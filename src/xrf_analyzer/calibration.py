from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .csv_input import read_csv_unique_headers


@dataclass
class Calibration:
    a_keV: float
    b_keV_per_channel: float
    rms_keV: float
    covariance: object
    points: list
    residuals_keV: list
    max_abs_residual_keV: float
    two_point: bool
    weighted: bool
    calibration_quality: str
    warnings: list

    def energy(self, channel):
        return self.a_keV + self.b_keV_per_channel * np.asarray(channel)

    def to_dict(self):
        result = asdict(self)
        result['covariance'] = None if self.covariance is None else self.covariance.tolist()
        return result


def fit_linear_calibration(path, fwhm_at_6kev=0.135, strict=False):
    df = read_csv_unique_headers(path, encoding='utf-8-sig')
    aliases = {'Channel': 'channel', 'Energy': 'energy_keV', 'energy': 'energy_keV',
               'energy(keV)': 'energy_keV'}
    df = df.rename(columns=aliases)
    if df.columns.duplicated().any() or not {'channel', 'energy_keV'} <= set(df):
        raise ValueError('Calibration requires unambiguous channel,energy_keV columns')
    c = pd.to_numeric(df.channel, errors='raise').to_numpy(dtype=float)
    e = pd.to_numeric(df.energy_keV, errors='raise').to_numpy(dtype=float)
    if len(c) < 2 or not np.isfinite(c).all() or not np.isfinite(e).all() or (c < 0).any():
        raise ValueError('Calibration needs >=2 finite nonnegative channels and finite energies')
    warnings = []
    # Explicit compatibility with the supplied instrument zero-origin calibration.
    if (e < 0).any() or ((e == 0) & (c != 0)).any():
        raise ValueError('Calibration energies must be >0, except the explicit (0,0) origin')
    if (e == 0).any():
        warnings.append('Accepted instrument origin (0,0); zero energy is allowed only at channel zero')
    if len(np.unique(c)) != len(c):
        raise ValueError('Duplicate calibration channels are not allowed')
    weighted = False
    w = np.ones(len(c))
    if 'energy_sigma_keV' in df:
        sigma = pd.to_numeric(df.energy_sigma_keV, errors='raise').to_numpy(dtype=float)
        present = ~np.isnan(sigma)
        if (sigma[present] <= 0).any() or not np.isfinite(sigma[present]).all():
            raise ValueError('energy_sigma_keV must be finite and >0 where provided')
        if present.all():
            w = 1 / sigma
            weighted = True
        elif present.any():
            warnings.append('Mixed missing energy uncertainties: used unweighted calibration')
    design = np.column_stack([np.ones(len(c)), c])
    coeff = np.linalg.lstsq(design * w[:, None], e * w, rcond=None)[0]
    if np.ptp(e) == 0 or coeff[1] <= 0:
        raise ValueError('Calibration slope must be positive')
    residual = e - design @ coeff
    cov = None
    if weighted or len(c) > 2:
        cov = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None]))
        if not weighted:
            cov *= (residual @ residual) / (len(c) - 2)
    limit = max(0.03, 0.25 * fwhm_at_6kev)
    bad = float(np.max(np.abs(residual))) > limit
    if bad:
        warnings.append('Calibration residual exceeds %.5g keV; no points were removed' % limit)
        if strict:
            raise ValueError(warnings[-1])
    if len(c) == 2 and not weighted:
        warnings.append('Two-point residuals do not estimate calibration statistical uncertainty')
    return Calibration(float(coeff[0]), float(coeff[1]), float(np.sqrt(np.mean(residual**2))),
                       cov, df[['channel', 'energy_keV']].to_dict('records'), residual.tolist(),
                       float(np.max(np.abs(residual))), len(c) == 2, weighted,
                       'warning' if bad else 'ok', warnings)
