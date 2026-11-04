"""Conservative acquisition-domain and empirical response initialization."""
from dataclasses import replace
import numpy as np
from .response import FWHM_FACTOR, sigma_kev


def infer_hard_lower_cutoff(spectrum, calibration, config):
    recorded = spectrum.metadata.get('hardware_lower_cutoff')
    if recorded is not None:
        first = recorded['first_retained_index']
        channel = recorded['first_retained_channel']
        return dict(recorded, energy_keV=float(calibration.energy(channel-.5)),
                    zero_prefix_channels=first,
                    first_counts=float(spectrum.counts[first]) if first < len(spectrum.counts) else None,
                    inferred=False, reason='recorded_Amptek_DP5_TLLD')
    positive = np.flatnonzero(spectrum.counts > 0)
    if not config.infer_low_energy_cutoff or not len(positive):
        return None
    first = int(positive[0])
    # A long exactly-zero prefix followed by a large abrupt step, not ordinary
    # low-count Poisson zeros. Record an inference, never overwrite input counts.
    if first < 32 or spectrum.counts[first] < 20 or spectrum.counts[first:first+3].sum() < 100:
        return None
    return {'energy_keV': float(calibration.energy(spectrum.channel[first]-.5)),
            'first_retained_channel': int(spectrum.channel[first]), 'zero_prefix_channels': first,
            'first_counts': float(spectrum.counts[first]), 'inferred': True,
            'reason': 'long_exact_zero_prefix_followed_by_abrupt_high_counts',
            'limitation': 'Acquisition threshold inferred from data; instrument confirmation required'}


def initialize_resolution(peaks, config):
    if not config.fit_response or not config.estimate_resolution_from_peaks or peaks.empty:
        return config, {'used': False}
    good = peaks[(peaks.approx_significance >= 10) & (peaks.local_deviance_per_dof < 5)
                 & ~peaks.quality_flags.str.contains('broad_structure|roi_truncated')]
    if len(good) < 2:
        return config, {'used': False, 'reason': 'Fewer than two well-fitted isolated observed peaks'}
    sig2 = (good.estimated_fwhm_keV.to_numpy()/FWHM_FACTOR)**2
    electronic = sig2-config.fano*config.pair_energy_kev*good.energy_keV.to_numpy()
    electronic = electronic[electronic > 0]
    if len(electronic) < 2:
        return config, {'used': False, 'reason': 'No consistent positive electronic width'}
    # The narrow half is less biased by unresolved multiplets or scattered peaks.
    electronic = np.sort(electronic)[:max(2, int(np.ceil(len(electronic)*.6)))]
    reference = FWHM_FACTOR*np.sqrt(np.median(electronic)+config.fano*config.pair_energy_kev*config.fwhm_ref_energy_kev)
    if not .75*config.fwhm_ref_kev <= reference <= 3*config.fwhm_ref_kev:
        return config, {'used': False, 'reason': 'Empirical resolution outside conservative initialization limits'}
    return replace(config, fwhm_ref_kev=float(reference)), {
        'used': True, 'method': 'narrow_isolated_peak_Fano_electronic_variance_median',
        'configured_reference_FWHM_keV': config.fwhm_ref_kev, 'effective_reference_FWHM_keV': float(reference),
        'peak_ids': good.peak_id.astype(int).tolist(), 'is_instrument_calibration': False}


def reclassify_peak_widths(peaks, config):
    """Interpret measured widths using the response actually used for matching.

    Initial peak discovery uses the configured resolution. After an empirical
    update, retaining its old broad flags would reject normal low-energy lines.
    Peak positions, areas, raw counts and local fit quality are unchanged.
    """
    peaks = peaks.copy()
    for index, peak in peaks.iterrows():
        flags = [flag for flag in peak.quality_flags.split(';') if flag and flag != 'broad_structure']
        if peak.estimated_fwhm_keV > 2.5*FWHM_FACTOR*float(sigma_kev(peak.energy_keV, config)):
            flags.append('broad_structure')
        peaks.at[index, 'quality_flags'] = ';'.join(flags)
    return peaks
