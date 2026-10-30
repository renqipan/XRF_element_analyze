from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from xrf_analyzer.atomic_lines import LINE_MAP, load_sensitivity
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.response import gaussian_response, sigma_kev


@pytest.fixture
def synthetic(tmp_path):
    def make(q=None, seed=42, exposure=1., energy_range=(1., 18.), branches=None,
             custom_lines=None, shift=0., width_scale=1., background=8.):
        q = {'Fe': 1000., 'Cu': 500., 'Pb': 2000.} if q is None else q
        import xraylib
        # Deliberately artificial common-scale S for tests; not physical sensitivity data.
        values = {'Fe': {'Kalpha1': 8., 'Kalpha2': 4., 'Kbeta1': 1.5},
                  'Cu': {'Kalpha1': 7., 'Kalpha2': 3.5, 'Kbeta1': 1.2},
                  'Pb': {'Lalpha1': 5., 'Lalpha2': .5, 'Lbeta1': 2.5, 'Lbeta2': .8, 'Lgamma1': .7},
                  'As': {'Kalpha1': 4., 'Kalpha2': 2., 'Kbeta1': .9},
                  'Co': {'Kalpha1': 8., 'Kalpha2': 4., 'Kbeta1': 1.4}}
        if branches:
            values.update(branches)
        records = []
        for symbol, vals in values.items():
            rec = {'Element': '%d %s' % (xraylib.SymbolToAtomicNumber(symbol), symbol)}
            rec.update({key: vals.get(key, '') for key in LINE_MAP})
            records.append(rec)
        sensitivity = tmp_path/'sensitivity.csv'
        pd.DataFrame(records).to_csv(sensitivity, index=False)
        calibration = tmp_path/'calibration.csv'
        calibration.write_text('channel,energy_keV\n100,2\n900,18\n')
        c = np.arange(0, 1001)
        e, edges = c*.02, (np.arange(1002)-.5)*.02
        config = AnalysisConfig(energy_range=energy_range, fit_response=True, elements=tuple(values), seed=seed)
        mu = np.full(len(c), background, dtype=float)
        table = load_sensitivity(sensitivity)
        for line in table.lines if custom_lines is None else custom_lines:
            mu += q.get(line.symbol, 0.)*line.sensitivity_cm2_g*gaussian_response(edges, line.energy_keV+shift,
                                                                                   sigma_kev(line.energy_keV, config)*width_scale)
        y = np.random.default_rng(seed).poisson(mu*exposure)
        spectrum = tmp_path/'sample.csv'
        pd.DataFrame({'channel': c, 'counts': y}).to_csv(spectrum, index=False)
        return spectrum, calibration, sensitivity, config, mu*exposure, table.lines
    return make
