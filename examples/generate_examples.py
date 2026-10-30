"""Generate reproducible Poisson samples from the supplied, unmodified S table.

Run from the repository root after installation: python examples/generate_examples.py
These are validation spectra, not measured samples and not incident source spectra.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from xrf_analyzer.atomic_lines import load_sensitivity
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.atomic_completion import complete_transitions
from xrf_analyzer.response import sigma_kev, gaussian_response
from xrf_analyzer.reporting import write_json, sha256


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'examples'
    sensitivity = root/'data/xrf_relative_sensitivity.csv'
    table = load_sensitivity(sensitivity)
    config = AnalysisConfig()
    lines, auxiliary, _ = complete_transitions(table.lines, config)
    channels = np.arange(2001)
    edges = (np.arange(2002)-.5)*.02
    energies = channels*.02
    (out/'calibration.csv').write_text('channel,energy_keV\n100,2\n1900,38\n', encoding='utf-8')
    cases = {'mixture_Fe_Cu_Pb': {'Fe': 1500., 'Cu': 750., 'Pb': 3000.},
             'overlap_Pb_As': {'Pb': 1600., 'As': 400.}}
    for i, (name, q) in enumerate(cases.items()):
        baseline = 10+18*np.exp(-energies/12)
        mu = baseline.copy()
        for line in lines:
            mu += q.get(line.symbol, 0.)*line.sensitivity_cm2_g*gaussian_response(
                edges, line.energy_keV+.006, float(sigma_kev(line.energy_keV, config))*1.08)
        shell_areas = {}
        for symbol in q:
            aux = [l for l in auxiliary if l.symbol == symbol]
            if not aux:
                continue
            shell_areas[symbol+'_L1_unscaled'] = q[symbol]
            total_rate = sum(l.relative_rate for l in aux)
            for line in aux:
                mu += q[symbol]*line.relative_rate/total_rate*gaussian_response(
                    edges, line.energy_keV+.006, float(sigma_kev(line.energy_keV, config))*1.08)
        y = np.random.default_rng(20261030+i).poisson(mu)
        pd.DataFrame({'channel': channels, 'counts': y, 'live_time_s': 30.}).to_csv(out/(name+'.csv'), index=False)
        write_json(out/(name+'.truth.json'), {'synthetic': True, 'generator_version': '0.2.0', 'seed': 20261030+i, 'q_truth': q,
                   'delta_E_keV': .006, 'resolution_scale': 1.08,
                   'sensitivity_sha256': sha256(sensitivity), 'branch_ratios': 'fixed_by_sensitivity_plus_same_shell_completion',
                   'auxiliary_shell_areas_counts': shell_areas,
                   'note': 'Known common-scale q values; not absolute g/cm2; sensitivity provenance still unverified'})
        print(out/(name+'.csv'))


if __name__ == '__main__':
    main()
