#!/usr/bin/env python3
"""Compare pending-list floors using saved fits; never refit or change a batch."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.candidate_reporting import filter_possible_candidates, PENDING
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.element_prior import apply_reporting_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', default='XRF_Analyze_results')
    p.add_argument('--floors', nargs='+', type=float, default=[0., 6., 9., 12., 16.])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.batch)
    manifest = json.loads((root/'batch_manifest.json').read_text())
    assert manifest['completed'], 'Wait for the batch to complete'
    results = {str(floor): {'possible_records': 0, 'samples': {}} for floor in args.floors}
    for sample in manifest['samples']:
        directory = root/sample['Sample_Output_Directory']
        summary = json.loads((directory/'fit_summary.json').read_text())
        cfg = AnalysisConfig(**summary['parameters'])
        cfg.fwhm_ref_kev = summary['response_initialization'].get('effective_reference_FWHM_keV', cfg.fwhm_ref_kev)
        table = pd.read_csv(directory/'fitted_lines.csv')
        fit = SimpleNamespace(delta=summary['delta_E_keV'], energy_gain=summary['energy_gain_correction'],
            resolution_scale=summary['resolution_scale'], line_centers={}, center_axis=cfg.find_peaks_axis)
        for row in table.itertuples():
            if pd.notna(row.Specified_Center_keV):
                fit.line_centers[(row.Element, row.Branch)] = row.Specified_Center_keV
        ident = SimpleNamespace(fit=fit,
            fit_lines=[LineCandidate(r.Z, r.Element, r.Branch, r.Reference_Energy_keV, r.Sensitivity_cm2_g)
                       for r in table.itertuples()], branch_evidence=summary['local_branch_evidence'],
            uncertainty_info=summary['uncertainty'], overlap_pairs=summary['overlap_pairs'])
        peaks = pd.read_csv(directory/'detected_peaks.csv')
        candidates = apply_reporting_policy(pd.read_csv(directory/'element_candidates.csv'),
            cfg.lanthanide_reporting, cfg.noble_gas_reporting)
        for floor in args.floors:
            selected = filter_possible_candidates(candidates, ident, peaks,
                replace(cfg, possible_min_delta_deviance=floor).validate())
            pending = selected.loc[selected.Status.isin(PENDING) & selected.q_Fit.notna() & selected.Reportable]
            results[str(floor)]['possible_records'] += len(pending)
            results[str(floor)]['samples'][sample['Spectrum_File']] = pending.Element.tolist()
    report = {'scope': 'reporting only; same saved likelihoods, statuses, peak areas and density',
              'manual_reference_used': False, 'variants': results}
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for floor, result in results.items():
        print('floor', floor, 'possible records', result['possible_records'])


if __name__ == '__main__':
    main()
