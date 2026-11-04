#!/usr/bin/env python3
"""Known-truth parameter comparison, with separate training/validation noise seeds.

This is software validation conditional on the supplied S table and response,
not a calibration of physical specimen composition or instrument sensitivity.
"""
import argparse
import contextlib
from dataclasses import replace
import io
import json
import logging
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from xrf_analyzer.atomic_lines import load_sensitivity, excitation_edge_kev
from xrf_analyzer.atomic_completion import complete_transitions
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.pipeline import analyze
from xrf_analyzer.response import gaussian_response, sigma_kev

CASES = [
    ('blank', {}, 0., 0.),
    ('Fe_Cu_Pb', {'Fe': 180., 'Cu': 80., 'Pb': 400.}, .08, 0.),
    ('weak_Ca', {'Ca': 30., 'Fe': 350., 'Pb': 1800.}, .10, -.002),
    ('Hg_Pb', {'Hg': 120., 'Pb': 1300.}, .09, -.003),
    ('Pb_As_overlap', {'Pb': 800., 'As': 120.}, .07, 0.),
    ('weak_Fe_Ni', {'Fe': 40., 'Ni': 40., 'Pb': 1300.}, .09, 0.),
    ('true_Tb', {'Tb': 500., 'Fe': 80., 'Pb': 700.}, .07, -.002),
    ('hidden_Tb', {'Tb': 100., 'Fe': 200., 'Pb': 700.}, .07, -.002),
    ('gain_drift', {'Fe': 250., 'Cu': 180., 'Pb': 1000.}, .12, -.005),
]
WHITELIST = ('Ca','Cr','Mn','Fe','Co','Ni','Cu','Zn','As','Pm','Tb','Hg','Pb')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config',default='configs/default.json')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--variants', nargs='*')
    parser.add_argument('--data-dir')
    parser.add_argument('--all-elements', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    parameters = json.loads((root/args.config).read_text())
    base = AnalysisConfig(**parameters)
    base.elements = None if args.all_elements else WHITELIST
    base.tube_voltage_kv = 50.
    # Match the measured MCA bin width and empirically broad SDD resolution.
    physical = replace(base, fwhm_ref_kev=.27)
    lines, _, _ = complete_transitions(load_sensitivity(root/'data/xrf_relative_sensitivity.csv').lines, physical)
    lines = [l for l in lines if excitation_edge_kev(l) < 50.]
    channel = np.arange(600)
    energy = channel*(22.08/299)
    edges = (np.arange(601)-.5)*(22.08/299)
    variants = {'current': {}, 'threshold_3': {'candidate_threshold_sigma': 3.},
                'threshold_4': {'candidate_threshold_sigma': 4.},
                'centroid_025': {'centroid_tolerance_kev': .025},
                'support_16': {'support_delta_deviance': 16., 'low_z_kalpha_min_delta':16.},
                'support_36': {'support_delta_deviance': 36., 'low_z_kalpha_min_delta':36.},
                'background_2': {'background_degree':2},
                'background_4': {'background_degree':4},
                'penalty_6': {'pattern_element_penalty':6.},
                'no_gain': {'fit_energy_gain':False},
                'kalpha_30': {'low_z_max_z':30},
                'no_gain_kalpha30': {'fit_energy_gain':False, 'low_z_max_z':30}}
    variants['balanced_26']={'candidate_threshold_sigma':3.5,'low_z_max_z':26}
    variants['balanced_30']={'candidate_threshold_sigma':3.5,'low_z_max_z':30}
    if args.validation:
        variants = {k:v for k,v in variants.items() if k in ('current','kalpha_30','no_gain','no_gain_kalpha30')}
    if args.variants:
        variants = {k:v for k,v in variants.items() if k in args.variants}
    report = {'base_parameters': base.to_dict(), 'generator_fwhm_ref_kev':.27,
              'seed_offset':args.seed_offset, 'full_dictionary':args.all_elements, 'truth_scope':'synthetic counts conditional on supplied S and Gaussian response',
              'variants':{}}
    logging.disable(logging.WARNING)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_context = (contextlib.nullcontext(args.data_dir) if args.data_dir else
                         tempfile.TemporaryDirectory(prefix='xrf-tuning-'))
    with temporary_context as temporary:
        folder = Path(temporary)
        folder.mkdir(parents=True,exist_ok=True)
        cal = folder/'calibration.csv'
        cal.write_text('channel,energy_keV\n0,0\n299,22.08\n')
        datasets = []
        for i,(name,q,delta,gain) in enumerate(CASES):
            mu = 12.+8*np.exp(-energy/12.)
            for line in lines:
                mu += q.get(line.symbol,0.)*line.sensitivity_cm2_g*gaussian_response(
                    edges,line.energy_keV*(1+gain)+delta,float(sigma_kev(line.energy_keV,physical)))
            y = np.random.default_rng(20261104+args.seed_offset+i).poisson(mu)
            path = folder/(name+'.csv')
            pd.DataFrame({'channel':channel,'counts':y}).to_csv(path,index=False)
            datasets.append((name,q,path))
        for label,overrides in variants.items():
            rows=[]
            for name,truth,path in datasets:
                cfg=replace(base,**overrides)
                with contextlib.redirect_stdout(io.StringIO()):
                    result=analyze(path,cal,root/'data/xrf_relative_sensitivity.csv',config=cfg)
                observed=set(result['identification'].candidates.loc[
                    result['identification'].candidates.Status=='supported','Element'])
                expected=set(truth)
                ratio_errors=[]
                density=result['density']
                for r in density.itertuples(index=False):
                    if (r.Element in truth and r.Reference_Element in truth and r.Element != r.Reference_Element
                            and np.isfinite(r.Relative_to_Reference)):
                        target=truth[r.Element]/truth[r.Reference_Element]
                        ratio_errors.append(abs(r.Relative_to_Reference/target-1))
                rows.append(dict(case=name,truth=sorted(expected),supported=sorted(observed),
                    false_positive=sorted(observed-expected),false_negative=sorted(expected-observed),
                    fit_success=result['identification'].fit.success,
                    ratio_relative_errors=ratio_errors,
                    fitted_symbols=result['identification'].fit.symbols,
                    response_initialization=result['summary']['response_initialization'],
                    centroid_alignment=result['summary']['centroid_alignment']['initial_alignment']['delta_E_keV'],
                    candidate_diagnostics=json.loads(result['identification'].candidates.loc[
                        result['identification'].candidates.Element.isin(expected | observed),
                        ['Element','Status','DeltaDeviance','NumIndependentEvidenceRegions','PatternScore','ConfidenceNotes']].to_json(orient='records'))))
            fp=sum(len(r['false_positive']) for r in rows)
            fn=sum(len(r['false_negative']) for r in rows)
            failed=sum(not r['fit_success'] for r in rows)
            errors=[e for r in rows for e in r['ratio_relative_errors']]
            report['variants'][label]=dict(overrides=overrides,false_positives=fp,false_negatives=fn,
                fit_failures=failed,weighted_loss=5*fp+3*fn+5*failed,
                median_ratio_relative_error=float(np.median(errors)) if errors else None,cases=rows)
            output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
            print(label,'FP',fp,'FN',fn,'failed',failed,'median ratio error',np.median(errors) if errors else None,flush=True)


if __name__ == '__main__':
    main()
