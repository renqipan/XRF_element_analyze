#!/usr/bin/env python3
"""Compare a numerical baseline to a completed batch; verify input and A/S invariants."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='results/optimization_20261104')
    parser.add_argument('--batch', help='Completed batch directory (default: ROOT/batch)')
    args=parser.parse_args()
    root=Path(args.root)
    before=json.loads((root/'before.json').read_text())['spectra']
    batch=Path(args.batch) if args.batch else root/'batch'
    manifest=json.loads((batch/'batch_manifest.json').read_text())
    if not manifest['completed']:
        raise SystemExit('Batch must complete before comparison')
    rows=[]
    density_errors=[]
    summaries={}
    for directory in sorted((batch/'spectra').iterdir()):
        path=directory/'fit_summary.json'
        if not path.exists():continue
        summary=json.loads(path.read_text())
        name=directory.name
        if name not in before:
            raise AssertionError('Unexpected input '+name)
        old=before[name]
        candidates=pd.read_csv(directory/'element_candidates.csv')
        density=pd.read_csv(directory/'relative_areal_density.csv')
        supported=candidates.loc[candidates.Status=='supported','Element'].tolist()
        supported_before=old['supported']
        same_hash=old['sha256']==summary['input_files']['sample_spectrum']['sha256']
        if not same_hash:raise AssertionError('Input changed: '+name)
        if not density.empty:
            valid=density.q_Area.notna() & density.Area_Counts.notna() & density.Sensitivity_Sum.gt(0)
            errors=abs(density.loc[valid,'q_Area']-density.loc[valid,'Area_Counts']/density.loc[valid,'Sensitivity_Sum'])
            density_errors.extend(errors.tolist())
        rows.append(dict(Spectrum=name,Before=','.join(supported_before),After=','.join(supported),
            Added=','.join(s for s in supported if s not in supported_before),
            Removed=','.join(s for s in supported_before if s not in supported),
            Before_Deviance_per_dof=old['deviance_per_dof'],After_Deviance_per_dof=summary['deviance_per_dof'],
            Fit_Success=summary['fit_status']=='success',Input_SHA256_Unchanged=same_hash,
            Numeric_Ratios=int(density.Relative_to_Reference.notna().sum())))
        summaries[name]=summary
    assert len(rows)==len(before)==manifest['sample_count']
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'comparison.csv',index=False)
    report=dict(spectra=len(rows),fits_successful=int(frame.Fit_Success.sum()),
        before_supported_records=sum(len(r['supported']) for r in before.values()),
        quantified_records=sum(len(s['reference']['fraction_denominator_elements']) for s in summaries.values()),
        after_supported_records=sum(len(row['After'].split(',')) if row['After'] else 0 for row in rows),
        numeric_ratios=int(frame.Numeric_Ratios.sum()),all_input_hashes_unchanged=bool(frame.Input_SHA256_Unchanged.all()),
        max_area_over_sensitivity_absolute_error=max(density_errors or [0.]),
        warning='Supported-set changes are not ground-truth accuracy; models differ, so deviance changes are descriptive.',
        rows=rows)
    (root/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    columns=['Spectrum','Before','After','Added','Removed']
    table='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    table+='\n'.join('| '+' | '.join(str(row[c]) for c in columns)+' |' for row in rows)+'\n'
    (root/'comparison_table.md').write_text(table)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
