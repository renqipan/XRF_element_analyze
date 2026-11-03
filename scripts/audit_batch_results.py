"""Read-only report audit: python scripts/audit_batch_results.py XRF_Analyze_results.

Checks every completed sample in a batch. This validates file consistency and
quantitative gating, not the correctness of an unknown sample's composition.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd


def same_value(a, b):
    missing=lambda value:pd.isna(value) or (isinstance(value,str) and value=='')
    if missing(a) and missing(b):
        return True
    if isinstance(a, (float, int, np.number)) and isinstance(b, (float, int, np.number)):
        return bool(np.isclose(a, b, rtol=1e-11, atol=1e-12))
    return str(a) == str(b)


def main(root):
    out = Path(root)
    manifest = json.loads((out/'batch_manifest.json').read_text())
    samples = pd.read_csv(out/'batch_summary.txt', sep='\t', encoding='utf-8-sig').fillna('')
    detail=json.loads((out/'density_details.json').read_text())
    aggregate=pd.DataFrame(detail['records'],columns=detail['columns'])
    from xrf_analyzer.batch import format_density_summary
    expected=format_density_summary(manifest['samples'],detail['records'])
    assert (out/'relative_areal_density_all.txt').read_text(encoding='utf-8-sig')==expected
    assert len(expected.splitlines())==4*len(samples)
    assert all(line=='' for line in expected.splitlines()[3::4])
    errors, records = [], []
    for _, sample in samples.iterrows():
        name = sample.Spectrum_File
        destination = out/sample.Sample_Output_Directory
        summary = json.loads((destination/'fit_summary.json').read_text())
        sample_rows = aggregate.loc[aggregate.Spectrum_File == name]
        local = []
        def check(condition, message):
            if not condition:
                local.append(message)
        check(len(sample_rows)>0, 'Sample missing from aggregate TXT')
        check(summary['fit_status']==sample.Fit_Status, 'Fit status disagrees with sample summary')
        if sample.Fit_Status != 'success':
            check(sample_rows.Element.isna().all(), 'Failed sample has aggregate element rows')
            check(sample_rows.Relative_to_Reference.isna().all(), 'Failed sample has ratios')
            records.append({'sample':name, 'fit_status':sample.Fit_Status, 'errors':local})
            errors.extend(name+': '+message for message in local)
            continue
        spec = pd.read_csv(destination/'fit_spectrum.csv')
        components = pd.read_csv(destination/'fit_components.csv')
        candidates = pd.read_csv(destination/'element_candidates.csv')
        density = pd.read_csv(destination/'relative_areal_density.csv')
        check(np.isfinite(spec.to_numpy()).all(), 'Nonfinite spectrum model or residual values')
        count_cols = ['observed_counts','model_total_counts','background_counts',
                      'scatter_or_artifact_counts','element_total_counts']
        check((spec[count_cols].to_numpy()>=0).all(), 'Negative model component or counts')
        model_error = np.max(abs(spec.model_total_counts-(spec.background_counts+
                             spec.scatter_or_artifact_counts+spec.element_total_counts)))
        check(np.allclose(spec.model_total_counts, spec.background_counts+
                          spec.scatter_or_artifact_counts+spec.element_total_counts,
                          rtol=1e-11, atol=1e-9), 'Model total != background+scatter+elements')
        check(np.allclose(spec.energy_keV,components.energy_keV), 'Energy coordinates differ')
        check((components.drop(columns=['energy_keV']).to_numpy()>=0).all(), 'Negative elemental component')
        check(np.allclose(spec.element_total_counts,components.drop(columns=['energy_keV']).sum(axis=1),
                          rtol=1e-11,atol=1e-9),'Element total != sum of exported element components')
        supported = set(candidates.loc[candidates.Status=='supported','Element'])
        possible = set(summary['elements_need_confirmation'])
        check(set(density.Element)==supported, 'Density table contains non-supported/missing supported elements')
        check(set(filter(None,sample.Supported_Elements.split(',')))==supported,
              'Supported elements in sample TXT do not match candidate statuses')
        check(set(filter(None,sample.Possible_Elements.split(',')))==possible,
              'Possible elements in sample TXT disagree with JSON')
        numeric = density.loc[density.Relative_to_Reference.notna()]
        check(numeric.Quantification_Status.eq('supported').all(), 'Invalid quantitative status has numeric ratio')
        check((numeric.Relative_to_Reference>0).all(), 'Reported numeric ratio is nonpositive')
        check(not set(numeric.Element)&possible, 'Pending element participates in reported ratios')
        if summary.get('quantification_method')=='local-peaks' and len(numeric):
            check(np.allclose(numeric.q_Area,numeric.Area_Counts/numeric.Sensitivity_Sum,
                              rtol=1e-10,atol=1e-12),'Primary-area density does not equal A/S')
            check(numeric.Quantification_Method.eq('local-peaks').all(),'Numeric ratios have the wrong measurement method')
            local_fit=pd.read_csv(destination/'quantification_fit.csv')
            check(np.allclose(local_fit.model_counts,local_fit.background_counts+local_fit.primary_counts+local_fit.nuisance_counts,
                              rtol=1e-10,atol=1e-9),'Local model components do not sum to total')
            check(np.isfinite(local_fit.to_numpy()[:,1:].astype(float)).all(),'Nonfinite local measurement model')
        if len(numeric):
            check(np.isclose(numeric.Fraction_among_Reported_Elements.sum(),1.), 'Reported fractions do not sum to 1')
        if not summary['quantitative_interpretation_valid']:
            check(density.Relative_to_Reference.isna().all(), 'Invalid quantitative sample has numeric ratios')
        for _, row in density.iterrows():
            matching = sample_rows.loc[sample_rows.Element==row.Element]
            check(len(matching)==1,'TXT row count differs for '+row.Element)
            if len(matching)==1:
                for col in density.columns:
                    check(same_value(row[col],matching.iloc[0][col]),'Detailed JSON differs: '+row.Element+'.'+col)
        hash_expected = summary['input_files']['sample_spectrum']['sha256']
        check(sample_rows.Spectrum_SHA256.eq(hash_expected).all(), 'Aggregate source hash differs')
        image = out/sample.Spectrum_Fit_Image
        check(image.is_file(), 'Root spectrum-fit image is missing')
        if image.is_file():
            check(hashlib.sha256(image.read_bytes()).digest()==hashlib.sha256((destination/(Path(name).stem+'_elements.png')).read_bytes()).digest(),
                  'Root fit image differs from sample image')
        errors.extend(name+': '+message for message in local)
        records.append({'sample':name,'channels':len(spec),'supported':sorted(supported),
                        'possible':sorted(possible),'numeric_ratio_count':len(numeric),
                        'max_model_sum_error':float(model_error),'errors':local})
    print(json.dumps({'batch_completed':manifest['completed'],'manifest_processed':manifest['processed_samples'],
                      'audited_samples':len(records),'aggregate_rows':len(aggregate),
                      'errors':errors,'samples':records},ensure_ascii=False,indent=2))
    return int(bool(errors))


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1] if len(sys.argv)>1 else 'XRF_Analyze_results'))
