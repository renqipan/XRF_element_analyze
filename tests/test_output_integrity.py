"""Reruns and case-sensitive inputs must retain unambiguous report artifacts."""
from dataclasses import replace

import pandas as pd

from xrf_analyzer.batch import run_batch
from xrf_analyzer.pipeline import analyze
from xrf_analyzer.reporting import export_analysis


def test_same_stem_same_format_case_variants_have_distinct_batch_images(synthetic, tmp_path):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 600.})
    config.elements = ('Fe',)
    source = tmp_path/'input'
    source.mkdir()
    (source/'same.csv').write_bytes(spectrum.read_bytes())
    frame = pd.read_csv(spectrum)
    frame['counts'] *= 2
    frame.to_csv(source/'same.CSV', index=False)
    output = tmp_path/'output'
    manifest = run_batch(source, output, calibration, sensitivity, config)
    assert manifest['failures'] == 0
    samples = manifest['samples']
    assert len({row['Spectrum_Fit_Image'].casefold() for row in samples}) == 2
    for row in samples:
        image = output/row['Spectrum_Fit_Image']
        original = output/row['Sample_Output_Directory']/'same_elements.png'
        assert image.read_bytes() == original.read_bytes()


def test_successful_rerun_without_intermediates_removes_old_intermediate_reports(synthetic, tmp_path):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 600.})
    result = analyze(spectrum, calibration, sensitivity, config=config)
    result['config'] = replace(result['config'], save_intermediate=True)
    output = tmp_path/'output'
    export_analysis(result, output)
    names = ('background_initial.csv', 'sensitivity_states.csv')
    assert all((output/name).is_file() for name in names)
    result['config'] = replace(result['config'], save_intermediate=False)
    export_analysis(result, output)
    assert all(not (output/name).exists() for name in names)
