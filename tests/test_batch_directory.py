import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import pytest
from xrf_analyzer.batch import run_batch, main, discover_spectra

ROOT = Path(__file__).resolve().parents[1]


def test_directory_batch_spaces_case_collisions_and_failure_continuation(synthetic, tmp_path):
    spectrum, calibration, sensitivity, cfg, _, _ = synthetic({'Fe': 600.})
    cfg.elements = ('Fe',)
    source = tmp_path/'input spectra'
    source.mkdir()
    (source/'same.csv').write_bytes(spectrum.read_bytes())
    (source/'sample space.CSV').write_bytes(spectrum.read_bytes())
    counts = pd.read_csv(spectrum).counts.astype(int)
    mca = '<<PMCA SPECTRUM>>\nLIVE_TIME - 30\n<<DATA>>\n'+'\n'.join(map(str, counts))+'\n<<END>>\nMCAC=%d;\n' % len(counts)
    (source/'same.MCA').write_text(mca)
    (source/'00 bad.csv').write_text('counts\n-1\n')
    (source/'ignored.txt').write_text('not a sample')
    output = tmp_path/'results'
    output.mkdir()
    (output/'relative_areal_density_all.csv').write_text('obsolete aggregate')
    (output/'batch_summary.csv').write_text('obsolete summary')
    manifest = run_batch(source, output, calibration, sensitivity, cfg)
    assert manifest['completed'] and manifest['sample_count'] == 4 and manifest['failures'] == 1
    assert manifest['successful_samples'] == 3
    summary = pd.read_csv(output/'batch_summary.txt', sep='\t')
    assert len(summary) == 4
    assert summary.loc[summary.Spectrum_File == '00 bad.csv', 'Fit_Status'].iloc[0] == 'failed'
    for name in ['same.csv', 'same.MCA', 'sample space.CSV']:
        assert (output/(name+'.spectrum_fit.png')).is_file()
        assert (output/'spectra'/name/'relative_areal_density.csv').is_file()
    aggregate = pd.read_csv(output/'relative_areal_density_all.txt', sep='\t')
    valid = aggregate.loc[aggregate.Record_Type == 'supported_element']
    assert set(valid.Element_Status) == {'supported'}
    assert set(valid.Spectrum_File) == {'same.csv', 'same.MCA', 'sample space.CSV'}
    assert valid.Relative_to_Reference.notna().all()
    assert np.allclose(valid.Relative_to_Reference, 1.)
    failure = aggregate.loc[aggregate.Record_Type == 'failed'].iloc[0]
    assert pd.isna(failure.Element) and pd.isna(failure.Relative_to_Reference)
    assert pd.notna(failure.Failure_Reason)
    assert not (output/'00 bad.csv.spectrum_fit.png').exists()
    saved = json.loads((output/'batch_manifest.json').read_text())
    assert saved['failures'] == 1 and saved['parameters']['tube_voltage_kv'] is None
    assert saved['aggregate_density_txt'] == 'relative_areal_density_all.txt'
    assert not (output/'relative_areal_density_all.csv').exists()
    assert not (output/'batch_summary.csv').exists()
    assert (output/'relative_areal_density_all.txt').read_bytes().startswith(b'\xef\xbb\xbf')
    assert set(valid.Ratio_Status) == {'available'}
    assert 'Possible_Elements' in aggregate.columns


def test_unsupported_or_unquantifiable_not_replaced_by_zero(synthetic, tmp_path):
    spectrum, calibration, sensitivity, cfg, _, _ = synthetic({'Fe': 600.})
    cfg.elements = ('Fe',)
    cfg.max_deviance_per_dof = .00001
    source = tmp_path/'input'
    source.mkdir()
    (source/'sample_40kv.CSV').write_bytes(spectrum.read_bytes())
    output = tmp_path/'results'
    manifest = run_batch(source, output, calibration, sensitivity, cfg)
    assert manifest['failures'] == 0  # Model mismatch is separate from optimizer failure.
    d = pd.read_csv(output/'relative_areal_density_all.txt', sep='\t')
    assert d.Record_Type.tolist() == ['supported_element']
    assert d.Relative_to_Reference.isna().all()
    assert d.Fraction_among_Reported_Elements.isna().all()
    assert d.Quantification_Status.tolist() == ['global_model_mismatch']
    assert d.Ratio_Status.tolist() == ['unavailable']
    assert d.Ratio_Unavailable_Reason.tolist() == ['global_model_mismatch']
    saved = json.loads((output/'spectra/sample_40kv.CSV/fit_summary.json').read_text())
    assert saved['parameters']['tube_voltage_kv'] is None  # Filename is not acquisition metadata.
    assert not pd.read_csv(output/'batch_summary.txt', sep='\t').Quantitative_Interpretation_Valid.any()


def test_no_supported_rows_still_records_each_sample(synthetic, tmp_path):
    spectrum, calibration, sensitivity, cfg, _, _ = synthetic({}, background=0)
    cfg.elements = ('Fe',)
    source = tmp_path/'input'
    source.mkdir()
    (source/'empty.csv').write_bytes(spectrum.read_bytes())
    out = tmp_path/'output'
    run_batch(source, out, calibration, sensitivity, cfg)
    d = pd.read_csv(out/'relative_areal_density_all.txt', sep='\t')
    assert d.Record_Type.tolist() == ['no_supported_elements']
    assert d.Element.isna().all() and d.Relative_to_Reference.isna().all()
    assert (out/'empty.csv.spectrum_fit.png').is_file()


def test_shell_runs_from_other_directory_and_forwards_options(synthetic, tmp_path):
    spectrum, calibration, sensitivity, _, _, _ = synthetic({'Fe': 500.})
    source = tmp_path/'with spaces'
    source.mkdir()
    (source/'a sample.csv').write_bytes(spectrum.read_bytes())
    output = tmp_path/'batch output'
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'energy_range': [4, 8], 'elements': ['Fe']}))
    env = os.environ.copy()
    env['PYTHON_BIN'] = str(ROOT/'.venv/bin/python')
    result = subprocess.run(['bash', str(ROOT/'analyze_all_spectrums.sh'), '--input-dir', str(source),
                             '--output-dir', str(output), '--calibration', str(calibration),
                             '--sensitivity', str(sensitivity), '--config', str(config),
                             '--find_peaks', '[6.4039, 7.058]'],
                            cwd=str(tmp_path), env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    s = json.loads((output/'spectra/a sample.csv/fit_summary.json').read_text())
    assert s['parameters']['find_peaks'] == [6.4039, 7.058]
    assert (output/'a sample.csv.spectrum_fit.png').exists()
    assert '批量处理完成' in result.stdout
    assert '统一面密度TXT' in result.stdout


def test_empty_missing_and_same_directories_fail_cleanly(tmp_path):
    empty = tmp_path/'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='No MCA or CSV'):
        discover_spectra(empty)
    assert main(['--input-dir', str(tmp_path/'missing'), '--output-dir', str(tmp_path/'out')]) == 2
    with pytest.raises(ValueError, match='must differ'):
        run_batch(empty, empty, 'not needed', 'not needed')


def test_main_returns_nonzero_for_failed_sample(synthetic, tmp_path):
    _, calibration, sensitivity, _, _, _ = synthetic()
    source = tmp_path/'input'
    source.mkdir()
    (source/'bad.csv').write_text('counts\n-2\n')
    assert main(['--input-dir', str(source), '--output-dir', str(tmp_path/'out'),
                 '--calibration', str(calibration), '--sensitivity', str(sensitivity),
                 '--elements', 'Fe', '--energy-range', '4', '8']) == 1
