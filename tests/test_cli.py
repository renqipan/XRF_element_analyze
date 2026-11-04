import json
from pathlib import Path
import pytest
from xrf_analyzer.cli import main, build_parser


@pytest.mark.parametrize('options,calibration,sensitivity', [
    ([], 'data/calibration.csv', 'data/xrf_relative_sensitivity.csv'),
    (['--calibration', 'custom_cal.csv'], 'custom_cal.csv', 'data/xrf_relative_sensitivity.csv'),
    (['--sensitivity', 'custom_s.csv'], 'data/calibration.csv', 'custom_s.csv'),
    (['--calibration', 'custom_cal.csv', '--sensitivity', 'custom_s.csv'], 'custom_cal.csv', 'custom_s.csv'),
])
def test_optional_input_paths_and_overrides(options, calibration, sensitivity):
    args = build_parser().parse_args(['analyze', '--spectrum', 'sample.mca', '--output-dir', 'out']+options)
    assert args.calibration == calibration
    assert args.sensitivity == sensitivity


@pytest.mark.parametrize('options', [['--spectrum', 'sample.mca'], ['--output-dir', 'out']])
def test_spectrum_and_output_directory_remain_required(options):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['analyze']+options)
    assert exc.value.code == 2


def test_cli_analyzes_with_default_input_files(synthetic, tmp_path, monkeypatch):
    spectrum, calibration, sensitivity, _, _, _ = synthetic({'Fe': 400.})
    data = tmp_path/'data'
    data.mkdir()
    (data/'calibration.csv').write_bytes(calibration.read_bytes())
    (data/'xrf_relative_sensitivity.csv').write_bytes(sensitivity.read_bytes())
    monkeypatch.chdir(tmp_path)
    out = tmp_path/'out'
    assert main(['analyze', '--spectrum', str(spectrum), '--output-dir', str(out),
                 '--elements', 'Fe', '--energy-range', '4', '8']) == 0
    summary = json.loads((out/'fit_summary.json').read_text())
    assert summary['fit_status'] == 'success'
    assert summary['input_files']['calibration']['path'] == str(data/'calibration.csv')
    assert summary['input_files']['sensitivity']['path'] == str(data/'xrf_relative_sensitivity.csv')


def test_cli_single_and_batch(synthetic, tmp_path):
    s, c, t, config, _, _ = synthetic({'Fe': 400})
    second = tmp_path/'second.csv'
    second.write_bytes(s.read_bytes())
    out = tmp_path/'batch'
    assert main(['analyze', '--spectrum', str(s), str(second), '--calibration', str(c),
                 '--sensitivity', str(t), '--output-dir', str(out), '--elements', 'Fe',
                 '--energy-range', '4', '8']) == 0
    assert (out/'sample/fit_summary.json').exists()
    assert (out/'second/fit_summary.json').exists()
    assert json.loads((out/'sample/fit_summary.json').read_text())['parameters']['tube_target'] == 'Ag'


def test_cli_bad_input_writes_failed_status(tmp_path):
    bad = tmp_path/'bad.csv'
    bad.write_text('counts\n-5\n')
    out = tmp_path/'out'
    out.mkdir()
    (out/'fitted_lines.csv').write_text('stale results')
    assert main(['analyze', '--spectrum', str(bad), '--calibration', 'missing.csv',
                 '--sensitivity', 'missing.csv', '--output-dir', str(out)]) == 1
    assert json.loads((out/'fit_summary.json').read_text())['fit_status'] == 'failed'
    assert not (out/'fitted_lines.csv').exists()


def test_cli_invalid_configuration(synthetic, tmp_path):
    s, c, t, _, _, _ = synthetic()
    assert main(['analyze', '--spectrum', str(s), '--calibration', str(c),
                 '--sensitivity', str(t), '--output-dir', str(tmp_path/'out'),
                 '--detector-efficiency', '.5']) == 2
