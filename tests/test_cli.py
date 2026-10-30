import json
from pathlib import Path
from xrf_analyzer.cli import main


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
