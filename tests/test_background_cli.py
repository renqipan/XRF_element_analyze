"""Public background options must behave identically in single/batch entrypoints."""
import json
from pathlib import Path

import pytest

from xrf_analyzer import cli
from xrf_analyzer.batch import _configuration_from_analysis_cli
from xrf_analyzer.config import AnalysisConfig


def _run_cli(monkeypatch, tmp_path, extra=(), settings=None):
    captured = []

    def analyze(spectrum, calibration, sensitivity, output_dir, config):
        captured.append(config)
        return {'summary': {'fit_status': 'success'}}

    monkeypatch.setattr(cli, 'analyze', analyze)
    args = ['analyze', '--spectrum', 'sample.mca', '--calibration', 'calibration.csv',
            '--sensitivity', 'sensitivity.csv', '--output-dir', str(tmp_path/'out')]
    if settings is not None:
        config_file = tmp_path/'config.json'
        config_file.write_text(json.dumps(settings))
        args += ['--config', str(config_file)]
    assert cli.main(args+list(extra)) == 0
    assert len(captured) == 1
    return captured[0]


def test_background_defaults_to_none(monkeypatch, tmp_path):
    assert AnalysisConfig().validate().background_spectrum is None
    assert _run_cli(monkeypatch, tmp_path).background_spectrum is None


@pytest.mark.parametrize('settings,extra,expected', [
    (None, (), 'Ag'),
    ({'tube_target': 'Mo'}, (), 'Mo'),
    ({'tube_target': 'Mo'}, ('--tube-target', 'Cu'), 'Cu'),
    ({'tube_target': None}, (), None),
])
def test_single_cli_target_default_and_overrides(monkeypatch, tmp_path, settings, extra, expected):
    assert _run_cli(monkeypatch, tmp_path, extra, settings).tube_target == expected


@pytest.mark.parametrize('option', ['--background_spectrum', '--background-spectrum',
                                    '--backgroud_spectrum', '--backgroud-spectrum',
                                    '--background'])
def test_background_filename_aliases_share_one_configuration_field(monkeypatch, tmp_path, option):
    config = _run_cli(monkeypatch, tmp_path, [option, 'custom background.mca'])
    assert Path(config.background_spectrum) == Path('custom background.mca')


@pytest.mark.parametrize('option', ['--background_spectrum', '--backgroud_spectrum', '--background'])
@pytest.mark.parametrize('token', ['None', 'none', 'NONE'])
def test_explicit_none_disables_background_in_all_supported_spellings(monkeypatch, tmp_path, option, token):
    config = _run_cli(monkeypatch, tmp_path, [option, token])
    assert config.background_spectrum is None


def test_cli_background_overrides_json_including_explicit_disable(monkeypatch, tmp_path):
    settings = {'background_spectrum': 'configured.mca'}
    assert _run_cli(monkeypatch, tmp_path, settings=settings).background_spectrum == 'configured.mca'
    disabled = _run_cli(monkeypatch, tmp_path, ['--background', 'None'], settings)
    assert disabled.background_spectrum is None
    overridden = _run_cli(monkeypatch, tmp_path, ['--backgroud_spectrum', 'override.mca'], settings)
    assert overridden.background_spectrum == 'override.mca'


def test_json_null_disables_background_without_being_replaced_by_default(monkeypatch, tmp_path):
    assert _run_cli(monkeypatch, tmp_path, settings={'background_spectrum': None}).background_spectrum is None


def test_batch_reuses_background_aliases_and_cli_overrides(tmp_path):
    config_file = tmp_path/'batch_config.json'
    config_file.write_text(json.dumps({'background_spectrum': 'configured.mca'}))
    disabled = _configuration_from_analysis_cli('calibration.csv', 'sensitivity.csv', tmp_path/'out',
                                                config_file, ['--background', 'None'])
    assert disabled.background_spectrum is None
    custom = _configuration_from_analysis_cli('calibration.csv', 'sensitivity.csv', tmp_path/'out',
                                              config_file, ['--backgroud_spectrum', 'custom.mca'])
    assert custom.background_spectrum == 'custom.mca'
