"""The short CLI and shafa preset must select the same analysis model."""
import json
from pathlib import Path

from xrf_analyzer import cli
from xrf_analyzer.config import AnalysisConfig, single_spectrum_defaults
from xrf_analyzer.batch import _configuration_from_analysis_cli


ROOT = Path(__file__).resolve().parents[1]


def test_single_preset_matches_complete_shafa_configuration_without_fixed_reference():
    preset = single_spectrum_defaults().validate()
    shafa = AnalysisConfig(**json.loads((ROOT/'configs/shafa.json').read_text())).validate()
    assert json.loads(json.dumps(preset.to_dict())) == json.loads(json.dumps(shafa.to_dict()))
    assert preset.reference_element is None


def test_short_and_explicit_cli_pass_the_same_configuration(monkeypatch, tmp_path):
    configs = []

    def capture(spectrum, calibration, sensitivity, output_dir, config):
        configs.append(config.to_dict())
        return {'summary': {'fit_status': 'success'}}

    monkeypatch.setattr(cli, 'analyze', capture)
    base = ['analyze', '--spectrum', 'sample.mca', '--output-dir', str(tmp_path/'out')]
    assert cli.main(base) == 0
    assert cli.main(base+['--config', str(ROOT/'configs/shafa.json')]) == 0
    assert json.loads(json.dumps(configs[0])) == json.loads(json.dumps(configs[1]))
    assert configs[0]['reference_element'] is None
    assert cli.main(base+['--reference-element', 'Fe']) == 0
    assert configs[2]['reference_element'] == 'Fe'


def test_json_and_cli_still_override_the_single_preset(monkeypatch, tmp_path):
    captured = []
    def capture(spectrum, calibration, sensitivity, output_dir, config):
        captured.append(config)
        return {'summary': {'fit_status': 'success'}}
    monkeypatch.setattr(cli, 'analyze', capture)
    path = tmp_path/'custom.json'
    path.write_text(json.dumps({'energy_range': [4., 8.], 'tube_voltage_kv': 40.,
                                'save_intermediate': False, 'tube_target': 'Cu'}))
    base = ['analyze', '--spectrum', 'sample.mca', '--output-dir', str(tmp_path/'out'), '--config', str(path)]
    assert cli.main(base) == 0
    assert captured[-1].energy_range == [4., 8.]
    assert captured[-1].tube_voltage_kv == 40.
    assert not captured[-1].save_intermediate
    assert captured[-1].tube_target == 'Cu'
    assert cli.main(base+['--energy-range', '2', '12', '--tube-voltage-kv', '45', '--tube-target', 'Mo']) == 0
    assert captured[-1].energy_range == [2., 12.]
    assert captured[-1].tube_voltage_kv == 45.
    assert captured[-1].tube_target == 'Mo'


def test_single_preset_does_not_change_generic_api_or_directory_batch_defaults():
    assert AnalysisConfig().tube_voltage_kv is None
    batch = _configuration_from_analysis_cli('cal', 'sensitivity', 'out', None, [])
    assert batch.tube_voltage_kv is None
    assert not batch.save_intermediate
