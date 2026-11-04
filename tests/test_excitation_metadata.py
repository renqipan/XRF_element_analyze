"""Explicit sample metadata must participate in sensitivity compatibility checks."""
import json

import pandas as pd
import pytest

from xrf_analyzer.pipeline import analyze


def _with_voltage(synthetic, sample_voltage, sensitivity_voltage, configured_voltage=None):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 600.})
    config.elements = ('Fe',)
    config.tube_voltage_kv = configured_voltage
    frame = pd.read_csv(spectrum)
    frame['tube_voltage_kV'] = sample_voltage
    frame.to_csv(spectrum, index=False)
    sensitivity.with_suffix('.metadata.json').write_text(
        json.dumps({'tube_voltage_kV': sensitivity_voltage}))
    return analyze(spectrum, calibration, sensitivity, config=config)


def test_sample_voltage_conflicting_with_sensitivity_withholds_ratios_without_cli_voltage(synthetic):
    result = _with_voltage(synthetic, 40., 50.)
    density = result['density']
    assert not density.empty
    assert density.Quantification_Status.eq('excitation_mismatch').all()
    assert density.Relative_to_Reference.isna().all()
    assert not result['summary']['quantitative_interpretation_valid']


@pytest.mark.parametrize('configured_voltage', [None, 50.])
def test_numerically_equal_voltage_metadata_keeps_ratios(synthetic, configured_voltage):
    result = _with_voltage(synthetic, 50., 50.000001, configured_voltage)
    assert result['density'].Quantification_Status.eq('supported').all()
    assert result['density'].Relative_to_Reference.notna().all()


def test_sample_voltage_conflict_with_explicit_configuration_still_withholds_ratios(synthetic):
    result = _with_voltage(synthetic, 40., 50., 50.)
    assert result['density'].Quantification_Status.eq('excitation_mismatch').all()
    assert result['density'].Relative_to_Reference.isna().all()
