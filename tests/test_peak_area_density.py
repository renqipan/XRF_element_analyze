"""Integration contract: measured local areas, rather than global fit quality, determine A/S."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.pipeline import analyze


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def shafa_local_density(tmp_path_factory):
    settings = json.loads((ROOT/'configs/shafa.json').read_text())
    settings['quantification_method'] = 'local-peaks'
    return analyze(ROOT/'SDD_Spectrum/shafa_50kv_20uA.mca',
                   ROOT/'data/calibration.csv', ROOT/'data/xrf_relative_sensitivity.csv',
                   tmp_path_factory.mktemp('shafa_local_density'), AnalysisConfig(**settings))


def test_real_supported_elements_have_measured_area_density_despite_global_mismatch(shafa_local_density):
    result = shafa_local_density
    summary, identification = result['summary'], result['identification']
    density = result['density'].set_index('Element')
    assert not summary['global_model_valid']
    assert summary['quantitative_interpretation_valid']
    assert result['local_quantification']
    for symbol in ['Ca', 'Fe', 'Pb']:
        row = density.loc[symbol]
        assert row.Element_Status == 'supported'
        assert row.Quantification_Status == 'supported'
        assert row.Apparent_Only
        assert row.Quantification_Method == 'local-peaks'
        assert np.isfinite(row.Relative_to_Reference) and row.Relative_to_Reference > 0
        assert np.isfinite(row.q_Area) and row.q_Area > 0
        assert row.Area_Counts > 0 and row.Sensitivity_Sum > 0
        assert row.q_Area == pytest.approx(row.Area_Counts/row.Sensitivity_Sum)
        # The physical global template and its exported amplitudes remain intact.
        i = identification.fit.symbols.index(symbol)
        assert row.q_Fit == pytest.approx(identification.fit.q[i])


def test_pending_elements_are_not_density_denominator_members(shafa_local_density):
    result = shafa_local_density
    candidates = result['identification'].candidates
    density = result['density']
    supported = set(candidates.loc[candidates.Status == 'supported', 'Element'])
    pending = set(result['summary']['elements_need_confirmation'])
    assert set(density.Element) == supported
    assert 'Cr' in pending
    # A failed Pm identity can now be not_supported rather than tentative;
    # either state must stay outside the quantitative denominator.
    assert candidates.set_index('Element').loc['Pm','Status'] != 'supported'
    assert not set(density.Element) & pending
    eligible = set(result['summary']['reference']['fraction_denominator_elements'])
    assert eligible and eligible <= supported and not eligible & pending
    numeric = density.loc[density.Relative_to_Reference.notna()]
    assert set(numeric.Element) == eligible
    assert numeric.Fraction_among_Reported_Elements.sum() == pytest.approx(1.)


def test_explicit_sensitivity_excitation_conflict_still_withholds_area_ratios(synthetic):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 900., 'Cu': 450.})
    config.quantification_method = 'local-peaks'
    config.elements = ('Fe', 'Cu')
    sensitivity.with_suffix('.metadata.json').write_text(json.dumps({'tube_voltage_kV': 40}))
    config.tube_voltage_kv = 50
    result = analyze(spectrum, calibration, sensitivity, config=config)
    assert result['identification'].candidates.Status.eq('supported').any()
    assert result['density'].Relative_to_Reference.isna().all()
    assert result['density'].Fraction_among_Reported_Elements.isna().all()
    assert not result['summary']['quantitative_interpretation_valid']
    assert result['density'].Quantification_Status.eq('excitation_mismatch').all()


def test_legacy_global_template_option_keeps_strict_global_quality_gate(synthetic):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 1000.})
    config.quantification_method = 'global-template'
    config.elements = ('Fe',)
    config.max_deviance_per_dof = .0001
    result = analyze(spectrum, calibration, sensitivity, config=config)
    assert result['summary']['analysis_status'] == 'model_mismatch'
    assert not result['summary']['global_model_valid']
    assert not result['summary']['quantitative_interpretation_valid']
    assert result['density'].Relative_to_Reference.isna().all()
    assert result['density'].Fraction_among_Reported_Elements.isna().all()


def test_unrelated_broad_structure_does_not_bias_known_primary_area_ratio(synthetic):
    spectrum, calibration, sensitivity, config, expectation, _ = synthetic({'Fe': 1000., 'Cu': 500.})
    config.quantification_method = 'local-peaks'
    config.elements = ('Fe', 'Cu')
    config.reference_element = 'Fe'
    config.fit_response = False
    frame = pd.read_csv(spectrum)
    # No counts from this remote structure reach the Fe/Cu Ka measurement windows.
    remote = 400.*np.exp(-.5*((frame.channel.to_numpy()*.02-14.)/.8)**2)
    frame['counts'] = np.random.default_rng(91).poisson(expectation+remote)
    frame.to_csv(spectrum, index=False)
    result = analyze(spectrum, calibration, sensitivity, config=config)
    assert not result['summary']['global_model_valid']
    assert result['summary']['quantitative_interpretation_valid']
    density = result['density'].set_index('Element')
    assert density.loc['Cu', 'Relative_to_Reference'] == pytest.approx(.5, rel=.08)
    assert density.loc['Fe', 'q_Area'] == pytest.approx(1000., rel=.08)
    assert density.loc['Cu', 'q_Area'] == pytest.approx(500., rel=.08)
    frame['counts'] = np.random.default_rng(91).poisson(expectation+3.*remote)
    frame.to_csv(spectrum, index=False)
    stronger = analyze(spectrum, calibration, sensitivity, config=config)
    other = stronger['density'].set_index('Element')
    for symbol in ['Fe', 'Cu']:
        assert other.loc[symbol, 'q_Area'] == pytest.approx(density.loc[symbol, 'q_Area'], rel=.015)
