"""Tube endpoints must exceed the initial-shell edge, not just the photon energy."""
from dataclasses import replace

import pandas as pd
import pytest
import xraylib

from xrf_analyzer.atomic_lines import LineCandidate, excitation_edge_kev
from xrf_analyzer.atomic_completion import complete_transitions
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.fit_model import instrument_lines
from xrf_analyzer.pipeline import analyze


@pytest.mark.parametrize('branch,shell', [
    ('Kalpha1', 'K'), ('Kalpha2', 'K'), ('Kbeta1', 'K'),
    ('Lalpha1', 'L3'), ('Lalpha2', 'L3'), ('Lbeta1', 'L2'),
    ('Lbeta2', 'L3'), ('Lgamma1', 'L2'), ('Malpha1', 'M5'),
])
def test_original_branches_use_their_initial_shell(branch, shell):
    line = LineCandidate(82, 'Pb', branch, 1., 1.)
    assert excitation_edge_kev(line) == xraylib.EdgeEnergy(82, getattr(xraylib, shell+'_SHELL'))


@pytest.mark.parametrize('completion', [False, True])
@pytest.mark.parametrize('voltage_source', ['configuration', 'sample_metadata'])
def test_below_fe_k_edge_cannot_identify_fe_even_when_kalpha_is_visible(synthetic, completion, voltage_source):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 1000.})
    config = replace(config, elements=('Fe',), complete_atomic_lines=completion)
    # Fe Ka=6.404 < 6.8, but Fe K edge=7.112 > 6.8 keV.
    if voltage_source == 'configuration':
        config.tube_voltage_kv = 6.8
    else:
        frame = pd.read_csv(spectrum)
        frame['tube_voltage_kV'] = 6.8
        frame.to_csv(spectrum, index=False)
    result = analyze(spectrum, calibration, sensitivity, config=config)
    assert result['identification'].fit.symbols == []
    assert result['density'].empty
    assert result['summary']['effective_parameters']['excitation_voltage_source'] == voltage_source
    assert any(r['reason'] == 'initial_shell_not_excitable_at_tube_voltage'
               for r in result['summary']['pruned_lines'])


def test_above_fe_k_edge_retains_fe(synthetic):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 1000.})
    config = replace(config, elements=('Fe',), tube_voltage_kv=7.2)
    result = analyze(spectrum, calibration, sensitivity, config=config)
    assert result['identification'].candidates.set_index('Element').loc['Fe', 'Status'] == 'supported'
    assert result['density'].Relative_to_Reference.notna().all()


def test_sample_voltage_metadata_also_constrains_target_scattering(synthetic):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 1000.})
    config = replace(config, elements=('Fe',), tube_target='Ag', energy_range=(1., 30.))
    frame = pd.read_csv(spectrum)
    frame['tube_voltage_kV'] = 24.
    frame.to_csv(spectrum, index=False)
    result = analyze(spectrum, calibration, sensitivity, config=config)
    assert not any(name.startswith('Ag_') for name in result['identification'].fit.nuisance_names)
    assert instrument_lines(result['config'], [19., 26.]) == []


def test_unexcitable_auxiliary_l1_cannot_enter_the_fit(synthetic, monkeypatch):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 1000.})
    config = replace(config, elements=('Fe', 'Pb'), tube_voltage_kv=14.)
    # Pb L3 is accessible at 14 keV, L1 is not (15.861 keV), although
    # an L1 fluorescence photon can lie below the endpoint.
    import xrf_analyzer.pipeline as pipeline
    original = pipeline.identify_elements
    captured = {}

    def inspect(*args, **kwargs):
        captured['auxiliary'] = kwargs['auxiliary_lines']
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, 'identify_elements', inspect)
    analyze(spectrum, calibration, sensitivity, config=config)
    assert not any(l.symbol == 'Pb' and l.parent_shell == 'L1' for l in captured['auxiliary'])


def test_completed_and_auxiliary_transitions_keep_shell_specific_edges():
    config = AnalysisConfig()
    lines, auxiliary, _ = complete_transitions([
        LineCandidate(82, 'Pb', 'Lalpha1', 10.5512, 5.),
        LineCandidate(82, 'Pb', 'Lbeta1', 12.6137, 2.5),
    ], config)
    assert any(l.parent_shell for l in lines)
    assert auxiliary
    for line in lines + auxiliary:
        shell = line.parent_shell or {'Lalpha1': 'L3', 'Lbeta1': 'L2'}[line.branch]
        assert excitation_edge_kev(line) == xraylib.EdgeEnergy(line.Z, getattr(xraylib, shell+'_SHELL'))
