"""Low-Z identity evidence is distinct from a quantitative ratio estimate."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xraylib

from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.low_z import assess_low_z_kalpha
from xrf_analyzer.response import gaussian_response, sigma_kev


def _case(symbol='Ca', beta_strength=.001, missing_beta=False, overlap=False,
          free_response=False):
    """Known expected spectrum with independent, explicit background/response.

    The branch diagnostic input represents the separately profiled family test.
    Deterministic expected counts keep weak-beta and overlap decisions independent
    of an arbitrary noise realization or the joint optimizer's initialization.
    """
    cfg = AnalysisConfig(fit_response=free_response).validate()
    energy = np.arange(1., 9.001, .02)
    edges = np.r_[energy-.01, energy[-1]+.01]
    z = xraylib.SymbolToAtomicNumber(symbol)
    lines = [LineCandidate(z, symbol, branch, xraylib.LineEnergy(z, constant), strength)
             for branch, constant, strength in [
                 ('Kalpha1', xraylib.KA1_LINE, 2.),
                 ('Kalpha2', xraylib.KA2_LINE, 1.),
                 ('Kbeta1', xraylib.KB1_LINE, beta_strength)]]
    q = 300.
    responses = [q*l.sensitivity_cm2_g*gaussian_response(
        edges, l.energy_keV, sigma_kev(l.energy_keV, cfg)) for l in lines]
    own = np.sum(responses, axis=0)
    background = np.full(len(energy), 8.)
    components = own[:, None]
    symbols = [symbol]
    q_values = [q]
    if overlap:
        # An unresolved competing element has more counts in the same Ka core.
        rival = 1.2*(responses[0]+responses[1])
        components = np.column_stack([own, rival])
        symbols.append('Pm')
        q_values.append(1.)
    model = background+components.sum(axis=1)
    counts = model-responses[-1] if missing_beta else model.copy()
    fit = SimpleNamespace(symbols=symbols, q=np.array(q_values),
                          matrix=components.copy(), t=np.ones(len(symbols)),
                          components=components, model=model, background=background,
                          resolution_scale=1., line_centers={}, center_axis='reference',
                          energy_gain=0., delta=0., gain_fitted=False)
    center = np.average([l.energy_keV for l in lines[:2]], weights=[2., 1.])
    peaks = pd.DataFrame([{'peak_id': 0, 'energy_keV': center, 'quality_flags': ''}])
    pairs = pd.DataFrame([{'Element': symbol, 'Branch': 'Kalpha1',
                           'Group_Affinity': 1., 'peak_id': 0}])
    beta_details = [{'Family': 'K', 'Branches': 'Kbeta1',
                     'Group_Observed': not missing_beta,
                     'Robust_Group_Observed': not missing_beta,
                     'Tied_vs_Free_DeltaDeviance': 100. if missing_beta else 0.}]
    return cfg, lines, peaks, pairs, counts, energy, edges, fit, beta_details


def _assess(case, details=None):
    cfg, lines, peaks, pairs, counts, energy, edges, fit, beta_details = case
    return assess_low_z_kalpha(lines[0].symbol, lines, peaks, pairs, None,
                              counts, energy, edges, fit, cfg,
                              beta_details if details is None else details)


@pytest.mark.parametrize('symbol', ['Ca', 'Cr'])
def test_strong_kalpha_weak_undetected_kbeta_can_support_identity(symbol):
    evidence = _assess(_case(symbol, missing_beta=True))
    assert evidence['Eligible']
    assert min(evidence['Kalpha_DeltaDeviance'], evidence['Wide_Kalpha_DeltaDeviance']) >= 25
    assert evidence['Expected_Kbeta_Information_Score'] < 5
    assert not evidence['Kbeta_Strongly_Contradicted']
    assert not evidence['Kbeta_Independently_Required']


@pytest.mark.parametrize('symbol', ['Al', 'Si'])
def test_single_unresolved_low_z_k_complex_does_not_require_two_groups(symbol):
    # branch_evidence intentionally has no separate groups in this limit.
    evidence = _assess(_case(symbol, missing_beta=True), details=[])
    assert evidence['Applicable'] and evidence['Eligible']
    assert evidence['Calibration_Anchored']


@pytest.mark.parametrize('symbol', ['Mn', 'Fe'])
def test_above_z24_does_not_use_kalpha_only_exception(symbol):
    evidence = _assess(_case(symbol))
    assert not evidence['Applicable']
    assert not evidence['Eligible']


def test_lone_kalpha_cannot_fit_and_independently_anchor_its_calibration():
    evidence = _assess(_case(free_response=True))
    assert not evidence['Calibration_Anchored']
    assert not evidence['Eligible']
    assert any('free calibration' in reason for reason in evidence['Reasons'])


def _other_anchor_case(primary_anchor=True):
    case = list(_case(free_response=True))
    cfg, _, peaks, pairs, _, energy, edges, fit, _ = case
    fe_energies = [xraylib.LineEnergy(26, line) for line in (xraylib.KA1_LINE, xraylib.KB1_LINE)]
    fe = sum(600.*gaussian_response(edges, center, sigma_kev(center, cfg)) for center in fe_energies)
    fit.symbols.append('Fe')
    fit.components = np.column_stack([fit.components, fe])
    fit.matrix = np.column_stack([fit.matrix, fe if primary_anchor else np.zeros(len(fe))])
    fit.t = np.ones(2)
    fit.q = np.r_[fit.q, 1.]
    fit.model += fe
    case[4] = fit.model.copy()
    for peak_id, (center, branch) in enumerate(zip(fe_energies, ['Kalpha1', 'Kbeta1']), 1):
        peaks.loc[len(peaks)] = [peak_id, center, '']
        pairs.loc[len(pairs)] = ['Fe', branch, 1., peak_id]
    return case


@pytest.mark.parametrize('primary_anchor', [False, True])
def test_other_element_anchor_requires_primary_template_not_free_auxiliary(primary_anchor):
    case = _other_anchor_case(primary_anchor)
    evidence = _assess(case)
    assert evidence['Calibration_Anchored'] == primary_anchor
    assert evidence['Eligible'] == primary_anchor


def test_auxiliary_branch_position_cannot_supply_second_independent_anchor():
    case = _other_anchor_case()
    assert _assess(case)['Eligible']
    pairs = case[3]
    pairs['Strength_Source'] = 'csv'
    pairs.loc[(pairs.Element == 'Fe') & (pairs.Branch == 'Kbeta1'),
              'Strength_Source'] = 'unscaled_auxiliary_shell'
    evidence = _assess(case)
    assert not evidence['Calibration_Anchored']
    assert not evidence['Eligible']


def test_low_z_peak_shared_by_another_candidate_is_not_independent_anchor():
    case = _other_anchor_case()
    assert _assess(case)['Eligible']
    pairs = case[3]
    # Fe has a primary anchor elsewhere and an alternative pairing to Ca's Ka.
    pairs.loc[(pairs.Element == 'Fe') & (pairs.Branch == 'Kbeta1'), 'peak_id'] = 0
    evidence = _assess(case)
    assert not evidence['Calibration_Anchored']
    assert not evidence['Eligible']


@pytest.mark.parametrize('broad', [False, True])
def test_second_anchor_needs_local_primary_signal_and_a_narrow_peak(broad):
    case = _other_anchor_case()
    assert _assess(case)['Eligible']
    if broad:
        case[2].loc[case[2].peak_id == 2, 'quality_flags'] = 'broad_structure'
    else:
        # A real-looking peak elsewhere cannot borrow information from strong Fe Ka.
        case[7].matrix[case[5] > 6.8, 1] = 0.
    evidence = _assess(case)
    assert not evidence['Calibration_Anchored'] and not evidence['Eligible']


def test_detectable_kbeta_absence_still_contradicts_low_z_family():
    evidence = _assess(_case('Cr', beta_strength=1., missing_beta=True))
    assert evidence['Expected_Kbeta_Information_Score'] >= 5
    assert evidence['Kbeta_Strongly_Contradicted']
    assert not evidence['Eligible']


def test_strong_but_unresolved_competing_element_prevents_single_region_support():
    evidence = _assess(_case(overlap=True))
    assert evidence['Kalpha_DeltaDeviance'] >= 25
    assert evidence['Kalpha_Share_vs_Strongest_Interferent'] < .7
    assert not evidence['Eligible']


def test_disabling_low_z_exception_keeps_strict_multi_region_path():
    case = _case()
    case[0].low_z_kalpha = False
    evidence = _assess(case)
    assert not evidence['Applicable'] and not evidence['Eligible']


@pytest.mark.parametrize('symbol', ['Al', 'Ca', 'Cr'])
def test_kalpha_only_identity_reaches_pipeline_supported_status(synthetic, symbol):
    from xrf_analyzer.pipeline import analyze
    spectrum, calibration, sensitivity, cfg, _, _ = synthetic(
        {symbol: 300.}, branches={symbol: {'Kalpha1': 2., 'Kalpha2': 1., 'Kbeta1': .001}})
    cfg.elements = (symbol,)
    cfg.fit_response = False
    cfg.estimate_resolution_from_peaks = False
    cfg.complete_atomic_lines = False
    result = analyze(spectrum, calibration, sensitivity, config=cfg)
    row = result['identification'].candidates.set_index('Element').loc[symbol]
    assert row.Status == 'supported'
    assert 'low_Z_Kalpha_dominant_support' in row.ConfidenceNotes
    evidence = next(d for d in result['summary']['low_z_kalpha_evidence'] if d['Element'] == symbol)
    assert evidence['Eligible']
    # The reported ratio includes only an element whose identity was supported.
    assert result['density'].Element.tolist() == [symbol]
