"""Independent net-area measurements must survive unrelated global mismatch."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.local_quantification import fit_local_quantification
from xrf_analyzer.response import gaussian_response, sigma_kev


def _case(lines, truth, statuses=None, supplied_q=None, extra=None, config=None, seed=19, max_energy=16.):
    cfg = config or AnalysisConfig(fit_response=False)
    energy = np.arange(1., max_energy+.001, .02)
    edges = np.r_[energy-.01, energy[-1]+.01]
    mean = np.full(len(energy), 8.)
    for line in lines:
        mean += truth.get((line.symbol, line.branch), truth.get(line.symbol, 0.))*line.sensitivity_cm2_g*gaussian_response(
            edges, line.energy_keV, sigma_kev(line.energy_keV, cfg))
    if extra is not None:
        mean += extra(energy, edges)
    counts = np.random.default_rng(seed).poisson(mean)
    symbols = list(dict.fromkeys(l.symbol for l in lines))
    fit = SimpleNamespace(symbols=symbols, q=np.array([(supplied_q or {}).get(s, truth.get(s, 1.)) for s in symbols]),
                          auxiliary_lines=[], auxiliary_areas={}, delta=0., energy_gain=0.,
                          resolution_scale=1., line_centers={}, center_axis='reference')
    candidates = pd.DataFrame([{'Element': s, 'Z': next(l.Z for l in lines if l.symbol == s),
                                'Status': (statuses or {}).get(s, 'supported')} for s in symbols])
    identification = SimpleNamespace(fit=fit, fit_lines=lines, candidates=candidates)
    return identification, counts, energy, edges, cfg


def _fit(case):
    return fit_local_quantification(*case)


def test_primary_area_divided_by_sum_s_recovers_known_scale_despite_wrong_global_q():
    lines = [LineCandidate(26, 'Fe', 'Kalpha1', 6.4039, 8.),
             LineCandidate(26, 'Fe', 'Kalpha2', 6.3909, 4.),
             LineCandidate(26, 'Fe', 'Kbeta1', 7.058, 1.5)]
    # Beta does not follow the global scalar q, and a distant broad feature is
    # deliberately absent from the fluorescence model. Neither changes Ka/S.
    case = _case(lines, {'Fe': 500., ('Fe', 'Kbeta1'): 20.}, supplied_q={'Fe': 200.},
                 extra=lambda e, edges: 20000.*gaussian_response(edges, 13., .4))
    result = _fit(case)
    row = result['elements'].iloc[0]
    assert row.Quantification_Status == 'local_area_estimate'
    assert row.q_Area == pytest.approx(500., rel=.05)
    assert row.Area_Counts/row.Sensitivity_Sum == pytest.approx(row.q_Area)
    assert row.Sensitivity_Sum == pytest.approx(12.)
    assert row.q_Std > 0 and row.q_Stat_Std > 0
    assert case[0].fit.q[0] == 200.  # The joint identity model is not overwritten.


def test_resolved_kalpha_doublet_measures_one_common_q_with_total_sensitivity():
    import xraylib
    lines = [LineCandidate(51, 'Sb', 'Kalpha1', xraylib.LineEnergy(51, xraylib.KA1_LINE), 1.5),
             LineCandidate(51, 'Sb', 'Kalpha2', xraylib.LineEnergy(51, xraylib.KA2_LINE), .75),
             LineCandidate(51, 'Sb', 'Kbeta1', xraylib.LineEnergy(51, xraylib.KB1_LINE), .25)]
    # The two alpha lines are resolved at this SDD width; beta deliberately has
    # a different scale so it cannot be used to force the primary amplitude.
    result = _fit(_case(lines, {'Sb': 200., ('Sb', 'Kbeta1'): 20.}, max_energy=32.))
    row = result['elements'].iloc[0]
    assert row.Quantification_Status == 'local_area_estimate'
    assert row.q_Area == pytest.approx(200., rel=.12)
    assert row.Sensitivity_Sum == pytest.approx(2.25)
    assert set(row.Selected_Branches.split('+')) == {'Kalpha1', 'Kalpha2'}
    assert result['audit'][0]['Primary_Group_Policy'] == 'same_alpha_doublet_common_area_over_summed_sensitivity'
    assert not any('Sb:Kalpha' in name for name in result['audit'][0]['Wide']['Nuisance_Groups'])


def test_shared_channels_retain_covariance_between_overlapping_primary_windows():
    lines = [LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.),
             LineCandidate(27, 'Co', 'Kalpha1', 6.58, 6.)]
    result = _fit(_case(lines, {'Fe': 500., 'Co': 350.}))
    values = result['elements'].set_index('Element')
    assert values.loc['Fe', 'q_Area'] == pytest.approx(500., rel=.06)
    assert values.loc['Co', 'q_Area'] == pytest.approx(350., rel=.06)
    covariance = result['stat_covariance']
    assert covariance[0, 1] < 0  # Deblending gives anti-correlated amplitudes.
    assert np.all(np.linalg.eigvalsh(covariance) > 0)
    assert result['covariance'][0, 1] != 0


def test_exact_unresolved_elemental_interference_is_not_falsely_quantified():
    lines = [LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.),
             LineCandidate(27, 'Co', 'Kalpha1', 6.4, 6.)]
    result = _fit(_case(lines, {'Fe': 500., 'Co': 350.}))
    assert result['elements'].q_Area.isna().all()
    assert set(result['elements'].Failure_Reason) == {'primary_area_not_identifiable'}
    assert not np.isfinite(result['covariance']).any()


def test_unconfirmed_nuisance_retains_same_shell_strong_anchor():
    # Fake L energies isolate the logical failure mode: a very weak competitor
    # gamma at Fe Ka cannot be amplified freely while ignoring its large beta.
    lines = [LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.),
             LineCandidate(61, 'Pm', 'Lbeta1', 5.9, 8.),
             LineCandidate(61, 'Pm', 'Lgamma1', 6.45, .2)]
    case = _case(lines, {'Fe': 500., 'Pm': 50.}, statuses={'Pm': 'tentative'})
    result = _fit(case)
    assert result['elements'].Element.tolist() == ['Fe']
    assert result['elements'].iloc[0].q_Area == pytest.approx(500., rel=.06)
    evidence = result['audit'][0]
    assert evidence['Accepted']
    assert any('Pm:L2' in x for x in evidence['Wide']['Interferent_Anchor_Extensions'])
    assert any('same_initial_shell_L2' in x for x in evidence['Wide']['Nuisance_Policies'])


def test_manual_reference_centers_remain_fixed_during_local_measurement():
    cfg = AnalysisConfig(find_peaks=(6.4,), fit_response=True)
    case = _case([LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.)], {'Fe': 500.}, config=cfg)
    result = _fit(case)
    assert result['elements'].iloc[0].Quantification_Status == 'local_area_estimate'
    assert result['audit'][0]['Narrow']['Local_Shift_keV'] == 0.
    assert result['audit'][0]['Wide']['Local_Shift_keV'] == 0.


def test_no_supported_elements_returns_schema_correct_empty_measurement():
    case = _case([LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.)], {'Fe': 500.}, statuses={'Fe': 'tentative'})
    result = _fit(case)
    assert result['elements'].empty
    assert {'Element', 'q_Area', 'q_Std', 'Quantification_Status'} <= set(result['elements'])
    assert result['covariance'].shape == (0, 0)


def test_fit_arrays_are_reviewable_and_add_up_to_total_model():
    result = _fit(_case([LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.)], {'Fe': 500.}))
    assert len(result['fits']) == 1
    fit = result['fits'][0]
    assert np.allclose(fit['model'], fit['primary_counts']+fit['background']+fit['nuisance_counts'])
    assert len(fit['indices']) == len(fit['model'])
    assert np.all(fit['background'] > 0)
