"""Physical contracts for weak candidates, auxiliary shells and report policy."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xrf_analyzer.atomic_completion import AuxiliaryLine
from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.element_prior import apply_reporting_policy
from xrf_analyzer.local_quantification import _groups, _tie_unconfirmed_nuisance
from xrf_analyzer.pattern_matching import discover_elements, anchored_primary_symbols


def test_single_kalpha_remains_testable_when_another_pattern_covers_its_peak():
    cfg = AnalysisConfig(fit_response=False, complete_atomic_lines=False)
    # Deliberately coincident toy competitor: these are not real Hg energies.
    # Peak coverage must propose hypotheses without deciding their identity.
    lines = [LineCandidate(20, 'Ca', 'Kalpha1', 3.69, 1.),
             LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 1.),
             LineCandidate(26, 'Fe', 'Kbeta1', 7.06, .1),
             LineCandidate(80, 'Hg', 'Lalpha1', 3.69, 1.),
             LineCandidate(80, 'Hg', 'Lbeta1', 6.4, 1.)]
    peaks = pd.DataFrame([dict(peak_id=i, energy_keV=e, approx_significance=s,
        centroid_std_keV=.005, estimated_fwhm_keV=.135, quality_flags='', net_area_counts=50.)
        for i, (e, s) in enumerate([(3.69, 5.), (6.4, 50.), (7.06, 3.6)], 1)])
    energy = np.arange(3., 9., .02)
    result = discover_elements(peaks, lines, [], energy, np.ones(len(energy))*10, cfg)
    assert 'Hg' in result.selected
    assert 'Ca' in result.selected
    assert result.rankings.set_index('Element').loc['Ca', 'PatternScore'] < 0
    assert 'Ca' in result.alignment['anchored_kalpha_candidates']


def test_supported_element_does_not_free_individual_unknown_shell_lines():
    cfg = AnalysisConfig(fit_response=False)
    primary = LineCandidate(80, 'Hg', 'Lalpha1', 9.99, 5.)
    auxiliary = [AuxiliaryLine(80, 'Hg', 'L1M2', 11., .25, 'L1'),
                 AuxiliaryLine(80, 'Hg', 'L1M3', 12., .75, 'L1')]
    fit = SimpleNamespace(symbols=['Hg'], q=np.array([100.]), auxiliary_lines=auxiliary,
        auxiliary_areas={'Hg_L1_unscaled': 400.}, delta=0., energy_gain=0.,
        resolution_scale=1., line_centers={}, center_axis='reference')
    ident = SimpleNamespace(fit=fit, fit_lines=[primary],
        candidates=pd.DataFrame([{'Element': 'Hg', 'Status': 'supported'}]))
    groups = _groups(ident, cfg, np.array([8., 14.]))
    nuisance = _tie_unconfirmed_nuisance(groups)
    shell = [g for g in nuisance if g['auxiliary']]
    assert len(shell) == 1
    assert shell[0]['expected'] == pytest.approx(400.)
    assert shell[0]['weights'].tolist() == pytest.approx([.25, .75])
    assert [g for g in nuisance if not g['auxiliary']][0]['confirmed']


@pytest.mark.parametrize('policy', ['supported-only', 'all-candidates'])
@pytest.mark.parametrize('gas_policy', ['supported-only', 'all-candidates'])
def test_rare_group_policy_preserves_fits_and_strong_rare_elements(policy, gas_policy):
    candidates = pd.DataFrame([
        {'Z': 20, 'Element': 'Ca', 'Status': 'tentative', 'q_Fit': 3.},
        {'Z': 57, 'Element': 'La', 'Status': 'tentative', 'q_Fit': 4.},
        {'Z': 61, 'Element': 'Pm', 'Status': 'unidentifiable', 'q_Fit': 8.},
        {'Z': 65, 'Element': 'Tb', 'Status': 'supported', 'q_Fit': 500.},
        {'Z': 71, 'Element': 'Lu', 'Status': 'ambiguous', 'q_Fit': 2.},
        {'Z': 18, 'Element': 'Ar', 'Status': 'tentative', 'q_Fit': 5.},
        {'Z': 54, 'Element': 'Xe', 'Status': 'supported', 'q_Fit': 500.},
        {'Z': 118, 'Element': 'Og', 'Status': 'unidentifiable', 'q_Fit': 1.}])
    original = candidates.copy(deep=True)
    result = apply_reporting_policy(candidates, policy, gas_policy).set_index('Element')
    assert result.loc['Ca', 'Reportable']
    assert result.loc['Tb', 'Reportable']
    assert result.loc['Tb', 'Status'] == 'supported'
    assert bool(result.loc['Pm', 'Reportable']) == (policy == 'all-candidates')
    assert bool(result.loc['La', 'Reportable']) == (policy == 'all-candidates')
    assert bool(result.loc['Lu', 'Reportable']) == (policy == 'all-candidates')
    assert result.loc['Xe', 'Reportable']
    assert bool(result.loc['Ar', 'Reportable']) == (gas_policy == 'all-candidates')
    assert bool(result.loc['Og', 'Reportable']) == (gas_policy == 'all-candidates')
    pd.testing.assert_frame_equal(candidates, original)
    pd.testing.assert_frame_equal(result.reset_index()[original.columns], original)


def test_invalid_lanthanide_policy_fails_before_analysis():
    with pytest.raises(ValueError, match='lanthanide_reporting'):
        AnalysisConfig(lanthanide_reporting='erase-all').validate()
    with pytest.raises(ValueError, match='noble_gas_reporting'):
        AnalysisConfig(noble_gas_reporting='erase-all').validate()


def test_strong_borrowed_minor_peaks_do_not_free_hypothetical_nuisance_lines():
    peaks = pd.DataFrame([{'peak_id': 1, 'approx_significance': 8., 'quality_flags': ''},
                          {'peak_id': 2, 'approx_significance': 80., 'quality_flags': ''}])
    pairs = pd.DataFrame([{'Element': 'Pm', 'Branch': 'Lalpha1', 'peak_id': 1,
                           'Group_Affinity': 1., 'Strength_Source': 'csv'},
                          {'Element': 'Pm', 'Branch': 'Lbeta2', 'peak_id': 2,
                           'Group_Affinity': 1., 'Strength_Source': 'csv'}])
    rankings = pd.DataFrame([{'Element': 'Pm', 'IndependentPeakRegions': 2, 'PatternScore': 10.}])
    cfg = AnalysisConfig()
    assert not anchored_primary_symbols(peaks, pairs, rankings, cfg)
    # Strong, balanced rare-element evidence still permits independent groups.
    peaks.loc[peaks.peak_id == 1, 'approx_significance'] = 60.
    assert anchored_primary_symbols(peaks, pairs, rankings, cfg) == {'Pm'}
