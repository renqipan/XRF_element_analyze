"""A stricter pending list must not alter identities, areas or density."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.candidate_reporting import filter_possible_candidates
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.pipeline import analyze


def _fixture(delta=6.5, anchored=True, status='tentative', branch=(), reviews=(), overlaps=()):
    cfg = AnalysisConfig(fit_response=False)
    candidates = pd.DataFrame([dict(Z=26, Element='Fe', Status=status, DeltaDeviance=delta,
        q_Fit=10., Reportable=True, Reporting_Reason='')])
    ident = SimpleNamespace(fit=SimpleNamespace(delta=0., energy_gain=0., resolution_scale=1.,
        center_axis='reference', line_centers={}), fit_lines=[LineCandidate(26, 'Fe', 'Kalpha1', 6.4039, 1.)],
        branch_evidence=branch, uncertainty_info={'low_z_counterfactual_reviews': reviews}, overlap_pairs=overlaps)
    peaks = pd.DataFrame([dict(energy_keV=6.4039 if anchored else 7.5,
                               approx_significance=12., quality_flags='')])
    return candidates, ident, peaks, cfg


@pytest.mark.parametrize('delta,anchored,reported', [(6.5,True,False), (15.,False,False), (15.,True,True)])
def test_pending_requires_stronger_evidence_and_observed_principal(delta, anchored, reported):
    args = _fixture(delta, anchored)
    original = args[0].copy(deep=True)
    out = filter_possible_candidates(*args)
    assert bool(out.loc[0, 'Reportable']) == reported
    pd.testing.assert_frame_equal(args[0], original)
    assert out.loc[0, 'Status'] == original.loc[0, 'Status']
    assert out.loc[0, 'q_Fit'] == original.loc[0, 'q_Fit']


def test_suppressed_global_area_can_still_have_robust_local_primary_evidence():
    branch = [dict(Element='Fe', Branches='Kalpha1', Branch_DeltaDeviance=40.,
                   Wide_Window_Branch_DeltaDeviance=35.)]
    out = filter_possible_candidates(*_fixture(delta=0., branch=branch))
    assert out.loc[0, 'Reportable']
    assert out.loc[0, 'Possible_Reporting_Evidence'] == 35.
    assert out.loc[0, 'Status'] == 'tentative'


def test_counterfactual_interpretation_stays_possible_without_promotion():
    review = dict(Element='Fe', Recovered_Possible_Interpretation=True,
        Counterfactual_Element_DeltaDeviance=50.,
        Counterfactual_Kalpha_Evidence={'Kalpha_DeltaDeviance':35., 'Wide_Kalpha_DeltaDeviance':32.})
    out = filter_possible_candidates(*_fixture(delta=0., reviews=[review]))
    assert out.loc[0, 'Reportable']
    assert out.loc[0, 'Possible_Reporting_Evidence_Source'].startswith('counterfactual')
    assert out.loc[0, 'Status'] == 'tentative'


def test_exactly_unresolved_group_is_not_erased_by_zero_individual_deletion_gain():
    pair = {'elements':['Fe','Co'], 'weighted_cosine':1.}
    out = filter_possible_candidates(*_fixture(delta=0., status='unidentifiable', overlaps=[pair]))
    assert out.loc[0, 'Reportable']
    assert out.loc[0, 'Status'] == 'unidentifiable'


def test_supported_identity_is_unaffected_and_zero_restores_legacy_pending_policy():
    args = _fixture(delta=0., anchored=False, status='supported')
    assert filter_possible_candidates(*args).loc[0, 'Reportable']
    candidates, ident, peaks, cfg = _fixture(delta=0., anchored=False)
    assert filter_possible_candidates(candidates, ident, peaks,
        replace(cfg, possible_min_delta_deviance=0.)).loc[0, 'Reportable']


@pytest.mark.parametrize('value', [-1., np.nan, np.inf])
def test_invalid_possible_floor_fails_before_fitting(value):
    with pytest.raises(ValueError, match='possible_min_delta_deviance'):
        AnalysisConfig(possible_min_delta_deviance=value).validate()


def test_real_reporting_tightening_preserves_every_fitted_parameter_and_density():
    root = Path(__file__).resolve().parents[1]
    spectrum = root/'XRF_SDD/pos1_40keV_80uA.mca'
    if not spectrum.exists():
        pytest.skip('Optional local measurement was removed')
    cfg = AnalysisConfig(**json.loads((root/'configs/default.json').read_text()))
    args = (spectrum, root/'data/calibration.csv', root/'data/xrf_relative_sensitivity.csv')
    before = analyze(*args, config=replace(cfg, possible_min_delta_deviance=0.))
    after = analyze(*args, config=cfg)
    np.testing.assert_array_equal(before['identification'].fit.parameters, after['identification'].fit.parameters)
    pd.testing.assert_frame_equal(before['density'], after['density'])
    pd.testing.assert_series_equal(before['identification'].candidates.Status, after['identification'].candidates.Status)
    assert set(after['summary']['elements_need_confirmation']) < set(before['summary']['elements_need_confirmation'])
    assert 'Ni' in after['summary']['elements_need_confirmation']
    assert 'Tc' not in after['summary']['elements_need_confirmation']
