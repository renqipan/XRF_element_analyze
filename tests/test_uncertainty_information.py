"""Information must respect fixed, boundary and unidentifiable parameters."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.local_quantification import fit_local_quantification
from xrf_analyzer.quantification import compute_relative_areal_density
from xrf_analyzer.response import gaussian_response, sigma_kev
from xrf_analyzer.uncertainty import estimate_covariance


def _fit(jacobian, alpha=1., fixed_alpha=False):
    jacobian = np.asarray(jacobian, dtype=float)
    return SimpleNamespace(
        symbols=['Fe'], parameters=np.array([100., 0., alpha]),
        bounds=[(0., None), (-3., 3.), (alpha, alpha) if fixed_alpha else (0., None)],
        nuisance_names=[], amplitude_scale=1., sensitivity_sums=np.array([1.]),
        jacobian=jacobian, model=np.full(jacobian.shape[0], 10.),
        q=np.array([100.]), success=True, response_at_boundary=False,
        background_scale=alpha, background_scale_index=2,
    )


def test_fixed_background_scale_has_zero_error_and_reduced_information():
    jacobian = np.array([[1., 1., 3.], [2., 1., 1.], [4., 1., 2.], [1., 1., 4.]])
    fit = _fit(jacobian, fixed_alpha=True)
    covariance, statuses, _ = estimate_covariance(fit)
    # Independently invert information for the two parameters actually fitted.
    expected = np.linalg.inv(jacobian[:, :2].T @ jacobian[:, :2] / 10.)
    assert covariance[0, 0] == pytest.approx(expected[0, 0])
    assert statuses == ['fisher_interior']
    assert fit.background_scale_uncertainty_status == 'fixed_parameter'
    assert fit.background_scale_std == 0.
    assert np.all(fit.background_scale_influence_sample == 0.)
    assert np.all(fit.background_scale_influence_reference == 0.)


def test_unidentifiable_scale_does_not_get_pseudoinverse_standard_error():
    fit = _fit([[1., 0., 0.], [0., 1., 1.], [0., 1., 1.], [0., 1., 1.]])
    covariance, statuses, info = estimate_covariance(fit)
    # Fe is independently measured, although the continuum/scale split is not.
    assert covariance[0, 0] == pytest.approx(10.)
    assert statuses == ['fisher_interior']
    assert info['singular_directions'] == 1
    assert fit.background_scale_uncertainty_status == 'not_estimable'
    assert fit.background_scale_std is None
    assert fit.background_scale_influence_sample is None
    assert fit.background_scale_influence_reference is None


def test_scale_at_nonnegative_boundary_has_no_symmetric_error_or_stale_influence():
    fit = _fit([[1., 1., 3.], [2., 1., 1.], [4., 1., 2.], [1., 1., 4.]])
    estimate_covariance(fit)
    assert fit.background_scale_std > 0
    fit.parameters[2] = fit.background_scale = 0.
    estimate_covariance(fit)
    assert fit.background_scale_uncertainty_status == 'nonnegative_boundary'
    assert fit.background_scale_std is None
    assert fit.background_scale_influence_sample is None
    assert fit.background_scale_influence_reference is None


def test_fixed_element_amplitude_is_not_treated_as_fitted_uncertainty():
    fit = _fit([[1., 1., 3.], [2., 1., 1.], [4., 1., 2.], [1., 1., 4.]])
    fit.bounds[0] = (100., 100.)
    covariance, statuses, _ = estimate_covariance(fit)
    assert statuses == ['fixed_parameter']
    assert np.isnan(covariance[0, 0])


def test_failed_fit_does_not_report_scale_information():
    fit = _fit([[1., 1., 3.], [2., 1., 1.], [4., 1., 2.], [1., 1., 4.]])
    fit.success = False
    estimate_covariance(fit)
    assert fit.background_scale_uncertainty_status == 'fit_failed'
    assert fit.background_scale_std is None


@pytest.mark.parametrize('scale_status,scale_std,expected', [
    ('not_estimable', None, 'background_scale_uncertainty_not_estimable'),
    ('nonnegative_boundary', None, 'background_scale_boundary_profile_required'),
    ('fixed_parameter', 0., 'local_area_estimate'),
])
def test_unresolved_background_scale_retains_conditional_area_and_withholds_ratio(
        scale_status, scale_std, expected):
    cfg = AnalysisConfig(fit_response=False)
    edges = np.arange(5., 8.001, .02)
    energy = (edges[:-1]+edges[1:])/2
    line = LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.)
    mean = 8.+4000.*gaussian_response(edges, 6.4, sigma_kev(6.4, cfg))
    counts = np.random.default_rng(42).poisson(mean)
    cfg._background_counts = np.full(len(energy), 3.)
    cfg._background_scale = 0.
    cfg._background_scale_std = scale_std
    fit = SimpleNamespace(
        symbols=['Fe'], q=np.array([500.]), auxiliary_lines=[], auxiliary_areas={},
        delta=0., energy_gain=0., resolution_scale=1., line_centers={}, center_axis='reference',
        background_scale_uncertainty_status=scale_status,
    )
    candidates = pd.DataFrame([{'Element': 'Fe', 'Z': 26, 'Status': 'supported',
                                'q_Fit': 500., 'Quantification_Status': 'supported'}])
    identification = SimpleNamespace(fit=fit, fit_lines=[line], candidates=candidates)
    local = fit_local_quantification(identification, counts, energy, edges, cfg)
    row = local['elements'].iloc[0]
    assert row.q_Area == pytest.approx(500., rel=.08)
    assert row.q_Std > 0
    assert row.Quantification_Status == expected
    density, _ = compute_relative_areal_density(identification, 'Fe', local)
    if scale_status == 'fixed_parameter':
        assert density.iloc[0].Relative_to_Reference == 1.
    else:
        assert pd.isna(density.iloc[0].Relative_to_Reference)
        assert density.iloc[0].Measurement_Failure_Reason == expected
