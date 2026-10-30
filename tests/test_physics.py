import numpy as np
import pytest
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.response import gaussian_response, response_with_derivatives, sigma_kev
from xrf_analyzer.baseline import snip_background
from xrf_analyzer.fit_model import fit_spectrum
from xrf_analyzer.uncertainty import estimate_covariance
from xrf_analyzer.atomic_lines import LineCandidate


def test_integrated_response_and_derivatives():
    edges = np.arange(0, 15, .02)
    p, d, w = response_with_derivatives(edges, [6.4], [.06], delta=.003, scale=1.05)
    assert p.sum() == pytest.approx(1)
    h = 1e-6
    pd = response_with_derivatives(edges, [6.4], [.06], delta=.003+h, scale=1.05)[0]
    pw = response_with_derivatives(edges, [6.4], [.06], delta=.003, scale=1.05+h)[0]
    assert np.allclose((pd-p)/h, d, atol=3e-5)
    assert np.allclose((pw-p)/h, w, atol=1e-6)
    assert gaussian_response(np.linspace(6.4, 8, 100), 6.4, .06).sum() == pytest.approx(.5)


def test_single_peak_recovers_area_center_and_width():
    config = AnalysisConfig(energy_range=(5, 8))
    edges = np.arange(5, 8.0001, .01)
    e = (edges[:-1]+edges[1:])/2
    line = LineCandidate(26, 'Fe', 'Kalpha1', 6.4039, 8.)
    sig = sigma_kev(line.energy_keV, config)
    mu = 10+50000*gaussian_response(edges, 6.4159, sig*1.1)
    y = np.random.default_rng(42).poisson(mu)
    fit = fit_spectrum(y, e, edges, np.full(len(e), 10.), [line], config, .03)
    assert fit.success
    assert fit.t[0] == pytest.approx(50000, rel=.025)
    assert fit.delta == pytest.approx(.012, abs=.003)
    assert fit.resolution_scale == pytest.approx(1.1, abs=.035)
    cov, states, _ = estimate_covariance(fit)
    assert states == ['fisher_interior'] and cov[0, 0] > 0


def test_background_finite_and_nonnegative():
    y = np.random.default_rng(3).poisson(5+np.linspace(0, 30, 500))
    b = snip_background(y, 25)
    assert np.isfinite(b).all() and (b >= 0).all()
    assert np.all(snip_background(np.zeros(50), 30) == 0)


def test_config_failure():
    with pytest.raises(ValueError):
        AnalysisConfig(fwhm_ref_kev=.01).validate()
    with pytest.raises(ValueError):
        AnalysisConfig(detector_efficiency=.8).validate()
    with pytest.raises(ValueError):
        AnalysisConfig(energy_range=(9, 2)).validate()
