"""Known target scattering must not become an elemental primary area."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xraylib

from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.fit_model import instrument_lines
from xrf_analyzer.local_quantification import _groups, fit_local_quantification
from xrf_analyzer.response import gaussian_response, sigma_kev


def _case(lines, config, scatter_center, scatter_sigma, scatter_area=2000.):
    edges = np.arange(19., 26.001, .02)
    energy = (edges[:-1]+edges[1:])/2
    mean = np.full(len(energy), 8.)
    for line in lines:
        mean += 500.*line.sensitivity_cm2_g*gaussian_response(
            edges, line.energy_keV, sigma_kev(line.energy_keV, config))
    mean += scatter_area*gaussian_response(edges, scatter_center, scatter_sigma)
    counts = np.random.default_rng(19).poisson(mean)
    fit = SimpleNamespace(symbols=['Pd'], q=np.array([500.]), auxiliary_lines=[],
        auxiliary_areas={}, delta=0., energy_gain=0., resolution_scale=1.,
        line_centers={}, center_axis='reference')
    candidates = pd.DataFrame([{'Element': 'Pd', 'Z': 46, 'Status': 'supported'}])
    ident = SimpleNamespace(fit=fit, fit_lines=lines, candidates=candidates)
    return ident, counts, energy, edges, config


def test_overlapping_compton_peak_does_not_inflate_primary_area():
    pd_alpha = xraylib.LineEnergy(46, xraylib.KA1_LINE)
    ag_alpha = xraylib.LineEnergy(47, xraylib.KA1_LINE)
    angle = np.rad2deg(np.arccos(1.-(ag_alpha/pd_alpha-1.)*510.99895/ag_alpha))
    cfg = AnalysisConfig(fit_response=False, tube_target='Ag', scatter_angle_deg=angle)
    lines = [LineCandidate(46, 'Pd', 'Kalpha1', pd_alpha, 4.),
             LineCandidate(46, 'Pd', 'Kalpha2', xraylib.LineEnergy(46, xraylib.KA2_LINE), 2.),
             LineCandidate(46, 'Pd', 'Kbeta1', xraylib.LineEnergy(46, xraylib.KB1_LINE), .7)]
    case = _case(lines, cfg, pd_alpha, 2*sigma_kev(pd_alpha, cfg))
    local = fit_local_quantification(*case)
    row = local['elements'].iloc[0]
    assert row.Quantification_Status == 'local_area_estimate'
    assert row.q_Area == pytest.approx(500., rel=.08)
    assert row.Sensitivity_Sum == 6.
    assert local['elements'].Element.tolist() == ['Pd']
    wide = local['audit'][0]['Wide']
    assert any('Ag_Compton_Kalpha' in name for name in wide['Nuisance_Groups'])
    assert 'instrument_scattering_free_area_nonquantitative' in wide['Nuisance_Policies']


def test_exact_primary_rayleigh_collision_is_not_identifiable():
    cfg = AnalysisConfig(fit_response=False, tube_target='Ag')
    # A synthetic exact collision isolates the identifiability contract. The
    # identity is supplied; this test does not assert that Pd emits at Ag Ka.
    _, center, width = instrument_lines(cfg, np.array([19., 26.]))[0]
    case = _case([LineCandidate(46, 'Pd', 'Kalpha1', center, 4.)], cfg, center, width)
    result = fit_local_quantification(*case)
    row = result['elements'].iloc[0]
    assert pd.isna(row.q_Area)
    assert row.Failure_Reason == 'primary_area_not_identifiable'


def test_observed_manual_centers_do_not_freeze_instrument_affine_response():
    cfg = AnalysisConfig(find_peaks=(21.17,), find_peaks_axis='observed', tube_target='Ag')
    line = LineCandidate(46, 'Pd', 'Kalpha1', 21.17, 4.)
    ident, _, _, edges, _ = _case([line], cfg, 24., .1, scatter_area=0.)
    ident.fit.delta = .03
    ident.fit.energy_gain = .003
    ident.fit.resolution_scale = 1.1
    ident.fit.center_axis = 'observed'
    ident.fit.line_centers = {('Pd', 'Kalpha1'): 21.17}
    groups = _groups(ident, cfg, edges)
    primary = next(g for g in groups if g['primary'])
    instrument = next(g for g in groups if g.get('instrument'))
    _, center, width = instrument_lines(cfg, edges)[0]
    assert primary['centers'][0] == 21.17
    assert instrument['centers'][0] == pytest.approx(center*1.003+.03)
    assert instrument['sigmas'][0] == pytest.approx(width*1.1)


def test_target_k_scattering_requires_voltage_above_target_k_edge():
    edges = np.array([19., 26.])
    # Ag Ka is 22.16 keV, but its K edge is about 25.51 keV.
    cfg = AnalysisConfig(tube_target='Ag', tube_voltage_kv=24.)
    assert instrument_lines(cfg, edges) == []
    cfg.tube_voltage_kv = 30.
    assert any('Rayleigh_Kalpha' in name for name, _, _ in instrument_lines(cfg, edges))
