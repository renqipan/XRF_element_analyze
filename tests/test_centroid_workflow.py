"""Known-truth regressions for centroid-first recognition and atomic completeness."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xraylib
from xrf_analyzer.atomic_lines import LINE_MAP, LineCandidate, load_sensitivity
from xrf_analyzer.atomic_completion import complete_transitions
from xrf_analyzer.calibration import fit_linear_calibration
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.io_spectrum import Spectrum
from xrf_analyzer.measurement import infer_hard_lower_cutoff
from xrf_analyzer.peak_detection import detect_peaks
from xrf_analyzer.pipeline import analyze
from xrf_analyzer.response import gaussian_response, sigma_kev
from xrf_analyzer.fit_model import fit_spectrum

ROOT = Path(__file__).resolve().parents[1]


def test_fitted_centroid_is_subchannel_with_joint_neighbor():
    cfg = AnalysisConfig(fit_energy_gain=False)
    edges = np.arange(5, 8.0001, .04)
    energy = .5*(edges[:-1]+edges[1:])
    truth = 6.4137
    mu = 12+30000*gaussian_response(edges, truth, .075)+1600*gaussian_response(edges, 6.75, .065)
    y = np.random.default_rng(73).poisson(mu)
    p = detect_peaks(np.arange(len(y)), energy, edges, y, np.full(len(y), 12.), cfg)
    main = p.loc[(p.energy_keV-truth).abs().idxmin()]
    assert abs(main.energy_keV-truth) < .003
    assert abs(main.maximum_energy_keV-truth) > abs(main.energy_keV-truth)
    assert main.centroid_std_keV > 0 and main.centroid_std_keV < .003
    assert main.net_area_counts == pytest.approx(30000, rel=.04)
    assert len(p) == 2


def test_same_shell_completion_adds_fe_beta3_and_pb_minor_lines():
    original = load_sensitivity(ROOT/'data/xrf_relative_sensitivity.csv').lines
    full, aux, audit = complete_transitions(original, AnalysisConfig())
    by_key = {(l.symbol, l.branch): l for l in full}
    fe = by_key[('Fe', 'Kbeta1')]
    beta3 = by_key[('Fe', 'KM2')]
    assert beta3.energy_keV == pytest.approx(fe.energy_keV)
    expected = fe.sensitivity_cm2_g*xraylib.RadRate(26, xraylib.KB3_LINE)/xraylib.RadRate(26, xraylib.KB1_LINE)
    assert beta3.sensitivity_cm2_g == pytest.approx(expected, rel=1e-5)
    assert beta3.strength_source == 'same_shell_rad_rate'
    assert by_key[('Pb', 'L3M1')].energy_keV == pytest.approx(9.1845)
    assert by_key[('Pb', 'L2M1')].energy_keV == pytest.approx(11.3493)
    assert any(l.symbol == 'Pb' and l.branch == 'L1M3' for l in aux)
    assert not any(l.symbol == 'Pb' and l.branch.startswith('L1') for l in full)
    for line in original:
        assert by_key[(line.symbol, line.branch)] == line
    # Pb K branches explicitly zero in the input stay absent.
    assert not any(l.symbol == 'Pb' and l.branch.startswith('K') for l in full)


def test_inconsistent_shell_not_completed():
    lines = [LineCandidate(26, 'Fe', 'Kalpha1', 6.4039, 8),
             LineCandidate(26, 'Fe', 'Kbeta1', 7.058, 100)]
    full, _, audit = complete_transitions(lines, AnalysisConfig())
    assert full == lines
    assert not next(a for a in audit if a['Shell'] == 'K')['CompletionEnabled']
    disabled, aux, _ = complete_transitions(lines, AnalysisConfig(complete_atomic_lines=False))
    assert disabled == lines and aux == []


def test_cutoff_requires_abrupt_high_counts_not_ordinary_poisson_zeros(tmp_path):
    cal = tmp_path/'cal.csv'
    cal.write_text('channel,energy_keV\n100,2\n900,18\n')
    c = fit_linear_calibration(cal)
    y = np.r_[np.zeros(50), np.full(200, 50.)]
    cutoff = infer_hard_lower_cutoff(Spectrum(np.arange(len(y)), y), c, AnalysisConfig())
    assert cutoff['first_retained_channel'] == 50
    assert cutoff['energy_keV'] == pytest.approx(.99)
    assert infer_hard_lower_cutoff(Spectrum(np.arange(len(y)), y), c,
                                   AnalysisConfig(infer_low_energy_cutoff=False)) is None
    weak = np.r_[np.zeros(50), np.ones(200)]
    assert infer_hard_lower_cutoff(Spectrum(np.arange(len(weak)), weak), c, AnalysisConfig()) is None
    assert infer_hard_lower_cutoff(Spectrum(np.arange(200), np.zeros(200)), c, AnalysisConfig()) is None


def test_shared_gain_and_offset_recover_linear_centroid_drift():
    cfg = AnalysisConfig(fwhm_ref_kev=.20, fit_energy_gain=True)
    edges = np.arange(4, 18.0001, .02)
    energy = (edges[:-1]+edges[1:])/2
    lines = [LineCandidate(26, 'Fe', 'Kalpha1', 6.4039, 8),
             LineCandidate(82, 'Pb', 'Lalpha1', 10.5512, 5),
             LineCandidate(82, 'Pb', 'Lgamma1', 14.7648, 1)]
    mu = np.full(len(energy), 12.)
    truth_gain, truth_shift = -.004, .12
    for line in lines:
        q = 10000/line.sensitivity_cm2_g if line.symbol == 'Fe' else 4000
        mu += q*line.sensitivity_cm2_g*gaussian_response(edges, line.energy_keV*(1+truth_gain)+truth_shift,
                                                       float(sigma_kev(line.energy_keV, cfg)))
    y = np.random.default_rng(321).poisson(mu)
    fit = fit_spectrum(y, energy, edges, np.full(len(y), 12.), lines, cfg, .25, initial_delta=.08)
    assert fit.success and fit.gain_fitted
    assert fit.energy_gain == pytest.approx(truth_gain, abs=.0006)
    assert fit.delta == pytest.approx(truth_shift, abs=.006)


def test_real_spectrum_multibranch_identity_and_auditable_pairings(tmp_path):
    result = analyze(ROOT/'SDD_Spectrum/shafa_50kv_20uA.mca', ROOT/'data/calibration.csv',
                     ROOT/'data/xrf_relative_sensitivity.csv', tmp_path/'out',
                     AnalysisConfig(energy_range=(1, 40), tube_voltage_kv=50))
    c = result['identification'].candidates.set_index('Element')
    assert c.loc['Fe', 'Status'] == 'supported'
    assert c.loc['Pb', 'Status'] == 'supported'
    assert c.loc['Pb', 'NumIndependentEvidenceRegions'] >= 3
    for competitor in ('As', 'Kr', 'Ga', 'Pm', 'Pa'):
        assert c.loc[competitor, 'Status'] != 'supported'
    assert result['summary']['acquisition_cutoff']['first_retained_channel'] == 41
    assert result['summary']['response_initialization']['used']
    assert result['summary']['deviance_per_dof'] < 15
    assert len(result['identification'].fit.symbols) < 15
    # Global shape mismatch does not invalidate independently measurable primary areas.
    assert not result['summary']['global_model_valid']
    assert result['summary']['quantitative_interpretation_valid']
    numeric = result['density'].loc[result['density'].Relative_to_Reference.notna()]
    assert not numeric.empty
    assert numeric.Element_Status.eq('supported').all()
    assert np.isfinite(numeric.Relative_to_Reference).all()
    assert (numeric.Relative_to_Reference > 0).all()
    pairs = pd.read_csv(tmp_path/'out/peak_line_matches.csv')
    pb = pairs.loc[(pairs.Element == 'Pb') & pairs.Selected_Element]
    assert {'Lalpha1', 'Lbeta1', 'Lgamma1', 'L3M1'} <= set(pb.Branch)
    assignments = pd.read_csv(tmp_path/'out/peak_assignments.csv')
    beta = assignments.loc[(assignments.Fitted_Centroid_keV-12.69).abs().idxmin()]
    assert beta.Primary_Element == 'Pb' and beta.Status == 'supported'
    aux = pd.read_csv(tmp_path/'out/auxiliary_fitted_lines.csv')
    assert (aux.Net_Area >= 0).all() and not aux.Quantitative_S_Available.any()
    spectrum = pd.read_csv(tmp_path/'out/fit_spectrum.csv')
    assert np.allclose(spectrum.model_total_counts, spectrum.background_counts+spectrum.scatter_or_artifact_counts+spectrum.element_total_counts)


def test_unresolved_shoulder_is_flagged_not_invented_as_two_peaks():
    cfg = AnalysisConfig(fit_energy_gain=False)
    edges = np.arange(5, 8.0001, .04)
    energy = .5*(edges[:-1]+edges[1:])
    mu = 12+30000*gaussian_response(edges, 6.4137, .075)+1600*gaussian_response(edges, 6.68, .065)
    y = np.random.default_rng(73).poisson(mu)
    p = detect_peaks(np.arange(len(y)), energy, edges, y, np.full(len(y), 12.), cfg)
    assert len(p) == 1
    assert 'local_shape_mismatch_or_unresolved_multiplet' in p.quality_flags.iloc[0]


def test_no_usable_lines_produces_empty_fit_not_exception(synthetic):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 500.})
    df = pd.read_csv(sensitivity)
    df.loc[df.Element == '26 Fe', list(LINE_MAP)] = 0
    df.to_csv(sensitivity, index=False)
    config.elements = ('Fe',)
    result = analyze(spectrum, calibration, sensitivity, config=config)
    assert result['identification'].fit.symbols == []
    assert result['identification'].candidates.iloc[0].Status == 'unidentifiable'
    assert result['density'].empty
