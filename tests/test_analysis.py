import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from xrf_analyzer.pipeline import analyze
from xrf_analyzer.reporting import fitted_line_tables
from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.fit_model import fit_spectrum
from xrf_analyzer.element_identification import identify_elements
from xrf_analyzer.peak_detection import detect_peaks
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.response import gaussian_response, sigma_kev


def run(data, output=None):
    s, c, t, cfg, _, _ = data
    return analyze(s, c, t, output, cfg)


def status(result, symbol):
    return result['identification'].candidates.set_index('Element').loc[symbol, 'Status']


def test_fe_supported(synthetic):
    result = run(synthetic({'Fe': 1200.}))
    assert status(result, 'Fe') == 'supported'
    assert result['identification'].fit.q[result['identification'].fit.symbols.index('Fe')] == pytest.approx(1200, rel=.04)
    assert status(result, 'Co') != 'supported'


def test_mixture_ratios_and_all_exports(synthetic, tmp_path):
    result = run(synthetic(), tmp_path/'out')
    assert result['summary']['fit_status'] == 'success'
    assert all(status(result, el) == 'supported' for el in ['Fe', 'Cu', 'Pb'])
    d = result['density'].set_index('Element')
    assert d.loc['Fe', 'q_Fit']/d.loc['Pb', 'q_Fit'] == pytest.approx(.5, rel=.07)
    assert d.loc['Cu', 'q_Fit']/d.loc['Fe', 'q_Fit'] == pytest.approx(.5, rel=.07)
    assert d.Fraction_among_Reported_Elements.sum() == pytest.approx(1)
    assert d.Apparent_Only.all()
    out = tmp_path/'out'
    required = ['calibration_result.json', 'detected_peaks.csv', 'element_candidates.csv', 'fitted_lines.csv',
                'relative_areal_density.csv', 'fit_spectrum.csv', 'fit_summary.json', 'spectrum_fit.png',
                'residuals.png', 'calibration.png', 'fitted_groups.csv']
    assert all((out/p).exists() for p in required)
    summary = json.loads((out/'fit_summary.json').read_text())
    assert summary['input_files']['sample_spectrum']['sha256'] and summary['parameters']['seed'] == 42
    assert len(pd.read_csv(out/'fit_spectrum.csv')) == len(result['counts'])
    lines = pd.read_csv(out/'fitted_lines.csv')
    assert (lines.Net_Area >= 0).all() and (lines.Sensitivity_cm2_g > 0).all()
    assert np.all(lines.groupby('Element').apply(lambda x: np.ptp(x.Net_Area/x.Sensitivity_cm2_g)) < 1e-8)
    groups = pd.read_csv(out/'fitted_groups.csv').set_index('Element')
    fealpha = lines[(lines.Element == 'Fe') & lines.Branch.isin(['Kalpha1', 'Kalpha2'])]
    assert groups.loc['Fe', 'Net_Area'] == pytest.approx(fealpha.Net_Area.sum())
    assert groups.loc['Fe', 'Area_Std_Counts'] == pytest.approx(fealpha.Area_Std_Counts.sum())


def test_exposure_error_scaling(synthetic):
    first = run(synthetic({'Fe': 1000.}, exposure=1))
    second = run(synthetic({'Fe': 1000.}, exposure=4))
    def rel(r):
        c = r['identification'].candidates.set_index('Element').loc['Fe']
        return c.q_Std/c.q_Fit
    assert rel(second)/rel(first) == pytest.approx(.5, rel=.12)


def test_zero_low_counts_and_unknown_peaks(synthetic):
    data = synthetic({}, background=0)
    result = run(data)
    assert not (result['identification'].candidates.Status == 'supported').any()
    assert result['density'].empty or result['density'].Fraction_among_Reported_Elements.isna().all()
    low = run(synthetic({'Fe': .2}, background=.1))
    assert low['summary']['fit_status'] == 'success'


def test_zero_sensitivity_never_identified(synthetic):
    data = synthetic({'Fe': 1000})
    df = pd.read_csv(data[2]).fillna('')
    df.loc[df.Element == '26 Fe', ['Kalpha1', 'Kalpha2', 'Kbeta1']] = 0
    df.to_csv(data[2], index=False)
    result = run(data)
    assert status(result, 'Fe') == 'unidentifiable'
    assert len(result['unassigned_peaks']) >= 1
    assert all(l.symbol != 'Fe' for l in result['identification'].fit_lines)


def test_pb_as_with_additional_lines(synthetic):
    result = run(synthetic({'Pb': 1500.}))
    assert status(result, 'Pb') == 'supported'
    assert status(result, 'As') != 'supported'


def test_weak_fe_beta_is_tentative(synthetic):
    result = run(synthetic({'Fe': 80}, branches={'Fe': {'Kalpha1': 8., 'Kalpha2': 4., 'Kbeta1': .001}}))
    assert status(result, 'Fe') == 'tentative'


def test_collinear_elements_no_precise_ratios():
    config = AnalysisConfig(fit_response=False, elements=('Fe', 'Co'), element_selection='manual')
    lines = [LineCandidate(26, 'Fe', 'Kalpha1', 6.4, 8.), LineCandidate(27, 'Co', 'Kalpha1', 6.4, 4.)]
    edges = np.arange(5, 8.001, .02)
    energy = (edges[:-1]+edges[1:])/2
    mu = 8+5000*gaussian_response(edges, 6.4, float(sigma_kev(6.4, config)))
    y = np.random.default_rng(42).poisson(mu)
    peaks = detect_peaks(np.arange(len(y)), energy, edges, y, np.full(len(y), 8), config)
    ident = identify_elements(y, energy, edges, np.full(len(y), 8), lines, peaks, config, .02)
    assert set(ident.candidates.Status) == {'unidentifiable'}
    assert not np.isfinite(ident.qcov).any()


def test_absorption_mismatch_warned(synthetic):
    data = synthetic({'Fe': 1500.})
    lines = data[-1]
    alternate = [LineCandidate(l.Z, l.symbol, l.branch, l.energy_keV,
                             l.sensitivity_cm2_g*(.1 if l.symbol == 'Fe' and l.branch == 'Kbeta1' else 1)) for l in lines]
    mismatched = synthetic({'Fe': 1500.}, custom_lines=alternate)
    mismatched[3].quantification_method = 'global-template'
    result = run(mismatched)
    assert any('branch_ratio_mismatch' in w for w in result['summary']['warnings'])
    assert status(result, 'Fe') == 'supported'  # Two measured energies can identify Fe even when ratios fail.
    assert result['identification'].candidates.set_index('Element').loc['Fe', 'Quantification_Status'] != 'supported'
    assert result['density'].Relative_to_Reference.isna().all()


def test_requested_unreliable_reference_is_not_replaced(synthetic):
    data = synthetic({'Fe': 1000})
    data[3].reference_element = 'Pb'
    result = run(data)
    assert result['summary']['reference']['Element'] is None
    assert result['density'].Relative_to_Reference.isna().all()


def test_fit_failure_does_not_report_quantities(synthetic):
    data = synthetic()
    data[3].max_iterations = 1
    result = run(data)
    assert result['summary']['fit_status'] == 'failed'
    assert result['density'].q_Fit.isna().all()
    tables = fitted_line_tables(result['identification'], result['counts'], result['edges'], result['config'])
    assert tables[0].empty


def test_pb_with_overlapped_beta_region(synthetic):
    data = synthetic({'Pb': 1500., 'Kr': 4500.}, branches={'Kr': {'Kalpha1': 8., 'Kalpha2': 4., 'Kbeta1': 1.2}})
    result = run(data)
    assert status(result, 'Pb') == 'supported'
    lb = next(l.energy_keV for l in data[-1] if l.symbol == 'Pb' and l.branch == 'Lbeta1')
    near = result['peaks'].loc[abs(result['peaks'].energy_keV-lb)<.2]
    assert len(near) <= 1  # Pb beta does not need its own local maximum, distinct from Kr.


def test_pb_as_restricted_region_not_quantified(synthetic):
    data = synthetic({'Pb': 600}, energy_range=(9.9, 10.9))
    result = run(data)
    assert status(result, 'Pb') in ('tentative', 'ambiguous', 'unidentifiable')
    assert status(result, 'As') != 'supported'
    assert result['density'].Fraction_among_Reported_Elements.isna().all()


def test_roi_truncated_area_not_renormalized(synthetic):
    data = synthetic({'Fe': 1000}, energy_range=(6.4, 7.5))
    data[3].fit_response = False
    result = run(data)
    lines, _, _ = fitted_line_tables(result['identification'], result['counts'], result['edges'], result['config'])
    fe = lines[lines.Element == 'Fe']
    assert fe.ROI_Truncated.any()
    assert (fe.Net_Area_in_ROI <= fe.Net_Area).all()


def test_excitation_mismatch_withholds_ratios(synthetic):
    data = synthetic({'Fe': 1000})
    sidecar = data[2].with_suffix('.metadata.json')
    sidecar.write_text(json.dumps({'tube_voltage_kV': 40}))
    data[3].tube_voltage_kv = 50
    result = run(data)
    assert result['density'].Fraction_among_Reported_Elements.isna().all()
    assert any('mismatch' in w for w in result['summary']['warnings'])


def test_bootstrap_reproducible_conditional_diagnostics(synthetic):
    data = synthetic({'Fe': 200.})
    data[3].elements = ('Fe',)
    data[3].bootstrap = 20
    a = run(data)
    b = run(data)
    assert a['summary']['bootstrap']['successful'] >= 20
    assert a['summary']['bootstrap']['q_samples'] == b['summary']['bootstrap']['q_samples']
    assert a['identification'].candidates.Upper95_q.notna().any()


def test_broad_unmodelled_structure_flagged(synthetic):
    data = synthetic({}, background=12)
    df = pd.read_csv(data[0])
    mu = 12+2000*np.exp(-.5*((df.channel*.02-11)/.4)**2)
    df.counts = np.random.default_rng(12).poisson(mu)
    df.to_csv(data[0], index=False)
    result = run(data)
    assert any('Structured residual' in w for w in result['summary']['warnings'])


def test_severe_global_mismatch_never_reports_normalized_content(synthetic):
    data = synthetic({'Fe': 1000})
    data[3].quantification_method = 'global-template'
    data[3].max_deviance_per_dof = .0001
    result = run(data)
    assert result['summary']['analysis_status'] == 'model_mismatch'
    assert not result['summary']['quantitative_interpretation_valid']
    assert status(result, 'Fe') == 'supported'  # Global quantitative quality does not erase identification.
    assert not (result['identification'].candidates.Quantification_Status == 'supported').any()
    assert result['density'].Fraction_among_Reported_Elements.isna().all()
    assert result['density'].Relative_to_Reference.isna().all()
