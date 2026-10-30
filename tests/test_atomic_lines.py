"""Regression checks for the single xraylib line-energy source."""
import pandas as pd
import pytest
import xraylib
from xrf_analyzer import atomic_lines
from xrf_analyzer.atomic_lines import LINE_MAP, load_sensitivity
from xrf_analyzer.pipeline import analyze


def write_sensitivity(path, entries):
    records = []
    for element, strengths in entries:
        records.append({'Element': element, **{branch: strengths.get(branch, '') for branch in LINE_MAP}})
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_nine_single_line_constants_and_native_kev(tmp_path, monkeypatch):
    strengths = [('26 Fe', {'Kalpha1': 8., 'Kalpha2': 4., 'Kbeta1': 1.5}),
                 ('82 Pb', {'Lalpha1': 5., 'Lalpha2': .5, 'Lbeta1': 2.5,
                            'Lbeta2': .8, 'Lgamma1': .7, 'Malpha1': 3.})]
    calls = []
    original = xraylib.LineEnergy
    def line_energy(z, constant):
        calls.append((z, constant))
        return original(z, constant)
    monkeypatch.setattr(atomic_lines.xraylib, 'LineEnergy', line_energy)
    table = load_sensitivity(write_sensitivity(tmp_path/'s.csv', strengths))
    assert calls == [(26, xraylib.KA1_LINE), (26, xraylib.KA2_LINE), (26, xraylib.KB1_LINE),
                     (82, xraylib.LA1_LINE), (82, xraylib.LA2_LINE), (82, xraylib.LB1_LINE),
                     (82, xraylib.LB2_LINE), (82, xraylib.LG1_LINE), (82, xraylib.MA1_LINE)]
    assert len(table.lines) == 9
    assert table.lines[0].energy_keV == original(26, xraylib.KA1_LINE)
    assert table.lines[0].energy_keV == pytest.approx(6.4, abs=.01)
    for line in table.lines:
        assert line.sensitivity_cm2_g == dict(strengths)[f'{line.Z} {line.symbol}'][line.branch]


def test_zero_and_missing_s_do_not_query_database(tmp_path, monkeypatch):
    def unexpected_query(*args):
        pytest.fail('No energy should be requested for zero or missing S')
    monkeypatch.setattr(atomic_lines.xraylib, 'LineEnergy', unexpected_query)
    table = load_sensitivity(write_sensitivity(tmp_path/'s.csv', [('26 Fe', {'Kalpha1': 0.})]))
    assert table.lines == []
    assert table.states[0]['State'] == 'zero'
    assert table.states[1]['State'] == 'missing'


def test_unavailable_database_line_is_removed(tmp_path, monkeypatch):
    original = xraylib.LineEnergy
    def line_energy(z, constant):
        if constant == xraylib.KB1_LINE:
            raise ValueError('Transition unavailable')
        return original(z, constant)
    monkeypatch.setattr(atomic_lines.xraylib, 'LineEnergy', line_energy)
    table = load_sensitivity(write_sensitivity(tmp_path/'s.csv',
                            [('26 Fe', {'Kalpha1': 8., 'Kbeta1': 1.5})]))
    assert [line.branch for line in table.lines] == ['Kalpha1']
    assert next(state for state in table.states if state['Branch'] == 'Kbeta1')['State'] == 'database_unavailable'
    assert any('Transition unavailable' in warning for warning in table.warnings)


def test_analysis_ignores_reference_csv_and_records_xraylib(synthetic, tmp_path):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 500.})
    config.elements = ('Fe',)
    # A reference file cannot silently become an input or override the atomic database.
    sensitivity.with_name('characteristic_lines_energy.csv').write_text('invalid,unused,reference\n')
    result = analyze(spectrum, calibration, sensitivity, tmp_path/'out', config)
    assert result['summary']['line_energy'] == {
        'source': 'xraylib', 'api': 'xraylib.LineEnergy', 'units': 'keV'}
    assert result['summary']['dependency_versions']['xraylib']
    assert set(result['summary']['input_files']) == {'sample_spectrum', 'calibration', 'sensitivity'}
    lines = pd.read_csv(tmp_path/'out/fitted_lines.csv')
    fe = lines.loc[lines.Branch == 'Kalpha1'].iloc[0]
    assert fe.Reference_Energy_keV == xraylib.LineEnergy(26, xraylib.KA1_LINE)
