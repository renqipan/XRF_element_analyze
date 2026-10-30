from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from xrf_analyzer.calibration import fit_linear_calibration
from xrf_analyzer.io_spectrum import load_spectrum
from xrf_analyzer.atomic_lines import load_sensitivity, LINE_MAP


def test_two_point_and_origin(tmp_path):
    p = tmp_path/'cal.csv'
    p.write_text('Channel,Energy\n100,2\n900,18\n')
    c = fit_linear_calibration(p)
    assert abs(c.a_keV) < 1e-12
    assert c.b_keV_per_channel == pytest.approx(.02)
    assert c.covariance is None and c.two_point
    p.write_text('channel,energy(keV)\n0,0\n299,22.08\n')
    assert fit_linear_calibration(p).b_keV_per_channel == pytest.approx(22.08/299)


@pytest.mark.parametrize('text', [
    'channel,energy_keV\n1,2\n1,3\n', 'channel,energy_keV\n1,3\n2,2\n',
    'channel,energy_keV\n1,3\n2,3\n', 'channel,energy_keV\n1,NaN\n2,3\n',
    'channel,bad\n1,2\n2,3\n', 'channel,energy_keV\n1,0\n2,3\n'])
def test_invalid_calibration(tmp_path, text):
    p = tmp_path/'cal.csv'
    p.write_text(text)
    with pytest.raises(ValueError):
        fit_linear_calibration(p)


def test_multipt_weighting_and_quality(tmp_path):
    p = tmp_path/'c.csv'
    p.write_text('channel,energy_keV,energy_sigma_keV\n100,2,.01\n500,10.01,.02\n900,18,.01\n')
    c = fit_linear_calibration(p)
    assert c.weighted and c.covariance is not None and c.rms_keV < .02
    p.write_text('channel,energy_keV,energy_sigma_keV\n100,2,.01\n500,11,\n900,18,.01\n')
    c = fit_linear_calibration(p)
    assert not c.weighted and c.calibration_quality == 'warning'
    with pytest.raises(ValueError):
        fit_linear_calibration(p, strict=True)


@pytest.mark.parametrize('origin', [0, 1])
def test_counts_and_mca_formats(tmp_path, origin):
    p = tmp_path/'a.csv'
    p.write_text('\ufeffcounts\n0\n2.0\n9\n1\n')
    a = load_spectrum(p, channel_start=origin)
    mca = '<<PMCA SPECTRUM>>\nLIVE_TIME - 30\nREAL_TIME - 31\n<<CALIBRATION>>\nLABEL - Channel\n0 0\n299 22.08\n<<DATA>>\n0\n2\n9\n1\n<<END>>\n<<DP5 CONFIGURATION>>\nMCAC=4;\n<<DP5 CONFIGURATION END>>\n'
    q = tmp_path/'a.MCA'
    q.write_text(mca)
    b = load_spectrum(q, channel_start=origin)
    assert np.array_equal(a.channel, b.channel) and np.array_equal(a.counts, b.counts)
    assert a.channel[0] == origin and b.live_time_s == 30
    q = tmp_path/'a.mac'
    q.write_text(mca)
    assert load_spectrum(q).counts.sum() == 12
    q.write_text('unknown format')
    with pytest.raises(ValueError, match='Unknown format'):
        load_spectrum(q)
    q.write_text(mca.replace('MCAC=4', 'MCAC=8'))
    with pytest.raises(ValueError, match='declared channel'):
        load_spectrum(q)


@pytest.mark.parametrize('data', ['counts\n-1\n', 'counts\n2.5\n', 'counts\nNaN\n',
                                 'channel,counts\n1,2\n3,2\n', 'channel,counts\n2,2\n1,2\n'])
def test_invalid_sample(tmp_path, data):
    p = tmp_path/'a.csv'
    p.write_text(data)
    with pytest.raises(ValueError):
        load_spectrum(p)


def test_missing_channels_explicit(tmp_path):
    p = tmp_path/'a.csv'
    p.write_text('channel,counts\n1,3\n3,7\n')
    s = load_spectrum(p, fill_missing_channels=True)
    assert list(s.counts) == [3, 0, 7] and s.metadata['filled_channels'] == [2]


def sensitivity_file(tmp_path, element='26 Fe', a='8', b='0'):
    p = tmp_path/'s.csv'
    p.write_text(','.join(['Element', *LINE_MAP])+'\n'+','.join([element, a, b]+['']*7)+'\n')
    return p


def test_sensitivity_zero_missing_separate(tmp_path):
    table = load_sensitivity(sensitivity_file(tmp_path))
    assert [s['State'] for s in table.states[:3]] == ['positive', 'zero', 'missing']
    assert len(table.lines) == 1 and table.lines[0].sensitivity_cm2_g == 8
    assert table.metadata['sensitivity_provenance'] == 'unverified'


@pytest.mark.parametrize('el,a', [('26 Cu', '1'), ('26 Fe', '-1'), ('26 Fe', 'inf')])
def test_invalid_sensitivity(tmp_path, el, a):
    with pytest.raises(ValueError):
        load_sensitivity(sensitivity_file(tmp_path, el, a))


def test_duplicate_sensitivity(tmp_path):
    p = sensitivity_file(tmp_path)
    p.write_text(p.read_text() + p.read_text().splitlines()[1]+'\n')
    with pytest.raises(ValueError):
        load_sensitivity(p)


def test_real_mca_structure_and_header_encoding():
    root = Path(__file__).resolve().parents[1]
    s = load_spectrum(root/'SDD_Spectrum/shafa_50kv_20uA.mca')
    assert len(s.counts) == 1024 and s.live_time_s == 30
    assert s.metadata['format'] == 'Amptek_PMCA_ASCII'
    assert len(s.metadata['embedded_calibration_points']) == 2
