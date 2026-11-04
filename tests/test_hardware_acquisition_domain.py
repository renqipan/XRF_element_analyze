"""Amptek TLLD is a recorded channel boundary, not a count-shape inference."""
from pathlib import Path

import numpy as np
import pytest

from xrf_analyzer.background_input import load_background
from xrf_analyzer.calibration import fit_linear_calibration
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.io_spectrum import load_spectrum
from xrf_analyzer.measurement import infer_hard_lower_cutoff
from xrf_analyzer.pipeline import analyze


ROOT = Path(__file__).resolve().parents[1]


def _mca(tmp_path, threshold='40', mode='NORM', offset='OFF', block=True, counts=None, firmware='6.10'):
    counts = np.r_[np.zeros(41, dtype=int), np.ones(215, dtype=int)] if counts is None else counts
    config = 'MCAS=%s;\nMCAC=256;\nSOFF=%s;\nTLLD=%s;\n' % (mode, offset, threshold)
    if block:
        config = '<<DP5 CONFIGURATION>>\n'+config+'<<DP5 CONFIGURATION END>>\n'
    path = tmp_path/'sample.mca'
    status = ('<<DPP STATUS>>\nFirmware: %s  Build: 4\n<<DPP STATUS END>>\n' % firmware) if firmware else ''
    path.write_text('<<PMCA SPECTRUM>>\n<<DATA>>\n'+'\n'.join(map(str, counts))+'\n<<END>>\n'+config+status)
    calibration = tmp_path/'calibration.csv'
    calibration.write_text('channel,energy_keV\n0,0\n255,25.5\n')
    return path, fit_linear_calibration(calibration)


@pytest.mark.parametrize('origin', [0, 1])
@pytest.mark.parametrize('threshold', ['40', '40.0'])
def test_recorded_threshold_excludes_device_indices_even_with_low_counts_and_inference_disabled(tmp_path, origin, threshold):
    path, calibration = _mca(tmp_path, threshold=threshold)
    spectrum = load_spectrum(path, channel_start=origin)
    original = spectrum.counts.copy()
    cutoff = infer_hard_lower_cutoff(spectrum, calibration, AnalysisConfig(infer_low_energy_cutoff=False))
    assert cutoff is not None
    assert not cutoff['inferred']
    assert cutoff['first_retained_channel'] == 41+origin
    assert cutoff['energy_keV'] == pytest.approx((40.5+origin)*.1)
    assert spectrum.metadata['hardware_lower_cutoff']['threshold_device_channel'] == 40
    assert np.array_equal(spectrum.counts, original)


@pytest.mark.parametrize('threshold', ['40.5', '-1', 'nan', '256'])
def test_invalid_or_fractional_explicit_threshold_is_not_rounded(tmp_path, threshold):
    path, _ = _mca(tmp_path, threshold=threshold)
    with pytest.raises(ValueError, match='TLLD'):
        load_spectrum(path)


@pytest.mark.parametrize('settings', [dict(threshold='OFF'), dict(mode='FAST'),
                                      dict(offset='3'), dict(block=False),
                                      dict(firmware='6.05'), dict(firmware=None)])
def test_unverified_modes_or_missing_configuration_do_not_create_a_hardware_domain(tmp_path, settings):
    path, calibration = _mca(tmp_path, **settings)
    spectrum = load_spectrum(path)
    assert infer_hard_lower_cutoff(spectrum, calibration, AnalysisConfig(infer_low_energy_cutoff=False)) is None


def test_nonzero_counts_in_hardware_rejected_channels_are_an_explicit_conflict(tmp_path):
    counts = np.ones(256, dtype=int)
    path, _ = _mca(tmp_path, counts=counts)
    with pytest.raises(ValueError, match='TLLD.*nonzero'):
        load_spectrum(path)


@pytest.mark.parametrize('threshold,first', [('0', 1), ('255', 256)])
def test_boundary_thresholds_keep_correct_domain_without_indexing_past_data(tmp_path, threshold, first):
    counts = np.r_[np.zeros(first, dtype=int), np.ones(256-first, dtype=int)]
    path, calibration = _mca(tmp_path, threshold=threshold, counts=counts)
    cutoff = infer_hard_lower_cutoff(load_spectrum(path), calibration, AnalysisConfig())
    assert cutoff['first_retained_channel'] == first
    assert cutoff['first_counts'] == (1 if first < 256 else None)


def test_missing_offset_does_not_claim_verified_channel_mapping(tmp_path):
    path, calibration = _mca(tmp_path)
    path.write_text(path.read_text().replace('SOFF=OFF;\n', ''))
    spectrum = load_spectrum(path)
    assert infer_hard_lower_cutoff(spectrum, calibration, AnalysisConfig(infer_low_energy_cutoff=False)) is None
    assert any('offset mapping' in message for message in spectrum.metadata['warnings'])


def test_background_cannot_supply_unmeasured_threshold_channels(tmp_path):
    path, calibration = _mca(tmp_path)
    sample = load_spectrum(path)
    with pytest.raises(ValueError, match='Background.*acquisition'):
        load_background(path, np.arange(30, 60), calibration, sample, AnalysisConfig())
    counts, _ = load_background(path, np.arange(41, 60), calibration, sample, AnalysisConfig())
    assert np.array_equal(counts, np.ones(19))


def test_pure_pb_real_spectrum_uses_recorded_cutoff_even_when_inference_is_disabled():
    result = analyze(ROOT/'data/backgroud_spectrum.mca', ROOT/'data/calibration.csv',
                     ROOT/'data/xrf_relative_sensitivity.csv',
                     config=AnalysisConfig(energy_range=(1., 40.), infer_low_energy_cutoff=False))
    cutoff = result['summary']['acquisition_cutoff']
    assert not cutoff['inferred']
    assert cutoff['first_retained_channel'] == 41
    assert result['channel'][0] == 41
    assert result['edges'][0] == pytest.approx(40.5*22.08/299)
    assert not any(l.symbol == 'Pb' and l.parent_shell.startswith('M')
                   for l in result['identification'].fit_lines)
    assert any('Recorded acquisition' in warning for warning in result['summary']['warnings'])
