"""ON/OFF likelihood inputs must not be the same recorded measurement."""
import numpy as np
import pandas as pd
import pytest

from xrf_analyzer.background_input import load_background
from xrf_analyzer.calibration import fit_linear_calibration
from xrf_analyzer.io_spectrum import load_spectrum
from xrf_analyzer.pipeline import analyze


@pytest.mark.parametrize('copied', [False, True])
def test_sample_or_byte_copy_cannot_be_its_own_independent_background(synthetic, tmp_path, copied):
    spectrum, calibration, sensitivity, config, _, _ = synthetic({'Fe': 500.})
    config.elements = ('Fe',)
    config.fit_response = False
    background = spectrum
    if copied:
        background = tmp_path/'copied_background.csv'
        background.write_bytes(spectrum.read_bytes())
    with pytest.raises(ValueError, match='same sample.*independent background'):
        analyze(spectrum, calibration, sensitivity, config=config, background_spectrum=background)


def test_equal_counts_with_distinct_measurement_headers_are_not_assumed_to_be_a_copy(synthetic, tmp_path):
    spectrum, calibration, _, config, _, _ = synthetic({'Fe': 500.})
    background = tmp_path/'another_measurement.csv'
    frame = pd.read_csv(spectrum)
    frame['live_time_s'] = 30.
    frame.to_csv(background, index=False)
    sample = load_spectrum(spectrum)
    counts, info = load_background(background, sample.channel, fit_linear_calibration(calibration),
                                   sample, config, sample_path=spectrum)
    assert np.array_equal(counts, sample.counts)
    assert info['enabled']
