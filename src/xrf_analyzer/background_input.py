"""Exact channel alignment: no interpolation of independent Poisson counts."""
from pathlib import Path
import numpy as np
from .io_spectrum import load_spectrum


def load_background(path, channels, calibration, sample, config):
    background=load_spectrum(path,config.channel_start,False)
    energy=calibration.energy(background.channel)
    tolerance=max(1e-6,.1*calibration.b_keV_per_channel)
    supplied=background.metadata.get('supplied_energy_keV')
    if supplied is not None and np.max(abs(np.asarray(supplied)-energy))>tolerance:
        raise ValueError('Background CSV energy conflicts with sample calibration; rebinning is not implicit')
    embedded=background.metadata.get('embedded_calibration_points',[])
    if any(abs(calibration.energy(c)-e)>tolerance for c,e in embedded):
        raise ValueError('Background embedded calibration conflicts with sample calibration')
    indices=np.searchsorted(background.channel,channels)
    if np.any(indices>=len(background.channel)) or not np.array_equal(background.channel[indices],channels):
        raise ValueError('Background spectrum must cover every analyzed sample channel exactly')
    counts=background.counts[indices]
    if counts.sum()<=0:
        raise ValueError('Background contains no counts in the analyzed energy range; scale is not identifiable')
    scale=sample.live_time_s/background.live_time_s if sample.live_time_s and background.live_time_s else 1.
    return counts,{'enabled':True,'path':str(Path(path).resolve()),'scale_initial':scale,
        'sample_live_time_s':sample.live_time_s,'reference_live_time_s':background.live_time_s,
        'alignment':'exact_channel_match_no_rebinning','conditions_verified':False,
        'model':'sample~Poisson(signal+continuum+alpha*b); reference~Poisson(b); b profiled',
        'limitations':'One global scale cannot correct background shape or geometry changes',
        'warnings':background.metadata.get('warnings',[])}
