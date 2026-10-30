import numpy as np
from scipy.special import ndtr

FWHM_FACTOR = 2.3548200450309493


def sigma_kev(energy, config):
    electronic2 = (config.fwhm_ref_kev / FWHM_FACTOR)**2 - config.fano * config.pair_energy_kev * config.fwhm_ref_energy_kev
    if electronic2 < 0:
        raise ValueError('Reference resolution is below Fano limit')
    return np.sqrt(electronic2 + config.fano * config.pair_energy_kev * np.maximum(energy, 0))


def gaussian_response(edges, center, sigma):
    """Probability in each physical channel. Never renormalize truncated responses."""
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError('Gaussian sigma must be positive')
    return np.maximum(np.diff(ndtr((np.asarray(edges) - center) / sigma)), 0)


def response_with_derivatives(edges, centers, sigmas, delta=0., scale=1.):
    sigmas = np.asarray(sigmas) * scale
    z = (edges[:, None] - np.asarray(centers)[None, :] - delta) / sigmas[None, :]
    phi = np.exp(-0.5 * z*z) / np.sqrt(2*np.pi)
    p = np.maximum(np.diff(ndtr(z), axis=0), 0)
    d_delta = (phi[:-1] - phi[1:]) / sigmas[None, :]
    d_scale = (z[:-1]*phi[:-1] - z[1:]*phi[1:]) / scale
    return p, d_delta, d_scale


def line_model_center(line, fit):
    key=(line.symbol,line.branch)
    base=fit.line_centers.get(key,line.energy_keV)
    if fit.center_axis=='observed' and key in fit.line_centers:
        return base
    return base*(1+fit.energy_gain)+fit.delta
