"""Profiled ON/OFF Poisson likelihood for a measured background spectrum.

Sample counts Y~Pois(s+alpha*b), reference counts B~Pois(b). The unknown
reference means b are profiled independently, rather than subtracting noisy
counts or treating the measured reference as an exact template. Alpha is the
coefficient of the *raw reference counts*, including any exposure ratio.
"""
import numpy as np
from scipy.special import xlogy


def _deviance(observed, expected):
    return float(2*np.sum(xlogy(observed, np.maximum(observed, 1e-300)/
                                np.maximum(expected, 1e-300))-(observed-expected)))


def profile_background(counts, signal, scale, background_counts):
    """Profile reference means and return envelope gradients/information.

    ``signal`` is the expected smooth residual background plus sample/instrument
    components, excluding the measured-background term. Information weights are
    the Schur complement after the per-channel reference means are eliminated:
    1/(sample_mean + scale**2 * reference_mean). The corresponding unprofiled
    parameter Jacobian uses ds/dtheta and reference_mean for d/dscale.
    """
    y, s, z = (np.asarray(value, dtype=float) for value in
               (counts, signal, background_counts))
    if y.shape != s.shape or z.shape != y.shape:
        raise ValueError('Measured background and sample must have identical aligned channel shapes')
    if (not np.isfinite(y).all() or not np.isfinite(z).all() or not np.isfinite(s).all()
            or (y < 0).any() or (z < 0).any() or (s < 0).any()
            or not np.isfinite(scale) or scale < 0):
        raise ValueError('ON/OFF Poisson counts, signal and scale must be finite and nonnegative')
    alpha = float(scale)
    if alpha == 0:
        latent = z.copy()
    else:
        # Positive root of alpha*(alpha+1)b^2 + C*b - B*s = 0.
        # Rationalize the root for C>=0 to avoid catastrophic cancellation.
        coefficient = alpha*(alpha+1.)
        middle = (alpha+1.)*s-alpha*(y+z)
        discriminant = np.hypot(middle, 2*np.sqrt(coefficient)*np.sqrt(z)*np.sqrt(s))
        latent = np.zeros_like(s)
        positive = middle >= 0
        denominator = discriminant+middle
        np.divide(2*z*s, denominator, out=latent, where=positive & (denominator > 0))
        latent[~positive] = (discriminant[~positive]-middle[~positive])/(2*coefficient)
    model = np.maximum(s+alpha*latent, 1e-12)
    residual = 1-y/model
    sample_deviance = _deviance(y, model)
    background_deviance = _deviance(z, latent)
    return {'model': model, 'latent_background': latent,
            'nll': float(np.sum(model-xlogy(y, model)+latent-xlogy(z, np.maximum(latent, 1e-300)))),
            'deviance': sample_deviance+background_deviance,
            'sample_deviance': sample_deviance, 'background_deviance': background_deviance,
            'signal_gradient': residual, 'scale_gradient': float(latent@residual),
            'information_weight': 1/np.maximum(model+alpha**2*latent, 1e-12)}
