import numpy as np
from scipy.ndimage import gaussian_filter1d


def snip_background(counts, iterations=30):
    """SNIP clipping in the log-log-square-root Poisson stabilizing transform."""
    y = gaussian_filter1d(np.asarray(counts, float), 0.7, mode='nearest')
    v = np.log1p(np.log1p(np.sqrt(y + 1)))
    for k in range(min(iterations, (len(y)-1)//2), 0, -1):
        v[k:-k] = np.minimum(v[k:-k], 0.5 * (v[:-2*k] + v[2*k:]))
    return np.maximum(np.expm1(np.expm1(v))**2 - 1, 0)
