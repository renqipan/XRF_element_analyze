from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import xlogy
from numpy.polynomial.legendre import legvander
import xraylib
from .response import response_with_derivatives, sigma_kev


@dataclass
class FitResult:
    success: bool
    message: str
    objective: float
    nll: float
    deviance: float
    symbols: list
    q: np.ndarray
    t: np.ndarray
    sensitivity_sums: np.ndarray
    background: np.ndarray
    scatter: np.ndarray
    components: np.ndarray
    model: np.ndarray
    matrix: np.ndarray
    jacobian: np.ndarray
    parameters: np.ndarray
    bounds: list
    amplitude_scale: float
    nuisance_names: list
    delta: float
    resolution_scale: float
    attempts: list
    n_parameters: int
    response_at_boundary: bool
    auxiliary_areas: dict
    auxiliary_lines: list
    energy_gain: float
    gain_fitted: bool
    line_centers: dict
    center_axis: str


def poisson_deviance(y, mu):
    y, mu = np.asarray(y, float), np.maximum(mu, 1e-12)
    return float(2*np.sum(xlogy(y, np.maximum(y, 1e-300)/mu) - (y-mu)))


def instrument_lines(config, edges):
    """Conservative target Rayleigh/optional Compton nuisance peaks; areas are free."""
    specs = []
    if config.tube_target:
        z = xraylib.SymbolToAtomicNumber(config.tube_target)
        for name, const in [('Kalpha', xraylib.KA1_LINE), ('Kbeta', xraylib.KB1_LINE)]:
            try:
                e = xraylib.LineEnergy(z, const)
            except ValueError:
                continue
            if edges[0] <= e <= edges[-1]:
                specs.append((config.tube_target + '_Rayleigh_' + name, e, float(sigma_kev(e, config))))
            if config.scatter_angle_deg is not None:
                ec = e / (1 + e/510.99895*(1-np.cos(np.deg2rad(config.scatter_angle_deg))))
                if edges[0] <= ec <= edges[-1]:
                    specs.append((config.tube_target + '_Compton_' + name, ec, 2*float(sigma_kev(ec, config))))
    return specs


def fit_spectrum(counts, energy, edges, background0, lines, config, delta_bound=0.02,
                 initial=None, fixed_areas=None, auxiliary_lines=(), initial_delta=0., line_centers=None, center_axis="reference"):
    line_centers = dict(line_centers or {})
    y = np.asarray(counts, float)
    symbols = sorted({line.symbol for line in lines}, key=xraylib.SymbolToAtomicNumber)
    sums = np.array([sum(l.sensitivity_cm2_g for l in lines if l.symbol == s) for s in symbols])
    extras = instrument_lines(config, edges)
    auxiliary_lines = [l for l in auxiliary_lines if l.symbol in symbols]
    aux_keys = sorted({(l.symbol, l.parent_shell) for l in auxiliary_lines})
    aux_names = ['%s_%s_unscaled' % key for key in aux_keys]
    aux_sums = {key: sum(l.relative_rate for l in auxiliary_lines if (l.symbol, l.parent_shell) == key) for key in aux_keys}
    n_aux = len(aux_keys)
    n, m = len(symbols), len(extras)+n_aux
    centers = np.array([line_centers.get((l.symbol,l.branch),l.energy_keV) for l in lines] + [s[1] for s in extras]
                       +[line_centers.get((l.symbol,l.branch),l.energy_keV) for l in auxiliary_lines])
    fixed_centers = np.array([center_axis=='observed' and (l.symbol,l.branch) in line_centers for l in lines]
                            +[False]*len(extras)+[center_axis=='observed' and (l.symbol,l.branch) in line_centers for l in auxiliary_lines],dtype=bool)
    sigmas = np.array([float(sigma_kev(l.energy_keV, config)) for l in lines] + [s[2] for s in extras]
                      + [float(sigma_kev(l.energy_keV, config)) for l in auxiliary_lines])
    weights = np.zeros((len(centers), n+m))
    for k, line in enumerate(lines):
        idx = symbols.index(line.symbol)
        weights[k, idx] = line.sensitivity_cm2_g / sums[idx]
    for j in range(len(extras)):
        weights[len(lines)+j, n+j] = 1.
    for k, line in enumerate(auxiliary_lines):
        key = (line.symbol, line.parent_shell)
        weights[len(lines)+len(extras)+k, n+len(extras)+aux_keys.index(key)] = line.relative_rate/aux_sums[key]
    h = legvander(2*(energy-energy[0])/max(energy[-1]-energy[0], 1e-12)-1, config.background_degree)
    b0 = np.maximum(background0, 0.1)
    amp_scale = max(float(y.sum()) / 10., 10.)
    n_amp = n+m
    beta_slice = slice(n_amp, n_amp+h.shape[1])
    gain_fitted = bool(config.fit_response and config.fit_energy_gain and np.sum(~fixed_centers) >= 2 and np.ptp(centers[~fixed_centers]) > 1.)
    npar = n_amp+h.shape[1]+(2+int(gain_fitted) if config.fit_response else 0)

    def matrices(delta, scale, gain=0.):
        if not len(centers):
            zero = np.zeros((len(y), 0))
            return zero, zero, zero, zero
        projected=centers*(1+gain)+delta
        projected[fixed_centers]=centers[fixed_centers]
        p, d, w = response_with_derivatives(edges, projected, sigmas, 0., scale)
        d[:,fixed_centers]=0.
        return p @ weights, d @ weights, w @ weights, (d*centers[None, :]) @ weights

    def evaluate(x, want_jac=False):
        response_start = n_amp+h.shape[1]
        delta, scale = x[response_start:response_start+2] if config.fit_response else (0., 1.)
        gain = x[response_start+2] if gain_fitted else 0.
        matrix, dm, wm, gm = matrices(delta, scale, gain)
        amplitudes = x[:n_amp]*amp_scale
        bg = b0*np.exp(h @ x[beta_slice])
        mu = np.maximum(bg + matrix @ amplitudes, 1e-12)
        if not want_jac:
            return mu, bg, matrix
        parts = [matrix*amp_scale, bg[:, None]*h]
        if config.fit_response:
            parts += [(dm @ amplitudes)[:, None], (wm @ amplitudes)[:, None]]
            if gain_fitted:
                parts += [(gm @ amplitudes)[:, None]]
        return mu, bg, matrix, np.column_stack(parts)

    def objective(x):
        mu, _, _, jac = evaluate(x, True)
        beta = x[beta_slice]
        reg = config.background_regularization
        # Deviance/2 is the original-count Poisson NLL up to a constant and is numerically stable.
        value = poisson_deviance(y, mu)/2 + 0.5*reg*(beta @ beta)
        grad = jac.T @ (1-y/mu)
        grad[beta_slice] += reg*beta
        return value, grad

    bounds = [(0., None)]*n_amp + [(-3., 3.)]*h.shape[1]
    if config.fit_response:
        bounds += [(-delta_bound, delta_bound), config.resolution_scale_bounds]
        if gain_fitted:
            bounds += [(-config.energy_gain_bound, config.energy_gain_bound)]
    if fixed_areas:
        for symbol, value in fixed_areas.items():
            if symbol in symbols:
                bounds[symbols.index(symbol)] = (value/amp_scale, value/amp_scale)
    mat, _, _, _ = matrices(initial_delta, 1.)
    if n_amp:
        try:
            area0 = nnls(mat / np.sqrt(b0+1)[:, None], np.maximum(y-b0, 0)/np.sqrt(b0+1))[0]
        except (RuntimeError, np.linalg.LinAlgError):
            area0 = np.full(n_amp, max(y.sum()/max(n_amp, 1), 1))
    else:
        area0 = np.empty(0)
    x0 = np.r_[area0/amp_scale, np.zeros(h.shape[1]),
               ([initial_delta, 1.]+([0.] if gain_fitted else [])) if config.fit_response else []]
    if initial is not None and len(initial) == len(x0):
        x0 = np.array(initial, copy=True)
    attempts, solutions = [], []
    for attempt in range(2):
        start = x0.copy()
        if attempt:
            start[:n_amp] = np.maximum(start[:n_amp]*0.65, 0.001)
            start[beta_slice] = 0.1
        for j, (lo, hi) in enumerate(bounds):
            start[j] = np.clip(start[j], lo, np.inf if hi is None else hi)
        opt = minimize(objective, start, jac=True, bounds=bounds, method='L-BFGS-B',
                       options={'maxiter': config.max_iterations, 'ftol': 1e-12, 'gtol': 1e-5, 'maxls': 40})
        attempts.append({'success': bool(opt.success), 'message': str(opt.message),
                         'iterations': int(opt.nit), 'objective': float(opt.fun)})
        solutions.append(opt)
        if opt.success and np.isfinite(opt.fun):
            break
    valid = [o for o in solutions if o.success and np.isfinite(o.fun)]
    opt = min(valid or solutions, key=lambda o: o.fun if np.isfinite(o.fun) else np.inf)
    mu, bg, mat, jac = evaluate(opt.x, True)
    t = opt.x[:n]*amp_scale
    components = mat[:, :n]*t[None, :]
    scatter = mat[:, n:n+len(extras)] @ (opt.x[n:n+len(extras)]*amp_scale)
    auxiliary_areas = {}
    for j, key in enumerate(aux_keys):
        col = n+len(extras)+j
        area = float(opt.x[col]*amp_scale)
        components[:, symbols.index(key[0])] += mat[:, col]*area
        auxiliary_areas['%s_%s_unscaled' % key] = area
    response_start = n_amp+h.shape[1]
    delta, scale = opt.x[response_start:response_start+2] if config.fit_response else (0., 1.)
    gain = float(opt.x[response_start+2]) if gain_fitted else 0.
    response_boundary = config.fit_response and ((np.any(~fixed_centers) and abs(delta) > .99*delta_bound) or
                            scale < config.resolution_scale_bounds[0]*1.01 or scale > config.resolution_scale_bounds[1]*.99)
    response_boundary |= gain_fitted and abs(gain) >= .99*config.energy_gain_bound
    return FitResult(bool(opt.success and np.isfinite(opt.fun)), str(opt.message), float(opt.fun),
                     float(np.sum(mu-xlogy(y, mu))), poisson_deviance(y, mu), symbols,
                     np.divide(t, sums), t, sums, bg, scatter, components, mu, mat, jac,
                     opt.x, bounds, amp_scale, [s[0] for s in extras]+aux_names, float(delta), float(scale),
                     attempts, npar, bool(response_boundary), auxiliary_areas, auxiliary_lines, gain, gain_fitted, line_centers, center_axis)
