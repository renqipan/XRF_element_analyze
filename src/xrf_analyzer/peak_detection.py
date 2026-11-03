"""Element-independent, sub-channel Poisson centroid fitting of observed peaks."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import minimize
from scipy.special import xlogy, ndtr
from .response import sigma_kev, gaussian_response, FWHM_FACTOR
from .measured_background import profile_background

PEAK_COLUMNS = ['peak_id', 'channel', 'energy_keV', 'local_prominence', 'estimated_fwhm_keV',
                'approx_significance', 'roi_start_keV', 'roi_end_keV', 'quality_flags',
                'maximum_channel', 'maximum_energy_keV', 'centroid_std_keV', 'net_area_counts',
                'area_std_counts', 'local_fit_status', 'local_deviance_per_dof']


def _local_fit(energy, edges, y, b0, centers0, sigmas0, nominal_sigmas, config, reference=None):
    alpha=getattr(config,'_background_scale',getattr(config,'_background_scale_initial',1.))
    if reference is not None:b0=np.maximum(b0-alpha*reference,.1)
    n = len(centers0)
    area_scale = max(y.sum()/max(n, 1), 10.)
    b0 = np.maximum(b0, .1)
    span = max(energy[-1]-energy[0], 1e-6)
    xcoord = (energy-energy.mean())/span
    h = np.column_stack([np.ones(len(y)), xcoord])
    bounds = []
    dw = np.median(np.diff(edges))
    for i in range(n):
        radius = max(2*sigmas0[i], dw)
        lo, hi = centers0[i]-radius, centers0[i]+radius
        if i:
            lo = max(lo, .5*(centers0[i-1]+centers0[i]))
        if i+1 < n:
            hi = min(hi, .5*(centers0[i]+centers0[i+1]))
        bounds.extend([(0., None), (lo, hi),
                       (max(.35*nominal_sigmas[i], .20*dw), max(3*nominal_sigmas[i], 2*sigmas0[i]))])
    bounds += [(-2., 2.), (-2., 2.)]
    start = []
    for center, sig in zip(centers0, sigmas0):
        p = gaussian_response(edges, center, sig)
        area = max(1., (np.maximum(y-b0, 0) @ p)/max(p @ p, 1e-12))
        start.extend([area/area_scale, center, sig])
    start += [0., 0.]

    def evaluate(x):
        bg = b0*np.exp(h @ x[-2:])
        mu = bg.copy()
        jac = np.zeros((len(y), len(x)))
        for i in range(n):
            area, center, sig = x[3*i:3*i+3]
            z = (edges-center)/sig
            phi = np.exp(-z*z/2)/np.sqrt(2*np.pi)
            p = np.maximum(np.diff(ndtr(z)), 0)
            mu += area*area_scale*p
            jac[:, 3*i] = area_scale*p
            jac[:, 3*i+1] = area*area_scale*(phi[:-1]-phi[1:])/sig
            jac[:, 3*i+2] = area*area_scale*(z[:-1]*phi[:-1]-z[1:]*phi[1:])/sig
        jac[:, -2:] = bg[:, None]*h
        return np.maximum(mu, 1e-12), jac

    def objective(x):
        mu, jac = evaluate(x)
        residual=1-y/mu
        loss = np.sum(mu-xlogy(y, mu)) + .05*(x[-2:] @ x[-2:])
        if reference is not None:
            from .measured_background import profile_background
            prof=profile_background(y,mu,alpha,reference)
            loss=prof['nll']+.05*(x[-2:]@x[-2:]);residual=prof['signal_gradient']
        grad = jac.T @ residual
        grad[-2:] += .1*x[-2:]
        return loss, grad

    opt = minimize(objective, start, jac=True, bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-5, 'maxls': 40})
    mu, jac = evaluate(opt.x)
    # Local information includes neighboring peaks and background. SVD detects rank loss.
    if reference is not None:
        prof=profile_background(y,mu,alpha,reference)
        j=jac*np.sqrt(prof['information_weight'][:,None])
    else:j = jac/np.sqrt(mu[:, None])
    norms = np.maximum(np.linalg.norm(j, axis=0), 1e-20)
    _, s, vt = np.linalg.svd(j/norms, full_matrices=False)
    good = s > max(s[0] if len(s) else 1, 1)*1e-7
    cov = ((vt[good].T/s[good]**2) @ vt[good])/norms[:, None]/norms[None, :]
    full_rank = good.all() and len(s) == len(opt.x)
    deviance = float(2*np.sum(xlogy(y, np.maximum(y, 1e-300)/mu)-(y-mu)))
    if reference is not None:
        deviance=prof['deviance']
    result = []
    for i in range(n):
        null_bounds = list(bounds)
        null_bounds[3*i] = (0., 0.)
        xnull = opt.x.copy()
        xnull[3*i] = 0.
        null = minimize(objective, xnull, jac=True, bounds=null_bounds, method='L-BFGS-B',
                        options={'maxiter': 400, 'ftol': 1e-10, 'maxls': 40})
        score = np.sqrt(max(0., 2*(null.fun-opt.fun))) if null.success and opt.success else 0.
        area, center, sig = opt.x[3*i:3*i+3]
        center_se = np.sqrt(max(cov[3*i+1, 3*i+1], 0)) if full_rank and area*area_scale > 1 else np.nan
        area_se = area_scale*np.sqrt(max(cov[3*i, 3*i], 0)) if full_rank else np.nan
        result.append(dict(center=float(center), sigma=float(sig), area=float(area*area_scale),
                           center_se=float(center_se), area_se=float(area_se), score=float(score),
                           success=bool(opt.success), deviance_per_dof=deviance/max(len(y)-len(opt.x), 1)))
    return result


def detect_peaks(channel, energy, edges, counts, background, config):
    width = float(np.median(np.diff(edges)))
    sig_chan = float(sigma_kev(np.median(energy), config))/width
    smooth = gaussian_filter1d(counts-background, max(.5, .5*sig_chan))
    indices, props = find_peaks(smooth, prominence=max(.5, np.sqrt(np.median(background)+1)),
                               distance=max(1, int(sig_chan)))
    widths = peak_widths(smooth, indices, rel_height=.5)[0] if len(indices) else []
    proposals = []
    for n, idx in enumerate(indices):
        nominal = float(sigma_kev(energy[idx], config))
        measured = max(widths[n]*width/FWHM_FACTOR, .5*nominal)
        sig = min(measured, 3*nominal)
        radius = max(4*sig, 4*nominal, 3*width)
        proposals.append(dict(idx=idx, prominence=props['prominences'][n], sigma=sig,
                              nominal=nominal, left=energy[idx]-radius, right=energy[idx]+radius))
    # Fit adjacent observed maxima jointly; never sequentially subtract overlapping peaks.
    groups = []
    for proposal in proposals:
        if groups and proposal['left'] < groups[-1]['right'] and len(groups[-1]['peaks']) < 6:
            groups[-1]['peaks'].append(proposal)
            groups[-1]['right'] = max(groups[-1]['right'], proposal['right'])
        else:
            groups.append(dict(left=proposal['left'], right=proposal['right'], peaks=[proposal]))
    rows = []
    for group in groups:
        positions = np.flatnonzero((energy >= group['left']) & (energy <= group['right']))
        if len(positions) < 6:
            continue
        a, b = positions[0], positions[-1]+1
        proposals = group['peaks']
        fitted = _local_fit(energy[a:b], edges[a:b+1], counts[a:b], background[a:b],
                            [energy[p['idx']] for p in proposals], [p['sigma'] for p in proposals],
                            [p['nominal'] for p in proposals], config,
                            None if getattr(config,'_background_counts',None) is None else config._background_counts[a:b])
        for proposal, f in zip(proposals, fitted):
            if not f['success'] or f['score'] < config.candidate_threshold_sigma:
                continue
            idx = proposal['idx']
            flags = ['poisson_fitted_centroid', 'local_score_not_global_p_value']
            if f['sigma'] > 2.5*proposal['nominal']:
                flags.append('broad_structure')
            if f['center']-3*f['sigma'] < edges[0] or f['center']+3*f['sigma'] > edges[-1]:
                flags.append('roi_truncated')
            if f['deviance_per_dof'] > 5:
                flags.append('local_shape_mismatch_or_unresolved_multiplet')
            ch = float(channel[0]+(f['center']-energy[0])/width)
            rows.append([0, ch, f['center'], float(proposal['prominence']), FWHM_FACTOR*f['sigma'],
                         f['score'], max(edges[0], f['center']-3*f['sigma']), min(edges[-1], f['center']+3*f['sigma']),
                         ';'.join(flags), int(channel[idx]), float(energy[idx]), f['center_se'],
                         f['area'], f['area_se'], 'success', f['deviance_per_dof']])
    result = pd.DataFrame(rows, columns=PEAK_COLUMNS).sort_values('energy_keV').reset_index(drop=True)
    result['peak_id'] = np.arange(1, len(result)+1)
    return result
