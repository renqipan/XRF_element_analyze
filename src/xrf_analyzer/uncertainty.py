import numpy as np


def estimate_covariance(fit):
    """Expected Poisson information with nuisance parameters, column-scaled SVD.

    No optimizer inverse-Hessian approximation is used. Boundary coefficients and
    singular directions receive no symmetric uncertainty.
    """
    n = len(fit.symbols)
    active = np.ones(len(fit.parameters), dtype=bool)
    n_amp = n + len(fit.nuisance_names)
    active[:n_amp] = fit.parameters[:n_amp]*fit.amplitude_scale > 1e-5
    idx = np.flatnonzero(active)
    weights=getattr(fit,'information_weights',None)
    weights=1/fit.model if weights is None else weights
    j = fit.jacobian[:, idx] * np.sqrt(weights[:,None])
    norms = np.linalg.norm(j, axis=0)
    safe_norms = np.maximum(norms, 1e-30)
    scaled = j / safe_norms
    _, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    good = singular > max(singular[0] if len(singular) else 0, 1)*1e-7
    cov_scaled = (vt[good].T / singular[good]**2) @ vt[good]
    cov_active = cov_scaled / safe_norms[:, None] / safe_norms[None, :]
    cov = np.full((len(fit.parameters), len(fit.parameters)), np.nan)
    cov[np.ix_(idx, idx)] = cov_active
    alpha_index=getattr(fit,'background_scale_index',None)
    if alpha_index is not None and np.isfinite(cov[alpha_index,alpha_index]):
        fit.background_scale_std=float(np.sqrt(max(0.,cov[alpha_index,alpha_index])))
        influence=cov[alpha_index,idx]@fit.jacobian[:,idx].T*weights
        fit.background_scale_influence_sample=influence
        fit.background_scale_influence_reference=-fit.background_scale*influence
    null_part = np.sum(vt[~good]**2, axis=0) if (~good).any() else np.zeros(len(idx))
    bad = set(idx[(null_part > 1e-5) | (norms <= 1e-15)].tolist())
    if len(singular) < len(idx):
        bad.update(idx.tolist())
    transform = fit.amplitude_scale / fit.sensitivity_sums
    qcov = cov[:n, :n] * transform[:, None] * transform[None, :]
    statuses = []
    for i in range(n):
        if not fit.success:
            status = 'fit_failed'
        elif not active[i]:
            status = 'nonnegative_boundary'
        elif i in bad or not np.isfinite(qcov[i, i]) or qcov[i, i] <= 0:
            status = 'not_estimable'
        elif fit.q[i] <= 2*np.sqrt(qcov[i, i]):
            status = 'near_boundary_profile_required'
        elif fit.response_at_boundary:
            status = 'response_at_boundary'
        else:
            status = 'fisher_interior'
        statuses.append(status)
    for i, status in enumerate(statuses):
        if status != 'fisher_interior':
            qcov[i, :] = np.nan
            qcov[:, i] = np.nan
    condition = float(singular[0]/singular[-1]) if len(singular) and singular[-1] > 0 else None
    return qcov, statuses, {'method': 'expected_poisson_information_nuisance_adjusted',
                            'scaled_jacobian_condition': condition, 'singular_directions': int((~good).sum())}


def ratio_interval(q, cov, i, ref):
    if i == ref:
        return 1., 0., 1., 1.
    ratio = q[i]/q[ref]
    variance = (cov[i, i] + ratio**2*cov[ref, ref] - 2*ratio*cov[i, ref]) / q[ref]**2
    if not np.isfinite(variance) or variance < -1e-12:
        return ratio, None, None, None
    sigma = np.sqrt(max(0, variance))
    return ratio, 100*sigma/ratio, max(0, ratio-1.96*sigma), ratio+1.96*sigma
