"""Measure identified primary-line areas without imposing distant branch ratios.

The fitted quantity is q=A/sum(S), on the sensitivity table's common scale.
It is an apparent thin-sample estimate, not an absolute mass or a validation of
unknown excitation/absorption corrections. Local count covariance is propagated
with per-channel influence functions, including overlapping measurement windows.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import xlogy
from .response import (sigma_kev, FWHM_FACTOR, line_model_center,
                       response_with_derivatives)


PRIMARY_BRANCHES = {'Kalpha1', 'Kalpha2', 'Lalpha1', 'Lalpha2', 'Malpha1'}


def _initial_shell(line):
    shell = getattr(line, 'parent_shell', '')
    if shell:
        return shell
    if line.branch.startswith('K'):
        return 'K'
    return {'Lalpha1': 'L3', 'Lalpha2': 'L3', 'Lbeta2': 'L3',
            'Lbeta1': 'L2', 'Lgamma1': 'L2', 'Malpha1': 'M5'}.get(line.branch, line.branch[:2])


def _tie_unconfirmed_nuisance(groups, allowed=None):
    """A faint hypothetical line cannot acquire an unlimited independent area.

    For an unconfirmed interferent preserve known *same initial-shell* branch
    ratios. L1/L2/L3 remain separate. Supported elements' resolved groups remain
    free, so a distant branch mismatch cannot force the measured main area.
    """
    result, buckets = [], {}
    for group in groups:
        if group['confirmed']:
            result.append(group)
            continue
        for k, line in enumerate(group['lines']):
            key = (group['Element'], group['auxiliary'], _initial_shell(line))
            if allowed is not None and key not in allowed:
                continue
            bucket = buckets.setdefault(key, {'Element': group['Element'], 'Z': group['Z'],
                'auxiliary': group['auxiliary'], 'confirmed': False, 'lines': [],
                'weights': [], 'centers': [], 'sigmas': [], 'expected': 0.})
            bucket['lines'].append(line)
            bucket['weights'].append(group['weights'][k]*group['S'])
            bucket['centers'].append(group['centers'][k])
            bucket['sigmas'].append(group['sigmas'][k])
            bucket['expected'] += group['expected']*group['weights'][k]
    for key, bucket in buckets.items():
        bucket['weights'] = np.asarray(bucket['weights'])
        bucket['S'] = float(bucket['weights'].sum())
        bucket['weights'] /= bucket['S']
        bucket['centers'] = np.asarray(bucket['centers'])
        bucket['sigmas'] = np.asarray(bucket['sigmas'])
        bucket['Branches'] = '+'.join(l.branch for l in bucket['lines'])
        bucket['Nuisance_Policy'] = 'unconfirmed_same_initial_shell_'+key[2]+'_ratios_retained'
        result.append(bucket)
    return result


def _groups(identification, config):
    fit = identification.fit
    confirmed = set(identification.candidates.loc[identification.candidates.Status == 'supported', 'Element'])
    groups = []
    for symbol in fit.symbols:
        entries = []
        for line in identification.fit_lines:
            if line.symbol == symbol:
                entries.append((line, line.sensitivity_cm2_g,
                                fit.q[fit.symbols.index(symbol)]*line.sensitivity_cm2_g))
        for shell in {l.parent_shell for l in fit.auxiliary_lines if l.symbol == symbol}:
            auxiliary = [l for l in fit.auxiliary_lines if l.symbol == symbol and l.parent_shell == shell]
            rate = sum(l.relative_rate for l in auxiliary)
            area = fit.auxiliary_areas.get(symbol+'_'+shell+'_unscaled', 0.)
            entries.extend((l, l.relative_rate, area*l.relative_rate/rate) for l in auxiliary)
        local = []
        for line, strength, expected in sorted(entries, key=lambda item: item[0].energy_keV):
            center = line_model_center(line, fit)
            sigma = float(sigma_kev(line.energy_keV, config))*fit.resolution_scale
            # Auxiliary shells have no known intensity scale relative to a
            # quantitative shell and may never be merged into its S sum.
            auxiliary = not hasattr(line, 'sensitivity_cm2_g')
            if (local and auxiliary == local[-1]['auxiliary']
                    and center-local[-1]['last'] <= .65*FWHM_FACTOR*sigma):
                group = local[-1]
            else:
                group = {'Element': symbol, 'Z': line.Z, 'lines': [], 'weights': [],
                         'centers': [], 'sigmas': [], 'expected': 0., 'auxiliary': auxiliary}
                local.append(group)
            group['lines'].append(line)
            group['weights'].append(float(strength))
            group['centers'].append(center)
            group['sigmas'].append(sigma)
            group['expected'] += max(0., float(expected))
            group['last'] = center
        for group in local:
            group['weights'] = np.asarray(group['weights'])
            group['S'] = float(group['weights'].sum())
            group['weights'] /= group['S']
            group['centers'] = np.asarray(group['centers'])
            group['sigmas'] = np.asarray(group['sigmas'])
            group['center'] = float(group['weights']@group['centers'])
            group['primary'] = not group['auxiliary'] and any(l.branch in PRIMARY_BRANCHES for l in group['lines'])
            group['confirmed'] = symbol in confirmed
            group['Branches'] = '+'.join(l.branch for l in group['lines'])
            groups.append(group)
    # A resolved alpha doublet still measures one elemental mass coefficient:
    # its two lines share an initial shell and known relative sensitivities.
    # Detection resolution must not split these into competing free amplitudes.
    # Keep all minor lines already unresolved with either member, while leaving
    # distant beta/gamma groups independent of the primary measurement.
    merged, primary = [], {}
    for group in groups:
        alpha_families = {l.branch[0]+'alpha' for l in group['lines'] if l.branch in PRIMARY_BRANCHES}
        if not group['primary'] or len(alpha_families) != 1:
            merged.append(group)
            continue
        key = (group['Element'], next(iter(alpha_families)))
        if key not in primary:
            primary[key] = group
            merged.append(group)
            continue
        anchor = primary[key]
        strengths = np.r_[anchor['weights']*anchor['S'], group['weights']*group['S']]
        anchor['S'] += group['S']
        anchor['weights'] = strengths/anchor['S']
        anchor['lines'] += group['lines']
        anchor['centers'] = np.r_[anchor['centers'], group['centers']]
        anchor['sigmas'] = np.r_[anchor['sigmas'], group['sigmas']]
        anchor['expected'] += group['expected']
        anchor['center'] = float(anchor['weights']@anchor['centers'])
        anchor['Branches'] = '+'.join(l.branch for l in anchor['lines'])
        anchor['Primary_Group_Policy'] = 'same_alpha_doublet_common_area_over_summed_sensitivity'
    return merged


def _profile(target, groups, counts, energy, edges, config, radius):
    low = np.min(target['centers']-radius*target['sigmas'])
    high = np.max(target['centers']+radius*target['sigmas'])
    audit = {'Background_Window_Sigmas': radius, 'Success': False}
    # If a hypothetical weak line could overlap the target, include the same
    # shell's strongest neighbouring group as an independent count constraint.
    # The window expands by at most 1.5 keV per side; no amplitude is fixed to
    # the global q. This prevents a free faint Pm line swallowing Fe Ka while
    # ignoring the accompanying much stronger Pm lines.
    def selected_nuisance(lo, hi):
        local_edges = np.array([max(edges[0], lo), min(edges[-1], hi)])
        selected = []
        for group in groups:
            if group is target:
                continue
            fraction = float((response_with_derivatives(local_edges, group['centers'], group['sigmas'])[0]
                              @group['weights']).sum())
            if group['expected']*fraction >= max(2., .001*max(target['expected'], 1.)):
                selected.append(group)
        return selected
    initial_nuisance = selected_nuisance(low, high)
    anchored_keys = {(g['Element'], g['auxiliary'], _initial_shell(l)) for g in initial_nuisance
                     if not g['confirmed'] for l in g['lines']}
    expansions = []
    original_low, original_high = low, high
    for key in anchored_keys:
        options = [g for g in groups if not g['confirmed'] and g['Element'] == key[0]
                   and g['auxiliary'] == key[1] and any(_initial_shell(l) == key[2] for l in g['lines'])]
        anchor = max(options, key=lambda g: g['expected'])
        anchor_low = float(np.min(anchor['centers']-2*anchor['sigmas']))
        anchor_high = float(np.max(anchor['centers']+2*anchor['sigmas']))
        if (original_low-1.5 <= anchor_low and anchor_high <= original_high+1.5
                and (anchor_low < low or anchor_high > high)):
            low, high = min(low, anchor_low), max(high, anchor_high)
            expansions.append(key[0]+':'+key[2]+':'+anchor['Branches'])
    ids = np.flatnonzero((energy >= low) & (energy <= high))
    audit['Interferent_Anchor_Extensions'] = expansions
    if len(ids) < 9:
        audit['Failure'] = 'too_few_local_channels'
        return audit
    a, b = int(ids[0]), int(ids[-1]+1)
    e, y, le = energy[a:b], counts[a:b], edges[a:b+1]
    # Give every relevant neighbouring group its own amplitude. Global q values
    # are only used to exclude numerically negligible nuisance groups.
    nuisance = selected_nuisance(le[0], le[-1])
    keys = {(g['Element'], g['auxiliary'], _initial_shell(l)) for g in nuisance
            if not g['confirmed'] for l in g['lines']}
    # Retain all same-shell transitions in a selected nuisance template, even
    # when some individual branches are weak or outside this window.
    expanded = [g for g in nuisance if g['confirmed']]
    expanded += [g for g in groups if not g['confirmed'] and any(
        (g['Element'], g['auxiliary'], _initial_shell(l)) in keys for l in g['lines'])]
    nearby = [target]+_tie_unconfirmed_nuisance(expanded, keys)
    ng = len(nearby)
    if len(y) < ng+6:
        audit['Failure'] = 'too_many_local_interferences'
        return audit
    h = np.column_stack([np.ones(len(e)), (e-e.mean())/max(np.ptp(e), 1e-9)])
    centers = np.concatenate([g['centers'] for g in nearby])
    widths = np.concatenate([g['sigmas'] for g in nearby])
    weights = np.zeros((len(centers), ng))
    offset = 0
    for k, group in enumerate(nearby):
        weights[offset:offset+len(group['lines']), k] = group['weights']
        offset += len(group['lines'])
    area_scale = max(float(y.sum())/max(ng, 1), 10.)
    # User-fixed centres and explicitly fixed response remain fixed. Automatic
    # data may refine a single local shift/width, never separate lines freely.
    refine = bool(config.fit_response and config.find_peaks is None)
    shift_bound = min(.12, .8*float(np.max(target['sigmas'])))
    bounds = [(0., None)]*ng+[(-12., 20.), (-8., 8.),
                              (-shift_bound, shift_bound) if refine else (0., 0.),
                              (.65, 1.8) if refine else (1., 1.)]

    def model(v, derivatives=False):
        response, ds, dw = response_with_derivatives(le, centers, widths, delta=v[-2], scale=v[-1])
        design = response@weights
        bg = np.exp(h@v[ng:ng+2])
        mu = np.maximum(bg+design@(v[:ng]*area_scale), 1e-12)
        if not derivatives:
            return mu
        jac = np.column_stack([design*area_scale, bg[:, None]*h,
                               (ds@weights)@(v[:ng]*area_scale),
                               (dw@weights)@(v[:ng]*area_scale)])
        return mu, jac

    baseline = max(float(np.percentile(y, 25)), .2)
    initial = np.r_[np.maximum([g['expected']/area_scale for g in nearby], 1e-5),
                    np.log(baseline), 0., 0., 1.]
    design = response_with_derivatives(le, centers, widths)[0]@weights
    try:
        initial[:ng] = nnls(design*area_scale, np.maximum(y-baseline, 0.))[0]
    except (ValueError, RuntimeError):
        pass

    def solve(null=False):
        actual_bounds = list(bounds)
        start = initial.copy()
        if null:
            actual_bounds[0] = (0., 0.)
            start[0] = 0.
        def objective(v):
            mu, jac = model(v, True)
            return float(np.sum(mu-xlogy(y, mu))), jac.T@(1-y/mu)
        return minimize(objective, start, jac=True, bounds=actual_bounds, method='L-BFGS-B',
                        options={'maxiter': 1200, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 50})

    full, null = solve(), solve(True)
    audit.update(Optimizer_Message=str(full.message), ROI_Start_keV=float(le[0]),
                 ROI_End_keV=float(le[-1]), Nuisance_Groups=[g['Element']+':'+g['Branches'] for g in nearby[1:]],
                 Nuisance_Policies=[g.get('Nuisance_Policy', 'supported_resolved_group_area_free') for g in nearby[1:]])
    if not full.success or not null.success:
        audit['Failure'] = 'local_optimization_failed'
        return audit
    mu, jac = model(full.x, True)
    free = np.array([lo != hi for lo, hi in bounds])
    j = jac[:, free]
    # Column normalization prevents intensity/background/energy parameter units
    # from determining a numerical identifiability decision.
    scales = np.linalg.norm(j/np.sqrt(mu[:, None]), axis=0)
    normalized = (j/np.maximum(scales, 1e-30))/np.sqrt(mu[:, None])
    _, singular, vh = np.linalg.svd(normalized, full_matrices=False)
    keep = singular > max(singular[0]*1e-7, 1e-12)
    null_weight = float(np.sum(vh[~keep, 0]**2)) if not keep.all() else 0.
    if null_weight > 1e-6:
        audit['Failure'] = 'primary_area_not_identifiable'
        return audit
    inverse = (vh[keep].T/(singular[keep]**2))@vh[keep]
    covariance = inverse/np.maximum(scales[:, None]*scales[None, :], 1e-30)
    influence = (covariance[0]@j.T/mu)*area_scale/target['S']
    q = float(full.x[0]*area_scale/target['S'])
    std = float(np.sqrt(max(covariance[0, 0], 0.))*area_scale/target['S'])
    deviance = float(2*np.sum(xlogy(y, np.maximum(y, 1e-300)/mu)-(y-mu)))
    dof = max(len(y)-int(free.sum()), 1)
    gain = max(0., 2*(null.fun-full.fun))
    area = float(full.x[0]*area_scale)
    area_fraction = float((response_with_derivatives(le, centers, widths, delta=full.x[-2], scale=full.x[-1])[0]@weights)[:, 0].sum())
    shape_boundary = bool(refine and (abs(full.x[-2]) >= .98*shift_bound or
                          full.x[-1] <= .655 or full.x[-1] >= 1.795))
    audit.update(Success=True, q_Area=q, q_Fisher_Std=std, Area_Counts=area,
                 Sensitivity_Sum=target['S'], Area_Fraction_in_Window=area_fraction,
                 DeltaDeviance=gain, Deviance=deviance, Degrees_of_Freedom=dof,
                 Deviance_per_dof=deviance/dof, Local_Shift_keV=float(full.x[-2]),
                 Local_Width_Scale=float(full.x[-1]), Response_At_Boundary=shape_boundary,
                 Local_Primary_Center_keV=float(target['weights']@target['centers']+full.x[-2]),
                 _indices=ids, _influence=influence, _model=mu)
    final_design = response_with_derivatives(le, centers, widths, delta=full.x[-2], scale=full.x[-1])[0]@weights
    audit['_primary_counts'] = final_design[:, 0]*area
    audit['_background'] = np.exp(h@full.x[ng:ng+2])
    audit['_nuisance_counts'] = np.maximum(mu-audit['_primary_counts']-audit['_background'], 0.)
    return audit


def fit_local_quantification(identification, counts, energy, edges, config):
    """Return primary-line A/S estimates and covariance for supported identities.

    ``elements`` has one row per supported fitted identity. Valid rows have
    Quantification_Status=local_area_estimate; rejected rows retain the reason.
    Covariance arrays use the DataFrame row order; invalid rows/columns are NaN.
    ``audit`` contains only JSON-serializable local-fit evidence.
    """
    counts, energy, edges = map(lambda a: np.asarray(a, float), (counts, energy, edges))
    groups = _groups(identification, config)
    candidates = identification.candidates
    supported = candidates.loc[(candidates.Status == 'supported') &
                               candidates.Element.isin(identification.fit.symbols)]
    records, audit, chosen = [], [], []
    for candidate in supported.itertuples(index=False):
        row = {'Element': candidate.Element, 'Z': candidate.Z, 'q_Area': np.nan, 'q_Std': np.nan,
               'q_Stat_Std': np.nan, 'Quantification_Status': 'local_area_not_estimable',
               'Failure_Reason': 'no_primary_group_in_measurable_range',
               'Notes': 'Conditional on supplied common-scale S; local Gaussian/background and candidate set; S/absorption systematics excluded'}
        options = [g for g in groups if g['Element'] == candidate.Element and g['primary']
                   and energy[0] < g['center'] < energy[-1]]
        # Select using observed support/available sensitivity, not the best
        # realized q, so selection cannot cherry-pick an upward noisy branch.
        options.sort(key=lambda g: g['S'], reverse=True)
        successful = None
        for group in options:
            narrow = _profile(group, groups, counts, energy, edges, config, 4.)
            wide = _profile(group, groups, counts, energy, edges, config, 6.)
            item = {'Element': candidate.Element, 'Branches': group['Branches'],
                    'Primary_Group_Policy': group.get('Primary_Group_Policy', 'unresolved_primary_group'),
                    'Sensitivity_Sum': group['S'], 'Narrow': {k:v for k,v in narrow.items() if not k.startswith('_')},
                    'Wide': {k:v for k,v in wide.items() if not k.startswith('_')}}
            audit.append(item)
            if not narrow['Success'] or not wide['Success']:
                row['Failure_Reason'] = narrow.get('Failure', wide.get('Failure', 'local_fit_failed'))
                continue
            fraction = min(narrow['Area_Fraction_in_Window'], wide['Area_Fraction_in_Window'])
            difference = abs(narrow['q_Area']-wide['q_Area'])
            relative_difference = difference/max(wide['q_Area'], 1e-20)
            detectable = min(narrow['DeltaDeviance'], wide['DeltaDeviance']) >= 9.
            if fraction < .85 or not detectable or relative_difference > .5 or wide['q_Area'] <= 0:
                row['Failure_Reason'] = ('primary_area_truncated' if fraction < .85 else
                    'primary_area_not_detectable' if not detectable else 'background_window_unstable')
                item['Accepted'] = False
                continue
            phi = max(1., narrow['Deviance_per_dof'], wide['Deviance_per_dof'])
            row.update(q_Area=wide['q_Area'], Quantification_Status='local_area_estimate', Failure_Reason='',
                       Selected_Branches=group['Branches'], Area_Counts=wide['Area_Counts'],
                       Sensitivity_Sum=group['S'], Background_Window_Relative_Difference=relative_difference,
                       Window_Systematic_q=difference, Local_Overdispersion=phi,
                       Local_Response_At_Boundary=wide['Response_At_Boundary'],
                       Method='local_primary_area_over_common_scale_sensitivity')
            item['Accepted'] = True
            successful = (wide, phi, difference)
            break
        records.append(row)
        chosen.append(successful)
    frame = pd.DataFrame(records, columns=None if records else [
        'Element', 'Z', 'q_Area', 'q_Std', 'q_Stat_Std', 'Quantification_Status',
        'Failure_Reason', 'Notes', 'Selected_Branches', 'Area_Counts', 'Sensitivity_Sum'])
    n = len(records)
    influence = np.zeros((n, len(counts)))
    mean_sum, mean_n = np.zeros(len(counts)), np.zeros(len(counts))
    for i, selected in enumerate(chosen):
        if selected is None:
            continue
        fit, _, _ = selected
        influence[i, fit['_indices']] = fit['_influence']
        mean_sum[fit['_indices']] += fit['_model']
        mean_n[fit['_indices']] += 1
    mean = mean_sum/np.maximum(mean_n, 1.)
    stat_covariance = (influence*mean)@influence.T
    inflation = np.array([np.sqrt(s[1]) if s is not None else 1. for s in chosen])
    covariance = stat_covariance*inflation[:, None]*inflation[None, :]
    for i, selected in enumerate(chosen):
        if selected is None:
            covariance[i, :] = covariance[:, i] = np.nan
            stat_covariance[i, :] = stat_covariance[:, i] = np.nan
        else:
            covariance[i, i] += selected[2]**2
            frame.loc[i, 'q_Stat_Std'] = np.sqrt(max(stat_covariance[i, i], 0.))
            frame.loc[i, 'q_Std'] = np.sqrt(max(covariance[i, i], 0.))
    fits = []
    for record, selected in zip(records, chosen):
        if selected is None:
            continue
        fit, _, _ = selected
        fits.append({'Element': record['Element'], 'Branches': record['Selected_Branches'],
                     'indices': fit['_indices'], 'model': fit['_model'],
                     'primary_counts': fit['_primary_counts'], 'background': fit['_background'],
                     'nuisance_counts': fit['_nuisance_counts'], 'center_keV': fit['Local_Primary_Center_keV']})
    return {'elements': frame, 'covariance': covariance, 'stat_covariance': stat_covariance, 'fits': fits,
            'audit': audit, 'method': 'local_primary_peak_area_over_sensitivity',
            'covariance_method': 'shared_channel_influence_Poisson_covariance; local_overdispersion_plus_window_difference',
            'assumptions': 'common_scale_S_assumed_applicable; apparent_thin_sample; S_systematics_not_included'}
