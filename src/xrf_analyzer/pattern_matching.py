"""Global peak-centroid/branch graph: align, cover peaks parsimoniously, retain alternatives."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .response import sigma_kev, FWHM_FACTOR
from .element_prior import choose_by_evidence, element_class

PAIR_COLUMNS = ['peak_id', 'Observed_Centroid_keV', 'Observed_Centroid_Std_keV', 'Z', 'Element', 'Branch',
                'Reference_Energy_keV', 'Aligned_Energy_keV', 'Energy_Residual_keV', 'Match_Tolerance_keV',
                'Strength_Source', 'Group_ID', 'Group_Centroid_keV', 'Group_Affinity', 'Selected_Element',
                'Assignment_Status']


@dataclass
class PatternResult:
    selected: set
    rankings: pd.DataFrame
    pairings: pd.DataFrame
    delta: float
    alignment: dict


def _groups(lines, auxiliary, config):
    groups = []
    for symbol in sorted({l.symbol for l in lines}):
        entries = [(l, l.sensitivity_cm2_g, l.strength_source) for l in lines if l.symbol == symbol]
        # Unknown shell has no cross-shell intensity scale: do not average its
        # centroid or its area with quantitatively anchored shells.
        subsets = [entries, [(l, l.relative_rate, 'unscaled_auxiliary_shell') for l in auxiliary if l.symbol == symbol]]
        for subset in subsets:
            local = []
            for line, strength, source in sorted(subset, key=lambda v: v[0].energy_keV):
                if local and line.energy_keV-local[-1]['last'] <= .65*FWHM_FACTOR*float(sigma_kev(line.energy_keV, config)):
                    local[-1]['entries'].append((line, strength, source))
                    local[-1]['last'] = line.energy_keV
                else:
                    local.append({'entries': [(line, strength, source)], 'last': line.energy_keV})
            for group in local:
                weights = np.array([x[1] for x in group['entries']])
                energies = np.array([x[0].energy_keV for x in group['entries']])
                center = float(weights @ energies/weights.sum())
                spread = float(np.sqrt(weights @ (energies-center)**2/weights.sum()))
                groups.append(dict(symbol=symbol, entries=group['entries'], center=center, spread=spread,
                                   strength=float(weights.sum()), Z=group['entries'][0][0].Z, family=group['entries'][0][0].branch[0],
                                   auxiliary=group['entries'][0][2]=='unscaled_auxiliary_shell', id=len(groups)+1))
    return groups


def discover_elements(peaks, lines, auxiliary, energy, counts, config, fixed_delta=None, fixed_gain=0.):
    groups = _groups(lines, auxiliary, config)
    symbols = sorted({l.symbol for l in lines}, key=lambda s: next(l.Z for l in lines if l.symbol == s))
    n, m = len(groups), len(peaks)
    observed = peaks.energy_keV.to_numpy()
    uncertainty = np.nan_to_num(peaks.centroid_std_keV.to_numpy(), nan=.05) if 'centroid_std_keV' in peaks else np.full(m, .03)
    scores = peaks.approx_significance.to_numpy()
    weights = np.log1p(scores)**1.2
    if m:
        weights *= np.where(peaks.quality_flags.str.contains('broad_structure').to_numpy(), .45, 1.)
    reference_centers = np.array([g['center'] for g in groups])
    centers = reference_centers*(1+fixed_gain)
    widths = FWHM_FACTOR*np.array([float(sigma_kev(c, config)) for c in centers])
    spreads = np.array([g['spread'] for g in groups])
    tolerances = np.sqrt(uncertainty[None, :]**2+config.centroid_tolerance_kev**2
                         +(.12*widths[:, None])**2+(.5*spreads[:, None])**2)
    # A broad observed maximum may be an unresolved multiplet. Expand centroid
    # compatibility only for broad components, not for ordinary precise peaks.
    if m and n:
        observed_sigma = peaks.estimated_fwhm_keV.to_numpy()/FWHM_FACTOR
        nominal_sigma = widths/FWHM_FACTOR
        extra = np.sqrt(np.maximum(observed_sigma[None, :]**2-1.8**2*nominal_sigma[:, None]**2, 0))*.3
        tolerances = np.hypot(tolerances, extra)
    symbol_groups = {s: np.array([i for i, g in enumerate(groups) if g['symbol']==s], dtype=int) for s in symbols}

    def evaluate(delta, details=False):
        residual = observed[None, :]-centers[:, None]-delta
        affinity = np.exp(-.5*(residual/np.maximum(tolerances, 1e-8))**2)
        affinity[abs(residual) > config.match_sigma*tolerances] = 0.
        cover = np.zeros((len(symbols), m))
        penalties = np.zeros(len(symbols))
        for si, symbol in enumerate(symbols):
            idx = symbol_groups[symbol]
            if not len(idx) or not m:
                continue
            anchored = [i for i in idx if not groups[i]['auxiliary']]
            cover[si] = affinity[anchored].max(axis=0) if anchored else 0.
            if np.sum(cover[si] >= .5) >= 2:
                cover[si] = np.maximum(cover[si], affinity[idx].max(axis=0))
            # Anchor expected missing lines to an observed primary group in the
            # SAME K/L/M family only. Do not assume equal attenuation across shells.
            for family in ('K', 'L', 'M'):
                ids = [i for i in idx if groups[i]['family']==family and not groups[i]['auxiliary']]
                if not ids:
                    continue
                anchors = [(groups[i]['strength'], i, int(np.argmax(affinity[i]))) for i in ids
                           if affinity[i].max() >= .5]
                if not anchors:
                    continue
                _, anchor, peak = max(anchors)
                area = float(peaks.iloc[peak].get('net_area_counts', 0))
                q = area/max(groups[anchor]['strength'], 1e-12)
                for i in ids:
                    # A masked/cropped peak, or a branch hidden under another
                    # observed complex, is not evidence that a line is absent.
                    if groups[i]['center'] < energy[0] or groups[i]['center'] > energy[-1]:
                        continue
                    hidden = np.any(abs(observed-(centers[i]+delta)) < .85*widths[i])
                    if affinity[i].max() >= .2 or hidden:
                        continue
                    expected = q*groups[i]['strength']
                    roi = abs(energy-centers[i]-delta) < 1.3*widths[i]
                    observed_total=float(counts[roi].sum())
                    # User hints are not an exhaustive list of absent lines.
                    # Penalize an unlisted branch only if even the raw ROI total
                    # cannot accommodate its expected area conservatively.
                    if config.find_peaks is not None and observed_total+3*np.sqrt(observed_total+1)>=expected:
                        continue
                    noise = np.sqrt(observed_total+expected+1)
                    signal = expected/noise
                    if signal >= config.candidate_threshold_sigma:
                        weak_beta=(config.low_z_kalpha and groups[i]['Z']<=config.low_z_max_z
                                   and all(l.branch.startswith('K') and not l.branch.startswith('Kalpha') for l,_,_ in groups[i]['entries'])
                                   and signal<config.low_z_beta_visibility_sigma)
                        penalties[si] += (0.25 if weak_beta else 1.)*min(6., np.log1p(signal))
        costs = config.pattern_element_penalty+penalties
        selected, explained = [], np.zeros(m)
        while len(selected) < 15:
            gains = np.array([weights @ np.maximum(cover[i]-explained, 0)-costs[i]
                              if i not in selected and symbols[i] != config.tube_target else -np.inf for i in range(len(symbols))])
            if not len(gains) or gains.max() <= 0:
                break
            best = choose_by_evidence(gains, symbols, config.element_prior, config.prior_tie_margin)
            if gains[best] <= 0:
                break
            selected.append(best)
            explained = np.maximum(explained, cover[best])
        objective = float(weights @ explained-sum(costs[i] for i in selected))
        if not details:
            return objective, selected
        return objective, selected, affinity, cover, penalties

    if fixed_delta is not None or not config.fit_response or not m or not n:
        delta = float(fixed_delta or 0.)
        search = []
    else:
        grid = np.linspace(-config.alignment_max_shift_kev, config.alignment_max_shift_kev, 101)
        search = [(float(d), *evaluate(d)) for d in grid]
        best = max(search, key=lambda x: x[1])
        # With no multi-region evidence, the common shift is unidentified. Do
        # not move a single observed peak onto an arbitrary element's line.
        detail = evaluate(best[0], True)
        multi_region = any(np.sum(detail[3][i] >= .5) >= 2 for i in best[2])
        delta = best[0] if multi_region else 0.
    objective, selected_idx, affinity, cover, penalties = evaluate(delta, True)
    selected = {symbols[i] for i in selected_idx}
    rankings = []
    for i, symbol in enumerate(symbols):
        groupids = symbol_groups[symbol]
        # Each observed maximum is only one independent evidence region, even
        # if several unresolved transitions match it.
        ids = [int(peaks.iloc[j].peak_id) for j in range(m) if cover[i, j] >= .5]
        gain = float(weights @ cover[i]-config.pattern_element_penalty-penalties[i])
        rankings.append({'Z': next(l.Z for l in lines if l.symbol==symbol), 'Element': symbol,
                         'PatternScore': gain, 'IndependentPeakRegions': len(ids),
                         'MatchedPeakIDs': ','.join(map(str, ids)), 'MissingLinePenalty': float(penalties[i]),
                         'Selected': symbol in selected, 'Element_Class': element_class(symbol, config.element_prior)})
    # Preserve a strong, single-region observation for review when it cannot
    # justify paying a full element complexity penalty.
    if not selected and m:
        best_single = [(float(weights @ cover[i]-penalties[i]), symbols[i]) for i in range(len(symbols))
                       if cover[i].max(initial=0) >= .5 and symbols[i] != config.tube_target]
        if best_single:
            best = max(x[0] for x in best_single)
            selected = {s for score, s in best_single if score >= best-1.5 and score > 0}
    if m and len(symbols):
        explained = np.max([cover[symbols.index(s)] for s in selected], axis=0) if selected else np.zeros(m)
        for j in range(m):
            if explained[j] >= .5 or scores[j] < config.candidate_threshold_sigma or 'broad_structure' in peaks.iloc[j].quality_flags:
                continue
            options = [(float(cover[i, j]-penalties[i]/6), symbols[i]) for i in range(len(symbols))
                       if cover[i, j] >= .3 and penalties[i] < 2 and symbols[i] != config.tube_target]
            if options:
                top = max(score for score, _ in options)
                options.sort(key=lambda item: (item[0], element_class(item[1], config.element_prior)=='common'), reverse=True)
                selected.update(s for score, s in options[:2] if score >= top-.2)
    # Greedy peak coverage is a proposal mechanism. An already-covered peak
    # may contain a second element, and the covering element can be wrong.
    # Retain positive multi-region alternatives for the joint likelihood test;
    # they do not become supported merely by entering this bounded shortlist.
    alternatives = [i for i in range(len(symbols)) if symbols[i] != config.tube_target
                    and np.sum(cover[i] >= .5) >= 2
                    and weights @ cover[i]-config.pattern_element_penalty-penalties[i] > 0]
    alternatives.sort(key=lambda i: float(weights @ cover[i]-penalties[i]), reverse=True)
    retained_alternatives = []
    for i in alternatives:
        ids = symbol_groups[symbols[i]]
        # Require an observed principal group, rather than borrowing large
        # peaks through weak minor branches of a speculative element.
        primary_anchor = False
        for family in ('K', 'L', 'M'):
            family_ids = [g for g in ids if not groups[g]['auxiliary'] and groups[g]['family'] == family]
            if not family_ids:
                continue
            strongest = max(family_ids, key=lambda g: groups[g]['strength'])
            matches = affinity[strongest] >= .5
            family_matches = np.max(affinity[family_ids], axis=0) >= .5
            if matches.any() and scores[matches].max() >= .5*scores[family_matches].max():
                primary_anchor = True
        if (primary_anchor and symbols[i] not in selected and len(selected) < 15
                and len(retained_alternatives) < 3):
            selected.add(symbols[i])
            retained_alternatives.append(symbols[i])
    pairings = []
    for i, g in enumerate(groups):
        for j in range(m):
            if affinity[i, j] <= 0:
                continue
            p = peaks.iloc[j]
            for line, _, source in g['entries']:
                pairings.append([int(p.peak_id), float(p.energy_keV), float(uncertainty[j]), line.Z, line.symbol,
                                 line.branch, line.energy_keV, line.energy_keV*(1+fixed_gain)+delta,
                                 float(p.energy_keV-line.energy_keV*(1+fixed_gain)-delta), config.match_sigma*float(tolerances[i, j]),
                                 source, g['id'], g['center'], float(affinity[i, j]), line.symbol in selected,
                                 'selected_group_candidate' if line.symbol in selected else 'alternative'])
    alignment = {'method': 'global_centroid_branch_graph_weighted_sparse_peak_cover', 'delta_E_keV': delta,
                 'search_bound_keV': config.alignment_max_shift_kev, 'score': objective,
                 'is_calibration_replacement': False, 'gain_for_matching': fixed_gain,
                 'element_prior': config.element_prior, 'prior_tie_margin': config.prior_tie_margin,
                 'retained_multiregion_alternatives': retained_alternatives, 'grid_solutions': [{'delta': d, 'score': score} for d, score, _ in search]}
    rankings = pd.DataFrame(rankings, columns=['Z', 'Element', 'PatternScore', 'IndependentPeakRegions',
                                                'MatchedPeakIDs', 'MissingLinePenalty', 'Selected', 'Element_Class'])
    rankings['Selected'] = rankings.Element.isin(selected)
    return PatternResult(selected, rankings, pd.DataFrame(pairings, columns=PAIR_COLUMNS), delta, alignment)
