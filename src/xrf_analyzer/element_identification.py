from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from .response import sigma_kev, FWHM_FACTOR, gaussian_response, line_model_center
from .fit_model import fit_spectrum
from .uncertainty import estimate_covariance
from .pattern_matching import discover_elements
from .manual_peaks import constrain_line_centers
from .element_prior import element_class, preferred_in_group
from .branch_evidence import evaluate_branch_evidence

CANDIDATE_COLUMNS = ['Z', 'Element', 'Status', 'Score', 'DeltaDeviance', 'NumObservableLines',
                     'NumIndependentEvidenceRegions', 'ExpectedMissingStrongLines', 'OverlapFlag',
                     'ConfidenceNotes', 'MatchedPeakIDs', 'OverlapElements', 'q_Fit', 'q_Std', 'Upper95_q',
                     'Uncertainty_Status', 'Quantification_Status', 'PatternScore', 'PatternMissingLinePenalty']


@dataclass
class IdentificationResult:
    fit: object
    candidates: pd.DataFrame
    qcov: np.ndarray
    uncertainty_status: list
    uncertainty_info: dict
    fit_lines: list
    comparisons: list
    overlap_pairs: list
    diagnostics: list
    pairings: object = None
    pattern_rankings: object = None
    alignment: object = None
    branch_evidence: object = None


def _regions(lines, fit, idx, energy, counts, config):
    """Evidence is measured in disjoint resolution regions, not line count."""
    groups = []
    for line in sorted(lines, key=lambda l: l.energy_keV):
        sig = float(sigma_kev(line.energy_keV, config))*fit.resolution_scale
        lo, hi = line_model_center(line,fit)-2.5*sig, line_model_center(line,fit)+2.5*sig
        if groups and lo <= groups[-1][1]:
            groups[-1][1] = max(hi, groups[-1][1])
            groups[-1][2].append(line)
        else:
            groups.append([lo, hi, [line]])
    evidence, missing, diagnostics, mismatch = 0, [], [], False
    for lo, hi, branches in groups:
        mask = (energy >= lo) & (energy <= hi)
        if not mask.any():
            continue
        component = fit.components[mask, idx]
        info = float(np.sum(component**2/fit.model[mask]))
        expected = float(component.sum())
        # Region residual uses the original-data residual, not hard-bound line ratios.
        residual_z = float(np.sum(counts[mask]-fit.model[mask])/np.sqrt(fit.model[mask].sum()+1))
        strength = np.sqrt(info)
        if strength >= 3:
            evidence += 1
        bad = strength >= 5 and abs(residual_z) >= 5
        mismatch |= bad
        if strength >= 5 and residual_z < -5:
            missing += [l.branch for l in branches]
        diagnostics.append({'Element': fit.symbols[idx], 'Branches': '+'.join(l.branch for l in branches),
                            'Family': '+'.join(sorted({l.branch[0] for l in branches})),
                            'ExpectedCounts': expected, 'InformationScore': strength,
                            'RegionResidualZ': residual_z, 'branch_ratio_mismatch': bool(bad)})
    return evidence, missing, diagnostics, mismatch


def identify_elements(counts, energy, edges, background, lines, peaks, config, delta_bound,
                      calibration_rms=0, auxiliary_lines=(), manual_response=None):
    all_symbols = sorted({l.symbol for l in lines}, key=lambda s: next(l.Z for l in lines if l.symbol == s))
    manual_reference = config.find_peaks is not None and config.find_peaks_axis=='reference'
    pattern = discover_elements(peaks, lines, auxiliary_lines, energy, counts, config, fixed_delta=0. if manual_reference else None)
    line_centers=constrain_line_centers(lines,auxiliary_lines,config,pattern.delta)
    initial_shift=manual_response['delta_E_keV'] if manual_reference else pattern.delta
    initial_alignment=dict(pattern.alignment)
    selected = set(all_symbols if config.element_selection == 'manual' else pattern.selected)
    selected.discard(config.tube_target)
    # Alignment is inferred only from a multi-region pattern. Respect an explicit
    # user bound; otherwise let that measured pattern initialize the global shift.
    if config.delta_max_kev is None:
        delta_bound = max(delta_bound, min(config.alignment_max_shift_kev, abs(initial_shift)+.04
                          +(config.energy_gain_bound*float(np.median(peaks.energy_keV)) if config.fit_energy_gain and len(peaks) else 0.)))
    initial_delta = float(np.clip(initial_shift, -delta_bound, delta_bound))
    comparisons = []
    for iteration in range(config.max_selection_iterations):
        fit_lines = [l for l in lines if l.symbol in selected]
        fit_auxiliary = [l for l in auxiliary_lines if l.symbol in selected]
        logging.info('Centroid-pattern candidates: %s; shared shift initial %.5f keV', sorted(selected), initial_delta)
        fit = fit_spectrum(counts, energy, edges, background, fit_lines, config, delta_bound,
                           auxiliary_lines=fit_auxiliary, initial_delta=initial_delta, line_centers=line_centers, center_axis=config.find_peaks_axis)
        if not fit.success or config.find_peaks is not None or config.element_selection == 'manual' or iteration == config.max_selection_iterations-1:
            break
        from .peak_detection import detect_peaks
        residual_peaks = detect_peaks(np.arange(len(counts)), energy, edges, counts, fit.model, config)
        residual_pattern = discover_elements(residual_peaks, lines, auxiliary_lines, energy, counts,
                                             config, fixed_delta=fit.delta, fixed_gain=fit.energy_gain)
        multi = set(residual_pattern.rankings.loc[residual_pattern.rankings.IndependentPeakRegions >= 2, 'Element'])
        additional = residual_pattern.selected.intersection(multi)-selected
        raw_pattern=discover_elements(peaks,lines,auxiliary_lines,energy,counts,config,fixed_delta=fit.delta,fixed_gain=fit.energy_gain)
        novel=set()
        for symbol in additional:
            ids=raw_pattern.pairings.loc[(raw_pattern.pairings.Element==symbol)&(raw_pattern.pairings.Group_Affinity>=.5),'peak_id']
            for peak_id in ids:
                peak=peaks.loc[peaks.peak_id==peak_id].iloc[0]
                center=peak.get('observed_energy_keV',peak.energy_keV)
                j=int(np.argmin(abs(energy-center)))
                if fit.components[j].sum() < .25*max(counts[j],1):
                    novel.add(symbol);break
        additional &= novel
        additional.discard(config.tube_target)
        if not additional:
            break
        logging.info('Residual multi-branch evidence adds: %s', sorted(additional))
        selected |= additional
        initial_delta = fit.delta
    # Recompute branch pairings with the final shared affine response, fixing
    # the old low-energy Ca mismatch caused by using only an initial offset.
    pattern=discover_elements(peaks,lines,auxiliary_lines,energy,counts,config,
                              fixed_delta=0. if manual_reference else fit.delta,
                              fixed_gain=0. if manual_reference else fit.energy_gain)
    cov, unc_status, unc_info = estimate_covariance(fit)
    rows, diag, overlap_pairs, branch_details = [], [], [], []
    weighted = fit.matrix[:, :len(fit.symbols)]/np.sqrt(fit.model[:, None])
    norms = np.linalg.norm(weighted, axis=0)
    cos = (weighted.T @ weighted)/np.maximum(norms[:, None]*norms[None, :], 1e-30)
    for i in range(len(fit.symbols)):
        for k in range(i+1, len(fit.symbols)):
            if cos[i, k] >= config.correlation_threshold:
                overlap_pairs.append({'elements': [fit.symbols[i], fit.symbols[k]], 'weighted_cosine': float(cos[i, k])})
    ranking = pattern.rankings.set_index('Element')
    for symbol in all_symbols:
        element_lines = [l for l in lines if l.symbol == symbol]
        z = element_lines[0].Z
        evidence = int(ranking.loc[symbol, 'IndependentPeakRegions'])
        matched_ids = ranking.loc[symbol, 'MatchedPeakIDs']
        pattern_score = float(ranking.loc[symbol, 'PatternScore'])
        penalty = float(ranking.loc[symbol, 'MissingLinePenalty'])
        if symbol not in fit.symbols:
            target = symbol == config.tube_target
            rows.append([z, symbol, 'ambiguous' if target else 'not_supported', pattern_score, None, len(element_lines),
                         evidence, '', target, 'target_scattering_confounded' if target else
                         'Not selected by global centroid/branch pattern; alternatives retained in peak_line_matches.csv; not evidence of absence',
                         matched_ids, '', None, None, None, 'not_estimable',
                         'not_estimable', pattern_score, penalty])
            continue
        i = fit.symbols.index(symbol)
        delta, score = None, None
        notes = []
        if fit.success:
            logging.info('Element deletion comparison: %s (%d/%d)', symbol, i+1, len(fit.symbols))
            aux_removed = [j for j, name in enumerate(fit.nuisance_names) if name.startswith(symbol+'_') and name.endswith('_unscaled')]
            if fit.t[i] <= 1e-5 and not any(fit.auxiliary_areas.get(fit.nuisance_names[j], 0) > 1e-5 for j in aux_removed):
                valid, delta, reduced_deviance = True, 0., fit.deviance
            else:
                removed = [i]+[len(fit.symbols)+j for j in aux_removed]
                reduced = fit_spectrum(counts, energy, edges, background,
                                       [l for l in fit_lines if l.symbol != symbol], config, delta_bound,
                                       initial=np.delete(fit.parameters, removed),
                                       auxiliary_lines=[l for l in fit_auxiliary if l.symbol != symbol],
                                       initial_delta=fit.delta, line_centers=line_centers, center_axis=config.find_peaks_axis)
                valid = reduced.success
                delta = max(0., 2*(reduced.nll-fit.nll)) if valid else None
                reduced_deviance = reduced.deviance
                if valid and reduced.nll < fit.nll-.1:
                    valid, delta = False, None
                    notes.append('Reduced model improves full fit; optimization instability')
            score = delta-(1+len(aux_removed))*np.log(len(counts)) if valid else None
            comparisons.append({'Element': symbol, 'reduced_fit_success': valid, 'DeltaDeviance': delta,
                                'reduced_deviance': reduced_deviance, 'full_deviance': fit.deviance,
                                'removed_parameters': 1+len(aux_removed),
                                'Score_definition': 'DeltaDeviance - removed_parameters*log(number_of_channels); ranking, not probability'})
        fitted_evidence, missing, diagnostics, mismatch = _regions(element_lines, fit, i, energy, np.asarray(counts), config)
        diag.extend(diagnostics)
        evidence = min(evidence, fitted_evidence)
        details=evaluate_branch_evidence(symbol,fit_lines,energy,edges,np.asarray(counts),fit,config) if config.branch_evidence else []
        wide=evaluate_branch_evidence(symbol,fit_lines,energy,edges,np.asarray(counts),fit,config,roi_sigma=6.) if details else []
        for item in details:
            alt=next((d for d in wide if d['Branches']==item['Branches']),None)
            item['Wide_Window_Branch_DeltaDeviance']=alt['Branch_DeltaDeviance'] if alt else None
            item['Robust_Group_Observed']=bool(item['Group_Observed'] and alt and alt['Group_Observed'])
            item['Ratio_Compatible']=bool(item['Tied_vs_Free_DeltaDeviance']<9 and alt and alt['Tied_vs_Free_DeltaDeviance']<9)
        branch_details.extend(details)
        observed_groups=sum(d['Robust_Group_Observed'] for d in details)
        robust_local=observed_groups>=2 and all(d['Ratio_Compatible'] for d in details)
        evidence=max(evidence,observed_groups)
        overlap = sorted({l.symbol for a in element_lines for l in fit_lines
                          if l.symbol != symbol and abs(a.energy_keV-l.energy_keV) <
                          FWHM_FACTOR*float(sigma_kev(a.energy_keV, config))*fit.resolution_scale})
        std = np.sqrt(cov[i, i]) if np.isfinite(cov[i, i]) else None
        if not fit.success or delta is None:
            status = 'not_supported'
            notes.append('fit_failed_or_comparison_failed')
        elif delta >= config.support_delta_deviance and evidence >= 2 and (pattern_score > 0 or robust_local):
            status = 'supported'
        elif delta >= config.tentative_delta_deviance or (fit.t[i] > 0 and evidence >= 1):
            status = 'tentative'
        else:
            status = 'not_supported'
        quantitative = 'supported' if status == 'supported' and unc_status[i] == 'fisher_interior' else 'not_estimable'
        if mismatch:
            notes.append('Identified energies retained; matrix_effect_or_model_mismatch; branch_ratio_mismatch')
            quantitative = 'branch_ratio_mismatch'
        if unc_status[i] == 'not_estimable' and status in ('supported', 'tentative'):
            status = 'unidentifiable'
            quantitative = 'not_estimable'
        if evidence < 2:
            notes.append('Fewer than two independent observed centroid regions')
        notes.append('Centroid/branch-pattern identification precedes fitting; fixed quantitative S with same-shell completion; scores are not probabilities')
        rows.append([z, symbol, status, score, delta, len(element_lines), evidence, '+'.join(missing), bool(overlap),
                     '; '.join(notes), matched_ids, ','.join(overlap), float(fit.q[i]) if fit.success else None,
                     std, None, unc_status[i], quantitative, pattern_score, penalty])
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    candidates['Element_Class']=candidates.Element.map(lambda symbol:element_class(symbol,config.element_prior))
    candidates['Preferred_By_Prior']=False
    candidates['Prior_Alternatives']=''
    # Check mathematically indistinguishable unselected alternatives as well:
    # choosing a common element does not make a data ambiguity disappear.
    for symbol in fit.symbols:
        for other in all_symbols:
            if other in fit.symbols or not ranking.loc[other,'IndependentPeakRegions']:
                continue
            if abs(float(ranking.loc[symbol,'PatternScore'])-float(ranking.loc[other,'PatternScore'])) > config.prior_tie_margin:
                continue
            vector=np.zeros(len(counts))
            for line in lines:
                if line.symbol==other:
                    vector+=line.sensitivity_cm2_g*gaussian_response(edges,line_model_center(line,fit),float(sigma_kev(line.energy_keV,config))*fit.resolution_scale)
            vector/=np.sqrt(fit.model)
            own=weighted[:,fit.symbols.index(symbol)]
            similarity=float(own@vector/max(np.linalg.norm(own)*np.linalg.norm(vector),1e-30))
            if similarity>=config.correlation_threshold:
                overlap_pairs.append({'elements':[symbol,other],'weighted_cosine':similarity,'unfitted_alternative':True})
    for pair in overlap_pairs:
        a, b = pair['elements']
        ids = candidates.Element.isin([a, b])
        indices = [fit.symbols.index(s) for s in [a, b] if s in fit.symbols]
        if sum(fit.t[indices]) > 1:
            candidates.loc[ids, 'Status'] = 'unidentifiable'
            preferred=preferred_in_group([a,b],config.element_prior)
            if preferred:
                candidates.loc[candidates.Element==preferred,'Preferred_By_Prior']=True
                candidates.loc[ids,'Prior_Alternatives']=a+','+b
                candidates.loc[ids,'ConfidenceNotes'] += '; common-element preference='+preferred+'; ambiguity retained; excluded from quantification'
            candidates.loc[ids, 'Quantification_Status'] = 'not_estimable'
            candidates.loc[ids, 'ConfidenceNotes'] += '; highly_collinear_templates: '+a+'/'+b
            candidates.loc[ids, 'q_Std'] = np.nan
            candidates.loc[ids, 'Uncertainty_Status'] = 'not_estimable'
            for i in indices:
                cov[i, :], cov[:, i], unc_status[i] = np.nan, np.nan, 'not_estimable'
    # A context preference can order tentative interpretations of the same
    # observed feature. It never promotes them to supported or quantitative.
    if config.element_prior!='none' and not pattern.pairings.empty:
        for _,group in pattern.pairings.groupby('peak_id'):
            symbols=set(group.loc[group.Group_Affinity>=.5,'Element'])
            active=candidates.loc[candidates.Element.isin(symbols) & candidates.Status.isin(['tentative','ambiguous','unidentifiable'])]
            preferred=preferred_in_group(active.Element.tolist(),config.element_prior)
            if preferred and len(active)>1 and (active.Element_Class=='uncommon').any():
                affinities=group.groupby('Element').Group_Affinity.max()
                if affinities[preferred]>=affinities.loc[active.Element].max()-min(.2,config.prior_tie_margin):
                    candidates.loc[candidates.Element==preferred,'Preferred_By_Prior']=True
                    candidates.loc[candidates.Element==preferred,'Prior_Alternatives']=','.join(sorted(active.Element))
                    candidates.loc[candidates.Element==preferred,'ConfidenceNotes'] += '; preferred common interpretation among unresolved candidates; not confirmation'
    if not pattern.pairings.empty:
        pattern.pairings['Selected_Element'] = pattern.pairings.Element.isin(fit.symbols)
        pattern.pairings['Preferred_By_Prior']=pattern.pairings.Element.map(candidates.set_index('Element').Preferred_By_Prior)
        pattern.pairings['Energy_Axis']=config.find_peaks_axis if config.find_peaks is not None else 'observed'
        pattern.pairings['Element_Status'] = pattern.pairings.Element.map(candidates.set_index('Element').Status)
        pattern.pairings['Model_Centroid_keV']=[line_centers.get((r.Element,r.Branch),r.Reference_Energy_keV) if config.find_peaks_axis=='observed' and (r.Element,r.Branch) in line_centers else line_centers.get((r.Element,r.Branch),r.Reference_Energy_keV)*(1+fit.energy_gain)+fit.delta for _,r in pattern.pairings.iterrows()]
        pattern.pairings['Model_Energy_Residual_keV'] = pattern.pairings.Observed_Centroid_keV-pattern.pairings.Model_Centroid_keV
        pattern.pairings['Group_Energy_Residual_keV'] = pattern.pairings.Observed_Centroid_keV-pattern.pairings.Group_Centroid_keV-pattern.delta
    pattern.rankings['Fitted'] = pattern.rankings.Element.isin(fit.symbols)
    pattern.alignment['effective_fit_delta_bound_keV'] = delta_bound
    pattern.alignment['initial_alignment']=initial_alignment
    return IdentificationResult(fit, candidates, cov, unc_status, unc_info, fit_lines, comparisons,
                                overlap_pairs, diag, pattern.pairings, pattern.rankings, pattern.alignment, branch_details)
