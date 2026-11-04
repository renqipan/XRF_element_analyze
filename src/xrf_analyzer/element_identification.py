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
from .low_z import assess_low_z_kalpha

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
    low_z_evidence: object = None


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
        variance=fit.model[mask].copy()
        if getattr(fit,'background_latent_counts',None) is not None:
            variance+=(fit.background_scale or 0.)**2*fit.background_latent_counts[mask]
        info=float(np.sum(component**2/variance))
        expected=float(component.sum())
        residual_z=float(np.sum(counts[mask]-fit.model[mask])/np.sqrt(variance.sum()+1))
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
    disabled_auxiliary=set()
    auxiliary_pruning=[]
    for iteration in range(config.max_selection_iterations):
        fit_lines = [l for l in lines if l.symbol in selected]
        fit_auxiliary = [l for l in auxiliary_lines if l.symbol in selected and l.symbol not in disabled_auxiliary]
        logging.info('Centroid-pattern candidates: %s; shared shift initial %.5f keV', sorted(selected), initial_delta)
        fit = fit_spectrum(counts, energy, edges, background, fit_lines, config, delta_bound,
                           auxiliary_lines=fit_auxiliary, initial_delta=initial_delta, line_centers=line_centers, center_axis=config.find_peaks_axis)
        if config.prune_unanchored_auxiliary:
            for _ in range(3):
                if not fit.success:break
                orphan=[]
                for symbol in {l.symbol for l in fit_auxiliary}:
                    j=fit.symbols.index(symbol)
                    primary=fit.matrix[:,j]*fit.t[j]
                    information=float(np.sqrt(np.sum(primary**2/fit.model)))
                    anchor_pairs=pattern.pairings.loc[(pattern.pairings.Element==symbol)
                        & pattern.pairings.Branch.isin(['Kalpha1','Kalpha2','Lalpha1','Lalpha2','Malpha1'])
                        & (pattern.pairings.Group_Affinity>=.5)
                        & (pattern.pairings.Strength_Source!='unscaled_auxiliary_shell')]
                    anchor_peaks=peaks.loc[peaks.peak_id.isin(anchor_pairs.peak_id)]
                    anchored=bool((anchor_peaks.approx_significance>=np.sqrt(config.support_delta_deviance)).any())
                    if information<max(config.candidate_threshold_sigma,np.sqrt(config.support_delta_deviance)) or not anchored:
                        orphan.append(symbol)
                        auxiliary_pruning.append({'Element':symbol,'Primary_Information_Score':information,
                            'Required_Primary_Information_Score':max(config.candidate_threshold_sigma,np.sqrt(config.support_delta_deviance)),
                            'Principal_Centroid_Anchored':anchored,
                            'Reason':'Independent auxiliary shells require detectable primary anchors'})
                if not orphan:break
                disabled_auxiliary.update(orphan)
                removed=[len(fit.symbols)+j for j,name in enumerate(fit.nuisance_names)
                         if any(name.startswith(symbol+'_') and name.endswith('_unscaled') for symbol in orphan)]
                fit_auxiliary=[l for l in fit_auxiliary if l.symbol not in disabled_auxiliary]
                logging.info('Disable unanchored auxiliary shells: %s',sorted(orphan))
                fit=fit_spectrum(counts,energy,edges,background,fit_lines,config,delta_bound,
                     initial=np.delete(fit.parameters,removed),auxiliary_lines=fit_auxiliary,
                     initial_delta=fit.delta,line_centers=line_centers,center_axis=config.find_peaks_axis)
        if not fit.success or config.find_peaks is not None or config.element_selection == 'manual' or iteration == config.max_selection_iterations-1:
            break
        from .peak_detection import detect_peaks
        residual_peaks = detect_peaks(np.arange(len(counts)), energy, edges, counts, fit.model, config)
        active_auxiliary = [l for l in auxiliary_lines if l.symbol not in disabled_auxiliary]
        residual_pattern = discover_elements(residual_peaks, lines, active_auxiliary, energy, counts,
                                             config, fixed_delta=fit.delta, fixed_gain=fit.energy_gain)
        multi = set(residual_pattern.rankings.loc[residual_pattern.rankings.IndependentPeakRegions >= 2, 'Element'])
        additional = residual_pattern.selected.intersection(multi)-selected
        raw_pattern=discover_elements(peaks,lines,active_auxiliary,energy,counts,config,fixed_delta=fit.delta,fixed_gain=fit.energy_gain)
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
    active_auxiliary = [l for l in auxiliary_lines if l.symbol not in disabled_auxiliary]
    pattern=discover_elements(peaks,lines,active_auxiliary,energy,counts,config,
                              fixed_delta=0. if manual_reference else fit.delta,
                              fixed_gain=0. if manual_reference else fit.energy_gain)
    cov, unc_status, unc_info = estimate_covariance(fit)
    unc_info['auxiliary_shell_pruning']=auxiliary_pruning
    rows, diag, overlap_pairs, branch_details, low_z_details = [], [], [], [], []
    weighted = fit.matrix[:, :len(fit.symbols)]/np.sqrt(fit.model[:, None])
    norms = np.linalg.norm(weighted, axis=0)
    cos = (weighted.T @ weighted)/np.maximum(norms[:, None]*norms[None, :], 1e-30)
    for i in range(len(fit.symbols)):
        for k in range(i+1, len(fit.symbols)):
            if cos[i, k] >= config.correlation_threshold:
                overlap_pairs.append({'elements': [fit.symbols[i], fit.symbols[k]], 'weighted_cosine': float(cos[i, k])})
    ranking = pattern.rankings.set_index('Element')
    # Local nuisance groups may be independent only with an actual multi-region
    # primary pattern. A high fitted amplitude alone can be borrowed from Fe.
    config._branch_confirmed_symbols=set(pattern.rankings.loc[
        (pattern.rankings.IndependentPeakRegions>=2) & (pattern.rankings.PatternScore>0),'Element'])
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
            item['Within_Shell_Ratio_Compatible']=bool(
                item.get('Within_Initial_Shell_DeltaDeviance') is not None
                and item['Within_Initial_Shell_DeltaDeviance']<9 and alt
                and alt.get('Within_Initial_Shell_DeltaDeviance') is not None
                and alt['Within_Initial_Shell_DeltaDeviance']<9)
        branch_details.extend(details)
        observed_groups=sum(d['Robust_Group_Observed'] for d in details)
        # Identity uses observed energies; a cross-shell intensity mismatch
        # belongs to quantification/model quality, not an energy-evidence veto.
        robust_local=observed_groups>=2 and all(d['Within_Shell_Ratio_Compatible'] for d in details)
        evidence=min(int(ranking.loc[symbol,'IndependentPeakRegions']),fitted_evidence)
        overlap = sorted({l.symbol for a in element_lines for l in fit_lines
                          if l.symbol != symbol and abs(a.energy_keV-l.energy_keV) <
                          FWHM_FACTOR*float(sigma_kev(a.energy_keV, config))*fit.resolution_scale})
        primary_pairs=pattern.pairings.loc[pattern.pairings.Strength_Source!='unscaled_auxiliary_shell']
        principal = [l for l in element_lines if l.branch in ('Kalpha1','Kalpha2','Lalpha1','Lalpha2','Malpha1')
                     and energy[0] <= line_model_center(l,fit) <= energy[-1]]
        if not principal:
            principal = [max(element_lines,key=lambda l:l.sensitivity_cm2_g)]
        principal_pairs=primary_pairs.loc[(primary_pairs.Element==symbol)
            & primary_pairs.Branch.isin([l.branch for l in principal]) & (primary_pairs.Group_Affinity>=.5)]
        principal_observed=bool(principal_pairs.peak_id.isin(peaks.loc[
            ~peaks.quality_flags.str.contains('broad_structure'),'peak_id']).any())
        # A blended beta can be measured without its own local maximum.
        # Count robust local groups only in a family with an observed alpha;
        # do not join a weak Kalpha to an unrelated fitted L-shell residual.
        anchored_families={branch[0] for branch in principal_pairs.Branch}
        local_family_evidence=max([sum(d['Robust_Group_Observed'] for d in details if d['Family']==family)
                                   for family in anchored_families]+[0])
        local_family_available=any(d['Family'] in anchored_families for d in details)
        severe_template_mismatch=fit.deviance/max(len(counts)-fit.n_parameters,1)>config.max_deviance_per_dof
        if severe_template_mismatch and local_family_available:
            evidence=local_family_evidence
        else:
            evidence=max(evidence,local_family_evidence)
        low_z=assess_low_z_kalpha(symbol,fit_lines,peaks,primary_pairs,ranking,
                                  np.asarray(counts),energy,edges,fit,config,details)
        if low_z['Applicable']:low_z_details.append(low_z)
        std = np.sqrt(cov[i, i]) if np.isfinite(cov[i, i]) else None
        if not fit.success or delta is None:
            status = 'not_supported'
            notes.append('fit_failed_or_comparison_failed')
        elif low_z['Eligible'] and delta >= config.tentative_delta_deviance:
            status='supported'
            notes.append('low_Z_Kalpha_dominant_support; independent Kbeta not required; single-region evidence retained; robust local evidence controls identity')
        elif (delta >= config.support_delta_deviance and evidence >= 2 and principal_observed
              and (pattern_score > 0 or robust_local)):
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
        if not principal_observed:
            notes.append('No precise observed principal alpha group; minor or auxiliary branches cannot establish identity alone')
        if evidence < 2 and not low_z['Eligible']:
            notes.append('Fewer than two independent observed centroid regions')
        notes.append('Centroid/branch-pattern identification precedes fitting; fixed quantitative S with same-shell completion; scores are not probabilities')
        rows.append([z, symbol, status, score, delta, len(element_lines), evidence, '+'.join(missing), bool(overlap),
                     '; '.join(notes), matched_ids, ','.join(overlap), float(fit.q[i]) if fit.success else None,
                     std, None, unc_status[i], quantitative, pattern_score, penalty])
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    candidates['Element_Class']=candidates.Element.map(lambda symbol:element_class(symbol,config.element_prior))
    candidates['Preferred_By_Prior']=False
    candidates['Prior_Alternatives']=''
    # A suppressed fitted amplitude is conditional on all competitors being present.
    # Recheck a suppressed low-Z Kalpha against an unconfirmed overlapping
    # competitor, including one with a failed identity but nonzero amplitude. This can recover a *possible* interpretation only:
    # the original full model, q, covariance and quantitative status stay intact.
    counterfactual_reviews = []
    if fit.success and config.low_z_kalpha:
        reviews = 0
        for row_index, candidate in candidates.iterrows():
            if (reviews >= 3 or candidate.Z > config.low_z_max_z or candidate.Status != 'not_supported'
                    or candidate.Element not in fit.symbols):
                continue
            symbol = candidate.Element
            alpha_pairs=primary_pairs.loc[(primary_pairs.Element==symbol)
                & primary_pairs.Branch.isin(['Kalpha1','Kalpha2']) & (primary_pairs.Group_Affinity>=.5)]
            if alpha_pairs.empty:
                continue
            peak_ids=set(alpha_pairs.peak_id)
            rivals=candidates.loc[candidates.Element.isin(str(candidate.OverlapElements).split(','))
                & candidates.Status.isin(['tentative','ambiguous','unidentifiable','not_supported'])]
            for _, rival_row in rivals.iterrows():
                if reviews >= 3:
                    break
                rival=rival_row.Element
                if rival not in fit.symbols or fit.t[fit.symbols.index(rival)] <= 1e-5:
                    continue
                rival_pairs=primary_pairs.loc[(primary_pairs.Element==rival)
                    & primary_pairs.peak_id.isin(peak_ids) & (primary_pairs.Group_Affinity>=.5)]
                if rival_pairs.empty:
                    continue
                details=[d for d in branch_details if d['Element']==rival]
                contradicted={d['Family'] for d in details if d['Tied_vs_Free_DeltaDeviance']>=9}
                # An unconfirmed rival need not have contradictory ratios to
                # suppress a valid alternative: its branch may be unresolvable
                # under a stronger element once interference is profiled.
                wide=evaluate_branch_evidence(rival,fit_lines,energy,edges,np.asarray(counts),fit,config,roi_sigma=6.)
                ratio_contradicted=any(d['Family'] in contradicted and d['Tied_vs_Free_DeltaDeviance']>=9 for d in wide)
                reviews += 1
                removed=[fit.symbols.index(rival)]+[len(fit.symbols)+j for j,name in enumerate(fit.nuisance_names)
                    if name.startswith(rival+'_') and name.endswith('_unscaled')]
                cf_lines=[l for l in fit_lines if l.symbol!=rival]
                cf_auxiliary=[l for l in fit_auxiliary if l.symbol!=rival]
                cf_fit=fit_spectrum(counts,energy,edges,background,cf_lines,config,delta_bound,
                    initial=np.delete(fit.parameters,removed),auxiliary_lines=cf_auxiliary,
                    initial_delta=fit.delta,line_centers=line_centers,center_axis=config.find_peaks_axis)
                audit={'Element':symbol,'Removed_Rival':rival,'Method':'leave_unconfirmed_rival_out; tentative_only',
                       'Counterfactual_Fit_Success':cf_fit.success,'Recovered_Possible_Interpretation':False,
                       'Rival_Ratio_Contradicted_In_Both_Windows':bool(ratio_contradicted),
                       'Full_Model_and_Quantification_Unchanged':True}
                if cf_fit.success:
                    cf_pattern=discover_elements(peaks,lines,[l for l in active_auxiliary if l.symbol!=rival],
                        energy,counts,config,fixed_delta=0. if manual_reference else cf_fit.delta,
                        fixed_gain=0. if manual_reference else cf_fit.energy_gain)
                    cf_pairs=cf_pattern.pairings.loc[cf_pattern.pairings.Strength_Source!='unscaled_auxiliary_shell']
                    cf_branches=evaluate_branch_evidence(symbol,cf_lines,energy,edges,np.asarray(counts),cf_fit,config)
                    cf_alpha=assess_low_z_kalpha(symbol,cf_lines,peaks,cf_pairs,cf_pattern.rankings.set_index('Element'),
                        np.asarray(counts),energy,edges,cf_fit,config,cf_branches)
                    cf_removed=[cf_fit.symbols.index(symbol)]+[len(cf_fit.symbols)+j for j,name in enumerate(cf_fit.nuisance_names)
                        if name.startswith(symbol+'_') and name.endswith('_unscaled')]
                    cf_null=fit_spectrum(counts,energy,edges,background,[l for l in cf_lines if l.symbol!=symbol],
                        config,delta_bound,initial=np.delete(cf_fit.parameters,cf_removed),
                        auxiliary_lines=[l for l in cf_auxiliary if l.symbol!=symbol],initial_delta=cf_fit.delta,
                        line_centers=line_centers,center_axis=config.find_peaks_axis)
                    delta=max(0.,2*(cf_null.nll-cf_fit.nll)) if cf_null.success and cf_null.nll>=cf_fit.nll-.1 else None
                    recovered=bool(cf_alpha['Eligible'] and delta is not None and delta>=config.support_delta_deviance)
                    audit.update(Counterfactual_Kalpha_Evidence=cf_alpha,Counterfactual_Element_DeltaDeviance=delta,
                                 Recovered_Possible_Interpretation=recovered)
                    if recovered:
                        candidates.loc[row_index,'Status']='tentative'
                        candidates.loc[row_index,'Quantification_Status']='not_estimable'
                        candidates.loc[row_index,'ConfidenceNotes'] += ('; Kalpha recovered only when unconfirmed overlapping '
                            +rival+' is removed; rival lacks independently confirmed identity; '
                            'counterfactual interpretation remains tentative; full-model q unchanged')
                counterfactual_reviews.append(audit)
                if audit['Recovered_Possible_Interpretation']:
                    break
    unc_info['low_z_counterfactual_reviews']=counterfactual_reviews
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
        pattern.pairings['Group_Energy_Residual_keV'] = pattern.pairings.Observed_Centroid_keV-pattern.pairings.Group_Centroid_keV*(1+pattern.alignment.get('gain_for_matching',0.))-pattern.delta
    pattern.rankings['Fitted'] = pattern.rankings.Element.isin(fit.symbols)
    pattern.alignment['effective_fit_delta_bound_keV'] = delta_bound
    pattern.alignment['initial_alignment']=initial_alignment
    return IdentificationResult(fit, candidates, cov, unc_status, unc_info, fit_lines, comparisons,
                                overlap_pairs, diag, pattern.pairings, pattern.rankings, pattern.alignment, branch_details, low_z_details)
