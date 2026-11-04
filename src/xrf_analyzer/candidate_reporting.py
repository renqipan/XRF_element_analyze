"""Conservative pending-list presentation, independent of fitting and density.

The evidence sources are different conditional tests, not calibrated p-values.
An unresolved shared peak may support alternative interpretations, never a
quantitative identity. No candidate status or fitted parameter is modified.
"""
import numpy as np

from .response import FWHM_FACTOR, sigma_kev, line_model_center


PRIMARY = {'Kalpha1', 'Kalpha2', 'Lalpha1', 'Lalpha2', 'Malpha1'}
PENDING = {'tentative', 'ambiguous', 'unidentifiable'}


def filter_possible_candidates(candidates, identification, peaks, config):
    result = candidates.copy()
    floor = config.possible_min_delta_deviance
    result['Possible_Reporting_Evidence'] = np.nan
    result['Possible_Reporting_Evidence_Source'] = ''
    result['Principal_Peak_Observed_For_Possibility'] = False
    if floor == 0:
        return result
    fit = identification.fit
    usable = peaks.loc[~peaks.quality_flags.str.contains('broad_structure')].copy()
    observed = usable.get('observed_energy_keV', usable.energy_keV).to_numpy()
    if (config.find_peaks is not None and config.find_peaks_axis == 'reference'
            and 'observed_energy_keV' not in usable):
        observed = observed*(1+fit.energy_gain)+fit.delta
    scores = usable.approx_significance.to_numpy()
    for index, row in result.iterrows():
        if (row.Status not in PENDING or not row.Reportable or row.q_Fit is None
                or not np.isfinite(row.q_Fit)):
            continue
        primary = [l for l in identification.fit_lines if l.symbol == row.Element and l.branch in PRIMARY]
        matched = np.zeros(len(usable), dtype=bool)
        for line in primary:
            radius = .5*FWHM_FACTOR*float(sigma_kev(line.energy_keV, config))*fit.resolution_scale
            matched |= abs(observed-line_model_center(line, fit)) <= radius
        anchored = bool(np.any(matched & (scores >= config.candidate_threshold_sigma)))
        evidence = []
        if row.DeltaDeviance is not None and np.isfinite(row.DeltaDeviance):
            evidence.append((float(row.DeltaDeviance), 'global_element_deletion'))
        for item in identification.branch_evidence or []:
            if item['Element'] != row.Element or 'alpha' not in item['Branches']:
                continue
            values = [item.get('Branch_DeltaDeviance'), item.get('Wide_Window_Branch_DeltaDeviance')]
            if all(v is not None and np.isfinite(v) for v in values):
                evidence.append((min(values), 'principal_group_both_local_windows'))
        for review in identification.uncertainty_info.get('low_z_counterfactual_reviews', []):
            if review['Element'] != row.Element or not review.get('Recovered_Possible_Interpretation'):
                continue
            alpha = review['Counterfactual_Kalpha_Evidence']
            values = [review['Counterfactual_Element_DeltaDeviance'],
                      alpha.get('Kalpha_DeltaDeviance'), alpha.get('Wide_Kalpha_DeltaDeviance')]
            if all(v is not None and np.isfinite(v) for v in values):
                evidence.append((min(values), 'counterfactual_conditional_interpretation_only'))
        # Exactly collinear templates may each have zero deletion gain. A
        # strong shared *observed* primary still warrants an unresolved group.
        collinear = any(row.Element in pair['elements'] and pair.get('weighted_cosine', 0.) >= config.correlation_threshold
                        for pair in identification.overlap_pairs)
        if row.Status == 'unidentifiable' and collinear and anchored:
            evidence.append((float(np.max(scores[matched])**2), 'observed_primary_of_unresolved_group'))
        gain, source = max(evidence, key=lambda v: v[0]) if evidence else (0., 'no_valid_evidence')
        result.loc[index, 'Possible_Reporting_Evidence'] = gain
        result.loc[index, 'Possible_Reporting_Evidence_Source'] = source
        result.loc[index, 'Principal_Peak_Observed_For_Possibility'] = anchored
        if not anchored or gain < floor:
            result.loc[index, 'Reportable'] = False
            result.loc[index, 'Reporting_Reason'] = 'possible_requires_observed_principal_peak_and_evidence_floor; hypothesis_retained_in_diagnostics'
    return result
