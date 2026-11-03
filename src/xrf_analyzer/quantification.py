import numpy as np
import pandas as pd
from .uncertainty import ratio_interval

DENSITY_COLUMNS = ['Z', 'Element', 'Element_Status', 'q_Fit', 'Relative_to_Reference',
                   'Fraction_among_Reported_Elements', 'Uncertainty_percent', 'CI95_Low', 'CI95_High',
                   'Reference_Element', 'Apparent_Only', 'Quality_Flags', 'q_Std', 'Upper95_q', 'Quantification_Status',
                   'q_Area', 'Area_Counts', 'Sensitivity_Sum', 'Selected_Branches', 'Quantification_Method',
                   'Global_Quantification_Status', 'q_Stat_Std', 'Quantification_Notes', 'Measurement_Failure_Reason']


def compute_relative_areal_density(identification, reference_element=None, local_quantification=None):
    if local_quantification is not None:
        return _area_density(identification, local_quantification, reference_element)
    fit, candidates = identification.fit, identification.candidates
    usable = candidates[(candidates.Status == 'supported') & (candidates.Quantification_Status == 'supported') & (candidates.q_Fit > 0)
                        & candidates.q_Std.notna()] if fit.success else candidates.iloc[:0]
    supported = usable.Element.tolist()
    reference_note = 'requested reference' if reference_element else 'minimum relative q standard error among supported identifiable elements'
    if reference_element is None and supported:
        reference_element = min(supported, key=lambda s: float(usable.loc[usable.Element == s, 'q_Std'].iloc[0]) / fit.q[fit.symbols.index(s)])
    valid_ref = reference_element in supported
    if not valid_ref:
        reference_note = 'No reliable supported reference; requested reference is not silently replaced'
    denominator = sum(fit.q[fit.symbols.index(s)] for s in supported)
    rows = []
    for _, c in candidates.iterrows():
        if c.Element not in fit.symbols or c.Status != 'supported':
            continue
        ratio = uncertainty = low = high = fraction = None
        if c.Element in supported:
            i = fit.symbols.index(c.Element)
            fraction = fit.q[i]/denominator
            if valid_ref:
                ratio, uncertainty, low, high = ratio_interval(fit.q, identification.qcov, i, fit.symbols.index(reference_element))
        rows.append([c.Z, c.Element, c.Status, c.q_Fit, ratio, fraction, uncertainty, low, high,
                     reference_element if valid_ref else None, True,
                     'apparent_relative_areal_density; statistical_only; fixed_branch_ratios; '
                     + ('reported_elements_only' if c.Element in supported else 'excluded_from_fraction_denominator; upper_limit_not_estimable_without_profile_or_bootstrap'),
                     c.q_Std, c.Upper95_q, c.Quantification_Status,
                     None, None, None, None, 'global-template', c.Quantification_Status,
                     c.q_Std, 'Global fixed-branch template; q_Fit is the quantitative scale', None])
    return pd.DataFrame(rows, columns=DENSITY_COLUMNS), {'Element': reference_element if valid_ref else None,
                                                       'selection_rule': reference_note,
                                                       'fraction_denominator_elements': supported}


def _area_density(identification, local, reference_element):
    """Use independently measured net areas, never overwrite the joint-fit q.

    A_i = common_flux_geometry_time * rho_i * S_i. Thus the unknown common
    factor cancels in (A_i/S_i)/(A_ref/S_ref). This is conditional on the supplied
    sensitivities and the apparent thin-sample approximation.
    """
    estimates = local['elements']
    order = estimates.Element.tolist()
    lookup = estimates.set_index('Element')
    covariance = np.asarray(local['covariance'], dtype=float)
    q = estimates.q_Area.to_numpy(dtype=float)
    supported = []
    candidates = identification.candidates
    for candidate in candidates.itertuples(index=False):
        if candidate.Status != 'supported' or candidate.Element not in lookup.index:
            continue
        row = lookup.loc[candidate.Element]
        if (row.Quantification_Status in ('supported','local_area_estimate') and
                np.isfinite(row.q_Area) and row.q_Area > 0 and
                np.isfinite(row.q_Std) and row.q_Std > 0):
            supported.append(candidate.Element)
    if reference_element is None and supported:
        reference_element = min(supported, key=lambda s: lookup.loc[s,'q_Std']/lookup.loc[s,'q_Area'])
        rule = 'minimum relative primary-area uncertainty among supported measurable elements'
    else:
        rule = 'requested reference' if reference_element else 'no measurable supported reference'
    valid_reference = reference_element in supported
    if not valid_reference:
        rule = 'No reliable supported reference; requested reference is not silently replaced'
    denominator = sum(float(lookup.loc[s,'q_Area']) for s in supported)
    rows = []
    for candidate in candidates.itertuples(index=False):
        if candidate.Status != 'supported' or candidate.Element not in identification.fit.symbols:
            continue
        estimate = lookup.loc[candidate.Element] if candidate.Element in lookup.index else None
        measurable = candidate.Element in supported
        ratio = uncertainty = low = high = fraction = None
        if measurable:
            fraction = float(estimate.q_Area)/denominator
            if valid_reference:
                ratio, uncertainty, low, high = ratio_interval(q, covariance, order.index(candidate.Element), order.index(reference_element))
        def value(name, default=None):
            return estimate.get(name, default) if estimate is not None else default
        status = 'supported' if measurable else value('Quantification_Status', 'local_area_not_estimable')
        flags = ('apparent_relative_areal_density; supplied_sensitivity_assumed; '
                 'local_primary_area; local_response_and_window_uncertainty; '
                 + ('reported_elements_only' if measurable else 'excluded_from_fraction_denominator'))
        boundary=value('Local_Response_At_Boundary',False)
        if pd.notna(boundary) and boundary:
            flags += '; local_response_at_boundary; approximate_uncertainty'
        if value('Background_Window_Relative_Difference',0.)>.2:
            flags += '; background_window_sensitive'
        rows.append([candidate.Z, candidate.Element, candidate.Status, candidate.q_Fit,
                     ratio, fraction, uncertainty, low, high, reference_element if valid_reference else None,
                     True, flags, value('q_Std'), None, status, value('q_Area'), value('Area_Counts'),
                     value('Sensitivity_Sum'), value('Selected_Branches'), 'local-peaks',
                     getattr(candidate, 'Global_Quantification_Status', candidate.Quantification_Status),
                     value('q_Stat_Std'), value('Notes', 'Conditional primary-peak estimate; not an absolute mass density'),
                     value('Failure_Reason')])
    return pd.DataFrame(rows, columns=DENSITY_COLUMNS), {
        'Element': reference_element if valid_reference else None, 'selection_rule': rule,
        'fraction_denominator_elements': supported, 'formula': '(net_area_i / sensitivity_i) / (net_area_reference / sensitivity_reference)',
        'scale_source': 'local_primary_peak_area; joint_fit_q_is_diagnostic'}
