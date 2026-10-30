import numpy as np
import pandas as pd
from .uncertainty import ratio_interval

DENSITY_COLUMNS = ['Z', 'Element', 'Element_Status', 'q_Fit', 'Relative_to_Reference',
                   'Fraction_among_Reported_Elements', 'Uncertainty_percent', 'CI95_Low', 'CI95_High',
                   'Reference_Element', 'Apparent_Only', 'Quality_Flags', 'q_Std', 'Upper95_q', 'Quantification_Status']


def compute_relative_areal_density(identification, reference_element=None):
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
                     c.q_Std, c.Upper95_q, c.Quantification_Status])
    return pd.DataFrame(rows, columns=DENSITY_COLUMNS), {'Element': reference_element if valid_ref else None,
                                                       'selection_rule': reference_note,
                                                       'fraction_denominator_elements': supported}
