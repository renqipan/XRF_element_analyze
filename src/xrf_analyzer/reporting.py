from pathlib import Path
import json
import hashlib
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr, xlogy
from .response import sigma_kev, FWHM_FACTOR, line_model_center

LINE_COLUMNS = ['Z', 'Element', 'Branch', 'Reference_Energy_keV', 'Centroid_keV', 'FWHM_keV',
                'Net_Area', 'Gross_Area', 'Uncertainty_of_Net_Area_percent', 'Net_Area_in_ROI',
                'ROI_Start_keV', 'ROI_End_keV', 'Area_Std_Counts', 'Element_Status', 'Overlap_Flag',
                'Uncertainty_Status', 'Notes', 'Sensitivity_cm2_g', 'ROI_Truncated', 'Fit_Status', 'Strength_Source', 'Quantification_Status', 'Specified_Center_keV', 'Center_Source']
ANALYSIS_OUTPUT_FILES = ('calibration_result.json', 'calibration.png', 'detected_peaks.csv',
    'element_candidates.csv', 'fitted_lines.csv', 'rejected_candidate_lines.csv', 'fitted_groups.csv',
    'relative_areal_density.csv', 'fit_spectrum.csv', 'fit_components.csv', 'fit_summary.json',
    'spectrum_fit.png', 'residuals.png', 'branch_diagnostics.csv', 'unassigned_peaks.csv',
    'sensitivity_states.csv', 'background_initial.csv', 'peak_line_matches.csv', 'peak_assignments.csv',
    'element_pattern_scores.csv', 'auxiliary_fitted_lines.csv', 'elements_need_confirmation.csv',
    'element_diagnostics.csv','element_classes.csv','branch_ratio_evidence.csv','calcium_analysis.json','calcium_region.png')

GROUP_COLUMNS = ['Z', 'Element', 'Branches', 'Centroid_keV', 'Moment_Equivalent_FWHM_keV',
                 'Net_Area', 'Area_Std_Counts', 'Notes']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def write_json(path, obj):
    Path(path).write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def fitted_line_tables(ident, counts, edges, config):
    fit = ident.fit
    rows, rejected = [], []
    if not fit.success:
        return pd.DataFrame(columns=LINE_COLUMNS), pd.DataFrame(columns=LINE_COLUMNS), pd.DataFrame(columns=GROUP_COLUMNS)
    for l in ident.fit_lines:
        i = fit.symbols.index(l.symbol)
        cand = ident.candidates.loc[ident.candidates.Element == l.symbol].iloc[0]
        centroid = line_model_center(l,fit)
        sig = float(sigma_kev(l.energy_keV, config))*fit.resolution_scale
        lo, hi = centroid-3*sig, centroid+3*sig
        fractions = np.maximum(0, np.minimum(edges[1:], hi)-np.maximum(edges[:-1], lo))/np.diff(edges)
        gross = float(fractions @ counts)
        area = float(fit.q[i]*l.sensitivity_cm2_g)
        var = ident.qcov[i, i]
        std = float(l.sensitivity_cm2_g*np.sqrt(var)) if np.isfinite(var) else None
        roi_probability = max(0., ndtr((min(edges[-1], hi)-centroid)/sig)-ndtr((max(edges[0], lo)-centroid)/sig))
        truncated = lo < edges[0] or hi > edges[-1]
        notes = 'fixed_by_sensitivity; correlated branches; overlapping Gross_Area values must not be summed'
        if truncated:
            notes += '; roi_truncated=true; full-area response not renormalized'
        row = [l.Z, l.symbol, l.branch, l.energy_keV, centroid, FWHM_FACTOR*sig, area, gross,
               100*std/area if std is not None and area > 0 else None, area*roi_probability,
               max(edges[0], lo), min(edges[-1], hi), std, cand.Status, bool(cand.OverlapFlag),
               cand.Uncertainty_Status, notes, l.sensitivity_cm2_g, truncated, 'success', l.strength_source, cand.Quantification_Status, fit.line_centers.get((l.symbol,l.branch)),
               'user_specified' if (l.symbol,l.branch) in fit.line_centers else 'xraylib']
        (rejected if cand.Status == 'not_supported' else rows).append(row)
    groups = []
    for symbol in fit.symbols:
        alpha = [l for l in ident.fit_lines if l.symbol == symbol and l.branch in ('Kalpha1', 'Kalpha2')]
        if len(alpha) != 2:
            continue
        if ident.candidates.loc[ident.candidates.Element == symbol, 'Status'].iloc[0] == 'not_supported':
            continue
        i = fit.symbols.index(symbol)
        s = np.array([l.sensitivity_cm2_g for l in alpha])
        e = np.array([line_model_center(l,fit) for l in alpha])
        mean = float(s @ e / s.sum())
        variances = np.array([float(sigma_kev(l.energy_keV, config))**2*fit.resolution_scale**2 for l in alpha])
        variance = float(s @ (variances+(e-mean)**2) / s.sum())
        std = s.sum()*np.sqrt(ident.qcov[i, i]) if np.isfinite(ident.qcov[i, i]) else None
        groups.append([alpha[0].Z, symbol, 'Kalpha1+Kalpha2', mean, FWHM_FACTOR*np.sqrt(variance),
                       fit.q[i]*s.sum(), std, 'Moments of composite distribution; shared q, not independent areas'])
    return pd.DataFrame(rows, columns=LINE_COLUMNS), pd.DataFrame(rejected, columns=LINE_COLUMNS), pd.DataFrame(groups, columns=GROUP_COLUMNS)


def export_analysis(result, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result['output_dir'] = str(out.resolve())
    write_json(out/'calibration_result.json', result['calibration_result'])
    write_json(out/'fit_summary.json', result['summary'])
    ident, fit = result['identification'], result['identification'].fit
    result['peaks'].to_csv(out/'detected_peaks.csv', index=False)
    ident.candidates.to_csv(out/'element_candidates.csv', index=False)
    ident.candidates.to_csv(out/'element_diagnostics.csv', index=False)
    pending=ident.candidates.loc[ident.candidates.Element.isin(result['summary']['elements_need_confirmation'])]
    pending.to_csv(out/'elements_need_confirmation.csv',index=False)
    result['element_classes'].to_csv(out/'element_classes.csv',index=False)
    pd.DataFrame(ident.branch_evidence,columns=['Element','Family','Branches','Background_Window_Sigmas','ROI_Start_keV','ROI_End_keV',
                 'Local_Free_Area_Counts','Expected_Area_From_Tied_Counts','Branch_DeltaDeviance','Tied_vs_Free_DeltaDeviance',
                 'Observed_Ratio_to_Strongest_Group','Expected_Ratio_to_Strongest_Group','Group_Observed',
                 'Wide_Window_Branch_DeltaDeviance','Robust_Group_Observed','Ratio_Compatible','Method']).to_csv(out/'branch_ratio_evidence.csv',index=False)
    ident.pairings.to_csv(out/'peak_line_matches.csv', index=False)
    ident.pattern_rankings.to_csv(out/'element_pattern_scores.csv', index=False)
    assignments = []
    for _, peak in result['peaks'].iterrows():
        active_symbols = set(ident.candidates.loc[ident.candidates.Status != 'not_supported', 'Element'])
        pairs = ident.pairings.loc[(ident.pairings.peak_id == peak.peak_id) & ident.pairings.Selected_Element
                                  & ident.pairings.Element.isin(active_symbols)]
        confirmed = set(ident.candidates.loc[ident.candidates.Status == 'supported', 'Element'])
        firm = pairs.loc[pairs.Element.isin(confirmed)]
        chosen = firm if len(firm) else pairs
        actual_center=peak.energy_keV
        if result['requested_config'].find_peaks is not None and result['requested_config'].find_peaks_axis=='reference':
            actual_center=peak.energy_keV*(1+fit.energy_gain)+fit.delta
        roi = (result['energy'] >= peak.roi_start_keV) & (result['energy'] <= peak.roi_end_keV)
        contributions = {symbol: float(fit.components[roi, i].sum()) for i, symbol in enumerate(fit.symbols)
                         if symbol in set(chosen.Element)}
        maximum = max(contributions.values(), default=0.)
        names = sorted((symbol for symbol in contributions if contributions[symbol] >= .1*maximum and maximum > 0),
                       key=lambda symbol: contributions[symbol], reverse=True)
        primary = names[0] if names else ''
        preferred=chosen.loc[chosen.Preferred_By_Prior.fillna(False),'Element'].unique().tolist()
        context_preferred=preferred[0] if not len(firm) and len(preferred)==1 else ''
        if context_preferred:
            primary=context_preferred
            names=[primary]+[name for name in names if name!=primary]
        primary_information = float(np.sqrt(np.sum(fit.components[roi, fit.symbols.index(primary)]**2/fit.model[roi]))) if primary else 0.
        minor = sorted(set(chosen.Element)-set(names))
        alternatives = ident.pairings.loc[(ident.pairings.peak_id == peak.peak_id) & ~ident.pairings.Selected_Element]
        assignments.append({'peak_id': int(peak.peak_id), 'Fitted_Centroid_keV': actual_center, 'Input_Center_keV':peak.energy_keV,
                            'Centroid_Std_keV': peak.centroid_std_keV,
                            'Elements': ','.join(names), 'Primary_Element': primary,
                            'Minor_Overlapping_Elements': ','.join(minor),
                            'Assignment_Model_Information_Score': primary_information,
                            'Preferred_By_Context_Prior':context_preferred,
                            'Branches': ';'.join(sorted({r.Element+':'+r.Branch for _, r in chosen.iterrows()})),
                            'Status': 'unassigned' if not names else ('supported' if len(firm) and primary_information >= result['config'].candidate_threshold_sigma else
                                      ('ambiguous' if pairs.Element_Status.isin(['ambiguous', 'unidentifiable']).any() else 'tentative')),
                            'Alternative_Elements': ','.join(sorted(set(alternatives.Element))),
                            'Notes': 'Observed complex center; unresolved branches share one evidence region'})
    pd.DataFrame(assignments, columns=['peak_id', 'Fitted_Centroid_keV', 'Centroid_Std_keV', 'Elements',
                 'Branches', 'Status', 'Alternative_Elements', 'Input_Center_keV', 'Primary_Element', 'Minor_Overlapping_Elements', 'Assignment_Model_Information_Score', 'Preferred_By_Context_Prior', 'Notes']).to_csv(out/'peak_assignments.csv', index=False)
    auxiliary = []
    for line in fit.auxiliary_lines:
        name = '%s_%s_unscaled' % (line.symbol, line.parent_shell)
        total_rate = sum(l.relative_rate for l in fit.auxiliary_lines if l.symbol == line.symbol and l.parent_shell == line.parent_shell)
        auxiliary.append({'Z': line.Z, 'Element': line.symbol, 'Branch': line.branch,
                          'Reference_Energy_keV': line.energy_keV,
                          'Centroid_keV': line_model_center(line,fit),
                          'Net_Area': fit.auxiliary_areas[name]*line.relative_rate/total_rate,
                          'Strength_Source': 'unscaled_shell_radiative_rates',
                          'Quantitative_S_Available': False,
                          'Notes': 'Free shell amplitude; not a sensitivity-derived areal density; uncertainty not estimated'})
    pd.DataFrame(auxiliary, columns=['Z', 'Element', 'Branch', 'Reference_Energy_keV', 'Centroid_keV', 'Net_Area',
                 'Strength_Source', 'Quantitative_S_Available', 'Notes']).to_csv(out/'auxiliary_fitted_lines.csv', index=False)
    result['density'].to_csv(out/'relative_areal_density.csv', index=False)
    lines, rejected, groups = fitted_line_tables(ident, result['counts'], result['edges'], result['config'])
    if result['summary']['analysis_status'] != 'completed':
        for table in (lines, rejected):
            table['Notes'] = table['Notes'] + '; analysis_status=' + result['summary']['analysis_status'] + '; model_conditional_statistical_errors_only'
    lines.to_csv(out/'fitted_lines.csv', index=False)
    rejected.to_csv(out/'rejected_candidate_lines.csv', index=False)
    groups.to_csv(out/'fitted_groups.csv', index=False)
    pd.DataFrame(ident.diagnostics, columns=['Element', 'Branches', 'Family', 'ExpectedCounts', 'InformationScore',
                 'RegionResidualZ', 'branch_ratio_mismatch']).to_csv(out/'branch_diagnostics.csv', index=False)
    result['unassigned_peaks'].to_csv(out/'unassigned_peaks.csv', index=False)
    y, mu = result['counts'], fit.model
    dev = 2*(xlogy(y, np.maximum(y, 1e-300)/mu) - (y-mu))
    spectrum = pd.DataFrame({'channel': result['channel'], 'energy_keV': result['energy'], 'observed_counts': y,
                            'model_total_counts': mu, 'background_counts': fit.background,
                            'scatter_or_artifact_counts': fit.scatter, 'element_total_counts': fit.components.sum(axis=1),
                            'pearson_residual': (y-mu)/np.sqrt(mu),
                            'deviance_residual': np.sign(y-mu)*np.sqrt(np.maximum(dev, 0))})
    spectrum.to_csv(out/'fit_spectrum.csv', index=False)
    components = pd.DataFrame(fit.components, columns=fit.symbols)
    components.insert(0, 'energy_keV', result['energy'])
    components.to_csv(out/'fit_components.csv', index=False)
    if result['config'].save_intermediate:
        pd.DataFrame(result['sensitivity'].states).to_csv(out/'sensitivity_states.csv', index=False)
        pd.DataFrame({'energy_keV': result['energy'], 'snip_counts': result['background0']}).to_csv(out/'background_initial.csv', index=False)
    calcium=ident.candidates.loc[ident.candidates.Element=='Ca']
    write_json(out/'calcium_analysis.json',{'candidate':calcium.to_dict('records'),
               'branch_evidence':[row for row in ident.branch_evidence if row['Element']=='Ca'],
               'interpretation':'Identity and quantitative quality are separate; weak Kbeta evidence requires confirmation near thresholds',
               'manual_peaks':result['summary']['manual_peaks']})
    from .visualization import make_plots,make_calcium_plot
    make_plots(result, out)
    make_calcium_plot(result,out)
