from __future__ import annotations
from pathlib import Path
from dataclasses import replace
from collections import Counter
import importlib.metadata
import logging
import time
import numpy as np
import pandas as pd
from .config import AnalysisConfig
from .calibration import fit_linear_calibration
from .io_spectrum import load_spectrum
from .atomic_lines import load_sensitivity, validate_symbol, excitation_edge_kev
from .atomic_completion import complete_transitions
from .measurement import infer_hard_lower_cutoff, initialize_resolution, reclassify_peak_widths
from .response import sigma_kev, gaussian_response, FWHM_FACTOR
from .baseline import snip_background
from .peak_detection import detect_peaks
from .manual_peaks import fit_requested_peaks
from .element_prior import classification_table, element_class, apply_reporting_policy
from .candidate_reporting import filter_possible_candidates
from .element_identification import identify_elements
from .fit_model import fit_spectrum
from .quantification import compute_relative_areal_density
from .reporting import sha256, export_analysis
from .console import print_run_summary


_USE_CONFIG_BACKGROUND=object()


def analyze(spectrum, calibration, sensitivity, output_dir=None, config=None, background_spectrum=_USE_CONFIG_BACKGROUND):
    start = time.monotonic()
    config = config or AnalysisConfig()
    if background_spectrum is not _USE_CONFIG_BACKGROUND:
        config=replace(config,background_spectrum=background_spectrum)
    config=config.validate()
    for symbol in (list(config.elements or ()) + list(config.exclude_elements)
                   + [s for s in (config.reference_element, config.tube_target) if s]):
        validate_symbol(symbol)
    spec = load_spectrum(spectrum, config.channel_start, config.fill_missing_channels)
    cal = fit_linear_calibration(calibration, FWHM_FACTOR*float(sigma_kev(6., config)), config.strict_calibration)
    table = load_sensitivity(sensitivity)
    atomic_lines, auxiliary_lines, completion_audit = complete_transitions(table.lines, config)
    cutoff = infer_hard_lower_cutoff(spec, cal, config)
    warnings = list(spec.metadata.get('warnings', [])) + cal.warnings + table.warnings
    energy_all = cal.energy(spec.channel)
    if not np.isfinite(energy_all).all() or energy_all[-1] <= 0 or energy_all[-1] > 1000:
        raise ValueError('Abnormal calibrated spectrum energy range')
    supplied = spec.metadata.get('supplied_energy_keV')
    if supplied is not None and np.max(np.abs(np.array(supplied)-energy_all)) > max(1e-6, cal.b_keV_per_channel*.1):
        warnings.append('CSV energy conflicts with explicit calibration; explicit calibration used')
    embedded = spec.metadata.get('embedded_calibration_points', [])
    if embedded and any(abs(cal.energy(p[0])-p[1]) > max(.001, .1*cal.b_keV_per_channel) for p in embedded):
        warnings.append('MCA embedded calibration conflicts with explicit calibration; explicit calibration used')
    mask = energy_all > 0
    if config.energy_range:
        mask &= (energy_all >= config.energy_range[0]) & (energy_all <= config.energy_range[1])
    if cutoff is not None:
        mask &= energy_all >= cutoff['energy_keV']
        label = 'Inferred' if cutoff['inferred'] else 'Recorded'
        warnings.append('%s acquisition lower cutoff %.5f keV: excluded channels are not treated as measured absence' % (label, cutoff['energy_keV']))
    if mask.sum() < max(config.background_degree+5, 8):
        raise ValueError('Fewer than required channels in requested energy range')
    channel, counts, energy = spec.channel[mask], spec.counts[mask], energy_all[mask]
    edges = cal.energy(np.r_[channel-.5, channel[-1]+.5])
    logging.info('Loaded %s: %d/%d channels, %.4g counts, energy %.4g..%.4g keV', spectrum, len(counts), len(spec.counts), counts.sum(), edges[0], edges[-1])
    iterations = config.snip_iterations or max(5, int(np.ceil(4*float(sigma_kev(np.median(energy), config))*FWHM_FACTOR/cal.b_keV_per_channel)))
    iterations = min(iterations, (len(counts)-1)//2)
    b0 = snip_background(counts, iterations)
    measured_counts=None
    background_info={'enabled':False,'path':None}
    if config.background_spectrum is not None:
        from .background_input import load_background
        measured_counts,background_info=load_background(config.background_spectrum,channel,cal,spec,config,sample_path=spectrum)
        b0=np.maximum(b0-background_info['scale_initial']*snip_background(measured_counts,iterations),.1)
        warnings.extend(background_info['warnings'])
        warnings.append('Measured background enabled with fitted global scale; acquisition/shape matching is not verified')
    if cutoff is not None:
        # SNIP has no samples below an acquisition boundary: extrapolate a low
        # interior background level instead of anchoring it to a truncated peak.
        k = min(iterations, len(b0)//4)
        edge_level = float(np.percentile(b0[k:3*k], 25))
        b0[:k] = np.minimum(b0[:k], edge_level)
        cutoff['background_edge_initialization'] = 'lower_quartile_of_adjacent_SNIP_background'
    manual_response=None
    config._background_counts=measured_counts
    config._background_scale_initial=background_info.get('scale_initial',1.)
    config._background_scale=config._background_scale_initial
    detection_background=b0+(config._background_scale_initial*measured_counts if measured_counts is not None else 0.)
    if config.find_peaks is None:
        peaks = detect_peaks(channel, energy, edges, counts, detection_background, config)
    else:
        peaks,manual_response=fit_requested_peaks(channel,energy,edges,counts,detection_background,config)
    logging.info('SNIP iterations=%d; observed candidate features=%d', iterations, len(peaks))
    fit_config, response_initialization = initialize_resolution(peaks, config)
    if manual_response is not None:
        limit=FWHM_FACTOR*np.sqrt(config.fano*config.pair_energy_kev*config.fwhm_ref_energy_kev)*1.001
        fit_config=replace(config,fwhm_ref_kev=max(limit,config.fwhm_ref_kev*manual_response['resolution_scale']),
                           centroid_tolerance_kev=max(config.centroid_tolerance_kev,config.manual_match_tolerance_kev/config.match_sigma))
        response_initialization={'used':True,'method':'user_centers_shared_response_fit','effective_reference_FWHM_keV':fit_config.fwhm_ref_kev}
    if config.find_peaks is None and response_initialization.get('used'):
        old_broad=int(peaks.quality_flags.str.contains('broad_structure').sum())
        peaks=reclassify_peak_widths(peaks,fit_config)
        response_initialization['broad_flags_before_update']=old_broad
        response_initialization['broad_flags_after_update']=int(peaks.quality_flags.str.contains('broad_structure').sum())
        response_initialization['peak_quality_response']='empirically_initialized_resolution'
    lines, exclusions = [], []
    fit_config._background_counts=measured_counts
    fit_config._background_scale_initial=config._background_scale_initial
    # Use declared measurement metadata when no voltage is configured; never
    # infer excitation from a filename or from an emission peak energy.
    excitation_voltage = (config.tube_voltage_kv if config.tube_voltage_kv is not None
                          else spec.metadata.get('tube_voltage_kV'))
    fit_config._excitation_voltage_kv = excitation_voltage
    for line in atomic_lines:
        reason = None
        if config.elements and line.symbol not in config.elements:
            reason = 'not_whitelisted'
        elif line.symbol in config.exclude_elements:
            reason = 'blacklisted'
        elif excitation_voltage is not None and excitation_edge_kev(line) >= excitation_voltage:
            reason = 'initial_shell_not_excitable_at_tube_voltage'
        elif gaussian_response(edges, line.energy_keV, float(sigma_kev(line.energy_keV, config))).sum() < 1e-6:
            reason = 'outside_measurable_energy_range'
        if reason:
            exclusions.append({'Element': line.symbol, 'Branch': line.branch, 'reason': reason})
        else:
            lines.append(line)
    delta_bound = config.delta_max_kev or max(.5*cal.b_keV_per_channel, cal.rms_keV, .02)
    observable_auxiliary = []
    for line in auxiliary_lines:
        if excitation_voltage is not None and excitation_edge_kev(line) >= excitation_voltage:
            exclusions.append({'Element': line.symbol, 'Branch': line.branch,
                               'reason': 'initial_shell_not_excitable_at_tube_voltage'})
        elif gaussian_response(edges, line.energy_keV, float(sigma_kev(line.energy_keV, fit_config))).sum() >= 1e-6:
            observable_auxiliary.append(line)
    auxiliary_lines = observable_auxiliary
    identification = identify_elements(counts, energy, edges, b0, lines, peaks, fit_config, delta_bound, cal.rms_keV, auxiliary_lines=auxiliary_lines, manual_response=manual_response)
    fit = identification.fit
    fit_config._background_scale=getattr(fit,'background_scale',None) or 0.
    fit_config._background_scale_std=getattr(fit,'background_scale_std',None)
    if measured_counts is not None:
        background_info.update(scale=fit.background_scale,scale_std=fit.background_scale_std,
                               scale_uncertainty_status=fit.background_scale_uncertainty_status,
                               sample_deviance=fit.sample_deviance,joint_deviance=fit.joint_deviance)
    # Keep all rows in the sensitivity table, even when there are no usable branches.
    for symbol in sorted({s['Element'] for s in table.states} - set(identification.candidates.Element)):
        if config.elements and symbol not in config.elements or symbol in config.exclude_elements:
            continue
        z = validate_symbol(symbol)
        row = {'Z': z, 'Element': symbol, 'Status': 'unidentifiable', 'NumObservableLines': 0,
               'NumIndependentEvidenceRegions': 0, 'OverlapFlag': False,
               'ConfidenceNotes': 'No positive measurable branches in analysis range; not evidence of absence',
               'Uncertainty_Status': 'not_estimable', 'Quantification_Status': 'not_estimable',
               'Element_Class':element_class(symbol,config.element_prior),'Preferred_By_Prior':False}
        identification.candidates = pd.concat([identification.candidates, pd.DataFrame([row])], ignore_index=True)
    identification.candidates = identification.candidates.sort_values('Z').reset_index(drop=True)
    identification.candidates = apply_reporting_policy(identification.candidates, config.lanthanide_reporting,
                                                        config.noble_gas_reporting)
    identification.candidates = filter_possible_candidates(identification.candidates, identification, peaks, fit_config)
    bootstrap_info = {'requested': config.bootstrap, 'successful': 0, 'method': None}
    if config.bootstrap and fit.success and fit.symbols:
        rng = np.random.default_rng(config.seed)
        samples = []
        for i in range(config.bootstrap):
            simulated = rng.poisson(fit.model)
            boot = fit_spectrum(simulated, energy, edges, b0, identification.fit_lines, fit_config,
                                identification.alignment['effective_fit_delta_bound_keV'],
                                auxiliary_lines=fit.auxiliary_lines, initial_delta=fit.delta,line_centers=fit.line_centers,center_axis=fit.center_axis,
                                background_counts=rng.poisson(fit.background_latent_counts) if measured_counts is not None else None)
            if boot.success:
                samples.append(boot.q)
            if (i+1) % 10 == 0:
                logging.info('Bootstrap %d/%d (%d successful)', i+1, config.bootstrap, len(samples))
        bootstrap_info.update(successful=len(samples), method='parametric_poisson_refit_all_parameters_conditional_candidate_set_and_snip',
                              selection_uncertainty_included=False, background_initialization_fixed=True)
        if len(samples) >= max(20, int(.8*config.bootstrap)):
            samples = np.array(samples)
            identification.uncertainty_info['bootstrap_covariance'] = np.cov(samples, rowvar=False).reshape(len(fit.symbols), len(fit.symbols)).tolist()
            for i, symbol in enumerate(fit.symbols):
                idx = identification.candidates.Element == symbol
                identification.candidates.loc[idx, 'Upper95_q'] = np.quantile(samples[:, i], .95)
            # Fisher remains authoritative for interior estimates; boundary bootstrap quantiles are conditional diagnostics.
            bootstrap_info['q_samples'] = samples.tolist()
        else:
            warnings.append('Insufficient successful bootstrap replicates; no bootstrap intervals reported (need >=20 and 80% success)')
    dof = max(0, len(counts)-fit.n_parameters)
    severe_mismatch = bool(dof and fit.deviance/dof > config.max_deviance_per_dof)
    identification.candidates['Global_Quantification_Status']=identification.candidates.Quantification_Status
    if severe_mismatch:
        idx = identification.candidates.Status == 'supported'
        identification.candidates.loc[idx, 'Global_Quantification_Status'] = 'global_model_mismatch'
        identification.candidates.loc[idx, 'Quantification_Status'] = 'global_model_mismatch'
        identification.candidates.loc[idx, 'ConfidenceNotes'] += '; severe_global_model_mismatch: joint-template q is diagnostic only'
        warnings.append('Severe global model mismatch: joint-template q is diagnostic; independently measured primary peak areas are evaluated separately')
    local_quantification=None
    if config.quantification_method=='local-peaks' and fit.success:
        from .local_quantification import fit_local_quantification
        local_quantification=fit_local_quantification(identification,counts,energy,edges,fit_config)
        local_lookup=local_quantification['elements'].set_index('Element')
        for symbol in identification.candidates.loc[identification.candidates.Status=='supported','Element']:
            row=local_lookup.loc[symbol] if symbol in local_lookup.index else None
            status=row.Quantification_Status if row is not None else 'local_area_not_estimable'
            if status=='local_area_estimate':status='supported'
            identification.candidates.loc[identification.candidates.Element==symbol,'Quantification_Status']=status
    density, reference_info = compute_relative_areal_density(identification, config.reference_element,local_quantification)
    if severe_mismatch:
        density['Quality_Flags'] += '; global_template_mismatch; joint_q_diagnostic_only'
    declared = table.metadata.get('declared', {})
    source_mismatch = False
    # An explicit voltage stored with the sample is evidence even when the user
    # did not repeat it in the CLI/config. Never infer conditions from filenames.
    sample_voltage = spec.metadata.get('tube_voltage_kV')
    actual_voltage = config.tube_voltage_kv if config.tube_voltage_kv is not None else sample_voltage
    for key, actual in [('tube_target', config.tube_target), ('tube_voltage_kV', actual_voltage)]:
        if key not in declared or actual is None:
            continue
        try:
            compatible = (bool(np.isclose(float(declared[key]), actual))
                          if key == 'tube_voltage_kV' else declared[key] == actual)
        except (TypeError, ValueError):
            compatible = False
        if not compatible:
            source_mismatch = True
            warnings.append('Sensitivity excitation metadata mismatch: ' + key + '; quantitative ratios withheld')
    if sample_voltage is not None and config.tube_voltage_kv is not None and not np.isclose(sample_voltage, config.tube_voltage_kv):
        source_mismatch = True
        warnings.append('Sample tube voltage conflicts with configured excitation; quantitative ratios withheld')
    if source_mismatch:
        for col in ['Relative_to_Reference', 'Fraction_among_Reported_Elements', 'Uncertainty_percent', 'CI95_Low', 'CI95_High']:
            density[col] = np.nan
        density['Quality_Flags'] += '; excitation_mismatch_ratios_withheld'
        density['Quantification_Status']='excitation_mismatch'
        density['Global_Quantification_Status']='excitation_mismatch'
        identification.candidates.loc[identification.candidates.Status=='supported','Quantification_Status']='excitation_mismatch'
        identification.candidates.loc[identification.candidates.Status=='supported','Global_Quantification_Status']='excitation_mismatch'
        reference_info = {'Element': None, 'selection_rule': 'excitation_mismatch_ratios_withheld', 'fraction_denominator_elements': []}
    accepted_symbols = set(identification.candidates.loc[identification.candidates.Status.isin(['supported', 'tentative']), 'Element'])
    selected_pairs = identification.pairings.loc[identification.pairings.Element.isin(accepted_symbols)]
    assigned = set(selected_pairs.peak_id.astype(int))
    unassigned = peaks.loc[~peaks.peak_id.isin(assigned)].copy()
    dof = max(0, len(counts)-fit.n_parameters)
    residual = (counts-fit.model)/np.sqrt(fit.model)
    if not fit.success:
        warnings.append('Joint optimization failed: quantities and symmetric uncertainties must not be interpreted')
    if fit.response_at_boundary:
        warnings.append('Global response parameter at bound: review calibration/resolution; no symmetric quantitative uncertainty')
    if dof and fit.deviance/dof > 2:
        warnings.append('Structured residual / overdispersion: matrix_effect_or_model_mismatch or unmodelled scattering/artifacts')
    if any(d['branch_ratio_mismatch'] for d in identification.diagnostics):
        warnings.append('branch_ratio_mismatch: regional observed residuals disagree with the fixed branch model')
    if config.tube_target:
        warnings.append('Target %s fluorescence and Rayleigh scattering cannot establish sample %s; sample target ratios withheld' % (config.tube_target, config.tube_target))
    if len(unassigned):
        warnings.append('%d observed features remain unassigned; review scattering, artifacts, or incomplete atomic dictionary' % len(unassigned))
    if counts.sum() == 0:
        warnings.append('Zero-count spectrum: no elemental detections or quantitative ratios')
    if not (identification.candidates.Quantification_Status == 'supported').any():
        warnings.append('No quantitatively valid elements; fraction denominator is empty; see separate identification statuses')
    needs_confirmation=identification.candidates.loc[identification.candidates.Status.isin(['tentative','ambiguous','unidentifiable']) & identification.candidates.q_Fit.notna() & identification.candidates.Reportable,'Element'].tolist()
    if needs_confirmation:
        warnings.append('需要进一步确定（待定）的元素: '+', '.join(needs_confirmation)+'；不参与相对面密度计算')
    versions = {}
    for package in ('numpy', 'scipy', 'pandas', 'matplotlib', 'xraylib', 'xrf-spectrum-analyzer'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unknown'
    files = {name: {'path': str(Path(p).resolve()), 'sha256': sha256(p)} for name, p in
             [('sample_spectrum', spectrum), ('calibration', calibration), ('sensitivity', sensitivity)]}
    if table.metadata.get('metadata_file'):
        files['sensitivity_metadata'] = {'path': table.metadata['metadata_file'], 'sha256': sha256(table.metadata['metadata_file'])}
    if config.background_spectrum is not None:
        files['background_spectrum']={'path':background_info['path'],'sha256':sha256(config.background_spectrum)}
    summary = dict(schema_version='3.0', measured_background=background_info, fit_status='success' if fit.success else 'failed',
                   analysis_status='fit_failed' if not fit.success else ('model_mismatch' if severe_mismatch else 'completed'),
                   quantitative_interpretation_valid=bool(fit.success and not source_mismatch and density.Relative_to_Reference.notna().any()),
                   global_model_valid=bool(fit.success and not severe_mismatch),
                   local_quantification_valid=bool(local_quantification is not None and not source_mismatch and reference_info['fraction_denominator_elements']),
                   normalized_fractions_available=bool(density.Fraction_among_Reported_Elements.notna().any()),
                   quantification_method=config.quantification_method,
                   input_files=files, dependency_versions=versions, parameters=config.to_dict(),
                   candidate_reporting={'possible_min_delta_deviance':config.possible_min_delta_deviance,
                       'requires_observed_principal_peak':bool(config.possible_min_delta_deviance),
                       'evidence_is_conditional_not_a_probability':True,
                       'suppressed_weak_candidates':identification.candidates.loc[
                           identification.candidates.Reporting_Reason.str.startswith('possible_requires'), 'Element'].tolist(),
                       'supported_statuses_and_quantification_unchanged':True},
                   effective_parameters={'background_method': 'snip', 'snip_iterations': iterations, 'delta_E_bound_keV': identification.alignment['effective_fit_delta_bound_keV'],
                                         'excitation_tube_voltage_kV': excitation_voltage,
                                         'excitation_voltage_source': ('configuration' if config.tube_voltage_kv is not None else
                                                                       'sample_metadata' if excitation_voltage is not None else 'unknown'),
                                         'effective_fwhm_ref_kev': fit_config.fwhm_ref_kev,
                                         'energy_edges_keV': [float(edges[0]), float(edges[-1])]},
                   calibration_quality=cal.calibration_quality, sensitivity=table.metadata,
                   line_energy={'source': 'xraylib', 'api': 'xraylib.LineEnergy', 'units': 'keV'},
                   manual_peaks=manual_response, element_prior={'profile':config.element_prior,'tie_margin':config.prior_tie_margin,'policy':'tie_break_only; data_ambiguity_is_not_confirmation',
                       'lanthanide_reporting':config.lanthanide_reporting,
                       'noble_gas_reporting':config.noble_gas_reporting,
                       'suppressed_lanthanide_candidates':identification.candidates.loc[
                           identification.candidates.Z.between(57,71) & ~identification.candidates.Reportable & identification.candidates.Status.isin(['tentative','ambiguous','unidentifiable']) & identification.candidates.q_Fit.notna(), 'Element'].tolist(),
                       'suppressed_noble_gas_candidates':identification.candidates.loc[
                           identification.candidates.Z.isin([2,10,18,36,54,86,118]) & ~identification.candidates.Reportable & identification.candidates.Status.isin(['tentative','ambiguous','unidentifiable']) & identification.candidates.q_Fit.notna(), 'Element'].tolist(),
                       'reporting_policy_does_not_modify_fit_or_establish_absence':True},
                   elements_need_confirmation=needs_confirmation,
                   excluded_from_density=identification.candidates.loc[identification.candidates.Status!='supported','Element'].tolist(),
                   local_branch_evidence=identification.branch_evidence,
                   low_z_kalpha_evidence=identification.low_z_evidence,
                   acquisition_cutoff=cutoff, response_initialization=response_initialization,
                   centroid_alignment=identification.alignment, atomic_completion=completion_audit,
                   identification_method='fitted_centroids_then_global_branch_pattern_then_joint_Poisson_fit',
                   unidentified_shell_amplitudes=fit.auxiliary_areas,
                   sensitivity_states=dict(Counter(s['State'] for s in table.states)),
                   pruned_lines=exclusions, sample_metadata=spec.metadata, branch_ratios='fixed_by_sensitivity_plus_same_shell_completion; unknown_shells_separate_nuisances',
                   model='raw_counts_poisson; centroid_pattern_selected; integrated_gaussian; q_i*S_i_line; unscaled_shell_nuisances; SNIP*exp(Legendre)',
                   objective=fit.objective, negative_log_likelihood_without_factorial=fit.nll,
                   poisson_deviance=fit.deviance, degrees_of_freedom=dof,
                   deviance_per_dof=fit.deviance/dof if dof else None,
                   model_comparisons=identification.comparisons, overlap_pairs=identification.overlap_pairs,
                   branch_diagnostics=identification.diagnostics, delta_E_keV=fit.delta,
                   resolution_scale=fit.resolution_scale, energy_gain_correction=fit.energy_gain,
                   gain_fitted=fit.gain_fitted, background_parameters=fit.parameters[
                       len(fit.symbols)+len(fit.nuisance_names):len(fit.symbols)+len(fit.nuisance_names)+config.background_degree+1].tolist(),
                   optimizer={'message': fit.message, 'attempts': fit.attempts}, nuisance_peaks=fit.nuisance_names,
                   uncertainty=identification.uncertainty_info, q_covariance=identification.qcov.tolist(),
                   covariance_element_order=fit.symbols, bootstrap=bootstrap_info,
                   reference=reference_info, apparent_relative_areal_density=True,
                   primary_peak_quantification=None if local_quantification is None else {
                       'method':local_quantification['method'],
                       'covariance_method':local_quantification.get('covariance_method'),
                       'formula':'A_i = common_flux_geometry_time * areal_density_i * S_i; ratio=(A_i/S_i)/(A_ref/S_ref)',
                       'elements':local_quantification['elements'].to_dict('records'),
                       'covariance_element_order':local_quantification['elements'].Element.tolist(),
                       'q_area_covariance':np.asarray(local_quantification['covariance']).tolist(),
                       'q_area_stat_covariance':np.asarray(local_quantification.get('stat_covariance',local_quantification['covariance'])).tolist(),
                       'audit':local_quantification['audit'],
                       'conditional_on':'supplied sensitivity; selected elements and local line shapes; thin-sample apparent density',
                       'global_template_q_unchanged':True},
                   assumptions={'thin_sample': True, 'self_absorption': False, 'full_energy_efficiency': 1.0,
                                'absolute_areal_density_available': False},
                   systematics_included=False, excitation_match_verified=False,
                   target_scattering_review='target specified' if config.tube_target else 'Tube target unspecified; candidate target fluorescence may be instrument scattering',
                   residual_anomalies={'abs_pearson_gt5': int(np.sum(abs(residual)>5)),
                                       'unassigned_peaks': unassigned.to_dict('records')},
                   warnings=warnings, elapsed_seconds=time.monotonic()-start)
    calibration_result = cal.to_dict()
    calibration_result.update(acquisition_cutoff=cutoff, channel_origin=int(spec.channel[0]), effective_energy_range_keV=[float(edges[0]), float(edges[-1])])
    result = dict(calibration=cal, calibration_result=calibration_result, sensitivity=table,
                  config=fit_config, requested_config=config, channel=channel, counts=counts, energy=energy, edges=edges,
                  background0=b0, peaks=peaks, identification=identification, density=density,
                  summary=summary, unassigned_peaks=unassigned, element_classes=classification_table(config.element_prior))
    result['local_quantification']=local_quantification
    if output_dir:
        export_analysis(result, output_dir)
    for warning in warnings:
        logging.warning(warning)
    logging.info('Completed %s in %.2fs: %s', spectrum, time.monotonic()-start, summary['fit_status'])
    print_run_summary(result)
    return result
