"""K-alpha-dominant identification without treating weak K-beta as a veto."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from .response import gaussian_response, sigma_kev, line_model_center


def _refined_alpha_profile(symbol, alpha, lines, counts, energy, edges, fit, config, radius):
    """Profile interference and local response under both identity hypotheses."""
    from types import SimpleNamespace
    import pandas as pd
    from .local_quantification import _groups, _profile
    provisional = []
    for j, other in enumerate(fit.symbols):
        component = fit.matrix[:, j]*fit.t[j]
        strong = np.sqrt(np.sum(component**2/fit.model)) >= np.sqrt(config.support_delta_deviance)
        anchored = other in getattr(config, '_branch_confirmed_symbols', set())
        provisional.append({'Element': other, 'Status': 'supported' if strong and anchored else 'tentative'})
    groups = _groups(SimpleNamespace(fit=fit, fit_lines=lines, candidates=pd.DataFrame(provisional)), config, edges)
    strength = sum(l.sensitivity_cm2_g for l in alpha)
    target = {'Element': symbol, 'Z': alpha[0].Z, 'lines': alpha,
              'weights': np.array([l.sensitivity_cm2_g/strength for l in alpha]), 'S': strength,
              'centers': np.array([line_model_center(l, fit) for l in alpha]),
              'sigmas': np.array([float(sigma_kev(l.energy_keV, config))*fit.resolution_scale for l in alpha]),
              'expected': fit.q[fit.symbols.index(symbol)]*strength,
              'confirmed': True, 'auxiliary': False, 'Branches': '+'.join(l.branch for l in alpha)}
    branches = {l.branch for l in alpha}
    neighbours = [g for g in groups if not (g['Element'] == symbol and
                  any(l.branch in branches for l in g['lines']))]
    neighbours = [dict(g, confirmed=True) if g['Element'] == symbol and not g['auxiliary'] else g
                  for g in neighbours]
    result = _profile(target, [target]+neighbours, counts, energy, edges, config, radius,
                      background_scale=getattr(fit, 'background_scale', 0.) or 0.)
    if not result['Success']:
        return None
    return {'delta': result['DeltaDeviance'], 'area': result['Area_Counts'], 'S': strength,
            'area_std': result['q_Fisher_Std']*strength*np.sqrt(max(1., result['Deviance_per_dof'])),
            'local_shift': result['Local_Shift_keV'], 'width_scale': result['Local_Width_Scale'],
            'deviance_per_dof': result['Deviance_per_dof'],
            'response_at_boundary': result['Response_At_Boundary']}


def _alpha_profile(lines, counts, energy, edges, fit, index, config, radius):
    centers=[line_model_center(l,fit) for l in lines]
    sigmas=[float(sigma_kev(l.energy_keV,config))*fit.resolution_scale for l in lines]
    ids=np.flatnonzero((energy>=min(c-radius*s for c,s in zip(centers,sigmas))) &
                       (energy<=max(c+radius*s for c,s in zip(centers,sigmas))))
    if len(ids)<7:return None
    a,b=ids[0],ids[-1]+1; y=counts[a:b];e=energy[a:b]
    template=sum(l.sensitivity_cm2_g*gaussian_response(edges[a:b+1],c,s)
                 for l,c,s in zip(lines,centers,sigmas))
    total_s=sum(l.sensitivity_cm2_g for l in lines);p=template/total_s
    other=np.maximum(fit.model[a:b]-fit.background[a:b]-fit.components[a:b,index],0.)
    h=np.column_stack([np.ones(len(e)),(e-e.mean())/max(np.ptp(e),1e-9)])
    scale=max(float(y.sum()),10.)
    reference=getattr(fit,'background_observed_counts',None)
    reference=reference[a:b] if reference is not None else None
    alpha=getattr(fit,'background_scale',0.) or 0.
    def solve(null=False):
        def obj(v):
            bg=np.exp(h@v[1:]);mu=np.maximum(other+bg+v[0]*scale*p,1e-12);r=1-y/mu
            loss=float(np.sum(mu-xlogy(y,mu)))
            if reference is not None:
                from .measured_background import profile_background
                prof=profile_background(y,mu,alpha,reference)
                loss=prof['nll'];r=prof['signal_gradient']
            return loss,np.r_[scale*(p@r),(bg[:,None]*h).T@r]
        bounds=[(0.,0.) if null else (0.,None),(-8.,15.),(-4.,4.)]
        return minimize(obj,[0. if null else .5,np.log(max(np.percentile(y,30),.2)),0.],jac=True,
                        bounds=bounds,method='L-BFGS-B',options={'maxiter':700,'ftol':1e-12,'maxls':40})
    full,null=solve(),solve(True)
    if not full.success or not null.success:return None
    return {'delta':max(0.,2*(null.fun-full.fun)),'area':float(full.x[0]*scale),'S':total_s}


def assess_low_z_kalpha(symbol, lines, peaks, pairs, ranking, counts, energy, edges,
                         fit, config, branch_details, *, profile_response=True):
    own=[l for l in lines if l.symbol==symbol]
    if not own or own[0].Z>config.low_z_max_z or not config.low_z_kalpha:
        return {'Element':symbol,'Applicable':False,'Eligible':False}
    alpha=[l for l in own if l.branch in ('Kalpha1','Kalpha2')]
    result={'Element':symbol,'Z':own[0].Z,'Applicable':bool(alpha),'Eligible':False,
            'Method':'low_Z_Kalpha_dominant; weak_Kbeta_not_required','Reasons':[]}
    if not alpha or symbol not in fit.symbols:return result
    index=fit.symbols.index(symbol)
    matched=pairs.loc[(pairs.Element==symbol)&pairs.Branch.isin(['Kalpha1','Kalpha2'])&(pairs.Group_Affinity>=.5)]
    matched_ids=set(matched.peak_id.astype(int))
    usable=peaks.loc[peaks.peak_id.isin(matched_ids) & ~peaks.quality_flags.str.contains('broad_structure')]
    if usable.empty:
        result['Reasons'].append('No precise observed Kalpha match');return result
    # Response must be fixed by an explicit calibration assumption or by other
    # observed strong branches; a single free-shift peak cannot confirm itself.
    anchor_points=[]
    for other in fit.symbols:
        if other==symbol:continue
        j=fit.symbols.index(other);component=fit.matrix[:,j]*fit.t[j]
        info=float(np.sqrt(np.sum(component**2/fit.model)))
        if info<config.candidate_threshold_sigma:continue
        candidate=pairs.loc[(pairs.Element==other)&(pairs.Group_Affinity>=.5)
                            & ~pairs.peak_id.isin(matched_ids)]
        if 'Strength_Source' in candidate:
            candidate=candidate.loc[candidate.Strength_Source!='unscaled_auxiliary_shell']
        anchors=peaks.loc[peaks.peak_id.isin(candidate.peak_id)
                          & ~peaks.quality_flags.str.contains('broad_structure')]
        for anchor in anchors.itertuples(index=False):
            radius=2*float(sigma_kev(anchor.energy_keV,config))*fit.resolution_scale
            roi=abs(energy-anchor.energy_keV)<=radius
            local_info=float(np.sqrt(np.sum(component[roi]**2/fit.model[roi])))
            if local_info>=config.candidate_threshold_sigma:
                anchor_points.append(float(anchor.energy_keV))
    anchored=not config.fit_response or (len(set(anchor_points))>=2 and
              (not fit.gain_fitted or np.ptp(anchor_points)>=1.))
    result['Calibration_Anchored']=bool(anchored)
    if not anchored:result['Reasons'].append('A lone Kalpha cannot identify its own free calibration')
    severe = (profile_response and getattr(fit, 'deviance', 0.)/max(len(counts)-getattr(fit, 'n_parameters', 0), 1)
              > config.max_deviance_per_dof)
    if severe:
        narrow = _refined_alpha_profile(symbol,alpha,lines,counts,energy,edges,fit,config,4.)
        wide = _refined_alpha_profile(symbol,alpha,lines,counts,energy,edges,fit,config,6.)
    else:
        narrow=_alpha_profile(alpha,counts,energy,edges,fit,index,config,4.)
        wide=_alpha_profile(alpha,counts,energy,edges,fit,index,config,6.)
    result['Local_Response_And_Interference_Profiled'] = severe
    if narrow is None or wide is None:
        result['Reasons'].append('Kalpha profile/background fitting failed');return result
    min_delta=min(narrow['delta'],wide['delta'])
    result.update(Kalpha_DeltaDeviance=narrow['delta'],Wide_Kalpha_DeltaDeviance=wide['delta'],
                  Kalpha_Area_Counts=narrow['area'])
    if severe:
        result['Local_Kalpha_Profiles'] = {'Narrow': narrow, 'Wide': wide}
    strong=min_delta>=config.low_z_kalpha_min_delta
    if not strong:result['Reasons'].append('Insufficient robust Kalpha evidence')
    # The wider continuum can include an acquisition-edge distortion or a
    # remote unmodelled structure. Require a good local fit and consistent
    # positive areas in both windows, rather than interpreting that remote
    # mismatch as absence of an otherwise well measured principal peak.
    area_consistent = not severe or abs(narrow['area']-wide['area']) <= max(
        .25*max(narrow['area'], wide['area']),
        3*np.hypot(narrow['area_std'], wide['area_std']))
    # Overdispersion is a model/quantitative warning, not proof that a stable
    # principal peak is absent. Its effect on area errors is retained above
    # and by local quantification. Do not reinstate a whole-window quality veto
    # on identity after measuring positive evidence in both windows.
    result['Local_Window_Shape_Mismatch'] = bool(severe and any(
        p['deviance_per_dof'] > config.max_deviance_per_dof for p in (narrow, wide)))
    shape_reliable = not severe or (area_consistent and
        all(not p['response_at_boundary'] for p in (narrow, wide)))
    if not shape_reliable:
        result['Reasons'].append('Local Kalpha areas are unstable or response reaches bounds')
    # Do not use single-region support if another fitted element contributes a
    # comparable amount within the Kalpha core. The multi-line route is unchanged.
    core=np.zeros(len(energy),bool)
    for l in alpha:
        core |= abs(energy-line_model_center(l,fit))<2*float(sigma_kev(l.energy_keV,config))*fit.resolution_scale
    own_signal=float(fit.components[core,index].sum())
    rival_signal=max([float(fit.components[core,j].sum()) for j in range(len(fit.symbols)) if j!=index]+[0.])
    share=own_signal/max(own_signal+rival_signal,1e-20)
    unique=share>=config.low_z_min_kalpha_share
    result['Kalpha_Share_vs_Strongest_Interferent']=share
    if not unique:result['Reasons'].append('Kalpha has a comparable fitted elemental interference')
    beta=[l for l in own if l.branch.startswith('K') and l not in alpha]
    predicted=np.zeros(len(energy));current=np.zeros(len(energy))
    q_alpha=.5*(narrow['area']/narrow['S']+wide['area']/wide['S'])
    for l in beta:
        response=gaussian_response(edges,line_model_center(l,fit),float(sigma_kev(l.energy_keV,config))*fit.resolution_scale)
        predicted+=q_alpha*l.sensitivity_cm2_g*response
        current+=fit.q[index]*l.sensitivity_cm2_g*response
    score=float(np.sqrt(np.sum(predicted**2/np.maximum(fit.model-current+predicted,1e-12))))
    beta_rows=[d for d in branch_details if d['Family']=='K' and 'Kalpha' not in d['Branches']]
    # A deletion test measures independent evidence, not absence. A beta hidden
    # under another element may have zero independent gain despite being there.
    # Claim a strong contradiction only if even the *raw* count budget cannot
    # accommodate its predicted beta area, without subtracting a fixed rival.
    beta_mask = np.zeros(len(energy), dtype=bool)
    for line in beta:
        beta_mask |= abs(energy-line_model_center(line,fit)) <= 2.5*float(sigma_kev(line.energy_keV,config))*fit.resolution_scale
    raw_beta = float(counts[beta_mask].sum())
    raw_budget = raw_beta+config.low_z_beta_visibility_sigma*np.sqrt(raw_beta+1.)
    expected_beta = float(predicted[beta_mask].sum())
    clearly_missing=bool(expected_beta > raw_budget and score>=config.low_z_beta_visibility_sigma and beta_rows and all(
        not d['Group_Observed'] and not d.get('Robust_Group_Observed',False)
        and d['Tied_vs_Free_DeltaDeviance']>=9 for d in beta_rows))
    result.update(Expected_Kbeta_Information_Score=score,Kbeta_Strongly_Contradicted=clearly_missing,
                  Expected_Kbeta_Counts_In_Observed_Region=expected_beta, Raw_Kbeta_Count_Budget=raw_budget,
                  Kbeta_Independently_Required=False)
    if clearly_missing:result['Reasons'].append('Kbeta should be detectable but its absence contradicts the family model')
    result['Eligible']=bool(anchored and strong and unique and shape_reliable and not clearly_missing)
    return result
