"""Local raw-count Poisson family/branch evidence, conditional on shared calibration."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from .response import gaussian_response, sigma_kev, FWHM_FACTOR, line_model_center


def evaluate_branch_evidence(symbol, lines, energy, edges, counts, fit, config, roi_sigma=4.):
    own = [l for l in lines if l.symbol == symbol]
    if not own or symbol not in fit.symbols or not fit.success:
        return []
    results = []
    for family in ('K', 'L', 'M'):
        selected = [l for l in own if l.branch.startswith(family)]
        if not selected:
            continue
        # Use separated observable resolution groups rather than single transitions.
        groups = []
        for line in sorted(selected, key=lambda l: l.energy_keV):
            center = line_model_center(line,fit)
            sigma = float(sigma_kev(line.energy_keV, config))*fit.resolution_scale
            if groups and center-groups[-1]['last'] < .65*FWHM_FACTOR*sigma:
                groups[-1]['lines'].append(line); groups[-1]['last'] = center
            else:
                groups.append({'lines': [line], 'last': center})
        if len(groups) < 2:
            continue
        low = min(line_model_center(l,fit)
                  -roi_sigma*float(sigma_kev(l.energy_keV,config))*fit.resolution_scale for l in selected)
        high = max(line_model_center(l,fit)
                   +roi_sigma*float(sigma_kev(l.energy_keV,config))*fit.resolution_scale for l in selected)
        ids=np.flatnonzero((energy>=low)&(energy<=high))
        if len(ids)<max(10,len(groups)+4):
            continue
        a,b=int(ids[0]),int(ids[-1]+1); e=energy[a:b]; y=counts[a:b]
        local_edges=edges[a:b+1]
        # A wide L complex must not be discarded, nor forced onto a single
        # log-linear continuum across several keV. A continuous piecewise
        # log-linear background supplies local slopes without fitting peaks.
        if high-low > 3.5:
            knots=np.linspace(e[0],e[-1],max(3,int(np.ceil((high-low)/1.5))+1))
            h=np.column_stack([np.interp(e,knots,np.eye(len(knots))[j]) for j in range(len(knots))])
            background_method='piecewise_log_linear_1p5keV_knots'
        else:
            h=np.column_stack([np.ones(len(e)),(e-e.mean())/max(np.ptp(e),1e-9)])
            background_method='local_log_linear'
        nb=h.shape[1]
        basis=[]; strengths=[]
        for group in groups:
            p=np.zeros(len(e)); strength=0.
            for l in group['lines']:
                center=line_model_center(l,fit)
                p+=l.sensitivity_cm2_g*gaussian_response(local_edges,center,float(sigma_kev(l.energy_keV,config))*fit.resolution_scale)
                strength+=l.sensitivity_cm2_g
            basis.append(p/max(strength,1e-20));strengths.append(strength)
        matrix=np.column_stack(basis); strengths=np.array(strengths)
        # Competing elements must be profiled too: fixing a strong Fe peak
        # would make its small residual look like independent Pm evidence.
        rivals=[fit.components[a:b,j] for j in range(len(fit.symbols))
                if j != fit.symbols.index(symbol)]
        scatter=getattr(fit,'scatter',None)
        if scatter is not None:rivals.append(np.asarray(scatter)[a:b])
        rivals=[np.maximum(r,0.) for r in rivals if np.sum(r)>max(.001,1e-5*y.sum())]
        rival_areas=np.array([r.sum() for r in rivals])
        nuisance=np.column_stack([r/area for r,area in zip(rivals,rival_areas)]) if rivals else np.zeros((len(e),0))
        reference=getattr(fit,'background_observed_counts',None)
        reference=reference[a:b] if reference is not None else None
        alpha=getattr(fit,'background_scale',0.) or 0.
        ng=len(groups); scale=max(y.sum()/ng,10.)
        def solve(tied=False, missing=None):
            if tied == 'initial_shell':
                # Only compare ratios within one initial vacancy shell. Mixed
                # unresolved groups remain free; no L2/L3 population is invented.
                from .atomic_lines import LINE_SHELL
                buckets={}
                for j,group in enumerate(groups):
                    shells={l.parent_shell or LINE_SHELL.get(l.branch,l.branch[:2]) for l in group['lines']}
                    key=next(iter(shells)) if len(shells)==1 else 'mixed_%d'%j
                    buckets.setdefault(key,[]).append(j)
                columns=[]
                for indices in buckets.values():
                    columns.append(matrix[:,indices]@strengths[indices]/strengths[indices].sum())
                design=np.column_stack(columns)
            else:
                design=(matrix@strengths)[:,None] if tied else matrix
                if tied: design/=strengths.sum()
            na=design.shape[1]
            design=np.column_stack([design,nuisance])
            nt=design.shape[1]
            def obj(v):
                bg=np.exp(h@v[-nb:]);mu=np.maximum(bg+design@(v[:nt]*scale),1e-12)
                r=1-y/mu
                loss=float(np.sum(mu-xlogy(y,mu)))
                if reference is not None:
                    from .measured_background import profile_background
                    prof=profile_background(y,mu,alpha,reference)
                    loss=prof['nll'];r=prof['signal_gradient']
                return loss,np.r_[scale*design.T@r,(bg[:,None]*h).T@r]
            level=np.log(max(np.percentile(y,30),.2))
            beta=np.full(nb,level) if high-low > 3.5 else np.array([level,0.])
            start=np.r_[np.full(na,.5),rival_areas/scale,beta]
            bounds=[(0,None)]*nt+([(-8,15)]*nb if high-low > 3.5 else [(-8,15),(-4,4)])
            if missing is not None:bounds[missing]=(0.,0.);start[missing]=0
            opt=minimize(obj,start,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'maxls':40})
            return opt
        full=solve();tied=solve(True);shell_tied=solve('initial_shell')
        if not full.success or not tied.success:
            continue
        areas=full.x[:ng]*scale
        anchor=int(np.argmax(areas))
        severe=(getattr(fit,'deviance',0.)/max(len(counts)-getattr(fit,'n_parameters',0),1)
                > config.max_deviance_per_dof)
        local_groups=None
        if severe:
            from types import SimpleNamespace
            import pandas as pd
            from .local_quantification import _groups
            provisional=[]
            for j,other_symbol in enumerate(fit.symbols):
                primary=fit.matrix[:,j]*fit.t[j]
                info=float(np.sqrt(np.sum(primary**2/fit.model)))
                anchored=other_symbol in getattr(config,'_branch_confirmed_symbols',set())
                provisional.append({'Element':other_symbol,'Status':'supported' if anchored and
                    info>=np.sqrt(config.support_delta_deviance) else 'tentative'})
            temporary=SimpleNamespace(fit=fit,fit_lines=lines,candidates=pd.DataFrame(provisional))
            local_groups=_groups(temporary,config,edges)
        for i,group in enumerate(groups):
            null=solve(missing=i)
            gain=max(0.,2*(null.fun-full.fun)) if null.success else None
            conditional_gain=gain
            local_failure=None
            shape_profiled=False
            core=np.zeros(len(e),bool)
            for line in group['lines']:
                core |= abs(e-line_model_center(line,fit))<2*float(sigma_kev(line.energy_keV,config))*fit.resolution_scale
            own_core=float((matrix[core,i]*areas[i]).sum())
            rival_core=float((nuisance[core]@(full.x[ng:ng+len(rivals)]*scale)).sum())
            # Spend local response refits on contested lines. This trigger only
            # controls extra work, never rejects a weak/trace branch by share.
            contested=rival_core>=max(1.,.1*own_core)
            if severe and contested and gain is not None and gain>=config.tentative_delta_deviance:
                shape_profiled=True
                # A misspecified global response can manufacture a weak line
                # beside Fe. Reuse the quantitative local model: resolved strong
                # interferents, background, and a common local shift/width all
                # refit under both hypotheses. Unconfirmed rivals keep their
                # same-shell constraints, so faint hypothetical lines cannot
                # freely swallow a measured primary.
                from .local_quantification import _profile
                target_lines=group['lines']
                target_s=sum(l.sensitivity_cm2_g for l in target_lines)
                target={'Element':symbol,'Z':target_lines[0].Z,'lines':target_lines,
                    'weights':np.array([l.sensitivity_cm2_g/target_s for l in target_lines]),
                    'centers':np.array([line_model_center(l,fit) for l in target_lines]),
                    'sigmas':np.array([float(sigma_kev(l.energy_keV,config))*fit.resolution_scale for l in target_lines]),
                    'S':target_s,'expected':fit.q[fit.symbols.index(symbol)]*target_s,
                    'confirmed':True,'auxiliary':False,'Branches':'+'.join(l.branch for l in target_lines)}
                branches={l.branch for l in target_lines}
                neighbours=[g for g in local_groups if not (g['Element']==symbol and
                    any(l.branch in branches for l in g['lines']))]
                neighbours=[dict(g,confirmed=True) if g['Element']==symbol and not g['auxiliary'] else g
                            for g in neighbours]
                profiled=_profile(target,[target]+neighbours,counts,energy,edges,config,roi_sigma,
                                  background_scale=getattr(fit,'background_scale',0.) or 0.)
                gain=profiled.get('DeltaDeviance') if profiled['Success'] else None
                local_failure=profiled.get('Failure')
            results.append({'Element':symbol,'Family':family,'Branches':'+'.join(l.branch for l in group['lines']),
                            'Background_Window_Sigmas':roi_sigma,'Background_Method':background_method,
                            'Background_Parameters':nb,'ROI_Start_keV':float(local_edges[0]),'ROI_End_keV':float(local_edges[-1]),
                            'Local_Free_Area_Counts':float(areas[i]),'Expected_Area_From_Tied_Counts':float(tied.x[0]*scale*strengths[i]/strengths.sum()),
                            'Branch_DeltaDeviance':gain,
                            'Observed_Ratio_to_Strongest_Group':float(areas[i]/areas[anchor]) if areas[anchor]>0 else None,
                            'Expected_Ratio_to_Strongest_Group':float(strengths[i]/strengths[anchor]),'Tied_vs_Free_DeltaDeviance':max(0.,2*(tied.fun-full.fun)),
                            'Within_Initial_Shell_DeltaDeviance':max(0.,2*(shell_tied.fun-full.fun)) if shell_tied.success else None,
                            'Group_Observed':bool(gain is not None and gain>=config.tentative_delta_deviance),
                            'Conditional_Response_DeltaDeviance':conditional_gain,
                            'Local_Response_Profiled':shape_profiled, 'Local_Profile_Failure':local_failure,
                            'Method':'local_Poisson_background_and_elemental_interference_profiled; conditional_not_global_p_value'})
    return results
