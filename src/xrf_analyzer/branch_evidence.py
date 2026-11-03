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
        if len(groups) < 2 or len(groups) > 8:
            continue
        low = min(line_model_center(l,fit)
                  -roi_sigma*float(sigma_kev(l.energy_keV,config))*fit.resolution_scale for l in selected)
        high = max(line_model_center(l,fit)
                   +roi_sigma*float(sigma_kev(l.energy_keV,config))*fit.resolution_scale for l in selected)
        if high-low > 3.5:  # A local background is not valid across an entire wide L complex.
            continue
        ids=np.flatnonzero((energy>=low)&(energy<=high))
        if len(ids)<max(10,len(groups)+4):
            continue
        a,b=int(ids[0]),int(ids[-1]+1); e=energy[a:b]; y=counts[a:b]
        local_edges=edges[a:b+1]; h=np.column_stack([np.ones(len(e)),(e-e.mean())/max(np.ptp(e),1e-9)])
        basis=[]; strengths=[]
        for group in groups:
            p=np.zeros(len(e)); strength=0.
            for l in group['lines']:
                center=line_model_center(l,fit)
                p+=l.sensitivity_cm2_g*gaussian_response(local_edges,center,float(sigma_kev(l.energy_keV,config))*fit.resolution_scale)
                strength+=l.sensitivity_cm2_g
            basis.append(p/max(strength,1e-20));strengths.append(strength)
        matrix=np.column_stack(basis); strengths=np.array(strengths)
        other=np.maximum(fit.model[a:b]-fit.background[a:b]-fit.components[a:b,fit.symbols.index(symbol)],0)
        reference=getattr(fit,'background_observed_counts',None)
        reference=reference[a:b] if reference is not None else None
        alpha=getattr(fit,'background_scale',0.) or 0.
        ng=len(groups); scale=max(y.sum()/ng,10.)
        def solve(tied=False, missing=None):
            design=(matrix@strengths)[:,None] if tied else matrix
            if tied: design/=strengths.sum()
            na=design.shape[1]
            def obj(v):
                bg=np.exp(h@v[-2:]);mu=np.maximum(other+bg+design@(v[:na]*scale),1e-12)
                r=1-y/mu
                loss=float(np.sum(mu-xlogy(y,mu)))
                if reference is not None:
                    from .measured_background import profile_background
                    prof=profile_background(y,mu,alpha,reference)
                    loss=prof['nll'];r=prof['signal_gradient']
                return loss,np.r_[scale*design.T@r,(bg[:,None]*h).T@r]
            start=np.r_[np.full(na,.5),np.log(max(np.percentile(y,30),.2)),0.]
            bounds=[(0,None)]*na+[(-8,15),(-4,4)]
            if missing is not None:bounds[missing]=(0.,0.);start[missing]=0
            opt=minimize(obj,start,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'maxls':40})
            return opt
        full=solve();tied=solve(True)
        if not full.success or not tied.success:
            continue
        areas=full.x[:ng]*scale
        anchor=int(np.argmax(areas))
        for i,group in enumerate(groups):
            null=solve(missing=i)
            gain=max(0.,2*(null.fun-full.fun)) if null.success else None
            results.append({'Element':symbol,'Family':family,'Branches':'+'.join(l.branch for l in group['lines']),
                            'Background_Window_Sigmas':roi_sigma,'ROI_Start_keV':float(local_edges[0]),'ROI_End_keV':float(local_edges[-1]),
                            'Local_Free_Area_Counts':float(areas[i]),'Expected_Area_From_Tied_Counts':float(tied.x[0]*scale*strengths[i]/strengths.sum()),
                            'Branch_DeltaDeviance':gain,
                            'Observed_Ratio_to_Strongest_Group':float(areas[i]/areas[anchor]) if areas[anchor]>0 else None,
                            'Expected_Ratio_to_Strongest_Group':float(strengths[i]/strengths[anchor]),'Tied_vs_Free_DeltaDeviance':max(0.,2*(tied.fun-full.fun)),
                            'Group_Observed':bool(gain is not None and gain>=config.tentative_delta_deviance),
                            'Method':'local_Poisson_background_refit_other_elements_fixed; conditional_not_global_p_value'})
    return results
