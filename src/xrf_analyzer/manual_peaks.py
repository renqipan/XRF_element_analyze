"""User-specified centers: no automatic peak discovery, only shared response fitting."""
import json
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import xlogy
from .response import sigma_kev, response_with_derivatives, FWHM_FACTOR
from .peak_detection import PEAK_COLUMNS


def parse_peak_energies(values):
    if values is None:
        return None
    text = ' '.join(values) if isinstance(values, (tuple, list)) else str(values)
    if text.strip().lower() == 'none':
        return None
    if ('[' in text) != (']' in text):
        raise ValueError('find_peaks list has unmatched brackets')
    tokens = re.split(r'[,\s]+', text.strip().strip('[]').strip())
    try:
        result = tuple(float(v) for v in tokens if v)
    except ValueError as exc:
        raise ValueError('find_peaks must contain numeric keV values') from exc
    return result


def requested_peak_groups(config):
    groups=[]
    for value in sorted(config.find_peaks):
        if groups and value-groups[-1][-1] <= .65*FWHM_FACTOR*float(sigma_kev(value,config)):
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def fit_requested_peaks(channel, energy, edges, counts, background, config):
    groups=requested_peak_groups(config)
    centers=np.array([np.mean(g) for g in groups]); sigmas=sigma_kev(centers,config)
    reference=config.find_peaks_axis=='reference'
    bound=config.alignment_max_shift_kev if reference else 0.
    if any(c+bound+4*s < edges[0] or c-bound-4*s > edges[-1] for c,s in zip(centers,sigmas)):
        raise ValueError('A find_peaks center is outside the measured energy range')
    mask=np.any(abs(energy[:,None]-centers[None,:]) < (bound+4*sigmas)[None,:],axis=1)
    y=counts[mask]; b0=np.maximum(background[mask],.1)
    reference=getattr(config,'_background_counts',None)
    reference=reference[mask] if reference is not None else None
    alpha=getattr(config,'_background_scale',1.)
    if reference is not None:b0=np.maximum(b0-alpha*reference,.1)
    x=(energy[mask]-np.mean(energy[mask]))/max(np.ptp(energy[mask]),1e-9)
    h=np.column_stack([np.ones(len(y)),x]); n=len(centers); scale=max(y.sum()/10,10.)
    gain_free=reference and config.fit_energy_gain and np.ptp(centers)>1
    def evaluate(v):
        delta,gain,width=v[n+2:]
        p,d,w=response_with_derivatives(edges,centers*(1+gain),sigmas,delta,width)
        p,d,w=p[mask],d[mask],w[mask]
        a=v[:n]*scale;bg=b0*np.exp(h@v[n:n+2]);mu=np.maximum(bg+p@a,1e-12)
        jac=np.column_stack([p*scale,bg[:,None]*h,d@a,(d*centers[None,:])@a,w@a])
        return mu,jac,p
    def obj(v):
        mu,jac,_=evaluate(v)
        residual=1-y/mu;loss=float(np.sum(mu-xlogy(y,mu)))
        if reference is not None:
            from .measured_background import profile_background
            prof=profile_background(y,mu,alpha,reference);loss=prof['nll'];residual=prof['signal_gradient']
        return loss+.05*np.sum(v[n:n+2]**2),jac.T@residual+np.r_[np.zeros(n),.1*v[n:n+2],np.zeros(3)]
    bounds=[(0,None)]*n+[(-3,3),(-3,3),(-bound,bound),
            (-config.energy_gain_bound,config.energy_gain_bound) if gain_free else (0.,0.),(.5,3.)]
    starts=[]
    for delta in (np.linspace(-bound,bound,11) if reference else [0.]):
        p=response_with_derivatives(edges,centers,sigmas,delta,1.5)[0][mask]
        try:a=nnls(p/np.sqrt(b0+1)[:,None],np.maximum(y-b0,0)/np.sqrt(b0+1))[0]
        except RuntimeError:a=np.full(n,max(y.sum()/n,1.))
        v=np.r_[a/scale,0.,0.,delta,0.,1.5]
        starts.append((obj(v)[0],v))
    solutions=[]
    for _,start in sorted(starts,key=lambda pair:pair[0])[:2]:
        opt=minimize(obj,start,jac=True,bounds=bounds,method='L-BFGS-B',
                     options={'maxiter':1600,'ftol':1e-12,'maxls':40})
        if opt.success:solutions.append(opt)
    if not solutions:
        raise ValueError('Shared-response fitting of user-specified peaks did not converge')
    opt=min(solutions,key=lambda r:r.fun);mu,jac,p=evaluate(opt.x)
    delta,gain,width=opt.x[n+2:];area=opt.x[:n]*scale
    variance=mu.copy()
    if reference is not None:
        from .measured_background import profile_background
        prof=profile_background(y,mu,alpha,reference)
        variance=prof['model']+alpha**2*prof['latent_background']
    rows=[]
    for i,(c,sig,requested) in enumerate(zip(centers,sigmas,groups)):
        observed=c*(1+gain)+delta
        information=float(np.sqrt(np.sum((area[i]*p[:,i])**2/variance)))
        raw_roi=abs(energy-observed)<3*sig*width
        maximum=int(np.flatnonzero(raw_roi)[np.argmax(counts[raw_roi])]) if raw_roi.any() else int(np.argmin(abs(energy-observed)))
        rows.append([i+1,float(channel[0]+(observed-energy[0])/np.median(np.diff(edges))),float(c),0.,
                     float(FWHM_FACTOR*sig*width),information,max(edges[0],observed-3*sig*width),
                     min(edges[-1],observed+3*sig*width),
                     'user_supplied_center; shared_response_only; unresolved_inputs_share_evidence',
                     int(channel[maximum]),float(energy[maximum]),0.,float(area[i]),np.nan,
                     'user_centers_fixed',np.nan])
    peaks=pd.DataFrame(rows,columns=PEAK_COLUMNS)
    peaks['observed_energy_keV']=centers*(1+gain)+delta
    peaks['requested_energies_keV']=[json.dumps(g) for g in groups]
    peaks['energy_axis']=config.find_peaks_axis
    return peaks,{'mode':'user_specified','axis':config.find_peaks_axis,'requested_energies_keV':list(config.find_peaks),
                  'independent_resolution_groups':len(groups),'delta_E_keV':float(delta),
                  'energy_gain':float(gain),'resolution_scale':float(width),
                  'automatic_discovery_used':False,'information_scores_are_global_p_values':False}


def constrain_line_centers(lines, auxiliary, config, alignment_delta=0., alignment_gain=0.):
    if config.find_peaks is None:
        return {}
    requested=np.array(config.find_peaks,float); result={}
    for line in list(lines)+list(auxiliary):
        predicted=line.energy_keV
        if config.find_peaks_axis=='observed':
            predicted=predicted*(1+alignment_gain)+alignment_delta
        i=int(np.argmin(abs(requested-predicted)))
        if abs(requested[i]-predicted)<=config.manual_match_tolerance_kev:
            result[(line.symbol,line.branch)]=float(requested[i])
    return result
