import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .response import line_model_center


def make_plots(result, out):
    fit, e, y = result['identification'].fit, result['energy'], result['counts']
    fig, (ax, res) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)
    ax.step(e, y, where='mid', color='0.45', lw=.8, label='Observed raw counts')
    ax.plot(e, fit.model, color='black', lw=1.2, label='Total Poisson model')
    ax.plot(e, result['background0'], '--', color='tab:orange', lw=.9, label='SNIP initial background')
    ax.plot(e, fit.background, ':', color='tab:brown', lw=1.3, label='Fitted continuous background')
    if getattr(fit,'background_scale',None) is not None:
        ax.plot(e,fit.measured_background_counts,'--',color='tab:purple',lw=.9,
                label='Measured background (scale=%.4g)' % fit.background_scale)
    if np.any(fit.scatter > 0):
        ax.plot(e, fit.scatter, '-.', color='tab:gray', lw=1., label='Instrument / scattering')
    candidates = result['identification'].candidates
    colors = plt.get_cmap('tab20')
    display = sorted(range(len(fit.symbols)), key=lambda i: fit.t[i], reverse=True)[:8]
    for rank, i in enumerate(display):
        symbol = fit.symbols[i]
        status = candidates.loc[candidates.Element == symbol, 'Status'].iloc[0]
        if status == 'not_supported' or fit.t[i] < max(5, y.sum()*1e-4):
            continue
        color = colors(rank % 20)
        style = '-' if status == 'supported' else '--'
        ax.plot(e, np.maximum(fit.components[:, i], 1e-5), lw=.9, linestyle=style, color=color, label='%s (%s)' % (symbol, status))
        strongest = sorted([l for l in result['identification'].fit_lines if l.symbol == symbol],
                           key=lambda l: l.sensitivity_cm2_g, reverse=True)[:2]
        for l in strongest:
            ax.axvline(line_model_center(l,fit), color=color, alpha=.25, lw=.6, linestyle=':')
    for _, peak in result['peaks'].iterrows():
        pairs = result['identification'].pairings
        confirmed = set(candidates.loc[candidates.Status == 'supported', 'Element'])
        names = sorted(set(pairs.loc[(pairs.peak_id == peak.peak_id) & pairs.Element.isin(confirmed), 'Element']))
        if names and peak.approx_significance >= 10:
            center=peak.energy_keV
            if result['requested_config'].find_peaks is not None and result['requested_config'].find_peaks_axis=='reference':
                center=center*(1+fit.energy_gain)+fit.delta
            pos = int(np.argmin(abs(e-center)))
            contributions = {s: fit.components[pos, fit.symbols.index(s)] for s in names if s in fit.symbols}
            maximum = max(contributions.values(), default=0.)
            names = sorted((s for s in contributions if contributions[s] >= .1*maximum and maximum > 0), key=lambda s: contributions[s], reverse=True)
            if not names:
                continue
            ax.annotate('/'.join(names), (e[pos], max(y[pos], 1)), xytext=(0, 12),
                        textcoords='offset points', ha='center', fontsize=8)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(bottom=0, top=max(3., float(max(y.max(), fit.model.max()))*2.5))
    ax.set_ylabel('Counts / channel')
    quality = result['summary']['analysis_status']
    ax.set_title('XRF joint spectrum fit | apparent thin-sample model | ' + quality.upper())
    ax.text(.01, .02, 'Up to 8 largest candidate components shown; all components in CSV',
            transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    residual = (y-fit.model)/np.sqrt(fit.model)
    res.axhline(0, color='black', lw=.6)
    res.axhspan(-3, 3, color='0.9')
    res.plot(e, residual, lw=.7)
    res.set_ylabel('Pearson residual')
    res.set_xlabel('Energy (keV)')
    fig.savefig(out/'spectrum_fit.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
    ax.plot(e, residual, lw=.7)
    ax.axhline(0, color='black', lw=.6)
    ax.set(xlabel='Energy (keV)', ylabel='Pearson residual', title='Signed residuals of original counts')
    fig.savefig(out/'residuals.png', dpi=140)
    plt.close(fig)
    cal = result['calibration']
    channels = np.array([p['channel'] for p in cal.points])
    energies = np.array([p['energy_keV'] for p in cal.points])
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    ax.scatter(channels, energies, label='Calibration points')
    ax.plot(channels, cal.energy(channels), label='Linear calibration')
    ax.set_ylabel('Energy (keV)')
    ax.legend()
    rax.axhline(0, color='black', lw=.6)
    rax.scatter(channels, cal.residuals_keV)
    rax.set(xlabel='Channel', ylabel='Residual (keV)')
    fig.savefig(out/'calibration.png', dpi=140)
    plt.close(fig)


def make_calcium_plot(result,out):
    fit=result['identification'].fit
    if 'Ca' not in fit.symbols:
        (out/'calcium_region.png').unlink(missing_ok=True)
        return
    e,y=result['energy'],result['counts']; roi=(e>=3.25)&(e<=4.65)
    if roi.sum()<5:return
    c=result['identification'].candidates.set_index('Element').loc['Ca']
    fig,(ax,res)=plt.subplots(2,1,figsize=(9,6),sharex=True,gridspec_kw={'height_ratios':[3,1]},constrained_layout=True)
    ax.step(e[roi],y[roi],where='mid',color='.35',label='Observed raw counts')
    ax.plot(e[roi],fit.model[roi],color='black',label='Full model')
    ax.plot(e[roi],fit.background[roi],':',label='Full-model background')
    ax.plot(e[roi],fit.components[roi,fit.symbols.index('Ca')],color='tab:blue',label='Ca model contribution')
    for family,branch in [('K alpha','Kalpha1'),('K beta','Kbeta1')]:
        lines=[l for l in result['identification'].fit_lines if l.symbol=='Ca' and l.branch==branch]
        if lines:
            center=line_model_center(lines[0],fit)
            ax.axvline(center,color='tab:blue',alpha=.45,linestyle='--')
            ax.text(center,.9*max(y[roi]),family,rotation=90,va='top',ha='right',fontsize=9)
    ax.set_title('Calcium low-energy evidence | identity: '+c.Status+' | density: '+c.Quantification_Status)
    ax.set_ylabel('Counts / channel');ax.set_ylim(bottom=0);ax.legend(fontsize=8)
    residual=(y-fit.model)/np.sqrt(fit.model)
    res.plot(e[roi],residual[roi]);res.axhline(0,color='black',lw=.6)
    res.set(xlabel='Calibrated observed energy (keV)',ylabel='Pearson residual')
    fig.savefig(out/'calcium_region.png',dpi=160);plt.close(fig)


def make_quantification_plots(result,out):
    local=result.get('local_quantification')
    fits=local.get('fits',[]) if local is not None else []
    if not fits:
        (out/'quantification_regions.png').unlink(missing_ok=True)
        return
    ncols=min(2,len(fits));nrows=(len(fits)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(7*ncols,4*nrows),squeeze=False,constrained_layout=True)
    for ax,profile in zip(axes.ravel(),fits):
        ids=np.asarray(profile['indices'],dtype=int)
        energy=result['energy'][ids]
        symbol=profile['Element']
        row=local['elements'].set_index('Element').loc[symbol]
        ax.step(energy,result['counts'][ids],where='mid',color='.4',label='Observed raw counts')
        ax.plot(energy,profile['model'],color='black',lw=1.1,label='Local Poisson fit')
        ax.plot(energy,profile['primary_counts'],color='tab:blue',label='Primary peak group')
        ax.plot(energy,profile['background'],':',color='tab:brown',label='Local background')
        ax.plot(energy,profile['nuisance_counts'],'--',color='tab:orange',label='Neighbouring groups')
        ax.set(title='%s: %s\nA/S = %.5g +/- %.2g' % (symbol,row.Selected_Branches,row.q_Area,row.q_Std),
               xlabel='Energy (keV)',ylabel='Counts / channel',ylim=(0,None))
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(fits):]:ax.set_visible(False)
    fig.suptitle('Primary peak area / sensitivity | apparent relative density',fontsize=13)
    fig.savefig(out/'quantification_regions.png',dpi=150)
    plt.close(fig)


def make_elements_plot(result,out):
    """Observed raw counts with supported elemental labels only."""
    from pathlib import Path
    e,y=result['energy'],result['counts']
    ident=result['identification'];fit=ident.fit
    supported=set(ident.candidates.loc[ident.candidates.Status=='supported','Element'])
    fig,ax=plt.subplots(figsize=(13,6),constrained_layout=True)
    ax.step(e,y,where='mid',color='.25',lw=.9,label='Observed raw count')
    for _,peak in result['peaks'].iterrows():
        pairs=ident.pairings
        names=set(pairs.loc[(pairs.peak_id==peak.peak_id)&pairs.Element.isin(supported)
                            &(pairs.Group_Affinity>0),'Element'])
        center=peak.energy_keV
        if result['requested_config'].find_peaks is not None and result['requested_config'].find_peaks_axis=='reference':
            center=center*(1+fit.energy_gain)+fit.delta
        pos=int(np.argmin(abs(e-center)))
        contributions={s:float(fit.components[pos,fit.symbols.index(s)]) for s in names if s in fit.symbols}
        maximum=max(contributions.values(),default=0.)
        names=sorted(s for s,v in contributions.items() if v>=.1*max(float(y[pos]),1.) and v>=.1*maximum)
        if names:
            ax.annotate('/'.join(names),(e[pos],max(y[pos],1)),xytext=(0,9),
                        textcoords='offset points',ha='center',fontsize=9)
    ax.set_yscale('symlog',linthresh=1)
    ax.set_ylim(0,max(3.,float(np.max(y))*3.))
    ax.set(xlabel='Energy (keV)',ylabel='Counts / channel')
    ax.legend(loc='upper right')
    name=Path(result['summary']['input_files']['sample_spectrum']['path']).stem
    fig.savefig(out/(name+'_elements.png'),dpi=160)
    plt.close(fig)
