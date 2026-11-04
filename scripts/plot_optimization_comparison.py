#!/usr/bin/env python3
"""Raw-count and fitted-spectrum comparison for two recovered local spectra."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'results/optimization_20261104'
NAMES=['shafa1_50kv_20uA.mca','shou_50keV_20uA_back3mm.mca']
fig,axes=plt.subplots(2,2,figsize=(13,7.2),gridspec_kw={'width_ratios':[3,2]},constrained_layout=True)
for row,name in enumerate(NAMES):
    before_path=ROOT/'XRF_Analyze_results/spectra'/name
    after_path=REVIEW/'batch/spectra'/name
    before=pd.read_csv(before_path/'fit_spectrum.csv')
    after=pd.read_csv(after_path/'fit_spectrum.csv')
    assert np.array_equal(before.channel,after.channel)
    assert np.array_equal(before.observed_counts,after.observed_counts)
    old_summary=json.loads((before_path/'fit_summary.json').read_text())
    summary=json.loads((after_path/'fit_summary.json').read_text())
    assert old_summary['input_files']['sample_spectrum']['sha256']==summary['input_files']['sample_spectrum']['sha256']
    candidates=pd.read_csv(after_path/'element_candidates.csv')
    selected=','.join(candidates.loc[candidates.Status=='supported','Element'])
    left,right=axes[row]
    for ax,xlim in [(left,(3,16)),(right,(3.3,4.35) if row==0 else (9.5,13.3))]:
        ax.step(after.energy_keV,after.observed_counts,where='mid',color='#343a40',lw=1,label='Observed raw counts',zorder=3)
        ax.plot(before.energy_keV,before.model_total_counts,color='#d17636',lw=1.3,alpha=.85,label='Before')
        ax.plot(after.energy_keV,after.model_total_counts,color='#176c9e',lw=1.4,label='After')
        ax.set_xlim(*xlim)
        if ax is right:
            visible=(after.energy_keV>=xlim[0]) & (after.energy_keV<=xlim[1])
            shown=max(after.loc[visible,'observed_counts'].max(),after.loc[visible,'model_total_counts'].max(),
                      before.loc[visible,'model_total_counts'].max())
            ax.set_ylim(0.,1.15*shown)
        ax.set_xlabel('Energy on supplied calibration (keV)')
        ax.set_ylabel('Counts / channel')
        ax.grid(alpha=.15)
    left.set_yscale('log')
    left.set_ylim(bottom=1.)
    left.set_title(name+'\nAfter: '+selected,loc='left',fontsize=10)
    right.set_title('Ca primary region' if row==0 else 'Hg / Pb primary complex',loc='left',fontsize=10)
axes[0,0].legend(loc='upper left',fontsize=8)
fig.suptitle('Candidate recovery and response consistency | same raw spectra',fontsize=13)
fig.savefig(REVIEW/'comparison.png',dpi=180)
plt.close(fig)
print(REVIEW/'comparison.png')
