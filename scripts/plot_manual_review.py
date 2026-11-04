#!/usr/bin/env python3
"""Scientific raw-count panels for the second-round identification review."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', required=True)
    p.add_argument('--after', default='XRF_Analyze_results/spectra')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    panels = [
        ('xiongzhen1.mca', 'Ca', (3.25, 4.5), [(3.6905, 'Ca Ka')]),
        ('yifu2_50kv_20uA.mca', 'Ca', (3.25, 4.5), [(3.6905, 'Ca Ka')]),
        ('yifu1_tou_50keV_20uA_back3mm.mca', 'Ca', (3.25, 4.5), [(3.6905, 'Ca Ka')]),
        ('shafa1_50kv_20uA.mca', 'Hg', (9.5, 12.35), [(9.989, 'Hg La'), (11.8238, 'Hg Lb')]),
        ('yifu_50keV_20uA_back3mm.mca', 'Cr', (5.05, 6.3), [(5.4147, 'Cr Ka'), (5.4324, 'Pm La')]),
        ('shu_50keV_20uA_back3mm.mca', 'Fe', (6., 7.5), [(6.4039, 'Fe Ka'), (6.2728, 'Tb La')])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (name, symbol, limits, markers) in zip(axes.flat, panels):
        old_dir, new_dir = Path(args.before)/name, Path(args.after)/name
        old = pd.read_csv(old_dir/'fit_spectrum.csv')
        new = pd.read_csv(new_dir/'fit_spectrum.csv')
        old_status = pd.read_csv(old_dir/'element_candidates.csv').set_index('Element').loc[symbol, 'Status']
        new_status = pd.read_csv(new_dir/'element_candidates.csv').set_index('Element').loc[symbol, 'Status']
        s = json.loads((new_dir/'fit_summary.json').read_text())
        mask = new.energy_keV.between(*limits)
        old_mask = old.energy_keV.between(*limits)
        ax.step(new.loc[mask, 'energy_keV'], new.loc[mask, 'observed_counts'], where='mid',
                color='.18', lw=1., label='Raw counts')
        ax.plot(old.loc[old_mask, 'energy_keV'], old.loc[old_mask, 'model_total_counts'],
                '--', color='#3182bd', lw=1.2, label='Previous fit')
        ax.plot(new.loc[mask, 'energy_keV'], new.loc[mask, 'model_total_counts'],
                color='#e6550d', lw=1.2, label='Updated fit')
        for i, (center, label) in enumerate(markers):
            center = center*(1+s['energy_gain_correction'])+s['delta_E_keV']
            ax.axvline(center, color=['#31a354', '#756bb1'][i % 2], ls=':', lw=1.)
            ax.text(center, .96-i*.1, label, transform=ax.get_xaxis_transform(),
                    ha='left', va='top', fontsize=8)
        ax.set(xlim=limits, xlabel='Energy (keV)', ylabel='Counts / channel',
               title=name+'\n'+symbol+': '+old_status+' -> '+new_status)
        ax.title.set_fontsize(9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.15)
        if symbol == 'Hg':
            ax.set_yscale('log')
            ax.set_ylim(bottom=max(1., float(new.loc[mask, 'observed_counts'].min())*.65))
        else:
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Raw-count evidence and before/after fits | classifications remain model-conditional', fontsize=13)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
