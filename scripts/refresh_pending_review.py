#!/usr/bin/env python3
"""Refresh spectra affected only by the final unsupported-rival review rule.

Run after a completed optimization batch. All input/config hashes must match.
Unchanged spectra reuse their already-verified reports. Production run_batch
regenerates every aggregate and image copy; no reports are fabricated.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from xrf_analyzer import batch
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.reporting import sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/optimization_20261104/batch'
cfg=AnalysisConfig(**json.loads((ROOT/'configs/default.json').read_text()))
manifest=json.loads((OUT/'batch_manifest.json').read_text())
assert manifest['completed'] and manifest['failures']==0
assert manifest['calibration']['sha256']==sha256(ROOT/'data/calibration.csv')
assert manifest['sensitivity']['sha256']==sha256(ROOT/'data/xrf_relative_sensitivity.csv')
original=batch.analyze
cached={}
affected=[]
for spectrum in batch.discover_spectra(ROOT/'XRF_SDD'):
    folder=OUT/'spectra'/spectrum.name
    summary=json.loads((folder/'fit_summary.json').read_text())
    assert summary['input_files']['sample_spectrum']['sha256']==sha256(spectrum)
    assert summary['parameters']==json.loads(json.dumps(cfg.to_dict()))
    candidates=pd.read_csv(folder/'element_candidates.csv')
    pairs=pd.read_csv(folder/'peak_line_matches.csv')
    pairs=pairs[(pairs.Strength_Source!='unscaled_auxiliary_shell') & (pairs.Group_Affinity>=.5)]
    changed=False
    for candidate in candidates.itertuples(index=False):
        potential=(candidate.Status=='not_supported' or 'counterfactual interpretation remains tentative' in str(candidate.ConfidenceNotes))
        if not potential or candidate.Z>cfg.low_z_max_z or pd.isna(candidate.q_Fit):continue
        alpha=pairs[(pairs.Element==candidate.Element)&pairs.Branch.isin(['Kalpha1','Kalpha2'])]
        overlaps=str(candidate.OverlapElements).split(',')
        rivals=candidates[(candidates.Status=='not_supported') & candidates.q_Fit.gt(0) & candidates.Element.isin(overlaps)]
        if not pairs[pairs.Element.isin(rivals.Element)&pairs.peak_id.isin(alpha.peak_id)].empty:
            changed=True
            break
    if changed:
        affected.append(spectrum.name)
    else:
        cached[spectrum.name]=dict(summary=summary,identification=SimpleNamespace(candidates=candidates),
                                 density=pd.read_csv(folder/'relative_areal_density.csv'))

def refresh(spectrum,calibration,sensitivity,output_dir,config):
    if Path(spectrum).name in cached:
        print('Reuse verified unaffected report:',Path(spectrum).name,flush=True)
        return cached[Path(spectrum).name]
    return original(spectrum,calibration,sensitivity,output_dir,config)

write_json(OUT.parent/'pending_refresh_audit.json',dict(
    rule_change='Review overlapping unconfirmed rivals including not_supported with nonzero fitted amplitude',
    reused=list(cached),recomputed=affected,
    input_and_configuration_hashes_verified=True,
    production_aggregation='xrf_analyzer.batch.run_batch'))
batch.analyze=refresh
try:
    result=batch.run_batch(ROOT/'XRF_SDD',OUT,ROOT/'data/calibration.csv',
                           ROOT/'data/xrf_relative_sensitivity.csv',cfg)
finally:
    batch.analyze=original
assert result['completed'] and result['failures']==0
