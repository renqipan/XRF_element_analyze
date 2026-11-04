"""Known-truth regressions for candidate recall and independent identity evidence."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.branch_evidence import evaluate_branch_evidence
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.element_identification import identify_elements
from xrf_analyzer.pattern_matching import discover_elements
from xrf_analyzer.peak_detection import detect_peaks
from xrf_analyzer.pipeline import analyze
from xrf_analyzer.response import gaussian_response, sigma_kev

ROOT = Path(__file__).resolve().parents[1]


def test_greedy_coverage_keeps_multiline_competitor_and_retains_ambiguity():
    cfg = AnalysisConfig(fit_response=False, complete_atomic_lines=False)
    edges = np.arange(5., 8.001, .02)
    energy = (edges[:-1]+edges[1:])/2
    # Deliberately indistinguishable identities: greedy coverage may propose
    # one first, but that is not evidence that the alternative is absent.
    lines = [LineCandidate(z, symbol, branch, center, strength)
             for z,symbol in [(26,'Fe'),(27,'Co')]
             for branch,center,strength in [('Kalpha1',6.4,8.),('Kbeta1',7.05,2.)]]
    own = sum(300*l.sensitivity_cm2_g*gaussian_response(
        edges,l.energy_keV,float(sigma_kev(l.energy_keV,cfg))) for l in lines if l.symbol=='Fe')
    counts = np.random.default_rng(704).poisson(8.+own)
    peaks = detect_peaks(np.arange(len(energy)),energy,edges,counts,np.full(len(energy),8.),cfg)
    proposed = discover_elements(peaks,lines,[],energy,counts,cfg)
    assert proposed.selected == {'Fe','Co'}
    assert proposed.alignment['retained_multiregion_alternatives']
    identification = identify_elements(counts,energy,edges,np.full(len(energy),8.),lines,peaks,cfg,.02)
    assert set(identification.candidates.Status) == {'unidentifiable'}


@pytest.mark.parametrize('many_groups',[False,True])
def test_wide_family_is_tested_and_background_shape_does_not_invent_lines(many_groups):
    cfg = AnalysisConfig(fit_response=False)
    edges = np.arange(4., 15.001, .025)
    energy = (edges[:-1]+edges[1:])/2
    centers = np.linspace(5.,13.,9) if many_groups else np.array([5.,10.])
    lines = [LineCandidate(80,'Hg','Lalpha1' if i==0 else 'Lbranch%d'%i,float(center),1.)
             for i,center in enumerate(centers)]
    component = sum(600.*gaussian_response(edges,c,float(sigma_kev(c,cfg))) for c in centers)
    background = np.exp(2.+.4*np.sin(energy/3.))
    fit = SimpleNamespace(symbols=['Hg'],success=True,delta=0.,energy_gain=0.,
        resolution_scale=1.,line_centers={},center_axis='reference',
        model=background+component,background=background,components=component[:,None])
    counts = np.random.default_rng(570).poisson(background+component)
    rows = evaluate_branch_evidence('Hg',lines,energy,edges,counts,fit,cfg)
    assert len(rows)==len(lines)
    assert all(row['Background_Method'].startswith('piecewise_log_linear') for row in rows)
    assert all(row['Branch_DeltaDeviance']>25 for row in rows)


def test_minor_branches_without_principal_peak_cannot_confirm_identity():
    cfg = AnalysisConfig(fit_response=False,complete_atomic_lines=False,
                         element_selection='manual',elements=('Tb',))
    edges = np.arange(4.,9.001,.02)
    energy = (edges[:-1]+edges[1:])/2
    lines = [LineCandidate(65,'Tb',branch,center,strength) for branch,center,strength in
             [('Lalpha1',6.272,5.),('Lbeta1',6.98,3.),('L3M1',5.546,1.)]]
    # An unrelated spectrum matches two minor transitions but has no Tb La.
    component = sum(500.*l.sensitivity_cm2_g*gaussian_response(
        edges,l.energy_keV,float(sigma_kev(l.energy_keV,cfg))) for l in lines if l.branch!='Lalpha1')
    counts = np.random.default_rng(512).poisson(10.+component)
    peaks = detect_peaks(np.arange(len(energy)),energy,edges,counts,np.full(len(energy),10.),cfg)
    result = identify_elements(counts,energy,edges,np.full(len(energy),10.),lines,peaks,cfg,.02)
    row = result.candidates.set_index('Element').loc['Tb']
    assert row.DeltaDeviance > 25
    assert row.Status != 'supported'
    assert 'No precise observed principal alpha group' in row.ConfidenceNotes


def test_real_ca_local_evidence_survives_global_branch_mismatch():
    path = ROOT/'XRF_SDD/shafa1_50kv_20uA.mca'
    if not path.exists():
        pytest.skip('Optional local measurement was removed')
    result = analyze(path,ROOT/'data/calibration.csv',ROOT/'data/xrf_relative_sensitivity.csv',
                     config=AnalysisConfig(energy_range=(1.,40.),tube_target='Ag'))
    row = result['identification'].candidates.set_index('Element').loc['Ca']
    evidence = next(d for d in result['summary']['low_z_kalpha_evidence'] if d['Element']=='Ca')
    assert evidence['Eligible']
    assert min(evidence['Kalpha_DeltaDeviance'],evidence['Wide_Kalpha_DeltaDeviance'])>25
    assert row.Status=='supported'
    assert result['density'].set_index('Element').loc['Ca','q_Area']>0


def test_unknown_cross_shell_population_is_not_within_shell_contradiction():
    cfg = AnalysisConfig(fit_response=False)
    edges = np.arange(4.,12.001,.02)
    energy = (edges[:-1]+edges[1:])/2
    lines = [LineCandidate(80,'Hg','Lalpha1',5.,2.,parent_shell='L3'),
             LineCandidate(80,'Hg','L3N1',7.,1.,parent_shell='L3'),
             LineCandidate(80,'Hg','Lbeta1',10.,1.,parent_shell='L2')]
    component = sum((300. if l.parent_shell=='L3' else 3000.)*l.sensitivity_cm2_g*
        gaussian_response(edges,l.energy_keV,float(sigma_kev(l.energy_keV,cfg))) for l in lines)
    background = np.full(len(energy),10.)
    fit = SimpleNamespace(symbols=['Hg'],success=True,delta=0.,energy_gain=0.,
        resolution_scale=1.,line_centers={},center_axis='reference',
        model=background+component,background=background,components=component[:,None])
    counts = np.random.default_rng(870).poisson(background+component)
    rows = evaluate_branch_evidence('Hg',lines,energy,edges,counts,fit,cfg)
    assert len(rows)==3
    assert rows[0]['Tied_vs_Free_DeltaDeviance']>25
    assert rows[0]['Within_Initial_Shell_DeltaDeviance']<9


def test_stable_fe_identity_is_not_vetoed_by_local_background_overdispersion():
    import json
    path = ROOT/'XRF_SDD/shou_50keV_20uA_back3mm.mca'
    if not path.exists():
        pytest.skip('Optional local measurement was removed')
    cfg = AnalysisConfig(**json.loads((ROOT/'configs/default.json').read_text()))
    result = analyze(path, ROOT/'data/calibration.csv', ROOT/'data/xrf_relative_sensitivity.csv', config=cfg)
    evidence = next(d for d in result['summary']['low_z_kalpha_evidence'] if d['Element']=='Fe')
    assert evidence['Local_Window_Shape_Mismatch']
    assert min(evidence['Kalpha_DeltaDeviance'], evidence['Wide_Kalpha_DeltaDeviance']) > 25
    assert evidence['Eligible']
    assert result['identification'].candidates.set_index('Element').loc['Fe', 'Status']=='supported'
    assert not result['summary']['global_model_valid']
    assert 'Fe' in result['density'].Element.tolist()


def test_empirical_resolution_updates_broad_flags_without_changing_measured_peaks():
    import pandas as pd
    from xrf_analyzer.measurement import reclassify_peak_widths
    peaks = pd.DataFrame([
        dict(peak_id=1,energy_keV=3.785,estimated_fwhm_keV=.297,
             quality_flags='poisson_fitted_centroid;broad_structure'),
        dict(peak_id=2,energy_keV=20.,estimated_fwhm_keV=1.3,
             quality_flags='poisson_fitted_centroid;broad_structure;local_shape_mismatch_or_unresolved_multiplet')])
    original = peaks.copy(deep=True)
    updated = reclassify_peak_widths(peaks,AnalysisConfig(fwhm_ref_kev=.26))
    assert 'broad_structure' not in updated.loc[0,'quality_flags']
    assert 'broad_structure' in updated.loc[1,'quality_flags']
    assert 'local_shape_mismatch' in updated.loc[1,'quality_flags']
    pd.testing.assert_frame_equal(peaks,original)
    pd.testing.assert_frame_equal(updated.drop(columns='quality_flags'),original.drop(columns='quality_flags'))


def test_branch_hidden_in_strong_interferent_does_not_borrow_its_fixed_amplitude():
    cfg=AnalysisConfig(fit_response=False)
    edges=np.arange(4.,7.501,.02)
    energy=(edges[:-1]+edges[1:])/2
    lines=[LineCandidate(80,'Hg','Lalpha1',5.,1.,parent_shell='L3'),
           LineCandidate(80,'Hg','Lbeta1',6.4,1.5,parent_shell='L2')]
    first=600.*gaussian_response(edges,5.,float(sigma_kev(5.,cfg)))
    shared=gaussian_response(edges,6.4,float(sigma_kev(6.4,cfg)))
    background=np.full(len(energy),10.)
    # The joint fit allocated some indistinguishable Fe counts to Hg beta.
    # A local beta test must allow Fe to take those counts back under the null.
    components=np.column_stack([first+900.*shared,5100.*shared])
    fit=SimpleNamespace(symbols=['Hg','Fe'],success=True,delta=0.,energy_gain=0.,
        resolution_scale=1.,line_centers={},center_axis='reference',
        model=background+components.sum(axis=1),background=background,components=components)
    counts=np.random.default_rng(918).poisson(fit.model)
    evidence=evaluate_branch_evidence('Hg',lines,energy,edges,counts,fit,cfg)
    alpha=next(row for row in evidence if row['Branches']=='Lalpha1')
    beta=next(row for row in evidence if row['Branches']=='Lbeta1')
    assert alpha['Branch_DeltaDeviance']>25
    assert beta['Branch_DeltaDeviance']<1
    assert not beta['Group_Observed']
    assert 'interference_profiled' in beta['Method']
