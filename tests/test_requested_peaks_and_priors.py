from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.manual_peaks import parse_peak_energies, requested_peak_groups, fit_requested_peaks
from xrf_analyzer.element_prior import SYMBOLS, classification_table, choose_by_evidence, element_class
from xrf_analyzer.pipeline import analyze
from xrf_analyzer.cli import main
from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.element_identification import identify_elements
from xrf_analyzer.peak_detection import detect_peaks
from xrf_analyzer.response import gaussian_response, sigma_kev

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('values', [['[3.69,','3.688,','6.4]'],['3.69','3.688','6.4'],['[3.69, 3.688, 6.4]']])
def test_manual_parser_accepts_cli_forms(values):
    assert parse_peak_energies(values)==(3.69,3.688,6.4)
    assert parse_peak_energies(['None']) is None


@pytest.mark.parametrize('values',[(),(0.,),(-1.,),(np.nan,),(np.inf,),(6.4,6.4),(True,)])
def test_manual_invalid_centers_fail_early(values):
    with pytest.raises(ValueError):AnalysisConfig(find_peaks=values).validate()


def test_classification_covers_all_118_and_profiles_are_reviewable():
    table=classification_table()
    assert len(table)==118 and table.Element.nunique()==118
    assert set(table.Class)=={'common','uncommon'}
    assert set(SYMBOLS)==set(table.Element)
    for symbol in ['Ca','Fe','Hg','Pb','Cr','Cu']:assert element_class(symbol)=='common'
    for symbol in ['Pm','Pa','Tc','Fr','Og']:assert element_class(symbol)=='uncommon'
    assert element_class('U','minerals')=='common'
    assert element_class('U','heritage')=='uncommon'
    assert len(pd.read_csv(ROOT/'data/common_elements.csv'))+len(pd.read_csv(ROOT/'data/uncommon_elements.csv'))==118


def test_prior_breaks_only_close_ties_and_does_not_erase_strong_rare_evidence():
    assert choose_by_evidence([10.,10.2],['Ca','Pm'])==0
    assert choose_by_evidence([10.,13.],['Ca','Pm'])==1
    assert choose_by_evidence([10.,10.2],['Ca','Pm'],profile='none')==1
    assert choose_by_evidence([-.1,.2],['Ca','Pm'])==1


def test_common_preference_preserves_unidentifiability():
    cfg=AnalysisConfig(fit_response=False,elements=('Ca','Pm'),element_selection='manual')
    lines=[LineCandidate(20,'Ca','Kalpha1',3.69,8.),LineCandidate(61,'Pm','Lalpha1',3.69,4.)]
    edges=np.arange(3,4.501,.02);energy=(edges[:-1]+edges[1:])/2
    y=np.random.default_rng(42).poisson(10+5000*gaussian_response(edges,3.69,float(sigma_kev(3.69,cfg))))
    peaks=detect_peaks(np.arange(len(y)),energy,edges,y,np.full(len(y),10.),cfg)
    result=identify_elements(y,energy,edges,np.full(len(y),10.),lines,peaks,cfg,.02)
    c=result.candidates.set_index('Element')
    assert c.loc['Ca','Preferred_By_Prior']
    assert c.loc['Ca','Status']=='unidentifiable'
    assert c.loc['Ca','Quantification_Status']!='supported'
    assert c.loc['Pm','Status']=='unidentifiable'


def test_manual_centers_replace_discovery_and_preserve_all_inputs(synthetic,tmp_path,monkeypatch):
    s,c,t,cfg,_,_=synthetic({'Fe':600.},shift=.055,width_scale=1.12)
    cfg.find_peaks=(6.4039,6.3909,7.058)
    cfg.elements=('Fe',)
    def forbidden(*args,**kwargs):pytest.fail('Automatic discovery must not run in find_peaks mode')
    monkeypatch.setattr('xrf_analyzer.pipeline.detect_peaks',forbidden)
    result=analyze(s,c,t,tmp_path/'manual',cfg)
    assert len(result['peaks'])==2  # unresolved Kalpha1/Kalpha2 are not independent evidence
    assert result['summary']['manual_peaks']['requested_energies_keV']==list(cfg.find_peaks)
    assert not result['summary']['manual_peaks']['automatic_discovery_used']
    assert result['identification'].candidates.set_index('Element').loc['Fe','Status']=='supported'
    for center in result['identification'].fit.line_centers.values():assert center in cfg.find_peaks
    lines=pd.read_csv(tmp_path/'manual/fitted_lines.csv')
    assert set(lines.Specified_Center_keV.dropna())<=set(cfg.find_peaks)
    density=pd.read_csv(tmp_path/'manual/relative_areal_density.csv')
    assert set(density.Element_Status)<= {'supported'}


def test_manual_observed_coordinates_are_not_shifted(synthetic):
    s,c,t,cfg,_,_=synthetic({'Fe':600.},shift=.055)
    cfg.find_peaks=(6.4589,7.113)
    cfg.find_peaks_axis='observed';cfg.elements=('Fe',)
    result=analyze(s,c,t,config=cfg)
    from xrf_analyzer.response import line_model_center
    fit=result['identification'].fit
    assert fit.success
    for line in result['identification'].fit_lines:
        if (line.symbol,line.branch) in fit.line_centers:
            assert line_model_center(line,fit) in cfg.find_peaks


@pytest.mark.parametrize('axis', ['reference', 'observed'])
@pytest.mark.parametrize('measured_background', [False, True])
def test_requested_response_keeps_axis_independent_of_background(axis, measured_background):
    # Two well separated peaks distinguish an affine gain from a common offset.
    # Background data must not replace the boolean selecting the energy axis.
    centers=np.array([6.4,12.6]);shift=.12;gain=-.004
    observed=centers*(1+gain)+shift
    cfg=AnalysisConfig(find_peaks=tuple(centers if axis=='reference' else observed),
                       find_peaks_axis=axis)
    edges=np.arange(4.,15.0001,.02);energy=(edges[:-1]+edges[1:])/2
    rng=np.random.default_rng(78)
    background=np.full(len(energy),8.)
    continuum=np.full(len(energy),5.)
    expected=continuum.copy()
    if measured_background:
        cfg._background_counts=rng.poisson(background)
        cfg._background_scale=1.3
        expected+=1.3*background
    for center,position in zip(centers,observed):
        expected+=20000*gaussian_response(edges,position,sigma_kev(center,cfg)*1.1)
    counts=rng.poisson(expected)
    detection_background=continuum+(1.3*cfg._background_counts if measured_background else 0.)
    peaks,response=fit_requested_peaks(np.arange(len(energy)),energy,edges,counts,
                                      detection_background,cfg)
    if axis=='reference':
        assert response['energy_gain']==pytest.approx(gain,abs=.0008)
        assert response['delta_E_keV']==pytest.approx(shift,abs=.006)
        assert peaks.observed_energy_keV.to_numpy()==pytest.approx(observed,abs=.004)
    else:
        assert response['energy_gain']==0.
        assert response['delta_E_keV']==0.
        assert peaks.observed_energy_keV.to_numpy()==pytest.approx(observed,abs=1e-12)
    assert np.isfinite(peaks.approx_significance).all()
    assert (peaks.approx_significance>5).all()


def test_absent_manual_hints_are_not_element_evidence(synthetic):
    s,c,t,cfg,_,_=synthetic({},background=.2)
    cfg.find_peaks=(6.4,7.06);cfg.elements=('Fe',)
    result=analyze(s,c,t,config=cfg)
    assert not (result['identification'].candidates.Status=='supported').any()
    assert result['density'].empty


def test_manual_mode_computes_supported_only_ratios(synthetic):
    s,c,t,cfg,_,_=synthetic({'Fe':1000.,'Cu':500.})
    cfg.find_peaks=(6.4039,7.058,8.0478,8.9053)
    cfg.elements=('Fe','Cu','Co','Pb');cfg.reference_element='Fe'
    result=analyze(s,c,t,config=cfg)
    density=result['density'].set_index('Element')
    assert set(density.index)=={'Fe','Cu'}
    assert density.loc['Cu','Relative_to_Reference']==pytest.approx(.5,rel=.10)
    assert density.Fraction_among_Reported_Elements.sum()==pytest.approx(1.)


def test_cli_find_peaks_alias(synthetic,tmp_path):
    s,c,t,_,_,_=synthetic({'Fe':600.})
    assert main(['analyze','--spectrum',str(s),'--calibration',str(c),'--sensitivity',str(t),
                 '--output-dir',str(tmp_path/'cli'),'--elements','Fe','--find_peaks','[6.4039, 7.058]'])==0
    summary=json.loads((tmp_path/'cli/fit_summary.json').read_text())
    assert summary['parameters']['find_peaks']==[6.4039,7.058]
    assert summary['manual_peaks']['mode']=='user_specified'


def test_real_calcium_branch_evidence_and_only_supported_output(tmp_path):
    cfg=AnalysisConfig(energy_range=(1.,40.),tube_voltage_kv=50)
    result=analyze(ROOT/'SDD_Spectrum/shafa_50kv_20uA.mca',ROOT/'data/calibration.csv',
                   ROOT/'data/xrf_relative_sensitivity.csv',tmp_path/'real',cfg)
    c=result['identification'].candidates.set_index('Element')
    assert c.loc['Ca','Status'] in ('supported','tentative')
    rows=[d for d in result['identification'].branch_evidence if d['Element']=='Ca']
    alpha=next(d for d in rows if 'Kalpha1' in d['Branches'])
    beta=next(d for d in rows if 'Kbeta1' in d['Branches'])
    assert alpha['Branch_DeltaDeviance']>100
    assert beta['Expected_Ratio_to_Strongest_Group']>0
    assert beta['Wide_Window_Branch_DeltaDeviance'] is not None
    assert (tmp_path/'real/calcium_region.png').exists()
    assert (tmp_path/'real/calcium_analysis.json').exists()
    assert set(result['density'].Element_Status)<= {'supported'}
    pending=pd.read_csv(tmp_path/'real/elements_need_confirmation.csv')
    assert not set(pending.Element).intersection(result['density'].Element)
    assert set(result['density'].Element)<=set(c.index[c.Status=='supported'])


def test_user_full_example_on_real_spectrum(tmp_path):
    values=(3.69,3.688,6.4,7.06,10.5,12.6,14.7,9.98,11.9)
    cfg=AnalysisConfig(energy_range=(1.,40.),tube_voltage_kv=50,find_peaks=values)
    result=analyze(ROOT/'SDD_Spectrum/shafa_50kv_20uA.mca',ROOT/'data/calibration.csv',
                   ROOT/'data/xrf_relative_sensitivity.csv',tmp_path/'manual',cfg)
    c=result['identification'].candidates.set_index('Element')
    assert all(c.loc[s,'Status']=='supported' for s in ['Fe','Hg','Pb'])
    assert c.loc['As','Status']!='supported'
    assert result['summary']['manual_peaks']['independent_resolution_groups']==8
    assert result['summary']['parameters']['find_peaks']==values
    fit=result['identification'].fit
    assert fit.line_centers and set(fit.line_centers.values())<=set(values)
    assert set(result['density'].Element_Status)<= {'supported'}
    assert (tmp_path/'manual/element_classes.csv').exists()


def test_extended_m_shell_preserves_input_and_derives_only_same_shell():
    from xrf_analyzer.atomic_completion import complete_transitions
    from xrf_analyzer.atomic_lines import load_sensitivity
    original=load_sensitivity(ROOT/'data/xrf_relative_sensitivity.csv').lines
    full,_,_=complete_transitions(original,AnalysisConfig())
    pb=[l for l in full if l.symbol=='Pb' and l.parent_shell=='M5']
    assert pb and all(l.strength_source=='same_shell_rad_rate' for l in pb)
    assert any(l.branch=='M5N6' for l in pb)
    known=next(l for l in original if l.symbol=='Pb' and l.branch=='Malpha1')
    assert next(l for l in full if l.symbol=='Pb' and l.branch=='Malpha1')==known
