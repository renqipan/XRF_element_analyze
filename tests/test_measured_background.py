import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from xrf_analyzer.measured_background import profile_background
from xrf_analyzer.fit_model import fit_spectrum
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.atomic_lines import LineCandidate
from xrf_analyzer.response import gaussian_response,sigma_kev
from xrf_analyzer.pipeline import analyze


def test_profile_matches_independent_numeric_background_optimization():
    y=np.array([0.,20.,55.,4.]);s=np.array([2.,5.,30.,1.]);b=np.array([0.,8.,12.,0.]);alpha=1.7
    result=profile_background(y,s,alpha,b)
    for i in range(len(y)):
        optimum=minimize_scalar(lambda x: (s[i]+alpha*x-y[i]*np.log(s[i]+alpha*x)+x-(b[i]*np.log(x) if b[i] else 0)),bounds=(1e-12,100),method='bounded')
        assert result['latent_background'][i]==pytest.approx(optimum.x,abs=1e-4)
    step=1e-5
    derivative=(profile_background(y,s,alpha+step,b)['nll']-profile_background(y,s,alpha-step,b)['nll'])/(2*step)
    assert derivative==pytest.approx(result['scale_gradient'],rel=1e-5)


def test_global_scale_and_signal_recover_known_truth_with_noisy_reference():
    cfg=AnalysisConfig(fit_response=False,background_degree=0)
    edges=np.arange(3.,15.001,.02);e=(edges[1:]+edges[:-1])/2
    line=LineCandidate(26,'Fe','Kalpha1',6.4,8.)
    b=8.+9000*gaussian_response(edges,10.,.08)
    rng=np.random.default_rng(51);reference=rng.poisson(b)
    counts=rng.poisson(2+1.7*b+500*8*gaussian_response(edges,6.4,sigma_kev(6.4,cfg)))
    fit=fit_spectrum(counts,e,edges,np.full(len(e),2.),[line],cfg,background_counts=reference)
    assert fit.success
    assert fit.background_scale==pytest.approx(1.7,rel=.05)
    assert fit.q[0]==pytest.approx(500,rel=.08)
    assert fit.background_scale_std>0
    assert np.allclose(fit.model,fit.background+fit.components.sum(axis=1)+fit.scatter)
    assert fit.deviance>=fit.sample_deviance


def test_background_environment_element_is_not_sample_composition(synthetic,tmp_path):
    spectrum,cal,sensitivity,cfg,_,lines=synthetic({'Fe':500.})
    cfg.elements=('Fe','Cu');cfg.fit_response=False;cfg.estimate_resolution_from_peaks=False
    c=np.arange(1001);edges=(np.arange(1002)-.5)*.02
    background=np.full(len(c),8.);signal=np.full(len(c),2.)
    for line in lines:
        response=gaussian_response(edges,line.energy_keV,sigma_kev(line.energy_keV,cfg))
        if line.symbol=='Cu':background+=400*line.sensitivity_cm2_g*response
        if line.symbol=='Fe':signal+=500*line.sensitivity_cm2_g*response
    background+=10000*gaussian_response(edges,12.,.08)
    rng=np.random.default_rng(911)
    pd.DataFrame({'channel':c,'counts':rng.poisson(signal+1.5*background)}).to_csv(spectrum,index=False)
    path=tmp_path/'background.csv'
    pd.DataFrame({'channel':c,'counts':rng.poisson(background)}).to_csv(path,index=False)
    result=analyze(spectrum,cal,sensitivity,config=cfg,background_spectrum=path)
    candidates=result['identification'].candidates.set_index('Element')
    assert candidates.loc['Fe','Status']=='supported'
    assert candidates.loc['Cu','Status']!='supported'
    assert result['summary']['measured_background']['scale']==pytest.approx(1.5,rel=.1)
    assert result['density'].set_index('Element').loc['Fe','q_Area']==pytest.approx(500,rel=.15)


def test_api_explicit_none_overrides_configured_background(synthetic):
    s,c,t,cfg,_,_=synthetic({'Fe':500.})
    cfg.background_spectrum='nonexistent_background.mca'
    result=analyze(s,c,t,config=cfg,background_spectrum=None)
    assert not result['summary']['measured_background']['enabled']
