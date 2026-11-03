"""Real-spectrum regression: an unconfirmed competitor is not a low-Z absence."""
from pathlib import Path
import json
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.pipeline import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_shafa_cr_remains_possible_under_pm_template_competition():
    settings=json.loads((ROOT/'configs/shafa.json').read_text())
    result=analyze(ROOT/'SDD_Spectrum/shafa_50kv_20uA.mca', ROOT/'data/calibration.csv',
                   ROOT/'data/xrf_relative_sensitivity.csv', config=AnalysisConfig(**settings))
    candidates=result['identification'].candidates.set_index('Element')
    assert candidates.loc['Cr','Status']=='tentative'
    assert candidates.loc['Cr','q_Fit']<=1e-6  # Original full fit is deliberately unchanged.
    assert candidates.loc['Cr','Quantification_Status']=='not_estimable'
    assert candidates.loc['Cr','Preferred_By_Prior']
    assert candidates.loc['Pm','Status']!='supported'
    assert 'Cr' not in result['density'].Element.tolist()
    assert 'Cr' in result['summary']['elements_need_confirmation']
    reviews=result['summary']['uncertainty']['low_z_counterfactual_reviews']
    cr=next(item for item in reviews if item['Element']=='Cr' and item['Removed_Rival']=='Pm')
    assert cr['Recovered_Possible_Interpretation']
    assert cr['Full_Model_and_Quantification_Unchanged']
    assert cr['Counterfactual_Element_DeltaDeviance']>=25
    alpha=cr['Counterfactual_Kalpha_Evidence']
    assert alpha['Kalpha_DeltaDeviance']>=25
    assert alpha['Wide_Kalpha_DeltaDeviance']>=25
    assert alpha['Calibration_Anchored']
    # Disabled independent shells may not re-enter centroid assignments or anchors.
    pairs=result['identification'].pairings
    assert not ((pairs.Element=='Pm') & (pairs.Strength_Source=='unscaled_auxiliary_shell')).any()
