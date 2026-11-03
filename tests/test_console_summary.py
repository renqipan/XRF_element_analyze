"""Public terminal output must distinguish identity from estimable ratios."""
from io import StringIO
from types import SimpleNamespace
import numpy as np
import pandas as pd
from xrf_analyzer.console import print_run_summary, print_failed_summary


def test_summary_reports_numeric_and_withheld_ratios_and_possible_elements():
    result = {
        'summary': {'input_files': {'sample_spectrum': {'path': '/tmp/sample.mca'}},
                    'elements_need_confirmation': ['Ag']},
        'identification': SimpleNamespace(candidates=pd.DataFrame({
            'Element': ['Ca', 'Fe', 'Ag'], 'Status': ['supported', 'supported', 'tentative']})),
        'density': pd.DataFrame([
            {'Element': 'Ca', 'Reference_Element': 'Fe', 'Relative_to_Reference': .25,
             'Quantification_Status': 'supported'},
            {'Element': 'Fe', 'Reference_Element': None, 'Relative_to_Reference': np.nan,
             'Quantification_Status': 'global_model_mismatch'}])}
    stream = StringIO()
    print_run_summary(result, stream)
    output = stream.getvalue()
    assert 'Ca, Fe' in output and 'Ca / Fe = 0.25' in output
    assert 'Fe：暂不可估计（global_model_mismatch）' in output
    assert '待定，不参与面密度计算）：Ag' in output


def test_failure_reports_no_fabricated_identity_or_ratio():
    stream = StringIO()
    print_failed_summary('/tmp/bad.csv', 'invalid counts', stream)
    assert 'bad.csv' in stream.getvalue() and '分析失败：invalid counts' in stream.getvalue()
    assert '无法判定' in stream.getvalue() and '不可估计' in stream.getvalue()
