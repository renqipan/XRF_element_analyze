"""Human-readable terminal results shared by single and directory analysis."""
from pathlib import Path
import math


def print_run_summary(result, stream=None):
    summary = result['summary']
    filename = Path(summary['input_files']['sample_spectrum']['path']).name
    candidates = result['identification'].candidates
    supported = candidates.loc[candidates.Status == 'supported', 'Element'].tolist()
    possible = summary.get('elements_need_confirmation', [])
    print('\n样品：' + filename, file=stream, flush=True)
    print('确定存在（证据支持）：' + (', '.join(supported) or '无'), file=stream)
    print('相对面密度比值（表观值）：', file=stream)
    if summary.get('quantification_method')=='local-peaks':
        print('  定量依据：局部净峰面积 / 相对灵敏度',file=stream)
    if not supported:
        print('  无可报告的支持元素', file=stream)
    for row in result['density'].itertuples(index=False):
        ratio = row.Relative_to_Reference
        if ratio is not None and math.isfinite(float(ratio)):
            print('  %s / %s = %.6g' % (row.Element, row.Reference_Element, ratio), file=stream)
        else:
            print('  %s：暂不可估计（%s）' % (row.Element, row.Quantification_Status), file=stream)
    print('可能存在（待定，不参与面密度计算）：' + (', '.join(possible) or '无'), file=stream, flush=True)


def print_failed_summary(spectrum, error, stream=None):
    print('\n样品：' + Path(spectrum).name, file=stream)
    print('分析失败：' + str(error), file=stream)
    print('确定存在：无法判定；相对面密度比值：不可估计；可能存在：无法判定', file=stream, flush=True)
