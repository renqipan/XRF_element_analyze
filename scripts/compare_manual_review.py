#!/usr/bin/env python3
"""Compare manual hypotheses with actual four-line batch TXT outputs.

This script is for review only; its reference JSON never enters the analyzer.
Missing suggestions and extra detections are disagreements, not measured errors.
"""
import argparse
import json
from pathlib import Path
import re

import pandas as pd


def read_summary(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    if len(lines) % 4 or any(lines[i] for i in range(3, len(lines), 4)):
        raise ValueError('Expected three content lines and one blank line per spectrum')
    result = {}
    for i in range(0, len(lines), 4):
        name = lines[i].split('：', 1)[1]
        pending = lines[i+2].split('：', 1)[1].strip()
        result[name] = {'supported': re.findall(r'\b([A-Z][a-z]?)=', lines[i+1].split('：', 1)[1]),
                        'possible': [] if pending == '无' else [s.strip() for s in pending.split(',')]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', default='XRF_Analyze_results')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    before = read_summary(args.before)
    after_root = Path(args.after)
    after = read_summary(after_root/'relative_areal_density_all.txt')
    reference = json.loads(Path(args.reference).read_text())
    rows = []
    for name, suggestion in reference['cases'].items():
        old, new = before[name], after[name]
        # Verify that the user's TXT agrees with the per-spectrum diagnostics.
        candidates = pd.read_csv(after_root/'spectra'/name/'element_candidates.csv')
        supported = set(candidates.loc[candidates.Status == 'supported', 'Element'])
        assert supported == set(new['supported']), name
        expected = set(suggestion['suggested_present'])
        rows.append({'Spectrum': name, 'Manual_Suggested': ','.join(suggestion['suggested_present']),
            'Manual_Uncertain': ','.join(suggestion['suggested_possible']),
            'Before_Supported': ','.join(old['supported']), 'Before_Possible': ','.join(old['possible']),
            'After_Supported': ','.join(new['supported']), 'After_Possible': ','.join(new['possible']),
            'Suggested_But_Still_Possible': ','.join(sorted(expected & set(new['possible']))),
            'Suggested_But_Not_Reported': ','.join(sorted(expected-supported-set(new['possible']))),
            'Additional_Supported': ','.join(sorted(supported-expected))})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'manual_comparison.csv', index=False)
    report = {'reference_is_ground_truth': False, 'reference_used_by_identification': False,
              'compared_txt': str((after_root/'relative_areal_density_all.txt').resolve()),
              'reference_source': reference['source'], 'rows': rows}
    (output/'manual_comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    cols = ['Spectrum', 'Manual_Suggested', 'Before_Supported', 'After_Supported', 'After_Possible']
    titles = ['文件', '人工建议（非真值）', '上一版确定', '本版确定', '本版待定']
    table = '| '+' | '.join(titles)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
    table += '\n'.join('| '+' | '.join(row[k] or '—' for k in cols)+' |' for row in rows)+'\n'
    (output/'manual_comparison.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
