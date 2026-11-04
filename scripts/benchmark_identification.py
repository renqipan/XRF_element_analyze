#!/usr/bin/env python3
"""Reproducible numerical benchmark; never overwrite normal analysis outputs."""
import argparse
import contextlib
import hashlib
import io
import json
import logging
from pathlib import Path
import time

import numpy as np
from xrf_analyzer.config import AnalysisConfig
from xrf_analyzer.pipeline import analyze


def summarize(result):
    ident = result['identification']
    return dict(fit_success=ident.fit.success, optimizer=result['summary']['optimizer'],
                fitted_symbols=ident.fit.symbols, response_initialization=result['summary']['response_initialization'],
                supported=ident.candidates.loc[ident.candidates.Status == 'supported', 'Element'].tolist(),
                tentative=ident.candidates.loc[(ident.candidates.Status == 'tentative'), 'Element'].tolist(),
                deviance_per_dof=result['summary']['deviance_per_dof'],
                delta=ident.fit.delta, gain=ident.fit.energy_gain,
                candidates=json.loads(ident.candidates.to_json(orient='records')),
                density=json.loads(result['density'].to_json(orient='records')),
                branch_evidence=ident.branch_evidence,
                low_z_evidence=ident.low_z_evidence,
                elapsed_seconds=result['summary']['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--input-dir', default='XRF_SDD')
    parser.add_argument('--output', required=True)
    parser.add_argument('--overrides', default='{}', help='JSON configuration overrides')
    parser.add_argument('--names', nargs='*')
    args = parser.parse_args()
    parameters = json.loads(Path(args.config).read_text())
    parameters.update(json.loads(args.overrides))
    logging.disable(logging.WARNING)
    paths = sorted(Path(args.input_dir).glob('*.mca'))
    if args.names:
        paths = [p for p in paths if any(name in p.name for name in args.names)]
    report = {'parameters': parameters, 'spectra': {}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        start = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            result = analyze(path, 'data/calibration.csv', 'data/xrf_relative_sensitivity.csv',
                             config=AnalysisConfig(**parameters))
        row = summarize(result)
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['spectra'][path.name] = row
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        print(path.name, 'supported='+','.join(row['supported']),
              'D/dof=%.3f' % row['deviance_per_dof'], '%.1fs' % (time.monotonic()-start), flush=True)


if __name__ == '__main__':
    main()
