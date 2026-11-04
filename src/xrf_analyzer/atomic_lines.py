from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import re
import numpy as np
import xraylib
from .csv_input import read_csv_unique_headers

LINE_MAP = dict(Kalpha1='KA1_LINE', Kalpha2='KA2_LINE', Kbeta1='KB1_LINE',
                Lalpha1='LA1_LINE', Lalpha2='LA2_LINE', Lbeta1='LB1_LINE',
                Lbeta2='LB2_LINE', Lgamma1='LG1_LINE', Malpha1='MA1_LINE')
LINE_SHELL = dict(Kalpha1='K', Kalpha2='K', Kbeta1='K',
                  Lalpha1='L3', Lalpha2='L3', Lbeta1='L2',
                  Lbeta2='L3', Lgamma1='L2', Malpha1='M5')


def excitation_edge_kev(line):
    """Minimum incident energy needed to create this line's initial vacancy.

    A fluorescence photon lies below its absorption edge. Testing emission
    energy against the tube endpoint alone admits physically impossible lines.
    Completed and auxiliary transitions carry their initial shell explicitly.
    """
    shell = line.parent_shell or LINE_SHELL[line.branch]
    return float(xraylib.EdgeEnergy(line.Z, getattr(xraylib, shell+'_SHELL')))


@dataclass(frozen=True)
class LineCandidate:
    Z: int
    symbol: str
    branch: str
    energy_keV: float
    sensitivity_cm2_g: float
    strength_source: str = "csv"
    parent_shell: str = ""


@dataclass
class SensitivityTable:
    lines: list
    states: list
    metadata: dict
    warnings: list


def validate_symbol(symbol):
    try:
        z = xraylib.SymbolToAtomicNumber(symbol)
    except ValueError as exc:
        raise ValueError('Unknown element symbol: ' + symbol) from exc
    if xraylib.AtomicNumberToSymbol(z) != symbol:
        raise ValueError('Use canonical element symbol: ' + symbol)
    return z


def load_sensitivity(path):
    """Read common-scale S and resolve positive branches with xraylib (native keV)."""
    path = Path(path)
    df = read_csv_unique_headers(path, encoding='utf-8-sig', keep_default_na=False)
    if df.columns.duplicated().any() or not {'Element', *LINE_MAP} <= set(df):
        raise ValueError('Sensitivity table requires Element and all nine branch columns')
    lines, states, warnings, seen = [], [], [], set()
    extra = set(df) - {'Element', *LINE_MAP}
    if extra:
        warnings.append('Ignored sensitivity columns: ' + ', '.join(sorted(extra)))
    for _, row in df.iterrows():
        match = re.fullmatch(r'\s*(\d+)\s+([A-Z][a-z]?)\s*', str(row.Element))
        if not match:
            raise ValueError('Element must be Z + symbol, e.g. 26 Fe')
        z, symbol = int(match[1]), match[2]
        if validate_symbol(symbol) != z or z in seen:
            raise ValueError('Inconsistent or duplicate sensitivity element: ' + row.Element)
        seen.add(z)
        for branch, constant in LINE_MAP.items():
            value = str(row[branch]).strip()
            if value == '' or value.lower() == 'nan':
                states.append(dict(Z=z, Element=symbol, Branch=branch, State='missing', S=None))
                continue
            s = float(value)
            if not np.isfinite(s) or s < 0:
                raise ValueError('Sensitivity must be finite and nonnegative: %s %s' % (symbol, branch))
            state = dict(Z=z, Element=symbol, Branch=branch, State='zero' if s == 0 else 'positive', S=s)
            states.append(state)
            if s == 0:
                continue
            try:
                energy = xraylib.LineEnergy(z, getattr(xraylib, constant))
                if not np.isfinite(energy) or energy <= 0:
                    raise ValueError('Invalid line energy')
            except ValueError as exc:
                state['State'] = 'database_unavailable'
                warnings.append('Removed %s %s: %s' % (symbol, branch, exc))
                logging.warning(warnings[-1])
                continue
            lines.append(LineCandidate(z, symbol, branch, energy, s))
    sidecar = path.with_suffix('.metadata.json')
    metadata = {'sensitivity_provenance': 'unverified', 'metadata_file': None}
    if sidecar.exists():
        declared = json.loads(sidecar.read_text(encoding='utf-8-sig'))
        if not isinstance(declared, dict):
            raise ValueError('Sensitivity metadata must be a JSON object')
        required = {'excitation_spectrum_sha256', 'normalization', 'xraylib_version', 'detector_efficiency_model'}
        metadata.update(declared=declared, metadata_file=str(sidecar.resolve()),
                        sensitivity_provenance='provided_unverified')
        if not required <= set(declared):
            warnings.append('Sensitivity metadata lacks required provenance fields')
    else:
        warnings.append('Sensitivity provenance unverified: missing .metadata.json; source matching is not established')
    return SensitivityTable(lines, states, metadata, warnings)
