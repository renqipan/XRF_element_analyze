"""Complete omitted transitions without inventing their excitation sensitivity.

Within one initial vacancy shell, S(line)/RadRate(line) is the same shell
production coefficient. This identity does NOT hold between L1, L2 and L3.
Unscaled shells are auxiliary shape components, never quantitative S values.
"""
from dataclasses import dataclass
import re
import numpy as np
import xraylib
from .atomic_lines import LINE_MAP, LineCandidate


@dataclass(frozen=True)
class AuxiliaryLine:
    Z: int
    symbol: str
    branch: str
    energy_keV: float
    relative_rate: float
    parent_shell: str


def parent_shell(constant):
    # Use explicit single-transition names to avoid KA/KB aggregate aliases.
    for name in dir(xraylib):
        if re.fullmatch(r'(?:K[LMNOP][1-7]+|L[123][MNOP][1-7]+|M[1-5][NOPQ][1-7]+)_LINE', name) and getattr(xraylib, name) == constant:
            return 'K' if name.startswith('K') else name[:2]
    return None


def complete_transitions(lines, config):
    if not config.complete_atomic_lines:
        return list(lines), [], []
    output, auxiliary, audit = list(lines), [], []
    supplied_constants = {getattr(xraylib, value) for value in LINE_MAP.values()}
    for symbol in sorted({l.symbol for l in lines}):
        original = [l for l in lines if l.symbol == symbol]
        z = original[0].Z
        anchors = {}
        for line in original:
            constant = getattr(xraylib, LINE_MAP[line.branch])
            shell = parent_shell(constant)
            if not shell:
                continue
            rate = xraylib.RadRate(z, constant)
            if rate > 0:
                anchors.setdefault(shell, []).append(line.sensitivity_cm2_g/rate)
        coefficients = {}
        for shell, estimates in anchors.items():
            center = float(np.median(estimates))
            spread = max(abs(np.array(estimates)/center-1))
            if spread <= config.shell_consistency_tolerance:
                coefficients[shell] = center
            audit.append({'Element': symbol, 'Shell': shell, 'AnchorConsistencyRelative': float(spread),
                          'CompletionEnabled': shell in coefficients,
                          'Method': 'same_initial_shell_S_over_RadRate; existing S unchanged'})
        seen = set(supplied_constants)
        for name in sorted(dir(xraylib)):
            if not re.fullmatch(r'(?:K[LMNOP][1-7]+|L[123][MNOP][1-7]+|M[1-5][NOPQ][1-7]+)_LINE', name):
                continue
            constant = getattr(xraylib, name)
            if constant in seen:
                continue
            seen.add(constant)
            shell = 'K' if name.startswith('K') else name[:2]
            # Only add an unknown L1 shell when independently anchored L2/L3 is active.
            unknown_l1 = shell == 'L1' and ('L2' in coefficients or 'L3' in coefficients)
            if shell not in coefficients and not unknown_l1:
                continue
            try:
                rate = xraylib.RadRate(z, constant)
                energy = xraylib.LineEnergy(z, constant)
            except ValueError:
                continue
            if rate < config.minimum_radiative_rate or energy <= 0:
                continue
            branch = name.removesuffix('_LINE') if hasattr(name, 'removesuffix') else name[:-5]
            if shell in coefficients:
                output.append(LineCandidate(z, symbol, branch, energy, coefficients[shell]*rate,
                                            'same_shell_rad_rate', shell))
            else:
                auxiliary.append(AuxiliaryLine(z, symbol, branch, energy, rate, shell))
    return output, auxiliary, audit
