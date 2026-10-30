"""Contextual frequency classes used only to break unresolved evidence ties.

These are reviewable domain heuristics, not elemental abundances, detection
limits, calibrated probabilities, or a whitelist. Strong data always wins.
"""
import pandas as pd

SYMBOLS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
           'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba '
           'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi '
           'Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
BASE = set('H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn As Se Br Rb Sr Y Zr Nb Mo Ag Cd In Sn Sb Cs Ba La Ce Nd W Au Hg Pb Bi'.split())
PROFILES = {
    'general': BASE,
    'heritage': set('H B C N O F Na Mg Al Si P S Cl K Ca Ti Cr Mn Fe Co Ni Cu Zn As Se Br Sr Zr Ag Cd Sn Sb Ba Au Hg Pb Bi'.split()),
    'archaeology': set('H Li B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn As Se Br Rb Sr Y Zr Nb Mo Ag Cd In Sn Sb Cs Ba La Ce Nd W Pt Au Hg Pb Bi'.split()),
    'minerals': BASE | set('Ga Ge Te I Pr Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta Tl Th U'.split()),
    'none': set(),
}


def element_class(symbol, profile='general'):
    if profile not in PROFILES:
        raise ValueError('Unknown element prior profile: '+profile)
    return 'common' if symbol in PROFILES[profile] else 'uncommon'


def choose_by_evidence(scores, symbols, profile='general', margin=.5):
    """Return a data maximum unless a common element is within a small tie band."""
    import numpy as np
    values = np.asarray(scores, float)
    best = int(np.argmax(values))
    if profile == 'none' or not np.isfinite(values[best]):
        return best
    tied = [i for i, s in enumerate(symbols) if values[i] >= values[best]-margin
            and (values[i]>0 or values[best]<=0) and element_class(s, profile) == 'common']
    return max(tied, key=lambda i: values[i]) if tied else best


def preferred_in_group(symbols, profile='general'):
    common = sorted(s for s in symbols if element_class(s, profile) == 'common')
    return common[0] if len(common) == 1 else None


def classification_table(profile='general'):
    return pd.DataFrame([{'Z': z, 'Element': symbol, 'Class': element_class(symbol, profile),
                          'Profile': profile, 'Policy': 'tie_break_only; not_evidence_of_presence',
                          **{p+'_class': element_class(symbol, p) for p in PROFILES if p != 'none'}}
                         for z, symbol in enumerate(SYMBOLS, 1)])
