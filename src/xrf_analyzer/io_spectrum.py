from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .csv_input import read_csv_unique_headers


@dataclass
class Spectrum:
    channel: np.ndarray
    counts: np.ndarray
    live_time_s: object = None
    metadata: dict = field(default_factory=dict)


def _validate(channel, counts, fill=False):
    channel, counts = np.asarray(channel, float), np.asarray(counts, float)
    if not len(counts) or not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Raw counts must be nonnegative finite integers')
    if not np.isfinite(channel).all() or (channel < 0).any() or not np.equal(channel, np.floor(channel)).all():
        raise ValueError('Channels must be nonnegative finite integers')
    if (np.diff(channel) <= 0).any():
        raise ValueError('Channels must be strictly increasing and unique')
    missing = []
    if (np.diff(channel) != 1).any():
        if not fill:
            raise ValueError('Missing channels; use --fill-missing-channels explicitly to fill with zeros')
        full = np.arange(int(channel[0]), int(channel[-1]) + 1)
        missing = np.setdiff1d(full, channel).tolist()
        out = np.zeros(len(full))
        out[(channel - full[0]).astype(int)] = counts
        channel, counts = full, out
    return channel.astype(int), counts, missing


def load_spectrum(path, channel_start=0, fill_missing_channels=False):
    path = Path(path)
    if channel_start not in (0, 1):
        raise ValueError('channel_start must be 0 or 1')
    if path.suffix.lower() in ('.mca', '.mac'):
        return _load_mca(path, channel_start, fill_missing_channels)
    if path.suffix.lower() != '.csv':
        raise ValueError('Unknown spectrum format; provide CSV or documented ASCII MCA')
    df = read_csv_unique_headers(path, encoding='utf-8-sig')
    if df.columns.duplicated().any() or 'counts' not in df:
        raise ValueError('Sample CSV requires a counts header')
    counts = pd.to_numeric(df.counts, errors='raise').to_numpy(float)
    explicit = 'channel' in df
    channel = pd.to_numeric(df.channel, errors='raise') if explicit else np.arange(len(df)) + channel_start
    channel, counts, missing = _validate(channel, counts, fill_missing_channels)
    meta = {'format': 'counts_csv', 'channel_origin': int(channel[0]), 'explicit_channels': explicit,
            'filled_channels': missing, 'warnings': []}
    if missing:
        meta['warnings'].append('Unmeasured channels explicitly filled with zero')
    for key in ('live_time_s', 'real_time_s', 'tube_voltage_kV'):
        if key in df:
            values = pd.to_numeric(df[key], errors='raise').dropna().unique()
            if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
                raise ValueError('Invalid or inconsistent spectrum metadata: ' + key)
            meta[key] = float(values[0])
    if 'energy_keV' in df and not missing:
        energies = pd.to_numeric(df.energy_keV, errors='raise').to_numpy(float)
        if not np.isfinite(energies).all():
            raise ValueError('Nonfinite supplied CSV energy')
        meta['supplied_energy_keV'] = energies.tolist()
    return Spectrum(channel, counts, meta.get('live_time_s'), meta)


def _record_dp5_acquisition_domain(lines, data_end, counts, channel_start, meta):
    """Interpret only the documented modern, normal-spectrum DP5 channel LLD.

    Amptek Programmer's Guide Rev B4 sections 1.33, 5.1.33/35/76:
    FW >= 6.06 uses zero-based TLLD and rejects device channels <= TLLD.
    DATA array indices stay separate from the user's 0/1-based channel labels.
    """
    begin, finish = '<<DP5 CONFIGURATION>>', '<<DP5 CONFIGURATION END>>'
    if lines.count(begin) != 1 or lines.count(finish) != 1:
        return
    a, b = lines.index(begin), lines.index(finish)
    if not data_end < a < b:
        return
    configuration = {}
    for key in ('MCAS', 'MCAC', 'SOFF', 'TLLD'):
        values = [m.group(1).strip() for line in lines[a+1:b]
                  for m in [re.fullmatch(key+r'\s*=\s*([^;]+);.*', line)] if m]
        if len(values) > 1:
            raise ValueError('Ambiguous DP5 configuration: duplicate '+key)
        if values:
            configuration[key] = values[0]
    meta['mca_configuration'] = configuration
    if configuration.get('MCAS') != 'NORM' or 'TLLD' not in configuration:
        return
    if configuration['TLLD'].upper() in ('OF', 'OFF'):
        return
    if (configuration.get('MCAC') not in tuple(str(n) for n in (256, 512, 1024, 2048, 4096, 8192))
            or int(configuration['MCAC']) != len(counts)):
        meta['warnings'].append('Recorded TLLD not applied: normal-spectrum MCAC mapping is not verified')
        return
    # Older firmware used one-based command values; do not silently reinterpret it.
    status_begin, status_end = '<<DPP STATUS>>', '<<DPP STATUS END>>'
    firmware = []
    if lines.count(status_begin) == lines.count(status_end) == 1:
        s, t = lines.index(status_begin), lines.index(status_end)
        if b < s < t:
            firmware = [m.groups() for line in lines[s+1:t]
                        for m in [re.fullmatch(r'Firmware:\s*(\d+)\.(\d+)(?:\s+Build:\s*\d+)?', line)] if m]
    if len(firmware) != 1 or tuple(map(int, firmware[0])) < (6, 6):
        meta['warnings'].append('Recorded TLLD not applied: firmware >=6.06 zero-based channel convention is not verified')
        return
    configuration['firmware_version'] = '.'.join(firmware[0])
    if 'SOFF' not in configuration:
        meta['warnings'].append('Recorded TLLD not applied: spectrum offset mapping is not verified')
        return
    offset = configuration['SOFF']
    if offset.upper() not in ('OF', 'OFF'):
        try:
            unshifted = float(offset) == 0.
        except ValueError:
            unshifted = False
        if not unshifted:
            meta['warnings'].append('Recorded TLLD not applied: nonzero spectrum offset mapping is not verified')
            return
    try:
        threshold = float(configuration['TLLD'])
    except ValueError as exc:
        raise ValueError('MCA TLLD must be OFF or a finite integer channel') from exc
    if not np.isfinite(threshold) or not threshold.is_integer() or not 0 <= threshold < len(counts):
        raise ValueError('MCA TLLD must be OFF or an integer channel in 0..MCAC-1; fractional thresholds are not defined')
    first = int(threshold)+1
    if np.any(counts[:first] != 0):
        raise ValueError('MCA TLLD metadata conflicts with nonzero counts in hardware-rejected channels')
    meta['hardware_lower_cutoff'] = {
        'source': 'Amptek_DP5_TLLD', 'threshold_device_channel': int(threshold),
        'first_retained_index': first, 'first_retained_channel': first+channel_start,
        'firmware_version': configuration['firmware_version'],
        'channel_convention': 'zero_based_device_indices; user_channel_origin_applied_separately',
        'limitation': 'Recorded channel LLD only; other trigger/acceptance effects are not inferred'}


def _load_mca(path, channel_start, fill):
    raw = path.read_bytes()
    if b'\x00' in raw:
        raise ValueError('Unknown binary format; provide a format sample or export CSV/MCA')
    warnings = []
    try:
        text = raw.decode('utf-8-sig')
        encoding = 'utf-8'
    except UnicodeDecodeError:
        # Amptek Windows exports can have a local-codepage degree sign in status headers.
        text = raw.decode('latin-1')
        encoding = 'latin-1'
        warnings.append('Non-UTF8 MCA headers decoded using latin-1; DATA remains strict ASCII numeric')
    lines = [s.strip() for s in text.splitlines()]
    if not lines or lines[0] != '<<PMCA SPECTRUM>>' or lines.count('<<DATA>>') != 1 or lines.count('<<END>>') != 1:
        raise ValueError('Unknown format: expected Amptek PMCA ASCII structure; export CSV/MCA')
    start, end = lines.index('<<DATA>>'), lines.index('<<END>>')
    if end <= start:
        raise ValueError('MCA END precedes DATA')
    data = lines[start+1:end]
    if not data or not all(re.fullmatch(r'[0-9]+(?:\.0+)?', s) for s in data):
        raise ValueError('MCA DATA must contain exactly one nonnegative integer per line')
    declared = re.findall(r'(?m)^\s*(?:MCAC\s*=|CHANNELS\s*[-=])\s*(\d+)', text)
    if any(int(n) != len(data) for n in declared):
        raise ValueError('MCA declared channel count disagrees with DATA length')
    meta = {'format': 'Amptek_PMCA_ASCII', 'encoding': encoding, 'channel_origin': channel_start,
            'warnings': warnings, 'header_lines': lines[:start], 'declared_channels': declared,
            'status_lines': lines[end+1:]}
    for label, key in [('LIVE_TIME', 'live_time_s'), ('REAL_TIME', 'real_time_s')]:
        values = re.findall(r'(?m)^' + label + r'\s*-\s*([\d.eE+-]+)', text)
        if values:
            val = float(values[0])
            if not np.isfinite(val) or val <= 0 or len(values) != 1:
                raise ValueError('Invalid MCA acquisition time: ' + label)
            meta[key] = val
    if '<<CALIBRATION>>' in lines:
        points = []
        for line in lines[lines.index('<<CALIBRATION>>')+1:start]:
            parts = line.split()
            if len(parts) == 2:
                try:
                    points.append([float(v) for v in parts])
                except ValueError:
                    pass
        meta['embedded_calibration_points'] = points
    channel, counts, _ = _validate(np.arange(len(data)) + channel_start, [float(s) for s in data], fill)
    _record_dp5_acquisition_domain(lines, end, counts, channel_start, meta)
    return Spectrum(channel, counts, meta.get('live_time_s'), meta)
