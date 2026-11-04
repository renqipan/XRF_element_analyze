"""Read CSV input without allowing pandas to hide duplicate column names."""
from collections import Counter
import csv
from pathlib import Path

import pandas as pd


def read_csv_unique_headers(path, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8-sig')
    with Path(path).open(encoding=encoding, newline='') as source:
        for header in csv.reader(source):
            # Match pandas' default skip_blank_lines behavior before the header.
            if header and (len(header) > 1 or header[0].strip()):
                duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
                if duplicates:
                    raise ValueError('Duplicate CSV column names: ' + ', '.join(duplicates))
                break
    return pd.read_csv(path, **kwargs)
