"""Reject ambiguous input columns before pandas automatically renames them."""
import csv

import pytest

from xrf_analyzer.atomic_lines import load_sensitivity
from xrf_analyzer.calibration import fit_linear_calibration
from xrf_analyzer.io_spectrum import load_spectrum


@pytest.mark.parametrize('reader,text', [
    (load_spectrum, 'channel,counts,counts\n0,2,200\n1,3,300\n'),
    (fit_linear_calibration, 'channel,energy_keV,energy_keV\n100,2,3\n900,18,27\n'),
])
def test_duplicate_input_header_is_not_silently_renamed(tmp_path, reader, text):
    path = tmp_path/'input.csv'
    path.write_text(text, encoding='utf-8-sig')
    with pytest.raises(ValueError, match='Duplicate CSV'):
        reader(path)


def test_duplicate_sensitivity_branch_is_not_silently_ignored(synthetic):
    _, _, path, _, _, _ = synthetic()
    with path.open(newline='', encoding='utf-8-sig') as source:
        rows = list(csv.reader(source))
    rows[0].append('Kalpha1')
    for row in rows[1:]:
        row.append('999')
    with path.open('w', newline='', encoding='utf-8-sig') as output:
        csv.writer(output).writerows(rows)
    with pytest.raises(ValueError, match='Duplicate CSV'):
        load_sensitivity(path)
