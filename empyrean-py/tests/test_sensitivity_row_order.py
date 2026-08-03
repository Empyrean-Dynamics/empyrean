"""The observation-Jacobian row order is a cross-layer contract — bd empyrean-9666l.

The engine returns a ``[6][n_params]`` observation Jacobian whose six rows are
``[range, RA, Dec, range-rate, RA-rate, Dec-rate]``, and every distribution
layer marshals it through unreordered. None of them used to say so, and a
consumer that read row 0 as RA got a range partial in AU instead of an RA
partial in degrees — finite, plausible, and wrong in both observable and unit.

The ``SENSITIVITY_ROW_*`` constants exist so nobody hand-indexes it. These
tests pin their values (a rotation is a breaking change for every consumer and
must be seen as one) and pin the C-ABI header as the shared source of truth,
so the Python names cannot drift away from the ones the C and Rust layers
publish under the same spelling.
"""

from __future__ import annotations

import re
from pathlib import Path

import empyrean
import pytest
from empyrean.ephemeris.sensitivity import (
    SENSITIVITY_ROW_DEC,
    SENSITIVITY_ROW_RA,
    SENSITIVITY_ROW_RANGE,
    SENSITIVITY_ROW_VDEC,
    SENSITIVITY_ROW_VRA,
    SENSITIVITY_ROW_VRANGE,
)

# Row name -> contract value. The single table every assertion below reads.
CONTRACT = {
    "RANGE": (SENSITIVITY_ROW_RANGE, 0),
    "RA": (SENSITIVITY_ROW_RA, 1),
    "DEC": (SENSITIVITY_ROW_DEC, 2),
    "VRANGE": (SENSITIVITY_ROW_VRANGE, 3),
    "VRA": (SENSITIVITY_ROW_VRA, 4),
    "VDEC": (SENSITIVITY_ROW_VDEC, 5),
}

# The generated C header, when this is a source checkout rather than an
# installed wheel. Skipped rather than failed when absent: a wheel is a
# legitimate way to run this suite and carries no header.
HEADER = Path(__file__).resolve().parents[2] / "include" / "empyrean.h"


@pytest.mark.parametrize(
    ("name", "constant", "expected"), [(n, c, e) for n, (c, e) in CONTRACT.items()]
)
def test_row_constant_has_its_contract_value(name: str, constant: int, expected: int) -> None:
    assert constant == expected, f"SENSITIVITY_ROW_{name} moved off its contract value"


def test_the_six_constants_cover_all_six_rows() -> None:
    """A duplicate or an out-of-range index would slice the wrong
    observable rather than fail, so distinctness is the assertion."""
    assert sorted(c for c, _ in CONTRACT.values()) == [0, 1, 2, 3, 4, 5]


def test_constants_are_reachable_from_the_package_root() -> None:
    """They document how to index a root-exported table, so they are
    importable from the same place that table is."""
    for name, (constant, _) in CONTRACT.items():
        assert getattr(empyrean, f"SENSITIVITY_ROW_{name}") == constant


def test_python_and_c_abi_agree_on_every_row() -> None:
    """The C header is the shipped contract; the Python constants are a
    mirror of it. A layer that rotates alone is exactly the failure the
    original defect had no way to surface."""
    if not HEADER.is_file():
        pytest.skip(f"generated C header not present at {HEADER} (installed-wheel run)")
    text = HEADER.read_text(encoding="utf-8")
    for name, (constant, _) in CONTRACT.items():
        define = f"EMPYREAN_SENSITIVITY_ROW_{name}"
        match = re.search(rf"^#define {define} (\d+)$", text, re.MULTILINE)
        assert match is not None, f"{define} is missing from {HEADER}"
        assert int(match.group(1)) == constant, (
            f"{define} disagrees between the C header and the Python constant"
        )
