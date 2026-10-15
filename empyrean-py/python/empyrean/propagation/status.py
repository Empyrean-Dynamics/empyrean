"""Per-orbit delivery status for batch results.

The engine carries a per-input-orbit delivery verdict: one orbit's failure
no longer ends a batch — the failing orbit is recorded and every other
orbit delivers its rows, and the call still succeeds. This module surfaces
that verdict as a first-class quivr table shared by
:class:`~empyrean.propagation.result.PropagationResult` and
:class:`~empyrean.ephemeris.result.EphemerisResult`: the accounting is the
product, so nothing raises for a partially-delivered batch.
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np
import pyarrow as pa
import quivr as qv


class OrbitOutcome(str, enum.Enum):
    """Delivery outcome of one input orbit in a batch.

    Subclasses ``str`` (like :class:`~empyrean.coordinates.enums.Frame`) so
    the value serializes directly into the ``outcome`` quivr string column.
    The member names match the C ABI constants
    ``EMPYREAN_ORBIT_OUTCOME_DELIVERED`` / ``_FAILED`` and the Rust wrapper's
    ``OrbitOutcome::{Delivered, Failed}`` variants.
    """

    DELIVERED = "delivered"
    """The orbit was propagated; its rows are the ``[first_row, first_row +
    num_rows)`` block of the result's flat table (which may be empty)."""

    FAILED = "failed"
    """The orbit failed; it delivered no rows and carries a classified
    ``error_code`` and the engine's ``error_message``."""


class OrbitStatuses(qv.Table):
    """One row per input orbit, in input order — a batch's delivery table.

    Delivery is read from :attr:`outcome` alone, never from
    :attr:`num_rows`, which is zero both for a failed orbit and for a
    delivered orbit whose marker stop fired before its first requested
    epoch.
    """

    # The input orbit's identifier and its zero-based index in the batch.
    orbit_id = qv.LargeStringColumn()
    orbit_index = qv.Int64Column()
    # An :class:`OrbitOutcome` value: "delivered" or "failed".
    outcome = qv.LargeStringColumn()
    # The EMPYREAN_*_FAILURE_* classification on a failed orbit; null on a
    # delivered one.
    error_code = qv.Int32Column(nullable=True)
    # The engine's failure text on a failed orbit; null on a delivered one.
    error_message = qv.LargeStringColumn(nullable=True)
    # Offset of the orbit's first delivered row in the flat result table;
    # null on a failed orbit (it occupies no block).
    first_row = qv.Int64Column(nullable=True)
    # Delivered row count for this orbit (0 for a failed orbit).
    num_rows = qv.Int64Column()

    @property
    def all_delivered(self) -> bool:
        """Whether every input orbit delivered (no ``FAILED`` row)."""
        return all(o == OrbitOutcome.DELIVERED.value for o in self.outcome.to_pylist())


def build_orbit_statuses(outcomes: dict[str, Any]) -> OrbitStatuses:
    """Build an :class:`OrbitStatuses` table from the extension's outcomes
    sub-dict.

    The extension carries, per input orbit, an integer ``status`` (0
    delivered / 1 failed), an ``error_code`` / ``first_row`` with ``-1``
    sentinels for "not applicable", and an ``error_message`` that is empty
    on a delivered orbit. The sentinels become genuine Arrow nulls here.
    """
    orbit_ids: list[str] = list(outcomes["orbit_id"])
    status = np.asarray(outcomes["status"], dtype=np.int64)
    error_codes = np.asarray(outcomes["error_code"], dtype=np.int64)
    error_messages: list[str] = list(outcomes["error_message"])
    first_rows = np.asarray(outcomes["first_row"], dtype=np.int64)
    num_rows = np.asarray(outcomes["num_rows"], dtype=np.int64)
    n = len(orbit_ids)

    delivered = status == 0
    outcome_values = [
        OrbitOutcome.DELIVERED.value if d else OrbitOutcome.FAILED.value for d in delivered
    ]

    # error_code / first_row: the -1 sentinels become null; error_message:
    # the empty-string placeholder on a delivered orbit becomes null.
    error_code_col = pa.array(
        [int(c) for c in error_codes], type=pa.int32(), mask=(error_codes < 0)
    )
    first_row_col = pa.array([int(f) for f in first_rows], type=pa.int64(), mask=(first_rows < 0))
    error_message_col = [m if m else None for m in error_messages]

    return OrbitStatuses.from_kwargs(
        orbit_id=orbit_ids,
        orbit_index=np.arange(n, dtype=np.int64),
        outcome=outcome_values,
        error_code=error_code_col,
        error_message=error_message_col,
        first_row=first_row_col,
        num_rows=num_rows,
    )
