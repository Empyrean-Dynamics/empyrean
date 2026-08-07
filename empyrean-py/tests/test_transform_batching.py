"""`transform_coordinates` crosses the FFI once per table, not once per row.

``empyrean.transform_coordinates`` has always been table-in / table-out at
the Python surface, but the binding underneath used to walk the table and
call the single-state entry point once per row. It now marshals the whole
table and makes **one** batched call.

Two things have to hold for that to be a scheduling change rather than a
numerical one, and both are asserted here:

* **Bit-identity.** Element ``i`` of an ``N``-row result must equal, bit
  for bit, the same row transformed on its own — elements, covariance,
  epoch, and the representation / frame / origin tags.
* **No mixing.** Distinct inputs must stay distinct — a batch that
  applies one row's Jacobian to its neighbours, or reuses a row, is the
  failure mode a single-row comparison catches and a summary statistic
  does not.

Deliberately **not** asserted here: a wall-clock ceiling. Measured A/B
against the per-row loop, the batched form is ~17% faster at N=1000 and
above and indistinguishable at N=1 — the engine's gravitational-parameter
and origin-shift memos are scoped to the Context, so they already amortize
across successive single-state calls and the saving is the per-call FFI
crossing rather than the shift itself. A margin that size cannot carry a
timing assertion. Bit-identity is the contract; the call shape is verified
by reading the binding.
"""

import empyrean
import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CometaryCoordinates,
    Frame,
    KeplerianCoordinates,
    Origin,
)

# A small spread of bound heliocentric states. Rows 0 and 2 deliberately
# share an epoch so both the memo-hit and memo-miss arms of the engine's
# origin-shift cache are exercised inside one batch.
_EPOCHS = [60000.0, 60123.5, 60000.0, 60321.25]
_STATES = [
    [0.9412, 0.3311, -0.0142, -0.006_12, 0.016_10, 0.000_31],
    [-1.2734, 0.8845, 0.0517, -0.007_44, -0.009_02, 0.000_11],
    [2.1180, -1.4402, -0.1893, 0.004_51, 0.006_88, -0.000_27],
    [0.4021, -1.6553, 0.0904, 0.010_02, 0.003_15, -0.000_52],
]


def _covariance(seed: float) -> np.ndarray:
    """A symmetric positive-definite 6×6, distinct per row."""
    rng = np.random.default_rng(int(seed * 1000))
    a = rng.normal(size=(6, 6)) * 1e-6
    return a @ a.T + np.eye(6) * 1e-8


def _table(n: int, *, with_covariance: bool) -> CartesianCoordinates:
    """An ``n``-row Cartesian table in ICRF about the Sun."""
    states = np.array(_STATES[:n], dtype=np.float64)
    kwargs = {
        "epoch": np.array(_EPOCHS[:n], dtype=np.float64),
        "x": states[:, 0],
        "y": states[:, 1],
        "z": states[:, 2],
        "vx": states[:, 3],
        "vy": states[:, 4],
        "vz": states[:, 5],
        "frame": Frame.ICRF.value,
        "origin": [str(Origin.SUN)] * n,
    }
    if with_covariance:
        kwargs["covariance"] = empyrean.coordinates.covariance.CartesianCovariance.from_matrix(
            np.stack([_covariance(_EPOCHS[i]) for i in range(n)])
        )
    return CartesianCoordinates.from_kwargs(**kwargs)


def _columns(table) -> dict:
    """Every numeric column of a coordinate table, as raw arrays."""
    names = [c for c in table.table.column_names if c != "covariance"]
    out = {c: table.column(c).to_numpy(zero_copy_only=False) for c in names}
    out["__frame__"] = table.frame
    return out


# ── Bit-identity ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("target", "frame", "origin"),
    [
        # Representation only.
        (KeplerianCoordinates, None, None),
        # Frame only.
        (CartesianCoordinates, Frame.ECLIPTICJ2000, None),
        # Origin only — forces the SPK origin shift, the memoized step.
        (CartesianCoordinates, None, Origin.SSB),
        # Every axis at once.
        (KeplerianCoordinates, Frame.ECLIPTICJ2000, Origin.SSB),
        (CometaryCoordinates, Frame.ECLIPTICJ2000, Origin.SSB),
    ],
)
def test_batch_elements_are_bit_identical_to_single_rows(target, frame, origin):
    """Row ``i`` of a batch equals that row transformed alone, bit for bit.

    Sabotaging the binding's output loop (sourcing element 1 from element
    0, the shape of a reused-row bug) fails this immediately.
    """
    n = 4
    batch = empyrean.transform_coordinates(
        _table(n, with_covariance=False), target, frame=frame, origin=origin
    )
    assert len(batch) == n, "the batch must return exactly one row per input row"

    batch_cols = _columns(batch)
    for i in range(n):
        alone = empyrean.transform_coordinates(
            _table(n, with_covariance=False)[i : i + 1], target, frame=frame, origin=origin
        )
        alone_cols = _columns(alone)
        assert batch_cols["__frame__"] == alone_cols["__frame__"], (
            f"element {i}: frame attribute diverged"
        )
        for name, values in alone_cols.items():
            if name == "__frame__":
                continue
            got = batch_cols[name][i : i + 1]
            np.testing.assert_array_equal(
                got,
                values,
                err_msg=(
                    f"element {i}, column {name!r}: batch elements must be "
                    f"BIT-identical to the single-row call"
                ),
            )


def test_covariance_survives_the_batch_bit_identically():
    """Covariance is propagated through the Jacobian per row, unmixed.

    The batch is the interesting case: one shared Jacobian applied to
    every row, or a row's matrix landing on its neighbour, are both
    plausible batching bugs and both show up as a mismatch here.
    """
    n = 4
    table = _table(n, with_covariance=True)
    batch = empyrean.transform_coordinates(
        table, KeplerianCoordinates, frame=Frame.ECLIPTICJ2000, origin=Origin.SSB
    )
    assert batch.covariance is not None
    batch_matrices = batch.covariance.to_matrix()

    for i in range(n):
        alone = empyrean.transform_coordinates(
            table[i : i + 1], KeplerianCoordinates, frame=Frame.ECLIPTICJ2000, origin=Origin.SSB
        )
        np.testing.assert_array_equal(
            batch_matrices[i],
            alone.covariance.to_matrix()[0],
            err_msg=(
                f"element {i}: the batched covariance must be BIT-identical to the single-row call"
            ),
        )

    # Distinct inputs must stay distinct — a shared-Jacobian bug collapses
    # them and would otherwise pass the comparison above only by accident.
    assert not np.array_equal(batch_matrices[0], batch_matrices[1]), (
        "distinct input covariances must not come back identical"
    )


def test_empty_table_round_trips():
    """A zero-row table is a no-op, not a crash and not a null result."""
    empty = _table(4, with_covariance=False)[0:0]
    out = empyrean.transform_coordinates(empty, KeplerianCoordinates)
    assert len(out) == 0
