"""The joint solved-parameter covariance — cross terms beyond the diagonal blocks.

A fit over the state and *P* parameters produces one ``(6+P) x (6+P)``
matrix. Its diagonal blocks have always crossed into Python: the ``6x6``
on the coordinates, the Marsden ``3x3`` on
:class:`~empyrean.orbits.nongrav.NonGravParams`, a DT variance, an AMRAT
variance, a per-segment thrust ``3x3``. The **off-diagonal** blocks are
what this module carries.

Leaving them behind is not a conservative simplification. A
block-diagonal covariance asserts that the data which produced the state
and the data which produced ``A2`` were independent, when they are the
same observations through the same fit. And a *propagated* joint has
non-zero state-parameter columns even when the input was block-diagonal,
because propagation itself generates that correlation — so a chained
propagation built on the ``6x6`` alone reports a tighter uncertainty than
the first leg supports.

The four homes
--------------

One covariance entry belongs to exactly one place:

===============================  ==========================================
 block                            home
===============================  ==========================================
 state-state                      ``coordinates.covariance``
 ``A_i``-``A_j``                  ``non_grav.covariance``
 state-``A_i``                    ``non_grav.non_grav_cross``
 ``dv_i``-``dv_j``, same segment  that segment's own ``3x3``
 everything else                  :class:`WideCross`
===============================  ==========================================

Note the placement divergence, called out rather than left to be
discovered: at the C ABI the state-Marsden border rides the *coordinate*,
beside the ``6x6`` it borders. Here it rides
:class:`~empyrean.orbits.nongrav.NonGravParams`, because this package's
covariance sub-table is a synthesized 21-column class with no natural slot
for an 18-value list. Same name, same contents, same units.

Why identity tags
-----------------

Entries name the **parameter**, never a column index. Which column a
parameter occupies depends on which *other* parameters the orbit declares
— adding an SRP AMRAT shifts the thrust columns by one — so an index
recorded against one orbit is wrong against the next, and the failure is
silent: every number finite, every gate passed, one parameter's
correlations attached to another.

Tags are rendered exactly as the engine renders them: ``"A1"``, ``"A2"``,
``"A3"``, ``"DT"``, ``"AMRAT"``, ``"thrust[0].x"``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import quivr as qv

__all__ = ["WideCross"]


class WideCross(qv.Table):
    """Row-aligned cross-covariance terms beyond the state+Marsden ``9x9``.

    Attached to each orbit table as a nullable sub-table column, so every
    existing table and round trip is unchanged: an orbit with no cross
    terms carries nulls in these columns.

    Storage follows the package's existing variable-width idiom (see
    :class:`~empyrean.ephemeris.sensitivity.ObservationSensitivities`):
    flat ``LargeListColumn`` payloads with the width recoverable from the
    data. Here the width IS ``len(columns)`` and every entry is
    self-identifying, so no companion width column is carried — a second
    source of truth for a number already in the data. Rows may differ in
    width freely, because nothing indexes across them.

    Absence
    -------

    A nullable sub-table column is **never** ``None`` — quivr returns a
    table of parent length regardless — so absence is per-row nulls, not
    a null sub-table. Test ``row_is_empty`` or the per-row accessors,
    never ``orbits.wide_cross is None``.

    A supplied entry whose six values are all zero is a **supplied zero
    correlation**, not an absence: it engages the engine's definiteness
    gate. To mean absent, omit the entry.

    Cross terms travel with the blocks they condition
    -------------------------------------------------

    A cross term is one half of a matrix whose other half is a parameter
    block, and the engine **refuses** the half without the whole: a
    state-Marsden border supplied on an orbit carrying no non-grav
    covariance is an error, not an ignored field. The same holds for the
    carrier's columns — a state-DT column needs the DT prior variance, a
    state-AMRAT column needs the SRP AMRAT prior variance.

    This matters when chaining a propagation. A propagated state carries
    the *propagated* cross terms but not the parameter blocks, which
    propagation passes through unchanged rather than restating on every
    output row, so a second leg is assembled from the output row plus
    the parameter blocks of the orbit that started the chain.
    """

    # Parameter identity per state-cross column, in the engine's own
    # rendering. Length is the number of state-cross columns on that row.
    columns = qv.LargeListColumn(qv.LargeStringColumn(), nullable=True)
    # 6 * len(columns), row-major: entry k covers positions 6k..6k+6, in
    # the coordinate's own element order, basis AND angular unit.
    state = qv.LargeListColumn(qv.Float64Column(), nullable=True)
    # Parameter-parameter terms: three parallel lists of equal length.
    pair_a = qv.LargeListColumn(qv.LargeStringColumn(), nullable=True)
    pair_b = qv.LargeListColumn(qv.LargeStringColumn(), nullable=True)
    pair_value = qv.LargeListColumn(qv.Float64Column(), nullable=True)

    def row_is_empty(self, i: int) -> bool:
        """Whether row ``i`` carries no cross terms at all.

        The per-row absence test. ``WideCross`` itself is never ``None``
        on a parent table, so this is what "this orbit has no joint"
        looks like.
        """
        cols = self.columns[i].as_py()
        pairs = self.pair_a[i].as_py()
        return not cols and not pairs

    def state_cross(self, i: int) -> dict[str, npt.NDArray[np.float64]]:
        """Row ``i``'s state-parameter columns, keyed by parameter tag.

        Each value is the 6-vector of covariances between the six state
        elements and that parameter, in the coordinate's own element
        order and units. Returns an empty dict when the row carries none.

        Reshaping accessor for the flat ``state`` payload — the same
        shape :meth:`ObservationSensitivities.jacobians_array` provides
        for its flat matrices, minus the homogeneity guard: identity tags
        let rows differ in width, because nothing indexes across them.
        """
        tags = self.columns[i].as_py()
        if not tags:
            return {}
        flat = np.asarray(self.state[i].as_py(), dtype=np.float64)
        expected = 6 * len(tags)
        if flat.size != expected:
            raise ValueError(
                f"row {i}: the state payload has {flat.size} values but "
                f"{len(tags)} tagged column(s) need {expected}. The payload is "
                "row-major 6-per-column; a mismatch means the two were written "
                "out of step, and reshaping anyway would attach one parameter's "
                "covariances to another."
            )
        block = flat.reshape(len(tags), 6)
        return {tag: block[k] for k, tag in enumerate(tags)}

    def param_cross(self, i: int) -> dict[tuple[str, str], float]:
        """Row ``i``'s parameter-parameter terms, keyed by ``(a, b)``.

        The key is canonicalized so ``(a, b)`` and ``(b, a)`` are one
        entry — the term is symmetric, and carrying it twice would be two
        numbers for one covariance.
        """
        a = self.pair_a[i].as_py() or []
        b = self.pair_b[i].as_py() or []
        v = self.pair_value[i].as_py() or []
        if not (len(a) == len(b) == len(v)):
            raise ValueError(
                f"row {i}: pair_a / pair_b / pair_value have lengths "
                f"{len(a)} / {len(b)} / {len(v)}. They are three parallel lists "
                "of one term each; unequal lengths mean the row was written "
                "inconsistently and any pairing of them would be invented."
            )
        return {(x, y) if x <= y else (y, x): float(w) for x, y, w in zip(a, b, v, strict=True)}

    @classmethod
    def from_entries(
        cls,
        state_cross: list[dict[str, npt.NDArray[np.float64] | list[float]] | None],
        param_cross: list[dict[tuple[str, str], float] | None],
    ) -> WideCross:
        """Build a table from per-row dicts, nulling rows that carry nothing.

        ``state_cross[i]`` maps a parameter tag to that row's 6-vector;
        ``param_cross[i]`` maps a canonical ``(a, b)`` pair to its value.
        Either may be ``None`` or empty for a row with no cross terms,
        which is written as nulls rather than as empty lists — empty
        lists would read as "supplied, and empty".
        """
        if len(state_cross) != len(param_cross):
            raise ValueError(
                f"state_cross has {len(state_cross)} rows and param_cross has "
                f"{len(param_cross)}; they are parallel per-orbit lists"
            )
        cols: list[list[str] | None] = []
        flat: list[list[float] | None] = []
        pa_: list[list[str] | None] = []
        pb_: list[list[str] | None] = []
        pv_: list[list[float] | None] = []
        for sc, pc_ in zip(state_cross, param_cross, strict=True):
            if sc:
                tags = sorted(sc)
                cols.append(tags)
                flat.append([float(v) for tag in tags for v in np.asarray(sc[tag]).ravel()])
            else:
                cols.append(None)
                flat.append(None)
            if pc_:
                keys = sorted(pc_)
                pa_.append([k[0] for k in keys])
                pb_.append([k[1] for k in keys])
                pv_.append([float(pc_[k]) for k in keys])
            else:
                pa_.append(None)
                pb_.append(None)
                pv_.append(None)
        return cls.from_kwargs(columns=cols, state=flat, pair_a=pa_, pair_b=pb_, pair_value=pv_)
