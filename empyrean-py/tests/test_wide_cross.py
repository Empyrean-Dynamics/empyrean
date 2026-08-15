"""The joint cross-covariance surface in Python.

Covers the representation itself — tags, reshaping, absence, and the
refusals that keep a malformed row from being reshaped into plausible
numbers — and then the marshaling end to end: what a propagation and a
fit put into these columns, and what a second leg gets when the columns
are fed back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    Epochs,
    NonGravParams,
    Origin,
    SRPParams,
    UncertaintyMethod,
    compute_b_planes,
    compute_impact_probabilities,
    determine,
    generate_ephemeris,
    propagate,
    read_ades,
    refine,
)
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.observers.observers import Observers
from empyrean.od.disposition import ParamDisposition
from empyrean.od.result import ODConfig, SolveFor
from empyrean.orbits.wide_cross import WideCross

FIXTURES = Path(__file__).parent / "fixtures"
APOPHIS_MULTIAPP = FIXTURES / "99942_apophis_multiapp.psv"

# Apophis at MJD 61000.0 TDB, heliocentric ecliptic — the same state the
# forcing-function fixture uses, so the two cannot drift apart in what
# they claim about the same object.
APOPHIS_STATE = {
    "epoch": 61000.0,
    "x": -7.85264914906904643e-02,
    "y": -8.19748051902064567e-01,
    "z": 4.18939515323390882e-02,
    "vx": 1.98751024968884596e-02,
    "vy": 1.32208844536140196e-03,
    "vz": 3.99496044422352188e-04,
}

# An Apophis-scale area-to-mass ratio (a ~370 m rock at ~2.7e10 kg) and a
# ~10% prior sigma. The AMRAT axis is what widens the layout past the
# state+Marsden 9x9: its state-cross column has no home but the carrier.
_AMRAT = 4.0e-6
_AMRAT_VARIANCE = 1.6e-13

# An inert Marsden block: a declared 3x3 with zero coefficients. Declared
# is what matters — it is the parameter block the state-Marsden border is
# conditioned on — while zero coefficients keep the force itself out of
# the trajectory, so a chained-propagation difference can only come from
# the covariance.
_MARSDEN_VARIANCES = [1e-20, 1e-20, 1e-20]


def test_an_orbit_table_carries_the_joint_columns() -> None:
    """The sub-table and the border attach to every orbit table."""
    schema = CartesianOrbits.empty().table.schema
    names = [f.name for f in schema]
    assert "wide_cross" in names

    wide = next(f for f in schema if f.name == "wide_cross")
    assert [x.name for x in wide.type] == [
        "columns",
        "state",
        "pair_a",
        "pair_b",
        "pair_value",
    ]

    # The border rides NonGravParams, beside the 3x3 it conditions.
    non_grav = next(f for f in schema if f.name == "non_grav")
    assert "non_grav_cross" in [x.name for x in non_grav.type]


def test_state_columns_reshape_by_tag() -> None:
    """The flat payload reshapes to one 6-vector per tagged parameter."""
    wc = WideCross.from_entries(
        state_cross=[{"DT": np.arange(6.0), "AMRAT": np.arange(6.0) + 10.0}],
        param_cross=[None],
    )
    got = wc.state_cross(0)
    assert set(got) == {"AMRAT", "DT"}
    # Tags are sorted on write, so AMRAT's block precedes DT's in the
    # flat payload; the accessor must key by tag, not by position.
    np.testing.assert_array_equal(got["DT"], np.arange(6.0))
    np.testing.assert_array_equal(got["AMRAT"], np.arange(6.0) + 10.0)


def test_pairs_are_symmetric_and_canonicalized() -> None:
    wc = WideCross.from_entries(state_cross=[None], param_cross=[{("DT", "AMRAT"): 1.5}])
    pairs = wc.param_cross(0)
    # Canonical order, whichever way it was supplied.
    assert pairs == {("AMRAT", "DT"): 1.5}


def test_absence_is_per_row_nulls_not_a_none_subtable() -> None:
    """A nullable sub-table is never None; absence is per-row."""
    orbits = CartesianOrbits.empty()
    # The quivr semantic this surface has to respect: the attribute is a
    # table of parent length regardless, so `is not None` is always true
    # and says nothing about content.
    assert orbits.wide_cross is not None

    wc = WideCross.from_entries(state_cross=[{"DT": np.zeros(6)}, None], param_cross=[None, None])
    assert not wc.row_is_empty(0)
    assert wc.row_is_empty(1)


def test_a_supplied_zero_entry_is_not_absence() -> None:
    """All-zero values are a supplied zero correlation, not an absence.

    The engine reads a supplied zero entry as a claim and runs the
    definiteness gate on it; only omitting the entry means absent.
    """
    wc = WideCross.from_entries(state_cross=[{"DT": np.zeros(6)}], param_cross=[None])
    assert not wc.row_is_empty(0)
    np.testing.assert_array_equal(wc.state_cross(0)["DT"], np.zeros(6))


def test_a_mismatched_payload_is_refused_rather_than_reshaped() -> None:
    """A row whose payload and tags disagree must not be reshaped.

    Reshaping anyway would attach one parameter's covariances to
    another — finite, plausible, and wrong.
    """
    wc = WideCross.from_kwargs(
        columns=[["DT", "AMRAT"]],
        state=[[1.0] * 6],  # one column's worth for two tags
        pair_a=[None],
        pair_b=[None],
        pair_value=[None],
    )
    with pytest.raises(ValueError, match="need 12"):
        wc.state_cross(0)


def test_unequal_pair_lists_are_refused() -> None:
    wc = WideCross.from_kwargs(
        columns=[None],
        state=[None],
        pair_a=[["DT"]],
        pair_b=[["AMRAT", "A1"]],
        pair_value=[[1.0]],
    )
    with pytest.raises(ValueError, match="parallel lists"):
        wc.param_cross(0)


class TestParamDisposition:
    """The tri-state, and its refusal to be a boolean."""

    def test_the_three_tags(self) -> None:
        assert ParamDisposition.SOLVED.value == "solved"
        assert ParamDisposition.CONSIDERED.value == "considered"
        assert ParamDisposition.FIXED.value == "fixed"
        assert ParamDisposition.parse("considered") is ParamDisposition.CONSIDERED

    def test_a_bool_is_refused_by_name(self) -> None:
        """`False` cannot say whether an axis is considered or fixed."""
        for value in (True, False):
            with pytest.raises(TypeError, match="not a bool"):
                ParamDisposition.parse(value)

    def test_an_unknown_tag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter disposition"):
            ParamDisposition.parse("estimated")

    def test_considered_is_not_solved(self) -> None:
        assert ParamDisposition.CONSIDERED.is_considered
        assert not ParamDisposition.CONSIDERED.is_solved
        assert not ParamDisposition.FIXED.is_solved


# ── The marshaling, end to end ───────────────────────────────────────
#
# Everything below runs the engine. The unit tests above pin what the
# columns MEAN; these pin that the columns are actually filled, that
# what fills them is the engine's own joint rather than something
# reconstructed on this side, and that a second leg handed those columns
# propagates a different covariance than one handed the 6x6 alone.


def _wide_layout_orbit() -> CartesianOrbits:
    """Apophis with a covariance, a declared Marsden block, and an AMRAT
    prior — the narrowest input whose joint needs both homes.

    The Marsden block alone would not do it: a Marsden-only layout's
    cross terms fit entirely inside the 6x3 border, and the carrier
    stays empty. The AMRAT axis is what puts a column in the carrier.
    """
    s = APOPHIS_STATE
    covariance = CartesianCovariance.from_matrix(
        np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])[None, :, :]
    )
    coords = CartesianCoordinates.from_kwargs(
        epoch=[s["epoch"]],
        x=[s["x"]],
        y=[s["y"]],
        z=[s["z"]],
        vx=[s["vx"]],
        vy=[s["vy"]],
        vz=[s["vz"]],
        covariance=covariance,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["WIDE"],
        object_id=["99942"],
        coordinates=coords,
        non_grav=NonGravParams.from_kwargs(
            a1=[0.0],
            a2=[0.0],
            a3=[0.0],
            model=["inverse_square"],
            covariance=[np.diag(_MARSDEN_VARIANCES).reshape(9).tolist()],
        ),
        srp=SRPParams.from_kwargs(amrat=[_AMRAT], cr=[1.0], amrat_variance=[_AMRAT_VARIANCE]),
    )


def _gravity_only_orbit() -> CartesianOrbits:
    """The same state with no parameter block at all."""
    wide = _wide_layout_orbit()
    return CartesianOrbits.from_kwargs(
        orbit_id=["GRAVITY_ONLY"],
        object_id=["99942"],
        coordinates=wide.coordinates,
    )


_EPOCHS = Epochs.from_mjd(np.array([61000.0 + 30.0 * i for i in range(4)]), scale="tdb")


class TestPropagatedJoint:
    """What a propagation puts into the two homes."""

    def test_a_propagated_state_carries_the_joint_it_computed(self) -> None:
        """Both homes are filled on every output row, tagged by identity.

        The values themselves are the engine's: this asserts they
        arrived finite and correctly shaped, and that the tag says AMRAT
        rather than a column index — an index recorded against one orbit
        is wrong against the next.
        """
        result = propagate(_wide_layout_orbit(), _EPOCHS)
        states = result.states

        assert states.non_grav is not None
        borders = states.non_grav.non_grav_cross.to_pylist()
        assert all(b is not None for b in borders), (
            "the propagated state-Marsden border is non-zero even from a "
            "block-diagonal input, because propagation generates the "
            f"correlation; got {borders}"
        )
        for i, b in enumerate(borders):
            assert len(b) == 18, f"row {i}: the border is 6x3 row-major, got {len(b)}"
            assert np.all(np.isfinite(b)), f"row {i}: non-finite border value"

        wide = states.wide_cross
        for i in range(len(states)):
            assert not wide.row_is_empty(i), f"row {i} carries no carrier entry"
            state_cross = wide.state_cross(i)
            assert set(state_cross) == {"AMRAT"}, (
                "the state-AMRAT column has no home but the carrier, and this "
                f"orbit declares an AMRAT prior; got {sorted(state_cross)}"
            )
            assert state_cross["AMRAT"].shape == (6,)
            assert np.all(np.isfinite(state_cross["AMRAT"]))
            # The Marsden block and AMRAT are both solved axes, so their
            # mixed pairs live here too — the one place they can.
            assert set(wide.param_cross(i)) == {
                ("A1", "AMRAT"),
                ("A2", "AMRAT"),
                ("A3", "AMRAT"),
            }

    def test_propagation_generates_the_correlation_it_reports(self) -> None:
        """The claim the whole surface rests on, measured.

        The input is block-diagonal: nothing correlates the state with
        AMRAT at t0, and the column there is exactly zero. It is
        non-zero at every later epoch because propagation itself
        generates the correlation — which is why a second leg handed
        only the 6x6 reports a tighter uncertainty than the first leg
        supports.

        Doubles as the check that the marshal carries each epoch's own
        row rather than broadcasting one: a broadcast would make these
        columns identical.
        """
        result = propagate(_wide_layout_orbit(), _EPOCHS)
        columns = [result.states.wide_cross.state_cross(i)["AMRAT"] for i in range(len(_EPOCHS))]

        np.testing.assert_array_equal(
            columns[0],
            np.zeros(6),
            err_msg="the input is block-diagonal, so the state-AMRAT column is zero at t0",
        )
        for i, column in enumerate(columns[1:], start=1):
            assert np.any(column != 0.0), (
                f"epoch {i}: the state-AMRAT column is still all zero after "
                "propagation, so nothing generated the correlation this surface "
                "exists to carry"
            )
        assert not np.array_equal(columns[1], columns[-1]), (
            "two different epochs report an identical state-AMRAT column, which "
            "means the marshal is broadcasting one row rather than carrying "
            "each epoch's own"
        )

    def test_a_gravity_only_propagation_reports_absence_not_zeros(self) -> None:
        """No declared parameter block, so no cross terms — and absence
        is null, never a row of zeros.

        A zero row is a supplied zero correlation that engages the
        engine's definiteness gate on a re-feed; writing one here would
        put a claim in the caller's mouth.
        """
        states = propagate(_gravity_only_orbit(), _EPOCHS).states
        for i in range(len(states)):
            assert states.wide_cross.row_is_empty(i)
        if states.non_grav is not None:
            assert all(b is None for b in states.non_grav.non_grav_cross.to_pylist())

    def test_dropping_the_marshal_hop_reinstates_the_silent_drop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sabotage: with the population hop removed, the columns go
        100% null — which is exactly what the no-silent-drops walk flags.

        Without this, a test asserting the columns are populated cannot
        distinguish "the marshal fills them" from "quivr happened to
        fill them", and the forcing function's green would not be
        attributable to this wiring.
        """
        import empyrean.propagation.propagate as propagate_module

        monkeypatch.setattr(
            propagate_module,
            "joint_columns_from_result",
            lambda result, n, prefix="": (None, None),
        )
        states = propagate(_wide_layout_orbit(), _EPOCHS).states

        n = len(states)
        # BOTH homes, not just the carrier. The marshal returns them as a
        # pair, so a regression that dropped only the border would leave
        # the carrier populated and a carrier-only assertion green while
        # `non_grav.non_grav_cross` silently reverted to all-null.
        assert states.wide_cross.columns.null_count == n
        assert states.wide_cross.state.null_count == n
        assert states.wide_cross.pair_a.null_count == n
        border_gone = states.non_grav is None or (states.non_grav.non_grav_cross.null_count == n)
        assert border_gone, (
            "with the marshal hop dropped the state-Marsden border must be "
            "100% null too — if it is not, something else is filling it and "
            "this test is not measuring the wiring it names"
        )


class TestTaggedReadbackJoint:
    """The joint on the provenance-tagged readback."""

    def test_the_tagged_readback_reports_the_same_joint_as_the_states(self) -> None:
        """Two surfaces, one engine row: they must not disagree.

        The tagged readback fetches its joint through a separate engine
        call from the one the states table uses. Both address the same
        (orbit, epoch), so any difference here is a marshaling defect on
        one side — and a caller reading the "honest" covariance would get
        cross terms that do not belong to it.
        """
        result = propagate(
            _wide_layout_orbit(), _EPOCHS, uncertainty_method="first_order", tagged_covariance=True
        )
        tagged = result.tagged_covariance
        assert tagged is not None

        state_borders = result.states.non_grav.non_grav_cross.to_pylist()
        tagged_borders = tagged.non_grav_cross.to_pylist()
        assert tagged_borders == state_borders

        for i in range(len(tagged)):
            np.testing.assert_array_equal(
                tagged.wide_cross.state_cross(i)["AMRAT"],
                result.states.wide_cross.state_cross(i)["AMRAT"],
            )

    def test_the_series_view_carries_the_joint_too(self) -> None:
        """``to_series`` must not be the place the joint falls off.

        The per-epoch dataclass is the ergonomic view of the same rows;
        a joint present in the table and absent from the series would be
        a drop that only shows up for callers who use the friendlier
        accessor.
        """
        result = propagate(
            _wide_layout_orbit(), _EPOCHS, uncertainty_method="first_order", tagged_covariance=True
        )
        series = result.tagged_covariance_series(0)
        assert len(series) == len(_EPOCHS)
        for entry in series:
            assert entry.non_grav_cross is not None
            assert entry.non_grav_cross.shape == (6, 3)
            assert set(entry.state_cross) == {"AMRAT"}
            assert entry.state_cross["AMRAT"].shape == (6,)

    def test_the_sampled_path_reports_what_it_has(self) -> None:
        """Sigma-point recovers the joint from its propagated cloud.

        Not a structural absence: a sample-based covariance still has
        state-parameter columns, and reporting none would understate the
        uncertainty of any leg chained onto it.
        """
        result = propagate(
            _wide_layout_orbit(),
            _EPOCHS,
            uncertainty_method=UncertaintyMethod.SIGMA_POINT,
            tagged_covariance=True,
        )
        tagged = result.tagged_covariance
        assert tagged is not None
        assert all(b is not None for b in tagged.non_grav_cross.to_pylist())
        for i in range(len(tagged)):
            assert set(tagged.wide_cross.state_cross(i)) == {"AMRAT"}


# ── Round trips ──────────────────────────────────────────────────────
#
# The two chains a user actually runs. Both end in the same
# discriminating shape: the same chain run WITHOUT the joint gives a
# different covariance. If the columns were dropped, or re-fed as zeros,
# the two runs would agree and these tests would pass vacuously.


def _relink(
    states: CartesianOrbits, row: int, parameter_source: CartesianOrbits
) -> CartesianOrbits:
    """Build a leg-2 orbit from one output row of leg 1.

    A propagation output carries the *propagated* cross terms but not
    the parameter blocks they are conditioned on — those are inputs that
    propagation passes through unchanged, so they come from the orbit
    that started the chain rather than being restated on every output
    row. This is the assembly a caller does, written once here.
    """
    ng = parameter_source.non_grav
    return CartesianOrbits.from_kwargs(
        orbit_id=[states.orbit_id[row].as_py()],
        object_id=[states.object_id[row].as_py()],
        coordinates=states.coordinates[row : row + 1],
        non_grav=NonGravParams.from_kwargs(
            a1=[ng.a1[0].as_py()],
            a2=[ng.a2[0].as_py()],
            a3=[ng.a3[0].as_py()],
            model=[ng.model[0].as_py()],
            covariance=[ng.covariance[0].as_py()],
            # The propagated border, from leg 1's output row.
            non_grav_cross=[states.non_grav.non_grav_cross[row].as_py()],
        ),
        srp=parameter_source.srp,
        wide_cross=states.wide_cross[row : row + 1],
    )


def _strip_joint(orbits: CartesianOrbits) -> CartesianOrbits:
    """The same orbit with both cross-term homes emptied.

    The chain a caller was forced into before these columns existed: the
    state block, and the parameter blocks, but nothing off-diagonal.
    """
    ng = orbits.non_grav
    return CartesianOrbits.from_kwargs(
        orbit_id=orbits.orbit_id.to_pylist(),
        object_id=orbits.object_id.to_pylist(),
        coordinates=orbits.coordinates,
        non_grav=NonGravParams.from_kwargs(
            a1=ng.a1.to_pylist(),
            a2=ng.a2.to_pylist(),
            a3=ng.a3.to_pylist(),
            model=ng.model.to_pylist(),
            covariance=ng.covariance.to_pylist(),
        ),
        srp=orbits.srp,
    )


def _position_variance(states: CartesianOrbits, row: int) -> float:
    """Trace of the position block at one output row."""
    matrices = states.coordinates.covariance.to_matrix()
    return float(sum(matrices[row][i][i] for i in range(3)))


def test_a_second_leg_consumes_the_first_legs_joint() -> None:
    """propagate -> re-feed -> propagate, against the single-leg answer.

    The reference is the same object propagated straight through in one
    leg: a chain that carries the joint should reproduce it, because
    splitting a propagation in two is a bookkeeping choice, not a
    physical one. A chain that drops the cross terms cannot — it hands
    leg 2 a block-diagonal covariance, asserting the state and the
    parameters were independent when one propagation produced both, and
    the result is a *tighter* uncertainty than the run supports.

    Both directions are asserted. Agreement alone would pass if the
    cross terms were negligible here; the second assertion is what shows
    they are not.
    """
    seed = _wide_layout_orbit()
    leg1 = propagate(seed, _EPOCHS).states
    last = len(leg1) - 1
    handover = leg1.coordinates.epoch[last].as_py()

    with_joint = _relink(leg1, last, seed)
    assert not with_joint.wide_cross.row_is_empty(0)
    without_joint = _strip_joint(with_joint)

    leg2_epochs = Epochs.from_mjd(np.array([handover, handover + 60.0]), scale="tdb")
    chained = _position_variance(propagate(with_joint, leg2_epochs).states, 1)
    block_diagonal = _position_variance(propagate(without_joint, leg2_epochs).states, 1)
    single_leg = _position_variance(
        propagate(
            seed,
            Epochs.from_mjd(np.array([APOPHIS_STATE["epoch"], handover + 60.0]), scale="tdb"),
        ).states,
        1,
    )

    assert np.isfinite(chained) and chained > 0.0
    assert chained == pytest.approx(single_leg, rel=1e-9), (
        f"the chained propagation ({chained:e}) does not reproduce the "
        f"single-leg answer ({single_leg:e}); carrying the joint across the "
        "handover is what makes the split invisible"
    )
    assert block_diagonal < chained, (
        f"dropping the cross terms must UNDERSTATE the uncertainty, and here it "
        f"reports {block_diagonal:e} against {chained:e}"
    )
    assert abs(chained - block_diagonal) / chained > 1e-3, (
        f"the block-diagonal chain differs from the joint chain by only "
        f"{abs(chained - block_diagonal) / chained:e}, so this fixture cannot "
        "tell a dropped joint from a carried one"
    )


def test_a_chained_leg_carries_its_own_joint_onward() -> None:
    """Leg 2's output is itself re-feedable.

    A chain that consumed a joint but stopped producing one would end
    silently after two legs.
    """
    seed = _wide_layout_orbit()
    leg1 = propagate(seed, _EPOCHS).states
    leg2 = propagate(
        _relink(leg1, len(leg1) - 1, seed),
        Epochs.from_mjd(
            np.array(
                [
                    leg1.coordinates.epoch[len(leg1) - 1].as_py(),
                    leg1.coordinates.epoch[len(leg1) - 1].as_py() + 60.0,
                ]
            ),
            scale="tdb",
        ),
    ).states
    assert not leg2.wide_cross.row_is_empty(1)
    assert leg2.non_grav.non_grav_cross[1].as_py() is not None


@pytest.fixture(scope="module")
def apophis_observations():
    if not APOPHIS_MULTIAPP.exists():
        pytest.skip(f"missing fixture: {APOPHIS_MULTIAPP}")
    optical, _radar = read_ades(APOPHIS_MULTIAPP)
    return optical


@pytest.fixture(scope="module")
def amrat_fit(apophis_observations):
    """A state+AMRAT fit — a real solve whose layout is wider than 6x6.

    Its state-AMRAT column has no home but the carrier, so this is the
    fit whose joint the file format and the forward model have to carry.
    """
    seed = determine(apophis_observations).single().orbit
    primed = CartesianOrbits.from_kwargs(
        orbit_id=seed.orbit_id.to_pylist(),
        object_id=seed.object_id.to_pylist(),
        coordinates=seed.coordinates,
        srp=SRPParams.from_kwargs(amrat=[1.5e-6], cr=[1.0], amrat_variance=[(5.0e-7) ** 2]),
    )
    return refine(
        primed, apophis_observations, config=ODConfig(solve_for_flags=SolveFor(amrat="solved"))
    )


def test_a_fit_reports_its_joint_on_the_orbit(amrat_fit) -> None:
    """The fitted orbit carries the fit's own cross terms.

    Tagged by identity, so the column is readable without knowing which
    layout the fit ran — the thing a slot index could not survive.
    """
    orbit = amrat_fit.orbit
    assert not orbit.wide_cross.row_is_empty(0)
    state_cross = orbit.wide_cross.state_cross(0)
    assert set(state_cross) == {"AMRAT"}
    assert np.all(np.isfinite(state_cross["AMRAT"]))
    # No Marsden block was solved, so there is no border to carry — and
    # absence is null rather than a zero block.
    if orbit.non_grav is not None:
        assert orbit.non_grav.non_grav_cross[0].as_py() is None


def test_a_fitted_joint_survives_a_file_round_trip_into_propagation(amrat_fit, tmp_path) -> None:
    """fit -> write -> read -> propagate.

    The columns must survive the file format byte for byte (asserted
    against the values the fit itself produced), and the orbit read back
    must still reach the engine as a joint — which the propagated
    covariance differing from the joint-stripped control is what proves.
    """
    fitted = amrat_fit.orbit
    path = tmp_path / "fitted.parquet"
    fitted.to_parquet(path)
    reloaded = CartesianOrbits.from_parquet(path)

    np.testing.assert_array_equal(
        reloaded.wide_cross.state_cross(0)["AMRAT"],
        fitted.wide_cross.state_cross(0)["AMRAT"],
    )
    assert reloaded.wide_cross.param_cross(0) == fitted.wide_cross.param_cross(0)

    # A decade — the span an impact assessment actually asks for, and
    # the span over which the state-AMRAT correlation matters. Over a
    # month the two answers agree to eight digits, so a short window
    # would leave this assertion unable to see a dropped joint.
    epoch = reloaded.coordinates.epoch[0].as_py()
    epochs = Epochs.from_mjd(np.array([epoch, epoch + 3650.0]), scale="tdb")
    with_joint = propagate(reloaded, epochs).states

    # The control keeps the AMRAT prior and drops only the cross terms,
    # so what it changes is exactly one thing: the fitted state-AMRAT
    # correlation becomes an asserted zero.
    stripped = CartesianOrbits.from_kwargs(
        orbit_id=reloaded.orbit_id.to_pylist(),
        object_id=reloaded.object_id.to_pylist(),
        coordinates=reloaded.coordinates,
        srp=reloaded.srp,
    )
    without_joint = propagate(stripped, epochs).states

    with_var = _position_variance(with_joint, 1)
    without_var = _position_variance(without_joint, 1)
    assert np.isfinite(with_var) and with_var > 0.0
    # Not a directional claim: a fitted correlation can tighten or widen
    # the propagated position variance depending on its sign. What must
    # not happen is the two agreeing, which would mean the columns came
    # back from the file but never reached the engine.
    assert abs(with_var - without_var) / with_var > 1e-3, (
        f"the re-imported fit propagates to the same variance as one with its "
        f"cross terms removed ({with_var:e} vs {without_var:e}) — the joint "
        "survived the file but did not reach the engine"
    )


# ── The forward model ────────────────────────────────────────────────
#
# propagate is not the only entry point that takes an orbit. An
# ephemeris, an impact probability and a B-plane are all computed
# against whatever covariance the orbit carries, so a fitted orbit fed
# to any of them with its cross terms dropped is conditioned on a
# block-diagonal covariance no fit and no propagation ever produced —
# and the reported uncertainty comes back too small.


@pytest.fixture(scope="module")
def joint_bearing_orbit() -> CartesianOrbits:
    """One orbit carrying a real propagated joint, and nothing invented.

    Produced by propagating the wide-layout fixture and re-linking its
    last row, so the cross terms are the engine's own.
    """
    seed = _wide_layout_orbit()
    leg1 = propagate(seed, _EPOCHS).states
    return _relink(leg1, len(leg1) - 1, seed)


def _window_end(orbits: CartesianOrbits) -> Epochs:
    """A close-approach window long enough to reach Apophis's encounter."""
    return Epochs.from_mjd([orbits.coordinates.epoch[0].as_py() + 3000.0], scale="tdb")


def test_impact_probabilities_condition_on_the_joint(joint_bearing_orbit) -> None:
    """The failure this surface exists to remove, at the entry point
    where it matters most.

    Dropping the cross terms does not perturb the miss distance — the
    nominal trajectory is identical — it shrinks the sigma around it. A
    tighter sigma on an unchanged miss distance is a smaller impact
    probability, which is the direction that gets an object cleared when
    it should not be.
    """
    end = _window_end(joint_bearing_orbit)
    with_joint = compute_impact_probabilities(joint_bearing_orbit, end, ["first_order"])
    without = compute_impact_probabilities(_strip_joint(joint_bearing_orbit), end, ["first_order"])

    assert len(with_joint) > 0, "the fixture must reach a close approach"
    assert len(with_joint) == len(without)

    # Same nominal geometry: only the uncertainty is at issue.
    assert with_joint.miss_distance_km.to_pylist() == without.miss_distance_km.to_pylist()

    sigma_with = np.asarray(with_joint.sigma_distance_km.to_pylist(), dtype=float)
    sigma_without = np.asarray(without.sigma_distance_km.to_pylist(), dtype=float)
    assert np.all(np.isfinite(sigma_with))
    assert np.all(sigma_without < sigma_with), (
        f"dropping the cross terms must UNDERSTATE the close-approach sigma; "
        f"got {sigma_without} against {sigma_with}"
    )
    shortfall = (sigma_with - sigma_without) / sigma_with
    assert np.all(shortfall > 0.01), (
        f"the joint moves the reported sigma by only {shortfall}, so this "
        "fixture cannot tell a carried joint from a dropped one"
    )


def test_b_planes_condition_on_the_joint(joint_bearing_orbit) -> None:
    """The same drop, seen as a B-plane error ellipse.

    The ellipse is the covariance projected into the encounter plane, so
    a dropped joint shrinks the very object a deflection assessment
    reads off.
    """
    end = _window_end(joint_bearing_orbit)
    with_joint = compute_b_planes(joint_bearing_orbit, end, ["first_order"])
    without = compute_b_planes(_strip_joint(joint_bearing_orbit), end, ["first_order"])

    assert len(with_joint) > 0
    assert len(with_joint) == len(without)

    cov_with = np.asarray(with_joint.cov_tt_km2.to_pylist(), dtype=float)
    cov_without = np.asarray(without.cov_tt_km2.to_pylist(), dtype=float)
    assert np.all(np.isfinite(cov_with))
    assert np.all(cov_without < cov_with), (
        f"dropping the cross terms must UNDERSTATE the B-plane covariance; "
        f"got {cov_without} against {cov_with}"
    )
    assert np.all((cov_with - cov_without) / cov_with > 0.01)


def test_dropping_the_impact_marshal_hop_restores_the_understatement(
    joint_bearing_orbit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sabotage: remove the hop, and the sigma collapses to the
    joint-stripped value.

    This is what ties the number in the test above to this wiring rather
    than to anything else about the fixture.
    """
    import empyrean.impact as impact_module

    end = _window_end(joint_bearing_orbit)
    stripped_sigma = compute_impact_probabilities(
        _strip_joint(joint_bearing_orbit), end, ["first_order"]
    ).sigma_distance_km.to_pylist()

    monkeypatch.setattr(impact_module, "extract_joint", lambda orbits: {})
    sabotaged = compute_impact_probabilities(
        joint_bearing_orbit, end, ["first_order"]
    ).sigma_distance_km.to_pylist()

    assert sabotaged == stripped_sigma, (
        "with the marshal hop removed, a joint-bearing orbit must report the "
        f"same sigma as one with no joint at all; got {sabotaged} against "
        f"{stripped_sigma}"
    )


def test_the_forward_model_validates_the_joint_it_is_given(joint_bearing_orbit) -> None:
    """A malformed joint is refused at every forward-model entry point.

    The discriminating property for the ephemeris path, whose reported
    covariance does not currently move when the cross terms are dropped:
    a refusal here can only come from the engine, so it proves the
    columns cross the boundary and are validated rather than being
    accepted and ignored.
    """
    ng = joint_bearing_orbit.non_grav
    # Scaled far past what the parameter block can support, so the
    # assembled joint is no longer positive semi-definite.
    absurd = [v * 1.0e6 for v in ng.non_grav_cross[0].as_py()]
    poisoned = CartesianOrbits.from_kwargs(
        orbit_id=joint_bearing_orbit.orbit_id.to_pylist(),
        object_id=joint_bearing_orbit.object_id.to_pylist(),
        coordinates=joint_bearing_orbit.coordinates,
        non_grav=NonGravParams.from_kwargs(
            a1=ng.a1.to_pylist(),
            a2=ng.a2.to_pylist(),
            a3=ng.a3.to_pylist(),
            model=ng.model.to_pylist(),
            covariance=ng.covariance.to_pylist(),
            non_grav_cross=[absurd],
        ),
        srp=joint_bearing_orbit.srp,
        wide_cross=joint_bearing_orbit.wide_cross,
    )
    epoch = poisoned.coordinates.epoch[0].as_py()
    observers = Observers.from_code("500", Epochs.from_mjd([epoch + 100.0], scale="tdb"))
    window_end = Epochs.from_mjd([epoch + 3000.0], scale="tdb")

    for what, call in (
        ("generate_ephemeris", lambda: generate_ephemeris(poisoned, observers)),
        (
            "compute_impact_probabilities",
            lambda: compute_impact_probabilities(poisoned, window_end, ["first_order"]),
        ),
        ("compute_b_planes", lambda: compute_b_planes(poisoned, window_end, ["first_order"])),
    ):
        with pytest.raises(RuntimeError, match="positive semi-definite") as excinfo:
            call()
        assert "joint" in str(excinfo.value), f"{what}: {excinfo.value}"


def test_an_ephemeris_of_a_joint_bearing_orbit_is_finite(joint_bearing_orbit) -> None:
    """A joint-bearing orbit generates a usable ephemeris.

    Deliberately not an assertion that the sky covariance *changes*: at
    the time of writing the engine's ephemeris covariance is unchanged
    by the cross terms, and pinning that would make a future fix look
    like a regression. What must hold either way is that the extra
    columns neither crash the call nor turn a covariance into NaN.
    """
    epoch = joint_bearing_orbit.coordinates.epoch[0].as_py()
    observers = Observers.from_code(
        "500", Epochs.from_mjd([epoch + 100.0, epoch + 400.0], scale="tdb")
    )
    ephemeris = generate_ephemeris(joint_bearing_orbit, observers).ephemeris

    assert len(ephemeris) == 2
    variances = np.asarray(ephemeris.coordinates.covariance.cov_lon_lon.to_pylist(), dtype=float)
    assert np.all(np.isfinite(variances)) and np.all(variances > 0.0)


def test_a_fit_reports_the_partition_it_ran(amrat_fit) -> None:
    """`dispositions` is what the fit DID, not what was requested.

    An axis can be requested and then not opened, and the solved
    covariance cannot settle it: its slot tags record what occupied a
    column, and a considered axis occupies none. This is also the only
    Python-visible answer to "would re-attaching a prior to this axis
    double-count it".
    """
    dispositions = amrat_fit.dispositions
    assert dispositions.amrat is ParamDisposition.SOLVED
    assert dispositions.marsden is ParamDisposition.FIXED
    assert dispositions.dt is ParamDisposition.FIXED
    # Declared-indexed, so the list is positional with the orbit's
    # correction covariances rather than a count of solved burns.
    assert all(d is ParamDisposition.FIXED for d in dispositions.thrust)

    # Covariance the fit was handed and declined to use, delivered as
    # payload. Empty here because the fit used everything it was given.
    assert amrat_fit.warnings == []
