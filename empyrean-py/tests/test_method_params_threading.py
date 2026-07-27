"""Per-method uncertainty parameters reach the engine on the impact path —
bd empyrean-zxtd.

``compute_impact_probabilities`` / ``compute_b_planes`` used to collapse each
method spec to a bare integer tag, and the Rust binding rebuilt every method
from engine defaults. A ``SigmaPoint(n_sigma=2.0)`` /
``MonteCarlo(n_samples=100_000, seed=...)`` / ``GaussianMixture(threshold=0.5,
...)`` therefore ran with default parameters, silently. Only *non-default*
specs were affected: a default-constructed dataclass lowers to the same seven
values the engine already used, which is what the invariant test at the bottom
of this file pins.

Every test here is marked PROVING (fails on the pre-fix code) or REGRESSION
GUARD (passes both before and after; it exists to keep a property from
regressing later) in its own docstring.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean.coordinates.coordinates import CartesianCoordinates, CometaryCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.coordinates.enums import Origin
from empyrean.impact import (
    _flatten_method_specs,
    compute_b_planes,
    compute_impact_probabilities,
)
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.config import (
    _UNCERTAINTY_PARAM_DEFAULTS,
    GaussianMixture,
    MonteCarlo,
    SigmaPoint,
    UncertaintyMethod,
    _uncertainty_method_params,
)

# 2008 TC3 on its final approach to Earth — the same offline cometary
# elements ``test_no_silent_drops`` uses. It is the cheapest fixture that
# actually produces impact rows (one-day window, no network), which the
# Monte-Carlo and Gaussian-mixture assertions below need: a method's
# parameters are only observable on a row it produced.
_TC3_COMETARY = {
    "epoch": 54746.0,
    "q": 0.8999568608039946,
    "e": 0.3120674404369712,
    "i": 2.542215283712214,
    "raan": 194.1011435928938,
    "ap": 234.44892519,
    "tp": 2_454_790.898_848_103_4 - 2_400_000.5,
}
_TC3_END_MJD_TDB = 54747.0


def _heliocentric_orbit() -> CartesianOrbits:
    """Near-1 AU, near-circular, no covariance, no encounter.

    Used only by the SigmaPoint-rejection tests: the engine rejects the
    non-default sigma-point knobs before any integration runs, so this
    fixture costs nothing and needs no close approach.
    """
    return CartesianOrbits.from_kwargs(
        orbit_id=["helio"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=[60000.0],
            x=[1.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.017],
            vz=[0.0],
            frame="ecliptic_j2000",
            origin=[str(Origin.SUN)],
        ),
    )


def _impactor_orbit() -> CartesianOrbits:
    """2008 TC3 with a small diagonal covariance, as Cartesian coordinates."""
    c = _TC3_COMETARY
    com = CometaryCoordinates.from_kwargs(
        epoch=[c["epoch"]],
        q=[c["q"]],
        e=[c["e"]],
        i=[c["i"]],
        raan=[c["raan"]],
        ap=[c["ap"]],
        tp=[c["tp"]],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    cart = empyrean.transform_coordinates(com, CartesianCoordinates)
    cov = np.diag([1e-12, 1e-12, 1e-12, 1e-16, 1e-16, 1e-16])[None, :, :]
    cart = cart.set_column("covariance", CartesianCovariance.from_matrix(cov))
    return CartesianOrbits.from_kwargs(
        orbit_id=["TC3"],
        object_id=["2008 TC3"],
        coordinates=cart,
    )


@pytest.fixture(scope="module")
def impactor() -> CartesianOrbits:
    empyrean.initialize()
    return _impactor_orbit()


# ══════════════════════════════════════════════════════════════════
#  SigmaPoint — threading surfaces the engine's loud rejection
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "compute",
    [compute_impact_probabilities, compute_b_planes],
    ids=["impact_probabilities", "b_planes"],
)
def test_non_default_sigma_point_is_rejected(compute) -> None:
    """PROVING.

    The canonical 2N+1 unscented set is parameter-free, so villeneuve rejects
    a non-default ``n_sigma`` loudly. Before the fix the impact path discarded
    ``n_sigma=2.0`` and handed the engine ``SigmaPoint { n_sigma: 1.0 }``, so
    the call succeeded and the user's request was silently reinterpreted —
    verified against the binding: tag 2 with the default columns returns rows
    without raising. Threading the parameter is what lets the rejection reach
    the caller.

    Runs on a covariance-free heliocentric fixture on purpose: the rejection
    fires during force-model setup, before any integration.
    """
    empyrean.initialize()
    with pytest.raises(Exception, match=r"[Ss]igma"):
        compute(
            _heliocentric_orbit(),
            end_epoch=60010.0,
            methods=[SigmaPoint(n_sigma=2.0)],
            body_filter=[Origin.EARTH],
        )


# ══════════════════════════════════════════════════════════════════
#  MonteCarlo — n_samples reaches the sampler
# ══════════════════════════════════════════════════════════════════


def test_monte_carlo_n_samples_reaches_engine(impactor: CartesianOrbits) -> None:
    """PROVING.

    ``mc_n_samples`` is the engine's own report of how many virtual asteroids
    it drew, so it is the one valid oracle for the requested sample count.
    Before the fix the binding built ``monte_carlo(1000)`` regardless, so this
    column read 1000 rather than 16.

    Deliberately no arithmetic tie between ``ip_mc``, ``mc_n_impacts`` and
    ``mc_n_samples``: ``ip_mc`` is normalised by the number of samples the
    engine actually propagated for that body, not by ``mc_n_samples``.
    """
    ips = compute_impact_probabilities(
        impactor,
        end_epoch=_TC3_END_MJD_TDB,
        methods=[MonteCarlo(n_samples=16, seed=11)],
        body_filter=[Origin.EARTH],
    )
    # Guard first: Monte-Carlo emits at most ONE row per body, while the
    # analytic methods emit one per close approach. An empty table would make
    # the set comparison below vacuously true.
    assert len(ips) > 0, "impactor fixture produced no Monte-Carlo impact row"
    assert set(ips.method.to_pylist()) == {"monte_carlo"}
    assert set(ips.mc_n_samples.to_pylist()) == {16}


# ══════════════════════════════════════════════════════════════════
#  GaussianMixture — splitter knobs reach the AGM
# ══════════════════════════════════════════════════════════════════


def test_gaussian_mixture_params_reach_engine(impactor: CartesianOrbits) -> None:
    """PROVING.

    ``threshold=0.0`` forces the splitter to fire and ``components_per_split=5``
    fixes the component count, so ``agm_components`` reads back 5 and ``ip_agm``
    is finite. Before the fix the binding built the mixture with engine defaults
    (``threshold=1.0``), the splitter never fired on this fixture, and both
    columns came back null — verified against the binding: tag 5 with the
    default columns returns ``agm_components = 0`` / ``ip_agm = NaN``, which the
    wrapper nulls.

    Both columns are nullable (a row below threshold reports null), so filter
    None out before comparing rather than asserting on the raw column.
    """
    ips = compute_impact_probabilities(
        impactor,
        end_epoch=_TC3_END_MJD_TDB,
        methods=[GaussianMixture(threshold=0.0, max_depth=1, components_per_split=5)],
        body_filter=[Origin.EARTH],
    )
    assert len(ips) > 0, "impactor fixture produced no Gaussian-mixture impact row"
    assert set(ips.method.to_pylist()) == {"gaussian_mixture"}

    components = [c for c in ips.agm_components.to_pylist() if c is not None]
    assert components, "AGM splitter did not fire — components_per_split never reached it"
    assert set(components) == {5}

    ip_agm = [p for p in ips.ip_agm.to_pylist() if p is not None]
    assert ip_agm, "AGM fired but reported no mixture-corrected impact probability"
    assert all(np.isfinite(p) for p in ip_agm)


# ══════════════════════════════════════════════════════════════════
#  Column alignment
# ══════════════════════════════════════════════════════════════════


def test_method_column_length_mismatch_raises() -> None:
    """PROVING (the validated channel did not exist before the fix).

    The seven parameter columns are consumed by index, never zipped. A short
    column must be a typed error naming the column and both lengths — a bare
    ``zip`` would truncate the method list and silently drop a requested
    method.
    """
    empyrean.initialize()
    from empyrean._empyrean_rs import _compute_impact_probabilities
    from empyrean.impact import _common_orbit_args

    args = _common_orbit_args(_heliocentric_orbit())
    tags, params = _flatten_method_specs(
        [UncertaintyMethod.FIRST_ORDER, UncertaintyMethod.SECOND_ORDER]
    )
    params["method_mc_n_samples"] = params["method_mc_n_samples"][:1]

    with pytest.raises(ValueError, match=r"method_mc_n_samples.*1.*2"):
        _compute_impact_probabilities(
            epochs=args["epochs"],
            elements=args["elements"],
            covariances=args["covariances"],
            has_covariance=args["has_covariance"],
            representations=args["representations"],
            frames=args["frames"],
            origins=args["origins"],
            end_mjd_tdb=60010.0,
            a1s=args["a1s"],
            a2s=args["a2s"],
            a3s=args["a3s"],
            method_tags=tags,
            **params,
        )


# ══════════════════════════════════════════════════════════════════
#  Behaviour preservation
# ══════════════════════════════════════════════════════════════════


def test_default_specs_lower_to_engine_defaults() -> None:
    """REGRESSION GUARD.

    A default-constructed dataclass must lower to exactly the engine defaults,
    which is what makes threading the parameters a no-op for every existing
    caller: ``SigmaPoint()`` and ``"sigma_point"`` produce identical columns.
    If a dataclass default ever drifts from
    ``_UNCERTAINTY_PARAM_DEFAULTS``, that equivalence silently breaks and this
    is the test that says so.
    """
    assert _uncertainty_method_params(SigmaPoint()) == _UNCERTAINTY_PARAM_DEFAULTS
    assert _uncertainty_method_params(MonteCarlo()) == _UNCERTAINTY_PARAM_DEFAULTS
    assert _uncertainty_method_params(GaussianMixture()) == _UNCERTAINTY_PARAM_DEFAULTS
    # Non-parameterized spec forms carry nothing of their own.
    assert _uncertainty_method_params(UncertaintyMethod.SIGMA_POINT) == _UNCERTAINTY_PARAM_DEFAULTS
    assert _uncertainty_method_params("monte_carlo") == _UNCERTAINTY_PARAM_DEFAULTS
    assert _uncertainty_method_params(5) == _UNCERTAINTY_PARAM_DEFAULTS

    # ... and the flattened columns agree, entry by entry.
    tags_dc, params_dc = _flatten_method_specs([SigmaPoint(), MonteCarlo(), GaussianMixture()])
    tags_enum, params_enum = _flatten_method_specs(
        [
            UncertaintyMethod.SIGMA_POINT,
            UncertaintyMethod.MONTE_CARLO,
            UncertaintyMethod.GAUSSIAN_MIXTURE,
        ]
    )
    assert tags_dc == tags_enum == [2, 3, 5]
    assert params_dc == params_enum
