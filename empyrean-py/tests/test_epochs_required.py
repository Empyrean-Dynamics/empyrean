"""Forcing-function: public time inputs are :class:`Epochs`, never bare numbers.

A Modified Julian Date is a clock reading, not an instant. 61000.5 read
as UTC and 61000.5 read as TDB name two moments about 69 seconds apart,
and the gap grows with every leap second. So no public entry point in
this package accepts a bare list, array, float or numpy scalar where a
time is expected: it would let the call site inherit a modelling
statement (which scale?) rather than make it.

Three gates live here:

1. :func:`test_no_public_time_parameter_admits_bare_numbers` scans every
   public callable in the package and fails if a time-like parameter
   admits ``ndarray`` / ``Sequence`` / ``float``. A newly added entry
   point cannot regress the rule without failing this test.
2. The per-entry-point refusal tests assert the actual ``TypeError``,
   and that its message names ``Epochs.from_mjd`` so the fix is in the
   traceback the user sees.
3. :func:`test_declared_scale_is_honored_end_to_end` proves the scale is
   read rather than assumed: the same instant handed in as UTC and as
   TDB — two MJD numbers ~69 s apart — propagates to bit-identical
   states, which is only possible if the declared scale is applied.

Three carve-outs, all deliberate:

* Parameters whose name pins the scale (``epoch_mjd_tdb``,
  ``end_mjd_tdb``) are already unambiguous — the name is the statement —
  and stay plain floats, matching the ``empyrean-core`` signatures they
  mirror.
* :class:`Epochs`' own constructors take bare numbers, because that is
  where numbers legitimately become epochs. They are exempted here and
  covered instead by
  :func:`test_epochs_constructors_require_an_explicit_scale`.
* The ``epoch`` column on the four coordinate tables is a plain float,
  MJD TDB by definition. It is a column you read, not a parameter you
  pass, so the scan below cannot see it in the first place —
  ``from_kwargs`` is inherited from quivr and never reaches
  :func:`_time_parameters`. It is listed here so the boundary is
  greppable: converting it for reuse as a time input is
  ``Epochs.from_mjd(..., scale="tdb")`` or
  :meth:`~empyrean.Epochs.from_orbits`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from typing import Any

import empyrean
import numpy as np
import pytest
from empyrean import CartesianCoordinates, CartesianOrbits, Epochs, Observers

# Epoch of the sample orbit below (MJD TDB), and a plain heliocentric
# state to propagate. Nothing here depends on the particular orbit — the
# tests care only that the same call is made two ways.
_ORBIT_EPOCH_MJD_TDB = 61000.0


@pytest.fixture
def sample_orbit() -> CartesianOrbits:
    """A single heliocentric orbit, no covariance — enough to propagate."""
    coordinates = CartesianCoordinates.from_kwargs(
        epoch=[_ORBIT_EPOCH_MJD_TDB],
        x=[2.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.01217],
        vz=[0.0],
        frame="eclipticj2000",
        origin=["Sun"],
    )
    return CartesianOrbits.from_kwargs(orbit_id=["epochs-gate"], coordinates=coordinates)


# ── Public-surface scan ───────────────────────────────────────────────

# A parameter carries a time if it is named for one. Scale-pinned names
# (``*_mjd_tdb``) are excluded below rather than here, so that the
# exclusion is explicit and greppable.
_TIME_PARAM = re.compile(r"^(epoch|epochs|time|times)$|_(epoch|epochs|time|times)$")

# ``epoch_mjd_tdb`` / ``end_mjd_tdb`` / ``epoch_mjd_utc``: the name is
# the scale statement, so a float is honest. These mirror the
# ``empyrean-core`` signatures one-for-one.
_SCALE_PINNED = re.compile(r"_mjd_(tdb|utc)$")

# Types a time parameter must not admit. A naive ``datetime`` carries no
# scale, for the same reason a ``float`` does not. ``str`` is refused on
# different grounds: a conforming ISO timestamp's trailing ``Z`` does
# state its scale, but times cross this API as a typed table rather than
# as text, and a parameter that accepted both would have to guess which
# of the two a caller meant.
_FORBIDDEN = ("ndarray", "Sequence", "float", "list", "int", "str", "datetime")

# ``Epochs`` is where numbers legitimately become epochs.
_EXEMPT_QUALNAMES = {"Epochs", "TimeScale"}


def _iter_package_modules() -> list[Any]:
    """Import and return every module in the ``empyrean`` package."""
    modules = [empyrean]
    for info in pkgutil.walk_packages(empyrean.__path__, prefix="empyrean."):
        if "._" in info.name or info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            modules.append(importlib.import_module(info.name))
        except Exception as exc:  # noqa: BLE001 — any import failure is a real failure
            pytest.fail(f"could not import {info.name}: {exc}")
    return modules


def _owned_by_empyrean(obj: Any) -> bool:
    return str(getattr(obj, "__module__", "")).startswith("empyrean")


def _public_callables() -> list[tuple[str, Any]]:
    """Every public function / method reachable from the package surface.

    Returns ``(label, callable)`` pairs, deduplicated by label, covering
    module-level functions and the public methods of public classes.
    """
    found: dict[str, Any] = {}
    for module in _iter_package_modules():
        for name, obj in vars(module).items():
            if name.startswith("_") or not _owned_by_empyrean(obj):
                continue
            if inspect.isfunction(obj):
                found.setdefault(f"{obj.__module__}.{obj.__qualname__}", obj)
            elif inspect.isclass(obj):
                if obj.__name__ in _EXEMPT_QUALNAMES:
                    continue
                for attr, member in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    func = member.__func__ if isinstance(member, classmethod) else member
                    if inspect.isfunction(func):
                        found.setdefault(f"{obj.__module__}.{obj.__name__}.{attr}", func)
    return sorted(found.items())


def _annotation_text(annotation: Any) -> str:
    """Render an annotation as source-like text, module prefixes stripped.

    Plain classes render by ``__name__``; everything else — unions above
    all — renders by ``str()``. Reaching for ``__name__`` first would be
    a silent hole: a ``X | Y`` union reports ``__name__ == "Union"`` on
    Python 3.14, which mentions neither arm, so every forbidden type
    would slip through unseen.
    """
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        text = annotation
    elif isinstance(annotation, type):
        text = annotation.__name__
    else:
        text = str(annotation)
    return re.sub(r"\b[\w.]+\.(\w+)", r"\1", text).replace('"', "").replace("'", "")


def test_annotation_rendering_shows_union_arms() -> None:
    """Guard the guard: a union must render its arms, not collapse to a word.

    ``(Epochs | np.ndarray).__name__`` is ``"Union"`` on Python 3.14 —
    a rendering that mentions no arm at all would make every gate below
    vacuously green.
    """
    assert _annotation_text(Epochs) == "Epochs"
    assert _annotation_text(Epochs | None) == "Epochs | None"
    widened = _annotation_text(Epochs | np.ndarray)
    assert "Epochs" in widened and "ndarray" in widened, widened


def _time_parameters() -> list[tuple[str, str, str]]:
    """``(label, parameter_name, annotation_text)`` for every time input."""
    out: list[tuple[str, str, str]] = []
    for label, func in _public_callables():
        try:
            signature = inspect.signature(func)
        except (ValueError, TypeError):  # pragma: no cover - C-level callables
            continue
        for name, param in signature.parameters.items():
            if _SCALE_PINNED.search(name) or not _TIME_PARAM.search(name):
                continue
            out.append((label, name, _annotation_text(param.annotation)))
    return out


def test_public_time_surface_is_non_empty() -> None:
    """The scan finds real entry points — a silent zero-case would pass."""
    found = _time_parameters()
    assert len(found) >= 8, f"scan found only {len(found)} time parameters: {found}"
    labels = {label for label, _, _ in found}
    for expected in (
        "empyrean.propagation.propagate.propagate",
        "empyrean.states.get_states",
        "empyrean.observers.observers.Observers.from_codes",
        "empyrean.impact.compute_impact_probabilities",
    ):
        assert expected in labels, f"{expected} missing from the scanned surface"


@pytest.mark.parametrize(
    ("label", "param", "annotation"),
    _time_parameters(),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_no_public_time_parameter_admits_bare_numbers(
    label: str, param: str, annotation: str
) -> None:
    """No public time parameter admits ndarray / Sequence / float.

    Add an entry point taking ``epochs: Epochs | np.ndarray`` and this
    fails on the new parameter. That is the point.
    """
    assert annotation, f"{label}({param}=...) carries no annotation"
    admitted = [t for t in _FORBIDDEN if re.search(rf"\b{t}\b", annotation)]
    assert not admitted, (
        f"{label}({param}: {annotation}) admits {', '.join(admitted)}. "
        f"A time input must be an Epochs table — a bare value carries no "
        f"time scale. Change the annotation to `Epochs` and coerce with "
        f"`_require_epochs` / `_epochs_mjd_tdb` / `_require_single_epoch`."
    )
    assert re.fullmatch(r"Epochs(\s*\|\s*None)?", annotation.strip()), (
        f"{label}({param}: {annotation}) should be annotated `Epochs`"
    )


def _epochs_unions() -> list[tuple[str, str, str]]:
    """Public parameters whose annotation mentions ``Epochs`` in a union.

    Complements the name-based scan above: a future parameter called
    something the ``_TIME_PARAM`` rule does not recognise (``mjds``,
    ``epoch_list``) still gets caught here the moment its annotation
    widens ``Epochs`` with a bare-number arm.
    """
    out: list[tuple[str, str, str]] = []
    for label, func in _public_callables():
        try:
            signature = inspect.signature(func)
        except (ValueError, TypeError):  # pragma: no cover - C-level callables
            continue
        for name, param in signature.parameters.items():
            text = _annotation_text(param.annotation)
            if "Epochs" in text and "|" in text:
                out.append((label, name, text))
    return out


@pytest.mark.parametrize(
    ("label", "param", "annotation"),
    _epochs_unions() or [("<none>", "<none>", "Epochs")],
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_epochs_is_never_widened_with_a_bare_number_arm(
    label: str, param: str, annotation: str
) -> None:
    """``Epochs | None`` is fine; ``Epochs | np.ndarray`` is the regression."""
    admitted = [t for t in _FORBIDDEN if re.search(rf"\b{t}\b", annotation)]
    assert not admitted, (
        f"{label}({param}: {annotation}) widens Epochs with {', '.join(admitted)}. "
        f"Drop the bare-number arm and refuse it instead."
    )


# ── Epochs' own constructors state their scale ────────────────────────


@pytest.mark.parametrize("name", ["from_mjd", "from_jd", "linspace", "arange"])
def test_epochs_constructors_require_an_explicit_scale(name: str) -> None:
    """``scale`` has no default on the constructors that take raw numbers."""
    scale = inspect.signature(getattr(Epochs, name)).parameters["scale"]
    assert scale.default is inspect.Parameter.empty, (
        f"Epochs.{name}() defaults `scale` to {scale.default!r}; users must "
        f"state the scale their numbers are in."
    )


@pytest.mark.parametrize("name", ["from_mjd", "from_jd"])
def test_epochs_constructors_refuse_a_missing_scale(name: str) -> None:
    """Omitting ``scale`` is a ``TypeError``, not a silent TDB assumption."""
    with pytest.raises(TypeError, match="scale"):
        getattr(Epochs, name)([60500.0])


def test_epochs_from_iso_keeps_its_output_scale_default() -> None:
    """ISO input states its own scale, so only the *output* scale defaults.

    ``from_iso`` requires the trailing ``Z``, so the input is UTC by
    format — nothing is left unstated. Its ``scale`` argument selects
    what comes back, and defaulting that to UTC changes no instant.
    """
    assert inspect.signature(Epochs.from_iso).parameters["scale"].default is not (
        inspect.Parameter.empty
    )
    default = Epochs.from_iso(["2026-01-01T12:00:00.000Z"])
    assert default.scale == "utc"
    assert (
        default.to_numpy()[0]
        == Epochs.from_iso(["2026-01-01T12:00:00.000Z"], scale="utc").to_numpy()[0]
    )


def test_epochs_now_keeps_its_utc_default() -> None:
    """``now()`` names its own clock, so the default stays."""
    assert inspect.signature(Epochs.now).parameters["scale"].default is not (
        inspect.Parameter.empty
    )
    assert Epochs.now().scale == "utc"


# ── Per-entry-point refusals ──────────────────────────────────────────

_BARE_TIMES = [
    pytest.param([61000.5, 61010.5], id="list"),
    pytest.param(np.array([61000.5, 61010.5]), id="ndarray"),
    pytest.param(61000.5, id="float"),
    pytest.param(np.float64(61000.5), id="numpy-scalar"),
]


def _assert_teaches_the_fix(message: str) -> None:
    """The refusal must name the type, the fix, and why it matters."""
    assert "Epochs" in message
    assert "Epochs.from_mjd" in message or "Epochs.from_iso" in message
    assert "no time scale" in message
    assert "69 seconds" in message


@pytest.mark.parametrize("bare", _BARE_TIMES)
def test_observers_from_code_refuses_bare_times(bare: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        Observers.from_code("689", bare)
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("bare", _BARE_TIMES)
def test_observers_from_codes_refuses_bare_times(bare: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        Observers.from_codes(["689"], bare)
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("bare", _BARE_TIMES)
def test_get_observer_states_refuses_bare_times(bare: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        empyrean.get_observer_states(["689"], bare)
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("bare", _BARE_TIMES)
def test_get_states_refuses_bare_times(bare: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        empyrean.get_states("Earth", "Sun", bare)
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("bare", _BARE_TIMES)
def test_propagate_refuses_bare_times(sample_orbit: Any, bare: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        empyrean.propagate(sample_orbit, bare)
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("bare", _BARE_TIMES)
def test_query_horizons_refuses_bare_times(bare: Any) -> None:
    """Refused at the entry point — before any network call is made."""
    with pytest.raises(TypeError) as excinfo:
        empyrean.query_horizons(["99942"], "500", bare)
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("bare", [61000.5, np.float64(61000.5), [61000.5]])
def test_compute_impact_probabilities_refuses_a_bare_end_epoch(
    sample_orbit: Any, bare: Any
) -> None:
    with pytest.raises(TypeError) as excinfo:
        empyrean.compute_impact_probabilities(sample_orbit, bare, ["first_order"])
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("bare", [61000.5, np.float64(61000.5), [61000.5]])
def test_compute_b_planes_refuses_a_bare_end_epoch(sample_orbit: Any, bare: Any) -> None:
    with pytest.raises(TypeError) as excinfo:
        empyrean.compute_b_planes(sample_orbit, bare, ["first_order"])
    _assert_teaches_the_fix(str(excinfo.value))


def test_end_epoch_refuses_a_multi_row_epochs(sample_orbit: Any) -> None:
    """A window has one end; two candidates is a question, not an answer."""
    with pytest.raises(ValueError, match="exactly one epoch, got 2"):
        empyrean.compute_impact_probabilities(
            sample_orbit,
            Epochs.from_mjd([61000.5, 61010.5], scale="tdb"),
            ["first_order"],
        )


@pytest.mark.parametrize("bare", _BARE_TIMES)
def test_built_system_propagate_refuses_bare_times(sample_orbit: Any, bare: Any) -> None:
    """The handle refuses under its own name, not its delegate's."""
    system = empyrean.build_system(empyrean.ForceModelTier.STANDARD, empyrean.Frame.ECLIPTICJ2000)
    with pytest.raises(TypeError) as excinfo:
        system.propagate(sample_orbit, bare)
    assert "BuiltSystem.propagate()" in str(excinfo.value)
    _assert_teaches_the_fix(str(excinfo.value))


@pytest.mark.parametrize("method", ["index_at", "up_to"])
@pytest.mark.parametrize("bare", [61010.0, np.float64(61010.0), [61010.0]])
def test_sensitivity_epoch_lookup_refuses_bare_times(
    sample_orbit: Any, method: str, bare: Any
) -> None:
    result = empyrean.propagate(
        sample_orbit,
        Epochs.from_mjd([61010.0], scale="tdb"),
        config=empyrean.PropagationConfig(compute_stm=True),
    )
    chain = result.sensitivity.select("orbit_id", "epochs-gate")
    with pytest.raises(TypeError) as excinfo:
        getattr(chain, method)(bare)
    _assert_teaches_the_fix(str(excinfo.value))


def test_refusal_names_the_entry_point_the_caller_used() -> None:
    """A delegating wrapper names itself, so the traceback is actionable."""
    for call, expected in (
        (lambda: Observers.from_code("689", [61000.5]), "Observers.from_code()"),
        (lambda: Observers.from_codes(["689"], [61000.5]), "Observers.from_codes()"),
        (
            lambda: empyrean.get_observer_states(["689"], [61000.5]),
            "get_observer_states()",
        ),
    ):
        with pytest.raises(TypeError) as excinfo:
            call()
        assert expected in str(excinfo.value), str(excinfo.value)


def test_iso_string_refusal_points_at_from_iso() -> None:
    """A string gets the ISO constructor, not the MJD one.

    And it is not told its timestamp carries no scale: the trailing
    ``Z`` states the scale, so the reason a string is refused is the
    type. Telling the caller otherwise would push them back into the
    hand-conversion this surface exists to remove.
    """
    with pytest.raises(TypeError) as excinfo:
        Observers.from_code("689", "2026-01-01T00:00:00.000Z")
    message = str(excinfo.value)
    assert "Epochs.from_iso" in message
    assert "no time scale" not in message
    assert "'Z'" in message


def test_refusal_is_not_a_deprecation_warning(
    sample_orbit: Any, recwarn: pytest.WarningsRecorder
) -> None:
    """The refusal raises. It never warns and proceeds on a TDB guess."""
    with pytest.raises(TypeError):
        empyrean.propagate(sample_orbit, [61000.5])
    deprecations = [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, f"bare times deprecated rather than refused: {deprecations}"


# ── Scale honesty, end to end ─────────────────────────────────────────

# 2026-01-01 12:00 UTC — comfortably inside the leap-second era, where
# TDB - UTC is ~69.18 s and the two clock readings are far enough apart
# to be unmistakable at double precision.
_HONESTY_EPOCH_UTC_ISO = "2026-01-01T12:00:00.000Z"


def test_the_two_scales_are_far_apart_at_this_epoch() -> None:
    """The premise: the same instant has two very different MJD values."""
    utc = Epochs.from_iso([_HONESTY_EPOCH_UTC_ISO], scale="utc")
    tdb = utc.to_tdb()
    offset_seconds = (tdb.to_numpy()[0] - utc.to_numpy()[0]) * 86400.0
    assert 69.0 < offset_seconds < 70.0, offset_seconds


def test_declared_scale_is_honored_end_to_end(sample_orbit: Any) -> None:
    """The same instant, declared two ways, propagates to the same state.

    ``rebuilt_utc`` and ``as_tdb`` hold *different numbers* — 61041.5
    UTC vs 61041.500800740 TDB, 69.18 s apart — that name the *same
    instant*. Each is reconstructed from its raw MJD rather than reused
    from the converted table, so the scale attribute is the only thing
    distinguishing them. If ``propagate`` ignored it and read both as
    TDB, the two runs would start 69 s apart and their x-components would
    land ~363 km apart — one component of a ~1460 km along-track
    displacement at this fixture's 21.1 km/s (see
    :func:`test_a_scale_lie_moves_the_state`). Instead they agree
    **bit for bit**: the declared UTC is converted to the identical
    float64 TDB, so the integrator sees one and the same input.
    """
    as_utc = Epochs.from_iso([_HONESTY_EPOCH_UTC_ISO], scale="utc")
    as_tdb = Epochs.from_mjd(as_utc.to_tdb().to_numpy(), scale="tdb")
    rebuilt_utc = Epochs.from_mjd(as_utc.to_numpy(), scale="utc")

    assert rebuilt_utc.to_numpy()[0] != as_tdb.to_numpy()[0]

    from_utc = empyrean.propagate(sample_orbit, rebuilt_utc)
    from_tdb = empyrean.propagate(sample_orbit, as_tdb)

    for axis in ("x", "y", "z", "vx", "vy", "vz"):
        left = getattr(from_utc.states.coordinates, axis).to_numpy(zero_copy_only=False)
        right = getattr(from_tdb.states.coordinates, axis).to_numpy(zero_copy_only=False)
        np.testing.assert_array_equal(
            left,
            right,
            err_msg=f"{axis} differs between the UTC- and TDB-declared forms of one instant",
        )


def test_a_scale_lie_moves_the_state(sample_orbit: Any) -> None:
    """The counter-check: mislabelling the scale changes the answer.

    Same number, two scale labels. If the entry point were ignoring the
    declared scale, these would agree — the fact that they differ is
    what makes the test above meaningful rather than vacuous.
    """
    mjd = Epochs.from_iso([_HONESTY_EPOCH_UTC_ISO], scale="utc").to_numpy()

    labelled_utc = empyrean.propagate(sample_orbit, Epochs.from_mjd(mjd, scale="utc"))
    labelled_tdb = empyrean.propagate(sample_orbit, Epochs.from_mjd(mjd, scale="tdb"))

    x_utc = labelled_utc.states.coordinates.x.to_numpy(zero_copy_only=False)[0]
    x_tdb = labelled_tdb.states.coordinates.x.to_numpy(zero_copy_only=False)[0]
    separation_km = abs(x_utc - x_tdb) * 1.495978707e8
    assert separation_km > 1.0, (
        f"the two scale labels landed {separation_km:.3f} km apart; the "
        f"declared scale is being ignored somewhere on the path"
    )
