"""Packed-joint wire refusals.

The engine refuses a covariance push it cannot honour — a malformed axis
layout, a triangle of the wrong length, a retired discriminant, a
covariance handed to the state-level transform — rather than accepting it
and quietly producing a wrong answer. Each refusal reaches Python as one
:class:`PushError` carrying the :class:`PushErrorVariant` that names the
cause and the engine's own message.

The extension raises its own low-level ``PushError`` (a ``ValueError``
subclass whose ``args`` are ``(variant_name, message)``);
:func:`translate_push_errors` wraps that into this richer class at every
boundary that can raise it, so a caller classifies a failure the same way
wherever it was raised. This is the single translation point.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["PushError", "PushErrorVariant", "translate_push_errors"]


class PushErrorVariant(str, enum.Enum):
    """The cause of a packed-joint wire refusal.

    Subclasses ``str`` so the variant serializes directly. The members
    mirror the wrapper's ``empyrean::PushError`` variants by name, one per
    failure axis — the engine never collapses two distinct refusals onto
    one tag.
    """

    LAYOUT_INVALID = "LayoutInvalid"
    """The axis-layout bitsets are inconsistent or out of range."""
    TRIANGLE_LENGTH = "TriangleLength"
    """The packed triangle length does not match the declared width."""
    WIDTH_MISMATCH = "WidthMismatch"
    """The joint width disagrees with the state/parameter layout."""
    UNKNOWN_FUNCTIONAL = "UnknownFunctional"
    """The functional tag is not a known code."""
    UNKNOWN_PROVENANCE = "UnknownProvenance"
    """The provenance tag is not a known code."""
    RETIRED_DISCRIMINANT = "RetiredDiscriminant"
    """A discriminant retired with the engine (e.g. third-order kind)."""
    EPOCH_MISMATCH = "EpochMismatch"
    """The joint epoch disagrees with the state it is pushed onto."""
    ENGINE_REFUSED = "EngineRefused"
    """The engine rejected the joint on a definiteness/consistency check."""
    UNSUPPORTED_INPUT_KIND = "UnsupportedInputKind"
    """The covariance kind is not accepted at this input."""
    UNKNOWN_ASSERTED_SOURCE = "UnknownAssertedSource"
    """An asserted joint named an unknown source."""
    TRANSFORM_COVARIANCE_UNSUPPORTED = "TransformCovarianceUnsupported"
    """A covariance-bearing state was handed to the state-level transform;
    use :func:`empyrean.transform_orbits` to move a covariance between
    bases."""


_VARIANT_BY_NAME = {v.value: v for v in PushErrorVariant}


class PushError(ValueError):
    """A packed-joint wire refusal, tagged with its cause.

    Subclasses :class:`ValueError` so existing ``except ValueError``
    handlers still catch it. The :attr:`variant` is the stable,
    machine-readable cause; :attr:`message` is the engine's own text.
    """

    def __init__(self, variant: PushErrorVariant, message: str) -> None:
        self.variant = variant
        self.message = message
        super().__init__(f"{variant.value}: {message}")


@contextmanager
def translate_push_errors() -> Iterator[None]:
    """Translate the extension's low-level ``PushError`` into :class:`PushError`.

    The one place the extension's ``(variant_name, message)`` args are
    mapped onto :class:`PushErrorVariant`. An unrecognized variant name is
    itself a skew between the extension and this package and is raised
    loudly rather than coerced.
    """
    from empyrean._empyrean_rs import PushError as _RsPushError

    try:
        yield
    except _RsPushError as exc:
        variant_name, message = exc.args
        try:
            variant = _VARIANT_BY_NAME[variant_name]
        except KeyError:
            raise RuntimeError(
                f"the compiled empyrean extension raised an unknown push-error "
                f"variant {variant_name!r}; it disagrees with this Python package "
                "about the refusal set — rebuild the wheel"
            ) from exc
        raise PushError(variant, message) from None
