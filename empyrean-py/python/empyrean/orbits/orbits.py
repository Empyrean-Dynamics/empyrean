"""Orbit types: coordinates + orbit_id + object_id + optional non-grav and photometric params."""

from collections.abc import Iterable
from typing import Protocol, TypeVar

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from empyrean.coordinates.coordinates import (
    CartesianCoordinates,
    CometaryCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
)
from empyrean.orbits.nongrav import NonGravParams
from empyrean.orbits.photometry import PhotometricParams
from empyrean.orbits.srp import SRPParams

T = TypeVar("T", bound=qv.Table)


class _IsInFn(Protocol):
    """Static signature for ``pyarrow.compute.is_in``.

    pyarrow generates its compute functions from the C++ function registry
    at import time, so ``pc.is_in`` has no statically declarable attribute
    (pyarrow ships ``py.typed`` but no stub for these dynamic functions).
    This Protocol pins the real signature; ``_is_in`` below binds the
    runtime-injected function object (read from the module dict) to it.
    """

    def __call__(
        self,
        values: pa.Array | pa.ChunkedArray,
        /,
        *,
        value_set: pa.Array,
    ) -> pa.BooleanArray: ...


_is_in: _IsInFn = pc.__dict__["is_in"]


def _select_by_string_column(table: T, column_name: str, values: Iterable[str]) -> T:
    """Filter a quivr Table to rows where ``column_name`` is in ``values``.

    Order of returned rows matches the input table, NOT the order of
    ``values``. Empty ``values`` returns an empty table of the same
    type. Missing IDs are silently skipped — caller validates if
    membership is required.
    """
    wanted = list(values)
    if not wanted:
        return table.empty()
    column = table.column(column_name)
    mask = _is_in(column, value_set=pa.array(wanted))
    return table.apply_mask(mask)


class CartesianOrbits(qv.Table):
    """Orbits in Cartesian coordinates."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    non_grav = NonGravParams.as_column(nullable=True)
    srp = SRPParams.as_column(nullable=True)
    photometric = PhotometricParams.as_column(nullable=True)

    def select_by_orbit_id(self, orbit_ids: Iterable[str]) -> "CartesianOrbits":
        """Return rows whose ``orbit_id`` is in ``orbit_ids``."""
        return _select_by_string_column(self, "orbit_id", orbit_ids)

    def select_by_object_id(self, object_ids: Iterable[str]) -> "CartesianOrbits":
        """Return rows whose ``object_id`` is in ``object_ids``."""
        return _select_by_string_column(self, "object_id", object_ids)


class KeplerianOrbits(qv.Table):
    """Orbits in Keplerian elements."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = KeplerianCoordinates.as_column()
    non_grav = NonGravParams.as_column(nullable=True)
    srp = SRPParams.as_column(nullable=True)
    photometric = PhotometricParams.as_column(nullable=True)

    def select_by_orbit_id(self, orbit_ids: Iterable[str]) -> "KeplerianOrbits":
        """Return rows whose ``orbit_id`` is in ``orbit_ids``."""
        return _select_by_string_column(self, "orbit_id", orbit_ids)

    def select_by_object_id(self, object_ids: Iterable[str]) -> "KeplerianOrbits":
        """Return rows whose ``object_id`` is in ``object_ids``."""
        return _select_by_string_column(self, "object_id", object_ids)


class CometaryOrbits(qv.Table):
    """Orbits in cometary elements."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = CometaryCoordinates.as_column()
    non_grav = NonGravParams.as_column(nullable=True)
    srp = SRPParams.as_column(nullable=True)
    photometric = PhotometricParams.as_column(nullable=True)

    def select_by_orbit_id(self, orbit_ids: Iterable[str]) -> "CometaryOrbits":
        """Return rows whose ``orbit_id`` is in ``orbit_ids``."""
        return _select_by_string_column(self, "orbit_id", orbit_ids)

    def select_by_object_id(self, object_ids: Iterable[str]) -> "CometaryOrbits":
        """Return rows whose ``object_id`` is in ``object_ids``."""
        return _select_by_string_column(self, "object_id", object_ids)


class SphericalOrbits(qv.Table):
    """Orbits in spherical coordinates."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    coordinates = SphericalCoordinates.as_column()
    non_grav = NonGravParams.as_column(nullable=True)
    srp = SRPParams.as_column(nullable=True)
    photometric = PhotometricParams.as_column(nullable=True)

    def select_by_orbit_id(self, orbit_ids: Iterable[str]) -> "SphericalOrbits":
        """Return rows whose ``orbit_id`` is in ``orbit_ids``."""
        return _select_by_string_column(self, "orbit_id", orbit_ids)

    def select_by_object_id(self, object_ids: Iterable[str]) -> "SphericalOrbits":
        """Return rows whose ``object_id`` is in ``object_ids``."""
        return _select_by_string_column(self, "object_id", object_ids)
