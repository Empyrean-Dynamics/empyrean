"""Covariance matrix types for each coordinate representation.

Each covariance class is a quivr Table with 21 Float64 columns
(lower-triangular elements of a 6x6 symmetric matrix) named
``cov_{label_i}_{label_j}``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

import numpy as np
import pyarrow as pa
import quivr as qv

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


class _CovarianceTable(Protocol):
    """Structural surface of a dynamically-built covariance table.

    Covariance classes are built dynamically by :func:`_make_covariance_class`,
    which injects ``_cov_names`` / ``_state_labels`` metadata and the
    ``from_matrix`` / ``from_kwargs`` constructors alongside the standard
    :class:`quivr.Table` API. This protocol captures exactly the members the
    helper functions touch, on both the instance side (``self``) and the class
    object side (``cls``, typed as ``type[_CovarianceTable]``).
    """

    _cov_names: ClassVar[list[str]]
    _state_labels: ClassVar[list[str]]

    def __len__(self) -> int: ...

    def column(self, column_name: str) -> pa.ChunkedArray: ...

    @classmethod
    def from_kwargs(cls, **kwargs: object) -> _CovarianceTable: ...

    @classmethod
    def from_matrix(cls, matrix: FloatArray) -> _CovarianceTable: ...


def _lower_tri_indices(dim: int) -> list[tuple[int, int]]:
    """Return (row, col) indices for lower-triangular traversal."""
    indices: list[tuple[int, int]] = []
    for i in range(dim):
        for j in range(i + 1):
            indices.append((i, j))
    return indices


def _cov_column_names(state_labels: list[str]) -> list[str]:
    """Generate covariance column names in lower-tri order."""
    names: list[str] = []
    for i in range(len(state_labels)):
        for j in range(i + 1):
            names.append(f"cov_{state_labels[j]}_{state_labels[i]}")
    return names


def _cov_from_matrix(cls: type[_CovarianceTable], matrix: FloatArray) -> _CovarianceTable:
    """Create from covariance matrices.

    Parameters
    ----------
    matrix : np.ndarray
        Covariance matrices with shape (N, 6, 6) or (6, 6).
    """
    if matrix.ndim == 2:
        matrix = matrix[np.newaxis, :, :]

    _n, dim, _ = matrix.shape
    if dim != 6:
        raise ValueError(f"Expected 6x6 matrices, got {dim}x{dim}")

    kwargs: dict[str, list[float]] = {}
    for name, (i, j) in zip(cls._cov_names, _lower_tri_indices(6), strict=False):
        kwargs[name] = matrix[:, i, j].tolist()

    return cls.from_kwargs(**kwargs)


def _cov_to_matrix(self: _CovarianceTable) -> FloatArray:
    """Return (N, 6, 6) numpy array."""
    n = len(self)
    mat: FloatArray = np.full((n, 6, 6), np.nan)

    for name, (i, j) in zip(self._cov_names, _lower_tri_indices(6), strict=False):
        col = self.column(name)
        vals = col.to_numpy(zero_copy_only=False)
        mat[:, i, j] = vals
        if i != j:
            mat[:, j, i] = vals

    return mat


def _cov_from_sigmas(cls: type[_CovarianceTable], sigmas: FloatArray) -> _CovarianceTable:
    """Create diagonal-only covariances from sigma values.

    Parameters
    ----------
    sigmas : np.ndarray
        Standard deviations with shape (N, 6) or (6,).
    """
    if sigmas.ndim == 1:
        sigmas = sigmas[np.newaxis, :]

    n, dim = sigmas.shape
    if dim != 6:
        raise ValueError(f"Expected 6 sigmas per row, got {dim}")

    mat: FloatArray = np.zeros((n, 6, 6))
    for k in range(6):
        mat[:, k, k] = sigmas[:, k] ** 2

    return cls.from_matrix(mat)


def _cov_sigmas(self: _CovarianceTable) -> FloatArray:
    """Return (N, 6) array of 1-sigma uncertainties (sqrt of diagonal)."""
    labels = self._state_labels
    n = len(self)
    result: FloatArray = np.full((n, 6), np.nan)
    for k, label in enumerate(labels):
        name = f"cov_{label}_{label}"
        col = self.column(name)
        vals = col.to_numpy(zero_copy_only=False)
        result[:, k] = np.sqrt(np.abs(vals))
    return result


def _make_covariance_class(
    class_name: str,
    coord_labels: list[str],
    docstring: str,
) -> type:
    """Create a covariance qv.Table subclass for a specific coordinate type.

    Each class has 21 nullable Float64 columns (lower-tri of 6x6)
    with names matching the coordinate labels.
    """
    cov_names = _cov_column_names(coord_labels)

    # Build the class namespace with 21 Float64 columns. The values are
    # genuinely heterogeneous (columns, metadata lists, methods, a property),
    # which is exactly the ``dict[str, Any]`` namespace that ``type()`` expects.
    namespace: dict[str, Any] = {"__doc__": docstring, "__module__": __name__}
    for name in cov_names:
        namespace[name] = qv.Float64Column(nullable=True)

    # Store metadata for methods
    namespace["_cov_names"] = cov_names
    namespace["_state_labels"] = coord_labels

    # Add methods
    namespace["from_matrix"] = classmethod(_cov_from_matrix)
    namespace["to_matrix"] = _cov_to_matrix
    namespace["from_sigmas"] = classmethod(_cov_from_sigmas)
    namespace["sigmas"] = property(_cov_sigmas)

    return type(class_name, (qv.Table,), namespace)


# Generate the four covariance classes
CartesianCovariance = _make_covariance_class(
    "CartesianCovariance",
    ["x", "y", "z", "vx", "vy", "vz"],
    "Covariance matrix for Cartesian state [x, y, z, vx, vy, vz].",
)

KeplerianCovariance = _make_covariance_class(
    "KeplerianCovariance",
    ["a", "e", "i", "raan", "ap", "ma"],
    "Covariance matrix for Keplerian elements [a, e, i, raan, ap, ma].",
)

CometaryCovariance = _make_covariance_class(
    "CometaryCovariance",
    ["q", "e", "i", "raan", "ap", "tp"],
    "Covariance matrix for cometary elements [q, e, i, raan, ap, tp].",
)

SphericalCovariance = _make_covariance_class(
    "SphericalCovariance",
    ["rho", "lon", "lat", "vrho", "vlon", "vlat"],
    "Covariance matrix for spherical coords [rho, lon, lat, vrho, vlon, vlat].",
)
