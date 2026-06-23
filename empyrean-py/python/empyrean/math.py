"""Covariance and mixture math primitives.

Public Python wrappers for the 6-dimensional state-covariance utilities
exposed by the engine: dominant-eigenvector extraction (used wherever
you need the principal axis of an uncertainty ellipsoid) and Gaussian
mixture splitting along that axis (the building block of the AGM
uncertainty propagation method).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixtureComponent:
    """One weighted component of a 6D Gaussian mixture.

    Returned by :func:`split_gaussian`.

    Attributes
    ----------
    weight : float
        Component weight (sum across components is 1.0).
    mean : numpy.ndarray
        6-element mean vector.
    covariance : numpy.ndarray
        6 × 6 covariance matrix.
    """

    weight: float
    mean: np.ndarray
    covariance: np.ndarray


def eigenvector_max_6x6(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Dominant eigenvalue / eigenvector of a 6 × 6 symmetric matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        6 × 6 symmetric (covariance-like) matrix.

    Returns
    -------
    eigenvalue : float
        Largest eigenvalue.
    eigenvector : numpy.ndarray
        Corresponding 6-element unit eigenvector.
    """
    from empyrean._empyrean_rs import _eigenvector_max_6x6

    arr = np.ascontiguousarray(matrix, dtype=np.float64)
    if arr.shape != (6, 6):
        raise ValueError(f"matrix must be 6x6, got shape {arr.shape}")
    eigenvalue, eigenvector = _eigenvector_max_6x6(arr)
    return float(eigenvalue), np.asarray(eigenvector, dtype=np.float64)


def split_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    k: int,
) -> list[MixtureComponent]:
    """Split a 6D Gaussian into ``k`` weighted components along the
    dominant eigenvector of the covariance.

    The split direction is the principal axis of the input covariance
    (matches the engine's adaptive Gaussian-mixture splitter).

    Parameters
    ----------
    mean : numpy.ndarray
        6-element mean vector of the input distribution.
    covariance : numpy.ndarray
        6 × 6 covariance of the input distribution.
    k : int
        Number of mixture components. Must be ≥ 1.

    Returns
    -------
    list[MixtureComponent]
        ``k`` weighted components whose marginal sums to the input
        Gaussian.
    """
    from empyrean._empyrean_rs import _split_gaussian

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    mean_arr = np.ascontiguousarray(mean, dtype=np.float64)
    cov_arr = np.ascontiguousarray(covariance, dtype=np.float64)
    if mean_arr.shape != (6,):
        raise ValueError(f"mean must have shape (6,), got {mean_arr.shape}")
    if cov_arr.shape != (6, 6):
        raise ValueError(f"covariance must be 6x6, got shape {cov_arr.shape}")

    result = _split_gaussian(mean_arr, cov_arr, k)
    weights = np.asarray(result["weights"], dtype=np.float64)
    means = np.asarray(result["means"], dtype=np.float64)
    covs = np.asarray(result["covariances"], dtype=np.float64)

    return [
        MixtureComponent(
            weight=float(weights[i]),
            mean=means[i].copy(),
            covariance=covs[i].copy(),
        )
        for i in range(k)
    ]
