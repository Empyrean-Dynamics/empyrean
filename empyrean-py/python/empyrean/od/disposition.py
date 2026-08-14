"""What a fit does with each parameter axis: solved, considered, or fixed."""

from __future__ import annotations

import enum

__all__ = ["ParamDisposition"]


class ParamDisposition(str, enum.Enum):
    """A parameter axis's treatment in a fit.

    Subclasses ``str`` so values serialize directly into the string
    columns that carry them, matching the convention
    :class:`~empyrean.propagation.tagged_covariance.CovarianceKind` and
    ``non_grav.model`` already use.

    The three are different operations with different mathematics, and
    using one where another is meant is silent — each produces a
    well-formed covariance:

    - :attr:`FIXED` — marginalized out of the prior. Contributes
      nothing and changes no number.
    - :attr:`SOLVED` — estimated from the data. Occupies a solved slot
      and comes back with a posterior variance.
    - :attr:`CONSIDERED` — not estimated, but uncertain: its prior
      uncertainty reaches the posterior through its measurement
      partials (Schmidt-Kalman consider analysis; Tapley, Byron D.,
      Schutz, Bob E., and Born, George H., *Statistical Orbit
      Determination*, Elsevier Academic Press, 2004, ch. 6).

    A considered axis is **not** a safety margin. Under an uncorrelated
    prior the correction strictly widens the posterior, but when the
    orbit supplies cross terms between the considered axis and the
    solved ones the correction is sign-indefinite and the posterior can
    come back **tighter**.

    No boolean coercion
    -------------------

    There is deliberately no ``from_bool`` and no truthiness contract
    beyond ``str``'s. A disposition is a modelling statement, and a
    conversion that turned ``True`` into ``SOLVED`` would let a call
    site inherit one rather than state it. Passing ``True`` where a
    disposition is expected raises rather than resolving:

    >>> ParamDisposition.parse(True)
    Traceback (most recent call last):
      ...
    TypeError: parameter disposition must be a string ...
    """

    FIXED = "fixed"
    SOLVED = "solved"
    CONSIDERED = "considered"

    @classmethod
    def parse(cls, value: object) -> ParamDisposition:
        """Parse a wire tag, refusing anything that is not one.

        A ``bool`` is refused by name rather than coerced — the whole
        point of the tri-state is that "not solved" is two different
        answers, and ``False`` cannot say which.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise TypeError(
                "parameter disposition must be a string, not a bool: True/False "
                "cannot say whether an unsolved axis is 'considered' (its "
                "uncertainty inflates the posterior) or 'fixed' (it contributes "
                "nothing). Pass 'solved', 'considered' or 'fixed'."
            )
        if not isinstance(value, str):
            raise TypeError(f"parameter disposition must be a string, got {type(value).__name__}")
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"unknown parameter disposition {value!r}; expected "
                f"{', '.join(repr(m.value) for m in cls)}"
            ) from None

    @property
    def is_solved(self) -> bool:
        """Whether this axis is estimated (and so occupies a solved slot)."""
        return self is ParamDisposition.SOLVED

    @property
    def is_considered(self) -> bool:
        """Whether this axis inflates the posterior through consider analysis."""
        return self is ParamDisposition.CONSIDERED
