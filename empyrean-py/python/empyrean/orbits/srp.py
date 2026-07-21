"""Solar-radiation-pressure force parameters."""

import quivr as qv


class SRPParams(qv.Table):
    """Solar-radiation-pressure force slot.

    An additive force slot — combinable with the Marsden
    :class:`~empyrean.orbits.nongrav.NonGravParams` on the same orbit.
    Attach it to an orbit table via ``orbits.srp = SRPParams(...)``.

    Only the product ``cr * amrat`` enters the dynamics, so a fitted AMRAT
    absorbs Cr (the JPL AMR convention); ``cr`` is fixed and never fitted —
    only :attr:`amrat` is a fittable parameter.
    """

    # Area-to-mass ratio AMRAT (m^2/kg) — the SRP-effective, fittable
    # parameter. Must be finite and > 0.
    amrat = qv.Float64Column()

    # Radiation-pressure coefficient Cr (typically 1.0-2.0; 1.0 = total
    # absorption). Fixed, never fitted. Must be finite and > 0.
    cr = qv.Float64Column()

    # Prior variance on AMRAT ((m^2/kg)^2); set to open + prior the AMRAT
    # column in a StateAndAMRAT / StateAndNonGravAndAMRAT refine. Null / <=0 =
    # no prior (AMRAT column stays closed; SRP is applied as a fixed force).
    amrat_variance = qv.Float64Column(nullable=True)
