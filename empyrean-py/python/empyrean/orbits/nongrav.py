"""Non-gravitational acceleration parameters."""

import quivr as qv


class NonGravParams(qv.Table):
    """Non-gravitational acceleration parameters.

    Model types:
        "marsden_water"  -- Marsden-Sekanina with standard H2O sublimation g(r)
        "inverse_square" -- Marsden-Sekanina with g(r) = 1/r^2 (Yarkovsky)
        "marsden"        -- Marsden-Sekanina with custom g(r) exponents
        "srp"            -- Solar radiation pressure from AMRAT (a1 = AMRAT in m^2/kg)

    For "marsden", the g(r) function is:
        g(r) = alpha * (r/r0)^{-m} * (1 + (r/r0)^n)^{-k}
    """

    a1 = qv.Float64Column()  # radial (AU/day^2) or AMRAT (m^2/kg)
    a2 = qv.Float64Column()  # transverse (AU/day^2)
    a3 = qv.Float64Column()  # normal (AU/day^2)
    model = qv.LargeStringColumn()

    # Marsden g(r) exponents (used when model="marsden")
    alpha = qv.Float64Column(nullable=True)  # normalizing constant
    r0 = qv.Float64Column(nullable=True)  # reference distance (AU)
    m = qv.Float64Column(nullable=True)  # power-law exponent
    n = qv.Float64Column(nullable=True)  # inner power-law exponent
    k = qv.Float64Column(nullable=True)  # outer damping exponent

    # SRP coefficient (used when model="srp")
    cr = qv.Float64Column(nullable=True)  # radiation pressure coefficient

    # Time delay for g(r) evaluation (days)
    dt = qv.Float64Column(nullable=True)  # outgassing peak offset from perihelion

    # Fitted non-grav 3x3 covariance for (A1, A2, A3), row-major flattened
    # (9 values). Populated by orbit determination (StateAndNonGrav fits) so a
    # fitted orbit re-feeds into a StateAndNonGrav refine without losing its
    # non-grav prior (empyrean-wo4n). Null for SBDB / hand-built / gravity-only.
    covariance = qv.LargeListColumn(qv.Float64Column(), nullable=True)
