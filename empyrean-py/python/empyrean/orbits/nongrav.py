"""Non-gravitational acceleration parameters."""

import quivr as qv


class NonGravParams(qv.Table):
    """Marsden-Sekanina non-gravitational acceleration parameters.

    Model types:
        "marsden_water"  -- Marsden-Sekanina with standard H2O sublimation g(r)
        "inverse_square" -- Marsden-Sekanina with g(r) = 1/r^2 (Yarkovsky)
        "marsden"        -- Marsden-Sekanina with custom g(r) exponents

    For "marsden", the g(r) function is:
        g(r) = alpha * (r/r0)^{-m} * (1 + (r/r0)^n)^{-k}

    Solar radiation pressure is a separate, additive force slot — see
    :class:`~empyrean.orbits.srp.SRPParams` (``orbits.srp``). It is NOT a
    NonGravParams model; ``a1``/``a2``/``a3`` here are always radial /
    transverse / normal Marsden accelerations (AU/day^2).
    """

    a1 = qv.Float64Column()  # radial (AU/day^2)
    a2 = qv.Float64Column()  # transverse (AU/day^2)
    a3 = qv.Float64Column()  # normal (AU/day^2)
    model = qv.LargeStringColumn()

    # Marsden g(r) exponents (used when model="marsden")
    alpha = qv.Float64Column(nullable=True)  # normalizing constant
    r0 = qv.Float64Column(nullable=True)  # reference distance (AU)
    m = qv.Float64Column(nullable=True)  # power-law exponent
    n = qv.Float64Column(nullable=True)  # inner power-law exponent
    k = qv.Float64Column(nullable=True)  # outer damping exponent

    # Time delay for g(r) evaluation (days)
    dt = qv.Float64Column(nullable=True)  # outgassing peak offset from perihelion

    # Prior variance on DT (days^2); set to open + prior the DT column in a
    # StateAndNonGravAndDT refine. Null / <=0 = no prior (DT column stays closed).
    dt_variance = qv.Float64Column(nullable=True)

    # Fitted non-grav 3x3 covariance for (A1, A2, A3), row-major flattened
    # (9 values). Populated by orbit determination (StateAndNonGrav fits) so a
    # fitted orbit re-feeds into a StateAndNonGrav refine without losing its
    # non-grav prior. Null for SBDB / hand-built / gravity-only.
    covariance = qv.LargeListColumn(qv.Float64Column(), nullable=True)

    # 6x3 row-major state-to-(A1, A2, A3) cross covariance, 18 values, in
    # the coordinate's own basis AND UNITS (angular rows in degrees for
    # Cometary / Keplerian / Spherical, matching coordinates.covariance).
    #
    # The other half of the matrix whose diagonal block is `covariance`
    # above, and it requires that block: a cross with no parameter block
    # to condition is refused by the engine, not merely incomplete.
    #
    # Named to match the C ABI's `CoordinateState.non_grav_cross` exactly.
    # It rides here rather than on the coordinates sub-table (where the C
    # ABI puts it) because this package's covariance class is 21
    # synthesized scalar columns with no slot for an 18-value list — a
    # placement divergence, not a semantic one.
    non_grav_cross = qv.LargeListColumn(qv.Float64Column(), nullable=True)
