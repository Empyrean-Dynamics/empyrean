# Glossary

Acronyms and conventions that appear across the empyrean docs.
Linked from the cookbook pages where the term first shows up.

## Astrodynamics

```{glossary}
STM
    State Transition Matrix $\Phi(t, t_0) = \partial \mathbf{x}(t) /
    \partial \mathbf{x}_0$. 6×6 in this codebase. Maps state-space
    perturbations from the initial epoch $t_0$ to a target time
    $t$. Linear-order covariance map: $\Sigma_t = \Phi \Sigma_0
    \Phi^\top$. See :class:`~empyrean.StateSensitivities`.

STT
    State Transition Tensor — the second-order analogue of the STM,
    $\Psi_{k,ab}(t, t_0) = \partial^2 x_k / \partial x_{0,a}\,
    \partial x_{0,b}$. 6×6×6. Required for Park-Scheeres
    second-order Gaussian covariance and $\kappa$. See
    :class:`~empyrean.StateSensitivities`.

IOD
    Initial Orbit Determination — the first-pass orbit fit from
    sparse astrometry that seeds differential correction. empyrean uses
    Gauss's method with Herget refinement or systematic ranging
    (admissible region + Manifold of Variations); the method actually
    used is reported on the result.

DC
    Differential Correction — the iterative least-squares orbit fit
    that refines the IOD seed with the full astrometric arc.

CA
    Close Approach — a local minimum in the planet-relative distance
    along a propagated trajectory. Detected per-body by villeneuve's
    event-detection layer.

VA
    Virtual Asteroid — a sample drawn from the orbit-uncertainty
    distribution. A VA sample is the population the
    :class:`~empyrean.MonteCarlo` and
    :class:`~empyrean.SigmaPoint` uncertainty methods sweep across.

IP
    Impact Probability — the probability that a sampled or analytically
    propagated state vector intersects an Earth-equivalent target.
    Reported by :func:`~empyrean.compute_impact_probabilities` under
    multiple uncertainty methods.

B-plane
    The encounter plane perpendicular to the hyperbolic excess
    velocity vector at a planetary close approach. Coordinates
    $(B \cdot T,\, B \cdot R)$ in the Kizner frame are the canonical
    reporting convention ($B \cdot R$ is the minimum-distance miss,
    $B \cdot T$ the timing / resonant-return coordinate).

Keyhole
    A small region in the B-plane that, if traversed, leads to an
    impact at the next encounter (resonant return). Keyhole detection
    feeds into multi-encounter IP rolldowns.

κ (kappa)
    Local nonlinearity diagnostic on a propagated chain — a scalar
    measure of how much the second-order STT contribution would shift
    the propagated mean relative to the linear map at each output
    epoch. Useful as a qualitative flag for which epochs are
    safely-linear vs need a sample-based check. See
    :meth:`~empyrean.StateSensitivities.kappa`.

EIH
    Einstein-Infeld-Hoffmann — the post-Newtonian general-relativity
    correction to N-body equations of motion. Included in
    `ForceModelTier.Basic` and above.

J2-J4
    The first three Earth gravitational zonal harmonic coefficients.
    Included in `ForceModelTier.Standard`.

Marsden non-grav
    The Marsden / Sekanina non-gravitational force model — radial,
    transverse, and normal accelerations $(A_1, A_2, A_3)$ modulated
    by a heliocentric distance-dependent function $g(r)$. Used to
    model Yarkovsky for asteroids and outgassing for comets. See
    :class:`~empyrean.NonGravParams`.

Yarkovsky effect
    Anisotropic thermal re-radiation that produces a small but
    measurable secular force on the orbit of a small body. Modelled
    via the Marsden $A_2$ coefficient.

ADES
    Astrometry Data Exchange Standard — the MPC-blessed astrometric
    observation format. Read with
    :func:`~empyrean.read_ades`; query with
    :func:`~empyrean.query_observations`.

PSV
    Pipe-Separated Values — the canonical ADES file format on disk.
```

## Reference frames and bodies

```{glossary}
ICRF
    International Celestial Reference Frame — the fundamental
    quasi-inertial reference frame for astronomy. Equivalent to
    J2000 equatorial coordinates at the milliarcsecond level.

EclipticJ2000
    Mean ecliptic and equinox of J2000.0 — the integration frame in
    villeneuve / empyrean. The default frame on propagation output.

ITRF93
    International Terrestrial Reference Frame (1993 realisation) —
    Earth-fixed, rotates with the Earth. Used for ground-station
    coordinate vectors.

NAIF ID
    JPL's integer identifier for Solar System bodies (e.g. 10 = Sun,
    399 = Earth, 301 = Moon). Used in the
    :class:`~empyrean.Origin` enum and `body_filter_naif` arguments.

SSB
    Solar System Barycenter — NAIF body 0. The integration origin
    in villeneuve / empyrean.

SPK
    Spacecraft and Planetary Kernel — JPL's binary file format for
    ephemerides. See :doc:`../cookbook/data-setup`.

BPC
    Binary PCK (planetary-constants kernel) — the kernel format for
    Earth-orientation, Moon-orientation, and other body-fixed-frame
    rotations.

GM
    Gravitational parameter $\mu = G M$. The
    ``gm_de440.tpc`` text-PCK file ships bundled in the wheel.

DE440
    JPL Development Ephemeris 440 — the planetary ephemeris empyrean
    integrates against. Pulled in via the ``naif-de440`` PyPI package.

SB441
    JPL's small-body mass perturber ephemeris family — released
    paired with DE441 but used here alongside DE440. The 16-asteroid
    "N16" cut is what `ForceModelTier.Standard` integrates against.

EOP
    Earth Orientation Parameters — drives ITRF93 ↔ ICRF and the
    Earth-fixed surface-station vectors. Pulled in via the
    ``naif-eop-*`` PyPI packages.
```

## Time scales

```{glossary}
MJD
    Modified Julian Date — JD - 2400000.5. The convention used
    everywhere in empyrean for raw time-as-float values.

TDB
    Barycentric Dynamical Time — the time scale of the integration
    and the default scale on every Epochs / MJD value at the API
    boundary unless explicitly converted.

UTC
    Coordinated Universal Time — leap-second-aware civil time.
    Convert in / out via :meth:`~empyrean.Epochs.to_utc` /
    :meth:`~empyrean.Epochs.from_iso`.

TT
    Terrestrial Time — TDB to within milliseconds for most uses;
    the Earth-orientation kernels (BPCs) are built on TT.

TAI
    International Atomic Time — TAI = UTC + leap seconds.
```

## Observation models

```{glossary}
Vereš et al. weighting
    The standard astrometric uncertainty model for MPC observations
    (Vereš et al. 2017). Combines a per-station floor with a
    per-catalog systematic. Driven by
    :class:`~empyrean.ODConfig` defaults.

Cook's distance
    Per-observation influence diagnostic — measures how much the fit
    moves when a single observation is removed. Reported in
    :class:`~empyrean.ObservationResults`. $D_i \gtrsim 1$ is
    typically a flag for review.

Leverage
    Per-observation diagonal of the projection matrix — bounded in
    $[0, 2]$. High-leverage observations sit at the edges of the
    arc and disproportionately constrain the fit; their residuals
    can be misleadingly small even when they're driving the solution.

Adaptive rejection
    The Carpino / Milani / Chesley 2003 adaptive $\chi^2$ rejection
    scheme. Default in :class:`~empyrean.ODConfig`.
```
