<!-- generated from empyrean-py/docs/api.md by agentdocs.py; do not edit -->

## Pipeline entry points
`propagate(orbits, epochs, config=…, *, force_model=…, uncertainty_method=…, num_threads=…, events=…, tagged_covariance=…, thrust_arcs=…)`
`generate_ephemeris(orbits, observers, config=…, *, force_model=…, uncertainty_method=…)`
`determine(observations, initial_orbits=…, *, radar=…, config=…)`
`evaluate(orbit, observations, *, config=…)`
`refine(orbit, observations, *, config=…)`
`transform_coordinates(coordinates, target, frame=…, origin=…)`
`get_states(target, center, epochs, frame=…)`
`get_observer_states(obs_codes, epochs, frame=…, origin=…)`
`compute_impact_probabilities(orbits, end_epoch, methods, body_filter=…)`
`compute_b_planes(orbits, end_epoch, methods, body_filter=…)`

## I/O helpers
`query_sbdb(names, cache_dir=…)`
`query_horizons(names, observer, epochs, cache_dir=…)`
`query_horizons_vectors(command, epoch, cache_dir=…)`
`query_observations(designations, cache_dir=…)`
`query_radar(designations, cache_dir=…)`
`read_ades(path_or_string)`
`initialize(*, data_dir=…, de440_path=…, gm_path=…, refresh=…)`
`download_data(*, data_dir=…)`
`default_data_dir()`
`versions()`
`version_string()`

## Coordinates & orbits
`CartesianCoordinates` (11 methods) — Cartesian state vectors.
`KeplerianCoordinates` (10 methods) — Keplerian orbital elements.
`CometaryCoordinates` (11 methods) — Cometary orbital elements.
`SphericalCoordinates` (9 methods) — Spherical coordinates (topocentric / observer-centric).
`CartesianCovariance` (24 methods) — Covariance matrix for Cartesian state [x, y, z, vx, vy, vz].
`KeplerianCovariance` (24 methods) — Covariance matrix for Keplerian elements [a, e, i, raan, ap, ma].
`CometaryCovariance` (24 methods) — Covariance matrix for cometary elements [q, e, i, raan, ap, tp].
`SphericalCovariance` (24 methods) — Covariance matrix for spherical coords [rho, lon, lat, vrho, vlon, vlat].
`CartesianOrbits` (9 methods) — Orbits in Cartesian coordinates.
`KeplerianOrbits` (9 methods) — Orbits in Keplerian elements.
`CometaryOrbits` (9 methods) — Orbits in cometary elements.
`SphericalOrbits` (9 methods) — Orbits in spherical coordinates.
`NonGravParams` (10 methods) — Marsden-Sekanina non-gravitational acceleration parameters.
`PhotometricParams` (covariance, g, g1, g12, g2, h, model) — Photometric parameters for apparent magnitude computation.
`SRPParams` (amrat, cr) — Solar-radiation-pressure force slot.
`PackedJoint` (from_wire, state_block, to_wire) — One packed joint covariance at a single ``(orbit, epoch)``.
`PackedJoints` (18 methods) — Row-aligned packed joint covariances — the orbit covariance column.
`Epochs` (19 methods) — Epochs as Modified Julian Dates with an explicit time scale.
`TimeScale` — Time scale for epoch values.
`Frame` — Reference frame for coordinate states.
`Origin` (asteroid, from_string) — Origin (center body) for a coordinate state.

### Continuous thrust
`ThrustParams` — Thrust parameters for one orbit: thrust arcs plus optional
`ThrustArc` — A single continuous-thrust arc with smooth on/off switching and
`SteeringLaw`
`ConstantRTN` — Constant RTN steering angles relative to the arc's central body.
`VelocityTangent` — Thrust aligned with the velocity vector relative to the central
`InertialFixed` — Fixed direction in the inertial frame (normalized internally).

## Observers
`Observers` (coordinates, from_code, from_codes, obs_code, observing_night) — Precomputed observer states as Cartesian coordinates.

## Propagation
`PropagationConfig` — Configuration for orbit propagation.
`PropagationResult` (from_dir, mixture_chains, rows_for, tagged_covariance_series, to_dir) — Result of orbit propagation.
`DiagnosticsConfig` — Per-trajectory diagnostic outputs (sensitivity, nonlinearity,
`AdvancedIntegratorConfig` — Integrator-tuning knobs.
`IntegratorChoice` — Integrator backend selector.
`OriginSwitchingConfig` — Trajectory splitting at body acceleration-dominance boundaries
`EphemerisOverlapPolicy` — What to do when the propagated state coincides with an SB441-N16
`ForceModelTier` — Force model tier for propagation.
`UncertaintyMethod` — Uncertainty propagation method — shortcut identifiers for the
`Auto` — Adaptive per-close-approach uncertainty-method selection.
`SigmaPoint` — Sigma-point (unscented) transform.
`MonteCarlo` — Monte Carlo sampling.
`EventConfig` — Configuration for event detection during propagation.
`Events` (count_by_type, select_orbit) — All events detected during propagation, grouped by type.
`EventSummary` (body, epochs, event_type, object_id, orbit_id) — One-row-per-event summary across all event types.
`CloseApproachStarts` (body, distance_au, distance_km, epochs, object_id, orbit_id) — Close approach zone entry events.
`CloseApproachEnds` (body, distance_au, distance_km, epochs, object_id, orbit_id) — Close approach zone exit events.
`Periapses` (13 methods) — Periapsis (closest approach) events within a CA zone.
`Impacts` (altitude_km, body, epochs, latitude_deg, longitude_deg, object_id, orbit_id, relative_velocity_au_day) — Nominal impact events (body surface intersection).
`PossibleImpacts` (15 methods) — Probabilistic impact assessments from covariance analysis.
`AtmosphericEntries` (9 methods) — Atmospheric entry events (Karman line inbound crossing).
`AtmosphericExits` (body, distance_au, epochs, object_id, orbit_id) — Atmospheric exit events.
`CaptureStarts` (12 methods) — Temporary gravitational capture start events.
`CaptureEnds` (13 methods) — Temporary gravitational capture end (escape) events.
`ShadowEntries` (body, epochs, illumination, object_id, orbit_id, shadow_fraction) — Shadow zone entry events (Sun occluded by a body).
`ShadowExits` (body, epochs, illumination, object_id, orbit_id, shadow_fraction) — Shadow zone exit events (Sun no longer occluded).
`CovarianceRegimeChanges` (9 methods) — Covariance-regime transitions from ``UncertaintyMethod.AUTO``.

### Tagged covariance
`TaggedCovariance` — Provenance-tagged covariance at a single ``(orbit, epoch)``.
`TaggedCovariances` (51 methods) — Per-``(orbit, epoch)`` provenance-tagged covariance readback.
`CovarianceKind` — How a covariance was derived.
`JointQuality` — Definiteness of a packed joint matrix.
`JointFunctional` — The functional a packed joint's second moment describes.
`JointProvenance` — Where a packed joint came from.
`AssertedSource` — Which external source asserted a joint's second moment.
`GaussianMixture` — Adaptive Gaussian mixture (AGM).

### System handles
`build_system(force_model=…, frame=…)`
`od_system(force_model=…)`
`BuiltSystem` (describe, determine, evaluate, generate_ephemeris, propagate, refine) — A reusable pre-built force-model handle.
`SystemDescription` — A reproducibility summary of a :class:`BuiltSystem`'s frozen force
`KernelRecord` — One entry of the kernel-identity manifest a handle captured at
`KernelKind` — Category of a loaded data file in a handle's kernel manifest.
`KernelProvenance` — Where a kernel in a handle's manifest came from.

## Ephemeris
`Ephemeris` (25 methods) — Predicted astrometric ephemeris for observed objects.
`EphemerisConfig` — Configuration for :func:`empyrean.generate_ephemeris`.
`EphemerisResult` (from_dir, rows_for, to_dir) — Result of :func:`empyrean.generate_ephemeris`.

## Impact probability and B-plane
`ImpactProbabilities` (29 methods) — Probabilistic impact assessments tagged by uncertainty method.
`BPlanes` (16 methods) — B-plane geometry breakdowns tagged by uncertainty method.

## Observation planning
`evaluate_plan(orbit, planned, config=…, orbit_id=…)`
`PlanResult` — The result of evaluating an observation plan.
`PlanMetrics` (10 methods) — Covariance summary metrics bracketing the campaign — two rows.
`PlanCandidates` (26 methods) — Per-candidate information gain — one row per planned observation.
`PlanEphemeris` (dec_deg, epochs, ra_deg) — Predicted sky position at each optical candidate's epoch.

### Planning configuration
`PlanningConfig` — Configuration for :func:`empyrean.evaluate_plan`.
`ObservatoryConfig` — Per-site astrometric assumptions and observability filters.
`STAGE_PRIOR`
`STAGE_POSTERIOR`
`PlannedObservation` (optical, radar) — One candidate observation: when it would be taken and what it
`PlannedObservationKind` — What a :class:`PlannedObservation` would measure.
`RadarMode` — Which radar observable(s) a candidate would measure.
`RadarStation` — A radar dish the planner can schedule against.

## Math primitives
`eigenvector_max_6x6(matrix)`
`split_gaussian(mean, covariance, k)`
`MixtureComponent` — One weighted component of a 6D Gaussian mixture.

## Sensitivity
`StateSensitivities` (19 methods) — Per-``(orbit, epoch)`` state-transition matrices and tensors.
`ObservationSensitivities` (13 methods) — Per-``(orbit, observer, epoch)`` observation Jacobians and Hessians.
`SENSITIVITY_ROW_RANGE`
`SENSITIVITY_ROW_RA`
`SENSITIVITY_ROW_DEC`
`SENSITIVITY_ROW_VRANGE`
`SENSITIVITY_ROW_VRA`
`SENSITIVITY_ROW_VDEC`

## Orbit determination
`ADESObservations` (35 methods) — ADES optical observations — full schema.
`ADESRadarObservations` (18 methods) — ADES radar observations — full schema.
`ODConfig` — Unified orbit-determination configuration.
`DetermineResult` — Result of orbit determination — returned by both
`DetermineResults` (single) — The result of a batch orbit determination — every object the
`DetermineFailure` — Why one object in a batch produced no orbit.
`EvaluateResult` — Result of orbit evaluation (residuals only, no fitting).
`ObservationResults` (39 methods) — Per-observation OD results — full upstream surface.
`ResidualSummary` — Aggregate residual statistics over a set of observations.
`FitSummary` (26 methods) — One row per **input** object of a batch orbit determination —
`BandStat` — Per-band photometric fit statistics. Mirrors ``empyrean::BandStat``.
`AcceptabilityReport` — Structured fit-quality verdict — mirrors
`GateRecord` — One model-ladder gate decision from the photometric fit. Mirrors
`SolvedCovariance` — Full tagged solved-parameter covariance from a wide OD fit.
`StallDelivery` — The numbers behind a fit delivered by the stable-stall acceptance
`CovarianceTrust` — Event-aware trust verdict on the delivered covariance, evaluated
`TrustGateEvent` — The intervening event named by an ``encounter_intervenes``
`StationBiases` (10 methods) — Per-station fitted nuisance biases.
`OutputEpoch` — Output epoch for the fitted orbit. Mirrors
`OutputEpochMode` — How :class:`OutputEpoch` selects the fitted-orbit epoch.
`Session` (9 methods) — Stateful orbit-determination handle.
`SessionDiff` — Pairwise diagnostic between two fits in the same session.

### Post-OD photometry
`PhotometryConfig` — Post-OD photometric-fit configuration. Mirrors
`PhotometryModel` — Photometric model for the post-OD phase-function fit.
`PhotometryResult` — Post-OD photometric solution — an H/G fit over the arc's

### OD configuration
`WeightingConfig` — Observation weighting pipeline. Mirrors
`WeightingLayer` — One element of the weighting pipeline. Tagged-union shape —
`WeightingLayerKind` — Discriminator for :class:`WeightingLayer` variants.
`WeightingPreset` — Preset selector for :class:`WeightingConfig`.
`SigmaPolicy` — How a weighting layer's σ combines with the per-observation
`DebiasingConfig` — Catalog-bias-correction configuration. Mirrors scott's
`DebiasingResolution` — Healpix resolution of a debiasing table.
`StationRADecConfig` — Per-station RA/Dec bias-fit configuration.
`RejectionConfig` — Outlier-rejection configuration. The active fields are
`RejectionKind` — Outlier-rejection strategy selector.
`SolveForParams` — Parameters to solve for in differential correction.
`SolveFor` — What the fit does with each parameter axis. Mirrors ``empyrean::SolveFor``.
`ParamDisposition` (parse) — A parameter axis's treatment in a fit.
`IODConfig` — IOD ranging tuning. Mirrors the IOD section of
`OriginPolicy` — Origin-policy selector for the OD pipeline. Mirrors
`OriginPolicyMode` — How :class:`OriginPolicy` selects the central body for IOD + DC.
`AutoEscalationPolicy` — Trigger thresholds for :attr:`SolveForParams.AUTO` escalation.
`AcceptabilityThresholds` — Thresholds for the post-DC fit-acceptability sub-checks. Mirrors
`CovarianceRepresentation` — Coordinate basis the OD output covariance is reported in.
