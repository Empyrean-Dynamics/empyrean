//! # empyrean
//!
//! High-precision Solar System dynamics — trajectory propagation,
//! ephemeris generation, orbit determination, and event analysis
//! (close approaches, occultations, eclipses, sphere-of-influence
//! crossings) for real Solar System bodies.
//!
//! Safe Rust wrapper over `empyrean-sys`. Use this crate to propagate
//! orbits, generate ephemerides, and determine orbits from observations
//! without writing `unsafe` code.
//!
//! ## Standard workflow
//!
//! Pull an orbit from JPL SBDB, propagate it forward, generate
//! ephemerides at an observatory, and inspect detected events:
//!
//! ```no_run
//! use empyrean::{Context, EphemerisConfig, Origin, PropagationConfig};
//!
//! let ctx = Context::from_data_dir(None)?;
//!
//! // 1. Pull Apophis from SBDB (CometaryCoordinates with covariance).
//! let batch = empyrean::query_sbdb(&["99942"], None)?;
//!
//! // 2. Propagate 10 years past the SBDB epoch.
//! let cfg = PropagationConfig::default();
//! let t0 = batch.orbits[0].state.epoch.mjd_tdb()?;
//! let epochs = vec![empyrean::Epoch::from_mjd_tdb(t0 + 10.0 * 365.25)];
//! let result = ctx.propagate(&batch.orbits, &epochs, &cfg)?;
//! println!("{} states, {} events", result.states.len(), result.events.len());
//!
//! // 3. Predict on-sky positions at Mauna Kea (MPC code 568).
//! let observers = ctx.get_observers(&["568"], &epochs)?;
//! let eph_cfg = EphemerisConfig::default();
//! let eph = ctx.generate_ephemeris(&batch.orbits, &observers, &eph_cfg)?;
//! # Ok::<(), empyrean::Error>(())
//! ```
//!
//! For close-approach analysis (impact probability, B-plane geometry),
//! see [`Context::compute_impact_probabilities`] and
//! [`Context::compute_b_planes`]. For OD from astrometric
//! observations, see [`Context::determine`] and the [`Session`] type
//! for interactive masking workflows.
//!
//! ## Coordinate transform
//!
//! ```no_run
//! use empyrean::{Context, CoordinateState, Frame, Origin, Representation};
//!
//! let ctx = Context::from_data_dir(None)?;
//! let input = CoordinateState::cometary(
//!     empyrean::Epoch::from_mjd_tdb(60200.0),
//!     [0.7461, 0.1914, 3.339, 204.446, 126.687, 60159.0],
//!     Frame::EclipticJ2000,
//!     Origin::SUN,
//! );
//! let cart = ctx.transform(&input, Representation::Cartesian, Frame::ICRF, Origin::SUN)?;
//! println!("x = {:.6} AU", cart.elements[0]);
//! # Ok::<(), empyrean::Error>(())
//! ```
//!
//! ## Quick reference
//!
//! | You want…                              | API                                              |
//! |----------------------------------------|--------------------------------------------------|
//! | Propagate orbits to target epochs      | [`Context::propagate`]                           |
//! | Predict observations at observatories  | [`Context::generate_ephemeris`]                  |
//! | Fit an orbit to observations           | [`Context::determine`]                           |
//! | Re-fit with a Bayesian prior           | [`Context::refine`]                              |
//! | Residuals only — no fit                | [`Context::evaluate`]                            |
//! | Stateful, mask-and-refit OD            | [`Session`]                                      |
//! | Impact probability                     | [`Context::compute_impact_probabilities`]        |
//! | B-plane geometry                       | [`Context::compute_b_planes`]                    |
//! | Convert between coordinate types       | [`Context::transform`]                           |
//! | Body / observer states                 | [`Context::get_states`] / [`Context::get_observers`] |
//! | Pull an orbit from JPL SBDB            | [`query_sbdb`]                                   |
//! | Pull predicted ephemeris from Horizons | [`query_horizons`]                               |
//! | Pull SSB state vectors from Horizons   | [`query_horizons_vectors`]                       |
//! | Pull observations from MPC             | [`query_observations`]                           |
//! | Pull radar astrometry from JPL         | [`query_radar`]                                  |
//! | Read ADES PSV observations             | [`Context::read_ades`]                           |
//! | Default data directory                 | [`default_data_dir`]                             |
//!
//! ## Conventions
//!
//! - **Distances** in AU; **velocities** in AU/day; **angles** in
//!   **degrees** at the API boundary (radians internally).
//! - **Epochs** are MJD on the **TDB** scale unless otherwise stated.
//!   See [`time::Epoch`] for time-scale-aware values and
//!   [`time::iso_to_mjd`] / [`time::mjd_to_iso`] for ISO 8601 interop.
//! - **Default integrator** is
//!   [`IntegratorChoice::GR15`] (median Horizons error ≈ 35 m).
//!   Switch to [`IntegratorChoice::DOP853`] via
//!   [`AdvancedIntegratorConfig::integrator`] for ~1.4× speed at the
//!   cost of ~10× position error.
//! - **Default frame** for propagation output is
//!   [`Frame::EclipticJ2000`] (the integration frame); set
//!   [`PropagationConfig::frame`] to [`Frame::ICRF`] for ICRF output.
#![warn(missing_docs)]

mod built_system;
mod context;
mod coordinate;
mod ephemeris;
mod error;
mod impact;
mod io;
mod math;
mod observers;
mod od;
mod orbit;
mod propagate;
mod query;
mod session;
mod states;
mod thrust;
pub mod time;
mod transform;
mod version;

pub use built_system::{
    BuiltSystem, BuiltSystemGuardError, KernelKind, KernelProvenance, KernelRecord,
    SystemDescription,
};
pub use context::{Context, default_data_dir, download_data};
pub use coordinate::{
    CoordinateState, Frame, Origin, Representation, frame_to_int, int_to_frame, int_to_rep,
    rep_to_int,
};
pub use ephemeris::{EphemerisConfig, EphemerisEntry, EphemerisResult, ObservationSensitivity};
pub use error::{Error, Result};
pub use impact::{BPlane, ImpactProbability};
pub use io::{
    OrbitBatch, read_orbits_csv, read_orbits_json, read_orbits_parquet, write_ephemeris_csv,
    write_ephemeris_json, write_ephemeris_parquet, write_events_csv, write_events_json,
    write_events_parquet, write_orbits_csv, write_orbits_json, write_orbits_parquet,
    write_residuals_csv, write_residuals_json, write_residuals_parquet,
};
pub use math::{MixtureComponent, eigenvector_max_6x6, split_gaussian};
pub use observers::Observer;
pub use od::{
    AcceptabilityReport, AcceptabilityThresholds, AutoEscalationPolicy, CovarianceRepresentation,
    DebiasingConfig, DebiasingResolution, DetermineResult, EvaluateResult, IODConfig, ODConfig,
    Observation, ObservationResidual, Observations, OriginPolicy, OutputEpoch, RadarMeasurement,
    RadarObservation, RejectionConfig, RejectionKind, RejectionReason, ResidualSummary,
    SigmaPolicy, SolveForParams, StationBias, StationRaDecConfig, WeightingConfig, WeightingLayer,
    WeightingPreset,
};
pub use orbit::{Orbit, PhaseFunction};
pub use propagate::{
    AdvancedIntegratorConfig, CovarianceKind, CovarianceQuality, DiagnosticsConfig, Event,
    EventConfig, ForceModelTier, IntegratorChoice, OriginSwitchingConfig, PropagatedState,
    PropagationConfig, PropagationResult, TaggedCovariance, TargetFunctional, UncertaintyMethod,
};
pub use query::{
    query_horizons, query_horizons_vectors, query_observations, query_radar, query_sbdb,
};
pub use session::{Session, SessionDiff};
pub use states::State;
pub use thrust::{SteeringLaw, ThrustArc, ThrustParams};
pub use time::{Epoch, TimeScale, iso_to_mjd, mjd_to_iso};
pub use version::{Versions, version_string, versions};

// Compile the Rust examples in both READMEs as doctests under
// `cargo test --doc` so they cannot rot against the public API. The
// `cfg(doctest)` gate keeps these synthetic structs out of the public
// rustdoc — they exist only during doc-test compilation. `../README.md`
// is the crate (crates.io) README; `../../README.md` is the top-level
// workspace README.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct CrateReadmeDoctests;

#[cfg(doctest)]
#[doc = include_str!("../../README.md")]
struct WorkspaceReadmeDoctests;
