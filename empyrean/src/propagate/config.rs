//! Propagation configuration: force model, uncertainty method,
//! integrator backend, event-detection toggles, and the top-level
//! [`PropagationConfig`] passed to [`Context::propagate`](super::Context::propagate).

use crate::coordinate::{Frame, Origin};
use crate::error::{Error, Result};

/// Force model tier. Each tier adds physics on top of the previous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ForceModelTier {
    /// Point-mass planets + Moon + Pluto. Fast, for visualization.
    Approximate = 0,
    /// Approximate + EIH general relativity + Sun J2.
    Basic = 1,
    /// Basic + 16 asteroid perturbers + Earth J2-J4 + Marsden non-grav. Default.
    Standard = 2,
}

/// Uncertainty propagation method.
///
/// Picks how the input covariance is mapped through the dynamics. The
/// default is [`UncertaintyMethod::FirstOrder`] — adequate for the bulk
/// of NEO work where the dynamics stay linear over the uncertainty
/// volume. Reach for [`UncertaintyMethod::SecondOrder`] near a
/// planetary close approach (for the second-order impact-probability
/// correction) and the sample-based methods when you need tail
/// probabilities or want to exercise the full distribution.
#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyMethod {
    /// First-order STM-only covariance propagation. Default.
    FirstOrder,
    /// Second-order — analytic Hessians, STM + STT.
    SecondOrder,
    /// Unscented sigma-point transform.
    SigmaPoint {
        /// Number of sigma deviations (default: 1.0).
        n_sigma: f64,
        /// Points per coordinate-plane pair (default: 8 → 120 total).
        samples_per_plane: usize,
    },
    /// Monte Carlo sampling.
    MonteCarlo {
        /// Number of random samples.
        n_samples: usize,
        /// RNG seed. `None` draws from `thread_rng` (not reproducible).
        seed: Option<u64>,
    },
    /// Adaptive method selection based on local nonlinearity. The engine
    /// escalates the uncertainty method automatically through close
    /// approaches and relaxes it elsewhere, combining the
    /// [`FirstOrder`](Self::FirstOrder),
    /// [`SecondOrder`](Self::SecondOrder), and adaptive-Gaussian-mixture
    /// methods. The switching points are caller-tunable via the fields
    /// below; the [`auto()`](Self::auto) constructor sets the engine
    /// defaults. References: Park & Scheeres 2006; DeMars-Bishop-Jah
    /// 2013; Roa et al. 2021 (Sentry-II).
    Auto {
        /// First-order nonlinearity tuning parameter.
        threshold_first: f64,
        /// Adaptive-mixture nonlinearity tuning parameter.
        threshold_mixture: f64,
        /// Impact-probability floor for the higher-order pass.
        threshold_ip_skip: f64,
        /// Adaptive-Gaussian-mixture maximum recursion depth.
        gmm_max_depth: usize,
        /// Adaptive-Gaussian-mixture components per split (odd).
        gmm_components_per_split: usize,
    },
    /// Adaptive Gaussian mixture (AGM) as a top-level method. The engine
    /// recursively splits the input Gaussian into a mixture wherever the
    /// local nonlinearity exceeds `threshold`, propagating each component
    /// and recombining at the output. Its distinctive product is the
    /// mixture-corrected impact probability at close approaches; away
    /// from encounters the output-state covariance is the linear
    /// \\( \Phi \Sigma \Phi^\top \\) mapping (like
    /// [`SecondOrder`](Self::SecondOrder)), so for a well-determined
    /// object it reads back very close to
    /// [`FirstOrder`](Self::FirstOrder) — that is expected, not a bug.
    /// Construct with [`gaussian_mixture()`](Self::gaussian_mixture) for
    /// the engine defaults. Reference: DeMars-Bishop-Jah (JGCD 2013).
    Mixture {
        /// Nonlinearity threshold above which the splitter fires
        /// (default: 1.0).
        threshold: f64,
        /// Maximum recursion depth for nested splitting (default: 3).
        max_depth: usize,
        /// Number of sub-Gaussians produced per split; the
        /// DeMars-Bishop-Jah splitting tables are tabulated only for odd
        /// counts (3 or 5). Default: 3.
        components_per_split: usize,
    },
}

impl UncertaintyMethod {
    /// SigmaPoint with default parameters (n_sigma = 1.0, samples_per_plane = 8).
    pub fn sigma_point() -> Self {
        Self::SigmaPoint {
            n_sigma: 1.0,
            samples_per_plane: 8,
        }
    }

    /// Monte Carlo with the given sample count and a fixed reproducibility seed.
    pub fn monte_carlo(n_samples: usize) -> Self {
        Self::MonteCarlo {
            n_samples,
            seed: Some(42),
        }
    }

    /// Auto with the engine-default thresholds.
    pub fn auto() -> Self {
        Self::Auto {
            threshold_first: 0.1,
            threshold_mixture: 10.0,
            threshold_ip_skip: 1e-12,
            gmm_max_depth: 3,
            gmm_components_per_split: 3,
        }
    }

    /// Adaptive Gaussian mixture with the engine-default parameters
    /// (threshold = 1.0, max_depth = 3, components_per_split = 3).
    pub fn gaussian_mixture() -> Self {
        Self::Mixture {
            threshold: 1.0,
            max_depth: 3,
            components_per_split: 3,
        }
    }

    pub(crate) fn to_ffi(&self) -> empyrean_sys::EmpyreanUncertaintyMethod {
        let mut out = empyrean_sys::EmpyreanUncertaintyMethod {
            tag: 0,
            sp_n_sigma: 0.0,
            sp_samples_per_plane: 0,
            mc_n_samples: 0,
            mc_seed_some: 0,
            mc_seed: 0,
            auto_threshold_first: 0.0,
            auto_threshold_mixture: 0.0,
            auto_threshold_ip_skip: 0.0,
            auto_gmm_max_depth: 0,
            auto_gmm_components_per_split: 0,
        };
        match *self {
            Self::FirstOrder => out.tag = 0,
            Self::SecondOrder => out.tag = 1,
            Self::SigmaPoint {
                n_sigma,
                samples_per_plane,
            } => {
                out.tag = 2;
                out.sp_n_sigma = n_sigma;
                out.sp_samples_per_plane = samples_per_plane as u64;
            }
            Self::MonteCarlo { n_samples, seed } => {
                out.tag = 3;
                out.mc_n_samples = n_samples as u64;
                out.mc_seed_some = u8::from(seed.is_some());
                out.mc_seed = seed.unwrap_or(0);
            }
            Self::Auto {
                threshold_first,
                threshold_mixture,
                threshold_ip_skip,
                gmm_max_depth,
                gmm_components_per_split,
            } => {
                out.tag = 4;
                out.auto_threshold_first = threshold_first;
                out.auto_threshold_mixture = threshold_mixture;
                out.auto_threshold_ip_skip = threshold_ip_skip;
                out.auto_gmm_max_depth = gmm_max_depth as u64;
                out.auto_gmm_components_per_split = gmm_components_per_split as u64;
            }
            Self::Mixture {
                threshold,
                max_depth,
                components_per_split,
            } => {
                // Top-level Mixture reuses the AGM parameter slots shared
                // with Auto (`auto_threshold_mixture` / `auto_gmm_max_depth`
                // / `auto_gmm_components_per_split`); the tag disambiguates
                // a standalone GaussianMixture from Auto, so no new FFI
                // fields are needed.
                out.tag = 5;
                out.auto_threshold_mixture = threshold;
                out.auto_gmm_max_depth = max_depth as u64;
                out.auto_gmm_components_per_split = components_per_split as u64;
            }
        }
        out
    }
}

/// Integrator backend selector.
///
/// Numbers below are median per-step position error against a JPL
/// Horizons reference orbit on a 1-year asteroid propagation; both
/// integrators converge with `epsilon = 1e-9`.
///
/// IAS15 is intentionally not built into this distribution — callers
/// needing it must use a custom engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegratorChoice {
    /// Gauss-Radau 15 (Everhart 1985; Rein &amp; Spiegel
    /// 2015). Default. Median Horizons error ≈ 35 m. Tightest accuracy
    /// at the cost of ~1.4× the wall-clock of DOP853.
    #[default]
    GR15,
    /// Dormand-Prince 8(5,3) (Hairer, Nørsett &amp; Wanner
    /// 1993, §II.5). ~1.4× faster than GR15 with looser median Horizons
    /// error (~358 m vs GR15's ~35 m). Reach for it on bulk surveys
    /// where 100-m-class accuracy is acceptable.
    DOP853,
}

/// Trajectory splitting at body Laplace SOIs (Amato/Baù/Bombardelli
/// 2017 §6). Default **enabled** — chaotic Earth-encounter
/// trajectories are re-centered on the dominant body during flybys,
/// preserving sub-meter precision. Set `enabled = false` to opt out.
///
/// At this surface, `enabled = true` selects every monitored body —
/// the per-body opt-in list is not exposed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OriginSwitchingConfig {
    /// Enable trajectory splitting. Default `true`.
    pub enabled: bool,
    /// Hysteresis band width for origin-switching.
    pub hysteresis: f64,
}

impl Default for OriginSwitchingConfig {
    fn default() -> Self {
        Self {
            // Harmonised with villeneuve's library default and the
            // empyrean-c `_DEFAULT` sentinel resolution. The whole
            // distribution chain — empyrean-core, empyrean-c (C ABI),
            // empyrean (this wrapper), empyrean-py, empyrean-cli —
            // defaults `enabled = true` for the planetary-science
            // brand promise (chaotic Earth-encounter trajectories
            // preserved at sub-meter precision via re-centering on
            // the dominant body during flybys).
            enabled: true,
            hysteresis: 0.2,
        }
    }
}

/// Integrator-tuning knobs.
///
/// Defaults are calibrated for production. Most callers don't touch
/// this — [`PropagationConfig::advanced`] exists to make the surface
/// complete and to enable bespoke runs (custom step bounds for tight
/// encounters, dense output for visualization, integrator backend
/// switching, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdvancedIntegratorConfig {
    /// Integrator backend (default GR15).
    pub integrator: IntegratorChoice,
    /// Truncation-error tolerance — relative b₆ for GR15, rtol for
    /// DOP853 (paired with a fixed atol = 1e-14).
    pub epsilon: f64,
    /// Initial step size in days. `None` = auto from orbital timescale.
    pub dt_initial: Option<f64>,
    /// Minimum allowed step size in days. `None` = auto.
    pub dt_min: Option<f64>,
    /// Encounter dynamical-timescale step floor divisor.
    pub encounter_timescale_divisor: f64,
    /// Maximum integration steps before aborting.
    pub max_steps: usize,
    /// Memory cap on the per-step b-coefficient cache.
    pub max_dense_steps: usize,
    /// Cache the integrator's per-step b-coefficients for fast
    /// interpolation (light-time iteration, dense output,
    /// arbitrary-epoch state queries).
    pub cache_integrator_steps: bool,
    /// Origin-switching trajectory splitting. Default enabled — see
    /// [`OriginSwitchingConfig`].
    pub origin_switching: OriginSwitchingConfig,
}

impl Default for AdvancedIntegratorConfig {
    fn default() -> Self {
        Self {
            integrator: IntegratorChoice::default(),
            epsilon: 1e-9,
            dt_initial: None,
            dt_min: None,
            encounter_timescale_divisor: 1000.0,
            max_steps: 10_000_000,
            max_dense_steps: 100_000,
            cache_integrator_steps: false,
            origin_switching: OriginSwitchingConfig::default(),
        }
    }
}

/// Per-trajectory diagnostic outputs.
///
/// All metrics off by default. Enable individual flags to populate the
/// matching diagnostic timeseries on the propagation result.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DiagnosticsConfig {
    /// Emit local sensitivity (∂x/∂x₀) per epoch.
    pub sensitivity: bool,
    /// Emit local nonlinearity diagnostic κ per epoch (requires
    /// second-order uncertainty propagation).
    pub nonlinearity: bool,
    /// Emit local Lyapunov-exponent estimates per epoch.
    pub lyapunov: bool,
    /// Emit keyhole-distance metrics around close approaches.
    pub keyholes: bool,
    /// Emit bifurcation-detection metrics.
    pub bifurcations: bool,
    /// Sample stride for timeseries output. `0` → engine default (1).
    pub sample_stride: usize,
    /// Threshold above which a `HighSensitivity` event is emitted.
    pub sensitivity_threshold: Option<f64>,
    /// Threshold above which a `ChaoticRegion` event is emitted.
    pub lyapunov_threshold: Option<f64>,
    /// Threshold above which a `HighNonlinearity` event is emitted
    /// (requires second-order uncertainty propagation).
    pub nonlinearity_threshold: Option<f64>,
}

/// Event-detection configuration. Controls which event kinds the
/// integrator looks for during propagation, plus body filtering and
/// dense-output cadence around close approaches.
#[derive(Debug, Clone, PartialEq)]
pub struct EventConfig {
    /// Detect close-approach start / end pairs against monitored bodies.
    pub close_approaches: bool,
    /// Detect impact events.
    pub impacts: bool,
    /// Detect atmospheric entry / exit at the Karman line.
    pub atmospheric: bool,
    /// Emit possible-impact records when a close approach has non-zero
    /// linearised impact probability.
    pub possible_impacts: bool,
    /// Detect umbral / penumbral shadow entry / exit.
    pub shadow_events: bool,
    /// Bodies to monitor. Empty = monitor every body the engine has.
    pub body_filter: Vec<Origin>,
    /// Insert dense output points around close approaches.
    pub dense_output: bool,
    /// Cadence (days) of dense output around close approaches.
    pub dense_output_cadence_days: f64,
    /// Master switch for per-substep event detection. Default `true`.
    ///
    /// **The only performance field on this struct.** The five flags
    /// above filter what is *emitted*; the detectors still run on every
    /// accepted integrator substep and still cost what they cost.
    /// Setting this `false` installs no observational detector and skips
    /// the per-substep dispatch entirely — measured at **1.5× faster**
    /// on a 64-orbit, covariance-free, two-epoch Standard-tier batch.
    ///
    /// # What it costs, precisely
    ///
    /// **State accuracy is unchanged.** The trajectory, the STM and the
    /// dense output come back bit-for-bit identical either way, because
    /// detection is observation and not dynamics. Origin-switch zones do
    /// alter the integrated trajectory and are **not** governed here —
    /// they follow [`OriginSwitchingConfig`](super::OriginSwitchingConfig)
    /// alone.
    ///
    /// **Everything the detectors produce is gone**, which is more than
    /// the event list. No events, no close approaches, and therefore no
    /// impact probabilities: the engine computes those from the nominal
    /// close approaches, so an empty CA list empties them too. The five
    /// flags above, [`body_filter`](Self::body_filter) and the enrichment
    /// pass are all moot, and
    /// [`PropagationResult::events`](super::PropagationResult::events)
    /// comes back empty.
    ///
    /// **Two uncertainty methods resolve themselves from that output, and
    /// combining them with detection off is refused** rather than served
    /// degraded — see [`validate`](super::PropagationConfig::validate).
    /// [`UncertaintyMethod::Auto`](super::UncertaintyMethod::Auto) picks
    /// its refinement windows from detected close approaches and gates
    /// its second pass on their impact probabilities, so with neither it
    /// would silently resolve `Linear` everywhere;
    /// [`UncertaintyMethod::Mixture`](super::UncertaintyMethod::Mixture)
    /// splits at those same close approaches, so with none it would never
    /// split. Every other method computes its covariance along the
    /// trajectory and is unaffected.
    pub detection_enabled: bool,
    /// Reference origin for dense encounter-trajectory output. Read only
    /// when [`dense_output`](Self::dense_output) is set. Default
    /// [`DenseOrigin::Bodycentric`].
    pub dense_origin: DenseOrigin,
    /// Which published definition of temporary capture the capture
    /// detector applies. Default [`CaptureCriterion::Population`].
    pub capture_criterion: CaptureCriterion,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            close_approaches: true,
            impacts: true,
            atmospheric: true,
            possible_impacts: true,
            shadow_events: true,
            body_filter: Vec::new(),
            dense_output: false,
            dense_output_cadence_days: 5.0 / 1440.0,
            detection_enabled: true,
            dense_origin: DenseOrigin::Bodycentric,
            capture_criterion: CaptureCriterion::Population,
        }
    }
}

/// Reference origin for the dense encounter trajectory
/// [`EventConfig::dense_output`] produces.
///
/// An origin change here is a pure translation by the body ephemeris,
/// which is deterministic — so the per-point covariance is identical in
/// either choice, and only the frame the arc arrives in differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseOrigin {
    /// Relative to the encounter body (Earth, Moon, …). Precise
    /// body-relative vectors through the encounter, and the natural
    /// frame for a body-centered close-approach view. **Default.**
    #[default]
    Bodycentric,
    /// Relative to the Solar System Barycenter, so the dense arc splices
    /// into a barycentric main trajectory with no client-side
    /// re-centering.
    Barycentric,
}

/// Which published definition of temporary capture the capture detector
/// applies.
///
/// Not a tolerance to tune: the three come from different papers and
/// genuinely disagree about which encounters count as a capture. Pick
/// the one the work is being compared against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureCriterion {
    /// Granvik+ 2012 / Fedorets+ 2018 composite: energy-bound within
    /// 3 Hill radii of the body. The canonical mini-moon *population*
    /// definition. **Default.**
    #[default]
    Population,
    /// Fedorets+ 2020 individual-object criterion: energy-bound within a
    /// tight body-specific scale — ≈ 1 lunar distance for Earth. Reach
    /// for it when comparing against per-object mini-moon
    /// characterization papers, whose reported capture window is
    /// anchored on a closest-approach distance rather than a
    /// Hill-sphere fraction.
    Individual,
    /// Energy-only: \\(\tfrac{1}{2}v_\text{rel}^2 - \mu/r < 0\\), with no
    /// distance gate beyond the close-approach tracking radius. The most
    /// permissive of the three — it fires on weak far-field couplings at
    /// the outer edge of the close-approach zone — and what the engine
    /// did before the criterion became selectable.
    EnergyOnly,
}

/// What to do when the propagated state coincides with an SB441-N16
/// perturber's own SPK ephemeris — the self-perturbation case, where a
/// body is both the thing being propagated and one of the forces acting
/// on it.
///
/// All sixteen SB441-N16 bodies (1 Ceres, 2 Pallas, 4 Vesta, 7 Iris, …)
/// are simultaneously members of the
/// [`ForceModelTier::Standard`] force model and legitimate objects to
/// propagate, so an orbit handed in for one of them sits on top of the
/// ephemeris the force model is reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EphemerisOverlapPolicy {
    /// Skip integration and return the matched perturber's SPK states at
    /// the requested epochs. **Default** — the historical behaviour,
    /// preserved bit for bit.
    ///
    /// The SPK is the authoritative solution for that body, so for the
    /// body itself these states are exact. What it costs, stated plainly:
    /// the caller's initial condition is **discarded** (two inputs within
    /// the detection threshold produce identical output, which makes the
    /// path useless for anything differential — an OD step, a covariance
    /// sweep), and **no trajectory is produced**, so there is no dense
    /// trajectory, no STM, and no sensitivity chain. Ephemeris and radar
    /// generation both read those, and therefore fail outright for an
    /// overlapped body under this policy.
    #[default]
    SubstituteSpk,
    /// Integrate normally, with the matched perturber removed from the
    /// force model for the remainder of that orbit's propagation, and
    /// report the overlap.
    ///
    /// Choose this when you need a trajectory rather than a table of SPK
    /// samples — which is always, for ephemeris generation.
    ExcludeAndIntegrate,
}

/// Top-level propagation configuration.
///
/// Force-model fields at the top, nested `events` / `diagnostics` /
/// `advanced` config bundles. Defaults mirror the engine defaults so
/// constructing one with [`PropagationConfig::default`] reproduces the
/// production hot path.
///
/// The default frame is [`Frame::EclipticJ2000`] (the integration
/// frame). Set `frame = Frame::ICRF` for ICRF output.
#[derive(Debug, Clone, PartialEq)]
pub struct PropagationConfig {
    // ── Force model ─────────────────────────────────────────
    /// Force-model tier preset.
    pub force_model: ForceModelTier,
    /// Bodies to exclude from the perturber set — useful when
    /// propagating an asteroid that the force model would otherwise
    /// include as a perturber (e.g. fitting Eros while excluding
    /// [`Origin::asteroid(433)`](Origin::asteroid)).
    pub excluded_perturbers: Vec<Origin>,

    // ── Uncertainty & STM ──────────────────────────────────
    /// Uncertainty propagation method.
    pub uncertainty_method: UncertaintyMethod,
    /// Force STM-producing integration even without input covariance.
    pub compute_stm: bool,

    // ── Frame, events, diagnostics ─────────────────────────
    /// Output reference frame.
    pub frame: Frame,
    /// Event-detection configuration.
    pub events: EventConfig,
    /// Per-trajectory diagnostic outputs.
    pub diagnostics: DiagnosticsConfig,

    // ── Parallelism ────────────────────────────────────────
    /// Threads for multi-orbit propagation. `None` = use all cores
    /// (Rayon default); `Some(n)` = exactly n threads. The non-zero
    /// type prevents the historical `Some(0)` footgun.
    pub num_threads: Option<std::num::NonZeroUsize>,

    // ── Integrator calibration ─────────────────────────────
    /// Integrator-tuning knobs (rarely touched).
    pub advanced: AdvancedIntegratorConfig,

    // ── Ephemeris-overlap policy ───────────────────────────
    /// What to do when the propagated state coincides with an SB441-N16
    /// perturber's own SPK ephemeris. Default
    /// [`EphemerisOverlapPolicy::SubstituteSpk`] — the historical behaviour. See
    /// [`EphemerisOverlapPolicy`] for what that costs and when to change it.
    ///
    /// Overlap **detection** is on by default and decides whether this
    /// field is consulted at all. The engine suppresses detection
    /// whenever [`excluded_perturbers`](Self::excluded_perturbers) is
    /// non-empty — a caller who named exclusions has already made the
    /// call this policy would make for them — so setting
    /// [`ExcludeAndIntegrate`](EphemerisOverlapPolicy::ExcludeAndIntegrate)
    /// alongside an explicit exclusion does nothing. Use one or the
    /// other. The detection toggle itself is not exposed here.
    pub ephemeris_overlap_policy: EphemerisOverlapPolicy,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            force_model: ForceModelTier::Standard,
            excluded_perturbers: Vec::new(),
            uncertainty_method: UncertaintyMethod::FirstOrder,
            compute_stm: false,
            frame: Frame::EclipticJ2000,
            events: EventConfig::default(),
            diagnostics: DiagnosticsConfig::default(),
            num_threads: None,
            advanced: AdvancedIntegratorConfig::default(),
            ephemeris_overlap_policy: EphemerisOverlapPolicy::default(),
        }
    }
}

/// The refusal for [`UncertaintyMethod::Auto`] under detection off.
///
/// Duplicated deliberately at the C boundary, so a raw C caller — who
/// never goes through this wrapper — gets the same sentence. The
/// duplicate cannot be cross-checked through the FFI (both sides refuse
/// the same pairing, so neither can provoke the other), so each side
/// instead pins its own copy against the written-out sentence in its own
/// test: `refusal_sentences_are_the_published_ones` here and in
/// `empyrean-c`. Editing one without the other fails that side's test.
pub(crate) const DETECTION_OFF_WITH_AUTO: &str = "detection_enabled = false is incompatible with uncertainty_method = Auto: \
     Auto selects its refinement windows from detected close approaches";

/// The refusal for [`UncertaintyMethod::Mixture`] under detection off.
/// Pinned the same way as [`DETECTION_OFF_WITH_AUTO`].
///
/// The method is named for the caller: `Mixture` in Rust,
/// `gaussian_mixture` in Python, tag 5 in C — all one method, so the
/// sentence names it the way the engine and the Python surface do.
pub(crate) const DETECTION_OFF_WITH_MIXTURE: &str = "detection_enabled = false is incompatible with uncertainty_method = GaussianMixture: \
     the mixture splits at detected close approaches";

impl PropagationConfig {
    /// Refuse a configuration whose uncertainty method cannot be served
    /// under the event settings it is paired with.
    ///
    /// One combination is refused, in two variants.
    /// [`EventConfig::detection_enabled`] `= false` installs no
    /// observational detector, so no close approach is found and no
    /// impact probability is computed — and two uncertainty methods
    /// resolve *themselves* from exactly those outputs:
    ///
    /// * [`UncertaintyMethod::Auto`] builds its second-pass refinement
    ///   windows from the first pass's close approaches and gates that
    ///   pass on their linear impact probabilities. With neither it does
    ///   not fail — it resolves `Linear` at every epoch and returns no
    ///   impact probabilities, which is a quieter and worse outcome than
    ///   an error.
    /// * [`UncertaintyMethod::Mixture`] (adaptive Gaussian mixture)
    ///   splits at detected close approaches. With none it never splits,
    ///   and the caller receives a single Gaussian under a name that
    ///   promises a mixture.
    ///
    /// Serving either would be a silent downgrade of a scientific
    /// output, so both are refused by name. Every other method computes
    /// its covariance along the trajectory and is unaffected by
    /// detection, so every other pairing is legal.
    ///
    /// Called by the propagation entry points before any work starts;
    /// the C boundary refuses the same pairing independently, so a raw C
    /// caller is covered too.
    ///
    /// This check belongs to the engine, not to the distribution — a
    /// refusal there would cover every caller including this one. It is
    /// filed for the engine, and this copy should be removed once a
    /// tracking bump makes the engine's own refusal reachable.
    pub fn validate(&self) -> Result<()> {
        if self.events.detection_enabled {
            return Ok(());
        }
        match self.uncertainty_method {
            UncertaintyMethod::Auto { .. } => Err(Error::invalid_input(DETECTION_OFF_WITH_AUTO)),
            UncertaintyMethod::Mixture { .. } => {
                Err(Error::invalid_input(DETECTION_OFF_WITH_MIXTURE))
            }
            _ => Ok(()),
        }
    }

    /// Build the C-ABI representation. Returns the FFI struct plus
    /// keepalive `Vec`s the FFI struct holds raw pointers into. Drop
    /// the keepalives only after the FFI call has returned.
    pub(crate) fn to_ffi_with(&self) -> (empyrean_sys::EmpyreanPropagationConfig, PropConfigKeep) {
        let perturbers: Vec<i32> = self
            .excluded_perturbers
            .iter()
            .map(|o| o.naif_id())
            .collect();
        let body_filter: Vec<i32> = self
            .events
            .body_filter
            .iter()
            .map(|o| o.naif_id())
            .collect();
        let cfg = empyrean_sys::EmpyreanPropagationConfig {
            force_model: self.force_model as i32,
            num_excluded_perturbers: perturbers.len(),
            excluded_perturbers_naif: if perturbers.is_empty() {
                std::ptr::null()
            } else {
                perturbers.as_ptr()
            },
            uncertainty_method: self.uncertainty_method.to_ffi(),
            compute_stm: u8::from(self.compute_stm),
            frame: self.frame as i32,
            events: empyrean_sys::EmpyreanEventConfig {
                close_approaches: u8::from(self.events.close_approaches),
                impacts: u8::from(self.events.impacts),
                atmospheric: u8::from(self.events.atmospheric),
                possible_impacts: u8::from(self.events.possible_impacts),
                shadow_events: u8::from(self.events.shadow_events),
                num_body_filter: body_filter.len(),
                body_filter_naif: if body_filter.is_empty() {
                    std::ptr::null()
                } else {
                    body_filter.as_ptr()
                },
                dense_output: u8::from(self.events.dense_output),
                dense_output_cadence_days: self.events.dense_output_cadence_days,
                // The wrapper always knows what it wants, so it never
                // writes `_DEFAULT` — that value exists for a C caller
                // who zeroed the struct, not for a layer holding a
                // resolved `bool`.
                detection_enabled: if self.events.detection_enabled {
                    empyrean_sys::EMPYREAN_EVENT_DETECTION_ON
                } else {
                    empyrean_sys::EMPYREAN_EVENT_DETECTION_OFF
                },
                dense_origin: match self.events.dense_origin {
                    DenseOrigin::Bodycentric => empyrean_sys::EMPYREAN_DENSE_ORIGIN_BODYCENTRIC,
                    DenseOrigin::Barycentric => empyrean_sys::EMPYREAN_DENSE_ORIGIN_BARYCENTRIC,
                },
                capture_criterion: match self.events.capture_criterion {
                    CaptureCriterion::Population => {
                        empyrean_sys::EMPYREAN_CAPTURE_CRITERION_POPULATION
                    }
                    CaptureCriterion::Individual => {
                        empyrean_sys::EMPYREAN_CAPTURE_CRITERION_INDIVIDUAL
                    }
                    CaptureCriterion::EnergyOnly => {
                        empyrean_sys::EMPYREAN_CAPTURE_CRITERION_ENERGY_ONLY
                    }
                },
            },
            diagnostics: empyrean_sys::EmpyreanDiagnosticsConfig {
                sensitivity: u8::from(self.diagnostics.sensitivity),
                nonlinearity: u8::from(self.diagnostics.nonlinearity),
                lyapunov: u8::from(self.diagnostics.lyapunov),
                keyholes: u8::from(self.diagnostics.keyholes),
                bifurcations: u8::from(self.diagnostics.bifurcations),
                sample_stride: self.diagnostics.sample_stride,
                sensitivity_threshold: self.diagnostics.sensitivity_threshold.unwrap_or(f64::NAN),
                lyapunov_threshold: self.diagnostics.lyapunov_threshold.unwrap_or(f64::NAN),
                nonlinearity_threshold: self.diagnostics.nonlinearity_threshold.unwrap_or(f64::NAN),
            },
            num_threads: self.num_threads.map_or(0, |n| n.get()),
            advanced: empyrean_sys::EmpyreanAdvancedIntegratorConfig {
                // Bindgen exposes the C `#define` macros as `u32`;
                // cast to `i32` to match the struct field type.
                integrator: match self.advanced.integrator {
                    IntegratorChoice::GR15 => empyrean_sys::EMPYREAN_INTEGRATOR_GR15 as i32,
                    IntegratorChoice::DOP853 => empyrean_sys::EMPYREAN_INTEGRATOR_DOP853 as i32,
                },
                epsilon: self.advanced.epsilon,
                dt_initial: self.advanced.dt_initial.unwrap_or(f64::NAN),
                dt_min: self.advanced.dt_min.unwrap_or(f64::NAN),
                encounter_timescale_divisor: self.advanced.encounter_timescale_divisor,
                max_steps: self.advanced.max_steps,
                max_dense_steps: self.advanced.max_dense_steps,
                cache_integrator_steps: u8::from(self.advanced.cache_integrator_steps),
                origin_switching: empyrean_sys::EmpyreanOriginSwitchingConfig {
                    // Tri-state encoding — wrapper bool
                    // maps to explicit ON / OFF tags (not DEFAULT) so
                    // an explicit `enabled = false` in user code
                    // reaches the engine as OFF, not as DEFAULT (which
                    // would resolve to the upstream default = true on
                    // the empyrean-c side and silently override the
                    // user's request).
                    // `as u8` casts: cbindgen emits these constants as
                    // `#define` and bindgen defaults to `u32` for those
                    // (unlike the explicitly-`u8` `EMPYREAN_UNCERTAINTY_*`
                    // constants which bindgen preserves via the typed
                    // const path). The cast is safe — the values are
                    // 0/1/2 and the field is `u8`.
                    enabled: if self.advanced.origin_switching.enabled {
                        empyrean_sys::EMPYREAN_ORIGIN_SWITCHING_ON as u8
                    } else {
                        empyrean_sys::EMPYREAN_ORIGIN_SWITCHING_OFF as u8
                    },
                    hysteresis: self.advanced.origin_switching.hysteresis,
                },
            },
            ephemeris_overlap_policy: match self.ephemeris_overlap_policy {
                EphemerisOverlapPolicy::SubstituteSpk => {
                    empyrean_sys::EMPYREAN_EPHEMERIS_OVERLAP_POLICY_SUBSTITUTE_SPK
                }
                EphemerisOverlapPolicy::ExcludeAndIntegrate => {
                    empyrean_sys::EMPYREAN_EPHEMERIS_OVERLAP_POLICY_EXCLUDE_AND_INTEGRATE
                }
            },
        };
        (
            cfg,
            PropConfigKeep {
                _perturbers: perturbers,
                _body_filter: body_filter,
            },
        )
    }
}

/// Keepalive bag for [`PropagationConfig::to_ffi_with`]'s heap arrays.
/// Hold this across the FFI call; drop after the call returns.
pub(crate) struct PropConfigKeep {
    _perturbers: Vec<i32>,
    _body_filter: Vec<i32>,
}

/// The [`EphemerisOverlapPolicy`] surface. The wrapper enum, the ABI integers it
/// marshals to, and the engine flag they select all ship together, so
/// these pin the mapping and the default.
#[cfg(test)]
mod overlap_policy_tests {
    use super::*;

    /// The default must be the historical behaviour, and must marshal as
    /// the ABI's zero — a config nobody touched has to keep meaning what
    /// it meant before the field existed.
    #[test]
    fn the_default_marshals_as_the_abi_zero() {
        assert_eq!(
            EphemerisOverlapPolicy::default(),
            EphemerisOverlapPolicy::SubstituteSpk
        );
        let (ffi, _keep) = PropagationConfig::default().to_ffi_with();
        assert_eq!(
            ffi.ephemeris_overlap_policy,
            empyrean_sys::EMPYREAN_EPHEMERIS_OVERLAP_POLICY_SUBSTITUTE_SPK
        );
        assert_eq!(ffi.ephemeris_overlap_policy, 0);
    }

    /// Both variants reach the wire as their contract values.
    #[test]
    fn both_variants_marshal_to_their_contract_values() {
        for (policy, want) in [
            (
                EphemerisOverlapPolicy::SubstituteSpk,
                empyrean_sys::EMPYREAN_EPHEMERIS_OVERLAP_POLICY_SUBSTITUTE_SPK,
            ),
            (
                EphemerisOverlapPolicy::ExcludeAndIntegrate,
                empyrean_sys::EMPYREAN_EPHEMERIS_OVERLAP_POLICY_EXCLUDE_AND_INTEGRATE,
            ),
        ] {
            let cfg = PropagationConfig {
                ephemeris_overlap_policy: policy,
                ..PropagationConfig::default()
            };
            let (ffi, _keep) = cfg.to_ffi_with();
            assert_eq!(
                ffi.ephemeris_overlap_policy, want,
                "{policy:?} marshals to {want}"
            );
        }
    }
}

/// Origin-switching defaults, pinned at the wrapper.
///
/// The default flipped to `enabled = true` upstream but five doc sites
/// kept saying "opt-in" — including one eleven lines below the `Default`
/// impl in this file. Nothing asserted the value, so nothing caught the
/// drift. This is that assertion.
#[cfg(test)]
mod origin_switching_default_tests {
    use super::*;

    /// Trajectory splitting is ON by default, both on its own config and
    /// as reached through [`AdvancedIntegratorConfig`], and it marshals
    /// to the ABI's explicit ON rather than the DEFAULT sentinel.
    #[test]
    fn origin_switching_is_enabled_by_default() {
        assert!(OriginSwitchingConfig::default().enabled);
        assert!(AdvancedIntegratorConfig::default().origin_switching.enabled);
        let (ffi, _keep) = PropagationConfig::default().to_ffi_with();
        assert_eq!(
            ffi.advanced.origin_switching.enabled,
            empyrean_sys::EMPYREAN_ORIGIN_SWITCHING_ON as u8,
            "the wrapper marshals its resolved default explicitly"
        );
    }
}

#[cfg(test)]
mod detection_refusal_tests {
    use super::*;

    /// Both refusal sentences are published contract — a caller greps
    /// for them, and the C boundary carries a byte-identical copy it
    /// cannot be cross-checked against. Pin them here so editing this
    /// side without the other fails.
    #[test]
    fn refusal_sentences_are_the_published_ones() {
        assert_eq!(
            DETECTION_OFF_WITH_AUTO,
            "detection_enabled = false is incompatible with uncertainty_method = Auto: Auto selects its refinement windows from detected close approaches"
        );
        assert_eq!(
            DETECTION_OFF_WITH_MIXTURE,
            "detection_enabled = false is incompatible with uncertainty_method = GaussianMixture: the mixture splits at detected close approaches"
        );
    }

    /// The refusal fires on the pairing, not on either half alone.
    #[test]
    fn only_the_pairing_is_refused() {
        let paired = |method: UncertaintyMethod, detection_enabled: bool| PropagationConfig {
            uncertainty_method: method,
            events: EventConfig {
                detection_enabled,
                ..EventConfig::default()
            },
            ..PropagationConfig::default()
        };

        assert_eq!(
            paired(UncertaintyMethod::auto(), false)
                .validate()
                .expect_err("Auto + detection off is refused")
                .message,
            DETECTION_OFF_WITH_AUTO
        );
        assert_eq!(
            paired(UncertaintyMethod::gaussian_mixture(), false)
                .validate()
                .expect_err("Mixture + detection off is refused")
                .message,
            DETECTION_OFF_WITH_MIXTURE
        );

        // Detection on serves both, and detection off serves every
        // method that does not read detector output. Without these the
        // test above would pass against a blanket refusal.
        for method in [
            UncertaintyMethod::auto(),
            UncertaintyMethod::gaussian_mixture(),
        ] {
            paired(method.clone(), true)
                .validate()
                .unwrap_or_else(|e| panic!("{method:?} with detection on is legal: {e}"));
        }
        for method in [
            UncertaintyMethod::FirstOrder,
            UncertaintyMethod::SecondOrder,
            UncertaintyMethod::sigma_point(),
            UncertaintyMethod::monte_carlo(64),
        ] {
            paired(method.clone(), false)
                .validate()
                .unwrap_or_else(|e| panic!("{method:?} with detection off is legal: {e}"));
        }
    }
}

/// The wrapper's own half of the event-config marshalling: enum → C
/// constant.
///
/// The C side is well covered the other way round — its
/// `build_event_config_from_c` has tests for the zero-init default,
/// every explicit value, and an off-ladder refusal. Those check
/// *constant → engine value*. Nothing checked *wrapper enum →
/// constant*, and a swapped arm there compiles, passes every one of
/// those tests, and hands a caller the wrong reference origin for its
/// dense encounter arc with no signal anywhere.
#[cfg(test)]
mod event_config_ffi_tests {
    use super::*;

    fn marshalled(events: EventConfig) -> empyrean_sys::EmpyreanEventConfig {
        let (ffi, _keep) = PropagationConfig {
            events,
            ..PropagationConfig::default()
        }
        .to_ffi_with();
        ffi.events
    }

    /// The master switch marshals to the explicit rungs, never to
    /// `_DEFAULT`: this layer holds a resolved `bool` and has no "unset"
    /// state to pass on.
    #[test]
    fn detection_enabled_marshals_to_the_explicit_rungs() {
        for (enabled, want) in [
            (true, empyrean_sys::EMPYREAN_EVENT_DETECTION_ON),
            (false, empyrean_sys::EMPYREAN_EVENT_DETECTION_OFF),
        ] {
            let ffi = marshalled(EventConfig {
                detection_enabled: enabled,
                ..EventConfig::default()
            });
            assert_eq!(ffi.detection_enabled, want, "detection_enabled = {enabled}");
            assert_ne!(
                ffi.detection_enabled,
                empyrean_sys::EMPYREAN_EVENT_DETECTION_DEFAULT,
                "the wrapper never writes the DEFAULT rung"
            );
        }
    }

    /// Every `DenseOrigin` variant round-trips to its own constant. The
    /// `assert_ne` against the sibling is what catches a swapped arm —
    /// equality alone would pass if both arms wrote the same value.
    #[test]
    fn every_dense_origin_marshals_to_its_own_constant() {
        let bodycentric = marshalled(EventConfig {
            dense_origin: DenseOrigin::Bodycentric,
            ..EventConfig::default()
        })
        .dense_origin;
        let barycentric = marshalled(EventConfig {
            dense_origin: DenseOrigin::Barycentric,
            ..EventConfig::default()
        })
        .dense_origin;

        assert_eq!(bodycentric, empyrean_sys::EMPYREAN_DENSE_ORIGIN_BODYCENTRIC);
        assert_eq!(barycentric, empyrean_sys::EMPYREAN_DENSE_ORIGIN_BARYCENTRIC);
        assert_ne!(bodycentric, barycentric, "the two arms must not be swapped");
    }

    /// Every `CaptureCriterion` variant round-trips to its own constant.
    /// These select among *published definitions of capture*, so a
    /// swapped arm silently answers a different paper's question.
    #[test]
    fn every_capture_criterion_marshals_to_its_own_constant() {
        let marshal = |criterion| {
            marshalled(EventConfig {
                capture_criterion: criterion,
                ..EventConfig::default()
            })
            .capture_criterion
        };
        let population = marshal(CaptureCriterion::Population);
        let individual = marshal(CaptureCriterion::Individual);
        let energy_only = marshal(CaptureCriterion::EnergyOnly);

        assert_eq!(
            population,
            empyrean_sys::EMPYREAN_CAPTURE_CRITERION_POPULATION
        );
        assert_eq!(
            individual,
            empyrean_sys::EMPYREAN_CAPTURE_CRITERION_INDIVIDUAL
        );
        assert_eq!(
            energy_only,
            empyrean_sys::EMPYREAN_CAPTURE_CRITERION_ENERGY_ONLY
        );
        assert_eq!(
            [population, individual, energy_only]
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3,
            "the three arms must be distinct"
        );
    }

    /// The wrapper's defaults marshal to the engine's defaults, so a
    /// caller who touches none of the three gets what it always did.
    #[test]
    fn the_defaults_marshal_to_the_engine_defaults() {
        let ffi = marshalled(EventConfig::default());
        assert_eq!(
            ffi.detection_enabled,
            empyrean_sys::EMPYREAN_EVENT_DETECTION_ON
        );
        assert_eq!(
            ffi.dense_origin,
            empyrean_sys::EMPYREAN_DENSE_ORIGIN_BODYCENTRIC
        );
        assert_eq!(
            ffi.capture_criterion,
            empyrean_sys::EMPYREAN_CAPTURE_CRITERION_POPULATION
        );
    }
}
