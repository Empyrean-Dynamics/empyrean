//! Propagation configuration: force model, uncertainty method,
//! integrator backend, event-detection toggles, and the top-level
//! [`PropagationConfig`] passed to [`Context::propagate`](super::Context::propagate).

use crate::coordinate::{Frame, Origin};

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
    /// Origin-switching trajectory splitting. Default disabled.
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
        }
    }
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
        }
    }
}

impl PropagationConfig {
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
