//! Outlier-rejection strategy for orbit determination.
//!
//! Two strategies ship: information-loss-weighted adaptive rejection
//! ([`RejectionKind::Adaptive`], the production default) and the
//! Carpino–Milani–Chesley (2003) χ²-with-hysteresis scheme
//! ([`RejectionKind::CMC2003`], for OrbFit / NEODyS interop).

/// Outlier-rejection strategy selector. The variant chosen determines
/// which fields of [`RejectionConfig`] are read; ignored fields keep
/// their defaults.
///
/// - [`Adaptive`](Self::Adaptive) reads `chi2_base` / `lambda` /
///   `max_threshold` and runs information-loss-weighted rejection.
/// - [`CMC2003`](Self::CMC2003) reads `chi2_rej` / `chi2_rec` and runs
///   χ²-with-hysteresis rejection. Recommended when reproducing
///   classical NEO orbit-fitting workflows or when interoperating with
///   OrbFit / NEODyS expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RejectionKind {
    /// Information-loss-weighted adaptive rejection (default).
    #[default]
    Adaptive,
    /// Carpino–Milani–Chesley (2003) χ²-with-hysteresis.
    CMC2003,
}

/// Outlier-rejection configuration. The active fields are determined
/// by [`kind`](Self::kind).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RejectionConfig {
    /// `true` → run rejection (default), `false` → skip it.
    pub enabled: bool,
    /// Strategy selector. Default [`RejectionKind::Adaptive`].
    pub kind: RejectionKind,
    /// Adaptive rejection χ²ᵦₐₛₑ. Used when
    /// `kind == RejectionKind::Adaptive`.
    pub chi2_base: f64,
    /// Adaptive rejection λ. Used when
    /// `kind == RejectionKind::Adaptive`.
    pub lambda: f64,
    /// Adaptive rejection effective-threshold cap. Used when
    /// `kind == RejectionKind::Adaptive`.
    pub max_threshold: f64,
    /// CMC2003 upper threshold (reject when χ² > chi2_rej). Used when
    /// `kind == RejectionKind::CMC2003`.
    pub chi2_rej: f64,
    /// CMC2003 lower threshold (recover when χ² < chi2_rec). Used when
    /// `kind == RejectionKind::CMC2003`. Must be strictly less than
    /// [`chi2_rej`](Self::chi2_rej) for hysteresis to break cycles.
    pub chi2_rec: f64,
    /// Maximum rejection-refit passes.
    pub max_passes: u32,
}

impl Default for RejectionConfig {
    fn default() -> Self {
        // Adaptive rejection enabled with the production-default χ²ᵦₐₛₑ
        // and λ. CMC2003 fields carry their own defaults so a caller
        // who flips `kind` without touching anything else gets a valid
        // config.
        Self {
            enabled: true,
            kind: RejectionKind::Adaptive,
            chi2_base: 9.21,
            lambda: 1.0,
            max_threshold: 100.0,
            chi2_rej: 8.0,
            chi2_rec: 7.0,
            max_passes: 4,
        }
    }
}
