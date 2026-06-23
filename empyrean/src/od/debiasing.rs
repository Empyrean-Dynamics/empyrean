//! Astrometric catalog-bias correction (EFCC2020).
//!
//! Applied during fit residual computation to remove pre-Gaia catalog
//! systematic biases before they reach the χ² accumulator.

/// Healpix resolution of a debiasing table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebiasingResolution {
    /// NSIDE = 64, ~35 MB. Production default.
    #[default]
    Standard,
    /// NSIDE = 256, ~567 MB.
    Hires,
}

/// Catalog-bias-correction configuration.
///
/// Default = `enabled` with `bias_dat_path = None`, which uses the
/// data-manager default lookup path (`~/.empyrean/data/bias.dat`) at
/// standard resolution. Set `enabled = false` to disable catalog
/// debiasing entirely.
#[derive(Debug, Clone, PartialEq)]
pub struct DebiasingConfig {
    /// `true` → on (default), `false` → no catalog debiasing.
    pub enabled: bool,
    /// Healpix resolution. Default Standard.
    pub resolution: DebiasingResolution,
    /// Optional path to bias.dat. `None` = use the data-manager
    /// default location.
    pub bias_dat_path: Option<std::path::PathBuf>,
}

impl Default for DebiasingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            resolution: DebiasingResolution::Standard,
            bias_dat_path: None,
        }
    }
}
