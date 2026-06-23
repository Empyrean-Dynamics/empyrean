//! Per-station nuisance-parameter configuration. Currently only RA/Dec
//! biases are exposed; future surfaces (e.g. timing) plug in here.

/// Per-station RA/Dec bias-fit configuration.
///
/// Schur-eliminated nuisance parameters that absorb per-station
/// pointing offsets, fit alongside the orbit. Defaults target modern
/// survey arcs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StationRaDecConfig {
    /// Default 1-sigma prior (arcsec).
    pub sigma_prior_arcsec: f64,
    /// Minimum observations per station required to allocate a bias
    /// parameter for that station.
    pub min_obs_per_station: usize,
}

impl Default for StationRaDecConfig {
    fn default() -> Self {
        // The "modern arc" defaults: 0.3″ sigma, 5-obs minimum.
        Self {
            sigma_prior_arcsec: 0.3,
            min_obs_per_station: 5,
        }
    }
}
