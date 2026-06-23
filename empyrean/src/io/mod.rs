//! File I/O for orbits, ephemerides, events, and residuals.
//!
//! Three file formats are supported per data type:
//!
//! - **Parquet** — round-trips covariance and is the canonical batch
//!   format. Recommended for data that flows back into the propagator
//!   (parquet's nullable covariance columns are the only way to
//!   round-trip uncertainty).
//! - **JSON** — human-readable, retains structure (covariance arrays
//!   when present), good for small batches and config-style use.
//! - **CSV** — flat row-per-record, drops covariance, easiest to feed
//!   into spreadsheets or shell pipelines.
//!
//! Orbits support read + write in all three formats; ephemeris,
//! events, and residuals are write-only — the propagator / OD pipeline
//! is the canonical producer.
//!
//! All conversion happens at the FFI boundary; the wrapper returns
//! Rust-friendly types ([`OrbitBatch`], [`EphemerisEntry`](crate::EphemerisEntry),
//! …) instead of raw C arrays.

mod ephemeris;
mod events;
mod orbits;
mod residuals;

pub use ephemeris::{write_ephemeris_csv, write_ephemeris_json, write_ephemeris_parquet};
pub use events::{write_events_csv, write_events_json, write_events_parquet};
pub use orbits::{
    OrbitBatch, read_orbits_csv, read_orbits_json, read_orbits_parquet, write_orbits_csv,
    write_orbits_json, write_orbits_parquet,
};
pub use residuals::{write_residuals_csv, write_residuals_json, write_residuals_parquet};

use std::ffi::CString;
use std::path::Path;

use crate::error::{Error, Result};

/// Path → CString helper shared across the I/O sub-modules.
pub(super) fn path_to_cstring(path: &Path) -> Result<CString> {
    let bytes = path
        .to_str()
        .ok_or_else(|| Error::invalid_input("path is not valid UTF-8"))?
        .as_bytes();
    CString::new(bytes).map_err(|_| Error::invalid_input("path contains a NUL byte"))
}
