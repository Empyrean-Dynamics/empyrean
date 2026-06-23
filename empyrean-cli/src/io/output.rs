//! Output writers for CLI commands.
//!
//! All writes go through the empyrean wrapper, which delegates to the
//! C ABI's I/O surface. The CLI never reaches into a private crate
//! directly.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use empyrean::{
    EphemerisEntry, Event, ObservationResidual, OrbitBatch, write_ephemeris_csv,
    write_ephemeris_json, write_ephemeris_parquet, write_events_csv, write_events_json,
    write_events_parquet, write_orbits_csv, write_orbits_json, write_orbits_parquet,
    write_residuals_csv, write_residuals_json, write_residuals_parquet,
};

/// Output file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Parquet,
    Json,
    Csv,
}

impl OutputFormat {
    fn extension(self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::Json => "json",
            Self::Csv => "csv",
        }
    }
}

fn out_path(dir: &Path, stem: &str, fmt: OutputFormat) -> PathBuf {
    dir.join(format!("{stem}.{}", fmt.extension()))
}

/// Write an orbit batch as `<dir>/<stem>.<ext>` in the chosen format.
pub fn write_orbits(dir: &Path, stem: &str, batch: &OrbitBatch, fmt: OutputFormat) -> Result<()> {
    std::fs::create_dir_all(dir).context("failed to create output directory")?;
    let path = out_path(dir, stem, fmt);
    match fmt {
        OutputFormat::Parquet => write_orbits_parquet(&path, batch),
        OutputFormat::Json => write_orbits_json(&path, batch),
        OutputFormat::Csv => write_orbits_csv(&path, batch),
    }
    .with_context(|| format!("failed to write {}", path.display()))?;
    eprintln!("  {} ({} rows)", path.display(), batch.len());
    Ok(())
}

/// Write a vector of events.
pub fn write_events(dir: &Path, stem: &str, events: &[Event], fmt: OutputFormat) -> Result<()> {
    std::fs::create_dir_all(dir).context("failed to create output directory")?;
    let path = out_path(dir, stem, fmt);
    match fmt {
        OutputFormat::Parquet => write_events_parquet(&path, events),
        OutputFormat::Json => write_events_json(&path, events),
        OutputFormat::Csv => write_events_csv(&path, events),
    }
    .with_context(|| format!("failed to write {}", path.display()))?;
    eprintln!("  {} ({} rows)", path.display(), events.len());
    Ok(())
}

/// Write a vector of ephemeris entries.
pub fn write_ephemeris(
    dir: &Path,
    stem: &str,
    entries: &[EphemerisEntry],
    fmt: OutputFormat,
) -> Result<()> {
    std::fs::create_dir_all(dir).context("failed to create output directory")?;
    let path = out_path(dir, stem, fmt);
    match fmt {
        OutputFormat::Parquet => write_ephemeris_parquet(&path, entries),
        OutputFormat::Json => write_ephemeris_json(&path, entries),
        OutputFormat::Csv => write_ephemeris_csv(&path, entries),
    }
    .with_context(|| format!("failed to write {}", path.display()))?;
    eprintln!("  {} ({} rows)", path.display(), entries.len());
    Ok(())
}

/// Write OD residuals.
pub fn write_residuals(
    path: &Path,
    residuals: &[ObservationResidual],
    fmt: OutputFormat,
) -> Result<()> {
    match fmt {
        OutputFormat::Parquet => write_residuals_parquet(path, residuals),
        OutputFormat::Json => write_residuals_json(path, residuals),
        OutputFormat::Csv => write_residuals_csv(path, residuals),
    }
    .with_context(|| format!("failed to write {}", path.display()))
}
