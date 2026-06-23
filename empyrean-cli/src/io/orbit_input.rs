use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use empyrean::OrbitBatch;

/// Resolve an orbit input source into an [`OrbitBatch`].
///
/// Exactly one of `object_ids` (SBDB lookup) or `input_path` (file
/// loaded by extension: `.parquet`, `.json`, `.csv`) must be provided.
pub fn load_orbits(
    object_ids: &Option<Vec<String>>,
    input_path: &Option<PathBuf>,
) -> Result<OrbitBatch> {
    match (object_ids, input_path) {
        (Some(names), None) => {
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let batch =
                empyrean::query_sbdb(&name_refs, None).context("failed to query JPL SBDB")?;
            eprintln!("Queried SBDB: {} object(s)", batch.len());
            Ok(batch)
        }
        (None, Some(path)) => {
            let batch = read_by_extension(path)?;
            eprintln!("Loaded {} orbit(s) from {}", batch.len(), path.display());
            Ok(batch)
        }
        (None, None) => bail!("specify --object-id or --input"),
        (Some(_), Some(_)) => bail!("--object-id and --input are mutually exclusive"),
    }
}

fn read_by_extension(path: &Path) -> Result<OrbitBatch> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("parquet") => empyrean::read_orbits_parquet(path)
            .with_context(|| format!("failed to read {}", path.display())),
        Some("json") => empyrean::read_orbits_json(path)
            .with_context(|| format!("failed to read {}", path.display())),
        Some("csv") => empyrean::read_orbits_csv(path)
            .with_context(|| format!("failed to read {}", path.display())),
        Some(other) => bail!(
            "unsupported orbit-file extension '.{}'; expected .parquet, .json, or .csv",
            other
        ),
        None => bail!("input path has no extension; expected .parquet, .json, or .csv"),
    }
}
