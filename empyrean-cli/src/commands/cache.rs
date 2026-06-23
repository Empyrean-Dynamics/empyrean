use std::path::PathBuf;

use anyhow::Result;

#[derive(clap::Subcommand)]
pub enum CacheCommand {
    /// Show cache location and size.
    Info,
    /// Delete all cached API responses.
    Clear,
}

fn cache_root() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".empyrean").join("cache")
}

fn dir_size(path: &std::path::Path) -> u64 {
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| {
            let meta = e.metadata().ok();
            if e.path().is_dir() {
                dir_size(&e.path())
            } else {
                meta.map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
}

fn fmt_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn run(cmd: CacheCommand) -> Result<()> {
    let root = cache_root();

    match cmd {
        CacheCommand::Info => {
            if !root.exists() {
                eprintln!("Cache directory: {} (empty)", root.display());
                return Ok(());
            }
            eprintln!("Cache directory: {}", root.display());
            for entry in std::fs::read_dir(&root)?.flatten() {
                if entry.path().is_dir() {
                    let name = entry.file_name();
                    let n_files = std::fs::read_dir(entry.path())
                        .map(|rd| rd.count())
                        .unwrap_or(0);
                    let size = dir_size(&entry.path());
                    eprintln!(
                        "  {}: {} files ({})",
                        name.to_string_lossy(),
                        n_files,
                        fmt_size(size),
                    );
                }
            }
            let total = dir_size(&root);
            eprintln!("  Total: {}", fmt_size(total));
        }
        CacheCommand::Clear => {
            if root.exists() {
                let size = dir_size(&root);
                std::fs::remove_dir_all(&root)?;
                eprintln!("Cleared cache: {} ({})", root.display(), fmt_size(size));
            } else {
                eprintln!("Cache already empty: {}", root.display());
            }
        }
    }

    Ok(())
}
