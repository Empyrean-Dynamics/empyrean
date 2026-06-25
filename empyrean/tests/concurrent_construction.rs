//! Regression: native context construction must be safe to call concurrently.
//!
//! The engine's first-init provisioning does writable-cache file I/O. With an
//! incomplete data directory, several `from_data_dir` calls at once raced to
//! re-provision the missing file and ~all failed with a path-less
//! `I/O error: ... No such file or directory (os error 2)`. The wrapper now
//! serializes native *construction*; concurrent *use* stays unserialized.
//!
//! This test reproduces the race cheaply: it seeds a temp dir with every kernel
//! symlinked from the populated default dir EXCEPT `bias.dat`, then races eight
//! constructions to re-fetch that one small file.

#![cfg(unix)]

use std::path::PathBuf;
use std::thread;

/// Seed a temp data dir = the default dir minus `bias.dat`. Returns `None`
/// (test skips) if the default dir can't be provisioned (e.g. no network).
fn seed_temp_missing_bias() -> Option<PathBuf> {
    // Provision the default dir once, serially, as the symlink source.
    let src = empyrean::download_data(None).ok()?;
    if !src.join("de440.bsp").exists() {
        return None;
    }

    let tmp = std::env::temp_dir().join(format!("empyrean-conc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).ok()?;
    for entry in std::fs::read_dir(&src).ok()?.flatten() {
        let name = entry.file_name();
        // Omit bias.dat (+ its .meta.json) so it must be re-fetched concurrently.
        if name.to_string_lossy().starts_with("bias.dat") {
            continue;
        }
        let _ = std::os::unix::fs::symlink(entry.path(), tmp.join(&name));
    }
    Some(tmp)
}

#[test]
fn concurrent_construction_reprovisions_without_racing() {
    let Some(tmp) = seed_temp_missing_bias() else {
        eprintln!("skipping concurrent-construction test: default data dir unavailable");
        return;
    };

    const THREADS: usize = 8;
    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let dir = tmp.clone();
            thread::spawn(move || empyrean::Context::from_data_dir(Some(&dir)).map(|_| ()))
        })
        .collect();

    let mut failures = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        match h.join().expect("construction thread panicked") {
            Ok(()) => {}
            Err(e) => failures.push(format!("thread {i}: {e}")),
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);

    assert!(
        failures.is_empty(),
        "{}/{THREADS} concurrent constructions raced:\n{}",
        failures.len(),
        failures.join("\n"),
    );
}
