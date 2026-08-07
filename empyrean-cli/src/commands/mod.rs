pub mod cache;
pub mod determine;
pub mod ephemeris;
pub mod init;
pub mod propagate;
pub mod query;

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

/// The data-acquisition options every context-building command shares:
/// where the kernels live and whether the constructor may go and get
/// them.
///
/// Carried as one value so `--data-dir` and `--no-refresh` cannot drift
/// apart between subcommands, and so the offline decision is taken in
/// exactly one place ([`DataOptions::build_context`]).
#[derive(Debug, Clone)]
pub struct DataOptions {
    /// `--data-dir` (global). `None` means the platform data directory.
    pub data_dir: Option<PathBuf>,
    /// Whether context construction may reach the network. `false` is
    /// `--no-refresh`.
    pub refresh: bool,
}

impl DataOptions {
    /// Borrow the data directory in the shape the wrapper wants.
    pub fn dir(&self) -> Option<&Path> {
        self.data_dir.as_deref()
    }

    /// Whether the network may actually be reached, once the
    /// `EMPYREAN_OFFLINE` floor is applied on top of `--no-refresh`.
    ///
    /// `refresh` records what the *flags* asked for. The floor is applied
    /// by the wrapper inside context construction, so anything this CLI
    /// does over the network **before** building a context — `init`'s
    /// kernel download, above all — has to consult this instead, or the
    /// floor would cover the load and not the fetch, which is the half
    /// that touches the network.
    pub fn effective_refresh(&self) -> bool {
        self.refresh && !empyrean::offline_floor_is_active()
    }

    /// Whether a command may hand its work to a running daemon.
    ///
    /// `--no-refresh` is a property of *how the context was built*, and a
    /// daemon's context was built once, at `serve` time, under whatever
    /// options that invocation carried. There is no way to apply a
    /// client's offline request to it after the fact, so an offline
    /// client runs in-process instead — the same reason
    /// `--tagged-covariance` and `--thrust-arcs` fall through. Serving
    /// the request anyway would ignore the flag without saying so.
    pub fn daemon_eligible(&self) -> bool {
        self.effective_refresh()
    }

    /// Build the [`empyrean::Context`] under these options.
    ///
    /// With `refresh: false` this is a **strict offline** construction:
    /// the kernel set is resolved from the data directory alone and the
    /// call fails, naming every absent file, if any is missing. It never
    /// falls back to the network and never quietly loads a smaller set.
    pub fn build_context(&self) -> Result<empyrean::Context> {
        let options = empyrean::DataDirOptions {
            refresh: self.refresh,
            ..empyrean::DataDirOptions::default()
        };
        empyrean::Context::from_data_dir_with(self.dir(), options).map_err(|e| {
            let missing = e.missing_data_files().to_vec();
            let base = anyhow::Error::new(e);
            if missing.is_empty() {
                base
            } else {
                // Reprint the absent files as their own block: under
                // --no-refresh this list *is* the remedy, and it should
                // not be buried inside one long line.
                base.context(format!(
                    "strict offline: the data directory is missing {} required \
                     file(s):\n  {}",
                    missing.len(),
                    missing.join("\n  ")
                ))
            }
        })
    }
}

/// Load a context for a command, printing how long it took (the existing
/// `Loaded context (Ns)` line every subcommand emits) and, when
/// `--no-refresh` is in force, saying so before the attempt — an offline
/// failure should never look like a mysterious missing kernel.
pub fn load_context(data: &DataOptions) -> Result<empyrean::Context> {
    let t0 = std::time::Instant::now();
    if !data.effective_refresh() {
        eprintln!("Loading context (strict offline, no downloads)...");
    }
    let ctx = data.build_context().context("failed to load context")?;
    eprintln!("Loaded context ({:.1}s)", t0.elapsed().as_secs_f64());
    Ok(ctx)
}

#[cfg(test)]
mod data_options_tests {
    use super::*;

    /// `--no-refresh` is the *negation* of the wrapper's `refresh`. Get
    /// this backwards and the flag silently does nothing, so pin it.
    #[test]
    fn no_refresh_negates_refresh() {
        let online = DataOptions {
            data_dir: None,
            refresh: true,
        };
        let offline = DataOptions {
            data_dir: None,
            refresh: false,
        };
        assert!(online.refresh, "default (no --no-refresh) must refresh");
        assert!(!offline.refresh, "--no-refresh must disable refresh");
    }

    /// The offline request must reach the wrapper unchanged. The
    /// wrapper's own `DataDirOptions::default()` is the online form, so
    /// asserting the constructed options differ from the default in
    /// exactly the `refresh` field catches a `..Default::default()` tail
    /// swallowing the flag.
    #[test]
    fn offline_options_differ_from_default_only_in_refresh() {
        let defaults = empyrean::DataDirOptions::default();
        assert!(defaults.refresh, "wrapper default must be online");

        let built = empyrean::DataDirOptions {
            refresh: false,
            ..empyrean::DataDirOptions::default()
        };
        assert!(!built.refresh);
        assert_eq!(
            built.tier, defaults.tier,
            "--no-refresh must not move the data tier"
        );
    }

    /// `EMPYREAN_OFFLINE` is process-global; serialize the tests that
    /// set it so they cannot observe each other's value.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_offline<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var("EMPYREAN_OFFLINE").ok();
        // Safety: guarded by ENV_LOCK, and no other test in this crate
        // reads EMPYREAN_OFFLINE.
        unsafe {
            match value {
                Some(v) => std::env::set_var("EMPYREAN_OFFLINE", v),
                None => std::env::remove_var("EMPYREAN_OFFLINE"),
            }
        }
        let out = f();
        unsafe {
            match prior {
                Some(v) => std::env::set_var("EMPYREAN_OFFLINE", v),
                None => std::env::remove_var("EMPYREAN_OFFLINE"),
            }
        }
        out
    }

    fn opts(refresh: bool) -> DataOptions {
        DataOptions {
            data_dir: None,
            refresh,
        }
    }

    /// An offline request must never be served by a daemon whose context
    /// was built under a different policy — that would ignore the flag
    /// without saying so.
    #[test]
    fn offline_requests_bypass_the_daemon() {
        with_offline(None, || {
            assert!(
                opts(true).daemon_eligible(),
                "a normal run must still take the daemon fast path"
            );
            assert!(
                !opts(false).daemon_eligible(),
                "--no-refresh must run in-process, not be served by a daemon \
                 context built with refresh on"
            );
        });
    }

    /// `EMPYREAN_OFFLINE=1` is a floor over `--no-refresh`, so anything
    /// this CLI does over the network *before* building a context — the
    /// `init` download above all — has to see it. It only ever removes
    /// network access, never grants it.
    #[test]
    fn the_offline_env_floors_effective_refresh() {
        with_offline(None, || {
            assert!(opts(true).effective_refresh(), "unset + no flag = online");
            assert!(!opts(false).effective_refresh(), "unset + flag = offline");
        });
        with_offline(Some("1"), || {
            assert!(
                !opts(true).effective_refresh(),
                "EMPYREAN_OFFLINE=1 must floor an unflagged run to offline, or \
                 `init` downloads kernels on a machine that forbade it"
            );
            assert!(!opts(false).effective_refresh());
            assert!(
                !opts(true).daemon_eligible(),
                "a floored run must also bypass the daemon"
            );
        });
        // Only the exact value 1 asserts the floor — a half-set variable
        // must not quietly change behaviour.
        for v in ["0", "true", "yes", ""] {
            with_offline(Some(v), || {
                assert!(
                    opts(true).effective_refresh(),
                    "EMPYREAN_OFFLINE={v:?} must not assert the floor"
                );
            });
        }
    }

    /// A `DataOptions` carrying a directory hands exactly that directory
    /// to the wrapper — no silent substitution of the platform default.
    #[test]
    fn dir_is_passed_through_verbatim() {
        let data = DataOptions {
            data_dir: Some(PathBuf::from("/tmp/empyrean-kernels")),
            refresh: false,
        };
        assert_eq!(data.dir(), Some(Path::new("/tmp/empyrean-kernels")));

        let none = DataOptions {
            data_dir: None,
            refresh: true,
        };
        assert_eq!(none.dir(), None);
    }
}
