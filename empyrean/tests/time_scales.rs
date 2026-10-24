//! Distribution time-scale contract (bead empyrean-8ctt5).
//!
//! An output epoch comes back in the scale the caller requested: a UTC
//! request yields a UTC-tagged, bit-for-bit echoed output. (The TDB tag on
//! an engine-found instant such as a kernel coverage window is enforced by
//! construction (see `EarthOrientationCoverage` in `context.rs`, built
//! with `Epoch::from_mjd_tdb`) and by the Python contract suite; a Rust
//! integration test for it would have to hard-code a BPC file path, which
//! the coverage reader takes instead of a `Context`.)

use std::path::PathBuf;

use empyrean::{Context, Epoch, Frame, Origin, TimeScale};

/// Resolve a usable data dir: `EMPYREAN_DATA_DIR` (CI) else
/// `~/.empyrean/data` (local). Returns `None` to skip when neither yields
/// a working Context, except under `EMPYREAN_DATA_DIR`, where the kernels
/// are supposed to be present and a skip would hide a regression.
fn try_context() -> Option<Context> {
    let declared = std::env::var("EMPYREAN_DATA_DIR").ok().map(PathBuf::from);
    let candidates = [
        declared.clone(),
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".empyrean/data")),
    ];
    for dir in candidates.into_iter().flatten() {
        match Context::from_data_dir(Some(&dir)) {
            Ok(ctx) => return Some(ctx),
            Err(e) if declared.is_some() => {
                panic!("EMPYREAN_DATA_DIR is set, so this test must not skip: {e}")
            }
            Err(_) => {}
        }
    }
    eprintln!("skipping: no data directory");
    None
}

#[test]
fn get_states_echoes_a_utc_request_as_utc() {
    let Some(ctx) = try_context() else {
        return;
    };
    let epochs = [Epoch::from_mjd_utc(60200.0), Epoch::from_mjd_utc(60201.0)];
    let states = ctx
        .get_states(Origin::Earth, Origin::Sun, &epochs, Frame::EclipticJ2000)
        .expect("Earth-Sun states at the requested epochs");
    assert_eq!(states.len(), epochs.len());
    for (s, req) in states.iter().zip(epochs.iter()) {
        // A UTC request yields a UTC-tagged output, echoed bit-for-bit, the
        // caller's own value, not a round-trip of the engine's TDB return.
        assert_eq!(s.epoch.scale(), TimeScale::UTC);
        assert_eq!(s.epoch.mjd(), req.mjd());
    }
}
