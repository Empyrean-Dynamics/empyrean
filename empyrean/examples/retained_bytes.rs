//! What a propagation result retains after its states have been read —
//! the measurement behind the per-(orbit, epoch) figures documented on
//! [`PropagationResult::into_states`](empyrean::propagate::PropagationResult::into_states).
//!
//! Runs one 64-orbit × 2-epoch call `REPS` times, accumulating either
//! whole `PropagationResult`s (the engine-side result stays alive behind
//! each one) or only the states `into_states` hands back, and reports
//! resident-set growth per accumulated call. Divided by the 128 cells of
//! the call, that is the per-(orbit, epoch) cost of each shape; the
//! difference between the two is what `into_states` releases.
//!
//! ```text
//! cargo run --release -p empyrean --example retained_bytes -- first  results
//! cargo run --release -p empyrean --example retained_bytes -- first  states
//! cargo run --release -p empyrean --example retained_bytes -- second results
//! cargo run --release -p empyrean --example retained_bytes -- second states
//! ```
//!
//! The first argument is the request shape (`first` = FirstOrder, the
//! raster case; `second` = SecondOrder with the STM, the derivative
//! case); the second selects which of the two accumulations to run.
//!
//! **One phase per process, deliberately.** Running both in one process
//! lets the first phase's freed pages back the second phase's
//! allocations, so the second reads far below what it costs — that
//! aliasing swung an earlier single-process version of this harness by
//! 40% between runs. Requires a data directory
//! (`EMPYREAN_DATA_DIR`, else `~/.empyrean/data`).
//!
//! Measurement only — not part of the crate's published surface, and not
//! run by CI. The `states` figure is a small delta against a
//! kernel-loaded baseline of over a gigabyte, so it carries page-
//! granularity noise the `results` figure does not; treat it as an
//! order of magnitude, not a pinned number.

use empyrean::{
    Context, CoordinateState, Epoch, Frame, Orbit, Origin, PropagationConfig, UncertaintyMethod,
};
use std::path::PathBuf;

const APOPHIS: [f64; 6] = [
    -0.078_526_491_490_690_46,
    -0.819_748_051_902_064_6,
    0.041_893_951_532_339_09,
    0.019_875_102_496_888_46,
    0.001_322_088_445_361_402,
    0.000_399_496_044_422_352_2,
];
const T0_MJD: f64 = 61000.0;
const N_ORBITS: usize = 64;
const N_EPOCHS: usize = 2;
const REPS: usize = 24;

fn rss_bytes() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .expect("rss kb")
        * 1024
}

fn ctx() -> Context {
    for dir in [
        std::env::var("EMPYREAN_DATA_DIR").ok().map(PathBuf::from),
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".empyrean/data")),
    ]
    .into_iter()
    .flatten()
    {
        if let Ok(c) = Context::from_data_dir(Some(&dir)) {
            return c;
        }
    }
    panic!("no data dir");
}

fn batch() -> Vec<Orbit> {
    let mut cov = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        cov[i][i] = 1.0e-14;
        cov[i + 3][i + 3] = 1.0e-20;
    }
    (0..N_ORBITS)
        .map(|i| {
            let mut e = APOPHIS;
            e[0] += (i as f64) * 1.0e-7;
            let state = CoordinateState::cartesian(
                Epoch::from_mjd_tdb(T0_MJD),
                e,
                Frame::EclipticJ2000,
                Origin::Sun,
            )
            .with_covariance(cov);
            Orbit::new(state).with_orbit_id(format!("o{i}"))
        })
        .collect()
}

fn main() {
    // One phase per process. Running both in one process lets the first
    // phase's freed pages back the second phase's allocations, so the
    // second reads far lower than it costs — which is what made an
    // earlier single-process version of this harness swing by 40%.
    let which = std::env::args().nth(1).unwrap_or_else(|| "second".into());
    let phase = std::env::args().nth(2).unwrap_or_else(|| "results".into());

    let ctx = ctx();
    let orbits = batch();
    let epochs: Vec<Epoch> = (0..N_EPOCHS)
        .map(|k| Epoch::from_mjd_tdb(T0_MJD + 30.0 * (k as f64 + 1.0)))
        .collect();
    let config = match which.as_str() {
        // The raster shape: no covariance readback wanted, no STT.
        "first" => PropagationConfig {
            uncertainty_method: UncertaintyMethod::FirstOrder,
            num_threads: std::num::NonZeroUsize::new(1),
            ..PropagationConfig::default()
        },
        _ => PropagationConfig {
            uncertainty_method: UncertaintyMethod::SecondOrder,
            compute_stm: true,
            num_threads: std::num::NonZeroUsize::new(1),
            ..PropagationConfig::default()
        },
    };

    // Warm every allocator arena and lazy kernel page before measuring.
    for _ in 0..3 {
        let _ = ctx
            .propagate(&orbits, &epochs, &config)
            .expect("warmup propagate")
            .into_states();
    }

    let base = rss_bytes();
    let mut kept_states = Vec::new();
    let mut kept_results = Vec::new();
    for _ in 0..REPS {
        let r = ctx.propagate(&orbits, &epochs, &config).expect("propagate");
        if phase == "states" {
            kept_states.push(r.into_states());
        } else {
            kept_results.push(r);
        }
    }
    let after = rss_bytes();

    let cells = (N_ORBITS * N_EPOCHS) as f64;
    let per_call = (after.saturating_sub(base)) as f64 / REPS as f64;
    println!(
        "{which}/{phase}: reps={REPS} cells={cells} rss {base} -> {after}  \
         {per_call:.0} B/call  {:.0} B/(orbit,epoch)",
        per_call / cells
    );
    drop(kept_states);
    drop(kept_results);
}
