//! Cross-channel parity: the empyrean Rust safe wrapper vs empyrean-core.
//!
//! Sibling of `empyrean-py/tests/test_core_parity.py` for the **wrapper**
//! channel. Both run the shared `parity_manifest.json` scenarios through
//! their own API, project to the same `"<scenario>/<op>/<table>/<index>/
//! <field>"` fingerprint vocabulary, and diff against the committed
//! `core_parity_oracle.json` (produced by empyrean-core's `parity-oracle`
//! binary). empyrean-core still carries every field; a field it populates
//! that the wrapper drops, NaNs, or diverges on is a contract violation
//! (bd empyrean-na4h.2, ADR cross-channel-parity).
//!
//! Testing the wrapper directly (not only transitively via empyrean-py)
//! gives attribution — a wrapper-layer drop fails here, localized — and
//! covers the wrapper as the independent Rust distribution surface.
//!
//! The wrapper's flat `Event` struct is a reduced projection: it has no
//! `shadow_fraction`/`illumination`, no periapsis relative state, and no
//! possible-impact probability payload. Those are the known C-ABI drops on
//! the manifest allow-list — absent here means "no column", which the diff
//! treats the same as the Python channel's NaN/0.0 sentinels.
//!
//! Needs kernels (EMPYREAN_DATA_DIR or the XDG default) and the
//! `libempyrean` dylib (built by empyrean-c). If the context can't init,
//! the test logs and returns (skips) rather than failing a kernel-less CI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use empyrean::{
    Context, CoordinateState, Epoch, Frame, Orbit, Origin, PropagationConfig, UncertaintyMethod,
};
use serde::Deserialize;

// ── Shared manifest + committed oracle ───────────────────────────────

#[derive(Deserialize)]
struct Grid {
    start: f64,
    step: f64,
    n: usize,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    epoch_mjd_tdb: f64,
    state: [f64; 6],
    cov_diag: [f64; 6],
    grid: Grid,
    ip_end_mjd: f64,
    #[serde(default)]
    rtol: Option<f64>,
    ephemeris_obs_code: Option<String>,
    #[serde(default)]
    ephemeris_epochs: Vec<f64>,
}

#[derive(Deserialize)]
struct DropSpec {
    table: String,
    field: String,
    #[allow(dead_code)]
    issue: String,
}

#[derive(Deserialize)]
struct Tolerances {
    rtol: f64,
    atol: f64,
    epoch_tol_days: f64,
}

#[derive(Deserialize)]
struct Manifest {
    scenarios: Vec<Scenario>,
    known_drops: Vec<DropSpec>,
    tolerances: Tolerances,
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../empyrean-py/tests/fixtures")
}

/// Finite -> Some, NaN/Inf -> None (the channel's NaN ≡ core's null).
fn num(x: f64) -> Option<f64> {
    if x.is_finite() { Some(x) } else { None }
}

/// MJD TDB of an epoch (always representable for these TDB fixtures).
fn mjd(e: &Epoch) -> f64 {
    e.mjd_tdb().expect("epoch mjd_tdb")
}

type Row = Vec<(&'static str, Option<f64>)>;

/// Sort rows by epoch, stamp `epoch_mjd_tdb`, flatten to fingerprint keys.
fn emit_table(
    fp: &mut BTreeMap<String, Option<f64>>,
    scenario: &str,
    op: &str,
    table: &str,
    mut rows: Vec<(f64, Row)>,
) {
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (epoch, fields)) in rows.into_iter().enumerate() {
        let base = format!("{scenario}/{op}/{table}/{i}");
        fp.insert(format!("{base}/epoch_mjd_tdb"), num(epoch));
        for (field, val) in fields {
            fp.insert(format!("{base}/{field}"), val);
        }
    }
}

fn build_orbit(s: &Scenario) -> Orbit {
    let mut cov = [[0.0_f64; 6]; 6];
    for (i, row) in cov.iter_mut().enumerate() {
        row[i] = s.cov_diag[i];
    }
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(s.epoch_mjd_tdb),
        s.state,
        Frame::EclipticJ2000,
        Origin::Sun,
    )
    .with_covariance(cov);
    Orbit::new(state)
        .with_orbit_id(s.name.clone())
        .with_object_id(s.name.clone())
}

/// Project the wrapper channel for one scenario into the oracle vocabulary.
fn channel_fingerprint(ctx: &Context, s: &Scenario) -> BTreeMap<String, Option<f64>> {
    let mut fp = BTreeMap::new();
    let orbit = build_orbit(s);

    // ── propagate -> events ──
    let epochs: Vec<Epoch> = (0..s.grid.n)
        .map(|i| Epoch::from_mjd_tdb(s.grid.start + s.grid.step * i as f64))
        .collect();
    let cfg = PropagationConfig {
        uncertainty_method: UncertaintyMethod::FirstOrder,
        ..Default::default()
    };
    let result = ctx
        .propagate(std::slice::from_ref(&orbit), &epochs, &cfg)
        .expect("propagate");

    let mut buckets: BTreeMap<&'static str, Vec<(f64, Row)>> = BTreeMap::new();
    for e in &result.events {
        let epoch = mjd(&e.epoch);
        let (table, fields): (&'static str, Row) = match e.event_type.as_str() {
            "close_approach_start" => (
                "close_approach_starts",
                vec![
                    ("distance_au", num(e.distance_au)),
                    ("distance_km", num(e.distance_km)),
                ],
            ),
            "close_approach_end" => (
                "close_approach_ends",
                vec![
                    ("distance_au", num(e.distance_au)),
                    ("distance_km", num(e.distance_km)),
                ],
            ),
            // Periapsis relative state now wired through the C ABI
            // (empyrean-14cz.1).
            "periapsis" => (
                "periapses",
                vec![
                    ("distance_au", num(e.distance_au)),
                    ("distance_km", num(e.distance_km)),
                    ("relative_velocity_au_day", num(e.relative_velocity_au_day)),
                    ("relative_x", num(e.relative_x)),
                    ("relative_y", num(e.relative_y)),
                    ("relative_z", num(e.relative_z)),
                    ("relative_vx", num(e.relative_vx)),
                    ("relative_vy", num(e.relative_vy)),
                    ("relative_vz", num(e.relative_vz)),
                ],
            ),
            "impact" => (
                "impacts",
                vec![
                    ("latitude_deg", num(e.impact_latitude_deg)),
                    ("longitude_deg", num(e.impact_longitude_deg)),
                    ("altitude_km", num(e.impact_altitude_km)),
                ],
            ),
            "atmospheric_entry" => (
                "atmospheric_entries",
                vec![("distance_au", num(e.distance_au))],
            ),
            "atmospheric_exit" => (
                "atmospheric_exits",
                vec![("distance_au", num(e.distance_au))],
            ),
            // shadow_fraction / illumination now wired through the C ABI
            // (empyrean-14cz.3).
            "shadow_entry" => (
                "shadow_entries",
                vec![
                    ("shadow_fraction", num(e.shadow_fraction)),
                    ("illumination", num(e.illumination)),
                ],
            ),
            "shadow_exit" => (
                "shadow_exits",
                vec![
                    ("shadow_fraction", num(e.shadow_fraction)),
                    ("illumination", num(e.illumination)),
                ],
            ),
            // possible-impact probability payload now wired through the C
            // ABI (empyrean-14cz.2). The miss distance is the generic event
            // distance. ip_second_order/nonlinearity/ip_agm/ip_mc are NaN
            // (→ null) for this first-order run; oracle emits null too.
            "possible_impact" => (
                "possible_impacts",
                vec![
                    ("miss_distance_au", num(e.distance_au)),
                    ("miss_distance_km", num(e.distance_km)),
                    ("relative_velocity_au_day", num(e.relative_velocity_au_day)),
                    ("effective_radius_au", num(e.effective_radius_au)),
                    ("effective_radius_km", num(e.effective_radius_km)),
                    ("sigma_distance_au", num(e.sigma_distance_au)),
                    ("ip_linear", num(e.ip_linear)),
                    ("ip_second_order", num(e.ip_second_order)),
                    ("nonlinearity", num(e.nonlinearity)),
                    ("ip_agm", num(e.ip_agm)),
                    ("ip_mc", num(e.ip_mc)),
                ],
            ),
            _ => continue,
        };
        buckets.entry(table).or_default().push((epoch, fields));
    }
    for (table, rows) in buckets {
        emit_table(&mut fp, &s.name, "propagate", table, rows);
    }

    // ── standalone compute_impact_probabilities (positive control) ──
    let end = Epoch::from_mjd_tdb(s.ip_end_mjd);
    let methods = [UncertaintyMethod::FirstOrder];
    let earth = [Origin::Earth];
    let ips = ctx
        .compute_impact_probabilities(std::slice::from_ref(&orbit), end, &methods, &earth)
        .expect("compute_impact_probabilities");
    let ip_rows: Vec<(f64, Row)> = ips
        .iter()
        .map(|ip| {
            (
                mjd(&ip.epoch),
                vec![
                    ("miss_distance_au", num(ip.miss_distance_au)),
                    ("miss_distance_km", num(ip.miss_distance_km)),
                    ("effective_radius_au", num(ip.effective_radius_au)),
                    ("effective_radius_km", num(ip.effective_radius_km)),
                    ("sigma_distance_au", num(ip.sigma_distance_au)),
                    ("sigma_distance_km", num(ip.sigma_distance_km)),
                    ("ip_linear", num(ip.ip_linear)),
                    ("relative_velocity_au_day", num(ip.relative_velocity_au_day)),
                    ("ip_second_order", num(ip.ip_second_order)),
                    ("nonlinearity", num(ip.nonlinearity)),
                    ("ip_agm", num(ip.ip_agm)),
                    ("ip_mc", num(ip.ip_mc)),
                ],
            )
        })
        .collect();
    emit_table(
        &mut fp,
        &s.name,
        "compute_impact_probabilities",
        "impact_probabilities",
        ip_rows,
    );

    // ── standalone compute_b_planes (positive control) ──
    let bps = ctx
        .compute_b_planes(std::slice::from_ref(&orbit), end, &methods, &earth)
        .expect("compute_b_planes");
    let bp_rows: Vec<(f64, Row)> = bps
        .iter()
        .map(|b| {
            (
                mjd(&b.epoch),
                vec![
                    ("b_dot_t_km", num(b.b_dot_t_km)),
                    ("b_dot_r_km", num(b.b_dot_r_km)),
                    ("b_mag_km", num(b.b_mag_km)),
                    ("v_inf_km_s", num(b.v_inf_km_s)),
                    ("effective_radius_km", num(b.effective_radius_km)),
                    ("body_radius_km", num(b.body_radius_km)),
                    ("cov_tt_km2", num(b.cov_b_plane[0])),
                    ("cov_tr_km2", num(b.cov_b_plane[1])),
                    ("cov_rr_km2", num(b.cov_b_plane[2])),
                    ("semi_major_3sig_km", num(b.semi_major_3sig_km)),
                    ("semi_minor_3sig_km", num(b.semi_minor_3sig_km)),
                    ("ellipse_angle_rad", num(b.ellipse_angle_rad)),
                    ("ip_linear", num(b.ip_linear)),
                ],
            )
        })
        .collect();
    emit_table(&mut fp, &s.name, "compute_b_planes", "b_planes", bp_rows);

    // ── generate_ephemeris sky-position scalars (positive control) ──
    if let Some(code) = &s.ephemeris_obs_code
        && !s.ephemeris_epochs.is_empty()
    {
        let eph_epochs: Vec<Epoch> = s
            .ephemeris_epochs
            .iter()
            .map(|&t| Epoch::from_mjd_tdb(t))
            .collect();
        let observers = ctx
            .get_observers(&[code.as_str()], &eph_epochs)
            .expect("observers");
        let entries = ctx
            .generate_ephemeris(
                std::slice::from_ref(&orbit),
                &observers,
                &Default::default(),
            )
            .expect("generate_ephemeris")
            .entries;
        let eph_rows: Vec<(f64, Row)> = entries
            .iter()
            .map(|e| {
                (
                    mjd(&e.epoch),
                    vec![
                        ("lon", num(e.ra_deg)),
                        ("lat", num(e.dec_deg)),
                        ("rho", num(e.rho_au)),
                        ("vrho", num(e.vrho_au_day)),
                        ("vlon", num(e.vra_deg_day)),
                        ("vlat", num(e.vdec_deg_day)),
                        ("light_time", num(e.light_time_days)),
                        ("phase_angle", num(e.phase_angle_deg)),
                        ("elongation", num(e.elongation_deg)),
                        ("heliocentric_distance", num(e.heliocentric_distance_au)),
                        ("mag", num(e.mag)),
                        ("mag_sigma", num(e.mag_sigma)),
                        ("zenith_angle", num(e.zenith_angle_deg)),
                        ("azimuth", num(e.azimuth_deg)),
                        ("hour_angle", num(e.hour_angle_deg)),
                        ("lunar_elongation", num(e.lunar_elongation_deg)),
                        ("position_angle", num(e.position_angle_deg)),
                        ("sky_rate", num(e.sky_rate_deg_day)),
                    ],
                )
            })
            .collect();
        emit_table(
            &mut fp,
            &s.name,
            "generate_ephemeris",
            "ephemeris",
            eph_rows,
        );
    }

    fp
}

#[test]
fn wrapper_core_parity_no_silent_value_drops() {
    let dir = data_dir();
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("parity_manifest.json")).unwrap())
            .expect("parse manifest");
    let oracle_raw: BTreeMap<String, serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("core_parity_oracle.json")).unwrap(),
    )
    .expect("parse oracle");
    let oracle: BTreeMap<String, Option<f64>> = oracle_raw
        .into_iter()
        .map(|(k, v)| (k, v.as_f64()))
        .collect();

    let ctx = match Context::from_data_dir(None) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP wrapper parity: context init failed (missing kernels?): {e}");
            return;
        }
    };

    let known_drops: std::collections::HashSet<(String, String)> = manifest
        .known_drops
        .iter()
        .map(|d| (d.table.clone(), d.field.clone()))
        .collect();
    let (rtol, atol, etol) = (
        manifest.tolerances.rtol,
        manifest.tolerances.atol,
        manifest.tolerances.epoch_tol_days,
    );
    // Per-scenario rtol override (the chaotic impactor needs a loose one —
    // oracle and dylib are separately compiled, ULP codegen differences
    // amplify through its near-surface encounter).
    let scenario_rtol: BTreeMap<String, f64> = manifest
        .scenarios
        .iter()
        .map(|s| (s.name.clone(), s.rtol.unwrap_or(rtol)))
        .collect();
    let close = |a: Option<f64>, b: Option<f64>, rtol: f64| -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => (x - y).abs() <= atol + rtol * x.abs().max(1.0),
            _ => false,
        }
    };
    // (table, field) from "<scenario>/<op>/<table>/<index>/<field>"
    let parse = |key: &str| -> (String, String) {
        let p: Vec<&str> = key.split('/').collect();
        (p[2].to_string(), p[p.len() - 1].to_string())
    };

    let mut channel: BTreeMap<String, Option<f64>> = BTreeMap::new();
    for s in &manifest.scenarios {
        channel.extend(channel_fingerprint(&ctx, s));
    }

    let mut new_violations: Vec<String> = Vec::new();
    let mut misalignments: Vec<String> = Vec::new();
    // (table, field) -> did EVERY occurrence match core? (reverse-enforce)
    let mut drop_all_match: BTreeMap<(String, String), bool> = BTreeMap::new();

    let mut all_keys: std::collections::BTreeSet<&String> = oracle.keys().collect();
    all_keys.extend(channel.keys());
    for key in all_keys {
        let (table, field) = parse(key);
        let core = oracle.get(key).copied().flatten();
        let in_oracle = oracle.contains_key(key);
        let in_channel = channel.contains_key(key);

        if field == "epoch_mjd_tdb" {
            match (
                oracle.get(key).copied().flatten(),
                channel.get(key).copied().flatten(),
            ) {
                (Some(a), Some(b)) if (a - b).abs() <= etol => {}
                (a, b) => misalignments.push(format!(
                    "{key}: core={a:?} chan={b:?} (event-stream divergence)"
                )),
            }
            continue;
        }

        let is_drop = known_drops.contains(&(table.clone(), field.clone()));

        if in_oracle && !in_channel {
            // Core emitted a key the wrapper has no column for.
            if core.is_none() {
                // Core has no value either (null) — not a drop.
            } else if is_drop {
                drop_all_match
                    .entry((table, field))
                    .and_modify(|m| *m = false)
                    .or_insert(false);
            } else {
                new_violations.push(format!(
                    "{key}: core={core:?} but wrapper has no such field"
                ));
            }
            continue;
        }
        if in_channel && !in_oracle {
            new_violations.push(format!("{key}: wrapper emitted a field core does not"));
            continue;
        }

        let chan = channel.get(key).copied().flatten();
        let scenario = key.split('/').next().unwrap_or("");
        let matched = close(
            core,
            chan,
            scenario_rtol.get(scenario).copied().unwrap_or(rtol),
        );
        if is_drop {
            let entry = drop_all_match.entry((table, field)).or_insert(true);
            *entry = *entry && matched;
        } else if !matched {
            new_violations.push(format!("{key}: core={core:?} wrapper={chan:?}"));
        }
    }

    let stale: Vec<String> = drop_all_match
        .iter()
        .filter(|(_, m)| **m)
        .map(|((t, f), _)| format!("({t}, {f})"))
        .collect();

    let mut msgs: Vec<String> = Vec::new();
    if !misalignments.is_empty() {
        msgs.push(format!(
            "Event-stream MISALIGNMENT:\n  {}",
            misalignments.join("\n  ")
        ));
    }
    if !new_violations.is_empty() {
        msgs.push(format!(
            "NEW core-vs-wrapper value drops/divergences (wire it through, or add to \
             known_drops in parity_manifest.json):\n  {}",
            new_violations.join("\n  ")
        ));
    }
    if !stale.is_empty() {
        msgs.push(format!(
            "STALE known_drops (every occurrence now matches core — remove from manifest):\n  {}",
            stale.join("\n  ")
        ));
    }
    assert!(msgs.is_empty(), "\n\n{}", msgs.join("\n\n"));
}
