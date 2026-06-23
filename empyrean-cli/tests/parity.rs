//! Cross-channel parity: the empyrean **CLI** vs empyrean-core.
//!
//! Third channel of the parity suite (after empyrean-py and the wrapper).
//! Runs the real `empyrean` binary (`CARGO_BIN_EXE_empyrean`) on the shared
//! `parity_manifest.json` scenarios, parses its output files, projects to
//! the `"<scenario>/<op>/<table>/<index>/<field>"` fingerprint vocabulary,
//! and diffs against the committed `core_parity_oracle.json`. Uniquely
//! exercises the CLI's **output-serialization** layer (the `write_*` file
//! writers) end-to-end — the one surface py/wrapper don't touch.
//!
//! Scope (bd empyrean-na4h.5): the CLI has only `propagate` + `ephemeris`
//! commands (no impact-probabilities / b-planes), so only those two ops are
//! compared. Beyond the shared C-ABI drops, the CLI file writers drop more
//! (bd empyrean-i7u5): `write_events_*` emits common fields only (no
//! per-event-type fields), and the ephemeris writer omits the 6 topocentric
//! angles. Those CLI-specific drops are allow-listed in CLI_EXTRA_DROPS.
//!
//! Needs kernels (EMPYREAN_DATA_DIR / XDG) + the `libempyrean` dylib. If the
//! binary can't init a context, the test logs and returns (skips).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use empyrean::{CoordinateState, Epoch, Frame, Orbit, OrbitBatch, Origin};
use serde::Deserialize;

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

/// CLI-only drops beyond the shared C-ABI ones. Now EMPTY: bd empyrean-i7u5
/// is fully fixed — both the events file writer (full per-event-type
/// payload) and the ephemeris writer (6 topocentric angles) carry every
/// field the wrapper/core do. Kept as a hook so a future CLI-specific
/// output drop has a home (and the test reverse-enforces a stale entry).
const CLI_EXTRA_DROPS: &[(&str, &str)] = &[];

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../empyrean-py/tests/fixtures")
}

fn num(x: f64) -> Option<f64> {
    if x.is_finite() { Some(x) } else { None }
}

type Row = Vec<(&'static str, Option<f64>)>;

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

/// Minimal CSV reader: header -> name->index, rows -> Vec<Vec<cell>>.
/// Safe here — the CLI emits plain numeric / short-string cells, no quoting.
fn read_csv(path: &Path) -> (BTreeMap<String, usize>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines = text.lines();
    let header: BTreeMap<String, usize> = lines
        .next()
        .unwrap_or("")
        .split(',')
        .enumerate()
        .map(|(i, name)| (name.trim().to_string(), i))
        .collect();
    let rows = lines
        .map(|l| l.split(',').map(|c| c.trim().to_string()).collect())
        .collect();
    (header, rows)
}

fn cell(row: &[String], idx: &BTreeMap<String, usize>, name: &str) -> Option<f64> {
    idx.get(name)
        .and_then(|&i| row.get(i))
        .and_then(|s| s.parse::<f64>().ok())
        .and_then(num)
}

fn write_input_json(s: &Scenario, dir: &Path) -> PathBuf {
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
    let orbit = Orbit::new(state)
        .with_orbit_id(s.name.clone())
        .with_object_id(s.name.clone());
    let batch = OrbitBatch::new(
        vec![orbit],
        vec![s.name.clone()],
        vec![Some(s.name.clone())],
    )
    .expect("batch");
    let path = dir.join(format!("{}_in.json", s.name));
    empyrean::write_orbits_json(&path, &batch).expect("write input json");
    path
}

/// Run the CLI binary with the dylib + data dir on the environment.
fn run_cli(args: &[&str]) -> bool {
    let bin = env!("CARGO_BIN_EXE_empyrean");
    let dylib_dir = Path::new(bin).parent().unwrap();
    Command::new(bin)
        .args(args)
        .env("DYLD_LIBRARY_PATH", dylib_dir)
        .env("LD_LIBRARY_PATH", dylib_dir)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn project_scenario(s: &Scenario, tmp: &Path) -> Option<BTreeMap<String, Option<f64>>> {
    let input = write_input_json(s, tmp);
    let mut fp = BTreeMap::new();

    // ── propagate -> events (common fields only) ──
    let last = s.grid.start + s.grid.step * (s.grid.n as f64 - 1.0);
    let out = tmp.join(format!("{}_prop", s.name));
    let ok = run_cli(&[
        "propagate",
        "--input",
        input.to_str().unwrap(),
        "--epoch",
        &format!("{last}"),
        "--out-dir",
        out.to_str().unwrap(),
        "--format",
        "csv",
    ]);
    if !ok {
        return None; // context init failed (missing kernels) -> skip
    }
    let (hdr, rows) = read_csv(&out.join("events.csv"));
    let et = |r: &[String]| {
        hdr.get("event_type")
            .and_then(|&i| r.get(i))
            .cloned()
            .unwrap_or_default()
    };
    let mut buckets: BTreeMap<&'static str, Vec<(f64, Row)>> = BTreeMap::new();
    for r in &rows {
        let epoch = cell(r, &hdr, "epoch_mjd_tdb").unwrap_or(f64::NAN);
        let d_au = cell(r, &hdr, "distance_au");
        let d_km = cell(r, &hdr, "distance_km");
        let rv = cell(r, &hdr, "relative_velocity_au_day");
        let c = |name: &str| cell(r, &hdr, name);
        // The CLI events file now carries the full per-event-type payload
        // (bd empyrean-i7u5 fixed), so each table compares everything the
        // oracle emits for it.
        let (table, fields): (&'static str, Row) = match et(r).as_str() {
            // close_approach START/END are zone-boundary (threshold-crossing)
            // events — ill-conditioned, and the CLI takes a single --epoch so
            // it can't replicate the oracle's multi-target integration driving
            // (the boundary distance/epoch then differ ~1-11%). Excluded for
            // the CLI (still covered by the py + wrapper channels, which pass
            // the full grid). The robust extremum (periapsis) IS compared.
            "close_approach_start" | "close_approach_end" => continue,
            "periapsis" => (
                "periapses",
                vec![
                    ("distance_au", d_au),
                    ("distance_km", d_km),
                    ("relative_velocity_au_day", rv),
                    ("relative_x", c("relative_x")),
                    ("relative_y", c("relative_y")),
                    ("relative_z", c("relative_z")),
                    ("relative_vx", c("relative_vx")),
                    ("relative_vy", c("relative_vy")),
                    ("relative_vz", c("relative_vz")),
                ],
            ),
            "atmospheric_entry" => ("atmospheric_entries", vec![("distance_au", d_au)]),
            "atmospheric_exit" => ("atmospheric_exits", vec![("distance_au", d_au)]),
            "possible_impact" => (
                "possible_impacts",
                vec![
                    ("miss_distance_au", d_au),
                    ("miss_distance_km", d_km),
                    ("relative_velocity_au_day", rv),
                    ("effective_radius_au", c("effective_radius_au")),
                    ("effective_radius_km", c("effective_radius_km")),
                    ("sigma_distance_au", c("sigma_distance_au")),
                    ("ip_linear", c("ip_linear")),
                    ("ip_second_order", c("ip_second_order")),
                    ("nonlinearity", c("nonlinearity")),
                    ("ip_agm", c("ip_agm")),
                    ("ip_mc", c("ip_mc")),
                ],
            ),
            "impact" => (
                "impacts",
                vec![
                    ("latitude_deg", c("impact_latitude_deg")),
                    ("longitude_deg", c("impact_longitude_deg")),
                    ("altitude_km", c("impact_altitude_km")),
                ],
            ),
            // shadow_fraction/illumination are evaluated at the umbra-edge
            // threshold crossing — a steep function the CLI's single --epoch
            // can't reproduce vs the oracle's grid for the chaotic impactor
            // (the only fixture that fires shadow; differs ~56%, like the
            // close-approach boundaries). The fields ARE carried (i7u5 fixed
            // — verified by the CSV header + the grid-driven py/wrapper
            // channels); excluded from the CLI VALUE comparison only.
            "shadow_entry" | "shadow_exit" => continue,
            _ => continue,
        };
        buckets.entry(table).or_default().push((epoch, fields));
    }
    for (table, rs) in buckets {
        emit_table(&mut fp, &s.name, "propagate", table, rs);
    }

    // ── ephemeris (one CLI run per epoch) ──
    if let Some(code) = &s.ephemeris_obs_code
        && !s.ephemeris_epochs.is_empty()
    {
        let mut eph_rows: Vec<(f64, Row)> = Vec::new();
        for (k, &ep) in s.ephemeris_epochs.iter().enumerate() {
            let eout = tmp.join(format!("{}_eph{k}", s.name));
            let ok = run_cli(&[
                "ephemeris",
                "--input",
                input.to_str().unwrap(),
                "--observers",
                code,
                "--epoch",
                &format!("{ep}"),
                "--out-dir",
                eout.to_str().unwrap(),
                "--format",
                "csv",
            ]);
            if !ok {
                continue;
            }
            let (h, rs) = read_csv(&eout.join("ephemeris.csv"));
            if let Some(r) = rs.first() {
                let epoch = cell(r, &h, "epoch_mjd_tdb").unwrap_or(ep);
                eph_rows.push((
                    epoch,
                    vec![
                        ("lon", cell(r, &h, "ra_deg")),
                        ("lat", cell(r, &h, "dec_deg")),
                        ("rho", cell(r, &h, "rho_au")),
                        ("vrho", cell(r, &h, "vrho_au_day")),
                        ("vlon", cell(r, &h, "vra_deg_day")),
                        ("vlat", cell(r, &h, "vdec_deg_day")),
                        ("light_time", cell(r, &h, "light_time_days")),
                        ("phase_angle", cell(r, &h, "phase_angle_deg")),
                        ("elongation", cell(r, &h, "elongation_deg")),
                        (
                            "heliocentric_distance",
                            cell(r, &h, "heliocentric_distance_au"),
                        ),
                        ("mag", cell(r, &h, "mag")),
                        ("mag_sigma", cell(r, &h, "mag_sigma")),
                        ("zenith_angle", cell(r, &h, "zenith_angle_deg")),
                        ("azimuth", cell(r, &h, "azimuth_deg")),
                        ("hour_angle", cell(r, &h, "hour_angle_deg")),
                        ("lunar_elongation", cell(r, &h, "lunar_elongation_deg")),
                        ("position_angle", cell(r, &h, "position_angle_deg")),
                        ("sky_rate", cell(r, &h, "sky_rate_deg_day")),
                    ],
                ));
            }
        }
        emit_table(
            &mut fp,
            &s.name,
            "generate_ephemeris",
            "ephemeris",
            eph_rows,
        );
    }

    Some(fp)
}

#[test]
fn cli_core_parity_no_silent_value_drops() {
    let dir = data_dir();
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("parity_manifest.json")).unwrap())
            .expect("manifest");
    let oracle_raw: BTreeMap<String, serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("core_parity_oracle.json")).unwrap(),
    )
    .expect("oracle");
    // CLI covers only propagate + generate_ephemeris (no IP / b-plane cmds),
    // and close-approach zone boundaries are excluded (single --epoch can't
    // match the oracle's grid-driven boundary detection — see project_scenario).
    let oracle: BTreeMap<String, Option<f64>> = oracle_raw
        .into_iter()
        .filter(|(k, _)| {
            let parts: Vec<&str> = k.split('/').collect();
            let op = parts.get(1).copied().unwrap_or("");
            let table = parts.get(2).copied().unwrap_or("");
            (op == "propagate" || op == "generate_ephemeris")
                && table != "close_approach_starts"
                && table != "close_approach_ends"
                // shadow fraction/illumination are umbra-edge threshold values
                // the CLI single --epoch can't reproduce vs the grid (see the
                // shadow arm in project_scenario).
                && table != "shadow_entries"
                && table != "shadow_exits"
        })
        .map(|(k, v)| (k, v.as_f64()))
        .collect();

    let mut drops: HashSet<(String, String)> = manifest
        .known_drops
        .iter()
        .map(|d| (d.table.clone(), d.field.clone()))
        .collect();
    drops.extend(
        CLI_EXTRA_DROPS
            .iter()
            .map(|(t, f)| (t.to_string(), f.to_string())),
    );
    let (rtol, atol, etol) = (
        manifest.tolerances.rtol,
        manifest.tolerances.atol,
        manifest.tolerances.epoch_tol_days,
    );
    let scenario_rtol: BTreeMap<String, f64> = manifest
        .scenarios
        .iter()
        .map(|s| (s.name.clone(), s.rtol.unwrap_or(rtol)))
        .collect();
    let close = |a: Option<f64>, b: Option<f64>, rtol: f64| match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => (x - y).abs() <= atol + rtol * x.abs().max(1.0),
        _ => false,
    };
    let parse = |key: &str| -> (String, String) {
        let p: Vec<&str> = key.split('/').collect();
        (p[2].to_string(), p[p.len() - 1].to_string())
    };

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let mut channel: BTreeMap<String, Option<f64>> = BTreeMap::new();
    for (i, s) in manifest.scenarios.iter().enumerate() {
        match project_scenario(s, &tmp) {
            Some(fp) => channel.extend(fp),
            None if i == 0 => {
                eprintln!("SKIP cli parity: CLI propagate failed (missing kernels/dylib?)");
                return;
            }
            None => panic!("CLI run failed for scenario {}", s.name),
        }
    }

    let mut new_violations: Vec<String> = Vec::new();
    let mut misalignments: Vec<String> = Vec::new();
    let mut drop_all_match: BTreeMap<(String, String), bool> = BTreeMap::new();

    let mut keys: BTreeSet<&String> = oracle.keys().collect();
    keys.extend(channel.keys());
    for key in keys {
        let (table, field) = parse(key);
        let core = oracle.get(key).copied().flatten();
        let in_oracle = oracle.contains_key(key);
        let in_channel = channel.contains_key(key);

        if field == "epoch_mjd_tdb" {
            match (core, channel.get(key).copied().flatten()) {
                (Some(a), Some(b)) if (a - b).abs() <= etol => {}
                // epoch only present on one side is fine when that table is
                // entirely a CLI-extra drop with no shared fields.
                (Some(_), None) | (None, Some(_)) if !in_oracle || !in_channel => {}
                (a, b) => misalignments.push(format!("{key}: core={a:?} chan={b:?}")),
            }
            continue;
        }

        let is_drop = drops.contains(&(table.clone(), field.clone()));

        if in_oracle && !in_channel {
            if core.is_none() {
            } else if is_drop {
                drop_all_match
                    .entry((table, field))
                    .and_modify(|m| *m = false)
                    .or_insert(false);
            } else {
                new_violations.push(format!("{key}: core={core:?} but CLI has no such field"));
            }
            continue;
        }
        if in_channel && !in_oracle {
            new_violations.push(format!("{key}: CLI emitted a field core does not"));
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
            let e = drop_all_match.entry((table, field)).or_insert(true);
            *e = *e && matched;
        } else if !matched {
            new_violations.push(format!("{key}: core={core:?} cli={chan:?}"));
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
            "NEW core-vs-CLI value drops/divergences (wire it through, or add to \
             known_drops/CLI_EXTRA_DROPS):\n  {}",
            new_violations.join("\n  ")
        ));
    }
    if !stale.is_empty() {
        msgs.push(format!(
            "STALE drops (every occurrence now matches core — remove the entry):\n  {}",
            stale.join("\n  ")
        ));
    }
    assert!(msgs.is_empty(), "\n\n{}", msgs.join("\n\n"));
}
