//! End-to-end tests for `empyrean show`.
//!
//! Fixtures are written by the **same writers the pipeline commands
//! use** — `crate::io::output` is a thin wrapper over these `empyrean::`
//! functions, so the bytes here are the bytes `propagate` / `determine`
//! produce, schemas and all. That matters: the orbits Parquet schema is
//! 82 columns wide and no hand-rolled fixture would reproduce it.
//!
//! Everything asserted here runs `show` **non-interactively** (stdout is
//! a pipe under `cargo test`), which is exactly the degraded path the
//! feature promises: a plain aligned stream of the whole table.
//!
//! Writing a fixture needs the `libempyrean` runtime. When it is absent
//! the writer-backed tests log and return, following the same convention
//! as `tests/parity.rs`; the reader-side tests below build their fixtures
//! with the Parquet writer directly and always run.

use std::process::Command;

use empyrean::{
    CoordinateState, Epoch, Frame, ObservationResidual, Orbit, OrbitBatch, Origin, RejectionReason,
    write_orbits_csv, write_orbits_json, write_orbits_parquet, write_residuals_csv,
    write_residuals_json, write_residuals_parquet,
};

// ---------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------

/// Run `empyrean show <args>` and return (stdout, stderr, success).
///
/// stdout is a pipe here, so `show` takes its non-interactive path.
fn show(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_empyrean"))
        .arg("show")
        .args(args)
        .output()
        .expect("run empyrean show");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

fn lines(s: &str) -> Vec<&str> {
    s.lines().collect()
}

/// Whether the engine runtime can be loaded. Writing a fixture needs it;
/// reading one does not.
fn engine_available() -> bool {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("probe.csv");
    std::panic::catch_unwind(|| write_orbits_csv(&path, &orbit_batch(1)).is_ok()).unwrap_or(false)
}

macro_rules! require_engine {
    () => {
        if !engine_available() {
            eprintln!(
                "SKIP: libempyrean is not loadable, so the CLI's own writers cannot \
                 produce a fixture. Set EMPYREAN_LIB to a matching build to run this test."
            );
            return;
        }
    };
}

// ---------------------------------------------------------------------
// Fixture builders — the CLI's own writers
// ---------------------------------------------------------------------

fn orbit_batch(n: usize) -> OrbitBatch {
    let mut orbits = Vec::new();
    let mut orbit_ids = Vec::new();
    let mut object_ids = Vec::new();
    for i in 0..n {
        let mut cov = [[0.0_f64; 6]; 6];
        for (j, row) in cov.iter_mut().enumerate() {
            row[j] = 1e-12 * (j as f64 + 1.0);
        }
        let state = CoordinateState::cartesian(
            Epoch::from_mjd_tdb(60000.0 + i as f64),
            [
                1.0 + i as f64,
                2.0,
                3.0,
                0.010_203_040_506_070_8,
                0.02,
                0.03,
            ],
            Frame::EclipticJ2000,
            Origin::Sun,
        )
        .with_covariance(cov);
        orbits.push(Orbit::new(state));
        orbit_ids.push(format!("orb{i}"));
        // Every other object has no object_id, which is how a null
        // reaches the table.
        object_ids.push((i % 2 == 0).then(|| format!("9994{i}")));
    }
    OrbitBatch::new(orbits, orbit_ids, object_ids).expect("batch")
}

/// A residual with every optional quantity absent — the shape a fit
/// produces before any influence or sky-motion pass has run. NaN
/// throughout, which is what those columns genuinely hold.
fn base_residual() -> ObservationResidual {
    ObservationResidual {
        obs_id: String::new(),
        obs_code: "500".into(),
        ast_cat: None,
        epoch: Epoch::from_mjd_tdb(60000.0),
        ra_residual_arcsec: f64::NAN,
        dec_residual_arcsec: f64::NAN,
        chi2: f64::NAN,
        dof: 2,
        probability: f64::NAN,
        selected: true,
        residual_cov_ra: f64::NAN,
        residual_cov_dec: f64::NAN,
        residual_cov_corr: f64::NAN,
        rejection_reason: RejectionReason::Accepted,
        rejection_criterion: f64::NAN,
        rejection_threshold: f64::NAN,
        rejection_effective_threshold: f64::NAN,
        rejection_information_loss: f64::NAN,
        cooks_distance: f64::NAN,
        leverage: f64::NAN,
        fractional_information: f64::NAN,
        along_track_arcsec: f64::NAN,
        cross_track_arcsec: f64::NAN,
        along_track_error_arcsec: f64::NAN,
        cross_track_error_arcsec: f64::NAN,
        track_position_angle_deg: f64::NAN,
        influence_information_loss: f64::NAN,
        along_cross_covariance_arcsec2: f64::NAN,
        radar: None,
    }
}

/// Residuals carrying both kinds of absence: a null (`ast_cat`) and NaN
/// floats (`chi2`, the rejection criteria).
fn residuals() -> Vec<ObservationResidual> {
    let accepted = ObservationResidual {
        obs_id: "obs1".into(),
        obs_code: "568".into(),
        ast_cat: Some("Gaia3".into()),
        epoch: Epoch::from_mjd_tdb(60000.5),
        ra_residual_arcsec: 0.123_456_789,
        dec_residual_arcsec: -0.25,
        chi2: 1.5,
        probability: 0.47,
        residual_cov_ra: 0.04,
        residual_cov_dec: 0.04,
        residual_cov_corr: 0.0,
        rejection_criterion: 1.5,
        rejection_threshold: 9.0,
        ..base_residual()
    };
    let rejected = ObservationResidual {
        obs_id: "obs2".into(),
        obs_code: "F51".into(),
        // No star catalogue recorded — a null, not a NaN.
        ast_cat: None,
        epoch: Epoch::from_mjd_tdb(60001.5),
        ra_residual_arcsec: 12.0,
        dec_residual_arcsec: 0.0,
        // Combined covariance unavailable, so chi2 stays NaN — an
        // uncomputable number, not an absent one.
        selected: false,
        rejection_reason: RejectionReason::ChiSquared,
        rejection_criterion: 144.0,
        rejection_threshold: 9.0,
        ..base_residual()
    };
    vec![accepted, rejected]
}

// ---------------------------------------------------------------------
// Piped mode: the whole table, aligned, no interactivity
// ---------------------------------------------------------------------

#[test]
fn piped_csv_streams_the_whole_table_aligned() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.csv");
    write_orbits_csv(&path, &orbit_batch(2)).unwrap();

    let (stdout, stderr, ok) =
        show(&[path.to_str().unwrap(), "--columns", "orbit_id,object_id,e0"]);
    assert!(ok, "show failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec![
            "orbit_id  object_id  e0",
            "orb0      99940      1",
            "orb1                 2",
        ],
        "a null object_id must render as an empty cell, not `null`"
    );
}

#[test]
fn piped_parquet_streams_the_whole_table_aligned() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(2)).unwrap();

    let (stdout, stderr, ok) = show(&[path.to_str().unwrap(), "--columns", "orbit_id,t,x,frame"]);
    assert!(ok, "show failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec![
            "orbit_id  t      x  frame",
            "orb0      60000  1  EclipticJ2000",
            "orb1      60001  2  EclipticJ2000",
        ]
    );
}

#[test]
fn piped_json_streams_the_whole_table_aligned() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.json");
    write_orbits_json(&path, &orbit_batch(2)).unwrap();

    let (stdout, stderr, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "orbit_id,epoch_mjd_tdb,elements",
    ]);
    assert!(ok, "show failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec![
            "orbit_id  epoch_mjd_tdb  elements",
            "orb0      60000          [1.0,2.0,3.0,0.0102030405060708,0.02,0.03]",
            "orb1      60001          [2.0,2.0,3.0,0.0102030405060708,0.02,0.03]",
        ],
        "a nested array must stay in its own cell, compactly"
    );
}

// ---------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------

#[test]
fn limit_caps_the_row_count() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(10)).unwrap();

    let (stdout, stderr, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "orbit_id",
        "--limit",
        "3",
    ]);
    assert!(ok, "show failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec!["orbit_id", "orb0", "orb1", "orb2"],
        "--limit must cut the rows, not the header"
    );
}

#[test]
fn no_header_omits_only_the_header() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(2)).unwrap();

    let (stdout, _, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "orbit_id",
        "--no-header",
    ]);
    assert!(ok);
    assert_eq!(lines(&stdout), vec!["orb0", "orb1"]);
}

#[test]
fn columns_subsets_and_reorders() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(1)).unwrap();

    let (stdout, _, ok) = show(&[path.to_str().unwrap(), "--columns", "z,y,x,orbit_id"]);
    assert!(ok);
    assert_eq!(lines(&stdout), vec!["z  y  x  orbit_id", "3  2  1  orb0"]);
}

/// The default is readable; `--full-precision` is exact. Both must be
/// true of the *same* value, or one of the two is lying.
#[test]
fn float_formatting_defaults_readable_and_goes_exact_on_demand() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(1)).unwrap();

    let (readable, _, ok) = show(&[path.to_str().unwrap(), "--columns", "vx", "--no-header"]);
    assert!(ok);
    assert_eq!(readable.trim(), "0.010203");

    let (exact, _, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "vx",
        "--no-header",
        "--full-precision",
    ]);
    assert!(ok);
    assert_eq!(exact.trim(), "0.0102030405060708");
    let back: f64 = exact.trim().parse().unwrap();
    assert_eq!(back, 0.010_203_040_506_070_8);
}

/// Six significant digits is the default, so a covariance entry that is
/// nine orders of magnitude below one still reads.
#[test]
fn small_covariance_entries_stay_legible() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(1)).unwrap();

    let (stdout, _, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "cov_00,cov_11,cov_55",
        "--no-header",
    ]);
    assert!(ok);
    assert_eq!(stdout.trim(), "1e-12   2e-12   6e-12");
}

// ---------------------------------------------------------------------
// Wide tables — the column window, driven non-interactively
// ---------------------------------------------------------------------

/// The orbits Parquet is 82 columns, most of them covariance. That width
/// is the reason the pager has a horizontal column window at all; here
/// the same selection is exercised through `--columns`.
#[test]
fn a_wide_table_is_readable_a_window_at_a_time() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(1)).unwrap();

    // Full width, header only, to establish the table really is wide.
    let (all, _, ok) = show(&[path.to_str().unwrap(), "--limit", "0"]);
    assert!(ok);
    let header: Vec<&str> = all.lines().next().unwrap().split_whitespace().collect();
    assert!(
        header.len() > 40,
        "fixture must be a wide table, got {} columns",
        header.len()
    );
    assert!(header.contains(&"cov_00"));
    assert!(header.contains(&"srp_amrat"));

    // A window over the far end of the table.
    let (window, _, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "cov_54,cov_55,covariance_representation",
        "--no-header",
    ]);
    assert!(ok);
    assert_eq!(window.trim(), "0       6e-12   cartesian");
}

// ---------------------------------------------------------------------
// Nulls, NaN, and empty tables
// ---------------------------------------------------------------------

/// Null and NaN must be distinguishable on screen: an object with no
/// identifier is not a number that could not be computed. The residual
/// tables carry the NaN side (an unevaluated χ²), the orbit tables the
/// null side (an orbit with no `object_id`).
#[test]
fn nulls_render_empty_and_nans_render_nan() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();

    let orbits = dir.path().join("states.parquet");
    write_orbits_parquet(&orbits, &orbit_batch(2)).unwrap();
    let (stdout, stderr, ok) = show(&[orbits.to_str().unwrap(), "--columns", "orbit_id,object_id"]);
    assert!(ok, "show failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec!["orbit_id  object_id", "orb0      99940", "orb1",],
        "an absent object_id is an empty cell, never the text `null`"
    );

    let resid = dir.path().join("residuals.parquet");
    write_residuals_parquet(&resid, &residuals()).unwrap();
    let (stdout, stderr, ok) = show(&[
        resid.to_str().unwrap(),
        "--columns",
        "chi2,probability,selected",
    ]);
    assert!(ok, "show failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec![
            "chi2  probability  selected",
            "1.5   0.47         true",
            "NaN   NaN          false",
        ],
        "an unevaluated χ² says NaN, never an empty cell"
    );
}

#[test]
fn nulls_and_nans_survive_csv_and_json_too() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();

    let csv = dir.path().join("residuals.csv");
    write_residuals_csv(&csv, &residuals()).unwrap();
    let json = dir.path().join("residuals.json");
    write_residuals_json(&json, &residuals()).unwrap();

    // CSV writes the literal `NaN`, so the distinction survives.
    let (stdout, stderr, ok) = show(&[csv.to_str().unwrap(), "--columns", "chi2,selected"]);
    assert!(ok, "show residuals.csv failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec!["chi2  selected", "1.5   true", "NaN   false"]
    );

    // JSON has no NaN literal, so the writer emits `null` for one. Reading
    // it back, an unevaluated χ² is indistinguishable from an absent one —
    // a property of the JSON *channel*, not of the reader, which shows
    // exactly what the file holds. Pinned here so the day the writer gains
    // a NaN encoding, this test says so.
    let (stdout, stderr, ok) = show(&[json.to_str().unwrap(), "--columns", "chi2,selected"]);
    assert!(ok, "show residuals.json failed: {stderr}");
    assert_eq!(
        lines(&stdout),
        vec!["chi2  selected", "1.5   true", "      false"],
        "JSON stores NaN as null, so it comes back as an empty cell"
    );

    // The null side, in the two text formats.
    let orbits_csv = dir.path().join("states.csv");
    write_orbits_csv(&orbits_csv, &orbit_batch(2)).unwrap();
    let orbits_json = dir.path().join("states.json");
    write_orbits_json(&orbits_json, &orbit_batch(2)).unwrap();
    for path in [&orbits_csv, &orbits_json] {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let (stdout, stderr, ok) =
            show(&[path.to_str().unwrap(), "--columns", "orbit_id,object_id"]);
        assert!(ok, "show {name} failed: {stderr}");
        assert_eq!(
            lines(&stdout),
            vec!["orbit_id  object_id", "orb0      99940", "orb1"],
            "{name}"
        );
    }
}

/// A zero-row table is a header and a count, not a crash and not silence.
#[test]
fn an_empty_table_prints_a_header_and_zero_rows() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();

    let parquet = dir.path().join("residuals.parquet");
    write_residuals_parquet(&parquet, &[]).unwrap();
    let (stdout, stderr, ok) = show(&[parquet.to_str().unwrap(), "--columns", "chi2,selected"]);
    assert!(ok, "an empty table must not fail: {stderr}");
    assert_eq!(lines(&stdout), vec!["chi2  selected", "0 rows"]);

    // An empty CSV is a zero-byte file: the writer emits no header, so
    // there is no schema to select from. It still reads as an empty
    // table rather than failing.
    let csv = dir.path().join("residuals.csv");
    write_residuals_csv(&csv, &[]).unwrap();
    assert_eq!(std::fs::read(&csv).unwrap().len(), 0);
    let (stdout, stderr, ok) = show(&[csv.to_str().unwrap()]);
    assert!(ok, "an empty csv must not fail: {stderr}");
    assert_eq!(lines(&stdout), vec!["0 rows"]);

    // Asking for a column of a file that declares none says why, rather
    // than printing "Available: " and nothing.
    let (_, stderr, ok) = show(&[csv.to_str().unwrap(), "--columns", "chi2"]);
    assert!(!ok);
    assert!(stderr.contains("declares no columns at all"), "{stderr}");
}

/// An empty JSON array carries no record to read a schema from, so there
/// is no header to print. It still reports its emptiness rather than
/// inventing columns or failing.
#[test]
fn an_empty_json_array_reports_zero_rows_with_no_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.json");
    std::fs::write(&path, "[]\n").unwrap();

    let (stdout, stderr, ok) = show(&[path.to_str().unwrap()]);
    assert!(ok, "{stderr}");
    assert_eq!(lines(&stdout), vec!["0 rows"]);
}

// ---------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------

#[test]
fn filter_keeps_only_matching_rows() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(3)).unwrap();

    let (stdout, stderr, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "orbit_id,object_id",
        "--filter",
        "ORB1",
    ]);
    assert!(ok, "{stderr}");
    assert_eq!(
        lines(&stdout),
        vec!["orbit_id  object_id", "orb1"],
        "the filter is case-insensitive and keeps the header"
    );
}

/// The filter tests the text on screen, so it finds NaN cells.
#[test]
fn filter_can_select_nan_rows() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("residuals.parquet");
    write_residuals_parquet(&path, &residuals()).unwrap();

    let (stdout, stderr, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "chi2,selected",
        "--filter",
        "nan",
    ]);
    assert!(ok, "{stderr}");
    assert_eq!(lines(&stdout), vec!["chi2  selected", "NaN   false"]);
}

#[test]
fn a_filter_matching_nothing_still_prints_the_header() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(2)).unwrap();

    let (stdout, _, ok) = show(&[
        path.to_str().unwrap(),
        "--columns",
        "orbit_id",
        "--filter",
        "nothing-matches-this",
    ]);
    assert!(ok);
    assert_eq!(lines(&stdout), vec!["orbit_id", "0 rows"]);
}

// ---------------------------------------------------------------------
// Directory listing
// ---------------------------------------------------------------------

#[test]
fn a_directory_lists_its_artifacts_with_descriptions() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    write_orbits_parquet(dir.path().join("states.parquet"), &orbit_batch(3)).unwrap();
    write_residuals_parquet(dir.path().join("residuals.parquet"), &residuals()).unwrap();
    // A non-artifact in the same directory must not stop the listing.
    std::fs::write(dir.path().join("notes.txt"), "not a table").unwrap();

    let (stdout, stderr, ok) = show(&["--out-dir", dir.path().to_str().unwrap()]);
    assert!(ok, "listing failed: {stderr}");

    let rows = lines(&stdout);
    assert_eq!(rows[0], dir.path().display().to_string());
    assert_eq!(rows[1], "");
    assert!(rows[2].starts_with("FILE"), "{:?}", rows[2]);
    assert!(
        rows[3].starts_with("residuals.parquet  2 "),
        "row count comes from the parquet footer: {:?}",
        rows[3]
    );
    assert!(
        rows[3].contains("Per-observation residuals"),
        "known artifacts get a description: {:?}",
        rows[3]
    );
    assert!(
        rows[4].starts_with("states.parquet     3 "),
        "{:?}",
        rows[4]
    );
    assert!(rows[4].contains("Propagated states"), "{:?}", rows[4]);
    assert_eq!(rows.len(), 5, "notes.txt must be skipped, not listed");
}

/// Piped, the listing is plain output with no picker numbers — there is
/// nothing to type a number into.
#[test]
fn a_piped_listing_has_no_picker() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    write_orbits_parquet(dir.path().join("states.parquet"), &orbit_batch(1)).unwrap();

    let (stdout, _, ok) = show(&[dir.path().to_str().unwrap()]);
    assert!(ok);
    assert!(!stdout.contains("Open ["), "piped output must not prompt");
    assert!(!stdout.contains(" 1."), "piped output must not be numbered");
}

/// A bare directory positional and `--out-dir` are the same thing.
#[test]
fn out_dir_and_the_positional_directory_agree() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    write_orbits_parquet(dir.path().join("states.parquet"), &orbit_batch(1)).unwrap();

    let (a, _, ok_a) = show(&[dir.path().to_str().unwrap()]);
    let (b, _, ok_b) = show(&["--out-dir", dir.path().to_str().unwrap()]);
    assert!(ok_a && ok_b);
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------
// Errors — every one names the file and what was tried
// ---------------------------------------------------------------------

#[test]
fn a_missing_file_fails_by_name() {
    let (_, stderr, ok) = show(&["/definitely/not/here/states.parquet"]);
    assert!(!ok, "a missing file must fail");
    assert!(
        stderr.contains("/definitely/not/here/states.parquet"),
        "{stderr}"
    );
    assert!(stderr.contains("no such file"), "{stderr}");
}

#[test]
fn an_unknown_format_fails_by_name_and_says_what_is_supported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mystery.bin");
    std::fs::write(&path, [0x00_u8, 0x01, 0x02, 0x03]).unwrap();

    let (_, stderr, ok) = show(&[path.to_str().unwrap()]);
    assert!(!ok);
    assert!(stderr.contains("mystery.bin"), "{stderr}");
    assert!(stderr.contains(".parquet"), "{stderr}");
    assert!(stderr.contains("\\x00"), "the bytes seen: {stderr}");
}

/// A file named `.parquet` that is not Parquet is a mislabelled file, and
/// saying so is more useful than guessing at its real format.
#[test]
fn a_mislabelled_parquet_fails_loudly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    std::fs::write(&path, "orbit_id,x\norb0,1\n").unwrap();

    let (_, stderr, ok) = show(&[path.to_str().unwrap()]);
    assert!(!ok);
    assert!(stderr.contains("states.parquet"), "{stderr}");
    assert!(stderr.contains("mislabelled or truncated"), "{stderr}");
}

/// A truncated Parquet has the magic but no readable footer.
#[test]
fn a_corrupt_parquet_fails_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    std::fs::write(&path, b"PAR1\x00\x00\x00\x00garbage").unwrap();

    let (_, stderr, ok) = show(&[path.to_str().unwrap()]);
    assert!(!ok);
    assert!(stderr.contains("states.parquet"), "{stderr}");
    assert!(stderr.contains("parquet footer"), "{stderr}");
}

#[test]
fn an_unknown_column_names_the_columns_that_exist() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("states.parquet");
    write_orbits_parquet(&path, &orbit_batch(1)).unwrap();

    let (_, stderr, ok) = show(&[path.to_str().unwrap(), "--columns", "not_a_column"]);
    assert!(!ok);
    assert!(
        stderr.contains("no column named `not_a_column`"),
        "{stderr}"
    );
    assert!(stderr.contains("orbit_id"), "{stderr}");
}

#[test]
fn a_directory_with_no_artifacts_says_so() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), "nothing here").unwrap();

    let (_, stderr, ok) = show(&[dir.path().to_str().unwrap()]);
    assert!(!ok);
    assert!(stderr.contains("no readable artifacts"), "{stderr}");
    assert!(stderr.contains(".parquet"), "{stderr}");
}

/// A CSV row that does not match its header is a corrupt file, not a row
/// to pad or skip.
#[test]
fn a_ragged_csv_fails_naming_the_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ragged.csv");
    std::fs::write(&path, "a,b,c\n1,2,3\n4,5\n").unwrap();

    let (_, stderr, ok) = show(&[path.to_str().unwrap()]);
    assert!(!ok);
    assert!(stderr.contains("ragged.csv:3"), "{stderr}");
    assert!(stderr.contains("expected 3 fields"), "{stderr}");
}

/// A JSON record with a key the header does not have would have to be
/// dropped to render. It is reported instead.
#[test]
fn json_schema_drift_is_reported_rather_than_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("drift.jsonl");
    std::fs::write(&path, "{\"a\": 1}\n{\"a\": 2, \"b\": 3}\n").unwrap();

    let (_, stderr, ok) = show(&[path.to_str().unwrap()]);
    assert!(!ok, "a dropped column must not pass silently");
    assert!(stderr.contains("drift.jsonl"), "{stderr}");
    assert!(stderr.contains("record 2"), "{stderr}");
    assert!(stderr.contains("`b`"), "{stderr}");
}

// ---------------------------------------------------------------------
// Reader-side behaviour that needs no engine
// ---------------------------------------------------------------------

/// JSONL and a pretty-printed JSON array are the same table.
#[test]
fn jsonl_and_a_json_array_read_identically() {
    let dir = tempfile::tempdir().unwrap();
    let jsonl = dir.path().join("t.jsonl");
    let array = dir.path().join("t.json");
    std::fs::write(
        &jsonl,
        "{\"a\": 1, \"b\": \"x\"}\n{\"a\": 2, \"b\": \"y\"}\n",
    )
    .unwrap();
    std::fs::write(
        &array,
        "[\n  {\n    \"a\": 1,\n    \"b\": \"x\"\n  },\n  {\n    \"a\": 2,\n    \"b\": \"y\"\n  }\n]\n",
    )
    .unwrap();

    let (a, _, ok_a) = show(&[jsonl.to_str().unwrap()]);
    let (b, _, ok_b) = show(&[array.to_str().unwrap()]);
    assert!(ok_a && ok_b);
    assert_eq!(lines(&a), vec!["a  b", "1  x", "2  y"]);
    assert_eq!(a, b);
}

/// A quoted CSV field containing a comma and a newline is one field of
/// one row.
#[test]
fn quoted_csv_fields_survive_the_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quoted.csv");
    std::fs::write(&path, "id,note\n1,\"a,b\"\n2,\"line1\nline2\"\n").unwrap();

    let (stdout, stderr, ok) = show(&[path.to_str().unwrap(), "--columns", "id"]);
    assert!(ok, "{stderr}");
    assert_eq!(
        lines(&stdout),
        vec!["id", "1", "2"],
        "an embedded newline must not split the row"
    );
}

/// A JSON file whose first record fixes a column the later records omit
/// renders the absence as a null.
#[test]
fn a_missing_json_key_renders_as_a_null_cell() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse.jsonl");
    std::fs::write(&path, "{\"a\": 1, \"b\": 2}\n{\"a\": 3}\n").unwrap();

    let (stdout, stderr, ok) = show(&[path.to_str().unwrap()]);
    assert!(ok, "{stderr}");
    assert_eq!(lines(&stdout), vec!["a  b", "1  2", "3"]);
}

/// Streaming, not slurping: a file far larger than any page renders its
/// first rows and its `--limit` promptly, and the row count is right.
#[test]
fn a_large_file_streams() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.csv");
    let mut body = String::from("i,v\n");
    for i in 0..200_000 {
        body.push_str(&format!("{i},{}\n", i as f64 * 0.5));
    }
    std::fs::write(&path, body).unwrap();

    let started = std::time::Instant::now();
    let (head, stderr, ok) = show(&[path.to_str().unwrap(), "--limit", "3"]);
    let elapsed = started.elapsed();
    assert!(ok, "{stderr}");
    assert_eq!(lines(&head), vec!["i  v", "0  0", "1  0.5", "2  1"]);
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "--limit 3 on a 200k-row file must not read the whole file (took {elapsed:?})"
    );

    let (all, _, ok) = show(&[path.to_str().unwrap(), "--columns", "i", "--no-header"]);
    assert!(ok);
    assert_eq!(all.lines().count(), 200_000);
}

/// The row count in a directory listing comes from the Parquet footer,
/// so it is exact and costs no decoding.
#[test]
fn parquet_row_counts_in_a_listing_are_exact() {
    require_engine!();
    let dir = tempfile::tempdir().unwrap();
    write_orbits_parquet(dir.path().join("states.parquet"), &orbit_batch(37)).unwrap();

    let (stdout, _, ok) = show(&[dir.path().to_str().unwrap()]);
    assert!(ok);
    assert!(
        stdout.contains("states.parquet  37"),
        "expected an exact count: {stdout}"
    );
}
