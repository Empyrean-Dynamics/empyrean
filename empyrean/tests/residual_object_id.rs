//! A residual row knows which object it belongs to, all the way to disk.
//!
//! Batch orbit determination fits one object per group and the CLI writes
//! every group's residuals into one flat file. Without an `object_id` on
//! the row that file is unattributable — the exact silent loss the batch
//! surface exists to prevent — so the column is asserted here at the
//! written-file boundary, not just in memory.
//!
//! Needs a `libempyrean` matching this source tree (the writers live
//! behind the C ABI). A mismatched or absent engine => skip.

use empyrean::{Epoch, ObservationResidual, RejectionReason, write_residuals_csv};

/// A residual row with only the fields the CSV schema reads populated;
/// everything else is the NaN / not-evaluated marker the ABI uses.
fn residual(obs_id: &str, object_id: Option<&str>, ra: f64) -> ObservationResidual {
    ObservationResidual {
        obs_id: obs_id.to_string(),
        object_id: object_id.map(str::to_string),
        obs_code: "703".to_string(),
        ast_cat: None,
        epoch: Epoch::from_mjd_tdb(60320.0),
        ra_residual_arcsec: ra,
        dec_residual_arcsec: -0.12,
        chi2: 1.4,
        dof: 2,
        probability: 0.5,
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

/// Rows from two different objects stay distinguishable in one file.
#[test]
fn residuals_csv_carries_the_object_id_of_each_row() {
    let dir = std::env::temp_dir().join(format!("empyrean-residual-oid-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("residuals.csv");

    let rows = vec![
        residual("obs-1", Some("2024 YR4"), 0.31),
        residual("obs-2", Some("433"), -0.08),
        // A single-object evaluate / refine row carries no grouping key.
        residual("obs-3", None, 0.02),
    ];

    if let Err(e) = write_residuals_csv(&path, &rows) {
        eprintln!(
            "skipping: residual writer unavailable ({}): {e}",
            path.display()
        );
        std::fs::remove_dir_all(&dir).ok();
        return;
    }

    let csv = std::fs::read_to_string(&path).expect("read residuals.csv");
    let mut lines = csv.lines();
    let header = lines.next().expect("residuals.csv must have a header");
    assert!(
        header.starts_with("object_id"),
        "object_id leads the residual schema so a batch file reads object-first: {header}"
    );

    let data: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(data.len(), 3, "every row is written\n{csv}");
    assert!(data[0].starts_with("2024 YR4,"), "{}", data[0]);
    assert!(data[1].starts_with("433,"), "{}", data[1]);
    // No key means an empty field — never another object's id.
    assert!(
        data[2].starts_with(','),
        "unkeyed row must be blank, not borrowed: {}",
        data[2]
    );

    std::fs::remove_dir_all(&dir).ok();
}
