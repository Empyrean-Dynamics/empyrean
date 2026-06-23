//! Residual I/O — write across parquet, JSON, and CSV.
//!
//! Orbit determination produces residuals via
//! [`DetermineResult`](crate::DetermineResult) /
//! [`EvaluateResult`](crate::EvaluateResult); only a write surface is
//! exposed here.

use std::path::Path;

use crate::error::{Error, Result};
use crate::od::ObservationResidual;

use super::path_to_cstring;

/// Build a C-ABI residuals array from the safe-Rust [`ObservationResidual`].
///
/// String fields (obs_id, ast_cat) are heap-allocated and parked in
/// `keep_strings` so the FFI pointers stay valid for the lifetime of
/// the returned Vec. Drop the keepalive after the FFI call returns.
fn residuals_to_ffi_array(
    residuals: &[ObservationResidual],
    keep_strings: &mut Vec<std::ffi::CString>,
) -> Result<Vec<empyrean_sys::EmpyreanObservationResult>> {
    fn opt_str_ptr(s: &str, keep: &mut Vec<std::ffi::CString>) -> *mut std::ffi::c_char {
        if s.is_empty() {
            return std::ptr::null_mut();
        }
        match std::ffi::CString::new(s) {
            Ok(c) => {
                let p = c.as_ptr() as *mut std::ffi::c_char;
                keep.push(c);
                p
            }
            Err(_) => std::ptr::null_mut(),
        }
    }
    let rejection_reason_to_int = |r: crate::od::RejectionReason| -> i32 {
        use crate::od::RejectionReason as R;
        match r {
            R::Accepted => 0,
            R::ChiSquared => 1,
            R::SigmaClip => 2,
            R::CooksDistance => 3,
            R::Adaptive => 4,
            R::UnsupportedObservatory => 5,
            R::CMC2003 => 6,
            R::RadarObservationsUnsupported => 7,
            R::OccultationObservationsUnsupported => 8,
            R::OutsideArc => 9,
            R::NotEvaluated => -1,
        }
    };
    residuals
        .iter()
        .map(|r| {
            let mut code = [0u8; 4];
            let bytes = r.obs_code.as_bytes();
            let take = bytes.len().min(3);
            code[..take].copy_from_slice(&bytes[..take]);
            Ok(empyrean_sys::EmpyreanObservationResult {
                obs_id: opt_str_ptr(&r.obs_id, keep_strings),
                obs_code: code,
                ast_cat: opt_str_ptr(r.ast_cat.as_deref().unwrap_or(""), keep_strings),
                epoch_mjd_tdb: r.epoch.mjd_tdb()?,
                ra_residual_arcsec: r.ra_residual_arcsec,
                dec_residual_arcsec: r.dec_residual_arcsec,
                chi2: r.chi2,
                dof: r.dof,
                probability: r.probability,
                selected: if r.selected { 1 } else { 0 },
                residual_cov_ra: r.residual_cov_ra,
                residual_cov_dec: r.residual_cov_dec,
                residual_cov_corr: r.residual_cov_corr,
                rejection_reason: rejection_reason_to_int(r.rejection_reason),
                rejection_criterion: r.rejection_criterion,
                rejection_threshold: r.rejection_threshold,
                rejection_effective_threshold: r.rejection_effective_threshold,
                rejection_information_loss: r.rejection_information_loss,
                cooks_distance: r.cooks_distance,
                leverage: r.leverage,
                fractional_information: r.fractional_information,
                along_track_arcsec: r.along_track_arcsec,
                cross_track_arcsec: r.cross_track_arcsec,
                along_track_error_arcsec: r.along_track_error_arcsec,
                cross_track_error_arcsec: r.cross_track_error_arcsec,
                track_position_angle_deg: r.track_position_angle_deg,
            })
        })
        .collect()
}

fn write_residuals_via<F>(path: &Path, residuals: &[ObservationResidual], c_call: F) -> Result<()>
where
    F: FnOnce(
        *const std::ffi::c_char,
        *const empyrean_sys::EmpyreanObservationResult,
        usize,
    ) -> i32,
{
    let path_c = path_to_cstring(path)?;
    let mut keep: Vec<std::ffi::CString> = Vec::new();
    let ffi = residuals_to_ffi_array(residuals, &mut keep)?;
    let code = c_call(path_c.as_ptr(), ffi.as_ptr(), ffi.len());
    drop(keep); // strings outlived the C call; safe to free now
    if code != 0 {
        return Err(Error::capture(code));
    }
    Ok(())
}

/// Write OD residuals to a parquet file.
pub fn write_residuals_parquet(
    path: impl AsRef<Path>,
    residuals: &[ObservationResidual],
) -> Result<()> {
    write_residuals_via(path.as_ref(), residuals, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_residuals_write_parquet(p, ptr, n)
    })
}

/// Write OD residuals to JSON.
pub fn write_residuals_json(
    path: impl AsRef<Path>,
    residuals: &[ObservationResidual],
) -> Result<()> {
    write_residuals_via(path.as_ref(), residuals, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_residuals_write_json(p, ptr, n)
    })
}

/// Write OD residuals to CSV.
pub fn write_residuals_csv(
    path: impl AsRef<Path>,
    residuals: &[ObservationResidual],
) -> Result<()> {
    write_residuals_via(path.as_ref(), residuals, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_residuals_write_csv(p, ptr, n)
    })
}
