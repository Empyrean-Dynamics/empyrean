//! Per-object fit-summary I/O — write across parquet, JSON, and CSV.
//!
//! A batch [`Context::determine`](crate::Context::determine) attempts one
//! fit per ADES object. The fit summary is the table of what happened to
//! each of them: one row per **input** object, whether or not it produced
//! an orbit. It is the artifact that makes a partially successful batch
//! readable — the orbit file only holds the objects that delivered.

use std::path::Path;

use crate::error::{Error, Result};
use crate::od::{DetermineEntry, DetermineResults};

use super::path_to_cstring;

/// One row of the fit summary: what orbit determination did with one
/// object.
///
/// Numeric fields are NaN when the quantity does not exist — a failed
/// object has no RMS, and a gate that could not be computed has no
/// value. They are never 0.0, which would read as a measurement at the
/// floor.
#[derive(Debug, Clone, PartialEq)]
pub struct FitSummaryRow {
    /// ADES object identifier.
    pub object_id: String,
    /// `"delivered"` or `"failed"`.
    pub status: String,
    /// Did the differential correction reach its stopping criterion?
    pub converged: bool,
    /// DC iterations used.
    pub iterations: u32,
    /// Observations this object contributed.
    pub n_obs: usize,
    /// Observations the fit retained.
    pub n_selected: usize,
    /// RA·cos(Dec) residual RMS (arcsec).
    pub rms_ra_arcsec: f64,
    /// Dec residual RMS (arcsec).
    pub rms_dec_arcsec: f64,
    /// Reduced χ² of the fit.
    pub reduced_chi2: f64,
    /// Aggregate fit-quality verdict.
    pub fit_acceptable: bool,
    /// Aggregate verdict on forward extrapolation: `fit_acceptable` AND
    /// the four selection / coverage axes below.
    pub extrapolation_acceptable: bool,
    /// Did the fit retain enough of its input?
    pub selection_fraction_ok: bool,
    /// Fraction of observations retained.
    pub selection_fraction: f64,
    /// Minimum retained fraction the gate required.
    pub selection_fraction_threshold: f64,
    /// Do the selected observations still span enough of the arc?
    pub selected_arc_coverage_ok: bool,
    /// Arc span over the selected observations only (days).
    pub selected_arc_days: f64,
    /// Selected-span / full-span ratio.
    pub selected_arc_fraction: f64,
    /// Minimum span ratio the gate required.
    pub selected_arc_fraction_threshold: f64,
    /// Were the most-recent observations kept?
    pub trailing_gap_ok: bool,
    /// Days between the last selected and the last full-arc observation.
    pub trailing_gap_days: f64,
    /// Largest trailing gap the gate allowed (days).
    pub trailing_gap_threshold_days: f64,
    /// Did σₐ / |a| pass its threshold?
    pub fractional_sigma_a_ok: bool,
    /// Measured σₐ / |a|.
    pub fractional_sigma_a: f64,
    /// Threshold for σₐ / |a|.
    pub fractional_sigma_a_threshold: f64,
    /// Width of the solved-parameter set (6 for a state-only fit).
    pub solve_for_width: u32,
    /// Failure message; `None` on a delivered object.
    pub error: Option<String>,
}

impl FitSummaryRow {
    /// The row describing one slot of a batch determine.
    pub fn from_entry(entry: &DetermineEntry) -> Self {
        match &entry.outcome {
            Ok(fit) => {
                let a = &fit.acceptability;
                Self {
                    object_id: entry.object_id.clone(),
                    status: "delivered".to_string(),
                    converged: fit.converged,
                    iterations: fit.iterations,
                    n_obs: fit.summary.num_obs,
                    n_selected: fit.summary.num_selected,
                    rms_ra_arcsec: fit.summary.rms_ra_arcsec,
                    rms_dec_arcsec: fit.summary.rms_dec_arcsec,
                    reduced_chi2: fit.summary.reduced_chi2,
                    fit_acceptable: a.fit_acceptable,
                    extrapolation_acceptable: a.extrapolation_acceptable,
                    selection_fraction_ok: a.selection_fraction_ok,
                    selection_fraction: a.selection_fraction_value,
                    selection_fraction_threshold: a.selection_fraction_threshold,
                    selected_arc_coverage_ok: a.selected_arc_coverage_ok,
                    selected_arc_days: a.selected_arc_days_value,
                    selected_arc_fraction: a.selected_arc_fraction_value,
                    selected_arc_fraction_threshold: a.selected_arc_fraction_threshold,
                    trailing_gap_ok: a.trailing_gap_ok,
                    trailing_gap_days: a.trailing_gap_days_value,
                    trailing_gap_threshold_days: a.trailing_gap_threshold,
                    fractional_sigma_a_ok: a.fractional_sigma_a_ok,
                    fractional_sigma_a: a.fractional_sigma_a_value,
                    fractional_sigma_a_threshold: a.fractional_sigma_a_threshold,
                    // A state-only fit reports no tagged solved
                    // covariance; its solved width is the 6-element state.
                    solve_for_width: fit
                        .solved_covariance
                        .as_ref()
                        .map(|sc| sc.width as u32)
                        .unwrap_or(6),
                    error: None,
                }
            }
            Err(failure) => Self {
                object_id: entry.object_id.clone(),
                status: "failed".to_string(),
                converged: false,
                iterations: 0,
                n_obs: 0,
                n_selected: 0,
                rms_ra_arcsec: f64::NAN,
                rms_dec_arcsec: f64::NAN,
                reduced_chi2: f64::NAN,
                fit_acceptable: false,
                extrapolation_acceptable: false,
                selection_fraction_ok: false,
                selection_fraction: f64::NAN,
                selection_fraction_threshold: f64::NAN,
                selected_arc_coverage_ok: false,
                selected_arc_days: f64::NAN,
                selected_arc_fraction: f64::NAN,
                selected_arc_fraction_threshold: f64::NAN,
                trailing_gap_ok: false,
                trailing_gap_days: f64::NAN,
                trailing_gap_threshold_days: f64::NAN,
                fractional_sigma_a_ok: false,
                fractional_sigma_a: f64::NAN,
                fractional_sigma_a_threshold: f64::NAN,
                solve_for_width: 0,
                error: Some(failure.message.clone()),
            },
        }
    }

    /// One row per object in a batch, in table order.
    pub fn from_results(results: &DetermineResults) -> Vec<Self> {
        results.iter().map(Self::from_entry).collect()
    }
}

/// Build the C-ABI array. The strings are parked in `keep` so the
/// borrowed pointers stay valid for the duration of the FFI call.
fn rows_to_ffi_array(
    rows: &[FitSummaryRow],
    keep: &mut Vec<std::ffi::CString>,
) -> Result<Vec<empyrean_sys::EmpyreanFitSummary>> {
    fn str_ptr(s: &str, keep: &mut Vec<std::ffi::CString>) -> *const std::ffi::c_char {
        match std::ffi::CString::new(s) {
            Ok(c) => {
                let p = c.as_ptr();
                keep.push(c);
                p
            }
            Err(_) => std::ptr::null(),
        }
    }
    Ok(rows
        .iter()
        .map(|r| empyrean_sys::EmpyreanFitSummary {
            object_id: str_ptr(&r.object_id, keep),
            status: str_ptr(&r.status, keep),
            converged: u8::from(r.converged),
            iterations: r.iterations,
            n_obs: r.n_obs,
            n_selected: r.n_selected,
            rms_ra_arcsec: r.rms_ra_arcsec,
            rms_dec_arcsec: r.rms_dec_arcsec,
            reduced_chi2: r.reduced_chi2,
            fit_acceptable: u8::from(r.fit_acceptable),
            extrapolation_acceptable: u8::from(r.extrapolation_acceptable),
            selection_fraction_ok: u8::from(r.selection_fraction_ok),
            selection_fraction: r.selection_fraction,
            selection_fraction_threshold: r.selection_fraction_threshold,
            selected_arc_coverage_ok: u8::from(r.selected_arc_coverage_ok),
            selected_arc_days: r.selected_arc_days,
            selected_arc_fraction: r.selected_arc_fraction,
            selected_arc_fraction_threshold: r.selected_arc_fraction_threshold,
            trailing_gap_ok: u8::from(r.trailing_gap_ok),
            trailing_gap_days: r.trailing_gap_days,
            trailing_gap_threshold_days: r.trailing_gap_threshold_days,
            fractional_sigma_a_ok: u8::from(r.fractional_sigma_a_ok),
            fractional_sigma_a: r.fractional_sigma_a,
            fractional_sigma_a_threshold: r.fractional_sigma_a_threshold,
            solve_for_width: r.solve_for_width,
            error: match &r.error {
                Some(m) => str_ptr(m, keep),
                None => std::ptr::null(),
            },
        })
        .collect())
}

fn write_via<F>(path: &Path, rows: &[FitSummaryRow], c_call: F) -> Result<()>
where
    F: FnOnce(*const std::ffi::c_char, *const empyrean_sys::EmpyreanFitSummary, usize) -> i32,
{
    let path_c = path_to_cstring(path)?;
    let mut keep: Vec<std::ffi::CString> = Vec::new();
    let ffi = rows_to_ffi_array(rows, &mut keep)?;
    let code = c_call(path_c.as_ptr(), ffi.as_ptr(), ffi.len());
    drop(keep); // strings outlived the C call; safe to free now
    if code != 0 {
        return Err(Error::capture(code));
    }
    Ok(())
}

/// Write the per-object fit summary to a parquet file.
pub fn write_fit_summary_parquet(path: impl AsRef<Path>, rows: &[FitSummaryRow]) -> Result<()> {
    write_via(path.as_ref(), rows, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_fit_summary_write_parquet(p, ptr, n)
    })
}

/// Write the per-object fit summary to JSON.
pub fn write_fit_summary_json(path: impl AsRef<Path>, rows: &[FitSummaryRow]) -> Result<()> {
    write_via(path.as_ref(), rows, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_fit_summary_write_json(p, ptr, n)
    })
}

/// Write the per-object fit summary to CSV.
pub fn write_fit_summary_csv(path: impl AsRef<Path>, rows: &[FitSummaryRow]) -> Result<()> {
    write_via(path.as_ref(), rows, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_fit_summary_write_csv(p, ptr, n)
    })
}
