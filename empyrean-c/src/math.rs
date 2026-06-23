//! C-compatible math utilities.

use std::panic::AssertUnwindSafe;

use empyrean_core::uncertainty::{eigenvector_max_6x6, split_gaussian};

use crate::set_last_error;

/// Find the dominant eigenvalue and eigenvector of a 6x6 symmetric matrix.
///
/// Returns 0 on success. `eigenvalue_out` receives the eigenvalue,
/// `eigenvector_out` receives the 6-element eigenvector.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_eigenvector_max_6x6(
    matrix: *const [[f64; 6]; 6],
    eigenvalue_out: *mut f64,
    eigenvector_out: *mut [f64; 6],
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if matrix.is_null() || eigenvalue_out.is_null() || eigenvector_out.is_null() {
            return -1;
        }
        let m = unsafe { &*matrix };
        let (eigenvector, eigenvalue) = eigenvector_max_6x6(m);
        unsafe {
            *eigenvalue_out = eigenvalue;
            *eigenvector_out = eigenvector;
        }
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_eigenvector_max_6x6");
            -99
        }
    }
}

/// Split a 6D Gaussian into K weighted components along the dominant
/// eigenvector of the covariance.
///
/// `weights_out`, `means_out`, `covariances_out` must point to arrays
/// of size K, K×6, and K×6×6 respectively. Returns 0 on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_split_gaussian(
    mean: *const [f64; 6],
    covariance: *const [[f64; 6]; 6],
    k: usize,
    weights_out: *mut f64,
    means_out: *mut [f64; 6],
    covariances_out: *mut [[f64; 6]; 6],
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if mean.is_null()
            || covariance.is_null()
            || weights_out.is_null()
            || means_out.is_null()
            || covariances_out.is_null()
            || k == 0
        {
            return -1;
        }

        let m = unsafe { &*mean };
        let c = unsafe { &*covariance };

        // split_gaussian needs a direction vector along which to split
        // the distribution. The dominant covariance eigenvector is the
        // standard choice (preserves the most uncertainty mass per
        // component) — match villeneuve's AGM splitter behavior.
        let (direction, _) = eigenvector_max_6x6(c);
        let components = match split_gaussian(m, c, &direction, k) {
            Ok(components) => components,
            Err(e) => {
                set_last_error(&format!("split_gaussian failed: {e:?}"));
                return -1;
            }
        };

        let weights_slice = unsafe { std::slice::from_raw_parts_mut(weights_out, k) };
        let means_slice = unsafe { std::slice::from_raw_parts_mut(means_out, k) };
        let covs_slice = unsafe { std::slice::from_raw_parts_mut(covariances_out, k) };

        for (i, (w, mean, cov)) in components.into_iter().enumerate() {
            weights_slice[i] = w;
            means_slice[i] = mean;
            covs_slice[i] = cov;
        }

        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_split_gaussian");
            -99
        }
    }
}
