//! Covariance and uncertainty math utilities.

use crate::error::{Error, Result};

/// Compute the largest eigenvalue and corresponding eigenvector of a
/// 6×6 symmetric matrix.
pub fn eigenvector_max_6x6(matrix: &[[f64; 6]; 6]) -> Result<(f64, [f64; 6])> {
    let mut eigenvalue: f64 = 0.0;
    let mut eigenvector: [f64; 6] = [0.0; 6];
    let code = unsafe {
        empyrean_sys::empyrean_eigenvector_max_6x6(matrix, &mut eigenvalue, &mut eigenvector)
    };
    if code != 0 {
        return Err(Error::capture(code));
    }
    Ok((eigenvalue, eigenvector))
}

/// A component of a Gaussian mixture produced by [`split_gaussian`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixtureComponent {
    /// Component weight (sums to 1 across the mixture).
    pub weight: f64,
    /// Component mean (6-element state vector).
    pub mean: [f64; 6],
    /// Component 6x6 covariance.
    pub covariance: [[f64; 6]; 6],
}

/// Split a Gaussian (mean, covariance) into a mixture of `k` smaller
/// Gaussians along the eigenvector of maximum variance.
///
/// Returns a mixture of `k` components whose weighted moments match the
/// input mean and covariance to second order.
pub fn split_gaussian(
    mean: &[f64; 6],
    covariance: &[[f64; 6]; 6],
    k: usize,
) -> Result<Vec<MixtureComponent>> {
    let mut weights: Vec<f64> = vec![0.0; k];
    let mut means: Vec<[f64; 6]> = vec![[0.0; 6]; k];
    let mut covs: Vec<[[f64; 6]; 6]> = vec![[[0.0; 6]; 6]; k];

    let code = unsafe {
        empyrean_sys::empyrean_split_gaussian(
            mean,
            covariance,
            k,
            weights.as_mut_ptr(),
            means.as_mut_ptr(),
            covs.as_mut_ptr(),
        )
    };
    if code != 0 {
        return Err(Error::capture(code));
    }

    Ok((0..k)
        .map(|i| MixtureComponent {
            weight: weights[i],
            mean: means[i],
            covariance: covs[i],
        })
        .collect())
}
