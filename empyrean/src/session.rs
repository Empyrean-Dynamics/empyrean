//! Stateful orbit-determination session.
//!
//! Mirrors `empyrean_core::determination::Session` (re-exported from
//! scott's session module). A [`Session`] owns observations, a mask of
//! currently-disabled observations, and a fit history. The typical
//! workflow is *fit → look at residuals → mask one bad night → re-fit
//! → compare χ²* — see the upstream Session docs for the rationale.
//!
//! ```no_run
//! use empyrean::{Context, ODConfig, Session};
//!
//! let ctx = Context::from_data_dir(None)?;
//! let observations = ctx.read_ades("apophis.psv")?;
//! let mut sess = Session::new(observations, ODConfig::default())?;
//! let _first = sess.refine(&ctx)?;
//! sess.mask(7)?;            // drop the 7th observation
//! let _second = sess.refine(&ctx)?;
//! let diff = sess.diff(0)?; // compare current fit to the first fit
//! println!("Δχ²_red = {:+.3}", diff.reduced_chi2_delta);
//! # Ok::<(), empyrean::Error>(())
//! ```

use crate::context::Context;
use crate::error::{Error, Result};
use crate::od::{DetermineResult, ODConfig, Observations};
use std::ptr::NonNull;

/// Pairwise diagnostic between two fits in the same session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionDiff {
    /// Δ reduced χ² (positive ⇒ current fit is worse than prior).
    pub reduced_chi2_delta: f64,
    /// Δ iteration count.
    pub iterations_delta: i64,
    /// Δ number of observations used (negative ⇒ observations were
    /// masked between prior and current).
    pub n_observations_delta: i64,
    /// Final update-norm convergence metric on the current fit.
    pub update_norm_current: f64,
    /// Final update-norm convergence metric on the prior fit.
    pub update_norm_prior: f64,
}

impl SessionDiff {
    fn from_ffi(d: &empyrean_sys::EmpyreanSessionDiff) -> Self {
        Self {
            reduced_chi2_delta: d.reduced_chi2_delta,
            iterations_delta: d.iterations_delta,
            n_observations_delta: d.n_observations_delta,
            update_norm_current: d.update_norm_current,
            update_norm_prior: d.update_norm_prior,
        }
    }
}

/// Stateful orbit-determination handle.
///
/// Owns the observation set, the mask state, and the fit history.
/// The wrapper holds an opaque pointer; the underlying state lives on
/// the C ABI heap and is freed when [`Session`] is dropped.
pub struct Session {
    raw: NonNull<empyrean_sys::EmpyreanSession>,
}

// Session is Send: the underlying scott::session::Session uses
// thread-local-free state. Not Sync — concurrent mutation is not
// safe; share via `Arc<Mutex<Session>>` if multiple threads need to
// touch one session.
unsafe impl Send for Session {}

impl Session {
    /// Construct a session over a fixed observation set.
    pub fn new(observations: Observations, config: ODConfig) -> Result<Self> {
        let (obs_ptr, obs_len) = observations.as_ffi_slice();
        let (ffi_config, _perturbers_keep) = config.to_ffi_with();
        let raw = unsafe { empyrean_sys::empyrean_session_new(obs_ptr, obs_len, &ffi_config) };
        // Session takes ownership of a copy of the observation array's
        // contents (via c_observations_to_optical → Vec<OpticalObservation>).
        // The original `Observations` lifetime is unchanged.
        let _ = observations;
        NonNull::new(raw)
            .map(|raw| Session { raw })
            .ok_or_else(Error::from_null_ptr)
    }

    /// Total observations in the session, masked or not.
    pub fn n_observations(&self) -> usize {
        unsafe { empyrean_sys::empyrean_session_n_observations(self.raw.as_ptr()) }
    }

    /// Number of observations currently masked.
    pub fn n_masked(&self) -> usize {
        unsafe { empyrean_sys::empyrean_session_n_masked(self.raw.as_ptr()) }
    }

    /// Number of observations active (not masked) in the next refine.
    pub fn n_active(&self) -> usize {
        unsafe { empyrean_sys::empyrean_session_n_active(self.raw.as_ptr()) }
    }

    /// Mask observation `idx`. Errors if out of bounds.
    pub fn mask(&mut self, idx: usize) -> Result<()> {
        let code = unsafe { empyrean_sys::empyrean_session_mask(self.raw.as_ptr(), idx) };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(())
    }

    /// Unmask observation `idx`. Errors if out of bounds.
    pub fn unmask(&mut self, idx: usize) -> Result<()> {
        let code = unsafe { empyrean_sys::empyrean_session_unmask(self.raw.as_ptr(), idx) };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(())
    }

    /// Clear all masks.
    pub fn unmask_all(&mut self) -> Result<()> {
        let code = unsafe { empyrean_sys::empyrean_session_unmask_all(self.raw.as_ptr()) };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(())
    }

    /// Whether observation `idx` is masked.
    pub fn is_masked(&self, idx: usize) -> bool {
        unsafe { empyrean_sys::empyrean_session_is_masked(self.raw.as_ptr(), idx) == 1 }
    }

    /// Run an OD refine using the current mask state.
    ///
    /// On the first call runs full IOD → DC; subsequent calls reuse
    /// the previous fit as the seed. Pushes a new entry onto the
    /// session history and returns it as a [`DetermineResult`].
    pub fn refine(&mut self, ctx: &Context) -> Result<DetermineResult> {
        let mut ffi_result = empyrean_sys::EmpyreanODResult::default();
        let code = unsafe {
            empyrean_sys::empyrean_session_refine(self.raw.as_ptr(), ctx.as_raw(), &mut ffi_result)
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let det = od_result_from_ffi(&ffi_result);
        unsafe { empyrean_sys::empyrean_od_result_free(&mut ffi_result) };
        det
    }

    /// Number of fits stored in the session history.
    pub fn history_len(&self) -> usize {
        unsafe { empyrean_sys::empyrean_session_history_len(self.raw.as_ptr()) }
    }

    /// Retrieve the i-th history entry.
    pub fn history(&self, idx: usize) -> Result<DetermineResult> {
        let mut ffi_result = empyrean_sys::EmpyreanODResult::default();
        let code = unsafe {
            empyrean_sys::empyrean_session_get_history(self.raw.as_ptr(), idx, &mut ffi_result)
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let det = od_result_from_ffi(&ffi_result);
        unsafe { empyrean_sys::empyrean_od_result_free(&mut ffi_result) };
        det
    }

    /// Diff the current fit against the `prior_idx`-th history entry.
    pub fn diff(&self, prior_idx: usize) -> Result<SessionDiff> {
        let mut ffi_diff = empyrean_sys::EmpyreanSessionDiff {
            reduced_chi2_delta: 0.0,
            iterations_delta: 0,
            n_observations_delta: 0,
            update_norm_current: 0.0,
            update_norm_prior: 0.0,
        };
        let code = unsafe {
            empyrean_sys::empyrean_session_diff(self.raw.as_ptr(), prior_idx, &mut ffi_diff)
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(SessionDiff::from_ffi(&ffi_diff))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { empyrean_sys::empyrean_session_free(self.raw.as_ptr()) }
    }
}

fn od_result_from_ffi(result: &empyrean_sys::EmpyreanODResult) -> Result<DetermineResult> {
    crate::od::ffi_od_result_to_rust_pub(result)
}
