//! Reusable pre-built force-model handle.
//!
//! A [`BuiltSystem`] assembles the force model — perturber set,
//! spherical-harmonic gravity, the post-Newtonian relativistic correction,
//! and the integration frame — **once** for a frozen
//! `{force_model, frame, divisor}` key, then reuses it across every
//! propagation or ephemeris call made through it. For workloads dominated by
//! short forward models (fit / predict / screen loops) the per-call
//! force-model assembly is the dominant cost; the handle amortizes it away.
//! Results are bit-identical to the one-shot [`Context::propagate`] /
//! [`Context::generate_ephemeris`] entry points on the matching key — the
//! handle only changes *when* the assembly happens, never the numerics.
//!
//! ```no_run
//! use empyrean::{Context, ForceModelTier, Frame, PropagationConfig};
//!
//! let ctx = Context::from_data_dir(None)?;
//! let batch = empyrean::query_sbdb(&["99942"], None)?;
//!
//! // Assemble the ~27-body force model once, at the engine-default divisor.
//! let system = ctx.built_system(ForceModelTier::Standard, Frame::EclipticJ2000, 0.0)?;
//!
//! // Reuse it across many short propagations with a matching config.
//! let cfg = PropagationConfig::default();
//! let epochs = vec![empyrean::Epoch::from_mjd_tdb(60800.0)];
//! let result = system.propagate(&ctx, &batch.orbits, &epochs, &cfg)?;
//! # Ok::<(), empyrean::Error>(())
//! ```
//!
//! # Identity guard
//!
//! Every forward-model call validates, before it runs, that (a) the handle
//! was built from the SAME ephemeris data the passed [`Context`] holds,
//! (b) the per-call config matches the frozen key, and (c) the source data
//! has not been mutated since the handle was built. Any mismatch — data
//! identity, frame, force model, divisor, or staleness — returns a loud,
//! distinct [`Error`] (classify it with [`Error::builtsystem_guard`]). The
//! handle never silently rebuilds and never serves wrong physics. **Rebuild
//! the handle after any `load_*` on the context** (a new kernel changes the
//! ephemeris data the frozen force model was assembled from).
//!
//! # Sharing across threads
//!
//! [`BuiltSystem`] is `Send + Sync` and every forward-model method takes
//! `&self`, so a single handle can be shared by reference across threads —
//! hand `&system` to as many workers as you like and run them concurrently.
//! It borrows nothing from the context after construction.
//!
//! A one-shot radar forward model is not exposed at this surface yet; it is a
//! deliberate, additive omission (the underlying C ABI has no one-shot radar
//! today).

use std::ffi::{CStr, c_char};
use std::path::PathBuf;
use std::ptr::NonNull;

use crate::context::Context;
use crate::coordinate::{Frame, int_to_frame};
use crate::ephemeris::{
    EphemerisConfig, EphemerisResult, marshal_ephemeris_result, observers_to_ffi,
};
use crate::error::{Error, Result};
use crate::observers::Observer;
use crate::orbit::{Orbit, orbits_to_ffi};
use crate::propagate::{
    ForceModelTier, PropagationConfig, PropagationResult, marshal_propagation_result,
};

// ────────────────────────────────────────────────────────────────────
// Identity-guard error classification
// ────────────────────────────────────────────────────────────────────

/// Which identity-guard axis a [`BuiltSystem`] forward-model call tripped.
///
/// A handle never silently rebuilds or serves wrong physics: a foreign
/// context, a stale handle, or a per-call config that diverges from the
/// frozen key each surfaces as a distinct, loud [`Error`]. Recover the axis
/// with [`Error::builtsystem_guard`] and match on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltSystemGuardError {
    /// The handle was NOT built from the ephemeris data the passed context
    /// holds. Rebuild it against the context you are calling with.
    DataMismatch,
    /// The per-call config's output frame diverges from the frozen frame.
    KeyMismatchFrame,
    /// The per-call config's force-model tier diverges from the frozen tier.
    KeyMismatchForceModel,
    /// The per-call config's encounter-timescale divisor diverges from the
    /// frozen divisor.
    KeyMismatchDivisor,
    /// The context's ephemeris data was mutated (a kernel load) after the
    /// handle was built. Rebuild the handle.
    Stale,
}

impl Error {
    /// Classify this error as one of the [`BuiltSystem`] identity-guard axes,
    /// or `None` for any non-guard error.
    ///
    /// Lets a caller react per axis to a forward-model rejection without
    /// pattern-matching raw integer codes — every guard axis is a distinct,
    /// loud failure, never a silent rebuild.
    pub fn builtsystem_guard(&self) -> Option<BuiltSystemGuardError> {
        match self.code {
            c if c == empyrean_sys::EMPYREAN_BUILTSYSTEM_DATA_MISMATCH => {
                Some(BuiltSystemGuardError::DataMismatch)
            }
            c if c == empyrean_sys::EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME => {
                Some(BuiltSystemGuardError::KeyMismatchFrame)
            }
            c if c == empyrean_sys::EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FORCE_MODEL => {
                Some(BuiltSystemGuardError::KeyMismatchForceModel)
            }
            c if c == empyrean_sys::EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR => {
                Some(BuiltSystemGuardError::KeyMismatchDivisor)
            }
            c if c == empyrean_sys::EMPYREAN_BUILTSYSTEM_STALE => {
                Some(BuiltSystemGuardError::Stale)
            }
            _ => None,
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Provenance description
// ────────────────────────────────────────────────────────────────────

/// Category of a loaded data file in a handle's kernel manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// SPK ephemeris kernel (planetary / small-body / spacecraft states).
    Spk,
    /// Binary PCK body-orientation kernel (Earth / Moon rotation).
    Bpc,
    /// Text PCK of gravitational parameters.
    Tpc,
    /// Gravity-field coefficient model (spherical-harmonics file or built-in field).
    Gravity,
    /// Observatory-code registry.
    ObsCodes,
}

/// Where a kernel in a handle's manifest came from.
///
/// Mirrors the engine's kernel-provenance record: each variant carries
/// exactly the identity fields that provenance can supply.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelProvenance {
    /// Loaded from a file on disk.
    File {
        /// Absolute path the kernel was loaded from.
        path: PathBuf,
        /// Lowercase-hex SHA-256 of the file's bytes (64 chars).
        sha256: String,
        /// Hashed file size in bytes.
        bytes: u64,
    },
    /// Handed over pre-loaded in memory (no path or hash known).
    InMemory,
    /// Synthesized from constants compiled into the engine.
    BuiltIn {
        /// Human-readable model name.
        name: String,
    },
}

/// One entry of the kernel-identity manifest a handle captured at
/// construction. Names the provenance of each loaded data file — never any
/// tuning rationale.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRecord {
    /// Category of the entry.
    pub kind: KernelKind,
    /// Where the entry came from, with its identity fields.
    pub provenance: KernelProvenance,
}

/// A reproducibility summary of a [`BuiltSystem`]'s frozen force model plus
/// the kernel-identity manifest it captured from the context.
///
/// Every field is populated by [`BuiltSystem::describe`] — nothing is
/// defaulted. Names the citable menu (tier, frame, GR, perturbers, BPC,
/// kernel hashes) so a run can be reproduced and audited.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemDescription {
    /// The frozen force-model tier.
    pub force_model: ForceModelTier,
    /// The frozen integration/output frame.
    pub frame: Frame,
    /// The frozen encounter-timescale divisor.
    pub encounter_timescale_divisor: f64,
    /// Whether the post-Newtonian (EIH) relativistic correction is enabled.
    pub relativistic: bool,
    /// Whether the N16 asteroid perturbers are included.
    pub asteroids: bool,
    /// Whether a BPC (body-fixed rotation) kernel is loaded.
    pub has_bpc: bool,
    /// NAIF ids of the perturbing bodies included in the force model.
    pub perturber_origins: Vec<i32>,
    /// The captured kernel-identity manifest, one record per loaded file.
    pub kernels: Vec<KernelRecord>,
}

// ────────────────────────────────────────────────────────────────────
// The handle
// ────────────────────────────────────────────────────────────────────

/// A reusable pre-built force-model handle.
///
/// Wraps an opaque native handle; the assembled force model and the captured
/// kernel manifest live on the libempyrean heap and are released when this
/// value is dropped. Construct with [`Context::built_system`]. See the
/// [module docs](self) for the identity guard and the cross-thread reuse
/// story.
pub struct BuiltSystem {
    raw: NonNull<empyrean_sys::EmpyreanBuiltSystem>,
}

// Safety: libempyrean documents `EmpyreanBuiltSystem` as `Send + Sync` — it
// holds only `Arc`-shared kernels and, after construction, borrows nothing
// from the context. A single `&BuiltSystem` may be shared across threads and
// every forward-model call takes `&self`.
unsafe impl Send for BuiltSystem {}
unsafe impl Sync for BuiltSystem {}

// Structural canary: `BuiltSystem` MUST stay `Send + Sync` so one handle can
// be shared by `&handle` across threads (its reason to exist — assemble once,
// reuse everywhere). This fails to compile if a future field breaks that.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BuiltSystem>();
};

impl Context {
    /// Assemble a reusable force-model handle for `(force_model, frame)`,
    /// freezing `encounter_timescale_divisor` into its key.
    ///
    /// Pass `0.0` for `encounter_timescale_divisor` to freeze the engine
    /// default; any positive value freezes that divisor into the key before
    /// it is used to validate calls. The handle snapshots this context's
    /// kernel manifest so [`BuiltSystem::describe`] is self-contained.
    ///
    /// Reuse the returned handle across many [`BuiltSystem::propagate`] /
    /// [`BuiltSystem::generate_ephemeris`] calls with a config matching the
    /// frozen key. **After loading any additional kernel into this context,
    /// rebuild the handle** — a stale handle is rejected loudly by every
    /// forward-model call, never silently reused.
    pub fn built_system(
        &self,
        force_model: ForceModelTier,
        frame: Frame,
        encounter_timescale_divisor: f64,
    ) -> Result<BuiltSystem> {
        let mut raw: *mut empyrean_sys::EmpyreanBuiltSystem = std::ptr::null_mut();
        let code = unsafe {
            empyrean_sys::empyrean_builtsystem_new(
                self.as_raw(),
                force_model as i32,
                frame as i32,
                encounter_timescale_divisor,
                &mut raw,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        NonNull::new(raw)
            .map(|raw| BuiltSystem { raw })
            .ok_or_else(Error::from_null_ptr)
    }
}

impl BuiltSystem {
    /// Borrow the raw FFI handle pointer (internal use).
    fn as_raw(&self) -> *const empyrean_sys::EmpyreanBuiltSystem {
        self.raw.as_ptr()
    }

    /// Propagate one or more orbits to a list of target epochs through the
    /// pre-built force model.
    ///
    /// Parallels [`Context::propagate`] but reuses the frozen force model.
    /// Before dispatch the identity guard runs: the handle must have been
    /// built from `ctx`'s ephemeris data, the `config` must match the frozen
    /// key, and the data must be unmutated since build. Any mismatch returns
    /// a loud, distinct [`Error`] (see [`Error::builtsystem_guard`]) — never
    /// a silent rebuild. On pass the result is bit-identical to the one-shot
    /// with the same config.
    pub fn propagate(
        &self,
        ctx: &Context,
        orbits: &[Orbit],
        epochs: &[crate::Epoch],
        config: &PropagationConfig,
    ) -> Result<PropagationResult> {
        let (ffi_orbits, _orbit_keep) = orbits_to_ffi(orbits)?;
        let (ffi_config, _config_keep) = config.to_ffi_with();
        let epochs_mjd_tdb: Vec<f64> = epochs
            .iter()
            .map(|e| e.mjd_tdb())
            .collect::<Result<Vec<_>>>()?;
        let mut ffi_result = empyrean_sys::EmpyreanPropagationResult::default();
        let code = unsafe {
            empyrean_sys::empyrean_builtsystem_propagate(
                self.as_raw(),
                ctx.as_raw(),
                ffi_orbits.as_ptr(),
                ffi_orbits.len(),
                epochs_mjd_tdb.as_ptr(),
                epochs_mjd_tdb.len(),
                &ffi_config,
                &mut ffi_result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        marshal_propagation_result(ffi_result, orbits.len())
    }

    /// Generate predicted ephemeris for orbits as seen by observers through
    /// the pre-built force model.
    ///
    /// Parallels [`Context::generate_ephemeris`] but reuses the frozen force
    /// model. Runs the same identity guard as [`BuiltSystem::propagate`]
    /// before dispatch; on pass the result is bit-identical to the one-shot.
    /// The ephemeris config carries no divisor knob, so a handle frozen at a
    /// non-default divisor is rejected here as a
    /// [`KeyMismatchDivisor`](BuiltSystemGuardError::KeyMismatchDivisor)
    /// rather than served under the wrong dynamics — build ephemeris-reuse
    /// handles at the default divisor (`0.0`).
    pub fn generate_ephemeris(
        &self,
        ctx: &Context,
        orbits: &[Orbit],
        observers: &[Observer],
        config: &EphemerisConfig,
    ) -> Result<EphemerisResult> {
        let (ffi_orbits, _orbit_keep) = orbits_to_ffi(orbits)?;
        let ffi_observers = observers_to_ffi(observers)?;
        let (ffi_config, _config_keep) = config.to_ffi_with();
        let mut ffi_result = empyrean_sys::EmpyreanEphemerisResult::default();
        let code = unsafe {
            empyrean_sys::empyrean_builtsystem_generate_ephemeris(
                self.as_raw(),
                ctx.as_raw(),
                ffi_orbits.as_ptr(),
                ffi_orbits.len(),
                ffi_observers.as_ptr(),
                ffi_observers.len(),
                &ffi_config,
                &mut ffi_result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(marshal_ephemeris_result(&mut ffi_result))
    }

    /// Return a full reproducibility summary of the frozen force model and
    /// the kernel manifest captured at construction.
    ///
    /// Every field of the returned [`SystemDescription`] is populated —
    /// nothing is defaulted. Self-contained: needs no context (the manifest
    /// was snapshotted when the handle was built).
    pub fn describe(&self) -> Result<SystemDescription> {
        let mut ffi = empyrean_sys::EmpyreanSystemDescription::default();
        let code = unsafe { empyrean_sys::empyrean_builtsystem_describe(self.as_raw(), &mut ffi) };
        if code != 0 {
            return Err(Error::capture(code));
        }
        // Copy every field out into owned Rust before releasing the C-owned
        // heap arrays, so nothing dangles and no field is dropped.
        let result = system_description_from_ffi(&ffi);
        unsafe { empyrean_sys::empyrean_builtsystem_description_free(&mut ffi) };
        result
    }
}

impl Drop for BuiltSystem {
    fn drop(&mut self) {
        unsafe { empyrean_sys::empyrean_builtsystem_free(self.raw.as_ptr()) }
    }
}

// ────────────────────────────────────────────────────────────────────
// FFI → safe marshaling (local helpers)
// ────────────────────────────────────────────────────────────────────

/// Marshal a populated FFI description into the owned [`SystemDescription`].
/// Reads pointers but takes no ownership — the caller frees the C arrays.
fn system_description_from_ffi(
    d: &empyrean_sys::EmpyreanSystemDescription,
) -> Result<SystemDescription> {
    let force_model = int_to_force_model_tier(d.force_model)?;
    let frame = int_to_frame(d.frame)?;
    let perturber_origins = if d.perturber_origins.is_null() || d.num_perturbers == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(d.perturber_origins, d.num_perturbers) }.to_vec()
    };
    let kernels = if d.kernels.is_null() || d.num_kernels == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(d.kernels, d.num_kernels) }
            .iter()
            .map(kernel_record_from_ffi)
            .collect::<Result<Vec<_>>>()?
    };
    Ok(SystemDescription {
        force_model,
        frame,
        encounter_timescale_divisor: d.encounter_timescale_divisor,
        relativistic: d.relativistic != 0,
        asteroids: d.asteroids != 0,
        has_bpc: d.has_bpc != 0,
        perturber_origins,
        kernels,
    })
}

/// Marshal one FFI kernel record into the owned [`KernelRecord`], populating
/// exactly the provenance fields the native record supplies.
fn kernel_record_from_ffi(r: &empyrean_sys::EmpyreanKernelRecord) -> Result<KernelRecord> {
    let kind = int_to_kernel_kind(r.kind)?;
    let provenance = match r.provenance {
        p if p == empyrean_sys::EMPYREAN_KERNEL_PROVENANCE_FILE as i32 => {
            let path = cstr_to_string(r.path)
                .ok_or_else(|| Error::invalid_input("FILE kernel record has a null path"))?;
            let sha256 = cstr_to_string(r.sha256)
                .ok_or_else(|| Error::invalid_input("FILE kernel record has a null sha256"))?;
            KernelProvenance::File {
                path: PathBuf::from(path),
                sha256,
                bytes: r.bytes,
            }
        }
        p if p == empyrean_sys::EMPYREAN_KERNEL_PROVENANCE_IN_MEMORY as i32 => {
            KernelProvenance::InMemory
        }
        p if p == empyrean_sys::EMPYREAN_KERNEL_PROVENANCE_BUILT_IN as i32 => {
            let name = cstr_to_string(r.name)
                .ok_or_else(|| Error::invalid_input("BUILT_IN kernel record has a null name"))?;
            KernelProvenance::BuiltIn { name }
        }
        other => {
            return Err(Error::invalid_input(format!(
                "unknown kernel provenance tag {other}"
            )));
        }
    };
    Ok(KernelRecord { kind, provenance })
}

/// Map a native kernel-kind tag to [`KernelKind`]. An unknown tag is a loud
/// error — never a silent fallback into a wrong category.
fn int_to_kernel_kind(v: i32) -> Result<KernelKind> {
    match v {
        x if x == empyrean_sys::EMPYREAN_KERNEL_KIND_SPK as i32 => Ok(KernelKind::Spk),
        x if x == empyrean_sys::EMPYREAN_KERNEL_KIND_BPC as i32 => Ok(KernelKind::Bpc),
        x if x == empyrean_sys::EMPYREAN_KERNEL_KIND_TPC as i32 => Ok(KernelKind::Tpc),
        x if x == empyrean_sys::EMPYREAN_KERNEL_KIND_GRAVITY as i32 => Ok(KernelKind::Gravity),
        x if x == empyrean_sys::EMPYREAN_KERNEL_KIND_OBSCODES as i32 => Ok(KernelKind::ObsCodes),
        other => Err(Error::invalid_input(format!(
            "unknown kernel kind tag {other}"
        ))),
    }
}

/// Map a force-model tier code (0/1/2) to [`ForceModelTier`]. An unknown code
/// is a loud error rather than a wrong tier.
fn int_to_force_model_tier(v: i32) -> Result<ForceModelTier> {
    match v {
        0 => Ok(ForceModelTier::Approximate),
        1 => Ok(ForceModelTier::Basic),
        2 => Ok(ForceModelTier::Standard),
        other => Err(Error::invalid_input(format!(
            "unknown force-model tier code {other}"
        ))),
    }
}

/// Copy a NUL-terminated C string into an owned `String`, or `None` for null.
fn cstr_to_string(p: *const c_char) -> Option<String> {
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CoordinateState, Epoch, Origin};

    // ── Structural checks (no ephemeris data required) ──────────

    /// The handle and its description must be `Send + Sync` so a single
    /// handle can be shared by `&handle` across threads (its reason to
    /// exist). A compile-time assertion, exercised as a test.
    #[test]
    fn handle_is_send_sync() {
        fn requires_send_sync<T: Send + Sync>() {}
        requires_send_sync::<BuiltSystem>();
        requires_send_sync::<SystemDescription>();
        requires_send_sync::<BuiltSystemGuardError>();
    }

    /// The identity-guard axes must classify to *distinct* variants so a
    /// caller can tell data / frame / force-model / divisor / stale apart.
    #[test]
    fn guard_codes_classify_distinctly() {
        let cases = [
            (
                empyrean_sys::EMPYREAN_BUILTSYSTEM_DATA_MISMATCH,
                BuiltSystemGuardError::DataMismatch,
            ),
            (
                empyrean_sys::EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME,
                BuiltSystemGuardError::KeyMismatchFrame,
            ),
            (
                empyrean_sys::EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FORCE_MODEL,
                BuiltSystemGuardError::KeyMismatchForceModel,
            ),
            (
                empyrean_sys::EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR,
                BuiltSystemGuardError::KeyMismatchDivisor,
            ),
            (
                empyrean_sys::EMPYREAN_BUILTSYSTEM_STALE,
                BuiltSystemGuardError::Stale,
            ),
        ];
        for (code, expected) in cases {
            let err = Error {
                code,
                message: String::new(),
            };
            assert_eq!(err.builtsystem_guard(), Some(expected), "code {code}");
        }
        // A non-guard code classifies to None.
        let ok = Error {
            code: -3,
            message: String::new(),
        };
        assert_eq!(ok.builtsystem_guard(), None);
    }

    // ── Full round-trip (gated on ephemeris data) ───────────────

    fn try_ctx() -> Option<Context> {
        Context::from_data_dir(None).ok()
    }

    /// A gravity-only heliocentric Cartesian orbit (AU, AU/day).
    fn base_orbit() -> Orbit {
        Orbit::new(CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::ICRF,
            Origin::SUN,
        ))
    }

    /// A Standard-tier, ICRF-output config matching a handle frozen the same
    /// way — so the identity guard passes and the numerics can be compared.
    fn standard_icrf_config() -> PropagationConfig {
        PropagationConfig {
            force_model: ForceModelTier::Standard,
            frame: Frame::ICRF,
            ..PropagationConfig::default()
        }
    }

    /// Propagation through the handle is bit-identical to the one-shot
    /// [`Context::propagate`] for the same orbit / config — the handle only
    /// changes *when* the force model is assembled, not the numerics.
    #[test]
    fn builtsystem_propagate_matches_one_shot() {
        let ctx = match try_ctx() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_propagate_matches_one_shot: no data dir");
                return;
            }
        };
        let cfg = standard_icrf_config();
        let orbits = [base_orbit()];
        let epochs = [
            Epoch::from_mjd_tdb(59000.0),
            Epoch::from_mjd_tdb(59010.0),
            Epoch::from_mjd_tdb(59030.0),
        ];

        let one_shot = ctx.propagate(&orbits, &epochs, &cfg).expect("one-shot");

        let system = ctx
            .built_system(ForceModelTier::Standard, Frame::ICRF, 0.0)
            .expect("build handle");
        let via_handle = system
            .propagate(&ctx, &orbits, &epochs, &cfg)
            .expect("handle propagate");

        assert_eq!(one_shot.states.len(), via_handle.states.len());
        assert_eq!(one_shot.states.len(), epochs.len());
        for (i, (a, b)) in one_shot
            .states
            .iter()
            .zip(via_handle.states.iter())
            .enumerate()
        {
            assert_eq!(
                a.epoch.mjd_tdb().unwrap(),
                b.epoch.mjd_tdb().unwrap(),
                "epoch[{i}] bit-identical"
            );
            assert_eq!(a.position, b.position, "position[{i}] bit-identical");
            assert_eq!(a.velocity, b.velocity, "velocity[{i}] bit-identical");
        }
    }

    /// Ephemeris generation through the handle is bit-identical to the
    /// one-shot [`Context::generate_ephemeris`], exercising the second
    /// forward-model method and the shared ephemeris marshaling.
    #[test]
    fn builtsystem_generate_ephemeris_matches_one_shot() {
        let ctx = match try_ctx() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_generate_ephemeris_matches_one_shot: no data dir");
                return;
            }
        };
        // ICRF-output ephemeris config so the handle key matches.
        let eph_cfg = EphemerisConfig {
            propagation: standard_icrf_config(),
            ..EphemerisConfig::default()
        };
        let orbits = [base_orbit()];
        let observers = [Observer {
            obs_code: "500".to_string(),
            epoch: Epoch::from_mjd_tdb(59000.0),
            position: [0.9, -0.42, -0.18],
            velocity: [0.0075, 0.0148, 0.0064],
            observing_night: -1,
        }];

        let one_shot = ctx
            .generate_ephemeris(&orbits, &observers, &eph_cfg)
            .expect("one-shot ephemeris");

        let system = ctx
            .built_system(ForceModelTier::Standard, Frame::ICRF, 0.0)
            .expect("build handle");
        let via_handle = system
            .generate_ephemeris(&ctx, &orbits, &observers, &eph_cfg)
            .expect("handle ephemeris");

        assert_eq!(one_shot.entries.len(), via_handle.entries.len());
        assert!(!one_shot.entries.is_empty(), "expected at least one entry");
        for (i, (a, b)) in one_shot
            .entries
            .iter()
            .zip(via_handle.entries.iter())
            .enumerate()
        {
            assert_eq!(a.ra_deg, b.ra_deg, "ra[{i}] bit-identical");
            assert_eq!(a.dec_deg, b.dec_deg, "dec[{i}] bit-identical");
            assert_eq!(a.rho_au, b.rho_au, "rho[{i}] bit-identical");
        }
    }

    /// `describe()` reports the frozen force model plus a fully-populated,
    /// non-empty kernel manifest — a FILE record carries a well-formed
    /// SHA-256 whose byte count matches the file on disk.
    #[test]
    fn builtsystem_describe_reports_provenance() {
        let ctx = match try_ctx() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_describe_reports_provenance: no data dir");
                return;
            }
        };
        let system = ctx
            .built_system(ForceModelTier::Standard, Frame::ICRF, 0.0)
            .expect("build handle");
        let desc = system.describe().expect("describe");

        assert_eq!(desc.force_model, ForceModelTier::Standard);
        assert_eq!(desc.frame, Frame::ICRF);
        assert_eq!(
            desc.encounter_timescale_divisor, 1000.0,
            "engine default divisor"
        );
        assert!(desc.relativistic, "Standard tier includes GR");
        assert!(desc.asteroids, "Standard tier includes N16 asteroids");
        assert!(desc.has_bpc, "Standard tier loads a BPC");
        assert!(
            !desc.perturber_origins.is_empty(),
            "Standard tier has perturbers"
        );
        assert!(!desc.kernels.is_empty(), "kernel manifest is non-empty");

        // Every FILE record is fully populated; cross-check the smallest one
        // against the file on disk (a dropped/defaulted field would fail).
        let mut checked_a_file = false;
        let mut smallest: Option<(&PathBuf, &String, u64)> = None;
        for rec in &desc.kernels {
            if let KernelProvenance::File {
                path,
                sha256,
                bytes,
            } = &rec.provenance
            {
                assert_eq!(sha256.len(), 64, "sha256 is 64 chars");
                assert!(
                    sha256.chars().all(|c| c.is_ascii_hexdigit()),
                    "sha256 is lowercase hex: {sha256}"
                );
                assert!(*bytes > 0, "FILE record has a nonzero byte count");
                if smallest
                    .as_ref()
                    .map(|(_, _, b)| *bytes < *b)
                    .unwrap_or(true)
                {
                    smallest = Some((path, sha256, *bytes));
                }
                checked_a_file = true;
            }
        }
        assert!(
            checked_a_file,
            "expected at least one FILE-provenance kernel"
        );
        let (path, _sha, bytes) = smallest.unwrap();
        let on_disk = std::fs::metadata(path).expect("kernel file exists").len();
        assert_eq!(on_disk, bytes, "manifest byte count matches file on disk");
    }

    /// A per-call config that diverges from the frozen key fires the identity
    /// guard loudly and by axis — never a silent rebuild under wrong physics.
    #[test]
    fn builtsystem_guard_fires_on_key_mismatch() {
        let ctx = match try_ctx() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_guard_fires_on_key_mismatch: no data dir");
                return;
            }
        };
        let system = ctx
            .built_system(ForceModelTier::Standard, Frame::ICRF, 0.0)
            .expect("build handle");
        let orbits = [base_orbit()];
        let epochs = [Epoch::from_mjd_tdb(59000.0), Epoch::from_mjd_tdb(59010.0)];

        // Force-model axis: frozen Standard, call requests Basic.
        let cfg_fm = PropagationConfig {
            force_model: ForceModelTier::Basic,
            frame: Frame::ICRF,
            ..PropagationConfig::default()
        };
        let err_fm = system
            .propagate(&ctx, &orbits, &epochs, &cfg_fm)
            .expect_err("force-model mismatch must error");
        assert_eq!(
            err_fm.builtsystem_guard(),
            Some(BuiltSystemGuardError::KeyMismatchForceModel),
            "force-model mismatch fires by axis: {err_fm}"
        );

        // Frame axis: frozen ICRF, call requests EclipticJ2000.
        let cfg_fr = PropagationConfig {
            force_model: ForceModelTier::Standard,
            frame: Frame::EclipticJ2000,
            ..PropagationConfig::default()
        };
        let err_fr = system
            .propagate(&ctx, &orbits, &epochs, &cfg_fr)
            .expect_err("frame mismatch must error");
        assert_eq!(
            err_fr.builtsystem_guard(),
            Some(BuiltSystemGuardError::KeyMismatchFrame),
            "frame mismatch fires by axis: {err_fr}"
        );
    }

    /// A handle used with a *different* context (distinct ephemeris-data
    /// instance) is rejected as a data mismatch — never silently rebuilt
    /// against the foreign kernels. The correct context still passes, proving
    /// the guard is specific.
    #[test]
    fn builtsystem_guard_fires_on_data_mismatch() {
        let ctx_a = match try_ctx() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_guard_fires_on_data_mismatch: no data dir");
                return;
            }
        };
        let ctx_b = match try_ctx() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_guard_fires_on_data_mismatch: second context");
                return;
            }
        };
        let system = ctx_a
            .built_system(ForceModelTier::Standard, Frame::ICRF, 0.0)
            .expect("build handle");
        let orbits = [base_orbit()];
        let epochs = [Epoch::from_mjd_tdb(59000.0), Epoch::from_mjd_tdb(59010.0)];
        let cfg = standard_icrf_config();

        // Foreign context → loud data mismatch.
        let err = system
            .propagate(&ctx_b, &orbits, &epochs, &cfg)
            .expect_err("foreign context must be rejected");
        assert_eq!(
            err.builtsystem_guard(),
            Some(BuiltSystemGuardError::DataMismatch),
            "foreign context fires the data-identity guard: {err}"
        );

        // Correct context → succeeds.
        system
            .propagate(&ctx_a, &orbits, &epochs, &cfg)
            .expect("correct context must pass");
    }
}
