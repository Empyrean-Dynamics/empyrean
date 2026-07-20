//! C ABI exports for the reusable pre-built force-model handle.
//!
//! Wraps [`empyrean_core::propagation::BuiltSystem`] — the force model
//! (perturber set, harmonics, relativistic correction, integration frame)
//! assembled **once** for a frozen `{force_model, frame, divisor}` key and
//! reused across every forward-model call made through it. For workloads
//! dominated by short propagations (fit / predict / screen loops) the
//! per-call force-model assembly is the dominant cost; the handle amortizes
//! it away. Results are bit-identical to the one-shot
//! [`empyrean_propagate`](crate::propagate::empyrean_propagate) /
//! [`empyrean_generate_ephemeris`](crate::ephemeris::empyrean_generate_ephemeris)
//! entry points on the matching key — the handle only changes *when* the
//! assembly happens.
//!
//! **Identity guard.** Every forward-model call validates, before it runs,
//! that (a) the handle was built from the SAME ephemeris data the passed
//! context holds, (b) the per-call config matches the frozen key, and (c)
//! the source data has not been mutated since the handle was built. Any
//! mismatch returns a loud, distinct error code — never a silent rebuild,
//! never wrong physics. Rebuild the handle after any context load.
//!
//! The handle is `Send + Sync` (it holds only `Arc`-shared kernels), so a
//! single `&EmpyreanBuiltSystem` may be shared across threads concurrently.
//! A one-shot radar forward model is not exposed at this surface; it is a
//! deliberate, additive omission (the C ABI has no one-shot radar today).

use std::ffi::c_char;
use std::panic::AssertUnwindSafe;

use empyrean_core::convert::frame_to_int;
use empyrean_core::data::{KernelKind, KernelProvenance, KernelRecord};
use empyrean_core::ephemeris::EphemerisGenerationError;
use empyrean_core::propagation::{BuiltSystem, PropagationError, SystemKeyMismatch};
use empyrean_core::time::Epoch;

use crate::ephemeris::{
    EmpyreanEphemerisConfig, EmpyreanEphemerisResult, build_ephemeris_config_from_c,
    build_observers_from_c, build_orbits_for_ephemeris, marshal_ephemeris_result,
};
use crate::observers::EmpyreanObserver;
use crate::propagate::{
    EmpyreanOrbit, EmpyreanPropagationConfig, EmpyreanPropagationResult,
    build_orbits_for_propagation, build_propagation_config_from_c, free_c_str, int_to_force_model,
    marshal_propagation_result, to_c_str,
};
use crate::{EmpyreanContext, set_last_error};

// ────────────────────────────────────────────────────────────────────
// Return codes — split by identity-guard axis
// ────────────────────────────────────────────────────────────────────

/// Success.
pub const EMPYREAN_BUILTSYSTEM_OK: i32 = 0;
/// A required pointer argument was null.
pub const EMPYREAN_BUILTSYSTEM_NULL_POINTER: i32 = -1;
/// An input was malformed (unknown force-model tier / frame code, a bad
/// orbit row, a kernel-manifest read failure, an unresolvable config).
pub const EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT: i32 = -2;
/// The forward model ran but failed for a reason unrelated to the identity
/// guard (integration / transform / missing data). `empyrean_last_error()`
/// carries the engine message.
pub const EMPYREAN_BUILTSYSTEM_PROPAGATION: i32 = -3;
/// Heap allocation for an output buffer failed.
pub const EMPYREAN_BUILTSYSTEM_ALLOC: i32 = -5;
/// Identity guard (a): the handle was NOT built from the ephemeris data the
/// passed context holds (`Arc` identity). The handle would otherwise serve
/// physics from a foreign kernel set — rebuild it against this context.
pub const EMPYREAN_BUILTSYSTEM_DATA_MISMATCH: i32 = -20;
/// Identity guard (b): the per-call config's output frame diverges from the
/// handle's frozen frame.
pub const EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME: i32 = -21;
/// Identity guard (b): the per-call config's force-model tier diverges from
/// the handle's frozen tier.
pub const EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FORCE_MODEL: i32 = -22;
/// Identity guard (b): the per-call config's encounter-timescale divisor
/// diverges from the handle's frozen divisor.
pub const EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR: i32 = -23;
/// Identity guard (c): the context's ephemeris data was mutated (a kernel
/// load) after the handle was built. The handle refuses to serve results
/// assembled from since-replaced kernels — rebuild it.
pub const EMPYREAN_BUILTSYSTEM_STALE: i32 = -24;
/// A panic was caught at the FFI boundary.
pub const EMPYREAN_BUILTSYSTEM_PANIC: i32 = -99;

// ── Kernel-record `kind` tag (EmpyreanKernelRecord.kind) ────────────
/// SPK ephemeris kernel (planetary / small-body / spacecraft states).
pub const EMPYREAN_KERNEL_KIND_SPK: i32 = 0;
/// Binary PCK body-orientation kernel (Earth / Moon rotation).
pub const EMPYREAN_KERNEL_KIND_BPC: i32 = 1;
/// Text PCK of gravitational parameters.
pub const EMPYREAN_KERNEL_KIND_TPC: i32 = 2;
/// Gravity-field coefficient model (spherical-harmonics file or built-in field).
pub const EMPYREAN_KERNEL_KIND_GRAVITY: i32 = 3;
/// Observatory-code registry.
pub const EMPYREAN_KERNEL_KIND_OBSCODES: i32 = 4;

// ── Kernel-record `provenance` tag (EmpyreanKernelRecord.provenance) ─
/// Loaded from a file on disk; `path`, `sha256`, and `bytes` are populated.
pub const EMPYREAN_KERNEL_PROVENANCE_FILE: i32 = 0;
/// Handed over pre-loaded in memory (no path known); hash fields are null/0.
pub const EMPYREAN_KERNEL_PROVENANCE_IN_MEMORY: i32 = 1;
/// Synthesized from constants compiled into the engine; `name` is populated.
pub const EMPYREAN_KERNEL_PROVENANCE_BUILT_IN: i32 = 2;

// ────────────────────────────────────────────────────────────────────
// Opaque handle
// ────────────────────────────────────────────────────────────────────

/// Opaque handle wrapping a pre-assembled force model plus the
/// kernel-identity snapshot taken from the context at construction (so
/// [`empyrean_builtsystem_describe`] is self-contained). Construct with
/// [`empyrean_builtsystem_new`]; release with [`empyrean_builtsystem_free`].
///
/// The handle is `Send + Sync`: hand `&EmpyreanBuiltSystem` (via a `const`
/// pointer) to as many threads as you like. It borrows nothing from the
/// context after construction.
pub struct EmpyreanBuiltSystem {
    system: BuiltSystem,
    kernel_manifest: Vec<KernelRecord>,
}

// Structural canary: the handle MUST stay `Send + Sync` so the C ABI can
// share `&EmpyreanBuiltSystem` across threads. `BuiltSystem` is `Send +
// Sync` (it holds only `Arc`-shared kernels) and the manifest snapshot is a
// plain owned `Vec`; this fails to compile if a future field breaks that.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EmpyreanBuiltSystem>();
};

// ────────────────────────────────────────────────────────────────────
// #[repr(C)] provenance structs
// ────────────────────────────────────────────────────────────────────

/// One entry of the kernel-identity manifest captured by a handle.
///
/// Names the provenance of each loaded data file — never any tuning
/// rationale. Free the owning [`EmpyreanSystemDescription`] with
/// [`empyrean_builtsystem_description_free`]; do not free these fields
/// individually.
#[repr(C)]
pub struct EmpyreanKernelRecord {
    /// Category of the entry (`EMPYREAN_KERNEL_KIND_*`).
    pub kind: i32,
    /// Where the entry came from (`EMPYREAN_KERNEL_PROVENANCE_*`).
    pub provenance: i32,
    /// Absolute path the kernel was loaded from. Non-null only for a
    /// `FILE` provenance; null otherwise. NUL-terminated UTF-8.
    pub path: *mut c_char,
    /// Lowercase-hex SHA-256 of the file's bytes (64 chars). Non-null only
    /// for a `FILE` provenance; null otherwise. NUL-terminated UTF-8.
    pub sha256: *mut c_char,
    /// Hashed file size in bytes. 0 unless `FILE` provenance.
    pub bytes: u64,
    /// Human-readable model name. Non-null only for a `BUILT_IN`
    /// provenance; null otherwise. NUL-terminated UTF-8.
    pub name: *mut c_char,
}

/// A reproducibility summary of a handle's frozen force model plus its
/// captured kernel-identity manifest.
///
/// Populated in full by [`empyrean_builtsystem_describe`] — every field
/// reflects the assembled system; nothing is defaulted. Names the citable
/// menu (tier, frame, GR, perturbers, BPC, kernel hashes) — never selection
/// logic or tuning rationale. Release the heap-owned arrays with
/// [`empyrean_builtsystem_description_free`].
#[repr(C)]
pub struct EmpyreanSystemDescription {
    /// Force-model tier: 0=Approximate, 1=Basic, 2=Standard.
    pub force_model: i32,
    /// Integration/output frame: 0=ICRF, 1=EclipticJ2000, 2=ITRF93.
    pub frame: i32,
    /// The frozen encounter-timescale divisor.
    pub encounter_timescale_divisor: f64,
    /// 1 if the post-Newtonian (EIH) GR correction is enabled, else 0.
    pub relativistic: u8,
    /// 1 if the N16 asteroid perturbers are included, else 0.
    pub asteroids: u8,
    /// 1 if a BPC (body-fixed rotation) kernel is loaded, else 0.
    pub has_bpc: u8,
    /// Heap array of perturbing-body NAIF ids (length `num_perturbers`).
    /// Null when `num_perturbers == 0`.
    pub perturber_origins: *mut i32,
    /// Number of entries in [`perturber_origins`](Self::perturber_origins).
    pub num_perturbers: usize,
    /// Heap array of kernel-identity records (length `num_kernels`). Null
    /// when `num_kernels == 0`.
    pub kernels: *mut EmpyreanKernelRecord,
    /// Number of entries in [`kernels`](Self::kernels).
    pub num_kernels: usize,
}

// ────────────────────────────────────────────────────────────────────
// Construction / destruction
// ────────────────────────────────────────────────────────────────────

/// Assemble a reusable force-model handle for `(force_model, frame)` from
/// `ctx`, freezing `encounter_timescale_divisor` into its key.
///
/// - `force_model`: tier code (0=Approximate, 1=Basic, 2=Standard).
/// - `frame`: output/integration frame code (0=ICRF, 1=EclipticJ2000, 2=ITRF93).
/// - `encounter_timescale_divisor`: pass `0.0` to freeze the engine default;
///   any positive value freezes that divisor into the key *before* it is
///   used to validate calls.
/// - `out`: receives the heap-allocated handle on success.
///
/// The handle snapshots `ctx`'s kernel manifest at construction so
/// [`empyrean_builtsystem_describe`] is self-contained. After you load any
/// additional kernel into `ctx`, rebuild the handle — a stale handle is
/// rejected loudly by every forward-model call, never silently reused.
///
/// Returns [`EMPYREAN_BUILTSYSTEM_OK`] on success; on error, `out` is left
/// untouched and `empyrean_last_error()` carries the message. The caller
/// owns the returned handle and must free it with
/// [`empyrean_builtsystem_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_builtsystem_new(
    ctx: *const EmpyreanContext,
    force_model: i32,
    frame: i32,
    encounter_timescale_divisor: f64,
    out: *mut *mut EmpyreanBuiltSystem,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || out.is_null() {
            set_last_error("null pointer argument");
            return EMPYREAN_BUILTSYSTEM_NULL_POINTER;
        }
        let ctx_ref = unsafe { &*ctx };

        let fm = match int_to_force_model(force_model) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e);
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };
        let fr = match empyrean_core::convert::int_to_frame(frame) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e.to_string());
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };

        // Snapshot the kernel manifest first so describe() is self-contained.
        // This hashes the loaded kernels (lazy, cached on the context) and
        // surfaces any file-read failure loudly rather than dropping the
        // provenance record.
        let manifest = match ctx_ref.kernel_manifest() {
            Ok(m) => m,
            Err(e) => {
                set_last_error(&format!("kernel manifest: {e}"));
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };

        let mut system = match ctx_ref.built_system(fm, fr) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&e.to_string());
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };
        // Sentinel: 0.0 keeps the engine default that `built_system` already
        // froze; any positive value freezes that divisor into the key before
        // it is used to validate calls. A non-positive non-zero value is not
        // a valid divisor and likewise falls back to the frozen default,
        // matching the per-call config sentinel convention.
        if encounter_timescale_divisor > 0.0 {
            system = system.with_encounter_timescale_divisor(encounter_timescale_divisor);
        }

        let handle = Box::new(EmpyreanBuiltSystem {
            system,
            kernel_manifest: manifest,
        });
        unsafe {
            *out = Box::into_raw(handle);
        }
        EMPYREAN_BUILTSYSTEM_OK
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_builtsystem_new");
            EMPYREAN_BUILTSYSTEM_PANIC
        }
    }
}

/// Free a handle previously returned by [`empyrean_builtsystem_new`].
/// Passing null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_builtsystem_free(handle: *mut EmpyreanBuiltSystem) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if !handle.is_null() {
            unsafe { drop(Box::from_raw(handle)) };
        }
    }));
}

// ────────────────────────────────────────────────────────────────────
// Forward-model calls (identity-guarded)
// ────────────────────────────────────────────────────────────────────

/// Propagate `orbits` to `times` through the pre-built handle.
///
/// Signature parallels the one-shot
/// [`empyrean_propagate`](crate::propagate::empyrean_propagate) but takes
/// `(handle, ctx, ...)`. Before dispatch the identity guard runs: the handle
/// must have been built from `ctx`'s ephemeris data
/// ([`EMPYREAN_BUILTSYSTEM_DATA_MISMATCH`]); the config must match the frozen
/// key ([`EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME`] /
/// `_FORCE_MODEL` / `_DIVISOR`); and the data must be unmutated since build
/// ([`EMPYREAN_BUILTSYSTEM_STALE`]). On pass, the result is bit-identical to
/// the one-shot with the same config.
///
/// On success populates `result_out`; free it with
/// [`empyrean_propagation_result_free`](crate::propagate::empyrean_propagation_result_free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_builtsystem_propagate(
    handle: *const EmpyreanBuiltSystem,
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    times_ptr: *const f64,
    num_times: usize,
    config: *const EmpyreanPropagationConfig,
    result_out: *mut EmpyreanPropagationResult,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null()
            || ctx.is_null()
            || orbits_ptr.is_null()
            || times_ptr.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return EMPYREAN_BUILTSYSTEM_NULL_POINTER;
        }
        let handle = unsafe { &*handle };
        let ctx_ref = unsafe { &*ctx };
        let config_ref = unsafe { &*config };
        let orbit_slice = unsafe { std::slice::from_raw_parts(orbits_ptr, num_orbits) };
        let times_slice = unsafe { std::slice::from_raw_parts(times_ptr, num_times) };

        // Identity guard (a): data identity. A handle built from a different
        // context must never silently integrate against this ctx's kernels.
        if !handle.system.built_from(ctx_ref.ephemeris_data()) {
            set_last_error(
                "BuiltSystem was not built from this context's ephemeris data — \
                 rebuild the handle after loading a new context or any kernel",
            );
            return EMPYREAN_BUILTSYSTEM_DATA_MISMATCH;
        }

        let (orbits, input_orbit_ids, input_object_ids) =
            match build_orbits_for_propagation(orbit_slice) {
                Ok(t) => t,
                Err(e) => {
                    set_last_error(&e);
                    return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
                }
            };
        let cfg = match build_propagation_config_from_c(config_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };
        let times: Vec<Epoch> = times_slice
            .iter()
            .map(|&t| Epoch::from_mjd_tdb(t))
            .collect();

        // Identity guards (b) + (c): dispatch through the frozen system. Its
        // check_fresh + check_key run before any integration and reject a
        // stale handle or a config that diverges on frame / force model /
        // divisor — mapped here to distinct loud codes. NO silent rebuild.
        let prop_result = match handle.system.propagate(&orbits, &times, &cfg) {
            Ok(res) => res,
            Err(e) => return map_prop_error_to_code(&e),
        };

        marshal_propagation_result(
            prop_result,
            times.len(),
            &input_orbit_ids,
            &input_object_ids,
            result_out,
        )
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_builtsystem_propagate");
            EMPYREAN_BUILTSYSTEM_PANIC
        }
    }
}

/// Generate predicted ephemeris for `orbits` and `observers` through the
/// pre-built handle.
///
/// Signature parallels the one-shot
/// [`empyrean_generate_ephemeris`](crate::ephemeris::empyrean_generate_ephemeris)
/// but takes `(handle, ctx, ...)`. Runs the same identity guard as
/// [`empyrean_builtsystem_propagate`] before dispatch; on pass the result is
/// bit-identical to the one-shot. Note the C-ABI ephemeris config carries no
/// divisor knob, so a handle frozen at a non-default divisor is rejected here
/// with [`EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR`] rather than served under
/// the wrong dynamics — build the handle with the default divisor for
/// ephemeris reuse.
///
/// On success populates `result_out`; free it with
/// [`empyrean_ephemeris_result_free`](crate::ephemeris::empyrean_ephemeris_result_free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_builtsystem_generate_ephemeris(
    handle: *const EmpyreanBuiltSystem,
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    observers_ptr: *const EmpyreanObserver,
    num_observers: usize,
    config: *const EmpyreanEphemerisConfig,
    result_out: *mut EmpyreanEphemerisResult,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null()
            || ctx.is_null()
            || orbits_ptr.is_null()
            || observers_ptr.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return EMPYREAN_BUILTSYSTEM_NULL_POINTER;
        }
        let handle = unsafe { &*handle };
        let ctx_ref = unsafe { &*ctx };
        let cfg_ref = unsafe { &*config };
        let orbit_slice = unsafe { std::slice::from_raw_parts(orbits_ptr, num_orbits) };
        let observer_slice = unsafe { std::slice::from_raw_parts(observers_ptr, num_observers) };

        // Identity guard (a): data identity.
        if !handle.system.built_from(ctx_ref.ephemeris_data()) {
            set_last_error(
                "BuiltSystem was not built from this context's ephemeris data — \
                 rebuild the handle after loading a new context or any kernel",
            );
            return EMPYREAN_BUILTSYSTEM_DATA_MISMATCH;
        }

        let orbits = match build_orbits_for_ephemeris(orbit_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };
        let observers = match build_observers_from_c(observer_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };
        let config = match build_ephemeris_config_from_c(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };

        // Identity guards (b) + (c): dispatch through the frozen system's
        // optical forward model (check_fresh + check_key run inside).
        let eph_result = match handle.system.generate_optical(&orbits, &observers, &config) {
            Ok(res) => res,
            Err(e) => return map_eph_error_to_code(&e),
        };

        marshal_ephemeris_result(&eph_result, result_out)
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_builtsystem_generate_ephemeris");
            EMPYREAN_BUILTSYSTEM_PANIC
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Provenance
// ────────────────────────────────────────────────────────────────────

/// Populate `out` with a full reproducibility summary of the handle's frozen
/// force model and its captured kernel manifest.
///
/// Every field is populated from the system description and the manifest
/// snapshot — no field is left defaulted. Returns
/// [`EMPYREAN_BUILTSYSTEM_OK`] on success. The caller owns the heap arrays
/// inside `out` and must release them with
/// [`empyrean_builtsystem_description_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_builtsystem_describe(
    handle: *const EmpyreanBuiltSystem,
    out: *mut EmpyreanSystemDescription,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || out.is_null() {
            set_last_error("null pointer argument");
            return EMPYREAN_BUILTSYSTEM_NULL_POINTER;
        }
        let handle = unsafe { &*handle };
        let desc = handle.system.describe();

        let force_model = match empyrean_core::ForceModelTier::try_from(desc.force_model) {
            Ok(f) => force_model_facade_to_int(f),
            Err(e) => {
                set_last_error(&e.to_string());
                return EMPYREAN_BUILTSYSTEM_INVALID_ARGUMENT;
            }
        };
        let frame = frame_to_int(desc.frame);

        // Perturber origins → owned NAIF-id array.
        let naifs: Vec<i32> = desc.perturber_origins.iter().map(|o| o.naif_id()).collect();
        let (perturber_origins, num_perturbers) = if naifs.is_empty() {
            (std::ptr::null_mut(), 0usize)
        } else {
            let layout = std::alloc::Layout::array::<i32>(naifs.len()).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut i32;
            if ptr.is_null() {
                set_last_error("allocation failed for perturber origins");
                return EMPYREAN_BUILTSYSTEM_ALLOC;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(naifs.as_ptr(), ptr, naifs.len());
            }
            (ptr, naifs.len())
        };

        // Kernel manifest → owned record array (each with owned C strings).
        let (kernels, num_kernels) = match alloc_kernel_records(&handle.kernel_manifest) {
            Ok(pair) => pair,
            Err(()) => {
                free_i32_array(perturber_origins, num_perturbers);
                set_last_error("allocation failed for kernel manifest");
                return EMPYREAN_BUILTSYSTEM_ALLOC;
            }
        };

        unsafe {
            *out = EmpyreanSystemDescription {
                force_model,
                frame,
                encounter_timescale_divisor: desc.encounter_timescale_divisor,
                relativistic: desc.relativistic as u8,
                asteroids: desc.asteroids as u8,
                has_bpc: desc.has_bpc as u8,
                perturber_origins,
                num_perturbers,
                kernels,
                num_kernels,
            };
        }
        EMPYREAN_BUILTSYSTEM_OK
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_builtsystem_describe");
            EMPYREAN_BUILTSYSTEM_PANIC
        }
    }
}

/// Free the heap arrays inside a description populated by
/// [`empyrean_builtsystem_describe`] (the perturber-id array and the kernel
/// records with their C strings). After this returns `desc` is
/// zero-initialized — safe to drop on the caller's stack. Passing null is a
/// no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_builtsystem_description_free(
    desc: *mut EmpyreanSystemDescription,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if desc.is_null() {
            return;
        }
        let d = unsafe { &mut *desc };
        free_i32_array(d.perturber_origins, d.num_perturbers);
        if !d.kernels.is_null() && d.num_kernels > 0 {
            for i in 0..d.num_kernels {
                let rec = unsafe { &*d.kernels.add(i) };
                unsafe {
                    free_c_str(rec.path);
                    free_c_str(rec.sha256);
                    free_c_str(rec.name);
                }
            }
            let layout = std::alloc::Layout::array::<EmpyreanKernelRecord>(d.num_kernels).unwrap();
            unsafe {
                std::alloc::dealloc(d.kernels as *mut u8, layout);
            }
        }
        d.perturber_origins = std::ptr::null_mut();
        d.num_perturbers = 0;
        d.kernels = std::ptr::null_mut();
        d.num_kernels = 0;
    }));
}

// ────────────────────────────────────────────────────────────────────
// Helpers (local to this module)
// ────────────────────────────────────────────────────────────────────

/// Map a handle-dispatch [`PropagationError`] onto its identity-guard C code.
/// The `KeyMismatch` axes and staleness each get a distinct loud code; every
/// other propagation failure surfaces its message under the generic
/// propagation code — never a silent success.
fn map_prop_error_to_code(e: &PropagationError) -> i32 {
    match e {
        PropagationError::KeyMismatch(SystemKeyMismatch::Frame { .. }) => {
            set_last_error(&format!("BuiltSystem key mismatch: {e}"));
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME
        }
        PropagationError::KeyMismatch(SystemKeyMismatch::ForceModel { .. }) => {
            set_last_error(&format!("BuiltSystem key mismatch: {e}"));
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FORCE_MODEL
        }
        PropagationError::KeyMismatch(SystemKeyMismatch::Divisor { .. }) => {
            set_last_error(&format!("BuiltSystem key mismatch: {e}"));
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR
        }
        PropagationError::StaleEphemerisData { .. } => {
            set_last_error(&format!("BuiltSystem is stale: {e}"));
            EMPYREAN_BUILTSYSTEM_STALE
        }
        other => {
            set_last_error(&format!("BuiltSystem propagation failed: {other}"));
            EMPYREAN_BUILTSYSTEM_PROPAGATION
        }
    }
}

/// Map a handle-dispatch [`EphemerisGenerationError`] onto its C code,
/// delegating the identity-guard axes to [`map_prop_error_to_code`].
fn map_eph_error_to_code(e: &EphemerisGenerationError) -> i32 {
    match e {
        EphemerisGenerationError::Propagation(inner) => map_prop_error_to_code(inner),
        other => {
            set_last_error(&format!("BuiltSystem ephemeris generation failed: {other}"));
            EMPYREAN_BUILTSYSTEM_PROPAGATION
        }
    }
}

/// Map the facade force-model tier to its C code.
fn force_model_facade_to_int(fm: empyrean_core::ForceModelTier) -> i32 {
    use empyrean_core::ForceModelTier as F;
    match fm {
        F::Approximate => 0,
        F::Basic => 1,
        F::Standard => 2,
        // `ForceModelTier` is #[non_exhaustive]; a tier added upstream that
        // this release does not yet map surfaces as -1, not a wrong id.
        _ => -1,
    }
}

/// Map a [`KernelKind`] to its C tag. Exhaustive on purpose: a new kernel
/// kind upstream is a compile error here, forcing the tag table to grow
/// rather than silently collapsing into a wrong category.
fn kernel_kind_to_int(kind: KernelKind) -> i32 {
    match kind {
        KernelKind::Spk => EMPYREAN_KERNEL_KIND_SPK,
        KernelKind::Bpc => EMPYREAN_KERNEL_KIND_BPC,
        KernelKind::Tpc => EMPYREAN_KERNEL_KIND_TPC,
        KernelKind::Gravity => EMPYREAN_KERNEL_KIND_GRAVITY,
        KernelKind::ObsCodes => EMPYREAN_KERNEL_KIND_OBSCODES,
    }
}

/// Allocate and populate a C array of [`EmpyreanKernelRecord`] from the
/// captured manifest. Each provenance variant populates exactly its own
/// fields (file: path/sha256/bytes; built-in: name; in-memory: none) — no
/// field is fabricated. Returns `(null, 0)` for an empty manifest, or
/// `Err(())` on allocation failure.
fn alloc_kernel_records(
    manifest: &[KernelRecord],
) -> Result<(*mut EmpyreanKernelRecord, usize), ()> {
    let n = manifest.len();
    if n == 0 {
        return Ok((std::ptr::null_mut(), 0));
    }
    let layout = std::alloc::Layout::array::<EmpyreanKernelRecord>(n).map_err(|_| ())?;
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanKernelRecord;
    if ptr.is_null() {
        return Err(());
    }
    for (i, rec) in manifest.iter().enumerate() {
        let kind = kernel_kind_to_int(rec.kind);
        let (provenance, path, sha256, bytes, name) = match &rec.provenance {
            KernelProvenance::File {
                path,
                sha256,
                bytes,
            } => (
                EMPYREAN_KERNEL_PROVENANCE_FILE,
                to_c_str(&path.to_string_lossy()),
                to_c_str(sha256),
                *bytes,
                std::ptr::null_mut(),
            ),
            KernelProvenance::InMemory => (
                EMPYREAN_KERNEL_PROVENANCE_IN_MEMORY,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0u64,
                std::ptr::null_mut(),
            ),
            KernelProvenance::BuiltIn { name } => (
                EMPYREAN_KERNEL_PROVENANCE_BUILT_IN,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0u64,
                to_c_str(name),
            ),
        };
        unsafe {
            ptr.add(i).write(EmpyreanKernelRecord {
                kind,
                provenance,
                path,
                sha256,
                bytes,
                name,
            });
        }
    }
    Ok((ptr, n))
}

/// Free an `i32` array allocated by [`empyrean_builtsystem_describe`].
fn free_i32_array(ptr: *mut i32, len: usize) {
    if !ptr.is_null() && len > 0 {
        let layout = std::alloc::Layout::array::<i32>(len).unwrap();
        unsafe {
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Structural checks (no ephemeris data required) ──────────

    /// The identity-guard axes must map to *distinct* codes: a caller that
    /// only pattern-matches numeric codes can tell data / frame / force-model
    /// / divisor / stale apart. Guards against an accidental collision if the
    /// table is edited.
    #[test]
    fn identity_guard_codes_are_distinct() {
        let codes = [
            EMPYREAN_BUILTSYSTEM_DATA_MISMATCH,
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME,
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FORCE_MODEL,
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR,
            EMPYREAN_BUILTSYSTEM_STALE,
        ];
        let mut sorted = codes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            codes.len(),
            "identity-guard codes must be distinct"
        );
    }

    // ── Full round-trip (gated on ephemeris data) ───────────────

    fn try_context() -> Option<EmpyreanContext> {
        empyrean_core::Context::from_data_dir(None).ok()
    }

    fn last_err() -> String {
        let p = crate::empyrean_last_error();
        if p.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    /// A gravity-only heliocentric Cartesian orbit (AU, AU/day).
    fn base_orbit() -> EmpyreanOrbit {
        EmpyreanOrbit {
            state: crate::CoordinateState {
                epoch_mjd_tdb: 59000.0,
                elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
                covariance: [[0.0; 6]; 6],
                has_covariance: 0,
                representation: 0, // Cartesian
                frame: 0,          // ICRF
                origin: 10,        // Sun (NAIF)
            },
            orbit_id: std::ptr::null(),
            object_id: std::ptr::null(),
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ng_alpha: 0.0,
            ng_r0: 0.0,
            ng_m: 0.0,
            ng_n: 0.0,
            ng_k: 0.0,
            non_grav_dt: f64::NAN,
            non_grav_dt_variance: f64::NAN,
            has_non_grav_covariance: 0,
            non_grav_covariance: [[0.0; 3]; 3],
            phot_system: -1,
            h_mag: f64::NAN,
            slope1: 0.0,
            slope2: 0.0,
            thrust_arcs: std::ptr::null(),
            n_thrust_arcs: 0,
            dv_corrections: std::ptr::null(),
            n_dv_corrections: 0,
            correction_covariances: std::ptr::null(),
            n_correction_covariances: 0,
        }
    }

    /// A Standard-tier, first-order config. `zeroed` gives the right
    /// sentinels for every field except the dt_* NaN-means-auto slots.
    fn standard_first_config() -> EmpyreanPropagationConfig {
        let mut cfg: EmpyreanPropagationConfig = unsafe { std::mem::zeroed() };
        cfg.force_model = 2; // Standard
        cfg.frame = 0; // ICRF
        cfg.uncertainty_method.tag = crate::propagate::EMPYREAN_UNCERTAINTY_FIRST;
        cfg.advanced.dt_initial = f64::NAN;
        cfg.advanced.dt_min = f64::NAN;
        cfg
    }

    /// A handle propagation is bit-identical to the one-shot `empyrean_propagate`
    /// for the same orbit/config — the handle only changes *when* the force
    /// model is assembled, not the numerics.
    #[test]
    fn builtsystem_propagate_matches_one_shot() {
        let ctx = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_propagate_matches_one_shot: no data dir");
                return;
            }
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;
        let cfg = standard_first_config();
        let times = [59000.0f64, 59010.0, 59030.0];
        let orbits = [base_orbit()];

        // One-shot baseline.
        let mut one_shot: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            crate::propagate::empyrean_propagate(
                ctx_ptr,
                orbits.as_ptr(),
                1,
                times.as_ptr(),
                times.len(),
                &cfg,
                &mut one_shot,
            )
        };
        assert_eq!(rc, 0, "one-shot propagate failed: {}", last_err());

        // Through the handle.
        let mut handle: *mut EmpyreanBuiltSystem = std::ptr::null_mut();
        let hc = unsafe { empyrean_builtsystem_new(ctx_ptr, 2, 0, 0.0, &mut handle) };
        assert_eq!(
            hc,
            EMPYREAN_BUILTSYSTEM_OK,
            "handle build failed: {}",
            last_err()
        );
        assert!(!handle.is_null());

        let mut via_handle: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let pc = unsafe {
            empyrean_builtsystem_propagate(
                handle,
                ctx_ptr,
                orbits.as_ptr(),
                1,
                times.as_ptr(),
                times.len(),
                &cfg,
                &mut via_handle,
            )
        };
        assert_eq!(
            pc,
            EMPYREAN_BUILTSYSTEM_OK,
            "handle propagate failed: {}",
            last_err()
        );

        assert_eq!(one_shot.num_states, via_handle.num_states);
        let n = one_shot.num_states;
        assert_eq!(n, times.len());
        for i in 0..n {
            let a = unsafe { &*one_shot.states.add(i) };
            let b = unsafe { &*via_handle.states.add(i) };
            assert_eq!(a.epoch_mjd_tdb, b.epoch_mjd_tdb, "epoch[{i}]");
            assert_eq!(a.x, b.x, "x[{i}] bit-identical");
            assert_eq!(a.y, b.y, "y[{i}] bit-identical");
            assert_eq!(a.z, b.z, "z[{i}] bit-identical");
            assert_eq!(a.vx, b.vx, "vx[{i}] bit-identical");
            assert_eq!(a.vy, b.vy, "vy[{i}] bit-identical");
            assert_eq!(a.vz, b.vz, "vz[{i}] bit-identical");
        }

        unsafe {
            crate::propagate::empyrean_propagation_result_free(&mut one_shot);
            crate::propagate::empyrean_propagation_result_free(&mut via_handle);
            empyrean_builtsystem_free(handle);
        }
    }

    /// `describe()` reports the frozen force-model/frame/divisor + a
    /// non-empty kernel manifest whose SHA-256 for a FILE record matches an
    /// independent digest of that file.
    #[test]
    fn builtsystem_describe_reports_provenance() {
        let ctx = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_describe_reports_provenance: no data dir");
                return;
            }
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;

        let mut handle: *mut EmpyreanBuiltSystem = std::ptr::null_mut();
        let hc = unsafe { empyrean_builtsystem_new(ctx_ptr, 2, 0, 0.0, &mut handle) };
        assert_eq!(
            hc,
            EMPYREAN_BUILTSYSTEM_OK,
            "handle build failed: {}",
            last_err()
        );

        let mut desc: EmpyreanSystemDescription = unsafe { std::mem::zeroed() };
        let dc = unsafe { empyrean_builtsystem_describe(handle, &mut desc) };
        assert_eq!(
            dc,
            EMPYREAN_BUILTSYSTEM_OK,
            "describe failed: {}",
            last_err()
        );

        assert_eq!(desc.force_model, 2, "Standard tier");
        assert_eq!(desc.frame, 0, "ICRF");
        assert_eq!(
            desc.encounter_timescale_divisor, 1000.0,
            "engine default divisor"
        );
        assert_eq!(desc.relativistic, 1, "Standard tier includes GR");
        assert_eq!(desc.asteroids, 1, "Standard tier includes N16 asteroids");
        assert_eq!(desc.has_bpc, 1, "Standard tier loads a BPC");
        assert!(desc.num_perturbers > 0, "Standard tier has perturbers");
        assert!(!desc.perturber_origins.is_null());
        assert!(desc.num_kernels > 0, "kernel manifest is non-empty");
        assert!(!desc.kernels.is_null());

        // Pick the smallest FILE-provenance record and re-digest it.
        use sha2::{Digest, Sha256};
        let mut smallest: Option<(String, String, u64)> = None;
        for i in 0..desc.num_kernels {
            let rec = unsafe { &*desc.kernels.add(i) };
            if rec.provenance != EMPYREAN_KERNEL_PROVENANCE_FILE {
                continue;
            }
            assert!(!rec.path.is_null(), "FILE record has a path");
            assert!(!rec.sha256.is_null(), "FILE record has a sha256");
            let path = unsafe { std::ffi::CStr::from_ptr(rec.path) }
                .to_str()
                .unwrap()
                .to_string();
            let sha = unsafe { std::ffi::CStr::from_ptr(rec.sha256) }
                .to_str()
                .unwrap()
                .to_string();
            assert_eq!(sha.len(), 64, "sha256 is 64 lowercase-hex chars");
            if smallest
                .as_ref()
                .map(|(_, _, b)| rec.bytes < *b)
                .unwrap_or(true)
            {
                smallest = Some((path, sha, rec.bytes));
            }
        }
        let (path, manifest_sha, manifest_bytes) =
            smallest.expect("expected at least one FILE-provenance kernel");
        let mut file = std::fs::File::open(&path).expect("open kernel file");
        let mut hasher = Sha256::new();
        let hashed_bytes = std::io::copy(&mut file, &mut hasher).expect("hash kernel file");
        let independent = format!("{:x}", hasher.finalize());
        assert_eq!(
            hashed_bytes, manifest_bytes,
            "byte count matches for {path}"
        );
        assert_eq!(
            independent, manifest_sha,
            "manifest SHA-256 matches an independent digest of {path}"
        );

        unsafe {
            empyrean_builtsystem_description_free(&mut desc);
            empyrean_builtsystem_free(handle);
        }
    }

    /// Ephemeris generation through the handle is bit-identical to the
    /// one-shot `empyrean_generate_ephemeris` for the same orbit / observer /
    /// config — exercising the second forward-model method and the shared
    /// ephemeris marshaling.
    #[test]
    fn builtsystem_generate_ephemeris_matches_one_shot() {
        let ctx = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_generate_ephemeris_matches_one_shot: no data dir");
                return;
            }
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;

        // Standard-tier, first-order ephemeris config with diagnostics on.
        let mut cfg: EmpyreanEphemerisConfig = unsafe { std::mem::zeroed() };
        cfg.propagation.force_model = 2;
        cfg.propagation.frame = 0;
        cfg.propagation.uncertainty_method.tag = crate::propagate::EMPYREAN_UNCERTAINTY_FIRST;
        cfg.propagation.advanced.dt_initial = f64::NAN;
        cfg.propagation.advanced.dt_min = f64::NAN;
        cfg.compute_diagnostics = 1;

        let orbits = [base_orbit()];
        // A geocentric-ish SSB observer state (AU, AU/day) at the orbit epoch.
        let observers = [EmpyreanObserver {
            obs_code: [b'5', b'0', b'0', 0],
            epoch_mjd_tdb: 59000.0,
            x: 0.9,
            y: -0.42,
            z: -0.18,
            vx: 0.0075,
            vy: 0.0148,
            vz: 0.0064,
            observing_night: -1,
        }];

        // One-shot baseline.
        let mut one_shot: EmpyreanEphemerisResult = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            crate::ephemeris::empyrean_generate_ephemeris(
                ctx_ptr,
                orbits.as_ptr(),
                1,
                observers.as_ptr(),
                1,
                &cfg,
                &mut one_shot,
            )
        };
        assert_eq!(rc, 0, "one-shot generate_ephemeris failed: {}", last_err());

        // Through the handle.
        let mut handle: *mut EmpyreanBuiltSystem = std::ptr::null_mut();
        let hc = unsafe { empyrean_builtsystem_new(ctx_ptr, 2, 0, 0.0, &mut handle) };
        assert_eq!(
            hc,
            EMPYREAN_BUILTSYSTEM_OK,
            "handle build failed: {}",
            last_err()
        );

        let mut via_handle: EmpyreanEphemerisResult = unsafe { std::mem::zeroed() };
        let ec = unsafe {
            empyrean_builtsystem_generate_ephemeris(
                handle,
                ctx_ptr,
                orbits.as_ptr(),
                1,
                observers.as_ptr(),
                1,
                &cfg,
                &mut via_handle,
            )
        };
        assert_eq!(
            ec,
            EMPYREAN_BUILTSYSTEM_OK,
            "handle generate_ephemeris failed: {}",
            last_err()
        );

        assert_eq!(one_shot.num_entries, via_handle.num_entries);
        assert!(one_shot.num_entries > 0, "expected at least one entry");
        for i in 0..one_shot.num_entries {
            let a = unsafe { &*one_shot.entries.add(i) };
            let b = unsafe { &*via_handle.entries.add(i) };
            assert_eq!(a.ra_deg, b.ra_deg, "ra[{i}] bit-identical");
            assert_eq!(a.dec_deg, b.dec_deg, "dec[{i}] bit-identical");
            assert_eq!(a.rho_au, b.rho_au, "rho[{i}] bit-identical");
        }

        unsafe {
            crate::ephemeris::empyrean_ephemeris_result_free(&mut one_shot);
            crate::ephemeris::empyrean_ephemeris_result_free(&mut via_handle);
            empyrean_builtsystem_free(handle);
        }
    }

    /// The identity guard fires loudly and by-axis for a per-call config that
    /// diverges from the frozen key — never a silent rebuild under the wrong
    /// dynamics.
    #[test]
    fn builtsystem_guard_fires_on_key_mismatch() {
        let ctx = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_guard_fires_on_key_mismatch: no data dir");
                return;
            }
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;

        let mut handle: *mut EmpyreanBuiltSystem = std::ptr::null_mut();
        let hc = unsafe { empyrean_builtsystem_new(ctx_ptr, 2, 0, 0.0, &mut handle) };
        assert_eq!(
            hc,
            EMPYREAN_BUILTSYSTEM_OK,
            "handle build failed: {}",
            last_err()
        );

        let times = [59000.0f64, 59010.0];
        let orbits = [base_orbit()];

        // Force-model axis: frozen Standard, call requests Basic.
        let mut cfg_fm = standard_first_config();
        cfg_fm.force_model = 1; // Basic
        let mut out_fm: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let rc_fm = unsafe {
            empyrean_builtsystem_propagate(
                handle,
                ctx_ptr,
                orbits.as_ptr(),
                1,
                times.as_ptr(),
                times.len(),
                &cfg_fm,
                &mut out_fm,
            )
        };
        assert_eq!(
            rc_fm,
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FORCE_MODEL,
            "force-model mismatch must fire by axis: {}",
            last_err()
        );

        // Frame axis: frozen ICRF, call requests EclipticJ2000.
        let mut cfg_fr = standard_first_config();
        cfg_fr.frame = 1; // EclipticJ2000
        let mut out_fr: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let rc_fr = unsafe {
            empyrean_builtsystem_propagate(
                handle,
                ctx_ptr,
                orbits.as_ptr(),
                1,
                times.as_ptr(),
                times.len(),
                &cfg_fr,
                &mut out_fr,
            )
        };
        assert_eq!(
            rc_fr,
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME,
            "frame mismatch must fire by axis: {}",
            last_err()
        );

        // Divisor axis: frozen at the engine default (built with 0.0), call
        // requests a non-default encounter-timescale divisor.
        let mut cfg_dv = standard_first_config();
        cfg_dv.advanced.encounter_timescale_divisor = 500.0;
        let mut out_dv: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let rc_dv = unsafe {
            empyrean_builtsystem_propagate(
                handle,
                ctx_ptr,
                orbits.as_ptr(),
                1,
                times.as_ptr(),
                times.len(),
                &cfg_dv,
                &mut out_dv,
            )
        };
        assert_eq!(
            rc_dv,
            EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR,
            "divisor mismatch must fire by axis: {}",
            last_err()
        );

        unsafe { empyrean_builtsystem_free(handle) };
    }

    /// A handle used with a *different* context (distinct ephemeris-data
    /// instance) is rejected as a data mismatch — never silently rebuilt
    /// against the foreign kernels.
    #[test]
    fn builtsystem_guard_fires_on_data_mismatch() {
        let ctx_a = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_guard_fires_on_data_mismatch: no data dir");
                return;
            }
        };
        let ctx_b = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping builtsystem_guard_fires_on_data_mismatch: second context");
                return;
            }
        };
        let a_ptr: *const EmpyreanContext = &ctx_a;
        let b_ptr: *const EmpyreanContext = &ctx_b;

        let mut handle: *mut EmpyreanBuiltSystem = std::ptr::null_mut();
        let hc = unsafe { empyrean_builtsystem_new(a_ptr, 2, 0, 0.0, &mut handle) };
        assert_eq!(
            hc,
            EMPYREAN_BUILTSYSTEM_OK,
            "handle build failed: {}",
            last_err()
        );

        let times = [59000.0f64, 59010.0];
        let orbits = [base_orbit()];
        let cfg = standard_first_config();

        // Wrong context → loud data mismatch.
        let mut out_bad: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let rc_bad = unsafe {
            empyrean_builtsystem_propagate(
                handle,
                b_ptr,
                orbits.as_ptr(),
                1,
                times.as_ptr(),
                times.len(),
                &cfg,
                &mut out_bad,
            )
        };
        assert_eq!(
            rc_bad,
            EMPYREAN_BUILTSYSTEM_DATA_MISMATCH,
            "foreign context must be rejected: {}",
            last_err()
        );

        // Correct context → succeeds (proves the guard is specific, not
        // rejecting everything).
        let mut out_ok: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let rc_ok = unsafe {
            empyrean_builtsystem_propagate(
                handle,
                a_ptr,
                orbits.as_ptr(),
                1,
                times.as_ptr(),
                times.len(),
                &cfg,
                &mut out_ok,
            )
        };
        assert_eq!(
            rc_ok,
            EMPYREAN_BUILTSYSTEM_OK,
            "correct context must pass: {}",
            last_err()
        );

        unsafe {
            crate::propagate::empyrean_propagation_result_free(&mut out_ok);
            empyrean_builtsystem_free(handle);
        }
    }
}
