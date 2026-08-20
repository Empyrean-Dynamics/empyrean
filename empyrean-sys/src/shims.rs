//! Free-function shims over the dynamically loaded engine, so callers use
//! `empyrean_sys::empyrean_*` exactly as with a statically linked library.
//!
//! GENERATED — every item here is derived mechanically from the
//! `impl EmpyreanLib` block in `bindings.rs` (same doc text, same signature
//! minus the receiver), so the two cannot drift. Regenerate alongside
//! `bindings.rs`; do not hand-edit.
use crate::{lib, *};
/** Return a pointer to the last error message (thread-local, null-terminated).

The pointer is valid until the next call that sets an error on the same
thread.*/
#[inline]
pub unsafe fn empyrean_last_error() -> *const ::std::os::raw::c_char {
    unsafe { lib().empyrean_last_error() }
}
/** Create a **minimal** `EmpyreanContext` from a DE440 SPK file and a
GM TPC file.

Loads ONLY the planetary ephemeris and gravitational parameters —
no Earth/Moon BPC kernels, no SB441-N16 asteroid perturbers, no
MPC observatory codes, no Earth gravity field. This is sufficient
for coordinate transforms and basic propagation under the
`Approximate` force model, but is **not** enough for production
orbit propagation, orbit determination, or topocentric ephemeris
generation. Most callers should use
[`empyrean_context_from_data_dir`] instead, which loads the full
Standard-tier kernel set (downloading any missing files).

Use [`empyrean_context_with_spk`] to chain additional SPK kernels
(e.g. SB441-N16) onto a context built by this function.

Returns a heap-allocated pointer on success, or null on error.
Call `empyrean_last_error()` to retrieve the error message when null is
returned.  The caller owns the returned pointer and must free it with
`empyrean_context_free()`.*/
#[inline]
pub unsafe fn empyrean_context_new_minimal(
    de440_path: *const ::std::os::raw::c_char,
    gm_path: *const ::std::os::raw::c_char,
) -> *mut EmpyreanContext {
    unsafe { lib().empyrean_context_new_minimal(de440_path, gm_path) }
}
/** Load an additional SPK kernel into an existing context, in place.

Useful for layering SB441-N16 asteroid perturbers or spacecraft
SPK kernels (JWST, Gaia, custom probes) on top of a context built
by [`empyrean_context_new_minimal`] or [`empyrean_context_from_data_dir`].
The merged context picks up the new kernel's body coverage on top
of what was already loaded.

Returns 0 on success; negative error code on failure. The context
pointer remains valid and unchanged when this function returns
non-zero — failure does not invalidate `ctx`.*/
#[inline]
pub unsafe fn empyrean_context_with_spk(
    ctx: *mut EmpyreanContext,
    spk_path: *const ::std::os::raw::c_char,
) -> i32 {
    unsafe { lib().empyrean_context_with_spk(ctx, spk_path) }
}
/** Create a new `EmpyreanContext` from a data directory.

Loads the full Standard-tier kernel set (DE440, SB441-N16, Earth/Moon
BPCs, GM, MPC observatory codes) from `data_dir`, downloading any
missing files. Pass null for `data_dir` to use the platform data
directory (`~/.local/share/empyrean/data` on Linux,
`~/Library/Application Support/empyrean/data` on macOS).

Returns a heap-allocated pointer on success, or null on error.
Call `empyrean_last_error()` to retrieve the error message when null is
returned. The caller owns the returned pointer and must free it with
`empyrean_context_free()`.*/
#[inline]
pub unsafe fn empyrean_context_from_data_dir(
    data_dir: *const ::std::os::raw::c_char,
) -> *mut EmpyreanContext {
    unsafe { lib().empyrean_context_from_data_dir(data_dir) }
}
/** Create a new `EmpyreanContext` from a data directory under explicit
options — the superset of [`empyrean_context_from_data_dir`].

Pass `options = NULL` (or a `memset(0)` struct) for exactly the
behaviour of [`empyrean_context_from_data_dir`]: acquire and load the
full Standard-tier kernel set, downloading anything missing.

# Strict offline

`options->refresh = EMPYREAN_DATA_REFRESH_OFF` is the reason this
entry point exists. It resolves the tier's kernels from `data_dir`
alone and fails if any is absent, naming **every** missing file
through [`empyrean_missing_data_files`] — villeneuve's kernels plus
the catalog-debiasing table — so an offline context can never come up
silently missing part of its data. No lower-tier fallback, no
download-just-this-one, no partially-loaded context.

This entry point reads **no environment variable** to decide
`refresh`. `EMPYREAN_OFFLINE` is a *channel*-level floor applied by
the layers above (the Rust wrapper, the CLI, the Python package); the
C ABI honours exactly what the caller passed.

Returns a heap-allocated pointer on success, or null on error. Call
`empyrean_last_error()` for the message, and
[`empyrean_missing_data_files`] for the structured file list when the
failure was a strict-offline shortfall. The caller owns the returned
pointer and must free it with `empyrean_context_free()`.*/
#[inline]
pub unsafe fn empyrean_context_from_data_dir_with(
    data_dir: *const ::std::os::raw::c_char,
    options: *const EmpyreanDataDirOptions,
) -> *mut EmpyreanContext {
    unsafe { lib().empyrean_context_from_data_dir_with(data_dir, options) }
}
/** Provision the complete Standard-tier kernel set into `data_dir`
**without building a context**.

Runs the same download-and-cache pass
[`empyrean_context_from_data_dir`] performs, but stops at the resolved
kernel paths: nothing is loaded or parsed, and no `EmpyreanContext` is
allocated. This is the cheap provisioning primitive behind the
wrapper's `download_data` — it exists so a caller that only wants to
warm a data directory does not pay for a full Standard-tier context
build (and discard). After it returns `0`, a later
[`empyrean_context_from_data_dir`] over the same directory loads with
no further downloads.

Pass `NULL` for `data_dir` to provision the platform data directory
(`~/.local/share/empyrean/data` on Linux, `~/Library/Application
Support/empyrean/data` on macOS).

Always reaches the network when a kernel is missing or its upstream
copy moved (the refreshing path); on a warm, complete directory it
issues only the staleness checks and downloads nothing.

Returns `0` on success. On failure returns the engine error code:
`-1` for a non-UTF-8 `data_dir` or another invalid argument, `-99` on
a caught panic, and `-2` when a required resource could not be
obtained. Call `empyrean_last_error()` for the message on any
non-zero return.

# Reading a `-2`

`-2` is the missing-data **category**, and two of its shapes carry a
signature this entry point's callers can key on:

* **Files absent, no fetch attempted** — `num_files > 0` from
  [`empyrean_missing_data_files`] names every file, and the message
  reads `"Missing data files: <names>"`. The
  remedy is to fetch or stage exactly those names.
* **A fetch was attempted and failed** — `num_files == 0` (the list
  is empty, which is not itself an error), and the message starts
  `"Data download failed: "`, naming the failing
  kernel by its URL. The remedy is connectivity, or — when the URL
  404s because an upstream rotated or withdrew a pinned kernel —
  staging that file by hand or moving to an engine whose pin is
  still served. Never local file repair.

The implications hold in one direction only: a populated list always
means the named-files shape, and the download prefix always means a
failed acquisition. A `-2` with an empty list and any other message
is one of the category's other residents (a kernel that read but
would not parse, a coverage gap) — the message states its own
remedy. An empty list after a `-2` therefore does **not** mean
"nothing was missing": read `empyrean_last_error()`.*/
#[inline]
pub unsafe fn empyrean_download_data(data_dir: *const ::std::os::raw::c_char) -> i32 {
    unsafe { lib().empyrean_download_data(data_dir) }
}
/** Retrieve the structured file list from the most recent
missing-data-files failure on this thread.

The companion to `empyrean_last_error()`: that returns the rendered
message, this returns the list the message was rendered from, so a
caller can fetch or report exactly those names instead of splitting a
string (file names may contain the separator).

Returns 0 and fills `out` on success. `out->num_files == 0` — with
`out->files` null — means the last error on this thread was not a
missing-data-files failure; it is not itself an error. Returns `-1`
for a null `out`, `-5` on allocation failure, `-99` on a caught panic.

The list is thread-local and is cleared by the next call that records
an error on this thread, so read it immediately after the failing
call. **The caller owns `out` and must release it with
[`empyrean_missing_data_files_free`].***/
#[inline]
pub unsafe fn empyrean_missing_data_files(out: *mut EmpyreanMissingDataFiles) -> i32 {
    unsafe { lib().empyrean_missing_data_files(out) }
}
/** Free an [`EmpyreanMissingDataFiles`] populated by
[`empyrean_missing_data_files`]. Passing a null or zeroed struct is a
no-op; the struct is left zeroed so a double free is safe.*/
#[inline]
pub unsafe fn empyrean_missing_data_files_free(out: *mut EmpyreanMissingDataFiles) {
    unsafe { lib().empyrean_missing_data_files_free(out) }
}
/**Free an `EmpyreanContext` previously returned by
`empyrean_context_from_data_dir`, `empyrean_context_from_data_dir_with`
or `empyrean_context_new_minimal`.

Passing null is a no-op.*/
#[inline]
pub unsafe fn empyrean_context_free(ctx: *mut EmpyreanContext) {
    unsafe { lib().empyrean_context_free(ctx) }
}
/** Return the platform XDG-compliant default data directory as a
heap-allocated, NUL-terminated UTF-8 string.

Mirrors the engine's `DataManager::new` resolution: honors
`EMPYREAN_DATA_DIR` first, then falls back to `dirs::data_dir()` —
`~/.local/share/empyrean/data/` on Linux, `~/Library/Application
Support/empyrean/data/` on macOS, `%APPDATA%\empyrean\data\` on
Windows. Cheap (no filesystem I/O).

Returns null on failure (non-UTF-8 path, NUL byte in path, panic).
Call `empyrean_last_error()` for details.

**The caller owns the returned pointer and must release it with
[`empyrean_string_free`].***/
#[inline]
pub unsafe fn empyrean_default_data_dir() -> *mut ::std::os::raw::c_char {
    unsafe { lib().empyrean_default_data_dir() }
}
/** Free a string returned by an empyrean C API function (e.g.,
[`empyrean_default_data_dir`], [`empyrean_version_string`]).

Passing null is a no-op. Passing any pointer not obtained from an
empyrean string-returning function is undefined behavior.*/
#[inline]
pub unsafe fn empyrean_string_free(s: *mut ::std::os::raw::c_char) {
    unsafe { lib().empyrean_string_free(s) }
}
/** Multi-line version report — `empyrean-core <ver>\nvilleneuve <ver>\n…`.

Mirrors [`empyrean_core::version_string`]. Useful for `--version`-style
output and for verifying the build provenance of a deployed cdylib.
Returns null on allocation failure (extremely unlikely — the strings
are short and `&'static` underneath); call `empyrean_last_error()` if
it does.

**The caller owns the returned pointer and must release it with
[`empyrean_string_free`].***/
#[inline]
pub unsafe fn empyrean_version_string() -> *mut ::std::os::raw::c_char {
    unsafe { lib().empyrean_version_string() }
}
/** Populate `out` with the per-crate versions of the empyrean stack.

Returns 0 on success, non-zero on failure (`empyrean_last_error()`
has the details). On failure `out` is left zero-initialized — no
allocation needs freeing.

**The caller owns the strings inside `out` and must release the
whole struct with [`empyrean_versions_free`] when done.***/
#[inline]
pub unsafe fn empyrean_versions(out: *mut EmpyreanVersions) -> i32 {
    unsafe { lib().empyrean_versions(out) }
}
/** Free the version strings inside `versions` (each was heap-allocated
by a previous successful [`empyrean_versions`] call). After this
returns, `versions` itself is zero-initialized — safe to drop on
the caller's stack.

Passing null is a no-op. Calling this twice on the same struct, or
passing a struct that wasn't populated by [`empyrean_versions`], is
undefined behavior.*/
#[inline]
pub unsafe fn empyrean_versions_free(versions: *mut EmpyreanVersions) {
    unsafe { lib().empyrean_versions_free(versions) }
}
/** Assemble a reusable force-model handle for `(force_model, frame)` from
`ctx`, freezing `encounter_timescale_divisor` into its key.

- `force_model`: tier code (0=Approximate, 1=Basic, 2=Standard).
- `frame`: output/integration frame code (0=ICRF, 1=EclipticJ2000, 2=ITRF93).
- `encounter_timescale_divisor`: pass `0.0` to freeze the engine default;
  any positive value freezes that divisor into the key *before* it is
  used to validate calls.
- `out`: receives the heap-allocated handle on success.

The handle snapshots `ctx`'s kernel manifest at construction so
[`empyrean_builtsystem_describe`] is self-contained. After you load any
additional kernel into `ctx`, rebuild the handle — a stale handle is
rejected loudly by every forward-model call, never silently reused.

Returns [`EMPYREAN_BUILTSYSTEM_OK`] on success; on error, `out` is left
untouched and `empyrean_last_error()` carries the message. The caller
owns the returned handle and must free it with
[`empyrean_builtsystem_free`].*/
#[inline]
pub unsafe fn empyrean_builtsystem_new(
    ctx: *const EmpyreanContext,
    force_model: i32,
    frame: i32,
    encounter_timescale_divisor: f64,
    out: *mut *mut EmpyreanBuiltSystem,
) -> i32 {
    unsafe {
        lib().empyrean_builtsystem_new(ctx, force_model, frame, encounter_timescale_divisor, out)
    }
}
/** Assemble a handle keyed for the **orbit-determination** entry points.

Equivalent to [`empyrean_builtsystem_new`] with the frame and the
encounter-timescale divisor an OD fit runs under already filled in, so
the recipe cannot be got wrong. Only the force-model tier is yours:
`ODConfig` exposes no frame or divisor knob, and the fit derives both
from the engine defaults.

Use this — not [`empyrean_builtsystem_new`] — for
[`empyrean_builtsystem_determine`] / `_evaluate` / `_refine`. A handle
built with `frame = 0` (ICRF), which every propagation example in this
header uses, is rejected on the OD path with
[`EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME`]: OD does not integrate in
ICRF, and the guard will not serve a fit under a frame the fit did not
ask for.

- `force_model`: tier code (0=Approximate, 1=Basic, 2=Standard). Must
  equal the `force_model` on the `EmpyreanODConfig` you pass to the OD
  call.
- `out`: receives the heap-allocated handle on success.

The handle it returns is an ordinary one — free it with
[`empyrean_builtsystem_free`], describe it with
[`empyrean_builtsystem_describe`], and use it for propagation too if a
matching config asks for the same frame.*/
#[inline]
pub unsafe fn empyrean_builtsystem_new_for_od(
    ctx: *const EmpyreanContext,
    force_model: i32,
    out: *mut *mut EmpyreanBuiltSystem,
) -> i32 {
    unsafe { lib().empyrean_builtsystem_new_for_od(ctx, force_model, out) }
}
/** Free a handle previously returned by [`empyrean_builtsystem_new`].
Passing null is a no-op.*/
#[inline]
pub unsafe fn empyrean_builtsystem_free(handle: *mut EmpyreanBuiltSystem) {
    unsafe { lib().empyrean_builtsystem_free(handle) }
}
/** Propagate `orbits` to `times` through the pre-built handle.

Signature parallels the one-shot
[`empyrean_propagate`](crate::propagate::empyrean_propagate) but takes
`(handle, ctx, ...)`. Before dispatch the identity guard runs: the handle
must have been built from `ctx`'s ephemeris data
([`EMPYREAN_BUILTSYSTEM_DATA_MISMATCH`]); the config must match the frozen
key ([`EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME`] /
`_FORCE_MODEL` / `_DIVISOR`); and the data must be unmutated since build
([`EMPYREAN_BUILTSYSTEM_STALE`]). On pass, the result is bit-identical to
the one-shot with the same config.

On success populates `result_out`; free it with
[`empyrean_propagation_result_free`](crate::propagate::empyrean_propagation_result_free).*/
#[inline]
pub unsafe fn empyrean_builtsystem_propagate(
    handle: *const EmpyreanBuiltSystem,
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    times_ptr: *const f64,
    num_times: usize,
    config: *const EmpyreanPropagationConfig,
    result_out: *mut EmpyreanPropagationResult,
) -> i32 {
    unsafe {
        lib().empyrean_builtsystem_propagate(
            handle, ctx, orbits_ptr, num_orbits, times_ptr, num_times, config, result_out,
        )
    }
}
/** Generate predicted ephemeris for `orbits` and `observers` through the
pre-built handle.

Signature parallels the one-shot
[`empyrean_generate_ephemeris`](crate::ephemeris::empyrean_generate_ephemeris)
but takes `(handle, ctx, ...)`. Runs the same identity guard as
[`empyrean_builtsystem_propagate`] before dispatch; on pass the result is
bit-identical to the one-shot. Note the C-ABI ephemeris config carries no
divisor knob, so a handle frozen at a non-default divisor is rejected here
with [`EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_DIVISOR`] rather than served under
the wrong dynamics — build the handle with the default divisor for
ephemeris reuse.

On success populates `result_out`; free it with
[`empyrean_ephemeris_result_free`](crate::ephemeris::empyrean_ephemeris_result_free).*/
#[inline]
pub unsafe fn empyrean_builtsystem_generate_ephemeris(
    handle: *const EmpyreanBuiltSystem,
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    observers_ptr: *const EmpyreanObserver,
    num_observers: usize,
    config: *const EmpyreanEphemerisConfig,
    result_out: *mut EmpyreanEphemerisResult,
) -> i32 {
    unsafe {
        lib().empyrean_builtsystem_generate_ephemeris(
            handle,
            ctx,
            orbits_ptr,
            num_orbits,
            observers_ptr,
            num_observers,
            config,
            result_out,
        )
    }
}
/** Run the full orbit-determination pipeline over every object in
`observations`, reusing the pre-built handle for every fit in the
batch.

Signature parallels the one-shot
[`empyrean_determine`](crate::od::empyrean_determine) but takes
`(handle, ctx, ...)`, exactly as
[`empyrean_builtsystem_propagate`] relates to `empyrean_propagate`.
Every other argument keeps its name, position and meaning. Results are
bit-identical to the one-shot with the same config and a matching
handle — the handle only changes *when* the force model is assembled.

# The handle must be an OD handle

Build it with [`empyrean_builtsystem_new_for_od`]. OD does not
integrate in ICRF, so a handle built the way the propagation examples
build theirs is rejected here with
[`EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME`].

The identity guard runs before any fitting and **refuses** on a
mismatch — data identity ([`EMPYREAN_BUILTSYSTEM_DATA_MISMATCH`]),
frozen key ([`EMPYREAN_BUILTSYSTEM_KEY_MISMATCH_FRAME`] /
`_FORCE_MODEL` / `_DIVISOR`), staleness
([`EMPYREAN_BUILTSYSTEM_STALE`]). It never degrades to a per-solve
assembly: a caller who asked to amortize and silently is not is
precisely the failure this entry point exists to make impossible.

# `auto_force_model` is refused, not tolerated

`EmpyreanODConfig::auto_force_model` lets the fit re-pick its own
force-model tier part-way through, which no frozen handle can follow:
the engine would drop this handle mid-fit and reassemble per solve
while this call still returned `0`. Passing it with a handle therefore
fails with [`EMPYREAN_BUILTSYSTEM_CONFIG_MUTATES_KEY`] and a message
naming the field. Set `auto_force_model = 0` and freeze the handle at
the tier you want, or drop the handle and call
[`empyrean_determine`](crate::od::empyrean_determine), which is free to
choose the tier.

# Return codes

The one-shot's codes, plus the guard codes above. On `0` and
[`EMPYREAN_DETERMINE_NONE_DELIVERED`](crate::od::EMPYREAN_DETERMINE_NONE_DELIVERED)
release `results_out` with
[`empyrean_determine_results_free`](crate::od::empyrean_determine_results_free).*/
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn empyrean_builtsystem_determine(
    handle: *const EmpyreanBuiltSystem,
    ctx: *const EmpyreanContext,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    radar: *const EmpyreanRadarObservation,
    num_radar: usize,
    initial_orbits: *const EmpyreanOrbit,
    num_initial_orbits: usize,
    config: *const EmpyreanODConfig,
    results_out: *mut EmpyreanDetermineResults,
) -> i32 {
    unsafe {
        lib().empyrean_builtsystem_determine(
            handle,
            ctx,
            observations,
            num_observations,
            radar,
            num_radar,
            initial_orbits,
            num_initial_orbits,
            config,
            results_out,
        )
    }
}
/** Evaluate residuals for a single orbit against observations, reusing
the pre-built handle.

Signature parallels the one-shot
[`empyrean_evaluate`](crate::od::empyrean_evaluate) with `handle`
prepended. Same identity guard, same code space, the same OD-handle
requirement and the same `auto_force_model` refusal as
[`empyrean_builtsystem_determine`]. Free `result_out` with
[`empyrean_evaluate_result_free`](crate::od::empyrean_evaluate_result_free).*/
#[inline]
pub unsafe fn empyrean_builtsystem_evaluate(
    handle: *const EmpyreanBuiltSystem,
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
    result_out: *mut EmpyreanEvaluateResult,
) -> i32 {
    unsafe {
        lib().empyrean_builtsystem_evaluate(
            handle,
            ctx,
            orbit,
            observations,
            num_observations,
            config,
            result_out,
        )
    }
}
/** Refine a single orbit estimate with new observations using a Bayesian
prior, reusing the pre-built handle.

Signature parallels the one-shot
[`empyrean_refine`](crate::od::empyrean_refine) with `handle`
prepended. Same identity guard, same code space, the same OD-handle
requirement and the same `auto_force_model` refusal as
[`empyrean_builtsystem_determine`]. This is
the measure-and-extend loop's entry point: build one handle, refine
many times through it. Free `result_out` with
[`empyrean_od_result_free`](crate::od::empyrean_od_result_free).*/
#[inline]
pub unsafe fn empyrean_builtsystem_refine(
    handle: *const EmpyreanBuiltSystem,
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    unsafe {
        lib().empyrean_builtsystem_refine(
            handle,
            ctx,
            orbit,
            observations,
            num_observations,
            config,
            result_out,
        )
    }
}
/** Populate `out` with a full reproducibility summary of the handle's frozen
force model and its captured kernel manifest.

Every field is populated from the system description and the manifest
snapshot — no field is left defaulted. Returns
[`EMPYREAN_BUILTSYSTEM_OK`] on success. The caller owns the heap arrays
inside `out` and must release them with
[`empyrean_builtsystem_description_free`].*/
#[inline]
pub unsafe fn empyrean_builtsystem_describe(
    handle: *const EmpyreanBuiltSystem,
    out: *mut EmpyreanSystemDescription,
) -> i32 {
    unsafe { lib().empyrean_builtsystem_describe(handle, out) }
}
/** Free the heap arrays inside a description populated by
[`empyrean_builtsystem_describe`] (the perturber-id array and the kernel
records with their C strings). After this returns `desc` is
zero-initialized — safe to drop on the caller's stack. Passing null is a
no-op.*/
#[inline]
pub unsafe fn empyrean_builtsystem_description_free(desc: *mut EmpyreanSystemDescription) {
    unsafe { lib().empyrean_builtsystem_description_free(desc) }
}
/** Generate predicted ephemeris for orbits and observers.

Returns 0 on success, negative error code on failure.
On success, `result_out` is populated with ephemeris entries:
`num_orbits * num_observers` rows, orbit-major, and within each orbit
in **observer-input order** (sensitivity rows follow the same order).
Each observer carries its own epoch, so positional pairing against
the input observers is safe within an orbit block.
The caller must free the result with `empyrean_ephemeris_result_free()`.*/
#[inline]
pub unsafe fn empyrean_generate_ephemeris(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    observers_ptr: *const EmpyreanObserver,
    num_observers: usize,
    config: *const EmpyreanEphemerisConfig,
    result_out: *mut EmpyreanEphemerisResult,
) -> i32 {
    unsafe {
        lib().empyrean_generate_ephemeris(
            ctx,
            orbits_ptr,
            num_orbits,
            observers_ptr,
            num_observers,
            config,
            result_out,
        )
    }
}
/** Free an ephemeris result previously returned by `empyrean_generate_ephemeris()`.*/
#[inline]
pub unsafe fn empyrean_ephemeris_result_free(result: *mut EmpyreanEphemerisResult) {
    unsafe { lib().empyrean_ephemeris_result_free(result) }
}
/** Run [`empyrean_core::impact::compute_impact_probabilities`] over a
caller-supplied set of [`UncertaintyMethod`] variants and return
the flattened IP records tagged by method.

Caller is responsible for freeing the result via
[`empyrean_compute_impact_probabilities_result_free`].

# Returns
`0` on success, negative on failure (see [`crate::set_last_error`]).

# Safety
All non-null pointer arguments must point to valid arrays of the
indicated length for the duration of the call. The output struct
is allocated by the caller; pointer fields inside it are allocated
here and owned by the result.*/
#[inline]
pub unsafe fn empyrean_compute_impact_probabilities(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    end_mjd_tdb: f64,
    methods_ptr: *const EmpyreanUncertaintyMethod,
    num_methods: usize,
    body_filter_naif: *const i32,
    num_body_filter: usize,
    result_out: *mut EmpyreanImpactProbabilitiesResult,
) -> i32 {
    unsafe {
        lib().empyrean_compute_impact_probabilities(
            ctx,
            orbits_ptr,
            num_orbits,
            end_mjd_tdb,
            methods_ptr,
            num_methods,
            body_filter_naif,
            num_body_filter,
            result_out,
        )
    }
}
/** Free the records array allocated by
[`empyrean_compute_impact_probabilities`]. After calling, the
struct's `records` is reset to null and `num_records` to 0.*/
#[inline]
pub unsafe fn empyrean_compute_impact_probabilities_result_free(
    result: *mut EmpyreanImpactProbabilitiesResult,
) {
    unsafe { lib().empyrean_compute_impact_probabilities_result_free(result) }
}
/** Run [`empyrean_core::impact::compute_b_planes`] over a caller-
supplied set of [`UncertaintyMethod`] variants and return the
flattened B-plane records tagged by method.

Caller is responsible for freeing the result via
[`empyrean_compute_b_planes_result_free`].

# Safety
Same contract as
[`empyrean_compute_impact_probabilities`].*/
#[inline]
pub unsafe fn empyrean_compute_b_planes(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    end_mjd_tdb: f64,
    methods_ptr: *const EmpyreanUncertaintyMethod,
    num_methods: usize,
    body_filter_naif: *const i32,
    num_body_filter: usize,
    result_out: *mut EmpyreanBPlanesResult,
) -> i32 {
    unsafe {
        lib().empyrean_compute_b_planes(
            ctx,
            orbits_ptr,
            num_orbits,
            end_mjd_tdb,
            methods_ptr,
            num_methods,
            body_filter_naif,
            num_body_filter,
            result_out,
        )
    }
}
/** Free the records array allocated by [`empyrean_compute_b_planes`].*/
#[inline]
pub unsafe fn empyrean_compute_b_planes_result_free(result: *mut EmpyreanBPlanesResult) {
    unsafe { lib().empyrean_compute_b_planes_result_free(result) }
}
/** Free a batch previously returned by an `empyrean_orbits_read_*`
function. Passing null is a no-op.*/
#[inline]
pub unsafe fn empyrean_orbits_batch_free(batch: *mut EmpyreanOrbitBatch) {
    unsafe { lib().empyrean_orbits_batch_free(batch) }
}
/** Read an orbits parquet file. Caller frees the result with
[`empyrean_orbits_batch_free`].*/
#[inline]
pub unsafe fn empyrean_orbits_read_parquet(
    path: *const ::std::os::raw::c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    unsafe { lib().empyrean_orbits_read_parquet(path, out) }
}
/** Write an orbit batch to a parquet file.*/
#[inline]
pub unsafe fn empyrean_orbits_write_parquet(
    path: *const ::std::os::raw::c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    unsafe { lib().empyrean_orbits_write_parquet(path, batch) }
}
/** Read an orbits JSON file (array of orbit-row objects). Caller frees
with [`empyrean_orbits_batch_free`].*/
#[inline]
pub unsafe fn empyrean_orbits_read_json(
    path: *const ::std::os::raw::c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    unsafe { lib().empyrean_orbits_read_json(path, out) }
}
/**Write an orbit batch to JSON.

This is **not** the engine's orbit schema — it is this crate's own
flat row shape (`OrbitRow`), and it is the least capable of the three
orbit formats. It carries the state, the 6×6 and the Marsden
coefficients with their g(r) exponents, and nothing else.

A batch carrying a state↔Marsden border or a wide cross-covariance
carrier is **refused by name**, pointing at parquet — the format is
unable to represent the joint, and writing the row short would
produce a file that reads back as a block-diagonal covariance with no
signal that anything was lost.

Fields this format drops without refusing, because they predate the
joint surface and a refusal would break callers who write them today:
the non-grav DT and its prior variance, the Marsden 3×3, the SRP slot
and the photometric block. Round-trip through parquet if you need
them.*/
#[inline]
pub unsafe fn empyrean_orbits_write_json(
    path: *const ::std::os::raw::c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    unsafe { lib().empyrean_orbits_write_json(path, batch) }
}
/** Read an orbits CSV file.

Uses the engine's own CSV reader, so the file is the same schema the
parquet path round-trips — covariance included. The
`_with_non_grav` reader is the one used: the plain reader drops the
Marsden A1/A2/A3 block, and an orbit that loses its non-gravitational
parameters on a round trip is silently a different orbit.*/
#[inline]
pub unsafe fn empyrean_orbits_read_csv(
    path: *const ::std::os::raw::c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    unsafe { lib().empyrean_orbits_read_csv(path, out) }
}
/** Write an orbit batch to CSV.

Routed through the engine's writer — the same `Orbits<AU>` the
parquet path writes — so CSV carries the full column set (state,
covariance, non-grav including `dt` / `dt_variance`, photometry, SRP)
rather than a flattened projection of it. A batch carrying a wide
cross-covariance the row schema cannot express is refused before the
file is created rather than written short.*/
#[inline]
pub unsafe fn empyrean_orbits_write_csv(
    path: *const ::std::os::raw::c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    unsafe { lib().empyrean_orbits_write_csv(path, batch) }
}
/** Write ephemeris entries to parquet using the villeneuve schema.*/
#[inline]
pub unsafe fn empyrean_ephemeris_write_parquet(
    path: *const ::std::os::raw::c_char,
    entries_ptr: *const EmpyreanEphemerisEntry,
    num_entries: usize,
) -> i32 {
    unsafe { lib().empyrean_ephemeris_write_parquet(path, entries_ptr, num_entries) }
}
/** Write ephemeris entries to JSON.*/
#[inline]
pub unsafe fn empyrean_ephemeris_write_json(
    path: *const ::std::os::raw::c_char,
    entries_ptr: *const EmpyreanEphemerisEntry,
    num_entries: usize,
) -> i32 {
    unsafe { lib().empyrean_ephemeris_write_json(path, entries_ptr, num_entries) }
}
/** Write ephemeris entries to CSV.*/
#[inline]
pub unsafe fn empyrean_ephemeris_write_csv(
    path: *const ::std::os::raw::c_char,
    entries_ptr: *const EmpyreanEphemerisEntry,
    num_entries: usize,
) -> i32 {
    unsafe { lib().empyrean_ephemeris_write_csv(path, entries_ptr, num_entries) }
}
/** Write events to parquet.*/
#[inline]
pub unsafe fn empyrean_events_write_parquet(
    path: *const ::std::os::raw::c_char,
    events_ptr: *const EmpyreanEvent,
    num_events: usize,
) -> i32 {
    unsafe { lib().empyrean_events_write_parquet(path, events_ptr, num_events) }
}
/** Write events to JSON.*/
#[inline]
pub unsafe fn empyrean_events_write_json(
    path: *const ::std::os::raw::c_char,
    events_ptr: *const EmpyreanEvent,
    num_events: usize,
) -> i32 {
    unsafe { lib().empyrean_events_write_json(path, events_ptr, num_events) }
}
/** Write events to CSV.*/
#[inline]
pub unsafe fn empyrean_events_write_csv(
    path: *const ::std::os::raw::c_char,
    events_ptr: *const EmpyreanEvent,
    num_events: usize,
) -> i32 {
    unsafe { lib().empyrean_events_write_csv(path, events_ptr, num_events) }
}
/** Write the per-object fit summary to parquet.*/
#[inline]
pub unsafe fn empyrean_fit_summary_write_parquet(
    path: *const ::std::os::raw::c_char,
    summaries_ptr: *const EmpyreanFitSummary,
    num_summaries: usize,
) -> i32 {
    unsafe { lib().empyrean_fit_summary_write_parquet(path, summaries_ptr, num_summaries) }
}
/** Write the per-object fit summary to JSON.*/
#[inline]
pub unsafe fn empyrean_fit_summary_write_json(
    path: *const ::std::os::raw::c_char,
    summaries_ptr: *const EmpyreanFitSummary,
    num_summaries: usize,
) -> i32 {
    unsafe { lib().empyrean_fit_summary_write_json(path, summaries_ptr, num_summaries) }
}
/** Write the per-object fit summary to CSV.*/
#[inline]
pub unsafe fn empyrean_fit_summary_write_csv(
    path: *const ::std::os::raw::c_char,
    summaries_ptr: *const EmpyreanFitSummary,
    num_summaries: usize,
) -> i32 {
    unsafe { lib().empyrean_fit_summary_write_csv(path, summaries_ptr, num_summaries) }
}
/** Write OD residuals to parquet.*/
#[inline]
pub unsafe fn empyrean_residuals_write_parquet(
    path: *const ::std::os::raw::c_char,
    obs_ptr: *const EmpyreanObservationResult,
    num_obs: usize,
) -> i32 {
    unsafe { lib().empyrean_residuals_write_parquet(path, obs_ptr, num_obs) }
}
/** Write OD residuals to JSON.*/
#[inline]
pub unsafe fn empyrean_residuals_write_json(
    path: *const ::std::os::raw::c_char,
    obs_ptr: *const EmpyreanObservationResult,
    num_obs: usize,
) -> i32 {
    unsafe { lib().empyrean_residuals_write_json(path, obs_ptr, num_obs) }
}
/** Write OD residuals to CSV.*/
#[inline]
pub unsafe fn empyrean_residuals_write_csv(
    path: *const ::std::os::raw::c_char,
    obs_ptr: *const EmpyreanObservationResult,
    num_obs: usize,
) -> i32 {
    unsafe { lib().empyrean_residuals_write_csv(path, obs_ptr, num_obs) }
}
/** Find the dominant eigenvalue and eigenvector of a 6x6 symmetric matrix.

Returns 0 on success. `eigenvalue_out` receives the eigenvalue,
`eigenvector_out` receives the 6-element eigenvector.*/
#[inline]
pub unsafe fn empyrean_eigenvector_max_6x6(
    matrix: *const [[f64; 6usize]; 6usize],
    eigenvalue_out: *mut f64,
    eigenvector_out: *mut [f64; 6usize],
) -> i32 {
    unsafe { lib().empyrean_eigenvector_max_6x6(matrix, eigenvalue_out, eigenvector_out) }
}
/** Split a 6D Gaussian into K weighted components along the dominant
eigenvector of the covariance.

`weights_out`, `means_out`, `covariances_out` must point to arrays
of size K, K×6, and K×6×6 respectively. Returns 0 on success.*/
#[inline]
pub unsafe fn empyrean_split_gaussian(
    mean: *const [f64; 6usize],
    covariance: *const [[f64; 6usize]; 6usize],
    k: usize,
    weights_out: *mut f64,
    means_out: *mut [f64; 6usize],
    covariances_out: *mut [[f64; 6usize]; 6usize],
) -> i32 {
    unsafe {
        lib().empyrean_split_gaussian(mean, covariance, k, weights_out, means_out, covariances_out)
    }
}
/** Compute observer states for given observatory codes and epochs, in a
caller-chosen `(frame, origin)` basis.

The result is the Cartesian product `obs_codes × epochs`, **code-major**:
all epochs for `obs_codes[0]`, then all epochs for `obs_codes[1]`, so
`observers[i * num_epochs + j]` is `(obs_codes[i], epochs[j])`.

# Choosing a basis

`frame = 0` (ICRF) with `origin = 0` (solar-system barycenter) is the
**construction basis** — the one every consumer of an observer state
requires, and the one to pass when the observers are headed for
`empyrean_generate_ephemeris`,
`empyrean_builtsystem_generate_ephemeris`, or orbit determination.
Requesting it takes no transform at all: the observers come back
exactly as constructed, bit for bit. Any other basis rotates and/or
translates them, which is what a consumer plotting observer geometry
wants (e.g. heliocentric ecliptic site positions via
`frame = 1`, `origin = 10`).

Each returned row carries the basis it is expressed in
([`EmpyreanObserver::frame`] / [`EmpyreanObserver::origin`]), so a
row is never ambiguous about which basis produced it.

# Returns

`0` on success; `result_out` is populated and the caller must free it
with `empyrean_observer_result_free()`.

Every rejection aborts the whole call — there is no partial output and
no silently untransformed entry. The failure codes are split by
**remedy**, not flattened, because a channel that collapses them tells
an operator to fix their input when the fix is on disk:

- `-1` — retry with different arguments: an observatory code is not in
  the registry; `frame` is one an observer cannot be rotated into
  (anything outside ICRF ↔ EclipticJ2000); `origin` names an MPC site
  rather than an SPK body; a null pointer or non-UTF-8 code.
- `-2` — load or fetch data, the arguments are fine: the code names a
  space telescope whose SPK is not loaded; `origin` has no SPK
  coverage at a requested epoch; an epoch falls outside the loaded
  BPC's window, or this context carries no BPC / no observatory
  registry at all.
- `-5` — allocation failure. `-99` — a panic was caught at the boundary.

Call `empyrean_last_error()` for the message in every failing case.*/
#[inline]
pub unsafe fn empyrean_get_observers(
    ctx: *const EmpyreanContext,
    obs_codes: *const *const ::std::os::raw::c_char,
    num_codes: usize,
    epochs_mjd_tdb: *const f64,
    num_epochs: usize,
    frame: i32,
    origin: i32,
    result_out: *mut EmpyreanObserverResult,
) -> i32 {
    unsafe {
        lib().empyrean_get_observers(
            ctx,
            obs_codes,
            num_codes,
            epochs_mjd_tdb,
            num_epochs,
            frame,
            origin,
            result_out,
        )
    }
}
/** Free an observer result previously returned by `empyrean_get_observers()`.

Passing a zeroed/null result is a no-op.*/
#[inline]
pub unsafe fn empyrean_observer_result_free(result: *mut EmpyreanObserverResult) {
    unsafe { lib().empyrean_observer_result_free(result) }
}
/** Runtime accessor for [`EMPYREAN_ABI_VERSION`] — lets a dynamically
linked consumer confirm the loaded library's frozen-shape contract
matches what it compiled against.*/
#[inline]
pub unsafe fn empyrean_abi_version() -> u32 {
    unsafe { lib().empyrean_abi_version() }
}
/** Read ADES PSV / MPC80 data from a string and pack into the C array.

`path_or_content` is a null-terminated UTF-8 string with the ADES
content directly (not a file path).*/
#[inline]
pub unsafe fn empyrean_read_ades(
    content: *const ::std::os::raw::c_char,
    observations_out: *mut *mut EmpyreanObservation,
    num_observations_out: *mut usize,
    radar_out: *mut *mut EmpyreanRadarObservation,
    num_radar_out: *mut usize,
) -> i32 {
    unsafe {
        lib().empyrean_read_ades(
            content,
            observations_out,
            num_observations_out,
            radar_out,
            num_radar_out,
        )
    }
}
/** Free an observation array previously returned by `empyrean_read_ades()`.
Copy a caller-owned array of [`EmpyreanObservation`] into a fresh
allocation that matches the layout produced by
[`empyrean_read_ades`].

The strings on the input observations (`perm_id` / `prov_id` /
`obs_time`) are duplicated into freshly-allocated `CString`s so the
returned array owns its own memory independent of the input.

On success populates `*out_ptr` with the new array and `*out_num`
with its length, both freeable with [`empyrean_observations_free`].

Returns 0 on success; negative error code on failure.*/
#[inline]
pub unsafe fn empyrean_observations_from_array(
    input: *const EmpyreanObservation,
    num: usize,
    out_ptr: *mut *mut EmpyreanObservation,
    out_num: *mut usize,
) -> i32 {
    unsafe { lib().empyrean_observations_from_array(input, num, out_ptr, out_num) }
}
#[inline]
pub unsafe fn empyrean_observations_free(observations: *mut EmpyreanObservation, num: usize) {
    unsafe { lib().empyrean_observations_free(observations, num) }
}
/** Copy a caller-owned array of [`EmpyreanRadarObservation`] into a fresh
allocation matching the layout produced by [`empyrean_read_ades`].

The nullable `*mut c_char` fields (`perm_id` / `prov_id` / `trk_sub` /
`obs_time` / `remarks`) are duplicated into freshly-allocated
`CString`s so the returned array owns its own memory independent of the
input. All scalar fields (including the ADES-native delay/Doppler
values) are copied verbatim — no unit conversion, nothing zeroed.

On success populates `*out_ptr` with the new array and `*out_num` with
its length, both freeable with [`empyrean_radar_observations_free`].

Returns 0 on success; negative error code on failure.*/
#[inline]
pub unsafe fn empyrean_radar_observations_from_array(
    input: *const EmpyreanRadarObservation,
    num: usize,
    out_ptr: *mut *mut EmpyreanRadarObservation,
    out_num: *mut usize,
) -> i32 {
    unsafe { lib().empyrean_radar_observations_from_array(input, num, out_ptr, out_num) }
}
/** Free a radar observation array previously returned by
[`empyrean_read_ades`] or [`empyrean_radar_observations_from_array`].*/
#[inline]
pub unsafe fn empyrean_radar_observations_free(
    observations: *mut EmpyreanRadarObservation,
    num: usize,
) {
    unsafe { lib().empyrean_radar_observations_free(observations, num) }
}
/** Run the full orbit determination pipeline over every object in
`observations`.

The observations are grouped by ADES object identifier (permID /
provID / trkSub) and each group is fitted independently, so one call
determines a whole batch. `results_out` receives one
[`EmpyreanODObjectResult`] per group, ordered by `object_id`.

When `num_initial_orbits > 0`, the supplied orbits are used as DC
seeds (one per ADES object_id encountered in `observations`,
matched by orbit index). Pass `null, 0` to let the IOD pipeline
produce its own seeds. A seed that matches no group is reported in
[`EmpyreanDetermineResults::unmatched_orbit_ids`], never dropped.

# Return codes

- `0` — the batch ran and **at least one** object delivered a fit.
  Individual failures do not abort the batch; check each slot's
  `delivered` flag. `results_out` is populated.
- [`EMPYREAN_DETERMINE_NONE_DELIVERED`] (`-4`) — the batch ran but
  every object failed. `results_out` IS populated with the per-object
  errors and must still be freed.
- `-1` — null pointer or malformed input; nothing is written.
- `-3` — a batch-level failure (an unparseable weighting config, an
  observation row with no identifier at all) aborted the run before
  any object was fitted; nothing is written.

A single-object input is not a special case: it produces a
one-row table.

# Ownership

On `0` and `-4`, release `results_out` with
[`empyrean_determine_results_free`]. On `-1` / `-3` there is nothing
to free.*/
#[inline]
pub unsafe fn empyrean_determine(
    ctx: *const EmpyreanContext,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    radar: *const EmpyreanRadarObservation,
    num_radar: usize,
    initial_orbits: *const EmpyreanOrbit,
    num_initial_orbits: usize,
    config: *const EmpyreanODConfig,
    results_out: *mut EmpyreanDetermineResults,
) -> i32 {
    unsafe {
        lib().empyrean_determine(
            ctx,
            observations,
            num_observations,
            radar,
            num_radar,
            initial_orbits,
            num_initial_orbits,
            config,
            results_out,
        )
    }
}
/** Free a batch result table previously written by
`empyrean_determine()`.

Releases every per-object slot (including the fits inside the
delivered ones), the per-object identifier / error strings, and the
unmatched-seed list. Safe to call on a table returned with
[`EMPYREAN_DETERMINE_NONE_DELIVERED`], and idempotent — the table is
left empty.*/
#[inline]
pub unsafe fn empyrean_determine_results_free(results: *mut EmpyreanDetermineResults) {
    unsafe { lib().empyrean_determine_results_free(results) }
}
/** Free an OD result previously returned by `empyrean_refine()`.

Batch `empyrean_determine()` results are released with
[`empyrean_determine_results_free`] instead.*/
#[inline]
pub unsafe fn empyrean_od_result_free(result: *mut EmpyreanODResult) {
    unsafe { lib().empyrean_od_result_free(result) }
}
/**Evaluate residuals for a single orbit against observations.

# A supplied joint covariance changes nothing here

Evaluation measures how well a FIXED orbit predicts observations; it
forms no prior and performs no estimation, so an orbit carrying a
state↔Marsden border or a wide carrier scores exactly as the same
orbit without one. Nothing is dropped — there is simply nothing for
the joint to affect, and this result type carries no orbit to echo
one back on. The nine other orbit-reading entry points consume it.*/
#[inline]
pub unsafe fn empyrean_evaluate(
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
    result_out: *mut EmpyreanEvaluateResult,
) -> i32 {
    unsafe {
        lib().empyrean_evaluate(
            ctx,
            orbit,
            observations,
            num_observations,
            config,
            result_out,
        )
    }
}
/** Free an evaluate result previously returned by `empyrean_evaluate()`.*/
#[inline]
pub unsafe fn empyrean_evaluate_result_free(result: *mut EmpyreanEvaluateResult) {
    unsafe { lib().empyrean_evaluate_result_free(result) }
}
/** Refine a single orbit estimate with new observations using a
Bayesian prior.*/
#[inline]
pub unsafe fn empyrean_refine(
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    unsafe {
        lib().empyrean_refine(
            ctx,
            orbit,
            observations,
            num_observations,
            config,
            result_out,
        )
    }
}
/** Evaluate an observation plan: how much each candidate observation would
tighten the prior orbit covariance.

`orbit` must carry a 6×6 Cartesian covariance (e.g. a `determine` result).
`planned` is an array of `num_planned` candidate observations. `orbit_id`
may be null (defaults to `"orbit_0"`). On success populates `*result_out`
(caller-allocated); free with [`empyrean_plan_result_free`].

Returns 0 on success, -1 for a null/invalid argument, -3 if planning fails
(missing/singular prior covariance, an infeasible or invalid candidate, an
ephemeris-generation error), -99 on an internal panic. The error message is
retrievable via `empyrean_last_error()`.*/
#[inline]
pub unsafe fn empyrean_evaluate_plan(
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    orbit_id: *const ::std::os::raw::c_char,
    planned: *const EmpyreanPlannedObservation,
    num_planned: usize,
    config: *const EmpyreanPlanningConfig,
    result_out: *mut EmpyreanPlanResult,
) -> i32 {
    unsafe {
        lib().empyrean_evaluate_plan(
            ctx,
            orbit,
            orbit_id,
            planned,
            num_planned,
            config,
            result_out,
        )
    }
}
/** Free the heap allocations inside a plan result populated by
[`empyrean_evaluate_plan`]. Does not free the caller-allocated struct.*/
#[inline]
pub unsafe fn empyrean_plan_result_free(result: *mut EmpyreanPlanResult) {
    unsafe { lib().empyrean_plan_result_free(result) }
}
/** Propagate orbits to the requested target times.

Returns 0 on success, negative error code on failure.
On success, `result_out` is populated with the propagated states.
The caller must free the result with `empyrean_propagation_result_free()`.

States are flat in orbit-major order; within each orbit, rows are in
**ascending epoch order, always** (engine guarantee since villeneuve
v1.18.0), regardless of request order. Positional pairing against an
ascending, duplicate-free request grid is exact; for any other
request shape, join on `epoch_mjd_tdb`.*/
#[inline]
pub unsafe fn empyrean_propagate(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    times_ptr: *const f64,
    num_times: usize,
    config: *const EmpyreanPropagationConfig,
    result_out: *mut EmpyreanPropagationResult,
) -> i32 {
    unsafe {
        lib().empyrean_propagate(
            ctx, orbits_ptr, num_orbits, times_ptr, num_times, config, result_out,
        )
    }
}
/** Free a propagation result previously returned by `empyrean_propagate()`.*/
#[inline]
pub unsafe fn empyrean_propagation_result_free(result: *mut EmpyreanPropagationResult) {
    unsafe { lib().empyrean_propagation_result_free(result) }
}
/** Resolved-kind tagged covariance at every output epoch for one orbit,
Cartesian basis. On success `out_series` owns the array; free with
[`empyrean_tagged_covariance_series_free`]. On error `out_series` is
left null and the detail is on `empyrean_last_error()`.

# Safety
`result` must be a valid pointer returned by `empyrean_propagate`;
`out_series` must be a valid pointer to write the result into.*/
#[inline]
pub unsafe fn empyrean_propagation_covariance_series_cartesian(
    result: *const EmpyreanPropagationResult,
    orbit_index: usize,
    out_series: *mut *mut EmpyreanTaggedCovarianceSeries,
) -> i32 {
    unsafe {
        lib().empyrean_propagation_covariance_series_cartesian(result, orbit_index, out_series)
    }
}
/** Resolved-kind tagged covariance at a single `(orbit_index,
epoch_index)`, Cartesian basis (the gm-free point query). `out` is
written on success.

# Safety
`result` and `out` must be valid pointers; `result` from `empyrean_propagate`.*/
#[inline]
pub unsafe fn empyrean_propagation_covariance_at_cartesian(
    result: *const EmpyreanPropagationResult,
    orbit_index: usize,
    epoch_index: usize,
    out: *mut EmpyreanTaggedCovariance,
) -> i32 {
    unsafe {
        lib().empyrean_propagation_covariance_at_cartesian(result, orbit_index, epoch_index, out)
    }
}
/** Free a series returned by
[`empyrean_propagation_covariance_series_cartesian`].

# Safety
`series` must be null or a pointer returned by that accessor, freed once.*/
#[inline]
pub unsafe fn empyrean_tagged_covariance_series_free(series: *mut EmpyreanTaggedCovarianceSeries) {
    unsafe { lib().empyrean_tagged_covariance_series_free(series) }
}
/** Query the JPL Small-Body Database for one or more orbits.

`object_ids` is an array of `num_object_ids` null-terminated UTF-8
designations / names / SPK IDs (e.g. `"apophis"`, `"99942"`,
`"2024 YR4"`, `"67P"`). `cache_dir` may be null to skip caching, or
a directory path where SBDB JSON responses are cached on disk.

On success the populated [`EmpyreanOrbitBatch`] must be released with
[`empyrean_orbits_batch_free`](crate::io::empyrean_orbits_batch_free).*/
#[inline]
pub unsafe fn empyrean_query_sbdb(
    object_ids: *const *const ::std::os::raw::c_char,
    num_object_ids: usize,
    cache_dir: *const ::std::os::raw::c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    unsafe { lib().empyrean_query_sbdb(object_ids, num_object_ids, cache_dir, out) }
}
/** Query JPL Horizons for predicted ephemeris records.

`object_ids` is an array of `num_object_ids` null-terminated UTF-8
designations / names / SPK IDs. `obs_code` is the MPC observatory
code as a null-terminated string. `times_mjd_tdb` carries
`num_times` epochs in MJD TDB.

On success populates an [`EmpyreanEphemerisResult`] with one entry
per `(object_id × epoch)`. Free with
[`empyrean_ephemeris_result_free`](crate::ephemeris::empyrean_ephemeris_result_free).

All angular values are converted to **degrees** at the FFI boundary
(Horizons natively returns radians).*/
#[inline]
pub unsafe fn empyrean_query_horizons(
    object_ids: *const *const ::std::os::raw::c_char,
    num_object_ids: usize,
    obs_code: *const ::std::os::raw::c_char,
    times_mjd_tdb: *const f64,
    num_times: usize,
    cache_dir: *const ::std::os::raw::c_char,
    out: *mut EmpyreanEphemerisResult,
) -> i32 {
    unsafe {
        lib().empyrean_query_horizons(
            object_ids,
            num_object_ids,
            obs_code,
            times_mjd_tdb,
            num_times,
            cache_dir,
            out,
        )
    }
}
/** Query JPL Horizons for a Cartesian state vector at a single epoch.

`command` is the Horizons COMMAND string as a null-terminated UTF-8
string (e.g. `"99942;"`, `"DES=C/2019 Q4;"`). `epoch_mjd_tdb` is
the epoch in MJD TDB. `cache_dir` may be null to skip caching, or
a directory path where Horizons JSON responses are cached on disk.

On success writes the position (AU) to `out_pos` (length 3) and the
velocity (AU/day) to `out_vel` (length 3) — both solar-system
barycenter (SSB) centered, ICRF.*/
#[inline]
pub unsafe fn empyrean_query_horizons_vectors(
    command: *const ::std::os::raw::c_char,
    epoch_mjd_tdb: f64,
    cache_dir: *const ::std::os::raw::c_char,
    out_pos: *mut f64,
    out_vel: *mut f64,
) -> i32 {
    unsafe {
        lib().empyrean_query_horizons_vectors(command, epoch_mjd_tdb, cache_dir, out_pos, out_vel)
    }
}
/** Query the MPC observations API for ADES records of one or more
designations.

`designations` is an array of `num_designations` null-terminated
UTF-8 designations (e.g. `"99942"`, `"2024 YR4"`, `"67P"`). The
MPC API returns ADES_DF JSON; this function parses each row into
the C-ABI [`EmpyreanObservation`] struct, filling the full ADES
schema (perm_id / prov_id / trk_sub / mode / sys / ctr / pos1-3 /
rms_corr / mag / rms_mag / band / ast_cat / phot_cat / phot_ap /
log_snr / seeing / exp / rms_fit / n_stars / notes / remarks) when
present in the JSON.

On success `*out_ptr` carries a heap-allocated array of length
`*out_num`. Free with
[`empyrean_observations_free`](crate::od::empyrean_observations_free).*/
#[inline]
pub unsafe fn empyrean_query_observations(
    designations: *const *const ::std::os::raw::c_char,
    num_designations: usize,
    cache_dir: *const ::std::os::raw::c_char,
    out_ptr: *mut *mut EmpyreanObservation,
    out_num: *mut usize,
) -> i32 {
    unsafe {
        lib().empyrean_query_observations(
            designations,
            num_designations,
            cache_dir,
            out_ptr,
            out_num,
        )
    }
}
/** Query the JPL `sb_radar` API for delay/Doppler radar astrometry of one
or more designations.

`designations` is an array of `num_designations` null-terminated UTF-8
designations (e.g. `"99942"`, `"2024 YR4"`). Asteroid radar astrometry
is a JPL SSD product — it is **not** served by the MPC observations API
(`empyrean_query_observations` returns only optical / occultation
records), so radar ships as its own live-query entry point. JPL
`sb_radar` JSON records are converted to ADES-native scott
`RadarObservation`s and packed into the C-ABI
[`EmpyreanRadarObservation`] struct (the same layout
[`empyrean_read_ades`](crate::od::empyrean_read_ades) emits): the delay
value is in seconds, its σ in microseconds, Doppler in Hz, frequency in
MHz, and the `com` flag is a tri-state i8. `cache_dir` may be null to
skip caching, or a directory path where `sb_radar` JSON responses are
cached on disk.

An object with no radar astrometry contributes no records (it is not an
error). A JPL record that fails to parse (missing required field, or an
unrecognised DSN station code) is rejected loudly rather than silently
dropped — the whole call fails so no radar quietly goes missing.

On success `*out_ptr` carries a heap-allocated array of length
`*out_num`. Free with
[`empyrean_radar_observations_free`](crate::od::empyrean_radar_observations_free).*/
#[inline]
pub unsafe fn empyrean_query_radar(
    designations: *const *const ::std::os::raw::c_char,
    num_designations: usize,
    cache_dir: *const ::std::os::raw::c_char,
    out_ptr: *mut *mut EmpyreanRadarObservation,
    out_num: *mut usize,
) -> i32 {
    unsafe {
        lib().empyrean_query_radar(designations, num_designations, cache_dir, out_ptr, out_num)
    }
}
/** Construct a new orbit-determination session over a fixed
observation set.

`config` is parsed by the same builder the one-shot
`empyrean_determine` / `empyrean_evaluate` / `empyrean_refine`
entry points use, so every field — weighting preset and additional
layers, debiasing, rejection, solve-for, origin and output-epoch
policy — resolves identically on both surfaces. A malformed
weighting layer is rejected here, before any fitting: the call
returns null with the reason in `empyrean_last_error`.

Returns a heap-allocated handle on success, or null on error.
The caller owns the returned pointer and must free it with
[`empyrean_session_free`].*/
#[inline]
pub unsafe fn empyrean_session_new(
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
) -> *mut EmpyreanSession {
    unsafe { lib().empyrean_session_new(observations, num_observations, config) }
}
/** Free a session previously returned by [`empyrean_session_new`].
Passing null is a no-op.*/
#[inline]
pub unsafe fn empyrean_session_free(session: *mut EmpyreanSession) {
    unsafe { lib().empyrean_session_free(session) }
}
/** Total number of observations in the session (masked or not).
Returns 0 if `session` is null.*/
#[inline]
pub unsafe fn empyrean_session_n_observations(session: *const EmpyreanSession) -> usize {
    unsafe { lib().empyrean_session_n_observations(session) }
}
/** Number of observations currently masked.*/
#[inline]
pub unsafe fn empyrean_session_n_masked(session: *const EmpyreanSession) -> usize {
    unsafe { lib().empyrean_session_n_masked(session) }
}
/** Number of observations active (not masked) in the next refine.*/
#[inline]
pub unsafe fn empyrean_session_n_active(session: *const EmpyreanSession) -> usize {
    unsafe { lib().empyrean_session_n_active(session) }
}
/** Mask the observation at `idx`. Returns 0 on success, -1 on null
or out-of-bounds.*/
#[inline]
pub unsafe fn empyrean_session_mask(session: *mut EmpyreanSession, idx: usize) -> i32 {
    unsafe { lib().empyrean_session_mask(session, idx) }
}
/** Unmask the observation at `idx`. Returns 0 on success, -1 on null
or out-of-bounds.*/
#[inline]
pub unsafe fn empyrean_session_unmask(session: *mut EmpyreanSession, idx: usize) -> i32 {
    unsafe { lib().empyrean_session_unmask(session, idx) }
}
/** Clear all masks. Returns 0 on success, -1 on null.*/
#[inline]
pub unsafe fn empyrean_session_unmask_all(session: *mut EmpyreanSession) -> i32 {
    unsafe { lib().empyrean_session_unmask_all(session) }
}
/** Whether the observation at `idx` is masked. Returns 1 = masked,
0 = active, 255 (-1 cast) on null/out-of-bounds.*/
#[inline]
pub unsafe fn empyrean_session_is_masked(session: *const EmpyreanSession, idx: usize) -> u8 {
    unsafe { lib().empyrean_session_is_masked(session, idx) }
}
/** Run an OD refine using the current mask state.

On the first call, runs the full IOD → DC pipeline. On subsequent
calls, uses the previously-fit orbit as the IOD seed (skipping the
IOD step). Pushes the new fit onto the session's history.

On success populates `result_out` with the latest fit. The caller
must free `result_out` with [`empyrean_od_result_free`](crate::od::empyrean_od_result_free).*/
#[inline]
pub unsafe fn empyrean_session_refine(
    session: *mut EmpyreanSession,
    ctx: *const EmpyreanContext,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    unsafe { lib().empyrean_session_refine(session, ctx, result_out) }
}
/** Number of fits in the session history.*/
#[inline]
pub unsafe fn empyrean_session_history_len(session: *const EmpyreanSession) -> usize {
    unsafe { lib().empyrean_session_history_len(session) }
}
/** Copy the i-th history entry into `result_out`. Returns 0 on
success, -1 on null/out-of-bounds. Caller frees `result_out` with
[`empyrean_od_result_free`](crate::od::empyrean_od_result_free).*/
#[inline]
pub unsafe fn empyrean_session_get_history(
    session: *const EmpyreanSession,
    idx: usize,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    unsafe { lib().empyrean_session_get_history(session, idx, result_out) }
}
/** Diff the current fit against an earlier history entry. Returns 0
on success, -1 if there is no current fit or `prior_idx` is out
of bounds.*/
#[inline]
pub unsafe fn empyrean_session_diff(
    session: *const EmpyreanSession,
    prior_idx: usize,
    diff_out: *mut EmpyreanSessionDiff,
) -> i32 {
    unsafe { lib().empyrean_session_diff(session, prior_idx, diff_out) }
}
/** Query body states relative to a center body at given epochs.

Returns 0 on success, negative error code on failure.
On success, `result_out` is populated with body states.
The caller must free the result with `empyrean_state_result_free()`.*/
#[inline]
pub unsafe fn empyrean_get_states(
    ctx: *const EmpyreanContext,
    target_naif_id: i32,
    center_naif_id: i32,
    epochs_mjd_tdb: *const f64,
    num_epochs: usize,
    frame: i32,
    result_out: *mut EmpyreanStateResult,
) -> i32 {
    unsafe {
        lib().empyrean_get_states(
            ctx,
            target_naif_id,
            center_naif_id,
            epochs_mjd_tdb,
            num_epochs,
            frame,
            result_out,
        )
    }
}
/** Free a state result previously returned by `empyrean_get_states()`.

Passing a zeroed/null result is a no-op.*/
#[inline]
pub unsafe fn empyrean_state_result_free(result: *mut EmpyreanStateResult) {
    unsafe { lib().empyrean_state_result_free(result) }
}
/** Parse an ISO 8601 UTC string (e.g. ``"2024-08-01T00:00:00.000Z"``)
to MJD in the requested target scale.

`scale` is `0` for UTC, `1` for TDB.

On success writes the MJD value to `*out_mjd` and returns 0.
On failure returns a negative code; consult
[`empyrean_last_error`](crate::empyrean_last_error).*/
#[inline]
pub unsafe fn empyrean_iso_to_mjd(
    iso: *const ::std::os::raw::c_char,
    scale: i32,
    out_mjd: *mut f64,
) -> i32 {
    unsafe { lib().empyrean_iso_to_mjd(iso, scale, out_mjd) }
}
/** Format an MJD value (in the given scale) as an ISO 8601 UTC string.

`scale` is `0` for UTC, `1` for TDB. Writes a null-terminated
string of length ≤ `buf_len-1` into `out_buf`. A 32-byte buffer is
always sufficient (typical output is 24 bytes:
``"2024-08-01T00:00:00.000Z"``).

Returns 0 on success; negative on failure.*/
#[inline]
pub unsafe fn empyrean_mjd_to_iso(
    mjd: f64,
    scale: i32,
    out_buf: *mut ::std::os::raw::c_char,
    buf_len: usize,
) -> i32 {
    unsafe { lib().empyrean_mjd_to_iso(mjd, scale, out_buf, buf_len) }
}
/** Transform a **batch** of coordinate states to a new representation,
frame, and/or origin.

The batched form carries the main name; the unit of work is
[`empyrean_transform_coordinates_single`]. Element `i` of `states_out`
is **bit-identical** to that function applied to `states[i]` with the
same target arguments, so batching is a scheduling choice and never a
numerical one.

`states` and `states_out` are both caller-owned arrays of exactly
`num_states` elements — nothing is heap-allocated here and there is no
result to free. They may not overlap. `num_states == 0` is a valid
no-op call (both pointers are then ignored and may be null).

# What the batch does and does not buy

Two costs in the transform pipeline do not depend on the individual
state and the engine pays each once per distinct key rather than once
per state: gravitational-parameter resolution (keyed on the origin)
and the origin shift (keyed on `(from, to, epoch)`). States sharing an
epoch — a catalogue snapshot, a sigma-point cloud — reuse one shift.

Note what that does **not** buy: those memos are scoped to the
context, not to the call, so a loop over
[`empyrean_transform_coordinates_single`] on the same context hits
them too. Measured against such a loop the batch is ~17% faster from a
thousand states up and indistinguishable at one — what it saves is the
per-call boundary crossing, not the shift. Reach for it for the shape
(one call, one error, one index) rather than for a large speedup. A
batch that reuses nothing costs the same as the equivalent
single-state loop rather than more.

# Returns

`0` on success. `-1` for a null pointer or an unresolvable target
basis — an argument-shaped failure of the call as a whole; `-2` when
an individual element fails, whether the input state is malformed,
the engine refuses it, or its result cannot be marshaled back out.
**Fail-fast, with the index**: the first failing element aborts the
batch, `states_out` is left untouched, and `empyrean_last_error()`
names the zero-based index of the element that failed along with its
underlying cause. No partial output is ever written — a batch either
transforms completely or not at all.*/
#[inline]
pub unsafe fn empyrean_transform_coordinates(
    ctx: *const EmpyreanContext,
    states: *const CoordinateState,
    num_states: usize,
    target_representation: i32,
    target_frame: i32,
    target_origin: i32,
    states_out: *mut CoordinateState,
) -> i32 {
    unsafe {
        lib().empyrean_transform_coordinates(
            ctx,
            states,
            num_states,
            target_representation,
            target_frame,
            target_origin,
            states_out,
        )
    }
}
/** Transform **one** coordinate state to a new representation, frame,
and/or origin.

The single-state unit of work; [`empyrean_transform_coordinates`] is
the batched form that transforms a whole array in one call. Reach for
the batch when a whole table is going to the same target basis — for
the call shape (one call, one error, one index), not for a large
speedup: the engine's memos are scoped to the context, so a loop over
this function on the same context amortizes them just as well. See
that function's docs for the measurement.

Covariance is propagated through the Jacobian of the transformation
when the input state carries one.

Returns 0 on success or a negative error code on failure.
Call `empyrean_last_error()` to retrieve the error message on failure.*/
#[inline]
pub unsafe fn empyrean_transform_coordinates_single(
    ctx: *const EmpyreanContext,
    input: *const CoordinateState,
    target_representation: i32,
    target_frame: i32,
    target_origin: i32,
    output: *mut CoordinateState,
) -> i32 {
    unsafe {
        lib().empyrean_transform_coordinates_single(
            ctx,
            input,
            target_representation,
            target_frame,
            target_origin,
            output,
        )
    }
}

/**The propagated joint's CROSS terms at a single `(orbit_index,
epoch_index)` — the state↔Marsden border and the wide carrier that
[`EmpyreanTaggedCovariance::matrix`] is the state block of.

# Why this is a separate call

[`EmpyreanTaggedCovariance`] is a plain-old-data struct: a caller
declares one on the stack, fills it through
[`empyrean_propagation_covariance_at_cartesian`], and frees nothing.
Putting the carrier on it would have made every such caller — code
that is correct today and recompiles without a murmur — leak two
allocations per call. Nothing would fail; memory would simply grow.
So the joint is opt-in through this call instead, and the tagged
covariance keeps its contract.

The same `(orbit_index, epoch_index)` addresses both surfaces, so a
consumer walking a series
([`empyrean_propagation_covariance_series_cartesian`]) can ask for
the joint of any entry without the series itself owning anything new.

# Absence is reported, never fabricated

`has_non_grav_cross = 0` with null carrier pointers means the engine
produced no cross terms at this row — not that they were zero. Every
uncertainty method that reaches this accessor carries the payload
when it has one, including the sampled paths, which recover the
state↔parameter columns from the cloud. A row genuinely without a
joint is one whose orbit declared no solved-parameter block.

# Ownership

On success `out` owns two heap arrays; release them with
[`empyrean_orbit_covariance_free`]. On any non-zero return `out` is
untouched and there is nothing to free.

# Returns

The same `EMPYREAN_TAGGED_COV_*` codes as the tagged-covariance
accessors, so a caller branches on one set.

# Safety
`result` and `out` must be valid pointers; `result` from
`empyrean_propagate`.*/
#[inline]
pub unsafe fn empyrean_propagation_joint_at(
    result: *const EmpyreanPropagationResult,
    orbit_index: usize,
    epoch_index: usize,
    out: *mut EmpyreanOrbitCovariance,
) -> i32 {
    unsafe { lib().empyrean_propagation_joint_at(result, orbit_index, epoch_index, out) }
}

/**Release the arrays owned by an [`EmpyreanOrbitCovariance`](crate::joint::EmpyreanOrbitCovariance)
written by [`empyrean_propagation_joint_at`].

Idempotent — the pointers are nulled and the counts zeroed — and a
null argument is a no-op. Do **not** call it on the `orbit_cov` of an
OD result or a propagated state: those are owned by their parent and
released by the parent's own free function.

# Safety
`cov` must be null or a struct written by
[`empyrean_propagation_joint_at`] and not already freed.*/
#[inline]
pub unsafe fn empyrean_orbit_covariance_free(cov: *mut EmpyreanOrbitCovariance) {
    unsafe { lib().empyrean_orbit_covariance_free(cov) }
}
