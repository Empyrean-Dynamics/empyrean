use std::panic::AssertUnwindSafe;

use empyrean_core::Origin;
use empyrean_core::convert::{
    coordinate_state_to_coordinates, coordinates_to_coordinate_state, int_to_frame,
    int_to_representation,
};

use crate::{CoordinateState, EmpyreanContext, set_last_error};

/// Resolve the three target-basis integers once for a whole call.
///
/// Shared by the batched and single-state entry points so neither can
/// drift in what it accepts: an unknown representation, an unknown frame,
/// or a NAIF id with no `Origin` is `-1` on both.
fn resolve_target(
    target_representation: i32,
    target_frame: i32,
    target_origin: i32,
) -> Result<
    (
        empyrean_core::coordinates::CoordinateRepresentation,
        empyrean_core::coordinates::Frame,
        Origin,
    ),
    String,
> {
    let target_rep = int_to_representation(target_representation).map_err(|e| e.to_string())?;
    let target_frm = int_to_frame(target_frame).map_err(|e| e.to_string())?;
    let target_orig = Origin::from_naif_id(target_origin)
        .ok_or_else(|| format!("unknown NAIF id: {target_origin}"))?;
    Ok((target_rep, target_frm, target_orig))
}

/// Marshal a whole transformed batch back to the flat C shape, or fail.
///
/// Two guards, both about the same thing — the caller's array must come
/// back complete or not at all.
///
/// **Row correspondence.** `expected` is what the caller's array is
/// sized for, and element `i` out belongs to element `i` in. Checking it
/// here rather than at the write loop matters because the natural write
/// (`zip`) would stop at the shorter side and report success, leaving
/// the tail of `states_out` untouched — and an untouched zeroed
/// `CoordinateState` reads back as a perfectly valid Cartesian / ICRF /
/// SSB state at MJD 0. Zero-fill presented as data is the failure the
/// output contract exists to prevent.
///
/// **All-or-nothing.** Every row is flattened before any is written,
/// because marshaling is itself fallible (an observatory origin has no
/// NAIF id), and a half-written output array is indistinguishable from a
/// complete one on the C side.
fn flatten_batch(
    transformed: &[empyrean_core::coordinates::Coordinates<
        empyrean_core::coordinates::AU,
        empyrean_core::coordinates::Degrees,
    >],
    expected: usize,
) -> Result<Vec<CoordinateState>, String> {
    if transformed.len() != expected {
        return Err(format!(
            "batch transform returned {} state(s) for {expected} input(s) — the row \
             correspondence the caller's array depends on is broken",
            transformed.len()
        ));
    }
    let mut flat = Vec::with_capacity(transformed.len());
    for (i, c) in transformed.iter().enumerate() {
        match coordinates_to_coordinate_state(c) {
            Ok(cs) => {
                let mut out = CoordinateState::from_empyrean(&cs);
                // The core flat type carries no border, so read it off
                // the transformed `Coordinates` directly. The engine
                // rotated it with the 6×6 it borders — dropping it here
                // would silently return half a joint through a basis
                // change, which is the failure this whole surface exists
                // to remove.
                let (has_border, cross) = crate::joint::border_to_c(c.extended_covariance());
                out.has_non_grav_cross = has_border;
                out.non_grav_cross = cross;
                flat.push(out);
            }
            Err(e) => return Err(format!("batch element {i} failed to transform: {e}")),
        }
    }
    Ok(flat)
}

/// Rebuild the engine coordinate a flat [`CoordinateState`] describes,
/// **including** its state↔Marsden border.
///
/// Both transform entry points take a bare `CoordinateState` with no
/// orbit beside it, so there is no `non_grav_covariance` in scope to
/// pair the border with. The engine only ever rotates the border's
/// `cross` half through the state Jacobian and never reads `params` on
/// this path, so a zero parameter block is inert here rather than a
/// substituted value — and the caller gets the same `cross` back,
/// rotated, with `params` still living on their orbit.
///
/// Note what the entry points therefore do and do not move: a coordinate
/// and its border, never a whole joint. An orbit's wide carrier is not
/// in scope at this signature — it is an orbit-level object, and the
/// coordinate does not carry it.
fn coordinate_state_to_bordered_coordinates(
    s: &CoordinateState,
) -> Result<
    empyrean_core::coordinates::Coordinates<
        empyrean_core::coordinates::AU,
        empyrean_core::coordinates::Degrees,
    >,
    String,
> {
    let coords = coordinate_state_to_coordinates(&s.to_empyrean()).map_err(|e| e.to_string())?;
    let ext =
        crate::joint::border_from_c(s.has_non_grav_cross, &s.non_grav_cross, [[0.0_f64; 3]; 3]);
    Ok(crate::joint::coordinates_with_extended(coords, ext))
}

/// Transform a **batch** of coordinate states to a new representation,
/// frame, and/or origin.
///
/// The batched form carries the main name; the unit of work is
/// [`empyrean_transform_coordinates_single`]. Element `i` of `states_out`
/// is **bit-identical** to that function applied to `states[i]` with the
/// same target arguments, so batching is a scheduling choice and never a
/// numerical one.
///
/// `states` and `states_out` are both caller-owned arrays of exactly
/// `num_states` elements — nothing is heap-allocated here and there is no
/// result to free. They may not overlap. `num_states == 0` is a valid
/// no-op call (both pointers are then ignored and may be null).
///
/// # What the batch does and does not buy
///
/// Two costs in the transform pipeline do not depend on the individual
/// state and the engine pays each once per distinct key rather than once
/// per state: gravitational-parameter resolution (keyed on the origin)
/// and the origin shift (keyed on `(from, to, epoch)`). States sharing an
/// epoch — a catalogue snapshot, a sigma-point cloud — reuse one shift.
///
/// Note what that does **not** buy: those memos are scoped to the
/// context, not to the call, so a loop over
/// [`empyrean_transform_coordinates_single`] on the same context hits
/// them too. Measured against such a loop the batch is ~17% faster from a
/// thousand states up and indistinguishable at one — what it saves is the
/// per-call boundary crossing, not the shift. Reach for it for the shape
/// (one call, one error, one index) rather than for a large speedup. A
/// batch that reuses nothing costs the same as the equivalent
/// single-state loop rather than more.
///
/// # Returns
///
/// `0` on success. `-1` for a null pointer or an unresolvable target
/// basis — an argument-shaped failure of the call as a whole; `-2` when
/// an individual element fails, whether the input state is malformed,
/// the engine refuses it, or its result cannot be marshaled back out.
/// **Fail-fast, with the index**: the first failing element aborts the
/// batch, `states_out` is left untouched, and `empyrean_last_error()`
/// names the zero-based index of the element that failed along with its
/// underlying cause. No partial output is ever written — a batch either
/// transforms completely or not at all.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_transform_coordinates(
    ctx: *const EmpyreanContext,
    states: *const CoordinateState,
    num_states: usize,
    target_representation: i32,
    target_frame: i32,
    target_origin: i32,
    states_out: *mut CoordinateState,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() {
            set_last_error("null context pointer");
            return -1;
        }
        // An empty batch touches nothing — mirror the engine, which
        // returns an empty Vec without consulting the SPK set.
        if num_states == 0 {
            return 0;
        }
        if states.is_null() || states_out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let in_slice = unsafe { std::slice::from_raw_parts(states, num_states) };

        let (target_rep, target_frm, target_orig) =
            match resolve_target(target_representation, target_frame, target_origin) {
                Ok(t) => t,
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            };

        // Element-attributed up front too: a malformed input state is
        // just as much "element i failed" as an engine-side failure, and
        // a caller with a 10⁵-row catalogue cannot find the bad row from
        // a bare message. It returns -2 for the same reason — the code is
        // the part of the contract a C caller branches on without
        // parsing the message, so an element failure must not arrive
        // wearing the code that means "your arguments are wrong".
        let mut coords_in = Vec::with_capacity(num_states);
        for (i, s) in in_slice.iter().enumerate() {
            match coordinate_state_to_bordered_coordinates(s) {
                Ok(c) => coords_in.push(c),
                Err(e) => {
                    set_last_error(&format!("batch element {i} failed to transform: {e}"));
                    return -2;
                }
            }
        }

        let transformed = match empyrean_core::coordinates::transform_coordinates(
            ctx_ref,
            &coords_in,
            target_rep,
            target_frm,
            target_orig,
        ) {
            Ok(t) => t,
            // The engine's `TransformError::Element` already carries the
            // failing index and renders as "batch element {i} failed to
            // transform: {cause}", so the index reaches the caller
            // through the normal message channel without being
            // re-derived here.
            Err(e) => {
                set_last_error(&e.to_string());
                return -2;
            }
        };

        let flat = match flatten_batch(&transformed, num_states) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e);
                return -2;
            }
        };

        let out_slice = unsafe { std::slice::from_raw_parts_mut(states_out, num_states) };
        out_slice.copy_from_slice(&flat);
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_transform_coordinates");
            -99
        }
    }
}

/// Transform **one** coordinate state to a new representation, frame,
/// and/or origin.
///
/// The single-state unit of work; [`empyrean_transform_coordinates`] is
/// the batched form that transforms a whole array in one call. Reach for
/// the batch when a whole table is going to the same target basis — for
/// the call shape (one call, one error, one index), not for a large
/// speedup: the engine's memos are scoped to the context, so a loop over
/// this function on the same context amortizes them just as well. See
/// that function's docs for the measurement.
///
/// Covariance is propagated through the Jacobian of the transformation
/// when the input state carries one.
///
/// Returns 0 on success or a negative error code on failure.
/// Call `empyrean_last_error()` to retrieve the error message on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_transform_coordinates_single(
    ctx: *const EmpyreanContext,
    input: *const CoordinateState,
    target_representation: i32,
    target_frame: i32,
    target_origin: i32,
    output: *mut CoordinateState,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || input.is_null() || output.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let input_state = unsafe { &*input };

        let (target_rep, target_frm, target_orig) =
            match resolve_target(target_representation, target_frame, target_origin) {
                Ok(t) => t,
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            };

        let coords_in = match coordinate_state_to_bordered_coordinates(input_state) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        match empyrean_core::coordinates::transform_coordinates_single(
            ctx_ref,
            coords_in,
            target_rep,
            target_frm,
            target_orig,
        ) {
            Ok(transformed) => match coordinates_to_coordinate_state(&transformed) {
                Ok(flat) => {
                    let mut out = CoordinateState::from_empyrean(&flat);
                    let (has_border, cross) =
                        crate::joint::border_to_c(transformed.extended_covariance());
                    out.has_non_grav_cross = has_border;
                    out.non_grav_cross = cross;
                    unsafe {
                        *output = out;
                    }
                    0
                }
                Err(e) => {
                    set_last_error(&e.to_string());
                    -2
                }
            },
            Err(e) => {
                set_last_error(&e.to_string());
                -2
            }
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_transform_coordinates_single");
            -99
        }
    }
}

/// The batch/single contract at the ABI boundary (the v3 batch-first
/// rename).
///
/// The batched symbol is the one that grew; the single-state symbol is
/// the same code path it always was, under a new name. What these pin is
/// the seam between them: element `i` of the batch must be the single
/// call on element `i`, **bit for bit**, or "batching is a scheduling
/// choice" is a false claim and every consumer that mixes the two paths
/// silently disagrees with itself.
///
/// Gated on a real data directory; skipped without one.
#[cfg(test)]
mod batch_contract_tests {
    use super::*;

    /// Cartesian, ICRF, heliocentric. Distinct epochs AND a shared epoch
    /// so the batch exercises both the memo-hit and memo-miss arms of the
    /// engine's origin-shift cache.
    fn inputs() -> Vec<CoordinateState> {
        let base = |epoch: f64, x: f64| CoordinateState {
            epoch_mjd_tdb: epoch,
            elements: [x, 0.1, 0.05, -0.005, 0.015, 0.001],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: 0, // Cartesian
            frame: 0,          // ICRF
            origin: 10,        // Sun
            has_non_grav_cross: 0,
            non_grav_cross: [[0.0; 3]; 6],
        };
        vec![
            base(59000.0, 1.0),
            base(59000.0, 1.4), // shares an epoch with [0] — memo hit
            base(59200.0, 2.3), // distinct epoch — memo miss
        ]
    }

    fn zeroed_out(n: usize) -> Vec<CoordinateState> {
        (0..n)
            .map(|_| CoordinateState {
                epoch_mjd_tdb: 0.0,
                elements: [0.0; 6],
                covariance: [[0.0; 6]; 6],
                has_covariance: 0,
                representation: 0,
                frame: 0,
                origin: 0,
                has_non_grav_cross: 0,
                non_grav_cross: [[0.0; 3]; 6],
            })
            .collect()
    }

    /// The central contract: batch element `i` == single on element `i`,
    /// bit for bit, across a representation + frame + origin change.
    #[test]
    fn batch_element_is_bit_identical_to_single() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping batch_element_is_bit_identical_to_single: no data dir");
            return;
        };
        let states = inputs();
        // Keplerian (1), EclipticJ2000 (1), SSB (0) — every axis of the
        // transform moves, and the origin change forces the SPK shift.
        let (rep, frame, origin) = (1, 1, 0);

        let mut batch_out = zeroed_out(states.len());
        let code = unsafe {
            empyrean_transform_coordinates(
                &ctx,
                states.as_ptr(),
                states.len(),
                rep,
                frame,
                origin,
                batch_out.as_mut_ptr(),
            )
        };
        assert_eq!(code, 0, "batch transform must succeed");

        for (i, s) in states.iter().enumerate() {
            let mut single = zeroed_out(1);
            let code = unsafe {
                empyrean_transform_coordinates_single(
                    &ctx,
                    s,
                    rep,
                    frame,
                    origin,
                    single.as_mut_ptr(),
                )
            };
            assert_eq!(code, 0, "single transform {i} must succeed");
            let (b, o) = (&batch_out[i], &single[0]);
            assert_eq!(
                b.elements, o.elements,
                "element {i}: batch elements must be BIT-identical to the single call"
            );
            assert_eq!(b.epoch_mjd_tdb, o.epoch_mjd_tdb, "element {i}: epoch");
            assert_eq!(b.representation, o.representation, "element {i}: rep");
            assert_eq!(b.frame, o.frame, "element {i}: frame");
            assert_eq!(b.origin, o.origin, "element {i}: origin");
            assert_eq!(b.has_covariance, o.has_covariance, "element {i}: has_cov");
            assert_eq!(b.covariance, o.covariance, "element {i}: covariance");
        }
    }

    /// The row-correspondence guard, exercised directly.
    ///
    /// It can only fire if the engine ever returns a different number of
    /// states than it was given, which no input can force from the C
    /// side — so testing it through `empyrean_transform_coordinates`
    /// would mean testing a branch no test can reach. Drive the marshal
    /// step itself instead: a short result must be an error, never a
    /// short write that the caller reads back as zero-filled data.
    #[test]
    fn a_short_result_is_refused_rather_than_short_written() {
        let coords: Vec<_> = inputs()
            .iter()
            .map(|s| {
                coordinate_state_to_coordinates(&s.to_empyrean())
                    .expect("the fixture states are well-formed")
            })
            .collect();
        assert_eq!(coords.len(), 3);

        // The honest case: lengths agree, every row marshals.
        let ok = flatten_batch(&coords, 3).expect("a full result marshals");
        assert_eq!(ok.len(), 3);

        // Short by one — a dropped row, a filtered element, an early
        // return upstream. This is the shape that used to be written out
        // as two good rows plus an untouched (zeroed) third, with a
        // success code on top.
        let err = flatten_batch(&coords[..2], 3)
            .expect_err("a short result must not be marshaled into a 3-row array");
        assert!(
            err.contains("2 state(s) for 3 input(s)"),
            "the error states both counts: {err}"
        );
        assert!(
            err.contains("row correspondence"),
            "the error names what broke: {err}"
        );

        // Long, too — a duplicated row breaks the correspondence just as
        // thoroughly as a dropped one.
        let mut four = coords.clone();
        four.push(coords[0]);
        assert!(
            flatten_batch(&four, 3).is_err(),
            "a long result is refused too"
        );
    }

    /// The border survives a basis change rather than being dropped.
    ///
    /// `CoordinateState` gained the state↔Marsden border in v4, and both
    /// transform entry points take a bare `CoordinateState` — so the
    /// natural implementation (flatten through the core type, which has
    /// no border) would silently return half a joint through every
    /// representation change. The engine rotates the border with the 6×6
    /// it borders; this pins that it reaches the caller.
    #[test]
    fn the_marsden_border_survives_a_basis_change() {
        let Some(ctx) =
            crate::testing::context_or_skip("the_marsden_border_survives_a_basis_change")
        else {
            return;
        };
        let mut input = inputs()[0];
        // A border needs its 6×6: the engine refuses a cross with no
        // state block to border, which is the guard this test must stay
        // on the right side of.
        for i in 0..6 {
            input.covariance[i][i] = 1.0e-8 * (i as f64 + 1.0);
        }
        input.has_covariance = 1;
        input.has_non_grav_cross = 1;
        input.non_grav_cross = [[1.0e-12, 2.0e-12, 3.0e-12]; 6];

        let mut out = zeroed_out(1);
        // Keplerian + EclipticJ2000: both the representation and the
        // frame move, so the border goes through a real Jacobian rather
        // than an identity.
        let code = unsafe {
            empyrean_transform_coordinates_single(&ctx, &input, 1, 1, 10, out.as_mut_ptr())
        };
        assert_eq!(code, 0, "the transform must accept a bordered state");

        assert_eq!(
            out[0].has_non_grav_cross, 1,
            "the border must reach the caller, not be dropped at the flat boundary"
        );
        assert!(
            out[0]
                .non_grav_cross
                .iter()
                .flatten()
                .all(|v| v.is_finite()),
            "every border entry must be finite after the rotation"
        );
        assert_ne!(
            out[0].non_grav_cross, input.non_grav_cross,
            "the border must be ROTATED, not passed through unchanged — a copy \
             would be a border describing the old basis attached to a new one"
        );

        // And the batch path agrees with the single path, border included.
        let mut batch_out = zeroed_out(1);
        let code = unsafe {
            empyrean_transform_coordinates(&ctx, &input, 1, 1, 1, 10, batch_out.as_mut_ptr())
        };
        assert_eq!(code, 0);
        assert_eq!(batch_out[0].has_non_grav_cross, out[0].has_non_grav_cross);
        assert_eq!(
            batch_out[0].non_grav_cross, out[0].non_grav_cross,
            "batch element 0 must be bit-identical to the single call, border included"
        );
    }

    /// A state with no border transforms exactly as it did before v4 —
    /// the zero-init contract, exercised through the shipped entry
    /// point rather than asserted about the struct.
    #[test]
    fn a_borderless_state_is_unchanged_by_the_new_field() {
        let Some(ctx) =
            crate::testing::context_or_skip("a_borderless_state_is_unchanged_by_the_new_field")
        else {
            return;
        };
        let input = inputs()[0];
        assert_eq!(input.has_non_grav_cross, 0, "the fixture carries no border");
        let mut out = zeroed_out(1);
        let code = unsafe {
            empyrean_transform_coordinates_single(&ctx, &input, 1, 1, 0, out.as_mut_ptr())
        };
        assert_eq!(code, 0);
        assert_eq!(out[0].has_non_grav_cross, 0, "no border in, no border out");
        assert_eq!(
            out[0].non_grav_cross, [[0.0; 3]; 6],
            "and no fabricated values"
        );
    }

    /// An empty batch is a no-op that touches neither pointer — the
    /// engine's own contract, mirrored so a caller with a zero-row
    /// catalogue does not have to special-case the call.
    #[test]
    fn an_empty_batch_is_a_no_op() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping an_empty_batch_is_a_no_op: no data dir");
            return;
        };
        let code = unsafe {
            empyrean_transform_coordinates(&ctx, std::ptr::null(), 0, 1, 1, 0, std::ptr::null_mut())
        };
        assert_eq!(
            code, 0,
            "an empty batch must succeed without touching anything"
        );
    }

    /// The two documented return codes mean different things and a C
    /// caller branches on them without parsing the message, so each has
    /// to be pinned to the failure class the header claims for it: `-1`
    /// is "your arguments are wrong", `-2` is "element `i` failed".
    #[test]
    fn the_return_codes_match_their_documented_meanings() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping the_return_codes_match_their_documented_meanings: no data dir");
            return;
        };
        let states = inputs();
        let mut out = zeroed_out(states.len());

        // An unresolvable target basis is a whole-call argument error.
        for (rep, frame, origin, what) in [
            (99, 1, 0, "representation"),
            (1, 99, 0, "frame"),
            (1, 1, 987_654, "origin NAIF id"),
        ] {
            let code = unsafe {
                empyrean_transform_coordinates(
                    &ctx,
                    states.as_ptr(),
                    states.len(),
                    rep,
                    frame,
                    origin,
                    out.as_mut_ptr(),
                )
            };
            assert_eq!(code, -1, "an unresolvable target {what} is -1, not -2");
        }

        // A null argument is the same class.
        let code = unsafe {
            empyrean_transform_coordinates(
                &ctx,
                std::ptr::null(),
                states.len(),
                1,
                1,
                0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, -1, "a null states pointer is -1");

        // A malformed element is an element failure, so -2 — even though
        // it is caught before the engine is ever entered.
        let mut bad = inputs();
        bad[1].representation = 99;
        let code = unsafe {
            empyrean_transform_coordinates(&ctx, bad.as_ptr(), bad.len(), 1, 1, 0, out.as_mut_ptr())
        };
        assert_eq!(
            code, -2,
            "a malformed input element is an element failure (-2), not an argument error (-1)"
        );
    }

    /// A failing element aborts the whole batch, names its index, and
    /// leaves the output array untouched — no partial write that a C
    /// caller would read back as a complete answer.
    #[test]
    fn a_failing_element_names_its_index_and_writes_nothing() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping a_failing_element_names_its_index_and_writes_nothing: no data dir");
            return;
        };
        let mut states = inputs();
        // Element 1 carries an unknown representation tag: the flat ->
        // Coordinates conversion refuses it.
        states[1].representation = 99;

        let sentinel = zeroed_out(states.len());
        let mut out = sentinel.clone();
        let code = unsafe {
            empyrean_transform_coordinates(
                &ctx,
                states.as_ptr(),
                states.len(),
                1,
                1,
                0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(
            code, -2,
            "a malformed element fails the batch with the element-failure code"
        );

        let msg = unsafe {
            std::ffi::CStr::from_ptr(crate::empyrean_last_error())
                .to_string_lossy()
                .into_owned()
        };
        assert!(
            msg.contains("element 1"),
            "the error names the failing element index: {msg}"
        );
        for (i, (got, want)) in out.iter().zip(sentinel.iter()).enumerate() {
            assert_eq!(
                got.elements, want.elements,
                "row {i} of the output array must be untouched on failure"
            );
        }
    }
}
