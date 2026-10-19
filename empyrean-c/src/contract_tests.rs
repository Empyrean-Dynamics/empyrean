//! 0.11.0 packed-joint contract tests (pass 14).
//!
//! These pin the rulings the adoption rests on at the C ABI boundary: the
//! retired `ThirdOrder` kind is refused by name; a non-grav uncertainty
//! round-trips ONLY on the packed joint; a malformed wire row returns the
//! SAME per-variant code at a direct push and through a file writer (the
//! typed `file_in_op` boundary); the bare coordinate transform refuses a
//! covariance by name; and `empyrean_transform_orbits` transports the joint.
//!
//! Synthetic inputs throughout: every orbit here is **Cartesian**, whose
//! state has no angular rows, so the degree <-> radian congruence the ABI
//! applies to angular state rows is the identity and the packed triangle
//! round-trips bit-for-bit — `to_bits`/exact equality is admissible on these
//! synthetic inputs (and only these).

use crate::CoordinateState;
use crate::io::EmpyreanOrbitBatch;
use crate::joint::wire_tags::{
    EMPYREAN_JOINT_ASSERTED_SOURCE_NONE, EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB,
    EMPYREAN_JOINT_PROVENANCE_ASSERTED,
};
use crate::joint::{
    EMPYREAN_PUSH_LAYOUT_INVALID, EMPYREAN_PUSH_RETIRED_DISCRIMINANT,
    EMPYREAN_PUSH_TRIANGLE_LENGTH, EMPYREAN_PUSH_UNKNOWN_ASSERTED_SOURCE,
    EMPYREAN_PUSH_UNKNOWN_FUNCTIONAL, EMPYREAN_PUSH_UNKNOWN_PROVENANCE,
    EMPYREAN_PUSH_UNSUPPORTED_INPUT_KIND, EMPYREAN_PUSH_WIDTH_MISMATCH, EmpyreanPackedJoint,
};
use crate::propagate::EmpyreanOrbit;

/// A packed lower triangle of the given width with a small positive diagonal
/// and zeros elsewhere — a well-formed, positive synthetic covariance.
fn diag_tri(width: usize) -> Vec<f64> {
    let n = width * (width + 1) / 2;
    let mut t = vec![0.0f64; n];
    for i in 0..width {
        t[i * (i + 1) / 2 + i] = 1.0e-8 * (i as f64 + 1.0);
    }
    t
}

/// A well-formed width-6 state-only joint borrowing `tri` (caller keeps it
/// alive). The per-variant tests then corrupt ONE field to trigger exactly
/// one `UncertaintyWireError`.
fn joint_borrowing(tri: &[f64], width: u8) -> EmpyreanPackedJoint {
    EmpyreanPackedJoint {
        layout_present: 0,
        layout_considered: 0,
        layout_marginalized: 0,
        width,
        tri: tri.as_ptr(),
        tri_len: tri.len(),
        kind: 0,    // Linear
        quality: 0, // output-only; re-derived by the engine on input
        cov_rank: 0,
        cov_null_axes: 0,
        cov_min_eig_corr: f64::NAN,
        functional: 0,                                        // State
        provenance: 0,                                        // Derived
        asserted_source: EMPYREAN_JOINT_ASSERTED_SOURCE_NONE, // no source on a derived joint
    }
}

/// A zeroed Cartesian orbit (ICRF, heliocentric) carrying `joint`. `tri` must
/// outlive every use. Cartesian + h_mag 0 + has_srp 0 + a1/a2/a3 0 ⇒ a pure
/// state orbit whose only uncertainty is the supplied joint.
fn orbit_with_joint(joint: EmpyreanPackedJoint) -> EmpyreanOrbit {
    // SAFETY: `EmpyreanOrbit` is `#[repr(C)]` of scalars and raw pointers with
    // no niche; an all-zero bit pattern is the documented `memset(0)` orbit
    // (null pointers, absent photometry/SRP/non-grav).
    let mut orbit: EmpyreanOrbit = unsafe { std::mem::zeroed() };
    orbit.state = CoordinateState {
        epoch_mjd_tdb: 59000.0,
        elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
        covariance: [[0.0; 6]; 6],
        has_covariance: 0,
        representation: 0, // Cartesian
        frame: 0,          // ICRF
        origin: 10,        // Sun
    };
    orbit.phot_system = -1;
    orbit.h_mag = f64::NAN;
    orbit.non_grav_dt = f64::NAN;
    orbit.covariance = joint;
    orbit
}

/// The push code at a DIRECT propagate-input boundary: `build_orbits_for_propagation`
/// returns the typed `AbiError` whose `.code` is the per-variant wire code.
fn push_code_direct(orbit: &EmpyreanOrbit) -> i32 {
    match crate::propagate::build_orbits_for_propagation(std::slice::from_ref(orbit)) {
        Ok(_) => 0,
        Err(e) => e.code.unwrap_or(-1),
    }
}

/// The push code through the FILE-WRITE boundary: `empyrean_orbits_write_csv`
/// converts the C row to an engine orbit first, so a malformed wire row is the
/// same refusal as at a direct push and must return the same code via the
/// typed `file_in_op` boundary. The CSV file is never created — the refusal
/// precedes `write_orbits_csv`.
fn push_code_csv(orbit: &EmpyreanOrbit) -> i32 {
    let id = std::ffi::CString::new("o0").unwrap();
    let mut id_ptr = id.as_ptr() as *mut std::ffi::c_char;
    let batch = EmpyreanOrbitBatch {
        orbits: orbit as *const EmpyreanOrbit as *mut EmpyreanOrbit,
        orbit_ids: &mut id_ptr as *mut *mut std::ffi::c_char,
        object_ids: std::ptr::null_mut(),
        num_orbits: 1,
    };
    let path = std::env::temp_dir().join("empyrean_pass14_push_code.csv");
    let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    unsafe { crate::io::empyrean_orbits_write_csv(cpath.as_ptr(), &batch) }
}

fn last_error() -> String {
    unsafe {
        std::ffi::CStr::from_ptr(crate::empyrean_last_error())
            .to_string_lossy()
            .into_owned()
    }
}

/// The mixture-component status and the packed joint's own
/// `quality` / `functional` / `provenance` wire discriminants are append-only
/// and pinned to the engine's encodings — exactly what `orbit_uncertainty_to_c`
/// writes into the tagged side columns. This also keeps the header-only consts
/// referenced from Rust.
#[test]
fn wire_discriminant_consts_are_stable() {
    use crate::joint::wire_tags::{
        EMPYREAN_JOINT_ASSERTED_SOURCE_EXTERNAL_FILE, EMPYREAN_JOINT_ASSERTED_SOURCE_NONE,
        EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB, EMPYREAN_JOINT_FUNCTIONAL_DELAY_DOPPLER,
        EMPYREAN_JOINT_FUNCTIONAL_SKY, EMPYREAN_JOINT_FUNCTIONAL_STATE,
        EMPYREAN_JOINT_PROVENANCE_ASSERTED, EMPYREAN_JOINT_PROVENANCE_DERIVED,
        EMPYREAN_JOINT_QUALITY_INDEFINITE, EMPYREAN_JOINT_QUALITY_POSITIVE_DEFINITE,
        EMPYREAN_JOINT_QUALITY_POSITIVE_SEMIDEFINITE, EMPYREAN_JOINT_QUALITY_REPAIRED,
    };
    use crate::propagate::{
        EMPYREAN_MIXTURE_COMPONENT_STATUS_CURVATURE_REFUSED,
        EMPYREAN_MIXTURE_COMPONENT_STATUS_FAILED, EMPYREAN_MIXTURE_COMPONENT_STATUS_RESOLVED,
        EMPYREAN_MIXTURE_COMPONENT_STATUS_SKY_LINEARIZATION_REFUSED,
        EMPYREAN_MIXTURE_COMPONENT_STATUS_UNRESOLVED,
    };
    use empyrean_core::propagation::{ComponentStatus, Functional, Provenance, Quality};
    use empyrean_core::uncertainty::component_status_wire;

    // Mixture-component status, each const pinned against the engine's own
    // `component_status_wire` — the one authority for the discriminant AND the
    // side value a refusal carries. `SkyLinearizationRefused` (code 4, new in
    // villeneuve 1.26.0) is append-only, so it takes 4 even though the engine
    // enum orders it before `Unresolved`; collapsing its const onto any other
    // code, or routing its `extent` through the wrong side slot, fails here.
    assert_eq!(EMPYREAN_MIXTURE_COMPONENT_STATUS_RESOLVED, 0);
    assert_eq!(EMPYREAN_MIXTURE_COMPONENT_STATUS_CURVATURE_REFUSED, 1);
    assert_eq!(EMPYREAN_MIXTURE_COMPONENT_STATUS_UNRESOLVED, 2);
    assert_eq!(EMPYREAN_MIXTURE_COMPONENT_STATUS_FAILED, 3);
    assert_eq!(
        EMPYREAN_MIXTURE_COMPONENT_STATUS_SKY_LINEARIZATION_REFUSED,
        4
    );
    assert_eq!(
        component_status_wire(ComponentStatus::Resolved),
        (EMPYREAN_MIXTURE_COMPONENT_STATUS_RESOLVED, None)
    );
    assert_eq!(
        component_status_wire(ComponentStatus::CurvatureRefused { rho: 2.5 }),
        (
            EMPYREAN_MIXTURE_COMPONENT_STATUS_CURVATURE_REFUSED,
            Some(2.5)
        )
    );
    assert_eq!(
        component_status_wire(ComponentStatus::Unresolved),
        (EMPYREAN_MIXTURE_COMPONENT_STATUS_UNRESOLVED, None)
    );
    assert_eq!(
        component_status_wire(ComponentStatus::Failed),
        (EMPYREAN_MIXTURE_COMPONENT_STATUS_FAILED, None)
    );
    // The side value for code 4 is the on-sky angular extent (radians), a
    // DIFFERENT quantity from code 1's dimensionless curvature ρ, carried
    // through the same slot the C `EmpyreanMixtureComponent::rho` field holds.
    assert_eq!(
        component_status_wire(ComponentStatus::SkyLinearizationRefused { extent: 0.5 }),
        (
            EMPYREAN_MIXTURE_COMPONENT_STATUS_SKY_LINEARIZATION_REFUSED,
            Some(0.5)
        )
    );

    // The packed joint's `quality` — each arm pinned against the engine's own
    // `Quality::wire_discriminant`.
    assert_eq!(
        EMPYREAN_JOINT_QUALITY_POSITIVE_DEFINITE,
        Quality::PositiveDefinite.wire_discriminant()
    );
    assert_eq!(
        EMPYREAN_JOINT_QUALITY_POSITIVE_SEMIDEFINITE,
        Quality::PositiveSemidefinite {
            rank: 6,
            null_axes: 0
        }
        .wire_discriminant()
    );
    assert_eq!(
        EMPYREAN_JOINT_QUALITY_INDEFINITE,
        Quality::Indefinite { min_eig_corr: -1.0 }.wire_discriminant()
    );
    assert_eq!(
        EMPYREAN_JOINT_QUALITY_REPAIRED,
        Quality::Repaired { min_eig_corr: -1.0 }.wire_discriminant()
    );

    // `functional` / `provenance`: the unit arms are pinned against the engine.
    // `Sky` / `DelayDoppler` and `Asserted` carry a side companion the C wire
    // never names (so they are not constructible here) — they are pinned by
    // their append-only value, verified against villeneuve `packed_joint.rs`.
    assert_eq!(
        EMPYREAN_JOINT_FUNCTIONAL_STATE,
        Functional::State.wire_discriminant()
    );
    assert_eq!(EMPYREAN_JOINT_FUNCTIONAL_SKY, 1);
    assert_eq!(EMPYREAN_JOINT_FUNCTIONAL_DELAY_DOPPLER, 2);
    assert_eq!(
        EMPYREAN_JOINT_PROVENANCE_DERIVED,
        Provenance::Derived.wire_discriminant()
    );
    assert_eq!(EMPYREAN_JOINT_PROVENANCE_ASSERTED, 1);

    // `asserted_source`: the two source values pin the engine's `AssertedSource`
    // encoding (SBDB 0 / ExternalFile 1). Like `Asserted`, the source enum is
    // not core-re-exported, so it is pinned by its append-only value, verified
    // against villeneuve `packed_joint.rs`. `_NONE` is the ABI's own
    // not-available sentinel (not an engine discriminant) and must sit above
    // every append-only source value so it never collides.
    // SBDB 0 / ExternalFile 1, and the none sentinel at 255 — above every
    // append-only engine source discriminant so it never collides.
    assert_eq!(EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB, 0);
    assert_eq!(EMPYREAN_JOINT_ASSERTED_SOURCE_EXTERNAL_FILE, 1);
    assert_eq!(EMPYREAN_JOINT_ASSERTED_SOURCE_NONE, 255);
}

/// Ruling: the retired `CovarianceKind::ThirdOrder` (old wire discriminant 2)
/// is refused BY NAME on input — `RetiredDiscriminant`, FFI code
/// `EMPYREAN_PUSH_RETIRED_DISCRIMINANT` (-45) — at a direct push AND through
/// the CSV writer, identically. (`EMPYREAN_COVARIANCE_KIND_THIRD_ORDER` no
/// longer exists in the header; its absence is a compile/header fact.)
#[test]
fn retired_third_order_kind_is_refused_by_name() {
    let tri = diag_tri(6);
    let mut joint = joint_borrowing(&tri, 6);
    joint.kind = 2; // the retired ThirdOrder discriminant
    let orbit = orbit_with_joint(joint);
    assert_eq!(push_code_direct(&orbit), EMPYREAN_PUSH_RETIRED_DISCRIMINANT);
    assert_eq!(push_code_csv(&orbit), EMPYREAN_PUSH_RETIRED_DISCRIMINANT);
}

/// Ruling: a non-grav uncertainty rides ONLY the packed joint. A width-9
/// Cartesian orbit (6 state + 3 Marsden axes) pushed with a joint comes back
/// through `orbits_to_batch` with its `layout_present`, `width` and `tri`
/// bit-identical. Cartesian has no angular rows, so the degree<->radian
/// congruence is the identity and exact equality holds on this synthetic input.
#[test]
fn nongrav_uncertainty_round_trips_only_on_the_joint() {
    let tri = diag_tri(9);
    let joint = EmpyreanPackedJoint {
        layout_present: 0b111, // Marsden A1, A2, A3
        width: 9,
        ..joint_borrowing(&tri, 9)
    };
    let orbit = orbit_with_joint(joint);
    let (orbits, _ids, _objs) =
        crate::propagate::build_orbits_for_propagation(std::slice::from_ref(&orbit))
            .expect("the width-9 Cartesian Marsden joint is accepted");
    let mut batch = crate::io::orbits_to_batch(&orbits).expect("marshal back to a batch");
    let out = unsafe { &*batch.orbits };
    assert_eq!(out.covariance.width, 9, "width survives the round trip");
    assert_eq!(
        out.covariance.layout_present, 0b111,
        "the Marsden axes survive on the joint"
    );
    let out_tri = unsafe { std::slice::from_raw_parts(out.covariance.tri, out.covariance.tri_len) };
    assert_eq!(
        out_tri,
        &tri[..],
        "the packed triangle round-trips bit-identical (Cartesian: no angular scaling)"
    );
    unsafe { crate::io::empyrean_orbits_batch_free(&mut batch) };
}

/// Ruling: the joint carries its asserted source at the C layer. An asserted
/// SBDB-class joint (synthetic Cartesian, `provenance = Asserted` plus a real
/// source discriminant) round-trips push → read-back with BOTH `provenance`
/// and `asserted_source` bit-identical, and the live joint propagates one step.
/// Before this field existed, the identical joint was refused on input as
/// `UnknownProvenance`: the source was lost on output and the facade rightly
/// refuses an asserted provenance without a source.
#[test]
fn asserted_sbdb_joint_round_trips_its_source_and_propagates() {
    let tri = diag_tri(6);
    let orbit = orbit_with_joint(EmpyreanPackedJoint {
        provenance: EMPYREAN_JOINT_PROVENANCE_ASSERTED,
        asserted_source: EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB,
        ..joint_borrowing(&tri, 6)
    });

    // (a) push → engine → read-back: provenance AND its source survive.
    let (orbits, _ids, _objs) =
        crate::propagate::build_orbits_for_propagation(std::slice::from_ref(&orbit))
            .expect("the asserted SBDB joint is accepted on input (the source is carried)");
    let mut batch = crate::io::orbits_to_batch(&orbits).expect("marshal back to a batch");
    let out = unsafe { &*batch.orbits };
    assert_eq!(
        out.covariance.provenance, EMPYREAN_JOINT_PROVENANCE_ASSERTED,
        "the asserted provenance survives the round trip"
    );
    assert_eq!(
        out.covariance.asserted_source, EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB,
        "the SBDB source survives the round trip on the joint"
    );
    let out_tri = unsafe { std::slice::from_raw_parts(out.covariance.tri, out.covariance.tri_len) };
    assert_eq!(
        out_tri,
        &tri[..],
        "the packed triangle round-trips bit-identical (Cartesian: no angular scaling)"
    );
    unsafe { crate::io::empyrean_orbits_batch_free(&mut batch) };

    // (b) the live asserted joint propagates one first-order step (a refused or
    // dropped joint would not reach a covariance here).
    let Some(ctx) = crate::testing::context_or_skip(
        "asserted_sbdb_joint_round_trips_its_source_and_propagates",
    ) else {
        return;
    };
    let cfg = first_order_config();
    let times = [orbit.state.epoch_mjd_tdb, orbit.state.epoch_mjd_tdb + 1.0];
    let mut result: crate::propagate::EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
    let pc = unsafe {
        crate::propagate::empyrean_propagate(
            &ctx,
            &orbit as *const EmpyreanOrbit,
            1,
            times.as_ptr(),
            times.len(),
            &cfg,
            &mut result,
        )
    };
    assert_eq!(
        pc,
        0,
        "the asserted joint propagates one step: {}",
        last_error()
    );
    assert_eq!(result.num_states, times.len());
    let last = unsafe { &*result.states.add(times.len() - 1) };
    assert_eq!(
        last.has_covariance, 1,
        "the asserted joint drives a live uncertainty propagation"
    );
    unsafe { crate::propagate::empyrean_propagation_result_free(&mut result) };
}

/// Root-cause pin (pass 20): a Keplerian (angular) joint and the flat state
/// view it was derived from agree to the bit after BOTH travel the C degree →
/// radian boundary — the joint through [`scale_tri_angular`], the flat through
/// core's dense `scale_angular_covariance`. The two scalings must apply the
/// identical diagonal congruence \\(D \Sigma D^\top\\); an earlier
/// `scale_tri_angular` scaled each angular VARIANCE (the diagonal) by `factor`
/// once instead of `factor²`, so a re-fed SBDB-class angular orbit disagreed
/// with its own flat state block by exactly `180/π` ≈ 57× on the angular
/// variances and was refused by name (the −1 "disagree on the state block").
/// The bug was invisible to a pure deg → engine → deg round trip (the two
/// opposite scalings cancel) and only surfaces against the independent flat
/// path, as here.
///
/// A diagonally-dominant (so positive-definite) degree-basis Keplerian state
/// block exercises every congruence case at once: the angular variances
/// (diagonal, `factor²`), angular×angular crosses (`factor²`), angular×linear
/// crosses (`factor` once), and linear×linear (untouched). Pushing the orbit
/// carrying BOTH the joint and the identical flat view must be accepted; the
/// paired mutation — scaling the flat's `i` variance by `1 + 1e-6` — must trip
/// the −1 refusal, proving the agreement gate is live.
#[test]
fn keplerian_angular_joint_refeed_agrees_with_its_flat_state_block() {
    // Degree-basis Keplerian state covariance (a, e | i, Ω, ω, M), symmetric
    // and diagonally dominant. Angular rows are 2,3,4,5.
    let cov_deg: [[f64; 6]; 6] = [
        [1.0e-10, 1.0e-11, 1.0e-11, 0.0, 0.0, 0.0],
        [1.0e-11, 1.0e-9, 1.0e-11, 0.0, 0.0, 0.0],
        [1.0e-11, 1.0e-11, 1.0e-8, 1.0e-10, 0.0, 0.0],
        [0.0, 0.0, 1.0e-10, 2.0e-8, 1.0e-10, 0.0],
        [0.0, 0.0, 0.0, 1.0e-10, 3.0e-8, 1.0e-10],
        [0.0, 0.0, 0.0, 0.0, 1.0e-10, 4.0e-8],
    ];
    // The joint carries the SAME 6×6 as a state-only (width-6) degree triangle.
    let mut tri = vec![0.0f64; 21];
    for i in 0..6 {
        for j in 0..=i {
            tri[i * (i + 1) / 2 + j] = cov_deg[i][j];
        }
    }
    let keplerian_orbit = |flat: [[f64; 6]; 6]| {
        let mut orbit: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        orbit.state = CoordinateState {
            epoch_mjd_tdb: 59000.0,
            elements: [1.0, 0.1, 10.0, 30.0, 40.0, 50.0], // a, e, i, Ω, ω, M (deg)
            covariance: flat,
            has_covariance: 1,
            representation: 1, // Keplerian
            frame: 0,          // ICRF
            origin: 10,        // Sun
        };
        orbit.phot_system = -1;
        orbit.h_mag = f64::NAN;
        orbit.non_grav_dt = f64::NAN;
        orbit.covariance = joint_borrowing(&tri, 6);
        orbit
    };

    // The flat view equals the joint's state block (both derived from one
    // object): after the identical congruence they agree to the bit and the
    // push is accepted.
    let agree = keplerian_orbit(cov_deg);
    crate::propagate::build_orbits_for_propagation(std::slice::from_ref(&agree)).unwrap_or_else(
        |e| {
            panic!(
                "the angular joint and its flat state block must agree after the \
                 degree→radian boundary: {} ({:?})",
                e.message, e.code
            )
        },
    );

    // Mutation: perturb only the flat `i` variance by 1 + 1e-6 — now the flat
    // and the joint contradict, and the gate refuses by name with code −1.
    let mut mutated = cov_deg;
    mutated[2][2] *= 1.0 + 1.0e-6;
    let disagree = keplerian_orbit(mutated);
    let err = crate::propagate::build_orbits_for_propagation(std::slice::from_ref(&disagree))
        .expect_err("a flat/joint state-block disagreement must be refused, not accepted");
    assert_eq!(
        err.code.unwrap_or(-1),
        -1,
        "a flat/joint state-block disagreement is refused by name (−1)"
    );
    assert!(
        err.message.contains("disagree on the state block"),
        "the refusal names the flat/joint state-block disagreement: {}",
        err.message
    );
}

/// The typed `file_in_op` boundary: one malformed wire row per reachable
/// `UncertaintyWireError` variant returns the SAME per-variant code (-40..-49)
/// at a direct push AND through `empyrean_orbits_write_csv`. `UnknownAssertedSource`
/// is now reachable from the C wire — an asserted joint carrying the `_NONE`
/// sentinel source is handed to the facade as `Some(255)` and refused by name.
/// (The remaining unexpressible variants — `EpochMismatch`, since the row epoch
/// is the basis epoch, and `EngineRefused`, an unreachable catch-all — are
/// covered by the engine facade's own tests.)
#[test]
fn push_refusal_codes_match_across_direct_and_csv() {
    // (label, mutate the width-6 joint, expected code)
    let tri6 = diag_tri(6);
    let tri9 = diag_tri(9);
    let short = diag_tri(6)[..20].to_vec(); // one short of 21 for width 6

    let cases: Vec<(&str, EmpyreanPackedJoint, i32)> = vec![
        (
            "layout_invalid: considered not a subset of present",
            EmpyreanPackedJoint {
                layout_considered: 0b1,
                ..joint_borrowing(&tri6, 6)
            },
            EMPYREAN_PUSH_LAYOUT_INVALID,
        ),
        (
            "triangle_length: tri shorter than the declared width implies",
            EmpyreanPackedJoint {
                tri: short.as_ptr(),
                tri_len: short.len(),
                ..joint_borrowing(&tri6, 6)
            },
            EMPYREAN_PUSH_TRIANGLE_LENGTH,
        ),
        (
            "width_mismatch: declared width disagrees with the layout",
            EmpyreanPackedJoint {
                width: 9,
                tri: tri9.as_ptr(),
                tri_len: tri9.len(),
                ..joint_borrowing(&tri9, 9) // layout_present 0 ⇒ implies width 6
            },
            EMPYREAN_PUSH_WIDTH_MISMATCH,
        ),
        (
            "unknown_functional",
            EmpyreanPackedJoint {
                functional: 99,
                ..joint_borrowing(&tri6, 6)
            },
            EMPYREAN_PUSH_UNKNOWN_FUNCTIONAL,
        ),
        (
            "unknown_provenance",
            EmpyreanPackedJoint {
                provenance: 99,
                ..joint_borrowing(&tri6, 6)
            },
            EMPYREAN_PUSH_UNKNOWN_PROVENANCE,
        ),
        (
            "unsupported_input_kind: a live non-linear kind",
            EmpyreanPackedJoint {
                kind: 1, // SecondOrder — a live non-linear kind
                ..joint_borrowing(&tri6, 6)
            },
            EMPYREAN_PUSH_UNSUPPORTED_INPUT_KIND,
        ),
        (
            "retired_discriminant: the old third-order kind",
            EmpyreanPackedJoint {
                kind: 2,
                ..joint_borrowing(&tri6, 6)
            },
            EMPYREAN_PUSH_RETIRED_DISCRIMINANT,
        ),
        (
            "unknown_asserted_source: an asserted joint carrying the none sentinel",
            EmpyreanPackedJoint {
                provenance: EMPYREAN_JOINT_PROVENANCE_ASSERTED,
                asserted_source: EMPYREAN_JOINT_ASSERTED_SOURCE_NONE,
                ..joint_borrowing(&tri6, 6)
            },
            EMPYREAN_PUSH_UNKNOWN_ASSERTED_SOURCE,
        ),
    ];

    for (label, joint, expected) in cases {
        let orbit = orbit_with_joint(joint);
        let direct = push_code_direct(&orbit);
        let csv = push_code_csv(&orbit);
        assert_eq!(direct, expected, "{label}: direct push code");
        assert_eq!(
            csv, expected,
            "{label}: the file-write path returns the same per-variant code, not a collapse"
        );
    }
}

/// `empyrean_transform_coordinates_single` transforms a bare state and refuses
/// a covariance-bearing one BY NAME with the dedicated code (-50), its message
/// pointing at `empyrean_transform_orbits`. Not a silent drop.
#[test]
fn transform_coordinates_refuses_a_covariance_bearing_state() {
    let Some(ctx) =
        crate::testing::context_or_skip("transform_coordinates_refuses_a_covariance_bearing_state")
    else {
        return;
    };
    let input = CoordinateState {
        epoch_mjd_tdb: 59000.0,
        elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
        covariance: {
            let mut c = [[0.0; 6]; 6];
            for (i, row) in c.iter_mut().enumerate() {
                row[i] = 1.0e-8 * (i as f64 + 1.0);
            }
            c
        },
        has_covariance: 1,
        representation: 0,
        frame: 0,
        origin: 10,
    };
    let mut out: CoordinateState = unsafe { std::mem::zeroed() };
    // Keplerian / EclipticJ2000 / Sun — a real basis change.
    let code = unsafe {
        crate::transform::empyrean_transform_coordinates_single(&ctx, &input, 1, 1, 10, &mut out)
    };
    assert_eq!(
        code,
        crate::transform::EMPYREAN_TRANSFORM_COVARIANCE_UNSUPPORTED,
        "a covariance-bearing state is refused, not transformed"
    );
    let msg = last_error();
    assert!(
        msg.contains("empyrean_transform_orbits"),
        "the refusal names the orbit-level entry point: {msg}"
    );
}

/// `empyrean_transform_orbits` transports the joint: a synthetic width-6
/// Cartesian orbit with a diagonal joint, transformed Cartesian -> Keplerian ->
/// Cartesian, returns its state and its joint's state block to a tight relative
/// band (the joint is re-expressed under the element Jacobian, not dropped).
#[test]
fn transform_orbits_round_trips_a_joint() {
    let Some(ctx) = crate::testing::context_or_skip("transform_orbits_round_trips_a_joint") else {
        return;
    };
    let tri = diag_tri(6);
    let joint = joint_borrowing(&tri, 6);
    let orbit = orbit_with_joint(joint);

    // Cartesian (0) -> Keplerian (1), ICRF (0), Sun (10).
    let mut kep: EmpyreanOrbitBatch = EmpyreanOrbitBatch::empty();
    let c1 =
        unsafe { crate::transform::empyrean_transform_orbits(&ctx, &orbit, 1, 1, 0, 10, &mut kep) };
    assert_eq!(c1, 0, "forward transform succeeds: {}", last_error());
    assert_eq!(kep.num_orbits, 1);
    assert_eq!(
        unsafe { &*kep.orbits }.covariance.width,
        6,
        "the joint is transported, not dropped"
    );

    // Keplerian -> Cartesian back.
    let mut cart: EmpyreanOrbitBatch = EmpyreanOrbitBatch::empty();
    let c2 = unsafe {
        crate::transform::empyrean_transform_orbits(&ctx, kep.orbits, 1, 0, 0, 10, &mut cart)
    };
    assert_eq!(c2, 0, "inverse transform succeeds: {}", last_error());

    let back = unsafe { &*cart.orbits };
    for i in 0..6 {
        let (got, want) = (back.state.elements[i], orbit.state.elements[i]);
        assert!(
            (got - want).abs() <= 1.0e-9 * want.abs().max(1.0e-12) + 1.0e-12,
            "state element {i} round-trips: got {got}, want {want}"
        );
    }
    let back_tri =
        unsafe { std::slice::from_raw_parts(back.covariance.tri, back.covariance.tri_len) };
    assert_eq!(
        back.covariance.width, 6,
        "the joint survives the round trip"
    );
    for (k, (&g, &w)) in back_tri.iter().zip(tri.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1.0e-9 * w.abs().max(1.0e-12) + 1.0e-18,
            "joint triangle entry {k} round-trips: got {g}, want {w}"
        );
    }

    unsafe {
        crate::io::empyrean_orbits_batch_free(&mut kep);
        crate::io::empyrean_orbits_batch_free(&mut cart);
    }
}

/// A first-order propagation config over the Standard force model — the
/// minimal config that transports a joint (mirrors the propagate module's own
/// fixture; kept local so the two test modules do not couple).
fn first_order_config() -> crate::propagate::EmpyreanPropagationConfig {
    // SAFETY: `#[repr(C)]` with scalar fields only; the all-zero pattern is the
    // documented default for every field except the NaN step sentinels set below.
    let mut cfg: crate::propagate::EmpyreanPropagationConfig = unsafe { std::mem::zeroed() };
    cfg.force_model = 2; // Standard
    cfg.uncertainty_method.tag = crate::propagate::EMPYREAN_UNCERTAINTY_FIRST;
    cfg.advanced.dt_initial = f64::NAN;
    cfg.advanced.dt_min = f64::NAN;
    cfg.advanced.cache_integrator_steps = 1;
    cfg
}

/// A single-orbit write batch borrowing `orbit` and `id` (both must outlive
/// the call) — the shape the C orbit writers consume.
fn write_batch(orbit: &EmpyreanOrbit, id: &mut *mut std::ffi::c_char) -> EmpyreanOrbitBatch {
    EmpyreanOrbitBatch {
        orbits: orbit as *const EmpyreanOrbit as *mut EmpyreanOrbit,
        orbit_ids: id as *mut *mut std::ffi::c_char,
        object_ids: std::ptr::null_mut(),
        num_orbits: 1,
    }
}

/// FILE-seam successor to the removed `io::orbit_file_joint_tests`: the ATOMIC
/// packed joint survives a Parquet orbit round trip and comes back live.
///
/// A width-9 Cartesian orbit (6 state + 3 Marsden axes) is written to Parquet,
/// read back through the C surface, and its `layout_present` / `width` / `tri`
/// come back bit-identical. Cartesian has no angular state rows, so the
/// degree <-> radian congruence the ABI applies to angular state is the
/// identity, and Parquet stores each `f64` in IEEE-754 binary with no
/// re-formatting — so exact equality (`to_bits`) is admissible on this
/// synthetic input (and only such synthetic Cartesian inputs). The read-back
/// orbit then propagates one step under a first-order uncertainty method and
/// delivers a covariance: the joint returned live, not as inert numbers.
#[test]
fn orbit_parquet_round_trips_the_atomic_joint_live() {
    let Some(ctx) =
        crate::testing::context_or_skip("orbit_parquet_round_trips_the_atomic_joint_live")
    else {
        return;
    };
    let tri = diag_tri(9);
    let mut orbit = orbit_with_joint(EmpyreanPackedJoint {
        layout_present: 0b111, // Marsden A1, A2, A3
        width: 9,
        ..joint_borrowing(&tri, 9)
    });
    // The Marsden coefficient point values the layout opens (the joint carries
    // their prior covariance; the orbit carries the values).
    orbit.a1 = 1.0e-10;
    orbit.a2 = 1.0e-11;
    orbit.a3 = 0.0;

    let id = std::ffi::CString::new("seam0").unwrap();
    let mut id_ptr = id.as_ptr() as *mut std::ffi::c_char;
    let batch = write_batch(&orbit, &mut id_ptr);
    let path = std::env::temp_dir().join("empyrean_pass15_seam.parquet");
    let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

    let wc = unsafe { crate::io::empyrean_orbits_write_parquet(cpath.as_ptr(), &batch) };
    assert_eq!(
        wc,
        0,
        "parquet write of the width-9 joint: {}",
        last_error()
    );

    let mut read_back = EmpyreanOrbitBatch::empty();
    let rc = unsafe { crate::io::empyrean_orbits_read_parquet(cpath.as_ptr(), &mut read_back) };
    assert_eq!(rc, 0, "parquet read: {}", last_error());
    assert_eq!(read_back.num_orbits, 1);

    let got = unsafe { &*read_back.orbits };
    assert_eq!(
        got.covariance.width, 9,
        "the joint width survives the file seam"
    );
    assert_eq!(
        got.covariance.layout_present, 0b111,
        "the Marsden axes survive on the joint"
    );
    let got_tri = unsafe { std::slice::from_raw_parts(got.covariance.tri, got.covariance.tri_len) };
    assert_eq!(
        got_tri,
        &tri[..],
        "the packed triangle round-trips bit-identical through Parquet"
    );

    // The read-back joint is LIVE: one first-order propagation step delivers a
    // covariance (a dropped or zeroed joint would not).
    let cfg = first_order_config();
    let times = [got.state.epoch_mjd_tdb, got.state.epoch_mjd_tdb + 1.0];
    let mut result: crate::propagate::EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
    let pc = unsafe {
        crate::propagate::empyrean_propagate(
            &ctx,
            read_back.orbits,
            1,
            times.as_ptr(),
            times.len(),
            &cfg,
            &mut result,
        )
    };
    assert_eq!(
        pc,
        0,
        "the read-back orbit propagates one step: {}",
        last_error()
    );
    assert_eq!(result.num_states, times.len());
    let last = unsafe { &*result.states.add(times.len() - 1) };
    assert_eq!(
        last.has_covariance, 1,
        "the read-back joint drives a live uncertainty propagation"
    );

    unsafe {
        crate::propagate::empyrean_propagation_result_free(&mut result);
        crate::io::empyrean_orbits_batch_free(&mut read_back);
    }
    let _ = std::fs::remove_file(&path);
}

/// The CSV orbit schema is state-only by design: a width-6 state joint
/// round-trips through it bit-identical, while a wider (parameter-carrying)
/// joint is REFUSED BY NAME on write rather than written short — Parquet is the
/// format that carries the atomic wide joint (above). This pins the loud
/// refusal: CSV has no null, so it cannot carry a parameter cross-covariance,
/// and there is no hidden fallback to a bare 6×6.
#[test]
fn orbit_csv_is_state_only_and_refuses_a_wider_joint_by_name() {
    // (a) a state-only width-6 joint round-trips bit-identical.
    let tri6 = diag_tri(6);
    let orbit6 = orbit_with_joint(joint_borrowing(&tri6, 6));
    let id6 = std::ffi::CString::new("csv6").unwrap();
    let mut id6_ptr = id6.as_ptr() as *mut std::ffi::c_char;
    let batch6 = write_batch(&orbit6, &mut id6_ptr);
    let path6 = std::env::temp_dir().join("empyrean_pass15_seam6.csv");
    let cpath6 = std::ffi::CString::new(path6.to_str().unwrap()).unwrap();

    let wc6 = unsafe { crate::io::empyrean_orbits_write_csv(cpath6.as_ptr(), &batch6) };
    assert_eq!(wc6, 0, "CSV write of a state-only joint: {}", last_error());
    let mut read6 = EmpyreanOrbitBatch::empty();
    let rc6 = unsafe { crate::io::empyrean_orbits_read_csv(cpath6.as_ptr(), &mut read6) };
    assert_eq!(rc6, 0, "CSV read: {}", last_error());
    let got6 = unsafe { &*read6.orbits };
    assert_eq!(got6.covariance.width, 6, "CSV carries the state 6×6");
    let got6_tri =
        unsafe { std::slice::from_raw_parts(got6.covariance.tri, got6.covariance.tri_len) };
    assert_eq!(
        got6_tri,
        &tri6[..],
        "the state triangle round-trips bit-identical through CSV (Cartesian: no scaling)"
    );
    unsafe { crate::io::empyrean_orbits_batch_free(&mut read6) };
    let _ = std::fs::remove_file(&path6);

    // (b) a width-9 Marsden joint is refused BY NAME on CSV write.
    let tri9 = diag_tri(9);
    let mut orbit9 = orbit_with_joint(EmpyreanPackedJoint {
        layout_present: 0b111,
        width: 9,
        ..joint_borrowing(&tri9, 9)
    });
    orbit9.a1 = 1.0e-10;
    orbit9.a2 = 1.0e-11;
    orbit9.a3 = 0.0;
    let id9 = std::ffi::CString::new("csv9").unwrap();
    let mut id9_ptr = id9.as_ptr() as *mut std::ffi::c_char;
    let batch9 = write_batch(&orbit9, &mut id9_ptr);
    let path9 = std::env::temp_dir().join("empyrean_pass15_seam9.csv");
    let cpath9 = std::ffi::CString::new(path9.to_str().unwrap()).unwrap();

    let wc9 = unsafe { crate::io::empyrean_orbits_write_csv(cpath9.as_ptr(), &batch9) };
    assert_ne!(wc9, 0, "the width-9 Marsden joint is refused on CSV write");
    let msg = last_error();
    assert!(
        msg.contains("ParameterCovarianceNotWritable") || msg.contains("cross-covariance"),
        "the CSV refusal names the parameter covariance: {msg}"
    );
    assert!(!path9.exists(), "the refused CSV file is never created");
    let _ = std::fs::remove_file(&path9);
}

/// Pass 22 (section A): under a SAMPLED method (Monte Carlo / sigma point) the
/// per-state output joint is resolved at the row's OWN kind from the DELIVERED
/// state row — the sample second moment the engine wrote once at assembly —
/// never from the sensitivity chain, which under a sampled method carries only
/// tag-carrier placeholders and refuses the sample moment BY NAME ("it lives on
/// the propagated states"). That refusal used to abort the whole
/// `empyrean_propagate` call with −5, because the per-row joint fill called the
/// chain query for EVERY row; the kind-dispatched fill now completes and
/// delivers a per-state joint tagged with the run's own kind.
///
/// "Same object, to the bit": the per-state `orbit_cov` is marshaled from
/// `states.covariance(i)`, and the flat `EmpyreanPropagatedState.covariance`
/// 6×6 beside it is that SAME row's `state_block()` — one publication site read
/// two ways. On a Cartesian output there is no angular degree <-> radian
/// congruence, so the packed triangle's lower entries equal the flat matrix
/// bit-for-bit.
#[test]
fn sampled_per_state_joint_reads_the_delivered_sample_moment() {
    use crate::propagate::{
        EMPYREAN_COVARIANCE_KIND_MONTE_CARLO, EMPYREAN_COVARIANCE_KIND_SIGMA_POINT,
        EMPYREAN_UNCERTAINTY_MONTE_CARLO, EMPYREAN_UNCERTAINTY_SIGMA_POINT,
    };
    let Some(ctx) = crate::testing::context_or_skip(
        "sampled_per_state_joint_reads_the_delivered_sample_moment",
    ) else {
        return;
    };

    // A Cartesian orbit whose only uncertainty is a tight width-6 state joint
    // (the regime the Python sampling contracts exercise). Tight + Cartesian ⇒
    // the sample moment is well-conditioned at a modest ensemble and the packed
    // triangle shares the flat field's basis exactly.
    let mut tri = vec![0.0f64; 6 * 7 / 2];
    for (i, d) in [1e-12, 1e-12, 1e-12, 1e-16, 1e-16, 1e-16]
        .iter()
        .enumerate()
    {
        tri[i * (i + 1) / 2 + i] = *d;
    }
    let orbit = orbit_with_joint(joint_borrowing(&tri, 6));

    // One sampled run: the per-state joint carries that method's own kind with a
    // state block bit-identical to the delivered flat covariance.
    let run = |tag: u8, kind_u8: u8, label: &str| {
        let mut cfg = first_order_config();
        cfg.uncertainty_method.tag = tag;
        cfg.uncertainty_method.sp_n_sigma = 1.0; // canonical sigma-point
        cfg.uncertainty_method.sp_samples_per_plane = 8; // construction (parameter-free)
        cfg.uncertainty_method.mc_n_samples = 64; // > the 7-draw rank floor
        cfg.uncertainty_method.mc_seed_some = 1;
        cfg.uncertainty_method.mc_seed = 7;
        let times = [orbit.state.epoch_mjd_tdb, orbit.state.epoch_mjd_tdb + 365.0];
        let mut result: crate::propagate::EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
        let pc = unsafe {
            crate::propagate::empyrean_propagate(
                &ctx,
                &orbit as *const EmpyreanOrbit,
                1,
                times.as_ptr(),
                times.len(),
                &cfg,
                &mut result,
            )
        };
        assert_eq!(
            pc,
            0,
            "{label}: a sampled propagate must COMPLETE (the chain refusal no longer \
             aborts the call): {}",
            last_error()
        );
        assert_eq!(result.num_states, times.len());

        let mut saw_covariance = false;
        for i in 0..result.num_states {
            let st = unsafe { &*result.states.add(i) };
            // The resolved kind tags the sampled method on every row, covariance
            // or not (an incomplete cloud clears the matrix but keeps the tag).
            assert_eq!(
                st.resolved_kind, kind_u8,
                "{label}: row {i} tags its own sampled kind"
            );
            if st.has_covariance == 0 {
                // Honest absence: a row below the rank floor / short cloud carries
                // no second moment, so the joint is legitimately empty — never −5.
                assert!(
                    st.orbit_cov.tri.is_null() && st.orbit_cov.tri_len == 0,
                    "{label}: a covariance-less sampled row carries an empty joint"
                );
                continue;
            }
            saw_covariance = true;
            assert_eq!(
                st.orbit_cov.kind, kind_u8,
                "{label}: the per-state joint is tagged with the sampled kind, \
                 not downgraded to linear"
            );
            assert_eq!(st.orbit_cov.width, 6, "{label}: a state-only sampled joint");
            let out_tri =
                unsafe { std::slice::from_raw_parts(st.orbit_cov.tri, st.orbit_cov.tri_len) };
            // The joint's state block IS the delivered row's flat covariance — one
            // object, two reads. Compare the packed lower triangle to the flat 6×6
            // bit-for-bit (Cartesian: no angular congruence).
            for r in 0..6 {
                for c in 0..=r {
                    let packed = out_tri[r * (r + 1) / 2 + c];
                    assert_eq!(
                        packed.to_bits(),
                        st.covariance[r][c].to_bits(),
                        "{label}: joint[{r}][{c}] must equal the flat sample covariance \
                         bit-for-bit (same object: states.covariance(i))"
                    );
                }
            }
        }
        assert!(
            saw_covariance,
            "{label}: at least one output row above the rank floor must deliver a \
             sampled per-state joint (an all-empty result would hide the −5 the \
             dispatch replaced)"
        );
        unsafe { crate::propagate::empyrean_propagation_result_free(&mut result) };
    };

    run(
        EMPYREAN_UNCERTAINTY_MONTE_CARLO,
        EMPYREAN_COVARIANCE_KIND_MONTE_CARLO,
        "monte_carlo",
    );
    run(
        EMPYREAN_UNCERTAINTY_SIGMA_POINT,
        EMPYREAN_COVARIANCE_KIND_SIGMA_POINT,
        "sigma_point",
    );
}
