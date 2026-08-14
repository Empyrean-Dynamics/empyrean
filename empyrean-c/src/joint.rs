//! The joint solved-parameter covariance at the C ABI — identity-tagged
//! cross terms in, posterior cross terms out.
//!
//! # What crosses here, and what does not
//!
//! A fitted orbit's uncertainty is one \\((6+P) \times (6+P)\\) matrix over
//! the state and every solved parameter. Until this module the C ABI could
//! carry only its diagonal blocks — the \\(6 \times 6\\) state covariance,
//! the Marsden \\(3 \times 3\\), a DT variance, an AMRAT variance, a
//! per-segment thrust \\(3 \times 3\\) — so a caller who fitted an orbit and
//! re-propagated it handed the engine a block-diagonal covariance and the
//! engine could not tell. The correlations are not a refinement on top of
//! the variances; they are what makes the joint the matrix the fit actually
//! computed.
//!
//! The joint is supplied in **four non-overlapping homes**, and one entry
//! belongs to exactly one of them:
//!
//! | Block | Home |
//! |---|---|
//! | state ↔ state | [`CoordinateState::covariance`](crate::CoordinateState::covariance) |
//! | \\(A_i \leftrightarrow A_j\\) | [`EmpyreanOrbit::non_grav_covariance`](crate::propagate::EmpyreanOrbit::non_grav_covariance) |
//! | state ↔ \\(A_i\\) | [`CoordinateState::non_grav_cross`](crate::CoordinateState::non_grav_cross) |
//! | \\(\Delta v_i \leftrightarrow \Delta v_j\\), same segment | `EmpyreanOrbit::correction_covariances[i]` |
//! | everything else | [`EmpyreanStateParamCross`] / [`EmpyreanParamPairCross`] |
//!
//! "Everything else" is state↔DT, state↔AMRAT, state↔\\(\Delta v\\), and
//! every mixed pair (DT↔\\(A_2\\), \\(A_1\\)↔AMRAT, AMRAT↔\\(\Delta v\\),
//! segment \\(i\\)↔segment \\(j\\)). The engine refuses a term with two
//! homes rather than merging or preferring one, so mirroring the partition
//! in the ABI's *shape* is what makes the common mistake unrepresentable
//! instead of merely refused.
//!
//! # Why the entries carry an identity and not a column index
//!
//! Which wide column a parameter occupies depends on which *other*
//! parameters the orbit declares: adding an SRP AMRAT to an orbit shifts
//! the thrust columns by one. A slot index recorded at one call site is
//! therefore wrong at the next, and the failure is silent — every number
//! finite, every gate passed, the fitted state↔AMRAT correlation attached
//! to \\(A_1\\). [`EmpyreanParamColumn`] names the parameter itself, and
//! the engine resolves the placement at assembly time.
//!
//! One consequence worth stating: because the tag is an identity, **the
//! entry order of the cross arrays is not contract**. Callers may emit in
//! any order.
//!
//! # A thrust `segment` is a DECLARED index, not a solved one
//!
//! `EmpyreanParamColumn::segment` indexes
//! `EmpyreanOrbit::correction_covariances` — the segments the orbit
//! **declares** — because that is the space the engine derives its
//! thrust columns in when the orbit is re-fed. Tagging with the solved
//! index instead would place a burn's cross terms on a different burn's
//! column the moment any segment is considered or fixed, which after
//! the tri-state is a routine case rather than an exotic one.
//!
//! Stated here explicitly because the engine enum this mirrors
//! (`ParamColumn::Thrust`) documents its own `segment` as "which solved
//! correction segment", which is stale: the estimator that populates
//! these carriers tags them by declared index, and the layout resolves
//! them by declared index. The C contract follows the code, not that
//! doc string. Reported upstream.
//!
//! # Units
//!
//! The state-side rows of every cross column are in the coordinate's own
//! element order, representation, frame **and angular unit** — the same
//! basis as the \\(6 \times 6\\) they border. For a Cometary orbit that
//! means the three angular rows are in DEGREES, matching
//! [`CoordinateState::covariance`](crate::CoordinateState::covariance)
//! exactly. [`push_orbit_with_joint`] converts the coordinate and its
//! carrier in one step through the engine's unit-aware entry point, so
//! there is no window in which a half-converted pair exists.

use empyrean_core::coordinates::{AU, Coordinates, Degrees, ExtendedCovariance};
use empyrean_core::orbits::Orbits;
use empyrean_core::propagation::{ParamColumn, WideCross};

use crate::propagate::EmpyreanOrbit;

// ── Parameter-column identity ─────────────────────────────────────
// Mirrors `empyrean_core::propagation::ParamColumn` one-for-one.

/// Marsden non-gravitational coefficient \\(A_{i+1}\\); `index` selects
/// which of \\(A_1, A_2, A_3\\).
pub const EMPYREAN_PARAM_COLUMN_MARSDEN: i32 = 0;
/// The Marsden non-grav time delay \\(\Delta T\\).
pub const EMPYREAN_PARAM_COLUMN_DT: i32 = 1;
/// The SRP area-to-mass ratio.
pub const EMPYREAN_PARAM_COLUMN_AMRAT: i32 = 2;
/// One component of one thrust \\(\Delta v\\) correction segment;
/// `segment` and `component` select which.
pub const EMPYREAN_PARAM_COLUMN_THRUST: i32 = 3;

/// The identity of one solved-parameter column, independent of the wide
/// slot it happens to occupy on any particular orbit.
///
/// The fields a given `kind` does not read must be **zero**. A non-zero
/// value there is refused rather than ignored, so a caller who sets
/// `segment` on a `DT` column learns immediately instead of silently
/// getting a different column than they named.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct EmpyreanParamColumn {
    /// One of the `EMPYREAN_PARAM_COLUMN_*` tags. Any other value is a
    /// loud argument error.
    pub kind: i32,
    /// MARSDEN: 0, 1 or 2, selecting \\(A_1\\), \\(A_2\\) or \\(A_3\\).
    /// Must be 0 for every other kind.
    pub index: u32,
    /// THRUST: index into `EmpyreanOrbit::correction_covariances`. Must
    /// be 0 for every other kind.
    pub segment: u32,
    /// THRUST: 0, 1 or 2, selecting x, y or z of that segment's
    /// \\(\Delta v\\). Must be 0 for every other kind.
    pub component: u32,
}

/// One state-to-parameter cross column: the 6-vector of covariances
/// between the six state elements and one solved parameter.
///
/// Rows are in the coordinate's own element order, representation, frame
/// and angular unit — the same basis AND THE SAME UNITS as the
/// \\(6 \times 6\\) in `state.covariance` that this borders. Cartesian:
/// \\((x, y, z, v_x, v_y, v_z)\\) in AU and AU/day. Cometary:
/// \\((q, e, i, \Omega, \omega, t_p)\\) with the three angular rows in
/// DEGREES. Supplying radians here and degrees there is a silent factor
/// of \\(180/\pi\\) on those rows.
///
/// The engine rotates these with the state when the orbit is transformed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EmpyreanStateParamCross {
    /// Which parameter this column covaries the state with.
    pub column: EmpyreanParamColumn,
    /// \\(\mathrm{Cov}(\text{element } r, \text{parameter})\\), \\(r\\) in
    /// \\(0..6\\).
    pub values: [f64; 6],
}

/// One parameter-to-parameter cross term.
///
/// Symmetric: \\((a, b)\\) and \\((b, a)\\) are the same entry, and
/// supplying both is a loud error rather than a merge or a
/// last-one-wins.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EmpyreanParamPairCross {
    /// One end of the pair.
    pub a: EmpyreanParamColumn,
    /// The other end. Must not name the same parameter as `a`.
    pub b: EmpyreanParamColumn,
    /// \\(\mathrm{Cov}(a, b)\\).
    pub value: f64,
}

/// A joint cross-covariance the library HANDS BACK — the half of a
/// covariance that is not a diagonal block.
///
/// # Why this exists as output at all
///
/// A propagated or fitted state's uncertainty is one
/// \\((6+P) \times (6+P)\\) matrix. Until this struct the C ABI reported
/// only its diagonal blocks, so a caller chaining one leg into the next
/// necessarily handed the engine a block-diagonal covariance — a
/// physically unmotivated claim that the state and \\(A_2\\) are
/// independent, when a single fit or a single propagation produced both.
/// The engine's own propagated border is non-zero **even from a
/// block-diagonal input**, because propagation itself generates the
/// correlation: that is precisely why a second leg must not be handed a
/// block-diagonal covariance.
///
/// # One shape, two producers
///
/// [`EmpyreanODResult::orbit`](crate::od::EmpyreanODResult::orbit) and
/// every row of a propagation result carry this same struct, with the
/// same field names and the same ownership. That is what makes leg
/// chaining uniform: whatever produced the state, its joint is read the
/// same way and re-fed the same way.
///
/// Each block is read off the object the engine already made coherent —
/// the fitted orbit, or the propagated state row — never re-derived from
/// a slot-tagged matrix. That matters: a slot-tagged covariance indexes
/// the *solved* layout, while a re-fed orbit's layout is derived from
/// what it *declares*, and the two disagree whenever the fit did not
/// solve every axis the orbit declares. Copying columns between them
/// would attach one parameter's correlations to another with every
/// number finite and every gate passed.
///
/// # Basis
///
/// The state-side rows are in the same basis **and the same units** as
/// the \\(6 \times 6\\) sitting beside this struct — Cartesian AU and
/// AU/day on a propagated state, and the fitted result's
/// `covariance_representation` on an OD result.
///
/// # Re-feeding
///
/// Assign across to the input orbit field by field — the result nests
/// what the orbit flattens:
///
/// ```c
/// orbit.state.has_non_grav_cross = st->orbit_cov.has_non_grav_cross;
/// memcpy(orbit.state.non_grav_cross,
///        st->orbit_cov.non_grav_cross, sizeof(double) * 18);
/// orbit.state_param_cross        = st->orbit_cov.state_param_cross;
/// orbit.n_state_param_cross      = st->orbit_cov.n_state_param_cross;
/// orbit.param_pair_cross         = st->orbit_cov.param_pair_cross;
/// orbit.n_param_pair_cross       = st->orbit_cov.n_param_pair_cross;
///
/// // ...and the diagonal blocks those crosses are conditioned on:
/// orbit.has_non_grav_covariance  = res->non_grav.has_covariance;
/// memcpy(orbit.non_grav_covariance,
///        res->non_grav.covariance, sizeof(double) * 9);
/// orbit.non_grav_dt_variance     = res->non_grav.dt_variance;
/// orbit.has_srp                  = res->has_srp;
/// orbit.srp_amrat_variance       = res->srp.amrat_variance;
/// ```
///
/// # Ownership
///
/// The two carrier arrays are **library-owned** and released with the
/// parent result — the opposite of the identically-named fields on
/// [`EmpyreanOrbit`], which are caller-owned and borrowed for the call.
/// So the pointer assignments above are valid only while the result
/// outlives the orbit; **copy them if it does not.**
#[repr(C)]
pub struct EmpyreanOrbitCovariance {
    /// 1 when [`non_grav_cross`](Self::non_grav_cross) carries the
    /// state↔Marsden border; 0 when there is none.
    pub has_non_grav_cross: u8,
    /// \\(6 \times 3\\) state-to-\\((A_1, A_2, A_3)\\) cross covariance,
    /// row-major, in the same basis and units as the \\(6 \times 6\\)
    /// beside it. Zeroed when `has_non_grav_cross = 0`.
    pub non_grav_cross: [[f64; 3]; 6],
    /// State-to-parameter cross columns for every parameter that is not
    /// Marsden. Null with a zero count when there are none.
    /// **Library-owned** — see the struct docs.
    pub state_param_cross: *mut EmpyreanStateParamCross,
    /// Number of entries in
    /// [`state_param_cross`](Self::state_param_cross).
    pub n_state_param_cross: usize,
    /// Parameter-to-parameter cross terms with no other home. Null with
    /// a zero count when there are none. **Library-owned** — see the
    /// struct docs.
    pub param_pair_cross: *mut EmpyreanParamPairCross,
    /// Number of entries in
    /// [`param_pair_cross`](Self::param_pair_cross).
    pub n_param_pair_cross: usize,
}

/// The absent joint — no border, no carrier, nothing to free.
///
/// Not `Default`, deliberately: this is a value with a meaning ("the
/// producer had no cross terms"), and every construction site states it
/// rather than inheriting it.
pub(crate) fn empty_orbit_covariance() -> EmpyreanOrbitCovariance {
    EmpyreanOrbitCovariance {
        has_non_grav_cross: 0,
        non_grav_cross: [[0.0; 3]; 6],
        state_param_cross: std::ptr::null_mut(),
        n_state_param_cross: 0,
        param_pair_cross: std::ptr::null_mut(),
        n_param_pair_cross: 0,
    }
}

/// Marshal an engine border + carrier pair into the owned C shape, or
/// **fail**.
///
/// The single producer for both output surfaces — the OD result and
/// every propagated state row — so the two cannot drift in what they
/// emit or in how it is owned.
///
/// # Allocation failure is a failure, not an absence
///
/// `Err` on a failed allocation, never a null-and-zero carrier. Absence
/// and unavailability are different claims at this boundary: a caller
/// reading `n_state_param_cross == 0` concludes *this fit produced no
/// cross terms* and re-feeds a block-diagonal joint, which is exactly
/// the physically unmotivated claim this whole surface exists to stop
/// them making. Reporting a delivered result whose carrier silently
/// vanished would be a hidden fallback in the feature built to remove
/// them. The caller sees the failure code and the message names the
/// allocation.
pub(crate) fn joint_to_c(
    ext: Option<&ExtendedCovariance>,
    cross: Option<&WideCross>,
    what: &str,
) -> Result<EmpyreanOrbitCovariance, String> {
    let (has_non_grav_cross, non_grav_cross) = border_to_c(ext);
    let (state_param_cross, n_state_param_cross, param_pair_cross, n_param_pair_cross) = match cross
    {
        Some(wc) => crate::io::carrier_to_owned_arrays(wc).ok_or_else(|| {
            format!(
                "allocation failed for the {what} wide cross-covariance arrays \
                 ({} state column(s), {} parameter pair(s))",
                wc.state_crosses().count(),
                wc.param_crosses().count()
            )
        })?,
        None => (std::ptr::null_mut(), 0, std::ptr::null_mut(), 0),
    };
    Ok(EmpyreanOrbitCovariance {
        has_non_grav_cross,
        non_grav_cross,
        state_param_cross,
        n_state_param_cross,
        param_pair_cross,
        n_param_pair_cross,
    })
}

/// Release the owned carrier arrays hanging off one
/// [`EmpyreanOrbitCovariance`] and leave it absent.
///
/// Idempotent: a second call on the same struct is a no-op, because the
/// pointers are nulled and the counts zeroed here.
///
/// # Safety
///
/// The arrays must be ones [`joint_to_c`] produced and not yet freed.
pub(crate) unsafe fn free_orbit_covariance(cov: &mut EmpyreanOrbitCovariance) {
    unsafe {
        crate::io::free_owned_side_array(cov.state_param_cross, cov.n_state_param_cross);
        crate::io::free_owned_side_array(cov.param_pair_cross, cov.n_param_pair_cross);
    }
    cov.state_param_cross = std::ptr::null_mut();
    cov.n_state_param_cross = 0;
    cov.param_pair_cross = std::ptr::null_mut();
    cov.n_param_pair_cross = 0;
}

/// Translate a C parameter-column tag into the engine's identity.
///
/// `n_segments` is the orbit's declared `n_correction_covariances`, which
/// is what bounds a THRUST segment index — the same count
/// `wide_layout` derives its thrust block from.
///
/// `what` names the array and entry the tag came from, so a refusal says
/// which of possibly dozens of entries is malformed.
///
/// Refuses (all argument errors, never physics):
/// - an unknown `kind` (E1);
/// - a MARSDEN `index` outside \\(0..3\\), a THRUST `component` outside
///   \\(0..3\\), or a THRUST `segment` at or beyond `n_segments` (E2);
/// - a non-zero `index` / `segment` / `component` on a kind that does not
///   read it (E3).
fn param_column_to_engine(
    c: &EmpyreanParamColumn,
    n_segments: usize,
    what: &str,
) -> Result<ParamColumn, String> {
    // E3 runs per-kind below rather than as one generic sweep, because
    // "which fields does this kind read" is exactly what the match arms
    // already say — a second, separate statement of it would be a copy
    // that can drift.
    let unused = |name: &str, value: u32, kind: &str| -> String {
        format!(
            "{what}: {name} = {value} is set on a {kind} column, which does not read it \
             (it must be 0). A column that names one parameter and carries another's \
             index is refused rather than silently resolved."
        )
    };
    match c.kind {
        EMPYREAN_PARAM_COLUMN_MARSDEN => {
            if c.index >= 3 {
                return Err(format!(
                    "{what}: Marsden index = {} is out of range; it selects A1/A2/A3 and \
                     must be 0, 1 or 2",
                    c.index
                ));
            }
            if c.segment != 0 {
                return Err(unused("segment", c.segment, "Marsden"));
            }
            if c.component != 0 {
                return Err(unused("component", c.component, "Marsden"));
            }
            Ok(ParamColumn::Marsden(c.index as usize))
        }
        EMPYREAN_PARAM_COLUMN_DT => {
            if c.index != 0 {
                return Err(unused("index", c.index, "DT"));
            }
            if c.segment != 0 {
                return Err(unused("segment", c.segment, "DT"));
            }
            if c.component != 0 {
                return Err(unused("component", c.component, "DT"));
            }
            Ok(ParamColumn::Dt)
        }
        EMPYREAN_PARAM_COLUMN_AMRAT => {
            if c.index != 0 {
                return Err(unused("index", c.index, "AMRAT"));
            }
            if c.segment != 0 {
                return Err(unused("segment", c.segment, "AMRAT"));
            }
            if c.component != 0 {
                return Err(unused("component", c.component, "AMRAT"));
            }
            Ok(ParamColumn::Amrat)
        }
        EMPYREAN_PARAM_COLUMN_THRUST => {
            if c.index != 0 {
                return Err(unused("index", c.index, "thrust"));
            }
            if c.component >= 3 {
                return Err(format!(
                    "{what}: thrust component = {} is out of range; it selects x/y/z and \
                     must be 0, 1 or 2",
                    c.component
                ));
            }
            if (c.segment as usize) >= n_segments {
                return Err(format!(
                    "{what}: thrust segment = {} but the orbit declares {n_segments} \
                     correction covariance(s); the segment index is an index into \
                     correction_covariances",
                    c.segment
                ));
            }
            Ok(ParamColumn::Thrust {
                segment: c.segment as usize,
                component: c.component as usize,
            })
        }
        other => Err(format!(
            "{what}: unknown parameter-column kind {other}; the legal tags are \
             {EMPYREAN_PARAM_COLUMN_MARSDEN} (Marsden), {EMPYREAN_PARAM_COLUMN_DT} (DT), \
             {EMPYREAN_PARAM_COLUMN_AMRAT} (AMRAT) and \
             {EMPYREAN_PARAM_COLUMN_THRUST} (thrust)"
        )),
    }
}

/// Translate the engine's identity into a C parameter-column tag.
///
/// The inverse of [`param_column_to_engine`], used on the result path.
/// Every field a kind does not read is written as 0, which is what the
/// input side requires of a caller.
pub(crate) fn param_column_from_engine(c: ParamColumn) -> EmpyreanParamColumn {
    match c {
        ParamColumn::Marsden(i) => EmpyreanParamColumn {
            kind: EMPYREAN_PARAM_COLUMN_MARSDEN,
            index: i as u32,
            segment: 0,
            component: 0,
        },
        ParamColumn::Dt => EmpyreanParamColumn {
            kind: EMPYREAN_PARAM_COLUMN_DT,
            index: 0,
            segment: 0,
            component: 0,
        },
        ParamColumn::Amrat => EmpyreanParamColumn {
            kind: EMPYREAN_PARAM_COLUMN_AMRAT,
            index: 0,
            segment: 0,
            component: 0,
        },
        ParamColumn::Thrust { segment, component } => EmpyreanParamColumn {
            kind: EMPYREAN_PARAM_COLUMN_THRUST,
            index: 0,
            segment: segment as u32,
            component: component as u32,
        },
    }
}

/// Build the wide cross-covariance carrier an [`EmpyreanOrbit`] supplies,
/// or `None` when it supplies none.
///
/// Both side arrays follow the established `thrust_arcs` contract: a
/// null pointer with a non-zero count is a loud argument error, and a
/// non-null pointer with a zero count is absent and never read.
///
/// Duplicate entries are **refused** rather than resolved. The engine's
/// setters are last-write-wins, which at an ABI boundary would turn "I
/// supplied it twice by accident" into "the engine used whichever came
/// later" — the two-numbers-for-one-entry failure the partition
/// discipline exists to prevent. A pair is a duplicate of its own
/// swapped form: \\((a, b)\\) and \\((b, a)\\) are one entry.
///
/// Partition violations (a Marsden term in the carrier, an intra-segment
/// thrust pair, a self-pair) are **not** screened here. They are the
/// engine's own refusals, raised with the orbit id and the assembled
/// layout in hand, and a second copy of the rule at this boundary would
/// have no mechanical link to the first.
pub(crate) fn empyrean_orbit_wide_cross(
    orbit: &EmpyreanOrbit,
) -> Result<Option<WideCross>, String> {
    crate::propagate::validate_side_array_ptr(
        orbit.state_param_cross,
        orbit.n_state_param_cross,
        "state_param_cross",
    )?;
    crate::propagate::validate_side_array_ptr(
        orbit.param_pair_cross,
        orbit.n_param_pair_cross,
        "param_pair_cross",
    )?;

    if orbit.n_state_param_cross == 0 && orbit.n_param_pair_cross == 0 {
        return Ok(None);
    }

    let n_segments = orbit.n_correction_covariances;
    let mut wide = WideCross::new();

    if orbit.n_state_param_cross > 0 {
        let entries = unsafe {
            std::slice::from_raw_parts(orbit.state_param_cross, orbit.n_state_param_cross)
        };
        let mut seen: Vec<ParamColumn> = Vec::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            let column =
                param_column_to_engine(&e.column, n_segments, &format!("state_param_cross[{i}]"))?;
            if let Some(first) = seen.iter().position(|c| *c == column) {
                return Err(format!(
                    "state_param_cross[{i}]: parameter {column} is already supplied at \
                     entry {first}. Entries are identified by their column tag, so a \
                     repeat is two values for one term rather than an ordering — supply \
                     it once."
                ));
            }
            seen.push(column);
            wide.set_state_cross(column, e.values);
        }
    }

    if orbit.n_param_pair_cross > 0 {
        let entries =
            unsafe { std::slice::from_raw_parts(orbit.param_pair_cross, orbit.n_param_pair_cross) };
        let mut seen: Vec<(ParamColumn, ParamColumn)> = Vec::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            let a = param_column_to_engine(&e.a, n_segments, &format!("param_pair_cross[{i}].a"))?;
            let b = param_column_to_engine(&e.b, n_segments, &format!("param_pair_cross[{i}].b"))?;
            // Canonicalize so the swapped form collides with the original:
            // (a, b) and (b, a) name one term, and supplying both is the
            // same two-values-for-one-entry error as a literal repeat.
            let key = if a <= b { (a, b) } else { (b, a) };
            if let Some(first) = seen.iter().position(|k| *k == key) {
                return Err(format!(
                    "param_pair_cross[{i}]: the pair ({a}, {b}) is already supplied at \
                     entry {first} (in either order — the term is symmetric). Supply it once."
                ));
            }
            seen.push(key);
            wide.set_param_cross(a, b, e.value);
        }
    }

    Ok(Some(wide))
}

/// Build the state↔Marsden border an [`EmpyreanOrbit`] supplies, or
/// `None` when it supplies none.
///
/// The C layer supplies only the `cross` half and derives `params` from
/// `non_grav_covariance`, which is the one place this ABI deliberately
/// subsets the engine's surface. The engine compares the two copies
/// exactly and raises `ExtendedCovarianceParamsMismatch` when they
/// disagree; deriving the second copy rather than asking for it makes
/// that failure **unreachable from C by construction** and removes a
/// two-copies-that-can-disagree hazard at the boundary. Every program
/// expressible in C is still a program the engine accepts.
///
/// The other guard stays reachable, deliberately: a border supplied with
/// `has_non_grav_covariance = 0` passes through with a zero `params`
/// block and the engine raises
/// `ExtendedCovarianceWithoutNonGravCovariance` from its own guard. The
/// shell does not re-implement that check.
pub(crate) fn empyrean_orbit_extended_covariance(
    orbit: &EmpyreanOrbit,
) -> Option<ExtendedCovariance> {
    let params = if orbit.has_non_grav_covariance != 0 {
        orbit.non_grav_covariance
    } else {
        [[0.0_f64; 3]; 3]
    };
    border_from_c(
        orbit.state.has_non_grav_cross,
        &orbit.state.non_grav_cross,
        params,
    )
}

/// Assemble an [`ExtendedCovariance`] from the C ABI's flat border pair
/// and a parameter block, or `None` when no border is supplied.
///
/// Shared by the orbit-marshaling path and
/// [`empyrean_transform_coordinates`](crate::transform::empyrean_transform_coordinates),
/// which carries a border through a basis change and so needs the same
/// presence rule from a bare `CoordinateState`.
pub(crate) fn border_from_c(
    has_border: u8,
    cross: &[[f64; 3]; 6],
    params: [[f64; 3]; 3],
) -> Option<ExtendedCovariance> {
    if has_border == 0 {
        return None;
    }
    Some(ExtendedCovariance::new(*cross, params))
}

/// Read a coordinate's border back out into the C ABI's flat pair.
///
/// The inverse of [`empyrean_orbit_extended_covariance`]'s `cross` half.
/// Only the cross is carried: the parameter block travels as the orbit's
/// `non_grav_covariance`, so returning it here too would be the
/// two-homes hazard the input side is shaped to prevent.
///
/// Returns `(0, zeros)` when the coordinate carries no border, which is
/// what a caller's `memset(0)` state already means.
pub(crate) fn border_to_c(ext: Option<&ExtendedCovariance>) -> (u8, [[f64; 3]; 6]) {
    match ext {
        Some(e) => (1, e.cross),
        None => (0, [[0.0_f64; 3]; 6]),
    }
}

/// Re-seat a coordinate's extended covariance without disturbing
/// anything else it carries.
///
/// `Coordinates` holds the border as the third field of every variant,
/// so this is a rebuild rather than a mutation — the same shape scott
/// uses when it writes a fitted border back onto an orbit.
pub(crate) fn coordinates_with_extended(
    coord: Coordinates<AU, Degrees>,
    ext: Option<ExtendedCovariance>,
) -> Coordinates<AU, Degrees> {
    match coord {
        Coordinates::Cartesian(c, cov, _) => Coordinates::Cartesian(c, cov, ext),
        Coordinates::Keplerian(c, cov, _) => Coordinates::Keplerian(c, cov, ext),
        Coordinates::Cometary(c, cov, _) => Coordinates::Cometary(c, cov, ext),
        Coordinates::Spherical(c, cov, _) => Coordinates::Spherical(c, cov, ext),
    }
}

/// Append one C-ABI orbit's coordinate to an `Orbits<AU>` batch **with**
/// its full joint covariance — the \\(6 \times 6\\), the state↔Marsden
/// border, and the wide carrier — converted from the caller's degrees to
/// the engine's stored radian basis in one step.
///
/// This is the single attach point for all five marshaling helpers. It
/// exists as one function rather than five inline blocks for the reason
/// the units channel makes unavoidable: the coordinate and its carrier
/// are one matrix and must convert together. Attaching a carrier after
/// `into_radians` borders a radian-scaled \\(6 \times 6\\) with a
/// degree-scaled carrier — for a Cometary orbit a factor of
/// \\(180/\pi\\) on three of six rows of every carrier column: finite,
/// plausible, accepted by the propagation gate, and wrong by 57×. There
/// is no API here that accepts a half-converted pair.
///
/// The carrier's partition discipline is not validated here; that is the
/// engine's `WideCross::validate`, run by the propagation gate with the
/// orbit id and the assembled layout in hand.
pub(crate) fn push_orbit_with_joint(
    orbits: &mut Orbits<AU>,
    orbit_id: String,
    coords: Coordinates<AU, Degrees>,
    orbit: &EmpyreanOrbit,
) -> Result<(), String> {
    let ext = empyrean_orbit_extended_covariance(orbit);
    let cross = empyrean_orbit_wide_cross(orbit)?;
    orbits
        .push_angular(orbit_id, coordinates_with_extended(coords, ext), cross)
        .map_err(|e| e.to_string())
}

/// The joint-covariance input surface: the identity tags, the four
/// argument-error classes, the zero-init contract, and the units channel.
///
/// These pin the boundary's own behaviour. The engine's refusals
/// (partition violations, definiteness, a cross for an undeclared
/// parameter) are deliberately NOT re-implemented here and are not
/// tested here either — they are tested where they live, and a second
/// copy at this boundary would have no mechanical link to the first.
#[cfg(test)]
mod joint_input_tests {
    use super::*;
    use empyrean_core::coordinates::{CoordinateRepresentation, Degrees};

    fn column(kind: i32) -> EmpyreanParamColumn {
        EmpyreanParamColumn {
            kind,
            index: 0,
            segment: 0,
            component: 0,
        }
    }

    fn marsden(i: u32) -> EmpyreanParamColumn {
        EmpyreanParamColumn {
            kind: EMPYREAN_PARAM_COLUMN_MARSDEN,
            index: i,
            segment: 0,
            component: 0,
        }
    }

    fn thrust(segment: u32, component: u32) -> EmpyreanParamColumn {
        EmpyreanParamColumn {
            kind: EMPYREAN_PARAM_COLUMN_THRUST,
            index: 0,
            segment,
            component,
        }
    }

    /// A zero-init orbit: the `memset(0)` a C caller gets for free.
    fn zeroed_orbit() -> EmpyreanOrbit {
        // SAFETY: `EmpyreanOrbit` is `#[repr(C)]` and every field is a
        // scalar, an array of scalars, or a raw pointer, so the all-zero
        // bit pattern is a valid value of the type. This is exactly the
        // caller-side `memset(0)` whose meaning the test is pinning.
        unsafe { std::mem::zeroed() }
    }

    /// Attach a carrier to a zero-init orbit, borrowing the caller's
    /// backing storage exactly as the C ABI does.
    fn with_carrier(
        orbit: &mut EmpyreanOrbit,
        state: &[EmpyreanStateParamCross],
        pairs: &[EmpyreanParamPairCross],
    ) {
        orbit.state_param_cross = state.as_ptr();
        orbit.n_state_param_cross = state.len();
        orbit.param_pair_cross = pairs.as_ptr();
        orbit.n_param_pair_cross = pairs.len();
    }

    // ── The zero-init contract ────────────────────────────────────

    /// `memset(0)` means "no joint supplied", on both structs. This is
    /// the audit as an executable assertion: every new field's zero
    /// value is a presence switch reading absent or a null pointer, and
    /// none has a non-zero engine default.
    #[test]
    fn a_zero_init_orbit_supplies_no_joint() {
        let o = zeroed_orbit();
        assert_eq!(o.state.has_non_grav_cross, 0, "border absent");
        assert!(o.state_param_cross.is_null(), "carrier pointer null");
        assert_eq!(o.n_state_param_cross, 0);
        assert!(o.param_pair_cross.is_null());
        assert_eq!(o.n_param_pair_cross, 0);

        assert!(
            empyrean_orbit_extended_covariance(&o).is_none(),
            "a zero-init orbit must produce no ExtendedCovariance"
        );
        assert_eq!(
            empyrean_orbit_wide_cross(&o).expect("a zero-init orbit is well-formed"),
            None,
            "a zero-init orbit must produce no carrier"
        );
    }

    /// A non-null pointer with a zero count is absent and never read —
    /// the `thrust_arcs` precedent. Pinned with a deliberately garbage
    /// pointer so a read would fault rather than pass.
    #[test]
    fn a_non_null_pointer_with_zero_count_is_absent() {
        let mut o = zeroed_orbit();
        o.state_param_cross = std::ptr::dangling();
        o.n_state_param_cross = 0;
        o.param_pair_cross = std::ptr::dangling();
        o.n_param_pair_cross = 0;
        assert_eq!(
            empyrean_orbit_wide_cross(&o).expect("absent, not read"),
            None
        );
    }

    /// The mirror rule: a non-zero count with a null pointer is a loud
    /// argument error, not a silent empty read.
    #[test]
    fn a_null_pointer_with_a_non_zero_count_is_refused() {
        for (field, set) in [("state_param_cross", 0), ("param_pair_cross", 1)] {
            let mut o = zeroed_orbit();
            if set == 0 {
                o.n_state_param_cross = 2;
            } else {
                o.n_param_pair_cross = 2;
            }
            let err = empyrean_orbit_wide_cross(&o).expect_err("null + count is refused");
            assert!(err.contains(field), "the error names the field: {err}");
            assert!(err.contains('2'), "the error names the count: {err}");
        }
    }

    // ── E1–E4: the four argument-error classes ────────────────────

    /// E1 — an unknown `kind` is refused by value, and the message
    /// lists the legal set rather than leaving the caller to guess.
    #[test]
    fn e1_an_unknown_column_kind_is_refused() {
        for bad in [-1, 4, 99] {
            let entries = [EmpyreanStateParamCross {
                column: column(bad),
                values: [1.0; 6],
            }];
            let mut o = zeroed_orbit();
            with_carrier(&mut o, &entries, &[]);
            let err = empyrean_orbit_wide_cross(&o).expect_err("unknown kind is refused");
            assert!(
                err.contains(&bad.to_string()),
                "the error names the offending value: {err}"
            );
            assert!(
                err.contains("state_param_cross[0]"),
                "the error names the entry: {err}"
            );
            assert!(err.contains("Marsden") && err.contains("thrust"), "{err}");
        }
    }

    /// E2 — an out-of-range Marsden index or thrust component.
    #[test]
    fn e2_an_out_of_range_index_or_component_is_refused() {
        let entries = [EmpyreanStateParamCross {
            column: marsden(3),
            values: [1.0; 6],
        }];
        let mut o = zeroed_orbit();
        with_carrier(&mut o, &entries, &[]);
        let err = empyrean_orbit_wide_cross(&o).expect_err("A4 does not exist");
        assert!(err.contains("A1/A2/A3"), "{err}");

        let entries = [EmpyreanStateParamCross {
            column: thrust(0, 3),
            values: [1.0; 6],
        }];
        let mut o = zeroed_orbit();
        o.n_correction_covariances = 1;
        with_carrier(&mut o, &entries, &[]);
        let err = empyrean_orbit_wide_cross(&o).expect_err("there is no fourth component");
        assert!(err.contains("x/y/z"), "{err}");
    }

    /// E2 — a thrust segment beyond what the orbit declares. The bound
    /// is `n_correction_covariances`, which is the space
    /// `wide_layout` derives its thrust block in, and the message says
    /// so rather than reporting a bare range.
    #[test]
    fn e2_a_thrust_segment_past_the_declared_count_is_refused() {
        let entries = [EmpyreanStateParamCross {
            column: thrust(2, 0),
            values: [1.0; 6],
        }];
        let mut o = zeroed_orbit();
        o.n_correction_covariances = 2; // segments 0 and 1 exist
        with_carrier(&mut o, &entries, &[]);
        let err = empyrean_orbit_wide_cross(&o).expect_err("segment 2 is not declared");
        assert!(
            err.contains("declares 2"),
            "the error names the count: {err}"
        );
        assert!(err.contains("correction_covariances"), "{err}");

        // And an orbit declaring NO segments admits no thrust tag at all,
        // rather than admitting segment 0 by default.
        let mut o = zeroed_orbit();
        with_carrier(&mut o, &entries[..], &[]);
        assert!(
            empyrean_orbit_wide_cross(&o).is_err(),
            "no declared segments means no legal thrust column"
        );
    }

    /// E3 — a field the kind does not read must be zero. Refused rather
    /// than ignored: a caller who sets `segment` on a DT column has
    /// named one parameter and described another, and silently
    /// resolving to the DT column would hand them a different number
    /// than they asked for.
    #[test]
    fn e3_a_field_the_kind_does_not_read_must_be_zero() {
        let cases: [(EmpyreanParamColumn, &str, &str); 4] = [
            (
                EmpyreanParamColumn {
                    kind: EMPYREAN_PARAM_COLUMN_DT,
                    index: 0,
                    segment: 1,
                    component: 0,
                },
                "segment",
                "DT",
            ),
            (
                EmpyreanParamColumn {
                    kind: EMPYREAN_PARAM_COLUMN_AMRAT,
                    index: 2,
                    segment: 0,
                    component: 0,
                },
                "index",
                "AMRAT",
            ),
            (
                EmpyreanParamColumn {
                    kind: EMPYREAN_PARAM_COLUMN_MARSDEN,
                    index: 1,
                    segment: 0,
                    component: 2,
                },
                "component",
                "Marsden",
            ),
            (
                EmpyreanParamColumn {
                    kind: EMPYREAN_PARAM_COLUMN_THRUST,
                    index: 1,
                    segment: 0,
                    component: 0,
                },
                "index",
                "thrust",
            ),
        ];
        for (col, field, kind) in cases {
            let entries = [EmpyreanStateParamCross {
                column: col,
                values: [1.0; 6],
            }];
            let mut o = zeroed_orbit();
            o.n_correction_covariances = 1;
            with_carrier(&mut o, &entries, &[]);
            let err = empyrean_orbit_wide_cross(&o)
                .expect_err("a set field the kind does not read is refused");
            assert!(
                err.contains(field),
                "the error names the field ({field}): {err}"
            );
            assert!(
                err.contains(kind),
                "the error names the kind ({kind}): {err}"
            );
            assert!(err.contains("must be 0"), "{err}");
        }
    }

    /// E4 — a repeated state column. The engine's setter is
    /// last-write-wins, which at an ABI boundary silently turns "I
    /// supplied it twice by accident" into "the engine used whichever
    /// came later".
    #[test]
    fn e4_a_repeated_state_column_is_refused() {
        let entries = [
            EmpyreanStateParamCross {
                column: column(EMPYREAN_PARAM_COLUMN_DT),
                values: [1.0; 6],
            },
            EmpyreanStateParamCross {
                column: column(EMPYREAN_PARAM_COLUMN_AMRAT),
                values: [2.0; 6],
            },
            EmpyreanStateParamCross {
                column: column(EMPYREAN_PARAM_COLUMN_DT),
                values: [9.0; 6],
            },
        ];
        let mut o = zeroed_orbit();
        with_carrier(&mut o, &entries, &[]);
        let err = empyrean_orbit_wide_cross(&o).expect_err("a repeated column is refused");
        assert!(
            err.contains("state_param_cross[2]"),
            "names the repeat: {err}"
        );
        assert!(err.contains("entry 0"), "names the original: {err}");
    }

    /// E4 — a repeated pair, including its SWAPPED form. This is the
    /// half a naive duplicate check misses: `(DT, AMRAT)` and
    /// `(AMRAT, DT)` are one symmetric term, so supplying both is two
    /// values for one entry rather than an ordering choice.
    #[test]
    fn e4_a_pair_collides_with_its_own_swapped_form() {
        let dt = column(EMPYREAN_PARAM_COLUMN_DT);
        let amrat = column(EMPYREAN_PARAM_COLUMN_AMRAT);
        let pairs = [
            EmpyreanParamPairCross {
                a: dt,
                b: amrat,
                value: 1.0,
            },
            EmpyreanParamPairCross {
                a: amrat,
                b: dt,
                value: 7.0,
            },
        ];
        let mut o = zeroed_orbit();
        with_carrier(&mut o, &[], &pairs);
        let err = empyrean_orbit_wide_cross(&o).expect_err("the swapped form is the same term");
        assert!(err.contains("param_pair_cross[1]"), "{err}");
        assert!(
            err.contains("either order"),
            "the error explains why: {err}"
        );

        // The honest case still passes: two DIFFERENT pairs sharing an
        // endpoint are not duplicates.
        let m0 = marsden(0);
        let pairs = [
            EmpyreanParamPairCross {
                a: dt,
                b: amrat,
                value: 1.0,
            },
            EmpyreanParamPairCross {
                a: dt,
                b: m0,
                value: 2.0,
            },
        ];
        let mut o = zeroed_orbit();
        with_carrier(&mut o, &[], &pairs);
        let carrier = empyrean_orbit_wide_cross(&o)
            .expect("distinct pairs sharing an endpoint are not duplicates")
            .expect("a carrier was supplied");
        assert_eq!(carrier.param_crosses().count(), 2);
    }

    // ── Identity round trip ───────────────────────────────────────

    /// Every legal tag survives C → engine → C unchanged, and the
    /// fields a kind does not read come back zero — which is what the
    /// input side requires, so a result can be re-fed without editing.
    #[test]
    fn every_identity_tag_round_trips_through_the_engine_form() {
        let mut cases = vec![
            column(EMPYREAN_PARAM_COLUMN_DT),
            column(EMPYREAN_PARAM_COLUMN_AMRAT),
        ];
        cases.extend((0..3).map(marsden));
        for s in 0..3 {
            cases.extend((0..3).map(|c| thrust(s, c)));
        }
        for c in cases {
            let engine = param_column_to_engine(&c, 3, "round trip").expect("legal tag");
            let back = param_column_from_engine(engine);
            assert_eq!(back, c, "tag must survive the round trip unchanged");
        }
    }

    // ── Ordering is not contract ──────────────────────────────────

    /// Entry order does not matter: the same entries in a shuffled
    /// order produce an identical carrier, because entries are keyed by
    /// identity rather than by position. This is the property that
    /// makes the whole surface immune to layout shifts.
    #[test]
    fn entry_order_is_not_contract() {
        let dt = column(EMPYREAN_PARAM_COLUMN_DT);
        let amrat = column(EMPYREAN_PARAM_COLUMN_AMRAT);
        let forward = [
            EmpyreanStateParamCross {
                column: dt,
                values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
            EmpyreanStateParamCross {
                column: amrat,
                values: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            },
        ];
        let reversed = [forward[1], forward[0]];

        let mut a = zeroed_orbit();
        with_carrier(&mut a, &forward, &[]);
        let mut b = zeroed_orbit();
        with_carrier(&mut b, &reversed, &[]);

        assert_eq!(
            empyrean_orbit_wide_cross(&a).unwrap(),
            empyrean_orbit_wide_cross(&b).unwrap(),
            "a shuffled emission order must produce the identical carrier"
        );
    }

    // ── The border ────────────────────────────────────────────────

    /// The border's parameter block is DERIVED from
    /// `non_grav_covariance` rather than asked for a second time, which
    /// is what makes the engine's `ExtendedCovarianceParamsMismatch`
    /// unreachable from C: there is no second copy to disagree.
    #[test]
    fn the_border_derives_its_parameter_block_from_the_orbit() {
        let mut o = zeroed_orbit();
        o.state.has_non_grav_cross = 1;
        o.state.non_grav_cross = [[1.0, 2.0, 3.0]; 6];
        o.has_non_grav_covariance = 1;
        o.non_grav_covariance = [[9.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 7.0]];

        let ext = empyrean_orbit_extended_covariance(&o).expect("a border was supplied");
        assert_eq!(ext.cross, o.state.non_grav_cross);
        assert_eq!(
            ext.params, o.non_grav_covariance,
            "params comes from the orbit's own 3x3, never from a second C-side copy"
        );
    }

    /// A border supplied WITHOUT its parameter block is passed through
    /// with a zero block rather than screened here, so the engine
    /// raises its own `ExtendedCovarianceWithoutNonGravCovariance`. The
    /// shell must not pre-empt an engine guard with a hand-maintained
    /// copy of the same rule.
    #[test]
    fn a_border_without_a_parameter_block_reaches_the_engine_guard() {
        let mut o = zeroed_orbit();
        o.state.has_non_grav_cross = 1;
        o.state.non_grav_cross = [[1.0, 2.0, 3.0]; 6];
        o.has_non_grav_covariance = 0;

        let ext = empyrean_orbit_extended_covariance(&o)
            .expect("the border is carried through, not dropped and not refused here");
        assert_eq!(
            ext.params, [[0.0; 3]; 3],
            "zero block, for the engine to catch"
        );
    }

    // ── The units channel ─────────────────────────────────────────

    /// The discriminating test for the attach point: a carrier supplied
    /// in the coordinate's own angular unit must be scaled by the same
    /// factor as the 6×6 it borders.
    ///
    /// A Cometary orbit's angular element rows are degrees on the ABI
    /// and radians in the engine. Attaching the carrier AFTER the
    /// conversion — which is what every marshal site did before this
    /// change, and what `set_wide_cross` still does — leaves those rows
    /// a factor of 180/π too large: finite, plausible, accepted by the
    /// propagation gate, and wrong by 57×.
    #[test]
    fn the_carrier_converts_with_the_coordinate_it_borders() {
        use empyrean_core::convert::coordinate_state_to_coordinates;

        let mut o = zeroed_orbit();
        o.state = crate::CoordinateState {
            epoch_mjd_tdb: 59000.0,
            // q, e, i, node, argperi, tp — the middle three are angles.
            elements: [1.0, 0.2, 10.0, 20.0, 30.0, 59000.0],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: 2, // Cometary
            frame: 0,
            origin: 10,
            has_non_grav_cross: 0,
            non_grav_cross: [[0.0; 3]; 6],
        };
        // One unit in every state row, so the scaling of each row is
        // readable directly off the stored carrier.
        let entries = [EmpyreanStateParamCross {
            column: column(EMPYREAN_PARAM_COLUMN_AMRAT),
            values: [1.0; 6],
        }];
        with_carrier(&mut o, &entries, &[]);

        let coords = coordinate_state_to_coordinates(&o.state.to_empyrean())
            .expect("a well-formed Cometary state");
        assert_eq!(coords.representation(), CoordinateRepresentation::Cometary);

        let mut orbits: Orbits<AU> = Orbits::empty();
        push_orbit_with_joint(&mut orbits, "c".to_string(), coords, &o)
            .expect("the joint attaches");

        let stored = orbits
            .wide_cross(0)
            .expect("the carrier reached the orbit")
            .state_cross(ParamColumn::Amrat)
            .copied()
            .expect("the AMRAT column is present");

        // Cometary's angular element indices are 2, 3, 4 (i, node,
        // argperi). Those rows are scaled to radians; q, e and tp are
        // not angles and are untouched.
        let to_rad = std::f64::consts::PI / 180.0;
        for (row, value) in stored.iter().enumerate() {
            let expected = if (2..5).contains(&row) { to_rad } else { 1.0 };
            assert!(
                (value - expected).abs() < 1e-15,
                "row {row}: expected {expected}, got {value} — the carrier must be \
                 scaled exactly as the 6x6 beside it is"
            );
        }

        // And the read-out is the same conversion in reverse, so a round
        // trip returns what was supplied. A read-out that converted the
        // coordinate and left the carrier alone would make the error
        // visible only on round trips, and read as a marshaling bug.
        let (_, back) = orbits
            .coordinates_angular::<Degrees>(0)
            .expect("row 0 exists");
        let back = back
            .expect("the carrier comes back")
            .state_cross(ParamColumn::Amrat)
            .copied()
            .expect("the AMRAT column survives");
        for (row, value) in back.iter().enumerate() {
            assert!(
                (value - 1.0).abs() < 1e-15,
                "row {row}: the degree round trip must return the supplied value, got {value}"
            );
        }
    }

    /// A Cartesian orbit has no angular rows, so the same carrier
    /// crosses unscaled. Paired with the Cometary case above, this is
    /// what distinguishes "converted correctly" from "not converted at
    /// all" — a broken attach point passes one of the two.
    #[test]
    fn a_cartesian_carrier_crosses_unscaled() {
        use empyrean_core::convert::coordinate_state_to_coordinates;

        let mut o = zeroed_orbit();
        o.state = crate::CoordinateState {
            epoch_mjd_tdb: 59000.0,
            elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: 0, // Cartesian
            frame: 0,
            origin: 10,
            has_non_grav_cross: 0,
            non_grav_cross: [[0.0; 3]; 6],
        };
        let entries = [EmpyreanStateParamCross {
            column: column(EMPYREAN_PARAM_COLUMN_DT),
            values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }];
        with_carrier(&mut o, &entries, &[]);

        let coords = coordinate_state_to_coordinates(&o.state.to_empyrean()).unwrap();
        let mut orbits: Orbits<AU> = Orbits::empty();
        push_orbit_with_joint(&mut orbits, "x".to_string(), coords, &o).unwrap();

        let stored = orbits
            .wide_cross(0)
            .unwrap()
            .state_cross(ParamColumn::Dt)
            .copied()
            .unwrap();
        assert_eq!(
            stored,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Cartesian has no angular rows, so nothing is scaled"
        );
    }

    /// An all-zero carrier ENTRY is a supplied zero correlation, not an
    /// absent one — the deliberate asymmetry with the border, where an
    /// all-zero cross reads as absent. The two readings differ, and a
    /// consumer should not have to discover that.
    #[test]
    fn an_all_zero_entry_is_supplied_rather_than_absent() {
        let entries = [EmpyreanStateParamCross {
            column: column(EMPYREAN_PARAM_COLUMN_DT),
            values: [0.0; 6],
        }];
        let mut o = zeroed_orbit();
        with_carrier(&mut o, &entries, &[]);
        let carrier = empyrean_orbit_wide_cross(&o)
            .expect("well-formed")
            .expect("a carrier was supplied");
        assert!(
            !carrier.is_empty(),
            "a supplied zero entry counts as supplied and engages the definiteness gate"
        );
        assert_eq!(carrier.state_cross(ParamColumn::Dt), Some(&[0.0; 6]));
    }
}
