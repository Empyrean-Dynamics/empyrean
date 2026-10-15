//! An orbit's full solved-parameter covariance as ONE self-describing
//! packed joint at the C ABI — the single covariance home, input and
//! output, at every width.
//!
//! A fitted or propagated orbit's uncertainty is one \\((6+P) \times (6+P)\\)
//! matrix over the state and every solved parameter (the Marsden
//! non-gravitational coefficients, the non-grav time delay, the SRP
//! area-to-mass ratio, and each thrust \\(\Delta v\\) segment). Rather than
//! scatter that matrix across diagonal-block and cross-term carriers — where
//! a caller who chained one leg into the next necessarily handed the engine a
//! block-diagonal covariance, a physically unmotivated claim that the state
//! and the parameters are independent — the whole joint travels as one
//! [`EmpyreanPackedJoint`]: three axis bitsets, the packed lower triangle, and
//! the tags that name what the joint describes.
//!
//! The same struct is the input orbit's covariance and every result's
//! covariance, with the same field names and the same ownership split
//! (caller-owned on input, library-owned on a result), so leg chaining is
//! uniform: whatever produced the state, its joint is read the same way and
//! re-fed the same way.
//!
//! [`c_to_orbit_uncertainty`] builds an engine
//! [`OrbitUncertainty`](empyrean_core::propagation::OrbitUncertainty) from the
//! wire through the facade, and [`orbit_uncertainty_to_c`] marshals one back
//! out. The triangle is in the engine's radian basis on both sides, so a
//! joint round-trips unchanged.

use empyrean_core::coordinates::{
    AU, Cartesian, CoordinateRepresentation, Coordinates, Degrees, Spherical,
};
use empyrean_core::orbits::Orbits;
use empyrean_core::propagation::{
    Basis, Derivation, OrbitUncertainty, Provenance, Quality, Uncertainty,
};
use empyrean_core::uncertainty::UncertaintyWireError;

use crate::propagate::EmpyreanOrbit;

// ── Packed-joint input-refusal FFI codes ────────────────────────────
//
// Every boundary that turns a caller-supplied packed joint into an engine
// `OrbitUncertainty` — a direct propagate / OD / ephemeris / impact push AND
// the orbit file-write path (`empyrean_orbits_write_parquet` / `_write_csv`,
// which convert C rows to engine orbits first) — refuses a malformed wire row
// by the SAME per-variant code, never a single code with the detail only in
// the message. The ten codes are the reserved block −40…−49, one per
// `empyrean_core::uncertainty::UncertaintyWireError` variant.

/// Packed joint refused: the three axis bitsets break an `AxisLayout` invariant.
pub const EMPYREAN_PUSH_LAYOUT_INVALID: i32 = -40;
/// Packed joint refused: the triangle length disagrees with the declared width.
pub const EMPYREAN_PUSH_TRIANGLE_LENGTH: i32 = -41;
/// Packed joint refused: the declared width disagrees with the layout's width.
pub const EMPYREAN_PUSH_WIDTH_MISMATCH: i32 = -42;
/// Packed joint refused: an unrecognized `functional` wire discriminant.
pub const EMPYREAN_PUSH_UNKNOWN_FUNCTIONAL: i32 = -43;
/// Packed joint refused: an unrecognized `provenance` wire discriminant.
pub const EMPYREAN_PUSH_UNKNOWN_PROVENANCE: i32 = -44;
/// Packed joint refused: a retired / reserved wire discriminant in a validated column.
pub const EMPYREAN_PUSH_RETIRED_DISCRIMINANT: i32 = -45;
/// Packed joint refused: the covariance basis epoch and the row epoch disagree.
pub const EMPYREAN_PUSH_EPOCH_MISMATCH: i32 = -46;
/// Packed joint refused: the engine rejected the assembled joint (loud catch-all).
pub const EMPYREAN_PUSH_ENGINE_REFUSED: i32 = -47;
/// Packed joint refused: a live non-linear `kind` an input prior cannot transport.
pub const EMPYREAN_PUSH_UNSUPPORTED_INPUT_KIND: i32 = -48;
/// Packed joint refused: an unrecognized `asserted_source` companion discriminant.
pub const EMPYREAN_PUSH_UNKNOWN_ASSERTED_SOURCE: i32 = -49;

/// Map a [`UncertaintyWireError`] variant to its reserved FFI code (−40…−49).
/// The match is exhaustive: a new engine variant is a compile error here
/// rather than a silent fall-through to one code.
pub(crate) fn uncertainty_wire_error_code(e: &UncertaintyWireError) -> i32 {
    match e {
        UncertaintyWireError::LayoutInvalid { .. } => EMPYREAN_PUSH_LAYOUT_INVALID,
        UncertaintyWireError::TriangleLength { .. } => EMPYREAN_PUSH_TRIANGLE_LENGTH,
        UncertaintyWireError::WidthMismatch { .. } => EMPYREAN_PUSH_WIDTH_MISMATCH,
        UncertaintyWireError::UnknownFunctional { .. } => EMPYREAN_PUSH_UNKNOWN_FUNCTIONAL,
        UncertaintyWireError::UnknownProvenance { .. } => EMPYREAN_PUSH_UNKNOWN_PROVENANCE,
        UncertaintyWireError::RetiredDiscriminant { .. } => EMPYREAN_PUSH_RETIRED_DISCRIMINANT,
        UncertaintyWireError::EpochMismatch { .. } => EMPYREAN_PUSH_EPOCH_MISMATCH,
        UncertaintyWireError::EngineRefused { .. } => EMPYREAN_PUSH_ENGINE_REFUSED,
        UncertaintyWireError::UnsupportedInputKind { .. } => EMPYREAN_PUSH_UNSUPPORTED_INPUT_KIND,
        UncertaintyWireError::UnknownAssertedSource { .. } => EMPYREAN_PUSH_UNKNOWN_ASSERTED_SOURCE,
    }
}

/// Wire discriminants for the tagged side columns of [`EmpyreanPackedJoint`]
/// — `quality`, `functional` and `provenance`. These are the C ABI contract:
/// documented in `include/empyrean.h` and read by C consumers of a result's
/// packed joint to interpret its tags, not by this crate's own Rust code
/// (which writes them straight from the engine's `wire_discriminant()`). The
/// module-level `allow(dead_code)` marks the whole set as the C contract
/// rather than tagging each constant; the values are pinned against the
/// engine's `Quality` / `Functional` / `Provenance` encodings in
/// `contract_tests::wire_discriminant_consts_are_stable`.
pub(crate) mod wire_tags {
    #![allow(dead_code)]

    // ── EmpyreanPackedJoint.quality — engine `Quality::wire_discriminant` ──
    /// All correlation-space eigenvalues positive; `cov_min_eig_corr` is NaN.
    pub const EMPYREAN_JOINT_QUALITY_POSITIVE_DEFINITE: u8 = 0;
    /// Exact zero variance on named axes; read `cov_rank` / `cov_null_axes`.
    pub const EMPYREAN_JOINT_QUALITY_POSITIVE_SEMIDEFINITE: u8 = 1;
    /// Minimum correlation-space eigenvalue below tolerance; `cov_min_eig_corr`
    /// carries that (negative) eigenvalue.
    pub const EMPYREAN_JOINT_QUALITY_INDEFINITE: u8 = 2;
    /// Explicitly repaired to PSD; `cov_min_eig_corr` is the pre-repair minimum.
    pub const EMPYREAN_JOINT_QUALITY_REPAIRED: u8 = 3;

    // ── EmpyreanPackedJoint.functional — engine `Functional::wire_discriminant` ──
    /// The second moment of the state axes.
    pub const EMPYREAN_JOINT_FUNCTIONAL_STATE: u8 = 0;
    /// Reception-epoch sky coordinates.
    pub const EMPYREAN_JOINT_FUNCTIONAL_SKY: u8 = 1;
    /// Delay / Doppler observables.
    pub const EMPYREAN_JOINT_FUNCTIONAL_DELAY_DOPPLER: u8 = 2;

    // ── EmpyreanPackedJoint.provenance — engine `Provenance::wire_discriminant` ──
    /// Derived by this engine.
    pub const EMPYREAN_JOINT_PROVENANCE_DERIVED: u8 = 0;
    /// Asserted by an external source at ingestion.
    pub const EMPYREAN_JOINT_PROVENANCE_ASSERTED: u8 = 1;

    // ── EmpyreanPackedJoint.asserted_source — engine `AssertedSource::wire_discriminant` ──
    //
    // The external source that asserted the moment, meaningful ONLY when
    // `provenance == EMPYREAN_JOINT_PROVENANCE_ASSERTED`. The two source values
    // pin the engine's `AssertedSource` encoding (not core-re-exported, so
    // pinned by their append-only value against villeneuve `packed_joint.rs`);
    // a derived joint carries `_NONE`, the ABI's not-available sentinel chosen
    // above every append-only engine discriminant so it never collides with a
    // future source.
    /// An orbit asserted from the JPL Small-Body Database.
    pub const EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB: u8 = 0;
    /// An orbit asserted from another external tool / file.
    pub const EMPYREAN_JOINT_ASSERTED_SOURCE_EXTERNAL_FILE: u8 = 1;
    /// No asserted source — the value `asserted_source` carries on any joint
    /// whose `provenance` is not `ASSERTED`.
    pub const EMPYREAN_JOINT_ASSERTED_SOURCE_NONE: u8 = 255;
}

/// A C-ABI error carrying an optional specific return code.
///
/// `code: Some(c)` pins the code a boundary must return — a per-variant
/// packed-joint refusal (−40…−49) reaches every push boundary this way, so a
/// malformed wire row refused at a file writer returns the same code it would
/// at a direct push. `code: None` is an ordinary error that takes the
/// boundary's own default code (its message still names what failed). One
/// mapping, shared by every push caller, its extern wrapper and both orbit
/// writers.
#[derive(Debug)]
pub(crate) struct AbiError {
    /// The specific code to return, or `None` to use the boundary default.
    pub code: Option<i32>,
    /// The human-readable message for `empyrean_last_error`.
    pub message: String,
}

impl AbiError {
    /// An error that pins a specific return code.
    pub(crate) fn coded(code: i32, message: String) -> Self {
        Self {
            code: Some(code),
            message,
        }
    }

    /// Prepend a context prefix to the message, keeping the code.
    pub(crate) fn prefixed(mut self, ctx: &str) -> Self {
        self.message = format!("{ctx}: {}", self.message);
        self
    }
}

impl From<String> for AbiError {
    fn from(message: String) -> Self {
        Self {
            code: None,
            message,
        }
    }
}

impl std::fmt::Display for AbiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// An orbit's full uncertainty as ONE self-describing packed joint — the
/// single covariance home at the C ABI, input and output, at every width.
///
/// # The axis table
///
/// The three `layout_*` bitsets index the canonical parameter-axis table
/// (bit → axis):
///
/// | bits | axis |
/// |---|---|
/// | 0, 1, 2 | Marsden \\(A_1, A_2, A_3\\) |
/// | 3 | non-grav time delay \\(\Delta T\\) (only alongside Marsden) |
/// | 4 | SRP area-to-mass ratio |
/// | 5 + 3·s + c | thrust segment `s`, \\(\Delta v\\) component `c` (x/y/z) |
///
/// `layout_present` are the axes with a column; `layout_considered` is a
/// subset of `present`; `layout_marginalized` is disjoint from `present`.
/// The joint width is `6 + popcount(layout_present)`.
///
/// A thrust segment's covariance (a burn prior) is the joint's thrust-axis
/// block(s); there is no separate thrust-covariance carrier.
///
/// `tri` is the packed LOWER triangle of the `width × width` symmetric
/// joint, row-major with `(i, j) → i·(i+1)/2 + j` for `j ≤ i`, of
/// `width·(width+1)/2` doubles, in the coordinate's own element space and
/// angular unit — the SAME units as the flat
/// [`CoordinateState::covariance`](crate::CoordinateState::covariance), so
/// the ABI carries one unit convention, not two. The leading
/// \\(6 \times 6\\) is the state block: position / velocity rows are AU and
/// AU/day, the angular state rows and columns are in **degrees** for the
/// angular representations (unscaled for Cartesian), and the parameter axes
/// (Marsden / DT / AMRAT / thrust) are always unscaled.
///
/// | representation | angular state rows/cols (in degrees) |
/// |---|---|
/// | Cartesian | none |
/// | Keplerian | \\(i, \Omega, \omega, M\\) (indices 2, 3, 4, 5) |
/// | Cometary | \\(i, \Omega, \omega\\) (indices 2, 3, 4) |
/// | Spherical | lon, lat, \\(\dot{\text{lon}}, \dot{\text{lat}}\\) (1, 2, 4, 5) |
///
/// Each boundary converts between this degree basis and the engine's radian
/// basis by the diagonal congruence \\(D \Sigma D^\top\\) (the scale on an
/// angular row/column, identity elsewhere): the input builder scales
/// degrees → radians before building the prior, and the output marshaler
/// scales radians → degrees after reading the engine triangle. A
/// degree → engine → degree round trip therefore returns the supplied joint
/// within a scaling band — never bit-identical across the congruence.
///
/// The tagged side columns name what the joint describes: `kind` is the
/// derivation (the `EMPYREAN_COVARIANCE_KIND_*` tag); `quality` the
/// definiteness (0 = positive-definite, 1 = positive-semidefinite with
/// `cov_rank` / `cov_null_axes`, 2 = indefinite and 3 = repaired with
/// `cov_min_eig_corr`); `functional` which second moment the axes describe
/// (0 = state); `provenance` whether the engine derived the moment
/// (0 = derived) or it was asserted at ingestion; `asserted_source` which
/// external source asserted it — an `EMPYREAN_JOINT_ASSERTED_SOURCE_*` value,
/// meaningful ONLY when `provenance == EMPYREAN_JOINT_PROVENANCE_ASSERTED` and
/// otherwise the documented `_NONE` (255) value. A side value that does
/// not apply is `NaN` (floating), 0 or the `_NONE` sentinel, the ABI's
/// not-available convention.
/// `kind` and `quality` are read on OUTPUT only — the engine re-derives them
/// when it ingests a prior.
///
/// # Ownership
///
/// On a result the library owns `tri` and releases it with the parent
/// result (via the result's `*_free`); on an input orbit the caller owns it
/// and it is borrowed for the call — the same split as the other carrier
/// arrays.
#[repr(C)]
pub struct EmpyreanPackedJoint {
    /// Axes with a column in the joint, as a bitset over the axis table.
    pub layout_present: u32,
    /// Considered axes — a subset of `layout_present`.
    pub layout_considered: u32,
    /// Axes whose identity is present but whose column is absent; disjoint
    /// from `layout_present`.
    pub layout_marginalized: u32,
    /// The joint width `N = 6 + popcount(layout_present)`.
    pub width: u8,
    /// The packed lower triangle, `width·(width+1)/2` doubles. See the
    /// struct docs for the index convention.
    pub tri: *const f64,
    /// Number of doubles in [`tri`](Self::tri).
    pub tri_len: usize,
    /// The derivation tag — an `EMPYREAN_COVARIANCE_KIND_*` value.
    pub kind: u8,
    /// The definiteness tag — an `EMPYREAN_JOINT_QUALITY_*` value (0–3); see
    /// the struct docs.
    pub quality: u8,
    /// The numerical rank; read only when `quality == 1`.
    pub cov_rank: u8,
    /// The null-axis bitset; read only when `quality == 1`.
    pub cov_null_axes: u32,
    /// The minimum correlation-space eigenvalue; read only when
    /// `quality == 2` or `3`, else `NaN`.
    pub cov_min_eig_corr: f64,
    /// Which second moment the axes describe — an `EMPYREAN_JOINT_FUNCTIONAL_*`
    /// value (0 = state).
    pub functional: u8,
    /// Whether the moment was derived or asserted — an
    /// `EMPYREAN_JOINT_PROVENANCE_*` value (0 = derived).
    pub provenance: u8,
    /// Which external source asserted the moment — an
    /// `EMPYREAN_JOINT_ASSERTED_SOURCE_*` value. Meaningful ONLY when
    /// `provenance == EMPYREAN_JOINT_PROVENANCE_ASSERTED`; on any other joint
    /// it carries the `EMPYREAN_JOINT_ASSERTED_SOURCE_NONE` (255) sentinel.
    pub asserted_source: u8,
}

/// The engine's pre-retention Gaussian-mixture accounting, carried beside a
/// covariance whose derivation is `Mixture` — the same six tallies the
/// reference channel reads off `Uncertainty::derivation()`. It rides WITH the
/// covariance it describes (every carrier that holds a `cov_kind` + packed
/// `joint`), NOT the retained-component status table
/// ([`EmpyreanMixtureChain`](crate::propagate::EmpyreanMixtureChain)), which is
/// a separate per-close-approach object.
///
/// Read presence from the carrier's `has_mixture_tally` flag, never from the
/// `kind`: a non-`Mixture` covariance leaves the tally absent. A `-1` in any
/// `n_*` field is the engine's `None` — "the producing site did not carry that
/// tally" — and is distinct from a real `0`. `n_components_total` and
/// `weight_delivered` are always carried on a `Mixture`; the four refusal /
/// failure counts may each be engine-`None` (`-1`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmpyreanMixtureTally {
    /// Total number of components before retention. Always present on a
    /// `Mixture`.
    pub n_components_total: u32,
    /// Pre-renormalization surviving mass. Always present on a `Mixture`.
    pub weight_delivered: f64,
    /// Members whose sub-propagation failed, or `-1` when the producing site
    /// did not carry the tally.
    pub n_failed: i64,
    /// Members still inside a zone at their horizon, or `-1` when not carried.
    pub n_unresolved: i64,
    /// Components refused for state-space curvature, or `-1` when not carried.
    pub n_curvature_refused: i64,
    /// Components refused because their projected spread overflowed the
    /// flat-sky linearization, or `-1` when not carried — engine-`None` on the
    /// propagation seam at this pin, so carried as `-1` (absent) by name,
    /// never fabricated as `0`.
    pub n_sky_linearization_refused: i64,
}

/// The absent mixture tally — the sentinel payload a carrier holds when
/// `has_mixture_tally == 0` (every non-`Mixture` covariance). Every `n_*`
/// field is the `-1` "not carried" sentinel, never `0`.
pub(crate) fn absent_mixture_tally() -> EmpyreanMixtureTally {
    EmpyreanMixtureTally {
        n_components_total: 0,
        weight_delivered: 0.0,
        n_failed: -1,
        n_unresolved: -1,
        n_curvature_refused: -1,
        n_sky_linearization_refused: -1,
    }
}

/// `(has_mixture_tally, tally)` for a delivered covariance's derivation.
///
/// A [`Derivation::Mixture`] yields `(1, …)` with the engine's six tallies
/// read straight off the carried [`MixtureComponents`]; every other derivation
/// yields `(0, absent)`. An engine `None` on any of the four optional counts
/// becomes `-1`, never `0` — the same no-fabricated-zeros contract the
/// reference channel keeps. Representation-generic (`R`), so the Cartesian
/// propagation seam and the spherical ephemeris seam read a mixture row
/// identically.
///
/// [`MixtureComponents`]: empyrean_core::propagation::MixtureComponents
pub(crate) fn mixture_tally_from_derivation<R>(
    derivation: &Derivation<R>,
) -> (u8, EmpyreanMixtureTally) {
    match derivation {
        Derivation::Mixture(mc) => (
            1,
            EmpyreanMixtureTally {
                n_components_total: mc.n_components_total as u32,
                weight_delivered: mc.weight_delivered,
                n_failed: mc.n_failed.map(|v| v as i64).unwrap_or(-1),
                n_unresolved: mc.n_unresolved.map(|v| v as i64).unwrap_or(-1),
                n_curvature_refused: mc.n_curvature_refused.map(|v| v as i64).unwrap_or(-1),
                n_sky_linearization_refused: mc
                    .n_sky_linearization_refused
                    .map(|v| v as i64)
                    .unwrap_or(-1),
            },
        ),
        _ => (0, absent_mixture_tally()),
    }
}

/// The absent joint — width 0, no triangle, nothing to free.
pub(crate) fn empty_packed_joint() -> EmpyreanPackedJoint {
    EmpyreanPackedJoint {
        layout_present: 0,
        layout_considered: 0,
        layout_marginalized: 0,
        width: 0,
        tri: std::ptr::null(),
        tri_len: 0,
        kind: crate::propagate::EMPYREAN_COVARIANCE_KIND_LINEAR,
        quality: 0,
        cov_rank: 0,
        cov_null_axes: 0,
        cov_min_eig_corr: f64::NAN,
        functional: 0,
        provenance: 0,
        asserted_source: wire_tags::EMPYREAN_JOINT_ASSERTED_SOURCE_NONE,
    }
}

/// Apply the diagonal congruence \\(D \Sigma D^\top\\) to a packed lower
/// triangle in place: entry \\((i, j)\\) is multiplied by \\(D_i D_j\\),
/// where \\(D_k = \text{factor}\\) on an angular state row `k` (`k` in
/// `angular`, all `< 6`) and \\(1\\) elsewhere — so position / velocity rows
/// and every parameter axis are left untouched.
///
/// `factor` is \\(\pi/180\\) for degrees → radians (the input boundary) and
/// \\(180/\pi\\) for the inverse (the output boundary). An angular row and
/// its matching column are each multiplied once, so an entry pairing two
/// angular elements carries the factor twice — the covariance analogue of
/// the state vector's own per-angle conversion, matching the flat
/// `CoordinateState.covariance` scaling. Indices beyond `tri.len()` are
/// skipped: a malformed short triangle is refused downstream by the engine,
/// by name, not silently here.
fn scale_tri_angular(tri: &mut [f64], angular: &[usize], factor: f64) {
    let packed = |i: usize, j: usize| i * (i + 1) / 2 + j; // j <= i
    for &a in angular {
        // Row a: entries (a, j) for j <= a.
        for j in 0..=a {
            let idx = packed(a, j);
            if idx < tri.len() {
                tri[idx] *= factor;
            }
        }
        // Column a: entries (i, a) for i > a, to the triangle's extent.
        let mut i = a + 1;
        loop {
            let idx = packed(i, a);
            if idx >= tri.len() {
                break;
            }
            tri[idx] *= factor;
            i += 1;
        }
        // Diagonal (a, a): an angular variance, so \\(D_a D_a = \text{factor}^2\\).
        // The row loop applied `factor` once and the column loop starts past
        // it, so apply the second factor here — matching the flat path's dense
        // `scale_angular_covariance` (which multiplies the whole row AND the
        // whole column, hitting the diagonal twice). Without this the angular
        // variance is off by `factor` and a re-fed joint disagrees with the
        // flat state block it was derived from (the −1 refusal).
        let d = packed(a, a);
        if d < tri.len() {
            tri[d] *= factor;
        }
    }
}

/// The packed lower triangle of an [`OrbitUncertainty`], regardless of
/// representation — the wire form read straight off the joint.
fn orbit_uncertainty_tri(ou: &OrbitUncertainty<AU>) -> &[f64] {
    use empyrean_core::propagation::OrbitUncertainty as OU;
    match ou {
        OU::Cartesian(u) => u.joint().tri(),
        OU::Keplerian(u) => u.joint().tri(),
        OU::Cometary(u) => u.joint().tri(),
        OU::Spherical(u) => u.joint().tri(),
    }
}

/// Marshal an engine [`OrbitUncertainty`] into the owned C packed-joint
/// struct: the layout bitsets, the packed lower triangle (library-owned),
/// and the tags read straight off the joint — nothing recomputed.
///
/// The returned `tri` is heap-allocated and must be released with
/// [`free_packed_joint`].
pub(crate) fn orbit_uncertainty_to_c(ou: &OrbitUncertainty<AU>) -> EmpyreanPackedJoint {
    // The wire triangle, read off the engine joint (radian basis), copied
    // into a library-owned heap buffer and scaled back to this ABI's degree
    // basis on the angular state rows/columns of the representation (the
    // inverse of the input builder's congruence).
    let mut tri_vec: Vec<f64> = orbit_uncertainty_tri(ou).to_vec();
    scale_tri_angular(
        &mut tri_vec,
        ou.representation().angular_element_indices(),
        180.0 / std::f64::consts::PI,
    );
    packed_joint_from_parts(
        ou.layout().bits(),
        ou.width() as u8,
        tri_vec,
        crate::propagate::covariance_kind_to_u8(ou.kind()),
        ou.quality(),
        ou.functional().wire_discriminant(),
        ou.provenance(),
    )
}

/// Marshal one ephemeris row's resolved SKY covariance
/// (`Uncertainty<Spherical<AU, Degrees>>`) into the owned C packed joint — the
/// same struct and the same tri packing the propagation rows carry, with
/// `functional == EMPYREAN_JOINT_FUNCTIONAL_SKY`.
///
/// The sky covariance is already published in this ABI's OUTPUT basis — the
/// facade's `to_degrees()` scaled the joint, so its angular rows/columns are in
/// degrees. Its packed triangle is therefore copied straight through with NO
/// further angular congruence (the opposite of [`orbit_uncertainty_to_c`],
/// whose engine joint is in radians and needs the rad→deg scaling). The row's
/// bare 6×6 `covariance` field is this joint's `state_block()`; the joint
/// beside it carries the `kind` tag and the self-describing layout the bare
/// matrix cannot. The returned `tri` is heap-allocated and must be released
/// with [`free_packed_joint`].
pub(crate) fn sky_uncertainty_to_c(
    unc: &Uncertainty<Spherical<AU, Degrees>>,
) -> EmpyreanPackedJoint {
    packed_joint_from_parts(
        unc.joint().layout().bits(),
        unc.joint().width() as u8,
        unc.joint().tri().to_vec(),
        crate::propagate::covariance_kind_to_u8(unc.kind()),
        unc.quality(),
        unc.functional().wire_discriminant(),
        unc.provenance(),
    )
}

/// Assemble the owned C packed joint from its already-marshaled parts: the
/// layout bitsets, the packed lower triangle **in this ABI's output basis**
/// (the caller has applied any representation scaling), the kind / functional
/// wire bytes, the live quality, and the provenance (whose asserted-source
/// companion is derived here). The triangle is copied into a library-owned
/// heap buffer released by [`free_packed_joint`]; nothing is recomputed.
///
/// Shared by [`orbit_uncertainty_to_c`] (state joints, rad→deg scaled) and
/// [`sky_uncertainty_to_c`] (sky joints, already in degrees) so both emit a
/// byte-identical wire struct for the same inputs.
fn packed_joint_from_parts(
    layout_bits: (u32, u32, u32),
    width: u8,
    tri_out_basis: Vec<f64>,
    kind: u8,
    quality: Quality,
    functional: u8,
    provenance: Provenance,
) -> EmpyreanPackedJoint {
    let (layout_present, layout_considered, layout_marginalized) = layout_bits;
    // The provenance and its asserted-source companion are read off the same
    // engine value: an asserted moment carries its source (written straight
    // from the engine's `AssertedSource::wire_discriminant`), a derived moment
    // carries the `_NONE` sentinel. Nothing is defaulted — the source rides the
    // wire so a re-fed asserted joint round-trips rather than being refused.
    let asserted_source = match provenance {
        Provenance::Asserted { source } => source.wire_discriminant(),
        Provenance::Derived => wire_tags::EMPYREAN_JOINT_ASSERTED_SOURCE_NONE,
    };
    let tri_boxed: Box<[f64]> = tri_out_basis.into_boxed_slice();
    let tri_len = tri_boxed.len();
    let tri = Box::into_raw(tri_boxed) as *const f64;
    let (quality_wire, cov_rank, cov_null_axes, cov_min_eig_corr) = match quality {
        Quality::PositiveDefinite => (0u8, 0u8, 0u32, f64::NAN),
        Quality::PositiveSemidefinite { rank, null_axes } => (1, rank, null_axes, f64::NAN),
        Quality::Indefinite { min_eig_corr } => (2, 0, 0, min_eig_corr),
        Quality::Repaired { min_eig_corr } => (3, 0, 0, min_eig_corr),
    };
    EmpyreanPackedJoint {
        layout_present,
        layout_considered,
        layout_marginalized,
        width,
        tri,
        tri_len,
        kind,
        quality: quality_wire,
        cov_rank,
        cov_null_axes,
        cov_min_eig_corr,
        functional,
        provenance: provenance.wire_discriminant(),
        asserted_source,
    }
}

/// Map a resolved Cartesian [`Uncertainty`] into the owned C packed joint.
///
/// This is the `UncertaintyQuery → OrbitUncertainty → C` step of the output
/// path: a lazy query resolved at the row's OWN delivered kind
/// (`UncertaintyQuery::covariance`, which builds a Linear row linearly and a
/// SecondOrder row at second order and never downgrades), or one entry of a
/// Cartesian covariance series, is an owned `Uncertainty<Cartesian<AU>>`.
/// Wrapping it as the representation-erased [`OrbitUncertainty`] lets the one
/// marshaler serialize it — the resolved state covariance is published as the
/// single packed joint, with its kind / quality / functional / provenance tags
/// read straight off the value. The returned `tri` is heap-allocated and must
/// be released with [`free_packed_joint`].
pub(crate) fn cartesian_uncertainty_to_c(unc: Uncertainty<Cartesian<AU>>) -> EmpyreanPackedJoint {
    orbit_uncertainty_to_c(&OrbitUncertainty::Cartesian(Box::new(unc)))
}

/// Release the library-owned triangle hanging off one
/// [`EmpyreanPackedJoint`] and leave it absent. Idempotent.
///
/// # Safety
///
/// `tri` must be a buffer [`orbit_uncertainty_to_c`] produced and not yet
/// freed, of length `tri_len`.
pub(crate) unsafe fn free_packed_joint(joint: &mut EmpyreanPackedJoint) {
    if !joint.tri.is_null() && joint.tri_len > 0 {
        let slice = unsafe { std::slice::from_raw_parts_mut(joint.tri as *mut f64, joint.tri_len) };
        drop(unsafe { Box::from_raw(slice as *mut [f64]) });
    }
    joint.tri = std::ptr::null();
    joint.tri_len = 0;
}

/// Build an engine [`OrbitUncertainty`] from one caller-supplied
/// [`EmpyreanPackedJoint`] — the INPUT dual of [`orbit_uncertainty_to_c`].
///
/// The construction goes through the engine facade (the distribution
/// depends on `empyrean-core` only, so the representation markers and the
/// `Derivation`/`Uncertainty` constructor the direct path needs are not
/// nameable here): the facade validates the layout bitsets, retypes the
/// packed triangle to the orbit's representation, and seeds a
/// [`Derivation::Linear`] prior — an input prior carries no higher-order
/// payload, and the engine re-derives the joint's `quality`/`kind`, so those
/// wire columns are output-only.
///
/// The caller's triangle is in the C degree basis (angular state rows in
/// degrees, matching the flat `CoordinateState.covariance`); it is scaled to
/// the engine's radian basis here by [`scale_tri_angular`] (the exact inverse
/// of [`orbit_uncertainty_to_c`]'s rad→deg scaling), so a degree → engine →
/// degree round trip returns the supplied joint. A layout that violates the
/// axis-table invariants, a triangle length that disagrees with the width, or
/// a retired / unknown `functional` / `provenance` discriminant is refused by
/// name rather than defaulted.
pub(crate) fn c_to_orbit_uncertainty(
    joint: &EmpyreanPackedJoint,
    representation: CoordinateRepresentation,
    basis: Basis,
) -> Result<OrbitUncertainty<AU>, AbiError> {
    // The triangle is borrowed read-only from the caller for the call. A
    // null pointer here means a joint was requested with no data behind it,
    // which is a loud argument error (not a wire refusal — the facade never
    // sees a null, it takes a slice) so it carries no pinned code and takes
    // the boundary's default.
    if joint.tri.is_null() {
        return Err(AbiError::from(
            "packed joint: tri is null but a joint of non-zero width was supplied".to_string(),
        ));
    }
    let tri_deg = unsafe { std::slice::from_raw_parts(joint.tri, joint.tri_len) };
    // Scale the caller's degree-basis triangle into the engine's radian
    // basis on the angular state rows/columns of this representation (the
    // diagonal congruence D Σ Dᵀ), leaving position/velocity and the
    // parameter axes unscaled. The inverse of the output marshaler, so a
    // degree → engine → degree round trip returns the supplied joint.
    let mut tri = tri_deg.to_vec();
    scale_tri_angular(
        &mut tri,
        representation.angular_element_indices(),
        std::f64::consts::PI / 180.0,
    );
    // The engine facade (core's `feature/uncertainty-from-wire`, landing on
    // the branch the local `[patch]` points at) validates the layout
    // bitsets, cross-checks `width` against BOTH the triangle length and the
    // layout-implied width (two distinct refusals), retypes the packed
    // triangle to the orbit's representation, refuses a retired `kind`
    // discriminant (the third-order 2 and the reserved 6), an unknown /
    // retired `functional` or `provenance` discriminant, and a basis / row
    // epoch disagreement — each BY NAME through a `UncertaintyWireError`
    // variant — and seeds a `Derivation::Linear` prior (an input prior
    // carries no higher-order payload; the engine re-derives quality). This
    // is the one unresolved import until that branch lands.
    //
    // `asserted_source` is the provenance companion: the joint carries its
    // asserted source in the `asserted_source` field, so when the provenance
    // discriminant is Asserted that byte is handed to the facade as
    // `Some(source)` (the facade refuses the `_NONE` sentinel or an unknown
    // source as `UnknownAssertedSource`, −49, by name); on any other provenance
    // `None` is passed and the field is ignored. Nothing is defaulted, and a
    // re-fed asserted (SBDB / external) joint round-trips rather than being
    // refused. `row_epoch` is the basis epoch: an input prior's covariance is
    // stated at the coordinate's own instant, so the facade's epoch check is
    // satisfied by construction.
    //
    // The variant-named message is preserved AND the `UncertaintyWireError`
    // variant is mapped to its own FFI return code (−40…−49) via
    // `AbiError::coded`, so the refusal reaches every push boundary — direct
    // call and file writer alike — as that code, never collapsed to one.
    let asserted_source = if joint.provenance == wire_tags::EMPYREAN_JOINT_PROVENANCE_ASSERTED {
        Some(joint.asserted_source)
    } else {
        None
    };
    empyrean_core::uncertainty::orbit_uncertainty_from_wire(
        representation,
        joint.layout_present,
        joint.layout_considered,
        joint.layout_marginalized,
        joint.width as usize,
        &tri,
        basis,
        joint.kind,
        joint.functional,
        joint.provenance,
        asserted_source,
        basis.epoch,
    )
    .map_err(|e| {
        AbiError::coded(
            uncertainty_wire_error_code(&e),
            format!("packed joint refused by the engine: {e}"),
        )
    })
}

/// Do two state-block covariances (both in the engine radian basis) agree to
/// a generous relative tolerance? Used only to catch a caller who supplied a
/// packed joint AND a flat state covariance that contradict each other — a
/// disagreement GATE, not a published value, so the band is wide enough that
/// the two independent deg→rad scaling paths never trip it on float noise.
fn state_blocks_agree(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> bool {
    for i in 0..6 {
        for j in 0..6 {
            let (x, y) = (a[i][j], b[i][j]);
            if (x - y).abs() > 1e-9 * (x.abs() + y.abs()) + 1e-18 {
                return false;
            }
        }
    }
    true
}

/// Append one C-ABI orbit's coordinate to an `Orbits<AU>` batch with its
/// uncertainty in one step.
///
/// The packed joint on [`EmpyreanOrbit::covariance`] is THE one authority for
/// the orbit's covariance (the whole \\((6+P)\\)-wide matrix). `state_cov` is
/// the flat state-6×6 view the convert tuple carries from
/// `CoordinateState.covariance` — for the coordinate-conversion API, not a
/// second home.
///
/// - Packed joint present (`width > 0`): it is built into an
///   [`OrbitUncertainty`] in the coordinate's representation and basis and
///   attached. If a flat `state_cov` was ALSO supplied, its state block must
///   agree with the joint's (both radian-basis here) or the caller gave
///   contradictory input — refused BY NAME rather than silently preferring
///   one.
/// - Packed joint absent: the flat `state_cov` (if any) is the orbit's only
///   uncertainty and is attached as-is; otherwise the orbit carries none.
///
/// The coordinate itself is stored in the engine's radian basis.
pub(crate) fn push_orbit_with_joint(
    orbits: &mut Orbits<AU>,
    orbit_id: String,
    coords: Coordinates<AU, Degrees>,
    state_cov: Option<OrbitUncertainty<AU>>,
    orbit: &EmpyreanOrbit,
) -> Result<(), AbiError> {
    let uncertainty = if orbit.covariance.width > 0 {
        let basis = Basis {
            frame: coords.frame(),
            origin: coords.origin(),
            epoch: coords.time().mjd_tdb(),
        };
        let joint_ou = c_to_orbit_uncertainty(&orbit.covariance, coords.representation(), basis)?;
        if let Some(flat) = state_cov.as_ref()
            && !state_blocks_agree(&joint_ou.state_block(), &flat.state_block())
        {
            return Err(AbiError::from(
                "orbit covariance: the packed joint and the flat state covariance disagree \
                 on the state block"
                    .to_string(),
            ));
        }
        Some(joint_ou)
    } else {
        state_cov
    };
    orbits
        .push(orbit_id, coords.into_radians())
        .map_err(|e| e.to_string())?;
    let i = orbits.len() - 1;
    orbits.set_uncertainty(i, uncertainty);
    Ok(())
}

#[cfg(test)]
mod mixture_tally_tests {
    use super::*;
    use empyrean_core::propagation::{MixtureComponents, Shift};

    /// Build a `Derivation::Mixture` from the facade types with the chosen
    /// per-axis tallies — no fixture produces a real split at this engine pin,
    /// so the marshal's reachability is proven structurally, from the same
    /// facade surface the reference channel reads.
    fn mixture(
        n_components_total: usize,
        weight_delivered: f64,
        n_failed: Option<usize>,
        n_unresolved: Option<usize>,
        n_curvature_refused: Option<usize>,
        n_sky_linearization_refused: Option<usize>,
    ) -> Derivation<Cartesian<AU>> {
        Derivation::Mixture(MixtureComponents::<Cartesian<AU>> {
            components: Vec::new().into_boxed_slice(),
            component_epoch: 0.0,
            collapse_shift: Shift::zeros(6),
            weight_delivered,
            n_failed,
            n_unresolved,
            n_curvature_refused,
            n_sky_linearization_refused,
            retained_depth: Some(2),
            n_components_total,
            split_criterion: Some(1.5),
        })
    }

    /// A mixture derivation reports the engine's six tallies through the
    /// marshal helper, flag set. Mutation: replacing `mc.n_components_total as
    /// u32` with `0` in [`mixture_tally_from_derivation`] drops the count and
    /// the first assert goes red.
    #[test]
    fn mixture_tally_from_derivation_fills_every_tally() {
        let d = mixture(7, 0.9, Some(2), Some(1), Some(3), Some(4));
        let (has, t) = mixture_tally_from_derivation(&d);
        assert_eq!(has, 1, "a mixture row is present");
        assert_eq!(t.n_components_total, 7);
        assert_eq!(t.weight_delivered, 0.9);
        assert_eq!(t.n_failed, 2);
        assert_eq!(t.n_unresolved, 1);
        assert_eq!(t.n_curvature_refused, 3);
        assert_eq!(t.n_sky_linearization_refused, 4);
    }

    /// An engine `None` on an optional count becomes `-1` (absent by name),
    /// never a fabricated `0`. Mutation: mapping `None` to `0` instead of `-1`
    /// in the marshal goes red here.
    #[test]
    fn mixture_tally_from_derivation_maps_engine_none_to_minus_one() {
        let d = mixture(5, 1.0, None, None, None, None);
        let (has, t) = mixture_tally_from_derivation(&d);
        assert_eq!(has, 1);
        assert_eq!(t.n_components_total, 5);
        assert_eq!(t.n_failed, -1, "a not-carried count is -1, not 0");
        assert_eq!(t.n_unresolved, -1);
        assert_eq!(t.n_curvature_refused, -1);
        assert_eq!(
            t.n_sky_linearization_refused, -1,
            "sky-linearization refusals are engine-None here, carried as -1"
        );
    }

    /// A non-mixture derivation leaves the tally absent at the marshal — the
    /// flag clear and every count `-1`. Mutation: returning `(1, …)` or a
    /// zero-filled tally for the `_` arm goes red.
    #[test]
    fn mixture_tally_from_derivation_leaves_non_mixture_absent() {
        for d in [
            Derivation::<Cartesian<AU>>::Linear,
            Derivation::<Cartesian<AU>>::SigmaPoint {
                mean_offset: None,
                n_nodes: 13,
            },
        ] {
            let (has, t) = mixture_tally_from_derivation(&d);
            assert_eq!(has, 0, "a non-mixture row carries no tally");
            assert_eq!(t.n_components_total, 0);
            assert_eq!(t.n_failed, -1);
            assert_eq!(t.n_sky_linearization_refused, -1);
        }
    }
}
