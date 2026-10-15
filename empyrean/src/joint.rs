//! The packed joint covariance — the one covariance carrier.
//!
//! # One matrix, one home
//!
//! A fit over the state and \\(P\\) solved parameters produces one
//! \\((6+P) \times (6+P)\\) matrix. Earlier releases split that matrix
//! across four carriers — a flat \\(6 \times 6\\) state block, a Marsden
//! \\(3 \times 3\\), a state↔Marsden border, and a wide cross for
//! everything else — and left a caller to reassemble them. The off-diagonal
//! blocks were the first casualty of that split: a block-diagonal covariance
//! asserts that the data which produced the state and the data which produced
//! \\(A_2\\) were independent, when they are the same observations through the
//! same fit, and the engine's *propagated* joint has non-zero
//! state↔parameter columns even when the input was block-diagonal.
//!
//! [`PackedJoint`] carries the whole matrix as one object: the axis
//! [`layout`](PackedJoint::layout), the joint [`width`](PackedJoint::width),
//! the packed lower triangle [`tri`](PackedJoint::tri), and the tags that name
//! what the matrix describes. There is no second home to disagree with it, so
//! chaining a leg reads one field whatever produced the state.
//!
//! # Mirrored, not re-exported
//!
//! [`PackedJoint`] mirrors the C ABI's `EmpyreanPackedJoint` by name and
//! layout rather than being the engine's own type. This crate depends on
//! `empyrean-sys` and nothing else — the C ABI is the only surface it is
//! permitted to call through — so identity with the engine's covariance type
//! is not available to it. Parity is a contract these types keep, testable
//! rather than compiler-enforced: the same joint marshaled across the boundary
//! and read back delivers the same triangle.

use crate::error::{Error, Result};

/// What a fit did with one parameter axis.
///
/// The three answers are different operations with different mathematics, and
/// using one where another is meant is silent — each produces a well-formed
/// covariance.
///
/// There is deliberately **no** `From<bool>` and no `Default`. An axis's
/// disposition is a modelling statement, and a conversion that turned `true`
/// into `Solved` would let a call site inherit one rather than state it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamDisposition {
    /// Marginalized out of the prior. Contributes nothing and changes no
    /// number.
    Fixed,
    /// Estimated from the data: occupies a solved slot and comes back with a
    /// posterior variance.
    Solved,
    /// Not estimated, but uncertain — its prior uncertainty reaches the
    /// posterior through its measurement partials (Schmidt–Kalman consider
    /// analysis; Tapley, Byron D., Schutz, Bob E., and Born, George H.,
    /// *Statistical Orbit Determination*, Elsevier Academic Press, 2004,
    /// ch. 6).
    ///
    /// Not a safety margin: under an uncorrelated prior the correction
    /// strictly widens the posterior, but with cross terms between the
    /// considered axis and the solved ones the correction is sign-indefinite
    /// and the posterior can come back **tighter**.
    Considered,
}

impl ParamDisposition {
    /// Whether this axis is estimated.
    pub const fn is_solved(self) -> bool {
        matches!(self, ParamDisposition::Solved)
    }

    /// Whether this axis inflates the posterior through consider analysis.
    pub const fn is_considered(self) -> bool {
        matches!(self, ParamDisposition::Considered)
    }

    /// The canonical wire tag: `"fixed"`, `"solved"`, `"considered"`.
    pub const fn as_tag(self) -> &'static str {
        match self {
            ParamDisposition::Fixed => "fixed",
            ParamDisposition::Solved => "solved",
            ParamDisposition::Considered => "considered",
        }
    }

    /// Parse the canonical wire tag. An unrecognized tag is an error, never a
    /// default.
    pub fn from_tag(tag: &str) -> Result<Self> {
        match tag {
            "fixed" => Ok(ParamDisposition::Fixed),
            "solved" => Ok(ParamDisposition::Solved),
            "considered" => Ok(ParamDisposition::Considered),
            other => Err(Error::invalid_input(format!(
                "unknown parameter disposition {other:?}; expected \"fixed\", \
                 \"solved\" or \"considered\""
            ))),
        }
    }

    pub(crate) const fn to_ffi(self) -> u8 {
        match self {
            ParamDisposition::Fixed => empyrean_sys::EMPYREAN_PARAM_FIXED,
            ParamDisposition::Solved => empyrean_sys::EMPYREAN_PARAM_SOLVED,
            ParamDisposition::Considered => empyrean_sys::EMPYREAN_PARAM_CONSIDERED,
        }
    }

    pub(crate) fn from_ffi(v: u8, field: &str) -> Result<Self> {
        match v {
            empyrean_sys::EMPYREAN_PARAM_FIXED => Ok(ParamDisposition::Fixed),
            empyrean_sys::EMPYREAN_PARAM_SOLVED => Ok(ParamDisposition::Solved),
            empyrean_sys::EMPYREAN_PARAM_CONSIDERED => Ok(ParamDisposition::Considered),
            other => Err(Error::invalid_input(format!(
                "the engine returned {other} for the {field} disposition, which is not \
                 fixed (0), solved (1) or considered (2)"
            ))),
        }
    }
}

/// The joint's axis layout — three bitsets over the engine's axis table.
///
/// [`present`](Self::present) names the axes that have a column in the joint
/// (the state six plus one bit per solved parameter);
/// [`considered`](Self::considered) is the subset of those that were carried by
/// consider analysis rather than solved; [`marginalized`](Self::marginalized)
/// names axes whose identity is present but whose column is absent, disjoint
/// from `present`. The layout names the parameter each column belongs to, so a
/// column's meaning does not depend on how many *other* parameters the orbit
/// declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AxisLayout {
    /// Axes with a column in the joint, as a bitset over the axis table.
    pub present: u32,
    /// Considered axes — a subset of [`present`](Self::present).
    pub considered: u32,
    /// Axes whose identity is present but whose column is absent; disjoint from
    /// [`present`](Self::present).
    pub marginalized: u32,
}

/// The definiteness of a packed joint's matrix.
///
/// Read on output only — the engine re-derives definiteness when it ingests a
/// prior, so an input joint's quality is not consulted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointQuality {
    /// All eigenvalues positive within round-off.
    PositiveDefinite,
    /// Rank-deficient but PSD: `rank` columns carry signal and `null_axes`
    /// names the null directions as a bitset over the axis table.
    PositiveSemidefinite {
        /// The numerical rank.
        rank: u8,
        /// The null-axis bitset.
        null_axes: u32,
    },
    /// At least one meaningfully negative eigenvalue; `min_eig_corr` is the
    /// most-negative correlation-space eigenvalue found.
    Indefinite {
        /// The most-negative correlation-space eigenvalue.
        min_eig_corr: f64,
    },
    /// Repaired to PSD; `min_eig_corr` is the correlation-space eigenvalue
    /// *before* the repair.
    Repaired {
        /// The most-negative correlation-space eigenvalue before the repair.
        min_eig_corr: f64,
    },
}

impl JointQuality {
    pub(crate) fn from_ffi(q: u8, rank: u8, null_axes: u32, min_eig_corr: f64) -> Result<Self> {
        Ok(match q as u32 {
            empyrean_sys::EMPYREAN_JOINT_QUALITY_POSITIVE_DEFINITE => Self::PositiveDefinite,
            empyrean_sys::EMPYREAN_JOINT_QUALITY_POSITIVE_SEMIDEFINITE => {
                Self::PositiveSemidefinite { rank, null_axes }
            }
            empyrean_sys::EMPYREAN_JOINT_QUALITY_INDEFINITE => Self::Indefinite { min_eig_corr },
            empyrean_sys::EMPYREAN_JOINT_QUALITY_REPAIRED => Self::Repaired { min_eig_corr },
            other => {
                return Err(Error::invalid_input(format!(
                    "the C ABI returned {other} for the joint quality tag, which is not \
                     positive-definite (0), positive-semidefinite (1), indefinite (2) or \
                     repaired (3)"
                )));
            }
        })
    }

    /// The `(quality, cov_rank, cov_null_axes, cov_min_eig_corr)` wire tuple.
    pub(crate) fn to_ffi(self) -> (u8, u8, u32, f64) {
        match self {
            Self::PositiveDefinite => (
                empyrean_sys::EMPYREAN_JOINT_QUALITY_POSITIVE_DEFINITE as u8,
                0,
                0,
                f64::NAN,
            ),
            Self::PositiveSemidefinite { rank, null_axes } => (
                empyrean_sys::EMPYREAN_JOINT_QUALITY_POSITIVE_SEMIDEFINITE as u8,
                rank,
                null_axes,
                f64::NAN,
            ),
            Self::Indefinite { min_eig_corr } => (
                empyrean_sys::EMPYREAN_JOINT_QUALITY_INDEFINITE as u8,
                0,
                0,
                min_eig_corr,
            ),
            Self::Repaired { min_eig_corr } => (
                empyrean_sys::EMPYREAN_JOINT_QUALITY_REPAIRED as u8,
                0,
                0,
                min_eig_corr,
            ),
        }
    }
}

/// Which second moment a packed joint's axes describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointFunctional {
    /// The Cartesian (or representation-native) state second moment.
    State,
    /// The sky-plane (right-ascension / declination) second moment.
    Sky,
    /// The radar delay / Doppler second moment.
    DelayDoppler,
}

impl JointFunctional {
    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        Ok(match v as u32 {
            empyrean_sys::EMPYREAN_JOINT_FUNCTIONAL_STATE => Self::State,
            empyrean_sys::EMPYREAN_JOINT_FUNCTIONAL_SKY => Self::Sky,
            empyrean_sys::EMPYREAN_JOINT_FUNCTIONAL_DELAY_DOPPLER => Self::DelayDoppler,
            other => {
                return Err(Error::invalid_input(format!(
                    "the C ABI returned {other} for the joint functional tag, which is not \
                     state (0), sky (1) or delay-Doppler (2)"
                )));
            }
        })
    }

    pub(crate) fn to_u8(self) -> u8 {
        (match self {
            Self::State => empyrean_sys::EMPYREAN_JOINT_FUNCTIONAL_STATE,
            Self::Sky => empyrean_sys::EMPYREAN_JOINT_FUNCTIONAL_SKY,
            Self::DelayDoppler => empyrean_sys::EMPYREAN_JOINT_FUNCTIONAL_DELAY_DOPPLER,
        }) as u8
    }
}

/// Whether a packed joint's second moment was derived by the engine or
/// asserted at ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointProvenance {
    /// The engine derived the moment from a fit or a propagation.
    Derived,
    /// The moment was asserted by the caller at ingestion.
    Asserted,
}

impl JointProvenance {
    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        Ok(match v as u32 {
            empyrean_sys::EMPYREAN_JOINT_PROVENANCE_DERIVED => Self::Derived,
            empyrean_sys::EMPYREAN_JOINT_PROVENANCE_ASSERTED => Self::Asserted,
            other => {
                return Err(Error::invalid_input(format!(
                    "the C ABI returned {other} for the joint provenance tag, which is not \
                     derived (0) or asserted (1)"
                )));
            }
        })
    }

    pub(crate) fn to_u8(self) -> u8 {
        (match self {
            Self::Derived => empyrean_sys::EMPYREAN_JOINT_PROVENANCE_DERIVED,
            Self::Asserted => empyrean_sys::EMPYREAN_JOINT_PROVENANCE_ASSERTED,
        }) as u8
    }
}

/// The external source that asserted a joint's second moment.
///
/// Carried only when [`provenance`](PackedJoint::provenance) is
/// [`Asserted`](JointProvenance::Asserted); a derived joint has no source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertedSource {
    /// The JPL Small-Body Database.
    SBDB,
    /// Another external orbit file or tool.
    ExternalFile,
}

impl AssertedSource {
    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        Ok(match v as u32 {
            empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB => Self::SBDB,
            empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_EXTERNAL_FILE => Self::ExternalFile,
            other => {
                return Err(Error::invalid_input(format!(
                    "the C ABI returned {other} for the joint asserted-source tag, which is \
                     not SBDB (0) or external-file (1)"
                )));
            }
        })
    }

    pub(crate) fn to_u8(self) -> u8 {
        (match self {
            Self::SBDB => empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB,
            Self::ExternalFile => empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_EXTERNAL_FILE,
        }) as u8
    }
}

/// A fitted or propagated joint covariance over the state and solved
/// parameters, as one packed object.
///
/// The matrix is stored as its packed lower triangle [`tri`](Self::tri) —
/// `width·(width+1)/2` doubles, entry \\((i, j)\\) for \\(j \le i\\) at index
/// \\(i(i+1)/2 + j\\). The leading \\(6 \times 6\\) is the state block; the
/// angular state rows and columns are in **degrees** for the angular
/// representations (unscaled for Cartesian), and the parameter axes
/// (Marsden / DT / AMRAT / thrust) are always unscaled.
///
/// This is the one authority for an orbit's covariance. A flat \\(6 \times 6\\)
/// state covariance read off a result is a derived read-only view of this
/// joint's state block ([`state_block`](Self::state_block)), never a second,
/// independently settable home.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedJoint {
    /// The axis layout — which columns are present, considered, or
    /// marginalized.
    pub layout: AxisLayout,
    /// The joint width `N = 6 + popcount(layout.present)`.
    pub width: u8,
    /// The packed lower triangle, `width·(width+1)/2` doubles.
    pub tri: Vec<f64>,
    /// How the covariance was derived — see
    /// [`CovarianceKind`](crate::CovarianceKind).
    pub kind: crate::propagate::CovarianceKind,
    /// The definiteness of the matrix (read on output only).
    pub quality: JointQuality,
    /// Which second moment the axes describe.
    pub functional: JointFunctional,
    /// Whether the moment was derived by the engine or asserted at ingestion.
    pub provenance: JointProvenance,
    /// Which external source asserted the moment. `Some` **iff**
    /// [`provenance`](Self::provenance) is [`Asserted`](JointProvenance::Asserted);
    /// a derived joint carries `None`. A mismatch — an asserted joint with no
    /// source, or a source on a derived joint — is refused at the FFI boundary
    /// rather than defaulted, so an SBDB-sourced orbit re-fed to the engine
    /// keeps its source instead of being silently re-tagged.
    pub asserted_source: Option<AssertedSource>,
}

impl PackedJoint {
    /// The leading \\(6 \times 6\\) state block as a dense symmetric matrix — a
    /// derived read-only view of [`tri`](Self::tri).
    ///
    /// The state block always occupies the first six rows and columns whatever
    /// parameters follow, so this reads off the leading triangle and mirrors it
    /// across the diagonal. It is a view: there is no setter, because the joint
    /// is the one authority and a settable \\(6 \times 6\\) beside it could
    /// disagree with the columns it borders.
    // Packed-triangle index arithmetic with a mirrored dual assignment; an
    // iterator form would obscure the `(i, j)` / `(j, i)` symmetry.
    #[allow(clippy::needless_range_loop)]
    pub fn state_block(&self) -> [[f64; 6]; 6] {
        let mut m = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let idx = i * (i + 1) / 2 + j;
                // Direct index, not a zero-filling `get`: construction
                // (`from_ffi`) refuses a joint whose triangle is shorter than its
                // packed length, so the leading 21 entries are always present. A
                // short triangle is a by-name refusal there, never a silent zero
                // read here.
                let v = self.tri[idx];
                m[i][j] = v;
                m[j][i] = v;
            }
        }
        m
    }

    /// Read a joint back from its C wire form. A width-0 struct is the absent
    /// joint and yields `None`.
    ///
    /// # Safety
    ///
    /// `c.tri` must be null or an array of `c.tri_len` doubles live for the
    /// duration of the call.
    pub(crate) unsafe fn from_ffi(c: &empyrean_sys::EmpyreanPackedJoint) -> Result<Option<Self>> {
        if c.width == 0 {
            return Ok(None);
        }
        // The triangle is borrowed read-only from the library-owned result for
        // the call and copied into an owned Vec — the wrapper's value outlives
        // the result it came from.
        let tri = if c.tri.is_null() || c.tri_len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(c.tri, c.tri_len) }.to_vec()
        };
        // A present joint (width ≥ 6) must carry its whole packed triangle —
        // `width·(width+1)/2` entries. A shorter one is refused here, by name,
        // rather than tolerated: the derived state-block view reads the leading
        // 21 entries, so a short triangle would make it zero-fill a covariance
        // cell silently. Validating at construction keeps `state_block` infallible
        // and means a malformed joint never exists, symmetric with the Python
        // decode (which refuses `tri.size < 21`).
        let packed_len = (c.width as usize) * (c.width as usize + 1) / 2;
        if c.width < 6 || tri.len() < packed_len {
            return Err(Error::invalid_input(format!(
                "packed joint has width {} and {} triangle entries; a present joint \
                 needs width ≥ 6 and a full {}-entry lower triangle (a shorter \
                 triangle would zero-fill the derived state block)",
                c.width,
                tri.len(),
                packed_len,
            )));
        }
        let provenance = JointProvenance::from_u8(c.provenance)?;
        // The source companion is meaningful ONLY for an asserted joint: there
        // it MUST name a recognized source (the `_NONE` sentinel or an unknown
        // value is refused by name). On any other provenance the byte is a
        // don't-care — the C ABI's zero/memset default is indistinguishable
        // from an explicit source there — so it is read as "no source" rather
        // than validated, matching the C facade, which hands the companion to
        // the engine only when the provenance is Asserted. The construction
        // side ([`to_ffi_borrowing`]) is where a derived-joint-with-a-source
        // mismatch is refused, because there the source is a stated value.
        let asserted_source = match provenance {
            JointProvenance::Asserted => Some(AssertedSource::from_u8(c.asserted_source)?),
            JointProvenance::Derived => None,
        };
        Ok(Some(Self {
            layout: AxisLayout {
                present: c.layout_present,
                considered: c.layout_considered,
                marginalized: c.layout_marginalized,
            },
            width: c.width,
            tri,
            kind: crate::propagate::CovarianceKind::from_u8(c.kind)?,
            quality: JointQuality::from_ffi(
                c.quality,
                c.cov_rank,
                c.cov_null_axes,
                c.cov_min_eig_corr,
            )?,
            functional: JointFunctional::from_u8(c.functional)?,
            provenance,
            asserted_source,
        }))
    }

    /// The `asserted_source` wire byte, enforcing the pairing invariant: an
    /// asserted joint MUST carry a source, a derived joint MUST NOT. A
    /// mismatch is refused rather than written as a defaulted byte the engine
    /// would then reject opaquely.
    fn asserted_source_wire(&self) -> Result<u8> {
        match (self.provenance, self.asserted_source) {
            (JointProvenance::Asserted, Some(s)) => Ok(s.to_u8()),
            (JointProvenance::Derived, None) => {
                Ok(empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_NONE as u8)
            }
            (JointProvenance::Asserted, None) => Err(Error::invalid_input(
                "an asserted joint must carry its asserted source, but none was set".to_string(),
            )),
            (JointProvenance::Derived, Some(_)) => Err(Error::invalid_input(
                "a derived joint must not carry an asserted source".to_string(),
            )),
        }
    }

    /// Build the C wire form, borrowing the triangle from `tri`.
    ///
    /// `tri` must be this joint's own triangle held in storage that outlives
    /// every use of the returned struct — the caller's keepalive owns it. The
    /// returned struct holds a raw pointer into `tri` and copies nothing. A
    /// provenance/source mismatch is refused (see
    /// [`asserted_source_wire`](Self::asserted_source_wire)).
    pub(crate) fn to_ffi_borrowing(
        &self,
        tri: &[f64],
    ) -> Result<empyrean_sys::EmpyreanPackedJoint> {
        let (quality, cov_rank, cov_null_axes, cov_min_eig_corr) = self.quality.to_ffi();
        let asserted_source = self.asserted_source_wire()?;
        Ok(empyrean_sys::EmpyreanPackedJoint {
            layout_present: self.layout.present,
            layout_considered: self.layout.considered,
            layout_marginalized: self.layout.marginalized,
            width: self.width,
            tri: if tri.is_empty() {
                std::ptr::null()
            } else {
                tri.as_ptr()
            },
            tri_len: tri.len(),
            kind: self.kind.to_u8(),
            quality,
            cov_rank,
            cov_null_axes,
            cov_min_eig_corr,
            functional: self.functional.to_u8(),
            provenance: self.provenance.to_u8(),
            asserted_source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagate::CovarianceKind;

    /// A synthetic width-9 Cartesian joint: 6 state + 3 Marsden. Cartesian has
    /// no angular rows, so nothing is scaled at any boundary and
    /// `to_bits`-level equality is admissible.
    fn width9_cartesian() -> PackedJoint {
        let n = 9;
        let tri: Vec<f64> = (0..n * (n + 1) / 2).map(|i| 1.0 + i as f64 * 0.5).collect();
        PackedJoint {
            // 3 parameter axes (the Marsden A1/A2/A3), so width = 6 + 3 = 9.
            layout: AxisLayout {
                present: 0b111,
                considered: 0,
                marginalized: 0,
            },
            width: 9,
            tri,
            kind: CovarianceKind::Linear,
            quality: JointQuality::PositiveDefinite,
            functional: JointFunctional::State,
            provenance: JointProvenance::Derived,
            asserted_source: None,
        }
    }

    #[test]
    fn a_width9_joint_round_trips_bit_identical() {
        // Synthetic Cartesian: the wrapper applies no basis scaling, so a
        // to_ffi → from_ffi round trip is a pure copy and must be bit-identical.
        let pj = width9_cartesian();
        let tri = pj.tri.clone();
        let c = pj.to_ffi_borrowing(&tri).expect("derived joint marshals");
        let back = unsafe { PackedJoint::from_ffi(&c) }
            .expect("round trips")
            .expect("non-zero width is present");
        assert_eq!(back.width, pj.width);
        assert_eq!(back.layout, pj.layout);
        assert_eq!(back.kind, pj.kind);
        assert_eq!(back.functional, pj.functional);
        assert_eq!(back.provenance, pj.provenance);
        assert_eq!(back.asserted_source, pj.asserted_source);
        for (a, b) in back.tri.iter().zip(pj.tri.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "triangle must round-trip bit-identical"
            );
        }
    }

    #[test]
    fn an_asserted_joint_round_trips_its_source() {
        // Mirror of the C surface: an asserted SBDB joint carries its source
        // across to_ffi → from_ffi with provenance AND source intact.
        let mut pj = width9_cartesian();
        pj.provenance = JointProvenance::Asserted;
        pj.asserted_source = Some(AssertedSource::SBDB);
        let tri = pj.tri.clone();
        let c = pj.to_ffi_borrowing(&tri).expect("asserted joint marshals");
        assert_eq!(
            c.asserted_source as u32,
            empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_SBDB,
            "the SBDB source is written on the wire, not the none sentinel"
        );
        let back = unsafe { PackedJoint::from_ffi(&c) }
            .expect("round trips")
            .expect("present");
        assert_eq!(back.provenance, JointProvenance::Asserted);
        assert_eq!(back.asserted_source, Some(AssertedSource::SBDB));
    }

    #[test]
    fn a_provenance_source_mismatch_is_refused_by_variant() {
        // (a) an asserted joint with no source is refused on the way out …
        let mut pj = width9_cartesian();
        pj.provenance = JointProvenance::Asserted;
        pj.asserted_source = None;
        let tri = pj.tri.clone();
        let err = pj
            .to_ffi_borrowing(&tri)
            .expect_err("an asserted joint with no source must be refused");
        assert!(err.message.contains("asserted"));
        // (b) … and a source on a derived joint is refused too.
        let mut pj = width9_cartesian();
        pj.provenance = JointProvenance::Derived;
        pj.asserted_source = Some(AssertedSource::SBDB);
        let err = pj
            .to_ffi_borrowing(&pj.tri.clone())
            .expect_err("a derived joint must not carry a source");
        assert!(err.message.contains("derived"));
        // (c) reading an asserted C joint carrying the none sentinel (no
        // source) is refused — an asserted joint requires a real source.
        let mut c = empyrean_sys::EmpyreanPackedJoint::default();
        let tri = [1.0f64; 21];
        c.width = 6;
        c.layout_present = 0b111_111;
        c.tri = tri.as_ptr();
        c.tri_len = tri.len();
        c.provenance = empyrean_sys::EMPYREAN_JOINT_PROVENANCE_ASSERTED as u8;
        c.asserted_source = empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_NONE as u8;
        let err = unsafe { PackedJoint::from_ffi(&c) }
            .expect_err("an asserted joint with the none sentinel source is refused");
        assert!(err.message.contains("asserted-source"));
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn the_flat_6x6_view_equals_the_joints_state_block() {
        let pj = width9_cartesian();
        let block = pj.state_block();
        // The state block reads the leading 6×6 of the packed triangle and
        // mirrors it, so (i, j) and (j, i) are both tri[i(i+1)/2 + j].
        for i in 0..6 {
            for j in 0..=i {
                let v = pj.tri[i * (i + 1) / 2 + j];
                assert_eq!(block[i][j], v);
                assert_eq!(block[j][i], v);
            }
        }
    }

    #[test]
    fn a_short_triangle_is_refused_at_construction() {
        // A width-9 joint declares a 45-entry packed triangle. A wire struct
        // carrying fewer is refused BY NAME at `from_ffi` — never accepted and
        // then silently zero-filled when the derived 6×6 state block is read.
        // Symmetric with the Python decode, which refuses `tri.size < 21`.
        let mut c = empyrean_sys::EmpyreanPackedJoint::default();
        let tri = [1.0f64; 20]; // short even of the 21-entry state block
        c.width = 9;
        c.layout_present = 0b111;
        c.tri = tri.as_ptr();
        c.tri_len = tri.len();
        let err = unsafe { PackedJoint::from_ffi(&c) }
            .expect_err("a short triangle must be refused, not zero-filled");
        assert!(
            err.message.contains("triangle") && err.message.contains("zero-fill"),
            "the refusal names the short triangle: {}",
            err.message
        );
    }

    #[test]
    fn the_retired_third_order_kind_is_refused_by_variant() {
        // Kind tag 2 was third-order, retired with the engine. There is no
        // `CovarianceKind::ThirdOrder` variant, and a wire struct carrying it is
        // refused by the kind decoder rather than defaulted — the wrapper has no
        // engine, so this mirrors the C surface's `-45` refusal at readback.
        let mut c = empyrean_sys::EmpyreanPackedJoint::default();
        let tri = [1.0f64; 21]; // width 6
        c.width = 6;
        c.layout_present = 0b111_111;
        c.tri = tri.as_ptr();
        c.tri_len = tri.len();
        c.kind = 2; // retired third-order discriminant
        let err = unsafe { PackedJoint::from_ffi(&c) }
            .expect_err("a retired kind discriminant must be refused, never defaulted");
        assert!(err.message.contains("covariance kind"));
    }

    #[test]
    fn dispositions_round_trip_and_reject_unknowns() {
        for d in [
            ParamDisposition::Fixed,
            ParamDisposition::Solved,
            ParamDisposition::Considered,
        ] {
            assert_eq!(ParamDisposition::from_tag(d.as_tag()).unwrap(), d);
            assert_eq!(ParamDisposition::from_ffi(d.to_ffi(), "x").unwrap(), d);
        }
        assert!(ParamDisposition::from_tag("true").is_err());
        assert!(ParamDisposition::from_ffi(3, "marsden").is_err());
    }
}
