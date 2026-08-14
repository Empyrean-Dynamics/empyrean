//! The joint solved-parameter covariance — the cross terms beyond the
//! diagonal blocks.
//!
//! # What this is for
//!
//! A fit over the state and \\(P\\) parameters produces one
//! \\((6+P) \times (6+P)\\) matrix. Its diagonal blocks — the
//! \\(6 \times 6\\) state covariance, the Marsden \\(3 \times 3\\), a DT
//! variance, an AMRAT variance, a per-segment thrust \\(3 \times 3\\) —
//! have always crossed this boundary. The **off-diagonal** blocks are
//! what these types carry.
//!
//! Leaving them behind is not a conservative simplification. A
//! block-diagonal covariance asserts that the data which produced the
//! state and the data which produced \\(A_2\\) were independent, when
//! they are the same observations through the same fit; and the engine's
//! *propagated* joint has non-zero state↔parameter columns even when the
//! input was block-diagonal, because propagation itself generates that
//! correlation. Chaining legs on the diagonal alone therefore reports a
//! tighter uncertainty than the propagation supports.
//!
//! # The four homes
//!
//! One covariance entry belongs to exactly one place, and supplying it
//! in two is refused rather than merged:
//!
//! | Block | Home |
//! |---|---|
//! | state ↔ state | [`CoordinateState::covariance`](crate::CoordinateState::covariance) |
//! | \\(A_i \leftrightarrow A_j\\) | [`Orbit::ng_covariance`](crate::Orbit::ng_covariance) |
//! | state ↔ \\(A_i\\) | [`CoordinateState::non_grav_cross`](crate::CoordinateState::non_grav_cross) |
//! | \\(\Delta v_i \leftrightarrow \Delta v_j\\), same segment | [`ThrustParams::correction_covariances`](crate::ThrustParams) |
//! | everything else | [`WideCross`] |
//!
//! # Mirrored, not re-exported
//!
//! [`ParamColumn`] and [`WideCross`] mirror the engine's types by name
//! and semantics rather than being the same types. This crate depends on
//! `empyrean-sys` and nothing else — the C ABI is the only surface it is
//! permitted to call through — so identity with the engine's types is
//! not available to it. Parity is a contract these types keep, and it is
//! testable rather than compiler-enforced: the same joint marshaled
//! through both channels must deliver the same σ.

use std::collections::BTreeMap;

/// Which solved parameter a cross term refers to.
///
/// Cross terms name the **parameter**, never a column index. Which
/// column a parameter occupies depends on which *other* parameters the
/// orbit declares — adding an SRP AMRAT shifts the thrust columns by one
/// — so an index recorded against one orbit is wrong against the next,
/// and the failure is silent: every number finite, every gate passed,
/// one parameter's correlations attached to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParamColumn {
    /// Marsden non-gravitational coefficient \\(A_{i+1}\\), `i` in
    /// `0..3`.
    Marsden(usize),
    /// The Marsden non-grav time delay \\(\Delta T\\).
    Dt,
    /// The SRP area-to-mass ratio.
    Amrat,
    /// One component of one thrust \\(\Delta v\\) correction segment.
    Thrust {
        /// Which **declared** correction segment — an index into
        /// [`ThrustParams::correction_covariances`](crate::ThrustParams),
        /// the space the engine emits its thrust columns in.
        ///
        /// Declared, not solved. A considered or fixed burn sitting
        /// between two solved ones is a routine case, and tagging by
        /// solved index would then place a burn's cross terms on a
        /// different burn's column.
        segment: usize,
        /// Which Cartesian component of that segment's \\(\Delta v\\),
        /// `0..3`.
        component: usize,
    },
}

impl ParamColumn {
    /// The canonical string form, matching the engine's own rendering:
    /// `A1`, `A2`, `A3`, `DT`, `AMRAT`, `thrust[0].x`.
    ///
    /// This is the tag the Python and file-format channels carry, so it
    /// is contract rather than presentation.
    pub fn as_tag(self) -> String {
        match self {
            ParamColumn::Marsden(i) => format!("A{}", i + 1),
            ParamColumn::Dt => "DT".to_string(),
            ParamColumn::Amrat => "AMRAT".to_string(),
            ParamColumn::Thrust { segment, component } => {
                let axis = ["x", "y", "z"].get(component).copied().unwrap_or("?");
                format!("thrust[{segment}].{axis}")
            }
        }
    }

    /// Parse the canonical string form.
    ///
    /// The inverse of [`as_tag`](Self::as_tag), and the single parse
    /// point for every string-tagged channel. An unrecognized tag is an
    /// error, never a default — a mis-parsed tag would attach one
    /// parameter's correlations to another.
    pub fn from_tag(tag: &str) -> crate::error::Result<Self> {
        let bad = || {
            crate::error::Error::invalid_input(format!(
                "unknown parameter-column tag {tag:?}; expected A1/A2/A3, DT, AMRAT, \
                 or thrust[<segment>].<x|y|z>"
            ))
        };
        match tag {
            "A1" => return Ok(ParamColumn::Marsden(0)),
            "A2" => return Ok(ParamColumn::Marsden(1)),
            "A3" => return Ok(ParamColumn::Marsden(2)),
            "DT" => return Ok(ParamColumn::Dt),
            "AMRAT" => return Ok(ParamColumn::Amrat),
            _ => {}
        }
        let rest = tag.strip_prefix("thrust[").ok_or_else(bad)?;
        let (seg, rest) = rest.split_once(']').ok_or_else(bad)?;
        let axis = rest.strip_prefix('.').ok_or_else(bad)?;
        let component = match axis {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => return Err(bad()),
        };
        Ok(ParamColumn::Thrust {
            segment: seg.parse::<usize>().map_err(|_| bad())?,
            component,
        })
    }

    /// The FFI tag for this identity.
    pub(crate) fn to_ffi(self) -> empyrean_sys::EmpyreanParamColumn {
        let (kind, index, segment, component) = match self {
            ParamColumn::Marsden(i) => {
                (empyrean_sys::EMPYREAN_PARAM_COLUMN_MARSDEN, i as u32, 0, 0)
            }
            ParamColumn::Dt => (empyrean_sys::EMPYREAN_PARAM_COLUMN_DT, 0, 0, 0),
            ParamColumn::Amrat => (empyrean_sys::EMPYREAN_PARAM_COLUMN_AMRAT, 0, 0, 0),
            ParamColumn::Thrust { segment, component } => (
                empyrean_sys::EMPYREAN_PARAM_COLUMN_THRUST,
                0,
                segment as u32,
                component as u32,
            ),
        };
        empyrean_sys::EmpyreanParamColumn {
            kind,
            index,
            segment,
            component,
        }
    }

    /// Read an identity back from its FFI tag.
    pub(crate) fn from_ffi(c: &empyrean_sys::EmpyreanParamColumn) -> crate::error::Result<Self> {
        match c.kind {
            empyrean_sys::EMPYREAN_PARAM_COLUMN_MARSDEN => {
                Ok(ParamColumn::Marsden(c.index as usize))
            }
            empyrean_sys::EMPYREAN_PARAM_COLUMN_DT => Ok(ParamColumn::Dt),
            empyrean_sys::EMPYREAN_PARAM_COLUMN_AMRAT => Ok(ParamColumn::Amrat),
            empyrean_sys::EMPYREAN_PARAM_COLUMN_THRUST => Ok(ParamColumn::Thrust {
                segment: c.segment as usize,
                component: c.component as usize,
            }),
            other => Err(crate::error::Error::invalid_input(format!(
                "the engine returned an unknown parameter-column kind {other}"
            ))),
        }
    }
}

/// Cross-covariance terms beyond the state+Marsden \\(9 \times 9\\):
/// state↔DT, state↔AMRAT, state↔\\(\Delta v\\), and every mixed
/// parameter pair.
///
/// # Keyed by identity, so order is not contract
///
/// Entries are stored against their [`ParamColumn`], and the engine
/// resolves placement when the joint is assembled. Two carriers holding
/// the same entries are equal regardless of the order they were built
/// in.
///
/// # Partition
///
/// The state↔Marsden border and the Marsden \\(3 \times 3\\) live on the
/// coordinate and the orbit respectively, and an intra-segment thrust
/// pair lives on that segment's own \\(3 \times 3\\). Supplying any of
/// those here is refused by the engine rather than merged — the terms
/// would then have two homes that could disagree. Cross-segment thrust
/// pairs have no other home and do belong here.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WideCross {
    state: BTreeMap<ParamColumn, [f64; 6]>,
    params: BTreeMap<(ParamColumn, ParamColumn), f64>,
}

impl WideCross {
    /// An empty carrier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the carrier holds no entries at all.
    ///
    /// Note this is about entry COUNT, not about values: an entry whose
    /// six values are all zero is a supplied zero correlation and makes
    /// this return `false`. Absence and a supplied zero are different
    /// claims, and only the first is expressed by omitting the entry.
    pub fn is_empty(&self) -> bool {
        self.state.is_empty() && self.params.is_empty()
    }

    /// Set the state↔parameter cross column for one parameter — the
    /// 6-vector of covariances between the six state elements and it.
    ///
    /// Rows are in the coordinate's own element order, representation,
    /// frame **and angular unit** — the same basis as the
    /// \\(6 \times 6\\) they border, so degrees for the angular rows of
    /// a Cometary or Keplerian state.
    pub fn set_state_cross(&mut self, column: ParamColumn, values: [f64; 6]) {
        self.state.insert(column, values);
    }

    /// The state↔parameter cross column for one parameter, if present.
    pub fn state_cross(&self, column: ParamColumn) -> Option<&[f64; 6]> {
        self.state.get(&column)
    }

    /// Every state↔parameter column, ascending by parameter.
    pub fn state_crosses(&self) -> impl Iterator<Item = (ParamColumn, &[f64; 6])> {
        self.state
            .iter()
            .map(|(k, v): (&ParamColumn, &[f64; 6])| (*k, v))
    }

    /// Set a parameter↔parameter cross term.
    ///
    /// Symmetric: \\((a, b)\\) and \\((b, a)\\) are one entry, and the
    /// key is canonicalized so setting one overwrites the other rather
    /// than creating a second term that could disagree with the first.
    pub fn set_param_cross(&mut self, a: ParamColumn, b: ParamColumn, value: f64) {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.params.insert(key, value);
    }

    /// A parameter↔parameter cross term, in either order.
    pub fn param_cross(&self, a: ParamColumn, b: ParamColumn) -> Option<f64> {
        let key = if a <= b { (a, b) } else { (b, a) };
        self.params.get(&key).copied()
    }

    /// Every parameter↔parameter term, ascending by canonical key.
    pub fn param_crosses(&self) -> impl Iterator<Item = (ParamColumn, ParamColumn, f64)> + '_ {
        self.params
            .iter()
            .map(|((a, b), v): (&(ParamColumn, ParamColumn), &f64)| (*a, *b, *v))
    }

    /// Marshal into the two FFI side arrays.
    ///
    /// The returned vectors are the storage the FFI struct borrows and
    /// must outlive the call — the wrapper's keepalive owns them.
    pub(crate) fn to_ffi_arrays(
        &self,
    ) -> (
        Vec<empyrean_sys::EmpyreanStateParamCross>,
        Vec<empyrean_sys::EmpyreanParamPairCross>,
    ) {
        let state = self
            .state_crosses()
            .map(|(column, values)| empyrean_sys::EmpyreanStateParamCross {
                column: column.to_ffi(),
                values: *values,
            })
            .collect();
        let pairs = self
            .param_crosses()
            .map(|(a, b, value)| empyrean_sys::EmpyreanParamPairCross {
                a: a.to_ffi(),
                b: b.to_ffi(),
                value,
            })
            .collect();
        (state, pairs)
    }

    /// Read a carrier back from the library-owned FFI arrays.
    ///
    /// # Safety
    ///
    /// `state` / `pairs` must be null or arrays of the given lengths,
    /// live for the duration of the call.
    pub(crate) unsafe fn from_ffi_arrays(
        state: *const empyrean_sys::EmpyreanStateParamCross,
        n_state: usize,
        pairs: *const empyrean_sys::EmpyreanParamPairCross,
        n_pairs: usize,
    ) -> crate::error::Result<Option<Self>> {
        if (state.is_null() || n_state == 0) && (pairs.is_null() || n_pairs == 0) {
            return Ok(None);
        }
        let mut out = WideCross::new();
        if !state.is_null() {
            for e in unsafe { std::slice::from_raw_parts(state, n_state) } {
                out.set_state_cross(ParamColumn::from_ffi(&e.column)?, e.values);
            }
        }
        if !pairs.is_null() {
            for e in unsafe { std::slice::from_raw_parts(pairs, n_pairs) } {
                out.set_param_cross(
                    ParamColumn::from_ffi(&e.a)?,
                    ParamColumn::from_ffi(&e.b)?,
                    e.value,
                );
            }
        }
        Ok(Some(out))
    }
}

/// What a fit did with one parameter axis.
///
/// The three answers are different operations with different
/// mathematics, and using one where another is meant is silent — each
/// produces a well-formed covariance.
///
/// There is deliberately **no** `From<bool>` and no `Default`. An axis's
/// disposition is a modelling statement, and a conversion that turned
/// `true` into `Solved` would let a call site inherit one rather than
/// state it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamDisposition {
    /// Marginalized out of the prior. Contributes nothing and changes no
    /// number.
    Fixed,
    /// Estimated from the data: occupies a solved slot and comes back
    /// with a posterior variance.
    Solved,
    /// Not estimated, but uncertain — its prior uncertainty reaches the
    /// posterior through its measurement partials (Schmidt–Kalman
    /// consider analysis; Tapley, Byron D., Schutz, Bob E., and Born,
    /// George H., *Statistical Orbit Determination*, Elsevier Academic
    /// Press, 2004, ch. 6).
    ///
    /// Not a safety margin: under an uncorrelated prior the correction
    /// strictly widens the posterior, but with cross terms between the
    /// considered axis and the solved ones the correction is
    /// sign-indefinite and the posterior can come back **tighter**.
    Considered,
}

impl ParamDisposition {
    /// Whether this axis is estimated.
    pub const fn is_solved(self) -> bool {
        matches!(self, ParamDisposition::Solved)
    }

    /// Whether this axis inflates the posterior through consider
    /// analysis.
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

    /// Parse the canonical wire tag. An unrecognized tag is an error,
    /// never a default.
    pub fn from_tag(tag: &str) -> crate::error::Result<Self> {
        match tag {
            "fixed" => Ok(ParamDisposition::Fixed),
            "solved" => Ok(ParamDisposition::Solved),
            "considered" => Ok(ParamDisposition::Considered),
            other => Err(crate::error::Error::invalid_input(format!(
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

    pub(crate) fn from_ffi(v: u8, field: &str) -> crate::error::Result<Self> {
        match v {
            empyrean_sys::EMPYREAN_PARAM_FIXED => Ok(ParamDisposition::Fixed),
            empyrean_sys::EMPYREAN_PARAM_SOLVED => Ok(ParamDisposition::Solved),
            empyrean_sys::EMPYREAN_PARAM_CONSIDERED => Ok(ParamDisposition::Considered),
            other => Err(crate::error::Error::invalid_input(format!(
                "the engine returned {other} for the {field} disposition, which is not \
                 fixed (0), solved (1) or considered (2)"
            ))),
        }
    }
}

/// A fitted or propagated joint's cross terms, as the library hands them
/// back.
///
/// The same shape rides an OD result's orbit and every propagated state,
/// so leg chaining reads one field whatever produced the state:
/// `determine → propagate` and `propagate → propagate` are the same
/// copy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JointCovariance {
    /// The \\(6 \times 3\\) state↔Marsden border, in the same basis and
    /// units as the \\(6 \times 6\\) it accompanies. `None` when the
    /// producer had none — never a zero block, which would read as a
    /// supplied zero correlation.
    pub non_grav_cross: Option<[[f64; 3]; 6]>,
    /// The wide carrier. `None` when there is none.
    pub wide_cross: Option<WideCross>,
}

impl JointCovariance {
    /// Whether this carries nothing at all.
    pub fn is_empty(&self) -> bool {
        self.non_grav_cross.is_none() && self.wide_cross.as_ref().is_none_or(WideCross::is_empty)
    }

    /// Read from the FFI shape.
    ///
    /// # Safety
    ///
    /// `cov`'s array pointers must be null or valid for their counts.
    pub(crate) unsafe fn from_ffi(
        cov: &empyrean_sys::EmpyreanOrbitCovariance,
    ) -> crate::error::Result<Self> {
        Ok(Self {
            non_grav_cross: (cov.has_non_grav_cross != 0).then_some(cov.non_grav_cross),
            wide_cross: unsafe {
                WideCross::from_ffi_arrays(
                    cov.state_param_cross,
                    cov.n_state_param_cross,
                    cov.param_pair_cross,
                    cov.n_param_pair_cross,
                )
            }?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tag_round_trips() {
        let mut cases = vec![ParamColumn::Dt, ParamColumn::Amrat];
        cases.extend((0..3).map(ParamColumn::Marsden));
        for s in 0..3 {
            cases.extend((0..3).map(|c| ParamColumn::Thrust {
                segment: s,
                component: c,
            }));
        }
        for c in cases {
            let tag = c.as_tag();
            assert_eq!(
                ParamColumn::from_tag(&tag).expect("round trips"),
                c,
                "tag {tag:?} must parse back to the identity that rendered it"
            );
            assert_eq!(
                ParamColumn::from_ffi(&c.to_ffi()).expect("ffi round trips"),
                c
            );
        }
    }

    #[test]
    fn the_tags_match_the_engines_rendering() {
        // These strings are contract — they are what the file formats
        // and the Python tables carry.
        assert_eq!(ParamColumn::Marsden(0).as_tag(), "A1");
        assert_eq!(ParamColumn::Marsden(2).as_tag(), "A3");
        assert_eq!(ParamColumn::Dt.as_tag(), "DT");
        assert_eq!(ParamColumn::Amrat.as_tag(), "AMRAT");
        assert_eq!(
            ParamColumn::Thrust {
                segment: 0,
                component: 0
            }
            .as_tag(),
            "thrust[0].x"
        );
        assert_eq!(
            ParamColumn::Thrust {
                segment: 2,
                component: 2
            }
            .as_tag(),
            "thrust[2].z"
        );
    }

    #[test]
    fn an_unknown_tag_is_an_error_never_a_default() {
        for bad in [
            "A4",
            "A0",
            "dt",
            "thrust[0].w",
            "thrust[].x",
            "thrust0.x",
            "",
        ] {
            assert!(
                ParamColumn::from_tag(bad).is_err(),
                "{bad:?} must not parse — a mis-parsed tag attaches one parameter's \
                 correlations to another"
            );
        }
    }

    #[test]
    fn a_pair_is_symmetric_and_stored_once() {
        let mut w = WideCross::new();
        w.set_param_cross(ParamColumn::Amrat, ParamColumn::Dt, 1.0);
        assert_eq!(
            w.param_cross(ParamColumn::Dt, ParamColumn::Amrat),
            Some(1.0)
        );
        // Setting the swapped form overwrites rather than adding a
        // second term that could disagree.
        w.set_param_cross(ParamColumn::Dt, ParamColumn::Amrat, 2.0);
        assert_eq!(w.param_crosses().count(), 1);
        assert_eq!(
            w.param_cross(ParamColumn::Amrat, ParamColumn::Dt),
            Some(2.0)
        );
    }

    #[test]
    fn entry_order_is_not_contract() {
        let a = {
            let mut w = WideCross::new();
            w.set_state_cross(ParamColumn::Dt, [1.0; 6]);
            w.set_state_cross(ParamColumn::Amrat, [2.0; 6]);
            w
        };
        let b = {
            let mut w = WideCross::new();
            w.set_state_cross(ParamColumn::Amrat, [2.0; 6]);
            w.set_state_cross(ParamColumn::Dt, [1.0; 6]);
            w
        };
        assert_eq!(a, b, "identity keying makes build order irrelevant");
    }

    #[test]
    fn a_supplied_zero_entry_is_not_absence() {
        let mut w = WideCross::new();
        w.set_state_cross(ParamColumn::Dt, [0.0; 6]);
        assert!(
            !w.is_empty(),
            "a supplied zero correlation is a claim; only omitting the entry means absent"
        );
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
        // `0`/`1` keep the meaning the pre-v4 booleans had.
        assert_eq!(
            ParamDisposition::from_ffi(0, "x").unwrap(),
            ParamDisposition::Fixed
        );
        assert_eq!(
            ParamDisposition::from_ffi(1, "x").unwrap(),
            ParamDisposition::Solved
        );
    }
}
