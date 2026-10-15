//! One by-name mapping from the wrapper's packed-joint / transform
//! refusals to CLI-level messages.
//!
//! The C ABI refuses a malformed covariance prior with a distinct code
//! (−40…−50), which the wrapper classifies as [`empyrean::PushError`].
//! The CLI surfaces each cause by name — one mapping, so a covariance a
//! command cannot push reads as what is wrong with it rather than as a
//! bare numeric code.
//!
//! Messages, not exit codes: the `determine` command reserves its
//! `0 / 3 / 4` process codes for per-object delivery outcomes
//! ([`crate::commands::determine::exit_code_for`]), so a push refusal is
//! an ordinary input error — it exits non-zero carrying its named cause,
//! never a code that would collide with a delivery outcome.

use anyhow::Error;
use empyrean::PushError;

/// A short, by-name label for each refusal axis — one arm per variant, no
/// shared fallback string.
pub fn label(push: PushError) -> &'static str {
    match push {
        PushError::LayoutInvalid => "covariance layout violates the axis-table invariants",
        PushError::TriangleLength => "covariance triangle length disagrees with its width",
        PushError::WidthMismatch => "covariance width disagrees with its layout",
        PushError::UnknownFunctional => "covariance functional discriminant is unknown or retired",
        PushError::UnknownProvenance => "covariance provenance discriminant is unknown or retired",
        PushError::RetiredDiscriminant => "covariance kind discriminant is retired (third order)",
        PushError::EpochMismatch => "covariance row epoch disagrees with the state epoch",
        PushError::EngineRefused => "the engine refused the covariance prior",
        PushError::UnsupportedInputKind => "the input covariance kind is not accepted on ingestion",
        PushError::UnknownAssertedSource => "an asserted covariance carries no recognized source",
        PushError::TransformCovarianceUnsupported => {
            "a covariance-bearing state was handed to the state-level transform; transform the \
             orbit so its joint rides the Jacobian"
        }
    }
}

/// Wrap a wrapper error under CLI `context`, naming the refusal axis when
/// the failure is a packed-joint / transform refusal; otherwise keep the
/// wrapper error (its own message and code) under `context`.
pub fn wrap<C>(err: empyrean::Error, context: C) -> Error
where
    C: std::fmt::Display + Send + Sync + 'static,
{
    match err.push_error() {
        Some(push) => Error::msg(format!("{context}: {} ({})", label(push), err.message)),
        None => Error::new(err).context(context),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const ALL: &[PushError] = &[
        PushError::LayoutInvalid,
        PushError::TriangleLength,
        PushError::WidthMismatch,
        PushError::UnknownFunctional,
        PushError::UnknownProvenance,
        PushError::RetiredDiscriminant,
        PushError::EpochMismatch,
        PushError::EngineRefused,
        PushError::UnsupportedInputKind,
        PushError::UnknownAssertedSource,
        PushError::TransformCovarianceUnsupported,
    ];

    /// Every variant maps to a distinct, non-empty label — the mapping is
    /// total and by-name, never a collapsed fallback.
    #[test]
    fn every_push_variant_has_a_distinct_label() {
        let mut seen = HashSet::new();
        for &p in ALL {
            let l = label(p);
            assert!(!l.is_empty(), "{p:?} has an empty label");
            assert!(seen.insert(l), "duplicate label for {p:?}: {l}");
        }
        assert_eq!(seen.len(), ALL.len());
    }

    /// A push refusal surfaces with its axis name and the wrapper message,
    /// under the CLI context.
    #[test]
    fn wrap_names_the_push_axis() {
        let err = empyrean::Error {
            code: PushError::TransformCovarianceUnsupported.code(),
            message: "detail".into(),
            missing_data_files: Vec::new(),
            orbit_index: None,
            orbit_id: None,
            epoch_mjd_tdb: None,
        };
        let s = format!("{:#}", wrap(err, "propagation failed"));
        assert!(s.contains("propagation failed"), "context lost: {s}");
        assert!(s.contains("transform the"), "axis name lost: {s}");
        assert!(s.contains("detail"), "wrapper message lost: {s}");
    }

    /// A non-push error keeps its own message under the context (no label).
    #[test]
    fn wrap_passes_through_a_non_push_error() {
        let err = empyrean::Error {
            code: -2,
            message: "missing data".into(),
            missing_data_files: Vec::new(),
            orbit_index: None,
            orbit_id: None,
            epoch_mjd_tdb: None,
        };
        let s = format!("{:#}", wrap(err, "load failed"));
        assert!(s.contains("load failed"), "context lost: {s}");
        assert!(s.contains("missing data"), "wrapper message lost: {s}");
    }
}
