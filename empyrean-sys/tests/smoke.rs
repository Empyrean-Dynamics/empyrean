//! Smoke test: verify the FFI linkage is correct by calling functions
//! that don't require any data files.
use empyrean_sys::*;
use std::ffi::CStr;

#[test]
fn last_error_is_empty_before_any_call() {
    unsafe {
        let err_ptr = empyrean_last_error();
        assert!(!err_ptr.is_null());
        let err_str = CStr::from_ptr(err_ptr).to_str().expect("valid utf8");
        assert_eq!(err_str, "");
    }
}

#[test]
fn context_new_with_missing_files_returns_null_and_sets_error() {
    unsafe {
        let bogus_spk = b"/nonexistent/de440.bsp\0";
        let bogus_gm = b"/nonexistent/gm.tpc\0";
        let ctx = empyrean_context_new_minimal(
            bogus_spk.as_ptr() as *const i8,
            bogus_gm.as_ptr() as *const i8,
        );
        assert!(ctx.is_null(), "expected null context for missing files");

        let err_ptr = empyrean_last_error();
        let err_str = CStr::from_ptr(err_ptr).to_str().expect("valid utf8");
        assert!(!err_str.is_empty(), "expected a non-empty error message");
    }
}

#[test]
fn rejection_constants_are_visible_through_ffi() {
    // Pin the wire values: changing these is a downstream-breaking
    // change. Adaptive and CMC2003 must have distinct codes so the
    // Python / wrapper layers can decode the per-obs reason.
    assert_eq!(EMPYREAN_REJECTION_ACCEPTED, 0);
    assert_eq!(EMPYREAN_REJECTION_ADAPTIVE, 4);
    assert_eq!(EMPYREAN_REJECTION_CMC2003, 6);
    assert_eq!(EMPYREAN_REJECTION_NOT_EVALUATED, -1);
    assert_ne!(EMPYREAN_REJECTION_ADAPTIVE, EMPYREAN_REJECTION_CMC2003);
}

#[test]
fn rejection_kind_constants_are_visible_through_ffi() {
    // Default kind is 0 = Adaptive so existing C callers that
    // zero-init EmpyreanRejectionConfig keep working.
    assert_eq!(EMPYREAN_REJECTION_KIND_ADAPTIVE, 0);
    assert_eq!(EMPYREAN_REJECTION_KIND_CMC2003, 1);
}

#[test]
fn rejection_config_struct_has_cmc2003_fields() {
    // Default-init must work (C side typically zero-inits).
    let c: EmpyreanRejectionConfig = Default::default();
    // Bindgen exposes fields as plain accessors; just touch them so
    // the build fails if any of the new fields disappear.
    let _ = c.kind;
    let _ = c.chi2_rej;
    let _ = c.chi2_rec;
}

#[test]
fn noise_model_constants_are_visible_through_ffi() {
    // Wire values for the observation error law. Gaussian is 0 so a
    // zero-initialized EmpyreanNoiseModel is the pre-noise-model default.
    assert_eq!(EMPYREAN_NOISE_MODEL_GAUSSIAN, 0);
    assert_eq!(EMPYREAN_NOISE_MODEL_STUDENT_T, 1);
    assert_eq!(EMPYREAN_NOISE_MODEL_NIGHT_SCALE, 2);
    assert_eq!(EMPYREAN_COVARIANCE_INFORMATION_OBSERVED, 0);
    assert_eq!(EMPYREAN_COVARIANCE_INFORMATION_EXPECTED, 1);
    // The blunder rejection reason and gate.
    assert_eq!(EMPYREAN_REJECTION_BLUNDER, 16);
    assert_eq!(EMPYREAN_REJECTION_KIND_BLUNDER, 2);
    // Rejection enable tri-state (zero-init = engine default), matching the
    // EMPYREAN_ORIGIN_SWITCHING_* DEFAULT/ON/OFF convention.
    assert_eq!(EMPYREAN_REJECTION_ENABLED_DEFAULT, 0);
    assert_eq!(EMPYREAN_REJECTION_ENABLED_ON, 1);
    assert_eq!(EMPYREAN_REJECTION_ENABLED_OFF, 2);
    // Sigma provenance tags are contiguous and distinct; UNKNOWN is its own
    // code, never folded into NOT_WEIGHTED.
    assert_eq!(EMPYREAN_SIGMA_SOURCE_REPORTED, 0);
    assert_eq!(EMPYREAN_SIGMA_SOURCE_NOT_WEIGHTED, 4);
    assert_eq!(EMPYREAN_SIGMA_SOURCE_UNKNOWN, 5);
    assert_ne!(
        EMPYREAN_SIGMA_SOURCE_UNKNOWN,
        EMPYREAN_SIGMA_SOURCE_NOT_WEIGHTED
    );
}

#[test]
fn od_error_codes_are_visible_and_the_slot_carries_the_field() {
    // NONE is 0 (delivered / not-applicable); the variant codes start at 1
    // and run contiguously to 43.
    assert_eq!(EMPYREAN_OD_ERROR_NONE, 0);
    assert_eq!(EMPYREAN_OD_ERROR_NOT_CARTESIAN, 1);
    assert_eq!(EMPYREAN_OD_ERROR_NOISE_MODEL_REJECTION_UNSET, 39);
    assert_eq!(EMPYREAN_OD_ERROR_NOISE_MODEL_IRLS_NOT_CONVERGED, 43);
    assert_ne!(
        EMPYREAN_OD_ERROR_NOISE_MODEL_REJECTION_UNSET,
        EMPYREAN_OD_ERROR_OBSERVED_INFORMATION_NOT_POSITIVE_DEFINITE
    );
    // The batch slot carries the specific OD code beside the failure kind.
    let slot: EmpyreanODObjectResult = Default::default();
    let _: i32 = slot.od_error_code;
    let _: i32 = slot.error_code;
}

#[test]
fn noise_model_and_result_structs_carry_the_new_fields() {
    // Touch the new struct fields so a dropped field fails the build,
    // and confirm the config structs still zero-init through Default.
    let nm: EmpyreanNoiseModel = Default::default();
    assert_eq!(nm.tag, 0);
    let _ = nm.nu;

    let w: EmpyreanWeightingConfig = Default::default();
    let _ = w.noise_model.tag;
    let _ = w.covariance_information;

    let layer: EmpyreanWeightingLayer = Default::default();
    let _ = layer.has_noise_model;
    let _ = layer.noise_model.nu;

    // The rejection tri-state is an unsigned byte; zero-init = engine
    // default (EMPYREAN_REJECTION_ENABLED_DEFAULT).
    let r: EmpyreanRejectionConfig = Default::default();
    let _: u8 = r.enabled;
    assert_eq!(r.enabled, EMPYREAN_REJECTION_ENABLED_DEFAULT as u8);

    let row: EmpyreanObservationResult = Default::default();
    let _ = row.noise_model;
    let _ = row.noise_model_nu;
    let _ = row.robust_weight;
    let _ = row.sigma_eff_ra_arcsec;
    let _ = row.sigma_source;
    let _ = row.sigma_scale_applied;

    let s: EmpyreanResidualSummary = Default::default();
    let _ = s.n_eff;
    let _ = s.expected_median_d2;
    let _ = s.expected_tail;

    let od: EmpyreanODResult = Default::default();
    let _ = od.rejection_kind;
    let _ = od.rejection_enabled;
}

#[test]
fn the_loaded_engine_reports_the_abi_version_this_crate_compiled_against() {
    // `dlsym` matches on symbol name, not signature, so a stale
    // `libempyrean` resolves every symbol and then reads the caller's
    // arguments through the wrong shape. Since ABI 3 that includes two
    // functions whose parameter *lists* changed, where the mismatch
    // writes a struct through an integer the caller passed by value.
    // `lib()` asserts the handshake at open time; this pins that the
    // engine actually reachable from this build agrees.
    let loaded = unsafe { empyrean_abi_version() };
    assert_eq!(
        loaded, EMPYREAN_ABI_VERSION,
        "the loaded libempyrean is built to a different ABI version than these bindings"
    );
}

/// The generated files carry the C ABI's doc text verbatim, and nothing that
/// compiles them ever reads it: `rustc`, `clippy` and `rustfmt` are all happy
/// with a doc comment full of garbage, and the CI doc gate runs only on
/// `-p empyrean`. So a generator that writes the file through a Latin-1 round
/// trip silently ships mojibake to docs.rs — which is exactly what happened
/// once, turning every em-dash in `shims.rs` into `â` on the public v3 surface.
///
/// A double-encoded character is a Latin-1 high byte followed by one or more
/// C1 continuation bytes; a correctly encoded `—` is a single `char` above
/// U+00FF and can never look like one. Check for the shape, not for a list of
/// known-bad strings.
#[test]
fn the_generated_sources_are_not_double_encoded() {
    fn mojibake_runs(src: &str) -> Vec<String> {
        let chars: Vec<char> = src.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            // A UTF-8 lead byte re-encoded as a `char` lands in U+00C0..=U+00FF.
            if ('\u{00c0}'..='\u{00ff}').contains(&chars[i]) {
                let mut j = i + 1;
                // Continuation bytes re-encode into U+0080..=U+00BF.
                while j < chars.len() && ('\u{0080}'..='\u{00bf}').contains(&chars[j]) {
                    j += 1;
                }
                if j > i + 1 {
                    out.push(chars[i..j].iter().collect());
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
        out
    }

    for (name, src) in [
        ("empyrean-sys/src/shims.rs", include_str!("../src/shims.rs")),
        (
            "empyrean-sys/src/bindings.rs",
            include_str!("../src/bindings.rs"),
        ),
    ] {
        let runs = mojibake_runs(src);
        assert!(
            runs.is_empty(),
            "{name} contains {} double-encoded run(s) — the file was written through a \
             Latin-1 round trip. Regenerate it reading the source as UTF-8. First few: {:?}",
            runs.len(),
            &runs[..runs.len().min(5)]
        );
    }
}
