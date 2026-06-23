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
