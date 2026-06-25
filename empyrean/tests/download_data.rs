//! Regression: `download_data` actually provisions a usable data directory and
//! is idempotent — it is no longer a no-op resolver.

#[test]
fn download_data_provisions_and_is_idempotent() {
    let dir = empyrean::download_data(None).expect("download_data must provision the data dir");
    assert!(
        dir.join("de440.bsp").exists(),
        "download_data must leave the core kernels on disk (de440.bsp under {})",
        dir.display(),
    );

    // The provisioned directory loads cleanly with no further downloads.
    empyrean::Context::from_data_dir(Some(&dir))
        .expect("from_data_dir over a download_data'd directory must load");

    // Idempotent: a second call returns the same directory and re-uses the
    // already-present files without error.
    let dir2 = empyrean::download_data(None).expect("download_data must be idempotent");
    assert_eq!(dir, dir2);
}
