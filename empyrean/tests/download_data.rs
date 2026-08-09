//! Regression: `download_data` actually provisions a usable data directory and
//! is idempotent — it is no longer a no-op resolver.
//!
//! Ignored by default: the provision path reaches NAIF / MPC / PDS
//! unconditionally, so a default `cargo test` stays offline-safe. Run the
//! behavioral form explicitly with `cargo test --test download_data --
//! --ignored` on a networked machine (the core `provision_data` pair
//! follows the same convention).

#[test]
#[ignore = "reaches the network (NAIF/MPC/PDS downloads); run with -- --ignored"]
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
