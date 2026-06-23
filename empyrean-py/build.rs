// PyO3 `extension-module` crates don't link libpython — the host
// interpreter resolves the `_Py*` symbols at import time. macOS's
// linker rejects undefined symbols by default, so a plain `cargo
// build` of the cdylib fails to link. Allow dynamic lookup, scoped to
// this crate's cdylib output only (the rest of the workspace keeps
// strict link-time symbol checking). maturin sets the equivalent
// itself when building the wheel.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo::rustc-cdylib-link-arg=-undefined");
        println!("cargo::rustc-cdylib-link-arg=dynamic_lookup");
    }
}
