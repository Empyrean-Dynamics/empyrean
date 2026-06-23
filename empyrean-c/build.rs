use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let header_out = PathBuf::from(&crate_dir).join("../include/empyrean.h");

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(
            cbindgen::Config::from_file(PathBuf::from(&crate_dir).join("cbindgen.toml"))
                .expect("Failed to read cbindgen.toml"),
        )
        .generate();

    match bindings {
        Ok(bindings) => {
            bindings.write_to_file(&header_out);
            println!(
                "cargo:warning=Generated C header at {}",
                header_out.display()
            );
        }
        Err(e) => {
            // Don't fail the build if header generation has issues -
            // the cdylib itself compiles independently. Warn instead.
            println!("cargo:warning=cbindgen failed: {e}. Header not regenerated.");
        }
    }
}
