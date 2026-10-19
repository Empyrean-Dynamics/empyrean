use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let header_out = PathBuf::from(&crate_dir).join("../include/empyrean.h");

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    // macOS cdylib link layout fix.
    //
    // Measured: the default rustc cdylib on aarch64-apple-darwin (minos 11.0)
    // is produced with the old-style LC_DYLD_INFO_ONLY bind/rebase layout and
    // an odd indirect-symbol count (251), which leaves the LINKEDIT string
    // pool at offset 4 mod 8. Apple's newer static linker (ld-27037) reads that
    // misaligned string pool and rejects the dylib with "mis-aligned LINKEDIT
    // string pool" when a C program is linked against it at build time (`cc`).
    // dlopen is unaffected, so the Python runtime loader never saw this, but a
    // consumer that LINKS the dylib does.
    //
    // -fixup_chains switches the image to LC_DYLD_CHAINED_FIXUPS, which packs
    // the indirect-symbol table to an even count and realigns LINKEDIT on an
    // 8-byte boundary; the dylib then both links and loads. The flag is cdylib-
    // scoped so it never reaches the engine's own rlib compilations.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-Wl,-fixup_chains");
    }

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
