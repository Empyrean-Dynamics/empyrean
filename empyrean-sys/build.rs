//! Build script for `empyrean-sys`.
//!
//! Resolves the absolute path of the prebuilt `libempyrean` shared library and
//! writes it to `$OUT_DIR/lib_path.rs` for the runtime loader. The library is
//! opened with `libloading` at run time — there is **no** link-time native
//! dependency, so no `install_name_tool` / `patchelf` / rpath / loader-path
//! environment is involved. Resolution order:
//!
//!   1. `EMPYREAN_LIB_DIR` — explicit override (offline / air-gapped / a
//!      locally built library).
//!   2. A sibling workspace build at `../target/release` (in-tree development),
//!      unless `EMPYREAN_FORCE_DOWNLOAD=1`.
//!   3. Download the prebuilt `libempyrean-<target>.tar.gz` for this crate's
//!      version from the GitHub release, verified against a pinned SHA-256, into
//!      a persistent per-version cache.
//!
//! Download and extraction are done in-process (ureq + flate2 + tar), so the
//! build needs no system `curl` / `wget` / `tar`. FFI bindings are pre-generated
//! and committed, so it needs no C header and no `libclang` / `bindgen` either.

use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const REPO: &str = "Empyrean-Dynamics/empyrean";

/// SHA-256 of each `libempyrean-<target>.tar.gz` release asset, pinned to this
/// crate version. Regenerated and pinned by `.github/workflows/release.yml`
/// at publish time (one `<asset-stem> <sha256>` pair per line; `#` comments
/// ignored).
const CHECKSUMS: &str = include_str!("checksums.txt");

/// `(asset stem, sha256)` for the host target, looked up from `checksums.txt`.
fn target_asset() -> Option<(String, String)> {
    let stem = match (target_os().as_str(), target_arch().as_str()) {
        ("macos", "aarch64") => "libempyrean-macos-aarch64",
        ("linux", "x86_64") => "libempyrean-linux-x86_64",
        _ => return None,
    };
    let sha = CHECKSUMS.lines().find_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (name, hash) = line.split_once(char::is_whitespace)?;
        (name.trim() == stem).then(|| hash.trim().to_string())
    })?;
    Some((stem.to_string(), sha))
}

fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").unwrap_or_default()
}
fn target_arch() -> String {
    env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default()
}

fn lib_filename() -> &'static str {
    match target_os().as_str() {
        "macos" => "libempyrean.dylib",
        "windows" => "empyrean.dll",
        _ => "libempyrean.so",
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=EMPYREAN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=EMPYREAN_FORCE_DOWNLOAD");

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("lib_path.rs");

    // docs.rs builds in a network-isolated sandbox and never loads the library.
    // Write a placeholder path so the crate compiles; the loader is lazy and is
    // never invoked there.
    if env::var_os("DOCS_RS").is_some() {
        fs::write(&out, "pub const LIB_PATH: &str = \"\";\n").expect("write lib_path.rs");
        return;
    }

    let lib_file = lib_filename();
    let lib_dir = resolve_lib_dir(lib_file);

    let lib_path = lib_dir.join(lib_file);
    assert!(
        lib_path.exists(),
        "libempyrean not found at {}. Set EMPYREAN_LIB_DIR to a directory containing {lib_file}.",
        lib_path.display(),
    );

    // The library is opened by absolute path at run time (libloading), so there
    // is no link-time dependency to emit — just record where it lives.
    let abs = lib_path.canonicalize().unwrap_or(lib_path);
    fs::write(
        &out,
        format!("pub const LIB_PATH: &str = {:?};\n", abs.to_string_lossy()),
    )
    .expect("write lib_path.rs");
    println!("cargo:rerun-if-changed={}", abs.display());
}

fn resolve_lib_dir(lib_file: &str) -> PathBuf {
    // 1. Explicit override.
    if let Ok(dir) = env::var("EMPYREAN_LIB_DIR") {
        return PathBuf::from(dir);
    }

    // 2. In-tree workspace build (development).
    let force = matches!(
        env::var("EMPYREAN_FORCE_DOWNLOAD").as_deref(),
        Ok("1") | Ok("true")
    );
    if !force {
        let ws = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../target/release");
        if ws.join(lib_file).exists() {
            return ws;
        }
    }

    // 3. Download the prebuilt, version-pinned library.
    download_prebuilt(lib_file)
}

fn download_prebuilt(lib_file: &str) -> PathBuf {
    let (stem, expected_sha) = target_asset().unwrap_or_else(|| {
        panic!(
            "No prebuilt libempyrean is published for target {}-{}. Build it from the engine \
             and point EMPYREAN_LIB_DIR at the directory containing {lib_file}.",
            target_arch(),
            target_os(),
        )
    });

    let cache = cache_dir();
    fs::create_dir_all(&cache).expect("create libempyrean cache dir");
    let lib_path = cache.join(lib_file);

    // A previously-downloaded, prepared library is reused as-is.
    if lib_path.exists() {
        return cache;
    }

    let url = format!("https://github.com/{REPO}/releases/download/v{VERSION}/{stem}.tar.gz");
    eprintln!("empyrean-sys: fetching prebuilt {stem} from {url}");

    // Download into memory with a pure-Rust HTTPS client (rustls), following
    // GitHub's redirect to the asset CDN — no system curl / wget / tar needed,
    // so the crate builds in minimal containers too.
    let resp = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("download libempyrean from {url}: {e}"));
    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("read libempyrean download from {url}: {e}"));

    // Verify the pinned SHA-256 before trusting the binary.
    let got = sha256_hex(&bytes);
    if got != expected_sha {
        panic!(
            "libempyrean checksum mismatch for {stem}.tar.gz\n  expected {expected_sha}\n  got      {got}\n\
             Refusing to use an unverified binary."
        );
    }

    // Extract the gzip-compressed tar in-process.
    let decoder = flate2::read::GzDecoder::new(&bytes[..]);
    tar::Archive::new(decoder)
        .unpack(&cache)
        .unwrap_or_else(|e| panic!("extract {stem}.tar.gz: {e}"));
    assert!(
        lib_path.exists(),
        "{stem}.tar.gz did not contain {lib_file}"
    );

    cache
}

fn cache_dir() -> PathBuf {
    let base = env::var("XDG_CACHE_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache"))
        })
        .unwrap_or_else(env::temp_dir);
    base.join("empyrean")
        .join("libempyrean")
        .join(VERSION)
        .join(format!("{}-{}", target_arch(), target_os()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
