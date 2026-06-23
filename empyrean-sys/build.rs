//! Build script for `empyrean-sys`.
//!
//! Locates `libempyrean` (the prebuilt engine shared library) and emits the
//! link directives. Resolution order:
//!
//!   1. `EMPYREAN_LIB_DIR` — explicit override (offline / air-gapped / a
//!      locally built library).
//!   2. A sibling workspace build at `../target/release` (in-tree development),
//!      unless `EMPYREAN_FORCE_DOWNLOAD=1`.
//!   3. Download the prebuilt `libempyrean-<target>.tar.gz` for this crate's
//!      version from the GitHub release, verified against a pinned SHA-256, into
//!      a persistent per-version cache.
//!
//! For the downloaded case the library's install name / soname is rewritten to
//! its absolute cache path, so binaries that link `empyrean` resolve it at run
//! time without any rpath, `DYLD_LIBRARY_PATH`, or `LD_LIBRARY_PATH` setup.
//!
//! FFI bindings are pre-generated and committed (`src/bindings.rs`), so building
//! this crate needs neither the C header nor `libclang` / `bindgen`.

use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    // docs.rs builds in a network-isolated sandbox and does not link the final
    // artifact. Skip locating/downloading libempyrean there; the committed
    // bindings still compile, so `cargo doc` succeeds.
    if env::var_os("DOCS_RS").is_some() {
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

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=empyrean");
    println!("cargo:rerun-if-changed={}", lib_path.display());
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

    bake_absolute_path(&lib_path);
    cache
}

/// Rewrite the library's recorded path to its absolute cache location so that
/// any binary linking it loads it from there at run time, with no rpath or
/// loader-path environment needed.
fn bake_absolute_path(lib_path: &Path) {
    let abs = lib_path
        .canonicalize()
        .unwrap_or_else(|_| lib_path.to_path_buf());
    let abs = path_str(&abs);
    match target_os().as_str() {
        "macos" => run(
            "install_name_tool",
            &["-id", abs, abs],
            "set absolute install_name on libempyrean.dylib",
        ),
        "linux" => {
            if which("patchelf") {
                run(
                    "patchelf",
                    &["--set-soname", abs, abs],
                    "set absolute soname on libempyrean.so",
                );
            } else {
                println!(
                    "cargo:warning=patchelf not found; if a binary linking `empyrean` fails to \
                     load libempyrean.so, install patchelf and rebuild, or add {} to \
                     LD_LIBRARY_PATH.",
                    lib_path.parent().unwrap().display()
                );
            }
        }
        _ => {}
    }
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

fn which(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run(cmd: &str, args: &[&str], what: &str) {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn `{cmd}` ({what}): {e}"));
    assert!(
        out.status.success(),
        "`{cmd}` failed ({what}):\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
}

fn path_str(p: &Path) -> &str {
    p.to_str().expect("non-UTF8 path")
}
