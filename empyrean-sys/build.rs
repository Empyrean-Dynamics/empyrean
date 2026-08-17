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
//!      version from the GitHub release, verified against a pinned SHA-256 —
//!      and, in a repository checkout, against the pinned SHA-256 of the
//!      `include/empyrean.h` those binaries were built from — into a
//!      persistent per-version cache.
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

// The pin parser, shared verbatim with the crate's tests.
#[path = "src/header_pin.rs"]
mod header_pin;

/// SHA-256 of each `libempyrean-<target>.tar.gz` release asset, pinned to this
/// crate version. Regenerated and pinned by `.github/workflows/release.yml`
/// at publish time (one `<asset-stem> <sha256>` pair per line; `#` comments
/// ignored).
const CHECKSUMS: &str = include_str!("checksums.txt");

/// `(asset stem, sha256)` for the host target, looked up from `checksums.txt`.
fn target_asset() -> Option<(String, String)> {
    let stem = match (target_os().as_str(), target_arch().as_str()) {
        ("macos", "aarch64") => "libempyrean-macos-aarch64",
        ("macos", "x86_64") => "libempyrean-macos-x86_64",
        ("linux", "x86_64") => "libempyrean-linux-x86_64",
        ("linux", "aarch64") => "libempyrean-linux-aarch64",
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

/// `include/empyrean.h` when this crate is being built inside the
/// repository, `None` when it is the packaged crate (the header sits
/// outside the package root, so it is not published; the bindings ship
/// pre-generated instead).
///
/// The header's presence alone is not proof of a checkout — a registry
/// cache is also a parent directory full of other crates — so the
/// sibling `empyrean-c` package, which is what generates the header, has
/// to be there too.
fn checkout_header() -> Option<PathBuf> {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    let header = root.join("include/empyrean.h");
    (header.exists() && root.join("empyrean-c/Cargo.toml").exists()).then_some(header)
}

/// Refuse a downloaded prebuilt whose ABI surface predates the header in
/// this checkout.
///
/// The tarball checksums prove the downloaded bytes are the bytes the
/// release served; they say nothing about struct layouts. Neither does
/// the load-time `empyrean_abi_version` handshake, which compares a
/// constant encoding only the base version — so a struct that grows
/// inside one release cycle passes it. The gap is real and silent: a
/// checkout whose `EmpyreanOrbit` is 912 bytes, linked against a pinned
/// prebuilt whose `EmpyreanOrbit` is 832, strides every orbit array
/// wrong and reads past the caller's buffer.
///
/// So `checksums.txt` also pins the SHA-256 of the `include/empyrean.h`
/// those binaries were built from, and this refuses the download when
/// the checkout's header hashes to anything else.
///
/// Only the download path is guarded. A local source build
/// (`cargo build -p empyrean-c --release`, resolution step 2) produces
/// the library and the header from one tree and needs no pin, and
/// `EMPYREAN_LIB_DIR` is an explicit override the caller owns.
///
/// In the packaged crate there is no `include/empyrean.h` to hash — the
/// header lives outside the package root — and none is needed: a
/// published `empyrean-sys x.y.z` and the `v x.y.z` release assets it
/// downloads are cut from the same commit, which is what the version
/// pins. The guard therefore applies exactly where the two can drift.
fn assert_prebuilt_matches_header() {
    let Some(header) = checkout_header() else {
        return;
    };
    println!("cargo:rerun-if-changed={}", header.display());
    let pinned = header_pin::pinned_header_sha(CHECKSUMS).unwrap_or_else(|| {
        panic!(
            "empyrean-sys/checksums.txt records no `{}` line, so the pinned prebuilt cannot be \
             checked against {}. Re-run .github/workflows/prepare-release.yml, which pins both, \
             or build the engine locally:\n    cargo build -p empyrean-c --release",
            header_pin::HEADER_PIN_PREFIX,
            header.display(),
        )
    });
    let bytes = fs::read(&header)
        .unwrap_or_else(|e| panic!("read {} for the ABI header pin: {e}", header.display()));
    let actual = sha256_hex(&bytes);
    if !header_pin::header_matches(pinned, &actual) {
        panic!(
            "the pinned prebuilt predates this header; build empyrean-c locally \
             (cargo build -p empyrean-c --release)\n\
             \n\
             empyrean-sys/checksums.txt pins the libempyrean v{VERSION} binaries, which were \
             built from an include/empyrean.h hashing to\n    {pinned}\n\
             but {} hashes to\n    {actual}\n\
             \n\
             The two describe different struct layouts, and nothing downstream can see that: \
             EMPYREAN_ABI_VERSION encodes only the base version, so the load-time handshake \
             passes and every orbit array is then strided at the wrong width. Refusing to \
             download.\n\
             \n\
             Either build the engine from this tree (the release profile specifically):\n    \
             cargo build -p empyrean-c --release\n\
             or point at an engine built from this exact header:\n    \
             EMPYREAN_LIB_DIR=/path/to/dir/containing/libempyrean",
            header.display(),
        );
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=EMPYREAN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=EMPYREAN_FORCE_DOWNLOAD");
    println!("cargo:rerun-if-changed=checksums.txt");

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
    // Before anything is fetched or reused from the cache: the pinned
    // binaries must match this checkout's ABI header.
    assert_prebuilt_matches_header();

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
    let resp = ureq::get(&url).call().unwrap_or_else(|e| {
        panic!(
            "could not download libempyrean from {url}: {e}\n\
             \n\
             No prebuilt engine is published for version {VERSION} until that \
             release is cut, so on an unreleased version this URL is expected \
             to 404. Two ways forward, either of which this build script \
             prefers over the download:\n\
             \n\
               1. Build the engine in this workspace first:\n\
                    cargo build -p empyrean-c --release\n\
                  (the release profile specifically — a debug build is not \
                  picked up), then re-run your build.\n\
               2. Point at an engine you already have:\n\
                    EMPYREAN_LIB_DIR=/path/to/dir/containing/libempyrean\n\
             \n\
             At run time, EMPYREAN_LIB overrides the resolved path."
        )
    });
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
