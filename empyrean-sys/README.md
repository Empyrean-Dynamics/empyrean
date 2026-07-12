<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean-sys">

# empyrean-sys
Low-level FFI bindings to the libempyrean astrodynamics C ABI

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://crates.io/crates/empyrean-sys"><img src="https://img.shields.io/crates/v/empyrean-sys.svg?style=flat-square&label=crates.io" alt="crates.io"></a>
<a href="https://docs.rs/empyrean-sys"><img src="https://img.shields.io/docsrs/empyrean-sys?style=flat-square&label=docs.rs" alt="docs.rs"></a>
<br>
<a href="Cargo.toml"><img src="https://img.shields.io/badge/rustc-1.90%2B-orange?style=flat-square&logo=rust" alt="MSRV 1.90"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD"><img src="https://img.shields.io/badge/source-BSD--3--Clause-blue.svg?style=flat-square" alt="Source license"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY"><img src="https://img.shields.io/badge/binary-proprietary-lightgrey.svg?style=flat-square" alt="Binary license"></a>
<a href="https://doi.org/10.5281/zenodo.21318471"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21318471-blue?style=flat-square" alt="DOI"></a>
<br>
<a href="https://claude.ai"><img src="https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square" alt="Built with Claude Code"></a>
<a href="https://www.empyrean-dynamics.com"><img src="https://img.shields.io/badge/Website-empyrean--dynamics.com-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Website"></a>
<a href="https://github.com/Empyrean-Dynamics"><img src="https://img.shields.io/badge/GitHub-Empyrean--Dynamics-1a1a2e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a>

---

empyrean-sys exposes the C ABI of `libempyrean` to Rust as raw,
`unsafe`, bindgen-generated declarations. It does not attempt to wrap,
type-check, or RAII-manage the underlying handles.

```toml
[dependencies]
empyrean-sys = "0.8"
```

```rust
use empyrean_sys::*;

// All entry points are unsafe; pointer ownership and lifetime are the
// caller's responsibility. See include/empyrean.h at the repository
// root for the authoritative C ABI documentation.
unsafe {
    // Null data_dir = the platform default data directory; downloads
    // any missing kernels. Returns null on error (see empyrean_last_error).
    let ctx: *mut EmpyreanContext = empyrean_context_from_data_dir(std::ptr::null());
    assert!(!ctx.is_null());
    empyrean_context_free(ctx);
}
```

**Most users want the safe wrapper instead** — see the
[`empyrean`](https://crates.io/crates/empyrean) crate, which builds on
empyrean-sys to provide typed handles, `Result`-returning entry points,
and Rust-native lifetime management.

## Runtime requirement

empyrean-sys opens `libempyrean.{dylib,so}` at run time via
`libloading` (dlopen). The library is distributed separately as a
binary release on
[GitHub](https://github.com/Empyrean-Dynamics/empyrean/releases) and
inside the published Python wheel. The path is resolved from the
`EMPYREAN_LIB` environment variable if set, else a `libempyrean.*`
sitting next to the loaded module, else a build-time location — an
`EMPYREAN_LIB_DIR` override, a sibling `../target/release` build, or
a checksum-pinned download from the GitHub release (in that order).
The FFI bindings are pre-generated and committed, so no C header,
libclang, or bindgen is needed to build.

Prebuilt engine binaries are currently published for four targets:
macOS arm64 (`macos-aarch64`), macOS x86_64 (`macos-x86_64`), Linux
x86_64 (`linux-x86_64`), and Linux aarch64 (`linux-aarch64`). On other
targets the build stops with an error unless `EMPYREAN_LIB_DIR` points
at an engine build.

## License

Source code in this crate is licensed under the
[BSD 3-Clause License](LICENSE). The closed-source `libempyrean`
runtime it loads at runtime is governed by a separate proprietary binary
license; see the main repository for the dual-license breakdown.

Copyright © 2024–2026 Joachim Moeyens. All rights reserved.
