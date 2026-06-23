//! Raw FFI bindings to `libempyrean`, the C shared library for empyrean's
//! astrodynamics engine.
//!
//! Prefer the safe wrapper crate [`empyrean`](https://docs.rs/empyrean)
//! unless you need direct access to the C ABI.
//!
//! # Linking `libempyrean`
//!
//! The build script resolves the prebuilt `libempyrean` shared library in this
//! order:
//!
//! 1. `EMPYREAN_LIB_DIR` — a directory containing the library (explicit
//!    override, offline use, or a locally built library).
//! 2. A sibling `../target/release` workspace build (in-tree development).
//! 3. Otherwise it downloads the version-matched, checksum-pinned prebuilt for
//!    your platform from the GitHub release and caches it under
//!    `~/.cache/empyrean`. Set `EMPYREAN_FORCE_DOWNLOAD=1` to force this path.
//!
//! The bindings below are pre-generated and committed, so building needs no C
//! header and no `libclang` / `bindgen`.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!("bindings.rs");
