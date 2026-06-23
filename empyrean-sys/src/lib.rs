//! Raw FFI bindings to `libempyrean`, the C shared library for empyrean's
//! astrodynamics engine.
//!
//! Prefer the safe wrapper crate [`empyrean`](https://docs.rs/empyrean)
//! unless you need direct access to the C ABI.
//!
//! # Local development
//!
//! The build script expects `libempyrean` and `empyrean.h` to be available.
//! By default it looks in the sibling `empyrean-internal` checkout's
//! `target/release` and `include` directories. Override with
//! `EMPYREAN_LIB_DIR` and `EMPYREAN_INCLUDE_DIR` environment variables.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
