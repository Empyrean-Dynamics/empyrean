//! Every entry point that can reach the engine before a [`Context`] exists
//! must open it through `ensure_engine_loaded` first.
//!
//! `empyrean_sys`'s free `empyrean_*` shims mirror the C ABI's signatures, so
//! they have nowhere to report "the library could not be opened" and panic
//! instead. A wrapper function that calls one without a `Context` in hand
//! therefore aborts the process on a broken install rather than returning
//! [`Error`](empyrean::Error) with the lookup diagnosis. That is what a
//! downstream caller hit in a container, and `Context::from_data_dir` being
//! guarded is not enough: `read_orbits_csv` or `query_sbdb` is just as
//! plausibly the first call a program makes.
//!
//! The rule cannot be checked by calling these functions, because a test
//! process cannot simulate a missing engine: the resolution order ends at the
//! build-time path, which by construction exists on the machine running the
//! test, and hiding it would break every other test in the binary. So this
//! checks the structure instead — that the guard is *there* — which is the
//! property that actually regresses when someone adds an entry point.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Functions that reach a shim without the guard, and are correct as they are.
///
/// Every entry is one of two shapes: it takes a `&Context`, so the engine
/// demonstrably loaded already, or it is a private helper on the far side of a
/// guarded public function. Anything else belongs guarded, not listed here.
const ALLOWED: &[(&str, &str)] = &[
    (
        "built_system.rs::new_for_od",
        "takes a &Context, which cannot exist unless the engine opened",
    ),
    (
        "context.rs::drain_missing_data_files",
        "drains a payload the engine just recorded, so it ran",
    ),
    (
        "error.rs::capture",
        "reads the error a completed FFI call left behind",
    ),
    (
        "error.rs::capture_location",
        "drains the position of the error a completed FFI call left behind; \
         only `capture` calls it, after that call returned",
    ),
    (
        "ephemeris.rs::marshal_ephemeris_result",
        "unpacks a result a Context method produced",
    ),
    (
        "io/orbits.rs::read_orbits_parquet",
        "delegates to read_orbits_via, which is guarded",
    ),
    (
        "io/orbits.rs::read_orbits_json",
        "delegates to read_orbits_via, which is guarded",
    ),
    (
        "io/orbits.rs::read_orbits_csv",
        "delegates to read_orbits_via, which is guarded",
    ),
    (
        "od/observation.rs::build_optical_ffi",
        "private helper called only from the guarded from_arrays",
    ),
    (
        "od/observation.rs::build_radar_ffi",
        "private helper called only from the guarded from_arrays",
    ),
];

/// One function found in the crate source.
struct Function {
    /// `<path relative to src>::<name>`, the key `ALLOWED` uses.
    id: String,
    /// Whether the body calls an `empyrean_sys::empyrean_*` shim.
    calls_shim: bool,
    /// Whether the body calls `ensure_engine_loaded`.
    guarded: bool,
    /// Whether the signature takes a receiver.
    has_receiver: bool,
    /// Whether the item sits under a `#[cfg(test)]` module.
    in_test_module: bool,
}

/// Whether `line` opens a function definition, and its name if so.
///
/// Handles the qualifier soup a signature can start with: `pub`,
/// `pub(crate)` / `pub(super)` / `pub(in path)`, `const`, `async`, `unsafe`,
/// `extern "C"`, in any order they legally appear.
fn function_name(line: &str) -> Option<&str> {
    let mut rest = line.trim_start();
    loop {
        // `pub` may be followed by a parenthesised scope with no space.
        if let Some(after) = rest.strip_prefix("pub") {
            let after = match after.strip_prefix('(') {
                Some(scoped) => scoped.split_once(')')?.1,
                None => after,
            };
            if after.starts_with(char::is_whitespace) {
                rest = after.trim_start();
                continue;
            }
        }
        let stripped = ["const ", "async ", "unsafe ", "extern \"C\" "]
            .iter()
            .find_map(|kw| rest.strip_prefix(kw));
        match stripped {
            Some(after) => rest = after.trim_start(),
            None => break,
        }
    }
    let rest = rest.strip_prefix("fn ")?;
    let end = rest
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

/// Whether a signature's first parameter is a receiver — `self`, `&self`,
/// `&mut self`, `&'a self`, `mut self`.
///
/// Parsed rather than substring-matched because a signature wraps across
/// lines: a `pub fn propagate(\n    &self,` would read as receiverless to a
/// naive `contains("(&self")`, and every `Context` method would then be
/// demanded to carry a guard it does not need.
fn takes_receiver(signature: &str) -> bool {
    // Start at the `fn` keyword: a `pub(crate)` scope carries its own
    // parentheses, and splitting on the first `(` in the line would find
    // those instead of the parameter list.
    let Some(after_fn) = signature.find("fn ").map(|i| &signature[i + 3..]) else {
        return false;
    };
    let Some((_, params)) = after_fn.split_once('(') else {
        return false;
    };
    let mut rest = params.trim_start();
    rest = rest.strip_prefix('&').unwrap_or(rest).trim_start();
    if let Some(after) = rest.strip_prefix('\'') {
        // A named lifetime on the receiver: skip it.
        let end = after
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(after.len());
        rest = after[end..].trim_start();
    }
    rest = rest.strip_prefix("mut ").unwrap_or(rest).trim_start();
    let Some(after) = rest.strip_prefix("self") else {
        return false;
    };
    // `self`, not `self_something`.
    !after.starts_with(|c: char| c.is_alphanumeric() || c == '_')
}

/// Every function in one file, with the facts this test keys on.
fn functions_in(file: &Path, rel: &str) -> Vec<Function> {
    let text = std::fs::read_to_string(file).expect("read a source file of this crate");
    let lines: Vec<&str> = text.lines().collect();
    let mut found = Vec::new();

    // Line at which the innermost `#[cfg(test)]` module body ends, if inside
    // one. Test code may call whatever it likes.
    let mut test_module_end = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("#[cfg(test)]")
            && i + 1 < lines.len()
            && let Some(end) = block_end(&lines, i + 1)
        {
            test_module_end = test_module_end.max(end);
        }
        let Some(name) = function_name(line) else {
            continue;
        };
        // Walk to the line that opens the body; the signature may wrap.
        let Some(open) = (i..lines.len()).find(|&j| lines[j].contains('{')) else {
            continue;
        };
        let signature = lines[i..=open].join("\n");
        let Some(close) = block_end(&lines, open) else {
            continue;
        };
        let body = lines[open..=close].join("\n");
        found.push(Function {
            id: format!("{rel}::{name}"),
            calls_shim: body.contains("empyrean_sys::empyrean_"),
            guarded: body.contains("ensure_engine_loaded"),
            has_receiver: takes_receiver(&signature),
            in_test_module: i <= test_module_end,
        });
    }
    found
}

/// Index of the line closing the brace block opened at or after `start`.
fn block_end(lines: &[&str], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut opened = false;
    for (i, line) in lines.iter().enumerate().skip(start) {
        for c in line.chars() {
            match c {
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if opened && depth <= 0 {
            return Some(i);
        }
    }
    None
}

/// Every `.rs` file under this crate's `src`, with its path relative to it.
fn source_files() -> Vec<(PathBuf, String)> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    let mut stack = vec![src.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read this crate's src tree") {
            let path = entry.expect("read a directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let rel = path
                    .strip_prefix(&src)
                    .expect("every file is under src")
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((path, rel));
            }
        }
    }
    assert!(!out.is_empty(), "the source sweep found no files to check");
    out
}

/// Functions that call a shim with no receiver and no guard.
fn unguarded() -> BTreeSet<String> {
    source_files()
        .iter()
        .flat_map(|(path, rel)| functions_in(path, rel))
        .filter(|f| f.calls_shim && !f.guarded && !f.has_receiver && !f.in_test_module)
        .map(|f| f.id)
        .collect()
}

#[test]
fn every_entry_point_that_can_precede_a_context_opens_the_engine_first() {
    let allowed: BTreeSet<String> = ALLOWED.iter().map(|(id, _)| (*id).to_string()).collect();
    let gaps: Vec<String> = unguarded().difference(&allowed).cloned().collect();

    assert!(
        gaps.is_empty(),
        "these reach an empyrean_sys shim with no Context and no ensure_engine_loaded, so on a \
         broken install they panic instead of returning the lookup diagnosis:\n  {}\n\nAdd \
         `crate::context::ensure_engine_loaded()?;` as the first statement, or — if the engine \
         is provably open by the time it runs — add it to ALLOWED in this file with the reason.",
        gaps.join("\n  ")
    );
}

#[test]
fn the_sweep_can_still_see_an_unguarded_entry_point() {
    // Positive control. The test above passes trivially if the sweep stops
    // finding functions — a parser that silently matches nothing is exactly
    // the failure mode a structural check has. So: the sweep must find a
    // healthy number of functions, must find the ones it is meant to exempt,
    // and must classify a known-guarded and a known-unguarded function
    // correctly.
    let files = source_files();
    let all: Vec<_> = files
        .iter()
        .flat_map(|(path, rel)| functions_in(path, rel))
        .collect();
    assert!(
        all.len() > 200,
        "the sweep found only {} functions; the parser has stopped working",
        all.len()
    );

    let shim_callers: Vec<_> = all.iter().filter(|f| f.calls_shim).collect();
    assert!(
        shim_callers.len() > 20,
        "only {} functions were seen calling a shim",
        shim_callers.len()
    );

    let find = |id: &str| {
        all.iter()
            .find(|f| f.id == id)
            .unwrap_or_else(|| panic!("the sweep did not find {id}"))
    };

    // A guarded entry point reads as guarded …
    let guarded = find("query.rs::query_sbdb");
    assert!(guarded.calls_shim && guarded.guarded && !guarded.has_receiver);

    // … an exempt private helper reads as unguarded, which is why ALLOWED
    // has to exist at all …
    let helper = find("error.rs::capture");
    assert!(helper.calls_shim && !helper.guarded);

    // … and a Context method reads as having a receiver, so it is never asked
    // for a guard it does not need.
    assert!(find("context.rs::as_raw").has_receiver);

    // Every exemption still names something real. A stale entry would quietly
    // widen the allowlist.
    let ids: BTreeSet<&str> = all.iter().map(|f| f.id.as_str()).collect();
    for (id, reason) in ALLOWED {
        assert!(
            ids.contains(id),
            "ALLOWED names {id}, which no longer exists"
        );
        assert!(!reason.is_empty(), "{id} is exempted with no reason");
    }
}
