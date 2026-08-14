//! Capability-parity gate: nothing reaches the C ABI without a consumer above
//! it or a recorded decision to have none.
//!
//! `empyrean_evaluate_plan` and `empyrean_plan_result_free` shipped for three
//! releases as dead planning surface — exported from the engine, declared in
//! the public header, mirrored into these bindings, and reachable from no
//! other channel: no safe Rust wrapper, no Python binding, no CLI subcommand.
//! Nothing failed, because nothing anywhere compared the exported set against
//! what the layers above it actually call. Dead ABI surface is worse than a
//! missing feature: it is documented, header-visible and effectively frozen
//! (no exported symbol has ever been removed), and every release spends review
//! effort re-discovering that it goes nowhere.
//!
//! # What is authoritative here
//!
//! `src/shims.rs` is **not** the source of truth for what the engine exports.
//! It sits at the end of a generator chain — `empyrean-c/src` → cbindgen →
//! `include/empyrean.h` → bindgen → `src/bindings.rs` → `src/shims.rs` — and
//! `tests/abi_surface.txt` is regenerated from `shims.rs` in turn. Comparing
//! those two alone would compare a file to the file it came from: it catches a
//! manifest nobody regenerated, and nothing else. A generator that drops a
//! symbol the engine really exports would leave every such check green on
//! precisely the bug this gate exists to catch.
//!
//! So the anchor is the **compiled `libempyrean`** — the artifact that
//! actually ships and that `dlsym` actually resolves against, read with `nm`.
//! Check 1 pins the manifest to the compiled engine; check 2 pins the shims to
//! the manifest, which is what localizes a failure to the generator rather than
//! to the manifest.
//!
//! # The four checks
//!
//! 1. `tests/abi_surface.txt` equals the `empyrean_*` symbols exported by the
//!    compiled engine library, both directions.
//! 2. `tests/abi_surface.txt` equals the exported set in `src/shims.rs`, both
//!    directions, and lists each name once.
//! 3. Every manifest symbol is either called from `empyrean/src` or listed in
//!    `tests/not_yet_wrapped.txt` with a reason.
//! 4. Every `not_yet_wrapped.txt` entry names a real, still-unwrapped symbol,
//!    appears once, and carries a reason — so an exception cannot outlive its
//!    exceptional case, and a merge that duplicates an entry cannot silently
//!    drop one of the two recorded decisions.
//!
//! A fifth test is not a parity check but a self-test of the consumption
//! scanner: check 3 is only as honest as the scanner's idea of what a call is,
//! and its comment / string / test-gating rules are pinned against a fixture
//! rather than assumed.
//!
//! The manifest is a plain sorted name list, regenerated from within
//! `empyrean-sys/` with:
//!
//! ```text
//! grep -o 'pub unsafe fn empyrean_[a-zA-Z0-9_]*' src/shims.rs \
//!     | sed 's/pub unsafe fn //' | sort -u > tests/abi_surface.txt
//! ```
//!
//! Regenerating it is the last step, not the fix: check 3 is what decides
//! whether the new symbol may ship.
//!
//! # What the consumer scan does not see
//!
//! The scan is textual, over the files under `empyrean/src`, and two gaps are
//! left open deliberately rather than papered over:
//!
//! - **Orphan files count.** A call in a `.rs` file that no `mod` declaration
//!   names is never compiled — rustc opens it, warns about it and lints it
//!   exactly never — yet this scan reads it off disk and credits it. Every
//!   file under `empyrean/src` is currently reachable from `lib.rs`, but this
//!   gate does not enforce that, so a wrapper added without its `mod` line
//!   would pass here while shipping no reachable API.
//! - **Only `#[cfg(test)]` is recognised as test-gated.** Its regions are
//!   skipped by brace depth, because a call that exists only in a unit test is
//!   not a capability any user can reach. Other gating forms
//!   (`#[cfg(all(test, …))]`, a feature-gated module) are not detected and
//!   would be credited.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `shims.rs` declares exactly one of these per entry point it wraps.
const EXPORT_DECL: &str = "pub unsafe fn ";

/// A capability is consumed when the wrapper *calls* it. A path that merely
/// appears (in a `use`, in prose, in a doc link) hands nothing to a caller.
const SYS_PATH: &str = "empyrean_sys::";

/// Mach-O prefixes every C symbol with an underscore; ELF does not.
#[cfg(target_os = "macos")]
const SYMBOL_PREFIX: &str = "_";
#[cfg(not(target_os = "macos"))]
const SYMBOL_PREFIX: &str = "";

/// `nm` flags that list **defined, external** symbols — the ones a consumer can
/// actually resolve. `-D` alone would also report undefined imports.
#[cfg(target_os = "macos")]
const NM_ARGS: &[&str] = &["-gU"];
#[cfg(not(target_os = "macos"))]
const NM_ARGS: &[&str] = &["-D", "--defined-only"];

/// One parsed line of a committed list file, carrying its line number so a
/// duplicate can be reported where it lives.
struct Entry {
    name: String,
    reason: String,
    line: usize,
}

fn sys_crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The safe wrapper this gate measures against.
///
/// Hard-required, never skipped: the gate runs from the repo workspace, where
/// `../empyrean/src` always exists, and a gate that quietly turns itself off
/// when its input is missing is not a gate — it is exactly the silence that
/// let the planning surface sit unwrapped for three releases. (`tests/` is not
/// in the crate's `include` list, so there is no published-package context in
/// which this path is legitimately absent.)
fn wrapper_src_dir() -> PathBuf {
    let dir = sys_crate_dir().join("../empyrean/src");
    assert!(
        dir.is_dir(),
        "capability-parity gate cannot run: the safe wrapper's sources are not at {}. \
         This gate must be run from the repo workspace, where the `empyrean` crate sits \
         beside `empyrean-sys`; it deliberately fails rather than skipping, because a \
         parity check that opts out when it cannot see one side proves nothing.",
        dir.display()
    );
    dir
}

/// The engine library this build resolves, following the same order
/// `empyrean-sys` itself uses: the `EMPYREAN_LIB` override first, then the
/// absolute path recorded at build time. The crate's middle step — a library
/// bundled beside the loaded module — is a packaging path for relocatable
/// artifacts (wheels) and never applies to a workspace test run.
fn engine_lib_path() -> PathBuf {
    let path = std::env::var_os("EMPYREAN_LIB")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(empyrean_sys::LIB_PATH));
    assert!(
        path.is_file(),
        "capability-parity gate cannot run: no engine library at {}. This is the path \
         empyrean-sys resolves and `smoke.rs` dlopens, so a test run that gets this far \
         has one; set EMPYREAN_LIB if the engine lives elsewhere.",
        path.display()
    );
    path
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Take the leading `[A-Za-z0-9_]` run of `s`, plus whatever follows it.
fn split_ident(s: &str) -> (&str, &str) {
    let end = s
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(s.len());
    s.split_at(end)
}

/// The code part of one Rust source line: string-literal contents blanked, and
/// everything from the first `//` outside a string dropped.
///
/// Both halves matter to the consumer scan. A trailing comment is the real
/// hazard — `let x = todo!(); // route through empyrean_sys::empyrean_foo(…)`
/// does not start with `//`, so a filter that only skips comment-leading lines
/// records that mention as a call, and an export whose last true caller was
/// deleted keeps passing on the note that survived. Blanking string contents
/// closes the same hole for a symbol named inside an error message.
///
/// Approximate by design: it tracks `"` and its backslash escapes, and does
/// not model raw strings, char literals or a string spanning two lines (none
/// of which occur in `empyrean/src`). Every inaccuracy is confined to one
/// line, and the direction is conservative — text that is dropped can only
/// *fail* to credit a consumer, never invent one.
fn code_only(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(c) = chars.next() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            // Keep the line's shape, drop its content.
            out.push(' ');
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(' ');
            }
            '/' if chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }
    out
}

/// Parse a committed list file: `name` or `name | reason`, skipping blanks and
/// `#` comments, keeping line numbers.
fn parse_entries(path: &Path) -> Vec<Entry> {
    read(path)
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'))
        .map(|(line, l)| match l.split_once('|') {
            Some((name, reason)) => Entry {
                name: name.trim().to_string(),
                reason: reason.trim().to_string(),
                line,
            },
            // A missing `|` parses as an empty reason rather than failing here,
            // so the reason requirement is reported by the test that owns it
            // instead of as a parse panic.
            None => Entry {
                name: l.to_string(),
                reason: String::new(),
                line,
            },
        })
        .collect()
}

fn manifest_entries() -> Vec<Entry> {
    parse_entries(&sys_crate_dir().join("tests/abi_surface.txt"))
}

fn decision_entries() -> Vec<Entry> {
    parse_entries(&sys_crate_dir().join("tests/not_yet_wrapped.txt"))
}

fn names(entries: &[Entry]) -> BTreeSet<String> {
    entries.iter().map(|e| e.name.clone()).collect()
}

/// Collapsing a list into a map or a set hides a repeated name — last write
/// wins, no diagnostic — in files whose entire job is to carry one record per
/// symbol. Report the repeats with their line numbers instead.
fn duplicate_report(entries: &[Entry], file: &str) -> String {
    let mut lines_by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for entry in entries {
        lines_by_name
            .entry(entry.name.as_str())
            .or_default()
            .push(entry.line);
    }
    let mut report = String::new();
    for (name, lines) in lines_by_name {
        if lines.len() > 1 {
            let lines: Vec<String> = lines.iter().map(usize::to_string).collect();
            report.push_str(&format!(
                "  {name} — appears {} times in {file} (lines {}). Keep one entry per symbol: \
                 the later line silently replaces the earlier one, so a merge that brings two \
                 records together loses whichever sorts first.\n",
                lines.len(),
                lines.join(", ")
            ));
        }
    }
    report
}

/// The exported set according to `src/shims.rs`, parsed at run time so the
/// check sees the file as committed rather than a snapshot baked in at compile
/// time.
fn shim_symbols() -> BTreeSet<String> {
    let src = read(&sys_crate_dir().join("src/shims.rs"));
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let Some(after) = line.trim_start().strip_prefix(EXPORT_DECL) else {
            continue;
        };
        let (name, _) = split_ident(after);
        if name.starts_with("empyrean_") {
            out.insert(name.to_string());
        }
    }
    out
}

/// The `empyrean_*` symbols the compiled engine actually exports.
fn engine_exported_symbols() -> BTreeSet<String> {
    let path = engine_lib_path();
    let output = Command::new("nm")
        .args(NM_ARGS)
        .arg(&path)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "capability-parity gate cannot run: failed to execute `nm {} {}`: {e}. \
                 `nm` ships with the platform toolchain (Xcode command line tools / binutils) \
                 and is what CI's own symbol sweep uses. This check fails rather than skips: \
                 without it the gate compares generated files to each other and cannot see a \
                 symbol the generator dropped.",
                NM_ARGS.join(" "),
                path.display()
            )
        });
    assert!(
        output.status.success(),
        "`nm {} {}` failed ({}): {}",
        NM_ARGS.join(" "),
        path.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let listing = String::from_utf8_lossy(&output.stdout);
    let mut out = BTreeSet::new();
    for line in listing.lines() {
        // `<address> <type> <name>`, or a bare name for an address-less entry.
        let Some(symbol) = line.split_whitespace().next_back() else {
            continue;
        };
        let symbol = symbol.strip_prefix(SYMBOL_PREFIX).unwrap_or(symbol);
        if symbol.starts_with("empyrean_") {
            out.insert(symbol.to_string());
        }
    }
    assert!(
        !out.is_empty(),
        "`nm` read {} and reported no empyrean_* exports at all. That file is not the engine, \
         or it is stripped, or it was built for another architecture — in any case this gate \
         cannot anchor to it.",
        path.display()
    );
    out
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to walk {}: {e}", dir.display()))
            .path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Every `empyrean_sys::<name>(` in one source file, ignoring comments, string
/// literals and `#[cfg(test)]` items.
fn consumed_in(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut skipping = false;
    // A `#[cfg(test)]` item is skipped until its own braces balance. Until the
    // first `{` is seen the depth is meaningless, so a brace-less item (a
    // gated `use` or `mod tests;`) ends the skip at its terminating `;`.
    let mut depth: i32 = 0;
    let mut awaiting_body = false;
    for line in src.lines() {
        let code = code_only(line);
        if !skipping && code.contains("#[cfg(test)]") {
            skipping = true;
            awaiting_body = true;
            depth = 0;
        }
        if skipping {
            for c in code.chars() {
                match c {
                    '{' => {
                        depth += 1;
                        awaiting_body = false;
                    }
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            if awaiting_body {
                if code.trim_end().ends_with(';') {
                    skipping = false;
                }
            } else if depth <= 0 {
                skipping = false;
            }
            continue;
        }
        let mut rest = code.as_str();
        while let Some(at) = rest.find(SYS_PATH) {
            let after = &rest[at + SYS_PATH.len()..];
            let (name, tail) = split_ident(after);
            if !name.is_empty() && tail.trim_start().starts_with('(') {
                out.insert(name.to_string());
            }
            rest = after;
        }
    }
    out
}

fn consumed_symbols() -> BTreeSet<String> {
    let dir = wrapper_src_dir();
    let mut files = Vec::new();
    rust_sources(&dir, &mut files);
    assert!(
        !files.is_empty(),
        "no .rs files under {} — the wrapper sources moved, and this gate is scanning nothing",
        dir.display()
    );
    files
        .iter()
        .flat_map(|file| consumed_in(&read(file)))
        .collect()
}

/// The scanner decides what check 3 will accept as proof that a capability is
/// reachable, so every rule it applies is pinned here against a fixture. The
/// destructive direction is the one that matters: each of these non-calls, if
/// credited, would let an export whose last real caller was deleted keep
/// passing on the text that survived it.
#[test]
fn the_consumption_scanner_credits_calls_and_nothing_else() {
    let fixture = r#"
//! empyrean_sys::empyrean_doc_module(ctx) in a module doc
/// [`empyrean_sys::empyrean_doc_item(ctx)`] in an item doc
// empyrean_sys::empyrean_line_comment(ctx)
use empyrean_sys::empyrean_imported_but_not_called;

pub fn wrapper() -> i32 {
    let msg = "call empyrean_sys::empyrean_in_a_string(ctx) yourself";
    let escaped = "a quote \" then empyrean_sys::empyrean_after_escaped_quote(ctx)";
    let _ = msg;
    let _ = escaped;
    unsafe { empyrean_sys::empyrean_real_call(ctx) } // empyrean_sys::empyrean_trailing(ctx)
}

#[cfg(test)]
mod tests {
    #[test]
    fn t() {
        let brace = "}{";
        let _ = brace;
        unsafe { empyrean_sys::empyrean_test_only(ctx) };
    }
}
"#;
    let found = consumed_in(fixture);
    assert_eq!(
        found,
        BTreeSet::from(["empyrean_real_call".to_string()]),
        "the scanner must credit exactly the one live call: a doc comment, a line comment, a \
         trailing comment, a string literal (escaped quotes included), a bare `use` import and \
         a `#[cfg(test)]` body are all non-consumption"
    );
}

#[test]
fn the_committed_manifest_matches_the_compiled_engines_exported_symbols() {
    let exported = engine_exported_symbols();
    let manifest = names(&manifest_entries());

    let mut report = String::new();
    for name in exported.difference(&manifest) {
        report.push_str(&format!(
            "  + {name} — exported by the compiled libempyrean, missing from \
             tests/abi_surface.txt\n"
        ));
    }
    for name in manifest.difference(&exported) {
        report.push_str(&format!(
            "  - {name} — listed in tests/abi_surface.txt, not exported by the compiled \
             libempyrean\n"
        ));
    }

    assert!(
        report.is_empty(),
        "the committed ABI surface manifest disagrees with the engine that actually ships:\n\
         {report}\n\
         `+` means the generator chain dropped a symbol the engine exports — regenerate \
         src/bindings.rs and src/shims.rs from include/empyrean.h, then the manifest from \
         shims.rs, and wire a consumer or record a decision for it.\n\
         `-` means the manifest names a symbol the engine does not export — either the \
         manifest is stale, or the resolved libempyrean is from another build (check \
         EMPYREAN_LIB and rebuild the engine)."
    );
}

#[test]
fn the_committed_manifest_matches_the_generated_shim_surface() {
    let entries = manifest_entries();
    let manifest = names(&entries);
    let shims = shim_symbols();

    let mut report = duplicate_report(&entries, "tests/abi_surface.txt");
    for name in shims.difference(&manifest) {
        report.push_str(&format!(
            "  + {name} — declared in src/shims.rs, missing from tests/abi_surface.txt\n"
        ));
    }
    for name in manifest.difference(&shims) {
        report.push_str(&format!(
            "  - {name} — listed in tests/abi_surface.txt, not declared in src/shims.rs\n"
        ));
    }

    assert!(
        report.is_empty(),
        "the committed ABI surface manifest disagrees with the generated shims:\n{report}\n\
         new export → add it to empyrean-sys/tests/abi_surface.txt AND wire a consumer or add \
         a not_yet_wrapped.txt entry with a reason.\n\
         removed export → dropping a symbol is a breaking ABI change; if that is deliberate, \
         remove the name from abi_surface.txt in the same commit.\n\
         If the engine check passed and this one did not, the shims are what drifted."
    );
}

#[test]
fn every_exported_symbol_is_consumed_or_carries_a_recorded_decision() {
    let manifest = names(&manifest_entries());
    let consumed = consumed_symbols();
    let decisions = names(&decision_entries());

    let mut report = String::new();
    for name in &manifest {
        if consumed.contains(name) || decisions.contains(name) {
            continue;
        }
        report.push_str(&format!(
            "  {name} — exported, but never called as `empyrean_sys::{name}(…)` from \
             empyrean/src outside comments and test-gated code, and no entry in \
             tests/not_yet_wrapped.txt\n"
        ));
    }

    assert!(
        report.is_empty(),
        "these C-ABI exports reach no channel above empyrean-sys:\n{report}\n\
         for each one: wire a consumer in the safe wrapper (and mirror it into empyrean-py / \
         empyrean-cli under the API-parity rule), or record the decision by adding \
         `<name> | <reason>` to empyrean-sys/tests/not_yet_wrapped.txt. Shipping an export \
         with neither is how the planning surface stayed dead for three releases."
    );
}

#[test]
fn every_not_yet_wrapped_entry_is_still_a_live_exception() {
    let manifest = names(&manifest_entries());
    let consumed = consumed_symbols();
    let entries = decision_entries();

    let mut report = duplicate_report(&entries, "tests/not_yet_wrapped.txt");
    for entry in &entries {
        let name = &entry.name;
        if !manifest.contains(name) {
            report.push_str(&format!(
                "  {name} (line {}) — not in tests/abi_surface.txt: the symbol does not exist \
                 (check the spelling) or it is gone. Remove the entry.\n",
                entry.line
            ));
            // Nothing further is meaningful for a symbol that is not exported.
            continue;
        }
        if consumed.contains(name) {
            report.push_str(&format!(
                "  {name} (line {}) — remove the entry: it is now wrapped, called from \
                 empyrean/src. A recorded decision that no longer holds hides the next one.\n",
                entry.line
            ));
        }
        if entry.reason.is_empty() {
            report.push_str(&format!(
                "  {name} (line {}) — no reason given. Write `{name} | <why this ships \
                 unwrapped>`; the list exists to carry the decision, not to grant the \
                 exemption.\n",
                entry.line
            ));
        }
    }

    assert!(
        report.is_empty(),
        "empyrean-sys/tests/not_yet_wrapped.txt is stale:\n{report}"
    );
}
