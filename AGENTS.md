# AGENTS.md

Guidance for coding agents working *inside this repository*. If you are
using the package rather than changing it, you want the skill instead:
`plugins/empyrean/skills/empyrean/SKILL.md`, installable with the lines on
<https://docs.empyrean-dynamics.com/agents.html>.

## What this repository is

- The empyrean distribution: `empyrean` (the Rust wrapper crate),
  `empyrean-sys` (FFI bindings), `empyrean-c` (the C ABI, built into
  `libempyrean`), `empyrean-cli` (the `empyrean` binary) and `empyrean-py`
  (the Python wheel). All four channels reach one engine binary; the
  engine's source is not in this repository, and the wrapper crates never
  depend on it directly, only on the C ABI.
- Pull requests from outside are not merged at present; open an issue
  first (see `CONTRIBUTING.md`). Issues route to the right repository:
  scenario problems to `empyrean-scenarios`, disagreements with a reference
  to `empyrean-validation`.

## Build and test

- Build the C ABI first; everything else links against it:
  `cargo build -p empyrean-c --release`.
- Rust: `cargo fmt`, then `cargo clippy --all-targets -- -D warnings`,
  then `cargo test`. Rustdoc builds with warnings denied in CI.
- Python (`empyrean-py`): build the wheel into a virtual environment with
  maturin, then `ruff check` and `pytest` (`-m "not slow"` for the quick
  set; tests marked `validation` need the kernels and the compiled engine).
- Docs (`empyrean-py/docs`): `make html`. The build also regenerates the
  generated region of `plugins/empyrean/skills/empyrean/SKILL.md`, the files
  under `plugins/empyrean/skills/empyrean/references/`, and writes `llms.txt`, `llms-full.txt` and
  a plain-Markdown copy of every page into the HTML output. Never edit the
  generated files by hand; `pytest` fails when they are stale.
- Versions are set together, by the maintainer, in the four `Cargo.toml`
  files, `empyrean-py/pyproject.toml`, `CITATION.cff` and
  `plugins/empyrean/.claude-plugin/plugin.json`; `prepare-release.yml` verifies they agree
  before a tag. Never bump a version as part of another change.
- The `release` branch is written only by the release workflow, which
  fast-forwards it to a final tag once PyPI and crates.io have both
  published; the plugin marketplace entry and Anthropic's plugin directory
  serve `plugins/empyrean` from it. Never push to `release` by hand.

## Where things are documented

- API reference and guides: <https://docs.empyrean-dynamics.com>
  (plain text for agents: `/llms.txt`, everything in one file: `/llms-full.txt`).
- Validation against JPL Horizons, ASSIST and the other channels:
  <https://validation.empyrean-dynamics.com/latest/validation_report.html>.
- Nine runnable scenarios with reference values:
  <https://github.com/Empyrean-Dynamics/empyrean-scenarios>.
