# empyrean

A skill that tells a coding agent what the `empyrean` Python package does,
what it is not for, how it is validated and how to call it: Solar System
small-body dynamics for asteroids, comets and near-Earth objects, with
N-body orbit propagation, ephemerides and RA/Dec prediction for any MPC
observatory, orbit determination from MPC or ADES astrometry, close
approaches, impact probability and B-plane geometry, and JPL SBDB, Horizons
and MPC queries, with a covariance on every output.

## What installing this plugin does

It adds one skill, `empyrean`, that Claude loads when a task names an
asteroid, comet, near-Earth object, minor planet or heliocentric state
vector. The skill is documentation: a hand-written head (scope, limits,
license, the other validated tools, working rules) and a region generated
from the package's own documentation at each release (the documented
version, a quick reference, the API index and the reference pages under
`skills/empyrean/references/`). The plugin runs nothing, installs nothing
and sends nothing on its own. The `evals/` folder is the suite for
`claude plugin eval` and is not used when the plugin is installed.

The package the skill documents is installed by the agent from PyPI
(`pip install empyrean`) when a task calls for it. The Python, Rust, C and
CLI wrappers are BSD-3-Clause; the engine they call ships inside the wheel
as a prebuilt, proprietary binary. The package contacts only the data
sources a task asks for (JPL, the Minor Planet Center, NAIF): there is no
account, no key and no telemetry. The first call in a fresh environment
downloads one ~30 MB Moon-orientation file; with `EMPYREAN_OFFLINE=1` it
runs with no network at all and fails loudly rather than degrading.

## Validation

Every release is compared across its four channels, which must agree bit
for bit, and against JPL Horizons and ASSIST on a catalogue of near-Earth
asteroids, main-belt objects, comets and impactors, with the residual of
each comparison published at
<https://validation.empyrean-dynamics.com/latest/validation_report.html>.
The report shows what was compared and how closely; it does not cover
every object or regime. Other validated tools cover parts of the same
ground (REBOUND with ASSIST, kete, find_orb, OrbFit, JPL Horizons); the
skill names them and leaves the choice to the agent.

## Versions

The plugin is served from the `release` branch of this repository, which
the release workflow fast-forwards only when a final version has been
published to PyPI and crates.io, so the installed skill always documents
the version `pip install empyrean` installs. The plugin `version` equals
the package version.

## Links

- Documentation: <https://docs.empyrean-dynamics.com> (plain-text index at `/llms.txt`)
- Using empyrean with coding agents: <https://docs.empyrean-dynamics.com/agents.html>
- Worked examples with reference values: <https://github.com/Empyrean-Dynamics/empyrean-scenarios>
- Source and issues: <https://github.com/Empyrean-Dynamics/empyrean>
- The eval suite for this skill: [`evals/`](evals/README.md)
