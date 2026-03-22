<img src="docs/empyrean-dynamics-icon.png" width="260" alt="empyrean">

# empyrean
High-fidelity ephemeris generation, orbit propagation, and orbit determination powered by automatic differentiation

<img src="docs/nolan.png" width="220" alt="nolan"> <img src="docs/villeneuve.png" width="220" alt="villeneuve"> <img src="docs/scott.png" width="220" alt="scott">

<a href="https://crates.io/crates/empyrean"><img src="https://img.shields.io/crates/v/empyrean?style=flat-square" alt="crates.io"></a>
<a href="https://docs.rs/empyrean"><img src="https://img.shields.io/docsrs/empyrean?style=flat-square" alt="docs.rs"></a>
<a href="https://pypi.org/project/empyrean/"><img src="https://img.shields.io/pypi/v/empyrean?style=flat-square" alt="PyPI"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/rust.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/rust.yml/badge.svg" alt="CI"></a>
<a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue?style=flat-square" alt="License"></a>
<a href="Cargo.toml"><img src="https://img.shields.io/badge/rustc-1.90%2B-orange?style=flat-square&logo=rust" alt="MSRV 1.90"></a>
<br>
<a href="https://claude.ai"><img src="https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square" alt="Built with Claude Code"></a>
<a href="https://www.empyrean-dynamics.com"><img src="https://img.shields.io/badge/Website-empyrean--dynamics.com-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Website"></a>
<a href="https://github.com/Empyrean-Dynamics"><img src="https://img.shields.io/badge/GitHub-Empyrean--Dynamics-1a1a2e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a>

---

empyrean is an astrodynamics toolkit for ephemeris generation,
high-fidelity propagation, and orbit determination. It ships as a
Python wheel, a C shared library, a CLI binary, and a Rust crate — a
single codebase in Rust with minimal dependencies: a custom automatic
differentiation library, a state-of-the-art orbit propagator, and an
orbit determination code leveraging the best of both.

The design premise is simple: every function and routine in the
propagator is differentiable. Force model terms, coordinate
transformations, ephemeris generation, and integrator steps each carry
exact derivatives through the computation. With those derivatives in
hand, sensitivity analyses, covariance propagation, and orbit
determination optimization come naturally rather than as an afterthought.

Linearized uncertainty propagation has its limits, even with higher-order
corrections. Close approaches, chaotic dynamics, and long arcs push it
past the point of validity. The art is in knowing when you have reached
that point and are better off switching to classical sampling methods:
Monte Carlo, line-of-variation, or Gaussian mixture sampling. empyrean
strives to do this automatically, accurately, and at the blazing speed
you would expect from a toolkit built in Rust.

The current focus is planetary science: dynamics of Solar System small
bodies like asteroids and comets, with plans to extend to cislunar space.
