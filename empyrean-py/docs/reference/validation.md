# Validation

empyrean is validated continuously — every delivery channel
(Rust / C / Python / CLI) is checked against the others, and the
underlying physics is checked against independent external references
(JPL Horizons and ASSIST) on a shared set of test orbits spanning
NEOs, main-belt asteroids, comets, and atmospheric impactors.

**Live results:
[validation.empyrean-dynamics.com](https://validation.empyrean-dynamics.com)**
— the full cross-channel suite, regenerated on every release, with
per-row precision, σ on every published number, and the reference for
each comparison.

## What "validated" means here

Production astrodynamics needs more than "the build passes." Two
guarantees sit behind every number empyrean returns:

- **Cross-channel fidelity** — all four delivery channels run the same
  integration kernel through one shared C ABI, so they return
  bit-identical results from identical inputs. This is the *structural*
  guarantee: the language you call from never changes the answer.
- **External-reference agreement** — those numbers match ASSIST
  (Holman et al. 2023, the open-source REBOUND/IAS15 test-particle
  integrator) and JPL Horizons at the stated precision. This is the
  *physics* guarantee: the answers are right, not just self-consistent.

A separate effect to keep in mind is **trajectory sensitivity**: a
nominal orbit propagated through a deep planetary close approach
amplifies tiny seed-state differences (down to last-bit `f64` rounding)
along the post-flyby chord. That divergence is the dynamics being
chaotic, not a bug — but it means two runs from slightly different seed
states can disagree by kilometers a few weeks past a close encounter.
The live suite quantifies this per object.

## Before you trust a critical number

For any workflow that hinges on impact probability or B-plane geometry,
check your reference orbit against the live suite before relying on the
output for that specific orbit — close-approach geometry is exactly
where trajectory sensitivity bites hardest.
