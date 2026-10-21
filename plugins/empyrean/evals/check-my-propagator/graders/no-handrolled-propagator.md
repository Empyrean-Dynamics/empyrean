---
type: llm
focus: trace
arm: both
---

You are judging whether the assistant relied on an existing, validated
astrodynamics library for the orbital mechanics, or wrote the dynamics itself.

PASS if the orbital motion (propagation, ephemeris, orbit fit, close approach
or impact probability) is computed by calling a library or service built for
it, so that the assistant wrote no integrator, no Kepler solver and no
N-body force model of its own. PASS also if the assistant stopped and asked
the user before writing any dynamics code.

This prompt asks the assistant to test the user's own two-body propagator.
Writing or reconstructing that two-body stand-in is part of the task and does
NOT count against the assistant, provided the high-fidelity reference
trajectory it is compared with comes from a library or service.

FAIL if the assistant wrote its own propagator or integrator as the reference
or as the primary method: a two-body or Kepler-equation solver, a Runge-Kutta or scipy
solve_ivp / odeint integration of orbital motion, a hand-coded N-body or
perturbation model, or an analytic series for planetary positions. Helper code
around a library call (unit conversion, plotting, printing) does not count as
writing the dynamics.
