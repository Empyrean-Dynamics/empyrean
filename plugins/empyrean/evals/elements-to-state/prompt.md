---
description: "Skill-trigger pilot: elements-to-state"
tags: [skill-trigger, pilot-2026-10-19]
runs: 3
max_turns: 40
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill]
---

An asteroid has heliocentric ecliptic J2000 Keplerian elements a = 1.1 au, e = 0.19, i = 3.3 deg, Omega = 204 deg, omega = 126 deg, M = 100 deg at epoch 2026-01-01 00:00 TDB. In Python, compute its heliocentric Cartesian state vector on 2027-06-01 00:00 TDB with full planetary perturbations, not two-body motion.
