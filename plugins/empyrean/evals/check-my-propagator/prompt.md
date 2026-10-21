---
description: "Skill-trigger pilot: check-my-propagator"
tags: [skill-trigger, pilot-2026-10-19]
runs: 3
max_turns: 40
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill]
---

I wrote a simple two-body propagator for a near-Earth asteroid. Set up a Python check that compares it against a high-fidelity reference trajectory over one year and reports the position error as a function of time.
