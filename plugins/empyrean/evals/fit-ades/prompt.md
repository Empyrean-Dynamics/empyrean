---
description: "Skill-trigger pilot: fit-ades"
tags: [skill-trigger, pilot-2026-10-19]
runs: 3
max_turns: 40
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill]
---

There is an ADES PSV file of optical astrometry at ./observations.psv. Fit an orbit to it in Python and report the post-fit RMS of the residuals in arcseconds and the orbital elements with their uncertainties.
