---
description: "Skill-trigger pilot: apophis-covariance"
tags: [skill-trigger, pilot-2026-10-19]
runs: 3
max_turns: 40
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill]
---

Propagate the orbit covariance of asteroid 99942 Apophis from its current epoch to its April 2029 Earth close approach in Python and report the 1-sigma position uncertainty at that time in km.
