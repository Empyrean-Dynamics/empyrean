---
description: "Skill-trigger pilot: bennu-approaches"
tags: [skill-trigger, pilot-2026-10-19]
runs: 3
max_turns: 40
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Skill]
---

List every close approach of asteroid 101955 Bennu to Earth between 2026 and 2100 that comes within 0.05 au, with dates and distances, computed in Python rather than looked up from a table.
