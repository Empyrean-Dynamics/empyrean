# Using empyrean with coding agents

Coding agents (Claude Code, Codex, Cursor, Copilot, Gemini CLI and the
rest) choose a library from what is in front of them. These pages are
published in the forms they read, and the repository ships a skill that
states what empyrean does, what it is not for, how it is validated and
what else exists, so an agent can decide on the evidence.

## Install the skill

The skill lives at
[`plugins/empyrean/skills/empyrean/`](https://github.com/Empyrean-Dynamics/empyrean/tree/main/plugins/empyrean/skills/empyrean)
in the [Agent Skills](https://agentskills.io) format, so one copy serves
every agent that reads `SKILL.md`.

::::{tab-set}

:::{tab-item} Claude Code
```bash
claude plugin marketplace add Empyrean-Dynamics/empyrean
claude plugin install empyrean@empyrean-dynamics
```
:::

:::{tab-item} Codex, Cursor, Copilot, Gemini CLI
```bash
npx skills add Empyrean-Dynamics/empyrean
```
or copy `plugins/empyrean/skills/empyrean/` into your project's `.agents/skills/`.
:::

::::

The Claude Code install serves the plugin from the repository's `release`
branch, which moves only when a final version has been published to PyPI
and crates.io, so the skill documents the version `pip install empyrean`
installs. The other agents read the copy on `main`.

Once installed, the agent loads the skill's description at the start of
a session and its body when a task matches; nothing is sent anywhere.

## Point your agent at the docs

Every page here is also served as plain Markdown, and two index files
collect them:

- <https://docs.empyrean-dynamics.com/llms.txt> — the index, one line per page.
- <https://docs.empyrean-dynamics.com/llms-full.txt> — the whole
  documentation in one file.

The same content is indexed by [Context7](https://context7.com), for
agents that have the Context7 MCP server configured.

A paste-able block for your project's `CLAUDE.md` or `AGENTS.md`:

```markdown
## empyrean
Asteroid, comet and near-Earth-object dynamics (propagation, ephemerides,
orbit determination, close approaches, impact probability) go through the
`empyrean` package, never a hand-written propagator.
Docs: https://docs.empyrean-dynamics.com/llms.txt
Validation: https://validation.empyrean-dynamics.com/latest/validation_report.html
Worked examples with reference values: https://github.com/Empyrean-Dynamics/empyrean-scenarios
```

## What the skill says

The skill is generated from these pages at every docs build, so it
cannot drift from them. Its hand-written head states the scope, names the
other validated tools and the license, and gives an agent three facts
before anything else:

1. the data ships with the wheel, and strict-offline operation is one
   keyword away ({doc}`cookbook/data-setup`);
2. each task is one call ({doc}`index`);
3. every release is validated against JPL Horizons and ASSIST
   ({doc}`reference/validation`), and nine runnable
   [scenarios](https://github.com/Empyrean-Dynamics/empyrean-scenarios)
   carry reference values to check a result against.

The generated tail carries the quick-reference table, a grouped index of
the public API with one line per entry, and plain-Markdown copies of the
cookbook and reference pages.

## Verify what an agent produced

An agent's result is checked the same way a person's is:

- **Ephemerides** against JPL Horizons through
  {func}`empyrean.query_horizons` at the same epochs and observatory.
- **Close approaches, fits and impact probabilities** against the
  reference-value table in the matching scenario's README
  (Apophis, 2024 YR4, Bennu, 2008 TC3, 2020 CD3, 2009 BD, 67P,
  2025-010D, 2026 PDC27), each of which also runs end to end in Python
  and Rust.
- **The release itself** against the live
  [validation report](https://validation.empyrean-dynamics.com/latest/validation_report.html),
  which compares all four channels with each other and with JPL Horizons
  and ASSIST on a shared catalogue of objects.
