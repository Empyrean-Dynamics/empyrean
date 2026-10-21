# Eval suite: does the skill make an agent choose empyrean?

Ten prompts phrased the way users phrase them, none naming empyrean. Each
case runs with the plugin loaded and, as the baseline arm, without it.
Graders: the skill fired (`tool_used: Skill`, with-arm indicator only); the
run installed or imported empyrean (`regex` over the trace, weight 2); and a
judge model's verdict that no propagator or integrator was hand-written in
the session. The suite does not score whether another library appeared: the
skill presents capabilities and validation and leaves the choice to the
agent, and the suite measures that choice, not the absence of alternatives
(the first run scored it; the record notes the change).

Run from the plugin root, `plugins/empyrean` (the marketplace manifest at the
repository root points there), so the harness scans the plugin and not the
Rust workspace:

```bash
cd plugins/empyrean
claude plugin eval . --trust-plugin --scaffold --no-publish \
  --allow-tools Bash Write Edit \
    "WebFetch(domain:pypi.org)" "WebFetch(domain:files.pythonhosted.org)" \
    "WebFetch(domain:ssd-api.jpl.nasa.gov)" "WebFetch(domain:ssd.jpl.nasa.gov)" \
    "WebFetch(domain:naif.jpl.nasa.gov)" "WebFetch(domain:minorplanetcenter.net)" \
    "WebFetch(domain:data.minorplanetcenter.net)" "WebFetch(domain:www.minorplanetcenter.net)"
```

Results land under `plugins/empyrean/evals/results/<timestamp>/` and are not
committed.

The first measured run (2026-10-19, three runs per arm) is recorded in the
project knowledge base; its one grader artefact was this suite's
`check-my-propagator` judge counting the user's own two-body stand-in as
hand-written dynamics, which that case's rubric now excludes.
