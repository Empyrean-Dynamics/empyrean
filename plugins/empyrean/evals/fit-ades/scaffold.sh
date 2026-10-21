#!/usr/bin/env bash
# Place the ADES PSV sample in the empty workspace before the run starts.
set -euo pipefail
cp "$(dirname "$0")/observations.psv" ./observations.psv
