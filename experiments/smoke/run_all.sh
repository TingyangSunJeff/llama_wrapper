#!/usr/bin/env bash
# Run all three knob-adaptation smoke tests.
#
# Usage:
#   ./run_all.sh            # Scenario A on CPU (edge-relevant), B and C on GPU
#   NGL=99 ./run_all.sh     # force GPU for all
#
# Requires the conda env python with aiohttp/numpy.
set -euo pipefail

PY="${PY:-/scratch2/tingyang/anaconda/envs/mynewenv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "### Scenario A (GGUF switch / throughput burst) -- CPU"
NGL="${NGL_A:-0}" BURST="${BURST:-12}" MAX_TOKENS="${MAX_TOKENS:-96}" PARALLEL="${PARALLEL:-4}" \
    "$PY" scenario_a.py
echo

echo "### Scenario B (context reshape / document unlock)"
"$PY" scenario_b.py
echo

echo "### Scenario C (parallel slots / anti-blocking)"
"$PY" scenario_c.py
echo

echo "### Scenario D (reconfiguration cost vs. cost-aware control) -- GPU"
NGL="${NGL_D:-99}" "$PY" scenario_d.py
