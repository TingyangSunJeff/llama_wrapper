# profile-suite

Research-grade profiling infrastructure for the paper
**"Switching-Cost-Aware Configuration Control for llama.cpp-Based Edge LLM Serving"**.

This suite turns the qualitative claims sketched by `experiments/smoke/` into paper-grade,
reproducible measurements: switching-cost (C_switch) decomposition, the config→performance
surface, the memory footprint surface, and the quality axis of the quant knob — all produced
under a reproducibility harness across the paper's platform set (A100 GPU now, Jetson-class
edge device as the stated target).

## Scope and isolation

This is **research infrastructure, not an upstream contribution**. Everything lives entirely
under `profile/` (`/scratch2/tingyang/llama.cpp/profile/`). The suite never creates, modifies,
or deletes any file outside `profile/`, and it does not touch the llama.cpp C++ engine.

## Target interpreter

All commands use:

```
/scratch2/tingyang/anaconda/envs/mynewenv/bin/python
```

## Layout

```
profile/
├── README.md
├── pyproject.toml          # package metadata; deps: aiohttp, pyyaml, matplotlib, numpy (dev: hypothesis)
├── profile_suite/          # importable package
│   ├── harness/            # shared measurement harness (server, client, sysprobe, logparse, stats, repro)
│   ├── modules/            # measurement modules (switch_cost, performance, memory, quality)
│   ├── platform/           # platform adapters (A100 CUDA, Jetson)
│   └── reporting/          # results.json + figures + tables
├── campaigns/              # input campaign definitions (YAML/JSON, git-tracked)
├── runs/                   # campaign-scoped outputs (manifests, raw logs, results, artifacts)
└── tests/                  # unit, property-based (Hypothesis), and slow A100 integration tests
```

## Development

Install (editable) with dev extras using the target interpreter, then run the fast test suite:

```
/scratch2/tingyang/anaconda/envs/mynewenv/bin/python -m pip install -e ".[dev]"
/scratch2/tingyang/anaconda/envs/mynewenv/bin/python -m pytest -m "not slow"
```
