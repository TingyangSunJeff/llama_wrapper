# Knob-Adaptation Smoke Tests

Scenarios A/B/C **show the gap**: each boots two *static* `llama-server` configs
and shows the config that is wrong for the scenario loses badly while the matching
one wins. That gap motivates runtime knob adaptation. They do **not** count the
cost of changing a knob.

Scenario D **adds the reconfiguration cost back in** and shows when a
switching-cost-aware control layer actually helps (and when naive adaptation
hurts). It is the bridge from "the gap exists" to "a controller is worth building."

| Scenario | Knob | Wrong config | Right config | Gap shown |
|---|---|---|---|---|
| A | model / quantization file | `Q8_0` | `Q4_K_M` | clears a burst faster (CPU) |
| B | context length (`-c`) | `ctx=2048` | `ctx=32768` | unlocks a long-doc task 2K cannot do |
| C | parallel slots (`-np`) | `np=1` | `np=4` | interactive TTFT drops ~58x |
| D | model file + **C_switch** | `static` / `always-switch` | `cost-aware` | controller wins across dwell times |

## Requirements

- A built `llama-server` (default: `build-cuda/bin/llama-server`).
- Models in `models/`: `gemma-3-1b-it-Q4_K_M.gguf`, `gemma-3-1b-it-Q8_0.gguf`.
  Scenario D defaults to the 8B pair
  (`Meta-Llama-3.1-8B-Instruct-{Q4_K_M,Q8_0}.gguf`) when present, for a clearer
  GPU regime sign-flip.
- Python with `aiohttp` (the `mynewenv` conda env has it).

## Run

```bash
# one at a time
/scratch2/tingyang/anaconda/envs/mynewenv/bin/python scenario_a.py
/scratch2/tingyang/anaconda/envs/mynewenv/bin/python scenario_b.py
/scratch2/tingyang/anaconda/envs/mynewenv/bin/python scenario_c.py
NGL=99 /scratch2/tingyang/anaconda/envs/mynewenv/bin/python scenario_d.py

# or all four
./run_all.sh
```

Knobs via env: `NGL` (GPU layers, 0 = CPU), `BURST`, `MAX_TOKENS`, `PARALLEL`
(Scenario A only). Scenario D also takes `LOW_N`, `DWELLS` (comma list of phase
durations, s), `N_PHASES`. Server binary via `LLAMA_SERVER`, models via
`LLAMA_MODEL_DIR`.

## Results observed

### gemma-3-1b-it (small model)

- **Scenario A** — CPU: `Q4_K_M` cleared a 12-user burst **1.25x** faster than
  `Q8_0` (8.65s vs 10.81s). On A100 GPU there was no throughput win (≈0.96x).
- **Scenario B** — `ctx=2048` rejected a 28K-token prompt with HTTP 400
  (`exceed_context_size_error`); `ctx=32768` ingested it and returned the buried
  passcode `MAGENTA-7731`. Feasibility gap, not a speed gap.
- **Scenario C** — interactive request TTFT was **4.44s** at `np=1` (head-of-line
  blocked behind a 512-token batch job) vs **0.08s** at `np=4` (continuous
  batching). ~**58x**.

### Meta-Llama-3.1-8B-Instruct (bigger model, A100 GPU) — Scenario A re-run

| metric | Q8_0 | Q4_K_M |
|---|---:|---:|
| burst makespan (np=8, 24 users, 256 tok) | **13.45s** | 17.58s |
| aggregate throughput (batched) | **456 tok/s** | 349 tok/s |
| single-stream tg128 (llama-bench, batch=1) | 131 t/s | **151 t/s** |
| VRAM used | 8864 MiB | **5664 MiB** |

## The model-file knob is regime-dependent (key finding)

The notes assume "switch to `Q4_K_M` to clear a burst 2-3x faster." Measurements
show this is **not** universally true — the sign of the throughput effect flips
with hardware and batch size:

- **Single-stream / bandwidth-bound** (batch≈1, or CPU): `Q4_K_M` is faster
  because token generation is dominated by weight memory traffic and Q4 weights
  are smaller. (8B: 151 vs 131 t/s; gemma-1B on CPU: 1.25x.)
- **Batched / compute-bound** (high concurrency on a datacenter GPU): `Q8_0` is
  often *faster*, with the largest gap in the **batch 4-12** window (8B: Q8 536
  vs Q4 373 tok/s at batch 8, +44%). This is **not** a simple "dequant is
  expensive" effect — it is driven by llama.cpp's batch-size-dependent CUDA
  kernel selection (GEMV at batch 1, per-type MMQ integer kernels for
  batch < 64, dequant + cuBLAS for batch >= 64). The relationship is
  non-monotonic. See `scenario_a_findings.md` for the full sweep, the source
  references, and the correction of an earlier oversimplified claim.
- **VRAM** is the one consistent Q4 win (8B: 3.2 GB less). On a big GPU that is
  the real lever: the freed memory buys more KV slots or longer context
  (capacity), which is an *indirect* throughput/latency benefit, not a per-token
  speedup.

Implication for the paper: the GGUF-file knob should be modeled as a
**throughput-vs-VRAM trade-off whose direction depends on the operating regime
(hardware + concurrency)**, not as a simple "smaller = faster" switch. A
switching-cost-aware controller must know which regime it is in.

## Scenario D: reconfiguration cost vs. a cost-aware control layer

**Terms used below:**
- **regime** — which workload situation we are in: `LOW` (one user at a time) or
  `BURST` (many concurrent users). The workload alternates between them.
- **sign-flip** — the faster config *reverses* with the regime (Q4 wins LOW, Q8
  wins BURST). If one config won both, you would never switch.
- **`C_switch`** — server downtime to change the knob (kill old server, boot new,
  warm up). Here ≈ 2.8s, during which nobody is served.
- **dwell** — how long a regime lasts before it flips (5s = fast-changing load;
  300s = stable load).
- **goodput** — useful tokens served over the whole run (higher = better); the
  score each policy gets.
- the four **policies**: `static-Q4`/`static-Q8` (never change), `always-switch`
  (adapt every phase but ignore `C_switch`), `cost-aware` (adapt only when it
  beats the downtime — the control layer we are arguing for).

A/B/C ignore the cost of *changing* a knob. Scenario D measures it and asks when
adapting pays off. It uses the quant knob (regime-dependent per Scenario A), a
workload that alternates between **LOW** (single-stream, Q4 wins) and **BURST**
(np=8 concurrent, Q8 wins), and replays that trace under four policies while
charging each reconfiguration its measured downtime.

What it measures live (8B pair, A100, ngl=99):

- **Regime sign-flip:** Q4 29.5 t/s vs Q8 24.0 t/s in LOW; Q8 92–93 t/s vs Q4
  63 t/s in BURST. Neither static config is best in both regimes.
- **Reconfiguration cost** of a GGUF swap: `C_switch = teardown (~0.2s) + boot
  (~2.5s) + first-token warmup (~0.16s) ≈ 2.8s`, the time the server serves
  nobody during a knob change.
- Two break-even dwell times (dwell = how long a regime lasts before it flips):
  - **single-switch break-even ≈ 9s** — when *one* switch into the better config
    pays off, looking only at that phase. This is *not* the threshold at which
    adapting beats a fixed config (see next).
  - **adaptation-pays dwell ≈ 51s** — when the cost-aware policy first actually
    beats the best static config. It is larger because the workload *alternates*:
    every switch into one config implies a later switch *back*, so adaptation
    pays the reload cost on the round trip, not just once.

Trace-replay goodput (6 alternating phases), one representative run:

| dwell | static-Q4 | static-Q8 | always-switch | **cost-aware** |
|---:|---:|---:|---:|---:|
| 5s   | 1387  | 1745   | 874    | **1745** |
| 30s  | 8321  | 10473  | 10013  | **10473** |
| 60s  | 16642 | 20945  | 20981  | **21013** |
| 300s | 83211 | 104726 | 108722 | **108722** |

Reading:

- **Short dwell (regimes flip fast):** `always-switch` is the *worst* policy
  (874) — it pays reload downtime it never recoups. The cost-aware layer
  **declines to switch**, matching the best static (1745), i.e. +100% over naive
  adaptation.
- **Mid dwell (e.g. 30s):** `always-switch` (10013) is *still worse* than just
  staying on Q8 (10473), even though 30s is well above the 9s single-switch
  break-even. Why: `always-switch` also detours *into Q4 for every LOW phase*,
  and Q4's small LOW edge (+81 tokens) does not cover the cost of switching back
  to Q8 for the next BURST phase (~2.8s × 92 t/s ≈ 262 tokens lost). The
  cost-aware layer **skips that bad detour** and just holds Q8, so it matches the
  best static. This is the crux: the single-switch break-even (9s) is not when
  round-trip adaptation wins (51s).
- **Long dwell (regimes durable):** the round-trip reload finally amortizes; the
  cost-aware layer **adapts** and beats the best static (108722 vs 104726, ~+4%).
- The cost-aware layer ties-or-wins at *every* dwell — it is the only policy good
  across both regimes. What makes that possible is **counting `C_switch`**: drop
  it and you get `always-switch` (loses at short/mid dwell); ignore adaptation and
  you get a static config (loses at long dwell).

Honest caveats:

- The long-dwell adaptation gain here is **modest (~4%)** because on the A100 Q8
  is a strong all-rounder (its BURST advantage dwarfs Q4's LOW advantage). The
  *dramatic* win is vs `always-switch` at short dwell. On a more balanced
  sign-flip (e.g. a memory-constrained edge GPU where Q4 frees VRAM for more
  slots/longer context, or CPU where Q4 decode wins decisively) the static-vs-
  adaptive gap widens.
- `cost-aware` is computed as the *optimal* keep-vs-reconfigure policy for the
  measured `C_switch` (DP over the config held per phase, with a regime-duration
  estimate). A deployed online controller approximates this; the experiment shows
  the achievable target, not a learned policy. This is the same accounting prior
  free-switching work skips by assuming `C_switch≈0` (which is exactly the
  `always-switch` column).
- Per-config/per-regime rates and `C_switch` are measured **live**; the
  dwell-time sweep is a deterministic accounting on top of those measurements
  (not a live mid-trace reconfiguration, which would be slower/flakier for a
  smoke test).

## Notes / limitations
- These are smoke tests: small model, short runs, single machine, no warmup
  control beyond `--no-warmup`, latency measured client-side.
- No adaptive policy is implemented in A/B/C. Each compares two fixed configs.
  Scenario D adds a policy comparison (static vs always-switch vs cost-aware) but
  the cost-aware policy is an accounting optimum over live-measured rates and
  `C_switch`, not a learned/online controller.
- Scenario B's gap is correctness/feasibility (truncation/rejection), which is
  arguably a stronger argument than latency: the wrong config cannot do the task.
