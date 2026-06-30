# C_switch Decomposition — MIG 1g.10gb Slice (a100-mig)

_C_switch decomposition by change type. Pinned_Build: b9412-6-g764f1e64a | Platform: a100-mig | Run_Repeat count: 3_

| Change type | Teardown (ms) | Boot (ms) | Warmup (ms) | C_switch total (ms) |
| --- | --- | --- | --- | --- |
| quant-only | 214.03 ± 0.06 | 2121.24 ± 12.24 | 138.45 ± 5.13 | 2473.71 ± 8.73 |
| ctx-only | 214.00 ± 0.05 | 2125.78 ± 15.50 | 139.82 ± 5.25 | 2479.60 ± 13.07 |
| slot-only | 214.03 ± 0.12 | 2124.17 ± 9.56 | 139.39 ± 4.45 | 2477.59 ± 8.82 |
| combined | 214.03 ± 0.08 | 2124.00 ± 14.19 | 139.94 ± 4.52 | 2477.97 ± 11.13 |

> Measured on a single MIG 1g.10gb slice of a 80 GB A100 (UUID:
> `MIG-ffd3804e-0c50-5327-a380-a6635cb1a15d`, ~9.5 GB usable, ~1/7 compute).
> Run under idle-box conditions immediately after the a100-cuda run.
> Model: Llama-3.1-8B (Q4_0 & Q4_K_M), ctx {2048, 4096}, slots {1, 4} —
> 2 repeats over 56 transitions.

---

## Analysis

### Headline numbers

Every reconfiguration on the MIG slice costs **~2.47–2.48 seconds**. Same three
phases as the full A100, but with notable differences:

- **Teardown** ~214 ms — **identical to the full A100** (214.01 vs 214.03 ms).
  This confirms teardown is purely CPU/OS and is completely unaffected by GPU
  partitioning or slice capacity.
- **Boot** ~2121–2126 ms — ~210 ms (~10%) slower than the full A100 (~1910 ms).
  However, the **variance collapses dramatically**: std is ±10–16 ms on MIG
  vs ±267–301 ms on the full A100. The MIG slice has a dedicated, fixed memory
  region and its own CUDA context; once the slice is provisioned, host→device
  weight copies and context init are highly deterministic.
- **Warmup** ~138–140 ms — **~3.2× slower** than the full A100 (~44 ms). This is
  the compute-bound phase (one prefill + one decode step), and it directly reflects
  the ~7× smaller SM count of the 1g.10gb slice relative to the full GPU.

### Per-knob comparison

The MIG data is the clearest per-knob result we have — variance is so low that even
tiny differences are visible:

| | quant-only | ctx-only | slot-only | combined |
|---|---|---|---|---|
| Total (ms) | 2473.7 | 2479.6 | 2477.6 | 2477.97 |
| Boot std | ±12 ms | ±16 ms | ±10 ms | ±14 ms |

All four rows are within **~6 ms of each other**, well inside the ±10–16 ms boot
std. On MIG, the knob type makes **no measurable difference** to C_switch cost.

### Why MIG gives tighter measurements

The MIG slice has an isolated memory partition. Every boot:
- copies the same ~4.5 GB model to the same dedicated DRAM region,
- initialises the same fixed CUDA context,
- allocates the same pre-sized KV cache.

There is no contention from other slices for memory bandwidth (the MIG memory
partitioning is hardware-enforced at the HBM level). The full A100 shares PCIe,
system RAM, and CUDA driver resources with all other processes, causing higher
variance. For reproducibility, MIG slices are better controlled environments.

### Full A100 vs MIG comparison summary

| Phase | Full A100 | MIG 1g.10gb | Ratio |
|---|---|---|---|
| Teardown | 214 ± 0.1 ms | 214 ± 0.1 ms | **1.00× (identical)** |
| Boot | 1910 ± 270 ms | 2123 ± 12 ms | 1.11× (MIG slower, less noisy) |
| Warmup | 44 ± 3 ms | 139 ± 5 ms | **3.2× (MIG slower — less compute)** |
| **Total** | **2170 ± 265 ms** | **2478 ± 10 ms** | **1.14×** |

**Key insight:** MIG is only 14% slower in total C_switch despite having ~1/7 the
compute, because **boot (PCIe-bound, shared) dominates**. The 5× compute
disadvantage only becomes visible in the warmup phase (~15% of total). This is the
core argument for why C_switch is an "irreducible" cost: even with dedicated
isolated hardware, a ~2.5 s downtime floor remains — it is driven by process
restart + weight loading, not compute capacity.

### Device isolation verified

The matching teardown (214 ms on both devices) and the 3.2× warmup gap provide
two independent confirmations that device pinning works correctly:
- teardown identical → the CPU op is device-independent (expected)
- warmup 3.2× gap → compute is genuinely isolated to the slice (verified)

The steady-state decode throughput gap (151 tok/s full A100 vs 29 tok/s MIG,
measured separately) further confirms 5.2× compute isolation.
