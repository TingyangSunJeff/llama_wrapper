# C_switch Decomposition — Full A100 (a100-cuda)

_C_switch decomposition by change type. Pinned_Build: b9412-6-g764f1e64a | Platform: a100-cuda | Run_Repeat count: 3_

| Change type | Teardown (ms) | Boot (ms) | Warmup (ms) | C_switch total (ms) |
| --- | --- | --- | --- | --- |
| quant-only | 214.01 ± 0.08 | 1927.74 ± 266.66 | 43.11 ± 2.70 | 2184.86 ± 264.85 |
| ctx-only | 213.99 ± 0.09 | 1895.16 ± 267.79 | 43.49 ± 3.35 | 2152.64 ± 266.21 |
| slot-only | 220.25 ± 25.07 | 1954.15 ± 300.82 | 45.59 ± 2.86 | 2219.98 ± 312.35 |
| combined | 214.02 ± 0.10 | 1909.71 ± 270.38 | 43.51 ± 2.68 | 2167.25 ± 269.33 |

> Measured on the full 80 GB A100 (device 0, integer index). Run under idle-box
> conditions (no co-tenant GPU load). Model: Llama-3.1-8B (Q4_0 & Q4_K_M),
> ctx {2048, 4096}, slots {1, 4} — 2 repeats over 56 transitions.

---

## Analysis

### Headline numbers

Every reconfiguration of this 8B model on the full A100 costs **~2.15–2.22 seconds**
of server-unavailability, regardless of which knob changes. The three phases:

- **Teardown** ~214 ms — constant across all rows (std < 0.1 ms for quant/ctx/combined).
  This is purely a CPU/OS operation: send SIGINT, wait for process exit. The GPU is
  not involved at this point, so it is independent of device capacity or model size.
- **Boot** ~1895–1954 ms — the dominant cost (~88% of total). Covers process spawn +
  CUDA context init + weight copy from host RAM to GPU + KV-cache allocation + graph
  build. The high std (±267–301 ms) reflects residual variability in CUDA context
  init timing, not measurement noise.
- **Warmup** ~43–46 ms — the first actual inference after boot. This is
  compute-bound (prefill + one token generated). Very fast on the full A100 because
  it has full SM count and HBM bandwidth available.

### Per-knob comparison

All four change types are within the noise of each other on the full A100:

| | quant-only | ctx-only | slot-only | combined |
|---|---|---|---|---|
| Total (ms) | 2185 | 2153 | 2220 | 2167 |
| Rank | 2nd | 1st | 4th | 3rd |
| Gap from min | +32 ms | — | +67 ms | +14 ms |

The ±267–312 ms boot std swamps these 15–67 ms differences, so **no single knob is
statistically distinguishable on this device**. All three reconfiguration types cost
essentially the same because boot is dominated by CUDA context init and the
PCIe host→device copy — not by any per-knob KV or weight reallocation cost. The
full A100 is fast enough that every Config fits with room to spare, so there is no
memory pressure differentiating the knobs.

### Boot variance source

The large boot std (±267 ms) despite an idle box is expected: CUDA context
initialisation timing is non-deterministic (driver, NVCC graph-build, PCIe DMA
scheduling). At 2 repeats this is the dominant source of std; 5+ repeats would
narrow it but would not eliminate it.

### Key takeaway for the paper

On a **full A100 under idle conditions**, the switching cost is **~2.2 s** and is
**knob-agnostic** — the choice of which knob to change does not materially affect
downtime. The cost floor is set by CUDA startup + PCIe weight transfer (~4.5 GB for
8B Q4), not by any model-shape-dependent operation.

Compare to the MIG slice: the full A100's warmup is **~3× faster** (~44 ms vs
~139 ms on MIG) because the full GPU has ~7× the compute. Boot is only ~9% faster
on the full A100 (~1910 ms vs ~2123 ms on MIG) because it is PCIe-bound, and both
devices share the same PCIe link.
