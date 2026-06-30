# Memory Footprint Decomposition — Full A100 (a100-cuda)

_Memory footprint decomposition by Config. Pinned_Build: b9412-6-g764f1e64a | Platform: a100-cuda | Run_Repeat count: 3_
_Model: Llama-3.1-8B (Q4_0 & Q4_K_M) · GPU: NVIDIA A100 80GB PCIe (device 0) · All values in MiB_

> Buffer sizes are deterministic per Config — the same Config always allocates
> identical buffers — so std = 0 across repeats. The ± column is omitted.

| Config | Weights (MiB) | KV per slot (MiB) | Scratch+overhead (MiB) | **Total (MiB)** |
| --- | ---: | ---: | ---: | ---: |
| Q4_0, c=2048, np=1 | 4156 | 256 | 747 | **5159** |
| Q4_0, c=2048, np=4 | 4156 |  64 | 747 | **4967** |
| Q4_0, c=4096, np=1 | 4156 | 512 | 747 | **5415** |
| Q4_0, c=4096, np=4 | 4156 | 128 | 747 | **5031** |
| Q4_K_M, c=2048, np=1 | 4403 | 256 | 748 | **5407** |
| Q4_K_M, c=2048, np=4 | 4403 |  64 | 748 | **5215** |
| Q4_K_M, c=4096, np=1 | 4403 | 512 | 748 | **5663** |
| Q4_K_M, c=4096, np=4 | 4403 | 128 | 748 | **5279** |

> All 8 configs fit within the 80 GB A100 with large headroom (~5.0–5.7 GB used).

---

## Analysis

### Three independent components

Every llama-server launch allocates three separate GPU memory regions. Their sizes
are **fixed at boot time** and do not change during inference:

**1. Weights (~4156–4403 MiB, ~80–84% of total)**
The model's tensor data, copied once from host RAM to GPU at boot. This is the
largest component and its size depends only on the quantization format:
- **Q4_0: 4156 MiB** (plain 4-bit, ~4.1 GB)
- **Q4_K_M: 4403 MiB** (k-quant 4-bit, ~4.3 GB, +247 MiB = +6%)

Q4_K_M is slightly larger because it uses a hierarchical super-block layout that
packs a small amount of extra scale metadata alongside the 4-bit weights. Both
formats are memory-comparable (< 5 GB for an 8B model) and leave > 74 GB free on
the A100. Changing `ctx_length` or `slot_count` has **zero effect** on weights —
the model is loaded once and shared.

**2. KV per slot (~64–512 MiB, ~1–10% of total)**
The Key-Value cache holds the attention context for in-flight sequences. Its total
size is:
```
KV_total = ctx_length × (n_heads_kv × head_dim × 2 bytes) × n_layers
         ≈ ctx_length × 0.125 MiB/token   (for this 8B Llama-3.1 at fp16)
```
So KV_total = 256 MiB at ctx=2048 and 512 MiB at ctx=4096. The **per-slot** value
is `KV_total / slot_count`:

| ctx \ slots | np=1 | np=4 |
|---|---|---|
| 2048 | 256 MiB | 64 MiB |
| 4096 | 512 MiB | 128 MiB |

This is the **core context-vs-concurrency trade-off**: you can have a few users
with long context, or many users with shorter context, within the same KV budget.
The slot-reshape knob shifts the partition — it does not increase total KV, it
redistributes it. Total KV is the same whether you use np=1 or np=4; what changes
is how much context each user gets.

**3. Scratch + overhead (~747–748 MiB, ~13–15% of total)**
CUDA compute buffers and compute graph allocations needed for each forward pass.
This is essentially constant across all configs for a given model (258 MiB for the
primary CUDA compute buffer + ~489 MiB allocator/context overhead). It does not
scale with ctx or slots — it depends only on the model architecture.

### What "Total" means for the controller

The total footprint determines whether a Config is feasible on a given device.
For this 8B model the feasibility range is:

| Range | Value |
|---|---|
| Minimum (Q4_0, c=2048, np=4) | 4967 MiB (~4.8 GB) |
| Maximum (Q4_K_M, c=4096, np=1) | 5663 MiB (~5.5 GB) |
| Full A100 total | 81920 MiB (80 GB) |
| Headroom (min config) | ~93% free |

All 8 configs are feasible on the full A100 with comfortable margin. The MIG
1g.10gb slice (~9728 MiB) could also fit all 8 configs (max ~5663 MiB < 9728 MiB),
but tighter (58% headroom vs the full GPU's 93%). An 8B Q8_0 model (~8 GB weights
alone) would OOM on the MIG slice — that is the shortfall the Memory_Profiler would
record at runtime.

### Key insight for the paper

The **knob-to-memory** mapping is the foundation of the controller's action space:

- **Quant knob** controls weights (the biggest component). Switching Q4_0 → Q4_K_M
  costs +247 MiB but gains quality; this is the only way to change weights without
  reloading.
- **ctx knob** controls KV_total. Doubling ctx (2048 → 4096) doubles KV_total
  (+256 MiB). This has no effect on weights or compute.
- **slot knob** controls the per-user KV budget. More slots = less context per user
  at the same KV_total. No additional memory is consumed — it is a partition choice.
- **None of the knobs are free**: each forces a full server restart (~2.2 s C_switch
  on the full A100) because all three allocations are fixed at boot time and cannot
  be resized live.

The controller therefore picks a point on this memory surface that fits the
device, subject to the C_switch cost of moving between points.
