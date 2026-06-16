# Scenario A deep-dive: why Q8_0 vs Q4_K_M throughput depends on batch size

This documents the investigation into *why* Q8_0 beat Q4_K_M in the Scenario A
server burst (np=8) on the A100, even though Q4 is smaller. It corrects an
earlier oversimplified explanation.

## TL;DR

- The winner between Q8_0 and Q4_K_M is **not** fixed; it depends on the decode
  **batch size** (number of sequences decoded together), and the dependence is
  **non-monotonic**.
- The driver is **CUDA kernel selection by batch size** in llama.cpp, verified
  in source — not a simple "dequantization is expensive" argument.

## Measurements (A100, Meta-Llama-3.1-8B-Instruct, ngl=99)

### Single-op micro-bench (llama-bench)
| test | Q8_0 | Q4_K_M |
|---|---:|---:|
| pp512 (prefill, compute-bound) | 4456 t/s | 4419 t/s (tie) |
| tg128 (decode, batch=1)        | 131 t/s  | **151 t/s** |

### Batched decode throughput, S_TG (llama-batched-bench, npp=16)
| batch | Q8_0 | Q4_K_M | winner |
|---:|---:|---:|---|
| 1  | 135  | 156  | Q4 +16% |
| 2  | 240  | 249  | Q4 +4%  |
| 4  | 386  | 337  | Q8 +15% |
| 6  | 480  | 361  | Q8 +33% |
| 8  | 536  | 373  | **Q8 +44%** |
| 10 | 754  | 715  | Q8 +5%  |
| 12 | 866  | 802  | Q8 +8%  |
| 16 | 951  | 1027 | Q4 +8%  (config-sensitive) |
| 32 | 1592 | 1663 | ~tie    (config-sensitive) |
| 64 | 2491 | 2362 | Q8 +5%  |
| 128| 3158 | 2900 | Q8 +9%  |

The large, reproducible effect is the **batch 4-12 window**, where Q8 is much
faster. The server burst used np=8, landing squarely in that window -> Q8 won
(456 vs 349 tok/s aggregate). At batch >=16 the gap is small and flips with
generation length / KV size, i.e. within run-to-run and config sensitivity.

## Mechanism (verified in ggml/src/ggml-cuda/)

`ggml_cuda_should_use_mmq()` (mmq.cu) on NVIDIA hardware with fp16 tensor cores:

```cpp
return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
```

with `MMQ_DP4A_MAX_BATCH_SIZE = 64` (mmq.cuh). `ne11` is the decode batch size.
There are effectively three matmul paths for quantized weights:

1. **batch == 1**: GEMV path (`mul_mat_vec_q`, MMVQ). Pure memory-bandwidth
   bound -> fewer weight bytes wins -> **Q4_K_M faster** (151 vs 131).
2. **2 <= batch < 64**: **MMQ** integer kernels, one specialized kernel *per
   quant type* with type-specific tiling/granularity
   (`mmq_get_granularity_device`, etc.). Q4_K's MMQ kernel is less efficient
   than Q8_0's in the batch ~4-12 range -> **Q8_0 faster** (peak +44% at 8).
3. **batch >= 64**: dequantize to fp16 + **cuBLAS** GEMM (same path for both
   types). Throughput converges; small residual differences remain.

So the relevant variable is *which kernel family runs and how efficient it is
for that quant type at that batch size*, not the per-weight dequant cost alone.
A monotone "dequant overhead" story cannot explain why Q4 wins at batch 1-2,
loses at 4-12, and ties again at >=16.

## Correction to the earlier claim

Earlier note said: "Under continuous batching the decode becomes compute-bound
and Q4_K_M's superblock dequantization overhead dominates." That is an
oversimplification and is not supported by the non-monotonic data. The accurate
statement is: the Q8/Q4 decode-throughput ordering is governed by llama.cpp's
batch-size-dependent CUDA kernel selection (GEMV vs MMQ vs cuBLAS) and the
per-type efficiency of those kernels.

## Caveats / threats to validity

- All numbers are A100-specific. Compute capability, tensor-core availability,
  and the `MMQ_DP4A_MAX_BATCH_SIZE` threshold change the picture on other GPUs
  and on CPU.
- llama-batched-bench measures steady-state decode at fixed batch; the live
  server has mixed prefill/decode, arrivals, and variable active-slot counts, so
  the effective batch size moves over time.
- Kernel selection thresholds are implementation details that change between
  llama.cpp versions (build b9418 here). Re-verify against the build in use.
- Differences at batch >=16 are within config sensitivity; do not over-read them.

## Why this matters for the paper

The model-file knob is **not** "smaller quant = faster." Its throughput effect
depends on the operating regime (hardware, and the time-varying decode batch
size driven by concurrency). The one consistent Q4 advantage is **VRAM** (8B:
~3.2 GB less), which buys capacity (more slots / longer context). A
switching-cost-aware controller therefore cannot assume a fixed sign for the
quantization knob's latency effect; it must condition on the regime.
