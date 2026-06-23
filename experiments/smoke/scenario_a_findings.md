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
- **Refinement (2026-06-22, repeated 3-way sweep):** the mid-batch slowdown is
  **specific to `Q4_K_M` (the k-quant)** — its MMQ decode kernel stalls in the
  batch ~4-16 window. Simple **`Q4_0` does not have this problem and beats both
  Q8_0 and Q4_K_M on throughput at almost every batch size.** So "4-bit loses
  when compute-bound" is wrong as stated; it is "the *k-quant* kernel is
  inefficient there." See the repeated sweep below.

## Measurements (A100, Meta-Llama-3.1-8B-Instruct, ngl=99)

> Quick legend: **decode** = generating output tokens (the part we care about);
> **prefill** = reading the prompt. **batch** = how many users generate at once.
> **tok/s** = tokens per second (higher is better). `ngl=99` = whole model on the
> GPU.

### Batched decode throughput — REPEATED 3-way sweep (2026-06-22)

Reproducible via `scenario_a_batchsweep.py` (npp=16, ntg=128, ctx=8192, ngl=99,
GPU 0, **5 runs + 1 discarded warmup**, raw logs in `results/`). Cells are decode
`speed_tg` in tok/s, **mean ± std** over the 5 runs.

| batch | Q8_0 | Q4_K_M | Q4_0 | winner (mean) |
|---:|---:|---:|---:|:--|
| 1   | 135 ± 1.5  | 154 ± 0.2  | 184 ± 2.5  | Q4_0 (+19%) |
| 2   | 242 ± 4.5  | 247 ± 0.3  | 325 ± 7.7  | Q4_0 (+32%) |
| 4   | 352 ± 9.2  | 316 ± 0.3  | 426 ± 12.1 | Q4_0 (+21%) |
| 8   | 501 ± 17.2 | 370 ± 0.2  | 599 ± 25.0 | Q4_0 (+20%) |
| 16  | 991 ± 77.9 | 1017 ± 0.8 | 1075 ± 89.5| Q4_0 (+6%)  |
| 32  | 1622 ± 51.5| 1664 ± 1.9 | 1726 ± 63.5| Q4_0 (+4%)  |
| 64  | 2427 ± 69.8| 2454 ± 2.2 | 2578 ± 50.7| Q4_0 (+5%)  |
| 128 | 3071 ± 51.4| 2907 ± 0.8 | 3069 ± 61.5| Q8_0 (+0%)  |

What the repeats establish:

- **The original Q8 > Q4_K_M mid-batch result reproduces and is NOT noise.** At
  batch 8, Q8_0 501 vs Q4_K_M 370 (+35%); Q4_K_M's std is **< 2 tok/s** across
  all batches, so its mid-batch deficit is rock-solid. Q4_K_M visibly **stalls**
  from batch 4→8 (316 → 370, barely moves) while Q8_0 (352 → 501) and Q4_0
  (426 → 599) scale normally — the K-quant MMQ kernel is the bottleneck there.
- **Q4_0 beats both Q8_0 and Q4_K_M at every batch except 128 (tie).** So the
  slowdown is a property of the **k-quant kernel, not of 4-bit weights**. A
  simple-format 4-bit (Q4_0) is the throughput winner across the whole sweep.
- **Where the noise lives:** batch 16 is the noisy point (Q8_0 ± 78, Q4_0 ± 90)
  — one of the 5 repeats ran at higher GPU clocks (its batch-8/16/32 were all
  elevated). Everywhere else variance is small. The mid-batch Q8>Q4_K_M gap is
  far larger than its std, so it is robust; the batch ≥ 16 orderings are within
  noise and should not be over-read.

## Why this happens (in plain language)

**The one idea that explains all of it: there are two possible bottlenecks.**
Each generate step can be limited by either
1. **moving data** — reading the model's weights out of GPU memory, or
2. **doing math** — the multiply-adds themselves.

Which one is the bottleneck decides whether "smaller model = faster."

**One user (batch 1) → limited by moving data (memory bandwidth).** To produce a single token the GPU
must read the *entire* model's weights but only does a little math with them. The
slow part is hauling the weights out of memory, so smaller weights = less to haul
= faster. That's why the 4-bit formats win at batch 1.
*Analogy:* carrying a giant cookbook to the kitchen to cook one small dish — the
carrying dominates, so a lighter cookbook helps.

**Many users at once (high batch) → limited by doing math.** When 8 users generate
together, the GPU reads each weight **once** and reuses it for all 8. The
data-hauling cost is now shared 8 ways, so it stops being the bottleneck — the
*math* becomes the limit instead.
*Analogy:* carry the cookbook once, then cook 16 dishes from it — the carrying no
longer matters; the cooking does.

**Why Q4_K_M then slows down, but Q4_0 doesn't.** The GPU can't multiply 4-bit
numbers directly — it first has to **unpack** them into a workable form, and how
hard that is depends on the format:
- Q8_0 and Q4_0 use *simple* packing → quick to unpack.
- Q4_K_M uses *fancy* packing (better quality, a bit smaller) → more work to
  unpack.

When the math/unpacking is the bottleneck (the middle batch range, ~4-16 users),
llama.cpp's code path for Q4_K_M is simply less efficient, so Q4_K_M slows down
there while Q8_0 and Q4_0 keep scaling. At very large batches everything falls
back to one common high-performance path and they even out again.

**That's the whole story:** smaller-is-faster only holds when you're limited by
*moving data* (one user / CPU / edge). When you're limited by *math* (many users
on a fast GPU), the **format's unpacking efficiency** decides it — and Q4_K_M's
happens to be weak in the middle range, while plain Q4_0's is fine.

<details>
<summary>Under the hood (exact source — skippable, for the curious)</summary>

Verified in `ggml/src/ggml-cuda/`. `ggml_cuda_should_use_mmq()` switches code
paths at `ne11 < MMQ_DP4A_MAX_BATCH_SIZE` (= 64), where `ne11` is the decode batch
size. Three matrix-multiply paths for quantized weights:
1. **batch 1:** a "matrix × vector" path (`mul_mat_vec_q`) — limited by memory
   bandwidth, so fewer weight bytes wins.
2. **batch 2-63:** the "MMQ" path. Note: MMQ does **not** avoid dequantization —
   it decodes each weight tile **on-chip** into integers and does int8
   dot-products (via `dp4a` / int8 tensor-core MMA), instead of writing a full
   fp16 copy of the matrix to memory (which is what path 3 does). So a format that
   is expensive to decode (Q4_K's hierarchical super-blocks) still pays for it
   here; Q8_0 (already int8) and Q4_0 (simple nibble) decode nearly for free.
3. **batch ≥ 64:** dequantize to fp16 in memory and use the standard cuBLAS
   matrix-multiply (same path for every format) → throughputs converge.

kernel — a GPU "kernel" is just a small program that runs on the GPU to do one specific job. Here it means the actual function that performs one matrix-multiply. The GPU has several different functions available for "multiply these weights," and it picks one depending on the situation. Think of a kernel as one specific recipe the GPU runs. (Nothing to do with an operating-system kernel — same word, unrelated.)

MMQ — short for "Mul-Mat-Quantized" = matrix-multiply using the compressed weights directly. It's llama.cpp's hand-written GPU kernel that multiplies with the 4-bit/8-bit weights more or less as-packed (using integer math), without first expanding them to full size. Crucially, there's a separate MMQ kernel hand-written for each format (one for Q8_0, one for Q4_0, one for Q4_K_M), and they're not equally well-tuned.

cuBLAS — NVIDIA's official, professionally-optimized matrix-math library (BLAS = "Basic Linear Algebra Subprograms"; cu = CUDA/NVIDIA). It's the standard, very fast multiply that works on full-size (fp16) numbers. llama.cpp uses it at large batch: it first unpacks the 4-bit weights into fp16, then hands the multiply to cuBLAS. Because cuBLAS is the same no matter what the original format was, all formats converge at large batch.

**Why the 2-63 range is non-monotonic and format-dependent** (mechanism class,
not yet profiled on this build):
- *Amortization of decode cost:* The per-tile decode cost is fixed; the useful math grows with batch width. At small batch the decode cost is spread over few columns and dominates, so cheap-decode formats (Q8_0, Q4_0) win big. As batch climbs, Q4_K_M's heavy decode gets amortized over more columns and it catches back up — which is exactly why the gap shrinks from +35% at batch 8 to a near-tie by batch 16–32.
- *Tile / "wave" quantization:* kernels compute fixed-width tiles, so a batch that
  doesn't fill the tile wastes part of it — efficiency rises in steps, not
  smoothly. llama.cpp picks **different tile shapes per format**
  (`mmq_get_granularity_*`), so each format's staircase is offset and the curves
  cross.
- *Tensor-core utilization:* the int8 MMA units have fixed shapes that need the
  batch wide enough to fill; formats can cross that efficiency threshold at
  different batch sizes.

So the real driver is *(how cheaply this format decodes) ÷ (how well this batch
fills the kernel's tile/MMA shapes)* — both format-specific. A precise
attribution of the batch-4-8 Q4_K dip would need a profiler (Nsight Compute) or a
batch-by-1 sweep to see the staircase.
</details>
</details>

**Why batch ≥ 64 perform different**

The flow at batch ≥ 64 is: read the compressed weights → dequantize them into a full fp16 copy in VRAM → hand that fp16 matrix to cuBLAS to multiply.

Walk through where a format could differ:

Dequant reads the compressed weights — Q4_0 reads ~half the bytes of Q8_0. This is the only place Q4_0 is smaller. Small advantage.
Dequant writes an fp16 copy — same size for every format (fp16). No advantage.
cuBLAS reads the fp16 matrix — same size for every format. No advantage.
cuBLAS does the multiply — identical fp16 GEMM regardless of original format. No advantage. And this step dominates the time at batch 128.

So once you expand to fp16, the weight is the same size for everyone (steps 2–4), and the multiply — the expensive part — is byte-for-byte identical across formats. Q4_0's only remaining edge is step 1, the one-time read of the compressed weights.


## Why this matters for the paper

The model-file knob is **not** "smaller quant = faster," and it is **not even
just Q4-vs-Q8** — it is a 3-way (≥) trade-off across **quality, throughput, and
VRAM**:

- **Throughput:** regime-dependent and *format*-dependent. Q4_0 is fastest across
  the batch sweep here; the Q8>Q4 mid-batch flip is specific to the **Q4_K_M
  k-quant kernel**. So a controller cannot assume a fixed sign for the quant
  knob's latency effect — it must condition on hardware, decode batch size, *and
  which quant format*.
- **Quality:** runs the other way — Q4_K_M > Q4_0 at equal bits. So the fastest
  format is not the best-quality one.
- **VRAM:** the one consistent 4-bit win (8B: ~3.2 GB less than Q8), buying
  capacity (more slots / longer context).

A switching-cost-aware controller therefore picks a point on a quality ×
throughput × VRAM surface that is regime-dependent — not a single "faster" knob.
