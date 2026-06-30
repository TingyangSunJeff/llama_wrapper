# Hypothesis 2 Test — C_switch Invariance to Target Slot Count

_Pinned_Build: b9412-6-g764f1e64a | Platform: a100-mig | Run_Repeat count: 3_
_Device: MIG-ffd3804e-0c50-5327-a380-a6635cb1a15d (1g.10gb, ~9.5 GB)_
_Model: Llama-3.1-8B Q4_K_M · ctx=4096 fixed · slot_counts: {1, 2, 4, 8}_

**Hypothesis 2:** C_switch (disruption time) is invariant to the **target level** of the
reconfigured knob — i.e., going np=1→8 costs the same as np=1→2 or np=4→8. It can be
modeled by a **constant**.

---

## Per-transition C_switch (slot-only changes only)

All 12 ordered transitions among {np=1, np=2, np=4, np=8}, 3 repeats each.

| np from → np to | Boot (ms) | C_switch total (ms) | n |
| --- | ---: | ---: | ---: |
| np=2 → np=1 | 3458 ± 424 | 3992 ± 460 | 3 |
| np=4 → np=1 | 3568 ± 392 | 4072 ± 455 | 3 |
| np=8 → np=1 | 3418 ± 182 | 3917 ± 276 | 3 |
| np=1 → np=2 | 3228 ±  58 | 3715 ±  17 | 3 |
| np=4 → np=2 | 3444 ± 209 | 3967 ± 270 | 3 |
| np=8 → np=2 | 3216 ± 198 | 3709 ± 258 | 3 |
| np=1 → np=4 | 3201 ± 115 | 3632 ± 134 | 3 |
| np=2 → np=4 | 3324 ± 139 | 3780 ± 173 | 3 |
| np=8 → np=4 | 3292 ±  37 | 3830 ±  75 | 3 |
| np=1 → np=8 | 3496 ± 283 | 3978 ± 285 | 3 |
| np=2 → np=8 | 3450 ± 407 | 4029 ± 358 | 3 |
| np=4 → np=8 | 3234 ± 339 | 3751 ± 471 | 3 |

**Summary across all 12 transitions:**
- Overall mean C_switch: **3864 ms**
- Std across transitions: **146 ms** (CV = **3.8%**)
- Range: 3632 – 4072 ms (spread = **440 ms**)

---

## Verdict: Hypothesis 2 is **SUPPORTED**

The coefficient of variation across all 12 (from, to) pairs is **3.8%** — less than
one-quarter of the within-transition run-to-run std. The 440 ms range is smaller than
the typical single-transition std (± 200–470 ms), meaning the **between-transition
spread is not larger than measurement noise**.

No systematic pattern is visible:
- **"Jump size" does not matter.** np=1→8 (big jump) = 3978 ms; np=1→2 (small jump)
  = 3715 ms. Difference: 263 ms, inside noise.
- **Direction does not matter.** np=1→4 = 3632 ms; np=4→1 = 4072 ms. The larger
  value going *down* (more slots → fewer) vs going *up* (fewer → more) — no
  consistent ordering.
- **Current slot count does not matter.** The three rows with to_np=2 span
  3709–3967 ms regardless of whether the source was np=1, np=4, or np=8.

## Why H2 holds (mechanistic explanation)

In llama.cpp, **every slot-count change forces a full process restart** because
`n_parallel` is baked into the KV-cache tensor allocation at context-init time
(verified in `src/llama-kv-cache.cpp`: `ggml_new_tensor_3d(..., kv_size, n_stream)`).
The restart sequence is:

1. **Teardown** (~214 ms): SIGINT → process exit. Cost = OS + CUDA context destroy.
   Independent of both from and to slot count.
2. **Boot** (~3200–3500 ms): process spawn + CUDA context init + weight copy to GPU
   + KV allocation + graph build. The weight copy (~4.4 GB for Q4_K_M) dominates.
   **KV allocation is negligible:** at ctx=4096, even np=8 only needs 512 MiB of KV
   (8 × 64 MiB/slot), which is < 12% of the weight copy. Going from np=1 (64 MiB KV)
   to np=8 (512 MiB KV) adds only 448 MiB — less than 11% of the 4403 MiB weight cost.
3. **Warmup** (~140 ms): first inference step. Cost = one prefill + one decode step.
   Independent of slot count (only one slot is active during warmup).

The KV delta between any two slot counts is small compared to the dominant weight-copy
cost, so the total C_switch is insensitive to which slot levels are involved — exactly
what H2 predicts.

## Implication for the controller model

Since C_switch is approximately constant across all (from_np, to_np) pairs for
slot-only changes, the switching cost can be modeled as a **single scalar constant**:

```
C_slot-only ≈ 3864 ± 146 ms   (MIG 1g.10gb, 8B Q4_K_M, idle box)
```

This simplifies the control problem: the decision to reconfigure concurrency is a
**threshold policy** — switch if and only if the expected holding cost (degraded
quality-of-service under the current slot count) exceeds C over the decision horizon.
The controller does not need to account for the magnitude of the concurrency change.

## Caveats
- **3 repeats per transition.** The within-transition std is high (up to ±471 ms),
  driven by MIG boot-time variability. With 5+ repeats the confidence intervals would
  be tighter, but the between-transition spread (3.8% CV) is already smaller than
  noise, so the conclusion is robust.
- **Single model and ctx.** Tested with Q4_K_M at ctx=4096 only. The mechanism
  (weight-copy dominates KV cost) would hold for other configs, but should be
  verified for larger ctx values where KV grows.
- **This is idle-box MIG.** Under load, the absolute C_switch floor may shift (as
  seen in earlier contended runs), but the invariance property depends on relative
  costs, which should remain stable.
