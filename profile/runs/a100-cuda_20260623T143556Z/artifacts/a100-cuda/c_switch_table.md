# Reconfiguration Cost (C_switch)

_C_switch decomposition by change type. Pinned_Build: b9412-6-g764f1e64a | Platform: a100-cuda | Run_Repeat count: 3_

| Change type | Teardown (ms) | Boot (ms) | Warmup (ms) | C_switch total (ms) |
| --- | --- | --- | --- | --- |
| slot-reshape | 259.97 ± 33.55 | 3062.41 ± 940.40 | 172.14 ± 182.03 | 3494.52 ± 964.10 |
| model-reload | 264.11 ± 30.24 | 2990.89 ± 936.51 | 191.62 ± 216.30 | 3446.63 ± 1122.09 |
| combined | 268.42 ± 25.79 | 3100.06 ± 973.53 | 151.12 ± 217.29 | 3519.61 ± 1181.49 |

> Model: gemma-3-1b-it (Q4_K_M & Q8_0) · GPU: NVIDIA A100 80GB PCIe · 3 repeats per transition.

---

## What is being measured

This table answers one question: **when you reconfigure a running llama.cpp server,
how long is it unavailable, and where does that time go?**

"Reconfiguring" means shutting down the current server and starting a new one with
different settings. The suite times that whole gap on a **single monotonic clock**
(a clock that only moves forward, immune to system time adjustments), so the
component phases always add up to the total.

## The four timestamps

For each reconfiguration the harness records four moments:

- **t0** — we send the shutdown signal (SIGINT) to the current server
- **t1** — the old server process has fully exited
- **t2** — the new server responds "ready" on its `/health` endpoint
- **t3** — the new server returns its first generated token to a request

## The columns (each is the gap between two of those moments)

- **Teardown (ms)** = `t1 - t0`. Time to cleanly shut down the old server (release
  the GPU, exit the process). ~260 ms here — fast and stable.
- **Boot (ms)** = `t2 - t1`. Time for the new server to start, load the model
  weights onto the GPU, allocate the KV cache, and report healthy. ~3000 ms — this
  is the bulk of the cost.
- **Warmup (ms)** = `t3 - t2`. After it reports "healthy," the time to actually
  serve the first token (the first request is always a bit slower). ~150–190 ms.
- **C_switch total (ms)** = `t3 - t0`. The full unavailability window — the sum of
  the three phases above. ~3500 ms. This is the headline number: roughly 3.5 seconds
  of downtime per reconfiguration.

By design the three phases reconcile to the total within 50 ms, so the breakdown is
trustworthy, not estimated.

## The rows — three kinds of reconfiguration

A "config" here is the triple {model file, context length, KV-slot count}. The row
label is decided purely by which fields change between the old and new config:

- **slot-reshape** — only the context length and/or slot count change; same model
  file (e.g. 1 slot → 4 slots, same model).
- **model-reload** — only the model file changes (e.g. Q4 → Q8 quantization, same
  shape).
- **combined** — the model file changes *and* the shape changes (both at once).

## How a single number becomes "mean ± std"

Each reconfiguration is run multiple times, and a single sample is never trusted:

- Before timing, one **warmup run is discarded** (to avoid cold-start artifacts like
  disk caching).
- It is then repeated until **3 successful runs** are collected (the production
  default is 5). Any run that times out or errors is excluded, not averaged in.
- The suite pools all successful runs for each change type and reports the **mean**
  (average) **± standard deviation** (run-to-run spread).

In this demo each row pools 4 distinct transitions × 3 repeats, so roughly a dozen
real measurements feed each line. The large std on Boot (±940 ms) reflects genuine
variation in process/model-load startup time — honest signal, not noise to hide.

## Takeaway

> Reconfiguring this model costs ~3.5 s of downtime, and ~87% of that is the new
> server booting (process start + weight load + KV allocation). Teardown and
> first-token warmup are cheap. For a 1B model the *type* of change barely matters
> because boot is dominated by fixed startup cost rather than weight reloading.

**Caveats:** these are small-model (1B) numbers at 3 repeats, so this is a methodology
demo rather than production figures. On the 8B model, model-reload and combined are
expected to separate from slot-reshape as weight loading grows.
