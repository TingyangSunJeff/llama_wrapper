# llama.cpp as the Substrate for Our Branch 2 Paper

## 1. Why we are looking at llama.cpp

The edge-deployed LLM setting Prof. Arslan described, a drone or small autonomous platform with no guaranteed ground link, rules out any serving stack that assumes a Python runtime, a container orchestrator, or a cloud control plane. vLLM fits servers, not companion computers. llama.cpp fits the edge by construction:

- Single self-contained C++ binary, no Python in the hot path.
- Loads one `.gguf` model file (weights plus quantization plus metadata in one file).
- Runs entirely on-device, on CPU only, a single GPU, several GPUs in one box, or across machines on a LAN via an RPC backend.
- Offers an HTTP server with an OpenAI-compatible API purely as a local interface for other on-device programs; nothing leaves the device.

In other words, llama.cpp is to edge LLM serving what vLLM is to server-class serving. Same job (inference engine plus scheduler), different target environment.

## 2. How llama.cpp actually works

### 2.1 The restaurant analogy

- **Kitchen** = the loaded model. One kitchen per server process. The chef is defined by the choice of model file and quantization level; a bigger chef (Q8_0) needs more counter space but cooks better, a smaller chef (Q2_K) fits anywhere but loses quality.
- **Tables (slots)** = pre-allocated KV-cache regions, one per concurrent conversation. Decided at startup via `--parallel N`. Each table has a placemat of a fixed size (`--ctx-size`) holding that conversation's running context.
- **Host (slot dispatcher)** = when a request arrives, picks a table by prefix similarity first, then LRU.
- **Waiter (continuous batching)** = every few milliseconds, collects one token of work from every active table, packs them into a single GPU batch, dispatches, delivers the tokens back to each user. This is how one GPU appears to serve many users at once without actually running parallel models.
- **Idle sleep** = if no one has shown up for N seconds, send the chef home and free all memory. Next customer wakes everything back up.
- **Router mode** = with `--models-dir`, a concierge manages a pool of kitchens (child processes), each serving a different model file. Capped by `--models-max`, evicts by LRU.

### 2.2 GPU memory layout

```
┌──────────────────────────────────────────┐
│ Model weights  (shared, read-only)       │  fixed at load time
├──────────────────────────────────────────┤
│ KV cache slot 1  (user A)                │  pre-allocated
│ KV cache slot 2  (user B)                │  at startup;
│ KV cache slot 3  (user C)                │  not resizable
│ KV cache slot 4  (user D)                │  at runtime
├──────────────────────────────────────────┤
│ Temp scratch   (reused every forward)    │
├──────────────────────────────────────────┤
│ Small runtime overhead                   │
└──────────────────────────────────────────┘
```

All slots share the same weights. This is what makes multi-user serving economical: you do not pay `N × weights`, you pay `weights + N × KV_per_slot`.

**Concrete example — Llama-3-8B on a 16 GB Jetson:**

| Component | Formula | Size |
| --- | --- | --- |
| Weights (Q4_K_M) | ~4.5 bits/param × 8 B params | ~4.5 GB |
| KV per slot (2K ctx, FP16 KV) | `2 × n_layers × n_ctx × d_model × 2` | ~1 GB |
| Temp scratch + overhead | — | ~1 GB |
| **Leftover for KV** | 16 − 4.5 − 1 | ~10 GB |
| **Feasible KV shapes** | — | 4 slots × 2K, or 2 × 4K, or 1 × 8K |

The weights decision (which quantization) is the gating step. After that, the fight is over how to spend the leftover memory on concurrency versus context length. This is exactly the trade-off our proposal's memory formula describes.

### 2.3 What happens when the model is bigger than one GPU

| Option | What it does | Cost |
| --- | --- | --- |
| `--n-gpu-layers N` | Keep N layers on GPU, rest on CPU | Slower per token (CPU/GPU ping-pong), but fits bigger models |
| `--split-mode layer` | Pipeline layers across several GPUs | Memory scales, throughput does not |
| `--split-mode row` / `tensor` | Split each layer across GPUs in parallel | More comm, but better latency when GPUs are well connected |
| `--rpc host:port,...` | Same as above but across machines on a LAN | Petals-style but trusted, not open internet |

These are placement levers. In the current llama.cpp they are all decided at startup and not adapted online.

## 3. What llama.cpp decides for you today

Every policy currently inside llama.cpp is a fixed threshold, LRU, or round-robin. None of them are workload-aware, and none of them account for reconfiguration cost.

### Fast timescale (per request / per token)

| Decision | Current policy | Code lives in |
| --- | --- | --- |
| Which slot serves a new request | Highest prefix similarity above 10% (adjustable), else LRU | `get_available_slot` |
| Batching across active slots | Everyone gets served each tick, capped by batch size | `update_slots` |
| Host-RAM prompt cache eviction | Plain LRU | `server_prompt_cache` |
| Which slot to purge to free KV | "First non-processing slot" (authors flagged this as TODO) | `try_clear_idle_slots` |

### Slow timescale (across servers, across time)

| Decision | Current policy | Code lives in |
| --- | --- | --- |
| Sleep when idle | Fixed threshold `--sleep-idle-seconds` | `handle_sleeping_state` |
| Evict a model when `--models-max` reached | Plain LRU, ignores switching cost | `unload_lru` |
| Which child serves a request | Dictated by the model name in the request body; no choice |  |

### Multi-GPU placement

Decided once at startup via `--split-mode`, `--tensor-split`, `--n-gpu-layers`. Never re-examined.

## 4. How this maps onto Branch 2

Our Branch 2 claim is: **in the current paper the expensive decisions are offline (static server-chain composition); the natural sequel is to let the configuration itself adapt over time, accounting for reconfiguration cost.** llama.cpp is a near-ideal substrate for that thesis for three reasons:

1. **The baseline policies are honest and simple.** LRU, fixed thresholds, and greedy heuristics are the textbook strawmen for any learned policy. Nothing to hide behind.
2. **The observables the controller needs are already exposed.** `GET /metrics`, `GET /slots`, `GET /props` give real-time throughput, queue lengths, per-slot state, and sleep status. These are the natural feature vectors for either a bandit or an MBRL policy.
3. **The control actions are already there, or nearly there.** Model load and unload, sleep and wake, slot KV save and restore, and LoRA swap are all HTTP-accessible. The one endpoint that is advertised but not yet implemented, `POST /props`, is a small, well-scoped extension point for changing runtime knobs (slot count, context length, KV dtype) through a sleep-wake cycle.

### 4.1 The three control levers, by cost

Branch 2 is essentially the question "keep or reconfigure, and if reconfigure, which lever?" llama.cpp gives us a clean ladder:

| Lever | Cost | What changes | Already in llama.cpp? |
| --- | --- | --- | --- |
| **Reassign slots** | ~free (per request) | Which conversation gets which table | Yes (heuristic) |
| **Reshape slots** (change `n_parallel`, `n_ctx`) | seconds (sleep/wake) | Memory split between concurrency and context length | Partially: sleep/wake exists, but always wakes with the same config |
| **Swap model / quantization** | tens of seconds | Which chef is in the kitchen | Yes in router mode, but eviction is plain LRU |

### 4.2 Why this is a genuine research gap

The llama.cpp code itself shows exactly where intelligence is missing:

- The slot-purge heuristic carries a TODO comment asking for a smarter policy.
- The LRU model eviction ignores that a big Q8 model costs 10× more to reload than a small Q2 one — a cost-unaware policy on a memory-pressured edge device.
- The sleep-wake mechanism always wakes with the same startup configuration, even if recent workload shape has clearly shifted.
- The multi-GPU placement is static, even though workload mix may change during a mission.

Every one of these is a concrete instance of the "reconfigure online with switching cost" question our proposal framed in the abstract.

### 4.3 How our proposed machinery fits

The three ideas from §4 of our proposal map cleanly onto this substrate:

- **Model-based RL.** Switching costs in llama.cpp are measurable offline, once, per platform — load time is dominated by disk throughput and model size, and table reshape cost is dominated by one sleep-wake cycle. Plug these as a known model of action costs; the learner only needs to estimate benefits (future workload, expected quality).
- **Hierarchical control.** Fast layer (learned slot dispatcher, above `get_available_slot`), medium layer (reshape decisions, through a smarter sleep-wake), slow layer (model/quant selection, above `unload_lru`). The timescale separation already mirrors the cost separation in the llama.cpp code.
- **Contextual bandits.** The router-level "which model to hold hot" decision is a natural arms-with-features problem: arms are model files, features are recent-workload statistics, reward is a latency and quality composite minus switching cost.

## 5. Where our code actually goes

Two options, neither of which requires us to become llama.cpp internals maintainers:

- **External controller (default).** A small Python process next to the server. Polls `/metrics` and `/slots` every few seconds. Decides "keep or reconfigure." If reconfiguring, calls the existing HTTP endpoints (`/models/load`, `/models/unload`, slot save/restore) or restarts the server with new flags. This is the exact pattern Llumnix uses above vLLM; we do the same above llama.cpp. All the research code is in Python.
- **Small C++ patch on top of that.** Fill in the stubbed `POST /props` handler so the controller can change runtime knobs (slot count, per-slot context, KV dtype) through a sleep-wake cycle without restarting the process. This is the only change we would need inside llama.cpp, and it is bounded and well scoped.

## 6. Baselines that come for free

We do not need to build strawmen — llama.cpp ships them:

| Baseline | What it is |
| --- | --- |
| Static config | Default, set once at startup |
| Sleep-on-idle | `--sleep-idle-seconds` as-is |
| LRU router eviction | `--models-dir --models-max N` |
| Prefix-similarity slot dispatch | Current `get_available_slot` |
| Llumnix-style adaptation (request-level only) | Our controller restricted to slot-level decisions |
| **Our policy** | Hierarchical, switching-cost-aware, across all three levers |

## 7. Minimum first experiment to propose

A clean first result, runnable on one Jetson-class box:

1. **Setup.** One llama.cpp server, `--models-dir` pointing at three quantizations of the same model (Q2_K, Q4_K_M, Q8_0), `--models-max 1` to force switching. Monitoring on.
2. **Workload.** Synthetic two-phase trace: interleaved short snappy requests and occasional long-context requests, with the mix shifting over a fixed schedule, to simulate changing mission phases.
3. **Baselines.** Static Q4_K_M (no adaptation), plus llama.cpp's built-in LRU router switching on-demand.
4. **Our policy.** A small contextual-bandit controller choosing the active quantization based on recent queue and prompt-length statistics, with a known switching cost.
5. **Metrics.** Per-phase tail latency, average quality (surrogate: perplexity or a small eval set), count and duration of reconfiguration events.

If this shows a clear gap over LRU, we have the first figure of the paper. It also lets us add more levers incrementally (slot reshape, multi-model) without rewriting the controller.

## 8. One-paragraph pitch for the group

> Llama.cpp is the edge-equivalent of vLLM: a single self-contained inference engine that owns one or more local GPUs, serves concurrent users through pre-allocated KV slots, and optionally manages multiple models via a lightweight in-process router. Every runtime policy it ships with is memoryless and cost-unaware: LRU eviction, fixed-threshold sleep, first-free slot purging. On an edge platform with bursty and shifting workloads, these baselines leave measurable performance on the table. Our next paper extends our offline server-chain composition to online configuration-level adaptation, using llama.cpp as the substrate: the observables we need are already exposed, the expensive control actions are already HTTP-accessible, and the missing piece is precisely the switching-cost-aware, hierarchical controller our current proposal already sketches in §4.
