# llama.cpp as the Substrate for Our Branch 2 Paper

## 1. Paper Direction

Our next-paper direction is **Branch 2: online configuration-level adaptation for LLM serving under switching cost**.

The current paper studies expensive serving decisions mostly as **offline/static composition** decisions. The natural sequel is to allow the serving configuration itself to adapt over time under time-varying workloads. The key question becomes:

> **When should the system keep its current serving configuration, and when is it worth paying the cost to reconfigure?**

We propose to study this question using **llama.cpp** as the substrate because llama.cpp exposes a clean set of edge-relevant configuration knobs: model/quantization file, context length, number of parallel slots, KV-cache allocation, sleep/wake behavior, and model load/unload behavior. llama.cpp is a widely used C/C++ inference engine for local and edge LLM inference, supporting GGUF model files and many quantized formats. [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)

---

## 2. Why We Are Looking at llama.cpp

The edge-deployed LLM setting — e.g., a drone, robot, or small autonomous platform with no guaranteed ground link — rules out serving stacks that assume a Python runtime, container orchestration, or cloud control plane.

vLLM fits server-class deployments. llama.cpp fits edge and local deployment by construction:

- Single self-contained C/C++ inference engine.
- Loads GGUF model files, where model weights and quantization metadata are packaged for local inference.
- Runs on CPU, local GPU, multiple GPUs in one box, and in some configurations across machines through llama.cpp-supported backends.
- Supports practical local deployment of quantized open-weight models, which is especially relevant for memory-constrained hardware.

In short:

> **llama.cpp is to edge/local LLM serving what vLLM is to server-class LLM serving.**

vLLM’s PagedAttention paper targets high-throughput server-side serving by reducing KV-cache fragmentation and improving batching efficiency. llama.cpp targets a different deployment regime: local, lightweight, memory-constrained inference.

**Key sources:**

- llama.cpp official repository: <https://github.com/ggml-org/llama.cpp>
- vLLM / PagedAttention: <https://arxiv.org/abs/2309.06180>

---

## 3. How llama.cpp Works Conceptually

### 3.1 Restaurant Analogy for one server

- **Kitchen** = the loaded model.  
  One model is loaded into a server process. The “chef” is defined by the chosen model file and quantization level. A higher-quality quantization such as Q8_0 consumes more memory, while a lower-bit quantization such as Q2_K or Q3_K consumes less memory but may lose quality. GGUF quantization formats in llama.cpp include legacy formats such as Q4_0/Q5_0/Q8_0 and K-quant formats such as Q3_K, Q4_K, Q5_K, and Q6_K.

- **Tables / slots** = pre-allocated KV-cache regions.  
  Each concurrent conversation/session needs KV-cache memory. In llama.cpp-style serving, this creates a fixed memory trade-off between the number of sessions and the context length.

- **Host / slot dispatcher** = request-to-slot assignment.  
  Current policies are simple heuristics such as prefix matching and LRU-style fallback, based on the implementation details we have inspected.

- **Waiter / continuous batching** = batching active slots together.  
  Like other LLM serving engines, batching is critical for throughput. vLLM formalized the importance of efficient KV memory management for high-throughput serving with PagedAttention.

- **Idle sleep** = free resources when idle.  
  llama.cpp has sleep/wake behavior that can free memory after idle periods, but the current behavior is threshold-based rather than workload-aware.

- **Router mode** = multiple model files managed by a lightweight server/router.  
  This creates a natural baseline for model/quantization switching, but default policies such as LRU eviction are not switching-cost-aware.

---

## 4. Memory Model and Configuration Trade-off

The key memory structure is:

```text
┌──────────────────────────────────────────┐
│ Model weights  (shared, read-only)       │  fixed at load time
├──────────────────────────────────────────┤
│ KV cache slot 1  (user A)                │  pre-allocated
│ KV cache slot 2  (user B)                │  at startup;
│ KV cache slot 3  (user C)                │  not easily resized
│ KV cache slot 4  (user D)                │  at runtime
├──────────────────────────────────────────┤
│ Temp scratch   (reused every forward)    │
├──────────────────────────────────────────┤
│ Small runtime overhead                   │
└──────────────────────────────────────────┘
```

All slots share the same model weights. Thus multi-user serving does not cost:

```text
N × model weights
```

but instead:

```text
model weights + N × KV_per_slot + temporary memory + runtime overhead
```

A standard decoder-only transformer KV-cache estimate is:

```text
M_KV_per_session = 2 × n_layers × n_ctx × d_model × b
```

where:

- `2` accounts for key and value tensors,
- `n_layers` is the number of transformer layers,
- `n_ctx` is the context length,
- `d_model` is the hidden dimension,
- `b` is bytes per KV element.

Note: this formula assumes one KV head per query head. Modern Llama-3 / Qwen2
models use **grouped-query attention (GQA)**, so the KV width is
`n_kv_heads × head_dim`, not the full `d_model`. For Llama-3-8B (32 query heads,
8 KV heads) the KV cache is ~**4× smaller** than the naive `d_model` estimate;
use `n_kv_heads × head_dim` in place of `d_model` for any concrete sizing.

This same memory structure appears in our proposal document: total memory is approximately model memory plus per-session KV memory plus temporary memory.

### Example: Llama-3-8B on a 16 GB Jetson-class Device

| Component | Formula / Intuition | Approximate Size |
|---|---|---:|
| Weights, Q4_K_M | about 4–5 bits per parameter for an 8B model | about 4–5 GB |
| KV per slot, 2K ctx, FP16 KV | `2 × n_layers × n_ctx × d_model × 2` | around 1 GB |
| Temp scratch + runtime overhead | backend-dependent | around 1 GB |
| Leftover for KV | `16 - weights - overhead` | around 10 GB |

The exact values depend on model architecture, backend, KV dtype, and llama.cpp build, but the structural trade-off is the important point: after choosing quantization/model size, the remaining memory must be divided between **context length** and **concurrency**.

This trade-off is also why llama.cpp quantization studies are directly relevant. The paper *Which Quantization Should I Use?* evaluates llama.cpp GGUF quantization schemes on Llama-3.1-8B-Instruct and reports model quality, perplexity, compression, quantization time, and CPU throughput across formats.

**Key source:**

- Uygar Kurt, *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct*, arXiv:2601.14277: <https://arxiv.org/pdf/2601.14277>

---

## 5. System Model for Our Paper

### 5.1 Platform

We should define the primary setting as:

> **One edge device or a small number of edge devices, each running an independent llama.cpp instance.**

The cleanest first version is **single-device**. A small multi-device extension is possible, but we should avoid claiming a full cloud-cluster orchestration problem, because that space already has strong related work such as QLM, Chiron, MuxServe, DistServe, and Splitwise.

Recommended wording:

> We focus on configuration adaptation for llama.cpp-style edge inference instances. The primary setting is a single edge server; a secondary extension considers a small number of independent edge servers without assuming elastic cloud provisioning, container orchestration, or high-bandwidth cluster control.

This lets us discuss multiple servers without colliding too strongly with distributed LLM serving papers.

### 5.2 Workload

The workload is a time-varying request stream with:

- arrival rate `λ(t)`,
- prompt length distribution,
- output length distribution,
- short interactive requests,
- occasional long-context requests,
- phase shifts over time.

This matches the workload motivation in many recent LLM serving papers: real workloads are heterogeneous, bursty, and hard to predict. Llumnix emphasizes request heterogeneity and unpredictability in LLM serving. MorphServe also motivates dynamic adaptation using bursty workloads and changing memory pressure.

### 5.3 Control Knobs

For the first paper, we should lock three main knobs.

#### Knob 1: Model / Quantization File

Examples:

```text
Q2_K, Q3_K, Q4_K_M, Q5_K_M, Q8_0
```

Effect:

- lower-bit quantization reduces model memory footprint,
- higher-bit quantization improves quality,
- switching model files incurs load/reload cost.

The llama.cpp quantization evaluation paper is useful here because it gives empirical evidence that different GGUF quantization formats trade off model size, quality, and throughput.

#### Knob 2: Context Length

Example:

```text
--ctx-size 2048
--ctx-size 4096
--ctx-size 8192
```

Effect:

- larger context improves ability to handle long prompts/sessions,
- larger context consumes more KV memory per slot,
- larger context reduces feasible concurrency.

KV-cache memory is a known bottleneck in LLM serving; vLLM/PagedAttention and KV-cache scheduling papers both emphasize KV cache as a core serving constraint.

#### Knob 3: Concurrency / Number of Slots

Example:

```text
--parallel 1
--parallel 2
--parallel 4
```

Effect:

- more slots reduce queueing delay under concurrent workloads,
- more slots consume more KV memory,
- more slots reduce feasible context length under fixed memory.

### **Performance Scenarios: Coarse-Grained Configuration Control**

In `llama.cpp` runtimes, total memory is typically split between **Model Weights** and the **KV Cache**. Here are three practical cases where reconfiguring these "coarse" knobs improves performance:

#### **Scenario A: Switching GGUF Files (VRAM headroom / capacity)**
*   **The Knob:** Switching between a higher-bit `Q8_0` and a lower-bit `Q4_K_M` GGUF file.
*   **What actually changes (measured — see `experiments/smoke/scenario_a_findings.md`):** the *consistent* effect is **VRAM**: `Q4_K_M` frees ~3.2 GB on Llama-3.1-8B, which buys more KV slots or longer context. The *throughput* effect is **regime-dependent and can flip sign** — `Q4_K_M` is faster only when bandwidth-bound (batch≈1, or CPU); under **batched GPU decode (the burst case) `Q8_0` was actually faster** in our runs (driven by llama.cpp's batch-size-dependent CUDA kernel selection, not a simple "dequant overhead").
*   **The Result:** the controller pays a **multi-second reload cost** to change the quant file mainly to **reshape the memory budget** (free VRAM for capacity), not as a guaranteed per-token speedup. A switching-cost-aware controller must condition the quant knob's benefit on the operating regime rather than assume "smaller = faster."

#### **Scenario B: Reshaping Context Length (Document Unlock)**
*   **The Knob:** Modifying the `-c` (context length) parameter.
*   **Performance Win:** A robot is initially in "chat mode" with a short **2K context**. Suddenly, it needs to analyze a **32K-token technical manual**.
*   **The Result:** The original 2K configuration would either **OOM (Out of Memory) or truncate the data**. The controller reconfigures the instance to a **32K context**. While this might require switching to a lower weight quantization to fit the larger KV cache, it "unlocks" the ability to perform the long-context task that was previously impossible.

#### **Scenario C: Modifying Parallel Slots (Anti-Blocking)**
*   **The Knob:** Changing the `-np` (number of parallel slots) parameter.
*   **Performance Win:** The system is currently running a single long-context "Batch" job with `-np 1`. A new interactive user arrives and is **Head-of-Line (HOL) blocked** for minutes.
*   **The Result:** The controller reconfigures the memory shape from `{np=1, ctx=32K}` to `{np=4, ctx=8K}`. By dividing the KV memory into **parallel slots**, the system can now use **Continuous Batching** to serve the interactive user and the batch job simultaneously, dropping interactive response time from minutes to milliseconds.
---

## 6. Decision Problem

At each decision epoch, the controller observes recent system state and chooses whether to keep or reconfigure.

### State

Possible state features:

```text
s_t = {
    recent arrival rate,
    queue length,
    average prompt length,
    prompt length variance,
    active slot usage,
    recent latency / TTFT / TPOT,
    current quantization,
    current context length,
    current number of slots,
    memory pressure,
    time since last reconfiguration
}
```

### Action

The action is either:

```text
keep current configuration
```

or choose a new configuration:

```text
(model/quantization, context length, number of slots)
```

For example:

```text
(Q4_K_M, 4096 ctx, 2 slots)
```

or:

```text
(Q8_0, 2048 ctx, 1 slot)
```

or:

```text
(Q3_K_M, 2048 ctx, 4 slots)
```

### Switching Cost

The switching cost depends on the action:

```text
C_switch(a_t) =
    0                                if keep current config
    C_slot_reshape                   if only n_ctx / n_parallel changes
    C_model_reload                   if model / quantization changes
    C_model_reload + C_slot_reshape  if both change
```

This is the key Branch 2 idea:

> **Reconfiguration is not free. The controller must learn when the future benefit justifies the immediate switching cost.**

This differentiates us from request-level scheduling systems such as Llumnix, which dynamically reschedules requests across instances but does not primarily solve model-file/context/concurrency reconfiguration. It also differentiates us from online KV-cache scheduling theory, which optimizes request scheduling under KV constraints rather than changing the engine configuration itself.

---

## 7. Objective

A possible reward/objective is:

```text
maximize expected sum over time of:
quality reward
- latency penalty
- memory pressure penalty
- switching cost
```

For example:

```text
R_t = Q(config, workload)
      - α × latency
      - β × SLO_violation
      - γ × memory_pressure
      - C_switch(action)
```

The objective can be framed as:

```text
max E[ Σ_t (R_t - C_switch(a_t)) ]
```

This naturally supports:

- model-based RL,
- contextual bandits,
- hierarchical control,
- threshold baselines,
- oracle/lookahead comparisons.

---

## 8. Why This Is a Branch 2 Problem

Branch 2 is about moving from:

```text
static/offline configuration
```

to:

```text
online configuration adaptation under switching cost
```

llama.cpp provides a clean substrate because:

1. **The baseline policies are simple.**  
   Existing behavior includes fixed thresholds, LRU-style eviction, static startup configuration, and simple request/slot heuristics.

2. **The observables are practical.**  
   A controller can monitor latency, queue length, active slots, memory pressure, and request length statistics.

3. **The actions map naturally to real knobs.**  
   Quantization/model file, context length, concurrency, sleep/wake behavior, and model load/unload correspond directly to llama.cpp runtime choices.

4. **The switching costs are measurable.**  
   Model load/reload time, sleep/wake time, and restart/reconfiguration time can be profiled offline per platform.

This gives a concrete systems version of the abstract problem from the proposal document: online adaptation of serving configuration under uncertainty and reconfiguration overhead.

---

# 9. Related Work and Novelty Check

> **Verified pass (2026-06).** Section 9★ below is the authoritative related-work
> writeup, based on full reads of the primary sources (not search snippets). The
> earlier snippet-based audit (old §9.1–9.10) and the duplicate novelty-claim
> (old §10) have been **removed** — their content is consolidated here and in
> §9★.5; unique neighbors are preserved as bullets in §9★.3. Section numbering
> jumps from §9★ to §11 for this reason. Full notes and citations:
> `litcheck_2026-06.md`.

# 9★. Related Work (verified, 2026-06)

## 9★.1 Positioning in one paragraph

Prior work on adaptive LLM serving clusters into four groups: (i) **runtime
morphing inside the engine** (MorphServe — quantized layer swapping + KV
resizing); (ii) **online parallelism reconfiguration** on GPU clusters (Flying
Serving DP↔TP, ParaDySe parallel-strategy switching, LoongServe/Seesaw/Shift-
Parallelism); (iii) **per-request mode/model routing** (ModeSwitch-LLM, HELIOS,
QLM, RouteLLM/OmniRouter); and (iv) **offline config search / cold-start
optimization** (AIConfigurator, ServerlessLLM and the serverless cold-start
line). **The single closest is LaTune (WWW '26): adaptive runtime-config tuning
for llama.cpp on edge devices** — but it tunes *cheap runtime params* and excludes
quant/context. None of them studies **switch-cost-aware *online* control of
llama.cpp edge memory-shape knobs — {GGUF/quant file, context length, KV slots} —
under an *irreducible* reconfiguration cost.** That gap, plus the regime-dependence
measurement, is our target.

## 9★.2 The threats to address head-on

**(A0) LaTune (WWW '26, Peking Univ.) — CLOSEST PAPER; read first.** Claims to be
the *first to systematically study configuration tuning for LLM inference engines
on edge devices*; backend is **llama.cpp**, evaluated on RTX 4090/3060, Apple M4,
and **Jetson Orin Nano**. Online adaptation to a time-varying resource budget
(max T(P) s.t. U(P) ≤ R_t) via parameter selection (Shapley) + knowledge transfer
+ two-stage optimization (offline MOBO Pareto front, online resource-aware
selection). *This pre-empts the generic "adaptive config tuning for edge
llama.cpp" framing — do not claim it.* **Two facts save our niche:** (1) it
**explicitly excludes** model-compression/quantization precision and fixes context
length, tuning only *runtime/system* params (threads, parallel, mem_pool,
gpu-layers, kv-offload, flash-attn, ubatch) — it even encodes quant/ctx/bpw as
*fixed task descriptors*, so our {quant file, context, slots} are exactly what it
leaves out; (2) its online step is **free selection** from a precomputed Pareto
set (~0.12 s) under a **rank-stability** assumption — it never models the cost of
*applying* a reconfiguration. Our wedge: those excluded knobs are *structurally
reload-inducing* on llama.cpp (startup-only `-c`/`-np`, GGUF swap), so the
controller must trade off an **irreducible switching cost**. Candid risk: a
reviewer could see us as "LaTune + quant/ctx + a cost term"; the defense is the
fundamentally different cost structure of memory-shape knobs. **Consider narrowing
the thesis to switch-cost-aware memory-shape reconfiguration and discussing scope
with advisor.**

**(A) ModeSwitch-LLM (arXiv:2605.23057) — closest framing.** Runs on the *same*
model+hardware we use (Llama-3.1-8B on a single A100), routes each request to a
fixed mode (FP16 / GPTQ-4bit / INT8 / speculative / prefix-cache / hybrids) using
cheap workload features, and compares rule-based vs learned routers vs a
constraint-aware oracle (≈ our planned controller study; rule-based wins).
*Difference we rely on:* it does **stateless per-request routing among free,
co-resident modes with ~0.01 ms overhead and no reload** — there is **no
switching-cost term** and no context/slot knob. Our problem is exactly the cost
it ignores: a single-process edge engine where changing the quant file, context,
or slot count forces a real teardown+reload, and modes cannot co-reside for free.

**(B) Flying Serving (arXiv:2602.22593) — closest mechanism for "online
reconfiguration."** vLLM-based, 8×H200; switches DP↔TP online via zero-copy
weight views, KV remap, and a pre-built communicator pool, reducing a switch from
a **146–292 s cold restart (Llama-70B) to ~15 ms**. Its three motivating
scenarios (burst→throughput, priority, long-context→pool memory) map almost 1:1
onto ours. *Difference we rely on:* it **engineers the switch cost to ≈0** on a
GPU cluster; on single-process edge llama.cpp that machinery does not exist
(`-c` and `-np` are startup-only), so the cost is **irreducible** and must be
*reasoned about*, not eliminated. Different knob (parallelism, not quant/ctx/
slots).

**(C) Shen, Ye, Glynn, Jaillet (arXiv:2602.07663, math.OC) — the theory
neighbor + a terminology trap.** Formalizes *online configuration selection +
admission control* (config examples: quantization, parallelism), with a
**switching-aware fluid oracle** and an SP-UCB-OLP algorithm achieving
Õ(√(KT)) regret. **Trap:** their "switching-aware" means *aware that mixing
configs over time is valuable* (exploiting complementary resource budgets) — in
their model **switching is FREE; there is no reconfiguration cost.** That is the
*opposite* premise to our "switching-cost-aware." It is pure theory (synthetic +
Alibaba cluster traces; no LLM/llama.cpp), assumes stationary i.i.d. arrivals,
and adds an admission-control layer we do not emphasize. *Implication:* do **not**
claim a novel online-learning algorithm/regret bound — cite this as the
theoretical foundation and optionally **use/adapt SP-UCB-OLP as our controller**.
Our contribution is the systems realization on edge llama.cpp + the irreducible
switch-cost extension they omit + non-stationary edge workloads (their future
work) + the regime-dependence measurement.

**(D) DeepServeCB (IEEE GAIIS 2026) — the closest match to our *planned method*.**
A contextual-bandit (LinUCB) scheduler for serverless multi-tenant LLM serving
whose Request Profiler features (prompt length, predicted output length, KV-cache
hit ratio + system state) and reward (SLO − cost − violation) are nearly identical
to the bandit we sketched in §12.4. *Consequence:* we **cannot** claim "contextual
bandit for adaptive LLM serving config" as novel. *Differences we rely on:* its
action space is batch size / concurrency / KV-eviction / warm-up — cheap per-epoch
scheduling knobs with **no switching cost, no quantization knob, no context-length
knob**; it runs on GPU cloud (vLLM, 8×A100), not edge llama.cpp; and it assumes a
stationary reward (non-stationary is its future work). Reuse its feature set and
reward shape; cite it as prior art for the controller; differentiate on the
irreducible switch cost + memory-shape knobs + edge substrate. (Caveat: minor
venue, citation-padded; treat as a novelty data point, not a systems competitor.)

## 9★.3 Other neighbors (cite + distinguish, lower threat)

- **MCAP / NVE (arXiv:2604.21026, 2026) — NEW framing threat, pre-empt it.**
  A *load-time recomputable* per-layer importance signal driving per-layer
  precision (W4A8/W4A16) + residency (GPU/RAM/SSD pager) for **edge
  memory-constrained** inference; uses **llama.cpp Q4_0 as baseline**. Its wording
  ("load-time control layer," "recomputable when conditions change," calibration
  set as a deployment-time knob) overlaps our online-adaptive pitch. *Differences
  we rely on:* (1) **no switching cost** — profile computed once, 0 ms on reload,
  applying a config is free; (2) knob = per-layer precision/residency, **not**
  {quant file, ctx, slots}; (3) **load-time static-per-deployment, not online over
  a non-stationary trace** (no control loop); (4) no regime-dependence. *Status:*
  arXiv preprint, solo author, small evals — cite as a framing data point, not a
  systems competitor. Differentiate on switch-cost-as-modeled-object + serving-level
  memory-shape knobs + online control.
- **SwapServeLLM (SC-W '25):** engine-agnostic whole-**model** hot-swap via GPU
  checkpointing; *eliminates* switch cost (like Flying Serving), GPU datacenter,
  model-file knob only. Low novelty threat; **high profiling value** (model-load
  decomposition + quant-dependent disk-vs-memory load times, incl. Ollama=llama.cpp
  path — see 9★.4).
- **EdgeLoRA (MobiSys '25):** multi-tenant **LoRA-adapter** serving built on
  llama.cpp, on Jetson/Pi. Knob = adapter selection + LRU adapter cache (no engine
  reload). Medium framing overlap (edge+llama.cpp+dynamic), low mechanism overlap;
  reuse its workload model (9★.4).
- **MorphServe (2506.02006):** fine-grained *in-engine* layer-precision swap + KV
  resize, state-preserving, low overhead. We do coarse-grained engine-level
  reconfiguration with explicit reload cost on edge. Closest technical cousin.
- **HELIOS (2504.10724):** online model + early-exit-layer selection; notable for
  explicitly comparing "load more layers vs switch model" by overhead and gating
  with hysteresis (Confidence Breach Counter + Re-assessment Interval). Different
  knob; GPU.
- **ParaDySe (2511.13198):** parallel-strategy switching for *training*; gives a
  reusable cost-model recipe (RF interpolation + polynomial extrapolation),
  OOM-constrained selection, and γ-hysteresis to suppress frequent switches.
- **AIConfigurator (2601.06288):** *offline* framework-agnostic config search via
  operator-decomposed profiled database (sub-second CPU search, no per-config GPU
  runs). Static, no switching cost; useful predictor methodology.
- **ServerlessLLM (2401.14351) + serverless cold-start line (ParaServe,
  HydraServe, Tangram):** optimize *model-load* latency on GPU serverless. They
  treat load as the cost to beat; none target edge llama.cpp memory-shape reload.
- **QLM (2407.00047), Llumnix (2406.03243), DistServe/Splitwise/Sarathi:** queue
  management, request rescheduling, prefill/decode disaggregation — orthogonal
  serving axes within a fixed engine config.
- **Edge quant+placement / scheduling (cite + distinguish):** DILEMMA
  (2503.01704, ILP joint layer quantization + placement across edge devices —
  static/semi-static, not online keep-vs-reconfigure); QLLMS (IEEE 11044591,
  quantization-adaptive edge *scheduling* via an Available Quantization Set +
  stable matching — a task→server assignment problem, not in-instance
  reconfiguration; its finding that **latency is not monotonic in bit-width**
  independently supports our regime-dependence point); Online Scheduling under KV
  constraints (2502.07115, theory: scheduling *within* a fixed config). None
  reconfigures the engine online under switching cost.
- **Cloud autoscaling / multi-LLM routing (orthogonal):** Chiron (2501.08090,
  hierarchical autoscaling — cloud instance provisioning, not edge knob reshaping);
  MuxServe (2404.02015), RouteLLM (2406.18665), OmniRouter (2502.20576) — route or
  multiplex among existing model endpoints, not local memory-shape reconfiguration.

## 9★.4 What we reuse, by component

- **quality(quant) + footprint(quant) — cite, don't re-run:** Kurt quant-eval
  (arXiv:2601.14277) — per-scheme downstream accuracy + WikiText-2 PPL with std
  errors (Tables 2/6) and size/footprint + conversion time (Table 5) for 13
  llama.cpp GGUF schemes on Llama-3.1-8B. Quality/footprint are model-intrinsic so
  they transfer to A100/Jetson; gives the quality axis of the quant knob for free.
  (Their CPU throughput table does NOT transfer — throughput is regime-dependent.)
- **Workload features + rule-first baseline:** ModeSwitch-LLM (start with rules
  before any bandit/RL; its rule≈oracle result is a strong prior).
- **Switch-cost hysteresis / two-timescale gating:** HELIOS (CBC + RI),
  ParaDySe (γ-smoothing).
- **Config-performance predictor without per-config GPU runs:** AIConfigurator
  (operator decomposition) + ParaDySe (RF/PR).
- **Switch-cost / load-cost decomposition for our own profiler:** **SwapServeLLM
  (Table 1: load + torch.compile + CUDA-graph capture — cleanest teardown→load→
  warmup template, maps onto a llama.cpp reload), and its quant-dependent disk-vs-
  memory model-load numbers incl. the Ollama=llama.cpp path (Fig 5)**; ServerlessLLM
  (storage→host→GPU tiers), ParaDySe (>31 s reset breakdown: imports/data/model-
  reinit/optimizer), Flying Serving (146–292 s cold-restart reference numbers).
- **Non-stationary workload generator:** EdgeLoRA (Gamma arrivals + power-law/Zipf-α
  popularity + cv burstiness) — adapt "adapter popularity" → "config/regime
  popularity."

## 9★.5 Verified novelty claim

> Existing work either (a) eliminates reconfiguration cost via in-engine or
> zero-copy mechanisms (MorphServe, Flying Serving), (b) routes among free
> co-resident modes per request (ModeSwitch-LLM, HELIOS), or (c) searches static
> configs offline (AIConfigurator). We study **switch-cost-aware online
> keep-vs-reconfigure control for llama.cpp-style edge serving**, over the
> memory-shape knob set {quant file, context length, KV slots}, where the
> reconfiguration cost is irreducible and must be traded off against future
> benefit under non-stationary, memory-constrained workloads. We also contribute
> the empirical finding that a knob's benefit can be **regime-dependent** (the
> quantization knob's latency effect changes sign with hardware and decode batch
> size), which a fixed-sign cost model would get wrong.

Do **not** claim novelty for: dynamic quantization, KV resizing, model swapping,
hierarchical control, parallelism switching, multi-model routing, **load-time
recomputable control signals (MCAP/NVE), engine-agnostic model hot-swapping
(SwapServeLLM), or edge multi-tenant LoRA serving (EdgeLoRA)** — all are covered
above. Claim novelty for the **combination**: edge llama.cpp substrate +
memory-shape knobs + explicit irreducible switching cost + online control under
non-stationarity + the regime-dependence measurement.

---

# 11. Refined Paper Scope

## Recommended Title

> **Switching-Cost-Aware Configuration Control for llama.cpp-Based Edge LLM Serving**

Alternative:

> **Online Memory-Shape Adaptation for Edge LLM Serving with llama.cpp**

## Core Action Space

```text
A = {
    quantization/model file,
    context length,
    number of parallel KV slots
}
```

## Core State Space

```text
S = {
    queue length,
    arrival rate,
    prompt length statistics,
    output length statistics,
    current config,
    memory pressure,
    recent latency,
    recent quality estimate,
    time since last switch
}
```

## Core Objective

```text
maximize latency-quality-memory reward
minus switching/reconfiguration cost
```

---

# 12. Proposed Algorithmic Framework

## 12.1 Baseline 1: Static Configuration

Choose one configuration at startup and never change.

Example:

```text
Q4_K_M, 4096 ctx, 2 slots
```

## 12.2 Baseline 2: llama.cpp Built-in Policies

Use existing llama.cpp policies:

- static startup flags,
- LRU model eviction,
- fixed sleep threshold,
- default slot dispatch.

These are simple and honest baselines.

## 12.3 Baseline 3: Reactive Threshold Policy

Example:

```text
if queue_length > threshold:
    switch to lower quantization and more slots
if long_prompt_fraction > threshold:
    switch to longer context
```

This is a simple non-learning baseline.

## 12.4 Proposed Policy: Contextual Bandit

Arms:

```text
(Q2_K, 2048 ctx, 4 slots)
(Q4_K_M, 4096 ctx, 2 slots)
(Q8_0, 2048 ctx, 1 slot)
...
```

Context:

```text
recent arrival rate,
queue length,
prompt length histogram,
current active sessions,
current configuration
```

Reward:

```text
quality - latency penalty - switching cost
```

This is a practical first algorithm before full model-based RL.

## 12.5 Later Extension: Model-Based RL

Use offline profiling for:

- model load time,
- throughput under each configuration,
- memory footprint,
- approximate quality,
- reconfiguration cost.

Then learn only workload dynamics and quality/latency response under mission-specific workloads.

This aligns with the proposal document’s suggested model-based RL, hierarchical control, and contextual bandit machinery.

---

# 13. Minimum First Experiment

## Setup

One llama.cpp server on a Jetson-class or laptop-GPU edge device.

Candidate model files:

```text
Llama-3-8B Q2_K
Llama-3-8B Q4_K_M
Llama-3-8B Q8_0
```

Use:

```text
--models-dir
--models-max 1
```

to force model/quantization switching.

## Workload

Synthetic two-phase or three-phase trace:

1. **Phase A:** many short interactive requests.
2. **Phase B:** fewer but longer-context requests.
3. **Phase C:** bursty mixed workload.

## Baselines

- Static Q4_K_M.
- Static Q2_K.
- Static Q8_0.
- llama.cpp LRU model switching.
- Reactive threshold policy.
- Our contextual-bandit policy.

## Metrics

- average latency,
- P95/P99 latency,
- TTFT,
- token throughput,
- request success rate,
- quality surrogate,
- number of switches,
- total switch time,
- time spent unavailable/reconfiguring.

## Expected First Figure

A plot showing:

```text
tail latency / quality / switching count
```

for static, LRU, threshold, and our switching-cost-aware policy.

---

# 14. Comparison Table

| Work | Main Focus | Dynamic? | Model / Quantization Adaptation? | Context / KV Capacity Adaptation? | Explicit Switching Cost? | Edge / llama.cpp Focus? | Relation to Us |
|---|---|---:|---:|---:|---:|---:|---|
| vLLM / PagedAttention | KV memory management | Yes, inside engine | No | KV paging | No | No | Engine-level baseline |
| Llumnix | Request rescheduling | Yes | No | Request/KV migration | Migration cost considered, not config switching | No | Request-level baseline |
| Online KV Scheduling | Scheduling theory | Yes | No | Fixed KV constraint | No config switching | No | Theoretical scheduling baseline |
| MorphServe | Runtime layer/KV adaptation | Yes | Layer-level precision | KV resizing | Low-overhead runtime transition, not coarse model-file reload | No | Closest technical cousin |
| QLM | Queue management | Yes | Model swapping operation | Not main focus | Not our reconfig-cost objective | Cloud/fixed-capacity | Important related work |
| Chiron | Autoscaling | Yes | No | Batch size / instances | Scale-up overhead discussed | Cloud | Autoscaling cousin |
| DILEMMA | Edge distributed placement | Mostly optimization | Layer quantization | Layer placement | Not our online switching setting | Edge, distributed | Important if multi-server |
| LaTune (WWW '26) | Adaptive edge config tuning | Yes (online re-select) | No (excludes quant) | No (fixes ctx) | No (free re-selection, rank-stability) | **Yes (edge + llama.cpp)** | **Closest paper; we take the knobs it excludes** |
| ModeSwitch-LLM | Per-request mode routing | Yes | Quant modes (co-resident) | No | No (~0.01 ms routing, no reload) | No (A100) | Closest framing; free routing vs our reload |
| MCAP / NVE | Load-time per-layer profiling | Load-time only | Per-layer precision | Per-layer residency/paging | No (0 ms, amortized) | Edge (llama.cpp baseline) | Framing threat; no switch cost, wrong knob set |
| llama.cpp Quant Eval | Static quantization evaluation | No | Evaluates GGUF formats | No | No | Yes | Empirical support for quality/memory tradeoff |
| Ours | Configuration control | Yes | Model-file / quantization switching | Context + slot reshaping | Yes | Yes | Target contribution |

---

## Bottom Line

The topic is still viable, but we must sharpen the claim.

The strongest defensible version is:

> **Switching-cost-aware online configuration control for llama.cpp-based edge LLM serving, focused on memory-shape adaptation through quantization/model-file choice, context length, and concurrency slots — under an irreducible reconfiguration cost and non-stationary workloads.**

Per the verified pass (§9★), the closest threats to pre-empt are **LaTune** (edge +
llama.cpp adaptive config tuning, but excludes our knobs and has no switch cost),
**ModeSwitch-LLM** (free per-request mode routing), and **MCAP/NVE** (load-time
recomputable per-layer signal, no switch cost). MorphServe and QLM remain technical
cousins. None of the sources we verified studies the llama.cpp keep-vs-reconfigure
controller problem as framed above.
