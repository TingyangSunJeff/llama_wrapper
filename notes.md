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

### 5.4 What We Exclude in the First Version

To keep the first paper scoped:

- no full multi-model routing problem,
- no prefill/decode disaggregation,
- no cloud autoscaling,
- no distributed tensor/pipeline parallelism as the main focus,
- no new quantization algorithm,
- no new KV-cache compression method.

These topics are already covered by nearby work such as RouteLLM/OmniRouter for model routing, DistServe/Splitwise for prefill-decode separation, Chiron for autoscaling, and KVQuant/QServe/MorphServe for quantization or KV adaptation.

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

This section is the corrected Step 2 audit.

## 9.1 Closest Work: MorphServe

**Paper:** *MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing*  
**Sources:**

- arXiv: <https://arxiv.org/abs/2506.02006>
- OpenReview: <https://openreview.net/forum?id=1JyePezdlF>
- GitHub: <https://github.com/MorphServe/MorphServe>

MorphServe is the closest related work found so far. It dynamically adapts LLM serving under bursty workloads by using **runtime quantized layer swapping** and **pressure-aware KV cache resizing**.

### Why It Is Close

MorphServe overlaps with our idea on:

- dynamic workload-aware LLM serving,
- precision/quantization adaptation,
- KV capacity adaptation,
- latency/quality/memory trade-offs.

### Key Difference

MorphServe performs **fine-grained runtime adaptation inside the inference engine**, swapping selected full-precision layers with quantized alternatives and resizing KV cache at runtime.

Our proposed work studies **coarse-grained configuration control for llama.cpp-style edge serving**, where the controller chooses among practical runtime configurations:

```text
model / quantization file
context length
number of parallel slots
```

and explicitly accounts for switching/reload/reconfiguration cost.

### Novelty Impact

We cannot claim:

> “No one has studied dynamic quantization or KV adaptation.”

That would be false because MorphServe exists.

A safer claim is:

> Prior work such as MorphServe studies runtime morphological adaptation inside serving engines; we study switching-cost-aware configuration control for llama.cpp-style edge inference, where adaptation occurs through practical model-file and memory-shape choices.

---

## 9.2 QLM: Queue Management with Model Swapping

**Paper:** *Queue Management for SLO-Oriented Large Language Model Serving*  
**Venue:** SoCC 2024  
**Sources:**

- arXiv: <https://arxiv.org/abs/2407.00047>
- Author page: <https://apatke.github.io/publications/socc24/>
- IBM page: <https://research.ibm.com/publications/queue-management-for-large-language-model-serving--1>
- GitHub: <https://github.com/QLM-project/QLM>

QLM is a queue management system for mixed interactive and batch LLM requests across different models and SLOs. It estimates request waiting time and orchestrates operations such as request pulling, eviction, load balancing, and model swapping.

### Difference from Our Work

QLM is primarily a **queue-management and orchestration** system. It does not focus on llama.cpp-style memory-shape decisions such as:

```text
which GGUF quantization file to load
how large the context length should be
how many KV slots to allocate
when the reload/reconfiguration cost is worth paying
```

### Novelty Impact

We should not frame our contribution as merely:

> “dynamic model swapping for LLM serving.”

QLM already includes model swapping as an LLM serving operation.

Our novelty should be framed around:

> **edge-local memory-shape configuration control with explicit switching cost.**

---

## 9.3 Chiron: Hierarchical Autoscaling

**Paper:** *Hierarchical Autoscaling for Large Language Model Serving with Chiron*  
**Sources:**

- arXiv: <https://arxiv.org/abs/2501.08090>
- IBM discussion: <https://research.ibm.com/blog/qlm-chiron-llm-orchestration>

Chiron studies hierarchical autoscaling for LLM serving, using queue size, utilization, and SLOs to scale serving instances and batch sizes.

### Difference from Our Work

Chiron targets cloud-style autoscaling and instance-level resource management. Our proposed work targets edge/local inference where new cloud instances cannot be elastically provisioned, and where the main control levers are local model-file choice, context length, and KV-slot allocation.

### Novelty Impact

We should not claim hierarchical control for LLM serving is new. Chiron already uses hierarchical autoscaling ideas.

Our contribution is hierarchical control over a different action space:

```text
fast: slot/request decisions
medium: context/concurrency reshaping
slow: model/quantization switching
```

---

## 9.4 DILEMMA: Joint Quantization and Distributed Edge Inference

**Paper:** *DILEMMA: Joint LLM Quantization and Distributed LLM Inference Over Edge Computing Systems*  
**Sources:**

- arXiv HTML: <https://arxiv.org/html/2503.01704v1>
- ResearchGate copy: <https://www.researchgate.net/publication/389580913_DILEMMA_Joint_LLM_Quantization_and_Distributed_LLM_Inference_Over_Edge_Computing_Systems>

DILEMMA jointly optimizes layer placement and layer quantization across edge computing systems using an integer linear programming formulation.

### Difference from Our Work

DILEMMA focuses on distributed layer placement and quantization across edge servers. Our work focuses on online runtime configuration adaptation for llama.cpp-style serving instances.

### Novelty Impact

If we include multiple edge servers, we must distinguish carefully:

> We are not solving static layer placement and quantization across edge servers. We are solving online keep-vs-reconfigure decisions for practical llama.cpp serving configurations.

---

## 9.5 QLLMS: Quantization-Adaptive Edge Scheduling 

**Paper:** *QLLMS: Quantization-Adaptive LLM Scheduling for Partially Informed Edge Processing*  
**Source found:** IEEE Xplore search result: <https://ieeexplore.ieee.org/abstract/document/11044591>

The search result indicates that QLLMS studies quantization-adaptive LLM scheduling for edge processing, where scheduling and quantization selection are jointly considered.

### Core idea
The key abstraction in QLLMS is the Available Quantization Set, or AQS. For each task-server pair, AQS records which quantization options are feasible under the task’s SLO constraints. For example, a task may be feasible on a cheaper GPU under 4-bit quantization but infeasible under FP16 due to memory or latency constraints. The paper defines AQS through the intersection of quantization choices satisfying multiple SLOs, including latency and perplexity requirements.
QLLMS has three major modules:

AQS Profiler
Profiles LLM tasks across GPU types and quantization levels to determine feasible quantization options under perplexity and latency constraints.

AQS Reconstructer
Handles partially informed edge systems where not all task-server-quantization profiles are known. It uses low-rank matrix completion and singular value thresholding to reconstruct missing SLO/profile entries from partial samples. 

Stable Matching Scheduler
Uses a many-to-one deferred acceptance algorithm to match LLM tasks to edge servers. Tasks prefer cheaper feasible servers, while servers rank tasks based on normalized service-satisfaction ratios. The paper proves the resulting matching has no blocking pairs and is stable.

### Novelty Impact

First, it supports the claim that quantization choice and scheduling/resource allocation are deeply coupled in edge LLM serving. 
Second, its measurement study supports our argument that quantization affects memory, latency, quality, and cost in nontrivial ways. In particular, latency is not always monotonic in bit-width, so a controller needs empirical profiles or learned performance models. 
Third, QLLMS leaves open the configuration-control layer that we want to study. It decides where a task should run and with what quantization; it does not decide when a long-running local serving instance should pay the cost to reload a different model file, reshape context length, or change slot concurrency.

QLLMS shows that quantization-aware scheduling is important for heterogeneous edge LLM serving, but it treats quantization as part of a task-server assignment problem. Our work studies a different control layer: an online llama.cpp-style serving instance must decide when to keep its current memory shape and when to pay a reconfiguration cost to change model/quantization file, context length, and KV-slot concurrency under changing workloads.

---

## 9.6 Online Scheduling with KV Cache Constraints

**Paper:** *Online Scheduling for LLM Inference with KV Cache Constraints*  
**arXiv:** `2502.07115`  
**Source:** <https://arxiv.org/abs/2502.07115>

This paper studies online scheduling under KV-cache memory constraints. It introduces a hindsight-optimal benchmark, proves limitations of deterministic online algorithms under arbitrary arrivals, and proposes an online scheduling algorithm.

### Difference from Our Work

It studies:

```text
fixed configuration → how to schedule requests
```

Our work studies:

```text
changing configuration → when and how to reconfigure
```

---

## 9.7 llama.cpp Quantization Evaluation

**Paper:** *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct*  
**arXiv:** `2601.14277`  
**Source:** <https://arxiv.org/pdf/2601.14277>

This paper evaluates llama.cpp GGUF quantization schemes on Llama-3.1-8B-Instruct, including 3–8 bit K-quant and legacy formats. It measures downstream performance, perplexity, model size, compression, quantization time, and CPU throughput.

### Difference from Our Work

It is a **static empirical evaluation** of quantization choices. It does not study online configuration control, switching cost, context-length adaptation, or concurrency-slot reshaping.

### Why It Is Useful

It can provide empirical support for our quality/latency/memory trade-off model across GGUF quantization options.

---

## 9.8 Llumnix: Dynamic Request Scheduling

**Paper:** *Llumnix: Dynamic Scheduling for Large Language Model Serving*  
**Venue:** OSDI 2024  
**Sources:**

- USENIX PDF: <https://www.usenix.org/system/files/osdi24-sun-biao.pdf>
- arXiv: <https://arxiv.org/abs/2406.03243>
- artifact repo: <https://github.com/alibaba/llm-scheduling-artifact>

Llumnix dynamically reschedules requests across multiple model instances and migrates request state to improve load balancing, reduce fragmentation, and handle priorities/SLOs.

### Difference from Our Work

Llumnix adapts **request placement** across fixed serving instances. Our work adapts the **serving configuration itself**.

This distinction is central:

```text
Llumnix: online scheduling within a fixed configuration
Ours: online reconfiguration of the serving engine
```

---

## 9.9 DistServe, Splitwise, and Sarathi-Serve

### DistServe

**Paper:** *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*  
**Sources:**

- arXiv: <https://arxiv.org/abs/2401.09670>
- USENIX PDF: <https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf>
- GitHub: <https://github.com/LLMServe/DistServe>

DistServe disaggregates prefill and decoding across different GPUs and co-optimizes phase-specific resource allocation and parallelism.

### Splitwise

**Paper:** *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*  
**Sources:**

- arXiv: <https://arxiv.org/abs/2311.18677>
- Microsoft blog: <https://www.microsoft.com/en-us/research/blog/splitwise-improves-gpu-usage-by-splitting-llm-inference-phases/>
- artifact: <https://zenodo.org/records/11003049>

Splitwise splits prompt computation and token generation across different machines to improve throughput, cost, and power efficiency.

### Sarathi-Serve

**Paper:** *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*  
**Sources:**

- USENIX: <https://www.usenix.org/conference/osdi24/presentation/agrawal>
- GitHub: <https://github.com/microsoft/sarathi-serve>
- Earlier SARATHI paper: <https://arxiv.org/abs/2308.16369>

Sarathi-Serve uses chunked-prefills and stall-free scheduling to improve the throughput-latency trade-off in LLM serving.

### Difference from Our Work

These systems optimize scheduling, phase separation, or cluster resource allocation. They do not focus on llama.cpp-style online configuration control over quantization, context length, and concurrency slots.

---

## 9.10 MuxServe, RouteLLM, and OmniRouter

### MuxServe

**Paper:** *MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving*  
**Sources:**

- arXiv: <https://arxiv.org/abs/2404.02015>
- GitHub: <https://github.com/hao-ai-lab/MuxServe>
- project blog: <https://haoailab.com/blogs/muxserve/>

MuxServe serves multiple LLMs efficiently using spatial-temporal multiplexing, model colocation, and adaptive batch scheduling.

### RouteLLM

**Paper:** *RouteLLM: Learning to Route LLMs with Preference Data*  
**Sources:**

- arXiv: <https://arxiv.org/abs/2406.18665>
- ICLR 2025: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html>
- OpenReview: <https://openreview.net/forum?id=8sSqNntaMr>
- GitHub: <https://github.com/lm-sys/RouteLLM>

RouteLLM learns routers that dynamically choose between stronger and weaker LLMs to trade off cost and quality.

### OmniRouter

**Paper:** *OmniRouter: Budget and Performance Controllable Multi-LLM Routing*  
**Sources:**

- arXiv: <https://arxiv.org/abs/2502.20576>
- ACM page: <https://dl.acm.org/doi/10.1145/3787470.3787480>

OmniRouter formulates multi-LLM routing as constrained optimization under budget/performance constraints.

### Difference from Our Work

These works route requests among existing model endpoints. Our work decides which local configuration should be active on a memory-limited edge inference engine.

---

# 10. Updated Novelty Claim

After the corrected literature check, the safe novelty claim is:

> Existing LLM serving work studies engine-level KV memory management, request scheduling, model routing, autoscaling, distributed phase disaggregation, and runtime precision/KV adaptation. However, there remains a gap for **switching-cost-aware online configuration control in llama.cpp-style edge serving**, where the system must decide when to change practical deployment knobs — model/quantization file, context length, and number of KV slots — under strict local memory constraints and non-negligible reconfiguration cost.

We should **not** claim:

- dynamic quantization is new,
- KV resizing is new,
- model swapping is new,
- hierarchical control is new,
- multi-model routing is new.

Those are already studied by MorphServe, QLM, Chiron, RouteLLM/OmniRouter, and related systems.

Instead, we should claim:

> The novelty is the **combination of llama.cpp-style edge constraints, memory-shape configuration, and explicit switching-cost-aware online control.**

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
| llama.cpp Quant Eval | Static quantization evaluation | No | Evaluates GGUF formats | No | No | Yes | Empirical support for quality/memory tradeoff |
| Ours | Configuration control | Yes | Model-file / quantization switching | Context + slot reshaping | Yes | Yes | Target contribution |

---

# 15. One-Paragraph Pitch for the Group

> llama.cpp is the edge-equivalent of vLLM: a lightweight inference engine that serves local users through memory-constrained model weights and KV slots, with practical deployment knobs such as GGUF quantization, context length, and parallel slots. Existing LLM serving work has studied KV paging, request scheduling, multi-model routing, autoscaling, phase disaggregation, and even runtime precision/KV adaptation. However, these works do not directly address the llama.cpp-style edge problem: deciding when to pay a nontrivial reconfiguration cost to change the active model/quantization file, context length, or KV-slot shape under a shifting workload. Our next paper studies this missing layer: switching-cost-aware online configuration control for edge LLM serving. The controller observes workload and system state, then decides whether to keep the current configuration or reconfigure the memory shape of the serving engine to trade off latency, quality, concurrency, and disruption cost.

---

# 16. Must-Read List Before Finalizing

1. **MorphServe** — closest technical overlap; dynamic quantized layer swapping and KV resizing.  
   <https://arxiv.org/abs/2506.02006>

2. **QLM** — queue management with model swapping operations.  
   <https://arxiv.org/abs/2407.00047>

3. **QLLMS** — potentially close quantization-adaptive edge scheduling; full paper needed.  
   <https://ieeexplore.ieee.org/abstract/document/11044591>

4. **DILEMMA** — joint quantization and distributed edge inference.  
   <https://arxiv.org/html/2503.01704v1>

5. **llama.cpp quantization evaluation** — useful empirical support for GGUF quantization trade-offs.  
   <https://arxiv.org/pdf/2601.14277>

6. **Online Scheduling with KV Cache Constraints** — theoretical scheduling baseline under KV constraints.  
   <https://arxiv.org/abs/2502.07115>

---

## Bottom Line

The topic is still viable, but we must sharpen the claim.

The strongest defensible version is:

> **Switching-cost-aware online configuration control for llama.cpp-based edge LLM serving, focused on memory-shape adaptation through quantization/model-file choice, context length, and concurrency slots.**

The main danger is **MorphServe**, and the second danger is **QLM/QLLMS**. But none of the sources we verified exactly study the llama.cpp controller problem as framed above.
