# llama.cpp as the Substrate for Our Branch 2 Paper

## 0. Refined Paper Direction

This note reframes our Branch 2 topic after reading **LaTune: Lightweight and Adaptive Configuration Tuning for LLM Inference on Edge Devices** https://dl.acm.org/doi/pdf/10.1145/3774904.3792382.

LaTune studies **adaptive runtime configuration tuning** for edge LLM inference. Its core question is:

> Given a fixed model, edge device, workload, and changing resource budget, which runtime configuration maximizes throughput while satisfying resource constraints?

Our topic should not simply duplicate this. Instead, we should go one level deeper into **llama.cpp-style edge serving dynamics**:

> Given an edge device with limited memory, multiple GGUF model/quantization variants, fixed KV-cache slots, and dynamic request arrivals, how should a controller jointly choose the model variant, runtime configuration, admission policy, and slot/scheduling behavior to optimize latency, throughput, robustness, and answer quality?

A possible title:

> **Multi-Level Adaptive Control for llama.cpp Edge Serving: Model Variant Selection, KV-Cache-Aware Configuration, and Dynamic Request Scheduling**

Alternative shorter titles:

- **Slot-Aware Adaptive Serving for Edge LLM Inference over llama.cpp**
- **KV-Cache- and Model-Variant-Aware Scheduling for Local LLM Serving on Edge Devices**
- **Beyond Static Tuning: Dynamic Request-Aware Control for llama.cpp Edge Inference**

---

## 1. Why We Are Looking at llama.cpp

The edge-deployed LLM setting — for example, a drone, robot, small autonomous platform, or local companion computer with no guaranteed ground link — rules out many serving stacks that assume a Python runtime, container orchestration, cloud control plane, or datacenter-grade GPU resources.

`llama.cpp` fits this setting because it provides:

- a single self-contained C/C++ inference engine;
- GGUF model files that package weights, quantization, tokenizer metadata, and model metadata;
- CPU, GPU, Metal, CUDA, Vulkan, and other backend support depending on build;
- local HTTP serving with OpenAI-compatible APIs;
- quantized model execution on consumer and embedded hardware;
- a slot-based serving structure with preallocated KV-cache regions;
- optional model-router behavior through multiple local model files.

A careful phrasing for the paper:

> `llama.cpp` plays a role analogous to vLLM in local edge deployment, but under very different design constraints: single-node execution, GGUF model packaging, tight memory budgets, CPU/GPU portability, and preallocated KV-cache slots.

This is better than overclaiming that `llama.cpp` is exactly the edge equivalent of vLLM. The analogy is useful, but the design space is different.

---

## 2. What LaTune Covers and What It Leaves Open

### 2.1 What LaTune Does

LaTune studies runtime configuration tuning for edge LLM inference engines. It uses three ideas:

1. **Parameter selection** — identify the few runtime parameters that dominate throughput and resource use.
2. **Knowledge transfer** — reuse good historical configurations from similar model/hardware/workload tasks.
3. **Two-stage optimization** — build a Pareto set offline, then choose the fastest feasible configuration online according to current resource availability.

LaTune’s target objective is mainly:

```text
maximize throughput
minimize resource usage
select feasible configuration under current resource budget
```

The paper evaluates across several edge devices, models, quantization levels, and load conditions. It reports that configuration rankings are often stable across resource levels, which justifies choosing from an offline Pareto set at runtime.

### 2.2 What LaTune Does Not Deeply Address

LaTune is very relevant, but it does not fully study the online serving dynamics that matter in `llama.cpp`:

- dynamic request arrivals;
- queueing delay;
- TTFT and TPOT under bursty workloads;
- slot assignment;
- KV-cache occupancy;
- prefix reuse;
- admission control;
- priority between real-time and background tasks;
- model/quantization switching cost;
- cold-start and model residency;
- tradeoff between answer quality and resource footprint.

LaTune also focuses on runtime-level parameters while mostly excluding model-compression decisions such as quantization precision from its main tuning scope. For our edge-serving setting, quantization/model-variant selection should be first-class because it changes memory feasibility, context capacity, concurrency, quality, and model loading cost.

### 2.3 Our Research Gap

LaTune asks:

> Which runtime configuration should a fixed local LLM inference setup use under changing resources?

Our work can ask:

> In a llama.cpp-style local serving system, how should the controller jointly manage model variant, runtime configuration, request admission, and slot/KV-cache resources under dynamic arrivals?

This gives us a distinct contribution:

```text
LaTune = adaptive configuration tuning
Our work = adaptive edge serving control
```

---

## 3. System Mental Model

### 3.1 Restaurant Analogy

- **Kitchen** = loaded model process.
- **Chef** = model variant and quantization level.
  - A larger chef, e.g. Q8 or a larger model, usually needs more memory but may provide better answer quality.
  - A smaller chef, e.g. Q4, Q3, or Q2, fits smaller devices and may serve more concurrent requests, but may reduce quality.
- **Tables** = slots.
  - Each slot has a preallocated KV-cache region.
  - The number of slots is controlled by server configuration such as `--parallel`.
- **Placemat** = context budget per slot.
  - Controlled by context size and KV-cache settings.
- **Host** = slot dispatcher.
  - Assigns incoming requests to available/reusable slots.
  - May use prefix similarity and LRU-like reuse.
- **Waiter** = continuous batching loop.
  - Collects work from active slots and dispatches token-generation work to the backend.
- **Concierge** = model router.
  - Manages multiple model files or child server processes.
  - Decides which model variant should serve which request.

This analogy is useful for explanation, but in the paper we should clearly separate actual engine behavior from our abstraction.

---

## 4. Engine Facts vs. Our Abstraction

### 4.1 Engine Facts

Facts we can rely on or verify experimentally:

- `llama.cpp` loads GGUF model files.
- Quantization level changes model memory footprint and usually changes speed/quality tradeoffs.
- The server supports concurrent slots / parallel sequences.
- KV cache consumes memory proportional to model architecture, context length, KV precision, and number of active slots.
- Runtime flags such as context size, parallelism, batch size, micro-batch size, GPU layers, and attention/KV options affect throughput and memory.
- The server exposes a local HTTP API, so a controller can be placed above it without modifying C++ code initially.

### 4.2 Our Abstraction

For research modeling, we can abstract the system as:

```text
Device
 ├── total memory
 ├── available memory over time
 ├── CPU/GPU compute capacity
 └── background load / thermal pressure

Model Variant
 ├── model family
 ├── parameter count
 ├── quantization level
 ├── model weight memory
 ├── loading time
 ├── expected quality score
 └── supported context length

llama.cpp Server Instance
 ├── loaded model variant
 ├── runtime configuration
 ├── N slots
 ├── KV cache per slot
 ├── continuous batching loop
 └── request queue / active slot set

Request
 ├── arrival time
 ├── prompt length
 ├── expected output length
 ├── task type
 ├── priority
 ├── prefix group / conversation id
 └── quality/latency requirement
```

---

## 5. Memory Model

The central memory equation is:

```text
Total memory used
≈ model weights
+ N_slots × KV_per_slot
+ temporary scratch / compute buffers
+ runtime overhead
```

Where:

```text
KV_per_slot
≈ function(model layers, hidden size, KV heads, context length, KV dtype)
```

This model creates the key tradeoff:

```text
larger model / higher quantization quality
    → more weight memory
    → fewer slots or smaller context
    → possibly better answer quality

smaller model / lower-bit quantization
    → less weight memory
    → more slots or larger context
    → possibly lower answer quality
```

This is the reason quantization/model variant must be included in our control problem.

---

## 6. Multi-Level Control Problem

We should model the system as having three timescales.

### 6.1 Slow Timescale: Model / Quantization Selection

Decision:

```text
Which GGUF model variant should be loaded or selected?
```

Examples:

```text
Llama-3-8B Q8_0
Llama-3-8B Q4_K_M
Llama-3-8B Q2_K
Phi/Qwen smaller model Q4/Q8
```

Important metrics:

- model load time;
- memory footprint;
- answer quality proxy;
- maximum feasible context length;
- maximum feasible slot count;
- cold-start cost;
- eviction/reload cost;
- whether the model can remain resident.

This decision should not be made for every request unless the router already keeps multiple model variants warm. Model switching is expensive and should be treated as slow adaptation.

### 6.2 Medium Timescale: Runtime Configuration Selection

Decision:

```text
How should llama.cpp be launched or configured?
```

Candidate knobs:

- `--parallel`
- `--ctx-size`
- `--batch-size`
- `--ubatch-size`
- `--n-gpu-layers`
- KV-cache type / offload options
- flash attention on/off
- CPU thread count

This overlaps with LaTune, but we will interpret these knobs through the lens of slots, KV cache, and dynamic serving.

### 6.3 Fast Timescale: Online Request Control

Decision:

```text
Which request should be sent now?
Which request should wait?
Which model/config should serve it?
Which priority class should be favored?
```

Control actions:

- admission control;
- queue ordering;
- priority scheduling;
- delay or reject background requests;
- route simple tasks to smaller/lower-bit model;
- reserve capacity for real-time requests;
- avoid overload that causes OOM or extreme TTFT.

If we modify `llama.cpp` later, we may also control:

- custom slot assignment;
- prefix-aware slot reuse;
- deadline-aware batching;
- token-level preemption;
- KV-cache eviction policy.

But the first implementation should use a Python control layer on top of `llama.cpp`.

---

## 7. Python Control Layer First

We should not modify `llama.cpp` at the beginning.

Initial architecture:

```text
Workload generator
        ↓
Python control layer
  ├── request queue
  ├── scheduler
  ├── admission controller
  ├── model/config manager
  ├── metrics logger
  └── resource monitor
        ↓
llama.cpp server process(es)
        ↓
local CPU/GPU/edge device
```

The Python layer can:

- launch `llama.cpp` with different configs;
- run one or multiple server processes;
- send HTTP requests;
- generate dynamic arrivals;
- implement priority queues;
- delay/reject requests;
- select among warmed model servers;
- collect latency and throughput metrics;
- monitor memory/GPU/CPU externally.

This is enough for the first paper experiments.

### When C++ Modification Becomes Necessary

We only need to modify `llama.cpp` if we require:

- custom internal slot assignment;
- token-level preemption;
- deadline-aware continuous batching;
- custom KV-cache eviction/reuse;
- detailed internal metrics not exposed externally.

Therefore:

```text
Phase 1 = Python controller + black/gray-box llama.cpp
Phase 2 = minimal llama.cpp modification if experiments prove it is necessary
```

---

## 8. Rank Stability Question

LaTune observes that configuration rankings can remain stable across resource levels: absolute TPS changes, but the relative order of configurations often stays similar.

We should test whether this remains true under dynamic request arrivals.

### 8.1 Static Rank Stability

Example:

```text
Config A: 100 TPS under idle, 50 TPS under high load
Config B: 80 TPS under idle, 38 TPS under high load
Config C: 60 TPS under idle, 25 TPS under high load
```

The absolute numbers drop, but the ranking remains:

```text
A > B > C
```

This is rank stability.

### 8.2 Dynamic-Arrival Rank Instability

Under real serving workloads, ranking may change because of:

- bursty arrivals;
- prompt length variability;
- prefill/decode phase interference;
- queueing delay;
- slot occupancy;
- KV-cache pressure;
- mixed short and long requests;
- real-time vs background priority;
- model switching cold-start cost.

Example:

```text
Static benchmark:
Config A has highest TPS.

Dynamic arrival benchmark:
Config A creates high TTFT or OOM risk.
Config B completes more requests with lower P95 latency.
```

If this happens, we can argue:

> Static configuration tuning is insufficient for edge LLM serving; dynamic request-aware and KV-cache-aware control is needed.

---

## 9. Proposed Experiments

### Experiment 1: Lightweight Model Variant Calibration

Goal:

> Build a small device-specific operating-region table for model/quantization variants, so the later controller knows which variants are feasible under different memory, latency, concurrency, and quality requirements.

Important clarification:

This experiment is **not** intended to be a full quantization benchmark paper. Prior work has already studied GGUF / `llama.cpp` quantization quality, perplexity, model size, throughput, latency, and edge-device behavior in much broader settings. Therefore, we should use existing work to justify the general quantization tradeoff and to narrow our search space. Our profiling should be a **lightweight calibration step** for our own device, build, workload, and serving controller.

Why this is still necessary:

```text
Published quantization results tell us general trends.
Our controller needs local operating regions.
```

Quantization behavior can depend on:

- target hardware;
- CPU/GPU backend;
- `llama.cpp` build;
- context length;
- `--parallel` / slot count;
- KV-cache pressure;
- workload length distribution;
- cold vs warm model state;
- dynamic request arrivals.

So we should not claim novelty from simply benchmarking Q4 vs Q8. Instead, we use profiling to parameterize the later adaptive controller.

#### 1.1 Existing Work We Can Reuse

We can cite existing work for the following points:

- Broad `llama.cpp` quantization evaluations already compare 3–8 bit K-quants and legacy formats, including downstream task accuracy, perplexity, CPU throughput, model size, compression, and quantization time.
- Edge quantization studies already evaluate quantized LLMs on constrained devices, including latency, energy, and accuracy.
- Public edge GGUF benchmark dashboards already report device-dependent throughput behavior, KV-cache collapse thresholds, and quality benchmarks across several devices and quantization variants.

Therefore, our profiling does **not** need to reproduce large benchmark suites such as MMLU, GSM8K, HumanEval, TruthfulQA, etc. We only need enough local data to support model/config/scheduling decisions.

#### 1.2 What We Should Profile Ourselves

We should profile only a small representative set of variants.

Minimal set:

```text
Same model family:
- Q8_0       high-quality / high-memory point
- Q4_K_M     balanced point
- Q2_K or Q3_K_M aggressive compression point
```

Better set if time allows:

```text
Same model family:
- Q8_0
- Q6_K
- Q4_K_M
- Q3_K_M
- Q2_K

Cross-size comparison:
- larger model with low-bit quantization
- smaller model with higher-bit quantization
```

The cross-size comparison is important because the controller may need to choose between:

```text
larger model Q3/Q4
vs.
smaller model Q8/Q4
```

This is more relevant to edge serving than simply asking which quantization is best for one model.

#### 1.3 Metrics to Collect

For each model variant, collect a compact profile:

```text
model_family
model_size
quantization
GGUF file size
load time
memory after load
peak RAM/VRAM during inference
single-request TPS
TTFT
TPOT
max feasible --parallel
max feasible --ctx-size
simple quality proxy
failure/OOM rate
```

We should especially focus on serving-level metrics that prior quantization studies may not fully cover for our setup:

- **load time**: needed to model cold-start and model-switching cost;
- **peak memory**: needed to decide whether a model can coexist with KV cache and slots;
- **TTFT**: needed for interactive or real-time edge requests;
- **TPOT**: needed to separate decode speed from prefill delay;
- **max feasible `--parallel`**: tells us how many slots the model can support;
- **max feasible `--ctx-size`**: tells us context capacity under memory limits;
- **quality proxy**: prevents the controller from always choosing the smallest/fastest variant.

#### 1.4 Quality Treatment

Do not make quality evaluation too large in the first version.

Recommended approach:

```text
Use quality as a constraint, not the main optimization objective.
```

That is:

```text
Only variants with quality_score ≥ Q_min are considered feasible.
Then optimize latency, throughput, memory, and robustness among feasible variants.
```

Possible lightweight quality proxies:

- a small task-specific prompt set;
- simple exact-match or multiple-choice questions;
- small summarization/instruction-following checklist;
- small domain-specific drone/edge command benchmark;
- optionally a small public benchmark subset if easy to automate.

The purpose is not to prove universal model quality. The purpose is to avoid obviously bad variants such as an overly aggressive quantization that fails the target task.

#### 1.5 Experimental Procedure

For each selected model variant:

```text
1. Launch llama.cpp server with a fixed baseline runtime configuration.
2. Measure server load time until ready.
3. Record memory after model load.
4. Send a warm-up request.
5. Run a fixed single-request benchmark.
6. Measure TPS, TTFT, TPOT, and peak memory.
7. Sweep --parallel until failure or unacceptable latency.
8. Sweep --ctx-size until failure or unacceptable latency.
9. Run the lightweight quality-proxy prompts.
10. Save all metrics to a CSV profile table.
```

Suggested CSV schema:

```text
model_family, model_size, quantization, gguf_file_size_GB,
load_time_s, memory_after_load_GB, peak_memory_GB,
ctx_size, parallel, batch_size, ubatch_size,
prompt_len, output_len,
single_request_tps, mean_ttft_s, p95_ttft_s,
mean_tpot_s, p95_latency_s,
max_feasible_parallel, max_feasible_ctx_size,
quality_score, quality_pass,
failure_rate, failure_reason
```

#### 1.6 Output of This Experiment

The output should be a small **model-variant operating-region table**:

```text
Variant A: high quality, high memory, low concurrency, long load time
Variant B: balanced quality/memory/speed, good default
Variant C: low memory, high concurrency, quality only acceptable for simple tasks
Variant D: smaller model, low TTFT, useful for real-time/simple requests
```

This table becomes an input to the later controller.

#### 1.7 Figures to Produce

Core figures:

1. **Memory vs. quantization**
   - x-axis: quantization variant
   - y-axis: memory after load / peak memory

2. **Latency and throughput vs. quantization**
   - x-axis: quantization variant
   - y-axis: TPS, TTFT, TPOT

3. **Quality-memory tradeoff**
   - x-axis: memory footprint
   - y-axis: quality proxy
   - color: quantization level

4. **Feasible serving region heatmap**
   - rows: model variants
   - columns: `--ctx-size` or `--parallel`
   - cell: success/failure or max feasible concurrency

#### 1.8 How to Position This Experiment in the Paper

Use this wording:

> Prior work has extensively profiled quantized LLM variants in terms of size, perplexity, accuracy, throughput, latency, and energy on local or edge devices. We therefore do not attempt to re-benchmark quantization broadly. Instead, we perform a lightweight device-specific calibration step to parameterize our controller's model/configuration selection under memory, KV-cache, slot, and dynamic-arrival constraints.

This framing keeps the experiment modest and avoids overclaiming novelty.

### Experiment 2: Static Runtime Configuration Sweep

Goal:

> Establish baseline rankings under static benchmarking.

Sweep:

```text
parallel ∈ {1, 2, 4, 8}
ctx_size ∈ {512, 1024, 2048, 4096}
batch_size ∈ {128, 256, 512}
ubatch_size ∈ {64, 128, 256}
gpu_layers ∈ feasible values
```

Metrics:

- TPS;
- memory;
- TTFT;
- TPOT;
- failure/OOM rate.

### Experiment 3: Dynamic Arrival Rank Stability

Goal:

> Test whether static rankings still hold when requests arrive dynamically.

Workloads:

```text
Poisson arrivals
bursty arrivals
short chat
long summarization
mixed short/long prompts
repeated-prefix conversations
real-time + background tasks
```

Rank configurations by:

- average TPS;
- completed requests per second;
- mean TTFT;
- P95 TTFT;
- P95 end-to-end latency;
- failure/OOM rate;
- deadline miss rate.

Main question:

```text
Does the best static configuration remain best under dynamic arrivals?
```

### Experiment 4: Python-Level Adaptive Controller

Goal:

> Show that a simple controller improves robustness and latency compared with fixed configurations.

Baselines:

1. default `llama.cpp` config;
2. best static TPS config;
3. fixed Q8 model;
4. fixed Q4 model;
5. LaTune-style best fixed model/config baseline;
6. our adaptive model/config/scheduling controller.

Simple controller policy:

```text
if request is real-time:
    route to low-latency model/config
    prioritize in queue
elif system load is high:
    use smaller/lower-memory model variant if available
    delay background requests
else:
    use higher-quality model variant
    batch more aggressively
```

### Experiment 5: Optional Internal llama.cpp Modification

Only if needed:

- expose slot status;
- expose KV-cache occupancy;
- implement custom slot assignment;
- implement priority-aware batching;
- implement prefix-aware eviction.

This should be future work unless Phase 1 shows a strong need.

---

## 10. Metrics

We should not only report TPS.

Important metrics:

```text
Throughput:
- output tokens per second
- completed requests per second

Latency:
- TTFT
- TPOT
- end-to-end latency
- P50/P95/P99 latency

Robustness:
- OOM/failure rate
- timeout rate
- deadline miss rate

Memory:
- peak RAM/VRAM
- KV-cache budget
- model residency count

Quality:
- benchmark score
- task success rate
- LLM-as-judge score if acceptable
- or use quality as a constraint rather than an objective

Adaptation cost:
- model load time
- restart time
- routing overhead
- controller overhead
```

---

## 11. Quality Treatment for Quantization

Quantization cannot be treated only as a memory optimization. It changes output quality.

Possible approaches:

### Option A: Quality Constraint

Only consider model variants that satisfy a minimum quality threshold:

```text
quality(model_variant, task_type) ≥ Q_min
```

Then optimize latency/throughput/memory within the feasible set.

### Option B: Utility Function

Define utility:

```text
utility = answer_quality
          - latency_penalty
          - memory_penalty
          - failure_penalty
```

### Option C: Task-Dependent Routing

Use higher-quality variants for complex tasks and smaller variants for simple tasks:

```text
complex summarization/reasoning → larger or higher-bit model
simple command/query → smaller or lower-bit model
background task → cheaper model/config
real-time task → lowest TTFT feasible model/config
```

For the first version, Option A or C is probably easiest.

---

## 12. Initial Implementation Plan

### Week 1: Benchmark Harness

Build scripts:

```text
launch_server.py
send_requests.py
collect_metrics.py
config_sweep.py
```

CSV schema:

```text
config_id, model_variant, quantization, parallel, ctx_size, batch_size,
ubatch_size, gpu_layers, workload, arrival_rate, prompt_len, output_len,
mean_tps, mean_ttft, p95_ttft, mean_tpot, p95_latency,
peak_memory, fail_rate, load_time
```

### Week 2: Dynamic Workload Generator

Implement:

```text
Poisson arrivals
bursty arrivals
short/long prompt mix
real-time/background priority mix
repeated-prefix workload
```

### Week 3: Rank Stability Study

Compare rankings under:

```text
static single request
fixed concurrency
dynamic Poisson arrivals
bursty arrivals
mixed prompt lengths
mixed priority workloads
```

### Week 4: Python Controller

Implement:

```text
priority queue
admission threshold
model/config routing rule
memory/load-aware decision rule
```

### Week 5: Evaluation and Paper Figures

Core figures:

1. memory breakdown across model variants;
2. static vs dynamic ranking heatmap;
3. latency-throughput tradeoff under dynamic arrivals;
4. controller vs fixed baselines;
5. failure rate under high load.

---

## 13. Contribution Statement Draft

This paper can claim:

1. **Measurement:** We show that static configuration rankings for local LLM inference can become unstable under dynamic request arrivals, especially when queueing, KV-cache pressure, and mixed prompt lengths are considered.

2. **System Model:** We introduce a llama.cpp-aware edge serving model that captures model variants, quantization, fixed KV-cache slots, context budgets, and dynamic arrivals.

3. **Controller:** We design a lightweight Python control layer that jointly manages model/config selection, request admission, and priority scheduling without requiring initial changes to `llama.cpp`.

4. **Evaluation:** We demonstrate improved latency, robustness, and memory feasibility compared with default serving, fixed quantization, and static best-configuration baselines.

---

## 14. Key Takeaway

LaTune is the closest related work, but our paper should not be another LaTune.

Our direction should be:

```text
from static configuration tuning
      ↓
to dynamic, model-variant-aware, slot/KV-cache-aware edge serving control
```

This direction is better aligned with `llama.cpp`, edge deployment, quantization tradeoffs, and our interest in performance modeling and request scheduling.
