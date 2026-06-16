"""Scenario A: Switching GGUF Files (Throughput Burst).

Knob: model / quantization file (Q8_0 vs Q4_K_M), with ctx and slots held equal.

Hypothesis: under a burst of concurrent users, the lighter Q4_K_M file clears
the burst faster (lower makespan / tail latency) than the heavier Q8_0 file.
If Q4 wins here, the GGUF-file knob is worth adapting when load spikes.

This only demonstrates the gap between two static configs; no controller.
"""

import asyncio
import os
import time

import common as C


BURST       = int(os.environ.get("BURST", "24"))       # concurrent users
PROMPT      = "Explain what a binary search tree is and how insertion works."
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "256"))  # decode-heavy
CTX         = int(os.environ.get("CTX", "4096"))
PARALLEL    = int(os.environ.get("PARALLEL", "8"))      # same for both configs

# Model files (override for a bigger model). Defaults: gemma-3-1b.
MODEL_Q8    = os.environ.get("MODEL_Q8", C.MODEL_Q8)
MODEL_Q4    = os.environ.get("MODEL_Q4", C.MODEL_Q4)
GPU_INDEX   = int(os.environ.get("GPU_INDEX", "0"))


async def run_burst(base):
    async with C.aiohttp.ClientSession() as s:
        t0 = time.time()
        tasks = [C.chat(s, base, PROMPT, MAX_TOKENS, tag=f"u{i}") for i in range(BURST)]
        results = await asyncio.gather(*tasks)
        makespan = time.time() - t0
    return results, makespan


def eval_config(model, label):
    base_vram = C.gpu_mem_used_mb(GPU_INDEX)
    with C.Server(model=model, ctx=CTX, parallel=PARALLEL, label=label) as srv:
        ready_vram = C.gpu_mem_used_mb(GPU_INDEX)
        results, makespan = asyncio.run(run_burst(srv.base))
        peak_vram = C.gpu_mem_used_mb(GPU_INDEX)
    summ = C.summarize(results)
    total_tokens = sum(r["tokens"] for r in results if r["ok"])
    summ["makespan"] = makespan
    summ["throughput"] = total_tokens / makespan if makespan else float("nan")
    summ["load_time"] = srv.load_time
    summ["vram_mb"] = peak_vram - base_vram if peak_vram == peak_vram else float("nan")
    return summ


def main():
    print("=" * 78)
    print(f"SCENARIO A  burst={BURST} concurrent, max_tokens={MAX_TOKENS}, "
          f"ctx={CTX}, np={PARALLEL}")
    print(f"  Q8 = {os.path.basename(MODEL_Q8)}")
    print(f"  Q4 = {os.path.basename(MODEL_Q4)}")
    print("=" * 78)

    q8 = eval_config(MODEL_Q8, "Q8_0")
    q4 = eval_config(MODEL_Q4, "Q4_K_M")

    def row(name, s):
        print(f"{name:>8} | ok {s['ok']}/{s['n']} | makespan {s['makespan']:6.2f}s "
              f"| tput {s['throughput']:7.1f} tok/s | lat_p95 {s['lat_p95']:6.2f}s "
              f"| vram {s['vram_mb']:7.0f} MiB | load {s['load_time']:5.2f}s")

    print("-" * 78)
    row("Q8_0", q8)
    row("Q4_K_M", q4)
    print("-" * 78)

    if q8["makespan"] > 0:
        speedup = q8["makespan"] / q4["makespan"]
        vram_save = q8["vram_mb"] - q4["vram_mb"]
        print(f"RESULT: Q4_K_M clears the burst {speedup:.2f}x faster than Q8_0 "
              f"(makespan {q8['makespan']:.2f}s -> {q4['makespan']:.2f}s)")
        print(f"        Q4_K_M uses {vram_save:.0f} MiB less VRAM "
              f"({q8['vram_mb']:.0f} -> {q4['vram_mb']:.0f} MiB)")
        verdict = "GAP CONFIRMED" if speedup > 1.15 else "no meaningful throughput gap"
        print(f"VERDICT: {verdict} -- model-file knob "
              f"{'matters' if speedup > 1.15 else 'gives mainly a VRAM win here'} "
              f"under burst")


if __name__ == "__main__":
    main()
