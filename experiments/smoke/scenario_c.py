"""Scenario C: Modifying Parallel Slots (Anti-Blocking).

Knob: number of parallel slots (-np), 1 vs 4, ctx held equal.

Setup: a long-running "batch" job (big prompt, large generation) starts first.
0.3s later a short *interactive* request arrives.

Hypothesis:
  - np=1 : the single slot is busy with the batch job, so the interactive
           request is Head-of-Line blocked until the batch finishes -> huge TTFT.
  - np=4 : continuous batching serves the interactive request alongside the
           batch job -> interactive TTFT stays small.

We report the interactive request's TTFT and latency in both configs. A large
np=1 / np=4 ratio confirms the slots knob is worth adapting. No controller.
"""

import asyncio
import time

import common as C


CTX             = 8192
BATCH_PROMPT    = "Write a long, detailed essay about the history of computing."
BATCH_TOKENS    = 512     # keeps the batch slot busy for a while
INTER_PROMPT    = "Say the single word: pong."
INTER_TOKENS    = 16
INTER_DELAY_S   = 0.3     # interactive arrives shortly after the batch job


async def run_workload(base):
    async with C.aiohttp.ClientSession() as s:
        batch = asyncio.create_task(
            C.chat(s, base, BATCH_PROMPT, BATCH_TOKENS, tag="batch"))
        await asyncio.sleep(INTER_DELAY_S)
        inter = await C.chat(s, base, INTER_PROMPT, INTER_TOKENS, tag="interactive")
        batch_res = await batch
    return inter, batch_res


def eval_config(parallel, label):
    with C.Server(model=C.MODEL_Q4, ctx=CTX, parallel=parallel, label=label) as srv:
        inter, batch = asyncio.run(run_workload(srv.base))
    return inter, batch


def main():
    print("=" * 70)
    print(f"SCENARIO C  batch_tokens={BATCH_TOKENS}, interactive arrives "
          f"+{INTER_DELAY_S}s, ctx={CTX}")
    print("=" * 70)

    inter1, batch1 = eval_config(1, "np=1")
    inter4, batch4 = eval_config(4, "np=4")

    def row(name, inter, batch):
        print(f"{name:>6} | interactive: TTFT {inter['ttft']:6.2f}s "
              f"latency {inter['latency']:6.2f}s ok={inter['ok']} "
              f"| batch: latency {batch['latency']:6.2f}s tokens {batch['tokens']}")

    print("-" * 70)
    row("np=1", inter1, batch1)
    row("np=4", inter4, batch4)
    print("-" * 70)

    t1 = inter1["ttft"] if inter1["ttft"] else float("nan")
    t4 = inter4["ttft"] if inter4["ttft"] else float("nan")
    ratio = t1 / t4 if t4 and t4 > 0 else float("nan")
    print(f"RESULT: interactive TTFT  np=1: {t1:.2f}s  vs  np=4: {t4:.2f}s "
          f"({ratio:.1f}x)")
    print(f"VERDICT: {'GAP CONFIRMED' if ratio > 2 else 'no gap'} -- adding slots "
          f"{'unblocks the interactive user' if ratio > 2 else 'did not help here'}")


if __name__ == "__main__":
    main()
