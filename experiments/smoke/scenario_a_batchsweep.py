"""Scenario A batch sweep: reproducible, repeated llama-batched-bench measurement.

Background: scenario_a_findings.md reported a non-monotonic decode-throughput
result (Q8_0 beats Q4_K_M in the batch 4-12 window on the A100) from a SINGLE
manual run of llama-batched-bench, with no repeats and no error bars. This
script makes that measurement reproducible and quantifies run-to-run noise:

  - sweeps the decode batch size (the -npl / parallel-sequences axis) for each
    model file,
  - runs each full sweep R times (plus a discarded warmup) to capture timing
    variance (GPU clocks/thermals/scheduling), and
  - reports mean +/- std of decode throughput (speed_tg) per (model, batch),
    plus the winner by mean, and saves raw jsonl logs + a markdown summary.

This is a steady-state micro-benchmark of the decode kernels, NOT the live
server. The decode batch is the number of sequences generated together (= -npl
here; emergent from concurrency on the live server). It does not touch -np slots
on a running server.

Usage:
  /scratch2/tingyang/anaconda/envs/mynewenv/bin/python scenario_a_batchsweep.py
  NPL=1,2,4,8,16 REPEATS=3 NTG=128 python scenario_a_batchsweep.py
"""

import json
import os
import subprocess
import statistics
import time
from datetime import datetime

REPO     = "/scratch2/tingyang/llama.cpp"
BIN      = os.environ.get("BATCHED_BENCH", f"{REPO}/build-cuda/bin/llama-batched-bench")
MODEL_DIR = os.environ.get("LLAMA_MODEL_DIR", f"{REPO}/models")
GPU_INDEX = os.environ.get("GPU_INDEX", "0")

# Models to compare. Skipped automatically if the file is missing.
MODELS = {
    "Q8_0":   f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    "Q4_K_M": f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "Q4_0":   f"{MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf",
}

NPL     = os.environ.get("NPL", "1,2,4,8,16,32,64,128")   # decode batch sizes
NPP     = os.environ.get("NPP", "16")                     # prompt tokens
NTG     = os.environ.get("NTG", "128")                    # gen tokens (decode-heavy)
CTX     = os.environ.get("CTX", "8192")
NGL     = os.environ.get("NGL", "99")
REPEATS = int(os.environ.get("REPEATS", "5"))
WARMUP  = int(os.environ.get("WARMUP", "1"))              # discarded runs

OUTDIR  = os.environ.get("OUTDIR", f"{REPO}/experiments/smoke/results")


def run_sweep_once(model_path):
    """One full -npl sweep. Returns {pl: speed_tg} parsed from jsonl, + raw text."""
    cmd = [
        BIN, "-m", model_path, "-c", CTX, "-ngl", NGL,
        "-npp", NPP, "-ntg", NTG, "-npl", NPL,
        "--output-format", "jsonl",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_INDEX)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    rows = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "pl" in obj and "speed_tg" in obj:
            rows[int(obj["pl"])] = float(obj["speed_tg"])
    if not rows:
        raise RuntimeError(f"no jsonl rows parsed for {model_path}; "
                           f"stderr tail:\n{proc.stderr[-500:]}")
    return rows, proc.stdout


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batches = [int(x) for x in NPL.split(",")]
    models = {k: v for k, v in MODELS.items() if os.path.exists(v)}
    missing = [k for k in MODELS if k not in models]

    print("=" * 78)
    print(f"SCENARIO A BATCH SWEEP  npp={NPP} ntg={NTG} ctx={CTX} ngl={NGL} "
          f"gpu={GPU_INDEX}")
    print(f"  repeats={REPEATS} (+{WARMUP} warmup discarded)  batches={batches}")
    print(f"  models: {', '.join(models)}"
          + (f"   (MISSING, skipped: {', '.join(missing)})" if missing else ""))
    print("=" * 78)

    # samples[model][pl] = list of speed_tg across repeats
    samples = {m: {b: [] for b in batches} for m in models}
    raw_log_path = f"{OUTDIR}/batchsweep_{stamp}.raw.jsonl"
    with open(raw_log_path, "w") as raw:
        for m, path in models.items():
            print(f"\n[{m}] {os.path.basename(path)}")
            for w in range(WARMUP):
                print(f"  warmup {w+1}/{WARMUP} ...", flush=True)
                run_sweep_once(path)
            for r in range(REPEATS):
                t0 = time.time()
                rows, stdout = run_sweep_once(path)
                for b in batches:
                    if b in rows:
                        samples[m][b].append(rows[b])
                raw.write(f"# model={m} repeat={r}\n")
                raw.write(stdout)
                print(f"  repeat {r+1}/{REPEATS} done in {time.time()-t0:.1f}s  "
                      + " ".join(f"b{b}:{rows.get(b, float('nan')):.0f}" for b in batches),
                      flush=True)

    def stats(vals):
        if not vals:
            return float("nan"), float("nan")
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    # --- console table ------------------------------------------------------
    mlist = list(models)
    print("\n" + "=" * 78)
    print("DECODE THROUGHPUT speed_tg (tok/s), mean +/- std over "
          f"{REPEATS} runs")
    hdr = f"{'batch':>5} | " + " | ".join(f"{m:>16}" for m in mlist) + " | winner(mean)"
    print(hdr)
    print("-" * len(hdr))
    lines_md = []
    for b in batches:
        cells, means = [], {}
        for m in mlist:
            mean, std = stats(samples[m][b])
            means[m] = mean
            cells.append(f"{mean:7.0f} +/-{std:5.1f}")
        winner = max(means, key=means.get)
        runner = sorted(means.values(), reverse=True)
        gap = (runner[0] - runner[1]) / runner[1] * 100 if len(runner) > 1 and runner[1] else 0.0
        print(f"{b:>5} | " + " | ".join(cells) + f" | {winner} (+{gap:.0f}%)")
        lines_md.append((b, {m: stats(samples[m][b]) for m in mlist}, winner, gap))

    # --- markdown summary ---------------------------------------------------
    md_path = f"{OUTDIR}/batchsweep_{stamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Scenario A batch sweep — {stamp}\n\n")
        f.write(f"- binary: `{BIN}`\n- gpu: CUDA_VISIBLE_DEVICES={GPU_INDEX}\n")
        f.write(f"- npp={NPP} ntg={NTG} ctx={CTX} ngl={NGL}\n")
        f.write(f"- repeats={REPEATS} (+{WARMUP} warmup discarded), batches={batches}\n")
        f.write(f"- raw log: `{os.path.basename(raw_log_path)}`\n\n")
        f.write("Decode throughput `speed_tg` (tok/s), mean ± std:\n\n")
        f.write("| batch | " + " | ".join(mlist) + " | winner (mean) | gap |\n")
        f.write("|---:" * (len(mlist) + 1) + "|:--|---:|\n")
        for b, st, winner, gap in lines_md:
            cells = " | ".join(f"{st[m][0]:.0f} ± {st[m][1]:.1f}" for m in mlist)
            f.write(f"| {b} | {cells} | **{winner}** | +{gap:.0f}% |\n")
        f.write("\nNotes:\n")
        f.write("- Steady-state decode micro-bench (`llama-batched-bench`), not the "
                "live server; `pl` = decode batch (sequences generated together).\n")
        f.write("- `std` is run-to-run timing noise (GPU clocks/thermal/scheduling), "
                "not sampling.\n")
        if "Q4_0" in mlist:
            f.write("- The Q4_0 GGUF was produced by **requantizing from Q8_0** "
                    "(`--allow-requantize`). Valid for a *throughput* comparison "
                    "(format/layout is correct); NOT a quality comparison "
                    "(quantizing from f16 would give different weights).\n")

    print("\nSaved:")
    print(f"  {md_path}")
    print(f"  {raw_log_path}")


if __name__ == "__main__":
    main()
