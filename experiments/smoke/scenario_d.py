"""Scenario D: Reconfiguration cost vs. a switching-cost-aware control layer.

Scenarios A/B/C only compare two *static* configs and ignore the cost of
actually changing a knob. This scenario adds that cost back in and asks the
real question: *given that a knob change forces a server teardown + reload +
warmup, when does adapting actually pay off, and when does a naive "always
adapt" policy hurt?*

We use the quant-file knob (Q4_K_M vs Q8_0) because we already verified it is
**regime-dependent** (see scenario_a_findings.md): Q4 wins single-stream /
bandwidth-bound decode, Q8 wins the mid-batch concurrent window on a datacenter
GPU. So neither static config is best in both regimes.

Workload model: the load alternates between two regimes, each lasting D seconds:
  - LOW   : interactive, ~single-stream  (Q4 tends to win)
  - BURST : many concurrent users        (Q8 tends to win, batch 4-12 window)

The driver does three things, all measured live:
  1. measure each config's token rate in each regime (4 short runs),
  2. measure the real reconfiguration cost C_switch of a GGUF swap
     (teardown -> boot -> first-token warmup),
  3. replay the alternating trace under several policies and a sweep of dwell
     times D, accounting for the downtime each reconfiguration costs.

Policies compared:
  - static-Q4 / static-Q8 : never reconfigure.
  - always-switch         : reconfigure to the regime-optimal config every phase.
  - cost-aware control     : reconfigure only if the future gain over the
                            remaining phase outweighs the reload downtime.
  - oracle                 : best achievable per phase (upper bound).

Expected story (the point of the experiment):
  - short D (regimes flip fast): the cost-aware controller declines to switch
    and matches the best static config, while always-switch loses to it
    (it pays reload downtime it never recoups).
  - long D (regimes are durable): the controller switches each phase, beating
    both static configs.
That crossover is the case where a switching-cost-aware control layer helps.
"""

import asyncio
import os
import time

import common as C


# Models: default to the 8B pair (clear GPU sign-flip) if present, else gemma-1B.
def _default_models():
    q4_8b = f"{C.MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    q8_8b = f"{C.MODEL_DIR}/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    if os.path.exists(q4_8b) and os.path.exists(q8_8b):
        return q4_8b, q8_8b
    return C.MODEL_Q4, C.MODEL_Q8


_DEF_Q4, _DEF_Q8 = _default_models()
MODEL_Q4   = os.environ.get("MODEL_Q4", _DEF_Q4)
MODEL_Q8   = os.environ.get("MODEL_Q8", _DEF_Q8)

CTX        = int(os.environ.get("CTX", "4096"))
PARALLEL   = int(os.environ.get("PARALLEL", "8"))    # slot count for BURST regime
BURST      = int(os.environ.get("BURST", "8"))       # concurrent users in BURST
LOW_N      = int(os.environ.get("LOW_N", "6"))       # sequential reqs in LOW
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
GPU_INDEX  = int(os.environ.get("GPU_INDEX", "0"))

PROMPT = "Explain what a binary search tree is and how insertion works."


# ---------------------------------------------------------------------------
# Live measurement helpers
# ---------------------------------------------------------------------------
async def _rate_low(base, n, max_tokens):
    """Sequential (concurrency 1) requests -> aggregate tokens/sec."""
    async with C.aiohttp.ClientSession() as s:
        t0 = time.time()
        toks = 0
        for i in range(n):
            r = await C.chat(s, base, PROMPT, max_tokens, tag=f"low{i}")
            if r["ok"]:
                toks += r["tokens"]
        dt = time.time() - t0
    return toks / dt if dt > 0 else float("nan")


async def _rate_burst(base, b, max_tokens):
    """b concurrent requests -> aggregate tokens / makespan."""
    async with C.aiohttp.ClientSession() as s:
        t0 = time.time()
        tasks = [C.chat(s, base, PROMPT, max_tokens, tag=f"b{i}") for i in range(b)]
        res = await asyncio.gather(*tasks)
        dt = time.time() - t0
    toks = sum(r["tokens"] for r in res if r["ok"])
    return toks / dt if dt > 0 else float("nan")


async def _warmup_ttft(base):
    """One tiny request after the server is healthy -> first-token latency."""
    async with C.aiohttp.ClientSession() as s:
        r = await C.chat(s, base, "Say: ready.", max_tokens=4, tag="warmup")
    return r["ttft"] if r["ttft"] is not None else r["latency"]


def _measure_config(model, label):
    """Boot a config, return (rates, server) keeping it running for caller."""
    srv = C.Server(model=model, ctx=CTX, parallel=PARALLEL, label=label)
    srv.__enter__()
    try:
        warm = asyncio.run(_warmup_ttft(srv.base))
        r_low = asyncio.run(_rate_low(srv.base, LOW_N, MAX_TOKENS))
        r_burst = asyncio.run(_rate_burst(srv.base, BURST, MAX_TOKENS))
    except Exception:
        srv.__exit__()
        raise
    return {"low": r_low, "burst": r_burst, "boot": srv.load_time, "warmup": warm}, srv


def _teardown(srv):
    t0 = time.time()
    srv.__exit__()
    return time.time() - t0


# ---------------------------------------------------------------------------
# Trace-replay accounting (uses the live-measured rates + C_switch)
# ---------------------------------------------------------------------------
def _phase_tokens(rate, dwell, downtime):
    """Tokens served in one phase of length `dwell` losing `downtime` to reload."""
    served = max(0.0, dwell - downtime)
    return rate * served


def replay(rates, cswitch, dwell, n_phases):
    """Return goodput (tokens) per policy for an alternating LOW/BURST trace.

    rates: {"Q4": {"low":r, "burst":r}, "Q8": {...}}
    Regime of phase i: LOW if i even else BURST.
    """
    regimes = ["low" if i % 2 == 0 else "burst" for i in range(n_phases)]
    best_cfg = {reg: ("Q4" if rates["Q4"][reg] >= rates["Q8"][reg] else "Q8")
                for reg in ("low", "burst")}

    out = {}

    # Static policies: never reconfigure, no downtime.
    for cfg in ("Q4", "Q8"):
        out[f"static-{cfg}"] = sum(_phase_tokens(rates[cfg][reg], dwell, 0.0)
                                   for reg in regimes)

    # Always-switch: jump to the regime-optimal config each phase; pay C_switch
    # whenever the config actually changes. This is the *cost-unaware* version of
    # adaptation -- it is what you get if you assume C_switch=0 (the implicit
    # assumption in prior free-switching work). It is our naive-adapt baseline.
    cur, tot, downtime = None, 0.0, 0.0
    for reg in regimes:
        opt = best_cfg[reg]
        dt = cswitch if (cur is not None and opt != cur) else 0.0
        tot += _phase_tokens(rates[opt][reg], dwell, dt)
        downtime += dt
        cur = opt
    out["always-switch"] = tot
    out["_always_downtime"] = downtime

    # Cost-aware control layer: the optimal keep-vs-reconfigure policy for the
    # *measured* C_switch (solved exactly by DP over the config held at the end
    # of each phase; phase-0 boot is free, every later real change pays C_switch).
    # A deployed online controller approximates this with a regime-duration
    # estimate; here we compute the achievable target. Note the only difference
    # from always-switch is that this one *counts* C_switch -- so it declines
    # switches whose benefit does not clear the reload downtime (incl. the cost
    # of eventually switching back).
    cfgs = ("Q4", "Q8")
    # dp[c] = (best tokens through current phase ending in config c, downtime)
    dp = {c: (_phase_tokens(rates[c][regimes[0]], dwell, 0.0), 0.0) for c in cfgs}
    for reg in regimes[1:]:
        nxt = {}
        for c in cfgs:
            best = None
            for p in cfgs:
                dt = cswitch if c != p else 0.0
                tok = dp[p][0] + _phase_tokens(rates[c][reg], dwell, dt)
                cand = (tok, dp[p][1] + dt)
                if best is None or cand[0] > best[0]:
                    best = cand
            nxt[c] = best
        dp = nxt
    best_end = max(dp.values(), key=lambda x: x[0])
    out["cost-aware"] = best_end[0]
    out["_ctrl_downtime"] = best_end[1]

    out["_best_cfg"] = best_cfg
    return out


def _single_switch_breakeven(rates, cswitch):
    """Break-even dwell for ONE switch in isolation (ignores the return trip).

    A single switch into the regime-optimal config pays off this phase when
    rate_opt*(D - Cs) > rate_cur*D, i.e. D > rate_opt*Cs/(rate_opt - rate_cur).
    Reported for the regime with the larger gain. NOTE: this is NOT the dwell at
    which *adaptation beats the best static* -- in an alternating workload every
    switch implies a later switch back, so the round-trip cost is higher (see
    _adaptation_pays_dwell).
    """
    thresholds = []
    for reg in ("low", "burst"):
        opt = "Q4" if rates["Q4"][reg] >= rates["Q8"][reg] else "Q8"
        other = "Q8" if opt == "Q4" else "Q4"
        r_opt, r_cur = rates[opt][reg], rates[other][reg]
        if r_opt > r_cur:
            thresholds.append(r_opt * cswitch / (r_opt - r_cur))
    return min(thresholds) if thresholds else float("nan")


def _adaptation_pays_dwell(rates, cswitch, n_phases, dmax=2000.0):
    """Smallest dwell at which the cost-aware policy strictly beats the best
    static config (i.e. round-trip adaptation actually wins). Found by scan."""
    d = 1.0
    while d <= dmax:
        rep = replay(rates, cswitch, d, n_phases)
        best_static = max(rep["static-Q4"], rep["static-Q8"])
        if rep["cost-aware"] > best_static * 1.001:
            return d
        d += 1.0
    return float("nan")


def main():
    print("=" * 80)
    print("SCENARIO D  reconfiguration cost vs. switching-cost-aware control")
    print(f"  Q4 = {os.path.basename(MODEL_Q4)}")
    print(f"  Q8 = {os.path.basename(MODEL_Q8)}")
    print(f"  ctx={CTX} np={PARALLEL}  LOW=seq({LOW_N})  BURST={BURST} concurrent  "
          f"max_tokens={MAX_TOKENS}  ngl={C.NGL}")
    print("=" * 80)

    # --- 1/2: measure config Q4, then swap to Q8 (timing the swap) ----------
    q4, srv4 = _measure_config(MODEL_Q4, "Q4_K_M")

    swap_t0 = time.time()
    teardown_s = _teardown(srv4)
    q8, srv8 = _measure_config(MODEL_Q8, "Q8_0")   # boot timing in q8["boot"]
    warmup_s = q8["warmup"]
    # C_switch = teardown old + boot new + first-token warmup new.
    cswitch = teardown_s + q8["boot"] + (warmup_s if warmup_s == warmup_s else 0.0)
    _teardown(srv8)

    rates = {"Q4": {"low": q4["low"], "burst": q4["burst"]},
             "Q8": {"low": q8["low"], "burst": q8["burst"]}}

    print("-" * 80)
    print("MEASURED token rate (tok/s):")
    print(f"{'config':>8} | {'LOW (seq)':>12} | {'BURST (np)':>12}")
    print(f"{'Q4_K_M':>8} | {rates['Q4']['low']:12.1f} | {rates['Q4']['burst']:12.1f}")
    print(f"{'Q8_0':>8} | {rates['Q8']['low']:12.1f} | {rates['Q8']['burst']:12.1f}")
    best_low = "Q4" if rates["Q4"]["low"] >= rates["Q8"]["low"] else "Q8"
    best_burst = "Q4" if rates["Q4"]["burst"] >= rates["Q8"]["burst"] else "Q8"
    print(f"  regime-optimal: LOW -> {best_low} , BURST -> {best_burst}")
    flip = best_low != best_burst
    print(f"  sign-flip across regimes: {flip} "
          f"({'switching can pay off' if flip else 'one config dominates -> dont switch'})")

    print("-" * 80)
    print("MEASURED reconfiguration cost C_switch (GGUF swap):")
    print(f"  teardown {teardown_s:5.2f}s + boot {q8['boot']:5.2f}s + warmup "
          f"{warmup_s:5.2f}s  =  C_switch {cswitch:5.2f}s")
    xover = _single_switch_breakeven(rates, cswitch)
    print(f"  single-switch break-even (one switch, ignoring the return trip): "
          f"{xover:6.2f}s")
    n_phases = int(os.environ.get("N_PHASES", "6"))
    adapt_d = _adaptation_pays_dwell(rates, cswitch, n_phases)
    print(f"  adaptation-pays dwell (cost-aware first beats the best static, "
          f"round trip included): {adapt_d:6.0f}s")

    # --- 3: dwell-time sweep ------------------------------------------------
    print("-" * 80)
    dwells = [float(x) for x in os.environ.get(
        "DWELLS", "5,15,30,60,120,300").split(",")]
    n_phases = int(os.environ.get("N_PHASES", "6"))
    print(f"TRACE REPLAY  ({n_phases} alternating LOW/BURST phases) -- goodput "
          f"(tokens served), higher is better")
    print("  static-*    = no control (one fixed config, chosen with foreknowledge)")
    print("  always-sw   = adapt every phase but IGNORE reload cost (assume C_switch=0)")
    print("  cost-aware  = control layer that COUNTS C_switch (keep-vs-reconfigure optimum)")
    hdr = (f"{'dwell(s)':>9} | {'static-Q4':>10} | {'static-Q8':>10} | "
           f"{'always-sw':>10} | {'cost-aware':>10} | winner / margin")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for d in dwells:
        rep = replay(rates, cswitch, d, n_phases)
        rows[d] = rep
        policies = {"static-Q4": rep["static-Q4"], "static-Q8": rep["static-Q8"],
                    "always-switch": rep["always-switch"], "cost-aware": rep["cost-aware"]}
        top = max(policies.values())
        winner = "cost-aware" if abs(rep["cost-aware"] - top) < 1e-6 \
            else max(policies, key=policies.get)
        runner = max(v for k, v in policies.items() if k != "cost-aware")
        margin = 100.0 * (rep["cost-aware"] - runner) / runner if runner else 0.0
        print(f"{d:9.0f} | {rep['static-Q4']:10.0f} | {rep['static-Q8']:10.0f} | "
              f"{rep['always-switch']:10.0f} | {rep['cost-aware']:10.0f} | "
              f"{winner} ({margin:+.1f}% vs next)")

    # --- verdict ------------------------------------------------------------
    print("-" * 80)
    short, long_ = rows[dwells[0]], rows[dwells[-1]]
    bs_short = max(short["static-Q4"], short["static-Q8"])
    bs_long = max(long_["static-Q4"], long_["static-Q8"])

    print("VERDICT:")
    if not flip:
        print("  No regime sign-flip measured here (one config dominates both "
              "regimes). The\n  cost-aware layer correctly keeps that config -- "
              "reconfiguration would only add\n  downtime. Re-run where the quant "
              "knob flips (8B on GPU, BURST in the batch\n  4-12 window) to see "
              "the adaptation gain.")
    else:
        print(f"  short dwell (D={dwells[0]:.0f}s, regimes flip fast): cost-aware "
              f"{short['cost-aware']:.0f} matches the\n    best static "
              f"{bs_short:.0f} (it DECLINES to switch) and beats naive always-switch "
              f"{short['always-switch']:.0f}\n    by "
              f"{100*(short['cost-aware']-short['always-switch'])/short['always-switch']:+.0f}% "
              f"-- always-switch pays reload downtime it never recoups.")
        print(f"  long  dwell (D={dwells[-1]:.0f}s, regimes durable): cost-aware "
              f"{long_['cost-aware']:.0f} beats the best static\n    {bs_long:.0f} by "
              f"{100*(long_['cost-aware']-bs_long)/bs_long:+.0f}% -- the one-time "
              f"reload now amortizes.")
        print("  => The control layer is the only policy good across BOTH regimes: "
              "it adapts when\n     a regime is durable and holds when it is not. "
              "What makes that possible is\n     COUNTING C_switch -- drop it "
              "(always-switch) and you lose at short dwell; ignore\n     adaptation "
              "(static) and you lose at long dwell.")


if __name__ == "__main__":
    main()
