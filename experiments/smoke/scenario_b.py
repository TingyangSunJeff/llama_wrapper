"""Scenario B: Reshaping Context Length (Document Unlock).

Knob: context length (-c), 2048 vs 32768.

Setup: a long "technical manual" (~28K tokens of filler) with a unique secret
passcode buried in the middle. We ask the model to report the passcode.

Hypothesis:
  - ctx=2048  : the prompt is far larger than the window -> the server either
                rejects the request or truncates away the needle -> WRONG answer.
  - ctx=32768 : the whole document fits -> the model can read the needle -> RIGHT.

This is a *feasibility* gap, not a speed gap: the small-context config is not
slow, it is simply unable to do the task. That alone justifies adapting ctx.
No controller is involved.
"""

import asyncio

import common as C


NEEDLE_CODE = "MAGENTA-7731"
QUESTION    = (f"You are reading a technical manual. Somewhere in it there is a "
              f"line that states the secret passcode. Report ONLY the secret "
              f"passcode, nothing else.")
TARGET_TOKENS = 28000   # comfortably > 2048 and < 32768


FILLER = (
    "The maintenance subsystem performs periodic diagnostics on the actuator "
    "array and logs telemetry to the onboard buffer for later review. ")
NEEDLE = f"\n\nIMPORTANT: The secret passcode is {NEEDLE_CODE}. Remember it.\n\n"


async def build_sized_prompt(session, base, target_tokens):
    """Build a prompt of ~target_tokens of filler with the needle centered.

    We measure filler token density on the server, scale the filler to the
    target, then insert the needle in the middle so a left/right truncation
    would drop it.
    """
    sample = FILLER * 100
    sample_tokens = await C.tokenize(session, base, sample)
    chars_per_token = len(sample) / max(sample_tokens, 1)
    target_chars = int(target_tokens * chars_per_token)

    filler = FILLER * (target_chars // len(FILLER) + 1)
    filler = filler[:target_chars]
    mid = len(filler) // 2
    document = filler[:mid] + NEEDLE + filler[mid:]
    prompt = f"{document}\n\n{QUESTION}"

    ntok = await C.tokenize(session, base, prompt)
    return prompt, ntok


async def measure(base, ctx_label, target_tokens):
    async with C.aiohttp.ClientSession() as s:
        prompt, ntok = await build_sized_prompt(s, base, target_tokens)
        r = await C.chat(s, base, prompt, max_tokens=24, tag=ctx_label)
    r["prompt_tokens"] = ntok
    r["found"] = (NEEDLE_CODE in r["text"])
    return r


def eval_config(ctx, label):
    with C.Server(model=C.MODEL_Q4, ctx=ctx, parallel=1, label=label,
                  extra=["--no-context-shift"]) as srv:
        res = asyncio.run(measure(srv.base, label, TARGET_TOKENS))
    return res


def main():
    print("=" * 70)
    print(f"SCENARIO B  needle={NEEDLE_CODE}  target_prompt~{TARGET_TOKENS} tokens")
    print("=" * 70)

    small = eval_config(2048, "ctx=2K")
    big   = eval_config(32768, "ctx=32K")

    def row(name, r):
        status = "OK" if r["ok"] else "FAIL"
        ans = r["text"].strip().replace("\n", " ")[:40]
        print(f"{name:>8} | {status:>4} | prompt_tok {r['prompt_tokens']:6d} "
              f"| needle_found={str(r['found']):>5} | latency {r['latency']:6.2f}s "
              f"| answer: {ans!r}")
        if not r["ok"]:
            print(f"         | error: {r['error']}")

    print("-" * 70)
    row("ctx=2K", small)
    row("ctx=32K", big)
    print("-" * 70)

    unlocked = (not small["found"]) and big["found"]
    print(f"RESULT: 2K finds needle={small['found']}, 32K finds needle={big['found']}")
    print(f"VERDICT: {'GAP CONFIRMED' if unlocked else 'no gap'} -- the 32K config "
          f"{'unlocks a task the 2K config cannot do' if unlocked else 'did not differ'}")


if __name__ == "__main__":
    main()
