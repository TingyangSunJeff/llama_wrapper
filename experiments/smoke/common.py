"""Shared harness for the knob-adaptation smoke tests.

These tests do NOT implement any adaptive policy. Each one stands up two
*static* llama-server configurations and shows that the config which is wrong
for a given scenario loses badly, while the matching config wins. That gap is
the whole motivation for runtime knob adaptation.

Knobs exercised (one per scenario, matching notes.md sec. "Performance
Scenarios: Coarse-Grained Configuration Control"):
  Scenario A: model / quantization file   (Q8_0 vs Q4_K_M)   -> throughput burst
  Scenario B: context length              (2K vs 32K)        -> document unlock
  Scenario C: parallel slots              (np=1 vs np=4)     -> anti-blocking
"""

import os
import sys
import time
import socket
import signal
import subprocess
import urllib.request
import urllib.error
import asyncio

# ---------------------------------------------------------------------------
# Configuration (override via environment if needed)
# ---------------------------------------------------------------------------
REPO       = "/scratch2/tingyang/llama.cpp"
SERVER_BIN = os.environ.get("LLAMA_SERVER", f"{REPO}/build-cuda/bin/llama-server")
MODEL_DIR  = os.environ.get("LLAMA_MODEL_DIR", f"{REPO}/models")

MODEL_Q4 = f"{MODEL_DIR}/gemma-3-1b-it-Q4_K_M.gguf"
MODEL_Q8 = f"{MODEL_DIR}/gemma-3-1b-it-Q8_0.gguf"

HOST = "127.0.0.1"
# Offload everything to GPU; A100 is idle. Set NGL=0 to force CPU.
NGL = int(os.environ.get("NGL", "99"))


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) != 0


def _pick_port(start: int = 8090) -> int:
    for p in range(start, start + 50):
        if _port_is_free(p):
            return p
    raise RuntimeError("no free port found")


class Server:
    """Context manager that boots a static llama-server and tears it down."""

    def __init__(self, model, ctx, parallel, port=None, ngl=NGL, extra=None, label=""):
        self.model    = model
        self.ctx      = ctx
        self.parallel = parallel
        self.ngl      = ngl
        self.extra    = extra or []
        self.label    = label or os.path.basename(model)
        self.port     = port or _pick_port()
        self.proc     = None
        self.base     = f"http://{HOST}:{self.port}"

    def __enter__(self):
        cmd = [
            SERVER_BIN,
            "-m", self.model,
            "-c", str(self.ctx),
            "-np", str(self.parallel),
            "-ngl", str(self.ngl),
            "--host", HOST,
            "--port", str(self.port),
            "--no-warmup",
        ] + self.extra
        print(f"[server:{self.label}] start  ctx={self.ctx} np={self.parallel} "
              f"ngl={self.ngl} port={self.port}", flush=True)
        t0 = time.time()
        # Capture logs so a crash is visible; keep them out of our stdout.
        self.log = open(f"/tmp/smoke_server_{self.port}.log", "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self._wait_ready()
        self.load_time = time.time() - t0
        print(f"[server:{self.label}] ready in {self.load_time:.2f}s", flush=True)
        return self

    def _wait_ready(self, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (code {self.proc.returncode}); "
                    f"see /tmp/smoke_server_{self.port}.log")
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                pass
            time.sleep(0.5)
        raise RuntimeError("server did not become healthy in time")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self.log.close()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Async client (streaming, so we can measure time-to-first-token)
# ---------------------------------------------------------------------------
import aiohttp
import json


async def chat(session, base, prompt, max_tokens, temperature=0.0, tag=""):
    """Send one streaming chat completion. Returns timing + text dict."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "cache_prompt": False,
    }
    submit = time.time()
    ttft = None
    text_parts = []
    n_tokens = 0
    error = None
    try:
        async with session.post(f"{base}/v1/chat/completions",
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=600)) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {"tag": tag, "ok": False, "error": f"HTTP {resp.status}: {body[:200]}",
                        "submit": submit, "ttft": None, "latency": time.time() - submit,
                        "tokens": 0, "text": ""}
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                chunk = delta.get("content")
                if chunk:
                    if ttft is None:
                        ttft = time.time() - submit
                    text_parts.append(chunk)
                    n_tokens += 1
    except Exception as e:  # noqa: BLE001 - smoke test, surface anything
        error = repr(e)
    latency = time.time() - submit
    return {
        "tag": tag,
        "ok": error is None,
        "error": error,
        "submit": submit,
        "ttft": ttft,
        "latency": latency,
        "tokens": n_tokens,
        "text": "".join(text_parts),
    }


async def tokenize(session, base, content):
    """Use the server /tokenize endpoint to count tokens precisely."""
    async with session.post(f"{base}/tokenize", json={"content": content}) as resp:
        obj = await resp.json()
        return len(obj.get("tokens", []))


def gpu_mem_used_mb(index=0):
    """Return MiB used on a given GPU via nvidia-smi (NaN if unavailable)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, timeout=5)
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(results):
    ok = [r for r in results if r["ok"]]
    lat = [r["latency"] for r in ok]
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    return {
        "n": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "lat_mean": (sum(lat) / len(lat)) if lat else float("nan"),
        "lat_p95": pct(lat, 95),
        "ttft_mean": (sum(ttft) / len(ttft)) if ttft else float("nan"),
        "ttft_p95": pct(ttft, 95),
    }
