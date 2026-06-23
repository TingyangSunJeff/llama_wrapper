"""Phase-timed ``llama-server`` lifecycle handle.

:class:`ServerHandle` generalizes the ``Server`` context manager from
``experiments/smoke/common.py`` into an explicitly phase-timed boot/teardown
handle for the profiling suite.

Two of the three C_switch phases are owned here (R1.1):

- **Boot** is the interval from launching ``llama-server`` to the server reporting
  ready (``/health`` returning 200). :meth:`ServerHandle.boot` measures it and
  returns a :class:`BootResult`. A boot that does not become healthy within
  ``boot_timeout_s`` (default 300 s) is reported as ``ready=False`` with an error
  reason so the caller can record a Boot-phase failure (R1.9).
- **Teardown** is the interval from issuing the shutdown signal (SIGINT) to OS
  process exit. :meth:`ServerHandle.teardown` measures it and returns the elapsed
  milliseconds.

(The third phase, Warmup / first-token, is owned by the measurement ``Client``.)

All phase timing uses a single monotonic clock (:func:`time.monotonic_ns`) so the
component timings never drift against wall-clock adjustments and reconcile to the
total within the design's 50 ms tolerance (see ``modules/switch_cost.py``).

Every boot writes a per-run log file (server stdout+stderr) at ``log_path`` so a
crash is diagnosable and the memory log lines are available to ``LogParser`` (R5.5).
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from ..config import Config

# Default GPU offload. Mirrors ``experiments/smoke/common.py``: offload everything
# to the GPU (A100 is the primary target); override with the ``NGL`` env var, set
# ``NGL=0`` to force CPU.
_DEFAULT_NGL = int(os.environ.get("NGL", "99"))

# Health-poll cadence and per-request timeout (seconds).
_POLL_INTERVAL_S = 0.5
_HEALTH_REQUEST_TIMEOUT_S = 2.0

# Grace period for a clean SIGINT shutdown before escalating to SIGKILL (seconds).
_TEARDOWN_KILL_GRACE_S = 30.0

_NS_PER_MS = 1_000_000.0


@dataclass
class BootResult:
    """Outcome of a single :meth:`ServerHandle.boot` attempt.

    Mirrors the design "Shared harness" ``server.py`` block:

    - ``ready``: ``True`` iff ``/health`` returned 200 within ``boot_timeout_s``.
    - ``boot_ms``: launch -> ``/health`` OK interval in milliseconds, measured on a
      monotonic clock. Populated even on failure (time spent before giving up).
    - ``load_time_ms``: server-reported model load time parsed from the log, if
      present; ``NaN`` when not parseable.
    - ``log_path``: absolute/related path to the per-run server log file.
    - ``error``: ``None`` on success, otherwise a human-readable failure reason
      (early process exit, or boot timeout) suitable for a Boot-phase failure
      record (R1.9).
    """

    ready: bool
    boot_ms: float
    load_time_ms: float
    log_path: str
    error: Optional[str]


class ServerHandle:
    """Launch, health-poll, and tear down a single ``llama-server`` instance.

    The handle is single-use per boot: call :meth:`boot`, run measurements against
    :attr:`base_url`, then call :meth:`teardown`. Booting launches the binary with
    the :class:`Config`'s knobs (``-m`` quant file, ``-c`` context length, ``-np``
    KV-slot count) and pins execution to ``gpu_index`` via ``CUDA_VISIBLE_DEVICES``.
    """

    def __init__(self, binary: str, config: Config, host: str, port: int,
                 gpu_index: int, log_path: str, boot_timeout_s: float = 300.0):
        self.binary = binary
        self.config = config
        self.host = host
        self.port = port
        self.gpu_index = gpu_index
        self.log_path = log_path
        self.boot_timeout_s = boot_timeout_s

        self._proc: Optional[subprocess.Popen] = None
        self._log_file = None

    # -- properties ---------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Base HTTP URL of the server, e.g. ``http://127.0.0.1:8090``."""
        return f"http://{self.host}:{self.port}"

    # -- lifecycle ----------------------------------------------------------

    def _build_command(self) -> list[str]:
        """Assemble the ``llama-server`` argv from the Config knobs.

        ``-m``/``-c``/``-np`` carry the Config knobs; ``-ngl`` controls GPU offload
        (env-overridable). ``--no-warmup`` keeps the Boot phase clean — the
        first-token warmup is a separate, explicitly measured phase (the ``Client``),
        so the server must not silently warm up during boot.
        """
        return [
            self.binary,
            "-m", self.config.quant_file,
            "-c", str(self.config.ctx_length),
            "-np", str(self.config.slot_count),
            "-ngl", str(_DEFAULT_NGL),
            "--host", self.host,
            "--port", str(self.port),
            "--no-warmup",
        ]

    def _child_env(self) -> dict:
        """Process environment with the GPU pinned via ``CUDA_VISIBLE_DEVICES`` (R5.3)."""
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        return env

    def boot(self) -> BootResult:
        """Launch the server and time launch -> ``/health`` OK (the Boot phase).

        Returns a :class:`BootResult`. On success ``ready=True`` and ``error=None``.
        If the process exits early or ``/health`` does not return 200 within
        ``boot_timeout_s``, returns ``ready=False`` with an ``error`` reason and the
        elapsed ``boot_ms`` (R1.9). The server log is written to ``log_path``.
        """
        cmd = self._build_command()
        self._log_file = open(self.log_path, "w")

        t_launch = time.monotonic_ns()
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=self._child_env(),
        )

        ready, error = self._wait_ready(t_launch)
        boot_ms = (time.monotonic_ns() - t_launch) / _NS_PER_MS

        # Best-effort: the elapsed boot time is authoritative; the server-reported
        # load time is parsed from the log when available, else NaN.
        load_time_ms = self._parse_load_time_ms()

        return BootResult(
            ready=ready,
            boot_ms=boot_ms,
            load_time_ms=load_time_ms,
            log_path=self.log_path,
            error=error,
        )

    def _wait_ready(self, t_launch: int) -> tuple[bool, Optional[str]]:
        """Poll ``/health`` until 200, the process dies, or the timeout elapses.

        Returns ``(ready, error)``: ``(True, None)`` on a healthy 200, otherwise
        ``(False, reason)``. Timing is measured against the monotonic launch instant.
        """
        deadline_ns = t_launch + int(self.boot_timeout_s * 1e9)
        health_url = f"{self.base_url}/health"

        while time.monotonic_ns() < deadline_ns:
            # Surface an early crash immediately rather than waiting out the timeout.
            if self._proc is not None and self._proc.poll() is not None:
                return False, (
                    f"server exited early (code {self._proc.returncode}); "
                    f"see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(
                    health_url, timeout=_HEALTH_REQUEST_TIMEOUT_S
                ) as resp:
                    if resp.status == 200:
                        return True, None
            except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
                # Server not accepting connections yet; keep polling.
                pass
            time.sleep(_POLL_INTERVAL_S)

        return False, (
            f"server did not report ready within boot timeout "
            f"({self.boot_timeout_s:g}s); see {self.log_path}"
        )

    def _parse_load_time_ms(self) -> float:
        """Parse a server-reported model load time (ms) from the log, else NaN.

        Best-effort: matches a ``load time = <N> ms`` style line if ``llama-server``
        emitted one. The measured ``boot_ms`` remains the authoritative Boot timing.
        """
        try:
            with open(self.log_path, "r", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            return float("nan")

        import re

        # e.g. "load time =    1234.56 ms" / "model load time = 1234 ms"
        match = re.search(r"load time\s*=\s*([\d.]+)\s*ms", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return float("nan")
        return float("nan")

    def teardown(self) -> float:
        """Send SIGINT and wait for exit; return the Teardown interval in ms.

        Measures from the instant the SIGINT is issued to the current process to OS
        process exit (``proc.wait()`` returning), on the monotonic clock. If the
        process does not exit within the grace period it is escalated to SIGKILL;
        the returned interval still spans signal -> actual exit. Returns ``0.0`` if
        there is no live process to tear down.
        """
        proc = self._proc

        if proc is None or proc.poll() is not None:
            # Nothing live to tear down; close the log and report no interval.
            self._close_log()
            return 0.0

        t_signal = time.monotonic_ns()
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=_TEARDOWN_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        teardown_ms = (time.monotonic_ns() - t_signal) / _NS_PER_MS

        self._close_log()
        return teardown_ms

    def _close_log(self) -> None:
        """Close the per-run log file handle, ignoring errors."""
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            finally:
                self._log_file = None


__all__ = ["BootResult", "ServerHandle"]
