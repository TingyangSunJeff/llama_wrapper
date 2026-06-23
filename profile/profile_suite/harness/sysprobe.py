"""Device-memory probing and environment capture for the profile-suite.

This module generalizes the single ``gpu_mem_used_mb()`` helper from
``experiments/smoke/common.py`` into a small :class:`SysProbe` facility that the
measurement harness uses to:

- sample per-GPU used / total device memory via ``nvidia-smi`` (R3.1),
- run a background peak sampler over a measurement interval (R3.1),
- capture the reproducibility environment (OS / GPU model / driver / CUDA /
  Python version, plus per-model checksums) recording the explicit
  :data:`~profile_suite.results.UNAVAILABLE` sentinel for any field that cannot
  be determined (R5.2, R5.8), and
- compute a sha256 content checksum for a resolved model file (R5.2).

All ``nvidia-smi`` access is best-effort: every probe degrades gracefully (to
``float("nan")`` for numeric samples or :data:`UNAVAILABLE` for environment
fields) rather than raising, so a campaign can continue on hosts where a field
is unavailable (R5.8).

See design.md "Shared harness" -> ``harness/sysprobe.py``::

    class SysProbe:
        def device_memory_used_mib(self, gpu_index: int) -> float: ...
        def device_memory_total_mib(self, gpu_index: int) -> float: ...
        def sample_peak(self, gpu_index: int, stop: threading.Event) -> float: ...
        def capture_environment(self) -> "EnvironmentCapture": ...
        def file_checksum(self, path: str) -> str: ...
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import threading

from ..results import UNAVAILABLE, EnvironmentCapture, ModelRef

# Cadence of the background peak sampler. The design specifies a roughly 50 ms
# polling interval so a short measurement workload is sampled densely enough to
# catch its true peak without saturating ``nvidia-smi``.
_PEAK_SAMPLE_INTERVAL_S: float = 0.05

# Default timeout (seconds) for any single ``nvidia-smi`` invocation.
_NVIDIA_SMI_TIMEOUT_S: float = 5.0


class SysProbe:
    """Best-effort device-memory probing and environment capture.

    The probe holds no mutable state; instances are cheap and reusable across a
    campaign. Every method is safe to call on a host without ``nvidia-smi`` or a
    CUDA driver: numeric probes return ``float("nan")`` and environment fields
    fall back to the :data:`UNAVAILABLE` sentinel.
    """

    # ------------------------------------------------------------------ #
    # Device memory samples
    # ------------------------------------------------------------------ #
    def device_memory_used_mib(self, gpu_index: int = 0) -> float:
        """Return MiB currently used on ``gpu_index`` (``nan`` if unavailable).

        Generalizes ``common.py`` ``gpu_mem_used_mb()`` to an explicit GPU index.
        """
        return self._query_gpu_float("memory.used", gpu_index)

    def device_memory_total_mib(self, gpu_index: int = 0) -> float:
        """Return total MiB installed on ``gpu_index`` (``nan`` if unavailable)."""
        return self._query_gpu_float("memory.total", gpu_index)

    def sample_peak(self, gpu_index: int, stop: threading.Event) -> float:
        """Sample used device memory until ``stop`` is set; return the peak MiB.

        Intended to run on a background thread for the duration of a measurement
        workload (R3.1): the caller boots a Config, starts this sampler, runs the
        workload, then sets ``stop`` and joins to obtain the observed peak. The
        loop polls roughly every 50 ms and ignores any individual ``nan`` sample
        so a transient ``nvidia-smi`` hiccup does not corrupt the peak.

        Returns ``float("nan")`` if no valid sample was ever obtained.
        """
        peak = float("nan")
        # Take at least one sample before checking the stop event so that very
        # short workloads still record a value.
        while True:
            used = self.device_memory_used_mib(gpu_index)
            if used == used:  # not NaN
                if peak != peak or used > peak:
                    peak = used
            if stop.is_set():
                break
            # Wait returns early (True) if the event is set during the interval,
            # which lets us stop promptly without an extra full-interval sleep.
            if stop.wait(_PEAK_SAMPLE_INTERVAL_S):
                # Take one final sample after being asked to stop to catch a peak
                # that occurred right at the end of the workload.
                used = self.device_memory_used_mib(gpu_index)
                if used == used and (peak != peak or used > peak):
                    peak = used
                break
        return peak

    # ------------------------------------------------------------------ #
    # Environment capture
    # ------------------------------------------------------------------ #
    def capture_environment(self) -> EnvironmentCapture:
        """Capture the reproducibility environment (R5.2, R5.8).

        Fills operating system, GPU model, driver version, CUDA version, and
        Python version. Any field that cannot be determined is recorded as the
        explicit :data:`UNAVAILABLE` sentinel so the campaign can continue. The
        per-model checksum map is populated separately by the caller via
        :meth:`file_checksum` / :meth:`model_ref`.
        """
        return EnvironmentCapture(
            os=self._capture_os(),
            gpu_model=self._capture_gpu_model(),
            driver_version=self._capture_driver_version(),
            cuda_version=self._capture_cuda_version(),
            python_version=self._capture_python_version(),
            models={},
        )

    def model_ref(self, path: str) -> ModelRef:
        """Build a :class:`ModelRef` (absolute path + sha256) for a model file."""
        import os

        return ModelRef(abs_path=os.path.abspath(path), sha256=self.file_checksum(path))

    def file_checksum(self, path: str) -> str:
        """Return the sha256 hex digest of the file at ``path``.

        Reads the file in chunks so large model files do not need to be loaded
        into memory at once. Returns :data:`UNAVAILABLE` if the file cannot be
        read (R5.8), keeping the harness resilient.
        """
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return UNAVAILABLE

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _capture_os(self) -> str:
        try:
            system = platform.system()
            release = platform.release()
            value = f"{system} {release}".strip()
            return value or UNAVAILABLE
        except Exception:  # noqa: BLE001 - capture must never abort the campaign
            return UNAVAILABLE

    def _capture_python_version(self) -> str:
        try:
            version = platform.python_version()
            return version or UNAVAILABLE
        except Exception:  # noqa: BLE001
            # Fall back to the raw interpreter version string.
            try:
                return sys.version.split()[0]
            except Exception:  # noqa: BLE001
                return UNAVAILABLE

    def _capture_gpu_model(self) -> str:
        value = self._query_gpu_str("name", 0)
        return value if value is not None else UNAVAILABLE

    def _capture_driver_version(self) -> str:
        value = self._query_gpu_str("driver_version", 0)
        return value if value is not None else UNAVAILABLE

    def _capture_cuda_version(self) -> str:
        """Determine the CUDA version reported by ``nvidia-smi``.

        ``nvidia-smi`` does not expose CUDA version through ``--query-gpu``, so we
        parse the human-readable header of plain ``nvidia-smi`` output, which
        contains a ``CUDA Version: <x.y>`` token.
        """
        try:
            out = subprocess.check_output(
                ["nvidia-smi"], text=True, timeout=_NVIDIA_SMI_TIMEOUT_S
            )
        except Exception:  # noqa: BLE001 - missing tool / driver
            return UNAVAILABLE
        marker = "CUDA Version:"
        for line in out.splitlines():
            idx = line.find(marker)
            if idx != -1:
                tail = line[idx + len(marker):].strip()
                token = tail.split()[0] if tail else ""
                # Strip any trailing table border characters.
                token = token.strip("|").strip()
                if token:
                    return token
        return UNAVAILABLE

    def _query_gpu_float(self, field: str, gpu_index: int) -> float:
        """Query a numeric ``--query-gpu`` field; return ``nan`` on any failure."""
        value = self._query_gpu_str(field, gpu_index)
        if value is None:
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return float("nan")

    def _query_gpu_str(self, field: str, gpu_index: int) -> str | None:
        """Query a single ``--query-gpu`` field for one GPU.

        Returns the stripped first-line value, or ``None`` if ``nvidia-smi`` is
        unavailable, times out, or produces no output.
        """
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu={field}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(gpu_index),
                ],
                text=True,
                timeout=_NVIDIA_SMI_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 - missing tool / bad index / timeout
            return None
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        value = lines[0]
        return value or None


__all__ = ["SysProbe", "_PEAK_SAMPLE_INTERVAL_S"]
