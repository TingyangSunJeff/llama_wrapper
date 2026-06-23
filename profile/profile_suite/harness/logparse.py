"""Parse memory-footprint lines from ``llama-server`` stdout (R3.2).

At startup ``llama-server`` logs the sizes of the backend buffers it allocates.
This module extracts the three categories the Memory_Profiler needs and sums
each across every backend (e.g. one line per CUDA device / CPU host buffer):

- **weights**  - the model-weights buffers, logged by ``load_tensors`` as
  ``<backend> model buffer size = X MiB`` (see ``src/llama-model.cpp``).
- **KV**       - the KV-cache buffers, logged by ``llama_kv_cache`` as
  ``<backend> KV buffer size = X MiB`` (see ``src/llama-kv-cache.cpp``).
- **compute**  - the compute/scratch buffers, logged by ``llama_context`` /
  ``llama_new_context_with_model`` as ``<backend> compute buffer size = X MiB``
  (see ``src/llama-context.cpp``).

Each category is the sum of *all* matching lines, so multi-device runs (one
buffer per GPU plus a host buffer) are accounted for in full. A category with
no matching line is reported as ``None`` (distinct from a measured ``0.0``).

Other ``* buffer size`` lines that ``llama-server`` may emit - ``output
buffer size``, ``RS buffer size``, ``LoRA buffer size`` - are intentionally
**not** matched: the three regexes below key on the exact category phrase.

The :class:`ServerMemoryReport` produced here is duck-typed by the pure
``profile_suite.modules.memory.decompose`` logic, which reads the
``weights_mib`` / ``kv_mib`` / ``compute_mib`` attributes.

See design.md "Shared harness" -> ``harness/logparse.py`` and "Memory
decomposition (R3.2, R3.3)".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern matches the category phrase followed by ``= <number> MiB``. The
# leading backend label (``CUDA0``, ``CPU``, ``HTP0-REPACK``, ...) is variable
# and irrelevant to the sum, so it is not captured. ``KV`` is matched
# case-sensitively via the literal phrase; the ``model``/``compute`` phrases are
# lowercase as emitted by the engine. Keying on the full phrase keeps
# ``output``/``RS``/``LoRA`` buffer lines from being mistaken for these.
_NUMBER = r"([-+]?\d+(?:\.\d+)?)"
_WEIGHTS_RE = re.compile(r"\bmodel buffer size\s*=\s*" + _NUMBER + r"\s*MiB")
_KV_RE = re.compile(r"\bKV buffer size\s*=\s*" + _NUMBER + r"\s*MiB")
_COMPUTE_RE = re.compile(r"\bcompute buffer size\s*=\s*" + _NUMBER + r"\s*MiB")


@dataclass(frozen=True)
class ServerMemoryReport:
    """Summed buffer sizes parsed from a ``llama-server`` log, in MiB (R3.2).

    Each field is the sum of every matching ``* buffer size = X MiB`` line for
    its category, or ``None`` when the log contains no such line. The field
    names match the attributes read by
    :func:`profile_suite.modules.memory.decompose`.
    """

    weights_mib: float | None  # sum of "* model buffer size = X MiB" lines
    kv_mib: float | None  # sum of "* KV buffer size = X MiB" lines
    compute_mib: float | None  # sum of "* compute buffer size = X MiB" lines


class LogParser:
    """Extract summed memory-buffer sizes from ``llama-server`` stdout."""

    def parse_memory(self, log_path: str) -> ServerMemoryReport:
        """Parse a server log file into a :class:`ServerMemoryReport`.

        Args:
            log_path: Path to the captured ``llama-server`` stdout/stderr log.

        Returns:
            A :class:`ServerMemoryReport` whose ``weights_mib`` / ``kv_mib`` /
            ``compute_mib`` are the summed MiB values for each category, or
            ``None`` for any category with no matching line.
        """
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            return self.parse_memory_text(handle.read())

    def parse_memory_text(self, text: str) -> ServerMemoryReport:
        """Parse already-read log text into a :class:`ServerMemoryReport`.

        Splitting this out from :meth:`parse_memory` lets callers parse a log
        held in memory (e.g. a streamed boot capture) without a round-trip to
        disk, and keeps the parsing logic directly unit-testable.

        Args:
            text: The full ``llama-server`` log contents.

        Returns:
            A :class:`ServerMemoryReport` summarising the buffer-size lines.
        """
        return ServerMemoryReport(
            weights_mib=_sum_matches(_WEIGHTS_RE, text),
            kv_mib=_sum_matches(_KV_RE, text),
            compute_mib=_sum_matches(_COMPUTE_RE, text),
        )


def _sum_matches(pattern: re.Pattern[str], text: str) -> float | None:
    """Sum every numeric match of ``pattern`` in ``text``.

    Returns ``None`` when there is no match, so an absent category is
    distinguishable from a category that legitimately summed to ``0.0``.
    """
    values = [float(m) for m in pattern.findall(text)]
    if not values:
        return None
    return sum(values)
