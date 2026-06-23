"""Serving configuration model and change-type labeling.

A :class:`Config` is one serving configuration: the tuple of the quant/model GGUF
file, the context length (``-c``), and the number of parallel KV slots (``-np``)
that defines a single ``llama-server`` instance.

The :meth:`Config.change_type` method labels the reconfiguration between two
Configs purely from the deltas of their fields (R1.3-R1.5):

- ``slot-reshape``  iff only ``ctx_length`` and/or ``slot_count`` differ
- ``model-reload``  iff only ``quant_file`` differs
- ``combined``      iff ``quant_file`` differs AND at least one of
  ``ctx_length`` / ``slot_count`` differs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChangeType = Literal["slot-reshape", "model-reload", "combined"]


@dataclass(frozen=True)
class Config:
    """A serving configuration: {quant/model file, context length, KV-slot count}."""

    quant_file: str
    ctx_length: int
    slot_count: int

    def change_type(self, other: "Config") -> ChangeType:
        """Label the reconfiguration from ``self`` to ``other`` by field deltas.

        Returns one of ``"slot-reshape"``, ``"model-reload"``, or ``"combined"``,
        derived purely from which fields differ between the two Configs.

        The two Configs are assumed to differ in at least one field (a real
        Config change). If they are identical there is no change to classify;
        a :class:`ValueError` is raised to surface the misuse rather than
        silently mislabeling it.
        """
        model_changed = self.quant_file != other.quant_file
        shape_changed = (
            self.ctx_length != other.ctx_length
            or self.slot_count != other.slot_count
        )

        if model_changed and shape_changed:
            return "combined"
        if model_changed:
            return "model-reload"
        if shape_changed:
            return "slot-reshape"

        raise ValueError(
            "change_type requires two differing Configs; "
            f"both Configs are identical: {self!r}"
        )


@dataclass(frozen=True)
class GridSpec:
    """Specification of a Config_Grid as the axes of a cross-product.

    Holds the three swept Knob axes:

    - ``quant_files``  — quant/model GGUF files
    - ``ctx_lengths``  — context lengths (``-c``)
    - ``slot_counts``  — parallel KV-slot counts (``-np``)

    Inputs may be supplied as any iterable (list, tuple, ...); each axis is
    normalized to a tuple so the spec is hashable and the expansion order is
    fixed. The cross-product is produced by :func:`expand_grid`.
    """

    quant_files: tuple[str, ...] = field(default=())
    ctx_lengths: tuple[int, ...] = field(default=())
    slot_counts: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        # Coerce each axis to a tuple so a frozen GridSpec is hashable and the
        # iteration order is stable regardless of the input container type.
        object.__setattr__(self, "quant_files", tuple(self.quant_files))
        object.__setattr__(self, "ctx_lengths", tuple(self.ctx_lengths))
        object.__setattr__(self, "slot_counts", tuple(self.slot_counts))


def expand_grid(spec: GridSpec) -> list[Config]:
    """Expand a :class:`GridSpec` into its Config_Grid (R1.6, R2.1, R8.2).

    Produces the cross-product of ``quant_files`` x ``ctx_lengths`` x
    ``slot_counts`` as a ``list[Config]`` in a stable, deterministic order via
    nested iteration (quant_files outermost, slot_counts innermost). The result
    is duplicate-free: duplicate values within any axis, or a Config that would
    otherwise be produced more than once, appear exactly once while preserving
    first-seen order.
    """
    seen: set[Config] = set()
    grid: list[Config] = []
    for quant_file in spec.quant_files:
        for ctx_length in spec.ctx_lengths:
            for slot_count in spec.slot_counts:
                cfg = Config(
                    quant_file=quant_file,
                    ctx_length=ctx_length,
                    slot_count=slot_count,
                )
                if cfg in seen:
                    continue
                seen.add(cfg)
                grid.append(cfg)
    return grid
