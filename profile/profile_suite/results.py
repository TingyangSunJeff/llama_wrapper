"""Result and manifest data models with JSON round-trip serialization.

This module defines the report-facing and persistence data models for the
profile-suite, plus ``to_dict``/``from_dict`` helpers that JSON-round-trip each
of them (and the :class:`~profile_suite.config.Config` they reference).

Models (see design.md "Data Models", "Run_Manifest", and the
"Reproducibility_Harness run loop"):

- :class:`Aggregate`          - mean/std over successful repeats + insufficiency flag
- :class:`ModelRef`           - a resolved model file: absolute path + sha256
- :class:`EnvironmentCapture` - captured environment; any field may be the explicit
                                 ``"unavailable"`` sentinel (R5.8)
- :class:`RunRepeatResult`    - one raw run (warmup or retained repeat)
- :class:`MeasurementPoint`   - one module x Config x axis unit with repeats + aggregates
- :class:`MeasuredValue`      - the report-facing measured record (R7.1)
- :class:`RunManifest`        - the machine-readable campaign record (R5.5, R5.6)

Serialization design:
    Every model exposes ``to_dict()`` returning a JSON-serializable ``dict`` and a
    ``from_dict(d)`` classmethod that reconstructs an equal instance, so that
    ``Model.from_dict(m.to_dict()) == m`` for any value (Property 12). The
    referenced :class:`Config` is serialized via the module-level
    :func:`config_to_dict` / :func:`config_from_dict` helpers so manifests can
    round-trip the swept Config_Grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .config import Config

# Explicit sentinel recorded for an environment field that cannot be determined
# when the environment is captured (R5.8). It is a normal string value so it
# round-trips through JSON unchanged.
UNAVAILABLE: str = "unavailable"

PointStatus = Literal["complete", "incomplete", "failed", "platform-infeasible"]


# --------------------------------------------------------------------------- #
# Config serialization helpers
# --------------------------------------------------------------------------- #
def config_to_dict(config: Config) -> dict[str, Any]:
    """Serialize a :class:`Config` to a JSON-serializable dict."""
    return {
        "quant_file": config.quant_file,
        "ctx_length": config.ctx_length,
        "slot_count": config.slot_count,
    }


def config_from_dict(d: dict[str, Any]) -> Config:
    """Reconstruct a :class:`Config` from a dict produced by :func:`config_to_dict`."""
    return Config(
        quant_file=d["quant_file"],
        ctx_length=int(d["ctx_length"]),
        slot_count=int(d["slot_count"]),
    )


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass
class Aggregate:
    """Mean/std over the successful repeats only, with an insufficiency flag.

    ``insufficient`` is ``True`` iff ``n_success < 5`` (R1.8, R4.7).
    """

    mean: float
    std: float
    n_success: int
    insufficient: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "n_success": self.n_success,
            "insufficient": self.insufficient,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Aggregate":
        return cls(
            mean=float(d["mean"]),
            std=float(d["std"]),
            n_success=int(d["n_success"]),
            insufficient=bool(d["insufficient"]),
        )


# --------------------------------------------------------------------------- #
# ModelRef
# --------------------------------------------------------------------------- #
@dataclass
class ModelRef:
    """A resolved model file: its absolute path and content checksum (R5.2)."""

    abs_path: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"abs_path": self.abs_path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelRef":
        return cls(abs_path=d["abs_path"], sha256=d["sha256"])


# --------------------------------------------------------------------------- #
# EnvironmentCapture
# --------------------------------------------------------------------------- #
@dataclass
class EnvironmentCapture:
    """Captured environment fields for reproducibility (R5.2, R5.8).

    Any scalar field may hold a normal value, ``None``, or the explicit
    :data:`UNAVAILABLE` sentinel when it could not be determined.
    """

    os: str | None
    gpu_model: str | None
    driver_version: str | None
    cuda_version: str | None
    python_version: str | None
    models: dict[str, ModelRef] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": self.os,
            "gpu_model": self.gpu_model,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "python_version": self.python_version,
            "models": {path: ref.to_dict() for path, ref in self.models.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvironmentCapture":
        return cls(
            os=d.get("os"),
            gpu_model=d.get("gpu_model"),
            driver_version=d.get("driver_version"),
            cuda_version=d.get("cuda_version"),
            python_version=d.get("python_version"),
            models={
                path: ModelRef.from_dict(ref)
                for path, ref in (d.get("models") or {}).items()
            },
        )


# --------------------------------------------------------------------------- #
# RunRepeatResult
# --------------------------------------------------------------------------- #
@dataclass
class RunRepeatResult:
    """One raw run: the discarded warmup or a retained Run_Repeat.

    ``metrics`` holds the per-run measured values (e.g. ``c_switch_ms``); ``error``
    carries the failure reason when ``ok`` is ``False``.
    """

    run_index: int
    discarded_warmup: bool
    ok: bool
    raw_log_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "discarded_warmup": self.discarded_warmup,
            "ok": self.ok,
            "raw_log_path": self.raw_log_path,
            "metrics": dict(self.metrics),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRepeatResult":
        return cls(
            run_index=int(d["run_index"]),
            discarded_warmup=bool(d["discarded_warmup"]),
            ok=bool(d["ok"]),
            raw_log_path=d["raw_log_path"],
            metrics={k: float(v) for k, v in (d.get("metrics") or {}).items()},
            error=d.get("error"),
        )


# --------------------------------------------------------------------------- #
# MeasurementPoint
# --------------------------------------------------------------------------- #
@dataclass
class MeasurementPoint:
    """The atomic measurement unit: one module x one Config x one axis value.

    Holds its raw :class:`RunRepeatResult`s and an :class:`Aggregate` per metric
    (computed over successful repeats only).
    """

    point_id: str
    module: str
    config: Config
    axis: dict[str, Any] = field(default_factory=dict)
    repeats: list[RunRepeatResult] = field(default_factory=list)
    aggregates: dict[str, Aggregate] = field(default_factory=dict)
    status: PointStatus = "complete"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "module": self.module,
            "config": config_to_dict(self.config),
            "axis": dict(self.axis),
            "repeats": [r.to_dict() for r in self.repeats],
            "aggregates": {k: agg.to_dict() for k, agg in self.aggregates.items()},
            "status": self.status,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasurementPoint":
        return cls(
            point_id=d["point_id"],
            module=d["module"],
            config=config_from_dict(d["config"]),
            axis=dict(d.get("axis") or {}),
            repeats=[RunRepeatResult.from_dict(r) for r in (d.get("repeats") or [])],
            aggregates={
                k: Aggregate.from_dict(v)
                for k, v in (d.get("aggregates") or {}).items()
            },
            status=d.get("status", "complete"),
            reason=d.get("reason"),
        )


# --------------------------------------------------------------------------- #
# MeasuredValue
# --------------------------------------------------------------------------- #
@dataclass
class MeasuredValue:
    """The report-facing measured record (R7.1).

    Carries the metric name, unit, mean, std, success count, any flags, the source
    Config (already serialized as a dict), the Platform, the axis, and a reference
    to the source Run_Manifest.
    """

    metric: str
    unit: str
    mean: float
    std: float
    n_success: int
    config: dict[str, Any]
    platform: str
    manifest_ref: str
    flags: list[str] = field(default_factory=list)
    axis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "mean": self.mean,
            "std": self.std,
            "n_success": self.n_success,
            "flags": list(self.flags),
            "config": dict(self.config),
            "platform": self.platform,
            "axis": dict(self.axis),
            "manifest_ref": self.manifest_ref,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeasuredValue":
        return cls(
            metric=d["metric"],
            unit=d["unit"],
            mean=float(d["mean"]),
            std=float(d["std"]),
            n_success=int(d["n_success"]),
            config=dict(d.get("config") or {}),
            platform=d["platform"],
            manifest_ref=d["manifest_ref"],
            flags=list(d.get("flags") or []),
            axis=dict(d.get("axis") or {}),
        )


# --------------------------------------------------------------------------- #
# RunManifest
# --------------------------------------------------------------------------- #
@dataclass
class RunManifest:
    """The machine-readable record of one campaign (R5.5, R5.6).

    Links the Pinned_Build, Platform descriptor, pinned device identifier, the
    environment capture, the swept Config_Grid, the Run_Repeat count, and the
    retained raw-log paths.
    """

    campaign_name: str
    pinned_build: str
    platform: str
    pinned_device: str
    environment: EnvironmentCapture
    config_grid: list[Config] = field(default_factory=list)
    run_repeats: int = 5
    raw_log_paths: dict[str, list[str]] = field(default_factory=dict)
    decode_batch_sizes: list[int] = field(default_factory=list)
    enabled_modules: list[str] = field(default_factory=list)
    noise_sensitive_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_name": self.campaign_name,
            "pinned_build": self.pinned_build,
            "platform": self.platform,
            "pinned_device": self.pinned_device,
            "environment": self.environment.to_dict(),
            "config_grid": [config_to_dict(c) for c in self.config_grid],
            "run_repeats": self.run_repeats,
            "raw_log_paths": {
                k: list(v) for k, v in self.raw_log_paths.items()
            },
            "decode_batch_sizes": list(self.decode_batch_sizes),
            "enabled_modules": list(self.enabled_modules),
            "noise_sensitive_metrics": list(self.noise_sensitive_metrics),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunManifest":
        return cls(
            campaign_name=d["campaign_name"],
            pinned_build=d["pinned_build"],
            platform=d["platform"],
            pinned_device=d["pinned_device"],
            environment=EnvironmentCapture.from_dict(d["environment"]),
            config_grid=[config_from_dict(c) for c in (d.get("config_grid") or [])],
            run_repeats=int(d.get("run_repeats", 5)),
            raw_log_paths={
                k: list(v) for k, v in (d.get("raw_log_paths") or {}).items()
            },
            decode_batch_sizes=[int(b) for b in (d.get("decode_batch_sizes") or [])],
            enabled_modules=list(d.get("enabled_modules") or []),
            noise_sensitive_metrics=list(d.get("noise_sensitive_metrics") or []),
        )
