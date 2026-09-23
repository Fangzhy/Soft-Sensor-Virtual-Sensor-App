"""Small, thread-safe, process-local store for fitted demo models.

Random run IDs act as capabilities: keep them in their originating session.
HTTP mode uses these as capabilities. The cloud adapter additionally checks
browser-session ownership. Neither mode provides durable authenticated accounts.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any
from uuid import uuid4


@dataclass
class ModelRun:
    """Retain the fitted estimator and compact evidence, never the API key."""

    model: Any
    model_name: str
    ranges: dict
    radius: float | None
    evidence: dict
    created: float = field(default_factory=monotonic)
    explanation: dict | None = None
    explanation_lock: Any = field(default_factory=Lock, repr=False)


class RunStore:
    """Evict oldest runs at capacity and expire after one hour by default."""

    def __init__(self, capacity: int = 16, ttl: float = 3600):
        self.capacity, self.ttl = capacity, ttl
        self._runs = OrderedDict()
        self._lock = Lock()

    def _expire(self):
        """Remove expired entries while the caller holds the store lock."""
        for key in list(self._runs):
            if monotonic() - self._runs[key].created >= self.ttl:
                del self._runs[key]

    def put(self, run: ModelRun) -> str:
        """Create a unique run ID without reusing another browser's model."""
        with self._lock:
            self._expire()
            while len(self._runs) >= self.capacity:
                self._runs.popitem(last=False)
            run_id = uuid4().hex
            self._runs[run_id] = run
            return run_id

    def get(self, run_id: str) -> ModelRun:
        """Raise KeyError when the run was evicted, expired, or never existed."""
        with self._lock:
            self._expire()
            return self._runs[run_id]


RUNS = RunStore()


def retain_run(model, x_train, features, evaluation, comparison=None) -> str:
    """Build server-owned evidence without sending raw rows to the LLM."""
    evidence = {
        "data_context": "Demo sensor data; UI uses synthetic independent snapshots. Not validated on a real process.",
        "target": "Solid concentration, mass percent (w/w)",
        "error_units": "percentage points; R2 is unitless",
        "model": evaluation.model,
        "split": {"training": evaluation.train_count, "test": evaluation.test_count,
                  "seed": evaluation.split_seed, "calibration": 0},
        "test_metrics": evaluation.test_metrics.model_dump(),
        "training_metrics": evaluation.train_metrics.model_dump(),
        "baseline_test_metrics": evaluation.baseline_test_metrics.model_dump(),
        "cv": None, "diagnostics": None, "optimizer_warnings": [],
    }
    radius = None
    if comparison is not None:
        evidence["cv"] = [{"model": item.model, "mean": item.mean.model_dump(),
                           "std": item.std.model_dump(), "warnings": item.warnings}
                          for item in comparison.results]
        evidence["selection_rule"] = comparison.selection_metric
        evidence["optimizer_warnings"] = comparison.warnings
        if comparison.diagnostics is not None:
            evidence["diagnostics"] = comparison.diagnostics.model_dump(exclude={"intervals"})
            radius = comparison.diagnostics.half_width
            evidence["split"]["calibration"] = comparison.diagnostics.calibration_count
    ranges = {name: {"min": float(x_train[:, i].min()), "max": float(x_train[:, i].max())}
              for i, name in enumerate(features)}
    return RUNS.put(ModelRun(model, evaluation.model, ranges, radius, evidence))
