"""
GPU telemetry helper using NVIDIA's NVML interface. Samples utilization, power,
and memory every ~0.5s during batch execution, recording per-batch aggregates and
estimating electricity cost via cost = ((avg_gpu_w + cpu_fudge)*hours)/1000 *
price_per_kwh.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - NVML optional
    pynvml = None
    _NVML_AVAILABLE = False


@dataclass
class BatchTelemetry:
    label: str
    avg_util_pct: Optional[float]
    avg_power_w: Optional[float]
    peak_vram_mb: Optional[float]
    elapsed_s: float


@dataclass
class TelemetryRecorder:
    sample_interval: float = 0.75
    cpu_overhead_w: float = 50.0
    batches: List[BatchTelemetry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._sampling = False
        self._samples: List[Dict[str, float]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("NVML init failed: %s", exc)
                self._nvml_ok = False
                self.note = "NVML unavailable"
            else:
                self._nvml_ok = True
                self.note = None
        else:
            self._nvml_ok = False
            self.note = "NVML unavailable"

    def _sample_loop(self) -> None:  # pragma: no cover - requires NVML
        handle_count = pynvml.nvmlDeviceGetCount()
        while self._sampling and self._stop_event and not self._stop_event.is_set():
            agg = {"util": 0.0, "power": 0.0, "vram": 0.0}
            for idx in range(handle_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    agg["util"] += utilization.gpu
                except pynvml.NVMLError:
                    # GPU utilization monitoring failed, continue without it
                    agg["util"] = None
                try:
                    agg["power"] += pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    agg["power"] = None
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    agg["vram"] = max(agg["vram"], mem_info.used / (1024 * 1024))
                except pynvml.NVMLError:
                    agg["vram"] = None
            if agg["util"] is not None:
                agg["util"] = agg["util"] / max(1, handle_count)
            if agg["power"] is not None:
                agg["power"] = agg["power"] / max(1, handle_count)
            self._samples.append(agg.copy())
            time.sleep(self.sample_interval)

    def start_batch(self) -> None:
        if not self._nvml_ok or self._sampling:
            return
        self._samples.clear()
        self._sampling = True
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def end_batch(self, label: str, elapsed_s: float) -> BatchTelemetry:
        if self._nvml_ok and self._sampling:
            assert self._stop_event and self._thread
            self._stop_event.set()
            self._thread.join(timeout=2)
            self._sampling = False
        if not self._samples:
            telemetry = BatchTelemetry(label, None, None, None, elapsed_s)
        else:
            util_vals = [s["util"] for s in self._samples if s["util"] is not None]
            power_vals = [s["power"] for s in self._samples if s["power"] is not None]
            vram_vals = [s["vram"] for s in self._samples if s["vram"] is not None]
            telemetry = BatchTelemetry(
                label=label,
                avg_util_pct=sum(util_vals) / len(util_vals) if util_vals else None,
                avg_power_w=sum(power_vals) / len(power_vals) if power_vals else None,
                peak_vram_mb=max(vram_vals) if vram_vals else None,
                elapsed_s=elapsed_s,
            )
        self.batches.append(telemetry)
        self._samples.clear()
        return telemetry

    def summarize(self, price_per_kwh: float, output_path: Path, elapsed_hours: float) -> Dict:
        if not self.batches:
            rollup = {
                "elapsed_hours": elapsed_hours,
                "avg_power_w": None,
                "electricity_cost": None,
                "batches": [],
                "note": self.note,
            }
            self._write_json(rollup, output_path)
            return rollup
        power_samples = [
            (batch.avg_power_w, batch.elapsed_s) for batch in self.batches if batch.avg_power_w is not None
        ]
        if not power_samples:
            avg_power = None
            cost = None
            if not self.note:
                self.note = "NVML provided no power samples"
        else:
            total_time = sum(duration for _, duration in power_samples)
            avg_power = sum(power * duration for power, duration in power_samples) / total_time
            cost = ((avg_power + self.cpu_overhead_w) * elapsed_hours) / 1000.0 * price_per_kwh
        rollup = {
            "elapsed_hours": elapsed_hours,
            "avg_power_w": avg_power,
            "electricity_cost": cost,
            "batches": [batch.__dict__ for batch in self.batches],
            "note": self.note,
        }
        self._write_json(rollup, output_path)
        return rollup

    def close(self) -> None:  # pragma: no cover - depends on NVML
        if self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def _write_json(payload: Dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
