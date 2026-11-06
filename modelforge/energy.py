"""GPU energy telemetry via NVML (GPU only; not whole-system power).

Uses the cumulative energy counter as an end-minus-start difference. If the
counter is unsupported (possible under WSL), falls back to integrating sampled
power and labels the method. If neither works, energy is reported as
unavailable (None), never zero.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class EnergyReading:
    method: str
    gpu_energy_joules: float | None
    seconds: float
    notes: list[str] = field(default_factory=list)


class GpuEnergyMeter:
    def __init__(self, device_index: int = 0, sample_seconds: float = 0.5) -> None:
        self.device_index = device_index
        self.sample_seconds = sample_seconds
        self._nvml = None
        self._handle = None
        self._start_mj = None
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self.notes: list[str] = []

    def __enter__(self) -> "GpuEnergyMeter":
        self._t0 = time.monotonic()
        try:
            import pynvml  # provided by nvidia-ml-py
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a run
            self.notes.append(f"NVML unavailable: {type(exc).__name__}")
            return self
        try:
            self._start_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"energy counter unsupported: {type(exc).__name__}; sampling power instead")
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self._samples.append((time.monotonic(), watts))
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"power sampling failed: {type(exc).__name__}")
                return
            self._stop.wait(self.sample_seconds)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def reading(self) -> EnergyReading:
        seconds = time.monotonic() - self._t0
        if self._start_mj is not None:
            try:
                end_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                return EnergyReading("nvml_total_energy_counter_delta", (end_mj - self._start_mj) / 1000.0, seconds, self.notes)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"energy counter read failed: {type(exc).__name__}")
        if len(self._samples) >= 2:
            joules = sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(self._samples, self._samples[1:]))
            return EnergyReading("nvml_power_samples_trapezoid", joules, seconds, self.notes)
        return EnergyReading("unavailable", None, seconds, self.notes)
