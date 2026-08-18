"""CPU-resource detection for DiatomicEA."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class CpuResources:
    """CPU resources available on the current machine."""

    physical_cores: int
    logical_cores: int
    recommended_workers: int


def recommended_worker_count(physical_cores: int) -> int:
    """Return a conservative default number of calculation workers.

    DiatomicEA intentionally leaves part of the machine unused so that
    the operating system and graphical interface remain responsive.
    """
    if physical_cores < 1:
        raise ValueError("physical_cores must be at least 1")

    if physical_cores <= 2:
        return 1

    reserve = max(1, round(physical_cores * 0.10))
    return max(1, physical_cores - reserve)


def detect_cpu_resources() -> CpuResources:
    """Detect physical and logical CPU resources."""
    logical = psutil.cpu_count(logical=True) or 1
    physical = psutil.cpu_count(logical=False)

    if physical is None or physical < 1:
        physical = logical

    return CpuResources(
        physical_cores=physical,
        logical_cores=logical,
        recommended_workers=recommended_worker_count(physical),
    )
