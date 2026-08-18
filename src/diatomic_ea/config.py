"""Compute-resource configuration for DiatomicEA."""

from __future__ import annotations

from dataclasses import dataclass

from diatomic_ea.resources import CpuResources, detect_cpu_resources


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """Resources assigned to DiatomicEA calculations."""

    workers: int
    threads_per_worker: int = 1

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

        if self.threads_per_worker < 1:
            raise ValueError("threads_per_worker must be at least 1")

    def validate_for(self, resources: CpuResources) -> None:
        """Validate this configuration against available hardware."""
        if self.workers > resources.logical_cores:
            raise ValueError(
                f"Requested {self.workers} workers, but only "
                f"{resources.logical_cores} logical CPU cores are available."
            )

    def worker_environment(self) -> dict[str, str]:
        """Environment variables used inside calculation workers."""
        threads = str(self.threads_per_worker)

        return {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
        }


def build_compute_config(
    workers: int | None = None,
    resources: CpuResources | None = None,
) -> ComputeConfig:
    """Create and validate a compute configuration.

    If no explicit worker count is provided, the conservative system
    recommendation is used.
    """
    if resources is None:
        resources = detect_cpu_resources()

    selected_workers = (
        resources.recommended_workers
        if workers is None
        else workers
    )

    config = ComputeConfig(
        workers=selected_workers,
        threads_per_worker=1,
    )
    config.validate_for(resources)

    return config