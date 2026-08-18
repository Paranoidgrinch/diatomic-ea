"""Tests for compute-resource configuration."""

import pytest

from diatomic_ea.config import ComputeConfig, build_compute_config
from diatomic_ea.resources import CpuResources


@pytest.fixture
def resources() -> CpuResources:
    return CpuResources(
        physical_cores=20,
        logical_cores=40,
        recommended_workers=18,
    )


def test_default_config_uses_recommendation(
    resources: CpuResources,
) -> None:
    config = build_compute_config(resources=resources)

    assert config.workers == 18
    assert config.threads_per_worker == 1


def test_explicit_worker_count(
    resources: CpuResources,
) -> None:
    config = build_compute_config(
        workers=12,
        resources=resources,
    )

    assert config.workers == 12


def test_too_many_workers_are_rejected(
    resources: CpuResources,
) -> None:
    with pytest.raises(ValueError):
        build_compute_config(
            workers=41,
            resources=resources,
        )


def test_zero_workers_are_rejected() -> None:
    with pytest.raises(ValueError):
        ComputeConfig(workers=0)


def test_worker_environment_limits_threading() -> None:
    config = ComputeConfig(workers=8)

    assert config.worker_environment() == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }