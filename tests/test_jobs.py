"""Tests for calculation jobs."""

import pytest

from diatomic_ea.jobs import (
    CalculationJob,
    CalculationMode,
    JobStatus,
)
from diatomic_ea.molecule import DiatomicMolecule


def test_new_job_is_queued() -> None:
    job = CalculationJob(
        molecule=DiatomicMolecule("Al", "O"),
    )

    assert job.status is JobStatus.QUEUED
    assert job.mode is CalculationMode.SCHEMA_F
    assert job.job_id


def test_smoke_test_mode() -> None:
    job = CalculationJob(
        molecule=DiatomicMolecule("Mg", "O"),
        mode=CalculationMode.SMOKE_TEST,
    )

    assert job.mode is CalculationMode.SMOKE_TEST


def test_valid_job_lifecycle() -> None:
    job = CalculationJob(
        molecule=DiatomicMolecule("Fe", "H"),
    )

    job.transition_to(JobStatus.RUNNING)
    job.transition_to(JobStatus.COMPLETED)

    assert job.status is JobStatus.COMPLETED


def test_invalid_job_transition_is_rejected() -> None:
    job = CalculationJob(
        molecule=DiatomicMolecule("Al", "O"),
    )

    with pytest.raises(ValueError):
        job.transition_to(JobStatus.COMPLETED)