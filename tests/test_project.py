"""Tests for persistent project files."""

import json

import pytest

from diatomic_ea.jobs import (
    CalculationJob,
    CalculationMode,
    JobStatus,
)
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.project import (
    PROJECT_FORMAT_VERSION,
    load_project,
    project_from_dict,
    save_project,
)
from diatomic_ea.queue import CalculationQueue


def make_job(
    atom_a: str,
    atom_b: str,
    *,
    mode: CalculationMode = CalculationMode.SCHEMA_F,
) -> CalculationJob:
    return CalculationJob(
        molecule=DiatomicMolecule(atom_a, atom_b),
        mode=mode,
    )


def test_project_round_trip(tmp_path) -> None:
    first = make_job("Al", "O")
    second = make_job(
        "Mg",
        "O",
        mode=CalculationMode.SMOKE_TEST,
    )

    queue = CalculationQueue([first, second])

    path = tmp_path / "test.dea.json"

    save_project(queue, path)
    restored = load_project(path)

    assert len(restored.jobs) == 2

    assert restored.jobs[0].job_id == first.job_id
    assert restored.jobs[0].molecule.formula == "AlO"
    assert restored.jobs[0].mode is CalculationMode.SCHEMA_F

    assert restored.jobs[1].job_id == second.job_id
    assert restored.jobs[1].molecule.formula == "MgO"
    assert (
        restored.jobs[1].mode
        is CalculationMode.SMOKE_TEST
    )


def test_terminal_status_is_preserved(tmp_path) -> None:
    job = make_job("Fe", "H")
    job.transition_to(JobStatus.RUNNING)
    job.transition_to(JobStatus.COMPLETED)

    path = tmp_path / "completed.dea.json"

    save_project(
        CalculationQueue([job]),
        path,
    )

    restored = load_project(path)

    assert (
        restored.jobs[0].status
        is JobStatus.COMPLETED
    )


def test_running_job_recovers_as_queued(tmp_path) -> None:
    job = make_job("Al", "O")
    job.transition_to(JobStatus.RUNNING)

    path = tmp_path / "interrupted.dea.json"

    save_project(
        CalculationQueue([job]),
        path,
    )

    restored = load_project(path)

    assert (
        restored.jobs[0].status
        is JobStatus.QUEUED
    )


def test_save_creates_parent_directories(
    tmp_path,
) -> None:
    job = make_job("Mg", "O")

    path = (
        tmp_path
        / "nested"
        / "projects"
        / "sample.dea.json"
    )

    save_project(
        CalculationQueue([job]),
        path,
    )

    assert path.exists()


def test_project_file_is_valid_json(
    tmp_path,
) -> None:
    job = make_job("Al", "O")
    path = tmp_path / "sample.dea.json"

    save_project(
        CalculationQueue([job]),
        path,
    )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert (
        data["format_version"]
        == PROJECT_FORMAT_VERSION
    )
    assert data["application"] == "DiatomicEA"


def test_unsupported_format_is_rejected() -> None:
    with pytest.raises(ValueError):
        project_from_dict(
            {
                "format_version": 999,
                "jobs": [],
            }
        )


def test_invalid_jobs_field_is_rejected() -> None:
    with pytest.raises(ValueError):
        project_from_dict(
            {
                "format_version": PROJECT_FORMAT_VERSION,
                "jobs": "not-a-list",
            }
        )


def test_missing_job_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        project_from_dict(
            {
                "format_version": PROJECT_FORMAT_VERSION,
                "jobs": [
                    {
                        "atom_a": "Al",
                        "atom_b": "O",
                    }
                ],
            }
        )