"""Persistent DiatomicEA project files."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from diatomic_ea import __version__
from diatomic_ea.jobs import (
    CalculationJob,
    CalculationMode,
    JobStatus,
)
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.queue import CalculationQueue


PROJECT_FORMAT_VERSION = 1


def _job_to_dict(job: CalculationJob) -> dict[str, str]:
    return {
        "job_id": job.job_id,
        "atom_a": job.molecule.atom_a,
        "atom_b": job.molecule.atom_b,
        "mode": job.mode.value,
        "status": job.status.value,
    }


def _job_from_dict(data: dict[str, Any]) -> CalculationJob:
    required = {
        "job_id",
        "atom_a",
        "atom_b",
        "mode",
        "status",
    }

    missing = required.difference(data)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            f"Project job is missing fields: {missing_text}"
        )

    try:
        mode = CalculationMode(data["mode"])
        status = JobStatus(data["status"])
    except ValueError as exc:
        raise ValueError(
            "Project contains an unsupported job mode or status."
        ) from exc

    # A running process cannot survive an application restart.
    # Recover it as queued so that the user may run it again.
    if status is JobStatus.RUNNING:
        status = JobStatus.QUEUED

    return CalculationJob(
        molecule=DiatomicMolecule(
            data["atom_a"],
            data["atom_b"],
        ),
        mode=mode,
        job_id=str(data["job_id"]),
        status=status,
    )


def project_to_dict(
    queue: CalculationQueue,
) -> dict[str, Any]:
    """Convert a calculation queue into project-file data."""
    return {
        "format_version": PROJECT_FORMAT_VERSION,
        "application": "DiatomicEA",
        "application_version": __version__,
        "jobs": [
            _job_to_dict(job)
            for job in queue.jobs
        ],
    }


def project_from_dict(
    data: dict[str, Any],
) -> CalculationQueue:
    """Create a calculation queue from project-file data."""
    if not isinstance(data, dict):
        raise ValueError("Project root must be a JSON object.")

    version = data.get("format_version")

    if version != PROJECT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported project format version: "
            f"{version!r}"
        )

    jobs_data = data.get("jobs")

    if not isinstance(jobs_data, list):
        raise ValueError(
            "Project field 'jobs' must be a list."
        )

    jobs: list[CalculationJob] = []

    for item in jobs_data:
        if not isinstance(item, dict):
            raise ValueError(
                "Each project job must be a JSON object."
            )

        jobs.append(_job_from_dict(item))

    return CalculationQueue(jobs)


def save_project(
    queue: CalculationQueue,
    path: str | Path,
) -> Path:
    """Atomically save a DiatomicEA project file."""
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = project_to_dict(queue)

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            json.dump(
                data,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

        raise

    return destination


def load_project(
    path: str | Path,
) -> CalculationQueue:
    """Load a DiatomicEA project file."""
    source = Path(path)

    with source.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    return project_from_dict(data)