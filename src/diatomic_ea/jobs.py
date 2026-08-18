"""Calculation-job definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from diatomic_ea.molecule import DiatomicMolecule


class CalculationMode(str, Enum):
    """Supported calculation workflows."""

    SCHEMA_F = "schema-f"
    SMOKE_TEST = "smoke-test"


class JobStatus(str, Enum):
    """Lifecycle state of a calculation job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {
        JobStatus.RUNNING,
        JobStatus.CANCELLED,
    },
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}


@dataclass(slots=True)
class CalculationJob:
    """A single queued molecular calculation."""

    molecule: DiatomicMolecule
    mode: CalculationMode = CalculationMode.SCHEMA_F
    job_id: str = field(
        default_factory=lambda: uuid4().hex
    )
    status: JobStatus = JobStatus.QUEUED

    def transition_to(self, new_status: JobStatus) -> None:
        """Perform a validated job-state transition."""
        allowed = _ALLOWED_TRANSITIONS[self.status]

        if new_status not in allowed:
            raise ValueError(
                f"Invalid job transition: "
                f"{self.status.value} -> {new_status.value}"
            )

        self.status = new_status