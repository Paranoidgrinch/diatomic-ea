"""Execution helpers for the DiatomicEA desktop interface.

Internal scientific identifiers are deliberately separated from
user-facing language in this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.jobs import (
    CalculationMode,
    JobStatus,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.production_plan import (
    PRODUCTION_PLAN_FILENAME,
)
from diatomic_ea.production_run import (
    PRODUCTION_STATUS_FILENAME,
)


@dataclass(frozen=True, slots=True)
class ProcessCommand:
    """One executable plus its command-line arguments."""

    program: str
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuiCalculationSpec:
    """Immutable settings captured when a GUI job enters the queue."""

    job_id: str
    molecule: DiatomicMolecule
    minimum_angstrom: float
    maximum_angstrom: float
    spin_max: int
    workers: int
    run_id: str
    threads_per_worker: int = 1

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError(
                "job_id must not be empty."
            )

        if not self.run_id:
            raise ValueError(
                "run_id must not be empty."
            )

        if self.minimum_angstrom <= 0:
            raise ValueError(
                "Minimum bond length must be positive."
            )

        if (
            self.maximum_angstrom
            <= self.minimum_angstrom
        ):
            raise ValueError(
                "Maximum bond length must be greater "
                "than minimum bond length."
            )

        if self.spin_max < 0:
            raise ValueError(
                "Maximum spin must not be negative."
            )

        if self.workers < 1:
            raise ValueError(
                "At least one worker is required."
            )

        if self.threads_per_worker < 1:
            raise ValueError(
                "At least one thread per worker is required."
            )

    def run_directory(
        self,
        output_root: str | Path,
    ) -> Path:
        return (
            Path(
                output_root
            )
            / self.molecule.formula
            / self.run_id
        )

    def plan_path(
        self,
        output_root: str | Path,
    ) -> Path:
        return (
            self.run_directory(
                output_root
            )
            / PRODUCTION_PLAN_FILENAME
        )

    def status_path(
        self,
        output_root: str | Path,
    ) -> Path:
        return (
            self.run_directory(
                output_root
            )
            / "logs"
            / PRODUCTION_STATUS_FILENAME
        )

    def final_result_path(
        self,
        output_root: str | Path,
    ) -> Path:
        return (
            self.run_directory(
                output_root
            )
            / "04_final"
            / "final_result.csv"
        )


def make_gui_run_id(
    molecule: DiatomicMolecule,
    job_id: str,
) -> str:
    """Create a stable, human-readable run id for one queued job."""

    cleaned_job_id = (
        job_id.strip()
    )

    if not cleaned_job_id:
        raise ValueError(
            "job_id must not be empty."
        )

    return (
        molecule.formula.lower()
        + "-"
        + cleaned_job_id[:12]
    )


def calculation_mode_label(
    mode: CalculationMode,
) -> str:
    """Translate internal workflow identifiers for the GUI."""

    if mode is CalculationMode.SCHEMA_F:
        return "Standard calculation"

    if mode is CalculationMode.SMOKE_TEST:
        return "System check"

    return "Calculation"


def job_status_label(
    status: JobStatus,
) -> str:
    """Translate internal job status values for display."""

    labels = {
        JobStatus.QUEUED: "Waiting",
        JobStatus.RUNNING: "Running",
        JobStatus.COMPLETED: "Completed",
        JobStatus.FAILED: "Failed",
        JobStatus.CANCELLED: "Cancelled",
    }

    return labels.get(
        status,
        "Unknown",
    )


def _number(
    value: float,
) -> str:
    return format(
        value,
        ".8g",
    )


def build_plan_command(
    spec: GuiCalculationSpec,
    *,
    output_root: str | Path,
    python_executable: str | None = None,
) -> ProcessCommand:
    """Build the command that prepares one calculation."""

    executable = (
        python_executable
        or sys.executable
    )

    return ProcessCommand(
        program=executable,
        arguments=(
            "-m",
            "diatomic_ea.production_plan",
            spec.molecule.atom_a,
            spec.molecule.atom_b,
            "--minimum",
            _number(
                spec.minimum_angstrom
            ),
            "--maximum",
            _number(
                spec.maximum_angstrom
            ),
            "--spin-max",
            str(
                spec.spin_max
            ),
            "--workers",
            str(
                spec.workers
            ),
            "--threads-per-worker",
            str(
                spec.threads_per_worker
            ),
            "--run-id",
            spec.run_id,
            "--output-root",
            str(
                Path(
                    output_root
                )
            ),
        ),
    )


def build_run_command(
    spec: GuiCalculationSpec,
    *,
    output_root: str | Path,
    python_executable: str | None = None,
    recover_stale_lock: bool = False,
) -> ProcessCommand:
    """Build the command that executes or resumes one calculation."""

    executable = (
        python_executable
        or sys.executable
    )

    arguments = [
        "-m",
        "diatomic_ea.production_run",
        "--plan",
        str(
            spec.plan_path(
                output_root
            )
        ),
        "--atom-a",
        spec.molecule.atom_a,
        "--atom-b",
        spec.molecule.atom_b,
        "--start",
    ]

    if recover_stale_lock:
        arguments.append(
            "--recover-stale-lock"
        )

    return ProcessCommand(
        program=executable,
        arguments=tuple(
            arguments
        ),
    )
