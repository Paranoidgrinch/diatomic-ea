"""Crash-resistant CSV storage for single-point results."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)


RAW_RESULT_COLUMNS = (
    "task_id",
    "molecule",
    "atom_a",
    "atom_b",
    "charge",
    "spin",
    "multiplicity",
    "functional",
    "basis",
    "bond_length_angstrom",
    "grid_level",
    "conv_tol",
    "max_cycle",
    "max_memory_mb",
    "threads_per_worker",
    "status",
    "error",
    "energy_hartree",
    "energy_ev",
    "converged",
    "used_level_shift_retry",
    "used_newton_retry",
    "electron_count",
    "alpha_electrons",
    "beta_electrons",
    "basis_label_a",
    "basis_label_b",
    "ecp_label_a",
    "ecp_label_b",
    "homo_hartree",
    "lumo_hartree",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "positive_homo_warning",
    "s2",
    "observed_multiplicity",
    "spin_contamination_warning",
    "pyscf_version",
    "elapsed_seconds",
)


def _optional(value) -> str | object:
    if value is None:
        return ""

    return value


def result_row(
    task: SinglePointTask,
    result: SinglePointResult,
) -> dict[str, object]:
    """Convert one task/result pair into a CSV row."""
    if task.task_id != result.task_id:
        raise ValueError(
            "Task/result identifier mismatch."
        )

    frontier = result.frontier

    return {
        "task_id": task.task_id,
        "molecule": task.molecule.formula,
        "atom_a": task.molecule.atom_a,
        "atom_b": task.molecule.atom_b,
        "charge": int(task.charge),
        "spin": task.spin,
        "multiplicity": task.multiplicity,
        "functional": task.functional,
        "basis": task.basis,
        "bond_length_angstrom": (
            task.bond_length_angstrom
        ),
        "grid_level": task.grid_level,
        "conv_tol": task.conv_tol,
        "max_cycle": task.max_cycle,
        "max_memory_mb": task.max_memory_mb,
        "threads_per_worker": (
            task.threads_per_worker
        ),
        "status": result.status.value,
        "error": result.error,
        "energy_hartree": result.energy_hartree,
        "energy_ev": result.energy_ev,
        "converged": result.converged,
        "used_level_shift_retry": (
            result.used_level_shift_retry
        ),
        "used_newton_retry": (
            result.used_newton_retry
        ),
        "electron_count": _optional(
            result.electron_count
        ),
        "alpha_electrons": _optional(
            result.alpha_electrons
        ),
        "beta_electrons": _optional(
            result.beta_electrons
        ),
        "basis_label_a": result.basis_label_a,
        "basis_label_b": result.basis_label_b,
        "ecp_label_a": result.ecp_label_a,
        "ecp_label_b": result.ecp_label_b,
        "homo_hartree": (
            frontier.homo_hartree
            if frontier is not None
            else ""
        ),
        "lumo_hartree": (
            frontier.lumo_hartree
            if frontier is not None
            else ""
        ),
        "homo_ev": (
            frontier.homo_ev
            if frontier is not None
            else ""
        ),
        "lumo_ev": (
            frontier.lumo_ev
            if frontier is not None
            else ""
        ),
        "gap_ev": (
            frontier.gap_ev
            if frontier is not None
            else ""
        ),
        "positive_homo_warning": (
            frontier.positive_homo_warning
            if frontier is not None
            else ""
        ),
        "s2": result.s2,
        "observed_multiplicity": (
            result.observed_multiplicity
        ),
        "spin_contamination_warning": (
            result.spin_contamination_warning
        ),
        "pyscf_version": result.pyscf_version,
        "elapsed_seconds": result.elapsed_seconds,
    }


class RawResultStore:
    """Append-only storage for recoverable calculation results."""

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

    def append(
        self,
        task: SinglePointTask,
        result: SinglePointResult,
    ) -> None:
        """Append and immediately flush one completed task."""
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_header = (
            not self.path.exists()
            or self.path.stat().st_size == 0
        )

        row = result_row(
            task,
            result,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=RAW_RESULT_COLUMNS,
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    def latest_rows(
        self,
    ) -> dict[str, dict[str, str]]:
        """Return the newest stored row for each task ID."""
        if not self.path.exists():
            return {}

        latest: dict[
            str,
            dict[str, str],
        ] = {}

        with self.path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                return {}

            missing = set(
                RAW_RESULT_COLUMNS
            ).difference(
                reader.fieldnames
            )

            if missing:
                raise ValueError(
                    "Raw result CSV is missing columns: "
                    + ", ".join(sorted(missing))
                )

            for row in reader:
                task_id = row.get(
                    "task_id",
                    "",
                )

                if task_id:
                    latest[task_id] = row

        return latest

    def recorded_task_ids(
        self,
    ) -> frozenset[str]:
        """Return every task ID that has a stored result."""
        return frozenset(
            self.latest_rows()
        )

    def finished_task_ids(
        self,
        *,
        retry_errors: bool = True,
    ) -> frozenset[str]:
        """Return task IDs that should be skipped on resume.

        By default error rows are retried, while successfully executed
        rows are considered finished even if SCF convergence diagnostics
        later cause the scientific analyzer to reject them.
        """
        rows = self.latest_rows()

        if not retry_errors:
            return frozenset(rows)

        return frozenset(
            task_id
            for task_id, row in rows.items()
            if (
                row.get("status")
                == SinglePointStatus.OK.value
            )
        )


def pending_tasks(
    tasks: Iterable[SinglePointTask],
    store: RawResultStore,
    *,
    retry_errors: bool = True,
) -> tuple[SinglePointTask, ...]:
    """Return tasks not already finished in the CSV store."""
    finished = store.finished_task_ids(
        retry_errors=retry_errors
    )

    return tuple(
        task
        for task in tasks
        if task.task_id not in finished
    )