"""Reduced real fast-grid integration validation.

This module intentionally does not run complete Schema F.

It validates the production execution path:

    electron-count resolution
        -> FastGridPlan
        -> multiprocessing
        -> platform single-point worker
        -> WSL/PySCF on Windows
        -> crash-resistant raw CSV
        -> resume without duplicate calculations

The calculation is backend validation only and must not be interpreted
as an electron-affinity prediction.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.compute_environment import (
    EXPECTED_PYSCF_VERSION,
)
from diatomic_ea.csv_store import (
    RawResultStore,
)
from diatomic_ea.electron_count_adapter import (
    run_platform_electron_count,
)
from diatomic_ea.grid import (
    BondGrid,
    FastGridPlan,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.runner import (
    FastGridRunSummary,
    execute_fast_grid_resumable,
)
from diatomic_ea.single_point import (
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.states import (
    ChargeState,
    build_state_scan_plan,
)


VALIDATION_MOLECULE = DiatomicMolecule(
    "F",
    "F",
)

VALIDATION_BASIS = "def2-SVP"

VALIDATION_FUNCTIONAL = "PBE"

VALIDATION_BOND_MIN = 1.40

VALIDATION_BOND_MAX = 1.44

VALIDATION_BOND_STEP = 0.04


@dataclass(frozen=True, slots=True)
class ReducedPipelineSmokeReport:
    """Summary of the real reduced pipeline integration test."""

    passed: bool
    neutral_electrons: int
    anion_electrons: int
    task_count: int
    first_attempted: int
    first_completed_ok: int
    first_remaining: int
    resume_attempted: int
    resume_already_finished: int
    csv_rows: int
    csv_path: str
    message: str


def build_reduced_validation_plan(
    *,
    neutral_electrons: int,
    anion_electrons: int,
) -> FastGridPlan:
    """Build four real validation tasks around the F2 bond region."""
    state_scan = build_state_scan_plan(
        neutral_electrons=neutral_electrons,
        anion_electrons=anion_electrons,
        spin_max=1,
    )

    bond_grid = BondGrid(
        minimum_angstrom=(
            VALIDATION_BOND_MIN
        ),
        maximum_angstrom=(
            VALIDATION_BOND_MAX
        ),
        step_angstrom=(
            VALIDATION_BOND_STEP
        ),
    )

    tasks: list[
        SinglePointTask
    ] = []

    for charge_scan in (
        state_scan.neutral,
        state_scan.anion,
    ):
        if len(
            charge_scan.states
        ) != 1:
            raise RuntimeError(
                "Reduced validation expected exactly "
                "one spin state per charge."
            )

        state = (
            charge_scan.states[0]
        )

        for bond_length in (
            bond_grid.values
        ):
            tasks.append(
                SinglePointTask(
                    molecule=(
                        VALIDATION_MOLECULE
                    ),
                    charge=state.charge,
                    spin=state.spin,
                    functional=(
                        VALIDATION_FUNCTIONAL
                    ),
                    basis=(
                        VALIDATION_BASIS
                    ),
                    bond_length_angstrom=(
                        bond_length
                    ),
                    grid_level=0,
                    conv_tol=1.0e-8,
                    max_cycle=100,
                    max_memory_mb=1000,
                    threads_per_worker=1,
                )
            )

    return FastGridPlan(
        molecule=VALIDATION_MOLECULE,
        state_scan=state_scan,
        bond_grid=bond_grid,
        tasks=tuple(
            tasks
        ),
    )


def _validate_raw_store(
    *,
    store: RawResultStore,
    plan: FastGridPlan,
    neutral_electrons: int,
    anion_electrons: int,
) -> tuple[
    bool,
    str,
]:
    """Validate persisted scientific execution fields for every task."""
    rows = store.latest_rows()

    if len(
        rows
    ) != plan.task_count:
        return (
            False,
            "Raw CSV row count does not match "
            "the reduced plan.",
        )

    for task in plan.tasks:
        row = rows.get(
            task.task_id
        )

        if row is None:
            return (
                False,
                "Raw CSV is missing task "
                f"{task.task_id}.",
            )

        if (
            row.get(
                "status"
            )
            != SinglePointStatus.OK.value
        ):
            return (
                False,
                "Task did not return OK status: "
                f"{task.task_id}.",
            )

        if (
            row.get(
                "converged",
                "",
            ).casefold()
            != "true"
        ):
            return (
                False,
                "Task did not converge: "
                f"{task.task_id}.",
            )

        try:
            energy = float(
                row[
                    "energy_hartree"
                ]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return (
                False,
                "Task has no valid energy: "
                f"{task.task_id}.",
            )

        if not math.isfinite(
            energy
        ):
            return (
                False,
                "Task has non-finite energy: "
                f"{task.task_id}.",
            )

        expected_electrons = (
            neutral_electrons
            if task.charge
            is ChargeState.NEUTRAL
            else anion_electrons
        )

        try:
            stored_electrons = int(
                row[
                    "electron_count"
                ]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return (
                False,
                "Task has no valid electron count: "
                f"{task.task_id}.",
            )

        if (
            stored_electrons
            != expected_electrons
        ):
            return (
                False,
                "Persisted electron count mismatch "
                f"for {task.task_id}.",
            )

        if (
            row.get(
                "pyscf_version"
            )
            != EXPECTED_PYSCF_VERSION
        ):
            return (
                False,
                "Persisted PySCF version mismatch "
                f"for {task.task_id}.",
            )

    return (
        True,
        "All reduced fast-grid rows are valid.",
    )


def run_reduced_pipeline_validation(
    *,
    output_directory: str | Path,
    max_workers: int = 2,
) -> ReducedPipelineSmokeReport:
    """Run and resume a four-task real production fast grid."""
    if max_workers < 1:
        raise ValueError(
            "max_workers must be at least 1."
        )

    output = Path(
        output_directory
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output
        / "reduced_fast_grid_raw.csv"
    )

    probe_r = (
        VALIDATION_BOND_MIN
        + VALIDATION_BOND_MAX
    ) / 2.0

    neutral_electrons = (
        run_platform_electron_count(
            molecule=VALIDATION_MOLECULE,
            charge=ChargeState.NEUTRAL,
            basis=VALIDATION_BASIS,
            bond_length_angstrom=probe_r,
            max_memory_mb=1000,
        )
    )

    anion_electrons = (
        run_platform_electron_count(
            molecule=VALIDATION_MOLECULE,
            charge=ChargeState.ANION,
            basis=VALIDATION_BASIS,
            bond_length_angstrom=probe_r,
            max_memory_mb=1000,
        )
    )

    if (
        anion_electrons
        != neutral_electrons + 1
    ):
        raise RuntimeError(
            "Anion electron count must be exactly "
            "one greater than the neutral count."
        )

    plan = (
        build_reduced_validation_plan(
            neutral_electrons=(
                neutral_electrons
            ),
            anion_electrons=(
                anion_electrons
            ),
        )
    )

    store = RawResultStore(
        csv_path
    )

    first = (
        execute_fast_grid_resumable(
            plan,
            store=store,
            max_workers=max_workers,
            retry_errors=True,
        )
    )

    rows_valid, validation_message = (
        _validate_raw_store(
            store=store,
            plan=plan,
            neutral_electrons=(
                neutral_electrons
            ),
            anion_electrons=(
                anion_electrons
            ),
        )
    )

    second = (
        execute_fast_grid_resumable(
            plan,
            store=store,
            max_workers=max_workers,
            retry_errors=True,
        )
    )

    csv_rows = len(
        store.latest_rows()
    )

    passed = (
        first.complete
        and first.attempted
        == plan.task_count
        and first.completed_ok
        == plan.task_count
        and first.completed_error == 0
        and rows_valid
        and second.complete
        and second.attempted == 0
        and second.already_finished
        == plan.task_count
        and csv_rows
        == plan.task_count
    )

    if passed:
        message = (
            "Real reduced fast-grid execution, "
            "CSV persistence, and resume passed."
        )
    else:
        message = (
            "Reduced fast-grid integration failed: "
            + validation_message
        )

    return ReducedPipelineSmokeReport(
        passed=passed,
        neutral_electrons=(
            neutral_electrons
        ),
        anion_electrons=(
            anion_electrons
        ),
        task_count=plan.task_count,
        first_attempted=(
            first.attempted
        ),
        first_completed_ok=(
            first.completed_ok
        ),
        first_remaining=(
            first.remaining_after_run
        ),
        resume_attempted=(
            second.attempted
        ),
        resume_already_finished=(
            second.already_finished
        ),
        csv_rows=csv_rows,
        csv_path=str(
            csv_path
        ),
        message=message,
    )


def _print_rows(
    store: RawResultStore,
) -> None:
    rows = store.latest_rows()

    for task_id in sorted(
        rows
    ):
        row = rows[
            task_id
        ]

        print(
            task_id,
            "| converged =",
            row.get(
                "converged"
            ),
            "| E / Ha =",
            row.get(
                "energy_hartree"
            ),
            "| PySCF =",
            row.get(
                "pyscf_version"
            ),
        )


def main() -> int:
    """Run the real reduced integration validation."""
    print()
    print(
        "DiatomicEA reduced fast-grid integration"
    )

    print(
        "========================================"
    )

    print()

    print(
        "BACKEND VALIDATION ONLY - "
        "NOT A SCIENTIFIC EA PREDICTION"
    )

    print()

    with tempfile.TemporaryDirectory(
        prefix="diatomic-ea-real-grid-"
    ) as temporary:
        try:
            report = (
                run_reduced_pipeline_validation(
                    output_directory=temporary,
                    max_workers=2,
                )
            )

        except Exception as exc:
            print(
                "Status: FAIL"
            )

            print(
                "Error:",
                str(
                    exc
                ),
            )

            return 1

        store = RawResultStore(
            report.csv_path
        )

        print(
            "Molecule:",
            VALIDATION_MOLECULE.formula,
        )

        print(
            "Method:",
            VALIDATION_FUNCTIONAL,
            "/",
            VALIDATION_BASIS,
        )

        print(
            "Neutral electrons:",
            report.neutral_electrons,
        )

        print(
            "Anion electrons:",
            report.anion_electrons,
        )

        print(
            "Tasks:",
            report.task_count,
        )

        print()

        _print_rows(
            store
        )

        print()

        print(
            "First run attempted:",
            report.first_attempted,
        )

        print(
            "First run OK:",
            report.first_completed_ok,
        )

        print(
            "First run remaining:",
            report.first_remaining,
        )

        print(
            "CSV rows:",
            report.csv_rows,
        )

        print(
            "Resume attempted:",
            report.resume_attempted,
        )

        print(
            "Resume already finished:",
            report.resume_already_finished,
        )

        print()

        print(
            "Status:",
            (
                "PASS"
                if report.passed
                else "FAIL"
            ),
        )

        print(
            "Message:",
            report.message,
        )

        return (
            0
            if report.passed
            else 1
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
