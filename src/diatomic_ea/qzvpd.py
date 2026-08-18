"""Generation of Schema F QZVPD refinement tasks."""

from __future__ import annotations

from dataclasses import dataclass

from diatomic_ea.grid import BondGrid
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.refinement import QZVPDCandidate
from diatomic_ea.schema_f import (
    SCHEMA_F,
    SchemaFSpec,
)
from diatomic_ea.single_point import SinglePointTask
from diatomic_ea.states import ChargeState


@dataclass(frozen=True, slots=True)
class QZVPDPlan:
    """All QZVPD single-point tasks for one molecule."""

    molecule: DiatomicMolecule
    candidates: tuple[QZVPDCandidate, ...]
    tasks: tuple[SinglePointTask, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def task_count(self) -> int:
        return len(self.tasks)


def build_qzvpd_plan(
    *,
    molecule: DiatomicMolecule,
    candidates: tuple[QZVPDCandidate, ...],
    threads_per_worker: int = 1,
    schema: SchemaFSpec = SCHEMA_F,
) -> QZVPDPlan:
    """Generate QZVPD refinement tasks from selected candidates."""
    if threads_per_worker < 1:
        raise ValueError(
            "threads_per_worker must be at least 1."
        )

    tasks: list[SinglePointTask] = []
    seen_task_ids: set[str] = set()

    for candidate in candidates:
        if candidate.molecule != molecule.formula:
            raise ValueError(
                "Candidate molecule does not match "
                f"requested molecule: "
                f"{candidate.molecule!r} != "
                f"{molecule.formula!r}."
            )

        charge = ChargeState(
            candidate.charge
        )

        bond_grid = BondGrid(
            minimum_angstrom=(
                candidate.r_min_angstrom
            ),
            maximum_angstrom=(
                candidate.r_max_angstrom
            ),
            step_angstrom=(
                schema.refinement
                .grid
                .step_angstrom
            ),
        )

        for bond_length in bond_grid.values:
            task = SinglePointTask(
                molecule=molecule,
                charge=charge,
                spin=candidate.spin,
                functional=(
                    candidate.functional
                ),
                basis=(
                    candidate.qzvpd_basis
                ),
                bond_length_angstrom=(
                    bond_length
                ),
                grid_level=(
                    schema.refinement
                    .grid
                    .grid_level
                ),
                conv_tol=(
                    schema.refinement
                    .grid
                    .conv_tol
                ),
                max_cycle=(
                    schema.refinement
                    .grid
                    .max_cycle
                ),
                max_memory_mb=(
                    schema.refinement
                    .grid
                    .max_memory_mb
                ),
                threads_per_worker=(
                    threads_per_worker
                ),
            )

            if task.task_id in seen_task_ids:
                continue

            seen_task_ids.add(
                task.task_id
            )

            tasks.append(task)

    return QZVPDPlan(
        molecule=molecule,
        candidates=candidates,
        tasks=tuple(tasks),
    )