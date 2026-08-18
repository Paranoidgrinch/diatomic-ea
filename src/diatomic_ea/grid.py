"""Schema F fast-grid task generation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.schema_f import SCHEMA_F, SchemaFSpec
from diatomic_ea.single_point import SinglePointTask
from diatomic_ea.states import (
    StateScanPlan,
    build_state_scan_plan,
)


@dataclass(frozen=True, slots=True)
class BondGrid:
    """Inclusive one-dimensional bond-length grid."""

    minimum_angstrom: float
    maximum_angstrom: float
    step_angstrom: float

    def __post_init__(self) -> None:
        if self.minimum_angstrom <= 0:
            raise ValueError(
                "minimum_angstrom must be positive."
            )

        if self.maximum_angstrom < self.minimum_angstrom:
            raise ValueError(
                "maximum_angstrom must be greater than "
                "or equal to minimum_angstrom."
            )

        if self.step_angstrom <= 0:
            raise ValueError(
                "step_angstrom must be positive."
            )

    @property
    def values(self) -> tuple[float, ...]:
        """Return decimal-safe grid points without overshooting."""
        current = Decimal(
            str(self.minimum_angstrom)
        )
        maximum = Decimal(
            str(self.maximum_angstrom)
        )
        step = Decimal(
            str(self.step_angstrom)
        )

        tolerance = Decimal("1e-12")
        values: list[float] = []

        while current <= maximum + tolerance:
            values.append(float(current))
            current += step

        return tuple(values)


@dataclass(frozen=True, slots=True)
class FastGridPlan:
    """All single points required for one Schema F fast grid."""

    molecule: DiatomicMolecule
    state_scan: StateScanPlan
    bond_grid: BondGrid
    tasks: tuple[SinglePointTask, ...]

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def bond_point_count(self) -> int:
        return len(self.bond_grid.values)


def build_fast_grid_plan(
    *,
    molecule: DiatomicMolecule,
    state_scan: StateScanPlan,
    minimum_angstrom: float,
    maximum_angstrom: float,
    threads_per_worker: int = 1,
    schema: SchemaFSpec = SCHEMA_F,
) -> FastGridPlan:
    """Create all Schema F fast-grid single-point tasks."""
    if threads_per_worker < 1:
        raise ValueError(
            "threads_per_worker must be at least 1."
        )

    bond_grid = BondGrid(
        minimum_angstrom=minimum_angstrom,
        maximum_angstrom=maximum_angstrom,
        step_angstrom=(
            schema.fast_grid.step_angstrom
        ),
    )

    tasks: list[SinglePointTask] = []

    for basis in schema.fast_bases:
        for functional in schema.functionals:
            for charge_scan in (
                state_scan.neutral,
                state_scan.anion,
            ):
                for state in charge_scan.states:
                    for bond_length in bond_grid.values:
                        tasks.append(
                            SinglePointTask(
                                molecule=molecule,
                                charge=state.charge,
                                spin=state.spin,
                                functional=functional,
                                basis=basis,
                                bond_length_angstrom=bond_length,
                                grid_level=(
                                    schema.fast_grid.grid_level
                                ),
                                conv_tol=(
                                    schema.fast_grid.conv_tol
                                ),
                                max_cycle=(
                                    schema.fast_grid.max_cycle
                                ),
                                max_memory_mb=(
                                    schema.fast_grid.max_memory_mb
                                ),
                                threads_per_worker=(
                                    threads_per_worker
                                ),
                            )
                        )

    return FastGridPlan(
        molecule=molecule,
        state_scan=state_scan,
        bond_grid=bond_grid,
        tasks=tuple(tasks),
    )


def build_fast_grid_plan_from_electron_counts(
    *,
    molecule: DiatomicMolecule,
    neutral_electrons: int,
    anion_electrons: int,
    spin_max: int,
    minimum_angstrom: float,
    maximum_angstrom: float,
    threads_per_worker: int = 1,
    schema: SchemaFSpec = SCHEMA_F,
) -> FastGridPlan:
    """Build state scans and fast-grid tasks in one step."""
    state_scan = build_state_scan_plan(
        neutral_electrons=neutral_electrons,
        anion_electrons=anion_electrons,
        spin_max=spin_max,
    )

    return build_fast_grid_plan(
        molecule=molecule,
        state_scan=state_scan,
        minimum_angstrom=minimum_angstrom,
        maximum_angstrom=maximum_angstrom,
        threads_per_worker=threads_per_worker,
        schema=schema,
    )