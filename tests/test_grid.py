"""Tests for Schema F fast-grid planning."""

import pytest

from diatomic_ea.grid import (
    BondGrid,
    build_fast_grid_plan_from_electron_counts,
)
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.states import ChargeState


def test_decimal_safe_bond_grid() -> None:
    grid = BondGrid(
        minimum_angstrom=1.0,
        maximum_angstrom=1.05,
        step_angstrom=0.025,
    )

    assert grid.values == (
        1.0,
        1.025,
        1.05,
    )


def test_noninteger_range_does_not_overshoot() -> None:
    grid = BondGrid(
        minimum_angstrom=1.0,
        maximum_angstrom=1.10,
        step_angstrom=0.03,
    )

    assert grid.values == (
        1.0,
        1.03,
        1.06,
        1.09,
    )


def test_single_point_grid() -> None:
    grid = BondGrid(
        minimum_angstrom=1.5,
        maximum_angstrom=1.5,
        step_angstrom=0.025,
    )

    assert grid.values == (1.5,)


def test_invalid_bond_grid() -> None:
    with pytest.raises(ValueError):
        BondGrid(
            minimum_angstrom=2.0,
            maximum_angstrom=1.0,
            step_angstrom=0.025,
        )


def test_schema_f_task_count() -> None:
    plan = build_fast_grid_plan_from_electron_counts(
        molecule=DiatomicMolecule("Al", "O"),
        neutral_electrons=20,
        anion_electrons=21,
        spin_max=3,
        minimum_angstrom=1.0,
        maximum_angstrom=1.05,
    )

    # 5 bases
    # x 4 functionals
    # x (2 neutral + 2 anion states)
    # x 3 bond lengths
    assert plan.task_count == 240
    assert plan.bond_point_count == 3


def test_first_task_is_deterministic() -> None:
    plan = build_fast_grid_plan_from_electron_counts(
        molecule=DiatomicMolecule("Al", "O"),
        neutral_electrons=20,
        anion_electrons=21,
        spin_max=3,
        minimum_angstrom=1.0,
        maximum_angstrom=1.05,
    )

    task = plan.tasks[0]

    assert task.basis == "def2-svp"
    assert task.functional == "PBE"
    assert task.charge is ChargeState.NEUTRAL
    assert task.spin == 0
    assert task.bond_length_angstrom == 1.0


def test_last_task_is_deterministic() -> None:
    plan = build_fast_grid_plan_from_electron_counts(
        molecule=DiatomicMolecule("Al", "O"),
        neutral_electrons=20,
        anion_electrons=21,
        spin_max=3,
        minimum_angstrom=1.0,
        maximum_angstrom=1.05,
    )

    task = plan.tasks[-1]

    assert task.basis == "def2-tzvpd"
    assert task.functional == "TPSSh"
    assert task.charge is ChargeState.ANION
    assert task.spin == 3
    assert task.bond_length_angstrom == 1.05


def test_task_ids_are_unique() -> None:
    plan = build_fast_grid_plan_from_electron_counts(
        molecule=DiatomicMolecule("Fe", "H"),
        neutral_electrons=10,
        anion_electrons=11,
        spin_max=3,
        minimum_angstrom=1.4,
        maximum_angstrom=1.45,
    )

    identifiers = [
        task.task_id
        for task in plan.tasks
    ]

    assert len(identifiers) == len(
        set(identifiers)
    )


def test_worker_thread_count_is_propagated() -> None:
    plan = build_fast_grid_plan_from_electron_counts(
        molecule=DiatomicMolecule("Mg", "O"),
        neutral_electrons=20,
        anion_electrons=21,
        spin_max=1,
        minimum_angstrom=1.5,
        maximum_angstrom=1.5,
        threads_per_worker=2,
    )

    assert all(
        task.threads_per_worker == 2
        for task in plan.tasks
    )