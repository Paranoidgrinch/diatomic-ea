"""Tests for QZVPD task generation."""

import pytest

from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.qzvpd import build_qzvpd_plan
from diatomic_ea.refinement import QZVPDCandidate
from diatomic_ea.states import ChargeState


def make_candidate(
    *,
    charge: int = 0,
    spin: int = 0,
    functional: str = "PBE",
    center: float = 1.60,
) -> QZVPDCandidate:
    return QZVPDCandidate(
        molecule="AlO",
        charge=charge,
        spin=spin,
        multiplicity=spin + 1,
        functional=functional,
        qzvpd_basis="def2-qzvpd",
        r_center_angstrom=center,
        r_min_angstrom=round(
            center - 0.10,
            6,
        ),
        r_max_angstrom=round(
            center + 0.10,
            6,
        ),
        source_basis="def2-tzvpd",
        source_energy_hartree=-100.0,
        method_minimum_count_for_spin=5,
        source_warnings=(),
    )


def test_one_candidate_generates_21_points() -> None:
    candidate = make_candidate()

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(candidate,),
    )

    assert plan.candidate_count == 1
    assert plan.task_count == 21


def test_qzvpd_grid_endpoints() -> None:
    candidate = make_candidate(
        center=1.60
    )

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(candidate,),
    )

    assert (
        plan.tasks[0].bond_length_angstrom
        == pytest.approx(1.50)
    )

    assert (
        plan.tasks[-1].bond_length_angstrom
        == pytest.approx(1.70)
    )


def test_qzvpd_scientific_settings() -> None:
    candidate = make_candidate()

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(candidate,),
    )

    task = plan.tasks[0]

    assert task.basis == "def2-qzvpd"
    assert task.functional == "PBE"
    assert task.grid_level == 4
    assert task.conv_tol == 1.0e-8
    assert task.max_cycle == 250
    assert task.max_memory_mb == 6000


def test_charge_and_spin_are_preserved() -> None:
    candidate = make_candidate(
        charge=-1,
        spin=3,
        functional="B3LYP",
    )

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(candidate,),
    )

    assert all(
        task.charge
        is ChargeState.ANION
        for task in plan.tasks
    )

    assert all(
        task.spin == 3
        for task in plan.tasks
    )

    assert all(
        task.functional == "B3LYP"
        for task in plan.tasks
    )


def test_worker_threads_are_propagated() -> None:
    candidate = make_candidate()

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(candidate,),
        threads_per_worker=2,
    )

    assert all(
        task.threads_per_worker == 2
        for task in plan.tasks
    )


def test_two_candidates_generate_two_grids() -> None:
    neutral = make_candidate(
        charge=0,
        spin=0,
    )

    anion = make_candidate(
        charge=-1,
        spin=1,
    )

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(
            neutral,
            anion,
        ),
    )

    assert plan.candidate_count == 2
    assert plan.task_count == 42


def test_task_ids_are_unique() -> None:
    candidate = make_candidate()

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(candidate,),
    )

    identifiers = [
        task.task_id
        for task in plan.tasks
    ]

    assert len(identifiers) == len(
        set(identifiers)
    )


def test_duplicate_candidate_tasks_are_deduplicated() -> None:
    candidate = make_candidate()

    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(
            candidate,
            candidate,
        ),
    )

    assert plan.candidate_count == 2
    assert plan.task_count == 21


def test_wrong_molecule_is_rejected() -> None:
    candidate = make_candidate()

    with pytest.raises(ValueError):
        build_qzvpd_plan(
            molecule=DiatomicMolecule(
                "Mg",
                "O",
            ),
            candidates=(candidate,),
        )


def test_empty_candidate_list_is_valid() -> None:
    plan = build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(),
    )

    assert plan.candidate_count == 0
    assert plan.task_count == 0
    assert plan.tasks == ()