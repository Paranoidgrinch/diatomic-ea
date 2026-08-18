"""Tests for UKS single-point infrastructure."""

import math

import pytest

from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.schema_f import HARTREE_TO_EV
from diatomic_ea.single_point import (
    SinglePointTask,
    compute_frontier_orbitals,
    format_float_for_id,
    functional_to_xc,
)
from diatomic_ea.states import ChargeState


@pytest.mark.parametrize(
    ("functional", "expected"),
    [
        ("PBE", "PBE"),
        ("pbe", "PBE"),
        ("B3LYP", "B3LYP"),
        ("PBE0", "PBE0"),
        ("TPSSh", "TPSSh"),
        ("TPSH", "TPSSh"),
    ],
)
def test_functional_mapping(
    functional: str,
    expected: str,
) -> None:
    assert functional_to_xc(
        functional
    ) == expected


def test_task_identifier() -> None:
    task = SinglePointTask(
        molecule=DiatomicMolecule("Al", "O"),
        charge=ChargeState.ANION,
        spin=1,
        functional="PBE0",
        basis="def2-tzvpd",
        bond_length_angstrom=1.625,
        grid_level=3,
        conv_tol=1.0e-8,
        max_cycle=200,
        max_memory_mb=4000,
    )

    assert task.task_id == (
        "alo_anion_spin1_"
        "PBE0_def2-tzvpd_r1p625"
    )


def test_multiplicity() -> None:
    task = SinglePointTask(
        molecule=DiatomicMolecule("Fe", "H"),
        charge=ChargeState.NEUTRAL,
        spin=4,
        functional="PBE",
        basis="def2-svp",
        bond_length_angstrom=1.5,
        grid_level=3,
        conv_tol=1.0e-8,
        max_cycle=200,
        max_memory_mb=4000,
    )

    assert task.multiplicity == 5


def test_frontier_orbitals_for_uks() -> None:
    mo_energy = (
        [-0.80, -0.40, 0.20],
        [-0.70, -0.30, 0.25],
    )

    mo_occ = (
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    result = compute_frontier_orbitals(
        mo_energy,
        mo_occ,
    )

    # Occupied alpha orbitals: -0.80, -0.40
    # Occupied beta orbital:    -0.70
    # Therefore HOMO = -0.40 Ha.
    #
    # Virtual alpha orbital:     0.20
    # Virtual beta orbitals:    -0.30, 0.25
    # Therefore LUMO = -0.30 Ha.
    assert result.homo_hartree == pytest.approx(
        -0.40
    )
    assert result.lumo_hartree == pytest.approx(
        -0.30
    )
    assert result.gap_ev == pytest.approx(
        0.10 * HARTREE_TO_EV
    )
    assert not result.positive_homo_warning


def test_positive_homo_warning() -> None:
    result = compute_frontier_orbitals(
        [-0.5, 0.1, 0.2],
        [1.0, 1.0, 0.0],
    )

    assert result.positive_homo_warning


def test_empty_virtual_space_returns_nan() -> None:
    result = compute_frontier_orbitals(
        [-0.5, -0.2],
        [1.0, 1.0],
    )

    assert math.isnan(
        result.lumo_hartree
    )
    assert math.isnan(
        result.gap_ev
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, "1p5"),
        (1.625, "1p625"),
        (2.0, "2"),
        (0.975, "0p975"),
    ],
)
def test_float_identifier_format(
    value: float,
    expected: str,
) -> None:
    assert format_float_for_id(
        value
    ) == expected

def test_mismatched_orbital_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_frontier_orbitals(
            [-0.5, -0.2],
            [1.0],
        )


def test_mismatched_orbital_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_frontier_orbitals(
            (
                [-0.5, 0.2],
                [-0.4, 0.3],
            ),
            [1.0, 0.0],
        )