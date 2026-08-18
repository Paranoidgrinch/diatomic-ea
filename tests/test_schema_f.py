"""Tests for the frozen Schema F specification."""

import dataclasses

import pytest

from diatomic_ea.schema_f import (
    HARTREE_TO_EV,
    SCHEMA_F,
    adiabatic_ea_ev,
)


def test_schema_f_identity() -> None:
    assert SCHEMA_F.schema_id == "schema-f-v1"
    assert (
        SCHEMA_F.electronic_structure_method
        == "UKS-DFT"
    )
    assert (
        SCHEMA_F.reference_pyscf_version
        == "2.13.0"
    )


def test_schema_f_functionals_are_frozen() -> None:
    assert SCHEMA_F.functionals == (
        "PBE",
        "B3LYP",
        "PBE0",
        "TPSSh",
    )


def test_schema_f_fast_bases_are_frozen() -> None:
    assert SCHEMA_F.fast_bases == (
        "def2-svp",
        "def2-tzvp",
        "def2-tzvpp",
        "def2-svpd",
        "def2-tzvpd",
    )


def test_fast_grid_parameters() -> None:
    grid = SCHEMA_F.fast_grid

    assert grid.step_angstrom == 0.025
    assert grid.grid_level == 3
    assert grid.conv_tol == 1.0e-8
    assert grid.max_cycle == 200
    assert grid.max_memory_mb == 4000


def test_qzvpd_parameters() -> None:
    refinement = SCHEMA_F.refinement

    assert refinement.basis == "def2-qzvpd"
    assert refinement.window_angstrom == 0.10
    assert refinement.grid.step_angstrom == 0.01
    assert refinement.grid.grid_level == 4
    assert refinement.grid.max_cycle == 250
    assert refinement.max_spins_per_charge == 2


def test_scf_rescue_parameters() -> None:
    rescue = SCHEMA_F.scf_rescue

    assert rescue.level_shift == 0.25
    assert rescue.newton_max_cycle == 80


def test_schema_is_immutable() -> None:
    with pytest.raises(
        dataclasses.FrozenInstanceError
    ):
        SCHEMA_F.schema_id = "changed"  # type: ignore[misc]


def test_adiabatic_ea_conversion() -> None:
    neutral = -100.0
    anion = -100.1

    result = adiabatic_ea_ev(
        neutral,
        anion,
    )

    assert result == pytest.approx(
        0.1 * HARTREE_TO_EV
    )