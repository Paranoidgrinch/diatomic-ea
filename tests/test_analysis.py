"""Tests for fast-grid scientific analysis."""

import pytest

from diatomic_ea.analysis import analyze_fast_grid
from diatomic_ea.csv_store import RawResultStore
from diatomic_ea.grid import BondGrid
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.schema_f import HARTREE_TO_EV
from diatomic_ea.single_point import (
    FrontierOrbitals,
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.states import ChargeState


def make_task(
    *,
    charge: ChargeState,
    spin: int,
    r: float,
    functional: str = "PBE",
    basis: str = "def2-svp",
) -> SinglePointTask:
    return SinglePointTask(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        charge=charge,
        spin=spin,
        functional=functional,
        basis=basis,
        bond_length_angstrom=r,
        grid_level=3,
        conv_tol=1.0e-8,
        max_cycle=200,
        max_memory_mb=4000,
    )


def write_point(
    store: RawResultStore,
    task: SinglePointTask,
    *,
    energy: float,
    converged: bool = True,
    positive_homo: bool = False,
    spin_warning: bool = False,
    status: SinglePointStatus = SinglePointStatus.OK,
) -> None:
    frontier = FrontierOrbitals(
        homo_hartree=(
            0.01
            if positive_homo
            else -0.2
        ),
        lumo_hartree=0.1,
        homo_ev=(
            0.272
            if positive_homo
            else -5.4
        ),
        lumo_ev=2.72,
        gap_ev=2.4,
        positive_homo_warning=positive_homo,
    )

    result = SinglePointResult(
        task_id=task.task_id,
        status=status,
        error=(
            ""
            if status is SinglePointStatus.OK
            else "test error"
        ),
        energy_hartree=energy,
        energy_ev=(
            energy * HARTREE_TO_EV
        ),
        converged=converged,
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=20,
        alpha_electrons=10,
        beta_electrons=10,
        basis_label_a="test",
        basis_label_b="test",
        ecp_label_a="",
        ecp_label_b="",
        frontier=frontier,
        s2=0.0,
        observed_multiplicity=(
            task.multiplicity
        ),
        spin_contamination_warning=spin_warning,
        pyscf_version="2.13.0",
        elapsed_seconds=0.1,
    )

    store.append(
        task,
        result,
    )


def make_grid() -> BondGrid:
    return BondGrid(
        minimum_angstrom=1.0,
        maximum_angstrom=1.05,
        step_angstrom=0.025,
    )


def test_state_minimum_finds_lowest_geometry(
    tmp_path,
) -> None:
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    first = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.0,
    )

    second = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.025,
    )

    write_point(
        store,
        first,
        energy=-100.0,
    )

    write_point(
        store,
        second,
        energy=-101.0,
    )

    analysis = analyze_fast_grid(
        store,
        make_grid(),
    )

    assert len(
        analysis.state_minima
    ) == 1

    minimum = analysis.state_minima[0]

    assert (
        minimum.point.bond_length_angstrom
        == pytest.approx(1.025)
    )

    assert (
        minimum.point.energy_hartree
        == pytest.approx(-101.0)
    )

    assert not minimum.grid_edge_warning


def test_charge_minimum_selects_lowest_spin(
    tmp_path,
) -> None:
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    singlet = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.025,
    )

    triplet = make_task(
        charge=ChargeState.NEUTRAL,
        spin=2,
        r=1.025,
    )

    write_point(
        store,
        singlet,
        energy=-100.0,
    )

    write_point(
        store,
        triplet,
        energy=-101.0,
    )

    analysis = analyze_fast_grid(
        store,
        make_grid(),
    )

    assert len(
        analysis.charge_minima
    ) == 1

    assert (
        analysis.charge_minima[0].spin
        == 2
    )


def test_grid_edge_is_hard_warning(
    tmp_path,
) -> None:
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    task = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.0,
    )

    write_point(
        store,
        task,
        energy=-100.0,
    )

    analysis = analyze_fast_grid(
        store,
        make_grid(),
    )

    minimum = analysis.state_minima[0]

    assert minimum.grid_edge_warning

    assert (
        "minimum_at_grid_edge"
        in minimum.hard_warnings
    )


def test_ea_uses_neutral_minus_anion_energy(
    tmp_path,
) -> None:
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    neutral = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.025,
    )

    anion = make_task(
        charge=ChargeState.ANION,
        spin=1,
        r=1.025,
    )

    write_point(
        store,
        neutral,
        energy=-100.0,
    )

    write_point(
        store,
        anion,
        energy=-100.1,
    )

    analysis = analyze_fast_grid(
        store,
        make_grid(),
    )

    assert len(
        analysis.method_eas
    ) == 1

    result = analysis.method_eas[0]

    assert result.ea_ev == pytest.approx(
        0.1 * HARTREE_TO_EV
    )

    assert (
        result.recommended_for_fast_summary
    )


def test_diagnostics_do_not_exclude_method(
    tmp_path,
) -> None:
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    neutral = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.025,
    )

    anion = make_task(
        charge=ChargeState.ANION,
        spin=1,
        r=1.025,
    )

    write_point(
        store,
        neutral,
        energy=-100.0,
        positive_homo=True,
    )

    write_point(
        store,
        anion,
        energy=-100.1,
        spin_warning=True,
    )

    analysis = analyze_fast_grid(
        store,
        make_grid(),
    )

    result = analysis.method_eas[0]

    assert (
        "positive_HOMO"
        in result.diagnostic_warnings
    )

    assert (
        "spin_contamination"
        in result.diagnostic_warnings
    )

    assert (
        result.recommended_for_fast_summary
    )


def test_nonconvergence_excludes_method(
    tmp_path,
) -> None:
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    neutral = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.025,
    )

    anion = make_task(
        charge=ChargeState.ANION,
        spin=1,
        r=1.025,
    )

    write_point(
        store,
        neutral,
        energy=-100.0,
        converged=False,
    )

    write_point(
        store,
        anion,
        energy=-100.1,
    )

    analysis = analyze_fast_grid(
        store,
        make_grid(),
    )

    result = analysis.method_eas[0]

    assert (
        "not_converged"
        in result.hard_warnings
    )

    assert not (
        result.recommended_for_fast_summary
    )


def test_error_result_is_ignored(
    tmp_path,
) -> None:
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    bad = make_task(
        charge=ChargeState.NEUTRAL,
        spin=0,
        r=1.025,
    )

    write_point(
        store,
        bad,
        energy=-999.0,
        status=SinglePointStatus.ERROR,
    )

    analysis = analyze_fast_grid(
        store,
        make_grid(),
    )

    assert analysis.points
    assert analysis.state_minima == ()
    assert analysis.charge_minima == ()
    assert analysis.method_eas == ()