"""Tests for QZVPD scientific analysis."""

import pytest

from diatomic_ea.csv_store import RawResultStore
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.qzvpd import (
    QZVPDPlan,
    build_qzvpd_plan,
)
from diatomic_ea.qzvpd_analysis import analyze_qzvpd
from diatomic_ea.refinement import QZVPDCandidate
from diatomic_ea.schema_f import HARTREE_TO_EV
from diatomic_ea.single_point import (
    FrontierOrbitals,
    SinglePointResult,
    SinglePointStatus,
)


def candidate(
    *,
    charge: int,
    spin: int,
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
        method_minimum_count_for_spin=4,
        source_warnings=(),
    )


def make_plan(
    *candidates: QZVPDCandidate,
) -> QZVPDPlan:
    return build_qzvpd_plan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=tuple(candidates),
    )


def find_task(
    plan: QZVPDPlan,
    *,
    charge: int,
    spin: int,
    functional: str,
    r: float,
):
    return next(
        task
        for task in plan.tasks
        if (
            int(task.charge) == charge
            and task.spin == spin
            and task.functional == functional
            and abs(
                task.bond_length_angstrom
                - r
            ) < 1.0e-9
        )
    )


def write_result(
    store: RawResultStore,
    task,
    *,
    energy: float,
    converged: bool = True,
    positive_homo: bool = False,
    spin_warning: bool = False,
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
        status=SinglePointStatus.OK,
        error="",
        energy_hartree=energy,
        energy_ev=energy * HARTREE_TO_EV,
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


def test_qzvpd_state_minimum(
    tmp_path,
) -> None:
    neutral = candidate(
        charge=0,
        spin=0,
    )

    plan = make_plan(neutral)

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    first = find_task(
        plan,
        charge=0,
        spin=0,
        functional="PBE",
        r=1.59,
    )

    second = find_task(
        plan,
        charge=0,
        spin=0,
        functional="PBE",
        r=1.60,
    )

    write_result(
        store,
        first,
        energy=-100.0,
    )

    write_result(
        store,
        second,
        energy=-101.0,
    )

    analysis = analyze_qzvpd(
        store,
        plan,
    )

    assert len(
        analysis.state_minima
    ) == 1

    minimum = analysis.state_minima[0]

    assert (
        minimum.point.bond_length_angstrom
        == pytest.approx(1.60)
    )

    assert not minimum.grid_edge_warning


def test_qzvpd_selects_lowest_spin(
    tmp_path,
) -> None:
    spin0 = candidate(
        charge=0,
        spin=0,
    )

    spin2 = candidate(
        charge=0,
        spin=2,
    )

    plan = make_plan(
        spin0,
        spin2,
    )

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    task0 = find_task(
        plan,
        charge=0,
        spin=0,
        functional="PBE",
        r=1.60,
    )

    task2 = find_task(
        plan,
        charge=0,
        spin=2,
        functional="PBE",
        r=1.60,
    )

    write_result(
        store,
        task0,
        energy=-100.0,
    )

    write_result(
        store,
        task2,
        energy=-101.0,
    )

    analysis = analyze_qzvpd(
        store,
        plan,
    )

    assert (
        analysis.charge_minima[0].spin
        == 2
    )


def test_qzvpd_ea(
    tmp_path,
) -> None:
    neutral = candidate(
        charge=0,
        spin=0,
    )

    anion = candidate(
        charge=-1,
        spin=1,
    )

    plan = make_plan(
        neutral,
        anion,
    )

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    neutral_task = find_task(
        plan,
        charge=0,
        spin=0,
        functional="PBE",
        r=1.60,
    )

    anion_task = find_task(
        plan,
        charge=-1,
        spin=1,
        functional="PBE",
        r=1.60,
    )

    write_result(
        store,
        neutral_task,
        energy=-100.0,
    )

    write_result(
        store,
        anion_task,
        energy=-100.1,
    )

    analysis = analyze_qzvpd(
        store,
        plan,
    )

    assert len(
        analysis.functional_eas
    ) == 1

    result = analysis.functional_eas[0]

    assert result.ea_ev == pytest.approx(
        0.1 * HARTREE_TO_EV
    )

    assert result.recommended_for_summary


def test_qzvpd_edge_is_hard_warning(
    tmp_path,
) -> None:
    neutral = candidate(
        charge=0,
        spin=0,
    )

    plan = make_plan(neutral)

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    edge_task = find_task(
        plan,
        charge=0,
        spin=0,
        functional="PBE",
        r=1.50,
    )

    write_result(
        store,
        edge_task,
        energy=-101.0,
    )

    analysis = analyze_qzvpd(
        store,
        plan,
    )

    minimum = analysis.state_minima[0]

    assert minimum.grid_edge_warning

    assert (
        "minimum_at_qzvpd_edge"
        in minimum.hard_warnings
    )


def test_qzvpd_nonconvergence_is_hard_warning(
    tmp_path,
) -> None:
    neutral = candidate(
        charge=0,
        spin=0,
    )

    plan = make_plan(neutral)

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    task = find_task(
        plan,
        charge=0,
        spin=0,
        functional="PBE",
        r=1.60,
    )

    write_result(
        store,
        task,
        energy=-101.0,
        converged=False,
    )

    analysis = analyze_qzvpd(
        store,
        plan,
    )

    assert (
        "not_converged"
        in analysis
        .state_minima[0]
        .hard_warnings
    )


def test_qzvpd_diagnostics_are_not_hard_failures(
    tmp_path,
) -> None:
    neutral = candidate(
        charge=0,
        spin=0,
    )

    anion = candidate(
        charge=-1,
        spin=1,
    )

    plan = make_plan(
        neutral,
        anion,
    )

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    neutral_task = find_task(
        plan,
        charge=0,
        spin=0,
        functional="PBE",
        r=1.60,
    )

    anion_task = find_task(
        plan,
        charge=-1,
        spin=1,
        functional="PBE",
        r=1.60,
    )

    write_result(
        store,
        neutral_task,
        energy=-100.0,
        positive_homo=True,
    )

    write_result(
        store,
        anion_task,
        energy=-100.1,
        spin_warning=True,
    )

    analysis = analyze_qzvpd(
        store,
        plan,
    )

    result = analysis.functional_eas[0]

    assert (
        "positive_HOMO"
        in result.diagnostic_warnings
    )

    assert (
        "spin_contamination"
        in result.diagnostic_warnings
    )

    assert result.recommended_for_summary