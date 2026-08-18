"""Tests for resumable raw-result CSV storage."""

import csv
import math

import pytest

from diatomic_ea.csv_store import (
    RAW_RESULT_COLUMNS,
    RawResultStore,
    pending_tasks,
)
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.single_point import (
    FrontierOrbitals,
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.states import ChargeState


def make_task(
    bond_length: float,
) -> SinglePointTask:
    return SinglePointTask(
        molecule=DiatomicMolecule("Al", "O"),
        charge=ChargeState.NEUTRAL,
        spin=0,
        functional="PBE",
        basis="def2-svp",
        bond_length_angstrom=bond_length,
        grid_level=3,
        conv_tol=1.0e-8,
        max_cycle=200,
        max_memory_mb=4000,
    )


def make_result(
    task: SinglePointTask,
    *,
    status: SinglePointStatus = SinglePointStatus.OK,
    energy: float = -100.0,
) -> SinglePointResult:
    frontier = FrontierOrbitals(
        homo_hartree=-0.2,
        lumo_hartree=-0.1,
        homo_ev=-5.4,
        lumo_ev=-2.7,
        gap_ev=2.7,
        positive_homo_warning=False,
    )

    return SinglePointResult(
        task_id=task.task_id,
        status=status,
        error=(
            ""
            if status is SinglePointStatus.OK
            else "test failure"
        ),
        energy_hartree=energy,
        energy_ev=energy * 27.211386245988,
        converged=(
            status is SinglePointStatus.OK
        ),
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=20,
        alpha_electrons=10,
        beta_electrons=10,
        basis_label_a="pyscf:def2-svp",
        basis_label_b="pyscf:def2-svp",
        ecp_label_a="",
        ecp_label_b="",
        frontier=frontier,
        s2=0.0,
        observed_multiplicity=1.0,
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=0.5,
    )


def test_append_creates_csv_with_header(
    tmp_path,
) -> None:
    task = make_task(1.5)
    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    store.append(
        task,
        make_result(task),
    )

    with store.path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == (
        RAW_RESULT_COLUMNS
    )
    assert len(rows) == 1
    assert rows[0]["task_id"] == task.task_id
    assert rows[0]["molecule"] == "AlO"


def test_multiple_results_are_appended(
    tmp_path,
) -> None:
    first = make_task(1.5)
    second = make_task(1.525)

    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    store.append(
        first,
        make_result(first),
    )
    store.append(
        second,
        make_result(second),
    )

    assert store.recorded_task_ids() == {
        first.task_id,
        second.task_id,
    }


def test_latest_row_wins_for_retried_task(
    tmp_path,
) -> None:
    task = make_task(1.5)

    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    store.append(
        task,
        make_result(
            task,
            status=SinglePointStatus.ERROR,
            energy=math.nan,
        ),
    )

    store.append(
        task,
        make_result(
            task,
            status=SinglePointStatus.OK,
            energy=-101.0,
        ),
    )

    latest = store.latest_rows()[
        task.task_id
    ]

    assert latest["status"] == "ok"
    assert float(
        latest["energy_hartree"]
    ) == pytest.approx(-101.0)


def test_errors_are_retried_by_default(
    tmp_path,
) -> None:
    first = make_task(1.5)
    second = make_task(1.525)

    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    store.append(
        first,
        make_result(
            first,
            status=SinglePointStatus.ERROR,
        ),
    )

    remaining = pending_tasks(
        [first, second],
        store,
    )

    assert remaining == (
        first,
        second,
    )


def test_successful_tasks_are_skipped(
    tmp_path,
) -> None:
    first = make_task(1.5)
    second = make_task(1.525)

    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    store.append(
        first,
        make_result(first),
    )

    remaining = pending_tasks(
        [first, second],
        store,
    )

    assert remaining == (second,)


def test_errors_can_be_considered_terminal(
    tmp_path,
) -> None:
    task = make_task(1.5)

    store = RawResultStore(
        tmp_path / "raw.csv"
    )

    store.append(
        task,
        make_result(
            task,
            status=SinglePointStatus.ERROR,
        ),
    )

    remaining = pending_tasks(
        [task],
        store,
        retry_errors=False,
    )

    assert remaining == ()