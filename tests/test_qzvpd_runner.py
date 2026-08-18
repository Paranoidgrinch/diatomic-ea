"""Tests for resumable QZVPD execution."""

from unittest.mock import patch

from diatomic_ea.csv_store import RawResultStore
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.progress import CalculationStage
from diatomic_ea.qzvpd import QZVPDPlan
from diatomic_ea.runner import execute_qzvpd_resumable
from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.states import ChargeState


def make_task(
    bond_length: float,
) -> SinglePointTask:
    return SinglePointTask(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        charge=ChargeState.NEUTRAL,
        spin=0,
        functional="PBE",
        basis="def2-qzvpd",
        bond_length_angstrom=bond_length,
        grid_level=4,
        conv_tol=1.0e-8,
        max_cycle=250,
        max_memory_mb=6000,
    )


def make_result(
    task: SinglePointTask,
) -> SinglePointResult:
    return SinglePointResult(
        task_id=task.task_id,
        status=SinglePointStatus.OK,
        error="",
        energy_hartree=-100.0,
        energy_ev=-2721.1386245988,
        converged=True,
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=20,
        alpha_electrons=10,
        beta_electrons=10,
        basis_label_a="test",
        basis_label_b="test",
        ecp_label_a="",
        ecp_label_b="",
        frontier=None,
        s2=0.0,
        observed_multiplicity=1.0,
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=0.1,
    )


def make_plan(
    tasks: tuple[SinglePointTask, ...],
) -> QZVPDPlan:
    return QZVPDPlan(
        molecule=DiatomicMolecule(
            "Al",
            "O",
        ),
        candidates=(),
        tasks=tasks,
    )


def fake_execute(
    items,
    *,
    worker,
    max_workers,
    reporter=None,
    stage=None,
    result_callback=None,
):
    results = []

    for task in items:
        result = make_result(task)

        if result_callback is not None:
            result_callback(
                task,
                result,
            )

        results.append(result)

    return tuple(results)


def test_qzvpd_run_persists_results(
    tmp_path,
) -> None:
    first = make_task(1.50)
    second = make_task(1.51)

    plan = make_plan(
        (
            first,
            second,
        )
    )

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    with patch(
        "diatomic_ea.runner.execute_process_batch",
        side_effect=fake_execute,
    ):
        summary = execute_qzvpd_resumable(
            plan,
            store=store,
            max_workers=2,
        )

    assert summary.total_planned == 2
    assert summary.already_finished == 0
    assert summary.attempted == 2
    assert summary.completed_ok == 2
    assert summary.completed_error == 0
    assert summary.remaining_after_run == 0
    assert summary.complete

    assert store.finished_task_ids() == {
        first.task_id,
        second.task_id,
    }


def test_qzvpd_resume_skips_finished_task(
    tmp_path,
) -> None:
    first = make_task(1.50)
    second = make_task(1.51)

    plan = make_plan(
        (
            first,
            second,
        )
    )

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    store.append(
        first,
        make_result(first),
    )

    seen = []

    def record_execute(
        items,
        **kwargs,
    ):
        seen.extend(items)

        return fake_execute(
            items,
            **kwargs,
        )

    with patch(
        "diatomic_ea.runner.execute_process_batch",
        side_effect=record_execute,
    ):
        summary = execute_qzvpd_resumable(
            plan,
            store=store,
            max_workers=2,
        )

    assert seen == [second]
    assert summary.already_finished == 1
    assert summary.attempted == 1
    assert summary.complete


def test_completed_qzvpd_plan_runs_no_workers(
    tmp_path,
) -> None:
    task = make_task(1.50)

    plan = make_plan(
        (task,)
    )

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    store.append(
        task,
        make_result(task),
    )

    with patch(
        "diatomic_ea.runner.execute_process_batch"
    ) as mocked:
        summary = execute_qzvpd_resumable(
            plan,
            store=store,
            max_workers=2,
        )

    mocked.assert_not_called()

    assert summary.complete
    assert summary.attempted == 0
    assert summary.already_finished == 1


def test_qzvpd_uses_refinement_progress_stage(
    tmp_path,
) -> None:
    task = make_task(1.50)

    plan = make_plan(
        (task,)
    )

    store = RawResultStore(
        tmp_path / "qzvpd.csv"
    )

    captured = {}

    def capture_execute(
        items,
        **kwargs,
    ):
        captured["stage"] = kwargs[
            "stage"
        ]

        return fake_execute(
            items,
            **kwargs,
        )

    with patch(
        "diatomic_ea.runner.execute_process_batch",
        side_effect=capture_execute,
    ):
        execute_qzvpd_resumable(
            plan,
            store=store,
            max_workers=1,
        )

    assert (
        captured["stage"]
        is CalculationStage.QZVPD_REFINEMENT
    )