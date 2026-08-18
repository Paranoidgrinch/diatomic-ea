"""Tests for the complete Schema F pipeline."""

from statistics import median
from unittest.mock import patch

import pytest

from diatomic_ea.csv_store import pending_tasks
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.pipeline import (
    PipelineRequest,
    run_schema_f_pipeline,
)
from diatomic_ea.runner import (
    FastGridRunSummary,
    QZVPDRunSummary,
)
from diatomic_ea.schema_f import HARTREE_TO_EV
from diatomic_ea.single_point import (
    FrontierOrbitals,
    SinglePointResult,
    SinglePointStatus,
)
from diatomic_ea.states import ChargeState


EA_HARTREE = {
    "PBE": 0.095,
    "B3LYP": 0.100,
    "PBE0": 0.105,
    "TPSSh": 0.110,
}


FUNCTIONAL_INDEX = {
    "PBE": 0,
    "B3LYP": 1,
    "PBE0": 2,
    "TPSSh": 3,
}


BASIS_INDEX = {
    "def2-svp": 0,
    "def2-tzvp": 1,
    "def2-tzvpp": 2,
    "def2-svpd": 3,
    "def2-tzvpd": 4,
    "def2-qzvpd": 5,
}


def electron_count_resolver(
    *,
    molecule,
    charge,
    basis,
    bond_length_angstrom,
    max_memory_mb,
) -> int:
    if charge is ChargeState.NEUTRAL:
        return 10

    if charge is ChargeState.ANION:
        return 11

    raise ValueError(
        "Unsupported charge."
    )


def synthetic_result(
    task,
) -> SinglePointResult:
    functional_index = FUNCTIONAL_INDEX[
        task.functional
    ]

    basis_index = BASIS_INDEX.get(
        task.basis,
        0,
    )

    target_r = 1.525

    energy = (
        -100.0
        - 0.01 * functional_index
        - 0.001 * basis_index
        + 0.5
        * (
            task.bond_length_angstrom
            - target_r
        ) ** 2
    )

    if task.charge is ChargeState.ANION:
        energy -= EA_HARTREE[
            task.functional
        ]

    electron_count = (
        10
        if task.charge
        is ChargeState.NEUTRAL
        else 11
    )

    alpha = (
        electron_count
        + task.spin
    ) // 2

    beta = (
        electron_count
        - task.spin
    ) // 2

    s = (
        task.spin / 2.0
    )

    return SinglePointResult(
        task_id=task.task_id,
        status=SinglePointStatus.OK,
        error="",
        energy_hartree=energy,
        energy_ev=(
            energy
            * HARTREE_TO_EV
        ),
        converged=True,
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=electron_count,
        alpha_electrons=alpha,
        beta_electrons=beta,
        basis_label_a="synthetic",
        basis_label_b="synthetic",
        ecp_label_a="",
        ecp_label_b="",
        frontier=FrontierOrbitals(
            homo_hartree=-0.2,
            lumo_hartree=0.1,
            homo_ev=(
                -0.2
                * HARTREE_TO_EV
            ),
            lumo_ev=(
                0.1
                * HARTREE_TO_EV
            ),
            gap_ev=(
                0.3
                * HARTREE_TO_EV
            ),
            positive_homo_warning=False,
        ),
        s2=(
            s * (s + 1.0)
        ),
        observed_multiplicity=(
            task.multiplicity
        ),
        spin_contamination_warning=False,
        pyscf_version="synthetic-test",
        elapsed_seconds=0.001,
    )


def fake_fast_runner(
    plan,
    *,
    store,
    max_workers,
    reporter=None,
    retry_errors=True,
    worker,
):
    remaining = pending_tasks(
        plan.tasks,
        store,
        retry_errors=retry_errors,
    )

    already_finished = (
        plan.task_count
        - len(remaining)
    )

    for task in remaining:
        store.append(
            task,
            worker(task),
        )

    return FastGridRunSummary(
        total_planned=plan.task_count,
        already_finished=already_finished,
        attempted=len(remaining),
        completed_ok=len(remaining),
        completed_error=0,
        remaining_after_run=0,
    )


def fake_qzvpd_runner(
    plan,
    *,
    store,
    max_workers,
    reporter=None,
    retry_errors=True,
    worker,
):
    remaining = pending_tasks(
        plan.tasks,
        store,
        retry_errors=retry_errors,
    )

    already_finished = (
        plan.task_count
        - len(remaining)
    )

    for task in remaining:
        store.append(
            task,
            worker(task),
        )

    return QZVPDRunSummary(
        total_planned=plan.task_count,
        already_finished=already_finished,
        attempted=len(remaining),
        completed_ok=len(remaining),
        completed_error=0,
        remaining_after_run=0,
    )


def request(
    run_id: str,
) -> PipelineRequest:
    return PipelineRequest(
        molecule=DiatomicMolecule(
            "H",
            "F",
        ),
        minimum_angstrom=1.50,
        maximum_angstrom=1.55,
        spin_max=1,
        workers=2,
        threads_per_worker=1,
        run_id=run_id,
    )


def run_synthetic_pipeline(
    tmp_path,
    run_id: str,
):
    with (
        patch(
            "diatomic_ea.pipeline.execute_fast_grid_resumable",
            side_effect=fake_fast_runner,
        ),
        patch(
            "diatomic_ea.pipeline.execute_qzvpd_resumable",
            side_effect=fake_qzvpd_runner,
        ),
    ):
        return run_schema_f_pipeline(
            request(run_id),
            output_root=tmp_path,
            electron_count_resolver=(
                electron_count_resolver
            ),
            worker=synthetic_result,
        )


def test_complete_pipeline_produces_result_files(
    tmp_path,
) -> None:
    result = run_synthetic_pipeline(
        tmp_path,
        "pipeline-test",
    )

    assert result.neutral_electrons == 10
    assert result.anion_electrons == 11

    assert result.fast_run.complete
    assert result.qzvpd_run.complete

    assert (
        result.estimate.functional_count
        == 4
    )

    assert (
        result.paths.fast_grid_csv
        .exists()
    )

    assert (
        result.paths.qzvpd_csv
        .exists()
    )

    assert (
        result.paths.final_result_csv
        .exists()
    )

    assert (
        result.paths.manifest_json
        .exists()
    )


def test_pipeline_produces_expected_task_counts(
    tmp_path,
) -> None:
    result = run_synthetic_pipeline(
        tmp_path,
        "task-counts",
    )

    assert result.fast_plan.task_count == 120

    assert result.qzvpd_plan.task_count == 168


def test_pipeline_produces_four_qzvpd_eas(
    tmp_path,
) -> None:
    result = run_synthetic_pipeline(
        tmp_path,
        "four-eas",
    )

    assert {
        item.functional
        for item
        in result.qzvpd_analysis.functional_eas
    } == {
        "PBE",
        "B3LYP",
        "PBE0",
        "TPSSh",
    }


def test_pipeline_wires_schema_f_statistics(
    tmp_path,
) -> None:
    result = run_synthetic_pipeline(
        tmp_path,
        "statistics",
    )

    qz_eas = [
        value * HARTREE_TO_EV
        for value
        in EA_HARTREE.values()
    ]

    expected = (
        median(qz_eas)
        + 0.0825
    )

    assert (
        result.estimate.predicted_ea_ev
        == pytest.approx(expected)
    )


def test_pipeline_is_resumable(
    tmp_path,
) -> None:
    first = run_synthetic_pipeline(
        tmp_path,
        "resume-test",
    )

    second = run_synthetic_pipeline(
        tmp_path,
        "resume-test",
    )

    assert first.paths.run_dir == (
        second.paths.run_dir
    )

    assert (
        second.fast_run.already_finished
        == second.fast_plan.task_count
    )

    assert second.fast_run.attempted == 0

    assert (
        second.qzvpd_run.already_finished
        == second.qzvpd_plan.task_count
    )

    assert second.qzvpd_run.attempted == 0


def test_invalid_electron_count_difference_fails(
    tmp_path,
) -> None:
    def bad_resolver(
        **kwargs,
    ):
        return 10

    with pytest.raises(
        RuntimeError,
        match="exactly one greater",
    ):
        run_schema_f_pipeline(
            request(
                "bad-electrons"
            ),
            output_root=tmp_path,
            electron_count_resolver=(
                bad_resolver
            ),
            worker=synthetic_result,
        )


def test_invalid_pipeline_worker_count() -> None:
    with pytest.raises(ValueError):
        PipelineRequest(
            molecule=DiatomicMolecule(
                "H",
                "F",
            ),
            minimum_angstrom=1.50,
            maximum_angstrom=1.55,
            spin_max=1,
            workers=0,
        )