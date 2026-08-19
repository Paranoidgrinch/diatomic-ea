"""Tests for the reduced real fast-grid integration harness."""

from diatomic_ea.csv_store import (
    RawResultStore,
    pending_tasks,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.reduced_pipeline_smoke import (
    build_reduced_validation_plan,
    run_reduced_pipeline_validation,
)
from diatomic_ea.runner import (
    FastGridRunSummary,
)
from diatomic_ea.schema_f import (
    HARTREE_TO_EV,
)
from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointStatus,
)
from diatomic_ea.states import (
    ChargeState,
)


def synthetic_result(
    task,
) -> SinglePointResult:
    electron_count = (
        18
        if task.charge
        is ChargeState.NEUTRAL
        else 19
    )

    alpha = (
        electron_count
        + task.spin
    ) // 2

    beta = (
        electron_count
        - task.spin
    ) // 2

    energy = (
        -198.0
        + 0.2
        * (
            task.bond_length_angstrom
            - 1.42
        ) ** 2
    )

    if (
        task.charge
        is ChargeState.ANION
    ):
        energy -= 0.10

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
        basis_label_a="pyscf:def2-SVP",
        basis_label_b="pyscf:def2-SVP",
        ecp_label_a="",
        ecp_label_b="",
        frontier=None,
        s2=0.0,
        observed_multiplicity=(
            task.multiplicity
        ),
        spin_contamination_warning=False,
        pyscf_version="2.13.0",
        elapsed_seconds=0.01,
    )


def fake_resumable_runner(
    plan,
    *,
    store,
    max_workers,
    reporter=None,
    retry_errors=True,
):
    remaining = pending_tasks(
        plan.tasks,
        store,
        retry_errors=retry_errors,
    )

    already_finished = (
        plan.task_count
        - len(
            remaining
        )
    )

    for task in remaining:
        store.append(
            task,
            synthetic_result(
                task
            ),
        )

    return FastGridRunSummary(
        total_planned=plan.task_count,
        already_finished=(
            already_finished
        ),
        attempted=len(
            remaining
        ),
        completed_ok=len(
            remaining
        ),
        completed_error=0,
        remaining_after_run=0,
    )


def test_reduced_plan_has_four_tasks() -> None:
    plan = (
        build_reduced_validation_plan(
            neutral_electrons=18,
            anion_electrons=19,
        )
    )

    assert plan.task_count == 4

    assert {
        task.charge
        for task in plan.tasks
    } == {
        ChargeState.NEUTRAL,
        ChargeState.ANION,
    }

    neutral_spins = {
        task.spin
        for task in plan.tasks
        if task.charge
        is ChargeState.NEUTRAL
    }

    anion_spins = {
        task.spin
        for task in plan.tasks
        if task.charge
        is ChargeState.ANION
    }

    assert neutral_spins == {
        0
    }

    assert anion_spins == {
        1
    }

    assert {
        task.bond_length_angstrom
        for task in plan.tasks
    } == {
        1.40,
        1.44,
    }


def test_reduced_plan_uses_validation_method() -> None:
    plan = (
        build_reduced_validation_plan(
            neutral_electrons=18,
            anion_electrons=19,
        )
    )

    assert {
        task.functional
        for task in plan.tasks
    } == {
        "PBE"
    }

    assert {
        task.basis
        for task in plan.tasks
    } == {
        "def2-SVP"
    }

    assert all(
        task.threads_per_worker
        == 1
        for task in plan.tasks
    )


def test_reduced_validation_persists_and_resumes(
    tmp_path,
    monkeypatch,
) -> None:
    counts = iter(
        (
            18,
            19,
        )
    )

    def fake_count(
        **kwargs,
    ):
        return next(
            counts
        )

    monkeypatch.setattr(
        "diatomic_ea.reduced_pipeline_smoke."
        "run_platform_electron_count",
        fake_count,
    )

    monkeypatch.setattr(
        "diatomic_ea.reduced_pipeline_smoke."
        "execute_fast_grid_resumable",
        fake_resumable_runner,
    )

    report = (
        run_reduced_pipeline_validation(
            output_directory=tmp_path,
            max_workers=2,
        )
    )

    assert report.passed

    assert (
        report.neutral_electrons
        == 18
    )

    assert (
        report.anion_electrons
        == 19
    )

    assert (
        report.task_count
        == 4
    )

    assert (
        report.first_attempted
        == 4
    )

    assert (
        report.first_completed_ok
        == 4
    )

    assert (
        report.first_remaining
        == 0
    )

    assert (
        report.csv_rows
        == 4
    )

    assert (
        report.resume_attempted
        == 0
    )

    assert (
        report.resume_already_finished
        == 4
    )

    store = RawResultStore(
        report.csv_path
    )

    assert len(
        store.latest_rows()
    ) == 4


def test_reduced_validation_rejects_bad_electron_counts(
    tmp_path,
    monkeypatch,
) -> None:
    counts = iter(
        (
            18,
            18,
        )
    )

    monkeypatch.setattr(
        "diatomic_ea.reduced_pipeline_smoke."
        "run_platform_electron_count",
        lambda **kwargs: next(
            counts
        ),
    )

    try:
        run_reduced_pipeline_validation(
            output_directory=tmp_path,
        )

    except RuntimeError as exc:
        assert (
            "one greater"
            in str(
                exc
            )
        )

    else:
        raise AssertionError(
            "Expected invalid electron counts "
            "to be rejected."
        )
