"""High-level resumable calculation runners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from diatomic_ea.csv_store import (
    RawResultStore,
    pending_tasks,
)
from diatomic_ea.executor import execute_process_batch
from diatomic_ea.grid import FastGridPlan
from diatomic_ea.progress import (
    CalculationStage,
    ProgressReporter,
)
from diatomic_ea.qzvpd import QZVPDPlan
from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointStatus,
    SinglePointTask,
)
from diatomic_ea.single_point_adapter import (
    run_platform_single_point,
)


@dataclass(frozen=True, slots=True)
class FastGridRunSummary:
    """Summary of one resumable fast-grid execution."""

    total_planned: int
    already_finished: int
    attempted: int
    completed_ok: int
    completed_error: int
    remaining_after_run: int

    @property
    def complete(self) -> bool:
        return self.remaining_after_run == 0


@dataclass(frozen=True, slots=True)
class QZVPDRunSummary:
    """Summary of one resumable QZVPD execution."""

    total_planned: int
    already_finished: int
    attempted: int
    completed_ok: int
    completed_error: int
    remaining_after_run: int

    @property
    def complete(self) -> bool:
        return self.remaining_after_run == 0


def _count_statuses(
    results: tuple[SinglePointResult, ...],
) -> tuple[int, int]:
    completed_ok = sum(
        result.status
        is SinglePointStatus.OK
        for result in results
    )

    return (
        completed_ok,
        len(results) - completed_ok,
    )


def execute_fast_grid_resumable(
    plan: FastGridPlan,
    *,
    store: RawResultStore,
    max_workers: int,
    reporter: ProgressReporter | None = None,
    retry_errors: bool = True,
    worker: Callable[
        [SinglePointTask],
        SinglePointResult,
    ] = run_platform_single_point,
) -> FastGridRunSummary:
    """Execute unfinished fast-grid tasks and persist each result."""
    total = plan.task_count

    remaining = pending_tasks(
        plan.tasks,
        store,
        retry_errors=retry_errors,
    )

    already_finished = (
        total - len(remaining)
    )

    if not remaining:
        return FastGridRunSummary(
            total_planned=total,
            already_finished=already_finished,
            attempted=0,
            completed_ok=0,
            completed_error=0,
            remaining_after_run=0,
        )

    def persist_result(
        task: SinglePointTask,
        result: SinglePointResult,
    ) -> None:
        store.append(
            task,
            result,
        )

    results = execute_process_batch(
        remaining,
        worker=worker,
        max_workers=max_workers,
        reporter=reporter,
        stage=CalculationStage.FAST_GRID,
        result_callback=persist_result,
    )

    completed_ok, completed_error = (
        _count_statuses(results)
    )

    remaining_after = len(
        pending_tasks(
            plan.tasks,
            store,
            retry_errors=retry_errors,
        )
    )

    return FastGridRunSummary(
        total_planned=total,
        already_finished=already_finished,
        attempted=len(results),
        completed_ok=completed_ok,
        completed_error=completed_error,
        remaining_after_run=remaining_after,
    )


def execute_qzvpd_resumable(
    plan: QZVPDPlan,
    *,
    store: RawResultStore,
    max_workers: int,
    reporter: ProgressReporter | None = None,
    retry_errors: bool = True,
    worker: Callable[
        [SinglePointTask],
        SinglePointResult,
    ] = run_platform_single_point,
) -> QZVPDRunSummary:
    """Execute unfinished QZVPD tasks and persist each result."""
    total = plan.task_count

    remaining = pending_tasks(
        plan.tasks,
        store,
        retry_errors=retry_errors,
    )

    already_finished = (
        total - len(remaining)
    )

    if not remaining:
        return QZVPDRunSummary(
            total_planned=total,
            already_finished=already_finished,
            attempted=0,
            completed_ok=0,
            completed_error=0,
            remaining_after_run=0,
        )

    def persist_result(
        task: SinglePointTask,
        result: SinglePointResult,
    ) -> None:
        store.append(
            task,
            result,
        )

    results = execute_process_batch(
        remaining,
        worker=worker,
        max_workers=max_workers,
        reporter=reporter,
        stage=(
            CalculationStage.QZVPD_REFINEMENT
        ),
        result_callback=persist_result,
    )

    completed_ok, completed_error = (
        _count_statuses(results)
    )

    remaining_after = len(
        pending_tasks(
            plan.tasks,
            store,
            retry_errors=retry_errors,
        )
    )

    return QZVPDRunSummary(
        total_planned=total,
        already_finished=already_finished,
        attempted=len(results),
        completed_ok=completed_ok,
        completed_error=completed_error,
        remaining_after_run=remaining_after,
    )