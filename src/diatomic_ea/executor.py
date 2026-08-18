"""Process-based execution for DiatomicEA calculations."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import TypeVar

from diatomic_ea.grid import FastGridPlan
from diatomic_ea.progress import (
    CalculationStage,
    ProgressReporter,
)
from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointTask,
    run_pyscf_single_point,
)


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class ProcessBatchResult:
    """Results from one ordered process batch."""

    results: tuple[object, ...]
    submitted: int
    completed: int


def execute_process_batch(
    items: Sequence[InputT],
    *,
    worker: Callable[[InputT], OutputT],
    max_workers: int,
    reporter: ProgressReporter | None = None,
    stage: CalculationStage = CalculationStage.FAST_GRID,
) -> tuple[OutputT, ...]:
    """Execute independent tasks in spawned worker processes.

    Results are returned in the same order as the supplied items,
    regardless of the order in which worker processes finish.
    """
    if max_workers < 1:
        raise ValueError(
            "max_workers must be at least 1."
        )

    total = len(items)

    if total == 0:
        return ()

    if reporter is not None:
        reporter.stage_started(
            stage,
            message=(
                f"Starting {total} tasks "
                f"with {max_workers} workers."
            ),
        )

    context = multiprocessing.get_context(
        "spawn"
    )

    results: list[OutputT | None] = [
        None
        for _ in items
    ]

    completed = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
    ) as pool:
        futures = {
            pool.submit(
                worker,
                item,
            ): index
            for index, item in enumerate(items)
        }

        try:
            for future in as_completed(futures):
                index = futures[future]

                try:
                    result = future.result()
                except Exception as exc:
                    if reporter is not None:
                        reporter.warning(
                            (
                                f"Worker task {index + 1} "
                                f"failed: {exc}"
                            ),
                            stage=stage,
                        )

                    for pending in futures:
                        pending.cancel()

                    raise

                results[index] = result
                completed += 1

                if reporter is not None:
                    reporter.advance(
                        stage,
                        completed=completed,
                        total=total,
                        message=(
                            f"Completed task "
                            f"{completed} of {total}."
                        ),
                    )

        finally:
            pass

    if reporter is not None:
        reporter.stage_completed(
            stage,
            message=(
                f"Completed {completed} tasks."
            ),
        )

    if any(
        result is None
        for result in results
    ):
        raise RuntimeError(
            "Process batch finished with missing results."
        )

    return tuple(
        result
        for result in results
        if result is not None
    )


def execute_fast_grid(
    plan: FastGridPlan,
    *,
    max_workers: int,
    reporter: ProgressReporter | None = None,
) -> tuple[SinglePointResult, ...]:
    """Execute all single points in a fast-grid plan."""
    results = execute_process_batch(
        plan.tasks,
        worker=run_pyscf_single_point,
        max_workers=max_workers,
        reporter=reporter,
        stage=CalculationStage.FAST_GRID,
    )

    return tuple(results)