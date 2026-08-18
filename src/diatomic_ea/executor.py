"""Process-based execution for DiatomicEA calculations."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from typing import TypeVar, cast

from diatomic_ea.grid import FastGridPlan
from diatomic_ea.progress import (
    CalculationStage,
    ProgressReporter,
)
from diatomic_ea.single_point import (
    SinglePointResult,
    run_pyscf_single_point,
)


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def execute_process_batch(
    items: Sequence[InputT],
    *,
    worker: Callable[[InputT], OutputT],
    max_workers: int,
    reporter: ProgressReporter | None = None,
    stage: CalculationStage = CalculationStage.FAST_GRID,
    result_callback: (
        Callable[[InputT, OutputT], None] | None
    ) = None,
) -> tuple[OutputT, ...]:
    """Execute independent tasks in spawned worker processes.

    Results are returned in the same order as the supplied items,
    regardless of the order in which worker processes finish.

    The optional result callback runs in the parent process immediately
    after each worker result becomes available. This allows crash-resistant
    persistence without coupling worker processes to CSV or GUI code.
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

    missing = object()

    results: list[object] = [
        missing
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

            if result_callback is not None:
                result_callback(
                    items[index],
                    result,
                )

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

    if any(
        result is missing
        for result in results
    ):
        raise RuntimeError(
            "Process batch finished with missing results."
        )

    if reporter is not None:
        reporter.stage_completed(
            stage,
            message=(
                f"Completed {completed} tasks."
            ),
        )

    return tuple(
        cast(OutputT, result)
        for result in results
    )


def execute_fast_grid(
    plan: FastGridPlan,
    *,
    max_workers: int,
    reporter: ProgressReporter | None = None,
) -> tuple[SinglePointResult, ...]:
    """Execute every single point in a fast-grid plan."""
    return execute_process_batch(
        plan.tasks,
        worker=run_pyscf_single_point,
        max_workers=max_workers,
        reporter=reporter,
        stage=CalculationStage.FAST_GRID,
    )