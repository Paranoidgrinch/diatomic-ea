"""Tests for process-based execution."""

import operator

import pytest

from diatomic_ea.executor import execute_process_batch
from diatomic_ea.progress import (
    CalculationStage,
    ProgressEventType,
    ProgressReporter,
)


def test_process_batch_preserves_input_order() -> None:
    results = execute_process_batch(
        [1, 2, 3, 4],
        worker=operator.neg,
        max_workers=2,
    )

    assert results == (
        -1,
        -2,
        -3,
        -4,
    )


def test_empty_process_batch() -> None:
    results = execute_process_batch(
        [],
        worker=operator.neg,
        max_workers=2,
    )

    assert results == ()


def test_invalid_worker_count() -> None:
    with pytest.raises(ValueError):
        execute_process_batch(
            [1],
            worker=operator.neg,
            max_workers=0,
        )


def test_progress_is_reported_from_parent() -> None:
    events = []

    reporter = ProgressReporter(
        "test-job",
        callback=events.append,
    )

    results = execute_process_batch(
        [1, 2, 3],
        worker=operator.neg,
        max_workers=2,
        reporter=reporter,
        stage=CalculationStage.FAST_GRID,
    )

    assert results == (-1, -2, -3)

    assert (
        events[0].event_type
        is ProgressEventType.STAGE_STARTED
    )

    advances = [
        event
        for event in events
        if (
            event.event_type
            is ProgressEventType.ADVANCE
        )
    ]

    assert len(advances) == 3

    assert sorted(
        event.completed
        for event in advances
    ) == [1, 2, 3]

    assert all(
        event.total == 3
        for event in advances
    )

    assert (
        events[-1].event_type
        is ProgressEventType.STAGE_COMPLETED
    )


def test_result_callback_receives_every_result() -> None:
    received = []

    results = execute_process_batch(
        [1, 2, 3],
        worker=operator.neg,
        max_workers=2,
        result_callback=(
            lambda item, result:
            received.append((item, result))
        ),
    )

    assert results == (-1, -2, -3)

    assert set(received) == {
        (1, -1),
        (2, -2),
        (3, -3),
    }