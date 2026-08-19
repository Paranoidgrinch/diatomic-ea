"""Tests for GUI-ready progress rate and ETA telemetry."""

import pytest

from diatomic_ea.progress import (
    CalculationStage,
    ProgressEvent,
    ProgressEventType,
)
from diatomic_ea.progress_metrics import (
    ProgressRateTracker,
    format_duration,
)


def stage_started():
    return ProgressEvent(
        job_id="test",
        event_type=(
            ProgressEventType.STAGE_STARTED
        ),
        stage=CalculationStage.FAST_GRID,
    )


def advance(
    completed,
    total=10,
):
    return ProgressEvent(
        job_id="test",
        event_type=(
            ProgressEventType.ADVANCE
        ),
        stage=CalculationStage.FAST_GRID,
        completed=completed,
        total=total,
    )


def test_rate_and_eta_are_calculated() -> None:
    tracker = ProgressRateTracker(
        window_seconds=30.0,
        minimum_elapsed_seconds=0.5,
    )

    tracker.update(
        stage_started(),
        now=100.0,
    )

    metrics = tracker.update(
        advance(
            1
        ),
        now=102.0,
    )

    assert (
        metrics.tasks_per_second
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics.eta_seconds
        == pytest.approx(
            18.0
        )
    )

    assert (
        metrics.elapsed_seconds
        == pytest.approx(
            2.0
        )
    )


def test_rate_updates_with_more_completed_tasks() -> None:
    tracker = ProgressRateTracker()

    tracker.update(
        stage_started(),
        now=0.0,
    )

    tracker.update(
        advance(
            1
        ),
        now=2.0,
    )

    metrics = tracker.update(
        advance(
            3
        ),
        now=4.0,
    )

    assert (
        metrics.tasks_per_second
        == pytest.approx(
            0.75
        )
    )

    assert (
        metrics.eta_seconds
        == pytest.approx(
            7.0
            / 0.75
        )
    )


def test_completed_stage_has_zero_eta() -> None:
    tracker = ProgressRateTracker()

    tracker.update(
        stage_started(),
        now=0.0,
    )

    metrics = tracker.update(
        advance(
            10
        ),
        now=5.0,
    )

    assert (
        metrics.tasks_per_second
        == pytest.approx(
            2.0
        )
    )

    assert (
        metrics.eta_seconds
        == pytest.approx(
            0.0
        )
    )


def test_stage_change_resets_rate() -> None:
    tracker = ProgressRateTracker()

    tracker.update(
        stage_started(),
        now=0.0,
    )

    tracker.update(
        advance(
            5
        ),
        now=5.0,
    )

    new_stage = ProgressEvent(
        job_id="test",
        event_type=(
            ProgressEventType.STAGE_STARTED
        ),
        stage=(
            CalculationStage
            .QZVPD_REFINEMENT
        ),
    )

    metrics = tracker.update(
        new_stage,
        now=6.0,
    )

    assert (
        metrics.stage
        is CalculationStage.QZVPD_REFINEMENT
    )

    assert (
        metrics.tasks_per_second
        is None
    )

    assert (
        metrics.eta_seconds
        is None
    )


def test_rate_waits_for_minimum_elapsed_time() -> None:
    tracker = ProgressRateTracker(
        minimum_elapsed_seconds=1.0,
    )

    tracker.update(
        stage_started(),
        now=0.0,
    )

    metrics = tracker.update(
        advance(
            1
        ),
        now=0.1,
    )

    assert (
        metrics.tasks_per_second
        is None
    )

    assert (
        metrics.eta_seconds
        is None
    )


@pytest.mark.parametrize(
    (
        "seconds",
        "expected",
    ),
    (
        (
            None,
            "--",
        ),
        (
            8.4,
            "8s",
        ),
        (
            95.0,
            "1m 35s",
        ),
        (
            3670.0,
            "1h 01m",
        ),
    ),
)
def test_duration_formatting(
    seconds,
    expected,
) -> None:
    assert (
        format_duration(
            seconds
        )
        == expected
    )
