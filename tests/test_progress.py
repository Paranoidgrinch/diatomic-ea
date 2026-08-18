"""Tests for calculation progress events."""

import pickle

import pytest

from diatomic_ea.progress import (
    CalculationStage,
    ProgressEvent,
    ProgressEventType,
    ProgressReporter,
)


def test_progress_fraction_and_percent() -> None:
    event = ProgressEvent(
        job_id="abc",
        event_type=ProgressEventType.ADVANCE,
        stage=CalculationStage.FAST_GRID,
        completed=25,
        total=100,
    )

    assert event.fraction == 0.25
    assert event.percent == 25.0


def test_event_without_counts_has_no_percent() -> None:
    event = ProgressEvent(
        job_id="abc",
        event_type=ProgressEventType.MESSAGE,
        message="Preparing calculation.",
    )

    assert event.fraction is None
    assert event.percent is None


def test_invalid_progress_is_rejected() -> None:
    with pytest.raises(ValueError):
        ProgressEvent(
            job_id="abc",
            event_type=ProgressEventType.ADVANCE,
            completed=101,
            total=100,
        )


def test_zero_total_is_rejected() -> None:
    with pytest.raises(ValueError):
        ProgressEvent(
            job_id="abc",
            event_type=ProgressEventType.ADVANCE,
            completed=0,
            total=0,
        )


def test_incomplete_count_pair_is_rejected() -> None:
    with pytest.raises(ValueError):
        ProgressEvent(
            job_id="abc",
            event_type=ProgressEventType.ADVANCE,
            completed=5,
        )


def test_reporter_calls_callback() -> None:
    received: list[ProgressEvent] = []

    reporter = ProgressReporter(
        "job-1",
        callback=received.append,
    )

    event = reporter.advance(
        CalculationStage.FAST_GRID,
        completed=7,
        total=10,
        message="Grid point 7 of 10.",
    )

    assert received == [event]
    assert event.percent == 70.0


def test_warning_event() -> None:
    reporter = ProgressReporter("job-1")

    event = reporter.warning(
        "SCF convergence required a retry.",
        stage=CalculationStage.FAST_GRID,
    )

    assert (
        event.event_type
        is ProgressEventType.WARNING
    )
    assert (
        event.stage
        is CalculationStage.FAST_GRID
    )


def test_events_are_pickle_serializable() -> None:
    event = ProgressEvent(
        job_id="job-1",
        event_type=ProgressEventType.ADVANCE,
        stage=CalculationStage.QZVPD_REFINEMENT,
        completed=8,
        total=20,
        message="Refining geometry.",
    )

    restored = pickle.loads(
        pickle.dumps(event)
    )

    assert restored == event