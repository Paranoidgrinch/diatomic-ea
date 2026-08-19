"""Tests for production progress telemetry persistence."""

import json
from types import SimpleNamespace

import pytest

from diatomic_ea.production_run import (
    PRODUCTION_STATUS_VERSION,
    _ConsoleProgress,
)
from diatomic_ea.progress import (
    CalculationStage,
    ProgressEvent,
    ProgressEventType,
)


class FakeClock:
    def __init__(
        self,
        value=0.0,
    ):
        self.value = float(
            value
        )

    def __call__(self):
        return self.value


def event(
    event_type,
    *,
    completed=None,
    total=None,
):
    return ProgressEvent(
        job_id="run",
        event_type=event_type,
        stage=CalculationStage.FAST_GRID,
        completed=completed,
        total=total,
    )


def test_status_version_is_two() -> None:
    assert (
        PRODUCTION_STATUS_VERSION
        == 2
    )


def test_console_progress_persists_rate_and_eta(
    tmp_path,
    capsys,
) -> None:
    clock = FakeClock(
        100.0
    )

    validated = SimpleNamespace(
        status_path=(
            tmp_path
            / "status.json"
        ),
        events_path=(
            tmp_path
            / "events.jsonl"
        ),
    )

    status = {}

    callback = _ConsoleProgress(
        validated,
        status,
        clock=clock,
        print_interval_seconds=2.0,
    )

    callback(
        event(
            ProgressEventType.STAGE_STARTED
        )
    )

    clock.value = 102.0

    callback(
        event(
            ProgressEventType.ADVANCE,
            completed=1,
            total=10,
        )
    )

    payload = json.loads(
        validated.status_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "tasks_per_second"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        payload[
            "eta_seconds"
        ]
        == pytest.approx(
            18.0
        )
    )

    assert (
        payload[
            "stage_elapsed_seconds"
        ]
        == pytest.approx(
            2.0
        )
    )

    output = (
        capsys.readouterr()
        .out
    )

    assert (
        "0.50 tasks/s"
        in output
    )

    assert (
        "ETA 18s"
        in output
    )


def test_time_interval_prints_before_next_five_percent(
    tmp_path,
    capsys,
) -> None:
    clock = FakeClock()

    validated = SimpleNamespace(
        status_path=(
            tmp_path
            / "status.json"
        ),
        events_path=(
            tmp_path
            / "events.jsonl"
        ),
    )

    callback = _ConsoleProgress(
        validated,
        {},
        clock=clock,
        print_interval_seconds=2.0,
    )

    callback(
        event(
            ProgressEventType.STAGE_STARTED
        )
    )

    clock.value = 1.0

    callback(
        event(
            ProgressEventType.ADVANCE,
            completed=1,
            total=100,
        )
    )

    capsys.readouterr()

    clock.value = 2.0

    callback(
        event(
            ProgressEventType.ADVANCE,
            completed=2,
            total=100,
        )
    )

    quiet = (
        capsys.readouterr()
        .out
    )

    assert quiet == ""

    clock.value = 3.1

    callback(
        event(
            ProgressEventType.ADVANCE,
            completed=3,
            total=100,
        )
    )

    output = (
        capsys.readouterr()
        .out
    )

    assert (
        "3/100"
        in output
    )

    assert (
        "tasks/s"
        in output
    )


def test_every_event_log_contains_gui_metrics(
    tmp_path,
) -> None:
    clock = FakeClock()

    validated = SimpleNamespace(
        status_path=(
            tmp_path
            / "status.json"
        ),
        events_path=(
            tmp_path
            / "events.jsonl"
        ),
    )

    callback = _ConsoleProgress(
        validated,
        {},
        clock=clock,
    )

    callback(
        event(
            ProgressEventType.STAGE_STARTED
        )
    )

    clock.value = 5.0

    callback(
        event(
            ProgressEventType.ADVANCE,
            completed=5,
            total=10,
        )
    )

    rows = [
        json.loads(
            line
        )
        for line
        in validated.events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    latest = rows[-1]

    assert (
        latest[
            "tasks_per_second"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        latest[
            "eta_seconds"
        ]
        == pytest.approx(
            5.0
        )
    )

    assert (
        latest[
            "stage_elapsed_seconds"
        ]
        == pytest.approx(
            5.0
        )
    )
