"""Tests for Qt-independent GUI state."""

import json
import os

from diatomic_ea.gui_state import (
    discover_latest_status,
    production_status_from_mapping,
    read_production_status,
)


def test_status_snapshot_formats_telemetry() -> None:
    snapshot = (
        production_status_from_mapping(
            {
                "state": "running",
                "stage": "fast-grid",
                "completed": 125,
                "total": 2000,
                "percent": 6.25,
                "tasks_per_second": 2.5,
                "eta_seconds": 750.0,
                "stage_elapsed_seconds": 50.0,
                "message": "Running.",
            }
        )
    )

    assert (
        snapshot.progress_percent
        == 6.25
    )

    assert (
        snapshot.completed_text
        == "125 / 2,000"
    )

    assert (
        snapshot.rate_text
        == "2.50 tasks/s"
    )

    assert (
        snapshot.eta_text
        == "12m 30s"
    )

    assert (
        snapshot.elapsed_text
        == "50s"
    )

    assert (
        snapshot.stage_text
        == "Fast Grid"
    )


def test_qzvpd_stage_name_is_formatted() -> None:
    snapshot = (
        production_status_from_mapping(
            {
                "state": "running",
                "stage": (
                    "qzvpd-refinement"
                ),
            }
        )
    )

    assert (
        snapshot.stage_text
        == "QZVPD Refinement"
    )


def test_completed_without_counts() -> None:
    snapshot = (
        production_status_from_mapping(
            {
                "state": "completed",
                "percent": 100.0,
            }
        )
    )

    assert (
        snapshot.completed_text
        == "Complete"
    )

    assert (
        snapshot.progress_percent
        == 100.0
    )


def test_read_production_status(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "production_status.json"
    )

    path.write_text(
        json.dumps(
            {
                "state": "running",
                "stage": "fast-grid",
                "completed": 4,
                "total": 20,
                "percent": 20.0,
                "tasks_per_second": 0.75,
                "eta_seconds": 21.3,
                "stage_elapsed_seconds": 8.0,
            }
        ),
        encoding="utf-8",
    )

    snapshot = (
        read_production_status(
            path
        )
    )

    assert (
        snapshot.source_path
        == str(
            path.resolve()
        )
    )

    assert (
        snapshot.rate_text
        == "0.75 tasks/s"
    )


def test_discover_latest_status(
    tmp_path,
) -> None:
    first = (
        tmp_path
        / "OH"
        / "run-a"
        / "logs"
        / "production_status.json"
    )

    second = (
        tmp_path
        / "AlO"
        / "run-b"
        / "logs"
        / "production_status.json"
    )

    first.parent.mkdir(
        parents=True,
    )

    second.parent.mkdir(
        parents=True,
    )

    first.write_text(
        "{}",
        encoding="utf-8",
    )

    second.write_text(
        "{}",
        encoding="utf-8",
    )

    os.utime(
        first,
        (
            100,
            100,
        ),
    )

    os.utime(
        second,
        (
            200,
            200,
        ),
    )

    assert (
        discover_latest_status(
            tmp_path
        )
        == second
    )


def test_missing_root_returns_none(
    tmp_path,
) -> None:
    assert (
        discover_latest_status(
            tmp_path
            / "missing"
        )
        is None
    )
