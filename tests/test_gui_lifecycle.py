"""Tests for safe GUI lifecycle behaviour."""

import os
from unittest.mock import patch

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

import pytest

pytest.importorskip(
    "PyQt5"
)

from PyQt5.QtWidgets import (
    QApplication,
)

from diatomic_ea.gui import (
    DiatomicEAMainWindow,
)
from diatomic_ea.jobs import (
    JobStatus,
)


class FakeCloseEvent:
    """Minimal close-event double."""

    def __init__(
        self,
    ) -> None:
        self.accepted = False
        self.ignored = False

    def accept(
        self,
    ) -> None:
        self.accepted = True

    def ignore(
        self,
    ) -> None:
        self.ignored = True


@pytest.fixture(scope="module")
def app():
    application = (
        QApplication.instance()
    )

    if application is None:
        application = QApplication(
            []
        )

    yield application


def make_window(
    app,
    tmp_path,
):
    return DiatomicEAMainWindow(
        auto_probe=False,
        status_root=tmp_path,
    )


def add_oh(
    window,
):
    window.atom_a_input.setText(
        "O"
    )

    window.atom_b_input.setText(
        "H"
    )

    window.add_current_molecule_to_queue()

    return (
        window
        .calculation_queue
        .jobs[0]
    )


def test_idle_window_can_close_normally(
    app,
    tmp_path,
) -> None:
    window = make_window(
        app,
        tmp_path,
    )

    event = FakeCloseEvent()

    window.closeEvent(
        event
    )

    assert event.accepted is True
    assert event.ignored is False


def test_active_calculation_keeps_window_open_by_default(
    app,
    tmp_path,
) -> None:
    window = make_window(
        app,
        tmp_path,
    )

    window._active_job_id = (
        "active-job"
    )

    window._calculation_process_is_active = (
        lambda: True
    )

    window._request_close_choice = (
        lambda: "keep_open"
    )

    event = FakeCloseEvent()

    window.closeEvent(
        event
    )

    assert event.accepted is False
    assert event.ignored is True

    assert (
        window._queue_stop_requested
        is False
    )

    assert (
        window._close_when_idle
        is False
    )

    window._active_job_id = None


def test_exit_after_current_does_not_kill_active_job(
    app,
    tmp_path,
) -> None:
    window = make_window(
        app,
        tmp_path,
    )

    window._active_job_id = (
        "active-job"
    )

    window._calculation_process_is_active = (
        lambda: True
    )

    window._request_close_choice = (
        lambda: "exit_after_current"
    )

    event = FakeCloseEvent()

    window.closeEvent(
        event
    )

    assert event.accepted is False
    assert event.ignored is True

    assert (
        window._queue_stop_requested
        is True
    )

    assert (
        window._close_when_idle
        is True
    )

    assert (
        "close after"
        in window.progress_message.text().casefold()
    )

    window._active_job_id = None


def test_running_job_cannot_be_removed_or_reordered(
    app,
    tmp_path,
) -> None:
    window = make_window(
        app,
        tmp_path,
    )

    job = add_oh(
        window
    )

    started = (
        window.calculation_queue.start_next()
    )

    assert started is job

    window.refresh_queue_view()

    window.queue_list.setCurrentRow(
        0
    )

    window._update_queue_buttons()

    assert (
        job.status
        is JobStatus.RUNNING
    )

    assert (
        window.queue_remove_button.isEnabled()
        is False
    )

    assert (
        window.queue_up_button.isEnabled()
        is False
    )

    assert (
        window.queue_down_button.isEnabled()
        is False
    )


def test_waiting_job_can_be_removed(
    app,
    tmp_path,
) -> None:
    window = make_window(
        app,
        tmp_path,
    )

    job = add_oh(
        window
    )

    assert (
        job.status
        is JobStatus.QUEUED
    )

    window.queue_list.setCurrentRow(
        0
    )

    window._update_queue_buttons()

    assert (
        window.queue_remove_button.isEnabled()
        is True
    )


def test_exit_after_current_schedules_close_only_after_finish(
    app,
    tmp_path,
) -> None:
    window = make_window(
        app,
        tmp_path,
    )

    job = add_oh(
        window
    )

    started = (
        window.calculation_queue.start_next()
    )

    assert started is job

    window._active_job_id = (
        job.job_id
    )

    window._queue_stop_requested = True
    window._close_when_idle = True

    with patch(
        "diatomic_ea.gui.QTimer.singleShot"
    ) as single_shot:
        window._finish_active_job(
            success=True,
            message=(
                "Electron-affinity calculation completed."
            ),
        )

    assert (
        job.status
        is JobStatus.COMPLETED
    )

    assert (
        window._active_job_id
        is None
    )

    assert (
        window._close_when_idle
        is False
    )

    single_shot.assert_called_once()

    delay, callback = (
        single_shot.call_args.args
    )

    assert delay == 0

    assert (
        callback
        == window.close
    )


def test_pending_exit_disables_new_queue_actions(
    app,
    tmp_path,
) -> None:
    window = make_window(
        app,
        tmp_path,
    )

    add_oh(
        window
    )

    window._close_when_idle = True
    window._queue_stop_requested = True

    window._update_queue_buttons()

    assert (
        window.add_queue_button.isEnabled()
        is False
    )

    assert (
        window.start_queue_button.isEnabled()
        is False
    )

    assert (
        window.resume_queue_button.isEnabled()
        is False
    )


def test_logical_active_job_without_process_does_not_block_close(
    app,
    tmp_path,
) -> None:
    """A mocked/planned job alone must not trigger the shutdown dialog."""

    window = make_window(
        app,
        tmp_path,
    )

    window._active_job_id = (
        "logical-job-only"
    )

    assert (
        window._active_process
        is None
    )

    event = FakeCloseEvent()

    window.closeEvent(
        event
    )

    assert event.accepted is True
    assert event.ignored is False

    window._active_job_id = None
