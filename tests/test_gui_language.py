"""Tests for user-facing GUI language and queue control."""

import os

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
from diatomic_ea.gui_state import (
    humanize_status_message,
    production_status_from_mapping,
    stage_display_name,
)
from diatomic_ea.jobs import (
    JobStatus,
)


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


@pytest.mark.parametrize(
    (
        "internal",
        "visible",
    ),
    (
        (
            "fast-grid",
            "Initial geometry scan",
        ),
        (
            "fast-grid-analysis",
            "Geometry analysis",
        ),
        (
            "qzvpd-refinement",
            "High-accuracy refinement",
        ),
        (
            "statistical-ea",
            "EA prediction",
        ),
        (
            "export",
            "Saving results",
        ),
    ),
)
def test_internal_stage_names_are_translated(
    internal,
    visible,
) -> None:
    assert (
        stage_display_name(
            internal
        )
        == visible
    )


def test_internal_method_language_is_removed_from_messages() -> None:
    message = (
        "Schema F statistical estimate completed."
    )

    visible = (
        humanize_status_message(
            message
        )
    )

    assert (
        "schema"
        not in visible.casefold()
    )

    assert (
        visible
        == "EA prediction completed."
    )


def test_internal_refinement_name_is_removed() -> None:
    snapshot = (
        production_status_from_mapping(
            {
                "state": "running",
                "stage": "qzvpd-refinement",
                "message": (
                    "QZVPD refinement is running."
                ),
            }
        )
    )

    assert (
        snapshot.stage_text
        == "High-accuracy refinement"
    )

    assert (
        "qzvpd"
        not in snapshot.message.casefold()
    )


def test_queue_item_uses_natural_english(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    window.atom_a_input.setText(
        "Al"
    )

    window.atom_b_input.setText(
        "O"
    )

    window.add_current_molecule_to_queue()

    text = (
        window.queue_list.item(
            0
        ).text()
    )

    assert (
        "Standard calculation"
        in text
    )

    assert (
        "schema"
        not in text.casefold()
    )

    window.close()


def test_settings_are_frozen_when_job_is_queued(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    window.atom_a_input.setText(
        "O"
    )

    window.atom_b_input.setText(
        "H"
    )

    window.minimum_bond_spin.setValue(
        0.75
    )

    window.maximum_bond_spin.setValue(
        1.35
    )

    window.spin_max_spin.setValue(
        3
    )

    window.worker_spin.setValue(
        2
    )

    window.add_current_molecule_to_queue()

    job = (
        window
        .calculation_queue
        .jobs[0]
    )

    calculation = (
        window.gui_job_specs[
            job.job_id
        ]
    )

    window.maximum_bond_spin.setValue(
        3.0
    )

    assert (
        calculation.minimum_angstrom
        == pytest.approx(
            0.75
        )
    )

    assert (
        calculation.maximum_angstrom
        == pytest.approx(
            1.35
        )
    )

    assert (
        calculation.spin_max
        == 3
    )

    assert (
        calculation.workers
        == 2
    )

    window.close()


def test_start_queue_builds_real_plan_command_without_running_it(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    window.atom_a_input.setText(
        "O"
    )

    window.atom_b_input.setText(
        "H"
    )

    window.add_current_molecule_to_queue()

    captured = {}

    def fake_start_process(
        command,
        *,
        phase,
    ):
        captured[
            "command"
        ] = command

        captured[
            "phase"
        ] = phase

    window._start_process = (
        fake_start_process
    )

    window.start_queue()

    job = (
        window
        .calculation_queue
        .jobs[0]
    )

    assert (
        job.status
        is JobStatus.RUNNING
    )

    assert (
        captured[
            "phase"
        ]
        == "planning"
    )

    assert (
        "diatomic_ea.production_plan"
        in captured[
            "command"
        ].arguments
    )

    window.close()


def test_stop_after_current_is_graceful(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    window._active_job_id = (
        "active-test-job"
    )

    window.stop_queue_after_current()

    assert (
        window._queue_stop_requested
        is True
    )

    assert (
        "after the current calculation"
        in window.progress_message.text()
    )

    window._active_job_id = None

    window.close()
