"""Headless tests for the PyQt5 desktop interface."""

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
    APP_TITLE,
    DiatomicEAMainWindow,
)
from diatomic_ea.gui_state import (
    production_status_from_mapping,
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


def test_main_window_constructs(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    assert (
        window.windowTitle()
        == APP_TITLE
    )

    assert (
        window.worker_spin.minimum()
        == 1
    )

    assert (
        window.worker_spin.value()
        >= 1
    )

    assert (
        window.queue_count()
        == 0
    )

    window.close()


def test_valid_molecule_can_be_queued(
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

    assert (
        window.queue_count()
        == 1
    )

    assert (
        window.queue_list.count()
        == 1
    )

    assert (
        "AlO"
        in window.queue_list.item(
            0
        ).text()
    )

    window.close()


def test_queue_can_be_reordered(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    for (
        atom_a,
        atom_b,
    ) in (
        (
            "Al",
            "O",
        ),
        (
            "Mg",
            "O",
        ),
    ):
        window.atom_a_input.setText(
            atom_a
        )

        window.atom_b_input.setText(
            atom_b
        )

        window.add_current_molecule_to_queue()

    window.queue_list.setCurrentRow(
        1
    )

    window.move_selected_up()

    assert (
        window
        .calculation_queue
        .jobs[0]
        .molecule
        .formula
        == "MgO"
    )

    window.close()


def test_dashboard_accepts_live_telemetry(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    snapshot = (
        production_status_from_mapping(
            {
                "state": "running",
                "stage": "fast-grid",
                "completed": 250,
                "total": 2000,
                "percent": 12.5,
                "tasks_per_second": 1.82,
                "eta_seconds": 961.5,
                "stage_elapsed_seconds": 137.4,
                "message": (
                    "Fast-grid calculation running."
                ),
            }
        )
    )

    window.apply_production_status(
        snapshot
    )

    assert (
        window.progress_bar.value()
        == 125
    )

    assert (
        window.progress_bar.format()
        == "12.5%"
    )

    assert (
        window.completed_value.text()
        == "250 / 2,000"
    )

    assert (
        window.rate_value.text()
        == "1.82 tasks/s"
    )

    assert (
        window.eta_value.text()
        == "16m 02s"
    )

    assert (
        window.elapsed_value.text()
        == "2m 17s"
    )

    window.close()


def test_auto_probe_can_be_disabled(
    app,
    tmp_path,
) -> None:
    window = (
        DiatomicEAMainWindow(
            auto_probe=False,
            status_root=tmp_path,
        )
    )

    window.show()

    app.processEvents()

    assert (
        window._active_probe
        is None
    )

    window.close()
