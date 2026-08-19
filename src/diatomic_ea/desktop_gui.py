"""Complete DiatomicEA PyQt5 desktop application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt5.QtCore import (
    QTimer,
    Qt,
    QUrl,
)
from PyQt5.QtGui import (
    QDesktopServices,
    QFont,
)
from PyQt5.QtWidgets import (
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diatomic_ea.gui import (
    DiatomicEAMainWindow,
    build_application,
)
from diatomic_ea.gui_persistence import (
    GuiPreferences,
    load_preferences,
    load_queue_session,
    save_preferences,
    save_queue_session,
)
from diatomic_ea.gui_results import (
    ResultReadError,
    format_energy,
    format_interval,
    read_calculation_result,
)
from diatomic_ea.jobs import (
    CalculationJob,
    JobStatus,
)
from diatomic_ea.queue import (
    CalculationQueue,
)


class DiatomicEADesktopWindow(
    DiatomicEAMainWindow
):
    """Complete desktop workflow built on the tested calculation GUI."""

    def __init__(
        self,
        *,
        auto_probe: bool = True,
        status_root: str | Path | None = None,
        persist_gui_state: bool = True,
    ) -> None:
        self._desktop_ready = False

        self._persist_gui_state = bool(
            persist_gui_state
        )

        self._selected_result_directory: (
            Path
            | None
        ) = None

        super().__init__(
            auto_probe=auto_probe,
            status_root=status_root,
        )

        # The raw telemetry path may contain internal method identifiers.
        # It is useful for debugging but not part of the user interface.
        self.status_source_label.hide()

        self.worker_hint.setText(
            (
                "CPU and memory limits are checked "
                "before the calculation starts."
            )
        )

        self.resume_queue_button.setText(
            "Resume / retry"
        )

        self.gui_state_root = (
            self.status_root
            / "_gui"
        )

        self.preferences_path = (
            self.gui_state_root
            / "preferences.json"
        )

        self.queue_session_path = (
            self.gui_state_root
            / "queue_session.json"
        )

        self._build_results_dock()

        self._restore_preferences()

        recovered = (
            self._restore_queue_session()
        )

        self._desktop_ready = True

        self._connect_persistent_controls()

        self.queue_list.currentItemChanged.connect(
            self._result_selection_changed
        )

        if recovered:
            self._queue_stop_requested = True

        self.refresh_queue_view()

        if recovered:
            self.progress_message.setText(
                (
                    "An interrupted calculation was recovered. "
                    "Select Resume / retry to continue it."
                )
            )

        self._select_latest_finished_job()

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def _connect_persistent_controls(
        self,
    ) -> None:
        for control in (
            self.minimum_bond_spin,
            self.maximum_bond_spin,
            self.spin_max_spin,
            self.worker_spin,
        ):
            control.valueChanged.connect(
                self._preference_changed
            )

    def _preference_changed(
        self,
        _value,
    ) -> None:
        self._save_preferences()

    def _current_preferences(
        self,
    ) -> GuiPreferences:
        return GuiPreferences(
            minimum_angstrom=float(
                self.minimum_bond_spin.value()
            ),
            maximum_angstrom=float(
                self.maximum_bond_spin.value()
            ),
            spin_max=int(
                self.spin_max_spin.value()
            ),
            workers=int(
                self.worker_spin.value()
            ),
        )

    def _restore_preferences(
        self,
    ) -> None:
        if not self._persist_gui_state:
            return

        preferences = load_preferences(
            self.preferences_path,
            fallback_workers=(
                self.resources.recommended_workers
            ),
        )

        self.minimum_bond_spin.setValue(
            preferences.minimum_angstrom
        )

        self.maximum_bond_spin.setValue(
            preferences.maximum_angstrom
        )

        self.spin_max_spin.setValue(
            min(
                preferences.spin_max,
                self.spin_max_spin.maximum(),
            )
        )

        self.worker_spin.setValue(
            min(
                preferences.workers,
                self.worker_spin.maximum(),
            )
        )

    def _save_preferences(
        self,
    ) -> None:
        if (
            not self._persist_gui_state
            or not self._desktop_ready
        ):
            return

        save_preferences(
            self.preferences_path,
            self._current_preferences(),
        )

    def _restore_queue_session(
        self,
    ) -> int:
        if not self._persist_gui_state:
            return 0

        try:
            session = load_queue_session(
                self.queue_session_path
            )
        except ValueError as exc:
            self.progress_message.setText(
                str(
                    exc
                )
            )

            return 0

        self.calculation_queue = (
            CalculationQueue(
                session.jobs
            )
        )

        self.gui_job_specs = dict(
            session.specs
        )

        return session.recovered_running_jobs

    def _save_queue_session(
        self,
    ) -> None:
        if (
            not self._persist_gui_state
            or not self._desktop_ready
        ):
            return

        save_queue_session(
            self.queue_session_path,
            jobs=self.calculation_queue.jobs,
            specs=self.gui_job_specs,
        )

    def refresh_queue_view(
        self,
    ) -> None:
        super().refresh_queue_view()

        self._save_queue_session()

    def apply_production_status(
        self,
        snapshot,
    ) -> None:
        """Present telemetry without exposing internal run identifiers."""

        super().apply_production_status(
            snapshot
        )

        spec = self._active_spec()

        if spec is not None:
            self.active_job_label.setText(
                spec.molecule.formula
            )

        elif snapshot.state.casefold() == "completed":
            self.active_job_label.setText(
                "Latest completed calculation"
            )

        else:
            self.active_job_label.setText(
                "Latest calculation"
            )

    # --------------------------------------------------------
    # Results panel
    # --------------------------------------------------------

    def _build_results_dock(
        self,
    ) -> None:
        dock = QDockWidget(
            "Calculation results",
            self,
        )

        dock.setObjectName(
            "resultsDock"
        )

        dock.setAllowedAreas(
            Qt.RightDockWidgetArea
            | Qt.LeftDockWidgetArea
        )

        dock.setMinimumWidth(
            365
        )

        container = QWidget()

        layout = QVBoxLayout(
            container
        )

        layout.setContentsMargins(
            18,
            18,
            18,
            18,
        )

        layout.setSpacing(
            12
        )

        self.result_molecule_label = QLabel(
            "No completed result selected"
        )

        title_font = QFont()

        title_font.setPointSize(
            13
        )

        title_font.setWeight(
            QFont.DemiBold
        )

        self.result_molecule_label.setFont(
            title_font
        )

        layout.addWidget(
            self.result_molecule_label
        )

        self.result_ea_value = QLabel(
            "-- eV"
        )

        ea_font = QFont()

        ea_font.setPointSize(
            22
        )

        ea_font.setWeight(
            QFont.DemiBold
        )

        self.result_ea_value.setFont(
            ea_font
        )

        layout.addWidget(
            self.result_ea_value
        )

        subtitle = QLabel(
            "Predicted electron affinity"
        )

        subtitle.setObjectName(
            "secondaryText"
        )

        layout.addWidget(
            subtitle
        )

        grid = QGridLayout()

        self.result_pi80 = QLabel(
            "--"
        )

        self.result_pi90 = QLabel(
            "--"
        )

        self.result_pi95 = QLabel(
            "--"
        )

        self.result_half_range = QLabel(
            "--"
        )

        rows = (
            (
                "80% prediction interval",
                self.result_pi80,
            ),
            (
                "90% prediction interval",
                self.result_pi90,
            ),
            (
                "95% prediction interval",
                self.result_pi95,
            ),
            (
                "Functional half-range",
                self.result_half_range,
            ),
        )

        for row_index, (
            title,
            value,
        ) in enumerate(
            rows
        ):
            grid.addWidget(
                QLabel(
                    title
                ),
                row_index,
                0,
            )

            grid.addWidget(
                value,
                row_index,
                1,
            )

        layout.addLayout(
            grid
        )

        functional_title = QLabel(
            "Functional electron affinities"
        )

        functional_title.setObjectName(
            "metricTitle"
        )

        layout.addWidget(
            functional_title
        )

        functional_grid = QGridLayout()

        self.result_functionals = {}

        for row_index, functional in enumerate(
            (
                "PBE",
                "B3LYP",
                "PBE0",
                "TPSSh",
            )
        ):
            functional_grid.addWidget(
                QLabel(
                    functional
                ),
                row_index,
                0,
            )

            value = QLabel(
                "--"
            )

            self.result_functionals[
                functional
            ] = value

            functional_grid.addWidget(
                value,
                row_index,
                1,
            )

        layout.addLayout(
            functional_grid
        )

        self.result_message = QLabel(
            (
                "Select a completed calculation "
                "to view its result."
            )
        )

        self.result_message.setObjectName(
            "secondaryText"
        )

        self.result_message.setWordWrap(
            True
        )

        layout.addWidget(
            self.result_message
        )

        layout.addStretch(
            1
        )

        controls = QHBoxLayout()

        controls.addStretch(
            1
        )

        self.open_results_button = QPushButton(
            "Open results folder"
        )

        self.open_results_button.setEnabled(
            False
        )

        self.open_results_button.clicked.connect(
            self.open_results_folder
        )

        controls.addWidget(
            self.open_results_button
        )

        layout.addLayout(
            controls
        )

        dock.setWidget(
            container
        )

        self.addDockWidget(
            Qt.RightDockWidgetArea,
            dock,
        )

        self.results_dock = dock

    def _clear_result(
        self,
        message: str,
    ) -> None:
        self.result_molecule_label.setText(
            "No completed result selected"
        )

        self.result_ea_value.setText(
            "-- eV"
        )

        self.result_pi80.setText(
            "--"
        )

        self.result_pi90.setText(
            "--"
        )

        self.result_pi95.setText(
            "--"
        )

        self.result_half_range.setText(
            "--"
        )

        for label in self.result_functionals.values():
            label.setText(
                "--"
            )

        self.result_message.setText(
            message
        )

        self._selected_result_directory = None

        self.open_results_button.setEnabled(
            False
        )

    def show_result_for_job(
        self,
        job_id: str | None,
    ) -> None:
        if job_id is None:
            self._clear_result(
                (
                    "Select a completed calculation "
                    "to view its result."
                )
            )

            return

        try:
            job = self.calculation_queue.get(
                job_id
            )
        except KeyError:
            self._clear_result(
                "The selected calculation is unavailable."
            )

            return

        spec = self.gui_job_specs.get(
            job_id
        )

        if spec is None:
            self._clear_result(
                "Calculation settings are unavailable."
            )

            return

        run_directory = spec.run_directory(
            self.status_root
        )

        if job.status is JobStatus.FAILED:
            self._clear_result(
                (
                    "This calculation failed. "
                    "Use Resume / retry to continue "
                    "from saved results."
                )
            )

            if run_directory.is_dir():
                self._selected_result_directory = (
                    run_directory
                )

                self.open_results_button.setEnabled(
                    True
                )

            return

        if job.status is JobStatus.RUNNING:
            self._clear_result(
                "This calculation is currently running."
            )

            return

        if job.status is JobStatus.QUEUED:
            self._clear_result(
                "This calculation is waiting in the queue."
            )

            return

        if job.status is not JobStatus.COMPLETED:
            self._clear_result(
                "No final result is available."
            )

            return

        result_path = spec.final_result_path(
            self.status_root
        )

        try:
            result = read_calculation_result(
                result_path
            )
        except ResultReadError as exc:
            self._clear_result(
                (
                    "The final result could not be read: "
                    + str(
                        exc
                    )
                )
            )

            if run_directory.is_dir():
                self._selected_result_directory = (
                    run_directory
                )

                self.open_results_button.setEnabled(
                    True
                )

            return

        self.result_molecule_label.setText(
            (
                result.molecule
                + " electron affinity"
            )
        )

        self.result_ea_value.setText(
            format_energy(
                result.predicted_ea_ev
            )
        )

        self.result_pi80.setText(
            format_interval(
                result.interval(
                    80
                )
            )
        )

        self.result_pi90.setText(
            format_interval(
                result.interval(
                    90
                )
            )
        )

        self.result_pi95.setText(
            format_interval(
                result.interval(
                    95
                )
            )
        )

        self.result_half_range.setText(
            (
                "±"
                + f"{result.functional_half_range_ev:.4f}"
                + " eV"
            )
        )

        values = dict(
            result.functional_eas_ev
        )

        for functional, label in self.result_functionals.items():
            label.setText(
                format_energy(
                    values[
                        functional
                    ]
                )
            )

        self.result_message.setText(
            (
                "Prediction based on all four "
                "standard functional calculations."
            )
        )

        self._selected_result_directory = (
            result_path.parent
        )

        self.open_results_button.setEnabled(
            True
        )

    def _result_selection_changed(
        self,
        _current=None,
        _previous=None,
    ) -> None:
        self.show_result_for_job(
            self._selected_job_id()
        )

    def _select_latest_finished_job(
        self,
    ) -> None:
        selected_row = None

        for index, job in enumerate(
            self.calculation_queue.jobs
        ):
            if job.status in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            ):
                selected_row = index

        if selected_row is not None:
            self.queue_list.setCurrentRow(
                selected_row
            )

    def open_results_folder(
        self,
    ) -> None:
        directory = (
            self._selected_result_directory
        )

        if (
            directory is None
            or not directory.exists()
        ):
            self.result_message.setText(
                "The results folder is not available."
            )

            return

        opened = QDesktopServices.openUrl(
            QUrl.fromLocalFile(
                str(
                    directory.resolve()
                )
            )
        )

        if not opened:
            self.result_message.setText(
                (
                    "The operating system could not "
                    "open the results folder."
                )
            )

    # --------------------------------------------------------
    # Failed-job retry
    # --------------------------------------------------------

    def _failed_jobs_exist(
        self,
    ) -> bool:
        return any(
            job.status is JobStatus.FAILED
            for job in self.calculation_queue.jobs
        )

    def _requeue_failed_jobs(
        self,
    ) -> int:
        rebuilt_jobs = []
        recovered = 0

        for job in self.calculation_queue.jobs:
            if job.status is JobStatus.FAILED:
                rebuilt_jobs.append(
                    CalculationJob(
                        molecule=job.molecule,
                        mode=job.mode,
                        job_id=job.job_id,
                        status=JobStatus.QUEUED,
                    )
                )

                recovered += 1
            else:
                rebuilt_jobs.append(
                    job
                )

        if recovered:
            self.calculation_queue = (
                CalculationQueue(
                    rebuilt_jobs
                )
            )

            self.refresh_queue_view()

        return recovered

    def _update_queue_buttons(
        self,
    ) -> None:
        super()._update_queue_buttons()

        if not hasattr(
            self,
            "resume_queue_button",
        ):
            return

        if (
            self._failed_jobs_exist()
            and self._active_job_id is None
            and not self._close_when_idle
        ):
            self.resume_queue_button.setEnabled(
                True
            )

    def resume_queue(
        self,
    ) -> None:
        if (
            self._active_job_id is not None
            or self._close_when_idle
        ):
            return

        recovered = self._requeue_failed_jobs()

        if recovered:
            self.progress_message.setText(
                (
                    "Failed calculation queued for retry. "
                    "Saved single-point results will be reused."
                )
            )

        super().resume_queue()

    # --------------------------------------------------------
    # Completion + shutdown hooks
    # --------------------------------------------------------

    def _finish_active_job(
        self,
        *,
        success: bool,
        message: str,
    ) -> None:
        completed_job_id = (
            self._active_job_id
        )

        super()._finish_active_job(
            success=success,
            message=message,
        )

        self._save_queue_session()

        if completed_job_id is not None:
            for index, job in enumerate(
                self.calculation_queue.jobs
            ):
                if job.job_id == completed_job_id:
                    self.queue_list.setCurrentRow(
                        index
                    )

                    break

            self.show_result_for_job(
                completed_job_id
            )

    def closeEvent(
        self,
        event,
    ) -> None:
        self._save_preferences()
        self._save_queue_session()

        super().closeEvent(
            event
        )


def main(
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--no-probe",
        action="store_true",
    )

    arguments = parser.parse_args(
        sys.argv[1:]
        if argv is None
        else argv
    )

    app = build_application()

    window = DiatomicEADesktopWindow(
        auto_probe=(
            not arguments.no_probe
            and not arguments.smoke
        )
    )

    window.show()

    if arguments.smoke:
        QTimer.singleShot(
            300,
            app.quit,
        )

    return int(
        app.exec_()
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
