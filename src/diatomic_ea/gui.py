"""PyQt5 desktop interface for DiatomicEA."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PyQt5.QtCore import (
    QObject,
    QProcess,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QFont,
)
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from diatomic_ea.compute_environment import (
    ComputeEnvironmentReport,
    inspect_compute_environment,
)
from diatomic_ea.gui_execution import (
    GuiCalculationSpec,
    ProcessCommand,
    build_plan_command,
    build_run_command,
    calculation_mode_label,
    job_status_label,
    make_gui_run_id,
)
from diatomic_ea.gui_state import (
    ProductionStatusSnapshot,
    discover_latest_status,
    humanize_status_message,
    read_production_status,
)
from diatomic_ea.jobs import (
    CalculationJob,
    JobStatus,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.queue import (
    CalculationQueue,
)
from diatomic_ea.resources import (
    CpuResources,
    detect_cpu_resources,
)


APP_TITLE = "DiatomicEA"

APP_TAGLINE = (
    "Fast and reproducible electron-affinity "
    "calculations for diatomic molecules."
)


class ComputeProbeSignals(QObject):
    """Signals emitted by the asynchronous environment probe."""

    finished = pyqtSignal(
        object
    )

    failed = pyqtSignal(
        str
    )


class ComputeProbe(QRunnable):
    """Inspect the compute environment outside the GUI thread."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.signals = (
            ComputeProbeSignals()
        )

    def run(
        self,
    ) -> None:
        try:
            report = (
                inspect_compute_environment()
            )

        except Exception as exc:
            self.signals.failed.emit(
                str(
                    exc
                )
            )

            return

        self.signals.finished.emit(
            report
        )


class DiatomicEAMainWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        *,
        auto_probe: bool = True,
        status_root: str | Path | None = None,
    ) -> None:
        super().__init__()

        self.resources: CpuResources = (
            detect_cpu_resources()
        )

        self.calculation_queue = (
            CalculationQueue()
        )

        self.gui_job_specs: dict[
            str,
            GuiCalculationSpec,
        ] = {}

        self._active_process: QProcess | None = None
        self._active_process_phase: str | None = None
        self._active_job_id: str | None = None
        self._queue_stop_requested = False
        self._close_when_idle = False
        self._process_output = ""

        self.thread_pool = (
            QThreadPool.globalInstance()
        )

        self._active_probe: (
            ComputeProbe
            | None
        ) = None

        self._auto_probe = bool(
            auto_probe
        )

        self._probe_started = False

        self.status_root = (
            self._resolve_status_root(
                status_root
            )
        )

        self.setWindowTitle(
            APP_TITLE
        )

        self.resize(
            1180,
            760,
        )

        self.setMinimumSize(
            980,
            650,
        )

        self._build_ui()
        self._apply_style()

        self.status_timer = QTimer(
            self
        )

        self.status_timer.setInterval(
            1000
        )

        self.status_timer.timeout.connect(
            self.poll_production_status
        )

        self.status_timer.start()

        self.poll_production_status()

    @staticmethod
    def _resolve_status_root(
        status_root: str | Path | None,
    ) -> Path:
        if status_root is not None:
            return Path(
                status_root
            )

        local_app_data = os.environ.get(
            "LOCALAPPDATA"
        )

        if local_app_data:
            return (
                Path(
                    local_app_data
                )
                / "DiatomicEA"
                / "production_runs"
            )

        return (
            Path.home()
            / ".diatomic-ea"
            / "production_runs"
        )

    def showEvent(
        self,
        event,
    ) -> None:
        super().showEvent(
            event
        )

        if (
            self._auto_probe
            and not self._probe_started
        ):
            self._probe_started = True

            QTimer.singleShot(
                0,
                self.refresh_compute_status,
            )

    def _request_close_choice(
        self,
    ) -> str:
        """Ask what to do when the user closes during a calculation."""

        dialog = QMessageBox(
            self
        )

        dialog.setWindowTitle(
            "Calculation still running"
        )

        dialog.setIcon(
            QMessageBox.Warning
        )

        dialog.setText(
            "An electron-affinity calculation is still running."
        )

        dialog.setInformativeText(
            (
                "Closing the application during the calculation "
                "could interrupt the compute process. "
                "Keep the window open, or let the current "
                "calculation finish and close automatically."
            )
        )

        keep_button = dialog.addButton(
            "Keep window open",
            QMessageBox.RejectRole,
        )

        exit_button = dialog.addButton(
            "Exit after current calculation",
            QMessageBox.AcceptRole,
        )

        dialog.setDefaultButton(
            keep_button
        )

        dialog.exec_()

        if (
            dialog.clickedButton()
            is exit_button
        ):
            return "exit_after_current"

        return "keep_open"

    def _calculation_process_is_active(
        self,
    ) -> bool:
        """Return whether a real child calculation process is alive."""

        process = self._active_process

        if process is None:
            return False

        return (
            process.state()
            != QProcess.NotRunning
        )

    def closeEvent(
        self,
        event,
    ) -> None:
        """Prevent destruction only while a real child process is alive."""

        if not self._calculation_process_is_active():
            event.accept()
            return

        choice = (
            self._request_close_choice()
        )

        if (
            choice
            == "exit_after_current"
        ):
            self._queue_stop_requested = True
            self._close_when_idle = True

            self.progress_message.setText(
                (
                    "The application will close after "
                    "the current calculation finishes."
                )
            )

            self._update_queue_buttons()

        event.ignore()

    def _build_ui(
        self,
    ) -> None:
        central = QWidget(
            self
        )

        self.setCentralWidget(
            central
        )

        root = QVBoxLayout(
            central
        )

        root.setContentsMargins(
            28,
            24,
            28,
            24,
        )

        root.setSpacing(
            18
        )

        root.addLayout(
            self._build_header()
        )

        body = QHBoxLayout()

        body.setSpacing(
            18
        )

        left = QVBoxLayout()

        left.setSpacing(
            16
        )

        left.addWidget(
            self._build_compute_panel()
        )

        left.addWidget(
            self._build_new_calculation_panel()
        )

        left.addWidget(
            self._build_queue_panel(),
            1,
        )

        right = QVBoxLayout()

        right.addWidget(
            self._build_progress_panel(),
            1,
        )

        body.addLayout(
            left,
            5,
        )

        body.addLayout(
            right,
            6,
        )

        root.addLayout(
            body,
            1,
        )

    def _build_header(
        self,
    ) -> QHBoxLayout:
        layout = QHBoxLayout()

        title_box = QVBoxLayout()

        title = QLabel(
            APP_TITLE
        )

        title.setObjectName(
            "appTitle"
        )

        title_font = QFont()

        title_font.setPointSize(
            23
        )

        title_font.setWeight(
            QFont.DemiBold
        )

        title.setFont(
            title_font
        )

        tagline = QLabel(
            APP_TAGLINE
        )

        tagline.setObjectName(
            "secondaryText"
        )

        title_box.addWidget(
            title
        )

        title_box.addWidget(
            tagline
        )

        layout.addLayout(
            title_box
        )

        layout.addStretch(
            1
        )

        self.compute_badge = QLabel(
            "Compute: not checked"
        )

        self.compute_badge.setObjectName(
            "statusBadge"
        )

        self.compute_badge.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(
            self.compute_badge
        )

        return layout

    def _build_compute_panel(
        self,
    ) -> QGroupBox:
        group = QGroupBox(
            "Compute environment"
        )

        layout = QGridLayout(
            group
        )

        layout.setColumnStretch(
            1,
            1,
        )

        layout.addWidget(
            QLabel(
                "Backend"
            ),
            0,
            0,
        )

        self.backend_value = QLabel(
            "Not checked"
        )

        layout.addWidget(
            self.backend_value,
            0,
            1,
        )

        layout.addWidget(
            QLabel(
                "State"
            ),
            1,
            0,
        )

        self.compute_state_value = QLabel(
            "Not checked"
        )

        layout.addWidget(
            self.compute_state_value,
            1,
            1,
        )

        layout.addWidget(
            QLabel(
                "Versions"
            ),
            2,
            0,
        )

        self.compute_versions_value = QLabel(
            "--"
        )

        self.compute_versions_value.setWordWrap(
            True
        )

        layout.addWidget(
            self.compute_versions_value,
            2,
            1,
        )

        layout.addWidget(
            QLabel(
                "CPU"
            ),
            3,
            0,
        )

        self.cpu_value = QLabel(
            (
                f"{self.resources.physical_cores} physical / "
                f"{self.resources.logical_cores} logical cores"
            )
        )

        layout.addWidget(
            self.cpu_value,
            3,
            1,
        )

        self.refresh_compute_button = QPushButton(
            "Refresh"
        )

        self.refresh_compute_button.clicked.connect(
            self.refresh_compute_status
        )

        layout.addWidget(
            self.refresh_compute_button,
            4,
            1,
            alignment=Qt.AlignRight,
        )

        return group

    def _build_new_calculation_panel(
        self,
    ) -> QGroupBox:
        group = QGroupBox(
            "New calculation"
        )

        layout = QFormLayout(
            group
        )

        atom_row = QHBoxLayout()

        self.atom_a_input = QLineEdit()

        self.atom_a_input.setPlaceholderText(
            "e.g. Al"
        )

        self.atom_a_input.setMaxLength(
            3
        )

        self.atom_b_input = QLineEdit()

        self.atom_b_input.setPlaceholderText(
            "e.g. O"
        )

        self.atom_b_input.setMaxLength(
            3
        )

        atom_row.addWidget(
            self.atom_a_input
        )

        separator = QLabel(
            "+"
        )

        separator.setAlignment(
            Qt.AlignCenter
        )

        atom_row.addWidget(
            separator
        )

        atom_row.addWidget(
            self.atom_b_input
        )

        atom_widget = QWidget()

        atom_widget.setLayout(
            atom_row
        )

        layout.addRow(
            "Atoms",
            atom_widget,
        )

        range_row = QHBoxLayout()

        self.minimum_bond_spin = QDoubleSpinBox()

        self.minimum_bond_spin.setDecimals(
            2
        )

        self.minimum_bond_spin.setRange(
            0.20,
            8.00,
        )

        self.minimum_bond_spin.setSingleStep(
            0.05
        )

        self.minimum_bond_spin.setValue(
            0.75
        )

        self.minimum_bond_spin.setSuffix(
            " Å"
        )

        self.minimum_bond_spin.setToolTip(
            "Lower bound of the initial bond-length scan."
        )

        self.maximum_bond_spin = QDoubleSpinBox()

        self.maximum_bond_spin.setDecimals(
            2
        )

        self.maximum_bond_spin.setRange(
            0.20,
            8.00,
        )

        self.maximum_bond_spin.setSingleStep(
            0.05
        )

        self.maximum_bond_spin.setValue(
            3.00
        )

        self.maximum_bond_spin.setSuffix(
            " Å"
        )

        self.maximum_bond_spin.setToolTip(
            "Upper bound of the initial bond-length scan."
        )

        range_row.addWidget(
            self.minimum_bond_spin
        )

        range_row.addWidget(
            QLabel(
                "to"
            )
        )

        range_row.addWidget(
            self.maximum_bond_spin
        )

        range_widget = QWidget()

        range_widget.setLayout(
            range_row
        )

        layout.addRow(
            "Bond-length scan",
            range_widget,
        )

        self.spin_max_spin = QSpinBox()

        self.spin_max_spin.setRange(
            0,
            15,
        )

        self.spin_max_spin.setValue(
            5
        )

        self.spin_max_spin.setToolTip(
            "Highest 2S value included in the electronic-state scan."
        )

        layout.addRow(
            "Maximum spin (2S)",
            self.spin_max_spin,
        )

        self.worker_spin = QSpinBox()

        self.worker_spin.setMinimum(
            1
        )

        self.worker_spin.setMaximum(
            max(
                1,
                self.resources.physical_cores,
            )
        )

        self.worker_spin.setValue(
            min(
                self.resources.recommended_workers,
                self.resources.physical_cores,
            )
        )

        self.worker_spin.setSuffix(
            " workers"
        )

        layout.addRow(
            "Parallelism",
            self.worker_spin,
        )

        self.worker_hint = QLabel(
            (
                "CPU recommendation: "
                f"{self.resources.recommended_workers}. "
                "The production planner also checks RAM."
            )
        )

        self.worker_hint.setObjectName(
            "secondaryText"
        )

        self.worker_hint.setWordWrap(
            True
        )

        layout.addRow(
            "",
            self.worker_hint,
        )

        self.add_queue_button = QPushButton(
            "Add to queue"
        )

        self.add_queue_button.clicked.connect(
            self.add_current_molecule_to_queue
        )

        layout.addRow(
            "",
            self.add_queue_button,
        )

        return group

    def _build_queue_panel(
        self,
    ) -> QGroupBox:
        group = QGroupBox(
            "Calculation queue"
        )

        layout = QVBoxLayout(
            group
        )

        self.queue_list = QListWidget()

        self.queue_list.setAlternatingRowColors(
            True
        )

        self.queue_list.setSelectionMode(
            QListWidget.SingleSelection
        )

        self.queue_list.currentItemChanged.connect(
            lambda _current, _previous: (
                self._update_queue_buttons()
            )
        )

        layout.addWidget(
            self.queue_list,
            1,
        )

        controls = QHBoxLayout()

        self.queue_up_button = QPushButton(
            "Move up"
        )

        self.queue_down_button = QPushButton(
            "Move down"
        )

        self.queue_remove_button = QPushButton(
            "Remove"
        )

        self.queue_up_button.clicked.connect(
            self.move_selected_up
        )

        self.queue_down_button.clicked.connect(
            self.move_selected_down
        )

        self.queue_remove_button.clicked.connect(
            self.remove_selected_queue_item
        )

        controls.addWidget(
            self.queue_up_button
        )

        controls.addWidget(
            self.queue_down_button
        )

        controls.addStretch(
            1
        )

        controls.addWidget(
            self.queue_remove_button
        )

        layout.addLayout(
            controls
        )

        execution_controls = QHBoxLayout()

        self.start_queue_button = QPushButton(
            "Start queue"
        )

        self.stop_queue_button = QPushButton(
            "Stop after current"
        )

        self.resume_queue_button = QPushButton(
            "Resume queue"
        )

        self.start_queue_button.clicked.connect(
            self.start_queue
        )

        self.stop_queue_button.clicked.connect(
            self.stop_queue_after_current
        )

        self.resume_queue_button.clicked.connect(
            self.resume_queue
        )

        self.stop_queue_button.setEnabled(
            False
        )

        self.resume_queue_button.setEnabled(
            False
        )

        execution_controls.addWidget(
            self.start_queue_button
        )

        execution_controls.addWidget(
            self.stop_queue_button
        )

        execution_controls.addWidget(
            self.resume_queue_button
        )

        layout.addLayout(
            execution_controls
        )

        self.queue_summary = QLabel(
            "0 jobs queued"
        )

        self.queue_summary.setObjectName(
            "secondaryText"
        )

        layout.addWidget(
            self.queue_summary
        )

        return group

    def _metric_card(
        self,
        title: str,
    ) -> tuple[
        QFrame,
        QLabel,
    ]:
        frame = QFrame()

        frame.setObjectName(
            "metricCard"
        )

        layout = QVBoxLayout(
            frame
        )

        layout.setContentsMargins(
            14,
            12,
            14,
            12,
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "metricTitle"
        )

        value = QLabel(
            "--"
        )

        value.setObjectName(
            "metricValue"
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            value
        )

        return (
            frame,
            value,
        )

    def _build_progress_panel(
        self,
    ) -> QGroupBox:
        group = QGroupBox(
            "Current calculation"
        )

        layout = QVBoxLayout(
            group
        )

        layout.setSpacing(
            18
        )

        self.active_job_label = QLabel(
            "No calculation running"
        )

        self.active_job_label.setObjectName(
            "currentJob"
        )

        layout.addWidget(
            self.active_job_label
        )

        self.stage_label = QLabel(
            "Idle"
        )

        self.stage_label.setObjectName(
            "stageLabel"
        )

        layout.addWidget(
            self.stage_label
        )

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            1000,
        )

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setFormat(
            "0.0%"
        )

        self.progress_bar.setMinimumHeight(
            28
        )

        layout.addWidget(
            self.progress_bar
        )

        metrics = QGridLayout()

        (
            task_card,
            self.completed_value,
        ) = self._metric_card(
            "Tasks"
        )

        (
            rate_card,
            self.rate_value,
        ) = self._metric_card(
            "Throughput"
        )

        (
            eta_card,
            self.eta_value,
        ) = self._metric_card(
            "ETA"
        )

        (
            elapsed_card,
            self.elapsed_value,
        ) = self._metric_card(
            "Stage elapsed"
        )

        metrics.addWidget(
            task_card,
            0,
            0,
        )

        metrics.addWidget(
            rate_card,
            0,
            1,
        )

        metrics.addWidget(
            eta_card,
            1,
            0,
        )

        metrics.addWidget(
            elapsed_card,
            1,
            1,
        )

        layout.addLayout(
            metrics
        )

        status_title = QLabel(
            "Status"
        )

        status_title.setObjectName(
            "metricTitle"
        )

        layout.addWidget(
            status_title
        )

        self.progress_message = QLabel(
            "Waiting for a calculation."
        )

        self.progress_message.setObjectName(
            "statusMessage"
        )

        self.progress_message.setWordWrap(
            True
        )

        self.progress_message.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum,
        )

        layout.addWidget(
            self.progress_message
        )

        layout.addStretch(
            1
        )

        self.status_source_label = QLabel(
            ""
        )

        self.status_source_label.setObjectName(
            "secondaryText"
        )

        self.status_source_label.setWordWrap(
            True
        )

        layout.addWidget(
            self.status_source_label
        )

        return group

    def _apply_style(
        self,
    ) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3f3f0;
            }

            QWidget {
                color: #282a27;
                font-size: 10pt;
            }

            QLabel#secondaryText {
                color: #696d67;
            }

            QLabel#statusBadge {
                background: #ededE9;
                border: 1px solid #bfc2bb;
                border-radius: 12px;
                padding: 7px 12px;
                font-weight: 600;
            }

            QLabel#currentJob {
                font-size: 16pt;
                font-weight: 600;
            }

            QLabel#stageLabel {
                color: #5c605a;
                font-size: 12pt;
            }

            QLabel#metricTitle {
                color: #747871;
                font-size: 9pt;
            }

            QLabel#metricValue {
                font-size: 16pt;
                font-weight: 600;
            }

            QLabel#statusMessage {
                background: #f8f8f5;
                border: 1px solid #dcddd7;
                border-radius: 7px;
                padding: 12px;
            }

            QGroupBox {
                background: #ffffff;
                border: 1px solid #d5d7d1;
                border-radius: 9px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 5px;
            }

            QFrame#metricCard {
                background: #f8f8f5;
                border: 1px solid #dcddd7;
                border-radius: 8px;
            }

            QLineEdit,
            QSpinBox,
            QListWidget {
                background: #ffffff;
                border: 1px solid #cdd0c9;
                border-radius: 6px;
                padding: 7px;
            }

            QPushButton {
                background: #414a42;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #515c52;
            }

            QPushButton:disabled {
                background: #b9bcb7;
                color: #eeeeeb;
            }

            QProgressBar {
                background: #e9eae5;
                border: 1px solid #c7cac3;
                border-radius: 7px;
                text-align: center;
            }

            QProgressBar::chunk {
                background: #647267;
                border-radius: 6px;
            }
            """
        )

    def refresh_compute_status(
        self,
    ) -> None:
        if self._active_probe is not None:
            return

        self.compute_badge.setText(
            "Compute: checking..."
        )

        self.backend_value.setText(
            "Checking..."
        )

        self.compute_state_value.setText(
            "Checking..."
        )

        self.compute_versions_value.setText(
            "--"
        )

        self.refresh_compute_button.setEnabled(
            False
        )

        probe = ComputeProbe()

        probe.signals.finished.connect(
            self._compute_probe_finished
        )

        probe.signals.failed.connect(
            self._compute_probe_failed
        )

        self._active_probe = probe

        self.thread_pool.start(
            probe
        )

    def _compute_probe_finished(
        self,
        report: object,
    ) -> None:
        self._active_probe = None

        self.refresh_compute_button.setEnabled(
            True
        )

        if not isinstance(
            report,
            ComputeEnvironmentReport,
        ):
            self._compute_probe_failed(
                "Unexpected compute environment result."
            )

            return

        backend_text = (
            report.backend
        )

        if report.distribution:
            backend_text += (
                " / "
                + report.distribution
            )

        self.backend_value.setText(
            backend_text
        )

        self.compute_state_value.setText(
            report.state.value
        )

        self.compute_versions_value.setText(
            (
                "Python "
                + (
                    report.python_version
                    or "?"
                )
                + "   |   PySCF "
                + (
                    report.pyscf_version
                    or "?"
                )
                + "   |   BSE "
                + (
                    report
                    .basis_set_exchange_version
                    or "?"
                )
            )
        )

        if report.ready:
            self.compute_badge.setText(
                "Compute: ready"
            )

            self.compute_badge.setStyleSheet(
                "background: #e7eee7;"
                "border-color: #9faf9f;"
            )

        else:
            self.compute_badge.setText(
                "Compute: attention required"
            )

            self.compute_badge.setStyleSheet(
                "background: #f2e8dd;"
                "border-color: #c8a889;"
            )

        self.compute_badge.setToolTip(
            report.message
        )

    def _compute_probe_failed(
        self,
        message: str,
    ) -> None:
        self._active_probe = None

        self.refresh_compute_button.setEnabled(
            True
        )

        self.compute_badge.setText(
            "Compute: check failed"
        )

        self.backend_value.setText(
            "Unavailable"
        )

        self.compute_state_value.setText(
            "Error"
        )

        self.compute_versions_value.setText(
            message
        )

    def add_current_molecule_to_queue(
        self,
    ) -> None:
        try:
            molecule = DiatomicMolecule(
                self.atom_a_input.text(),
                self.atom_b_input.text(),
            )

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid molecule",
                str(
                    exc
                ),
            )

            return

        minimum = float(
            self.minimum_bond_spin.value()
        )

        maximum = float(
            self.maximum_bond_spin.value()
        )

        if maximum <= minimum:
            QMessageBox.warning(
                self,
                "Invalid bond-length range",
                (
                    "The upper bond length must be "
                    "greater than the lower bond length."
                ),
            )

            return

        job = CalculationJob(
            molecule=molecule
        )

        spec = GuiCalculationSpec(
            job_id=job.job_id,
            molecule=molecule,
            minimum_angstrom=minimum,
            maximum_angstrom=maximum,
            spin_max=int(
                self.spin_max_spin.value()
            ),
            workers=int(
                self.worker_spin.value()
            ),
            run_id=make_gui_run_id(
                molecule,
                job.job_id,
            ),
        )

        self.calculation_queue.add(
            job
        )

        self.gui_job_specs[
            job.job_id
        ] = spec

        self.atom_a_input.clear()
        self.atom_b_input.clear()

        self.refresh_queue_view()

    def refresh_queue_view(
        self,
    ) -> None:
        selected_id = (
            self._selected_job_id()
        )

        self.queue_list.clear()

        selected_row = -1

        for index, job in enumerate(
            self.calculation_queue.jobs
        ):
            item = QListWidgetItem(
                (
                    f"{job.molecule.formula}"
                    "    ·    "
                    f"{calculation_mode_label(job.mode)}"
                    "    ·    "
                    f"{job_status_label(job.status)}"
                )
            )

            spec = self.gui_job_specs.get(
                job.job_id
            )

            if spec is not None:
                item.setToolTip(
                    (
                        f"Bond-length scan: "
                        f"{spec.minimum_angstrom:.2f}–"
                        f"{spec.maximum_angstrom:.2f} Å\n"
                        f"Maximum spin (2S): {spec.spin_max}\n"
                        f"Workers: {spec.workers}"
                    )
                )
            else:
                item.setToolTip(
                    job.job_id
                )

            item.setData(
                Qt.UserRole,
                job.job_id,
            )

            self.queue_list.addItem(
                item
            )

            if (
                selected_id
                == job.job_id
            ):
                selected_row = index

        if selected_row >= 0:
            self.queue_list.setCurrentRow(
                selected_row
            )

        count = len(
            self.calculation_queue
        )

        waiting = sum(
            1
            for job
            in self.calculation_queue.jobs
            if job.status is JobStatus.QUEUED
        )

        self.queue_summary.setText(
            (
                f"{count} total · {waiting} waiting"
            )
        )

        self._update_queue_buttons()

    def _selected_job_id(
        self,
    ) -> str | None:
        item = (
            self.queue_list.currentItem()
        )

        if item is None:
            return None

        value = item.data(
            Qt.UserRole
        )

        if value is None:
            return None

        return str(
            value
        )

    def remove_selected_queue_item(
        self,
    ) -> None:
        job_id = (
            self._selected_job_id()
        )

        if job_id is None:
            return

        try:
            removed = self.calculation_queue.remove(
                job_id
            )

            self.gui_job_specs.pop(
                removed.job_id,
                None,
            )

        except (
            ValueError,
            KeyError,
        ) as exc:
            QMessageBox.warning(
                self,
                "Cannot remove job",
                str(
                    exc
                ),
            )

            return

        self.refresh_queue_view()

    def move_selected_up(
        self,
    ) -> None:
        job_id = (
            self._selected_job_id()
        )

        if job_id is None:
            return

        try:
            self.calculation_queue.move_up(
                job_id
            )

        except (
            ValueError,
            KeyError,
        ) as exc:
            QMessageBox.warning(
                self,
                "Cannot move job",
                str(
                    exc
                ),
            )

            return

        self.refresh_queue_view()

    def move_selected_down(
        self,
    ) -> None:
        job_id = (
            self._selected_job_id()
        )

        if job_id is None:
            return

        try:
            self.calculation_queue.move_down(
                job_id
            )

        except (
            ValueError,
            KeyError,
        ) as exc:
            QMessageBox.warning(
                self,
                "Cannot move job",
                str(
                    exc
                ),
            )

            return

        self.refresh_queue_view()

    def _has_waiting_jobs(
        self,
    ) -> bool:
        return any(
            job.status is JobStatus.QUEUED
            for job
            in self.calculation_queue.jobs
        )

    def _update_queue_buttons(
        self,
    ) -> None:
        active = (
            self._active_job_id
            is not None
        )

        waiting = (
            self._has_waiting_jobs()
        )

        selected_queued = False

        selected_id = (
            self._selected_job_id()
        )

        if selected_id is not None:
            try:
                selected_job = (
                    self.calculation_queue.get(
                        selected_id
                    )
                )

            except KeyError:
                selected_job = None

            if selected_job is not None:
                selected_queued = (
                    selected_job.status
                    is JobStatus.QUEUED
                )

        self.queue_up_button.setEnabled(
            selected_queued
        )

        self.queue_down_button.setEnabled(
            selected_queued
        )

        self.queue_remove_button.setEnabled(
            selected_queued
        )

        self.add_queue_button.setEnabled(
            not self._close_when_idle
        )

        self.start_queue_button.setEnabled(
            waiting
            and not active
            and not self._queue_stop_requested
            and not self._close_when_idle
        )

        self.stop_queue_button.setEnabled(
            active
            and not self._queue_stop_requested
            and not self._close_when_idle
        )

        self.resume_queue_button.setEnabled(
            waiting
            and not active
            and self._queue_stop_requested
            and not self._close_when_idle
        )

    def start_queue(
        self,
    ) -> None:
        if (
            self._active_job_id is not None
            or self._close_when_idle
        ):
            return

        self._queue_stop_requested = False

        self._launch_next_waiting_job()

    def resume_queue(
        self,
    ) -> None:
        if (
            self._active_job_id is not None
            or self._close_when_idle
        ):
            return

        self._queue_stop_requested = False

        self.progress_message.setText(
            "Queue resumed."
        )

        self._launch_next_waiting_job()

    def stop_queue_after_current(
        self,
    ) -> None:
        self._queue_stop_requested = True

        if self._active_job_id is None:
            self.progress_message.setText(
                "Queue stopped."
            )

        else:
            self.progress_message.setText(
                (
                    "The queue will stop after the "
                    "current calculation finishes."
                )
            )

        self._update_queue_buttons()

    def _launch_next_waiting_job(
        self,
    ) -> None:
        if self._queue_stop_requested:
            self._update_queue_buttons()
            return

        if self._active_job_id is not None:
            return

        job = self.calculation_queue.start_next()

        if job is None:
            self.active_job_label.setText(
                "Queue complete"
            )

            self.stage_label.setText(
                "Idle"
            )

            self.progress_message.setText(
                "No calculations are waiting."
            )

            self._update_queue_buttons()

            return

        spec = self.gui_job_specs.get(
            job.job_id
        )

        if spec is None:
            job.transition_to(
                JobStatus.FAILED
            )

            self.refresh_queue_view()

            self.progress_message.setText(
                "Queued calculation settings are missing."
            )

            QTimer.singleShot(
                0,
                self._launch_next_waiting_job,
            )

            return

        self._active_job_id = (
            job.job_id
        )

        self.active_job_label.setText(
            spec.molecule.formula
        )

        self.stage_label.setText(
            "Preparing calculation"
        )

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setFormat(
            "0.0%"
        )

        self.completed_value.setText(
            "-- / --"
        )

        self.rate_value.setText(
            "-- tasks/s"
        )

        self.eta_value.setText(
            "--"
        )

        self.elapsed_value.setText(
            "--"
        )

        self.progress_message.setText(
            (
                "Checking the calculation setup "
                "and preparing the run."
            )
        )

        self.refresh_queue_view()

        command = build_plan_command(
            spec,
            output_root=self.status_root,
        )

        self._start_process(
            command,
            phase="planning",
        )

    def _active_spec(
        self,
    ) -> GuiCalculationSpec | None:
        if self._active_job_id is None:
            return None

        return self.gui_job_specs.get(
            self._active_job_id
        )

    def _start_process(
        self,
        command: ProcessCommand,
        *,
        phase: str,
    ) -> None:
        if self._active_process is not None:
            raise RuntimeError(
                "A GUI calculation process is already active."
            )

        process = QProcess(
            self
        )

        process.setProcessChannelMode(
            QProcess.MergedChannels
        )

        process.setProgram(
            command.program
        )

        process.setArguments(
            list(
                command.arguments
            )
        )

        process.readyReadStandardOutput.connect(
            self._read_process_output
        )

        process.finished.connect(
            self._process_finished
        )

        process.errorOccurred.connect(
            self._process_error
        )

        self._process_output = ""
        self._active_process_phase = phase
        self._active_process = process

        process.start()

    def _read_process_output(
        self,
    ) -> None:
        process = self._active_process

        if process is None:
            return

        raw = bytes(
            process.readAllStandardOutput()
        )

        if not raw:
            return

        text = raw.decode(
            "utf-8",
            errors="replace",
        )

        self._process_output = (
            self._process_output
            + text
        )[-16000:]

    def _process_error(
        self,
        error,
    ) -> None:
        if (
            error
            != QProcess.FailedToStart
        ):
            return

        self._release_process()

        self._finish_active_job(
            success=False,
            message=(
                "Could not start the calculation process."
            ),
        )

    def _release_process(
        self,
    ) -> tuple[
        str | None,
        str,
    ]:
        phase = (
            self._active_process_phase
        )

        output = (
            self._process_output
        )

        process = self._active_process

        self._active_process = None
        self._active_process_phase = None
        self._process_output = ""

        if process is not None:
            process.deleteLater()

        return (
            phase,
            output,
        )

    def _process_finished(
        self,
        exit_code: int,
        exit_status,
    ) -> None:
        if self._active_process is None:
            return

        phase, output = (
            self._release_process()
        )

        success = (
            exit_code == 0
            and exit_status
            == QProcess.NormalExit
        )

        if not success:
            lines = [
                line.strip()
                for line
                in output.splitlines()
                if line.strip()
            ]

            detail = (
                lines[-1]
                if lines
                else (
                    "Calculation process exited "
                    f"with code {exit_code}."
                )
            )

            self._finish_active_job(
                success=False,
                message=humanize_status_message(
                    detail
                ),
            )

            return

        if phase == "planning":
            self._launch_active_calculation()

            return

        if phase == "calculation":
            self._finish_active_job(
                success=True,
                message=(
                    "Electron-affinity calculation completed."
                ),
            )

            return

        self._finish_active_job(
            success=False,
            message=(
                "Unknown calculation-process state."
            ),
        )

    def _launch_active_calculation(
        self,
    ) -> None:
        spec = self._active_spec()

        if spec is None:
            self._finish_active_job(
                success=False,
                message=(
                    "Calculation settings are missing."
                ),
            )

            return

        plan_path = (
            spec.plan_path(
                self.status_root
            )
        )

        if not plan_path.is_file():
            self._finish_active_job(
                success=False,
                message=(
                    "Calculation setup was not created."
                ),
            )

            return

        self.stage_label.setText(
            "Preparing calculation"
        )

        self.progress_message.setText(
            "Starting electron-affinity calculation."
        )

        command = build_run_command(
            spec,
            output_root=self.status_root,
        )

        self._start_process(
            command,
            phase="calculation",
        )

    def _finish_active_job(
        self,
        *,
        success: bool,
        message: str,
    ) -> None:
        job_id = (
            self._active_job_id
        )

        self._active_job_id = None

        if job_id is not None:
            try:
                job = (
                    self.calculation_queue.get(
                        job_id
                    )
                )

                if (
                    job.status
                    is JobStatus.RUNNING
                ):
                    job.transition_to(
                        (
                            JobStatus.COMPLETED
                            if success
                            else JobStatus.FAILED
                        )
                    )

            except (
                KeyError,
                ValueError,
            ):
                pass

        self.progress_message.setText(
            humanize_status_message(
                message
            )
        )

        self.refresh_queue_view()

        if self._close_when_idle:
            self._close_when_idle = False
            self._queue_stop_requested = True

            self.active_job_label.setText(
                "Calculation finished"
            )

            self.stage_label.setText(
                "Idle"
            )

            self.progress_message.setText(
                (
                    "Current calculation finished. "
                    "Closing DiatomicEA."
                )
            )

            self._update_queue_buttons()

            QTimer.singleShot(
                0,
                self.close,
            )

            return

        if self._queue_stop_requested:
            self.active_job_label.setText(
                "Queue stopped"
            )

            self.stage_label.setText(
                "Idle"
            )

            self._update_queue_buttons()

            return

        QTimer.singleShot(
            0,
            self._launch_next_waiting_job,
        )

    def poll_production_status(
        self,
    ) -> None:
        active_spec = (
            self._active_spec()
        )

        if active_spec is not None:
            active_path = (
                active_spec.status_path(
                    self.status_root
                )
            )

            if active_path.is_file():
                path = active_path
            else:
                return

        else:
            path = discover_latest_status(
                self.status_root
            )

            if path is None:
                return

        try:
            snapshot = (
                read_production_status(
                    path
                )
            )

        except ValueError:
            return

        self.apply_production_status(
            snapshot
        )

    def apply_production_status(
        self,
        snapshot: ProductionStatusSnapshot,
    ) -> None:
        run_id = None

        if snapshot.source_path:
            source = Path(
                snapshot.source_path
            )

            if len(
                source.parents
            ) >= 3:
                run_id = (
                    source
                    .parent
                    .parent
                    .name
                )

        self.active_job_label.setText(
            run_id
            or snapshot.state.title()
        )

        self.stage_label.setText(
            snapshot.stage_text
        )

        progress_value = int(
            round(
                snapshot.progress_percent
                * 10.0
            )
        )

        self.progress_bar.setValue(
            progress_value
        )

        self.progress_bar.setFormat(
            f"{snapshot.progress_percent:.1f}%"
        )

        self.completed_value.setText(
            snapshot.completed_text
        )

        self.rate_value.setText(
            snapshot.rate_text
        )

        self.eta_value.setText(
            snapshot.eta_text
        )

        self.elapsed_value.setText(
            snapshot.elapsed_text
        )

        self.progress_message.setText(
            snapshot.message
            or snapshot.state.title()
        )

        if snapshot.source_path:
            self.status_source_label.setText(
                (
                    "Telemetry: "
                    + snapshot.source_path
                )
            )

    def queue_count(
        self,
    ) -> int:
        """Return current queue size."""

        return len(
            self.calculation_queue
        )


def build_application() -> QApplication:
    """Create or reuse QApplication."""

    existing = (
        QApplication.instance()
    )

    if existing is not None:
        return existing

    app = QApplication(
        [
            sys.argv[0]
        ]
    )

    app.setApplicationName(
        APP_TITLE
    )

    app.setOrganizationName(
        "DiatomicEA"
    )

    return app


def main(
    argv: list[str] | None = None,
) -> int:
    """Launch the PyQt5 desktop application."""

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

    window = DiatomicEAMainWindow(
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
