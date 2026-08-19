"""PyQt5 desktop interface for DiatomicEA."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PyQt5.QtCore import (
    QObject,
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
from diatomic_ea.gui_state import (
    ProductionStatusSnapshot,
    discover_latest_status,
    read_production_status,
)
from diatomic_ea.jobs import (
    CalculationJob,
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

        self.calculation_queue.add(
            CalculationJob(
                molecule=molecule
            )
        )

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
                    f"{job.mode.value}"
                    "    ·    "
                    f"{job.status.value}"
                )
            )

            item.setData(
                Qt.UserRole,
                job.job_id,
            )

            item.setToolTip(
                job.job_id
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

        self.queue_summary.setText(
            (
                f"{count} job queued"
                if count == 1
                else f"{count} jobs queued"
            )
        )

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
            self.calculation_queue.remove(
                job_id
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

    def poll_production_status(
        self,
    ) -> None:
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
