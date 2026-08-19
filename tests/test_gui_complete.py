"""Tests for the completed desktop application."""

import csv
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

from diatomic_ea.desktop_gui import (
    DiatomicEADesktopWindow,
)
from diatomic_ea.gui_execution import (
    GuiCalculationSpec,
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


FIELDS = [
    "molecule",
    "model_id",
    "n_functionals",
    "ea_pbe_eV",
    "ea_b3lyp_eV",
    "ea_pbe0_eV",
    "ea_tpssh_eV",
    "median_qz_eV",
    "half_range_qz_eV",
    "bias_correction_eV",
    "predicted_ea_eV",
    "scale_eV",
    "pi80_half_width_eV",
    "pi80_lower_eV",
    "pi80_upper_eV",
    "pi90_half_width_eV",
    "pi90_lower_eV",
    "pi90_upper_eV",
    "pi95_half_width_eV",
    "pi95_lower_eV",
    "pi95_upper_eV",
]


ROW = {
    "molecule": "OH",
    "model_id": "internal-only",
    "n_functionals": "4",
    "ea_pbe_eV": "1.9829184758392013",
    "ea_b3lyp_eV": "1.8566987970372117",
    "ea_pbe0_eV": "1.6131671704161916",
    "ea_tpssh_eV": "1.6239377722463588",
    "median_qz_eV": "1.7403182846417853",
    "half_range_qz_eV": "0.18487565271150486",
    "bias_correction_eV": "0.0825",
    "predicted_ea_eV": "1.8228182846417853",
    "scale_eV": "0.1517492341243265",
    "pi80_half_width_eV": "0.20243347832185157",
    "pi80_lower_eV": "1.6203848063199338",
    "pi80_upper_eV": "2.025251762963637",
    "pi90_half_width_eV": "0.2843780647489879",
    "pi90_lower_eV": "1.5384402198927973",
    "pi90_upper_eV": "2.1071963493907733",
    "pi95_half_width_eV": "0.33172382579577775",
    "pi95_lower_eV": "1.4910944588460076",
    "pi95_upper_eV": "2.1545421104375633",
}


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


def build_completed_window(
    tmp_path,
):
    window = DiatomicEADesktopWindow(
        auto_probe=False,
        status_root=tmp_path,
        persist_gui_state=False,
    )

    molecule = DiatomicMolecule(
        "O",
        "H",
    )

    job = CalculationJob(
        molecule=molecule,
        job_id="completed-job",
        status=JobStatus.COMPLETED,
    )

    spec = GuiCalculationSpec(
        job_id=job.job_id,
        molecule=molecule,
        minimum_angstrom=0.75,
        maximum_angstrom=1.35,
        spin_max=3,
        workers=2,
        run_id="oh-result-test",
    )

    window.calculation_queue = (
        CalculationQueue(
            (
                job,
            )
        )
    )

    window.gui_job_specs = {
        job.job_id: spec,
    }

    result_path = spec.final_result_path(
        tmp_path
    )

    result_path.parent.mkdir(
        parents=True,
    )

    with result_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        writer.writerow(
            ROW
        )

    window.refresh_queue_view()

    return (
        window,
        job,
    )


def test_completed_result_is_displayed(
    app,
    tmp_path,
) -> None:
    window, job = build_completed_window(
        tmp_path
    )

    window.show_result_for_job(
        job.job_id
    )

    assert (
        window.result_ea_value.text()
        == "1.8228 eV"
    )

    assert (
        window.result_pi90.text()
        == "1.5384 to 2.1072 eV"
    )

    assert (
        window.result_half_range.text()
        == "±0.1849 eV"
    )

    assert (
        window.status_source_label.isHidden()
        is True
    )

    visible = " ".join(
        (
            window.result_molecule_label.text(),
            window.result_message.text(),
        )
    ).casefold()

    assert "schema" not in visible
    assert "qzvpd" not in visible

    window.close()


def test_failed_job_can_be_retried(
    app,
    tmp_path,
) -> None:
    window = DiatomicEADesktopWindow(
        auto_probe=False,
        status_root=tmp_path,
        persist_gui_state=False,
    )

    molecule = DiatomicMolecule(
        "Al",
        "O",
    )

    failed = CalculationJob(
        molecule=molecule,
        job_id="failed-job",
        status=JobStatus.FAILED,
    )

    spec = GuiCalculationSpec(
        job_id=failed.job_id,
        molecule=molecule,
        minimum_angstrom=1.0,
        maximum_angstrom=2.5,
        spin_max=5,
        workers=2,
        run_id="alo-failed",
    )

    window.calculation_queue = (
        CalculationQueue(
            (
                failed,
            )
        )
    )

    window.gui_job_specs = {
        failed.job_id: spec,
    }

    assert (
        window._requeue_failed_jobs()
        == 1
    )

    assert (
        window.calculation_queue.jobs[
            0
        ].status
        is JobStatus.QUEUED
    )

    window.close()


def test_preferences_survive_restart(
    app,
    tmp_path,
) -> None:
    first = DiatomicEADesktopWindow(
        auto_probe=False,
        status_root=tmp_path,
        persist_gui_state=True,
    )

    first.minimum_bond_spin.setValue(
        0.85
    )

    first.maximum_bond_spin.setValue(
        2.75
    )

    first.spin_max_spin.setValue(
        7
    )

    first._save_preferences()

    first.close()

    second = DiatomicEADesktopWindow(
        auto_probe=False,
        status_root=tmp_path,
        persist_gui_state=True,
    )

    assert (
        second.minimum_bond_spin.value()
        == pytest.approx(
            0.85
        )
    )

    assert (
        second.maximum_bond_spin.value()
        == pytest.approx(
            2.75
        )
    )

    assert (
        second.spin_max_spin.value()
        == 7
    )

    second.close()



def test_internal_run_identifier_is_not_displayed(
    app,
    tmp_path,
) -> None:
    from diatomic_ea.gui_state import (
        production_status_from_mapping,
    )

    window = DiatomicEADesktopWindow(
        auto_probe=False,
        status_root=tmp_path,
        persist_gui_state=False,
    )

    source = (
        tmp_path
        / "OH"
        / "oh-full-schema-f-v1"
        / "logs"
        / "production_status.json"
    )

    snapshot = production_status_from_mapping(
        {
            "state": "completed",
            "stage": "export",
            "percent": 100.0,
            "message": "Calculation completed.",
        },
        source_path=str(
            source
        ),
    )

    window.apply_production_status(
        snapshot
    )

    assert (
        "schema"
        not in window.active_job_label.text().casefold()
    )

    assert (
        "qzvpd"
        not in window.active_job_label.text().casefold()
    )

    assert (
        window.active_job_label.text()
        == "Latest completed calculation"
    )

    window.close()
