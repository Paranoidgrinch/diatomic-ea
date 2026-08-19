"""Real desktop integration smoke using a completed calculation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

from diatomic_ea.desktop_gui import (
    DiatomicEADesktopWindow,
)
from diatomic_ea.gui import (
    build_application,
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


def run_smoke(
    plan_path: str | Path,
    *,
    atom_a: str,
    atom_b: str,
) -> None:
    plan_file = Path(
        plan_path
    ).resolve()

    if not plan_file.is_file():
        raise RuntimeError(
            "Production plan was not found."
        )

    payload = json.loads(
        plan_file.read_text(
            encoding="utf-8"
        )
    )

    molecule = DiatomicMolecule(
        atom_a,
        atom_b,
    )

    if payload["molecule"] != molecule.formula:
        raise RuntimeError(
            "Plan molecule does not match."
        )

    output_root = (
        plan_file.parent.parent.parent
    )

    job_id = "gui-resume-smoke"

    job = CalculationJob(
        molecule=molecule,
        job_id=job_id,
    )

    spec = GuiCalculationSpec(
        job_id=job_id,
        molecule=molecule,
        minimum_angstrom=float(
            payload[
                "minimum_angstrom"
            ]
        ),
        maximum_angstrom=float(
            payload[
                "maximum_angstrom"
            ]
        ),
        spin_max=int(
            payload[
                "spin_max"
            ]
        ),
        workers=int(
            payload[
                "requested_workers"
            ]
        ),
        threads_per_worker=int(
            payload[
                "threads_per_worker"
            ]
        ),
        run_id=str(
            payload[
                "run_id"
            ]
        ),
    )

    app = build_application()

    window = DiatomicEADesktopWindow(
        auto_probe=False,
        status_root=output_root,
        persist_gui_state=False,
    )

    window.calculation_queue = (
        CalculationQueue(
            (
                job,
            )
        )
    )

    window.gui_job_specs = {
        job_id: spec,
    }

    window.refresh_queue_view()

    window.show()

    app.processEvents()

    window.start_queue()

    deadline = (
        time.monotonic()
        + 120.0
    )

    while time.monotonic() < deadline:
        app.processEvents()

        if (
            job.status
            in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            )
            and window._active_process is None
        ):
            break

        time.sleep(
            0.02
        )

    if job.status is not JobStatus.COMPLETED:
        detail = (
            window.progress_message.text()
        )

        window._active_job_id = None

        window.close()

        raise RuntimeError(
            (
                "Desktop integration smoke failed: "
                + detail
            )
        )

    window.show_result_for_job(
        job_id
    )

    app.processEvents()

    if "eV" not in window.result_ea_value.text():
        raise RuntimeError(
            "Completed EA was not displayed."
        )

    print(
        "Molecule:",
        molecule.formula,
    )

    print(
        "Displayed EA:",
        window.result_ea_value.text(),
    )

    print(
        "90% prediction interval:",
        window.result_pi90.text(),
    )

    print(
        "Queue state:",
        job.status.value,
    )

    print(
        "Production subprocess active:",
        window._active_process is not None,
    )

    print(
        "Status: PASS"
    )

    window.close()

    app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        required=True,
    )

    parser.add_argument(
        "--atom-a",
        required=True,
    )

    parser.add_argument(
        "--atom-b",
        required=True,
    )

    arguments = parser.parse_args()

    try:
        run_smoke(
            arguments.plan,
            atom_a=arguments.atom_a,
            atom_b=arguments.atom_b,
        )
    except Exception as exc:
        print(
            "Status: FAIL"
        )

        print(
            "Error:",
            str(
                exc
            ),
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
