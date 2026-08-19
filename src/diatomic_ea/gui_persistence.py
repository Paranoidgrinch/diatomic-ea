"""Persistent settings and queue recovery for the desktop GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.gui_execution import (
    GuiCalculationSpec,
)
from diatomic_ea.jobs import (
    CalculationJob,
    CalculationMode,
    JobStatus,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)


GUI_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class GuiPreferences:
    minimum_angstrom: float
    maximum_angstrom: float
    spin_max: int
    workers: int


@dataclass(frozen=True, slots=True)
class LoadedQueueSession:
    jobs: tuple[CalculationJob, ...]
    specs: dict[str, GuiCalculationSpec]
    recovered_running_jobs: int


def default_preferences(
    workers: int,
) -> GuiPreferences:
    return GuiPreferences(
        minimum_angstrom=0.70,
        maximum_angstrom=3.00,
        spin_max=5,
        workers=max(
            1,
            int(
                workers
            ),
        ),
    )


def _write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def save_preferences(
    path: str | Path,
    preferences: GuiPreferences,
) -> None:
    _write_json(
        Path(
            path
        ),
        {
            "version": GUI_STATE_VERSION,
            "minimum_angstrom": preferences.minimum_angstrom,
            "maximum_angstrom": preferences.maximum_angstrom,
            "spin_max": preferences.spin_max,
            "workers": preferences.workers,
        },
    )


def load_preferences(
    path: str | Path,
    *,
    fallback_workers: int,
) -> GuiPreferences:
    fallback = default_preferences(
        fallback_workers
    )

    source = Path(
        path
    )

    if not source.is_file():
        return fallback

    try:
        payload = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )

        minimum = float(
            payload[
                "minimum_angstrom"
            ]
        )

        maximum = float(
            payload[
                "maximum_angstrom"
            ]
        )

        spin_max = int(
            payload[
                "spin_max"
            ]
        )

        workers = int(
            payload[
                "workers"
            ]
        )

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return fallback

    if (
        minimum <= 0
        or maximum <= minimum
        or spin_max < 0
        or workers < 1
    ):
        return fallback

    return GuiPreferences(
        minimum_angstrom=minimum,
        maximum_angstrom=maximum,
        spin_max=spin_max,
        workers=workers,
    )


def save_queue_session(
    path: str | Path,
    *,
    jobs: tuple[CalculationJob, ...],
    specs: dict[str, GuiCalculationSpec],
) -> None:
    serialized_jobs = []

    for job in jobs:
        spec = specs.get(
            job.job_id
        )

        if spec is None:
            continue

        serialized_jobs.append(
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "mode": job.mode.value,
                "atom_a": spec.molecule.atom_a,
                "atom_b": spec.molecule.atom_b,
                "minimum_angstrom": spec.minimum_angstrom,
                "maximum_angstrom": spec.maximum_angstrom,
                "spin_max": spec.spin_max,
                "workers": spec.workers,
                "threads_per_worker": spec.threads_per_worker,
                "run_id": spec.run_id,
            }
        )

    _write_json(
        Path(
            path
        ),
        {
            "version": GUI_STATE_VERSION,
            "jobs": serialized_jobs,
        },
    )


def load_queue_session(
    path: str | Path,
) -> LoadedQueueSession:
    source = Path(
        path
    )

    if not source.is_file():
        return LoadedQueueSession(
            jobs=(),
            specs={},
            recovered_running_jobs=0,
        )

    try:
        payload = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "The saved calculation queue could not be read."
        ) from exc

    rows = payload.get(
        "jobs"
    )

    if not isinstance(
        rows,
        list,
    ):
        raise ValueError(
            "The saved calculation queue is invalid."
        )

    jobs = []
    specs = {}
    recovered_running_jobs = 0

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            raise ValueError(
                "The saved calculation queue contains an invalid job."
            )

        molecule = DiatomicMolecule(
            str(
                row[
                    "atom_a"
                ]
            ),
            str(
                row[
                    "atom_b"
                ]
            ),
        )

        job_id = str(
            row[
                "job_id"
            ]
        )

        saved_status = JobStatus(
            str(
                row[
                    "status"
                ]
            )
        )

        if saved_status is JobStatus.RUNNING:
            restored_status = JobStatus.QUEUED
            recovered_running_jobs += 1
        else:
            restored_status = saved_status

        job = CalculationJob(
            molecule=molecule,
            mode=CalculationMode(
                str(
                    row.get(
                        "mode",
                        CalculationMode.SCHEMA_F.value,
                    )
                )
            ),
            job_id=job_id,
            status=restored_status,
        )

        spec = GuiCalculationSpec(
            job_id=job_id,
            molecule=molecule,
            minimum_angstrom=float(
                row[
                    "minimum_angstrom"
                ]
            ),
            maximum_angstrom=float(
                row[
                    "maximum_angstrom"
                ]
            ),
            spin_max=int(
                row[
                    "spin_max"
                ]
            ),
            workers=int(
                row[
                    "workers"
                ]
            ),
            threads_per_worker=int(
                row.get(
                    "threads_per_worker",
                    1,
                )
            ),
            run_id=str(
                row[
                    "run_id"
                ]
            ),
        )

        jobs.append(
            job
        )

        specs[
            job_id
        ] = spec

    return LoadedQueueSession(
        jobs=tuple(
            jobs
        ),
        specs=specs,
        recovered_running_jobs=recovered_running_jobs,
    )
