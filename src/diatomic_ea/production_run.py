"""Safe launcher for a prepared full Schema F production run.

The launcher does not define scientific settings itself.

It loads an existing immutable production plan, verifies that the
currently installed compute worker still matches that plan, prevents
two launchers from writing to the same run simultaneously, and then
executes the existing resumable Schema F pipeline.

Raw single-point results remain the authoritative resume checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from diatomic_ea.compute_provenance import (
    collect_compute_provenance,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.pipeline import (
    PipelineRequest,
    PipelineResult,
    run_schema_f_pipeline,
)
from diatomic_ea.production_plan import (
    PRODUCTION_PLAN_FILENAME,
    PRODUCTION_PLAN_VERSION,
)
from diatomic_ea.progress import (
    ProgressEvent,
    ProgressEventType,
    ProgressReporter,
)
from diatomic_ea.schema_f import (
    SCHEMA_F,
)


PRODUCTION_STATUS_VERSION = 1

PRODUCTION_STATUS_FILENAME = (
    "production_status.json"
)

PRODUCTION_EVENTS_FILENAME = (
    "production_events.jsonl"
)

PRODUCTION_LOCK_FILENAME = (
    "production.lock"
)


class ProductionLaunchError(RuntimeError):
    """Raised when a prepared production run cannot safely start."""


@dataclass(frozen=True, slots=True)
class ValidatedProductionRun:
    """Prepared production run after compatibility validation."""

    plan_path: Path
    run_directory: Path
    output_root: Path
    molecule: DiatomicMolecule
    request: PipelineRequest
    plan: dict[str, Any]
    provenance: dict[str, Any]
    status_path: Path
    events_path: Path
    lock_path: Path
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProductionRunOutcome:
    """Outcome returned by the production launcher."""

    started: bool
    completed: bool
    run_id: str
    run_directory: str
    status_path: str
    final_result_csv: str | None
    manifest_json: str | None
    message: str


def _utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _load_json_object(
    path: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError as exc:
        raise ProductionLaunchError(
            f"Production plan does not exist: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProductionLaunchError(
            "Production plan is not valid JSON."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ProductionLaunchError(
            "Production plan must contain a JSON object."
        )

    return payload


def _required_int(
    payload: dict[str, Any],
    name: str,
) -> int:
    try:
        value = int(
            payload[
                name
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProductionLaunchError(
            f"Production plan has invalid {name!r}."
        ) from exc

    return value


def _required_float(
    payload: dict[str, Any],
    name: str,
) -> float:
    try:
        value = float(
            payload[
                name
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProductionLaunchError(
            f"Production plan has invalid {name!r}."
        ) from exc

    return value


def _required_text(
    payload: dict[str, Any],
    name: str,
) -> str:
    try:
        value = str(
            payload[
                name
            ]
        ).strip()
    except KeyError as exc:
        raise ProductionLaunchError(
            f"Production plan is missing {name!r}."
        ) from exc

    if not value:
        raise ProductionLaunchError(
            f"Production plan has empty {name!r}."
        )

    return value


def _provenance_compute(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    compute = provenance.get(
        "compute"
    )

    if not isinstance(
        compute,
        dict,
    ):
        raise ProductionLaunchError(
            "Current compute provenance contains "
            "no compute identity."
        )

    return compute


def _verify_current_provenance(
    plan: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    compatibility = provenance.get(
        "compatibility"
    )

    if not isinstance(
        compatibility,
        dict,
    ):
        raise ProductionLaunchError(
            "Current compute provenance contains "
            "no compatibility record."
        )

    if not bool(
        compatibility.get(
            "verified",
            False,
        )
    ):
        raise ProductionLaunchError(
            "Current compute provenance is not verified."
        )

    if not bool(
        plan.get(
            "provenance_verified",
            False,
        )
    ):
        raise ProductionLaunchError(
            "Prepared production plan was not "
            "created with verified provenance."
        )

    compute = _provenance_compute(
        provenance
    )

    current_backend = str(
        provenance.get(
            "backend",
            "",
        )
    )

    planned_backend = str(
        plan.get(
            "compute_backend",
            "",
        )
    )

    if current_backend != planned_backend:
        raise ProductionLaunchError(
            "Compute backend changed since planning: "
            f"planned {planned_backend!r}, "
            f"current {current_backend!r}."
        )

    comparisons = (
        (
            "distribution",
            plan.get(
                "compute_distribution"
            ),
            compute.get(
                "distribution"
            ),
        ),
        (
            "PySCF version",
            plan.get(
                "pyscf_version"
            ),
            compute.get(
                "pyscf_version"
            ),
        ),
        (
            "worker wheel SHA-256",
            plan.get(
                "worker_wheel_sha256"
            ),
            compute.get(
                "worker_wheel_sha256"
            ),
        ),
    )

    for (
        label,
        planned,
        current,
    ) in comparisons:
        if planned != current:
            raise ProductionLaunchError(
                f"{label} changed since planning: "
                f"planned {planned!r}, "
                f"current {current!r}."
            )


def validate_prepared_production(
    plan_path: str | Path,
    *,
    atom_a: str,
    atom_b: str,
    provenance: dict[str, Any] | None = None,
) -> ValidatedProductionRun:
    """Load and validate one immutable production plan."""
    resolved_plan = Path(
        plan_path
    ).resolve()

    if (
        resolved_plan.name
        != PRODUCTION_PLAN_FILENAME
    ):
        raise ProductionLaunchError(
            "Expected production plan filename "
            f"{PRODUCTION_PLAN_FILENAME!r}."
        )

    plan = _load_json_object(
        resolved_plan
    )

    if (
        _required_int(
            plan,
            "plan_version",
        )
        != PRODUCTION_PLAN_VERSION
    ):
        raise ProductionLaunchError(
            "Unsupported production plan version."
        )

    schema_id = _required_text(
        plan,
        "schema_id",
    )

    if schema_id != SCHEMA_F.schema_id:
        raise ProductionLaunchError(
            "Production plan does not use the "
            "current frozen Schema F preset."
        )

    molecule = DiatomicMolecule(
        atom_a,
        atom_b,
    )

    planned_formula = _required_text(
        plan,
        "molecule",
    )

    if molecule.formula != planned_formula:
        raise ProductionLaunchError(
            "Supplied atoms do not match the "
            "planned molecule: "
            f"{molecule.formula!r} != "
            f"{planned_formula!r}."
        )

    run_directory = Path(
        _required_text(
            plan,
            "run_directory",
        )
    ).resolve()

    if (
        resolved_plan.parent
        != run_directory
    ):
        raise ProductionLaunchError(
            "Production plan is not located in "
            "its recorded run directory."
        )

    run_id = _required_text(
        plan,
        "run_id",
    )

    if (
        run_directory.name
        != run_id
    ):
        raise ProductionLaunchError(
            "Run directory name does not match run_id."
        )

    if (
        run_directory.parent.name
        != molecule.formula
    ):
        raise ProductionLaunchError(
            "Molecule directory does not match "
            "the planned molecule."
        )

    current_provenance = (
        collect_compute_provenance()
        if provenance is None
        else provenance
    )

    _verify_current_provenance(
        plan,
        current_provenance,
    )

    requested_workers = _required_int(
        plan,
        "requested_workers",
    )

    recommended_workers = _required_int(
        plan,
        "recommended_workers",
    )

    if requested_workers < 1:
        raise ProductionLaunchError(
            "Planned worker count must be at least 1."
        )

    if recommended_workers < 1:
        raise ProductionLaunchError(
            "Recommended worker count must be at least 1."
        )

    warnings: list[str] = []

    if (
        requested_workers
        > recommended_workers
    ):
        warnings.append(
            "Planned worker count exceeds the "
            "conservative resource recommendation."
        )

    minimum = _required_float(
        plan,
        "minimum_angstrom",
    )

    maximum = _required_float(
        plan,
        "maximum_angstrom",
    )

    spin_max = _required_int(
        plan,
        "spin_max",
    )

    threads = _required_int(
        plan,
        "threads_per_worker",
    )

    request = PipelineRequest(
        molecule=molecule,
        minimum_angstrom=minimum,
        maximum_angstrom=maximum,
        spin_max=spin_max,
        workers=requested_workers,
        threads_per_worker=threads,
        run_id=run_id,
    )

    output_root = (
        run_directory
        .parent
        .parent
    )

    logs_directory = (
        run_directory
        / "logs"
    )

    logs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return ValidatedProductionRun(
        plan_path=resolved_plan,
        run_directory=run_directory,
        output_root=output_root,
        molecule=molecule,
        request=request,
        plan=plan,
        provenance=current_provenance,
        status_path=(
            logs_directory
            / PRODUCTION_STATUS_FILENAME
        ),
        events_path=(
            logs_directory
            / PRODUCTION_EVENTS_FILENAME
        ),
        lock_path=(
            logs_directory
            / PRODUCTION_LOCK_FILENAME
        ),
        warnings=tuple(
            warnings
        ),
    )


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def _append_event(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
            )
        )

        handle.write(
            "\n"
        )

        handle.flush()


def _read_previous_state(
    path: Path,
) -> str | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    value = payload.get(
        "state"
    )

    if value is None:
        return None

    return str(
        value
    )


def _base_status(
    validated: ValidatedProductionRun,
    *,
    state: str,
) -> dict[str, Any]:
    now = _utc_now()

    return {
        "status_version": (
            PRODUCTION_STATUS_VERSION
        ),
        "run_id": (
            validated.request.run_id
        ),
        "molecule": (
            validated.molecule.formula
        ),
        "state": state,
        "stage": None,
        "completed": None,
        "total": None,
        "percent": None,
        "message": "",
        "updated_at_utc": now,
        "worker_wheel_sha256": (
            validated.plan.get(
                "worker_wheel_sha256"
            )
        ),
        "requested_workers": (
            validated.request.workers
        ),
        "threads_per_worker": (
            validated.request
            .threads_per_worker
        ),
        "resume_supported": True,
    }


class _ConsoleProgress:
    """Persist every event while printing throttled task progress."""

    def __init__(
        self,
        validated: ValidatedProductionRun,
        status: dict[str, Any],
    ) -> None:
        self.validated = validated
        self.status = status
        self._last_bucket: dict[
            str,
            int,
        ] = {}

    def _write_status(
        self,
    ) -> None:
        self.status[
            "updated_at_utc"
        ] = _utc_now()

        _atomic_write_json(
            self.validated.status_path,
            self.status,
        )

    def __call__(
        self,
        event: ProgressEvent,
    ) -> None:
        event_payload = {
            "timestamp_utc": (
                _utc_now()
            ),
            "job_id": event.job_id,
            "event_type": (
                event.event_type.value
            ),
            "stage": (
                None
                if event.stage is None
                else event.stage.value
            ),
            "completed": (
                event.completed
            ),
            "total": event.total,
            "percent": (
                event.percent
            ),
            "message": (
                event.message
            ),
        }

        _append_event(
            self.validated.events_path,
            event_payload,
        )

        self.status[
            "stage"
        ] = event_payload[
            "stage"
        ]

        self.status[
            "message"
        ] = event.message

        self.status[
            "completed"
        ] = event.completed

        self.status[
            "total"
        ] = event.total

        self.status[
            "percent"
        ] = event.percent

        should_print = (
            event.event_type
            is not ProgressEventType.ADVANCE
        )

        should_write_status = (
            should_print
        )

        if (
            event.event_type
            is ProgressEventType.ADVANCE
            and event.stage is not None
            and event.percent is not None
        ):
            stage_name = (
                event.stage.value
            )

            bucket = int(
                event.percent
                // 5
            )

            previous = (
                self._last_bucket.get(
                    stage_name
                )
            )

            if (
                previous != bucket
                or event.completed
                == event.total
            ):
                self._last_bucket[
                    stage_name
                ] = bucket

                should_print = True
                should_write_status = True

        if should_write_status:
            self._write_status()

        if should_print:
            stage_text = (
                ""
                if event.stage is None
                else (
                    "["
                    + event.stage.value
                    + "] "
                )
            )

            if (
                event.completed is not None
                and event.total is not None
                and event.percent is not None
            ):
                print(
                    stage_text
                    + f"{event.completed}/"
                    + f"{event.total} "
                    + f"({event.percent:.1f}%)"
                )

            elif event.message:
                print(
                    stage_text
                    + event.message
                )


@contextmanager
def _production_lock(
    validated: ValidatedProductionRun,
    *,
    recover_stale_lock: bool,
) -> Iterator[None]:
    path = validated.lock_path

    if (
        recover_stale_lock
        and path.exists()
    ):
        path.unlink()

    try:
        descriptor = os.open(
            path,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise ProductionLaunchError(
            "Production run is already locked. "
            f"Lock file: {path}. "
            "If the previous process is definitely "
            "not running, use --recover-stale-lock."
        ) from exc

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_at_utc": (
                            _utc_now()
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        yield

    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _completed_files_exist(
    validated: ValidatedProductionRun,
) -> bool:
    final_csv = (
        validated.run_directory
        / "04_final"
        / "final_result.csv"
    )

    manifest = (
        validated.run_directory
        / "04_final"
        / "manifest.json"
    )

    return (
        final_csv.is_file()
        and manifest.is_file()
    )


def run_prepared_production(
    validated: ValidatedProductionRun,
    *,
    recover_stale_lock: bool = False,
) -> ProductionRunOutcome:
    """Run or resume the prepared production calculation."""
    previous_state = (
        _read_previous_state(
            validated.status_path
        )
    )

    if (
        previous_state
        == "completed"
        and _completed_files_exist(
            validated
        )
    ):
        return ProductionRunOutcome(
            started=False,
            completed=True,
            run_id=(
                validated.request.run_id
                or ""
            ),
            run_directory=str(
                validated.run_directory
            ),
            status_path=str(
                validated.status_path
            ),
            final_result_csv=str(
                validated.run_directory
                / "04_final"
                / "final_result.csv"
            ),
            manifest_json=str(
                validated.run_directory
                / "04_final"
                / "manifest.json"
            ),
            message=(
                "Production run is already complete."
            ),
        )

    with _production_lock(
        validated,
        recover_stale_lock=(
            recover_stale_lock
        ),
    ):
        status = _base_status(
            validated,
            state="running",
        )

        status[
            "started_at_utc"
        ] = _utc_now()

        status[
            "resumed_from_state"
        ] = previous_state

        _atomic_write_json(
            validated.status_path,
            status,
        )

        callback = _ConsoleProgress(
            validated,
            status,
        )

        reporter = ProgressReporter(
            job_id=(
                validated.request.run_id
                or validated.molecule.formula
            ),
            callback=callback,
        )

        try:
            result: PipelineResult = (
                run_schema_f_pipeline(
                    validated.request,
                    output_root=(
                        validated.output_root
                    ),
                    reporter=reporter,
                    compute_provenance=(
                        validated.provenance
                    ),
                )
            )

        except KeyboardInterrupt:
            status[
                "state"
            ] = "interrupted"

            status[
                "message"
            ] = (
                "Production run interrupted. "
                "Persisted raw results can be resumed."
            )

            status[
                "interrupted_at_utc"
            ] = _utc_now()

            _atomic_write_json(
                validated.status_path,
                status,
            )

            raise

        except Exception as exc:
            status[
                "state"
            ] = "failed"

            status[
                "message"
            ] = str(
                exc
            )

            status[
                "failed_at_utc"
            ] = _utc_now()

            _atomic_write_json(
                validated.status_path,
                status,
            )

            raise

        status[
            "state"
        ] = "completed"

        status[
            "stage"
        ] = "export"

        status[
            "completed"
        ] = None

        status[
            "total"
        ] = None

        status[
            "percent"
        ] = 100.0

        status[
            "message"
        ] = (
            "Full Schema F production run completed."
        )

        status[
            "completed_at_utc"
        ] = _utc_now()

        status[
            "final_result_csv"
        ] = str(
            result.paths.final_result_csv
            .resolve()
        )

        status[
            "manifest_json"
        ] = str(
            result.paths.manifest_json
            .resolve()
        )

        _atomic_write_json(
            validated.status_path,
            status,
        )

        return ProductionRunOutcome(
            started=True,
            completed=True,
            run_id=(
                validated.request.run_id
                or ""
            ),
            run_directory=str(
                validated.run_directory
            ),
            status_path=str(
                validated.status_path
            ),
            final_result_csv=str(
                result.paths
                .final_result_csv
                .resolve()
            ),
            manifest_json=str(
                result.paths
                .manifest_json
                .resolve()
            ),
            message=(
                "Full Schema F production "
                "run completed."
            ),
        )


def _print_validation(
    validated: ValidatedProductionRun,
) -> None:
    plan = validated.plan

    print()
    print(
        "DiatomicEA production launch validation"
    )

    print(
        "======================================"
    )

    print()

    print(
        "Schema:",
        plan[
            "schema_id"
        ],
    )

    print(
        "Molecule:",
        validated.molecule.formula,
    )

    print(
        "Run ID:",
        validated.request.run_id,
    )

    print(
        "Run directory:",
        validated.run_directory,
    )

    print(
        "Fast-grid tasks:",
        plan[
            "fast_grid_tasks"
        ],
    )

    print(
        "QZVPD task upper bound:",
        plan[
            "qzvpd_task_upper_bound"
        ],
    )

    print(
        "Workers:",
        validated.request.workers,
    )

    print(
        "Worker wheel SHA-256:",
        plan.get(
            "worker_wheel_sha256"
        ),
    )

    print(
        "PySCF:",
        plan.get(
            "pyscf_version"
        ),
    )

    print(
        "Compute identity:",
        "MATCH",
    )

    for warning in (
        validated.warnings
    ):
        print(
            "WARNING:",
            warning,
        )


def main() -> int:
    """Validate, start, or resume one prepared production run."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute one prepared "
            "full Schema F production run."
        )
    )

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

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--check-only",
        action="store_true",
    )

    mode.add_argument(
        "--start",
        action="store_true",
    )

    parser.add_argument(
        "--recover-stale-lock",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        validated = (
            validate_prepared_production(
                args.plan,
                atom_a=args.atom_a,
                atom_b=args.atom_b,
            )
        )

    except Exception as exc:
        print()
        print(
            "STATUS: FAIL"
        )

        print(
            "Error:",
            str(
                exc
            ),
        )

        return 1

    _print_validation(
        validated
    )

    if args.check_only:
        print()
        print(
            "SCIENTIFIC EXECUTION HAS NOT STARTED"
        )

        print(
            "STATUS: READY"
        )

        return 0

    print()
    print(
        "======================================="
    )

    print(
        " STARTING FULL SCHEMA F PRODUCTION RUN"
    )

    print(
        "======================================="
    )

    print()

    print(
        "Raw results are persisted incrementally."
    )

    print(
        "Re-running this same command resumes "
        "unfinished tasks."
    )

    print()

    try:
        outcome = run_prepared_production(
            validated,
            recover_stale_lock=(
                args.recover_stale_lock
            ),
        )

    except KeyboardInterrupt:
        print()
        print(
            "STATUS: INTERRUPTED"
        )

        print(
            "Run can be resumed with the same command."
        )

        return 130

    except Exception as exc:
        print()
        print(
            "STATUS: FAIL"
        )

        print(
            "Error:",
            str(
                exc
            ),
        )

        print(
            "Persisted successful raw results "
            "remain available for resume."
        )

        return 1

    print()
    print(
        "STATUS: COMPLETE"
    )

    print(
        "Message:",
        outcome.message,
    )

    print(
        "Final result:",
        outcome.final_result_csv
        or "n/a",
    )

    print(
        "Manifest:",
        outcome.manifest_json
        or "n/a",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
