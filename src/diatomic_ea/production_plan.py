"""Preparation of persistent full Schema F production runs.

This module does not execute the production calculation.

It resolves the real compute environment, electron counts, electronic
state space, exact fast-grid task count, QZVPD upper bound, and resource
requirements before a long calculation is allowed to start.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from diatomic_ea.compute_provenance import (
    collect_compute_provenance,
)
from diatomic_ea.electron_count_adapter import (
    run_platform_electron_count,
)
from diatomic_ea.grid import (
    BondGrid,
    FastGridPlan,
    build_fast_grid_plan_from_electron_counts,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
)
from diatomic_ea.resources import (
    CpuResources,
    detect_cpu_resources,
)
from diatomic_ea.run_layout import (
    RunPaths,
    create_run_paths,
)
from diatomic_ea.schema_f import (
    SCHEMA_F,
)
from diatomic_ea.states import (
    ChargeState,
)


PRODUCTION_PLAN_VERSION = 1

PRODUCTION_PLAN_FILENAME = (
    "00_production_plan.json"
)


@dataclass(frozen=True, slots=True)
class ProductionRunRequest:
    """User-controlled scientific extent of one production run."""

    molecule: DiatomicMolecule
    minimum_angstrom: float
    maximum_angstrom: float
    spin_max: int
    workers: int
    threads_per_worker: int = 1
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.minimum_angstrom <= 0:
            raise ValueError(
                "minimum_angstrom must be positive."
            )

        if (
            self.maximum_angstrom
            < self.minimum_angstrom
        ):
            raise ValueError(
                "maximum_angstrom must be greater "
                "than or equal to minimum_angstrom."
            )

        if self.spin_max < 0:
            raise ValueError(
                "spin_max must not be negative."
            )

        if self.workers < 1:
            raise ValueError(
                "workers must be at least 1."
            )

        if self.threads_per_worker < 1:
            raise ValueError(
                "threads_per_worker must be at least 1."
            )


@dataclass(frozen=True, slots=True)
class ProductionPlan:
    """Resolved immutable plan for one full Schema F run."""

    plan_version: int
    schema_id: str
    molecule: str
    run_id: str
    run_directory: str

    minimum_angstrom: float
    maximum_angstrom: float
    spin_max: int

    neutral_electrons: int
    anion_electrons: int
    neutral_spins: tuple[int, ...]
    anion_spins: tuple[int, ...]

    functionals: tuple[str, ...]
    fast_bases: tuple[str, ...]
    fast_grid_step_angstrom: float
    fast_grid_points: int
    fast_grid_tasks: int

    qzvpd_basis: str
    qzvpd_window_angstrom: float
    qzvpd_step_angstrom: float
    qzvpd_points_per_candidate: int
    qzvpd_candidate_upper_bound: int
    qzvpd_task_upper_bound: int
    total_task_upper_bound: int

    physical_cores: int
    logical_cores: int
    detected_memory_mb: int
    requested_workers: int
    cpu_recommended_workers: int
    memory_recommended_workers: int
    recommended_workers: int
    threads_per_worker: int

    fast_memory_per_worker_mb: int
    qzvpd_memory_per_worker_mb: int
    qzvpd_concurrent_memory_upper_mb: int

    compute_backend: str
    compute_distribution: str | None
    compute_python: str | None
    pyscf_version: str | None
    worker_wheel_sha256: str | None
    provenance_verified: bool

    scientific_execution_started: bool
    resumable: bool


def _verified_provenance() -> dict[str, object]:
    provenance = (
        collect_compute_provenance()
    )

    compatibility = provenance.get(
        "compatibility"
    )

    if not isinstance(
        compatibility,
        dict,
    ):
        raise RuntimeError(
            "Compute provenance contains no "
            "compatibility record."
        )

    if not bool(
        compatibility.get(
            "verified",
            False,
        )
    ):
        raise RuntimeError(
            "Compute provenance is not verified."
        )

    return provenance


def _resolve_electron_counts(
    request: ProductionRunRequest,
) -> tuple[int, int]:
    probe_r = (
        request.minimum_angstrom
        + request.maximum_angstrom
    ) / 2.0

    neutral = int(
        run_platform_electron_count(
            molecule=request.molecule,
            charge=ChargeState.NEUTRAL,
            basis=SCHEMA_F.fast_bases[0],
            bond_length_angstrom=probe_r,
            max_memory_mb=(
                SCHEMA_F
                .fast_grid
                .max_memory_mb
            ),
        )
    )

    anion = int(
        run_platform_electron_count(
            molecule=request.molecule,
            charge=ChargeState.ANION,
            basis=SCHEMA_F.fast_bases[0],
            bond_length_angstrom=probe_r,
            max_memory_mb=(
                SCHEMA_F
                .fast_grid
                .max_memory_mb
            ),
        )
    )

    if neutral < 1:
        raise RuntimeError(
            "Neutral electron count is invalid."
        )

    if anion != neutral + 1:
        raise RuntimeError(
            "Anion electron count must be exactly "
            "one greater than the neutral count."
        )

    return (
        neutral,
        anion,
    )


def _qzvpd_points_per_candidate() -> int:
    center = 1.0

    grid = BondGrid(
        minimum_angstrom=(
            center
            - SCHEMA_F
            .refinement
            .window_angstrom
        ),
        maximum_angstrom=(
            center
            + SCHEMA_F
            .refinement
            .window_angstrom
        ),
        step_angstrom=(
            SCHEMA_F
            .refinement
            .grid
            .step_angstrom
        ),
    )

    return len(
        grid.values
    )


def _memory_mb() -> int:
    return int(
        psutil.virtual_memory().total
        // (
            1024
            * 1024
        )
    )


def _memory_worker_limit(
    *,
    total_memory_mb: int,
) -> int:
    available_budget = int(
        total_memory_mb
        * 0.75
    )

    per_worker = (
        SCHEMA_F
        .refinement
        .grid
        .max_memory_mb
    )

    return max(
        1,
        available_budget
        // per_worker,
    )


def _resource_worker_recommendation(
    *,
    resources: CpuResources,
    total_memory_mb: int,
) -> tuple[int, int]:
    memory_workers = (
        _memory_worker_limit(
            total_memory_mb=(
                total_memory_mb
            )
        )
    )

    recommended = max(
        1,
        min(
            resources.recommended_workers,
            memory_workers,
        ),
    )

    return (
        memory_workers,
        recommended,
    )


def _compute_identity(
    provenance: dict[str, object],
) -> tuple[
    str,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    backend = str(
        provenance.get(
            "backend",
            "unknown",
        )
    )

    compute = provenance.get(
        "compute"
    )

    if not isinstance(
        compute,
        dict,
    ):
        return (
            backend,
            None,
            None,
            None,
            None,
        )

    def optional_text(
        name: str,
    ) -> str | None:
        value = compute.get(
            name
        )

        if value is None:
            return None

        text = str(
            value
        ).strip()

        return (
            text
            if text
            else None
        )

    return (
        backend,
        optional_text(
            "distribution"
        ),
        optional_text(
            "python_version"
        ),
        optional_text(
            "pyscf_version"
        ),
        optional_text(
            "worker_wheel_sha256"
        ),
    )


def build_production_plan(
    request: ProductionRunRequest,
    *,
    output_root: str | Path,
    provenance: dict[str, object],
    neutral_electrons: int,
    anion_electrons: int,
    resources: CpuResources,
    total_memory_mb: int,
) -> tuple[
    ProductionPlan,
    FastGridPlan,
    RunPaths,
]:
    """Build an exact pre-execution plan without running DFT tasks."""
    paths = create_run_paths(
        output_root=output_root,
        molecule=request.molecule,
        run_id=request.run_id,
    )

    fast_plan = (
        build_fast_grid_plan_from_electron_counts(
            molecule=request.molecule,
            neutral_electrons=(
                neutral_electrons
            ),
            anion_electrons=(
                anion_electrons
            ),
            spin_max=request.spin_max,
            minimum_angstrom=(
                request.minimum_angstrom
            ),
            maximum_angstrom=(
                request.maximum_angstrom
            ),
            threads_per_worker=(
                request.threads_per_worker
            ),
            schema=SCHEMA_F,
        )
    )

    neutral_spins = tuple(
        state.spin
        for state
        in fast_plan.state_scan.neutral.states
    )

    anion_spins = tuple(
        state.spin
        for state
        in fast_plan.state_scan.anion.states
    )

    qz_points = (
        _qzvpd_points_per_candidate()
    )

    qz_candidate_upper = (
        len(
            SCHEMA_F.functionals
        )
        * 2
        * SCHEMA_F
        .refinement
        .max_spins_per_charge
    )

    qz_task_upper = (
        qz_candidate_upper
        * qz_points
    )

    (
        memory_workers,
        recommended_workers,
    ) = _resource_worker_recommendation(
        resources=resources,
        total_memory_mb=(
            total_memory_mb
        ),
    )

    (
        backend,
        distribution,
        compute_python,
        pyscf_version,
        wheel_hash,
    ) = _compute_identity(
        provenance
    )

    compatibility = provenance.get(
        "compatibility",
        {},
    )

    verified = bool(
        compatibility.get(
            "verified",
            False,
        )
        if isinstance(
            compatibility,
            dict,
        )
        else False
    )

    plan = ProductionPlan(
        plan_version=(
            PRODUCTION_PLAN_VERSION
        ),
        schema_id=(
            SCHEMA_F.schema_id
        ),
        molecule=request.molecule.formula,
        run_id=paths.run_id,
        run_directory=str(
            paths.run_dir.resolve()
        ),
        minimum_angstrom=(
            request.minimum_angstrom
        ),
        maximum_angstrom=(
            request.maximum_angstrom
        ),
        spin_max=request.spin_max,
        neutral_electrons=(
            neutral_electrons
        ),
        anion_electrons=(
            anion_electrons
        ),
        neutral_spins=(
            neutral_spins
        ),
        anion_spins=(
            anion_spins
        ),
        functionals=(
            SCHEMA_F.functionals
        ),
        fast_bases=(
            SCHEMA_F.fast_bases
        ),
        fast_grid_step_angstrom=(
            SCHEMA_F
            .fast_grid
            .step_angstrom
        ),
        fast_grid_points=(
            fast_plan
            .bond_point_count
        ),
        fast_grid_tasks=(
            fast_plan.task_count
        ),
        qzvpd_basis=(
            SCHEMA_F
            .refinement
            .basis
        ),
        qzvpd_window_angstrom=(
            SCHEMA_F
            .refinement
            .window_angstrom
        ),
        qzvpd_step_angstrom=(
            SCHEMA_F
            .refinement
            .grid
            .step_angstrom
        ),
        qzvpd_points_per_candidate=(
            qz_points
        ),
        qzvpd_candidate_upper_bound=(
            qz_candidate_upper
        ),
        qzvpd_task_upper_bound=(
            qz_task_upper
        ),
        total_task_upper_bound=(
            fast_plan.task_count
            + qz_task_upper
        ),
        physical_cores=(
            resources.physical_cores
        ),
        logical_cores=(
            resources.logical_cores
        ),
        detected_memory_mb=(
            total_memory_mb
        ),
        requested_workers=(
            request.workers
        ),
        cpu_recommended_workers=(
            resources
            .recommended_workers
        ),
        memory_recommended_workers=(
            memory_workers
        ),
        recommended_workers=(
            recommended_workers
        ),
        threads_per_worker=(
            request.threads_per_worker
        ),
        fast_memory_per_worker_mb=(
            SCHEMA_F
            .fast_grid
            .max_memory_mb
        ),
        qzvpd_memory_per_worker_mb=(
            SCHEMA_F
            .refinement
            .grid
            .max_memory_mb
        ),
        qzvpd_concurrent_memory_upper_mb=(
            request.workers
            * SCHEMA_F
            .refinement
            .grid
            .max_memory_mb
        ),
        compute_backend=backend,
        compute_distribution=(
            distribution
        ),
        compute_python=(
            compute_python
        ),
        pyscf_version=(
            pyscf_version
        ),
        worker_wheel_sha256=(
            wheel_hash
        ),
        provenance_verified=(
            verified
        ),
        scientific_execution_started=False,
        resumable=True,
    )

    return (
        plan,
        fast_plan,
        paths,
    )


def _plan_path(
    paths: RunPaths,
) -> Path:
    return (
        paths.run_dir
        / PRODUCTION_PLAN_FILENAME
    )


def _comparable_plan(
    payload: dict[str, object],
) -> dict[str, object]:
    ignored = {
        "run_directory",
    }

    return {
        key: value
        for key, value
        in payload.items()
        if key not in ignored
    }


def write_production_plan(
    plan: ProductionPlan,
    paths: RunPaths,
) -> Path:
    """Persist plan and reject incompatible reuse of the run id."""
    path = _plan_path(
        paths
    )

    payload = json.loads(
        json.dumps(
            asdict(
                plan
            ),
            sort_keys=True,
        )
    )

    if path.exists():
        try:
            existing = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Existing production plan "
                "is not valid JSON."
            ) from exc

        if not isinstance(
            existing,
            dict,
        ):
            raise RuntimeError(
                "Existing production plan "
                "is not a JSON object."
            )

        if (
            _comparable_plan(
                existing
            )
            != _comparable_plan(
                payload
            )
        ):
            raise RuntimeError(
                "Existing run_id belongs to a "
                "different production plan. "
                "Choose another run_id."
            )

        return path

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


def prepare_production_run(
    request: ProductionRunRequest,
    *,
    output_root: str | Path = (
        "production_runs"
    ),
) -> tuple[
    ProductionPlan,
    Path,
]:
    """Resolve and persist a production run without executing it."""
    provenance = (
        _verified_provenance()
    )

    (
        neutral,
        anion,
    ) = _resolve_electron_counts(
        request
    )

    resources = (
        detect_cpu_resources()
    )

    memory_mb = (
        _memory_mb()
    )

    (
        plan,
        _fast_plan,
        paths,
    ) = build_production_plan(
        request,
        output_root=output_root,
        provenance=provenance,
        neutral_electrons=neutral,
        anion_electrons=anion,
        resources=resources,
        total_memory_mb=memory_mb,
    )

    plan_path = (
        write_production_plan(
            plan,
            paths,
        )
    )

    return (
        plan,
        plan_path,
    )


def _hours_from_seconds(
    seconds: float,
) -> float:
    return seconds / 3600.0


def estimate_wall_time_hours(
    *,
    plan: ProductionPlan,
    fast_seconds_per_task: float,
    qzvpd_seconds_per_task: float,
) -> tuple[
    float,
    float,
    float,
]:
    """Return naive fast, QZVPD-upper, and total wall-time estimates."""
    if fast_seconds_per_task <= 0:
        raise ValueError(
            "fast_seconds_per_task must be positive."
        )

    if qzvpd_seconds_per_task <= 0:
        raise ValueError(
            "qzvpd_seconds_per_task must be positive."
        )

    parallelism = max(
        1,
        plan.requested_workers,
    )

    fast_seconds = (
        plan.fast_grid_tasks
        * fast_seconds_per_task
        / parallelism
    )

    qz_seconds = (
        plan.qzvpd_task_upper_bound
        * qzvpd_seconds_per_task
        / parallelism
    )

    return (
        _hours_from_seconds(
            fast_seconds
        ),
        _hours_from_seconds(
            qz_seconds
        ),
        _hours_from_seconds(
            fast_seconds
            + qz_seconds
        ),
    )


def _print_plan(
    plan: ProductionPlan,
    *,
    plan_path: Path,
) -> None:
    print()
    print(
        "DiatomicEA full Schema F production plan"
    )

    print(
        "======================================="
    )

    print()

    print(
        "SCIENTIFIC EXECUTION HAS NOT STARTED"
    )

    print()

    print(
        "Schema:",
        plan.schema_id,
    )

    print(
        "Molecule:",
        plan.molecule,
    )

    print(
        "Run ID:",
        plan.run_id,
    )

    print(
        "Run directory:",
        plan.run_directory,
    )

    print()

    print(
        "Bond range / Angstrom:",
        plan.minimum_angstrom,
        "to",
        plan.maximum_angstrom,
    )

    print(
        "Spin max:",
        plan.spin_max,
    )

    print(
        "Neutral electrons:",
        plan.neutral_electrons,
    )

    print(
        "Neutral spins:",
        plan.neutral_spins,
    )

    print(
        "Anion electrons:",
        plan.anion_electrons,
    )

    print(
        "Anion spins:",
        plan.anion_spins,
    )

    print()

    print(
        "Functionals:",
        ", ".join(
            plan.functionals
        ),
    )

    print(
        "Fast bases:",
        ", ".join(
            plan.fast_bases
        ),
    )

    print(
        "Fast-grid points:",
        plan.fast_grid_points,
    )

    print(
        "Exact fast-grid tasks:",
        plan.fast_grid_tasks,
    )

    print()

    print(
        "QZVPD points / candidate:",
        plan.qzvpd_points_per_candidate,
    )

    print(
        "QZVPD candidate upper bound:",
        plan.qzvpd_candidate_upper_bound,
    )

    print(
        "QZVPD task upper bound:",
        plan.qzvpd_task_upper_bound,
    )

    print(
        "Total task upper bound:",
        plan.total_task_upper_bound,
    )

    print()

    print(
        "Physical cores:",
        plan.physical_cores,
    )

    print(
        "Logical cores:",
        plan.logical_cores,
    )

    print(
        "Detected RAM / MiB:",
        plan.detected_memory_mb,
    )

    print(
        "Requested workers:",
        plan.requested_workers,
    )

    print(
        "CPU recommendation:",
        plan.cpu_recommended_workers,
    )

    print(
        "Memory recommendation:",
        plan.memory_recommended_workers,
    )

    print(
        "Combined recommendation:",
        plan.recommended_workers,
    )

    print(
        "QZVPD concurrent memory upper / MiB:",
        plan.qzvpd_concurrent_memory_upper_mb,
    )

    if (
        plan.requested_workers
        > plan.recommended_workers
    ):
        print()
        print(
            "WARNING: requested worker count exceeds "
            "the conservative resource recommendation."
        )

    print()

    print(
        "Backend:",
        plan.compute_backend,
    )

    print(
        "Distribution:",
        plan.compute_distribution
        or "n/a",
    )

    print(
        "Compute Python:",
        plan.compute_python
        or "n/a",
    )

    print(
        "PySCF:",
        plan.pyscf_version
        or "n/a",
    )

    print(
        "Worker wheel SHA-256:",
        plan.worker_wheel_sha256
        or "n/a",
    )

    print(
        "Provenance verified:",
        plan.provenance_verified,
    )

    print()

    print(
        "Plan file:",
        plan_path,
    )

    print(
        "Resumable:",
        plan.resumable,
    )

    print()

    print(
        "STATUS: READY FOR EXPLICIT PRODUCTION START"
    )


def main() -> int:
    """Prepare one persistent full Schema F production run."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a full Schema F production "
            "run without starting DFT execution."
        )
    )

    parser.add_argument(
        "atom_a",
    )

    parser.add_argument(
        "atom_b",
    )

    parser.add_argument(
        "--minimum",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--maximum",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--spin-max",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        default="production_runs",
    )

    args = parser.parse_args()

    request = ProductionRunRequest(
        molecule=DiatomicMolecule(
            args.atom_a,
            args.atom_b,
        ),
        minimum_angstrom=args.minimum,
        maximum_angstrom=args.maximum,
        spin_max=args.spin_max,
        workers=args.workers,
        threads_per_worker=(
            args.threads_per_worker
        ),
        run_id=args.run_id,
    )

    try:
        (
            plan,
            path,
        ) = prepare_production_run(
            request,
            output_root=args.output_root,
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

    _print_plan(
        plan,
        plan_path=path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
