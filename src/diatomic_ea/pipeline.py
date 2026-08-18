"""End-to-end Schema F calculation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.analysis import (
    FastGridAnalysis,
    analyze_fast_grid,
)
from diatomic_ea.csv_store import RawResultStore
from diatomic_ea.grid import (
    FastGridPlan,
    build_fast_grid_plan_from_electron_counts,
)
from diatomic_ea.manifest import (
    build_reproducibility_manifest,
    write_reproducibility_manifest,
)
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.progress import (
    CalculationStage,
    ProgressReporter,
)
from diatomic_ea.qzvpd import (
    QZVPDPlan,
    build_qzvpd_plan,
)
from diatomic_ea.qzvpd_analysis import (
    QZVPDAnalysis,
    analyze_qzvpd,
)
from diatomic_ea.refinement import (
    QZVPDCandidate,
    select_qzvpd_candidates,
)
from diatomic_ea.result_export import (
    FinalSchemaFRecord,
    final_record_from_estimate,
    write_final_result_csv,
)
from diatomic_ea.run_layout import (
    RunPaths,
    create_run_paths,
)
from diatomic_ea.runner import (
    FastGridRunSummary,
    QZVPDRunSummary,
    execute_fast_grid_resumable,
    execute_qzvpd_resumable,
)
from diatomic_ea.schema_f import SCHEMA_F
from diatomic_ea.schema_f_statistics import (
    SchemaFEstimate,
    estimate_schema_f,
)
from diatomic_ea.single_point import (
    SinglePointResult,
    SinglePointTask,
    determine_electron_count,
    run_pyscf_single_point,
)
from diatomic_ea.states import ChargeState


ElectronCountResolver = Callable[..., int]

SinglePointWorker = Callable[
    [SinglePointTask],
    SinglePointResult,
]


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """User-level inputs for one complete Schema F calculation."""

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
                "maximum_angstrom must be greater than "
                "or equal to minimum_angstrom."
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
class PipelineResult:
    """Products of one complete Schema F run."""

    paths: RunPaths
    neutral_electrons: int
    anion_electrons: int
    fast_plan: FastGridPlan
    fast_run: FastGridRunSummary
    fast_analysis: FastGridAnalysis
    qzvpd_candidates: tuple[
        QZVPDCandidate,
        ...,
    ]
    qzvpd_plan: QZVPDPlan
    qzvpd_run: QZVPDRunSummary
    qzvpd_analysis: QZVPDAnalysis
    estimate: SchemaFEstimate
    final_record: FinalSchemaFRecord


def _resolve_electron_counts(
    request: PipelineRequest,
    resolver: ElectronCountResolver,
) -> tuple[int, int]:
    """Resolve explicit neutral and anionic electron counts."""
    probe_r = (
        request.minimum_angstrom
        + request.maximum_angstrom
    ) / 2.0

    neutral = int(
        resolver(
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
        resolver(
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
            "one greater than the neutral count. "
            f"Received neutral={neutral}, "
            f"anion={anion}."
        )

    return (
        neutral,
        anion,
    )


def run_schema_f_pipeline(
    request: PipelineRequest,
    *,
    output_root: str | Path = "runs",
    reporter: ProgressReporter | None = None,
    electron_count_resolver: ElectronCountResolver = (
        determine_electron_count
    ),
    worker: SinglePointWorker = run_pyscf_single_point,
) -> PipelineResult:
    """Run the complete strict Paper-1 Schema F workflow."""
    paths = create_run_paths(
        output_root=output_root,
        molecule=request.molecule,
        run_id=request.run_id,
    )

    if reporter is not None:
        reporter.job_started(
            message=(
                "Starting complete Schema F calculation "
                f"for {request.molecule.formula}."
            )
        )

    try:
        (
            neutral_electrons,
            anion_electrons,
        ) = _resolve_electron_counts(
            request,
            electron_count_resolver,
        )

        fast_plan = (
            build_fast_grid_plan_from_electron_counts(
                molecule=request.molecule,
                neutral_electrons=neutral_electrons,
                anion_electrons=anion_electrons,
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
            )
        )

        fast_store = RawResultStore(
            paths.fast_grid_csv
        )

        fast_run = execute_fast_grid_resumable(
            fast_plan,
            store=fast_store,
            max_workers=request.workers,
            reporter=reporter,
            retry_errors=True,
            worker=worker,
        )

        if not fast_run.complete:
            raise RuntimeError(
                "Fast grid remains incomplete after execution. "
                f"{fast_run.remaining_after_run} task(s) "
                "still require successful results."
            )

        if reporter is not None:
            reporter.stage_started(
                CalculationStage.FAST_GRID_ANALYSIS,
                message=(
                    "Analyzing fast-grid minima and "
                    "selecting QZVPD candidates."
                ),
            )

        fast_analysis = analyze_fast_grid(
            fast_store,
            fast_plan.bond_grid,
        )

        qzvpd_candidates = (
            select_qzvpd_candidates(
                fast_analysis
            )
        )

        if not qzvpd_candidates:
            raise RuntimeError(
                "Fast-grid analysis produced no reliable "
                "QZVPD refinement candidates."
            )

        if reporter is not None:
            reporter.stage_completed(
                CalculationStage.FAST_GRID_ANALYSIS,
                message=(
                    f"Selected {len(qzvpd_candidates)} "
                    "QZVPD candidate(s)."
                ),
            )

        qzvpd_plan = build_qzvpd_plan(
            molecule=request.molecule,
            candidates=qzvpd_candidates,
            threads_per_worker=(
                request.threads_per_worker
            ),
        )

        if not qzvpd_plan.tasks:
            raise RuntimeError(
                "QZVPD refinement plan contains no tasks."
            )

        qzvpd_store = RawResultStore(
            paths.qzvpd_csv
        )

        qzvpd_run = execute_qzvpd_resumable(
            qzvpd_plan,
            store=qzvpd_store,
            max_workers=request.workers,
            reporter=reporter,
            retry_errors=True,
            worker=worker,
        )

        if not qzvpd_run.complete:
            raise RuntimeError(
                "QZVPD refinement remains incomplete "
                "after execution. "
                f"{qzvpd_run.remaining_after_run} task(s) "
                "still require successful results."
            )

        if reporter is not None:
            reporter.stage_started(
                CalculationStage.STATISTICAL_EA,
                message=(
                    "Reducing QZVPD minima and "
                    "evaluating Schema F."
                ),
            )

        qzvpd_analysis = analyze_qzvpd(
            qzvpd_store,
            qzvpd_plan,
        )

        estimate = estimate_schema_f(
            qzvpd_analysis
        )

        if reporter is not None:
            reporter.stage_completed(
                CalculationStage.STATISTICAL_EA,
                message=(
                    "Schema F statistical estimate "
                    "completed."
                ),
            )

        final_record = (
            final_record_from_estimate(
                estimate
            )
        )

        if reporter is not None:
            reporter.stage_started(
                CalculationStage.EXPORT,
                message=(
                    "Writing final result and "
                    "reproducibility manifest."
                ),
            )

        write_final_result_csv(
            paths.final_result_csv,
            final_record,
        )

        manifest = (
            build_reproducibility_manifest(
                estimate=estimate,
                fast_grid_task_count=(
                    fast_plan.task_count
                ),
                qzvpd_task_count=(
                    qzvpd_plan.task_count
                ),
                fast_grid_raw_csv=(
                    paths.fast_grid_csv
                ),
                qzvpd_raw_csv=(
                    paths.qzvpd_csv
                ),
            )
        )

        write_reproducibility_manifest(
            paths.manifest_json,
            manifest,
        )

        if reporter is not None:
            reporter.stage_completed(
                CalculationStage.EXPORT,
                message=(
                    "Final result files written."
                ),
            )

            reporter.job_completed(
                message=(
                    "Schema F calculation completed."
                )
            )

        return PipelineResult(
            paths=paths,
            neutral_electrons=neutral_electrons,
            anion_electrons=anion_electrons,
            fast_plan=fast_plan,
            fast_run=fast_run,
            fast_analysis=fast_analysis,
            qzvpd_candidates=qzvpd_candidates,
            qzvpd_plan=qzvpd_plan,
            qzvpd_run=qzvpd_run,
            qzvpd_analysis=qzvpd_analysis,
            estimate=estimate,
            final_record=final_record,
        )

    except Exception as exc:
        if reporter is not None:
            reporter.job_failed(
                message=str(exc)
            )

        raise