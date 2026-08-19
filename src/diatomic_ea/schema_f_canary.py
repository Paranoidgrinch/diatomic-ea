"""Real reduced-cost end-to-end workflow canary.

This module exercises the same scientific reduction stages used by
Schema F, but deliberately uses reduced numerical settings.

It is therefore NEVER a scientific Schema F electron-affinity result.

The canary validates:

    platform electron-count resolution
    -> fast-grid generation
    -> real multiprocessing calculations
    -> raw CSV persistence
    -> fast-grid analysis
    -> QZVPD candidate selection
    -> real QZVPD refinement
    -> QZVPD analysis
    -> four-functional statistical reduction
    -> reproducibility provenance
    -> resumability

The final numerical estimate is diagnostic only.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.analysis import (
    analyze_fast_grid,
)
from diatomic_ea.compute_provenance import (
    collect_compute_provenance,
)
from diatomic_ea.csv_store import (
    RawResultStore,
)
from diatomic_ea.electron_count_adapter import (
    run_platform_electron_count,
)
from diatomic_ea.grid import (
    FastGridPlan,
    build_fast_grid_plan_from_electron_counts,
)
from diatomic_ea.molecule import (
    DiatomicMolecule,
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
    select_qzvpd_candidates,
)
from diatomic_ea.runner import (
    execute_fast_grid_resumable,
    execute_qzvpd_resumable,
)
from diatomic_ea.schema_f import (
    SCHEMA_F,
    GridSpec,
    RefinementSpec,
    SCFRescueSpec,
    SchemaFSpec,
)
from diatomic_ea.schema_f_statistics import (
    SchemaFEstimate,
    estimate_schema_f,
)
from diatomic_ea.single_point_adapter import (
    run_platform_single_point,
)
from diatomic_ea.states import (
    ChargeState,
)


CANARY_ID = (
    "schema-f-workflow-canary-v1"
)

CANARY_WARNING = (
    "WORKFLOW CANARY ONLY - "
    "NOT A SCIENTIFIC SCHEMA F EA PREDICTION"
)

CANARY_MOLECULE = (
    DiatomicMolecule(
        "O",
        "H",
    )
)

CANARY_SCHEMA = SchemaFSpec(
    schema_id=CANARY_ID,
    reference_pyscf_version=(
        SCHEMA_F.reference_pyscf_version
    ),
    electronic_structure_method=(
        SCHEMA_F.electronic_structure_method
    ),
    functionals=SCHEMA_F.functionals,
    fast_bases=(
        "def2-svp",
    ),
    fast_grid=GridSpec(
        step_angstrom=0.10,
        grid_level=1,
        conv_tol=1.0e-7,
        max_cycle=120,
        max_memory_mb=1500,
    ),
    refinement=RefinementSpec(
        basis="def2-qzvpd",
        window_angstrom=0.08,
        grid=GridSpec(
            step_angstrom=0.04,
            grid_level=1,
            conv_tol=1.0e-7,
            max_cycle=160,
            max_memory_mb=2000,
        ),
        max_spins_per_charge=1,
    ),
    scf_rescue=SCFRescueSpec(
        level_shift=(
            SCHEMA_F
            .scf_rescue
            .level_shift
        ),
        newton_max_cycle=(
            SCHEMA_F
            .scf_rescue
            .newton_max_cycle
        ),
    ),
    all_electron_through_atomic_number=(
        SCHEMA_F
        .all_electron_through_atomic_number
    ),
)

CANARY_MINIMUM_ANGSTROM = 0.80

CANARY_MAXIMUM_ANGSTROM = 1.10

CANARY_SPIN_MAX = 1


@dataclass(frozen=True, slots=True)
class CanaryResult:
    """Products of one real reduced-cost workflow canary."""

    passed: bool
    output_directory: str
    fast_grid_csv: str
    qzvpd_csv: str
    report_json: str
    neutral_electrons: int
    anion_electrons: int
    fast_task_count: int
    candidate_count: int
    qzvpd_task_count: int
    fast_resume_attempted: int
    qzvpd_resume_attempted: int
    estimate: SchemaFEstimate
    message: str


def build_canary_fast_plan(
    *,
    neutral_electrons: int,
    anion_electrons: int,
) -> FastGridPlan:
    """Build the intentionally reduced real fast-grid plan."""
    return (
        build_fast_grid_plan_from_electron_counts(
            molecule=CANARY_MOLECULE,
            neutral_electrons=(
                neutral_electrons
            ),
            anion_electrons=(
                anion_electrons
            ),
            spin_max=CANARY_SPIN_MAX,
            minimum_angstrom=(
                CANARY_MINIMUM_ANGSTROM
            ),
            maximum_angstrom=(
                CANARY_MAXIMUM_ANGSTROM
            ),
            threads_per_worker=1,
            schema=CANARY_SCHEMA,
        )
    )


def _estimate_payload(
    estimate: SchemaFEstimate,
) -> dict[str, object]:
    return {
        "model_id": estimate.model_id,
        "functional_eas_eV": {
            functional: value
            for functional, value
            in estimate.functional_eas_ev
        },
        "median_qz_eV": (
            estimate.median_qz_ev
        ),
        "half_range_qz_eV": (
            estimate.half_range_qz_ev
        ),
        "diagnostic_predicted_ea_eV": (
            estimate.predicted_ea_ev
        ),
        "diagnostic_scale_eV": (
            estimate.scale_ev
        ),
        "prediction_intervals_eV": {
            str(
                interval.confidence_percent
            ): {
                "lower": (
                    interval.lower_ev
                ),
                "upper": (
                    interval.upper_ev
                ),
                "half_width": (
                    interval.half_width_ev
                ),
            }
            for interval
            in estimate.intervals
        },
    }


def build_canary_report_payload(
    *,
    neutral_electrons: int,
    anion_electrons: int,
    fast_plan: FastGridPlan,
    qzvpd_plan: QZVPDPlan,
    qzvpd_analysis: QZVPDAnalysis,
    estimate: SchemaFEstimate,
    provenance: dict[str, object],
    fast_grid_csv: str | Path,
    qzvpd_csv: str | Path,
) -> dict[str, object]:
    """Build a report that cannot be confused with a real result."""
    return {
        "report_type": (
            "diatomic_ea_workflow_canary"
        ),
        "canary_id": CANARY_ID,
        "scientific_result": False,
        "schema_f_result": False,
        "warning": CANARY_WARNING,
        "molecule": (
            CANARY_MOLECULE.formula
        ),
        "purpose": (
            "End-to-end software and compute "
            "backend validation."
        ),
        "reduced_settings": {
            "functionals": list(
                CANARY_SCHEMA.functionals
            ),
            "fast_bases": list(
                CANARY_SCHEMA.fast_bases
            ),
            "fast_grid": {
                "minimum_angstrom": (
                    CANARY_MINIMUM_ANGSTROM
                ),
                "maximum_angstrom": (
                    CANARY_MAXIMUM_ANGSTROM
                ),
                "step_angstrom": (
                    CANARY_SCHEMA
                    .fast_grid
                    .step_angstrom
                ),
                "grid_level": (
                    CANARY_SCHEMA
                    .fast_grid
                    .grid_level
                ),
            },
            "refinement": {
                "basis": (
                    CANARY_SCHEMA
                    .refinement
                    .basis
                ),
                "window_angstrom": (
                    CANARY_SCHEMA
                    .refinement
                    .window_angstrom
                ),
                "step_angstrom": (
                    CANARY_SCHEMA
                    .refinement
                    .grid
                    .step_angstrom
                ),
                "grid_level": (
                    CANARY_SCHEMA
                    .refinement
                    .grid
                    .grid_level
                ),
            },
        },
        "electron_counts": {
            "neutral": neutral_electrons,
            "anion": anion_electrons,
        },
        "execution": {
            "fast_grid_tasks": (
                fast_plan.task_count
            ),
            "qzvpd_candidates": (
                qzvpd_plan
                .candidate_count
            ),
            "qzvpd_tasks": (
                qzvpd_plan.task_count
            ),
        },
        "qzvpd_functionals": [
            {
                "functional": (
                    result.functional
                ),
                "ea_eV": result.ea_ev,
                "recommended_for_summary": (
                    result
                    .recommended_for_summary
                ),
                "hard_warnings": list(
                    result.hard_warnings
                ),
                "diagnostic_warnings": list(
                    result
                    .diagnostic_warnings
                ),
            }
            for result
            in qzvpd_analysis.functional_eas
        ],
        "diagnostic_statistical_reduction": (
            _estimate_payload(
                estimate
            )
        ),
        "compute_provenance": provenance,
        "raw_files": {
            "fast_grid_csv": str(
                Path(
                    fast_grid_csv
                ).resolve()
            ),
            "qzvpd_csv": str(
                Path(
                    qzvpd_csv
                ).resolve()
            ),
        },
    }


def write_canary_report(
    path: str | Path,
    payload: dict[str, object],
) -> Path:
    """Write the dedicated non-scientific canary JSON report."""
    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return destination


def _verify_provenance(
    provenance: dict[str, object],
) -> None:
    compatibility = provenance.get(
        "compatibility"
    )

    if not isinstance(
        compatibility,
        dict,
    ):
        raise RuntimeError(
            "Compute provenance contains "
            "no compatibility record."
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


def run_real_canary(
    *,
    output_directory: str | Path,
    max_workers: int = 2,
) -> CanaryResult:
    """Execute the complete reduced-cost real workflow canary."""
    if max_workers < 1:
        raise ValueError(
            "max_workers must be at least 1."
        )

    output = Path(
        output_directory
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    fast_csv = (
        output
        / "canary_fast_grid_raw.csv"
    )

    qzvpd_csv = (
        output
        / "canary_qzvpd_raw.csv"
    )

    report_json = (
        output
        / "canary_report.json"
    )

    provenance = (
        collect_compute_provenance()
    )

    _verify_provenance(
        provenance
    )

    probe_r = 0.97

    neutral_electrons = (
        run_platform_electron_count(
            molecule=CANARY_MOLECULE,
            charge=ChargeState.NEUTRAL,
            basis="def2-svp",
            bond_length_angstrom=probe_r,
            max_memory_mb=1500,
        )
    )

    anion_electrons = (
        run_platform_electron_count(
            molecule=CANARY_MOLECULE,
            charge=ChargeState.ANION,
            basis="def2-svp",
            bond_length_angstrom=probe_r,
            max_memory_mb=1500,
        )
    )

    if neutral_electrons != 9:
        raise RuntimeError(
            "OH neutral electron-count validation "
            f"failed: {neutral_electrons}."
        )

    if anion_electrons != 10:
        raise RuntimeError(
            "OH anion electron-count validation "
            f"failed: {anion_electrons}."
        )

    fast_plan = (
        build_canary_fast_plan(
            neutral_electrons=(
                neutral_electrons
            ),
            anion_electrons=(
                anion_electrons
            ),
        )
    )

    fast_store = RawResultStore(
        fast_csv
    )

    fast_run = (
        execute_fast_grid_resumable(
            fast_plan,
            store=fast_store,
            max_workers=max_workers,
            retry_errors=True,
            worker=(
                run_platform_single_point
            ),
        )
    )

    if not fast_run.complete:
        raise RuntimeError(
            "Canary fast grid remains "
            "incomplete after execution."
        )

    fast_analysis = (
        analyze_fast_grid(
            fast_store,
            fast_plan.bond_grid,
        )
    )

    candidates = (
        select_qzvpd_candidates(
            fast_analysis,
            schema=CANARY_SCHEMA,
        )
    )

    if len(
        candidates
    ) != 8:
        raise RuntimeError(
            "Canary expected exactly 8 "
            "QZVPD candidates "
            "(4 functionals x 2 charges), "
            f"received {len(candidates)}."
        )

    qzvpd_plan = (
        build_qzvpd_plan(
            molecule=CANARY_MOLECULE,
            candidates=candidates,
            threads_per_worker=1,
            schema=CANARY_SCHEMA,
        )
    )

    qzvpd_store = (
        RawResultStore(
            qzvpd_csv
        )
    )

    qzvpd_run = (
        execute_qzvpd_resumable(
            qzvpd_plan,
            store=qzvpd_store,
            max_workers=max_workers,
            retry_errors=True,
            worker=(
                run_platform_single_point
            ),
        )
    )

    if not qzvpd_run.complete:
        raise RuntimeError(
            "Canary QZVPD refinement remains "
            "incomplete after execution."
        )

    qzvpd_analysis = (
        analyze_qzvpd(
            qzvpd_store,
            qzvpd_plan,
        )
    )

    if len(
        qzvpd_analysis.functional_eas
    ) != 4:
        raise RuntimeError(
            "Canary expected four functional "
            "QZVPD electron affinities, received "
            f"{len(qzvpd_analysis.functional_eas)}."
        )

    unreliable = [
        result.functional
        for result
        in qzvpd_analysis.functional_eas
        if not result.recommended_for_summary
    ]

    if unreliable:
        raise RuntimeError(
            "Canary QZVPD analysis contains "
            "hard warnings for: "
            + ", ".join(
                unreliable
            )
        )

    estimate = (
        estimate_schema_f(
            qzvpd_analysis
        )
    )

    if estimate.functional_count != 4:
        raise RuntimeError(
            "Canary statistical reduction did "
            "not contain all four functionals."
        )

    fast_resume = (
        execute_fast_grid_resumable(
            fast_plan,
            store=fast_store,
            max_workers=max_workers,
            retry_errors=True,
            worker=(
                run_platform_single_point
            ),
        )
    )

    qzvpd_resume = (
        execute_qzvpd_resumable(
            qzvpd_plan,
            store=qzvpd_store,
            max_workers=max_workers,
            retry_errors=True,
            worker=(
                run_platform_single_point
            ),
        )
    )

    if fast_resume.attempted != 0:
        raise RuntimeError(
            "Fast-grid resume attempted "
            "duplicate calculations."
        )

    if qzvpd_resume.attempted != 0:
        raise RuntimeError(
            "QZVPD resume attempted "
            "duplicate calculations."
        )

    report_payload = (
        build_canary_report_payload(
            neutral_electrons=(
                neutral_electrons
            ),
            anion_electrons=(
                anion_electrons
            ),
            fast_plan=fast_plan,
            qzvpd_plan=qzvpd_plan,
            qzvpd_analysis=(
                qzvpd_analysis
            ),
            estimate=estimate,
            provenance=provenance,
            fast_grid_csv=fast_csv,
            qzvpd_csv=qzvpd_csv,
        )
    )

    write_canary_report(
        report_json,
        report_payload,
    )

    passed = (
        fast_run.complete
        and qzvpd_run.complete
        and fast_resume.attempted == 0
        and qzvpd_resume.attempted == 0
        and estimate.functional_count == 4
        and report_json.is_file()
    )

    return CanaryResult(
        passed=passed,
        output_directory=str(
            output.resolve()
        ),
        fast_grid_csv=str(
            fast_csv.resolve()
        ),
        qzvpd_csv=str(
            qzvpd_csv.resolve()
        ),
        report_json=str(
            report_json.resolve()
        ),
        neutral_electrons=(
            neutral_electrons
        ),
        anion_electrons=(
            anion_electrons
        ),
        fast_task_count=(
            fast_plan.task_count
        ),
        candidate_count=len(
            candidates
        ),
        qzvpd_task_count=(
            qzvpd_plan.task_count
        ),
        fast_resume_attempted=(
            fast_resume.attempted
        ),
        qzvpd_resume_attempted=(
            qzvpd_resume.attempted
        ),
        estimate=estimate,
        message=(
            "Real reduced-cost end-to-end "
            "workflow canary passed."
            if passed
            else
            "Workflow canary failed."
        ),
    )


def main() -> int:
    """Run the real end-to-end workflow canary."""
    print()
    print(
        "DiatomicEA real workflow canary"
    )

    print(
        "=============================="
    )

    print()

    print(
        CANARY_WARNING
    )

    print()

    output = Path(
        tempfile.mkdtemp(
            prefix=(
                "diatomic-ea-canary-"
            )
        )
    )

    print(
        "Output directory:",
        output,
    )

    print(
        "Molecule:",
        CANARY_MOLECULE.formula,
    )

    print(
        "Fast basis:",
        CANARY_SCHEMA.fast_bases[0],
    )

    print(
        "QZVPD basis:",
        CANARY_SCHEMA.refinement.basis,
    )

    print(
        "Functionals:",
        ", ".join(
            CANARY_SCHEMA.functionals
        ),
    )

    print()

    try:
        result = run_real_canary(
            output_directory=output,
            max_workers=2,
        )

    except Exception as exc:
        print()
        print(
            "Status: FAIL"
        )

        print(
            "Error:",
            str(
                exc
            ),
        )

        print()
        print(
            "Partial output retained at:",
            output,
        )

        return 1

    print()
    print(
        "Neutral electrons:",
        result.neutral_electrons,
    )

    print(
        "Anion electrons:",
        result.anion_electrons,
    )

    print(
        "Fast-grid tasks:",
        result.fast_task_count,
    )

    print(
        "QZVPD candidates:",
        result.candidate_count,
    )

    print(
        "QZVPD tasks:",
        result.qzvpd_task_count,
    )

    print()

    for (
        functional,
        value,
    ) in result.estimate.functional_eas_ev:
        print(
            functional,
            "diagnostic QZVPD EA / eV:",
            value,
        )

    print()

    print(
        "Diagnostic statistical EA / eV:",
        result.estimate.predicted_ea_ev,
    )

    print()

    print(
        CANARY_WARNING
    )

    print()

    print(
        "Fast resume attempted:",
        result.fast_resume_attempted,
    )

    print(
        "QZVPD resume attempted:",
        result.qzvpd_resume_attempted,
    )

    print(
        "Report:",
        result.report_json,
    )

    print()

    print(
        "Status:",
        (
            "PASS"
            if result.passed
            else "FAIL"
        ),
    )

    print(
        "Message:",
        result.message,
    )

    return (
        0
        if result.passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
