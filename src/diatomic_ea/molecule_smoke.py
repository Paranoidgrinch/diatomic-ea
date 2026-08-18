"""Synthetic end-to-end molecule smoke test.

This module deliberately does not perform electronic-structure
calculations. It exercises the complete DiatomicEA execution path with
deterministic synthetic energies.

Values produced by this module are NOT scientific EA predictions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.manifest import (
    write_reproducibility_manifest,
)
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.pipeline import (
    PipelineRequest,
    run_schema_f_pipeline,
)
from diatomic_ea.schema_f import HARTREE_TO_EV
from diatomic_ea.single_point import (
    FrontierOrbitals,
    SinglePointResult,
    SinglePointStatus,
)
from diatomic_ea.states import ChargeState


SMOKE_NOTICE = """\
DIATOMICEA SYNTHETIC MOLECULE SMOKE TEST

THIS RUN IS NOT A SCIENTIFIC ELECTRON-AFFINITY CALCULATION.

NO PySCF ELECTRONIC-STRUCTURE ENERGIES WERE USED.
ALL ENERGIES IN THIS RUN ARE SYNTHETIC TEST VALUES.
THE EA-LIKE OUTPUT EXISTS ONLY TO TEST SOFTWARE EXECUTION.
"""


_SYNTHETIC_EA_HARTREE = {
    "PBE": 0.095,
    "B3LYP": 0.100,
    "PBE0": 0.105,
    "TPSSh": 0.110,
}


_FUNCTIONAL_INDEX = {
    "PBE": 0,
    "B3LYP": 1,
    "PBE0": 2,
    "TPSSh": 3,
}


_BASIS_INDEX = {
    "def2-svp": 0,
    "def2-tzvp": 1,
    "def2-tzvpp": 2,
    "def2-svpd": 3,
    "def2-tzvpd": 4,
    "def2-qzvpd": 5,
}


@dataclass(frozen=True, slots=True)
class MoleculeSmokeReport:
    """Result of a successful synthetic end-to-end smoke test."""

    run_dir: Path
    final_result_csv: Path
    manifest_json: Path
    synthetic_predicted_ea_ev: float
    fast_grid_task_count: int
    qzvpd_task_count: int


def synthetic_electron_count(
    *,
    molecule,
    charge,
    basis,
    bond_length_angstrom,
    max_memory_mb,
) -> int:
    """Return deterministic smoke-test electron counts."""
    if charge is ChargeState.NEUTRAL:
        return 10

    if charge is ChargeState.ANION:
        return 11

    raise ValueError(
        "Synthetic smoke test supports only "
        "neutral and anionic states."
    )


def synthetic_single_point(
    task,
) -> SinglePointResult:
    """Return one deterministic synthetic single-point result."""
    functional_index = (
        _FUNCTIONAL_INDEX[
            task.functional
        ]
    )

    basis_index = _BASIS_INDEX.get(
        task.basis,
        0,
    )

    equilibrium_r = 1.525

    energy = (
        -100.0
        - 0.01 * functional_index
        - 0.001 * basis_index
        + 0.5
        * (
            task.bond_length_angstrom
            - equilibrium_r
        ) ** 2
    )

    if task.charge is ChargeState.ANION:
        energy -= (
            _SYNTHETIC_EA_HARTREE[
                task.functional
            ]
        )

    electron_count = (
        10
        if task.charge
        is ChargeState.NEUTRAL
        else 11
    )

    alpha_electrons = (
        electron_count
        + task.spin
    ) // 2

    beta_electrons = (
        electron_count
        - task.spin
    ) // 2

    s_value = (
        task.spin / 2.0
    )

    return SinglePointResult(
        task_id=task.task_id,
        status=SinglePointStatus.OK,
        error="",
        energy_hartree=energy,
        energy_ev=(
            energy
            * HARTREE_TO_EV
        ),
        converged=True,
        used_level_shift_retry=False,
        used_newton_retry=False,
        electron_count=electron_count,
        alpha_electrons=alpha_electrons,
        beta_electrons=beta_electrons,
        basis_label_a="synthetic-smoke",
        basis_label_b="synthetic-smoke",
        ecp_label_a="",
        ecp_label_b="",
        frontier=FrontierOrbitals(
            homo_hartree=-0.2,
            lumo_hartree=0.1,
            homo_ev=(
                -0.2
                * HARTREE_TO_EV
            ),
            lumo_ev=(
                0.1
                * HARTREE_TO_EV
            ),
            gap_ev=(
                0.3
                * HARTREE_TO_EV
            ),
            positive_homo_warning=False,
        ),
        s2=(
            s_value
            * (s_value + 1.0)
        ),
        observed_multiplicity=(
            task.multiplicity
        ),
        spin_contamination_warning=False,
        pyscf_version="synthetic-smoke",
        elapsed_seconds=0.001,
    )


def _mark_smoke_outputs(
    run_dir: Path,
    final_dir: Path,
    manifest_path: Path,
) -> None:
    """Mark generated files as synthetic test output."""
    run_marker = (
        run_dir
        / "SMOKE_TEST_ONLY.txt"
    )

    final_marker = (
        final_dir
        / "SMOKE_TEST_ONLY.txt"
    )

    run_marker.write_text(
        SMOKE_NOTICE,
        encoding="utf-8",
    )

    final_marker.write_text(
        SMOKE_NOTICE,
        encoding="utf-8",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    manifest["smoke_test"] = {
        "synthetic": True,
        "scientific_prediction": False,
        "notice": (
            "Synthetic software validation only. "
            "Not a scientific EA prediction."
        ),
    }

    write_reproducibility_manifest(
        manifest_path,
        manifest,
    )


def run_molecule_smoke_test(
    *,
    output_root: str | Path,
    workers: int = 1,
) -> MoleculeSmokeReport:
    """Run the complete synthetic molecule pipeline."""
    if workers < 1:
        raise ValueError(
            "workers must be at least 1."
        )

    request = PipelineRequest(
        molecule=DiatomicMolecule(
            "H",
            "F",
        ),
        minimum_angstrom=1.50,
        maximum_angstrom=1.55,
        spin_max=1,
        workers=workers,
        threads_per_worker=1,
    )

    result = run_schema_f_pipeline(
        request,
        output_root=output_root,
        electron_count_resolver=(
            synthetic_electron_count
        ),
        worker=synthetic_single_point,
    )

    _mark_smoke_outputs(
        result.paths.run_dir,
        result.paths.final_dir,
        result.paths.manifest_json,
    )

    return MoleculeSmokeReport(
        run_dir=result.paths.run_dir,
        final_result_csv=(
            result.paths.final_result_csv
        ),
        manifest_json=(
            result.paths.manifest_json
        ),
        synthetic_predicted_ea_ev=(
            result.estimate.predicted_ea_ev
        ),
        fast_grid_task_count=(
            result.fast_plan.task_count
        ),
        qzvpd_task_count=(
            result.qzvpd_plan.task_count
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone smoke-test command parser."""
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "diatomic_ea.molecule_smoke"
        ),
        description=(
            "Run the synthetic DiatomicEA "
            "end-to-end molecule smoke test."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="smoke_runs",
        help=(
            "Root directory for synthetic "
            "smoke-test output."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of synthetic worker processes."
        ),
    )

    return parser


def main() -> int:
    """Run the command-line smoke test."""
    parser = build_parser()

    args = parser.parse_args()

    print()
    print(
        "DiatomicEA molecule smoke test"
    )

    print(
        "=============================="
    )

    print()

    print(
        "SYNTHETIC TEST ONLY - "
        "NOT A SCIENTIFIC EA PREDICTION"
    )

    print()

    try:
        report = run_molecule_smoke_test(
            output_root=args.output_dir,
            workers=args.workers,
        )

    except Exception as exc:
        print(
            "[FAIL] End-to-end molecule "
            f"smoke test failed: {exc}"
        )

        return 1

    print(
        "[PASS] Complete synthetic "
        "Schema F pipeline executed."
    )

    print(
        "Fast-grid tasks : "
        f"{report.fast_grid_task_count}"
    )

    print(
        "QZVPD tasks      : "
        f"{report.qzvpd_task_count}"
    )

    print(
        "Run directory    : "
        f"{report.run_dir}"
    )

    print(
        "Final CSV        : "
        f"{report.final_result_csv}"
    )

    print(
        "Manifest         : "
        f"{report.manifest_json}"
    )

    print()

    print(
        "Synthetic EA-like result: "
        f"{report.synthetic_predicted_ea_ev:.6f} eV"
    )

    print()

    print(
        "IMPORTANT: This value was generated "
        "from synthetic energies."
    )

    print(
        "It is NOT a scientific electron-affinity "
        "prediction."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )