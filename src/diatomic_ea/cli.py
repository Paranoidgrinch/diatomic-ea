"""Command-line interface for DiatomicEA."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from diatomic_ea import __version__
from diatomic_ea.config import build_compute_config
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.resources import detect_cpu_resources
from diatomic_ea.schema_f import SCHEMA_F
from diatomic_ea.states import build_state_scan_plan
from diatomic_ea.smoke import run_system_smoke_test


def _show_system_info() -> int:
    resources = detect_cpu_resources()

    print(f"DiatomicEA {__version__}")
    print()
    print("Detected CPU resources")
    print("----------------------")
    print(f"Physical CPU cores : {resources.physical_cores}")
    print(f"Logical CPU cores  : {resources.logical_cores}")
    print(f"Recommended workers: {resources.recommended_workers}")

    return 0


def _show_compute_config(
    workers: int | None,
) -> int:
    resources = detect_cpu_resources()

    try:
        config = build_compute_config(
            workers=workers,
            resources=resources,
        )
    except ValueError as exc:
        print(
            f"Invalid compute configuration: {exc}"
        )
        return 2

    print("Compute configuration")
    print("---------------------")
    print(f"Workers            : {config.workers}")
    print(
        "Threads per worker : "
        f"{config.threads_per_worker}"
    )
    print()
    print("Worker thread limits")

    for name, value in (
        config.worker_environment().items()
    ):
        print(f"{name}={value}")

    return 0


def _show_molecule(
    atom_a: str,
    atom_b: str,
) -> int:
    try:
        molecule = DiatomicMolecule(
            atom_a,
            atom_b,
        )
    except ValueError as exc:
        print(f"Invalid molecule: {exc}")
        return 2

    print("Diatomic molecule")
    print("-----------------")
    print(f"Atom A : {molecule.atom_a}")
    print(f"Atom B : {molecule.atom_b}")
    print(f"Formula: {molecule.formula}")

    return 0


def _show_method_info() -> int:
    print("Schema F")
    print("========")
    print(f"Schema ID           : {SCHEMA_F.schema_id}")
    print(
        "Reference PySCF     : "
        f"{SCHEMA_F.reference_pyscf_version}"
    )
    print(
        "Electronic method   : "
        f"{SCHEMA_F.electronic_structure_method}"
    )
    print(
        "Functionals         : "
        + ", ".join(SCHEMA_F.functionals)
    )
    print(
        "Fast-grid bases     : "
        + ", ".join(SCHEMA_F.fast_bases)
    )
    print(
        "Fast-grid step      : "
        f"{SCHEMA_F.fast_grid.step_angstrom} A"
    )
    print(
        "QZVPD basis         : "
        f"{SCHEMA_F.refinement.basis}"
    )
    print(
        "QZVPD window        : +/-"
        f"{SCHEMA_F.refinement.window_angstrom} A"
    )
    print(
        "QZVPD step          : "
        f"{SCHEMA_F.refinement.grid.step_angstrom} A"
    )
    print(
        "Max spins / charge  : "
        f"{SCHEMA_F.refinement.max_spins_per_charge}"
    )
    print()
    print(
        "Schema F is a fixed scientific preset."
    )

    return 0

def _show_state_scan(
    neutral_electrons: int,
    anion_electrons: int,
    spin_max: int,
) -> int:
    try:
        plan = build_state_scan_plan(
            neutral_electrons=neutral_electrons,
            anion_electrons=anion_electrons,
            spin_max=spin_max,
        )
    except ValueError as exc:
        print(f"Invalid state scan: {exc}")
        return 2

    print("Electronic-state scan")
    print("=====================")

    for scan in (
        plan.neutral,
        plan.anion,
    ):
        label = (
            "Neutral"
            if scan.charge == 0
            else "Anion"
        )

        spins = ", ".join(
            str(state.spin)
            for state in scan.states
        )

        multiplicities = ", ".join(
            str(state.multiplicity)
            for state in scan.states
        )

        print()
        print(
            f"{label} electrons      : "
            f"{scan.electron_count}"
        )
        print(
            f"{label} PySCF spins     : "
            f"{spins}"
        )
        print(
            f"{label} multiplicities  : "
            f"{multiplicities}"
        )

    return 0

def _run_smoke_test(
    output_dir: str,
) -> int:
    print(f"DiatomicEA {__version__}")
    print()
    print("System smoke test")
    print("=================")

    report = run_system_smoke_test(
        output_dir=output_dir,
    )

    for check in report.checks:
        symbol = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        print(
            f"[{symbol}] "
            f"{check.name}: "
            f"{check.message}"
        )

    print()
    print(
        f"{report.passed_count} passed, "
        f"{report.failed_count} failed"
    )

    if report.passed:
        print()
        print(
            "System ready for DiatomicEA "
            "base calculations."
        )
        return 0

    print()
    print(
        "System smoke test failed."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Create the DiatomicEA command-line parser."""
    parser = argparse.ArgumentParser(
        prog="diatomic-ea",
        description=(
            "Fast and reproducible electron-affinity "
            "calculations for diatomic molecules."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"DiatomicEA {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    subparsers.add_parser(
        "system-info",
        help="Show detected CPU resources.",
    )

    config_parser = subparsers.add_parser(
        "compute-config",
        help=(
            "Show calculation resource "
            "configuration."
        ),
    )

    config_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of calculation workers "
            "to use."
        ),
    )

    molecule_parser = subparsers.add_parser(
        "molecule",
        help=(
            "Validate and display a "
            "diatomic molecule."
        ),
    )

    molecule_parser.add_argument("atom_a")
    molecule_parser.add_argument("atom_b")

    subparsers.add_parser(
        "method-info",
        help="Show the frozen Schema F specification.",
    )
    state_parser = subparsers.add_parser(
        "state-scan",
        help="Display a neutral/anion spin-state scan.",
    )
    state_parser.add_argument(
        "--neutral-electrons",
        type=int,
        required=True,
    )
    state_parser.add_argument(
        "--anion-electrons",
        type=int,
        required=True,
    )
    state_parser.add_argument(
        "--spin-max",
        type=int,
        required=True,
    )
    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help=(
            "Run DiatomicEA system "
            "smoke tests."
        ),
    )

    smoke_parser.add_argument(
        "--output-dir",
        default="results",
        help=(
            "Directory to test for "
            "result-file output."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the DiatomicEA command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "system-info":
        return _show_system_info()

    if args.command == "compute-config":
        return _show_compute_config(
            args.workers
        )

    if args.command == "molecule":
        return _show_molecule(
            args.atom_a,
            args.atom_b,
        )

    if args.command == "method-info":
        return _show_method_info()
    if args.command == "state-scan":
        return _show_state_scan(
            args.neutral_electrons,
            args.anion_electrons,
            args.spin_max,
        )
    if args.command == "smoke-test":
        return _run_smoke_test(
            args.output_dir
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())