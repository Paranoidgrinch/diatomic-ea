"""Command-line interface for DiatomicEA."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from diatomic_ea import __version__
from diatomic_ea.config import build_compute_config
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.resources import detect_cpu_resources


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


def _show_compute_config(workers: int | None) -> int:
    resources = detect_cpu_resources()

    try:
        config = build_compute_config(
            workers=workers,
            resources=resources,
        )
    except ValueError as exc:
        print(f"Invalid compute configuration: {exc}")
        return 2

    print("Compute configuration")
    print("---------------------")
    print(f"Workers            : {config.workers}")
    print(f"Threads per worker : {config.threads_per_worker}")
    print()
    print("Worker thread limits")

    for name, value in config.worker_environment().items():
        print(f"{name}={value}")

    return 0


def _show_molecule(atom_a: str, atom_b: str) -> int:
    try:
        molecule = DiatomicMolecule(atom_a, atom_b)
    except ValueError as exc:
        print(f"Invalid molecule: {exc}")
        return 2

    print("Diatomic molecule")
    print("-----------------")
    print(f"Atom A : {molecule.atom_a}")
    print(f"Atom B : {molecule.atom_b}")
    print(f"Formula: {molecule.formula}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the DiatomicEA command-line parser."""
    parser = argparse.ArgumentParser(
        prog="diatomic-ea",
        description=(
            "Fast and reproducible electron-affinity calculations "
            "for diatomic molecules."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"DiatomicEA {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "system-info",
        help="Show detected CPU resources.",
    )

    config_parser = subparsers.add_parser(
        "compute-config",
        help="Show calculation resource configuration.",
    )
    config_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of calculation workers to use.",
    )

    molecule_parser = subparsers.add_parser(
        "molecule",
        help="Validate and display a diatomic molecule.",
    )
    molecule_parser.add_argument("atom_a")
    molecule_parser.add_argument("atom_b")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DiatomicEA command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "system-info":
        return _show_system_info()

    if args.command == "compute-config":
        return _show_compute_config(args.workers)

    if args.command == "molecule":
        return _show_molecule(
            args.atom_a,
            args.atom_b,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())