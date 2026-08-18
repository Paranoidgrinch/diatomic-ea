"""Command-line interface for DiatomicEA."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from diatomic_ea import __version__
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DiatomicEA command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "system-info":
        return _show_system_info()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
