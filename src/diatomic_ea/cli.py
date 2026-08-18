"""Command-line interface for DiatomicEA."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from diatomic_ea import __version__


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the DiatomicEA command-line interface."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
