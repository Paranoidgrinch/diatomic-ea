"""Basic package tests."""

from pathlib import Path
import tomllib

from diatomic_ea import __version__
from diatomic_ea.cli import build_parser


def test_version_is_defined() -> None:
    pyproject_path = (
        Path(__file__).resolve().parents[1]
        / "pyproject.toml"
    )

    metadata = tomllib.loads(
        pyproject_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        __version__
        == metadata["project"]["version"]
    )


def test_cli_program_name() -> None:
    parser = build_parser()
    assert parser.prog == "diatomic-ea"
