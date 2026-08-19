"""Basic package tests."""

from importlib.metadata import version

from diatomic_ea import __version__
from diatomic_ea.cli import build_parser


def test_version_is_defined() -> None:
    assert __version__ == version("diatomic-ea")


def test_cli_program_name() -> None:
    parser = build_parser()
    assert parser.prog == "diatomic-ea"
