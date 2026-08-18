"""Basic package tests."""

from diatomic_ea import __version__
from diatomic_ea.cli import build_parser


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0.dev0"


def test_cli_program_name() -> None:
    parser = build_parser()
    assert parser.prog == "diatomic-ea"
