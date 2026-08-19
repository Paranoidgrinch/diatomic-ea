"""Tests for stdin forwarding through the WSL bridge."""

from types import SimpleNamespace
from unittest.mock import patch

from diatomic_ea.wsl import (
    run_wsl_command,
)


def completed():
    return SimpleNamespace(
        returncode=0,
        stdout=b"ok\n",
        stderr=b"",
    )


def test_run_wsl_command_forwards_utf8_stdin() -> None:
    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            return_value=completed(),
        ) as mocked,
    ):
        result = run_wsl_command(
            (
                "cat",
            ),
            distribution="Ubuntu-24.04",
            input_text='{"hello":"world"}',
            timeout=12.0,
        )

    assert result.succeeded
    assert result.stdout == "ok\n"

    command = mocked.call_args.args[0]
    kwargs = mocked.call_args.kwargs

    assert command == [
        "wsl.exe",
        "--distribution",
        "Ubuntu-24.04",
        "--",
        "cat",
    ]

    assert (
        kwargs["input"]
        == b'{"hello":"world"}'
    )

    assert (
        kwargs["timeout"]
        == 12.0
    )


def test_run_wsl_command_without_stdin_passes_none() -> None:
    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            return_value=completed(),
        ) as mocked,
    ):
        run_wsl_command(
            (
                "true",
            )
        )

    kwargs = mocked.call_args.kwargs

    assert (
        kwargs["input"]
        is None
    )
