"""Tests for the WSL command bridge."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from diatomic_ea.wsl import (
    WSLCommandError,
    decode_wsl_output,
    inspect_wsl,
    list_wsl_distributions,
    run_wsl_command,
    run_wsl_shell,
)


def completed(
    *,
    returncode=0,
    stdout=b"",
    stderr=b"",
):
    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_decode_utf8_output() -> None:
    data = (
        "Ubuntu\nDebian\n"
        .encode("utf-8")
    )

    assert decode_wsl_output(
        data
    ) == "Ubuntu\nDebian\n"


def test_decode_utf16_output() -> None:
    data = (
        "Ubuntu\nDebian\n"
        .encode("utf-16-le")
    )

    assert decode_wsl_output(
        data
    ) == "Ubuntu\nDebian\n"


def test_list_wsl_distributions() -> None:
    output = (
        "Ubuntu-24.04\nDebian\n"
        .encode("utf-16-le")
    )

    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="C:/Windows/System32/wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            return_value=completed(
                stdout=output
            ),
        ),
    ):
        distributions = (
            list_wsl_distributions()
        )

    assert distributions == (
        "Ubuntu-24.04",
        "Debian",
    )


def test_inspect_wsl_without_executable() -> None:
    with patch(
        "diatomic_ea.wsl.shutil.which",
        return_value=None,
    ):
        availability = inspect_wsl()

    assert not availability.ready
    assert not availability.executable_found
    assert availability.distributions == ()


def test_run_default_distribution_command() -> None:
    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            return_value=completed(
                stdout=b"Linux\n"
            ),
        ) as mocked,
    ):
        result = run_wsl_command(
            (
                "uname",
                "-s",
            )
        )

    assert result.succeeded
    assert result.stdout == "Linux\n"

    command = mocked.call_args.args[0]

    assert command == [
        "wsl.exe",
        "--",
        "uname",
        "-s",
    ]


def test_run_named_distribution_command() -> None:
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
                "python3",
                "--version",
            ),
            distribution="Ubuntu-24.04",
        )

    command = mocked.call_args.args[0]

    assert command == [
        "wsl.exe",
        "--distribution",
        "Ubuntu-24.04",
        "--",
        "python3",
        "--version",
    ]


def test_checked_command_raises() -> None:
    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            return_value=completed(
                returncode=7,
                stderr=b"failure",
            ),
        ),
    ):
        with pytest.raises(
            WSLCommandError
        ):
            run_wsl_command(
                ("false",),
                check=True,
            )


def test_run_wsl_shell() -> None:
    with patch(
        "diatomic_ea.wsl.run_wsl_command"
    ) as mocked:
        run_wsl_shell(
            "printf hello",
            distribution="Ubuntu",
            timeout=12.0,
        )

    mocked.assert_called_once_with(
        (
            "sh",
            "-lc",
            "printf hello",
        ),
        distribution="Ubuntu",
        timeout=12.0,
        check=False,
    )