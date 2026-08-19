"""Tests for the WSL command bridge."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from diatomic_ea.wsl import (
    WSLCommandError,
    WSL_TIMEOUT_RETURN_CODE,
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


def test_decode_string_output() -> None:
    assert (
        decode_wsl_output(
            "already decoded"
        )
        == "already decoded"
    )


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


def test_list_timeout_returns_empty_tuple() -> None:
    timeout = subprocess.TimeoutExpired(
        cmd=[
            "wsl.exe",
            "--list",
            "--quiet",
        ],
        timeout=15.0,
    )

    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            side_effect=timeout,
        ),
    ):
        assert (
            list_wsl_distributions()
            == ()
        )


def test_inspect_wsl_without_executable() -> None:
    with patch(
        "diatomic_ea.wsl.shutil.which",
        return_value=None,
    ):
        availability = inspect_wsl()

    assert not availability.ready
    assert not availability.executable_found
    assert not availability.timed_out
    assert availability.distributions == ()


def test_inspect_wsl_timeout_is_reported() -> None:
    timeout = subprocess.TimeoutExpired(
        cmd=[
            "wsl.exe",
            "--list",
            "--quiet",
        ],
        timeout=5.0,
    )

    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            side_effect=timeout,
        ),
    ):
        availability = inspect_wsl(
            timeout=5.0
        )

    assert not availability.ready
    assert availability.executable_found
    assert availability.timed_out

    assert (
        "timed out after 5 seconds"
        in availability.message
    )


def test_run_command_timeout_is_result() -> None:
    timeout = subprocess.TimeoutExpired(
        cmd=[
            "wsl.exe",
            "--",
            "uname",
        ],
        timeout=2.0,
        output=b"partial",
        stderr=b"",
    )

    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            side_effect=timeout,
        ),
    ):
        result = run_wsl_command(
            ("uname",),
            timeout=2.0,
        )

    assert result.timed_out
    assert (
        result.returncode
        == WSL_TIMEOUT_RETURN_CODE
    )
    assert result.stdout == "partial"
    assert "timed out" in result.stderr


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


def test_checked_timeout_raises() -> None:
    timeout = subprocess.TimeoutExpired(
        cmd=[
            "wsl.exe",
            "--",
            "sleep",
            "10",
        ],
        timeout=1.0,
    )

    with (
        patch(
            "diatomic_ea.wsl.shutil.which",
            return_value="wsl.exe",
        ),
        patch(
            "diatomic_ea.wsl.subprocess.run",
            side_effect=timeout,
        ),
    ):
        with pytest.raises(
            WSLCommandError
        ):
            run_wsl_command(
                (
                    "sleep",
                    "10",
                ),
                timeout=1.0,
                check=True,
            )


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