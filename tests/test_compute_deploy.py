"""Tests for exact DiatomicEA wheel deployment into WSL."""

import hashlib
import json
from unittest.mock import patch

import pytest

from diatomic_ea.compute_deploy import (
    deploy_wsl_wheel,
    sha256_file,
    windows_path_to_wsl,
)
from diatomic_ea.compute_environment import (
    WSL_COMPUTE_PYTHON,
    WSL_COMPUTE_VENV,
)
from diatomic_ea.single_point_protocol import (
    SINGLE_POINT_PROTOCOL_VERSION,
)
from diatomic_ea.wsl import (
    WSLCommandResult,
)


def result(
    *,
    returncode=0,
    stdout="",
    stderr="",
):
    return WSLCommandResult(
        command=("wsl.exe",),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_sha256_file(tmp_path) -> None:
    path = (
        tmp_path
        / "test.whl"
    )

    path.write_bytes(
        b"abc"
    )

    assert (
        sha256_file(
            path
        )
        == hashlib.sha256(
            b"abc"
        ).hexdigest()
    )


def test_windows_path_translation(tmp_path) -> None:
    wheel = (
        tmp_path
        / "package.whl"
    )

    wheel.write_bytes(
        b"wheel"
    )

    with patch(
        "diatomic_ea.compute_deploy.run_wsl_command",
        return_value=result(
            stdout="/mnt/c/package.whl\n"
        ),
    ) as command:
        translated = windows_path_to_wsl(
            wheel,
            distribution="Ubuntu-24.04",
        )

    assert (
        translated
        == "/mnt/c/package.whl"
    )

    arguments = (
        command.call_args.args[0]
    )

    assert arguments[0:3] == (
        "wslpath",
        "-a",
        "-u",
    )

    supplied_path = arguments[3]

    assert "\\" not in supplied_path
    assert "/" in supplied_path


def test_successful_wheel_deployment(tmp_path) -> None:
    wheel = (
        tmp_path
        / "diatomic_ea-test.whl"
    )

    wheel.write_bytes(
        b"exact wheel bytes"
    )

    expected_hash = sha256_file(
        wheel
    )

    verify_payload = json.dumps(
        {
            "package_version": "0.1.0.dev0",
            "module_path": (
                WSL_COMPUTE_VENV
                + "/lib/python3.12/"
                "site-packages/diatomic_ea/"
                "__init__.py"
            ),
            "worker_module": (
                WSL_COMPUTE_VENV
                + "/lib/python3.12/"
                "site-packages/diatomic_ea/"
                "single_point_worker.py"
            ),
            "protocol_version": (
                SINGLE_POINT_PROTOCOL_VERSION
            ),
            "wheel_sha256": (
                expected_hash
            ),
        }
    )

    with patch(
        "diatomic_ea.compute_deploy.run_wsl_command",
        side_effect=[
            result(
                stdout=(
                    "/mnt/c/"
                    "diatomic_ea-test.whl\n"
                )
            ),
            result(),
            result(),
            result(
                stdout=verify_payload
            ),
        ],
    ) as command:
        deployment = deploy_wsl_wheel(
            wheel,
            distribution="Ubuntu-24.04",
        )

    assert deployment.success

    assert (
        deployment.wheel_sha256
        == expected_hash
    )

    calls = command.call_args_list

    install_arguments = (
        calls[1].args[0]
    )

    assert install_arguments[0] == (
        WSL_COMPUTE_PYTHON
    )

    assert install_arguments[1:4] == (
        "-m",
        "pip",
        "install",
    )

    assert (
        "--force-reinstall"
        in install_arguments
    )

    assert (
        "--no-deps"
        in install_arguments
    )

    assert (
        calls[1].kwargs["user"]
        == "root"
    )

    assert (
        calls[2].kwargs["user"]
        == "root"
    )


def test_install_failure_is_reported(tmp_path) -> None:
    wheel = (
        tmp_path
        / "package.whl"
    )

    wheel.write_bytes(
        b"wheel"
    )

    with patch(
        "diatomic_ea.compute_deploy.run_wsl_command",
        side_effect=[
            result(
                stdout="/mnt/c/package.whl\n"
            ),
            result(
                returncode=1,
                stderr="pip failed",
            ),
        ],
    ):
        deployment = deploy_wsl_wheel(
            wheel,
            distribution="Ubuntu-24.04",
        )

    assert not deployment.success

    assert (
        "pip failed"
        in deployment.message
    )


def test_hash_mismatch_is_rejected(tmp_path) -> None:
    wheel = (
        tmp_path
        / "package.whl"
    )

    wheel.write_bytes(
        b"wheel"
    )

    verify_payload = json.dumps(
        {
            "package_version": "0.1.0.dev0",
            "module_path": (
                WSL_COMPUTE_VENV
                + "/site-packages/"
                "diatomic_ea/__init__.py"
            ),
            "worker_module": (
                WSL_COMPUTE_VENV
                + "/site-packages/"
                "diatomic_ea/"
                "single_point_worker.py"
            ),
            "protocol_version": (
                SINGLE_POINT_PROTOCOL_VERSION
            ),
            "wheel_sha256": (
                "wrong-hash"
            ),
        }
    )

    with patch(
        "diatomic_ea.compute_deploy.run_wsl_command",
        side_effect=[
            result(
                stdout="/mnt/c/package.whl\n"
            ),
            result(),
            result(),
            result(
                stdout=verify_payload
            ),
        ],
    ):
        deployment = deploy_wsl_wheel(
            wheel,
            distribution="Ubuntu-24.04",
        )

    assert not deployment.success

    assert (
        "hash"
        in deployment.message.lower()
    )


def test_non_wheel_path_is_rejected(tmp_path) -> None:
    path = (
        tmp_path
        / "not-a-wheel.txt"
    )

    path.write_text(
        "x",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=".whl",
    ):
        deploy_wsl_wheel(
            path
        )
