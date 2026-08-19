"""Release/distribution contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import diatomic_ea


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"


def test_release_version_is_consistent() -> None:
    pyproject = (
        ROOT / "pyproject.toml"
    ).read_text(
        encoding="utf-8-sig"
    )

    match = re.search(
        r'^version\s*=\s*"([^"]+)"\s*$',
        pyproject,
        re.MULTILINE,
    )

    assert match is not None
    assert match.group(1) == VERSION
    assert diatomic_ea.__version__ == VERSION


def test_release_metadata_and_gui_entry_point() -> None:
    pyproject = (
        ROOT / "pyproject.toml"
    ).read_text(
        encoding="utf-8-sig"
    )

    assert 'license = "GPL-3.0-only"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert (
        'diatomic-ea-gui = "diatomic_ea.desktop_gui:main"'
        in pyproject
    )
    assert "setuptools>=77" in pyproject


def test_release_files_exist() -> None:
    required = (
        "LICENSE",
        "docs/INSTALLATION.md",
        "docs/USER_GUIDE.md",
        "scripts/build_release.py",
        "scripts/install_windows.ps1",
        "scripts/uninstall_windows.ps1",
        "scripts/install_linux.sh",
        "scripts/uninstall_linux.sh",
    )

    for relative in required:
        assert (
            ROOT / relative
        ).is_file(), relative


def test_windows_installers_are_repo_location_independent() -> None:
    installer = (
        ROOT / "scripts/install_windows.ps1"
    ).read_text(
        encoding="utf-8"
    )

    uninstaller = (
        ROOT / "scripts/uninstall_windows.ps1"
    ).read_text(
        encoding="utf-8"
    )

    assert "$PSScriptRoot" in installer
    assert "compute_bootstrap" in installer
    assert "compute_deploy" in installer
    assert "compute_smoke" in installer
    assert "Restart-Computer" not in installer
    assert "shutdown.exe" not in installer
    assert "Restart-Computer" not in uninstaller


def test_linux_installer_is_native_compute_install() -> None:
    installer = (
        ROOT / "scripts/install_linux.sh"
    ).read_text(
        encoding="utf-8"
    )

    assert installer.startswith(
        "#!/usr/bin/env bash\nset -euo pipefail"
    )
    assert "pyscf==2.13.0" in installer
    assert "basis-set-exchange" in installer
    assert "compute_smoke" in installer
    assert "wsl" not in installer.casefold()


def test_readme_describes_stable_release() -> None:
    readme = (
        ROOT / "README.md"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "Version **1.0.0** is the first stable release."
        in readme
    )
    assert "Early development" not in readme
    assert "GPL-3.0-only" in readme
    assert "rc" not in VERSION.casefold()
