#!/usr/bin/env python3
"""Build reproducible DiatomicEA release archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BUNDLE_FILES = (
    "README.md",
    "LICENSE",
    "docs/INSTALLATION.md",
    "docs/USER_GUIDE.md",
    "scripts/install_windows.ps1",
    "scripts/uninstall_windows.ps1",
    "scripts/install_linux.sh",
    "scripts/uninstall_linux.sh",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_version(root: str | Path) -> str:
    text = (Path(root) / "pyproject.toml").read_text(encoding="utf-8-sig")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not read project version from pyproject.toml.")
    return match.group(1)


def build_release(root: str | Path, out: str | Path | None = None) -> Path:
    root_path = Path(root).resolve()
    version = project_version(root_path)
    dist = Path(out).resolve() if out is not None else root_path / "dist"
    python_dist = dist / "python"

    if dist.exists():
        shutil.rmtree(dist)
    python_dist.mkdir(parents=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--outdir",
            str(python_dist),
        ],
        cwd=root_path,
        check=True,
    )

    wheels = list(python_dist.glob("diatomic_ea-*.whl"))
    sdists = list(python_dist.glob("diatomic_ea-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("Expected exactly one wheel and one source archive.")

    for relative in BUNDLE_FILES:
        if not (root_path / relative).is_file():
            raise RuntimeError(f"Missing release file: {relative}")

    bundle = dist / f"DiatomicEA-{version}.zip"
    prefix = f"DiatomicEA-{version}"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(wheels[0], f"{prefix}/{wheels[0].name}")
        for relative in BUNDLE_FILES:
            archive.write(root_path / relative, f"{prefix}/{relative}")

    artifacts = (wheels[0], sdists[0], bundle)
    sums = "\n".join(f"{sha256_file(path)}  {path.name}" for path in artifacts) + "\n"
    (dist / "SHA256SUMS.txt").write_text(sums, encoding="ascii")

    manifest = {
        "version": version,
        "artifacts": [
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
    }
    (dist / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Version: {version}")
    for path in artifacts:
        print(f"Artifact: {path}")
    print(f"Checksums: {dist / 'SHA256SUMS.txt'}")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    build_release(args.root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
