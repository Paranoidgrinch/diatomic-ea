"""Standard run-directory layout for DiatomicEA."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from diatomic_ea.molecule import DiatomicMolecule


_SAFE_COMPONENT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
)


def _validate_component(
    value: str,
    *,
    label: str,
) -> str:
    value = value.strip()

    if not value:
        raise ValueError(
            f"{label} must not be empty."
        )

    if value in {".", ".."}:
        raise ValueError(
            f"Invalid {label}: {value!r}."
        )

    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{label} contains unsafe characters: "
            f"{value!r}."
        )

    return value


def new_run_id() -> str:
    """Create a compact sortable unique run identifier."""
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    suffix = uuid.uuid4().hex[:8]

    return f"{timestamp}-{suffix}"


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem locations belonging to one calculation run."""

    output_root: Path
    molecule_dir: Path
    run_dir: Path
    run_id: str
    fast_grid_dir: Path
    fast_analysis_dir: Path
    qzvpd_dir: Path
    final_dir: Path
    logs_dir: Path

    @property
    def fast_grid_csv(self) -> Path:
        return (
            self.fast_grid_dir
            / "raw_results.csv"
        )

    @property
    def qzvpd_csv(self) -> Path:
        return (
            self.qzvpd_dir
            / "raw_results.csv"
        )

    @property
    def final_result_csv(self) -> Path:
        return (
            self.final_dir
            / "final_result.csv"
        )

    @property
    def manifest_json(self) -> Path:
        return (
            self.final_dir
            / "manifest.json"
        )


def create_run_paths(
    *,
    output_root: str | Path,
    molecule: DiatomicMolecule,
    run_id: str | None = None,
) -> RunPaths:
    """Create or reopen the standard directory tree for one run."""
    root = Path(
        output_root
    )

    molecule_name = _validate_component(
        molecule.formula,
        label="molecule name",
    )

    resolved_run_id = _validate_component(
        (
            new_run_id()
            if run_id is None
            else run_id
        ),
        label="run_id",
    )

    molecule_dir = (
        root / molecule_name
    )

    run_dir = (
        molecule_dir
        / resolved_run_id
    )

    fast_grid_dir = (
        run_dir
        / "01_fast_grid"
    )

    fast_analysis_dir = (
        run_dir
        / "02_fast_analysis"
    )

    qzvpd_dir = (
        run_dir
        / "03_qzvpd"
    )

    final_dir = (
        run_dir
        / "04_final"
    )

    logs_dir = (
        run_dir
        / "logs"
    )

    for directory in (
        fast_grid_dir,
        fast_analysis_dir,
        qzvpd_dir,
        final_dir,
        logs_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return RunPaths(
        output_root=root,
        molecule_dir=molecule_dir,
        run_dir=run_dir,
        run_id=resolved_run_id,
        fast_grid_dir=fast_grid_dir,
        fast_analysis_dir=fast_analysis_dir,
        qzvpd_dir=qzvpd_dir,
        final_dir=final_dir,
        logs_dir=logs_dir,
    )