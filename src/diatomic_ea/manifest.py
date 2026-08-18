"""Reproducibility manifests for DiatomicEA calculations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

from diatomic_ea import __version__
from diatomic_ea.schema_f import SCHEMA_F
from diatomic_ea.schema_f_statistics import (
    SCHEMA_F_CALIBRATION,
    SchemaFEstimate,
)


def sha256_file(
    path: str | Path,
) -> str:
    """Return the SHA-256 digest of one file."""
    source = Path(path)

    digest = hashlib.sha256()

    with source.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def _package_version(
    distribution_name: str,
) -> str | None:
    try:
        return importlib.metadata.version(
            distribution_name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        value = result.stdout.strip()

        return value or None

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        return bool(
            result.stdout.strip()
        )

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None


def _file_entry(
    path: str | Path | None,
) -> dict[str, object] | None:
    if path is None:
        return None

    source = Path(path)

    if not source.exists():
        return {
            "path": str(source),
            "exists": False,
            "sha256": None,
            "size_bytes": None,
        }

    return {
        "path": str(source),
        "exists": True,
        "sha256": sha256_file(
            source
        ),
        "size_bytes": source.stat().st_size,
    }


def build_reproducibility_manifest(
    *,
    estimate: SchemaFEstimate,
    fast_grid_task_count: int,
    qzvpd_task_count: int,
    fast_grid_raw_csv: str | Path | None = None,
    qzvpd_raw_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build a complete reproducibility record."""
    if fast_grid_task_count < 0:
        raise ValueError(
            "fast_grid_task_count must not be negative."
        )

    if qzvpd_task_count < 0:
        raise ValueError(
            "qzvpd_task_count must not be negative."
        )

    intervals = {
        str(interval.confidence_percent): {
            "conformal_quantile": (
                interval.conformal_quantile
            ),
            "half_width_eV": (
                interval.half_width_ev
            ),
            "lower_eV": interval.lower_ev,
            "upper_eV": interval.upper_ev,
        }
        for interval in estimate.intervals
    }

    return {
        "manifest_version": 1,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "application": {
            "name": "DiatomicEA",
            "version": __version__,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        },
        "environment": {
            "python_version": (
                platform.python_version()
            ),
            "python_implementation": (
                platform.python_implementation()
            ),
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "pyscf_version": (
                _package_version("pyscf")
            ),
            "basis_set_exchange_version": (
                _package_version(
                    "basis-set-exchange"
                )
            ),
        },
        "workflow": {
            "schema_id": SCHEMA_F.schema_id,
            "reference_pyscf_version": (
                SCHEMA_F.reference_pyscf_version
            ),
            "electronic_structure_method": (
                SCHEMA_F
                .electronic_structure_method
            ),
            "functionals": list(
                SCHEMA_F.functionals
            ),
            "fast_bases": list(
                SCHEMA_F.fast_bases
            ),
            "fast_grid": {
                "step_angstrom": (
                    SCHEMA_F
                    .fast_grid
                    .step_angstrom
                ),
                "grid_level": (
                    SCHEMA_F
                    .fast_grid
                    .grid_level
                ),
                "conv_tol": (
                    SCHEMA_F
                    .fast_grid
                    .conv_tol
                ),
                "max_cycle": (
                    SCHEMA_F
                    .fast_grid
                    .max_cycle
                ),
                "max_memory_mb": (
                    SCHEMA_F
                    .fast_grid
                    .max_memory_mb
                ),
            },
            "qzvpd_refinement": {
                "basis": (
                    SCHEMA_F.refinement.basis
                ),
                "window_angstrom": (
                    SCHEMA_F
                    .refinement
                    .window_angstrom
                ),
                "step_angstrom": (
                    SCHEMA_F
                    .refinement
                    .grid
                    .step_angstrom
                ),
                "grid_level": (
                    SCHEMA_F
                    .refinement
                    .grid
                    .grid_level
                ),
                "conv_tol": (
                    SCHEMA_F
                    .refinement
                    .grid
                    .conv_tol
                ),
                "max_cycle": (
                    SCHEMA_F
                    .refinement
                    .grid
                    .max_cycle
                ),
                "max_memory_mb": (
                    SCHEMA_F
                    .refinement
                    .grid
                    .max_memory_mb
                ),
                "max_spins_per_charge": (
                    SCHEMA_F
                    .refinement
                    .max_spins_per_charge
                ),
            },
            "scf_rescue": {
                "level_shift": (
                    SCHEMA_F
                    .scf_rescue
                    .level_shift
                ),
                "newton_max_cycle": (
                    SCHEMA_F
                    .scf_rescue
                    .newton_max_cycle
                ),
            },
        },
        "schema_f_calibration": {
            "model_id": (
                SCHEMA_F_CALIBRATION.model_id
            ),
            "central_slope": (
                SCHEMA_F_CALIBRATION
                .central_slope
            ),
            "bias_correction_eV": (
                SCHEMA_F_CALIBRATION
                .bias_correction_ev
            ),
            "scale_intercept": (
                SCHEMA_F_CALIBRATION
                .scale_intercept
            ),
            "scale_half_range_coefficient": (
                SCHEMA_F_CALIBRATION
                .scale_half_range_coefficient
            ),
            "conformal_quantiles": {
                str(confidence): quantile
                for confidence, quantile
                in (
                    SCHEMA_F_CALIBRATION
                    .conformal_quantiles
                )
            },
        },
        "calculation": {
            "molecule": estimate.molecule,
            "fast_grid_task_count": (
                fast_grid_task_count
            ),
            "qzvpd_task_count": (
                qzvpd_task_count
            ),
        },
        "schema_f_result": {
            "functional_eas_eV": {
                functional: value
                for functional, value
                in estimate.functional_eas_ev
            },
            "median_qz_eV": (
                estimate.median_qz_ev
            ),
            "half_range_qz_eV": (
                estimate.half_range_qz_ev
            ),
            "bias_correction_eV": (
                estimate.bias_correction_ev
            ),
            "predicted_ea_eV": (
                estimate.predicted_ea_ev
            ),
            "scale_eV": estimate.scale_ev,
            "prediction_intervals": intervals,
        },
        "raw_files": {
            "fast_grid": _file_entry(
                fast_grid_raw_csv
            ),
            "qzvpd": _file_entry(
                qzvpd_raw_csv
            ),
        },
    }


def write_reproducibility_manifest(
    path: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """Atomically write a reproducibility manifest."""
    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(
                handle.name
            )

            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
            )

            handle.write("\n")
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

        raise

    return destination