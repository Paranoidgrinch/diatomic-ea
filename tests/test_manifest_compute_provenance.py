"""Tests for compute provenance in reproducibility manifests."""

from diatomic_ea.manifest import (
    build_reproducibility_manifest,
)
from diatomic_ea.schema_f_statistics import (
    estimate_schema_f_from_values,
)


def estimate():
    return estimate_schema_f_from_values(
        molecule="AlO",
        functional_eas_ev={
            "PBE": 1.0,
            "B3LYP": 1.1,
            "PBE0": 1.2,
            "TPSSh": 1.3,
        },
    )


def test_manifest_preserves_compute_provenance() -> None:
    provenance = {
        "backend": "wsl",
        "ready": True,
        "compute": {
            "distribution": (
                "Ubuntu-24.04"
            ),
            "worker_wheel_sha256": (
                "a" * 64
            ),
        },
        "compatibility": {
            "verified": True,
        },
    }

    manifest = (
        build_reproducibility_manifest(
            estimate=estimate(),
            fast_grid_task_count=10,
            qzvpd_task_count=20,
            compute_provenance=(
                provenance
            ),
        )
    )

    assert (
        manifest[
            "compute_provenance"
        ]
        == provenance
    )


def test_manifest_allows_missing_compute_provenance() -> None:
    manifest = (
        build_reproducibility_manifest(
            estimate=estimate(),
            fast_grid_task_count=10,
            qzvpd_task_count=20,
        )
    )

    assert (
        manifest[
            "compute_provenance"
        ]
        is None
    )
