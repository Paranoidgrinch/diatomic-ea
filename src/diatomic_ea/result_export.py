"""Final user-facing Schema F result export."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.schema_f_statistics import SchemaFEstimate


FINAL_RESULT_COLUMNS = (
    "molecule",
    "model_id",
    "n_functionals",
    "ea_pbe_eV",
    "ea_b3lyp_eV",
    "ea_pbe0_eV",
    "ea_tpssh_eV",
    "median_qz_eV",
    "half_range_qz_eV",
    "bias_correction_eV",
    "predicted_ea_eV",
    "scale_eV",
    "pi80_half_width_eV",
    "pi80_lower_eV",
    "pi80_upper_eV",
    "pi90_half_width_eV",
    "pi90_lower_eV",
    "pi90_upper_eV",
    "pi95_half_width_eV",
    "pi95_lower_eV",
    "pi95_upper_eV",
)


@dataclass(frozen=True, slots=True)
class FinalSchemaFRecord:
    """One complete final Schema F result."""

    molecule: str
    model_id: str
    n_functionals: int
    ea_pbe_ev: float
    ea_b3lyp_ev: float
    ea_pbe0_ev: float
    ea_tpssh_ev: float
    median_qz_ev: float
    half_range_qz_ev: float
    bias_correction_ev: float
    predicted_ea_ev: float
    scale_ev: float
    pi80_half_width_ev: float
    pi80_lower_ev: float
    pi80_upper_ev: float
    pi90_half_width_ev: float
    pi90_lower_ev: float
    pi90_upper_ev: float
    pi95_half_width_ev: float
    pi95_lower_ev: float
    pi95_upper_ev: float


def final_record_from_estimate(
    estimate: SchemaFEstimate,
) -> FinalSchemaFRecord:
    """Convert a Schema F estimate into the final table record."""
    functional_values = dict(
        estimate.functional_eas_ev
    )

    pi80 = estimate.interval(80)
    pi90 = estimate.interval(90)
    pi95 = estimate.interval(95)

    return FinalSchemaFRecord(
        molecule=estimate.molecule,
        model_id=estimate.model_id,
        n_functionals=estimate.functional_count,
        ea_pbe_ev=functional_values["PBE"],
        ea_b3lyp_ev=functional_values["B3LYP"],
        ea_pbe0_ev=functional_values["PBE0"],
        ea_tpssh_ev=functional_values["TPSSh"],
        median_qz_ev=estimate.median_qz_ev,
        half_range_qz_ev=estimate.half_range_qz_ev,
        bias_correction_ev=estimate.bias_correction_ev,
        predicted_ea_ev=estimate.predicted_ea_ev,
        scale_ev=estimate.scale_ev,
        pi80_half_width_ev=pi80.half_width_ev,
        pi80_lower_ev=pi80.lower_ev,
        pi80_upper_ev=pi80.upper_ev,
        pi90_half_width_ev=pi90.half_width_ev,
        pi90_lower_ev=pi90.lower_ev,
        pi90_upper_ev=pi90.upper_ev,
        pi95_half_width_ev=pi95.half_width_ev,
        pi95_lower_ev=pi95.lower_ev,
        pi95_upper_ev=pi95.upper_ev,
    )


def final_record_row(
    record: FinalSchemaFRecord,
) -> dict[str, object]:
    """Convert a final result record to a CSV-compatible mapping."""
    return {
        "molecule": record.molecule,
        "model_id": record.model_id,
        "n_functionals": record.n_functionals,
        "ea_pbe_eV": record.ea_pbe_ev,
        "ea_b3lyp_eV": record.ea_b3lyp_ev,
        "ea_pbe0_eV": record.ea_pbe0_ev,
        "ea_tpssh_eV": record.ea_tpssh_ev,
        "median_qz_eV": record.median_qz_ev,
        "half_range_qz_eV": record.half_range_qz_ev,
        "bias_correction_eV": record.bias_correction_ev,
        "predicted_ea_eV": record.predicted_ea_ev,
        "scale_eV": record.scale_ev,
        "pi80_half_width_eV": record.pi80_half_width_ev,
        "pi80_lower_eV": record.pi80_lower_ev,
        "pi80_upper_eV": record.pi80_upper_ev,
        "pi90_half_width_eV": record.pi90_half_width_ev,
        "pi90_lower_eV": record.pi90_lower_ev,
        "pi90_upper_eV": record.pi90_upper_ev,
        "pi95_half_width_eV": record.pi95_half_width_ev,
        "pi95_lower_eV": record.pi95_lower_ev,
        "pi95_upper_eV": record.pi95_upper_ev,
    }


def write_final_result_csv(
    path: str | Path,
    record: FinalSchemaFRecord,
) -> Path:
    """Atomically write one final Schema F result CSV."""
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
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(
                handle.name
            )

            writer = csv.DictWriter(
                handle,
                fieldnames=FINAL_RESULT_COLUMNS,
            )

            writer.writeheader()

            writer.writerow(
                final_record_row(
                    record
                )
            )

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