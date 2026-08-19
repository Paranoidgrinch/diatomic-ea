"""Qt-independent state helpers for the DiatomicEA GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.progress_metrics import (
    format_duration,
)


@dataclass(frozen=True, slots=True)
class ProductionStatusSnapshot:
    """GUI-friendly view of production_status.json."""

    state: str
    stage: str | None
    completed: int | None
    total: int | None
    percent: float | None
    tasks_per_second: float | None
    eta_seconds: float | None
    stage_elapsed_seconds: float | None
    message: str
    updated_at_utc: str | None
    source_path: str | None = None

    @property
    def progress_percent(self) -> float:
        if self.percent is None:
            return 0.0

        return max(
            0.0,
            min(
                100.0,
                self.percent,
            ),
        )

    @property
    def completed_text(self) -> str:
        if (
            self.completed is None
            or self.total is None
        ):
            if (
                self.state.casefold()
                == "completed"
            ):
                return "Complete"

            return "-- / --"

        return (
            f"{self.completed:,}"
            " / "
            f"{self.total:,}"
        )

    @property
    def rate_text(self) -> str:
        if self.tasks_per_second is None:
            return "-- tasks/s"

        return (
            f"{self.tasks_per_second:.2f}"
            " tasks/s"
        )

    @property
    def eta_text(self) -> str:
        return format_duration(
            self.eta_seconds
        )

    @property
    def elapsed_text(self) -> str:
        return format_duration(
            self.stage_elapsed_seconds
        )

    @property
    def stage_text(self) -> str:
        if not self.stage:
            return "Idle"

        words = (
            self.stage
            .replace(
                "_",
                "-",
            )
            .split(
                "-"
            )
        )

        return " ".join(
            (
                "QZVPD"
                if word.casefold()
                == "qzvpd"
                else word.title()
            )
            for word in words
        )


def _optional_int(
    value: object,
) -> int | None:
    if value is None:
        return None

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def _optional_float(
    value: object,
) -> float | None:
    if value is None:
        return None

    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def production_status_from_mapping(
    payload: dict[str, object],
    *,
    source_path: str | None = None,
) -> ProductionStatusSnapshot:
    """Normalize persisted telemetry for presentation."""

    stage_value = payload.get(
        "stage"
    )

    updated_value = payload.get(
        "updated_at_utc"
    )

    return ProductionStatusSnapshot(
        state=str(
            payload.get(
                "state",
                "unknown",
            )
        ),
        stage=(
            None
            if stage_value is None
            else str(
                stage_value
            )
        ),
        completed=_optional_int(
            payload.get(
                "completed"
            )
        ),
        total=_optional_int(
            payload.get(
                "total"
            )
        ),
        percent=_optional_float(
            payload.get(
                "percent"
            )
        ),
        tasks_per_second=_optional_float(
            payload.get(
                "tasks_per_second"
            )
        ),
        eta_seconds=_optional_float(
            payload.get(
                "eta_seconds"
            )
        ),
        stage_elapsed_seconds=_optional_float(
            payload.get(
                "stage_elapsed_seconds"
            )
        ),
        message=str(
            payload.get(
                "message",
                "",
            )
        ),
        updated_at_utc=(
            None
            if updated_value is None
            else str(
                updated_value
            )
        ),
        source_path=source_path,
    )


def read_production_status(
    path: str | Path,
) -> ProductionStatusSnapshot:
    """Read one atomically written production status file."""

    source = Path(
        path
    )

    try:
        payload = json.loads(
            source.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Could not read production status: "
            f"{source}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Production status must contain "
            "a JSON object."
        )

    return production_status_from_mapping(
        payload,
        source_path=str(
            source.resolve()
        ),
    )


def discover_latest_status(
    output_root: str | Path,
) -> Path | None:
    """Find the most recently modified production status."""

    root = Path(
        output_root
    )

    if not root.is_dir():
        return None

    candidates = [
        path
        for path
        in root.glob(
            "*/*/logs/production_status.json"
        )
        if path.is_file()
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime_ns
        ),
    )
