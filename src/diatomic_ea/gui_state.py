"""Qt-independent presentation state for the DiatomicEA GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from diatomic_ea.progress_metrics import (
    format_duration,
)


_STAGE_LABELS = {
    "preparation": "Preparing calculation",
    "fast-grid": "Initial geometry scan",
    "fast-grid-analysis": "Geometry analysis",
    "qzvpd-refinement": "High-accuracy refinement",
    "statistical-ea": "EA prediction",
    "export": "Saving results",
}


_MESSAGE_REPLACEMENTS = (
    (
        "Full Schema F production run",
        "Electron-affinity calculation",
    ),
    (
        "complete Schema F calculation",
        "electron-affinity calculation",
    ),
    (
        "Complete Schema F calculation",
        "Electron-affinity calculation",
    ),
    (
        "Schema F statistical estimate",
        "EA prediction",
    ),
    (
        "Schema F calculation",
        "electron-affinity calculation",
    ),
    (
        "evaluating Schema F",
        "calculating the EA prediction",
    ),
    (
        "QZVPD refinement",
        "high-accuracy refinement",
    ),
    (
        "QZVPD candidate",
        "high-accuracy candidate",
    ),
    (
        "QZVPD",
        "high-accuracy",
    ),
    (
        "Fast-grid",
        "Initial geometry scan",
    ),
    (
        "fast-grid",
        "initial geometry scan",
    ),
    (
        "Fast grid",
        "Initial geometry scan",
    ),
    (
        "fast grid",
        "initial geometry scan",
    ),
)


def stage_display_name(
    stage: str | None,
) -> str:
    """Return natural English for an internal calculation stage."""

    if not stage:
        return "Idle"

    normalized = (
        stage.strip()
        .casefold()
        .replace(
            "_",
            "-",
        )
    )

    known = _STAGE_LABELS.get(
        normalized
    )

    if known is not None:
        return known

    return " ".join(
        word.title()
        for word
        in normalized.split(
            "-"
        )
    )


def humanize_status_message(
    message: str,
) -> str:
    """Remove implementation-specific terminology from GUI messages."""

    result = str(
        message
    )

    for (
        internal,
        visible,
    ) in _MESSAGE_REPLACEMENTS:
        result = result.replace(
            internal,
            visible,
        )

    return result


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
        return stage_display_name(
            self.stage
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
    """Normalize persisted calculation telemetry for presentation."""

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
        message=humanize_status_message(
            str(
                payload.get(
                    "message",
                    "",
                )
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
    """Read one atomically written calculation-status file."""

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
            "Could not read calculation status: "
            f"{source}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Calculation status must contain "
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
    """Find the most recently modified calculation status."""

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
