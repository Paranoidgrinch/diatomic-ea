"""stdin/stdout worker for one DiatomicEA single-point calculation.

Input:
    One versioned SinglePointTask JSON document on stdin.

Output:
    One versioned SinglePointResult JSON document on stdout.

Diagnostics and protocol errors are written to stderr.
"""

from __future__ import annotations

import sys

from diatomic_ea.single_point import (
    run_pyscf_single_point,
)
from diatomic_ea.single_point_protocol import (
    SinglePointProtocolError,
    dumps_result,
    loads_task,
)


def execute_task_json(
    task_json: str,
) -> str:
    """Execute one serialized single-point task."""
    if not task_json.strip():
        raise SinglePointProtocolError(
            "Single-point worker received empty input."
        )

    task = loads_task(
        task_json
    )

    result = run_pyscf_single_point(
        task
    )

    return dumps_result(
        result
    )


def main() -> int:
    """Read one task from stdin and emit one result to stdout."""
    try:
        task_json = sys.stdin.read()

        result_json = execute_task_json(
            task_json
        )

    except SinglePointProtocolError as exc:
        print(
            "PROTOCOL_ERROR:",
            str(exc),
            file=sys.stderr,
        )

        return 2

    except Exception as exc:
        print(
            "WORKER_ERROR:",
            repr(exc),
            file=sys.stderr,
        )

        return 3

    sys.stdout.write(
        result_json
    )

    sys.stdout.write(
        "\n"
    )

    sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
