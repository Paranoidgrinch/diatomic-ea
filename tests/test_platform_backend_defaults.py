"""Tests that production orchestration uses platform backends."""

import inspect

from diatomic_ea.electron_count_adapter import (
    run_platform_electron_count,
)
from diatomic_ea.executor import (
    execute_fast_grid,
)
from diatomic_ea.pipeline import (
    run_schema_f_pipeline,
)
from diatomic_ea.runner import (
    execute_fast_grid_resumable,
    execute_qzvpd_resumable,
)
from diatomic_ea.single_point_adapter import (
    run_platform_single_point,
)


def test_pipeline_defaults_to_platform_backends() -> None:
    parameters = inspect.signature(
        run_schema_f_pipeline
    ).parameters

    assert (
        parameters[
            "electron_count_resolver"
        ].default
        is run_platform_electron_count
    )

    assert (
        parameters[
            "worker"
        ].default
        is run_platform_single_point
    )


def test_fast_grid_runner_defaults_to_platform_worker() -> None:
    parameters = inspect.signature(
        execute_fast_grid_resumable
    ).parameters

    assert (
        parameters[
            "worker"
        ].default
        is run_platform_single_point
    )


def test_qzvpd_runner_defaults_to_platform_worker() -> None:
    parameters = inspect.signature(
        execute_qzvpd_resumable
    ).parameters

    assert (
        parameters[
            "worker"
        ].default
        is run_platform_single_point
    )


def test_executor_fast_grid_uses_platform_worker() -> None:
    source = inspect.getsource(
        execute_fast_grid
    )

    assert (
        "worker=run_platform_single_point"
        in source
    )

    assert (
        "worker=run_pyscf_single_point"
        not in source
    )
