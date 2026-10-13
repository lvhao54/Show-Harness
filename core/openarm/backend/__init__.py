"""Vendored OpenArm MuJoCo backend.

Self-contained copy of the OpenArm simulation modules Show-Harness runs, so deployments
no longer need an ``embodied_ai-llm`` checkout. See ``NOTICE.md`` for provenance and what
was trimmed.

The arm model, pedestal, meshes and the official IK all come from the ``openarm-mujoco`` /
``openarm-control`` wheels (``requirements/requirements-openarm.txt``); only the runtime
glue lives here.
"""
from __future__ import annotations

from typing import Any

from core.openarm.backend.scenes import is_vlabench_assets_present, vlabench_assets_dir

__all__ = [
    "SimulationBackend",
    "is_vlabench_assets_present",
    "vlabench_assets_dir",
]


def __getattr__(name: str) -> Any:
    """Import the MuJoCo-backed simulation lazily.

    ``scenes`` is pure path logic, and every submodule import runs this package first, so
    an eager ``simulation`` import would drag ``mujoco`` / ``openarm_control`` in for a
    scene lookup -- and turn :func:`core.openarm.sim.import_openarm_backend`'s install
    hint into a bare ``ModuleNotFoundError`` on a venv without the wheels.
    """
    if name == "SimulationBackend":
        from core.openarm.backend.simulation import SimulationBackend  # noqa: PLC0415

        return SimulationBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
