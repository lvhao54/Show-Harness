"""Vendored OpenArm MuJoCo backend.

Self-contained copy of the OpenArm simulation modules Show-Harness runs, so deployments
no longer need an ``embodied_ai-llm`` checkout. See ``NOTICE.md`` for provenance and what
was trimmed.

The arm model, pedestal, meshes and the official IK all come from the ``openarm-mujoco`` /
``openarm-control`` wheels (``requirements/requirements-openarm.txt``); only the runtime
glue lives here.
"""
from __future__ import annotations

from core.openarm.backend.scenes import is_vlabench_assets_present, vlabench_assets_dir
from core.openarm.backend.simulation import SimulationBackend

__all__ = [
    "SimulationBackend",
    "is_vlabench_assets_present",
    "vlabench_assets_dir",
]
