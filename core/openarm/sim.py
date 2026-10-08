"""Build the OpenArm MuJoCo backend from the vendored simulator.

The physics, camera and IK glue lives in :mod:`core.openarm.backend` -- a trimmed copy of
the ``embodied_ai-llm`` OpenArm runtime (see ``backends/NOTICE.md``). The robot model,
pedestal, meshes and the official ``openarm_control`` inverse kinematics come from the
``openarm-mujoco`` / ``openarm-control`` wheels installed by
``bash scripts/setup.sh base --openarm``.

Everything needed for the default cube tabletop ships with the repository. The
``insert_flower`` scene references VLABench meshes that are not redistributable here, so
that scene additionally requires a one-time fetch:
``bash scripts/model/download_vlabench_assets.sh``.

The vendored modules import only ``mujoco`` / ``openarm_control`` / ``openarm_mujoco`` (all
three measured to run on Show-Harness's ``numpy<2`` pin), so the upstream project's
FastAPI/pydantic/placo stack and its ``numpy>=2`` requirement never enter this venv.
"""
from __future__ import annotations

from typing import Any

from core.openarm.backend.scenes import (
    OPENARM_SCENES as _SCENE_FILES,
    is_vlabench_assets_present,
    scene_file,
    vlabench_assets_dir,
)

#: Scenes the vendored simulator ships. ``None`` is the three-cube tabletop;
#: ``insert_flower`` adds the VLABench vase + three roses (needs downloaded meshes).
OPENARM_SCENES = tuple(_SCENE_FILES)

__all__ = [
    "OPENARM_SCENES",
    "OpenArmDependencyError",
    "import_openarm_backend",
    "is_vlabench_assets_present",
    "make_openarm_backend",
    "scene_file",
    "vlabench_assets_dir",
]


class OpenArmDependencyError(RuntimeError):
    """``mujoco`` / ``openarm_control`` / ``openarm_mujoco`` is not importable here."""


def import_openarm_backend() -> Any:
    """Import and return the vendored ``SimulationBackend`` class.

    Raises :class:`OpenArmDependencyError` with the install hint when the simulation
    wheels are missing from this interpreter.
    """
    try:
        from core.openarm.backend.simulation import (  # noqa: PLC0415
            SimulationBackend,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        missing = exc.name or "a dependency"
        raise OpenArmDependencyError(
            f"Cannot import the OpenArm simulation: missing {missing!r}. Install the "
            "simulator stack into this venv with:\n"
            "  bash scripts/setup.sh base --openarm\n"
            "(or: uv pip install --python .venv/bin/python mujoco openarm-control "
            "openarm-mujoco pyopengl)"
        ) from exc
    return SimulationBackend


def make_openarm_backend(
    *,
    scene: str | None = None,
    control_hz: float = 60.0,
    cameras: tuple[str, ...] | None = None,
) -> Any:
    """Construct one OpenArm MuJoCo backend (physics + cameras), started and holding.

    ``scene`` is one of :data:`OPENARM_SCENES`; an unknown name raises with the list, so
    a typo in the config fails at startup instead of silently loading the cube table.
    """
    _ensure_render_backend()
    if scene not in OPENARM_SCENES:
        raise ValueError(
            f"Unknown OpenArm scene {scene!r}; available: "
            f"{[s for s in OPENARM_SCENES if s] or ['(default cubes)']}"
        )
    if scene is not None and not is_vlabench_assets_present():
        raise FileNotFoundError(
            f"Scene {scene!r} needs the VLABench meshes, which are not bundled with this "
            "repository. Fetch them once (pinned revision, sha256-verified) with:\n"
            "  bash scripts/model/download_vlabench_assets.sh\n"
            f"They install under {vlabench_assets_dir()}."
        )
    backend_cls = import_openarm_backend()
    kwargs: dict[str, Any] = {"control_hz": float(control_hz), "scene": scene}
    if cameras is not None:
        kwargs["cameras"] = tuple(cameras)
    backend = backend_cls(**kwargs)
    backend.start()
    return backend


def _ensure_render_backend() -> None:
    """Select an offscreen GL backend before MuJoCo imports it.

    MuJoCo resolves its GL backend from ``MUJOCO_GL`` when the FIRST renderer is created, so
    this must run before any capture. Without it MuJoCo takes the interactive ``glfw`` path,
    which cannot create a context on a headless machine -- the failure surfaces later as a
    GLFW error and a blank render rather than at startup. ``egl`` is the headless default
    here, matching ``core.sim.maniskill_task``.
    Set ``SHOWHARNESS_OPENARM_GL`` (or ``MUJOCO_GL``) to override, e.g. ``osmesa`` on a box
    with no EGL device.
    """
    from core.openarm.backend._gl import ensure_render_backend

    ensure_render_backend()
