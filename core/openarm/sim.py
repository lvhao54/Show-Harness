"""Locate and construct the OpenArm MuJoCo backend from an ``embodied_ai-llm`` checkout.

The OpenArm model, IK and camera stack are not vendored into Show-Harness. They already
exist, verified on hardware and in sim, in ``embodied_ai-llm``
(``src/robot_runtime/robots/openarm/``), so this module imports them from that checkout
exactly as :mod:`core.sim.robolab_task` imports RoboLab and
:mod:`core.sim.maniskill_scenes` imports RLinf.

Resolution order for the checkout (first hit wins):

1. ``openarm_root`` passed by the caller (the robot config's ``openarm_root:`` key),
2. ``OPENARM_ROOT`` in the environment,
3. the common sibling checkouts of this repo:
   ``../embodied_ai-llm``, ``~/embodied_ai-llm``, ``~/ws/luke/embodied_ai-llm``.

Step 3 exists because the two projects are normally checked out side by side; it keeps the
usual layout working with no configuration, and resolution still fails loudly (listing
every path it tried) rather than silently falling back to a different world.

Only the simulation modules are imported: ``robot_runtime.robots.openarm.simulation``
needs ``mujoco`` + ``openarm_control`` + ``openarm_mujoco`` and nothing else from that
project, so the runtime's FastAPI/pydantic/placo stack is never loaded -- and, importantly,
its ``numpy>=2`` pin does not apply. Show-Harness's ``numpy<2`` is satisfied by the same
``mujoco``/``openarm_control`` wheels (measured), which is why both live in one venv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

#: Checkout locations tried, in order, when nothing overrides the path.
def _default_openarm_roots() -> tuple[Path, ...]:
    here = Path(__file__).resolve()
    # .../Show-Harness/core/openarm/sim.py -> the repo root, then its parent.
    repo_root = here.parents[2]
    candidates = (
        repo_root.parent / "embodied_ai-llm",
        Path.home() / "embodied_ai-llm",
        Path.home() / "ws" / "luke" / "embodied_ai-llm",
    )
    # Preserve order, drop duplicates (the sibling guess often IS ~/ws/luke/...).
    return tuple(dict.fromkeys(p.resolve() for p in candidates))


#: Kept as a name for callers that only need a representative path (the first guess).
DEFAULT_OPENARM_ROOT = _default_openarm_roots()[0]

#: Scenes the upstream checkout ships. ``None`` is its three-cube tabletop;
#: ``insert_flower`` adds the VLABench vase + three roses.
OPENARM_SCENES = (None, "insert_flower")


class OpenArmCheckoutError(RuntimeError):
    """The ``embodied_ai-llm`` checkout (or one of its sim dependencies) is missing."""


def resolve_openarm_root(openarm_root: str | os.PathLike[str] | None = None) -> Path:
    """The ``embodied_ai-llm`` checkout to import the OpenArm sim from."""
    if openarm_root:
        return Path(openarm_root).expanduser()
    env = os.environ.get("OPENARM_ROOT")
    if env:
        return Path(env).expanduser()
    candidates = _default_openarm_roots()
    for candidate in candidates:
        if _has_openarm_sim(candidate):
            return candidate
    # Nothing found: return the first guess so the error message is concrete.
    return candidates[0]


def _has_openarm_sim(root: Path) -> bool:
    return (root / "src" / "robot_runtime" / "robots" / "openarm" / "simulation.py").is_file()


def _ensure_importable(root: Path) -> None:
    """Put the checkout's ``src/`` on ``sys.path`` so ``robot_runtime`` resolves."""
    src = root / "src"
    if not _has_openarm_sim(root):
        tried = "\n".join(f"  - {p}" for p in _default_openarm_roots())
        raise OpenArmCheckoutError(
            f"No OpenArm simulation in {root!r}: expected "
            f"{src / 'robot_runtime/robots/openarm/simulation.py'}. Point OPENARM_ROOT "
            "(or the robot config's openarm_root:) at an embodied_ai-llm checkout.\n"
            f"Default locations tried:\n{tried}"
        )
    entry = str(src)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def import_openarm_backend(root: Path | None = None) -> Any:
    """Import and return the upstream ``SimulationBackend`` class.

    Raises :class:`OpenArmCheckoutError` with the install hint when ``mujoco`` /
    ``openarm_control`` / ``openarm_mujoco`` are not importable in this interpreter.
    """
    resolved = resolve_openarm_root(root)
    _ensure_importable(resolved)
    try:
        from robot_runtime.robots.openarm.simulation import (  # noqa: PLC0415
            SimulationBackend,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        missing = exc.name or "a dependency"
        raise OpenArmCheckoutError(
            f"Cannot import the OpenArm simulation from {resolved!r}: missing {missing!r}. "
            "Install it into this venv with:\n"
            "  uv pip install --python .venv/bin/python mujoco openarm-control openarm-mujoco\n"
            "(the checkout itself is imported from src/, not installed)."
        ) from exc
    return SimulationBackend


def make_openarm_backend(
    *,
    openarm_root: str | os.PathLike[str] | None = None,
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
    backend_cls = import_openarm_backend(openarm_root)
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
    preferred = os.environ.get("SHOWHARNESS_OPENARM_GL", "egl")
    os.environ.setdefault("MUJOCO_GL", preferred)
