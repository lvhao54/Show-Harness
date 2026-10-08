"""Select MuJoCo's offscreen GL backend before anything imports ``mujoco``.

MuJoCo reads ``MUJOCO_GL`` when it first creates a GL context, so the value has to be in
the environment before the first render. On a headless box the interactive ``glfw``
backend cannot create a context and the failure surfaces much later as a blank render plus
a GLFW warning, so every module in this package that imports MuJoCo calls
:func:`ensure_render_backend` first.

``egl`` is the headless default and matches ``core.sim.maniskill_task``. Override with
``SHOWHARNESS_OPENARM_GL`` (or ``MUJOCO_GL``), e.g. ``osmesa`` on a box with no EGL device.
"""
from __future__ import annotations

import os

_APPLIED = False


def ensure_render_backend() -> None:
    """Set ``MUJOCO_GL`` (idempotent) -- call before the first ``import mujoco``."""
    global _APPLIED
    if _APPLIED:
        return
    os.environ.setdefault("MUJOCO_GL", os.environ.get("SHOWHARNESS_OPENARM_GL", "egl"))
    _APPLIED = True
