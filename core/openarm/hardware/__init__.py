"""OpenArm2 Pedestal hardware layer: CAN dispatch, cameras and the robot duck type.

The simulator lives in :mod:`core.openarm.backend`; this package is the real-rig
counterpart. It vendors the thread-owned CAN send strategy from ``embodied_ai-llm``
(Apache-2.0, see ``NOTICE.md``) and exposes the same four-method robot contract the
simulator does, so the atomic controller, plugins and runners are literally the same code
on both backends.

Importing this package opens no devices: construction validates configuration only, and
``start()`` (called by the session) is what touches CAN interfaces, motors and cameras.
The heavy modules are imported lazily, so importing this package in an environment without
the vendor wheels still works and only the accessor that needs them raises.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "OpenArmDependencyError",
    "OpenArmPedestalBackend",
    "OpenArmRealRobot",
    "make_pedestal_backend",
    "require_openarm_driver",
]


class OpenArmDependencyError(RuntimeError):
    """The OpenArm hardware driver stack (or OpenCV) is not importable here."""


def require_openarm_driver() -> None:
    """Raise with an install hint when the hardware driver stack is missing."""
    try:
        import openarm_driver  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise OpenArmDependencyError(
            "Cannot import the OpenArm hardware driver: missing "
            f"{exc.name or 'openarm_driver'!r}. Install the real-rig stack into this "
            "venv with:\n"
            "  bash scripts/setup.sh base --openarm-real"
        ) from exc


def __getattr__(name: str) -> Any:
    """Lazily resolve the backend/robot classes (and their own missing-wheel errors)."""
    if name == "OpenArmPedestalBackend":
        from core.openarm.hardware.backend import (  # noqa: PLC0415
            OpenArmPedestalBackend,
        )

        return OpenArmPedestalBackend
    if name == "make_pedestal_backend":
        from core.openarm.hardware.backend import make_pedestal_backend  # noqa: PLC0415

        return make_pedestal_backend
    if name == "OpenArmRealRobot":
        from core.openarm.hardware.robot import OpenArmRealRobot  # noqa: PLC0415

        return OpenArmRealRobot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
