"""Backend state dataclasses for the vendored OpenArm MuJoCo backend.

Vendored from ``embodied_ai-llm`` (``src/robot_runtime/control/backend.py``,
Apache-2.0, Enactic Inc.); only the state dataclass the simulator backend
returns is kept -- the hardware-driver protocols and the entry-point loader are
not used by the simulation path. See ``NOTICE.md`` in this directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendState:
    """One measured snapshot, in the ``arm_origin`` site frame."""

    right: list[float]  # [x,y,z,qw,qx,qy,qz], metres, arm_origin
    left: list[float]
    joints_right: list[float]  # seven joints then gripper, radians
    joints_left: list[float]
    source_time: float  # simulation time or hardware acquisition time
    healthy: bool = True
    fault: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "end_effector_pose": {"right": list(self.right), "left": list(self.left)},
            "joint_state": {
                "right": list(self.joints_right),
                "left": list(self.joints_left),
            },
            "source_time": self.source_time,
            "healthy": self.healthy,
            "fault": self.fault,
        }
