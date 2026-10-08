"""Shared setup and command conversion for the official openarm_control IK.

Vendored from ``embodied_ai-llm`` (``src/robot_runtime/robots/openarm/kinematics.py``,
Apache-2.0, Enactic Inc.). See ``NOTICE.md`` in this directory.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from openarm_control import ArmSetup, IKParams, Kinematics


def create_kinematics(
    xml: str, control_hz: float = 60, config: Mapping[str, Any] | None = None,
) -> tuple[ArmSetup, Kinematics]:
    config = config or {}
    setup = ArmSetup.from_args(
        xml=str(xml),
        mode="bimanual",
        frame_right="right_ee_control_point",
        frame_type_right="site",
        frame_left="left_ee_control_point",
        frame_type_left="site",
        keyframe="home",
        origin_frame="arm_origin",
        origin_frame_type="site",
    )
    params = IKParams(
        dt=1 / control_hz,
        max_iters=int(config.get("ik_max_iters", 5)),
        position_cost=float(config.get("ik_position_cost", 12)),
        orientation_cost=float(config.get("ik_orientation_cost", 1.5)),
        frame_position_error_limit=float(config.get("ik_frame_position_error_limit", 0.02)),
        frame_orientation_error_limit=float(config.get("ik_frame_orientation_error_limit", 0.25)),
        nullspace_cost=float(config.get("ik_nullspace_cost", 8.5)),
        kinetic_energy_cost=float(config.get("ik_kinetic_energy_cost", 2e-5)),
    )
    return setup, Kinematics(setup, params)


def solve_bimanual(
    kin: Kinematics, measured, right, left, grippers, *,
    dtype=float, error="Inverse kinematics failed",
):
    """Solve against measured joints, leaving command dispatch to each backend."""
    kin.sync(measured)
    kin.update_measured_state(measured)
    kin.set_target("right", np.asarray(right, dtype=dtype))
    kin.set_target("left", np.asarray(left, dtype=dtype))
    kin.set_gripper("right", float(grippers["right"]))
    kin.set_gripper("left", float(grippers["left"]))
    solved = kin.solve()
    if solved is None or not np.isfinite(solved).all():
        raise RuntimeError(error)
    return solved
