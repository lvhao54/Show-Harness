"""Shared setup and command conversion for the official openarm_control IK.

Vendored from ``embodied_ai-llm`` (``src/robot_runtime/robots/openarm/kinematics.py``,
Apache-2.0, Enactic Inc.). See ``NOTICE.md`` in this directory.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import mink
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
        max_iters=int(config.get("ik_max_iters", 10)),
        position_cost=float(config.get("ik_position_cost", 12)),
        orientation_cost=float(config.get("ik_orientation_cost", 12)),
        frame_position_error_limit=float(config.get("ik_frame_position_error_limit", 0.02)),
        frame_orientation_error_limit=float(config.get("ik_frame_orientation_error_limit", 0.25)),
        nullspace_cost=float(config.get("ik_nullspace_cost", 2)),
        kinetic_energy_cost=float(config.get("ik_kinetic_energy_cost", 2e-5)),
    )
    kin = Kinematics(setup, params)
    # The stock solver bounds joint limits/singularities but has no collision limit.
    # Reaching the centre of this table otherwise drives link4 into the pedestal:
    # the wrist loses height/orientation while the solver keeps accepting commands.
    # Keep this robot-only constraint out of task geometry (grasp contacts stay legal).
    # openarm-control 0.4.0 exposes no public limit hook; isolate its private seam here.
    solver = getattr(kin, "_ik", None)
    if solver is None or not isinstance(getattr(solver, "_limits", None), list):
        raise RuntimeError("Unsupported openarm-control IK limit API; use requirements-openarm.txt")
    solver._limits.append(mink.CollisionAvoidanceLimit(
        setup.model,
        [(["openarm_body_link0_collision"],
          ["link4_right_collision_00", "link4_left_collision_00"])],
        minimum_distance_from_collisions=0.006,
        collision_detection_distance=0.07,
        gain=0.5,
    ))
    return setup, kin


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
