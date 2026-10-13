"""The setpoint leash: a blocked move must not wind force into the setpoint.

Without it the controller integrates every commanded step, so MV_DOWN repeated against a
support drives the target ever deeper while the arm cannot follow. On the OpenArm sim
that stored press slid a held block through the pads and, on RELEASE, snapped the wrist
down so the opening fingers struck the support and toppled the stack.
"""
from __future__ import annotations

import unittest

import numpy as np

from core.config import load_yaml
from interpreters.real_atomic_controller import RealAtomicController


class BlockedRobot:
    """Tracks each setpoint at once (as the OpenArm sim does), but not below ``contact_z``."""

    def __init__(self, contact_z: float) -> None:
        self.contact_z = contact_z
        self.pose = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
        self.width = 0.02
        self.on_gripper = None

    def get_ee_pose(self) -> np.ndarray:
        return self.pose.copy()

    def get_gripper_position(self) -> np.ndarray:
        return np.array([self.width])

    def control_gripper(self, close: bool) -> None:
        if self.on_gripper is not None:
            self.on_gripper()
        self.width = 0.02 if close else 0.08

    def update_desired_ee_pose(self, pose7) -> None:
        pose = np.asarray(pose7, dtype=float).copy()
        pose[2] = max(pose[2], self.contact_z)
        self.pose = pose


def _controller(robot: BlockedRobot, leash) -> RealAtomicController:
    controller = RealAtomicController.from_primitives_config(
        robot,
        load_yaml("configs/primitives_openarm.yaml"),
        step_m=0.02,
        settle_steps=2,
        settle_dt_s=0.0,
        gripper_settle_s=0.0,
        gripper_min_settle_s=0.0,
        setpoint_leash_m=leash,
        verbose=False,
    )
    controller.sync_from_robot()
    return controller


class SetpointLeashTests(unittest.TestCase):
    def test_without_a_leash_blocked_descents_wind_up_the_setpoint(self) -> None:
        robot = BlockedRobot(contact_z=0.17)
        controller = _controller(robot, leash=None)
        for _ in range(4):
            controller.step("MV_DOWN")
        self.assertAlmostEqual(float(robot.pose[2]), 0.17)
        self.assertAlmostEqual(float(controller.target_pose[2]), 0.12)

    def test_leash_holds_the_measured_pose_after_a_blocked_descent(self) -> None:
        robot = BlockedRobot(contact_z=0.17)
        controller = _controller(robot, leash=0.004)
        for _ in range(4):
            result = controller.step("MV_DOWN")
            self.assertGreaterEqual(float(controller.target_pose[2]), 0.17 - 1e-9)
        self.assertIn("setpoint leash", result.note)

    def test_leash_leaves_free_motion_alone(self) -> None:
        robot = BlockedRobot(contact_z=0.0)
        controller = _controller(robot, leash=0.004)
        result = controller.step("MV_DOWN")
        self.assertAlmostEqual(float(controller.target_pose[2]), 0.18)
        self.assertEqual(result.note, "")

    def test_release_lets_go_from_the_measured_pose(self) -> None:
        robot = BlockedRobot(contact_z=0.17)
        controller = _controller(robot, leash=0.004)
        controller.step("GRASP")
        controller.step("MV_DOWN")
        controller.step("MV_DOWN")
        # Simulate a press left in the setpoint, then check what RELEASE commands.
        controller._target_pos[2] = 0.15
        seen = []
        robot.on_gripper = lambda: seen.append(float(controller.target_pose[2]))
        controller.step("RELEASE")
        self.assertAlmostEqual(seen[0], 0.17)

    def test_reanchoring_never_goes_below_the_z_floor(self) -> None:
        robot = BlockedRobot(contact_z=0.0)
        controller = _controller(robot, leash=0.004)
        controller.set_z_floor(0.19)
        robot.pose[2] = 0.185  # measured below the floor, e.g. sagging under load
        controller.step("RELEASE")
        self.assertGreaterEqual(float(controller.target_pose[2]), 0.19)


if __name__ == "__main__":
    unittest.main()
