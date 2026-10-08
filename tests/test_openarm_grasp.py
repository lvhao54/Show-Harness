"""Physical regressions: a configured floor must permit a real grasp, not just IK."""
from __future__ import annotations

import unittest

import numpy as np

from core.config import load_yaml
from core.openarm.sim import OpenArmDependencyError, import_openarm_backend


def _available():
    try:
        import_openarm_backend()
        return True
    except OpenArmDependencyError:
        return False


@unittest.skipUnless(_available(), "OpenArm dependencies unavailable")
class OpenArmPhysicalGraspTests(unittest.TestCase):
    def _run(self, floor=None):
        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig
        from core.openarm.controller import make_openarm_controller

        session = OpenArmSession(OpenArmSessionConfig(verbose=False)).connect()
        self.addCleanup(session.close)
        cfg = load_yaml("configs/robot_openarm.yaml")
        if floor is not None:
            cfg["z_floor_m"] = floor
        controller = make_openarm_controller(cfg, session.robot, verbose=False)
        controller.sync_from_robot()
        initial = np.array(session.scene_truth()["objects"]["box_red"]["position"])
        table = session.backend.model.geom("table_top").id
        sequence = ["RELEASE"] + ["MV_RIGHT"] * 5 + ["MV_FWD"] + ["MV_DOWN"] * 2
        for token in sequence:
            controller.step(token)
            # Test actual contacts, not an assumed 65 mm offset from the wrist.
            for contact in session.backend.data.contact:
                ids = list(contact.geom)
                if table in ids:
                    other = ids[1 - ids.index(table)]
                    self.assertNotIn("right", session.backend.model.geom(other).name)
        grasp = controller.step("GRASP")
        width = float(session.robot.get_gripper_position()[0])
        for _ in range(3):
            controller.step("MV_UP")
        final = np.array(session.scene_truth()["objects"]["box_red"]["position"])
        self.assertEqual(session.robot._ik_refusals, 0)
        return controller, grasp, width, final[2] - initial[2]

    def test_default_floor_grasps_and_lifts_cube(self):
        controller, grasp, width, lift = self._run()
        self.assertFalse(grasp.grasp_empty)
        self.assertGreater(width, 0.035)
        self.assertLess(width, 0.055)
        self.assertLess(width, controller.gripper_close_threshold_m)
        self.assertGreater(lift, 0.05)

    def test_previous_floor_reproduces_empty_grasp(self):
        _, grasp, _, lift = self._run(floor=-0.225)
        self.assertTrue(grasp.grasp_empty)
        self.assertLess(lift, 0.005)

    def test_default_controller_transports_and_releases_a_stable_stack(self):
        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig
        from core.openarm.controller import make_openarm_controller

        session = OpenArmSession(
            OpenArmSessionConfig(success_check="red_on_green", verbose=False)
        ).connect()
        self.addCleanup(session.close)
        controller = make_openarm_controller(
            load_yaml("configs/robot_openarm.yaml"), session.robot, verbose=False
        )
        controller.sync_from_robot()
        actions = (
            ["RELEASE"] + ["MV_RIGHT"] * 5 + ["MV_FWD"] + ["MV_DOWN"] * 2
            + ["GRASP"] + ["MV_UP"] * 3 + ["MV_LEFT"] * 13
            + ["MV_DOWN", "RELEASE"] + ["MV_UP"] * 2
        )
        for token in actions:
            controller.step(token)
            for contact in session.backend.data.contact:
                names = {session.backend.model.geom(g).name for g in contact.geom}
                self.assertNotEqual(
                    names, {"openarm_body_link0_collision", "link4_right_collision_00"},
                    "IK must route the elbow around the pedestal during central transport",
                )
        self.assertTrue(session.success(), "cube must remain supported after release and retreat")
        self.assertEqual(session.robot._ik_refusals, 0)
