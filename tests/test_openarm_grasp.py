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
    def _run(self, floor=None, body="box_red", lateral=("MV_RIGHT",) * 5):
        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig
        from core.openarm.controller import make_openarm_controller

        session = OpenArmSession(OpenArmSessionConfig(verbose=False)).connect()
        self.addCleanup(session.close)
        cfg = load_yaml("configs/robot_openarm.yaml")
        if floor is not None:
            cfg["z_floor_m"] = floor
        controller = make_openarm_controller(cfg, session.robot, verbose=False)
        controller.sync_from_robot()
        initial = np.array(session.scene_truth()["objects"][body]["position"])
        table = session.backend.model.geom("table_top").id
        sequence = ["RELEASE", *lateral, "MV_FWD"] + ["MV_DOWN"] * 2
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
        final = np.array(session.scene_truth()["objects"][body]["position"])
        self.assertEqual(session.robot._ik_refusals, 0)
        return controller, grasp, width, final[2] - initial[2]

    def test_default_floor_grasps_and_lifts_red_block(self):
        controller, grasp, width, lift = self._run()
        self.assertFalse(grasp.grasp_empty)
        # The jaws close along world Y, across the block's 1 cm width -- above the
        # 6 mm empty-grasp threshold, so a real hold is not mistaken for air.
        self.assertGreater(width, 0.007)
        self.assertLess(width, 0.012)
        self.assertGreater(width, controller.grasp_min_width_m)
        self.assertLess(width, controller.gripper_close_threshold_m)
        self.assertGreater(lift, 0.05)

    def test_default_floor_grasps_and_lifts_45mm_cube(self):
        controller, grasp, width, lift = self._run(body="box_green", lateral=("MV_LEFT",) * 8)
        self.assertFalse(grasp.grasp_empty)
        self.assertGreater(width, 0.035)
        self.assertLess(width, 0.055)
        self.assertLess(width, controller.gripper_close_threshold_m)
        self.assertGreater(lift, 0.05)

    def test_previous_floor_reproduces_empty_grasp_on_45mm_cube(self):
        # The 6 cm red block is tall enough for the old floor; the cubes are not.
        _, grasp, _, lift = self._run(
            floor=-0.225, body="box_green", lateral=("MV_LEFT",) * 8
        )
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
        self.assertTrue(session.success(), "block must remain supported after release and retreat")
        self.assertEqual(session.robot._ik_refusals, 0)

    def _stack_after_pressing(self, leash):
        """Stack, but press three times into the support before releasing."""
        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig
        from core.openarm.controller import make_openarm_controller

        session = OpenArmSession(
            OpenArmSessionConfig(success_check="red_on_green", verbose=False)
        ).connect()
        self.addCleanup(session.close)
        cfg = load_yaml("configs/robot_openarm.yaml")
        cfg["setpoint_leash_m"] = leash
        controller = make_openarm_controller(cfg, session.robot, verbose=False)
        controller.sync_from_robot()
        actions = (
            ["RELEASE"] + ["MV_RIGHT"] * 5 + ["MV_FWD"] + ["MV_DOWN"] * 2
            + ["GRASP"] + ["MV_UP"] * 3 + ["MV_LEFT"] * 13
            + ["MV_DOWN"] * 3 + ["RELEASE"] + ["MV_UP"] * 2
        )
        for token in actions:
            controller.step(token)
        return session.success()

    def test_pressing_into_the_support_still_releases_a_stable_stack(self):
        self.assertTrue(self._stack_after_pressing(leash=0.004))

    def test_without_the_leash_pressing_topples_the_tall_block(self):
        # The wound-up press slides the block through the pads; on release the
        # opening fingers strike the green cube and the block falls.
        self.assertFalse(self._stack_after_pressing(leash=None))
