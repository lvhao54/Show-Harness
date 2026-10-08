"""Task scoring uses independent simulator state, never policy observations."""
from __future__ import annotations

import copy
import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", os.environ.get("SHOWHARNESS_OPENARM_GL", "egl"))

import numpy as np

from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig


class SuccessConfigurationTests(unittest.TestCase):
    def test_unknown_predicate_fails_before_connecting_simulator(self) -> None:
        session = OpenArmSession(OpenArmSessionConfig(success_check="red_on_grean"))
        with patch.object(session, "_connect_openarm") as connect:
            with self.assertRaisesRegex(ValueError, "Unknown success_check.*red_on_grean"):
                session.connect()
        connect.assert_not_called()
        self.assertIsNone(session.backend)


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") and importlib.util.find_spec("openarm_mujoco"),
    "OpenArm MuJoCo models are not installed",
)
class RedOnGreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        from core.openarm.backend.scenes import openarm_scene

        cls.mujoco = mujoco
        with openarm_scene() as path:
            cls.model_template = mujoco.MjModel.from_xml_path(path)

    def setUp(self) -> None:
        # Each test has its own model so a contact fixture cannot alter another test.
        self.model = copy.copy(self.model_template)
        self.data = self.mujoco.MjData(self.model)
        self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.mujoco.mj_forward(self.model, self.data)
        self.session = OpenArmSession(
            OpenArmSessionConfig(success_check="red_on_green", verbose=False)
        )
        self.session.backend = SimpleNamespace(model=self.model, data=self.data)

    def _place(self, name: str, position) -> None:
        joint = self.model.joint(f"{name}_free").id
        address = int(self.model.jnt_qposadr[joint])
        self.data.qpos[address:address + 7] = [*position, 1.0, 0.0, 0.0, 0.0]
        self.mujoco.mj_forward(self.model, self.data)

    def _stack(self) -> None:
        # 0.1 mm penetration is a normal contact solver tolerance, not a glued cube.
        self._place("box_red", [0.4, 0.0, 0.4674])

    def test_initial_scene_has_no_stack(self) -> None:
        self.assertFalse(self.session.success())

    def test_stationary_released_stack_succeeds(self) -> None:
        self._stack()
        self.assertTrue(self.session.success())

    def test_airborne_red_cube_fails_even_when_nearly_aligned(self) -> None:
        # Still within the positional tolerance, but with no supporting contact.
        self._place("box_red", [0.4, 0.0, 0.4685])
        self.assertFalse(self.session.success())

    def test_airborne_pair_of_cubes_fails(self) -> None:
        self._place("box_green", [0.4, 0.0, 0.5225])
        self._place("box_red", [0.4, 0.0, 0.5674])
        self.assertFalse(self.session.success())

    def test_offset_stack_fails_even_with_red_green_contact(self) -> None:
        self._place("box_red", [0.43, 0.0, 0.4674])
        self.assertFalse(self.session.success())

    def test_stack_still_touching_a_gripper_fails(self) -> None:
        self._stack()
        self.assertTrue(self.session.success())
        # Add the robot/cube contact to MuJoCo's live contact buffer. This keeps
        # the qpos fixture and real support contacts identical, isolating the
        # difference between a released stack and one still touching a pad.
        contact = self.mujoco.MjContact()
        contact.geom1 = self.model.geom("box_red_geom").id
        contact.geom2 = self.model.geom("finger_inner_right_collision_00").id
        contact.dist = 0.0
        contact.frame[:] = np.eye(3).ravel()
        self.assertEqual(self.mujoco.mj_addContact(self.model, self.data, contact), 0)
        self.assertFalse(self.session.success())

    def test_moving_stack_fails(self) -> None:
        for cube in ("box_red", "box_green"):
            for dof_offset, speed in ((0, 0.05), (3, 0.5)):
                with self.subTest(cube=cube, dof_offset=dof_offset):
                    self._stack()
                    self.data.qvel[:] = 0.0
                    dof = int(self.model.jnt_dofadr[self.model.joint(f"{cube}_free").id])
                    self.data.qvel[dof + dof_offset] = speed
                    self.mujoco.mj_forward(self.model, self.data)
                    self.assertFalse(self.session.success())

    def test_predicate_does_not_add_ground_truth_to_observation(self) -> None:
        self._stack()
        self.assertTrue(self.session.success())
        self.session.robot = SimpleNamespace(
            get_ee_pose=lambda: np.zeros(7),
            get_gripper_position=lambda: [0.08],
        )
        frames = {
            "camera_head": np.zeros((8, 8, 3), dtype=np.uint8),
            "camera_wrist_right": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        with patch.object(self.session, "_render_all", return_value=frames):
            observation = self.session.get_observation()
        self.assertNotIn("success", observation)
        self.assertNotIn("environment_success", observation)
        self.assertNotIn("scene_truth", observation)
        self.assertNotIn("objects", observation)


if __name__ == "__main__":
    unittest.main()
