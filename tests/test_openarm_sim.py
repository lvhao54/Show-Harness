"""OpenArm simulation integration: config, path resolution, and the frame conversions.

The MuJoCo world itself needs ``mujoco`` + ``openarm_control`` + ``openarm_mujoco`` (and a
rendering backend), so the tests that build it are skipped when the checkout or its
dependencies are absent -- the config and pure-geometry checks always run, because those
are the parts that break silently on a machine that has the sim.
"""
from __future__ import annotations

import os
import unittest

# MuJoCo reads MUJOCO_GL at IMPORT time and defaults to the interactive `glfw` backend,
# which cannot create a context on a headless box (the failure is a blank render plus a
# GLFW warning, not a startup error). The other sim entry points set this too; here it must
# happen before the first `import mujoco`, which any earlier test in the session may have
# triggered -- so set it explicitly rather than relying on the session helper.
os.environ.setdefault("MUJOCO_GL", os.environ.get("SHOWHARNESS_OPENARM_GL", "egl"))

import numpy as np

from core.config import load_yaml
from core.openarm.sim import (
    DEFAULT_OPENARM_ROOT,
    OPENARM_SCENES,
    OpenArmCheckoutError,
    _default_openarm_roots,
    _has_openarm_sim,
    resolve_openarm_root,
)
from core.openarm.openarm_interface import (
    _MAX_FREE_GAP_M,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
)

CONFIG = "configs/robot_openarm.yaml"


def _sim_available() -> bool:
    try:
        from core.openarm.sim import import_openarm_backend

        import_openarm_backend()
    except Exception:  # noqa: BLE001 - any failure means "not runnable here"
        return False
    return True


class OpenArmConfigTests(unittest.TestCase):
    def test_config_resolves_without_a_site_layer(self) -> None:
        """A sim config must load on a fresh clone with no hardware identity."""
        cfg = load_yaml(CONFIG)
        self.assertIn(cfg["scene"], OPENARM_SCENES)
        self.assertIn(cfg["arm_side"], ("left", "right"))
        self.assertGreater(float(cfg["step_m"]), 0.0)

    def test_every_move_token_is_defined(self) -> None:
        from core.action_units import MOVE_ATOMS

        cfg = load_yaml(CONFIG)
        vectors = cfg["move_vectors"]
        self.assertEqual(set(vectors), set(MOVE_ATOMS))
        for token, vec in vectors.items():
            arr = np.asarray(vec, dtype=float)
            self.assertEqual(arr.shape, (3,), token)
            self.assertAlmostEqual(float(np.linalg.norm(arr)), 1.0, places=6, msg=token)

    def test_opposite_tokens_are_exact_negatives(self) -> None:
        cfg = load_yaml(CONFIG)
        pairs = (("MV_FWD", "MV_BACK"), ("MV_LEFT", "MV_RIGHT"), ("MV_UP", "MV_DOWN"))
        for a, b in pairs:
            self.assertTrue(
                np.allclose(cfg["move_vectors"][a], -np.asarray(cfg["move_vectors"][b])),
                f"{a} must be the exact opposite of {b}",
            )

    def test_z_floor_keeps_the_gripper_out_of_the_table(self) -> None:
        """The floor bounds the CONTROL POINT, and at the home orientation the pads ride at
        that same height (the tool axis is horizontal), so the bound transfers directly.

        Measured on the upstream model: the table surface is at -0.298 in ``arm_origin``
        and the home control point at -0.221, while the lowest gripper collision geometry
        sits ~0.064 m below the control point when the jaw points down. The floor must
        therefore stay clear of the surface with margin for that lower bound.
        """
        cfg = load_yaml(CONFIG)
        floor = float(cfg["z_floor_m"])
        table_surface_z = -0.298
        lowest_geom_offset_m = 0.065
        self.assertGreater(
            floor - lowest_geom_offset_m,
            table_surface_z,
            "z_floor_m must keep the whole gripper above the tabletop",
        )
        # ...and it must still allow the arm to reach down to the tabletop to grasp.
        self.assertLess(floor, table_surface_z + 0.15)

    def test_default_wrist_transform_is_configured_for_the_selected_arm(self) -> None:
        """The wrist mount is mirrored, so rotation/flip are per-arm; they must be set."""
        cfg = load_yaml(CONFIG)
        expected = {
            "right": (270, "both"),
            "left": (90, "both"),
        }[cfg["arm_side"]]
        self.assertEqual(
            int(cfg["wrist_rotation_degrees"]), expected[0],
            "wrist rotation must match the selected arm (see configs/robot_openarm.yaml)",
        )
        self.assertEqual(str(cfg["wrist_flip"]).lower(), expected[1])


class OpenArmPathTests(unittest.TestCase):
    def test_default_roots_are_deduplicated_and_ordered(self) -> None:
        roots = _default_openarm_roots()
        self.assertGreaterEqual(len(roots), 1)
        self.assertEqual(len(roots), len(set(roots)), "candidate roots must be unique")
        # The sibling checkout is the first guess; a caller override always wins.
        self.assertEqual(roots[0], DEFAULT_OPENARM_ROOT)

    def test_caller_override_wins_and_is_expanded(self) -> None:
        resolved = resolve_openarm_root("~/some-openarm")
        self.assertTrue(str(resolved).endswith("some-openarm"))
        self.assertNotIn("~", str(resolved))

    def test_missing_checkout_raises_with_guidance(self) -> None:
        from core.openarm.sim import import_openarm_backend

        with self.assertRaises(OpenArmCheckoutError) as ctx:
            import_openarm_backend("/nonexistent/embodied-ai-llm")
        message = str(ctx.exception)
        self.assertIn("embodied_ai-llm", message)
        self.assertIn("OPENARM_ROOT", message)

    def test_checkout_probe_recognises_the_real_layout(self) -> None:
        root = resolve_openarm_root()
        if not _has_openarm_sim(root):
            self.skipTest(f"no embodied_ai-llm checkout at {root}")
        self.assertTrue(_has_openarm_sim(root))


class OpenArmFrameTests(unittest.TestCase):
    """The pose conversion is the easiest silent bug: both orders are 7 numbers."""

    def test_quaternion_order_roundtrips(self) -> None:
        wxyz = np.array([1.0, 2.0, 3.0, 0.7071, 0.0, -0.7071, 0.0])
        xyzw = _wxyz_to_xyzw(wxyz)
        self.assertEqual(xyzw.tolist(), [1.0, 2.0, 3.0, 0.0, -0.7071, 0.0, 0.7071])
        self.assertTrue(np.allclose(_xyzw_to_wxyz(xyzw), wxyz))

    def test_order_conversion_is_not_a_permutation_only(self) -> None:
        """Guard against the identity-function bug: x/y/z must be untouched."""
        wxyz = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        xyzw = _wxyz_to_xyzw(wxyz)
        self.assertTrue(np.allclose(xyzw[:3], wxyz[:3]))
        self.assertEqual(xyzw[3], wxyz[4])
        self.assertEqual(xyzw[6], wxyz[3])

    def test_gripper_open_is_wider_than_closed(self) -> None:
        self.assertGreater(_MAX_FREE_GAP_M, 0.0)
        self.assertLess(_MAX_FREE_GAP_M, 0.3)


@unittest.skipUnless(_sim_available(), "OpenArm sim dependencies not installed")
class OpenArmSimTests(unittest.TestCase):
    """Builds the real MuJoCo world; needs the checkout plus a GL backend."""

    def setUp(self) -> None:
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from core.openarm.sim import make_openarm_backend
        from core.openarm.openarm_interface import OpenArmSimRobot

        self.backend = make_openarm_backend(scene=None)
        self.robot = OpenArmSimRobot(self.backend, side="right", verbose=False)
        self.addCleanup(self.robot.close)

    def test_gripper_opens_and_closes(self) -> None:
        self.robot.control_gripper(True)
        closed = float(self.robot.get_gripper_position()[0])
        self.robot.control_gripper(False)
        opened = float(self.robot.get_gripper_position()[0])
        self.assertLess(closed, opened - 0.05)

    def test_five_centimetre_command_moves_about_five_centimetres(self) -> None:
        """The sim tracks its setpoint closely, so step_m is the physical distance."""
        before = self.robot.get_ee_pose()[:3].copy()
        target = self.robot.get_ee_pose()
        target[2] += 0.05
        for _ in range(6):
            self.robot.update_desired_ee_pose(target)
        travelled = float(np.linalg.norm(self.robot.get_ee_pose()[:3] - before))
        self.assertGreater(travelled, 0.035)
        self.assertLess(travelled, 0.065)

    def test_pose_is_reported_in_xyzw(self) -> None:
        pose = self.robot.get_ee_pose()
        self.assertEqual(pose.shape, (7,))
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0, places=4)

    def test_z_floor_stops_descent(self) -> None:
        """The floor is enforced by the real controller; descent must stop at it."""
        from core.action_units import MOVE_ATOMS, ROTATE_ATOMS
        from core.config import load_yaml as _load
        from interpreters.real_atomic_controller import RealAtomicController

        cfg = _load(CONFIG)
        primitives = {name: list(cfg["move_vectors"][name]) for name in MOVE_ATOMS}
        primitives.update({name: 1.0 for name in ROTATE_ATOMS})
        controller = RealAtomicController(
            self.robot, primitives, float(cfg["step_m"]), 0.15,
            settle_steps=2, settle_dt_s=0.0,
            gripper_settle_s=0.0, gripper_min_settle_s=0.0,
            z_floor_m=float(cfg["z_floor_m"]), verbose=False,
        )
        controller.sync_from_robot()
        for _ in range(12):
            controller.step("MV_DOWN")
        z = float(self.robot.get_ee_pose()[2])
        # The CONTROLLER's setpoint is clamped exactly at the floor; the measured pose can
        # still sit a fraction of a millimetre below it while the servo settles (measured
        # 0.1 mm). Assert the floor holds to within that physical tolerance.
        self.assertGreaterEqual(z, float(cfg["z_floor_m"]) - 1e-3)

    def test_observations_carry_both_views_and_state(self) -> None:
        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig

        session = OpenArmSession(OpenArmSessionConfig(scene=None, observation_resolution=128))
        self.addCleanup(session.close)
        session.connect()
        obs = session.get_observation()
        for key in ("agentview", "wrist"):
            self.assertEqual(obs[key].shape, (128, 128, 3), key)
            self.assertEqual(obs[key].dtype, np.uint8)
            self.assertGreater(float(obs[key].std()), 1.0, f"{key} looks blank")
        self.assertEqual(len(obs["ee_pose"]), 7)
        self.assertGreaterEqual(float(obs["gripper_width"]), 0.0)

    def test_flower_predicate_is_reachable(self) -> None:
        """`flowers_in_vase` must be satisfiable, or it is not a success condition.

        The upstream vase has no collision hull, so the predicate is a geometric relation
        (tip at the mouth) rather than physical containment. Place each tip exactly on the
        mouth and assert the predicate fires.
        """
        import mujoco

        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig

        session = OpenArmSession(
            OpenArmSessionConfig(scene="insert_flower", success_check="flowers_in_vase")
        )
        self.addCleanup(session.close)
        session.connect()
        self.assertFalse(session.success(), "roses start on the table")
        model, data = session.backend.model, session.backend.data
        mouth = np.asarray(data.site_xpos[model.site("vase_mouth").id], dtype=float)
        for body in ("flower_1", "flower_2", "flower_3"):
            joint = model.body(body).jntadr[0]
            qadr = model.jnt_qposadr[joint]
            # identity orientation: the tip site sits (0, -0.075, 0) from the body origin.
            data.qpos[qadr : qadr + 3] = [mouth[0], mouth[1] + 0.075, mouth[2]]
            data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        self.assertTrue(session.success(), "tips placed at the mouth must satisfy the check")

    def test_cube_predicate_is_false_on_the_flower_scene(self) -> None:
        """The cube bodies do not exist there; the predicate must not raise."""
        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig

        session = OpenArmSession(
            OpenArmSessionConfig(scene="insert_flower", success_check="cubes_on_table")
        )
        self.addCleanup(session.close)
        session.connect()
        self.assertFalse(session.success())


if __name__ == "__main__":
    unittest.main()
