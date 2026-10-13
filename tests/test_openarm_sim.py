"""OpenArm simulation integration: config, vendored scenes, and the frame conversions.

The MuJoCo world itself needs ``mujoco`` + ``openarm_control`` + ``openarm_mujoco`` (and a
rendering backend), so the tests that build it are skipped when those wheels are absent --
the config, scene-resolution and pure-geometry checks always run, because those are the
parts that break silently on a machine that has the sim.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

# MuJoCo reads MUJOCO_GL at IMPORT time and defaults to the interactive `glfw` backend,
# which cannot create a context on a headless box (the failure is a blank render plus a
# GLFW warning, not a startup error). The other sim entry points set this too; here it must
# happen before the first `import mujoco`, which any earlier test in the session may have
# triggered -- so set it explicitly rather than relying on the session helper.
os.environ.setdefault("MUJOCO_GL", os.environ.get("SHOWHARNESS_OPENARM_GL", "egl"))

import numpy as np

from core.config import load_yaml
from core.openarm.sim import (
    OPENARM_SCENES,
    OpenArmDependencyError,
    import_openarm_backend,
    is_vlabench_assets_present,
    scene_file,
    vlabench_assets_dir,
)
from core.openarm.openarm_interface import (
    _MAX_FREE_GAP_M,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
)

CONFIG = "configs/robot_openarm.yaml"


def _sim_available() -> bool:
    try:
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

    def test_grasp_height_limits_are_shared_and_below_home(self) -> None:
        # Actual collision clearance and pick/lift are exercised by test_openarm_grasp.
        fine = load_yaml(CONFIG)
        zero = load_yaml("configs/robot_openarm_zero.yaml")
        self.assertEqual(fine["z_floor_m"], zero["z_floor_m"])
        self.assertLess(float(fine["z_floor_m"]), -0.240)
        self.assertGreater(float(fine["z_floor_m"]), -0.265)

    def test_wrist_transform_is_configured_per_arm(self) -> None:
        """The wrist mount is mirrored, so rotation/flip are per-arm; both must be set."""
        from core.launch import openarm_wrist_transform

        cfg = load_yaml(CONFIG)
        for side, expected in (("right", (270, "both")), ("left", (90, "both"))):
            with self.subTest(side=side):
                rotation, flip = openarm_wrist_transform(cfg, side)
                self.assertEqual((rotation, flip.lower()), expected)

    def test_driving_an_arm_without_its_own_wrist_transform_is_refused(self) -> None:
        """Flat wrist keys describe the config's arm only; --arm must not reuse them."""
        from types import SimpleNamespace

        from core.launch import openarm_session_config, openarm_wrist_transform

        cfg = {"arm_side": "right", "wrist_rotation_degrees": 270, "wrist_flip": "both"}
        self.assertEqual(openarm_wrist_transform(cfg, "right"), (270, "both"))
        with self.assertRaisesRegex(ValueError, "wrist_by_arm.left"):
            openarm_wrist_transform(cfg, "left")
        with self.assertRaisesRegex(ValueError, "wrist_by_arm.left"):
            openarm_session_config(cfg, SimpleNamespace(scene=None, arm="left"))

    def test_arm_override_reaches_the_session_with_that_arms_transform(self) -> None:
        from types import SimpleNamespace

        from core.launch import openarm_session_config

        session_cfg = openarm_session_config(
            load_yaml(CONFIG), SimpleNamespace(scene=None, arm="left")
        )
        self.assertEqual(session_cfg.arm_side, "left")
        self.assertEqual(
            (session_cfg.wrist_rotation_degrees, session_cfg.wrist_flip), (90, "both")
        )


class OpenArmSceneTests(unittest.TestCase):
    """The simulator is self-contained: no embodied_ai-llm checkout is consulted."""

    def test_scene_files_are_bundled(self) -> None:
        for scene in OPENARM_SCENES:
            path = scene_file(scene)
            self.assertTrue(path.is_file(), f"scene {scene!r} missing at {path}")
            self.assertIn("<mujoco", path.read_text(encoding="utf-8"))

    def test_unknown_scene_raises_with_the_available_list(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            scene_file("no_such_scene")
        self.assertIn("insert_flower", str(ctx.exception))

    def test_default_scene_needs_no_external_checkout(self) -> None:
        """The cube tabletop ships with the repo; building it must not need a checkout."""
        if not _sim_available():
            self.skipTest("OpenArm wheels not installed")
        from core.openarm.sim import make_openarm_backend

        backend = make_openarm_backend(scene=None)
        self.addCleanup(backend.close)
        self.assertEqual(backend.frame, "arm_origin")

    def test_vlabench_gate_points_at_the_fetch_script(self) -> None:
        """`insert_flower` must fail with the fetch command, not a mesh-not-found trace."""
        if is_vlabench_assets_present():
            self.skipTest("VLABench meshes already downloaded")
        if not _sim_available():
            self.skipTest("OpenArm wheels not installed")
        from core.openarm.sim import make_openarm_backend

        with self.assertRaises(FileNotFoundError) as ctx:
            make_openarm_backend(scene="insert_flower")
        message = str(ctx.exception)
        self.assertIn("download_vlabench_assets.sh", message)
        self.assertIn(str(vlabench_assets_dir()), message)

    def test_missing_dependency_error_names_the_setup_command(self) -> None:
        """The import error an install-less clone hits must name the fix."""
        import builtins

        real_import = builtins.__import__

        def _block_mujoco(name, *args, **kwargs):
            if name == "mujoco":
                raise ModuleNotFoundError("No module named 'mujoco'", name="mujoco")
            return real_import(name, *args, **kwargs)

        import sys

        cached = sys.modules.pop("core.openarm.backend.simulation", None)
        self.addCleanup(
            lambda: sys.modules.__setitem__("core.openarm.backend.simulation", cached)
            if cached is not None
            else None
        )
        with mock.patch.object(builtins, "__import__", _block_mujoco):
            with self.assertRaises(OpenArmDependencyError) as ctx:
                import_openarm_backend()
        message = str(ctx.exception)
        self.assertIn("mujoco", message)
        self.assertIn("--openarm", message)

    def test_scene_helpers_import_without_the_simulator_wheels(self) -> None:
        """``core.openarm.sim`` must import on a base venv, or the hint above is dead code."""
        import subprocess
        import sys

        probe = (
            "import sys\n"
            "for name in ('mujoco', 'openarm_control', 'openarm_mujoco'):\n"
            "    sys.modules[name] = None  # behave as if not installed\n"
            "from core.openarm.sim import OpenArmDependencyError, import_openarm_backend\n"
            "try:\n"
            "    import_openarm_backend()\n"
            "except OpenArmDependencyError as exc:\n"
            "    print('HINT', exc)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HINT", result.stdout)
        self.assertIn("setup.sh base --openarm", result.stdout)


class OpenArmZeroShotWiringTests(unittest.TestCase):
    """The zero-shot harness (planner + plugin stack) can drive the OpenArm sim.

    These are the seams the launcher uses; the rollout itself needs a VLM, so the tests
    stop at the wiring: hardware selection, primitives resolution, session construction and
    the controller factory.
    """

    ZERO_CONFIG = "configs/robot_openarm_zero.yaml"

    def test_launcher_recognises_openarm_hardware(self) -> None:
        from core.launch import resolve_hardware
        from types import SimpleNamespace

        cfg = load_yaml(self.ZERO_CONFIG)
        args = SimpleNamespace(robot_config=self.ZERO_CONFIG)
        self.assertEqual(resolve_hardware(args, cfg), "openarm")

    def test_primitives_default_is_the_openarm_file(self) -> None:
        from types import SimpleNamespace

        from core.launch import resolve_primitives_path

        cfg = load_yaml(self.ZERO_CONFIG)
        cfg.pop("primitives_config", None)
        path = resolve_primitives_path(
            SimpleNamespace(primitives_config=None), cfg, "openarm"
        )
        self.assertTrue(path.endswith("primitives_openarm.yaml"), path)
        self.assertTrue(Path(path).is_file(), path)

    def test_controller_and_finetuned_move_vectors_agree(self) -> None:
        """One action-unit mapping: the primitives file and the fine-tuned config.

        The fine-tuned entry point layers ``move_vectors`` from robot_openarm.yaml over the
        primitives file, so the two representations must not drift.
        """
        primitives = load_yaml("configs/primitives_openarm.yaml")["atomic_primitives"]
        fine = load_yaml(CONFIG)["move_vectors"]
        for token, vec in fine.items():
            self.assertIn(token, primitives, token)
            self.assertTrue(
                np.allclose(primitives[token], vec),
                f"{token}: primitives {primitives[token]} != move_vectors {vec}",
            )

    def test_zero_and_fine_configs_share_the_world(self) -> None:
        """Share physics and safety; preprocessing belongs to each policy path."""
        zero = load_yaml(self.ZERO_CONFIG)
        fine = load_yaml(CONFIG)
        for key in (
            "scene",
            "arm_side",
            "control_hz",
            "sim_ticks_per_command",
            "observation_resolution",
            "agentview_crop_aspect",
            "wrist_crop_aspect",
            "step_m",
            "z_floor_m",
            "setpoint_leash_m",
            "empty_width_m",
            "open_width_m",
            "gripper_close_threshold_m",
        ):
            self.assertEqual(zero.get(key), fine.get(key), f"{key} drifted between the modes")

    def test_each_mode_keeps_the_camera_contract_its_policy_needs(self) -> None:
        """Preserve legacy fine-tuned transforms and upright zero-shot views.

        The zero-shot OpenArm prompt describes its execution-time FWD/BACK
        exchange explicitly; it does not treat the wrist side view as overhead.
        """
        from core.launch import openarm_wrist_transform

        zero = load_yaml(self.ZERO_CONFIG)
        fine = load_yaml(CONFIG)
        self.assertEqual(openarm_wrist_transform(fine, "right"), (270, "both"))
        # Zero-shot: native orientation, no flip -- for either arm.
        self.assertEqual(str(zero.get("agentview_flip", "none")).lower(), "none")
        self.assertEqual(int(zero.get("agentview_rotation_degrees", 0)), 0)
        for side in ("right", "left"):
            self.assertEqual(openarm_wrist_transform(zero, side), (0, "none"), side)
        # ...compensated in the token->motion mapping.
        from core.launch import execution_token_swap_pairs
        from core.config import resolve_vlm_config

        resolved = dict(zero)
        resolved["vlm"] = resolve_vlm_config(zero, backend="gateway")
        self.assertEqual(
            execution_token_swap_pairs(resolved),
            (("MV_FWD", "MV_BACK"),),
            "the zero-shot OpenArm config must swap FWD/BACK at execution time",
        )

    def test_zero_shot_depth_axis_is_corrected_exactly_once(self) -> None:
        """Guard against fixing the same mismatch twice (flip AND swap), which would
        cancel out and look like no fix at all."""
        zero = load_yaml(self.ZERO_CONFIG)
        flipped = str(zero.get("agentview_flip", "none")).lower() != "none"
        self.assertFalse(flipped)

    def test_session_defaults_to_the_config_arm(self) -> None:
        """The Piper default arm must not leak into the OpenArm rig (it is camera-specific)."""
        from types import SimpleNamespace

        from core.launch import openarm_session_config

        cfg = load_yaml(self.ZERO_CONFIG)
        config = openarm_session_config(cfg, SimpleNamespace(scene=None, arm=None))
        self.assertEqual(config.arm_side, cfg["arm_side"])

    def test_controller_factory_disables_the_floor_on_request(self) -> None:
        if not _sim_available():
            self.skipTest("OpenArm wheels not installed")
        from core.openarm.controller import make_openarm_controller
        from core.openarm.openarm_interface import OpenArmSimRobot
        from core.openarm.sim import make_openarm_backend

        cfg = load_yaml(self.ZERO_CONFIG)
        backend = make_openarm_backend(scene=None)
        robot = OpenArmSimRobot(backend, verbose=False)
        self.addCleanup(robot.close)
        controller = make_openarm_controller(cfg, robot, z_floor_m=None, verbose=False)
        self.assertIsNone(controller.z_floor_m)
        controller = make_openarm_controller(cfg, robot, verbose=False)
        self.assertAlmostEqual(controller.z_floor_m, float(cfg["z_floor_m"]))

    def test_zero_shot_controller_uses_the_sim_settle_contract(self) -> None:
        """The real-rig settle defaults (the ``robot:`` block) must not leak in.

        The zero-shot launcher builds its kwargs from the hardware defaults; the OpenArm
        branch has to drop them, or a rollout would sleep like a Franka between gripper
        commands and step the wrong number of physics ticks per token.
        """
        if not _sim_available():
            self.skipTest("OpenArm wheels not installed")
        import argparse

        from core.config import load_yaml as _load
        from core.launch import make_controller, make_session, resolve_primitives_path

        cfg = _load(self.ZERO_CONFIG)
        args = argparse.Namespace(
            scene=None, arm=None, primitives_config=None, mock_robot=False,
            no_z_floor=False, z_floor_m=None, z_floor=False, z_floor_name=None,
            mock_cameras=False,
        )
        session = make_session(cfg, args, "openarm")
        session.connect()
        self.addCleanup(session.close)
        primitives = _load(resolve_primitives_path(args, cfg, "openarm"))
        controller = make_controller(cfg, primitives, session, args, "openarm")
        self.assertEqual(controller.settle_steps, int(cfg["settle_steps"]))
        self.assertEqual(controller.gripper_settle_s, float(cfg["gripper_settle_s"]))
        self.assertEqual(controller.gripper_min_settle_s, float(cfg["gripper_min_settle_s"]))
        self.assertEqual(controller.grasp_open_width_m, float(cfg["open_width_m"]))
        self.assertAlmostEqual(controller.step_m, float(cfg["step_m"]))
        self.assertAlmostEqual(controller.grasp_min_width_m, float(cfg["empty_width_m"]))

    def test_recovery_off_also_turns_off_the_empty_grasp_reopen(self) -> None:
        """plugins.recovery gates the controller's auto-reopen, as on Franka and Piper."""
        if not _sim_available():
            self.skipTest("OpenArm wheels not installed")
        import argparse

        from core.config import load_yaml as _load
        from core.launch import make_controller, make_session, resolve_primitives_path

        cfg = _load(self.ZERO_CONFIG)
        cfg["plugins"] = {**cfg["plugins"], "recovery": False}
        args = argparse.Namespace(
            scene=None, arm=None, primitives_config=None, mock_robot=False,
            no_z_floor=False, z_floor_m=None, z_floor=False, z_floor_name=None,
            mock_cameras=False,
        )
        session = make_session(cfg, args, "openarm")
        session.connect()
        self.addCleanup(session.close)
        primitives = _load(resolve_primitives_path(args, cfg, "openarm"))
        controller = make_controller(cfg, primitives, session, args, "openarm")
        self.assertIsNone(controller.grasp_min_width_m)

    def test_franka_defaults_do_not_leak_into_the_openarm_config(self) -> None:
        """No Franka table height as a floor or table reference; step_m is the step."""
        import argparse

        from core.launch import TABLE_CONTACT_Z_M, build_config, table_height_m

        args = argparse.Namespace(
            vlm_url=None, model=None, vlm_backend=None, fine_step_m=None
        )
        robot_cfg = load_yaml(self.ZERO_CONFIG)
        robot_cfg.pop("z_floor_m")
        robot_cfg["step_m"] = 0.03
        cfg = build_config(args, robot_cfg)
        # A config without a floor stays floorless, so startup demands a measurement.
        self.assertIsNone(cfg["z_floor_m"])
        self.assertAlmostEqual(cfg["fine_step_m"], 0.03)
        args.fine_step_m = 0.01
        self.assertAlmostEqual(build_config(args, robot_cfg)["fine_step_m"], 0.01)
        # Without a floor, OpenArm has no table reference; Franka keeps its constant.
        self.assertIsNone(table_height_m({"hardware": "openarm"}, None))
        self.assertAlmostEqual(table_height_m({"hardware": "openarm"}, -0.25), -0.25)
        self.assertAlmostEqual(table_height_m({"hardware": "franka"}, None), TABLE_CONTACT_Z_M)

    def test_default_task_follows_the_scene_that_was_built(self) -> None:
        from scripts.run_openarm_mvtoken import _default_task

        self.assertIn("vase", _default_task("insert_flower"))
        self.assertIn("red block", _default_task(None))


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
    """Builds the real MuJoCo world; needs the sim wheels plus a GL backend.

    The default cube scene is self-contained. Tests that load ``insert_flower`` call
    :meth:`_require_flower_scene`, which skips when its VLABench meshes have not been
    downloaded -- a fresh clone must pass without them.
    """

    @staticmethod
    def _require_flower_scene() -> None:
        if not is_vlabench_assets_present():
            raise unittest.SkipTest(
                "insert_flower meshes not downloaded; run "
                "bash scripts/model/download_vlabench_assets.sh"
            )

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

    def test_rollout_and_live_view_threads_can_both_capture(self) -> None:
        """The EGL context is thread-bound; captures from two threads must both work."""
        import threading

        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig

        session = OpenArmSession(OpenArmSessionConfig(scene=None, observation_resolution=64))
        self.addCleanup(session.close)
        session.connect()
        # Only the views the policy reads are rendered.
        self.assertEqual(
            set(session.backend.cameras), {"camera_head", "camera_wrist_right"}
        )
        shown: list = []
        live = threading.Thread(target=lambda: shown.append(session.get_camera_frames()))
        live.start()
        live.join(60)
        self.assertIsNotNone(shown[0], "the live-view thread could not capture")
        for _ in range(3):
            obs = session.get_observation()
            self.assertGreater(float(obs["agentview"].std()), 1.0)
        session.close()
        self.assertIsNone(session.get_camera_frames())

    def test_flower_predicate_is_reachable(self) -> None:
        """`flowers_in_vase` must be satisfiable, or it is not a success condition.

        The upstream vase has no collision hull, so the predicate is a geometric relation
        (tip at the mouth) rather than physical containment. Place each tip exactly on the
        mouth and assert the predicate fires.
        """
        import mujoco

        from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig

        self._require_flower_scene()
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

        self._require_flower_scene()
        session = OpenArmSession(
            OpenArmSessionConfig(scene="insert_flower", success_check="cubes_on_table")
        )
        self.addCleanup(session.close)
        session.connect()
        self.assertFalse(session.success())


if __name__ == "__main__":
    unittest.main()
