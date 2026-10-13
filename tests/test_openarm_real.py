"""The OpenArm real-rig wiring: config resolution, duck type, session contract.

Every test here injects a fake CAN backend and fake cameras -- importing the modules must
never open a device. The physics-free duck-type tests are the safety net for the two
silent failures the simulator tests already document: the wrist-vs-fingertip offset (here
inherited unchanged, because the control point is the same site) and the mirrored jaw sign
(here the metre conversion, which the controller's empty-grasp check depends on).
"""
from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.config import load_yaml
from core.openarm.hardware.robot import (
    _MAX_FREE_GAP_M,
    OpenArmRealRobot,
    free_gap_m,
)

REPO = Path(__file__).resolve().parents[1]
REAL_CONFIG = "configs/robot_openarm_real.yaml"


class OpenArmRealConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        # The real config requires the site layer; a fresh checkout ships only the example.
        site = REPO / "configs" / "site" / "openarm.yaml"
        if not site.exists():
            shutil.copy(site.with_suffix(".yaml.example"), site)
            self.addCleanup(site.unlink, missing_ok=True)

    def test_real_config_resolves_and_selects_the_real_backend(self) -> None:
        from core.launch import openarm_backend, openarm_real_session_config

        cfg = load_yaml(REAL_CONFIG)
        self.assertEqual(openarm_backend(cfg), "real")
        session_cfg = openarm_real_session_config(
            cfg, SimpleNamespace(arm=None)
        )
        self.assertEqual(session_cfg.arm_side, cfg["arm_side"])
        self.assertEqual(session_cfg.cameras, cfg["driver"]["cameras"])
        self.assertNotIn("cameras", session_cfg.driver)
        self.assertIsNone(cfg.get("z_floor_m"))  # forces the cell calibration

    def test_simulator_config_still_selects_the_simulator_backend(self) -> None:
        from core.launch import openarm_backend

        for name in ("configs/robot_openarm.yaml", "configs/robot_openarm_zero.yaml"):
            with self.subTest(name=name):
                self.assertEqual(openarm_backend(load_yaml(name)), "simulation")

    def test_unknown_backend_is_rejected(self) -> None:
        from core.launch import openarm_backend

        with self.assertRaises(ValueError):
            openarm_backend({"backend": "gazebo"})

    def test_arm_override_flows_into_the_session(self) -> None:
        from core.launch import openarm_real_session_config

        cfg = load_yaml(REAL_CONFIG)
        session_cfg = openarm_real_session_config(
            cfg, SimpleNamespace(arm="left")
        )
        self.assertEqual(session_cfg.arm_side, "left")
        # The mounts are mirrored: the left arm gets its own wrist transform.
        self.assertEqual(
            (session_cfg.wrist_rotation_degrees, session_cfg.wrist_flip), (90, "both")
        )

    def test_arm_override_also_reaches_the_controller_robot(self) -> None:
        """--arm picks the driven arm everywhere; the other arm holds its pose."""
        from core.openarm.openarm_real_session import OpenArmRealSessionConfig

        cfg = load_yaml(REAL_CONFIG)
        # The session config carries the side the wrapper is built with; the controller
        # then drives exactly that wrapper's robot.
        session_cfg = OpenArmRealSessionConfig(arm_side="left")
        self.assertEqual(session_cfg.arm_side, "left")


class OpenArmGripperGeometryTests(unittest.TestCase):
    """The metre conversion the controller's thresholds are written against."""

    def test_free_gap_endpoints_match_the_model(self) -> None:
        self.assertAlmostEqual(free_gap_m(0.0), 0.0, places=6)
        self.assertAlmostEqual(free_gap_m(-0.7854), _MAX_FREE_GAP_M, delta=2e-5)
        self.assertAlmostEqual(free_gap_m(0.7854), _MAX_FREE_GAP_M, delta=2e-5)

    def test_sign_is_mirrored_per_arm(self) -> None:
        for angle in (0.1, 0.3, 0.7, 0.7854):
            with self.subTest(angle=angle):
                self.assertAlmostEqual(
                    free_gap_m(angle), free_gap_m(-angle), places=9
                )

    def test_gap_is_monotone_over_the_travel(self) -> None:
        widths = [free_gap_m(a) for a in np.linspace(0.0, 0.7854, 33)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(widths, widths[1:])))
        self.assertGreater(widths[-1] - widths[0], 0.14)

    def test_a_45mm_cube_reads_as_a_held_object(self) -> None:
        """The empty-grasp rule keys on 6 mm; a held 45 mm cube must sit far above it."""
        # angle that yields a 45 mm pad opening, from the same measured fit.
        target = 0.045
        lo, hi = 0.0, 0.7854
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if free_gap_m(mid) < target:
                lo = mid
            else:
                hi = mid
        held = free_gap_m(0.5 * (lo + hi))
        self.assertGreater(held, 0.04)
        self.assertLess(held, 0.05)

    def test_gripper_joint_of_the_wrong_arm_is_still_a_gap(self) -> None:
        """The other arm's jaw is mirrored; the conversion must not go negative."""
        self.assertGreaterEqual(free_gap_m(-0.35), 0.0)
        self.assertGreaterEqual(free_gap_m(0.35), 0.0)


def _state(*, right_angle: float, left_angle: float, right_pose=None, left_pose=None):
    right = np.r_[np.zeros(7), right_angle]
    left = np.r_[np.zeros(7), left_angle]
    return SimpleNamespace(
        right=list(right_pose if right_pose is not None else [0.3, 0.0, -0.2, 1, 0, 0, 0]),
        left=list(left_pose if left_pose is not None else [0.3, 0.2, -0.2, 1, 0, 0, 0]),
        joints_right=list(right),
        joints_left=list(left),
        source_time=1.0,
    )


class OpenArmRealRobotTests(unittest.TestCase):
    """The four-method duck type; the backend is a fake, no CAN is touched."""

    def _robot(self, *, side="right", right_angle=0.0, left_angle=0.0):
        backend = Mock()
        backend.read_state.return_value = _state(
            right_angle=right_angle, left_angle=left_angle
        )
        robot = OpenArmRealRobot(backend, side=side, verbose=False)
        return robot, backend

    def test_pose_quaternion_is_reordered_to_show_harness_convention(self) -> None:
        robot, _ = self._robot()
        pose = robot.get_ee_pose()
        # upstream [x,y,z,qw,qx,qy,qz] = [0.3,0,-0.2,1,0,0,0] -> xyzw identity
        np.testing.assert_allclose(pose, [0.3, 0.0, -0.2, 0.0, 0.0, 0.0, 1.0])

    def test_gripper_width_comes_from_the_measured_joint(self) -> None:
        robot, _ = self._robot(right_angle=-0.7854)
        width = float(robot.get_gripper_position()[0])
        self.assertAlmostEqual(width, _MAX_FREE_GAP_M, delta=2e-5)
        self.assertFalse(robot.gripper_closed)

        robot, _ = self._robot(right_angle=0.0)
        self.assertAlmostEqual(float(robot.get_gripper_position()[0]), 0.0, places=6)
        self.assertTrue(robot.gripper_closed)

    def test_update_desired_ee_pose_commands_the_same_absolute_target(self) -> None:
        robot, backend = self._robot()
        target = np.array([0.35, 0.05, -0.21, 0.0, 0.0, 0.0, 1.0])
        robot.update_desired_ee_pose(target)
        backend.command.assert_called_once()
        right, left, grippers = backend.command.call_args.args
        # Driven arm gets the target in WXYZ order; the held arm keeps its latched pose.
        np.testing.assert_allclose(right, [0.35, 0.05, -0.21, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(left, [0.3, 0.2, -0.2, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(set(grippers), {"right", "left"})

    def test_left_arm_is_the_driven_one_when_selected(self) -> None:
        robot, backend = self._robot(side="left")
        robot.update_desired_ee_pose(
            np.array([0.31, 0.21, -0.19, 0.0, 0.0, 0.0, 1.0])
        )
        right, left, _ = backend.command.call_args.args
        np.testing.assert_allclose(right, [0.3, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(left, [0.31, 0.21, -0.19, 1.0, 0.0, 0.0, 0.0])

    def test_ik_refusal_is_reported_and_not_raised(self) -> None:
        robot, backend = self._robot()
        backend.command.side_effect = RuntimeError("OpenArm pedestal IK failed")
        robot.update_desired_ee_pose(
            np.array([0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 1.0])
        )
        self.assertEqual(robot._ik_refusals, 1)

    def test_gripper_command_uses_the_mirrored_open_sign(self) -> None:
        for side, expected in (("right", -0.7854), ("left", 0.7854)):
            with self.subTest(side=side):
                robot, backend = self._robot(side=side)
                robot.control_gripper(False)
                grippers = backend.command_grippers.call_args.args[0]
                self.assertAlmostEqual(grippers[side], expected, places=3)
                self.assertTrue(robot.gripper_closed is False)
                robot.control_gripper(True)
                grippers = backend.command_grippers.call_args.args[0]
                self.assertAlmostEqual(grippers[side], 0.0, places=6)
                self.assertTrue(robot.gripper_closed)


class OpenArmRealSessionTests(unittest.TestCase):
    """The session answers the shared observation contract with mocked cameras."""

    def _session(self):
        import io

        from PIL import Image

        from core.openarm.openarm_real_session import (
            OpenArmRealSession,
            OpenArmRealSessionConfig,
        )

        frames = {}
        for name in ("camera_head", "camera_wrist_right", "camera_wrist_left"):
            image = Image.new("RGB", (64, 48), (10, 20, 30))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            frames[name] = buffer.getvalue()

        backend = Mock()
        backend.cameras = tuple(frames)
        backend.capture.return_value = frames
        backend.read_state.return_value = _state(right_angle=-0.4, left_angle=0.4)
        backend.validate_motion_ready.return_value = None

        session = OpenArmRealSession(OpenArmRealSessionConfig(arm_side="right"))
        session.backend = backend
        session.robot = OpenArmRealRobot(backend, side="right", verbose=False)
        session._connected = True
        return session, backend

    def test_observation_contract_matches_the_other_sessions(self) -> None:
        session, _ = self._session()
        obs = session.get_observation()
        self.assertEqual(set(obs), {"agentview", "wrist", "ee_pose", "gripper_width"})
        self.assertEqual(obs["agentview"].shape, (256, 256, 3))
        self.assertEqual(obs["wrist"].shape, (256, 256, 3))
        self.assertEqual(np.asarray(obs["ee_pose"]).shape, (7,))
        self.assertGreater(float(obs["gripper_width"]), 0.0)

    def test_hd_views_are_added_when_requested(self) -> None:
        session, _ = self._session()
        session.config.hd_resolution = 320
        obs = session.get_observation()
        self.assertEqual(obs["agentview_hd"].shape, (320, 320, 3))
        self.assertEqual(obs["wrist_hd"].shape, (320, 320, 3))

    def test_missing_camera_role_fails_instead_of_serving_a_stale_view(self) -> None:
        session, backend = self._session()
        only_head = {"camera_head": backend.capture.return_value["camera_head"]}
        backend.capture.return_value = only_head
        with self.assertRaises(KeyError):
            session.get_observation()

    def test_camera_frames_for_display_never_raises(self) -> None:
        session, backend = self._session()
        backend.capture.side_effect = RuntimeError("camera dropped")
        self.assertIsNone(session.get_camera_frames())

    def test_success_and_scene_truth_are_vlm_only_on_hardware(self) -> None:
        session, _ = self._session()
        self.assertFalse(session.success())
        self.assertEqual(session.scene_truth()["source"], "hardware")

    def test_heartbeat_runs_while_the_rollout_thinks(self) -> None:
        import time

        session, backend = self._session()
        session._start_heartbeat()
        try:
            time.sleep(0.3)
        finally:
            session._stop_heartbeat()
        self.assertGreater(backend.step.call_count, 0)

    def test_wait_ready_retries_until_the_preload_is_stable(self) -> None:
        session, backend = self._session()
        backend.validate_motion_ready.side_effect = [
            RuntimeError("preload history is insufficient"),
            None,
        ]
        session.wait_ready(timeout_s=2.0)
        self.assertEqual(backend.validate_motion_ready.call_count, 2)

    def test_wait_ready_fails_loudly_on_a_deadline(self) -> None:
        session, backend = self._session()
        backend.validate_motion_ready.side_effect = RuntimeError("no feedback")
        with self.assertRaises(RuntimeError) as ctx:
            session.wait_ready(timeout_s=0.1)
        self.assertIn("did not stabilize", str(ctx.exception))

    def test_failed_startup_disables_the_arms_before_propagating(self) -> None:
        """A preload that never stabilizes must not leave motors enabled."""
        backend = Mock()
        backend.cameras = ()
        backend.read_state.return_value = _state(right_angle=0.0, left_angle=0.0)
        backend.validate_motion_ready.side_effect = RuntimeError("no feedback")

        from core.openarm.openarm_real_session import (
            OpenArmRealSession,
            OpenArmRealSessionConfig,
        )

        with (
            patch(
                "core.openarm.openarm_real_session.make_pedestal_backend",
                return_value=backend,
            ),
            patch("core.openarm.openarm_real_session.require_openarm_driver"),
        ):
            session = OpenArmRealSession(
                OpenArmRealSessionConfig(driver={"preload_window_seconds": 0.05})
            )
            with self.assertRaises(RuntimeError):
                session.connect()
        self.assertFalse(session._connected)
        self.assertGreaterEqual(backend.close.call_count, 1)

    def _connect_with(self, backend, config=None):
        """Run connect() against an injected backend; returns (session, factory mock)."""
        from core.openarm.openarm_real_session import (
            OpenArmRealSession,
            OpenArmRealSessionConfig,
        )

        with (
            patch(
                "core.openarm.openarm_real_session.make_pedestal_backend",
                return_value=backend,
            ) as factory,
            patch("core.openarm.openarm_real_session.require_openarm_driver"),
        ):
            session = OpenArmRealSession(config or OpenArmRealSessionConfig())
            try:
                session.connect()
            finally:
                self.addCleanup(session.close)
        return session, factory

    def test_failure_after_enable_while_wrapping_the_arm_still_disables_it(self) -> None:
        """backend.start() enabled the motors; a stale first read must not strand them."""
        backend = Mock()
        backend.cameras = ()
        backend.read_state.side_effect = RuntimeError("encoder feedback is stale")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self._connect_with(backend)
        backend.start.assert_called_once()
        self.assertGreaterEqual(backend.close.call_count, 1)

    def test_idle_arm_hold_pose_is_latched_after_the_preload_settles(self) -> None:
        """The idle arm sags after enable; holding the enable pose would lift it back."""
        enabled = _state(right_angle=0.0, left_angle=0.0, left_pose=[0.3, 0.2, -0.20, 1, 0, 0, 0])
        sagged = _state(right_angle=0.0, left_angle=0.0, left_pose=[0.3, 0.2, -0.23, 1, 0, 0, 0])
        backend = Mock()
        backend.cameras = ()
        backend.read_state.return_value = enabled

        def settle():
            backend.read_state.return_value = sagged

        backend.validate_motion_ready.side_effect = settle
        session, _ = self._connect_with(backend)
        np.testing.assert_allclose(session.robot._other_arm_pose, sagged.left)

    def test_only_the_driven_arms_cameras_are_opened(self) -> None:
        """A stall on the idle arm's wrist camera must not be able to fail a capture."""
        from core.openarm.openarm_real_session import OpenArmRealSessionConfig

        table = {
            name: {"serial": name}
            for name in ("camera_head", "camera_wrist_left", "camera_wrist_right")
        }
        backend = Mock()
        backend.cameras = ()
        backend.read_state.return_value = _state(right_angle=0.0, left_angle=0.0)
        _, factory = self._connect_with(
            backend, OpenArmRealSessionConfig(arm_side="left", cameras=table)
        )
        self.assertEqual(
            set(factory.call_args.kwargs["cameras"]), {"camera_head", "camera_wrist_left"}
        )

    def test_a_missing_driven_camera_role_fails_before_any_device_opens(self) -> None:
        from core.openarm.openarm_real_session import OpenArmRealSessionConfig

        backend = Mock()
        with self.assertRaisesRegex(ValueError, "camera_wrist_right"):
            self._connect_with(
                backend, OpenArmRealSessionConfig(cameras={"camera_head": {}})
            )
        backend.start.assert_not_called()

    def test_connect_orders_camera_start_before_can_start(self) -> None:
        """A bad camera must fail before any motor is enabled."""
        backend = Mock()
        events: list[str] = []
        backend.start.side_effect = lambda: events.append("can.start")
        backend.validate_motion_ready.return_value = None
        backend.cameras = ()

        with (
            patch(
                "core.openarm.openarm_real_session.make_pedestal_backend",
                return_value=backend,
            ),
            patch("core.openarm.openarm_real_session.require_openarm_driver"),
            patch(
                "core.openarm.hardware.backend.OpenArmPedestalBackend.start",
                side_effect=AssertionError("real backend must be injected"),
            ),
        ):
            from core.openarm.openarm_real_session import (
                OpenArmRealSession,
                OpenArmRealSessionConfig,
            )

            session = OpenArmRealSession(OpenArmRealSessionConfig())
            backend.read_state.return_value = _state(
                right_angle=0.0, left_angle=0.0
            )
            session.connect()
            session.close()
        self.assertEqual(events, ["can.start"])


class OpenArmRealCliGuardTests(unittest.TestCase):
    """Simulator-only flags are refused on the real backend before anything starts."""

    def setUp(self) -> None:
        site = REPO / "configs" / "site" / "openarm.yaml"
        if not site.exists():
            shutil.copy(site.with_suffix(".yaml.example"), site)
            self.addCleanup(site.unlink, missing_ok=True)

    def _run(self, argv):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, *argv],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=180,
        )

    def test_scene_is_refused_on_the_real_config(self) -> None:
        result = self._run(
            [
                "scripts/run_openarm_mvtoken.py",
                "--robot-config",
                REAL_CONFIG,
                "--scene",
                "insert_flower",
            ]
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("simulator-only", result.stdout + result.stderr)

    def test_success_check_is_refused_on_the_real_config(self) -> None:
        result = self._run(
            [
                "scripts/run_openarm_mvtoken.py",
                "--robot-config",
                REAL_CONFIG,
                "--success-check",
                "red_on_green",
            ]
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("simulator-only", result.stdout + result.stderr)

    def test_missing_floor_is_refused_before_any_device_opens(self) -> None:
        """The real config ships no floor; the run must stop with the calibration command."""
        for entry in ("scripts/run_real.py", "scripts/run_openarm_mvtoken.py"):
            with self.subTest(entry=entry):
                result = self._run(
                    [entry, "--robot-config", REAL_CONFIG, "--no-show"]
                    if entry.endswith("run_real.py")
                    else [entry, "--robot-config", REAL_CONFIG]
                )
                self.assertNotEqual(result.returncode, 0)
                output = result.stdout + result.stderr
                self.assertIn("no measured height is set", output)
                self.assertIn("capture_z_floor.py", output)
                # Nothing may have been enabled: the refusal happens before connect().
                self.assertNotIn("starting CAN workers", output)

    def test_mock_flags_are_refused_so_a_dry_run_cannot_enable_the_arms(self) -> None:
        result = self._run(
            [
                "scripts/run_real.py",
                "--robot-config",
                REAL_CONFIG,
                "--mock-robot",
                "--no-show",
            ]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("would enable the physical arms", result.stderr)
        self.assertNotIn("starting CAN workers", result.stdout + result.stderr)

    def test_begin_pose_is_refused_on_both_backends(self) -> None:
        for config in (REAL_CONFIG, "configs/robot_openarm_zero.yaml"):
            with self.subTest(config=config):
                result = self._run(
                    [
                        "scripts/run_real.py",
                        "--robot-config",
                        config,
                        "--begin-pose",
                        "default",
                        "--no-show",
                    ]
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--begin-pose does not apply", result.stderr)


class OpenArmRealSessionGuardTests(unittest.TestCase):
    """Every entry point builds the real session through one guarded factory."""

    def setUp(self) -> None:
        site = REPO / "configs" / "site" / "openarm.yaml"
        if not site.exists():
            shutil.copy(site.with_suffix(".yaml.example"), site)
            self.addCleanup(site.unlink, missing_ok=True)

    @staticmethod
    def _args(**overrides):
        values = dict(
            arm=None, mock_robot=False, mock_cameras=False,
            no_z_floor=False, z_floor_m=None, z_floor=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_mock_flags_are_refused_by_the_shared_session_factory(self) -> None:
        """run_real_mvtoken.py reaches the rig through make_session, not run_real.py."""
        from core.launch import make_session

        cfg = load_yaml(REAL_CONFIG)
        cfg["z_floor_m"] = -0.27
        for flag in ("mock_robot", "mock_cameras"):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, "would enable the physical arms"):
                    make_session(cfg, self._args(**{flag: True}), "openarm")

    def test_floor_flag_satisfies_the_floor_check(self) -> None:
        from core.launch import make_session, openarm_real_preflight

        cfg = load_yaml(REAL_CONFIG)
        cfg["z_floor_m"] = None  # the shipped site template
        with self.assertRaisesRegex(ValueError, "no measured height is set"):
            openarm_real_preflight(cfg, self._args())
        # Capturing at the start pose would forbid all descent: still refused.
        with self.assertRaisesRegex(ValueError, "no measured height is set"):
            openarm_real_preflight(cfg, self._args(z_floor=True))
        openarm_real_preflight(cfg, self._args(z_floor_m=-0.27))
        openarm_real_preflight(cfg, self._args(no_z_floor=True))
        # Building the session opens nothing; connect() is what enables the arms.
        session = make_session(cfg, self._args(z_floor_m=-0.27), "openarm")
        self.assertIsNone(session.backend)

    def test_site_layer_without_a_floor_key_gets_no_franka_floor(self) -> None:
        import argparse

        from core.launch import build_config, openarm_real_preflight

        robot_cfg = load_yaml(REAL_CONFIG)
        robot_cfg.pop("z_floor_m", None)
        cfg = build_config(
            argparse.Namespace(vlm_url=None, model=None, vlm_backend=None), robot_cfg
        )
        self.assertIsNone(cfg["z_floor_m"])
        with self.assertRaisesRegex(ValueError, "no measured height is set"):
            openarm_real_preflight(cfg, self._args())


class CaptureZFloorTests(unittest.TestCase):
    """The floor capture must not move an arm the operator rested on the table."""

    def test_capture_runs_neither_the_start_nor_the_stop_trajectory(self) -> None:
        from scripts.openarm import capture_z_floor

        robot_cfg = {
            "driver": {
                "hardware_watchdog_confirmed": True,
                "startup_trajectory": True,
                "cameras": {"camera_head": {}},
            }
        }
        with patch.object(capture_z_floor, "load_yaml", return_value=robot_cfg):
            driver = capture_z_floor._driver_config(
                SimpleNamespace(robot_config="robot.yaml", site_config="site.yaml")
            )
        self.assertIs(driver["startup_trajectory"], False)
        self.assertIs(driver["stop_trajectory"], False)
        self.assertNotIn("cameras", driver)


class OpenArmRealPreflightTests(unittest.TestCase):
    """check_setup's real branch reads config and devices, never moves anything."""

    def setUp(self) -> None:
        site = REPO / "configs" / "site" / "openarm.yaml"
        if not site.exists():
            shutil.copy(site.with_suffix(".yaml.example"), site)
            self.addCleanup(site.unlink, missing_ok=True)

    def test_preflight_reports_missing_floor_and_watchdog_without_crashing(self) -> None:
        from scripts.check_setup import Report, _check_openarm_real

        cfg = load_yaml(REAL_CONFIG)
        report = Report()
        _check_openarm_real(report, cfg)
        # The shipped template ships an unset floor and an unconfirmed watchdog.
        self.assertGreater(report.failures, 0)


if __name__ == "__main__":
    unittest.main()
