"""OpenArm backend integration tests inject cameras and CAN workers; no hardware I/O."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import core.openarm.hardware.cameras as camera_module
import core.openarm.hardware.dispatch as dispatch_module
import core.openarm.hardware.backend as hardware


class OpenArmHardwareIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.camera_config = {
            "camera_wrist_left": {"vendor": "04b4", "serial": "CAM_LEFT"},
            "camera_wrist_right": {"vendor": "04b4", "usb_port": "3-3"},
            "camera_head": {"vendor": "2b03"},
        }
        self.config = {
            "hardware_watchdog_confirmed": True,
            "cameras": self.camera_config,
        }
        self.frames = {name: f"jpeg:{name}".encode() for name in self.camera_config}
        self.timestamps = dict(zip(self.camera_config, (101.1, 101.2, 101.3)))
        self.metadata = {
            "camera_info": {
                "source": "hardware_rgb",
                "depth_available": False,
                "hardware_synchronized": False,
                "cameras": dict.fromkeys(self.camera_config, {}),
            }
        }
        self.camera = Mock()
        self.camera.names = tuple(self.camera_config)
        self.camera.capture_times = dict(self.timestamps)
        self.camera.start.side_effect = lambda: self.events.append("camera.start")
        self.camera.capture.side_effect = lambda: (
            self.events.append("camera.capture") or dict(self.frames)
        )
        self.camera.observation_metadata.return_value = self.metadata
        self.camera.close.side_effect = lambda: self.events.append("camera.close")
        self.dispatch = Mock()
        self.dispatch.start.side_effect = lambda: self.events.append("dispatch.start")
        self.dispatch.close.side_effect = lambda: self.events.append("dispatch.close")
        self.right = np.arange(8, dtype=float) / 20
        self.left = -np.arange(8, dtype=float) / 30
        self.samples = {
            "right": SimpleNamespace(qpos=self.right, captured_at=99.6),
            "left": SimpleNamespace(qpos=self.left, captured_at=99.2),
        }
        self.dispatch.samples.return_value = self.samples
        self.applied = {
            "right": np.r_[np.ones(7), -0.15],
            "left": np.r_[np.ones(7), 0.25],
        }
        self.dispatch.applied_targets.return_value = self.applied
        self.dispatch.preload_bias.return_value = {
            "right": np.zeros(8), "left": np.zeros(8),
        }
        self.setup, self.kin = object(), Mock()
        self.poses = np.array([[0.2, -0.1, 0, 1, 0, 0, 0], [0.2, 0.1, 0, 1, 0, 0, 0]])
        self.kin.fk_bimanual.return_value = self.poses
        self.camera_factory = self.patch("OpenArmCameras", return_value=self.camera)
        self.dispatch_factory = self.patch(
            "BimanualDispatcher", return_value=self.dispatch
        )
        self.kinematics_factory = self.patch(
            "create_kinematics", return_value=(self.setup, self.kin)
        )
        self.patch("openarm_pedestal_xml", return_value="/model/pedestal.xml")

    def patch(self, name, **kwargs):
        patcher = patch.object(hardware, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def backend(self, **overrides):
        return hardware.OpenArmPedestalBackend({**self.config, **overrides})

    def fail(self, event, error):
        def failure(*_args, **_kwargs):
            self.events.append(event)
            raise error

        return failure

    def test_constructor_configures_resources_but_never_starts_io(self):
        backend = self.backend(
            control_hz=50,
            max_camera_age_seconds=0.3,
            camera_startup_timeout_seconds=4,
            camera_close_timeout_seconds=0.5,
            jpeg_quality=80,
            kinematics_xml="/custom/pedestal.xml",
        )
        self.assertEqual(self.events, [])
        self.assertIs(backend.setup, self.setup)
        self.assertIs(backend.kin, self.kin)
        self.assertEqual(backend.cameras, tuple(self.camera_config))
        self.camera.start.assert_not_called()
        self.dispatch.start.assert_not_called()
        self.camera_factory.assert_called_once_with(
            self.camera_config,
            max_age_seconds=0.3,
            jpeg_quality=80,
            startup_timeout=4,
            close_timeout=0.5,
        )
        args = self.kinematics_factory.call_args.args
        self.assertEqual(args[:2], ("/custom/pedestal.xml", 50))
        self.assertEqual(self.dispatch_factory.call_args.args[0], args[2])
        self.assertTrue(backend.hardware_watchdog)
        self.assertEqual((backend.mode, backend.frame), ("real", "arm_origin"))

    def test_camera_start_failure_never_starts_can_and_cleans_both_resources(self):
        original = RuntimeError("USB camera missing")
        self.camera.start.side_effect = self.fail("camera.start", original)
        backend = self.backend()
        with self.assertRaises(RuntimeError) as caught:
            backend.start()
        self.assertIs(caught.exception, original)
        self.dispatch.start.assert_not_called()
        self.camera.capture.assert_not_called()
        self.assertEqual(
            self.events, ["camera.start", "dispatch.close", "camera.close"]
        )
        self.assertFalse(backend._started)

    def test_can_start_failure_closes_camera_after_control_cleanup(self):
        original = RuntimeError("CAN carrier down")
        self.dispatch.start.side_effect = self.fail("dispatch.start", original)
        backend = self.backend()
        with self.assertRaises(RuntimeError) as caught:
            backend.start()
        self.assertIs(caught.exception, original)
        self.assertEqual(
            self.events,
            ["camera.start", "dispatch.start", "dispatch.close", "camera.close"],
        )
        self.assertFalse(backend._started)

    def test_start_checks_frames_after_can_start_and_rejects_duplicate_start(self):
        backend = self.backend()
        backend.start()
        self.assertEqual(
            self.events, ["camera.start", "dispatch.start", "camera.capture"]
        )
        self.assertTrue(backend._started)
        with self.assertRaisesRegex(RuntimeError, "already started"):
            backend.start()
        self.assertEqual(self.dispatch.start.call_count, 1)

    def test_camera_failure_after_can_start_disarms_and_preserves_primary_error(self):
        original = RuntimeError("camera frame is stale")
        self.camera.capture.side_effect = self.fail("camera.capture", original)
        self.dispatch.close.side_effect = self.fail(
            "dispatch.close", RuntimeError("disable timed out")
        )
        backend = self.backend()
        with self.assertRaises(RuntimeError) as caught:
            backend.start()
        self.assertIs(caught.exception, original)
        self.assertEqual(
            self.events,
            [
                "camera.start",
                "dispatch.start",
                "camera.capture",
                "dispatch.close",
                "camera.close",
            ],
        )
        self.assertIn("disable timed out", " ".join(original.__notes__))
        self.assertFalse(backend._started)

    def test_three_camera_frames_metadata_and_original_times_are_forwarded(self):
        backend = self.backend()
        self.assertEqual(backend.capture(), self.frames)
        self.assertIs(backend.observation_metadata(), self.metadata)
        self.assertEqual(backend.capture_times, self.timestamps)
        times = backend.capture_times
        times["camera_head"] = 200
        self.assertEqual(backend.capture_times["camera_head"], 101.3)
        self.camera.capture.side_effect = RuntimeError("camera stopped")
        with self.assertRaisesRegex(RuntimeError, "camera stopped"):
            backend.capture()

    def test_read_state_uses_measured_joints_and_oldest_encoder_source_time(self):
        backend = self.backend()
        state = backend.read_state()
        self.assertEqual(state.source_time, 99.2)
        self.assertEqual(state.joints_right, self.right.tolist())
        self.assertEqual(state.joints_left, self.left.tolist())
        self.assertEqual(state.right, self.poses[0].tolist())
        self.assertEqual(state.left, self.poses[1].tolist())
        args = self.kin.fk_bimanual.call_args.args
        np.testing.assert_array_equal(args[0], self.right)
        np.testing.assert_array_equal(args[1], self.left)
        self.dispatch.samples.side_effect = RuntimeError("stale encoder")
        with self.assertRaisesRegex(RuntimeError, "stale encoder"):
            backend.read_state()

    def test_command_seeds_ik_from_both_measured_arms_then_dispatches_solution(self):
        backend = self.backend()
        target_right, target_left = (
            [0.3, -0.2, 0, 1, 0, 0, 0],
            [0.3, 0.2, 0, 1, 0, 0, 0],
        )
        grippers = {"right": -0.2, "left": 0.3}
        solved = np.arange(16, dtype=float) / 100
        with patch.object(hardware, "solve_bimanual", return_value=solved) as solve:
            backend.command(target_right, target_left, grippers)
        args = solve.call_args.args
        self.assertIs(args[0], self.kin)
        np.testing.assert_array_equal(args[1], np.r_[self.right, self.left])
        self.assertEqual(args[2:], (target_right, target_left, grippers))
        np.testing.assert_array_equal(
            self.dispatch.command.call_args.args[0], solved[:8]
        )
        np.testing.assert_array_equal(
            self.dispatch.command.call_args.args[1], solved[8:]
        )
        self.assertEqual(backend.gripper_targets, {"right": -0.15, "left": 0.25})
        copied = backend.gripper_targets
        copied["left"] = 0.7
        self.assertEqual(backend.gripper_targets["left"], 0.25)

    def test_failed_ik_never_submits_or_changes_applied_gripper_targets(self):
        backend = self.backend()
        with patch.object(
            hardware, "solve_bimanual", side_effect=RuntimeError("IK failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "IK failed"):
                backend.command([0] * 7, [0] * 7, {"right": 0, "left": 0})
        self.dispatch.command.assert_not_called()
        self.assertEqual(backend.gripper_targets, {"right": -0.15, "left": 0.25})

    def test_readiness_is_read_only_and_missing_preload_never_runs_ik(self):
        backend = self.backend()
        backend.validate_motion_ready()
        self.dispatch.preload_bias.assert_called_once_with(capture=False)
        self.dispatch.preload_bias.side_effect = ValueError("preload not stable")
        with patch.object(hardware, "solve_bimanual") as solve:
            with self.assertRaisesRegex(ValueError, "not stable"):
                backend.command([0] * 7, [0] * 7, {"right": 0, "left": 0})
        solve.assert_not_called()
        self.dispatch.command.assert_not_called()

    def test_step_and_hold_delegate_without_camera_io_or_ik(self):
        backend = self.backend()
        with patch.object(
            hardware, "solve_bimanual", side_effect=AssertionError("hold ran IK")
        ):
            backend.step(1 / 60)
            backend.hold()
        self.dispatch.heartbeat.assert_called_once_with()
        self.dispatch.hold.assert_called_once_with()
        self.camera.capture.assert_not_called()

    def test_close_disarms_before_camera_cleanup_even_if_both_fail(self):
        backend = self.backend()
        backend.start()
        self.events.clear()
        control_error = RuntimeError("CAN cleanup failed")
        self.dispatch.close.side_effect = self.fail("dispatch.close", control_error)
        self.camera.close.side_effect = self.fail(
            "camera.close", RuntimeError("camera cleanup timed out")
        )
        with self.assertRaises(RuntimeError) as caught:
            backend.close()
        self.assertIs(caught.exception, control_error)
        self.assertEqual(self.events, ["dispatch.close", "camera.close"])
        self.assertFalse(backend._started)
        self.assertIn("camera cleanup timed out", " ".join(control_error.__notes__))

    def test_control_cleanup_can_run_independently_of_camera_shutdown(self):
        backend = self.backend()
        backend.start()
        self.events.clear()
        backend.close_control()
        self.assertEqual(self.events, ["dispatch.close"])
        self.assertFalse(backend._started)
        self.camera.close.assert_not_called()
        backend.close_camera()
        self.assertEqual(self.events, ["dispatch.close", "camera.close"])

    def test_legacy_empty_camera_list_and_invalid_options(self):
        self.backend(cameras=[])
        self.assertEqual(self.camera_factory.call_args.args, ({},))
        for changes in (
            {"hardware_watchdog_confirmed": 1},
            {"hardware_watchdog_confirmed": False},
            {"profile": "openarm_cell"},
            {"driver_config": ""},
            {"kinematics_xml": None},
            {"unknown": "option"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.backend(**changes)


class OpenArmShippedConfigTests(unittest.TestCase):
    """The shipped real-rig YAML builds the backend with no device I/O."""

    def _resolved_driver(self) -> tuple[dict, dict]:
        from core.config import load_yaml

        root = Path(__file__).resolve().parents[1]
        # The site layer is required by the shipped config; a fresh clone ships only the
        # example. Copy it in for this test run, then resolve the real config.
        site = root / "configs" / "site" / "openarm.yaml"
        if not site.exists():
            site.write_text(
                (root / "configs" / "site" / "openarm.yaml.example").read_text()
            )
            self.addCleanup(site.unlink, missing_ok=True)
        cfg = load_yaml(root / "configs" / "robot_openarm_real.yaml")
        driver = dict(cfg["driver"])
        cameras = dict(driver.pop("cameras", {}) or {})
        return driver, cameras

    def test_shipped_config_constructs_with_real_model_and_no_device_io(self):
        driver, cameras = self._resolved_driver()
        names = tuple(cameras)
        self.assertIs(driver["hardware_watchdog_confirmed"], False)
        config = {**driver, "hardware_watchdog_confirmed": True}
        with (
            patch.object(
                dispatch_module,
                "make_profile",
                side_effect=AssertionError("opened vendor profile"),
            ),
            patch.object(
                dispatch_module,
                "make_driver",
                side_effect=AssertionError("opened CAN"),
            ),
            patch.object(
                camera_module,
                "video_devices",
                side_effect=AssertionError("inspected camera devices"),
            ),
            patch.object(
                camera_module.threading.Thread,
                "start",
                side_effect=AssertionError("started worker"),
            ),
        ):
            backend = hardware.OpenArmPedestalBackend({**config, "cameras": cameras})
        self.assertEqual(backend.cameras, names)
        self.assertEqual(tuple(backend.setup.sides), ("right", "left"))
        self.assertFalse(backend._started)
        self.assertEqual(backend._dispatch._threads, [])
        self.assertEqual(backend._camera_group._workers, {})
        self.assertEqual(backend._dispatch.send_hz, 200)
        self.assertEqual(backend._dispatch.cutoff_hz, 5)


if __name__ == "__main__":
    unittest.main()
