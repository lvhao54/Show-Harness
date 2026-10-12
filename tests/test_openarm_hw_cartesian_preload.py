"""Real OpenArm IK with a synthetic loaded servo; no camera or CAN I/O.

The plant has a fixed position-servo deflection: at equilibrium measured arm
joints equal the dispatched target minus the support bias.  The representative
measured configuration is independent of task recordings.  This exercises the
IK-to-dispatch boundary that mocks returning fixed IK solutions cannot cover.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import core.openarm.hardware.backend as hardware
from core.openarm.hardware.preload import PreloadTracker


SIDES = ("right", "left")
MEASURED = {
    "right": np.array([
        -0.0528343633, 0.0101091020, -0.0051499199, 1.4906157015,
        -0.0131609064, 0.0814450294, -0.0204089418, -0.7854,
    ]),
    "left": np.array([
        0.0528343633, -0.0078202487, 0.0051499199, 1.4906157015,
        0.0158312352, -0.0684748608, 0.0108720531, 0.7854,
    ]),
}


class LoadedServoDispatcher:
    """Deterministic first-order position servo with fixed gravity deflection."""

    def __init__(self):
        self.measured = {side: values.copy() for side, values in MEASURED.items()}
        # A supported nominal pose is different from its loaded measurement.
        supported = np.array([0, 0, 0, np.pi / 2, 0, 0, 0, 0], dtype=float)
        self.bias = {}
        self.applied = {}
        for side in SIDES:
            self.bias[side] = supported - self.measured[side]
            self.bias[side][-1] = 0  # The jaw is not an arm support channel.
            self.applied[side] = self.measured[side] + self.bias[side]
        self.sent = []
        self.time = 0.3
        self.tracker = PreloadTracker()
        for timestamp in (0.0, 0.1, 0.2, 0.3):
            for side in SIDES:
                self.tracker.add(
                    side, self.measured[side], self.applied[side], timestamp,
                )

    def start(self):
        pass

    def close(self):
        pass

    def samples(self):
        return {
            side: SimpleNamespace(qpos=values.copy(), captured_at=self.time)
            for side, values in self.measured.items()
        }

    def applied_targets(self):
        return {side: values.copy() for side, values in self.applied.items()}

    def preload_bias(self):
        return self.tracker.capture()

    def command(self, right, left):
        targets = dict(zip(SIDES, (right, left)))
        self.time += 1 / 60
        for side in SIDES:
            self.applied[side] = np.asarray(targets[side], dtype=float).copy()
            equilibrium = self.applied[side] - self.bias[side]
            # The servo moves only partway per tick, exposing lag as well as
            # gravity deflection; repeated feedback must not accumulate bias.
            self.measured[side] += 0.3 * (equilibrium - self.measured[side])
            self.tracker.add(
                side, self.measured[side], self.applied[side], self.time,
            )
        self.sent.append(self.applied_targets())

    def command_grippers(self, grippers):
        targets = self.applied_targets()
        for side in SIDES:
            targets[side][-1] = grippers[side]
        self.command(targets["right"], targets["left"])


class OpenArmCartesianPreloadTests(unittest.TestCase):
    def setUp(self):
        self.dispatch = LoadedServoDispatcher()
        self.camera = Mock()
        self.camera.names = ()
        self.camera.capture.return_value = {}
        # Keep the actual model and IK. Only the device resources are replaced.
        with patch.object(hardware, "BimanualDispatcher", return_value=self.dispatch), \
                patch.object(hardware, "OpenArmCameras", return_value=self.camera):
            self.backend = hardware.OpenArmPedestalBackend({
                "hardware_watchdog_confirmed": True,
            })
        self.backend.start()
        self.addCleanup(self.backend.close)
        self.initial = self.poses()
        self.supported = self.dispatch.applied_targets()
        self.grippers = {"right": -0.7854, "left": 0.7854}

    def poses(self):
        state = self.backend.read_state()
        return {"right": np.asarray(state.right), "left": np.asarray(state.left)}

    def tick(self, targets):
        self.backend.command(targets["right"], targets["left"], self.grippers)
        return self.poses()

    def assert_pose_near(self, measured, targets, position=0.002, orientation=0.02):
        for side in SIDES:
            with self.subTest(side=side):
                actual, desired = measured[side], targets[side]
                self.assertLess(np.linalg.norm(actual[:3] - desired[:3]), position)
                qa = actual[3:] / np.linalg.norm(actual[3:])
                qb = desired[3:] / np.linalg.norm(desired[3:])
                angle = 2 * np.arccos(np.clip(abs(np.dot(qa, qb)), 0, 1))
                self.assertLess(angle, orientation)

    def assert_bias_unchanged(self):
        captured = self.dispatch.preload_bias()
        for side in SIDES:
            np.testing.assert_allclose(captured[side], self.dispatch.bias[side], atol=1e-12)

    def test_zero_displacement_preserves_first_supported_command_and_does_not_sag(self):
        measured = self.tick(self.initial)
        for side in SIDES:
            # The real IK's nullspace regularization may move redundant joints
            # slightly, but it must not remove the ~0.05--0.08 rad support bias.
            np.testing.assert_allclose(
                self.dispatch.sent[0][side][:7], self.supported[side][:7], atol=0.0003,
            )
        for _ in range(120):
            measured = self.tick(self.initial)
            for side in SIDES:
                self.assertGreater(measured[side][2], self.initial[side][2] - 0.001)
        self.assert_pose_near(measured, self.initial)
        self.assert_bias_unchanged()

    def test_upward_25mm_trajectory_has_no_initial_drop_and_settles_without_bias_growth(self):
        targets = {side: pose.copy() for side, pose in self.initial.items()}
        for step in range(181):
            fraction = step / 180
            smooth = fraction * fraction * (3 - 2 * fraction)
            for side in SIDES:
                targets[side][2] = self.initial[side][2] + 0.025 * smooth
            measured = self.tick(targets)
            for side in SIDES:
                self.assertGreater(measured[side][2], self.initial[side][2] - 0.001)
        for _ in range(120):
            measured = self.tick(targets)
        self.assert_pose_near(measured, targets)
        settled = self.dispatch.applied_targets()
        for _ in range(120):
            measured = self.tick(targets)
        self.assert_pose_near(measured, targets)
        for side in SIDES:
            np.testing.assert_allclose(
                self.dispatch.applied[side][:7], settled[side][:7], atol=0.0005,
            )
        self.assert_bias_unchanged()

    def test_gripper_close_open_retains_preload_across_repeated_cartesian_commands(self):
        self.tick(self.initial)  # Capture the settled preload before jaw moves.
        for grip in (0.0, 0.7854, 0.0, 0.7854):
            before = self.dispatch.applied_targets()
            self.grippers = {"right": -grip, "left": grip}
            self.backend.command_grippers(self.grippers)
            for side in SIDES:
                np.testing.assert_array_equal(
                    self.dispatch.applied[side][:7], before[side][:7],
                )
                self.assertEqual(self.dispatch.applied[side][-1], self.grippers[side])
            for _ in range(30):
                measured = self.tick(self.initial)
                for side in SIDES:
                    self.assertAlmostEqual(
                        self.dispatch.applied[side][-1], self.grippers[side], places=6,
                    )
            self.assert_pose_near(measured, self.initial)
            self.assert_bias_unchanged()


if __name__ == "__main__":
    unittest.main()
