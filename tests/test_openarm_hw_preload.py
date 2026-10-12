"""Tests for one-shot OpenArm preload estimation."""

from __future__ import annotations

import unittest

import numpy as np

from core.openarm.hardware.preload import PreloadTracker


class OpenArmPreloadTests(unittest.TestCase):
    def add_window(
        self,
        tracker: PreloadTracker,
        *,
        measured_right=None,
        applied_right=None,
        measured_left=None,
        applied_left=None,
        jaw_delta: float = 0.0,
        times=(0.0, 0.1, 0.2, 0.3),
    ):
        measured_right = (
            np.zeros(8)
            if measured_right is None
            else np.asarray(measured_right, dtype=float)
        )
        measured_left = (
            np.zeros(8)
            if measured_left is None
            else np.asarray(measured_left, dtype=float)
        )
        applied_right = (
            np.zeros(8)
            if applied_right is None
            else np.asarray(applied_right, dtype=float)
        )
        applied_left = (
            np.zeros(8)
            if applied_left is None
            else np.asarray(applied_left, dtype=float)
        )
        for timestamp in times:
            right_applied = applied_right.copy()
            left_applied = applied_left.copy()
            right_applied[7] += jaw_delta * timestamp
            left_applied[7] -= jaw_delta * timestamp
            tracker.add("right", measured_right, right_applied, timestamp)
            tracker.add("left", measured_left, left_applied, timestamp)

    def test_captures_stable_sag_and_zeros_jaw(self):
        tracker = PreloadTracker()
        measured_right = np.r_[np.linspace(0.1, 0.7, 7), -0.7]
        measured_left = np.r_[np.linspace(-0.1, -0.7, 7), 0.7]
        bias_right = np.r_[np.array([0.03, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01]), 0]
        bias_left = -bias_right
        self.add_window(
            tracker,
            measured_right=measured_right,
            applied_right=measured_right + bias_right,
            measured_left=measured_left,
            applied_left=measured_left + bias_left,
        )
        captured = tracker.capture()
        np.testing.assert_allclose(captured["right"], bias_right)
        np.testing.assert_allclose(captured["left"], bias_left)

    def test_moving_applied_target_is_rejected(self):
        tracker = PreloadTracker()
        for timestamp in (0.0, 0.1, 0.2, 0.3):
            applied = np.zeros(8)
            applied[2] = timestamp * 0.1
            tracker.add("right", np.zeros(8), applied, timestamp)
            tracker.add("left", np.zeros(8), np.zeros(8), timestamp)
        with self.assertRaisesRegex(ValueError, "applied arm joints are not stable"):
            tracker.capture()

    def test_moving_measurement_is_rejected(self):
        tracker = PreloadTracker()
        for timestamp in (0.0, 0.1, 0.2, 0.3):
            measured = np.zeros(8)
            measured[1] = timestamp * 0.1
            tracker.add("right", measured, np.zeros(8), timestamp)
            tracker.add("left", np.zeros(8), np.zeros(8), timestamp)
        with self.assertRaisesRegex(ValueError, "measured arm joints are not stable"):
            tracker.capture()

    def test_insufficient_history_is_rejected(self):
        tracker = PreloadTracker()
        for timestamp in (0.0, 0.1, 0.2):
            tracker.add("right", np.zeros(8), np.zeros(8), timestamp)
            tracker.add("left", np.zeros(8), np.zeros(8), timestamp)
        with self.assertRaisesRegex(ValueError, "history is insufficient"):
            tracker.capture()

    def test_jaw_changes_do_not_affect_stability_or_bias(self):
        tracker = PreloadTracker()
        self.add_window(tracker, jaw_delta=10.0)
        captured = tracker.capture()
        np.testing.assert_array_equal(captured["right"][7], 0.0)
        np.testing.assert_array_equal(captured["left"][7], 0.0)

    def test_offset_bound_is_checked_only_for_arm_joints(self):
        tracker = PreloadTracker(max_offset=0.15)
        measured = np.zeros(8)
        applied = np.zeros(8)
        applied[3] = 0.151
        applied[7] = 100.0
        self.add_window(
            tracker,
            measured_right=measured,
            applied_right=applied,
            measured_left=measured,
            applied_left=np.zeros(8),
        )
        with self.assertRaisesRegex(ValueError, "preload exceeds max_offset"):
            tracker.capture()

    def test_capture_requires_both_sides_and_returns_copies(self):
        tracker = PreloadTracker()
        for timestamp in (0.0, 0.1, 0.2, 0.3):
            tracker.add("right", np.zeros(8), np.zeros(8), timestamp)
        with self.assertRaisesRegex(ValueError, "requires samples for left"):
            tracker.capture()

        tracker = PreloadTracker()
        self.add_window(tracker)
        first = tracker.capture()
        first["right"][0] = 99.0
        second = tracker.capture()
        self.assertEqual(second["right"][0], 0.0)

    def test_irregular_cadence_retains_a_full_window(self):
        tracker = PreloadTracker()
        self.add_window(tracker, times=(0.0, 0.09, 0.17, 0.26, 0.34))
        captured = tracker.capture()
        np.testing.assert_array_equal(captured["right"], np.zeros(8))

    def test_preview_does_not_freeze_and_sampling_gap_requires_new_window(self):
        tracker = PreloadTracker()
        self.add_window(tracker)
        np.testing.assert_array_equal(tracker.preview()["right"], np.zeros(8))
        for side in ("right", "left"):
            tracker.add(side, np.zeros(8), np.zeros(8), 2.0)
        with self.assertRaisesRegex(ValueError, "history is insufficient"):
            tracker.capture()
        self.add_window(
            tracker,
            applied_right=np.r_[np.full(7, .03), 0],
            applied_left=np.r_[np.full(7, -.03), 0],
            times=(2.1, 2.2, 2.3, 2.4, 2.5),
        )
        np.testing.assert_allclose(tracker.capture()["right"][:7], .03)

    def test_old_unstable_samples_expire_after_a_stable_window(self):
        tracker = PreloadTracker()
        moving = np.r_[np.full(7, 0.1), 0.0]
        for side in ("right", "left"):
            tracker.add(side, moving, moving, 0.0)
        self.add_window(tracker, times=(0.07, 0.16, 0.25, 0.34, 0.43))
        captured = tracker.capture()
        np.testing.assert_array_equal(captured["right"], np.zeros(8))

    def test_inputs_are_copied_and_backward_time_rejected(self):
        tracker = PreloadTracker()
        measured = np.zeros(8)
        applied = np.r_[np.full(7, 0.03), 0.0]
        for side in ("right", "left"):
            tracker.add(side, measured, applied, 0.0)
        measured[:] = 99.0
        applied[:] = 99.0
        self.add_window(
            tracker,
            applied_right=np.r_[np.full(7, 0.03), 0.0],
            applied_left=np.r_[np.full(7, 0.03), 0.0],
            times=(0.1, 0.2, 0.3),
        )
        captured = tracker.capture()
        np.testing.assert_allclose(captured["right"][:7], 0.03)
        with self.assertRaisesRegex(ValueError, "monotonic_time moved backwards"):
            tracker.add("right", np.zeros(8), np.zeros(8), 0.1)

    def test_capture_is_frozen_after_later_samples(self):
        tracker = PreloadTracker()
        self.add_window(
            tracker,
            applied_right=np.r_[np.full(7, 0.03), 0.0],
            applied_left=np.r_[np.full(7, -0.03), 0.0],
        )
        first = tracker.capture()
        for timestamp in (0.4, 0.5, 0.6):
            tracker.add("right", np.zeros(8), np.r_[np.full(7, 0.1), 0.0], timestamp)
            tracker.add("left", np.zeros(8), np.r_[np.full(7, -0.1), 0.0], timestamp)
        second = tracker.capture()
        np.testing.assert_allclose(second["right"], first["right"])
        np.testing.assert_allclose(second["left"], first["left"])


if __name__ == "__main__":
    unittest.main()
