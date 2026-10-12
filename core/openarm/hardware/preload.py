"""Bounded, one-shot preload estimation for OpenArm joint targets.

The hardware driver uses position PD commands without a torque feed-forward
channel.  A settled arm can therefore have a small, repeatable difference
between its last commanded arm joints and encoder feedback.  ``PreloadTracker``
can capture that difference once from a stable history so a Cartesian command
can preserve the existing support preload without continuously adapting to
feedback drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np

SIDES: Final = ("right", "left")
_JOINT_COUNT: Final = 8
_ARM_JOINT_COUNT: Final = 7


@dataclass(frozen=True)
class _Sample:
    timestamp: float
    measured: np.ndarray
    applied: np.ndarray


class PreloadTracker:
    """Capture one stable, bounded arm preload from measured/applied samples.

    Samples cover the configured time window plus one boundary sample.
    Stability is checked independently for measured and applied arm joints; the gripper
    channel is intentionally excluded from both checks and the returned bias.
    Once :meth:`capture` succeeds, its result is frozen and later samples do
    not adapt it.
    """

    def __init__(
        self,
        window: float = 0.3,
        stability_tolerance: float = 0.005,
        max_offset: float = 0.15,
    ) -> None:
        values = {}
        for name, value in (
            ("window", window),
            ("stability_tolerance", stability_tolerance),
            ("max_offset", max_offset),
        ):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and positive") from exc
            if isinstance(value, bool) or not np.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be finite and positive")
            values[name] = numeric
        self.window = values["window"]
        self.stability_tolerance = values["stability_tolerance"]
        self.max_offset = values["max_offset"]
        self._history: dict[str, deque[_Sample]] = {
            side: deque() for side in SIDES
        }
        self._captured: dict[str, np.ndarray] | None = None

    @staticmethod
    def _vector(value: object, name: str) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain eight finite joint values") from exc
        if vector.shape != (_JOINT_COUNT,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain eight finite joint values")
        return vector.copy()

    def add(
        self,
        side: str,
        measured: object,
        applied: object,
        monotonic_time: float,
    ) -> None:
        """Append one feedback/command sample for ``side``.

        ``monotonic_time`` must not move backwards for a given arm.  Samples
        older than the newest sample's window are discarded except for the
        one sample at or immediately before its left boundary.  That sample
        brackets a full window even when the sampling cadence has jitter.
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES!r}")
        try:
            timestamp = float(monotonic_time)
        except (TypeError, ValueError) as exc:
            raise ValueError("monotonic_time must be finite") from exc
        if isinstance(monotonic_time, bool) or not np.isfinite(timestamp):
            raise ValueError("monotonic_time must be finite")
        measured_vector = self._vector(measured, f"{side} measured")
        applied_vector = self._vector(applied, f"{side} applied")

        history = self._history[side]
        if history and timestamp < history[-1].timestamp:
            raise ValueError(
                f"{side} monotonic_time moved backwards: "
                f"{timestamp} < {history[-1].timestamp}"
            )
        if history and timestamp - history[-1].timestamp > self.window:
            history.clear()
        sample = _Sample(timestamp, measured_vector, applied_vector)
        if history and timestamp == history[-1].timestamp:
            history[-1] = sample
        else:
            history.append(sample)
        cutoff = timestamp - self.window
        # Retain just one sample at/before the left boundary.  Dropping every
        # sample older than cutoff would usually leave a span slightly below
        # window and reject capture forever under an irregular real clock.
        while len(history) > 1 and history[1].timestamp <= cutoff:
            history.popleft()

    @staticmethod
    def _copy_biases(biases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {side: value.copy() for side, value in biases.items()}

    def _capture_side(self, side: str) -> np.ndarray:
        history = self._history[side]
        if not history:
            raise ValueError(f"preload capture requires samples for {side}")

        span = history[-1].timestamp - history[0].timestamp
        epsilon = max(1e-12, self.window * 1e-9)
        if span + epsilon < self.window:
            raise ValueError(
                f"{side} preload history is insufficient: covers {span:.6f}s, "
                f"requires {self.window:.6f}s"
            )

        measured = np.stack(
            [sample.measured[:_ARM_JOINT_COUNT] for sample in history]
        )
        applied = np.stack(
            [sample.applied[:_ARM_JOINT_COUNT] for sample in history]
        )
        measured_range = np.ptp(measured, axis=0)
        applied_range = np.ptp(applied, axis=0)
        tolerance = self.stability_tolerance
        if np.any(measured_range > tolerance):
            joints = np.flatnonzero(measured_range > tolerance).tolist()
            raise ValueError(
                f"{side} measured arm joints are not stable at indices {joints}: "
                f"max range {float(np.max(measured_range)):.6f} > {tolerance:.6f}"
            )
        if np.any(applied_range > tolerance):
            joints = np.flatnonzero(applied_range > tolerance).tolist()
            raise ValueError(
                f"{side} applied arm joints are not stable at indices {joints}: "
                f"max range {float(np.max(applied_range)):.6f} > {tolerance:.6f}"
            )

        offsets = applied - measured
        absolute_offsets = np.max(np.abs(offsets), axis=0)
        if np.any(absolute_offsets > self.max_offset):
            joints = np.flatnonzero(absolute_offsets > self.max_offset).tolist()
            raise ValueError(
                f"{side} preload exceeds max_offset at indices {joints}: "
                f"max {float(np.max(absolute_offsets)):.6f} > {self.max_offset:.6f}"
            )

        # Average a stable window to suppress encoder quantization noise.  The
        # jaw is deliberately zero even if its command and feedback differ.
        bias = np.zeros(_JOINT_COUNT, dtype=float)
        bias[:_ARM_JOINT_COUNT] = np.mean(offsets, axis=0)
        return bias

    def preview(self) -> dict[str, np.ndarray]:
        """Validate the current history without freezing or changing it."""
        if self._captured is not None:
            return self._copy_biases(self._captured)
        return {side: self._capture_side(side) for side in SIDES}

    def capture(self) -> dict[str, np.ndarray]:
        """Return a frozen, copied preload bias for both arms.

        The first successful capture validates both histories.  A later call
        returns the same values even if new samples have been added.
        """
        if self._captured is None:
            captured = self.preview()
            self._captured = self._copy_biases(captured)
        return self._copy_biases(self._captured)
