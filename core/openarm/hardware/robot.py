"""OpenArm2 Pedestal robot: the duck type :class:`RealAtomicController` drives.

This is the real-rig twin of :class:`core.openarm.openarm_interface.OpenArmSimRobot`.
Both implement exactly the four methods ``RealAtomicController`` calls
(``get_ee_pose`` / ``get_gripper_position`` / ``control_gripper`` /
``update_desired_ee_pose``), so the setpoint integration, per-command clamps, Z floor,
gripper settle, empty-grasp detection and every plugin run here unchanged.

Differences from the simulator wrapper, all of them properties of hardware:

* ``update_desired_ee_pose`` resolves one Cartesian target to joint targets **once** per
  call. The arm itself runs a 200 Hz filtered position loop, and ``RealAtomicController``
  already re-commands each setpoint ``settle_steps`` times, so no physics is stepped here.
* The jaw is a position servo with no width encoder. ``get_gripper_position`` converts the
  measured finger joint angle through the same geometry the MJCF model uses (measured from
  the official model: 0 rad closed, then a monotone opening), because the controller's
  empty-grasp and close/open discriminators work in metres.
* The other arm is held at a latched pose, exactly like the simulator: one Show-Harness
  rollout drives one arm and must not disturb the other. The session re-latches it with
  :meth:`OpenArmRealRobot.sync_hold` once the support preload has settled -- the pose read
  at construction is taken right after enable, before the arm has sagged under gravity.

Frames and units are identical to the simulator path:
``arm_origin`` metres with the Show-Harness ``[x,y,z,qx,qy,qz,qw]`` quaternion order; the
upstream ``openarm_control`` API reports ``[x,y,z,qw,qx,qy,qz]`` and the conversions
happen only here.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

#: Finger joint travel, radians (both arms; the sign is mirrored). 0 is CLOSED.
_GRIPPER_MAX_RAD = 0.7854
_GRIPPER_CLOSED_RAD = 0.0
_GRIPPER_OPEN_RAD = {"left": _GRIPPER_MAX_RAD, "right": -_GRIPPER_MAX_RAD}

#: Finger-pad half-thicknesses summed along the jaw-normal axis, from the official
#: collision geoms; converts the kinematic pad-centre separation into a free gap.
_PAD_HALF_THICKNESS_SUM_M = 0.00995

#: Cubic fit of pad-centre separation vs. |finger joint angle| (radians), measured on the
#: official Pedestal MJCF at 41 samples: max error 0.011 mm. The linkage is non-linear, so
#: a linear ratio would overstate the opening near the closed end -- exactly where the
#: empty-grasp test decides.
_PAD_CENTRE_FIT = (0.009125916702, 0.1917855893, 0.0124754596, -0.03093486746)

#: Free gap at full travel; used only to adopt the measured state at startup.
_MAX_FREE_GAP_M = 0.14252

#: IK failures to report before going quiet (a stuck arm would otherwise flood a rollout).
_MAX_IK_REFUSAL_WARNINGS = 5


def _wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    """``[x,y,z,qw,qx,qy,qz]`` -> ``[x,y,z,qx,qy,qz,qw]`` (Show-Harness convention)."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    return np.asarray([p[0], p[1], p[2], p[4], p[5], p[6], p[3]], dtype=float)


def _xyzw_to_wxyz(pose: np.ndarray) -> np.ndarray:
    """``[x,y,z,qx,qy,qz,qw]`` -> ``[x,y,z,qw,qx,qy,qz]`` (upstream convention)."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    return np.asarray([p[0], p[1], p[2], p[6], p[3], p[4], p[5]], dtype=float)


def free_gap_m(finger_angle_rad: float) -> float:
    """Free opening (metres) between the pads for a measured finger joint angle.

    The magnitude is what matters -- the joint sign is mirrored between the arms.
    """
    angle = abs(float(finger_angle_rad))
    centre = sum(
        coefficient * angle**power
        for power, coefficient in enumerate(_PAD_CENTRE_FIT)
    )
    return max(0.0, min(_MAX_FREE_GAP_M, centre - _PAD_HALF_THICKNESS_SUM_M))


class OpenArmRealRobot:
    """The four-method robot duck type, backed by the OpenArm2 Pedestal CAN backend."""

    LOG_TAG = "openarm-real"

    def __init__(
        self,
        backend: Any,
        side: str = "right",
        *,
        verbose: bool = True,
    ) -> None:
        side = str(side).strip().lower()
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.backend = backend
        self.side = side
        self.other_side = "left" if side == "right" else "right"
        self.verbose = bool(verbose)
        # Gripper joint targets (radians), one per arm; the untouched arm keeps the
        # measured value so holding it cannot unload its jaw.
        measured = self._joint_positions()
        self._gripper_rad = {
            self.side: float(measured[self.side][7]),
            self.other_side: float(measured[self.other_side][7]),
        }
        self._gripper_closed = (
            free_gap_m(self._gripper_rad[self.side]) <= 0.5 * _MAX_FREE_GAP_M
        )
        # The other arm's held control-point pose, latched once at construction.
        self._other_arm_pose: Optional[np.ndarray] = None
        self._ik_refusals = 0
        self.sync_hold()

    # -- helpers -----------------------------------------------------------
    def _joint_positions(self) -> dict[str, np.ndarray]:
        """Measured eight-slot driver joints per arm from the CAN backend."""
        state = self.backend.read_state()
        return {
            "right": np.asarray(state.joints_right, dtype=float).copy(),
            "left": np.asarray(state.joints_left, dtype=float).copy(),
        }

    def _upstream_pose(self, side: str) -> np.ndarray:
        state = self.backend.read_state()
        return np.asarray(
            state.right if side == "right" else state.left, dtype=float
        ).copy()

    def sync_hold(self) -> None:
        """Latch the untouched arm's measured pose and adopt the jaw state."""
        self._other_arm_pose = self._upstream_pose(self.other_side)
        measured = self._joint_positions()
        self._gripper_rad = {
            self.side: float(measured[self.side][7]),
            self.other_side: float(measured[self.other_side][7]),
        }
        width = self.get_gripper_position()[0]
        self._gripper_closed = width <= 0.5 * _MAX_FREE_GAP_M

    # -- the duck type -----------------------------------------------------
    def get_ee_pose(self) -> np.ndarray:
        """Driven arm's control-point pose, ``[x,y,z,qx,qy,qz,qw]`` in ``arm_origin``."""
        return _wxyz_to_xyzw(self._upstream_pose(self.side))

    def get_gripper_position(self) -> np.ndarray:
        """Free opening between the finger pads, metres (0 == fully closed)."""
        angle = float(self._joint_positions()[self.side][7])
        return np.asarray([free_gap_m(angle)], dtype=float)

    def control_gripper(self, close: bool) -> None:
        """Drive the jaw to its closed/open joint target.

        Returns as soon as the target has been dispatched; ``RealAtomicController``'s
        settle polling watches the measured width that follows.
        """
        self._gripper_rad[self.side] = (
            _GRIPPER_CLOSED_RAD if close else _GRIPPER_OPEN_RAD[self.side]
        )
        self.backend.command_grippers(dict(self._gripper_rad))
        self._gripper_closed = bool(close)
        if self.verbose:
            state = "CLOSE" if close else "OPEN"
            width = float(self.get_gripper_position()[0]) * 1000.0
            print(f"[{self.LOG_TAG}] gripper -> {state} ({width:.1f}mm)")

    def update_desired_ee_pose(self, pose7: np.ndarray) -> None:
        """Resolve one absolute Cartesian setpoint to joint targets and dispatch it.

        ``pose7`` is ``[x,y,z,qx,qy,qz,qw]``. The untouched arm is commanded back to its
        latched pose in the same bimanual solve, which is what keeps it still. A refused
        IK solve is reported and skipped rather than raised: the controller only ever
        sends clamped single steps, and failing the rollout on one local-solver miss would
        be worse than holding for that tick. The Z floor and the step budget still bound
        the motion.
        """
        target = _xyzw_to_wxyz(pose7)
        assert self._other_arm_pose is not None  # set in __init__/sync_hold
        right, left = (
            (target, self._other_arm_pose)
            if self.side == "right"
            else (self._other_arm_pose, target)
        )
        try:
            self.backend.command(right, left, dict(self._gripper_rad))
        except RuntimeError as exc:
            self._report_ik_refusal(target, exc)

    def _report_ik_refusal(self, target: np.ndarray, exc: BaseException) -> None:
        self._ik_refusals += 1
        if self.verbose and self._ik_refusals <= _MAX_IK_REFUSAL_WARNINGS:
            print(
                f"[{self.LOG_TAG}] IK refused a setpoint "
                f"({np.round(np.asarray(target)[:3], 4).tolist()}): {exc}; "
                "holding this step"
            )

    # -- extras used by the session ---------------------------------------
    @property
    def gripper_closed(self) -> bool:
        """Whether the last commanded gripper state was CLOSED."""
        return bool(self._gripper_closed)

    def close(self) -> None:
        """Stop the arm workers, run the vendor stop trajectory and disable the motors."""
        self.backend.close()
