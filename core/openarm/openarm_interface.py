"""OpenArm MuJoCo robot: the duck type :class:`RealAtomicController` drives.

``RealAtomicController`` talks to a robot through exactly four methods
(``get_ee_pose`` / ``get_gripper_position`` / ``control_gripper`` /
``update_desired_ee_pose``). This class implements them against the upstream
``embodied_ai-llm`` MuJoCo backend, so the **entire** real-arm stack -- setpoint
integration, per-command clamps, the Z floor, gripper settle + empty-grasp detection, and
the ``smooth`` / ``variable_step`` / ``rotation`` plugin mounts -- runs unchanged in
simulation. Nothing about OpenArm's dynamics is reimplemented here.

Three OpenArm-specific facts drive the code below; each was measured against the model
rather than assumed:

1. **Frames.** Upstream reports poses for ``{side}_ee_control_point`` in the
   ``arm_origin`` site frame, as ``[x, y, z, qw, qx, qy, qz]``. Show-Harness uses
   ``[x, y, z, qx, qy, qz, qw]``, so every pose crossing this boundary is reordered.
   ``arm_origin`` sits at world ``z = 0.698``, so the frame's negative Z is *below the
   robot's own base*; the two conventions never mix because only this class converts.

2. **The control point is the wrist, not the TCP.** The upstream model's
   ``*_ee_control_point`` site sits at the wrist flange; the finger pads are a further
   **16.38 cm along the approach axis** (``finger_pad_midpoint_in_control_frame`` =
   ``[-0.0007, 0, -0.1638]``). Commands are therefore issued to the wrist site -- moving
   the *pads* onto an object means targeting the wrist 16.4 cm back along the approach
   axis. This is the single easiest thing to get wrong here: a token sequence that looks
   correct in the logs will grasp thin air if the offset is ignored.

3. **One command is not one simulation tick.** The upstream IK is a *damped local
   solver* (``ik_max_iters=5``): a single solve lands only part-way to a distant target,
   so the arm tracks properly only when the target is re-solved every control tick -- the
   same way ``embodied_ai-llm``'s runtime control thread drives it. Each
   :meth:`update_desired_ee_pose` here therefore re-solves and advances the physics by
   ``sim_ticks_per_command`` steps.

The gripper is a two-finger parallel jaw: each finger joint spans ``[-0.7854, 0]`` on the
right arm and ``[0, 0.7854]`` on the left (the sign is mirrored; the magnitude is not), and
an equality constraint drives the second finger from the first, so both move together.
``0`` is fully CLOSED and the magnitude's extreme is fully OPEN. Measured pad-centre
separation runs 9.13 mm closed -> 152.47 mm open, i.e. a 142.52 mm free gap -- a wide
stroke, which is why the 45 mm cubes in the default upstream scene are graspable but the
per-arm joint *signs* are the easy thing to invert.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Finger-pad half-thicknesses along the jaw-normal axis, summed. Subtracting this from the
# pad-centre separation converts a kinematic distance into the free opening a user thinks
# of as "gripper width". Read from the model's collision geoms
# (finger_{inner,outer}_{side}_collision_00, half-extent 0.00497/0.00498 m).
_PAD_HALF_THICKNESS_SUM_M = 0.00995

# Per-side gripper joint targets. `closed` is 0 for both; the OPEN sign follows the joint
# range, which is mirrored between the arms.
_GRIPPER_OPEN_RAD = {"left": 0.7854, "right": -0.7854}
_GRIPPER_CLOSED_RAD = 0.0

#: Measured free gap at full travel: pad-centre separation 152.47 mm minus the 9.95 mm of
#: pad half-thicknesses. Used only as the closed/open discriminator when adopting the
#: gripper's starting state.
_MAX_FREE_GAP_M = 0.14252

#: How many IK-refusal warnings to print before going quiet (a stuck arm can otherwise
#: flood the console once per step for the rest of the episode).
_MAX_IK_REFUSAL_WARNINGS = 5


def _wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    """``[x,y,z,qw,qx,qy,qz]`` -> ``[x,y,z,qx,qy,qz,qw]`` (Show-Harness convention)."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    return np.asarray([p[0], p[1], p[2], p[4], p[5], p[6], p[3]], dtype=float)


def _xyzw_to_wxyz(pose: np.ndarray) -> np.ndarray:
    """``[x,y,z,qx,qy,qz,qw]`` -> ``[x,y,z,qw,qx,qy,qz]`` (upstream convention)."""
    p = np.asarray(pose, dtype=float).reshape(-1)
    return np.asarray([p[0], p[1], p[2], p[6], p[3], p[4], p[5]], dtype=float)


class OpenArmSimRobot:
    """The four-method robot duck type, backed by the upstream OpenArm MuJoCo backend.

    Args:
        backend: an upstream ``robot_runtime.robots.openarm.simulation.SimulationBackend``
            (see :func:`core.openarm.sim.make_openarm_backend`), already started.
        side: which arm this controller drives -- ``"right"`` or ``"left"``. The other arm
            holds the pose it had at construction, so a single-arm Show-Harness loop never
            disturbs it.
        sim_ticks_per_command: physics ticks advanced per setpoint command. At the
            backend's 60 Hz control rate, 6 ticks == 0.1 s, which pairs with the real
            controller's ``settle_dt_s=0.1`` default: one atomic token then costs
            ``settle_steps * sim_ticks_per_command`` ticks of simulated time.
        gripper_ticks: physics ticks advanced per gripper command. The jaw is a position
            servo, so it needs a few ticks to travel and load up on an object.
        settle_ticks: ticks advanced before the first read after a reset/re-home.
        verbose: print a one-line summary per command.
    """

    LOG_TAG = "openarm-sim"

    def __init__(
        self,
        backend: Any,
        side: str = "right",
        sim_ticks_per_command: int = 6,
        gripper_ticks: int = 60,
        settle_ticks: int = 60,
        verbose: bool = True,
    ) -> None:
        side = str(side).strip().lower()
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.backend = backend
        self.side = side
        self.other_side = "left" if side == "right" else "right"
        self.sim_ticks_per_command = max(1, int(sim_ticks_per_command))
        self.gripper_ticks = max(1, int(gripper_ticks))
        self.settle_ticks = max(1, int(settle_ticks))
        self.verbose = bool(verbose)
        self.dt = 1.0 / float(getattr(backend, "control_hz", 60.0) or 60.0)

        # Gripper command latched here; the upstream backend keeps applying the last
        # target through every physics step, so a held width is stable without re-sending.
        self._gripper_rad = {self.side: _GRIPPER_CLOSED_RAD, self.other_side: _GRIPPER_CLOSED_RAD}
        self._gripper_closed = True
        # The other arm's held pose, refreshed on every sync so it never drifts.
        self._other_arm_pose: Optional[np.ndarray] = None
        self._last_command_s: float = 0.0
        self._ik_refusals = 0
        self.sync_hold()

    # -- lifecycle ---------------------------------------------------------
    def sync_hold(self) -> None:
        """Latch the untouched arm's current pose and settle the gripper at its measured state."""
        pose = self._upstream_pose(self.other_side)
        self._other_arm_pose = np.asarray(pose, dtype=float).copy()
        width = self.get_gripper_position()[0]
        self._gripper_closed = width <= 0.5 * self._max_free_gap_m()
        self._step(self.settle_ticks)

    def _step(self, ticks: int) -> None:
        for _ in range(max(0, int(ticks))):
            self.backend.step(self.dt)
        self._last_command_s = float(self.backend.read_state().source_time)

    # -- the duck type -----------------------------------------------------
    def _upstream_pose(self, side: str) -> np.ndarray:
        state = self.backend.read_state()
        return np.asarray(state.right if side == "right" else state.left, dtype=float)

    def get_ee_pose(self) -> np.ndarray:
        """Driven arm's control-point pose, ``[x,y,z,qx,qy,qz,qw]`` in ``arm_origin``.

        This is the *wrist* control point, not the finger pads -- see the module docstring.
        """
        return _wxyz_to_xyzw(self._upstream_pose(self.side))

    def get_gripper_position(self) -> np.ndarray:
        """Free opening between the finger pads, in metres (0 == fully closed)."""
        return np.asarray([self._free_gap_m()], dtype=float)

    def control_gripper(self, close: bool) -> None:
        """Drive the jaw to its closed/open joint target and let it settle."""
        self._gripper_rad[self.side] = (
            _GRIPPER_CLOSED_RAD if close else _GRIPPER_OPEN_RAD[self.side]
        )
        self.backend.command_grippers(dict(self._gripper_rad))
        self._step(self.gripper_ticks)
        self._gripper_closed = bool(close)
        if self.verbose:
            state = "CLOSE" if close else "OPEN"
            print(
                f"[{self.LOG_TAG}] gripper -> {state} "
                f"({self.get_gripper_position()[0] * 1000:.1f}mm)"
            )

    def update_desired_ee_pose(self, pose7: np.ndarray) -> None:
        """Re-solve IK toward one absolute setpoint and advance the physics.

        Mirrors the upstream runtime's control thread: the target is re-solved from the
        *measured* state every tick, because the damped local solver converges only
        incrementally. ``pose7`` is ``[x,y,z,qx,qy,qz,qw]``; the untouched arm is held at
        its latched pose.

        The upstream solver is also *local*: a target far from the current pose (or outside
        the reachable set) makes ``solve_bimanual`` return ``None`` and the runtime raises.
        A real arm never teleports, and Show-Harness's controller hands us setpoints that
        are one clamped step apart, so a single refusal is treated as "this step could not
        be solved" rather than a fatal error: the target is re-solved as a few interpolated
        waypoints, and a still-unreachable target is reported once and skipped, leaving the
        arm at its last good joint state. Failing the whole rollout on one refused IK solve
        would be worse than holding -- the Z floor and the episode's own step budget still
        bound the motion.
        """
        target = _xyzw_to_wxyz(pose7)
        assert self._other_arm_pose is not None  # set in __init__
        measured = self._upstream_pose(self.side)
        # Interpolate from the measured pose to the target across the ticks of this command,
        # so each individual solve starts near its target (the upstream solver's assumption).
        ticks = self.sim_ticks_per_command
        for i in range(1, ticks + 1):
            frac = i / float(ticks)
            blended = measured + (target - measured) * frac
            right, left = (
                (blended, self._other_arm_pose)
                if self.side == "right"
                else (self._other_arm_pose, blended)
            )
            try:
                self.backend.command(right, left, dict(self._gripper_rad))
            except RuntimeError as exc:
                self._ik_refusals += 1
                if self.verbose and self._ik_refusals <= _MAX_IK_REFUSAL_WARNINGS:
                    print(
                        f"[{self.LOG_TAG}] IK refused a setpoint "
                        f"({np.round(target[:3], 4).tolist()}): {exc}; holding this step"
                    )
                break
            self.backend.step(self.dt)
        self._last_command_s = float(self.backend.read_state().source_time)

    # -- extras used by the session ---------------------------------------
    def _free_gap_m(self) -> float:
        """Pad-centre separation minus the pad half-thicknesses, clamped at 0.

        Read straight off the primitives (``geom_xpos``), not from the joint angle: the
        linkage is non-linear near the closed end, so an angle-derived width would be
        optimistic exactly where the empty-grasp test matters.
        """
        model, data = self.backend.model, self.backend.data
        points = []
        for part in ("inner", "outer"):
            geom = model.geom(f"finger_{part}_{self.side}_collision_00").id
            points.append(np.asarray(data.geom_xpos[geom], dtype=float).copy())
        centre_gap = float(np.linalg.norm(points[0] - points[1]))
        return max(0.0, centre_gap - _PAD_HALF_THICKNESS_SUM_M)

    def _max_free_gap_m(self) -> float:
        """The model's fully-open free gap, used as the closed/open discriminator."""
        return _MAX_FREE_GAP_M

    @property
    def gripper_closed(self) -> bool:
        """Whether the last commanded gripper state was CLOSED."""
        return bool(self._gripper_closed)

    def sim_time_s(self) -> float:
        """Simulated seconds elapsed (from the upstream backend's own clock)."""
        return float(self.backend.read_state().source_time)

    def close(self) -> None:
        """Best-effort teardown of the upstream backend."""
        try:
            self.backend.close()
        except Exception as exc:  # noqa: BLE001 - teardown must never mask a result
            if self.verbose:
                print(f"[{self.LOG_TAG}] backend close error: {exc}")
