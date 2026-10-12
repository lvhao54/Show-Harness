"""OpenArm2 Pedestal hardware backend using AceOpenarm's thread-owned send strategy.

The runtime owns Cartesian trajectories and measured completion; independent
CAN workers own both vendor drivers and maintain filtered position commands.
Construction validates configuration and loads kinematics without device I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.openarm.backend._gl import ensure_render_backend

ensure_render_backend()  # MUJOCO_GL must be set before the first mujoco import

import numpy as np
from openarm_mujoco.v2 import openarm_pedestal_xml

from core.openarm.backend.kinematics import create_kinematics, solve_bimanual
from core.openarm.backend.state import BackendState
from core.openarm.hardware.cameras import OpenArmCameras
from core.openarm.hardware.dispatch import BimanualDispatcher


class OpenArmPedestalBackend:
    """RobotBackend bridge for a bimanual OpenArm2 mounted on a pedestal."""

    mode = "real"
    frame = "arm_origin"

    def __init__(self, config: Mapping[str, Any]):
        config = dict(config)
        allowed = {
            "profile",
            "driver_config",
            "kinematics_xml",
            "hardware_watchdog_confirmed",
            "right_can_interface",
            "left_can_interface",
            "control_hz",
            "send_hz",
            "cutoff_hz",
            "startup_trajectory",
            "max_feedback_seconds",
            "max_dispatch_seconds",
            "control_timeout_seconds",
            "startup_timeout_seconds",
            "shutdown_timeout_seconds",
            "preload_window_seconds",
            "preload_stability_tolerance",
            "preload_max_offset",
            "cameras",
            "max_camera_age_seconds",
            "camera_startup_timeout_seconds",
            "camera_close_timeout_seconds",
            "jpeg_quality",
            "ik_max_iters",
            "ik_position_cost",
            "ik_orientation_cost",
            "ik_frame_position_error_limit",
            "ik_frame_orientation_error_limit",
            "ik_nullspace_cost",
            "ik_kinetic_energy_cost",
        }
        if set(config) - allowed:
            raise ValueError(
                f"Unknown OpenArm driver options: {sorted(set(config) - allowed)}"
            )
        for name in ("driver_config", "kinematics_xml"):
            if name in config and (
                not isinstance(config[name], str) or not config[name].strip()
            ):
                raise ValueError(
                    f"OpenArm {name} must be a nonempty path or profile name"
                )
        if config.get("profile", "openarm_pedestal") != "openarm_pedestal":
            raise ValueError(
                "OpenArm2 hardware adapter requires profile=openarm_pedestal"
            )
        if config.get("hardware_watchdog_confirmed") is not True:
            raise ValueError(
                "Set driver.hardware_watchdog_confirmed=true only after verifying "
                "the OpenArm motor watchdog and physical stop behavior"
            )
        self._config = config
        camera_config = config.get("cameras", {})
        if camera_config == []:  # Preserve historical state-only commissioning TOML.
            camera_config = {}
        self._camera_group = OpenArmCameras(
            camera_config,
            max_age_seconds=config.get("max_camera_age_seconds", 0.2),
            jpeg_quality=config.get("jpeg_quality", 90),
            startup_timeout=config.get("camera_startup_timeout_seconds", 5),
            close_timeout=config.get("camera_close_timeout_seconds", 1),
        )
        self.cameras = self._camera_group.names
        self._dispatch = BimanualDispatcher(config)
        control_hz = _bounded(config.get("control_hz", 60), "control_hz", 1, 1000)
        xml = config.get("kinematics_xml") or openarm_pedestal_xml()
        self.setup, self.kin = create_kinematics(
            xml,
            control_hz,
            config,
        )
        self._started = False

    @property
    def gripper_targets(self) -> dict[str, float]:
        return {
            side: float(joints[-1])
            for side, joints in self._dispatch.applied_targets().items()
        }

    @property
    def hardware_watchdog(self) -> bool:
        # Explicit local confirmation, not a claim that host threads implement
        # the independent motor/controller watchdog.
        return self._config.get("hardware_watchdog_confirmed") is True

    def start(self) -> None:
        if self._started:
            raise RuntimeError("OpenArm hardware backend is already started")
        try:
            # Bad/missing camera identities or frames must fail before enabling
            # a motor. Every capture worker is independent of the CAN workers.
            self._camera_group.start()
            self._dispatch.start()
            self._camera_group.capture()
            self._started = True
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"OpenArm startup cleanup also failed: {cleanup}")
            raise

    def read_state(self) -> BackendState:
        samples = self._dispatch.samples()
        right_joints = samples["right"].qpos
        left_joints = samples["left"].qpos
        poses = self.kin.fk_bimanual(right_joints, left_joints)
        return BackendState(
            poses[0].astype(float).tolist(),
            poses[1].astype(float).tolist(),
            right_joints.tolist(),
            left_joints.tolist(),
            min(sample.captured_at for sample in samples.values()),
        )

    def command(
        self, right: list[float], left: list[float], grippers: dict[str, float]
    ) -> None:
        samples = self._dispatch.samples()
        preload = self._dispatch.preload_bias()
        solved = solve_bimanual(
            self.kin,
            np.r_[samples["right"].qpos, samples["left"].qpos],
            right,
            left,
            grippers,
            error="OpenArm pedestal inverse kinematics failed",
        )
        # The vendor position loop has no torque feed-forward. Preserve the
        # stable startup/hold support offset while following the Cartesian IK
        # target; the jaw channel is never biased.
        solved[:7] += preload["right"][:7]
        solved[8:15] += preload["left"][:7]
        # This waits for both workers to actually dispatch this generation;
        # retained grippers reflect driver.last_command after safety limiting.
        self._dispatch.command(solved[:8], solved[8:])

    def validate_motion_ready(self) -> None:
        """Ensure a stable support preload is available before accepting move."""
        self._dispatch.preload_bias(capture=False)

    def command_grippers(self, grippers: dict[str, float]) -> None:
        # Do not solve IK for a jaw-only action. The dispatcher starts from the
        # last command actually sent to each arm, preserving holding torque.
        self._dispatch.command_grippers(grippers)

    def step(self, dt: float) -> None:
        del dt
        # Workers retain idle targets, but must stop if runtime itself stalls.
        self._dispatch.heartbeat()

    def hold(self) -> None:
        self._dispatch.hold()

    @property
    def capture_times(self) -> dict[str, float]:
        return dict(self._camera_group.capture_times)

    def capture(self) -> dict[str, bytes]:
        return self._camera_group.capture()

    def observation_metadata(self) -> dict[str, Any]:
        return self._camera_group.observation_metadata()

    def close_camera(self) -> None:
        self._camera_group.close()

    def close_control(self) -> None:
        try:
            self._dispatch.close()
        finally:
            self._started = False

    def close(self) -> None:
        error = None
        for cleanup in (self.close_control, self.close_camera):
            try:
                cleanup()
            except Exception as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"OpenArm cleanup also failed: {exc}")
        if error is not None:
            raise error


def create_backend(config: dict[str, Any]) -> OpenArmPedestalBackend:
    """Entry-point constructor: one ``[driver]`` config table -> one backend."""
    return OpenArmPedestalBackend(config)


def make_pedestal_backend(
    *, driver: Mapping[str, Any] | None = None, cameras: Mapping[str, Any] | None = None,
) -> OpenArmPedestalBackend:
    """Build the backend from Show-Harness's resolved config blocks.

    ``driver`` is the robot config's ``driver:`` mapping (CAN interfaces, rates, safety
    confirmations); ``cameras`` is the optional named camera table. Unknown driver keys
    are rejected by the backend, so a config typo fails at construction.
    """
    options = dict(driver or {})
    if cameras is not None:
        options["cameras"] = dict(cameras)
    return OpenArmPedestalBackend(options)


def _bounded(value: Any, name: str, lower: float, upper: float) -> float:
    """Validated float, replacing the runtime's RobotError-raising helper."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"OpenArm {name} must be a finite number within [{lower},{upper}]")
    number = float(value)
    if not np.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"OpenArm {name} must be within [{lower},{upper}]")
    return number
