"""MuJoCo driver for the OpenArm Pedestal; owns the physics and the camera capture.

Vendored from ``embodied_ai-llm``
(``src/robot_runtime/robots/openarm/simulation.py``, Apache-2.0, Enactic Inc.).
The scene XMLs it loads are bundled next to this package; see ``NOTICE.md``.
"""

from __future__ import annotations

from core.openarm.backend._gl import ensure_render_backend

ensure_render_backend()  # MUJOCO_GL must be set before the first mujoco import

import mujoco
import numpy as np

from core.openarm.backend.cameras import DEFAULT_CAMERAS
from core.openarm.backend.kinematics import create_kinematics, solve_bimanual
from core.openarm.backend.mujoco_backend import MujocoBackend
from core.openarm.backend.scenes import openarm_scene
from core.openarm.backend.state import BackendState


class SimulationBackend(MujocoBackend):
    mode = "simulation"
    frame = "arm_origin"
    hardware_watchdog = False

    def __init__(self, xml: str | None = None, control_hz: float = 60, cameras=DEFAULT_CAMERAS,
                 *, scene: str | None = None):
        with openarm_scene(xml, scene=scene) as scene_path:
            self.setup, self.kin = create_kinematics(scene_path, control_hz)
        super().__init__(self.setup.model, self.setup.data, cameras)
        self._qcmd = self._joints()
        self._gripper_targets = None
        self._gravity_data = mujoco.MjData(self.model)
        self._arm_actuators = np.array([
            self.model.actuator(f"{side}_joint{joint}_ctrl").id
            for side in ("right", "left") for joint in range(1, 8)
        ])
        self._arm_dofs = np.concatenate([
            self.setup.joint_resolver.arm_dof_indices(side)
            for side in ("right", "left")
        ])
        for actuator, dof in zip(self._arm_actuators, self._arm_dofs):
            joint = self.model.actuator_trnid[actuator, 0]
            if not (
                self.model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_dyntype[actuator] == mujoco.mjtDyn.mjDYN_NONE
                and self.model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
                and self.model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_AFFINE
                and self.model.jnt_dofadr[joint] == dof
                and np.array_equal(self.model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])
            ):
                raise ValueError("Gravity compensation requires direct affine arm actuators")
        self._arm_bias = self.model.actuator_biasprm[self._arm_actuators, 0].copy()

    def _joints(self):
        joints, grippers = self.setup.joint_resolver.get_driver(
            self.data.qpos, "bimanual"
        )
        return np.r_[joints[:7], grippers[0], joints[7:], grippers[1]]

    @property
    def gripper_targets(self) -> dict[str, float]:
        with self.guard():
            if self._gripper_targets is None:
                joints = self._joints()
                return {"right": float(joints[7]), "left": float(joints[15])}
            return dict(self._gripper_targets)

    def start(self):
        # Fixtures may reposition the model after construction. Seed from the
        # startup measurement, never a constructor-time or previous-run target.
        self._gripper_targets = None
        self.hold()

    def read_state(self):
        with self.guard():
            mujoco.mj_kinematics(self.model, self.data)
            joints = self._joints()
            return BackendState(
                self.setup.read_ee_pose("right").tolist(),
                self.setup.read_ee_pose("left").tolist(),
                joints[:8].tolist(),
                joints[8:].tolist(),
                float(self.data.time),
            )

    def observation_metadata(self, prepared=None):
        """Report robot-only pad geometry in the same frozen state as the JPEGs.

        These are collision-mesh bounds centers, not object/contact data or a
        calibrated grasp target. The centers move with the finger joints.
        """
        if prepared is None:
            with self.guard():
                return self._observation_metadata(self.model, self.data)
        return self._observation_metadata(self._camera_model, prepared)

    def _observation_metadata(self, model, data):
        metadata = super().observation_metadata(None if data is self.data else data)
        geometry = {}
        for side in ("right", "left"):
            control_name = f"{side}_ee_control_point"
            pad_names = [f"finger_{part}_{side}_collision_00" for part in ("inner", "outer")]
            # Custom XML may omit the default pads; do not guess a transform.
            if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
                   for name in pad_names):
                continue
            control = data.site(control_name)
            rotation = control.xmat.reshape(3, 3)
            centers = [rotation.T @ (data.geom(name).xpos - control.xpos) for name in pad_names]
            geometry[side] = {
                "source": "robot_model_kinematics",
                "scope": "simulation robot geometry only; not object depth, contact or grasp evidence",
                "control_frame": control_name,
                "position_unit": "metres",
                "finger_pad_centers_in_control_frame": [center.tolist() for center in centers],
                "finger_pad_midpoint_in_control_frame": np.mean(centers, axis=0).tolist(),
                "usage": "pad_midpoint_in_arm_origin = control_position + control_rotation @ midpoint; "
                         "move targets are control-point poses, not pad or object positions. "
                         "Confirm alignment and clearance in the images.",
            }
        if geometry:
            metadata["end_effector_geometry"] = geometry
        return metadata

    def command(self, right, left, grippers):
        with self.guard():
            self._qcmd = solve_bimanual(
                self.kin, self._joints(), right, left, grippers, dtype=np.float32,
            )

    def command_grippers(self, grippers):
        with self.guard():
            self._qcmd[7] = grippers["right"]
            self._qcmd[15] = grippers["left"]

    def _compensate_gravity(self):
        # Compute gravity at the measured pose, with zero velocity so Coriolis
        # forces are not fed back. A separate MjData leaves live state untouched.
        self._gravity_data.qpos[:] = self.data.qpos
        self._gravity_data.qvel[:] = 0
        mujoco.mj_fwdPosition(self.model, self._gravity_data)
        mujoco.mj_fwdVelocity(self.model, self._gravity_data)
        # Add feedforward INSIDE each actuator's affine force law. MuJoCo still
        # clamps the combined PD + gravity torque to the actuator/joint limits.
        # qfrc_applied or body gravcomp would bypass that actuator force budget.
        self.model.actuator_biasprm[self._arm_actuators, 0] = (
            self._arm_bias + self._gravity_data.qfrc_bias[self._arm_dofs]
        )

    def _write_controls(self):
        if self._gripper_targets is None:
            joints = self._joints()
            self._gripper_targets = {"right": float(joints[7]), "left": float(joints[15])}
        resolver = self.setup.joint_resolver
        for side, offset in (("right", 0), ("left", 8)):
            resolver.set_ctrl(self.data.ctrl, self._qcmd[offset:offset + 8], side)
            self._gripper_targets[side] = float(self._qcmd[offset + 7])

    def step(self, dt):
        with self.guard():
            self._write_controls()
            self._compensate_gravity()
            self._step_physics(dt)

    def hold(self):
        with self.guard():
            self._qcmd = self._joints().copy()
            if self._gripper_targets is not None:
                self._qcmd[7] = self._gripper_targets["right"]
                self._qcmd[15] = self._gripper_targets["left"]
            self._write_controls()
            self._compensate_gravity()
