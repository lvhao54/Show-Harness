"""OpenArm simulation session: build the world, expose observations, answer success checks.

The environment-setup half of an OpenArm rollout. It owns the upstream MuJoCo backend and
the :class:`~core.openarm.openarm_interface.OpenArmSimRobot` wrapper, and exposes the same
``get_observation()`` contract the Franka session does, so
:class:`~core.runners.mvtoken.MvTokenRunner` drives it without modification.

Cameras
-------
The upstream model ships three: ``camera_head`` (a fixed third-person view, 1280x720) and
two eye-in-hand views, ``camera_wrist_right`` / ``camera_wrist_left`` (960x600 each). The
driven arm's wrist camera supplies the wrist view; the head camera supplies the agentview,
because per-arm wrist views move with the arm and cannot serve as the global reference the
prompt's ``MV_LEFT``/``MV_RIGHT`` rules are written against.

Both views go through :func:`core.record.images.prepare_view` -- the same
rotate/crop/letterbox contract every other embodiment uses -- so the frames this session
hands to the policy are shaped like the ones it was trained on.

Task success
------------
Three predicates ship, all read from MuJoCo state that the policy never sees:

* ``"cubes_on_table"`` -- the default scene's three cubes are all still resting on the
  tabletop (nothing knocked off). A cheap "did the arm wreck the scene" check.
* ``"red_on_green"`` -- the red cube rests on the green cube, which rests on the
  table; both cubes are stationary and clear of robot contact.
* ``"flowers_in_vase"`` -- the ``insert_flower`` scene's task: every rose tip is brought to
  the vase mouth. **This is a GEOMETRIC relation, not physical containment**: the upstream
  vase model carries no collision hull through the aperture (measured with a downward
  raycast grid -- every ray passes straight to the table), so a released rose simply falls
  to the tabletop. The predicate therefore matches VLABench's own relation-based scoring
  ("is the stem at the container") rather than asking whether the rose stayed in. It is
  satisfiable while the arm holds a rose at the mouth, which is the behaviour the
  ``insert_flower`` scene exists to exercise.

``"none"`` disables the check, which is what the fine-tuned MVTOKEN loop wants: that policy
emits ``DONE`` as its own terminal signal and never consults an environment predicate.
"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image

from core.openarm.openarm_interface import OpenArmSimRobot
from core.openarm.sim import make_openarm_backend
from core.record.images import prepare_view

#: Success predicates the session understands (``"none"`` disables checking).
SUCCESS_CHECKS = ("none", "cubes_on_table", "red_on_green", "flowers_in_vase")

#: Default scene's cube bodies, from the upstream ``openarm2.xml`` keyframe.
_CUBE_BODIES = ("box_red", "box_green", "box_blue")
#: Rose bodies in the upstream ``insert_flower`` scene.
_FLOWER_BODIES = ("flower_1", "flower_2", "flower_3")


@dataclass
class OpenArmSessionConfig:
    """World + camera settings for one OpenArm simulation session."""

    scene: Optional[str] = None
    arm_side: str = "right"
    control_hz: float = 60.0
    #: Physics ticks advanced per setpoint command (see OpenArmSimRobot).
    sim_ticks_per_command: int = 6
    gripper_ticks: int = 60
    settle_ticks: int = 60
    #: Square render size fed to the policy (mirrors the real rigs' observation_resolution).
    observation_resolution: Optional[int] = 256
    #: Optional high-res render (``agentview_hd`` / ``wrist_hd``) for glyph-reading plugins.
    hd_resolution: Optional[int] = None
    #: Camera transforms -- see core.record.images.prepare_view.
    agentview_rotation_degrees: int = 0
    wrist_rotation_degrees: int = 0
    agentview_flip: str = "none"
    wrist_flip: str = "none"
    agentview_crop_aspect: Optional[float] = None
    wrist_crop_aspect: Optional[float] = None
    success_check: str = "none"
    verbose: bool = True


class OpenArmSession:
    """Owns the OpenArm MuJoCo world, its robot wrapper, and its camera captures."""

    LOG_TAG = "openarm-session"

    def __init__(self, config: Optional[OpenArmSessionConfig] = None) -> None:
        self.config = config or OpenArmSessionConfig()
        self.backend: Any = None
        self.robot: Optional[OpenArmSimRobot] = None
        self._connected = False
        self._origin_z = 0.0
        # Serializes captures between the rollout and a live-display poller.
        self._camera_lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "OpenArmSession":
        """Build the MuJoCo world, start it holding, and wrap the driven arm."""
        self._success_check_name()  # Reject typos before allocating the simulator.
        self._connect_openarm()
        self._connected = True
        return self

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def _connect_openarm(self) -> None:
        cfg = self.config
        self._log(
            f"[{self.LOG_TAG}] building MuJoCo world (scene={cfg.scene or 'default cubes'}) ..."
        )
        self.backend = make_openarm_backend(
            scene=cfg.scene or None,
            control_hz=float(cfg.control_hz),
        )
        # arm_origin is the frame everything is reported in; its world Z is the offset
        # between world coordinates (used by the scene predicates) and API coordinates.
        self._origin_z = float(self.backend.data.site("arm_origin").xpos[2])
        self.robot = OpenArmSimRobot(
            self.backend,
            side=cfg.arm_side,
            sim_ticks_per_command=int(cfg.sim_ticks_per_command),
            gripper_ticks=int(cfg.gripper_ticks),
            settle_ticks=int(cfg.settle_ticks),
            verbose=cfg.verbose,
        )
        pose = self.robot.get_ee_pose()
        self._log(
            f"[{self.LOG_TAG}]   arm={cfg.arm_side} ee_pose="
            f"{np.round(pose, 4).tolist()} "
            f"gripper={self.robot.get_gripper_position()[0] * 1000:.1f}mm"
        )
        self._log(f"[{self.LOG_TAG}]   cameras={list(self.backend.cameras)}")

    def close(self) -> None:
        if self.robot is not None:
            self.robot.close()
        self._connected = False

    def __enter__(self) -> "OpenArmSession":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- camera helpers ----------------------------------------------------
    def agentview_camera(self) -> str:
        """The third-person camera name (fixed head camera; not the wrist-driven view)."""
        return "camera_head"

    def wrist_camera(self) -> str:
        """The driven arm's eye-in-hand camera name."""
        return f"camera_wrist_{self.config.arm_side}"

    def _render_all(self) -> dict[str, np.ndarray]:
        """One frozen-state capture, decoded to HWC uint8 RGB."""
        if not self._connected or self.backend is None:
            raise RuntimeError("OpenArmSession.get_observation called before connect()")
        with self._camera_lock:
            frames = self.backend.capture()
        out: dict[str, np.ndarray] = {}
        for name, payload in frames.items():
            out[name] = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
        return out

    def _view(self, frame: np.ndarray, *, wrist: bool, res: Optional[int]) -> np.ndarray:
        cfg = self.config
        shaped = prepare_view(
            frame,
            rotation_degrees=(
                cfg.wrist_rotation_degrees if wrist else cfg.agentview_rotation_degrees
            ),
            flip=cfg.wrist_flip if wrist else cfg.agentview_flip,
            crop_aspect=cfg.wrist_crop_aspect if wrist else cfg.agentview_crop_aspect,
            square_size=res,
        )
        return np.ascontiguousarray(shaped)

    # -- observations ------------------------------------------------------
    def get_observation(self) -> dict[str, Any]:
        """Capture a fresh observation.

        Returns the same keys the real sessions do -- ``agentview``, ``wrist``, ``ee_pose``,
        ``gripper_width`` -- plus ``agentview_hd`` / ``wrist_hd`` when ``hd_resolution`` is
        set. Both halves come from ONE frozen physics state, so the images and the pose
        describe the same instant.
        """
        assert self.robot is not None  # set in connect()
        cfg = self.config
        frames = self._render_all()
        agent_raw = self._require_frame(frames, self.agentview_camera())
        wrist_raw = self._require_frame(frames, self.wrist_camera())
        obs: dict[str, Any] = {
            "agentview": self._view(agent_raw, wrist=False, res=cfg.observation_resolution),
            "wrist": self._view(wrist_raw, wrist=True, res=cfg.observation_resolution),
            "ee_pose": self.robot.get_ee_pose(),
            "gripper_width": float(self.robot.get_gripper_position()[0]),
        }
        if cfg.hd_resolution:
            obs["agentview_hd"] = self._view(agent_raw, wrist=False, res=cfg.hd_resolution)
            obs["wrist_hd"] = self._view(wrist_raw, wrist=True, res=cfg.hd_resolution)
        return obs

    @staticmethod
    def _require_frame(frames: dict[str, np.ndarray], name: str) -> np.ndarray:
        try:
            return frames[name]
        except KeyError as exc:
            raise KeyError(
                f"camera {name!r} not rendered; available: {sorted(frames)}"
            ) from exc

    def get_camera_frames(self) -> Optional[dict[str, np.ndarray]]:
        """Latest two views for a LIVE DISPLAY poller; never raises.

        Mirrors ``FrankaSession.get_camera_frames``: a cosmetic display thread must not be
        able to perturb or crash the rollout, so any failure returns ``None``.
        """
        if not self._connected or self.backend is None:
            return None
        try:
            cfg = self.config
            frames = self._render_all()
            return {
                "agentview": self._view(
                    self._require_frame(frames, self.agentview_camera()),
                    wrist=False,
                    res=cfg.observation_resolution,
                ),
                "wrist": self._view(
                    self._require_frame(frames, self.wrist_camera()),
                    wrist=True,
                    res=cfg.observation_resolution,
                ),
            }
        except Exception:  # noqa: BLE001 - display is best-effort by contract
            return None

    # -- task predicates ---------------------------------------------------
    def _success_check_name(self) -> str:
        check = str(self.config.success_check or "none").strip().lower()
        if check not in SUCCESS_CHECKS:
            raise ValueError(
                f"Unknown success_check {check!r}; expected one of {list(SUCCESS_CHECKS)}"
            )
        return check

    def success(self) -> bool:
        """Whether the configured predicate currently holds (``"none"`` -> False)."""
        check = self._success_check_name()
        if check == "none":
            return False
        if self.backend is None:
            return False
        if check == "cubes_on_table":
            return self._cubes_on_table()
        if check == "red_on_green":
            return self._red_on_green()
        return self._flowers_in_vase()

    def _cubes_on_table(self) -> bool:
        """All three cubes still resting on the tabletop (not knocked off the edge)."""
        model, data = self.backend.model, self.backend.data
        if self.config.scene is not None:
            # The cube bodies exist only in the default tabletop scene.
            return False
        for body in _CUBE_BODIES:
            geom = model.geom(f"{body}_geom").id
            x, y, z = (float(v) for v in data.geom_xpos[geom])
            # Table top is at world z ~0.40; below that the cube has left the surface.
            if z < 0.40:
                return False
            # Tabletop footprint is 0.82 x 1.10 m centred at x=0.47, y=0.
            if not (0.06 <= x <= 0.88 and -0.55 <= y <= 0.55):
                return False
        return True

    def _red_on_green(self) -> bool:
        """A released, stationary stack in the default cube scene.

        Geometry alone can mistake a held or airborne cube for a completed stack.
        Require supporting contacts (red/green and green/table), a centered red
        cube, low free-joint velocities, and no contact with either robot arm.
        This reads simulator truth only when evaluating; observations remain visual.
        """
        if self.config.scene:
            return False
        model, data = self.backend.model, self.backend.data
        red = model.geom("box_red_geom").id
        green = model.geom("box_green_geom").id
        table = model.geom("table_top").id
        red_pos, green_pos = data.geom_xpos[red], data.geom_xpos[green]
        red_extent = np.abs(data.geom_xmat[red].reshape(3, 3)) @ model.geom_size[red]
        green_extent = np.abs(data.geom_xmat[green].reshape(3, 3)) @ model.geom_size[green]
        # Keep the red center of mass inside the support footprint with a 10%
        # margin. The vertical tolerance permits the solver's small penetration.
        if np.any(np.abs((red_pos - green_pos)[:2]) > 0.9 * green_extent[:2]):
            return False
        expected_height = red_extent[2] + green_extent[2]
        if abs(float(red_pos[2] - green_pos[2]) - expected_height) > 0.003:
            return False
        for body in ("box_red", "box_green"):
            dof = int(model.jnt_dofadr[model.joint(f"{body}_free").id])
            velocity = data.qvel[dof:dof + 6]
            if np.linalg.norm(velocity[:3]) > 0.01 or np.linalg.norm(velocity[3:]) > 0.1:
                return False

        robot_geoms = {
            geom_id for geom_id in range(model.ngeom)
            if model.body(int(model.geom_bodyid[geom_id])).name.startswith("openarm_")
        }
        red_supported = green_supported = False
        for contact in data.contact:
            if contact.dist > 0.001:
                continue
            pair = {int(contact.geom1), int(contact.geom2)}
            if (red in pair or green in pair) and pair & robot_geoms:
                return False
            # Side contacts cannot support a stack, even if the centers line up.
            if abs(float(contact.frame[2])) < 0.9:
                continue
            red_supported |= pair == {red, green}
            green_supported |= pair == {green, table}
        return red_supported and green_supported

    def _flowers_in_vase(self) -> bool:
        """Every rose tip is within the vase mouth aperture.

        A tip counts when its horizontal offset from the mouth centre is inside the
        aperture and its height is within a band around the rim -- i.e. the stem is at the
        opening. See the module docstring: this is a geometric relation, not a physics
        containment test, because the upstream vase has no collision hull below the rim.

        The band is generous on the high side because the grasp necessarily stops short of
        the rim: the pads are 0.1638 m ahead of the commanded control point, so the model
        cannot reach arbitrarily low without driving the wrist through the table.
        """
        model, data = self.backend.model, self.backend.data
        mouth = np.asarray(data.site_xpos[model.site("vase_mouth").id], dtype=float)
        # Mouth aperture: the vase is scaled 0.65 from VLABench, giving ~0.065 m across;
        # allow for the stem's own radius when deciding "at the opening".
        radius = 0.045
        above, below = 0.12, 0.02
        for body in _FLOWER_BODIES:
            tip = np.asarray(data.site_xpos[model.site(f"{body}_tip").id], dtype=float)
            horizontal = float(np.linalg.norm((tip - mouth)[:2]))
            dz = float(tip[2] - mouth[2])
            if horizontal > radius or not (-below <= dz <= above):
                return False
        return True

    # -- introspection -----------------------------------------------------
    def scene_truth(self) -> dict[str, Any]:
        """Object poses in ``arm_origin`` for logging/analysis only, never for the policy.

        Mirrors the upstream diagnostics contract: this is simulator ground truth for
        debugging, and must not be fed to the model.
        """
        model, data = self.backend.model, self.backend.data
        origin = np.array([0.0, 0.0, self._origin_z])
        out: dict[str, Any] = {
            "source": "mujoco_ground_truth",
            "frame": "arm_origin",
            "objects": {},
        }
        if self.config.scene == "insert_flower":
            mouth = np.asarray(data.site_xpos[model.site("vase_mouth").id], dtype=float)
            out["objects"]["vase_mouth"] = {
                "position": np.round(mouth - origin, 5).tolist()
            }
            for body in _FLOWER_BODIES:
                tip = np.asarray(data.site_xpos[model.site(f"{body}_tip").id], dtype=float)
                out["objects"][f"{body}_tip"] = {
                    "position": np.round(tip - origin, 5).tolist()
                }
        else:
            # The default scene's objects; only present in the cube tabletop model.
            for body in _CUBE_BODIES:
                pos = np.asarray(data.geom_xpos[model.geom(f"{body}_geom").id], float)
                out["objects"][body] = {
                    "position": np.round(pos - origin, 5).tolist()
                }
        return out

    def sim_time_s(self) -> float:
        """Simulated seconds elapsed in this world."""
        return 0.0 if self.backend is None else float(self.backend.read_state().source_time)
