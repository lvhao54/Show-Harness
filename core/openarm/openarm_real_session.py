"""OpenArm2 Pedestal hardware session: enable CAN, expose observations, hold safely.

The real-rig counterpart of :class:`core.openarm.openarm_session.OpenArmSession`. It owns
the hardware backend (CAN workers + USB cameras), wraps the driven arm in
:class:`~core.openarm.hardware.robot.OpenArmRealRobot`, and answers exactly the same
``get_observation()`` contract -- so :class:`~core.runners.real.RealEpisodeRunner`,
:class:`~core.runners.mvtoken.MvTokenRunner` and every plugin run here unchanged.

Cameras
-------
Three USB cameras are addressed by hardware identity, never by video index: the two wrist
cameras and the ZED Mini head camera (side-by-side frame, one eye delivered). The head
camera is the AgentView and the driven arm's wrist camera the wrist view; which is which
comes from the config, exactly like the simulator. Only those two are opened -- the idle
arm's wrist camera is never read, so it must not be able to fail a capture. Every capture
is a fresh complete bundle -- a stale frame fails the observation instead of being served
as new.

Safety
------
Two things gate motion before a rollout can start:

* ``driver.hardware_watchdog_confirmed`` must be true. This is an operator statement that
  the motor/controller watchdog and physical stop behaviour have been verified on this
  cell -- Python process liveness is not a hardware watchdog.
* The Z floor is resolved by the shared launcher (``openarm_real_preflight``, before the
  session is even built) and refuses to "capture at start": a raised start pose would
  forbid all descent and a low one protects nothing. Name a measured ``z_floor_m`` (or
  pass ``--z-floor-m``) before an autonomous run.

Stopping is the vendor's own path: ``close()`` stops the workers, runs the configured stop
trajectory and disables the motors, then releases the cameras. A ``SIGKILL``, a power loss
or a wedged native call can still defeat it -- this is a software stop, not an E-stop.

Control heartbeat
-----------------
The vendor dispatch loop stops an arm if the owner stops feeding it a control heartbeat
within ``driver.control_timeout_seconds``. A Show-Harness rollout spends most of its wall
clock inside VLM inference between two setpoint commands, so the session runs that
heartbeat itself on a small daemon thread. This is the stand-in for the reference runtime's
60 Hz control thread: if the process wedges badly enough to starve the heartbeat thread,
the arms run their stop trajectory and disable, which is the behaviour the timeout exists
for.
"""
from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

from core.openarm.hardware import make_pedestal_backend, require_openarm_driver
from core.openarm.hardware.robot import OpenArmRealRobot
from core.record.images import prepare_view


@dataclass
class OpenArmRealSessionConfig:
    """Hardware + camera settings for one OpenArm Pedestal session."""

    arm_side: str = "right"
    #: The robot config's ``driver:`` block, passed through to the CAN backend.
    driver: dict[str, Any] = field(default_factory=dict)
    #: Named camera table (``driver.cameras``): USB identity + capture size per role.
    cameras: dict[str, Any] = field(default_factory=dict)

    #: Square render size fed to the policy.
    observation_resolution: Optional[int] = 256
    #: Optional high-res render (``agentview_hd`` / ``wrist_hd``) for glyph-reading plugins.
    hd_resolution: Optional[int] = None

    #: Camera transforms -- see core.record.images.prepare_view. The physical mounts are
    #: fixed, so unlike the simulator these describe the actual rig and must be measured.
    agentview_rotation_degrees: int = 0
    wrist_rotation_degrees: int = 0
    agentview_flip: str = "none"
    wrist_flip: str = "none"
    agentview_crop_aspect: Optional[float] = None
    wrist_crop_aspect: Optional[float] = None

    verbose: bool = True


class OpenArmRealSession:
    """Owns the OpenArm Pedestal CAN workers, cameras and the driven-arm robot wrapper."""

    LOG_TAG = "openarm-real-session"

    def __init__(self, config: Optional[OpenArmRealSessionConfig] = None) -> None:
        self.config = config or OpenArmRealSessionConfig()
        self.backend: Any = None
        self.robot: Optional[OpenArmRealRobot] = None
        self._connected = False
        # Serializes captures between the rollout and a live-display poller.
        self._camera_lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> "OpenArmRealSession":
        """Validate the driver stack, start CAN + cameras, and wrap the driven arm.

        Startup is all-or-nothing: if anything fails after the motors were enabled -- the
        robot wrapper's first feedback read, or a preload that never stabilizes (arms
        powered off, on the wrong bus, or not holding still) -- the motors are stopped and
        disabled before the error propagates. Leaving them enabled with no heartbeat would
        be the worst outcome of a failed startup.
        """
        require_openarm_driver()
        try:
            self._connect_openarm()
            self._connected = True
            self._start_heartbeat()
            self.wait_ready()
            # The idle arm sagged under gravity while the preload settled; latch its hold
            # pose only now, or the first command would lift it back to the enable pose.
            assert self.robot is not None  # set in _connect_openarm()
            self.robot.sync_hold()
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"OpenArm startup cleanup also failed: {cleanup}")
            raise
        return self

    def _start_heartbeat(self) -> None:
        """Feed the dispatch loop while Show-Harness thinks between commands."""
        timeout = float(self.config.driver.get("control_timeout_seconds", 0.5) or 0.5)
        interval = max(0.02, min(0.2, timeout / 3.0))
        self._heartbeat_stop.clear()

        def pump() -> None:
            while not self._heartbeat_stop.wait(interval):
                try:
                    self.backend.step(0.0)
                except Exception:  # noqa: BLE001 - a worker fault surfaces in command()
                    return

        self._heartbeat_thread = threading.Thread(
            target=pump, name="openarm-real-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread.ident is not None:
            thread.join(max(0.0, 1.0))
        self._heartbeat_thread = None

    def wait_ready(self, timeout_s: Optional[float] = None) -> None:
        """Block until the support preload is stable enough to accept a Cartesian command.

        The preload tracker needs a full ``preload_window_seconds`` of stable
        commanded-vs-measured samples before it will latch the offset. The reference
        runtime reports a retryable error until then; here the session waits once at
        startup, so the first rollout step does not fail on a technicality.
        """
        window = float(self.config.driver.get("preload_window_seconds", 0.3) or 0.3)
        deadline = time.monotonic() + (
            float(timeout_s) if timeout_s is not None else max(2.0, 4.0 * window)
        )
        last: Optional[BaseException] = None
        while True:
            try:
                self.backend.validate_motion_ready()
                self._log(f"[{self.LOG_TAG}]   support preload is stable; arm is ready")
                return
            except (RuntimeError, ValueError) as exc:
                last = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "OpenArm support preload did not stabilize before the deadline "
                    f"({last}); check that both arms are powered, on their own CAN bus, "
                    "and holding still."
                ) from last
            time.sleep(0.05)

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    def _connect_openarm(self) -> None:
        cfg = self.config
        self._log(f"[{self.LOG_TAG}] starting CAN workers and cameras ...")
        self.backend = make_pedestal_backend(
            driver=cfg.driver, cameras=self._driven_cameras()
        )
        self.backend.start()
        self.robot = OpenArmRealRobot(
            self.backend,
            side=cfg.arm_side,
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
        """Stop the arm workers and cameras; never raises through the caller."""
        self._stop_heartbeat()
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not mask a result
                self._log(f"[{self.LOG_TAG}] backend close error: {exc}")
        self._connected = False

    def __enter__(self) -> "OpenArmRealSession":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- camera helpers ----------------------------------------------------
    def agentview_camera(self) -> str:
        """The third-person camera name (fixed head camera; not either wrist view)."""
        return "camera_head"

    def wrist_camera(self) -> str:
        """The driven arm's eye-in-hand camera name."""
        return f"camera_wrist_{self.config.arm_side}"

    def _driven_cameras(self) -> dict[str, Any]:
        """The two camera roles this rollout reads, from the site's camera table.

        The idle arm's wrist camera is never opened: every capture requires each open
        camera to be fresh, so a stall on a view nobody reads would abort the rollout.
        An empty table (state-only commissioning) passes through unchanged.
        """
        table = self.config.cameras
        if not table:
            return {}
        wanted = (self.agentview_camera(), self.wrist_camera())
        missing = [name for name in wanted if name not in table]
        if missing:
            raise ValueError(
                f"driver.cameras has no {missing} role(s); available: {sorted(table)}. "
                "Fill in configs/site/openarm.yaml from the example."
            )
        return {name: table[name] for name in wanted}

    def _render_all(self) -> dict[str, np.ndarray]:
        """One fresh complete capture, decoded from the backend's JPEGs to RGB."""
        if not self._connected or self.backend is None:
            raise RuntimeError(
                "OpenArmRealSession.get_observation called before connect()"
            )
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
        """Capture one fresh observation.

        Returns the same keys every other session does -- ``agentview``, ``wrist``,
        ``ee_pose``, ``gripper_width`` -- plus ``agentview_hd`` / ``wrist_hd`` when
        ``hd_resolution`` is set. The three camera frames carry independent host
        timestamps; the pose is read after them, so a stalled camera fails here instead of
        silently mixing a stale image with fresh feedback.
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
                f"camera {name!r} not captured; available: {sorted(frames)}"
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

    # -- introspection -----------------------------------------------------
    def success(self) -> bool:
        """No independent task predicate on hardware; the VLM owns completion."""
        return False

    def scene_truth(self) -> dict[str, Any]:
        """No simulator ground truth on hardware; returns an empty, labeled record."""
        return {"source": "hardware", "frame": "arm_origin", "objects": {}}
