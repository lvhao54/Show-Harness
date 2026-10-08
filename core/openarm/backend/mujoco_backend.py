"""Small shared lifecycle helpers for MuJoCo robot backends.

Vendored from ``embodied_ai-llm`` (``src/robot_runtime/robots/mujoco.py``,
Apache-2.0, Enactic Inc.). See ``NOTICE.md`` in this directory.

The robot-specific modules keep ownership of controls, state conversion and
safety policy. This class centralizes state snapshots, camera renderer
lifecycle and fractional physics-step accounting.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import threading

from core.openarm.backend._gl import ensure_render_backend

ensure_render_backend()  # MUJOCO_GL must be set before the first mujoco import

import mujoco

from core.openarm.backend.cameras import CameraRenderer, camera_metadata, camera_sizes


class MujocoBackend:
    """Reusable MuJoCo runtime plumbing for simulation robot adapters."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cameras: tuple[str, ...],
    ) -> None:
        self.model = model
        self.data = data
        self.cameras = tuple(cameras)
        self._sizes = camera_sizes(model, self.cameras)
        self._renderer = None
        self._physics_budget = 0.0
        # Runtime control and camera capture can run on different threads.
        # Protect MuJoCo model/data access independently from RobotRuntime's
        # state lock; rendering uses a copied MjData after this lock is released.
        self._state_lock = threading.RLock()
        self._render_lock = threading.Lock()
        # Rendering owns a model as well as data: OpenArm gravity compensation
        # mutates the live model's actuator parameters on every control tick.
        # Allocate once, before control starts; do not allocate MjData or create
        # GL contexts inside the control critical section.
        self._camera_model = copy.copy(model)
        self._camera_data = mujoco.MjData(self._camera_model)

    @contextmanager
    def guard(self):
        with self._state_lock:
            yield

    def _step_physics(self, dt: float) -> None:
        """Advance whole MuJoCo steps while retaining fractional elapsed time."""
        self._physics_budget += dt
        count = int((self._physics_budget + 1e-12) / self.model.opt.timestep)
        if count:
            mujoco.mj_step(self.model, self.data, nstep=count)
            self._physics_budget -= count * self.model.opt.timestep

    @property
    def camera_origin_site(self):
        return "arm_origin"

    def prepare_capture(self):
        """Freeze live data; Runtime calls this under its control/capture locks.

        The reusable buffer belongs to the capture thread until publication.
        Camera model topology/geometry is fixed at backend construction; the
        control thread's changing actuator bias is copied with the physics data.
        """
        with self.guard():
            mujoco.mj_copyData(self._camera_data, self.model, self.data)
            self._camera_model.actuator_biasprm[:] = self.model.actuator_biasprm
        return self._camera_data

    def observation_metadata(self, prepared=None):
        """Use the JPEG's frozen transforms, or inspect live geometry offline."""
        if prepared is None:
            with self.guard():
                return {"camera_info": camera_metadata(
                    self.model, self.data, self._sizes, origin_site=self.camera_origin_site,
                )}
        return {"camera_info": camera_metadata(
            self._camera_model, prepared, self._sizes, origin_site=self.camera_origin_site,
        )}

    def capture(self, prepared=None):
        """Render a prepared state, without a control/backend/viewer lock.

        No-argument capture remains available for offline, single-threaded use.
        Runtime supplies the buffer prepared alongside its measured feedback.
        GL creation, use and destruction stay on the capture owner thread.
        """
        with self._render_lock:
            if prepared is None:
                prepared = self.prepare_capture()
            # mj_step leaves derived transforms at the previous substep.
            mujoco.mj_kinematics(self._camera_model, prepared)
            mujoco.mj_camlight(self._camera_model, prepared)
            if self._renderer is None:
                self._renderer = CameraRenderer(self._camera_model, self._sizes)
            return self._renderer.render_all(prepared)

    def close_camera(self) -> None:
        with self._render_lock:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None

    def start(self):
        self.hold()

    def close(self):
        self.hold()
