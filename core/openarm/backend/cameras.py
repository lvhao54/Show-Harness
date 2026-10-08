"""Camera rendering following enactic/dora-openarm-mujoco's CameraRenderer.

Vendored from ``embodied_ai-llm`` (``src/robot_runtime/cameras/mujoco.py``,
Apache-2.0, Enactic Inc.). See ``NOTICE.md`` in this directory.

Keep each MJCF camera's projection and resolution. A renderer pool must be
created, used and closed on the same thread (including with a desktop viewer).
"""

from __future__ import annotations

import io
from contextlib import ExitStack

from core.openarm.backend._gl import ensure_render_backend

ensure_render_backend()  # MUJOCO_GL must be set before the first mujoco import

import mujoco
import numpy as np
from PIL import Image

DEFAULT_CAMERAS = (
    "camera_wrist_left",
    "camera_wrist_right",
    "camera_head",
)


def camera_sizes(
    model: mujoco.MjModel,
    cameras: tuple[str, ...],
    max_width: int | None = None,
    max_height: int | None = None,
) -> dict[str, tuple[int, int]]:
    """Return native (width, height), or fit optional bounds without stretching."""
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("Select at least one camera, without duplicate names")
    for bound in (max_width, max_height):
        if bound is not None and bound <= 0:
            raise ValueError("Image dimensions must be positive")
    sizes = {}
    for name in cameras:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            raise ValueError(f"Camera {name!r} not found in model")
        width, height = map(int, model.cam_resolution[cam_id])
        if min(width, height) <= 0:
            raise ValueError(f"Camera {name!r} has no positive MJCF resolution")
        scale = min(
            1.0,
            max_width / width if max_width is not None else 1.0,
            max_height / height if max_height is not None else 1.0,
        )
        sizes[name] = (max(1, round(width * scale)), max(1, round(height * scale)))
    return sizes


def camera_metadata(
    model: mujoco.MjModel, data: mujoco.MjData,
    sizes: dict[str, tuple[int, int]], *, origin_site: str = "arm_origin",
) -> dict:
    """Describe RGB sensor geometry at the captured state, never scene objects.

    ``origin_site`` is the robot's API reference (SO101 calls baseframe
    arm_origin). MuJoCo camera axes are right/up/back; published optical axes
    are right/down/forward. Intrinsics describe the actual render dimensions.
    """
    origin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, origin_site)
    if origin_id < 0:
        raise ValueError(f"Camera metadata requires the robot reference site {origin_site!r}")
    origin = data.site_xpos[origin_id]
    world_to_arm = data.site_xmat[origin_id].reshape(3, 3).T
    cameras, extrinsics = {}, {}
    for name, (width, height) in sizes.items():
        camera_id = model.camera(name).id
        rotation = world_to_arm @ data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = world_to_arm @ (data.cam_xpos[camera_id] - origin)
        extrinsics[name] = {
            "source": "mujoco_model", "status": "exact_in_simulation",
            "parent_frame": "arm_origin", "child_frame": f"{name}_optical",
            "reference_point": "optical_center", "camera_axes": "x right, y down, z forward",
            "transform_direction": "camera_optical_to_arm_origin",
            "matrix": transform.tolist(),
            "optical_axis_in_arm": rotation[:, 2].tolist(),
            "image_right_axis_in_arm": rotation[:, 0].tolist(),
            "image_down_axis_in_arm": rotation[:, 1].tolist(),
        }
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        vertical, horizontal, clip = (np.zeros(2, dtype=np.float32) for _ in range(3))
        # Ask the rendering library for its frustum: supports fovy, explicit
        # sensor intrinsics, asymmetric principal points and orthographic views.
        mujoco.mjv_cameraFrustum(vertical, horizontal, clip, model, camera)
        top, bottom = float(vertical[0]), -float(vertical[1])
        halfwidth = float(sum(horizontal)) / 2 or (top - bottom) * width / height / 2
        center = float(horizontal[1] - horizontal[0]) / 2
        scale_x, scale_y = width / (2 * halfwidth), height / (top - bottom)
        cx, cy = width / 2 - center * scale_x - .5, top * scale_y - .5
        orthographic = model.cam_projection[camera_id] == mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
        entry = {
            "resolution": [width, height],
            "projection": "orthographic" if orthographic else "pinhole",
            "pixel_coordinates": "u right, v down; top-left pixel center is (0, 0)",
            "principal_point_px": [cx, cy],
            "mount": "world_fixed" if model.cam_bodyid[camera_id] == 0 else "body_attached",
            "pose_updates_each_frame": True,
        }
        if orthographic:
            entry["pixels_per_meter"] = [scale_x, scale_y]
        else:
            fx, fy = scale_x * float(clip[0]), scale_y * float(clip[0])
            entry["intrinsic_matrix"] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        cameras[name] = entry
    return {
        "source": "simulation_rgb", "simulation_only": True, "depth_available": False,
        "camera_to_arm_origin_calibrated": True,
        "calibration_scope": "rendered simulation only; no hardware calibration or object depth",
        "cameras": cameras, "extrinsics": extrinsics,
    }


class CameraRenderer:
    """One offscreen renderer per camera, allocated at that camera's resolution."""

    def __init__(
        self, model: mujoco.MjModel, sizes: dict[str, tuple[int, int]]
    ) -> None:
        self._resources = ExitStack()
        self._ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in sizes
        }
        # Enlarge the framebuffer BEFORE constructing any renderer.
        model.vis.global_.offwidth = max(
            model.vis.global_.offwidth, *(w for w, _ in sizes.values())
        )
        model.vis.global_.offheight = max(
            model.vis.global_.offheight, *(h for _, h in sizes.values())
        )
        self._renderers = {}
        try:
            for name, (width, height) in sizes.items():
                self._renderers[name] = self._resources.enter_context(
                    mujoco.Renderer(model, height=height, width=width)
                )
        except BaseException:
            self.close()
            raise

    def render_all(self, data: mujoco.MjData) -> dict[str, bytes]:
        """Return RGB JPEGs from the same simulation state, with no resizing."""
        frames = {}
        for name, renderer in self._renderers.items():
            renderer.update_scene(data, camera=self._ids[name])
            out = io.BytesIO()
            Image.fromarray(renderer.render()).save(out, format="JPEG", quality=90)
            frames[name] = out.getvalue()
        return frames

    def close(self) -> None:
        self._resources.close()

    def __enter__(self) -> CameraRenderer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def configure_viewer_camera(viewer: object) -> None:
    """Frame the Pedestal, both arms and the tabletop task area."""
    viewer.cam.azimuth = 150
    viewer.cam.elevation = -25
    viewer.cam.distance = 1.8
    viewer.cam.lookat[:] = [0.25, 0.0, 0.4]
