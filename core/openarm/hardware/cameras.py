"""V4L2 RGB cameras for OpenArm, using AceOpenarm's USB identity mappings.

Each physical camera owns its OpenCV capture in one worker thread. Frame times
are host read-completion times, not hardware synchronization or exposure times.
A side-by-side ZED UVC frame can be delivered whole (``view="stereo"``) or cut
into one eye (``view="left"`` or ``"right"``); no depth or calibration is
inferred. Construction validates configuration without opening any device.
"""

from __future__ import annotations

import math
import os
import re
import struct
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _is_video_capture(path):
    """VIDIOC_QUERYCAP probe: distinguish image nodes from UVC metadata nodes."""
    import fcntl

    capability = bytearray(104)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        fcntl.ioctl(descriptor, 0x80685600, capability, True)
    finally:
        os.close(descriptor)
    capabilities, device_capabilities = struct.unpack_from("=II", capability, 84)
    if capabilities & 0x80000000:  # V4L2_CAP_DEVICE_CAPS
        capabilities = device_capabilities
    return bool(capabilities & (0x00000001 | 0x00001000))  # VIDEO_CAPTURE[_MPLANE]


_SYS_VIDEO = Path("/sys/class/video4linux")
_DEV_ROOT = Path("/dev")
_IDENTITY_KEYS = ("vendor", "product", "serial", "usb_port")


def validate_camera_configs(config: Mapping) -> dict[str, dict[str, Any]]:
    """Copy and validate logical names and explicit device/USB identity settings."""
    if not isinstance(config, Mapping):
        raise ValueError("OpenArm driver.cameras must be a table of named cameras")
    result = {}
    explicit = set()
    for name, supplied in config.items():
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError(
                "Camera names must contain only letters, digits, '_' or '-'"
            )
        if not isinstance(supplied, Mapping):
            raise ValueError(f"Camera {name} settings must be a table")
        settings = dict(supplied)
        allowed = {
            "type",
            "index_or_path",
            "width",
            "height",
            "fps",
            "fourcc",
            "view",
            *_IDENTITY_KEYS,
        }
        if set(settings) - allowed:
            raise ValueError(
                f"Unsupported camera settings for {name}: {sorted(set(settings) - allowed)}"
            )
        if settings.setdefault("type", "opencv") != "opencv":
            raise ValueError(f"Camera {name} type must be 'opencv'")
        view = settings.setdefault("view", "stereo")
        if view not in ("stereo", "left", "right"):
            raise ValueError(f"Camera {name} view must be stereo, left, or right")
        if (
            view in ("left", "right")
            and "width" in settings
            and settings["width"] % 2
        ):
            raise ValueError(
                f"Camera {name} view={view} needs an even side-by-side width"
            )
        if "index_or_path" in settings:
            if any(key in settings for key in _IDENTITY_KEYS):
                raise ValueError(
                    f"Camera {name} must use index_or_path or USB identity, not both"
                )
            device = settings["index_or_path"]
            if not (
                (type(device) is int and device >= 0)
                or (isinstance(device, str) and Path(device).is_absolute())
            ):
                raise ValueError(
                    f"Camera {name} index_or_path must be a nonnegative index or absolute path"
                )
            # Pure lexical check here: resolving symlinks belongs to start().
            key = (
                str(_DEV_ROOT / f"video{device}")
                if type(device) is int
                else str(Path(device))
            )
            if key in explicit:
                raise ValueError(
                    f"Physical camera {device} is configured more than once"
                )
            explicit.add(key)
        elif "vendor" not in settings:
            raise ValueError(
                f"Camera {name} needs index_or_path or a USB vendor identity"
            )
        for key in ("vendor", "product"):
            if key in settings:
                value = settings[key]
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-fA-F]{4}", value) is None
                ):
                    raise ValueError(
                        f"Camera {name} {key} must be a four-digit USB hex ID"
                    )
                settings[key] = value.lower()
        for key in ("serial", "usb_port"):
            if key in settings and (
                not isinstance(settings[key], str) or not settings[key].strip()
            ):
                raise ValueError(f"Camera {name} {key} must be a nonempty string")
        for key in ("width", "height"):
            if key in settings and (
                type(settings[key]) is not int or settings[key] <= 0
            ):
                raise ValueError(f"Camera {name} {key} must be a positive integer")
        if "fps" in settings:
            value = settings["fps"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Camera {name} fps must be positive and finite")
        if "fourcc" in settings:
            value = settings["fourcc"]
            if not isinstance(value, str) or len(value) != 4 or not value.isascii():
                raise ValueError(
                    f"Camera {name} fourcc must contain four ASCII characters"
                )
        result[name] = settings
    return result


def video_devices() -> list[dict[str, Any]]:
    """Enumerate USB video image nodes, retaining identity inspection failures.

    Unreadable matching nodes must not turn an ambiguous USB selection into an
    apparently unique one. V4L2 metadata nodes are excluded from image matches.
    """
    devices = []
    for entry in sorted(_SYS_VIDEO.glob("video*")):
        try:
            usb = next(
                parent
                for parent in entry.resolve().parents
                if (parent / "idVendor").exists()
            )
        except StopIteration:
            continue  # Non-USB cameras remain available through an explicit path.
        path = _DEV_ROOT / entry.name
        try:
            serial = usb / "serial"
            identity = {
                "device": str(path),
                "vendor": (usb / "idVendor").read_text().strip().lower(),
                "product": (usb / "idProduct").read_text().strip().lower(),
                "usb_port": usb.name,
                "serial": serial.read_text().strip() or None
                if serial.exists()
                else None,
            }
        except OSError as exc:
            raise RuntimeError(
                f"Cannot inspect USB identity for {path}: {exc}"
            ) from exc
        try:
            if not _is_video_capture(path):
                continue
        except OSError as exc:
            identity["capture_error"] = str(exc)
        devices.append(identity)
    return devices


def _matches(device: Mapping, config: Mapping) -> bool:
    return all(
        device.get(key) == config[key] for key in _IDENTITY_KEYS if key in config
    )


def _deliver_view(name: str, settings: Mapping, image: np.ndarray) -> np.ndarray:
    """Return the configured delivered image without resampling pixels.

    ``stereo`` keeps the original side-by-side frame. ``left``/``right`` cut
    one eye at exactly half the captured width, so a horizontal shift cannot
    blend the two views together.
    """
    view = settings.get("view", "stereo")
    if view == "stereo":
        return image
    width = image.shape[1]
    if width % 2:
        raise RuntimeError(
            f"Camera {name} view={view} needs an even side-by-side width, got {width}"
        )
    half = width // 2
    if view == "left":
        return image[:, :half]
    return image[:, half:]


def resolve_devices(config: Mapping) -> dict[str, dict[str, Any]]:
    """Resolve every role before any camera is opened; reject shared devices."""
    inventory = video_devices()
    selected = {}
    identities = set()
    for name, settings in config.items():
        if "index_or_path" in settings:
            value = settings["index_or_path"]
            path = _DEV_ROOT / f"video{value}" if type(value) is int else Path(value)
            resolved = path.resolve()
            known = [
                item for item in inventory if Path(item["device"]).resolve() == resolved
            ]
            item = dict(known[0]) if known else {"device": str(path)}
        else:
            matches = [item for item in inventory if _matches(item, settings)]
            unavailable = [item for item in matches if item.get("capture_error")]
            if unavailable:
                raise RuntimeError(
                    f"Cannot inspect matching camera {name}: {unavailable}"
                )
            if len(matches) != 1:
                reason = (
                    "Ambiguous camera mapping"
                    if matches
                    else "No matching USB video camera"
                )
                raise RuntimeError(
                    f"{reason} for {name}; specify serial or usb_port: {matches}"
                )
            item = dict(matches[0])
            resolved = Path(item["device"]).resolve()
        if item.get("capture_error"):
            raise RuntimeError(f"Cannot inspect camera {name}: {item['capture_error']}")
        # Multiple capture nodes of one USB camera are still one physical camera.
        identity = (
            ("usb", item["usb_port"])
            if item.get("usb_port")
            else ("path", str(resolved))
        )
        if identity in identities:
            raise ValueError(
                f"Physical camera {item['device']} is configured more than once (role {name})"
            )
        identities.add(identity)
        selected[name] = item
    return selected


def _positive(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass
class _Camera:
    name: str
    settings: dict[str, Any]
    device: dict[str, Any]
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    frame: np.ndarray | None = None
    captured_at: float | None = None
    captured_monotonic: float | None = None
    sequence: int = 0
    error: BaseException | None = None


class OpenArmCameras:
    """Latest-frame RGB acquisition, with bounded start and close waits.

    A blocked native camera read cannot be forcibly interrupted safely from a
    second thread. Workers are daemons and close reports a timeout instead of
    blocking robot shutdown indefinitely; each worker releases its own device.
    """

    def __init__(
        self,
        config: Mapping,
        *,
        max_age_seconds: float = 0.2,
        jpeg_quality: int = 90,
        startup_timeout: float = 5.0,
        close_timeout: float = 1.0,
    ):
        self.config = validate_camera_configs(config)
        self.names = tuple(self.config)
        self.max_age_seconds = _positive(max_age_seconds, "max_age_seconds")
        self.startup_timeout = _positive(startup_timeout, "startup_timeout")
        self.close_timeout = _positive(close_timeout, "close_timeout")
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer from 1 to 100")
        self.jpeg_quality = jpeg_quality
        self.capture_times: dict[str, float] = {}
        self._workers: dict[str, _Camera] = {}
        self._stop = threading.Event()
        self._started = False
        self._cv2 = None

    def start(self) -> None:
        if self._started or any(
            w.thread is not None and w.thread.is_alive() for w in self._workers.values()
        ):
            raise RuntimeError("OpenArm cameras are already started or still closing")
        if not self.config:
            self._stop = threading.Event()
            self.capture_times = {}
            self._started = True
            return
        selected = resolve_devices(self.config)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenArm camera extra: uv sync --extra openarm"
            ) from exc
        self._cv2 = cv2
        self._stop = threading.Event()
        self.capture_times = {}
        self._workers = {
            name: _Camera(name, settings, selected[name])
            for name, settings in self.config.items()
        }
        self._started = True
        try:
            for worker in self._workers.values():
                worker.thread = threading.Thread(
                    target=self._run,
                    args=(worker,),
                    name=f"openarm-camera-{worker.name}",
                    daemon=True,
                )
                worker.thread.start()
            deadline = time.monotonic() + self.startup_timeout
            for worker in self._workers.values():
                if not worker.ready.wait(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(
                        f"Camera {worker.name} did not provide a frame before the startup deadline"
                    )
                with worker.lock:
                    if worker.error is not None:
                        raise RuntimeError(
                            f"Camera {worker.name} acquisition failed: {worker.error}"
                        ) from worker.error
            self.capture()  # Validate a fresh complete bundle before declaring startup.
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(f"Camera cleanup also failed: {cleanup_error}")
            raise

    def _run(self, worker: _Camera) -> None:
        cv2 = self._cv2
        capture = None
        try:
            capture = cv2.VideoCapture(worker.device["device"], cv2.CAP_V4L2)
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {worker.device['device']}")
            settings = worker.settings
            if "fourcc" in settings:
                capture.set(
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings["fourcc"])
                )
            for key, prop in (
                ("width", cv2.CAP_PROP_FRAME_WIDTH),
                ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                ("fps", cv2.CAP_PROP_FPS),
            ):
                if key in settings:
                    capture.set(prop, settings[key])
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self._stop.is_set():
                ok, frame = capture.read()
                monotonic, wall = time.monotonic(), time.time()
                if self._stop.is_set():
                    break
                if not ok:
                    raise RuntimeError("Frame capture failed")
                if (
                    not isinstance(frame, np.ndarray)
                    or frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape[2] != 3
                    or min(frame.shape[:2]) <= 0
                ):
                    raise RuntimeError("Camera must return a nonempty uint8 BGR image")
                if ("width" in settings and frame.shape[1] != settings["width"]) or (
                    "height" in settings and frame.shape[0] != settings["height"]
                ):
                    raise RuntimeError(
                        f"Camera returned {frame.shape[1]}x{frame.shape[0]}, different from requested dimensions"
                    )
                with worker.lock:
                    worker.frame = frame.copy()
                    worker.captured_monotonic = monotonic
                    worker.captured_at = wall
                    worker.sequence += 1
                worker.ready.set()
        except BaseException as exc:
            with worker.lock:
                worker.error = exc
                worker.frame = None
                worker.captured_at = worker.captured_monotonic = None
            worker.ready.set()
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    with worker.lock:
                        if worker.error is None:
                            worker.error = exc
            with worker.lock:
                worker.frame = None
                worker.captured_at = worker.captured_monotonic = None

    def capture(self) -> dict[str, bytes]:
        """Encode a fresh complete bundle; preserve each original acquisition time."""
        self.capture_times = {}
        if not self._started:
            raise RuntimeError("OpenArm cameras are not started")
        frames, timestamps, monotonic_times = {}, {}, {}
        for name, worker in self._workers.items():
            settings = self.config[name]
            if not worker.lock.acquire(timeout=self.max_age_seconds):
                raise TimeoutError(f"Camera {name} frame lock exceeded deadline")
            try:
                if worker.error is not None:
                    raise RuntimeError(
                        f"Camera {name} acquisition failed: {worker.error}"
                    ) from worker.error
                if worker.thread is None or not worker.thread.is_alive():
                    raise RuntimeError(f"Camera {name} acquisition stopped")
                if (
                    worker.frame is None
                    or worker.captured_at is None
                    or worker.captured_monotonic is None
                    or not 0
                    <= time.monotonic() - worker.captured_monotonic
                    <= self.max_age_seconds
                ):
                    raise RuntimeError(f"Camera {name} frame is stale")
                image = _deliver_view(name, settings, worker.frame).copy()
                timestamps[name] = worker.captured_at
                monotonic_times[name] = worker.captured_monotonic
            finally:
                worker.lock.release()
            ok, encoded = self._cv2.imencode(
                ".jpg", image, [self._cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ok:
                raise RuntimeError(f"Camera {name} JPEG encoding failed")
            frames[name] = encoded.tobytes()
        # Encoding later cameras must not publish a bundle whose first frame aged out.
        if any(
            not 0 <= time.monotonic() - timestamp <= self.max_age_seconds
            for timestamp in monotonic_times.values()
        ):
            raise RuntimeError("Camera bundle became stale during JPEG encoding")
        if not self._started or self._stop.is_set():
            raise RuntimeError("OpenArm cameras stopped during capture")
        for name, worker in self._workers.items():
            if (
                worker.error is not None
                or worker.thread is None
                or not worker.thread.is_alive()
            ):
                raise RuntimeError(f"Camera {name} acquisition stopped during capture")
        self.capture_times = timestamps
        return frames

    def metadata(self) -> dict[str, Any]:
        cameras = {}
        for name, settings in self.config.items():
            view = settings.get("view", "stereo")
            item = {
                "type": "opencv",
                "views": [name],
                "view": view,
                "requested": dict(settings),
            }
            worker = self._workers.get(name)
            if worker is not None:
                with worker.lock:
                    item["device"] = dict(worker.device)
                    if worker.frame is not None:
                        width, height = worker.frame.shape[1], worker.frame.shape[0]
                        if view in ("left", "right"):
                            width //= 2
                        item["width"], item["height"] = width, height
            cameras[name] = item
        return {
            "source": "hardware_rgb",
            "depth_available": False,
            "camera_to_arm_origin_calibrated": False,
            "hardware_synchronized": False,
            "timestamp_source": "host_frame_read_completed",
            "cameras": cameras,
        }

    def observation_metadata(self) -> dict[str, Any]:
        return {"camera_info": self.metadata()}

    def close(self) -> None:
        """Stop acquisition without waiting indefinitely for a blocked native read."""
        self._stop.set()
        self._started = False
        self.capture_times = {}
        deadline = time.monotonic() + self.close_timeout
        for worker in self._workers.values():
            if worker.thread is not None and worker.thread.ident is not None:
                worker.thread.join(max(0.0, deadline - time.monotonic()))
        alive = [
            name
            for name, worker in self._workers.items()
            if worker.thread is not None and worker.thread.is_alive()
        ]
        if alive:
            raise TimeoutError(
                f"Camera acquisition did not stop before deadline: {alive}"
            )
