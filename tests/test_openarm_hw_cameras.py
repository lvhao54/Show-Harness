"""OpenArm V4L2 camera tests use mocked devices only; never open a real camera."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import core.openarm.hardware.cameras as cameras


def device(
    index, *, serial="CAM_LEFT", port="3-2", vendor="04b4", product="4950", **extra
):
    return {
        "device": f"/dev/video{index}",
        "serial": serial,
        "usb_port": port,
        "vendor": vendor,
        "product": product,
        **extra,
    }


class OpenArmCameraConfigTests(unittest.TestCase):
    def test_constructor_does_not_discover_or_open_devices(self):
        config = {
            "camera_wrist_left": {
                "vendor": "04B4",
                "product": "4950",
                "serial": "CAM_LEFT",
                "width": 960,
                "height": 600,
                "fps": 30,
            }
        }
        with patch.object(
            cameras,
            "video_devices",
            side_effect=AssertionError("device discovery during construction"),
        ):
            group = cameras.OpenArmCameras(config)
        self.assertEqual(group.names, ("camera_wrist_left",))
        self.assertEqual(group.config["camera_wrist_left"]["vendor"], "04b4")
        self.assertEqual(config["camera_wrist_left"]["vendor"], "04B4")
        self.assertFalse(group.metadata()["depth_available"])
        self.assertFalse(group.metadata()["hardware_synchronized"])

    def test_invalid_configuration_and_duplicate_explicit_indices(self):
        invalid = [
            None,
            [],
            {"bad.name": {"index_or_path": 0}},
            {"cam": None},
            {"cam": {}},
            {"cam": {"index_or_path": True}},
            {"cam": {"index_or_path": "auto"}},
            {"cam": {"index_or_path": "video0"}},
            {"cam": {"index_or_path": 0, "vendor": "04b4"}},
            {"cam": {"vendor": "4b4"}},
            {"cam": {"vendor": "04b4", "serial": ""}},
        ]
        for changes in [
            {"width": True},
            {"height": 0},
            {"fps": False},
            {"fps": float("inf")},
            {"fourcc": "RGB"},
            {"type": "zed_m"},
            {"view": "both"},
            {"view": "left", "width": 7},
            {"unknown": 1},
        ]:
            invalid.append({"cam": {"index_or_path": 0, **changes}})
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                cameras.OpenArmCameras(config)
        with self.assertRaisesRegex(ValueError, "more than once"):
            cameras.OpenArmCameras(
                {"a": {"index_or_path": 0}, "b": {"index_or_path": "/dev/video0"}}
            )
        for kwargs in [
            {"max_age_seconds": 0},
            {"startup_timeout": False},
            {"close_timeout": -1},
            {"jpeg_quality": 101},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cameras.OpenArmCameras({}, **kwargs)

    def test_identity_binding_rejects_ambiguity_and_cross_role_conflict(self):
        inventory = [device(0), device(2, serial=None, port="3-3")]
        left = {"vendor": "04b4", "product": "4950", "serial": "CAM_LEFT"}
        right = {"vendor": "04b4", "product": "4950", "usb_port": "3-3"}
        with patch.object(cameras, "video_devices", return_value=inventory):
            result = cameras.resolve_devices(
                {"camera_wrist_left": left, "camera_wrist_right": right}
            )
            self.assertEqual(result["camera_wrist_left"]["device"], "/dev/video0")
            self.assertEqual(result["camera_wrist_right"]["device"], "/dev/video2")
            with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                cameras.resolve_devices({"cam": {"vendor": "04b4"}})
            with self.assertRaisesRegex(ValueError, "more than once"):
                cameras.resolve_devices({"left": left, "other": {"index_or_path": 0}})
            with self.assertRaisesRegex(RuntimeError, "No matching"):
                cameras.resolve_devices({"left": {**left, "serial": "MISSING"}})

    def test_unreadable_match_cannot_hide_a_second_device(self):
        inventory = [
            device(0),
            device(2, port="3-3", capture_error="permission denied"),
        ]
        with patch.object(cameras, "video_devices", return_value=inventory):
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                cameras.resolve_devices({"cam": {"vendor": "04b4"}})

    def test_resolved_aliases_and_multiple_nodes_of_one_camera_are_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            node = Path(directory) / "video0"
            node.touch()
            alias = Path(directory) / "camera-left"
            alias.symlink_to(node)
            with patch.object(cameras, "video_devices", return_value=[]):
                with self.assertRaisesRegex(ValueError, "more than once"):
                    cameras.resolve_devices(
                        {
                            "a": {"index_or_path": str(node)},
                            "b": {"index_or_path": str(alias)},
                        }
                    )
        with patch.object(
            cameras, "video_devices", return_value=[device(0), device(1)]
        ):
            with self.assertRaisesRegex(ValueError, "more than once"):
                cameras.resolve_devices(
                    {"a": {"index_or_path": 0}, "b": {"index_or_path": 1}}
                )

    def test_discovery_excludes_metadata_and_keeps_usb_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sys_video = root / "class"
            sys_video.mkdir()
            usb = root / "usb" / "3-3"
            usb.mkdir(parents=True)
            for name, value in [
                ("idVendor", "04B4"),
                ("idProduct", "4950"),
                ("serial", "CAM_LEFT"),
            ]:
                (usb / name).write_text(value)
            for index in range(2):
                target = usb / "interface" / f"video{index}"
                target.mkdir(parents=True)
                (sys_video / f"video{index}").symlink_to(target)
            with (
                patch.object(cameras, "_SYS_VIDEO", sys_video),
                patch.object(cameras, "_is_video_capture", side_effect=[True, False]),
            ):
                found = cameras.video_devices()
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0], device(0, port="3-3"))


class FakeCapture:
    def __init__(self, frame=None):
        self.frame = np.zeros((4, 8, 3), dtype=np.uint8) if frame is None else frame
        self.owner = None
        self.fail = False
        self.block = threading.Event()
        self.block.set()
        self.read_entered = threading.Event()
        self.released = threading.Event()
        self.properties = []
        self.opened = True

    def _owned(self):
        current = threading.get_ident()
        if self.owner is None:
            self.owner = current
        assert self.owner == current, "camera operated from a second thread"

    def isOpened(self):
        self._owned()
        return self.opened

    def set(self, key, value):
        self._owned()
        self.properties.append((key, value))
        return True

    def read(self):
        self._owned()
        self.read_entered.set()
        self.block.wait()
        # Simulate camera pacing without making tests race a hot writer loop.
        time.sleep(0.002)
        return not self.fail, self.frame

    def release(self):
        self._owned()
        self.released.set()


class OpenArmCameraCaptureTests(unittest.TestCase):
    def setUp(self):
        self.instances = {"/dev/video0": FakeCapture(), "/dev/video2": FakeCapture()}
        self.encoded_images = []
        self.cv = SimpleNamespace(
            CAP_V4L2=200,
            CAP_PROP_FOURCC=6,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FPS=5,
            CAP_PROP_BUFFERSIZE=38,
            IMWRITE_JPEG_QUALITY=1,
            VideoWriter_fourcc=lambda *value: "".join(value),
            VideoCapture=lambda path, backend: self.instances[path],
            imencode=self.encode,
        )
        self.patches = [
            patch.dict("sys.modules", {"cv2": self.cv}),
            patch.object(cameras, "video_devices", return_value=[]),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.groups = []
        self.addCleanup(self.cleanup_groups)

    def encode(self, extension, image, params):
        self.encoded_images.append(image.copy())
        return True, np.array([255, 216, 255, 217], dtype=np.uint8)

    def cleanup_groups(self):
        for instance in self.instances.values():
            instance.block.set()
        for group in self.groups:
            group.close()

    def group(self, config=None, **kwargs):
        if config is None:
            config = {
                "camera_head": {
                    "index_or_path": 0,
                    "width": 8,
                    "height": 4,
                    "fps": 30,
                    "fourcc": "YUYV",
                }
            }
        group = cameras.OpenArmCameras(config, **kwargs)
        self.groups.append(group)
        return group

    def test_worker_ownership_original_timestamps_and_unsplit_stereo(self):
        group = self.group()
        capture = self.instances["/dev/video0"]
        capture.frame[:, :4, 2] = 255  # OpenCV BGR, sent directly to its JPEG encoder.
        group.start()
        capture.block.clear()
        time.sleep(0.01)  # Allow the already in-flight read to settle.
        first = group.capture()
        timestamp = group.capture_times["camera_head"]
        self.assertEqual(first, {"camera_head": b"\xff\xd8\xff\xd9"})
        self.assertEqual(group.capture(), first)
        self.assertEqual(group.capture_times["camera_head"], timestamp)
        self.assertEqual(self.encoded_images[-1].shape, (4, 8, 3))
        np.testing.assert_array_equal(self.encoded_images[-1][0, 0], [0, 0, 255])
        self.assertIn((self.cv.CAP_PROP_FOURCC, "YUYV"), capture.properties)
        self.assertNotEqual(capture.owner, threading.get_ident())
        self.assertEqual(group.metadata()["cameras"]["camera_head"]["width"], 8)

    def test_each_camera_acquires_independently(self):
        config = {
            "camera_wrist_left": {"index_or_path": 0},
            "camera_wrist_right": {"index_or_path": 2},
        }
        group = self.group(config)
        group.start()
        self.assertNotEqual(
            self.instances["/dev/video0"].owner, self.instances["/dev/video2"].owner
        )
        self.assertEqual(tuple(group.capture()), tuple(config))
        self.assertEqual(set(group.capture_times), set(config))

    def test_single_eye_views_cut_side_by_side_frames_without_resampling(self):
        config = {
            "camera_head": {
                "index_or_path": 0,
                "width": 8,
                "height": 4,
                "fps": 30,
                "fourcc": "YUYV",
            }
        }
        # BGR red in the left half and the unset blue channel in the right half
        # make it explicit which pixels each view delivered to the encoder.
        for view, expected_top_left, expected_size in (
            ("left", [0, 0, 255], (4, 4)),
            ("right", [255, 0, 0], (4, 4)),
            ("stereo", [0, 0, 255], (8, 4)),
        ):
            with self.subTest(view=view):
                capture = FakeCapture()
                capture.frame[:, :4, 2] = 255  # BGR left half only.
                capture.frame[:, 4:, 0] = 255  # Distinct BGR blue right half.
                self.instances["/dev/video0"] = capture
                group = self.group(
                    {"camera_head": {**config["camera_head"], "view": view}}
                )
                group.start()
                encoded = group.capture()["camera_head"]
                self.assertEqual(encoded, b"\xff\xd8\xff\xd9")
                image = self.encoded_images[-1]
                self.assertEqual(image.shape, (expected_size[1], expected_size[0], 3))
                np.testing.assert_array_equal(image[0, 0], expected_top_left)
                metadata = group.metadata()["cameras"]["camera_head"]
                self.assertEqual((metadata["width"], metadata["height"]), expected_size)
                self.assertIn(metadata["view"], ("left", "right", "stereo"))
                group.close()

    def test_stale_frame_and_failed_capture_do_not_retain_public_times(self):
        group = self.group()
        group.start()
        worker = group._workers["camera_head"]
        self.instances["/dev/video0"].block.clear()
        time.sleep(0.01)
        with worker.lock:
            worker.captured_monotonic = time.monotonic() - 1
        with self.assertRaisesRegex(RuntimeError, "stale"):
            group.capture()
        self.assertEqual(group.capture_times, {})
        self.instances["/dev/video0"].fail = True
        self.instances["/dev/video0"].block.set()
        self.assertTrue(self.instances["/dev/video0"].released.wait(1))
        with self.assertRaisesRegex(RuntimeError, "acquisition failed"):
            group.capture()
        self.assertIsNone(worker.frame)
        self.assertEqual(group.capture_times, {})

    def test_encoding_failure_and_encoding_age_do_not_publish_bundle(self):
        group = self.group()
        group.start()
        self.cv.imencode = lambda *_: (False, None)
        with self.assertRaisesRegex(RuntimeError, "JPEG encoding"):
            group.capture()
        self.assertEqual(group.capture_times, {})

        def slow_encode(*args):
            time.sleep(0.03)
            return self.encode(*args)

        group.max_age_seconds = 0.02
        self.cv.imencode = slow_encode
        with self.assertRaisesRegex(RuntimeError, "stale"):
            group.capture()
        self.assertEqual(group.capture_times, {})

    def test_invalid_frames_fail_start_and_cleanup_the_device(self):
        for frame in [
            np.zeros((4, 7, 3), np.uint8),
            np.zeros((4, 8, 4), np.uint8),
            np.zeros((4, 8, 3), np.float32),
        ]:
            with self.subTest(shape=frame.shape):
                capture = FakeCapture(frame)
                self.instances["/dev/video0"] = capture
                group = self.group()
                with self.assertRaisesRegex(RuntimeError, "acquisition failed"):
                    group.start()
                self.assertTrue(capture.released.is_set())
                self.assertFalse(group._started)

    def test_close_is_bounded_when_native_read_blocks_and_restart_is_rejected(self):
        capture = self.instances["/dev/video0"]
        group = self.group(close_timeout=0.02)
        group.start()
        capture.block.clear()
        time.sleep(0.01)
        before = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "did not stop"):
            group.close()
        self.assertLess(time.monotonic() - before, 0.2)
        with self.assertRaisesRegex(RuntimeError, "still closing"):
            group.start()
        self.assertEqual(group.capture_times, {})
        capture.block.set()
        self.assertTrue(capture.released.wait(1))
        group.close()

    def test_start_deadline_is_bounded_and_missing_devices_fail_before_open(self):
        capture = self.instances["/dev/video0"]
        capture.block.clear()
        group = self.group(startup_timeout=0.02, close_timeout=0.02)
        before = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "startup deadline"):
            group.start()
        self.assertLess(time.monotonic() - before, 0.2)
        with self.assertRaisesRegex(RuntimeError, "not started"):
            group.capture()
        group = self.group({"cam": {"vendor": "04b4"}})
        with self.assertRaisesRegex(RuntimeError, "No matching"):
            group.start()
        self.assertEqual(group._workers, {})

    def test_empty_configuration_and_idempotent_close(self):
        group = self.group({})
        group.start()
        self.assertEqual(group.capture(), {})
        group.close()
        group.close()
        group.start()
        self.assertEqual(group.capture(), {})


if __name__ == "__main__":
    unittest.main()
