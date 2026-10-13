"""Real OpenArm dispatch contract; all CAN and motors are in-memory fakes."""

import math
import threading
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.openarm.hardware import dispatch


class FakeLink:
    def __init__(self):
        self.responses = 1
        self.rejected_commands = 0
        self.malformed_frames = 0
        self.last_response = time.monotonic()
        self.responsive = True

    def ever_responded(self):
        return self.responses > 0

    def seconds_since_response(self):
        return time.monotonic() - self.last_response

    def is_stale(self, timeout):
        return self.seconds_since_response() > timeout


class FakeDriver:
    def __init__(self, side, interface):
        self.side, self.interface = side, interface
        self.owner = threading.get_ident()
        self.links = [FakeLink() for _ in range(8)]
        self.num_mit_motors, self.gripper_posforce = 7, True
        self.measured = np.r_[np.zeros(7), -0.3 if side == "right" else 0.3]
        self.last_command = self.measured.copy()
        self.last_command_dispatch_timestamp_ns = None
        self.started, self.enabled = False, False
        self.safety_stop_reason = "test rejection"
        self.sent = []
        self.events = threading.Condition()
        self.disabled = threading.Event()
        self.disable_calls = 0
        self.disable_noops = 0
        self.disable_errors = 0
        self.disable_fetch_errors = 0
        self.disable_baseline = None
        self.disabled_axes = None
        self.reject = False
        self.clamp_grip = None
        self.counter_fault = None
        self.receive_error = False
        self.before_send = None
        self.fetch_count = 0
        self.respond_every = 1
        self.startup_steps = 0
        self.stop_steps = 0
        self.stop_entered = threading.Event()
        self.stop_gate = None
        self.stop_reject = False
        self.startup_entered = threading.Event()
        self.startup_gate = None
        self.safety_checker = SimpleNamespace(
            check=lambda *_args, **_kwargs: SimpleNamespace(
                is_safe=True, force_stop=False
            )
        )
        self.bus = SimpleNamespace(
            **{
                key: SimpleNamespace(count=0)
                for key in ("write_no_buffer", "write_net_down", "write_other")
            }
        )
        self.openarm = SimpleNamespace(
            get_arm=lambda: SimpleNamespace(get_link_stats=lambda i: self.links[i]),
            get_gripper=lambda: SimpleNamespace(get_link_stats=lambda i: self.links[7]),
            get_bus_status=lambda: self.bus,
            is_link_running=lambda: True,
            disable_all=self.disable,
        )

    def assert_owner(self):
        assert threading.get_ident() == self.owner, "cross-thread native CAN access"

    def enable(self):
        self.assert_owner()
        self.enabled = self.started = True
        self.last_command = self.measured.copy()

    def disable(self):
        self.assert_owner()
        self.disable_calls += 1
        if self.disable_baseline is None:
            self.disable_baseline = [link.responses for link in self.links]
        if self.disable_errors:
            self.disable_errors -= 1
            raise OSError("transient disable CAN failure")
        if self.disable_noops:
            self.disable_noops -= 1
            return  # Native write succeeded, but a motor never received it.
        self.enabled = False
        self.disabled.set()

    def fetch_state(self, **_):
        self.assert_owner()
        self.fetch_count += 1
        if self.disable_calls and self.disable_fetch_errors:
            self.disable_fetch_errors -= 1
            raise OSError("transient shutdown feedback failure")
        for link in self.links:
            if link.responsive and self.fetch_count % self.respond_every == 0:
                link.responses += 1
                link.last_response = time.monotonic()
        return {"qpos": self.measured.copy()}

    def get_health(self):
        self.assert_owner()
        if self.disabled_axes is not None and self.disable_calls:
            statuses = [
                "DISABLED" if i in self.disabled_axes else "ENABLED" for i in range(8)
            ]
        else:
            statuses = ["ENABLED" if self.enabled else "DISABLED"] * 8
        return statuses, {"carrier": True}

    def send_position(self, values):
        self.assert_owner()
        if self.before_send:
            self.before_send(values)
        if self.reject:
            return False
        self.last_command = np.asarray(values).copy()
        if self.clamp_grip is not None:
            self.last_command[-1] = self.clamp_grip
        self.last_command_dispatch_timestamp_ns = time.time_ns()
        with self.events:
            self.sent.append(self.last_command.copy())
            self.events.notify_all()
        if self.counter_fault == "write":
            self.bus.write_no_buffer.count += 1
        elif self.counter_fault == "reject":
            self.links[-1].rejected_commands += 1
        elif self.counter_fault == "malformed":
            self.links[2].malformed_frames += 1
        if self.receive_error:
            raise OSError("receive failed after dispatch")
        return True

    def move_to_start_position(self):
        self.assert_owner()
        self.startup_entered.set()
        if self.startup_gate:
            self.startup_gate.wait(1)
        for _ in range(20):
            checked = self.safety_checker.check(self.last_command, driver=self)
            if not checked.is_safe and checked.force_stop:
                return False
            self.startup_steps += 1
            self.send_position(self.last_command)
            self.fetch_state()
        return True

    def move_to_stop_position(self):
        self.assert_owner()
        self.stop_entered.set()
        if self.stop_gate:
            self.stop_gate.wait(1)
        for _ in range(20):
            if self.stop_reject:
                return False
            self.stop_steps += 1
            self.send_position(self.last_command)
            self.fetch_state()
        return True

    def wait_sends(self, count, timeout=1):
        deadline = time.monotonic() + timeout
        with self.events:
            while len(self.sent) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("timed out waiting for fake sends")
                self.events.wait(remaining)


class DispatcherFixture:
    def __init__(self, *, configure=None, **config):
        self.stack = ExitStack()
        self.drivers = {}
        self.profile = SimpleNamespace(
            get_can_interface=lambda side: f"calibrated-{side}",
            get_stop_config=lambda: {"moves": [{"duration": 0.0}]},
        )
        self.profile_mock = self.stack.enter_context(
            patch.object(dispatch, "make_profile", return_value=self.profile)
        )

        def create(side, _profile, interface):
            driver = FakeDriver(side, interface)
            self.drivers[side] = driver
            if configure:
                configure(driver)
            return driver

        self.stack.enter_context(
            patch.object(dispatch, "make_driver", side_effect=create)
        )
        self.stack.enter_context(
            patch.object(
                dispatch, "enable_at_measurement", side_effect=lambda d: d.enable()
            )
        )
        self.dispatcher = dispatch.BimanualDispatcher(
            {"startup_timeout_seconds": 1, "shutdown_timeout_seconds": 1, **config}
        )

    def start(self):
        self.dispatcher.start()
        return self.dispatcher

    def close(self):
        try:
            self.dispatcher.close()
        except (RuntimeError, TimeoutError):
            pass  # Tests inspect the original worker failure explicitly.
        self.stack.close()


class OpenArmDispatchTests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = DispatcherFixture(**kwargs)
        self.addCleanup(fixture.close)
        return fixture

    def test_each_native_driver_is_owned_by_one_thread_and_idle_keeps_sending(self):
        fixture = self.fixture(driver_config="local-calibration.yaml")
        robot = fixture.start()
        fixture.profile_mock.assert_called_once_with("local-calibration.yaml")
        right, left = fixture.drivers["right"], fixture.drivers["left"]
        self.assertNotEqual(right.owner, left.owner)
        self.assertEqual(right.interface, "calibrated-right_arm")
        right.wait_sends(len(right.sent) + 3)
        np.testing.assert_array_equal(robot.applied_targets()["right"], right.measured)
        robot.close()
        self.assertTrue(right.disabled.is_set() and left.disabled.is_set())

    def test_close_retries_lost_disable_and_confirms_fresh_feedback_for_both_arms(self):
        fixture = self.fixture(configure=lambda d: setattr(d, "disable_noops", 1))
        robot = fixture.start()
        before_stop = {side: len(d.sent) for side, d in fixture.drivers.items()}
        robot.close()
        counts = {}
        for side, driver in fixture.drivers.items():
            self.assertGreaterEqual(driver.disable_calls, 2)
            self.assertEqual(driver.stop_steps, 20)
            self.assertGreater(len(driver.sent), before_stop[side])
            self.assertFalse(driver.enabled)
            self.assertFalse(driver.started)
            self.assertTrue(all(
                link.responses > before
                for link, before in zip(driver.links, driver.disable_baseline)
            ))
            counts[side] = len(driver.sent), driver.disable_calls
        robot.close()
        self.assertEqual(
            counts,
            {side: (len(d.sent), d.disable_calls) for side, d in fixture.drivers.items()},
        )

    def test_close_runs_stop_trajectory_before_disable(self):
        fixture = self.fixture()
        robot = fixture.start()
        for driver in fixture.drivers.values():
            driver.stop_entered.clear()

        robot.close()

        for driver in fixture.drivers.values():
            self.assertTrue(driver.stop_entered.is_set())
            self.assertEqual(driver.stop_steps, 20)
            self.assertFalse(driver.enabled)

    def test_stop_trajectory_can_be_disabled_for_in_place_shutdown(self):
        """capture_z_floor.py disables where the operator placed the arm."""
        fixture = self.fixture(stop_trajectory=False)
        robot = fixture.start()
        for driver in fixture.drivers.values():
            driver.stop_entered.clear()

        robot.close()

        for driver in fixture.drivers.values():
            self.assertFalse(driver.stop_entered.is_set())
            self.assertFalse(driver.enabled)
            self.assertTrue(driver.disabled.is_set())

    def test_stop_trajectory_failure_is_reported_after_disable(self):
        fixture = self.fixture()
        robot = fixture.start()
        for driver in fixture.drivers.values():
            driver.stop_reject = True

        with self.assertRaisesRegex(RuntimeError, "stop trajectory was rejected"):
            robot.close()

        for driver in fixture.drivers.values():
            self.assertFalse(driver.enabled)
            self.assertTrue(driver.disabled.is_set())

    def test_shutdown_budget_includes_configured_stop_duration(self):
        profile = SimpleNamespace(
            get_can_interface=lambda side: f"calibrated-{side}",
            get_stop_config=lambda: {"moves": [
                {"duration": 2.0}, {"duration": 3.0},
            ]},
        )
        with patch.object(dispatch, "make_profile", return_value=profile):
            robot = dispatch.BimanualDispatcher({"shutdown_timeout_seconds": 0.5})
            with patch.object(dispatch, "make_driver", side_effect=RuntimeError("stop")):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    robot.start()
        self.assertEqual(robot.stop_duration, 5.0)

    def test_close_recovers_from_transient_disable_and_feedback_errors(self):
        for fault in ("disable_errors", "disable_fetch_errors"):
            with self.subTest(fault=fault):
                fixture = self.fixture(configure=lambda d: setattr(d, fault, 1))
                robot = fixture.start()
                robot.close()
                for driver in fixture.drivers.values():
                    if fault == "disable_errors":
                        self.assertGreaterEqual(driver.disable_calls, 2)
                    self.assertEqual(getattr(driver, fault), 0)
                    self.assertFalse(driver.enabled)
                    self.assertFalse(driver.started)
                    self.assertTrue(all(
                        link.responses > before
                        for link, before in zip(driver.links, driver.disable_baseline)
                    ))
                fixture.close()

    def test_close_rejects_cached_disabled_status_without_fresh_axis_response(self):
        fixture = self.fixture(shutdown_timeout_seconds=0.08)
        robot = fixture.start()
        driver = fixture.drivers["right"]
        # A DISABLED status cached locally cannot acknowledge a disable frame.
        driver.links[2].responsive = False
        started = time.monotonic()
        with self.assertRaises((RuntimeError, TimeoutError)) as caught:
            robot.close()
        self.assertIn("right", str(caught.exception))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(fixture.drivers["left"].disabled.is_set())

    def test_close_rejects_a_single_motor_that_remains_enabled(self):
        fixture = self.fixture(shutdown_timeout_seconds=0.08)
        robot = fixture.start()
        fixture.drivers["right"].disabled_axes = set(range(7))
        started = time.monotonic()
        with self.assertRaises((RuntimeError, TimeoutError)) as caught:
            robot.close()
        self.assertIn("right", str(caught.exception))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(fixture.drivers["left"].disabled.is_set())

    def test_filter_tracks_actual_driver_clamp_and_copies_applied_targets(self):
        fixture = self.fixture()
        robot = fixture.start()
        right = fixture.drivers["right"]
        right.clamp_grip = -0.27
        robot.command(np.r_[np.full(7, 0.2), 0], np.r_[np.full(7, 0.2), 0])
        applied = robot.applied_targets()
        self.assertEqual(applied["right"][-1], -0.27)
        self.assertGreater(applied["right"][0], 0)
        self.assertLess(applied["right"][0], 0.2)
        applied["right"][:] = 99
        self.assertLess(robot.applied_targets()["right"][0], 1)

    def test_hold_discards_filter_and_retains_last_sent_gripper_without_unloading(self):
        fixture = self.fixture(control_timeout_seconds=0.04)
        robot = fixture.start()
        robot.command(np.r_[np.ones(7), 0], np.r_[np.ones(7), 0])
        before_hold = robot.applied_targets()
        for driver in fixture.drivers.values():
            driver.measured[:7] = -0.15
        robot.hold()
        held = robot.applied_targets()
        for side, driver in fixture.drivers.items():
            # Holding freezes the last successfully dispatched arm target;
            # rebasing to the sagged encoder would unload the arm.
            self.assertTrue(np.all(held[side][:7] >= before_hold[side][:7]))
            self.assertTrue(np.all(held[side][:7] < 1))
            # Deliberately different measured jaw: hold must retain command.
            self.assertNotEqual(held[side][-1], driver.measured[-1])
            driver.measured[:7] = -0.2
            count = len(driver.sent)
            driver.wait_sends(count + 12)  # Longer than heartbeat expiry.
            np.testing.assert_array_equal(driver.sent[-1], held[side])
        robot.samples()  # Successful hold does not depend on runtime heartbeat.

    def test_preload_hold_is_fixed_across_repeated_stop_and_gripper_commands(self):
        fixture = self.fixture(preload_window_seconds=0.02)
        robot = fixture.start()
        bias = np.r_[np.full(7, 0.06), 0]
        for driver in fixture.drivers.values():
            driver.measured[:7] -= bias[:7]
        for driver in fixture.drivers.values():
            driver.wait_sends(len(driver.sent) + 10)
        preview = robot.preload_bias(capture=False)
        self.assertIsNone(robot._arm_bias)
        learned = robot.preload_bias()
        for side in dispatch.SIDES:
            np.testing.assert_allclose(preview[side], bias)
            np.testing.assert_allclose(learned[side], bias)

        robot.command(np.r_[np.full(7, .3), -.2], np.r_[np.full(7, .3), .2])
        for driver in fixture.drivers.values():
            driver.measured[:7] = .1
        robot.hold()
        first = robot.applied_targets()
        for side in dispatch.SIDES:
            np.testing.assert_allclose(first[side][:7], .16)
        for driver in fixture.drivers.values():
            driver.measured[:7] = .05  # Later lag must not rebase repeated holds.
        robot.command_grippers({"right": -.1, "left": .1})
        robot.hold()
        robot.hold()
        for side in dispatch.SIDES:
            np.testing.assert_array_equal(robot.applied_targets()[side][:7], first[side][:7])
            np.testing.assert_allclose(robot.preload_bias()[side], bias)

        robot.command(np.r_[np.full(7, .2), -.1], np.r_[np.full(7, .2), .1])
        robot.hold()
        for side in dispatch.SIDES:
            np.testing.assert_allclose(robot.applied_targets()[side][:7], .11)

    def test_missing_preload_window_rejects_without_faulting_workers(self):
        fixture = self.fixture(preload_window_seconds=2)
        robot = fixture.start()
        with self.assertRaisesRegex(ValueError, "history is insufficient"):
            robot.preload_bias(capture=False)
        robot.hold()
        robot.samples()
        self.assertIsNone(robot._arm_bias)

    def test_active_target_heartbeat_expiry_disables_both(self):
        fixture = self.fixture(control_timeout_seconds=0.03)
        robot = fixture.start()
        robot.command(np.zeros(8), np.zeros(8))
        for driver in fixture.drivers.values():
            self.assertTrue(driver.disabled.wait(1))
            self.assertEqual(driver.stop_steps, 20)
        with self.assertRaisesRegex(RuntimeError, "heartbeat expired"):
            robot.samples()

    def test_encoder_never_responded_prevents_enable(self):
        def configure(driver):
            if driver.side == "right":
                driver.links[-1].responses = 0
                driver.links[-1].responsive = False

        fixture = self.fixture(configure=configure, max_feedback_seconds=0.02)
        with self.assertRaisesRegex(RuntimeError, "startup feedback exceeded"):
            fixture.start()
        self.assertTrue(all(not d.enabled for d in fixture.drivers.values()))
        self.assertTrue(all(not d.sent for d in fixture.drivers.values()))

    def test_startup_allows_responses_spread_over_multiple_refreshes(self):
        fixture = self.fixture(configure=lambda d: setattr(d, "respond_every", 3))
        fixture.start()
        self.assertTrue(all(d.fetch_count >= 3 for d in fixture.drivers.values()))

    def test_per_axis_stale_feedback_faults_even_while_fetch_returns_arrays(self):
        fixture = self.fixture(max_feedback_seconds=0.03)
        robot = fixture.start()
        last = fixture.drivers["right"].links[-1]
        last.responsive = False
        last.last_response = time.monotonic() - 0.1
        self.assertTrue(fixture.drivers["right"].disabled.wait(1))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            robot.samples()

    def test_bus_or_axis_write_failure_is_not_acknowledged_or_retained(self):
        for fault, counter in (
            ("write", "bus.write_no_buffer"),
            ("reject", "motor[7].rejected_commands"),
            ("malformed", "motor[2].malformed_frames"),
        ):
            with self.subTest(fault=fault):
                fixture = self.fixture()
                robot = fixture.start()
                old = robot.applied_targets()["right"].copy()
                fixture.drivers["right"].counter_fault = fault
                with self.assertRaisesRegex(RuntimeError, "counter fault") as caught:
                    robot.command(np.zeros(8), np.zeros(8))
                self.assertIn(f"{counter} 0->1", str(caught.exception))
                np.testing.assert_array_equal(robot.applied_targets()["right"], old)
                self.assertTrue(fixture.drivers["left"].disabled.wait(1))
                fixture.close()

    def test_hold_after_worker_fault_reports_cause_without_resuming_dispatch(self):
        fixture = self.fixture()
        robot = fixture.start()
        fixture.drivers["right"].counter_fault = "write"
        for driver in fixture.drivers.values():
            self.assertTrue(driver.disabled.wait(1))
        sent_counts = {side: len(driver.sent) for side, driver in fixture.drivers.items()}
        generation = robot._generation

        with self.assertRaisesRegex(RuntimeError, "bus.write_no_buffer 0->1"):
            robot.hold()

        self.assertEqual(robot._generation, generation)
        self.assertTrue(robot._stop.is_set())
        for side, driver in fixture.drivers.items():
            self.assertFalse(driver.enabled)
            self.assertEqual(driver.stop_steps, 20)
            self.assertEqual(len(driver.sent), sent_counts[side])

    def test_receive_error_after_send_retains_successfully_dispatched_grip(self):
        fixture = self.fixture()
        robot = fixture.start()
        driver = fixture.drivers["right"]
        driver.receive_error = True
        driver.clamp_grip = -0.24
        with self.assertRaisesRegex(RuntimeError, "receive failed"):
            robot.command(np.zeros(8), np.zeros(8))
        self.assertEqual(robot.applied_targets()["right"][-1], -0.24)

    def test_command_waits_for_both_actual_dispatches(self):
        fixture = self.fixture()
        robot = fixture.start()
        gate, entered, done = threading.Event(), threading.Event(), threading.Event()

        def block(_):
            entered.set()
            gate.wait(1)

        fixture.drivers["left"].before_send = block
        error = []

        def command():
            try:
                robot.command(np.zeros(8), np.zeros(8))
            except Exception as exc:
                error.append(exc)
            finally:
                done.set()

        caller = threading.Thread(target=command)
        caller.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(done.is_set())
        gate.set()
        caller.join(1)
        self.assertTrue(done.is_set())
        self.assertEqual(error, [])

    def test_feedback_timestamp_uses_oldest_axis_response(self):
        fixture = self.fixture()
        robot = fixture.start()
        driver = fixture.drivers["right"]
        driver.links[-1].responsive = False
        driver.links[-1].last_response = time.monotonic() - 0.08
        driver.wait_sends(len(driver.sent) + 2)
        sample = robot.samples()["right"]
        self.assertGreater(time.time() - sample.captured_at, 0.075)

    def test_partial_constructor_failure_never_enables_peer(self):
        fixture = self.fixture()
        original = dispatch.make_driver
        with patch.object(
            dispatch,
            "make_driver",
            side_effect=lambda side, p, i: (
                (_ for _ in ()).throw(OSError("missing left CAN"))
                if side == "left"
                else original(side, p, i)
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing left CAN"):
                fixture.start()
        self.assertFalse(fixture.drivers["right"].enabled)
        self.assertEqual(fixture.drivers["right"].sent, [])

    def test_opt_in_startup_counter_fault_cancels_before_more_waypoints(self):
        def configure(driver):
            if driver.side == "right":
                driver.counter_fault = "write"

        fixture = self.fixture(configure=configure, startup_trajectory=True)
        with self.assertRaisesRegex(RuntimeError, "startup CAN command counter fault") as caught:
            fixture.start()
        self.assertIn("bus.write_no_buffer 0->1", str(caught.exception))
        self.assertEqual(fixture.drivers["right"].startup_steps, 1)
        self.assertTrue(all(d.disabled.is_set() for d in fixture.drivers.values()))

    def test_startup_checker_is_removed_before_runtime_dispatch(self):
        def configure(driver):
            driver.original_checker = driver.safety_checker

        fixture = self.fixture(configure=configure, startup_trajectory=True)
        robot = fixture.start()
        for driver in fixture.drivers.values():
            self.assertIs(driver.safety_checker, driver.original_checker)
        robot.close()

    def test_dispatch_timeout_stops_workers_and_does_not_ack_a_queued_target(self):
        fixture = self.fixture(max_dispatch_seconds=0.02)
        robot = fixture.start()
        gate, entered = threading.Event(), threading.Event()

        def block(_):
            entered.set()
            gate.wait(1)

        fixture.drivers["left"].before_send = block
        try:
            with self.assertRaisesRegex(TimeoutError, "dispatch exceeded"):
                robot.command(np.zeros(8), np.zeros(8))
            self.assertTrue(entered.is_set())
        finally:
            gate.set()
        self.assertTrue(fixture.drivers["left"].disabled.wait(1))
        self.assertTrue(fixture.drivers["right"].disabled.wait(1))

    def test_invalid_rates_rejected_without_native_driver(self):
        for name, value in (
            ("send_hz", 0),
            ("cutoff_hz", math.inf),
            ("cutoff_hz", 101),
            ("startup_trajectory", "false"),
            ("stop_trajectory", "false"),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                dispatch.BimanualDispatcher({name: value})
