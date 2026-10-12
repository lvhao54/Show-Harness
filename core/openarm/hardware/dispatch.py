"""Thread-owned OpenArm CAN dispatch, adapted from AceOpenarm's real driver loop.

Each arm owns its native driver for its entire lifetime. Runtime targets are
acknowledged only after an actual send, and the low-pass filter follows the
vendor driver's final (possibly limited) command. Shutdown runs the configured
stop trajectory before disabling every enabled arm. All tests inject drivers;
importing this module opens no devices.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.openarm.hardware.preload import PreloadTracker

SIDES = ("right", "left")


def _positive(config, name, default, maximum):
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"OpenArm {name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"OpenArm {name} must be positive and at most {maximum}")
    return value


def _joints(value, description):
    values = np.asarray(value, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise RuntimeError(
            f"OpenArm {description} must contain eight finite joint values"
        )
    return values.copy()


def make_profile(name):
    from openarm_driver import Config

    return Config(name)


def make_driver(side, profile, interface):
    from openarm_driver import SingleArmDriver

    return SingleArmDriver(f"{side}_arm", profile, can_interface=interface)


def stop_trajectory_duration(profile):
    """Return the configured per-arm stop trajectory duration in seconds."""
    try:
        moves = profile.get_stop_config().get("moves", [])
    except Exception as exc:
        raise ValueError("OpenArm stop trajectory configuration is invalid") from exc
    if not isinstance(moves, list):
        raise ValueError("OpenArm stop trajectory moves must be a list")
    duration = 0.0
    for index, move in enumerate(moves):
        if not isinstance(move, dict):
            raise ValueError(f"OpenArm stop move {index} must be a mapping")
        value = move.get("duration")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(
                f"OpenArm stop move {index} duration must be a nonnegative finite number"
            )
        duration += float(value)
    return duration


def enable_at_measurement(driver):
    """Enable without the vendor start hook, which otherwise executes homing."""
    import openarm_can as oa

    driver.openarm.set_callback_mode_all(oa.CallbackMode.STATE)
    driver.openarm.enable_all()
    driver.set_latest_state(timeout_us=500)
    driver.openarm.refresh_all()
    driver.set_latest_state(timeout_us=500)
    driver.last_command = _joints(driver.latest_state["qpos"], "startup feedback")
    driver.last_command_time_s = time.monotonic()
    driver.last_command_dispatch_timestamp_ns = None
    driver.started = True


@dataclass(frozen=True)
class ArmSample:
    qpos: np.ndarray
    captured_at: float
    captured_monotonic: float


class BimanualDispatcher:
    """Bounded mailbox plus independent 200 Hz send/feedback loops.

    ``command`` waits until both arms have sent the submitted generation. It
    does not imply the filtered endpoint was reached; ``applied_targets`` is
    always the last command actually accepted by the vendor. A hold replaces
    the mailbox and bypasses the filter at the next native call boundary.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.send_hz = _positive(config, "send_hz", 200, 1000)
        self.cutoff_hz = _positive(config, "cutoff_hz", 5, self.send_hz / 2)
        self.feedback_timeout = _positive(config, "max_feedback_seconds", 0.2, 2)
        self.dispatch_timeout = _positive(config, "max_dispatch_seconds", 0.2, 2)
        self.control_timeout = _positive(config, "control_timeout_seconds", 0.5, 5)
        self.startup_timeout = _positive(config, "startup_timeout_seconds", 30, 120)
        self.shutdown_timeout = _positive(config, "shutdown_timeout_seconds", 5, 30)
        self.stop_duration = 0.0
        self._preload = PreloadTracker(
            window=_positive(config, "preload_window_seconds", 0.3, 2),
            stability_tolerance=_positive(config, "preload_stability_tolerance", 0.005, 0.02),
            max_offset=_positive(config, "preload_max_offset", 0.15, 0.2),
        )
        self._arm_bias: dict[str, np.ndarray] | None = None
        self._held_arms: dict[str, np.ndarray] = {}
        for name in ("right_can_interface", "left_can_interface"):
            value = config.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"OpenArm {name} must be a nonempty interface name")
        if type(config.get("startup_trajectory", False)) is not bool:
            raise ValueError("OpenArm startup_trajectory must be a boolean")
        self._condition = threading.Condition()
        self._submit_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._errors: list[BaseException] = []
        self._constructed: set[str] = set()
        self._preflight_ready: set[str] = set()
        self._ready: set[str] = set()
        self._samples: dict[str, ArmSample] = {}
        self._applied: dict[str, np.ndarray] = {}
        self._mailbox: dict[str, tuple[int, np.ndarray | None]] = {}
        self._acked = dict.fromkeys(SIDES, -1)
        self._generation = 0
        self._control_active = False
        self._last_control = 0.0
        # A single deadline bounds all shutdown work after the first stop or
        # worker failure. It prevents a late close() from granting another
        # full timeout while a worker is already trying to disable its arm.
        self._shutdown_deadline: float | None = None
        self._shutdown_pending: dict[str, str] = {}

    def _error_locked(self):
        if self._errors:
            raise RuntimeError(
                f"OpenArm CAN worker failed: {self._errors[0]}"
            ) from self._errors[0]
        if self._stop.is_set():
            raise RuntimeError("OpenArm CAN dispatch has stopped")

    def _fail(self, error):
        with self._condition:
            self._errors.append(error)
            self._stop.set()
            if self._shutdown_deadline is None:
                self._shutdown_deadline = (
                    time.monotonic() + self.stop_duration + self.shutdown_timeout
                )
            self._condition.notify_all()

    def _run_stop_trajectory(self, side, driver):
        if not getattr(driver, "started", False):
            return
        before = self._counters(driver)
        if driver.move_to_stop_position() is False:
            raise RuntimeError(f"{side} stop trajectory was rejected")
        self._sample(side, driver)
        changes = self._counter_changes(before, self._counters(driver))
        if changes:
            raise RuntimeError(f"{side} stop CAN command counter fault: {changes}")

    def _disable_and_confirm(self, side, driver):
        """Disable one arm and verify fresh DISABLED feedback for every axis.

        ``get_health()`` may expose the native cache, so a cached DISABLED
        state cannot acknowledge the disable command. Record per-axis response
        counts before the first ``disable_all()``, then refresh state on every
        retry and require every axis to both answer and report
        DISABLED. Retries stay on the driver's owner thread until the shared
        shutdown deadline expires.
        """
        with self._condition:
            deadline = self._shutdown_deadline
        if deadline is None:
            deadline = time.monotonic() + self.shutdown_timeout

        baseline = None
        with self._condition:
            self._shutdown_pending[side] = "disable confirmation pending"
        while True:
            try:
                try:
                    if baseline is None:
                        baseline = [link.responses for link in self._links(driver)]
                finally:
                    # Always attempt disable, even if diagnostics fail or a
                    # preceding native call returned after the deadline.
                    driver.openarm.disable_all()
                # Refresh after every disable attempt; status alone is not
                # enough because the native object can retain old responses.
                driver.fetch_state(refresh=True)
                statuses, bus = driver.get_health()
                links = self._links(driver)
                fresh = [
                    index
                    for index, (link, count) in enumerate(zip(links, baseline))
                    if link.responses <= count
                ]
                enabled = [
                    index
                    for index, status in enumerate(statuses)
                    if status != "DISABLED"
                ]
                if (
                    not driver.openarm.is_link_running()
                    or not bus.get("carrier", False)
                ):
                    raise RuntimeError(f"{side} CAN carrier down during shutdown")
                if len(statuses) != 8:
                    raise RuntimeError(
                        f"{side} shutdown returned {len(statuses)} motor statuses"
                    )
                if fresh:
                    raise RuntimeError(
                        f"{side} shutdown feedback stale on motor indices {fresh}"
                    )
                if enabled:
                    raise RuntimeError(
                        f"{side} motors still enabled at indices {enabled}: {statuses}"
                    )
                with self._condition:
                    self._shutdown_pending.pop(side, None)
                return
            except Exception as exc:
                with self._condition:
                    self._shutdown_pending[side] = str(exc)
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(0.01, remaining))
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{side} motors did not confirm DISABLED before shutdown "
                        f"deadline ({exc})"
                    ) from exc

    def start(self):
        if self._threads:
            raise RuntimeError("OpenArm dispatcher cannot be started twice")
        profile = make_profile(
            self.config.get("driver_config")
            or self.config.get("profile", "openarm_pedestal")
        )
        self.stop_duration = stop_trajectory_duration(profile)
        interfaces = {
            side: self.config.get(f"{side}_can_interface")
            or profile.get_can_interface(f"{side}_arm")
            for side in SIDES
        }
        if interfaces["right"] == interfaces["left"]:
            raise ValueError("OpenArm arms must use separate CAN interfaces")
        self._threads = [
            threading.Thread(
                target=self._run_arm,
                args=(side, profile, interfaces[side]),
                name=f"openarm-{side}",
                daemon=True,
            )
            for side in SIDES
        ]
        try:
            for thread in self._threads:
                thread.start()
            deadline = time.monotonic() + self.startup_timeout
            with self._condition:
                while self._ready != set(SIDES):
                    self._error_locked()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("OpenArm startup exceeded its deadline")
                    self._condition.wait(remaining)
                self._error_locked()
                self._last_control = time.monotonic()
                self._control_active = False
        except BaseException as exc:
            self._fail(exc)
            try:
                self.close()
            except Exception:
                pass  # Preserve the failure that caused startup to abort.
            raise

    @staticmethod
    def _links(driver):
        collections = [(driver.openarm.get_arm(), driver.num_mit_motors)]
        if driver.gripper_posforce:
            collections.append((driver.openarm.get_gripper(), 1))
        links = [
            collection.get_link_stats(index)
            for collection, count in collections
            for index in range(count)
        ]
        if len(links) != 8:
            raise RuntimeError("OpenArm driver must expose eight motor link states")
        return links

    @classmethod
    def _counters(cls, driver):
        bus = driver.openarm.get_bus_status()
        counters = {
            f"bus.{name}": getattr(bus, name).count
            for name in ("write_no_buffer", "write_net_down", "write_other")
        }
        for index, link in enumerate(cls._links(driver)):
            for name in ("rejected_commands", "malformed_frames"):
                counters[f"motor[{index}].{name}"] = getattr(link, name)
        return counters

    @staticmethod
    def _counter_changes(before, after):
        return ", ".join(
            f"{name} {before[name]}->{value}"
            for name, value in after.items()
            if value != before[name]
        )

    def _fresh_startup_sample(self, side, driver, *, enabled, baseline):
        deadline = time.monotonic() + self.feedback_timeout
        while True:
            if self._stop.is_set():
                raise RuntimeError("OpenArm startup cancelled")
            try:
                return self._sample(side, driver, enabled=enabled, fresh_after=baseline)
            except RuntimeError as exc:
                # Serial CAN replies can span several bounded recv windows.
                # Only incomplete/stale feedback is retryable; motor/bus faults
                # abort immediately without enabling or starting a trajectory.
                if not any(
                    text in str(exc)
                    for text in (
                        "stale",
                        "never responded",
                        "incomplete fresh",
                        "enable feedback pending",
                    )
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{side} startup feedback exceeded its deadline"
                    ) from exc
                self._stop.wait(min(0.001, max(0, deadline - time.monotonic())))

    def _sample(self, side, driver, *, enabled=True, fresh_after=None):
        # Native motor caches have independent ages. A fast fetch is not proof
        # that all eight encoders have responded, especially during startup.
        started = time.monotonic()
        state = driver.fetch_state(refresh=True)
        joints = _joints(state["qpos"], f"{side} feedback")
        statuses, bus = driver.get_health()
        if not driver.openarm.is_link_running() or not bus.get("carrier", False):
            raise RuntimeError(f"{side} CAN carrier down")
        allowed = {"ENABLED"} if enabled else {"ENABLED", "DISABLED"}
        if len(statuses) != 8 or any(status not in allowed for status in statuses):
            if (
                fresh_after is not None
                and enabled
                and len(statuses) == 8
                and all(status in {"ENABLED", "DISABLED"} for status in statuses)
            ):
                raise RuntimeError(f"{side} enable feedback pending")
            raise RuntimeError(f"{side} motor status {statuses}")
        links = self._links(driver)
        if any(
            not link.ever_responded() or link.is_stale(self.feedback_timeout)
            for link in links
        ):
            raise RuntimeError(f"{side} encoder feedback is stale or never responded")
        if fresh_after is not None and any(
            link.responses <= count for link, count in zip(links, fresh_after)
        ):
            raise RuntimeError(f"{side} has incomplete fresh encoder responses")
        now_mono, now_wall = time.monotonic(), time.time()
        age = max(float(link.seconds_since_response()) for link in links)
        if not math.isfinite(age) or age < 0 or age > self.feedback_timeout:
            raise RuntimeError(f"{side} encoder feedback is stale")
        if now_mono - started > self.feedback_timeout:
            raise TimeoutError(f"{side} encoder feedback exceeded its deadline")
        sample = ArmSample(joints, now_wall - age, now_mono - age)
        if enabled:
            with self._condition:
                self._samples[side] = sample
        return sample

    def _send(self, side, driver, target):
        before = self._counters(driver)
        timestamp = getattr(driver, "last_command_dispatch_timestamp_ns", None)
        error = None
        try:
            if driver.send_position(target) is False:
                raise RuntimeError(
                    f"{side}-arm command rejected by driver safety: "
                    f"{getattr(driver, 'safety_stop_reason', 'unknown')}"
                )
        except BaseException as exc:
            error = exc
        changes = self._counter_changes(before, self._counters(driver))
        if changes:
            raise RuntimeError(
                f"{side} CAN write or motor command counter fault: {changes}"
            ) from error
        sent = getattr(driver, "last_command_dispatch_timestamp_ns", None)
        # A receive error may occur after the native write. Retain the actually
        # sent target before propagating that feedback fault.
        if error is None or (sent is not None and sent != timestamp):
            applied = _joints(driver.last_command, f"{side} applied command")
            with self._condition:
                self._applied[side] = applied
        if error is not None:
            raise error
        return applied

    def _run_arm(self, side, profile, interface):
        driver = None
        try:
            # Native construction, enable, send, receive and disable all occur
            # on this one thread, including startup and failure cleanup.
            driver = make_driver(side, profile, interface)
            with self._condition:
                self._constructed.add(side)
                self._condition.notify_all()
                while self._constructed != set(SIDES) and not self._stop.is_set():
                    self._condition.wait()
                if self._stop.is_set():
                    return
            baseline = [link.responses for link in self._links(driver)]
            self._fresh_startup_sample(side, driver, enabled=False, baseline=baseline)
            with self._condition:
                self._preflight_ready.add(side)
                self._condition.notify_all()
                while self._preflight_ready != set(SIDES) and not self._stop.is_set():
                    self._condition.wait()
                if self._stop.is_set():
                    return
            # Split vendor start so fresh ENABLED feedback is checked before
            # any opt-in startup trajectory can use the native state cache.
            baseline = [link.responses for link in self._links(driver)]
            enable_at_measurement(driver)
            startup_sample = self._fresh_startup_sample(
                side, driver, enabled=True, baseline=baseline
            )
            # The native enable helper may have seeded last_command from an
            # older cache read. Rebase filtering and any opt-in startup path on
            # the feedback that passed the post-enable freshness gate.
            driver.last_command = startup_sample.qpos.copy()
            driver.last_command_time_s = time.monotonic()
            driver.last_command_dispatch_timestamp_ns = None
            if self.config.get("startup_trajectory", False):
                from openarm_driver import CheckResult, CompositeChecker

                dispatcher = self
                before = self._counters(driver)
                original_checker = driver.safety_checker

                class StartupCancellation:
                    def check(self, _position, **_kwargs):
                        try:
                            changes = dispatcher._counter_changes(
                                before, dispatcher._counters(driver)
                            )
                            if changes:
                                raise RuntimeError(
                                    f"{side} startup CAN command counter fault: {changes}"
                                )
                            statuses, bus = driver.get_health()
                            if not bus.get("carrier", False) or any(
                                status != "ENABLED" for status in statuses
                            ):
                                raise RuntimeError(f"{side} startup motor/bus fault")
                            if any(
                                link.is_stale(dispatcher.feedback_timeout)
                                for link in dispatcher._links(driver)
                            ):
                                raise RuntimeError(f"{side} startup feedback is stale")
                        except Exception as exc:
                            dispatcher._fail(exc)
                        if dispatcher._stop.is_set():
                            return CheckResult(
                                is_safe=False,
                                force_stop=True,
                                message="OpenArm startup cancelled",
                            )
                        return CheckResult(is_safe=True)

                # The startup checker protects only the opt-in home trajectory.
                # Do not leave its startup counter baseline installed for the
                # runtime loop: normal CAN health is checked by _send(), and a
                # later counter increment would otherwise be mislabeled as a
                # startup fault on every subsequent command.
                driver.safety_checker = CompositeChecker(
                    [StartupCancellation(), original_checker]
                )
                try:
                    if driver.move_to_start_position() is False:
                        raise RuntimeError(f"{side} startup trajectory was rejected")
                    changes = self._counter_changes(before, self._counters(driver))
                    if changes:
                        raise RuntimeError(
                            f"{side} startup CAN command counter fault: {changes}"
                        )
                finally:
                    driver.safety_checker = original_checker
            if self._stop.is_set():
                return
            filtered = _joints(driver.last_command, f"{side} startup command")
            self._sample(side, driver)
            with self._condition:
                self._mailbox[side] = (0, filtered.copy())
            interval = 1 / self.send_hz
            alpha = -math.expm1(-2 * math.pi * self.cutoff_hz * interval)
            deadline = time.monotonic()
            hold_generation = -1
            while not self._stop.is_set():
                with self._condition:
                    if (
                        self._control_active
                        and time.monotonic() - self._last_control > self.control_timeout
                    ):
                        raise TimeoutError("OpenArm runtime control heartbeat expired")
                    generation, raw = self._mailbox[side]
                    if raw is not None:
                        raw = raw.copy()
                    # A hold wakes immediately; normal command updates respect
                    # the fixed send clock instead of bursting at runtime rate.
                    immediate_hold = raw is None and generation != hold_generation
                    remaining = deadline - time.monotonic()
                    if remaining > 0 and not immediate_hold:
                        self._condition.wait(remaining)
                        continue
                holding = raw is None
                if holding:
                    sample = self._sample(side, driver)
                    with self._condition:
                        raw = self._applied[side].copy()
                        if side in self._held_arms:
                            raw[:7] = self._held_arms[side]
                        elif self._arm_bias is not None:
                            raw[:7] = sample.qpos[:7] + self._arm_bias[side][:7]
                        # Before a preload is validated, freeze the last sent
                        # target. Never unload the PD loop by zeroing its bias.
                    filtered = raw.copy()
                    hold_generation = generation
                    with self._condition:
                        if self._mailbox[side][0] == generation:
                            self._mailbox[side] = (generation, raw.copy())
                else:
                    filtered = filtered + alpha * (raw - filtered)
                with self._condition:
                    if self._stop.is_set():
                        break
                    # Do not send a superseded copied target after a hold has
                    # replaced the mailbox at a native-call boundary.
                    if self._mailbox[side][0] != generation:
                        continue
                filtered = self._send(side, driver, filtered)
                sample = self._sample(side, driver)
                with self._condition:
                    if self._arm_bias is None:
                        # Use collection time for the stable-window clock;
                        # per-axis sample freshness is checked in _sample.
                        self._preload.add(
                            side, sample.qpos, filtered, time.monotonic()
                        )
                    if holding and self._mailbox[side][0] == generation:
                        # A safety-limited hold must remain at the command that
                        # was actually sent, rather than retaining filter work.
                        self._mailbox[side] = (generation, filtered.copy())
                        self._held_arms[side] = filtered[:7].copy()
                    self._acked[side] = generation
                    self._ready.add(side)
                    self._condition.notify_all()
                deadline += interval
                now = time.monotonic()
                if deadline <= now:
                    # Skip expired slots, never catch up with a burst of sends.
                    deadline += (math.floor((now - deadline) / interval) + 1) * interval
        except BaseException as exc:
            self._fail(exc)
        finally:
            if driver is not None:
                try:
                    # Run the configured stop trajectory before disabling. This
                    # is the same native-owner thread used for all motor I/O, so
                    # no runtime command can race the shutdown sequence.
                    self._run_stop_trajectory(side, driver)
                except BaseException as exc:
                    self._fail(exc)
                try:
                    self._disable_and_confirm(side, driver)
                except BaseException as exc:
                    self._fail(exc)
                finally:
                    driver.started = False
            with self._condition:
                self._condition.notify_all()

    def _submit(self, targets, *, arm_motion=False):
        with self._submit_lock:
            with self._condition:
                self._error_locked()
                if self._ready != set(SIDES):
                    raise RuntimeError("OpenArm dispatcher is not ready")
                if arm_motion:
                    self._held_arms.clear()
                self._generation += 1
                generation = self._generation
                self._last_control = time.monotonic()
                self._control_active = targets["right"] is not None
                for side in SIDES:
                    self._mailbox[side] = (generation, targets[side])
                self._condition.notify_all()
                deadline = time.monotonic() + self.dispatch_timeout
                while any(self._acked[side] < generation for side in SIDES):
                    self._error_locked()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        error = TimeoutError(
                            "OpenArm target dispatch exceeded its deadline"
                        )
                        self._fail(error)
                        raise error
                    self._condition.wait(remaining)
                self._error_locked()

    def command(self, right, left):
        self._submit(
            {
                "right": _joints(right, "right target"),
                "left": _joints(left, "left target"),
            },
            arm_motion=True,
        )

    def command_grippers(self, grippers):
        """Dispatch only jaw channels and retain the last arm joint command."""
        # Prefer the current mailbox target. It may be ahead of the last
        # physical send while an arm move is filtering toward its endpoint;
        # rebasing from ``_applied`` here would step that arm backwards. When
        # the mailbox is a hold, use the last applied command instead.
        with self._condition:
            self._error_locked()
            targets = {}
            for side in SIDES:
                _generation, raw = self._mailbox.get(side, (-1, None))
                if raw is None:
                    raw = self._applied.get(side)
                if raw is None:
                    raise RuntimeError("OpenArm has no complete applied command state")
                targets[side] = raw.copy()
        targets["right"][-1] = float(grippers["right"])
        targets["left"][-1] = float(grippers["left"])
        self._submit(targets)

    def preload_bias(self, *, capture=True):
        """Latch a stable, bounded support offset once for this connection."""
        with self._condition:
            self._error_locked()
            now = time.monotonic()
            if set(self._samples) != set(SIDES):
                raise RuntimeError("OpenArm has no complete joint feedback")
            if any(
                now - sample.captured_monotonic > self.feedback_timeout
                for sample in self._samples.values()
            ):
                raise RuntimeError("OpenArm encoder feedback is stale")
            if self._arm_bias is None:
                if not capture:
                    return self._preload.preview()
                self._arm_bias = self._preload.capture()
            return {side: bias.copy() for side, bias in self._arm_bias.items()}

    def hold(self):
        self._submit(dict.fromkeys(SIDES))

    def heartbeat(self):
        with self._condition:
            self._error_locked()
            self._last_control = time.monotonic()

    def samples(self):
        with self._condition:
            self._error_locked()
            now = time.monotonic()
            if set(self._samples) != set(SIDES):
                raise RuntimeError("OpenArm has no complete joint feedback")
            if any(
                now - sample.captured_monotonic > self.feedback_timeout
                for sample in self._samples.values()
            ):
                raise RuntimeError("OpenArm encoder feedback is stale")
            return {
                side: ArmSample(
                    sample.qpos.copy(), sample.captured_at, sample.captured_monotonic
                )
                for side, sample in self._samples.items()
            }

    def applied_targets(self):
        # Remain readable after a fault, preserving partial dispatch evidence.
        with self._condition:
            if set(self._applied) != set(SIDES):
                raise RuntimeError("OpenArm has no complete applied command state")
            return {side: joints.copy() for side, joints in self._applied.items()}

    def close(self):
        with self._condition:
            self._stop.set()
            if self._shutdown_deadline is None:
                self._shutdown_deadline = (
                    time.monotonic() + self.stop_duration + self.shutdown_timeout
                )
            deadline = self._shutdown_deadline
            self._condition.notify_all()
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(max(0, deadline - time.monotonic()))
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        with self._condition:
            shutdown = "; ".join(
                f"{side} shutdown disable confirmation failed: {detail}"
                for side, detail in self._shutdown_pending.items()
            )
            failure = (
                f"OpenArm CAN worker failed: {self._errors[0]}"
                if self._errors else ""
            )
            if alive:
                details = "; ".join(part for part in (failure, shutdown) if part)
                raise TimeoutError(
                    f"OpenArm CAN workers did not stop: {alive}"
                    + (f"; {details}" if details else "")
                ) from (self._errors[0] if self._errors else None)
            if self._errors:
                raise RuntimeError(
                    failure + (f"; {shutdown}" if shutdown else "")
                ) from self._errors[0]
