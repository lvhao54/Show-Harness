"""Build the real-arm atomic controller against the OpenArm simulation.

The OpenArm sim is driven by the *unmodified* real-arm control stack
(:class:`~interpreters.real_atomic_controller.RealAtomicController`): the session's
:class:`~core.openarm.openarm_interface.OpenArmSimRobot` implements the same four-method
robot duck type the hardware adapters do, so setpoint integration, the Z floor,
empty-grasp detection and the plugin mounts are the code that runs on a real arm.

Token semantics come from ``configs/primitives_openarm.yaml`` (axis signs measured against
the head camera, the per-token step, and the safety clamps). The fine-tuned entry point
carries the same values under ``move_vectors`` in ``configs/robot_openarm.yaml``; a test
asserts the two stay in sync.

Two entry points build controllers from here -- ``scripts/run_openarm_mvtoken.py`` (one
action token per step) and the zero-shot path in :mod:`core.launch` -- so both get the
same defaults, and the robot config is the only place a value is written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]

#: Fallback primitives file, matching ``resolve_primitives_path``'s OpenArm default.
DEFAULT_PRIMITIVES = ROOT / "configs" / "primitives_openarm.yaml"

#: Sentinel: "derive the Z floor from the config" (as opposed to an explicit ``None``,
#: which means the caller already resolved "floor disabled").
_UNSET: Any = object()


def make_openarm_controller(
    cfg: dict[str, Any],
    robot: Any,
    primitives_cfg: Optional[dict[str, Any]] = None,
    *,
    z_floor_m: Any = _UNSET,
    extra_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> Any:
    """Build a :class:`RealAtomicController` for the OpenArm sim from a resolved config.

    ``cfg`` is the resolved robot config (the keys ``configs/robot_openarm.yaml``
    documents). ``primitives_cfg`` defaults to ``configs/primitives_openarm.yaml``.

    ``z_floor_m`` uses a sentinel so "not given" (derive from the config) is distinct from
    an explicit ``None`` (floor disabled): the zero-shot launcher resolves
    ``--no-z-floor`` / ``--z-floor-m`` before construction and passes the result here.

    ``extra_kwargs`` is anything else the caller already resolved -- the zero-shot harness
    threads its plugin mounts (``variable_step_plugin``, ``smooth_plugin``, ``up_step_m``,
    ``table_height_m``, ...) through it. Everything is merged into ONE kwargs mapping
    before the call, so a value present on both sides is simply overridden, never passed
    twice.
    """
    from core.config import load_yaml
    from interpreters.real_atomic_controller import RealAtomicController

    if primitives_cfg is None:
        primitives_cfg = load_yaml(DEFAULT_PRIMITIVES)
    if z_floor_m is _UNSET:
        z_floor_m = cfg.get("z_floor_m") if bool(cfg.get("enable_z_floor", True)) else None

    kwargs: dict[str, Any] = dict(
        motion_frame=str(cfg.get("motion_frame", "base")),
        step_m=float(cfg.get("step_m", primitives_cfg.get("step_m", 0.02))),
        yaw_step_rad=float(primitives_cfg.get("yaw_step_rad", 0.15)),
        max_position_delta_m=float(
            primitives_cfg.get("osc_translation_output_max_m", 0.05)
        ),
        max_rotation_delta_rad=float(
            primitives_cfg.get("osc_rotation_output_max_rad", 0.5)
        ),
        # This backend's own settle contract, NOT the real-rig ``robot:`` block: the sim
        # steps physics instead of waiting on hardware. One atomic token costs
        # settle_steps x sim_ticks_per_command physics ticks.
        settle_steps=int(cfg.get("settle_steps", 6)),
        settle_dt_s=float(cfg.get("settle_dt_s", 0.001)),
        gripper_close_threshold_m=float(cfg.get("gripper_close_threshold_m", 0.02)),
        grasp_open_width_m=float(cfg.get("open_width_m", 0.06)),
        gripper_settle_s=float(cfg.get("gripper_settle_s", 0.0)),
        gripper_min_settle_s=float(cfg.get("gripper_min_settle_s", 0.0)),
        z_floor_m=None if z_floor_m is None else float(z_floor_m),
        verbose=bool(verbose),
    )
    if cfg.get("empty_width_m") is not None:
        kwargs["grasp_min_width_m"] = float(cfg["empty_width_m"])
    # The caller's already-resolved values win (they may come from CLI flags or plugins).
    kwargs.update(extra_kwargs or {})
    return RealAtomicController.from_primitives_config(robot, primitives_cfg, **kwargs)
