#!/usr/bin/env python3
"""Show-Harness OpenArm simulation: MVTOKEN atomic-token policy (no planner/stage).

Simulator sibling of ``scripts/run_maniskill_mvtoken.py``: the same flat MVTOKEN control
loop, driving an OpenArm bimanual rig in MuJoCo through the official ``openarm_control``
inverse kinematics. It builds a :class:`~core.openarm.openarm_session.OpenArmSession` and
drives it with the **unmodified** :class:`~core.runners.mvtoken.MvTokenRunner` -- the
session answers the same ``get_observation()`` contract the real sessions do, and
:class:`~core.openarm.openarm_interface.OpenArmSimRobot` implements the four-method robot
duck type the real-arm controller already speaks. The whole real stack therefore runs
here unchanged: setpoint integration, the Z safety floor, empty-grasp detection, and the
``auto_release`` plugin.

The simulator is self-contained: the MuJoCo physics/IK/camera glue is vendored under
``core/openarm/backend/`` and the robot model comes from the ``openarm-mujoco`` wheel, so
no external checkout is needed. The ``insert_flower`` scene additionally needs its
VLABench meshes, which are not redistributable with this repo -- fetch them once with
``bash scripts/model/download_vlabench_assets.sh``.

CROSS-DOMAIN experiment: the released ``_sim`` LoRA was trained on RoboLab + ManiSkill
Franka/Piper rollouts, so behaviour on a bimanual OpenArm with a different gripper and
camera geometry is exploratory. Verify the two image contracts and the step size before
reading anything into a success rate:

    python scripts/run_openarm_mvtoken.py --dump-views
    python scripts/run_openarm_mvtoken.py --probe-axes

Start the vLLM server first (Terminal 1) -- serve the adapter the config's vlm_backend
names, on :8000 (scripts/serve_vlm.sh):
    MODEL=<base> FAMILY=qwen3_5 \\\\
      LORA=qwen3_5_2b_showharness_sim=<path> bash scripts/serve_vlm.sh

Then (Terminal 2):
    python scripts/run_openarm_mvtoken.py --version v3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The entry lives one level below the repo root; make the root importable
# so core/, plugins/ and friends resolve when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_secrets_env, load_yaml
from core.launch import openarm_backend, openarm_wrist_transform
from core.openarm.openarm_session import OpenArmSession, OpenArmSessionConfig
from core.record.episode_logger import EpisodeLogger
from core.runners.mvtoken import MvTokenRunner
from core.vlm.mvtoken_roles import MvTokenController
from interpreters.real_atomic_controller import RealAtomicController
from plugins.auto_release import AutoReleasePlugin
from plugins.config import PluginsConfig

# Shared sim config/client wiring (no startup calibration -- the atomic controller maps
# MV_* -> base-frame XYZ directly).
from core.sim.launch import build_config, make_vlm_client

#: Where rollouts land when the config names no ``log_dir`` (the real config does not).
DEFAULT_LOG_DIR = "rollouts/openarm"
#: Step budget when the config names no ``max_steps``.
DEFAULT_MAX_STEPS = 80


def _prompt_path(prompts_dir: Path, version: str) -> Path:
    """``prompts/<version>/mvtoken_generator_lite.txt`` with the v4 per-embodiment fallback.

    v4 ships no shared lite prompt, so it falls back to the Franka overhead one -- the
    OpenArm agentview is also a fixed third-person view, matching that framing.
    """
    base = prompts_dir / version
    names = ("mvtoken_generator_lite.txt", "franka_mvtoken_lite.txt")
    for name in names:
        candidate = base / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"No lite prompt under {base} (looked for {' / '.join(names)}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show-Harness OpenArm sim: MVTOKEN atomic-token policy (no planner/stage)"
    )
    parser.add_argument(
        "--robot-config", default=str(ROOT / "configs" / "robot_openarm.yaml")
    )
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument(
        "--version",
        default="v3",
        help="Prompt version subfolder under prompts/ (e.g. v3, v4), matching the served "
        "LoRA's training data.",
    )
    parser.add_argument(
        "--scene", default=None,
        help="OpenArm scene: omit for the default three-cube tabletop, or `insert_flower`.",
    )
    parser.add_argument("--arm", default=None, choices=(None, "left", "right"))
    parser.add_argument(
        "--success-check", default=None,
        help="none | cubes_on_table | red_on_green | flowers_in_vase (default: the config's value).",
    )
    parser.add_argument(
        "--z-floor-m", type=float, default=None,
        help="Override the config's control-point Z floor (arm_origin frame, metres).",
    )
    parser.add_argument("--no-z-floor", action="store_true", help="Disable the Z floor.")
    parser.add_argument(
        "--vlm-backend", default=os.environ.get("VLM_BACKEND")
    )
    parser.add_argument(
        "--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL")
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--probe-axes", action="store_true",
        help="Before the rollout, step each MV_* token in isolation and record the observed "
        "control-point delta into calibration.json (verifies the axis/sign mapping).",
    )
    parser.add_argument(
        "--dump-views", action="store_true",
        help="Write the two prepared views to the run dir and exit, so the camera framing "
        "and transforms can be checked by eye before any rollout.",
    )
    parser.add_argument(
        "--prompt-log-every", type=int, default=20,
        help="Save the exact VLM prompt + returned token every N steps (0 disables).",
    )
    parser.add_argument(
        "--debug", action="store_true", default=os.environ.get("DEBUG") == "1"
    )
    # build_config reads these; the OpenArm path does not use them.
    parser.set_defaults(task_suite_name=None, task_id=None, episode_index=None)
    return parser.parse_args()


def _session_config(cfg: dict, args: argparse.Namespace) -> OpenArmSessionConfig:
    arm_side = str(args.arm or cfg.get("arm_side", "right"))
    wrist_rotation, wrist_flip = openarm_wrist_transform(cfg, arm_side)
    return OpenArmSessionConfig(
        scene=args.scene if args.scene is not None else cfg.get("scene"),
        arm_side=arm_side,
        control_hz=float(cfg.get("control_hz", 60.0)),
        sim_ticks_per_command=int(cfg.get("sim_ticks_per_command", 6)),
        gripper_ticks=int(cfg.get("gripper_ticks", 60)),
        settle_ticks=int(cfg.get("settle_ticks", 60)),
        observation_resolution=(
            int(cfg["observation_resolution"]) if cfg.get("observation_resolution") else None
        ),
        hd_resolution=int(cfg["hd_resolution"]) if cfg.get("hd_resolution") else None,
        agentview_rotation_degrees=int(cfg.get("agentview_rotation_degrees", 0)),
        wrist_rotation_degrees=wrist_rotation,
        agentview_flip=str(cfg.get("agentview_flip", "none")),
        wrist_flip=wrist_flip,
        agentview_crop_aspect=cfg.get("agentview_crop_aspect"),
        wrist_crop_aspect=cfg.get("wrist_crop_aspect"),
        success_check=str(args.success_check or cfg.get("success_check", "none")),
    )


def _make_controller(cfg: dict, args: argparse.Namespace, robot) -> RealAtomicController:
    """The real-arm controller, unchanged -- only the robot duck type is simulated.

    Token semantics come from the shared ``configs/primitives_openarm.yaml``, with the
    robot config's own ``move_vectors`` layered on top when present (the same convention
    ``configs/robot_maniskill.yaml`` and ``robot_robolab.yaml`` use). The robot config is
    authoritative because the axis signs follow the CAMERA: the OpenArm head view's
    image-left is world +Y, which differs from the real Franka rig the Franka primitives
    file was written for. Verify with ``--probe-axes`` after any change to the scene,
    camera or mount.
    """
    from core.action_units import MOVE_ATOMS
    from core.config import load_yaml as _load
    from core.openarm.controller import DEFAULT_PRIMITIVES, make_openarm_controller

    prims = _load(cfg.get("primitives_config") or str(DEFAULT_PRIMITIVES))
    move_vectors = cfg.get("move_vectors") or {}
    missing = [name for name in MOVE_ATOMS if name not in move_vectors]
    if missing and missing != list(MOVE_ATOMS):
        raise ValueError(
            f"move_vectors must define all of {list(MOVE_ATOMS)}; missing {missing}"
        )
    if not missing:
        prims = dict(prims)
        prims["atomic_primitives"] = {
            **prims.get("atomic_primitives", {}),
            **{name: list(move_vectors[name]) for name in MOVE_ATOMS},
        }
    if args.no_z_floor:
        z_floor = None
    elif args.z_floor_m is not None:
        z_floor = args.z_floor_m
    elif not bool(cfg.get("enable_z_floor", True)):
        z_floor = None
    else:
        z_floor = cfg.get("z_floor_m")
    return make_openarm_controller(cfg, robot, prims, z_floor_m=z_floor, verbose=True)


def _probe_axes(session: OpenArmSession, controller: RealAtomicController) -> dict:
    """Step each MV_* token in isolation and measure the control-point displacement."""
    from core.action_units import MOVE_ATOMS

    controller.sync_from_robot()
    out: dict[str, list[float]] = {}
    for token in MOVE_ATOMS:
        before = controller.robot.get_ee_pose()[:3].copy()
        controller.step(token)
        after = controller.robot.get_ee_pose()[:3].copy()
        out[token] = [round(float(v), 5) for v in (after - before)]
    return out


def _dump_views(session: OpenArmSession, out_dir: Path) -> int:
    """Write the two prepared views so the framing/transform can be inspected."""
    from PIL import Image

    obs = session.get_observation()
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("agentview", "wrist"):
        path = out_dir / f"{key}.png"
        Image.fromarray(obs[key]).save(path)
        print(f"  {key:10s} {obs[key].shape} -> {path}")
    print(f"EE pose: {[round(float(v), 4) for v in obs['ee_pose']]}")
    print(f"gripper width: {obs['gripper_width'] * 1000:.1f} mm")
    return 0


def main() -> int:
    args = parse_args()
    load_secrets_env()
    robot_cfg = load_yaml(args.robot_config)
    cfg = build_config(args, robot_cfg)

    prompt_path = _prompt_path(Path(args.prompts_dir), args.version)
    prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    print(f"Prompt: {prompt_path} (version {args.version})")

    backend = openarm_backend(cfg)
    # Everything the run reads from the config is resolved before connect(): on the real
    # rig connect() enables both arms, so a config error must not first surface after it.
    log_root = Path(ROOT) / str(cfg.get("log_dir") or DEFAULT_LOG_DIR)
    max_steps = int(cfg.get("max_steps") or DEFAULT_MAX_STEPS)
    if backend == "real":
        if args.scene is not None or args.success_check is not None:
            print(
                "[openarm] --scene / --success-check are simulator-only: the real rig has "
                "no MJCF world and no independent scene predicate. Use "
                "configs/robot_openarm.yaml for the simulator.",
                file=sys.stderr,
            )
            return 2
        # The launcher's own session factory is used so this path and run_real.py cannot
        # drift on camera transforms, the driver block, or the pre-connect safety checks
        # (it refuses an unmeasured Z floor before any device is opened).
        from core.launch import make_openarm_real_session

        what = "hardware"
        try:
            session = make_openarm_real_session(cfg, args)
        except ValueError as exc:
            print(f"\n[openarm] {exc}", file=sys.stderr)
            return 2
    else:
        what = "the simulation"
        try:
            session = OpenArmSession(_session_config(cfg, args))
        except ValueError as exc:
            print(f"\n[openarm] {exc}", file=sys.stderr)
            return 2
    try:
        # A failed connect() has already stopped and disabled anything it enabled.
        session.connect()
    except Exception as exc:  # noqa: BLE001 - startup guidance beats a traceback
        print(f"\n[openarm] could not start {what}: {exc}", file=sys.stderr)
        return 2

    try:
        logger = EpisodeLogger(log_root, 0, variant="OA")
        if args.dump_views:
            return _dump_views(session, logger.run_dir)

        client = make_vlm_client(args, cfg)
        vlm_cfg = cfg["vlm"]
        print(
            f"VLM backend: {vlm_cfg.get('backend', '?')} "
            f"(model={vlm_cfg['model']}, base_url={vlm_cfg['base_url']})"
        )
        if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
            print(f"Waiting for VLM endpoint {vlm_cfg['base_url']}/models ...")
            client.health_check(
                wait_s=float(vlm_cfg.get("startup_wait_s", 0)),
                poll_s=float(vlm_cfg.get("startup_poll_s", 5)),
            )
            print("VLM endpoint is ready.")

        controller = _make_controller(cfg, args, session.robot)
        controller.sync_from_robot()

        if args.probe_axes:
            axes = _probe_axes(session, controller)
            logger.write_calibration({"move_axis_controlpoint_delta": axes})
            print("Axis probe (control-point delta per MV_* token, arm_origin metres):")
            for token, delta in axes.items():
                print(f"  {token:8s} -> {delta}")

        # Auto-release safety rule (plugins.auto_release): reopen a closed gripper whose
        # measured width shows it is holding nothing.
        plugins_cfg = PluginsConfig.from_config(cfg)
        auto_release = AutoReleasePlugin(
            enabled=plugins_cfg.enabled("auto_release", default=True),
            empty_width_m=float(cfg.get("empty_width_m", 0.006)),
        )
        if auto_release.enabled:
            print(
                "Auto-release: ON (reopen a closed gripper narrower than "
                f"{auto_release.empty_width_m:.4f} m)."
            )

        task = str(cfg.get("task") or "").strip() or _default_task(
            getattr(session.config, "scene", None)
        )
        print(f"Task: {task}")
        logger.write_metadata(
            {
                "backend": backend,
                "scene": getattr(session.config, "scene", None),
                "arm_side": session.config.arm_side,
                "success_check": getattr(session.config, "success_check", "none"),
                "task": task,
                "robot_config": cfg,
                "prompt_file": str(prompt_path.resolve()),
                "prompt_version": args.version,
                "control_loop": "openarm_mvtoken",
                "z_floor_m": controller.z_floor_m,
                "debug": args.debug,
            }
        )

        runner = MvTokenRunner(
            session=session,
            controller=controller,
            agent=MvTokenController(client=client, prompt_template=prompt_template),
            logger=logger,
            task=task,
            gripper_color=str(cfg.get("gripper_color", "black")),
            max_steps=max_steps,
            loop_period_s=float(cfg.get("loop_period_s", 0.0)),
            use_wrist_image=bool(cfg.get("use_wrist_image", True)),
            video_fps=float(cfg.get("v0", {}).get("video_fps", 2.0)),
            debug=bool(args.debug),
            auto_release=auto_release,
            prompt_log_every=int(args.prompt_log_every),
        )
        result = runner.run()

        env_success = session.success()
        print(f"Episode success: {env_success}")
        print(f"Steps: {result.steps}")
        print(f"End reason: {result.end_reason}")
        print(f"Run dir: {result.run_dir}")
        print(f"Video: {result.video_path}")

        # Environment ground truth is logged for analysis, never shown to the policy.
        try:
            logger.write_planner_diagnostics({"scene_truth": session.scene_truth()})
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail a rollout
            print(f"[openarm] could not write scene truth: {exc}")
        # An explicitly requested physics predicate must not be bypassed by DONE. The
        # hardware session has none, so completion there is the policy's own DONE.
        checked = (
            backend == "simulation"
            and str(getattr(session.config, "success_check", "none") or "none").strip().lower()
            != "none"
        )
        completed = env_success if checked else result.end_reason == "done"
        return 0 if completed else 2
    finally:
        session.close()


def _default_task(scene: str | None) -> str:
    """Task text when the config carries none (the sim configs are task-free).

    Keyed on the scene the session actually built, so ``--scene`` picks the matching task.
    """
    if scene == "insert_flower":
        return "Insert the roses into the vase"
    return "Pick up the red block and place it on the table"


if __name__ == "__main__":
    raise SystemExit(main())
