#!/usr/bin/env python3
"""Capture the Z safety floor (minimum control-point height) for ONE OpenArm arm.

The floor is the ``arm_origin`` height of the driven arm's control point when its gripper
just rests on YOUR tabletop. The controller blocks every commanded descent below it, so an
autonomous run refuses to start until this is a measured number rather than a guess.

    # rest the RIGHT gripper on the tabletop first, then read + persist the height
    python scripts/openarm/capture_z_floor.py --arm right --write

    # print only, do not touch the config
    python scripts/openarm/capture_z_floor.py --arm right

This starts ONLY the CAN workers for the requested arm (no cameras), enables it, reads the
control-point height and disables it again. It moves nothing: the vendor's startup trajectory
is never run, so hand-guide or jog the gripper into contact before calling it.

The value is written into the site layer (configs/site/openarm.yaml), which is per-cell and
gitignored, under ``z_floor_m``. ``--site-config`` points at a different file.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml

DEFAULT_ROBOT_CONFIG = ROOT / "configs" / "robot_openarm_real.yaml"
DEFAULT_SITE_CONFIG = ROOT / "configs" / "site" / "openarm.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one OpenArm arm's Z safety floor (control-point height)."
    )
    parser.add_argument(
        "--robot-config",
        default=str(DEFAULT_ROBOT_CONFIG),
        help="Robot config whose driver block this run uses (default: the real rig).",
    )
    parser.add_argument(
        "--site-config",
        default=str(DEFAULT_SITE_CONFIG),
        help="Site layer to read and (with --write) update the floor in.",
    )
    parser.add_argument("--arm", required=True, choices=("left", "right"))
    parser.add_argument(
        "--write", action="store_true", help="Write z_floor_m into the site config."
    )
    return parser.parse_args()


def _driver_config(args: argparse.Namespace) -> dict:
    """The resolved driver block, with the site layer's floor key removed."""
    cfg = load_yaml(args.robot_config)
    driver = dict(cfg.get("driver", {}) or {})
    driver.pop("cameras", None)
    if not driver.get("hardware_watchdog_confirmed"):
        raise SystemExit(
            "driver.hardware_watchdog_confirmed is false. Capturing the floor enables the "
            "motors (they hold their measured position; the start trajectory is off), so "
            "verify the motor watchdog and the physical stop on this robot, set "
            f"`hardware_watchdog_confirmed: true` in {args.site_config}, and re-run."
        )
    # Capturing a floor must not run the vendor startup trajectory underneath the operator.
    driver["startup_trajectory"] = False
    return driver


def _write_floor(site_path: Path, value: float) -> bool:
    """Set the top-level ``z_floor_m`` in the site YAML, preserving everything else."""
    text = site_path.read_text()
    lines = text.splitlines(keepends=True)
    import re

    pattern = re.compile(r"^(z_floor_m\s*:).*$", re.MULTILINE)
    replacement = f"z_floor_m: {value:.5f}   # measured; blocks all descent below it\n"
    if pattern.search(text):
        updated = pattern.sub(replacement.rstrip("\n"), text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        updated = text + "\n# Z safety floor (arm_origin, metres); measured on the cell.\n" + replacement
    site_path.write_text(updated)
    return updated != text


def main() -> int:
    args = parse_args()
    driver = _driver_config(args)
    from core.openarm.hardware import make_pedestal_backend
    from core.openarm.hardware.robot import OpenArmRealRobot

    backend = make_pedestal_backend(driver=driver, cameras={})
    robot = None
    exit_code = 0
    try:
        backend.start()
        robot = OpenArmRealRobot(backend, side=args.arm, verbose=False)
        pose = np.asarray(robot.get_ee_pose(), dtype=float)
        z = float(pose[2])
        if not math.isfinite(z):
            raise RuntimeError(f"non-finite height read back: {z}")
        print(
            f"[capture-z-floor] {args.arm} arm control point: "
            f"xyz={np.round(pose[:3], 5).tolist()}"
        )
        print(f"[capture-z-floor] measured floor z_floor_m = {z:.5f} m")
        print(
            "[capture-z-floor] downward motion below this height will be blocked. "
            f"Make sure the {args.arm} gripper was resting on the tabletop."
        )
        if args.write:
            site = Path(args.site_config)
            if _write_floor(site, z):
                print(f"[capture-z-floor] wrote z_floor_m: {z:.5f} into {site}")
            else:
                print(f"[capture-z-floor] no change written to {site}")
                exit_code = 1
    except Exception as exc:  # noqa: BLE001 - surface clearly
        print(f"[capture-z-floor] ERROR: {exc}")
        exit_code = 1
    finally:
        try:
            if robot is not None:
                robot.close()
            else:
                backend.close()
        except Exception:  # noqa: BLE001
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
