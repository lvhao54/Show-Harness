"""OpenArm prompts must match its side-mounted wrist camera and wrist-height frame."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.config import load_yaml, resolve_vlm_config
from core.launch import make_runner
from core.openarm.prompting import ARM_SIDE_MARKER, OpenArmProprioceptionPlugin, apply_arm_side
from core.prompting.prompt_loader import load_prompt_dir


def test_height_is_permitted_travel_not_fingertip_clearance():
    plugin = OpenArmProprioceptionPlugin(fine_step_m=0.02)
    text = plugin.render({"eef_pos": [0.2, -0.15, -0.220]}, -0.250)
    assert "descend 3.0 cm" in text
    assert "not distance" in text
    assert "above the table" not in text
    text = plugin.render({"eef_pos": [0.2, 0, -0.18]}, -0.250, holding=True)
    assert "Holding an object" in text
    assert "MV_DOWN first" not in text


def test_lateral_position_is_reported_against_the_robot_centre_line():
    plugin = OpenArmProprioceptionPlugin(fine_step_m=0.02)
    assert "29 cm RIGHT of the robot's centre line" in plugin.render(
        {"eef_pos": [0.2, -0.29, -0.2]}, -0.250
    )
    assert "12 cm LEFT" in plugin.render({"eef_pos": [0.2, 0.12, -0.2]}, -0.250)
    assert "on the robot's centre line" in plugin.render({"eef_pos": [0.2, 0.004, -0.2]}, -0.250)


def test_blocked_descent_advice_depends_on_holding():
    plugin = OpenArmProprioceptionPlugin(fine_step_m=0.02)
    stalled = {"eef_pos": [0.2, 0, -0.21], "descend_moved_m": 0.001, "descend_commanded_m": 0.02}
    holding = plugin.render(stalled, -0.250, holding=True)
    # A supported held object must not be dragged sideways (it tips over).
    assert "Never slide it sideways" in holding
    assert "re-align horizontally" not in holding
    assert "re-align horizontally" in plugin.render(stalled, -0.250, holding=False)


def test_arm_side_is_named_in_the_prompt():
    prompt = load_prompt_dir("prompts")["controller_openarm_prompt"]
    assert ARM_SIDE_MARKER in prompt
    named = apply_arm_side(prompt, "right")
    assert ARM_SIDE_MARKER not in named
    assert "You control the RIGHT arm only" in named
    assert "LEFT side of the image" in apply_arm_side(prompt, "left")
    with pytest.raises(ValueError):
        apply_arm_side(prompt, "both")


def test_zero_runner_uses_openarm_prompt_and_independent_score(tmp_path):
    cfg = load_yaml("configs/robot_openarm_zero.yaml")
    cfg["vlm"] = resolve_vlm_config(cfg, backend="gateway")
    session = SimpleNamespace(success=Mock(return_value=False))
    controller = SimpleNamespace(
        z_floor_m=cfg["z_floor_m"], step_m=0.02,
        variable_step_plugin=SimpleNamespace(enabled=False), rotation_plugin=None,
    )
    runner = make_runner(
        cfg=cfg, prompts_cfg=load_prompt_dir("prompts"), client=None,
        session=session, controller=controller,
        logger=SimpleNamespace(run_dir=tmp_path), debug=False,
    )
    agent = runner.controls.controller.agent
    assert "HORIZONTAL parallel gripper" in agent.prompt_template
    assert "You control the RIGHT arm only" in agent.prompt_template
    assert ARM_SIDE_MARKER not in agent.prompt_template
    assert "{gripper_proprio}" in agent.prompt_template
    assert isinstance(agent.proprio_plugin, OpenArmProprioceptionPlugin)
    assert runner.task_success_check == session.success
    assert not runner.action_chunk_plugin.enabled
    session.success.assert_not_called()
