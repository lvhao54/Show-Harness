"""OpenArm controller context: wrist height, lateral position, and the driven arm.

The safety floor is not the table surface, so height is reported as permitted travel.
"""
from __future__ import annotations

from plugins.proprioception.plugin import ProprioceptionPlugin

#: Marker in prompts/controller_openarm.txt replaced by the driven arm's side.
ARM_SIDE_MARKER = "<ARM_SIDE>"


def apply_arm_side(prompt: str, arm_side: str) -> str:
    """Name the driven arm in the controller prompt (``right`` -> ``RIGHT``).

    The head camera looks forward from the robot's midline, so the right arm is on the
    image right; without the name the model has no anchor for which gripper is its own.
    """
    side = str(arm_side).strip().lower()
    if side not in ("left", "right"):
        raise ValueError(f"arm_side must be 'left' or 'right', got {arm_side!r}")
    return prompt.replace(ARM_SIDE_MARKER, side.upper())


class OpenArmProprioceptionPlugin(ProprioceptionPlugin):
    """Report measured travel remaining, without inventing fingertip clearance."""

    def render(self, proprio, table_height_m, holding=False):
        if not self.enabled or not proprio or table_height_m is None:
            return ""
        eef = proprio.get("eef_pos") or []
        if len(eef) < 3:
            return ""
        gap_cm = max(0.0, (float(eef[2]) - float(table_height_m)) * 100)
        block = (
            f"The wrist can descend {gap_cm:.1f} cm before its calibrated grasp-height limit; "
            f"{self._step_sizes_text()}. This is remaining permitted travel, not distance "
            "from the finger pads to the table. "
            f"{self._lateral_text(float(eef[1]))}"
        )
        if holding:
            block += " Holding an object: lift clear, then carry; descend only to place."
        if self._descend_stall_line(proprio):
            if holding:
                block += (
                    " Last descent was blocked: the held object is resting on something. "
                    "If it is centered on the destination, finish this stage (DONE in a "
                    "PLACE stage, RELEASE in a RELEASE stage); otherwise MV_UP before "
                    "re-aligning. Never slide it sideways or press further."
                )
            else:
                block += " Last descent was blocked or limited; re-align horizontally instead of repeating MV_DOWN."
        return block

    @staticmethod
    def _lateral_text(y_m: float) -> str:
        """Where the finger gap is across the robot (arm_origin +Y is the robot's left).

        The fingers point along +X, so the gap shares the wrist's Y. The head camera sits
        on the robot's midline, so that line is AgentView's vertical centre line.
        """
        cm = abs(y_m) * 100
        if cm < 1.0:
            where = "on the robot's centre line"
        else:
            where = f"{cm:.0f} cm {'LEFT' if y_m > 0 else 'RIGHT'} of the robot's centre line"
        return (
            f"Your finger gap is {where} (AgentView's vertical centre line is the robot's "
            "centre line)."
        )
