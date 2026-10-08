"""OpenArm wrist-height context; the safety floor is not the table surface."""
from __future__ import annotations

from plugins.proprioception.plugin import ProprioceptionPlugin


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
            "from the finger pads to the table."
        )
        if holding:
            block += " Holding an object: lift clear, then carry; descend only to place."
        if self._descend_stall_line(proprio):
            block += " Last descent was blocked or limited; re-align horizontally instead of repeating MV_DOWN."
        return block
