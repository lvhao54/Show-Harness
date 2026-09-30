"""OpenArm embodiment: MuJoCo simulation session and the robot duck type.

Mirrors ``core/franka`` and ``core/piper``: :class:`~core.openarm.openarm_session.OpenArmSession`
owns the environment (here an in-process MuJoCo world) and
:class:`~core.openarm.openarm_interface.OpenArmSimRobot` implements the four-method
duck type :class:`~interpreters.real_atomic_controller.RealAtomicController` drives.

The MuJoCo model itself is NOT vendored here. ``embodied_ai-llm`` already has a verified
OpenArm simulation (official ``openarm_control`` IK, gravity feed-forward, three cameras),
so this package resolves that checkout and reuses it -- the same "point at an external
checkout" arrangement ``core.sim.robolab_task`` uses for ``ROBOLAB_ROOT``.
"""
