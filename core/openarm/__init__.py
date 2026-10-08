"""OpenArm embodiment: MuJoCo simulation session and the robot duck type.

Mirrors ``core/franka`` and ``core/piper``: :class:`~core.openarm.openarm_session.OpenArmSession`
owns the environment (here an in-process MuJoCo world) and
:class:`~core.openarm.openarm_interface.OpenArmSimRobot` implements the four-method
duck type :class:`~interpreters.real_atomic_controller.RealAtomicController` drives.

Two entry points share this world: the fine-tuned MVTOKEN loop
(``scripts/run_openarm_mvtoken.py``) and the zero-shot plugin harness
(``scripts/run_real.py`` with ``configs/robot_openarm_zero.yaml``).

The simulator is self-contained. The MuJoCo driver, cameras and scene files live in
:mod:`core.openarm.backend` (vendored from ``embodied_ai-llm``, Apache-2.0 -- see its
``NOTICE.md``); the robot model, meshes and the official ``openarm_control`` IK come from
the ``openarm-mujoco`` / ``openarm-control`` wheels. No external checkout is consulted.
"""
