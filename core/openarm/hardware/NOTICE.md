# Vendored OpenArm hardware backend

`dispatch.py`, `preload.py` and `cameras.py` are a trimmed, self-contained copy of the
OpenArm2 Pedestal hardware layer from the `embodied_ai-llm` project
(`openarm2-sim-bridge`), so Show-Harness can drive the real CAN-connected arms without
checking that repository out. Files were copied from `src/robot_runtime/` at commit
`4ac2625f0f7d0ab9f8ca1ba9b0dc4e55b05f7e02`:

| this directory | upstream source |
| --- | --- |
| `dispatch.py` | `robots/openarm/dispatch.py` |
| `preload.py` | `robots/openarm/preload.py` |
| `cameras.py` | `cameras/openarm.py` (the `_is_video_capture` V4L2 probe is inlined) |
| `backend.py` | `robots/openarm/hardware.py` (the runtime's `RobotError`-raising `bounded` was replaced by a local `_bounded`) |

Copyright 2026 Enactic, Inc. -- licensed under the Apache License, Version 2.0; you may
obtain a copy at <http://www.apache.org/licenses/LICENSE-2.0>. Local changes include
import paths, dropping the FastAPI/pydantic runtime layer, and the `make_pedestal_backend`
constructor used by the Show-Harness session. The vendor motor driver, CAN transport,
kinematics solver and the robot model/meshes are **not** vendored: they come from the
`openarm-driver`, `openarm-can`, `openarm-control` and `openarm-mujoco` wheels
(`requirements/requirements-openarm-real.txt`).

`robot.py` and `core/openarm/openarm_real_session.py` are Show-Harness originals: they
adapt the vendored backend to the four-method robot duck type and the shared session
contract.
