# Vendored OpenArm MuJoCo backend

The Python modules in this directory are a trimmed, self-contained copy of the OpenArm
simulation from the `embodied_ai-llm` project (`openarm2-sim-bridge`), so Show-Harness can
run the OpenArm simulator without checking that repository out. Files were copied from
`src/robot_runtime/` at commit `6bb76dabde2a638834f43882679df81145db07c0`:

| this directory | upstream source |
| --- | --- |
| `state.py` | `control/backend.py` (only `BackendState`; the hardware-driver protocols were dropped) |
| `cameras.py` | `cameras/mujoco.py` |
| `mujoco_backend.py` | `robots/mujoco.py` |
| `kinematics.py` | `robots/openarm/kinematics.py` |
| `simulation.py` | `robots/openarm/simulation.py` |
| `scenes.py` | new loader for the bundled scene XMLs (replaces `assets/__init__.py`) |
| `assets/openarm2.xml`, `assets/openarm2_insert_flower.xml` | `assets/` (scene furniture only) |

Copyright 2026 Enactic, Inc. — licensed under the Apache License, Version 2.0; you may
obtain a copy at <http://www.apache.org/licenses/LICENSE-2.0>. The changes are limited to
import paths, the scene loader, and dropping the hardware-driver half of the runtime. The
robot model, pedestal, meshes and the inverse-kinematics solver are **not** vendored: they
come from the `openarm-mujoco` and `openarm-control` wheels
(`requirements/requirements-openarm.txt`).

`assets/vlabench/` carries the scene assembly XMLs plus the VLABench provenance files. Its
meshes/textures (~25 MB) are not bundled; fetch them with
`bash scripts/model/download_vlabench_assets.sh`. See `assets/vlabench/README.md` and
`assets/vlabench/SOURCE.json` for the pinned upstream revisions and what is known about
their licensing.
