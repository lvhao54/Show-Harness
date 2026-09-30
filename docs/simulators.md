# Simulators

Show-Harness integrates three benchmark simulators, plus two arm-specific simulation rigs:
the LeRobot SO-101 on an Isaac Lab scene and the bimanual OpenArm in MuJoCo
([SO-101](#so-101-isaac-lab-workshop), [OpenArm](#openarm-mujoco)). They serve two roles:

1. **Zero-shot evaluation** — run the deployment pipelines (the subgoal planner
   stack or the fine-tuned action model, see `docs/finetuned.md`) in sim
   with no robot attached.
2. **Sim-to-real training data** — generate demonstrations on the same action
   lattice the real robot deploys with (single-axis, ~2 cm per token), so sim
   rollouts mix directly with real teleop rollouts for fine-tuning.

Each integration keeps the deployment contracts: the same nine-token action
vocabulary, the same image transforms (`core/record/images.py`), and measured
per-config step calibration so one token means ~2 cm of physical travel.

Installing a simulator's dependencies: `bash scripts/setup.sh base --openarm`
adds the OpenArm stack to `.venv`. ManiSkill and RoboLab each want their own
interpreter (see below), because their pins conflict with the base environment.

## LIBERO

The standard manipulation benchmark, plus the two robustness suites built on it.
All three are separate checkouts that install as the same `libero` package, so
they share one runner and one camera contract and differ only in which
interpreter you point at them:

| checkout | what it adds | protocol |
| --- | --- | --- |
| LIBERO | the original suites | `--suite <name> --episodes N` |
| LIBERO-plus | seven perturbation dimensions over one large suite | `--libero-plus`, every task once |
| LIBERO-PRO | perturbation suites beside the base one (`*_object`, `*_swap`, `*_lan`, `*_task`, `*_temp`, and the `*_env_*` / `*_with_*` / numbered families) | `--suite <perturbed> --tasks <base task names>` |

- `scripts/run_libero_mvtoken.py` — entry point; config `configs/robot_libero.yaml`.
- `scripts/libero/eval_batch.sh <model> <n_episodes> [suite ...]` — batch eval,
  one process per suite. `PY=` selects the checkout, `RUN_ARGS=` the protocol;
  everything else is identical across the three.
- `bash scripts/setup.sh libero <checkout>` builds that checkout's venv
  (Python 3.10, torch 2.6.0+cu124, this repo's requirements installed alongside).

```bash
PY=../LIBERO-PRO/.venv/bin/python bash scripts/libero/eval_batch.sh <adapter> 10 \
    libero_spatial libero_spatial_object libero_spatial_swap libero_spatial_lan libero_spatial_task
PY=../LIBERO-plus/.venv/bin/python RUN_ARGS=--libero-plus \
    bash scripts/libero/eval_batch.sh <adapter> 1 libero_spatial
```

Add `RUN_ARGS="--ignore-done"` when comparing against published numbers: the
other VLAs have no stop action, so their episodes end only on the environment's
success check or the step limit.

Facts that do not fail loudly (the reason each is handled in code):

- **The task sentence must come from the bddl**, never `task.language`:
  LIBERO-plus appends the variant name to it, and LIBERO-PRO derives it from the
  file stem, so in exactly the two suites that perturb the instruction
  (`*_lan`, `*_task`) it reports the *un*perturbed sentence. `bddl_language()`
  in the runner reads the bddl, the way the data generator did.
- **agentview is rendered upside down.** Upstream un-flips it with
  `obs[cam][::-1]`, a vertical flip; this repo rotates 180°. Both stand the
  scene upright but differ by a horizontal mirror, and they disagree about which
  way is right — under `[::-1]` world +y moves left on screen, under the
  rotation it moves right, and `MV_RIGHT` has to look like right. Our frames are
  therefore mirrored with respect to upstream figures. Generation and inference
  both read `configs/robot_libero.yaml`, so changing it invalidates every frame
  the policy was trained on.
- **Fixtures are not in the state vector.** Cabinets and stoves have no joints,
  so LIBERO samples their placement into `model.body_pos` on every `reset()`, a
  few mm apart each time — measured at 2.7% of agentview pixels off by >20/255.
  A single-episode re-run is therefore *not* the same scene as that episode
  inside a full run. Store `fixture_poses()` with the rollout and
  `set_fixture_poses()` before re-rendering; `--replay` already does.

## ManiSkill

Fine-tuned policy only, in ManiSkill 3's translation-only `pd_ee_delta_pos`
control mode (rotation locked — the assumption the atomic-token policy makes).

- `scripts/run_maniskill_mvtoken.py` — entry point; config `configs/robot_maniskill.yaml`.
  One config covers both protocols: it defaults to the fixed layout preset, and
  `--traj-id random --layout wide` reproduces the randomized object layout the
  training data was generated with. `--env-id` switches scenes.
- `scripts/maniskill/eval_batch.sh <config> <model> <n_episodes> [max_steps] [tag]`
  — batch eval over consecutive seeds; prints the closed-loop success rate.

ManiSkill needs its own Python environment; the
runner's remaining dependencies (numpy, requests, pyyaml, PIL, imageio) are
standard. Scenes are declared as one `SceneSpec` row each in
`core/sim/maniskill_scenes.py`. The default `BlockPAP-v1` is a real2sim replica
of the real Franka rig (table, pedestal, block + coaster, calibrated front
camera) and needs an RLinf checkout (`RLINF_ROOT`); `BlockStack-v1` is the same
rig with a stacking task. Stock tasks (`PickCube-v1`, `StackCube-v1`) run too
but with a much larger domain gap.

```bash
python scripts/run_maniskill_mvtoken.py --version v3 --model <adapter> --max-steps 60 --probe-axes
```

`--probe-axes` records the measured per-token TCP delta into `calibration.json`.

Calibration facts (load-bearing; full derivations in the yaml comments):

- `step_m: 0.026` x `sim_steps_per_decision: 2` is the commanded setting that
  achieves ~20.2 mm per decision (PD lag makes achieved < commanded) — see
  `configs/robot_maniskill.yaml`.
- `wrist_flip: both` and `agentview_square_size: 256` are training contracts;
  read the yaml comments before touching either, and regenerate data after any
  camera change.

Training-data generation lives in `scripts/trajectory/real2sim/`: a
simulator-agnostic core (`atomic_tokenizer.py` — token vocabulary, closed-loop
2 cm execution, Manhattan/RDP/chase planners, teleop-format writer) plus one
backend per simulator (`backends/maniskill.py`, `backends/robolab.py`). It
produces rollouts in exactly the real-teleop format, so sim and real data mix
without special cases. See `scripts/trajectory/real2sim/README.md`.

## RoboLab (Isaac Lab)

NVIDIA's [RoboLab](https://github.com/NVLabs/RoboLab) benchmark: 120 authored
Isaac Sim manipulation tasks with automated success predicates and photoreal
rendering.

- `scripts/run_robolab_mvtoken.py` — entry point; config `configs/robot_robolab.yaml`.
- `scripts/robolab/eval_batch.sh <model> <n_episodes> [task ...]` — batch eval,
  one process per task (Isaac Sim's cold start dominates otherwise).

The RoboLab checkout location comes from the `ROBOLAB_ROOT` environment
variable (see `robolab_root()` in `core/sim/robolab_task.py`). Isaac Sim pins
Python 3.11 with its own large dependency set, so run this repo's scripts with
the RoboLab venv's interpreter — it also satisfies everything the runner needs.
First launch requires accepting Isaac Sim's EULA (`OMNI_KIT_ACCEPT_EULA=YES`),
and `libGLU.so.1` must be loadable or Isaac Sim segfaults during stage creation
with a misleading backtrace; `launch_isaac` in `core/sim/robolab_task.py`
checks for it up front and prints the fix.

```bash
python scripts/run_robolab_mvtoken.py --list-tasks                                     # no Isaac Sim needed
python scripts/run_robolab_mvtoken.py --task RubiksCubeTask --dump-views --probe-axes --no-rollout   # calibration only, no VLM
python scripts/run_robolab_mvtoken.py --version v3 --task RubiksCubeTask --episodes 5
```

`--task` takes the task class name; `--episodes N` reuses one Isaac Sim app and
env across episodes; `--gui` shows the viewport (default headless).

Embodiment. The sim Franka wears the real rig's short yellow fingertips, not
the stock black fingers — for a policy that reads pixels, the stock finger is a
distribution shift in the middle of every frame.
`assets/robolab_franka/panda_short_finger.usda` replaces both finger visuals
and collision meshes (rebuild with
`scripts/trajectory/real2sim/robolab/make_short_finger_asset.py`; set
`ROBOLAB_PANDA_USD` to compare against the stock robot). The camera geometry
also mirrors the real rig rather than RoboLab's DROID default: Panda hand, and
the wrist camera centered between the fingers looking down the grasp axis
(`core/sim/robolab_franka.py`).

Calibration facts (load-bearing; measured tables in the yaml comments):

- RoboLab's relative differential-IK achieves a constant ~28% of any commanded
  delta, so `step_m: 0.072` commanded yields ~20.1 mm measured per decision —
  see `configs/robot_robolab.yaml`, including why settle steps do not help.
- `wrist_rotation_degrees: 270` / `wrist_flip: none` are measured for the Panda
  hand (fingertips at the top of the frame, no mirror). The camera contract
  lives only in this yaml; the runner and the data generators both read it via
  `core.config.camera_contract()` — never restate it elsewhere.

Training-data generation uses the same real2sim core. The RoboLab oracle parses
each task's own subtask declaration, so any single-object pick-and-place task
among the 120 generates without code changes (others raise `UnsupportedTask`).
Generate per task with `real2sim/robolab/record_demos.py` then
`follow_tokenize.py` (commands in
[real2sim/README.md](../scripts/trajectory/real2sim/README.md)), looping over the
task names `--list-tasks` prints to build a set. Quality gate before training:
`python scripts/robolab/check_dataset.py <dir>` (nonzero exit = do not train on
it). Prefer rotation-insensitive objects — the vocabulary has no wrist-rotation
token, so elongated objects are ungraspable.

## SO-101 (Isaac Lab Workshop)

The [SO-101](https://github.com/TheRobotStudio/SO-ARM100) is LeRobot's low-cost 5-DoF arm.
It runs here on NVIDIA's
[Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
(tag `v1.0`, Isaac Sim 5.1 / Isaac Lab), on its "vials → rack" task
`Lerobot-So101-Teleop-Vials-To-Rack-DR`. This integration covers simulation only for now.
The real-arm backend (a LeRobot follower behind the same controller) is a follow-up, once it
has been validated on hardware.

| Path | What it is |
| --- | --- |
| `interpreters/so101_kinematics.py` | numpy FK/IK on the USD's own joint chain (`configs/so101_chain.json`) |
| `interpreters/so101_atomic_controller.py` | `So101ArmController` (motion core, I/O-agnostic) + `So101AtomicExec` (token executor) |
| `configs/primitives_so101.yaml` | MV_* vectors, rotation units, measured execution numbers |
| `configs/robot_so101_workshop.yaml` | task, scene variants, **the camera contract** |
| `scripts/trajectory/real2sim/backends/so101_workshop.py` | `AtomicSimEnv` backend: joints, cameras, contact sensor, success |
| `scripts/trajectory/real2sim/so101_workshop/` | `run_in_docker.sh`, `check_views.py`, `verify_motion.py`, USD extraction scripts, privileged `scene.py` |

Setup. The Workshop brings its own Isaac Sim / Isaac Lab stack in a Docker image, so nothing
from it is installed into this repo's environments. As with RoboLab, its dependencies stay
isolated from the base environment. Build the image once from a checkout at tag `v1.0`, then
run any script of this repo inside it through the wrapper. The wrapper mounts the repo
read-only, puts it on `PYTHONPATH`, and writes outputs to `rollouts/so101_workshop/`.

```bash
git clone -b v1.0 https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop ~/sim2real/Sim-to-Real-SO-101-Workshop
(cd ~/sim2real/Sim-to-Real-SO-101-Workshop && docker build -t teleop-docker -f docker/sim/Dockerfile .)
export WORKSHOP_ROOT=~/sim2real/Sim-to-Real-SO-101-Workshop
W=scripts/trajectory/real2sim/so101_workshop
bash $W/run_in_docker.sh $W/check_views.py   --out /workspace/out/check_views     # camera contract, exit 1 on mismatch
bash $W/run_in_docker.sh $W/verify_motion.py --out /workspace/out/verify_motion   # execution numbers, exit 1 on FAIL
```

Rotation units on 5 DoF. The shared vocabulary is unchanged, and the interpreter states what
the arm can realise. Pitch, Elbow and Wrist_Pitch are parallel, so the tool axis can only tilt
within the arm plane:

| Unit | On SO-101 |
| --- | --- |
| `RT_PITCH_FWD` / `RT_PITCH_BACK` | tool pitch in the arm plane, TCP held; about the arm-plane normal, which equals the shared world axis only while the arm points along +X |
| `RT_YAW_CW` / `RT_YAW_CCW` | hand turned through Wrist_Roll, TCP held; about the tool axis, which equals the shared world vertical only while the tool points straight down |
| `RT_ROLL_LEFT` / `RT_ROLL_RIGHT` | **not realisable**: always refused with reason `unsupported_on_5dof`; the arm does not move |

So neither realisable rotation is about a fixed world axis in every pose. For yaw the gap grows
with the tilt of the hand. In the default roll mode, at tool pitch *p* (90° = straight down) one
token gives 10° × sin *p* about the vertical, and the rest tips the jaw's closing axis out of
horizontal. At the 47° used for the measurements below, that is 7.3° of yaw per token and about
7° / 14° / 20° of tilt after one / two / three tokens (kinematics, reproduced on the unit-test
arm). That reaches the 13–22° of tilt at which the fixed finger dips and grasps fail (module
docstring of `interpreters/so101_atomic_controller.py`).

Refusals are whole-token and deterministic. Every token is checked before it runs (end-point
and mid-point IK feasibility, joint-limit margin, the whole hand's collision hull above the
table, workspace box). `So101AtomicExec.check()` answers without moving, so a generator never
records a frame for a token that would be refused. The backend does not claim
`supports_rotation`, because that contract means an arbitrary world-axis rotation. Execute
tokens with `So101AtomicExec`, not the generic `AtomicExec`.

Calibration facts (load-bearing; measured tables in the yaml comments):

- The SO-101's joint gains are low. The executor servos to a fixed target, capped at
  5 mm per 30 Hz control step, learns gravity sag at rest, and returns only once the arm is
  at rest. Measured by `verify_motion.py` (pitch 47°): one MV token moves 19.8–20.1 mm (96 tokens), at
  most 0.8 mm off-axis and 0.7 mm of overshoot. 80 random tokens and their reverse return
  within 0.1 mm. RT tokens turn 9.9–10.1° with at most 1.5 mm of TCP drift.
- Camera contract (`configs/robot_so101_workshop.yaml`): agentview `external_D455` rotated
  90°, wrist `ego` rotated 180°, both letterboxed to 256. Raw, neither view matches the
  shared prompt geometry. `check_views.py` measures it: agentview by TCP projection, wrist on
  the rendered frames (scene shift per executed token, and the fingers found as the pixels
  that ride with the camera). The rotated agentview is portrait, so its letterbox bars are
  left/right.
- Success is the Workshop's own `vial_placed` term, and it is lenient. It fires at the frame
  the fingers open if a just-held vial is within 45° of vertical and inside the rack's
  footprint, and it stays set. It does not check the hole, the vial staying upright, or which
  vial it was, so review rollouts by eye before trusting a success rate.

## OpenArm (MuJoCo)

The bimanual OpenArm in MuJoCo, driven through the official `openarm_control`
bimanual IK. Unlike the three benchmark simulators above, the real-arm stack runs here **unchanged**:
the session answers the same `get_observation()` contract the Franka session
does, and `core/openarm/openarm_interface.py` implements the four-method robot
duck type (`get_ee_pose` / `get_gripper_position` / `control_gripper` /
`update_desired_ee_pose`) that `RealAtomicController` already speaks. Setpoint
integration, the Z floor, empty-grasp detection and the `auto_release` plugin
are therefore the same code as on hardware.

- `scripts/run_openarm_mvtoken.py` — entry point; config `configs/robot_openarm.yaml`.
  `--scene insert_flower` swaps the default three-cube tabletop for the VLABench
  vase + three roses; `--arm left|right` picks the driven arm (the other holds
  its start pose).
- `--dump-views` and `--probe-axes` are the two calibration checks. Run both
  before reading any rollout.

The model, IK and cameras are **not vendored**. They come from an
`embodied_ai-llm` checkout, located via the config's `openarm_root`,
`OPENARM_ROOT`, or the sibling-checkout search in `core/openarm/sim.py`.
Only that project's simulation modules are imported; its FastAPI/pydantic/placo
runtime is never loaded, which is why the base environment's `numpy<2` works
here (measured; `openarm_control` and `mujoco` do not require numpy 2).

Three parts of the embodiment are easy to get wrong, and each is asserted by a
test in `tests/test_openarm_sim.py`:

- **The control point is the wrist, not the fingertips.** The
  `*_ee_control_point` site sits 0.1638 m behind the finger pads along the
  approach axis. A token sequence that looks right in the logs grasps air if
  that offset is ignored — the jaw midpoint, not the commanded pose, has to
  land on the object.
- **The wrist transform is per-arm.** The eye-in-hand cameras are mirrored
  (right: `rotation 270 / flip both`; left: `90 / both`). Measured at the home
  pose by projecting world +X / +Y, and the fingertips, into the wrist frame;
  the contract is "fingertips at the top, image-left agrees with the agentview".
- **The axis signs follow the camera, not the robot.** The head camera's
  image-left is world **+Y**, so `MV_LEFT` is `+Y` — the opposite sign to the
  real Franka primitives file. `--probe-axes` measures this and
  `configs/robot_openarm.yaml` documents it; inverting it produces the classic
  left/right oscillation.

Measured facts (load-bearing; from the model, not assumed):

- The sim tracks its setpoint closely — 20.1 mm achieved per 20 mm commanded on
  all three axes — so unlike RoboLab's relative IK, `step_m` IS the travelled
  distance and needs no lag compensation.
- The gripper free gap runs 0.0 m (closed) to 0.1425 m (open). `0` rad is
  CLOSED for both arms; the sign is mirrored.
- The head camera is 1280x720 (16:9) while the training rigs are 4:3, so the
  agentview is centre-cropped to 4:3 before letterboxing to 256.
- The upstream vase has **no collision hull below the rim** (a downward raycast
  grid through the mouth reaches the table). `success_check: flowers_in_vase`
  is therefore a geometric relation — tip at the mouth, matching VLABench's own
  relation-based scoring — and is not a test of physical containment.

Cross-domain caveat, stronger here than for the other simulators: the
released `_sim` policy was trained on Franka/Piper single-arm rollouts, so a
bimanual OpenArm with a different gripper, camera and workspace is well outside
its training distribution. Numbers from this harness are exploratory.
