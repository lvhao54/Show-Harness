# Simulators

Show-Harness integrates three benchmark simulators, plus two arm-specific simulation rigs:
the LeRobot SO-101 on an Isaac Lab scene and the bimanual OpenArm in MuJoCo
([SO-101](#so-101-isaac-lab-workshop), [OpenArm](#openarm-mujoco)). The OpenArm Pedestal
rig also runs the same contracts on real CAN hardware
([OpenArm2 Pedestal (real hardware)](#openarm2-pedestal-real-hardware)). They serve two
roles:

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
adds the OpenArm MuJoCo stack to `.venv`, and `--openarm-real` adds the OpenArm CAN
hardware stack. ManiSkill and RoboLab each want their own
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

Two entry points share one world:

| mode | entry point | config |
| --- | --- | --- |
| zero-shot (frontier VLM + plugin stack) | `scripts/run_real.py` | `configs/robot_openarm_zero.yaml` |
| fine-tuned (one action token per step) | `scripts/run_openarm_mvtoken.py` | `configs/robot_openarm.yaml` |

Both drive the same session and the same controller factory
(`core/openarm/controller.py`); only the decision loop differs. A test asserts
the two configs agree on the world and safety values. Camera preprocessing is
policy-specific: zero-shot uses upright native views; the fine-tuned path retains
its legacy wrist transform.

`--scene insert_flower` swaps the default three-cube tabletop for the VLABench
vase + three roses; also select `--success-check flowers_in_vase` for that task.
`--arm left|right` picks the driven arm (the other holds its start pose). The
two wrist mounts are mirror images, so each arm's wrist transform is its own
`wrist_by_arm.<side>` entry in the config, and an arm without one is refused
rather than given the other arm's (rotated or mirrored) view. On the zero-shot
entry point `--dump-views` and `--probe-axes` are not defined, so calibrate once
with the mvtoken entry point before reading any rollout.

The simulator is **self-contained**:

- the physics/IK/camera glue is vendored in `core/openarm/backend/`
  (Apache-2.0, provenance in `backend/NOTICE.md`);
- the robot model, pedestal, meshes and the official `openarm_control` IK come
  from the `openarm-mujoco` / `openarm-control` wheels;
- the default `openarm2.xml` scene ships with the repo.

No `embodied_ai-llm` checkout is involved. The wheels are installed by
`bash scripts/setup.sh base --openarm`; the upstream project's
FastAPI/pydantic/placo stack and its `numpy>=2` pin never enter this venv
(measured; `openarm_control` and `mujoco` run on `numpy<2`).

The one exception is the `insert_flower` scene: its VLABench meshes and textures
(~25 MB) are not redistributable with this repository, so they are fetched on
demand at a pinned revision with sha256 verification:

```bash
bash scripts/model/download_vlabench_assets.sh
```

Three parts of the embodiment are easy to get wrong, and each is asserted by a
test in `tests/test_openarm_sim.py`:

- **The control point is the wrist, not the fingertips.** The
  `*_ee_control_point` site sits 0.1638 m behind the finger pads along the
  approach axis. A token sequence that looks right in the logs grasps air if
  that offset is ignored — the jaw midpoint, not the commanded pose, has to
  land on the object.
- **The wrist view is a side view.** Native wrist image height mostly represents
  vertical height, not table depth. Zero-shot uses `controller_openarm.txt`, native
  upright views and the configured FWD/BACK exchange at execution. The legacy
  fine-tuned transforms remain right `270/both`, left `90/both`; they are a
  deployment convention, not evidence that the adapter was trained on OpenArm.
- **The axis signs follow the camera, not the robot.** The head camera's
  image-left is world **+Y**, so `MV_LEFT` is `+Y` — the opposite sign to the
  real Franka primitives file. `--probe-axes` measures this and
  `configs/primitives_openarm.yaml` owns it (the fine-tuned config repeats the
  same vectors under `move_vectors`, and a test asserts they match); inverting
  it produces the classic left/right oscillation.

Measured facts (load-bearing; from the model, not assumed):

- The sim tracks its setpoint closely — 20.1 mm achieved per 20 mm commanded on
  all three axes — so unlike RoboLab's relative IK, `step_m` IS the travelled
  distance and needs no lag compensation.
- The gripper free gap runs 0.0 m (closed) to 0.1425 m (open). `0` rad is
  CLOSED for both arms; the sign is mirrored. A held 45 mm cube leaves a 45 mm
  gap, so the OPEN/CLOSED threshold is 60 mm, not the old 20 mm.
- The red object is an upright 2 x 1 x 6 cm (x, y, z) block; green and blue are
  45 mm cubes. The jaws close along world Y, across the 1 cm face, so a held
  red block reads ~9 mm, 3 mm above the 6 mm empty-grasp threshold. The grasp
  window along the approach axis is about ±10 mm of pad offset. On the 20 mm
  token grid that is one depth, one `MV_FWD` from home; the neighbouring depths
  close empty. At -15 mm the pads catch an edge and report a hold without
  lifting. Lateral error up to ~24 mm is pushed into line by the closing jaws.
  At 26 mm the shoving finger tilts the block ~14° before it is gripped.
- The horizontal tool's calibrated wrist floor is **-0.250 m** in `arm_origin`.
  The former -0.225 m floor left only about 1 mm of pad/cube overlap and empty
  grasps on the 45 mm cubes (the 6 cm block is reachable from either floor).
  Proprioception reports travel remaining to the wrist floor, not an invented
  fingertip-to-table distance.
- IK includes a robot-only link4/pedestal collision constraint and a stronger
  orientation task. Without both, transport to the central cube collides with
  the pedestal or tilts the tool. The limit extension uses the tested
  `openarm-control==0.4.0` solver seam.
- `setpoint_leash_m: 0.004` (both sim configs) is the controller's anti-windup.
  A move stops once the measured wrist trails its setpoint by more than 4 mm,
  which happens on contact or at the reach limit (free motion lags ≤ 1.1 mm).
  The setpoint then re-anchors on the measured pose, and gripper actions
  re-anchor first. Without it, each blocked `MV_DOWN` wound the setpoint deeper.
  A held red block slid ~33 mm up through the pads, and on `RELEASE` the
  rotating fingers struck the green cube's top edge and toppled the stack.
  Runaway lateral commands also wound the target ~1.5 m past reach until the
  IK flipped and the wrist dropped below the floor. The real-rig config does
  not set it: the lag is read right after each command, which assumes a
  blocking `update_desired_ee_pose` like the simulator's.
- Zero-shot re-observes after every action; three coarse open-loop moves could
  overshoot the small cubes. A stage timeout fails the episode instead of
  advancing from an unfinished GRASP to LIFT.
- `success_check: red_on_green` independently verifies the default stack task
  after final DONE: support contacts, low velocities and no arm contact. The
  score and object poses are never policy inputs. Change the predicate with
  the task; `none` keeps VLM-only completion.
- The head camera is 1280x720 (16:9) while the training rigs are 4:3, so the
  agentview is centre-cropped to 4:3 before letterboxing to 256.
- The upstream vase has **no collision hull below the rim** (a downward raycast
  grid through the mouth reaches the table). `success_check: flowers_in_vase`
  is therefore a geometric relation — tip at the mouth, matching VLABench's own
  relation-based scoring — and is not a test of physical containment.

Validate mechanics without a model before spending on a rollout:

```bash
.venv/bin/python -m pytest tests/test_openarm_grasp.py tests/test_openarm_success.py -q
.venv/bin/python scripts/run_real.py --robot-config configs/robot_openarm_zero.yaml \
  --vlm-backend gateway --no-show
```

The physics regression uses fixed tokens (not a VLM): it reproduces the old empty
close on a 45 mm cube, checks the corrected grasp/lift of the red block and a
cube, then transports, releases and retreats from a stable red-on-green stack
without elbow/pedestal contact. It also presses the held block three times into
the green cube before releasing: the stack stands with the leash and topples
without it. It establishes
that the control path can solve the task; it is not an autonomous policy score.

### Zero-shot validation note (2026-10-10)

Five fresh zero-shot runs of `red_on_green` were executed on this branch against the
gateway backend (`gpt-6.1-sol`, medium reasoning), with the config otherwise identical to
the earlier reference run:

| run | steps | end reason | physically stacked |
| --- | --- | --- | --- |
| 1 | 65 | max_steps_exceeded | no |
| 2 | 92 | subgoal_step_cap_exceeded | no (released 4.8 cm short of the green cube) |
| 3 | 30 | gateway HTTP 499 (aborted) | n/a |
| 4 | 52 | subgoal_step_cap_exceeded | no (cube nudged off the table) |
| 5 | 38 | plan_complete | **yes** (red-green dz = 45.0 mm, dxy = 5.1 mm) |

So 1 of 5 attempts completed. Two of the four failures were infrastructure- or
budget-shaped rather than policy-shaped: run 3 died on a reverse-proxy 499 mid-response
(now classified retryable), and runs 1/2/4 were moving toward the target when either the
episode or the per-stage step cap expired. Run 4 was a genuine policy failure: the model
kept emitting `MV_RIGHT` in the MOVE stage and pushed the cube off the tabletop.

The successful run is logged with independent ground truth in
`planner_diagnostics.json` (`success_check.holds: true` plus final object positions), which
is what makes it a scored result rather than a self-report. That file is now written on the
zero-shot path too, not only by the fine-tuned entry point.

Both fixes this validation surfaced are in the tree: the gateway 499 retry class
(`core/vlm/vlm_client.py`) and the zero-shot scene-truth record (`scripts/run_real.py`).

### Zero-shot red-block stacking (2026-10-13)

The red cube became an upright block, first 1 x 2 x 6 cm (x, y, z), now 2 x 1 x 6 cm. Same
gateway backend and settings as above (`gpt-6.1-sol`, medium reasoning, 65 steps); success
is the independent `red_on_green` check. Rounds 1-4 used the 1 x 2 x 6 block:

| round | change | stacked |
| --- | --- | --- |
| 1 | block only | 0/3 |
| 2 | + setpoint leash, driven-arm name, lateral proprioception, placement rules | 1/3 |
| 3 | + stage-aware placement, wrist-view centering before GRASP | 2/5 |
| 4 | centering relaxed to the middle third of the wrist image | **5/5** (28-34 steps) |

What each failure was:

- **Round 1, wrong arm.** Once lifted, the block is hidden behind the wrist in the 256 px
  AgentView, and the prompt never said which arm was driven. The model judged the
  block "left of green" and drove right until it left the table. The prompt now names
  the driven arm and its image side. Proprioception reports the finger gap's offset
  from the robot centre line, which is AgentView's vertical centre line.
- **Round 1, release topple.** Pressing after contact wound up the setpoint. The block
  slid through the pads, and the opening fingers struck green (see
  `setpoint_leash_m`). The blocked-descent hint used to say "re-align horizontally" even
  while holding. Now it says stop, finish the stage, or lift before re-aligning.
- **Round 2, re-grasp.** The model released inside PLACE although the plan has a separate
  RELEASE stage, then re-grasped the block. The supported-object rule now finishes the
  current stage: DONE in PLACE, RELEASE in RELEASE.
- **Round 2, slip in transport.** A grasp 26 mm off-centre lets one finger shove the block
  first, so it is gripped tilted and works loose. Grasps within about 24 mm hold.
- **Round 3, centering dither.** The block sits between two wrist positions on the 2 cm
  grid, and "exactly centred" made the model alternate MV_LEFT/MV_RIGHT until the stage
  cap. The middle third of the wrist image is about +-16 mm.

All round-4 runs start from the same keyframe, and the model's choices barely vary
(the grasp phase is the same nine tokens each time). Read 5/5 as the failure modes being
fixed for this layout, not as a success rate over varied object positions.

The 2 x 1 x 6 block (jaws now across its 1 cm face) then stacked **5/5** (26-32 steps) with
no further change. One pass was lucky: the model ended LIFT after a single 4 cm step,
leaving the block's bottom ~6 mm below green's top. Carried sideways, the block pushed the
green cube 6 cm before the model placed it on top. The other four lifted 8-12 cm and carried
clear.

The released `_sim` adapter was trained on other simulators, not OpenArm. The
local validated experiments that emitted only MV_DOWN are failed cross-domain
policy transfer; changing the floor cannot supply missing visual/motor training.
Use the zero-shot path to establish an autonomous baseline, or collect OpenArm
trajectories and train a corresponding adapter before treating fine-tuned results
as supported performance. The zero-shot prompt is specific to the fixed horizontal
tool and native camera mounting; revalidate it if either changes.

## OpenArm2 Pedestal (real hardware)

The same rig has a second backend: `configs/robot_openarm_real.yaml` drives the physical
CAN-connected arms through the **identical** session/controller/plugin stack. The two
backends differ only in what answers the robot and session contracts, so a token that
works in the simulator reaches the same code path on hardware. A `backend:` key in the
robot config selects it; the simulator configs do not set it (default `simulation`).

| what | simulator | hardware |
| --- | --- | --- |
| config | `configs/robot_openarm.yaml` / `robot_openarm_zero.yaml` | `configs/robot_openarm_real.yaml` |
| session | `core/openarm/openarm_session.py` (MuJoCo) | `core/openarm/openarm_real_session.py` (CAN + USB cameras) |
| robot duck type | `OpenArmSimRobot` | `core/openarm/hardware/robot.py` |
| install | `bash scripts/setup.sh base --openarm` | `bash scripts/setup.sh base --openarm-real` |
| site layer | none | `configs/site/openarm.yaml` (required) |

Setup, in order:

```bash
# 1. hardware stack + site identity
bash scripts/setup.sh base --openarm-real
cp configs/site/openarm.yaml.example configs/site/openarm.yaml

# 2. edit site/openarm.yaml: CAN interfaces, camera USB identities
# 3. rest one gripper on the table, measure the Z floor (writes the site layer)
python scripts/openarm/capture_z_floor.py --arm right --write

# 4. read-only preflight: driver wheels, CAN links, cameras, key, floor
python scripts/check_setup.py --robot-config configs/robot_openarm_real.yaml

# 5. run
python scripts/run_real.py --robot-config configs/robot_openarm_real.yaml --no-show
```

What is load-bearing on hardware:

- **`driver.hardware_watchdog_confirmed` gates motion.** The backend refuses to enable
  motors until an operator has verified the motor/controller watchdog and the physical
  stop on this cell; Python process liveness is not a watchdog.
- **The Z floor is a measured per-cell value**, not the simulator's `-0.250` constant.
  The launcher refuses to capture at the start pose (a raised start forbids all descent, a
  low one protects nothing), so `z_floor_m` must be a number before an autonomous run.
- **Each arm owns a CAN interface** (`can0` / `can1`), never shared. The vendor's
  thread-owned dispatcher enables by measured position (no homing), sends at `send_hz`
  through a 5 Hz low-pass, and runs its own stop trajectory + disable on shutdown.
- **A support preload is latched once**, from a stable window of commanded-vs-measured arm
  joints (jaw excluded). Until it is stable the session waits at startup rather than
  dispatching a command that would unload the position loop.
- **The jaw has no width encoder.** The measured finger joint angle is converted through
  the same geometry the MJCF model uses (a measured cubic fit, 0.011 mm max error,
  0.0-0.143 m free gap), because the controller's empty-grasp and close/open thresholds
  are written in metres. The open sign is mirrored per arm.
- **A heartbeat keeps the arms enabled while the VLM thinks.** The dispatcher stops an arm
  if it stops hearing from its owner within `control_timeout_seconds`; the session feeds
  it from a small daemon thread, standing in for the reference runtime's control thread. A
  process wedged badly enough to starve that thread still triggers the stop path.
- **Cameras are addressed by USB identity, never by video index.** The head camera is the
  AgentView and the driven arm's wrist camera the wrist view; a missing or stale frame
  fails the observation instead of being served as new.
- **No independent success predicate.** The simulator's scene checks read MuJoCo state; on
  hardware the policy's own `DONE` is the terminal signal, and `--scene` /
  `--success-check` are rejected for the real config.

The vendored hardware layer lives in `core/openarm/hardware/` (CAN dispatch, preload,
cameras) and is exercised by `tests/test_openarm_real.py` plus the ported dispatcher,
preload and camera suites (`tests/test_openarm_hw_*.py`); those tests inject fake drivers
and cameras and never open a real device. The reference implementation and its own
end-to-end validation are described in `embodied_ai-llm/docs/openarm-real.md`; the tests
here prove the contract and the wiring, not that a given cell is calibrated. Complete the
physical stop, watchdog, workspace and floor checks on your rig before an autonomous run.
