# armThrow

PyBullet + PPO prototype for the ME5406 Project 2 throwing task.

The current codebase trains a 3-DOF arm to throw a ball at a 3D target. The target may be on the ground or in the air. The main entry point is [`train.py`](train.py).

## Current Status

- Main CLI entry point lives in [`train.py`](train.py).
- Core implementation is now split across dedicated modules:
  - [`env.py`](env.py)
  - [`callbacks.py`](callbacks.py)
  - [`config.py`](config.py)
- The current recommended setup uses:
  - `reward_mode: distance_progress`
  - `observation_mode: arm_target_release`
  - split control parameters: `accel_scale`, `motor_force_limit`, `joint_velocity_limit`
  - curriculum training: `fixed_center -> narrow_random -> random_full`

## Repository Layout

- [`train.py`](train.py): CLI entry point and PPO setup
- [`env.py`](env.py): environment dynamics, reward, observation, termination logic
- [`callbacks.py`](callbacks.py): GIF recording, W&B logging, eval callbacks
- [`config.py`](config.py): config loading, normalization, run directory helpers
- [`arm.urdf`](arm.urdf): 3-DOF arm model
- [`visualize_arm.py`](visualize_arm.py): manual visualization/debug script
- [`configs/base.yaml`](configs/base.yaml): default interactive training config
- [`configs/`](configs): frozen experiment configs and curriculum stages
- `runs/`: local training outputs, saved models, W&B local files; new additions should stay local

## Environment Setup

### Conda / Micromamba

From the repository root:

```bash
conda env create -f environment.yml
conda activate armThrow
```

Or with micromamba:

```bash
micromamba create -f environment.yml
micromamba activate armThrow
```

### W&B

If you want experiment tracking:

```bash
wandb login
```

If you do not want W&B, set `logging.use_wandb: false` in the config you run.

## Quick Start

### 1. Default run

This is the simplest way to start training:

```bash
python train.py --config configs/base.yaml
```

Notes:
- `base.yaml` keeps `render: true` by default, so this opens the PyBullet GUI.
- For long runs, use:

```bash
python train.py --config configs/base.yaml --no-render
```

### 2. Load a pretrained stage

The training script supports warm-starting from an existing checkpoint:

```bash
python train.py --config <config.yaml> --load-model <path/to/model.zip>
```

This is how the curriculum stages are chained together.

You can also override the training seed without editing the YAML:

```bash
python train.py --config <config.yaml> --seed 123
```

The resolved config actually used for a run is saved to `runs/<run>/config.yaml`.

## Current Default Training Recipe

The current default recipe is the most robust curriculum chain tested so far:

1. [`configs/fixed_center_stage1_300k.yaml`](configs/fixed_center_stage1_300k.yaml)
2. [`configs/narrow_random_stage2_300k.yaml`](configs/narrow_random_stage2_300k.yaml)
3. [`configs/random_full_stage3_300k.yaml`](configs/random_full_stage3_300k.yaml)

Run them in order:

```bash
python train.py --config configs/fixed_center_stage1_300k.yaml
python train.py --config configs/narrow_random_stage2_300k.yaml --load-model runs/<stage1_run>/model.zip
python train.py --config configs/random_full_stage3_300k.yaml --load-model runs/<stage2_run>/model.zip
```

Interpretation:
- `stage1` learns basic arm swing + release behavior on a fixed target
- `stage2` teaches the policy to use `target_pos` under a narrow random target range
- `stage3` transfers the policy to the full random target range

Only the final model from `stage3` should be treated as the final model.

Why this is the default:
- it is the strongest **multi-seed validated** schedule tested so far
- it is more robust than the more aggressive `150k / 250k / 400k` branch on seeds `7`, `42`, and `123`

## Current Best Single-Seed Candidate

The strongest single-seed fixed-budget chain tested so far is:

1. [`configs/fixed_center_stage1_150k.yaml`](configs/fixed_center_stage1_150k.yaml)
2. [`configs/narrow_random_stage2_250k_from_stage1_150k.yaml`](configs/narrow_random_stage2_250k_from_stage1_150k.yaml)
3. [`configs/random_full_stage3_400k_after_stage2_250k.yaml`](configs/random_full_stage3_400k_after_stage2_250k.yaml)

This chain produced the best single-seed result, but it is not the default because it showed high variance in multi-seed revalidation.

## Current Assessment

The 3-stage curriculum is currently the best validated structure for this repository.

- The current default fixed-budget schedule is `300k / 300k / 300k`.
- The strongest single-seed candidate is `150k / 250k / 400k`.
- These should be treated differently:
  - `300k / 300k / 300k`: default training recipe
  - `150k / 250k / 400k`: experimental aggressive candidate

Current validation basis:

- `150k / 250k / 400k`
  - seed `42`: `success_rate = 0.80`, `mean_final_distance = 0.223`
  - seed `7`: `success_rate = 0.05`, `mean_final_distance = 0.544`
  - seed `123`: `success_rate = 0.45`, `mean_final_distance = 0.603`
- `300k / 300k / 300k`
  - seed `42`: `success_rate = 0.70`, `mean_final_distance = 0.290`
  - seed `7`: `success_rate = 0.20`, `mean_final_distance = 0.596`
  - seed `123`: `success_rate = 0.70`, `mean_final_distance = 0.278`

Aggregate interpretation:

- `150k / 250k / 400k`
  - mean `success_rate ≈ 0.43`
  - mean `mean_final_distance ≈ 0.457`
- `300k / 300k / 300k`
  - mean `success_rate ≈ 0.53`
  - mean `mean_final_distance ≈ 0.388`

So the aggressive schedule wins on the best seed, but loses on robustness.

## Empirical Findings Behind the Current Schedule

These findings are useful for collaborators because they explain why the current schedule looks the way it does.

- The current best single-seed chain (`150k / 250k / 400k`) did not come from intuition alone. It came from direct schedule comparisons.
- Stage 2 is **not** monotonic in practice. In other words, "train Stage 2 longer" is not automatically better.
- Under the `stage1 = 150k` setup:
  - `stage2 = 200k` was too short and led to weak transfer into Stage 3
  - `stage2 = 250k` performed better
  - `stage2 = 300k` was worse than `250k`
- This means the narrow-random phase appears to have a useful window rather than a simple "the longer the better" trend.

Single-seed evidence for the Stage 2 comparison:

- `150k / 200k / 400k`
  - Stage 2: `valid/success_rate = 0.20`
  - Final Stage 3: `valid/success_rate = 0.65`
  - Final Stage 3: `valid/mean_final_distance = 0.366`

- `150k / 250k / 400k`
  - Stage 2: `valid/success_rate = 0.45`
  - Final Stage 3: `valid/success_rate = 0.80`
  - Final Stage 3: `valid/mean_final_distance = 0.223`

- `150k / 300k / 400k` was not promoted because the Stage 2 branch itself had already degraded relative to `250k`

Practical interpretation:

- Stage 2 should be treated as a transition phase with a useful operating window
- The goal is to switch once the policy has clearly started solving the narrow-random task, not to maximize Stage 2 duration
- This is why the stage-switch guidance focuses on the **first stable useful window** instead of a fixed "longer is better" assumption
- However, the `150k / 250k / 400k` branch showed too much seed sensitivity to replace the default outright

## Stage Switching Guidance

The curriculum is not meant to be blindly hard-coded forever. The switching guidance below is an **experimental decision rule**, not code that runs automatically.

- Stage 1 -> Stage 2
  - `valid/mean_final_distance` has dropped and then plateaued
  - `valid/mean_release_step` is stable
  - `valid/mean_release_ball_speed` is stable
  - nonzero success is not required yet

- Stage 2 -> Stage 3
  - do not assume "longer is always better"
  - switch when Stage 2 first reaches a stable useful window:
    - `valid/success_rate` is roughly in the `0.4 - 0.5` range
    - `valid/min_distance_to_target` is already close to `target_radius`
    - `valid/mean_final_distance` is no longer rapidly improving, or has started to wobble

- Stop Stage 3
  - `valid/success_rate` has plateaued on the full-random task
  - `valid/mean_final_distance` is no longer improving
  - multiple eval windows are stable

Important:
- Stage 2 is not monotonic in practice.
- A tested `150k / 200k / 400k` schedule underperformed because Stage 2 was cut too early.
- A tested `150k / 250k / 400k` schedule is the best single-seed branch, but it is not robust enough to replace the default yet.
- Until a better schedule wins on multiple seeds, use `300k / 300k / 300k` as the default and treat more aggressive schedules as experimental candidates.

## Config Semantics

### Observation modes

- `arm_target_only`
  - joint angles + joint velocities + target position
- `arm_target_release`
  - `arm_target_only` + `released_flag`
  - current recommended mode
- `full_throw_state`
  - adds raw ball position and velocity
  - kept for ablation only; not the recommended default

### Reward modes

- `absolute_distance`
  - legacy shaping
- `distance_progress`
  - rewards progress toward the target instead of static proximity persistence
  - current recommended mode

### Target definition

- `target.mode: fixed`
  - one fixed 3D target
- `target.mode: random`
  - sample target coordinates from the configured ranges

### Success condition

Success is currently defined as:
- the ball has been released
- the ball enters the 3D target radius
- the ball is still above ground (`z >= 0.05`)

This matches the current project interpretation: the target may be on the ground or in the air.

## Key W&B Metrics

The most useful metrics to monitor are:

- `valid/success_rate`
- `valid/mean_final_distance`
- `valid/min_distance_to_target`
- `valid/mean_ep_length`
- `valid/release_rate`
- `valid/mean_release_step`
- `valid/mean_release_ball_speed`
- `valid/ground_miss_rate`
- `valid/timeout_no_release_rate`
- `valid/timeout_after_release_rate`

These are more informative than reward alone.

## Reproducing Recent Ablations

Useful frozen configs in [`configs/`](configs):

- Reward comparisons:
  - [`reward_compare_before_300k.yaml`](configs/reward_compare_before_300k.yaml)
  - [`reward_compare_after_300k.yaml`](configs/reward_compare_after_300k.yaml)
- Observation/control ablations:
  - [`random_full_pre679_compat_300k.yaml`](configs/random_full_pre679_compat_300k.yaml)
  - [`random_full_post679_300k.yaml`](configs/random_full_post679_300k.yaml)
  - [`random_full_only7_ablation_300k.yaml`](configs/random_full_only7_ablation_300k.yaml)
  - [`random_full_obsv2_300k.yaml`](configs/random_full_obsv2_300k.yaml)

## Known Limitations

- The target is represented numerically but is not yet visualized in the GUI.
- Training logic has been split into modules, but the repository is still small and intentionally lightweight rather than fully packaged.
