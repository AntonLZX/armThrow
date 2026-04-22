# armThrow

PyBullet + PPO prototype for the ME5406 Project 2 throwing task.

The current codebase trains a 3-DOF arm to throw a ball at a 3D target. The target may be on the ground or in the air.

## Model Validation
Final artifacts for our three reported result families are stored under `models/`:

- `models/physics_baseline/`
- `models/best_single_run/`
- `models/curriculum_auto/`

Each directory contains the exported `model.zip`. Where available, we also include
success / failure GIFs, final-frame PNGs, and short text summaries. The configs used
to train the models are stored under `configs/`.

To validate these models using our test script, run
```bash
python test.py
        --model models/best_single_run/model.zip 
        --config configs/best_single_run/random_full_single_run_1500k.yaml
```

To re-generate gifs from the final model, run
```bash
python capture_success.py 
        --config configs/best_single_run/random_full_single_run_1500k.yaml
        --model models/best_single_run/model.zip
        --output-dir tmp/success_capture
```


## Repository Layout

- [`train.py`](train.py): CLI entry point and PPO setup
- [`env.py`](env.py): environment dynamics, reward, observation, termination logic
- [`callbacks.py`](callbacks.py): GIF recording, W&B logging, eval callbacks
- [`config.py`](config.py): config loading, normalization, run directory helpers
- [`capture_success.py`](capture_success.py): local utility to search for a successful rollout and save a GIF/PNG proof
- [`arm.urdf`](arm.urdf): 3-DOF arm model
- [`visualize_arm.py`](visualize_arm.py): manual visualization/debug script
- [`configs/`](configs): frozen experiment configs and curriculum stages
- [`physics_baseline.py`](physics_baseline.py): scripted physics controller for ArmThrow
- [`sweep.py`](sweep.py): Wandb hyperparameter sweep runner
- [`test.py`](test.py): Validation / testing script for any ArmThrow model

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
python train.py --config configs/best_single_run/random_full_single_run_1500k.yaml
```

Render is set to false, but you can turn it on using
```bash
python train.py --config configs/best_single_run/random_full_single_run_1500k.yaml --render
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

### 3. Automatic curriculum entry

The repo also provides a dedicated automatic stage-switch runner:

```bash
python train_curriculum.py --config configs/curriculum_auto/curriculum_auto_target_best_working.yaml
```

This entry point will:

- run `stage1 -> stage2 -> stage3` sequentially
- evaluate `valid/*` metrics after each configured chunk
- switch or stop stages based on rules defined in the curriculum YAML
- support `threshold.consecutive` for "must pass N evals in a row"
- support `window_stat` for rolling-window readiness / stop targets
- save per-stage artifacts such as `config.yaml`, `eval_metrics.jsonl`, and `stage_summary.json`
- save run-level artifacts such as `curriculum_config.yaml`, `curriculum_events.jsonl`, and `curriculum_summary.json`

The final reported automatic curriculum recipe in this submission branch is:

```bash
python train_curriculum.py --config configs/curriculum_auto/curriculum_auto_target_best_working.yaml
```

It uses the three stage configs:

- `configs/curriculum_auto/fixed_center_stage1_auto_target_500k.yaml`
- `configs/curriculum_auto/narrow_random_stage2_auto_target_500k.yaml`
- `configs/curriculum_auto/random_full_stage3_auto_target_900k.yaml`

The automatic logic keeps `min_timesteps: 0` for all three stages and uses
`max_timesteps` only as a safety cap. Stage switching is triggered by rolling-window
targets reverse-engineered from the strongest fixed-budget runs:

- `stage1 -> stage2`
  - `valid/release_rate >= 0.95` for 4 consecutive evals
  - recent-5 `valid/min_distance_to_target` max `<= 0.47`
  - recent-5 `valid/mean_final_distance` max `<= 0.54`
- `stage2 -> stage3`
  - `valid/release_rate >= 0.95` for 4 consecutive evals
  - recent-6 `valid/success_rate` max `>= 0.65`
  - recent-6 `valid/min_distance_to_target` min `<= 0.10`
  - recent-6 `valid/mean_final_distance` min `<= 0.35`
- `stage3 stop`
  - recent-5 `valid/success_rate` mean `>= 0.99`
  - recent-5 `valid/success_rate` min `>= 0.95`
  - recent-5 `valid/ground_miss_rate` mean `<= 0.02`
  - recent-5 `valid/mean_final_distance` mean `<= 0.08`

In the validated best automatic run (`seed=42`), these rules triggered at:

- stage1 switch: `301056` timesteps
- stage2 switch: `294912` timesteps
- stage3 stop: `466944` timesteps

The corresponding stage-3 tail metrics were:

- tail-5 success mean: `0.99`
- tail-5 success minimum: `0.95`
- tail-5 ground-miss mean: `0.01`
- tail-5 mean-final-distance mean: `0.0722`

The original manual chain is still available via `train.py` plus `--load-model`
if you want to replay the stages explicitly.

## Evaluation

Two scripts are provided for evaluating a trained model after training is complete.

### test.py — Quantitative validation

`test.py` loads a `model.zip` and runs four structured test suites, printing a full report to stdout.

```bash
# Minimal usage — uses built-in defaults matching the final curriculum stage
python test.py --model model.zip

# With the config that was used during training (recommended)
python test.py --model runs/<run>/model.zip --config runs/<run>/config.yaml

# More episodes for stable statistics
python test.py --model model.zip --n-episodes 100

# With PyBullet GUI (slow, useful for visual debugging)
python test.py --model model.zip --render
```

**Suite 1 — Core Performance** runs `--n-episodes` episodes against the random target range and
reports the env's built-in metrics:

- Success rate and release rate
- Mean / std reward and mean episode length
- Mean final distance and mean minimum distance to target
- Termination breakdown (success / ground miss / timeout variants)

**Suite 2 — Sanity Checks** runs eight targeted assertions:

| Check | What it verifies |
|---|---|
| Ball release | Ball is released (constraint removed) in at least one episode |
| Release timing | Mean release step is well before the timeout limit |
| Joint velocity limits | `max_abs_joint_velocity` never exceeds the configured `joint_velocity_limit` |
| Action bounds | All model outputs stay within `[-1, 1]` (the action space contract) |
| Joint angle range | Joint angles never exceed ±4π (guards against unconstrained spinning) |
| Release ball speed | Speed at release is physically plausible (0.01–30 m/s) |
| Gravity / physics | Ball z-position descends from its peak after release (gravity is active) |
| Ball movement | Ball travels forward at least 0.5 m from the arm base after release |

**Suite 3 — Progressive Difficulty** tests success rate across five target tiers to check that
performance degrades gracefully as difficulty increases:

| Tier | Target |
|---|---|
| Easy | Fixed center [2.0, 0.0, 0.5] — trained distribution centre |
| Medium | Random [1.8–2.2, ±0.2, 0.4–0.6] — trained range |
| Hard | Random [1.5–2.5, ±0.5, 0.3–0.7] — wider than training |
| Extreme | Fixed far target [3.0, 0.0, 0.5] — out-of-distribution distance |
| Aerial | Fixed high target [2.0, 0.0, 1.5] — out-of-distribution height |

Also checks that success rate is monotonically non-increasing with difficulty (with 5% slack for
episode variance).

**Suite 4 — Summary** prints a combined PASS / FAIL verdict for all checks.

### capture_success.py — Visual proof (GIF + PNG)

`capture_success.py` rolls out the policy repeatedly and saves rendered clips of the best success
and the worst failure found within a fixed attempt budget.

```bash
python capture_success.py \
  --config runs/<run_dir>/config.yaml \
  --model runs/<run_dir>/model.zip \
  --output-dir tmp/success_capture
```

If deterministic rollout does not find a success quickly, retry with stochastic actions:

```bash
python capture_success.py \
  --config runs/<run_dir>/config.yaml \
  --model runs/<run_dir>/model.zip \
  --output-dir tmp/success_capture \
  --stochastic
```

For each captured episode the script writes three files:

- `success_seed<N>.gif` / `worst_failure_seed<N>.gif` — animated rollout
- `success_seed<N>_final.png` / `worst_failure_seed<N>_final.png` — final frame
- `success_seed<N>_summary.txt` / `worst_failure_seed<N>_summary.txt` — text record with
  `final_distance_to_target`, `min_distance_to_target`, `release_step`, `release_ball_speed`,
  `target`, `target_radius`, and `total_reward`

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--config` | *(required)* | Config YAML used to build the env |
| `--model` | *(required)* | Path to `model.zip` |
| `--output-dir` | `tmp/success_capture` | Directory to write GIF / PNG / summary |
| `--seed` | `42` | Base evaluation seed |
| `--max-attempts` | `50` | Episode budget to search for a success |
| `--stochastic` | off | Use stochastic policy instead of deterministic rollout |

The script exits with code `0` if a success was found, `1` otherwise.



## Training Recipes

### Physics Baseline

```bash
python train.py --config configs/physics_baseline.yaml
```

### Best Single Run
```bash
python train.py --config configs/best_single_run/random_full_single_run_1500k.yaml
```


### Automatic Curriculum

```bash
python train_curriculum.py --config configs/curriculum_auto/curriculum_auto_target_best_working.yaml
```

This is the final target-based automatic curriculum recipe used in our report-facing
results. It keeps all three stages on low `min_timesteps` and lets the curriculum
advance only when the rolling-window readiness targets are met.




## Automatic Curriculum Condition Semantics

The automatic runner accepts three condition families:

- `threshold`
  - fields: `metric`, `op`, `value`
  - optional: `consecutive`
  - `consecutive: N` means the threshold must pass on the most recent `N` evals

- `window_stat`
  - fields: `metric`, `statistic`, `window`, `op`, `value`
  - `statistic` can be `mean`, `min`, `max`, or `last`
  - use this for target-style gates such as "recent success mean >= 0.99" or "recent ground miss mean <= 0.02"

- `plateau`
  - fields: `metric`, `mode`, `window`
  - optional legacy field: `min_delta`
  - optional new fields: `rel_min_delta`, `patience`
  - if `rel_min_delta` is set, plateau is decided by **relative improvement** instead of absolute improvement
  - `patience: P` means the last `P` overlapping plateau windows must all satisfy the plateau rule

Compatibility notes:

- existing YAMLs without `consecutive` still behave as a single-eval threshold check
- existing YAMLs without `rel_min_delta` or `patience` still use the old absolute `min_delta` behavior

Practical tuning guidance:

- to make a threshold gate stricter, raise `consecutive`
- to make a plateau easier to trigger, raise `rel_min_delta`
- to make a plateau more stable but slower, raise `patience`
- to make stage1 switch earlier, lower `stage1.min_timesteps`
- to delay stage2 or stage3 stopping, lower `rel_min_delta`, raise `patience`, or raise the readiness threshold

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
- `visualize_target: true`
  - renders the target sphere, center marker, and success highlight in both GUI runs and recorded frames

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


# AI Declaration
Claude Sonnet 4.6 and ChatGPT were used to help write this code, mostly for clear code comments, documentation, and improving efficiency of base logic. We are responsible for the content and quality of the submitted work.