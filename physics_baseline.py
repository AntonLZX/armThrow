"""
physics_baseline.py — Scripted physics controller for ArmThrow.

The controller uses analytical projectile motion to decide exactly when to
release the ball so that it hits the target.  No learning is involved.

Strategy
--------
Phase 1 YAW_ALIGN:
    Only joint 0 moves, driving toward arctan2(target_y, target_x).
    Joints 1 & 2 are held stationary.
    Completes when |yaw_error| < 0.05 rad (~3°), or after 60 steps.

Phase 2 WINDUP (per-target depth):
    Joints 1 & 2 swing backward at full scale to build negative velocity.
    Joint 0 holds the aligned yaw with a gentle P-controller.
    Depth (steps) is computed per episode from the required throw arc.

Phase 3 SWING:
    Joints 1 & 2 swing forward at a per-target effective scale.
    At every step the controller forward-simulates the projectile that would
    result from releasing *right now*.  As soon as that trajectory passes
    within target_radius the ball is released.
    If the swing completes without release (no trajectory window found),
    the controller retries from Phase 1 with a slightly higher scale
    (up to max_attempts times per episode).

Previous two-phase strategy (replaced):
Phase A WINDUP  (windup_steps):
    Joint 0 (assumed yaw) rotates to face the target's horizontal angle
    *simultaneously* with joints 1 & 2 winding backward.
    Problem: joint 0 rarely converges before the swing starts, causing
    the ball velocity to point in the wrong direction → timeout_no_release.

Phase 2 SWING:
    All joints swing forward at maximum effort.
    At every step the controller forward-simulates the projectile that would
    result from releasing *right now* (ball position + current ball velocity +
    gravity).  As soon as that simulated trajectory passes within
    target_radius of the target the ball is released.

The resulting model.zip is loadable via PhysicsBaseline.load() and has the
same predict() interface as an SB3 model, so it works with capture_success.py.

Usage
-----
    # Run evaluation and log to wandb
    python physics_baseline.py --config configs/physics_baseline.yaml

    # Evaluate only, no wandb
    python physics_baseline.py --config configs/physics_baseline.yaml --no-wandb

    # Use with capture_success.py (capture_success.py handles loading)
    python capture_success.py --config configs/physics_baseline.yaml \\
        --model runs/<run_dir>/model.zip

Note: the env config MUST use observation_mode: "full_throw_state" so the
      predict() method can read ball position, ball velocity and target from
      the observation vector.
"""

import json
import zipfile
from pathlib import Path

import math

import numpy as np
import yaml

from callbacks import WANDB_AVAILABLE, wandb
from config import _coerce_float, _coerce_int, make_run_dir, resolve_wandb_name, set_seed
from env import ArmThrowEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def normalize_physics_config(cfg):
    if not isinstance(cfg, dict):
        raise TypeError("Top-level config must be a mapping")

    cfg["seed"] = _coerce_int(cfg.get("seed", 42), "seed", minimum=0)

    for section in ("env", "algo", "logging"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise TypeError(f"Config section '{section}' must be present and be a mapping")

    env = cfg["env"]
    algo = cfg["algo"]
    logging_cfg = cfg["logging"]

    # env
    env["max_steps"] = _coerce_int(env.get("max_steps"), "env.max_steps", minimum=1)
    env["end_effector_link_index"] = _coerce_int(
        env.get("end_effector_link_index"), "env.end_effector_link_index", minimum=0
    )
    env["accel_scale"] = _coerce_float(env.get("accel_scale"), "env.accel_scale", minimum=0.0)
    env["motor_force_limit"] = _coerce_float(
        env.get("motor_force_limit"), "env.motor_force_limit", minimum=0.0
    )
    env["joint_velocity_limit"] = _coerce_float(
        env.get("joint_velocity_limit", 10.0), "env.joint_velocity_limit", minimum=0.0
    )
    env["target_radius"] = _coerce_float(env.get("target_radius", 0.1), "env.target_radius", minimum=0.0)
    env["release_success_bonus"] = _coerce_float(
        env.get("release_success_bonus", 1.0), "env.release_success_bonus"
    )

    # Force full_throw_state — needed so ball velocity is available in obs
    if env.get("observation_mode") != "full_throw_state":
        print(
            f"[physics_baseline] observation_mode forced to 'full_throw_state' "
            f"(was {env.get('observation_mode')!r})"
        )
    env["observation_mode"] = "full_throw_state"

    # algo
    algo["name"] = str(algo.get("name", "Physics"))
    algo["windup_steps"] = _coerce_int(algo.get("windup_steps", 30), "algo.windup_steps", minimum=1)
    algo["swing_scale"] = _coerce_float(algo.get("swing_scale", 1.0), "algo.swing_scale", minimum=0.0)
    algo["n_eval_episodes"] = _coerce_int(
        algo.get("n_eval_episodes", 200), "algo.n_eval_episodes", minimum=1
    )

    if "gif_every_n_episodes" in logging_cfg:
        logging_cfg["gif_every_n_episodes"] = _coerce_int(
            logging_cfg["gif_every_n_episodes"], "logging.gif_every_n_episodes", minimum=1
        )

    return cfg


def load_physics_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_physics_config(cfg)


# ---------------------------------------------------------------------------
# Physics controller
# ---------------------------------------------------------------------------

class PhysicsBaseline:
    """
    Scripted physics controller.  Implements the SB3 model predict() interface.

    Observation layout expected (full_throw_state, dim=16):
        obs[0:3]   joint angles
        obs[3:6]   joint velocities
        obs[6:9]   ball position
        obs[9:12]  ball velocity
        obs[12:15] target position
        obs[15]    released flag
    """

    # Used by capture_success.py to distinguish from PPO models
    POLICY_TYPE = "PhysicsBaseline"

    # Estimated ball speed at full swing_scale=1.0, calibrated from observed
    # mean release speed (~12.3 m/s) with joint_velocity_limit=10 rad/s.
    _V_MAX_AT_UNIT_SCALE = 12.3  # m/s

    # Maximum steps per phase before the controller forces a transition
    _MAX_YAW_STEPS    = 60   # 1.0 s  — time to converge joint 0 to target yaw
    _MAX_WINDUP_STEPS = 80   # 1.3 s  — upper bound; per-target value is shorter
    _MAX_SWING_STEPS  = 80   # 1.3 s  — if no release by here, retry

    def __init__(self, env_cfg, windup_steps=30, swing_scale=1.0):
        self.env_cfg = env_cfg
        self.windup_steps = windup_steps  # kept for save/load compatibility
        self.swing_scale = swing_scale
        self.device = "cpu"  # SB3 interface compatibility
        self._reset_state()

    # -- Episode management --

    def _reset_state(self):
        """Reset all per-episode mutable state."""
        self._phase = "yaw_align"   # yaw_align → windup → swing → (retry)
        self._phase_step = 0        # steps elapsed in current phase
        self._attempt = 0           # retry counter
        self._max_attempts = 3
        # Per-episode computed parameters (set in _compute_throw_params)
        self._target_yaw = 0.0
        self._effective_scale = self.swing_scale
        self._effective_windup_steps = self.windup_steps
        self._params_ready = False

    def reset_episode(self):
        """Must be called at the start of each episode."""
        self._reset_state()

    # -- Per-target throw parameter computation --

    def _compute_throw_params(self, obs):
        """
        Compute per-episode throw parameters from the first observation.

        Uses projectile equations to derive:
          _target_yaw              — direction joint 0 must face
          _effective_scale         — fraction of swing_scale to use (tuned to
                                     the minimum speed that reaches the target,
                                     with a 30% margin)
          _effective_windup_steps  — how many steps to spend in windup phase,
                                     scaled to the required throw arc

        Requires full_throw_state obs (dim ≥ 16) so ball position is available.
        Falls back to defaults if obs is too short.
        """
        if len(obs) < 16:
            # Not full_throw_state; use defaults from __init__
            self._params_ready = True
            return

        ball_pos = obs[6:9]
        target   = obs[12:15]

        # --- Yaw: joint 0 must face the horizontal direction of the target ---
        self._target_yaw = math.atan2(float(target[1]), float(target[0]))

        # --- Projectile: compute minimum release speed to reach the target ---
        dx = float(target[0]) - float(ball_pos[0])
        dy = float(target[1]) - float(ball_pos[1])
        dz = float(target[2]) - float(ball_pos[2])
        D  = math.sqrt(dx ** 2 + dy ** 2)   # horizontal distance at release
        g  = 9.81

        # Solve for the minimum launch speed v_min such that any real-valued
        # launch angle exists:
        #   discriminant of tan(θ) quadratic = D² - 4A(dz + A) ≥ 0,
        #   where A = g·D²/(2v²).
        # Setting discriminant = 0 and solving for A:
        #   4A² + 4·dz·A - D² = 0  →  A_min = (-dz + √(dz² + D²)) / 2
        inner = max(dz ** 2 + D ** 2, 0.0)
        A_min = (-dz + math.sqrt(inner)) / 2.0
        if D > 0.05 and A_min > 1e-6:
            v_min = math.sqrt(g * D ** 2 / (2.0 * A_min))
        else:
            v_min = self._V_MAX_AT_UNIT_SCALE * self.swing_scale  # fallback: use max

        # Target speed: 30% above minimum so the discriminant is comfortably positive
        v_max    = self._V_MAX_AT_UNIT_SCALE * self.swing_scale
        v_target = min(v_min * 1.3, v_max)
        # Floor at 40% of max so the arm always has enough momentum to swing cleanly
        self._effective_scale = max(v_target / v_max, 0.4) * self.swing_scale

        # --- Windup depth: enough arc to sweep through the release window ---
        # The release window is the range of arm angles at which _trajectory_hits
        # fires.  Farther or higher targets require higher launch angles, which
        # means more of the upward arc must be covered → deeper windup.
        # Rule of thumb derived from projectile geometry:
        #   close targets (D < 1.2 m or dz ≤ 0):  ~25 steps  (≈ 0.4 s)
        #   medium targets (D 1.2–2.0 m):          ~40 steps  (≈ 0.7 s)
        #   far / high targets (D > 2.0 m or dz > 0.5 m): ~60 steps (≈ 1.0 s)
        if D > 2.0 or dz > 0.5:
            self._effective_windup_steps = 60
        elif D > 1.2 or dz > 0.0:
            self._effective_windup_steps = 40
        else:
            self._effective_windup_steps = 25

        self._params_ready = True

    # -- SB3-compatible predict --

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        """Return (action, state) matching the SB3 model.predict() signature."""
        obs = np.asarray(obs, dtype=np.float32)
        batched = obs.ndim == 2
        if not batched:
            obs = obs[None]

        if episode_start is None:
            episode_start = np.zeros(len(obs), dtype=bool)
        else:
            episode_start = np.asarray(episode_start, dtype=bool)
            if episode_start.ndim == 0:
                episode_start = episode_start[None]

        actions = []
        for i, o in enumerate(obs):
            if episode_start[i]:
                self._reset_state()
            actions.append(self._step_action(o))

        actions = np.array(actions, dtype=np.float32)
        return (actions[0] if not batched else actions), state

    def _step_action(self, obs):
        """
        Compute action for a single observation using a three-phase strategy.

        Phase yaw_align:  drive joint 0 to target yaw; joints 1 & 2 stationary.
        Phase windup:     drive joints 1 & 2 backward; joint 0 holds yaw.
        Phase swing:      drive joints 1 & 2 forward; release when trajectory hits.
                          On failure, retry from yaw_align (up to _max_attempts).
        """
        released = obs[15] > 0.5
        if released:
            return np.zeros(4, dtype=np.float32)

        # Compute per-target parameters once, on the very first step
        if not self._params_ready:
            self._compute_throw_params(obs)

        joint_ang = obs[0:3]
        ball_pos  = obs[6:9]
        ball_vel  = obs[9:12]
        target    = obs[12:15]
        radius    = float(self.env_cfg.get("target_radius", 0.1))

        self._phase_step += 1
        s = self._effective_scale

        # Shared yaw correction term (used with different gains per phase)
        yaw_error   = self._target_yaw - float(joint_ang[0])
        yaw_error   = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi  # wrap to [-π, π]
        yaw_aligned = abs(yaw_error) < 0.05   # ~3° tolerance

        # ------------------------------------------------------------------
        # Phase yaw_align
        # Only joint 0 moves.  Joints 1 & 2 are held at zero velocity so
        # they do not interfere with the alignment dynamics.
        # Transitions when yaw error is within tolerance or step cap is hit.
        # ------------------------------------------------------------------
        if self._phase == "yaw_align":
            j0_vel = float(np.clip(yaw_error * 8.0, -self.swing_scale, self.swing_scale))
            if yaw_aligned or self._phase_step >= self._MAX_YAW_STEPS:
                self._phase      = "windup"
                self._phase_step = 0
            return np.array([j0_vel, 0.0, 0.0, -1.0], dtype=np.float32)

        # ------------------------------------------------------------------
        # Phase windup
        # Joints 1 & 2 wind backward at full swing_scale to build energy.
        # Joint 0 holds the aligned yaw with a gentle P-gain (the arm is
        # already aligned; this just corrects small drift).
        # Depth is _effective_windup_steps, capped at _MAX_WINDUP_STEPS.
        # ------------------------------------------------------------------
        elif self._phase == "windup":
            j0_vel   = float(np.clip(yaw_error * 4.0, -self.swing_scale, self.swing_scale))
            max_wind = min(self._effective_windup_steps, self._MAX_WINDUP_STEPS)
            if self._phase_step >= max_wind:
                self._phase      = "swing"
                self._phase_step = 0
            return np.array([j0_vel, -self.swing_scale, -self.swing_scale, -1.0], dtype=np.float32)

        # ------------------------------------------------------------------
        # Phase swing
        # Joints 1 & 2 swing forward at _effective_scale.
        # Joint 0 holds yaw at reduced gain (0.15×) to avoid fighting the throw.
        # Release when _trajectory_hits() confirms the current trajectory hits.
        # Retry from yaw_align if the swing step cap is reached without release.
        # ------------------------------------------------------------------
        elif self._phase == "swing":
            j0_vel = float(np.clip(yaw_error * 4.0, -self.swing_scale, self.swing_scale)) * 0.15

            if self._trajectory_hits(ball_pos, ball_vel, target, radius):
                return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            if self._phase_step >= self._MAX_SWING_STEPS:
                if self._attempt < self._max_attempts - 1:
                    self._attempt       += 1
                    self._phase          = "yaw_align"
                    self._phase_step     = 0
                    # Slightly increase scale in case the trajectory window was
                    # just above the speed that the previous attempt reached
                    self._effective_scale = min(
                        self._effective_scale * 1.15, self.swing_scale
                    )

            return np.array([j0_vel, s, s, -1.0], dtype=np.float32)

        # Fallback: all attempts exhausted, hold still
        return np.zeros(4, dtype=np.float32)

    def _trajectory_hits(self, pos, vel, target, radius):
        """
        Forward-simulate projectile from (pos, vel) under gravity and check
        whether it passes within radius of target.

        Uses 4× oversampled timesteps to avoid missing a narrow crossing window.
        """
        g  = 9.81
        dt = 1.0 / 240.0   # 4× the env's 60 Hz timestep
        tol = radius * 1.05  # small margin for discrete-time rounding

        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        tx, ty, tz = float(target[0]), float(target[1]), float(target[2])

        for i in range(1, 1200):   # up to 5 s of simulated flight
            t  = i * dt
            x  = px + vx * t
            y  = py + vy * t
            z  = pz + vz * t - 0.5 * g * t * t
            if z < 0.05:           # hit the ground (same threshold as env.py)
                break
            d = ((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2) ** 0.5
            if d <= tol:
                return True
        return False

    # -- Persistence --

    def save(self, path):
        """Save as model.zip containing JSON metadata (no neural-network weights)."""
        path = str(path)
        if not path.endswith(".zip"):
            path += ".zip"
        data = {
            "policy_type": self.POLICY_TYPE,
            "windup_steps": self.windup_steps,
            "swing_scale": self.swing_scale,
            "env_cfg": {k: v for k, v in self.env_cfg.items()},
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("policy_data.json", json.dumps(data, indent=2, default=str))
        print(f"Saved physics baseline to {path}")
        return path

    @classmethod
    def load(cls, path, env=None, **kwargs):
        """Load a PhysicsBaseline from a model.zip produced by save()."""
        path = str(path)
        if not path.endswith(".zip"):
            path += ".zip"
        with zipfile.ZipFile(path, "r") as zf:
            if "policy_data.json" not in zf.namelist():
                raise ValueError(f"Not a PhysicsBaseline model.zip: {path}")
            data = json.loads(zf.read("policy_data.json"))
        if data.get("policy_type") != cls.POLICY_TYPE:
            raise ValueError(
                f"Expected policy_type={cls.POLICY_TYPE!r}, got {data.get('policy_type')!r}"
            )
        return cls(
            env_cfg=data["env_cfg"],
            windup_steps=int(data["windup_steps"]),
            swing_scale=float(data["swing_scale"]),
        )

    @staticmethod
    def is_physics_baseline(path):
        """Return True if path looks like a PhysicsBaseline model.zip."""
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                if "policy_data.json" not in zf.namelist():
                    return False
                data = json.loads(zf.read("policy_data.json"))
                return data.get("policy_type") == PhysicsBaseline.POLICY_TYPE
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Evaluation  (same metric keys as WandbEvalCallback / train_imitation.py)
# ---------------------------------------------------------------------------

def _finite_mean(vals):
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_std(vals):
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.std(finite)) if finite else float("nan")


def evaluate(baseline, env_cfg, n_episodes, seed):
    """
    Run n_episodes with the PhysicsBaseline and return a metrics dict whose
    keys match WandbEvalCallback (valid/*) and train_imitation (reward/, control/).
    """
    eval_cfg = {**env_cfg, "render": False}
    env = ArmThrowEnv(eval_cfg)
    env.reset(seed=seed)

    lengths, final_dists, min_dists = [], [], []
    successes, releases, release_steps, release_speeds = [], [], [], []
    ground_misses, timeout_no_release, timeout_after_release = [], [], []
    pre_release_penalties, shapings, release_bonuses = [], [], []
    success_bonuses, failure_penalties = [], []
    max_joint_vels, mean_joint_vels, action_norms = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        baseline.reset_episode()
        done = truncated = False
        ep_len = 0
        last_info = {}

        while not (done or truncated):
            action, _ = baseline.predict(obs)
            obs, _, done, truncated, last_info = env.step(action)
            ep_len += 1

        lengths.append(ep_len)
        final_dists.append(last_info.get("final_distance_to_target", np.nan))
        min_dists.append(last_info.get("min_distance_to_target", np.nan))
        successes.append(float(last_info.get("success", 0.0)))
        releases.append(float(last_info.get("released", 0.0)))
        release_steps.append(last_info.get("release_step", np.nan))
        release_speeds.append(last_info.get("release_ball_speed", np.nan))
        ground_misses.append(float(last_info.get("termination_ground_miss", 0.0)))
        timeout_no_release.append(float(last_info.get("termination_timeout_no_release", 0.0)))
        timeout_after_release.append(float(last_info.get("termination_timeout_after_release", 0.0)))
        pre_release_penalties.append(last_info.get("reward_pre_release_penalty", np.nan))
        shapings.append(last_info.get("reward_shaping_component", np.nan))
        release_bonuses.append(last_info.get("reward_release_bonus_component", np.nan))
        success_bonuses.append(last_info.get("reward_success_bonus_component", np.nan))
        failure_penalties.append(last_info.get("reward_failure_penalty_component", np.nan))
        max_joint_vels.append(last_info.get("max_abs_joint_velocity", np.nan))
        mean_joint_vels.append(last_info.get("mean_abs_joint_velocity", np.nan))
        action_norms.append(last_info.get("mean_action_norm", np.nan))

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            sr = float(np.mean(successes))
            print(f"  [{ep + 1:4d}/{n_episodes}] running success_rate={sr:.3f}")

    env.close()

    return {
        "valid/mean_ep_length":          float(np.mean(lengths)),
        "valid/success_rate":            float(np.mean(successes)),
        "valid/release_rate":            float(np.mean(releases)),
        "valid/mean_final_distance":     _finite_mean(final_dists),
        "valid/std_final_distance":      _finite_std(final_dists),
        "valid/min_distance_to_target":  _finite_mean(min_dists),
        "valid/mean_release_step":       _finite_mean(release_steps),
        "valid/mean_release_ball_speed": _finite_mean(release_speeds),
        "valid/ground_miss_rate":        float(np.mean(ground_misses)),
        "valid/timeout_no_release_rate": float(np.mean(timeout_no_release)),
        "valid/timeout_after_release_rate": float(np.mean(timeout_after_release)),
        "reward/pre_release_penalty":    _finite_mean(pre_release_penalties),
        "reward/shaping":                _finite_mean(shapings),
        "reward/release_bonus":          _finite_mean(release_bonuses),
        "reward/success_bonus":          _finite_mean(success_bonuses),
        "reward/failure_penalty":        _finite_mean(failure_penalties),
        "control/max_abs_joint_velocity":  _finite_mean(max_joint_vels),
        "control/mean_abs_joint_velocity": _finite_mean(mean_joint_vels),
        "control/mean_action_norm":        _finite_mean(action_norms),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path="configs/physics_baseline.yaml", no_wandb=False, seed_override=None):
    cfg = load_physics_config(config_path)
    if seed_override is not None:
        cfg["seed"] = _coerce_int(seed_override, "seed_override", minimum=0)
    set_seed(cfg["seed"])

    run_dir = make_run_dir(cfg["logging"]["save_dir"], cfg["run_name"])
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    baseline = PhysicsBaseline(
        env_cfg=cfg["env"],
        windup_steps=cfg["algo"]["windup_steps"],
        swing_scale=cfg["algo"]["swing_scale"],
    )

    use_wandb = cfg["logging"]["use_wandb"] and not no_wandb
    if use_wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is enabled in config but not installed.")
        wandb.init(
            project=cfg["logging"]["project"],
            entity=cfg["logging"]["entity"],
            name=resolve_wandb_name(cfg["logging"], run_dir),
            config=cfg,
            tags=cfg["logging"].get("tags", []),
            notes=cfg["logging"].get("notes", ""),
            sync_tensorboard=False,
            dir=str(run_dir),
        )

    n_episodes = cfg["algo"]["n_eval_episodes"]
    print(f"\nEvaluating physics baseline over {n_episodes} episodes...")
    metrics = evaluate(baseline, cfg["env"], n_episodes=n_episodes, seed=cfg["seed"])

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if use_wandb:
        wandb.log({**metrics, "global_timestep": 0})

    if cfg["logging"]["save_model"]:
        baseline.save(str(run_dir / "model"))

    if use_wandb:
        wandb.finish()

    print(f"\nDone. Run saved to: {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Physics baseline controller for ArmThrow")
    parser.add_argument("--config", type=str, default="configs/physics_baseline.yaml")
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging regardless of config setting",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    args = parser.parse_args()

    main(config_path=args.config, no_wandb=args.no_wandb, seed_override=args.seed)
