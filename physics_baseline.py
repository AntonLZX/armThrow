"""
physics_baseline.py — Scripted physics controller for ArmThrow.

The controller uses analytical projectile motion to decide exactly when to
release the ball so that it hits the target.  No learning is involved.

Strategy
--------
Phase 1 YAW_ALIGN:
    Only joint 0 moves to align directly with target yaw
    Joints 1 & 2 are held stationary.

Phase 2 WINDUP (per-target depth):
    Joints 1 & 2 swing backward at full scale to build negative velocity.
    Joint 0 holds the aligned yaw.
    Depth (steps) is computed per episode from the required throw arc.

Phase 3 SWING:
    Joints 1 & 2 swing forward at a per-target effective scale.
    At every step the controller forward-simulates the projectile that would
    result from releasing *right now*.  As soon as that trajectory passes
    within target_radius the ball is released.
    If the swing completes without release (no trajectory window found),
    the controller retries from Phase 1 with a slightly higher scale
    (up to max_attempts times per episode).
Usage
-----
    # Run evaluation and log to wandb
    python physics_baseline.py --config configs/physics_baseline.yaml

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
from metrics import evaluate_episodes


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
    _MAX_YAW_STEPS    = 40   # 0.67 s — PD control converges faster than pure-P;
                             #          saved 20 steps are available for retries
    _MAX_WINDUP_STEPS = 80   # 1.3 s  — upper bound; per-target value is shorter
    _MAX_SWING_STEPS  = 80   # 1.3 s  — if no release by here, retry

    def __init__(self, env_cfg, windup_steps=30, swing_scale=1.0):
        self.env_cfg = env_cfg
        self.windup_steps = windup_steps  # kept for save/load compatibility
        self.swing_scale = swing_scale
        self.device = "cpu"  # SB3 interface compatibility
        self._reset_state()

    def _reset_state(self):
        """Reset all per-episode mutable state."""
        self._phase = "yaw_align"   # yaw_align → windup → swing → (retry)
        self._phase_step = 0        # steps elapsed in current phase
        self._attempt = 0           # retry counter
        self._max_attempts = 3
        # Per-episode computed parameters (set in _compute_throw_params)
        self._target_yaw = 0.0
        self._effective_scale = self.swing_scale
        self._original_scale = self.swing_scale   # stored for retry bracketing
        self._effective_windup_steps = self.windup_steps
        self._v_min_threshold = 0.0              # minimum speed before _trajectory_hits may release
        self._params_ready = False
        self._env = None            # set via reset_episode(env=...) to enable direct snap

    def reset_episode(self, env=None):
        """Must be called at the start of each episode.

        Pass the ArmThrowEnv instance to enable direct joint-0 snapping:
            model.reset_episode(env=env)
        When env is provided the yaw-align phase is skipped entirely —
        joint 0 is teleported to the target yaw via resetJointState and
        then held at zero velocity throughout windup and swing.
        """
        self._reset_state()
        self._env = env

    # -- Per-target throw parameter computation --

    def _compute_throw_params(self, obs):
        """
        Compute per-episode throw parameters from the first observation.

        Uses projectile equations to derive:
          _target_yaw              — direction joint 0 must face
          _effective_scale         — fraction of swing_scale to use (tuned to
                                     the minimum speed that reaches the target,
                                     with a 50% margin)
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

        # Yaw: joint 0 must face the horizontal direction of the target
        self._target_yaw = math.atan2(float(target[1]), float(target[0]))

        # If the caller supplied the env, snap joint 0 directly to the target
        # yaw in one step and skip the velocity-ramp yaw-align phase entirely.
        if self._env is not None:
            import pybullet as p
            p.resetJointState(
                self._env.robot_id, 0,
                targetValue=self._target_yaw,
                targetVelocity=0.0,
                physicsClientId=self._env.client,
            )
            self._phase = "windup"
            self._phase_step = 0

        # Projectile: compute minimum release speed to reach the target 
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

        # Target speed: 50% above minimum so the discriminant is comfortably positive
        # and the release window is wide enough to be caught reliably.
        v_max    = self._V_MAX_AT_UNIT_SCALE * self.swing_scale
        v_target = min(v_min * 1.5, v_max)
        # Floor at 20% so close targets get a proportionate low-speed swing
        # (40% was too high: it forced ~5 m/s for targets needing only 3 m/s,
        # making the trajectory-check window extremely narrow).
        self._effective_scale = max(v_target / v_max, 0.2) * self.swing_scale
        self._original_scale  = self._effective_scale  # stored for retry bracketing

        # Release gate: don't allow _trajectory_hits to trigger until the ball
        # is moving at least 20% above v_min.  At exactly v_min there is only
        # one valid launch angle (45°), so the success window has zero width —
        # any 1-step lag or velocity-direction error causes a miss.  The 1.2×
        # factor opens the window to a usable angular range while staying well
        # below v_target (1.5×), so the gate never blocks the intended release.
        self._v_min_threshold = v_min * 1.2

        # Windup depth: enough arc to sweep through the release window
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

    # predict compatible with stable baselines

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
        j0_ang_vel = float(obs[3])   # actual joint 0 angular velocity (for PD yaw)
        ball_pos  = obs[6:9]
        ball_vel  = obs[9:12]
        target    = obs[12:15]
        radius    = float(self.env_cfg.get("target_radius", 0.1))
        joint_vel_limit = float(self.env_cfg.get("joint_velocity_limit", 10.0))

        self._phase_step += 1
        s = self._effective_scale

        # Shared yaw error (wrapped to [-π, π])
        yaw_error = self._target_yaw - float(joint_ang[0])
        yaw_error = (yaw_error + math.pi) % (2.0 * math.pi) - math.pi
        # Tighter tolerance: 0.03 rad (~1.7°).  At 5 m/s to a 2m target,
        # 3° gives ~0.1m lateral deviation — right at the 0.1m radius edge.
        yaw_aligned = abs(yaw_error) < 0.03


        # Phase yaw_align
        # PD control on joint 0 (obs[3] = actual angular velocity) to damp
        # oscillation and converge faster
        # Joints 1 & 2 are driven to zero so they start the windup from rest.
        if self._phase == "yaw_align":
            # Kp=8, Kd=2 — derivative term brakes the approach and prevents
            # overshoot
            j0_vel = float(np.clip(
                yaw_error * 8.0 - j0_ang_vel * 2.0,
                -self.swing_scale, self.swing_scale,
            ))
            if yaw_aligned or self._phase_step >= self._MAX_YAW_STEPS:
                self._phase      = "windup"
                self._phase_step = 0
            return np.array([j0_vel, 0.0, 0.0, -1.0], dtype=np.float32)


        # Phase windup
        # Joints 1 & 2 wind backward at effective_scale (not full swing_scale).
        # Winding up at the same scale used for the throw means no deceleration
        # phase at the start of swing — the arm transitions smoothly from
        # -s·ω_max to +s·ω_max, cutting the wasted swing steps from ~30 to ~12.
        # Exit early if the joints are already saturated at the target velocity.
        elif self._phase == "windup":
            # When joint 0 was snapped directly, hold it at zero velocity.
            # Otherwise use a soft P-controller to maintain yaw alignment.
            j0_vel = 0.0 if self._env is not None else float(np.clip(
                yaw_error * 4.0 - j0_ang_vel * 1.0,
                -self.swing_scale, self.swing_scale,
            ))
            max_wind = min(self._effective_windup_steps, self._MAX_WINDUP_STEPS)
            # Adaptive early exit: joints 1 & 2 have reached target winding velocity
            joints_wound_up = (
                float(obs[4]) <= -s * joint_vel_limit * 0.85
                and float(obs[5]) <= -s * joint_vel_limit * 0.85
                and self._phase_step >= 5
            )
            if self._phase_step >= max_wind or joints_wound_up:
                self._phase      = "swing"
                self._phase_step = 0
            return np.array([j0_vel, -s, -s, -1.0], dtype=np.float32)


        # Phase swing
        # Joints 1 & 2 drive forward at effective_scale.
        # Joint 0 uses a very light PD correction (0.12×) to avoid fighting
        # the throw while still correcting slow drift.
        # Release when _trajectory_hits() confirms the current trajectory hits.
        # On failure, bracket the scale: attempt 1 tries 0.80×, attempt 2
        # tries 1.20×, covering both "too fast" and "too slow" cases.
        elif self._phase == "swing":
            # When joint 0 was snapped directly, hold at zero velocity.
            # Otherwise use a light correction to counteract slow drift.
            j0_vel = 0.0 if self._env is not None else float(np.clip(
                yaw_error * 4.0 - j0_ang_vel * 1.0,
                -self.swing_scale, self.swing_scale,
            )) * 0.12

            ball_speed = float(np.linalg.norm(ball_vel))
            if ball_speed >= self._v_min_threshold and \
                    self._trajectory_hits(ball_pos, ball_vel, target, radius):
                return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            if self._phase_step >= self._MAX_SWING_STEPS:
                if self._attempt < self._max_attempts - 1:
                    self._attempt       += 1
                    self._phase          = "yaw_align"
                    self._phase_step     = 0
                    # Bracket: retry 1 tries lower, retry 2 tries higher.
                    if self._attempt == 1:
                        self._effective_scale = max(
                            self._original_scale * 0.80, 0.15
                        )
                    else:
                        self._effective_scale = min(
                            self._original_scale * 1.20, self.swing_scale
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
    eval_cfg = {**cfg["env"], "render": False}
    eval_env = ArmThrowEnv(eval_cfg)
    metrics = evaluate_episodes(baseline, eval_env, n_episodes=n_episodes, seed=cfg["seed"], verbose=True)
    eval_env.close()

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
