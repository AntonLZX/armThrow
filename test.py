"""
test.py — Validation / testing script for any ArmThrow model.

Usage:
    python test.py --model ppo_arm_throw.zip
    python test.py --model runs/<run>/model.zip --config runs/<run>/config.yaml
    python test.py --model ppo_arm_throw.zip --render --n-episodes 20

The script runs four test suites and prints a structured report:

    1. Core Performance Metrics   — success rate, reward, episode length,
                                    release rate, min/final distance to target
    2. Sanity Checks              — ball release, velocity limits, action bounds,
                                    joint angle bounds, physics (gravity), release timing
    3. Progressive Difficulty     — success rate as target distance increases
    4. Summary Pass/Fail          — overall verdict
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import yaml
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from callbacks import WANDB_AVAILABLE, wandb
from env import ArmThrowEnv
from metrics import build_test_wandb_payload
from physics_baseline import PhysicsBaseline


# Algo names that are saved as PPO-compatible model.zip files
_SB3_ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}

# Try order for auto-detection: PPO first
_SB3_TRY_ORDER = [PPO, SAC, TD3, A2C, DDPG]


def load_model(model_path, config_path=None):
    """
    Load a model from model_path.  Returns (model, algo_name).

    Detection order:
      1. PhysicsBaseline  — detected by policy_data.json inside the zip
      2. Config algo.name — used when --config is provided
      3. Auto-detect      — tries each SB3 class in turn as a fallback
    """
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    # 1. Physics baseline
    if PhysicsBaseline.is_physics_baseline(str(path)):
        print("  Detected: PhysicsBaseline")
        return PhysicsBaseline.load(str(path)), "Physics"

    # 2. Config-declared algo
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        algo_name = str(raw.get("algo", {}).get("name", "PPO")).upper()
        load_name = algo_name
        cls = _SB3_ALGO_MAP.get(load_name)
        if cls is None:
            print(f"  Warning: unknown algo '{algo_name}' in config — falling back to auto-detect")
        else:
            print(f"  Detected from config: {algo_name}")
            return cls.load(str(path), device="cpu"), algo_name

    # 3. Auto-detect by trying each SB3 class
    print("  No config provided — auto-detecting algo...")
    for cls in _SB3_TRY_ORDER:
        try:
            model = cls.load(str(path), device="cpu")
            print(f"  Auto-detected: {cls.__name__}")
            return model, cls.__name__
        except Exception:
            continue

    raise ValueError(
        f"Could not load {path} with any known algorithm. "
        "Provide --config to specify the algo explicitly."
    )


def _build_default_env_cfg(render: bool = False, obs_mode: str = "arm_target_release") -> dict:
    """Minimal env config mirroring the final-stage training defaults."""
    return {
        "arm_urdf": "arm.urdf",
        "render": render,
        "max_steps": 240,
        "end_effector_link_index": 3,
        "accel_scale": 50.0,
        "motor_force_limit": 50.0,
        "joint_velocity_limit": 10.0,
        "target_radius": 0.1,
        "release_success_bonus": 1.0,
        "reward_mode": "distance_progress",
        "observation_mode": obs_mode,
        "visualize_target": False,
        "target": {
            "mode": "random",
            "fixed": [2.0, 0.0, 0.5],
            "random": {"x": [1.8, 2.2], "y": [-0.2, 0.2], "z": [0.4, 0.6]},
        },
    }


def _load_env_cfg_from_yaml(config_path: str, render: bool = False) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("env", raw)
    cfg["render"] = render
    cfg.setdefault("reward_mode", "distance_progress")
    cfg.setdefault("visualize_target", False)
    return cfg


def _prepare_env_cfg(config_path, render, algo_name):
    """Build env cfg and enforce observation_mode for PhysicsBaseline."""
    if config_path:
        cfg = _load_env_cfg_from_yaml(config_path, render)
    else:
        obs_mode = "full_throw_state" if algo_name == "Physics" else "arm_target_release"
        cfg = _build_default_env_cfg(render, obs_mode)

    if algo_name == "Physics":
        if cfg.get("observation_mode") != "full_throw_state":
            print("  [info] Forcing observation_mode=full_throw_state for PhysicsBaseline")
            cfg["observation_mode"] = "full_throw_state"

    return cfg


def run_episode(model, env: ArmThrowEnv, seed: int | None = None):
    """
    Roll out one episode and return a rich dict of per-episode statistics.
    Handles both SB3 models and PhysicsBaseline (reset_episode).
    """
    # PhysicsBaseline is stateful — must reset its internal step counter
    if hasattr(model, "reset_episode"):
        model.reset_episode()

    obs, _ = env.reset(seed=seed)

    ep_reward = 0.0
    ep_length = 0
    info_final = {}

    released = False
    release_step = None
    release_ball_speed = None
    ball_released_at_all = False

    # For physics check: ball z positions after release
    z_after_release = []

    # For action bound check
    max_action_abs = 0.0

    # For joint angle tracking — obs[0:3] are joint angles in all observation modes
    max_joint_angle = 0.0

    # For joint velocity tracking — obs[3:6] are joint velocities in all observation modes
    max_joint_vel_obs = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        max_action_abs = max(max_action_abs, float(np.max(np.abs(action))))
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1

        max_joint_angle = max(max_joint_angle, float(np.max(np.abs(obs[:3]))))
        max_joint_vel_obs = max(max_joint_vel_obs, float(np.max(np.abs(obs[3:6]))))

        if info.get("released") and not released:
            released = True
            ball_released_at_all = True
            release_step = info.get("release_step")
            release_ball_speed = info.get("release_ball_speed")

        if released and env.ball_id is not None:
            ball_pos, _ = p.getBasePositionAndOrientation(env.ball_id, physicsClientId=env.client)
            z_after_release.append(ball_pos[2])

        info_final = info
        if terminated or truncated:
            break

    # Physics check: after release the ball should peak then descend (gravity)
    gravity_ok = None
    if len(z_after_release) >= 5:
        peak_z = max(z_after_release)
        last_z = z_after_release[-1]
        gravity_ok = (peak_z > 0.05) and (last_z < peak_z)
    elif released:
        gravity_ok = True  # too short post-release to falsify

    return {
        "reward": ep_reward,
        "length": ep_length,
        "success": bool(info_final.get("success", 0)),
        "released": ball_released_at_all,
        "release_step": release_step,
        "release_ball_speed": release_ball_speed,
        "final_distance": info_final.get("final_distance_to_target", float("nan")),
        "min_distance": info_final.get("min_distance_to_target", float("nan")),
        "max_abs_joint_velocity": info_final.get("max_abs_joint_velocity", float("nan")),
        "max_action_abs": max_action_abs,
        "max_joint_angle": max_joint_angle,
        "max_joint_vel_obs": max_joint_vel_obs,
        "gravity_ok": gravity_ok,
        "termination_type": (
            "success" if info_final.get("termination_success")
            else "ground_miss" if info_final.get("termination_ground_miss")
            else "timeout_no_release" if info_final.get("termination_timeout_no_release")
            else "timeout_after_release"
        ),
        "target_pos": np.array([
            info_final.get("target_x", float("nan")),
            info_final.get("target_y", float("nan")),
            info_final.get("target_z", float("nan")),
        ]),
        "target_radius": info_final.get("target_radius", float("nan")),
    }


def run_core_performance(model, env_cfg: dict, n_episodes: int, seed: int):
    """Suite 1: standard random-target performance over N episodes."""
    print(f"\n{'='*60}")
    print(f"  SUITE 1: Core Performance ({n_episodes} episodes, random targets)")
    print(f"{'='*60}")

    env = ArmThrowEnv(env_cfg)
    episodes = []
    for i in range(n_episodes):
        ep = run_episode(model, env, seed=seed + i)
        episodes.append(ep)
        status = "SUCCESS" if ep["success"] else ep["termination_type"]
        print(f"  ep {i+1:3d}  reward={ep['reward']:7.3f}  dist={ep['final_distance']:.3f}  "
              f"min_dist={ep['min_distance']:.3f}  {status}")
    env.close()

    rewards = [e["reward"] for e in episodes]
    successes = [e["success"] for e in episodes]
    final_dists = [e["final_distance"] for e in episodes if not math.isnan(e["final_distance"])]
    min_dists = [e["min_distance"] for e in episodes if not math.isnan(e["min_distance"])]
    lengths = [e["length"] for e in episodes]
    release_rates = [e["released"] for e in episodes]

    terminations: dict[str, int] = {}
    for e in episodes:
        terminations[e["termination_type"]] = terminations.get(e["termination_type"], 0) + 1

    print(f"\n  --- Results ---")
    print(f"  Success rate:          {np.mean(successes)*100:.1f}%  ({sum(successes)}/{n_episodes})")
    print(f"  Release rate:          {np.mean(release_rates)*100:.1f}%")
    print(f"  Mean reward:           {np.mean(rewards):.3f}  (std={np.std(rewards):.3f})")
    print(f"  Mean episode length:   {np.mean(lengths):.1f}")
    print(f"  Mean final distance:   {np.mean(final_dists):.3f}")
    print(f"  Mean min distance:     {np.mean(min_dists):.3f}")
    print(f"  Best min distance:     {np.min(min_dists):.3f}")
    print(f"  Termination breakdown: {terminations}")

    return {
        "success_rate": float(np.mean(successes)),
        "release_rate": float(np.mean(release_rates)),
        "mean_reward": float(np.mean(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
        "mean_final_dist": float(np.mean(final_dists)),
        "mean_min_dist": float(np.mean(min_dists)),
        "episodes": episodes,
    }


def run_sanity_checks(episodes: list, env_cfg: dict, model, seed: int):
    """Suite 2: per-episode and aggregate sanity checks."""
    print(f"\n{'='*60}")
    print(f"  SUITE 2: Sanity Checks")
    print(f"{'='*60}")

    joint_vel_limit = env_cfg["joint_velocity_limit"]
    n = len(episodes)
    results = {}

    # Check 1: Ball release
    release_count = sum(e["released"] for e in episodes)
    release_ok = release_count > 0
    print(f"\n  [{'PASS' if release_ok else 'FAIL'}] Ball release: "
          f"{release_count}/{n} episodes released the ball")
    results["ball_releases"] = release_ok

    # Check 2: Release timing
    release_steps = [e["release_step"] for e in episodes
                     if e["released"] and e["release_step"] is not None]
    if release_steps:
        mean_release = np.mean(release_steps)
        max_steps = env_cfg["max_steps"]
        early_release_ok = float(mean_release) < max_steps * 0.9
        print(f"  [{'PASS' if early_release_ok else 'WARN'}] Release timing: "
              f"mean step {mean_release:.1f} / {max_steps} (should be well before timeout)")
        results["release_timing"] = early_release_ok
    else:
        print(f"  [SKIP] Release timing: no releases recorded")
        results["release_timing"] = None

    # Check 3: Joint velocity limits
    max_vels = [e["max_abs_joint_velocity"] for e in episodes
                if not math.isnan(e["max_abs_joint_velocity"])]
    if max_vels:
        global_max_vel = max(max_vels)
        vel_ok = global_max_vel <= joint_vel_limit + 1e-3
        print(f"  [{'PASS' if vel_ok else 'FAIL'}] Joint velocity limits: "
              f"peak={global_max_vel:.4f} <= limit={joint_vel_limit}")
        results["velocity_limit"] = vel_ok
    else:
        print(f"  [SKIP] Joint velocity limits: no data")
        results["velocity_limit"] = None

    # Check 4: Action bounds
    max_actions = [e["max_action_abs"] for e in episodes]
    global_max_action = max(max_actions)
    action_ok = global_max_action <= 1.0 + 1e-5
    print(f"  [{'PASS' if action_ok else 'FAIL'}] Action bounds: "
          f"max |action| = {global_max_action:.6f} (should be <= 1.0)")
    results["action_bounds"] = action_ok

    # Check 5: Joint angle sanity
    # Values beyond ±4π (two full rotations) suggest unconstrained spinning
    max_angles = [e["max_joint_angle"] for e in episodes]
    global_max_angle = max(max_angles)
    angle_ok = global_max_angle < 4 * math.pi
    print(f"  [{'PASS' if angle_ok else 'WARN'}] Joint angle range: "
          f"max |angle| = {global_max_angle:.3f} rad "
          f"(warn if > {4*math.pi:.2f} rad / 2 full rotations)")
    results["joint_angle_range"] = angle_ok

    #Check 6: Release ball speed
    # A 3-DOF arm with joint_vel_limit=10 rad/s and ~1 m links gives ~10 m/s tip speed.
    # Allowing up to 30 m/s as a generous ceiling; below 0.01 m/s means the arm barely moved.
    release_speeds = [e["release_ball_speed"] for e in episodes
                      if e["released"] and e["release_ball_speed"] is not None
                      and not math.isnan(e["release_ball_speed"])]
    if release_speeds:
        mean_speed = np.mean(release_speeds)
        max_speed = max(release_speeds)
        min_speed = min(release_speeds)
        speed_ok = min_speed > 0.01 and max_speed < 30.0
        print(f"  [{'PASS' if speed_ok else 'FAIL'}] Release ball speed: "
              f"mean={mean_speed:.2f}  min={min_speed:.2f}  max={max_speed:.2f} m/s "
              f"(expected 0.01–30 m/s)")
        results["release_speed"] = speed_ok
    else:
        print(f"  [SKIP] Release ball speed: no releases recorded")
        results["release_speed"] = None

    # Check 7: Gravity / physics 
    gravity_checks = [e["gravity_ok"] for e in episodes if e["gravity_ok"] is not None]
    if gravity_checks:
        gravity_pass = sum(gravity_checks)
        gravity_ok = gravity_pass == len(gravity_checks)
        print(f"  [{'PASS' if gravity_ok else 'FAIL'}] Gravity (ball descends after peak): "
              f"{gravity_pass}/{len(gravity_checks)} episodes")
        results["gravity"] = gravity_ok
    else:
        print(f"  [SKIP] Gravity: ball never released in any episode")
        results["gravity"] = None

    # Check 8: Ball movement probe
    # Run one fresh episode with a fixed center target and verify the ball
    # actually travels forward (x > 0.5 m from the arm base at origin).
    print(f"\n  [INFO] Running single-episode trajectory probe for ball movement check...")
    probe_cfg = dict(env_cfg)
    probe_cfg["target"] = {
        "mode": "fixed",
        "fixed": [2.0, 0.0, 0.5],
        "random": {"x": [2.0, 2.0], "y": [0.0, 0.0], "z": [0.5, 0.5]},
    }
    probe_env = ArmThrowEnv(probe_cfg)
    if hasattr(model, "reset_episode"):
        model.reset_episode()
    obs, _ = probe_env.reset(seed=seed)
    ball_positions_x = []
    released_flag = False
    for _ in range(env_cfg["max_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = probe_env.step(action)
        if probe_env.ball_id is not None:
            bp, _ = p.getBasePositionAndOrientation(
                probe_env.ball_id, physicsClientId=probe_env.client
            )
            ball_positions_x.append(bp[0])
        if info.get("released"):
            released_flag = True
        if terminated or truncated:
            break
    probe_env.close()

    if released_flag and ball_positions_x:
        max_x_reach = max(ball_positions_x)
        ball_moves_ok = max_x_reach > 0.5
        print(f"  [{'PASS' if ball_moves_ok else 'FAIL'}] Ball movement: "
              f"max x reached = {max_x_reach:.2f} m (should be > 0.5 m from arm base)")
        results["ball_movement"] = ball_moves_ok
    else:
        print(f"  [SKIP] Ball movement: ball not released in probe episode")
        results["ball_movement"] = None

    return results


def run_progressive_difficulty(model, env_cfg: dict, n_episodes: int, seed: int):
    """
    Suite 3: test success rate as target distance / difficulty increases.
    Each tier uses a fixed center target to isolate distance as the variable.
    The observation_mode is inherited from env_cfg so Physics gets full_throw_state.
    """
    print(f"\n{'='*60}")
    print(f"  SUITE 3: Progressive Difficulty ({n_episodes} eps per tier)")
    print(f"{'='*60}")

    tiers = [
        ("Easy    (x=2.0, y=0.0, z=0.5 — fixed center)",
         {"mode": "fixed", "fixed": [2.0, 0.0, 0.5],
          "random": {"x": [2.0, 2.0], "y": [0.0, 0.0], "z": [0.5, 0.5]}}),
        ("Medium  (x=[1.8,2.2], y=[-0.2,0.2], z=[0.4,0.6] — trained range)",
         {"mode": "random", "fixed": [2.0, 0.0, 0.5],
          "random": {"x": [1.8, 2.2], "y": [-0.2, 0.2], "z": [0.4, 0.6]}}),
        ("Hard    (x=[2.2,2.8], y=[-0.5,0.5], z=[0.5,0.7] — wider range)",
         {"mode": "random", "fixed": [2.5, 0.0, 0.5],
          "random": {"x": [2.2, 2.8], "y": [-0.5, 0.5], "z": [0.5, 0.7]}}),
        ("Extreme (x=3.0, y=0.0, z=0.5 — far fixed target)",
         {"mode": "fixed", "fixed": [3.0, 0.0, 0.5],
          "random": {"x": [3.0, 3.0], "y": [0.0, 0.0], "z": [0.5, 0.5]}}),
        ("Aerial  (x=2.0, y=0.0, z=1.5 — high fixed target)",
         {"mode": "fixed", "fixed": [2.0, 0.0, 1.5],
          "random": {"x": [2.0, 2.0], "y": [0.0, 0.0], "z": [1.5, 1.5]}}),
    ]

    tier_results = []
    for label, target_cfg in tiers:
        tier_env_cfg = dict(env_cfg)
        tier_env_cfg["target"] = target_cfg
        env = ArmThrowEnv(tier_env_cfg)
        successes = []
        final_dists = []
        for i in range(n_episodes):
            ep = run_episode(model, env, seed=seed + i)
            successes.append(ep["success"])
            final_dists.append(ep["final_distance"])
        env.close()

        sr = np.mean(successes)
        md = np.nanmean(final_dists)
        tier_results.append({"label": label, "success_rate": float(sr), "mean_final_dist": float(md)})
        print(f"  {label}")
        print(f"    success_rate={sr*100:.1f}%  mean_final_distance={md:.3f}")

    # Soft check: easy success_rate should be >= hard with 5% slack
    easy_sr = tier_results[0]["success_rate"]
    hard_sr = tier_results[2]["success_rate"]
    monotonic_ok = easy_sr >= hard_sr - 0.05
    print(f"\n  [{'PASS' if monotonic_ok else 'WARN'}] Difficulty monotonicity: "
          f"easy={easy_sr*100:.1f}% >= hard={hard_sr*100:.1f}% (5% slack)")

    return tier_results, monotonic_ok


# Report

def print_summary(core, sanity, difficulty, monotonic_ok, algo_name):
    print(f"\n{'='*60}")
    print(f"  SUITE 4: Summary Report  [{algo_name}]")
    print(f"{'='*60}")

    sr = core["success_rate"]
    rr = core["release_rate"]
    performance_ok = sr >= 0.1
    print(f"\n  Core Performance:")
    print(f"    Success rate:        {sr*100:.1f}%  "
          f"{'[OK]' if sr >= 0.5 else '[LOW]' if sr >= 0.1 else '[FAIL]'}")
    print(f"    Release rate:        {rr*100:.1f}%  {'[OK]' if rr >= 0.8 else '[LOW]'}")
    print(f"    Mean reward:         {core['mean_reward']:.3f}")
    print(f"    Mean final distance: {core['mean_final_dist']:.3f} m")

    print(f"\n  Sanity Checks:")
    all_sanity = []
    for k, v in sanity.items():
        if v is None:
            status = "SKIP"
        elif v:
            status = "PASS"
            all_sanity.append(True)
        else:
            status = "FAIL"
            all_sanity.append(False)
        print(f"    {k:<25s} [{status}]")

    sanity_ok = all(all_sanity) if all_sanity else False

    print(f"\n  Progressive Difficulty:")
    for tier in difficulty:
        print(f"    {tier['label'][:50]:<50s} {tier['success_rate']*100:.1f}%")
    print(f"    Monotonicity check:  {'[PASS]' if monotonic_ok else '[WARN]'}")

    overall_ok = performance_ok and sanity_ok
    print(f"\n{'='*60}")
    print(f"  OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    if not performance_ok:
        print(f"  WARNING: Success rate {sr*100:.1f}% is below the minimum 10% bar.")
    if not sanity_ok:
        print(f"  WARNING: One or more sanity checks failed (see above).")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate any ArmThrow model (PPO, A2C, SAC, TD3, DDPG, BC, GAIL, Physics)."
    )
    parser.add_argument(
        "--model", type=str, default="ppo_arm_throw.zip",
        help="Path to model.zip (default: ppo_arm_throw.zip)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to the config YAML used during training. Recommended — used to set "
             "algo name and observation_mode automatically."
    )
    parser.add_argument(
        "--n-episodes", type=int, default=50,
        help="Episodes per test suite (default: 50)"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable PyBullet GUI rendering (slow)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="W&B project name. When set, logs test results to wandb."
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity (team or username). Optional."
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name. Defaults to 'test-<model_basename>'."
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ArmThrow Model Validation")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Config:     {args.config or '(built-in defaults)'}")
    print(f"  Episodes:   {args.n_episodes} per suite")
    print(f"  Seed:       {args.seed}")
    print(f"  Render:     {args.render}")

    print(f"\n  Loading model...")
    try:
        model, algo_name = load_model(args.model, args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  Algorithm:  {algo_name}")

    # Build env config with correct observation_mode for this algo
    env_cfg = _prepare_env_cfg(args.config, render=args.render, algo_name=algo_name)
    env_cfg["visualize_target"] = False

    # Suite 1
    core = run_core_performance(model, env_cfg, args.n_episodes, args.seed)

    # Suite 2
    sanity = run_sanity_checks(core["episodes"], env_cfg, model, args.seed)

    # Suite 3 (fewer episodes — each tier is a fresh env)
    difficulty, monotonic_ok = run_progressive_difficulty(
        model, env_cfg, max(args.n_episodes // 5, 10), args.seed
    )

    # Suite 4
    print_summary(core, sanity, difficulty, monotonic_ok, algo_name)

    # Optional wandb logging
    if args.wandb_project:
        if not WANDB_AVAILABLE:
            print("WARNING: --wandb-project set but wandb is not installed. Skipping.", file=sys.stderr)
        else:
            run_name = args.wandb_run_name or f"test-{Path(args.model).stem}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={"model": args.model, "algo": algo_name,
                        "n_episodes": args.n_episodes, "seed": args.seed},
            )
            payload = build_test_wandb_payload(core, sanity, difficulty)
            payload["test/monotonicity_ok"] = float(monotonic_ok)
            wandb.log(payload)
            wandb.finish()
            print(f"  Logged test results to wandb run '{run_name}' in project '{args.wandb_project}'")


if __name__ == "__main__":
    main()
