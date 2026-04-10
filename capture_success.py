"""
capture_success.py — Capture GIF + PNG proof of a successful and failed rollout.

Supports: PPO, A2C, SAC, TD3, DDPG (all SB3 algorithms), BC and GAIL
(saved as PPO-compatible zips), and the scripted PhysicsBaseline.

Usage:
    python capture_success.py \\
        --config runs/<run_dir>/config.yaml \\
        --model runs/<run_dir>/model.zip \\
        --output-dir tmp/success_capture

    # If deterministic rollout misses, try stochastic actions:
    python capture_success.py \\
        --config runs/<run_dir>/config.yaml \\
        --model runs/<run_dir>/model.zip \\
        --output-dir tmp/success_capture \\
        --stochastic

Outputs per episode captured:
    <label>_seed<N>.gif           — animated rollout
    <label>_seed<N>_final.png     — final frame
    <label>_seed<N>_summary.txt   — text record with key metrics

Exit code: 0 if a success was found, 1 otherwise.
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import pybullet as p
import yaml
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from config import _coerce_float, _coerce_int
from env import ArmThrowEnv
from physics_baseline import PhysicsBaseline


# ---------------------------------------------------------------------------
# Model loading  (same logic as test.py — kept self-contained)
# ---------------------------------------------------------------------------

_BC_GAIL_ALIASES = {"BC", "GAIL"}

_SB3_ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}

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
        print(f"  Detected: PhysicsBaseline")
        return PhysicsBaseline.load(str(path)), "Physics"

    # 2. Config-declared algo
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        algo_name = str(raw.get("algo", {}).get("name", "PPO")).upper()
        load_name = "PPO" if algo_name in _BC_GAIL_ALIASES else algo_name
        cls = _SB3_ALGO_MAP.get(load_name)
        if cls is None:
            print(f"  Warning: unknown algo '{algo_name}' in config — falling back to auto-detect")
        else:
            print(f"  Detected from config: {algo_name}" +
                  (f" (loaded as PPO)" if algo_name in _BC_GAIL_ALIASES else ""))
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


# ---------------------------------------------------------------------------
# Env config loading
# ---------------------------------------------------------------------------

def load_env_config(config_path, algo_name):
    """
    Load and validate the env section from a config YAML.
    Works with all training configs (PPO, Physics, BC, GAIL, etc.)
    since only the env section is validated here.
    Forces observation_mode=full_throw_state for PhysicsBaseline.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env = cfg["env"]

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
    env["observation_mode"] = str(env.get("observation_mode", "arm_target_release"))
    env["release_success_bonus"] = _coerce_float(
        env.get("release_success_bonus", 1.0), "env.release_success_bonus"
    )

    if algo_name == "Physics" and env["observation_mode"] != "full_throw_state":
        print(f"  [info] Forcing observation_mode=full_throw_state for PhysicsBaseline "
              f"(was {env['observation_mode']!r})")
        env["observation_mode"] = "full_throw_state"

    env["render"] = False
    env["visualize_target"] = True
    return env


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def capture_frame(env, width=640, height=480):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.5, 0.0, 0.5],
        distance=3.0,
        yaw=45,
        pitch=-30,
        roll=0,
        upAxisIndex=2,
        physicsClientId=env.client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=100,
        physicsClientId=env.client,
    )
    _, _, rgb_array, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        physicsClientId=env.client,
    )
    return np.array(rgb_array, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_once(env, model, episode_seed, deterministic):
    """
    Roll out one episode and return (frames, info, step_count, total_reward).
    Handles PhysicsBaseline (reset_episode) and all SB3 models uniformly.
    """
    # PhysicsBaseline is stateful — reset its internal step counter each episode
    if hasattr(model, "reset_episode"):
        model.reset_episode()

    frames = []
    obs, _ = env.reset(seed=episode_seed)
    frames.append(capture_frame(env))

    done = False
    truncated = False
    last_info = {}
    step_count = 0
    total_reward = 0.0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, last_info = env.step(action)
        last_info = last_info
        total_reward += float(reward)
        step_count += 1
        frames.append(capture_frame(env))

    hold_frames = 12 if float(last_info.get("success", 0.0)) >= 1.0 else 4
    frames.extend([frames[-1]] * hold_frames)
    return frames, last_info, step_count, total_reward


# ---------------------------------------------------------------------------
# Save episode
# ---------------------------------------------------------------------------

def save_episode(frames, info, step_count, total_reward, seed, label, output_dir):
    gif_path = output_dir / f"{label}_seed{seed}.gif"
    png_path = output_dir / f"{label}_seed{seed}_final.png"
    summary_path = output_dir / f"{label}_seed{seed}_summary.txt"

    imageio.mimsave(str(gif_path), frames, fps=30)
    imageio.imwrite(str(png_path), frames[-1])

    summary_lines = [
        f"label={label}",
        f"seed={seed}",
        f"steps={step_count}",
        f"total_reward={total_reward:.4f}",
        f"success={info.get('success', float('nan'))}",
        f"final_distance_to_target={info.get('final_distance_to_target', float('nan'))}",
        f"min_distance_to_target={info.get('min_distance_to_target', float('nan'))}",
        f"release_step={info.get('release_step', float('nan'))}",
        f"release_ball_speed={info.get('release_ball_speed', float('nan'))}",
        f"target=({info.get('target_x', float('nan'))}, "
        f"{info.get('target_y', float('nan'))}, "
        f"{info.get('target_z', float('nan'))})",
        f"target_radius={info.get('target_radius', float('nan'))}",
        f"gif={gif_path}",
        f"png={png_path}",
    ]
    summary_path.write_text(
        "\n".join(str(line) for line in summary_lines) + "\n", encoding="utf-8"
    )
    print(f"Saved {label} GIF     : {gif_path}")
    print(f"Saved {label} PNG     : {png_path}")
    print(f"Saved {label} summary : {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Capture GIF/PNG proof of success and failure for any ArmThrow model."
    )
    parser.add_argument(
        "--config", required=True,
        help="Config YAML used to build the env (required)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to model.zip"
    )
    parser.add_argument(
        "--output-dir", default="tmp/success_capture",
        help="Directory to write GIF/PNG/summary (default: tmp/success_capture)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base evaluation seed (default: 42)"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=50,
        help="Episode budget to search for a success (default: 50)"
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic policy actions instead of deterministic rollout"
    )
    args = parser.parse_args()

    print(f"\n  Loading model from {args.model} ...")
    model, algo_name = load_model(args.model, args.config)
    print(f"  Algorithm: {algo_name}")

    env_cfg = load_env_config(args.config, algo_name)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = ArmThrowEnv(env_cfg)

    best_success = None   # (frames, info, step_count, total_reward, seed)
    worst_failure = None  # (frames, info, step_count, total_reward, seed)

    try:
        for attempt in range(args.max_attempts):
            episode_seed = args.seed + attempt
            frames, info, step_count, total_reward = rollout_once(
                env=env,
                model=model,
                episode_seed=episode_seed,
                deterministic=not args.stochastic,
            )
            success = float(info.get("success", 0.0)) >= 1.0
            print(
                f"attempt={attempt + 1}  seed={episode_seed}  success={success}  "
                f"total_reward={total_reward:.3f}  "
                f"final_distance={info.get('final_distance_to_target', float('nan')):.3f}  "
                f"steps={step_count}"
            )

            if success:
                if best_success is None or total_reward > best_success[3]:
                    best_success = (frames, info, step_count, total_reward, episode_seed)
            else:
                if worst_failure is None or total_reward < worst_failure[3]:
                    worst_failure = (frames, info, step_count, total_reward, episode_seed)

    finally:
        env.close()

    if best_success is not None:
        frames, info, step_count, total_reward, seed = best_success
        save_episode(frames, info, step_count, total_reward, seed, "success", output_dir)
    else:
        print("No successful rollout found within the attempt budget.")

    if worst_failure is not None:
        frames, info, step_count, total_reward, seed = worst_failure
        save_episode(frames, info, step_count, total_reward, seed, "worst_failure", output_dir)
    else:
        print("No failed rollout found (all episodes succeeded).")

    return 0 if best_success is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
