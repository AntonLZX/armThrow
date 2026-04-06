import argparse
from pathlib import Path

import imageio
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

import yaml
from config import _coerce_float, _coerce_int
from env import ArmThrowEnv


def load_env_config(path):
    """Load a config YAML and return only the validated env section.

    Works with both PPO (train.py) and imitation (train_imitation.py) configs
    since it only validates the env section, not the algo section.
    """
    with open(path, "r") as f:
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
    return env


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


def rollout_once(env, model, episode_seed, deterministic):
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
        obs, reward, done, truncated, info = env.step(action)
        last_info = info
        total_reward += float(reward)
        step_count += 1
        frames.append(capture_frame(env))

    hold_frames = 12 if float(last_info.get("success", 0.0)) >= 1.0 else 4
    frames.extend([frames[-1]] * hold_frames)
    return frames, last_info, step_count, total_reward


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
        f"target=({info.get('target_x', float('nan'))}, {info.get('target_y', float('nan'))}, {info.get('target_z', float('nan'))})",
        f"target_radius={info.get('target_radius', float('nan'))}",
        f"gif={gif_path}",
        f"png={png_path}",
    ]
    summary_path.write_text("\n".join(str(line) for line in summary_lines) + "\n", encoding="utf-8")
    print(f"Saved {label} GIF     : {gif_path}")
    print(f"Saved {label} PNG     : {png_path}")
    print(f"Saved {label} summary : {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config used to build the env")
    parser.add_argument("--model", required=True, help="Path to model.zip")
    parser.add_argument("--output-dir", default="tmp/success_capture", help="Where to write GIF/PNG")
    parser.add_argument("--seed", type=int, default=42, help="Base evaluation seed")
    parser.add_argument("--max-attempts", type=int, default=50, help="How many episodes to try")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic rollout",
    )
    args = parser.parse_args()

    env_cfg = load_env_config(args.config)
    env_cfg["render"] = False
    env_cfg["visualize_target"] = True

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = ArmThrowEnv(env_cfg)
    model = PPO.load(str(Path(args.model).expanduser().resolve()), env=env, device="auto")

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
                f"attempt={attempt + 1} seed={episode_seed} success={success} "
                f"total_reward={total_reward:.3f} "
                f"final_distance={info.get('final_distance_to_target', float('nan')):.3f} "
                f"steps={step_count}"
            )

            if success:
                # Keep the first success found (or replace with higher-reward success)
                if best_success is None or total_reward > best_success[3]:
                    best_success = (frames, info, step_count, total_reward, episode_seed)
            else:
                # Track the failure with the lowest total reward
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
