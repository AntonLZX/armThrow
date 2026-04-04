import argparse
from pathlib import Path

import imageio
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from config import load_config
from env import ArmThrowEnv


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
    return np.array(rgb_array, dtype=np.uint8)[:, :, :3]


def rollout_once(env, model, episode_seed, deterministic):
    frames = []
    obs, _ = env.reset(seed=episode_seed)
    frames.append(capture_frame(env))

    done = False
    truncated = False
    last_info = {}
    step_count = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, truncated, info = env.step(action)
        last_info = info
        step_count += 1
        frames.append(capture_frame(env))

    hold_frames = 12 if float(last_info.get("success", 0.0)) >= 1.0 else 4
    frames.extend([frames[-1]] * hold_frames)
    return frames, last_info, step_count


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

    cfg = load_config(args.config)
    cfg["env"]["render"] = False
    cfg["env"]["visualize_target"] = True

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = ArmThrowEnv(cfg["env"])
    model = PPO.load(str(Path(args.model).expanduser().resolve()), env=env, device="auto")

    try:
        for attempt in range(args.max_attempts):
            episode_seed = args.seed + attempt
            frames, info, step_count = rollout_once(
                env=env,
                model=model,
                episode_seed=episode_seed,
                deterministic=not args.stochastic,
            )
            success = float(info.get("success", 0.0)) >= 1.0
            print(
                f"attempt={attempt + 1} seed={episode_seed} success={success} "
                f"final_distance={info.get('final_distance_to_target', float('nan')):.3f} "
                f"steps={step_count}"
            )
            if not success:
                continue

            gif_path = output_dir / f"success_seed{episode_seed}.gif"
            png_path = output_dir / f"success_seed{episode_seed}_final.png"
            summary_path = output_dir / f"success_seed{episode_seed}_summary.txt"

            imageio.mimsave(str(gif_path), frames, fps=30)
            imageio.imwrite(str(png_path), frames[-1])

            summary_lines = [
                f"seed={episode_seed}",
                f"steps={step_count}",
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
            print(f"Saved success GIF to: {gif_path}")
            print(f"Saved success PNG to: {png_path}")
            print(f"Saved success summary to: {summary_path}")
            return 0
    finally:
        env.close()

    print("No successful rollout found within the attempt budget.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
