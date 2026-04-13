from pathlib import Path

import imageio
import numpy as np
import pybullet as p
from stable_baselines3.common.callbacks import BaseCallback

from env import ArmThrowEnv
from metrics import _finite_mean, _finite_std, evaluate_episodes

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WandbCallback = None
    WANDB_AVAILABLE = False


class EpisodeRecorderCallback(BaseCallback):
    """Records rollout GIFs every N completed episodes."""

    def __init__(self, save_path=None, env_cfg=None, every_n_episodes=200, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.env_cfg = env_cfg
        self.every_n_episodes = max(1, int(every_n_episodes))
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                self.episode_count += 1
                if self.episode_count % self.every_n_episodes == 0:
                    print(f"Recording episode {self.episode_count} to GIF...")
                    self.record_episode(self.episode_count)
        return True

    def record_episode(self, episode_number):
        if self.env_cfg is None:
            print("Warning: env_cfg not provided. Skipping GIF recording.")
            return

        rec_cfg = self.env_cfg.copy()
        rec_cfg["render"] = False
        rec_env = ArmThrowEnv(rec_cfg)

        frames = []
        obs, _ = rec_env.reset()
        done = False
        truncated = False
        last_info = {}

        while not (done or truncated):
            frame = self.capture_frame(rec_env)
            if frame is not None:
                frames.append(frame)

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, truncated, last_info = rec_env.step(action)

        final_frame = self.capture_frame(rec_env)
        if final_frame is not None:
            frames.append(final_frame)
            hold_count = 12 if float(last_info.get("success", 0.0)) >= 1.0 else 4
            frames.extend([final_frame] * hold_count)

        rec_env.close()

        if frames and self.save_path:
            gif_path = Path(self.save_path) / f"episode_{episode_number:05d}.gif"
            try:
                imageio.mimsave(str(gif_path), frames, fps=30)
                print(f"Saved episode {episode_number} GIF ({len(frames)} frames) to {gif_path}")
            except Exception as e:
                print(f"Error saving GIF: {e}")
        else:
            print(f"Warning: No frames captured (frames={len(frames)}, save_path={self.save_path})")

    def capture_frame(self, env):
        try:
            if not hasattr(env, "client") or env.client is None:
                return None

            width = 640
            height = 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0, 0.5],
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
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None


class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=20, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = evaluate_episodes(self.model, self.eval_env, self.n_eval_episodes)
        wandb.log({**metrics, "global_timestep": int(self.num_timesteps)}, step=self.num_timesteps)
        return True


class WandbEpisodeCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                ep = info["episode"]
                payload = {
                    "episode/reward": float(ep["r"]),
                    "episode/length": float(ep["l"]),
                    "episode/success": float(info.get("success", 0.0)),
                    "episode/released": float(info.get("released", 0.0)),
                    "episode/final_distance": float(info.get("final_distance_to_target", np.nan)),
                    "episode/release_step": float(info.get("release_step", np.nan)),
                    "episode/release_ball_speed": float(info.get("release_ball_speed", np.nan)),
                    "global_timestep": int(self.num_timesteps),
                }
                wandb.log(payload, step=self.num_timesteps)
        return True


class WandbTrainStatsCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        values = getattr(self.logger, "name_to_value", {})
        payload = {"global_timestep": int(self.num_timesteps)}

        for key in [
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/entropy_loss",
            "train/loss",
            "time/fps",
        ]:
            if key in values and values[key] is not None:
                try:
                    payload[key] = float(values[key])
                except (TypeError, ValueError):
                    pass

        if len(payload) > 1:
            wandb.log(payload, step=self.num_timesteps)

        return True
