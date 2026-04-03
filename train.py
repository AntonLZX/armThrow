import os
import time
import uuid
import yaml
import shutil
import random
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pybullet as p
import pybullet_data
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

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
        """Records one episode and saves as GIF."""
        
        if self.env_cfg is None:
            print("Warning: env_cfg not provided. Skipping GIF recording.")
            return

        # Create a fresh environment for recording (non-rendering)
        from pathlib import Path
        rec_cfg = self.env_cfg.copy()
        rec_cfg["render"] = False
        rec_env = ArmThrowEnv(rec_cfg)

        frames = []
        obs, _ = rec_env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            # Capture frame
            frame = self.capture_frame(rec_env)
            if frame is not None:
                frames.append(frame)

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = rec_env.step(action)

        rec_env.close()

        # Save GIF
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
        """Capture a frame from PyBullet."""
        try:
            if not hasattr(env, 'client') or env.client is None:
                return None
                
            width = 640
            height = 480
            
            # Get camera view matrix (looking at the scene)
            camera_distance = 3.0
            camera_yaw = 45
            camera_pitch = -30
            
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.5, 0, 0.5],
                distance=camera_distance,
                yaw=camera_yaw,
                pitch=camera_pitch,
                roll=0,
                upAxisIndex=2,
                physicsClientId=env.client
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width / height,
                nearVal=0.1,
                farVal=100,
                physicsClientId=env.client
            )
            
            _, _, rgb_array, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=env.client
            )
            
            # Convert to uint8 and return
            return np.array(rgb_array, dtype=np.uint8)[:, :, :3]
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


class WandbEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=20, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        rewards = []
        lengths = []
        final_distances = []
        successes = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            last_info = {}

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                ep_reward += reward
                ep_len += 1
                last_info = info

            rewards.append(ep_reward)
            lengths.append(ep_len)
            final_distances.append(last_info.get("landing_xy_dist", np.nan))
            successes.append(last_info.get("success", 0.0))

        payload = {
            "valid/mean_reward": float(np.mean(rewards)),
            "valid/std_reward": float(np.std(rewards)),
            "valid/mean_ep_length": float(np.mean(lengths)),
            "valid/success_rate": float(np.mean(successes)),
            "valid/mean_final_distance": float(np.nanmean(final_distances)),
            "global_timestep": int(self.num_timesteps),
        }

        wandb.log(payload, step=self.num_timesteps)
        return True

class WandbEpisodeCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done and "episode" in info:
                ep = info["episode"]
                wandb.log({
                    "episode/reward": float(ep["r"]),
                    "episode/length": float(ep["l"]),
                    "global_timestep": int(self.num_timesteps),
                }, step=self.num_timesteps)
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class ArmThrowEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.render_enabled = cfg["render"]
        self.arm_urdf = cfg["arm_urdf"]
        self.max_steps = cfg["max_steps"]
        self.end_effector_link_index = cfg["end_effector_link_index"]
        self.torque_scale = cfg["torque_scale"]
        self.release_success_bonus = float(cfg.get("release_success_bonus", 1.0))

        self.client = p.connect(p.GUI if self.render_enabled else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        # Action space: [angular_accel_1, angular_accel_2, angular_accel_3, release_trigger]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.dt = 1.0 / 60.0
        self.n_joints = 3

        self.robot_id = None
        self.ball_id = None
        self.cid = None
        self.released = False
        self.step_count = 0
        self.target_pos = np.array([2.0, 0.0, 0.5], dtype=np.float32)
        self.joint_velocities = np.zeros(self.n_joints, dtype=np.float32)
        self.target_radius = 0.1  # Radius around target for early termination

    def _sample_target(self):
        target_cfg = self.cfg["target"]
        if target_cfg["mode"] == "fixed":
            return np.array(target_cfg["fixed"], dtype=np.float32)

        xr = target_cfg["random"]["x"]
        yr = target_cfg["random"]["y"]
        zr = target_cfg["random"]["z"]
        return np.array([
            self.np_random.uniform(xr[0], xr[1]),
            self.np_random.uniform(yr[0], yr[1]),
            self.np_random.uniform(zr[0], zr[1]),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)

        self.target_pos = self._sample_target()

        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.robot_id = p.loadURDF(
            self.arm_urdf,
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client,
        )

        self.ball_id = p.loadURDF(
            "sphere_small.urdf",
            [0.1, 0.0, 1.0],
            physicsClientId=self.client,
        )

        for j in range(self.n_joints):
            p.setJointMotorControl2(
                self.robot_id, j, p.VELOCITY_CONTROL, force=0,
                physicsClientId=self.client
            )
            p.resetJointState(
                self.robot_id,
                j,
                targetValue=float(self.np_random.uniform(-0.2, 0.2)),
                targetVelocity=float(self.np_random.uniform(-0.05, 0.05)),
                physicsClientId=self.client,
            )

        self.cid = p.createConstraint(
            self.robot_id,
            self.end_effector_link_index,
            self.ball_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0, 0],
            physicsClientId=self.client,
        )

        self.released = False
        self.step_count = 0
        self.joint_velocities = np.zeros(self.n_joints, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        released_this_step = False

        # Convert action (angular acceleration) to velocity via integration
        # a_desired = action[:3] * accel_scale
        # v_new = v_old + a * dt
        accel_scale = self.torque_scale  # Reuse torque scale for acceleration scale
        self.joint_velocities += action[:3] * accel_scale * self.dt
        
        # Apply velocity control to joints (only before release)
        if not self.released:
            for i in range(self.n_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=float(self.joint_velocities[i]),
                    force=float(self.torque_scale),
                    physicsClientId=self.client,
                )
        else:
            # After release, freeze all joint movements
            self.joint_velocities = np.zeros(self.n_joints, dtype=np.float32)
            for i in range(self.n_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=self.torque_scale,
                    physicsClientId=self.client,
                )

        # Handle release
        if action[3] > 0.5 and not self.released:
            p.removeConstraint(self.cid, physicsClientId=self.client)
            self.cid = None
            self.released = True
            released_this_step = True

        p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1

        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_pos = np.array(ball_pos, dtype=np.float32)

        dist_to_target = np.linalg.norm(ball_pos - self.target_pos)
        
        # Reward logic:
        # - Before release: control cost only
        # - In air (released & z >= 0.05): accumulate distance reward
        # - Landed (z < 0.05): landing bonus only
        reward = 0.0
        
        if not self.released:
            # Before release: penalize control effort
            reward -= 0.0005 * float(np.sum(np.square(action[:3])))
        elif ball_pos[2] >= 0.05:
            # Ball in air: accumulate distance-based reward
            reward = float(np.exp(-1.5 * dist_to_target))
        # else: Ball landed, reward = 0 until landing bonus below

        if released_this_step:
            reward += self.release_success_bonus

        terminated = False
        truncated = False

        # Landing termination: ball hits ground after release
        if self.released and ball_pos[2] < 0.05:
            terminated = True

        # Early termination: ball reaches target in air (within radius)
        if self.released and dist_to_target < self.target_radius and ball_pos[2] >= 0.05:
            terminated = True
            reward += 10.0  # Bonus for hitting target while in air

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "distance_to_target": float(dist_to_target),
            "released": self.released,
            "target_x": float(self.target_pos[0]),
            "target_y": float(self.target_pos[1]),
            "target_z": float(self.target_pos[2]),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        joint_states = p.getJointStates(
            self.robot_id, list(range(self.n_joints)), physicsClientId=self.client
        )
        angles = np.array([s[0] for s in joint_states], dtype=np.float32)
        vels = np.array([s[1] for s in joint_states], dtype=np.float32)
        return np.concatenate([angles, vels, self.target_pos]).astype(np.float32)

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(base_dir: str, run_name: str):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(base_dir) / f"{stamp}_{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(config_path="configs/base.yaml", render=None):
    cfg = load_config(config_path)
    # Override render from command line if provided
    if render is not None:
        cfg["env"]["render"] = render
    set_seed(cfg["seed"])

    run_dir = make_run_dir(cfg["logging"]["save_dir"], cfg["run_name"])
    shutil.copy(config_path, run_dir / "config.yaml")

    env = ArmThrowEnv(cfg["env"])
    env = Monitor(env, filename=str(run_dir / "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg["algo"]["learning_rate"],
        n_steps=cfg["algo"]["n_steps"],
        batch_size=cfg["algo"]["batch_size"],
        n_epochs=cfg["algo"]["n_epochs"],
        gamma=cfg["algo"]["gamma"],
        gae_lambda=cfg["algo"]["gae_lambda"],
        clip_range=cfg["algo"]["clip_range"],
        ent_coef=cfg["algo"]["ent_coef"],
        verbose=1,
        device="auto",
        seed=cfg["seed"],
        tensorboard_log=str(run_dir / "tb"),
    )
    print(f"Training device selected: {model.device}")

    logger = configure(str(run_dir), ["stdout", "csv"])
    model.set_logger(logger)

    callbacks = []

    gif_every_n_episodes = cfg.get("logging", {}).get("gif_every_n_episodes", 10)
    callbacks.append(
        EpisodeRecorderCallback(
            save_path=run_dir,
            env_cfg=cfg["env"],
            every_n_episodes=gif_every_n_episodes,
            verbose=1,
        )
    )

    if cfg["logging"]["use_wandb"]:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is enabled in config but not installed.")

        # set up eval env (always non-rendering to avoid multiple GUI connections)
        eval_cfg = cfg["env"].copy()
        eval_cfg["render"] = False
        eval_env = ArmThrowEnv(eval_cfg)
        # seed means it is reproducable for all runs
        eval_env.reset(seed=123)
        
        wandb.init(
            project=cfg["logging"]["project"],
            entity=cfg["logging"]["entity"],
            name=cfg["logging"]["name"],
            config=cfg,
            tags=cfg["logging"]["tags"],
            notes=cfg["logging"]["notes"],
            sync_tensorboard=cfg["logging"]["sync_tensorboard"],
            dir=str(run_dir),
        )
        callbacks.extend([
            WandbCallback(
                model_save_path=str(run_dir / "wandb_models"),
                model_save_freq=0,
                verbose=1,
            ),
            WandbEpisodeCallback(),
            WandbTrainStatsCallback(log_freq=1000),
            WandbEvalCallback(eval_env=eval_env, eval_freq=5000, n_eval_episodes=20),
        ])
        

    model.learn(
        total_timesteps=cfg["algo"]["total_timesteps"],
        callback=CallbackList(callbacks) if callbacks else None,
    )

    if cfg["logging"]["save_model"]:
        model.save(str(run_dir / "model"))

    env.close()

    if cfg["logging"]["use_wandb"] and WANDB_AVAILABLE:
        wandb.finish()

    print(f"Done. Run saved to: {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering during training")
    args = parser.parse_args()

    render_override = None
    if args.render:
        render_override = True
    elif args.no_render:
        render_override = False
    
    main(args.config, render=render_override)