import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pybullet as p
import pybullet_data


class ArmThrowEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.render_enabled = cfg["render"]
        self.arm_urdf = cfg["arm_urdf"]
        self.max_steps = cfg["max_steps"]
        self.end_effector_link_index = cfg["end_effector_link_index"]
        self.accel_scale = float(cfg["accel_scale"])
        self.motor_force_limit = float(cfg["motor_force_limit"])
        self.joint_velocity_limit = float(cfg.get("joint_velocity_limit", 10.0))
        self.release_success_bonus = float(cfg.get("release_success_bonus", 1.0))
        self.reward_mode = cfg.get("reward_mode", "distance_progress")
        self.target_radius = float(cfg.get("target_radius", 0.1))
        self.observation_mode = cfg.get("observation_mode", "arm_target_release")
        if self.reward_mode not in {"absolute_distance", "distance_progress"}:
            raise ValueError(
                f"Unsupported reward_mode={self.reward_mode!r}; expected 'absolute_distance' or 'distance_progress'"
            )
        if self.observation_mode not in {"arm_target_only", "arm_target_release", "full_throw_state"}:
            raise ValueError(
                f"Unsupported observation_mode={self.observation_mode!r}; expected 'arm_target_only', "
                "'arm_target_release', or 'full_throw_state'"
            )

        self.client = p.connect(p.GUI if self.render_enabled else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = {
            "arm_target_only": 9,
            "arm_target_release": 10,
            "full_throw_state": 16,
        }[self.observation_mode]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.dt = 1.0 / 60.0
        self.n_joints = 3

        self.robot_id = None
        self.ball_id = None
        self.cid = None
        self.released = False
        self.step_count = 0
        self.target_pos = np.array([2.0, 0.0, 0.5], dtype=np.float32)
        self.joint_velocities = np.zeros(self.n_joints, dtype=np.float32)
        self.prev_dist_to_target = None
        self.best_dist_to_target = None
        self.release_step = None
        self.release_distance_to_target = None
        self.release_ball_speed = None
        self.termination_type = None
        self.pre_release_penalty_sum = 0.0
        self.shaping_reward_sum = 0.0
        self.release_bonus_sum = 0.0
        self.success_bonus_sum = 0.0
        self.failure_penalty_sum = 0.0
        self.action_norm_sum = 0.0
        self.cumulative_abs_joint_velocity = 0.0
        self.max_abs_joint_velocity = 0.0

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
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        initial_dist = float(np.linalg.norm(np.array(ball_pos, dtype=np.float32) - self.target_pos))
        self.prev_dist_to_target = initial_dist
        self.best_dist_to_target = initial_dist
        self.release_step = None
        self.release_distance_to_target = None
        self.release_ball_speed = None
        self.termination_type = None
        self.pre_release_penalty_sum = 0.0
        self.shaping_reward_sum = 0.0
        self.release_bonus_sum = 0.0
        self.success_bonus_sum = 0.0
        self.failure_penalty_sum = 0.0
        self.action_norm_sum = 0.0
        self.cumulative_abs_joint_velocity = 0.0
        self.max_abs_joint_velocity = 0.0
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        released_this_step = False
        success = False
        pre_release_penalty = 0.0
        shaping_component = 0.0
        release_bonus_component = 0.0
        success_bonus_component = 0.0
        failure_penalty_component = 0.0
        action_norm = float(np.linalg.norm(action[:3]))

        self.joint_velocities += action[:3] * self.accel_scale * self.dt
        self.joint_velocities = np.clip(
            self.joint_velocities,
            -self.joint_velocity_limit,
            self.joint_velocity_limit,
        ).astype(np.float32)

        if not self.released:
            for i in range(self.n_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=float(self.joint_velocities[i]),
                    force=float(self.motor_force_limit),
                    physicsClientId=self.client,
                )
        else:
            self.joint_velocities = np.zeros(self.n_joints, dtype=np.float32)
            for i in range(self.n_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=float(self.motor_force_limit),
                    physicsClientId=self.client,
                )

        self.action_norm_sum += action_norm
        current_abs_joint_velocity = float(np.max(np.abs(self.joint_velocities)))
        self.max_abs_joint_velocity = max(self.max_abs_joint_velocity, current_abs_joint_velocity)
        self.cumulative_abs_joint_velocity += float(np.mean(np.abs(self.joint_velocities)))

        if action[3] > 0.5 and not self.released:
            p.removeConstraint(self.cid, physicsClientId=self.client)
            self.cid = None
            self.released = True
            released_this_step = True

        p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1

        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_pos = np.array(ball_pos, dtype=np.float32)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
        ball_vel = np.array(ball_vel, dtype=np.float32)
        ball_speed = float(np.linalg.norm(ball_vel))

        dist_to_target = np.linalg.norm(ball_pos - self.target_pos)

        prev_dist_to_target = (
            float(self.prev_dist_to_target)
            if self.prev_dist_to_target is not None
            else float(dist_to_target)
        )
        phi_t = float(np.exp(-1.5 * dist_to_target))
        phi_prev = float(np.exp(-1.5 * prev_dist_to_target))

        reward = 0.0

        if self.reward_mode == "absolute_distance":
            if not self.released:
                pre_release_penalty -= 0.0005 * float(np.sum(np.square(action[:3])))
            elif ball_pos[2] >= 0.05:
                shaping_component += phi_t
        else:
            if not self.released:
                pre_release_penalty -= 0.0005 * float(np.sum(np.square(action[:3])))
                pre_release_penalty -= 0.001
            elif ball_pos[2] >= 0.05:
                shaping_component += 2.0 * (phi_t - phi_prev)

        if released_this_step:
            release_bonus_component += self.release_success_bonus

        terminated = False
        truncated = False

        if self.released and ball_pos[2] < 0.05:
            terminated = True
            self.termination_type = "ground_miss"

        if self.released and dist_to_target < self.target_radius and ball_pos[2] >= 0.05:
            terminated = True
            success = True
            success_bonus_component += 10.0
            self.termination_type = "success"

        if self.step_count >= self.max_steps:
            truncated = True
            if not success and not terminated:
                self.termination_type = "timeout_after_release" if self.released else "timeout_no_release"

        if self.reward_mode == "distance_progress" and (terminated or truncated) and not success:
            failure_penalty_component -= 0.5 * float(dist_to_target)

        reward = (
            pre_release_penalty
            + shaping_component
            + release_bonus_component
            + success_bonus_component
            + failure_penalty_component
        )

        self.prev_dist_to_target = float(dist_to_target)
        if self.best_dist_to_target is None:
            self.best_dist_to_target = float(dist_to_target)
        else:
            self.best_dist_to_target = min(float(self.best_dist_to_target), float(dist_to_target))

        if released_this_step:
            self.release_step = self.step_count
            self.release_distance_to_target = float(dist_to_target)
            self.release_ball_speed = ball_speed

        self.pre_release_penalty_sum += pre_release_penalty
        self.shaping_reward_sum += shaping_component
        self.release_bonus_sum += release_bonus_component
        self.success_bonus_sum += success_bonus_component
        self.failure_penalty_sum += failure_penalty_component

        info = {
            "distance_to_target": float(dist_to_target),
            "final_distance_to_target": float(dist_to_target),
            "min_distance_to_target": float(self.best_dist_to_target),
            "released": self.released,
            "released_this_step": released_this_step,
            "success": float(success),
            "release_step": float(self.release_step) if self.release_step is not None else float("nan"),
            "release_distance_to_target": (
                float(self.release_distance_to_target)
                if self.release_distance_to_target is not None
                else float("nan")
            ),
            "release_ball_speed": (
                float(self.release_ball_speed)
                if self.release_ball_speed is not None
                else float("nan")
            ),
            "termination_success": float(self.termination_type == "success"),
            "termination_ground_miss": float(self.termination_type == "ground_miss"),
            "termination_timeout_no_release": float(self.termination_type == "timeout_no_release"),
            "termination_timeout_after_release": float(self.termination_type == "timeout_after_release"),
            "reward_pre_release_penalty": float(self.pre_release_penalty_sum),
            "reward_shaping_component": float(self.shaping_reward_sum),
            "reward_release_bonus_component": float(self.release_bonus_sum),
            "reward_success_bonus_component": float(self.success_bonus_sum),
            "reward_failure_penalty_component": float(self.failure_penalty_sum),
            "max_abs_joint_velocity": float(self.max_abs_joint_velocity),
            "mean_abs_joint_velocity": float(self.cumulative_abs_joint_velocity / max(self.step_count, 1)),
            "mean_action_norm": float(self.action_norm_sum / max(self.step_count, 1)),
            "target_x": float(self.target_pos[0]),
            "target_y": float(self.target_pos[1]),
            "target_z": float(self.target_pos[2]),
            "target_radius": float(self.target_radius),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _get_ball_state(self):
        if self.ball_id is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
        ball_vel, _ = p.getBaseVelocity(self.ball_id, physicsClientId=self.client)
        return np.array(ball_pos, dtype=np.float32), np.array(ball_vel, dtype=np.float32)

    def _get_obs(self):
        joint_states = p.getJointStates(
            self.robot_id, list(range(self.n_joints)), physicsClientId=self.client
        )
        angles = np.array([s[0] for s in joint_states], dtype=np.float32)
        vels = np.array([s[1] for s in joint_states], dtype=np.float32)
        released_flag = np.array([1.0 if self.released else 0.0], dtype=np.float32)
        if self.observation_mode == "arm_target_only":
            return np.concatenate([angles, vels, self.target_pos]).astype(np.float32)
        if self.observation_mode == "arm_target_release":
            return np.concatenate([angles, vels, self.target_pos, released_flag]).astype(np.float32)
        ball_pos, ball_vel = self._get_ball_state()
        return np.concatenate([angles, vels, ball_pos, ball_vel, self.target_pos, released_flag]).astype(np.float32)

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None
