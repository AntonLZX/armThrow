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
        self.visualize_target = bool(cfg.get("visualize_target", True))
        # Reward scaling factors for distance_progress mode
        self.pre_release_action_penalty = float(cfg.get("pre_release_action_penalty", 0.0005))
        self.pre_release_const_penalty = float(cfg.get("pre_release_const_penalty", 0.001))
        self.progress_shaping_scale = float(cfg.get("progress_shaping_scale", 2.0))
        self.progress_shaping_function = cfg.get("progress_shaping_function", "exponential")
        self.progress_shaping_param = float(cfg.get("progress_shaping_param", 1.5))
        
        if self.progress_shaping_function not in {"exponential", "tanh", "polynomial_3", "polynomial_5", "inverse_square"}:
            raise ValueError(
                f"Unsupported progress_shaping_function={self.progress_shaping_function!r}; "
                "expected one of: 'exponential', 'tanh', 'polynomial_3', 'polynomial_5', 'inverse_square'"
            )
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
        self.pre_release_abs_yaw_error_sum = 0.0
        self.pre_release_abs_yaw_error_min = None
        self.pre_release_joint0_yaw_min = None
        self.pre_release_joint0_yaw_max = None
        self.pre_release_alignment_steps = 0
        self.release_signed_yaw_error = None
        self.release_abs_yaw_error = None
        self.target_shell_id = None
        self.target_center_id = None
        self.target_success_halo_id = None
        self.target_debug_item_ids = []
        self.target_success_text_id = None

    def _compute_shaping_potential(self, distance: float) -> float:
        """
        Compute the shaping potential based on distance and selected function.
        
        Different function options:
        - exponential: exp(-param * distance)
        - tanh: tanh(param / distance) [inverse distance tanh]
        - polynomial_3: 1 / (1 + distance^3)
        - polynomial_5: 1 / (1 + distance^5)
        - inverse_square: 1 / (1 + distance^2)
        """
        if self.progress_shaping_function == "exponential":
            return float(np.exp(-self.progress_shaping_param * distance))
        elif self.progress_shaping_function == "tanh":
            # Use param as scaling factor for the input
            return float(np.tanh(self.progress_shaping_param / (distance + 1e-6)))
        elif self.progress_shaping_function == "polynomial_3":
            return float(1.0 / (1.0 + self.progress_shaping_param * (distance ** 3)))
        elif self.progress_shaping_function == "polynomial_5":
            return float(1.0 / (1.0 + self.progress_shaping_param * (distance ** 5)))
        elif self.progress_shaping_function == "inverse_square":
            return float(1.0 / (1.0 + self.progress_shaping_param * (distance ** 2)))
        else:
            return float(np.exp(-self.progress_shaping_param * distance))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _compute_target_yaw_rad(self) -> float:
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        target_delta_xy = np.asarray(self.target_pos[:2], dtype=np.float32) - np.asarray(
            base_pos[:2], dtype=np.float32
        )
        return float(np.arctan2(target_delta_xy[1], target_delta_xy[0]))

    def _sample_target(self):
        target_cfg = self.cfg["target"]
        if target_cfg["mode"] == "fixed":
            return np.array(target_cfg["fixed"], dtype=np.float32)

        random_cfg = target_cfg["random"]
        sampling_mode = random_cfg.get("sampling_mode", "uniform_xyz")
        
        if sampling_mode == "sphere_with_cylinder_exclusion":
            return self._sample_sphere_with_cylinder_exclusion(random_cfg)
        else:
            # Default: uniform x/y/z sampling
            xr = random_cfg["x"]
            yr = random_cfg["y"]
            zr = random_cfg["z"]
            return np.array([
                self.np_random.uniform(xr[0], xr[1]),
                self.np_random.uniform(yr[0], yr[1]),
                self.np_random.uniform(zr[0], zr[1]),
            ], dtype=np.float32)

    def _sample_sphere_with_cylinder_exclusion(self, random_cfg):
        """Sample a point uniformly in a sphere, excluding a cylinder."""
        sphere_cfg = random_cfg["sphere"]
        cylinder_cfg = random_cfg["cylinder_exclusion"]
        
        sphere_center = np.array(sphere_cfg["center"], dtype=np.float32)
        sphere_radius = float(sphere_cfg["radius"])
        
        cylinder_center = np.array(cylinder_cfg["center"], dtype=np.float32)
        cylinder_radius = float(cylinder_cfg["radius"])
        cylinder_height = float(cylinder_cfg["height"])
        
        max_attempts = 1000
        for _ in range(max_attempts):
            # Sample uniformly in sphere
            while True:
                u = self.np_random.uniform(-1, 1, 3)
                r_sq = np.sum(u ** 2)
                if r_sq <= 1.0:
                    break
            
            # Scale to sphere
            point = sphere_center + u * sphere_radius
            
            # Check if point is in cylinder exclusion zone
            rel_pos = point - cylinder_center
            horizontal_dist = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            vertical_dist = np.abs(rel_pos[2])
            
            # If not in exclusion cylinder, return this point
            if horizontal_dist > cylinder_radius or vertical_dist > cylinder_height / 2.0:
                return point.astype(np.float32)
        
        # Fallback if max attempts exceeded (shouldn't happen with reasonable geometry)
        return sphere_center.astype(np.float32)


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
        self._create_target_visuals()

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
        self.pre_release_abs_yaw_error_sum = 0.0
        self.pre_release_abs_yaw_error_min = None
        self.pre_release_joint0_yaw_min = None
        self.pre_release_joint0_yaw_max = None
        self.pre_release_alignment_steps = 0
        self.release_signed_yaw_error = None
        self.release_abs_yaw_error = None
        return self._get_obs(), {}

    def _create_visual_sphere(self, radius, rgba, position):
        visual_shape_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=float(radius),
            rgbaColor=rgba,
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[float(v) for v in position],
            useMaximalCoordinates=True,
            physicsClientId=self.client,
        )

    def _create_target_visuals(self):
        self.target_shell_id = None
        self.target_center_id = None
        self.target_success_halo_id = None
        self.target_debug_item_ids = []
        self.target_success_text_id = None

        if not self.visualize_target:
            return

        self.target_shell_id = self._create_visual_sphere(
            radius=self.target_radius,
            rgba=[1.0, 0.2, 0.2, 0.28],
            position=self.target_pos,
        )
        center_radius = min(max(self.target_radius * 0.2, 0.018), 0.04)
        self.target_center_id = self._create_visual_sphere(
            radius=center_radius,
            rgba=[1.0, 0.95, 0.25, 0.95],
            position=self.target_pos,
        )

        line_half_length = max(self.target_radius * 1.35, 0.06)
        x, y, z = [float(v) for v in self.target_pos]
        line_specs = [
            ([x - line_half_length, y, z], [x + line_half_length, y, z], [1.0, 0.35, 0.35]),
            ([x, y - line_half_length, z], [x, y + line_half_length, z], [0.35, 0.95, 1.0]),
            ([x, y, z - line_half_length], [x, y, z + line_half_length], [0.5, 1.0, 0.45]),
            ([x, y, 0.02], [x, y, z], [1.0, 0.8, 0.3]),
        ]
        for line_from, line_to, color in line_specs:
            debug_id = p.addUserDebugLine(
                line_from,
                line_to,
                color,
                lineWidth=2.0,
                physicsClientId=self.client,
            )
            self.target_debug_item_ids.append(debug_id)

        label_position = [x, y, z + max(self.target_radius * 1.5, 0.08)]
        label_id = p.addUserDebugText(
            f"TARGET r={self.target_radius:.2f}",
            label_position,
            textColorRGB=[1.0, 0.95, 0.25],
            textSize=1.3,
            physicsClientId=self.client,
        )
        self.target_debug_item_ids.append(label_id)

    def _highlight_success(self):
        if not self.visualize_target:
            return

        if self.target_shell_id is not None:
            p.changeVisualShape(
                self.target_shell_id,
                -1,
                rgbaColor=[0.2, 1.0, 0.35, 0.35],
                physicsClientId=self.client,
            )
        if self.target_center_id is not None:
            p.changeVisualShape(
                self.target_center_id,
                -1,
                rgbaColor=[0.1, 1.0, 0.1, 1.0],
                physicsClientId=self.client,
            )
        if self.target_success_halo_id is None:
            self.target_success_halo_id = self._create_visual_sphere(
                radius=self.target_radius * 1.45,
                rgba=[0.2, 1.0, 0.35, 0.12],
                position=self.target_pos,
            )

        text_position = [
            float(self.target_pos[0]),
            float(self.target_pos[1]),
            float(self.target_pos[2] + max(self.target_radius * 2.0, 0.12)),
        ]
        self.target_success_text_id = p.addUserDebugText(
            "SUCCESS",
            text_position,
            textColorRGB=[0.15, 1.0, 0.2],
            textSize=1.5,
            physicsClientId=self.client,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        was_released = self.released
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

        joint0_yaw_rad = float(p.getJointState(self.robot_id, 0, physicsClientId=self.client)[0])
        target_yaw_rad = self._compute_target_yaw_rad()
        signed_yaw_error_rad = self._wrap_to_pi(target_yaw_rad - joint0_yaw_rad)
        abs_yaw_error_rad = abs(signed_yaw_error_rad)

        if not was_released:
            self.pre_release_alignment_steps += 1
            self.pre_release_abs_yaw_error_sum += abs_yaw_error_rad
            if self.pre_release_abs_yaw_error_min is None:
                self.pre_release_abs_yaw_error_min = abs_yaw_error_rad
            else:
                self.pre_release_abs_yaw_error_min = min(self.pre_release_abs_yaw_error_min, abs_yaw_error_rad)

            if self.pre_release_joint0_yaw_min is None:
                self.pre_release_joint0_yaw_min = joint0_yaw_rad
                self.pre_release_joint0_yaw_max = joint0_yaw_rad
            else:
                self.pre_release_joint0_yaw_min = min(self.pre_release_joint0_yaw_min, joint0_yaw_rad)
                self.pre_release_joint0_yaw_max = max(self.pre_release_joint0_yaw_max, joint0_yaw_rad)

        dist_to_target = np.linalg.norm(ball_pos - self.target_pos)

        prev_dist_to_target = (
            float(self.prev_dist_to_target)
            if self.prev_dist_to_target is not None
            else float(dist_to_target)
        )
        phi_t = self._compute_shaping_potential(dist_to_target)
        phi_prev = self._compute_shaping_potential(prev_dist_to_target)

        reward = 0.0

        if self.reward_mode == "absolute_distance":
            if not self.released:
                pre_release_penalty -= self.pre_release_action_penalty * float(np.sum(np.square(action[:3])))
            elif ball_pos[2] >= 0.05:
                shaping_component += phi_t
        else:
            if not self.released:
                pre_release_penalty -= self.pre_release_action_penalty * float(np.sum(np.square(action[:3])))
                pre_release_penalty -= self.pre_release_const_penalty
            elif ball_pos[2] >= 0.05:
                shaping_component += self.progress_shaping_scale * (phi_t - phi_prev)

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
            self._highlight_success()

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
            self.release_signed_yaw_error = signed_yaw_error_rad
            self.release_abs_yaw_error = abs_yaw_error_rad

        self.pre_release_penalty_sum += pre_release_penalty
        self.shaping_reward_sum += shaping_component
        self.release_bonus_sum += release_bonus_component
        self.success_bonus_sum += success_bonus_component
        self.failure_penalty_sum += failure_penalty_component

        mean_pre_release_abs_yaw_error = float("nan")
        if self.pre_release_alignment_steps > 0:
            mean_pre_release_abs_yaw_error = float(
                self.pre_release_abs_yaw_error_sum / self.pre_release_alignment_steps
            )

        pre_release_joint0_range = float("nan")
        if self.pre_release_joint0_yaw_min is not None and self.pre_release_joint0_yaw_max is not None:
            pre_release_joint0_range = float(self.pre_release_joint0_yaw_max - self.pre_release_joint0_yaw_min)

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
            "release_signed_yaw_error_rad": (
                float(self.release_signed_yaw_error)
                if self.release_signed_yaw_error is not None
                else float("nan")
            ),
            "release_abs_yaw_error_rad": (
                float(self.release_abs_yaw_error)
                if self.release_abs_yaw_error is not None
                else float("nan")
            ),
            "mean_pre_release_abs_yaw_error_rad": mean_pre_release_abs_yaw_error,
            "min_pre_release_abs_yaw_error_rad": (
                float(self.pre_release_abs_yaw_error_min)
                if self.pre_release_abs_yaw_error_min is not None
                else float("nan")
            ),
            "pre_release_joint0_range_rad": pre_release_joint0_range,
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
