import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time

class ArmThrowEnv(gym.Env):
    def __init__(self, render=False):
        super(ArmThrowEnv, self).__init__()
        # Connect to PyBullet
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Action: [Torque1, Torque2, Torque3, Release_Trigger]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Obs: [Joint Angles (3), Joint Velocities (3), Target Position (3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.target_pos = np.array([2.0, 0.0, 0.5]) # Your (xt, yt, zt)
        self.reset()

    def reset(self, seed=None, options=None):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        
        # Load Plane and Robot (You'll need a .urdf for your 3-DOF arm)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("your_arm.urdf", [0, 0, 0], useFixedBase=True)
        
        # Load the object to be thrown (the "ball")
        self.ball_id = p.loadURDF("sphere_small.urdf", [0.1, 0, 1.0])
        
        # Constraint to keep ball in "cup" until released
        self.cid = p.createConstraint(self.robot_id, 3, self.ball_id, -1, 
                                     p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.1])
        
        self.released = False
        return self._get_obs(), {}

    def step(self, action):
        # 1. Apply Torques
        torques = action[:3] * 50.0 # Scale to Newton-meters
        for i in range(3):
            p.setJointMotorControl2(self.robot_id, i, p.TORQUE_CONTROL, force=torques[i])
        
        # 2. Handle Release
        if action[3] > 0.5 and not self.released:
            p.removeConstraint(self.cid)
            self.released = True
            
        p.stepSimulation()
        
        # 3. Reward Logic (The Curve)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        dist = np.linalg.norm(np.array(ball_pos) - self.target_pos)
        reward = np.exp(-1.0 * dist) # Your Reward Curve
        
        # Check if ball hit ground
        terminated = bool(ball_pos[2] < 0.05 and self.released)
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Implementation of joint state gathering
        joint_states = p.getJointStates(self.robot_id, [0, 1, 2])
        angles = [s[0] for s in joint_states]
        vels = [s[1] for s in joint_states]
        return np.concatenate([angles, vels, self.target_pos]).astype(np.float32)