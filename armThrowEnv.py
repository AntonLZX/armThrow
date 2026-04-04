"""Deprecated compatibility shim for the legacy `armThrowEnv.py` entry point."""

from copy import deepcopy
import warnings

from env import ArmThrowEnv as _CurrentArmThrowEnv


_LEGACY_DEFAULT_CFG = {
    "arm_urdf": "arm.urdf",
    "render": False,
    "max_steps": 240,
    "end_effector_link_index": 3,
    "accel_scale": 50.0,
    "motor_force_limit": 50.0,
    "joint_velocity_limit": 10.0,
    "target_radius": 0.1,
    "release_success_bonus": 1.0,
    "reward_mode": "absolute_distance",
    "observation_mode": "arm_target_only",
    "target": {
        "mode": "fixed",
        "fixed": [2.0, 0.0, 0.5],
        "random": {
            "x": [2.0, 2.0],
            "y": [0.0, 0.0],
            "z": [0.5, 0.5],
        },
    },
}


class ArmThrowEnv(_CurrentArmThrowEnv):
    """Backward-compatible wrapper around the current `env.ArmThrowEnv`."""

    def __init__(self, render=False, cfg=None):
        warnings.warn(
            "armThrowEnv.ArmThrowEnv is deprecated; import ArmThrowEnv from env.py instead.",
            FutureWarning,
            stacklevel=2,
        )

        if cfg is None:
            resolved_cfg = deepcopy(_LEGACY_DEFAULT_CFG)
            resolved_cfg["render"] = bool(render)
        else:
            resolved_cfg = deepcopy(cfg)
            resolved_cfg.setdefault("render", bool(render))

        super().__init__(resolved_cfg)
