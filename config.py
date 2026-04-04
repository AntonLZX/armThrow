import random
import time
import uuid
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_config(cfg)


def _coerce_int(value, path: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be an integer, got boolean")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"{path} must be an integer-compatible value, got {value!r}")
        result = int(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError as exc:
            raise TypeError(f"{path} must be numeric, got {value!r}") from exc
        if not numeric.is_integer():
            raise TypeError(f"{path} must be an integer-compatible value, got {value!r}")
        result = int(numeric)
    else:
        raise TypeError(f"{path} must be an integer, got {type(value).__name__}")

    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}, got {result}")
    return result


def _coerce_float(value, path: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be a float, got boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be numeric, got {value!r}") from exc

    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}, got {result}")
    return result


def normalize_config(cfg):
    if not isinstance(cfg, dict):
        raise TypeError("Top-level config must be a mapping")

    cfg["seed"] = _coerce_int(cfg.get("seed", 42), "seed", minimum=0)

    for section in ("env", "algo", "logging"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise TypeError(f"Config section '{section}' must be present and be a mapping")

    algo = cfg["algo"]
    env = cfg["env"]
    logging_cfg = cfg["logging"]

    algo["total_timesteps"] = _coerce_int(algo.get("total_timesteps"), "algo.total_timesteps", minimum=1)
    algo["n_steps"] = _coerce_int(algo.get("n_steps"), "algo.n_steps", minimum=1)
    algo["batch_size"] = _coerce_int(algo.get("batch_size"), "algo.batch_size", minimum=1)
    algo["n_epochs"] = _coerce_int(algo.get("n_epochs"), "algo.n_epochs", minimum=1)
    algo["learning_rate"] = _coerce_float(algo.get("learning_rate"), "algo.learning_rate", minimum=0.0)
    algo["gamma"] = _coerce_float(algo.get("gamma"), "algo.gamma", minimum=0.0)
    algo["gae_lambda"] = _coerce_float(algo.get("gae_lambda"), "algo.gae_lambda", minimum=0.0)
    algo["clip_range"] = _coerce_float(algo.get("clip_range"), "algo.clip_range", minimum=0.0)
    algo["ent_coef"] = _coerce_float(algo.get("ent_coef"), "algo.ent_coef", minimum=0.0)

    env["max_steps"] = _coerce_int(env.get("max_steps"), "env.max_steps", minimum=1)
    env["end_effector_link_index"] = _coerce_int(
        env.get("end_effector_link_index"), "env.end_effector_link_index", minimum=0
    )
    accel_scale_raw = env.get("accel_scale")
    motor_force_limit_raw = env.get("motor_force_limit")

    if accel_scale_raw is None:
        raise TypeError("env.accel_scale must be provided")
    if motor_force_limit_raw is None:
        raise TypeError("env.motor_force_limit must be provided")

    env["accel_scale"] = _coerce_float(accel_scale_raw, "env.accel_scale", minimum=0.0)
    env["motor_force_limit"] = _coerce_float(
        motor_force_limit_raw, "env.motor_force_limit", minimum=0.0
    )
    env["joint_velocity_limit"] = _coerce_float(
        env.get("joint_velocity_limit", 10.0), "env.joint_velocity_limit", minimum=0.0
    )
    env["target_radius"] = _coerce_float(env.get("target_radius", 0.1), "env.target_radius", minimum=0.0)
    env["observation_mode"] = str(env.get("observation_mode", "arm_target_release"))
    if env["observation_mode"] not in {"arm_target_only", "arm_target_release", "full_throw_state"}:
        raise ValueError(
            "env.observation_mode must be 'arm_target_only', 'arm_target_release', or 'full_throw_state', "
            f"got {env['observation_mode']!r}"
        )
    env["release_success_bonus"] = _coerce_float(
        env.get("release_success_bonus", 1.0), "env.release_success_bonus"
    )

    if "gif_every_n_episodes" in logging_cfg:
        logging_cfg["gif_every_n_episodes"] = _coerce_int(
            logging_cfg["gif_every_n_episodes"], "logging.gif_every_n_episodes", minimum=1
        )

    return cfg


def make_run_dir(base_dir: str, run_name: str):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(base_dir) / f"{stamp}_{run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_wandb_name(logging_cfg, run_dir: Path):
    configured = logging_cfg.get("name")
    if configured in (None, "", "run-name", "auto"):
        return run_dir.name
    return configured
