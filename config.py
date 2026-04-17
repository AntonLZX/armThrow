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


def load_curriculum_config(path: str):
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_curriculum_config(cfg, config_path)


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


def _coerce_bool(value, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise TypeError(f"{path} must be a boolean, got {value!r}")


def _coerce_str(value, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{path} must not be empty")
    return stripped


def _resolve_path(path_value: str, source_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return (source_dir / path).resolve()
    return path.resolve()


def _normalize_curriculum_condition(condition, path: str) -> dict:
    if not isinstance(condition, dict):
        raise TypeError(f"{path} must be a mapping")

    condition_type = str(condition.get("type", "threshold")).strip().lower()
    metric = _coerce_str(condition.get("metric"), f"{path}.metric")

    if condition_type == "threshold":
        op = str(condition.get("op", ">=")).strip()
        if op not in {">=", ">", "<=", "<", "=="}:
            raise ValueError(f"{path}.op must be one of >=, >, <=, <, ==, got {op!r}")
        return {
            "type": "threshold",
            "metric": metric,
            "op": op,
            "value": _coerce_float(condition.get("value"), f"{path}.value"),
            "consecutive": _coerce_int(condition.get("consecutive", 1), f"{path}.consecutive", minimum=1),
        }

    if condition_type == "plateau":
        mode = str(condition.get("mode", "min")).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError(f"{path}.mode must be 'min' or 'max', got {mode!r}")
        rel_min_delta = condition.get("rel_min_delta")
        if rel_min_delta is not None:
            rel_min_delta = _coerce_float(rel_min_delta, f"{path}.rel_min_delta", minimum=0.0)
        return {
            "type": "plateau",
            "metric": metric,
            "mode": mode,
            "window": _coerce_int(condition.get("window", 3), f"{path}.window", minimum=2),
            "min_delta": _coerce_float(condition.get("min_delta", 0.0), f"{path}.min_delta", minimum=0.0),
            "rel_min_delta": rel_min_delta,
            "patience": _coerce_int(condition.get("patience", 1), f"{path}.patience", minimum=1),
        }

    raise ValueError(f"{path}.type must be 'threshold' or 'plateau', got {condition_type!r}")


def _normalize_stage_decision(decision, path: str, is_last_stage: bool) -> dict:
    if decision is None:
        return {
            "action": "stop" if is_last_stage else "switch",
            "logic": "all",
            "conditions": [],
        }
    if not isinstance(decision, dict):
        raise TypeError(f"{path} must be a mapping")

    action_default = "stop" if is_last_stage else "switch"
    action = str(decision.get("action", action_default)).strip().lower()
    if action not in {"switch", "stop"}:
        raise ValueError(f"{path}.action must be 'switch' or 'stop', got {action!r}")

    logic = str(decision.get("logic", "all")).strip().lower()
    if logic not in {"all", "any"}:
        raise ValueError(f"{path}.logic must be 'all' or 'any', got {logic!r}")

    conditions_raw = decision.get("conditions", [])
    if not isinstance(conditions_raw, list):
        raise TypeError(f"{path}.conditions must be a list")

    return {
        "action": action,
        "logic": logic,
        "conditions": [
            _normalize_curriculum_condition(condition, f"{path}.conditions[{index}]")
            for index, condition in enumerate(conditions_raw)
        ],
    }


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


def normalize_curriculum_config(cfg, source_path: Path):
    if not isinstance(cfg, dict):
        raise TypeError("Top-level curriculum config must be a mapping")

    run_name = _coerce_str(cfg.get("run_name", "ppo_arm_throw_curriculum_auto"), "run_name")
    seed = _coerce_int(cfg.get("seed", 42), "seed", minimum=0)

    logging_cfg = cfg.get("logging", {})
    if not isinstance(logging_cfg, dict):
        raise TypeError("Curriculum logging section must be a mapping")

    save_dir = str(logging_cfg.get("save_dir", "runs"))
    use_wandb = _coerce_bool(logging_cfg.get("use_wandb", True), "logging.use_wandb")
    project = logging_cfg.get("project")
    entity = logging_cfg.get("entity")
    tags = logging_cfg.get("tags", [])
    notes = logging_cfg.get("notes", "")
    sync_tensorboard = _coerce_bool(logging_cfg.get("sync_tensorboard", False), "logging.sync_tensorboard")
    if project is not None:
        project = _coerce_str(project, "logging.project")
    if entity is not None:
        entity = _coerce_str(entity, "logging.entity")
    if not isinstance(tags, list):
        raise TypeError("logging.tags must be a list when provided")

    curriculum_cfg = cfg.get("curriculum")
    if not isinstance(curriculum_cfg, dict):
        raise TypeError("Curriculum config section 'curriculum' must be present and be a mapping")

    eval_freq = _coerce_int(curriculum_cfg.get("eval_freq", 5000), "curriculum.eval_freq", minimum=1)
    n_eval_episodes = _coerce_int(
        curriculum_cfg.get("n_eval_episodes", 20),
        "curriculum.n_eval_episodes",
        minimum=1,
    )
    deterministic_eval = _coerce_bool(
        curriculum_cfg.get("deterministic_eval", True),
        "curriculum.deterministic_eval",
    )

    stages_raw = curriculum_cfg.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise TypeError("curriculum.stages must be a non-empty list")

    normalized_stages = []
    for index, stage in enumerate(stages_raw):
        stage_path = f"curriculum.stages[{index}]"
        if not isinstance(stage, dict):
            raise TypeError(f"{stage_path} must be a mapping")

        stage_name = _coerce_str(stage.get("name", f"stage{index + 1}"), f"{stage_path}.name")
        config_ref = _coerce_str(stage.get("config") or stage.get("config_path"), f"{stage_path}.config")
        resolved_config_path = _resolve_path(config_ref, source_path.parent)
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"{stage_path}.config not found: {resolved_config_path}")

        stage_training_cfg = load_config(str(resolved_config_path))
        stage_seed = stage.get("seed")
        if stage_seed is None:
            stage_training_cfg["seed"] = seed
        else:
            stage_training_cfg["seed"] = _coerce_int(stage_seed, f"{stage_path}.seed", minimum=0)

        min_timesteps = _coerce_int(stage.get("min_timesteps", 0), f"{stage_path}.min_timesteps", minimum=0)
        max_timesteps = _coerce_int(
            stage.get("max_timesteps", stage_training_cfg["algo"]["total_timesteps"]),
            f"{stage_path}.max_timesteps",
            minimum=1,
        )
        if min_timesteps > max_timesteps:
            raise ValueError(
                f"{stage_path}.min_timesteps must be <= {stage_path}.max_timesteps, "
                f"got {min_timesteps} > {max_timesteps}"
            )

        normalized_stages.append({
            "name": stage_name,
            "config": config_ref,
            "resolved_config_path": str(resolved_config_path),
            "seed": stage_training_cfg["seed"],
            "min_timesteps": min_timesteps,
            "max_timesteps": max_timesteps,
            "decision": _normalize_stage_decision(
                stage.get("decision"),
                f"{stage_path}.decision",
                is_last_stage=index == len(stages_raw) - 1,
            ),
            "training_config": stage_training_cfg,
        })

    return {
        "seed": seed,
        "run_name": run_name,
        "logging": {
            "save_dir": save_dir,
            "use_wandb": use_wandb,
            "project": project,
            "entity": entity,
            "tags": tags,
            "notes": notes,
            "sync_tensorboard": sync_tensorboard,
        },
        "curriculum": {
            "eval_freq": eval_freq,
            "n_eval_episodes": n_eval_episodes,
            "deterministic_eval": deterministic_eval,
            "source_path": str(source_path),
            "stages": normalized_stages,
        },
    }


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
