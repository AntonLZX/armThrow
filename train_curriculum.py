import argparse
import copy
import json
import math
import re
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from callbacks import (
    EpisodeRecorderCallback,
    WANDB_AVAILABLE,
    WandbCallback,
    WandbEpisodeCallback,
    WandbTrainStatsCallback,
    wandb,
)
from config import load_curriculum_config, make_run_dir, resolve_wandb_name, set_seed
from env import ArmThrowEnv
from metrics import evaluate_episodes
from train import _build_algo_kwargs, _get_algo_class


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "stage"


def _write_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, sort_keys=False, allow_nan=False)
        f.write("\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_jsonl(path, payload):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")


def _evaluate_policy_metrics(model, eval_env, n_eval_episodes=20, deterministic=True, seed_base=123):
    if deterministic:
        return evaluate_episodes(model, eval_env, n_eval_episodes, seed=seed_base)

    lengths = []
    final_distances = []
    min_distances = []
    successes = []
    releases = []
    release_steps = []
    release_ball_speeds = []
    termination_ground_misses = []
    termination_timeout_no_release = []
    termination_timeout_after_release = []
    pre_release_penalties = []
    shapings = []
    release_bonuses = []
    success_bonuses = []
    failure_penalties = []
    max_joint_velocities = []
    mean_joint_velocities = []
    mean_action_norms = []

    for episode_index in range(n_eval_episodes):
        seed = None if seed_base is None else seed_base + episode_index
        obs, _ = eval_env.reset(seed=seed)
        if hasattr(model, "reset_episode"):
            model.reset_episode(env=eval_env)
        done = False
        truncated = False
        ep_len = 0
        last_info = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, truncated, last_info = eval_env.step(action)
            ep_len += 1

        lengths.append(ep_len)
        final_distances.append(last_info.get("final_distance_to_target", math.nan))
        min_distances.append(last_info.get("min_distance_to_target", math.nan))
        successes.append(float(last_info.get("success", 0.0)))
        releases.append(float(last_info.get("released", 0.0)))
        release_steps.append(last_info.get("release_step", math.nan))
        release_ball_speeds.append(last_info.get("release_ball_speed", math.nan))
        termination_ground_misses.append(float(last_info.get("termination_ground_miss", 0.0)))
        termination_timeout_no_release.append(float(last_info.get("termination_timeout_no_release", 0.0)))
        termination_timeout_after_release.append(float(last_info.get("termination_timeout_after_release", 0.0)))
        pre_release_penalties.append(last_info.get("reward_pre_release_penalty", math.nan))
        shapings.append(last_info.get("reward_shaping_component", math.nan))
        release_bonuses.append(last_info.get("reward_release_bonus_component", math.nan))
        success_bonuses.append(last_info.get("reward_success_bonus_component", math.nan))
        failure_penalties.append(last_info.get("reward_failure_penalty_component", math.nan))
        max_joint_velocities.append(last_info.get("max_abs_joint_velocity", math.nan))
        mean_joint_velocities.append(last_info.get("mean_abs_joint_velocity", math.nan))
        mean_action_norms.append(last_info.get("mean_action_norm", math.nan))

    def _finite_mean(values):
        finite_values = [float(v) for v in values if v is not None and math.isfinite(v)]
        return float(sum(finite_values) / len(finite_values)) if finite_values else float("nan")

    def _finite_std(values):
        finite_values = [float(v) for v in values if v is not None and math.isfinite(v)]
        if not finite_values:
            return float("nan")
        mean_value = sum(finite_values) / len(finite_values)
        variance = sum((value - mean_value) ** 2 for value in finite_values) / len(finite_values)
        return float(math.sqrt(variance))

    episode_count = len(lengths)
    return {
        "valid/mean_ep_length": float(sum(lengths) / episode_count),
        "valid/success_rate": float(sum(successes) / episode_count),
        "valid/release_rate": float(sum(releases) / episode_count),
        "valid/mean_final_distance": _finite_mean(final_distances),
        "valid/std_final_distance": _finite_std(final_distances),
        "valid/min_distance_to_target": _finite_mean(min_distances),
        "valid/mean_release_step": _finite_mean(release_steps),
        "valid/mean_release_ball_speed": _finite_mean(release_ball_speeds),
        "valid/ground_miss_rate": float(sum(termination_ground_misses) / episode_count),
        "valid/timeout_no_release_rate": float(sum(termination_timeout_no_release) / episode_count),
        "valid/timeout_after_release_rate": float(sum(termination_timeout_after_release) / episode_count),
        "reward/pre_release_penalty": _finite_mean(pre_release_penalties),
        "reward/shaping": _finite_mean(shapings),
        "reward/release_bonus": _finite_mean(release_bonuses),
        "reward/success_bonus": _finite_mean(success_bonuses),
        "reward/failure_penalty": _finite_mean(failure_penalties),
        "control/max_abs_joint_velocity": _finite_mean(max_joint_velocities),
        "control/mean_abs_joint_velocity": _finite_mean(mean_joint_velocities),
        "control/mean_action_norm": _finite_mean(mean_action_norms),
    }


def _compare_threshold(value: float, op: str, target: float) -> bool:
    if op == ">=":
        return value >= target
    if op == ">":
        return value > target
    if op == "<=":
        return value <= target
    if op == "<":
        return value < target
    if op == "==":
        return value == target
    raise ValueError(f"Unsupported threshold operator: {op}")


def _evaluate_condition(condition: dict, history: list[dict]) -> dict:
    latest = history[-1]
    metric = condition["metric"]

    if condition["type"] == "threshold":
        value = float(latest.get(metric, float("nan")))
        passed = math.isfinite(value) and _compare_threshold(value, condition["op"], condition["value"])
        return {
            "type": "threshold",
            "metric": metric,
            "passed": passed,
            "value": value,
            "target": condition["value"],
            "op": condition["op"],
        }

    if condition["type"] == "plateau":
        window = condition["window"]
        if len(history) < window:
            return {
                "type": "plateau",
                "metric": metric,
                "passed": False,
                "reason": "insufficient_history",
                "window": window,
            }

        recent_values = [float(entry.get(metric, float("nan"))) for entry in history[-window:]]
        if not all(math.isfinite(value) for value in recent_values):
            return {
                "type": "plateau",
                "metric": metric,
                "passed": False,
                "reason": "non_finite_values",
                "window": window,
                "values": recent_values,
            }

        first_value = recent_values[0]
        if condition["mode"] == "min":
            best_value = min(recent_values)
            improvement = first_value - best_value
        else:
            best_value = max(recent_values)
            improvement = best_value - first_value

        passed = improvement <= condition["min_delta"]
        return {
            "type": "plateau",
            "metric": metric,
            "passed": passed,
            "mode": condition["mode"],
            "window": window,
            "values": recent_values,
            "improvement": improvement,
            "min_delta": condition["min_delta"],
        }

    raise ValueError(f"Unsupported condition type: {condition['type']}")


def _evaluate_stage_decision(stage: dict, history: list[dict], current_timesteps: int) -> dict:
    decision = stage["decision"]
    min_timesteps = stage["min_timesteps"]

    if current_timesteps < min_timesteps:
        return {
            "triggered": False,
            "action": decision["action"],
            "logic": decision["logic"],
            "checks": [],
            "current_timesteps": current_timesteps,
            "min_timesteps": min_timesteps,
            "reason": "below_min_timesteps",
        }

    if not decision["conditions"]:
        return {
            "triggered": False,
            "action": decision["action"],
            "logic": decision["logic"],
            "checks": [],
            "current_timesteps": current_timesteps,
            "min_timesteps": min_timesteps,
            "reason": "no_conditions_configured",
        }

    checks = [_evaluate_condition(condition, history) for condition in decision["conditions"]]
    triggered = all(check["passed"] for check in checks) if decision["logic"] == "all" else any(
        check["passed"] for check in checks
    )
    return {
        "triggered": triggered,
        "action": decision["action"],
        "logic": decision["logic"],
        "checks": checks,
        "current_timesteps": current_timesteps,
        "min_timesteps": min_timesteps,
        "reason": "conditions_met" if triggered else "conditions_not_met",
    }


def _prepare_stage_cfg(stage: dict) -> dict:
    cfg = copy.deepcopy(stage["training_config"])
    cfg["seed"] = stage["seed"]

    curriculum_logging = stage["curriculum_logging"]

    cfg["logging"]["use_wandb"] = curriculum_logging["use_wandb"]
    if curriculum_logging["project"]:
        cfg["logging"]["project"] = curriculum_logging["project"]
    if curriculum_logging["entity"]:
        cfg["logging"]["entity"] = curriculum_logging["entity"]
    cfg["logging"]["sync_tensorboard"] = curriculum_logging["sync_tensorboard"]

    tags = list(cfg["logging"].get("tags", []))
    tags.extend(curriculum_logging.get("tags", []))
    tags.extend(["curriculum-auto", stage["name"]])
    cfg["logging"]["tags"] = list(dict.fromkeys(tags))

    existing_notes = str(cfg["logging"].get("notes", "") or "").strip()
    curriculum_notes = str(curriculum_logging.get("notes", "") or "").strip()
    auto_note = f"Automatic curriculum stage: {stage['name']}"
    note_parts = [part for part in [existing_notes, curriculum_notes, auto_note] if part]
    cfg["logging"]["notes"] = " | ".join(dict.fromkeys(note_parts))

    configured_name = cfg["logging"].get("name")
    if configured_name in (None, "", "run-name", "auto"):
        cfg["logging"]["name"] = f"{stage['curriculum_group']} {stage['name']}"

    return cfg


def _init_stage_session(stage_cfg: dict, stage: dict, stage_dir: Path, load_model_path: str | None):
    stage_dir.mkdir(parents=True, exist_ok=False)
    _write_yaml(stage_dir / "config.yaml", stage_cfg)

    env = ArmThrowEnv(stage_cfg["env"])
    env = Monitor(env, filename=str(stage_dir / "monitor.csv"))

    algo_name = stage_cfg["algo"].get("name", "PPO").upper()
    algo_class = _get_algo_class(algo_name)

    if load_model_path:
        model_path = Path(load_model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Initial model not found: {model_path}")
        model = algo_class.load(str(model_path), env=env, device="auto")
        model.num_timesteps = 0
        model.verbose = 1
        model.tensorboard_log = str(stage_dir / "tb")
        print(f"Loaded initial weights from: {model_path}")
    else:
        kwargs = _build_algo_kwargs(algo_name, stage_cfg["algo"], stage_cfg["seed"], str(stage_dir / "tb"))
        model = algo_class("MlpPolicy", env, **kwargs)

    logger = configure(str(stage_dir), ["stdout", "csv"])
    model.set_logger(logger)

    callbacks = [
        EpisodeRecorderCallback(
            save_path=stage_dir,
            env_cfg=stage_cfg["env"],
            every_n_episodes=stage_cfg.get("logging", {}).get("gif_every_n_episodes", 10),
            verbose=1,
        )
    ]

    eval_cfg = copy.deepcopy(stage_cfg["env"])
    eval_cfg["render"] = False
    eval_env = ArmThrowEnv(eval_cfg)

    wandb_active = False
    wandb_run_url = None
    wandb_run_id = None
    wandb_run_name = None
    if stage_cfg["logging"]["use_wandb"]:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is enabled in config but not installed.")

        run = wandb.init(
            project=stage_cfg["logging"]["project"],
            entity=stage_cfg["logging"]["entity"],
            name=resolve_wandb_name(stage_cfg["logging"], stage_dir),
            config=stage_cfg,
            tags=stage_cfg["logging"]["tags"],
            notes=stage_cfg["logging"]["notes"],
            sync_tensorboard=stage_cfg["logging"]["sync_tensorboard"],
            dir=str(stage_dir),
            group=stage["curriculum_group"],
            job_type=f"curriculum_stage_{stage['name']}",
            reinit="finish_previous",
        )
        callbacks.extend([
            WandbCallback(
                model_save_path=str(stage_dir / "wandb_models"),
                model_save_freq=0,
                verbose=1,
            ),
            WandbEpisodeCallback(),
            WandbTrainStatsCallback(log_freq=1000),
        ])
        wandb_active = True
        wandb_run_url = getattr(run, "url", None)
        wandb_run_id = getattr(run, "id", None)
        wandb_run_name = getattr(run, "name", None)

    return {
        "model": model,
        "env": env,
        "eval_env": eval_env,
        "callbacks": CallbackList(callbacks) if callbacks else None,
        "wandb_active": wandb_active,
        "wandb_run_url": wandb_run_url,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
    }


def _close_stage_session(session: dict):
    session["env"].close()
    session["eval_env"].close()
    if session["wandb_active"]:
        wandb.finish()


def _log_wandb_payload(session: dict, payload: dict, step: int):
    if not session["wandb_active"]:
        return
    wandb.log(_json_safe(payload), step=step)


def _run_stage(stage: dict, stage_index: int, stage_count: int, curriculum_cfg: dict, root_run_dir: Path, previous_model_path):
    stage_cfg = _prepare_stage_cfg(stage)
    stage_dir = root_run_dir / f"{stage_index + 1:02d}_{_slugify(stage['name'])}"
    stage_event_base = {
        "stage_index": stage_index + 1,
        "stage_name": stage["name"],
        "stage_dir": str(stage_dir),
    }
    append_jsonl(root_run_dir / "curriculum_events.jsonl", {
        "type": "stage_start",
        **stage_event_base,
        "source_config": stage["resolved_config_path"],
        "load_model_path": previous_model_path,
    })

    session = _init_stage_session(stage_cfg, stage, stage_dir, previous_model_path)
    eval_history = []
    stage_decision_snapshot = None
    stage_stop_reason = "max_timesteps_cap"
    chunk_count = 0

    try:
        _log_wandb_payload(session, {
            "curriculum/event_name": "stage_start",
            "curriculum/stage_index": stage_index + 1,
            "curriculum/stage_name": stage["name"],
            "curriculum/min_timesteps": stage["min_timesteps"],
            "curriculum/max_timesteps": stage["max_timesteps"],
            "curriculum/load_model": 0 if previous_model_path is None else 1,
        }, step=0)

        while session["model"].num_timesteps < stage["max_timesteps"]:
            remaining = stage["max_timesteps"] - session["model"].num_timesteps
            chunk_timesteps = min(curriculum_cfg["curriculum"]["eval_freq"], remaining)
            reset_num_timesteps = session["model"].num_timesteps == 0
            session["model"].learn(
                total_timesteps=chunk_timesteps,
                callback=session["callbacks"],
                reset_num_timesteps=reset_num_timesteps,
            )
            chunk_count += 1

            metrics = _evaluate_policy_metrics(
                session["model"],
                session["eval_env"],
                n_eval_episodes=curriculum_cfg["curriculum"]["n_eval_episodes"],
                deterministic=curriculum_cfg["curriculum"]["deterministic_eval"],
                seed_base=123,
            )
            metrics["global_timestep"] = int(session["model"].num_timesteps)
            metrics["chunk_index"] = chunk_count
            eval_history.append(metrics)

            append_jsonl(stage_dir / "eval_metrics.jsonl", metrics)
            append_jsonl(root_run_dir / "curriculum_events.jsonl", {
                "type": "eval",
                **stage_event_base,
                "metrics": metrics,
            })
            _log_wandb_payload(session, {
                **{key: value for key, value in metrics.items() if key.startswith(("valid/", "reward/", "control/"))},
                "curriculum/event_name": "eval",
                "curriculum/stage_index": stage_index + 1,
                "curriculum/stage_name": stage["name"],
                "curriculum/chunk_index": chunk_count,
                "curriculum/current_stage_timestep": int(session["model"].num_timesteps),
            }, step=int(session["model"].num_timesteps))

            stage_decision_snapshot = _evaluate_stage_decision(
                stage,
                eval_history,
                current_timesteps=int(session["model"].num_timesteps),
            )
            if stage_decision_snapshot["triggered"]:
                stage_stop_reason = f"decision_{stage_decision_snapshot['action']}"
                break

        model_path = stage_dir / "model"
        session["model"].save(str(model_path))

        if stage_stop_reason.startswith("decision_"):
            requested_action = stage_decision_snapshot["action"]
        elif stage_index < stage_count - 1:
            requested_action = "switch"
        else:
            requested_action = "stop"

        if stage_index == stage_count - 1:
            next_action = "stop"
        elif requested_action == "switch":
            next_action = "advance"
        else:
            next_action = "stop"

        stage_summary = {
            "stage_index": stage_index + 1,
            "stage_name": stage["name"],
            "stage_dir": str(stage_dir),
            "source_config_path": stage["resolved_config_path"],
            "load_model_path": previous_model_path,
            "model_path": str(model_path) + ".zip",
            "min_timesteps": stage["min_timesteps"],
            "max_timesteps": stage["max_timesteps"],
            "completed_timesteps": int(session["model"].num_timesteps),
            "chunk_count": chunk_count,
            "stop_reason": stage_stop_reason,
            "requested_action": requested_action,
            "next_action": next_action,
            "latest_metrics": eval_history[-1] if eval_history else {},
            "decision_snapshot": stage_decision_snapshot or {},
            "eval_history_path": str(stage_dir / "eval_metrics.jsonl"),
            "wandb_run_url": session["wandb_run_url"],
            "wandb_run_id": session["wandb_run_id"],
            "wandb_run_name": session["wandb_run_name"],
        }
        _write_json(stage_dir / "stage_summary.json", stage_summary)
        append_jsonl(root_run_dir / "curriculum_events.jsonl", {
            "type": "stage_end",
            **stage_event_base,
            "summary": stage_summary,
        })
        _log_wandb_payload(session, {
            "curriculum/event_name": "stage_end",
            "curriculum/stage_index": stage_index + 1,
            "curriculum/stage_name": stage["name"],
            "curriculum/stop_reason": stage_stop_reason,
            "curriculum/requested_action": requested_action,
            "curriculum/next_action": next_action,
            "curriculum/is_terminal_stage_end": 1 if next_action != "advance" else 0,
            "curriculum/final_status": stage_stop_reason if next_action != "advance" else "stage_complete",
            "curriculum/completed_timesteps": int(session["model"].num_timesteps),
            "curriculum/min_timesteps": stage["min_timesteps"],
            "curriculum/max_timesteps": stage["max_timesteps"],
            **{key: value for key, value in (eval_history[-1] if eval_history else {}).items() if key.startswith("valid/")},
        }, step=int(session["model"].num_timesteps))
        return stage_summary
    finally:
        _close_stage_session(session)


def main(config_path: str):
    curriculum_cfg = load_curriculum_config(config_path)
    set_seed(curriculum_cfg["seed"])

    root_run_dir = make_run_dir(curriculum_cfg["logging"]["save_dir"], curriculum_cfg["run_name"])
    _write_yaml(root_run_dir / "curriculum_config.yaml", curriculum_cfg)

    stage_summaries = []
    previous_model_path = None
    final_status = "completed"
    curriculum_group = root_run_dir.name

    stages = curriculum_cfg["curriculum"]["stages"]
    for index, stage in enumerate(stages):
        stage = copy.deepcopy(stage)
        stage["curriculum_logging"] = curriculum_cfg["logging"]
        stage["curriculum_group"] = curriculum_group
        stage_summary = _run_stage(
            stage,
            stage_index=index,
            stage_count=len(stages),
            curriculum_cfg=curriculum_cfg,
            root_run_dir=root_run_dir,
            previous_model_path=previous_model_path,
        )
        stage_summaries.append(stage_summary)
        previous_model_path = stage_summary["model_path"]

        if stage_summary["next_action"] != "advance":
            if index < len(stages) - 1:
                final_status = f"stopped_early_at_{stage['name']}"
            break

    curriculum_summary = {
        "run_name": curriculum_cfg["run_name"],
        "root_run_dir": str(root_run_dir),
        "seed": curriculum_cfg["seed"],
        "curriculum_group": curriculum_group,
        "status": final_status,
        "stage_count_requested": len(stages),
        "stage_count_completed": len(stage_summaries),
        "final_model_path": stage_summaries[-1]["model_path"] if stage_summaries else None,
        "stages": stage_summaries,
    }
    _write_json(root_run_dir / "curriculum_summary.json", curriculum_summary)

    print(f"Curriculum run saved to: {root_run_dir}")
    print(f"Curriculum status: {final_status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to curriculum yaml")
    args = parser.parse_args()
    main(args.config)
