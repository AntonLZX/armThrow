import argparse
import copy
import csv
import math
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
from config import (
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _coerce_str,
    load_config,
    make_run_dir,
    resolve_wandb_name,
    set_seed,
)
from env import ArmThrowEnv
from train import _build_algo_kwargs, _get_algo_class
from train_curriculum import _evaluate_policy_metrics, _json_safe, _write_json, _write_yaml, append_jsonl


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")


def _resolve_path(path_value: str, source_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return (source_dir / path).resolve()
    return path.resolve()


def _normalize_logging(logging_cfg: dict) -> dict:
    if not isinstance(logging_cfg, dict):
        raise TypeError("logging must be a mapping")

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
        raise TypeError("logging.tags must be a list")

    return {
        "save_dir": save_dir,
        "use_wandb": use_wandb,
        "project": project,
        "entity": entity,
        "tags": tags,
        "notes": str(notes or "").strip(),
        "sync_tensorboard": sync_tensorboard,
    }


def _normalize_stage_spec(stage_spec, path: str, source_dir: Path, seed: int) -> dict:
    if not isinstance(stage_spec, dict):
        raise TypeError(f"{path} must be a mapping")

    stage_name = _coerce_str(stage_spec.get("name"), f"{path}.name")
    config_ref = _coerce_str(stage_spec.get("config"), f"{path}.config")
    resolved_config_path = _resolve_path(config_ref, source_dir)
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"{path}.config not found: {resolved_config_path}")

    resolved_load_model_path = None
    if stage_spec.get("load_model") is not None:
        load_model_ref = _coerce_str(stage_spec.get("load_model"), f"{path}.load_model")
        resolved_load_model_path = _resolve_path(load_model_ref, source_dir)
        if not resolved_load_model_path.exists():
            raise FileNotFoundError(f"{path}.load_model not found: {resolved_load_model_path}")

    training_cfg = load_config(str(resolved_config_path))
    stage_seed = stage_spec.get("seed")
    if stage_seed is None:
        training_cfg["seed"] = seed
    else:
        training_cfg["seed"] = _coerce_int(stage_seed, f"{path}.seed", minimum=0)

    max_timesteps = _coerce_int(
        stage_spec.get("max_timesteps", training_cfg["algo"]["total_timesteps"]),
        f"{path}.max_timesteps",
        minimum=1,
    )

    return {
        "name": stage_name,
        "config": config_ref,
        "resolved_config_path": str(resolved_config_path),
        "resolved_load_model_path": str(resolved_load_model_path) if resolved_load_model_path else None,
        "seed": training_cfg["seed"],
        "max_timesteps": max_timesteps,
        "training_config": training_cfg,
    }


def load_cutpoint_sweep_config(path: str) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise TypeError("Top-level cutpoint sweep config must be a mapping")

    seed = _coerce_int(cfg.get("seed", 42), "seed", minimum=0)
    run_name = _coerce_str(cfg.get("run_name", "curriculum-cutpoint-sweep"), "run_name")
    logging_cfg = _normalize_logging(cfg.get("logging", {}))

    comparison_anchors = cfg.get("comparison_anchors", [])
    if not isinstance(comparison_anchors, list):
        raise TypeError("comparison_anchors must be a list when provided")

    sweep_cfg = cfg.get("sweep")
    if not isinstance(sweep_cfg, dict):
        raise TypeError("sweep section must be present and be a mapping")

    eval_freq = _coerce_int(sweep_cfg.get("eval_freq", 5000), "sweep.eval_freq", minimum=1)
    n_eval_episodes = _coerce_int(sweep_cfg.get("n_eval_episodes", 20), "sweep.n_eval_episodes", minimum=1)
    deterministic_eval = _coerce_bool(
        sweep_cfg.get("deterministic_eval", True),
        "sweep.deterministic_eval",
    )
    success_threshold = _coerce_float(
        sweep_cfg.get("success_threshold", 0.05),
        "sweep.success_threshold",
        minimum=0.0,
    )
    stage2_early_window_timesteps = _coerce_int(
        sweep_cfg.get("stage2_early_window_timesteps", 50000),
        "sweep.stage2_early_window_timesteps",
        minimum=1,
    )
    stage3_early_window_timesteps = _coerce_int(
        sweep_cfg.get("stage3_early_window_timesteps", 100000),
        "sweep.stage3_early_window_timesteps",
        minimum=1,
    )
    bad_regime_ground_miss_threshold = _coerce_float(
        sweep_cfg.get("bad_regime_ground_miss_threshold", 0.9),
        "sweep.bad_regime_ground_miss_threshold",
        minimum=0.0,
    )

    prelude_raw = sweep_cfg.get("prelude_stages", [])
    if not isinstance(prelude_raw, list):
        raise TypeError("sweep.prelude_stages must be a list")
    prelude_stages = [
        _normalize_stage_spec(stage_spec, f"sweep.prelude_stages[{index}]", config_path.parent, seed)
        for index, stage_spec in enumerate(prelude_raw)
    ]

    upstream_raw = sweep_cfg.get("upstream")
    if not isinstance(upstream_raw, dict):
        raise TypeError("sweep.upstream must be a mapping")
    upstream = _normalize_stage_spec(upstream_raw, "sweep.upstream", config_path.parent, seed)
    cutpoints_raw = upstream_raw.get("cutpoints")
    if not isinstance(cutpoints_raw, list) or not cutpoints_raw:
        raise TypeError("sweep.upstream.cutpoints must be a non-empty list")

    cutpoints = sorted({
        _coerce_int(value, "sweep.upstream.cutpoints[]", minimum=1)
        for value in cutpoints_raw
    })
    if cutpoints[-1] > upstream["max_timesteps"]:
        raise ValueError(
            "Largest upstream cutpoint exceeds sweep.upstream.max_timesteps: "
            f"{cutpoints[-1]} > {upstream['max_timesteps']}"
        )
    upstream["cutpoints"] = cutpoints

    downstream_raw = sweep_cfg.get("downstream_chain")
    if not isinstance(downstream_raw, list) or not downstream_raw:
        raise TypeError("sweep.downstream_chain must be a non-empty list")
    downstream_chain = [
        _normalize_stage_spec(stage_spec, f"sweep.downstream_chain[{index}]", config_path.parent, seed)
        for index, stage_spec in enumerate(downstream_raw)
    ]

    return {
        "seed": seed,
        "run_name": run_name,
        "logging": logging_cfg,
        "comparison_anchors": comparison_anchors,
        "sweep": {
            "source_path": str(config_path),
            "eval_freq": eval_freq,
            "n_eval_episodes": n_eval_episodes,
            "deterministic_eval": deterministic_eval,
            "success_threshold": success_threshold,
            "stage2_early_window_timesteps": stage2_early_window_timesteps,
            "stage3_early_window_timesteps": stage3_early_window_timesteps,
            "bad_regime_ground_miss_threshold": bad_regime_ground_miss_threshold,
            "prelude_stages": prelude_stages,
            "upstream": upstream,
            "downstream_chain": downstream_chain,
        },
    }


def _build_stage_logging(stage_cfg: dict, root_logging: dict, *, name: str, tags: list[str], notes: str) -> dict:
    cfg = copy.deepcopy(stage_cfg)
    cfg["logging"]["use_wandb"] = root_logging["use_wandb"]
    if root_logging["project"]:
        cfg["logging"]["project"] = root_logging["project"]
    if root_logging["entity"]:
        cfg["logging"]["entity"] = root_logging["entity"]
    cfg["logging"]["sync_tensorboard"] = root_logging["sync_tensorboard"]
    cfg["logging"]["name"] = name
    merged_tags = list(cfg["logging"].get("tags", []))
    merged_tags.extend(root_logging.get("tags", []))
    merged_tags.extend(tags)
    cfg["logging"]["tags"] = list(dict.fromkeys(merged_tags))
    note_parts = [
        str(cfg["logging"].get("notes", "") or "").strip(),
        root_logging.get("notes", ""),
        notes,
    ]
    cfg["logging"]["notes"] = " | ".join(part for part in note_parts if part)
    return cfg


def _make_wandb_name(
    *,
    batch_id: str,
    candidate_type: str,
    stage_name: str,
    s1_cut: str,
    s2_cut: str,
    budget: int,
    seed: int,
) -> str:
    return (
        f"{batch_id}_"
        f"curriculum-cutpoint-sweep-"
        f"{_slugify(candidate_type)}-"
        f"{_slugify(stage_name)}-"
        f"s1cut-{_slugify(s1_cut)}-"
        f"s2cut-{_slugify(s2_cut)}-"
        f"budget-{budget}-"
        f"seed-{seed}"
    )


def _init_training_session(stage_cfg: dict, stage_dir: Path, *, group: str, job_type: str, load_model_path: str | None):
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
            group=group,
            job_type=job_type,
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


def _close_session(session: dict):
    session["env"].close()
    session["eval_env"].close()
    if session["wandb_active"]:
        wandb.finish()


def _log_wandb_payload(session: dict, payload: dict, step: int):
    if not session["wandb_active"]:
        return
    wandb.log(_json_safe(payload), step=step)


def _metric_value(entry: dict, key: str):
    value = entry.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_timestep_meeting(history: list[dict], metric: str, threshold: float, op: str = ">="):
    for entry in history:
        value = _metric_value(entry, metric)
        if value is None:
            continue
        if op == ">=" and value >= threshold:
            return int(entry["global_timestep"])
        if op == ">" and value > threshold:
            return int(entry["global_timestep"])
        if op == "<=" and value <= threshold:
            return int(entry["global_timestep"])
        if op == "<" and value < threshold:
            return int(entry["global_timestep"])
    return None


def _window_metric_mean(history: list[dict], metric: str, window_timesteps: int):
    values = []
    for entry in history:
        if int(entry["global_timestep"]) > window_timesteps:
            break
        value = _metric_value(entry, metric)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def _normalized_metric_auc(history: list[dict], metric: str, window_timesteps: int):
    if not history:
        return None

    area = 0.0
    prev_timestep = 0
    last_value = None
    for entry in history:
        current_timestep = min(int(entry["global_timestep"]), window_timesteps)
        value = _metric_value(entry, metric)
        if value is None:
            if last_value is None:
                continue
            value = last_value
        if current_timestep > prev_timestep:
            area += value * (current_timestep - prev_timestep)
            prev_timestep = current_timestep
        last_value = value
        if int(entry["global_timestep"]) >= window_timesteps:
            break

    if last_value is not None and prev_timestep < window_timesteps:
        area += last_value * (window_timesteps - prev_timestep)

    return float(area / window_timesteps)


def _best_metric(history: list[dict], metric: str, mode: str):
    values = [_metric_value(entry, metric) for entry in history]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if mode == "max":
        return float(max(values))
    if mode == "min":
        return float(min(values))
    raise ValueError(f"Unsupported best-metric mode: {mode}")


def _final_metric(history: list[dict], metric: str):
    if not history:
        return None
    return _metric_value(history[-1], metric)


def _time_to_exit_bad_regime(history: list[dict], *, success_threshold: float, ground_miss_threshold: float):
    for entry in history:
        success_rate = _metric_value(entry, "valid/success_rate")
        ground_miss_rate = _metric_value(entry, "valid/ground_miss_rate")
        if success_rate is None and ground_miss_rate is None:
            continue
        success_ok = success_rate is not None and success_rate >= success_threshold
        ground_miss_ok = ground_miss_rate is not None and ground_miss_rate < ground_miss_threshold
        if success_ok or ground_miss_ok:
            return int(entry["global_timestep"])
    return None


def _stage_budget(stage_spec: dict) -> int:
    return int(stage_spec["max_timesteps"])


def _run_stage(
    *,
    root_run_dir: Path,
    root_logging: dict,
    group: str,
    eval_freq: int,
    n_eval_episodes: int,
    deterministic_eval: bool,
    stage_spec: dict,
    stage_role: str,
    stage_dir: Path,
    candidate_type: str,
    batch_id: str,
    s1_cut_label: str,
    s2_cut_label: str,
    load_model_path: str | None,
    capture_cutpoints: list[int] | None = None,
):
    effective_load_model_path = load_model_path or stage_spec.get("resolved_load_model_path")
    budget = _stage_budget(stage_spec)
    semantic_name = _make_wandb_name(
        batch_id=batch_id,
        candidate_type=candidate_type,
        stage_name=stage_spec["name"],
        s1_cut=s1_cut_label,
        s2_cut=s2_cut_label,
        budget=budget,
        seed=stage_spec["seed"],
    )
    stage_cfg = _build_stage_logging(
        stage_spec["training_config"],
        root_logging,
        name=semantic_name,
        tags=[
            "curriculum-cutpoint-sweep",
            _slugify(candidate_type),
            _slugify(stage_role),
            f"s1cut-{_slugify(s1_cut_label)}",
            f"s2cut-{_slugify(s2_cut_label)}",
            f"budget-{budget}",
            f"seed-{stage_spec['seed']}",
        ],
        notes=(
            f"cutpoint_sweep stage_role={stage_role} "
            f"s1_cut={s1_cut_label} s2_cut={s2_cut_label} "
            f"load_model={effective_load_model_path or 'none'}"
        ),
    )
    stage_cfg["seed"] = stage_spec["seed"]
    stage_cfg["algo"]["total_timesteps"] = budget

    append_jsonl(root_run_dir / "sweep_events.jsonl", {
        "type": "stage_start",
        "stage_role": stage_role,
        "stage_name": stage_spec["name"],
        "candidate_type": candidate_type,
        "stage_dir": str(stage_dir),
        "load_model_path": effective_load_model_path,
        "requested_s1_cut": s1_cut_label,
        "requested_s2_cut": s2_cut_label,
    })

    session = _init_training_session(
        stage_cfg,
        stage_dir,
        group=group,
        job_type=f"cutpoint_sweep_{_slugify(stage_role)}",
        load_model_path=effective_load_model_path,
    )

    checkpoint_targets = list(sorted(capture_cutpoints or []))
    checkpoint_records = []
    eval_history = []
    chunk_index = 0

    try:
        _log_wandb_payload(session, {
            "sweep/event_name": "stage_start",
            "sweep/stage_role": stage_role,
            "sweep/stage_name": stage_spec["name"],
            "sweep/candidate_type": candidate_type,
            "sweep/requested_s1_cut": s1_cut_label,
            "sweep/requested_s2_cut": s2_cut_label,
            "sweep/load_model": 0 if effective_load_model_path is None else 1,
            "sweep/stage_budget": budget,
        }, step=0)

        while session["model"].num_timesteps < budget:
            remaining = budget - session["model"].num_timesteps
            chunk_timesteps = min(eval_freq, remaining)
            reset_num_timesteps = session["model"].num_timesteps == 0
            session["model"].learn(
                total_timesteps=chunk_timesteps,
                callback=session["callbacks"],
                reset_num_timesteps=reset_num_timesteps,
            )
            chunk_index += 1
            current_timestep = int(session["model"].num_timesteps)

            metrics = _evaluate_policy_metrics(
                session["model"],
                session["eval_env"],
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic_eval,
                seed_base=123,
            )
            metrics["global_timestep"] = current_timestep
            metrics["chunk_index"] = chunk_index
            eval_history.append(metrics)

            append_jsonl(stage_dir / "eval_metrics.jsonl", metrics)
            append_jsonl(root_run_dir / "sweep_events.jsonl", {
                "type": "eval",
                "stage_role": stage_role,
                "stage_name": stage_spec["name"],
                "candidate_type": candidate_type,
                "requested_s1_cut": s1_cut_label,
                "requested_s2_cut": s2_cut_label,
                "metrics": metrics,
            })
            _log_wandb_payload(session, {
                **{key: value for key, value in metrics.items() if key.startswith(("valid/", "reward/", "control/"))},
                "sweep/event_name": "eval",
                "sweep/stage_role": stage_role,
                "sweep/stage_name": stage_spec["name"],
                "sweep/candidate_type": candidate_type,
                "sweep/requested_s1_cut": s1_cut_label,
                "sweep/requested_s2_cut": s2_cut_label,
                "sweep/chunk_index": chunk_index,
                "sweep/current_stage_timestep": current_timestep,
                "sweep/stage_budget": budget,
            }, step=current_timestep)

            while checkpoint_targets and current_timestep >= checkpoint_targets[0]:
                requested_cutpoint = checkpoint_targets.pop(0)
                checkpoint_base = stage_dir / "checkpoints" / f"cut_{requested_cutpoint:07d}"
                checkpoint_base.parent.mkdir(parents=True, exist_ok=True)
                session["model"].save(str(checkpoint_base))
                checkpoint_record = {
                    "requested_cutpoint": requested_cutpoint,
                    "actual_timestep": current_timestep,
                    "model_path": str(checkpoint_base) + ".zip",
                    "stage_role": stage_role,
                    "stage_name": stage_spec["name"],
                    "candidate_type": candidate_type,
                    "latest_metrics": metrics,
                }
                checkpoint_records.append(checkpoint_record)
                append_jsonl(stage_dir / "checkpoint_manifest.jsonl", checkpoint_record)
                append_jsonl(root_run_dir / "sweep_events.jsonl", {
                    "type": "checkpoint_saved",
                    "stage_role": stage_role,
                    "stage_name": stage_spec["name"],
                    "candidate_type": candidate_type,
                    "checkpoint": checkpoint_record,
                })
                _log_wandb_payload(session, {
                    "sweep/event_name": "checkpoint_saved",
                    "sweep/stage_role": stage_role,
                    "sweep/stage_name": stage_spec["name"],
                    "sweep/candidate_type": candidate_type,
                    "sweep/requested_cutpoint": requested_cutpoint,
                    "sweep/actual_cutpoint": current_timestep,
                }, step=current_timestep)

        final_model_base = stage_dir / "model"
        session["model"].save(str(final_model_base))
        stage_summary = {
            "stage_role": stage_role,
            "stage_name": stage_spec["name"],
            "stage_dir": str(stage_dir),
            "source_config_path": stage_spec["resolved_config_path"],
            "load_model_path": effective_load_model_path,
            "model_path": str(final_model_base) + ".zip",
            "budget_timesteps": budget,
            "completed_timesteps": int(session["model"].num_timesteps),
            "chunk_count": chunk_index,
            "requested_s1_cut": s1_cut_label,
            "requested_s2_cut": s2_cut_label,
            "candidate_type": candidate_type,
            "eval_history": eval_history,
            "checkpoint_records": checkpoint_records,
            "eval_history_path": str(stage_dir / "eval_metrics.jsonl"),
            "wandb_run_url": session["wandb_run_url"],
            "wandb_run_id": session["wandb_run_id"],
            "wandb_run_name": session["wandb_run_name"],
        }
        _write_json(stage_dir / "stage_summary.json", stage_summary)
        append_jsonl(root_run_dir / "sweep_events.jsonl", {
            "type": "stage_end",
            "stage_role": stage_role,
            "stage_name": stage_spec["name"],
            "candidate_type": candidate_type,
            "requested_s1_cut": s1_cut_label,
            "requested_s2_cut": s2_cut_label,
            "summary": stage_summary,
        })
        _log_wandb_payload(session, {
            "sweep/event_name": "stage_end",
            "sweep/stage_role": stage_role,
            "sweep/stage_name": stage_spec["name"],
            "sweep/candidate_type": candidate_type,
            "sweep/requested_s1_cut": s1_cut_label,
            "sweep/requested_s2_cut": s2_cut_label,
            "sweep/completed_timesteps": int(session["model"].num_timesteps),
            **{key: value for key, value in (eval_history[-1] if eval_history else {}).items() if key.startswith("valid/")},
        }, step=int(session["model"].num_timesteps))
        return stage_summary
    finally:
        _close_session(session)


def _derive_candidate_row(
    *,
    candidate_id: str,
    candidate_type: str,
    seed: int,
    fixed_s1_cut: int | None,
    fixed_s2_cut: int | None,
    upstream_requested_cut: int,
    upstream_actual_cut: int,
    downstream_summaries: list[dict],
    success_threshold: float,
    stage2_window: int,
    stage3_window: int,
    ground_miss_threshold: float,
):
    row = {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "seed": seed,
        "requested_upstream_cut": upstream_requested_cut,
        "actual_upstream_cut": upstream_actual_cut,
        "s1_cut": fixed_s1_cut,
        "s2_cut": fixed_s2_cut,
    }

    for summary in downstream_summaries:
        stage_name = summary["stage_name"]
        history = summary["eval_history"]
        if stage_name == "stage2":
            row.update({
                "stage2_budget": summary["budget_timesteps"],
                "stage2_early_recovery": _normalized_metric_auc(history, "valid/success_rate", stage2_window),
                "stage2_time_to_first_success": _first_timestep_meeting(
                    history,
                    "valid/success_rate",
                    success_threshold,
                    op=">=",
                ),
                "stage2_time_to_exit_bad_transfer_regime": _time_to_exit_bad_regime(
                    history,
                    success_threshold=success_threshold,
                    ground_miss_threshold=ground_miss_threshold,
                ),
                "stage2_best_success_rate_within_budget": _best_metric(history, "valid/success_rate", "max"),
                "stage2_final_success_rate": _final_metric(history, "valid/success_rate"),
                "stage2_final_ground_miss_rate": _final_metric(history, "valid/ground_miss_rate"),
                "stage2_final_timeout_no_release_rate": _final_metric(history, "valid/timeout_no_release_rate"),
                "stage2_final_timeout_after_release_rate": _final_metric(history, "valid/timeout_after_release_rate"),
                "stage2_final_mean_final_distance": _final_metric(history, "valid/mean_final_distance"),
                "stage2_final_min_distance": _final_metric(history, "valid/min_distance_to_target"),
                "stage2_early_ground_miss_mean": _window_metric_mean(
                    history,
                    "valid/ground_miss_rate",
                    stage2_window,
                ),
                "stage2_early_timeout_no_release_mean": _window_metric_mean(
                    history,
                    "valid/timeout_no_release_rate",
                    stage2_window,
                ),
                "stage2_wandb_run_url": summary["wandb_run_url"],
            })
        elif stage_name == "stage3":
            row.update({
                "stage3_budget": summary["budget_timesteps"],
                "stage3_time_to_first_success": _first_timestep_meeting(
                    history,
                    "valid/success_rate",
                    success_threshold,
                    op=">=",
                ),
                "stage3_best_success_rate": _best_metric(history, "valid/success_rate", "max"),
                "stage3_final_success_rate": _final_metric(history, "valid/success_rate"),
                "stage3_final_min_distance": _final_metric(history, "valid/min_distance_to_target"),
                "stage3_final_mean_final_distance": _final_metric(history, "valid/mean_final_distance"),
                "stage3_success_auc_0_window": _normalized_metric_auc(
                    history,
                    "valid/success_rate",
                    stage3_window,
                ),
                "stage3_wandb_run_url": summary["wandb_run_url"],
            })

    return row


def _write_results_table(path: Path, rows: list[dict]):
    if not rows:
        _write_json(path.with_suffix(".json"), [])
        return

    ordered_fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in ordered_fieldnames:
                ordered_fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _write_json(path.with_suffix(".json"), rows)


def main(config_path: str):
    sweep_cfg = load_cutpoint_sweep_config(config_path)
    set_seed(sweep_cfg["seed"])

    root_run_dir = make_run_dir(sweep_cfg["logging"]["save_dir"], sweep_cfg["run_name"])
    batch_id = root_run_dir.name.split("_")[0]
    _write_yaml(root_run_dir / "sweep_config.yaml", sweep_cfg)
    _write_json(root_run_dir / "comparison_anchors.json", sweep_cfg["comparison_anchors"])

    group = root_run_dir.name
    prelude_summaries = []
    previous_model_path = None
    for index, prelude_stage in enumerate(sweep_cfg["sweep"]["prelude_stages"]):
        stage_dir = root_run_dir / "prelude" / f"{index + 1:02d}_{_slugify(prelude_stage['name'])}"
        summary = _run_stage(
            root_run_dir=root_run_dir,
            root_logging=sweep_cfg["logging"],
            group=group,
            eval_freq=sweep_cfg["sweep"]["eval_freq"],
            n_eval_episodes=sweep_cfg["sweep"]["n_eval_episodes"],
            deterministic_eval=sweep_cfg["sweep"]["deterministic_eval"],
            stage_spec=prelude_stage,
            stage_role=f"prelude_{index + 1}",
            stage_dir=stage_dir,
            candidate_type="prelude_stage",
            batch_id=batch_id,
            s1_cut_label="na",
            s2_cut_label="na",
            load_model_path=previous_model_path,
        )
        prelude_summaries.append(summary)
        previous_model_path = summary["model_path"]

    upstream_stage = sweep_cfg["sweep"]["upstream"]
    bank_dir = root_run_dir / "upstream_bank" / f"01_{_slugify(upstream_stage['name'])}"
    bank_summary = _run_stage(
        root_run_dir=root_run_dir,
        root_logging=sweep_cfg["logging"],
        group=group,
        eval_freq=sweep_cfg["sweep"]["eval_freq"],
        n_eval_episodes=sweep_cfg["sweep"]["n_eval_episodes"],
        deterministic_eval=sweep_cfg["sweep"]["deterministic_eval"],
        stage_spec=upstream_stage,
        stage_role="upstream_bank",
        stage_dir=bank_dir,
        candidate_type="upstream_bank",
        batch_id=batch_id,
        s1_cut_label="bank" if upstream_stage["name"] == "stage1_bank" else (
            str(prelude_summaries[-1]["completed_timesteps"]) if prelude_summaries else "na"
        ),
        s2_cut_label="bank" if upstream_stage["name"] == "stage2_bank" else "na",
        load_model_path=previous_model_path,
        capture_cutpoints=upstream_stage["cutpoints"],
    )

    checkpoint_records = bank_summary["checkpoint_records"]
    _write_json(root_run_dir / "bank_index.json", checkpoint_records)

    candidate_rows = []
    candidate_manifest = []
    prelude_stage1_cut = None
    for summary in prelude_summaries:
        if summary["stage_name"] == "stage1":
            prelude_stage1_cut = summary["completed_timesteps"]

    for index, checkpoint in enumerate(checkpoint_records):
        upstream_requested_cut = int(checkpoint["requested_cutpoint"])
        upstream_actual_cut = int(checkpoint["actual_timestep"])
        candidate_id = f"candidate_{index + 1:02d}_{upstream_requested_cut}"
        candidate_dir = root_run_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=False)

        if upstream_stage["name"] == "stage1_bank":
            s1_cut = upstream_actual_cut
            s2_cut = None
            s1_cut_label = str(upstream_requested_cut)
            s2_cut_label = "na"
        else:
            s1_cut = prelude_stage1_cut
            s2_cut = upstream_actual_cut
            s1_cut_label = str(prelude_stage1_cut) if prelude_stage1_cut is not None else "na"
            s2_cut_label = str(upstream_requested_cut)

        downstream_summaries = []
        load_model_path = checkpoint["model_path"]
        for stage_index, downstream_stage in enumerate(sweep_cfg["sweep"]["downstream_chain"]):
            stage_dir = candidate_dir / f"{stage_index + 1:02d}_{_slugify(downstream_stage['name'])}"
            summary = _run_stage(
                root_run_dir=root_run_dir,
                root_logging=sweep_cfg["logging"],
                group=group,
                eval_freq=sweep_cfg["sweep"]["eval_freq"],
                n_eval_episodes=sweep_cfg["sweep"]["n_eval_episodes"],
                deterministic_eval=sweep_cfg["sweep"]["deterministic_eval"],
                stage_spec=downstream_stage,
                stage_role=f"candidate_downstream_{stage_index + 1}",
                stage_dir=stage_dir,
                candidate_type="sweep_candidate",
                batch_id=batch_id,
                s1_cut_label=s1_cut_label,
                s2_cut_label=s2_cut_label,
                load_model_path=load_model_path,
            )
            downstream_summaries.append(summary)
            load_model_path = summary["model_path"]

        row = _derive_candidate_row(
            candidate_id=candidate_id,
            candidate_type="sweep_candidate",
            seed=sweep_cfg["seed"],
            fixed_s1_cut=s1_cut,
            fixed_s2_cut=s2_cut,
            upstream_requested_cut=upstream_requested_cut,
            upstream_actual_cut=upstream_actual_cut,
            downstream_summaries=downstream_summaries,
            success_threshold=sweep_cfg["sweep"]["success_threshold"],
            stage2_window=sweep_cfg["sweep"]["stage2_early_window_timesteps"],
            stage3_window=sweep_cfg["sweep"]["stage3_early_window_timesteps"],
            ground_miss_threshold=sweep_cfg["sweep"]["bad_regime_ground_miss_threshold"],
        )
        candidate_rows.append(row)

        candidate_record = {
            "candidate_id": candidate_id,
            "upstream_requested_cut": upstream_requested_cut,
            "upstream_actual_cut": upstream_actual_cut,
            "checkpoint_model_path": checkpoint["model_path"],
            "s1_cut": s1_cut,
            "s2_cut": s2_cut,
            "candidate_dir": str(candidate_dir),
            "downstream_stage_summaries": downstream_summaries,
            "result_row": row,
        }
        candidate_manifest.append(candidate_record)
        _write_json(candidate_dir / "candidate_meta.json", candidate_record)

    _write_json(root_run_dir / "candidate_manifest.json", candidate_manifest)
    _write_results_table(root_run_dir / "results_table.csv", candidate_rows)

    root_summary = {
        "run_name": sweep_cfg["run_name"],
        "root_run_dir": str(root_run_dir),
        "seed": sweep_cfg["seed"],
        "comparison_anchors": sweep_cfg["comparison_anchors"],
        "prelude_summaries": prelude_summaries,
        "upstream_bank_summary": bank_summary,
        "candidate_count": len(candidate_rows),
        "results_table_csv": str(root_run_dir / "results_table.csv"),
        "results_table_json": str(root_run_dir / "results_table.json"),
    }
    _write_json(root_run_dir / "sweep_summary.json", root_summary)

    print(f"Cutpoint sweep run saved to: {root_run_dir}")
    print(f"Generated {len(candidate_rows)} sweep candidates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to cutpoint sweep yaml")
    args = parser.parse_args()
    main(args.config)
