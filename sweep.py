"""
Wandb hyperparameter sweep runner.

Usage:
    python sweep.py --config configs/sweep_example.yaml           # create sweep + run 1 agent
    python sweep.py --config configs/sweep_example.yaml --count 5 # run 5 trials
    python sweep.py --sweep-id <id> --count 5                     # join an existing sweep

The config follows the same env/logging/algo structure as train.py, plus a top-level
`sweep:` section containing the wandb sweep definition (method, metric, parameters).
Sweep parameters are flat keys that override fields inside cfg["algo"].
"""

import argparse
import copy
from pathlib import Path

import yaml

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from callbacks import (
    WANDB_AVAILABLE,
    WandbCallback,
    WandbEpisodeCallback,
    WandbEvalCallback,
    WandbTrainStatsCallback,
    wandb,
)
from config import make_run_dir, set_seed
from env import ArmThrowEnv
from train import _build_algo_kwargs, _get_algo_class

# Flat sweep-parameter keys that map directly into cfg["algo"]
_ALGO_PARAM_KEYS = {
    "algo_name", "learning_rate", "batch_size", "gamma", "ent_coef",
    "n_steps", "n_epochs", "gae_lambda", "clip_range",
    "tau", "buffer_size", "learning_starts", "train_freq",
}


def _load_raw(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_sweep_params(cfg: dict, sweep_params: dict) -> dict:
    """Merge flat wandb sweep parameters into cfg["algo"]."""
    cfg = copy.deepcopy(cfg)
    for key, value in sweep_params.items():
        if key == "algo_name":
            cfg["algo"]["name"] = value
        elif key in _ALGO_PARAM_KEYS:
            cfg["algo"][key] = value
    return cfg


def run_trial(base_cfg: dict):
    """Single sweep trial. Called inside wandb.agent()."""
    wandb.init(
        project=base_cfg["logging"]["project"],
        entity=base_cfg["logging"].get("entity"),
        tags=base_cfg["logging"].get("tags", []),
        notes=base_cfg["logging"].get("notes", ""),
        sync_tensorboard=base_cfg["logging"].get("sync_tensorboard", False),
        settings=wandb.Settings(init_timeout=300),
    )
    sweep_params = dict(wandb.config)
    cfg = _apply_sweep_params(base_cfg, sweep_params)

    set_seed(cfg["seed"])

    run_dir = make_run_dir(cfg["logging"]["save_dir"], cfg["run_name"])
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = ArmThrowEnv(cfg["env"])
    env = Monitor(env, filename=str(run_dir / "monitor.csv"))

    algo_name = cfg["algo"].get("name", "PPO").upper()
    algo_class = _get_algo_class(algo_name)
    kwargs = _build_algo_kwargs(algo_name, cfg["algo"], cfg["seed"], str(run_dir / "tb"))
    model = algo_class("MlpPolicy", env, **kwargs)
    print(f"[sweep] algo={algo_name}  device={model.device}  run_dir={run_dir}")

    logger = configure(str(run_dir), ["stdout", "csv"])
    model.set_logger(logger)

    eval_cfg = cfg["env"].copy()
    eval_cfg["render"] = False
    eval_env = ArmThrowEnv(eval_cfg)
    eval_env.reset(seed=123)

    callbacks = [
        WandbCallback(model_save_path=str(run_dir / "wandb_models"), model_save_freq=0, verbose=0),
        WandbEpisodeCallback(),
        WandbTrainStatsCallback(log_freq=1000),
        WandbEvalCallback(eval_env=eval_env, eval_freq=5000, n_eval_episodes=10, eval_seed=123),
    ]

    model.learn(
        total_timesteps=cfg["algo"]["total_timesteps"],
        callback=CallbackList(callbacks),
    )

    if cfg["logging"].get("save_model", True):
        model.save(str(run_dir / "model"))

    env.close()
    eval_env.close()
    wandb.finish()


def main():
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is required for sweeps. Install it with: pip install wandb")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to sweep config yaml")
    parser.add_argument("--sweep-id", type=str, default=None, help="Join an existing sweep by ID")
    parser.add_argument("--count", type=int, default=1, help="Number of trials to run")
    args = parser.parse_args()

    if args.sweep_id is None and args.config is None:
        parser.error("Provide either --config to create a sweep or --sweep-id to join one")

    if args.sweep_id:
        # Join an existing sweep — load a config just for the base env/logging settings
        if args.config is None:
            parser.error("--config is required alongside --sweep-id (for env/logging base settings)")
        base_cfg = _load_raw(args.config)
        sweep_id = args.sweep_id
        project = base_cfg["logging"]["project"]
        entity = base_cfg["logging"].get("entity")
    else:
        base_cfg = _load_raw(args.config)
        sweep_def = base_cfg.pop("sweep")  # extract wandb sweep definition
        project = base_cfg["logging"]["project"]
        entity = base_cfg["logging"].get("entity")
        sweep_id = wandb.sweep(sweep_def, project=project, entity=entity)
        print(f"Created sweep: {sweep_id}")

    wandb.agent(
        sweep_id,
        function=lambda: run_trial(base_cfg),
        project=project,
        entity=entity,
        count=args.count,
    )


if __name__ == "__main__":
    main()
