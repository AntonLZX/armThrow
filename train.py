from pathlib import Path

import torch
import yaml

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from callbacks import (
    EpisodeRecorderCallback,
    WANDB_AVAILABLE,
    WandbCallback,
    WandbEpisodeCallback,
    WandbEvalCallback,
    WandbTrainStatsCallback,
    wandb,
)
from config import _coerce_int, load_config, make_run_dir, resolve_wandb_name, set_seed
from env import ArmThrowEnv


_ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DDPG": DDPG}

# Which hyperparameter keys each algorithm accepts
_ALGO_KEYS = {
    "PPO":  {"learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef"},
    "A2C":  {"learning_rate", "n_steps", "gamma", "gae_lambda", "ent_coef"},
    "SAC":  {"learning_rate", "batch_size", "gamma", "ent_coef", "tau", "buffer_size", "learning_starts", "train_freq"},
    "TD3":  {"learning_rate", "batch_size", "gamma", "tau", "buffer_size", "learning_starts", "train_freq"},
    "DDPG": {"learning_rate", "batch_size", "gamma", "tau", "buffer_size", "learning_starts", "train_freq"},
}


def _get_algo_class(name: str):
    cls = _ALGO_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from: {list(_ALGO_CLASSES)}")
    return cls


def _build_algo_kwargs(algo_name: str, algo_cfg: dict, seed: int, tb_log: str) -> dict:
    accepted = _ALGO_KEYS[algo_name]
    kwargs = {k: v for k, v in algo_cfg.items() if k in accepted}
    # Detect GPU: try CUDA, fall back to CPU with diagnostic info
    if torch.cuda.is_available():
        device = "cuda"
        print("✓ GPU (CUDA) detected and will be used")
    else:
        device = "cpu"
        print("⚠  GPU not available. You may need to:")
        print("   - Update NVIDIA driver (run: nvidia-smi to check version)")
        print("   - Check PyTorch CUDA compatibility")
        print("   - Training will proceed on CPU (slower, but functional)")
    kwargs.update(verbose=1, device=device, seed=seed, tensorboard_log=tb_log)
    return kwargs


def main(config_path="configs/base.yaml", render=None, load_model_path=None, seed_override=None):
    cfg = load_config(config_path)
    if render is not None:
        cfg["env"]["render"] = render
    if seed_override is not None:
        cfg["seed"] = _coerce_int(seed_override, "seed_override", minimum=0)
    set_seed(cfg["seed"])

    run_dir = make_run_dir(cfg["logging"]["save_dir"], cfg["run_name"])
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = ArmThrowEnv(cfg["env"])
    env = Monitor(env, filename=str(run_dir / "monitor.csv"))

    algo_name = cfg["algo"].get("name", "PPO").upper()
    algo_class = _get_algo_class(algo_name)

    if load_model_path:
        model_path = Path(load_model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Initial model not found: {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = algo_class.load(str(model_path), env=env, device=device)
        model.verbose = 1
        model.tensorboard_log = str(run_dir / "tb")
        print(f"Loaded initial weights from: {model_path}")
    else:
        kwargs = _build_algo_kwargs(algo_name, cfg["algo"], cfg["seed"], str(run_dir / "tb"))
        model = algo_class("MlpPolicy", env, **kwargs)
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

        eval_cfg = cfg["env"].copy()
        eval_cfg["render"] = False
        eval_env = ArmThrowEnv(eval_cfg)
        eval_env.reset(seed=123)

        wandb.init(
            project=cfg["logging"]["project"],
            entity=cfg["logging"]["entity"],
            name=resolve_wandb_name(cfg["logging"], run_dir),
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
            WandbEvalCallback(eval_env=eval_env, eval_freq=5000, n_eval_episodes=20, eval_seed=123),
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
    parser.add_argument("--load-model", type=str, default=None, help="Path to an existing PPO model.zip to warm-start from")
    parser.add_argument("--seed", type=int, default=None, help="Override the training seed from the config")
    args = parser.parse_args()

    render_override = None
    if args.render:
        render_override = True
    elif args.no_render:
        render_override = False

    main(args.config, render=render_override, load_model_path=args.load_model, seed_override=args.seed)
