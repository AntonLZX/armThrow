"""
Batch training script to experiment with different reward scaling factor combinations.

Runs multiple training trials with different combinations of:
- pre_release_action_penalty: penalty multiplier for joint action during pre-release
- pre_release_const_penalty: constant penalty during pre-release phase
- progress_shaping_scale: scaling factor for distance progress reward

Usage:
    python batch_train_rewards.py                    # Run with default parameter combinations
    python batch_train_rewards.py --base-config configs/base.yaml
    python batch_train_rewards.py --trials 5         # Run first 5 combinations
    python batch_train_rewards.py --dry-run           # Print combinations without training
    python batch_train_rewards.py --compare-only      # Show summary of existing runs
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from itertools import product

import yaml
import numpy as np


# Default parameter combinations to test
DEFAULT_COMBINATIONS = {
    "pre_release_action_penalty": [0.0001, 0.0005, 0.001],
    "pre_release_const_penalty": [0.0005, 0.001, 0.002],
    "progress_shaping_scale": [1.0, 2.0, 3.0],
}


def load_base_config(config_path: str) -> dict:
    """Load base training configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def create_batch_configs(
    base_cfg: dict,
    combinations: dict,
    output_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    """
    Generate configuration variants with different reward scaling factors.
    
    Returns list of (filepath, config_dict) tuples for each combination.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    param_names = list(combinations.keys())
    param_values = [combinations[name] for name in param_names]
    all_combinations = list(product(*param_values))
    
    configs = []
    for idx, param_combo in enumerate(all_combinations):
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # Deep copy
        
        # Apply parameter override
        for param_name, param_value in zip(param_names, param_combo):
            cfg["env"][param_name] = param_value
        
        # Create descriptive run name
        param_str = "_".join(
            f"{name[:10]}={val:.4g}" 
            for name, val in zip(param_names, param_combo)
        )
        cfg["run_name"] = f"reward_sweep_{idx:03d}_{param_str}"
        
        # Save config to file
        config_path = output_dir / f"reward_sweep_{idx:03d}.yaml"
        if not dry_run:
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        
        configs.append({
            "index": idx,
            "config_path": config_path,
            "config": cfg,
            "params": dict(zip(param_names, param_combo)),
        })
    
    return configs


def run_training(
    config_path: Path,
    render: bool = False,
    load_model_path: str = None,
) -> dict:
    """Run a single training trial and return results."""
    cmd = ["python", "train.py", f"--config={config_path}"]
    
    if render:
        cmd.append("--render")
    
    if load_model_path:
        cmd.append(f"--load={load_model_path}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=None)
        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "return_code": -1,
            "error": "Training timeout",
        }
    except Exception as e:
        return {
            "success": False,
            "return_code": -1,
            "error": str(e),
        }


def extract_run_metrics(run_dir: Path) -> dict:
    """Extract performance metrics from a training run directory."""
    metrics = {
        "run_dir": str(run_dir),
        "success_rate": None,
        "mean_reward": None,
        "best_distance": None,
        "total_timesteps": None,
    }
    
    # Try to read from monitor.csv if it exists
    monitor_path = run_dir / "monitor.csv"
    if monitor_path.exists():
        try:
            with open(monitor_path, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Last line should have the latest stats
                    last_line = lines[-1].strip()
                    parts = last_line.split(",")
                    if len(parts) >= 5:
                        metrics["mean_reward"] = float(parts[1])
                        metrics["total_timesteps"] = int(float(parts[2]))
        except Exception:
            pass
    
    # Try to extract from wandb logs if available
    wandb_log_dir = run_dir / "wandb" / "latest-run"
    if wandb_log_dir.exists():
        try:
            files_run_path = wandb_log_dir / "files" / "run"
            if files_run_path.exists():
                # Count episodes to estimate success rate
                events_file = list(files_run_path.glob("**/events.out.tfevents*"))
                if events_file:
                    # Basic check that run was created
                    metrics["wandb_tracked"] = True
        except Exception:
            pass
    
    return metrics


def format_combination(params: dict) -> str:
    """Format parameter combination as string."""
    lines = []
    for name, value in params.items():
        lines.append(f"  {name:.<40} {value:.6g}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Batch train with different reward scaling factor combinations",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Base configuration file to use",
    )
    parser.add_argument(
        "--combinations",
        type=lambda x: json.loads(x),
        default=None,
        help="JSON string with parameter combinations. "
             "Default: {\"pre_release_action_penalty\": [0.0001, 0.0005, 0.001], ...}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/reward_sweep",
        help="Directory to save generated config files",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Only run first N combinations (useful for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combinations without running training",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during training",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only analyze existing runs, don't train new ones",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to pre-trained model to load for all trials",
    )
    
    args = parser.parse_args()
    
    # Load base config
    base_cfg = load_base_config(args.base_config)
    print(f"Loaded base config from: {args.base_config}")
    
    # Use provided combinations or defaults
    combinations = args.combinations or DEFAULT_COMBINATIONS
    
    # Make sure reward_mode is set to distance_progress
    base_cfg["env"]["reward_mode"] = "distance_progress"
    
    output_dir = Path(args.output_dir)
    
    # Generate configurations
    print(f"\nGenerating configurations with parameter combinations...")
    configs = create_batch_configs(
        base_cfg,
        combinations,
        output_dir,
        dry_run=args.dry_run,
    )
    
    print(f"Generated {len(configs)} configuration variants:\n")
    
    # Limit trials if requested
    if args.trials:
        configs = configs[:args.trials]
        print(f"Running first {len(configs)} trials.\n")
    
    # Print combinations
    for cfg_info in configs:
        print(f"\n[{cfg_info['index']:03d}] Config: {cfg_info['config_path'].name}")
        print(format_combination(cfg_info["params"]))
    
    if args.dry_run:
        print(f"\n[DRY RUN] Skipping training. Generated {len(configs)} configs.")
        return 0
    
    if args.compare_only:
        print(f"\n[COMPARE ONLY] Analyzing existing runs in {output_dir}...")
        print("Not running new training.\n")
        return 0
    
    # Run training for each configuration
    print(f"\n{'='*70}")
    print(f"Starting training for {len(configs)} configurations...")
    print(f"{'='*70}\n")
    
    results = []
    start_time = time.time()
    
    for idx, cfg_info in enumerate(configs):
        cfg_index = cfg_info["index"]
        config_path = cfg_info["config_path"]
        params = cfg_info["params"]
        
        print(f"\n[{idx+1}/{len(configs)}] Running trial {cfg_index}:")
        print(format_combination(params))
        print(f"Config: {config_path}")
        
        trial_start = time.time()
        
        # Run training
        success = run_training(
            config_path,
            render=not args.no_render,
            load_model_path=args.load_model,
        )
        
        trial_time = time.time() - trial_start
        
        # Collect metrics
        run_dir = Path(base_cfg["logging"]["save_dir"]) / cfg_info["config"]["run_name"]
        metrics = extract_run_metrics(run_dir)
        metrics["trial_time_sec"] = trial_time
        
        result_record = {
            "trial_index": cfg_index,
            **params,
            **metrics,
        }
        results.append(result_record)
        
        status = "✓ SUCCESS" if success["success"] else "✗ FAILED"
        print(f"{status} (took {trial_time:.1f}s)")
        
        if metrics.get("mean_reward") is not None:
            print(f"  Mean reward: {metrics['mean_reward']:.4f}")
        if metrics.get("total_timesteps") is not None:
            print(f"  Total timesteps: {metrics['total_timesteps']}")
    
    total_time = time.time() - start_time
    
    # Save results summary
    summary_path = output_dir / "batch_results.csv"
    if results:
        # Write CSV with results
        fieldnames = list(results[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*70}")
        print(f"Batch training completed in {total_time/60:.1f} minutes")
        print(f"{'='*70}")
        print(f"\nResults saved to: {summary_path}")
        
        # Print summary table
        print(f"\n{'Summary of all trials:':^70}")
        print(f"{'-'*70}")
        print(f"Trial | pre_release_action | pre_release_const | shaping_scale")
        print(f"      | penalty            | penalty           | ")
        print(f"{'-'*70}")
        for record in results:
            trial_id = record.get("trial_index", "?")
            action_pen = record.get("pre_release_action_penalty", "?")
            const_pen = record.get("pre_release_const_penalty", "?")
            shaping = record.get("progress_shaping_scale", "?")
            print(f"{trial_id:3d}  | {action_pen:17.5g} | {const_pen:16.5g} | {shaping:13.5g}")
        print(f"{'-'*70}")
    else:
        print(f"\nNo results collected.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
