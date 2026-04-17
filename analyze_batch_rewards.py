"""
Analyze and visualize results from batch reward scaling experiments.

Usage:
    python analyze_batch_rewards.py                           # Analyze latest batch
    python analyze_batch_rewards.py configs/reward_sweep
    python analyze_batch_rewards.py --plot                    # Generate plots
    python analyze_batch_rewards.py --best                    # Show best performing config
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def load_batch_results(output_dir: Path) -> Optional[list[dict]]:
    """Load batch results from CSV file."""
    results_path = output_dir / "batch_results.csv"
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return None
    
    results = []
    with open(results_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key not in ["run_dir", "trial_index"]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            results.append(row)
    
    return results


def print_results_table(results: list[dict], sort_by: str = "mean_reward") -> None:
    """Print results in a formatted table."""
    if not results:
        print("No results to display.")
        return
    
    # Sort results
    if sort_by == "mean_reward":
        try:
            results_sorted = sorted(
                results,
                key=lambda x: float(x.get("mean_reward") or float("-inf")),
                reverse=True,
            )
            sort_label = "Mean Reward ↓"
        except (ValueError, TypeError):
            results_sorted = results
            sort_label = "Trial Index"
    else:
        results_sorted = results
        sort_label = "Trial Index"
    
    # Extract parameter names
    params = [k for k in results[0].keys() 
              if k not in ["trial_index", "mean_reward", "total_timesteps", 
                          "run_dir", "success_rate", "best_distance", 
                          "wandb_tracked", "trial_time_sec"]]
    
    # Print header
    print(f"\n{'Trial':<6} {sort_label:<20}", end="")
    for param in params:
        print(f" {param:<18}", end="")
    print(f" {'Reward':<12} {'Time(s)':<10}")
    print("-" * (6 + 20 + len(params) * 20 + 22))
    
    # Print rows
    for row in results_sorted:
        trial_id = int(row.get("trial_index", -1))
        mean_reward = float(row.get("mean_reward") or 0)
        trial_time = float(row.get("trial_time_sec") or 0)
        
        print(f"{trial_id:<6} {mean_reward:>19.4f}", end="")
        
        for param in params:
            value = float(row.get(param) or 0)
            print(f" {value:>17.6g}", end="")
        
        print(f" {mean_reward:>11.4f} {trial_time:>9.1f}")


def print_statistics(results: list[dict]) -> None:
    """Print summary statistics."""
    if not results:
        return
    
    rewards = [float(r.get("mean_reward") or 0) for r in results]
    times = [float(r.get("trial_time_sec") or 0) for r in results]
    
    print(f"\n{'Statistics':=^60}")
    print(f"Total trials:        {len(results)}")
    print(f"Mean reward:         {np.mean(rewards):>10.4f}")
    print(f"Std reward:          {np.std(rewards):>10.4f}")
    print(f"Max reward:          {np.max(rewards):>10.4f}")
    print(f"Min reward:          {np.min(rewards):>10.4f}")
    print(f"Total training time: {sum(times):>10.1f}s ({sum(times)/60:.1f}m)")
    print(f"Average trial time:  {np.mean(times):>10.1f}s")
    print("=" * 60)


def find_best_config(results: list[dict]) -> Optional[dict]:
    """Find the best performing configuration."""
    if not results:
        return None
    
    best = max(results, key=lambda x: float(x.get("mean_reward") or float("-inf")))
    return best


def print_best_config(results: list[dict]) -> None:
    """Print the best performing configuration."""
    best = find_best_config(results)
    
    if best is None:
        print("No results available.")
        return
    
    print(f"\n{'Best Performing Configuration':=^60}")
    print(f"Trial Index: {int(best.get('trial_index', -1))}")
    print(f"\nHyperparameters:")
    
    params = [k for k in best.keys() 
              if k not in ["trial_index", "mean_reward", "total_timesteps", 
                          "run_dir", "success_rate", "best_distance", 
                          "wandb_tracked", "trial_time_sec"]]
    
    for param in params:
        value = float(best.get(param) or 0)
        print(f"  {param:<40} {value:.8g}")
    
    print(f"\nPerformance:")
    print(f"  Mean Reward:      {float(best.get('mean_reward') or 0):.6f}")
    print(f"  Total Timesteps:  {int(float(best.get('total_timesteps') or 0))}")
    print(f"  Training Time:    {float(best.get('trial_time_sec') or 0):.1f}s")
    print(f"  Run Directory:    {best.get('run_dir', 'N/A')}")
    print("=" * 60)


def generate_plots(results: list[dict], output_dir: Path) -> None:
    """Generate comparison plots (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")
        print("Install with: pip install matplotlib")
        return
    
    if not results:
        print("No results to plot.")
        return
    
    # Extract parameters and rewards
    params = [k for k in results[0].keys() 
              if k not in ["trial_index", "mean_reward", "total_timesteps", 
                          "run_dir", "success_rate", "best_distance", 
                          "wandb_tracked", "trial_time_sec"]]
    
    rewards = [float(r.get("mean_reward") or 0) for r in results]
    
    # Create plots for each parameter
    n_params = len(params)
    if n_params == 0:
        print("No parameters to plot.")
        return
    
    fig, axes = plt.subplots(1, min(3, n_params), figsize=(15, 4))
    if n_params == 1:
        axes = [axes]
    
    for idx, param in enumerate(params[:3]):
        param_values = [float(r.get(param) or 0) for r in results]
        
        ax = axes[idx]
        ax.scatter(param_values, rewards, alpha=0.6, s=100)
        ax.set_xlabel(param)
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"Reward vs {param}")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "reward_comparison.png"
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze batch reward scaling experiment results",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=str,
        default="configs/reward_sweep",
        help="Directory containing batch results",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Show best performing configuration",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="mean_reward",
        choices=["mean_reward", "trial_index"],
        help="Column to sort results by",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.results_dir)
    
    print(f"Loading results from: {output_dir}")
    results = load_batch_results(output_dir)
    
    if results is None:
        return 1
    
    print(f"Loaded {len(results)} trial results.\n")
    
    if args.best:
        print_best_config(results)
    else:
        print_results_table(results, sort_by=args.sort_by)
        print_statistics(results)
    
    if args.plot:
        generate_plots(results, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
