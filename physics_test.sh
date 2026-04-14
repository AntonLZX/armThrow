#!/usr/bin/env bash
# physics_test.sh — Validate progressive-difficulty tier targets using PhysicsBaseline.
#
# Runs the analytical physics controller against every tier defined in
# run_progressive_difficulty and reports whether each target is reachable.
# A tier is considered reachable if the physics baseline achieves at least
# MIN_SUCCESS_RATE success within N_EPISODES.
#
# Usage:
#   bash physics_test.sh [N_EPISODES [MIN_SUCCESS_RATE]]
#
# Examples:
#   bash physics_test.sh            # 20 episodes, 10% threshold (defaults)
#   bash physics_test.sh 50         # 50 episodes, 10% threshold
#   bash physics_test.sh 50 0.25    # 50 episodes, 25% threshold
#
# Exit code: 0 if all tiers pass, 1 if any tier fails the threshold.

set -euo pipefail

cd "$(dirname "$0")"

N_EPISODES="${1:-20}"
MIN_SUCCESS_RATE="${2:-0.10}"
SEED=42

echo "============================================================"
echo "  ArmThrow: Physics Baseline Target Reachability Test"
echo "  Episodes per tier : $N_EPISODES"
echo "  Pass threshold    : $MIN_SUCCESS_RATE success rate"
echo "  Seed              : $SEED"
echo "============================================================"

python3 - <<PYEOF
import sys
import numpy as np
from env import ArmThrowEnv
from physics_baseline import PhysicsBaseline

N_EPISODES = int($N_EPISODES)
MIN_SUCCESS_RATE = float($MIN_SUCCESS_RATE)
SEED = int($SEED)

BASE_ENV_CFG = {
    "arm_urdf": "arm.urdf",
    "render": False,
    "max_steps": 240,
    "end_effector_link_index": 3,
    "accel_scale": 50.0,
    "motor_force_limit": 50.0,
    "joint_velocity_limit": 10.0,
    "target_radius": 0.1,
    "release_success_bonus": 1.0,
    "reward_mode": "distance_progress",
    "observation_mode": "full_throw_state",
    "visualize_target": False,
}

# Mirror the tiers defined in test.py:run_progressive_difficulty exactly.
TIERS = [
    (
        "Easy    (x=2.0, y=0.0, z=0.5 — fixed center)",
        {"mode": "fixed", "fixed": [2.0, 0.0, 0.5],
         "random": {"x": [2.0, 2.0], "y": [0.0, 0.0], "z": [0.5, 0.5]}},
    ),
    (
        "Medium  (x=[1.8,2.2], y=[-0.2,0.2], z=[0.4,0.6] — trained range)",
        {"mode": "random", "fixed": [2.0, 0.0, 0.5],
         "random": {"x": [1.8, 2.2], "y": [-0.2, 0.2], "z": [0.4, 0.6]}},
    ),
    (
        "Hard    (x=[2.2,2.8], y=[-0.5,0.5], z=[0.5,0.7] — wider range)",
        {"mode": "random", "fixed": [2.0, 0.0, 0.5],
         "random": {"x": [2.2, 2.8], "y": [-0.5, 0.5], "z": [0.5, 0.7]}},
    ),
    (
        "Extreme (x=3.0, y=0.0, z=0.5 — far fixed target)",
        {"mode": "fixed", "fixed": [3.0, 0.0, 0.5],
         "random": {"x": [3.0, 3.0], "y": [0.0, 0.0], "z": [0.5, 0.5]}},
    ),
    (
        "Aerial  (x=2.0, y=0.0, z=1.5 — high fixed target)",
        {"mode": "fixed", "fixed": [2.0, 0.0, 1.5],
         "random": {"x": [2.0, 2.0], "y": [0.0, 0.0], "z": [1.5, 1.5]}},
    ),
]

results = []
all_pass = True

for label, target_cfg in TIERS:
    cfg = {**BASE_ENV_CFG, "target": target_cfg}
    model = PhysicsBaseline(env_cfg=cfg, windup_steps=30, swing_scale=1.0)
    env = ArmThrowEnv(cfg)

    successes = []
    min_dists = []
    release_counts = 0
    for i in range(N_EPISODES):
        model.reset_episode()
        obs, _ = env.reset(seed=SEED + i)
        success = False
        ep_min_dist = float("inf")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            d = info.get("distance_to_target", float("inf"))
            if d < ep_min_dist:
                ep_min_dist = d
            if info.get("success"):
                success = True
            if terminated or truncated:
                if info.get("released"):
                    release_counts += 1
                break
        successes.append(success)
        min_dists.append(ep_min_dist)
    env.close()

    sr = float(np.mean(successes))
    mean_min_dist = float(np.nanmean(min_dists))
    best_min_dist = float(np.nanmin(min_dists))
    release_rate = release_counts / N_EPISODES
    passed = sr >= MIN_SUCCESS_RATE
    if not passed:
        all_pass = False

    status = "PASS" if passed else "FAIL"
    results.append((label, sr, mean_min_dist, best_min_dist, release_rate, passed))
    print(f"\n  [{status}] {label}")
    print(f"         success_rate={sr*100:.1f}%  "
          f"mean_min_dist={mean_min_dist:.3f}m  "
          f"best_min_dist={best_min_dist:.3f}m  "
          f"release_rate={release_rate*100:.0f}%  "
          f"(threshold={MIN_SUCCESS_RATE*100:.0f}%)")

print()
print("=" * 60)
if all_pass:
    print("  RESULT: ALL TIERS REACHABLE — physics baseline meets the")
    print(f"  {MIN_SUCCESS_RATE*100:.0f}% success threshold on every tier.")
else:
    print("  RESULT: SOME TIERS MAY BE UNREACHABLE")
    print()
    print("  Tiers that failed the reachability threshold:")
    for label, sr, mmd, bmd, rr, passed in results:
        if not passed:
            diag = []
            if rr < 0.5:
                diag.append(f"low release rate ({rr*100:.0f}%) — arm may not build enough velocity")
            if bmd > 0.5:
                diag.append(f"best approach {bmd:.3f}m — target may be outside throw envelope")
            diag_str = "; ".join(diag) if diag else "unknown cause"
            print(f"    - {label}")
            print(f"      success={sr*100:.1f}%, best_min_dist={bmd:.3f}m — {diag_str}")
print("=" * 60)

sys.exit(0 if all_pass else 1)
PYEOF
