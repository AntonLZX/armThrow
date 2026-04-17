"""
metrics.py — Shared metric helpers for ArmThrow.

Provides:
  _finite_mean / _finite_std  — NaN-safe aggregation helpers.
  evaluate_episodes           — canonical evaluation loop used by
                                WandbEvalCallback, physics_baseline, and test.

Canonical wandb namespaces
--------------------------
  valid/*    — per-eval-batch episode statistics (success, distance, …)
  reward/*   — reward component breakdown (from env info dict)
  control/*  — joint velocity and action-norm statistics
  episode/*  — per-training-episode metrics (logged by WandbEpisodeCallback)
  test/*     — test.py sanity checks and difficulty-tier results
"""

import numpy as np


# ---------------------------------------------------------------------------
# NaN-safe aggregation
# ---------------------------------------------------------------------------

def _finite_mean(vals):
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_std(vals):
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.std(finite)) if finite else float("nan")


# ---------------------------------------------------------------------------
# Canonical evaluation loop
# ---------------------------------------------------------------------------

def evaluate_episodes(model, env, n_episodes, seed=None, verbose=False, deterministic=True):
    """
    Roll out n_episodes with model on env and return a dict of canonical metrics.

    Works with SB3 models and PhysicsBaseline — reset_episode() is called
    automatically when the model exposes that method.

    Args:
        model:       SB3-compatible model with predict(obs, deterministic=...).
        env:         ArmThrowEnv instance (already created; not closed here).
        n_episodes:  Number of episodes to evaluate.
        seed:        Optional base seed; episode i uses seed + i.
        verbose:     Print a running success-rate line every 10 % of episodes.
        deterministic:
                     Whether to use deterministic actions when calling model.predict().

    Returns:
        Dict with the keys listed in EVAL_METRIC_KEYS below.
    """
    lengths = []
    final_dists, min_dists = [], []
    successes, releases = [], []
    release_steps, release_speeds = [], []
    ground_misses, timeout_no_release, timeout_after_release = [], [], []
    pre_release_penalties, shapings, release_bonuses = [], [], []
    success_bonuses, failure_penalties = [], []
    max_joint_vels, mean_joint_vels, action_norms = [], [], []

    for i in range(n_episodes):
        ep_seed = (seed + i) if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        if hasattr(model, "reset_episode"):
            model.reset_episode(env=env)
        done = truncated = False
        ep_len = 0
        last_info = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, truncated, last_info = env.step(action)
            ep_len += 1

        lengths.append(ep_len)
        final_dists.append(last_info.get("final_distance_to_target", np.nan))
        min_dists.append(last_info.get("min_distance_to_target", np.nan))
        successes.append(float(last_info.get("success", 0.0)))
        releases.append(float(last_info.get("released", 0.0)))
        release_steps.append(last_info.get("release_step", np.nan))
        release_speeds.append(last_info.get("release_ball_speed", np.nan))
        ground_misses.append(float(last_info.get("termination_ground_miss", 0.0)))
        timeout_no_release.append(float(last_info.get("termination_timeout_no_release", 0.0)))
        timeout_after_release.append(float(last_info.get("termination_timeout_after_release", 0.0)))
        pre_release_penalties.append(last_info.get("reward_pre_release_penalty", np.nan))
        shapings.append(last_info.get("reward_shaping_component", np.nan))
        release_bonuses.append(last_info.get("reward_release_bonus_component", np.nan))
        success_bonuses.append(last_info.get("reward_success_bonus_component", np.nan))
        failure_penalties.append(last_info.get("reward_failure_penalty_component", np.nan))
        max_joint_vels.append(last_info.get("max_abs_joint_velocity", np.nan))
        mean_joint_vels.append(last_info.get("mean_abs_joint_velocity", np.nan))
        action_norms.append(last_info.get("mean_action_norm", np.nan))

        if verbose and (i + 1) % max(1, n_episodes // 10) == 0:
            print(f"  [{i + 1:4d}/{n_episodes}] running success_rate={float(np.mean(successes)):.3f}")

    return {
        # --- episode quality ---
        "valid/mean_ep_length":             float(np.mean(lengths)),
        "valid/success_rate":               float(np.mean(successes)),
        "valid/release_rate":               float(np.mean(releases)),
        "valid/mean_final_distance":        _finite_mean(final_dists),
        "valid/std_final_distance":         _finite_std(final_dists),
        "valid/min_distance_to_target":     _finite_mean(min_dists),
        "valid/mean_release_step":          _finite_mean(release_steps),
        "valid/mean_release_ball_speed":    _finite_mean(release_speeds),
        # --- termination breakdown ---
        "valid/ground_miss_rate":           float(np.mean(ground_misses)),
        "valid/timeout_no_release_rate":    float(np.mean(timeout_no_release)),
        "valid/timeout_after_release_rate": float(np.mean(timeout_after_release)),
        # --- reward components (NaN when not in env info) ---
        "reward/pre_release_penalty":       _finite_mean(pre_release_penalties),
        "reward/shaping":                   _finite_mean(shapings),
        "reward/release_bonus":             _finite_mean(release_bonuses),
        "reward/success_bonus":             _finite_mean(success_bonuses),
        "reward/failure_penalty":           _finite_mean(failure_penalties),
        # --- control quality ---
        "control/max_abs_joint_velocity":   _finite_mean(max_joint_vels),
        "control/mean_abs_joint_velocity":  _finite_mean(mean_joint_vels),
        "control/mean_action_norm":         _finite_mean(action_norms),
    }


# ---------------------------------------------------------------------------
# test.py result → wandb payload helper
# ---------------------------------------------------------------------------

# Tier order matches run_progressive_difficulty() in test.py
_DIFFICULTY_TIER_NAMES = ("easy", "medium", "hard", "extreme", "aerial")


def build_test_wandb_payload(core: dict, sanity: dict, difficulty: list) -> dict:
    """
    Convert test.py suite results into a flat wandb-ready dict.

    Core metrics reuse the canonical valid/* keys so they can be compared
    directly against training-time eval runs.  Sanity checks and difficulty
    tiers use the test/* namespace.
    """
    payload = {
        # reuse canonical valid/* names for direct cross-run comparison
        "valid/success_rate":           core["success_rate"],
        "valid/release_rate":           core["release_rate"],
        "valid/mean_ep_length":         core["mean_ep_length"],
        "valid/mean_final_distance":    core["mean_final_dist"],
        "valid/min_distance_to_target": core["mean_min_dist"],
        # reward is per-episode mean; no valid/* equivalent
        "test/mean_reward":             core["mean_reward"],
    }

    # Sanity checks: True → 1.0, False → 0.0, None → skip
    for key, val in sanity.items():
        if val is not None:
            payload[f"test/sanity_{key}"] = float(val)

    # Progressive difficulty tiers
    for i, tier in enumerate(difficulty):
        name = _DIFFICULTY_TIER_NAMES[i] if i < len(_DIFFICULTY_TIER_NAMES) else str(i)
        payload[f"test/difficulty_{name}_success_rate"] = tier["success_rate"]
        payload[f"test/difficulty_{name}_mean_final_dist"] = tier["mean_final_dist"]

    return payload
