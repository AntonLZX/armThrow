#!/usr/bin/env bash
# Batch Training Quick Reference Cheat Sheet
# Copy & paste commands directly into terminal

# ============================================================================
# STEP 1: TEST THE SETUP
# ============================================================================

# Generate 2 sample configs (no training)
python batch_train_rewards.py --trials 2 --dry-run

# Run 3 quick trials to verify everything works
python batch_train_rewards.py --trials 3 --no-render


# ============================================================================
# STEP 2: ANALYZE RESULTS
# ============================================================================

# Show results table sorted by performance
python analyze_batch_rewards.py

# Show best configuration found
python analyze_batch_rewards.py --best

# Generate plots comparing parameters to rewards
python analyze_batch_rewards.py --plot


# ============================================================================
# STEP 3: RUN FULL BATCH
# ============================================================================

# Run all 27 trials with default parameter grid (3×3×3)
python batch_train_rewards.py --no-render

# Run in background so you can work on other things
python batch_train_rewards.py --no-render > batch_log.txt 2>&1 &


# ============================================================================
# ADVANCED: CUSTOM PARAMETER GRIDS
# ============================================================================

# Test only specific values (fewer trials = faster testing)
python batch_train_rewards.py \
  --combinations '{"pre_release_action_penalty": [0.0001, 0.001], "pre_release_const_penalty": [0.001], "progress_shaping_scale": [1.0, 2.0, 3.0]}' \
  --no-render

# Focused ablation study on one parameter
python batch_train_rewards.py \
  --combinations '{"pre_release_action_penalty": [0.0001, 0.0003, 0.0005, 0.001, 0.003], "pre_release_const_penalty": [0.001], "progress_shaping_scale": [2.0]}' \
  --no-render

# Large parameter space exploration
python batch_train_rewards.py \
  --combinations '{"pre_release_action_penalty": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005], "pre_release_const_penalty": [0.0005, 0.001, 0.002], "progress_shaping_scale": [1.0, 1.5, 2.0, 2.5, 3.0]}' \
  --no-render


# ============================================================================
# ADVANCED: TRANSFER LEARNING
# ============================================================================

# Use a pre-trained model as starting point for all trials
python batch_train_rewards.py \
  --load-model runs/my_best_model/model.zip \
  --no-render


# ============================================================================
# ADVANCED: MULTIPLE BATCHES
# ============================================================================

# Run first batch with small values
python batch_train_rewards.py \
  --output-dir configs/sweep_small \
  --combinations '{"pre_release_action_penalty": [0.00001, 0.0001, 0.001], "pre_release_const_penalty": [0.0005, 0.001], "progress_shaping_scale": [0.5, 1.0, 1.5]}' \
  --no-render

# Run second batch with large values  
python batch_train_rewards.py \
  --output-dir configs/sweep_large \
  --combinations '{"pre_release_action_penalty": [0.01, 0.05, 0.1], "pre_release_const_penalty": [0.01, 0.05], "progress_shaping_scale": [3.0, 5.0, 10.0]}' \
  --no-render

# Compare both batches
python analyze_batch_rewards.py configs/sweep_small --best
python analyze_batch_rewards.py configs/sweep_large --best


# ============================================================================
# MONITORING & DEBUGGING
# ============================================================================

# Watch progress (updates every 30 seconds)
watch -n 30 'python analyze_batch_rewards.py | head -40'

# Check specific trial's monitor log
tail -f runs/reward_sweep_000_*/monitor.csv

# View training logs with TensorBoard
tensorboard --logdir runs/reward_sweep_000_*/tb

# See generated config for a trial
cat configs/reward_sweep/reward_sweep_000.yaml


# ============================================================================
# CLEANUP & MAINTENANCE
# ============================================================================

# Remove old experiment results to save space
rm -rf configs/reward_sweep_old/
rm -rf runs/reward_sweep_old_*/

# Archive results for later comparison
tar -czf batch_results_backup.tar.gz configs/reward_sweep/ runs/reward_sweep_*/

# Clear only CSV results but keep model files
rm configs/reward_sweep/batch_results.csv


# ============================================================================
# USING BEST RESULTS IN PRODUCTION
# ============================================================================

# After finding best config from analyze_batch_rewards.py --best,
# edit configs/base.yaml to include the best parameters:

cat >> configs/base.yaml << 'EOF'

# Best parameters from reward scaling batch (update these with actual values)
# env:
#   pre_release_action_penalty: 0.0005   
#   pre_release_const_penalty: 0.001
#   progress_shaping_scale: 2.0
EOF

# Then use in training:
python train.py --config=configs/base.yaml


# ============================================================================
# HELP & REFERENCE
# ============================================================================

# See all batch training options
python batch_train_rewards.py --help

# See all analysis options  
python analyze_batch_rewards.py --help

# Print example usage patterns
python BATCH_EXAMPLES.py

# Read full documentation
cat BATCH_TRAINING_README.md
cat SETUP_SUMMARY.md
