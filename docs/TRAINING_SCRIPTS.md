# ES Training Scripts for Beam Steering

This directory contains three production-ready Evolution Strategy training scripts for optimizing plasma rod configurations toward different target angles.

## Scripts Overview

### `train_0deg.py`
Trains the ES agent to steer electromagnetic waves toward the **bottom receiver (0°)**.

**Usage:**
```bash
python src/scripts/train_0deg.py
```

**Output:** `./results/es_outputs/0deg_{timestamp}/`

---

### `train_90deg.py`
Trains the ES agent to steer electromagnetic waves toward the **right/forward receiver (90°)**.

**Usage:**
```bash
python src/scripts/train_90deg.py
```

**Output:** `./results/es_outputs/90deg_{timestamp}/`

---

### `train_180deg.py`
Trains the ES agent to steer electromagnetic waves toward the **top receiver (180°)**.

**Usage:**
```bash
python src/scripts/train_180deg.py
```

**Output:** `./results/es_outputs/180deg_{timestamp}/`

---

## Configuration

All three scripts use the same ES configuration:
- **Population size:** N = 100
- **Iterations:** 1000 total iterations
- **Perturbation scale:** σ₀ = 0.3, decays at rate 0.999/iteration
- **Adam optimizer:** η = 0.02, β₁ = 0.9, β₂ = 0.999
- **Reward function:** R = P_target − λ × Σ P_other (λ = 0.5)
- **Checkpointing:** Every 10 iterations

## Output Structure

Each training run creates the following directory structure:

```
results/es_outputs/
└── 0deg_20251213_153344/          # Timestamped output directory
    ├── checkpoint_00000/           # Iteration 0
    │   ├── metadata.json           # Hyperparameters & metrics
    │   ├── best_rho.npy            # Optimized rod configuration
    │   ├── best_eps_r.npy          # Permittivity map
    │   ├── best_Ez.npy             # Electric field
    │   ├── top_k_summary.json      # Top-10 configurations summary
    │   └── top_k_configs/
    │       ├── rank_01_rho.npy     # Best configuration
    │       ├── rank_02_rho.npy     # 2nd best
    │       └── ...
    ├── checkpoint_00010/
    ├── checkpoint_00020/
    │   ...
    └── final_training_history.png  # Training curves (4 subplots)
```

## Monitoring Progress

As the script runs, you'll see:
- **Iteration progress:** `[100/1000] (10.0%)` with population sampling progress
- **Gradient statistics:** `||g||` norm for convergence monitoring
- **Adam update sizes:** `||Δθ||` norm to track step size effectiveness
- **Reward metrics:** Best reward (↑ if improved), population mean ± std, target power, crosstalk
- **Checkpoint saves:** Confirmation when checkpoints are saved

Example output:
```
[  100/1000] (10.0%) Sampling population... [20/20] Done
  → Computing ES gradient (rank-weighted perturbations)... ||g||=1.2345e-02
  → Applying Adam update... ||Δθ||=3.4567e-03
  Summary: R_best=3.8284e+04 ↑ | R_pop=2.1547e+04±8.9334e+03 | P_target=1.2e+05 | crosstalk=0.234 | σ=0.293
  ✓ Checkpoint saved: checkpoint_00100/
```

## Post-Training Analysis

After training completes, you can analyze results:

1. **View training history:** Check `final_training_history.png` for 4-panel visualization
2. **Inspect top configurations:** Load `top_k_configs/rank_*_rho.npy` for best designs
3. **Compare angles:** Run all three scripts and compare performance across 0°, 90°, 180°
