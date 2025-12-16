# Agent Development Checklist

This checklist ensures every ES/training agent is production-ready for AWS Batch deployment.

---

## 1. Configuration Parity

- [ ] **Hyperparameters match baseline** (ES_CONFIG)
  - [ ] `N = 100` (population size)
  - [ ] `sigma_0 = 0.3` (initial perturbation)
  - [ ] `sigma_decay = 0.999` (per-iteration decay)
  - [ ] `eta = 0.02` (Adam learning rate)
  - [ ] `beta_1 = 0.9`, `beta_2 = 0.999` (Adam momentum)
  - [ ] `adam_eps = 1e-8`
  - [ ] `n_iterations = 1000`
  - [ ] `log_every = 10`
  - [ ] `frequency_hz = 6e9` (6 GHz)
  - [ ] `n_jobs = 1` (default, overridden at runtime)

- [ ] **Any differing parameters are intentional and documented**
  - Document WHY the parameter differs
  - Example: ES-Multi uses `lambda_variance=0.5` instead of `lambda_crosstalk=0.5`

---

## 2. History Logging (Required Fields)

All agents MUST log these fields for comparison:

| Field | Description | Required |
|-------|-------------|----------|
| `iteration` | Current iteration number | ✅ |
| `reward_best` | Best reward so far | ✅ |
| `reward_mean` | Population mean reward | ✅ |
| `reward_std` | Population std reward | ✅ |
| `power_0deg` | Power at 0° receiver | ✅ |
| `power_90deg` | Power at 90° receiver | ✅ |
| `power_180deg` | Power at 180° receiver | ✅ |
| `power_target` | Target power (varies by agent) | ✅ |
| `power_other_sum` | Sum of non-target powers | ✅ |
| `crosstalk_ratio` | Balance/crosstalk metric | ✅ |
| `sigma` | Current perturbation scale | ✅ |
| `eta_effective` | Adam update norm | ✅ |

Additional fields are allowed but must not break existing analysis scripts.

---

## 3. Checkpoint Structure

Each checkpoint directory must contain:

```
checkpoint_XXXXX/
├── metadata.json          # Config, iteration, best metrics
├── best_rho.npy           # Best rod configuration (8×8)
├── best_eps_r.npy         # Full permittivity map
├── best_Ez.npy            # Electric field solution
├── training_history.json  # Full history up to this point
├── top_k_summary.json     # Top-K configurations summary
└── top_k_configs/
    ├── rank_01_rho.npy
    ├── rank_02_rho.npy
    └── ...
```

### metadata.json Required Fields:
- `iteration`
- `reward_best`
- `config` (full hyperparameters)
- `power_0deg`, `power_90deg`, `power_180deg`

---

## 4. S3 Upload

- [ ] **`_upload_checkpoint_to_s3()` method exists**
- [ ] **Called in `train()` after every `save_checkpoint()`**
- [ ] **Checks `S3_BUCKET` env var before uploading**
- [ ] **Uploads ALL files in checkpoint directory recursively**
- [ ] **Uses correct S3 prefix**: `{output_dir.name}/checkpoint_XXXXX/`

### S3 Upload Code Pattern:
```python
s3_bucket = os.environ.get('S3_BUCKET')
if s3_bucket:
    try:
        self._upload_checkpoint_to_s3(iteration, s3_bucket)
    except Exception as e:
        print(f"S3 upload failed: {e}")
```

---

## 5. Training Script

Each agent needs a corresponding training script in `src/scripts/`:

- [ ] **Script exists**: `train_{name}.py`
- [ ] **Has shebang**: `#!/usr/bin/env python3`
- [ ] **Adds project root to sys.path**
- [ ] **Prints clear header with reward function**
- [ ] **Checks S3_BUCKET env var and reports status**
- [ ] **Uses unbuffered output**: `python3 -u`

### Training Script Template:
```python
#!/usr/bin/env python3
"""
Train {AgentName}: {Brief description}

Reward Function:
    R = {formula}

Usage:
    python3 -u src/scripts/train_{name}.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.{module} import {Agent}, {CONFIG}

if __name__ == '__main__':
    print("=" * 80)
    print("{AGENT NAME}: {Description}")
    print("=" * 80)
    # ... training code
```

---

## 6. Functional Tests

Run before deployment:

### Quick Test (2 iterations)
```bash
python3 -c "
from src.training.{module} import {Agent}, {CONFIG}
config = {CONFIG}.copy()
config['N'] = 10
config['n_iterations'] = 2
config['log_every'] = 1
config['n_jobs'] = 4
agent = {Agent}(config=config, output_dir='es_outputs/test_{name}')
best_rho, best_reward = agent.train(verbose=True)
print(f'Test passed: reward={best_reward:.4e}')
"
```

### Checklist:
- [ ] Agent initializes without error
- [ ] Training runs for 2 iterations
- [ ] All history fields populated
- [ ] Checkpoint created with all files
- [ ] Verbose output shows correct reward formula

---

## 7. Docker & AWS Batch

- [ ] **Dockerfile CMD updated** (if new default script)
- [ ] **Docker image rebuilt**: `docker build --platform linux/amd64 -t {image}:latest .`
- [ ] **Pushed to ECR**: `docker push {ecr_url}:latest`
- [ ] **Job definition exists or updated**
- [ ] **S3_BUCKET env var in job definition**
- [ ] **jobRoleArn has S3 access**

---

## 8. Comparison Readiness

For fair comparison between agents:

- [ ] **Same λ value** (0.5) where applicable
- [ ] **Same iteration count** (1000)
- [ ] **Same population size** (100)
- [ ] **Same random seed handling** (or multiple runs)
- [ ] **Individual powers logged** (power_0deg, power_90deg, power_180deg)

### Key Comparison Metrics:

| Comparison | ES-Single | ES-Multi |
|------------|-----------|----------|
| P_0° performance | `power_0deg` from 0° run | `power_0deg` |
| P_90° performance | `power_90deg` from 90° run | `power_90deg` |
| P_180° performance | `power_180deg` from 180° run | `power_180deg` |

---

## Agent Status

| Agent | Config ✓ | History ✓ | Checkpoint ✓ | S3 ✓ | Script ✓ | Tested ✓ |
|-------|----------|-----------|--------------|------|----------|----------|
| ESAgent (0°) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ESAgent (90°) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ESAgent (180°) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ESMultiAgent | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ES+NN Agent | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

---

## Quick Validation Command

Run this to validate any agent:

```bash
python3 -c "
from src.training.{module} import {Agent}, {CONFIG}
from src.training.es_agent import ES_CONFIG

# 1. Config check
print('Config matches ES_CONFIG:', all(
    {CONFIG}.get(k) == v for k, v in ES_CONFIG.items() 
    if k not in ['lambda_crosstalk', 'lambda_variance']
))

# 2. History fields check
agent = {Agent}(...)
required = ['iteration', 'reward_best', 'reward_mean', 'reward_std',
            'power_0deg', 'power_90deg', 'power_180deg', 'sigma']
print('All required fields:', all(f in agent.history for f in required))

# 3. S3 method check
print('S3 upload method:', hasattr(agent, '_upload_checkpoint_to_s3'))
"
```
