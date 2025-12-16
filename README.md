# CS229 Beam Steering - Electromagnetic Optimization

High-speed Evolution Strategy training pipeline for optimizing plasma rod configurations to steer electromagnetic waves.

## Project Structure

```
cs229-beam-steering/
├── README.md                          # This file
├── Dockerfile                         # Docker image definition
├── .dockerignore                      # Docker build exclusions
├── requirements.txt                   # Python dependencies
├── 
├── docs/                              # Documentation
│   ├── DOCKER.md                      # Docker setup guide
│   ├── DESIGN_DECISIONS.md            # Physics & algorithm choices
│   ├── TRAINING_SCRIPTS.md            # Training script reference
│   ├── SPEED_OPTIMIZATION.md          # Performance optimization report
│   └── AWS_RESOURCES.md               # AWS infrastructure reference
│
├── config/                            # Configuration files
│   └── scripts/                       # AWS deployment scripts
│       ├── aws_setup.sh               # Initial AWS infrastructure
│       ├── phase3_aws_setup.sh        # Comprehensive setup
│       ├── phase3_aws_setup_optimized.sh  # Speed-optimized setup
│       └── update_compute_env.sh      # Update existing environment
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── constants.py                   # Simulation constants
│   ├── simulation.py                  # Physics simulation (ceviche)
│   ├── visualization.py               # Plotting utilities
│   │
│   ├── training/                      # ES algorithm
│   │   ├── __init__.py
│   │   └── es_agent.py                # ESAgent class with joblib parallelization
│   │
│   ├── scripts/                       # Training executables
│   │   ├── train_0deg.py              # Train for 0° target
│   │   ├── train_90deg.py             # Train for 90° target
│   │   └── train_180deg.py            # Train for 180° target
│   │
│   └── tests/                         # Unit tests
│       └── test_parallelization.py    # Verify parallelization correctness/speedup
│
├── notebooks/                         # Jupyter notebooks (exploration)
│   ├── beam_steering_dev.ipynb
│   ├── es_train_test.ipynb
│   └── test_es_outputs/               # Outputs from notebook tests
│
└── results/                           # Training outputs
    ├── es_outputs/                    # Local training runs
    │   ├── 0deg_{timestamp}/
    │   ├── 90deg_{timestamp}/
    │   └── 180deg_{timestamp}/
    └── aws_outputs/                   # (For Phase 6 - download from S3)
        ├── train_0deg_results/
        ├── train_90deg_results/
        └── train_180deg_results/
```

## Quick Start

### Local Training (CPU/GPU)

```bash
# Train for 0° target angle
python src/scripts/train_0deg.py

# Train for 90° target angle (uses n_jobs=10 parallelization)
python src/scripts/train_90deg.py

# Train for 180° target angle
python src/scripts/train_180deg.py
```

**Expected performance:**
- Serial (n_jobs=1): ~14 hours per run
- Parallel (n_jobs=10 on M3 Pro): ~4.4 hours per run (3.19× speedup)

### Docker (Local Testing)

```bash
# Build image
docker build -t cs229-trainer:latest .

# Test imports
docker run --rm --entrypoint python3 cs229-trainer:latest \
  -c "from src.training import ESAgent, ES_CONFIG; print('✅ OK')"

# Run training in container
docker run --rm -v $(pwd)/results/es_outputs:/app/results/es_outputs \
  cs229-trainer:latest src/scripts/train_90deg.py
```

### AWS Batch Deployment (Phase 4-5)

```bash
# Phase 4: Push to ECR
ECR_URI="745854319073.dkr.ecr.us-west-2.amazonaws.com/cs229-beam-steering-trainer"
docker tag cs229-trainer:latest $ECR_URI:latest
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_URI
docker push $ECR_URI:latest

# Phase 5: Submit jobs (instructions in AWS Batch console)
# Queue: cs229-beam-steering-queue
# Job Definition: cs229-beam-steering-train
# Override command: src/scripts/train_0deg.py (or train_90deg.py, train_180deg.py)
```

## Configuration

### ES Parameters
Edit `src/training/es_agent.py` or pass config dict to ESAgent:

```python
config = {
    'N': 100,              # Population size
    'n_iterations': 1000,  # Total iterations
    'sigma_0': 0.3,        # Initial perturbation
    'sigma_decay': 0.999,  # Decay per iteration
    'eta': 0.02,           # Adam learning rate
    'n_jobs': 10,          # Parallel jobs (joblib)
}
```

### Physical Parameters
- **Frequency**: 3.75 GHz (below plasma frequency → blocking)
- **Array**: 8×8 rods @ 2.1 cm spacing
- **Rod diameter**: 2 cm
- **Max plasma frequency**: 15 GHz (at 30V)
- **Control parameter ρ**: Normalized plasma frequency [0.0, 1.0]

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Parallelization** | 3.19× speedup (n_jobs=10) |
| **Per-job runtime** | ~4.4 hours (AWS r5.4xlarge) |
| **Total for 3 jobs** | ~13.5 hours (sequential) or ~4.5 hours (parallel) |
| **Cost per job** | $2.30 (On-Demand, no Spot) |
| **Total cost** | $6.90 for 3 jobs |
| **Interruption risk** | Zero (100% On-Demand instances) |

## AWS Infrastructure

**Status**: ✅ Fully configured and ready for Phase 4

- **Compute**: r5.4xlarge On-Demand instances (16 vCPU, 128 GB RAM)
- **Storage**: S3 bucket `cs229-beam-steering-results`
- **Registry**: ECR repo `cs229-beam-steering-trainer`
- **Orchestration**: AWS Batch with optimized Compute Environment & Job Queue
- **Pricing Model**: On-Demand only (no Spot interruption risk)

See `docs/AWS_RESOURCES.md` for detailed infrastructure reference.

## Documentation

- **`DOCKER.md`**: Docker image building and testing
- **`DESIGN_DECISIONS.md`**: Physics model and algorithm justification
- **`TRAINING_SCRIPTS.md`**: Training script usage and output format
- **`SPEED_OPTIMIZATION.md`**: Performance optimization details and tradeoffs
- **`AWS_RESOURCES.md`**: AWS infrastructure, account IDs, and connection info

## Development

### Running Tests

```bash
# Test parallelization (correctness, speedup, reproducibility)
python src/tests/test_parallelization.py
```

### Project Phases

- ✅ **Phase 1**: Parallelization implementation & testing
- ✅ **Phase 2**: Docker containerization
- ✅ **Phase 3**: AWS infrastructure setup (speed-optimized)
- ⏳ **Phase 4**: Push Docker image to ECR
- ⏳ **Phase 5**: Submit 3 training jobs to AWS Batch
- ⏳ **Phase 6**: Download results and analyze

## Author Notes

This project prioritizes **speed over cost**. All infrastructure choices (On-Demand instances, no Spot interruptions, joblib parallelization, optimal resource allocation) are optimized for guaranteed fast completion.

- **Expected result**: 3 trained electromagnetic beam steering designs in ~4-5 hours wall-clock time
- **Cost**: ~$7 total (irrelevant to project goals)
- **Reliability**: 100% On-Demand, zero interruption risk
