# Docker Setup for CS229 Beam Steering ES Training

## Overview
Docker containerization for the Evolution Strategy training pipeline enables reproducible execution on AWS Batch without environment-specific issues.

## Files Created

### 1. `Dockerfile`
- **Base**: `python:3.11-slim` (minimal, 266MB image)
- **Features**:
  - System dependencies: `build-essential`, `ca-certificates`
  - Python packages: numpy, scipy, matplotlib, ceviche, joblib, boto3
  - ENTRYPOINT: `python3 -u` (allows script override)
  - Default CMD: `src/train_90deg.py` (can be overridden)

### 2. `.dockerignore`
Excludes unnecessary files from image:
- Development artifacts: `.git`, `__pycache__`, `.venv`, `.pytest_cache`
- Training outputs: `es_outputs/`, `test_*`, `outputs/`
- Documentation: `*.md`, `*.ipynb`, `img/`, `slides/`, `notes/`
- AWS setup scripts (will be managed separately in Phase 3)

### 3. `requirements.txt`
Core dependencies:
- **Simulation**: ceviche==0.1.3, autograd==1.8.0
- **Numerics**: numpy==2.0.2, scipy==1.13.1, matplotlib==3.9.4
- **Parallelization**: joblib==1.5.2
- **AWS**: boto3>=1.26.0 (for S3 upload support)
- **Utilities**: python-dateutil, pillow

## Building the Image

```bash
cd /Users/sze/project-lorentz/cs229-beam-steering
docker build -t cs229-trainer:latest .
```

**Result**: 266MB image named `cs229-trainer:latest`

## Testing Locally

### Test 1: Import validation
```bash
docker run --rm --entrypoint python3 cs229-trainer:latest \
  -c "from src.training import ESAgent, ES_CONFIG; print('✅ Imports OK')"
```

### Test 2: Run training
```bash
docker run --rm \
  -v $(pwd)/results/es_outputs:/app/results/es_outputs \
  --entrypoint python3 \
  cs229-trainer:latest \
  src/scripts/train_90deg.py
```

### Test 3: Custom configuration
```bash
docker run --rm --entrypoint python3 cs229-trainer:latest << 'EOF'
from src.training import ESAgent, ES_CONFIG

config = ES_CONFIG.copy()
config['N'] = 20
config['n_iterations'] = 5
config['n_jobs'] = 4

agent = ESAgent(target_angle=90, config=config)
agent.train(verbose=True)
EOF
```

## Usage for AWS Batch

In AWS Batch Job Definition, override the default CMD:

```json
{
  "containerProperties": {
    "image": "123456789.dkr.ecr.us-west-2.amazonaws.com/cs229-trainer:latest",
    "entryPoint": ["python3", "-u"],
    "command": ["src/scripts/train_90deg.py"],
    "environment": [
      {"name": "PYTHONUNBUFFERED", "value": "1"}
    ]
  }
}
```

This allows AWS Batch to override the command while keeping the container lightweight and reusable.

## Image Size Breakdown

- Base Python: ~200MB
- Dependencies: ~66MB
- Project code: <1MB
- **Total**: 266MB

## Testing Status

- ✅ Image builds successfully
- ✅ Imports work in container
- ✅ ESAgent initializes correctly
- ✅ Ready for AWS deployment
