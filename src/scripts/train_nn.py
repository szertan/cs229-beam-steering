#!/usr/bin/env python3
"""
Training script for ES+NN beam steering agent.

The ES+NN agent optimizes a neural network that maps angle → design.
This allows generalization to unseen angles after training.

Key advantage: Model weights are saved at each checkpoint and can be
loaded later to evaluate at ANY angle (0°, 45°, 90°, 135°, 180°, etc.)

Reward Function (same as ES-Single, summed over training angles):
    R = Σ_θ [P_target(θ) - λ * Σ P_other(θ)]

Usage:
    python src/scripts/train_nn.py
    
Environment:
    S3_BUCKET: If set, uploads checkpoints to this S3 bucket
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.es_nn_agent import ESNNAgent, ES_NN_CONFIG


def main():
    print("=" * 70)
    print("ES+NN BEAM STEERING TRAINING")
    print("=" * 70)
    print()
    print("Agent: ES+NN (Neural Network optimized by Evolution Strategies)")
    print(f"Architecture: [sin(θ), cos(θ)] → 64 → 128 → 64 → sigmoid → 8×8")
    print(f"Training angles: {ES_NN_CONFIG['training_angles']}")
    print()
    print("Reward Function:")
    print(f"  R = Σ_θ [P_target(θ) - {ES_NN_CONFIG['lambda_crosstalk']} * Σ P_other(θ)]")
    print()
    print("Key Features:")
    print("  ✓ Model weights saved at every checkpoint")
    print("  ✓ Can evaluate at ANY angle after training (including unseen angles)")
    print("  ✓ sin/cos encoding for smooth angle interpolation")
    print()
    
    # Check S3
    s3_bucket = os.environ.get('S3_BUCKET')
    if s3_bucket:
        print(f"S3 Upload: ENABLED → s3://{s3_bucket}/")
    else:
        print("S3 Upload: DISABLED (set S3_BUCKET env var to enable)")
    print()
    print("=" * 70)
    print()
    
    # Create and train agent
    agent = ESNNAgent(seed=42)
    
    print(f"Neural Network Parameters: {agent.n_params:,}")
    print()
    
    # Train
    history = agent.train(verbose=True)
    
    # Summary
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Reward: {agent.best_reward:.4f}")
    print(f"Best Powers: {agent.best_powers}")
    print(f"Output Directory: {agent.output_dir}")
    print()
    print("To evaluate at unseen angles:")
    print("  from src.training.es_nn_agent import load_model")
    print(f"  model = load_model(Path('{agent.output_dir}/checkpoint_01000'))")
    print("  design_45deg = model.forward(45)  # Unseen angle!")
    print("=" * 70)


if __name__ == '__main__':
    main()
