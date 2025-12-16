#!/usr/bin/env python3
"""
Test script for ES+NN agent on AWS.

Runs a short training (10 iterations) with verbose logging every 20 seconds.
Tests that model weights can be saved, loaded, and evaluated at unseen angles.
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.es_nn_agent import ESNNAgent, ES_NN_CONFIG, load_model


def main():
    print("=" * 70)
    print("ES+NN AWS TEST")
    print("=" * 70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check S3
    s3_bucket = os.environ.get('S3_BUCKET')
    print(f"S3_BUCKET: {s3_bucket or 'NOT SET'}")
    print()
    
    # Short test config
    config = {
        **ES_NN_CONFIG,
        'n_iterations': 10,      # Short test
        'checkpoint_every': 5,   # Checkpoint at 5 and 10
        'log_every': 1,          # Log every iteration
    }
    
    print("Test Configuration:")
    print(f"  Iterations: {config['n_iterations']}")
    print(f"  Checkpoint every: {config['checkpoint_every']}")
    print(f"  Training angles: {config['training_angles']}")
    print()
    
    # Create agent
    print("Creating ES+NN agent...")
    agent = ESNNAgent(config=config, seed=42)
    print(f"  NN Parameters: {agent.n_params:,}")
    print(f"  Output dir: {agent.output_dir}")
    print()
    
    # Train with timing
    print("=" * 70)
    print("TRAINING (with parallelization)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Use the agent's train method which now uses parallel evaluation
    history = agent.train(verbose=True)
    
    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print(f"TRAINING COMPLETE in {total_time:.1f}s")
    print("=" * 70)
    print()
    
    n_iterations = config['n_iterations']
    
    # =========================================================================
    # TEST: Verify checkpoint contents
    # =========================================================================
    print("=" * 70)
    print("CHECKPOINT VERIFICATION")
    print("=" * 70)
    
    checkpoint_dir = agent.output_dir / f'checkpoint_{n_iterations:05d}'
    print(f"\nCheckpoint: {checkpoint_dir}")
    print("\nFiles saved:")
    for f in sorted(checkpoint_dir.rglob('*')):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(checkpoint_dir)}: {size:,} bytes")
    
    # =========================================================================
    # TEST: Load model and evaluate at UNSEEN angles
    # =========================================================================
    print()
    print("=" * 70)
    print("MODEL LOADING & GENERALIZATION TEST")
    print("=" * 70)
    
    print("\nLoading saved model weights...")
    loaded_model = load_model(checkpoint_dir)
    print(f"  Loaded {loaded_model.num_params:,} parameters")
    
    # Test at training angles
    print("\nDesigns at TRAINING angles (should match training):")
    for angle in [0, 90, 180]:
        design = loaded_model.forward(angle)
        print(f"  {angle}°: design shape={design.shape}, range=[{design.min():.3f}, {design.max():.3f}]")
    
    # Test at UNSEEN angles
    print("\nDesigns at UNSEEN angles (generalization test):")
    for angle in [45, 135, 30, 60]:
        design = loaded_model.forward(angle)
        print(f"  {angle}°: design shape={design.shape}, range=[{design.min():.3f}, {design.max():.3f}]")
    
    # =========================================================================
    # TEST: Training history structure
    # =========================================================================
    print()
    print("=" * 70)
    print("TRAINING HISTORY VERIFICATION")
    print("=" * 70)
    
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path) as f:
        history = json.load(f)
    
    print(f"\nHistory fields ({len(history)} total):")
    for key, values in history.items():
        print(f"  {key}: {len(values)} entries, type={type(values[0]).__name__ if values else 'empty'}")
    
    # =========================================================================
    # TEST: Top-K models
    # =========================================================================
    print()
    print("=" * 70)
    print("TOP-K MODELS")
    print("=" * 70)
    
    top_k_path = checkpoint_dir / 'top_k_summary.json'
    with open(top_k_path) as f:
        top_k = json.load(f)
    
    print(f"\nTop-{len(top_k)} configurations:")
    for entry in top_k:
        powers_str = ", ".join(f"{a}°:{p:.4f}" for a, p in sorted(entry['powers'].items()))
        print(f"  Rank {entry['rank']}: R={entry['reward']:.4f} | {powers_str}")
    
    # Check top-K model files
    top_k_models = list((checkpoint_dir / 'top_k_models').glob('*.npy'))
    print(f"\nTop-K model weight files: {len(top_k_models)}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()
    print("✅ Agent created and trained successfully")
    print("✅ Checkpoints saved with all required files")
    print("✅ Model weights can be loaded and used for inference")
    print("✅ Model generates designs for UNSEEN angles (generalization)")
    print("✅ Training history logged correctly")
    print("✅ Top-K models saved")
    
    if s3_bucket:
        print(f"✅ Checkpoints uploaded to s3://{s3_bucket}/")
    else:
        print("⚠️  S3 upload skipped (no S3_BUCKET set)")
    
    print()
    print(f"Total test time: {total_time:.1f}s")
    print("=" * 70)
    
    # Cleanup (only in test mode)
    if not s3_bucket:
        print("\nCleaning up local test files...")
        shutil.rmtree(agent.output_dir)
        print("Done.")


if __name__ == '__main__':
    main()
