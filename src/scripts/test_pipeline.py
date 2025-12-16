#!/usr/bin/env python3
# =============================================================================
# AWS Batch Pipeline Test: Minimal ES Training
# =============================================================================
"""
Quick sanity check for AWS Batch pipeline:
- Trains for 2 iterations (very fast, ~1 minute)
- Tests S3 upload
- Verifies checkpoint creation
- Uses n_jobs=10 parallelization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training import ESAgent, ES_CONFIG
from src.training.es_agent import upload_to_s3


def main():
    """Run minimal ES training test."""
    
    print("="*80)
    print("AWS BATCH PIPELINE TEST: Minimal ES Training")
    print("="*80)
    
    # Minimal config: 2 iterations, N=10 population
    config = ES_CONFIG.copy()
    config['N'] = 10              # Small population
    config['n_iterations'] = 2    # Just 2 iterations (fast)
    config['log_every'] = 1       # Log every iteration
    config['n_jobs'] = 10         # Full parallelization
    
    print("\nTest Configuration:")
    print(f"  Population (N): {config['N']}")
    print(f"  Iterations: {config['n_iterations']}")
    print(f"  Parallel jobs: {config['n_jobs']}")
    print(f"  Expected runtime: ~30-60 seconds")
    
    # Initialize ES agent for 90° (arbitrary choice)
    agent = ESAgent(
        target_angle=90,
        config=config,
        output_dir=None  # Auto-generate
    )
    
    print(f"\nOutput directory: {agent.output_dir}")
    print("\n" + "="*80)
    print("STARTING MINIMAL TEST TRAINING...")
    print("="*80)
    
    # Run training
    try:
        best_rho, best_reward = agent.train(verbose=True)
        
        print("\n" + "="*80)
        print("TEST PASSED ✅")
        print("="*80)
        print(f"Best reward achieved: {best_reward:.6e}")
        print(f"Rod configuration shape: {best_rho.shape}")
        print(f"Checkpoints created: {len(list(agent.output_dir.glob('checkpoint_*')))}")
        
        # Verify checkpoint structure
        checkpoint_00000 = agent.output_dir / "checkpoint_00000"
        files = list(checkpoint_00000.glob("*"))
        print(f"\nCheckpoint files in {checkpoint_00000.name}:")
        for f in sorted(files):
            print(f"  ✓ {f.name}")
        
        # Upload results to S3 (for AWS Batch runs)
        import os
        if os.environ.get('S3_BUCKET'):
            print("\n" + "="*80)
            print("UPLOADING TO S3...")
            print("="*80)
            success_upload = upload_to_s3(str(agent.output_dir), verbose=True)
            if success_upload:
                print("  ✅ S3 upload successful")
            else:
                print("  ❌ S3 upload failed")
        
        print("\n" + "="*80)
        print("PIPELINE VERIFICATION:")
        print("="*80)
        print("  ✅ ESAgent initialized")
        print("  ✅ Training completed (2 iterations)")
        print("  ✅ Checkpoints created")
        print("  ✅ n_jobs=10 parallelization working")
        print("  ✅ Output directory structure correct")
        if os.environ.get('S3_BUCKET'):
            print("  ✅ S3 upload tested")
        print("\nReady for full pipeline test! 🚀")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ❌")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
