#!/usr/bin/env python3
"""
Train ES-Multi: Optimize ONE design for ALL steering angles.

This baseline experiment demonstrates that a single static design cannot
effectively serve multiple steering angles. The optimizer will produce
a "compromise" design with mediocre performance at all angles.

Reward Function:
    R = Σ P_i − λ · Var(P_i)
    R = (P_0 + P_90 + P_180) − 0.5 · Var(P_0, P_90, P_180)
    
    Uses the SAME λ=0.5 as ES-Single for fair comparison.

Usage:
    python3 -u src/scripts/train_multi.py

Expected runtime: ~6-8 hours with n_jobs=16
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.es_multi_agent import ESMultiAgent, ES_MULTI_CONFIG

if __name__ == '__main__':
    lambda_var = ES_MULTI_CONFIG.get('lambda_variance', 0.5)
    
    print("=" * 80)
    print("ES-MULTI: Multi-Angle Optimization Baseline")
    print("=" * 80)
    print(f"Optimizing for angles: [0°, 90°, 180°]")
    print(f"Reward function: R = Σ P_i − λ · Var(P_i)  where λ={lambda_var}")
    print(f"                 (Same λ as ES-Single's crosstalk penalty)")
    print(f"Expected result: Compromise design with mediocre performance")
    print("=" * 80)
    
    # Check for S3 bucket
    s3_bucket = os.environ.get('S3_BUCKET')
    if s3_bucket:
        print(f"S3 uploads enabled: {s3_bucket}")
    else:
        print("S3 uploads disabled (no S3_BUCKET env var)")
    
    # Override config for AWS (r5.4xlarge has 16 vCPU)
    config = ES_MULTI_CONFIG.copy()
    config['n_jobs'] = 16  # Parallelize on 16 cores
    
    print(f"Parallel jobs: {config['n_jobs']}")
    
    # Create agent and train
    agent = ESMultiAgent(config=config)
    
    best_rho, best_reward = agent.train(verbose=True)
    
    # Save final results
    print(f"\nFinal best reward: {best_reward:.6e}")
    print(f"Results saved to: {agent.output_dir}")
    
    # Plot if not in AWS
    if not s3_bucket:
        try:
            agent.plot_history(save_path=str(agent.output_dir / "training_history.png"))
            print(f"Training history plot saved")
        except Exception as e:
            print(f"Could not save plot: {e}")
