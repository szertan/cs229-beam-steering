#!/usr/bin/env python3
# =============================================================================
# ES Training Script: 0° Beam Steering (Bottom Receiver)
# =============================================================================
"""
Evolution Strategy training for 0° target angle (bottom receiver).

This script trains an ES agent to steer electromagnetic waves toward the 
bottom receiver (0°) while suppressing crosstalk to 90° and 180° receivers.

Configuration:
    - Target angle: 0°
    - Population size: N=100
    - Iterations: 1000
    - Perturbation scale: σ₀=0.3, decay=0.999/iteration
    - Adam optimizer: η=0.02, β₁=0.9, β₂=0.999

Output:
    - Checkpoints saved to ./results/es_outputs/0deg_{timestamp}/
    - Each checkpoint contains: metadata.json, best_rho.npy, best_eps_r.npy, best_Ez.npy
    - Top-10 configurations saved to top_k_configs/ subdirectory
    - Training history and plots in checkpoint_* directories
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training import ESAgent, ES_CONFIG
from src.training.es_agent import upload_to_s3


def main():
    """Run ES training for 0° target angle."""
    
    print("="*80)
    print("ES TRAINING: 0° BEAM STEERING (BOTTOM RECEIVER)")
    print("="*80)
    
    # Production configuration: N=100, 1000 iterations
    config = ES_CONFIG.copy()
    config['N'] = 100          # Population size
    config['n_iterations'] = 1000  # Total training iterations
    config['log_every'] = 10   # Save checkpoint every 10 iterations
    config['n_jobs'] = 16      # Parallelize on 16 cores (r5.4xlarge has 16 vCPU)
    
    print("\nConfiguration:")
    print(f"  Target angle: 0° (bottom receiver)")
    print(f"  Population size (N): {config['N']}")
    print(f"  Total iterations: {config['n_iterations']}")
    print(f"  Initial perturbation (σ₀): {config['sigma_0']}")
    print(f"  Perturbation decay: {config['sigma_decay']}/iteration")
    print(f"  Adam learning rate (η): {config['eta']}")
    print(f"  Reward function: R = P_target − λ * Σ P_other (λ={config['lambda_crosstalk']})")
    print(f"  Operating frequency: {config['frequency_hz']/1e9} GHz")
    
    # Initialize ES agent for 0° target
    agent = ESAgent(
        target_angle=0,
        config=config,
        output_dir=None  # Auto-generate: results/es_outputs/0deg_{timestamp}
    )
    
    print(f"\nOutput directory: {agent.output_dir}")
    print("\n" + "="*80)
    print("STARTING TRAINING...")
    print("="*80)
    
    # Run training
    best_rho, best_reward = agent.train(verbose=True)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best reward achieved: {best_reward:.6e}")
    print(f"Rod configuration shape: {best_rho.shape}")
    print(f"Rod ρ range: [{best_rho.min():.4f}, {best_rho.max():.4f}]")
    print(f"\nCheckpoints saved to: {agent.output_dir}")
    print(f"Total checkpoint directories: {len(list(agent.output_dir.glob('checkpoint_*')))}")
    
    # Print top-10 summary
    print("\n" + "="*80)
    print("TOP-10 BEST CONFIGURATIONS")
    print("="*80)
    agent.print_top_k_summary()
    
    # Plot and save training history
    print("Saving training history plot...")
    history_path = agent.output_dir / 'final_training_history.png'
    agent.plot_history(save_path=str(history_path))
    print(f"✓ History plot saved to: {history_path}")
    
    # Final S3 upload of history plot (checkpoints already uploaded during training)
    import os
    if os.environ.get('S3_BUCKET'):
        print("Uploading final training history plot to S3...")
        upload_to_s3(str(agent.output_dir), verbose=False)
        print(f"✓ All results uploaded to s3://{os.environ['S3_BUCKET']}/{agent.output_dir.name}/")
    
    print("\n" + "="*80)
    print("✓ 0° training complete. Ready for analysis or ES+NN training.")
    print("="*80)


if __name__ == '__main__':
    main()
