# =============================================================================
# ES AGENT: Evolution Strategy Training for Beam Steering Optimization
# =============================================================================
"""
Evolution Strategy trainer for optimizing plasma rod configurations toward
target angles using the FDFD simulation pipeline.

Design Choices & Justifications
================================

1. REWARD FUNCTION (Target Tranmission Maximization + Crosstalk Penalty)
   - R(θ) = P_target − λ * Σ_{other_angles} P_other
   - Why: Directly optimizes signal-to-crosstalk trade-off; λ gives explicit control
   - λ (fixed): 0.5 (balances target power and crosstalk suppression)
   - Justification: Target power + crosstalk penalty is more interpretable and
     gives sharper gradients than normalized ratios, improving ES convergence.

2. POPULATION SIZE (N = 100)
   - Why: Balances stable gradient estimates with reasonable compute cost
   - Larger N → smoother gradients but more simulation evals per iteration
   - 100 is a proven sweet spot for 64-dimensional rod parameter space

3. PERTURBATION SCALE (σ₀ = 0.3, decay = 0.999 per iteration)
   - Why: σ₀=0.3 explores meaningful changes in rod parameter space
   - Decay 0.999 halves σ over ~693 iterations → sustained exploration then refinement
   - For 1000 iterations: gradual transition from exploration to exploitation

4. STEP SIZE OPTIMIZATION (Adam-on-ES)
   - Algorithm: Apply Adam optimizer to ES gradient estimates
   - Why: Per-parameter adaptive scaling, momentum, robust to noise
   - Base learning rate η = 0.02; β₁=0.9, β₂=0.999, ε=1e-8 (standard Adam defaults)
   - NO explicit crosstalk (λ) adaptation — keep λ=0.5 fixed throughout training
   - Justification: Adam removes need for manual α tuning; handles 64-D rod space
     with differing sensitivities; empirically robust with ES gradient estimates

5. RANKING SCHEME (Linear, Centered)
   - Map rewards to zero-mean ranks: w_i = rank_i − (N+1)/2
   - ES gradient: g ∝ (1/(N*σ)) * Σ_i w_i * ε_i
   - Why: Robust to absolute reward scale, standard in ES literature

6. POPULATION STRATEGY (Warm-start)
   - Retain best candidate between iterations; sample perturbations around it
   - Why: Preserves progress; σ decay provides exploration → exploitation transition
   - Mitigates local optima via multiple independent runs (user can run multiple seeds)

7. ITERATIONS & LOGGING
   - Total iterations: 1000 (chosen for AWS compute budget and design refinement)
   - Log every 10 iterations (rich dataset for later ES+NN research)
   - Checkpoint: metadata.json + best_rho.npy + best_eps_r.npy + best_Ez.npy + visualization

8. FREQUENCY (Fixed at 6 GHz)
   - Matches lab hardware and earlier analysis
   - Fixed for initial runs; can sweep later if broadband designs needed

9. PERMITTIVITY PARAMETERIZATION (ρ = normalized plasma frequency)
   - rho ∈ [0, 1] maps to omega_p = rho * OMEGA_P_MAX
   - Drude model: ε(ω) = 1 − (ω_p / ω)²
   - Why: Physically meaningful (maps to hardware voltage control); enables
     natural discretization; connects design to experimental implementation
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime
from joblib import Parallel, delayed

# Import simulation and constants
from ..simulation import (
    rods_to_permittivity,
    add_source_waveguide,
    add_receiver_waveguide,
    create_source,
    run_simulation,
    measure_power_at_receiver,
)
from ..constants import (
    N_RODS,
    FREQUENCY,
    RECEIVERS,
)


# =============================================================================
# ES HYPERPARAMETERS (from design choices above)
# =============================================================================

ES_CONFIG = {
    # Population & perturbation
    'N': 100,                          # Population size
    'sigma_0': 0.3,                    # Initial perturbation scale
    'sigma_decay': 0.999,              # Per-iteration decay of σ
    
    # Adam optimizer on ES gradient
    'eta': 0.02,                       # Base learning rate
    'beta_1': 0.9,                     # Adam momentum parameter
    'beta_2': 0.999,                   # Adam second moment decay
    'adam_eps': 1e-8,                  # Adam numerical stability
    'clip_norm': None,                 # Optional max update norm (None = no clipping)
    
    # Reward function
    'lambda_crosstalk': 0.5,           # Crosstalk penalty weight (fixed, not adaptive)
    
    # Iterations
    'n_iterations': 1000,              # Total training iterations
    'log_every': 10,                   # Save checkpoint every N iterations
    
    # Frequency
    'frequency_hz': FREQUENCY,         # Operating frequency (6 GHz)
    
    # Parallelization
    'n_jobs': 1,                       # Number of parallel jobs (1=serial, -1=all cores, 10=recommended for M3 Pro)
}


# =============================================================================
# ES AGENT CLASS
# =============================================================================

class ESAgent:
    """
    Evolution Strategy agent for beam steering optimization.
    
    Implements warm-start ES with:
    - Linear-ranked centered weights for gradient estimation
    - Adam optimizer for adaptive per-parameter step sizes
    - Efficient population sampling around best candidate
    - Comprehensive logging and checkpointing
    """
    
    def __init__(self, target_angle: int, config: Dict = None, output_dir: str = None):
        """
        Initialize ES agent for a target angle.
        
        Args:
            target_angle: Target receiver angle (0, 90, or 180 degrees)
            config: Dict of hyperparameters (default: ES_CONFIG)
            output_dir: Directory for checkpoints (default: ./es_outputs/{angle}_{timestamp})
        """
        if target_angle not in RECEIVERS:
            raise ValueError(f"Invalid target_angle={target_angle}. Valid: {list(RECEIVERS.keys())}")
        
        self.target_angle = target_angle
        self.config = config or ES_CONFIG.copy()
        self.n_jobs = self.config.get('n_jobs', 1)  # Extract n_jobs from config
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"es_outputs/{target_angle}deg_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parameters
        self.theta_best = np.random.uniform(0, 1, (N_RODS, N_RODS))  # Start random
        self.reward_best = -np.inf
        
        # Adam state
        self.m = np.zeros_like(self.theta_best)  # First moment
        self.v = np.zeros_like(self.theta_best)  # Second moment
        self.t_adam = 0  # Adam timestep counter
        
        # History for logging
        self.history = {
            'iteration': [],
            'reward_best': [],
            'reward_mean': [],
            'reward_std': [],
            'power_target': [],
            'power_other_sum': [],
            'power_0deg': [],      # Individual receiver powers for cross-angle analysis
            'power_90deg': [],
            'power_180deg': [],
            'crosstalk_ratio': [],
            'sigma': [],
            'eta_effective': [],  # Effective per-parameter lr from Adam
        }
        
        # Top-K best configurations tracking
        self.top_k = 10
        self.top_k_configs = []  # List of (reward, rho_array, iteration)
    
    def evaluate_configuration(self, rod_states: np.ndarray, verbose: bool = False) -> Tuple[float, Dict]:
        """
        Evaluate reward for a rod configuration.
        
        Args:
            rod_states: N_RODS × N_RODS array of ρ values ∈ [0, 1]
            verbose: If True, print simulation details
        
        Returns:
            (reward, powers_dict) where:
                reward = P_target − λ * Σ P_other
                powers_dict = {angle: power}
        """
        try:
            # Build permittivity map
            eps_r = rods_to_permittivity(rod_states)
            eps_r = add_source_waveguide(eps_r)
            eps_r = add_receiver_waveguide(eps_r, self.target_angle)
            
            # Run FDFD simulation
            Ez = run_simulation(eps_r, frequency=self.config['frequency_hz'], verbose=verbose)
            
            # Measure power at all receivers (for logging crosstalk)
            powers = {}
            for angle in RECEIVERS.keys():
                powers[angle] = measure_power_at_receiver(Ez, angle)
            
            # Compute reward: target power minus crosstalk penalty
            p_target = powers[self.target_angle]
            p_others = sum(powers[a] for a in powers if a != self.target_angle)
            reward = p_target - self.config['lambda_crosstalk'] * p_others
            
            return reward, powers
        
        except Exception as e:
            # If simulation fails, return very low reward
            print(f"Simulation failed for this configuration: {e}")
            return -1e10, {a: 0.0 for a in RECEIVERS.keys()}
    
    def compute_es_gradient(self, perturbations: np.ndarray, rewards: np.ndarray, sigma: float) -> np.ndarray:
        """
        Compute ES gradient estimate using centered linear ranking.
        
        Args:
            perturbations: N × 8 × 8 array of Gaussian perturbations
            rewards: Length-N array of reward evaluations
            sigma: Current perturbation scale
        
        Returns:
            gradient: 8×8 gradient estimate
        """
        N = len(rewards)
        
        # Rank rewards (ascending; rank 1 = worst, N = best)
        ranks = np.argsort(np.argsort(rewards)) + 1
        
        # Center ranks to have zero mean
        centered_ranks = ranks - (N + 1) / 2.0
        
        # ES gradient: g ∝ (1/(N*σ)) * Σ_i w_i * ε_i
        # Shape: (N, 8, 8) weighted by (N, 1, 1) -> average over N -> (8, 8)
        g = np.mean(centered_ranks[:, np.newaxis, np.newaxis] * perturbations, axis=0) / sigma
        
        return g
    
    def update_top_k(self, reward: float, rho: np.ndarray, iteration: int):
        """
        Track top-K best configurations found during training.
        
        Args:
            reward: Reward value for this configuration
            rho: 8×8 array of plasma frequency parameters
            iteration: Iteration number when found
        """
        self.top_k_configs.append({
            'reward': reward,
            'rho': rho.copy(),
            'iteration': iteration
        })
        
        # Sort by reward (descending) and keep top-k
        self.top_k_configs.sort(key=lambda x: x['reward'], reverse=True)
        self.top_k_configs = self.top_k_configs[:self.top_k]
    
    def get_top_k_summary(self):
        """
        Get summary of top-K configurations as a list of dicts.
        
        Returns:
            List of dicts with keys: rank, reward, iteration, rho_min, rho_max
        """
        summary = []
        for i, config in enumerate(self.top_k_configs, 1):
            summary.append({
                'rank': i,
                'reward': config['reward'],
                'iteration': config['iteration'],
                'rho_min': float(config['rho'].min()),
                'rho_max': float(config['rho'].max()),
                'rho_mean': float(config['rho'].mean()),
            })
        return summary
    
    def get_top_k_config(self, rank: int) -> np.ndarray:
        """
        Retrieve the rho configuration for a specific top-K rank (1-indexed).
        
        Args:
            rank: Rank number (1 = best, 10 = 10th best)
        
        Returns:
            8×8 rho array
        """
        if rank < 1 or rank > len(self.top_k_configs):
            raise ValueError(f"Rank must be in [1, {len(self.top_k_configs)}]")
        return self.top_k_configs[rank - 1]['rho']
    
    def print_top_k_summary(self):
        """
        Print a formatted summary of top-K configurations.
        """
        summary = self.get_top_k_summary()
        print("\n" + "="*80)
        print(f"TOP-{len(summary)} CONFIGURATIONS")
        print("="*80)
        print(f"{'Rank':<6} {'Reward':<15} {'Iteration':<12} {'ρ Range':<20} {'ρ Mean':<10}")
        print("-"*80)
        for item in summary:
            rho_range = f"[{item['rho_min']:.4f}, {item['rho_max']:.4f}]"
            print(f"{item['rank']:<6} {item['reward']:<15.6e} {item['iteration']:<12} {rho_range:<20} {item['rho_mean']:<10.4f}")
        print("="*80 + "\n")
    
    def adam_step(self, g: np.ndarray) -> np.ndarray:
        """
        Apply Adam optimizer to gradient estimate.
        
        Args:
            g: Gradient estimate (8×8 array)
        
        Returns:
            delta_theta: Parameter update (8×8 array)
        """
        self.t_adam += 1
        eta = self.config['eta']
        beta_1 = self.config['beta_1']
        beta_2 = self.config['beta_2']
        eps = self.config['adam_eps']
        
        # Update biased moments (both g and m, v are 8×8)
        self.m = beta_1 * self.m + (1 - beta_1) * g
        self.v = beta_2 * self.v + (1 - beta_2) * (g ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - beta_1 ** self.t_adam)
        v_hat = self.v / (1 - beta_2 ** self.t_adam)
        
        # Update
        delta_theta = eta * m_hat / (np.sqrt(v_hat) + eps)
        
        # Optional clipping
        if self.config['clip_norm'] is not None:
            norm = np.linalg.norm(delta_theta)
            if norm > self.config['clip_norm']:
                delta_theta = delta_theta / norm * self.config['clip_norm']
        
        return delta_theta
    
    def train(self, verbose: bool = True):
        """
        Run the ES training loop for n_iterations.
        
        Args:
            verbose: If True, print progress updates
        """
        N = self.config['N']
        n_iters = self.config['n_iterations']
        sigma_0 = self.config['sigma_0']
        sigma_decay = self.config['sigma_decay']
        log_every = self.config['log_every']
        
        if verbose:
            print(f"Starting ES training for target angle {self.target_angle}°")
            print(f"Population: N={N}, σ₀={sigma_0}, decay={sigma_decay}")
            print(f"Adam: η={self.config['eta']}, β₁={self.config['beta_1']}, β₂={self.config['beta_2']}")
            print(f"Reward: R = P_target − λ * Σ P_other (λ={self.config['lambda_crosstalk']})")
            print(f"Output: {self.output_dir}")
        
        # Evaluate initial configuration
        reward_init, powers_init = self.evaluate_configuration(self.theta_best, verbose=False)
        self.reward_best = reward_init
        
        for iteration in range(n_iters):
            # Current perturbation scale
            sigma = sigma_0 * (sigma_decay ** iteration)
            
            # Progress indicator
            if verbose:
                progress_pct = 100 * (iteration + 1) / n_iters
                print(f"\n[{iteration+1:4d}/{n_iters}] ({progress_pct:5.1f}%) Sampling population...", end='', flush=True)
            
            # Sample perturbations and evaluate
            eps_population = np.random.randn(N, N_RODS, N_RODS)  # N × 8 × 8
            
            if verbose:
                print(f"  Evaluating population ({N} configurations)", end='', flush=True)
            
            # Parallel evaluation using joblib
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self.evaluate_configuration)(
                    self.theta_best + sigma * eps_population[i],
                    verbose=False
                )
                for i in range(N)
            )
            
            # Unpack results
            rewards = np.array([r[0] for r in results])
            all_powers = [r[1] for r in results]
            
            if verbose:
                print(" Done")
            
            # Find best in this iteration
            best_idx = np.argmax(rewards)
            reward_iter_best = rewards[best_idx]
            powers_iter_best = all_powers[best_idx]
            
            # Compute ES gradient
            if verbose:
                print(f"  → Computing ES gradient (rank-weighted perturbations)...", end='', flush=True)
            g = self.compute_es_gradient(eps_population, rewards, sigma)
            if verbose:
                print(f" ||g||={np.linalg.norm(g):.4e}")
            
            # Adam update
            if verbose:
                print(f"  → Applying Adam update...", end='', flush=True)
            delta_theta = self.adam_step(g)
            if verbose:
                print(f" ||Δθ||={np.linalg.norm(delta_theta):.4e}")
            
            self.theta_best = self.theta_best + delta_theta
            
            # Clip to [0, 1] (valid ρ range)
            self.theta_best = np.clip(self.theta_best, 0, 1)
            
            # Re-evaluate after update
            reward_after, powers_after = self.evaluate_configuration(self.theta_best, verbose=False)
            
            # Track if improvement
            if reward_after > self.reward_best:
                self.reward_best = reward_after
                powers_best = powers_after
            else:
                powers_best = powers_iter_best
            
            # Update top-K tracking (always track this iteration's best)
            self.update_top_k(reward_after, self.theta_best, iteration)
            
            # Logging
            p_target = powers_best[self.target_angle]
            p_others = sum(powers_best[a] for a in powers_best if a != self.target_angle)
            crosstalk_ratio = p_others / max(1e-12, p_target)
            
            self.history['iteration'].append(iteration)
            self.history['reward_best'].append(self.reward_best)
            self.history['reward_mean'].append(np.mean(rewards))
            self.history['reward_std'].append(np.std(rewards))
            self.history['power_target'].append(p_target)
            self.history['power_other_sum'].append(p_others)
            self.history['power_0deg'].append(float(powers_best.get(0, 0)))
            self.history['power_90deg'].append(float(powers_best.get(90, 0)))
            self.history['power_180deg'].append(float(powers_best.get(180, 0)))
            self.history['crosstalk_ratio'].append(crosstalk_ratio)
            self.history['sigma'].append(sigma)
            
            # Compute effective learning rate (norm of Adam update)
            eta_eff = np.linalg.norm(delta_theta)
            self.history['eta_effective'].append(eta_eff)
            
            # Summary output
            if verbose:
                improvement = "↑" if reward_after > self.reward_best else "→"
                print(f"  Summary: R_best={self.reward_best:.3e} {improvement} | "
                      f"R_pop={np.mean(rewards):.3e}±{np.std(rewards):.2e} | "
                      f"P_target={p_target:.3e} | "
                      f"crosstalk={crosstalk_ratio:.3f} | "
                      f"σ={sigma:.4e}")
            
            # Save checkpoint and upload to S3
            if (iteration + 1) % log_every == 0 or iteration == n_iters - 1:
                if verbose:
                    print(f"  ✓ Checkpoint saved: checkpoint_{iteration:05d}/")
                self.save_checkpoint(iteration)
                
                # Upload to S3 after every checkpoint (if S3_BUCKET is set)
                s3_bucket = os.environ.get('S3_BUCKET')
                if s3_bucket:
                    if verbose:
                        print(f"  ⬆ Uploading to S3...", end='', flush=True)
                    try:
                        self._upload_checkpoint_to_s3(iteration, s3_bucket)
                        if verbose:
                            print(f" Done")
                    except Exception as e:
                        if verbose:
                            print(f" Failed: {e}")
        
        if verbose:
            print(f"\n" + "="*80)
            print(f"Training complete for target angle {self.target_angle}°")
            print(f"="*80)
            print(f"Final best reward: {self.reward_best:.6e}")
            print(f"Final rod config range: [{self.theta_best.min():.4f}, {self.theta_best.max():.4f}]")
            print(f"Output directory: {self.output_dir}")
            print(f"Total iterations: {n_iters} | Checkpoints saved: {(n_iters + log_every - 1) // log_every}")
            print(f"="*80 + "\n")
        
        return self.theta_best, self.reward_best
    
    def save_checkpoint(self, iteration: int):
        """Save checkpoint (metadata, parameters, Ez, permittivity, images)."""
        checkpoint_dir = self.output_dir / f"checkpoint_{iteration:05d}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Metadata
        metadata = {
            'iteration': iteration,
            'target_angle': self.target_angle,
            'timestamp': datetime.now().isoformat(),
            'frequency_hz': self.config['frequency_hz'],
            'hyperparameters': self.config,
            'reward_best': float(self.reward_best),
            'history_latest': {
                'reward_mean': float(self.history['reward_mean'][-1]) if self.history['reward_mean'] else None,
                'power_target': float(self.history['power_target'][-1]) if self.history['power_target'] else None,
                'crosstalk_ratio': float(self.history['crosstalk_ratio'][-1]) if self.history['crosstalk_ratio'] else None,
            }
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Best rod configuration
        np.save(checkpoint_dir / 'best_rho.npy', self.theta_best)
        
        # Full permittivity map
        eps_r = rods_to_permittivity(self.theta_best)
        eps_r = add_source_waveguide(eps_r)
        eps_r = add_receiver_waveguide(eps_r, self.target_angle)
        np.save(checkpoint_dir / 'best_eps_r.npy', eps_r)
        
        # Electric field
        Ez = run_simulation(eps_r, frequency=self.config['frequency_hz'], verbose=False)
        np.save(checkpoint_dir / 'best_Ez.npy', Ez)
        
        # Save top-K configurations summary
        top_k_summary = self.get_top_k_summary()
        with open(checkpoint_dir / 'top_k_summary.json', 'w') as f:
            json.dump(top_k_summary, f, indent=2)
        
        # Save top-K rho configurations (as individual .npy files)
        top_k_dir = checkpoint_dir / 'top_k_configs'
        top_k_dir.mkdir(exist_ok=True)
        for config in self.top_k_configs:
            rank = len([c for c in self.top_k_configs if c['reward'] > config['reward']]) + 1
            np.save(top_k_dir / f'rank_{rank:02d}_rho.npy', config['rho'])
        
        # Save full training history as JSON (for easy analysis without parsing all checkpoints)
        # Convert numpy types to native Python for JSON serialization
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
        
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def _upload_checkpoint_to_s3(self, iteration: int, s3_bucket: str):
        """
        Upload a single checkpoint directory to S3.
        
        Args:
            iteration: Checkpoint iteration number
            s3_bucket: S3 bucket name
        """
        import boto3
        
        checkpoint_dir = self.output_dir / f"checkpoint_{iteration:05d}"
        s3_prefix = self.output_dir.name  # e.g., "90deg_20251214_120000"
        
        s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
        
        # Upload all files in the checkpoint directory
        for file_path in checkpoint_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.output_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                s3_client.upload_file(str(file_path), s3_bucket, s3_key)
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history (requires matplotlib).
        
        Args:
            save_path: If provided, save figure to this path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping history plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        iter_list = self.history['iteration']
        
        # Reward evolution
        axes[0, 0].plot(iter_list, self.history['reward_best'], label='Best', linewidth=2)
        axes[0, 0].plot(iter_list, self.history['reward_mean'], label='Mean', linewidth=2)
        axes[0, 0].fill_between(iter_list,
                                 np.array(self.history['reward_mean']) - np.array(self.history['reward_std']),
                                 np.array(self.history['reward_mean']) + np.array(self.history['reward_std']),
                                 alpha=0.2)
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward Evolution')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('symlog')  # Handle potentially negative rewards
        
        # Power distribution
        axes[0, 1].plot(iter_list, self.history['power_target'], label='Target', linewidth=2)
        axes[0, 1].plot(iter_list, self.history['power_other_sum'], label='Other (sum)', linewidth=2)
        axes[0, 1].set_ylabel('Power (|Ez|²)')
        axes[0, 1].set_title('Power at Receivers')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Crosstalk ratio
        axes[1, 0].plot(iter_list, self.history['crosstalk_ratio'], linewidth=2, color='red')
        axes[1, 0].axhline(y=0.1, color='green', linestyle='--', label='Target (10%)')
        axes[1, 0].set_ylabel('Crosstalk Ratio')
        axes[1, 0].set_title('Crosstalk Metric: P_other / P_target')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Perturbation scale (σ) and effective learning rate
        axes[1, 1].plot(iter_list, self.history['sigma'], label='σ (perturbation)', linewidth=2)
        axes[1, 1].plot(iter_list, self.history['eta_effective'], label='η_eff (Adam step norm)', linewidth=2)
        axes[1, 1].set_ylabel('Scale')
        axes[1, 1].set_title('Exploration & Step Size')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved history plot: {save_path}")
        else:
            plt.show()


# =============================================================================
# S3 Upload Functions
# =============================================================================

def upload_to_s3(local_dir: str, s3_bucket: str = None, s3_prefix: str = None, 
                 verbose: bool = True) -> bool:
    """
    Upload training results directory to S3.
    
    Args:
        local_dir: Local directory path containing training outputs
        s3_bucket: S3 bucket name (default: from S3_BUCKET env var)
        s3_prefix: S3 key prefix (default: derived from local_dir name)
        verbose: Print progress
    
    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("ERROR: boto3 not installed. Cannot upload to S3.")
        return False
    
    # Get bucket from environment if not provided
    if s3_bucket is None:
        s3_bucket = os.environ.get('S3_BUCKET', 'cs229-beam-steering-results')
    
    # Default prefix is the directory name
    local_path = Path(local_dir)
    if s3_prefix is None:
        s3_prefix = local_path.name
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"UPLOADING RESULTS TO S3")
        print(f"{'='*80}")
        print(f"  Local directory: {local_path}")
        print(f"  S3 destination:  s3://{s3_bucket}/{s3_prefix}/")
    
    try:
        s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
        
        # Count files to upload
        files_to_upload = list(local_path.rglob('*'))
        files_to_upload = [f for f in files_to_upload if f.is_file()]
        
        if verbose:
            print(f"  Files to upload: {len(files_to_upload)}")
        
        uploaded_count = 0
        for file_path in files_to_upload:
            # Compute S3 key
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{relative_path}"
            
            # Upload file
            s3_client.upload_file(str(file_path), s3_bucket, s3_key)
            uploaded_count += 1
            
            if verbose and uploaded_count % 10 == 0:
                print(f"    Uploaded {uploaded_count}/{len(files_to_upload)} files...")
        
        if verbose:
            print(f"\n  ✅ Successfully uploaded {uploaded_count} files to S3")
            print(f"  📦 Results available at: s3://{s3_bucket}/{s3_prefix}/")
            print(f"{'='*80}\n")
        
        return True
        
    except ClientError as e:
        print(f"\n  ❌ S3 upload failed: {e}")
        return False
    except Exception as e:
        print(f"\n  ❌ Unexpected error during S3 upload: {e}")
        return False


def download_from_s3(s3_bucket: str, s3_prefix: str, local_dir: str, 
                     verbose: bool = True) -> bool:
    """
    Download training results from S3 to local directory.
    
    Args:
        s3_bucket: S3 bucket name
        s3_prefix: S3 key prefix (e.g., '90deg_20251214_120000')
        local_dir: Local directory to download to
        verbose: Print progress
    
    Returns:
        True if download succeeded, False otherwise
    
    Example:
        download_from_s3('cs229-beam-steering-results', '90deg_20251214_120000', 
                        './results/aws_outputs/90deg_run1')
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("ERROR: boto3 not installed. Cannot download from S3.")
        return False
    
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"DOWNLOADING RESULTS FROM S3")
        print(f"{'='*80}")
        print(f"  S3 source:       s3://{s3_bucket}/{s3_prefix}/")
        print(f"  Local directory: {local_path}")
    
    try:
        s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
        
        # List all objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
        
        downloaded_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                s3_key = obj['Key']
                
                # Compute local file path
                relative_key = s3_key[len(s3_prefix):].lstrip('/')
                if not relative_key:  # Skip the prefix itself if it's a "directory"
                    continue
                    
                local_file = local_path / relative_key
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                s3_client.download_file(s3_bucket, s3_key, str(local_file))
                downloaded_count += 1
                
                if verbose and downloaded_count % 10 == 0:
                    print(f"    Downloaded {downloaded_count} files...")
        
        if verbose:
            print(f"\n  ✅ Successfully downloaded {downloaded_count} files")
            print(f"  📁 Results saved to: {local_path}")
            print(f"{'='*80}\n")
        
        return True
        
    except ClientError as e:
        print(f"\n  ❌ S3 download failed: {e}")
        return False
    except Exception as e:
        print(f"\n  ❌ Unexpected error during S3 download: {e}")
        return False


def list_s3_results(s3_bucket: str = None, verbose: bool = True) -> list:
    """
    List all training results available in S3.
    
    Args:
        s3_bucket: S3 bucket name (default: cs229-beam-steering-results)
        verbose: Print results
    
    Returns:
        List of result prefixes (e.g., ['0deg_20251214_120000', '90deg_20251214_120000'])
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("ERROR: boto3 not installed.")
        return []
    
    if s3_bucket is None:
        s3_bucket = os.environ.get('S3_BUCKET', 'cs229-beam-steering-results')
    
    try:
        s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
        
        # List top-level prefixes (training runs)
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Delimiter='/')
        
        prefixes = []
        if 'CommonPrefixes' in response:
            for prefix_obj in response['CommonPrefixes']:
                prefix = prefix_obj['Prefix'].rstrip('/')
                prefixes.append(prefix)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"AVAILABLE TRAINING RESULTS IN S3")
            print(f"{'='*80}")
            print(f"  Bucket: s3://{s3_bucket}/")
            print(f"\n  Training runs found: {len(prefixes)}")
            for p in sorted(prefixes):
                print(f"    - {p}")
            print(f"{'='*80}\n")
        
        return prefixes
        
    except ClientError as e:
        print(f"ERROR: Could not list S3 bucket: {e}")
        return []


# =============================================================================
# Convenience Functions
# =============================================================================

def run_es_training(target_angle: int, n_iterations: int = 1000, 
                    config: Dict = None, output_dir: str = None,
                    verbose: bool = True, upload_s3: bool = False) -> Tuple[np.ndarray, float]:
    """
    Convenience function to run ES training end-to-end.
    
    Args:
        target_angle: Target receiver angle (0, 90, or 180)
        n_iterations: Number of iterations (default: 1000)
        config: Custom config dict (default: ES_CONFIG)
        output_dir: Output directory for checkpoints
        verbose: Print progress
        upload_s3: If True, upload results to S3 after training
    
    Returns:
        (best_rho, best_reward): Optimized rod configuration and final reward
    """
    cfg = config or ES_CONFIG.copy()
    cfg['n_iterations'] = n_iterations
    
    agent = ESAgent(target_angle=target_angle, config=cfg, output_dir=output_dir)
    best_rho, best_reward = agent.train(verbose=verbose)
    
    # Plot and save
    plot_path = agent.output_dir / 'training_history.png'
    agent.plot_history(save_path=str(plot_path))
    
    # Upload to S3 if requested
    if upload_s3:
        upload_to_s3(str(agent.output_dir), verbose=verbose)
    
    return best_rho, best_reward


# =============================================================================
# Module exports
# =============================================================================
__all__ = [
    'ESAgent',
    'run_es_training',
    'ES_CONFIG',
    'upload_to_s3',
    'download_from_s3',
    'list_s3_results',
]
