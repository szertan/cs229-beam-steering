# =============================================================================
# ES MULTI-ANGLE AGENT: Optimize ONE Design for ALL Angles
# =============================================================================
"""
ES-Multi: Evolution Strategy that tries to find a single rod configuration
that performs well for ALL three steering angles (0°, 90°, 180°) simultaneously.

Purpose:
    This is a BASELINE experiment to demonstrate that a single static design
    cannot effectively serve multiple steering angles. It motivates the need
    for angle-conditional design generation (ES+NN).

Reward Function:
    R = Σ P_i  −  λ · Var(P_i)
    R = (P_0 + P_90 + P_180) − λ · Var(P_0, P_90, P_180)
    
    where:
        Σ P_i = P_0 + P_90 + P_180           (total power to all angles)
        Var(P_i) = (1/3) Σ (P_i - P̄)²        (variance of powers)
        P̄ = (P_0 + P_90 + P_180) / 3         (mean power)
        λ = 0.5                              (balance penalty weight)
    
    This "Sum − Variance Penalty" formulation:
    - Maximizes total power transmitted to all three angles
    - Penalizes imbalance between angles (via variance term)
    - Uses the SAME λ=0.5 as ES-Single for fair comparison

Comparison to ES-Single:
    ┌─────────────┬────────────────────────────────────┬──────────────────────────┐
    │ Method      │ Reward Function                    │ Interpretation           │
    ├─────────────┼────────────────────────────────────┼──────────────────────────┤
    │ ES-Single   │ R = P_target − λ · Σ P_other       │ Max target, penalize     │
    │             │                                    │ leakage to other angles  │
    ├─────────────┼────────────────────────────────────┼──────────────────────────┤
    │ ES-Multi    │ R = Σ P_i − λ · Var(P_i)           │ Max total, penalize      │
    │             │                                    │ imbalance between angles │
    └─────────────┴────────────────────────────────────┴──────────────────────────┘
    
    Both use λ=0.5, making the comparison fair and interpretable.

Why Sum − Variance (not min)?
    1. Parallel structure to ES-Single (both: maximize X − λ·penalty)
    2. Same hyperparameter λ=0.5 (no new tuning required)
    3. Smooth gradients (all angles contribute signal, unlike min)
    4. Interpretable: "total power minus imbalance penalty"
    5. Gives ES-Multi its best shot with smooth optimization

Expected Result:
    ES-Multi will produce a mediocre "compromise" design that:
    - Doesn't steer strongly to any single angle
    - Has roughly equal (but low) power at all exits
    - Performs worse at each angle than the angle-specific ES-Single
    
    This demonstrates that angle-conditional design is necessary.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from datetime import datetime
from joblib import Parallel, delayed

# Import simulation and constants
from ..simulation import (
    rods_to_permittivity,
    add_source_waveguide,
    add_receiver_waveguide,
    run_simulation,
    measure_power_at_receiver,
)
from ..constants import (
    N_RODS,
    FREQUENCY,
    RECEIVERS,
    TRAINING_ANGLES,
)

# Import ES config from es_agent
from .es_agent import ES_CONFIG


# =============================================================================
# ES MULTI-ANGLE CONFIG
# =============================================================================

ES_MULTI_CONFIG = ES_CONFIG.copy()
# Use SAME λ=0.5 as ES-Single for fair comparison
# In ES-Multi: λ penalizes variance (imbalance) instead of crosstalk
ES_MULTI_CONFIG['lambda_variance'] = 0.5  # Renamed for clarity, same value as lambda_crosstalk


# =============================================================================
# ES MULTI-ANGLE AGENT CLASS
# =============================================================================

class ESMultiAgent:
    """
    Evolution Strategy agent for multi-angle optimization.
    
    Finds ONE design that maximizes total power to all angles while penalizing
    imbalance. Uses R = Σ P_i − λ · Var(P_i) with λ=0.5.
    
    This is a baseline to show that single designs cannot effectively steer
    to multiple angles, motivating angle-conditional design generation.
    """
    
    def __init__(self, config: Dict = None, output_dir: str = None):
        """
        Initialize ES-Multi agent.
        
        Args:
            config: Dict of hyperparameters (default: ES_MULTI_CONFIG)
            output_dir: Directory for checkpoints (default: ./es_outputs/multi_{timestamp})
        """
        self.target_angles = TRAINING_ANGLES  # [0, 90, 180]
        self.config = config or ES_MULTI_CONFIG.copy()
        self.n_jobs = self.config.get('n_jobs', 1)
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"es_outputs/multi_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parameters
        self.theta_best = np.random.uniform(0, 1, (N_RODS, N_RODS))
        self.reward_best = -np.inf
        
        # Adam state
        self.m = np.zeros_like(self.theta_best)
        self.v = np.zeros_like(self.theta_best)
        self.t_adam = 0
        
        # History for logging (matching es_agent.py structure where applicable)
        self.history = {
            'iteration': [],
            'reward_best': [],      # Σ P_i − λ · Var(P_i) for ES-Multi
            'reward_mean': [],
            'reward_std': [],
            'power_target': [],     # For ES-Multi: power_sum (what we're maximizing)
            'power_other_sum': [],  # For ES-Multi: same as power_sum (all angles are "targets")
            'power_0deg': [],
            'power_90deg': [],
            'power_180deg': [],
            'crosstalk_ratio': [],  # For ES-Multi: max_power / min_power (balance metric)
            'sigma': [],
            'eta_effective': [],
            # Additional ES-Multi specific metrics
            'power_sum': [],        # Σ P_i = P_0 + P_90 + P_180
            'power_variance': [],   # Var(P_i) - the penalty term
            'power_min': [],        # min(P_0, P_90, P_180)
            'power_max': [],        # max(P_0, P_90, P_180)
            'lambda_variance': [],  # λ used (should be 0.5)
        }
        
        # Top-K best configurations tracking
        self.top_k = 10
        self.top_k_configs = []
    
    def evaluate_configuration(self, rod_states: np.ndarray, verbose: bool = False) -> Tuple[float, Dict]:
        """
        Evaluate reward for a rod configuration across ALL angles.
        
        Args:
            rod_states: N_RODS × N_RODS array of ρ values ∈ [0, 1]
            verbose: If True, print simulation details
        
        Returns:
            (reward, powers_dict) where:
                reward = Σ P_i − λ · Var(P_i)  -- sum minus variance penalty
                powers_dict = {angle: power}
        """
        try:
            # Build permittivity map with ALL receiver waveguides
            eps_r = rods_to_permittivity(rod_states)
            eps_r = add_source_waveguide(eps_r)
            for angle in self.target_angles:
                eps_r = add_receiver_waveguide(eps_r, angle)
            
            # Run FDFD simulation
            Ez = run_simulation(eps_r, frequency=self.config['frequency_hz'], verbose=verbose)
            
            # Measure power at all receivers
            powers = {}
            for angle in self.target_angles:
                powers[angle] = measure_power_at_receiver(Ez, angle)
            
            # =================================================================
            # REWARD: R = Σ P_i − λ · Var(P_i)
            # =================================================================
            # Sum of powers (total transmission to all angles)
            power_sum = sum(powers.values())
            
            # Variance of powers (imbalance penalty)
            # Var = (1/n) * Σ (P_i - P̄)²
            power_mean = power_sum / len(powers)
            power_variance = sum((p - power_mean) ** 2 for p in powers.values()) / len(powers)
            
            # Reward = sum - λ * variance
            # Same λ=0.5 as ES-Single uses for crosstalk penalty
            lambda_var = self.config.get('lambda_variance', 0.5)
            reward = power_sum - lambda_var * power_variance
            
            return reward, powers
        
        except Exception as e:
            print(f"Simulation failed: {e}")
            return -1e10, {a: 0.0 for a in self.target_angles}
    
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
    
    def adam_step(self, gradient: np.ndarray) -> np.ndarray:
        """
        Apply Adam optimizer update.
        """
        self.t_adam += 1
        eta = self.config['eta']
        beta_1 = self.config['beta_1']
        beta_2 = self.config['beta_2']
        eps = self.config['adam_eps']
        
        self.m = beta_1 * self.m + (1 - beta_1) * gradient
        self.v = beta_2 * self.v + (1 - beta_2) * (gradient ** 2)
        
        m_hat = self.m / (1 - beta_1 ** self.t_adam)
        v_hat = self.v / (1 - beta_2 ** self.t_adam)
        
        delta = eta * m_hat / (np.sqrt(v_hat) + eps)
        
        if self.config.get('clip_norm'):
            norm = np.linalg.norm(delta)
            if norm > self.config['clip_norm']:
                delta = delta * self.config['clip_norm'] / norm
        
        return delta
    
    def update_top_k(self, reward: float, rho: np.ndarray, iteration: int):
        """Track top-K best configurations."""
        self.top_k_configs.append({
            'reward': reward,
            'rho': rho.copy(),
            'iteration': iteration
        })
        self.top_k_configs.sort(key=lambda x: x['reward'], reverse=True)
        self.top_k_configs = self.top_k_configs[:self.top_k]
    
    def get_top_k_summary(self) -> List[Dict]:
        """Get summary of top-K configurations."""
        return [
            {
                'rank': i + 1,
                'reward': cfg['reward'],
                'iteration': cfg['iteration']
            }
            for i, cfg in enumerate(self.top_k_configs)
        ]
    
    def train(self, verbose: bool = True):
        """
        Run ES-Multi training.
        
        Returns:
            (best_rho, best_reward): Optimized configuration and reward
        """
        N = self.config['N']
        sigma_0 = self.config['sigma_0']
        sigma_decay = self.config['sigma_decay']
        n_iters = self.config['n_iterations']
        log_every = self.config['log_every']
        
        if verbose:
            lambda_var = self.config.get('lambda_variance', 0.5)
            print(f"\n{'='*80}")
            print(f"ES-MULTI TRAINING: Optimizing for ALL angles ({self.target_angles})")
            print(f"{'='*80}")
            print(f"Population: N={N} | Iterations: {n_iters}")
            print(f"Reward: R = Σ P_i − λ · Var(P_i)  where λ={lambda_var}")
            print(f"        (Same λ as ES-Single's crosstalk penalty)")
            print(f"Output: {self.output_dir}")
            print(f"{'='*80}\n")
        
        for iteration in range(n_iters):
            sigma = sigma_0 * (sigma_decay ** iteration)
            
            if verbose:
                print(f"[{iteration+1:4d}/{n_iters}] ({100*(iteration+1)/n_iters:5.1f}%) "
                      f"Sampling population...", end='', flush=True)
            
            # Sample perturbations
            perturbations = np.random.randn(N, N_RODS, N_RODS) * sigma
            candidates = self.theta_best + perturbations
            candidates = np.clip(candidates, 0, 1)
            
            if verbose:
                print(f"  Evaluating population ({N} configurations)", end='', flush=True)
            
            # Evaluate population (parallel)
            if self.n_jobs != 1:
                results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                    delayed(self.evaluate_configuration)(candidates[i], verbose=False)
                    for i in range(N)
                )
            else:
                results = [self.evaluate_configuration(candidates[i], verbose=False) for i in range(N)]
            
            rewards = np.array([r[0] for r in results])
            powers_list = [r[1] for r in results]
            
            if verbose:
                print(f" Done")
            
            # Find best in population
            best_idx = np.argmax(rewards)
            if rewards[best_idx] > self.reward_best:
                self.reward_best = rewards[best_idx]
                self.theta_best = candidates[best_idx].copy()
                powers_best = powers_list[best_idx]
            else:
                # Re-evaluate current best to get powers
                _, powers_best = self.evaluate_configuration(self.theta_best, verbose=False)
            
            # Compute ES gradient
            if verbose:
                print(f"    → Computing ES gradient...", end='', flush=True)
            g = self.compute_es_gradient(perturbations, rewards, sigma)
            if verbose:
                print(f" ||g||={np.linalg.norm(g):.4e}")
            
            # Adam update
            if verbose:
                print(f"    → Applying Adam update...", end='', flush=True)
            delta_theta = self.adam_step(g)
            if verbose:
                print(f" ||Δθ||={np.linalg.norm(delta_theta):.4e}")
            
            self.theta_best = self.theta_best + delta_theta
            self.theta_best = np.clip(self.theta_best, 0, 1)
            
            # Re-evaluate after update
            reward_after, powers_after = self.evaluate_configuration(self.theta_best, verbose=False)
            if reward_after > self.reward_best:
                self.reward_best = reward_after
                powers_best = powers_after
            
            # Update top-K
            self.update_top_k(reward_after, self.theta_best, iteration)
            
            # Logging (match es_agent.py history structure)
            p_0 = powers_best.get(0, 0)
            p_90 = powers_best.get(90, 0)
            p_180 = powers_best.get(180, 0)
            p_min = min(p_0, p_90, p_180)
            p_max = max(p_0, p_90, p_180)
            p_sum = p_0 + p_90 + p_180
            p_mean = p_sum / 3
            p_variance = ((p_0 - p_mean)**2 + (p_90 - p_mean)**2 + (p_180 - p_mean)**2) / 3
            lambda_var = self.config.get('lambda_variance', 0.5)
            
            # For ES-Multi: crosstalk_ratio = max/min (how unbalanced the design is)
            # A perfect multi-angle design would have ratio = 1.0
            crosstalk_ratio = p_max / max(1e-12, p_min)
            
            self.history['iteration'].append(iteration)
            self.history['reward_best'].append(self.reward_best)
            self.history['reward_mean'].append(np.mean(rewards))
            self.history['reward_std'].append(np.std(rewards))
            self.history['power_target'].append(float(p_sum))       # ES-Multi: sum is the "target"
            self.history['power_other_sum'].append(float(p_sum))    # Same as power_target for ES-Multi
            self.history['power_0deg'].append(float(p_0))
            self.history['power_90deg'].append(float(p_90))
            self.history['power_180deg'].append(float(p_180))
            self.history['crosstalk_ratio'].append(float(crosstalk_ratio))
            self.history['sigma'].append(sigma)
            self.history['eta_effective'].append(np.linalg.norm(delta_theta))
            # ES-Multi specific
            self.history['power_sum'].append(float(p_sum))
            self.history['power_variance'].append(float(p_variance))
            self.history['power_min'].append(float(p_min))
            self.history['power_max'].append(float(p_max))
            self.history['lambda_variance'].append(float(lambda_var))
            
            if verbose:
                # Show reward breakdown: R = Σ P_i − λ · Var(P_i)
                print(f"  Summary: R_best={self.reward_best:.3e} (Σ={p_sum:.2e} − {lambda_var}·Var={p_variance:.2e})")
                print(f"           P_0={p_0:.2e} | P_90={p_90:.2e} | P_180={p_180:.2e} | "
                      f"balance={1/crosstalk_ratio:.2f} | σ={sigma:.4e}")
            
            # Checkpoint
            if (iteration + 1) % log_every == 0 or iteration == n_iters - 1:
                if verbose:
                    print(f"  ✓ Checkpoint saved: checkpoint_{iteration:05d}/")
                self.save_checkpoint(iteration)
                
                # Upload to S3
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
            print(f"\n{'='*80}")
            print(f"ES-Multi training complete!")
            print(f"{'='*80}")
            print(f"Final best reward (min power): {self.reward_best:.6e}")
            print(f"Final powers: P_0={p_0:.2e}, P_90={p_90:.2e}, P_180={p_180:.2e}")
            print(f"Output: {self.output_dir}")
            print(f"{'='*80}\n")
        
        return self.theta_best, self.reward_best
    
    def save_checkpoint(self, iteration: int):
        """Save checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint_{iteration:05d}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Get current powers
        _, powers = self.evaluate_configuration(self.theta_best, verbose=False)
        
        # Metadata
        metadata = {
            'iteration': iteration,
            'target_angles': self.target_angles,
            'reward_best': float(self.reward_best),
            'power_0deg': float(powers.get(0, 0)),
            'power_90deg': float(powers.get(90, 0)),
            'power_180deg': float(powers.get(180, 0)),
            'power_min': float(min(powers.values())),
            'power_sum': float(sum(powers.values())),
            'config': {k: v for k, v in self.config.items() if not callable(v)},
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Best rod configuration
        np.save(checkpoint_dir / 'best_rho.npy', self.theta_best)
        
        # Full permittivity map (with all waveguides)
        eps_r = rods_to_permittivity(self.theta_best)
        eps_r = add_source_waveguide(eps_r)
        for angle in self.target_angles:
            eps_r = add_receiver_waveguide(eps_r, angle)
        np.save(checkpoint_dir / 'best_eps_r.npy', eps_r)
        
        # Electric field
        Ez = run_simulation(eps_r, frequency=self.config['frequency_hz'], verbose=False)
        np.save(checkpoint_dir / 'best_Ez.npy', Ez)
        
        # Top-K configurations
        top_k_dir = checkpoint_dir / 'top_k_configs'
        top_k_dir.mkdir(exist_ok=True)
        
        for i, cfg in enumerate(self.top_k_configs):
            np.save(top_k_dir / f'rank_{i+1:02d}_rho.npy', cfg['rho'])
        
        with open(checkpoint_dir / 'top_k_summary.json', 'w') as f:
            json.dump(self.get_top_k_summary(), f, indent=2)
        
        # Training history
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _upload_checkpoint_to_s3(self, iteration: int, s3_bucket: str):
        """Upload checkpoint to S3."""
        import boto3
        s3_client = boto3.client('s3')
        
        checkpoint_dir = self.output_dir / f"checkpoint_{iteration:05d}"
        s3_prefix = f"{self.output_dir.name}/checkpoint_{iteration:05d}"
        
        for file_path in checkpoint_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(checkpoint_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                s3_client.upload_file(str(file_path), s3_bucket, s3_key)
    
    def plot_history(self, save_path: str = None):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        iters = self.history['iteration']
        
        # Reward (min power)
        ax = axes[0, 0]
        ax.plot(iters, self.history['reward_best'], 'b-', label='Best (min)', linewidth=2)
        ax.plot(iters, self.history['reward_mean'], 'g--', alpha=0.7, label='Pop mean')
        ax.fill_between(iters,
                        np.array(self.history['reward_mean']) - np.array(self.history['reward_std']),
                        np.array(self.history['reward_mean']) + np.array(self.history['reward_std']),
                        alpha=0.2, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward (min power)')
        ax.set_title('ES-Multi: Reward = min(P_0, P_90, P_180)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Individual powers
        ax = axes[0, 1]
        ax.plot(iters, self.history['power_0deg'], 'r-', label='P_0°', linewidth=2)
        ax.plot(iters, self.history['power_90deg'], 'g-', label='P_90°', linewidth=2)
        ax.plot(iters, self.history['power_180deg'], 'b-', label='P_180°', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Power')
        ax.set_title('Power at Each Angle')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sigma decay
        ax = axes[1, 0]
        ax.plot(iters, self.history['sigma'], 'purple', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('σ')
        ax.set_title('Perturbation Scale')
        ax.grid(True, alpha=0.3)
        
        # Power sum vs min
        ax = axes[1, 1]
        ax.plot(iters, self.history['power_sum'], 'orange', label='Sum', linewidth=2)
        ax.plot(iters, self.history['power_min'], 'purple', label='Min', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Power')
        ax.set_title('Sum vs Min Power')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# =============================================================================
# Module exports
# =============================================================================
__all__ = [
    'ESMultiAgent',
    'ES_MULTI_CONFIG',
]
