"""
ES+NN Agent: Evolution Strategies optimizing a Neural Network for beam steering.

The NN takes angle as input (sin/cos encoding) and outputs an 8x8 design.
This allows:
1. Training on discrete angles (0°, 90°, 180°)
2. Generalization to unseen angles (45°, 135°, etc.)
3. Saving/loading model weights for later evaluation

Architecture:
    Input: [sin(θ), cos(θ)]  (2 units)
    Hidden: 64 → ReLU → 128 → ReLU → 64 → ReLU
    Output: 64 → Sigmoid → reshape(8, 8)

Reward Function (per angle, same as ES-Single):
    R_θ = P_target(θ) - λ * Σ P_other(θ)
    
Total Reward (sum over training angles):
    R_total = Σ_θ R_θ
"""

import numpy as np
import json
import time
import os
import boto3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
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

# Import ES config for consistency
from .es_agent import ES_CONFIG


# =============================================================================
# Neural Network (NumPy implementation for ES compatibility)
# =============================================================================

@dataclass
class NNLayer:
    """A single dense layer."""
    weights: np.ndarray  # shape: (in_features, out_features)
    biases: np.ndarray   # shape: (out_features,)
    
    @property
    def num_params(self) -> int:
        return self.weights.size + self.biases.size


class BeamSteeringNN:
    """
    Neural network that maps angle → 8x8 design.
    
    Architecture:
        [sin(θ), cos(θ)] → 64 → 128 → 64 → 64 → sigmoid → reshape(8,8)
    """
    
    def __init__(self, seed: int = 42):
        """Initialize network with Xavier initialization."""
        rng = np.random.default_rng(seed)
        
        # Layer sizes: input=2, hidden=[64, 128, 64], output=64
        layer_sizes = [2, 64, 128, 64, 64]
        
        self.layers: List[NNLayer] = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            
            # Xavier initialization
            scale = np.sqrt(2.0 / (in_size + out_size))
            weights = rng.normal(0, scale, (in_size, out_size))
            biases = np.zeros(out_size)
            
            self.layers.append(NNLayer(weights=weights, biases=biases))
        
        self._num_params = sum(layer.num_params for layer in self.layers)
    
    @property
    def num_params(self) -> int:
        return self._num_params
    
    def forward(self, angle_deg: float) -> np.ndarray:
        """
        Forward pass: angle → 8x8 design.
        
        Args:
            angle_deg: Beam angle in degrees
            
        Returns:
            8x8 array with values in [0, 1]
        """
        # Encode angle as [sin, cos]
        angle_rad = np.deg2rad(angle_deg)
        x = np.array([np.sin(angle_rad), np.cos(angle_rad)])
        
        # Forward through hidden layers with ReLU
        for layer in self.layers[:-1]:
            x = x @ layer.weights + layer.biases
            x = np.maximum(0, x)  # ReLU
        
        # Output layer with sigmoid
        x = x @ self.layers[-1].weights + self.layers[-1].biases
        x = 1 / (1 + np.exp(-x))  # Sigmoid
        
        # Reshape to 8x8
        return x.reshape(8, 8)
    
    def get_flat_params(self) -> np.ndarray:
        """Get all parameters as a flat vector."""
        params = []
        for layer in self.layers:
            params.append(layer.weights.flatten())
            params.append(layer.biases.flatten())
        return np.concatenate(params)
    
    def set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set all parameters from a flat vector."""
        idx = 0
        for layer in self.layers:
            # Weights
            w_size = layer.weights.size
            layer.weights = flat_params[idx:idx + w_size].reshape(layer.weights.shape)
            idx += w_size
            
            # Biases
            b_size = layer.biases.size
            layer.biases = flat_params[idx:idx + b_size].reshape(layer.biases.shape)
            idx += b_size
    
    def save(self, path: Path) -> None:
        """Save model weights to file."""
        params = self.get_flat_params()
        np.save(path, params)
    
    def load(self, path: Path) -> None:
        """Load model weights from file."""
        params = np.load(path)
        self.set_flat_params(params)
    
    def save_architecture(self, path: Path) -> None:
        """Save architecture info for reconstruction."""
        info = {
            'layer_sizes': [2, 64, 128, 64, 64],
            'num_params': self.num_params,
            'layer_shapes': [
                {'weights': list(layer.weights.shape), 'biases': list(layer.biases.shape)}
                for layer in self.layers
            ]
        }
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)


# =============================================================================
# ES+NN Configuration
# =============================================================================

ES_NN_CONFIG = {
    # ES hyperparameters (same as ES-Single/Multi)
    'N': 100,                    # Population size
    'sigma_0': 0.3,              # Initial noise std (in param space)
    'sigma_decay': 0.999,        # Decay rate per iteration
    'eta': 0.02,                 # Learning rate
    'beta_1': 0.9,               # Adam momentum
    'beta_2': 0.999,             # Adam RMSprop
    'adam_eps': 1e-8,            # Adam epsilon
    
    # Training
    'n_iterations': 1000,
    'log_every': 10,
    'checkpoint_every': 100,
    
    # Reward function (same λ as ES-Single for fair comparison)
    'lambda_crosstalk': 0.5,
    
    # Training angles
    'training_angles': [0, 90, 180],
    
    # Simulation
    'frequency_hz': 6e9,
    
    # Parallelization (use -1 for all cores, or specific number)
    'n_jobs': -1,
}


# =============================================================================
# Standalone evaluation function for parallel execution
# =============================================================================

def _evaluate_nn_candidate(theta: np.ndarray, training_angles: List[int], 
                           frequency_hz: float, lambda_crosstalk: float) -> Tuple[float, Dict[int, float]]:
    """
    Evaluate a single NN parameter vector (standalone function for joblib).
    
    This is a module-level function so it can be pickled for parallel execution.
    
    Args:
        theta: Flat NN parameter vector
        training_angles: List of angles to evaluate
        frequency_hz: Simulation frequency
        lambda_crosstalk: Reward penalty weight
        
    Returns:
        (total_reward, powers_dict)
    """
    # Create a temporary NN and set params
    nn = BeamSteeringNN(seed=0)  # Seed doesn't matter, we're setting params
    nn.set_flat_params(theta)
    
    total_reward = 0.0
    powers = {}
    
    for angle in training_angles:
        # Get design from NN
        rho = nn.forward(angle)
        
        # Build permittivity map with ALL receiver waveguides
        eps_r = rods_to_permittivity(rho)
        eps_r = add_source_waveguide(eps_r)
        for recv_angle in training_angles:
            eps_r = add_receiver_waveguide(eps_r, recv_angle)
        
        # Run FDFD simulation
        Ez = run_simulation(eps_r, frequency=frequency_hz, verbose=False)
        
        # Measure power at all receivers
        port_powers = {}
        for recv_angle in training_angles:
            port_powers[recv_angle] = measure_power_at_receiver(Ez, recv_angle)
        
        # Store power at target port
        powers[angle] = port_powers[angle]
        
        # Compute reward (same as ES-Single)
        target_power = port_powers[angle]
        other_power = sum(p for a, p in port_powers.items() if a != angle)
        reward = target_power - lambda_crosstalk * other_power
        
        total_reward += reward
    
    return total_reward, powers


# =============================================================================
# ES+NN Agent
# =============================================================================

class ESNNAgent:
    """
    Evolution Strategies agent that optimizes neural network weights.
    
    The NN maps angle → design, allowing generalization to unseen angles.
    """
    
    def __init__(self, config: Optional[Dict] = None, seed: int = 42):
        """Initialize ES+NN agent."""
        self.config = {**ES_NN_CONFIG, **(config or {})}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Initialize neural network
        self.nn = BeamSteeringNN(seed=seed)
        self.n_params = self.nn.num_params
        
        # Current parameters (what ES optimizes)
        self.theta = self.nn.get_flat_params()
        
        # Best found
        self.best_theta = self.theta.copy()
        self.best_reward = -np.inf
        self.best_powers = {}  # powers at each angle for best
        
        # Adaptive sigma
        self.sigma = self.config['sigma_0']
        
        # Adam optimizer state
        self.m = np.zeros(self.n_params)  # First moment
        self.v = np.zeros(self.n_params)  # Second moment
        self.t = 0  # Timestep
        
        # Training angles
        self.training_angles = self.config['training_angles']
        
        # Parallelization
        self.n_jobs = self.config['n_jobs']
        
        # History tracking
        self.history = {
            'iteration': [],
            'reward_best': [],
            'reward_mean': [],
            'reward_std': [],
            # Per-angle powers for best design
            'power_0deg': [],
            'power_90deg': [],
            'power_180deg': [],
            # Aggregate metrics
            'power_sum': [],
            'power_min': [],
            'power_max': [],
            # ES state
            'sigma': [],
            'eta_effective': [],
            # NN-specific
            'param_norm': [],
            'grad_norm': [],
        }
        
        # Output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'es_outputs/nn_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Top-K tracking
        self.top_k = 5
        self.top_k_configs: List[Tuple[float, np.ndarray, Dict[int, float]]] = []
    
    def evaluate_nn(self, theta: np.ndarray) -> Tuple[float, Dict[int, float]]:
        """
        Evaluate NN parameters across all training angles.
        
        Returns:
            total_reward: Sum of per-angle rewards
            powers: Dict mapping angle → power at target port
        """
        # Set NN parameters
        self.nn.set_flat_params(theta)
        
        total_reward = 0.0
        powers = {}
        
        for angle in self.training_angles:
            # Get design from NN (8x8 array in [0, 1])
            rho = self.nn.forward(angle)
            
            # Build permittivity map with ALL receiver waveguides
            eps_r = rods_to_permittivity(rho)
            eps_r = add_source_waveguide(eps_r)
            for recv_angle in self.training_angles:
                eps_r = add_receiver_waveguide(eps_r, recv_angle)
            
            # Run FDFD simulation
            Ez = run_simulation(eps_r, frequency=self.config['frequency_hz'], verbose=False)
            
            # Measure power at all receivers
            port_powers = {}
            for recv_angle in self.training_angles:
                port_powers[recv_angle] = measure_power_at_receiver(Ez, recv_angle)
            
            # Store power at target port
            powers[angle] = port_powers[angle]
            
            # Compute reward (same as ES-Single)
            target_power = port_powers[angle]
            other_power = sum(p for a, p in port_powers.items() if a != angle)
            reward = target_power - self.config['lambda_crosstalk'] * other_power
            
            total_reward += reward
        
        return total_reward, powers
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Run ES optimization on NN parameters.
        
        Returns:
            Final training history
        """
        if verbose:
            print("=" * 70)
            print("ES+NN TRAINING")
            print("=" * 70)
            print(f"Neural Network: {self.n_params:,} parameters")
            print(f"Architecture: 2 → 64 → 128 → 64 → 64 → sigmoid → 8×8")
            print(f"Training angles: {self.training_angles}")
            print(f"Reward: Σ_θ [P_target(θ) - {self.config['lambda_crosstalk']} * Σ P_other(θ)]")
            print(f"Population: N={self.config['N']}, σ₀={self.config['sigma_0']}")
            print(f"Parallel jobs: {self.n_jobs}")
            print(f"Output: {self.output_dir}")
            print("=" * 70)
        
        n_iterations = self.config['n_iterations']
        N = self.config['N']
        
        for iteration in range(1, n_iterations + 1):
            iter_start = time.time()
            
            # Generate perturbations
            epsilon = self.rng.standard_normal((N, self.n_params))
            
            # Build list of perturbed parameters
            perturbed_thetas = [self.theta + self.sigma * epsilon[i] for i in range(N)]
            
            if verbose:
                print(f"[{iteration:4d}/{n_iterations}] Evaluating {N} candidates...", end='', flush=True)
            
            # Parallel evaluation using joblib (same pattern as ES-Single)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_evaluate_nn_candidate)(
                    theta,
                    self.training_angles,
                    self.config['frequency_hz'],
                    self.config['lambda_crosstalk']
                )
                for theta in perturbed_thetas
            )
            
            # Unpack results
            rewards = np.array([r[0] for r in results])
            all_powers = [r[1] for r in results]
            
            if verbose:
                print(" Done", flush=True)
            
            # Compute gradient using fitness shaping (rank-based)
            ranks = np.argsort(np.argsort(rewards))  # Rank from 0 to N-1
            centered_ranks = ranks - (N - 1) / 2      # Center around 0
            
            # Gradient estimate
            grad = (1 / (N * self.sigma)) * (epsilon.T @ centered_ranks)
            
            # Adam update
            self.t += 1
            self.m = self.config['beta_1'] * self.m + (1 - self.config['beta_1']) * grad
            self.v = self.config['beta_2'] * self.v + (1 - self.config['beta_2']) * (grad ** 2)
            
            m_hat = self.m / (1 - self.config['beta_1'] ** self.t)
            v_hat = self.v / (1 - self.config['beta_2'] ** self.t)
            
            eta_effective = self.config['eta'] / (np.sqrt(np.mean(v_hat)) + self.config['adam_eps'])
            
            # Update parameters
            self.theta = self.theta + eta_effective * m_hat
            
            # Decay sigma
            self.sigma *= self.config['sigma_decay']
            
            # Track best
            best_idx = np.argmax(rewards)
            if rewards[best_idx] > self.best_reward:
                self.best_reward = rewards[best_idx]
                self.best_theta = self.theta + self.sigma * epsilon[best_idx]
                self.best_powers = all_powers[best_idx]
            
            # Update top-K
            for i in range(N):
                theta_i = self.theta + self.sigma * epsilon[i]
                self._update_top_k(rewards[i], theta_i, all_powers[i])
            
            # Log history
            if iteration % self.config['log_every'] == 0 or iteration == 1:
                self.nn.set_flat_params(self.best_theta)
                
                self.history['iteration'].append(iteration)
                self.history['reward_best'].append(float(self.best_reward))
                self.history['reward_mean'].append(float(np.mean(rewards)))
                self.history['reward_std'].append(float(np.std(rewards)))
                
                # Per-angle powers
                self.history['power_0deg'].append(float(self.best_powers.get(0, 0)))
                self.history['power_90deg'].append(float(self.best_powers.get(90, 0)))
                self.history['power_180deg'].append(float(self.best_powers.get(180, 0)))
                
                # Aggregates
                power_values = list(self.best_powers.values())
                self.history['power_sum'].append(float(sum(power_values)))
                self.history['power_min'].append(float(min(power_values)))
                self.history['power_max'].append(float(max(power_values)))
                
                # ES state
                self.history['sigma'].append(float(self.sigma))
                self.history['eta_effective'].append(float(eta_effective))
                
                # NN-specific
                self.history['param_norm'].append(float(np.linalg.norm(self.theta)))
                self.history['grad_norm'].append(float(np.linalg.norm(grad)))
                
                if verbose:
                    iter_time = time.time() - iter_start
                    powers_str = ", ".join(f"{a}°:{p:.4f}" for a, p in sorted(self.best_powers.items()))
                    print(f"[{iteration:4d}/{n_iterations}] "
                          f"R={self.best_reward:.4f} | "
                          f"Powers: {powers_str} | "
                          f"σ={self.sigma:.4f} | "
                          f"{iter_time:.1f}s")
            
            # Checkpoint
            if iteration % self.config['checkpoint_every'] == 0:
                self.save_checkpoint(iteration)
                self._upload_checkpoint_to_s3(iteration)
        
        # Final checkpoint
        self.save_checkpoint(n_iterations)
        self._upload_checkpoint_to_s3(n_iterations)
        
        if verbose:
            print("=" * 70)
            print("TRAINING COMPLETE")
            print(f"Best reward: {self.best_reward:.4f}")
            print(f"Best powers: {self.best_powers}")
            print("=" * 70)
        
        return self.history
    
    def _update_top_k(self, reward: float, theta: np.ndarray, powers: Dict[int, float]) -> None:
        """Update top-K configurations."""
        if len(self.top_k_configs) < self.top_k:
            self.top_k_configs.append((reward, theta.copy(), powers.copy()))
            self.top_k_configs.sort(key=lambda x: x[0], reverse=True)
        elif reward > self.top_k_configs[-1][0]:
            self.top_k_configs[-1] = (reward, theta.copy(), powers.copy())
            self.top_k_configs.sort(key=lambda x: x[0], reverse=True)
    
    def save_checkpoint(self, iteration: int) -> Path:
        """
        Save checkpoint with model weights.
        
        This is the key advantage over ES-Single/Multi: we save the actual
        model that can be loaded and evaluated at ANY angle later.
        """
        checkpoint_dir = self.output_dir / f'checkpoint_{iteration:05d}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set NN to best params for saving
        self.nn.set_flat_params(self.best_theta)
        
        # 1. Save NN weights (the key file!)
        self.nn.save(checkpoint_dir / 'model_weights.npy')
        self.nn.save_architecture(checkpoint_dir / 'model_architecture.json')
        
        # 2. Save best designs at each training angle
        for angle in self.training_angles:
            rho = self.nn.forward(angle)
            np.save(checkpoint_dir / f'design_{angle}deg.npy', rho)
            
            # Also save eps_r and field
            eps_r = rods_to_permittivity(rho)
            eps_r = add_source_waveguide(eps_r)
            for recv_angle in self.training_angles:
                eps_r = add_receiver_waveguide(eps_r, recv_angle)
            Ez = run_simulation(eps_r, frequency=self.config['frequency_hz'], verbose=False)
            np.save(checkpoint_dir / f'eps_r_{angle}deg.npy', eps_r)
            np.save(checkpoint_dir / f'Ez_{angle}deg.npy', Ez)
        
        # 3. Metadata
        metadata = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'best_reward': float(self.best_reward),
            'best_powers': {str(k): float(v) for k, v in self.best_powers.items()},
            'training_angles': self.training_angles,
            'sigma': float(self.sigma),
            'config': {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in self.config.items()},
            'nn_num_params': self.n_params,
            'seed': self.seed,
        }
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 4. Training history
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 5. Top-K summary
        top_k_summary = []
        for rank, (reward, theta, powers) in enumerate(self.top_k_configs):
            top_k_summary.append({
                'rank': rank + 1,
                'reward': float(reward),
                'powers': {str(k): float(v) for k, v in powers.items()},
            })
        with open(checkpoint_dir / 'top_k_summary.json', 'w') as f:
            json.dump(top_k_summary, f, indent=2)
        
        # 6. Save top-K model weights
        top_k_dir = checkpoint_dir / 'top_k_models'
        top_k_dir.mkdir(exist_ok=True)
        for rank, (reward, theta, powers) in enumerate(self.top_k_configs):
            np.save(top_k_dir / f'rank_{rank+1}_weights.npy', theta)
        
        return checkpoint_dir
    
    def _upload_checkpoint_to_s3(self, iteration: int) -> None:
        """Upload checkpoint to S3 if S3_BUCKET is set."""
        bucket = os.environ.get('S3_BUCKET')
        if not bucket:
            return
        
        try:
            s3 = boto3.client('s3')
            checkpoint_dir = self.output_dir / f'checkpoint_{iteration:05d}'
            
            # Upload all files in checkpoint
            for file_path in checkpoint_dir.rglob('*'):
                if file_path.is_file():
                    s3_key = f"{self.output_dir.name}/{checkpoint_dir.name}/{file_path.relative_to(checkpoint_dir)}"
                    s3.upload_file(str(file_path), bucket, s3_key)
            
            print(f"  → Uploaded checkpoint to s3://{bucket}/{self.output_dir.name}/{checkpoint_dir.name}/")
        except Exception as e:
            print(f"  → S3 upload failed: {e}")
    
    def evaluate_angle(self, angle_deg: float) -> Dict:
        """
        Evaluate the best model at any angle (including unseen angles).
        
        This is the key capability of ES+NN vs ES-Single/Multi.
        """
        self.nn.set_flat_params(self.best_theta)
        rho = self.nn.forward(angle_deg)
        
        # Build simulation with all training receivers
        eps_r = rods_to_permittivity(rho)
        eps_r = add_source_waveguide(eps_r)
        for recv_angle in self.training_angles:
            eps_r = add_receiver_waveguide(eps_r, recv_angle)
        Ez = run_simulation(eps_r, frequency=self.config['frequency_hz'], verbose=False)
        
        # Measure power at all ports
        port_powers = {}
        for recv_angle in self.training_angles:
            port_powers[recv_angle] = measure_power_at_receiver(Ez, recv_angle)
        
        return {
            'angle': angle_deg,
            'design': rho,
            'port_powers': port_powers,
            'Ez': Ez,
        }


def load_model(checkpoint_path: Path) -> BeamSteeringNN:
    """
    Load a trained model from checkpoint.
    
    Usage:
        model = load_model(Path('es_outputs/nn_20251214_.../checkpoint_01000'))
        design = model.forward(45)  # Get design for 45 degrees
    """
    nn = BeamSteeringNN()
    nn.load(checkpoint_path / 'model_weights.npy')
    return nn
