#!/usr/bin/env python3
"""
Generate figure showing ES+NN designs and mode collapse at 180°.

This shows the key finding: the NN learns to sacrifice 180° to maximize 0° and 90°.
"""

import sys
import os
sys.path.insert(0, '/Users/sze/project-lorentz/ceviche')

import numpy as np
import matplotlib.pyplot as plt
import json

# Load constants
exec(open('/Users/sze/project-lorentz/cs229-beam-steering/src/constants.py').read())

# Paths
NN_CHECKPOINT = '/Users/sze/project-lorentz/cs229-beam-steering/es_outputs/nn_from_s3/checkpoint_00200'
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

ANGLES = [0, 90, 180]

def create_nn_figure():
    """Create figure showing NN designs and mode collapse."""
    print("Creating ES+NN Figure...")
    
    # Load metadata
    with open(os.path.join(NN_CHECKPOINT, 'metadata.json')) as f:
        metadata = json.load(f)
    
    powers = metadata['best_powers']
    print(f"  Powers: 0°={powers['0']:.1f}, 90°={powers['90']:.1f}, 180°={powers['180']:.1f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    
    for idx, angle in enumerate(ANGLES):
        # Load design (rho)
        rho = np.load(os.path.join(NN_CHECKPOINT, f'design_{angle}deg.npy'))
        Ez = np.load(os.path.join(NN_CHECKPOINT, f'Ez_{angle}deg.npy'))
        
        print(f"  Loaded {angle}° data: rho shape={rho.shape}, Ez shape={Ez.shape}")
        
        # Top row: Rod patterns (ρ values)
        ax_rho = axes[0, idx]
        im_rho = ax_rho.imshow(rho.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
        
        # Title with power
        power = float(powers[str(angle)])
        if power > 1000:
            power_str = f'P={power:.2e}'
        else:
            power_str = f'P={power:.1f}'
        ax_rho.set_title(f'{angle}° Design\n({power_str})')
        ax_rho.set_xlabel('Row')
        ax_rho.set_ylabel('Column')
        
        # Add grid
        ax_rho.set_xticks(np.arange(-0.5, N_RODS, 1), minor=True)
        ax_rho.set_yticks(np.arange(-0.5, N_RODS, 1), minor=True)
        ax_rho.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        ax_rho.set_xticks(range(N_RODS))
        ax_rho.set_yticks(range(N_RODS))
        
        # Bottom row: Field patterns (log scale)
        ax_field = axes[1, idx]
        field_intensity = np.abs(Ez.T)**2
        field_log = np.log10(field_intensity + 1e-10)
        vmax = np.percentile(field_log, 99)
        vmin = vmax - 4
        im_field = ax_field.imshow(field_log, cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
        ax_field.set_title(f'{angle}° Field |Ez|²')
        ax_field.set_xlabel('x')
        ax_field.set_ylabel('y')
    
    # Add colorbars
    fig.colorbar(im_rho, ax=axes[0, :], shrink=0.6, label='ρ (plasma density)', pad=0.02)
    fig.colorbar(im_field, ax=axes[1, :], shrink=0.6, label='log₁₀|Ez|²', pad=0.02)
    
    # Add overall title
    fig.suptitle('ES+NN: Mode Collapse at 180°\n(Network sacrifices backward steering to maximize forward angles)', 
                 fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'nn_mode_collapse.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating ES+NN Mode Collapse Figure")
    print("=" * 60)
    
    create_nn_figure()
    
    print("\n" + "=" * 60)
    print("✓ Figure generated!")
    print("=" * 60)
