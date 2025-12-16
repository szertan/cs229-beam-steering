#!/usr/bin/env python3
"""
Generate figures for the CS229 final report.

Uses the existing simulation and visualization modules.
Reads data from final_report/data/ (downloaded from S3).
Saves figures to final_report/figures/
"""

import sys
import os

# Add paths
sys.path.insert(0, '/Users/sze/project-lorentz/ceviche')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# Load constants directly (avoiding import issues)
exec(open('/Users/sze/project-lorentz/cs229-beam-steering/src/constants.py').read())

# Define colors for receivers
COLORS_RECV = {
    0: '#e63946',    # Red (Bottom)
    90: '#f4a261',   # Orange (Right/Forward)
    180: '#e9c46a'   # Gold/Yellow (Top)
}

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

ANGLES = [0, 90, 180]


# =============================================================================
# FIGURE 1: Domain Schematic
# =============================================================================
def create_domain_schematic():
    """Create a schematic of the simulation domain."""
    print("Creating Figure 1: Domain Schematic...")
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    
    # PML regions (gray)
    pml_color = '#DDDDDD'
    ax.add_patch(patches.Rectangle((0, 0), NPML, Ny, facecolor=pml_color, edgecolor='none'))
    ax.add_patch(patches.Rectangle((Nx-NPML, 0), NPML, Ny, facecolor=pml_color, edgecolor='none'))
    ax.add_patch(patches.Rectangle((0, 0), Nx, NPML, facecolor=pml_color, edgecolor='none'))
    ax.add_patch(patches.Rectangle((0, Ny-NPML), Nx, NPML, facecolor=pml_color, edgecolor='none'))
    
    # Rod array (8x8 circles)
    for i in range(N_RODS):
        for j in range(N_RODS):
            cx = ROD_ARRAY_X0 + i * ROD_SPACING_GRID
            cy = ROD_ARRAY_Y0 + j * ROD_SPACING_GRID
            circle = patches.Circle((cx, cy), ROD_RADIUS_GRID, 
                                     facecolor='#4169E1', edgecolor='black', 
                                     linewidth=0.5, alpha=0.8)
            ax.add_patch(circle)
    
    # Source waveguide (left) - parallel plates
    wg_color = '#333333'
    ax.plot([0, SOURCE_X+15], [SOURCE_WG_Y_TOP, SOURCE_WG_Y_TOP], color=wg_color, linewidth=3)
    ax.plot([0, SOURCE_X+15], [SOURCE_WG_Y_BOT, SOURCE_WG_Y_BOT], color=wg_color, linewidth=3)
    
    # Source arrow
    ax.annotate('', xy=(SOURCE_X+35, Ny//2), xytext=(SOURCE_X, Ny//2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(SOURCE_X-15, Ny//2, 'Source\n(6 GHz)', fontsize=9, ha='center', color='red')
    
    # Receiver waveguides
    for angle in ANGLES:
        recv = RECEIVERS[angle]
        color = COLORS_RECV[angle]
        
        if recv['type'] == 'vertical_waveguide':  # 90° (right)
            y_top = recv['wg_y_center'] + WAVEGUIDE_WIDTH // 2
            y_bot = recv['wg_y_center'] - WAVEGUIDE_WIDTH // 2
            ax.plot([recv['wg_x_start'], Nx], [y_top, y_top], color=wg_color, linewidth=3)
            ax.plot([recv['wg_x_start'], Nx], [y_bot, y_bot], color=wg_color, linewidth=3)
            ax.text(Nx-20, recv['wg_y_center'], f'{angle}°', fontsize=11, 
                    ha='center', fontweight='bold', color=color)
        else:  # 0° (bottom) or 180° (top)
            x_left = recv['wg_x_center'] - WAVEGUIDE_WIDTH // 2
            x_right = recv['wg_x_center'] + WAVEGUIDE_WIDTH // 2
            y_start = recv['wg_y_start']
            y_end = recv['wg_y_end']
            
            if angle == 0:
                ax.plot([x_left, x_left], [0, y_end], color=wg_color, linewidth=3)
                ax.plot([x_right, x_right], [0, y_end], color=wg_color, linewidth=3)
                ax.text(recv['wg_x_center'], 15, f'{angle}°', fontsize=11, 
                        ha='center', fontweight='bold', color=color)
            else:  # 180
                ax.plot([x_left, x_left], [y_start, Ny], color=wg_color, linewidth=3)
                ax.plot([x_right, x_right], [y_start, Ny], color=wg_color, linewidth=3)
                ax.text(recv['wg_x_center'], Ny-15, f'{angle}°', fontsize=11, 
                        ha='center', fontweight='bold', color=color)
    
    # Labels
    ax.text(NPML//2, Ny//2, 'PML', fontsize=9, ha='center', va='center', 
            rotation=90, color='gray')
    ax.text(ROD_ARRAY_X0 + 3.5*ROD_SPACING_GRID, ROD_ARRAY_Y0 + 3.5*ROD_SPACING_GRID, 
            '8×8\nPlasma\nRods', fontsize=10, ha='center', va='center', fontweight='bold')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x (grid points)')
    ax.set_ylabel('y (grid points)')
    ax.set_title('Simulation Domain')
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'domain_schematic.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# FIGURE 2: Training Curves
# =============================================================================
def create_training_curves():
    """Plot training convergence for all three angles."""
    print("\nCreating Figure 2: Training Curves...")
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    colors = {0: '#1f77b4', 90: '#ff7f0e', 180: '#2ca02c'}
    
    for angle in ANGLES:
        history_path = os.path.join(DATA_DIR, f'{angle}deg', 'training_history.json')
        with open(history_path) as f:
            history = json.load(f)
        
        iters = history['iteration']
        rewards = history['reward_best']
        ax.plot(iters, rewards, label=f'{angle}°', color=colors[angle], linewidth=1.5)
        print(f"  Loaded {angle}° history: {len(iters)} iterations")
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Reward')
    ax.set_title('ES-Single Training Convergence')
    ax.legend(title='Target Angle')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'training_curves.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# FIGURE 3: Optimized Designs and Field Patterns
# =============================================================================
def create_field_patterns():
    """Show optimized rod patterns and resulting field distributions."""
    print("\nCreating Figure 3: Field Patterns...")
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    
    for idx, angle in enumerate(ANGLES):
        # Load data
        rho = np.load(os.path.join(DATA_DIR, f'{angle}deg', 'best_rho.npy'))
        Ez = np.load(os.path.join(DATA_DIR, f'{angle}deg', 'best_Ez.npy'))
        print(f"  Loaded {angle}° data: rho={rho.shape}, Ez={Ez.shape}")
        
        # Top row: Rod patterns (ρ values)
        ax_rho = axes[0, idx]
        im_rho = ax_rho.imshow(rho.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax_rho.set_title(f'{angle}° Design (ρ)')
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
    
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, 'field_patterns.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating figures for CS229 Final Report")
    print("=" * 60)
    
    create_domain_schematic()
    create_training_curves()
    create_field_patterns()
    
    print("\n" + "=" * 60)
    print("✓ All figures generated!")
    print("=" * 60)
