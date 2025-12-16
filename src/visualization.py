# =============================================================================
# VISUALIZATION: Plotting Functions for Analysis and Debugging
# =============================================================================
"""
Visualization functions for the beam steering simulation.

Main functions:
    visualize_simulation: Comprehensive 3-panel display (permittivity, field, power)
    plot_permittivity: Plot just the permittivity map
    plot_field: Plot just the electric field
    plot_power_comparison: Compare power distributions
    plot_rod_pattern: Visualize the 8×8 rod pattern as a heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .constants import (
    N_RODS,
    SOURCE_X, PROBE_X,
    SOURCE_WG_Y_TOP, SOURCE_WG_Y_BOT,
    WAVEGUIDE_WIDTH, PLATE_THICKNESS,
    RECEIVERS, RECEIVER_LINE_LENGTH,
    ROD_ARRAY_X0, NPML,
)


# =============================================================================
# Color Schemes
# =============================================================================

# Permittivity colormap: black for air (ε=1), red/orange/yellow for plasma (ε<0)
COLORS_EPS = ['red', 'orangered', 'orange', 'gold', 'black']
EPS_CMAP = LinearSegmentedColormap.from_list('plasma_eps', COLORS_EPS)

# Receiver colors: warm red-orange-yellow palette
COLORS_RECV = {
    0: '#e63946',    # Red (Bottom)
    90: '#f4a261',   # Orange (Right/Forward)
    180: '#e9c46a'   # Gold/Yellow (Top)
}

# Metal plate color
PLATE_COLOR = '#4a4a4a'  # Dark gray


def visualize_simulation(eps_r, Ez, powers, title=None, save_path=None):
    """
    Comprehensive visualization of a beam steering simulation.
    
    Shows three panels:
    1. Permittivity map (rod configuration)
    2. Field magnitude |Ez| with receiver positions and waveguide plates
    3. Power bar chart at each receiver
    
    Args:
        eps_r: Permittivity map (Nx × Ny)
        Ez: Electric field (Nx × Ny)
        powers: Dict mapping angle → power (e.g., {0: 1e4, 90: 2e5, 180: 5e4})
        title: Optional title for the figure
        save_path: If provided, save figure to this path instead of displaying
    
    Returns:
        fig: The matplotlib figure object
    """
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Panel 1: Permittivity Map
    # -------------------------------------------------------------------------
    # Clip to reasonable range for visualization (ignore extreme metal values)
    eps_display = np.clip(eps_r, -16, 1)
    
    im0 = axes[0].imshow(eps_display.T, origin='lower', cmap=EPS_CMAP, 
                         vmin=-16, vmax=1)
    axes[0].set_title('Permittivity εr\n(red=plasma, black=air)')
    cbar0 = plt.colorbar(im0, ax=axes[0], label='ε')
    
    # Mark source and probe lines (subtle)
    axes[0].axvline(SOURCE_X, color='yellow', linestyle='--', alpha=0.7)
    axes[0].axvline(PROBE_X, color='green', linestyle='--', alpha=0.7)
    
    # -------------------------------------------------------------------------
    # Panel 2: Field Magnitude with Receivers and Waveguide Plates
    # -------------------------------------------------------------------------
    field_magnitude = np.abs(Ez)
    im1 = axes[1].imshow(field_magnitude.T, origin='lower', cmap='hot')
    axes[1].set_title('|Ez| Field Magnitude with Receivers')
    plt.colorbar(im1, ax=axes[1], label='|Ez|')
    
    # Draw waveguide plates as rectangles
    _draw_waveguide_plates(axes[1])
    
    # Draw receiver lines on top
    for angle, info in RECEIVERS.items():
        rx, ry = info['receiver_x'], info['receiver_y']
        half_len = RECEIVER_LINE_LENGTH // 2
        
        if info['line_orientation'] == 'horizontal':
            axes[1].plot([rx - half_len, rx + half_len], [ry, ry], 
                       color=COLORS_RECV[angle], linewidth=3, 
                       label=f"{angle}° ({info['name']})")
        else:
            axes[1].plot([rx, rx], [ry - half_len, ry + half_len], 
                       color=COLORS_RECV[angle], linewidth=3, 
                       label=f"{angle}° ({info['name']})")
    
    axes[1].legend(loc='upper right', fontsize=8)
    
    # -------------------------------------------------------------------------
    # Panel 3: Power Bar Chart
    # -------------------------------------------------------------------------
    angles_list = list(powers.keys())
    powers_list = list(powers.values())
    bars = axes[2].bar([f"{a}°\n({RECEIVERS[a]['name']})" for a in angles_list], 
                      powers_list, color=[COLORS_RECV[a] for a in angles_list])
    axes[2].set_ylabel('Integrated Power (|Ez|²)')
    axes[2].set_title('Power at Each Receiver')
    axes[2].set_yscale('log')
    
    # Add value labels on bars
    for bar, power in zip(bars, powers_list):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{power:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig


def _draw_waveguide_plates(ax):
    """Helper function to draw waveguide plate rectangles on an axis."""
    
    # Source waveguide plates (horizontal, on left)
    src_plate_x_start = NPML
    src_plate_x_end = ROD_ARRAY_X0 - 5
    
    # Top plate
    ax.add_patch(plt.Rectangle(
        (src_plate_x_start, SOURCE_WG_Y_TOP), 
        src_plate_x_end - src_plate_x_start, PLATE_THICKNESS,
        facecolor=PLATE_COLOR, edgecolor='white', linewidth=0.5
    ))
    # Bottom plate
    ax.add_patch(plt.Rectangle(
        (src_plate_x_start, SOURCE_WG_Y_BOT - PLATE_THICKNESS), 
        src_plate_x_end - src_plate_x_start, PLATE_THICKNESS,
        facecolor=PLATE_COLOR, edgecolor='white', linewidth=0.5
    ))
    
    # Receiver waveguide plates - loop over all receivers
    for angle, receiver in RECEIVERS.items():
        if receiver['type'] == 'vertical_waveguide':
            # Horizontal plates (top and bottom) for waveguides opening left/right
            y_top = receiver['wg_y_center'] + WAVEGUIDE_WIDTH // 2
            y_bot = receiver['wg_y_center'] - WAVEGUIDE_WIDTH // 2
            x_start = receiver['wg_x_start']
            x_end = receiver['wg_x_end']
            
            ax.add_patch(plt.Rectangle(
                (x_start, y_top), 
                x_end - x_start, PLATE_THICKNESS,
                facecolor=PLATE_COLOR, edgecolor='white', linewidth=0.5
            ))
            ax.add_patch(plt.Rectangle(
                (x_start, y_bot - PLATE_THICKNESS), 
                x_end - x_start, PLATE_THICKNESS,
                facecolor=PLATE_COLOR, edgecolor='white', linewidth=0.5
            ))
        else:  # horizontal_waveguide
            # Vertical plates (left and right) for waveguides opening up/down
            x_left = receiver['wg_x_center'] - WAVEGUIDE_WIDTH // 2
            x_right = receiver['wg_x_center'] + WAVEGUIDE_WIDTH // 2
            y_start = receiver['wg_y_start']
            y_end = receiver['wg_y_end']
            
            ax.add_patch(plt.Rectangle(
                (x_left - PLATE_THICKNESS, y_start), 
                PLATE_THICKNESS, y_end - y_start,
                facecolor=PLATE_COLOR, edgecolor='white', linewidth=0.5
            ))
            ax.add_patch(plt.Rectangle(
                (x_right, y_start), 
                PLATE_THICKNESS, y_end - y_start,
                facecolor=PLATE_COLOR, edgecolor='white', linewidth=0.5
            ))


def plot_permittivity(eps_r, title="Permittivity Map", save_path=None):
    """
    Plot just the permittivity map.
    
    Args:
        eps_r: Permittivity map (Nx × Ny)
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    eps_display = np.clip(eps_r, -16, 1)
    im = ax.imshow(eps_display.T, origin='lower', cmap=EPS_CMAP, vmin=-16, vmax=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='ε')
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return fig


def plot_field(Ez, title="|Ez| Field Magnitude", save_path=None):
    """
    Plot just the electric field magnitude.
    
    Args:
        Ez: Complex electric field (Nx × Ny)
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    field_magnitude = np.abs(Ez)
    im = ax.imshow(field_magnitude.T, origin='lower', cmap='hot')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='|Ez|')
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return fig


def plot_power_comparison(powers_dict, labels=None, title="Power Comparison", save_path=None):
    """
    Compare power distributions from multiple simulations.
    
    Args:
        powers_dict: Dict of {label: {angle: power}} or list of power dicts
        labels: Labels for each simulation (if powers_dict is a list)
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(powers_dict, list):
        # List of power dicts
        if labels is None:
            labels = [f"Sim {i+1}" for i in range(len(powers_dict))]
        data = {label: powers for label, powers in zip(labels, powers_dict)}
    else:
        data = powers_dict
    
    angles = list(list(data.values())[0].keys())
    x = np.arange(len(angles))
    width = 0.8 / len(data)
    
    for i, (label, powers) in enumerate(data.items()):
        offset = (i - len(data)/2 + 0.5) * width
        values = [powers[a] for a in angles]
        ax.bar(x + offset, values, width, label=label)
    
    ax.set_ylabel('Power (|Ez|²)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}°" for a in angles])
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return fig


def plot_rod_pattern(rod_states, title="Rod Pattern (ρ values)", save_path=None):
    """
    Visualize the 8×8 rod pattern as a heatmap.
    
    Args:
        rod_states: N_RODS × N_RODS array of per-rod design values. Each
            entry is ``rho`` (ρ), the normalized plasma-frequency control in
            [0, 1]. The mapping rho → permittivity is performed by
            ``rho_to_permittivity`` in ``src/constants.py`` (Drude model).
        title: Plot title
        save_path: Optional path to save the figure
        
    Returns:
        fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(rod_states.T, origin='lower', cmap='viridis', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel('Row')
    ax.set_ylabel('Column')
    plt.colorbar(im, ax=ax, label='ρ')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, N_RODS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_RODS, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Add text annotations
    for i in range(N_RODS):
        for j in range(N_RODS):
            val = rod_states[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(i, j, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return fig


# =============================================================================
# Module-level exports
# =============================================================================
__all__ = [
    'visualize_simulation',
    'plot_permittivity',
    'plot_field',
    'plot_power_comparison',
    'plot_rod_pattern',
    'EPS_CMAP',
    'COLORS_RECV',
    'PLATE_COLOR',
]
