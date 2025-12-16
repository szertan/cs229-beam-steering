# =============================================================================
# CONSTANTS: Simulation Parameters for Microwave Lab Setup
# =============================================================================
"""
Physical and grid parameters for the plasma metamaterial beam steering simulation.

These parameters match the real 8×8 plasma rod array in the lab.
See DESIGN_DECISIONS.md for full justification of each choice.

Constants are organized into:
    - Physical constants (speed of light, etc.)
    - Operating frequency and wavelength
    - Lab hardware parameters (rod dimensions, spacing)
    - Plasma frequency parameters (Drude model)
    - Grid resolution and domain layout
    - Source and receiver waveguide configuration
    - Angle definitions for training/evaluation
"""

import numpy as np

# =============================================================================
# Physical Constants
# =============================================================================
SPEED_OF_LIGHT = 3e8        # m/s
METAL_EPSILON = -1e4        # Large negative permittivity to simulate PEC (metal)

# =============================================================================
# Operating Frequency
# =============================================================================
# We choose 6 GHz based on analysis of rod/wavelength ratios:
#   - d/λ = 0.40 (sub-wavelength rods, clean scattering)
#   - a/λ = 0.42 (safely below Bragg condition of 0.5)
#   - At ρ=1: ε = -5.25 (strong but not extreme contrast)
# This matches Jesse's experimental setup.
FREQUENCY = 6e9             # 6 GHz microwave
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY  # ~5 cm
OMEGA = 2 * np.pi * FREQUENCY  # Angular frequency (rad/s)

# =============================================================================
# Lab Hardware Parameters (FIXED - must match physical setup)
# =============================================================================
N_RODS = 8                  # 8×8 rod array
ROD_DIAMETER = 0.02         # Rod diameter: 2 cm
ROD_RADIUS_M = ROD_DIAMETER / 2  # Rod radius: 1 cm
LATTICE_CONSTANT = 0.021    # Lattice constant: 2.1 cm (center-to-center spacing)
                            # This is rod diameter (2cm) + gap (1mm)

# =============================================================================
# Plasma Frequency Parameters (from Jesse's characterization)
# =============================================================================
# Jesse's data shows plasma frequency is LINEAR with voltage:
#   - At V=0:  ω_p ≈ 0 GHz
#   - At V=30: ω_p ≈ 15 GHz
# We use ρ ∈ [0,1] as NORMALIZED PLASMA FREQUENCY (not electron density!)
# This gives us direct mapping: ρ → voltage → plasma frequency → permittivity

OMEGA_P_MAX = 2 * np.pi * 15e9   # Max plasma frequency: 15 GHz (at 30V)
RHO_LEVELS = 11                   # Discrete levels: 0.0, 0.1, 0.2, ..., 1.0

# =============================================================================
# Grid Resolution
# =============================================================================
RESOLUTION = 25                    # Grid points per lattice constant
DL = LATTICE_CONSTANT / RESOLUTION # Grid spacing in meters (~0.84 mm)
ROD_RADIUS_GRID = int(ROD_RADIUS_M / DL)  # Rod radius in grid points (~11)
ROD_SPACING_GRID = RESOLUTION      # Rod spacing = 1 lattice constant = res grid points

# Convenience aliases (for notebook compatibility)
res = RESOLUTION
dl = DL
rod_radius_grid = ROD_RADIUS_GRID
rod_spacing_grid = ROD_SPACING_GRID

# =============================================================================
# Simulation Domain Layout
# =============================================================================
# Layout needs space for:
#   - Source waveguide on the left
#   - 8×8 rod array in the center  
#   - Receiver waveguides on right, top, and bottom

NPML = 2 * RESOLUTION              # PML thickness: 2 lattice constants
SOURCE_GAP = 2 * RESOLUTION        # Gap for source waveguide
RECEIVER_GAP = 2 * RESOLUTION      # Gap for receiver waveguides

# X direction: PML | source_wg | rods | receiver_wg | PML
NX = 2 * NPML + SOURCE_GAP + N_RODS * RESOLUTION + RECEIVER_GAP

# Y direction: PML | bottom_receiver_wg | rods | top_receiver_wg | PML
NY = 2 * NPML + RECEIVER_GAP + N_RODS * RESOLUTION + RECEIVER_GAP

# Convenience aliases
Npml = NPML
Nx = NX
Ny = NY
source_gap = SOURCE_GAP
receiver_gap = RECEIVER_GAP

# =============================================================================
# Key Positions in Grid
# =============================================================================
# Source position (left side)
SOURCE_X = NPML + SOURCE_GAP // 2
SOURCE_Y_CENTER = NY // 2

# Rod array position (center of domain)
ROD_ARRAY_X0 = NPML + SOURCE_GAP  # First rod column
ROD_ARRAY_Y0 = NPML + RECEIVER_GAP  # First rod row (bottom)

# Rod array boundaries (useful for placing receivers OUTSIDE)
ROD_ARRAY_X_END = ROD_ARRAY_X0 + (N_RODS - 1) * ROD_SPACING_GRID + ROD_RADIUS_GRID
ROD_ARRAY_Y_END = ROD_ARRAY_Y0 + (N_RODS - 1) * ROD_SPACING_GRID + ROD_RADIUS_GRID
ROD_ARRAY_Y_START = ROD_ARRAY_Y0 - ROD_RADIUS_GRID

# Rod array center (for reference)
ROD_ARRAY_CENTER_X = ROD_ARRAY_X0 + (N_RODS - 1) * ROD_SPACING_GRID // 2
ROD_ARRAY_CENTER_Y = ROD_ARRAY_Y0 + (N_RODS - 1) * ROD_SPACING_GRID // 2

# Legacy variable (for backwards compatibility)
PROBE_X = ROD_ARRAY_X_END + RESOLUTION

# =============================================================================
# Waveguide Parameters (shared by source and receivers)
# =============================================================================
WAVEGUIDE_WIDTH = 2 * RESOLUTION           # Aperture width: 2 lattice constants (~4.2 cm)
PLATE_THICKNESS = 3                        # Metal plate thickness in grid points

# =============================================================================
# Source Waveguide (left side, horizontal)
# =============================================================================
SOURCE_WG_LENGTH = SOURCE_GAP  # = 2 * res = 50 grid points
SOURCE_WG_Y_TOP = SOURCE_Y_CENTER + WAVEGUIDE_WIDTH // 2
SOURCE_WG_Y_BOT = SOURCE_Y_CENTER - WAVEGUIDE_WIDTH // 2

# Legacy names for compatibility
WAVEGUIDE_Y_TOP = SOURCE_WG_Y_TOP
WAVEGUIDE_Y_BOT = SOURCE_WG_Y_BOT

# =============================================================================
# Receiver Waveguides (positioned OUTSIDE the rod array)
# =============================================================================
# Each receiver has:
#   - A waveguide (two parallel metal plates)
#   - A line receiver inside the waveguide
#   - Positioned just outside the rod array
#
# Design: Receiver waveguide length = half of source waveguide length
#         Receiver line is at the BACK of the waveguide (away from rods)

# Receiver waveguide length = half of source
RECEIVER_WG_LENGTH = SOURCE_WG_LENGTH // 2  # = res = 25 grid points

RECEIVER_LINE_LENGTH = WAVEGUIDE_WIDTH - 4  # Line receiver length (inside waveguide)

# Distance from rod array boundary to waveguide opening (same for all receivers)
RECV_GAP = RESOLUTION // 2  # 12 grid points gap between rods and waveguide opening

# 90° Receiver: RIGHT of rod array (vertical waveguide, opens to the left)
RECV_90_X = ROD_ARRAY_X_END + RECV_GAP
RECV_90_Y_CENTER = ROD_ARRAY_CENTER_Y
RECV_90_WG_X_START = RECV_90_X
RECV_90_WG_X_END = RECV_90_X + RECEIVER_WG_LENGTH

# 0° Receiver: BOTTOM of rod array (horizontal waveguide, opens upward)
RECV_0_Y = ROD_ARRAY_Y_START - RECV_GAP
RECV_0_X_CENTER = ROD_ARRAY_CENTER_X
RECV_0_WG_Y_END = RECV_0_Y
RECV_0_WG_Y_START = RECV_0_Y - RECEIVER_WG_LENGTH

# 180° Receiver: TOP of rod array (horizontal waveguide, opens downward)
RECV_180_Y = ROD_ARRAY_Y_END + RECV_GAP
RECV_180_X_CENTER = ROD_ARRAY_CENTER_X
RECV_180_WG_Y_START = RECV_180_Y
RECV_180_WG_Y_END = RECV_180_Y + RECEIVER_WG_LENGTH

# =============================================================================
# Receiver Definitions Dictionary
# =============================================================================
# NOTE: Receiver line is at the BACK of the waveguide (farthest from rods)
RECEIVERS = {
    0: {    # 0° = Down (-y direction)
        'name': 'Bottom',
        'type': 'horizontal_waveguide',  # Waveguide runs in Y direction
        'receiver_x': RECV_0_X_CENTER,
        'receiver_y': RECV_0_WG_Y_START + 2,  # At back of waveguide (+ small margin)
        'wg_x_center': RECV_0_X_CENTER,
        'wg_y_start': RECV_0_WG_Y_START,
        'wg_y_end': RECV_0_WG_Y_END,
        'line_orientation': 'horizontal',  # Receiver line runs along x
    },
    90: {   # 90° = Right (+x direction, straight through)
        'name': 'Right',
        'type': 'vertical_waveguide',  # Waveguide runs in X direction
        'receiver_x': RECV_90_WG_X_END - 2,  # At back of waveguide (- small margin)
        'receiver_y': RECV_90_Y_CENTER,
        'wg_x_start': RECV_90_WG_X_START,
        'wg_x_end': RECV_90_WG_X_END,
        'wg_y_center': RECV_90_Y_CENTER,
        'line_orientation': 'vertical',  # Receiver line runs along y
    },
    180: {  # 180° = Up (+y direction)
        'name': 'Top',
        'type': 'horizontal_waveguide',  # Waveguide runs in Y direction
        'receiver_x': RECV_180_X_CENTER,
        'receiver_y': RECV_180_WG_Y_END - 2,  # At back of waveguide (- small margin)
        'wg_x_center': RECV_180_X_CENTER,
        'wg_y_start': RECV_180_WG_Y_START,
        'wg_y_end': RECV_180_WG_Y_END,
        'line_orientation': 'horizontal',  # Receiver line runs along x
    },
}

RECEIVER_LENGTH = RECEIVER_LINE_LENGTH  # For compatibility

# =============================================================================
# Angle Definitions
# =============================================================================
# Convention: 0° = down (-y), 90° = right (+x, straight through), 180° = up (+y)
TRAINING_ANGLES = [0, 90, 180]        # Angles to optimize for (with receivers)
GENERALIZATION_ANGLES = [30, 60, 120, 150]  # Angles for generalization testing
EVAL_ANGLES = [0, 30, 60, 90, 120, 150, 180]  # All evaluation angles

# =============================================================================
# Derived Quantities (useful for analysis)
# =============================================================================
# Permittivity range
EPS_MIN = 1 - (OMEGA_P_MAX / OMEGA) ** 2  # At ρ=1 (most negative)
EPS_MAX = 1.0  # At ρ=0 (air)

# =============================================================================
# Helper Functions for ρ ↔ Permittivity Conversion
# =============================================================================

def rho_to_plasma_frequency(rho):
    """
    Convert normalized control parameter ρ to plasma frequency.
    
    Args:
        rho: Normalized plasma frequency ∈ [0, 1]
             0 = plasma off (V=0)
             1 = max plasma (V=30V, ω_p=15 GHz)
    
    Returns:
        omega_p: Plasma frequency in rad/s
    """
    return rho * OMEGA_P_MAX


def plasma_frequency_to_permittivity(omega_p):
    """
    Convert plasma frequency to permittivity using Drude model.
    
    ε(ω) = 1 - (ω_p/ω)²
    
    Args:
        omega_p: Plasma frequency in rad/s
    
    Returns:
        epsilon: Relative permittivity (can be negative!)
    """
    if omega_p == 0:
        return 1.0  # Air
    return 1.0 - (omega_p / OMEGA) ** 2


def rho_to_permittivity(rho):
    """
    Full chain: ρ → plasma frequency → permittivity.
    
    This is the key function that converts our optimization variable (ρ)
    to the physical quantity ceviche needs (permittivity).
    
    Args:
        rho: Normalized plasma frequency ∈ [0, 1]
    
    Returns:
        epsilon: Relative permittivity
    """
    omega_p = rho_to_plasma_frequency(rho)
    return plasma_frequency_to_permittivity(omega_p)


def discretize_rho(rho, n_levels=RHO_LEVELS):
    """
    Snap continuous ρ values to discrete levels.
    
    Args:
        rho: Continuous ρ value(s) in [0, 1]
        n_levels: Number of discrete levels (default: 11)
    
    Returns:
        Discretized ρ snapped to nearest level (0.0, 0.1, ..., 1.0)
    """
    step = 1.0 / (n_levels - 1)
    return np.round(rho / step) * step


def get_rod_positions():
    """
    Get the grid positions of all rod centers.
    
    Returns:
        positions: List of (x, y) tuples for each rod center
    """
    positions = []
    for row in range(N_RODS):
        for col in range(N_RODS):
            x = ROD_ARRAY_X0 + row * ROD_SPACING_GRID
            y = ROD_ARRAY_Y0 + col * ROD_SPACING_GRID
            positions.append((x, y))
    return positions


# =============================================================================
# Module-level exports
# =============================================================================
__all__ = [
    # Physical constants
    'SPEED_OF_LIGHT', 'METAL_EPSILON',
    # Frequency
    'FREQUENCY', 'WAVELENGTH', 'OMEGA',
    # Hardware
    'N_RODS', 'ROD_DIAMETER', 'ROD_RADIUS_M', 'LATTICE_CONSTANT',
    # Plasma
    'OMEGA_P_MAX', 'RHO_LEVELS',
    # Grid
    'RESOLUTION', 'DL', 'ROD_RADIUS_GRID', 'ROD_SPACING_GRID',
    'res', 'dl', 'rod_radius_grid', 'rod_spacing_grid',
    # Domain
    'NPML', 'NX', 'NY', 'Npml', 'Nx', 'Ny',
    'SOURCE_GAP', 'RECEIVER_GAP', 'source_gap', 'receiver_gap',
    # Positions
    'SOURCE_X', 'SOURCE_Y_CENTER',
    'ROD_ARRAY_X0', 'ROD_ARRAY_Y0',
    'ROD_ARRAY_X_END', 'ROD_ARRAY_Y_END', 'ROD_ARRAY_Y_START',
    'ROD_ARRAY_CENTER_X', 'ROD_ARRAY_CENTER_Y',
    'PROBE_X',
    # Waveguides
    'WAVEGUIDE_WIDTH', 'PLATE_THICKNESS',
    'SOURCE_WG_LENGTH', 'SOURCE_WG_Y_TOP', 'SOURCE_WG_Y_BOT',
    'WAVEGUIDE_Y_TOP', 'WAVEGUIDE_Y_BOT',
    'RECEIVER_WG_LENGTH', 'RECEIVER_LINE_LENGTH', 'RECV_GAP',
    'RECEIVER_LENGTH',
    # Receivers
    'RECEIVERS',
    'RECV_90_X', 'RECV_90_Y_CENTER', 'RECV_90_WG_X_START', 'RECV_90_WG_X_END',
    'RECV_0_Y', 'RECV_0_X_CENTER', 'RECV_0_WG_Y_START', 'RECV_0_WG_Y_END',
    'RECV_180_Y', 'RECV_180_X_CENTER', 'RECV_180_WG_Y_START', 'RECV_180_WG_Y_END',
    # Angles
    'TRAINING_ANGLES', 'GENERALIZATION_ANGLES', 'EVAL_ANGLES',
    # Derived
    'EPS_MIN', 'EPS_MAX',
    # Functions
    'rho_to_plasma_frequency', 'plasma_frequency_to_permittivity',
    'rho_to_permittivity', 'discretize_rho', 'get_rod_positions',
]
