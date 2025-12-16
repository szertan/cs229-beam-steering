# =============================================================================
# SIMULATION: Core FDFD Simulation Functions
# =============================================================================
"""
Functions for running FDFD simulations of the plasma metamaterial array.

Each function does ONE thing:
        rods_to_permittivity: Convert 8×8 rod states to 2D permittivity map (rods only)
        add_source_waveguide: Add source waveguide plates to permittivity map
        add_receiver_waveguide: Add ONE receiver waveguide at a specified angle
        create_source: Create line source excitation
        run_simulation: Solve Maxwell's equations, return Ez field
        measure_power_at_receiver: Integrate |Ez|² at a receiver

Notes about the design variable (ρ):
        - In this codebase the per-rod design variable is named ``rho`` (ρ).
        - ``rho`` is a *normalized plasma-frequency control* in the range [0, 1].
            Concretely: omega_p = rho * OMEGA_P_MAX, where OMEGA_P_MAX is defined
            in ``src/constants.py`` (default 15 GHz in our setup).
        - The physical permittivity used by the FDFD solver is computed via the
            Drude-model mapping implemented in ``rho_to_permittivity(rho)``:
                    epsilon(omega) = 1 - (omega_p / omega)**2
            and exposed to this module as the helper ``_rho_to_eps``.
        - Throughout the simulation pipeline we treat ``rod_states`` as an
            8×8 array of these normalized ``rho`` values; ``rods_to_permittivity``
            converts them into the full-grid permittivity map ``eps_r``.
"""

import numpy as np
import sys

# Add ceviche to path if needed
sys.path.insert(0, '/Users/sze/project-lorentz/ceviche')

from ceviche import fdfd_ez

from .constants import (
    N_RODS, OMEGA, DL, NPML,
    NX, NY, Nx, Ny,
    ROD_ARRAY_X0, ROD_ARRAY_Y0,
    ROD_SPACING_GRID, ROD_RADIUS_GRID,
    METAL_EPSILON, PLATE_THICKNESS, WAVEGUIDE_WIDTH,
    SOURCE_X, SOURCE_WG_Y_TOP, SOURCE_WG_Y_BOT,
    WAVEGUIDE_Y_TOP, WAVEGUIDE_Y_BOT,
    RECEIVERS, RECEIVER_LINE_LENGTH,
    FREQUENCY, WAVELENGTH, RESOLUTION,
    rho_to_permittivity as _rho_to_eps,
    res, dl,
    Npml,
)


# =============================================================================
# Permittivity Map Construction
# =============================================================================

def rods_to_permittivity(rod_states):
    """
    Convert the 8×8 rod state matrix into a 2D permittivity map.
    
    This function ONLY places the plasma rods on a blank (air) grid.
    Use add_source_waveguide() and add_receiver_waveguide() to add waveguides.
    
    Args:
        rod_states: N_RODS × N_RODS array (8×8 = 64 values).
                    Each entry is ``rho`` (ρ), the *normalized plasma-frequency*
                    control in the range [0, 1]. Concretely:
                        omega_p = rho * OMEGA_P_MAX
                    and permittivity is computed via the Drude mapping
                        epsilon(omega) = 1 - (omega_p / OMEGA)**2
                    - rho = 0: rod is off (air, ε = 1)
                    - rho = 1: rod at maximum plasma frequency (ε ≈ -5.25 at 6 GHz)
    
    Returns:
        eps_r: Nx × Ny array of permittivity values (air + rods only)
    """
    # Initialize entire domain as air (permittivity = 1.0)
    eps_r = np.ones((Nx, Ny))
    
    # Add plasma rods
    for row_idx in range(N_RODS):
        for col_idx in range(N_RODS):
            
            # Get the ρ value for this rod
            rho = rod_states[row_idx, col_idx]
            
            # Skip if rod is off (ρ ≈ 0)
            if rho < 0.01:
                continue
            
            # Convert ρ to permittivity using Drude model
            rod_permittivity = _rho_to_eps(rho)
            
            # Calculate the center position of this rod in grid coordinates
            rod_center_x = ROD_ARRAY_X0 + row_idx * ROD_SPACING_GRID
            rod_center_y = ROD_ARRAY_Y0 + col_idx * ROD_SPACING_GRID
            
            # Fill in a circular region around the rod center
            for delta_x in range(-ROD_RADIUS_GRID, ROD_RADIUS_GRID + 1):
                for delta_y in range(-ROD_RADIUS_GRID, ROD_RADIUS_GRID + 1):
                    
                    # Check if this point is inside the circular rod
                    distance_squared = delta_x**2 + delta_y**2
                    is_inside_rod = distance_squared <= ROD_RADIUS_GRID**2
                    
                    if is_inside_rod:
                        grid_x = rod_center_x + delta_x
                        grid_y = rod_center_y + delta_y
                        
                        # Bounds check
                        if 0 <= grid_x < Nx and 0 <= grid_y < Ny:
                            eps_r[grid_x, grid_y] = rod_permittivity
    
    return eps_r


def add_source_waveguide(eps_r):
    """
    Add source waveguide metal plates to an existing permittivity map.
    
    The source waveguide is on the left side, with horizontal plates
    that guide the wave toward the rod array.
    
    Args:
        eps_r: Nx × Ny permittivity array (modified in place)
    
    Returns:
        eps_r: The same array with source waveguide added
    """
    plate_x_start = Npml
    plate_x_end = ROD_ARRAY_X0 - 5  # Stop before rods
    
    # Top plate
    for t in range(PLATE_THICKNESS):
        y_top = SOURCE_WG_Y_TOP + t
        if y_top < Ny:
            eps_r[plate_x_start:plate_x_end, y_top] = METAL_EPSILON
    
    # Bottom plate
    for t in range(PLATE_THICKNESS):
        y_bot = SOURCE_WG_Y_BOT - t
        if y_bot >= 0:
            eps_r[plate_x_start:plate_x_end, y_bot] = METAL_EPSILON
    
    return eps_r


def add_receiver_waveguide(eps_r, angle):
    """
    Add ONE receiver waveguide at the specified angle.
    
    Call this function multiple times with different angles to add
    multiple receivers.
    
    Args:
        eps_r: Nx × Ny permittivity array (modified in place)
        angle: Receiver angle (0, 90, or 180 degrees)
    
    Returns:
        eps_r: The same array with receiver waveguide added
    
    Raises:
        ValueError: If angle is not a valid receiver angle
    """
    if angle not in RECEIVERS:
        raise ValueError(f"No receiver defined for angle {angle}°. Available: {list(RECEIVERS.keys())}")
    
    receiver = RECEIVERS[angle]
    
    if receiver['type'] == 'vertical_waveguide':
        # 90° Receiver (RIGHT): horizontal plates (top and bottom)
        y_top = receiver['wg_y_center'] + WAVEGUIDE_WIDTH // 2
        y_bot = receiver['wg_y_center'] - WAVEGUIDE_WIDTH // 2
        x_start = receiver['wg_x_start']
        x_end = receiver['wg_x_end']
        
        for t in range(PLATE_THICKNESS):
            if y_top + t < Ny:
                eps_r[x_start:x_end, y_top + t] = METAL_EPSILON
            if y_bot - t >= 0:
                eps_r[x_start:x_end, y_bot - t] = METAL_EPSILON
    
    else:  # horizontal_waveguide (0° and 180°)
        # Vertical plates (left and right)
        x_left = receiver['wg_x_center'] - WAVEGUIDE_WIDTH // 2
        x_right = receiver['wg_x_center'] + WAVEGUIDE_WIDTH // 2
        y_start = receiver['wg_y_start']
        y_end = receiver['wg_y_end']
        
        for t in range(PLATE_THICKNESS):
            if x_left - t >= 0:
                eps_r[x_left - t, y_start:y_end] = METAL_EPSILON
            if x_right + t < Nx:
                eps_r[x_right + t, y_start:y_end] = METAL_EPSILON
    
    return eps_r


# =============================================================================
# Source Creation
# =============================================================================

def create_source(amplitude=1e3):
    """
    Create the spatial distribution of the source (WHERE to inject energy).
    
    This function only defines the LOCATION and AMPLITUDE of the source.
    It does NOT set the frequency - that is determined by run_simulation().
    
    In FDFD, the source is a current distribution. The frequency of the
    resulting wave is set by the solver, not by the source pattern.
    
    Args:
        amplitude: Source amplitude (scales all fields linearly)
    
    Returns:
        source: Nx × Ny complex array with line source in waveguide
    """
    source = np.zeros((Nx, Ny), dtype=complex)
    
    # Line source spanning the waveguide aperture
    margin = 2
    y_start = WAVEGUIDE_Y_BOT + margin
    y_end = WAVEGUIDE_Y_TOP - margin
    
    source[SOURCE_X, y_start:y_end] = amplitude
    
    return source


# =============================================================================
# FDFD Simulation
# =============================================================================

def run_simulation(eps_r, frequency=None, verbose=True):
    """
    Run a ceviche FDFD simulation with the given permittivity map.
    
    This solves Maxwell's equations at the specified frequency.
    The frequency parameter sets the frequency of the input wave.
    
    Note: In FDFD, frequency is a property of the SOLVER, not the source.
    The entire simulation operates at one frequency (frequency domain).
    
    Args:
        eps_r: Nx × Ny permittivity array (fully constructed)
        frequency: Frequency of the input wave in Hz (default: 6 GHz)
        verbose: If True, print simulation parameters
    
    Returns:
        Ez: Complex electric field Ez(x,y) across the domain
    """
    # Use default frequency if not specified
    if frequency is None:
        frequency = FREQUENCY
    
    omega = 2 * np.pi * frequency
    wavelength = 3e8 / frequency
    
    if verbose:
        print(f"--- Simulation ---")
        print(f"  Input wave frequency: f = {frequency/1e9:.2f} GHz, λ = {wavelength*100:.2f} cm")
        print(f"  Grid: {Nx}×{Ny}, resolution = {res} pts/a, dl = {dl*1000:.2f} mm")
    
    # Create FDFD simulation object (frequency is set here)
    simulation = fdfd_ez(omega, DL, eps_r, [NPML, NPML])
    
    # Create source excitation (location only)
    source = create_source()
    
    # Solve Maxwell's equations
    Ex, Ey, Ez = simulation.solve(source)
    
    return Ez


# =============================================================================
# Power Measurement
# =============================================================================

def measure_power_at_receiver(Ez, angle):
    """
    Measure the integrated power at a receiver for a given angle.
    
    The receiver is a line segment inside a waveguide that integrates |Ez|² 
    along its length.
    
    Args:
        Ez: Complex electric field from simulation (Nx × Ny)
        angle: Target angle (0, 90, or 180 degrees)
    
    Returns:
        power: Integrated |Ez|² at the receiver (scalar)
    """
    if angle not in RECEIVERS:
        raise ValueError(f"No receiver defined for angle {angle}°. Available: {list(RECEIVERS.keys())}")
    
    receiver = RECEIVERS[angle]
    rx, ry = receiver['receiver_x'], receiver['receiver_y']
    half_len = RECEIVER_LINE_LENGTH // 2
    
    if receiver['line_orientation'] == 'horizontal':
        x_start = max(0, rx - half_len)
        x_end = min(Nx, rx + half_len)
        power = np.sum(np.abs(Ez[x_start:x_end, ry])**2)
    else:  # vertical
        y_start = max(0, ry - half_len)
        y_end = min(Ny, ry + half_len)
        power = np.sum(np.abs(Ez[rx, y_start:y_end])**2)
    
    return power


# =============================================================================
# Module-level exports
# =============================================================================
__all__ = [
    'rods_to_permittivity',
    'add_source_waveguide',
    'add_receiver_waveguide',
    'create_source',
    'run_simulation',
    'measure_power_at_receiver',
]
