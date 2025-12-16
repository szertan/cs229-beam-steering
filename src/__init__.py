# =============================================================================
# CS229 Beam Steering - Inverse Design Environment
# =============================================================================
"""
This package provides the simulation environment for beam steering optimization
using an 8×8 plasma metamaterial rod array.

Modules:
    constants: Physical parameters, grid setup, receiver configuration
    simulation: FDFD simulation functions (each does ONE thing)
    visualization: Plotting functions for analysis and debugging

Quick Start:
    from src import (
        rods_to_permittivity, add_source_waveguide, add_receiver_waveguide,
        run_simulation, measure_power_at_receiver,
        RECEIVERS, TRAINING_ANGLES, N_RODS
    )
    import numpy as np
    
    # Create rod pattern and build permittivity map
    rod_states = np.zeros((N_RODS, N_RODS))
    eps_r = rods_to_permittivity(rod_states)
    eps_r = add_source_waveguide(eps_r)
    for angle in TRAINING_ANGLES:
        eps_r = add_receiver_waveguide(eps_r, angle)
    
    # Run simulation
    Ez = run_simulation(eps_r)
    
    # Measure power at receivers
    powers = {angle: measure_power_at_receiver(Ez, angle) for angle in TRAINING_ANGLES}
"""

# Re-export everything from submodules for convenience
from .constants import *
from .simulation import *
from .visualization import *

__version__ = "0.1.0"
__author__ = "SZE"
