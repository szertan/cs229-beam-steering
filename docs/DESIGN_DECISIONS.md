# Design Decisions Log

## Project: CS229 Beam Steering with Evolution Strategies

---

### Decision 1: Angle Selection (2025-12-09)

**Context**: Need to choose which angles to train on and which to evaluate.

**Key Insight**: This is NOT a traditional ML train/test split because:
- We don't have a dataset
- We use real-time simulation (ceviche) as our "data source"
- The optimizer learns directly from physics, not from pre-collected data

**Decision**:
- **Training angles**: 0°, 90°, 180° (optimize the network to produce good designs for these)
- **Evaluation angles**: 0°, 30°, 60°, 90°, 120°, 150°, 180° (all 7 angles)

**Rationale**:
1. Primary objective: Produce good designs for 0°, 90°, 180°
2. Secondary objective: See if designs generalize to intermediate angles (30°, 60°, 120°, 150°)
3. Even if generalization fails, understanding *why* is scientifically valuable

**Reporting plan**:
- Report performance on training angles (0°, 90°, 180°) as "trained performance"
- Report performance on other angles (30°, 60°, 120°, 150°) as "generalization analysis"
- Analyze failure modes if generalization is poor

---

### Decision 2: Physical Parameters - Lab Setup (2025-12-09)

**Context**: Parameters must match the real plasma metamaterial array in the lab for sim-to-real transfer.

**Decision - Fixed Lab Parameters**:
| Parameter | Value | Source |
|-----------|-------|--------|
| Grid size | 8 × 8 rods | Lab hardware |
| Rod diameter | 2 cm | Lab hardware |
| Rod spacing | 2.1 cm (center-to-center) | Lab hardware (2cm rod + 1mm gap) |
| Operating frequency | 3.75 GHz (λ ≈ 8 cm) | Below plasma frequency for blocking |
| Max plasma frequency | 15 GHz | Jesse's voltage characterization at 30V |

**Decision - Simulation Grid**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Resolution | 25 points per lattice constant | Balance speed vs accuracy |
| Grid spacing | 0.84 mm | = 2.1cm / 25 |
| Total grid | ~425 × 400 points | Fits 8×8 array + source + probes + PML |

**Rationale**: 
- Physical parameters are fixed by lab hardware - no flexibility
- Simulation resolution can be adjusted (higher for accuracy, lower for speed)
- 25 points/lattice gives ~11 grid points per rod radius - adequate for circular geometry

---

### Decision 2b: Plasma Physics Model (2025-12-09)

**Context**: Need to map rod control parameter to electromagnetic permittivity for simulation.

**Physics Background**:

The plasma frequency depends on electron density:
$$\omega_p = \sqrt{\frac{n_e e^2}{m_e \varepsilon_0}} \propto \sqrt{n_e}$$

The permittivity follows the Drude model:
$$\varepsilon(\omega) = 1 - \frac{\omega_p^2}{\omega^2}$$

When ω < ω_p: ε < 0 → wave is evanescent (blocked)
When ω > ω_p: ε > 0 → wave propagates

**Key Empirical Finding** (from Jesse's characterization):
- Plasma frequency is approximately **linear with voltage**: ω_p ∝ V
- At 30V (max), ω_p ≈ 15 GHz

**Decision - Control Parameter ρ**:

We define ρ as **normalized plasma frequency** (NOT electron density):
```
ρ ∈ {0.0, 0.1, 0.2, ..., 1.0}    (11 discrete levels)
ω_p = ρ × 15 GHz
ε = 1 - (ω_p / ω)²
```

**Why ρ = normalized plasma frequency (not density)?**
1. Linear relationship to voltage (lab control knob)
2. Directly interpretable: ρ=0.5 means half of max plasma frequency
3. Simple mapping to permittivity
4. Easy sim-to-real transfer: ρ → V is linear

**Resulting Permittivity Range** (at f = 3.75 GHz):

| ρ | ω_p (GHz) | ε |
|---|-----------|---|
| 0.0 | 0 | 1.0 (air) |
| 0.3 | 4.5 | -0.44 |
| 0.5 | 7.5 | -3.0 |
| 0.7 | 10.5 | -6.84 |
