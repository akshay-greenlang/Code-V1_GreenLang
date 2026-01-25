# -*- coding: utf-8 -*-
"""GL-015 INSULSCAN - Property-based tests.

Property-based test package using Hypothesis for testing physical invariants.
Validates that calculations respect fundamental physical laws and constraints.

Invariants Tested:
    - Heat loss is non-negative (when operating temp > ambient)
    - Condition score bounded in [0, 100]
    - Payback period is positive when savings > 0
    - Thicker insulation reduces heat loss
    - Larger surface area increases heat loss
    - Higher temperature differential increases heat loss
    - Insulated surface loss < bare surface loss

Physical Laws Verified:
    - First Law of Thermodynamics (energy conservation)
    - Fourier's Law (heat conduction)
    - Stefan-Boltzmann Law (thermal radiation)
    - Newton's Law of Cooling (convection)

Author: GL-TestEngineer
Version: 1.0.0
"""

__all__ = [
    "test_heat_loss_properties",
    "test_invariants",
]
