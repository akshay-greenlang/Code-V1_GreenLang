"""
Property-Based Testing Module for GL-003 UnifiedSteam

This module provides comprehensive property-based tests using Hypothesis
to validate thermodynamic calculations, sensor input handling, and
state machine behavior.

Test Categories:
- Thermodynamic consistency (Maxwell relations, region boundaries)
- Input validation and fuzzing (NaN/Inf handling, boundary conditions)
- State machine transitions (safety states, optimization states)
- Calculation invariants (quality bounds, monotonicity)

Author: GL-TestEngineer
Version: 1.0.0
"""

__all__ = [
    'test_thermodynamics_properties',
    'test_input_validation',
    'test_state_machines',
]
