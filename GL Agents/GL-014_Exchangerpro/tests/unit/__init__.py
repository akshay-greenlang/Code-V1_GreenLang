# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Unit Tests Package

Unit tests for the Heat Exchanger Optimizer agent's core calculators:
- Heat duty calculations (Q = m*Cp*dT)
- LMTD calculations with F-factor correction
- Epsilon-NTU method for all flow configurations
- UA calculations from LMTD and NTU methods
- Pressure drop calculations (Darcy-Weisbach)
- Fouling prediction ML tests
- Cleaning optimizer tests

Target coverage: 85%+

Author: GL-TestEngineer
Version: 1.0.0
"""

__all__ = [
    "test_heat_duty",
    "test_lmtd",
    "test_epsilon_ntu",
    "test_ua_calculator",
    "test_pressure_drop",
    "test_fouling_predictor",
    "test_optimizer",
]
