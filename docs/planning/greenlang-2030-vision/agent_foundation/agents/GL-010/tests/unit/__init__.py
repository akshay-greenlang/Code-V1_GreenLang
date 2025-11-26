# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH Unit Tests

Unit tests for individual components of the EmissionsComplianceAgent.
Each test module focuses on a specific calculator or component with
85%+ coverage target.

Test Modules:
    - test_nox_calculator.py: NOx emission calculations (25+ tests)
    - test_sox_calculator.py: SOx emission calculations (20+ tests)
    - test_co2_calculator.py: CO2 emission calculations (25+ tests)
    - test_particulate_calculator.py: PM calculations (18+ tests)
    - test_emission_factors.py: Emission factor lookups (22+ tests)
    - test_compliance_checker.py: Compliance checking (25+ tests)
    - test_violation_detector.py: Violation detection (20+ tests)
    - test_orchestrator.py: Agent orchestration (28+ tests)
    - test_tools.py: Tool wrapper validation (24+ tests)

Total: 200+ unit tests

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

__all__ = [
    "test_nox_calculator",
    "test_sox_calculator",
    "test_co2_calculator",
    "test_particulate_calculator",
    "test_emission_factors",
    "test_compliance_checker",
    "test_violation_detector",
    "test_orchestrator",
    "test_tools",
]
