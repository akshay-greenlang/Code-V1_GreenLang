# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Unit Tests Package.

This package contains comprehensive unit tests for the SteamQualityController agent,
targeting 95%+ code coverage across all calculators and components.

Test Modules:
- test_steam_quality_calculator: Dryness fraction, superheat, steam state tests
- test_desuperheater_calculator: Injection rate, control signal tests
- test_pressure_control_calculator: Valve position, PID control tests
- test_moisture_analyzer: Moisture content, condensation risk tests
- test_tools: SteamQualityTools method tests
- test_config: Configuration validation tests

Standards Compliance:
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- ASME PTC 19.11: Steam and Water Sampling, Conditioning, and Analysis
- ISO 11042: Gas turbines - Exhaust gas emission

Coverage Target: 95%+
Determinism: All tests verify bit-perfect reproducibility

Author: GL-TestEngineer
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'GL-TestEngineer'
