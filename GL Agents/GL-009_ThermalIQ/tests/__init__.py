# -*- coding: utf-8 -*-
"""
GL-009 THERMALIQ Test Suite

Comprehensive test suite for the ThermalIQ thermal efficiency calculator agent.
This test suite provides 85%+ code coverage and validates:

- First Law (energy) efficiency calculations
- Second Law (exergy) efficiency calculations
- Heat balance validation and closure
- Sankey diagram generation
- Fluid property library accuracy
- Explainability and reporting features
- API endpoints and integration
- Golden value validation against reference standards

Test Categories:
- Unit Tests: Test individual functions and methods
- Integration Tests: Test component interactions
- Performance Tests: Validate latency and throughput targets
- Compliance Tests: Validate regulatory requirements
- Property-Based Tests: Hypothesis-driven testing

Standards Compliance Testing:
- ASME PTC 4.1 - Steam Generating Units
- ASME PTC 46 - Overall Plant Performance
- ISO 50001:2018 - Energy Management Systems
- IAPWS-IF97 - Steam Property Tables

Author: GL-TestEngineer
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GL-TestEngineer"

# Test markers for categorization
TEST_MARKERS = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions",
    "performance": "Performance and benchmark tests",
    "compliance": "Regulatory compliance validation tests",
    "golden": "Golden value validation against reference data",
    "slow": "Tests that take longer than 5 seconds",
    "requires_coolprop": "Tests requiring CoolProp library",
    "requires_kafka": "Tests requiring Kafka connection",
}

# Coverage targets
COVERAGE_TARGET_PERCENT = 85

# Performance targets
PERFORMANCE_TARGETS = {
    "first_law_calculation_ms": 5.0,
    "second_law_calculation_ms": 10.0,
    "sankey_generation_ms": 50.0,
    "full_analysis_ms": 100.0,
    "api_response_ms": 200.0,
}

# Tolerance values for numerical comparisons
TOLERANCE = {
    "efficiency_percent": 0.01,  # 0.01% tolerance
    "energy_kw": 0.1,  # 0.1 kW tolerance
    "temperature_c": 0.1,  # 0.1 C tolerance
    "pressure_bar": 0.001,  # 0.001 bar tolerance
    "heat_balance_closure": 0.02,  # 2% closure per ASME PTC 4.1
}
