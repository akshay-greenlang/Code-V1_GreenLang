"""
GL-011 FUELCRAFT Test Suite

Comprehensive test suite for the GL-011 Fuel Optimization Agent.
Targets 85%+ code coverage across all modules.

Test Categories:
    - Unit Tests: Individual function and class testing
    - Integration Tests: Component interaction testing
    - Performance Tests: Throughput and latency validation
    - Compliance Tests: Regulatory requirement validation

Run tests with:
    pytest greenlang/agents/process_heat/gl_011_fuel_optimization/tests/ -v --cov

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

__all__ = [
    "test_config",
    "test_schemas",
    "test_heating_value",
    "test_fuel_pricing",
    "test_fuel_blending",
    "test_fuel_switching",
    "test_inventory",
    "test_cost_optimization",
    "test_optimizer",
]
