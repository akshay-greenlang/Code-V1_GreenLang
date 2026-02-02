"""
GL-009 THERMALIQ Agent - Test Suite

Comprehensive test suite for the Thermal Fluid Systems Agent.
Target: 85%+ code coverage with deterministic validation.

Test Modules:
    - test_schemas.py: Input/output schema validation
    - test_config.py: Configuration validation
    - test_fluid_properties.py: Thermal fluid property calculations
    - test_exergy.py: Exergy (2nd Law) analysis
    - test_degradation.py: Fluid degradation monitoring
    - test_expansion_tank.py: Expansion tank sizing (API 660)
    - test_heat_transfer.py: Heat transfer coefficients
    - test_safety.py: Safety interlock logic
    - test_analyzer.py: Main analyzer integration

Run tests:
    pytest greenlang/agents/process_heat/gl_009_thermal_fluid/tests/ -v --cov=greenlang.agents.process_heat.gl_009_thermal_fluid
"""

__version__ = "1.0.0"
