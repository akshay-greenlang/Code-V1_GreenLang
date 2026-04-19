# -*- coding: utf-8 -*-
"""
Compliance Tests Package for GreenLang Process Heat Agents.

This package contains comprehensive compliance validation tests for:
    - EPA 40 CFR Part 60 (NSPS - New Source Performance Standards)
    - EPA 40 CFR Part 75 (CEMS - Continuous Emission Monitoring)
    - EPA 40 CFR Part 98 (GHG Mandatory Reporting)
    - IEC 61511 (Functional Safety - Safety Instrumented Systems)
    - NFPA 85 (Boiler and Combustion Systems Hazards Code)

Test Modules:
    - test_epa_part60_compliance.py: NSPS emission limit tests
    - test_epa_part75_cems.py: CEMS data quality tests
    - test_epa_part98_ghg.py: GHG calculation accuracy tests
    - test_iec61511_safety.py: Functional safety calculation tests
    - test_nfpa85_combustion.py: Combustion safety timing tests

Usage:
    # Run all compliance tests
    pytest tests/compliance/ -v -m compliance

    # Run specific regulatory standard tests
    pytest tests/compliance/test_epa_part98_ghg.py -v

    # Run only safety-related tests
    pytest tests/compliance/test_iec61511_safety.py tests/compliance/test_nfpa85_combustion.py -v

Fixtures:
    All shared fixtures are defined in conftest.py, including:
    - EPA Part 60 emission limits
    - EPA Part 75 QA/QC requirements
    - EPA Part 98 emission factors and GWP values
    - IEC 61511 SIL targets and failure rates
    - NFPA 85 timing requirements and state transitions
    - PFD calculator for safety calculations
    - GHG calculator for emission calculations

Pass/Fail Criteria:
    Each test class documents specific pass/fail criteria based on
    regulatory requirements. Tests are marked with @pytest.mark.compliance
    for easy identification.

Author: GL-TestEngineer
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GL-TestEngineer"

# Define test categories for pytest markers
COMPLIANCE_TEST_CATEGORIES = [
    "epa_part60",    # NSPS emission standards
    "epa_part75",    # CEMS monitoring
    "epa_part98",    # GHG reporting
    "iec61511",      # Functional safety
    "nfpa85",        # Combustion safety
]

# Regulatory standards covered
REGULATORY_STANDARDS = {
    "EPA Part 60": "New Source Performance Standards (NSPS)",
    "EPA Part 75": "Continuous Emission Monitoring Systems (CEMS)",
    "EPA Part 98": "Greenhouse Gas Mandatory Reporting",
    "IEC 61511": "Functional Safety - Safety Instrumented Systems",
    "IEC 61508": "Functional Safety - General Requirements",
    "NFPA 85": "Boiler and Combustion Systems Hazards Code",
    "NFPA 86": "Standard for Ovens and Furnaces",
    "ASME PTC 4.1": "Steam Generating Units Performance Test Code",
    "API 556": "Instrumentation, Control, and Protective Systems",
    "API 560": "Fired Heaters for General Refinery Service",
}
