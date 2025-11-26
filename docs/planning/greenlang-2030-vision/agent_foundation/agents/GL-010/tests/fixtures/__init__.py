# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH Test Fixtures

Test data fixtures for the EmissionsComplianceAgent test suite.
Provides standardized test data for emissions calculations,
regulatory compliance, and workflow testing.

Fixture Files:
    - emissions_test_cases.json: Known-result test cases (50+ cases)
    - regulatory_limits.json: Multi-jurisdiction limits (~300 lines)

Test Case Categories:
    - NOx Test Cases: EPA Method 19 reference values
    - SOx Test Cases: Stoichiometric S to SO2 conversions
    - CO2 Test Cases: AP-42 emission factors
    - PM Test Cases: Filterable and condensable PM
    - Compliance Test Cases: Pass/fail boundary conditions
    - Edge Cases: Zero values, maximum ranges, dilution scenarios

Regulatory Limit Sources:
    - EPA NSPS (40 CFR Part 60)
    - EPA MACT (40 CFR Part 63)
    - EU IED (Directive 2010/75/EU)
    - China MEE (GB 13223-2011)
    - State Limits (CA SCAQMD, TX TCEQ, NY DEC)

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import json
from pathlib import Path

# Fixture directory path
FIXTURES_DIR = Path(__file__).parent


def load_emissions_test_cases():
    """Load emissions test cases from JSON file."""
    with open(FIXTURES_DIR / "emissions_test_cases.json", "r") as f:
        return json.load(f)


def load_regulatory_limits():
    """Load regulatory limits from JSON file."""
    with open(FIXTURES_DIR / "regulatory_limits.json", "r") as f:
        return json.load(f)


__all__ = [
    "FIXTURES_DIR",
    "load_emissions_test_cases",
    "load_regulatory_limits",
]
