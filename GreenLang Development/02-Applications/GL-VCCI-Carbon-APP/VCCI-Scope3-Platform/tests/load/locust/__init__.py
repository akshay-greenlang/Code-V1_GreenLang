# -*- coding: utf-8 -*-
"""
GL-VCCI Load Testing Suite - Locust Test Modules
================================================

This package contains all Locust load test scenarios organized by category.

Modules:
- ingestion_tests: Data ingestion load tests (5 scenarios)
- api_tests: API endpoint load tests (5 scenarios)
- calculation_tests: Calculation engine load tests (4 scenarios)
- database_tests: Database and cache load tests (3 scenarios)
- endurance_tests: Long-running endurance tests (3 scenarios)

Total: 20 comprehensive load test scenarios

Author: GL-VCCI Team
Phase: Phase 6 - Testing & Validation
Version: 2.0
"""

__version__ = "2.0.0"
__author__ = "GL-VCCI Team"

__all__ = [
    "ingestion_tests",
    "api_tests",
    "calculation_tests",
    "database_tests",
    "endurance_tests"
]
