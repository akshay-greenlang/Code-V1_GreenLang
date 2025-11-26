# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH Integration Tests

Integration tests for component interactions and external system
integrations of the EmissionsComplianceAgent.

Test Modules:
    - test_cems_integration.py: CEMS data acquisition (18+ tests)
    - test_regulatory_reporting.py: EPA/EU reporting (15+ tests)
    - test_full_pipeline.py: End-to-end pipeline (15+ tests)

Total: 50+ integration tests

Integration Test Categories:
    - CEMS Connection: Data acquisition, calibration, quality validation
    - Regulatory Reporting: EPA ECMPS, EU E-PRTR, submission validation
    - Pipeline Integration: Multi-source aggregation, trend analysis

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

__all__ = [
    "test_cems_integration",
    "test_regulatory_reporting",
    "test_full_pipeline",
]
