# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Test Suite Package.

Comprehensive test suite for the SteamQualityController agent including:
    - Unit tests (calculators, tools, config)
    - Integration tests (connectors, orchestrator)
    - End-to-end tests (complete workflows)
    - Performance tests (benchmarks, scalability)
    - Determinism tests (reproducibility, provenance)
    - Security tests (validation, auth, safety)

Test Standards:
    - Target coverage: 95%+
    - All tests deterministic
    - Performance benchmarks with clear targets
    - OWASP Top 10 security coverage

Author: GreenLang Industrial Optimization Team
Agent ID: GL-012
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-012"

# Pytest markers used in this test suite
PYTEST_MARKERS = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions",
    "e2e": "End-to-end workflow tests",
    "performance": "Performance benchmark tests",
    "determinism": "Determinism and reproducibility tests",
    "security": "Security validation tests",
    "slow": "Tests that take longer than 5 seconds",
    "golden": "Golden file validation tests",
    "iapws": "IAPWS-IF97 standard compliance tests",
}

# Test categories
TEST_CATEGORIES = [
    "unit",
    "integration",
    "e2e",
    "performance",
    "determinism",
    "security",
]
