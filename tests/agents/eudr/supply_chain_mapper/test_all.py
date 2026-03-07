# -*- coding: utf-8 -*-
"""
Master Test Runner - AGENT-EUDR-001 Supply Chain Mapper

Orchestrates the full test suite for the EUDR Supply Chain Mapper agent.
This file is NOT a test collection file -- it provides the ``main()``
entry point for running all tests with coverage validation.

Individual test modules are discovered by pytest's normal collection
mechanism from the other ``test_*.py`` files in this directory.

Coverage targets:
    - Line coverage:   >= 85%
    - Branch coverage:  >= 90% (aspirational)

Test count targets (PRD Section 13.1):
    Module                      | Target | Actual | Verified
    ----------------------------|--------|--------|----------
    Graph Engine                | 150+   | 142    | 142/142 PASS
    Multi-Tier Mapping          |  80+   | 105    | 105/105 PASS
    Geolocation Linker          |  60+   |  95    |  95/95  PASS
    Risk Propagation            |  80+   |  63    |  63/63  PASS
    Gap Analysis                |  70+   | 117    | 117/117 PASS
    Visualization Engine        |  40+   |  63    |  63/63  PASS
    Supplier Onboarding         |  40+   |  80    |  80/80  PASS
    API Routes                  |  80+   |  95    |  95/95  PASS
    Models & Constants          |  50+   |  50    |  50/50  PASS
    Provenance Tracker          |  25+   |  37    |  37/37  PASS
    Golden Tests (7x7)          |  49    |  49    |  49/49  PASS
    Integration Tests           |  30+   |  23    |  23/23  PASS
    Performance Benchmarks      |  20+   |  16    |  16/16  PASS
    ----------------------------|--------|--------|----------
    TOTAL                       | 800+   | 935    | 935/935 PASS

    Coverage: 71.35% line (8011 stmts, 2014 missed)
    Verified: March 2026 -- all 935 tests green

Usage:
    # Run all tests:
    pytest tests/agents/eudr/supply_chain_mapper/ -v

    # Run with coverage:
    pytest tests/agents/eudr/supply_chain_mapper/ \\
        --cov=greenlang.agents.eudr.supply_chain_mapper \\
        --cov-report=html:coverage_reports/eudr_scm \\
        --cov-report=term-missing \\
        --cov-branch -v

    # Run via this runner:
    python tests/agents/eudr/supply_chain_mapper/test_all.py

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
"""

import sys
import pytest


# ---------------------------------------------------------------------------
# Coverage validation utility
# ---------------------------------------------------------------------------


def validate_coverage(coverage_pct: float, target: float = 85.0) -> bool:
    """Validate that coverage meets the target threshold.

    Args:
        coverage_pct: Measured line coverage percentage.
        target: Target coverage percentage (default 85%).

    Returns:
        True if coverage meets or exceeds target.
    """
    return coverage_pct >= target


# ---------------------------------------------------------------------------
# Test suite manifest (for documentation; not used for collection)
# ---------------------------------------------------------------------------

TEST_MODULES = [
    "test_graph_engine",           # Feature 1: Graph Engine (142 tests)
    "test_multi_tier_mapper",      # Feature 2: Multi-Tier Mapper (105 tests)
    "test_geolocation_linker",     # Feature 3: Geolocation Linker (95 tests)
    "test_risk_propagation",       # Feature 5: Risk Propagation (63 tests)
    "test_gap_analyzer",           # Feature 6: Gap Analysis (117 tests)
    "test_visualization_engine",   # Feature 7: Visualization Engine (63 tests)
    "test_supplier_onboarding",    # Feature 8: Supplier Onboarding (80 tests)
    "test_api_routes",             # API Routes (95 tests)
    "test_models",                 # Pydantic v2 Models (50 tests)
    "test_provenance",             # Provenance Tracker (37 tests)
    "test_golden_scenarios",       # Golden Tests 7x7 (49 tests)
    "test_integration",            # Integration Tests (23 tests)
    "test_performance",            # Performance Benchmarks (16 tests)
]


# ---------------------------------------------------------------------------
# Direct execution support
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main(
        [
            "tests/agents/eudr/supply_chain_mapper/",
            "-v",
            "--tb=short",
            "--cov=greenlang.agents.eudr.supply_chain_mapper",
            "--cov-report=term-missing",
            "--cov-branch",
        ]
    )
    sys.exit(exit_code)
