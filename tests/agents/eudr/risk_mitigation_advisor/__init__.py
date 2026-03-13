# -*- coding: utf-8 -*-
"""
Test package for AGENT-EUDR-025 Risk Mitigation Advisor Agent.

Comprehensive test suite covering all 8 processing engines, API layer,
ML pipeline, golden scenarios, performance benchmarks, determinism
verification, upstream integration, and security/RBAC validation.

Target: 920+ tests, 85%+ coverage.

Test modules:
    conftest                          -- 80+ shared fixtures
    test_strategy_selection_engine    -- Engine 1: ~70 tests
    test_remediation_plan_design_engine -- Engine 2: ~70 tests
    test_capacity_building_manager_engine -- Engine 3: ~60 tests
    test_measure_library_engine       -- Engine 4: ~60 tests
    test_effectiveness_tracking_engine -- Engine 5: ~70 tests
    test_continuous_monitoring_engine  -- Engine 6: ~60 tests
    test_cost_benefit_optimizer_engine -- Engine 7: ~70 tests
    test_stakeholder_collaboration_engine -- Engine 8: ~60 tests
    test_api_strategy_routes          -- Strategy API: ~40 tests
    test_api_remediation_routes       -- Remediation API: ~50 tests
    test_api_optimizer_routes         -- Optimizer API: ~40 tests
    test_api_integration              -- End-to-end API: ~60 tests
    test_ml_models                    -- ML pipeline: ~50 tests
    test_golden_scenarios             -- 15 golden scenarios
    test_performance                  -- Performance: ~25 tests
    test_determinism                  -- Determinism: ~35 tests
    test_integration_with_upstream    -- Upstream: ~50 tests
    test_security                     -- Security/RBAC: ~60 tests

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
"""
