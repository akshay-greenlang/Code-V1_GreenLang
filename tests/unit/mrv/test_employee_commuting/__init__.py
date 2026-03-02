# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-020 (Employee Commuting Agent).

GHG Protocol Scope 3 Category 7: Employee Commuting
Agent ID: GL-MRV-S3-007

Test modules:
- conftest.py: Shared pytest fixtures (~200 lines)
- test_models.py: Enums, constants, input/result models, helpers (55 tests)
- test_config.py: Config dataclasses, env loading, validation (40 tests)
- test_metrics.py: Prometheus metrics, NoOp fallback (25 tests)
- test_provenance.py: SHA-256 chain, Merkle tree (30 tests)
- test_employee_commuting_database.py: EF lookups (50 tests)
- test_personal_vehicle_calculator.py: Car/motorcycle calcs (50 tests)
- test_public_transit_calculator.py: Bus/rail/subway/ferry calcs (45 tests)
- test_active_transport_calculator.py: Cycling/walking/e-bike calcs (40 tests)
- test_telework_calculator.py: WFH energy calcs (45 tests)
- test_compliance_checker.py: 7-framework compliance (45 tests)
- test_employee_commuting_pipeline.py: 10-stage pipeline (40 tests)
- test_setup.py: Service facade (35 tests)
- test_api.py: API endpoints (40 tests)

Total: 540+ tests

Author: GreenLang Platform Team
Date: February 2026
"""
