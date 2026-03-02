# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-022: Downstream Transportation & Distribution Agent.

Tests the GL-MRV-S3-009 agent for GHG Protocol Scope 3 Category 9
downstream transportation and distribution emissions calculations.

Package: greenlang.downstream_transportation
Agent ID: GL-MRV-S3-009
API: /api/v1/downstream-transportation
DB Migration: V073
Metrics Prefix: gl_dto_

Test modules:
- conftest.py: Shared pytest fixtures (~650 lines)
- test_models.py: Enums, constants, Pydantic models (~200 tests, ~1800 lines)
- test_config.py: Configuration, singleton, env overrides (~100 tests, ~950 lines)
- test_downstream_transport_database.py: EF lookups (~60 tests, ~600 lines)
- test_distance_based_calculator.py: Tonne-km calculations (~60 tests, ~700 lines)
- test_spend_based_calculator.py: EEIO spend-based calculations (~40 tests, ~500 lines)
- test_average_data_calculator.py: Industry average channel calc (~40 tests, ~500 lines)
- test_warehouse_distribution.py: DC/cold storage/last-mile (~50 tests, ~600 lines)
- test_compliance_checker.py: 7-framework compliance checks (~65 tests, ~700 lines)
- test_provenance.py: SHA-256 chain, Merkle tree (~60 tests, ~800 lines)
- test_downstream_transport_pipeline.py: End-to-end pipeline (~50 tests, ~600 lines)
- test_api.py: API endpoints and routes (~30 tests, ~500 lines)
- test_setup.py: Service facade and factory (~25 tests, ~250 lines)

Total: 700+ tests
"""
