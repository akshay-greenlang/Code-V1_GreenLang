# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-021 (Upstream Leased Assets Agent).

Test modules:
- conftest.py: Shared pytest fixtures (~400 lines)
- test_models.py: Enums, constants, input/result models, helpers (206 tests)
- test_config.py: Configuration dataclasses, env loading, singleton (110 tests)
- test_upstream_leased_database.py: EF lookups, building EUI, fallback (56 tests)
- test_building_calculator.py: Building energy calculations (36 tests)
- test_vehicle_fleet_calculator.py: Vehicle fleet calculations (22 tests)
- test_equipment_calculator.py: Equipment energy calculations (20 tests)
- test_it_assets_calculator.py: IT asset calculations with PUE (22 tests)
- test_provenance.py: SHA-256 chain, Merkle tree, thread safety (65 tests)
- test_compliance_checker.py: Regulatory compliance checks (65 tests)
- test_upstream_leased_pipeline.py: End-to-end pipeline (45 tests)
- test_api.py: API endpoints and routes (30 tests)
- test_setup.py: Service facade and factory (28 tests)

Total: 705+ tests
"""
