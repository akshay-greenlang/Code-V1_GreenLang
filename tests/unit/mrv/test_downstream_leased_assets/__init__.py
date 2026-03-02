# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-026 (Downstream Leased Assets Agent).

GHG Protocol Scope 3, Category 13: Downstream Leased Assets.
Calculates emissions from assets OWNED by the reporter and LEASED TO
other entities (reporter is LESSOR). Mirror of Cat 8 from lessor perspective.

Agent ID: GL-MRV-S3-013
Table Prefix: gl_dla_

Test modules:
- conftest.py: Shared pytest fixtures (~1,500 lines)
- test_models.py: Enums, constants, input/result models, helpers (190+ tests)
- test_config.py: Configuration dataclasses, env loading, singleton (50 tests)
- test_downstream_asset_database.py: EF lookups, EUI benchmarks (100+ tests)
- test_asset_specific_calculator.py: Metered energy calculations (35 tests)
- test_average_data_calculator.py: Benchmark-based calculations (40 tests)
- test_spend_based_calculator.py: EEIO spend-based calculations (30 tests)
- test_hybrid_aggregator.py: Multi-method aggregation (25 tests)
- test_compliance_checker.py: Regulatory compliance checks (45 tests)
- test_downstream_leased_assets_pipeline.py: E2E pipeline (25 tests)
- test_provenance.py: SHA-256 chain, Merkle tree, thread safety (58 tests)
- test_setup.py: Service facade and factory (30 tests)
- test_api.py: API endpoints and routes (25 tests)

Total: 600+ tests
"""

AGENT_ID = "GL-MRV-S3-013"
AGENT_COMPONENT = "AGENT-MRV-026"
VERSION = "1.0.0"
TABLE_PREFIX = "gl_dla_"
