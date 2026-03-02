# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-025 (End-of-Life Treatment of Sold Products).

GHG Protocol Scope 3, Category 12: End-of-Life Treatment of Sold Products.
Calculates emissions from the waste disposal and treatment of products sold
by the reporting company at the end of their life, disposed of by downstream
consumers and third parties.

Agent ID: GL-MRV-S3-012
Package: greenlang.end_of_life_treatment
API Prefix: /api/v1/end-of-life-treatment
DB Migration: V076
Metrics Prefix: gl_eol_
Table Prefix: gl_eol_

Test modules:
- conftest.py: Shared pytest fixtures, engine mocks, singleton resets (~1100 lines)
- test_models.py: Enums, constants, input/result models, helpers (150+ tests)
- test_config.py: Configuration dataclasses, env loading, singleton (50+ tests)
- test_eol_product_database.py: Material EF lookups, compositions, treatment mixes (70+ tests)
- test_waste_type_specific_calculator.py: Landfill FOD, incineration, recycling (50+ tests)
- test_average_data_calculator.py: Product category composite EFs (40+ tests)
- test_producer_specific_calculator.py: EPD, take-back, EPR calculations (35+ tests)
- test_hybrid_aggregator.py: Method waterfall, avoided emissions, circularity (40+ tests)
- test_compliance_checker.py: 7 frameworks, DC rules, boundary validation (45+ tests)
- test_provenance.py: SHA-256 chain, Merkle tree, seal/verify (60+ tests)
- test_end_of_life_pipeline.py: 10-stage pipeline, batch, portfolio (25+ tests)
- test_api.py: 22 endpoints, validation, error responses (30+ tests)
- test_setup.py: Service singleton, engine init, health check (30+ tests)

Total: 600+ tests

Author: GL-TestEngineer
Date: February 2026
"""

AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_eol_"
