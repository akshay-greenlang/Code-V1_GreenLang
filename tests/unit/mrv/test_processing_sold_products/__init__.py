# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-023 (Processing of Sold Products Agent).

GHG Protocol Scope 3, Category 10: Processing of Sold Products.
Calculates emissions from downstream processing of intermediate products
sold by the reporting company, where processing is not controlled by
the reporting company.

Agent ID: GL-MRV-S3-010
Package: greenlang.agents.mrv.processing_sold_products
API Prefix: /api/v1/processing-sold-products
DB Migration: V074
Metrics Prefix: gl_psp_
Table Prefix: gl_psp_

Test modules:
- conftest.py: Shared pytest fixtures, engine mocks, singleton resets (~1000 lines)
- test_models.py: Enums, constants, input/result models, helpers (120+ tests)
- test_config.py: Configuration dataclasses, env loading, singleton (70+ tests)
- test_processing_database.py: EF lookups, currency, CPI, chains (30+ tests)
- test_site_specific_calculator.py: Direct/energy/fuel methods (35+ tests)
- test_average_data_calculator.py: Process EF, energy intensity, chains (30+ tests)
- test_spend_based_calculator.py: EEIO, currency conversion, CPI (25+ tests)
- test_hybrid_aggregator.py: Waterfall, gap-fill, allocation, hotspot (30+ tests)
- test_compliance_checker.py: 7 frameworks, 8 DC rules, boundary (40+ tests)
- test_provenance.py: SHA-256 chain, Merkle, determinism (50+ tests)
- test_processing_pipeline.py: 10-stage pipeline, batch, portfolio (30+ tests)
- test_api.py: 20 endpoints, validation, error responses (25+ tests)
- test_setup.py: Service singleton, engine init, health check (25+ tests)

Total: 700+ tests

Author: GL-TestEngineer
Date: February 2026
"""

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"
