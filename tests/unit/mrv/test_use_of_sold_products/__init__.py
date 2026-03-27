# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-024 (Use of Sold Products Agent).

GHG Protocol Scope 3, Category 11: Use of Sold Products.
Calculates total expected lifetime emissions from the USE of goods and
services sold by the reporting company in the reporting period.

Agent ID: GL-MRV-S3-011
Package: greenlang.agents.mrv.use_of_sold_products
API Prefix: /api/v1/use-of-sold-products
DB Migration: V075
Metrics Prefix: gl_usp_
Table Prefix: gl_usp_

Test modules:
- conftest.py: Shared pytest fixtures, engine mocks, singleton resets (~1000 lines)
- test_models.py: Enums, constants, input/result models, helpers (120+ tests)
- test_config.py: Configuration dataclasses, env loading, singleton (70+ tests)
- test_product_use_database.py: Product profiles, fuel EFs, refrigerant GWPs (35+ tests)
- test_direct_emissions_calculator.py: Fuel combustion, refrigerant leak, chemical (35+ tests)
- test_indirect_emissions_calculator.py: Electricity, heating, steam/cooling (30+ tests)
- test_fuels_feedstocks_calculator.py: Fuel sales, feedstock oxidation (25+ tests)
- test_lifetime_modeling.py: Lifetimes, degradation, Weibull, fleet (30+ tests)
- test_compliance_checker.py: 7 frameworks, 8 DC rules, boundary (40+ tests)
- test_provenance.py: SHA-256 chain, Merkle, determinism (50+ tests)
- test_use_of_sold_products_pipeline.py: 10-stage pipeline, batch (30+ tests)
- test_api.py: 22 endpoints, validation, error responses (25+ tests)
- test_setup.py: Service singleton, engine init, health check (25+ tests)

Total: 700+ tests

Author: GL-TestEngineer
Date: February 2026
"""

AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_usp_"
