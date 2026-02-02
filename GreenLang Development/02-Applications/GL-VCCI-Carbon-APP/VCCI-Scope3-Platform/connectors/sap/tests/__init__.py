# -*- coding: utf-8 -*-
"""
SAP S/4HANA Connector Test Suite
GL-VCCI Scope 3 Platform

Comprehensive unit and integration tests for SAP connector.

Test Coverage:
--------------
- Configuration management and validation
- OAuth 2.0 authentication and token caching
- OData client operations (queries, pagination, error handling)
- Data extractors (MM, SD, FI modules)
- Data mappers (PO, GR, Delivery, Transport)
- Celery job scheduling and execution
- Utilities (retry logic, rate limiting, audit logging, deduplication)
- Integration scenarios

Requirements:
-------------
- pytest >= 7.0.0
- pytest-mock >= 3.10.0
- pytest-cov >= 4.0.0
- freezegun >= 1.2.0
- responses >= 0.22.0

Run Tests:
----------
# All tests with coverage
pytest connectors/sap/tests/ -v --cov=connectors.sap --cov-report=html

# Specific test file
pytest connectors/sap/tests/test_config.py -v

# With markers
pytest connectors/sap/tests/ -v -m "not integration"

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

__version__ = "1.0.0"
__author__ = "GL-VCCI Development Team"
