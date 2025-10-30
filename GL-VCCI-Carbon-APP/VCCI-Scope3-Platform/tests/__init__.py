# GL-VCCI Test Suite
# Comprehensive test suite for VCCI Scope 3 Platform

"""
VCCI Test Suite
===============

Comprehensive test coverage for the VCCI Scope 3 Platform.

Test Structure:
--------------
tests/
  ├── agents/              # Agent-level tests (1,200+ tests)
  │   ├── test_intake_agent.py
  │   ├── test_calculator_agent.py
  │   ├── test_hotspot_agent.py
  │   ├── test_engagement_agent.py
  │   └── test_reporting_agent.py
  ├── connectors/          # ERP connector tests (150+ tests)
  │   ├── test_sap_connector.py
  │   ├── test_oracle_connector.py
  │   └── test_workday_connector.py
  ├── integration/         # End-to-end integration tests (50+ scenarios)
  │   ├── test_e2e_pipeline.py
  │   └── test_multi_tenant.py
  ├── performance/         # Performance and load tests
  │   ├── test_throughput.py
  │   └── test_scalability.py
  └── security/            # Security tests
      ├── test_authentication.py
      └── test_encryption.py

Running Tests:
-------------
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test module
pytest tests/agents/test_calculator_agent.py -v

# Run integration tests only
pytest tests/integration/ -v

# Run performance tests
pytest tests/performance/ -v --benchmark-only
```

Test Goals:
----------
- Unit test coverage: >90%
- Integration test coverage: All critical paths
- Performance benchmarks: All agents meet SLAs
- Security tests: All attack vectors covered

Fixtures:
---------
Common test fixtures are available in conftest.py:
- sample_procurement_data
- sample_emission_factors
- mock_sap_connection
- test_database
- mock_llm_client
"""

__version__ = "1.0.0"
