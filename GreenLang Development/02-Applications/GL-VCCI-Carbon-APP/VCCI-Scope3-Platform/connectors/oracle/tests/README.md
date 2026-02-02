# Oracle Fusion Cloud Connector - Test Suite

Comprehensive unit tests for the Oracle Fusion Cloud connector with 50+ tests covering all components.

## Overview

- **Total Tests**: 52+
- **Target Coverage**: 90%+
- **Test Files**: 7
- **Total Lines**: ~1,350+

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest fixtures and mocks (~180 lines)
├── test_config.py              # Configuration tests (12 tests)
├── test_auth.py                # Authentication tests (15 tests)
├── test_client.py              # REST client tests (22 tests)
├── test_extractors.py          # Extractor tests (25 tests)
├── test_mappers.py             # Mapper tests (30 tests)
└── test_integration.py         # Integration tests (8 tests)
```

## Test Categories

### Configuration Tests (test_config.py)
- OAuth2 configuration creation and validation
- REST endpoint configuration
- Retry and rate limit configuration
- Environment variable loading
- Configuration validation
- Global config singleton pattern

**Key Tests**:
- `test_config_from_env` - Environment variable loading
- `test_config_default_endpoints` - Default endpoint initialization
- `test_config_validation` - Configuration validation

### Authentication Tests (test_auth.py)
- Token cache operations (set, get, invalidate, clear)
- OAuth 2.0 token acquisition
- Token caching and reuse
- Token refresh on expiration
- Authentication error handling
- Multi-environment support
- Thread-safe token management

**Key Tests**:
- `test_acquire_token_success` - Successful token acquisition
- `test_token_caching` - Token caching behavior
- `test_force_token_refresh` - Token refresh logic
- `test_multi_environment_support` - Multiple environments

### REST Client Tests (test_client.py)
- Rate limiter functionality
- Query builder operations
- GET/POST/PATCH requests
- Pagination handling
- Error handling (404, 401, 429, 503, timeout)
- Token refresh on 401
- Retry logic with exponential backoff
- Oracle-specific response parsing

**Key Tests**:
- `test_pagination` - Oracle links-based pagination
- `test_error_handling_401_with_retry` - Token refresh
- `test_query_builder_chaining` - Query parameter building
- `test_rate_limit_error` - Rate limiting

### Extractor Tests (test_extractors.py)
- Base extractor functionality
- Delta extraction by timestamp
- Query parameter building
- Pagination link extraction
- Procurement extractor (PO, Requisitions, Suppliers)
- SCM extractor (Shipments, Transport Orders)
- Financials extractor (Fixed Assets)
- Batch processing
- Error handling

**Key Tests**:
- `test_build_query_params_with_delta` - Delta extraction
- `test_get_all_records` - Pagination handling
- `test_extract_success` - Extraction result tracking
- `test_get_purchase_orders` - Procurement extraction

### Mapper Tests (test_mappers.py)
- Purchase Order mapping
- Requisition mapping
- Shipment mapping
- Transport Order mapping
- Unit standardization (Oracle → VCCI)
- Currency conversion
- Transport mode mapping
- Distance conversion (miles → km)
- Region inference
- Metadata generation
- Batch mapping

**Key Tests**:
- `test_map_purchase_order_basic` - PO mapping
- `test_convert_currency_eur_to_usd` - Currency conversion
- `test_standardize_unit_known_units` - Unit conversion
- `test_map_transport_distance_conversion` - Distance conversion

### Integration Tests (test_integration.py)
- End-to-end procurement flow
- Multi-module extraction
- Extraction + mapping pipeline
- Error recovery with retry
- Throughput/batch processing
- Delta extraction scenarios
- Multi-environment configuration

**Key Tests**:
- `test_full_procurement_flow` - Complete E2E flow
- `test_multi_module_extraction` - Multiple modules
- `test_error_recovery_with_retry` - Retry logic
- `test_throughput_batch_processing` - Large datasets

## Running Tests

### Prerequisites

```bash
pip install pytest pytest-cov pytest-mock responses
```

### Run All Tests

```bash
# From the oracle connector directory
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/oracle

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term
```

### Run Specific Test Files

```bash
# Configuration tests only
pytest tests/test_config.py -v

# Authentication tests only
pytest tests/test_auth.py -v

# REST client tests only
pytest tests/test_client.py -v

# Extractor tests only
pytest tests/test_extractors.py -v

# Mapper tests only
pytest tests/test_mappers.py -v

# Integration tests only
pytest tests/test_integration.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run specific test class
pytest tests/test_config.py::TestOracleConnectorConfig -v

# Run specific test method
pytest tests/test_auth.py::TestOracleAuthHandler::test_acquire_token_success -v

# Run tests matching pattern
pytest tests/ -k "test_pagination" -v
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html

# Open coverage report
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html

# Terminal coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Fixtures

### Configuration Fixtures
- `mock_env_vars` - Mock environment variables
- `oauth_config` - Test OAuth configuration
- `oracle_config` - Test Oracle connector configuration

### Authentication Fixtures
- `token_cache` - Fresh token cache
- `auth_handler` - Test auth handler
- `mock_oauth_response` - Mock OAuth token response

### Client Fixtures
- `rest_client` - Test REST client
- `rate_limiter` - Test rate limiter
- `query_builder` - Test query builder

### Sample Data Fixtures
- `sample_purchase_order` - Oracle PO header data
- `sample_po_line` - Oracle PO line data
- `sample_requisition` - Oracle requisition data
- `sample_shipment` - Oracle shipment data
- `sample_transport_order` - Oracle transport order data
- `sample_supplier` - Oracle supplier data

### Response Fixtures
- `oracle_rest_response_single` - Single item response
- `oracle_rest_response_paginated` - Paginated response
- `oracle_rest_response_empty` - Empty response

## Coverage Goals

| Component | Target Coverage | Critical Paths |
|-----------|----------------|----------------|
| Config | 90%+ | Environment loading, validation |
| Auth | 95%+ | Token acquisition, caching |
| Client | 90%+ | Request handling, pagination |
| Extractors | 90%+ | Delta extraction, pagination |
| Mappers | 95%+ | Data transformation, unit conversion |
| Overall | 90%+ | All critical business logic |

## Mocking Strategy

### HTTP Mocking
- Uses `responses` library for HTTP request mocking
- Mock OAuth token endpoints
- Mock Oracle REST API responses
- Mock error scenarios (404, 401, 429, 503, timeout)

### Client Mocking
- Mock Oracle REST client for extractor tests
- Mock Redis client for caching tests
- Mock auth handlers where needed

### Data Mocking
- Realistic Oracle data samples
- Cover edge cases (missing fields, invalid data)
- Multiple data formats (single, batch, paginated)

## Test Patterns

### Arrange-Act-Assert
All tests follow AAA pattern:
```python
def test_example():
    # Arrange - Setup test data and mocks
    config = OracleConnectorConfig(...)

    # Act - Execute the operation
    result = config.validate()

    # Assert - Verify results
    assert len(result) == 0
```

### Parametrized Tests
Use pytest parametrize for testing multiple scenarios:
```python
@pytest.mark.parametrize("unit,expected", [
    ("KG", "kg"),
    ("EA", "items"),
    ("LTR", "liters")
])
def test_unit_conversion(unit, expected):
    assert standardize_unit(unit) == expected
```

### Exception Testing
Test error handling properly:
```python
def test_missing_config():
    with pytest.raises(ValueError, match="required"):
        OracleConnectorConfig.from_env()
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Oracle Connector Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Best Practices

1. **Isolation**: Each test is independent
2. **Mocking**: External dependencies are mocked
3. **Clarity**: Descriptive test names explain intent
4. **Coverage**: Aim for 90%+ code coverage
5. **Speed**: Tests run quickly (<1 second each)
6. **Maintainability**: Use fixtures to reduce duplication

## Troubleshooting

### Import Errors
```bash
# Add parent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../../.."
```

### Missing Dependencies
```bash
pip install pytest pytest-cov pytest-mock responses pydantic requests
```

### Fixture Not Found
Ensure `conftest.py` is in the tests directory and properly imported.

### Response Mocking Issues
Check that `@responses.activate` decorator is used on test methods.

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Add fixtures to `conftest.py` if reusable
3. Update this README with new test counts
4. Ensure coverage remains above 90%
5. Run all tests before committing

## Version History

- **1.0.0** (2025-11-06): Initial comprehensive test suite
  - 52+ tests across 7 files
  - 90%+ coverage target
  - Full Oracle connector coverage

## Contact

For issues or questions about tests:
- Review test output and error messages
- Check fixture definitions in `conftest.py`
- Verify Oracle connector implementation matches test expectations
- Consult Oracle Fusion Cloud REST API documentation
