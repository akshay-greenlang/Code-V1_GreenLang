# SAP S/4HANA Connector Test Suite

Comprehensive unit and integration tests for the SAP S/4HANA connector for the GL-VCCI Scope 3 Carbon Platform.

## Overview

**Test Statistics:**
- **Total Files:** 9 test modules + 1 fixture file
- **Total Tests:** 163 tests
- **Line Count:** ~2,030 lines
- **Coverage Target:** 90%+ overall, 95%+ for critical paths

## Test Structure

```
connectors/sap/tests/
├── __init__.py                 # Test suite initialization
├── conftest.py                 # Shared fixtures and mocks (~200 lines)
├── test_config.py              # Configuration tests (15 tests)
├── test_auth.py                # Authentication tests (18 tests)
├── test_client.py              # OData client tests (25 tests)
├── test_extractors.py          # Extractor tests (30 tests)
├── test_mappers.py             # Mapper tests (35 tests)
├── test_jobs.py                # Celery job tests (20 tests)
├── test_utils.py               # Utility tests (25 tests)
├── test_integration.py         # Integration tests (10 tests)
└── README.md                   # This file
```

## Test Coverage

### Configuration Tests (test_config.py)
- OAuth2Config validation
- RetryConfig validation with exponential backoff
- RateLimitConfig validation
- ODataEndpoint configuration
- Environment-based configuration loading
- Configuration validation and error handling

### Authentication Tests (test_auth.py)
- TokenCache thread-safe operations
- OAuth token acquisition and caching
- Token refresh on expiration
- Token invalidation
- Multi-environment token management
- Authentication error handling (401, 403, timeouts)

### OData Client Tests (test_client.py)
- OData query building ($filter, $select, $top, $skip, $orderby, $expand)
- Pagination handling (@odata.nextLink)
- GET/POST operations
- Rate limiting enforcement
- Retry logic with exponential backoff
- Timeout and connection error handling
- Authentication integration
- HTTP error handling (4xx, 5xx)

### Extractor Tests (test_extractors.py)
- Base extractor functionality
- MM extractor (PO, GR, Vendor, Material)
- SD extractor (Delivery, Transport)
- FI extractor (Fixed Assets)
- Delta extraction with timestamps
- Batch processing and pagination
- Field selection optimization
- Error handling and retry logic
- Pydantic model validation

### Mapper Tests (test_mappers.py)
- Purchase Order mapper (SAP → procurement_v1.0)
- Goods Receipt mapper (SAP → logistics_v1.0)
- Delivery mapper (SAP → logistics_v1.0)
- Transport mapper (SAP → logistics_v1.0)
- Unit standardization (17 SAP units → VCCI units)
- Currency conversion (8 currencies → USD)
- Transport mode mapping (10 SAP types → ISO 14083)
- Region inference
- Missing data handling
- Batch mapping
- Metadata and lineage generation

### Job Tests (test_jobs.py)
- Celery task execution
- Delta sync jobs (PO, Deliveries, Capital Goods)
- Job scheduling configuration
- Job failure handling and retries
- Progress tracking
- Timestamp management (last sync)
- Health check tasks

### Utility Tests (test_utils.py)
- Retry logic with exponential backoff
- Backoff calculation and jitter
- Rate limiter (token bucket)
- Audit logger (API calls, auth events, errors, lineage)
- Deduplication cache
- Batch operations

### Integration Tests (test_integration.py)
- End-to-end pipeline (Extract → Map → Ingest)
- SAP sandbox connection (stubbed)
- Throughput scenarios (100K records/hour)
- Multi-module extraction (MM, SD, FI)
- Error recovery scenarios
- Data quality validation

## Requirements

Install test dependencies:

```bash
pip install pytest pytest-mock pytest-cov freezegun responses
```

Or from requirements file:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run All Tests

```bash
# Run all tests with verbose output
pytest connectors/sap/tests/ -v

# Run with coverage report
pytest connectors/sap/tests/ -v --cov=connectors.sap --cov-report=html

# Run with coverage report (terminal)
pytest connectors/sap/tests/ -v --cov=connectors.sap --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Configuration tests only
pytest connectors/sap/tests/test_config.py -v

# Authentication tests only
pytest connectors/sap/tests/test_auth.py -v

# OData client tests only
pytest connectors/sap/tests/test_client.py -v

# All mapper tests
pytest connectors/sap/tests/test_mappers.py -v
```

### Run Tests by Marker

```bash
# Run only unit tests (fast)
pytest connectors/sap/tests/ -v -m "unit"

# Run only integration tests (slow)
pytest connectors/sap/tests/ -v -m "integration"

# Skip integration tests
pytest connectors/sap/tests/ -v -m "not integration"

# Run only authentication tests
pytest connectors/sap/tests/ -v -m "auth"
```

### Run Specific Test Classes or Methods

```bash
# Run specific test class
pytest connectors/sap/tests/test_config.py::TestSAPConnectorConfig -v

# Run specific test method
pytest connectors/sap/tests/test_auth.py::TestTokenCache::test_should_store_and_retrieve_token -v
```

### Run Tests with Different Verbosity

```bash
# Minimal output
pytest connectors/sap/tests/

# Verbose output
pytest connectors/sap/tests/ -v

# Very verbose output (shows test docstrings)
pytest connectors/sap/tests/ -vv

# Show print statements
pytest connectors/sap/tests/ -v -s
```

## Coverage Reports

### Generate HTML Coverage Report

```bash
pytest connectors/sap/tests/ --cov=connectors.sap --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Generate Coverage Report with Missing Lines

```bash
pytest connectors/sap/tests/ --cov=connectors.sap --cov-report=term-missing
```

### Coverage Thresholds

```bash
# Fail if coverage below 90%
pytest connectors/sap/tests/ --cov=connectors.sap --cov-fail-under=90
```

## Test Fixtures

### Available Fixtures (conftest.py)

**Configuration Fixtures:**
- `mock_env_vars`: Mock environment variables
- `sap_config`: SAP connector configuration
- `oauth_config`: OAuth2 configuration

**Authentication Fixtures:**
- `mock_oauth_token_response`: Mock OAuth token response
- `mock_oauth_error_response`: Mock OAuth error response
- `token_cache`: Fresh token cache instance

**Client Fixtures:**
- `mock_sap_client`: Mock SAP OData client
- `odata_query_builder`: OData query builder

**Sample Data Fixtures:**
- `sample_po_data`: Sample purchase order data
- `sample_gr_data`: Sample goods receipt data
- `sample_delivery_data`: Sample delivery data
- `sample_transport_data`: Sample transport data
- `sample_vendor_data`: Sample vendor master data
- `sample_material_data`: Sample material master data

**OData Response Fixtures:**
- `mock_odata_response_single`: Mock single entity response
- `mock_odata_response_collection`: Mock collection response
- `mock_odata_response_paginated`: Mock paginated response
- `mock_odata_error_response`: Mock error response

**Infrastructure Fixtures:**
- `mock_redis_client`: Mock Redis client
- `mock_celery_app`: Mock Celery application
- `mock_db_session`: Mock database session
- `frozen_time`: Freeze time for testing
- `mock_responses`: Mock HTTP responses

## Test Data

Sample SAP data is provided in fixtures to match actual SAP S/4HANA OData structures:

- Purchase Orders with line items
- Goods Receipts with movement data
- Outbound Deliveries with shipping info
- Transportation Orders with route details
- Vendor Master data
- Material Master data

## Writing New Tests

### Test Naming Convention

```python
def test_should_{what_it_does}_when_{condition}():
    """Test description."""
    # Arrange
    # Act
    # Assert
```

### Example Test

```python
def test_should_map_po_to_procurement_record(sample_po_data):
    """Test mapping SAP PO to VCCI procurement schema."""
    # Arrange
    mapper = PurchaseOrderMapper(tenant_id="tenant-001")
    po = sample_po_data[0]

    # Act
    records = mapper.map_purchase_order(po)

    # Assert
    assert len(records) == 1
    assert records[0].procurement_id == "PROC-4500000001-00010"
    assert records[0].supplier_name == "Acme Corporation"
```

### Using Fixtures

```python
def test_with_mock_client(mock_sap_client, sample_po_data):
    """Test using mock SAP client."""
    mock_sap_client.get.return_value = sample_po_data

    result = mock_sap_client.get("purchase_orders")

    assert result == sample_po_data
```

### Parametrized Tests

```python
@pytest.mark.parametrize("sap_unit,vcci_unit", [
    ("KG", "kg"),
    ("TO", "tonnes"),
    ("EA", "items"),
])
def test_should_standardize_unit(sap_unit, vcci_unit):
    """Test unit standardization."""
    mapper = PurchaseOrderMapper()

    result = mapper._standardize_unit(sap_unit)

    assert result == vcci_unit
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: SAP Connector Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Run tests with coverage
      run: |
        pytest connectors/sap/tests/ -v \
          --cov=connectors.sap \
          --cov-report=xml \
          --cov-fail-under=90

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:/path/to/Code-V1_GreenLang"
```

**Missing Dependencies:**
```bash
pip install pytest pytest-mock pytest-cov freezegun responses
```

**Fixture Not Found:**
- Check `conftest.py` for fixture definition
- Ensure fixture is in scope (same directory or parent)

**Test Discovery Issues:**
```bash
# Run with test discovery debug
pytest --collect-only connectors/sap/tests/
```

## Best Practices

1. **Test Isolation:** Each test should be independent
2. **Mock External Dependencies:** Use mocks for HTTP, Redis, Celery, DB
3. **Descriptive Names:** Use descriptive test names and docstrings
4. **Arrange-Act-Assert:** Follow AAA pattern
5. **Parametrize:** Use parametrized tests for similar scenarios
6. **Coverage:** Aim for 90%+ coverage, 95%+ for critical paths
7. **Fast Tests:** Unit tests should run in milliseconds
8. **Integration Tests:** Mark with `@pytest.mark.integration`

## Contact

For questions or issues with tests:
- Development Team: GL-VCCI Development Team
- Phase: 4 (Weeks 24-26)
- Date: 2025-11-06

## Version

Version: 1.0.0
