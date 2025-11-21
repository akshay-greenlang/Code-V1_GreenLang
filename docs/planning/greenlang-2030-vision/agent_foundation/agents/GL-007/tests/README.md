# GL-007 FurnacePerformanceMonitor Test Suite

Comprehensive test suite for GL-007 FurnacePerformanceMonitor agent with **85%+ coverage target**.

## Test Suite Overview

| Test Category | Files | Tests | Coverage Target | Status |
|--------------|-------|-------|----------------|--------|
| Unit Tests | 3 | 60+ | 90%+ | Ready |
| Integration Tests | 1 | 20+ | 85%+ | Ready |
| E2E Tests | 1 | 15+ | 80%+ | Ready |
| **TOTAL** | **5** | **95+** | **85%+** | **Ready** |

## Quick Start

### 1. Install Dependencies

```bash
cd GreenLang_2030/agent_foundation/agents/GL-007
pip install -r requirements-test.txt
```

Required packages:
- pytest>=7.4.0
- pytest-cov>=4.1.0
- pytest-asyncio>=0.21.0
- pytest-benchmark>=4.0.0
- pytest-xdist>=3.3.0
- pytest-timeout>=2.1.0
- numpy>=1.24.0
- faker>=19.0.0

### 2. Run All Tests

```bash
# Run full test suite with coverage
pytest

# Run with verbose output
pytest -v

# Run specific test category
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/e2e/           # E2E tests only
```

### 3. View Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Open report in browser
open tests/coverage_html/index.html  # macOS
start tests/coverage_html/index.html # Windows
```

## Test Structure

```
tests/
├── __init__.py                          # Test suite initialization
├── conftest.py                          # Shared fixtures and configuration
├── pytest.ini                           # Pytest configuration
├── README.md                            # This file
│
├── fixtures/                            # Test data fixtures
│   ├── thermal_efficiency_test_cases.json
│   ├── anomaly_detection_test_cases.json
│   └── ...
│
├── unit/                                # Unit tests (90%+ coverage)
│   ├── __init__.py
│   ├── test_thermal_efficiency.py      # 25+ tests
│   ├── test_fuel_consumption.py        # 20+ tests
│   └── test_anomaly_detection.py       # 15+ tests
│
├── integration/                         # Integration tests (85%+ coverage)
│   ├── __init__.py
│   └── test_external_systems.py        # 20+ tests
│
└── e2e/                                 # End-to-end tests (80%+ coverage)
    ├── __init__.py
    └── test_complete_workflows.py      # 15+ tests
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual functions and methods in isolation.

**Files:**
- `test_thermal_efficiency.py` - Thermal efficiency calculation tests
- `test_fuel_consumption.py` - Fuel consumption analysis tests
- `test_anomaly_detection.py` - Anomaly detection tests

**Coverage:**
- ✅ Input validation
- ✅ Calculation accuracy (ASME PTC 4.1 compliance)
- ✅ Error handling
- ✅ Edge cases
- ✅ Provenance tracking
- ✅ Parametrized tests for multiple fuel types

**Run:**
```bash
pytest tests/unit/ -v
```

### Integration Tests (`tests/integration/`)

Test integration with external systems and other agents.

**Files:**
- `test_external_systems.py` - DCS, CEMS, CMMS, ERP integration tests

**Coverage:**
- ✅ DCS/PLC integration (OPC UA)
- ✅ CEMS integration (Modbus)
- ✅ CMMS integration (work order creation)
- ✅ ERP integration (fuel pricing, scheduling)
- ✅ Agent coordination (GL-001, GL-002, GL-004, GL-005, GL-006)
- ✅ Data persistence

**Run:**
```bash
pytest tests/integration/ -v
```

### End-to-End Tests (`tests/e2e/`)

Test complete workflows from data ingestion to actionable insights.

**Files:**
- `test_complete_workflows.py` - Complete workflow tests

**Workflows:**
- ✅ Real-time monitoring workflow
- ✅ Predictive maintenance workflow
- ✅ Optimization workflow
- ✅ Compliance reporting workflow
- ✅ Multi-furnace coordination workflow

**Run:**
```bash
pytest tests/e2e/ -v
```

## Test Markers

Tests are marked with custom markers for selective execution:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only e2e tests
pytest -m e2e

# Run only performance tests
pytest -m performance

# Run only compliance tests
pytest -m compliance

# Run only ASME PTC 4.1 tests
pytest -m asme_ptc

# Exclude slow tests
pytest -m "not slow"
```

### Available Markers

- `unit` - Unit tests
- `integration` - Integration tests
- `e2e` - End-to-end tests
- `performance` - Performance benchmarks
- `compliance` - Regulatory compliance tests
- `accuracy` - Calculation accuracy tests
- `safety` - Safety and error handling tests
- `slow` - Tests taking >1 second
- `asme_ptc` - ASME PTC 4.1 compliance tests
- `iso_50001` - ISO 50001 compliance tests
- `epa_cems` - EPA CEMS compliance tests

## Key Test Fixtures

### Agent Fixtures
- `agent_config` - Test agent configuration
- `mock_agent` - Mock GL-007 agent instance

### Furnace Data Fixtures
- `sample_furnace_data` - Standard furnace operating data
- `sample_thermal_efficiency_input` - Thermal efficiency calculation input
- `sample_fuel_consumption_data` - 24 hours of fuel consumption data
- `sample_equipment_inventory` - Equipment list for maintenance prediction
- `sample_condition_monitoring_data` - Condition monitoring sensor data
- `sample_multi_furnace_data` - Multi-furnace fleet data

### External System Mocks
- `mock_dcs_client` - Mock DCS/PLC client
- `mock_cems_client` - Mock CEMS client
- `mock_cmms_client` - Mock CMMS client
- `mock_erp_client` - Mock ERP client

### Validation Helpers
- `assert_thermal_efficiency_valid` - Validate efficiency calculation results
- `assert_fuel_consumption_valid` - Validate consumption analysis results
- `assert_provenance_deterministic` - Validate deterministic provenance

### Test Data Generators
- `test_data_generator` - Generate realistic test data
  - `generate_furnace_timeseries()` - Generate time series data with optional anomalies

## Performance Benchmarks

Performance targets for GL-007:

| Operation | Target Latency | Actual | Status |
|-----------|---------------|--------|--------|
| Thermal Efficiency Calc | <50ms | TBD | Target |
| Fuel Consumption Analysis | <100ms | TBD | Target |
| Maintenance Prediction | <200ms | TBD | Target |
| Anomaly Detection | <80ms | TBD | Target |
| Multi-Furnace Optimization | <3000ms | TBD | Target |

**Run performance tests:**
```bash
pytest -m performance --benchmark-only
```

## Compliance Testing

### ASME PTC 4.1 Compliance

Tests validate thermal efficiency calculations meet ASME PTC 4.1 uncertainty requirements (±1.5%).

**Run:**
```bash
pytest -m asme_ptc
```

### ISO 50001 Compliance

Tests validate energy performance indicators (EnPIs) follow ISO 50001 methodology.

**Run:**
```bash
pytest -m iso_50001
```

### EPA CEMS Compliance

Tests validate emissions monitoring and reporting meet EPA CEMS requirements.

**Run:**
```bash
pytest -m epa_cems
```

## Test Data Files

### Thermal Efficiency Test Cases
`fixtures/thermal_efficiency_test_cases.json`

5 test cases with known expected values:
- Baseline natural gas furnace
- High efficiency operation
- Degraded performance
- Coal-fired furnace
- Hydrogen furnace (future)

### Anomaly Detection Test Cases
`fixtures/anomaly_detection_test_cases.json`

5 test cases for anomaly detection:
- Temperature spike
- Efficiency drift
- CO emissions spike
- Normal operation (no false positives)
- Multiple simultaneous anomalies

## Running Tests in CI/CD

### GitHub Actions

```yaml
name: GL-007 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements-test.txt
      - run: pytest --cov --cov-fail-under=85
      - run: pytest --benchmark-only
```

### GitLab CI

```yaml
test:
  script:
    - pip install -r requirements-test.txt
    - pytest --cov --cov-fail-under=85 --junitxml=report.xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      junit: report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## Test Development Guidelines

### Writing New Tests

1. **Name tests descriptively:**
   ```python
   def test_thermal_efficiency_with_high_stack_temperature():
       """Test efficiency degradation with high stack temperature."""
   ```

2. **Use fixtures for test data:**
   ```python
   def test_calculation(sample_furnace_data, assert_thermal_efficiency_valid):
       result = calculate_efficiency(sample_furnace_data)
       assert_thermal_efficiency_valid(result)
   ```

3. **Test both success and failure cases:**
   ```python
   def test_valid_input():
       result = calculate(valid_input)
       assert result.success

   def test_invalid_input():
       with pytest.raises(ValueError):
           calculate(invalid_input)
   ```

4. **Use parametrize for multiple scenarios:**
   ```python
   @pytest.mark.parametrize("fuel_type,expected_range", [
       ("natural_gas", (78, 88)),
       ("coal", (70, 80)),
   ])
   def test_efficiency_by_fuel(fuel_type, expected_range):
       ...
   ```

### Test Coverage Best Practices

- ✅ Test all public methods
- ✅ Test error conditions
- ✅ Test edge cases (zero, negative, maximum values)
- ✅ Test integration points
- ✅ Test compliance requirements
- ✅ Validate calculation accuracy
- ✅ Test performance targets

## Troubleshooting

### Common Issues

**Issue: Tests failing with import errors**
```bash
# Solution: Install test dependencies
pip install -r requirements-test.txt
```

**Issue: Coverage below 85%**
```bash
# Solution: Identify uncovered code
pytest --cov --cov-report=term-missing

# Focus on uncovered lines
```

**Issue: Slow test execution**
```bash
# Solution: Run in parallel
pytest -n auto

# Skip slow tests during development
pytest -m "not slow"
```

**Issue: Fixtures not found**
```bash
# Solution: Ensure conftest.py is in tests/ directory
# Check fixture scope (session, module, function)
```

## Code Coverage Report

After running tests, view detailed coverage:

```bash
# Generate HTML report
pytest --cov=src --cov-report=html

# View report
open tests/coverage_html/index.html
```

**Coverage Goals:**
- Overall: **85%+**
- Unit tests: **90%+**
- Integration tests: **85%+**
- E2E tests: **80%+**

## Continuous Monitoring

### Pre-commit Hooks

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
pytest tests/unit/ -v
if [ $? -ne 0 ]; then
    echo "Unit tests failed. Commit aborted."
    exit 1
fi
```

### Test Reports

Generate test reports:

```bash
# JUnit XML report
pytest --junitxml=tests/report.xml

# HTML report
pytest --html=tests/report.html --self-contained-html
```

## Support

For test-related questions:
- Team: GreenLang QA Team
- Slack: #gl-furnace-monitor-tests
- Email: qa@greenlang.ai

## Version History

- **v1.0.0** (2025-11-21) - Initial comprehensive test suite
  - 95+ tests across unit, integration, e2e
  - 85%+ coverage target
  - Full fixture library
  - Complete documentation

---

**Created:** 2025-11-21
**Last Updated:** 2025-11-21
**Version:** 1.0.0
**Status:** Production-Ready ✅
