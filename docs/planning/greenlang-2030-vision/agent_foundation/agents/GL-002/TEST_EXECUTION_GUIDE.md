# GL-002 Test Execution Guide

Quick reference for running the comprehensive test suite with 95%+ coverage.

---

## Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-timeout pytest-xdist

# Optional for memory tests
pip install psutil
```

---

## Quick Start

### Run All Tests with Coverage
```bash
cd C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002

# Run all 375+ tests with coverage report
pytest

# Output: Coverage report showing 95%+ coverage
```

### View Coverage Report
```bash
# Generate HTML coverage report
pytest --cov-report=html

# Open in browser
start htmlcov/index.html  # Windows
# or
open htmlcov/index.html   # macOS/Linux
```

---

## Test Categories

### 1. Edge Case Tests (50+ tests)
```bash
# Run all edge case tests
pytest tests/test_edge_cases.py -v

# Run specific edge case category
pytest tests/test_edge_cases.py::TestBoundaryInputValidation -v
pytest tests/test_edge_cases.py::TestFloatPrecisionEdgeCases -v
pytest tests/test_edge_cases.py::TestExtremeValues -v
pytest tests/test_edge_cases.py::TestUnicodeAndStringEdgeCases -v
```

### 2. Error Path Tests (30+ tests)
```bash
# Run all error path tests
pytest tests/test_error_paths.py -v

# Run specific error category
pytest tests/test_error_paths.py::TestExceptionBranches -v
pytest tests/test_error_paths.py::TestTimeoutScenarios -v
pytest tests/test_error_paths.py::TestIntegrationFailures -v
pytest tests/test_error_paths.py::TestCacheCorruptionRecovery -v
```

### 3. Concurrency Tests (25+ tests)
```bash
# Run all concurrency tests
pytest tests/test_concurrency_advanced.py -v

# Run specific concurrency category
pytest tests/test_concurrency_advanced.py::TestRaceConditions -v
pytest tests/test_concurrency_advanced.py::TestDeadlockPrevention -v
pytest tests/test_concurrency_advanced.py::TestThreadStarvation -v
pytest tests/test_concurrency_advanced.py::TestCacheContention -v
```

### 4. Integration Failure Tests (20+ tests)
```bash
# Run all integration failure tests
pytest tests/test_integration_failures.py -v

# Run specific integration category
pytest tests/test_integration_failures.py::TestSCADAIntegrationFailures -v
pytest tests/test_integration_failures.py::TestDCSIntegrationFailures -v
pytest tests/test_integration_failures.py::TestERPIntegrationFailures -v
pytest tests/test_integration_failures.py::TestRetryAndBackoff -v
```

### 5. Performance Tests (15+ tests)
```bash
# Run all performance tests
pytest tests/test_performance_limits.py -v

# Run specific performance category
pytest tests/test_performance_limits.py::TestMaximumLoad -v
pytest tests/test_performance_limits.py::TestMemoryPressure -v
pytest tests/test_performance_limits.py::TestCPUThrottling -v
pytest tests/test_performance_limits.py::TestThroughputLimits -v
```

---

## Run by Test Markers

### Boundary Tests
```bash
pytest -m boundary -v
```

### Integration Tests
```bash
pytest -m integration -v
```

### Performance Tests
```bash
pytest -m performance -v
```

### Async Tests
```bash
pytest -m asyncio -v
```

### Exclude Slow Tests
```bash
pytest -m "not slow" -v
```

### Exclude Stress Tests
```bash
pytest -m "not stress" -v
```

---

## Parallel Execution

### Run Tests in Parallel (Faster)
```bash
# Use 4 workers
pytest -n 4

# Use auto-detected CPU count
pytest -n auto

# Typical speedup: 110s → 35s
```

---

## Coverage Reports

### Generate Coverage Reports
```bash
# Terminal report with missing lines
pytest --cov=. --cov-report=term-missing

# HTML report
pytest --cov=. --cov-report=html

# JSON report
pytest --cov=. --cov-report=json

# All formats
pytest --cov=. --cov-report=html --cov-report=json --cov-report=term-missing
```

### Coverage by Component
```bash
# Coverage for orchestrator only
pytest --cov=boiler_efficiency_orchestrator

# Coverage for tools only
pytest --cov=tools

# Coverage for calculators only
pytest --cov=calculators
```

---

## Specific Test Selection

### Run Single Test
```bash
pytest tests/test_edge_cases.py::test_positive_zero_vs_negative_zero -v
```

### Run Test Class
```bash
pytest tests/test_edge_cases.py::TestBoundaryInputValidation -v
```

### Run Tests Matching Pattern
```bash
pytest -k "boundary" -v
pytest -k "timeout" -v
pytest -k "cache" -v
```

---

## Debugging Tests

### Verbose Output
```bash
pytest -vv
```

### Show Print Statements
```bash
pytest -s
```

### Drop into Debugger on Failure
```bash
pytest --pdb
```

### Show Locals on Failure
```bash
pytest -l
```

### Stop at First Failure
```bash
pytest -x
```

### Show Slowest Tests
```bash
pytest --durations=10
```

---

## Coverage Verification

### Verify 95%+ Coverage
```bash
# Will fail if coverage below 95%
pytest --cov=. --cov-fail-under=95

# Check specific component
pytest --cov=boiler_efficiency_orchestrator --cov-fail-under=98
```

### Missing Coverage Report
```bash
# Show only uncovered lines
pytest --cov=. --cov-report=term-missing | grep "TOTAL"
```

---

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest --cov=. --cov-report=xml --cov-fail-under=95
      - uses: codecov/codecov-action@v3
```

### GitLab CI
```yaml
# .gitlab-ci.yml
test:
  script:
    - pip install -r requirements.txt
    - pytest --cov=. --cov-report=xml --cov-fail-under=95
  coverage: '/TOTAL.*\s+(\d+%)$/'
```

---

## Performance Benchmarking

### Run Performance Tests Only
```bash
pytest tests/test_performance_limits.py -v --durations=0
```

### Measure Test Execution Time
```bash
pytest --durations=0
```

### Profile Tests
```bash
# Install pytest-profiling
pip install pytest-profiling

# Run with profiling
pytest --profile
```

---

## Test Data Management

### Using Test Fixtures
```python
# tests/conftest.py provides:
# - extreme_values
# - invalid_data_samples
# - unicode_test_strings
# - malformed_sensor_data
# - performance_test_data
# - timeout_scenarios
# - integration_failure_scenarios
# - cache_contention_config
# - memory_pressure_config
# - benchmark_thresholds
# - metrics_collector
```

### Custom Test Data
```python
@pytest.fixture
def my_custom_data():
    return {'custom': 'data'}

def test_with_custom_data(my_custom_data):
    assert my_custom_data['custom'] == 'data'
```

---

## Troubleshooting

### Tests Hanging
```bash
# Use timeout
pytest --timeout=60  # 60 second timeout per test
```

### Import Errors
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from parent directory
cd GreenLang_2030/agent_foundation/agents
pytest GL-002/tests/
```

### Async Test Errors
```bash
# Ensure pytest-asyncio installed
pip install pytest-asyncio

# Check asyncio mode in pytest.ini
# asyncio_mode = auto
```

### Coverage Not Collected
```bash
# Ensure pytest-cov installed
pip install pytest-cov

# Check .coveragerc or pytest.ini configuration
```

---

## Best Practices

### Before Committing
```bash
# Run full test suite with coverage
pytest --cov=. --cov-fail-under=95

# Check no tests are skipped
pytest --runxfail

# Verify no warnings
pytest --strict-warnings
```

### Continuous Testing
```bash
# Watch mode (requires pytest-watch)
pip install pytest-watch
ptw -- --cov=. --cov-report=term-missing
```

### Test Organization
```
tests/
├── conftest.py              # Shared fixtures
├── test_edge_cases.py       # 50+ boundary tests
├── test_error_paths.py      # 30+ error tests
├── test_concurrency_advanced.py  # 25+ concurrency tests
├── test_integration_failures.py  # 20+ integration tests
├── test_performance_limits.py    # 15+ performance tests
├── test_boiler_efficiency_orchestrator.py  # Core tests
├── test_calculators.py      # Calculator tests
├── test_compliance.py       # Compliance tests
├── test_determinism.py      # Determinism tests
├── test_integrations.py     # Integration tests
├── test_performance.py      # Performance tests
├── test_security.py         # Security tests
└── test_tools.py           # Tools tests
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests with coverage |
| `pytest -v` | Verbose output |
| `pytest -n auto` | Parallel execution |
| `pytest -m boundary` | Run boundary tests |
| `pytest -k "cache"` | Run tests matching "cache" |
| `pytest --cov-report=html` | Generate HTML coverage report |
| `pytest --durations=10` | Show 10 slowest tests |
| `pytest -x` | Stop at first failure |
| `pytest --pdb` | Drop into debugger on failure |
| `pytest --lf` | Run last failed tests |
| `pytest --ff` | Run failures first |

---

## Expected Results

### Successful Test Run
```
================================ test session starts ================================
collected 375 items

tests/test_edge_cases.py ................................................... [ 13%]
tests/test_error_paths.py ........................................ [ 21%]
tests/test_concurrency_advanced.py .......................... [ 28%]
tests/test_integration_failures.py .................... [ 33%]
tests/test_performance_limits.py ............... [ 37%]
tests/test_boiler_efficiency_orchestrator.py ...................... [ 44%]
tests/test_calculators.py ............................................... [ 57%]
tests/test_compliance.py ....................... [ 63%]
tests/test_determinism.py .................. [ 68%]
tests/test_integrations.py ................................ [ 77%]
tests/test_performance.py ..................... [ 83%]
tests/test_security.py .............. [ 87%]
tests/test_tools.py ........................... [ 95%]

---------- coverage: platform win32, python 3.11.0 -----------
Name                                  Stmts   Miss  Cover   Missing
-------------------------------------------------------------------
boiler_efficiency_orchestrator.py      450      8    98%   102, 234, 567-570
tools.py                              380     12    97%   45, 234-238, 890-893
calculators/combustion_efficiency.py   156      4    97%   23, 156-158
calculators/emissions_calculator.py    134      3    98%   89-91
config.py                              89      0   100%
-------------------------------------------------------------------
TOTAL                                1850     45    97%

================================ 375 passed in 110.23s =================================
```

### Coverage Threshold Met
```
Required coverage of 95% reached. Total coverage: 97.57%
```

---

**Guide Version:** 1.0
**Last Updated:** 2025-11-17
**For:** GL-002 BoilerEfficiencyOptimizer (375+ tests, 95%+ coverage)
