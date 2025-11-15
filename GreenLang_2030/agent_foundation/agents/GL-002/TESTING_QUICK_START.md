# GL-002: Boiler Efficiency Optimizer - Testing Quick Start

## Overview

This guide provides quick instructions for running tests on the GL-002 Boiler Efficiency Optimizer. The test suite ensures reliability, performance, and compliance with industrial standards.

## Prerequisites

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Required packages:
# - pytest>=7.0.0
# - pytest-asyncio>=0.21.0
# - pytest-cov>=4.0.0
# - pytest-benchmark>=4.0.0
# - pytest-mock>=3.10.0
# - hypothesis>=6.70.0
# - factory-boy>=3.2.0
# - faker>=18.0.0
```

## Quick Test Commands

### Run All Tests

```bash
# Run complete test suite
pytest

# Run with coverage report
pytest --cov=gl002 --cov-report=html

# Run in parallel (faster)
pytest -n auto

# Verbose output
pytest -v

# Run with markers
pytest -m "not slow"
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Performance tests only
pytest tests/performance/

# Security tests only
pytest tests/security/

# Smoke tests (quick validation)
pytest -m smoke
```

## Test Categories

### 1. Unit Tests

**Purpose:** Test individual components in isolation

```bash
# Run all unit tests
pytest tests/unit/ -v

# Test specific modules
pytest tests/unit/test_efficiency_calculator.py
pytest tests/unit/test_combustion_optimizer.py
pytest tests/unit/test_nox_predictor.py

# Run with coverage for specific module
pytest tests/unit/test_efficiency_calculator.py --cov=gl002.calculators.efficiency
```

**Example Unit Test:**

```python
# tests/unit/test_efficiency_calculator.py
def test_boiler_efficiency_direct_method():
    """Test efficiency calculation using direct method."""
    calculator = EfficiencyCalculator()

    result = calculator.calculate_direct({
        'steam_flow': 50000,
        'steam_pressure': 600,
        'steam_temperature': 485,
        'feedwater_temperature': 230,
        'fuel_flow': 3500,
        'fuel_heating_value': 18500
    })

    assert 84 <= result['efficiency'] <= 86
    assert result['heat_rate'] > 0
```

### 2. Integration Tests

**Purpose:** Test component interactions and external integrations

```bash
# Run all integration tests
pytest tests/integration/ -v

# Test specific integrations
pytest tests/integration/test_opc_ua_integration.py
pytest tests/integration/test_database_integration.py
pytest tests/integration/test_api_integration.py

# Skip external dependency tests
pytest tests/integration/ -m "not external"
```

**Example Integration Test:**

```python
# tests/integration/test_api_integration.py
@pytest.mark.integration
async def test_optimization_api_endpoint():
    """Test optimization API endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/optimize",
            json={
                "boiler_id": "BOILER001",
                "sensor_data": {
                    "steam_flow": 50000,
                    "fuel_flow": 3500
                }
            }
        )

    assert response.status_code == 200
    assert "recommendations" in response.json()
```

### 3. Performance Tests

**Purpose:** Validate system performance and scalability

```bash
# Run performance tests
pytest tests/performance/ -v

# Run with benchmarks
pytest tests/performance/ --benchmark-only

# Generate performance report
pytest tests/performance/ --benchmark-autosave

# Compare with baseline
pytest tests/performance/ --benchmark-compare=baseline
```

**Example Performance Test:**

```python
# tests/performance/test_optimization_performance.py
@pytest.mark.benchmark
def test_optimization_speed(benchmark):
    """Benchmark optimization speed."""
    optimizer = EfficiencyOptimizer(test_config)
    test_data = generate_sensor_data()

    result = benchmark(optimizer.optimize, test_data)

    assert result.efficiency > 0.80
    assert benchmark.stats['mean'] < 0.5  # Less than 500ms
```

### 4. Security Tests

**Purpose:** Validate security measures and vulnerability checks

```bash
# Run security tests
pytest tests/security/ -v

# Run SQL injection tests
pytest tests/security/test_sql_injection.py

# Run authentication tests
pytest tests/security/test_authentication.py

# Run with security scanning
bandit -r gl002/ -f json -o security_report.json
```

## Test Configuration

### pytest.ini Configuration

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --tb=short
    --disable-warnings
    -ra
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    security: Security tests
    smoke: Quick smoke tests
    slow: Slow running tests
    external: Tests requiring external services
asyncio_mode = auto
```

### Test Environment Setup

```bash
# Set test environment variables
export GL002_ENV=test
export DATABASE_URL=postgresql://test:test@localhost/gl002_test
export REDIS_URL=redis://localhost:6379/1
export OPC_TEST_SERVER=opc.tcp://localhost:4840
export LOG_LEVEL=DEBUG

# Or use .env.test file
cp .env.example .env.test
# Edit .env.test with test configurations
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Run tests
      run: |
        pytest --cov=gl002 --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Local CI Simulation

```bash
# Run tests as CI would
./scripts/ci_test.sh

# Script contents:
#!/bin/bash
set -e

echo "Running linters..."
flake8 gl002/
black --check gl002/
mypy gl002/

echo "Running security checks..."
bandit -r gl002/
safety check

echo "Running tests..."
pytest --cov=gl002 --cov-report=term-missing

echo "Checking coverage..."
coverage report --fail-under=90
```

## Test Data Management

### Generate Test Data

```python
# tests/factories.py
import factory
from faker import Faker

fake = Faker()

class BoilerDataFactory(factory.Factory):
    class Meta:
        model = dict

    steam_flow = factory.Faker('random_int', min=10000, max=100000)
    steam_pressure = factory.Faker('random_int', min=400, max=800)
    steam_temperature = factory.Faker('random_int', min=400, max=500)
    fuel_flow = factory.Faker('random_int', min=1000, max=5000)
    o2_percent = factory.Faker('pyfloat', min_value=2.0, max_value=5.0)
```

### Load Test Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def sample_sensor_data():
    """Load sample sensor data."""
    data_file = Path(__file__).parent / "fixtures" / "sensor_data.json"
    with open(data_file) as f:
        return json.load(f)

@pytest.fixture
def mock_opc_server():
    """Mock OPC UA server for testing."""
    with patch('gl002.integrations.opc_ua.Client') as mock:
        yield mock
```

## Coverage Reporting

### Generate Coverage Reports

```bash
# HTML report
pytest --cov=gl002 --cov-report=html
# Open htmlcov/index.html in browser

# Terminal report
pytest --cov=gl002 --cov-report=term-missing

# XML report (for CI)
pytest --cov=gl002 --cov-report=xml

# JSON report
pytest --cov=gl002 --cov-report=json
```

### Coverage Requirements

```yaml
# Minimum coverage requirements
overall: 90%
core_modules: 95%
calculators: 95%
integrations: 85%
api: 90%
utilities: 80%
```

## Debugging Tests

### Run Tests with Debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace

# Show local variables on failure
pytest -l

# Show print statements
pytest -s

# Maximum verbosity
pytest -vvv
```

### Debug Specific Test

```python
# Add breakpoint in test
def test_complex_calculation():
    data = generate_test_data()
    import pdb; pdb.set_trace()  # Breakpoint
    result = calculate_efficiency(data)
    assert result > 0.80
```

## Mock External Services

### Mock Configuration

```python
# tests/mocks.py
class MockOPCServer:
    """Mock OPC UA server for testing."""

    def __init__(self):
        self.connected = False
        self.data = {}

    def connect(self):
        self.connected = True

    def get_value(self, node_id):
        return self.data.get(node_id, 0.0)

    def set_test_data(self, data):
        self.data = data

# Usage in tests
@pytest.fixture
def mock_opc():
    server = MockOPCServer()
    server.set_test_data({
        'ns=2;s=SteamFlow': 50000,
        'ns=2;s=Pressure': 600
    })
    return server
```

## Test Best Practices

### 1. Test Naming

```python
# Good test names
def test_efficiency_calculation_with_valid_data_returns_expected_range():
def test_optimizer_handles_missing_sensor_data_gracefully():
def test_api_authentication_rejects_invalid_token():

# Bad test names
def test_1():
def test_efficiency():
def test_works():
```

### 2. Test Organization

```
tests/
├── unit/
│   ├── core/
│   ├── calculators/
│   └── utilities/
├── integration/
│   ├── api/
│   ├── database/
│   └── external/
├── performance/
│   └── benchmarks/
├── security/
│   └── vulnerabilities/
└── fixtures/
    └── data/
```

### 3. Assertion Guidelines

```python
# Use specific assertions
assert result == expected  # Not: assert result

# Use appropriate matchers
assert 0.84 <= efficiency <= 0.86  # Not: assert efficiency > 0

# Include helpful messages
assert efficiency > 0.80, f"Efficiency {efficiency} below minimum threshold"
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Add project to Python path
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   ```

2. **Database Connection Errors**
   ```bash
   # Use test database
   export DATABASE_URL=sqlite:///test.db
   ```

3. **Slow Tests**
   ```bash
   # Skip slow tests
   pytest -m "not slow"

   # Set timeout
   pytest --timeout=30
   ```

4. **Flaky Tests**
   ```bash
   # Retry flaky tests
   pytest --reruns 3 --reruns-delay 1
   ```

## Resources

- **Documentation:** https://docs.greenlang.io/gl002/testing
- **CI Dashboard:** https://ci.greenlang.io/gl002
- **Coverage Reports:** https://codecov.io/gh/greenlang/gl002
- **Test Results:** https://tests.greenlang.io/gl002

---

*Happy Testing! For questions, contact the GL-002 team at gl002-dev@greenlang.io*