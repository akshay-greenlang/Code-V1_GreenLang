# Testing Quick Start Guide

Quick reference for running tests in GreenLang Agent Factory 5.0.

---

## Installation

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

---

## Running Tests

### All Tests
```bash
pytest
```

### By Test Type
```bash
# Unit tests (fast, <5ms each)
pytest -m unit

# Integration tests (require services)
pytest -m integration

# End-to-end tests (full workflows)
pytest -m e2e

# Performance tests
pytest -m performance

# Security tests
pytest -m security

# Slow tests only
pytest -m slow
```

### By Directory
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/

# Security tests
pytest tests/security/
```

### Specific Files
```bash
# LLM integration tests
pytest tests/unit/core/test_llm_integration.py

# Agent pipeline tests
pytest tests/integration/test_agent_pipeline.py

# Single test
pytest tests/unit/core/test_llm_integration.py::TestAnthropicProvider::test_successful_api_call
```

---

## Coverage

### Generate Coverage Report
```bash
# HTML report
pytest --cov=greenlang_core --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=greenlang_core --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=greenlang_core --cov-report=xml
```

### Coverage Threshold
```bash
# Fail if coverage < 85%
pytest --cov-fail-under=85
```

---

## Performance Testing

### Load Testing with Locust
```bash
# Web UI
locust -f tests/performance/test_load_stress.py

# Headless
locust -f tests/performance/test_load_stress.py \
  --headless \
  --users 1000 \
  --spawn-rate 10 \
  --run-time 5m \
  --host https://api.greenlang.example.com
```

### Benchmarking
```bash
# Run benchmarks
pytest tests/performance/ --benchmark-only

# Save benchmark
pytest tests/performance/ --benchmark-save=baseline

# Compare to baseline
pytest tests/performance/ --benchmark-compare=baseline
```

---

## Security Testing

### Static Analysis
```bash
# Bandit (Python security)
bandit -r greenlang_core/

# Safety (dependencies)
safety check

# Gitleaks (secrets)
gitleaks detect --source .
```

### Dynamic Analysis
```bash
# OWASP ZAP baseline scan
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable \
  zap-baseline.py \
  -t https://api.greenlang.example.com
```

---

## Test Utilities

### Watch Mode
```bash
# Re-run tests on file change
pytest-watch
```

### Parallel Execution
```bash
# Run tests in parallel
pytest -n auto
```

### Verbose Output
```bash
# Show detailed output
pytest -vv

# Show local variables on failure
pytest -vv --showlocals
```

### Stop on First Failure
```bash
pytest -x
```

### Debug Mode
```bash
# Drop into debugger on failure
pytest --pdb
```

---

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run tests
  run: |
    pytest tests/ \
      --cov=greenlang_core \
      --cov-report=xml \
      --junitxml=junit.xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

---

## Test Data

### Generate Test Data
```python
from tests.fixtures.data_factory import TestDataFactory

factory = TestDataFactory()

# Generate shipments
shipments = factory.generate_shipment(num_records=100)

# Generate users
user = factory.generate_user(role='admin')

# Generate emissions
emissions = factory.generate_emission_record()
```

---

## Fixtures

### Common Fixtures

```python
def test_example(
    db_session,              # Database session
    redis_client,            # Redis client
    mock_anthropic_client,   # Mock LLM client
    sample_shipment_data,    # Test shipment data
    valid_user,              # Test user
    data_factory             # Data generator
):
    # Your test code
    pass
```

---

## Markers

### Use Markers
```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.integration
def test_database():
    pass

@pytest.mark.slow
def test_long_running():
    pass

@pytest.mark.performance
def test_load():
    pass
```

### Run Marked Tests
```bash
# Skip slow tests
pytest -m "not slow"

# Only integration tests
pytest -m integration

# Unit OR integration
pytest -m "unit or integration"

# Unit AND NOT slow
pytest -m "unit and not slow"
```

---

## Environment Variables

```bash
# Test database
export TEST_DATABASE_URL="postgresql://postgres:test@localhost:5432/greenlang_test"

# Test Redis
export TEST_REDIS_URL="redis://localhost:6379/0"

# API keys (for integration tests)
export ANTHROPIC_API_KEY="sk-ant-test-key"
export OPENAI_API_KEY="sk-openai-test-key"

# Test mode
export GL_ENVIRONMENT="test"
```

---

## Troubleshooting

### Tests Fail with Network Error
```bash
# Allow network for integration tests
pytest -m integration
```

### Tests Timeout
```bash
# Increase timeout (default: 300s)
pytest --timeout=600
```

### Database Connection Error
```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d postgres

# Check connection
psql postgresql://postgres:test@localhost:5432/greenlang_test
```

### Redis Connection Error
```bash
# Start test Redis
docker-compose -f docker-compose.test.yml up -d redis

# Check connection
redis-cli -h localhost -p 6379 ping
```

### Coverage Not Generated
```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Run with coverage explicitly
pytest --cov=greenlang_core --cov-report=html
```

---

## Best Practices

### 1. Run Tests Before Commit
```bash
# Quick sanity check
pytest -m unit --maxfail=1

# Full check
pytest
```

### 2. Write Tests First (TDD)
```python
def test_new_feature():
    # Write test first
    result = new_feature()
    assert result == expected
```

### 3. Use Fixtures
```python
@pytest.fixture
def my_fixture():
    # Setup
    resource = create_resource()
    yield resource
    # Teardown
    resource.cleanup()
```

### 4. Parametrize Tests
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected
```

### 5. Mock External Services
```python
from unittest.mock import patch

@patch('greenlang_core.llm.anthropic_client')
def test_with_mock(mock_client):
    mock_client.generate.return_value = "mocked response"
    # Test code
```

---

## Quick Commands Cheat Sheet

```bash
# Run all tests
pytest

# Unit tests only
pytest -m unit

# With coverage
pytest --cov

# Watch mode
ptw

# Parallel
pytest -n auto

# Stop on fail
pytest -x

# Verbose
pytest -vv

# Specific file
pytest tests/unit/core/test_llm_integration.py

# Specific test
pytest tests/unit/core/test_llm_integration.py::test_successful_api_call

# Failed tests only
pytest --lf

# Generate HTML coverage
pytest --cov --cov-report=html && open htmlcov/index.html
```

---

## Resources

- **Full Strategy:** `TESTING_STRATEGY.md`
- **Implementation Summary:** `TESTING_IMPLEMENTATION_SUMMARY.md`
- **Pytest Docs:** https://docs.pytest.org/
- **Coverage Docs:** https://coverage.readthedocs.io/
- **Locust Docs:** https://docs.locust.io/

---

**Last Updated:** 2025-01-14
**Version:** 1.0
