# GreenLang Testing Guide

Quick reference for running and maintaining the comprehensive test suite.

---

## Quick Start

### Run All Tests
```bash
# All tests
pytest

# With verbose output
pytest -v

# With coverage
pytest --cov=. --cov-report=html
```

### Run Specific Test Suites

```bash
# CBAM tests
pytest GL-CBAM-APP/CBAM-Importer-Copilot/tests/ -v

# CSRD tests
pytest GL-CSRD-APP/CSRD-Reporting-Platform/tests/ -v

# VCCI tests
pytest GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/tests/ -v

# Shared services
pytest greenlang/services/tests/ -v

# Benchmarks
pytest tests/benchmarks/ --benchmark-only
```

### Run by Marker

```bash
# V2 refactored tests
pytest -m v2 -v

# Critical path tests
pytest -m critical -v

# Infrastructure tests
pytest -m infrastructure -v

# Performance tests
pytest -m performance -v

# Agent tests
pytest -m agents -v

# Integration tests
pytest -m integration -v
```

---

## Test Requirements

### Install Dependencies

```bash
# Core testing dependencies
pip install pytest pytest-cov pytest-benchmark

# Additional dependencies
pip install pandas numpy scipy
pip install unittest-mock
```

---

## Test Categories

### Unit Tests (60%)
- Fast, isolated tests
- Mock external dependencies
- Test individual functions/methods

### Integration Tests (30%)
- Test component interactions
- May require databases/services
- Validate data flow

### End-to-End Tests (10%)
- Full pipeline execution
- Real or near-real scenarios
- Performance validation

---

## Performance Benchmarks

### Run Benchmarks
```bash
# All benchmarks
pytest tests/benchmarks/benchmark_all.py --benchmark-only

# Specific benchmark
pytest tests/benchmarks/benchmark_all.py::TestAgentExecutionSpeed -v

# Save baseline
pytest tests/benchmarks/ --benchmark-save=baseline

# Compare to baseline
pytest tests/benchmarks/ --benchmark-compare=baseline
```

### Benchmark Targets
- V2 Agent Overhead: <5%
- Cache Hit Rate: 30%+
- LLM Cost Savings: 30%+
- Throughput: 100+ records/sec
- Memory: <500MB

---

## Coverage Reports

### Generate HTML Report
```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html
```

### Generate Terminal Report
```bash
pytest --cov=. --cov-report=term-missing
```

### Coverage Thresholds
- Overall: 90%+
- Critical paths: 95%+
- Infrastructure: 85%+

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt pytest pytest-cov
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Writing New Tests

### Test Structure Template

```python
import pytest

@pytest.mark.unit
class TestMyComponent:
    """Test MyComponent functionality."""

    def test_basic_operation(self):
        """Test basic operation works correctly."""
        component = MyComponent()
        result = component.process(input_data)
        assert result == expected_output

    def test_error_handling(self):
        """Test error handling."""
        component = MyComponent()
        with pytest.raises(ValueError):
            component.process(invalid_data)
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using fixture."""
    assert sample_data["key"] == "value"
```

---

## Test Markers

### Available Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.v2` - V2 refactored tests
- `@pytest.mark.agents` - Agent tests
- `@pytest.mark.services` - Service tests
- `@pytest.mark.infrastructure` - Infrastructure tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.benchmark` - Benchmark tests
- `@pytest.mark.critical` - Critical path tests
- `@pytest.mark.slow` - Slow tests (skip in quick runs)
- `@pytest.mark.llm` - Requires LLM API

### Skip Slow Tests

```bash
pytest -m "not slow"
```

---

## Debugging Tests

### Run Single Test
```bash
pytest path/to/test_file.py::TestClass::test_method -v
```

### Show Print Statements
```bash
pytest -s
```

### Drop to Debugger on Failure
```bash
pytest --pdb
```

### Show Local Variables on Failure
```bash
pytest -l
```

---

## Common Issues

### Import Errors
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Fixtures
- Check conftest.py files
- Ensure fixture scope is correct

### Slow Tests
- Use `@pytest.mark.slow` marker
- Mock expensive operations
- Use smaller test datasets

---

## Test Maintenance

### Best Practices

1. **Keep tests independent** - No shared state
2. **Use descriptive names** - `test_what_when_expected`
3. **One assertion per test** - When possible
4. **Mock external dependencies** - For unit tests
5. **Update tests with code** - Keep in sync

### Regular Tasks

- [ ] Run full test suite weekly
- [ ] Review coverage reports monthly
- [ ] Update benchmarks quarterly
- [ ] Refactor slow tests as needed

---

## Getting Help

### View Test Documentation
```bash
pytest --markers  # List all markers
pytest --fixtures # List all fixtures
pytest --help     # Full help
```

### Resources
- Pytest docs: https://docs.pytest.org/
- Coverage.py: https://coverage.readthedocs.io/
- pytest-benchmark: https://pytest-benchmark.readthedocs.io/

---

**Last Updated:** 2025-11-09
**Testing & QA Team**
