# GreenLang Testing Framework - Quick Start Guide

## Installation

```bash
# Navigate to testing directory
cd C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\testing

# Install required packages
pip install pytest pytest-cov pytest-asyncio pytest-benchmark
pip install numpy pandas psutil faker
```

## Running Tests

### Run All Tests
```bash
# Verbose mode with coverage
pytest -v --cov=. --cov-report=html --cov-report=term

# Quick run (no coverage)
pytest -v
```

### Run by Test Suite
```bash
# Unit tests only
pytest unit_tests/ -v

# Integration tests only
pytest integration_tests/ -v

# Performance tests only
pytest performance_tests/ -v

# Security tests only
pytest security_tests/ -v
```

### Run by Marker
```bash
# Run unit tests
pytest -m unit -v

# Run integration tests
pytest -m integration -v

# Run performance tests
pytest -m performance -v

# Run security tests
pytest -m security -v

# Run slow tests
pytest -m slow -v
```

### Run Specific Test File
```bash
# Memory systems tests
pytest unit_tests/test_memory_systems.py -v

# Capabilities tests
pytest unit_tests/test_capabilities.py -v

# Intelligence tests
pytest unit_tests/test_intelligence.py -v

# RAG system tests
pytest integration_tests/test_rag_system.py -v

# Multi-agent tests
pytest integration_tests/test_multi_agent_workflows.py -v

# Load/stress tests
pytest performance_tests/test_load_stress.py -v

# Security tests
pytest security_tests/test_security_vulnerabilities.py -v
```

### Run Specific Test
```bash
# Run single test class
pytest unit_tests/test_memory_systems.py::TestShortTermMemory -v

# Run single test method
pytest unit_tests/test_memory_systems.py::TestShortTermMemory::test_add_to_working_memory -v
```

## Viewing Results

### Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=. --cov-report=html

# Open in browser (Windows)
start htmlcov/index.html

# Open in browser (macOS)
open htmlcov/index.html

# Open in browser (Linux)
xdg-open htmlcov/index.html
```

### Terminal Coverage
```bash
# Show coverage in terminal
pytest --cov=. --cov-report=term
```

## Test Organization

```
testing/
├── agent_test_framework.py       # Core testing infrastructure
├── quality_validators.py          # 12-dimension quality framework
├── conftest.py                    # Pytest configuration
│
├── unit_tests/
│   ├── test_memory_systems.py     # Memory: STM, LTM, Episodic, Semantic
│   ├── test_capabilities.py       # Planning, Reasoning, Meta-cognition, Tools
│   └── test_intelligence.py       # LLM orchestration, Prompts, Context
│
├── integration_tests/
│   ├── test_rag_system.py         # RAG pipeline integration
│   └── test_multi_agent_workflows.py  # Multi-agent coordination
│
├── performance_tests/
│   └── test_load_stress.py        # Load testing, stress testing
│
└── security_tests/
    └── test_security_vulnerabilities.py  # Security & vulnerability tests
```

## Key Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.asyncio` - Async tests

## Example Test Execution

```bash
# Run memory tests with coverage
pytest unit_tests/test_memory_systems.py -v --cov=. --cov-report=term

# Run performance tests (includes benchmarks)
pytest performance_tests/test_load_stress.py -v -m performance

# Run all tests excluding slow tests
pytest -v -m "not slow"

# Run only async tests
pytest -v -m asyncio

# Run with detailed output
pytest -vv --tb=short
```

## Common Options

```bash
# Verbose output
-v, --verbose

# Very verbose output
-vv

# Show print statements
-s

# Stop on first failure
-x, --exitfirst

# Run last failed tests
--lf, --last-failed

# Show coverage
--cov=.

# Coverage report format
--cov-report=html
--cov-report=term
--cov-report=xml

# Run in parallel (requires pytest-xdist)
-n 4  # Use 4 CPU cores

# Show slowest tests
--durations=10
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install Dependencies
        run: |
          pip install pytest pytest-cov pytest-asyncio
          pip install numpy pandas psutil faker

      - name: Run Tests
        run: pytest -v --cov=. --cov-report=xml

      - name: Upload Coverage
        uses: codecov/codecov-action@v2
```

## Coverage Targets

- **Overall Coverage:** 90%+
- **Unit Tests:** 95%+
- **Integration Tests:** 85%+
- **Critical Paths:** 100%

## Performance Targets

- **Agent Creation:** <100ms P99
- **Message Passing:** <10ms P99
- **STM Retrieval:** <50ms P99
- **LTM Hot Tier:** <50ms P99
- **LTM Cold Tier:** <200ms P99
- **Throughput:** >1000 agents/second
- **Concurrent Agents:** 10,000+

## Troubleshooting

### Import Errors
```bash
# Add parent directory to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/agent_foundation"
```

### Async Test Warnings
```bash
# Install pytest-asyncio
pip install pytest-asyncio
```

### Coverage Not Showing
```bash
# Install pytest-cov
pip install pytest-cov

# Ensure running from correct directory
cd testing/
```

### Slow Tests
```bash
# Skip slow tests
pytest -v -m "not slow"

# Only run fast tests
pytest -v -m "not performance and not slow"
```

## Best Practices

1. **Always run tests before committing**
   ```bash
   pytest -v --cov=. --cov-report=term
   ```

2. **Write tests for new features**
   - Unit tests for individual components
   - Integration tests for workflows
   - Performance tests for critical paths

3. **Maintain >90% coverage**
   ```bash
   pytest --cov=. --cov-report=term --cov-fail-under=90
   ```

4. **Use markers for organization**
   ```python
   @pytest.mark.unit
   @pytest.mark.integration
   @pytest.mark.performance
   ```

5. **Document test purpose**
   ```python
   def test_feature():
       """Test that feature works correctly."""
       # Arrange, Act, Assert
   ```

## Quick Commands Cheat Sheet

```bash
# Full test suite with coverage
pytest -v --cov=. --cov-report=html

# Fast unit tests only
pytest unit_tests/ -v

# Performance benchmarks
pytest performance_tests/ -v -m performance

# Security audit
pytest security_tests/ -v -m security

# Parallel execution
pytest -n 4 -v

# Stop on first failure
pytest -x -v

# Re-run failures
pytest --lf -v

# Show slowest 10 tests
pytest --durations=10
```

---

**Last Updated:** 2025-11-15
**Framework Version:** 2.0
**Status:** Production Ready ✓
