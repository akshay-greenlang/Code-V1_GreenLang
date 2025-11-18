# GL-005 Test Suite - Quick Start Guide

## Prerequisites

```bash
# Install dependencies
pip install pytest pytest-asyncio pytest-cov pytest-timeout pydantic
```

## Quick Start (30 seconds)

```bash
# Navigate to GL-005 directory
cd C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/agents/GL-005

# Run all tests
pytest tests/ -v

# Expected output: 89 tests passed
```

## Run Specific Test Categories

### Unit Tests (Fast - ~10 seconds)
```bash
pytest tests/unit/ -v
# 72 tests covering orchestrator, calculators, tools, config
```

### Integration Tests (Slower - ~60 seconds)
```bash
pytest tests/integration/ -v
# 17 tests covering E2E workflows, safety, determinism
```

### Specific Test Files

**Orchestrator Tests** (17 tests):
```bash
pytest tests/unit/test_orchestrator.py -v
```

**Calculator Tests** (27 tests):
```bash
pytest tests/unit/test_calculators.py -v
```

**Safety Tests** (14 tests):
```bash
pytest tests/integration/test_safety_interlocks.py -v
```

**Determinism Tests** (11 tests):
```bash
pytest tests/integration/test_determinism_validation.py -v
```

**E2E Workflow Tests** (12 tests):
```bash
pytest tests/integration/test_e2e_control_workflow.py -v
```

## Run by Test Marker

### Performance Tests
```bash
pytest -m performance -v
# Tests control loop latency, throughput
```

### Determinism Tests
```bash
pytest -m determinism -v
# Tests zero-hallucination guarantee
```

### Safety Tests
```bash
pytest -m safety -v
# Tests safety interlocks
```

### Boundary Tests
```bash
pytest -m boundary -v
# Tests edge cases and boundary conditions
```

## Coverage Reports

### Generate HTML Coverage Report
```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

### Terminal Coverage Report
```bash
pytest tests/ --cov=. --cov-report=term-missing
# Shows coverage with missing line numbers
```

### Check Coverage Threshold
```bash
pytest tests/ --cov=. --cov-fail-under=85
# Fails if coverage below 85%
```

## Performance Benchmarking

### Show Slowest Tests
```bash
pytest tests/ --durations=10
# Shows 10 slowest tests
```

### Performance Tests Only
```bash
pytest -m performance -v
# Validates <100ms control loop latency
```

## Test Selection

### Run Single Test
```bash
pytest tests/unit/test_calculators.py::TestStabilityIndexCalculator::test_stability_index_high_stability -v
```

### Run Test Class
```bash
pytest tests/unit/test_orchestrator.py::TestCombustionControlOrchestratorInitialization -v
```

### Exclude Slow Tests
```bash
pytest -m "not slow" -v
# Skips tests marked as slow
```

## Debugging Failed Tests

### Verbose Output
```bash
pytest tests/ -vv
# Extra verbose output
```

### Show Print Statements
```bash
pytest tests/ -s
# Show print() output
```

### Stop on First Failure
```bash
pytest tests/ -x
# Stop at first failure
```

### Run Last Failed Tests
```bash
pytest tests/ --lf
# Re-run only failed tests
```

## Continuous Integration

### Run Full CI Pipeline
```bash
pytest tests/ -v --cov=. --cov-report=xml --cov-report=term-missing --cov-fail-under=85
# Full CI test suite with coverage
```

### Quick Smoke Test
```bash
pytest tests/unit/ -v --maxfail=1
# Quick validation, stop on first failure
```

## Test Statistics

```bash
# Count total tests
pytest --collect-only tests/ | grep "test session starts"

# Current: 89 tests
```

## Common Issues

### Issue: Tests timeout
**Solution**: Increase timeout in pytest.ini
```ini
timeout = 600
```

### Issue: Coverage below threshold
**Solution**: Generate HTML report to see missing coverage
```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html
```

### Issue: Mock servers port conflict
**Solution**: Change ports in tests/.env.example
```bash
TEST_MOCK_DCS_PORT=4841
TEST_MOCK_PLC_PORT=5503
```

## Expected Results

### All Tests Pass
```
============ 89 passed in XX.XXs ============
```

### Coverage Report
```
TOTAL                    3800    320    92%
```

### Performance Benchmarks
- Control loop latency: <100ms ✓
- Safety check time: <20ms ✓
- Emergency shutdown: <1000ms ✓

## Test Suite Structure

```
tests/
├── unit/                  # 72 tests
│   ├── test_orchestrator.py     # 17 tests
│   ├── test_calculators.py      # 27 tests
│   ├── test_tools.py            # 14 tests
│   └── test_config.py           # 14 tests
└── integration/           # 17 tests
    ├── test_e2e_control_workflow.py      # 12 tests
    ├── test_safety_interlocks.py         # 14 tests
    └── test_determinism_validation.py    # 11 tests
```

## Next Steps

1. **Run tests**: `pytest tests/ -v`
2. **Check coverage**: `pytest tests/ --cov=. --cov-report=html`
3. **Review results**: Open `htmlcov/index.html`
4. **Fix failures**: Use `-vv` and `-s` flags for debugging

## Support

- See `tests/README.md` for detailed documentation
- See `TEST_SUITE_SUMMARY.md` for implementation details
- Check test logs: `tests/logs/gl005_tests.log`

---

**Quick Reference**:
- All tests: `pytest tests/ -v`
- Unit only: `pytest tests/unit/ -v`
- Integration only: `pytest tests/integration/ -v`
- With coverage: `pytest tests/ --cov=. --cov-report=html`
- Performance: `pytest -m performance -v`
- Determinism: `pytest -m determinism -v`
