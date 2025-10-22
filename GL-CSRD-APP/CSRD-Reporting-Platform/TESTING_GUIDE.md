# CSRD Calculator Agent - Testing Guide

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
# From project root
pytest tests/test_calculator_agent.py -v

# Or simply
pytest
```

---

## Test Commands Reference

### By Test Type

#### Fast Tests Only (Unit Tests)
```bash
pytest tests/test_calculator_agent.py -v -m "unit"
```
**Best for:** Development, quick validation
**Duration:** ~10 seconds
**Tests:** 70+ tests

#### Critical Tests Only
```bash
pytest tests/test_calculator_agent.py -v -m "critical"
```
**Best for:** Pre-commit checks, zero hallucination verification
**Duration:** ~30 seconds
**Tests:** 20+ critical tests

#### Integration Tests
```bash
pytest tests/test_calculator_agent.py -v -m "integration"
```
**Best for:** End-to-end validation
**Duration:** ~1 minute
**Tests:** 15+ integration tests

#### Exclude Slow Tests (Recommended for Development)
```bash
pytest tests/test_calculator_agent.py -v -m "not slow and not performance"
```
**Best for:** Regular development workflow
**Duration:** ~1 minute
**Tests:** 80+ tests

#### Performance Tests Only
```bash
pytest tests/test_calculator_agent.py -v -m "performance"
```
**Best for:** Performance benchmarking
**Duration:** ~2 minutes
**Tests:** 4 performance tests

---

## Coverage Reports

### Generate HTML Coverage Report
```bash
pytest tests/test_calculator_agent.py --cov=agents.calculator_agent --cov-report=html
```
Then open: `htmlcov/index.html`

### Terminal Coverage Report
```bash
pytest tests/test_calculator_agent.py --cov=agents.calculator_agent --cov-report=term
```

### Coverage with Missing Lines
```bash
pytest tests/test_calculator_agent.py --cov=agents.calculator_agent --cov-report=term-missing
```

---

## Running Specific Tests

### Single Test Class
```bash
pytest tests/test_calculator_agent.py::TestCalculatorAgentInitialization -v
```

### Single Test Method
```bash
pytest tests/test_calculator_agent.py::TestCalculatorAgentInitialization::test_calculator_agent_initialization -v
```

### Multiple Test Classes
```bash
pytest tests/test_calculator_agent.py::TestGHGScope1Calculations tests/test_calculator_agent.py::TestGHGScope2Calculations -v
```

### By Keyword (Pattern Matching)
```bash
# Run all GHG tests
pytest tests/test_calculator_agent.py -k "ghg" -v

# Run all Scope 1 tests
pytest tests/test_calculator_agent.py -k "scope1" -v

# Run all reproducibility tests
pytest tests/test_calculator_agent.py -k "reproducibility" -v

# Run all emission factor tests
pytest tests/test_calculator_agent.py -k "emission_factor" -v
```

---

## Test Output Options

### Verbose Output
```bash
pytest tests/test_calculator_agent.py -v
```

### Very Verbose (Show Each Test)
```bash
pytest tests/test_calculator_agent.py -vv
```

### Quiet Mode (Summary Only)
```bash
pytest tests/test_calculator_agent.py -q
```

### Show Print Statements
```bash
pytest tests/test_calculator_agent.py -v -s
```

### Show Local Variables on Failure
```bash
pytest tests/test_calculator_agent.py -v -l
```

### Full Traceback
```bash
pytest tests/test_calculator_agent.py -v --tb=long
```

### Short Traceback (Default)
```bash
pytest tests/test_calculator_agent.py -v --tb=short
```

### No Traceback
```bash
pytest tests/test_calculator_agent.py -v --tb=no
```

---

## Filtering and Selection

### Stop on First Failure
```bash
pytest tests/test_calculator_agent.py -x
```

### Stop After N Failures
```bash
pytest tests/test_calculator_agent.py --maxfail=3
```

### Run Last Failed Tests
```bash
pytest tests/test_calculator_agent.py --lf
```

### Run Failed Tests First
```bash
pytest tests/test_calculator_agent.py --ff
```

### Run New Tests (Since Last Run)
```bash
pytest tests/test_calculator_agent.py --nf
```

---

## Performance Testing

### Measure Test Duration
```bash
pytest tests/test_calculator_agent.py -v --durations=10
```
Shows 10 slowest tests

### Profile Tests
```bash
pytest tests/test_calculator_agent.py -v --profile
```

---

## Parallel Execution (Faster)

### Install pytest-xdist
```bash
pip install pytest-xdist
```

### Run Tests in Parallel
```bash
# Auto-detect CPU cores
pytest tests/test_calculator_agent.py -n auto

# Use specific number of workers
pytest tests/test_calculator_agent.py -n 4
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Test Calculator Agent

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/test_calculator_agent.py -v --cov=agents.calculator_agent --cov-report=xml
      - uses: codecov/codecov-action@v3
```

### Pre-commit Hook
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
pytest tests/test_calculator_agent.py -v -m "critical and not slow"
if [ $? -ne 0 ]; then
    echo "Critical tests failed. Commit aborted."
    exit 1
fi
```

---

## Test Development Workflow

### 1. **Development** (Fast Feedback)
```bash
pytest tests/test_calculator_agent.py -v -m "unit" -k "your_feature"
```

### 2. **Pre-Commit** (Quick Validation)
```bash
pytest tests/test_calculator_agent.py -v -m "critical and not slow"
```

### 3. **Pre-Push** (Comprehensive)
```bash
pytest tests/test_calculator_agent.py -v -m "not slow"
```

### 4. **Full Suite** (Before Release)
```bash
pytest tests/test_calculator_agent.py -v --cov=agents.calculator_agent --cov-report=html
```

---

## Debugging Tests

### Run Test in Python Debugger (pdb)
```bash
pytest tests/test_calculator_agent.py::TestClass::test_method --pdb
```

### Drop to Debugger on Failure
```bash
pytest tests/test_calculator_agent.py -v --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb
```

### Print Captured Output on Failure
```bash
pytest tests/test_calculator_agent.py -v --capture=no
```

---

## Common Test Scenarios

### Zero Hallucination Verification
```bash
pytest tests/test_calculator_agent.py -v -m "critical" -k "reproducibility or hallucination"
```

### GHG Calculations Only
```bash
pytest tests/test_calculator_agent.py -v -k "scope1 or scope2 or scope3"
```

### Formula Coverage Check
```bash
pytest tests/test_calculator_agent.py::TestAllFormulaCoverage -v
```

### Performance Benchmarking
```bash
pytest tests/test_calculator_agent.py -v -m "performance" --durations=0
```

### Edge Cases and Error Handling
```bash
pytest tests/test_calculator_agent.py::TestEdgeCasesAndBoundaries tests/test_calculator_agent.py::TestErrorHandling -v
```

---

## Interpreting Test Results

### Successful Run
```
====================== test session starts ======================
collected 100 items

tests/test_calculator_agent.py::TestCalculatorAgentInitialization::test_calculator_agent_initialization PASSED [  1%]
tests/test_calculator_agent.py::TestCalculatorAgentInitialization::test_calculator_agent_loads_formulas PASSED [  2%]
...
====================== 100 passed in 45.23s =====================
```
✅ All tests passed!

### Failed Test
```
====================== FAILURES =================================
___________ TestClass::test_method ____________

    def test_method(self):
>       assert result == expected
E       AssertionError: assert 10 == 11

tests/test_calculator_agent.py:123: AssertionError
====================== 1 failed, 99 passed in 45.23s ============
```
❌ Fix the failing test

### Coverage Report
```
Name                           Stmts   Miss  Cover   Missing
------------------------------------------------------------
agents/calculator_agent.py       420      0   100%
------------------------------------------------------------
TOTAL                            420      0   100%
```
✅ 100% coverage achieved!

---

## Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Solution: Install dependencies
pip install -r requirements.txt

# Or install specific packages
pip install pytest pydantic pyyaml pandas
```

### Issue: No tests collected
```bash
# Solution: Check you're in the correct directory
cd GL-CSRD-APP/CSRD-Reporting-Platform

# Verify test files exist
ls tests/

# Run with verbose collection
pytest --collect-only tests/test_calculator_agent.py
```

### Issue: Tests timing out
```bash
# Solution: Increase timeout
pytest tests/test_calculator_agent.py -v --timeout=300
```

### Issue: Import errors
```bash
# Solution: Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/test_calculator_agent.py -v
```

---

## Best Practices

### ✅ DO:
- Run unit tests frequently during development
- Run critical tests before committing
- Run full suite before pushing
- Check coverage regularly (aim for 100%)
- Add tests for new features
- Update tests when refactoring

### ❌ DON'T:
- Skip tests before committing
- Ignore failing tests
- Remove tests to "fix" failures
- Push code with <90% coverage
- Disable critical tests

---

## Test Maintenance

### Adding New Tests
1. Follow existing test patterns
2. Use descriptive test names
3. Add appropriate markers (`@pytest.mark.unit`, etc.)
4. Update this guide if needed

### Updating Tests
1. Run all affected tests
2. Verify coverage doesn't decrease
3. Check performance impact
4. Update documentation

### Removing Tests
1. Verify test is truly obsolete
2. Ensure coverage is maintained
3. Document reason for removal

---

## Performance Targets

| Test Type | Target Duration | Status |
|-----------|----------------|---------|
| Unit Tests | <30 seconds | ✅ |
| Critical Tests | <1 minute | ✅ |
| Integration Tests | <2 minutes | ✅ |
| Full Suite | <5 minutes | ✅ |
| Performance Tests | <3 minutes | ✅ |

| Calculation | Target | Status |
|-------------|--------|---------|
| Single Metric | <5ms | ✅ |
| Batch (7 metrics) | <35ms | ✅ |
| Large Batch (10k) | <50s | ✅ |

---

## Resources

### Documentation
- **Test Summary:** `TEST_CALCULATOR_SUMMARY.md`
- **Test File:** `tests/test_calculator_agent.py`
- **Coverage Report:** `htmlcov/index.html` (after running with --cov)

### pytest Documentation
- Official Docs: https://docs.pytest.org/
- Markers: https://docs.pytest.org/en/latest/example/markers.html
- Fixtures: https://docs.pytest.org/en/latest/fixture.html

### Coverage
- Coverage.py: https://coverage.readthedocs.io/

---

## Quick Reference Card

```bash
# Most Common Commands

# Development (fast)
pytest -v -m "unit"

# Pre-commit (critical)
pytest -v -m "critical and not slow"

# Full suite
pytest -v

# With coverage
pytest -v --cov=agents.calculator_agent --cov-report=html

# Performance check
pytest -v -m "performance" --durations=0

# Debug specific test
pytest -v tests/test_calculator_agent.py::TestClass::test_method --pdb
```

---

**Happy Testing! 🧪**

For questions or issues, refer to:
- `TEST_CALCULATOR_SUMMARY.md` - Comprehensive test documentation
- `tests/test_calculator_agent.py` - Source code with inline documentation
