# INTL-103 CI Configuration

## Overview

This document describes the CI/CD configuration for INTL-103 ("No Naked Numbers" enforcement) as part of the GreenLang Intelligence Runtime.

**Related Specification**: INTL-103 - Tool Runtime with "No Naked Numbers" Enforcement
**DoD Gap 7**: Document CI job configuration (15 min)

## Test Suite

The INTL-103 test suite validates that:
1. All numeric values in tool outputs are wrapped in `Quantity {value, unit}`
2. Currency units are non-convertible (tagged dimensions)
3. Version strings are only allowed in code blocks
4. Property/fuzz tests validate digit scanner robustness
5. Golden tests ensure byte-exact reproducibility
6. Performance benchmarks meet p95 < 200ms latency requirement
7. Unit conversion system achieves ≥ 80% code coverage

### Test Files

| Test File | Purpose | Coverage Target |
|-----------|---------|-----------------|
| `tests/intelligence/test_tools_runtime.py` | Core runtime functionality | Tool runtime, schemas, errors |
| `tests/intelligence/test_golden_replay.py` | Deterministic replay validation | Runtime reproducibility |
| `tests/intelligence/test_performance_benchmark.py` | Performance requirements | p95 < 200ms, no memory leaks |
| `tests/intelligence/test_units_coverage.py` | Unit system edge cases | units.py ≥ 80% coverage |

## CI Job Configuration

### GitHub Actions Workflow

**File**: `.github/workflows/intl-103-runtime.yml`

```yaml
name: INTL-103 Tool Runtime Tests

on:
  push:
    branches: [main, develop]
    paths:
      - 'greenlang/intelligence/runtime/**'
      - 'tests/intelligence/**'
  pull_request:
    branches: [main, develop]
    paths:
      - 'greenlang/intelligence/runtime/**'
      - 'tests/intelligence/**'

jobs:
  runtime-tests:
    name: Tool Runtime Tests
    runs-on: ${{ matrix.os }}

    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ['3.11', '3.12', '3.13']

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,test]
          pip install pytest pytest-cov hypothesis pydantic jsonschema jsonpath-ng pint

      - name: Run INTL-103 runtime tests
        run: |
          pytest tests/intelligence/test_tools_runtime.py \
                 tests/intelligence/test_golden_replay.py \
                 tests/intelligence/test_performance_benchmark.py \
                 tests/intelligence/test_units_coverage.py \
                 --cov=greenlang.intelligence.runtime \
                 --cov-report=xml \
                 --cov-report=term \
                 --cov-fail-under=80 \
                 -v

      - name: Upload coverage reports
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: intl-103-runtime
          name: intl-103-${{ matrix.os }}-${{ matrix.python-version }}

      - name: Archive golden files
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: golden-files-${{ matrix.os }}-${{ matrix.python-version }}
          path: tests/goldens/
          retention-days: 7

  performance-gate:
    name: Performance Gate
    runs-on: ubuntu-latest
    needs: runtime-tests

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,test]
          pip install pytest hypothesis

      - name: Run performance benchmarks
        run: |
          pytest tests/intelligence/test_performance_benchmark.py \
                 -v -s \
                 --tb=short

      - name: Check performance requirements
        run: |
          echo "✓ p95 latency < 200ms requirement enforced by test assertions"
          echo "✓ p99 latency < 500ms requirement enforced by test assertions"
          echo "✓ No memory leak requirement enforced by test assertions"

  coverage-gate:
    name: Coverage Gate
    runs-on: ubuntu-latest
    needs: runtime-tests

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev,test]
          pip install pytest pytest-cov

      - name: Check units.py coverage ≥ 80%
        run: |
          pytest tests/intelligence/test_tools_runtime.py \
                 tests/intelligence/test_units_coverage.py \
                 --cov=greenlang.intelligence.runtime.units \
                 --cov-fail-under=80 \
                 --cov-report=term-missing

      - name: Verify coverage threshold met
        run: |
          echo "✓ units.py coverage ≥ 80% requirement enforced"
```

## Local Development

### Running Tests Locally

```bash
# Run all INTL-103 tests
pytest tests/intelligence/test_tools_runtime.py \
       tests/intelligence/test_golden_replay.py \
       tests/intelligence/test_performance_benchmark.py \
       tests/intelligence/test_units_coverage.py \
       -v

# Run with coverage report
pytest tests/intelligence/test_tools_runtime.py \
       tests/intelligence/test_units_coverage.py \
       --cov=greenlang.intelligence.runtime \
       --cov-report=html \
       --cov-report=term-missing

# Run performance benchmarks only
pytest tests/intelligence/test_performance_benchmark.py -v -s

# Run property/fuzz tests only
pytest tests/intelligence/test_tools_runtime.py::TestPropertyFuzz -v
```

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: intl-103-runtime-tests
      name: INTL-103 Runtime Tests
      entry: pytest tests/intelligence/test_tools_runtime.py -q
      language: system
      pass_filenames: false
      files: ^greenlang/intelligence/runtime/
```

## Exit Criteria

The CI job will fail if any of the following are not met:

1. **All tests pass**: 100% pass rate required
2. **Coverage threshold**: `greenlang.intelligence.runtime.units` must have ≥ 80% coverage
3. **Performance gate**: p95 latency < 200ms for tool execution
4. **Golden file match**: Byte-exact reproducibility verified
5. **No memory leaks**: Object count growth < 1000 over 100 iterations

## Monitoring & Alerts

### Coverage Tracking

- **Tool**: Codecov
- **Threshold**: 80% minimum for units.py
- **Alert**: Fail PR if coverage drops below threshold

### Performance Tracking

- **Metrics collected**:
  - p50, p95, p99 latency
  - Mean validation overhead
  - Throughput (informational)
  - Object count growth

- **Alerts**:
  - Fail if p95 > 200ms
  - Fail if p99 > 500ms
  - Fail if object growth > 1000

### Golden File Drift

- **Check**: SHA256 hash comparison
- **Alert**: Fail if golden output changes without explicit update
- **Update process**: Delete golden file, re-run test, commit new golden

## Troubleshooting

### Common CI Failures

#### 1. Coverage below 80%

**Symptom**: `ERROR: Coverage failure: total of X.XX is less than fail-under=80.00`

**Fix**: Add tests to cover missing branches in units.py:
```python
# Example: Cover unknown dimension branch
def test_validate_dimension_unknown(ureg):
    qty = Quantity(value=100, unit="kg")
    with pytest.raises(GLValidationError):
        ureg.validate_dimension(qty, "unknown_dimension")
```

#### 2. Performance regression

**Symptom**: `AssertionError: p95 latency (XXX.XXms) exceeds 200ms threshold`

**Fix**:
1. Profile tool execution: `python -m cProfile -s cumulative ...`
2. Check for added I/O or network calls
3. Verify schema validation overhead is minimal
4. Review git diff for performance-impacting changes

#### 3. Golden file mismatch

**Symptom**: `AssertionError: Output does not match golden reference!`

**Fix**:
1. Investigate why output changed
2. If intentional: `rm tests/goldens/runtime_no_naked_numbers.json`
3. Re-run test to regenerate golden
4. Commit updated golden file with explanation

#### 4. Property test failures

**Symptom**: Hypothesis finds counter-example

**Fix**:
1. Review counter-example input
2. Determine if it's a valid edge case or test issue
3. Either fix code to handle edge case or update test assumptions
4. Add regression test for the specific input

## Maintenance

### Updating Golden Files

Golden files should be updated when:
- Tool runtime output format changes intentionally
- New provenance fields are added
- Claim resolution logic is modified

**Process**:
```bash
# Delete existing golden
rm tests/goldens/runtime_no_naked_numbers.json
rm tests/goldens/runtime_no_naked_numbers.sha256

# Regenerate (will skip on first run)
pytest tests/intelligence/test_golden_replay.py::TestGoldenReplay::test_golden_no_naked_numbers_scenario

# Re-run to verify
pytest tests/intelligence/test_golden_replay.py::TestGoldenReplay::test_golden_no_naked_numbers_scenario

# Commit updated golden
git add tests/goldens/
git commit -m "chore: Update INTL-103 golden files for [reason]"
```

### Performance Baseline Updates

If performance characteristics change due to architecture improvements:

1. Update threshold in `test_performance_benchmark.py`:
   ```python
   assert p95 < NEW_THRESHOLD_MS
   ```
2. Document reason in commit message
3. Update this CI documentation to reflect new baseline

## References

- **Specification**: `docs/intelligence/INTL-103-Tool-Runtime.md`
- **DoD Status**: `docs/intelligence/INTL-103-DoD-Status-Report.md`
- **Test Coverage Report**: Generated in CI artifacts
- **Performance Benchmarks**: `tests/intelligence/test_performance_benchmark.py`

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2025-10-02 | Claude | Initial CI configuration documentation for INTL-103 |
