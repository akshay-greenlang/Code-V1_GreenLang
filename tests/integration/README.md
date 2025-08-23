# GreenLang Integration Tests

Comprehensive integration test suite for GreenLang workflows, ensuring bulletproof reliability across all scenarios.

## Test Structure

```
tests/integration/
├── conftest.py                           # Test configuration and fixtures
├── test_workflow_commercial_e2e.py       # Commercial building E2E tests
├── test_workflow_india_e2e.py           # India-specific workflow tests
├── test_workflow_portfolio_e2e.py       # Portfolio aggregation tests
├── test_workflow_cross_country_e2e.py   # Cross-country comparison tests
├── test_workflow_cli_e2e.py             # CLI command tests
├── test_workflow_parallel_and_caching.py # Parallel execution & caching
├── test_workflow_errors_and_validation.py # Error handling tests
├── test_workflow_provenance_and_versions.py # Provenance tracking
├── test_workflow_reproducibility_and_perf.py # Reproducibility & performance
├── test_workflow_plugins_and_contracts.py # Plugin discovery & contracts
├── test_workflow_assistant_mocked.py     # Assistant with mocked LLM
├── test_backward_compatibility.py        # Backward compatibility tests
└── utils/                                # Test utilities
    ├── normalizers.py                    # Data normalization helpers
    ├── io.py                             # I/O helpers
    └── net_guard.py                      # Network blocking utilities
```

## Running Tests

### Run all integration tests
```bash
pytest -m integration -q
```

### Run specific test category
```bash
# Commercial workflow tests
pytest tests/integration/test_workflow_commercial_e2e.py -v

# Performance tests only
pytest -m performance -v

# With coverage
pytest -m integration --cov=greenlang --cov-report=html
```

### Run tests in parallel (requires pytest-xdist)
```bash
pytest -m integration -n auto
```

## Test Categories

### 1. End-to-End Scenarios
- **Commercial Building E2E**: Complete workflow for office buildings
- **India-Specific**: BEE/EECB compliance, India recommendations
- **Portfolio Analysis**: Multi-building aggregation
- **Cross-Country**: Different emission factors by country
- **CLI E2E**: Command-line interface testing

### 2. Engine Behavior
- **Parallel Execution**: Concurrent fuel calculations
- **Caching**: Result caching and invalidation
- **Error Handling**: Graceful failure modes
- **Validation**: Input validation and type checking

### 3. Data Integrity
- **Provenance**: Dataset version tracking
- **Reproducibility**: Deterministic results
- **Performance**: < 2s for single building
- **Numerical Invariants**: sum(by_fuel) ≈ total

### 4. Extensibility
- **Plugin Discovery**: Dynamic agent loading
- **Contract Testing**: Agent interface compliance
- **Assistant Integration**: LLM-driven workflows
- **Backward Compatibility**: Version migration

## Key Assertions

### Success Path
- `result.success is True`
- `emissions.total_co2e_kg > 0`
- `'by_fuel' in emissions`
- `benchmark.rating in ['Excellent', 'Good', 'Average', 'Poor']`
- `len(recommendations) >= 3`

### Numerical Invariants
- `sum(by_fuel.values()) ≈ total_co2e_kg` (ε ≤ 1e-9)
- `total_co2e_tons == total_co2e_kg / 1000` (ε ≤ 1e-6)
- All emissions ≥ 0
- Percentages sum to ~100%

### Dataset-Driven
- Factors from `global_emission_factors.json`
- No hardcoded emission values
- Provenance includes version/source/date

### Performance Gates
- Single building < 2s
- Portfolio (50 buildings) < 5s
- Memory usage < 100MB

## Fixtures

### Core Fixtures
- `dataset()`: Emission factors and benchmarks
- `workflow_runner()`: In-process workflow execution
- `runner()`: Click CLI runner
- `mock_llm()`: Deterministic LLM responses

### Utilities
- `assert_close()`: Numerical comparison with tolerance
- `normalize_text()`: Strip timestamps/paths
- `block_network()`: Enforce offline testing
- `tmp_outdir()`: Temporary output directory

## Golden Snapshots

Located in `tests/integration/snapshots/`:
- `reports/commercial_india_office.json`: Reference output
- `cli/calc_building.md`: CLI markdown report

Update snapshots:
```python
compare_snapshots(actual, snapshot_path, update=True)
```

## Network Isolation

All tests run offline with blocked network access:
- Socket connections blocked
- HTTP requests mocked
- Deterministic data from fixtures

## Continuous Integration

### GitHub Actions
```yaml
- name: Run Integration Tests
  run: |
    pytest -m integration -q --disable-warnings
    pytest -m performance --timeout=10
```

### Pre-commit Hook
```bash
#!/bin/bash
pytest -m integration -q --tb=line
```

## Adding New Tests

1. Create test file: `test_workflow_<feature>_e2e.py`
2. Mark with `@pytest.mark.integration`
3. Use fixtures for data/workflows
4. Assert numerical invariants
5. Add golden snapshot if needed
6. Document in this README

## Troubleshooting

### Test Failures
- Check `pytest -vv` for detailed output
- Verify fixtures in `tests/fixtures/`
- Check dataset version compatibility
- Review snapshots for changes

### Performance Issues
- Use `@pytest.mark.timeout(5)` for limits
- Profile with `pytest --profile`
- Check parallel execution conflicts
- Review caching behavior

### Flaky Tests
- Ensure deterministic seeds
- Check for race conditions
- Verify network is blocked
- Review timestamp normalization

## Coverage Goals

- Line coverage: > 80%
- Branch coverage: > 70%
- All workflows tested
- All error paths covered
- Performance budgets enforced