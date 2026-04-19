# GreenLang Emission Factors - Comprehensive Test Suite

## Overview

This test suite provides **90%+ test coverage** for the GreenLang emission factors infrastructure, validating 500+ emission factors across database, API, calculation engine, and integration layers.

**Coverage Target: 90%+**

## Test Suite Organization

```
tests/
├── test_emission_factors.py          # Original database tests (85% coverage)
├── test_database_comprehensive.py    # Expanded database tests (500 factors, concurrency, fallback)
├── test_api.py                        # Original API tests (87% coverage)
├── test_api_performance.py            # API load tests (1200 req/sec target, cache validation)
├── test_calculation_engine.py         # Calculation tests (determinism, audit, multi-gas)
├── test_integration_e2e.py            # End-to-end workflows (CSRD, CBAM, VCCI)
├── test_performance_benchmarks.py     # Performance tests (10,000+ calculations, benchmarking)
├── conftest_emission_factors.py       # Shared fixtures and test data generators
├── test_data_generator.py             # Test data generation utilities
└── README_TEST_SUITE.md               # This file
```

## Quick Start

### Install Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-benchmark pytest-asyncio psutil

# Or install from requirements
pip install -r tests/requirements_test.txt
```

### Run All Tests

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=greenlang --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/ -v -m unit              # Unit tests only
pytest tests/ -v -m integration       # Integration tests only
pytest tests/ -v -m performance       # Performance tests only
```

### Run Individual Test Files

```bash
# Database tests
pytest tests/test_database_comprehensive.py -v

# API tests
pytest tests/test_api_performance.py -v

# Calculation engine tests
pytest tests/test_calculation_engine.py -v

# Integration tests
pytest tests/test_integration_e2e.py -v

# Performance benchmarks
pytest tests/test_performance_benchmarks.py -v --benchmark-only
```

## Test Categories

### 1. Database Tests (test_database_comprehensive.py)

**Coverage Target: 90%+**

- **500 Factor Import**: Validates import of 500+ emission factors
- **Query Performance**: <10ms target for single factor lookup
- **Geographic Fallback**: Tests country → regional → global fallback logic
- **Temporal Fallback**: Tests year-specific → latest available fallback
- **Concurrent Access**: Multi-threaded read/write operations
- **Database Integrity**: Constraints, foreign keys, indexes

**Key Tests:**
- `test_import_500_factors()` - Import 500+ factors successfully
- `test_single_factor_lookup_performance()` - <10ms lookup target
- `test_concurrent_reads()` - Multi-threaded concurrent reads
- `test_geographic_fallback()` - Geographic fallback logic
- `test_database_constraints()` - Integrity constraints

**Run:**
```bash
pytest tests/test_database_comprehensive.py -v
```

### 2. API Performance Tests (test_api_performance.py)

**Coverage Target: 87%+ → 90%+**

- **Load Testing**: 1200 req/sec target throughput
- **Cache Hit Rate**: >90% cache hit rate validation
- **Response Times**: P50 < 30ms, P95 < 50ms, P99 < 100ms
- **Error Handling**: Comprehensive error scenarios
- **Rate Limiting**: Rate limit enforcement (if enabled)
- **Pagination**: Efficient pagination handling

**Key Tests:**
- `test_sustained_load_calculations()` - 100 req/sec sustained load
- `test_cache_hit_rate_for_factors()` - >90% cache hit rate
- `test_calculation_response_time()` - P95 < 50ms
- `test_batch_calculation_throughput()` - 500+ calc/sec
- `test_concurrent_different_endpoints()` - Mixed endpoint load

**Run:**
```bash
pytest tests/test_api_performance.py -v
```

### 3. Calculation Engine Tests (test_calculation_engine.py)

**Coverage Target: 94%+**

- **Determinism**: Bit-perfect reproducibility (same input → same output)
- **Audit Trail**: Complete audit trail for every calculation
- **Multi-Gas Decomposition**: CO2, CH4, N2O breakdown accuracy
- **Uncertainty Quantification**: Monte Carlo simulation
- **Provenance Hashing**: SHA-256 provenance hashing
- **Edge Cases**: Zero, negative, very large values

**Key Tests:**
- `test_same_input_same_output()` - Determinism guarantee
- `test_audit_trail_creation()` - Audit trail completeness
- `test_gas_decomposition_calculation()` - Multi-gas accuracy
- `test_monte_carlo_simulation()` - Uncertainty quantification
- `test_provenance_hash_determinism()` - Provenance hashing

**Run:**
```bash
pytest tests/test_calculation_engine.py -v
```

### 4. Integration & E2E Tests (test_integration_e2e.py)

**Coverage Target: Full workflows**

- **End-to-End Workflows**: Query → Calculate → Audit Trail
- **CSRD Reporting**: Scope 1+2 reporting workflow
- **CBAM Imports**: Carbon border adjustment calculations
- **VCCI Scope 3**: Value chain calculations
- **YAML Import**: Import pipeline validation
- **Batch Processing**: Multi-record processing

**Key Tests:**
- `test_query_calculate_audit_workflow()` - Complete workflow
- `test_csrd_reporting_workflow()` - CSRD Scope 1+2
- `test_cbam_import_calculation()` - CBAM embedded emissions
- `test_yaml_import_factors()` - YAML import pipeline
- `test_batch_calculation_workflow()` - Fleet calculations

**Run:**
```bash
pytest tests/test_integration_e2e.py -v
```

### 5. Performance Benchmarks (test_performance_benchmarks.py)

**Coverage Target: Performance validation**

- **Batch Processing**: 10,000+ calculations
- **Throughput**: >100 calc/sec (single-threaded), >500 calc/sec (parallel)
- **Memory Usage**: <500 MB for 10k calculations
- **Scalability**: Linear scaling validation
- **Database Size Impact**: Performance vs. DB size

**Key Tests:**
- `test_batch_10000_calculations()` - 10k calculation batch
- `test_batch_parallel_processing()` - Multi-threaded processing
- `test_memory_usage_large_batch()` - Memory profiling
- `test_linear_scaling_calculations()` - Scalability validation
- `test_database_size_impact()` - DB size vs. performance

**Run:**
```bash
pytest tests/test_performance_benchmarks.py -v --benchmark-only
```

## Test Markers

Tests are organized using pytest markers:

```python
# Run specific categories
pytest -m unit              # Unit tests (fast, isolated)
pytest -m integration       # Integration tests (database, API)
pytest -m performance       # Performance tests (benchmarking)
pytest -m compliance        # Compliance tests (regulatory)
pytest -m e2e               # End-to-end tests (full workflows)

# Application-specific
pytest -m csrd              # CSRD reporting tests
pytest -m cbam              # CBAM import tests
pytest -m vcci              # VCCI Scope 3 tests
```

## Test Data Generation

### Using the Test Data Generator

```python
from tests.test_data_generator import EmissionFactorGenerator

# Generate 500 realistic factors
generator = EmissionFactorGenerator(seed=42)
factors = generator.generate_realistic_factors(count=500)

# Generate gas vectors
gas_vectors = []
for factor in factors:
    vectors = generator.generate_gas_vectors(factor)
    gas_vectors.extend(vectors)

# Generate calculation scenarios
scenarios = generator.generate_calculation_scenarios(count=100)

# Save to JSON
from tests.test_data_generator import save_test_data_to_json
save_test_data_to_json('test_data.json', count=500)
```

### Fixtures

```python
# Using shared fixtures
def test_example(populated_db, emission_factor_client, performance_timer):
    """Example test using shared fixtures."""

    # Use performance timer
    performance_timer.start()

    # Use client
    factor = emission_factor_client.get_factor('diesel_us_2024')

    # Measure performance
    performance_timer.stop()

    assert factor.factor_id == 'diesel_us_2024'
    assert performance_timer.elapsed_ms() < 10  # <10ms target
```

## Coverage Reports

### Generate HTML Coverage Report

```bash
# Run tests with coverage
pytest tests/ --cov=greenlang --cov-report=html

# Open report
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
xdg-open htmlcov/index.html # Linux
```

### Coverage Breakdown

```bash
# Show coverage by module
pytest tests/ --cov=greenlang --cov-report=term-missing

# Coverage targets
# - Database layer: 90%+
# - API layer: 90%+
# - Calculation engine: 94%+
# - Integration: Full workflows
# - Overall: 90%+
```

## Performance Targets

### Database Performance
- Single factor lookup: **<10ms**
- Category query: **<50ms**
- Search query: **<100ms**
- 500 factor import: **<5 seconds**

### API Performance
- Health check: P95 **<100ms**
- Factor lookup: P95 **<30ms**
- Calculation: P50 **<30ms**, P95 **<50ms**, P99 **<100ms**
- List factors: P95 **<200ms**

### Calculation Performance
- Single calculation: **<10ms**
- 1,000 calculations: **<10 seconds** (>100 calc/sec)
- 10,000 calculations: **<100 seconds** (>100 calc/sec)
- Parallel processing: **>500 calc/sec**

### Cache Performance
- Cache hit rate: **>90%**
- Cached request speedup: **2-10x** faster

## CI/CD Integration

### GitHub Actions

```yaml
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
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/requirements_test.txt

    - name: Run tests with coverage
      run: |
        pytest tests/ -v --cov=greenlang --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

    - name: Check coverage threshold
      run: |
        pytest tests/ --cov=greenlang --cov-fail-under=90
```

### GitLab CI

```yaml
test:
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - pip install -r tests/requirements_test.txt
    - pytest tests/ -v --cov=greenlang --cov-report=term --cov-fail-under=90
  coverage: '/TOTAL.*\s+(\d+%)$/'
```

## Troubleshooting

### Common Issues

**Issue: Tests fail with database connection errors**
```bash
# Solution: Ensure database path is writable
export TMPDIR=/tmp
pytest tests/test_database_comprehensive.py -v
```

**Issue: Performance tests fail intermittently**
```bash
# Solution: Run with more iterations or higher timeout
pytest tests/test_performance_benchmarks.py -v --benchmark-min-rounds=10
```

**Issue: Coverage below 90%**
```bash
# Solution: Check which modules need more tests
pytest tests/ --cov=greenlang --cov-report=term-missing

# Focus on modules with low coverage
pytest tests/test_<module>.py -v --cov=greenlang.<module>
```

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_<feature>_<scenario>.py`
2. **Add docstrings**: Explain what the test validates
3. **Use markers**: Tag with appropriate markers (`@pytest.mark.unit`, etc.)
4. **Add fixtures**: Create reusable fixtures in `conftest_emission_factors.py`
5. **Update this README**: Document new test categories

## Test Suite Statistics

- **Total Test Files**: 8
- **Total Tests**: 100+ (and growing)
- **Test Categories**: 5 (Database, API, Calculation, Integration, Performance)
- **Coverage Target**: 90%+
- **Performance Tests**: 20+
- **Integration Tests**: 15+
- **Unit Tests**: 65+

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [GreenLang Documentation](../README.md)

## License

Copyright (c) 2024 GreenLang. All rights reserved.
