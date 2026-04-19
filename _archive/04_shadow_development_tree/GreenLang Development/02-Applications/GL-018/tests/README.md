# GL-018 FLUEFLOW Test Suite

Comprehensive test suite for GL-018 FLUEFLOW flue gas analyzer with **85%+ coverage target**.

## Test Structure

```
tests/
├── unit/                          # Unit tests (95%+ coverage target)
│   ├── test_combustion_analyzer.py
│   ├── test_efficiency_calculator.py
│   ├── test_air_fuel_ratio_calculator.py
│   ├── test_emissions_calculator.py
│   ├── test_config.py
│   └── test_flue_gas_analyzer_agent.py
├── integration/                   # Integration tests (80%+ coverage target)
│   ├── test_scada_integration.py
│   ├── test_api_endpoints.py
│   └── test_end_to_end.py
├── test_data/                     # Test data and fixtures
│   ├── asme_ptc_reference.json
│   └── test_scenarios.yaml
├── conftest.py                    # Pytest fixtures and configuration
└── README.md                      # This file
```

## Running Tests

### Run All Tests
```bash
cd GL-018
pytest
```

### Run with Coverage Report
```bash
pytest --cov=calculators --cov=config --cov=flue_gas_analyzer_agent --cov-report=html
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Calculator tests only (95%+ coverage)
pytest -m calculator

# Performance tests
pytest -m performance

# Critical tests (must pass)
pytest -m critical
```

### Run Specific Test Files
```bash
pytest tests/unit/test_combustion_analyzer.py
pytest tests/unit/test_efficiency_calculator.py
pytest tests/unit/test_emissions_calculator.py
```

## Test Categories (Pytest Markers)

- **`@pytest.mark.unit`**: Unit tests for individual functions/methods
- **`@pytest.mark.integration`**: Integration tests with external systems
- **`@pytest.mark.calculator`**: Calculator-specific tests (require 95%+ coverage)
- **`@pytest.mark.performance`**: Performance and benchmark tests
- **`@pytest.mark.compliance`**: Regulatory compliance tests (EPA, ASME)
- **`@pytest.mark.provenance`**: Provenance and determinism tests
- **`@pytest.mark.slow`**: Tests that take >5 seconds
- **`@pytest.mark.critical`**: Critical path tests (must pass)

## Coverage Targets

| Module                         | Target Coverage |
|--------------------------------|-----------------|
| **Calculators (all)**          | **95%+**        |
| - combustion_analyzer.py       | 95%+            |
| - efficiency_calculator.py     | 95%+            |
| - air_fuel_ratio_calculator.py | 95%+            |
| - emissions_calculator.py      | 95%+            |
| **Agent Orchestrator**         | **90%+**        |
| - flue_gas_analyzer_agent.py   | 90%+            |
| **Configuration**              | **85%+**        |
| - config.py                    | 85%+            |
| **Integrations**               | **80%+**        |
| - scada_integration.py         | 80%+            |
| **Overall Project**            | **85%+**        |

## Test Fixtures (conftest.py)

### Calculator Fixtures
- `combustion_analyzer`: CombustionAnalyzer instance
- `efficiency_calculator`: EfficiencyCalculator instance
- `air_fuel_ratio_calculator`: AirFuelRatioCalculator instance
- `emissions_calculator`: EmissionsCalculator instance

### Input Data Fixtures - Natural Gas
- `natural_gas_combustion_input`: Valid natural gas combustion input
- `natural_gas_efficiency_input`: Valid natural gas efficiency input
- `natural_gas_air_fuel_input`: Valid natural gas air-fuel ratio input
- `natural_gas_emissions_input`: Valid natural gas emissions input

### Input Data Fixtures - Fuel Oil
- `fuel_oil_combustion_input`: Valid fuel oil combustion input
- `fuel_oil_efficiency_input`: Valid fuel oil efficiency input

### Input Data Fixtures - Coal
- `coal_combustion_input`: Valid coal combustion input

### Edge Case Fixtures
- `low_O2_combustion_input`: Low O2 (rich combustion)
- `high_O2_combustion_input`: High O2 (excessive air)
- `wet_basis_combustion_input`: Wet basis input (requires conversion)

### Test Data Generators
- `scada_data_generator`: Generate mock SCADA time-series data
- `benchmark_dataset`: Large dataset for performance testing (1000 records)
- `excess_air_test_cases`: Parameterized excess air test cases
- `fuel_properties_test_cases`: All fuel types test cases
- `emissions_conversion_test_cases`: Unit conversion test cases

### Validation Helpers
- `provenance_validator`: Validate provenance record structure
- `tolerance_checker`: Floating point comparison helper

## Test Data Files

### ASME PTC 4.1 Reference Data
**File**: `test_data/asme_ptc_reference.json`

Contains validated test cases from ASME PTC 4.1 standards:
- Natural gas optimal conditions
- Fuel oil standard conditions
- Coal typical conditions
- Low O2 rich combustion
- High O2 lean combustion
- Emissions test cases
- Known conversion factors

## Test Development Guidelines

### 1. Test Coverage Requirements
- **Unit tests**: Test every public method and function
- **Edge cases**: Test boundary conditions (min/max values)
- **Error handling**: Test all ValueError paths
- **Provenance**: Test determinism (same input → same hash)
- **Performance**: Validate <5ms per calculation

### 2. Test Naming Convention
```python
def test_<method_name>_<scenario>():
    """Test description."""
    pass

# Examples:
def test_natural_gas_optimal_combustion():
def test_invalid_O2_raises_error():
def test_provenance_determinism():
```

### 3. Assertion Guidelines
- Use `pytest.approx()` for floating point comparisons
- Specify relative tolerance: `pytest.approx(expected, rel=0.01)` (1%)
- Validate provenance: `assert verify_provenance(provenance) is True`
- Check SHA-256 hash length: `assert len(provenance_hash) == 64`

### 4. Parametrized Tests
Use `@pytest.mark.parametrize` for multiple test scenarios:
```python
@pytest.mark.parametrize("O2_pct,expected_excess_air", [
    (3.5, 20.0),
    (4.0, 23.5),
    (5.0, 31.25),
])
def test_excess_air_calculation(O2_pct, expected_excess_air):
    result = calculate_excess_air_from_O2(O2_pct)
    assert result == pytest.approx(expected_excess_air, rel=0.01)
```

## Known Test Values (ASME PTC 4.1)

| Test Case                | Input            | Expected Output      |
|--------------------------|------------------|----------------------|
| Excess Air from O2       | O2 = 3.5%        | Excess Air = 20.0%   |
| Excess Air from O2       | O2 = 4.0%        | Excess Air = 23.5%   |
| Stack Loss (Siegert)     | 180°C, 12% CO2   | Stack Loss = 6.7%    |
| NOx ppm to mg/Nm³        | 100 ppm          | 205.25 mg/Nm³        |
| CO ppm to mg/Nm³         | 100 ppm          | 124.93 mg/Nm³        |
| Lambda from O2           | O2 = 3.5%        | λ = 1.200            |

## Performance Benchmarks

| Metric                        | Target            | Actual |
|-------------------------------|-------------------|--------|
| Single calculation time       | <5 ms             | TBD    |
| Batch processing throughput   | >1000 records/sec | TBD    |
| Test suite execution time     | <2 minutes        | TBD    |
| Memory usage per calculation  | <1 MB             | TBD    |

## CI/CD Integration

### GitHub Actions / GitLab CI
```yaml
- name: Run Tests with Coverage
  run: |
    pytest --cov --cov-report=xml --cov-report=term-missing

- name: Upload Coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    fail_ci_if_error: true
    verbose: true
```

### Coverage Badges
Add to README.md:
```markdown
![Coverage](https://img.shields.io/codecov/c/github/yourorg/greenlang)
```

## Troubleshooting

### Issue: Tests failing with import errors
**Solution**: Ensure parent directory is in Python path:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

### Issue: Coverage below 85%
**Solution**:
1. Run `pytest --cov --cov-report=html`
2. Open `htmlcov/index.html`
3. Identify uncovered lines
4. Add tests for missing coverage

### Issue: Performance tests failing
**Solution**:
- Run on dedicated machine (no background processes)
- Use `@pytest.mark.performance` to skip in CI
- Increase timeout if needed

## Continuous Improvement

### Adding New Tests
1. Create test file in appropriate directory (`unit/` or `integration/`)
2. Follow naming convention: `test_<module_name>.py`
3. Add fixtures to `conftest.py` if reusable
4. Add test data to `test_data/` if needed
5. Run tests: `pytest tests/unit/test_new_module.py`
6. Verify coverage: `pytest --cov=module --cov-report=term-missing`

### Test Review Checklist
- [ ] All public methods tested
- [ ] Edge cases covered (boundary values)
- [ ] Error handling tested (ValueError paths)
- [ ] Provenance determinism verified
- [ ] Performance targets met (<5ms)
- [ ] Coverage target achieved (95%+ for calculators)
- [ ] Documentation complete (docstrings)

## References

- **ASME PTC 4.1**: Fired Steam Generators Performance Test Code
- **EPA Method 19**: Determination of SO2 Removal Efficiency
- **ISO 10396**: Stationary Source Emissions - Sampling
- **EN 14181**: Quality Assurance of Automated Measuring Systems

## Contact

For questions or issues with the test suite:
- **Author**: GL-TestEngineer
- **Version**: 1.0.0
- **Last Updated**: December 2025
