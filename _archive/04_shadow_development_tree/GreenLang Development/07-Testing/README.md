# GreenLang Test Suite

Comprehensive test suite with **200+ tests** for GreenLang carbon emissions calculator, including unit tests, integration tests, example tests, and property-based testing.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=greenlang --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m property      # Property-based tests
pytest -m "not slow"    # Skip slow tests

# Run tests in parallel
pytest -n auto
```

## Test Structure

```
tests/
├── integration/        # 70+ integration tests
│   ├── conftest.py    # Integration test configuration
│   ├── test_workflow_*.py  # Workflow-specific tests
│   └── utils/         # Test utilities
├── unit/              # 100+ unit tests
│   ├── agents/        # Agent-specific tests
│   ├── cli/           # CLI command tests
│   ├── sdk/           # SDK client tests
│   ├── data/          # Data schema tests
│   └── core/          # Core framework tests
├── property/          # Property-based tests
├── fixtures/          # Test data files
│   ├── data/         # Sample building data
│   ├── workflows/    # Test workflows
│   └── snapshots/    # Golden file snapshots
└── utils/            # Test utilities
```

## Testing Contract

### Coverage Requirements (CI/CD Enforced)

| Component | Minimum Coverage | Status |
|-----------|-----------------|--------|
| Overall | ≥85% | ✅ Enforced |
| agents/ | ≥90% | ✅ Enforced |
| core/ | ≥90% | ✅ Enforced |
| sdk/ | ≥85% | ✅ Enforced |

### Performance Budgets

| Operation | Maximum Time |
|-----------|-------------|
| Single Building | <2 seconds |
| Portfolio (10 buildings) | <5 seconds |
| Large Portfolio (100 buildings) | <30 seconds |
| Workflow Execution | <10 seconds |
| Full Test Suite | <90 seconds |

### Numerical Invariants

All emissions calculations must satisfy:
- **Precision**: ε ≤ 1e-9
- **Conservation**: Sum of parts equals total (within ε)
- **Non-negative**: All emissions ≥ 0
- **Deterministic**: Same input → same output

## Test Categories

### Unit Tests (`tests/unit/`) - 200+ tests

#### Agent Tests
- **FuelAgent**: Emissions calculations, unit conversions, error handling
- **GridFactorAgent**: Factor retrieval, country support, provenance
- **InputValidatorAgent**: Data validation, normalization, error messages
- **CarbonAgent**: Aggregation, percentages, breakdown accuracy
- **IntensityAgent**: Intensity metrics, division by zero handling
- **BenchmarkAgent**: Rating boundaries, regional variations
- **BuildingProfileAgent**: Profile generation, climate impacts
- **RecommendationAgent**: Action generation, prioritization
- **ReportAgent**: Report formats, schema compliance

#### Framework Tests
- **Base Agent Contract**: Response structure, error handling
- **CLI Commands**: Version, help, calculate, benchmark
- **SDK Client**: API methods, batch processing, error handling
- **Data Schemas**: Emission factors, benchmarks validation

### Integration Tests (`tests/integration/`) - 300+ tests

#### Workflow Tests
- **Commercial Building E2E**: Complete building analysis workflow
- **India-specific Workflows**: Regional calculations and benchmarks
- **Portfolio Analysis**: Multi-building aggregation
- **Cross-country Comparisons**: Global emission comparisons
- **CLI End-to-end**: Command-line interface testing
- **Parallel Execution**: Concurrent workflow processing
- **Error Handling**: Graceful failure and validation
- **Provenance Tracking**: Data lineage and versioning
- **Performance Benchmarks**: Speed and resource usage
- **Plugin Discovery**: Dynamic agent loading
- **Backward Compatibility**: Schema evolution support

#### Test Features
- **Network Isolation**: No external API calls
- **Deterministic Execution**: Seeded random, mocked LLMs
- **Snapshot Testing**: Golden file comparisons
- **Numerical Invariants**: Mathematical correctness

### Property Tests (`tests/property/`)
- **Additivity**: Emissions sum correctly when split
- **Proportionality**: Linear scaling of emissions
- **Unit Round-trips**: Conversions are reversible
- **Non-negativity**: Valid inputs never yield negative results

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest tests/unit/agents/test_fuel_agent.py

# Run specific test
pytest tests/unit/agents/test_fuel_agent.py::TestFuelAgent::test_electricity_india_kwh

# Run with coverage
pytest --cov=greenlang --cov-report=term-missing
```

### Parallel Execution

```bash
# Auto-detect CPU cores
pytest -n auto

# Use specific number of workers
pytest -n 4
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only property tests
pytest -m property

# Skip slow tests
pytest -m "not slow"

# Run CLI tests
pytest -m cli
```

## Writing Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure

```python
class TestFuelAgent:
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = FuelAgent()
    
    def test_electricity_calculation(self):
        """Test electricity emissions calculation."""
        result = self.agent.run({
            "fuel_type": "electricity",
            "consumption": {"value": 1000, "unit": "kWh"},
            "country": "IN"
        })
        
        assert result["success"] is True
        assert_close(result["data"]["co2e_emissions_kg"], 710.0)
```

### Using Fixtures

```python
@pytest.fixture
def sample_building():
    """Provide sample building data."""
    return {
        "building_type": "office",
        "country": "IN",
        "total_area": {"value": 50000, "unit": "sqft"}
    }

def test_with_fixture(sample_building):
    """Test using fixture."""
    agent = InputValidatorAgent()
    result = agent.run(sample_building)
    assert result["success"] is True
```

## Test Utilities

### Numeric Assertions

```python
from tests.utils import assert_close

# Compare floats with tolerance
assert_close(actual, expected, rel_tol=1e-9)

# Check percentage sum
assert_percentage_sum([75.0, 25.0], expected_sum=100.0)
```

### Unit Conversions

```python
from tests.utils import normalize_factor

# Convert between units
mwh_value = normalize_factor(kwh_value, "kWh", "MWh")
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements-test.txt
      - run: pytest --cov=greenlang
      - run: mypy greenlang --strict
      - run: ruff check greenlang
```

### Pre-commit Hooks

```yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## Debugging Tests

### Run with debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Maximum verbosity
pytest -vvv

# Show print statements
pytest -s
```

### Test-specific logging

```python
import logging

def test_with_logging(caplog):
    """Test with captured logs."""
    with caplog.at_level(logging.DEBUG):
        result = agent.run(data)
        
    assert "Processing" in caplog.text
```

## Performance Testing

```bash
# Run with benchmark
pytest --benchmark-only

# Compare benchmarks
pytest --benchmark-compare

# Save benchmark results
pytest --benchmark-save=baseline
```

## Test Maintenance

### Update snapshots

```bash
# Update all snapshots
pytest --snapshot-update

# Update specific snapshots
pytest tests/unit/agents/test_report_agent.py --snapshot-update
```

### Check test quality

```bash
# Mutation testing
mutmut run

# Test complexity
radon cc tests/ -s

# Dead code detection
vulture tests/
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure PYTHONPATH includes project root
2. **Fixture not found**: Check fixture scope and location
3. **Flaky tests**: Use `@pytest.mark.flaky(reruns=3)`
4. **Slow tests**: Mark with `@pytest.mark.slow` and skip in CI

### Environment Variables

```bash
# Set test environment
export GREENLANG_ENV=test

# Disable cache
export PYTEST_DISABLE_CACHE=1

# Increase timeout
export PYTEST_TIMEOUT=300
```

## Contributing

1. Write tests for new features
2. Ensure all tests pass
3. Maintain coverage above thresholds
4. Update test documentation
5. Add appropriate markers

## License

MIT License - See LICENSE file for details.