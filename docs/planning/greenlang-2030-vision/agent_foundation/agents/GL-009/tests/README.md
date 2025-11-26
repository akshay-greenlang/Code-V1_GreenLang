# GL-009 THERMALIQ Test Suite

Comprehensive test suite for the ThermalEfficiencyCalculator agent targeting 90%+ code coverage.

## Overview

This test suite provides thorough testing of all GL-009 components:

- **Unit Tests**: Isolated component testing (calculators, tools)
- **Integration Tests**: External system integration (connectors, APIs)
- **End-to-End Tests**: Complete workflow testing
- **Determinism Tests**: Reproducibility and zero-hallucination verification
- **Performance Tests**: Throughput and latency validation

## Test Structure

```
tests/
├── __init__.py                              # Test package initialization
├── conftest.py                              # Shared fixtures (400+ lines)
├── pytest.ini                               # Pytest configuration
├── README.md                                # This file
│
├── unit/                                    # Unit tests (isolated)
│   ├── __init__.py
│   ├── test_first_law_efficiency.py        # First Law calculator (400+ lines, 28 tests)
│   ├── test_second_law_efficiency.py       # Second Law calculator (350+ lines, 18 tests)
│   ├── test_heat_loss_calculator.py        # Heat loss calculator (400+ lines, 22 tests)
│   ├── test_sankey_generator.py            # Sankey diagram (350+ lines, 15 tests)
│   ├── test_benchmark_calculator.py        # Benchmarking (300+ lines, 12 tests)
│   ├── test_orchestrator.py                # Main orchestrator (500+ lines, 28 tests)
│   └── test_tools.py                       # LangChain tools (400+ lines, 22 tests)
│
├── integration/                             # Integration tests
│   ├── __init__.py
│   ├── test_connectors.py                  # External connectors (400+ lines, 18 tests)
│   └── test_api.py                         # FastAPI endpoints (350+ lines, 18 tests)
│
├── e2e/                                     # End-to-end tests
│   ├── __init__.py
│   └── test_complete_workflow.py           # Complete workflows (500+ lines, 12 tests)
│
├── determinism/                             # Determinism tests
│   ├── __init__.py
│   └── test_reproducibility.py             # Reproducibility (300+ lines, 12 tests)
│
└── fixtures/                                # Test data
    └── thermal_efficiency_test_cases.json  # Known test cases (200+ lines)
```

## Test Statistics

- **Total Test Files**: 15
- **Total Test Count**: 180+
- **Target Coverage**: 90%+
- **Total Lines of Test Code**: 4,500+

### Coverage by Component

| Component | Test File | Test Count | Target Coverage |
|-----------|-----------|------------|-----------------|
| First Law Calculator | test_first_law_efficiency.py | 28 | 95%+ |
| Second Law Calculator | test_second_law_efficiency.py | 18 | 90%+ |
| Heat Loss Calculator | test_heat_loss_calculator.py | 22 | 92%+ |
| Sankey Generator | test_sankey_generator.py | 15 | 88%+ |
| Benchmark Calculator | test_benchmark_calculator.py | 12 | 85%+ |
| Orchestrator | test_orchestrator.py | 28 | 90%+ |
| Tools | test_tools.py | 22 | 92%+ |
| Connectors | test_connectors.py | 18 | 85%+ |
| API Endpoints | test_api.py | 18 | 88%+ |
| E2E Workflows | test_complete_workflow.py | 12 | 85%+ |
| Determinism | test_reproducibility.py | 12 | 95%+ |

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Unit Tests Only

```bash
pytest tests/unit/
```

### Run Integration Tests

```bash
pytest tests/integration/ -m integration
```

### Run E2E Tests

```bash
pytest tests/e2e/ -m e2e
```

### Run Determinism Tests

```bash
pytest tests/determinism/ -m determinism
```

### Run with Coverage Report

```bash
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
```

The HTML coverage report will be generated in `tests/coverage_html/index.html`.

### Run Specific Test File

```bash
pytest tests/unit/test_first_law_efficiency.py -v
```

### Run Specific Test

```bash
pytest tests/unit/test_first_law_efficiency.py::TestFirstLawEfficiencyCalculator::test_calculate_basic_efficiency -v
```

### Run Tests Excluding Slow Tests

```bash
pytest tests/ -m "not slow"
```

### Run Tests in Parallel

```bash
pytest tests/ -n auto  # Requires pytest-xdist
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests (isolated component testing)
- `@pytest.mark.integration`: Integration tests (requires external systems)
- `@pytest.mark.e2e`: End-to-end workflow tests
- `@pytest.mark.slow`: Slow tests (typically E2E tests)
- `@pytest.mark.performance`: Performance/benchmark tests
- `@pytest.mark.determinism`: Determinism/reproducibility tests
- `@pytest.mark.asyncio`: Async tests

## Test Fixtures

Key fixtures available in `conftest.py`:

### Configuration Fixtures
- `test_config`: Test configuration settings
- `temp_database_path`: Temporary database for testing
- `temp_cache_dir`: Temporary cache directory

### Sample Data Fixtures
- `sample_reference_environment`: Reference environment for exergy
- `sample_surface_geometry`: Surface geometry for heat loss
- `sample_insulation_layers`: Insulation layers
- `sample_flue_gas_composition`: Flue gas composition
- `sample_thermal_efficiency_input`: Thermal efficiency inputs
- `sample_boiler_parameters`: Boiler operating parameters
- `sample_exergy_streams`: Exergy stream samples

### Mock Data Fixtures
- `mock_energy_meter_data`: Mock energy meter readings
- `mock_historian_data`: Mock historian time-series data
- `known_test_cases`: Test cases with known results
- `benchmark_data`: Industry benchmark data

### Mock Connector Fixtures
- `mock_energy_meter_connector`: Mock energy meter connector
- `mock_historian_connector`: Mock historian connector
- `mock_database_connection`: Mock database connection
- `mock_redis_cache`: Mock Redis cache

## Coverage Requirements

### Overall Target: 90%+

### Minimum Coverage by Component:
- **Calculators**: 90%+
  - First Law Efficiency: 95%+
  - Second Law Efficiency: 90%+
  - Heat Loss Calculator: 92%+
  - Benchmark Calculator: 85%+

- **Orchestrator**: 90%+
- **Tools**: 92%+
- **Connectors**: 85%+
- **API Endpoints**: 88%+

### What's Tested

#### First Law Efficiency Calculator
- ✅ Basic efficiency calculation
- ✅ Energy balance validation
- ✅ Multiple inputs/outputs
- ✅ Edge cases (0%, 100% efficiency)
- ✅ Input validation
- ✅ Provenance hash generation
- ✅ Calculation steps audit trail
- ✅ Warning generation
- ✅ Direct and indirect methods
- ✅ Precision rounding
- ✅ Loss breakdown

#### Second Law Efficiency Calculator
- ✅ Exergy efficiency calculation
- ✅ Stream exergy calculation
- ✅ Fuel exergy (multiple fuel types)
- ✅ Heat transfer exergy
- ✅ Combustion irreversibility
- ✅ Heat transfer irreversibility
- ✅ Reference environment handling
- ✅ Irreversibility breakdown
- ✅ First Law comparison
- ✅ Provenance tracking

#### Heat Loss Calculator
- ✅ Radiation loss (Stefan-Boltzmann)
- ✅ Natural convection (multiple orientations)
- ✅ Forced convection
- ✅ Conduction (single/multiple layers)
- ✅ Flue gas loss
- ✅ Unburned fuel loss
- ✅ Total loss calculation
- ✅ Loss breakdown percentages
- ✅ Temperature validation
- ✅ Geometry validation

#### Orchestrator
- ✅ All 8 operation modes
- ✅ Cache hit/miss scenarios
- ✅ Error handling
- ✅ Retry logic
- ✅ Provenance tracking
- ✅ Audit trail generation
- ✅ Batch processing
- ✅ Async operations
- ✅ Resource cleanup
- ✅ Health checks

#### Connectors
- ✅ Energy meter connection/disconnection
- ✅ Historian time-series queries
- ✅ SCADA Modbus/OPC-UA
- ✅ ERP integration
- ✅ Connection retry logic
- ✅ Data validation
- ✅ Mock server testing

#### API Endpoints
- ✅ Health check endpoints
- ✅ Calculation endpoints
- ✅ Error responses (400, 401, 404, 422, 500)
- ✅ Authentication (API key, JWT, OAuth2)
- ✅ Rate limiting
- ✅ CORS headers

#### Determinism
- ✅ Same input = same output
- ✅ Provenance hash consistency
- ✅ Bit-perfect reproducibility
- ✅ No randomness in calculations
- ✅ Cross-version determinism
- ✅ Seed verification
- ✅ Floating-point determinism

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov pytest-asyncio
      - run: pytest tests/ --cov=. --cov-report=xml --cov-fail-under=90
      - uses: codecov/codecov-action@v3
```

## Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_calculate_efficiency_with_zero_output`
2. **One assertion per test** (when possible)
3. **Use fixtures** for common setup
4. **Mock external dependencies**
5. **Test edge cases and error conditions**
6. **Validate provenance hashes**
7. **Check determinism** (same input = same output)
8. **Document complex test logic**

### Test Organization

- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test interactions with external systems
- **E2E tests**: Test complete workflows from start to finish
- **Performance tests**: Validate throughput and latency targets

### Coverage Guidelines

- Aim for 90%+ overall coverage
- 100% coverage of critical calculation paths
- Test all error handling paths
- Test all input validation
- Test boundary conditions

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Add parent directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Async Test Failures**
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio
```

**Coverage Not Generated**
```bash
# Install coverage dependencies
pip install pytest-cov coverage
```

**Slow Tests**
```bash
# Skip slow tests
pytest tests/ -m "not slow"
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Ensure 90%+ coverage** for new code
3. **Add fixtures** to `conftest.py` if reusable
4. **Update test counts** in this README
5. **Run full test suite** before committing

## License

Copyright © 2025 GreenLang Foundation. All rights reserved.

## Author

GL-TestEngineer
Version: 1.0.0
