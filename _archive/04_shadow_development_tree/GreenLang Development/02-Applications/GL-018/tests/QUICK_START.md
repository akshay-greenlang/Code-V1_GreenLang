# GL-018 FLUEFLOW - Test Suite Quick Start

## Install Dependencies

```bash
cd GL-018
pip install -r tests/requirements-test.txt
```

## Run Tests

### All Tests with Coverage
```bash
pytest --cov --cov-report=html --cov-report=term-missing
```

### Unit Tests Only (95%+ coverage)
```bash
pytest tests/unit/ -v
```

### Integration Tests Only
```bash
pytest tests/integration/ -v
```

### Specific Calculator Tests
```bash
pytest tests/unit/test_combustion_analyzer.py -v
pytest tests/unit/test_efficiency_calculator.py -v
pytest tests/unit/test_emissions_calculator.py -v
```

### Critical Tests Only
```bash
pytest -m critical -v
```

### Performance Tests
```bash
pytest -m performance -v
```

## View Coverage Report

After running tests with `--cov-report=html`:
```bash
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

## Coverage Targets

- **Overall**: 85%+ ✓
- **Calculators**: 95%+ ✓
- **Agent**: 90%+ ✓
- **Config**: 85%+ ✓

## Test Structure

```
tests/
├── unit/                              # 250+ unit tests
│   ├── test_combustion_analyzer.py    # 60+ tests (95%+)
│   ├── test_efficiency_calculator.py  # 50+ tests (95%+)
│   ├── test_air_fuel_ratio_calculator.py  # 40+ tests (95%+)
│   └── test_emissions_calculator.py   # 45+ tests (95%+)
├── integration/                       # 40+ integration tests
│   └── test_end_to_end.py            # Complete workflows
├── test_data/
│   └── asme_ptc_reference.json       # ASME PTC 4.1 reference data
├── conftest.py                        # 30+ fixtures
├── pytest.ini                         # Test configuration
├── requirements-test.txt              # Dependencies
├── README.md                          # Full documentation
├── TEST_SUMMARY.md                    # Detailed summary
└── QUICK_START.md                     # This file
```

## Common Commands

### Run and open coverage report in one command
```bash
pytest --cov --cov-report=html && start htmlcov/index.html
```

### Run with parallel execution (faster)
```bash
pytest -n auto --cov
```

### Run with verbose output and show durations
```bash
pytest -v --durations=10
```

### Run only failed tests from last run
```bash
pytest --lf
```

### Run specific test by name
```bash
pytest -k "test_natural_gas_optimal"
```

## Interpreting Results

### Coverage Report
- **Green**: Good coverage (>90%)
- **Yellow**: Moderate coverage (70-90%)
- **Red**: Low coverage (<70%)

### Test Status
- **PASSED**: Test succeeded ✓
- **FAILED**: Test failed (assertion error)
- **ERROR**: Test error (exception)
- **SKIPPED**: Test skipped (conditional)

## Troubleshooting

### Import Errors
Ensure you're in the GL-018 directory:
```bash
cd GL-018
pytest
```

### Coverage Below Target
View missing coverage:
```bash
pytest --cov --cov-report=term-missing
```

### Slow Tests
Run without performance tests:
```bash
pytest -m "not performance"
```

## Next Steps

1. ✓ Install dependencies: `pip install -r tests/requirements-test.txt`
2. ✓ Run all tests: `pytest --cov`
3. ✓ View coverage report: `open htmlcov/index.html`
4. ✓ Verify 85%+ coverage achieved
5. ✓ Add tests to CI/CD pipeline
6. ✓ Run tests on every commit

## Support

- **Full Documentation**: See `tests/README.md`
- **Test Summary**: See `tests/TEST_SUMMARY.md`
- **Reference Data**: See `tests/test_data/asme_ptc_reference.json`

---

**Quick Tip**: Run `pytest --help` to see all available options!
