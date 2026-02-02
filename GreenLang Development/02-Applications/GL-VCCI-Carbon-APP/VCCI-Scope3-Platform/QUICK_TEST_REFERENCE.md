# Quick Test Reference Guide
## GL-VCCI Scope 3 Platform

**For**: Developers and QA Engineers
**Date**: 2025-11-09
**Version**: 1.0.0

---

## Test Suite Overview

**Total Tests**: 1,722 tests
**New Tests Added**: 491 tests
**Test Coverage**: ~90%+

---

## Quick Commands

### Run All Tests
```bash
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform
pytest tests/ -v
```

### Run Specific Component Tests

#### Intake Agent (184 tests)
```bash
pytest tests/agents/intake/test_intake_agent.py -v
```

#### Calculator - Category 1 (35 tests)
```bash
pytest tests/agents/calculator/test_category_1.py -v
```

#### Calculator - Category 4 (30 tests)
```bash
pytest tests/agents/calculator/test_category_4.py -v
```

#### Calculator - Category 6 (25 tests)
```bash
pytest tests/agents/calculator/test_category_6.py -v
```

#### Hotspot/Engagement/Reporting (160 tests)
```bash
pytest tests/agents/test_comprehensive_suite.py -v
```

#### Integration Tests (30 tests)
```bash
pytest tests/integration/test_end_to_end_suite.py -v
```

#### Performance Tests (30 tests)
```bash
pytest tests/performance/test_performance_suite.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=services --cov-report=html --cov-report=term-missing
```

### Run Fast Tests Only
```bash
pytest tests/ -m "not slow" -v
```

---

## Test File Locations

### New Test Files
```
tests/agents/intake/test_intake_agent.py           (184 tests)
tests/agents/calculator/test_category_1.py         (35 tests)
tests/agents/calculator/test_category_4.py         (30 tests)
tests/agents/calculator/test_category_6.py         (25 tests)
tests/agents/test_comprehensive_suite.py           (160 tests)
tests/integration/test_end_to_end_suite.py         (30 tests)
tests/performance/test_performance_suite.py        (30 tests)
```

### Existing Test Files
```
tests/agents/calculator/test_category_*.py         (12 files)
tests/agents/engagement/                           (4 files)
tests/agents/hotspot/                              (2 files)
tests/agents/reporting/                            (3 files)
tests/connectors/                                  (20+ files)
tests/e2e/                                         (5 files)
tests/load/                                        (6 files)
tests/resilience/                                  (4 files)
tests/services/                                    (15+ files)
```

---

## Test Categories

### By Component
- **Intake Agent**: CSV/Excel ingestion, validation, entity resolution (184 tests)
- **Calculator Agent**: All 15 Scope 3 categories (300+ tests total)
- **Hotspot Agent**: Pareto, segmentation, insights (60 tests)
- **Engagement Agent**: Supplier selection, emails, tracking (50 tests)
- **Reporting Agent**: Reports, XBRL, PDF, multi-format (50 tests)
- **Integration**: End-to-end workflows (30 tests)
- **Performance**: Latency, throughput, load (30 tests)

### By Type
- **Unit Tests**: 70% (1,205 tests)
- **Integration Tests**: 20% (344 tests)
- **Performance Tests**: 5% (86 tests)
- **E2E Tests**: 5% (87 tests)

---

## Common Test Patterns

### Async Tests
```python
@pytest.mark.asyncio
async def test_something(calculator, mock_factor_broker):
    result = await calculator.calculate(input_data)
    assert result.emissions_kgco2e > 0
```

### Mocking External Services
```python
mock_factor = Mock(
    value=2.5,
    unit="kgCO2e/kg",
    source="test_db"
)
mock_factor_broker.resolve = AsyncMock(return_value=mock_factor)
```

### Data Validation Tests
```python
def test_validation_error():
    input_data = Category1Input(quantity=-1000.0)  # Invalid
    with pytest.raises(DataValidationError):
        await calculator.calculate(input_data)
```

---

## Fixtures Available

### From `conftest.py`
- `mock_factor_broker`: Mocked FactorBroker
- `mock_llm_client`: Mocked LLM client
- `mock_uncertainty_engine`: Mocked UncertaintyEngine
- `mock_provenance_builder`: Mocked ProvenanceChainBuilder
- `sample_tier1_input`: Sample Tier 1 input data
- `sample_tier2_input`: Sample Tier 2 input data
- `sample_tier3_input`: Sample Tier 3 input data

### From Individual Test Files
- `intake_agent`: ValueChainIntakeAgent instance
- `calculator`: Category-specific calculator instance
- `sample_csv_file`: Temporary CSV file
- `sample_excel_file`: Temporary Excel file

---

## Test Naming Convention

```
test_<component>_<scenario>_<expected_outcome>
```

**Examples**:
- `test_tier1_basic_calculation`
- `test_csv_with_unicode`
- `test_validation_negative_quantity`
- `test_pareto_80_20_rule`

---

## Performance Benchmarks

### Target SLAs
- Single calculation: P50 < 100ms, P95 < 200ms, P99 < 500ms
- Batch 10k records: < 30 seconds
- Batch 100k records: < 5 minutes
- Database queries: Factor lookup < 10ms, Entity resolution < 50ms

### Test Performance
```bash
pytest tests/performance/ --benchmark-only
```

---

## Troubleshooting

### Tests Failing Due to Missing Dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Tests Failing Due to Mock Issues
- Check that mocks are properly configured
- Verify AsyncMock is used for async functions
- Reset mocks between tests (autouse fixture handles this)

### Tests Timing Out
- Check if async tests have `@pytest.mark.asyncio` decorator
- Verify network-dependent tests are mocked
- Consider increasing timeout in pytest.ini

---

## Coverage Reports

### Generate HTML Coverage Report
```bash
pytest tests/ --cov=services --cov-report=html
# Open htmlcov/index.html in browser
```

### Generate Terminal Coverage Report
```bash
pytest tests/ --cov=services --cov-report=term-missing
```

### Coverage Targets
- **Overall**: 85%+ (Currently: ~90%+) ✅
- **Critical Paths**: 100% ✅
- **Error Handlers**: 100% ✅

---

## CI/CD Integration

### GitHub Actions Workflows
```
.github/workflows/greenlang-first-enforcement.yml
.github/workflows/security-scan.yml
.github/workflows/performance-regression.yml
```

### Pre-commit Hooks
Tests run automatically on commit (if configured)

---

## Best Practices

1. **Write Independent Tests**: No test should depend on another
2. **Use Descriptive Names**: Test name should explain what is tested
3. **Follow AAA Pattern**: Arrange, Act, Assert
4. **Mock External Services**: Never call real APIs in tests
5. **Test Edge Cases**: Zero, negative, very large values
6. **Test Error Paths**: Validate error handling
7. **Performance Test**: Include assertions on execution time
8. **Clean Up Resources**: Use fixtures for setup/teardown

---

## Getting Help

### Documentation
- `TEST_SUITE_COMPLETION_REPORT.md`: Full test suite documentation
- `README.md`: Platform overview
- `conftest.py`: Available fixtures

### Common Issues
- **Import errors**: Check `PYTHONPATH` includes project root
- **Async errors**: Ensure `@pytest.mark.asyncio` is present
- **Mock errors**: Verify mock configuration matches function signature

---

**Last Updated**: 2025-11-09
**Maintained By**: Test Suite Completion Team 1
