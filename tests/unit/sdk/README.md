# GreenLang SDK Unit Tests

Comprehensive test suite for the GreenLang framework's agent system.

## Quick Start

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all SDK tests
pytest tests/unit/sdk/ -v

# Run with coverage
pytest tests/unit/sdk/ --cov=greenlang.agents --cov-report=html --cov-report=term
```

## Test Files

### test_base_agent.py (886 lines)
Tests for `greenlang.agents.base.BaseAgent` - the foundation for all agents.

**Run:**
```bash
pytest tests/unit/sdk/test_base_agent.py -v
```

**Key Test Classes:**
- `TestAgentConfig` - Configuration validation
- `TestStatsTracker` - Metrics collection
- `TestBaseAgentHooks` - Pre/post execution hooks
- `TestBaseAgentErrorHandling` - Exception handling
- `TestBaseAgentResourceLoading` - File resource loading
- `TestBaseAgentEdgeCases` - Concurrent execution, large inputs

### test_data_processor.py (894 lines)
Tests for `greenlang.agents.data_processor.BaseDataProcessor` - batch data processing.

**Run:**
```bash
pytest tests/unit/sdk/test_data_processor.py -v
```

**Key Test Classes:**
- `TestDataProcessorConfig` - Batch size, workers validation
- `TestRecordProcessing` - Single record transformation
- `TestSequentialProcessing` - Sequential batch processing
- `TestParallelProcessing` - Parallel batch processing (2-4 workers)
- `TestProcessingStatistics` - Success rates, metrics
- `TestEdgeCases` - Performance comparison, order preservation

### test_calculator.py (897 lines)
Tests for `greenlang.agents.calculator.BaseCalculator` - mathematical operations.

**Run:**
```bash
pytest tests/unit/sdk/test_calculator.py -v
```

**Key Test Classes:**
- `TestCalculatorConfig` - Precision, cache configuration
- `TestPrecision` - Decimal rounding (0-28 places)
- `TestCaching` - Result caching, LRU eviction
- `TestSafeDivide` - Division by zero handling
- `TestUnitConverter` - Energy, mass, volume conversions
- `TestDeterminism` - Reproducible calculations

### test_reporter.py (927 lines)
Tests for `greenlang.agents.reporter.BaseReporter` - multi-format reporting.

**Run:**
```bash
pytest tests/unit/sdk/test_reporter.py -v
```

**Key Test Classes:**
- `TestReporterConfig` - Output format configuration
- `TestMarkdownRendering` - Markdown report generation
- `TestHTMLRendering` - HTML with CSS styling
- `TestJSONRendering` - JSON structured output
- `TestTableRendering` - Table formatting (Markdown/HTML)
- `TestEdgeCases` - Large datasets, Unicode, special characters

## Running Specific Tests

### By test class:
```bash
pytest tests/unit/sdk/test_base_agent.py::TestBaseAgentHooks -v
pytest tests/unit/sdk/test_calculator.py::TestCaching -v
```

### By test method:
```bash
pytest tests/unit/sdk/test_data_processor.py::TestParallelProcessing::test_parallel_processing -v
pytest tests/unit/sdk/test_reporter.py::TestMarkdownRendering::test_render_markdown_with_table -v
```

### By keyword:
```bash
# Run all caching tests
pytest tests/unit/sdk/ -k "cache" -v

# Run all validation tests
pytest tests/unit/sdk/ -k "validation" -v

# Run all error handling tests
pytest tests/unit/sdk/ -k "error" -v

# Run all edge case tests
pytest tests/unit/sdk/ -k "edge" -v
```

### With markers:
```bash
# Run all unit tests
pytest tests/unit/sdk/ -m unit -v

# Skip slow tests
pytest tests/unit/sdk/ -m "not slow" -v
```

## Coverage Reports

### Generate HTML coverage report:
```bash
pytest tests/unit/sdk/ \
  --cov=greenlang.agents \
  --cov-report=html \
  --cov-report=term-missing

# View report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Generate XML coverage (for CI):
```bash
pytest tests/unit/sdk/ \
  --cov=greenlang.agents \
  --cov-report=xml \
  --cov-report=term
```

### Coverage by module:
```bash
pytest tests/unit/sdk/test_base_agent.py \
  --cov=greenlang.agents.base \
  --cov-report=term-missing

pytest tests/unit/sdk/test_data_processor.py \
  --cov=greenlang.agents.data_processor \
  --cov-report=term-missing

pytest tests/unit/sdk/test_calculator.py \
  --cov=greenlang.agents.calculator \
  --cov-report=term-missing

pytest tests/unit/sdk/test_reporter.py \
  --cov=greenlang.agents.reporter \
  --cov-report=term-missing
```

## Debugging Tests

### Run with verbose output:
```bash
pytest tests/unit/sdk/test_base_agent.py -vv
```

### Show print statements:
```bash
pytest tests/unit/sdk/test_base_agent.py -s
```

### Run with PDB on failure:
```bash
pytest tests/unit/sdk/test_base_agent.py --pdb
```

### Show local variables on failure:
```bash
pytest tests/unit/sdk/test_base_agent.py -l
```

### Stop on first failure:
```bash
pytest tests/unit/sdk/ -x
```

### Show slowest tests:
```bash
pytest tests/unit/sdk/ --durations=10
```

## Parallel Execution

### Run tests in parallel:
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with 4 workers
pytest tests/unit/sdk/ -n 4

# Run with auto-detected CPU count
pytest tests/unit/sdk/ -n auto
```

## Test Statistics

| Metric | Value |
|--------|-------|
| Total Lines | 3,604 |
| Test Files | 4 |
| Test Classes | 52 |
| Test Methods | ~229 |
| Helper Classes | 21 |

## Test Coverage Areas

### BaseAgent (56 tests)
- ✅ Configuration & initialization
- ✅ Lifecycle management (validate, execute, cleanup)
- ✅ Pre/post execution hooks
- ✅ Metrics & statistics tracking
- ✅ Resource loading & caching
- ✅ Error handling & recovery
- ✅ Concurrent execution

### DataProcessor (58 tests)
- ✅ Batch processing (sequential & parallel)
- ✅ Record validation & transformation
- ✅ Error collection & thresholds
- ✅ Progress tracking
- ✅ Processing statistics
- ✅ Performance comparison

### Calculator (59 tests)
- ✅ High-precision arithmetic (0-28 decimal places)
- ✅ Calculation caching (LRU)
- ✅ Step-by-step tracking
- ✅ Unit conversion (energy, mass, volume)
- ✅ Safe division (zero handling)
- ✅ Deterministic results

### Reporter (56 tests)
- ✅ Multi-format output (Markdown, HTML, JSON, Excel)
- ✅ Data aggregation
- ✅ Section management
- ✅ Table & list rendering
- ✅ Summary generation
- ✅ Unicode & special characters

## CI/CD Integration

### GitHub Actions example:
```yaml
- name: Run SDK tests
  run: |
    pytest tests/unit/sdk/ \
      --cov=greenlang.agents \
      --cov-report=xml \
      --cov-report=term \
      -v

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

### Expected results:
- All tests should pass
- Coverage should be > 95%
- No warnings or errors

## Common Issues

### Import errors:
```bash
# Install package in development mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

### Missing dependencies:
```bash
# Install all test dependencies
pip install -e ".[test]"

# Or manually:
pip install pytest pytest-cov pytest-mock tqdm
```

### Python version:
Tests require Python >= 3.10 (as specified in pyproject.toml)

## Contributing

When adding new tests:

1. Follow existing naming conventions (`test_*` for functions, `Test*` for classes)
2. Add descriptive docstrings
3. Group related tests in classes
4. Use helper classes for reusable test agents
5. Test both happy path and error cases
6. Include edge cases and boundary values
7. Add concurrent execution tests where applicable

## Contact

For questions or issues:
- Repository: https://github.com/greenlang/greenlang
- Documentation: https://greenlang.io/docs
- Discord: https://discord.gg/greenlang
