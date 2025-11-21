# GreenLang Agent Foundation Testing Framework

## Overview

Comprehensive testing framework for GreenLang AI agents with **90%+ test coverage target**, 12-dimension quality validation, and full lifecycle testing capabilities.

## Architecture

### Test Categories

1. **Unit Tests** (`unit_tests/`)
   - Test individual components in isolation
   - Target: 95%+ coverage
   - Fast execution (<1s per test)

2. **Integration Tests** (`integration_tests/`)
   - Test component interactions
   - Multi-agent workflows
   - External system integrations

3. **End-to-End Tests** (`e2e_tests/`)
   - Complete workflow validation
   - Real-world scenarios
   - Production-like environments

4. **Performance Tests** (`performance_tests/`)
   - Load testing
   - Stress testing
   - Benchmark validation

## Key Features

### 1. Full Lifecycle Testing

Test all agent lifecycle states from Architecture spec (lines 58-66):

```python
from testing.agent_test_framework import AgentTestCase, AgentState

class TestAgentLifecycle(AgentTestCase):
    def test_full_lifecycle(self):
        agent = self.create_mock_agent(MyAgent)

        # Test complete lifecycle
        self.assert_lifecycle_transition(agent, AgentState.CREATED, AgentState.INITIALIZING)
        self.assert_lifecycle_transition(agent, AgentState.INITIALIZING, AgentState.READY)
        self.assert_lifecycle_transition(agent, AgentState.READY, AgentState.RUNNING)
        # ... and so on
```

### 2. 12-Dimension Quality Framework

Based on ISO 25010 standards (Architecture lines 1099-1221):

```python
from testing.quality_validators import ComprehensiveQualityValidator

validator = ComprehensiveQualityValidator(target_score=0.8)
report = validator.validate_agent(my_agent, test_data)

print(f"Overall Score: {report.overall_score:.1%}")
print(f"Passed: {report.passed}")
print(f"Recommendations: {report.recommendations}")
```

**Quality Dimensions:**
1. Functional Quality
2. Performance Efficiency
3. Compatibility
4. Usability
5. Reliability
6. Security
7. Maintainability
8. Portability
9. Scalability
10. Interoperability
11. Reusability
12. Testability

### 3. Deterministic Testing

Zero-hallucination guarantees with reproducible results:

```python
from testing.agent_test_framework import DeterministicLLMProvider

llm = DeterministicLLMProvider(seed=42)
response = llm.generate("Test prompt")

# Always returns same response for same prompt
assert response["deterministic"] == True
```

### 4. Provenance Tracking

Validate SHA-256 provenance chains:

```python
from testing.agent_test_framework import ProvenanceValidator

result = agent.process(input_data)
is_valid, errors = ProvenanceValidator.validate_chain(result.provenance_chain)

assert is_valid
assert len(errors) == 0
```

### 5. Performance Validation

Validate against Architecture performance targets (lines 22-28):

```python
from testing.agent_test_framework import PerformanceTestRunner

runner = PerformanceTestRunner()

# Test agent creation: <100ms target
result = runner.test_agent_creation_performance(MyAgent, iterations=100)
assert result["p99_ms"] < 100

# Test message passing: <10ms target
result = runner.test_message_passing_performance(num_agents=10)
assert result["p99_ms"] < 10

# Test memory retrieval: <50ms target
result = runner.test_memory_retrieval_performance(memory_size=10000)
assert result["p99_ms"] < 50
```

## Usage

### Running Tests

```bash
# Run all tests
pytest

# Run specific category
pytest unit_tests/
pytest integration_tests/
pytest e2e_tests/
pytest performance_tests/

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific markers
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m compliance

# Run with performance monitoring
pytest --benchmark-only
```

### Configuration

See `pytest.ini` for complete configuration:

- **Coverage Target:** 90%
- **Timeout:** 300s
- **Parallel Execution:** Available with pytest-xdist
- **Test Markers:** unit, integration, e2e, performance, security, compliance

### Test Fixtures

Available in `conftest.py`:

```python
def test_my_agent(
    mock_agent_config,      # Mock agent configuration
    mock_llm_provider,      # Deterministic LLM
    mock_vector_store,      # Mock vector database
    mock_rag_system,        # Mock RAG system
    test_data_generator,    # Test data generation
    quality_validator,      # Quality validation
    performance_runner      # Performance testing
):
    # Your test code here
    pass
```

## Test Data Generation

```python
from testing.agent_test_framework import TestDataGenerator

generator = TestDataGenerator(seed=42)

# Generate agent configs
configs = generator.generate_agent_configs(count=10)

# Generate test messages
messages = generator.generate_test_messages(count=100)

# Generate memory entries
memories = generator.generate_memory_entries(count=1000)

# Generate carbon emissions data
carbon_data = generator.generate_carbon_data(count=100)
```

## Quality Validation

### Individual Dimension

```python
from testing.quality_validators import PerformanceValidator

validator = PerformanceValidator(target_score=0.8)
metric = validator.validate(agent, test_data)

print(f"Score: {metric.score:.1%}")
print(f"Passed: {metric.passed}")
print(f"Measurements: {metric.measurements}")
```

### Comprehensive Report

```python
from testing.quality_validators import ComprehensiveQualityValidator

validator = ComprehensiveQualityValidator(target_score=0.8)
report = validator.validate_agent(agent, test_data)

# Generate HTML report
validator.generate_html_report(report, "quality_report.html")
```

## Performance Benchmarking

```python
from testing.agent_test_framework import PerformanceTestRunner

runner = PerformanceTestRunner()

# Test agent creation
result = runner.test_agent_creation_performance(MyAgent, iterations=100)

# Test message passing
result = runner.test_message_passing_performance(num_agents=10)

# Test memory retrieval
result = runner.test_memory_retrieval_performance(memory_size=10000)

# Generate report
report = runner.generate_report()
print(f"Pass Rate: {report['summary']['pass_rate']:.1%}")
```

## Coverage Analysis

```python
from testing.agent_test_framework import CoverageAnalyzer

analyzer = CoverageAnalyzer(target_coverage=0.90)

# Analyze modules
analyzer.analyze_module("core.agent_base", 0.95)
analyzer.analyze_module("intelligence.llm_orchestration", 0.88)

# Generate report
report = analyzer.get_report()
print(f"Overall Coverage: {report['overall_coverage']:.1%}")
print(f"Passed: {report['passed']}")
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      - name: Run tests
        run: pytest --cov=. --cov-report=xml --cov-fail-under=90
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### GitLab CI

```yaml
test:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest --cov=. --cov-report=html --cov-fail-under=90
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

## Best Practices

### 1. Test Structure

```python
class TestMyAgent(AgentTestCase):
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.agent = self.create_mock_agent(MyAgent)

    def test_initialization(self):
        """Test agent initialization."""
        # Arrange - prepare test data
        config = {"name": "test"}

        # Act - execute functionality
        agent = MyAgent(config)

        # Assert - verify results
        self.assertEqual(agent.config["name"], "test")
```

### 2. Mocking LLMs

```python
def test_with_mock_llm(self, mock_llm_provider):
    """Test with deterministic LLM."""
    agent = MyAgent(llm_client=mock_llm_provider)

    result = agent.process_with_llm("Test prompt")

    # LLM responses are deterministic
    assert result is not None
```

### 3. Performance Testing

```python
def test_performance(self):
    """Test performance requirements."""
    with self.assert_performance(max_duration_ms=100, max_memory_mb=50):
        result = self.agent.process(large_dataset)

    assert result is not None
```

### 4. Zero-Hallucination

```python
def test_calculation_accuracy(self):
    """Test calculation accuracy."""
    result = agent.calculate_emissions(quantity=100, factor=2.5)
    expected = 250.0

    self.assert_zero_hallucination(result, expected, tolerance=1e-6)
```

## Coverage Targets

| Component | Target | Current |
|-----------|--------|---------|
| Core Agents | 95% | - |
| Memory Systems | 90% | - |
| Intelligence Layer | 85% | - |
| Communication | 90% | - |
| Overall | 90% | - |

## Performance Targets

From Architecture document (lines 22-28):

| Metric | Target |
|--------|--------|
| Agent Creation | <100ms |
| Message Passing | <10ms |
| Memory Retrieval (Recent) | <50ms |
| Memory Retrieval (Long-term) | <200ms |
| LLM Call Average | <2s |
| LLM Call P99 | <5s |
| Concurrent Agents | 10,000+ |

## Quality Targets

| Dimension | Target |
|-----------|--------|
| Functional Quality | >90% |
| Performance | >85% |
| Reliability | >95% |
| Security | >90% |
| Maintainability | >80% |
| Overall | >80% |

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Coverage Not Met**
   ```bash
   pytest --cov=. --cov-report=term-missing
   # Check missing lines in report
   ```

3. **Performance Test Failures**
   ```bash
   pytest -m performance --benchmark-autosave
   # Compare benchmarks over time
   ```

4. **Async Test Issues**
   ```bash
   pytest --asyncio-mode=auto
   ```

## Documentation

- [Agent Architecture](../Agent_Foundation_Architecture.md)
- [Testing Best Practices](./docs/testing_best_practices.md)
- [CI/CD Integration Guide](./docs/cicd_integration.md)
- [Coverage Reports](./htmlcov/index.html)

## Contributing

1. Write tests first (TDD approach)
2. Maintain 90%+ coverage
3. Use pytest fixtures
4. Follow naming conventions
5. Document test purpose
6. Include performance assertions
7. Validate quality dimensions

## Support

For questions or issues:
- Review documentation
- Check existing tests for examples
- Run tests with `-v` flag for details
- Check coverage reports for gaps

---

**Version:** 1.0.0
**Last Updated:** November 2024
**Maintainer:** GreenLang Test Engineering Team