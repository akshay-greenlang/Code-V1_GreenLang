# GreenLang Testing Framework

A comprehensive testing framework specifically designed for GreenLang infrastructure-based applications.

## Features

- **Agent Testing**: Test individual agents and pipelines with performance monitoring
- **LLM Testing**: Mock LLM responses, track tokens and costs, test caching
- **Cache Testing**: Test cache operations, hit rates, TTL, and performance
- **Database Testing**: Test database operations with automatic transaction rollback
- **Integration Testing**: End-to-end testing with Docker support
- **Mock Objects**: Complete mocks for all infrastructure components
- **Fixtures**: Sample data for common testing scenarios
- **Custom Assertions**: Specialized assertions for sustainability and emissions testing

## Quick Start

```python
from greenlang.testing import AgentTestCase

class TestMyAgent(AgentTestCase):
    def test_agent_execution(self):
        result = self.run_agent(MyAgent(), input_data)
        self.assert_success(result)
        self.assert_performance(result, max_time=2.0)
```

## Installation

The testing framework is included with GreenLang:

```bash
pip install greenlang
```

For development:

```bash
pip install greenlang[dev]
```

## Documentation

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for complete documentation.

## Directory Structure

```
greenlang/testing/
├── __init__.py              # Main module exports
├── agent_test.py            # Agent test cases
├── llm_test.py             # LLM test cases
├── cache_test.py           # Cache test cases
├── database_test.py        # Database test cases
├── integration_test.py     # Integration test cases
├── mocks.py                # Mock implementations
├── assertions.py           # Custom assertions
├── TESTING_GUIDE.md        # Complete testing guide
├── README.md               # This file
├── fixtures/               # Test fixtures
│   ├── sample_emissions_data.json
│   ├── sample_suppliers.yaml
│   ├── sample_config.yaml
│   ├── mock_llm_responses.json
│   └── test_database_schema.sql
├── templates/              # Test templates
│   ├── test_agent_template.py
│   ├── test_pipeline_template.py
│   ├── test_llm_template.py
│   └── test_integration_template.py
└── examples/               # Example tests
    ├── test_example_agent.py
    ├── test_example_llm.py
    ├── test_example_cache.py
    └── test_example_database.py
```

## Test Cases

### AgentTestCase

Test individual GreenLang agents:

```python
from greenlang.testing import AgentTestCase

class TestEmissionsAgent(AgentTestCase):
    def test_calculate_emissions(self):
        result = self.run_agent(agent, input_data)
        self.assert_success(result)
        self.assert_output_schema(result, schema)
```

### PipelineTestCase

Test agent pipelines:

```python
from greenlang.testing import PipelineTestCase

class TestEmissionsPipeline(PipelineTestCase):
    def test_full_pipeline(self):
        result = self.run_pipeline(pipeline, input_data)
        self.assert_pipeline_success(result)
        self.assert_all_stages_completed(result)
```

### LLMTestCase

Test LLM integrations:

```python
from greenlang.testing import LLMTestCase

class TestLLM(LLMTestCase):
    def test_with_mock(self):
        with self.mock_llm_response("Test response"):
            result = my_llm_function()
            self.assertEqual(result, "Test response")
```

### CacheTestCase

Test caching:

```python
from greenlang.testing import CacheTestCase

class TestCache(CacheTestCase):
    def test_cache_hit_rate(self):
        self.set_cache("key", "value")
        self.get_cache("key")
        self.assert_hit_rate(min_rate=0.8)
```

### DatabaseTestCase

Test database operations:

```python
from greenlang.testing import DatabaseTestCase

class TestDB(DatabaseTestCase):
    def test_insert_query(self):
        with self.db_transaction():
            self.db.insert("users", {"name": "John"})
            self.assert_record_exists("users", {"name": "John"})
```

### IntegrationTestCase

Test end-to-end workflows:

```python
from greenlang.testing import IntegrationTestCase

class TestIntegration(IntegrationTestCase):
    def test_full_workflow(self):
        result = self.run_end_to_end_test(test_data)
        self.assert_integration_success(result)
```

## Mock Objects

Complete mock implementations for testing without infrastructure:

- `MockChatSession`: Mock LLM interactions
- `MockCacheManager`: Mock cache operations
- `MockDatabaseManager`: Mock database operations
- `MockValidationFramework`: Mock validation
- `MockTelemetryManager`: Mock telemetry

```python
from greenlang.testing import MockChatSession

mock_chat = MockChatSession()
mock_chat.add_response("Mocked LLM response")
```

## Custom Assertions

Specialized assertions for GreenLang testing:

```python
from greenlang.testing import (
    assert_agent_result_valid,
    assert_schema_valid,
    assert_performance,
    assert_cache_hit_rate,
    assert_no_hallucination,
    assert_deterministic,
)
```

## Fixtures

Pre-built test data in `fixtures/`:

- `sample_emissions_data.json`: Sample emissions records
- `sample_suppliers.yaml`: Sample supplier data
- `sample_config.yaml`: Sample configuration
- `mock_llm_responses.json`: Pre-defined LLM responses
- `test_database_schema.sql`: Test database schema

```python
class TestMyAgent(AgentTestCase):
    def test_with_fixture(self):
        data = self.load_fixture('sample_emissions_data.json')
        result = self.run_agent(agent, data)
```

## Templates

Ready-to-use test templates in `templates/`:

- `test_agent_template.py`: Template for agent tests
- `test_pipeline_template.py`: Template for pipeline tests
- `test_llm_template.py`: Template for LLM tests
- `test_integration_template.py`: Template for integration tests

Copy a template and customize for your needs.

## Examples

Complete working examples in `examples/`:

- `test_example_agent.py`: Agent testing examples
- `test_example_llm.py`: LLM testing examples
- `test_example_cache.py`: Cache testing examples
- `test_example_database.py`: Database testing examples

Run examples:

```bash
python -m pytest greenlang/testing/examples/
```

## Best Practices

1. **Use fixtures** for test data instead of hardcoding
2. **Mock infrastructure** to avoid external dependencies
3. **Test performance** with `assert_performance()`
4. **Use transactions** for database tests (automatic rollback)
5. **Test error scenarios** not just happy paths
6. **Verify determinism** for critical agents
7. **Track metrics** (tokens, costs, cache hits)

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_my_agent.py

# Run with coverage
pytest --cov=greenlang --cov-report=html

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_my_agent.py::TestMyAgent::test_basic_execution
```

## CI/CD Integration

Example GitHub Actions workflow:

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
          python-version: '3.11'
      - run: pip install -e .[dev]
      - run: pytest tests/ --cov=greenlang
```

## Contributing

When adding new testing features:

1. Add test case to appropriate module
2. Add examples to `examples/`
3. Update documentation in `TESTING_GUIDE.md`
4. Add fixtures if needed
5. Write tests for your tests!

## License

Part of the GreenLang project. See main LICENSE file.

## Support

For questions and support:
- Documentation: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- Examples: `examples/` directory
- Templates: `templates/` directory

## Version

Version 1.0.0 - Complete testing framework for GreenLang infrastructure applications
