# GreenLang Testing Framework Guide

Complete guide to testing GreenLang infrastructure-based applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Testing Agents](#testing-agents)
4. [Testing LLM Integration](#testing-llm-integration)
5. [Testing Caching](#testing-caching)
6. [Testing Databases](#testing-databases)
7. [Integration Testing](#integration-testing)
8. [Best Practices](#best-practices)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

The GreenLang Testing Framework provides comprehensive tools for testing infrastructure-based applications built with GreenLang. It includes:

- **Test Cases**: Specialized base classes for different testing scenarios
- **Mocks**: Mock implementations of infrastructure components
- **Fixtures**: Sample data for testing
- **Assertions**: Custom assertions for validating results
- **Templates**: Ready-to-use test templates

### Installation

```python
from greenlang.testing import (
    AgentTestCase,
    PipelineTestCase,
    LLMTestCase,
    CacheTestCase,
    DatabaseTestCase,
    IntegrationTestCase
)
```

---

## Quick Start

### Basic Agent Test

```python
from greenlang.testing import AgentTestCase
from your_module import EmissionsCalculatorAgent

class TestEmissionsAgent(AgentTestCase):
    def test_calculate_emissions(self):
        agent = EmissionsCalculatorAgent()

        input_data = {
            "quantity": 1000,
            "unit": "kg",
            "material": "steel"
        }

        result = self.run_agent(agent, input_data)

        self.assert_success(result)
        self.assertGreater(result['result']['emissions'], 0)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_emissions_agent.py

# Run with coverage
python -m pytest --cov=greenlang tests/
```

---

## Testing Agents

### AgentTestCase

Base class for testing GreenLang agents.

#### Basic Usage

```python
from greenlang.testing import AgentTestCase

class TestMyAgent(AgentTestCase):
    def setUp(self):
        super().setUp()
        self.agent = MyAgent()
        self.test_data = {"key": "value"}

    def test_agent_execution(self):
        result = self.run_agent(self.agent, self.test_data)
        self.assert_success(result)
```

#### Available Methods

**`run_agent(agent, input_data, **kwargs)`**
- Runs an agent and tracks performance
- Returns dict with `result`, `execution_time`, `memory_used`, `success`

**`run_agent_batch(agent, input_batch, **kwargs)`**
- Runs agent with multiple inputs
- Returns list of results

**`assert_success(result)`**
- Asserts agent execution succeeded

**`assert_output_schema(result, schema)`**
- Validates output against JSON schema or Pydantic model

**`assert_performance(result, max_time, max_memory)`**
- Validates execution time and memory usage

**`assert_deterministic(agent, input_data, runs=3)`**
- Validates agent produces consistent results

#### Example: Testing Output Schema

```python
def test_output_schema(self):
    schema = {
        "type": "object",
        "properties": {
            "emissions": {"type": "number"},
            "unit": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["emissions", "unit"]
    }

    result = self.run_agent(self.agent, self.test_data)
    self.assert_output_schema(result, schema)
```

#### Example: Testing Performance

```python
def test_performance(self):
    result = self.run_agent(self.agent, self.test_data)

    # Max 2 seconds, max 100MB memory
    self.assert_performance(
        result,
        max_time=2.0,
        max_memory=100 * 1024 * 1024
    )
```

#### Example: Testing with Mocks

```python
def test_with_mock_infrastructure(self):
    # Set up mock LLM response
    self.mock_chat.add_response("Calculated emissions: 2500 kg CO2e")

    with self.mock_infrastructure():
        result = self.run_agent(self.agent, self.test_data)
        self.assert_success(result)
```

#### Example: Loading Fixtures

```python
def test_with_fixture_data(self):
    # Load sample emissions data
    fixture = self.load_fixture('sample_emissions_data.json')

    result = self.run_agent(self.agent, fixture)
    self.assert_success(result)
```

### PipelineTestCase

For testing agent pipelines.

```python
from greenlang.testing import PipelineTestCase

class TestEmissionsPipeline(PipelineTestCase):
    def test_full_pipeline(self):
        pipeline = EmissionsCalculationPipeline()

        result = self.run_pipeline(pipeline, self.input_data)

        self.assert_pipeline_success(result)
        self.assert_all_stages_completed(result)
        self.assert_pipeline_performance(result, max_time=10.0)
```

#### Testing Individual Stages

```python
def test_individual_stages(self):
    result = self.run_pipeline(self.pipeline, self.input_data)

    # Validate stage 1 output
    self.assert_stage_output(
        result,
        "data_validation",
        {"type": "object", "properties": {"valid": {"type": "boolean"}}}
    )

    # Validate stage 2 output
    self.assert_stage_output(
        result,
        "emissions_calculation",
        {"type": "object", "properties": {"total_emissions": {"type": "number"}}}
    )
```

---

## Testing LLM Integration

### LLMTestCase

For testing LLM interactions, caching, token counting, and cost tracking.

#### Mocking Single Response

```python
from greenlang.testing import LLMTestCase

class TestLLMIntegration(LLMTestCase):
    def test_basic_llm_call(self):
        with self.mock_llm_response("Calculated: 2500 kg CO2e"):
            result = my_llm_function("Calculate emissions")
            self.assertEqual(result, "Calculated: 2500 kg CO2e")
```

#### Mocking Multiple Responses

```python
def test_multiple_calls(self):
    responses = [
        "First response",
        "Second response",
        "Third response"
    ]

    with self.mock_llm_responses(responses):
        r1 = my_llm_function("Prompt 1")
        r2 = my_llm_function("Prompt 2")
        r3 = my_llm_function("Prompt 3")

        self.assertEqual(r1, "First response")
        self.assertEqual(r2, "Second response")
        self.assertEqual(r3, "Third response")
```

#### Testing Caching

```python
def test_caching_works(self):
    # First call - cache miss
    with self.mock_llm_response("Result", cached=False):
        result1 = my_llm_function("Same prompt")

    # Second call - cache hit
    with self.mock_llm_response("Result", cached=True):
        result2 = my_llm_function("Same prompt")
        self.assert_cache_hit(expected=True)
```

#### Testing Token Counting

```python
def test_token_usage(self):
    with self.mock_llm_response("Response", tokens=1000):
        result = my_llm_function("Long prompt")

    # Assert total tokens within limit
    self.assert_total_tokens(max_tokens=1500)
```

#### Testing Cost Tracking

```python
def test_cost_tracking(self):
    with self.mock_llm_response("Response", tokens=1000, cost=0.01):
        result = my_llm_function("Prompt")

    # Assert cost within budget
    self.assert_total_cost(max_cost=0.02)
```

#### Testing Streaming Responses

```python
def test_streaming(self):
    chunks = ["Hello", " ", "world", "!"]

    with self.mock_streaming_response(chunks):
        collected = []
        for chunk in my_streaming_function("Test"):
            collected.append(chunk)

        self.assertEqual(collected, chunks)
```

#### Testing Response Format

```python
def test_json_response(self):
    json_response = '{"emissions": 2500, "unit": "kg CO2e"}'

    with self.mock_llm_response(json_response):
        result = my_llm_function("Get JSON")
        self.assert_response_format(result, 'json')
```

#### Getting LLM Metrics

```python
def test_metrics(self):
    with self.mock_llm_response("Response", tokens=100, cost=0.001):
        my_llm_function("Test")

    metrics = self.get_llm_metrics()

    # Metrics include:
    # - total_calls
    # - cache_hits, cache_misses
    # - cache_hit_rate
    # - total_tokens, avg_tokens_per_call
    # - total_cost, avg_cost_per_call
```

---

## Testing Caching

### CacheTestCase

For testing cache operations, performance, TTL, and invalidation.

```python
from greenlang.testing import CacheTestCase

class TestCaching(CacheTestCase):
    def test_basic_cache_operations(self):
        # Set value
        self.set_cache("key", "value")

        # Get value
        value, time, is_hit = self.get_cache("key")

        self.assertEqual(value, "value")
        self.assertTrue(is_hit)
```

#### Testing Cache Hit Rate

```python
def test_hit_rate(self):
    # Set some values
    self.set_cache("key1", "value1")
    self.set_cache("key2", "value2")

    # Hit
    self.get_cache("key1")
    # Hit
    self.get_cache("key2")
    # Miss
    self.get_cache("key3")

    # Assert hit rate at least 66%
    self.assert_hit_rate(min_rate=0.66)
```

#### Testing TTL Expiration

```python
def test_ttl_expiration(self):
    import time

    # Set with 1 second TTL
    self.set_cache("temp_key", "temp_value", ttl=1)

    # Should exist immediately
    self.assert_cache_hit("temp_key")

    # Wait for expiration
    time.sleep(1.1)

    # Should be expired
    self.assert_cache_miss("temp_key")
```

#### Testing Cache Performance

```python
def test_cache_performance(self):
    # Perform many operations
    for i in range(1000):
        self.set_cache(f"key_{i}", f"value_{i}")
        self.get_cache(f"key_{i}")

    # Assert average operation time < 1ms
    self.assert_cache_performance(max_avg_time=0.001)
```

#### Simulating Cache Load

```python
def test_under_load(self):
    # Simulate 10000 operations with 70% hit rate
    self.simulate_cache_load(
        num_operations=10000,
        hit_probability=0.7
    )

    stats = self.get_cache_stats()
    self.assertGreater(stats['hit_rate'], 0.65)
```

---

## Testing Databases

### DatabaseTestCase

For testing database operations with automatic rollback.

```python
from greenlang.testing import DatabaseTestCase

class TestDatabase(DatabaseTestCase):
    def test_insert_and_query(self):
        with self.db_transaction():
            # Insert data
            self.db.insert("users", {
                "name": "John Doe",
                "email": "john@example.com"
            })

            # Query data
            results, _ = self.execute_query(
                "SELECT * FROM users WHERE name = :name",
                {"name": "John Doe"}
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["email"], "john@example.com")

        # Transaction automatically rolled back
```

#### Testing Record Existence

```python
def test_record_existence(self):
    with self.db_transaction():
        self.db.insert("users", {"name": "Jane", "email": "jane@example.com"})

        # Assert record exists
        self.assert_record_exists("users", {"name": "Jane"})

        # Delete
        self.db.delete("users", {"name": "Jane"})

        # Assert record no longer exists
        self.assert_record_not_exists("users", {"name": "Jane"})
```

#### Testing Query Performance

```python
def test_query_performance(self):
    with self.db_transaction():
        # Create test data
        for i in range(1000):
            self.db.insert("products", {"name": f"Product {i}", "price": i * 10})

        # Query should complete in under 100ms
        self.assert_query_performance(
            "SELECT * FROM products WHERE price > :price",
            max_time=0.1,
            params={"price": 5000}
        )
```

#### Loading Fixtures

```python
def test_with_fixtures(self):
    with self.db_transaction():
        self.load_fixtures({
            'users': [
                {'name': 'John', 'email': 'john@example.com'},
                {'name': 'Jane', 'email': 'jane@example.com'},
            ],
            'products': [
                {'name': 'Product 1', 'price': 100},
                {'name': 'Product 2', 'price': 200},
            ]
        })

        self.assert_record_count('users', expected_count=2)
        self.assert_record_count('products', expected_count=2)
```

---

## Integration Testing

### IntegrationTestCase

For end-to-end integration testing.

```python
from greenlang.testing import IntegrationTestCase

class TestFullStack(IntegrationTestCase):
    def _execute_end_to_end(self, test_data):
        # Your end-to-end logic here
        app = MyApplication()
        result = app.run(test_data)
        return result

    def test_complete_workflow(self):
        test_data = {"input": "test"}

        result = self.run_end_to_end_test(
            test_data,
            expected_output={"status": "success"}
        )

        self.assert_integration_success(result)
```

#### Testing with Docker Services

```python
def test_with_docker(self):
    if not self.docker_available:
        self.skipTest("Docker not available")

    with self.docker_services("docker-compose.test.yml"):
        # Services running
        result = self.run_end_to_end_test(self.test_data)
        self.assert_integration_success(result)
        # Services stopped automatically
```

#### Testing with Environment Variables

```python
def test_with_environment(self):
    env = {
        "API_KEY": "test_key",
        "DEBUG": "true",
        "DATABASE_URL": "postgresql://test:test@localhost/test"
    }

    with self.temporary_environment(env):
        result = self.run_end_to_end_test(self.test_data)
        self.assert_integration_success(result)
```

#### Testing Service Availability

```python
def test_services_running(self):
    # Start services
    self.start_service("api", ["python", "api.py"])
    self.start_service("worker", ["python", "worker.py"])

    # Wait for ready
    self.wait_for_services_ready()

    # Assert running
    self.assert_all_services_running()

    # Test
    result = self.run_end_to_end_test(self.test_data)
    self.assert_integration_success(result)
```

---

## Best Practices

### 1. Use Fixtures for Test Data

```python
# Good
def test_with_fixture(self):
    data = self.load_fixture('sample_emissions_data.json')
    result = self.run_agent(self.agent, data)

# Avoid
def test_hardcoded_data(self):
    data = {"field1": "value1", "field2": "value2", ...}  # Long hardcoded data
```

### 2. Test Performance

```python
def test_performance_requirements(self):
    result = self.run_agent(self.agent, self.input_data)

    # Define clear performance requirements
    self.assert_performance(
        result,
        max_time=2.0,  # Must complete in 2 seconds
        max_memory=100 * 1024 * 1024  # Max 100MB
    )
```

### 3. Mock External Dependencies

```python
def test_without_real_llm(self):
    # Don't make real LLM calls in tests
    self.mock_chat.add_response("Mocked response")

    with self.mock_infrastructure():
        result = self.run_agent(self.agent, self.input_data)
```

### 4. Test Error Scenarios

```python
def test_handles_invalid_input(self):
    invalid_input = {}

    with self.assertRaises(ValidationError):
        self.run_agent(self.agent, invalid_input)

def test_handles_llm_failure(self):
    # Simulate LLM failure
    self.mock_chat.add_response(None)  # No response

    with self.assertRaises(LLMError):
        self.run_agent(self.agent, self.input_data)
```

### 5. Use Transactions for Database Tests

```python
def test_database_operations(self):
    # Always use transactions for automatic rollback
    with self.db_transaction():
        # Your test code
        # Changes rolled back automatically
        pass
```

### 6. Test Determinism

```python
def test_consistent_results(self):
    # Agent should produce same results for same input
    self.assert_deterministic(
        self.agent,
        self.input_data,
        runs=5
    )
```

### 7. Organize Tests Logically

```
tests/
├── unit/
│   ├── test_agents/
│   │   ├── test_emissions_agent.py
│   │   └── test_supplier_agent.py
│   ├── test_llm/
│   │   └── test_llm_integration.py
│   └── test_utils/
├── integration/
│   ├── test_pipelines/
│   │   └── test_emissions_pipeline.py
│   └── test_end_to_end/
│       └── test_full_workflow.py
└── fixtures/
    ├── sample_data.json
    └── test_config.yaml
```

---

## Advanced Topics

### Custom Assertions

```python
from greenlang.testing.assertions import (
    assert_no_hallucination,
    assert_response_contains,
    assert_numeric_range
)

def test_no_hallucination(self):
    source = "Steel production emits 2.5 kg CO2e per kg"
    response = my_llm_function(source)

    assert_no_hallucination(response, source, min_similarity=0.7)

def test_response_content(self):
    response = my_function()

    assert_response_contains(
        response,
        ["emissions", "CO2", "calculation"],
        case_sensitive=False
    )

def test_numeric_validation(self):
    emissions = calculate_emissions(data)

    assert_numeric_range(
        emissions,
        min_value=0,
        max_value=10000,
        inclusive=True
    )
```

### Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("quantity,expected", [
    (100, 250),
    (500, 1250),
    (1000, 2500),
])
def test_emissions_calculation(self, quantity, expected):
    result = self.calculate_emissions(quantity)
    self.assertEqual(result, expected)
```

### Testing Async Agents

```python
import asyncio

class TestAsyncAgent(AgentTestCase):
    def test_async_agent(self):
        async def run_test():
            result = await self.agent.run_async(self.input_data)
            return result

        result = asyncio.run(run_test())
        self.assert_success(result)
```

### Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

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
        pip install -e .
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=greenlang --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Summary

The GreenLang Testing Framework provides everything you need to comprehensively test your infrastructure-based applications:

- **AgentTestCase**: Test individual agents
- **PipelineTestCase**: Test agent pipelines
- **LLMTestCase**: Test LLM integrations
- **CacheTestCase**: Test caching mechanisms
- **DatabaseTestCase**: Test database operations
- **IntegrationTestCase**: Test end-to-end workflows

Use fixtures, mocks, and custom assertions to write robust, maintainable tests that ensure your GreenLang applications work correctly in all scenarios.

For more examples, see the templates in `greenlang/testing/templates/`.
