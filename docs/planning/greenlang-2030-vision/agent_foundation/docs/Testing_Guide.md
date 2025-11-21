# Testing Guide

## Comprehensive Testing for GreenLang Agents

Complete guide to testing agents with unit, integration, and end-to-end testing strategies.

---

## Testing Philosophy

- **Test Early, Test Often**: Write tests alongside code
- **Coverage Targets**: 90%+ for core components
- **Test Pyramid**: Many unit tests, fewer integration tests, some E2E tests
- **Fast Feedback**: Tests should run quickly

---

## Testing Framework

### Setup

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Run tests
pytest tests/

# With coverage
pytest --cov=greenlang --cov-report=html tests/
```

### Test Structure

```
tests/
├── unit/
│   ├── test_base_agent.py
│   ├── test_memory.py
│   └── test_intelligence.py
├── integration/
│   ├── test_workflows.py
│   └── test_database.py
├── e2e/
│   └── test_complete_scenarios.py
├── fixtures/
│   └── test_data.py
└── conftest.py
```

---

## Unit Testing

### Testing Base Agent

```python
import pytest
from greenlang import BaseAgent, AgentConfig

class TestBaseAgent:
    """Unit tests for BaseAgent."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return AgentConfig(
            name="test-agent",
            version="1.0.0",
            timeout=30
        )

    @pytest.fixture
    async def agent(self, config):
        """Create test agent."""
        agent = BaseAgent(config)
        await agent.initialize()
        yield agent
        await agent.stop()

    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state == AgentState.READY
        assert agent.name == "test-agent"

    @pytest.mark.asyncio
    async def test_lifecycle(self, agent):
        """Test lifecycle transitions."""
        await agent.start()
        assert agent.state == AgentState.RUNNING

        await agent.pause()
        assert agent.state == AgentState.PAUSED

        await agent.resume()
        assert agent.state == AgentState.RUNNING

        await agent.stop()
        assert agent.state == AgentState.TERMINATED

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling."""
        with pytest.raises(ValidationError):
            await agent.process({'invalid': 'data'})
```

### Testing Memory Systems

```python
class TestMemorySystem:
    """Test memory operations."""

    @pytest.fixture
    async def memory(self):
        """Create test memory."""
        return MemoryManager(MemoryConfig())

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, memory):
        """Test basic store/retrieve."""
        key = await memory.store('test_key', {'data': 'value'})
        retrieved = await memory.retrieve('test_key')
        assert retrieved['data'] == 'value'

    @pytest.mark.asyncio
    async def test_search(self, memory):
        """Test memory search."""
        await memory.store('key1', {'content': 'carbon emissions'})
        await memory.store('key2', {'content': 'solar energy'})

        results = await memory.search('carbon', top_k=1)
        assert len(results) == 1
        assert 'carbon' in results[0]['content']

    @pytest.mark.asyncio
    async def test_consolidation(self, memory):
        """Test memory consolidation."""
        # Add items to short-term
        for i in range(10):
            await memory.short_term.store(f'key{i}', f'value{i}')

        # Consolidate
        stats = await memory.consolidate()
        assert stats['consolidated'] > 0
```

### Testing with Mocks

```python
from unittest.mock import AsyncMock, Mock, patch

class TestAgentWithMocks:
    """Test using mocks."""

    @pytest.fixture
    def agent(self):
        """Create agent with mocked dependencies."""
        agent = MyAgent(test_config)
        agent.llm = AsyncMock()
        agent.db = AsyncMock()
        agent.memory = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_llm_interaction(self, agent):
        """Test LLM interaction."""
        agent.llm.generate.return_value = "test response"

        result = await agent.process({'query': 'test'})

        agent.llm.generate.assert_called_once()
        assert result['response'] == "test response"

    @pytest.mark.asyncio
    async def test_database_interaction(self, agent):
        """Test database interaction."""
        agent.db.query.return_value = [{'id': 1, 'name': 'test'}]

        result = await agent.fetch_data('test')

        agent.db.query.assert_called_once_with("SELECT * FROM test")
        assert len(result) == 1
```

---

## Integration Testing

### Testing Multi-Agent Workflows

```python
@pytest.mark.integration
class TestWorkflow:
    """Integration tests for workflows."""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator with real agents."""
        orch = AgentOrchestrator()

        orch.register(DataCollectorAgent(config1))
        orch.register(ProcessorAgent(config2))
        orch.register(ReporterAgent(config3))

        yield orch

        await orch.shutdown()

    @pytest.mark.asyncio
    async def test_complete_workflow(self, orchestrator):
        """Test complete workflow execution."""
        workflow = {
            'steps': [
                {'agent': 'collector', 'action': 'collect'},
                {'agent': 'processor', 'action': 'process'},
                {'agent': 'reporter', 'action': 'report'}
            ]
        }

        result = await orchestrator.execute_workflow(
            workflow,
            {'source': 'test_data'}
        )

        assert result['status'] == 'completed'
        assert len(result['steps']) == 3
        assert result['report'] is not None
```

### Testing Database Integration

```python
@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database operations."""

    @pytest.fixture
    async def db(self):
        """Create test database."""
        db = Database(test_db_url)
        await db.create_tables()
        yield db
        await db.drop_tables()

    @pytest.mark.asyncio
    async def test_crud_operations(self, db):
        """Test CRUD operations."""
        # Create
        agent_id = await db.create_agent({'name': 'test'})
        assert agent_id is not None

        # Read
        agent = await db.get_agent(agent_id)
        assert agent['name'] == 'test'

        # Update
        await db.update_agent(agent_id, {'name': 'updated'})
        agent = await db.get_agent(agent_id)
        assert agent['name'] == 'updated'

        # Delete
        await db.delete_agent(agent_id)
        agent = await db.get_agent(agent_id)
        assert agent is None
```

---

## End-to-End Testing

### Complete Scenario Testing

```python
@pytest.mark.e2e
class TestCompleteScenarios:
    """End-to-end tests."""

    @pytest.mark.asyncio
    async def test_carbon_calculation_scenario(self):
        """Test complete carbon calculation workflow."""
        # Setup
        agent = CarbonCalculatorAgent(production_config)
        await agent.initialize()

        # Execute complete scenario
        input_data = {
            'company': 'Test Corp',
            'activities': [
                {'type': 'electricity', 'amount': 1000, 'unit': 'kWh'},
                {'type': 'natural_gas', 'amount': 500, 'unit': 'therms'}
            ]
        }

        result = await agent.calculate_emissions(input_data)

        # Verify
        assert result['total_emissions'] > 0
        assert 'scope_1' in result
        assert 'scope_2' in result
        assert result['report_generated'] == True

        # Cleanup
        await agent.stop()

    @pytest.mark.asyncio
    async def test_compliance_reporting_scenario(self):
        """Test compliance reporting workflow."""
        agent = ComplianceAgent(production_config)

        # Full workflow
        result = await agent.generate_report({
            'regulation': 'CSRD',
            'company_data': test_company_data,
            'year': 2024
        })

        assert result['compliant'] == True
        assert result['report_url'] is not None
        assert result['audit_trail'] is not None
```

---

## Performance Testing

### Load Testing

```python
import time
import asyncio

class TestPerformance:
    """Performance tests."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test agent throughput."""
        agent = HighPerformanceAgent(config)

        start = time.time()
        tasks = [agent.process({'id': i}) for i in range(1000)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start

        throughput = len(results) / duration
        assert throughput > 100  # >100 requests/second

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_latency(self):
        """Test response latency."""
        agent = LowLatencyAgent(config)

        latencies = []
        for _ in range(100):
            start = time.time()
            await agent.process({'test': 'data'})
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        p50 = sorted(latencies)[50]
        p95 = sorted(latencies)[95]
        p99 = sorted(latencies)[99]

        assert p50 < 100  # <100ms at p50
        assert p95 < 500  # <500ms at p95
        assert p99 < 2000  # <2s at p99
```

---

## Test Fixtures

### Shared Fixtures

```python
# conftest.py
import pytest
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Standard test configuration."""
    return AgentConfig(
        name="test-agent",
        version="1.0.0",
        timeout=30,
        memory_enabled=False  # Disable for speed
    )

@pytest.fixture
async def test_db():
    """Test database connection."""
    db = Database(test_db_url)
    await db.connect()
    yield db
    await db.disconnect()

@pytest.fixture
def mock_llm():
    """Mock LLM client."""
    llm = AsyncMock()
    llm.generate.return_value = "test response"
    llm.embed.return_value = [0.1] * 768
    return llm
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest --cov=greenlang --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Best Practices

### Do's
✅ Write tests before code (TDD)
✅ Test edge cases
✅ Use descriptive test names
✅ Keep tests independent
✅ Use fixtures for setup
✅ Mock external dependencies
✅ Test error paths
✅ Maintain test data

### Don'ts
❌ Don't skip tests
❌ Don't test implementation details
❌ Don't have flaky tests
❌ Don't hardcode test data
❌ Don't commit failing tests
❌ Don't have slow tests
❌ Don't ignore warnings

---

**Last Updated**: November 2024
**Version**: 1.0.0