# Best Practices

## Development Best Practices for GreenLang Agents

Production-proven patterns and practices for building robust, scalable agents.

---

## Architecture Best Practices

### 1. Agent Design Principles

**Single Responsibility**
```python
# Good: Each agent has one clear purpose
class CarbonCalculatorAgent(BaseAgent):
    """Calculate carbon emissions only."""

class DataValidatorAgent(BaseAgent):
    """Validate data only."""

# Bad: Agent does too much
class SuperAgent(BaseAgent):
    """Calculate, validate, report, analyze..."""
```

**Composition Over Inheritance**
```python
# Good: Compose capabilities
class MyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.calculator = Calculator()
        self.validator = Validator()
        self.reporter = Reporter()

# Bad: Deep inheritance hierarchy
class MyAgent(ReportingAgent, CalculationAgent, ValidationAgent):
    pass
```

---

## Performance Best Practices

### 1. Use Async Operations

```python
# Good: Async for I/O
async def process(self, data):
    results = await asyncio.gather(
        self.fetch_from_db(data),
        self.call_api(data),
        self.generate_llm_response(data)
    )
    return self.combine_results(results)

# Bad: Synchronous blocking
def process(self, data):
    db_result = self.fetch_from_db(data)  # Blocks
    api_result = self.call_api(data)  # Blocks
    llm_result = self.generate_llm_response(data)  # Blocks
    return self.combine_results([db_result, api_result, llm_result])
```

### 2. Implement Caching

```python
from functools import lru_cache

class OptimizedAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.cache = LRUCache(maxsize=1000)

    @lru_cache(maxsize=100)
    def expensive_computation(self, key):
        """Cached computation."""
        return complex_calculation(key)

    async def get_with_cache(self, key):
        """Redis-backed cache."""
        if cached := await self.redis.get(key):
            return cached

        result = await self.compute(key)
        await self.redis.set(key, result, ex=3600)
        return result
```

### 3. Batch Operations

```python
# Good: Batch processing
async def process_batch(self, items):
    embeddings = await self.llm.embed_batch(
        [item['text'] for item in items]
    )
    return await self.db.insert_many(
        [{'item': item, 'embedding': emb}
         for item, emb in zip(items, embeddings)]
    )

# Bad: One at a time
async def process_items(self, items):
    for item in items:
        embedding = await self.llm.embed(item['text'])
        await self.db.insert({'item': item, 'embedding': embedding})
```

---

## Error Handling Best Practices

### 1. Specific Error Handling

```python
# Good: Specific exception handling
try:
    result = await self.process(data)
except ValidationError as e:
    self.logger.error(f"Validation failed: {e}")
    return {'error': 'invalid_input', 'details': str(e)}
except LLMError as e:
    self.logger.error(f"LLM call failed: {e}")
    return await self.use_fallback(data)
except DatabaseError as e:
    self.logger.critical(f"Database error: {e}")
    await self.alert_ops_team(e)
    raise

# Bad: Catch-all
try:
    result = await self.process(data)
except Exception as e:
    self.logger.error("Something went wrong")
    return None
```

### 2. Graceful Degradation

```python
async def process_with_fallback(self, data):
    """Process with multiple fallback levels."""
    try:
        # Try primary method
        return await self.primary_process(data)
    except PrimaryError:
        try:
            # Fall back to secondary
            return await self.secondary_process(data)
        except SecondaryError:
            # Last resort: basic processing
            return await self.basic_process(data)
```

---

## Security Best Practices

### 1. Input Validation

```python
from pydantic import BaseModel, validator

class InputData(BaseModel):
    """Validated input schema."""
    user_id: str
    amount: float
    description: str

    @validator('amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount must be positive')
        return v

    @validator('description')
    def validate_description(cls, v):
        # Sanitize input
        return v.strip()[:1000]

async def process(self, data: dict):
    """Process with validation."""
    validated = InputData(**data)  # Raises ValidationError if invalid
    return await self._process_validated(validated)
```

### 2. Secrets Management

```python
# Good: Environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ConfigurationError("OPENAI_API_KEY not set")

# Good: External secrets management
from azure.keyvault.secrets import SecretClient
secret_client = SecretClient(vault_url, credential)
api_key = secret_client.get_secret("openai-api-key").value

# Bad: Hardcoded secrets
api_key = "sk-xxxxxxxxxxxxx"  # Never do this!
```

### 3. Rate Limiting

```python
from aiolimiter import AsyncLimiter

class RateLimitedAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # 100 requests per minute
        self.limiter = AsyncLimiter(100, 60)

    async def process(self, data):
        async with self.limiter:
            return await self._process(data)
```

---

## Testing Best Practices

### 1. Comprehensive Test Coverage

```python
import pytest

class TestMyAgent:
    """Test suite for MyAgent."""

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        config = AgentConfig(name="test-agent")
        return MyAgent(config)

    @pytest.mark.asyncio
    async def test_valid_input(self, agent):
        """Test with valid input."""
        result = await agent.process({'data': 'valid'})
        assert result['status'] == 'success'

    @pytest.mark.asyncio
    async def test_invalid_input(self, agent):
        """Test error handling."""
        with pytest.raises(ValidationError):
            await agent.process({'invalid': 'data'})

    @pytest.mark.asyncio
    async def test_timeout(self, agent):
        """Test timeout handling."""
        with pytest.raises(TimeoutError):
            await agent.process_with_timeout({'data': 'slow'}, timeout=1)

    @pytest.mark.integration
    async def test_end_to_end(self, agent):
        """Integration test."""
        result = await agent.full_process({'data': 'complete'})
        assert result['completed'] == True
```

### 2. Mock External Dependencies

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mocks(agent):
    """Test with mocked dependencies."""
    # Mock LLM
    agent.llm = AsyncMock()
    agent.llm.generate.return_value = "mocked response"

    # Mock database
    agent.db = AsyncMock()
    agent.db.query.return_value = [{'id': 1}]

    result = await agent.process({'test': 'data'})

    # Verify mocks were called
    agent.llm.generate.assert_called_once()
    agent.db.query.assert_called_once()
```

---

## Observability Best Practices

### 1. Structured Logging

```python
import structlog

class ObservableAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.logger = structlog.get_logger(
            agent_id=self.id,
            agent_name=self.name
        )

    async def process(self, data):
        self.logger.info(
            "processing_started",
            input_size=len(data),
            timestamp=datetime.utcnow()
        )

        try:
            result = await self._process(data)
            self.logger.info(
                "processing_completed",
                output_size=len(result),
                duration_ms=self.get_duration()
            )
            return result

        except Exception as e:
            self.logger.error(
                "processing_failed",
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise
```

### 2. Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

class InstrumentedAgent(BaseAgent):
    # Define metrics
    requests_total = Counter(
        'agent_requests_total',
        'Total requests',
        ['agent_name', 'status']
    )

    request_duration = Histogram(
        'agent_request_duration_seconds',
        'Request duration',
        ['agent_name']
    )

    active_requests = Gauge(
        'agent_active_requests',
        'Active requests',
        ['agent_name']
    )

    async def process(self, data):
        self.active_requests.labels(agent_name=self.name).inc()

        try:
            with self.request_duration.labels(agent_name=self.name).time():
                result = await self._process(data)

            self.requests_total.labels(
                agent_name=self.name,
                status='success'
            ).inc()

            return result

        except Exception as e:
            self.requests_total.labels(
                agent_name=self.name,
                status='error'
            ).inc()
            raise

        finally:
            self.active_requests.labels(agent_name=self.name).dec()
```

---

## Memory Management Best Practices

### 1. Efficient Memory Usage

```python
# Good: Generator for large datasets
def process_large_file(self, file_path):
    with open(file_path) as f:
        for line in f:
            yield self.process_line(line)

# Bad: Load entire file into memory
def process_large_file(self, file_path):
    with open(file_path) as f:
        lines = f.readlines()  # Loads entire file
        return [self.process_line(line) for line in lines]
```

### 2. Memory Pruning

```python
class MemoryEfficientAgent(BaseAgent):
    async def maintain_memory(self):
        """Periodic memory maintenance."""
        while self.is_running:
            await asyncio.sleep(300)  # Every 5 minutes

            # Clear old cache entries
            await self.cache.clear_expired()

            # Prune low-priority memories
            await self.memory.prune(threshold=0.3)

            # Consolidate memories
            await self.memory.consolidate()
```

---

## Scalability Best Practices

### 1. Stateless Design

```python
# Good: Stateless agent
class StatelessAgent(BaseAgent):
    async def process(self, data):
        # All state passed in data
        # No shared state between requests
        return await self.pure_function(data)

# Bad: Shared mutable state
class StatefulAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.shared_state = {}  # Problematic for scaling

    async def process(self, data):
        self.shared_state[data['id']] = data  # Race conditions!
        return await self.process_with_state(data)
```

### 2. Resource Pooling

```python
class PooledAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # Connection pooling
        self.db_pool = create_pool(
            min_size=5,
            max_size=20
        )

    async def process(self, data):
        async with self.db_pool.acquire() as conn:
            return await conn.execute(query, data)
```

---

## Production Checklist

### Pre-Deployment
- [ ] All tests passing (unit, integration, e2e)
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Alerts set up
- [ ] Rollback plan ready

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics flowing
- [ ] Logs aggregating
- [ ] No error spikes
- [ ] Performance acceptable
- [ ] User feedback positive

---

## Do's and Don'ts

### Do's
✅ Use async/await for I/O operations
✅ Implement comprehensive error handling
✅ Add logging and metrics
✅ Write tests for all functionality
✅ Validate all inputs
✅ Use type hints
✅ Document your code
✅ Handle secrets securely
✅ Implement graceful degradation
✅ Monitor performance

### Don'ts
❌ Don't block the event loop
❌ Don't ignore errors
❌ Don't hardcode credentials
❌ Don't skip input validation
❌ Don't create memory leaks
❌ Don't use print() for logging
❌ Don't deploy without tests
❌ Don't skip documentation
❌ Don't ignore security
❌ Don't premature optimize

---

## Additional Resources

- [Agent Development Guide](Agent_Development_Guide.md)
- [Testing Guide](Testing_Guide.md)
- [Deployment Guide](Deployment_Guide.md)
- [Security Guidelines](Security_Guide.md)

---

**Last Updated**: November 2024
**Version**: 1.0.0
**Maintainer**: GreenLang Best Practices Team