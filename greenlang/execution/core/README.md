# GreenLang Core - Composability Framework

## Overview

The GreenLang Composability Framework (GLEL - GreenLang Expression Language) provides a powerful, intuitive way to compose agent pipelines similar to LangChain's LCEL, but optimized for GreenLang's zero-hallucination and provenance tracking requirements.

## Key Features

### 1. Pipe Operator (`|`) for Intuitive Chaining
```python
chain = intake_agent | validation_agent | calculation_agent | reporting_agent
result = chain.invoke(input_data)
```

### 2. Parallel Execution
```python
parallel = RunnableParallel({
    "emissions": emissions_agent,
    "compliance": compliance_agent,
    "risk": risk_agent
})
results = await parallel.ainvoke(input_data)
```

### 3. Zero-Hallucination Guarantees
```python
# Wrap calculation agents to ensure deterministic processing
safe_calc = ZeroHallucinationWrapper(
    AgentRunnable(calculation_agent),
    validation_rules=[validate_numeric_inputs]
)
```

### 4. Complete Provenance Tracking
- SHA-256 hashing at every step
- Full audit trail with parent-child relationships
- Processing time tracking
- Metadata support

### 5. Production-Ready Error Handling
```python
# Add retry logic
reliable = agent.with_retry(max_retries=3, delay_ms=1000)

# Add fallback
resilient = primary.with_fallback(backup_agent)
```

## File Structure

```
greenlang/core/
├── composability.py      # Main framework implementation
└── README.md            # This file

examples/
└── composability_examples.py  # Comprehensive examples

tests/
└── test_composability.py      # Complete test suite

docs/
└── composability_guide.md     # Developer documentation
```

## Quick Start

### 1. Import the Framework
```python
from greenlang.core.composability import (
    AgentRunnable,
    RunnableSequence,
    RunnableParallel,
    RunnableConfig
)
```

### 2. Wrap Your Agents
```python
# Any GreenLang agent can be made composable
runnable_intake = AgentRunnable(your_intake_agent)
runnable_calc = AgentRunnable(your_calculation_agent)
```

### 3. Create Chains
```python
# Sequential processing
chain = runnable_intake | runnable_calc

# Parallel processing
parallel = RunnableParallel({
    "path1": chain1,
    "path2": chain2
})
```

### 4. Execute
```python
# Synchronous
result = chain.invoke(input_data)

# Asynchronous
result = await chain.ainvoke(input_data)

# Streaming
async for chunk in chain.astream(input_data):
    process(chunk)
```

## Core Components

### BaseRunnable
Abstract base class for all runnable components.

### AgentRunnable
Wrapper that makes any GreenLang agent composable.

### RunnableSequence
Sequential execution of multiple runnables (created with `|` operator).

### RunnableParallel
Parallel execution of multiple runnables.

### RetryRunnable
Automatic retry logic for handling transient failures.

### FallbackRunnable
Fallback to alternative runnable on failure.

### RunnableLambda
Wrap simple functions as runnables.

### RunnableBranch
Conditional routing based on input data.

### ZeroHallucinationWrapper
Ensures deterministic calculations without LLM involvement.

## Advanced Patterns

### Map-Reduce
```python
# Process multiple items in parallel, then aggregate
map_reduce = create_map_reduce_chain(
    mapper=item_processor,
    reducer=aggregator
)
result = await map_reduce.ainvoke(items)
```

### Conditional Branching
```python
branch = RunnableBranch(
    branches=[
        (is_large_company, large_company_chain),
        (is_medium_company, medium_company_chain)
    ],
    default=small_company_chain
)
```

### Complex Pipeline
```python
pipeline = (
    intake.with_retry(3) |
    validation |
    safe_calculation |
    parallel_assessments |
    reporting
)
```

## Zero-Hallucination Principles

### Allowed (Deterministic)
- Database lookups for emission factors
- Python arithmetic calculations
- Formula evaluation from YAML/JSON
- Pandas/NumPy aggregations

### Not Allowed (Hallucination Risk)
- LLM calls for numeric calculations
- ML predictions for compliance values
- Unvalidated external API calls

### LLM Usage (Allowed for Non-Numeric)
- Classification and categorization
- Entity resolution and matching
- Narrative generation
- Materiality assessments

## Configuration

```python
config = RunnableConfig(
    max_retries=3,              # Retry attempts
    retry_delay_ms=1000,        # Delay between retries
    timeout_seconds=30.0,       # Execution timeout
    batch_size=100,             # Batch processing size
    enable_streaming=False,     # Enable streaming mode
    enable_provenance=True,     # Track provenance
    parallel_workers=4,         # Parallel worker count
    metadata={}                 # Custom metadata
)

result = chain.invoke(input_data, config)
```

## Testing

Run the test suite:
```bash
python -m pytest tests/test_composability.py -v
```

Run the examples:
```bash
python examples/composability_examples.py
```

## Performance Considerations

1. **Use Async for I/O**: Leverage async methods for database and API calls
2. **Batch Processing**: Process multiple items efficiently with `batch()`
3. **Parallel Execution**: Use `RunnableParallel` for independent operations
4. **Streaming**: Stream large results to avoid memory issues
5. **Caching**: Implement caching in agents for expensive operations

## Migration from Traditional Chains

### Before (Manual Chaining)
```python
try:
    result1 = agent1.process(input)
    if not result1:
        raise Error()
    result2 = agent2.process(result1)
    if not result2:
        raise Error()
    result3 = agent3.process(result2)
except Exception as e:
    # Handle errors
    pass
```

### After (GLEL)
```python
chain = (
    AgentRunnable(agent1) |
    AgentRunnable(agent2) |
    AgentRunnable(agent3)
).with_retry(3)

result = chain.invoke(input)
```

## Best Practices

1. **Always use type hints** for clarity and IDE support
2. **Enable provenance** in production environments
3. **Add retry logic** to external service calls
4. **Use parallel execution** for independent operations
5. **Validate early** in the chain to fail fast
6. **Wrap calculations** with ZeroHallucinationWrapper
7. **Stream large results** to manage memory
8. **Batch multiple inputs** for efficiency

## Support

For questions or issues:
1. Review examples in `examples/composability_examples.py`
2. Check test cases in `tests/test_composability.py`
3. Consult the developer guide in `docs/composability_guide.md`

## License

Part of the GreenLang framework. All rights reserved.