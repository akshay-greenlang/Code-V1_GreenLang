# GreenLang Composability Framework (GLEL)

## GreenLang Expression Language Developer Guide

### Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Installation & Setup](#installation--setup)
4. [Basic Usage](#basic-usage)
5. [Advanced Patterns](#advanced-patterns)
6. [Zero-Hallucination Guarantees](#zero-hallucination-guarantees)
7. [Provenance Tracking](#provenance-tracking)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)
10. [API Reference](#api-reference)
11. [Migration Guide](#migration-guide)
12. [Best Practices](#best-practices)

---

## Introduction

The GreenLang Expression Language (GLEL) is a composability framework inspired by LangChain's LCEL, but optimized for GreenLang's specific requirements:

- **Zero-hallucination guarantees** for regulatory calculations
- **Complete provenance tracking** with SHA-256 hashing
- **Intuitive pipe operator** (`|`) for chaining agents
- **Parallel and sequential execution** patterns
- **Async/streaming support** with backpressure
- **Comprehensive error handling** with retry logic

### Key Benefits

1. **Developer-Friendly**: Use the pipe operator to chain agents naturally
2. **Type-Safe**: Full type hints and Pydantic validation throughout
3. **Production-Ready**: Built-in retry, fallback, and error handling
4. **Audit-Compliant**: Complete provenance tracking for every calculation
5. **Performance-Optimized**: Async support, parallel execution, and batching

---

## Core Concepts

### Runnables

A **Runnable** is the basic building block of GLEL. Every GreenLang agent can be wrapped as a Runnable:

```python
from greenlang.core.composability import AgentRunnable

# Wrap any GreenLang agent
runnable = AgentRunnable(your_agent)
```

### The Pipe Operator

Chain runnables together using the pipe operator (`|`):

```python
# Create a processing pipeline
chain = intake_agent | validation_agent | calculation_agent | reporting_agent

# Execute the chain
result = chain.invoke(input_data)
```

### Parallel Execution

Run multiple agents in parallel:

```python
from greenlang.core.composability import RunnableParallel

parallel = RunnableParallel({
    "emissions": emissions_agent,
    "compliance": compliance_agent,
    "risk": risk_agent
})

results = await parallel.ainvoke(input_data)
```

---

## Installation & Setup

```python
# Import the composability framework
from greenlang.core.composability import (
    AgentRunnable,
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    RetryRunnable,
    FallbackRunnable,
    ZeroHallucinationWrapper,
    RunnableConfig,
    create_sequential_chain,
    create_parallel_chain,
    create_map_reduce_chain
)
```

---

## Basic Usage

### Simple Sequential Chain

```python
# Create your agents
intake = IntakeAgent()
validation = ValidationAgent()
calculation = CalculationAgent()
reporting = ReportingAgent()

# Method 1: Using pipe operator
chain = (
    AgentRunnable(intake) |
    AgentRunnable(validation) |
    AgentRunnable(calculation) |
    AgentRunnable(reporting)
)

# Method 2: Using utility function
chain = create_sequential_chain(intake, validation, calculation, reporting)

# Execute the chain
result = chain.invoke({"company_name": "GreenCorp", "period": "2024-Q1"})
```

### Parallel Processing

```python
# Process multiple aspects in parallel
parallel = RunnableParallel({
    "emissions": AgentRunnable(emissions_agent),
    "water": AgentRunnable(water_agent),
    "waste": AgentRunnable(waste_agent)
})

# Execute all branches simultaneously
results = await parallel.ainvoke(activity_data)

# Access individual results
emissions_result = results["emissions"]
water_result = results["water"]
waste_result = results["waste"]
```

### Adding Retry Logic

```python
# Add retry logic to any agent
reliable_agent = AgentRunnable(unreliable_agent).with_retry(
    max_retries=3,
    delay_ms=1000
)

# Use in a chain
chain = intake_agent | reliable_agent | reporting_agent
```

### Fallback Patterns

```python
# Use fallback data source if primary fails
primary_source = AgentRunnable(primary_database_agent)
fallback_source = AgentRunnable(cached_data_agent)

data_source = FallbackRunnable(primary_source, fallback_source)

# Will automatically use fallback if primary fails
chain = intake_agent | data_source | calculation_agent
```

---

## Advanced Patterns

### Conditional Branching

```python
def is_large_company(data):
    return data.get("employees", 0) > 250

def is_medium_company(data):
    return 50 < data.get("employees", 0) <= 250

# Different processing paths based on company size
branch = RunnableBranch(
    branches=[
        (is_large_company, large_company_pipeline),
        (is_medium_company, medium_company_pipeline)
    ],
    default=small_company_pipeline
)

result = branch.invoke(company_data)
```

### Map-Reduce Pattern

```python
# Process multiple facilities and aggregate results
facility_processor = (
    AgentRunnable(intake_agent) |
    AgentRunnable(calculation_agent)
)

aggregator = AgentRunnable(aggregation_agent)

# Create map-reduce chain
map_reduce = create_map_reduce_chain(
    mapper=facility_processor,
    reducer=aggregator
)

# Process all facilities in parallel, then aggregate
facilities = [facility1, facility2, facility3]
aggregated_result = await map_reduce.ainvoke(facilities)
```

### Lambda Functions in Chains

```python
def add_metadata(data):
    return {**data, "processed_at": datetime.now().isoformat()}

def calculate_intensity(data):
    emissions = data.get("emissions", 0)
    revenue = data.get("revenue", 1)
    return {**data, "intensity": emissions / revenue}

# Use lambda functions for simple transformations
chain = (
    AgentRunnable(intake_agent) |
    RunnableLambda(add_metadata) |
    AgentRunnable(calculation_agent) |
    RunnableLambda(calculate_intensity) |
    AgentRunnable(reporting_agent)
)
```

### Streaming Results

```python
# Stream results from each step
async for chunk in chain.astream(input_data):
    print(f"Step: {chunk['step']}")
    print(f"Output: {chunk['output']}")
    # Process intermediate results as they arrive
```

---

## Zero-Hallucination Guarantees

### The Zero-Hallucination Wrapper

Ensure calculations are deterministic and never use LLMs for numeric values:

```python
# Define validation rules
def validate_numeric_inputs(data):
    """Ensure all inputs are numeric."""
    for key, value in data.get("activity_data", {}).items():
        if not isinstance(value, (int, float)):
            return False
    return True

def validate_positive_values(data):
    """Ensure all values are non-negative."""
    for value in data.get("activity_data", {}).values():
        if isinstance(value, (int, float)) and value < 0:
            return False
    return True

# Wrap calculation agent with zero-hallucination guarantees
safe_calculation = ZeroHallucinationWrapper(
    AgentRunnable(calculation_agent),
    validation_rules=[validate_numeric_inputs, validate_positive_values]
)

# Use in chain - will validate inputs and ensure deterministic calculations
chain = intake_agent | validation_agent | safe_calculation | reporting_agent
```

### Allowed vs Not Allowed

**ALLOWED (Deterministic):**
```python
# Database lookups
emission_factor = db.lookup_emission_factor(material_id)

# Python arithmetic
emissions = activity_data * emission_factor

# Formula evaluation
result = formula_engine.evaluate(formula_id, inputs)

# Pandas aggregations
total = df.groupby('category')['emissions'].sum()
```

**NOT ALLOWED (Hallucination Risk):**
```python
# LLM for calculations - NEVER DO THIS
emissions = llm.calculate_emissions(activity_data)

# ML predictions for compliance values
value = ml_model.predict(features)

# Unvalidated external APIs
result = external_api.get_value()
```

---

## Provenance Tracking

### Automatic Provenance

Every chain execution automatically tracks provenance:

```python
config = RunnableConfig(enable_provenance=True)

result = chain.invoke(input_data, config)

# Access provenance information
chain_provenance = result['_chain_provenance']
chain_hash = result['_chain_hash']

# Each step has:
# - agent_id: Identifier of the agent
# - input_hash: SHA-256 of input
# - output_hash: SHA-256 of output
# - processing_time_ms: Execution time
# - parent_hash: Link to previous step
```

### Custom Provenance

Add custom provenance metadata:

```python
class CustomAgent:
    def process(self, input_data):
        result = self.calculate(input_data)

        # Add custom provenance
        provenance = {
            "calculation_method": "IPCC_2021_TIER1",
            "data_sources": ["ERP", "IoT_Sensors"],
            "confidence_level": 0.95,
            "timestamp": datetime.now().isoformat()
        }

        return {**result, "_provenance_metadata": provenance}
```

---

## Error Handling

### Retry Strategies

```python
# Simple retry with exponential backoff
chain = (
    AgentRunnable(agent1).with_retry(max_retries=3, delay_ms=1000) |
    AgentRunnable(agent2).with_retry(max_retries=5, delay_ms=2000) |
    AgentRunnable(agent3)
)
```

### Fallback Chains

```python
# Multiple fallback options
primary = AgentRunnable(primary_agent)
secondary = AgentRunnable(secondary_agent)
tertiary = AgentRunnable(tertiary_agent)

# Chain fallbacks
resilient = FallbackRunnable(
    primary,
    FallbackRunnable(secondary, tertiary)
)
```

### Error Recovery in Parallel

```python
parallel = RunnableParallel({
    "critical": AgentRunnable(critical_agent).with_retry(5),
    "optional": FallbackRunnable(
        AgentRunnable(optional_agent),
        RunnableLambda(lambda x: {"status": "skipped"})
    )
})

# Critical branch will retry, optional will fallback to default
results = await parallel.ainvoke(input_data)
```

---

## Performance Optimization

### Async Execution

```python
# All chains support async execution
async def process_data(input_data):
    # Async chain execution
    result = await chain.ainvoke(input_data)

    # Async streaming
    async for chunk in chain.astream(input_data):
        process_chunk(chunk)

    # Async batch processing
    results = await chain.abatch(input_list)

    return result
```

### Batch Processing

```python
config = RunnableConfig(batch_size=100)

# Process large dataset in batches
inputs = [data for data in large_dataset]
results = chain.batch(inputs, config)

# Async batch for better performance
results = await chain.abatch(inputs, config)
```

### Parallel Workers

```python
config = RunnableConfig(parallel_workers=8)

# Use more workers for CPU-intensive tasks
parallel = RunnableParallel(
    {f"worker_{i}": processor for i in range(8)}
)

results = await parallel.ainvoke(input_data, config)
```

---

## API Reference

### Core Classes

#### AgentRunnable
```python
AgentRunnable(agent, name=None)
# Wraps a GreenLang agent as a runnable

Methods:
- invoke(input, config=None) -> output
- ainvoke(input, config=None) -> output
- stream(input, config=None) -> Iterator
- astream(input, config=None) -> AsyncIterator
- batch(inputs, config=None) -> List[output]
- abatch(inputs, config=None) -> List[output]
- with_retry(max_retries, delay_ms) -> RetryRunnable
- with_fallback(fallback) -> FallbackRunnable
- with_config(**kwargs) -> AgentRunnable
```

#### RunnableSequence
```python
RunnableSequence(runnables)
# Sequential execution of runnables

# Create with pipe operator
chain = runnable1 | runnable2 | runnable3
```

#### RunnableParallel
```python
RunnableParallel(runnables: Dict[str, Runnable])
# Parallel execution of runnables

parallel = RunnableParallel({
    "branch1": runnable1,
    "branch2": runnable2
})
```

#### RunnableConfig
```python
RunnableConfig(
    max_retries=3,
    retry_delay_ms=1000,
    timeout_seconds=None,
    batch_size=100,
    enable_streaming=False,
    enable_provenance=True,
    parallel_workers=4,
    error_handler=None,
    metadata={}
)
```

---

## Migration Guide

### From Traditional Agent Chains

**Before (Traditional):**
```python
# Manual chaining with error-prone data passing
intake_result = intake_agent.process(input_data)
if not intake_result:
    raise ValueError("Intake failed")

validation_result = validation_agent.process(intake_result)
if not validation_result.get("valid"):
    raise ValueError("Validation failed")

calculation_result = calculation_agent.process(validation_result)
report = reporting_agent.process(calculation_result)
```

**After (GLEL):**
```python
# Clean, composable chain with automatic error propagation
chain = (
    AgentRunnable(intake_agent) |
    AgentRunnable(validation_agent) |
    AgentRunnable(calculation_agent) |
    AgentRunnable(reporting_agent)
)

report = chain.invoke(input_data)
```

### Adding to Existing Agents

Your existing agents don't need modification:

```python
# Existing agent
class YourExistingAgent:
    def process(self, input_data):
        # Your existing logic
        return result

# Make it composable
composable = AgentRunnable(YourExistingAgent())

# Use in chains
chain = composable | other_agent
```

---

## Best Practices

### 1. Use Type Hints

```python
from typing import Dict, Any

def transform(data: Dict[str, Any]) -> Dict[str, Any]:
    """Always use type hints for clarity."""
    return {**data, "transformed": True}

lambda_runnable = RunnableLambda(transform)
```

### 2. Enable Provenance for Production

```python
# Always enable provenance in production
production_config = RunnableConfig(
    enable_provenance=True,
    metadata={"environment": "production"}
)

result = chain.invoke(input_data, production_config)
```

### 3. Handle Errors Gracefully

```python
# Combine retry and fallback for resilience
critical_chain = (
    AgentRunnable(data_source).with_retry(3).with_fallback(cache_agent) |
    AgentRunnable(processor).with_retry(2) |
    AgentRunnable(reporter)
)
```

### 4. Use Parallel for Independent Operations

```python
# Good: Parallel for independent operations
parallel = RunnableParallel({
    "emissions": emissions_calc,
    "water": water_calc,
    "waste": waste_calc
})

# Bad: Sequential for independent operations
chain = emissions_calc | water_calc | waste_calc  # Slower!
```

### 5. Validate Early

```python
# Validate at the beginning of chains
chain = (
    RunnableLambda(validate_input) |
    AgentRunnable(expensive_calculation) |  # Only runs if validation passes
    AgentRunnable(reporting)
)
```

### 6. Use Zero-Hallucination for Calculations

```python
# Always wrap calculation agents
safe_calc = ZeroHallucinationWrapper(
    AgentRunnable(calculation_agent),
    validation_rules=[validate_numeric, validate_positive]
)
```

### 7. Stream for Large Results

```python
# Stream large results to avoid memory issues
async for chunk in chain.astream(large_input):
    await process_and_store(chunk)
    # Process incrementally
```

### 8. Batch for Multiple Inputs

```python
# Process multiple inputs efficiently
config = RunnableConfig(batch_size=100)
results = await chain.abatch(thousand_inputs, config)
```

---

## Examples Repository

Find complete working examples in:
- `examples/composability_examples.py` - All patterns demonstrated
- `tests/test_composability.py` - Comprehensive test suite

---

## Support

For questions or issues:
1. Check the examples in `examples/composability_examples.py`
2. Review test cases in `tests/test_composability.py`
3. Contact the GreenLang development team

---

## Conclusion

The GreenLang Expression Language (GLEL) provides a powerful, intuitive way to compose agent pipelines while maintaining:
- Zero-hallucination guarantees for regulatory compliance
- Complete provenance tracking for audit trails
- Production-ready error handling and resilience
- High performance with async and parallel execution

Start using GLEL today to build more maintainable, reliable, and auditable agent pipelines!