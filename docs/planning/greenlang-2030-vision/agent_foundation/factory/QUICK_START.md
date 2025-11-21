# GreenLang Agent Factory - Quick Start Guide

## 5-Minute Quick Start

This guide will get you creating production-ready agents in less than 5 minutes.

## Installation

Ensure you have the GreenLang agent foundation installed:

```bash
cd C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation
pip install -r requirements.txt
```

## Quick Create: Calculator Agent

### Option 1: Using Convenience Function (Fastest)

```python
from factory.agent_factory import create_calculator_agent

# Create a simple calculator agent in one function call
result = create_calculator_agent(
    name="CarbonCalculator",
    formulas={"emissions": "activity_data * emission_factor"},
    input_schema={"activity_data": "float", "emission_factor": "float"},
    output_schema={"emissions": "float"}
)

print(f"✓ Agent created in {result.generation_time_ms:.0f}ms")
print(f"  Code: {result.code_path}")
print(f"  Tests: {result.test_path}")
print(f"  Quality: {result.quality_score:.1f}%")
```

**Expected Output:**
```
✓ Agent created in 67ms
  Code: C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\factory\generated_agents\CarbonCalculator\carboncalculator.py
  Tests: C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\factory\generated_agents\CarbonCalculator\test_carboncalculator.py
  Quality: 82.5%
```

### Option 2: Full Control with Specification

```python
from factory.agent_factory import AgentFactory, AgentSpecification
from pathlib import Path

# 1. Initialize factory
factory = AgentFactory()

# 2. Define specification
spec = AgentSpecification(
    name="CarbonCalculator",
    type="calculator",
    description="Calculate carbon emissions from activity data",
    input_schema={
        "activity_data": "float",
        "emission_factor": "float"
    },
    output_schema={
        "emissions": "float"
    },
    calculation_formulas={
        "emissions": "activity_data * emission_factor"
    }
)

# 3. Create agent
result = factory.create_agent(spec)

# 4. Check results
if result.success:
    print(f"✓ Success! Quality: {result.quality_score}%")
    print(f"  Agent ID: {result.agent_id}")
    print(f"  Files generated: {result.lines_of_code} LOC")
else:
    print("✗ Failed:", result.errors)
```

## Quick Create: Stateless Agent

```python
from factory.agent_factory import create_stateless_agent

result = create_stateless_agent(
    name="DataValidator",
    description="Validate input data format and ranges",
    input_schema={"data": "Dict[str, Any]"},
    output_schema={"valid": "bool", "errors": "List[str]"}
)

print(f"✓ Stateless agent created: {result.agent_name}")
```

## Create Multiple Agents in Batch

```python
from factory.agent_factory import AgentFactory, AgentSpecification

factory = AgentFactory()

# Define multiple agents
specs = [
    AgentSpecification(
        name="Scope1Calculator",
        type="calculator",
        description="Scope 1 emissions",
        input_schema={"fuel": "float"},
        output_schema={"emissions": "float"},
        calculation_formulas={"emissions": "fuel * 2.5"}
    ),
    AgentSpecification(
        name="Scope2Calculator",
        type="calculator",
        description="Scope 2 emissions",
        input_schema={"electricity": "float"},
        output_schema={"emissions": "float"},
        calculation_formulas={"emissions": "electricity * 0.5"}
    )
]

# Create all in parallel
results = factory.create_agent_batch(specs, parallel=True)

print(f"✓ Created {len(results)} agents")
```

## Running Generated Agents

After creation, test your generated agent:

```python
import asyncio
from generated_agents.CarbonCalculator.carboncalculator import CarbonCalculator
from base_agent import AgentConfig

async def test_agent():
    # Initialize
    config = AgentConfig(name="CarbonCalculator")
    agent = CarbonCalculator(config)
    await agent.initialize()

    # Execute
    input_data = {
        "activity_data": 100.0,
        "emission_factor": 2.5
    }

    result = await agent.execute(input_data)

    print(f"Result: {result.result}")
    print(f"Provenance: {result.result.get('provenance_hash', 'N/A')}")

    # Cleanup
    await agent.terminate()

# Run
asyncio.run(test_agent())
```

## Run Generated Tests

```bash
# Run all tests for your agent
pytest generated_agents/CarbonCalculator/test_carboncalculator.py -v

# With coverage
pytest generated_agents/CarbonCalculator/test_carboncalculator.py --cov -v
```

## Validate Quality

```python
from factory.validation import AgentValidator
from pathlib import Path

validator = AgentValidator()

result = validator.validate_agent(
    code_path=Path("./generated_agents/CarbonCalculator/carboncalculator.py"),
    test_path=Path("./generated_agents/CarbonCalculator/test_carboncalculator.py")
)

print(f"Quality Score: {result.quality_score}%")
print(f"Test Coverage: {result.test_coverage}%")
print(f"Code Quality: {result.code_quality_score}%")
print(f"Security: {result.security_score}%")
print(f"Deployable: {result.is_valid}")
```

## Create a Pack

```python
from factory.pack_builder import PackBuilder, PackMetadata
from pathlib import Path

builder = PackBuilder()

metadata = PackMetadata(
    name="CarbonPack",
    version="1.0.0",
    description="Carbon calculation agents",
    agent_type="calculator",
    domain="carbon"
)

pack_id = builder.create_pack(
    agent_dir=Path("./generated_agents/CarbonCalculator"),
    metadata=metadata
)

print(f"✓ Pack created: {pack_id}")
```

## Deploy to Kubernetes (Optional)

```python
from factory.deployment import KubernetesDeployer, DeploymentConfig
from pathlib import Path

deployer = KubernetesDeployer()

config = DeploymentConfig(
    name="carbon-calculator",
    replicas=3,
    image="greenlang/carbon-calculator:latest",
    enable_autoscaling=True
)

deployment_id = deployer.deploy(
    Path("./generated_agents/CarbonCalculator"),
    config
)

status = deployer.status(deployment_id)
print(f"✓ Deployed: {status.status}")
print(f"  Replicas: {status.replicas_ready}/{status.replicas_total}")
```

## Check Factory Stats

```python
from factory.agent_factory import AgentFactory

factory = AgentFactory()

# ... create some agents ...

# Get statistics
stats = factory.get_metrics()

print(f"Total Agents: {stats['agents_created']}")
print(f"Average Time: {stats['average_generation_time_ms']:.2f}ms")
print(f"Fastest: {stats['fastest_ms']:.2f}ms")
print(f"Slowest: {stats['slowest_ms']:.2f}ms")
```

## Common Patterns

### Pattern 1: Calculator Agent with Multiple Formulas

```python
spec = AgentSpecification(
    name="MultiCalculator",
    type="calculator",
    description="Multi-metric calculator",
    input_schema={
        "activity": "float",
        "factor": "float",
        "quantity": "float"
    },
    output_schema={
        "total": "float",
        "intensity": "float",
        "average": "float"
    },
    calculation_formulas={
        "total": "activity * factor",
        "intensity": "total / quantity",
        "average": "(activity + quantity) / 2"
    }
)
```

### Pattern 2: Stateful Agent with History

```python
spec = AgentSpecification(
    name="TrendAnalyzer",
    type="stateful",
    description="Analyze trends over time",
    input_schema={"value": "float", "timestamp": "str"},
    output_schema={"trend": "str", "change": "float"}
)
```

### Pattern 3: Batch Processing Pipeline

```python
# Define a pipeline of agents
pipeline_specs = [
    AgentSpecification(name="DataIngestion", type="stateless", ...),
    AgentSpecification(name="DataValidation", type="stateless", ...),
    AgentSpecification(name="DataCalculation", type="calculator", ...),
    AgentSpecification(name="ResultAggregation", type="stateful", ...),
]

results = factory.create_agent_batch(pipeline_specs, parallel=True)
```

## Troubleshooting

### Issue: Generation taking too long

```python
# Enable caching
factory = AgentFactory(cache_templates=True)

# Use parallel execution
factory = AgentFactory(parallel_execution=True, max_workers=8)
```

### Issue: Low quality score

```python
# Check validation details
result = factory.create_agent(spec, validate=True)
if result.validation_result:
    for error in result.validation_result.errors:
        print(f"Error: {error['message']}")
    for warning in result.validation_result.warnings:
        print(f"Warning: {warning['message']}")
```

### Issue: Tests failing

```python
# Generate with comprehensive tests
result = factory.create_agent(
    spec,
    generate_tests=True,
    validate=True
)

# Check test coverage
print(f"Tests: {result.test_count}")
print(f"Coverage: {result.validation_result.test_coverage}%")
```

## Next Steps

1. **Review Generated Code:** Check `generated_agents/{agent_name}/`
2. **Run Tests:** `pytest generated_agents/{agent_name}/test_*.py -v`
3. **Customize:** Edit generated code as needed
4. **Validate:** Run quality validation
5. **Deploy:** Use deployment system or manual deployment

## Examples Directory

Full working examples available in:
```
C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\factory\examples\
├── create_calculator_agent.py     # Single agent creation
└── batch_create_agents.py         # Batch creation
```

Run examples:
```bash
cd C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\factory\examples
python create_calculator_agent.py
python batch_create_agents.py
```

## Documentation

- **Full Documentation:** `factory/README.md`
- **Implementation Summary:** `factory/IMPLEMENTATION_SUMMARY.md`
- **Architecture:** `Agent_Foundation_Architecture.md`

## Support

For issues or questions:
1. Check documentation in `README.md`
2. Review examples in `examples/`
3. Check validation results for specific errors
4. Review generated code for customization needs

---

**Quick Start Complete!** You should now be able to create production-ready agents in <5 minutes.

**Version:** 1.0.0
**Status:** Production Ready
