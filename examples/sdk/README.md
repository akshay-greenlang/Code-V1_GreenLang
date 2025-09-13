# GreenLang SDK Examples

This directory contains comprehensive examples demonstrating how to use the GreenLang SDK programmatically for building, executing, and managing carbon accounting workflows.

## Overview

The SDK examples cover three main areas:

1. **Basic Usage** (`basic_usage.py`) - Fundamental SDK operations and patterns
2. **Pipeline Creation** (`pipeline_creation.py`) - Building pipelines programmatically
3. **Pack Loading** (`pack_loading.py`) - Managing and executing packs through code

## Examples

### 1. Basic Usage Example (`basic_usage.py`)

Demonstrates fundamental GreenLang SDK usage patterns:

#### **Key Features**
- Agent configuration and registration
- Workflow creation and execution
- Input validation and data processing
- Result handling and formatting
- Error handling and logging

#### **Usage**
```bash
python examples/sdk/basic_usage.py
```

#### **What It Does**
```python
# Set up agents
carbon_agent = CarbonAgent()
validator = InputValidatorAgent()
report_agent = ReportAgent()

# Create workflow
workflow = Workflow(
    name="basic_carbon_accounting",
    steps=[
        validate_input,
        calculate_scope1,
        calculate_scope2,
        generate_report
    ]
)

# Execute with sample data
results = orchestrator.execute_workflow("carbon_analysis", input_data)
```

#### **Sample Output**
```
======================================================
CARBON FOOTPRINT ANALYSIS RESULTS
======================================================
Facility: Sample Office Building
Total Emissions: 247.50 tons CO2e
  Scope 1 (Direct): 132.65 tons CO2e
  Scope 2 (Electricity): 114.85 tons CO2e
Emissions Intensity: 9.90 lbs CO2e/sqft
Data Quality: high
Analysis Timestamp: 2024-09-13T15:30:45
======================================================
```

### 2. Pipeline Creation Example (`pipeline_creation.py`)

Shows how to build complex pipelines programmatically:

#### **Key Features**
- Builder pattern for pipeline construction
- Dynamic pipeline generation from configuration
- Template-based pipeline creation
- Pipeline modification and enhancement
- Export to YAML for reuse

#### **Usage**
```bash
python examples/sdk/pipeline_creation.py
```

#### **Builder Pattern Example**
```python
pipeline = (PipelineBuilder("facility-analysis", version=1)
    .set_variables({"reporting_year": 2024})
    .add_validation_step("validate-input", "${inputs.facility_data}")
    .add_carbon_calculation_step("calculate-scope1", scope=1)
    .add_carbon_calculation_step("calculate-scope2", scope=2, parallel=True)
    .add_aggregation_step("total-emissions", ["scope1", "scope2"])
    .add_report_generation_step("generate-report", template="summary")
    .build())

# Export to YAML
pipeline.to_yaml("generated_pipeline.yaml")
```

#### **Template Examples**
```python
# Basic facility analysis
basic_pipeline = PipelineTemplates.create_basic_facility_analysis("Office Building")

# Portfolio analysis with 8 facilities
portfolio_pipeline = PipelineTemplates.create_portfolio_analysis("Corporate Portfolio", 8)

# Dynamic pipeline from configuration
dynamic_pipeline = create_dynamic_pipeline_from_config(config_dict)
```

### 3. Pack Loading Example (`pack_loading.py`)

Demonstrates advanced pack management and execution:

#### **Key Features**
- Pack discovery and installation
- Metadata inspection and validation
- Dependency resolution
- Pipeline execution from packs
- Registry management

#### **Usage**
```bash
python examples/sdk/pack_loading.py
```

#### **Pack Management**
```python
# Create registry and add repositories
registry = PackRegistry()
registry.add_repository("examples", "examples/packs")

# Discover available packs
available_packs = registry.discover_packs("examples/packs")

# Install packs
for pack in available_packs:
    registry.install_pack(pack["name"])

# Execute pack pipeline
result = registry.pack_loader.execute_pack_pipeline(
    pack_name="enterprise-carbon-suite",
    pipeline_name="main-analysis.yaml",
    inputs=facility_data
)
```

#### **Pack Inspection**
```python
# Load and inspect pack metadata
metadata = pack_loader.load_pack("examples/packs/advanced")

print(f"Name: {metadata.name}")
print(f"Version: {metadata.version}")
print(f"Dependencies: {metadata.dependencies}")
print(f"Pipelines: {metadata.pipelines}")
print(f"Agents: {metadata.agents}")
```

## Advanced Patterns

### 1. **Error Handling and Resilience**

```python
# Configure error handling
step = PipelineStep(
    name="calculate-emissions",
    agent="CarbonAgent",
    action="calculate_total",
    inputs={"data": "${inputs.facility_data}"},
    error_handling={
        "on_failure": "retry",
        "retry_count": 3,
        "retry_delay": 5,
        "fallback_action": "use_conservative_estimate"
    }
)
```

### 2. **Conditional Execution**

```python
# Add conditional step
renewable_step = PipelineStep(
    name="analyze-renewables",
    agent="SolarOffsetAgent",
    action="calculate_offset",
    inputs={"renewable_kwh": "${inputs.renewable_generation}"},
    conditional={
        "condition": "${inputs.renewable_generation} > 0",
        "on_skip": {"renewable_offset": 0}
    }
)
```

### 3. **Parallel Processing**

```python
# Create parallel processing group
builder.add_parallel_processing_group(
    base_name="process-facility",
    agent="CarbonAgent",
    action="calculate_emissions",
    data_partitions=["${partition_1}", "${partition_2}", "${partition_3}"],
    depends_on=["data-preprocessing"]
)
```

### 4. **Dynamic Configuration**

```python
# Create pipeline from configuration
config = {
    "name": "dynamic-analysis",
    "steps": [
        {"type": "validation", "name": "validate", "data_source": "${inputs.data}"},
        {"type": "carbon_calculation", "name": "calculate", "scope": 1},
        {"type": "report", "name": "report", "template": "summary"}
    ]
}

pipeline = create_dynamic_pipeline_from_config(config)
```

## Integration Examples

### 1. **Web Application Integration**

```python
from flask import Flask, request, jsonify
import greenlang

app = Flask(__name__)
orchestrator = Orchestrator()

@app.route('/analyze', methods=['POST'])
def analyze_facility():
    facility_data = request.json

    # Execute carbon analysis
    results = orchestrator.execute_workflow("carbon_analysis", facility_data)

    return jsonify(results)
```

### 2. **Batch Processing**

```python
def process_facility_batch(facilities: List[Dict]):
    """Process multiple facilities in batch"""
    results = []

    for facility in facilities:
        try:
            result = orchestrator.execute_workflow("facility_analysis", {
                "facility_data": facility
            })
            results.append({"facility_id": facility["id"], "result": result})
        except Exception as e:
            results.append({"facility_id": facility["id"], "error": str(e)})

    return results
```

### 3. **Monitoring and Logging**

```python
import logging
from greenlang import Orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create orchestrator with monitoring
orchestrator = Orchestrator()
orchestrator.add_execution_listener(lambda event: logger.info(f"Step completed: {event}"))

# Execute with monitoring
results = orchestrator.execute_workflow("analysis", data)
```

## Testing Examples

### 1. **Unit Testing Workflows**

```python
import unittest
from greenlang import CarbonAgent, Orchestrator

class TestCarbonWorkflow(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Orchestrator()
        self.orchestrator.register_agent("carbon", CarbonAgent())

    def test_basic_calculation(self):
        input_data = {"electricity_kwh": 1000, "natural_gas_mmbtu": 100}
        result = self.orchestrator.execute_workflow("basic_calc", input_data)

        self.assertIsNotNone(result)
        self.assertIn("total_emissions", result)
        self.assertGreater(result["total_emissions"], 0)
```

### 2. **Integration Testing**

```python
def test_full_pipeline_integration():
    """Test complete pipeline from input to output"""

    # Load pack
    registry = PackRegistry()
    registry.install_pack("test-pack")

    # Execute pipeline
    result = registry.pack_loader.execute_pack_pipeline(
        "test-pack",
        "main-pipeline.yaml",
        test_facility_data
    )

    # Verify results
    assert result["status"] == "completed"
    assert "total_emissions" in result["outputs"]
    assert result["outputs"]["total_emissions"] > 0
```

## Best Practices

### 1. **Error Handling**
- Always configure appropriate error handling for each step
- Use retry mechanisms for transient failures
- Implement fallback strategies for critical calculations
- Log errors with sufficient context for debugging

### 2. **Performance Optimization**
- Use parallel processing for independent calculations
- Implement caching for expensive operations
- Partition large datasets for distributed processing
- Set appropriate timeouts for long-running operations

### 3. **Code Organization**
- Use builder patterns for complex pipeline construction
- Create reusable templates for common workflows
- Separate configuration from implementation
- Implement proper logging and monitoring

### 4. **Testing**
- Write unit tests for individual components
- Implement integration tests for complete workflows
- Use mock data for development and testing
- Validate outputs against expected results

## File Structure

```
examples/sdk/
├── README.md                        # This documentation
├── basic_usage.py                   # Fundamental SDK usage patterns
├── pipeline_creation.py             # Programmatic pipeline building
├── pack_loading.py                  # Pack management and execution
├── generated_basic_pipeline.yaml    # Generated pipeline example
├── generated_portfolio_pipeline.yaml # Generated portfolio pipeline
├── generated_custom_pipeline.yaml   # Generated custom pipeline
└── requirements.txt                 # Python dependencies for examples
```

## Prerequisites

### Python Dependencies
```bash
pip install pyyaml
pip install logging
pip install pathlib
pip install dataclasses
pip install typing
```

### GreenLang Installation
```bash
pip install greenlang
# or for development
pip install -e .
```

## Running the Examples

### Individual Examples
```bash
# Basic usage patterns
python examples/sdk/basic_usage.py

# Pipeline creation
python examples/sdk/pipeline_creation.py

# Pack loading and management
python examples/sdk/pack_loading.py
```

### All Examples
```bash
# Run all SDK examples
cd examples/sdk
python -c "
import basic_usage
import pipeline_creation
import pack_loading

basic_usage.main()
pipeline_creation.main()
pack_loading.main()
"
```

## Common Issues and Solutions

### 1. **Import Errors**
```bash
# Ensure GreenLang is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. **Pack Loading Issues**
```bash
# Verify pack structure
gl pack validate examples/packs/advanced/pack.yaml
```

### 3. **Pipeline Execution Failures**
```bash
# Enable debug logging
export GREENLANG_LOG_LEVEL=DEBUG
python examples/sdk/basic_usage.py
```

## Next Steps

After working through these examples:

1. **Explore Advanced Features**: Look at the [complex pipeline examples](../pipelines/complex/)
2. **Study Pack Structure**: Examine the [advanced pack example](../packs/advanced/)
3. **Build Custom Solutions**: Use these patterns to create your own workflows
4. **Contribute**: Add your own examples and improvements

## Related Documentation

- [Pipeline Specification](../../docs/specs/GL_PIPELINE_SPEC_V1.md)
- [Pack Schema](../../docs/specs/PACK_SCHEMA_V1.md)
- [API Documentation](../../docs/api/)
- [Developer Guides](../../docs/guides/)

## Support

For SDK-related questions:
- Check the [examples index](../README.md)
- Review the [API documentation](../../docs/api/)
- Submit issues on the [project repository](https://github.com/greenlang/greenlang)