# GreenLang API Reference

**Complete API Documentation for GreenLang Framework v0.3.0**

This comprehensive reference documents all major classes, functions, and utilities in the GreenLang framework.

---

## Table of Contents

1. [Core SDK](#core-sdk)
   - [Agent](#agent)
   - [Pipeline](#pipeline)
   - [Context](#context)
   - [Result](#result)
2. [Provenance Framework](#provenance-framework)
   - [Decorators](#provenance-decorators)
   - [ProvenanceContext](#provenancecontext)
   - [ProvenanceRecord](#provenancerecord)
3. [Emission Calculations](#emission-calculations)
   - [EmissionFactorService](#emissionfactorservice)
   - [FuelConsumption](#fuelconsumption)
4. [Data Models](#data-models)
   - [BuildingData](#buildingdata)
   - [EmissionResult](#emissionresult)
5. [Utilities](#utilities)
   - [BatchProcessor](#batchprocessor)
   - [CSVLoader](#csvloader)
   - [DataValidator](#datavalidator)
6. [CLI Commands](#cli-commands)

---

## Core SDK

### Agent

Base class for all GreenLang agents. Agents are stateless computation units that process typed inputs and produce typed outputs.

#### Class Definition

```python
from greenlang.sdk import Agent
from typing import TypeVar, Generic

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")

class Agent(ABC, Generic[TInput, TOutput]):
    """Base Agent abstraction for climate intelligence"""
```

#### Constructor

```python
def __init__(self, metadata: Optional[Metadata] = None):
    """
    Initialize agent with metadata.

    Args:
        metadata (Optional[Metadata]): Agent metadata (id, name, version, etc.)
    """
```

#### Methods

##### `validate(input_data: TInput) -> bool`

Validate input data before processing.

**Parameters:**
- `input_data` (TInput): Input data to validate

**Returns:**
- `bool`: True if valid, False otherwise

**Example:**
```python
def validate(self, input_data: BuildingInput) -> bool:
    """Validate building input data"""
    return input_data.area_sqft > 0 and len(input_data.fuels) > 0
```

##### `process(input_data: TInput) -> TOutput`

Process input and produce output. Must be implemented by subclasses.

**Parameters:**
- `input_data` (TInput): Validated input data

**Returns:**
- `TOutput`: Processed output

**Example:**
```python
def process(self, input_data: BuildingInput) -> dict:
    """Calculate building emissions"""
    emissions = self.calculate_emissions(input_data)
    return {"total_emissions_tons": emissions}
```

##### `run(input_data: TInput) -> Result`

Execute agent with validation and error handling.

**Parameters:**
- `input_data` (TInput): Input data

**Returns:**
- `Result`: Result object with success status, data, and metadata

**Example:**
```python
agent = EmissionsAgent()
input_data = BuildingInput(name="Office", area_sqft=10000, fuels=[...])
result = agent.run(input_data)

if result.success:
    print(f"Emissions: {result.data['total_emissions_tons']}")
else:
    print(f"Error: {result.error}")
```

##### `describe() -> Dict[str, Any]`

Get agent description including capabilities and schemas.

**Returns:**
- `dict`: Agent metadata, input schema, and output schema

**Example:**
```python
agent = EmissionsAgent()
description = agent.describe()

print(f"Agent ID: {description['metadata']['id']}")
print(f"Input Schema: {description['input_schema']}")
```

#### Complete Example

```python
from greenlang.sdk import Agent, Result, Metadata
from greenlang.provenance.decorators import traced
from pydantic import BaseModel, Field
from typing import Dict

class FuelInput(BaseModel):
    """Input model with automatic validation"""
    fuel_type: str = Field(..., description="Type of fuel")
    consumption: float = Field(..., gt=0, description="Consumption amount")
    unit: str = Field(default="kWh", description="Unit of measurement")

class EmissionsAgent(Agent[FuelInput, Dict]):
    """Agent to calculate emissions from fuel consumption"""

    def __init__(self):
        super().__init__(
            metadata=Metadata(
                id="emissions_calculator",
                name="Emissions Calculator Agent",
                version="1.0.0",
                description="Calculate CO2 emissions from fuel consumption",
                tags=["emissions", "carbon", "scope1", "scope2"]
            )
        )

    def validate(self, input_data: FuelInput) -> bool:
        """Validation handled by Pydantic"""
        return True

    @traced(save_path="provenance/emissions.json", track_inputs=True, track_outputs=True)
    def process(self, input_data: FuelInput) -> Dict:
        """Calculate emissions with automatic provenance tracking"""
        from greenlang.emissions import EmissionFactorService

        ef_service = EmissionFactorService()
        emissions = ef_service.calculate_emissions(
            fuel_type=input_data.fuel_type,
            consumption=input_data.consumption,
            unit=input_data.unit
        )

        return {
            "emissions_kg": emissions.kg,
            "emissions_tons": emissions.tons,
            "fuel_type": input_data.fuel_type,
            "consumption": input_data.consumption,
            "unit": input_data.unit,
            "emission_factor": emissions.factor_used
        }

# Usage
agent = EmissionsAgent()
input_data = FuelInput(fuel_type="electricity", consumption=1000, unit="kWh")
result = agent.run(input_data)

assert result.success
assert result.data["emissions_tons"] > 0
```

---

### Pipeline

Base class for orchestrating multiple agents in sequence or parallel.

#### Class Definition

```python
from greenlang.sdk import Pipeline

class Pipeline(ABC):
    """Base Pipeline abstraction"""
```

#### Constructor

```python
def __init__(self, metadata: Optional[Metadata] = None):
    """
    Initialize pipeline.

    Args:
        metadata (Optional[Metadata]): Pipeline metadata
    """
```

#### Methods

##### `add_agent(agent: Agent) -> Pipeline`

Add an agent to the pipeline.

**Parameters:**
- `agent` (Agent): Agent to add to pipeline

**Returns:**
- `Pipeline`: Self (for method chaining)

**Example:**
```python
pipeline = EmissionsPipeline()
pipeline.add_agent(DataValidator())\
        .add_agent(EmissionsCalculator())\
        .add_agent(ReportGenerator())
```

##### `execute(input_data: Any) -> Result`

Execute the pipeline with input data.

**Parameters:**
- `input_data` (Any): Initial input data

**Returns:**
- `Result`: Final pipeline result

**Example:**
```python
pipeline = EmissionsPipeline()
result = pipeline.execute({"building_data": {...}})

if result.success:
    print(f"Pipeline completed: {result.data}")
```

##### `describe() -> Dict[str, Any]`

Get pipeline description.

**Returns:**
- `dict`: Pipeline metadata, agent list, and flow definition

#### Complete Example

```python
from greenlang.sdk import Pipeline, Context, Result
from pathlib import Path

class BuildingEmissionsPipeline(Pipeline):
    """Pipeline for building emissions analysis"""

    def __init__(self):
        super().__init__(
            metadata=Metadata(
                id="building_emissions_pipeline",
                name="Building Emissions Analysis Pipeline",
                version="1.0.0"
            )
        )
        self.data_loader = DataLoaderAgent()
        self.validator = DataValidatorAgent()
        self.calculator = EmissionsCalculatorAgent()
        self.reporter = ReportGeneratorAgent()

    def execute(self, input_data: Dict) -> Result:
        """Execute pipeline stages"""
        ctx = Context(
            inputs=input_data,
            artifacts_dir=Path("output/artifacts"),
            metadata={"pipeline": "building_emissions"}
        )

        try:
            # Stage 1: Load data
            load_result = self.data_loader.run(input_data)
            if not load_result.success:
                return Result(success=False, error=f"Data loading failed: {load_result.error}")
            ctx.add_step_result("load", load_result)

            # Stage 2: Validate data
            validate_result = self.validator.run(load_result.data)
            if not validate_result.success:
                return Result(success=False, error=f"Validation failed: {validate_result.error}")
            ctx.add_step_result("validate", validate_result)

            # Stage 3: Calculate emissions
            calc_result = self.calculator.run(validate_result.data)
            if not calc_result.success:
                return Result(success=False, error=f"Calculation failed: {calc_result.error}")
            ctx.add_step_result("calculate", calc_result)

            # Stage 4: Generate report
            report_result = self.reporter.run(calc_result.data)
            if not report_result.success:
                return Result(success=False, error=f"Reporting failed: {report_result.error}")
            ctx.add_step_result("report", report_result)

            # Return final result
            return ctx.to_result()

        except Exception as e:
            return Result(success=False, error=str(e))

# Usage
pipeline = BuildingEmissionsPipeline()
result = pipeline.execute({"csv_path": "buildings.csv"})

if result.success:
    print(f"Pipeline completed successfully")
    print(f"Total emissions: {result.data['calculate']['outputs']['total_emissions_tons']}")
else:
    print(f"Pipeline failed: {result.error}")
```

---

### Context

Execution context for pipelines and agents. Manages inputs, outputs, artifacts, and metadata.

#### Class Definition

```python
from greenlang.sdk import Context

class Context:
    """Execution context for pipelines and agents"""
```

#### Constructor

```python
def __init__(
    self,
    inputs: Optional[Dict[str, Any]] = None,
    artifacts_dir: Optional[Path] = None,
    profile: str = "dev",
    backend: str = "local",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Initialize execution context.

    Args:
        inputs (Optional[Dict]): Input data dictionary
        artifacts_dir (Optional[Path]): Directory for storing artifacts
        profile (str): Execution profile (dev, staging, prod)
        backend (str): Execution backend (local, k8s, etc.)
        metadata (Optional[Dict]): Additional metadata
    """
```

#### Attributes

- `inputs` (dict): Input data for the pipeline
- `data` (dict): Alias for inputs (compatibility)
- `artifacts_dir` (Path): Directory where artifacts are stored
- `artifacts` (Dict[str, Artifact]): Dictionary of artifacts
- `steps` (dict): Results from pipeline steps
- `metadata` (dict): Execution metadata
- `start_time` (datetime): Context start time

#### Methods

##### `add_artifact(name: str, path: Path, type: str = "file", **metadata) -> Artifact`

Add an artifact to the context.

**Parameters:**
- `name` (str): Artifact name
- `path` (Path): Path to artifact file
- `type` (str): Artifact type (default: "file")
- `**metadata`: Additional metadata

**Returns:**
- `Artifact`: Created artifact object

##### `save_artifact(name: str, content: Any, type: str = "json", **metadata) -> Artifact`

Save content as an artifact file.

**Parameters:**
- `name` (str): Artifact name
- `content` (Any): Content to save
- `type` (str): Format (json, yaml, text, csv)
- `**metadata`: Additional metadata

**Returns:**
- `Artifact`: Created artifact object

##### `add_step_result(name: str, result: Result)`

Add a pipeline step result to the context.

**Parameters:**
- `name` (str): Step name
- `result` (Result): Step result

##### `get_step_output(step_name: str) -> Optional[Any]`

Get output from a previous step.

**Parameters:**
- `step_name` (str): Name of the step

**Returns:**
- `Any`: Step output or None

##### `to_result() -> Result`

Convert context to a Result object.

**Returns:**
- `Result`: Result containing all step results

#### Complete Example

```python
from greenlang.sdk import Context, Result
from pathlib import Path
import json

# Create context
ctx = Context(
    inputs={"building_id": "B001", "data_path": "input.csv"},
    artifacts_dir=Path("output/artifacts"),
    profile="production",
    metadata={"user": "analyst@company.com", "project": "Q4_analysis"}
)

# Save an artifact
emissions_data = {"total": 125.5, "breakdown": {...}}
artifact = ctx.save_artifact(
    name="emissions_result",
    content=emissions_data,
    type="json",
    description="Final emissions calculation"
)

# Add step results
validation_result = Result(success=True, data={"valid_rows": 100})
ctx.add_step_result("validation", validation_result)

calculation_result = Result(success=True, data={"emissions": 125.5})
ctx.add_step_result("calculation", calculation_result)

# Retrieve step output
validation_data = ctx.get_step_output("validation")
print(f"Validated {validation_data['valid_rows']} rows")

# Convert to final result
final_result = ctx.to_result()
print(f"Pipeline success: {final_result.success}")
print(f"Duration: {final_result.metadata['duration']:.2f} seconds")
```

---

### Result

Standard result container for agent and pipeline executions.

#### Class Definition

```python
from greenlang.sdk.base import Result
from dataclasses import dataclass

@dataclass
class Result:
    """Standard result container"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Attributes

- `success` (bool): Whether execution succeeded
- `data` (Any): Output data (if successful)
- `error` (str): Error message (if failed)
- `metadata` (dict): Additional metadata

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert result to dictionary.

**Returns:**
- `dict`: Dictionary representation

#### Example

```python
from greenlang.sdk.base import Result

# Success result
success_result = Result(
    success=True,
    data={"emissions_tons": 125.5, "intensity": 0.45},
    metadata={"agent": "emissions_calculator", "version": "1.0.0", "duration_ms": 123}
)

# Error result
error_result = Result(
    success=False,
    error="Invalid input: consumption cannot be negative",
    metadata={"agent": "emissions_calculator", "input_validation_failed": True}
)

# Check result
if success_result.success:
    emissions = success_result.data["emissions_tons"]
    print(f"Calculated emissions: {emissions:.2f} tCO2e")
else:
    print(f"Error: {success_result.error}")

# Convert to dict for serialization
result_dict = success_result.to_dict()
import json
json.dump(result_dict, open("result.json", "w"))
```

---

## Provenance Framework

### Provenance Decorators

Decorators for automatic provenance tracking.

#### `@traced`

Automatically track provenance for functions.

**Signature:**
```python
def traced(
    record_id: Optional[str] = None,
    save_path: Optional[str] = None,
    track_inputs: bool = True,
    track_outputs: bool = True
)
```

**Parameters:**
- `record_id` (Optional[str]): Custom record ID
- `save_path` (Optional[str]): Path to save provenance record
- `track_inputs` (bool): Whether to track input arguments (default: True)
- `track_outputs` (bool): Whether to track output results (default: True)

**Example:**
```python
from greenlang.provenance.decorators import traced

@traced(save_path="provenance/calculations.json", track_inputs=True, track_outputs=True)
def calculate_emissions(fuel_type: str, consumption: float) -> float:
    """Calculate emissions with automatic provenance"""
    factor = get_emission_factor(fuel_type)
    emissions = consumption * factor
    return emissions

# Provenance automatically recorded when function is called
result = calculate_emissions("electricity", 1000)
# Provenance saved to: provenance/calculations.json
```

#### `@track_provenance`

Track provenance for class methods.

**Signature:**
```python
def track_provenance(
    context_attr: str = "_provenance_context",
    save_on_completion: bool = False
)
```

**Parameters:**
- `context_attr` (str): Name of context attribute on class (default: "_provenance_context")
- `save_on_completion` (bool): Save provenance when method completes (default: False)

**Example:**
```python
from greenlang.provenance.decorators import track_provenance
from greenlang.provenance.records import ProvenanceContext

class DataPipeline:
    """Pipeline with automatic provenance tracking"""

    def __init__(self):
        self._provenance_context = ProvenanceContext("data_pipeline")

    @track_provenance()
    def load_data(self, path: str):
        """Load data - automatically tracked"""
        data = pd.read_csv(path)
        return data

    @track_provenance()
    def transform_data(self, data):
        """Transform data - automatically tracked"""
        transformed = data.apply(transformation)
        return transformed

    @track_provenance(save_on_completion=True)
    def save_results(self, data, output_path: str):
        """Save results - provenance saved on completion"""
        data.to_csv(output_path)

# Usage - provenance tracked automatically
pipeline = DataPipeline()
data = pipeline.load_data("input.csv")
transformed = pipeline.transform_data(data)
pipeline.save_results(transformed, "output.csv")
# Provenance automatically saved after save_results()
```

#### Context Manager: `provenance_tracker`

Context manager for provenance tracking.

**Example:**
```python
from greenlang.provenance.decorators import provenance_tracker

with provenance_tracker("my_operation", save_path="provenance/op.json") as ctx:
    # Do work
    data = load_data("input.csv")
    ctx.record_input("input.csv", {"rows": len(data)})

    # Process
    result = process(data)
    ctx.record_output("output.csv", {"rows": len(result)})

    # Save
    result.to_csv("output.csv")

# Provenance automatically saved when context exits
```

---

### ProvenanceContext

Context for tracking provenance during execution.

#### Constructor

```python
from greenlang.provenance.records import ProvenanceContext

def __init__(self, name: str, record_id: Optional[str] = None):
    """
    Initialize provenance context.

    Args:
        name (str): Operation name
        record_id (Optional[str]): Custom record ID (auto-generated if not provided)
    """
```

#### Methods

##### `record_input(name: str, metadata: Dict)`

Record an input to the operation.

##### `record_output(name: str, metadata: Dict)`

Record an output from the operation.

##### `record_agent_execution(agent_name: str, start_time: str, end_time: str, duration_seconds: float, metadata: Dict = None)`

Record agent execution details.

##### `finalize(output_path: Optional[str] = None) -> ProvenanceRecord`

Finalize and optionally save provenance record.

#### Complete Example

```python
from greenlang.provenance.records import ProvenanceContext
from datetime import datetime

# Create context
ctx = ProvenanceContext(name="emissions_calculation", record_id="calc_001")

# Record input
ctx.record_input(
    name="building_data",
    metadata={
        "source": "input.csv",
        "rows": 100,
        "columns": ["building_id", "electricity_kwh", "gas_therms"]
    }
)

# Do calculation
start_time = datetime.now()
result = calculate_emissions(building_data)
end_time = datetime.now()

# Record execution
ctx.record_agent_execution(
    agent_name="EmissionsCalculator",
    start_time=start_time.isoformat(),
    end_time=end_time.isoformat(),
    duration_seconds=(end_time - start_time).total_seconds(),
    metadata={"total_emissions": result["total"]}
)

# Record output
ctx.record_output(
    name="emissions_result",
    metadata={
        "total_emissions_tons": result["total"],
        "output_file": "emissions.json"
    }
)

# Finalize and save
provenance = ctx.finalize(output_path="provenance/calc_001.json")
print(f"Provenance saved: {provenance.record_id}")
```

---

## Emission Calculations

### EmissionFactorService

Service for retrieving emission factors and calculating emissions.

#### Constructor

```python
from greenlang.emissions import EmissionFactorService

def __init__(self, region: str = "US", year: Optional[int] = None):
    """
    Initialize emission factor service.

    Args:
        region (str): Geographic region (default: "US")
        year (Optional[int]): Year for emission factors (default: current year)
    """
```

#### Methods

##### `calculate_emissions(fuel_type: str, consumption: float, unit: str) -> EmissionResult`

Calculate emissions for fuel consumption.

**Parameters:**
- `fuel_type` (str): Type of fuel (electricity, natural_gas, diesel, etc.)
- `consumption` (float): Consumption amount
- `unit` (str): Unit of measurement (kWh, therms, liters, etc.)

**Returns:**
- `EmissionResult`: Emissions result with kg, tons, and metadata

##### `get_factor(fuel_type: str, region: str = None) -> float`

Get emission factor for a fuel type.

**Parameters:**
- `fuel_type` (str): Type of fuel
- `region` (Optional[str]): Region (uses service default if not provided)

**Returns:**
- `float`: Emission factor (kgCO2e per unit)

#### Complete Example

```python
from greenlang.emissions import EmissionFactorService

# Initialize service
ef_service = EmissionFactorService(region="US", year=2024)

# Calculate emissions for electricity
elec_result = ef_service.calculate_emissions(
    fuel_type="electricity",
    consumption=10000,
    unit="kWh"
)

print(f"Electricity emissions: {elec_result.tons:.2f} tCO2e")
print(f"Factor used: {elec_result.factor_used} kgCO2e/kWh")

# Calculate emissions for natural gas
gas_result = ef_service.calculate_emissions(
    fuel_type="natural_gas",
    consumption=500,
    unit="therms"
)

print(f"Gas emissions: {gas_result.tons:.2f} tCO2e")

# Get emission factor directly
diesel_factor = ef_service.get_factor("diesel", region="US")
print(f"Diesel emission factor: {diesel_factor} kgCO2e/liter")

# Calculate for different regions
ef_service_uk = EmissionFactorService(region="UK")
uk_elec_result = ef_service_uk.calculate_emissions("electricity", 10000, "kWh")
print(f"UK electricity emissions: {uk_elec_result.tons:.2f} tCO2e")
```

---

## Utilities

### BatchProcessor

Process large datasets in batches with optional parallelization.

#### Constructor

```python
from greenlang.utils import BatchProcessor

def __init__(
    self,
    agent: Agent,
    batch_size: int = 100,
    parallel: bool = False,
    num_workers: int = 4
):
    """
    Initialize batch processor.

    Args:
        agent (Agent): Agent to process each item
        batch_size (int): Number of items per batch (default: 100)
        parallel (bool): Enable parallel processing (default: False)
        num_workers (int): Number of parallel workers (default: 4)
    """
```

#### Methods

##### `process_batch(data: List[Any]) -> List[Result]`

Process a batch of data items.

**Parameters:**
- `data` (List[Any]): List of data items to process

**Returns:**
- `List[Result]`: List of results for each item

#### Example

```python
from greenlang.utils import BatchProcessor
from greenlang.agents import EmissionsAgent

# Create agent
agent = EmissionsAgent()

# Create batch processor
processor = BatchProcessor(
    agent=agent,
    batch_size=500,
    parallel=True,
    num_workers=8
)

# Load large dataset
buildings_data = load_buildings_from_csv("buildings.csv")  # 10,000+ rows

# Process in batches
print(f"Processing {len(buildings_data)} buildings...")
results = processor.process_batch(buildings_data)

# Analyze results
successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

total_emissions = sum(r.data["emissions_tons"] for r in successful)
print(f"Total emissions: {total_emissions:.2f} tCO2e")
```

---

### CSVLoader

Load and validate CSV data.

#### Constructor

```python
from greenlang.utils import CSVLoader

def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
    """
    Initialize CSV loader.

    Args:
        delimiter (str): CSV delimiter (default: ",")
        encoding (str): File encoding (default: "utf-8")
    """
```

#### Methods

##### `load(filepath: str, model: Type[BaseModel] = None) -> List[Dict]`

Load CSV file with optional Pydantic model validation.

**Parameters:**
- `filepath` (str): Path to CSV file
- `model` (Optional[Type[BaseModel]]): Pydantic model for validation

**Returns:**
- `List[Dict]`: Loaded and validated data

#### Example

```python
from greenlang.utils import CSVLoader
from pydantic import BaseModel, Field

# Define model
class BuildingRow(BaseModel):
    building_id: str
    electricity_kwh: float = Field(..., gt=0)
    gas_therms: float = Field(..., gt=0)

# Load and validate
loader = CSVLoader()
data = loader.load("buildings.csv", model=BuildingRow)

print(f"Loaded {len(data)} validated rows")
for row in data[:5]:  # First 5 rows
    print(f"Building {row.building_id}: {row.electricity_kwh} kWh")
```

---

## CLI Commands

### `gl version`

Display GreenLang version.

```bash
gl version
# Output: GreenLang v0.3.0
```

### `gl doctor`

Run health check on GreenLang installation.

```bash
gl doctor --verbose
# Checks: Python version, dependencies, configuration, etc.
```

### `gl init`

Initialize a new GreenLang project or agent.

```bash
# Initialize new agent
gl init agent my_emissions_agent --template basic

# Initialize new pipeline
gl init pipeline my_pipeline --template data_processing
```

### `gl calc`

Quick calculation using CLI.

```bash
# Simple calculation
gl calc --fuel-type electricity --consumption 1000 --unit kWh

# With multiple fuels
gl calc --fuels "electricity:1000:kWh,natural_gas:100:therms" --output result.json

# From input file
gl calc --input building_data.json --output results.json
```

### `gl pack`

Manage GreenLang packs.

```bash
# List installed packs
gl pack list

# Install a pack
gl pack install emissions-calculator

# Create new pack
gl pack create my-custom-pack

# Validate pack
gl pack verify my-pack/
```

### `gl run`

Run a pipeline.

```bash
# Run pipeline from YAML
gl run pipeline.yaml --input data.json --output results/

# Run with specific profile
gl run pipeline.yaml --profile production --backend k8s
```

---

## Error Handling

All agents and utilities use consistent error handling:

```python
from greenlang.sdk import Agent, Result

class MyAgent(Agent):
    def process(self, input_data):
        try:
            result = do_calculation(input_data)
            return result
        except ValueError as e:
            # Validation errors
            raise ValueError(f"Invalid input: {e}")
        except Exception as e:
            # Unexpected errors
            raise RuntimeError(f"Calculation failed: {e}")

# Errors are captured in Result
result = agent.run(input_data)
if not result.success:
    print(f"Error: {result.error}")
    # Handle error appropriately
```

---

## Type Hints

GreenLang extensively uses type hints for better IDE support:

```python
from typing import Dict, List, Optional, Union
from greenlang.sdk import Agent, Result
from pydantic import BaseModel

class MyAgent(Agent[InputModel, OutputModel]):
    """Agent with explicit type parameters"""

    def process(self, input_data: InputModel) -> OutputModel:
        """Type-checked processing"""
        result: OutputModel = calculate(input_data)
        return result

# Type checking with mypy
# mypy src/ --strict
```

---

## Best Practices

1. **Always use Pydantic models** for input validation
2. **Add @traced decorator** for provenance tracking
3. **Use type hints** for better code quality
4. **Implement proper error handling** in all agents
5. **Test agents thoroughly** with unit and integration tests
6. **Document your agents** with clear docstrings
7. **Use BatchProcessor** for large datasets
8. **Enable caching** in production environments

---

## See Also

- [Quick Start Guide](QUICKSTART.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Examples](../examples/)

---

**GreenLang v0.3.0 - The Climate Intelligence Platform**

*For questions or issues, visit: [github.com/greenlang/greenlang/issues](https://github.com/greenlang/greenlang/issues)*
