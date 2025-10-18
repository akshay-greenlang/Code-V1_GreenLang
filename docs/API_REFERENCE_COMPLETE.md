# GreenLang Framework API Reference

**Version:** 1.0.0
**Last Updated:** 2025-10-18

This comprehensive API reference documents all public classes, methods, and functions in the GreenLang framework.

---

## Table of Contents

1. [greenlang.agents](#greenlangagents)
   - [BaseAgent](#baseagent)
   - [BaseCalculator](#basecalculator)
   - [BaseReporter](#basereporter)
   - [AgentConfig](#agentconfig)
   - [AgentResult](#agentresult)
   - [AgentMetrics](#agentmetrics)
2. [greenlang.provenance](#greenlangprovenance)
   - [ProvenanceRecord](#provenancerecord)
   - [ProvenanceContext](#provenancecontext)
   - [RunLedger](#runledger)
   - [Decorators](#provenance-decorators)
3. [greenlang.validation](#greenlangvalidation)
   - [ValidationFramework](#validationframework)
   - [SchemaValidator](#schemavalidator)
   - [RulesEngine](#rulesengine)
   - [ValidationResult](#validationresult)
4. [greenlang.io](#greenlangio)
   - [DataReader](#datareader)
   - [DataWriter](#datawriter)

---

## greenlang.agents

The agents module provides base classes for building domain-specific processing agents with lifecycle management, metrics tracking, and provenance integration.

### BaseAgent

**Module:** `greenlang.agents.base`

Base class for all GreenLang agents providing lifecycle management, metrics collection, and provenance tracking.

#### Class Signature

```python
class BaseAgent(ABC):
    def __init__(self, config: Optional[AgentConfig] = None)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Optional[AgentConfig]` | `None` | Agent configuration. If None, creates default config from class metadata |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | `AgentConfig` | Agent configuration object |
| `logger` | `logging.Logger` | Logger instance for this agent |
| `stats` | `StatsTracker` | Execution statistics tracker |

#### Methods

##### `run(input_data: Dict[str, Any]) -> AgentResult`

Execute the agent with full lifecycle management.

**Parameters:**
- `input_data` (Dict[str, Any]): Input data dictionary

**Returns:**
- `AgentResult`: Execution results with metrics and metadata

**Example:**
```python
from greenlang.agents import BaseAgent, AgentConfig, AgentResult

class MyAgent(BaseAgent):
    def execute(self, input_data):
        result = {"output": input_data["value"] * 2}
        return AgentResult(success=True, data=result)

config = AgentConfig(name="MyAgent", description="Example agent")
agent = MyAgent(config)
result = agent.run({"value": 42})

print(f"Success: {result.success}")
print(f"Data: {result.data}")
print(f"Execution time: {result.metrics.execution_time_ms}ms")
```

##### `execute(input_data: Dict[str, Any]) -> AgentResult` (Abstract)

Core execution logic - must be implemented by subclasses.

**Parameters:**
- `input_data` (Dict[str, Any]): Input data dictionary

**Returns:**
- `AgentResult`: Execution results

**Example:**
```python
class DataProcessor(BaseAgent):
    def execute(self, input_data):
        # Process data
        processed = self.process_records(input_data["records"])

        return AgentResult(
            success=True,
            data={"processed_records": processed},
            metadata={"count": len(processed)}
        )
```

##### `validate_input(input_data: Dict[str, Any]) -> bool`

Validate input data before execution. Override to add custom validation logic.

**Parameters:**
- `input_data` (Dict[str, Any]): Input data to validate

**Returns:**
- `bool`: True if valid, False otherwise

**Example:**
```python
class ValidatingAgent(BaseAgent):
    def validate_input(self, input_data):
        if "required_field" not in input_data:
            self.logger.error("Missing required field")
            return False
        return True

    def execute(self, input_data):
        # Execute knowing input is valid
        return AgentResult(success=True, data={})
```

##### `preprocess(input_data: Dict[str, Any]) -> Dict[str, Any]`

Preprocess input data before execution.

**Parameters:**
- `input_data` (Dict[str, Any]): Raw input data

**Returns:**
- `Dict[str, Any]`: Preprocessed input data

**Example:**
```python
class PreprocessingAgent(BaseAgent):
    def preprocess(self, input_data):
        # Normalize values
        input_data["normalized_value"] = input_data["value"] / 100.0
        return input_data

    def execute(self, input_data):
        # Use preprocessed data
        result = input_data["normalized_value"] * 2
        return AgentResult(success=True, data={"result": result})
```

##### `postprocess(result: AgentResult) -> AgentResult`

Postprocess result after execution.

**Parameters:**
- `result` (AgentResult): Raw execution result

**Returns:**
- `AgentResult`: Postprocessed result

**Example:**
```python
class PostprocessingAgent(BaseAgent):
    def postprocess(self, result):
        # Add computed metadata
        if result.success:
            result.metadata["processed_at"] = datetime.now().isoformat()
        return result

    def execute(self, input_data):
        return AgentResult(success=True, data={"value": 100})
```

##### `initialize()`

Override to add custom initialization logic. Called after constructor.

**Example:**
```python
class InitializingAgent(BaseAgent):
    def initialize(self):
        self.cache = {}
        self.logger.info("Agent initialized with cache")

    def execute(self, input_data):
        # Use initialized resources
        return AgentResult(success=True, data={})
```

##### `cleanup()`

Cleanup resources after execution. Override to add custom cleanup logic.

**Example:**
```python
class CleanupAgent(BaseAgent):
    def cleanup(self):
        if hasattr(self, 'connection'):
            self.connection.close()
            self.logger.info("Connection closed")
```

##### `load_resource(resource_path: str) -> Any`

Load a resource file with caching.

**Parameters:**
- `resource_path` (str): Path to resource file

**Returns:**
- `Any`: Loaded resource data

**Raises:**
- `FileNotFoundError`: If resource doesn't exist

**Example:**
```python
class ResourceAgent(BaseAgent):
    def execute(self, input_data):
        # Load configuration from file (cached)
        config = self.load_resource("config/settings.json")

        return AgentResult(success=True, data={"config": config})
```

##### `add_pre_hook(hook: Callable)`

Add a pre-execution hook function.

**Parameters:**
- `hook` (Callable): Function called before execution

**Example:**
```python
def log_input(agent, input_data):
    agent.logger.info(f"Processing input: {input_data.keys()}")

agent = MyAgent()
agent.add_pre_hook(log_input)
```

##### `add_post_hook(hook: Callable)`

Add a post-execution hook function.

**Parameters:**
- `hook` (Callable): Function called after execution

**Example:**
```python
def log_result(agent, result):
    agent.logger.info(f"Result: {result.success}")

agent = MyAgent()
agent.add_post_hook(log_result)
```

##### `get_stats() -> Dict[str, Any]`

Get execution statistics.

**Returns:**
- `Dict[str, Any]`: Statistics including executions, success rate, timing

**Example:**
```python
agent = MyAgent()
# Run multiple times
for i in range(10):
    agent.run({"value": i})

stats = agent.get_stats()
print(f"Executions: {stats['executions']}")
print(f"Success rate: {stats['success_rate']}%")
print(f"Average time: {stats['avg_time_ms']}ms")
```

##### `reset_stats()`

Reset execution statistics.

**Example:**
```python
agent.reset_stats()
```

---

### BaseCalculator

**Module:** `greenlang.agents.calculator`

Base class for calculator agents providing high-precision arithmetic, caching, and calculation tracing.

#### Class Signature

```python
class BaseCalculator(BaseAgent):
    def __init__(self, config: Optional[CalculatorConfig] = None)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Optional[CalculatorConfig]` | `None` | Calculator configuration |

#### Methods

##### `calculate(inputs: Dict[str, Any]) -> Any` (Abstract)

Perform the calculation. Must be implemented by subclasses.

**Parameters:**
- `inputs` (Dict[str, Any]): Input values for calculation

**Returns:**
- `Any`: Calculation result

**Example:**
```python
class CarbonCalculator(BaseCalculator):
    def calculate(self, inputs):
        energy_kwh = inputs['energy_kwh']
        emission_factor = inputs['emission_factor']

        # Record calculation step
        self.add_calculation_step(
            step_name="carbon_emission",
            formula="energy_kwh * emission_factor",
            inputs=inputs,
            result=energy_kwh * emission_factor,
            units="kgCO2e"
        )

        return energy_kwh * emission_factor
```

##### `validate_calculation_inputs(inputs: Dict[str, Any]) -> bool`

Validate calculation inputs. Override to add custom validation.

**Parameters:**
- `inputs` (Dict[str, Any]): Inputs to validate

**Returns:**
- `bool`: True if valid, False otherwise

**Example:**
```python
class ValidatedCalculator(BaseCalculator):
    def validate_calculation_inputs(self, inputs):
        required = ['energy_kwh', 'emission_factor']
        for field in required:
            if field not in inputs:
                self.logger.error(f"Missing field: {field}")
                return False
        return True

    def calculate(self, inputs):
        return inputs['energy_kwh'] * inputs['emission_factor']
```

##### `round_decimal(value: Union[float, Decimal], precision: Optional[int] = None) -> Decimal`

Round a value to specified precision.

**Parameters:**
- `value` (Union[float, Decimal]): Value to round
- `precision` (Optional[int]): Decimal places (uses config.precision if None)

**Returns:**
- `Decimal`: Rounded value

**Example:**
```python
calculator = CarbonCalculator()
rounded = calculator.round_decimal(3.14159, precision=2)
print(rounded)  # 3.14
```

##### `safe_divide(numerator: float, denominator: float) -> Optional[float]`

Safely divide two numbers handling division by zero.

**Parameters:**
- `numerator` (float): Numerator
- `denominator` (float): Denominator

**Returns:**
- `Optional[float]`: Division result, or None if division by zero and allowed

**Raises:**
- `ZeroDivisionError`: If division by zero and not allowed

**Example:**
```python
class IntensityCalculator(BaseCalculator):
    def calculate(self, inputs):
        total = inputs['total_emissions']
        area = inputs['area']

        # Safe division
        intensity = self.safe_divide(total, area)
        if intensity is None:
            self.logger.warning("Area is zero")
            return 0.0

        return intensity
```

##### `convert_units(value: float, from_unit: str, to_unit: str) -> float`

Convert value between units.

**Parameters:**
- `value` (float): Value to convert
- `from_unit` (str): Source unit
- `to_unit` (str): Target unit

**Returns:**
- `float`: Converted value

**Raises:**
- `ValueError`: If units incompatible or unknown

**Example:**
```python
calculator = EnergyCalculator()

# Convert energy
kwh_to_mwh = calculator.convert_units(1000, "kWh", "MWh")
print(kwh_to_mwh)  # 1.0

# Convert mass
kg_to_ton = calculator.convert_units(1000, "kg", "t")
print(kg_to_ton)  # 1.0
```

##### `add_calculation_step(step_name: str, formula: str, inputs: Dict[str, Any], result: Any, units: Optional[str] = None)`

Record a calculation step for traceability.

**Parameters:**
- `step_name` (str): Name of the step
- `formula` (str): Formula or expression
- `inputs` (Dict[str, Any]): Input values
- `result` (Any): Calculated result
- `units` (Optional[str]): Units of result

**Example:**
```python
class DetailedCalculator(BaseCalculator):
    def calculate(self, inputs):
        # Step 1: Calculate base
        base = inputs['value'] * 1.2
        self.add_calculation_step(
            "apply_multiplier",
            "value * 1.2",
            {"value": inputs['value']},
            base,
            units="kWh"
        )

        # Step 2: Apply tax
        tax = base * 0.1
        self.add_calculation_step(
            "apply_tax",
            "base * 0.1",
            {"base": base},
            tax,
            units="kWh"
        )

        return base + tax
```

##### `clear_cache()`

Clear the calculation cache.

**Example:**
```python
calculator = CarbonCalculator()
calculator.clear_cache()
```

#### Configuration Options

**CalculatorConfig** extends `AgentConfig` with:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `precision` | `int` | `6` | Decimal precision for calculations |
| `enable_caching` | `bool` | `True` | Enable caching of results |
| `cache_size` | `int` | `128` | Maximum cached results |
| `validate_inputs` | `bool` | `True` | Validate inputs before calculation |
| `allow_division_by_zero` | `bool` | `False` | Allow division by zero (returns None) |

**Example:**
```python
config = CalculatorConfig(
    name="PrecisionCalculator",
    precision=8,
    enable_caching=True,
    cache_size=256
)
calculator = MyCalculator(config)
```

---

### BaseReporter

**Module:** `greenlang.agents.reporter`

Base class for reporting agents supporting multiple output formats.

#### Class Signature

```python
class BaseReporter(BaseAgent):
    def __init__(self, config: Optional[ReporterConfig] = None)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Optional[ReporterConfig]` | `None` | Reporter configuration |

#### Methods

##### `aggregate_data(input_data: Dict[str, Any]) -> Dict[str, Any]` (Abstract)

Aggregate input data for reporting. Must be implemented by subclasses.

**Parameters:**
- `input_data` (Dict[str, Any]): Raw input data

**Returns:**
- `Dict[str, Any]`: Aggregated data dictionary

**Example:**
```python
class SalesReporter(BaseReporter):
    def aggregate_data(self, input_data):
        sales = input_data['sales']
        return {
            'total_sales': sum(sales),
            'average_sale': sum(sales) / len(sales),
            'max_sale': max(sales),
            'min_sale': min(sales),
            'count': len(sales)
        }

    def build_sections(self, aggregated_data):
        return [
            ReportSection(
                title="Sales Summary",
                content=f"Total: ${aggregated_data['total_sales']:,.2f}",
                section_type="text"
            )
        ]
```

##### `build_sections(aggregated_data: Dict[str, Any]) -> List[ReportSection]` (Abstract)

Build report sections from aggregated data. Must be implemented by subclasses.

**Parameters:**
- `aggregated_data` (Dict[str, Any]): Aggregated data

**Returns:**
- `List[ReportSection]`: List of report sections

**Example:**
```python
class DetailedReporter(BaseReporter):
    def aggregate_data(self, input_data):
        return {"total": sum(input_data['values'])}

    def build_sections(self, aggregated_data):
        sections = []

        # Summary section
        sections.append(ReportSection(
            title="Summary",
            content=f"Total: {aggregated_data['total']}",
            level=2,
            section_type="text"
        ))

        # Details table
        sections.append(ReportSection(
            title="Details",
            content=[
                {"metric": "Total", "value": aggregated_data['total']},
                {"metric": "Average", "value": aggregated_data.get('avg', 0)}
            ],
            level=2,
            section_type="table"
        ))

        return sections
```

##### `add_section(title: str, content: Any, level: int = 2, section_type: str = "text")`

Add a section to the report.

**Parameters:**
- `title` (str): Section title
- `content` (Any): Section content
- `level` (int): Heading level (1-6)
- `section_type` (str): Type of section (text, table, list, chart)

**Example:**
```python
reporter = SalesReporter()
reporter.add_section("Executive Summary", "Q4 performance exceeded targets", level=1)
reporter.add_section("Key Metrics", [
    {"metric": "Revenue", "value": "$1.2M"},
    {"metric": "Growth", "value": "15%"}
], level=2, section_type="table")
```

##### `generate_summary(aggregated_data: Dict[str, Any]) -> str`

Generate summary text from aggregated data.

**Parameters:**
- `aggregated_data` (Dict[str, Any]): Aggregated data

**Returns:**
- `str`: Summary text in markdown format

**Example:**
```python
class CustomReporter(BaseReporter):
    def generate_summary(self, aggregated_data):
        lines = [
            f"## Executive Summary",
            f"",
            f"Total Revenue: ${aggregated_data['revenue']:,.2f}",
            f"Growth Rate: {aggregated_data['growth_rate']}%",
            f"Customer Count: {aggregated_data['customers']:,}"
        ]
        return "\n".join(lines)
```

##### `render_markdown() -> str`

Render report as Markdown.

**Returns:**
- `str`: Markdown formatted report

**Example:**
```python
reporter = SalesReporter()
result = reporter.run({"sales": [100, 200, 300]})
markdown = reporter.render_markdown()

with open("report.md", "w") as f:
    f.write(markdown)
```

##### `render_html() -> str`

Render report as HTML with styling.

**Returns:**
- `str`: HTML formatted report

**Example:**
```python
reporter = SalesReporter()
result = reporter.run({"sales": [100, 200, 300]})
html = reporter.render_html()

with open("report.html", "w") as f:
    f.write(html)
```

##### `render_json() -> str`

Render report as JSON.

**Returns:**
- `str`: JSON formatted report

**Example:**
```python
reporter = SalesReporter()
result = reporter.run({"sales": [100, 200, 300]})
json_report = reporter.render_json()

with open("report.json", "w") as f:
    f.write(json_report)
```

##### `render_excel(output_path: str)`

Render report as Excel file.

**Parameters:**
- `output_path` (str): Path to save Excel file

**Raises:**
- `ImportError`: If openpyxl not available

**Example:**
```python
reporter = SalesReporter()
result = reporter.run({"sales": [100, 200, 300]})
reporter.render_excel("report.xlsx")
```

#### Configuration Options

**ReporterConfig** extends `AgentConfig` with:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_format` | `str` | `"markdown"` | Output format (markdown, html, json, excel) |
| `include_summary` | `bool` | `True` | Include summary section |
| `include_details` | `bool` | `True` | Include detailed sections |
| `include_charts` | `bool` | `False` | Include charts/visualizations |
| `template_path` | `Optional[str]` | `None` | Custom template path |

**Example:**
```python
config = ReporterConfig(
    name="QuarterlyReport",
    output_format="html",
    include_summary=True,
    include_charts=True
)
reporter = QuarterlyReporter(config)
```

---

### AgentConfig

**Module:** `greenlang.agents.base`

Configuration model for agents using Pydantic.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Name of the agent |
| `description` | `str` | Required | Description of agent's purpose |
| `version` | `str` | `"0.0.1"` | Agent version |
| `enabled` | `bool` | `True` | Whether agent is enabled |
| `parameters` | `Dict[str, Any]` | `{}` | Agent-specific parameters |
| `enable_metrics` | `bool` | `True` | Enable metrics collection |
| `enable_provenance` | `bool` | `True` | Enable provenance tracking |
| `resource_paths` | `List[str]` | `[]` | Paths to resource files |
| `log_level` | `str` | `"INFO"` | Logging level |

**Example:**
```python
from greenlang.agents import AgentConfig

config = AgentConfig(
    name="DataProcessor",
    description="Processes incoming data streams",
    version="1.0.0",
    enabled=True,
    parameters={
        "batch_size": 100,
        "timeout": 30
    },
    enable_metrics=True,
    log_level="DEBUG"
)
```

---

### AgentResult

**Module:** `greenlang.agents.base`

Result model from agent execution.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `success` | `bool` | Required | Whether execution was successful |
| `data` | `Dict[str, Any]` | `{}` | Result data |
| `error` | `Optional[str]` | `None` | Error message if failed |
| `metadata` | `Dict[str, Any]` | `{}` | Additional metadata |
| `metrics` | `Optional[AgentMetrics]` | `None` | Execution metrics |
| `provenance_id` | `Optional[str]` | `None` | Provenance record ID |
| `timestamp` | `Optional[datetime]` | `None` | Execution timestamp |

**Example:**
```python
from greenlang.agents import AgentResult, AgentMetrics

# Success result
result = AgentResult(
    success=True,
    data={"processed_records": 1000},
    metadata={"source": "api"},
    metrics=AgentMetrics(
        execution_time_ms=250.5,
        records_processed=1000
    )
)

# Error result
error_result = AgentResult(
    success=False,
    error="Database connection failed",
    metadata={"attempted_at": "2025-10-18T10:00:00"}
)
```

---

### AgentMetrics

**Module:** `greenlang.agents.base`

Metrics collected during agent execution.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `execution_time_ms` | `float` | `0.0` | Execution time in milliseconds |
| `input_size` | `int` | `0` | Size of input data |
| `output_size` | `int` | `0` | Size of output data |
| `records_processed` | `int` | `0` | Number of records processed |
| `cache_hits` | `int` | `0` | Number of cache hits |
| `cache_misses` | `int` | `0` | Number of cache misses |
| `custom_metrics` | `Dict[str, float]` | `{}` | Custom agent-specific metrics |

**Example:**
```python
from greenlang.agents import AgentMetrics

metrics = AgentMetrics(
    execution_time_ms=125.5,
    input_size=1024,
    output_size=2048,
    records_processed=100,
    cache_hits=75,
    cache_misses=25,
    custom_metrics={
        "api_calls": 10,
        "retry_count": 2
    }
)
```

---

## greenlang.provenance

The provenance module provides tools for tracking data lineage, execution history, and audit trails.

### ProvenanceRecord

**Module:** `greenlang.provenance.records`

Complete provenance record for audit trails.

#### Class Signature

```python
@dataclass
class ProvenanceRecord:
    record_id: str
    generated_at: str
    environment: Dict[str, Any]
    dependencies: Dict[str, str]
    configuration: Dict[str, Any]
    agent_execution: List[Dict[str, Any]] = field(default_factory=list)
    data_lineage: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    input_file_hash: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert to dictionary.

**Returns:**
- `Dict[str, Any]`: Dictionary representation

**Example:**
```python
from greenlang.provenance import ProvenanceRecord

record = ProvenanceRecord(
    record_id="run-20251018-001",
    generated_at="2025-10-18T10:00:00Z",
    environment={"python_version": "3.11", "platform": "linux"},
    dependencies={"pandas": "2.0.0", "numpy": "1.24.0"},
    configuration={"batch_size": 100}
)

record_dict = record.to_dict()
```

##### `to_json(indent: int = 2) -> str`

Convert to JSON string.

**Parameters:**
- `indent` (int): JSON indentation

**Returns:**
- `str`: JSON string

**Example:**
```python
json_str = record.to_json(indent=2)
print(json_str)
```

##### `save(path: str)`

Save provenance record to file.

**Parameters:**
- `path` (str): File path to save to

**Example:**
```python
record.save("provenance/run-001.json")
```

##### `from_dict(data: Dict[str, Any]) -> ProvenanceRecord` (classmethod)

Create from dictionary.

**Parameters:**
- `data` (Dict[str, Any]): Dictionary with provenance data

**Returns:**
- `ProvenanceRecord`: ProvenanceRecord instance

**Example:**
```python
data = {
    "record_id": "run-001",
    "generated_at": "2025-10-18T10:00:00Z",
    "environment": {},
    "dependencies": {},
    "configuration": {}
}
record = ProvenanceRecord.from_dict(data)
```

##### `from_json(json_str: str) -> ProvenanceRecord` (classmethod)

Create from JSON string.

**Parameters:**
- `json_str` (str): JSON string

**Returns:**
- `ProvenanceRecord`: ProvenanceRecord instance

**Example:**
```python
json_str = '{"record_id": "run-001", ...}'
record = ProvenanceRecord.from_json(json_str)
```

##### `load(path: str) -> ProvenanceRecord` (classmethod)

Load provenance record from file.

**Parameters:**
- `path` (str): File path to load from

**Returns:**
- `ProvenanceRecord`: Loaded provenance record

**Example:**
```python
record = ProvenanceRecord.load("provenance/run-001.json")
```

---

### ProvenanceContext

**Module:** `greenlang.provenance.records`

Runtime provenance tracking context.

#### Class Signature

```python
class ProvenanceContext:
    def __init__(self, name: str = "default", record_id: Optional[str] = None)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"default"` | Context name |
| `record_id` | `Optional[str]` | `None` | Optional record ID (auto-generated if None) |

#### Methods

##### `record_input(source: str, metadata: Optional[Dict[str, Any]] = None)`

Record an input source.

**Parameters:**
- `source` (str): Input source (file path, URL, etc.)
- `metadata` (Optional[Dict[str, Any]]): Optional metadata about input

**Example:**
```python
from greenlang.provenance import ProvenanceContext

ctx = ProvenanceContext("data_pipeline")
ctx.record_input("data/input.csv", {"rows": 10000, "format": "csv"})
```

##### `record_output(destination: str, metadata: Optional[Dict[str, Any]] = None)`

Record an output.

**Parameters:**
- `destination` (str): Output destination
- `metadata` (Optional[Dict[str, Any]]): Optional metadata about output

**Example:**
```python
ctx.record_output("data/output.json", {"rows": 9500, "format": "json"})
```

##### `record_agent_execution(agent_name: str, start_time: str, end_time: str, duration_seconds: float, input_records: int = 0, output_records: int = 0, metadata: Optional[Dict[str, Any]] = None)`

Record agent execution details.

**Parameters:**
- `agent_name` (str): Name of the agent
- `start_time` (str): Start timestamp (ISO 8601)
- `end_time` (str): End timestamp (ISO 8601)
- `duration_seconds` (float): Execution duration
- `input_records` (int): Number of input records processed
- `output_records` (int): Number of output records produced
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

**Example:**
```python
from datetime import datetime

start = datetime.now()
# ... do work ...
end = datetime.now()
duration = (end - start).total_seconds()

ctx.record_agent_execution(
    agent_name="DataValidator",
    start_time=start.isoformat(),
    end_time=end.isoformat(),
    duration_seconds=duration,
    input_records=10000,
    output_records=9500,
    metadata={"errors_found": 500}
)
```

##### `record_validation(validation_results: Dict[str, Any])`

Record validation results.

**Parameters:**
- `validation_results` (Dict[str, Any]): Validation outcome

**Example:**
```python
ctx.record_validation({
    "valid": True,
    "errors": 0,
    "warnings": 3
})
```

##### `set_configuration(config: Dict[str, Any])`

Set configuration snapshot.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary

**Example:**
```python
ctx.set_configuration({
    "batch_size": 100,
    "timeout": 30,
    "retry_count": 3
})
```

##### `add_metadata(key: str, value: Any)`

Add metadata key-value pair.

**Parameters:**
- `key` (str): Metadata key
- `value` (Any): Metadata value

**Example:**
```python
ctx.add_metadata("run_id", "run-001")
ctx.add_metadata("user", "admin")
```

##### `to_record() -> ProvenanceRecord`

Convert context to ProvenanceRecord.

**Returns:**
- `ProvenanceRecord`: ProvenanceRecord with all collected data

**Example:**
```python
record = ctx.to_record()
record.save("provenance.json")
```

##### `finalize(output_path: Optional[str] = None) -> ProvenanceRecord`

Finalize and save provenance record.

**Parameters:**
- `output_path` (Optional[str]): Optional path to save record

**Returns:**
- `ProvenanceRecord`: Final ProvenanceRecord

**Example:**
```python
record = ctx.finalize(output_path="provenance/run-001.json")
```

---

### RunLedger

**Module:** `greenlang.provenance.ledger`

Append-only ledger for tracking all pipeline executions.

#### Class Signature

```python
class RunLedger:
    def __init__(self, ledger_path: Optional[Path] = None)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ledger_path` | `Optional[Path]` | `None` | Path to ledger file (defaults to ~/.greenlang/ledger.jsonl) |

#### Methods

##### `record_run(pipeline: str, inputs: Dict[str, Any], outputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str`

Record a pipeline execution in the ledger.

**Parameters:**
- `pipeline` (str): Pipeline name or reference
- `inputs` (Dict[str, Any]): Input data for the pipeline
- `outputs` (Dict[str, Any]): Output data from the pipeline
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

**Returns:**
- `str`: Run ID (UUID)

**Example:**
```python
from greenlang.provenance import RunLedger

ledger = RunLedger()

run_id = ledger.record_run(
    pipeline="carbon_calculation",
    inputs={"energy_kwh": 1000, "factor": 0.5},
    outputs={"emissions_kg": 500},
    metadata={"duration": 1.5, "backend": "local"}
)

print(f"Recorded run: {run_id}")
```

##### `get_run(run_id: str) -> Optional[Dict[str, Any]]`

Retrieve a specific run by ID.

**Parameters:**
- `run_id` (str): Run UUID

**Returns:**
- `Optional[Dict[str, Any]]`: Run entry or None if not found

**Example:**
```python
run = ledger.get_run(run_id)
if run:
    print(f"Pipeline: {run['pipeline']}")
    print(f"Timestamp: {run['timestamp']}")
```

##### `list_runs(pipeline: Optional[str] = None, limit: int = 100, since: Optional[datetime] = None) -> List[Dict[str, Any]]`

List runs from the ledger.

**Parameters:**
- `pipeline` (Optional[str]): Filter by pipeline name
- `limit` (int): Maximum number of entries to return
- `since` (Optional[datetime]): Only return runs since this timestamp

**Returns:**
- `List[Dict[str, Any]]`: List of run entries

**Example:**
```python
from datetime import datetime, timedelta

# Get recent runs
since_date = datetime.now() - timedelta(days=7)
recent_runs = ledger.list_runs(
    pipeline="carbon_calculation",
    limit=50,
    since=since_date
)

for run in recent_runs:
    print(f"{run['timestamp']}: {run['pipeline']}")
```

##### `find_duplicate_runs(input_hash: str, pipeline: Optional[str] = None) -> List[Dict[str, Any]]`

Find runs with the same input hash (for deduplication).

**Parameters:**
- `input_hash` (str): SHA-256 hash of inputs
- `pipeline` (Optional[str]): Filter by pipeline name

**Returns:**
- `List[Dict[str, Any]]`: List of matching run entries

**Example:**
```python
import hashlib
import json

inputs = {"energy_kwh": 1000}
input_hash = hashlib.sha256(
    json.dumps(inputs, sort_keys=True).encode()
).hexdigest()

duplicates = ledger.find_duplicate_runs(input_hash, "carbon_calculation")
print(f"Found {len(duplicates)} duplicate runs")
```

##### `get_statistics(pipeline: Optional[str] = None, days: int = 30) -> Dict[str, Any]`

Get execution statistics from the ledger.

**Parameters:**
- `pipeline` (Optional[str]): Filter by pipeline name
- `days` (int): Number of days to look back

**Returns:**
- `Dict[str, Any]`: Statistics dictionary

**Example:**
```python
stats = ledger.get_statistics(pipeline="carbon_calculation", days=30)

print(f"Total runs: {stats['total_runs']}")
print(f"Unique inputs: {stats['unique_inputs']}")
print(f"Avg per day: {stats['average_per_day']:.2f}")
```

##### `verify_reproducibility(input_hash: str, output_hash: str, pipeline: str) -> bool`

Verify if outputs are reproducible for given inputs.

**Parameters:**
- `input_hash` (str): Expected input hash
- `output_hash` (str): Expected output hash
- `pipeline` (str): Pipeline name

**Returns:**
- `bool`: True if all runs with same inputs produce same outputs

**Example:**
```python
is_reproducible = ledger.verify_reproducibility(
    input_hash="abc123...",
    output_hash="def456...",
    pipeline="carbon_calculation"
)

if is_reproducible:
    print("Pipeline is reproducible!")
```

##### `export_to_json(output_path: Path, pipeline: Optional[str] = None, days: int = 30) -> Path`

Export ledger entries to JSON file.

**Parameters:**
- `output_path` (Path): Path to output JSON file
- `pipeline` (Optional[str]): Filter by pipeline name
- `days` (int): Number of days to export

**Returns:**
- `Path`: Path to exported file

**Example:**
```python
from pathlib import Path

export_path = ledger.export_to_json(
    output_path=Path("exports/ledger_export.json"),
    pipeline="carbon_calculation",
    days=90
)

print(f"Exported to: {export_path}")
```

---

### Provenance Decorators

**Module:** `greenlang.provenance.decorators`

#### `@traced`

Decorator to automatically track provenance for functions.

**Parameters:**
- `record_id` (Optional[str]): Optional custom record ID
- `save_path` (Optional[str]): Optional path to save provenance record
- `track_inputs` (bool): Whether to track input arguments
- `track_outputs` (bool): Whether to track output results

**Example:**
```python
from greenlang.provenance.decorators import traced

@traced(save_path="provenance/process_data.json")
def process_data(input_file, config):
    # Load data
    data = load_file(input_file)

    # Process
    result = transform(data, config)

    return result

# Provenance automatically recorded when called
result = process_data("data.csv", {"batch_size": 100})
```

#### `@track_provenance`

Decorator to track provenance for class methods.

**Parameters:**
- `context_attr` (str): Name of the context attribute on the class
- `save_on_completion` (bool): Whether to save provenance when method completes

**Example:**
```python
from greenlang.provenance import ProvenanceContext
from greenlang.provenance.decorators import track_provenance

class DataPipeline:
    def __init__(self):
        self._provenance_context = ProvenanceContext("pipeline")

    @track_provenance()
    def load_data(self, path):
        # Load data - automatically tracked
        return data

    @track_provenance()
    def transform_data(self, data):
        # Transform data - automatically tracked
        return transformed

    def finalize(self):
        # Save provenance record
        self._provenance_context.finalize("provenance.json")
```

#### `provenance_tracker` Context Manager

Context manager for provenance tracking.

**Parameters:**
- `name` (str): Operation name
- `record_id` (Optional[str]): Optional record ID
- `save_path` (Optional[str]): Optional path to save provenance

**Example:**
```python
from greenlang.provenance.decorators import provenance_tracker

with provenance_tracker("my_operation", save_path="provenance.json") as ctx:
    # Do work
    data = load_data("input.csv")
    ctx.record_input("input.csv", {"rows": len(data)})

    # Process
    result = process(data)
    ctx.record_output("output.csv", {"rows": len(result)})

# Provenance automatically saved
```

---

## greenlang.validation

The validation module provides a flexible framework for multi-layer data validation.

### ValidationFramework

**Module:** `greenlang.validation.framework`

Core validation framework supporting multiple validation strategies.

#### Class Signature

```python
class ValidationFramework:
    def __init__(self)
```

#### Methods

##### `add_validator(name: str, validator_func: Callable[[Any], ValidationResult], config: Optional[Validator] = None)`

Register a validator function.

**Parameters:**
- `name` (str): Unique name for the validator
- `validator_func` (Callable): Function that takes data and returns ValidationResult
- `config` (Optional[Validator]): Optional validator configuration

**Example:**
```python
from greenlang.validation import ValidationFramework, ValidationResult, ValidationError

def check_required_fields(data):
    result = ValidationResult(valid=True)
    required = ['name', 'email', 'age']

    for field in required:
        if field not in data:
            result.add_error(ValidationError(
                field=field,
                message=f"Required field '{field}' is missing",
                severity=ValidationSeverity.ERROR,
                validator="required_fields"
            ))

    return result

framework = ValidationFramework()
framework.add_validator("required_fields", check_required_fields)
```

##### `validate(data: Any, validators: Optional[List[str]] = None, stop_on_error: bool = False) -> ValidationResult`

Validate data using registered validators.

**Parameters:**
- `data` (Any): Data to validate
- `validators` (Optional[List[str]]): List of validator names to use (all if None)
- `stop_on_error` (bool): Stop on first error

**Returns:**
- `ValidationResult`: Aggregated validation result

**Example:**
```python
framework = ValidationFramework()
framework.add_validator("schema", schema_validator)
framework.add_validator("business_rules", rules_validator)

data = {"name": "John", "age": 30}
result = framework.validate(data)

if not result.valid:
    print(result.get_summary())
    for error in result.errors:
        print(f"  - {error}")
```

##### `validate_batch(data_list: List[Any], validators: Optional[List[str]] = None) -> List[ValidationResult]`

Validate a batch of data items.

**Parameters:**
- `data_list` (List[Any]): List of data items to validate
- `validators` (Optional[List[str]]): List of validator names to use

**Returns:**
- `List[ValidationResult]`: List of validation results

**Example:**
```python
data_batch = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": -5},  # Invalid
    {"age": 30}  # Missing name
]

results = framework.validate_batch(data_batch)

for i, result in enumerate(results):
    print(f"Record {i}: {result.get_summary()}")
```

##### `get_validation_summary(results: List[ValidationResult]) -> Dict[str, Any]`

Get summary statistics for batch validation.

**Parameters:**
- `results` (List[ValidationResult]): List of validation results

**Returns:**
- `Dict[str, Any]`: Summary statistics

**Example:**
```python
results = framework.validate_batch(data_batch)
summary = framework.get_validation_summary(results)

print(f"Total: {summary['total']}")
print(f"Passed: {summary['passed']}")
print(f"Failed: {summary['failed']}")
print(f"Pass rate: {summary['pass_rate']}%")
```

##### `add_pre_validator(validator_func: Callable)`

Add a pre-validation hook.

**Example:**
```python
def log_validation_start(data):
    print(f"Starting validation of {len(data)} records")

framework.add_pre_validator(log_validation_start)
```

##### `add_post_validator(validator_func: Callable)`

Add a post-validation hook.

**Example:**
```python
def log_validation_end(data, result):
    print(f"Validation complete: {result.get_summary()}")

framework.add_post_validator(log_validation_end)
```

---

### SchemaValidator

**Module:** `greenlang.validation.schema`

JSON Schema validator with enhanced error reporting.

#### Class Signature

```python
class SchemaValidator:
    def __init__(self, schema: Dict[str, Any])
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `Dict[str, Any]` | Required | JSON Schema dictionary |

#### Methods

##### `validate(data: Any) -> ValidationResult`

Validate data against schema.

**Parameters:**
- `data` (Any): Data to validate

**Returns:**
- `ValidationResult`: Validation result with errors if any

**Example:**
```python
from greenlang.validation import SchemaValidator

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

validator = SchemaValidator(schema)

# Valid data
result = validator.validate({"name": "John", "age": 30, "email": "john@example.com"})
assert result.valid

# Invalid data
result = validator.validate({"name": "John", "age": -5})
assert not result.valid
print(result.errors)
```

##### `from_file(schema_path: str) -> SchemaValidator` (classmethod)

Create validator from JSON schema file.

**Parameters:**
- `schema_path` (str): Path to JSON schema file

**Returns:**
- `SchemaValidator`: SchemaValidator instance

**Example:**
```python
validator = SchemaValidator.from_file("schemas/person.json")
result = validator.validate(data)
```

---

### RulesEngine

**Module:** `greenlang.validation.rules`

Business rules validation engine.

#### Class Signature

```python
class RulesEngine:
    def __init__(self)
```

#### Methods

##### `add_rule(rule: Rule)`

Add a validation rule.

**Parameters:**
- `rule` (Rule): Rule to add

**Example:**
```python
from greenlang.validation import RulesEngine, Rule, RuleOperator

engine = RulesEngine()

# Age rule
age_rule = Rule(
    name="check_age",
    field="age",
    operator=RuleOperator.GREATER_EQUAL,
    value=18,
    message="Age must be 18 or older",
    severity=ValidationSeverity.ERROR
)
engine.add_rule(age_rule)

# Email rule
email_rule = Rule(
    name="check_email",
    field="email",
    operator=RuleOperator.REGEX,
    value=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
    message="Invalid email format"
)
engine.add_rule(email_rule)
```

##### `add_rule_set(rule_set: RuleSet)`

Add a rule set.

**Parameters:**
- `rule_set` (RuleSet): RuleSet to add

**Example:**
```python
from greenlang.validation import RuleSet

user_rules = RuleSet(
    name="user_validation",
    description="User data validation rules",
    rules=[age_rule, email_rule]
)

engine.add_rule_set(user_rules)
```

##### `validate(data: Dict[str, Any], rule_set_name: Optional[str] = None) -> ValidationResult`

Validate data against rules.

**Parameters:**
- `data` (Dict[str, Any]): Data to validate
- `rule_set_name` (Optional[str]): Optional rule set name to use

**Returns:**
- `ValidationResult`: Validation result

**Example:**
```python
data = {"age": 25, "email": "user@example.com"}
result = engine.validate(data)

if not result.valid:
    for error in result.errors:
        print(f"{error.field}: {error.message}")
```

##### `load_rules_from_dict(rules_config: List[Dict[str, Any]])`

Load rules from configuration dictionary.

**Parameters:**
- `rules_config` (List[Dict[str, Any]]): List of rule configurations

**Example:**
```python
rules_config = [
    {
        "name": "check_age",
        "field": "age",
        "operator": ">=",
        "value": 18,
        "message": "Must be 18 or older"
    },
    {
        "name": "check_score",
        "field": "score",
        "operator": "<=",
        "value": 100,
        "message": "Score cannot exceed 100"
    }
]

engine.load_rules_from_dict(rules_config)
```

#### Supported Operators

- `==` (EQUALS): Equality check
- `!=` (NOT_EQUALS): Inequality check
- `>` (GREATER_THAN): Greater than
- `>=` (GREATER_EQUAL): Greater than or equal
- `<` (LESS_THAN): Less than
- `<=` (LESS_EQUAL): Less than or equal
- `in` (IN): Value in list
- `not_in` (NOT_IN): Value not in list
- `contains` (CONTAINS): String contains substring
- `regex` (REGEX): Regular expression match
- `is_null` (IS_NULL): Value is null
- `not_null` (NOT_NULL): Value is not null

---

### ValidationResult

**Module:** `greenlang.validation.framework`

Result of validation process.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `valid` | `bool` | Required | Whether validation passed |
| `errors` | `List[ValidationError]` | `[]` | List of errors |
| `warnings` | `List[ValidationError]` | `[]` | List of warnings |
| `info` | `List[ValidationError]` | `[]` | List of info messages |
| `metadata` | `Dict[str, Any]` | `{}` | Additional metadata |
| `timestamp` | `datetime` | Auto | Validation timestamp |

#### Methods

##### `add_error(error: ValidationError)`

Add an error to the result.

##### `merge(other: ValidationResult)`

Merge another validation result into this one.

##### `get_summary() -> str`

Get a summary of validation results.

**Example:**
```python
result = ValidationResult(valid=True)

# Add errors
result.add_error(ValidationError(
    field="age",
    message="Age is too low",
    severity=ValidationSeverity.ERROR,
    validator="age_check"
))

# Get summary
print(result.get_summary())  # "Validation FAILED: 1 errors, 0 warnings"

# Error count
print(f"Errors: {result.get_error_count()}")
print(f"Warnings: {result.get_warning_count()}")
```

---

## greenlang.io

The IO module provides multi-format data reading and writing capabilities.

### DataReader

**Module:** `greenlang.io.readers`

Multi-format data reader with automatic format detection.

#### Class Signature

```python
class DataReader:
    def __init__(self, default_encoding: str = "utf-8")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_encoding` | `str` | `"utf-8"` | Default text encoding |

#### Supported Formats

- JSON (.json)
- CSV (.csv)
- TSV (.tsv)
- TXT (.txt)
- YAML (.yaml, .yml) - requires PyYAML
- Excel (.xlsx) - requires openpyxl
- Excel (.xls) - requires xlrd
- Parquet (.parquet) - requires pyarrow
- XML (.xml) - requires lxml

#### Methods

##### `read(file_path: Union[str, Path], **kwargs) -> Any`

Read data from file with automatic format detection.

**Parameters:**
- `file_path` (Union[str, Path]): Path to file
- `**kwargs`: Format-specific options

**Returns:**
- `Any`: Loaded data (format depends on file type)

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If format not supported

**Example:**
```python
from greenlang.io import DataReader

reader = DataReader()

# JSON
data = reader.read("data.json")

# CSV with custom delimiter
data = reader.read("data.csv", csv_delimiter=";", csv_has_header=True)

# Excel with specific sheet
data = reader.read("data.xlsx", sheet_name="Sheet2")

# YAML
config = reader.read("config.yaml")
```

##### `get_supported_formats() -> List[str]`

Get list of supported file formats.

**Returns:**
- `List[str]`: List of supported extensions

**Example:**
```python
reader = DataReader()
formats = reader.get_supported_formats()
print(f"Supported formats: {formats}")
```

#### Format-Specific Options

**CSV/TSV:**
- `csv_delimiter` (str): Delimiter character (default: ",")
- `csv_has_header` (bool): Whether CSV has header row (default: True)

**Excel:**
- `sheet_name` (Union[int, str]): Sheet name or index (default: 0)

**Example:**
```python
# CSV without header
data = reader.read("data.csv", csv_has_header=False)

# TSV (tab-separated)
data = reader.read("data.tsv")

# Excel specific sheet
data = reader.read("workbook.xlsx", sheet_name="Q4_Sales")
```

---

### DataWriter

**Module:** `greenlang.io.writers`

Multi-format data writer.

#### Class Signature

```python
class DataWriter:
    def __init__(self, default_encoding: str = "utf-8")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_encoding` | `str` | `"utf-8"` | Default text encoding |

#### Supported Formats

- JSON (.json)
- CSV (.csv)
- TSV (.tsv)
- TXT (.txt)
- YAML (.yaml, .yml) - requires PyYAML
- Excel (.xlsx) - requires openpyxl
- Parquet (.parquet) - requires pyarrow

#### Methods

##### `write(data: Any, file_path: Union[str, Path], **kwargs)`

Write data to file with automatic format detection.

**Parameters:**
- `data` (Any): Data to write
- `file_path` (Union[str, Path]): Path to output file
- `**kwargs`: Format-specific options

**Raises:**
- `ValueError`: If format not supported

**Example:**
```python
from greenlang.io import DataWriter

writer = DataWriter()

# JSON with custom indent
data = {"name": "John", "age": 30}
writer.write(data, "output.json", indent=4)

# CSV from list of dictionaries
records = [
    {"name": "Alice", "score": 95},
    {"name": "Bob", "score": 87}
]
writer.write(records, "output.csv")

# Excel with auto-width columns
writer.write(records, "output.xlsx", auto_width=True)

# YAML
writer.write(config, "config.yaml")
```

##### `get_supported_formats() -> List[str]`

Get list of supported file formats.

**Returns:**
- `List[str]`: List of supported extensions

#### Format-Specific Options

**JSON:**
- `indent` (int): JSON indentation (default: 2)

**CSV/TSV:**
- `csv_delimiter` (str): Delimiter character (default: ",")

**Excel:**
- `sheet_name` (str): Sheet name (default: "Sheet1")
- `auto_width` (bool): Auto-size columns (default: True)

**Example:**
```python
# JSON compact
writer.write(data, "compact.json", indent=None)

# CSV with semicolon delimiter
writer.write(records, "data.csv", csv_delimiter=";")

# Excel with custom sheet name
writer.write(records, "report.xlsx", sheet_name="Sales_Data")
```

---

## Convenience Functions

### `read_file(file_path: Union[str, Path], **kwargs) -> Any`

**Module:** `greenlang.io.readers`

Convenience function to read a file.

**Example:**
```python
from greenlang.io import read_file

data = read_file("data.json")
csv_data = read_file("data.csv", csv_delimiter=";")
```

### `write_file(data: Any, file_path: Union[str, Path], **kwargs)`

**Module:** `greenlang.io.writers`

Convenience function to write data to file.

**Example:**
```python
from greenlang.io import write_file

write_file(data, "output.json", indent=2)
write_file(records, "output.csv")
```

---

## Common Patterns

### Pattern 1: Building a Custom Agent

```python
from greenlang.agents import BaseAgent, AgentConfig, AgentResult

class MyCustomAgent(BaseAgent):
    def __init__(self):
        config = AgentConfig(
            name="CustomAgent",
            description="Performs custom processing",
            version="1.0.0"
        )
        super().__init__(config)

    def validate_input(self, input_data):
        # Validate required fields
        if "data" not in input_data:
            self.logger.error("Missing 'data' field")
            return False
        return True

    def execute(self, input_data):
        # Process data
        data = input_data["data"]
        result = self.process(data)

        return AgentResult(
            success=True,
            data={"result": result},
            metadata={"processed_count": len(data)}
        )

    def process(self, data):
        # Custom processing logic
        return [item * 2 for item in data]

# Use the agent
agent = MyCustomAgent()
result = agent.run({"data": [1, 2, 3, 4, 5]})
print(result.data)  # {"result": [2, 4, 6, 8, 10]}
```

### Pattern 2: Building a Calculator Agent

```python
from greenlang.agents import BaseCalculator, CalculatorConfig

class EmissionCalculator(BaseCalculator):
    def __init__(self):
        config = CalculatorConfig(
            name="EmissionCalculator",
            precision=4,
            enable_caching=True
        )
        super().__init__(config)

    def validate_calculation_inputs(self, inputs):
        required = ['energy_kwh', 'emission_factor']
        return all(field in inputs for field in required)

    def calculate(self, inputs):
        energy = inputs['energy_kwh']
        factor = inputs['emission_factor']

        # Calculate emissions
        emissions = energy * factor

        # Record step
        self.add_calculation_step(
            step_name="calculate_emissions",
            formula="energy_kwh * emission_factor",
            inputs=inputs,
            result=emissions,
            units="kgCO2e"
        )

        return emissions

# Use the calculator
calculator = EmissionCalculator()
result = calculator.run({
    "inputs": {
        "energy_kwh": 1000,
        "emission_factor": 0.5
    }
})

print(f"Emissions: {result.result_value} kgCO2e")
print(f"Cached: {result.cached}")
```

### Pattern 3: Building a Reporter Agent

```python
from greenlang.agents import BaseReporter, ReporterConfig, ReportSection

class SalesReporter(BaseReporter):
    def __init__(self):
        config = ReporterConfig(
            name="SalesReport",
            output_format="html",
            include_summary=True
        )
        super().__init__(config)

    def aggregate_data(self, input_data):
        sales = input_data['sales']
        return {
            'total_sales': sum(sales),
            'average_sale': sum(sales) / len(sales),
            'max_sale': max(sales),
            'min_sale': min(sales),
            'count': len(sales)
        }

    def build_sections(self, aggregated_data):
        sections = []

        # Summary section
        sections.append(ReportSection(
            title="Executive Summary",
            content=f"Total Sales: ${aggregated_data['total_sales']:,.2f}",
            level=2,
            section_type="text"
        ))

        # Details table
        sections.append(ReportSection(
            title="Sales Metrics",
            content=[
                {"Metric": "Total", "Value": f"${aggregated_data['total_sales']:,.2f}"},
                {"Metric": "Average", "Value": f"${aggregated_data['average_sale']:,.2f}"},
                {"Metric": "Maximum", "Value": f"${aggregated_data['max_sale']:,.2f}"},
                {"Metric": "Minimum", "Value": f"${aggregated_data['min_sale']:,.2f}"}
            ],
            level=2,
            section_type="table"
        ))

        return sections

# Use the reporter
reporter = SalesReporter()
result = reporter.run({"sales": [100, 200, 150, 300, 250]})

# Get HTML report
html_report = reporter.render_html()
with open("sales_report.html", "w") as f:
    f.write(html_report)
```

### Pattern 4: Multi-Layer Validation

```python
from greenlang.validation import ValidationFramework, SchemaValidator, RulesEngine, Rule, RuleOperator

# Define schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age"]
}

# Create validators
schema_validator = SchemaValidator(schema)

rules_engine = RulesEngine()
rules_engine.add_rule(Rule(
    name="check_age",
    field="age",
    operator=RuleOperator.GREATER_EQUAL,
    value=18,
    message="Must be 18 or older"
))

# Create framework
framework = ValidationFramework()
framework.add_validator("schema", schema_validator.validate)
framework.add_validator("business_rules", rules_engine.validate)

# Validate data
data = {"name": "John", "age": 25, "email": "john@example.com"}
result = framework.validate(data)

if result.valid:
    print("Validation passed!")
else:
    print("Validation failed:")
    for error in result.errors:
        print(f"  - {error}")
```

### Pattern 5: Provenance Tracking

```python
from greenlang.provenance import ProvenanceContext

# Create context
ctx = ProvenanceContext("data_pipeline")

# Record input
ctx.record_input("data/input.csv", {"rows": 10000})

# Set configuration
ctx.set_configuration({
    "batch_size": 100,
    "validation_enabled": True
})

# Process data (record agent execution)
from datetime import datetime

start = datetime.now()
# ... do processing ...
end = datetime.now()

ctx.record_agent_execution(
    agent_name="DataProcessor",
    start_time=start.isoformat(),
    end_time=end.isoformat(),
    duration_seconds=(end - start).total_seconds(),
    input_records=10000,
    output_records=9500
)

# Record output
ctx.record_output("data/output.json", {"rows": 9500})

# Finalize and save
record = ctx.finalize(output_path="provenance/pipeline.json")
print(f"Provenance saved: {record.record_id}")
```

### Pattern 6: Reading and Writing Data

```python
from greenlang.io import DataReader, DataWriter

# Read various formats
reader = DataReader()

json_data = reader.read("config.json")
csv_data = reader.read("data.csv", csv_has_header=True)
excel_data = reader.read("report.xlsx", sheet_name="Q4")

# Write various formats
writer = DataWriter()

writer.write(json_data, "output.json", indent=2)
writer.write(csv_data, "output.csv")
writer.write(excel_data, "output.xlsx", sheet_name="Results")

# Batch processing
for file_path in ["data1.csv", "data2.csv", "data3.csv"]:
    data = reader.read(file_path)
    processed = process_data(data)
    output_path = file_path.replace(".csv", "_processed.json")
    writer.write(processed, output_path)
```

---

## Type Definitions

The framework uses TypedDict for type safety. Key types are defined in `greenlang.agents.types`:

### Core Types

- `FuelInput` / `FuelOutput`
- `BoilerInput` / `BoilerOutput`
- `GridFactorInput` / `GridFactorOutput`
- `CarbonInput` / `CarbonOutput`
- `IntensityInput` / `IntensityOutput`
- `BenchmarkInput` / `BenchmarkOutput`
- `RecommendationInput` / `RecommendationOutput`
- `ReportInput` / `ReportOutput`
- `WorkflowInput` / `WorkflowOutput`
- `PortfolioInput` / `PortfolioOutput`

---

## Error Handling

All framework classes follow consistent error handling patterns:

```python
try:
    agent = MyAgent()
    result = agent.run(input_data)

    if result.success:
        # Process successful result
        print(result.data)
    else:
        # Handle agent-level error
        print(f"Error: {result.error}")

except FileNotFoundError as e:
    # Handle missing files
    print(f"File not found: {e}")

except ValueError as e:
    # Handle invalid inputs
    print(f"Invalid input: {e}")

except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
```

---

## Performance Considerations

### Caching

Calculators automatically cache results based on input hashes:

```python
calculator = CarbonCalculator()

# First call - calculated
result1 = calculator.run({"inputs": {"energy": 1000}})
print(result1.cached)  # False

# Second call - cached
result2 = calculator.run({"inputs": {"energy": 1000}})
print(result2.cached)  # True

# Clear cache if needed
calculator.clear_cache()
```

### Batch Processing

Use batch methods for better performance:

```python
framework = ValidationFramework()
results = framework.validate_batch(data_list)

# Get summary statistics
summary = framework.get_validation_summary(results)
print(f"Pass rate: {summary['pass_rate']}%")
```

---

## Configuration Best Practices

1. **Use configuration objects** for better type safety:
```python
config = AgentConfig(
    name="MyAgent",
    version="1.0.0",
    enable_metrics=True
)
agent = MyAgent(config)
```

2. **Externalize configuration** using YAML or JSON:
```python
import yaml

with open("config.yaml") as f:
    config_dict = yaml.safe_load(f)

config = AgentConfig(**config_dict)
```

3. **Use environment-specific configs**:
```python
import os

env = os.getenv("ENV", "dev")
config = AgentConfig.from_file(f"config/{env}.yaml")
```

---

## Logging

The framework uses Python's standard logging module:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Agent will use configured logger
agent = MyAgent()
agent.run(input_data)  # Logs execution details
```

---

## Appendix: Version History

- **1.0.0** (2025-10-18): Initial API reference release

---

**End of API Reference**
