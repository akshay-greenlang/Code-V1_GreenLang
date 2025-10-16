# GreenLang Agent Framework Documentation (v0.3.0)

Complete documentation for the GreenLang agent framework - build production-ready climate intelligence agents in minutes.

## Quick Navigation

### Getting Started
- **[Quick Start Guide](QUICK_START.md)** - Get running in 5 minutes
- **[Installation](../README.md#installation)** - Installation instructions
- **[Your First Agent](QUICK_START.md#your-first-agent-in-5-minutes)** - Create an agent in 5 minutes

### Core Documentation
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all agent classes (SDK)
- **[Migration Guide](MIGRATION_GUIDE.md)** - Migrate from custom code to framework
- **[Architecture Guide](ARCHITECTURE.md)** - Framework design and internals

### Examples
- **[Examples Directory](examples/)** - 10 complete, runnable examples
- **[Examples Index](examples/README.md)** - Overview of all examples

---

## Framework Overview

The GreenLang agent framework provides four specialized base classes:

```
BaseAgent
│
├── BaseDataProcessor    → Batch processing & ETL
├── BaseCalculator       → Mathematical computations
└── BaseReporter        → Data aggregation & reporting
```

### BaseAgent
Foundation class for all agents with:
- Lifecycle management (init, validate, execute, cleanup)
- Automatic metrics collection
- Provenance tracking hooks
- Resource loading
- Error handling

**Use for:** Custom agents with any logic

**Documentation:** [API Reference - BaseAgent](API_REFERENCE.md#baseagent)

**Examples:**
- [Example 01: Basic Agent](examples/example_01_basic_agent.py)

---

### BaseDataProcessor
Specialized class for batch data transformation:
- Parallel processing with configurable workers
- Record-level validation and transformation
- Error collection and reporting
- Progress tracking
- Automatic metrics

**Use for:** ETL, data transformation, batch processing

**Documentation:** [API Reference - BaseDataProcessor](API_REFERENCE.md#basedataprocessor)

**Examples:**
- [Example 02: Data Processor](examples/example_02_data_processor.py)
- [Example 05: Batch Processing](examples/example_05_batch_processing.py)
- [Example 06: Parallel Processing](examples/example_06_parallel_processing.py)

---

### BaseCalculator
Specialized class for mathematical operations:
- High-precision decimal arithmetic
- Deterministic calculations
- Result caching for performance
- Calculation step tracking
- Unit conversion support

**Use for:** Calculations, formulas, mathematical models

**Documentation:** [API Reference - BaseCalculator](API_REFERENCE.md#basecalculator)

**Examples:**
- [Example 03: Calculator Agent](examples/example_03_calculator.py)
- [Example 08: With Provenance](examples/example_08_with_provenance.py)

---

### BaseReporter
Specialized class for reporting:
- Multi-format output (Markdown, HTML, JSON, Excel)
- Data aggregation utilities
- Template-based reporting
- Summary generation
- Section management

**Use for:** Reports, dashboards, summaries

**Documentation:** [API Reference - BaseReporter](API_REFERENCE.md#basereporter)

**Examples:**
- [Example 04: Reporter Agent](examples/example_04_reporter.py)
- [Example 09: Multi-Format Reports](examples/example_09_multi_format_reports.py)

---

## Documentation by Use Case

### For First-Time Users
1. **[Quick Start Guide](QUICK_START.md)** - Your first agent in 5 minutes
2. **[Example 01: Basic Agent](examples/example_01_basic_agent.py)** - Simple greeting agent
3. **[Example 02: Data Processor](examples/example_02_data_processor.py)** - Temperature converter

### For Developers Building Agents
1. **[API Reference](API_REFERENCE.md)** - Complete API documentation
2. **[Examples Directory](examples/)** - 10 runnable examples
3. **[Common Patterns](QUICK_START.md#common-patterns)** - Agent composition, validation, etc.

### For Teams Migrating Code
1. **[Migration Guide](MIGRATION_GUIDE.md)** - Step-by-step migration process
2. **[Before/After Examples](MIGRATION_GUIDE.md#before-and-after-comparisons)** - See the transformation
3. **[CBAM Case Study](examples/example_10_cbam_pipeline.py)** - 680→230 lines (66% reduction)

### For Production Deployments
1. **[Architecture Guide](ARCHITECTURE.md)** - Design principles and patterns
2. **[Example 08: Provenance](examples/example_08_with_provenance.py)** - Regulatory compliance
3. **[Example 06: Parallel Processing](examples/example_06_parallel_processing.py)** - Performance optimization

---

## Key Features

### ✓ Lifecycle Management
Every agent has a consistent lifecycle:
```
initialize() → validate_input() → preprocess() → execute() → postprocess() → cleanup()
```

### ✓ Automatic Metrics
Track execution statistics automatically:
- Execution count
- Success/failure rates
- Average execution time
- Custom counters and timers

### ✓ Provenance Tracking
Add audit trails with lifecycle hooks:
```python
agent.add_pre_hook(record_inputs)
agent.add_post_hook(record_outputs)
```

### ✓ Error Handling
Built-in error handling with detailed reporting:
- Input validation errors
- Processing errors
- Error collection (batch processing)
- Error statistics

### ✓ Configuration Management
Type-safe configuration with Pydantic:
```python
config = CalculatorConfig(
    name="MyCalculator",
    precision=6,
    enable_caching=True
)
```

---

## Code Examples

### Simple Agent
```python
from greenlang.agents import BaseAgent, AgentConfig, AgentResult

class HelloAgent(BaseAgent):
    def __init__(self):
        config = AgentConfig(name="HelloAgent", description="Greet users")
        super().__init__(config)

    def execute(self, input_data):
        name = input_data.get('name', 'World')
        return AgentResult(
            success=True,
            data={"greeting": f"Hello, {name}!"}
        )

# Use it
agent = HelloAgent()
result = agent.run({"name": "Alice"})
print(result.data['greeting'])  # "Hello, Alice!"
```

### Data Processor
```python
from greenlang.agents import BaseDataProcessor, DataProcessorConfig

class TemperatureConverter(BaseDataProcessor):
    def __init__(self):
        config = DataProcessorConfig(
            name="TempConverter",
            batch_size=100,
            parallel_workers=4
        )
        super().__init__(config)

    def process_record(self, record):
        fahrenheit = record['temp_f']
        celsius = (fahrenheit - 32) * 5 / 9
        record['temp_c'] = round(celsius, 2)
        return record

# Use it
processor = TemperatureConverter()
result = processor.run({
    "records": [
        {"id": 1, "temp_f": 32},
        {"id": 2, "temp_f": 68},
    ]
})
print(f"Processed {result.records_processed} records")
```

### Calculator
```python
from greenlang.agents import BaseCalculator, CalculatorConfig

class EmissionsCalculator(BaseCalculator):
    def __init__(self):
        config = CalculatorConfig(
            name="EmissionsCalc",
            precision=4,
            enable_caching=True
        )
        super().__init__(config)

    def calculate(self, inputs):
        kwh = inputs['electricity_kwh']
        factor = inputs.get('emission_factor', 0.5)
        emissions = kwh * factor

        self.add_calculation_step(
            step_name="Calculate Emissions",
            formula="kwh × factor",
            inputs={'kwh': kwh, 'factor': factor},
            result=emissions,
            units="kg CO2"
        )

        return emissions

# Use it
calculator = EmissionsCalculator()
result = calculator.run({
    "inputs": {"electricity_kwh": 1000, "emission_factor": 0.45}
})
print(f"Emissions: {result.result_value:.2f} kg CO2")
```

### Reporter
```python
from greenlang.agents import BaseReporter, ReporterConfig, ReportSection

class EnergyReporter(BaseReporter):
    def __init__(self):
        config = ReporterConfig(
            name="Energy Report",
            output_format='markdown'
        )
        super().__init__(config)

    def aggregate_data(self, input_data):
        readings = input_data['readings']
        return {
            'total_kwh': sum(r['kwh'] for r in readings),
            'avg_kwh': sum(r['kwh'] for r in readings) / len(readings)
        }

    def build_sections(self, aggregated_data):
        return [
            ReportSection(
                title="Summary",
                content=f"Total: {aggregated_data['total_kwh']} kWh",
                section_type="text"
            )
        ]

# Use it
reporter = EnergyReporter()
result = reporter.run({
    "readings": [{"kwh": 100}, {"kwh": 150}, {"kwh": 120}]
})
print(result.data['report'])
```

---

## Performance Benefits

### Code Reduction
- **Basic calculations:** 250 lines → 50 lines (80% reduction)
- **Data pipelines:** 400 lines → 80 lines (80% reduction)
- **CBAM compliance:** 680 lines → 230 lines (66% reduction)

### Built-in Features
Instead of building yourself:
- ✓ Validation
- ✓ Error handling
- ✓ Metrics tracking
- ✓ Provenance tracking
- ✓ Batch processing
- ✓ Parallel execution
- ✓ Progress bars
- ✓ Caching
- ✓ Logging

### Maintenance
- Framework updates improve all agents automatically
- No need to maintain boilerplate code
- Community-driven improvements
- Regular emission factor updates

---

## Best Practices

### 1. Use Appropriate Base Class
- **BaseAgent:** Generic agents with custom logic
- **BaseDataProcessor:** Batch transformation
- **BaseCalculator:** Mathematical calculations
- **BaseReporter:** Report generation

### 2. Implement Validation
```python
def validate_input(self, input_data):
    if 'required_field' not in input_data:
        self.logger.error("Missing required field")
        return False
    return True
```

### 3. Add Calculation Steps (Calculators)
```python
self.add_calculation_step(
    step_name="Step Name",
    formula="a + b",
    inputs={"a": 1, "b": 2},
    result=3,
    units="units"
)
```

### 4. Use Configuration Objects
```python
config = DataProcessorConfig(
    name="MyProcessor",
    batch_size=1000,
    parallel_workers=4,
    enable_progress=True
)
```

### 5. Track Custom Metrics
```python
self.stats.increment('records_processed')
self.stats.add_time('external_api_time', 125.5)
```

---

## Troubleshooting

### Import Errors
```bash
pip install greenlang-cli==0.3.0
```

### Missing Progress Bars
```bash
pip install tqdm
```

### Missing Excel Export
```bash
pip install openpyxl
```

### Agent Not Executing
- Check `validate_input()` returns True
- Check `config.enabled` is True
- Review logs for error messages

### Performance Issues
- Enable parallel processing: `parallel_workers=4`
- Increase batch size: `batch_size=1000`
- Enable caching: `enable_caching=True`

---

## Community & Support

### Get Help
- **Discord:** [Join our community](https://discord.gg/greenlang)
- **GitHub Issues:** [Report bugs](https://github.com/greenlang/greenlang/issues)
- **Documentation:** You're reading it!

### Contributing
- Found a bug? Open an issue
- Have an improvement? Submit a PR
- Want to help? Join our Discord

---

## Version Information

- **Framework Version:** 0.3.0
- **Release Date:** October 2025
- **Python Compatibility:** 3.8+
- **Dependencies:** pydantic
- **Optional:** tqdm (progress bars), openpyxl (Excel export)

---

## What's Next?

After exploring the documentation:

1. **Try the Quick Start:** [QUICK_START.md](QUICK_START.md)
2. **Run an Example:** [examples/example_01_basic_agent.py](examples/example_01_basic_agent.py)
3. **Read the API Reference:** [API_REFERENCE.md](API_REFERENCE.md)
4. **Migrate Your Code:** [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
5. **Join the Community:** [Discord](https://discord.gg/greenlang)

---

**GreenLang v0.3.0 - The Climate Intelligence Platform**

*Build production-ready climate intelligence agents in minutes, not weeks.*

[GitHub](https://github.com/greenlang/greenlang) | [Documentation](https://greenlang.io/docs) | [Discord](https://discord.gg/greenlang)
