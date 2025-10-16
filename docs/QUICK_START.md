# GreenLang Framework v0.3.0 - Quick Start Guide

Get up and running with the GreenLang agent framework in 5 minutes!

## Table of Contents
- [Installation](#installation)
- [Your First Agent in 5 Minutes](#your-first-agent-in-5-minutes)
- [Agent Types Overview](#agent-types-overview)
- [Basic Examples](#basic-examples)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Install GreenLang

```bash
# Install from PyPI
pip install greenlang-cli==0.3.0

# Verify installation
python -c "from greenlang.agents import BaseAgent; print('GreenLang installed successfully!')"
```

### Optional Dependencies

For advanced features, install optional dependencies:

```bash
# For parallel processing with progress bars
pip install tqdm

# For Excel report generation
pip install openpyxl

# For decimal arithmetic (included in Python standard library)
# No additional installation needed
```

---

## Your First Agent in 5 Minutes

Let's create a simple carbon emissions calculator agent!

### Step 1: Create Your Agent Class

```python
from greenlang.agents import BaseCalculator, CalculatorConfig
from typing import Dict, Any

class SimpleCarbonCalculator(BaseCalculator):
    """Calculate carbon emissions from electricity consumption."""

    def __init__(self):
        config = CalculatorConfig(
            name="SimpleCarbonCalculator",
            description="Calculate CO2 emissions from electricity",
            precision=2
        )
        super().__init__(config)

    def calculate(self, inputs: Dict[str, Any]) -> float:
        """
        Calculate emissions: kWh × emission_factor = kg CO2
        """
        kwh = inputs['electricity_kwh']
        emission_factor = inputs.get('emission_factor', 0.5)  # Default: 0.5 kg CO2/kWh

        # Log the calculation step for transparency
        self.add_calculation_step(
            step_name="Calculate Emissions",
            formula="electricity_kwh × emission_factor",
            inputs={"kwh": kwh, "factor": emission_factor},
            result=kwh * emission_factor,
            units="kg CO2"
        )

        return kwh * emission_factor

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Ensure required inputs are present and valid."""
        if 'electricity_kwh' not in inputs:
            self.logger.error("Missing required input: electricity_kwh")
            return False

        if inputs['electricity_kwh'] < 0:
            self.logger.error("electricity_kwh must be non-negative")
            return False

        return True
```

### Step 2: Use Your Agent

```python
# Create an instance
calculator = SimpleCarbonCalculator()

# Run a calculation
result = calculator.run({
    "inputs": {
        "electricity_kwh": 1000,
        "emission_factor": 0.45
    }
})

# Check the result
if result.success:
    print(f"Emissions: {result.result_value:.2f} kg CO2")
    print(f"Execution time: {result.metrics.execution_time_ms:.2f}ms")

    # View calculation steps
    for step in result.calculation_steps:
        print(f"\n{step.step_name}:")
        print(f"  Formula: {step.formula}")
        print(f"  Result: {step.result} {step.units}")
else:
    print(f"Error: {result.error}")
```

### Step 3: Check Agent Statistics

```python
# Run multiple calculations
for kwh in [500, 1000, 1500, 2000]:
    calculator.run({"inputs": {"electricity_kwh": kwh}})

# View statistics
stats = calculator.get_stats()
print(f"\nAgent Statistics:")
print(f"  Total executions: {stats['executions']}")
print(f"  Success rate: {stats['success_rate']}%")
print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
```

**Output:**
```
Emissions: 450.00 kg CO2
Execution time: 2.45ms

Calculate Emissions:
  Formula: electricity_kwh × emission_factor
  Result: 450.0 kg CO2

Agent Statistics:
  Total executions: 4
  Success rate: 100.0%
  Average time: 2.31ms
```

Congratulations! You've created and run your first GreenLang agent!

---

## Agent Types Overview

GreenLang provides four specialized base classes for different use cases:

```
BaseAgent
│
├── BaseDataProcessor    → Batch processing & transformations
├── BaseCalculator       → Mathematical computations
└── BaseReporter        → Data aggregation & reporting
```

### When to Use Each Agent Type

| Agent Type | Use For | Key Features |
|------------|---------|--------------|
| **BaseAgent** | Custom agents, any logic | Full control, lifecycle management |
| **BaseDataProcessor** | ETL, data transformation, batch jobs | Parallel processing, error collection |
| **BaseCalculator** | Math, formulas, computations | High precision, caching, calculation trace |
| **BaseReporter** | Reports, dashboards, summaries | Multi-format output (MD, HTML, JSON, Excel) |

---

## Basic Examples

### Example 1: DataProcessor Agent

Transform a batch of temperature readings from Fahrenheit to Celsius:

```python
from greenlang.agents import BaseDataProcessor, DataProcessorConfig
from typing import Dict, Any

class TemperatureConverter(BaseDataProcessor):
    """Convert temperature readings from Fahrenheit to Celsius."""

    def __init__(self):
        config = DataProcessorConfig(
            name="TemperatureConverter",
            description="Convert temperature data F→C",
            batch_size=100,
            parallel_workers=4,
            enable_progress=True
        )
        super().__init__(config)

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single temperature reading."""
        fahrenheit = record['temperature_f']
        celsius = (fahrenheit - 32) * 5 / 9

        return {
            'sensor_id': record['sensor_id'],
            'timestamp': record['timestamp'],
            'temperature_f': fahrenheit,
            'temperature_c': round(celsius, 2)
        }

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate record has required fields."""
        return all(key in record for key in ['sensor_id', 'timestamp', 'temperature_f'])

# Use the processor
converter = TemperatureConverter()

# Sample data
temperature_data = [
    {'sensor_id': 'S1', 'timestamp': '2025-01-01T00:00:00', 'temperature_f': 32},
    {'sensor_id': 'S2', 'timestamp': '2025-01-01T00:00:00', 'temperature_f': 68},
    {'sensor_id': 'S3', 'timestamp': '2025-01-01T00:00:00', 'temperature_f': 98.6},
]

result = converter.run({"records": temperature_data})

if result.success:
    print(f"Processed {result.records_processed} records")
    print(f"Failed {result.records_failed} records")
    print("\nConverted data:")
    for record in result.data['records']:
        print(f"  {record['sensor_id']}: {record['temperature_f']}°F = {record['temperature_c']}°C")
```

**Output:**
```
Processing batches: 100%|████████████| 1/1 [00:00<00:00, 245.12it/s]
Processed 3 records
Failed 0 records

Converted data:
  S1: 32°F = 0.0°C
  S2: 68°F = 20.0°C
  S3: 98.6°F = 37.0°C
```

### Example 2: Calculator Agent

Calculate the payback period for a solar panel installation:

```python
from greenlang.agents import BaseCalculator, CalculatorConfig
from typing import Dict, Any

class SolarPaybackCalculator(BaseCalculator):
    """Calculate payback period for solar panel installation."""

    def __init__(self):
        config = CalculatorConfig(
            name="SolarPaybackCalculator",
            description="Calculate solar panel ROI",
            precision=2,
            enable_caching=True
        )
        super().__init__(config)

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate installation costs, savings, and payback period."""

        # Extract inputs
        panel_cost = inputs['panel_cost_usd']
        installation_cost = inputs['installation_cost_usd']
        annual_generation_kwh = inputs['annual_generation_kwh']
        electricity_rate = inputs['electricity_rate_per_kwh']

        # Step 1: Calculate total investment
        total_cost = panel_cost + installation_cost
        self.add_calculation_step(
            step_name="Total Investment",
            formula="panel_cost + installation_cost",
            inputs={"panel_cost": panel_cost, "installation_cost": installation_cost},
            result=total_cost,
            units="USD"
        )

        # Step 2: Calculate annual savings
        annual_savings = annual_generation_kwh * electricity_rate
        self.add_calculation_step(
            step_name="Annual Savings",
            formula="annual_generation_kwh × electricity_rate",
            inputs={"generation": annual_generation_kwh, "rate": electricity_rate},
            result=annual_savings,
            units="USD/year"
        )

        # Step 3: Calculate payback period
        payback_years = self.safe_divide(total_cost, annual_savings)
        if payback_years is not None:
            self.add_calculation_step(
                step_name="Payback Period",
                formula="total_cost ÷ annual_savings",
                inputs={"total_cost": total_cost, "annual_savings": annual_savings},
                result=payback_years,
                units="years"
            )

        return {
            'total_cost': total_cost,
            'annual_savings': annual_savings,
            'payback_years': payback_years
        }

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate all required inputs are present."""
        required_keys = [
            'panel_cost_usd',
            'installation_cost_usd',
            'annual_generation_kwh',
            'electricity_rate_per_kwh'
        ]
        return all(key in inputs for key in required_keys)

# Use the calculator
calculator = SolarPaybackCalculator()

result = calculator.run({
    "inputs": {
        "panel_cost_usd": 15000,
        "installation_cost_usd": 5000,
        "annual_generation_kwh": 12000,
        "electricity_rate_per_kwh": 0.15
    }
})

if result.success:
    values = result.result_value
    print(f"Total Investment: ${values['total_cost']:,.2f}")
    print(f"Annual Savings: ${values['annual_savings']:,.2f}/year")
    print(f"Payback Period: {values['payback_years']:.1f} years")
    print(f"\nCalculation Steps:")
    for step in result.calculation_steps:
        print(f"  • {step.step_name}: {step.result} {step.units}")
```

**Output:**
```
Total Investment: $20,000.00
Annual Savings: $1,800.00/year
Payback Period: 11.1 years

Calculation Steps:
  • Total Investment: 20000 USD
  • Annual Savings: 1800.0 USD/year
  • Payback Period: 11.11 years
```

### Example 3: Reporter Agent

Generate a monthly energy consumption report:

```python
from greenlang.agents import BaseReporter, ReporterConfig, ReportSection
from typing import Dict, Any, List

class EnergyConsumptionReporter(BaseReporter):
    """Generate monthly energy consumption reports."""

    def __init__(self, output_format='markdown'):
        config = ReporterConfig(
            name="Energy Consumption Report",
            description="Monthly building energy analysis",
            output_format=output_format,
            include_summary=True,
            include_details=True
        )
        super().__init__(config)

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate energy consumption data."""
        readings = input_data['readings']

        total_kwh = sum(r['kwh'] for r in readings)
        avg_kwh = total_kwh / len(readings)
        max_reading = max(readings, key=lambda r: r['kwh'])
        min_reading = min(readings, key=lambda r: r['kwh'])

        return {
            'total_consumption_kwh': round(total_kwh, 2),
            'average_daily_kwh': round(avg_kwh, 2),
            'peak_day': max_reading['date'],
            'peak_consumption_kwh': max_reading['kwh'],
            'lowest_day': min_reading['date'],
            'lowest_consumption_kwh': min_reading['kwh'],
            'num_days': len(readings)
        }

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """Build report sections."""
        sections = []

        # Consumption details section
        details_table = [
            {'Metric': 'Total Consumption', 'Value': f"{aggregated_data['total_consumption_kwh']:,.2f} kWh"},
            {'Metric': 'Average Daily', 'Value': f"{aggregated_data['average_daily_kwh']:,.2f} kWh"},
            {'Metric': 'Peak Day', 'Value': f"{aggregated_data['peak_day']} ({aggregated_data['peak_consumption_kwh']:.2f} kWh)"},
            {'Metric': 'Lowest Day', 'Value': f"{aggregated_data['lowest_day']} ({aggregated_data['lowest_consumption_kwh']:.2f} kWh)"},
        ]

        sections.append(ReportSection(
            title="Consumption Breakdown",
            content=details_table,
            level=2,
            section_type="table"
        ))

        # Recommendations
        avg_kwh = aggregated_data['average_daily_kwh']
        if avg_kwh > 100:
            recommendations = [
                "Consider upgrading to LED lighting to reduce consumption by 20-30%",
                "Schedule HVAC optimization audit",
                "Install occupancy sensors in common areas"
            ]
        else:
            recommendations = ["Energy usage is within optimal range"]

        sections.append(ReportSection(
            title="Recommendations",
            content=recommendations,
            level=2,
            section_type="list"
        ))

        return sections

# Use the reporter
reporter = EnergyConsumptionReporter(output_format='markdown')

# Sample data
monthly_data = {
    'month': 'January 2025',
    'readings': [
        {'date': '2025-01-01', 'kwh': 95.3},
        {'date': '2025-01-02', 'kwh': 102.7},
        {'date': '2025-01-03', 'kwh': 88.4},
        {'date': '2025-01-04', 'kwh': 125.8},
        {'date': '2025-01-05', 'kwh': 91.2},
    ]
}

result = reporter.run(monthly_data)

if result.success:
    print(result.data['report'])
    print(f"\nGenerated {result.data['sections_count']} sections")
```

**Output:**
```markdown
# Energy Consumption Report Report

**Generated:** 2025-10-16 14:23:45

## Summary

- **Total Consumption Kwh**: 503.40
- **Average Daily Kwh**: 100.68
- **Peak Day**: 2025-01-04
- **Peak Consumption Kwh**: 125.80
- **Lowest Day**: 2025-01-03
- **Lowest Consumption Kwh**: 88.40
- **Num Days**: 5

## Consumption Breakdown

| Metric | Value |
| --- | --- |
| Total Consumption | 503.40 kWh |
| Average Daily | 100.68 kWh |
| Peak Day | 2025-01-04 (125.80 kWh) |
| Lowest Day | 2025-01-03 (88.40 kWh) |

## Recommendations

- Consider upgrading to LED lighting to reduce consumption by 20-30%
- Schedule HVAC optimization audit
- Install occupancy sensors in common areas

Generated 3 sections
```

---

## Common Patterns

### Pattern 1: Agent Composition

Combine multiple agents into a pipeline:

```python
from greenlang.agents import BaseAgent, AgentConfig, AgentResult
from typing import Dict, Any

class DataPipeline(BaseAgent):
    """Compose multiple agents into a pipeline."""

    def __init__(self, agents):
        config = AgentConfig(
            name="DataPipeline",
            description="Orchestrate multiple agents"
        )
        super().__init__(config)
        self.agents = agents

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Run agents in sequence."""
        data = input_data
        results = []

        for agent in self.agents:
            result = agent.run(data)

            if not result.success:
                return AgentResult(
                    success=False,
                    error=f"Agent {agent.config.name} failed: {result.error}"
                )

            # Pass output to next agent
            data = result.data
            results.append(result)

        return AgentResult(
            success=True,
            data=data,
            metadata={'pipeline_results': results}
        )

# Use the pipeline
converter = TemperatureConverter()
reporter = EnergyConsumptionReporter()

pipeline = DataPipeline(agents=[converter, reporter])
result = pipeline.run({"records": temperature_data})
```

### Pattern 2: Custom Validation

Add sophisticated input validation:

```python
class ValidatedCalculator(BaseCalculator):
    """Calculator with comprehensive validation."""

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Multi-level validation."""

        # Check required fields
        required = ['value', 'unit']
        if not all(key in inputs for key in required):
            self.logger.error(f"Missing required fields: {required}")
            return False

        # Check data types
        if not isinstance(inputs['value'], (int, float)):
            self.logger.error("'value' must be numeric")
            return False

        # Check value ranges
        if inputs['value'] < 0:
            self.logger.error("'value' must be non-negative")
            return False

        # Check valid units
        valid_units = ['kWh', 'MWh', 'GJ']
        if inputs['unit'] not in valid_units:
            self.logger.error(f"'unit' must be one of: {valid_units}")
            return False

        return True
```

### Pattern 3: Error Recovery

Handle errors gracefully in data processors:

```python
class RobustDataProcessor(BaseDataProcessor):
    """Data processor with error recovery strategies."""

    def __init__(self):
        config = DataProcessorConfig(
            name="RobustProcessor",
            collect_errors=True,  # Don't fail immediately
            max_errors=10,        # Stop after 10 errors
            validate_records=True
        )
        super().__init__(config)

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process with fallback values."""
        try:
            value = float(record.get('value', 0))
        except ValueError:
            # Fallback to default if conversion fails
            self.logger.warning(f"Could not convert value, using 0")
            value = 0

        return {
            'id': record.get('id', 'unknown'),
            'value': value,
            'processed': True
        }

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Lenient validation."""
        return isinstance(record, dict)
```

### Pattern 4: Custom Metrics

Track domain-specific metrics:

```python
class MetricsTrackingAgent(BaseAgent):
    """Agent that tracks custom metrics."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute with custom metric tracking."""

        # Your business logic here
        processing_steps = input_data.get('steps', [])

        for step in processing_steps:
            # Track custom metrics
            self.stats.increment('steps_processed')

            if step['type'] == 'critical':
                self.stats.increment('critical_steps')

        # Track timing for specific operations
        import time
        start = time.time()
        # ... do expensive operation ...
        duration_ms = (time.time() - start) * 1000
        self.stats.add_time('expensive_operation', duration_ms)

        return AgentResult(
            success=True,
            data={'processed_steps': len(processing_steps)}
        )

    def get_custom_metrics(self):
        """Retrieve custom metrics."""
        stats = self.get_stats()
        return {
            'steps_processed': stats['custom_counters'].get('steps_processed', 0),
            'critical_steps': stats['custom_counters'].get('critical_steps', 0),
            'expensive_op_time': stats['custom_timers'].get('expensive_operation', 0)
        }
```

### Pattern 5: Lifecycle Hooks

Add pre and post-execution hooks:

```python
def log_execution(agent, input_data):
    """Hook to log before execution."""
    print(f"[{agent.config.name}] Starting execution with {len(input_data)} keys")

def save_result(agent, result):
    """Hook to save result after execution."""
    if result.success:
        print(f"[{agent.config.name}] Completed successfully in {result.metrics.execution_time_ms:.2f}ms")
    else:
        print(f"[{agent.config.name}] Failed: {result.error}")

# Add hooks to any agent
calculator = SimpleCarbonCalculator()
calculator.add_pre_hook(log_execution)
calculator.add_post_hook(save_result)

# Hooks will be called automatically
result = calculator.run({"inputs": {"electricity_kwh": 100}})
```

---

## Troubleshooting

### Issue: Agent execution fails silently

**Symptom:** Agent returns success=False but no clear error message.

**Solution:** Enable debug logging:

```python
import logging

# Configure logging to see detailed error messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or configure for specific agent
calculator = SimpleCarbonCalculator()
calculator.logger.setLevel(logging.DEBUG)
```

### Issue: DataProcessor is too slow

**Symptom:** Processing large datasets takes too long.

**Solutions:**

1. **Enable parallel processing:**
```python
config = DataProcessorConfig(
    parallel_workers=8,  # Use 8 threads
    batch_size=1000      # Larger batches
)
```

2. **Disable validation for trusted data:**
```python
config = DataProcessorConfig(
    validate_records=False  # Skip validation
)
```

3. **Disable progress bar in production:**
```python
config = DataProcessorConfig(
    enable_progress=False  # No visual progress
)
```

### Issue: Calculator results are inconsistent

**Symptom:** Same inputs produce slightly different outputs.

**Solution:** Ensure deterministic mode and proper precision:

```python
config = CalculatorConfig(
    deterministic=True,  # Ensure consistent results
    precision=6          # Set appropriate precision
)
```

### Issue: Reporter fails with Excel export

**Symptom:** `ImportError: openpyxl required for Excel export`

**Solution:** Install the optional dependency:

```bash
pip install openpyxl
```

### Issue: Memory usage is too high

**Symptom:** Agent consumes excessive memory.

**Solutions:**

1. **Reduce cache size:**
```python
config = CalculatorConfig(
    enable_caching=True,
    cache_size=32  # Smaller cache
)
```

2. **Process in smaller batches:**
```python
config = DataProcessorConfig(
    batch_size=100  # Smaller batches
)
```

3. **Clear caches periodically:**
```python
calculator.clear_cache()
calculator.reset_stats()
```

### Issue: Validation is too strict

**Symptom:** Agent rejects valid inputs.

**Solution:** Customize validation logic:

```python
class LenientAgent(BaseCalculator):
    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Less strict validation."""
        # Only check critical fields
        return 'critical_field' in inputs
```

### Issue: Need to debug calculation steps

**Symptom:** Results don't match expectations.

**Solution:** Inspect calculation trace:

```python
result = calculator.run({"inputs": {...}})

if result.success:
    print("Calculation Steps:")
    for i, step in enumerate(result.calculation_steps, 1):
        print(f"\nStep {i}: {step.step_name}")
        print(f"  Formula: {step.formula}")
        print(f"  Inputs: {step.inputs}")
        print(f"  Result: {step.result} {step.units}")
```

### Issue: Agent statistics are reset unexpectedly

**Symptom:** Statistics show zero executions after multiple runs.

**Solution:** Reuse agent instances instead of creating new ones:

```python
# ❌ Bad: Creates new instance each time
for data in dataset:
    agent = MyAgent()  # Statistics reset
    agent.run(data)

# ✅ Good: Reuse instance
agent = MyAgent()
for data in dataset:
    agent.run(data)  # Statistics accumulate

stats = agent.get_stats()  # Now shows all executions
```

### Getting Help

If you're still stuck:

1. **Check the logs:** Set log level to DEBUG for detailed information
2. **Read the API Reference:** See [API_REFERENCE.md](API_REFERENCE.md) for detailed documentation
3. **Review examples:** Check [docs/examples/](examples/) for more code samples
4. **Ask for help:** Open an issue on GitHub or join our Discord community

---

## Next Steps

Now that you've got the basics, explore more advanced topics:

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Migration Guide](MIGRATION_GUIDE.md)** - Migrate existing code to the framework
- **[Examples](examples/)** - 10+ complete, runnable examples
- **[Architecture Guide](ARCHITECTURE.md)** - Deep dive into framework design

Happy building with GreenLang! 🌍
