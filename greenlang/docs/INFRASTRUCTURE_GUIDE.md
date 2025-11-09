# GreenLang Infrastructure Guide

Complete guide to the GreenLang SDK infrastructure components.

## Table of Contents

1. [Validation Framework](#validation-framework)
2. [Cache System](#cache-system)
3. [Telemetry System](#telemetry-system)
4. [Provenance Framework](#provenance-framework)
5. [Agent Templates](#agent-templates)

---

## Validation Framework

The validation framework provides multi-layer validation with JSON Schema, business rules, and data quality checks.

### Quick Start

```python
from greenlang.validation import ValidationFramework, SchemaValidator, RulesEngine, Rule, RuleOperator

# Create framework
framework = ValidationFramework()

# Add JSON Schema validator
schema = {
    "type": "object",
    "properties": {
        "emissions": {"type": "number", "minimum": 0},
        "source": {"type": "string"}
    },
    "required": ["emissions", "source"]
}
schema_validator = SchemaValidator(schema)
framework.add_validator("schema", schema_validator.validate)

# Add business rules
rules_engine = RulesEngine()
rules_engine.add_rule(Rule(
    name="emissions_threshold",
    field="emissions",
    operator=RuleOperator.LESS_THAN,
    value=1000,
    message="Emissions exceed threshold"
))
framework.add_validator("business_rules", rules_engine.validate)

# Validate data
data = {"emissions": 250, "source": "Scope 1"}
result = framework.validate(data)

if result.valid:
    print("Validation passed!")
else:
    print(f"Errors: {result.get_summary()}")
    for error in result.errors:
        print(f"  - {error}")
```

### Features

- **JSON Schema Validation**: Standard JSON Schema Draft 7 support
- **Business Rules Engine**: Flexible rule-based validation
- **Data Quality Checks**: Automatic quality assessment
- **Custom Validators**: Register custom validation functions
- **Batch Validation**: Validate multiple items efficiently

### API Reference

#### ValidationFramework

- `add_validator(name, func, config)`: Register validator
- `validate(data, validators=None)`: Run validation
- `validate_batch(data_list)`: Validate multiple items
- `enable_validator(name)`: Enable validator
- `disable_validator(name)`: Disable validator

#### SchemaValidator

- `validate(data)`: Validate against JSON schema
- `from_file(path)`: Load schema from file

#### RulesEngine

- `add_rule(rule)`: Add validation rule
- `validate(data)`: Validate against rules
- `load_rules_from_dict(config)`: Load rules from config

---

## Cache System

Multi-layer caching (L1 Memory, L2 Redis, L3 Disk) with intelligent orchestration.

### Quick Start

```python
from greenlang.cache import CacheManager, get_cache_manager, initialize_cache_manager

# Initialize cache manager
initialize_cache_manager(
    enable_l1=True,
    enable_l2=True,
    enable_l3=True,
    redis_url="redis://localhost:6379"
)

cache_manager = get_cache_manager()

# Cache data
await cache_manager.set("my_key", {"data": "value"}, ttl=3600)

# Retrieve data
data = await cache_manager.get("my_key")

# Get or compute pattern
async def expensive_computation():
    # ... expensive operation ...
    return {"result": "computed_value"}

result = await cache_manager.get_or_compute(
    key="computation_key",
    compute_fn=expensive_computation,
    ttl=3600
)
```

### Architecture

```
┌─────────────┐
│ L1 Memory   │ ← Fastest (LRU, in-memory)
├─────────────┤
│ L2 Redis    │ ← Distributed (shared across instances)
├─────────────┤
│ L3 Disk     │ ← Persistent (survives restarts)
└─────────────┘
```

### Features

- **L1 Memory Cache**: Ultra-fast LRU cache with configurable size
- **L2 Redis Cache**: Distributed caching with circuit breaker
- **L3 Disk Cache**: Persistent file-based cache
- **Automatic Promotion**: Hot data moves to faster layers
- **TTL Management**: Time-based expiration
- **Invalidation**: Pattern-based, event-based, version-based

### API Reference

#### CacheManager

- `get(key)`: Retrieve from cache
- `set(key, value, ttl)`: Store in cache
- `delete(key)`: Remove from cache
- `get_or_compute(key, compute_fn, ttl)`: Get or compute pattern
- `invalidate_pattern(pattern)`: Invalidate by pattern
- `get_analytics()`: Get cache statistics

---

## Telemetry System

Comprehensive observability with metrics, logging, and tracing.

### Quick Start

```python
from greenlang.telemetry import (
    MetricsCollector,
    StructuredLogger,
    TracingManager,
    get_metrics_collector,
    get_logger
)

# Metrics
metrics = get_metrics_collector()
metrics.increment("calculations.total")
metrics.record("calculation.duration", 1.23)

# Structured logging
logger = get_logger(__name__)
logger.info("Processing data", extra={
    "rows": 1000,
    "source": "emissions.csv"
})

# Tracing
tracer = TracingManager()
with tracer.trace_operation("data_processing"):
    # ... processing code ...
    tracer.add_attribute("records_processed", 1000)
```

### Features

- **Prometheus Metrics**: Counter, Gauge, Histogram, Summary
- **Structured Logging**: JSON logging with context
- **OpenTelemetry Tracing**: Distributed tracing support
- **Health Checks**: Readiness and liveness probes
- **Performance Monitoring**: Automatic latency tracking

### API Reference

#### MetricsCollector

- `increment(name, value=1, labels)`: Increment counter
- `gauge(name, value, labels)`: Set gauge value
- `record(name, value, labels)`: Record histogram value
- `get_metrics()`: Get all metrics

#### StructuredLogger

- `debug/info/warning/error(message, **kwargs)`: Log with context
- `add_context(key, value)`: Add persistent context
- `with_context(context_dict)`: Context manager

---

## Provenance Framework

Enterprise-grade provenance tracking for regulatory compliance.

### Quick Start

```python
from greenlang.provenance import ProvenanceTracker

# Create tracker
tracker = ProvenanceTracker(
    name="emissions_calculation",
    auto_hash_files=True
)

# Track operation
with tracker.track_operation("data_intake"):
    tracker.track_file_input("emissions_data.csv")

    # ... process data ...

    tracker.track_file_output("cleaned_data.parquet")

# Track transformation
tracker.track_data_transformation(
    source="emissions_data.csv",
    destination="cleaned_data.parquet",
    transformation="data cleaning and validation",
    input_records=1000,
    output_records=950
)

# Save provenance record
tracker.save_record("provenance.json")

# Verify integrity
is_valid = tracker.verify_integrity("cleaned_data.parquet")
```

### Features

- **Automatic Lineage Tracking**: Captures data flow automatically
- **SHA-256 Hashing**: File integrity verification
- **Chain of Custody**: Audit trail for regulatory compliance
- **Environment Capture**: System and dependency information
- **Decorator Support**: `@track_with_provenance` decorator

### API Reference

#### ProvenanceTracker

- `track_operation(name)`: Context manager for operations
- `track_file_input(path)`: Track input file
- `track_file_output(path)`: Track output file
- `track_data_transformation(...)`: Track transformation
- `track_agent_execution(...)`: Track agent run
- `get_record()`: Get provenance record
- `save_record(path)`: Save to file
- `verify_integrity(path)`: Verify file integrity

---

## Agent Templates

Production-ready agent templates for common sustainability workflows.

### IntakeAgent

Multi-format data ingestion with validation.

```python
from greenlang.agents.templates import IntakeAgent, DataFormat

# Create agent with schema
schema = {
    "required": ["emissions", "source"],
    "types": {
        "emissions": "number",
        "source": "string"
    }
}
agent = IntakeAgent(schema=schema)

# Ingest CSV file
result = await agent.ingest(
    file_path="emissions.csv",
    format=DataFormat.CSV,
    validate=True
)

if result.success:
    print(f"Ingested {result.rows_read} rows")
    data = result.data  # pandas DataFrame
else:
    for issue in result.validation_issues:
        print(f"{issue.severity}: {issue.message}")

# Streaming for large files
results = await agent.ingest_streaming(
    file_path="large_emissions.csv",
    format=DataFormat.CSV,
    chunk_size=10000
)
```

**Supported Formats**: CSV, JSON, Excel, XML, Parquet, Avro, ORC, Feather

### CalculatorAgent

Zero-hallucination calculations with provenance.

```python
from greenlang.agents.templates import CalculatorAgent

# Create agent
agent = CalculatorAgent()

# Register formula
def calculate_emissions(activity_data, emission_factor):
    return activity_data * emission_factor

agent.register_formula(
    "scope1_emissions",
    calculate_emissions,
    required_inputs=["activity_data", "emission_factor"]
)

# Calculate
result = await agent.calculate(
    formula_name="scope1_emissions",
    inputs={
        "activity_data": 1000,
        "emission_factor": 2.5,
        "unit": "kg CO2e"
    }
)

print(f"Result: {result.value} {result.unit}")
print(f"Provenance: {result.provenance.hash}")

# Parallel batch processing
inputs_list = [
    {"activity_data": 1000, "emission_factor": 2.5},
    {"activity_data": 2000, "emission_factor": 3.0},
    {"activity_data": 1500, "emission_factor": 2.8},
]

results = await agent.batch_calculate(
    formula_name="scope1_emissions",
    inputs_list=inputs_list,
    parallel=True,
    use_processes=True  # CPU-intensive calculations
)
```

### ReportingAgent

Multi-format export with compliance checking.

```python
from greenlang.agents.templates import ReportingAgent, ReportFormat, ComplianceFramework

# Create agent
agent = ReportingAgent()

# Generate report
result = await agent.generate_report(
    data=emissions_df,
    format=ReportFormat.EXCEL,
    output_path="emissions_report.xlsx",
    check_compliance=[ComplianceFramework.GHG_PROTOCOL, ComplianceFramework.CSRD]
)

if result.success:
    print(f"Report saved to {result.file_path}")

    # Check compliance
    for check in result.compliance_checks:
        if check.passed:
            print(f"{check.framework}: PASSED")
        else:
            print(f"{check.framework}: FAILED")
            for issue in check.issues:
                print(f"  - {issue}")

# Generate with charts
chart_configs = [
    {
        "type": "bar",
        "x": "category",
        "y": "emissions",
        "title": "Emissions by Category"
    },
    {
        "type": "line",
        "x": "date",
        "y": "emissions",
        "title": "Emissions Over Time"
    }
]

result = await agent.generate_with_charts(
    data=emissions_df,
    chart_configs=chart_configs,
    format=ReportFormat.HTML,
    output_path="emissions_report.html"
)
```

**Supported Formats**: JSON, Excel, CSV, HTML, PDF, XBRL, Parquet, XML, Markdown, YAML

---

## Best Practices

### Validation

1. **Layer your validation**: Use schema + rules + quality checks
2. **Set appropriate severity levels**: ERROR for critical, WARNING for important
3. **Provide clear error messages**: Help users fix issues
4. **Validate early**: Catch errors at ingestion

### Caching

1. **Set appropriate TTLs**: Balance freshness and performance
2. **Use get_or_compute**: Prevents thundering herd
3. **Monitor cache hit rates**: Optimize cache strategy
4. **Invalidate strategically**: Use patterns and events

### Telemetry

1. **Add context to logs**: Include relevant metadata
2. **Track key metrics**: Business and technical KPIs
3. **Use distributed tracing**: For multi-service workflows
4. **Set up alerts**: Proactive issue detection

### Provenance

1. **Track at key points**: Ingestion, transformation, export
2. **Hash critical files**: Verify data integrity
3. **Store records with outputs**: Enable reproducibility
4. **Audit regularly**: Verify provenance chain

### Agents

1. **Validate inputs**: Use schemas with IntakeAgent
2. **Use parallel processing**: For batch calculations
3. **Track provenance**: Enable full auditability
4. **Check compliance**: Validate against frameworks

---

## Performance Tips

### Validation
- Disable unused validators
- Use batch validation for multiple items
- Cache schema validators

### Caching
- Enable all cache layers for best performance
- Use Redis for distributed deployments
- Configure L1 cache size based on memory

### Telemetry
- Use sampling for high-volume traces
- Batch metric writes
- Compress log data

### Provenance
- Disable file hashing for performance-critical paths
- Use async tracking
- Store records in distributed storage

---

## Troubleshooting

### Common Issues

**Validation fails unexpectedly**
- Check jsonschema is installed
- Verify schema format
- Review validation rules

**Cache misses are high**
- Check TTL configuration
- Monitor eviction rates
- Review cache size

**Metrics not appearing**
- Verify Prometheus is running
- Check metrics endpoint
- Review metric names

**Provenance tracking slow**
- Disable file hashing if not needed
- Use async operations
- Batch provenance writes

---

## Migration Guide

### From Legacy Systems

If migrating from legacy validation:

```python
# Old
def validate_data(data):
    if data["value"] < 0:
        raise ValueError("Invalid")

# New
from greenlang.validation import ValidationFramework, RulesEngine, Rule

framework = ValidationFramework()
rules = RulesEngine()
rules.add_rule(Rule(
    name="positive_value",
    field="value",
    operator=RuleOperator.GREATER_EQUAL,
    value=0
))
framework.add_validator("rules", rules.validate)
result = framework.validate(data)
```

---

## Support

For issues, questions, or contributions:

- GitHub Issues: [greenlang/issues](https://github.com/greenlang/greenlang/issues)
- Documentation: [docs.greenlang.io](https://docs.greenlang.io)
- Community: [discuss.greenlang.io](https://discuss.greenlang.io)
