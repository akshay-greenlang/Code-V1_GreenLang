# GreenLang Examples Gallery

A comprehensive collection of production-ready examples demonstrating GreenLang's capabilities for building climate-aware applications.

## Overview

This gallery contains 10+ fully-functional examples covering core concepts, best practices, and advanced patterns for climate intelligence applications.

## Quick Start

```bash
# Run any example directly
python examples/01_simple_agent.py
python examples/02_data_processor.py
# ... etc

# Or use the provided sample data
ls examples/data/
```

## Examples Index

### 1. Simple Agent - Basic Data Processing
**File:** `01_simple_agent.py`

Learn the fundamentals of creating agents with GreenLang SDK.

**Demonstrates:**
- Creating a basic agent from scratch
- Input validation
- Emissions calculations using emission factors
- Error handling
- Multi-country support

**Key Concepts:**
- `Agent` base class
- `validate()` and `process()` methods
- `Result` objects
- Working with emission factor datasets

**Run:**
```bash
python examples/01_simple_agent.py
```

**Expected Output:**
- US office building: ~26.95 tCO2e
- UK office building: ~16.95 tCO2e (lower grid intensity)
- Error handling demonstration

---

### 2. CSV Batch Processor with Error Handling
**File:** `02_data_processor.py`

Process multiple buildings from CSV files with comprehensive error handling.

**Demonstrates:**
- Batch processing CSV data
- Row-by-row error handling
- Processing statistics tracking
- Success/failure reporting
- Output file generation

**Key Concepts:**
- CSV data ingestion
- Error accumulation patterns
- Statistical summaries
- Result aggregation

**Run:**
```bash
python examples/02_data_processor.py
```

**Expected Output:**
- Processes 5 buildings from sample_buildings.csv
- Success rate: 100%
- Total emissions: ~129.65 tCO2e
- Generates JSON output file

---

### 3. Calculator with Caching and Determinism
**File:** `03_calculator_cached.py`

High-performance calculator with caching and reproducible results.

**Demonstrates:**
- Function-level caching (LRU cache)
- Deterministic execution with seeded randomness
- Performance benchmarking
- Cache hit/miss tracking
- Reproducibility verification

**Key Concepts:**
- `@lru_cache` decorator
- Random seed management
- Hash-based caching
- Performance optimization

**Run:**
```bash
python examples/03_calculator_cached.py
```

**Expected Output:**
- Identical results across runs (same seed)
- Different results with different seeds
- 1000+ calculations/second with caching
- Cache hit rate: >95%

---

### 4. Multi-Format Reporter (Markdown, HTML, Excel)
**File:** `04_multi_format_reporter.py`

Generate comprehensive reports in multiple output formats.

**Demonstrates:**
- Markdown report generation
- HTML report with styling
- JSON export
- CSV data export
- Multi-format output pipeline

**Key Concepts:**
- `Report` base class
- Template-based reporting
- Format-specific generation
- File I/O management

**Run:**
```bash
python examples/04_multi_format_reporter.py
```

**Expected Output:**
- 4 report files generated
- Markdown: emissions_report.md
- HTML: emissions_report.html
- JSON: emissions_report.json
- CSV: emissions_breakdown.csv

---

### 5. Complete Provenance and Audit Trail
**File:** `05_provenance_tracking.py`

Full audit trail with input/output hashing and run ledger.

**Demonstrates:**
- Provenance tracking
- Input/output hashing (SHA-256)
- Run ledger management
- Reproducibility verification
- Audit trail queries
- Ledger export

**Key Concepts:**
- `RunLedger` class
- `stable_hash()` function
- Execution metadata
- Compliance tracking

**Run:**
```bash
python examples/05_provenance_tracking.py
```

**Expected Output:**
- Run ID for each calculation
- Input/output hashes
- Reproducibility verification
- Ledger statistics
- Exported ledger JSON

---

### 6. Schema and Business Rules Validation
**File:** `06_validation_framework.py`

Comprehensive validation framework with schema and business rules.

**Demonstrates:**
- JSON Schema validation
- Business rules validation
- Data quality checks
- Multi-stage validation
- Warning vs error handling
- Validation error reporting

**Key Concepts:**
- `Validator` base class
- Schema definition
- Business logic validation
- Data quality metrics

**Run:**
```bash
python examples/06_validation_framework.py
```

**Expected Output:**
- Valid data passes all checks
- Invalid data caught with clear errors
- Warnings for data quality issues
- Detailed error messages

---

### 7. Unit Conversion Calculator
**File:** `07_unit_converter.py`

Universal calculator accepting any common energy/area unit.

**Demonstrates:**
- Energy unit conversions (kWh, MWh, GJ, BTU)
- Area unit conversions (sqm, sqft, acre)
- Automatic unit detection
- Conversion tracking
- Multi-unit input support

**Key Concepts:**
- `Transform` class
- Unit conversion factors
- Normalization pipelines
- Conversion verification

**Run:**
```bash
python examples/07_unit_converter.py
```

**Expected Output:**
- All unit conversions verified
- 50 MWh = 50,000 kWh confirmed
- Mixed unit processing successful
- Automatic normalization applied

---

### 8. Parallel Batch Processing
**File:** `08_parallel_processing.py`

High-performance parallel processing with ThreadPoolExecutor.

**Demonstrates:**
- Parallel execution with thread pools
- Serial vs parallel comparison
- Worker count optimization
- Progress tracking
- Performance benchmarking

**Key Concepts:**
- `ThreadPoolExecutor`
- Concurrent processing
- Speedup calculations
- Throughput metrics

**Run:**
```bash
python examples/08_parallel_processing.py
```

**Expected Output:**
- Serial: ~2.0 seconds for 20 buildings
- Parallel (4 workers): ~0.5 seconds
- Speedup: 4x
- Optimal worker count identified

---

### 9. Custom Decorators for Agent Methods
**File:** `09_custom_decorators.py`

Demonstrates custom decorators for enhanced functionality.

**Demonstrates:**
- `@deterministic`: Reproducible results
- `@cached`: Result caching
- `@traced`: Execution tracing
- `@timed`: Performance measurement
- `@validated`: Input validation
- Decorator composition

**Key Concepts:**
- Python decorators
- Function wrapping
- Metadata tracking
- Performance optimization

**Run:**
```bash
python examples/09_custom_decorators.py
```

**Expected Output:**
- All decorators demonstrated
- Cache hit/miss tracking
- Execution trace logs
- Timing measurements
- Validation enforcement

---

### 10. Complete Multi-Agent Pipeline
**File:** `10_complete_pipeline.py`

End-to-end pipeline with intake, calculation, and reporting.

**Demonstrates:**
- Multi-agent orchestration
- Pipeline composition
- Stage-by-stage execution
- Error propagation
- Comprehensive output generation

**Pipeline Stages:**
1. **IntakeAgent**: Load CSV data
2. **CalculatorAgent**: Calculate emissions
3. **ReportGeneratorAgent**: Create reports

**Run:**
```bash
python examples/10_complete_pipeline.py
```

**Expected Output:**
- 3 stages executed successfully
- 5 buildings processed
- Total emissions calculated
- Reports generated (MD, JSON, CSV)

---

## Sample Data

### `examples/data/` Directory

**sample_buildings.csv**
- 5 sample buildings with energy data
- Includes area, electricity, gas consumption

**sample_energy.csv**
- Time-series energy data
- Multiple buildings and dates

**emission_factors.json**
- Emission factors for US, UK, CA
- Electricity and natural gas factors
- Source attribution

**calculation_inputs.json**
- Sample structured input data
- Building metadata
- Energy consumption details

---

## Running the Examples

### Prerequisites

```bash
# Install GreenLang
pip install greenlang-cli

# Or install from source
pip install -e .
```

### Run Individual Examples

```bash
# Basic examples
python examples/01_simple_agent.py
python examples/02_data_processor.py
python examples/03_calculator_cached.py

# Advanced examples
python examples/05_provenance_tracking.py
python examples/08_parallel_processing.py
python examples/10_complete_pipeline.py
```

### Run All Examples

```bash
# Run all examples sequentially
for i in {01..10}; do
    echo "Running example $i..."
    python examples/${i}_*.py
done
```

---

## Output Directories

Examples generate output in the following directories:

```
examples/out/
├── batch_processing_results.json      # Example 02 output
├── reports/                           # Example 04 reports
│   ├── emissions_report.md
│   ├── emissions_report.html
│   ├── emissions_report.json
│   └── emissions_breakdown.csv
├── ledger/                            # Example 05 ledger
│   ├── calculations.jsonl
│   └── ledger_export.json
└── pipeline_reports/                  # Example 10 reports
    ├── emissions_report.md
    ├── emissions_report.json
    └── emissions_summary.csv
```

---

## Best Practices Demonstrated

### 1. Agent Design
- Clear separation of validation and processing
- Type hints for inputs/outputs
- Comprehensive error handling
- Logging at appropriate levels

### 2. Data Handling
- Schema validation before processing
- Business rules enforcement
- Unit normalization
- Error accumulation patterns

### 3. Performance
- Caching for expensive operations
- Parallel processing for batch jobs
- Benchmarking and optimization
- Resource management

### 4. Reproducibility
- Deterministic execution
- Seed management
- Input/output hashing
- Provenance tracking

### 5. Output Generation
- Multiple format support
- Comprehensive reporting
- Audit trail creation
- Export capabilities

---

## Common Patterns

### Pattern 1: Agent Creation
```python
from greenlang.sdk.base import Agent, Result, Metadata

class MyAgent(Agent[InputType, OutputType]):
    def __init__(self):
        metadata = Metadata(id="my_agent", name="My Agent", version="1.0.0")
        super().__init__(metadata)

    def validate(self, input_data: InputType) -> bool:
        # Validation logic
        return True

    def process(self, input_data: InputType) -> OutputType:
        # Processing logic
        return result
```

### Pattern 2: Error Handling
```python
result = agent.run(input_data)

if result.success:
    # Process successful result
    data = result.data
else:
    # Handle error
    print(f"Error: {result.error}")
```

### Pattern 3: Batch Processing
```python
results = []
errors = []

for item in batch:
    result = agent.run(item)
    if result.success:
        results.append(result.data)
    else:
        errors.append({"item": item, "error": result.error})
```

### Pattern 4: Pipeline Composition
```python
# Stage 1
stage1_result = agent1.run(input_data)

# Stage 2 (uses stage 1 output)
if stage1_result.success:
    stage2_result = agent2.run(stage1_result.data)

# Stage 3
if stage2_result.success:
    stage3_result = agent3.run(stage2_result.data)
```

---

## Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Solution: Install GreenLang
pip install -e .
```

**Issue: File not found errors**
```bash
# Solution: Run from project root
cd /path/to/Code-V1_GreenLang
python examples/01_simple_agent.py
```

**Issue: Permission errors on output**
```bash
# Solution: Ensure write permissions
chmod -R 755 examples/out/
```

---

## Next Steps

### After Running Examples

1. **Modify the examples** to use your own data
2. **Combine patterns** from multiple examples
3. **Create custom agents** for your use case
4. **Build pipelines** specific to your workflow
5. **Deploy to production** following the deployment guide

### Learn More

- [GreenLang Documentation](../../docs/)
- [SDK Reference](../../core/greenlang/sdk/)
- [Agent Development Guide](../../docs/agent-development.md)
- [Best Practices](../../docs/best-practices.md)

---

## Contributing

Found an issue or want to add an example?

1. Open an issue on GitHub
2. Submit a pull request
3. Join our Discord community

---

## License

MIT License - see LICENSE file for details

---

## Support

- **Documentation:** [greenlang.io/docs](https://greenlang.io/docs)
- **GitHub:** [github.com/greenlang/greenlang](https://github.com/greenlang/greenlang)
- **Discord:** [discord.gg/greenlang](https://discord.gg/greenlang)
- **Email:** support@greenlang.io

---

**Built with GreenLang - The Climate Operating System**

*Save the planet, one calculation at a time.*
