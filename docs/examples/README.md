# GreenLang Framework Examples

This directory contains 10 comprehensive, runnable examples demonstrating the GreenLang agent framework (v0.3.0).

## Quick Start

Each example is self-contained and can be run independently:

```bash
# Run any example
python example_01_basic_agent.py
python example_02_data_processor.py
# ... etc
```

## Examples Overview

### Example 01: Basic Agent
**File:** `example_01_basic_agent.py`

Learn the fundamentals of creating a simple agent.

**Concepts:**
- Creating custom agents
- Implementing the execute method
- Input validation
- Running agents and checking results
- Agent statistics

**Difficulty:** Beginner

---

### Example 02: Data Processor Agent
**File:** `example_02_data_processor.py`

Process data in batches with automatic error handling.

**Concepts:**
- Batch processing
- Record-level validation
- Error collection
- Progress tracking
- Parallel vs sequential processing

**Difficulty:** Beginner

---

### Example 03: Calculator Agent
**File:** `example_03_calculator.py`

Perform high-precision mathematical calculations with caching.

**Concepts:**
- High-precision decimal arithmetic
- Calculation step tracking
- Result caching
- Input validation
- Division by zero handling

**Difficulty:** Intermediate

---

### Example 04: Reporter Agent
**File:** `example_04_reporter.py`

Generate professional reports in multiple formats.

**Concepts:**
- Data aggregation
- Multi-section reports
- Markdown/HTML/JSON output
- Custom report styling
- Summary generation

**Difficulty:** Intermediate

---

### Example 05: Batch Processing
**File:** `example_05_batch_processing.py`

Efficiently process large datasets.

**Concepts:**
- Large dataset handling
- Batch size configuration
- Error handling at scale
- Performance optimization
- Aggregated results

**Difficulty:** Intermediate

---

### Example 06: Parallel Processing
**File:** `example_06_parallel_processing.py`

Improve performance with parallel processing.

**Concepts:**
- Sequential vs parallel execution
- Worker configuration
- Performance benchmarking
- Thread safety
- Optimal configuration

**Difficulty:** Advanced

---

### Example 07: Custom Validation
**File:** `example_07_custom_validation.py`

Implement sophisticated validation logic.

**Concepts:**
- Multi-level validation
- Detailed error messages
- Business rule validation
- Data quality checks
- Warning vs error handling

**Difficulty:** Intermediate

---

### Example 08: Agent with Provenance
**File:** `example_08_with_provenance.py`

Track provenance for regulatory compliance.

**Concepts:**
- Lifecycle hooks
- Input/output tracking
- Audit trail generation
- Provenance export
- Data integrity verification

**Difficulty:** Advanced

---

### Example 09: Multi-Format Reports
**File:** `example_09_multi_format_reports.py`

Generate reports in Markdown, HTML, and JSON.

**Concepts:**
- Multi-format output
- Custom styling
- Professional layouts
- Format comparison
- Stakeholder reports

**Difficulty:** Intermediate

---

### Example 10: CBAM Pipeline
**File:** `example_10_cbam_pipeline.py`

Complete CBAM (Carbon Border Adjustment Mechanism) compliance pipeline.

**Concepts:**
- Multi-agent pipelines
- Agent chaining
- Regulatory compliance
- Real-world use case
- Code reduction (680→230 lines)

**Difficulty:** Advanced

---

## Learning Path

### For Beginners
Start with these examples in order:
1. Example 01: Basic Agent
2. Example 02: Data Processor Agent
3. Example 04: Reporter Agent

### For Intermediate Users
After completing the beginner examples:
4. Example 03: Calculator Agent
5. Example 05: Batch Processing
6. Example 07: Custom Validation
7. Example 09: Multi-Format Reports

### For Advanced Users
For complex scenarios and production use:
8. Example 06: Parallel Processing
9. Example 08: Agent with Provenance
10. Example 10: CBAM Pipeline

---

## Running the Examples

### Requirements

```bash
# Install GreenLang with examples dependencies
pip install greenlang-cli==0.3.0

# Optional: Install tqdm for progress bars
pip install tqdm

# Optional: Install openpyxl for Excel reports
pip install openpyxl
```

### Running Individual Examples

```bash
# Navigate to examples directory
cd docs/examples

# Run an example
python example_01_basic_agent.py

# Run with verbose output (if supported)
python example_06_parallel_processing.py --verbose
```

### Expected Output

Each example produces:
- Console output showing execution steps
- Success/failure indicators (✓/✗)
- Performance metrics
- Statistics summaries

Some examples also create files:
- `example_04`: Creates `energy_report.html`
- `example_08`: Creates `provenance_output/` directory
- `example_09`: Creates `reports_output/` directory
- `example_10`: Creates `cbam_reports/` directory

---

## Key Concepts Demonstrated

### Agent Lifecycle
- `initialize()`: Setup
- `validate_input()`: Input validation
- `preprocess()`: Data preprocessing
- `execute()`: Main logic
- `postprocess()`: Result postprocessing
- `cleanup()`: Resource cleanup

### Configuration
All agents use configuration objects:
- `AgentConfig`: Base configuration
- `DataProcessorConfig`: Batch processing settings
- `CalculatorConfig`: Precision and caching
- `ReporterConfig`: Output format settings

### Results
All agents return `AgentResult` objects with:
- `success`: Boolean status
- `data`: Output data
- `error`: Error message (if failed)
- `metadata`: Additional information
- `metrics`: Performance metrics
- `timestamp`: Execution timestamp

### Statistics
All agents track statistics:
- Execution count
- Success/failure rate
- Average execution time
- Custom metrics (cache hits, records processed, etc.)

---

## Common Patterns

### Pattern 1: Basic Agent
```python
class MyAgent(BaseAgent):
    def __init__(self):
        config = AgentConfig(name="MyAgent", description="...")
        super().__init__(config)

    def execute(self, input_data):
        # Your logic here
        return AgentResult(success=True, data={...})
```

### Pattern 2: Data Processor
```python
class MyProcessor(BaseDataProcessor):
    def process_record(self, record):
        # Transform single record
        return transformed_record

    def validate_record(self, record):
        # Validate single record
        return is_valid
```

### Pattern 3: Calculator
```python
class MyCalculator(BaseCalculator):
    def calculate(self, inputs):
        # Perform calculation
        result = compute(inputs)

        # Track calculation step
        self.add_calculation_step(
            step_name="Step 1",
            formula="a + b",
            inputs=inputs,
            result=result
        )

        return result
```

### Pattern 4: Reporter
```python
class MyReporter(BaseReporter):
    def aggregate_data(self, input_data):
        # Aggregate data
        return aggregated

    def build_sections(self, aggregated_data):
        # Build report sections
        return [section1, section2, ...]
```

---

## Troubleshooting

### Import Errors

```bash
# If you get import errors, ensure GreenLang is installed
pip install greenlang-cli==0.3.0

# Or install from source
pip install -e /path/to/greenlang
```

### Missing Dependencies

```bash
# For progress bars
pip install tqdm

# For Excel reports (Example 04)
pip install openpyxl
```

### Permission Errors

If you get permission errors when creating output directories:

```bash
# Run with appropriate permissions
# Or change output directory in the example code
```

---

## Next Steps

After completing these examples:

1. **Read the API Reference**: See `docs/API_REFERENCE.md` for complete API documentation
2. **Review Migration Guide**: See `docs/MIGRATION_GUIDE.md` for migrating existing code
3. **Check Architecture Guide**: See `docs/ARCHITECTURE.md` for design principles
4. **Join the Community**: Discord, GitHub Discussions

---

## Contributing

Found a bug or have an improvement? Please:
1. Open an issue on GitHub
2. Submit a pull request
3. Join our Discord community

---

## License

These examples are part of the GreenLang project and are released under the MIT License.

---

**GreenLang v0.3.0 - The Climate Intelligence Platform**

For more information, visit: [github.com/greenlang/greenlang](https://github.com/greenlang/greenlang)
