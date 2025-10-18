# Examples Gallery - Quick Start Guide

## Overview

This directory contains 10 production-ready examples demonstrating GreenLang's capabilities for building climate-aware applications. Each example is fully runnable and includes sample data.

## File Structure

```
examples/
├── data/                              # Sample data files
│   ├── sample_buildings.csv          # Building energy data (5 buildings)
│   ├── sample_energy.csv             # Time-series energy data
│   ├── emission_factors.json         # Emission factors (US, UK, CA)
│   └── calculation_inputs.json       # Structured input data
│
├── 01_simple_agent.py                # Basic agent creation
├── 02_data_processor.py              # CSV batch processing
├── 03_calculator_cached.py           # Caching & determinism
├── 04_multi_format_reporter.py       # Multi-format reports
├── 05_provenance_tracking.py         # Audit trail & provenance
├── 06_validation_framework.py        # Schema & business rules
├── 07_unit_converter.py              # Unit conversions
├── 08_parallel_processing.py         # Parallel batch processing
├── 09_custom_decorators.py           # Custom decorators
├── 10_complete_pipeline.py           # Multi-agent pipeline
│
├── gallery_README.md                 # Comprehensive documentation
├── QUICK_START.md                    # This file
└── out/                              # Generated outputs (created on run)
```

## Quick Reference

| # | Example | Key Concepts | Run Time |
|---|---------|--------------|----------|
| 01 | Simple Agent | Agent basics, validation, error handling | <1s |
| 02 | Data Processor | CSV processing, batch operations, stats | <1s |
| 03 | Calculator Cached | Caching, determinism, performance | ~2s |
| 04 | Multi-Format Reporter | Markdown, HTML, JSON, CSV output | <1s |
| 05 | Provenance Tracking | Audit trail, hashing, reproducibility | ~1s |
| 06 | Validation Framework | Schema, business rules, data quality | <1s |
| 07 | Unit Converter | Energy/area units, normalization | <1s |
| 08 | Parallel Processing | ThreadPool, performance optimization | ~3s |
| 09 | Custom Decorators | @cached, @timed, @traced, @validated | ~2s |
| 10 | Complete Pipeline | Multi-agent orchestration, end-to-end | ~1s |

## Running Examples

### Run a Single Example

```bash
# Navigate to project root
cd /path/to/Code-V1_GreenLang

# Run any example
python examples/01_simple_agent.py
python examples/02_data_processor.py
# etc.
```

### Run All Examples

```bash
# Linux/Mac
for f in examples/0[0-9]_*.py; do
    echo "Running $f..."
    python "$f"
    echo ""
done

# Windows PowerShell
Get-ChildItem examples\0*_*.py | ForEach-Object {
    Write-Host "Running $_..."
    python $_
}
```

## Expected Outputs

### Console Output
All examples print formatted output showing:
- Input data
- Processing steps
- Results/statistics
- File paths for generated outputs

### Generated Files
Examples create output in `examples/out/`:

```
examples/out/
├── batch_processing_results.json       # Example 02
├── reports/                            # Example 04
│   ├── emissions_report.md
│   ├── emissions_report.html
│   ├── emissions_report.json
│   └── emissions_breakdown.csv
├── ledger/                             # Example 05
│   ├── calculations.jsonl
│   └── ledger_export.json
└── pipeline_reports/                   # Example 10
    ├── emissions_report.md
    ├── emissions_report.json
    └── emissions_summary.csv
```

## Common Use Cases

### "I want to calculate emissions for a single building"
→ Start with **Example 01** (Simple Agent)

### "I need to process a CSV file with many buildings"
→ Start with **Example 02** (Data Processor)

### "I need reproducible, auditable calculations"
→ Start with **Example 05** (Provenance Tracking)

### "I need to generate reports in multiple formats"
→ Start with **Example 04** (Multi-Format Reporter)

### "I need to process large batches quickly"
→ Start with **Example 08** (Parallel Processing)

### "I want to build a complete workflow"
→ Start with **Example 10** (Complete Pipeline)

## Modifying Examples

### Use Your Own Data

Replace the sample data files:

```python
# In any example, change the data path:
data_file = Path(__file__).parent / "data" / "your_data.csv"
```

### Adjust Emission Factors

Edit `examples/data/emission_factors.json`:

```json
{
  "factors": {
    "electricity": {
      "YOUR_COUNTRY": {
        "value": 0.500,
        "unit": "kgCO2e/kWh",
        "source": "Your Source",
        "year": 2025
      }
    }
  }
}
```

### Combine Multiple Examples

```python
# Combine patterns from multiple examples
from examples.example_01_simple_agent import EmissionsCalculatorAgent
from examples.example_04_multi_format_reporter import EmissionsReporter

# Calculate
agent = EmissionsCalculatorAgent()
result = agent.run(your_data)

# Report
reporter = EmissionsReporter()
report = reporter.generate(result.data, format="markdown")
```

## Troubleshooting

### Import Errors

```bash
# Install GreenLang in development mode
pip install -e .

# Verify installation
python -c "from greenlang.sdk.base import Agent; print('Success!')"
```

### File Not Found

```bash
# Ensure you're in the project root
pwd  # Should show .../Code-V1_GreenLang

# Check data files exist
ls examples/data/
```

### Permission Errors

```bash
# Create output directory
mkdir -p examples/out

# Set permissions (Linux/Mac)
chmod -R 755 examples/out/
```

## Key Concepts Summary

### Agents
- Stateless computation units
- Type-safe input/output
- Built-in validation and error handling
- Composable into pipelines

### Validation
- Schema validation (structure)
- Business rules (logic)
- Data quality checks
- Multi-stage validation

### Performance
- Function caching (LRU)
- Parallel processing (ThreadPool)
- Batching strategies
- Benchmarking patterns

### Reproducibility
- Deterministic execution
- Seeded randomness
- Input/output hashing
- Audit trails

### Reporting
- Multiple output formats
- Template-based generation
- Data export (JSON, CSV)
- Visualization-ready output

## Next Steps

1. **Run all examples** to see the full range of capabilities
2. **Read gallery_README.md** for detailed documentation
3. **Modify examples** to work with your data
4. **Build your own agents** following the patterns
5. **Create pipelines** combining multiple agents

## Resources

- **Full Documentation:** `gallery_README.md`
- **GreenLang SDK:** `../../core/greenlang/sdk/`
- **Main README:** `../../README.md`
- **Agent Development:** `../../docs/agent-development.md`

## Support

Questions? Issues?
- GitHub: https://github.com/greenlang/greenlang
- Discord: https://discord.gg/greenlang
- Email: support@greenlang.io

---

**Happy Building!**

*Built with GreenLang - The Climate Operating System*
