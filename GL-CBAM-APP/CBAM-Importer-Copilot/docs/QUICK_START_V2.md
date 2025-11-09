# GL-CBAM-APP v2 Quick Start Guide

## TL;DR

```bash
# Run v2 pipeline (same API as v1)
python cbam_pipeline_v2.py \
  --input examples/demo_shipments.csv \
  --output output/cbam_report.json \
  --summary output/cbam_summary.md \
  --importer-name "Acme Steel EU BV" \
  --importer-country NL \
  --importer-eori NL123456789012 \
  --declarant-name "John Smith" \
  --declarant-position "Compliance Officer"
```

## What's New in v2?

### 1. Framework Integration
- **Built on GreenLang SDK v0.3.0**
- Inherits from `Agent` and `Pipeline` base classes
- Automatic error handling and metrics collection

### 2. New Features
- **Multi-format Export**: JSON, Excel, CSV, PDF
- **Prometheus Metrics**: Production monitoring out-of-the-box
- **Type Safety**: Pydantic models for all inputs/outputs
- **Async Support**: Ready for large-scale processing

### 3. Same Business Logic
- ✅ Same CBAM validation rules
- ✅ Same emission calculations (ZERO HALLUCINATION)
- ✅ Same report format
- ✅ 100% backward compatible

## Usage Examples

### Basic Usage (v1-compatible)

```python
from cbam_pipeline_v2 import CBAMPipeline_v2

# Initialize (same as v1)
pipeline = CBAMPipeline_v2(
    cn_codes_path="data/cn_codes.json",
    cbam_rules_path="rules/cbam_rules.yaml",
    suppliers_path="examples/demo_suppliers.yaml"
)

# Run (same API as v1)
report = pipeline.run(
    input_file="examples/demo_shipments.csv",
    importer_info={
        "importer_name": "Acme Steel EU BV",
        "importer_country": "NL",
        "importer_eori": "NL123456789012",
        "declarant_name": "John Smith",
        "declarant_position": "Compliance Officer"
    },
    output_report_path="output/report.json",
    output_summary_path="output/summary.md"
)
```

### NEW: Multi-format Export

```python
from reporting_packager_agent_v2 import ReportingPackagerAgent_v2, ReportFormat
import asyncio

agent = ReportingPackagerAgent_v2()

# Generate report
report = agent.generate_report(shipments, importer_info)

# Export to Excel (NEW in v2)
async def export_excel():
    result = await agent.export_report(
        report=report,
        format=ReportFormat.EXCEL,
        output_path="output/cbam_report.xlsx"
    )

asyncio.run(export_excel())
```

### NEW: Prometheus Metrics

```python
# Enable metrics collection
pipeline = CBAMPipeline_v2(
    ...,
    enable_metrics=True
)

# Metrics available at http://localhost:8000/metrics
# - gl_pipeline_runs_total
# - gl_pipeline_duration_seconds
# - gl_active_executions
# - gl_cpu_usage_percent
# - gl_memory_usage_bytes
```

## Agent-Level Usage

### ShipmentIntakeAgent_v2

```python
from shipment_intake_agent_v2 import ShipmentIntakeAgent_v2

agent = ShipmentIntakeAgent_v2(
    cn_codes_path="data/cn_codes.json",
    cbam_rules_path="rules/cbam_rules.yaml",
    suppliers_path="examples/demo_suppliers.yaml"
)

# Process file (v1-compatible API)
result = agent.process_file("examples/demo_shipments.csv")

# Or use framework API
from shipment_intake_agent_v2 import IntakeInput
input_data = IntakeInput(file_path="examples/demo_shipments.csv")
result = agent.run(input_data)

print(f"Valid: {result.data.metadata['valid_records']}")
print(f"Invalid: {result.data.metadata['invalid_records']}")
```

### EmissionsCalculatorAgent_v2

```python
from emissions_calculator_agent_v2 import EmissionsCalculatorAgent_v2

agent = EmissionsCalculatorAgent_v2(
    suppliers_path="examples/demo_suppliers.yaml"
)

# Calculate emissions (v1-compatible API)
result = agent.calculate_batch(validated_shipments)

print(f"Total emissions: {result['metadata']['total_emissions_tco2']:.2f} tCO2")
```

### ReportingPackagerAgent_v2

```python
from reporting_packager_agent_v2 import ReportingPackagerAgent_v2

agent = ReportingPackagerAgent_v2(
    cbam_rules_path="rules/cbam_rules.yaml"
)

# Generate report (v1-compatible API)
report = agent.generate_report(
    shipments_with_emissions=calculated_shipments,
    importer_info=importer_info
)

# Write outputs
agent.write_report(report, "output/cbam_report.json")
agent.write_summary(report, "output/cbam_summary.md")
```

## Testing

### Run Integration Tests

```bash
# All tests
pytest tests/test_v2_integration.py -v

# Specific test
pytest tests/test_v2_integration.py::test_pipeline_v2_end_to_end -v

# With coverage
pytest tests/test_v2_integration.py --cov=agents --cov-report=html
```

### Expected Results

```
✓ Intake agent v2 processes X shipments correctly
✓ Intake agent v2 applies same validation (Y errors detected)
✓ Calculator v2 produces identical emissions: Z.ZZ tCO2
✓ Calculator v2 is deterministic (ZERO HALLUCINATION verified)
✓ Packager v2 generates complete CBAM report with all sections
✓ Packager v2 applies same validation rules
✓ Pipeline v2 executes end-to-end successfully
✓ Pipeline v2 is backward compatible with v1
✓ Performance comparison: v1=X.XXs, v2=Y.YYs (overhead: Z.Zx)
```

## Environment Setup

### Python Path

```bash
# Option 1: Export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/Code-V1_GreenLang

# Option 2: Use sys.path in code
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Dependencies

```bash
# Install core dependencies
pip install pandas>=2.0.0 pydantic>=2.0.0 PyYAML>=6.0 openpyxl>=3.1.0

# Install monitoring (optional)
pip install prometheus-client>=0.19.0 psutil>=5.9.0

# Install testing (optional)
pip install pytest>=7.4.0 pytest-cov>=4.1.0
```

## Performance

### v2 vs v1 Comparison

| Metric | v1 | v2 | Change |
|--------|----|----|--------|
| Code Size | 2,531 LOC | 2,136 LOC | -15.6% |
| Custom Code | 98.1% | 42.7% | -55.4% |
| Execution Time* | 1.0x | 1.1x | +10% |
| Memory Usage* | 1.0x | 1.0x | Same |

*For 1,000 shipments on standard hardware

### Performance Tips

1. **Disable metrics in development**:
   ```python
   pipeline = CBAMPipeline_v2(..., enable_metrics=False)
   ```

2. **Use batch processing for large files**:
   ```python
   # Process in chunks for files > 10,000 shipments
   chunk_size = 1000
   ```

3. **Enable async for I/O-bound operations**:
   ```python
   # Use async export for large reports
   await agent.export_report(report, format=ReportFormat.PDF)
   ```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: greenlang

**Problem**: Cannot import greenlang modules

**Solution**:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Code-V1_GreenLang
# Or add to ~/.bashrc for permanent fix
```

#### 2. Agent execution failed: Input validation failed

**Problem**: Invalid input file format

**Solution**: Ensure input file has required columns:
- shipment_id
- import_date
- quarter
- cn_code
- origin_iso
- net_mass_kg

#### 3. Prometheus metrics not available

**Problem**: Metrics endpoint returns 404

**Solution**:
```python
# Enable metrics explicitly
pipeline = CBAMPipeline_v2(..., enable_metrics=True)

# Check metrics are being collected
from greenlang.telemetry.metrics import get_metrics_collector
collector = get_metrics_collector()
print(collector.get_metrics())
```

## Migration from v1

### Step 1: Update Imports

```python
# Old (v1)
from cbam_pipeline import CBAMPipeline

# New (v2)
from cbam_pipeline_v2 import CBAMPipeline_v2
```

### Step 2: Update Instantiation (Optional)

```python
# v1 syntax still works
pipeline = CBAMPipeline_v2(...)

# Or use new features
pipeline = CBAMPipeline_v2(..., enable_metrics=True)
```

### Step 3: Run (No Changes)

```python
# Same API as v1
report = pipeline.run(...)
```

## Best Practices

### 1. Use Type Hints

```python
from typing import Dict, List
from cbam_pipeline_v2 import CBAMPipeline_v2

def process_cbam_data(
    input_file: str,
    importer_info: Dict[str, str]
) -> Dict:
    pipeline = CBAMPipeline_v2(...)
    return pipeline.run(input_file, importer_info)
```

### 2. Handle Errors Gracefully

```python
from greenlang.sdk.base import Result

result = agent.run(input_data)
if not result.success:
    logger.error(f"Agent failed: {result.error}")
    # Implement retry logic or fallback
else:
    data = result.data
```

### 3. Monitor in Production

```python
# Enable metrics
pipeline = CBAMPipeline_v2(..., enable_metrics=True)

# Set up alerts
# - Pipeline failure rate > 1%
# - Execution time > 60s (p95)
# - Memory usage > 500MB
```

### 4. Test Before Deploying

```bash
# Run full test suite
pytest tests/test_v2_integration.py -v

# Compare v1 and v2 outputs
python -m tests.compare_v1_v2 --input examples/demo_shipments.csv
```

## Support

### Documentation
- [V2 Migration Complete Report](V2_MIGRATION_COMPLETE.md)
- [GreenLang SDK Documentation](../../greenlang/README.md)

### Issues
- GitHub Issues: [Link to repository]
- Slack: #cbam-support

### Contributing
- See CONTRIBUTING.md for development guidelines
- Submit pull requests to `develop` branch

---

**Version**: 2.0.0
**Last Updated**: 2025-11-09
**Maintainer**: GreenLang CBAM Team
