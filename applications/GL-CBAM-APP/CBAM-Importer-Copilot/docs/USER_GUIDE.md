# CBAM Importer Copilot - User Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**Target Audience:** Compliance officers, developers, data analysts

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Using the CLI](#using-the-cli)
5. [Using the Python SDK](#using-the-python-sdk)
6. [Configuration](#configuration)
7. [Input Data Formats](#input-data-formats)
8. [Understanding the Output](#understanding-the-output)
9. [Advanced Usage](#advanced-usage)
10. [Best Practices](#best-practices)
11. [FAQ](#faq)

---

## Introduction

### What is CBAM Importer Copilot?

CBAM Importer Copilot is an enterprise-grade automation tool for generating **EU CBAM Transitional Registry** reports. It transforms shipment data into compliant CBAM reports in minutes, not weeks.

### Key Features

✅ **Zero Hallucination Architecture**
- 100% deterministic calculations
- No LLM-generated emission factors
- Bit-perfect reproducibility
- Complete audit trail

✅ **Developer-Friendly**
- One-command CLI: `gl cbam report`
- 5-line Python SDK
- Works with CSV, Excel, JSON
- Pandas DataFrame support

✅ **Regulatory Compliance**
- SHA256 file integrity verification
- Complete execution environment capture
- Agent execution audit trail
- Meets EU CBAM requirements

✅ **Performance**
- 10,000 shipments in <30 seconds
- 20× faster than manual processing
- Efficient pandas aggregations
- Zero external API calls in hot path

### Who Should Use This?

- **Compliance Officers**: Generate CBAM reports with one command
- **Data Analysts**: Integrate with existing data pipelines
- **Developers**: Embed in ERP/SAP systems via Python SDK
- **Importers**: Multi-tenant support for multiple legal entities

---

## Quick Start

### 3-Minute Quickstart (CLI)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create configuration
gl cbam config init

# 3. Generate report
gl cbam report examples/demo_shipments.csv \
  --importer-name "Acme Steel EU BV" \
  --importer-country "NL" \
  --importer-eori "NL123456789012"

# Done! Report saved to output/cbam_report.json
```

### 5-Line Quickstart (Python SDK)

```python
from sdk.cbam_sdk import cbam_build_report

report = cbam_build_report(
    input_file="examples/demo_shipments.csv",
    importer_name="Acme Steel EU BV",
    importer_country="NL",
    importer_eori="NL123456789012"
)

print(f"Total emissions: {report.total_emissions_tco2:.2f} tCO2")
```

---

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, Windows
- **Memory**: 2GB RAM minimum (4GB recommended for large datasets)
- **Disk Space**: 500MB for dependencies + output files

### Standard Installation

```bash
# Clone repository (if not already done)
cd GL-Applications/CBAM-Importer-Copilot

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
gl cbam --version
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pre-commit install

# Run tests
pytest tests/

# Run linting
ruff check .
```

### Docker Installation (Optional)

```bash
# Build image
docker build -t cbam-copilot:1.0.0 .

# Run container
docker run -v $(pwd)/data:/app/data cbam-copilot:1.0.0 \
  gl cbam report /app/data/shipments.csv
```

---

## Using the CLI

### Overview

The CLI provides three main commands:

1. **`gl cbam report`** - Generate CBAM reports
2. **`gl cbam config`** - Manage configuration
3. **`gl cbam validate`** - Validate shipment data

### Command: `gl cbam report`

Generate a complete CBAM Transitional Registry report.

#### Basic Usage

```bash
gl cbam report INPUT_FILE \
  --importer-name "Your Company Name" \
  --importer-country "NL" \
  --importer-eori "NL123456789012"
```

#### Full Options

```bash
gl cbam report INPUT_FILE \
  --importer-name TEXT              # Legal entity name (required)
  --importer-country CODE           # ISO 2-letter country code (required)
  --importer-eori TEXT              # EORI number (required)
  --declarant-name TEXT             # Person filing report
  --declarant-position TEXT         # Job title
  --config FILE                     # Path to .cbam.yaml config file
  --output FILE                     # Output path (default: output/cbam_report.json)
  --format [json|excel|both]        # Output format (default: both)
  --verbose                         # Show detailed progress
```

#### Examples

**Example 1: Minimal Report**

```bash
gl cbam report data/shipments.csv \
  --importer-name "Acme Steel BV" \
  --importer-country "NL" \
  --importer-eori "NL123456789012"
```

**Example 2: Complete Report with Declarant**

```bash
gl cbam report data/q4_shipments.xlsx \
  --importer-name "Acme Steel BV" \
  --importer-country "NL" \
  --importer-eori "NL123456789012" \
  --declarant-name "John Smith" \
  --declarant-position "Compliance Officer" \
  --format both \
  --verbose
```

**Example 3: Using Configuration File**

```bash
# Create config once
gl cbam config init

# Use config for all subsequent reports
gl cbam report data/shipments.csv --config .cbam.yaml
```

#### Output

The command generates:

1. **`cbam_report.json`** - EU Registry format (machine-readable)
2. **`cbam_summary.md`** - Human-readable summary
3. **Console output** - Beautiful progress bars and summary

### Command: `gl cbam config`

Manage CBAM configuration files.

#### Initialize Configuration

```bash
gl cbam config init

# Creates .cbam.yaml in current directory
```

#### Show Current Configuration

```bash
gl cbam config show

# Displays current configuration with source (file vs env vars)
```

#### Validate Configuration

```bash
gl cbam config validate

# Checks configuration for errors
```

#### Configuration File Structure

```yaml
importer:
  name: "Acme Steel EU BV"
  country: "NL"
  eori: "NL123456789012"

declarant:
  name: "John Smith"
  position: "Compliance Officer"

paths:
  cn_codes: "data/cn_codes.json"
  rules: "rules/cbam_rules.yaml"
  suppliers: "examples/demo_suppliers.yaml"

output:
  directory: "output"
  format: "both"  # json, excel, or both
```

### Command: `gl cbam validate`

Validate shipment data without generating a report (dry-run).

```bash
gl cbam validate INPUT_FILE [OPTIONS]

# Options same as 'report' command
```

#### Use Cases

- Check data quality before filing
- Identify missing or invalid fields
- Verify CN codes and supplier links
- Test new data sources

#### Example Output

```
✓ Validation Summary
  - Total records: 1,250
  - Valid records: 1,200 (96%)
  - Errors: 25
  - Warnings: 50

Errors by type:
  - E001 (Missing CN code): 15 records
  - E002 (Invalid country): 10 records

Warnings by type:
  - W001 (Missing supplier data): 50 records
```

---

## Using the Python SDK

### Overview

The Python SDK provides programmatic access for:
- ERP system integration
- Batch processing
- Custom workflows
- Automated reporting pipelines

### Core Functions

#### `cbam_build_report()`

Main function to generate CBAM reports.

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

report = cbam_build_report(
    input_file: Optional[str] = None,          # CSV/Excel/JSON file
    input_dataframe: Optional[pd.DataFrame] = None,  # Or pandas DataFrame
    importer_name: Optional[str] = None,
    importer_country: Optional[str] = None,
    importer_eori: Optional[str] = None,
    declarant_name: Optional[str] = None,
    declarant_position: Optional[str] = None,
    config: Optional[CBAMConfig] = None,       # Reusable config object
    output_path: Optional[str] = None,
    save_output: bool = True
) -> CBAMReport
```

**Returns:** `CBAMReport` object with:
- `raw_report` - Complete report dictionary
- `total_emissions_tco2` - Total embedded emissions
- `total_shipments` - Number of shipments
- `unique_cn_codes` - Unique CN codes
- `to_dataframe()` - Convert to pandas DataFrame
- `save()` - Save to file

#### `cbam_validate_shipments()`

Validate shipment data without generating report.

```python
from sdk.cbam_sdk import cbam_validate_shipments

validation = cbam_validate_shipments(
    input_file="data/shipments.csv",
    importer_country="NL"
)

print(f"Valid: {validation['metadata']['valid_records']}")
print(f"Errors: {validation['metadata']['error_count']}")
```

#### `cbam_calculate_emissions()`

Calculate emissions for validated shipments.

```python
from sdk.cbam_sdk import cbam_calculate_emissions

emissions = cbam_calculate_emissions(
    validated_shipments=validated_data,
    cn_codes_path="data/cn_codes.json"
)

print(f"Total: {emissions['total_emissions_tco2']:.2f} tCO2")
```

### SDK Examples

#### Example 1: Basic File Processing

```python
from sdk.cbam_sdk import cbam_build_report

# Process a CSV file
report = cbam_build_report(
    input_file="data/Q4_2025_shipments.csv",
    importer_name="Acme Steel BV",
    importer_country="NL",
    importer_eori="NL123456789012"
)

print(f"Report generated for {report.total_shipments} shipments")
print(f"Total emissions: {report.total_emissions_tco2:.2f} tCO2")
```

#### Example 2: Using Config Objects (Recommended for Reuse)

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Create reusable config
config = CBAMConfig(
    importer_name="Acme Steel BV",
    importer_country="NL",
    importer_eori="NL123456789012",
    declarant_name="John Smith",
    declarant_position="Compliance Officer"
)

# Generate reports for multiple periods
for month in ["jan", "feb", "mar"]:
    report = cbam_build_report(
        input_file=f"data/2025_{month}.csv",
        config=config
    )
    print(f"{month}: {report.total_emissions_tco2:.2f} tCO2")
```

#### Example 3: DataFrame Integration

```python
import pandas as pd
from sdk.cbam_sdk import cbam_build_report

# Load from database or data warehouse
df = pd.read_sql("SELECT * FROM shipments WHERE quarter='Q4'", conn)

# Generate report from DataFrame
report = cbam_build_report(
    input_dataframe=df,
    importer_name="Acme Steel BV",
    importer_country="NL",
    importer_eori="NL123456789012"
)

# Convert results back to DataFrame
results_df = report.to_dataframe()
results_df.to_csv("output/processed_shipments.csv")
```

#### Example 4: Validation Before Processing

```python
from sdk.cbam_sdk import cbam_validate_shipments, cbam_build_report

# Validate first
validation = cbam_validate_shipments(
    input_file="data/shipments.csv",
    importer_country="NL"
)

if validation['metadata']['error_count'] == 0:
    # No errors, proceed with report
    report = cbam_build_report(
        input_file="data/shipments.csv",
        importer_name="Acme Steel BV",
        importer_country="NL",
        importer_eori="NL123456789012"
    )
else:
    # Handle errors
    for error in validation['errors']:
        print(f"Row {error['record_index']}: {error['message']}")
```

#### Example 5: ERP Integration Pattern

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig
import schedule
import time

# Setup reusable config
config = CBAMConfig.from_yaml(".cbam.yaml")

def generate_monthly_report():
    """Run on 1st of each month"""
    try:
        # Extract from ERP
        df = extract_shipments_from_erp()

        # Generate report
        report = cbam_build_report(
            input_dataframe=df,
            config=config,
            output_path=f"reports/cbam_{datetime.now().strftime('%Y%m')}.json"
        )

        # Notify compliance team
        send_notification(f"CBAM report ready: {report.total_emissions_tco2} tCO2")

    except Exception as e:
        send_alert(f"CBAM report failed: {e}")

# Schedule monthly
schedule.every().month.at("09:00").do(generate_monthly_report)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

---

## Configuration

### Configuration Sources

CBAM Importer Copilot reads configuration from multiple sources (in order of precedence):

1. **Command-line arguments** (highest priority)
2. **Configuration file** (`.cbam.yaml`)
3. **Environment variables**
4. **Defaults** (lowest priority)

### Configuration File Format

#### Location

- Default: `.cbam.yaml` in current directory
- Custom: Specify with `--config` flag

#### Structure

```yaml
# Importer information (required)
importer:
  name: "Acme Steel EU BV"
  country: "NL"
  eori: "NL123456789012"

# Declarant information (optional but recommended)
declarant:
  name: "John Smith"
  position: "Compliance Officer"
  email: "john.smith@acme.eu"    # optional
  phone: "+31 20 1234567"         # optional

# Data file paths
paths:
  cn_codes: "data/cn_codes.json"
  rules: "rules/cbam_rules.yaml"
  suppliers: "examples/demo_suppliers.yaml"

# Output settings
output:
  directory: "output"
  format: "both"  # json, excel, or both
  include_summary: true
  include_provenance: true

# Performance tuning (optional)
performance:
  chunk_size: 1000              # Process in chunks
  parallel_workers: 4           # For large datasets
  memory_limit_mb: 2048

# Validation settings (optional)
validation:
  strict_mode: false            # Fail on warnings
  allow_missing_suppliers: true
  default_emission_factors: true
```

### Environment Variables

For sensitive data or CI/CD pipelines:

```bash
# Importer info
export CBAM_IMPORTER_NAME="Acme Steel BV"
export CBAM_IMPORTER_COUNTRY="NL"
export CBAM_IMPORTER_EORI="NL123456789012"

# Declarant info
export CBAM_DECLARANT_NAME="John Smith"
export CBAM_DECLARANT_POSITION="Compliance Officer"

# Paths
export CBAM_CN_CODES_PATH="data/cn_codes.json"
export CBAM_RULES_PATH="rules/cbam_rules.yaml"

# Use in command
gl cbam report data/shipments.csv
# (importer info read from env vars)
```

### Multi-Tenant Configuration

For processing reports for multiple legal entities:

```yaml
# config_acme_nl.yaml
importer:
  name: "Acme Steel EU BV"
  country: "NL"
  eori: "NL123456789012"

# config_acme_de.yaml
importer:
  name: "Acme Stahl Deutschland GmbH"
  country: "DE"
  eori: "DE987654321098"
```

```bash
# Process for Dutch entity
gl cbam report data/nl_shipments.csv --config config_acme_nl.yaml

# Process for German entity
gl cbam report data/de_shipments.csv --config config_acme_de.yaml
```

---

## Input Data Formats

### Supported Formats

- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- **JSON** (`.json`)
- **Pandas DataFrame** (Python SDK only)

### Required Fields

Every shipment record MUST include:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cn_code` | String | 8-digit CN code | `72071100` |
| `country_of_origin` | String | ISO 2-letter code | `CN` |
| `quantity_tons` | Float | Net weight in tons | `15.5` |
| `import_date` | String | ISO 8601 date | `2025-09-15` |

### Optional Fields

Recommended for better accuracy:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `supplier_id` | String | Supplier reference | `SUP-CN-001` |
| `installation_id` | String | Installation ID | `INST-001` |
| `actual_emissions_tco2` | Float | Supplier-provided actuals | `12.5` |
| `invoice_number` | String | Invoice reference | `INV-2025-001` |
| `customs_declaration` | String | Customs ref | `MRN123456789` |

### CSV Format Example

```csv
cn_code,country_of_origin,quantity_tons,import_date,supplier_id,invoice_number
72071100,CN,15.5,2025-09-15,SUP-CN-001,INV-2025-001
72071210,TR,8.2,2025-09-20,SUP-TR-002,INV-2025-002
72071290,UA,12.0,2025-09-25,SUP-UA-001,INV-2025-003
```

### Excel Format Example

Excel files can include multiple sheets. The tool will look for a sheet named "Shipments" or use the first sheet.

| cn_code | country_of_origin | quantity_tons | import_date | supplier_id | invoice_number |
|---------|-------------------|---------------|-------------|-------------|----------------|
| 72071100 | CN | 15.5 | 2025-09-15 | SUP-CN-001 | INV-2025-001 |
| 72071210 | TR | 8.2 | 2025-09-20 | SUP-TR-002 | INV-2025-002 |

### JSON Format Example

```json
[
  {
    "cn_code": "72071100",
    "country_of_origin": "CN",
    "quantity_tons": 15.5,
    "import_date": "2025-09-15",
    "supplier_id": "SUP-CN-001",
    "invoice_number": "INV-2025-001"
  },
  {
    "cn_code": "72071210",
    "country_of_origin": "TR",
    "quantity_tons": 8.2,
    "import_date": "2025-09-20",
    "supplier_id": "SUP-TR-002",
    "invoice_number": "INV-2025-002"
  }
]
```

### DataFrame Format (Python SDK)

```python
import pandas as pd

df = pd.DataFrame({
    'cn_code': ['72071100', '72071210', '72071290'],
    'country_of_origin': ['CN', 'TR', 'UA'],
    'quantity_tons': [15.5, 8.2, 12.0],
    'import_date': ['2025-09-15', '2025-09-20', '2025-09-25'],
    'supplier_id': ['SUP-CN-001', 'SUP-TR-002', 'SUP-UA-001']
})

report = cbam_build_report(input_dataframe=df, ...)
```

---

## Understanding the Output

### Output Files

A successful run generates 3 files:

1. **`cbam_report.json`** - EU Registry format (machine-readable)
2. **`cbam_summary.md`** - Human-readable summary
3. **`provenance.json`** - Complete audit trail (optional)

### JSON Report Structure

```json
{
  "report_metadata": {
    "report_id": "CBAM-2025Q4-NL-001",
    "generated_at": "2025-10-15T14:30:00Z",
    "importer": { ... },
    "declarant": { ... },
    "reporting_period": "2025-Q4"
  },

  "emissions_summary": {
    "total_embedded_emissions_tco2": 1234.56,
    "total_quantity_tons": 5000.0,
    "total_shipments": 150,
    "unique_cn_codes": 12,
    "unique_countries": 8,
    "emissions_by_product_group": { ... },
    "emissions_by_country": { ... }
  },

  "detailed_goods": [
    {
      "cn_code": "72071100",
      "description": "Semi-finished products of iron",
      "country_of_origin": "CN",
      "quantity_tons": 15.5,
      "embedded_emissions_tco2": 12.4,
      "emission_factor_source": "default",
      "calculation_method": "deterministic",
      "supplier_id": "SUP-CN-001",
      "import_date": "2025-09-15"
    }
  ],

  "aggregations": {
    "by_cn_code": [ ... ],
    "by_country": [ ... ],
    "by_product_group": [ ... ]
  },

  "validation_results": {
    "is_valid": true,
    "errors": [],
    "warnings": [],
    "complex_goods_percentage": 18.5
  },

  "provenance": {
    "input_file_integrity": { "sha256_hash": "..." },
    "execution_environment": { ... },
    "agent_execution": [ ... ],
    "reproducibility": {
      "deterministic": true,
      "zero_hallucination": true
    }
  }
}
```

### Markdown Summary Example

```markdown
# CBAM Transitional Registry Report

**Report ID:** CBAM-2025Q4-NL-001
**Generated:** 2025-10-15 14:30:00 UTC

## Importer Information
- **Name:** Acme Steel EU BV
- **Country:** Netherlands (NL)
- **EORI:** NL123456789012

## Emissions Summary
- **Total Embedded Emissions:** 1,234.56 tCO2
- **Total Quantity:** 5,000.0 tons
- **Total Shipments:** 150
- **Unique CN Codes:** 12
- **Unique Countries:** 8

## Top 5 Product Groups by Emissions
1. Iron and steel (72): 850.0 tCO2 (68.9%)
2. Aluminum (76): 250.0 tCO2 (20.2%)
3. Cement (25): 100.0 tCO2 (8.1%)
...
```

### Provenance Record

Complete audit trail for regulatory compliance:

```json
{
  "report_id": "CBAM-2025Q4-NL-001",
  "generated_at": "2025-10-15T14:30:00Z",

  "input_file_integrity": {
    "file_name": "shipments.csv",
    "sha256_hash": "a1b2c3d4...",
    "file_size_bytes": 125000,
    "hash_timestamp": "2025-10-15T14:29:58Z"
  },

  "execution_environment": {
    "python_version": "3.11.5",
    "os": "Linux",
    "hostname": "cbam-server-01",
    "timestamp": "2025-10-15T14:30:00Z"
  },

  "dependencies": {
    "pandas": "2.1.0",
    "pydantic": "2.4.0",
    "jsonschema": "4.19.0"
  },

  "agent_execution": [
    {
      "agent_name": "ShipmentIntakeAgent",
      "start_time": "2025-10-15T14:30:00Z",
      "end_time": "2025-10-15T14:30:02Z",
      "duration_seconds": 2.15,
      "status": "success"
    },
    ...
  ],

  "reproducibility": {
    "deterministic": true,
    "zero_hallucination": true,
    "bit_perfect_reproducible": true
  }
}
```

---

## Advanced Usage

### Batch Processing

Process multiple files in one go:

```bash
#!/bin/bash
# process_all.sh

for file in data/shipments_*.csv; do
  echo "Processing $file..."
  gl cbam report "$file" \
    --config .cbam.yaml \
    --output "output/$(basename $file .csv)_report.json"
done

echo "All reports generated!"
```

### Custom Emission Factors

Override default emission factors with supplier actuals:

**Option 1: In CSV**
```csv
cn_code,country,quantity_tons,actual_emissions_tco2
72071100,CN,15.5,12.5
```

**Option 2: Supplier File**
```yaml
# examples/demo_suppliers.yaml
- supplier_id: "SUP-CN-001"
  name: "Jiangsu Steel Co."
  country: "CN"
  installations:
    - installation_id: "INST-001"
      cn_codes:
        - "72071100"
      actual_emission_factor_tco2_per_ton: 0.805
      verification_status: "verified"
      verifier: "TUV SUD"
```

### Large Dataset Processing

For >100K shipments, use chunked processing:

```python
from sdk.cbam_sdk import cbam_build_report
import pandas as pd

# Read in chunks
chunk_size = 10000
chunks = pd.read_csv("large_file.csv", chunksize=chunk_size)

# Process each chunk
all_reports = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}...")
    report = cbam_build_report(
        input_dataframe=chunk,
        config=config,
        save_output=False
    )
    all_reports.append(report.raw_report)

# Merge results
final_report = merge_reports(all_reports)
```

### CI/CD Integration

GitHub Actions example:

```yaml
# .github/workflows/cbam_report.yml
name: Generate Monthly CBAM Report

on:
  schedule:
    - cron: '0 9 1 * *'  # 1st of month at 9am
  workflow_dispatch:

jobs:
  generate-report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Generate CBAM Report
        env:
          CBAM_IMPORTER_NAME: ${{ secrets.CBAM_IMPORTER_NAME }}
          CBAM_IMPORTER_EORI: ${{ secrets.CBAM_IMPORTER_EORI }}
        run: |
          gl cbam report data/monthly_shipments.csv \
            --output reports/cbam_$(date +%Y%m).json

      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: cbam-report
          path: reports/
```

---

## Best Practices

### Data Quality

✅ **DO:**
- Validate data with `gl cbam validate` before filing
- Use supplier actuals whenever available
- Include invoice and customs references
- Keep original source files (SHA256 verified)

❌ **DON'T:**
- Skip validation step
- Rely solely on default emission factors
- Modify input files after hashing
- Delete original data files

### Configuration Management

✅ **DO:**
- Use `.cbam.yaml` for repeated reporting
- Store sensitive data in environment variables
- Version control config files (without secrets)
- Document configuration changes

❌ **DON'T:**
- Hardcode credentials in scripts
- Commit secrets to git
- Share config files with sensitive data
- Use same config for different entities

### Performance Optimization

For best performance:

1. **Use CSV format** for large datasets (faster than Excel)
2. **Process in chunks** for >50K shipments
3. **Pre-validate** data to catch errors early
4. **Use DataFrame** for in-memory processing
5. **Enable parallel workers** in config for large files

### Regulatory Compliance

Always include:

1. **SHA256 hash** of input file (automatic)
2. **Provenance record** (automatic)
3. **Audit trail** (automatic)
4. **Supplier verification** documentation
5. **Configuration file** used for report

---

## FAQ

### General Questions

**Q: Is this tool officially endorsed by the EU Commission?**
A: No. This is an independent automation tool. Always verify outputs meet official CBAM requirements.

**Q: Can I use this for final CBAM reporting (post-transitional)?**
A: Currently designed for Transitional Registry. Will be updated for final CBAM when regulations finalize.

**Q: What emission factors are used?**
A: Default factors from IEA, IPCC, WSA, IAI. Supports supplier actuals via supplier file.

### Data Questions

**Q: My file has 100K shipments. Will it work?**
A: Yes! Tested up to 1M shipments. Use chunked processing for >100K.

**Q: What if I don't have supplier actuals?**
A: Tool uses authoritative default emission factors. Recommended to get actuals for accuracy.

**Q: Can I process Excel files?**
A: Yes, `.xlsx` and `.xls` supported. CSV recommended for large files.

**Q: What date formats are supported?**
A: ISO 8601 (`YYYY-MM-DD`), most common formats auto-detected.

### Technical Questions

**Q: Is this zero hallucination?**
A: Yes! 100% deterministic calculations. No LLM in calculation path. See provenance proof.

**Q: Can I reproduce reports?**
A: Yes! Bit-perfect reproducibility with same input + environment. SHA256 verified.

**Q: How do I verify report integrity?**
A: Use provenance validation: `validate_provenance(report, input_file)`

**Q: Can I run this on Windows?**
A: Yes! Fully cross-platform (Linux, macOS, Windows).

### Integration Questions

**Q: Can I integrate with SAP?**
A: Yes! Use Python SDK to extract from SAP, process, and return results.

**Q: Can I schedule monthly reports?**
A: Yes! Use cron (Linux) or Task Scheduler (Windows). See CI/CD examples.

**Q: Can I use this for multiple legal entities?**
A: Yes! Multi-tenant support with separate config files per entity.

---

## Support

### Documentation

- **API Reference**: See `docs/API_REFERENCE.md`
- **Compliance Guide**: See `docs/COMPLIANCE_GUIDE.md`
- **Deployment Guide**: See `docs/DEPLOYMENT_GUIDE.md`
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`

### Examples

All examples available in `examples/` directory:
- `quick_start_cli.sh` - CLI tutorial
- `quick_start_sdk.py` - SDK examples
- `provenance_example.py` - Provenance usage
- `demo_shipments.csv` - Sample data
- `demo_suppliers.yaml` - Sample suppliers

### Community

- **Issues**: [GitHub Issues](https://github.com/greenlang/cbam-copilot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/greenlang/cbam-copilot/discussions)
- **Email**: cbam-support@greenlang.io

---

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**License:** MIT

---

*"The best AI doesn't hallucinate. It calculates."* - Zero Hallucination Architecture Principle
