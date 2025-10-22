# CSRD Platform - User Guide

**Complete Guide to the CSRD/ESRS Digital Reporting Platform**

Version 1.0.0 | Last Updated: 2025-10-18

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [CLI Command Reference](#cli-command-reference)
6. [SDK API Reference](#sdk-api-reference)
7. [Configuration Guide](#configuration-guide)
8. [Workflow Tutorials](#workflow-tutorials)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)
11. [Glossary](#glossary)

---

## Introduction

### About CSRD

The EU Corporate Sustainability Reporting Directive (CSRD) is a regulation requiring approximately 50,000 companies globally to report detailed sustainability information according to European Sustainability Reporting Standards (ESRS).

**Key Facts:**
- **Effective Date**: First reports due Q1 2025 for large public companies
- **Scope**: Large companies (>500 employees), listed SMEs, non-EU companies with significant EU operations
- **Standards**: 12 ESRS standards covering Environment, Social, and Governance topics
- **Format**: Digital reporting using ESEF (European Single Electronic Format) with XBRL tagging

### About This Platform

The CSRD/ESRS Digital Reporting Platform automates the entire CSRD reporting process with:

**Core Features:**
- **6-Agent Pipeline**: Automated workflow from raw data to submission-ready reports
- **Zero Hallucination**: 100% accurate calculations with complete audit trail
- **<30 Minute Processing**: Fast turnaround for 10,000+ data points
- **XBRL Digital Tagging**: ESEF-compliant submission packages
- **Multi-Standard Support**: Integrates TCFD, GRI, SASB data into ESRS format

**Who Should Use This Platform:**
- Sustainability/ESG managers preparing CSRD reports
- Finance teams responsible for regulatory compliance
- External auditors verifying CSRD submissions
- Consultants supporting companies with CSRD implementation
- Software developers integrating CSRD functionality

---

## Getting Started

### Prerequisites

**System Requirements:**
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.11 or higher
- **Memory**: Minimum 4 GB RAM (8 GB recommended for large datasets)
- **Storage**: 2 GB free disk space

**Required Knowledge:**
- Basic command-line usage
- Understanding of ESG/sustainability metrics
- Familiarity with CSRD/ESRS standards (recommended)

**Optional:**
- Python programming (for SDK usage)
- XBRL knowledge (for report validation)
- Docker (for containerized deployment)

### 5-Minute Quick Start

Follow these steps to generate your first CSRD report:

**Step 1: Clone Repository**
```bash
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Set API Key (for materiality assessment)**
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

**Step 4: Run Demo**
```bash
python examples/quick_start.py
```

**Expected Output:**
- Processing time: 2-5 minutes
- Output directory: `output/quick_start_demo/`
- Files: Complete CSRD report, summary, validation results, audit trail

---

## Installation

### Standard Installation

**1. Install Python 3.11+**

Check your Python version:
```bash
python --version
# Should show: Python 3.11.x or higher
```

If you need to install Python:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python@3.11`
- **Linux**: `sudo apt install python3.11 python3.11-venv`

**2. Create Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**3. Install Platform**
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "from sdk.csrd_sdk import csrd_build_report; print('✓ Installation successful')"
```

**4. Configure Environment**
```bash
# Copy example configuration
cp config/csrd_config.example.yaml config/csrd_config.yaml

# Edit with your settings
nano config/csrd_config.yaml  # or use your preferred editor
```

### Docker Installation

**1. Build Docker Image**
```bash
docker build -t csrd-platform .
```

**2. Run Container**
```bash
docker run -it \
  -v $(pwd)/output:/app/output \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  csrd-platform \
  python examples/quick_start.py
```

### Development Installation

For developers who want to modify the platform:

```bash
# Clone repository
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linter
ruff check .
```

---

## Data Preparation

### ESG Data Format

The platform accepts ESG data in multiple formats: CSV, JSON, Excel, or Parquet.

**Required Fields:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `metric_code` | string | ESRS data point code | "E1-1", "S1-9" |
| `metric_name` | string | Human-readable metric name | "Scope 1 GHG emissions" |
| `value` | number/string/boolean | Metric value | 12500 |
| `unit` | string | Unit of measurement | "tCO2e", "GJ", "FTE" |
| `period_start` | string | Period start date (YYYY-MM-DD) | "2024-01-01" |
| `period_end` | string | Period end date (YYYY-MM-DD) | "2024-12-31" |

**Optional Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `data_quality` | string | Quality indicator: high, medium, low |
| `source_document` | string | Reference to source file/system |
| `verification_status` | string | verified or unverified |
| `notes` | string | Additional context or comments |
| `subsidiary_id` | string | For multi-entity reporting |
| `location` | string | Geographic location |

**Example CSV:**
```csv
metric_code,metric_name,value,unit,period_start,period_end,data_quality
E1-1,Scope 1 GHG Emissions,12500,tCO2e,2024-01-01,2024-12-31,high
E1-2,Scope 2 GHG Emissions (location-based),8300,tCO2e,2024-01-01,2024-12-31,high
E1-6,Total Energy Consumption,185000,GJ,2024-01-01,2024-12-31,medium
S1-1,Total Employees,1250,FTE,2024-01-01,2024-12-31,high
S1-6,Work-related accidents,3,count,2024-01-01,2024-12-31,high
G1-1,Board gender diversity,45,percent,2024-01-01,2024-12-31,high
```

**Example JSON:**
```json
[
  {
    "metric_code": "E1-1",
    "metric_name": "Scope 1 GHG Emissions",
    "value": 12500,
    "unit": "tCO2e",
    "period_start": "2024-01-01",
    "period_end": "2024-12-31",
    "data_quality": "high",
    "source_document": "GHG Inventory Report 2024.xlsx",
    "verification_status": "verified"
  }
]
```

### Company Profile Format

Create a JSON file with your company information:

**Example: `company_profile.json`**
```json
{
  "company_id": "550e8400-e29b-41d4-a716-446655440000",
  "legal_name": "Acme Manufacturing EU B.V.",
  "lei_code": "549300ABC123DEF456GH",
  "country": "NL",
  "registered_address": {
    "street": "Industrieweg 42",
    "city": "Amsterdam",
    "postal_code": "1043 AB",
    "country": "NL"
  },
  "sector": {
    "nace_code": "25.11",
    "nace_description": "Manufacture of metal structures",
    "industry": "Manufacturing"
  },
  "company_size": {
    "employee_count": 1250,
    "employee_count_fte": 1250,
    "size_category": "Large",
    "revenue_eur": 450000000,
    "total_assets_eur": 280000000
  },
  "reporting_period": {
    "fiscal_year": 2024,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "reporting_framework": "CSRD"
  },
  "contact": {
    "sustainability_officer": {
      "name": "Jane Smith",
      "title": "Chief Sustainability Officer",
      "email": "jane.smith@acme-eu.com"
    }
  }
}
```

**Required Fields:**
- `legal_name`: Official company name
- `lei_code`: Legal Entity Identifier (20 characters)
- `country`: ISO 3166-1 alpha-2 country code
- `sector.nace_code`: EU NACE industry classification
- `reporting_period.fiscal_year`: Reporting year

### ESRS Data Points Catalog

The platform includes a catalog of 1,082 ESRS data points covering all 12 standards.

**Data Point Structure:**
```json
{
  "E1-1": {
    "metric_code": "E1-1",
    "metric_name": "Scope 1 GHG emissions",
    "description": "Total gross Scope 1 GHG emissions in tonnes of CO2 equivalent",
    "unit": "tCO2e",
    "esrs_standard": "E1",
    "mandatory": true,
    "data_type": "quantitative",
    "calculation_formula": "SUM(direct_emissions_sources)",
    "validation_rules": [
      "value >= 0",
      "unit == 'tCO2e'"
    ]
  }
}
```

**Access the catalog:**
```python
import json
with open('data/esrs_data_points.json', 'r') as f:
    catalog = json.load(f)

# Find a specific data point
e1_1 = catalog['E1-1']
print(e1_1['metric_name'])
print(e1_1['description'])
```

### Data Quality Requirements

The platform validates data quality against these criteria:

**1. Completeness (40% weight)**
- All mandatory ESRS data points present
- No missing values for required fields
- Period coverage is complete (no gaps)

**2. Accuracy (30% weight)**
- Values are within reasonable ranges
- Units are consistent and correct
- Calculations can be verified

**3. Consistency (20% weight)**
- Data is internally consistent (e.g., totals match sums)
- Time series data shows logical progression
- Cross-references are valid

**4. Timeliness (10% weight)**
- Data is from the correct reporting period
- Recent data is preferred over historical data
- Updates are properly timestamped

**Minimum Quality Score:** 80/100 (configurable)

**Improve Data Quality:**
1. Use standardized templates
2. Implement data validation at source
3. Regular data quality checks
4. Document data collection processes
5. Train staff on ESRS requirements

---

## CLI Command Reference

### csrd run

Execute the complete 6-agent CSRD reporting pipeline.

**Syntax:**
```bash
csrd run --input <esg_data> --company-profile <company.json> [OPTIONS]
```

**Required Arguments:**
- `--input, -i`: Path to ESG data file (CSV/JSON/Excel/Parquet)
- `--company-profile, -p`: Path to company profile JSON

**Optional Arguments:**
- `--output-dir, -o`: Output directory (default: ./output)
- `--config, -c`: Configuration file path
- `--skip-materiality`: Skip materiality assessment
- `--skip-audit`: Skip compliance audit
- `--verbose, -v`: Verbose output
- `--quiet, -q`: Minimal output (errors only)
- `--format`: Output format: xbrl, json, both (default: both)

**Examples:**
```bash
# Basic usage
csrd run --input data.csv --company-profile company.json

# With custom output directory
csrd run -i data.csv -p company.json -o reports/2024

# Skip materiality (faster, for testing)
csrd run -i data.csv -p company.json --skip-materiality

# Verbose mode
csrd run -i data.csv -p company.json -v

# Custom configuration
csrd run -i data.csv -p company.json -c custom_config.yaml
```

**Output Files:**
- `00_complete_report.json` - Complete report package
- `00_summary.md` - Human-readable summary
- `01_validated_data.json` - Validated ESG data
- `02_materiality_assessment.json` - Materiality results
- `03_calculated_metrics.json` - Calculated ESRS metrics
- `04_aggregated_data.json` - Multi-framework data
- `05_xbrl_report.xhtml` - XBRL-tagged report
- `06_audit_compliance.json` - Compliance validation

**Exit Codes:**
- `0` - Success
- `1` - Error (pipeline failed)
- `2` - Warning (compliance issues)

---

### csrd validate

Validate ESG data without running the full pipeline (IntakeAgent only).

**Syntax:**
```bash
csrd validate --input <esg_data> [OPTIONS]
```

**Required Arguments:**
- `--input, -i`: Path to ESG data file

**Optional Arguments:**
- `--output, -o`: Output validation report path
- `--verbose, -v`: Show detailed validation errors
- `--quiet, -q`: Minimal output

**Examples:**
```bash
# Validate data
csrd validate --input data.csv

# Save validation report
csrd validate -i data.csv -o validation_report.json

# Verbose mode (see all errors)
csrd validate -i data.csv -v
```

**Output:**
```json
{
  "metadata": {
    "total_records": 150,
    "valid_records": 145,
    "invalid_records": 5,
    "data_quality_score": 87.5,
    "quality_threshold_met": true
  },
  "validation_issues": [
    {
      "severity": "error",
      "error_code": "INVALID_UNIT",
      "message": "Invalid unit 'kg' for metric E1-1, expected 'tCO2e'",
      "field": "unit",
      "row": 12
    }
  ]
}
```

---

### csrd calculate

Calculate ESRS metrics only (CalculatorAgent).

**Syntax:**
```bash
csrd calculate --input <validated_data.json> [OPTIONS]
```

**Required Arguments:**
- `--input, -i`: Path to validated data (from IntakeAgent)

**Optional Arguments:**
- `--output, -o`: Output calculated metrics path
- `--verbose, -v`: Show calculation details
- `--quiet, -q`: Minimal output

**Examples:**
```bash
# Calculate metrics
csrd calculate --input validated_data.json

# Save calculations
csrd calculate -i validated_data.json -o metrics.json -v
```

**Key Features:**
- **Zero Hallucination**: No LLM involvement
- **Deterministic**: Same inputs always produce same outputs
- **Fast**: <5ms per metric
- **Traceable**: Complete provenance for every calculation

---

### csrd audit

Run compliance audit only (AuditAgent).

**Syntax:**
```bash
csrd audit --report <report.json> [OPTIONS]
```

**Required Arguments:**
- `--report, -r`: Path to CSRD report (JSON or XBRL)

**Optional Arguments:**
- `--output, -o`: Output audit report path
- `--verbose, -v`: Show detailed compliance issues
- `--quiet, -q`: Minimal output

**Examples:**
```bash
# Audit a report
csrd audit --report complete_report.json

# Verbose mode
csrd audit -r report.json -v -o audit_results.json
```

**Validation Checks:**
- 200+ ESRS compliance rules
- ESEF technical standards
- Data quality thresholds
- Cross-validation checks
- Taxonomy alignment

**Output:**
```json
{
  "compliance_status": "PASS",
  "total_rules_checked": 215,
  "rules_passed": 212,
  "rules_failed": 3,
  "rules_warning": 5,
  "critical_failures": 0,
  "audit_ready": true
}
```

---

### csrd materialize

Run double materiality assessment only (MaterialityAgent).

**Syntax:**
```bash
csrd materialize --input <data.json> --company-profile <company.json> [OPTIONS]
```

**Required Arguments:**
- `--input, -i`: Validated ESG data
- `--company-profile, -p`: Company profile JSON

**Optional Arguments:**
- `--output, -o`: Output materiality assessment
- `--verbose, -v`: Show detailed assessment
- `--quiet, -q`: Minimal output

**Examples:**
```bash
# Run materiality assessment
csrd materialize -i data.json -p company.json

# Save results
csrd materialize -i data.json -p company.json -o materiality.json
```

**Warning:** This uses AI/LLM and requires human review!

---

### csrd config

Manage CSRD configuration.

**Syntax:**
```bash
csrd config [--init | --show] [OPTIONS]
```

**Options:**
- `--init`: Create new configuration interactively
- `--show`: Display current configuration
- `--path`: Configuration file path (default: .csrd.yaml)

**Examples:**
```bash
# Create new config
csrd config --init

# Show current config
csrd config --show

# Custom config path
csrd config --init --path custom.yaml
```

---

## SDK API Reference

### csrd_build_report()

Main SDK function to generate a complete CSRD report in one call.

**Function Signature:**
```python
def csrd_build_report(
    esg_data: Union[str, Path, pd.DataFrame],
    company_profile: Union[str, Path, Dict],
    config: Optional[CSRDConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    skip_materiality: bool = False,
    skip_audit: bool = False,
    verbose: bool = False,
    **kwargs
) -> CSRDReport
```

**Parameters:**
- `esg_data`: ESG data as file path or DataFrame
- `company_profile`: Company profile as file path or dictionary
- `config`: CSRDConfig object (optional)
- `output_dir`: Directory to save all outputs
- `skip_materiality`: Skip materiality assessment
- `skip_audit`: Skip compliance audit
- `verbose`: Enable verbose logging
- `**kwargs`: Additional configuration overrides

**Returns:**
- `CSRDReport`: Complete report object with results

**Example:**
```python
from sdk.csrd_sdk import csrd_build_report, CSRDConfig

# Create configuration
config = CSRDConfig(
    company_name="Acme Corp",
    company_lei="549300ABC123DEF456GH",
    reporting_year=2024,
    sector="Manufacturing"
)

# Generate report
report = csrd_build_report(
    esg_data="data/esg_data.csv",
    company_profile="data/company.json",
    config=config,
    output_dir="output/csrd_2024"
)

# Access results
print(f"Compliance: {report.compliance_status.compliance_status}")
print(f"Material topics: {report.materiality.material_topics_count}")
print(f"GHG emissions: {report.metrics.total_ghg_emissions_tco2e:.2f} tCO2e")

# Save outputs
report.save_json("report.json")
report.save_summary("summary.md")
```

---

### CSRDConfig Class

Configuration object for CSRD pipeline.

**Constructor:**
```python
CSRDConfig(
    company_name: str,
    company_lei: str,
    reporting_year: int,
    sector: str,
    country: str = "DE",
    employee_count: Optional[int] = None,
    revenue: Optional[float] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    llm_api_key: Optional[str] = None,
    quality_threshold: float = 0.80,
    impact_materiality_threshold: float = 5.0,
    financial_materiality_threshold: float = 5.0,
    **kwargs
)
```

**Class Methods:**
```python
# Load from YAML
config = CSRDConfig.from_yaml("csrd_config.yaml")

# Load from environment variables
config = CSRDConfig.from_env()

# Load from dictionary
config = CSRDConfig.from_dict(config_dict)

# Convert to dictionary
config_dict = config.to_dict()
```

---

### CSRDReport Class

Report object returned by `csrd_build_report()`.

**Properties:**
```python
report.report_id                    # Unique report ID
report.company_info                 # Company information dict
report.reporting_period             # Reporting period dict
report.materiality                  # MaterialityAssessment object
report.metrics                      # ESRSMetrics object
report.compliance_status            # ComplianceStatus object
report.is_compliant                 # bool: Pass/fail status
report.is_audit_ready               # bool: Audit readiness
report.material_standards           # List of material ESRS standards
report.warnings                     # List of warning messages
report.processing_time_total_minutes  # Total processing time
```

**Methods:**
```python
# Get summary text
summary = report.summary()

# Convert to dictionary
report_dict = report.to_dict()

# Convert to JSON string
json_str = report.to_json(indent=2)

# Save to file
report.save_json("report.json")
report.save_summary("summary.md")

# Convert to DataFrame
df = report.to_dataframe()
```

---

## Configuration Guide

### Configuration File Structure

The platform uses YAML configuration files. Here's a complete example:

**File: `csrd_config.yaml`**
```yaml
# Company Information
company:
  name: "Acme Manufacturing EU B.V."
  lei: "549300ABC123DEF456GH"
  country: "NL"
  sector: "Manufacturing"
  reporting_year: 2024
  employee_count: 1250
  revenue: 450000000

# Data Paths
paths:
  esrs_data_points: "data/esrs_data_points.json"
  emission_factors: "data/emission_factors.json"
  esrs_formulas: "data/esrs_formulas.yaml"
  compliance_rules: "rules/esrs_compliance_rules.yaml"
  data_quality_rules: "rules/data_quality_rules.yaml"

# Agent Configuration
agents:
  intake:
    enabled: true
    data_quality_threshold: 0.80

  materiality:
    enabled: true
    requires_human_review: true
    impact_threshold: 5.0
    financial_threshold: 5.0

  calculator:
    enabled: true
    zero_hallucination: true
    deterministic: true

  audit:
    enabled: true
    execute_all_rules: true

# LLM Configuration
llm:
  openai:
    api_key_env_var: "OPENAI_API_KEY"
    default_model: "gpt-4o"
    max_retries: 3
    timeout_seconds: 60

  anthropic:
    api_key_env_var: "ANTHROPIC_API_KEY"
    default_model: "claude-3-5-sonnet-20241022"

# Performance Targets
performance:
  total_pipeline_max_minutes: 30
  materiality_agent_max_minutes: 10
  calculator_agent_ms_per_metric: 5

# Quality Thresholds
quality:
  data_completeness_min: 0.80
  calculation_accuracy: 1.00
  xbrl_validation_required: true
```

---

## Workflow Tutorials

### Tutorial 1: First-Time Setup

**Goal:** Set up the platform and generate your first report.

**Time:** 15 minutes

**Steps:**

1. **Install Platform**
```bash
pip install -r requirements.txt
```

2. **Prepare Your Data**
   - Export ESG data to CSV
   - Create company profile JSON
   - (Use templates in `examples/` as reference)

3. **Set API Key**
```bash
export OPENAI_API_KEY='your-key-here'
```

4. **Generate Report**
```bash
csrd run \
  --input your_data.csv \
  --company-profile your_company.json \
  --output-dir output/first_report \
  --verbose
```

5. **Review Results**
   - Check `output/first_report/00_summary.md`
   - Review compliance status
   - Address any validation issues

---

### Tutorial 2: Iterative Development

**Goal:** Iteratively improve data quality before final report.

**Steps:**

1. **Validate Data First**
```bash
csrd validate --input data.csv --verbose > validation.log
```

2. **Review Validation Issues**
   - Open `validation.log`
   - Fix data quality issues
   - Re-validate until score > 80

3. **Run Calculations Only**
```bash
csrd calculate --input validated_data.json
```

4. **Generate Final Report**
```bash
csrd run --input data.csv --company-profile company.json
```

---

### Tutorial 3: Batch Processing

**Goal:** Process multiple subsidiaries or reporting periods.

**Python Script:**
```python
from sdk.csrd_sdk import csrd_build_report, CSRDConfig
from pathlib import Path

# Define subsidiaries
subsidiaries = [
    {"name": "Acme NL", "file": "data/acme_nl.csv"},
    {"name": "Acme DE", "file": "data/acme_de.csv"},
    {"name": "Acme FR", "file": "data/acme_fr.csv"}
]

# Process each subsidiary
for sub in subsidiaries:
    print(f"Processing {sub['name']}...")

    config = CSRDConfig(
        company_name=sub['name'],
        company_lei=f"549300{sub['name'][-2:]}",
        reporting_year=2024,
        sector="Manufacturing"
    )

    report = csrd_build_report(
        esg_data=sub['file'],
        company_profile="company.json",
        config=config,
        output_dir=f"output/{sub['name'].replace(' ', '_')}"
    )

    print(f"✓ {sub['name']} complete: {report.compliance_status.compliance_status}")
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Data Quality Below Threshold

**Symptom:**
```
WARNING: Data quality score (72.3) below threshold (80.0)
```

**Solutions:**
1. Review validation issues in detail:
   ```bash
   csrd validate --input data.csv --verbose
   ```

2. Check for:
   - Missing mandatory data points
   - Invalid units
   - Out-of-range values
   - Incorrect metric codes

3. Fix data and re-validate

---

#### Issue 2: API Key Not Found

**Symptom:**
```
WARNING: OPENAI_API_KEY not set in environment
Materiality assessment will be skipped
```

**Solutions:**
1. Set environment variable:
   ```bash
   export OPENAI_API_KEY='sk-...'
   ```

2. Or add to config:
   ```python
   config.llm_api_key = "sk-..."
   ```

3. For persistent setup (Linux/macOS):
   ```bash
   echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
   source ~/.bashrc
   ```

---

#### Issue 3: Out of Memory

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. Process data in smaller batches
2. Increase system memory
3. Use chunked processing:
   ```python
   # Process in chunks of 1000 records
   chunk_size = 1000
   for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
       # Process chunk
   ```

---

#### Issue 4: Compliance Validation Failed

**Symptom:**
```
Compliance Status: FAIL
Critical Issues: 3
```

**Solutions:**
1. Review audit report:
   ```bash
   csrd audit --report report.json --verbose
   ```

2. Check failed rules:
   ```python
   for rule in report.compliance_status.failed_rules:
       print(f"{rule['rule_id']}: {rule['message']}")
   ```

3. Address critical issues first
4. Re-run pipeline

---

## FAQ

### General Questions

**Q: How long does it take to generate a CSRD report?**
A: Processing time depends on data size:
- 1,000 data points: ~5 minutes
- 10,000 data points: ~15 minutes
- 50,000 data points: ~45 minutes

**Q: Does the platform work offline?**
A: Partially. Data validation, calculations, and audit work offline. Materiality assessment requires internet connection for LLM API.

**Q: Can I use my own emission factors?**
A: Yes, customize `data/emission_factors.json` with your factors.

**Q: Is the platform GDPR compliant?**
A: Yes, but you are responsible for:
- Not including personal data in ESG data
- Proper data access controls
- Audit trail retention policies

---

### Technical Questions

**Q: What file formats are supported?**
A: CSV, JSON, Excel (.xlsx), and Parquet for ESG data. JSON or YAML for company profile.

**Q: Can I customize the ESRS data points catalog?**
A: Yes, edit `data/esrs_data_points.json` to add custom data points.

**Q: How is "zero hallucination" achieved?**
A: All calculations use:
1. Database lookups (emission factors, etc.)
2. Python arithmetic only (no LLM)
3. Deterministic formulas
4. Complete provenance tracking

**Q: Can I run the platform on ARM Macs?**
A: Yes, all dependencies support ARM architecture.

**Q: How do I integrate with SAP/Oracle ERP?**
A: Export ESG data from your ERP to CSV, then use the platform. See `docs/ERP_INTEGRATION.md` for details.

---

### Regulatory Questions

**Q: Is the output submission-ready?**
A: The XBRL output is ESEF-compliant and ready for submission after:
1. Human review of materiality assessment
2. External assurance (if required)
3. Board approval

**Q: What ESRS standards are covered?**
A: All 12 Set 1 standards:
- Environment: E1-E5
- Social: S1-S4
- Governance: G1
- Cross-cutting: ESRS-1, ESRS-2

**Q: Does this replace external auditors?**
A: No. The platform generates audit-ready reports but external assurance is still required by regulation.

---

## Glossary

**CSRD**: Corporate Sustainability Reporting Directive (EU Directive 2022/2464)

**ESRS**: European Sustainability Reporting Standards (Set 1: 12 standards)

**ESEF**: European Single Electronic Format (XBRL-based format for digital reporting)

**XBRL**: eXtensible Business Reporting Language (XML-based standard for business reporting)

**iXBRL**: Inline XBRL (human-readable HTML with embedded XBRL tags)

**Double Materiality**: Assessment methodology considering both:
- **Impact Materiality**: Company's impact on environment/society
- **Financial Materiality**: Sustainability matters' effect on company value

**LEI**: Legal Entity Identifier (20-character code for companies)

**NACE**: Nomenclature of Economic Activities (EU industry classification)

**Scope 1/2/3 Emissions**: GHG emissions classification:
- Scope 1: Direct emissions from owned sources
- Scope 2: Indirect emissions from purchased energy
- Scope 3: Other indirect emissions in value chain

**tCO2e**: Tonnes of CO2 equivalent (standard unit for GHG emissions)

**Zero Hallucination**: Guarantee that all calculations are 100% deterministic with no LLM involvement

**Provenance**: Complete record of data lineage and calculation methodology

**Materiality Threshold**: Minimum score (0-10) for a topic to be considered material

**Data Quality Score**: Composite score (0-100) based on completeness, accuracy, consistency, and timeliness

---

## Additional Resources

**Official Documentation:**
- [EFRAG ESRS Standards](https://www.efrag.org/en/activities/esrs)
- [EU CSRD Directive](https://eur-lex.europa.eu/eli/dir/2022/2464)
- [ESEF Reporting Manual](https://www.esma.europa.eu/esef)

**Platform Documentation:**
- [API Reference](API_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Architecture Guide](ARCHITECTURE.md)

**Support:**
- Email: csrd@greenlang.io
- GitHub Issues: [Report bugs](https://github.com/akshay-greenlang/Code-V1_GreenLang/issues)
- Documentation: https://greenlang.io/docs

---

**Last Updated:** 2025-10-18
**Version:** 1.0.0
**License:** MIT
