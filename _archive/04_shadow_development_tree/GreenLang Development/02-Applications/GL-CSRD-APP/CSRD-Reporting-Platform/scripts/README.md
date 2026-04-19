# CSRD Platform Utility Scripts

This directory contains utility scripts for running, testing, and validating the CSRD/ESRS Digital Reporting Platform.

## Overview

Four essential utility scripts are provided:

1. **`benchmark.py`** - Performance benchmarking tool
2. **`validate_schemas.py`** - Schema validation tool
3. **`generate_sample_data.py`** - Test data generator
4. **`run_full_pipeline.py`** - Complete pipeline runner

All scripts use the `Click` CLI framework and provide rich terminal output via the `rich` library.

---

## 1. benchmark.py - Performance Benchmarking Tool

Measures and reports performance metrics for the CSRD pipeline.

### Features

- Benchmark all 6 agents individually
- Benchmark complete end-to-end pipeline
- Test multiple dataset sizes (tiny, small, medium, large, xlarge)
- Memory profiling and throughput measurement
- Generate reports in JSON and Markdown formats
- Compare against performance targets

### Usage

```bash
# Basic usage - benchmark with small dataset
python scripts/benchmark.py --dataset-size small

# Benchmark specific agents
python scripts/benchmark.py --dataset-size medium --agents intake

# Benchmark complete pipeline with large dataset
python scripts/benchmark.py --dataset-size large --agents pipeline

# Custom output paths
python scripts/benchmark.py --dataset-size small \
  --output results/benchmark.json \
  --markdown results/benchmark.md

# Use existing data (skip generation)
python scripts/benchmark.py --dataset-size medium --skip-data-generation
```

### Dataset Sizes

| Size    | Records | Target Time  | Use Case                    |
|---------|---------|--------------|----------------------------|
| tiny    | 10      | 5s           | Quick smoke tests          |
| small   | 100     | 30s          | Development testing        |
| medium  | 1,000   | 3 min        | Integration testing        |
| large   | 10,000  | 30 min       | Production target          |
| xlarge  | 50,000  | 90 min       | Stress testing             |

### Options

```
--dataset-size      Dataset size to benchmark [tiny|small|medium|large|xlarge]
--agents            Which agents to benchmark [all|intake|calculator|pipeline]
--output            Output file for JSON results (default: benchmark_report.json)
--markdown          Output file for Markdown report (default: benchmark_report.md)
--skip-data-generation  Skip data generation (use existing benchmark data)
```

### Output

**JSON Report Structure:**
```json
{
  "benchmark_id": "benchmark_1234567890",
  "timestamp": "2024-10-18T16:30:00",
  "dataset_size": "small",
  "dataset_records": 100,
  "target_time_seconds": 30,
  "total_duration_seconds": 25.5,
  "meets_target": true,
  "agent_benchmarks": [
    {
      "agent_name": "IntakeAgent",
      "duration_seconds": 2.5,
      "input_records": 100,
      "output_records": 98,
      "throughput_records_per_sec": 40.0,
      "memory_peak_mb": 125.3,
      "status": "success"
    }
  ],
  "summary": {
    "agents_tested": 6,
    "agents_successful": 6,
    "total_memory_peak_mb": 450.2,
    "average_throughput": 250.5,
    "slowest_agent": "ReportingAgent",
    "fastest_agent": "CalculatorAgent"
  }
}
```

**Markdown Report:**
- Summary statistics
- Agent performance table
- Complete pipeline metrics
- Performance analysis
- Target comparison

### Performance Targets

| Agent              | Target                        |
|-------------------|-------------------------------|
| IntakeAgent       | 1,000 records/sec             |
| CalculatorAgent   | 500 calculations/sec          |
| ReportingAgent    | 50 XBRL tags/sec              |
| AuditAgent        | 100 rules/sec                 |
| Complete Pipeline | < 30 min for 10,000 records   |

---

## 2. validate_schemas.py - Schema Validation Tool

Validates JSON schemas and example data files.

### Features

- Validate all JSON schemas for correctness
- Test example data against schemas
- Check schema completeness and coverage
- Report validation errors with detailed diagnostics
- Verify ESRS data point coverage

### Usage

```bash
# Validate all schemas and examples
python scripts/validate_schemas.py

# Validate specific schema
python scripts/validate_schemas.py --schema schemas/esg_data.schema.json

# Validate specific example
python scripts/validate_schemas.py --example examples/demo_esg_data.csv

# Custom directories
python scripts/validate_schemas.py \
  --schemas-dir path/to/schemas \
  --examples-dir path/to/examples

# Check ESRS coverage
python scripts/validate_schemas.py --check-esrs-coverage

# Save validation report
python scripts/validate_schemas.py --output validation_results.json
```

### Options

```
--schema            Validate specific schema file
--example           Validate specific example data file
--schemas-dir       Directory containing schemas (default: schemas/)
--examples-dir      Directory containing example data (default: examples/)
--output            Output file for validation report (default: validation_report.json)
--check-esrs-coverage  Check ESRS data point coverage
```

### Validated Schemas

1. **company_profile.schema.json** - Company profile structure
2. **esg_data.schema.json** - ESG data point structure
3. **materiality.schema.json** - Materiality assessment structure
4. **csrd_report.schema.json** - CSRD report structure

### Output

**Validation Report:**
```json
{
  "timestamp": "2024-10-18T16:30:00",
  "overall_valid": true,
  "schemas_validated": 4,
  "examples_validated": 3,
  "total_errors": 0,
  "total_warnings": 2,
  "schema_results": [...],
  "example_results": [...]
}
```

**Console Output:**
- Validation status table
- Error details with file paths
- Warning summaries
- ESRS coverage statistics

### Exit Codes

- `0` - All validations passed
- `1` - Validation errors found

---

## 3. generate_sample_data.py - Test Data Generator

Generates realistic ESG test data for development and testing.

### Features

- Generate CSV, JSON, and Excel formats
- Support all ESRS standards (E1-E5, S1-S4, G1)
- Create company profiles
- Generate materiality assessments
- Configurable data sizes (10 to 50,000+ metrics)
- Realistic value ranges and units
- Proper ESRS metric codes

### Usage

```bash
# Generate 100 data points in CSV format
python scripts/generate_sample_data.py --size 100 --format csv

# Generate data in all formats
python scripts/generate_sample_data.py --size 1000 --format all --output data/test_data

# Generate specific ESRS standards
python scripts/generate_sample_data.py \
  --size 500 \
  --esrs-standards E1 E3 S1 \
  --format json

# Generate complete test dataset
python scripts/generate_sample_data.py \
  --size 1000 \
  --format all \
  --generate-company-profile \
  --generate-materiality \
  --company-name "Test Corporation"

# Different reporting year
python scripts/generate_sample_data.py --size 100 --year 2023
```

### Options

```
--size              Number of data points to generate (default: 100)
--format            Output format [csv|json|excel|all] (default: csv)
--output            Output file path without extension (default: sample_esg_data)
--esrs-standards    ESRS standards to include (can specify multiple)
                    Options: E1, E2, E3, E4, E5, S1, S2, S3, S4, G1
--year              Reporting year (default: 2024)
--generate-company-profile    Also generate a company profile
--generate-materiality        Also generate a materiality assessment
--company-name      Company name for profile generation (default: Sample Corporation)
```

### ESRS Standards Coverage

**Environmental (E):**
- **E1** - Climate Change (9 metrics)
- **E2** - Pollution (5 metrics)
- **E3** - Water and Marine Resources (5 metrics)
- **E4** - Biodiversity and Ecosystems (3 metrics)
- **E5** - Circular Economy (6 metrics)

**Social (S):**
- **S1** - Own Workforce (14 metrics)
- **S2** - Workers in Value Chain (4 metrics)
- **S3** - Affected Communities (4 metrics)
- **S4** - Consumers and End-Users (4 metrics)

**Governance (G):**
- **G1** - Business Conduct (8 metrics)

**Total:** 62 unique metric templates

### Generated Data Quality

- **Data Quality Distribution:** 60% high, 30% medium, 10% low
- **Verification Status:** 50% verified, 30% third-party assured, 20% unverified
- **Value Ranges:** Realistic ranges based on industry benchmarks
- **Units:** Proper ESRS-compliant units (tCO2e, GJ, m3, FTE, EUR, %, etc.)

### Output Examples

**CSV Format:**
```csv
metric_code,metric_name,value,unit,period_start,period_end,data_quality,source_document,verification_status,notes
E1-1,Scope 1 GHG Emissions,12500.5,tCO2e,2024-01-01,2024-12-31,high,SAP ERP System,verified,Generated data point 1
...
```

**JSON Format:**
```json
{
  "data_points": [
    {
      "metric_code": "E1-1",
      "metric_name": "Scope 1 GHG Emissions",
      "value": 12500.5,
      "unit": "tCO2e",
      "period_start": "2024-01-01",
      "period_end": "2024-12-31",
      "data_quality": "high",
      "source_document": "SAP ERP System",
      "verification_status": "verified",
      "notes": "Generated data point 1"
    }
  ]
}
```

**Company Profile:**
- Realistic company information
- LEI code, NACE codes
- Subsidiaries and operations
- Sustainability governance
- Previous reporting history

**Materiality Assessment:**
- Material topics by ESRS standard
- Impact and financial materiality scores
- Stakeholder consultation details

---

## 4. run_full_pipeline.py - Complete Pipeline Runner

Runs the complete CSRD reporting pipeline with monitoring and reporting.

### Features

- Run complete 6-agent CSRD pipeline
- Single company or batch processing
- Configuration file support
- Progress monitoring with rich output
- Error handling and recovery
- Summary report generation (Markdown)
- Performance metrics tracking

### Usage

**Single Company Mode:**
```bash
# Basic usage
python scripts/run_full_pipeline.py \
  --esg-data examples/demo_esg_data.csv \
  --company-profile examples/demo_company_profile.json

# Custom configuration and output
python scripts/run_full_pipeline.py \
  --esg-data data/company_esg_2024.xlsx \
  --company-profile data/company_profile.json \
  --config config/custom_config.yaml \
  --output output/company_report_2024

# Without summary report
python scripts/run_full_pipeline.py \
  --esg-data data.csv \
  --company-profile profile.json \
  --generate-summary=false
```

**Batch Processing Mode:**
```bash
# Process multiple companies
python scripts/run_full_pipeline.py \
  --batch batch_config.json \
  --output output/batch_reports

# Parallel processing (not yet implemented)
python scripts/run_full_pipeline.py \
  --batch batch_config.json \
  --parallel \
  --output output/batch_reports
```

### Options

```
--esg-data          ESG data file (CSV/JSON/Excel/Parquet)
--company-profile   Company profile JSON file
--config            CSRD configuration file (default: config/csrd_config.yaml)
--output            Output directory for reports (default: output/csrd_report)
--batch             Batch configuration file (JSON/YAML) with multiple companies
--parallel          Run batch processing in parallel (not yet implemented)
--generate-summary  Generate summary report in Markdown (default: true)
--resume            Resume a failed pipeline run (pipeline ID) - not yet implemented
```

### Batch Configuration Format

**JSON Example:**
```json
{
  "companies": [
    {
      "company_id": "company_001",
      "legal_name": "Acme Corporation",
      "esg_data_file": "data/acme_esg_2024.csv",
      "company_profile": {
        "legal_name": "Acme Corporation",
        "country": "NL",
        ...
      }
    },
    {
      "company_id": "company_002",
      "legal_name": "Beta Industries",
      "esg_data_file": "data/beta_esg_2024.xlsx",
      "company_profile": {
        ...
      }
    }
  ]
}
```

### Pipeline Stages

The script runs all 6 agents in sequence:

1. **IntakeAgent** - Data ingestion and validation
2. **MaterialityAgent** - Double materiality assessment (AI-powered)
3. **CalculatorAgent** - ESRS metrics calculation (deterministic)
4. **AggregatorAgent** - Cross-framework aggregation
5. **ReportingAgent** - CSRD report generation with XBRL tagging
6. **AuditAgent** - Compliance validation

### Output Files

**Single Company:**
```
output/csrd_report/
├── intermediate/
│   ├── 01_intake_validated.json
│   ├── 02_materiality_assessment.json
│   ├── 03_calculated_metrics.json
│   ├── 04_aggregated_data.json
│   ├── 05_csrd_report.json
│   └── 06_compliance_audit.json
├── pipeline_result.json
├── pipeline_summary.md
└── csrd_esef_package.zip
```

**Batch Processing:**
```
output/batch_reports/
├── company_001/
│   ├── intermediate/...
│   └── pipeline_result.json
├── company_002/
│   ├── intermediate/...
│   └── pipeline_result.json
└── batch_report_batch_1234567890.json
```

### Summary Report

The generated `pipeline_summary.md` includes:

- Pipeline execution metadata
- Performance summary (time, throughput)
- Agent performance breakdown
- Data quality and compliance status
- Agent execution details
- Configuration used

### Exit Codes

- `0` - Pipeline completed successfully (compliant)
- `1` - Pipeline failed or non-compliant

### Console Output

Rich terminal output includes:

- Real-time progress bars
- Agent execution status
- Performance metrics
- Validation results
- Error messages with context
- Summary tables

---

## Installation & Dependencies

All scripts require the dependencies listed in `requirements.txt`:

```bash
# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
- `click` - CLI framework
- `rich` - Terminal formatting
- `pandas` - Data manipulation
- `pydantic` - Data validation
- `jsonschema` - Schema validation
- `openpyxl` - Excel support
- `pyyaml` - YAML support

---

## Common Workflows

### 1. Development Testing

```bash
# Generate test data
python scripts/generate_sample_data.py --size 100 --format all

# Validate schemas
python scripts/validate_schemas.py

# Run pipeline
python scripts/run_full_pipeline.py \
  --esg-data sample_esg_data.csv \
  --company-profile examples/demo_company_profile.json
```

### 2. Performance Testing

```bash
# Generate benchmark data
python scripts/generate_sample_data.py --size 10000 --format csv --output benchmark_data

# Run benchmark
python scripts/benchmark.py --dataset-size large

# Review results
cat benchmark_report.md
```

### 3. Production Deployment

```bash
# Validate all schemas before deployment
python scripts/validate_schemas.py --check-esrs-coverage

# Run with production data
python scripts/run_full_pipeline.py \
  --esg-data production_data/esg_2024.xlsx \
  --company-profile production_data/company_profile.json \
  --config config/production_config.yaml \
  --output output/csrd_2024_report

# Verify compliance
cat output/csrd_2024_report/pipeline_summary.md
```

### 4. Batch Processing

```bash
# Create batch configuration
cat > batch_config.json <<EOF
{
  "companies": [
    {
      "company_id": "company_001",
      "esg_data_file": "data/company_001.csv",
      "company_profile": {...}
    },
    {
      "company_id": "company_002",
      "esg_data_file": "data/company_002.csv",
      "company_profile": {...}
    }
  ]
}
EOF

# Run batch
python scripts/run_full_pipeline.py \
  --batch batch_config.json \
  --output output/batch_reports
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Ensure you're in the project root
cd /path/to/CSRD-Reporting-Platform

# Run scripts from project root
python scripts/benchmark.py --dataset-size small
```

**2. Missing Dependencies**

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import pandas, click, rich, jsonschema; print('OK')"
```

**3. Configuration Errors**

```bash
# Check config file exists
ls config/csrd_config.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/csrd_config.yaml'))"
```

**4. Data File Issues**

```bash
# Check file format
file data/esg_data.csv

# Test with generated data first
python scripts/generate_sample_data.py --size 10
python scripts/run_full_pipeline.py \
  --esg-data sample_esg_data.csv \
  --company-profile examples/demo_company_profile.json
```

---

## Performance Tips

1. **Use appropriate dataset sizes for testing:**
   - Development: tiny/small (10-100 records)
   - Integration: medium (1,000 records)
   - Production: large (10,000 records)

2. **Enable caching when running multiple benchmarks:**
   ```bash
   python scripts/benchmark.py --dataset-size small --skip-data-generation
   ```

3. **For large datasets, monitor memory usage:**
   ```bash
   # Benchmark includes memory profiling
   python scripts/benchmark.py --dataset-size large --agents all
   ```

4. **Use batch processing for multiple companies:**
   ```bash
   # More efficient than running individually
   python scripts/run_full_pipeline.py --batch companies.json
   ```

---

## Contributing

When adding new utility scripts:

1. Follow the existing pattern (Click CLI + Rich output)
2. Include comprehensive help text
3. Add error handling
4. Generate output reports
5. Update this README

---

## Support

For issues or questions:

1. Check the main [README.md](../README.md)
2. Review [TESTING_GUIDE.md](../TESTING_GUIDE.md)
3. See agent-specific documentation in `docs/`

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

**Last Updated:** 2024-10-18
**Version:** 1.0.0
**Author:** GreenLang CSRD Team
