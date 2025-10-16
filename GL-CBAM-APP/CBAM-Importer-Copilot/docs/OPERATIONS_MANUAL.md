# CBAM IMPORTER COPILOT - COMPLETE OPERATIONS MANUAL

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**Audience:** Developers, DevOps, System Administrators, Power Users

---

## 📚 TABLE OF CONTENTS

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Building from Source](#building-from-source)
5. [CLI Operations](#cli-operations)
6. [SDK Operations](#sdk-operations)
7. [Configuration Management](#configuration-management)
8. [Common Workflows](#common-workflows)
9. [Advanced Usage](#advanced-usage)
10. [Integration Patterns](#integration-patterns)
11. [Maintenance & Operations](#maintenance--operations)
12. [Troubleshooting](#troubleshooting)
13. [Quick Reference](#quick-reference)

---

## 1. INTRODUCTION

### What is CBAM Importer Copilot?

The **CBAM Importer Copilot** is an AI-powered automation system for EU Carbon Border Adjustment Mechanism (CBAM) compliance reporting. It transforms a 5-day manual process into a 10-minute automated workflow.

### Key Capabilities

- ✅ **Automated Data Processing** - Ingests CSV, Excel, JSON shipment data
- ✅ **Zero Hallucination Calculations** - 100% deterministic emissions calculations
- ✅ **EU Compliance** - Implements all CBAM transitional registry requirements
- ✅ **Enterprise Provenance** - SHA256 file integrity + complete audit trails
- ✅ **Lightning Fast** - 10,000 shipments processed in ~30 seconds

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   CBAM IMPORTER COPILOT                      │
│              3-Agent AI System on GreenLang                  │
└─────────────────────────────────────────────────────────────┘

INPUT: shipments.csv/xlsx/json (any size)
  ↓
┌──────────────────────────────┐
│ Agent 1: Intake & Validation │  50+ CBAM rules
│ Performance: 1000 rec/sec    │  CN code enrichment
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ Agent 2: Emissions Calculator│  🔒 Zero Hallucination
│ Performance: <3ms/record     │  100% deterministic
└──────────────────────────────┘
  ↓
┌──────────────────────────────┐
│ Agent 3: Report Packager     │  EU Registry format
│ Performance: <1s for 10K     │  Provenance included
└──────────────────────────────┘
  ↓
OUTPUT: cbam_report.json + provenance
```

---

## 2. PREREQUISITES

### System Requirements

#### Minimum Requirements
- **OS:** Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)
- **CPU:** 2 cores, 2.0 GHz
- **RAM:** 4 GB
- **Disk:** 500 MB free space
- **Python:** 3.9 or higher

#### Recommended for Production
- **OS:** Windows Server 2019+, Ubuntu 22.04 LTS
- **CPU:** 4+ cores, 3.0 GHz
- **RAM:** 8 GB
- **Disk:** 2 GB free space (for data + logs)
- **Python:** 3.11+ (best performance)

### Required Software

#### 1. Python 3.9+

**Check Python version:**
```bash
python --version
# Expected output: Python 3.9.x or higher
```

**Install Python (if needed):**

**Windows:**
```bash
# Download from python.org
# Or use Chocolatey:
choco install python --version=3.11.0
```

**macOS:**
```bash
brew install python@3.11
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### 2. pip (Python Package Manager)

**Check pip version:**
```bash
pip --version
# Expected output: pip 23.x or higher
```

**Upgrade pip:**
```bash
python -m pip install --upgrade pip
```

#### 3. Git (for source installation)

**Check Git:**
```bash
git --version
# Expected output: git version 2.x
```

**Install Git:**
- **Windows:** Download from git-scm.com
- **macOS:** `brew install git`
- **Linux:** `sudo apt install git`

#### 4. GreenLang CLI (optional, for pack installation)

**Install GreenLang CLI:**
```bash
pip install greenlang-cli
gl --version
```

### Optional Tools

- **Docker** - For containerized deployment
- **PostgreSQL** - For persistent data storage
- **Redis** - For caching (high-volume scenarios)

### Network Requirements

- **Outbound HTTPS:** For downloading dependencies (pypi.org)
- **No inbound connections required** (on-premise tool)

---

## 3. INSTALLATION

### Method 1: Quick Install (Recommended)

**For end users who want to get started immediately:**

```bash
# Step 1: Create isolated environment
python -m venv cbam-env

# Step 2: Activate environment
# Windows:
cbam-env\Scripts\activate
# macOS/Linux:
source cbam-env/bin/activate

# Step 3: Install from requirements
cd path/to/GL-CBAM-APP/CBAM-Importer-Copilot
pip install -r requirements.txt

# Step 4: Verify installation
python -c "import cbam_pipeline; print('✓ Installation successful!')"
```

**Expected output:**
```
✓ Installation successful!
```

**Time to complete:** ~2 minutes

---

### Method 2: GreenLang Pack Installation

**For GreenLang users (future Hub publication):**

```bash
# Install from GreenLang Hub
gl pack install cbam-importer-demo

# Verify installation
gl cbam --version
```

**Status:** Not yet published to Hub (Phase 10 complete, publication pending)

---

### Method 3: Development Installation

**For developers who want to modify the code:**

```bash
# Step 1: Clone repository (if from Git)
git clone https://github.com/your-org/cbam-importer-copilot.git
cd cbam-importer-copilot

# Step 2: Create development environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Step 3: Install in editable mode
pip install -e .

# Step 4: Install development dependencies
pip install -r requirements-dev.txt

# Step 5: Run tests to verify
python -m pytest tests/ -v
```

**Expected output:**
```
======================== 212 passed in 30.5s ========================
```

---

### Method 4: Docker Installation

**For containerized deployment:**

```bash
# Step 1: Build Docker image
cd path/to/GL-CBAM-APP/CBAM-Importer-Copilot
docker build -t cbam-importer:1.0.0 .

# Step 2: Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  cbam-importer:1.0.0

# Step 3: Verify
docker run cbam-importer:1.0.0 python -c "import cbam_pipeline; print('✓ OK')"
```

**Note:** Dockerfile not included in v1.0.0 (add in Phase 10 if needed)

---

### Verification Checklist

After installation, verify all components:

```bash
# 1. Check Python environment
python --version
pip list | grep -E "(pandas|pydantic|jsonschema|pyyaml|openpyxl)"

# 2. Check core modules
python -c "from cbam_pipeline import CBAMPipeline; print('✓ Pipeline OK')"
python -c "from agents.shipment_intake_agent import ShipmentIntakeAgent; print('✓ Agent 1 OK')"
python -c "from agents.emissions_calculator_agent import EmissionsCalculatorAgent; print('✓ Agent 2 OK')"
python -c "from agents.reporting_packager_agent import ReportingPackagerAgent; print('✓ Agent 3 OK')"

# 3. Check CLI (if installed via GreenLang)
gl cbam --help

# 4. Check SDK
python -c "from sdk.cbam_sdk import cbam_build_report; print('✓ SDK OK')"

# 5. Check data files
ls -la data/cn_codes.json
ls -la data/emission_factors.py
ls -la rules/cbam_rules.yaml
```

**All checks should pass with "✓ OK" or similar success message.**

---

## 4. BUILDING FROM SOURCE

### Complete Build Process

This section covers building the CBAM Importer Copilot from scratch.

#### Step 1: Prepare Development Environment

```bash
# Create project directory
mkdir cbam-build
cd cbam-build

# Create virtual environment
python -m venv build-env

# Activate environment
# Windows:
build-env\Scripts\activate
# macOS/Linux:
source build-env/bin/activate

# Upgrade build tools
python -m pip install --upgrade pip setuptools wheel build
```

#### Step 2: Clone or Copy Source Code

**Option A: From Git repository**
```bash
git clone https://github.com/your-org/cbam-importer-copilot.git
cd cbam-importer-copilot
```

**Option B: From local directory**
```bash
cp -r C:/Users/aksha/Code-V1_GreenLang/GL-CBAM-APP/CBAM-Importer-Copilot .
cd CBAM-Importer-Copilot
```

#### Step 3: Verify Directory Structure

```bash
# Check all required directories exist
ls -la

# Expected structure:
# agents/
# cli/
# config/
# data/
# docs/
# examples/
# provenance/
# rules/
# schemas/
# scripts/
# sdk/
# specs/
# tests/
# cbam_pipeline.py
# requirements.txt
# pack.yaml
# gl.yaml
# README.md
# LICENSE
```

#### Step 4: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Verify key dependencies
pip list | grep pandas
pip list | grep pydantic
pip list | grep jsonschema
```

**Expected versions:**
- pandas >= 2.0.0
- pydantic >= 2.0.0
- jsonschema >= 4.0.0
- pyyaml >= 6.0
- openpyxl >= 3.1.0

#### Step 5: Build Python Package (Optional)

**If creating distributable package:**

```bash
# Create pyproject.toml (if not exists)
cat > pyproject.toml <<EOF
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cbam-importer-copilot"
version = "1.0.0"
description = "EU CBAM Compliance Automation"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "pandas>=2.0.0",
    "pydantic>=2.0.0",
    "jsonschema>=4.0.0",
    "pyyaml>=6.0",
    "openpyxl>=3.1.0",
    "click>=8.0.0",
    "rich>=13.0.0"
]

[project.scripts]
cbam = "cli.cbam_commands:cli"
EOF

# Build package
python -m build

# Output: dist/cbam_importer_copilot-1.0.0.tar.gz
#         dist/cbam_importer_copilot-1.0.0-py3-none-any.whl
```

#### Step 6: Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run full test suite
python -m pytest tests/ -v --tb=short

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term --cov-report=html

# Expected output:
# ======================== 212 passed in 30.5s ========================
# Coverage: 85%+
```

#### Step 7: Run Security Scan

```bash
# Install security tools
pip install bandit safety

# Run security scan
scripts/security_scan.bat  # Windows
# or
./scripts/security_scan.sh  # macOS/Linux (if exists)

# Expected output:
# ✓ No high-severity issues found
# Security Score: 92/100 (A Grade)
```

#### Step 8: Build Documentation (Optional)

```bash
# Install documentation tools
pip install mkdocs mkdocs-material

# Build HTML docs
mkdocs build

# Serve locally for preview
mkdocs serve
# Open browser to http://127.0.0.1:8000
```

#### Step 9: Create Distribution

```bash
# Create source distribution
tar -czf cbam-importer-copilot-1.0.0-src.tar.gz \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='*.pyc' \
  .

# Create binary distribution (wheel)
python -m build --wheel

# Verify distributions
ls -lh dist/
# cbam_importer_copilot-1.0.0-py3-none-any.whl
# cbam_importer_copilot-1.0.0.tar.gz
```

#### Step 10: Install Built Package

```bash
# Install from wheel
pip install dist/cbam_importer_copilot-1.0.0-py3-none-any.whl

# Verify installation
python -c "import cbam_pipeline; print(cbam_pipeline.__version__)"
# Expected: 1.0.0

cbam --version
# Expected: CBAM Importer Copilot v1.0.0
```

---

### Build Troubleshooting

#### Issue: "No module named 'setuptools'"
**Solution:**
```bash
pip install --upgrade setuptools wheel
```

#### Issue: "Microsoft Visual C++ required" (Windows)
**Solution:**
```bash
# Install Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Issue: Tests fail with import errors
**Solution:**
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # macOS/Linux
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

---

## 5. CLI OPERATIONS

### Overview

The CBAM Importer Copilot provides 3 main CLI commands:

1. `gl cbam report` - Generate CBAM compliance reports
2. `gl cbam config` - Manage configuration
3. `gl cbam validate` - Validate shipment data

**Note:** If not installed via GreenLang, replace `gl cbam` with `python -m cli.cbam_commands`

---

### Command 1: Generate Reports

#### Basic Usage

```bash
gl cbam report \
  --input examples/demo_shipments.csv \
  --output output/cbam_report.json \
  --importer-country NL \
  --importer-name "Your Company Name" \
  --importer-eori NL123456789
```

**Expected output:**
```
========================================
CBAM Importer Copilot - Report Generator
========================================

✓ Reading shipments from: examples/demo_shipments.csv
  Total records: 5

✓ Validating CBAM rules...
  Errors: 0
  Warnings: 0

✓ Calculating embedded emissions...
  Records processed: 5
  Total emissions: 192.85 tCO2

✓ Generating compliance report...
  Report structure: Valid
  Aggregations: Complete

✓ Creating provenance record...
  File hash: a3f5c8d2e9b1...
  Timestamp: 2025-10-15T14:23:45Z

✓ Report saved to: output/cbam_report.json

========================================
Report Generation Complete!
========================================

Summary:
  - Total Shipments: 5
  - Total Emissions: 192.85 tCO2
  - Complex Goods: 1 (20.0%)
  - Processing Time: 0.8s
```

---

#### Advanced Usage

**With configuration file:**
```bash
gl cbam report \
  --input shipments.csv \
  --config config/my_config.yaml \
  --output output/report.json
```

**With supplier data:**
```bash
gl cbam report \
  --input shipments.csv \
  --config config/my_config.yaml \
  --suppliers data/suppliers.yaml \
  --output output/report.json
```

**With provenance enabled:**
```bash
gl cbam report \
  --input shipments.csv \
  --config config/my_config.yaml \
  --output output/report.json \
  --provenance
```

**Multiple output formats:**
```bash
# JSON output (default)
gl cbam report --input shipments.csv --output report.json --format json

# Excel output
gl cbam report --input shipments.csv --output report.xlsx --format excel

# CSV output
gl cbam report --input shipments.csv --output report.csv --format csv
```

**Verbose mode (debugging):**
```bash
gl cbam report \
  --input shipments.csv \
  --output report.json \
  --importer-country NL \
  --importer-name "Test Co" \
  --importer-eori NL123456789 \
  --verbose
```

**Expected verbose output:**
```
[DEBUG] Loading CN codes from: data/cn_codes.json
[DEBUG] Loading CBAM rules from: rules/cbam_rules.yaml
[DEBUG] Loading emission factors from: data/emission_factors.py
[INFO] Processing 1000 shipments...
[DEBUG] Agent 1: Validating record 1/1000...
[DEBUG] Agent 1: Validating record 2/1000...
...
[INFO] Agent 1 complete: 1000 records validated in 1.2s
[DEBUG] Agent 2: Calculating emissions for 1000 records...
...
```

---

#### Complete Parameter Reference

```bash
gl cbam report [OPTIONS]

Required (if no config file):
  --input PATH              Input file (CSV/Excel/JSON)
  --output PATH             Output file path
  --importer-country CODE   2-letter country code (e.g., NL, DE, FR)
  --importer-name TEXT      Company name
  --importer-eori TEXT      EORI number

Optional:
  --config PATH             Configuration YAML file
  --suppliers PATH          Supplier data YAML/JSON file
  --format [json|excel|csv] Output format (default: json)
  --provenance              Enable provenance tracking
  --verbose                 Enable verbose logging
  --help                    Show help message
```

---

### Command 2: Configuration Management

#### Initialize Configuration

**Create new configuration file:**
```bash
gl cbam config init --output config/my_config.yaml
```

**Expected output:**
```
========================================
CBAM Configuration Initializer
========================================

✓ Configuration template created: config/my_config.yaml

Next steps:
  1. Edit config/my_config.yaml with your details
  2. Run: gl cbam config validate --config config/my_config.yaml
  3. Use: gl cbam report --config config/my_config.yaml

See docs/USER_GUIDE.md for configuration options.
```

**Generated config file structure:**
```yaml
# CBAM Importer Copilot Configuration
# Version: 1.0.0

importer:
  name: "Your Company Name"
  eori_number: "YOUR_EORI_HERE"  # e.g., NL123456789
  country: "NL"  # 2-letter ISO code

data_sources:
  cn_codes_path: "data/cn_codes.json"
  cbam_rules_path: "rules/cbam_rules.yaml"
  emission_factors_path: "data/emission_factors.py"
  suppliers_path: null  # Optional: "data/suppliers.yaml"

processing:
  enable_provenance: true
  strict_validation: true
  performance_mode: false

output:
  default_format: "json"
  include_summary: true
  include_aggregations: true
```

---

#### Interactive Configuration

**Create config with prompts:**
```bash
gl cbam config init --interactive
```

**Interactive session:**
```
========================================
CBAM Configuration Setup (Interactive)
========================================

Enter your company details:

Importer Country (2-letter code, e.g., NL): NL
Importer Name: My Import Company
Importer EORI Number: NL123456789

Data Sources:
Use default CN codes file? (Y/n): Y
Use default CBAM rules file? (Y/n): Y
Use default emission factors? (Y/n): Y
Provide suppliers file? (y/N): N

Processing Options:
Enable provenance tracking? (Y/n): Y
Use strict validation? (Y/n): Y

✓ Configuration saved to: cbam_config.yaml
```

---

#### Show Configuration

**Display current configuration:**
```bash
gl cbam config show --config config/my_config.yaml
```

**Expected output:**
```
========================================
CBAM Configuration
========================================

Importer Information:
  Name: My Import Company
  Country: NL
  EORI: NL123456789

Data Sources:
  CN Codes: data/cn_codes.json (30 codes)
  CBAM Rules: rules/cbam_rules.yaml (50+ rules)
  Emission Factors: data/emission_factors.py (14 variants)
  Suppliers: None

Processing Settings:
  Provenance: Enabled
  Strict Validation: Enabled
  Performance Mode: Disabled

Output Settings:
  Default Format: JSON
  Include Summary: Yes
  Include Aggregations: Yes

========================================
Configuration is valid and ready to use.
========================================
```

---

#### Validate Configuration

**Check configuration validity:**
```bash
gl cbam config validate --config config/my_config.yaml
```

**Expected output (valid config):**
```
✓ Configuration file is valid
✓ All required fields present
✓ Data sources accessible
✓ Importer details complete

Configuration is ready for use.
```

**Expected output (invalid config):**
```
✗ Configuration validation failed

Errors:
  - Missing required field: importer.eori_number
  - Invalid country code: 'XX' (must be 2-letter ISO code)
  - File not found: data/custom_cn_codes.json

Please fix these issues and try again.
```

---

#### Edit Configuration

**Open configuration in default editor:**
```bash
gl cbam config edit --config config/my_config.yaml
```

**Opens the file in:**
- **Windows:** Notepad
- **macOS:** TextEdit or vim
- **Linux:** nano or vim

---

### Command 3: Data Validation

#### Basic Validation

**Validate shipment data without generating report:**
```bash
gl cbam validate --input shipments.csv --importer-country NL
```

**Expected output:**
```
========================================
CBAM Data Validation
========================================

✓ Reading data from: shipments.csv
  Total records: 100

✓ Validating CBAM compliance rules...

Validation Results:
  ✓ Records passed: 95 (95.0%)
  ✗ Records failed: 5 (5.0%)

Errors Found:
  E001 - Missing CN code: 2 records (rows 15, 42)
  E003 - Invalid country code: 1 record (row 78)
  W002 - Missing supplier ID: 2 records (rows 23, 91)

========================================
Validation Complete
========================================

Next steps:
  1. Fix errors in rows: 15, 42, 78
  2. Review warnings in rows: 23, 91
  3. Re-run validation
  4. Generate report when all errors fixed
```

---

#### Detailed Validation

**Get detailed validation report:**
```bash
gl cbam validate \
  --input shipments.csv \
  --importer-country NL \
  --detailed
```

**Expected output:**
```
========================================
CBAM Data Validation (Detailed)
========================================

File Information:
  Path: shipments.csv
  Size: 25,340 bytes
  Records: 100
  Format: CSV

Validation Summary:
  ✓ Passed: 95 records
  ✗ Failed: 5 records
  ⚠ Warnings: 2 records

Detailed Errors:

Row 15:
  Error Code: E001
  Severity: ERROR
  Field: cn_code
  Issue: Missing required field
  Value: (empty)
  Fix: Add valid 8-digit CN code

Row 42:
  Error Code: E001
  Severity: ERROR
  Field: cn_code
  Issue: Missing required field
  Value: (empty)
  Fix: Add valid 8-digit CN code

Row 78:
  Error Code: E003
  Severity: ERROR
  Field: country_of_origin
  Issue: Invalid country code
  Value: 'XX'
  Fix: Use valid 2-letter ISO code (e.g., CN, IN, RU)

Detailed Warnings:

Row 23:
  Warning Code: W002
  Severity: WARNING
  Field: supplier_id
  Issue: Missing supplier ID
  Value: (empty)
  Impact: Cannot link to supplier actuals
  Recommendation: Add supplier ID if available

Row 91:
  Warning Code: W002
  Severity: WARNING
  Field: supplier_id
  Issue: Missing supplier ID
  Value: (empty)
  Impact: Cannot link to supplier actuals
  Recommendation: Add supplier ID if available

========================================
Fix these issues before generating report
========================================
```

---

#### Validation Output to File

**Save validation results to JSON:**
```bash
gl cbam validate \
  --input shipments.csv \
  --importer-country NL \
  --output validation_results.json \
  --format json
```

**Output file structure:**
```json
{
  "metadata": {
    "input_file": "shipments.csv",
    "total_records": 100,
    "validated_at": "2025-10-15T14:30:00Z"
  },
  "summary": {
    "passed": 95,
    "failed": 5,
    "warnings": 2
  },
  "errors": [
    {
      "row": 15,
      "code": "E001",
      "severity": "ERROR",
      "field": "cn_code",
      "message": "Missing required field",
      "value": null
    }
  ],
  "warnings": [
    {
      "row": 23,
      "code": "W002",
      "severity": "WARNING",
      "field": "supplier_id",
      "message": "Missing supplier ID",
      "value": null
    }
  ],
  "is_valid": false
}
```

---

#### Strict Validation Mode

**Treat warnings as errors:**
```bash
gl cbam validate \
  --input shipments.csv \
  --importer-country NL \
  --strict
```

**In strict mode:**
- Warnings become errors
- Exit code: 1 if any warnings or errors
- Useful for CI/CD pipelines

---

## 6. SDK OPERATIONS

### Overview

The CBAM Importer Copilot SDK provides a simple Python API for programmatic access.

### Core Components

1. **`cbam_build_report()`** - Main function to generate reports
2. **`CBAMConfig`** - Configuration dataclass
3. **`CBAMReport`** - Report result object
4. **`cbam_validate_shipments()`** - Validation function
5. **`cbam_calculate_emissions()`** - Calculation function

---

### Basic Usage

#### 5-Line Quick Start

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Load configuration
config = CBAMConfig.from_yaml('config/my_config.yaml')

# Generate report
report = cbam_build_report('shipments.csv', config)

# Access results
print(f"Total Emissions: {report.total_emissions_tco2} tCO2")
```

**Output:**
```
Total Emissions: 1,250.45 tCO2
```

---

### Detailed SDK Usage

#### Example 1: Generate Report from CSV

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Method 1: Using config file
config = CBAMConfig.from_yaml('config/my_config.yaml')
report = cbam_build_report(
    input_file='data/shipments.csv',
    config=config,
    save_output=True,
    output_path='output/cbam_report.json'
)

# Method 2: Using config object
config = CBAMConfig(
    importer_name="My Company",
    importer_country="NL",
    importer_eori="NL123456789",
    cn_codes_path="data/cn_codes.json",
    cbam_rules_path="rules/cbam_rules.yaml",
    enable_provenance=True
)

report = cbam_build_report(
    input_file='data/shipments.csv',
    config=config
)

# Access report properties
print(f"Report ID: {report.report_id}")
print(f"Generated At: {report.generated_at}")
print(f"Total Emissions: {report.total_emissions_tco2} tCO2")
print(f"Total Shipments: {report.total_shipments}")
print(f"Complex Goods %: {report.complex_goods_percentage}%")

# Get detailed goods
for good in report.detailed_goods[:5]:  # First 5 goods
    print(f"  CN Code: {good.cn_code}")
    print(f"  Quantity: {good.quantity_tons} tons")
    print(f"  Emissions: {good.embedded_emissions_tco2} tCO2")
    print()
```

---

#### Example 2: Work with pandas DataFrame

```python
import pandas as pd
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Load shipments into DataFrame
df = pd.read_csv('shipments.csv')

# Filter shipments (e.g., last month only)
df_filtered = df[df['import_date'] >= '2025-09-01']

# Generate report from DataFrame
config = CBAMConfig.from_yaml('config/my_config.yaml')
report = cbam_build_report(
    input_data=df_filtered,  # Pass DataFrame directly
    config=config
)

# Convert report back to DataFrame
report_df = report.to_dataframe()

# Analyze with pandas
print(report_df.groupby('product_group')['embedded_emissions_tco2'].sum())

# Output:
# product_group
# cement         125.30
# iron_steel     890.15
# aluminum       235.00
# Name: embedded_emissions_tco2, dtype: float64
```

---

#### Example 3: Validate Before Processing

```python
from sdk.cbam_sdk import cbam_validate_shipments, CBAMConfig

# Load configuration
config = CBAMConfig.from_yaml('config/my_config.yaml')

# Validate shipments
validation_result = cbam_validate_shipments(
    input_file='shipments.csv',
    config=config
)

# Check validation results
if validation_result['is_valid']:
    print("✓ All shipments valid, proceeding with report generation...")
    from sdk.cbam_sdk import cbam_build_report
    report = cbam_build_report('shipments.csv', config)
else:
    print("✗ Validation failed:")
    for error in validation_result['errors']:
        print(f"  Row {error['row']}: {error['message']}")

    # Fix errors before proceeding
    sys.exit(1)
```

---

#### Example 4: Calculate Emissions Only

```python
from sdk.cbam_sdk import cbam_calculate_emissions, CBAMConfig
import pandas as pd

# Load validated shipments
df = pd.read_csv('validated_shipments.csv')

# Calculate emissions
config = CBAMConfig.from_yaml('config/my_config.yaml')
emissions_data = cbam_calculate_emissions(
    validated_shipments=df.to_dict('records'),
    config=config
)

# Access calculation results
total_emissions = emissions_data['total_emissions_tco2']
shipments_with_emissions = emissions_data['shipments_with_emissions']

print(f"Total Embedded Emissions: {total_emissions} tCO2")

# Save to file
import json
with open('emissions_data.json', 'w') as f:
    json.dump(emissions_data, f, indent=2)
```

---

#### Example 5: Export to Excel

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Generate report
config = CBAMConfig.from_yaml('config/my_config.yaml')
report = cbam_build_report('shipments.csv', config)

# Export to Excel with multiple sheets
report.to_excel(
    output_path='output/cbam_report.xlsx',
    include_summary=True,
    include_aggregations=True,
    include_provenance=True
)

# Output file structure:
# Sheet 1: Summary (emissions summary, validation results)
# Sheet 2: Detailed Goods (all shipments with emissions)
# Sheet 3: Aggregations (by CN code, country, product group)
# Sheet 4: Provenance (file integrity, audit trail)
```

---

#### Example 6: Batch Processing

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure
config = CBAMConfig.from_yaml('config/my_config.yaml')

# Process all CSV files in directory
input_dir = Path('data/monthly_shipments')
output_dir = Path('output/monthly_reports')
output_dir.mkdir(exist_ok=True)

for csv_file in input_dir.glob('*.csv'):
    logger.info(f"Processing {csv_file.name}...")

    try:
        # Generate report
        report = cbam_build_report(
            input_file=str(csv_file),
            config=config,
            save_output=True,
            output_path=str(output_dir / f"{csv_file.stem}_report.json")
        )

        logger.info(f"  ✓ Success: {report.total_emissions_tco2} tCO2")

    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        continue

logger.info("Batch processing complete!")
```

---

### CBAMConfig Reference

```python
from sdk.cbam_sdk import CBAMConfig

# Create from dictionary
config = CBAMConfig(
    # Required
    importer_name="My Company",
    importer_country="NL",
    importer_eori="NL123456789",

    # Data sources
    cn_codes_path="data/cn_codes.json",
    cbam_rules_path="rules/cbam_rules.yaml",
    emission_factors_path="data/emission_factors.py",
    suppliers_path="data/suppliers.yaml",  # Optional

    # Processing options
    enable_provenance=True,
    strict_validation=True,
    performance_mode=False
)

# Create from YAML file
config = CBAMConfig.from_yaml('config/my_config.yaml')

# Create from dictionary
config_dict = {
    'importer': {
        'name': 'My Company',
        'country': 'NL',
        'eori_number': 'NL123456789'
    },
    'data_sources': {
        'cn_codes_path': 'data/cn_codes.json',
        'cbam_rules_path': 'rules/cbam_rules.yaml'
    },
    'processing': {
        'enable_provenance': True
    }
}
config = CBAMConfig.from_dict(config_dict)

# Save to YAML
config.to_yaml('config/saved_config.yaml')

# Access properties
print(config.importer_name)        # "My Company"
print(config.importer_country)     # "NL"
print(config.enable_provenance)    # True
```

---

### CBAMReport Reference

```python
# After generating report
report = cbam_build_report('shipments.csv', config)

# Properties
report.report_id                    # "CBAM-2025Q3-12345"
report.generated_at                 # "2025-10-15T14:23:45Z"
report.total_emissions_tco2         # 1250.45
report.total_shipments              # 1000
report.total_quantity_tons          # 5000.0
report.complex_goods_percentage     # 15.5
report.is_valid                     # True

# Collections
report.detailed_goods               # List[dict] - All shipments with emissions
report.aggregations                 # Dict - Aggregated by CN code, country, group
report.validation_results           # Dict - Validation summary
report.provenance                   # Dict - File integrity & audit trail

# Methods
report.to_dict()                    # Convert to dictionary
report.to_json(indent=2)            # Convert to JSON string
report.to_dataframe()               # Convert to pandas DataFrame
report.save('output/report.json')   # Save to file
report.to_excel('output/report.xlsx')  # Export to Excel

# Example: Filter high-emission goods
high_emission_goods = [
    good for good in report.detailed_goods
    if good['embedded_emissions_tco2'] > 10.0
]
```

---

## 7. CONFIGURATION MANAGEMENT

### Configuration File Structure

Complete `cbam_config.yaml` reference:

```yaml
# ==================================================
# CBAM Importer Copilot Configuration
# Version: 1.0.0
# ==================================================

# --------------------------------------------------
# Importer Information (REQUIRED)
# --------------------------------------------------
importer:
  name: "Your Company Name"
  eori_number: "NL123456789"  # EU EORI number
  country: "NL"  # 2-letter ISO country code

  # Optional fields
  address: "123 Business Street, Amsterdam"
  contact_email: "compliance@yourcompany.com"
  contact_phone: "+31 20 1234567"

# --------------------------------------------------
# Data Sources
# --------------------------------------------------
data_sources:
  # CN Codes mapping (REQUIRED)
  cn_codes_path: "data/cn_codes.json"

  # CBAM validation rules (REQUIRED)
  cbam_rules_path: "rules/cbam_rules.yaml"

  # Emission factors database (REQUIRED)
  emission_factors_path: "data/emission_factors.py"

  # Supplier actual emissions (OPTIONAL)
  suppliers_path: "data/suppliers.yaml"  # null if not used

  # Custom emission factors (OPTIONAL)
  custom_factors_path: null  # "data/custom_factors.yaml"

# --------------------------------------------------
# Processing Options
# --------------------------------------------------
processing:
  # Enable provenance tracking (recommended: true)
  enable_provenance: true

  # Strict validation mode (treat warnings as errors)
  strict_validation: false

  # Performance mode (skip non-critical validations)
  performance_mode: false

  # Parallel processing (for large datasets)
  use_parallel: false
  max_workers: 4

  # Batch size for processing
  batch_size: 1000

# --------------------------------------------------
# Output Settings
# --------------------------------------------------
output:
  # Default output format
  default_format: "json"  # json | excel | csv

  # Include sections in output
  include_summary: true
  include_detailed_goods: true
  include_aggregations: true
  include_provenance: true
  include_validation_results: true

  # Output directory
  output_directory: "output"

  # File naming pattern
  filename_pattern: "cbam_report_{date}_{time}.json"

# --------------------------------------------------
# Validation Rules
# --------------------------------------------------
validation:
  # Require all mandatory fields
  enforce_mandatory_fields: true

  # Validate CN codes against database
  validate_cn_codes: true

  # Validate country codes (ISO 3166-1 alpha-2)
  validate_country_codes: true

  # Check quantity ranges
  min_quantity_tons: 0.001
  max_quantity_tons: 100000.0

  # Date validation
  allow_future_dates: false
  earliest_date: "2023-10-01"  # CBAM start date

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging:
  # Log level: DEBUG | INFO | WARNING | ERROR
  level: "INFO"

  # Log to file
  log_to_file: true
  log_file_path: "logs/cbam.log"

  # Log rotation
  max_log_size_mb: 10
  backup_count: 5

# --------------------------------------------------
# Performance Tuning
# --------------------------------------------------
performance:
  # Cache CN codes in memory
  cache_cn_codes: true

  # Cache emission factors
  cache_emission_factors: true

  # Chunk size for large files
  chunk_size: 10000

  # Memory limit (MB)
  max_memory_mb: 2048

# --------------------------------------------------
# Integration
# --------------------------------------------------
integration:
  # ERP system integration
  erp_system: null  # "SAP" | "Oracle" | "Custom"
  erp_api_endpoint: null
  erp_api_key_env: "CBAM_ERP_API_KEY"

  # Database connection (for persistent storage)
  database_url: null  # "postgresql://user:pass@host:port/db"

  # Webhook for notifications
  webhook_url: null
  webhook_events: []  # ["report_complete", "validation_failed"]
```

---

### Environment Variables

Override configuration with environment variables:

```bash
# Importer information
export CBAM_IMPORTER_NAME="My Company"
export CBAM_IMPORTER_COUNTRY="NL"
export CBAM_IMPORTER_EORI="NL123456789"

# Data sources
export CBAM_CN_CODES_PATH="/custom/path/cn_codes.json"
export CBAM_RULES_PATH="/custom/path/cbam_rules.yaml"
export CBAM_SUPPLIERS_PATH="/custom/path/suppliers.yaml"

# Processing options
export CBAM_ENABLE_PROVENANCE="true"
export CBAM_STRICT_VALIDATION="false"

# Output settings
export CBAM_OUTPUT_FORMAT="excel"
export CBAM_OUTPUT_DIR="/custom/output"

# Logging
export CBAM_LOG_LEVEL="DEBUG"
export CBAM_LOG_FILE="/var/log/cbam/app.log"

# Integration
export CBAM_ERP_API_KEY="secret_api_key_here"
export CBAM_DATABASE_URL="postgresql://user:pass@localhost:5432/cbam"

# Run with environment variables
gl cbam report --input shipments.csv --output report.json
```

**Priority order (highest to lowest):**
1. Command-line arguments
2. Environment variables
3. Configuration file
4. Default values

---

### Multi-Tenant Configuration

**Setup for multiple companies:**

```bash
# Company A configuration
config/
  company_a_config.yaml
  company_b_config.yaml
  company_c_config.yaml

# Run for each company
gl cbam report --input data/company_a.csv --config config/company_a_config.yaml --output output/company_a_report.json
gl cbam report --input data/company_b.csv --config config/company_b_config.yaml --output output/company_b_report.json
gl cbam report --input data/company_c.csv --config config/company_c_config.yaml --output output/company_c_report.json
```

**Automated multi-tenant processing:**

```python
from pathlib import Path
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Process all companies
companies = ['company_a', 'company_b', 'company_c']

for company in companies:
    config = CBAMConfig.from_yaml(f'config/{company}_config.yaml')

    report = cbam_build_report(
        input_file=f'data/{company}.csv',
        config=config,
        save_output=True,
        output_path=f'output/{company}_report.json'
    )

    print(f"{company}: {report.total_emissions_tco2} tCO2")
```

---

## 8. COMMON WORKFLOWS

### Workflow 1: First-Time Setup

**Step-by-step guide for new users:**

```bash
# Step 1: Install CBAM Importer Copilot
python -m venv cbam-env
source cbam-env/bin/activate  # or cbam-env\Scripts\activate on Windows
cd path/to/GL-CBAM-APP/CBAM-Importer-Copilot
pip install -r requirements.txt

# Step 2: Verify installation
python -c "from cbam_pipeline import CBAMPipeline; print('✓ Installation OK')"

# Step 3: Create configuration
mkdir -p config output logs
gl cbam config init --output config/my_config.yaml

# Step 4: Edit configuration
# Open config/my_config.yaml and fill in your company details:
#   - importer.name
#   - importer.eori_number
#   - importer.country

# Step 5: Validate configuration
gl cbam config validate --config config/my_config.yaml

# Step 6: Test with demo data
gl cbam report \
  --input examples/demo_shipments.csv \
  --config config/my_config.yaml \
  --output output/test_report.json \
  --provenance

# Step 7: Review output
cat output/test_report.json | jq '.emissions_summary'

# Expected output:
# {
#   "total_embedded_emissions_tco2": 192.85,
#   "total_shipments": 5,
#   "total_quantity_tons": 75.5,
#   ...
# }

# Step 8: Process real data
gl cbam report \
  --input data/my_shipments.csv \
  --config config/my_config.yaml \
  --output output/cbam_report.json \
  --provenance

# Success! You're now ready for production use.
```

---

### Workflow 2: Monthly Reporting

**Typical monthly CBAM reporting workflow:**

```bash
# Month: September 2025

# Step 1: Extract shipment data from ERP
# (This varies by ERP system - export to CSV/Excel)

# Step 2: Place file in input directory
cp /path/from/erp/september_2025_shipments.csv data/input/

# Step 3: Validate data
gl cbam validate \
  --input data/input/september_2025_shipments.csv \
  --config config/production_config.yaml \
  --detailed

# Step 4: Fix any validation errors
# Edit CSV file to correct errors
# Re-run validation until clean

# Step 5: Generate monthly report
gl cbam report \
  --input data/input/september_2025_shipments.csv \
  --config config/production_config.yaml \
  --output output/cbam_report_2025_09.json \
  --provenance \
  --format json

# Step 6: Generate Excel version for review
gl cbam report \
  --input data/input/september_2025_shipments.csv \
  --config config/production_config.yaml \
  --output output/cbam_report_2025_09.xlsx \
  --format excel

# Step 7: Review report
# Open output/cbam_report_2025_09.xlsx in Excel
# Check emissions summary, aggregations, validations

# Step 8: Archive monthly data
mkdir -p archive/2025/09
cp data/input/september_2025_shipments.csv archive/2025/09/
cp output/cbam_report_2025_09.* archive/2025/09/

# Step 9: Submit to EU CBAM Registry
# Upload cbam_report_2025_09.json to EU portal
# (Manual upload currently - API integration in v2.0)

# Step 10: Document submission
echo "2025-09 CBAM report submitted on $(date)" >> logs/submissions.log
```

---

### Workflow 3: Quarterly Reporting with Aggregation

**Aggregate 3 months into quarterly report:**

```python
#!/usr/bin/env python
"""
Quarterly CBAM Report Generator
Combines 3 monthly reports into quarterly submission
"""

import pandas as pd
from sdk.cbam_sdk import cbam_build_report, CBAMConfig
from pathlib import Path

# Configuration
quarter = "2025-Q3"
months = ['2025-07', '2025-08', '2025-09']
config = CBAMConfig.from_yaml('config/production_config.yaml')

# Step 1: Load all monthly shipments
monthly_data = []
for month in months:
    file_path = f"archive/{month[:4]}/{month[5:]}/shipments.csv"
    df = pd.read_csv(file_path)
    df['reporting_month'] = month
    monthly_data.append(df)

# Step 2: Combine into quarterly dataset
quarterly_df = pd.concat(monthly_data, ignore_index=True)

print(f"Quarterly Dataset: {len(quarterly_df)} shipments")
print(f"  July: {len(monthly_data[0])} shipments")
print(f"  August: {len(monthly_data[1])} shipments")
print(f"  September: {len(monthly_data[2])} shipments")

# Step 3: Generate quarterly report
report = cbam_build_report(
    input_data=quarterly_df,
    config=config,
    save_output=True,
    output_path=f'output/cbam_report_{quarter}.json'
)

# Step 4: Print quarterly summary
print(f"\n{quarter} CBAM Report Summary:")
print(f"  Total Emissions: {report.total_emissions_tco2:,.2f} tCO2")
print(f"  Total Shipments: {report.total_shipments:,}")
print(f"  Total Quantity: {report.total_quantity_tons:,.2f} tons")
print(f"  Complex Goods: {report.complex_goods_percentage:.1f}%")

# Step 5: Export to Excel for submission
report.to_excel(f'output/cbam_report_{quarter}.xlsx')

print(f"\n✓ Quarterly report saved:")
print(f"  JSON: output/cbam_report_{quarter}.json")
print(f"  Excel: output/cbam_report_{quarter}.xlsx")
```

---

### Workflow 4: Data Quality Check

**Pre-submission data quality workflow:**

```bash
#!/bin/bash
# data_quality_check.sh
# Comprehensive data quality check before generating final report

INPUT_FILE="data/input/shipments.csv"
CONFIG="config/production_config.yaml"
OUTPUT_DIR="output/quality_check"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "CBAM Data Quality Check"
echo "========================================="
echo ""

# Check 1: File integrity
echo "[1/6] Checking file integrity..."
if [ ! -f "$INPUT_FILE" ]; then
    echo "✗ Error: Input file not found"
    exit 1
fi

FILE_SIZE=$(wc -c < "$INPUT_FILE")
echo "✓ File exists: $FILE_SIZE bytes"

# Check 2: Basic validation
echo ""
echo "[2/6] Running validation..."
gl cbam validate \
  --input "$INPUT_FILE" \
  --config "$CONFIG" \
  --output "$OUTPUT_DIR/validation.json" \
  --format json

VALIDATION_STATUS=$?
if [ $VALIDATION_STATUS -ne 0 ]; then
    echo "✗ Validation failed - check $OUTPUT_DIR/validation.json"
    cat "$OUTPUT_DIR/validation.json" | jq '.errors'
    exit 1
fi

echo "✓ Validation passed"

# Check 3: Record counts
echo ""
echo "[3/6] Checking record counts..."
TOTAL_RECORDS=$(wc -l < "$INPUT_FILE")
HEADER_LINES=1
DATA_RECORDS=$((TOTAL_RECORDS - HEADER_LINES))

echo "✓ Total records: $DATA_RECORDS"

if [ $DATA_RECORDS -eq 0 ]; then
    echo "✗ Error: No data records found"
    exit 1
fi

# Check 4: Duplicate detection
echo ""
echo "[4/6] Checking for duplicates..."
python -c "
import pandas as pd
df = pd.read_csv('$INPUT_FILE')
duplicates = df[df.duplicated(subset=['invoice_number'], keep=False)]
if len(duplicates) > 0:
    print(f'⚠ Warning: {len(duplicates)} duplicate records found')
    duplicates.to_csv('$OUTPUT_DIR/duplicates.csv', index=False)
    print(f'  See: $OUTPUT_DIR/duplicates.csv')
else:
    print('✓ No duplicates found')
"

# Check 5: Date range validation
echo ""
echo "[5/6] Checking date ranges..."
python -c "
import pandas as pd
df = pd.read_csv('$INPUT_FILE')
df['import_date'] = pd.to_datetime(df['import_date'])
min_date = df['import_date'].min()
max_date = df['import_date'].max()
print(f'✓ Date range: {min_date.date()} to {max_date.date()}')
"

# Check 6: Test report generation
echo ""
echo "[6/6] Test report generation..."
gl cbam report \
  --input "$INPUT_FILE" \
  --config "$CONFIG" \
  --output "$OUTPUT_DIR/test_report.json" \
  --provenance \
  > "$OUTPUT_DIR/generation.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Report generation successful"

    # Show summary
    cat "$OUTPUT_DIR/test_report.json" | jq '.emissions_summary' > "$OUTPUT_DIR/summary.txt"
    cat "$OUTPUT_DIR/summary.txt"
else
    echo "✗ Report generation failed"
    cat "$OUTPUT_DIR/generation.log"
    exit 1
fi

echo ""
echo "========================================="
echo "Data Quality Check: PASSED"
echo "========================================="
echo ""
echo "Ready for final report generation!"
```

---

### Workflow 5: Automated CI/CD Pipeline

**GitLab CI/CD example:**

```yaml
# .gitlab-ci.yml
# Automated CBAM reporting pipeline

stages:
  - validate
  - test
  - report
  - archive

variables:
  PYTHON_VERSION: "3.11"
  CONFIG_FILE: "config/production_config.yaml"

# Stage 1: Validate input data
validate_data:
  stage: validate
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - |
      gl cbam validate \
        --input data/input/monthly_shipments.csv \
        --config ${CONFIG_FILE} \
        --output validation_results.json \
        --detailed
  artifacts:
    paths:
      - validation_results.json
    expire_in: 30 days
  only:
    - schedules  # Run on scheduled pipelines

# Stage 2: Run tests
run_tests:
  stage: test
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - python -m pytest tests/ -v --cov=.
  coverage: '/TOTAL.*\s+(\d+%)$/'
  only:
    - schedules

# Stage 3: Generate CBAM report
generate_report:
  stage: report
  image: python:${PYTHON_VERSION}
  script:
    - pip install -r requirements.txt
    - mkdir -p output
    - |
      gl cbam report \
        --input data/input/monthly_shipments.csv \
        --config ${CONFIG_FILE} \
        --output output/cbam_report_$(date +%Y_%m).json \
        --provenance
    - |
      gl cbam report \
        --input data/input/monthly_shipments.csv \
        --config ${CONFIG_FILE} \
        --output output/cbam_report_$(date +%Y_%m).xlsx \
        --format excel
  artifacts:
    paths:
      - output/cbam_report_*.json
      - output/cbam_report_*.xlsx
    expire_in: 1 year
  dependencies:
    - validate_data
  only:
    - schedules

# Stage 4: Archive to S3
archive_reports:
  stage: archive
  image: python:${PYTHON_VERSION}
  script:
    - pip install awscli
    - |
      aws s3 cp output/cbam_report_$(date +%Y_%m).json \
        s3://my-cbam-reports/$(date +%Y)/$(date +%m)/ \
        --region eu-west-1
    - |
      aws s3 cp output/cbam_report_$(date +%Y_%m).xlsx \
        s3://my-cbam-reports/$(date +%Y)/$(date +%m)/ \
        --region eu-west-1
  dependencies:
    - generate_report
  only:
    - schedules
```

---

## 9. ADVANCED USAGE

### Custom Emission Factors

**Add company-specific emission factors:**

1. **Create custom factors file:**

```yaml
# data/custom_emission_factors.yaml

custom_factors:
  - supplier_id: "SUP-CN-001"
    product_type: "steel"
    production_method: "electric_arc_furnace"
    emission_factor_tco2_per_ton: 1.85
    direct_emissions: 0.45
    indirect_emissions: 1.40
    data_source: "Supplier verification report 2025-Q2"
    verified: true
    verification_date: "2025-07-15"

  - supplier_id: "SUP-IN-042"
    product_type: "cement"
    production_method: "dry_process"
    emission_factor_tco2_per_ton: 0.78
    direct_emissions: 0.52
    indirect_emissions: 0.26
    data_source: "Third-party audit ABC-2025-789"
    verified: true
    verification_date: "2025-08-01"
```

2. **Update configuration:**

```yaml
# config/my_config.yaml

data_sources:
  cn_codes_path: "data/cn_codes.json"
  cbam_rules_path: "rules/cbam_rules.yaml"
  emission_factors_path: "data/emission_factors.py"
  suppliers_path: "data/suppliers.yaml"
  custom_factors_path: "data/custom_emission_factors.yaml"  # Add this
```

3. **Use in processing:**

```python
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

config = CBAMConfig.from_yaml('config/my_config.yaml')

# Custom factors will be used when supplier_id matches
report = cbam_build_report('shipments.csv', config)

# Check which records used custom factors
for good in report.detailed_goods:
    if good.get('emission_factor_source') == 'supplier_actual':
        print(f"CN {good['cn_code']}: Used supplier actual ({good['emission_factor']} tCO2/ton)")
```

---

### Parallel Processing

**Process large datasets in parallel:**

```python
from sdk.cbam_sdk import CBAMConfig
from cbam_pipeline import CBAMPipeline
import pandas as pd
from multiprocessing import Pool, cpu_count

def process_chunk(args):
    """Process a chunk of shipments"""
    chunk_df, config, chunk_id = args

    # Save chunk to temp file
    temp_file = f'/tmp/chunk_{chunk_id}.csv'
    chunk_df.to_csv(temp_file, index=False)

    # Process chunk
    pipeline = CBAMPipeline(
        cn_codes_path=config.cn_codes_path,
        cbam_rules_path=config.cbam_rules_path,
        enable_provenance=False  # Disable for performance
    )

    result = pipeline.run(
        input_file=temp_file,
        importer_info={
            'name': config.importer_name,
            'country': config.importer_country,
            'eori_number': config.importer_eori
        },
        output_path=None  # Don't save individual chunks
    )

    return result

# Load large dataset
df = pd.read_csv('large_shipments.csv')  # 100,000 records
print(f"Total records: {len(df):,}")

# Split into chunks
chunk_size = 10000
chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
print(f"Split into {len(chunks)} chunks")

# Prepare arguments
config = CBAMConfig.from_yaml('config/my_config.yaml')
chunk_args = [(chunk, config, i) for i, chunk in enumerate(chunks)]

# Process in parallel
num_workers = cpu_count() - 1  # Leave one core free
print(f"Processing with {num_workers} workers...")

with Pool(processes=num_workers) as pool:
    results = pool.map(process_chunk, chunk_args)

# Combine results
total_emissions = sum(r['emissions_summary']['total_embedded_emissions_tco2'] for r in results)
total_shipments = sum(r['emissions_summary']['total_shipments'] for r in results)

print(f"\nResults:")
print(f"  Total Emissions: {total_emissions:,.2f} tCO2")
print(f"  Total Shipments: {total_shipments:,}")
```

---

### API Integration

**Integrate with external APIs:**

```python
import requests
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Step 1: Fetch data from ERP API
erp_api_url = "https://erp.company.com/api/v1/shipments"
headers = {"Authorization": f"Bearer {os.getenv('ERP_API_TOKEN')}"}

response = requests.get(
    f"{erp_api_url}?start_date=2025-09-01&end_date=2025-09-30",
    headers=headers
)
response.raise_for_status()

shipments = response.json()['data']
print(f"Fetched {len(shipments)} shipments from ERP")

# Step 2: Convert to DataFrame
import pandas as pd
df = pd.DataFrame(shipments)

# Step 3: Map ERP fields to CBAM schema
df_mapped = df.rename(columns={
    'hs_code': 'cn_code',
    'origin_country': 'country_of_origin',
    'net_weight_kg': 'quantity_tons',
    'import_timestamp': 'import_date',
    'vendor_id': 'supplier_id',
    'invoice_ref': 'invoice_number'
})

# Convert kg to tons
df_mapped['quantity_tons'] = df_mapped['quantity_tons'] / 1000

# Step 4: Generate CBAM report
config = CBAMConfig.from_yaml('config/my_config.yaml')
report = cbam_build_report(
    input_data=df_mapped,
    config=config,
    save_output=True,
    output_path='output/cbam_report.json'
)

# Step 5: Upload report to compliance system
compliance_api_url = "https://compliance.company.com/api/v1/cbam/reports"
upload_response = requests.post(
    compliance_api_url,
    headers=headers,
    json=report.to_dict()
)
upload_response.raise_for_status()

print(f"✓ Report uploaded to compliance system")
print(f"  Report ID: {upload_response.json()['report_id']}")
```

---

### Database Integration

**Store reports in PostgreSQL:**

```python
import psycopg2
import json
from sdk.cbam_sdk import cbam_build_report, CBAMConfig
from datetime import datetime

# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="cbam_db",
    user="cbam_user",
    password=os.getenv('DB_PASSWORD')
)

# Create tables (first run only)
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cbam_reports (
            id SERIAL PRIMARY KEY,
            report_id VARCHAR(50) UNIQUE,
            generated_at TIMESTAMP,
            reporting_period VARCHAR(10),
            total_emissions_tco2 DECIMAL(12, 2),
            total_shipments INTEGER,
            complex_goods_percentage DECIMAL(5, 2),
            report_data JSONB,
            provenance_data JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cbam_shipments (
            id SERIAL PRIMARY KEY,
            report_id VARCHAR(50) REFERENCES cbam_reports(report_id),
            cn_code VARCHAR(8),
            country_of_origin VARCHAR(2),
            quantity_tons DECIMAL(12, 3),
            embedded_emissions_tco2 DECIMAL(12, 3),
            import_date DATE,
            supplier_id VARCHAR(50),
            invoice_number VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    conn.commit()

# Generate report
config = CBAMConfig.from_yaml('config/my_config.yaml')
report = cbam_build_report('shipments.csv', config)

# Insert report
with conn.cursor() as cur:
    cur.execute("""
        INSERT INTO cbam_reports (
            report_id, generated_at, reporting_period,
            total_emissions_tco2, total_shipments,
            complex_goods_percentage, report_data, provenance_data
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        report.report_id,
        datetime.fromisoformat(report.generated_at.replace('Z', '+00:00')),
        "2025-Q3",
        report.total_emissions_tco2,
        report.total_shipments,
        report.complex_goods_percentage,
        json.dumps(report.to_dict()),
        json.dumps(report.provenance)
    ))

    # Insert individual shipments
    for good in report.detailed_goods:
        cur.execute("""
            INSERT INTO cbam_shipments (
                report_id, cn_code, country_of_origin,
                quantity_tons, embedded_emissions_tco2,
                import_date, supplier_id, invoice_number
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            report.report_id,
            good['cn_code'],
            good['country_of_origin'],
            good['quantity_tons'],
            good['embedded_emissions_tco2'],
            good['import_date'],
            good.get('supplier_id'),
            good.get('invoice_number')
        ))

    conn.commit()

print(f"✓ Report saved to database")
print(f"  Report ID: {report.report_id}")
print(f"  Shipments: {len(report.detailed_goods)}")

# Query reports
with conn.cursor() as cur:
    cur.execute("""
        SELECT reporting_period, total_emissions_tco2, total_shipments
        FROM cbam_reports
        ORDER BY generated_at DESC
        LIMIT 10
    """)

    print("\nRecent Reports:")
    for row in cur.fetchall():
        period, emissions, shipments = row
        print(f"  {period}: {emissions:,.2f} tCO2 ({shipments:,} shipments)")

conn.close()
```

---

## 10. INTEGRATION PATTERNS

### SAP Integration

```python
"""
SAP ERP Integration for CBAM Reporting
Fetches shipment data from SAP and generates CBAM reports
"""

from pyrfc import Connection
import pandas as pd
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# SAP Connection
sap_conn = Connection(
    user=os.getenv('SAP_USER'),
    passwd=os.getenv('SAP_PASSWORD'),
    ashost='sap.company.com',
    sysnr='00',
    client='100'
)

# Call SAP RFC function to get shipments
result = sap_conn.call(
    'Z_GET_CBAM_SHIPMENTS',  # Custom SAP function
    IV_START_DATE='20250901',
    IV_END_DATE='20250930'
)

shipments = result['ET_SHIPMENTS']
print(f"Retrieved {len(shipments)} shipments from SAP")

# Convert to DataFrame
df = pd.DataFrame(shipments)

# Map SAP fields to CBAM schema
df_mapped = df.rename(columns={
    'MATNR': 'cn_code',  # Material number → CN code
    'LAND1': 'country_of_origin',
    'MENGE': 'quantity_tons',
    'DATUM': 'import_date',
    'LIFNR': 'supplier_id',
    'VBELN': 'invoice_number'
})

# Generate CBAM report
config = CBAMConfig.from_yaml('config/sap_config.yaml')
report = cbam_build_report(df_mapped, config)

# Upload back to SAP (optional)
sap_conn.call(
    'Z_UPLOAD_CBAM_REPORT',
    IV_REPORT_ID=report.report_id,
    IV_EMISSIONS=report.total_emissions_tco2,
    IT_REPORT_DATA=report.to_dict()
)

sap_conn.close()
print("✓ Report uploaded to SAP")
```

---

### Oracle ERP Integration

```python
"""
Oracle ERP Cloud Integration
Uses REST API to fetch and update CBAM data
"""

import requests
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Oracle ERP Cloud credentials
oracle_base_url = "https://your-company.oraclecloud.com"
oracle_user = os.getenv('ORACLE_USER')
oracle_password = os.getenv('ORACLE_PASSWORD')

# Authenticate
auth = (oracle_user, oracle_password)

# Fetch shipments via REST API
response = requests.get(
    f"{oracle_base_url}/fscmRestApi/resources/11.13.18.05/shipments",
    auth=auth,
    params={
        'q': 'CreationDate >= "2025-09-01" AND CreationDate < "2025-10-01"',
        'limit': 10000
    }
)
response.raise_for_status()

shipments = response.json()['items']

# Transform to CBAM format
import pandas as pd
df = pd.DataFrame(shipments)
df_mapped = df.rename(columns={
    'HSCode': 'cn_code',
    'CountryOfOrigin': 'country_of_origin',
    'Weight': 'quantity_tons',
    'ShipmentDate': 'import_date',
    'VendorId': 'supplier_id',
    'InvoiceNumber': 'invoice_number'
})

# Generate report
config = CBAMConfig.from_yaml('config/oracle_config.yaml')
report = cbam_build_report(df_mapped, config)

# Create custom object in Oracle to store report
requests.post(
    f"{oracle_base_url}/fscmRestApi/resources/11.13.18.05/customObjects/CBAMReports",
    auth=auth,
    json={
        'ReportId': report.report_id,
        'TotalEmissions': report.total_emissions_tco2,
        'ReportData': report.to_dict()
    }
)

print("✓ Report stored in Oracle ERP Cloud")
```

---

### Webhook Notifications

```python
"""
Send webhook notifications on report completion
"""

from sdk.cbam_sdk import cbam_build_report, CBAMConfig
import requests

def send_webhook(event_type, data):
    """Send webhook notification"""
    webhook_url = os.getenv('CBAM_WEBHOOK_URL')
    if not webhook_url:
        return

    payload = {
        'event': event_type,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✓ Webhook sent: {event_type}")
    except Exception as e:
        print(f"⚠ Webhook failed: {e}")

# Generate report with webhooks
try:
    config = CBAMConfig.from_yaml('config/my_config.yaml')

    # Notify start
    send_webhook('report.started', {
        'input_file': 'shipments.csv'
    })

    # Generate report
    report = cbam_build_report('shipments.csv', config)

    # Notify completion
    send_webhook('report.completed', {
        'report_id': report.report_id,
        'total_emissions_tco2': report.total_emissions_tco2,
        'total_shipments': report.total_shipments,
        'output_path': 'output/report.json'
    })

except Exception as e:
    # Notify failure
    send_webhook('report.failed', {
        'error': str(e),
        'input_file': 'shipments.csv'
    })
    raise
```

---

## 11. MAINTENANCE & OPERATIONS

### Log Management

**Configure logging:**

```python
# config/logging.yaml

version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/cbam.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/cbam_errors.log
    maxBytes: 10485760
    backupCount: 10

loggers:
  cbam:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

**Use in code:**

```python
import logging
import logging.config
import yaml

# Load logging configuration
with open('config/logging.yaml', 'r') as f:
    log_config = yaml.safe_load(f)
    logging.config.dictConfig(log_config)

logger = logging.getLogger('cbam')

# Use logger
logger.info("Starting CBAM report generation...")
logger.debug(f"Configuration: {config}")

try:
    report = cbam_build_report('shipments.csv', config)
    logger.info(f"Report generated: {report.report_id}")
except Exception as e:
    logger.error(f"Report generation failed: {e}", exc_info=True)
    raise
```

---

### Backup & Recovery

**Automated backup script:**

```bash
#!/bin/bash
# backup_cbam.sh
# Daily backup of CBAM data and reports

BACKUP_DIR="/backups/cbam/$(date +%Y/%m)"
mkdir -p "$BACKUP_DIR"

# Backup input data
echo "Backing up input data..."
tar -czf "$BACKUP_DIR/input_$(date +%Y%m%d).tar.gz" data/input/

# Backup generated reports
echo "Backing up reports..."
tar -czf "$BACKUP_DIR/reports_$(date +%Y%m%d).tar.gz" output/

# Backup configuration
echo "Backing up configuration..."
tar -czf "$BACKUP_DIR/config_$(date +%Y%m%d).tar.gz" config/

# Backup logs
echo "Backing up logs..."
tar -czf "$BACKUP_DIR/logs_$(date +%Y%m%d).tar.gz" logs/

# Backup database (if using PostgreSQL)
if command -v pg_dump &> /dev/null; then
    echo "Backing up database..."
    pg_dump cbam_db | gzip > "$BACKUP_DIR/database_$(date +%Y%m%d).sql.gz"
fi

# Cleanup old backups (keep 90 days)
find /backups/cbam -type f -mtime +90 -delete

# Upload to S3 (optional)
if command -v aws &> /dev/null; then
    echo "Uploading to S3..."
    aws s3 sync "$BACKUP_DIR" s3://my-cbam-backups/$(date +%Y/%m)/ \
        --storage-class STANDARD_IA
fi

echo "Backup complete: $BACKUP_DIR"
```

**Schedule with cron:**

```bash
# crontab -e
# Daily backup at 2 AM
0 2 * * * /path/to/backup_cbam.sh >> /var/log/cbam/backup.log 2>&1
```

---

### Monitoring

**Health check script:**

```python
#!/usr/bin/env python
"""
CBAM Health Check
Monitors system health and sends alerts
"""

import sys
import psutil
from pathlib import Path
from sdk.cbam_sdk import CBAMConfig

def check_disk_space():
    """Check available disk space"""
    usage = psutil.disk_usage('/')
    free_gb = usage.free / (1024**3)

    if free_gb < 1:
        return False, f"Low disk space: {free_gb:.2f} GB free"
    return True, f"Disk space OK: {free_gb:.2f} GB free"

def check_memory():
    """Check available memory"""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)

    if available_gb < 1:
        return False, f"Low memory: {available_gb:.2f} GB available"
    return True, f"Memory OK: {available_gb:.2f} GB available"

def check_configuration():
    """Check configuration file validity"""
    try:
        config = CBAMConfig.from_yaml('config/production_config.yaml')
        return True, "Configuration valid"
    except Exception as e:
        return False, f"Configuration error: {e}"

def check_data_files():
    """Check required data files exist"""
    required_files = [
        'data/cn_codes.json',
        'data/emission_factors.py',
        'rules/cbam_rules.yaml'
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            return False, f"Missing file: {file_path}"

    return True, "All data files present"

def check_import_pipeline():
    """Test import pipeline"""
    try:
        from cbam_pipeline import CBAMPipeline
        return True, "Pipeline import OK"
    except Exception as e:
        return False, f"Pipeline import failed: {e}"

# Run all checks
checks = [
    ("Disk Space", check_disk_space),
    ("Memory", check_memory),
    ("Configuration", check_configuration),
    ("Data Files", check_data_files),
    ("Pipeline Import", check_import_pipeline)
]

all_passed = True
for name, check_func in checks:
    passed, message = check_func()
    status = "✓" if passed else "✗"
    print(f"{status} {name}: {message}")

    if not passed:
        all_passed = False

sys.exit(0 if all_passed else 1)
```

**Monitor with Prometheus:**

```python
# monitoring/prometheus_exporter.py

from prometheus_client import start_http_server, Gauge, Counter
import time
from cbam_pipeline import CBAMPipeline

# Metrics
reports_generated = Counter('cbam_reports_generated_total', 'Total reports generated')
processing_duration = Gauge('cbam_processing_duration_seconds', 'Processing duration')
total_emissions = Gauge('cbam_total_emissions_tco2', 'Total emissions processed')
shipments_processed = Counter('cbam_shipments_processed_total', 'Total shipments processed')

# Start metrics server
start_http_server(8000)

print("Prometheus metrics available at http://localhost:8000")
print("Press Ctrl+C to stop")

# Keep running
while True:
    time.sleep(60)
```

---

## 12. TROUBLESHOOTING

### Common Issues & Solutions

#### Issue 1: Import Error

**Symptom:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas; print(pandas.__version__)"
```

---

#### Issue 2: File Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/cn_codes.json'
```

**Solution:**
```bash
# Check current working directory
pwd

# Ensure running from project root
cd /path/to/GL-CBAM-APP/CBAM-Importer-Copilot

# Verify file exists
ls -la data/cn_codes.json

# If missing, re-copy from repository
```

---

#### Issue 3: Permission Denied

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'output/report.json'
```

**Solution:**
```bash
# Check permissions
ls -la output/

# Fix permissions
chmod 755 output/
chmod 644 output/*.json

# Or run with appropriate user
sudo -u cbam_user gl cbam report ...
```

---

#### Issue 4: Out of Memory

**Symptom:**
```
MemoryError: Unable to allocate array with shape (1000000,) and data type object
```

**Solution:**

**Option 1: Increase memory limit**
```bash
# Increase Python memory limit
export PYTHONMAXMEMORY=8G
```

**Option 2: Process in chunks**
```python
# Process large files in chunks
import pandas as pd

chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    process_chunk(chunk)
```

**Option 3: Enable performance mode**
```yaml
# config/my_config.yaml
processing:
  performance_mode: true
  chunk_size: 5000
```

---

#### Issue 5: Validation Errors

**Symptom:**
```
Validation failed: Missing CN code (E001) in 50 records
```

**Solution:**

**Step 1: Export validation errors**
```bash
gl cbam validate \
  --input shipments.csv \
  --config config/my_config.yaml \
  --output validation_errors.json \
  --detailed
```

**Step 2: Review errors**
```bash
cat validation_errors.json | jq '.errors[] | {row, field, message}'
```

**Step 3: Fix in bulk (Python)**
```python
import pandas as pd

# Load data
df = pd.read_csv('shipments.csv')

# Fill missing CN codes (if pattern exists)
df.loc[df['cn_code'].isna(), 'cn_code'] = '00000000'  # or derive from product

# Save corrected data
df.to_csv('shipments_fixed.csv', index=False)
```

---

## 13. QUICK REFERENCE

### Essential Commands

```bash
# INSTALLATION
python -m venv cbam-env
source cbam-env/bin/activate  # Windows: cbam-env\Scripts\activate
pip install -r requirements.txt

# CONFIGURATION
gl cbam config init --output config/my_config.yaml
gl cbam config validate --config config/my_config.yaml

# VALIDATION
gl cbam validate --input shipments.csv --config config/my_config.yaml

# REPORT GENERATION
gl cbam report \
  --input shipments.csv \
  --config config/my_config.yaml \
  --output output/report.json \
  --provenance

# TESTING
python -m pytest tests/ -v

# HELP
gl cbam --help
gl cbam report --help
gl cbam config --help
gl cbam validate --help
```

---

### SDK Quick Reference

```python
# Import
from sdk.cbam_sdk import cbam_build_report, CBAMConfig

# Load config
config = CBAMConfig.from_yaml('config/my_config.yaml')

# Generate report (file input)
report = cbam_build_report('shipments.csv', config)

# Generate report (DataFrame input)
import pandas as pd
df = pd.read_csv('shipments.csv')
report = cbam_build_report(df, config)

# Access results
print(report.total_emissions_tco2)
print(report.total_shipments)
print(report.complex_goods_percentage)

# Export
report.save('output/report.json')
report.to_excel('output/report.xlsx')
df = report.to_dataframe()
```

---

### Configuration Quick Reference

```yaml
# Minimal configuration
importer:
  name: "Company Name"
  eori_number: "NL123456789"
  country: "NL"

data_sources:
  cn_codes_path: "data/cn_codes.json"
  cbam_rules_path: "rules/cbam_rules.yaml"
  emission_factors_path: "data/emission_factors.py"

processing:
  enable_provenance: true
```

---

### Error Codes

| Code | Severity | Description |
|------|----------|-------------|
| E001 | ERROR | Missing required field (cn_code, country, quantity, date) |
| E002 | ERROR | Invalid data type |
| E003 | ERROR | Invalid country code (not ISO 3166-1 alpha-2) |
| E004 | ERROR | Invalid CN code format (must be 8 digits) |
| E005 | ERROR | Quantity out of range |
| E006 | ERROR | Invalid date format |
| E007 | ERROR | Future import date |
| E008 | ERROR | CN code not in CBAM scope |
| E009 | ERROR | Duplicate record (same invoice number) |
| E010 | ERROR | Invalid supplier reference |
| W001 | WARNING | Missing optional field |
| W002 | WARNING | Missing supplier ID (will use defaults) |
| W003 | WARNING | Unusual quantity (very high/low) |
| W004 | WARNING | Old import date (>1 year) |
| W005 | WARNING | Complex good (requires additional documentation) |

---

### Performance Benchmarks

| Operation | Dataset Size | Expected Time | Throughput |
|-----------|--------------|---------------|------------|
| Validation | 1,000 records | <1s | 1000+/s |
| Validation | 10,000 records | <5s | 2000+/s |
| Calculation | 1,000 records | <2s | 500+/s |
| Calculation | 10,000 records | <30s | 333+/s |
| Full Pipeline | 1,000 records | <3s | 333+/s |
| Full Pipeline | 10,000 records | <30s | 333+/s |

---

### File Locations

```
GL-CBAM-APP/
└── CBAM-Importer-Copilot/
    ├── agents/                  # AI agent implementations
    ├── cli/                     # CLI commands
    ├── config/                  # Configuration files
    ├── data/                    # Data files
    │   ├── cn_codes.json       # CN code mappings
    │   ├── emission_factors.py # Emission factors database
    │   └── input/              # Input shipments (user-provided)
    ├── docs/                    # Documentation
    ├── examples/                # Example files
    │   ├── demo_shipments.csv  # Sample data
    │   └── quick_start_*.py    # Quick start examples
    ├── logs/                    # Application logs
    ├── output/                  # Generated reports
    ├── provenance/              # Provenance utilities
    ├── rules/                   # CBAM validation rules
    ├── schemas/                 # JSON schemas
    ├── scripts/                 # Utility scripts
    ├── sdk/                     # Python SDK
    ├── tests/                   # Test suite
    ├── cbam_pipeline.py         # Main pipeline
    ├── requirements.txt         # Python dependencies
    ├── pack.yaml                # GreenLang pack definition
    └── README.md                # Project README
```

---

### Support Resources

- **User Guide:** `docs/USER_GUIDE.md`
- **API Reference:** `docs/API_REFERENCE.md`
- **Compliance Guide:** `docs/COMPLIANCE_GUIDE.md`
- **Deployment Guide:** `docs/DEPLOYMENT_GUIDE.md`
- **Troubleshooting:** `docs/TROUBLESHOOTING.md`
- **This Manual:** `docs/OPERATIONS_MANUAL.md`

---

### Version Information

**CBAM Importer Copilot:** v1.0.0
**GreenLang Platform:** v1.0+
**Python:** 3.9+
**Last Updated:** 2025-10-15

---

**END OF OPERATIONS MANUAL**

*For questions or support, refer to the documentation or create an issue in the project repository.*
