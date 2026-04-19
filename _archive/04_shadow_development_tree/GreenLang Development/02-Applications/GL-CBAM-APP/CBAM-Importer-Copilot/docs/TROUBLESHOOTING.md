# CBAM Importer Copilot - Troubleshooting Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**Target Audience:** All users, support teams

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Data Validation Errors](#data-validation-errors)
4. [Performance Problems](#performance-problems)
5. [Configuration Issues](#configuration-issues)
6. [Output & Reporting Problems](#output--reporting-problems)
7. [Integration Issues](#integration-issues)
8. [Common Error Messages](#common-error-messages)
9. [Getting Help](#getting-help)

---

## Quick Diagnostics

### Run Self-Diagnostic

Before troubleshooting, run the built-in diagnostic:

```bash
gl cbam doctor

# Expected output:
# ✓ Python version: 3.11.5
# ✓ Dependencies installed: 12/12
# ✓ Data files accessible
# ✓ Configuration valid
# ✓ Permissions correct
# ✓ All systems operational
```

### Check Version

```bash
gl cbam --version

# Expected: cbam-importer-copilot 1.0.0
# If different or error: reinstall application
```

### Verify Installation

```bash
# Test with example data
gl cbam report examples/demo_shipments.csv \
  --importer-name "Test" \
  --importer-country "NL" \
  --importer-eori "NL000000000000"

# Expected: Report generated successfully
# Check: output/cbam_report.json exists
```

---

## Installation Issues

### Problem: `gl cbam` command not found

**Symptoms:**
```bash
$ gl cbam --version
bash: gl: command not found
```

**Cause:** Application not installed or not in PATH

**Solution 1: Activate virtual environment**
```bash
# Navigate to application directory
cd /path/to/cbam-importer-copilot

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verify
gl cbam --version
```

**Solution 2: Reinstall application**
```bash
cd /path/to/cbam-importer-copilot
pip install -e .

# Verify
which gl  # Should show path to executable
```

**Solution 3: Add to PATH (system-wide)**
```bash
# Find GL executable
find ~ -name "gl" -type f 2>/dev/null

# Add to PATH in ~/.bashrc
echo 'export PATH="$PATH:/path/to/venv/bin"' >> ~/.bashrc
source ~/.bashrc
```

---

### Problem: `ImportError: No module named 'pandas'`

**Symptoms:**
```bash
$ gl cbam report ...
ImportError: No module named 'pandas'
```

**Cause:** Dependencies not installed

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify pandas installed
python -c "import pandas; print(pandas.__version__)"
# Expected: 2.1.0 or higher
```

**If still failing:**
```bash
# Force reinstall
pip install --force-reinstall pandas

# Or upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Problem: `ModuleNotFoundError: No module named 'sdk'`

**Symptoms:**
```bash
$ python my_script.py
ModuleNotFoundError: No module named 'sdk'
```

**Cause:** Python path not configured

**Solution 1: Install as package**
```bash
cd /path/to/cbam-importer-copilot
pip install -e .
```

**Solution 2: Add to PYTHONPATH**
```python
# At top of your script
import sys
from pathlib import Path

# Add application directory to path
app_dir = Path(__file__).parent.parent / "cbam-importer-copilot"
sys.path.insert(0, str(app_dir))

# Now can import
from sdk.cbam_sdk import cbam_build_report
```

---

### Problem: Permission denied errors on Windows

**Symptoms:**
```
PermissionError: [WinError 5] Access is denied
```

**Cause:** Insufficient permissions or antivirus blocking

**Solution 1: Run as Administrator**
- Right-click Command Prompt
- Select "Run as Administrator"
- Navigate to directory and retry

**Solution 2: Exclude from antivirus**
- Add `cbam-importer-copilot` directory to antivirus exclusions
- Particularly important for Windows Defender

**Solution 3: Use user-specific installation**
```bash
# Install in user directory
pip install --user -r requirements.txt
```

---

## Data Validation Errors

### Problem: `E001: Missing CN code`

**Error message:**
```
E001: Missing CN code in row 5
Field: cn_code
Severity: error
```

**Cause:** CSV file missing CN code column or empty values

**Solution:**
```bash
# Check CSV structure
head -1 data/shipments.csv
# Expected: cn_code,country_of_origin,quantity_tons,import_date,...

# Find rows with missing CN codes
awk -F',' 'NR>1 && $1=="" {print "Row", NR}' data/shipments.csv

# Fix: Add CN codes to rows or remove invalid rows
```

**Prevention:**
```python
# Validate before processing
from sdk.cbam_sdk import cbam_validate_shipments

validation = cbam_validate_shipments(
    input_file="data/shipments.csv",
    importer_country="NL"
)

if validation['metadata']['error_count'] > 0:
    print("Errors found:")
    for error in validation['errors']:
        print(f"  Row {error['record_index']}: {error['message']}")
```

---

### Problem: `E003: Unknown CN code`

**Error message:**
```
E003: Unknown CN code '12345678' in row 10
CN code '12345678' not found in CBAM Annex I
```

**Cause:** CN code not covered by CBAM regulations

**Solution 1: Verify CN code is correct**
```bash
# Check official CBAM Annex I
# Only these product groups are covered:
# - 25: Cement
# - 27: Electricity
# - 28-31: Fertilizers
# - 72-73: Iron and Steel
# - 76: Aluminum

# Correct the CN code in input file
```

**Solution 2: Update CN codes database (if valid but missing)**
```json
// data/cn_codes.json
{
  "12345678": {
    "description": "New product description",
    "product_group": "iron_steel",
    "cbam_covered": true,
    "default_emission_factor_tco2_per_ton": 1.5
  }
}
```

---

### Problem: `E007: Invalid quantity (negative/zero)`

**Error message:**
```
E007: Invalid quantity in row 15
Value: -5.5
Severity: error
Message: Quantity must be positive number
```

**Cause:** Negative or zero quantity in input data

**Solution:**
```bash
# Find negative quantities
awk -F',' 'NR>1 && $3<=0 {print "Row", NR, "Quantity:", $3}' data/shipments.csv

# Fix in input file (correct data entry error)
# OR remove invalid rows
awk -F',' 'NR==1 || $3>0' data/shipments.csv > data/shipments_clean.csv
```

---

### Problem: `W001: Missing supplier data`

**Warning message:**
```
W001: Missing supplier data in row 20
Field: supplier_id
Impact: Will use default emission factors
```

**Cause:** No supplier_id provided (optional field)

**Impact:** Not an error, but reduces accuracy

**Solution (if supplier actuals available):**

```yaml
# examples/demo_suppliers.yaml
suppliers:
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
        verification_date: "2025-09-01"
```

```bash
# Use supplier file in report
gl cbam report data/shipments.csv \
  --suppliers examples/demo_suppliers.yaml \
  --importer-name "Acme Steel BV" \
  --importer-country "NL" \
  --importer-eori "NL123456789012"
```

---

## Performance Problems

### Problem: Processing very slow (>10 minutes for 10K shipments)

**Symptoms:**
- Processing takes much longer than expected
- High CPU usage
- System becomes unresponsive

**Diagnosis:**
```bash
# Check system resources
top  # Linux/Mac
taskmgr  # Windows

# Look for:
# - CPU usage >90%
# - Memory usage >80%
# - Disk I/O wait time
```

**Solution 1: Use chunked processing**
```python
# For large files (>50K shipments)
import pandas as pd
from sdk.cbam_sdk import cbam_build_report

# Read in chunks
chunk_size = 10000
chunks = pd.read_csv("large_file.csv", chunksize=chunk_size)

# Process each chunk
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}...")
    report = cbam_build_report(
        input_dataframe=chunk,
        config=config,
        output_path=f"output/chunk_{i}.json",
        save_output=True
    )
```

**Solution 2: Optimize configuration**
```yaml
# config/performance.yaml
performance:
  chunk_size: 5000              # Smaller chunks
  parallel_workers: 4           # Match CPU cores
  memory_limit_mb: 8192         # Set appropriate limit
  use_chunked_reading: true     # Enable chunking
```

**Solution 3: Use CSV instead of Excel**
```bash
# Excel files are slower to process
# Convert to CSV first
python -c "import pandas as pd; pd.read_excel('data.xlsx').to_csv('data.csv', index=False)"

# Then process CSV
gl cbam report data.csv ...
```

---

### Problem: Out of memory errors

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Cause:** Dataset too large for available RAM

**Solution 1: Process in chunks (recommended)**
```python
# Use chunked reading
import pandas as pd
from sdk.cbam_sdk import cbam_build_report

chunks = pd.read_csv("large_file.csv", chunksize=5000)
all_reports = []

for chunk in chunks:
    report = cbam_build_report(
        input_dataframe=chunk,
        config=config,
        save_output=False
    )
    all_reports.append(report.raw_report)

# Merge results
final_report = merge_reports(all_reports)
```

**Solution 2: Increase system memory**
```bash
# If using Docker, increase memory limit
docker run -m 8g cbam-copilot ...

# If VM, increase RAM allocation
```

**Solution 3: Use lighter data types**
```python
# Optimize DataFrame dtypes
import pandas as pd

df = pd.read_csv("shipments.csv", dtype={
    'cn_code': 'str',
    'country_of_origin': 'str',
    'quantity_tons': 'float32',  # Instead of float64
    'import_date': 'str'
})
```

---

## Configuration Issues

### Problem: `Configuration file not found`

**Error message:**
```
FileNotFoundError: Configuration file '.cbam.yaml' not found
```

**Cause:** No configuration file in current directory

**Solution 1: Create configuration**
```bash
gl cbam config init

# Creates .cbam.yaml in current directory
```

**Solution 2: Specify config path**
```bash
gl cbam report data/shipments.csv \
  --config /path/to/config.yaml \
  --importer-name "..." \
  ...
```

**Solution 3: Provide all parameters directly**
```bash
# Don't use config file, provide all params
gl cbam report data/shipments.csv \
  --importer-name "Acme Steel BV" \
  --importer-country "NL" \
  --importer-eori "NL123456789012" \
  --declarant-name "John Smith" \
  --declarant-position "Compliance Officer"
```

---

### Problem: `Invalid EORI format`

**Error message:**
```
ValidationError: Invalid EORI format 'NL12345'
Expected format: 2-letter country code + 12 digits
Example: NL123456789012
```

**Cause:** EORI number doesn't match expected format

**Solution:**
```bash
# EORI format: CC + 12 digits
# CC = ISO 3166-1 alpha-2 country code
# Examples:
#   NL123456789012 ✓
#   DE987654321098 ✓
#   FR111222333444 ✓
#   NL12345 ✗ (too short)
#   123456789012 ✗ (missing country)

# Fix in config file or command line
```

---

### Problem: Environment variables not being read

**Symptoms:**
- Set `CBAM_IMPORTER_NAME` but tool doesn't recognize it
- Must provide all parameters manually

**Diagnosis:**
```bash
# Check if environment variable is set
echo $CBAM_IMPORTER_NAME

# If empty, variable not set
```

**Solution 1: Export variables**
```bash
# Use export (not just assignment)
export CBAM_IMPORTER_NAME="Acme Steel BV"  # ✓
CBAM_IMPORTER_NAME="Acme Steel BV"         # ✗ Won't work

# Verify
echo $CBAM_IMPORTER_NAME
```

**Solution 2: Use .env file**
```bash
# Create .env file
cat > .env <<EOF
CBAM_IMPORTER_NAME="Acme Steel BV"
CBAM_IMPORTER_COUNTRY="NL"
CBAM_IMPORTER_EORI="NL123456789012"
EOF

# Load before running
source .env
gl cbam report ...
```

**Solution 3: Use configuration file instead**
```bash
# More reliable than environment variables
gl cbam config init
# Edit .cbam.yaml
gl cbam report --config .cbam.yaml ...
```

---

## Output & Reporting Problems

### Problem: Report generated but JSON is invalid

**Symptoms:**
```bash
cat output/cbam_report.json
# Shows garbled or incomplete JSON
```

**Diagnosis:**
```bash
# Validate JSON
python -m json.tool output/cbam_report.json > /dev/null

# If error: JSON is malformed
```

**Solution 1: Regenerate report**
```bash
# Remove corrupted file
rm output/cbam_report.json

# Regenerate
gl cbam report data/shipments.csv ... --output output/cbam_report.json
```

**Solution 2: Check disk space**
```bash
# May be truncated due to full disk
df -h

# If <10% free, clean up disk
```

---

### Problem: Excel file won't open

**Error message (in Excel):**
```
Excel found unreadable content in 'cbam_report.xlsx'
```

**Cause:** Corrupted Excel file or incompatible format

**Solution 1: Use openpyxl format**
```python
from sdk.cbam_sdk import cbam_build_report

report = cbam_build_report(...)

# Save with explicit engine
report.to_excel("output/cbam_report.xlsx", engine='openpyxl')
```

**Solution 2: Try LibreOffice**
```bash
# Sometimes Excel is picky, LibreOffice more forgiving
libreoffice --calc output/cbam_report.xlsx
```

**Solution 3: Use JSON and convert**
```bash
# Save as JSON first (always works)
report.save("output/cbam_report.json")

# Convert to Excel manually
python -c "
import pandas as pd
import json

with open('output/cbam_report.json') as f:
    report = json.load(f)

df = pd.DataFrame(report['detailed_goods'])
df.to_excel('output/cbam_report.xlsx', index=False)
"
```

---

### Problem: Provenance file missing

**Symptoms:**
- Report generated successfully
- But no `provenance.json` file

**Cause:** Provenance disabled in configuration

**Solution:**
```yaml
# config/.cbam.yaml
output:
  include_provenance: true  # Ensure this is true

# OR via command line
gl cbam report ... --include-provenance
```

---

### Problem: SHA256 hash doesn't match

**Error message:**
```
✗ Provenance validation failed
Error: File hash mismatch
Expected: a1b2c3d4e5f6...
Actual: x9y8z7w6v5u4...
```

**Cause:** Input file has been modified since report was generated

**Solution:**
```bash
# If file should be same:
# - Check if file was accidentally edited
# - Restore from backup

# If file was intentionally changed:
# - Regenerate report with new file
gl cbam report data/shipments_updated.csv ...
```

---

## Integration Issues

### Problem: Pandas DataFrame integration not working

**Symptoms:**
```python
import pandas as pd
from sdk.cbam_sdk import cbam_build_report

df = pd.read_sql(...)
report = cbam_build_report(input_dataframe=df, ...)

# Error: TypeError or ValueError
```

**Solution 1: Check required columns**
```python
# DataFrame must have required columns
required_cols = ['cn_code', 'country_of_origin', 'quantity_tons', 'import_date']

missing = set(required_cols) - set(df.columns)
if missing:
    print(f"Missing columns: {missing}")
    # Add missing columns
```

**Solution 2: Check data types**
```python
# Ensure correct dtypes
df['cn_code'] = df['cn_code'].astype(str)
df['country_of_origin'] = df['country_of_origin'].astype(str)
df['quantity_tons'] = pd.to_numeric(df['quantity_tons'])
df['import_date'] = pd.to_datetime(df['import_date']).dt.strftime('%Y-%m-%d')

# Now process
report = cbam_build_report(input_dataframe=df, ...)
```

---

### Problem: S3 integration fails

**Symptoms:**
```python
import boto3
s3 = boto3.client('s3')
s3.download_file('cbam-input', 'shipments.csv', 'local.csv')

# Error: botocore.exceptions.NoCredentialsError
```

**Cause:** AWS credentials not configured

**Solution 1: Configure AWS CLI**
```bash
aws configure

# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region
# - Output format
```

**Solution 2: Use environment variables**
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="eu-west-1"
```

**Solution 3: Use IAM role (on EC2)**
```bash
# Attach IAM role to EC2 instance with S3 permissions
# No credentials needed in code
```

---

## Common Error Messages

### `FileNotFoundError: [Errno 2] No such file or directory`

**Cause:** Input file path incorrect

**Solution:**
```bash
# Use absolute path
gl cbam report /full/path/to/shipments.csv ...

# OR navigate to directory first
cd /path/to/data
gl cbam report shipments.csv ...

# OR use relative path from current directory
gl cbam report ./data/shipments.csv ...
```

---

### `UnicodeDecodeError: 'utf-8' codec can't decode byte`

**Cause:** Input file not in UTF-8 encoding

**Solution:**
```bash
# Check file encoding
file -i data/shipments.csv
# Example output: data/shipments.csv: text/plain; charset=iso-8859-1

# Convert to UTF-8
iconv -f ISO-8859-1 -t UTF-8 data/shipments.csv > data/shipments_utf8.csv

# On Windows:
# Open in Notepad++
# Encoding → Convert to UTF-8
# Save
```

---

### `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

**Cause:** JSON file is empty or corrupted

**Solution:**
```bash
# Check if file is empty
wc -l output/cbam_report.json

# If 0 lines: regenerate report
gl cbam report data/shipments.csv ... --output output/cbam_report.json

# Check JSON validity
python -m json.tool output/cbam_report.json
```

---

### `PermissionError: [Errno 13] Permission denied`

**Cause:** Insufficient permissions to write output

**Solution:**
```bash
# Check permissions
ls -la output/

# Create output directory if missing
mkdir -p output

# Set permissions
chmod 755 output/

# OR use different output path
gl cbam report ... --output ~/my_reports/cbam_report.json
```

---

## Getting Help

### Self-Service Resources

**1. Documentation:**
- User Guide: `docs/USER_GUIDE.md`
- API Reference: `docs/API_REFERENCE.md`
- Compliance Guide: `docs/COMPLIANCE_GUIDE.md`
- Deployment Guide: `docs/DEPLOYMENT_GUIDE.md`

**2. Examples:**
- CLI examples: `examples/quick_start_cli.sh`
- SDK examples: `examples/quick_start_sdk.py`
- Provenance examples: `examples/provenance_example.py`

**3. Run built-in help:**
```bash
gl cbam --help
gl cbam report --help
gl cbam config --help
```

### Diagnostic Information

**When reporting issues, include:**

```bash
# 1. Version information
gl cbam --version

# 2. System information
uname -a  # Linux/Mac
systeminfo  # Windows

# 3. Python information
python --version
pip list | grep -E "(pandas|pydantic|jsonschema)"

# 4. Run diagnostic
gl cbam doctor > diagnostic.txt

# 5. Error output
gl cbam report ... 2>&1 | tee error.log
```

### Community Support

**GitHub Issues:**
- Bug reports: https://github.com/greenlang/cbam-copilot/issues
- Feature requests: https://github.com/greenlang/cbam-copilot/discussions

**Email Support:**
- Technical issues: cbam-support@greenlang.io
- Compliance questions: cbam-compliance@greenlang.io

### Professional Services

**For enterprise support:**
- Custom deployment assistance
- Integration consulting
- Training sessions
- Priority bug fixes

Contact: enterprise@greenlang.io

---

## Debug Mode

### Enable Verbose Logging

```bash
# Set log level to DEBUG
export CBAM_LOG_LEVEL=DEBUG

# Run with verbose flag
gl cbam report ... --verbose

# Check logs
tail -f /home/cbam/logs/cbam.log
```

### Python Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from sdk.cbam_sdk import cbam_build_report

# Now all internal operations logged
report = cbam_build_report(...)
```

### Interactive Debugging

```python
# Add breakpoint
import pdb

from sdk.cbam_sdk import cbam_build_report

pdb.set_trace()  # Execution pauses here
report = cbam_build_report(...)

# In pdb prompt:
# - n: next line
# - c: continue
# - p variable: print variable
# - q: quit
```

---

## FAQ - Quick Answers

**Q: Why is processing slow?**
A: Use chunked processing for large files, convert Excel to CSV, check system resources

**Q: How to fix "command not found"?**
A: Activate virtual environment, reinstall package, or add to PATH

**Q: What if validation fails?**
A: Run `gl cbam validate` first to identify specific errors, fix data, then regenerate

**Q: Can I reprocess same file?**
A: Yes, but output will be identical (bit-perfect reproducibility)

**Q: How to update tool?**
A: `git pull && pip install -r requirements.txt --upgrade`

**Q: Is my data secure?**
A: Yes, all processing is local, no data sent externally

---

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**License:** MIT

---

*For additional help, see `docs/USER_GUIDE.md` or contact support@greenlang.io*
