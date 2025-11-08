# GL-VCCI CLI - Quick Reference Guide

## Installation & Setup

```bash
# Navigate to platform directory
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform

# Install dependencies
pip install typer rich

# Test CLI
python -m cli.main --help
```

## Command Overview

| Command | Purpose | Example |
|---------|---------|---------|
| `vcci status` | Check platform health | `vcci status --detailed` |
| `vcci intake` | Ingest data files | `vcci intake file --file data.csv` |
| `vcci engage` | Manage campaigns | `vcci engage create --name "Q1 2026"` |
| `vcci pipeline` | Run full workflow | `vcci pipeline run --input data/ --output results/` |
| `vcci calculate` | Calculate emissions | `vcci calculate --category 1 --input data.csv` |
| `vcci analyze` | Analyze results | `vcci analyze --input results.json --type hotspot` |
| `vcci report` | Generate reports | `vcci report --input results.json --format ghg-protocol` |

---

## 1. INTAKE Commands

### Single File Ingestion
```bash
# Basic ingestion
vcci intake file --file suppliers.csv

# Specify format explicitly
vcci intake file --file data.json --format json

# Custom entity type
vcci intake file --file products.xlsx --entity-type product

# With output
vcci intake file --file data.csv --output results.json --verbose
```

### Batch Processing
```bash
# Process entire directory
vcci intake batch --directory data/

# Filter by pattern
vcci intake batch --directory data/ --pattern "*.csv"

# Custom entity type
vcci intake batch --directory suppliers/ --entity-type supplier
```

### Status
```bash
# Overall status
vcci intake status

# Specific batch
vcci intake status --batch-id BATCH-20251108-ABC123
```

**Supported Formats:** CSV, JSON, Excel (.xlsx, .xls), XML, PDF

---

## 2. ENGAGE Commands

### List Campaigns
```bash
# All campaigns
vcci engage list

# Filter by status
vcci engage list --status active

# Limit results
vcci engage list --limit 20
```

### Create Campaign
```bash
# Standard campaign
vcci engage create --name "Q1 2026 Data Collection" --template standard

# Custom duration
vcci engage create --name "Urgent Request" --template urgent --duration 30

# From supplier file
vcci engage create --name "Top 100" --suppliers suppliers.csv

# Dry run mode
vcci engage create --name "Test" --template standard --dry-run
```

### Send Emails
```bash
# Send all emails
vcci engage send --campaign-id CAMP-ABC123

# Dry run preview
vcci engage send --campaign-id CAMP-ABC123 --dry-run

# Limited test
vcci engage send --campaign-id CAMP-ABC123 --limit 10
```

### Campaign Status
```bash
# Quick status
vcci engage status --campaign-id CAMP-ABC123

# Detailed analytics
vcci engage status --campaign-id CAMP-ABC123 --detailed
```

### Leaderboard
```bash
# Top 10 suppliers
vcci engage leaderboard --campaign-id CAMP-ABC123

# Top 20
vcci engage leaderboard --campaign-id CAMP-ABC123 --top 20
```

---

## 3. PIPELINE Commands

### Run Pipeline
```bash
# All categories
vcci pipeline run --input data/ --output results/ --categories all

# Specific categories
vcci pipeline run --input data.csv --output results/ --categories 1,4,15

# Custom format
vcci pipeline run --input data/ --output results/ --format cdp

# Performance options
vcci pipeline run --input data/ --output results/ --no-llm --no-mc

# Full configuration
vcci pipeline run \
  --input data/suppliers.csv \
  --output results/q4_2025/ \
  --categories 1,2,4,15 \
  --format ghg-protocol \
  --llm \
  --mc
```

### Pipeline Status
```bash
# Recent runs
vcci pipeline status

# Specific run
vcci pipeline status --run-id RUN-20251108120000
```

**Pipeline Stages:**
1. Intake (data ingestion)
2. Calculate (emissions calculation)
3. Analyze (hotspot analysis)
4. Report (automated reporting)

---

## 4. CALCULATE Commands

```bash
# Basic calculation
vcci calculate --category 1 --input procurement.csv

# With output
vcci calculate --category 15 --input investments.json --output results.json

# Disable features
vcci calculate --category 4 --input transport.csv --no-llm --no-mc

# All options
vcci calculate \
  --category 1 \
  --input data.csv \
  --output results.json \
  --llm \
  --mc
```

**Categories:** 1-15 (all Scope 3 categories)

---

## 5. ANALYZE Commands

```bash
# Hotspot analysis
vcci analyze --input scope3_results.json --type hotspot

# Pareto analysis
vcci analyze --input scope3_results.json --type pareto

# Trend analysis
vcci analyze --input scope3_results.json --type trend
```

**Analysis Types:** hotspot, pareto, trend

---

## 6. REPORT Commands

```bash
# GHG Protocol report
vcci report --input scope3.json --format ghg-protocol --output report.pdf

# CDP report
vcci report --input scope3.json --format cdp

# TCFD report
vcci report --input scope3.json --format tcfd --output tcfd_report.pdf

# CSRD report
vcci report --input scope3.json --format csrd
```

**Report Formats:** ghg-protocol, cdp, tcfd, csrd

---

## 7. UTILITY Commands

### Status
```bash
# Simple status
vcci status

# Detailed status
vcci status --detailed
```

### Categories
```bash
# List all categories
vcci categories

# Summary view
vcci categories --summary
```

### Config
```bash
# Show configuration
vcci config --show

# Set value
vcci config --set llm.provider --value openai
```

### Info
```bash
# Platform information
vcci info
```

### Version
```bash
# Show version
vcci --version
```

### Help
```bash
# Main help
vcci --help

# Command help
vcci intake --help
vcci engage --help
vcci pipeline --help
```

---

## Common Workflows

### Workflow 1: Quick Pipeline
```bash
vcci pipeline run --input data.csv --output results/ --categories all
```

### Workflow 2: Manual Steps
```bash
# 1. Ingest
vcci intake file --file suppliers.csv

# 2. Calculate
vcci calculate --category 1 --input suppliers.csv --output cat1.json

# 3. Analyze
vcci analyze --input cat1.json --type hotspot

# 4. Report
vcci report --input cat1.json --format ghg-protocol --output report.pdf
```

### Workflow 3: Batch Processing
```bash
# 1. Batch ingest
vcci intake batch --directory data/q4_2025/

# 2. Run pipeline
vcci pipeline run --input data/q4_2025/ --output results/q4_2025/ --categories all

# 3. Check results
vcci pipeline status
```

### Workflow 4: Supplier Engagement
```bash
# 1. Create campaign
vcci engage create --name "Q1 2026" --template standard --duration 90

# 2. Preview emails
vcci engage send --campaign-id CAMP-001 --dry-run --limit 5

# 3. Send emails
vcci engage send --campaign-id CAMP-001

# 4. Monitor
vcci engage status --campaign-id CAMP-001 --detailed

# 5. Check leaderboard
vcci engage leaderboard --campaign-id CAMP-001
```

---

## Global Options

```bash
# Verbose output
vcci <command> --verbose

# JSON output
vcci <command> --json

# Custom config
vcci --config my_config.yaml <command>

# Help for any command
vcci <command> --help
```

---

## Output Locations

```
results/
├── scope3_report_RUN-*.json    # Pipeline reports
├── cat1.json                   # Category calculations
├── report.pdf                  # Generated reports
└── batch_results/              # Batch processing results
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (agent not available, file not found, etc.) |
| 2 | Invalid arguments |

---

## Tips & Tricks

### 1. Use Auto-Detection
```bash
# No need to specify format
vcci intake file --file data.csv    # Auto-detects CSV
vcci intake file --file data.xlsx   # Auto-detects Excel
```

### 2. Dry Run First
```bash
# Always test before sending emails
vcci engage send --campaign-id CAMP-001 --dry-run
```

### 3. Save Intermediate Results
```bash
# Save outputs for debugging
vcci calculate --category 1 --input data.csv --output step1.json
vcci analyze --input step1.json --type hotspot
```

### 4. Use Verbose for Debugging
```bash
# Get detailed error information
vcci intake file --file data.csv --verbose
```

### 5. Check Status Often
```bash
# Monitor long-running operations
vcci pipeline status
vcci engage status --campaign-id CAMP-001
```

### 6. Chain Commands
```bash
# Use && for sequential operations
vcci intake file --file data.csv && \
vcci calculate --category 1 --input data.csv --output results.json && \
vcci report --input results.json --format ghg-protocol
```

---

## Troubleshooting

### Agent Not Available
```bash
# Error: "SupplierEngagementAgent not available"
# Solution: Install required packages
pip install -r requirements.txt
```

### Format Not Detected
```bash
# Error: "Could not detect format"
# Solution: Specify format explicitly
vcci intake file --file data.txt --format csv
```

### Campaign Not Found
```bash
# Error: "Campaign not found"
# Solution: List campaigns first
vcci engage list
```

### Invalid Category
```bash
# Error: "Invalid category"
# Solution: Use 1-15 or 'all'
vcci pipeline run --input data/ --output results/ --categories 1,4,15
```

---

## Performance Tips

### Disable LLM for Speed
```bash
# Faster but less intelligent
vcci calculate --category 1 --input data.csv --no-llm
vcci pipeline run --input data/ --output results/ --no-llm
```

### Disable Monte Carlo
```bash
# Faster but no uncertainty quantification
vcci calculate --category 1 --input data.csv --no-mc
```

### Limit Email Batches
```bash
# Send in smaller batches
vcci engage send --campaign-id CAMP-001 --limit 50
```

### Batch Files Together
```bash
# More efficient than individual files
vcci intake batch --directory data/
```

---

## Getting Help

1. **Command Help:** `vcci <command> --help`
2. **Platform Info:** `vcci info`
3. **Documentation:** See CLI_COMMANDS_SUMMARY.md
4. **Examples:** Built into command help text

---

**Quick Start:** `vcci info` → `vcci status` → `vcci pipeline run --input data/ --output results/ --categories all`
