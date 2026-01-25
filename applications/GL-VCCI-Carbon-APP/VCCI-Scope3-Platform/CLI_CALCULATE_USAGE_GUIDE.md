# VCCI CLI Calculate Command - Usage Guide

## Overview

The `vcci calculate` command has been upgraded from demo/simulation mode to use the **real Scope3CalculatorAgent**. It now performs actual emissions calculations using the production-ready calculator engine.

## What Changed

### Before (Demo Mode)
```python
# Showed fake/simulated results
console.print("[green]Simulated: 1,234.56 tCO2e[/green]")
```

### After (Production Mode)
```python
# Real calculations using Scope3CalculatorAgent
factor_broker = FactorBroker()
agent = Scope3CalculatorAgent(factor_broker=factor_broker)
result = await agent.calculate_by_category(category, data)
# Display actual emissions with real data quality, tier info, and uncertainty
```

## Key Features

### 1. Real Calculator Integration
- **Scope3CalculatorAgent**: Production-ready agent supporting all 15 categories
- **FactorBroker**: Multi-source emission factor resolution (ecoinvent, DESNZ, EPA, proxy)
- **3-Tier Waterfall** (Category 1): Supplier-specific → Product averages → Spend-based
- **ISO 14083 Compliance** (Category 4): Standardized transport emissions
- **PCAF Standard** (Category 15): Financed emissions methodology

### 2. Multi-Format Data Loading
- **JSON**: Single object or array of objects
- **CSV**: Automatic type conversion and header parsing
- **Excel**: Support for .xlsx and .xls files (requires openpyxl)

### 3. Batch Processing
- Process multiple records in parallel
- Progress tracking with Rich progress bars
- Detailed success/failure reporting
- Aggregated emissions totals

### 4. Rich Output Formatting
- Beautiful terminal output using Rich library
- Color-coded results (green for success, yellow for warnings, red for errors)
- Tables for batch results
- Detailed provenance and quality information

### 5. Comprehensive Error Handling
- Input validation errors
- Calculator-specific errors
- Graceful fallback for missing dependencies
- Verbose mode for debugging

## Command Syntax

```bash
vcci calculate --category <1-15> --input <file> [OPTIONS]
```

## Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| --category | -cat | int | Required | Scope 3 category (1-15) |
| --input | -i | Path | Required | Input data file (JSON/CSV/Excel) |
| --output | -o | Path | None | Output file path (JSON) |
| --llm/--no-llm | | bool | True | Enable/disable LLM intelligence |
| --mc/--no-mc | | bool | True | Enable/disable Monte Carlo uncertainty |
| --batch | | bool | False | Process file as batch (multiple records) |
| --tenant | -t | string | "cli-user" | Tenant identifier |
| --verbose | -v | bool | False | Show detailed processing information |

## Usage Examples

### Category 1: Purchased Goods & Services

**Single Record (Tier 1 - Supplier PCF)**
```bash
vcci calculate \
  --category 1 \
  --input examples/sample_category1_single.json

# Output:
# ✓ Calculation complete
# Category: 1
# Total Emissions: 2.50 tCO2e
# Data Tier: Tier 1 (Supplier-specific)
# Data Quality: 90.0%
# Uncertainty: ±15.0%
```

**Batch Processing**
```bash
vcci calculate \
  --category 1 \
  --input examples/sample_category1_batch.csv \
  --batch \
  --output results.json

# Processes all rows in CSV
# Shows summary table with totals
# Saves detailed results to JSON
```

**With Verbose Output**
```bash
vcci calculate \
  --category 1 \
  --input examples/sample_category1_single.json \
  --verbose

# Shows:
# - Loaded records count
# - Agent initialization details
# - Detailed breakdown table
# - Factor source and methodology
# - Full provenance chain
```

### Category 4: Upstream Transportation & Distribution

**ISO 14083 Compliant Transport**
```bash
vcci calculate \
  --category 4 \
  --input examples/sample_category4_transport.json

# Calculates using ISO 14083 standard:
# emissions = distance × weight × emission_factor
```

**Fast Mode (No Monte Carlo)**
```bash
vcci calculate \
  --category 4 \
  --input transport_data.csv \
  --batch \
  --no-mc

# Disables Monte Carlo uncertainty propagation
# Significantly faster for large batches
```

### Category 6: Business Travel

**Flights + Hotels + Ground Transport**
```bash
vcci calculate \
  --category 6 \
  --input examples/sample_category6_travel.json

# Calculates:
# - Flight emissions (with radiative forcing)
# - Hotel stay emissions
# - Ground transport emissions
# - Aggregated total
```

### Category 15: Investments (PCAF)

```bash
vcci calculate \
  --category 15 \
  --input investments.json \
  --output pcaf_results.json

# Uses PCAF Standard for financed emissions
# Supports all asset classes
# Calculates attribution factors
```

## Input Data Formats

### JSON Format

**Single Record:**
```json
{
  "product_name": "Steel Rebar",
  "quantity": 1000,
  "quantity_unit": "kg",
  "region": "US",
  "supplier_pcf": 2.5,
  "supplier_pcf_uncertainty": 0.15
}
```

**Multiple Records:**
```json
[
  {
    "product_name": "Steel Rebar",
    "quantity": 1000,
    "quantity_unit": "kg",
    "region": "US",
    "supplier_pcf": 2.5
  },
  {
    "product_name": "Aluminum Sheets",
    "quantity": 500,
    "quantity_unit": "kg",
    "region": "US",
    "supplier_pcf": 8.2
  }
]
```

### CSV Format

```csv
product_name,quantity,quantity_unit,region,supplier_pcf,supplier_pcf_uncertainty
Steel Rebar,1000,kg,US,2.5,0.15
Aluminum Sheets,500,kg,US,8.2,0.20
Concrete Mix,2000,kg,US,0.15,0.10
```

**Features:**
- Automatic header detection
- Numeric type conversion (int/float)
- Empty value handling

### Excel Format

Supported: `.xlsx`, `.xls`

**Requirements:**
```bash
pip install openpyxl
```

**Format:**
- First row = headers
- Subsequent rows = data
- Active sheet is used

## Output Examples

### Single Record Output

```
╭─────────────────── Category 1 Results ────────────────────╮
│ ✓ Calculation complete                                    │
│                                                            │
│ Category: 1                                                │
│ Input: sample_category1_single.json                        │
│ Monte Carlo: Enabled                                       │
│                                                            │
│ Results:                                                   │
│ Total Emissions: 2.50 tCO2e                               │
│ Data Tier: Tier 1 (Supplier-specific)                    │
│ Data Quality: 90.0%                                       │
│ Uncertainty: ±15.0%                                       │
╰────────────────────────────────────────────────────────────╯
```

### Batch Output

```
                  Category 1 Batch Results
╭──────────────────┬─────────────────────────────────────╮
│ Metric           │                               Value │
├──────────────────┼─────────────────────────────────────┤
│ Total Records    │                                   5 │
│ Successful       │                                   5 │
│ Failed           │                                   0 │
│ Total Emissions  │                            18.75 tCO2e │
│ Avg DQI Score    │                               87.0% │
│ Processing Time  │                              1.23s │
╰──────────────────┴─────────────────────────────────────╯
```

### JSON Output File

```json
{
  "emissions_kgco2e": 2500.0,
  "emissions_tco2e": 2.5,
  "tier": "tier_1",
  "data_quality": {
    "dqi_score": 90.0,
    "tier": "tier_1",
    "completeness": 100.0,
    "accuracy": 95.0
  },
  "uncertainty": {
    "relative_uncertainty_pct": 15.0,
    "absolute_uncertainty_kgco2e": 375.0,
    "confidence_interval_95_lower": 2125.0,
    "confidence_interval_95_upper": 2875.0
  },
  "provenance": {
    "calculation_date": "2025-11-08T10:30:00",
    "methodology": "GHG Protocol Scope 3 Category 1",
    "emission_factor_source": "Supplier-specific PCF",
    "data_sources": ["ABC Steel Co."],
    "calculation_id": "calc-20251108-abc123"
  }
}
```

## Error Handling

### Input Validation Error
```
╭──────────── Validation Error ─────────────╮
│ Input Error: Unsupported file format: .txt│
│ Supported formats: .json, .csv, .xlsx, .xls│
╰────────────────────────────────────────────╯
```

### Calculator Error
```
╭──────────── Calculation Failed ────────────╮
│ Calculation Error: Missing required field │
│ 'quantity' in input data                   │
│                                            │
│ Category: 1                                │
│ Input: bad_data.json                       │
╰────────────────────────────────────────────╯
```

### Module Not Available
```
╭──────────── Initialization Error ──────────╮
│ Error: Calculator agent not available.     │
│                                            │
│ Details: No module named 'services.agents'│
│                                            │
│ Please ensure the calculator module is    │
│ properly installed.                        │
╰────────────────────────────────────────────╯
```

## Performance Tips

### For Large Batches

1. **Disable Monte Carlo** for faster processing:
   ```bash
   vcci calculate --category 1 --input large_file.csv --batch --no-mc
   ```

2. **Use CSV instead of JSON** for better memory efficiency

3. **Split very large files** into smaller batches:
   ```bash
   # Process 1000 records at a time
   split -l 1000 large_file.csv batch_
   for file in batch_*; do
     vcci calculate --category 1 --input "$file" --batch
   done
   ```

### For Single Records

1. **Use JSON format** for better type safety
2. **Enable verbose mode** for debugging: `--verbose`
3. **Save results** for later analysis: `--output results.json`

## Integration with Other Commands

### End-to-End Workflow

```bash
# 1. Ingest supplier data
vcci intake file --file suppliers.csv --entity-type supplier

# 2. Calculate emissions (batch)
vcci calculate --category 1 --input procurement.csv --batch --output cat1_results.json

# 3. Analyze results
vcci analyze --input cat1_results.json --type hotspot

# 4. Generate report
vcci report --input cat1_results.json --format ghg-protocol --output report.pdf
```

### Pipeline Integration

```bash
# Full automated pipeline
vcci pipeline run \
  --input data/ \
  --output results/ \
  --categories 1,4,6,15 \
  --enable-engagement

# Includes:
# - Data ingestion
# - Emission calculations (all specified categories)
# - Analysis and hotspot detection
# - Supplier engagement campaigns
# - Report generation
```

## Supported Categories

| Category | Name | Status | Special Features |
|----------|------|--------|------------------|
| 1 | Purchased Goods & Services | ✓ | 3-tier waterfall, LLM classification |
| 2 | Capital Goods | ✓ | LLM classification |
| 3 | Fuel & Energy Activities | ✓ | T&D losses |
| 4 | Transportation (Upstream) | ✓ | ISO 14083 compliance |
| 5 | Waste Operations | ✓ | LLM categorization |
| 6 | Business Travel | ✓ | Radiative forcing |
| 7 | Employee Commuting | ✓ | LLM patterns |
| 8 | Leased Assets (Upstream) | ✓ | Area-based |
| 9 | Transportation (Downstream) | ✓ | Last-mile delivery |
| 10 | Processing Sold Products | ✓ | Industry-specific |
| 11 | Use of Sold Products | ✓ | Lifetime modeling |
| 12 | End-of-Life Treatment | ✓ | Material analysis |
| 13 | Leased Assets (Downstream) | ✓ | LLM building type |
| 14 | Franchises | ✓ | LLM control |
| 15 | Investments | ✓ | PCAF Standard |

## Technical Architecture

### Component Stack

```
CLI (Typer + Rich)
    ↓
Scope3CalculatorAgent
    ↓
├── Category Calculators (1-15)
├── FactorBroker (multi-source cascading)
├── UncertaintyEngine (Monte Carlo)
├── ProvenanceBuilder (audit trail)
└── DataQualityEngine (DQI scoring)
```

### Async Processing

The CLI uses `asyncio.run()` to execute the async calculator methods:

```python
async def run_single():
    return await agent.calculate_by_category(category, data)

result = asyncio.run(run_single())
```

This ensures compatibility with the async calculator architecture while maintaining a synchronous CLI interface.

## Troubleshooting

### Common Issues

**Issue**: "Calculator agent not available"
- **Solution**: Ensure you're in the VCCI platform directory and dependencies are installed

**Issue**: "Unsupported file format"
- **Solution**: Use .json, .csv, .xlsx, or .xls files only

**Issue**: Excel files not loading
- **Solution**: Install openpyxl: `pip install openpyxl`

**Issue**: Slow batch processing
- **Solution**: Disable Monte Carlo with `--no-mc` flag

**Issue**: Missing required fields
- **Solution**: Check input data structure matches category requirements (use `--verbose` for details)

## Next Steps

1. **Try the examples**:
   ```bash
   cd examples/
   vcci calculate --category 1 --input sample_category1_single.json
   ```

2. **Create your own data files** based on the templates in `examples/`

3. **Integrate with pipelines** using `vcci pipeline run`

4. **Generate reports** with `vcci report`

5. **Explore other commands** with `vcci --help`

## Support

- **Documentation**: See `README.md` and `CLI_QUICK_REFERENCE.md`
- **Examples**: Check `examples/` directory
- **Help**: Run `vcci calculate --help`
- **Status**: Run `vcci status --detailed`

---

**Version**: 1.0.0
**Date**: 2025-11-08
**Team**: D (CLI Implementation Specialist)
