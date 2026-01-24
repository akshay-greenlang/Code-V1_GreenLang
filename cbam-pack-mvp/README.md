# GreenLang CBAM Pack

**EU Carbon Border Adjustment Mechanism Compliance Tool**

A deterministic CLI tool for generating EU CBAM Transitional Registry XML reports with full audit bundles for compliance and auditability.

## Features

- **CBAM Transitional Period Support**: Full compliance with EU CBAM reporting requirements (2024-2025)
- **Steel & Aluminum**: Support for CN codes 72xx, 73xx (Steel) and 76xx (Aluminum)
- **Defaults-First Approach**: Built-in emission factors with supplier-specific override capability
- **Full Audit Trail**: Complete provenance tracking with claims, assumptions, and lineage
- **Zero Hallucination**: Deterministic calculations using Decimal arithmetic
- **Local-First**: All processing happens locally - no data leaves your machine

## Installation

```bash
# Install from source
cd cbam-pack-mvp
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Your Config File

Create a `cbam.yaml` configuration file:

```yaml
declarant:
  name: "Your Company GmbH"
  eori_number: "DE123456789012345"
  address:
    street: "123 Example Street"
    city: "Berlin"
    postal_code: "10115"
    country: "DE"
  contact:
    name: "John Doe"
    email: "cbam@example.com"

reporting_period:
  quarter: "Q1"
  year: 2025

settings:
  aggregation: "aggregate_by_cn_country"
```

### 2. Prepare Your Import Ledger

Create an `imports.csv` file with your CBAM-regulated imports:

```csv
line_id,quarter,year,cn_code,product_description,country_of_origin,quantity,unit
IMP-001,Q1,2025,72061000,Carbon steel ingots,CN,50000,kg
IMP-002,Q1,2025,76011000,Primary aluminum ingots,RU,120,tonnes
```

### 3. Run the Pipeline

```bash
gl-cbam run cbam --config cbam.yaml --imports imports.csv --out ./output/
```

## Output Artifacts

The pipeline generates:

| File | Description |
|------|-------------|
| `cbam_report.xml` | EU CBAM Registry submission file |
| `report_summary.xlsx` | Human-readable Excel summary |
| `audit/claims.json` | Formal emission claims |
| `audit/lineage.json` | Data provenance graph |
| `audit/assumptions.json` | All assumptions made |
| `audit/run_manifest.json` | Execution metadata |
| `audit/checksums.json` | SHA-256 hashes |

## Commands

### Run Pipeline

```bash
gl-cbam run cbam -c <config.yaml> -i <imports.csv> -o <output_dir>

Options:
  -c, --config    Path to CBAM config YAML file (required)
  -i, --imports   Path to import ledger CSV/XLSX (required)
  -o, --out       Output directory (required)
  -v, --verbose   Enable verbose logging
  --dry-run       Validate only, don't generate outputs
```

### Validate Only

```bash
gl-cbam validate -c <config.yaml> -i <imports.csv>
```

### Show Version

```bash
gl-cbam version
```

## Supported CN Codes

| Product Category | CN Prefix | Examples |
|-----------------|-----------|----------|
| Iron & Steel | 72xx | 72061000, 72082500 |
| Steel Articles | 73xx | 73011000, 73021000 |
| Aluminum | 76xx | 76011000, 76012000 |

## Supplier-Specific Data

To use supplier-specific emission factors instead of defaults, add these optional columns:

```csv
supplier_id,installation_id,supplier_direct_emissions,supplier_indirect_emissions,supplier_certificate_ref
SUP-001,INST-001,1.65,0.38,CERT-2025-001
```

## Architecture

The CBAM Pack implements a 7-agent pipeline:

1. **Orchestrator**: Coordinates pipeline execution
2. **Schema Validator**: Validates input files
3. **Unit Normalizer**: Converts units to tonnes
4. **Factor Library**: Provides emission factors
5. **Calculator**: Computes embedded emissions
6. **XML Exporter**: Generates compliant XML
7. **Evidence Packager**: Creates audit bundle

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Support

- Documentation: https://greenlang.in/docs
- Issues: https://github.com/greenlang/greenlang/issues
- Email: support@greenlang.in
