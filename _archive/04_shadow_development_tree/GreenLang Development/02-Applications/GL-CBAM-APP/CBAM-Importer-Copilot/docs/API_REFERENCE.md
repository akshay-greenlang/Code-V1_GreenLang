# CBAM Importer Copilot - API Reference

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**Target Audience:** Developers, system integrators

---

## Table of Contents

1. [Overview](#overview)
2. [Python SDK](#python-sdk)
3. [CLI Commands](#cli-commands)
4. [Core Modules](#core-modules)
5. [Data Models](#data-models)
6. [Error Codes](#error-codes)
7. [Type Definitions](#type-definitions)

---

## Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  ┌──────────────┐                  ┌──────────────┐             │
│  │  CLI (Bash)  │                  │ Python SDK   │             │
│  └──────────────┘                  └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                         │
│  ┌────────────────────────────────────────────────────────┐     │
│  │               cbam_pipeline.py                         │     │
│  │  (Coordinates agent execution & provenance)            │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       AI AGENT LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Intake Agent │→ │ Calculator   │→ │ Packager     │          │
│  │              │  │ Agent        │  │ Agent        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       DATA & RULES LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ CN Codes DB  │  │ Emission     │  │ CBAM Rules   │          │
│  │              │  │ Factors DB   │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Module Organization

```
cbam_importer_copilot/
├── sdk/
│   ├── __init__.py
│   └── cbam_sdk.py          # Public SDK API
├── cli/
│   ├── __init__.py
│   └── cbam_commands.py      # CLI commands
├── agents/
│   ├── __init__.py
│   ├── shipment_intake_agent.py
│   ├── emissions_calculator_agent.py
│   └── reporting_packager_agent.py
├── provenance/
│   ├── __init__.py
│   └── provenance_utils.py   # Provenance tracking
├── data/
│   ├── emission_factors.py   # Emission factors DB
│   └── cn_codes.json         # CN code mappings
└── cbam_pipeline.py          # Pipeline orchestration
```

---

## Python SDK

### Core Functions

#### `cbam_build_report()`

**Main entry point for generating CBAM reports.**

```python
def cbam_build_report(
    input_file: Optional[str] = None,
    input_dataframe: Optional[pd.DataFrame] = None,
    importer_name: Optional[str] = None,
    importer_country: Optional[str] = None,
    importer_eori: Optional[str] = None,
    declarant_name: Optional[str] = None,
    declarant_position: Optional[str] = None,
    config: Optional[CBAMConfig] = None,
    cn_codes_path: str = "data/cn_codes.json",
    cbam_rules_path: str = "rules/cbam_rules.yaml",
    suppliers_path: Optional[str] = None,
    output_path: Optional[str] = None,
    save_output: bool = True
) -> CBAMReport
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_file` | `str` | No* | `None` | Path to CSV/Excel/JSON file |
| `input_dataframe` | `pd.DataFrame` | No* | `None` | Pandas DataFrame with shipments |
| `importer_name` | `str` | Yes** | `None` | Legal entity name |
| `importer_country` | `str` | Yes** | `None` | ISO 2-letter country code |
| `importer_eori` | `str` | Yes** | `None` | EORI number |
| `declarant_name` | `str` | No | `None` | Person filing report |
| `declarant_position` | `str` | No | `None` | Job title |
| `config` | `CBAMConfig` | No | `None` | Reusable config object |
| `cn_codes_path` | `str` | No | `"data/cn_codes.json"` | CN codes database path |
| `cbam_rules_path` | `str` | No | `"rules/cbam_rules.yaml"` | Validation rules path |
| `suppliers_path` | `str` | No | `None` | Supplier actuals file path |
| `output_path` | `str` | No | `"output/cbam_report.json"` | Output file path |
| `save_output` | `bool` | No | `True` | Save to file |

\* Either `input_file` OR `input_dataframe` must be provided

\** Required unless provided via `config` object

**Returns:** `CBAMReport` object

**Raises:**

- `ValueError` - Missing required parameters or invalid input
- `FileNotFoundError` - Input file not found
- `ValidationError` - Data validation failures (if strict mode)

**Example:**

```python
from sdk.cbam_sdk import cbam_build_report

report = cbam_build_report(
    input_file="data/shipments.csv",
    importer_name="Acme Steel BV",
    importer_country="NL",
    importer_eori="NL123456789012",
    declarant_name="John Smith",
    declarant_position="Compliance Officer"
)

print(f"Total emissions: {report.total_emissions_tco2:.2f} tCO2")
print(f"Valid shipments: {report.total_shipments}")
```

---

#### `cbam_validate_shipments()`

**Validate shipment data without generating report (dry-run).**

```python
def cbam_validate_shipments(
    input_file: Optional[str] = None,
    input_dataframe: Optional[pd.DataFrame] = None,
    importer_country: str = "NL",
    cn_codes_path: str = "data/cn_codes.json",
    cbam_rules_path: str = "rules/cbam_rules.yaml"
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_file` | `str` | No* | `None` | Path to input file |
| `input_dataframe` | `pd.DataFrame` | No* | `None` | Pandas DataFrame |
| `importer_country` | `str` | No | `"NL"` | Importer country code |
| `cn_codes_path` | `str` | No | `"data/cn_codes.json"` | CN codes path |
| `cbam_rules_path` | `str` | No | `"rules/cbam_rules.yaml"` | Rules path |

\* Either `input_file` OR `input_dataframe` must be provided

**Returns:** Dictionary with validation results

```python
{
    "metadata": {
        "total_records": 100,
        "valid_records": 95,
        "error_count": 3,
        "warning_count": 2
    },
    "errors": [
        {
            "record_index": 5,
            "field": "cn_code",
            "error_code": "E001",
            "message": "Missing CN code",
            "severity": "error"
        }
    ],
    "warnings": [ ... ],
    "validated_shipments": [ ... ]  # Only valid records
}
```

**Example:**

```python
from sdk.cbam_sdk import cbam_validate_shipments

validation = cbam_validate_shipments(
    input_file="data/shipments.csv",
    importer_country="NL"
)

if validation['metadata']['error_count'] == 0:
    print("✓ All records valid!")
else:
    print(f"✗ {validation['metadata']['error_count']} errors found")
    for error in validation['errors']:
        print(f"  Row {error['record_index']}: {error['message']}")
```

---

#### `cbam_calculate_emissions()`

**Calculate emissions for validated shipments.**

```python
def cbam_calculate_emissions(
    validated_shipments: Dict[str, Any],
    cn_codes_path: str = "data/cn_codes.json",
    suppliers_path: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `validated_shipments` | `Dict` | Yes | Output from `cbam_validate_shipments()` |
| `cn_codes_path` | `str` | No | CN codes database path |
| `suppliers_path` | `str` | No | Supplier actuals file path |

**Returns:** Dictionary with emissions calculations

```python
{
    "total_emissions_tco2": 1234.56,
    "total_quantity_tons": 5000.0,
    "shipments_with_emissions": [
        {
            "cn_code": "72071100",
            "quantity_tons": 15.5,
            "embedded_emissions_tco2": 12.4,
            "emission_factor_tco2_per_ton": 0.8,
            "emission_factor_source": "default",
            "calculation_method": "deterministic"
        }
    ],
    "emissions_by_product_group": { ... },
    "emissions_by_country": { ... }
}
```

**Example:**

```python
from sdk.cbam_sdk import cbam_validate_shipments, cbam_calculate_emissions

# Step 1: Validate
validated = cbam_validate_shipments(input_file="data/shipments.csv")

# Step 2: Calculate emissions
emissions = cbam_calculate_emissions(
    validated_shipments=validated,
    suppliers_path="examples/demo_suppliers.yaml"
)

print(f"Total emissions: {emissions['total_emissions_tco2']:.2f} tCO2")
```

---

### Data Classes

#### `CBAMConfig`

**Reusable configuration for repeated report generation.**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CBAMConfig:
    """CBAM configuration for repeated use."""

    importer_name: str
    importer_country: str
    importer_eori: str
    declarant_name: Optional[str] = None
    declarant_position: Optional[str] = None
    cn_codes_path: str = "data/cn_codes.json"
    cbam_rules_path: str = "rules/cbam_rules.yaml"
    suppliers_path: Optional[str] = None
    output_directory: str = "output"
    output_format: str = "both"  # "json", "excel", or "both"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CBAMConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return cls(
            importer_name=config['importer']['name'],
            importer_country=config['importer']['country'],
            importer_eori=config['importer']['eori'],
            declarant_name=config.get('declarant', {}).get('name'),
            declarant_position=config.get('declarant', {}).get('position'),
            cn_codes_path=config.get('paths', {}).get('cn_codes', 'data/cn_codes.json'),
            # ... etc
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        # Implementation
```

**Usage:**

```python
from sdk.cbam_sdk import CBAMConfig, cbam_build_report

# Create config once
config = CBAMConfig(
    importer_name="Acme Steel BV",
    importer_country="NL",
    importer_eori="NL123456789012",
    declarant_name="John Smith",
    declarant_position="Compliance Officer"
)

# Use for multiple reports
for month in ["jan", "feb", "mar"]:
    report = cbam_build_report(
        input_file=f"data/2025_{month}.csv",
        config=config
    )
```

---

#### `CBAMReport`

**Report result with convenient access methods.**

```python
from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd

@dataclass
class CBAMReport:
    """CBAM report result with convenient accessors."""

    raw_report: Dict[str, Any]

    @property
    def report_id(self) -> str:
        """Get report ID."""
        return self.raw_report.get("report_metadata", {}).get("report_id", "")

    @property
    def total_emissions_tco2(self) -> float:
        """Get total embedded emissions in tCO2."""
        return self.raw_report.get("emissions_summary", {}).get(
            "total_embedded_emissions_tco2", 0.0
        )

    @property
    def total_quantity_tons(self) -> float:
        """Get total quantity in tons."""
        return self.raw_report.get("emissions_summary", {}).get(
            "total_quantity_tons", 0.0
        )

    @property
    def total_shipments(self) -> int:
        """Get total number of shipments."""
        return self.raw_report.get("emissions_summary", {}).get(
            "total_shipments", 0
        )

    @property
    def unique_cn_codes(self) -> int:
        """Get number of unique CN codes."""
        return self.raw_report.get("emissions_summary", {}).get(
            "unique_cn_codes", 0
        )

    @property
    def unique_countries(self) -> int:
        """Get number of unique origin countries."""
        return self.raw_report.get("emissions_summary", {}).get(
            "unique_countries", 0
        )

    @property
    def is_valid(self) -> bool:
        """Check if report passed all validations."""
        return self.raw_report.get("validation_results", {}).get(
            "is_valid", False
        )

    @property
    def errors(self) -> List[Dict[str, Any]]:
        """Get all validation errors."""
        return self.raw_report.get("validation_results", {}).get("errors", [])

    @property
    def warnings(self) -> List[Dict[str, Any]]:
        """Get all validation warnings."""
        return self.raw_report.get("validation_results", {}).get("warnings", [])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert detailed goods to pandas DataFrame."""
        return pd.DataFrame(self.raw_report.get("detailed_goods", []))

    def save(self, output_path: str) -> None:
        """Save report to JSON file."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.raw_report, f, indent=2)

    def to_excel(self, output_path: str) -> None:
        """Save report to Excel file with multiple sheets."""
        with pd.ExcelWriter(output_path) as writer:
            # Detailed goods
            self.to_dataframe().to_excel(
                writer, sheet_name='Detailed Goods', index=False
            )

            # Aggregations
            agg = self.raw_report.get("aggregations", {})
            pd.DataFrame(agg.get("by_cn_code", [])).to_excel(
                writer, sheet_name='By CN Code', index=False
            )
            pd.DataFrame(agg.get("by_country", [])).to_excel(
                writer, sheet_name='By Country', index=False
            )
```

**Usage:**

```python
from sdk.cbam_sdk import cbam_build_report

report = cbam_build_report(...)

# Access summary data
print(f"Report ID: {report.report_id}")
print(f"Total emissions: {report.total_emissions_tco2:.2f} tCO2")
print(f"Total shipments: {report.total_shipments}")
print(f"Valid: {report.is_valid}")

# Convert to DataFrame
df = report.to_dataframe()
df.to_csv("output/processed_shipments.csv")

# Save to Excel
report.to_excel("output/cbam_report.xlsx")
```

---

## CLI Commands

### `gl cbam report`

Generate CBAM Transitional Registry report.

**Syntax:**

```bash
gl cbam report INPUT_FILE [OPTIONS]
```

**Options:**

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--importer-name TEXT` | String | Yes* | Legal entity name |
| `--importer-country CODE` | String | Yes* | ISO 2-letter country code |
| `--importer-eori TEXT` | String | Yes* | EORI number |
| `--declarant-name TEXT` | String | No | Person filing report |
| `--declarant-position TEXT` | String | No | Job title |
| `--config FILE` | Path | No | Path to .cbam.yaml |
| `--output FILE` | Path | No | Output path (default: output/cbam_report.json) |
| `--format [json\|excel\|both]` | String | No | Output format (default: both) |
| `--verbose` | Flag | No | Show detailed progress |

\* Required unless provided in `--config` file

**Exit Codes:**

- `0` - Success
- `1` - General error
- `2` - Invalid input data
- `3` - Validation failures
- `4` - File not found

**Example:**

```bash
gl cbam report data/shipments.csv \
  --importer-name "Acme Steel BV" \
  --importer-country "NL" \
  --importer-eori "NL123456789012" \
  --declarant-name "John Smith" \
  --declarant-position "Compliance Officer" \
  --format both \
  --verbose
```

---

### `gl cbam config`

Manage CBAM configuration.

**Sub-commands:**

#### `init`

Initialize configuration file.

```bash
gl cbam config init [--output FILE]

# Creates .cbam.yaml with template
```

#### `show`

Display current configuration.

```bash
gl cbam config show [--config FILE]

# Shows configuration with source (file/env/default)
```

#### `validate`

Validate configuration file.

```bash
gl cbam config validate [--config FILE]

# Checks configuration for errors
```

---

### `gl cbam validate`

Validate shipment data without generating report.

**Syntax:**

```bash
gl cbam validate INPUT_FILE [OPTIONS]
```

**Options:** Same as `gl cbam report`

**Exit Codes:**

- `0` - All validations passed
- `1` - General error
- `2` - Validation failures found

**Example:**

```bash
gl cbam validate data/shipments.csv \
  --importer-country "NL" \
  --verbose
```

---

## Core Modules

### cbam_pipeline.py

**Pipeline orchestration and provenance tracking.**

#### `CBAMPipeline`

```python
class CBAMPipeline:
    """End-to-end CBAM report generation pipeline."""

    def __init__(
        self,
        cn_codes_path: str = "data/cn_codes.json",
        cbam_rules_path: str = "rules/cbam_rules.yaml",
        suppliers_path: Optional[str] = None,
        enable_provenance: bool = True
    ):
        """Initialize pipeline with data sources."""

    def run(
        self,
        input_file: str,
        importer_info: Dict[str, Any],
        output_path: str = "output/cbam_report.json"
    ) -> Dict[str, Any]:
        """
        Run complete pipeline.

        Returns complete report with provenance.
        """

    def validate_only(
        self,
        input_file: str,
        importer_country: str = "NL"
    ) -> Dict[str, Any]:
        """Validate input data only (dry-run)."""
```

**Usage:**

```python
from cbam_pipeline import CBAMPipeline

pipeline = CBAMPipeline(
    cn_codes_path="data/cn_codes.json",
    cbam_rules_path="rules/cbam_rules.yaml",
    suppliers_path="examples/demo_suppliers.yaml"
)

report = pipeline.run(
    input_file="data/shipments.csv",
    importer_info={
        "name": "Acme Steel BV",
        "country": "NL",
        "eori": "NL123456789012"
    },
    output_path="output/cbam_report.json"
)
```

---

### Provenance Module

#### `hash_file()`

Calculate SHA256 hash for file integrity.

```python
def hash_file(
    file_path: str,
    algorithm: str = "sha256"
) -> Dict[str, Any]:
    """
    Calculate cryptographic hash of file.

    Returns:
        {
            "file_path": str,
            "file_name": str,
            "file_size_bytes": int,
            "hash_algorithm": str,
            "hash_value": str,
            "hash_timestamp": str (ISO 8601),
            "verification": str (command to verify)
        }
    """
```

**Example:**

```python
from provenance import hash_file

file_hash = hash_file("data/shipments.csv")
print(f"SHA256: {file_hash['hash_value']}")
print(f"Verify with: {file_hash['verification']}")
```

---

#### `get_environment_info()`

Capture complete execution environment.

```python
def get_environment_info() -> Dict[str, Any]:
    """
    Capture execution environment for reproducibility.

    Returns:
        {
            "timestamp": str,
            "python": {
                "version": str,
                "version_info": dict,
                "implementation": str,
                "compiler": str,
                "executable": str
            },
            "system": {
                "os": str,
                "release": str,
                "machine": str,
                "processor": str,
                "architecture": str,
                "hostname": str
            },
            "process": {
                "pid": int,
                "cwd": str,
                "user": str
            }
        }
    """
```

**Example:**

```python
from provenance import get_environment_info

env = get_environment_info()
print(f"Python: {env['python']['version']}")
print(f"OS: {env['system']['os']} {env['system']['release']}")
```

---

#### `create_provenance_record()`

Create complete provenance record.

```python
def create_provenance_record(
    report_id: str,
    input_file: str,
    configuration: Dict[str, Any],
    agent_executions: List[Dict[str, Any]],
    validation_results: Dict[str, Any]
) -> ProvenanceRecord:
    """
    Create complete provenance record for audit trail.

    Returns ProvenanceRecord dataclass with all metadata.
    """
```

**Example:**

```python
from provenance import create_provenance_record

provenance = create_provenance_record(
    report_id="CBAM-2025Q4-001",
    input_file="data/shipments.csv",
    configuration={"importer": {...}},
    agent_executions=[...],
    validation_results={...}
)

provenance.save("output/provenance.json")
```

---

#### `validate_provenance()`

Validate provenance integrity.

```python
def validate_provenance(
    provenance: ProvenanceRecord,
    input_file: str
) -> Dict[str, Any]:
    """
    Validate provenance record integrity.

    Checks:
    - File hash matches original
    - All required fields present
    - Agent execution chain valid

    Returns:
        {
            "is_valid": bool,
            "checks_passed": int,
            "errors": List[str],
            "warnings": List[str],
            "error_details": List[Dict]
        }
    """
```

**Example:**

```python
from provenance import ProvenanceRecord, validate_provenance

# Load provenance
provenance = ProvenanceRecord.load("output/provenance.json")

# Validate
validation = validate_provenance(
    provenance,
    input_file="data/shipments.csv"
)

if validation['is_valid']:
    print("✓ Provenance verified")
else:
    print("✗ Validation failed")
    for error in validation['errors']:
        print(f"  - {error}")
```

---

## Data Models

### Shipment Schema

```python
{
    "cn_code": str,                      # 8-digit CN code (required)
    "country_of_origin": str,            # ISO 2-letter code (required)
    "quantity_tons": float,              # Net weight in tons (required)
    "import_date": str,                  # ISO 8601 date (required)
    "supplier_id": str,                  # Supplier reference (optional)
    "installation_id": str,              # Installation ID (optional)
    "actual_emissions_tco2": float,      # Supplier actuals (optional)
    "invoice_number": str,               # Invoice ref (optional)
    "customs_declaration": str,          # Customs ref (optional)
    "product_description": str           # Description (optional)
}
```

### Report Output Schema

```python
{
    "report_metadata": {
        "report_id": str,
        "generated_at": str (ISO 8601),
        "importer": {
            "name": str,
            "country": str,
            "eori": str
        },
        "declarant": {
            "name": str,
            "position": str
        },
        "reporting_period": str
    },

    "emissions_summary": {
        "total_embedded_emissions_tco2": float,
        "total_quantity_tons": float,
        "total_shipments": int,
        "unique_cn_codes": int,
        "unique_countries": int,
        "emissions_by_product_group": {...},
        "emissions_by_country": {...}
    },

    "detailed_goods": [
        {
            "cn_code": str,
            "description": str,
            "country_of_origin": str,
            "quantity_tons": float,
            "embedded_emissions_tco2": float,
            "emission_factor_tco2_per_ton": float,
            "emission_factor_source": str,  # "default" or "supplier"
            "calculation_method": str,       # "deterministic"
            "supplier_id": str,
            "import_date": str
        }
    ],

    "aggregations": {
        "by_cn_code": [...],
        "by_country": [...],
        "by_product_group": [...]
    },

    "validation_results": {
        "is_valid": bool,
        "errors": List[Dict],
        "warnings": List[Dict],
        "complex_goods_percentage": float
    },

    "provenance": {
        "input_file_integrity": {...},
        "execution_environment": {...},
        "agent_execution": [...],
        "reproducibility": {...}
    }
}
```

---

## Error Codes

### Validation Errors (E-codes)

| Code | Description | Severity | Resolution |
|------|-------------|----------|------------|
| `E001` | Missing CN code | Error | Add valid 8-digit CN code |
| `E002` | Invalid CN code format | Error | Use format: 12345678 |
| `E003` | Unknown CN code | Error | Check against CBAM Annex I |
| `E004` | Missing country of origin | Error | Add ISO 2-letter code |
| `E005` | Invalid country code | Error | Use ISO 3166-1 alpha-2 |
| `E006` | Missing quantity | Error | Add quantity in tons |
| `E007` | Invalid quantity (negative/zero) | Error | Must be positive number |
| `E008` | Missing import date | Error | Add date in YYYY-MM-DD |
| `E009` | Invalid date format | Error | Use ISO 8601: YYYY-MM-DD |
| `E010` | Date outside reporting period | Error | Check reporting period |

### Validation Warnings (W-codes)

| Code | Description | Severity | Impact |
|------|-------------|----------|--------|
| `W001` | Missing supplier data | Warning | Will use default emission factors |
| `W002` | Unusually high emission factor | Warning | Review calculation |
| `W003` | Complex goods classification | Warning | May need manual review |
| `W004` | Missing optional field | Warning | Reduces data quality |
| `W005` | Duplicate shipment detected | Warning | Check for duplicates |

### System Errors

| Code | Description | Recovery |
|------|-------------|----------|
| `S001` | File not found | Check file path |
| `S002` | Invalid file format | Use CSV/Excel/JSON |
| `S003` | Memory limit exceeded | Process in chunks |
| `S004` | Database connection failed | Check data file paths |
| `S005` | Configuration error | Validate config file |

---

## Type Definitions

### TypeScript-style Type Definitions

```typescript
// For documentation purposes (Python uses type hints)

type CBAMReport = {
  report_metadata: ReportMetadata;
  emissions_summary: EmissionsSummary;
  detailed_goods: DetailedGood[];
  aggregations: Aggregations;
  validation_results: ValidationResults;
  provenance: Provenance;
}

type ReportMetadata = {
  report_id: string;
  generated_at: string;  // ISO 8601
  importer: Importer;
  declarant?: Declarant;
  reporting_period: string;
}

type EmissionsSummary = {
  total_embedded_emissions_tco2: number;
  total_quantity_tons: number;
  total_shipments: number;
  unique_cn_codes: number;
  unique_countries: number;
  emissions_by_product_group: Record<string, number>;
  emissions_by_country: Record<string, number>;
}

type DetailedGood = {
  cn_code: string;
  description: string;
  country_of_origin: string;
  quantity_tons: number;
  embedded_emissions_tco2: number;
  emission_factor_tco2_per_ton: number;
  emission_factor_source: "default" | "supplier";
  calculation_method: "deterministic";
  supplier_id?: string;
  import_date: string;
}

type Provenance = {
  input_file_integrity: FileHash;
  execution_environment: Environment;
  dependencies: Record<string, string>;
  agent_execution: AgentExecution[];
  reproducibility: Reproducibility;
}
```

---

## Version History

### 1.0.0 (2025-10-15)

**Initial release**

- Core SDK with `cbam_build_report()`, `cbam_validate_shipments()`, `cbam_calculate_emissions()`
- CLI commands: `report`, `config`, `validate`
- 3-agent pipeline: Intake, Calculator, Packager
- Enterprise provenance tracking
- SHA256 file integrity
- Zero hallucination architecture
- Supports CSV, Excel, JSON, pandas DataFrames
- Multi-tenant configuration support
- Complete audit trail
- Regulatory compliance features

---

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**License:** MIT

---

*For usage examples, see `docs/USER_GUIDE.md`*
*For compliance information, see `docs/COMPLIANCE_GUIDE.md`*
