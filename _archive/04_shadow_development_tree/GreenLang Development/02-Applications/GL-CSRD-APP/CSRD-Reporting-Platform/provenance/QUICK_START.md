# Provenance Framework - Quick Start Guide

## Installation

The provenance framework is already included in the CSRD Reporting Platform. No additional installation required.

```python
from provenance import *
```

---

## 5-Minute Quick Start

### 1. Hash a File (Data Integrity)

```python
from provenance import hash_file

# Hash input file for integrity verification
file_info = hash_file("data/esg_data.csv")

print(f"File: {file_info['file_name']}")
print(f"SHA256: {file_info['hash_value']}")
print(f"Size: {file_info['human_readable_size']}")

# Later, verify file hasn't changed
new_hash = hash_file("data/esg_data.csv")
assert new_hash['hash_value'] == file_info['hash_value']  # ✓ Verified
```

---

### 2. Track a Calculation

```python
from provenance import track_calculation_lineage

# Track a calculation with complete lineage
lineage = track_calculation_lineage(
    metric_code="E1-1",
    metric_name="Total GHG Emissions",
    formula="Scope1 + Scope2 + Scope3",
    input_values={
        "Scope1": 1000.0,
        "Scope2": 500.0,
        "Scope3": 2000.0
    },
    output_value=3500.0,
    output_unit="tCO2e",
    formula_type="sum"
)

print(f"Hash: {lineage.hash}")  # SHA-256 for verification
print(f"Output: {lineage.output_value} {lineage.output_unit}")
```

---

### 3. Track Data Source

```python
from provenance import create_data_source

# Excel source
source = create_data_source(
    source_type="excel",
    file_path="data/emissions.xlsx",
    sheet_name="Scope1",
    row_index=10,
    column_name="emissions_tCO2e",
    cell_reference="D10"
)

print(f"Source ID: {source.source_id}")
print(f"File hash: {source.file_hash[:16]}...")
```

---

### 4. Create Provenance Record

```python
from provenance import create_provenance_record

# Create a complete provenance record
record = create_provenance_record(
    agent_name="CalculatorAgent",
    operation="calculate_ghg_emissions",
    inputs={"metrics": ["E1-1", "E1-2"]},
    outputs={"calculated": 2, "failed": 0},
    duration_seconds=0.05,
    status="success"
)

print(f"Record ID: {record.record_id}")
print(f"Timestamp: {record.timestamp}")
```

---

### 5. Create Audit Package

```python
from provenance import create_audit_package

# Create complete ZIP audit package
audit_pkg = create_audit_package(
    provenance_records=[record],
    output_path="output/audit_package.zip",
    include_config="config/csrd_config.yaml"
)

print(f"Audit package: {audit_pkg}")
```

---

## Common Use Cases

### Use Case 1: CalculatorAgent Integration

```python
from provenance import track_calculation_lineage, create_provenance_record

class CalculatorAgent:
    def __init__(self):
        self.provenance_records = []

    def calculate_metric(self, metric_code, formula_spec, input_data):
        # Perform calculation
        result = self._evaluate_formula(formula_spec, input_data)

        # Track lineage
        lineage = track_calculation_lineage(
            metric_code=metric_code,
            metric_name=formula_spec["metric_name"],
            formula=formula_spec["formula"],
            input_values=input_data,
            output_value=result,
            output_unit=formula_spec["unit"],
            formula_type=formula_spec.get("calculation_type", "expression")
        )

        # Store provenance
        self.provenance_records.append(lineage)

        return result, lineage
```

---

### Use Case 2: IntakeAgent Integration

```python
from provenance import create_data_source, hash_file

class IntakeAgent:
    def read_esg_data(self, input_file):
        # Hash file first
        file_hash = hash_file(input_file)

        # Read data
        df = pd.read_csv(input_file)

        # Track source for each data point
        for idx, row in df.iterrows():
            source = create_data_source(
                source_type="csv",
                file_path=input_file,
                file_hash=file_hash["hash_value"],
                row_index=idx,
                column_name="emissions"
            )

            # Add source to enriched data
            row['data_source'] = source.dict()

        return df
```

---

### Use Case 3: AuditAgent Integration

```python
from provenance import (
    create_audit_package,
    generate_audit_report,
    build_lineage_graph
)

class AuditAgent:
    def generate_auditor_package(self, provenance_records):
        # Build lineage graph
        calculation_lineages = [
            r.calculation_lineage for r in provenance_records
            if r.calculation_lineage
        ]
        G = build_lineage_graph(calculation_lineages)

        # Generate audit report
        report = generate_audit_report(
            provenance_records=provenance_records,
            output_path="output/audit_report.md"
        )

        # Create audit package
        audit_pkg = create_audit_package(
            provenance_records=provenance_records,
            output_path="output/audit_package.zip",
            include_config="config/csrd_config.yaml",
            include_lineage_graph=True
        )

        return {
            "audit_package": str(audit_pkg),
            "audit_report": "output/audit_report.md",
            "lineage_graph": G
        }
```

---

## CLI Usage

```bash
# Hash a file
python -m provenance.provenance_utils hash-file data/esg_data.csv

# Capture environment
python -m provenance.provenance_utils capture-env \
    --config config/csrd_config.yaml \
    --output environment.json

# Verify file integrity
python -m provenance.provenance_utils verify-hash \
    data/esg_data.csv \
    a1b2c3d4e5f6789...
```

---

## Best Practices

### 1. Always Hash Input Files

```python
# ✓ Good
file_hash = hash_file(input_file)
# Use hash in provenance records

# ✗ Bad
# Not tracking file integrity
```

---

### 2. Track All Calculations

```python
# ✓ Good
lineage = track_calculation_lineage(
    metric_code="E1-1",
    formula="A+B",
    input_values={"A": 100, "B": 200},
    output_value=300,
    output_unit="tCO2e"
)

# ✗ Bad
result = a + b  # No provenance tracking
```

---

### 3. Include Intermediate Steps

```python
# ✓ Good
lineage = track_calculation_lineage(
    ...,
    intermediate_steps=[
        "Scope 1: 1000.0 tCO2e",
        "Scope 2: 500.0 tCO2e",
        "Scope 3: 2000.0 tCO2e",
        "Total: 3500.0 tCO2e"
    ]
)

# ✗ Bad
lineage = track_calculation_lineage(...)  # No steps
```

---

### 4. Track Data Sources

```python
# ✓ Good
source = create_data_source(
    source_type="excel",
    file_path="data.xlsx",
    sheet_name="Sheet1",
    row_index=10,
    column_name="emissions"
)

# ✗ Bad
# No source tracking - can't trace back to origin
```

---

### 5. Capture LLM Models

```python
# ✓ Good (for MaterialityAgent)
env = capture_environment(
    llm_models={
        "materiality": "gpt-4o",
        "narratives": "claude-3.5-sonnet"
    }
)

# ✗ Bad
env = capture_environment()  # Missing LLM info
```

---

## API Reference

### Core Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `hash_file(path)` | Hash file for integrity | Dict with hash details |
| `hash_data(data)` | Hash arbitrary data | SHA-256 hex string |
| `capture_environment()` | Capture runtime env | EnvironmentSnapshot |
| `get_dependency_versions()` | Get package versions | Dict[str, str] |

### Data Source Tracking

| Function | Purpose | Returns |
|----------|---------|---------|
| `create_data_source(...)` | Create data source record | DataSource |

### Calculation Lineage

| Function | Purpose | Returns |
|----------|---------|---------|
| `track_calculation_lineage(...)` | Track calculation | CalculationLineage |

### Provenance Records

| Function | Purpose | Returns |
|----------|---------|---------|
| `create_provenance_record(...)` | Create provenance record | ProvenanceRecord |

### Graph Analysis

| Function | Purpose | Returns |
|----------|---------|---------|
| `build_lineage_graph(lineages)` | Build dependency graph | networkx.DiGraph |
| `get_calculation_path(graph, metric)` | Get calc path | List[str] |

### Serialization

| Function | Purpose | Returns |
|----------|---------|---------|
| `serialize_provenance(records)` | Serialize to JSON | Dict |
| `save_provenance_json(records, path)` | Save to file | None |

### Audit Package

| Function | Purpose | Returns |
|----------|---------|---------|
| `create_audit_package(...)` | Create ZIP package | Path |
| `generate_audit_report(records)` | Generate Markdown report | str |

---

## Data Models

### DataSource

```python
@dataclass
class DataSource:
    source_id: str              # Auto-generated UUID
    source_type: str            # csv, json, excel, database, etc.
    file_path: Optional[str]    # Absolute path to file
    file_hash: Optional[str]    # SHA-256 of file
    sheet_name: Optional[str]   # Excel sheet
    row_index: Optional[int]    # Row number
    column_name: Optional[str]  # Column name
    cell_reference: Optional[str]  # Excel cell (e.g., "D10")
    timestamp: str              # ISO 8601 timestamp
```

### CalculationLineage

```python
@dataclass
class CalculationLineage:
    lineage_id: str             # Auto-generated UUID
    metric_code: str            # ESRS metric code
    metric_name: str            # Metric name
    formula: str                # Formula string
    formula_type: str           # sum, division, etc.
    input_values: Dict          # Input variables
    output_value: Any           # Result
    output_unit: str            # Unit
    hash: str                   # SHA-256 of formula+inputs
    intermediate_steps: List[str]  # Calculation steps
    dependencies: List[str]     # Dependent metrics
```

### EnvironmentSnapshot

```python
@dataclass
class EnvironmentSnapshot:
    snapshot_id: str            # Auto-generated UUID
    timestamp: str              # ISO 8601
    python_version: str         # Full version
    python_major: int
    python_minor: int
    python_micro: int
    platform: str               # Windows, Linux, Darwin
    machine: str                # x86_64, ARM64, etc.
    package_versions: Dict      # Package → version
    llm_models: Dict            # Purpose → model
```

### ProvenanceRecord

```python
@dataclass
class ProvenanceRecord:
    record_id: str              # Auto-generated UUID
    timestamp: str              # ISO 8601
    agent_name: str             # Agent name
    operation: str              # Operation type
    inputs: Dict                # Input data
    outputs: Dict               # Output data
    environment: EnvironmentSnapshot
    calculation_lineage: Optional[CalculationLineage]
    duration_seconds: float
    status: str                 # success, warning, error
    errors: List[str]
    warnings: List[str]
```

---

## Troubleshooting

### Import Error

```python
# Error: ModuleNotFoundError: No module named 'provenance'

# Solution: Make sure you're in the right directory
import sys
sys.path.append('C:/Users/aksha/Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform')
from provenance import *
```

### NetworkX Not Found

```python
# Error: ModuleNotFoundError: No module named 'networkx'

# Solution: Install networkx
pip install networkx
```

### Hash Verification Failed

```python
# Error: Hash mismatch

# Cause: File was modified after hashing
# Solution: Re-hash the file
new_hash = hash_file("data/esg_data.csv")
```

---

## Examples Directory

See complete examples in:
```
examples/
├── basic_usage.py              # Basic usage examples
├── calculator_integration.py   # CalculatorAgent integration
├── audit_package_demo.py       # Audit package creation
└── lineage_graph_demo.py       # Dependency graph visualization
```

---

## Support

For questions or issues:
- **Documentation**: See `PROVENANCE_FRAMEWORK_SUMMARY.md`
- **Code**: `provenance/provenance_utils.py`
- **Team**: GreenLang CSRD Team

---

**Last Updated**: 2025-10-18
**Version**: 1.0.0
