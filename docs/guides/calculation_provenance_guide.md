# GreenLang Calculation Provenance Guide

**Version:** 1.0.0
**Author:** GreenLang Team
**Last Updated:** 2025-12-01

## Overview

The **CalculationProvenance** module provides standardized audit trail tracking for all GreenLang calculators, based on CSRD, CBAM, and GL-001 through GL-010 best practices. It ensures zero-hallucination calculations with complete traceability for regulatory compliance.

## Features

- **Step-by-step calculation recording** - Every operation is tracked
- **SHA-256 hash-based integrity verification** - Tamper-proof audit trails
- **Complete audit trail with timestamps** - Full traceability
- **Standard reference tracking** - EPA, ISO, ASME, GHG Protocol, etc.
- **Data source lineage** - Track where every data point came from
- **Zero-hallucination guarantees** - Deterministic, reproducible calculations

## Quick Start

### Basic Usage

```python
from greenlang.core.provenance import CalculationProvenance, OperationType

# 1. Create provenance record
provenance = CalculationProvenance.create(
    agent_name="EmissionsCalculator",
    agent_version="1.0.0",
    calculation_type="scope1_emissions",
    input_data={
        "fuel_consumption_kg": 1000,
        "fuel_type": "natural_gas"
    },
    standards_applied=["GHG Protocol Scope 1"],
    data_sources=["EPA eGRID 2023"]
)

# 2. Record calculation steps
provenance.add_step(
    operation=OperationType.LOOKUP,
    description="Lookup emission factor for natural gas",
    inputs={"fuel_type": "natural_gas"},
    output=0.18414,
    data_source="EPA eGRID 2023",
    standard_reference="EPA AP-42"
)

provenance.add_step(
    operation=OperationType.MULTIPLY,
    description="Calculate total CO2 emissions",
    inputs={
        "fuel_consumption_kg": 1000,
        "emission_factor": 0.18414
    },
    output=184.14,
    formula="emissions = fuel_consumption * emission_factor",
    standard_reference="GHG Protocol Scope 1"
)

# 3. Finalize with output
provenance.finalize(output_data={
    "total_emissions_kg_co2": 184.14,
    "emissions_per_kg_fuel": 0.18414
})

# 4. Export for storage
record = provenance.to_dict()
print(record["calculation_id"])  # Unique calculation ID
print(record["input_hash"])      # SHA-256 hash for deduplication
print(record["output_hash"])     # SHA-256 hash for integrity
```

### Using with BaseCalculator

The `BaseCalculator` class automatically tracks provenance when enabled:

```python
from greenlang.agents.calculator import BaseCalculator, CalculatorConfig

class EmissionsCalculator(BaseCalculator):
    def calculate(self, inputs):
        fuel_kg = inputs["fuel_consumption_kg"]
        fuel_type = inputs["fuel_type"]

        # Lookup emission factor
        ef = self.lookup_emission_factor(fuel_type)

        # Record step in provenance
        self.record_provenance_step(
            operation="lookup",
            description=f"Lookup emission factor for {fuel_type}",
            inputs={"fuel_type": fuel_type},
            output=ef,
            data_source="EPA eGRID 2023",
            standard_reference="EPA AP-42"
        )

        # Calculate emissions
        emissions = fuel_kg * ef

        # Record calculation step
        self.record_provenance_step(
            operation="multiply",
            description="Calculate total emissions",
            inputs={"fuel_kg": fuel_kg, "emission_factor": ef},
            output=emissions,
            formula="emissions = fuel_kg * emission_factor"
        )

        return emissions

    def lookup_emission_factor(self, fuel_type):
        # Your lookup logic here
        return 0.18414  # Example: natural gas

# Configure with provenance enabled
config = CalculatorConfig(
    name="EmissionsCalculator",
    enable_provenance=True,
    agent_version="1.0.0"
)

calculator = EmissionsCalculator(config)

# Execute calculation
result = calculator.execute({
    "inputs": {
        "fuel_consumption_kg": 1000,
        "fuel_type": "natural_gas"
    },
    "calculation_type": "scope1_emissions"
})

# Access provenance record
if result.provenance:
    print(f"Calculation ID: {result.provenance['calculation_id']}")
    print(f"Steps: {len(result.provenance['steps'])}")
    print(f"Duration: {result.provenance['duration_ms']:.2f}ms")
```

## Storage and Retrieval

### Storing Provenance Records

```python
from greenlang.core.provenance.storage import SQLiteProvenanceStorage

# Create storage
storage = SQLiteProvenanceStorage("provenance.db")

# Store provenance record
calc_id = storage.store(provenance)
print(f"Stored calculation: {calc_id}")
```

### Retrieving Records

```python
# Retrieve by calculation ID
retrieved = storage.retrieve(calc_id)

if retrieved:
    print(f"Agent: {retrieved.metadata.agent_name}")
    print(f"Type: {retrieved.metadata.calculation_type}")
    print(f"Steps: {len(retrieved.steps)}")

    # Verify integrity
    integrity = retrieved.verify_integrity()
    if integrity["input_hash_valid"] and integrity["output_hash_valid"]:
        print("✓ Integrity verified")
```

### Querying Records

```python
from datetime import datetime, timedelta

# Query by agent name
emissions_calcs = storage.query(
    agent_name="EmissionsCalculator",
    limit=100
)

# Query by calculation type
scope1_calcs = storage.query(
    calculation_type="scope1_emissions",
    limit=50
)

# Query by time range
week_ago = datetime.utcnow() - timedelta(days=7)
recent_calcs = storage.query(
    start_time=week_ago,
    limit=100
)

# Query calculations with errors
failed_calcs = storage.query(
    has_errors=True,
    limit=50
)

# Find calculations with same inputs (deduplication)
duplicates = storage.find_by_input_hash(provenance.input_hash)
if len(duplicates) > 1:
    print(f"Found {len(duplicates)} calculations with same inputs")

# Find calculations using specific data source
defra_calcs = storage.find_by_data_source("DEFRA 2024")
print(f"Found {len(defra_calcs)} calculations using DEFRA 2024")
```

### Storage Statistics

```python
# Get storage statistics
stats = storage.get_statistics()

print(f"Total calculations: {stats['total_calculations']}")
print(f"Unique agents: {stats['unique_agents']}")
print(f"Records with errors: {stats['records_with_errors']}")
print(f"Average duration: {stats['average_duration_ms']:.2f}ms")
print(f"Calculation types: {stats['calculation_types']}")
```

## Operation Types

The `OperationType` enum defines standard operations:

```python
from greenlang.core.provenance import OperationType

# Standard operations
OperationType.LOOKUP        # Database/table lookup
OperationType.ADD           # Addition
OperationType.SUBTRACT      # Subtraction
OperationType.MULTIPLY      # Multiplication
OperationType.DIVIDE        # Division
OperationType.AGGREGATE     # Sum, average, etc.
OperationType.CONVERT       # Unit conversion
OperationType.VALIDATE      # Data validation
OperationType.TRANSFORM     # Data transformation
OperationType.INTERPOLATE   # Interpolation
OperationType.EXTRAPOLATE   # Extrapolation
```

## Complete Example: Scope 1 Emissions Calculator

```python
from greenlang.agents.calculator import BaseCalculator, CalculatorConfig
from greenlang.core.provenance import OperationType
from greenlang.core.provenance.storage import SQLiteProvenanceStorage

class Scope1EmissionsCalculator(BaseCalculator):
    """
    Calculate Scope 1 GHG emissions per GHG Protocol.

    Zero-Hallucination Guarantee:
    - Deterministic emission factor lookups
    - Mathematical calculations only (no LLM)
    - Complete provenance tracking
    """

    def __init__(self):
        config = CalculatorConfig(
            name="Scope1EmissionsCalculator",
            agent_version="2.1.0",
            enable_provenance=True,
            precision=6
        )
        super().__init__(config)

        # Emission factors database (simplified)
        self.emission_factors = {
            "natural_gas": 0.18414,  # kg CO2/kg fuel
            "diesel": 0.26760,
            "gasoline": 0.23120,
        }

    def calculate(self, inputs):
        """Calculate Scope 1 emissions."""
        fuel_consumption_kg = inputs["fuel_consumption_kg"]
        fuel_type = inputs["fuel_type"]

        # Step 1: Validate inputs
        if fuel_consumption_kg <= 0:
            raise ValueError("Fuel consumption must be positive")

        self.record_provenance_step(
            operation=OperationType.VALIDATE,
            description="Validate fuel consumption is positive",
            inputs={"fuel_consumption_kg": fuel_consumption_kg},
            output=True,
            standard_reference="GHG Protocol"
        )

        # Step 2: Lookup emission factor
        if fuel_type not in self.emission_factors:
            raise ValueError(f"Unknown fuel type: {fuel_type}")

        emission_factor = self.emission_factors[fuel_type]

        self.record_provenance_step(
            operation=OperationType.LOOKUP,
            description=f"Lookup emission factor for {fuel_type}",
            inputs={"fuel_type": fuel_type},
            output=emission_factor,
            data_source="EPA eGRID 2023",
            standard_reference="EPA AP-42 Table 1.4-1"
        )

        # Step 3: Calculate direct emissions (kg CO2)
        emissions_kg = fuel_consumption_kg * emission_factor

        self.record_provenance_step(
            operation=OperationType.MULTIPLY,
            description="Calculate direct CO2 emissions",
            inputs={
                "fuel_consumption_kg": fuel_consumption_kg,
                "emission_factor": emission_factor
            },
            output=emissions_kg,
            formula="emissions_kg_co2 = fuel_consumption_kg * emission_factor",
            standard_reference="GHG Protocol Scope 1 Equation 3.1"
        )

        # Step 4: Convert to tonnes CO2e
        emissions_tonnes = emissions_kg / 1000

        self.record_provenance_step(
            operation=OperationType.CONVERT,
            description="Convert kg CO2 to tonnes CO2e",
            inputs={"emissions_kg": emissions_kg},
            output=emissions_tonnes,
            formula="emissions_tonnes = emissions_kg / 1000"
        )

        return {
            "total_emissions_tonnes_co2e": round(emissions_tonnes, 6),
            "emissions_kg_co2": round(emissions_kg, 6),
            "emission_factor_used": emission_factor,
            "fuel_type": fuel_type
        }

# Usage
calculator = Scope1EmissionsCalculator()

result = calculator.execute({
    "inputs": {
        "fuel_consumption_kg": 5000,
        "fuel_type": "natural_gas"
    },
    "calculation_type": "scope1_emissions"
})

print(f"Result: {result.result_value}")
print(f"Success: {result.success}")

# Store provenance
if result.provenance:
    storage = SQLiteProvenanceStorage("scope1_provenance.db")
    calc_id = storage.store(
        CalculationProvenance.from_dict(result.provenance)
    )
    print(f"Stored provenance: {calc_id}")

    # Generate audit summary
    from greenlang.core.provenance import CalculationProvenance
    prov = CalculationProvenance.from_dict(result.provenance)
    summary = prov.get_audit_summary()
    print(f"\nAudit Summary:")
    print(summary["summary"])
    print(f"Standards: {summary['standards_applied']}")
    print(f"Data Sources: {summary['data_sources']}")
```

## Audit Trail Export

### Export Audit Report

```python
from datetime import datetime, timedelta

# Export audit report for last month
month_ago = datetime.utcnow() - timedelta(days=30)

report_path = storage.export_audit_report(
    "audit_report_2025_12.json",
    start_time=month_ago
)

print(f"Exported audit report to: {report_path}")
```

### Generate Compliance Report

```python
# Get all calculations for compliance reporting
calculations = storage.query(
    calculation_type="scope1_emissions",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 12, 31),
    has_errors=False,
    limit=10000
)

# Generate compliance report
total_emissions = sum(
    calc.output_data.get("total_emissions_tonnes_co2e", 0)
    for calc in calculations
)

print(f"Total Scope 1 Emissions (2025): {total_emissions:.2f} tonnes CO2e")
print(f"Number of calculations: {len(calculations)}")

# Verify all calculations passed integrity checks
integrity_passed = sum(
    1 for calc in calculations
    if calc.verify_integrity()["input_hash_valid"] and
       calc.verify_integrity()["output_hash_valid"]
)

print(f"Integrity verified: {integrity_passed}/{len(calculations)}")
```

## Best Practices

### 1. Always Record Data Sources

```python
# ✓ GOOD: Track data source
provenance.add_step(
    operation=OperationType.LOOKUP,
    description="Lookup emission factor",
    inputs={"fuel_type": "natural_gas"},
    output=0.18414,
    data_source="EPA eGRID 2023",  # ✓ Data source tracked
    standard_reference="EPA AP-42"
)

# ✗ BAD: Missing data source
provenance.add_step(
    operation=OperationType.LOOKUP,
    description="Lookup emission factor",
    inputs={"fuel_type": "natural_gas"},
    output=0.18414,
    # ✗ No data source - can't verify where data came from
)
```

### 2. Use Standard References

```python
# ✓ GOOD: Reference specific standard and section
provenance.add_step(
    operation=OperationType.MULTIPLY,
    description="Calculate emissions",
    inputs={"fuel_kg": 1000, "ef": 0.18414},
    output=184.14,
    formula="emissions = fuel_kg * ef",
    standard_reference="GHG Protocol Scope 1 Equation 3.1"  # ✓ Specific
)

# ✗ BAD: Vague reference
provenance.add_step(
    operation=OperationType.MULTIPLY,
    description="Calculate emissions",
    inputs={"fuel_kg": 1000, "ef": 0.18414},
    output=184.14,
    standard_reference="Some standard"  # ✗ Too vague
)
```

### 3. Record All Calculation Steps

```python
# ✓ GOOD: Every step recorded
ef = self.lookup_emission_factor(fuel_type)
self.record_provenance_step(...)  # ✓ Lookup recorded

emissions = fuel_kg * ef
self.record_provenance_step(...)  # ✓ Calculation recorded

# ✗ BAD: Missing steps
ef = self.lookup_emission_factor(fuel_type)
# ✗ Lookup not recorded - where did EF come from?

emissions = fuel_kg * ef
self.record_provenance_step(...)  # Only final step recorded
```

### 4. Use Descriptive Operation Descriptions

```python
# ✓ GOOD: Clear description
provenance.add_step(
    operation=OperationType.LOOKUP,
    description="Lookup natural gas emission factor for US-CA region",  # ✓ Clear
    inputs={"fuel_type": "natural_gas", "region": "US-CA"},
    output=0.18414
)

# ✗ BAD: Vague description
provenance.add_step(
    operation=OperationType.LOOKUP,
    description="Get value",  # ✗ What value? From where?
    inputs={"fuel_type": "natural_gas"},
    output=0.18414
)
```

### 5. Always Finalize Provenance

```python
# ✓ GOOD: Provenance finalized
provenance = CalculationProvenance.create(...)
provenance.add_step(...)
provenance.add_step(...)
provenance.finalize(output_data={"result": 184.14})  # ✓ Finalized

# ✗ BAD: Provenance not finalized
provenance = CalculationProvenance.create(...)
provenance.add_step(...)
# ✗ Never finalized - no output hash, no duration
```

### 6. Verify Integrity Before Trust

```python
# ✓ GOOD: Always verify integrity
retrieved = storage.retrieve(calc_id)
if retrieved:
    integrity = retrieved.verify_integrity()
    if integrity["input_hash_valid"] and integrity["output_hash_valid"]:
        # ✓ Safe to use
        result = retrieved.output_data
    else:
        # ✗ Data may be tampered with
        raise ValueError("Provenance integrity check failed")

# ✗ BAD: Use without verification
retrieved = storage.retrieve(calc_id)
result = retrieved.output_data  # ✗ Could be tampered with
```

## Troubleshooting

### Hash Mismatch After Deserialization

**Problem:** Hash verification fails after loading from storage.

**Solution:** Ensure you're using the same serialization method:

```python
# Use canonical JSON serialization
import json
data = json.dumps(provenance.to_dict(), sort_keys=True)
```

### Missing Provenance in Calculator Results

**Problem:** `result.provenance` is `None`.

**Solution:** Ensure provenance is enabled in config:

```python
config = CalculatorConfig(
    name="MyCalculator",
    enable_provenance=True,  # ← Must be True
    agent_version="1.0.0"
)
```

### Duplicate Calculation IDs

**Problem:** Same calculation ID for different calculations.

**Solution:** Calculation IDs include timestamp - if you're creating many calculations in the same microsecond, they may collide. Add a delay or use a counter:

```python
import time
for i in range(100):
    provenance = CalculationProvenance.create(...)
    time.sleep(0.001)  # 1ms delay
```

## Migration Guide

### Migrating from Old CalculationStep

If you're using the old `CalculationStep` from `BaseCalculator`:

```python
# Old way
self.add_calculation_step(
    step_name="Calculate emissions",
    formula="emissions = fuel * ef",
    inputs={"fuel": 1000, "ef": 0.18414},
    result=184.14
)

# New way (with provenance)
self.record_provenance_step(
    operation=OperationType.MULTIPLY,
    description="Calculate emissions",
    formula="emissions = fuel * ef",
    inputs={"fuel": 1000, "ef": 0.18414},
    output=184.14,
    standard_reference="GHG Protocol"
)
```

Both methods work - the new method adds provenance tracking for regulatory compliance.

## See Also

- [GreenLang Determinism Guide](./determinism_guide.md)
- [Zero-Hallucination Principles](../zero_hallucination.md)
- [Regulatory Compliance](../compliance/csrd_cbam.md)
- [GHG Protocol Implementation](../standards/ghg_protocol.md)

## Support

For questions or issues:
- GitHub Issues: https://github.com/GreenLang/greenlang/issues
- Documentation: https://docs.greenlang.org
- Email: support@greenlang.org
