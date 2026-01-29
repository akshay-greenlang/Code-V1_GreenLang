# GL-FOUND-X-002: Schema Compiler & Validator

## Overview

The **Schema Compiler & Validator** (GL-FOUND-X-002) is a core foundation agent in the GreenLang Climate OS platform. It provides comprehensive schema validation, type coercion, unit consistency checking, and machine-fixable error suggestions for all data payloads across the GreenLang ecosystem.

**Agent ID:** GL-FOUND-X-002
**Layer:** Foundation & Governance
**Version:** 1.0.0
**Family:** SchemaFamily
**Estimated Variants:** 1,500

---

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **JSON Schema Validation** | Full Draft-07 specification compliance with detailed error reporting |
| **Type Coercion** | Safe automatic type conversion with complete audit trails |
| **Unit Consistency** | Validates unit families (mass_co2e, energy, volume, etc.) across fields |
| **Fix Suggestions** | Machine-fixable hints with code snippets for rapid error resolution |
| **Schema Registry** | Version-controlled schema storage with semantic versioning |
| **Provenance Tracking** | SHA-256 hashes for audit trails and reproducibility |

---

## Quick Start

### Installation

```bash
pip install greenlang-sdk
```

### Python SDK - Basic Validation

```python
from greenlang.agents.foundation import SchemaCompilerAgent

# Initialize the agent
agent = SchemaCompilerAgent()

# Define your payload
payload = {
    "emissions": [
        {
            "fuel_type": "Natural Gas",
            "quantity": 1000,
            "unit": "therms",
            "co2e_emissions_kg": 5300.0,
            "scope": 1
        }
    ],
    "organization_id": "ORG-001"
}

# Validate against built-in schema
result = agent.run({
    "payload": payload,
    "schema_id": "gl-emissions-input"
})

if result.data["is_valid"]:
    print("Validation passed!")
else:
    print(f"Errors: {result.data['validation_result']['errors']}")
```

### CLI - Quick Validation

```bash
# Validate a YAML file against a schema
greenlang schema validate data.yaml --schema emissions/activity@1.3.0

# Validate with JSON output (for CI/CD)
greenlang schema validate data.yaml -s emissions/activity@1.3.0 --format json --quiet

# Batch validation with glob patterns
greenlang schema validate --glob "data/*.yaml" -s emissions/activity@1.3.0
```

### REST API - HTTP Request

```bash
curl -X POST "https://api.greenlang.io/v1/schema/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "payload": {
      "emissions": [{"fuel_type": "Gas", "co2e_emissions_kg": 100}]
    },
    "schema_ref": {
      "schema_id": "gl-emissions-input",
      "version": "1.0.0"
    }
  }'
```

---

## Architecture

```
+---------------------+
|   Input Payload     |
|   (YAML/JSON)       |
+----------+----------+
           |
           v
+----------+----------+
|   Schema Registry   |
|   (Version Control) |
+----------+----------+
           |
           v
+----------+----------+
|  Type Coercion      |
|  Engine             |
+----------+----------+
           |
           v
+----------+----------+
|  JSON Schema        |
|  Validator          |
+----------+----------+
           |
           v
+----------+----------+
|  Unit Consistency   |
|  Checker            |
+----------+----------+
           |
           v
+----------+----------+
|  Fix Suggestion     |
|  Generator          |
+----------+----------+
           |
           v
+----------+----------+
|  Validation Report  |
|  + Provenance Hash  |
+---------------------+
```

---

## Validation Pipeline

The Schema Compiler & Validator processes payloads through seven phases:

1. **Parse** - Parse YAML/JSON payload into structured data
2. **Compile** - Compile schema to intermediate representation (IR)
3. **Structural** - Validate types and required fields
4. **Constraints** - Validate ranges, patterns, and enum values
5. **Units** - Validate unit dimensions and consistency
6. **Rules** - Evaluate cross-field business rules
7. **Lint** - Check for typos, deprecated fields, and best practices

---

## Built-in Schemas

The Schema Registry comes pre-configured with these GreenLang standard schemas:

| Schema ID | Description | Tags |
|-----------|-------------|------|
| `gl-emissions-input` | Standard input schema for emissions data | emissions, input, core |
| `gl-activity-data` | Schema for activity data inputs | activity, input, core |
| `gl-calculation-result` | Schema for calculation output results | calculation, output, core |

---

## Unit Families

The validator recognizes these unit families for consistency checking:

| Family | Units | Base Unit |
|--------|-------|-----------|
| `mass_co2e` | kgCO2e, tCO2e, gCO2e, MtCO2e, GtCO2e | kgCO2e |
| `mass` | kg, g, t, mt, lb, ton | kg |
| `energy` | kWh, MWh, GWh, TWh, J, kJ, MJ, GJ, TJ, BTU, therm | kWh |
| `volume` | L, m3, gal, barrel, ft3 | L |
| `area` | m2, ft2, ha, acre, km2 | m2 |
| `distance` | km, m, mi, ft, nm | km |
| `time` | s, min, h, d, wk, mo, yr | h |
| `currency` | USD, EUR, GBP, JPY, CNY, CHF, CAD, AUD | USD |
| `percentage` | %, percent, pct | % |
| `dimensionless` | count, unit, each, pcs | unit |

---

## Zero-Hallucination Guarantees

GL-FOUND-X-002 implements GreenLang's zero-hallucination principles:

| Guarantee | Implementation |
|-----------|----------------|
| Complete Lineage | Every validation result has traceable provenance_hash |
| Deterministic Execution | Same inputs always produce same outputs |
| Coercion Tracking | All type coercions are recorded with before/after values |
| Rule-Based Hints | All error suggestions derived from schema rules, never inferred |
| SHA-256 Provenance | Cryptographic hashes for audit trail integrity |

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [CLI Reference](cli.md) | Command-line interface usage and examples |
| [REST API](api.md) | HTTP endpoints, request/response formats |
| [Python SDK](sdk.md) | Python library usage guide |
| [Migration Guide](migration.md) | Migrating from existing validation solutions |
| [Error Codes](error_codes.md) | Complete error code reference |

---

## Example Use Cases

### 1. Emissions Data Validation

```python
# Validate emissions data from ERP import
result = agent.validate(
    payload=erp_emissions_data,
    schema_id="gl-emissions-input",
    enable_coercion=True,  # Auto-convert "100" -> 100
    enable_unit_check=True  # Verify unit consistency
)

if not result.is_valid:
    for fix in result.fix_suggestions:
        if fix.auto_fixable:
            print(f"Auto-fix available: {fix.code_snippet}")
```

### 2. Batch Processing for Supplier Data

```python
# Validate 1000 supplier submissions
payloads = load_supplier_submissions()

result = agent.run({
    "payloads": payloads,
    "schema_id": "gl-supplier-pcf",
    "enable_coercion": True
})

# Get summary statistics
print(f"Valid: {result.data['summary']['valid_count']}")
print(f"Invalid: {result.data['summary']['error_count']}")
```

### 3. CI/CD Pipeline Integration

```bash
# In your CI pipeline
greenlang schema validate \
  --glob "data/**/*.yaml" \
  --schema emissions/activity@1.3.0 \
  --format sarif \
  --fail-on-warnings \
  > validation-results.sarif

# Exit code 0 = all valid, 1 = validation failed, 2 = system error
```

---

## Performance

| Metric | Value |
|--------|-------|
| Single payload validation | < 10ms |
| Batch validation (1000 items) | < 5s |
| Schema compilation | < 50ms |
| Cache hit ratio (typical) | > 95% |
| Maximum batch size | 1000 items |
| Maximum payload size | 10 MB |

---

## Support

- **Documentation:** https://docs.greenlang.io/schema
- **API Status:** https://status.greenlang.io
- **Community:** https://community.greenlang.io
- **Issues:** https://github.com/greenlang/greenlang/issues

---

**GL-FOUND-X-002: Schema Compiler & Validator**
*Ensuring data quality and compliance across the GreenLang Climate OS*
