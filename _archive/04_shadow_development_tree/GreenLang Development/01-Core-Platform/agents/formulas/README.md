# GreenLang Formula Versioning System

Centralized formula management with version control, audit trails, rollback capability, and A/B testing support.

## Overview

The Formula Versioning System provides enterprise-grade formula management inspired by CSRD's `esrs_formulas.yaml` approach, but with complete version control and audit capabilities.

### Key Features

- **Version Control**: Full version history with rollback capability
- **Zero-Hallucination**: Deterministic calculations with complete provenance
- **Audit Trail**: SHA-256 hashing and execution logging
- **Database-Backed**: SQLite (dev) / PostgreSQL (prod) storage
- **Migration Support**: Import from YAML and Python modules
- **A/B Testing**: Test formula variants before production deployment
- **Dependency Resolution**: Automatic topological sorting
- **CLI Management**: Command-line tools for all operations

## Architecture

```
greenlang/formulas/
├── __init__.py           # Package exports
├── schema.sql            # Database schema
├── models.py             # Pydantic data models
├── repository.py         # Database access layer
├── engine.py             # Formula execution engine
├── manager.py            # High-level API
├── migration.py          # Import from YAML/Python
├── cli.py                # Command-line interface
└── tests/
    ├── test_formula_manager.py
    ├── test_formula_versioning.py
    └── test_formula_execution.py
```

## Quick Start

### 1. Installation

```bash
# Install GreenLang
pip install greenlang

# Or install in development mode
cd greenlang
pip install -e .
```

### 2. Initialize Database

```python
from greenlang.formulas import FormulaManager

# Initialize with SQLite database
manager = FormulaManager("formulas.db")
```

### 3. Create Your First Formula

```python
from greenlang.formulas.models import FormulaCategory

# Create formula metadata
formula_id = manager.create_formula(
    formula_code="E1-1",
    formula_name="Total Scope 1 GHG Emissions",
    category=FormulaCategory.EMISSIONS,
    description="Sum of all Scope 1 emission sources",
    standard_reference="ESRS E1",
    created_by="john.doe@company.com"
)

# Create initial version
version_data = {
    'formula_expression': 'stationary + mobile + process + fugitive',
    'calculation_type': 'sum',
    'required_inputs': ['stationary', 'mobile', 'process', 'fugitive'],
    'output_unit': 'tCO2e',
    'deterministic': True,
    'zero_hallucination': True,
}

version_id = manager.create_new_version(
    formula_code="E1-1",
    formula_data=version_data,
    change_notes="Initial version based on ESRS E1 guidance",
    auto_activate=True
)
```

### 4. Execute Formula

```python
# Execute with input data
result = manager.execute_formula(
    formula_code="E1-1",
    input_data={
        'stationary': 1000,
        'mobile': 500,
        'process': 200,
        'fugitive': 50
    },
    agent_name="EmissionsCalculatorAgent"
)

print(f"Total Scope 1 Emissions: {result} tCO2e")
# Output: Total Scope 1 Emissions: 1750 tCO2e
```

### 5. Get Full Execution Result with Provenance

```python
# Get complete result with audit trail
result = manager.execute_formula_full(
    formula_code="E1-1",
    input_data={
        'stationary': 1000,
        'mobile': 500,
        'process': 200,
        'fugitive': 50
    },
    agent_name="EmissionsCalculatorAgent",
    calculation_id="CALC-2025-001",
    user_id="john.doe@company.com"
)

print(f"Output: {result.output_value} tCO2e")
print(f"Input Hash: {result.input_hash}")
print(f"Output Hash: {result.output_hash}")
print(f"Execution Time: {result.execution_time_ms:.2f}ms")
print(f"Status: {result.execution_status}")
```

## Version Management

### Creating New Versions

```python
# Create improved version with updated calculation
version_data = {
    'formula_expression': 'stationary + mobile + process + fugitive + biogenic',
    'calculation_type': 'sum',
    'required_inputs': ['stationary', 'mobile', 'process', 'fugitive', 'biogenic'],
    'output_unit': 'tCO2e',
}

v2_id = manager.create_new_version(
    formula_code="E1-1",
    formula_data=version_data,
    change_notes="Added biogenic emissions per updated ESRS guidance",
    auto_activate=False  # Don't activate yet
)
```

### Activating Versions

```python
# Activate version 2
manager.activate_version("E1-1", version_number=2)

# Activate with specific effective date
from datetime import date
manager.activate_version(
    "E1-1",
    version_number=2,
    effective_from=date(2025, 1, 1)
)
```

### Rolling Back

```python
# Rollback to previous version (creates new version as copy)
new_version_id = manager.rollback_to_version(
    formula_code="E1-1",
    version_number=1
)

# This creates version 3 which is a copy of version 1
# Version 2 remains in history but is deprecated
```

### Comparing Versions

```python
# Compare two versions
comparison = manager.compare_versions(
    formula_code="E1-1",
    version_a=1,
    version_b=2
)

print(f"Expression Changed: {comparison.expression_changed}")
print(f"Added Inputs: {comparison.added_inputs}")
print(f"Removed Inputs: {comparison.removed_inputs}")
print(f"Performance Diff: {comparison.avg_time_diff_pct:+.1f}%")
```

## Formula Migration

### From YAML (CSRD esrs_formulas.yaml)

```python
from greenlang.formulas.migration import FormulaMigrator

migrator = FormulaMigrator(manager)

# Migrate all formulas from YAML
stats = migrator.migrate_from_yaml(
    yaml_path="GL-CSRD-APP/CSRD-Reporting-Platform/data/esrs_formulas.yaml",
    auto_activate=True
)

print(f"Migrated {stats['success']}/{stats['total']} formulas")
```

### From Python (CBAM emission_factors.py)

```python
# Migrate emission factors from Python module
stats = migrator.migrate_from_python(
    python_path="GL-CBAM-APP/CBAM-Importer-Copilot/data/emission_factors.py",
    auto_activate=True
)

print(f"Migrated {stats['success']} emission factors")
```

### Custom Formula Import

```python
# Define custom formulas
custom_formulas = [
    {
        'formula_code': 'BOILER_EFF_001',
        'formula_name': 'Boiler Thermal Efficiency',
        'category': 'efficiency',
        'formula_expression': '(energy_output / energy_input) * 100',
        'calculation_type': 'percentage',
        'required_inputs': ['energy_output', 'energy_input'],
        'output_unit': '%',
        'standard_reference': 'ASME PTC 4.1',
    },
    # ... more formulas
]

stats = migrator.migrate_custom_formulas(custom_formulas)
```

## CLI Usage

### List Formulas

```bash
# List all formulas
greenlang formula list

# Filter by category
greenlang formula list --category emissions

# Limit results
greenlang formula list --limit 20
```

### Show Formula Details

```bash
# Show active version
greenlang formula show E1-1

# Show specific version
greenlang formula show E1-1 --version 2
```

### List Versions

```bash
# List all versions of a formula
greenlang formula versions E1-1
```

### Activate Version

```bash
# Activate version
greenlang formula activate E1-1 --version 2

# Activate with effective date
greenlang formula activate E1-1 --version 2 --from-date 2025-01-01
```

### Rollback

```bash
# Rollback to previous version
greenlang formula rollback E1-1 --to-version 1
```

### Compare Versions

```bash
# Compare two versions
greenlang formula compare E1-1 --versions 1,2
```

### Execute Formula

```bash
# Execute formula with JSON input
greenlang formula execute E1-1 --input '{"stationary": 1000, "mobile": 500, "process": 200, "fugitive": 50}'

# Execute specific version
greenlang formula execute E1-1 --version 1 --input '{"stationary": 1000, "mobile": 500, "process": 200, "fugitive": 50}'
```

### Migrate Formulas

```bash
# Migrate from YAML
greenlang formula migrate esrs_formulas.yaml --type yaml

# Migrate from Python
greenlang formula migrate emission_factors.py --type python

# Don't auto-activate
greenlang formula migrate esrs_formulas.yaml --type yaml --no-auto-activate
```

## Supported Calculation Types

The formula engine supports the following calculation types:

### 1. Sum

```python
{
    'calculation_type': 'sum',
    'formula_expression': 'value1 + value2 + value3',
    'required_inputs': ['value1', 'value2', 'value3']
}
```

### 2. Subtraction

```python
{
    'calculation_type': 'subtraction',
    'formula_expression': 'total - adjustment',
    'required_inputs': ['total', 'adjustment']
}
```

### 3. Multiplication

```python
{
    'calculation_type': 'multiplication',
    'formula_expression': 'activity_data * emission_factor',
    'required_inputs': ['activity_data', 'emission_factor']
}
```

### 4. Division

```python
{
    'calculation_type': 'division',
    'formula_expression': 'emissions / revenue',
    'required_inputs': ['emissions', 'revenue']
}
```

### 5. Percentage

```python
{
    'calculation_type': 'percentage',
    'formula_expression': '(renewable / total) * 100',
    'required_inputs': ['renewable', 'total']
}
```

### 6. Custom Expression

```python
{
    'calculation_type': 'custom_expression',
    'formula_expression': 'value1 * 2 + value2 / 100',
    'required_inputs': ['value1', 'value2']
}
```

**Security Note**: Custom expressions are evaluated in a restricted Python environment with only safe math operations (no file I/O, imports, or dangerous functions).

## Dependency Management

Formulas can depend on other formulas for complex calculations:

```python
# Create base formula
manager.create_formula(
    formula_code="E1-1",
    formula_name="Scope 1 Emissions",
    category=FormulaCategory.EMISSIONS
)

# Create dependent formula
manager.create_formula(
    formula_code="E1-4",
    formula_name="Total GHG Emissions",
    category=FormulaCategory.EMISSIONS
)

# Add dependency
manager.add_dependency(
    formula_code="E1-4",
    version_number=1,
    depends_on_formula_code="E1-1",
    dependency_type="required"
)

# Engine will automatically resolve dependencies
result = manager.execute_formula(
    formula_code="E1-4",
    input_data={...}
)
```

## Database Schema

### Tables

- **formulas**: Formula metadata (code, name, category)
- **formula_versions**: Version-specific data (expression, inputs, validation)
- **formula_dependencies**: Formula dependency graph
- **formula_execution_log**: Complete execution audit trail
- **formula_ab_tests**: A/B test configurations (future)
- **formula_migration_log**: Migration tracking

### Views

- **v_active_formulas**: Currently active formulas
- **v_formula_dependencies**: Dependency tree

## Best Practices

### 1. Always Use Version Control

Never edit formulas in-place. Always create new versions:

```python
# ✅ CORRECT: Create new version
manager.create_new_version(
    formula_code="E1-1",
    formula_data=updated_data,
    change_notes="Fixed calculation bug in stationary emissions"
)

# ❌ WRONG: Don't try to edit existing versions
# (Repository doesn't even provide update methods)
```

### 2. Provide Detailed Change Notes

```python
# ✅ GOOD
change_notes = """
Fixed calculation bug where fugitive emissions were double-counted.

Changed from:
  stationary + mobile + process + fugitive + fugitive_refrigerants

To:
  stationary + mobile + process + (fugitive + fugitive_refrigerants)

Ticket: JIRA-1234
Reviewed by: jane.smith@company.com
"""

# ❌ BAD
change_notes = "fixed bug"
```

### 3. Test Before Activating

```python
# Create version
version_id = manager.create_new_version(
    formula_code="E1-1",
    formula_data=new_version_data,
    change_notes="Updated calculation",
    auto_activate=False  # Don't activate yet
)

# Test with sample data
result = manager.execute_formula(
    formula_code="E1-1",
    input_data=test_data,
    version_number=2  # Test specific version
)

# Verify result
assert result == expected_value

# Activate after successful testing
manager.activate_version("E1-1", version_number=2)
```

### 4. Use Effective Dates for Regulatory Changes

```python
# Activate new version on regulatory effective date
manager.activate_version(
    formula_code="E1-1",
    version_number=2,
    effective_from=date(2025, 7, 1)  # ESRS Set 2 effective date
)
```

### 5. Maintain Audit Trail

```python
# Always provide context in execution
result = manager.execute_formula_full(
    formula_code="E1-1",
    input_data=emissions_data,
    agent_name="EmissionsCalculatorAgent",
    calculation_id=f"CALC-{year}-{report_id}",
    user_id=current_user.email
)

# This creates complete audit trail in database
```

## Zero-Hallucination Guarantees

All formulas are executed deterministically with no LLM involvement in numeric calculations:

### ✅ ALLOWED (Deterministic)

```python
# Database lookups
emission_factor = db.lookup("natural_gas", "combustion")

# Python arithmetic
emissions = activity_data * emission_factor

# Formula evaluation
result = engine.execute("E1-1", inputs)
```

### ❌ NOT ALLOWED (Non-Deterministic)

```python
# LLM for numeric calculations
emissions = llm.calculate_emissions(data)  # NEVER

# Unvalidated external API calls
factor = external_api.get_factor()  # NO PROVENANCE

# ML model predictions for compliance values
value = ml_model.predict(features)  # NOT FOR REGULATORY
```

## Performance

### Caching

The execution engine uses caching for dependency resolution:

```python
# Dependencies are cached within execution context
engine.clear_cache()  # Clear when switching contexts
```

### Benchmarking

```python
# Get execution statistics
version = manager.get_version("E1-1", 1)
print(f"Average execution time: {version.avg_execution_time_ms:.2f}ms")
print(f"Total executions: {version.execution_count}")
```

### Database Optimization

The schema includes indexes on common query patterns:

- Formula code lookup
- Active version lookup
- Execution log queries
- Dependency resolution

## Troubleshooting

### Formula Not Found

```python
formula = manager.get_formula("E1-1")
if not formula:
    print("Formula doesn't exist. Create it first.")
    manager.create_formula(...)
```

### No Active Version

```python
active = manager.get_active_formula("E1-1")
if not active:
    print("No active version. Activate one:")
    manager.activate_version("E1-1", version_number=1)
```

### Execution Fails

```python
try:
    result = manager.execute_formula("E1-1", input_data)
except ValidationError as e:
    print(f"Input validation failed: {e}")
    # Check required inputs
    version = manager.get_active_formula("E1-1")
    print(f"Required inputs: {version.required_inputs}")
except ProcessingError as e:
    print(f"Execution failed: {e}")
```

## API Reference

See module docstrings for complete API documentation:

- `FormulaManager`: High-level management API
- `FormulaRepository`: Database access layer
- `FormulaExecutionEngine`: Execution engine
- `FormulaMigrator`: Import utilities

## Contributing

When adding new calculation types:

1. Add enum value to `CalculationType` in `models.py`
2. Implement calculation method in `FormulaExecutionEngine._execute_calculation()`
3. Add unit tests in `tests/test_formula_execution.py`
4. Update this documentation

## License

Copyright © 2025 GreenLang. All rights reserved.
