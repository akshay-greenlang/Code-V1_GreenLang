# Formula Versioning System - Quick Reference

## Installation

```bash
pip install greenlang
```

## Python API Quick Start

```python
from greenlang.formulas import FormulaManager
from greenlang.formulas.models import FormulaCategory

# Initialize
manager = FormulaManager("formulas.db")

# Create formula
manager.create_formula(
    formula_code="E1-1",
    formula_name="Total Scope 1 GHG Emissions",
    category=FormulaCategory.EMISSIONS
)

# Create version
version_data = {
    'formula_expression': 'stationary + mobile + process + fugitive',
    'calculation_type': 'sum',
    'required_inputs': ['stationary', 'mobile', 'process', 'fugitive'],
    'output_unit': 'tCO2e',
}
manager.create_new_version("E1-1", version_data, "Initial", auto_activate=True)

# Execute
result = manager.execute_formula(
    "E1-1",
    {'stationary': 1000, 'mobile': 500, 'process': 200, 'fugitive': 50}
)

print(f"Result: {result} tCO2e")  # Output: 1750 tCO2e
```

## CLI Commands

```bash
# List formulas
greenlang formula list
greenlang formula list --category emissions

# Show details
greenlang formula show E1-1
greenlang formula show E1-1 --version 2

# List versions
greenlang formula versions E1-1

# Execute
greenlang formula execute E1-1 --input '{"stationary": 1000, "mobile": 500, "process": 200, "fugitive": 50}'

# Activate version
greenlang formula activate E1-1 --version 2
greenlang formula activate E1-1 --version 2 --from-date 2025-01-01

# Rollback
greenlang formula rollback E1-1 --to-version 1

# Compare
greenlang formula compare E1-1 --versions 1,2

# Migrate
greenlang formula migrate esrs_formulas.yaml --type yaml
greenlang formula migrate emission_factors.py --type python
```

## Calculation Types

| Type | Expression Example | Use Case |
|------|-------------------|----------|
| `sum` | `value1 + value2 + value3` | Total emissions |
| `subtraction` | `total - adjustment` | Net values |
| `multiplication` | `activity * factor` | Emissions calculation |
| `division` | `emissions / revenue` | Intensity metrics |
| `percentage` | `(part / total) * 100` | Percentages |
| `custom_expression` | `value1 * 2 + value2 / 100` | Complex formulas |

## Version Management

```python
# Create new version
manager.create_new_version(
    formula_code="E1-1",
    formula_data={...},
    change_notes="Updated calculation",
    auto_activate=False  # Test first
)

# Test new version
result = manager.execute_formula(
    "E1-1",
    input_data={...},
    version_number=2  # Specific version
)

# Activate after testing
manager.activate_version("E1-1", version_number=2)

# Rollback if needed
manager.rollback_to_version("E1-1", version_number=1)
```

## Migration

```python
from greenlang.formulas.migration import FormulaMigrator

migrator = FormulaMigrator(manager)

# From YAML
stats = migrator.migrate_from_yaml("esrs_formulas.yaml")

# From Python
stats = migrator.migrate_from_python("emission_factors.py")

# Custom
custom_formulas = [
    {
        'formula_code': 'CUSTOM-001',
        'formula_name': 'Custom Formula',
        'category': 'emissions',
        'formula_expression': 'value1 + value2',
        'calculation_type': 'sum',
        'required_inputs': ['value1', 'value2'],
        'output_unit': 'tCO2e',
    }
]
stats = migrator.migrate_custom_formulas(custom_formulas)

print(f"Migrated: {stats['success']}/{stats['total']}")
```

## Common Patterns

### Get Active Formula
```python
active = manager.get_active_formula("E1-1")
print(f"Version: {active.version_number}")
print(f"Expression: {active.formula_expression}")
```

### Execute with Provenance
```python
result = manager.execute_formula_full(
    "E1-1",
    input_data={...},
    agent_name="EmissionsAgent",
    calculation_id="CALC-2025-001",
    user_id="user@company.com"
)

print(f"Output: {result.output_value}")
print(f"Input Hash: {result.input_hash}")
print(f"Execution Time: {result.execution_time_ms}ms")
```

### Compare Versions
```python
comparison = manager.compare_versions("E1-1", version_a=1, version_b=2)

if comparison.expression_changed:
    print("Expression changed!")
if comparison.inputs_changed:
    print(f"Added: {comparison.added_inputs}")
    print(f"Removed: {comparison.removed_inputs}")
```

### List All Versions
```python
versions = manager.list_versions("E1-1")
for v in versions:
    print(f"v{v.version_number}: {v.version_status} - {v.change_notes}")
```

## Database Locations

```bash
# Development
~/.greenlang/formulas.db

# Custom location (CLI)
greenlang formula list --db /path/to/formulas.db

# Custom location (Python)
manager = FormulaManager("/path/to/formulas.db")

# Environment variable
export GREENLANG_FORMULA_DB=/path/to/formulas.db
```

## Error Handling

```python
from greenlang.exceptions import ValidationError, ProcessingError

try:
    result = manager.execute_formula("E1-1", input_data)
except ValidationError as e:
    print(f"Input validation failed: {e}")
    # Check required inputs
    version = manager.get_active_formula("E1-1")
    print(f"Required: {version.required_inputs}")
except ProcessingError as e:
    print(f"Execution failed: {e}")
```

## Formula Categories

- `emissions` - GHG emissions calculations
- `energy` - Energy consumption and renewable %
- `water` - Water consumption and recycling
- `waste` - Waste generation and diversion
- `workforce` - Employee metrics
- `efficiency` - Efficiency calculations
- `cost` - Cost and savings
- `compliance` - Regulatory compliance
- `utility` - Unit conversions and helpers

## File Locations

```
greenlang/formulas/
├── __init__.py           # Import from here
├── manager.py            # FormulaManager class
├── models.py             # Pydantic models
├── repository.py         # Database access
├── engine.py             # Execution engine
├── migration.py          # FormulaMigrator
├── cli.py                # CLI commands
├── schema.sql            # Database schema
├── README.md             # Full documentation
└── examples/
    ├── basic_usage.py    # Usage examples
    └── migrate_existing.py  # Migration examples
```

## Best Practices

1. **Always version**: Never edit in-place, create new versions
2. **Test before activate**: Use `auto_activate=False`, test, then activate
3. **Detailed change notes**: Document what changed and why
4. **Use effective dates**: For regulatory changes
5. **Maintain audit trail**: Always provide agent_name, user_id, calculation_id
6. **Validate inputs**: Use validation_rules in version_data
7. **Zero-hallucination**: Never use LLM for numeric calculations

## Support

- **Documentation**: `greenlang/formulas/README.md`
- **Examples**: `greenlang/formulas/examples/`
- **Tests**: `greenlang/formulas/tests/`
- **Schema**: `greenlang/formulas/schema.sql`

---

**Quick Help**:
```bash
greenlang formula --help
greenlang formula list --help
greenlang formula execute --help
```
