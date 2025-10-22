# Validation Rules - Quick Reference Guide

## 🚀 Get Started in 30 Seconds

```python
from greenlang.validation.rules import RulesEngine, Rule
import yaml

# Load rules
with open('examples/validation_rules/data_import_validation.yaml') as f:
    config = yaml.safe_load(f)

# Create engine and add rules
engine = RulesEngine()
for rule_dict in config['rules']:
    engine.add_rule(Rule(**rule_dict))

# Validate your data
result = engine.validate(your_data)

# Check result
if result.valid:
    print("✓ Validation passed!")
else:
    for error in result.errors:
        print(f"✗ {error}")
```

---

## 📁 Which File Should I Use?

| Your Task | Use This File | Rule Set |
|-----------|---------------|----------|
| Validating CSV/JSON imports | `data_import_validation.yaml` | `standard_validation` |
| Validating Scope 1 calculations | `calculation_validation.yaml` | `scope_1_validation` |
| Validating Scope 2 calculations | `calculation_validation.yaml` | `scope_2_validation` |
| Validating Scope 3 calculations | `calculation_validation.yaml` | `scope_3_validation` |
| Validating final reports | `report_validation.yaml` | `cdp_submission` or `iso_14064_compliance` |
| Quick checks only | Any file | `quick_validation` |

---

## 🔧 Common Code Snippets

### Load Specific Rule Set

```python
import yaml
from greenlang.validation.rules import RulesEngine, Rule

with open('calculation_validation.yaml') as f:
    config = yaml.safe_load(f)

# Get the scope_1_validation rule set
rule_set = next(rs for rs in config['rule_sets']
                if rs['name'] == 'scope_1_validation')

engine = RulesEngine()
for rule_name in rule_set['rules']:
    rule_dict = next(r for r in config['rules']
                     if r['name'] == rule_name)
    engine.add_rule(Rule(**rule_dict))

result = engine.validate(data)
```

### Integrate with GreenLang Agent

```python
from greenlang.sdk.base import Agent, Metadata
from greenlang.validation.rules import RulesEngine, Rule
import yaml

class ValidatedAgent(Agent):
    def __init__(self, validation_file):
        super().__init__(Metadata(id="validated", name="Validated Agent", version="1.0.0"))

        # Load validation rules
        with open(validation_file) as f:
            config = yaml.safe_load(f)

        self.engine = RulesEngine()
        for rule_dict in config['rules']:
            self.engine.add_rule(Rule(**rule_dict))

    def validate(self, data):
        result = self.engine.validate(data)
        if not result.valid:
            for error in result.errors:
                self.logger.error(str(error))
            return False
        return True

    def process(self, data):
        # Validation runs automatically
        return {"status": "success", "data": data}

# Usage
agent = ValidatedAgent('data_import_validation.yaml')
result = agent.run(your_data)
```

### Chain Multiple Validators

```python
from greenlang.validation.framework import ValidationFramework
import yaml

def load_yaml_validator(yaml_file):
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    engine = RulesEngine()
    for rule_dict in config['rules']:
        engine.add_rule(Rule(**rule_dict))

    return lambda data: engine.validate(data)

# Create framework
framework = ValidationFramework()
framework.add_validator("import", load_yaml_validator('data_import_validation.yaml'))
framework.add_validator("calc", load_yaml_validator('calculation_validation.yaml'))
framework.add_validator("report", load_yaml_validator('report_validation.yaml'))

# Validate through pipeline
result = framework.validate(data, validators=["import", "calc", "report"])
```

---

## 🎯 Operator Cheat Sheet

| Operator | Example | What It Checks |
|----------|---------|----------------|
| `not_null` | `field: "building_id"` | Field exists and is not null |
| `is_null` | `field: "optional_field"` | Field is null or missing |
| `==` | `value: 100` | Exactly equals 100 |
| `!=` | `value: 0` | Not equal to 0 |
| `>` | `value: 0` | Greater than 0 |
| `>=` | `value: 100` | Greater than or equal to 100 |
| `<` | `value: 1000` | Less than 1000 |
| `<=` | `value: 100` | Less than or equal to 100 |
| `in` | `value: ["A", "B", "C"]` | Value is in the list |
| `not_in` | `value: ["invalid"]` | Value is not in the list |
| `contains` | `value: "keyword"` | String contains "keyword" |
| `regex` | `value: "^\\d{4}$"` | Matches regex pattern |

---

## 🎨 Severity Level Quick Guide

```yaml
# ERROR: Critical - stops processing
severity: "error"
message: "Building ID is required"

# WARNING: Important - needs attention
severity: "warning"
message: "Data quality score is below threshold"

# INFO: Nice to have - recommendation
severity: "info"
message: "Consider adding location data for better accuracy"
```

---

## 💡 Common Validation Patterns

### Email Validation
```yaml
- name: "email_format"
  field: "contact_email"
  operator: "regex"
  value: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
  severity: "error"
  message: "Invalid email format"
```

### Date Validation (YYYY-MM-DD)
```yaml
- name: "date_format"
  field: "reporting_date"
  operator: "regex"
  value: "^\\d{4}-\\d{2}-\\d{2}$"
  severity: "error"
  message: "Date must be YYYY-MM-DD format"
```

### Positive Number
```yaml
- name: "value_positive"
  field: "energy_kwh"
  operator: ">"
  value: 0
  severity: "error"
  message: "Energy must be positive"
```

### Range Check
```yaml
- name: "percentage_range"
  field: "renewable_percent"
  operator: ">="
  value: 0
  severity: "error"

- name: "percentage_max"
  field: "renewable_percent"
  operator: "<="
  value: 100
  severity: "error"
```

### Enum Validation
```yaml
- name: "unit_valid"
  field: "energy_unit"
  operator: "in"
  value: ["kWh", "MWh", "GWh"]
  severity: "error"
  message: "Unit must be kWh, MWh, or GWh"
```

### Conditional Validation
```yaml
- name: "conditional_check"
  field: "renewable_kwh"
  operator: "not_null"
  severity: "warning"
  message: "Renewable kWh should be provided"
  condition: "exists:renewable_percentage"
```

---

## 🐛 Troubleshooting

### YAML Won't Load
```python
import yaml

# Check YAML syntax
with open('validation_rules.yaml') as f:
    try:
        config = yaml.safe_load(f)
        print("✓ YAML is valid")
    except yaml.YAMLError as e:
        print(f"✗ YAML error: {e}")
```

### Rules Not Matching
```python
# Test a single rule
engine = RulesEngine()
rule = Rule(
    name="test",
    field="your_field",
    operator=">=",
    value=0
)
engine.add_rule(rule)

test_data = {"your_field": -1}
result = engine.validate(test_data)

print(f"Valid: {result.valid}")
if not result.valid:
    print(f"Error: {result.errors[0].message}")
```

### Regex Not Working
```python
import re

# Test regex separately
pattern = r"^\d{4}-\d{2}-\d{2}$"
test_value = "2024-01-15"

if re.match(pattern, test_value):
    print("✓ Pattern matches")
else:
    print("✗ Pattern doesn't match")
```

---

## 📊 Sample Data Templates

### For data_import_validation.yaml
```json
{
  "building_id": "NYC-001234",
  "facility_name": "Office Tower",
  "building_area_sqft": 50000,
  "building_type": "Office",
  "energy_data": {
    "electricity_kwh": 125000,
    "electricity_unit": "kWh",
    "gas_therms": 5000,
    "gas_unit": "therms"
  },
  "location": {
    "city": "New York",
    "country_code": "US"
  },
  "reporting_year": 2024,
  "reporting_date": "2024-01-15",
  "contact_email": "contact@example.com"
}
```

### For calculation_validation.yaml
```json
{
  "calculation_id": "CALC-2024-001",
  "facility_id": "FAC-001",
  "energy_consumption": 125000,
  "energy_unit": "kWh",
  "emission_factor": 0.45,
  "emission_unit": "kg_co2e",
  "calculation_method": "activity_based",
  "emission_scope": "scope_2",
  "reporting_period_start": "2024-01-01",
  "reporting_period_end": "2024-12-31",
  "grid_emission_factor": 0.45,
  "renewable_percentage": 25
}
```

### For report_validation.yaml
```json
{
  "report_metadata": {
    "title": "2024 Carbon Emissions Report",
    "version": "1.0",
    "report_date": "2024-12-31",
    "report_id": "RPT2024001"
  },
  "organization": {
    "name": "Example Corp"
  },
  "emissions_data": {
    "total_emissions_tonnes_co2e": 5000,
    "scope_1_total": 1000,
    "scope_2_total": 3000,
    "scope_3_total": 1000,
    "units": "tonnes CO2e"
  },
  "methodology": {
    "calculation_approach": "GHG Protocol",
    "organizational_boundary": "Operational control",
    "boundary_approach": "operational_control"
  },
  "compliance": {
    "reporting_standard": "GHG Protocol"
  }
}
```

---

## 🔗 Quick Links

- **Full Documentation**: `README.md`
- **Summary**: `SUMMARY.md`
- **Test Script**: `test_validation_rules.py`
- **Data Import Rules**: `data_import_validation.yaml`
- **Calculation Rules**: `calculation_validation.yaml`
- **Report Rules**: `report_validation.yaml`

---

## ⚡ Pro Tips

1. **Start with Quick Validation**: Use `quick_validation` rule set first, then add more rules
2. **Test Incrementally**: Add rules one at a time and test
3. **Use Appropriate Severity**: Don't make everything an error
4. **Clear Messages**: Write error messages that explain how to fix the issue
5. **Conditional Rules**: Use conditions to avoid false positives on optional fields
6. **Document Custom Rules**: Add comments explaining why custom rules exist

---

**Need More Help?** See the full README.md for detailed documentation and examples.
