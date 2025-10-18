# GreenLang Validation Rules Examples

This directory contains comprehensive YAML validation rule files demonstrating the full capabilities of the GreenLang Validation Framework. These production-ready examples can be used directly or customized for your specific carbon accounting and emissions reporting needs.

## 📁 Files Overview

### 1. `data_import_validation.yaml`
**Purpose**: Validate imported data from CSV/JSON sources

**Use Cases**:
- CSV file imports
- JSON data ingestion
- API data validation
- Bulk data uploads
- Data quality checks during ETL processes

**Key Features**:
- Required field validation
- Data type and format validation (email, date, phone)
- Enumeration validation (building types, units, country codes)
- Numeric range constraints
- String content validation
- Cross-field business logic
- Conditional validation rules

**Rule Count**: 70+ comprehensive validation rules

---

### 2. `calculation_validation.yaml`
**Purpose**: Validate carbon calculation inputs and parameters

**Use Cases**:
- Scope 1, 2, and 3 emissions calculations
- Carbon footprint analysis
- Energy modeling inputs
- Emission factor validation
- Unit conversion verification

**Key Features**:
- Numeric value range validation
- Unit consistency checks
- Calculation method validation
- Precision requirements
- Business logic rules (energy > 0, factor validation)
- Fuel-specific validation
- Time period validation
- Reference data validation

**Rule Count**: 90+ calculation-specific rules

---

### 3. `report_validation.yaml`
**Purpose**: Validate carbon emissions reports and disclosures

**Use Cases**:
- GHG Inventory Reports
- Carbon Disclosure Project (CDP) submissions
- TCFD disclosures
- Sustainability reports
- ISO 14064 compliant reporting
- Annual carbon footprint reports

**Key Features**:
- Required section validation
- Data quality checks
- Aggregation validation
- Compliance and standards validation
- Format and structure validation
- Intensity metrics validation
- Supporting documentation checks

**Rule Count**: 80+ report validation rules

---

## 🚀 Quick Start

### Basic Usage

```python
from greenlang.validation.rules import RulesEngine, Rule
import yaml

# Load validation rules from YAML file
with open('examples/validation_rules/data_import_validation.yaml') as f:
    config = yaml.safe_load(f)

# Create rules engine
engine = RulesEngine()

# Add rules from configuration
for rule_dict in config['rules']:
    rule = Rule(**rule_dict)
    engine.add_rule(rule)

# Validate your data
data = {
    "building_id": "NYC-001234",
    "facility_name": "Corporate Office Tower",
    "building_area_sqft": 50000,
    "energy_data": {
        "electricity_kwh": 125000,
        "electricity_unit": "kWh",
        "gas_therms": 5000,
        "gas_unit": "therms"
    },
    "reporting_year": 2024
}

result = engine.validate(data)

# Check results
if result.valid:
    print("✓ Validation passed!")
else:
    print(f"✗ Validation failed with {len(result.errors)} errors:")
    for error in result.errors:
        print(f"  - {error}")
```

### Using Rule Sets

Each YAML file includes predefined rule sets for different scenarios:

```python
# Load specific rule set
with open('examples/validation_rules/calculation_validation.yaml') as f:
    config = yaml.safe_load(f)

engine = RulesEngine()

# Load only Scope 1 validation rules
scope_1_rules = next(rs for rs in config['rule_sets'] if rs['name'] == 'scope_1_validation')
for rule_name in scope_1_rules['rules']:
    # Find and add the rule
    rule_dict = next(r for r in config['rules'] if r['name'] == rule_name)
    rule = Rule(**rule_dict)
    engine.add_rule(rule)

# Validate Scope 1 calculation inputs
calculation_data = {
    "fuel_type": "natural_gas",
    "fuel_quantity": 10000,
    "emission_factor": 0.0053,
    "calculation_method": "fuel_based",
    "emission_scope": "scope_1",
    "heating_value": 1026,
    "reporting_period_start": "2024-01-01",
    "reporting_period_end": "2024-12-31"
}

result = engine.validate(calculation_data)
```

### Integration with Validation Framework

```python
from greenlang.validation.framework import ValidationFramework, ValidationResult
from greenlang.validation.rules import RulesEngine, Rule
import yaml

# Create validation framework
framework = ValidationFramework()

# Load and register YAML-based validators
def create_yaml_validator(yaml_file_path):
    with open(yaml_file_path) as f:
        config = yaml.safe_load(f)

    engine = RulesEngine()
    for rule_dict in config['rules']:
        rule = Rule(**rule_dict)
        engine.add_rule(rule)

    return lambda data: engine.validate(data)

# Register validators
framework.add_validator(
    "data_import",
    create_yaml_validator('examples/validation_rules/data_import_validation.yaml')
)

framework.add_validator(
    "calculation",
    create_yaml_validator('examples/validation_rules/calculation_validation.yaml')
)

framework.add_validator(
    "report",
    create_yaml_validator('examples/validation_rules/report_validation.yaml')
)

# Validate through framework
result = framework.validate(your_data, validators=["data_import", "calculation"])
```

---

## 📖 Rule Operators Reference

All 12 validation operators are demonstrated across the example files:

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equals | `value: 100` matches exactly 100 |
| `!=` | Not equals | `value: 0` fails if value is 0 |
| `>` | Greater than | `value: 0` requires value > 0 |
| `>=` | Greater than or equal | `value: 100` requires value >= 100 |
| `<` | Less than | `value: 1000` requires value < 1000 |
| `<=` | Less than or equal | `value: 100` requires value <= 100 |
| `in` | In list/set | `value: ["A", "B", "C"]` |
| `not_in` | Not in list/set | `value: ["invalid", "error"]` |
| `contains` | String contains | `value: "keyword"` in field |
| `regex` | Regex pattern match | `value: "^\\d{4}-\\d{2}-\\d{2}$"` |
| `is_null` | Field is null | Checks if field is null/missing |
| `not_null` | Field is not null | Checks if field exists and is not null |

---

## 🎯 Severity Levels

Each rule can have one of three severity levels:

- **`error`**: Critical validation failure - data should not be processed
- **`warning`**: Non-critical issue - data can be processed but needs attention
- **`info`**: Informational message - data quality could be improved

```yaml
rules:
  - name: "required_field"
    field: "building_id"
    operator: "not_null"
    severity: "error"  # Critical - stops processing

  - name: "recommended_field"
    field: "location.postal_code"
    operator: "not_null"
    severity: "warning"  # Important but not critical

  - name: "optional_enhancement"
    field: "building_year_built"
    operator: "not_null"
    severity: "info"  # Nice to have
```

---

## 🔧 Conditional Rules

Rules can be conditionally applied based on the presence of other fields:

```yaml
rules:
  - name: "renewable_data_when_percentage_exists"
    field: "energy_data.renewable_kwh"
    operator: "not_null"
    severity: "warning"
    message: "Renewable kWh should be provided when renewable percentage is specified"
    enabled: true
    condition: "exists:renewable_energy_percentage"
```

The `condition` parameter uses simplified expressions:
- `exists:field_name` - Only apply rule if field exists
- More complex conditions can be added via custom code

---

## 📊 Pre-defined Rule Sets

Each YAML file includes organized rule sets for common scenarios:

### `data_import_validation.yaml`
- **quick_validation**: Essential validations only
- **standard_validation**: Comprehensive production import validation
- **strict_validation**: All rules including warnings and info
- **format_validation**: Format and type checks only
- **range_validation**: Numeric range checks only

### `calculation_validation.yaml`
- **scope_1_validation**: Scope 1 direct emissions
- **scope_2_validation**: Scope 2 purchased electricity
- **scope_3_validation**: Scope 3 value chain emissions
- **precision_validation**: High precision audited calculations
- **quick_validation**: Essential calculation checks

### `report_validation.yaml`
- **basic_report**: Minimum valid report requirements
- **cdp_submission**: CDP reporting requirements
- **iso_14064_compliance**: ISO 14064-1 compliance
- **comprehensive_report**: All validation rules
- **quick_quality_check**: Fast draft report validation

---

## 💡 Customization Guide

### Adding Custom Rules

Add new rules to existing YAML files:

```yaml
rules:
  # Your existing rules...

  # Add a new custom rule
  - name: "custom_energy_threshold"
    field: "energy_data.electricity_kwh"
    operator: "<="
    value: 500000
    severity: "warning"
    message: "Electricity consumption exceeds your organization's threshold"
    enabled: true
    condition: "exists:energy_data.electricity_kwh"
```

### Creating Custom Rule Sets

Define your own rule sets for specific workflows:

```yaml
rule_sets:
  # Add your custom rule set
  - name: "my_custom_validation"
    description: "Custom validation for specific use case"
    enabled: true
    rules:
      - building_id_required
      - energy_consumption_positive
      - custom_energy_threshold  # Your custom rule
```

### Modifying Severity Levels

Adjust severity levels to match your organization's requirements:

```yaml
rules:
  - name: "scope_3_emissions_check"
    field: "emissions_data.scope_3_total"
    operator: "not_null"
    severity: "error"  # Changed from "warning" to make it mandatory
    message: "Scope 3 emissions are required by our organization"
    enabled: true
```

### Disabling Rules

Temporarily disable rules without deleting them:

```yaml
rules:
  - name: "optional_check"
    field: "some_field"
    operator: "not_null"
    severity: "info"
    message: "Optional check"
    enabled: false  # Rule is defined but not executed
```

---

## 🧪 Testing Your Validation Rules

### Test Script Template

```python
#!/usr/bin/env python3
"""
Test validation rules with sample data
"""
from greenlang.validation.rules import RulesEngine, Rule
import yaml
import json

def test_validation_rules(yaml_file, test_data_file):
    """Test validation rules against sample data"""

    # Load rules
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    engine = RulesEngine()
    for rule_dict in config['rules']:
        rule = Rule(**rule_dict)
        engine.add_rule(rule)

    # Load test data
    with open(test_data_file) as f:
        test_data = json.load(f)

    # Run validation
    result = engine.validate(test_data)

    # Report results
    print(f"\n{'='*70}")
    print(f"Validation Results for: {yaml_file}")
    print(f"{'='*70}")
    print(f"Status: {'PASSED ✓' if result.valid else 'FAILED ✗'}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Info: {len(result.info)}")

    if result.errors:
        print(f"\n{'Errors:':-^70}")
        for error in result.errors:
            print(f"  ✗ [{error.field}] {error.message}")

    if result.warnings:
        print(f"\n{'Warnings:':-^70}")
        for warning in result.warnings:
            print(f"  ⚠ [{warning.field}] {warning.message}")

    return result

# Run tests
if __name__ == "__main__":
    test_validation_rules(
        'examples/validation_rules/data_import_validation.yaml',
        'test_data/sample_building_data.json'
    )
```

### Sample Test Data

Create test data files to validate your rules:

```json
{
  "building_id": "NYC-001234",
  "facility_name": "Corporate Office Tower",
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
    "country_code": "US",
    "postal_code": "10001"
  },
  "reporting_year": 2024,
  "contact_email": "facilities@example.com"
}
```

---

## 🔍 Common Patterns and Best Practices

### 1. Validate Required Fields First

Order rules to check required fields before validating their content:

```yaml
rules:
  # First: Check field exists
  - name: "email_required"
    field: "contact_email"
    operator: "not_null"
    severity: "error"
    message: "Email is required"

  # Then: Validate format (with condition)
  - name: "email_format"
    field: "contact_email"
    operator: "regex"
    value: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    severity: "error"
    message: "Email must be valid format"
    condition: "exists:contact_email"
```

### 2. Use Appropriate Severity Levels

- **error**: Data quality issues that prevent processing
- **warning**: Data quality issues that need attention but don't block processing
- **info**: Suggestions for improvement

### 3. Provide Clear Error Messages

```yaml
# Bad: Generic message
message: "Validation failed"

# Good: Specific, actionable message
message: "Building area must be between 100 and 50M square feet"

# Better: Includes context and expected value
message: "Electricity consumption of {value} kWh exceeds maximum threshold of 100M kWh - verify data accuracy"
```

### 4. Group Related Rules

Use descriptive comments to organize rules:

```yaml
rules:
  # ----- Building Identification -----
  - name: "building_id_required"
    # ...

  - name: "facility_name_required"
    # ...

  # ----- Energy Data Fields -----
  - name: "energy_data_required"
    # ...
```

### 5. Use Conditional Rules for Optional Fields

```yaml
# Only validate format if field is provided
- name: "phone_format"
  field: "contact_phone"
  operator: "regex"
  value: "^\\+?[1-9]\\d{1,14}$"
  severity: "warning"
  message: "Phone should be in international format"
  condition: "exists:contact_phone"
```

---

## 📚 Integration Examples

### With GreenLang Agents

```python
from greenlang.sdk.base import Agent, Metadata, Result
from greenlang.validation.rules import RulesEngine, Rule
import yaml

class ValidatedDataImporter(Agent):
    """Data importer with YAML-based validation"""

    def __init__(self, validation_rules_path):
        metadata = Metadata(
            id="validated_importer",
            name="Validated Data Importer",
            version="1.0.0"
        )
        super().__init__(metadata)

        # Load validation rules
        with open(validation_rules_path) as f:
            config = yaml.safe_load(f)

        self.engine = RulesEngine()
        for rule_dict in config['rules']:
            rule = Rule(**rule_dict)
            self.engine.add_rule(rule)

    def validate(self, input_data):
        """Validate input data"""
        result = self.engine.validate(input_data)

        if not result.valid:
            for error in result.errors:
                self.logger.error(f"Validation error: {error}")
            return False

        for warning in result.warnings:
            self.logger.warning(f"Validation warning: {warning}")

        return True

    def process(self, input_data):
        """Process validated data"""
        # Validation happens automatically via validate() method
        return {
            "status": "success",
            "data": input_data,
            "message": "Data imported and validated successfully"
        }

# Usage
agent = ValidatedDataImporter('examples/validation_rules/data_import_validation.yaml')
result = agent.run(your_data)
```

### With Pipeline Validation

```python
from greenlang.sdk.pipeline import Pipeline
from greenlang.validation.rules import RulesEngine, Rule
import yaml

# Create validation stage
def create_validation_stage(yaml_file):
    with open(yaml_file) as f:
        config = yaml.safe_load(f)

    engine = RulesEngine()
    for rule_dict in config['rules']:
        rule = Rule(**rule_dict)
        engine.add_rule(rule)

    def validate_stage(data):
        result = engine.validate(data)
        if not result.valid:
            raise ValueError(f"Validation failed: {result.get_summary()}")
        return data

    return validate_stage

# Build pipeline
pipeline = Pipeline()
pipeline.add_stage("import", import_data_function)
pipeline.add_stage("validate", create_validation_stage('data_import_validation.yaml'))
pipeline.add_stage("calculate", calculate_emissions_function)
pipeline.add_stage("validate_report", create_validation_stage('report_validation.yaml'))
pipeline.add_stage("export", export_report_function)

# Run pipeline
result = pipeline.run(input_data)
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Rules not loading from YAML
```python
# Solution: Check YAML syntax
import yaml

with open('validation_rules.yaml') as f:
    try:
        config = yaml.safe_load(f)
        print("YAML is valid")
    except yaml.YAMLError as e:
        print(f"YAML syntax error: {e}")
```

**Issue**: Regex patterns not matching
```python
# Solution: Test regex patterns separately
import re

pattern = r"^\d{4}-\d{2}-\d{2}$"
test_value = "2024-01-15"

if re.match(pattern, test_value):
    print("Pattern matches")
else:
    print("Pattern does not match")
```

**Issue**: Conditional rules not working
```python
# Ensure condition syntax is correct
# Supported: "exists:field_name"
# The field path should match your data structure

condition: "exists:energy_data.electricity_kwh"  # Nested field
condition: "exists:building_id"  # Top-level field
```

---

## 📖 Additional Resources

- **GreenLang Validation Framework Documentation**: `/docs/validation/framework.md`
- **Rules Engine API Reference**: `/docs/validation/rules.md`
- **Example Integration Code**: `/examples/06_validation_framework.py`
- **Test Suite**: `/tests/unit/validation/`

---

## 🤝 Contributing

To contribute new validation rules or improve existing ones:

1. Follow the YAML structure and naming conventions
2. Include comprehensive comments explaining each rule
3. Test rules with sample data
4. Update this README with any new patterns or examples
5. Submit a pull request with your changes

---

## 📄 License

These validation rule examples are part of the GreenLang framework and are subject to the same license terms.

---

## 📞 Support

For questions or issues with validation rules:
- Review the GreenLang documentation
- Check existing test cases for examples
- Consult the validation framework source code
- Reach out to the GreenLang team

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0
**Maintainer**: GreenLang Validation Framework Team
