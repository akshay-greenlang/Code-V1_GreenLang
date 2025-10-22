# Validation Rules Examples - Summary

## Files Created

### 1. **data_import_validation.yaml** (567 lines, 18 KB)
Comprehensive validation rules for CSV/JSON data imports.

**Statistics:**
- **Total Rules**: 70+
- **Operators Demonstrated**: All 12 (==, !=, >, >=, <, <=, in, not_in, contains, regex, is_null, not_null)
- **Severity Levels**: Error, Warning, Info
- **Rule Sets**: 5 predefined sets
  - quick_validation
  - standard_validation
  - strict_validation
  - format_validation
  - range_validation

**Key Sections:**
1. Required Field Validation (not_null)
2. Data Type and Format Validation (regex, contains)
3. Enumeration Validation (in, not_in)
4. Numeric Range Validation (>, >=, <, <=, ==, !=)
5. Data Completeness Validation (not_null, is_null)
6. Cross-Field Business Logic
7. String Length and Content Validation
8. Conditional Validation Rules

**Sample Rules:**
- Building ID format validation
- Email format validation (regex)
- Date format validation (YYYY-MM-DD)
- Building type enumeration
- Area range constraints (100 - 50M sqft)
- Energy consumption validation
- Country code validation

---

### 2. **calculation_validation.yaml** (831 lines, 24 KB)
Validation rules for carbon calculation inputs and parameters.

**Statistics:**
- **Total Rules**: 90+
- **Operators Demonstrated**: All 12
- **Severity Levels**: Error, Warning, Info
- **Rule Sets**: 5 predefined sets
  - scope_1_validation (Direct emissions)
  - scope_2_validation (Purchased electricity)
  - scope_3_validation (Value chain emissions)
  - precision_validation (High precision audited)
  - quick_validation (Essential checks)

**Key Sections:**
1. Numeric Value Validation
2. Unit Validation (energy, mass, volume, distance, emissions)
3. Calculation Method Validation
4. Precision and Accuracy Requirements
5. Business Logic Rules (GWP, grid factors, renewable %)
6. Fuel-Specific Validation
7. Time Period Validation
8. Reference and Source Data Validation
9. Null/Missing Value Validation

**Sample Rules:**
- Energy consumption > 0
- Emission factor validation (0 - 1000 range)
- Unit enumeration (kWh, MWh, GWh, etc.)
- Calculation method validation (activity_based, spend_based, etc.)
- Emission scope validation (Scope 1, 2, 3)
- Grid emission factor range (0 - 2.0 kg CO2e/kWh)
- Renewable percentage (0-100%)
- GWP timeframe validation (20, 100, 500 years)

---

### 3. **report_validation.yaml** (807 lines, 26 KB)
Validation rules for carbon emissions reports and disclosures.

**Statistics:**
- **Total Rules**: 80+
- **Operators Demonstrated**: All 12
- **Severity Levels**: Error, Warning, Info
- **Rule Sets**: 5 predefined sets
  - basic_report (Minimum requirements)
  - cdp_submission (CDP reporting)
  - iso_14064_compliance (ISO 14064-1)
  - comprehensive_report (All rules)
  - quick_quality_check (Fast draft validation)

**Key Sections:**
1. Required Report Sections
2. Data Quality Validation
3. Numeric Validation for Emissions Data
4. Aggregation and Breakdown Validation
5. Compliance and Standards Validation
6. Format and Structure Validation
7. Contextual Information Validation
8. Attachments and Supporting Documentation

**Sample Rules:**
- Report title and metadata requirements
- Scope 1, 2, 3 emissions required
- Data completeness score (>85%)
- Total emissions validation (non-negative)
- Year-over-year change validation
- Reporting standard validation (GHG Protocol, ISO 14064, etc.)
- Report ID format validation
- Emissions units validation
- Intensity metrics validation

---

### 4. **README.md** (709 lines, 19 KB)
Comprehensive documentation and usage guide.

**Contents:**
1. Files Overview
2. Quick Start Guide
3. Basic Usage Examples
4. Rule Sets Usage
5. Integration Examples
6. Rule Operators Reference (all 12 operators)
7. Severity Levels Explanation
8. Conditional Rules Guide
9. Pre-defined Rule Sets Documentation
10. Customization Guide
11. Testing Guidelines
12. Common Patterns and Best Practices
13. Integration Examples (with Agents and Pipelines)
14. Troubleshooting
15. Additional Resources

---

### 5. **test_validation_rules.py** (Python test script)
Automated test suite for validation rules.

**Features:**
- YAML syntax validation
- Rule loading tests
- All 12 operators demonstration
- Sample data validation
- Comprehensive test reporting

**Test Phases:**
1. YAML Syntax Validation
2. Rule Loading into RulesEngine
3. Operator Demonstration (all 12)
4. Sample Data Validation

---

## Operator Coverage Matrix

All files demonstrate all 12 rule operators:

| Operator | data_import | calculation | report | Description |
|----------|------------|-------------|--------|-------------|
| == | ✓ | ✓ | ✓ | Equals |
| != | ✓ | ✓ | ✓ | Not equals |
| > | ✓ | ✓ | ✓ | Greater than |
| >= | ✓ | ✓ | ✓ | Greater than or equal |
| < | ✓ | ✓ | ✓ | Less than |
| <= | ✓ | ✓ | ✓ | Less than or equal |
| in | ✓ | ✓ | ✓ | In list/set |
| not_in | ✓ | ✓ | ✓ | Not in list/set |
| contains | ✓ | ✓ | ✓ | String contains |
| regex | ✓ | ✓ | ✓ | Regex pattern match |
| is_null | ✓ | ✓ | ✓ | Field is null |
| not_null | ✓ | ✓ | ✓ | Field is not null |

---

## Severity Level Distribution

### data_import_validation.yaml
- **Error**: ~35 rules (critical failures)
- **Warning**: ~25 rules (needs attention)
- **Info**: ~10 rules (recommendations)

### calculation_validation.yaml
- **Error**: ~50 rules (invalid calculations)
- **Warning**: ~30 rules (questionable values)
- **Info**: ~10 rules (best practices)

### report_validation.yaml
- **Error**: ~30 rules (incomplete reports)
- **Warning**: ~30 rules (missing recommended sections)
- **Info**: ~20 rules (quality enhancements)

---

## Rule Set Coverage

Each file includes 5 predefined rule sets ranging from quick validation (essential rules only) to comprehensive validation (all rules).

### Quick Start for Each File

**Data Import:**
```python
# Use standard_validation rule set for production imports
```

**Calculations:**
```python
# Use scope_1_validation, scope_2_validation, or scope_3_validation
# based on calculation type
```

**Reports:**
```python
# Use cdp_submission or iso_14064_compliance for specific standards
# Use basic_report for minimum viable reports
```

---

## Key Features Demonstrated

1. **Nested Field Validation**: Rules can validate nested fields (e.g., `energy_data.electricity_kwh`)

2. **Conditional Rules**: Rules with `condition: "exists:field_name"` only apply when field exists

3. **Custom Error Messages**: Each rule has descriptive, actionable error messages

4. **Comprehensive Comments**: Every section is thoroughly documented with inline comments

5. **Production-Ready**: Files are immediately usable with the ValidationFramework

6. **Standards Compliance**: Rules aligned with GHG Protocol, ISO 14064, CDP, and other standards

7. **Industry Best Practices**: Validation rules based on carbon accounting industry standards

8. **Extensible**: Easy to add custom rules or modify existing ones

---

## Usage Examples

### Load and Use Single File
```python
from greenlang.validation.rules import RulesEngine, Rule
import yaml

with open('data_import_validation.yaml') as f:
    config = yaml.safe_load(f)

engine = RulesEngine()
for rule_dict in config['rules']:
    rule = Rule(**rule_dict)
    engine.add_rule(rule)

result = engine.validate(your_data)
```

### Load Specific Rule Set
```python
# Load only the 'quick_validation' rule set
quick_rules = next(rs for rs in config['rule_sets']
                   if rs['name'] == 'quick_validation')

engine = RulesEngine()
for rule_name in quick_rules['rules']:
    rule_dict = next(r for r in config['rules']
                     if r['name'] == rule_name)
    rule = Rule(**rule_dict)
    engine.add_rule(rule)
```

### Integrate with ValidationFramework
```python
from greenlang.validation.framework import ValidationFramework

framework = ValidationFramework()
framework.add_validator("import", create_validator('data_import_validation.yaml'))
framework.add_validator("calc", create_validator('calculation_validation.yaml'))
framework.add_validator("report", create_validator('report_validation.yaml'))

result = framework.validate(data, validators=["import", "calc"])
```

---

## Real-World Applications

### 1. Data Import Pipeline
Use `data_import_validation.yaml` to validate:
- CSV file uploads
- API data ingestion
- Bulk data imports
- User-submitted forms

### 2. Calculation Engine
Use `calculation_validation.yaml` to validate:
- Carbon footprint calculations
- Emissions factor applications
- Energy conversions
- Scope 1, 2, 3 calculations

### 3. Report Generation
Use `report_validation.yaml` to validate:
- Annual GHG inventories
- CDP submissions
- Sustainability reports
- Compliance disclosures

---

## Testing

Run the test script to verify everything works:

```bash
python examples/validation_rules/test_validation_rules.py
```

The test script will:
1. Validate YAML syntax
2. Load rules into RulesEngine
3. Demonstrate all 12 operators
4. Test with sample data
5. Generate comprehensive test report

---

## Next Steps

1. **Review**: Read the README.md for detailed usage instructions
2. **Customize**: Modify rules to match your specific requirements
3. **Integrate**: Use with GreenLang agents and pipelines
4. **Extend**: Add new rules for your use cases
5. **Test**: Run test_validation_rules.py to validate your changes

---

## Summary Statistics

- **Total Files**: 5 (3 YAML, 1 README, 1 test script)
- **Total Lines**: 2,914
- **Total Size**: ~87 KB
- **Total Validation Rules**: 240+
- **Operators Demonstrated**: 12/12 (100%)
- **Severity Levels**: 3 (error, warning, info)
- **Predefined Rule Sets**: 15
- **Documentation**: Comprehensive inline comments + README
- **Test Coverage**: Automated test script included

---

**Created**: 2025-01-15
**Version**: 1.0.0
**Framework Compatibility**: GreenLang ValidationFramework v1.0.0+
