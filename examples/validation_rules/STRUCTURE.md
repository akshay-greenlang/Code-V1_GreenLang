# Validation Rules File Structure

## 📂 Directory Structure

```
examples/validation_rules/
│
├── data_import_validation.yaml      # Data import validation rules (567 lines)
├── calculation_validation.yaml      # Calculation input validation (831 lines)
├── report_validation.yaml           # Report output validation (807 lines)
│
├── README.md                         # Comprehensive documentation (709 lines)
├── SUMMARY.md                        # High-level summary (11 KB)
├── QUICK_REFERENCE.md                # Quick start guide (9.7 KB)
├── STRUCTURE.md                      # This file - structural overview
│
└── test_validation_rules.py         # Automated test suite (14 KB)
```

---

## 🗂️ YAML File Internal Structure

Each YAML validation file follows this consistent structure:

```yaml
# ============================================================================
# File Header with Description and Usage Instructions
# ============================================================================

metadata:
  name: "Rule Set Name"
  version: "1.0.0"
  description: "What this validates"
  author: "GreenLang Validation Framework"
  last_updated: "2025-01-15"
  applicable_to:
    - "Use case 1"
    - "Use case 2"

# ============================================================================
# SECTION 1: Category Name
# ============================================================================
# Description of what this section validates
# Operators used: list of operators
# ============================================================================

rules:
  # ----- Subcategory -----
  - name: "descriptive_rule_name"
    field: "field.path.to.validate"
    operator: "comparison_operator"
    value: expected_value_or_range
    severity: "error|warning|info"
    message: "Clear, actionable error message"
    enabled: true
    condition: "exists:optional_field"  # Optional

  # ... more rules ...

# ============================================================================
# SECTION 2: Another Category
# ============================================================================
# ... more sections ...

# ============================================================================
# Rule Sets: Organized groups for different scenarios
# ============================================================================

rule_sets:
  - name: "rule_set_name"
    description: "When to use this rule set"
    enabled: true
    rules:
      - rule_name_1
      - rule_name_2
      # ... or "all" for all rules

  # ... more rule sets ...

# ============================================================================
# Configuration Options
# ============================================================================

config:
  stop_on_error: false
  max_errors: 100
  include_values_in_errors: true
  verbose_logging: false
  # ... file-specific config ...
```

---

## 📋 data_import_validation.yaml Structure

```
data_import_validation.yaml (567 lines, 18 KB)
│
├── Metadata (lines 1-35)
│   ├── name: "Data Import Validation Rules"
│   ├── version: "1.0.0"
│   └── applicable_to: CSV imports, JSON ingestion, API validation
│
├── SECTION 1: Required Field Validation (lines 36-78)
│   ├── Building identification (building_id, facility_name)
│   ├── Location data (city, address)
│   └── Energy data fields (electricity, gas)
│   └── Operator: not_null
│
├── SECTION 2: Data Type and Format Validation (lines 79-142)
│   ├── Email format (regex)
│   ├── Date format YYYY-MM-DD (regex)
│   ├── Building ID format (regex)
│   ├── Phone number format (regex)
│   └── String contains validation (contains)
│   └── Operators: regex, contains
│
├── SECTION 3: Enumeration Validation (lines 143-216)
│   ├── Building type (Office, Retail, Industrial, etc.)
│   ├── Energy units (kWh, MWh, GWh)
│   ├── Gas units (therms, m3, MMBtu, GJ)
│   ├── Country codes (US, CA, GB, etc.)
│   └── Data quality status (not invalid/rejected)
│   └── Operators: in, not_in
│
├── SECTION 4: Numeric Range Validation (lines 217-313)
│   ├── Building area (100 - 50M sqft)
│   ├── Energy consumption (0 - 100M kWh)
│   ├── Gas consumption (0 - 10M therms)
│   ├── Employee count (0 - 100K)
│   ├── Reporting year (2000 - 2025)
│   └── Renewable percentage (0 - 100%)
│   └── Operators: >, >=, <, <=, ==, !=
│
├── SECTION 5: Data Completeness (lines 314-352)
│   ├── Optional but recommended fields
│   ├── Address, postal code, building year
│   └── Green certification check
│   └── Operators: not_null, is_null
│
├── SECTION 6: Cross-Field Business Logic (lines 353-401)
│   ├── Data quality indicators
│   ├── Completeness scores (≥80%)
│   ├── Confidence scores (≥70%)
│   └── Occupancy type validation
│   └── Operators: >=, in, not_null
│
├── SECTION 7: String Length and Content (lines 402-425)
│   ├── Non-empty string validation
│   └── Length constraints
│   └── Operators: !=, regex
│
├── SECTION 8: Conditional Validation (lines 426-455)
│   ├── Rules that only apply when other fields exist
│   └── Operator: not_null with conditions
│
├── Rule Sets (lines 456-520)
│   ├── quick_validation (6 essential rules)
│   ├── standard_validation (14 production rules)
│   ├── strict_validation (all rules)
│   ├── format_validation (format checks only)
│   └── range_validation (numeric ranges only)
│
└── Configuration (lines 521-567)
    ├── stop_on_error: false
    ├── max_errors: 100
    └── error_templates
```

---

## 🧮 calculation_validation.yaml Structure

```
calculation_validation.yaml (831 lines, 24 KB)
│
├── Metadata (lines 1-35)
│   ├── name: "Calculation Input Validation Rules"
│   └── applicable_to: Scope 1/2/3 calculations, carbon footprint
│
├── SECTION 1: Numeric Value Validation (lines 36-120)
│   ├── Energy consumption (> 0, ≤ 1B)
│   ├── Emission factors (> 0, ≤ 1000)
│   ├── Activity data (≥ 0)
│   ├── Conversion factors (> 0)
│   └── Uncertainty ranges (0 - 100%)
│   └── Operators: >, >=, <, <=, !=
│
├── SECTION 2: Unit Validation (lines 121-222)
│   ├── Energy units (kWh, MWh, GWh, etc.)
│   ├── Mass units (kg, tonnes, lb, etc.)
│   ├── Volume units (m3, L, gallons, etc.)
│   ├── Distance units (km, miles, etc.)
│   ├── Emission units (kg_co2e, tonnes_co2e, etc.)
│   └── Invalid unit checks
│   └── Operators: in, not_in
│
├── SECTION 3: Calculation Method Validation (lines 223-293)
│   ├── Calculation method (activity_based, spend_based, etc.)
│   ├── Emission scope (Scope 1, 2, 3)
│   ├── Allocation method (physical, economic, etc.)
│   └── Data quality tier (1-5)
│   └── Operators: not_null, in
│
├── SECTION 4: Precision Requirements (lines 294-325)
│   ├── Decimal precision (0-10 places)
│   └── Rounding method validation
│   └── Operators: >=, <=, in
│
├── SECTION 5: Business Logic Rules (lines 326-422)
│   ├── GWP timeframe (20, 100, 500 years)
│   ├── Grid emission factors (0 - 2.0 kg CO2e/kWh)
│   ├── Renewable percentage (0 - 100%)
│   ├── Sequestration rate (≥ 0)
│   ├── Offset credits validation
│   └── Biogenic carbon (≥ 0)
│   └── Operators: >=, <=, >, in
│
├── SECTION 6: Fuel-Specific Validation (lines 423-491)
│   ├── Fuel type enumeration
│   ├── Deprecated fuel check
│   ├── Heating value validation
│   └── Heating value type (HHV, LHV)
│   └── Operators: in, not_in, >
│
├── SECTION 7: Time Period Validation (lines 492-546)
│   ├── Reporting period start/end (required)
│   ├── Reporting year (1990 - 2030)
│   └── Date format (YYYY-MM-DD)
│   └── Operators: not_null, >=, <=, regex
│
├── SECTION 8: Reference Data Validation (lines 547-605)
│   ├── Emission factor source (EPA, DEFRA, etc.)
│   ├── Compliance standard (GHG Protocol, ISO, etc.)
│   ├── Database version format
│   └── Factor region requirement
│   └── Operators: not_null, in, regex
│
├── SECTION 9: Null/Missing Value Validation (lines 606-635)
│   ├── Calculation ID (required)
│   ├── Facility ID (required)
│   └── Baseline year (optional info)
│   └── Operators: not_null, is_null
│
├── Rule Sets (lines 636-760)
│   ├── scope_1_validation (Direct emissions)
│   ├── scope_2_validation (Purchased electricity)
│   ├── scope_3_validation (Value chain)
│   ├── precision_validation (Audited calculations)
│   └── quick_validation (Essential checks)
│
└── Configuration (lines 761-831)
    ├── enforce_unit_consistency: true
    ├── require_source_documentation: true
    └── precision settings
```

---

## 📊 report_validation.yaml Structure

```
report_validation.yaml (807 lines, 26 KB)
│
├── Metadata (lines 1-36)
│   ├── name: "Report Output Validation Rules"
│   └── applicable_to: GHG inventory, CDP, TCFD, sustainability reports
│
├── SECTION 1: Required Report Sections (lines 37-132)
│   ├── Report metadata (title, version, date)
│   ├── Organization info (name)
│   ├── Executive summary
│   ├── Emissions data (Scope 1, 2, 3 totals)
│   ├── Methodology section
│   └── Calculation approach, emission factors
│   └── Operators: not_null, !=
│
├── SECTION 2: Data Quality Validation (lines 133-200)
│   ├── Completeness score (≥85%)
│   ├── Accuracy score (≥80%)
│   ├── Overall quality rating
│   ├── Missing data percentage (≤15%)
│   └── Estimated data threshold (≤30%)
│   └── Operators: >=, <=, in
│
├── SECTION 3: Numeric Emissions Validation (lines 201-309)
│   ├── Total emissions (≥ 0, > 0, ≤ 100M)
│   ├── Scope-specific totals (≥ 0)
│   ├── Biogenic emissions (≥ 0)
│   ├── Offset credits (≥ 0)
│   └── Year-over-year change (-90% to +500%)
│   └── Operators: >=, >, <=, !=
│
├── SECTION 4: Aggregation and Breakdown (lines 310-365)
│   ├── Emissions by category
│   ├── Emissions by source
│   ├── Emissions by facility
│   ├── Minimum Scope 3 categories (≥ 1)
│   └── Category coverage (≥ 80%)
│   └── Operators: not_null, >=
│
├── SECTION 5: Compliance and Standards (lines 366-435)
│   ├── Reporting standard (GHG Protocol, ISO, etc.)
│   ├── Verification status
│   ├── Assurance level (limited, reasonable)
│   ├── Organizational boundary (required)
│   └── Boundary approach (operational, financial, equity)
│   └── Operators: not_null, in
│
├── SECTION 6: Format and Structure (lines 436-523)
│   ├── Report ID format (alphanumeric)
│   ├── Version format (semantic versioning)
│   ├── Date format (YYYY-MM-DD)
│   ├── Reporting period format
│   ├── Currency code (ISO 4217)
│   └── Emissions units (tonnes CO2e, etc.)
│   └── Operators: regex, in, !=
│
├── SECTION 7: Contextual Information (lines 524-617)
│   ├── Organization description
│   ├── Industry sector
│   ├── Employee count
│   ├── Annual revenue
│   ├── Intensity metrics (per revenue, per employee)
│   ├── Emissions targets
│   ├── Baseline/target year
│   ├── Uncertainties documentation
│   └── Data gaps documentation
│   └── Operators: not_null, >=, is_null
│
├── SECTION 8: Supporting Documentation (lines 618-660)
│   ├── Calculation sheets count
│   ├── Supporting documents
│   └── Verification statement
│   └── Operators: >=, not_null, is_null
│
├── Rule Sets (lines 661-750)
│   ├── basic_report (Minimum requirements)
│   ├── cdp_submission (CDP reporting)
│   ├── iso_14064_compliance (ISO standard)
│   ├── comprehensive_report (All rules)
│   └── quick_quality_check (Fast validation)
│
└── Configuration (lines 751-807)
    ├── require_executive_summary: true
    ├── minimum_completeness_score: 0.85
    ├── allowed_output_formats
    └── required/recommended sections
```

---

## 🎯 Operator Distribution Across Files

### data_import_validation.yaml
| Operator | Count | Primary Use |
|----------|-------|-------------|
| not_null | 20 | Required fields |
| regex | 8 | Format validation (email, date, phone) |
| in | 12 | Enumerations (types, units, codes) |
| >= / <= | 15 | Numeric ranges |
| > / < | 8 | Strict ranges |
| == / != | 5 | Exact matches / non-zero |
| not_in | 2 | Invalid values |
| contains | 2 | String content |
| is_null | 2 | Optional field checks |

### calculation_validation.yaml
| Operator | Count | Primary Use |
|----------|-------|-------------|
| >= / <= | 25 | Value ranges (factors, percentages) |
| > / < | 12 | Strict positive/negative |
| in | 20 | Unit validation, methods |
| not_null | 15 | Required calculation params |
| != | 5 | Non-zero checks |
| regex | 4 | Date/version formats |
| not_in | 3 | Deprecated values |
| == | 2 | Exact comparisons |
| is_null | 1 | Optional baseline |

### report_validation.yaml
| Operator | Count | Primary Use |
|----------|-------|-------------|
| not_null | 25 | Required sections |
| >= / <= | 20 | Quality scores, ranges |
| in | 15 | Standards, formats, ratings |
| regex | 8 | ID/date/version formats |
| > / < | 5 | Thresholds |
| != | 4 | Non-empty checks |
| is_null | 3 | Optional documentation |

---

## 🔄 Validation Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    Data Pipeline                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Data Import Validation                        │
│  File: data_import_validation.yaml                      │
│  ├─ Check required fields                               │
│  ├─ Validate formats (email, date, phone)               │
│  ├─ Check enumerations (types, units)                   │
│  ├─ Validate numeric ranges                             │
│  └─ Verify data completeness                            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Calculation Validation                        │
│  File: calculation_validation.yaml                      │
│  ├─ Validate numeric inputs                             │
│  ├─ Check unit consistency                              │
│  ├─ Verify calculation methods                          │
│  ├─ Validate emission factors                           │
│  └─ Check business logic rules                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  [Calculation Processing]                                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Report Validation                             │
│  File: report_validation.yaml                           │
│  ├─ Check required sections                             │
│  ├─ Validate data quality                               │
│  ├─ Verify emissions totals                             │
│  ├─ Check compliance standards                          │
│  ├─ Validate report format                              │
│  └─ Verify supporting documentation                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               Final Report Output                        │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 Rule Severity Philosophy

### Error (Critical - Stops Processing)
- Missing required fields
- Invalid data types
- Out-of-range critical values
- Invalid calculation methods
- Incomplete required sections

### Warning (Needs Attention - Processing Continues)
- Missing recommended fields
- Unusual but possible values
- Low data quality scores
- Missing optional documentation
- Format deviations

### Info (Recommendations - FYI Only)
- Enhancement suggestions
- Best practice recommendations
- Optional field availability
- Data quality improvements
- Documentation completeness

---

## 🎨 Naming Conventions

### Rule Names
Format: `category_description`

Examples:
- `building_id_required`
- `email_format`
- `energy_unit_valid`
- `scope_1_non_negative`
- `report_title_required`

### Field Paths
Format: Dot notation for nested fields

Examples:
- `building_id` (top-level)
- `energy_data.electricity_kwh` (nested)
- `location.country_code` (nested)
- `emissions_data.scope_1_total` (nested)

### Rule Set Names
Format: `purpose_level`

Examples:
- `quick_validation`
- `standard_validation`
- `scope_1_validation`
- `cdp_submission`
- `iso_14064_compliance`

---

## 🔗 Integration Points

```python
# With GreenLang ValidationFramework
ValidationFramework
  ├─> data_import_validation.yaml
  ├─> calculation_validation.yaml
  └─> report_validation.yaml

# With GreenLang Agents
Agent.validate()
  └─> RulesEngine (loaded from YAML)

# With GreenLang Pipelines
Pipeline.add_stage("validate")
  └─> YAML-based validator

# With Custom Code
Your Application
  ├─> Load YAML rules
  ├─> Create RulesEngine
  └─> Validate data
```

---

**Visual Guide Version**: 1.0.0
**Last Updated**: 2025-01-15
