# GreenLang Operator Training

**Document Version:** 1.0
**Last Updated:** December 2025
**Audience:** Operators, Data Entry Personnel, Analysts
**Prerequisites:** Completed [01_getting_started.md](01_getting_started.md)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Operator Role Overview](#operator-role-overview)
3. [Daily Workflows](#daily-workflows)
4. [Data Entry and Validation](#data-entry-and-validation)
5. [Running Calculations](#running-calculations)
6. [Understanding Results](#understanding-results)
7. [Quality Assurance](#quality-assurance)
8. [Reporting](#reporting)
9. [Alarm Management](#alarm-management)
10. [Best Practices](#best-practices)
11. [Certification Checklist](#certification-checklist)

---

## Introduction

This training module prepares operators to use GreenLang effectively for daily emissions calculations, data management, and compliance reporting. Upon completion, you will be able to:

- Enter and validate emissions data
- Run calculations and interpret results
- Generate compliance reports
- Manage alarms and notifications
- Ensure data quality standards

---

## Operator Role Overview

### Responsibilities

| Area | Responsibilities |
|------|-----------------|
| **Data Entry** | Input fuel consumption, activity data, and meter readings |
| **Calculations** | Execute emissions calculations and verify results |
| **Validation** | Ensure data quality and flag anomalies |
| **Reporting** | Generate daily, weekly, and monthly reports |
| **Monitoring** | Monitor alarms and respond to notifications |

### Access Levels

Operators typically have the following permissions:

```yaml
operator_permissions:
  calculations:
    - create
    - read
    - update_draft
  reports:
    - generate
    - view
    - export
  data:
    - input
    - validate
  alarms:
    - acknowledge
    - view
  settings:
    - view_own_profile
```

---

## Daily Workflows

### Morning Checklist

```
[ ] 1. Log into GreenLang dashboard
[ ] 2. Review overnight alarms and notifications
[ ] 3. Check data sync status from ERP/IoT systems
[ ] 4. Verify pending calculations from previous day
[ ] 5. Review data quality scores
```

### Standard Daily Workflow

```
08:00 - 09:00  | Review overnight data imports
09:00 - 10:00  | Enter manual data (meter readings, receipts)
10:00 - 11:00  | Validate data and fix anomalies
11:00 - 12:00  | Run daily calculations
12:00 - 13:00  | Break
13:00 - 14:00  | Review calculation results
14:00 - 15:00  | Generate daily reports
15:00 - 16:00  | Address any alarms or issues
16:00 - 17:00  | Prepare data for next day
```

### End-of-Day Checklist

```
[ ] 1. All daily calculations completed
[ ] 2. No unacknowledged critical alarms
[ ] 3. Data quality score > 95%
[ ] 4. Reports generated and distributed
[ ] 5. Notes added for any anomalies
```

---

## Data Entry and Validation

### Data Entry Interface

Access the data entry interface:

```
Dashboard > Data Entry > New Entry
```

### Supported Data Types

| Category | Data Types | Units |
|----------|-----------|-------|
| **Fuels** | Diesel, Gasoline, Natural Gas, LPG, Coal | liters, kg, m3 |
| **Electricity** | Grid, Solar, Wind | kWh |
| **Transport** | Distance, Fuel Used | km, liters |
| **Process** | Production Volume, Raw Materials | tonnes, units |
| **Waste** | Landfill, Recycled, Incinerated | kg |

### Manual Data Entry Example

```python
# Using the Python SDK for data entry
from greenlang.data import DataEntry

entry = DataEntry()

# Add fuel consumption record
entry.add_fuel_record(
    date="2025-12-07",
    fuel_type="diesel",
    quantity=500,
    unit="liters",
    source="fleet_vehicles",
    facility_id="FAC-001",
    notes="Weekly fuel delivery"
)

# Validate before submission
validation = entry.validate()
if validation.is_valid:
    result = entry.submit()
    print(f"Entry ID: {result.entry_id}")
else:
    print(f"Validation errors: {validation.errors}")
```

### Bulk Data Import

For large datasets, use bulk import:

```bash
# Import from CSV
greenlang data import --file fuel_data.csv --type fuel_consumption

# Import from Excel
greenlang data import --file monthly_data.xlsx --sheet "Fuel Data"
```

### CSV Format Requirements

```csv
date,fuel_type,quantity,unit,facility_id,source,notes
2025-12-01,diesel,100,liters,FAC-001,fleet,Monday delivery
2025-12-02,diesel,150,liters,FAC-001,fleet,Tuesday delivery
2025-12-03,natural_gas,500,cubic_meters,FAC-001,heating,
```

### Data Validation Rules

GreenLang automatically validates data against these rules:

| Rule | Description | Action |
|------|-------------|--------|
| **Range Check** | Values within expected ranges | Warning if out of range |
| **Consistency** | Data consistent with historical patterns | Flag anomalies |
| **Completeness** | Required fields present | Reject if missing |
| **Format** | Correct data types and formats | Reject if invalid |
| **Duplicates** | No duplicate entries | Warning on potential duplicates |

### Handling Validation Errors

```python
# Check for validation errors
validation = entry.validate()

for error in validation.errors:
    print(f"Field: {error.field}")
    print(f"Issue: {error.message}")
    print(f"Suggested fix: {error.suggestion}")

# Common error handling
if validation.has_error("out_of_range"):
    # Value is unusual - verify and override if correct
    entry.override_validation(
        field="quantity",
        reason="Confirmed correct - large delivery"
    )
```

---

## Running Calculations

### Single Calculation

```python
from greenlang import Calculator

calc = Calculator()

# Run a single calculation
result = calc.calculate(
    fuel_type="diesel",
    quantity=1000,
    unit="liters",
    region="US"
)

print(f"CO2e: {result.emissions_kg_co2e} kg")
print(f"Provenance: {result.provenance_hash}")
```

### Batch Calculations

For processing multiple records:

```python
# Batch calculation for daily data
batch_result = calc.calculate_batch(
    data_source="2025-12-07",  # Date or data source ID
    facility_id="FAC-001"
)

print(f"Records processed: {batch_result.count}")
print(f"Total emissions: {batch_result.total_kg_co2e} kg")
print(f"Processing time: {batch_result.duration_ms} ms")
```

### Scheduled Calculations

Calculations can run on schedule:

```yaml
# In calculation_schedule.yaml
schedules:
  - name: daily_fuel_calculations
    cron: "0 6 * * *"  # 6 AM daily
    calculation_type: fuel_emissions
    facilities:
      - FAC-001
      - FAC-002

  - name: monthly_scope2_calculations
    cron: "0 7 1 * *"  # 7 AM on 1st of month
    calculation_type: electricity_emissions
    facilities: all
```

### Calculation Status Tracking

```python
# Check calculation status
status = calc.get_status(calculation_id="calc_abc123")

print(f"Status: {status.state}")
# States: PENDING, PROCESSING, COMPLETED, FAILED

if status.state == "FAILED":
    print(f"Error: {status.error_message}")
```

---

## Understanding Results

### Result Components

Each calculation result includes:

```python
result = {
    "calculation_id": "calc_abc123",
    "status": "COMPLETED",

    # Emissions breakdown
    "emissions": {
        "co2": 2500.0,      # kg CO2
        "ch4": 0.5,         # kg CH4
        "n2o": 0.1,         # kg N2O
        "co2e": 2680.0      # kg CO2 equivalent (total)
    },

    # Methodology details
    "methodology": {
        "standard": "GHG Protocol",
        "scope": 1,
        "emission_factor_source": "EPA",
        "emission_factor_version": "2024"
    },

    # Provenance (audit trail)
    "provenance": {
        "hash": "sha256:3a7f8c2b...",
        "timestamp": "2025-12-07T12:00:00Z",
        "inputs_hash": "sha256:abc123...",
        "ef_hash": "sha256:def456..."
    },

    # Quality indicators
    "quality": {
        "data_quality_score": 0.95,
        "confidence_level": "HIGH",
        "uncertainty_percent": 5.2
    }
}
```

### Interpreting Quality Scores

| Score | Level | Meaning | Action |
|-------|-------|---------|--------|
| 0.95 - 1.00 | Excellent | High-quality primary data | No action needed |
| 0.85 - 0.94 | Good | Minor data gaps or estimates | Review flagged items |
| 0.70 - 0.84 | Moderate | Some estimates or defaults used | Improve data sources |
| < 0.70 | Low | Significant data quality issues | Immediate attention required |

### Comparing Results

```python
# Compare current vs previous period
comparison = calc.compare(
    current_period="2025-Q4",
    previous_period="2025-Q3",
    facility_id="FAC-001"
)

print(f"Change: {comparison.percent_change}%")
print(f"Absolute change: {comparison.absolute_change} kg CO2e")
print(f"Trend: {comparison.trend}")  # INCREASING, DECREASING, STABLE
```

---

## Quality Assurance

### Data Quality Dashboard

Access: `Dashboard > Quality Assurance > Data Quality`

Key metrics to monitor:

| Metric | Target | Description |
|--------|--------|-------------|
| **Completeness** | > 98% | All required fields populated |
| **Accuracy** | > 95% | Data within expected ranges |
| **Timeliness** | < 24h | Data entered within 24 hours |
| **Consistency** | > 90% | Data consistent with patterns |

### Quality Checks

```python
from greenlang.quality import QualityChecker

qc = QualityChecker()

# Run quality check on recent data
report = qc.check(
    date_range=("2025-12-01", "2025-12-07"),
    facility_id="FAC-001"
)

print(f"Overall score: {report.overall_score}")
print(f"Issues found: {report.issue_count}")

for issue in report.issues:
    print(f"  - {issue.type}: {issue.description}")
```

### Handling Anomalies

When anomalies are detected:

1. **Review the flagged data**
2. **Verify with source documents**
3. **Correct if error found**
4. **Document if correct but unusual**

```python
# Document an unusual but correct value
entry.add_annotation(
    field="quantity",
    note="Correct - large quarterly delivery",
    verified_by="John Smith",
    verification_date="2025-12-07"
)
```

---

## Reporting

### Standard Report Types

| Report | Frequency | Description |
|--------|-----------|-------------|
| **Daily Summary** | Daily | Previous day's emissions |
| **Weekly Rollup** | Weekly | Week's emissions by category |
| **Monthly Report** | Monthly | Full monthly breakdown |
| **Quarterly Compliance** | Quarterly | Regulatory submission format |
| **Annual Report** | Yearly | Complete annual emissions |

### Generating Reports

```python
from greenlang.reports import ReportGenerator

rg = ReportGenerator()

# Generate monthly report
report = rg.generate(
    report_type="monthly",
    period="2025-12",
    facility_id="FAC-001",
    format="pdf"
)

report.save("emissions_report_2025_12.pdf")

# Generate for regulatory submission
compliance_report = rg.generate(
    report_type="cbam_quarterly",
    period="2025-Q4",
    format="xml"  # CBAM XML format
)
```

### Report Distribution

```python
# Configure automatic distribution
distribution = ReportDistribution(
    report_type="weekly_summary",
    recipients=[
        "manager@company.com",
        "compliance@company.com"
    ],
    format="pdf",
    schedule="every_monday_9am"
)

distribution.enable()
```

---

## Alarm Management

### Alarm Types

GreenLang follows ISA-18.2 alarm management standards:

| Priority | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **Critical** | Immediate safety/compliance risk | < 5 minutes | Emissions limit exceeded |
| **High** | Significant issue requiring attention | < 1 hour | Data sync failure |
| **Medium** | Important but not urgent | < 4 hours | Quality score dropped |
| **Low** | Informational, minor issue | < 24 hours | Unusual data pattern |

### Alarm Dashboard

Access: `Dashboard > Alarms > Active Alarms`

### Acknowledging Alarms

```python
from greenlang.alarms import AlarmManager

am = AlarmManager()

# View active alarms
active = am.get_active_alarms()

for alarm in active:
    print(f"ID: {alarm.id}")
    print(f"Priority: {alarm.priority}")
    print(f"Message: {alarm.message}")
    print(f"Time: {alarm.triggered_at}")

# Acknowledge an alarm
am.acknowledge(
    alarm_id="ALM-001",
    operator="jsmith",
    notes="Investigating data sync issue"
)
```

### Alarm Response Procedures

1. **Receive alarm notification**
2. **Acknowledge within response time**
3. **Investigate root cause**
4. **Take corrective action**
5. **Document resolution**
6. **Clear alarm when resolved**

```python
# Clear alarm with resolution notes
am.clear(
    alarm_id="ALM-001",
    resolution="Restarted data sync service, backfill completed",
    operator="jsmith"
)
```

---

## Best Practices

### Data Entry

1. **Enter data daily** - Don't let data accumulate
2. **Verify before submit** - Double-check entries
3. **Document anomalies** - Add notes for unusual values
4. **Use bulk import** - For large datasets
5. **Keep source documents** - For audit trail

### Calculations

1. **Run at consistent times** - Maintain schedule
2. **Review before finalizing** - Check for errors
3. **Compare to previous periods** - Catch anomalies
4. **Verify provenance** - Ensure reproducibility
5. **Document methodology changes** - Track updates

### Quality Assurance

1. **Monitor quality scores daily** - Catch issues early
2. **Address red flags immediately** - Don't ignore warnings
3. **Validate data sources** - Ensure accuracy
4. **Regular reconciliation** - Match to source systems
5. **Continuous improvement** - Track and reduce errors

### Reporting

1. **Generate on schedule** - Don't miss deadlines
2. **Review before distribution** - Quality check
3. **Archive all reports** - Maintain records
4. **Track changes** - Document corrections
5. **Regulatory awareness** - Know requirements

---

## Certification Checklist

Complete the following to be certified as a GreenLang Operator:

### Knowledge Assessment

```
[ ] Understand GreenLang architecture
[ ] Know emission factor sources
[ ] Understand GHG Protocol scopes
[ ] Know data quality requirements
[ ] Understand alarm priorities
```

### Practical Skills

```
[ ] Successfully enter manual data
[ ] Complete bulk data import
[ ] Run single and batch calculations
[ ] Generate standard reports
[ ] Acknowledge and clear alarms
[ ] Handle validation errors
```

### Exercises Completed

```
[ ] Exercise 01: Basic Calculations
[ ] Exercise 02: Data Validation
[ ] Exercise 03: Report Generation
[ ] Exercise 04: Alarm Management
```

### Supervisor Sign-off

```
Operator Name: ___________________
Date Certified: __________________
Supervisor: _____________________
Signature: ______________________
```

---

## Quick Reference Card

### Common Commands (CLI)

```bash
# Data entry
greenlang data add --type fuel --file data.csv
greenlang data validate --date 2025-12-07

# Calculations
greenlang calc run --facility FAC-001 --date 2025-12-07
greenlang calc status --id calc_abc123

# Reports
greenlang report generate --type monthly --period 2025-12
greenlang report export --id rpt_xyz789 --format pdf

# Alarms
greenlang alarm list --status active
greenlang alarm ack --id ALM-001 --note "Investigating"
```

### Keyboard Shortcuts (Dashboard)

| Shortcut | Action |
|----------|--------|
| `Ctrl + N` | New data entry |
| `Ctrl + R` | Run calculation |
| `Ctrl + G` | Generate report |
| `Ctrl + A` | View alarms |
| `Ctrl + Q` | Quality dashboard |
| `Ctrl + H` | Help |

---

## Next Steps

- Complete hands-on exercises in [training_exercises/](training_exercises/)
- Review [05_troubleshooting_workshop.md](05_troubleshooting_workshop.md)
- Schedule certification assessment with supervisor
- Join the Operators community forum

---

**Training Complete!** You are now ready to operate GreenLang effectively.
