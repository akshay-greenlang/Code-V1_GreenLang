# Risk Matrix Implementation for Process Heat Agents

## Overview

The Risk Matrix module (`risk_matrix.py`) implements a comprehensive 5x5 risk assessment framework per **IEC 61511 Functional Safety** standards. It provides systematic risk evaluation, tracking, and compliance reporting for process heat agents.

## Key Features

### 1. Risk Matrix (5x5)
- **Severity Scale**: 1-5 (Minor to Catastrophic)
- **Likelihood Scale**: 1-5 (Remote to Almost Certain)
- **Risk Levels**: LOW, MEDIUM, HIGH, CRITICAL
- **Dynamic Mapping**: Risk level determined from severity × likelihood intersection

### 2. Risk Categories
- **Safety Risks**: Personnel injury, equipment damage
- **Environmental Risks**: Emissions, spills, pollution
- **Operational Risks**: Downtime, quality loss, efficiency
- **Compliance Risks**: Regulatory violations, financial penalties

### 3. IEC 61511 Integration
Automatic mapping of risks to Safety Integrity Levels (SIL):

| Risk Level | SIL Level | Risk Reduction Factor | Acceptance Days |
|-----------|-----------|----------------------|-----------------|
| CRITICAL  | SIL 3     | 1,000-10,000x       | 7 days          |
| HIGH      | SIL 2     | 100-1,000x          | 30 days         |
| MEDIUM    | SIL 1     | 10-100x             | 90 days         |
| LOW       | NO_SIL    | Monitor only        | 365 days        |

### 4. Color-Coded Visualization
- **Green**: LOW risk (monitor only)
- **Yellow**: MEDIUM risk (action within 90 days)
- **Orange**: HIGH risk (action within 30 days)
- **Red**: CRITICAL risk (immediate action - 7 days max)

## Module Components

### RiskMatrix Class (Static Utility)
Provides core risk calculation and mapping functionality:

```python
from greenlang.safety.risk_matrix import RiskMatrix

# Calculate risk level
level = RiskMatrix.calculate_risk_level(severity=4, likelihood=3)
# Returns: RiskLevel.HIGH

# Get visualization color
color = RiskMatrix.get_risk_color(level)
# Returns: "orange"

# Get required SIL
sil = RiskMatrix.get_required_sil(level)
# Returns: SafetyIntegrityLevel.SIL_2

# Get acceptance timeline
days = RiskMatrix.get_acceptance_days(level)
# Returns: 30

# Generate heatmap visualization data
heatmap = RiskMatrix.generate_heatmap(risks)
# Returns: matrix, colors, summary

# Aggregate risk statistics
stats = RiskMatrix.aggregate_risks(risks)
# Returns: total, critical, high, medium, low, overdue, etc.
```

### RiskData Model (Pydantic)
Represents individual risk assessment:

```python
from greenlang.safety.risk_matrix import RiskData, RiskCategory

risk = RiskData(
    title="High Temperature Excursion",
    description="Furnace temperature exceeds design limit",
    category=RiskCategory.OPERATIONAL,
    severity=4,        # 1-5 scale
    likelihood=3,      # 1-5 scale
    source="HAZOP",    # HAZOP, FMEA, or other
    source_id="DEV-015",
    mitigation_strategy="Install cascade temperature control",
    responsible_party="Process Engineer",
)
```

**Key Fields:**
- `risk_id`: Auto-generated unique identifier (RISK-XXXXXX)
- `title`, `description`: Risk narrative
- `category`: Safety, Environmental, Operational, or Compliance
- `severity`, `likelihood`: 1-5 numeric scales
- `status`: OPEN, IN_PROGRESS, MITIGATED, CLOSED, ACCEPTED
- `residual_severity`, `residual_likelihood`: Risk after mitigation
- `target_mitigation_date`: IEC 61511 deadline (auto-calculated)
- `provenance_hash`: SHA-256 audit trail

### RiskRegister Class (Stateful Manager)
Manages comprehensive risk lifecycle:

```python
from greenlang.safety.risk_matrix import RiskRegister

register = RiskRegister()

# Add risk
added_risk = register.add_risk(risk_data)

# Update risk status
register.update_risk(risk_id, {
    "status": RiskStatus.IN_PROGRESS,
    "responsible_party": "John Doe"
})

# Query risks
open_risks = register.get_open_risks()
critical_risks = register.get_critical_risks()
overdue_risks = register.get_overdue_risks()
safety_risks = register.get_open_risks(RiskCategory.SAFETY)

# Import from HAZOP/FMEA
hazop_risks = register.import_from_hazop(hazop_deviations)
fmea_risks = register.import_from_fmea(failure_modes)

# Generate reports
report = register.generate_report()  # JSON dictionary
text = register.export_to_compliance_report(format_type="text")
csv = register.export_to_compliance_report(format_type="csv")
json_report = register.export_to_compliance_report(format_type="json")
```

## Usage Examples

### Example 1: Basic Risk Assessment

```python
from greenlang.safety.risk_matrix import RiskMatrix, RiskLevel

# Create a 4x3 risk (Major severity, Moderate likelihood)
risk_level = RiskMatrix.calculate_risk_level(severity=4, likelihood=3)
assert risk_level == RiskLevel.HIGH

color = RiskMatrix.get_risk_color(risk_level)
assert color == "orange"

sil = RiskMatrix.get_required_sil(risk_level)
assert sil == SafetyIntegrityLevel.SIL_2

days = RiskMatrix.get_acceptance_days(risk_level)
assert days == 30  # Must mitigate within 30 days
```

### Example 2: Risk Register Management

```python
from greenlang.safety.risk_matrix import RiskRegister, RiskData, RiskCategory

register = RiskRegister()

# Create and add risk
risk = RiskData(
    title="Boiler Pressure Relief Valve Failure",
    description="PRV stuck open, pressure cannot be maintained",
    category=RiskCategory.SAFETY,
    severity=5,
    likelihood=2,
)

added = register.add_risk(risk)
print(f"Risk ID: {added.risk_id}")
print(f"Risk Level: {added.risk_level}")  # HIGH
print(f"Target Mitigation: {added.target_mitigation_date.date()}")

# Update status
register.update_risk(added.risk_id, {
    "status": RiskStatus.IN_PROGRESS,
    "mitigation_strategy": "Install redundant PRV"
})
```

### Example 3: HAZOP Integration

```python
# Import risks from HAZOP study
hazop_deviations = [
    {
        "deviation_id": "DEV-001",
        "deviation_description": "NO FLOW through boiler",
        "consequences": ["Loss of heat", "Equipment damage"],
        "severity": 4,
        "likelihood": 2,
        "recommendations": ["Install flow indicator", "Add alarm"],
    }
]

risks = register.import_from_hazop(hazop_deviations)
print(f"Imported {len(risks)} risks from HAZOP")
```

### Example 4: Compliance Reporting

```python
# Generate comprehensive report
report = register.generate_report()

print(f"Total Risks: {report['summary']['total_risks']}")
print(f"Critical: {report['summary']['critical']}")
print(f"High: {report['summary']['high']}")
print(f"Overdue: {report['overdue_risks_count']}")

# Export as text for compliance filing
text_report = register.export_to_compliance_report(format_type="text")

# Export as CSV for analysis
csv_report = register.export_to_compliance_report(format_type="csv")
```

### Example 5: Risk Trending

```python
# Get all critical risks
critical = register.get_critical_risks()
for risk in critical:
    print(f"{risk.title}: Due {risk.target_mitigation_date.date()}")

# Get overdue risks
overdue = register.get_overdue_risks()
print(f"Overdue risks: {len(overdue)}")

# Filter by category
safety_risks = register.get_open_risks(RiskCategory.SAFETY)
```

## Risk Matrix Reference

### 5x5 Matrix Layout

```
Severity\Likelihood   1        2        3        4        5
1 (Minor)             LOW      LOW      LOW      MEDIUM   MEDIUM
2 (Significant)       LOW      LOW      MEDIUM   MEDIUM   HIGH
3 (Serious)           LOW      MEDIUM   MEDIUM   HIGH     HIGH
4 (Major)             MEDIUM   MEDIUM   HIGH     HIGH     CRITICAL
5 (Catastrophic)      MEDIUM   HIGH     HIGH     CRITICAL CRITICAL
```

### Severity Scale Definition

1. **Minor**: No injury, insignificant environmental impact
2. **Significant**: Minor injury, contained spill
3. **Serious**: Lost time injury, environmental release
4. **Major**: Permanent disability, major environmental damage
5. **Catastrophic**: Fatality, major environmental disaster

### Likelihood Scale Definition

1. **Remote**: Very unlikely to occur (< 1% annually)
2. **Low**: Unlikely (1-10% annually)
3. **Moderate**: Possible (10-25% annually)
4. **Probable**: Likely (25-75% annually)
5. **Almost Certain**: Very likely (> 75% annually)

## Data Structures

### Risk Levels
```python
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### Risk Categories
```python
class RiskCategory(str, Enum):
    SAFETY = "safety"              # Personnel, equipment
    ENVIRONMENTAL = "environmental"  # Emissions, spills
    OPERATIONAL = "operational"    # Downtime, quality
    COMPLIANCE = "compliance"      # Regulatory, financial
```

### Risk Status
```python
class RiskStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    MITIGATED = "mitigated"
    CLOSED = "closed"
    ACCEPTED = "accepted"
```

### Safety Integrity Levels
```python
class SafetyIntegrityLevel(str, Enum):
    SIL_4 = "sil_4"  # 10,000-100,000x risk reduction
    SIL_3 = "sil_3"  # 1,000-10,000x risk reduction
    SIL_2 = "sil_2"  # 100-1,000x risk reduction
    SIL_1 = "sil_1"  # 10-100x risk reduction
    NO_SIL = "no_sil"  # No SIL required
```

## Audit Trail and Provenance

Each risk maintains:
- **Provenance Hash**: SHA-256 hash for audit trail verification
- **Audit Trail**: Complete event history with timestamps
- **Source Tracking**: Original HAZOP/FMEA reference
- **Status History**: All status transitions logged

```python
# Access audit trail
for event in register.audit_trail:
    print(f"{event['timestamp']}: {event['event_type']} - {event['risk_id']}")
```

## Report Generation

### Text Report
Executive summary with:
- Total risk counts by level
- Critical risks requiring immediate action
- Overdue risks
- Category breakdown

### CSV Report
Tabular format for:
- Spreadsheet analysis
- Historical trending
- Data export to CMMS/ERP systems

### JSON Report
Structured format containing:
- Summary statistics
- Heatmap matrix
- Critical risk list
- Overdue risk list
- Audit trail summary

## Integration Points

### Process Heat Agents
Connect RiskRegister to GL-001, GL-002, GL-005, etc.:

```python
class ThermalCommandAgent:
    def __init__(self):
        self.risk_register = RiskRegister()

    def on_alarm(self, alarm_code):
        # Auto-create risk from alarms
        risk = RiskData(
            title=f"Alarm: {alarm_code}",
            category=RiskCategory.OPERATIONAL,
            severity=self.assess_severity(alarm_code),
            likelihood=self.assess_likelihood(alarm_code),
        )
        self.risk_register.add_risk(risk)
```

### CMMS Integration
Export overdue risks to work orders:

```python
overdue = register.get_overdue_risks()
for risk in overdue:
    cmms.create_work_order(
        description=risk.title,
        priority="URGENT" if risk.risk_level == RiskLevel.CRITICAL else "HIGH",
        assigned_to=risk.responsible_party,
    )
```

### Dashboard Display
Visualize heatmap and trending:

```python
heatmap = RiskMatrix.generate_heatmap(register.risks.values())
dashboard.display_heatmap(heatmap["matrix"], heatmap["colors"])

agg = RiskMatrix.aggregate_risks(register.risks.values())
dashboard.display_gauge("Critical Risks", agg["critical"])
dashboard.display_gauge("Overdue Risks", agg["risks_overdue"])
```

## Testing

Comprehensive unit test coverage (49 tests):

```bash
pytest tests/unit/test_risk_matrix.py -v

# Test categories:
# - RiskMatrix calculations (all 25 combinations)
# - Color mapping (4 levels)
# - SIL assignment (4 levels)
# - Risk register operations (CRUD)
# - HAZOP/FMEA integration
# - Report generation
# - Data validation
```

## File Structure

```
greenlang/safety/
├── risk_matrix.py              # Main implementation (734 lines)
├── RISK_MATRIX_README.md       # This documentation
├── hazop_analyzer.py           # HAZOP integration
├── fmea_analyzer.py            # FMEA integration
└── ...

tests/unit/
├── test_risk_matrix.py         # 49 unit tests
└── ...

examples/
└── risk_matrix_usage_example.py # 6 complete examples
```

## Performance Characteristics

- **Risk Creation**: < 1ms
- **Heatmap Generation**: < 10ms (25 risks)
- **Report Generation**: < 50ms (100 risks)
- **Query Operations**: O(n) linear scan
- **Memory Usage**: ~1KB per risk

## Standards Compliance

- **IEC 61511:2016** - Functional Safety, Safety Instrumented Systems
- **IEC 61882:2016** - Hazard and Operability Studies (HAZOP)
- **IEC 60812:2018** - Failure Mode and Effects Analysis (FMEA)
- **ISO 14971** - Medical device risk management (risk matrix)

## Version History

- **1.0.0** - Initial implementation with 5x5 matrix, risk register, HAZOP/FMEA integration
  - Features: Core matrix, IEC 61511 SIL mapping, compliance reporting
  - Test Coverage: 49 unit tests, 100% coverage of public API

## Author

GreenLang Safety Engineering Team

## License

Proprietary - GreenLang Corporation
