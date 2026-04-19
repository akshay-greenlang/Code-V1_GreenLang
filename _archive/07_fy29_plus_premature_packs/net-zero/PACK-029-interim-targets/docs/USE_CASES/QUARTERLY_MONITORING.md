# Use Case Guide: Quarterly Emissions Monitoring

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Workflow:** Quarterly Monitoring Workflow

---

## Table of Contents

1. [Use Case Overview](#use-case-overview)
2. [Personas and Roles](#personas-and-roles)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Interpreting the Quarterly Report](#interpreting-the-quarterly-report)
6. [RAG Status Thresholds](#rag-status-thresholds)
7. [Alert Configuration](#alert-configuration)
8. [Escalation Procedures](#escalation-procedures)
9. [Quarterly Board Reporting](#quarterly-board-reporting)
10. [Worked Example: Manufacturing Company](#worked-example-manufacturing-company)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

---

## Use Case Overview

### Scenario

A sustainability manager at a manufacturing company needs to track quarterly emissions against their SBTi-aligned interim targets. The company set a 1.5C-aligned target in 2024 to reduce Scope 1+2 emissions by 46.5% by 2030 from a 2021 baseline of 80,000 tCO2e. They need to know whether they are on track each quarter and, if not, what corrective actions to take.

### Business Value

| Benefit | Description |
|---------|-------------|
| Early warning | Detect deviations within 3 months instead of waiting for annual review |
| Proactive management | Trigger corrective actions before gaps become unrecoverable |
| Board confidence | Provide quarterly progress updates with clear RAG status |
| SBTi compliance | Demonstrate continuous monitoring per SBTi best practices |
| CDP scoring | Show active management of targets (CDP Management score) |

### Workflow Phases

```
Phase 1: DataCollection
    Collect actual emissions for the quarter from MRV Bridge
    |
    v
Phase 2: ProgressCheck
    Compare actuals against quarterly target using Quarterly Monitoring Engine
    |
    v
Phase 3: TrendAnalysis
    Analyze trend direction, velocity, and annualized projection
    |
    v
Phase 4: QuarterlyReport
    Generate quarterly progress report with RAG status and recommendations
```

---

## Personas and Roles

| Persona | Role | Responsibilities |
|---------|------|------------------|
| Sustainability Manager | Primary user | Run quarterly monitoring, analyze results, escalate |
| Data Analyst | Data provider | Ensure emissions data is complete and accurate |
| Facility Manager | Operations | Provide operational context for variances |
| CFO | Executive sponsor | Review quarterly dashboard, approve corrective actions |
| Board ESG Committee | Governance | Receive quarterly RAG summary |

### RBAC Permissions

| Permission | Sustainability Manager | Data Analyst | CFO |
|-----------|----------------------|-------------|-----|
| Run quarterly monitoring | Yes | No | No |
| View quarterly report | Yes | Yes | Yes |
| Configure alert thresholds | Yes | No | No |
| Approve corrective actions | No | No | Yes |
| Export to CDP | Yes | No | No |

---

## Prerequisites

### Before First Quarterly Monitoring

1. **Interim targets set**: Run the Interim Target Setting Workflow at least once
2. **MRV Bridge configured**: Emissions data sources connected for all material scopes
3. **Alert channels configured**: Email/Slack/Teams notifications set up
4. **Roles assigned**: RBAC permissions for all participants
5. **Quarterly schedule established**: Calendar reminders for data collection deadlines

### Before Each Quarterly Run

1. **Emissions data complete**: All facilities have reported for the quarter
2. **Data quality checked**: No significant gaps or outliers in the data
3. **Activity data available**: Revenue, production, or other activity metrics for the quarter

### Data Collection Timeline

```
Quarter End (e.g., March 31)
    |
    +-- Week 1-2: Facility data collection
    |
    +-- Week 3: Data quality review and validation
    |
    +-- Week 4: Run Quarterly Monitoring Workflow
    |
    +-- Week 5: Report to management and board
```

---

## Step-by-Step Walkthrough

### Step 1: Collect Quarterly Emissions Data

```python
from integrations.mrv_bridge import MRVBridge

# Initialize MRV Bridge
mrv = MRVBridge(config=pack_config)

# Collect emissions for Q1 2025
quarterly_data = await mrv.collect_quarterly_emissions(
    entity_id="entity-001",
    year=2025,
    quarter=1,
    scopes=["scope_1", "scope_2"],
)

# Verify data completeness
print(f"Data completeness: {quarterly_data.completeness_pct}%")
print(f"Scope 1: {quarterly_data.scope_1_tco2e:,.0f} tCO2e")
print(f"Scope 2: {quarterly_data.scope_2_tco2e:,.0f} tCO2e")
print(f"Total: {quarterly_data.total_tco2e:,.0f} tCO2e")
```

### Step 2: Run Quarterly Monitoring Engine

```python
from engines.quarterly_monitoring_engine import QuarterlyMonitoringEngine

# Initialize engine
engine = QuarterlyMonitoringEngine()

# Prepare input
monitoring_input = QuarterlyMonitoringInput(
    entity_id="entity-001",
    year=2025,
    quarter=1,
    actual_emissions=quarterly_data.total_tco2e,
    target_emissions=Decimal("16700"),  # Quarterly target (annual / 4)
    previous_quarters=[
        {"year": 2024, "quarter": 4, "actual": Decimal("17200")},
        {"year": 2024, "quarter": 3, "actual": Decimal("17500")},
        {"year": 2024, "quarter": 2, "actual": Decimal("16800")},
        {"year": 2024, "quarter": 1, "actual": Decimal("16500")},
    ],
    activity_data={
        "revenue_m_usd": Decimal("145"),
    },
)

# Run monitoring
result = await engine.calculate(monitoring_input)
```

### Step 3: Review RAG Status

```python
print(f"RAG Status: {result.rag_status}")         # GREEN, AMBER, or RED
print(f"Actual: {result.actual_emissions:,.0f} tCO2e")
print(f"Target: {result.target_emissions:,.0f} tCO2e")
print(f"Gap: {result.gap_tco2e:,.0f} tCO2e")
print(f"Gap %: {result.gap_pct:.1f}%")
print(f"Trend: {result.trend_direction}")          # IMPROVING, STABLE, WORSENING
print(f"Velocity: {result.trend_velocity}")        # Rate of change
print(f"Annualized projection: {result.annualized_projection:,.0f} tCO2e")
```

### Step 4: Generate Quarterly Report

```python
from templates.quarterly_progress_report import QuarterlyProgressReport

# Generate report
report = QuarterlyProgressReport()
output = report.render(
    monitoring_result=result,
    format="html",  # or "md", "json", "pdf"
    include_chart=True,
)

# Save report
with open("Q1_2025_Progress_Report.html", "w") as f:
    f.write(output)
```

### Step 5: Trigger Alerts (Automatic)

```python
# Alerts are generated automatically based on RAG status
# Configuration determines alert channels and escalation

# View generated alerts
for alert in result.alerts:
    print(f"Alert: {alert.type} - {alert.severity}")
    print(f"  Message: {alert.message}")
    print(f"  Channel: {alert.delivery_channel}")
    print(f"  Sent to: {alert.recipients}")
```

---

## Interpreting the Quarterly Report

### Report Sections

```
+-------------------------------------------------------+
| QUARTERLY PROGRESS REPORT                              |
| Entity: Acme Manufacturing | Q1 2025                  |
+-------------------------------------------------------+
|                                                        |
| RAG STATUS: [GREEN]                                    |
|                                                        |
| Actual: 16,200 tCO2e    Target: 16,700 tCO2e          |
| Gap: -500 tCO2e (0.5 tCO2e below target = ON TRACK)   |
|                                                        |
+-------------------------------------------------------+
| TREND ANALYSIS                                          |
|                                                        |
| Direction: IMPROVING (emissions decreasing)             |
| Velocity: -3.2% quarter-over-quarter                    |
| Annualized projection: 63,500 tCO2e                    |
| Annual target: 66,800 tCO2e                             |
| Projected surplus: 3,300 tCO2e                          |
|                                                        |
+-------------------------------------------------------+
| QUARTERLY TRAJECTORY                                    |
|                                                        |
|  Q1'24  Q2'24  Q3'24  Q4'24  Q1'25  Target            |
|  16,500 16,800 17,500 17,200 16,200 16,700             |
|  GREEN  GREEN  AMBER  AMBER  GREEN  --                  |
|                                                        |
+-------------------------------------------------------+
| KEY OBSERVATIONS                                        |
|                                                        |
| 1. Q1 2025 emissions returned to GREEN after two        |
|    AMBER quarters (Q3-Q4 2024)                          |
| 2. Q3 2024 spike attributed to expanded production      |
|    line (activity effect: +1,200 tCO2e)                 |
| 3. Solar PV installation (completed Q4 2024) now        |
|    delivering -800 tCO2e/quarter                        |
| 4. Intensity improved 8.2% vs Q1 2024                   |
|                                                        |
+-------------------------------------------------------+
| RECOMMENDATION                                          |
|                                                        |
| Continue current trajectory. No corrective action       |
| needed. Monitor Q2 to confirm improvement trend.        |
|                                                        |
+-------------------------------------------------------+
```

### RAG Status Interpretation

| RAG | Meaning | Management Response |
|-----|---------|---------------------|
| GREEN | On or below target | Continue current approach |
| AMBER | 1-5% above target | Investigate root cause, monitor closely |
| RED | >5% above target | Trigger corrective action planning |

---

## RAG Status Thresholds

### Default Thresholds

```python
RAG_THRESHOLDS = {
    "GREEN": {
        "gap_pct_max": Decimal("0"),     # At or below target
        "description": "On track"
    },
    "AMBER": {
        "gap_pct_min": Decimal("0"),     # Above target
        "gap_pct_max": Decimal("5"),     # Up to 5% above
        "description": "Slight deviation, monitor closely"
    },
    "RED": {
        "gap_pct_min": Decimal("5"),     # More than 5% above
        "description": "Significant deviation, corrective action needed"
    }
}
```

### Customizing Thresholds

Organizations can customize RAG thresholds based on their risk tolerance:

```python
# Conservative thresholds (low tolerance for deviation)
custom_thresholds = {
    "AMBER": {"gap_pct_max": Decimal("2.5")},
    "RED": {"gap_pct_min": Decimal("2.5")},
}

# Lenient thresholds (higher tolerance)
custom_thresholds = {
    "AMBER": {"gap_pct_max": Decimal("10")},
    "RED": {"gap_pct_min": Decimal("10")},
}
```

### Consecutive Quarter Escalation

PACK-029 escalates the alert level when multiple consecutive quarters are off-track:

```
1 AMBER quarter     -> Standard AMBER alert
2 AMBER quarters    -> Elevated AMBER alert + management notification
3 AMBER quarters    -> Auto-upgrade to RED + corrective action trigger
1 RED quarter       -> RED alert + executive notification
2 RED quarters      -> RED alert + board notification + mandatory corrective action
```

---

## Alert Configuration

### Alert Types

| Alert Type | Trigger | Severity | Default Channel |
|-----------|---------|----------|-----------------|
| `target_exceeded` | Actual > target | Medium | Email |
| `trend_worsening` | 2+ quarters of deterioration | Medium | Slack |
| `red_status` | RAG = RED | High | Email + Slack |
| `consecutive_amber` | 3+ AMBER quarters | High | Email + Teams |
| `annualized_miss` | Projected annual > annual target | Medium | Email |
| `data_quality_low` | Data completeness < 90% | Low | Email |

### Configuration Example

```python
alert_config = AlertConfiguration(
    entity_id="entity-001",
    channels={
        "email": {
            "enabled": True,
            "recipients": ["sustainability@acme.com", "cfo@acme.com"],
        },
        "slack": {
            "enabled": True,
            "webhook": "https://hooks.slack.com/services/T.../B.../...",
            "channel": "#sustainability-alerts",
        },
        "teams": {
            "enabled": True,
            "webhook": "https://outlook.office.com/webhook/...",
        },
    },
    escalation_rules=[
        {"condition": "rag_status == RED", "notify": ["cfo@acme.com", "board-esg@acme.com"]},
        {"condition": "consecutive_amber >= 3", "notify": ["cfo@acme.com"]},
    ],
)
```

---

## Escalation Procedures

### Escalation Matrix

```
Severity    | Response Time | Escalation Path
------------|--------------|------------------
LOW         | 5 days       | Sustainability Manager
MEDIUM      | 3 days       | Sustainability Manager -> Head of Operations
HIGH        | 1 day        | Sustainability Manager -> CFO -> Board ESG Committee
CRITICAL    | Same day     | CFO -> CEO -> Board Chair
```

### Escalation Actions by RAG Status

| RAG | Immediate Action | Within 1 Week | Within 1 Month |
|-----|------------------|---------------|----------------|
| GREEN | None required | Review report | File for records |
| AMBER | Review root cause | Brief management | Plan corrective actions |
| RED | Alert management | Run variance analysis | Submit corrective action plan |

---

## Quarterly Board Reporting

### Executive Dashboard Content

```
+-----------------------------------------------------------+
| QUARTERLY EMISSIONS DASHBOARD -- BOARD SUMMARY              |
| Period: Q1 2025 | Prepared: April 15, 2025                 |
+-----------------------------------------------------------+
|                                                            |
| OVERALL STATUS: [GREEN] ON TRACK                           |
|                                                            |
| Scope 1+2:  16,200 tCO2e (target: 16,700) | GREEN         |
| Scope 3:    28,500 tCO2e (target: 30,000) | GREEN         |
|                                                            |
| Progress to 2030 target: 32.3% achieved (vs 33.3% plan)   |
| Carbon budget remaining: 412,000 tCO2e                     |
| Estimated target achievement year: 2029.8 (ahead of plan)  |
|                                                            |
+-----------------------------------------------------------+
| KEY METRICS                                                 |
|                                                            |
| Metric           | Q1 2025 | Q1 2024 | Change              |
| Scope 1+2 (tCO2e)| 16,200  | 16,500  | -1.8%               |
| Revenue (M USD)  | 145     | 130     | +11.5%              |
| Intensity (t/M$) | 111.7   | 126.9   | -12.0%              |
| Renewable %      | 65%     | 45%     | +20pp               |
|                                                            |
+-----------------------------------------------------------+
| ACTIONS THIS QUARTER                                        |
|                                                            |
| - Solar PV Phase 2 operational (Feb 2025)                   |
| - Fleet electrification: 5 additional EVs deployed          |
| - HVAC upgrade completed at Plant B                         |
|                                                            |
| NEXT QUARTER OUTLOOK                                        |
|                                                            |
| - Expected GREEN status to continue                         |
| - Heat pump installation begins at Plant A (Q2)             |
| - Supplier engagement program launches                      |
+-----------------------------------------------------------+
```

---

## Worked Example: Manufacturing Company

### Company Profile

| Attribute | Value |
|-----------|-------|
| Company | Acme Manufacturing Ltd |
| Sector | Industrial manufacturing |
| Base year | 2021 |
| Base year Scope 1+2 | 80,000 tCO2e |
| Near-term target | 42,800 tCO2e by 2030 (46.5% reduction) |
| Annual target 2025 | 66,800 tCO2e |
| Quarterly target Q1 2025 | 16,700 tCO2e |

### Q1 2025 Data Collection

```python
# Facility data arrives via MRV Bridge
facility_data = {
    "Plant A": {"scope_1": Decimal("5200"), "scope_2": Decimal("3100")},
    "Plant B": {"scope_1": Decimal("3800"), "scope_2": Decimal("2200")},
    "HQ": {"scope_1": Decimal("300"), "scope_2": Decimal("1600")},
}

total_q1 = sum(
    f["scope_1"] + f["scope_2"]
    for f in facility_data.values()
)
# Total: 16,200 tCO2e
```

### Q1 2025 Monitoring Result

```python
result = await engine.calculate(QuarterlyMonitoringInput(
    entity_id="acme-001",
    year=2025,
    quarter=1,
    actual_emissions=Decimal("16200"),
    target_emissions=Decimal("16700"),
    previous_quarters=[
        {"year": 2024, "quarter": 4, "actual": Decimal("17200")},
        {"year": 2024, "quarter": 3, "actual": Decimal("17500")},
        {"year": 2024, "quarter": 2, "actual": Decimal("16800")},
        {"year": 2024, "quarter": 1, "actual": Decimal("16500")},
    ],
))

# Result:
# rag_status: GREEN
# gap: -500 tCO2e (below target)
# trend_direction: IMPROVING
# trend_velocity: -5.8% vs previous quarter
# annualized_projection: 63,500 tCO2e (below annual target of 66,800)
```

### Management Narrative

> "Q1 2025 emissions of 16,200 tCO2e are 500 tCO2e below the quarterly target, reflecting a GREEN RAG status. This represents a meaningful improvement after two AMBER quarters in H2 2024, driven by the completion of Solar PV Phase 2 in February and the HVAC upgrade at Plant B. Year-over-year, Q1 emissions decreased 1.8% despite 11.5% revenue growth, indicating a 12.0% improvement in carbon intensity. The annualized projection of 63,500 tCO2e is well below the 2025 annual target of 66,800 tCO2e, providing a 3,300 tCO2e buffer."

---

## Troubleshooting

### Common Issues

| Issue | Cause | Resolution |
|-------|-------|-----------|
| Missing quarterly data | MRV Bridge not configured for some facilities | Check MRV Bridge health; add missing data sources |
| Unusually high emissions | Data quality issue or operational event | Run Data Quality Profiler; check for outliers |
| RAG always GREEN despite no action | Target may be set too conservatively | Review target ambition; consider SBTi 1.5C |
| Alerts not being received | Alert channel misconfigured | Verify webhook URLs and email settings |
| Seasonal spikes causing RED | Normal seasonality not accounted for | Enable seasonal adjustment in monitoring config |
| Cannot compare to same quarter last year | Less than 4 quarters of data | Wait for full year; use available data |

### Data Quality Checks

```python
# Before running quarterly monitoring
quality_checks = [
    data.completeness_pct >= Decimal("90"),    # At least 90% data completeness
    data.total_tco2e > Decimal("0"),            # Non-zero emissions
    data.total_tco2e < previous_quarter * Decimal("2"),  # No implausible spikes
    data.total_tco2e > previous_quarter * Decimal("0.5"), # No implausible drops
]

if not all(quality_checks):
    print("WARNING: Data quality checks failed. Review before proceeding.")
```

---

## FAQ

**Q: How quickly after quarter-end can I run monitoring?**
A: As soon as emissions data is complete. Most organizations achieve this within 3-4 weeks of quarter-end. PACK-029 runs the calculation in under 500ms once data is available.

**Q: What if my data is not complete by the deadline?**
A: Run monitoring with available data (noting the completeness percentage). PACK-029 flags data below 90% completeness as a data quality alert. You can re-run when complete data arrives.

**Q: Can I adjust the quarterly target for seasonality?**
A: Yes. Enable seasonal adjustment in the monitoring configuration. PACK-029 will distribute the annual target across quarters based on historical seasonal patterns rather than using a flat 25% per quarter.

**Q: What happens if I miss a quarter?**
A: Run the next quarter with available data. PACK-029 will use the year-to-date actual vs. year-to-date target to account for the gap. Missing quarters are flagged but do not prevent monitoring.

**Q: Can I run monthly monitoring instead of quarterly?**
A: Monthly monitoring is planned for v1.1.0. Currently, the minimum frequency is quarterly. You can collect monthly data and aggregate to quarterly for PACK-029.

**Q: How do I handle restated data from a previous quarter?**
A: Re-run the affected quarter with corrected data. PACK-029 will update the audit trail showing the original and corrected values, maintaining full traceability.

---

**End of Quarterly Monitoring Use Case Guide**
