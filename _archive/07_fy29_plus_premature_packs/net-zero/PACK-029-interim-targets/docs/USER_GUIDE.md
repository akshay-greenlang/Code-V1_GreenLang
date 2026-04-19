# PACK-029 Interim Targets Pack -- User Guide

**Pack ID:** PACK-029-interim-targets
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Step 1: Set Interim Targets](#step-1-set-interim-targets)
4. [Step 2: Monitor Progress Quarterly](#step-2-monitor-progress-quarterly)
5. [Step 3: Annual Progress Review](#step-3-annual-progress-review)
6. [Step 4: Investigate Variances](#step-4-investigate-variances)
7. [Step 5: Plan Corrective Actions](#step-5-plan-corrective-actions)
8. [Step 6: Annual Reporting](#step-6-annual-reporting)
9. [Step 7: Recalibrate Targets](#step-7-recalibrate-targets)
10. [Dashboard Overview](#dashboard-overview)
11. [Best Practices](#best-practices)
12. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Introduction

This guide walks through the complete PACK-029 user journey, from setting interim targets through monitoring, variance investigation, corrective action, and regulatory reporting. Each step corresponds to one of the 7 workflows in the pack.

### Who This Guide Is For

- **Sustainability Directors** managing net-zero programs
- **Climate Target Managers** responsible for interim milestones
- **CDP/TCFD Reporting Leads** preparing annual disclosures
- **SBTi Submission Leads** validating targets against SBTi criteria

### Prerequisites

Before starting, ensure:

1. PACK-029 is installed and configured (see [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md))
2. Baseline emissions data is available (either from PACK-021 or direct input)
3. Long-term net-zero target is defined (year and reduction percentage)
4. You have `target_manager` role or higher

---

## Getting Started

### Choosing Your Preset

PACK-029 provides 7 presets. Select the one that matches your ambition level:

| Preset | Best For | Annual Rate | Near-Term Target |
|--------|----------|-------------|-----------------|
| SBTi 1.5C | Most ambitious organizations | 4.2%/yr | 42% by 2030 |
| SBTi WB2C | Organizations meeting SBTi minimum | 2.5%/yr | 25% by 2030 |
| Race to Zero | Campaign participants | 7.0%/yr | 50% by 2030 |
| Corporate Net-Zero | Standard net-zero commitment | 4.2%/yr | 42% by 2030 + FLAG |
| Financial Institution | Banks/asset managers | 4.2%/yr | Portfolio alignment |
| SME Simplified | Small/medium enterprises | 2.5%/yr | Scope 1+2 only |
| Manufacturing | Manufacturing companies | 4.2%/yr | Intensity + absolute |

```python
from config.pack_config import PackConfig

# Load a preset
config = PackConfig.from_preset("sbti_15c")
print(f"Preset: {config.preset_name}")
print(f"Ambition: {config.ambition_level}")
print(f"Annual Rate: {config.annual_rate_pct}%/yr")
```

---

## Step 1: Set Interim Targets

### InterimTargetSettingWorkflow

This workflow creates your complete set of interim milestones from baseline to net-zero.

### 1.1 Prepare Baseline Data

```python
from engines.interim_target_engine import BaselineData

baseline = BaselineData(
    base_year=2021,
    scope_1_tco2e=150_000,       # Direct emissions (tCO2e)
    scope_2_tco2e=80_000,        # Purchased electricity (tCO2e)
    scope_3_tco2e=450_000,       # Value chain (tCO2e)
    scope_2_method="market_based",
    is_flag_sector=False,         # Set True for agriculture/forestry
    flag_emissions_tco2e=0,
)

print(f"Total Baseline: {baseline.scope_1_tco2e + baseline.scope_2_tco2e + baseline.scope_3_tco2e:,.0f} tCO2e")
# Total Baseline: 680,000 tCO2e
```

### 1.2 Define Long-Term Target

```python
from engines.interim_target_engine import LongTermTarget

long_term = LongTermTarget(
    target_year=2050,
    reduction_pct=90,            # 90% reduction from baseline
    residual_emissions_pct=10,   # Maximum 10% residual
    net_zero_year=2050,
    includes_scope_3=True,
)
```

### 1.3 Run the Workflow

```python
from workflows.interim_target_setting_workflow import InterimTargetSettingWorkflow

workflow = InterimTargetSettingWorkflow(preset="sbti_15c")
result = await workflow.execute(
    entity_name="GlobalManufacturing Inc.",
    entity_id="gm-001",
    baseline=baseline,
    long_term_target=long_term,
)

# Review scope-specific timelines
for timeline in result.interim_targets.scope_timelines:
    print(f"\n{timeline.scope}:")
    print(f"  Baseline: {timeline.baseline_tco2e:,.0f} tCO2e")
    print(f"  Near-term ({timeline.near_term_year}): "
          f"{timeline.near_term_target_tco2e:,.0f} tCO2e "
          f"({timeline.near_term_reduction_pct}% reduction)")
    print(f"  Long-term ({timeline.long_term_year}): "
          f"{timeline.long_term_target_tco2e:,.0f} tCO2e "
          f"({timeline.long_term_reduction_pct}% reduction)")
    print(f"  Annual rate: {timeline.annual_rate_pct}%/yr")
```

### 1.4 Review SBTi Validation

```python
validation = result.sbti_validation

print(f"\nSBTi Validation Summary:")
print(f"  Compliant: {validation.is_compliant}")
print(f"  Checks: {validation.passed_checks}/{validation.total_checks} passed")
print(f"  Failed: {validation.failed_checks}")
print(f"  Warnings: {validation.warning_checks}")

for note in validation.validation_notes:
    print(f"  - {note}")
```

### 1.5 Generate Target Report

```python
from templates.interim_targets_summary import InterimTargetsSummaryTemplate

template = InterimTargetsSummaryTemplate()
report = template.render(
    interim_result=result.interim_targets,
    format="html",
)

# Save to file
with open("interim_targets_report.html", "w") as f:
    f.write(report)
```

### 1.6 Understanding Pathway Shapes

Choose the pathway shape that best matches your reduction strategy:

**Linear** (recommended for most organizations):
- Equal reduction each year
- Simple to communicate and track

**Front-Loaded** (for organizations planning early action):
- Faster reductions in early years, slower later
- Good when easy wins are available now

**Back-Loaded** (not recommended for SBTi):
- Slower early reductions, accelerating later
- Warning: May not meet SBTi near-term requirements

**Milestone-Based** (for organizations with specific commitments):
- Custom milestones at defined years
- Useful when specific initiatives have known timelines

**Constant Rate** (compound annual reduction):
- Same percentage reduction each year (exponential decay)
- Larger absolute reductions early, smaller later

---

## Step 2: Monitor Progress Quarterly

### QuarterlyMonitoringWorkflow

Run this workflow every quarter to track actual emissions against interim targets.

### 2.1 Collect Quarterly Data

```python
quarterly_data = {
    "scope_1_tco2e": 35_200,    # Q3 2025 Scope 1 actual
    "scope_2_tco2e": 18_500,    # Q3 2025 Scope 2 actual
    "scope_3_tco2e": 108_000,   # Q3 2025 Scope 3 actual
}
```

### 2.2 Run Quarterly Monitoring

```python
from workflows.quarterly_monitoring_workflow import QuarterlyMonitoringWorkflow

workflow = QuarterlyMonitoringWorkflow()
result = await workflow.execute(
    entity_id="gm-001",
    quarter="2025-Q3",
    actual_data=quarterly_data,
)

# Review RAG status
print(f"\nQ3 2025 Monitoring Results:")
print(f"  Overall Status: {result.overall_rag}")

for scope_result in result.scope_results:
    icon = {"green": "[OK]", "amber": "[!!]", "red": "[XX]"}[scope_result.rag_status]
    print(f"  {icon} {scope_result.scope}: "
          f"Actual {scope_result.actual_tco2e:,.0f} vs "
          f"Target {scope_result.target_tco2e:,.0f} "
          f"({scope_result.variance_pct:+.1f}%)")

# Check alerts
if result.alerts:
    print(f"\n  Alerts ({len(result.alerts)}):")
    for alert in result.alerts:
        print(f"    [{alert.severity}] {alert.message}")
```

### 2.3 Interpret RAG Status

| Status | What It Means | What To Do |
|--------|--------------|------------|
| GREEN | On track (within 5% of target) | Continue current trajectory, no action needed |
| AMBER | Slightly off track (5-15% above target) | Review causes, prepare contingency plans |
| RED | Significantly off track (>15% above target) | Immediate corrective action required |

### 2.4 Track Trends

```python
print(f"\nTrend Analysis:")
print(f"  Direction: {result.trend_direction}")
print(f"  Velocity: {result.trend_velocity_pct_per_quarter:+.1f}% per quarter")
print(f"  Consecutive quarters on trend: {result.trend_streak}")

# Annualized projection
print(f"\nAnnualized Projection:")
print(f"  Projected annual: {result.annualized_projection_tco2e:,.0f} tCO2e")
print(f"  Annual target: {result.annual_target_tco2e:,.0f} tCO2e")
```

---

## Step 3: Annual Progress Review

### AnnualProgressReviewWorkflow

Run this workflow annually to produce a comprehensive progress assessment.

### 3.1 Run Annual Review

```python
from workflows.annual_progress_review_workflow import AnnualProgressReviewWorkflow

workflow = AnnualProgressReviewWorkflow()
result = await workflow.execute(
    entity_id="gm-001",
    reporting_year=2025,
    annual_actual={
        "scope_1_tco2e": 140_000,
        "scope_2_tco2e": 72_000,
        "scope_3_tco2e": 420_000,
    },
)
```

### 3.2 Review Year-over-Year Comparison

```python
yoy = result.yoy_comparison
print(f"\nYear-over-Year (2024 -> 2025):")
print(f"  Previous: {yoy.previous_year_tco2e:,.0f} tCO2e")
print(f"  Current:  {yoy.current_year_tco2e:,.0f} tCO2e")
print(f"  Change:   {yoy.absolute_change_tco2e:+,.0f} tCO2e ({yoy.percentage_change:+.1f}%)")
print(f"  Required: {yoy.required_change_pct:+.1f}% per year")
print(f"  On Track: {'Yes' if abs(yoy.percentage_change) >= abs(yoy.required_change_pct) else 'No'}")
```

### 3.3 Check Carbon Budget

```python
budget = result.cumulative_budget
print(f"\nCarbon Budget Status:")
print(f"  Total Budget:     {budget.total_budget_tco2e:,.0f} tCO2e")
print(f"  Consumed:         {budget.consumed_tco2e:,.0f} tCO2e ({budget.burn_rate_pct:.1f}%)")
print(f"  Remaining:        {budget.remaining_tco2e:,.0f} tCO2e")
print(f"  Years Remaining:  {budget.years_until_exhaustion:.1f} years")
```

### 3.4 Review Forward Projection

```python
projection = result.forward_projection
print(f"\nForward Projection:")
print(f"  Projected 2030: {projection.projected_2030_tco2e:,.0f} tCO2e")
print(f"  Target 2030:    {projection.target_2030_tco2e:,.0f} tCO2e")
print(f"  Gap:            {projection.gap_tco2e:+,.0f} tCO2e")
print(f"  On-Track Probability: {projection.on_track_probability:.0%}")
```

---

## Step 4: Investigate Variances

### VarianceInvestigationWorkflow

When monitoring reveals off-track performance, use LMDI decomposition to understand why.

### 4.1 Run LMDI Decomposition

```python
from workflows.variance_investigation_workflow import VarianceInvestigationWorkflow

workflow = VarianceInvestigationWorkflow()
result = await workflow.execute(
    entity_id="gm-001",
    period_start="2024",
    period_end="2025",
    emissions_start=680_000,
    emissions_end=632_000,
    activity_data_start={"revenue_musd": 2000, "production_tonnes": 500_000},
    activity_data_end={"revenue_musd": 2200, "production_tonnes": 520_000},
)
```

### 4.2 Interpret Decomposition Results

```python
decomp = result.decomposition

print(f"\nLMDI Variance Decomposition (2024 -> 2025):")
print(f"  Total Change:     {decomp.total_change_tco2e:+,.0f} tCO2e")
print(f"  Activity Effect:  {decomp.activity_effect_tco2e:+,.0f} tCO2e "
      f"({decomp.activity_effect_pct:+.1f}%)")
print(f"  Intensity Effect: {decomp.intensity_effect_tco2e:+,.0f} tCO2e "
      f"({decomp.intensity_effect_pct:+.1f}%)")
print(f"  Structural Effect:{decomp.structural_effect_tco2e:+,.0f} tCO2e "
      f"({decomp.structural_effect_pct:+.1f}%)")
print(f"  Perfect Decomp:   {decomp.is_perfect_decomposition}")
```

### 4.3 Understanding the Effects

| Effect | Positive Means | Negative Means | Typical Causes |
|--------|---------------|----------------|----------------|
| Activity | Business grew, adding emissions | Business contracted | Revenue growth, production increase, expansion |
| Intensity | Intensity worsened | Intensity improved | Efficiency gains, technology upgrades, fuel switching |
| Structural | Structural shift increased emissions | Structural shift reduced emissions | Product mix changes, division size changes, outsourcing |

### 4.4 Generate Variance Report

```python
from templates.variance_waterfall_report import VarianceWaterfallReportTemplate

template = VarianceWaterfallReportTemplate()
report = template.render(
    variance_result=result,
    format="html",
    include_waterfall_chart=True,
)
```

---

## Step 5: Plan Corrective Actions

### CorrectiveActionPlanningWorkflow

When a gap to target is identified, this workflow generates an optimized corrective action plan.

### 5.1 Run Corrective Action Planning

```python
from workflows.corrective_action_planning_workflow import CorrectiveActionPlanningWorkflow

workflow = CorrectiveActionPlanningWorkflow()
result = await workflow.execute(
    entity_id="gm-001",
    gap_tco2e=22_000,
    max_budget_usd=10_000_000,
    max_years=3,
    optimization_strategy="cost_effective",
)
```

### 5.2 Review Selected Initiatives

```python
print(f"\nCorrective Action Plan:")
print(f"  Gap to Target:  {result.gap_tco2e:,.0f} tCO2e")
print(f"  Initiatives Selected: {len(result.selected_initiatives)}")
print(f"  Total Reduction: {result.total_reduction_tco2e:,.0f} tCO2e")
print(f"  Gap Closure:    {result.gap_closure_pct:.1f}%")
print(f"  Total Cost:     ${result.total_cost_usd:,.0f}")
print(f"  Years to Close: {result.years_to_close}")

print(f"\nInitiative Portfolio (ranked by cost-effectiveness):")
for i, init in enumerate(result.selected_initiatives, 1):
    print(f"  {i}. {init.name}")
    print(f"     Reduction:  {init.reduction_tco2e:,.0f} tCO2e")
    print(f"     Cost:       ${init.cost_usd:,.0f}")
    print(f"     $/tCO2e:    ${init.cost_per_tco2e:,.0f}")
    print(f"     Timeline:   {init.start_year} - {init.completion_year}")
```

### 5.3 Review Scenario Analysis

```python
print(f"\nScenario Analysis:")
for scenario_name, scenario in result.scenarios.items():
    print(f"  {scenario_name}: {scenario.closure_pct:.1f}% gap closure "
          f"in {scenario.years:.1f} years")
```

---

## Step 6: Annual Reporting

### AnnualReportingWorkflow

Generate annual regulatory disclosures for CDP, TCFD, and SBTi.

### 6.1 Run Annual Reporting

```python
from workflows.annual_reporting_workflow import AnnualReportingWorkflow

workflow = AnnualReportingWorkflow()
result = await workflow.execute(
    entity_id="gm-001",
    reporting_year=2025,
    exports=["cdp", "tcfd", "sbti"],
)
```

### 6.2 CDP C4.1 and C4.2 Export

```python
cdp = result.cdp_export

# C4.1: Text description of interim targets
print(f"\nCDP C4.1 Text:")
print(f"  {cdp.c4_1_text}")

# C4.2: Target details table
print(f"\nCDP C4.2 Rows ({len(cdp.c4_2_rows)}):")
for row in cdp.c4_2_rows:
    print(f"  Scope: {row.scope}")
    print(f"  Base Year: {row.base_year}")
    print(f"  Target Year: {row.target_year}")
    print(f"  Reduction: {row.reduction_pct}%")
    print(f"  Status: {row.progress_status}")
```

### 6.3 TCFD Metrics and Targets

```python
tcfd = result.tcfd_export

print(f"\nTCFD Metrics and Targets:")
for section in tcfd.sections:
    print(f"  {section.title}:")
    print(f"    {section.content[:200]}...")
```

### 6.4 SBTi Annual Disclosure

```python
sbti = result.sbti_disclosure

print(f"\nSBTi Annual Disclosure:")
print(f"  Status: {sbti.status}")
print(f"  Near-Term Progress: {sbti.near_term_progress_pct:.1f}%")
print(f"  On Track: {sbti.is_on_track}")
```

---

## Step 7: Recalibrate Targets

### TargetRecalibrationWorkflow

Recalibrate targets when significant changes occur.

### 7.1 Trigger Events

| Event | When to Recalibrate | Threshold |
|-------|--------------------|-----------|
| Acquisition | Entity acquires another company | >5% baseline change |
| Divestment | Entity sells a division | >5% baseline change |
| Methodology | GHG calculation method changes | Any change |
| Base Year Update | New data for base year | Any change |
| Scope Change | Organizational boundary changes | Any change |

### 7.2 Run Recalibration

```python
from workflows.target_recalibration_workflow import TargetRecalibrationWorkflow

workflow = TargetRecalibrationWorkflow()
result = await workflow.execute(
    entity_id="gm-001",
    trigger="acquisition",
    trigger_details={
        "acquired_entity": "SubCo Ltd.",
        "acquired_emissions_tco2e": 45_000,
        "acquisition_date": "2025-09-01",
    },
)

print(f"\nRecalibration Summary:")
print(f"  Trigger: {result.trigger_type}")
print(f"  Old Baseline: {result.old_baseline_tco2e:,.0f} tCO2e")
print(f"  New Baseline: {result.new_baseline_tco2e:,.0f} tCO2e")
print(f"  Change: {result.baseline_change_pct:+.1f}%")
print(f"  Milestones Adjusted: {len(result.adjusted_milestones)}")
```

---

## Dashboard Overview

### Executive Dashboard

The executive dashboard provides a single-page view of net-zero progress:

```
+------------------------------------------------------------------+
|  NET ZERO PROGRESS DASHBOARD - GlobalManufacturing Inc.           |
|                                                                    |
|  [GREEN] Overall Status: ON TRACK                                 |
|  Temperature Score: 1.5C | SBTi: Validated                       |
+------------------------------------------------------------------+
|                                                                    |
|  SCOPE 1+2                    SCOPE 3                             |
|  [GREEN] On Track            [AMBER] Slightly Off Track           |
|  140,000 / 145,000 tCO2e     420,000 / 405,000 tCO2e            |
|  -3.4% variance              +3.7% variance                      |
|                                                                    |
+------------------------------------------------------------------+
|  CARBON BUDGET                TREND (4 quarters)                  |
|  Consumed: 21.6%              Direction: Improving                |
|  Remaining: 2,745,000 tCO2e   Velocity: -2.1%/quarter            |
|  Years Left: 18.2             Streak: 3 quarters                  |
+------------------------------------------------------------------+
|  5-YEAR MILESTONES            CORRECTIVE ACTIONS                  |
|  2025: [OK] 10% reduction     Active: 2 initiatives              |
|  2030: [--] 42% target        Planned: 3 initiatives             |
|  2035: [--] 60% target        Budget: $4.2M allocated            |
|  2040: [--] 76% target        Gap Closure: 85%                   |
+------------------------------------------------------------------+
```

---

## Best Practices

### Target Setting

1. **Use a recent base year**: SBTi recommends no earlier than 2019 for near-term targets
2. **Include all material scopes**: Scope 3 must be included if >= 40% of total emissions
3. **Choose linear pathway**: Unless you have specific reasons for front/back-loaded
4. **Validate before submission**: Always run SBTi 21-criteria validation before submitting

### Monitoring

1. **Monitor quarterly at minimum**: Annual monitoring misses off-track signals
2. **Automate data collection**: Integrate with MRV agents for automated emissions data
3. **Set up alerts**: Configure email/Slack/Teams alerts for AMBER and RED status
4. **Track trends, not just snapshots**: Three consecutive quarters of deterioration is a stronger signal than one bad quarter

### Variance Investigation

1. **Run LMDI for every annual review**: Even when on-track, understanding why helps
2. **Distinguish activity from intensity**: Business growth (activity) is different from inefficiency (intensity)
3. **Investigate structural effects**: Changes in product mix or division size matter

### Corrective Actions

1. **Start with low-cost initiatives**: LED lighting, HVAC optimization, fleet management
2. **Budget realistically**: Include implementation costs, not just equipment
3. **Plan for pessimistic scenario**: Use the pessimistic scenario timeline
4. **Track initiative delivery**: Monitor corrective action implementation quarterly

---

## Troubleshooting Common Issues

### "No interim targets found for entity"

**Cause:** Interim targets have not been set for this entity ID.

**Solution:** Run the Interim Target Setting Workflow first:
```python
workflow = InterimTargetSettingWorkflow(preset="sbti_15c")
result = await workflow.execute(entity_id="your-entity-id", ...)
```

### "Quarterly data not matching expected scope breakdown"

**Cause:** Quarterly actual data format does not match the expected scope structure.

**Solution:** Ensure quarterly data includes all expected scopes:
```python
quarterly_data = {
    "scope_1_tco2e": 35200,    # Required
    "scope_2_tco2e": 18500,    # Required
    "scope_3_tco2e": 108000,   # Optional if scope 3 not in targets
}
```

### "SBTi validation fails on criterion 9 (base year recency)"

**Cause:** Base year is more than 2 years before the submission year.

**Solution:** Update base year to a more recent year or provide justification for older base year.

### "LMDI decomposition shows NaN values"

**Cause:** Zero or negative emissions in one of the periods (LMDI requires positive values).

**Solution:** Ensure both period start and end emissions are positive:
```python
# Both must be positive
assert emissions_start > 0
assert emissions_end > 0
```

### "Corrective action plan exceeds budget"

**Cause:** Available initiatives cost more than the budget constraint.

**Solution:** Either increase the budget or accept partial gap closure:
```python
result = engine.plan(
    gap_tco2e=22_000,
    max_budget_usd=5_000_000,  # Increase budget
    optimization_strategy="maximum_reduction",  # Or use different strategy
)
```

---

**End of User Guide**
