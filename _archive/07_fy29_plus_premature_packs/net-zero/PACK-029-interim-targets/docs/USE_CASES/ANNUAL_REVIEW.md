# Use Case Guide: Annual Progress Review

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Workflow:** Annual Progress Review Workflow

---

## Table of Contents

1. [Use Case Overview](#use-case-overview)
2. [Personas and Roles](#personas-and-roles)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Year-Over-Year Comparison](#year-over-year-comparison)
6. [Carbon Budget Assessment](#carbon-budget-assessment)
7. [Trend Forecasting](#trend-forecasting)
8. [Pathway Adherence Scoring](#pathway-adherence-scoring)
9. [LMDI Variance Analysis Integration](#lmdi-variance-analysis-integration)
10. [Annual Report Generation](#annual-report-generation)
11. [Regulatory Reporting Integration](#regulatory-reporting-integration)
12. [Worked Example: Services Company](#worked-example-services-company)
13. [Troubleshooting](#troubleshooting)
14. [FAQ](#faq)

---

## Use Case Overview

### Scenario

The Head of Sustainability at a professional services firm needs to conduct the annual review of their net-zero progress. The company committed to SBTi-aligned targets in 2023, with a 2021 base year and a 2030 near-term target. Three years into the journey, they need a comprehensive assessment of where they stand, whether they are on track, what drove the changes, and what the trajectory looks like going forward.

### Business Value

| Benefit | Description |
|---------|-------------|
| Strategic assessment | Comprehensive view of decarbonization progress |
| Board reporting | Data-driven annual sustainability report to the board |
| CDP submission | Automated C4.1/C4.2 data for annual CDP response |
| TCFD compliance | Metrics and Targets pillar content for annual report |
| SBTi compliance | Annual progress disclosure per SBTi requirements |
| Budget planning | Carbon budget status informs next year's CapEx decisions |
| Course correction | Identify and address trajectory deviations early |

### Workflow Phases

```
Phase 1: AnnualDataCollect
    Aggregate full-year emissions from quarterly data via MRV Bridge
    |
    v
Phase 2: YoYComparison
    Compare current year against previous year and base year
    using Annual Review Engine
    |
    v
Phase 3: BudgetCheck
    Assess cumulative carbon budget status
    using Carbon Budget Tracker Engine
    |
    v
Phase 4: TrendForecast
    Project forward trajectory to 2030
    using Trend Extrapolation Engine
    |
    v
Phase 5: AnnualReport
    Generate comprehensive annual progress report
    using Annual Progress Report Template
```

---

## Personas and Roles

| Persona | Role | Annual Review Responsibilities |
|---------|------|-------------------------------|
| Head of Sustainability | Report owner | Orchestrate review, present to board |
| Sustainability Analyst | Data preparer | Compile data, run engines, draft report |
| Operations VP | Operational insight | Explain facility-level variances |
| CFO | Financial sponsor | Review investment requirements |
| CEO | Executive accountability | Present to board and external stakeholders |
| External auditor | Assurance provider | Verify emissions data and target progress |

---

## Prerequisites

### Data Requirements

| Data Item | Source | Required |
|-----------|--------|----------|
| Full-year Scope 1 emissions | MRV Bridge (Agents 001-008) | Yes |
| Full-year Scope 2 emissions | MRV Bridge (Agents 009-013) | Yes |
| Full-year Scope 3 emissions | MRV Bridge (Agents 014-030) | If material |
| Revenue / production data | Finance system | Yes (for intensity) |
| 4 quarterly monitoring results | Quarterly Monitoring Engine | Recommended |
| Previous year annual review | Database | For YoY comparison |
| Base year emissions | PACK-021 Bridge | Yes |
| Interim target milestones | Interim Target Engine | Yes |

### Timeline

| Week | Activity | Owner |
|------|----------|-------|
| Jan W1-2 | Finalize Q4 data, complete annual emissions | Sustainability Analyst |
| Jan W3 | Run data quality checks | Sustainability Analyst |
| Jan W4 | Run Annual Review Workflow | Sustainability Analyst |
| Feb W1 | Run Variance Analysis (LMDI) | Sustainability Analyst |
| Feb W2 | Draft annual report | Sustainability Analyst |
| Feb W3 | Internal review | Head of Sustainability + CFO |
| Feb W4 | Board presentation | Head of Sustainability |
| Mar-Apr | External assurance (if applicable) | External auditor |
| May-Jul | CDP submission using annual review data | Sustainability Analyst |

---

## Step-by-Step Walkthrough

### Step 1: Aggregate Full-Year Emissions

```python
from integrations.mrv_bridge import MRVBridge

mrv = MRVBridge(config=pack_config)

# Collect full-year emissions
annual_data = await mrv.collect_annual_emissions(
    entity_id="entity-001",
    year=2024,
    scopes=["scope_1", "scope_2", "scope_3"],
)

print(f"Full-year 2024 emissions:")
print(f"  Scope 1: {annual_data.scope_1_tco2e:,.0f} tCO2e")
print(f"  Scope 2 (location): {annual_data.scope_2_location_tco2e:,.0f} tCO2e")
print(f"  Scope 2 (market): {annual_data.scope_2_market_tco2e:,.0f} tCO2e")
print(f"  Scope 3: {annual_data.scope_3_tco2e:,.0f} tCO2e")
print(f"  Total (market): {annual_data.total_market_tco2e:,.0f} tCO2e")
```

### Step 2: Run Annual Review Engine

```python
from engines.annual_review_engine import AnnualReviewEngine

engine = AnnualReviewEngine()

review_input = AnnualReviewInput(
    entity_id="entity-001",
    reporting_year=2024,
    actual_emissions={
        "scope_1": Decimal("22000"),
        "scope_2_market": Decimal("13000"),
        "scope_3": Decimal("48000"),
    },
    target_emissions={
        "scope_1_2": Decimal("37200"),  # 2024 milestone from interim targets
        "scope_3": Decimal("52800"),    # 2024 milestone for Scope 3
    },
    base_year_emissions={
        "scope_1": Decimal("28000"),
        "scope_2_market": Decimal("18000"),
        "scope_3": Decimal("55000"),
    },
    previous_year_emissions={
        "scope_1": Decimal("24000"),
        "scope_2_market": Decimal("15000"),
        "scope_3": Decimal("50000"),
    },
    activity_data={
        "revenue_m_usd": Decimal("620"),
        "previous_year_revenue": Decimal("550"),
        "base_year_revenue": Decimal("480"),
        "headcount": 4200,
    },
)

result = await engine.calculate(review_input)
```

### Step 3: Assess Carbon Budget

```python
from engines.carbon_budget_tracker_engine import CarbonBudgetTrackerEngine

budget_engine = CarbonBudgetTrackerEngine()

budget_input = CarbonBudgetInput(
    entity_id="entity-001",
    base_year=2021,
    target_year=2030,
    base_year_emissions=Decimal("46000"),   # Scope 1+2 base year
    target_year_emissions=Decimal("24610"),  # 46.5% reduction
    annual_actuals={
        2022: Decimal("43000"),
        2023: Decimal("39000"),
        2024: Decimal("35000"),
    },
)

budget_result = await budget_engine.calculate(budget_input)

print(f"Carbon budget status:")
print(f"  Total budget: {budget_result.total_budget:,.0f} tCO2e")
print(f"  Budget used: {budget_result.budget_used:,.0f} tCO2e")
print(f"  Budget remaining: {budget_result.budget_remaining:,.0f} tCO2e")
print(f"  % used: {budget_result.pct_used:.1f}%")
print(f"  Burn rate: {budget_result.current_burn_rate:,.0f} tCO2e/year")
print(f"  Years until exhaustion: {budget_result.years_remaining:.1f}")
```

### Step 4: Run Trend Forecast

```python
from engines.trend_extrapolation_engine import TrendExtrapolationEngine

trend_engine = TrendExtrapolationEngine()

trend_input = TrendExtrapolationInput(
    entity_id="entity-001",
    historical_emissions=[
        {"year": 2021, "emissions": Decimal("46000")},
        {"year": 2022, "emissions": Decimal("43000")},
        {"year": 2023, "emissions": Decimal("39000")},
        {"year": 2024, "emissions": Decimal("35000")},
    ],
    forecast_horizon=6,  # Project to 2030
    target_year=2030,
    target_emissions=Decimal("24610"),
)

trend_result = await trend_engine.calculate(trend_input)

print(f"Selected model: {trend_result.selected_model}")
print(f"2030 projection: {trend_result.forecast_2030:,.0f} tCO2e")
print(f"2030 target: 24,610 tCO2e")
print(f"Projected gap/surplus: {trend_result.gap_to_target:,.0f} tCO2e")
print(f"On track: {trend_result.on_track}")
```

### Step 5: Generate Annual Report

```python
from templates.annual_progress_report import AnnualProgressReport

report_template = AnnualProgressReport()

annual_report = report_template.render(
    annual_review=result,
    budget_status=budget_result,
    trend_forecast=trend_result,
    variance_analysis=variance_result,  # If run separately
    format="html",
    include_charts=True,
)

# Save report
with open("Annual_Progress_Report_2024.html", "w") as f:
    f.write(annual_report)
```

---

## Year-Over-Year Comparison

### Comparison Table

The Annual Review Engine generates a comprehensive year-over-year comparison:

```
Year-Over-Year Comparison (2024 vs 2023)
=========================================

| Metric                  | 2023      | 2024      | Change   | % Change |
|-------------------------|-----------|-----------|----------|----------|
| Scope 1 (tCO2e)        | 24,000    | 22,000    | -2,000   | -8.3%    |
| Scope 2 Market (tCO2e) | 15,000    | 13,000    | -2,000   | -13.3%   |
| Scope 1+2 (tCO2e)      | 39,000    | 35,000    | -4,000   | -10.3%   |
| Scope 3 (tCO2e)        | 50,000    | 48,000    | -2,000   | -4.0%    |
| Total (tCO2e)          | 89,000    | 83,000    | -6,000   | -6.7%    |
| Revenue (M USD)        | 550       | 620       | +70      | +12.7%   |
| Intensity (tCO2e/M$)   | 161.8     | 133.9     | -27.9    | -17.2%   |
| Headcount              | 3,800     | 4,200     | +400     | +10.5%   |
| Per capita (tCO2e/emp) | 23.4      | 19.8      | -3.6     | -15.4%   |
```

### Base Year Comparison

```
Progress from Base Year (2024 vs 2021)
=======================================

| Metric                  | 2021      | 2024      | Reduction | % of Target |
|                         | (Base)    | (Current) | Achieved  | Achieved    |
|-------------------------|-----------|-----------|-----------|-------------|
| Scope 1+2 (tCO2e)      | 46,000    | 35,000    | 11,000    | 51.4%       |
| Scope 3 (tCO2e)        | 55,000    | 48,000    | 7,000     | 46.7%       |
| Total (tCO2e)          | 101,000   | 83,000    | 18,000    | 50.0%       |

Target: Scope 1+2 = 24,610 tCO2e by 2030 (46.5% reduction = 21,390 tCO2e to cut)
Progress: 11,000 / 21,390 = 51.4% of the way to the 2030 target
Years elapsed: 3 of 9 (33.3% of time elapsed)
Status: AHEAD OF SCHEDULE (51.4% progress in 33.3% of time)
```

---

## Carbon Budget Assessment

### Cumulative Carbon Budget

The carbon budget represents the total cumulative emissions allowed between the base year and target year:

```
Carbon Budget Calculation (Trapezoidal Integration):

Year    | Target  | Actual  | Cumulative Target | Cumulative Actual | Budget Status
--------|---------|---------|-------------------|-------------------|-------------
2021    | 46,000  | 46,000  | 46,000            | 46,000            | On budget
2022    | 43,620  | 43,000  | 89,620            | 89,000            | Under budget
2023    | 41,240  | 39,000  | 130,860           | 128,000           | Under budget
2024    | 38,860  | 35,000  | 169,720           | 163,000           | Under budget
...
2030    | 24,610  | --      | --                 | --                | --

Total budget (2021-2030): 317,745 tCO2e (area under target pathway)
Budget used (2021-2024): 163,000 tCO2e
Budget remaining: 154,745 tCO2e
Budget utilization: 51.3% used in 44.4% of time = UNDER BUDGET
```

### Budget Burn Rate Analysis

```python
# Current burn rate
burn_rate = budget_result.budget_used / years_elapsed  # 40,750 tCO2e/year

# Allowed burn rate for remaining period
allowed_rate = budget_result.budget_remaining / years_remaining  # 25,791 tCO2e/year

# Burn rate ratio
burn_ratio = burn_rate / allowed_rate  # 1.58

# Interpretation
if burn_ratio < 1.0:
    status = "UNDER BUDGET - current rate is sustainable"
elif burn_ratio < 1.2:
    status = "ON BUDGET - minor acceleration needed"
elif burn_ratio < 1.5:
    status = "SLIGHTLY OVER BUDGET - reduction acceleration needed"
else:
    status = "OVER BUDGET - significant corrective action required"
```

---

## Trend Forecasting

### Forward Projection

The Trend Extrapolation Engine projects the emissions trajectory forward:

```
Historical + Projected Emissions (Linear Regression)
=====================================================

Year    | Actual  | Projected | Target  | Gap
--------|---------|-----------|---------|--------
2021    | 46,000  | --        | 46,000  | 0
2022    | 43,000  | --        | 43,620  | -620
2023    | 39,000  | --        | 41,240  | -2,240
2024    | 35,000  | --        | 38,860  | -3,860
2025    | --      | 31,200    | 36,480  | -5,280
2026    | --      | 27,500    | 34,100  | -6,600
2027    | --      | 23,800    | 31,720  | -7,920
2028    | --      | 20,100    | 29,340  | -9,240
2029    | --      | 16,400    | 26,960  | -10,560
2030    | --      | 12,700    | 24,610  | -11,910

Model: Linear regression (R^2 = 0.994, MAPE = 1.8%)
Forecast: 2030 projection of 12,700 tCO2e is WELL BELOW target of 24,610 tCO2e
Surplus: 11,910 tCO2e below target
Conclusion: Company is significantly ahead of schedule
```

### Confidence Intervals

```
Year    | Point    | 80% CI Lower | 80% CI Upper | 95% CI Lower | 95% CI Upper
--------|----------|-------------|-------------|-------------|-------------
2025    | 31,200   | 29,500      | 32,900      | 28,800      | 33,600
2026    | 27,500   | 25,000      | 30,000      | 23,800      | 31,200
2027    | 23,800   | 20,400      | 27,200      | 18,700      | 28,900
2028    | 20,100   | 15,700      | 24,500      | 13,500      | 26,700
2029    | 16,400   | 11,000      | 21,800      | 8,300       | 24,500
2030    | 12,700   | 6,300       | 19,100      | 3,100       | 22,300
```

---

## Pathway Adherence Scoring

### Adherence Score (0-100)

PACK-029 computes a pathway adherence score based on how closely actual emissions follow the target pathway:

```
Adherence Score = 100 - SUM( weight_t * |deviation_t| ) * penalty_factor

Where:
    deviation_t = (actual_t - target_t) / target_t * 100  (% deviation)
    weight_t = more recent years weighted higher
    penalty_factor = extra penalty for positive deviations (above target)
```

### Score Interpretation

| Score Range | Rating | Interpretation |
|-------------|--------|---------------|
| 90-100 | Excellent | Closely following target pathway |
| 80-89 | Good | Minor deviations, on track overall |
| 70-79 | Fair | Some deviations, monitor closely |
| 60-69 | Concerning | Significant deviations, corrective action needed |
| Below 60 | Poor | Major trajectory issues, urgent action required |

### Example Calculation

```
Year | Target  | Actual  | Deviation (%) | Weighted Score
-----|---------|---------|---------------|---------------
2022 | 43,620  | 43,000  | -1.4%         | +0.5 (below = bonus)
2023 | 41,240  | 39,000  | -5.4%         | +1.8 (below = bonus)
2024 | 38,860  | 35,000  | -9.9%         | +3.3 (below = bonus)

Adherence Score: 100 + 5.6 (bonuses for being below target) = 100 (capped)
Rating: EXCELLENT
```

---

## LMDI Variance Analysis Integration

### Connecting Annual Review to Variance Analysis

After the annual review, PACK-029 can run a detailed LMDI decomposition to explain the drivers of emission change:

```python
from engines.variance_analysis_engine import VarianceAnalysisEngine

variance_engine = VarianceAnalysisEngine()

variance_input = VarianceAnalysisInput(
    entity_id="entity-001",
    period_0={"year": 2023, "emissions": Decimal("39000"), "activity": Decimal("550")},
    period_t={"year": 2024, "emissions": Decimal("35000"), "activity": Decimal("620")},
    decomposition_type="additive",
)

variance_result = await variance_engine.calculate(variance_input)

print(f"Activity effect: {variance_result.activity_effect:,.0f} tCO2e")
print(f"Intensity effect: {variance_result.intensity_effect:,.0f} tCO2e")
print(f"Total change: {variance_result.total_change:,.0f} tCO2e")
print(f"Perfect decomposition: {variance_result.is_perfect_decomposition}")
```

### Variance Narrative for Annual Report

```
Variance Analysis: 2023 to 2024
================================

Total emissions decreased by 4,000 tCO2e (-10.3%):

LMDI Decomposition:
    Activity effect:   +4,800 tCO2e   Revenue grew 12.7%, which would have
                                        increased emissions by 4,800 tCO2e
                                        if intensity had remained constant.

    Intensity effect:  -8,800 tCO2e   Emission intensity improved 20.4%
                                        (from 70.9 to 56.5 tCO2e/M USD),
                                        removing 8,800 tCO2e.

    Total:             -4,000 tCO2e   = 4,800 + (-8,800) = -4,000 (PERFECT)

Key drivers of intensity improvement:
    1. Green electricity PPA (Jan 2024):         -3,500 tCO2e
    2. Office consolidation (Mar 2024):          -1,800 tCO2e
    3. Business travel policy changes:           -1,500 tCO2e
    4. IT server virtualization:                 -1,200 tCO2e
    5. LED lighting upgrade:                       -800 tCO2e
    Total identified:                            -8,800 tCO2e

Conclusion: Strong decarbonization despite significant business growth.
The company is decoupling emissions from revenue growth effectively.
```

---

## Annual Report Generation

### Report Structure

The Annual Progress Report includes:

```
1. Executive Summary
   - Overall status (GREEN/AMBER/RED)
   - Key metrics at a glance
   - Headline achievement

2. Year-Over-Year Comparison
   - Scope 1, 2, 3 emissions table
   - Intensity metrics
   - Absolute and relative changes

3. Progress Against Targets
   - Base year comparison
   - % of target achieved
   - Pathway adherence score

4. Carbon Budget Status
   - Cumulative budget tracking
   - Remaining budget
   - Burn rate analysis

5. Variance Analysis (LMDI)
   - Activity, intensity, structural effects
   - Waterfall chart
   - Root cause narrative

6. Forward Projection
   - Trend forecast to 2030
   - Confidence intervals
   - Model selection rationale

7. Corrective Actions (if off-track)
   - Gap quantification
   - Initiative portfolio
   - Implementation timeline

8. Next Year Outlook
   - Planned initiatives
   - Expected trajectory
   - Risks and mitigations

9. Appendix
   - Methodology notes
   - Data quality assessment
   - SBTi validation status
   - Provenance hashes
```

---

## Regulatory Reporting Integration

### CDP Annual Submission

```python
# Generate CDP export from annual review data
from templates.cdp_export_template import CDPExportTemplate

cdp_template = CDPExportTemplate()
cdp_package = cdp_template.render(
    annual_review=result,
    interim_targets=target_result,
    variance_analysis=variance_result,
    format="xlsx",  # CDP Online Response System format
)
```

### TCFD Annual Disclosure

```python
# Generate TCFD Metrics and Targets content
from templates.tcfd_disclosure_template import TCFDDisclosureTemplate

tcfd_template = TCFDDisclosureTemplate()
tcfd_content = tcfd_template.render(
    annual_review=result,
    interim_targets=target_result,
    trend_forecast=trend_result,
    format="md",  # For integration into sustainability report
)
```

### SBTi Annual Progress

```python
# Generate SBTi annual disclosure
from integrations.sbti_portal_bridge import SBTiPortalBridge

sbti_bridge = SBTiPortalBridge(config=pack_config)
sbti_disclosure = await sbti_bridge.prepare_annual_disclosure(
    annual_review=result,
    sbti_validation=validation_result,
)
```

---

## Worked Example: Services Company

### Company Profile

| Attribute | Value |
|-----------|-------|
| Company | Bright Consulting Group |
| Sector | Professional services |
| Employees | 4,200 |
| Revenue | USD 620M (2024) |
| Base year | 2021 |
| Base year Scope 1+2 | 46,000 tCO2e |
| Base year Scope 3 | 55,000 tCO2e |
| Near-term target (2030) | Scope 1+2: 24,610 tCO2e (-46.5%) |
| Near-term target (2030) | Scope 3: 41,250 tCO2e (-25%) |

### 2024 Annual Review Results

```
Overall Status: GREEN (ahead of schedule)

Scope 1+2:
    2024 actual: 35,000 tCO2e
    2024 target: 38,860 tCO2e
    Gap: -3,860 tCO2e (BELOW target)
    Reduction from base year: 23.9%
    % of target achieved: 51.4%
    Years elapsed: 33.3% of timeline

Scope 3:
    2024 actual: 48,000 tCO2e
    2024 target: 52,800 tCO2e
    Gap: -4,800 tCO2e (BELOW target)
    Reduction from base year: 12.7%
    % of target achieved: 50.9%

Carbon Budget:
    Total budget: 317,745 tCO2e (Scope 1+2)
    Budget used: 163,000 tCO2e (51.3%)
    Budget remaining: 154,745 tCO2e
    Status: UNDER BUDGET

Pathway Adherence Score: 100/100 (EXCELLENT)

Trend Forecast (2030):
    Projected Scope 1+2: 12,700 tCO2e
    Target: 24,610 tCO2e
    Surplus: 11,910 tCO2e
    Status: WELL AHEAD OF TARGET

Board Summary:
    "Bright Consulting is significantly ahead of its 2030 SBTi target.
    Scope 1+2 emissions have been reduced 23.9% from the 2021 baseline
    (target: 46.5% by 2030). At the current rate of reduction, the company
    will achieve its 2030 target approximately 4 years early (by 2026).
    Revenue has grown 29.2% over the same period, demonstrating successful
    decoupling of emissions from business growth."
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Resolution |
|-------|-------|-----------|
| Annual emissions differ from sum of quarters | Restatements or late data | Reconcile quarterly vs annual; use annual as authoritative |
| Base year emissions changed | Acquisition, divestment, or methodology | Run Target Recalibration Engine |
| Scope 3 data incomplete | Supplier data lag | Note coverage %; estimate missing using PACK-029 |
| LMDI shows unexpected structural effect | Organizational changes | Review segment definitions |
| Forecast diverges significantly from target | Unusual recent year (e.g., COVID) | Consider excluding anomalous years or using Holt's damping |
| Pathway adherence score drops | Single bad year | Assess whether structural or temporary |

---

## FAQ

**Q: When should I run the annual review?**
A: As soon as full-year emissions data is finalized, typically in January or February for the previous year. Run before CDP submission deadline (usually May-July).

**Q: Can I include Scope 3 in the annual review if data is estimated?**
A: Yes. PACK-029 tracks data quality alongside emissions. Include estimates but flag them as such. The data quality assessment will reflect the estimation level.

**Q: How does the annual review relate to quarterly monitoring?**
A: The annual review aggregates all four quarters and provides a comprehensive full-year perspective. Quarterly monitoring provides early warning; annual review provides the definitive assessment for reporting.

**Q: What if my annual result contradicts my quarterly monitoring?**
A: This can happen if Q4 data was restated or if quarterly data was preliminary. The annual review uses authoritative full-year data and should be treated as definitive.

**Q: Should I re-run the annual review after an external audit?**
A: Yes, if the audit results in material changes to emissions data. Re-running will update the provenance hash and audit trail, maintaining traceability.

**Q: How do I handle the annual review if we made an acquisition mid-year?**
A: First run the Target Recalibration Engine to adjust the base year. Then run the annual review using the recalculated targets. PACK-029 handles the pro-rata allocation automatically.

---

**End of Annual Review Use Case Guide**
