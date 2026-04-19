# Use Case Guide: Corrective Action Planning

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Workflow:** Corrective Action Planning Workflow

---

## Table of Contents

1. [Use Case Overview](#use-case-overview)
2. [Personas and Roles](#personas-and-roles)
3. [Prerequisites](#prerequisites)
4. [Trigger Conditions](#trigger-conditions)
5. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
6. [Gap Assessment](#gap-assessment)
7. [Initiative Identification](#initiative-identification)
8. [MACC Optimization](#macc-optimization)
9. [Scenario Analysis](#scenario-analysis)
10. [Action Plan Approval Process](#action-plan-approval-process)
11. [Implementation Tracking](#implementation-tracking)
12. [Worked Example: Retail Company](#worked-example-retail-company)
13. [Worked Example: Financial Services](#worked-example-financial-services)
14. [Integration with Other Workflows](#integration-with-other-workflows)
15. [Troubleshooting](#troubleshooting)
16. [FAQ](#faq)

---

## Use Case Overview

### Scenario

A retail company has been tracking its emissions quarterly against SBTi-aligned interim targets. After two consecutive AMBER quarters (Q3 and Q4 2024) and a RED result in Q1 2025, the sustainability team needs to develop a corrective action plan to close the 8,000 tCO2e gap and get back on track before the next annual SBTi progress disclosure.

### Business Value

| Benefit | Description |
|---------|-------------|
| Gap closure | Structured approach to closing the emissions gap |
| Cost optimization | MACC analysis selects the most cost-effective initiatives |
| Board credibility | Demonstrates proactive management of targets |
| SBTi compliance | Maintains pathway adherence for SBTi annual disclosure |
| CDP scoring | Shows corrective action capability (Management/Leadership score) |
| Budget justification | Data-driven case for decarbonization investment |
| Risk mitigation | Reduces risk of missing near-term target deadline |

### Workflow Phases

```
Phase 1: GapQuantification
    Quantify the current gap, forward gap, and acceleration needed
    using Corrective Action Engine + Annual Review data
    |
    v
Phase 2: InitiativeScanning
    Identify available abatement initiatives from initiative library
    and PACK-028 abatement levers
    |
    v
Phase 3: MACCOptimization
    Build MACC, optimize portfolio under budget/timeline constraints
    using Corrective Action Engine
    |
    v
Phase 4: ScheduleGen
    Create phased implementation schedule with milestones
    |
    v
Phase 5: ActionPlanReport
    Generate corrective action plan report for approval
    using Corrective Action Plan Report Template
```

---

## Personas and Roles

| Persona | Role | Corrective Action Responsibilities |
|---------|------|-------------------------------------|
| Sustainability Manager | Plan owner | Develop plan, present to leadership |
| Operations Director | Implementation lead | Assess feasibility, commit to delivery |
| Procurement Manager | Supplier engagement | Evaluate renewable/green procurement options |
| Facilities Manager | Building operations | Identify and deliver efficiency improvements |
| Fleet Manager | Transport operations | Plan fleet electrification and optimization |
| CFO | Budget approver | Approve capital expenditure for initiatives |
| Board ESG Committee | Governance | Review and endorse corrective action plan |

---

## Prerequisites

### Data Requirements

| Data Item | Source | Required |
|-----------|--------|----------|
| Current emissions (latest quarter) | Quarterly Monitoring Engine | Yes |
| Target emissions (annual + quarterly) | Interim Target Engine | Yes |
| LMDI variance analysis | Variance Analysis Engine | Recommended |
| Available budget | Finance team | Yes |
| Initiative catalog | Internal + PACK-028 | Yes |
| Implementation constraints | Operations team | Recommended |
| Previous corrective action results | Database | If applicable |

### When to Initiate

The Corrective Action Planning Workflow should be triggered when:

1. **Quarterly monitoring returns RED** (>5% above target)
2. **Three consecutive AMBER** quarters (automatic escalation)
3. **Annual review shows off-track** trajectory
4. **Trend forecast projects** missing the near-term target
5. **Carbon budget burn rate** exceeds allowed rate by >20%
6. **Management request** for proactive planning (even if on track)

---

## Trigger Conditions

### Automatic Triggers

```python
# Alert Generation Engine triggers
corrective_action_triggers = [
    {
        "condition": "rag_status == 'RED'",
        "action": "Initiate Corrective Action Planning Workflow",
        "urgency": "HIGH",
        "deadline": "Within 30 days",
    },
    {
        "condition": "consecutive_amber_quarters >= 3",
        "action": "Initiate Corrective Action Planning Workflow",
        "urgency": "MEDIUM",
        "deadline": "Within 45 days",
    },
    {
        "condition": "annualized_projection > annual_target * 1.10",
        "action": "Recommend Corrective Action review",
        "urgency": "MEDIUM",
        "deadline": "Before next quarterly review",
    },
    {
        "condition": "carbon_budget_burn_ratio > 1.20",
        "action": "Recommend Corrective Action review",
        "urgency": "MEDIUM",
        "deadline": "Within 60 days",
    },
]
```

### Manual Triggers

Organizations may also initiate corrective action planning proactively:
- Before SBTi annual disclosure (to address any gaps)
- After a significant organizational change (acquisition, new facility)
- As part of annual budget planning (to justify decarbonization CapEx)
- When new abatement technologies become available

---

## Step-by-Step Walkthrough

### Step 1: Quantify the Gap

```python
from engines.corrective_action_engine import CorrectiveActionEngine

engine = CorrectiveActionEngine()

gap_input = GapQuantificationInput(
    entity_id="entity-001",
    current_year=2025,
    target_year=2030,
    actual_emissions=Decimal("73000"),      # Latest annualized
    target_emissions=Decimal("65000"),       # 2025 annual target
    base_year_emissions=Decimal("80000"),    # 2021 baseline
    target_year_emissions=Decimal("42800"),  # 2030 target
    historical_actuals={
        2022: Decimal("76000"),
        2023: Decimal("75000"),
        2024: Decimal("73000"),
    },
)

gap_result = await engine.quantify_gap(gap_input)

print(f"Current year gap: {gap_result.current_gap:,.0f} tCO2e")
print(f"Gap as % of target: {gap_result.gap_pct:.1f}%")
print(f"Years behind pathway: {gap_result.years_behind:.1f}")
print(f"Required annual rate: {gap_result.required_annual_rate:.1f}%")
print(f"Original annual rate: {gap_result.original_annual_rate:.1f}%")
print(f"Acceleration needed: {gap_result.acceleration_needed:.1f}%")
```

### Step 2: Identify Available Initiatives

```python
# Load initiative catalog
initiatives = await engine.scan_initiatives(
    entity_id="entity-001",
    scopes=["scope_1", "scope_2"],
    budget_max=Decimal("2000000"),
    timeline_max_months=24,
)

print(f"Found {len(initiatives)} available initiatives:")
for init in initiatives:
    print(f"  - {init.name}: {init.abatement_potential:,.0f} tCO2e, "
          f"${init.capital_cost:,.0f}, {init.implementation_time_months} months")
```

### Step 3: Build MACC and Optimize Portfolio

```python
optimization_input = MACCOptimizationInput(
    gap_tco2e=gap_result.current_gap,
    initiatives=initiatives,
    budget_constraint=Decimal("2000000"),
    timeline_months=24,
    discount_rate=Decimal("0.05"),
)

portfolio = await engine.optimize_portfolio(optimization_input)

print(f"\nOptimized Portfolio:")
print(f"  Selected initiatives: {len(portfolio.selected_initiatives)}")
print(f"  Total abatement: {portfolio.total_abatement:,.0f} tCO2e")
print(f"  Total capital: ${portfolio.total_capital:,.0f}")
print(f"  Annual operating impact: ${portfolio.annual_operating_cost:,.0f}")
print(f"  Average cost/tCO2e: ${portfolio.average_cost_per_tco2e:,.0f}")
print(f"  Gap closed: {portfolio.gap_closed}")
print(f"  NPV: ${portfolio.npv:,.0f}")
print(f"  Payback: {portfolio.payback_years:.1f} years")
```

### Step 4: Run Scenario Analysis

```python
scenarios = await engine.analyze_scenarios(portfolio)

for scenario_name, scenario in scenarios.items():
    print(f"\n{scenario_name.upper()} Scenario:")
    print(f"  Abatement: {scenario.total_abatement:,.0f} tCO2e")
    print(f"  Cost: ${scenario.total_cost:,.0f}")
    print(f"  Gap closed: {scenario.gap_closed}")
    if scenario.catch_up_year:
        print(f"  Catch-up year: {scenario.catch_up_year}")
    else:
        print(f"  Residual gap: {scenario.residual_gap:,.0f} tCO2e")
```

### Step 5: Generate Implementation Schedule

```python
schedule = await engine.generate_schedule(
    portfolio=portfolio,
    start_date="2025-04-01",
    max_parallel_initiatives=3,
    annual_budget_phasing=[
        Decimal("800000"),   # Year 1
        Decimal("1200000"),  # Year 2
    ],
)

for phase in schedule.phases:
    print(f"\n{phase.name} (Month {phase.start_month}-{phase.end_month}):")
    for init in phase.initiatives:
        print(f"  - {init.name}: {init.abatement_by_year_end:,.0f} tCO2e")
    print(f"  Phase capital: ${phase.capital:,.0f}")
    print(f"  Phase abatement: {phase.abatement:,.0f} tCO2e")
```

### Step 6: Generate Action Plan Report

```python
from templates.corrective_action_plan_report import CorrectiveActionPlanReport

report_template = CorrectiveActionPlanReport()

action_plan = report_template.render(
    gap_analysis=gap_result,
    portfolio=portfolio,
    scenarios=scenarios,
    schedule=schedule,
    variance_analysis=variance_result,  # Optional: explains root cause
    format="html",
    include_charts=True,
)

with open("Corrective_Action_Plan_2025.html", "w") as f:
    f.write(action_plan)
```

---

## Gap Assessment

### Gap Visualization

```
Emissions Trajectory: Actual vs Target
========================================

  80,000 |* (2021 baseline)
         | *
  76,000 |  * (2022 actual)
         |   *   ...target pathway
  72,000 |    * (2023 actual)
         |    *   *
  68,000 |     * (2024 actual)
         |      *   *
  64,000 |       ....* (2025 target = 65,000)
         |        GAP *
  60,000 |            * * (2025 actual projected = 73,000)
         |
                    Gap = 8,000 tCO2e
```

### Gap Components

```json
{
  "gap_assessment": {
    "current_gap": {
      "value_tco2e": 8000,
      "as_pct_of_target": 12.3,
      "classification": "Significant (>5%)"
    },
    "forward_gap": {
      "value_tco2e": 42000,
      "description": "Cumulative gap through 2030 under current trajectory"
    },
    "acceleration_needed": {
      "current_annual_rate_pct": 2.5,
      "required_annual_rate_pct": 6.1,
      "acceleration_factor": 2.4,
      "description": "Must reduce 2.4x faster than current rate"
    },
    "years_behind": {
      "value": 2.1,
      "description": "Approximately 2 years behind the planned pathway"
    }
  }
}
```

---

## Initiative Identification

### Initiative Sources

| Source | Description | PACK-029 Integration |
|--------|-------------|---------------------|
| Internal catalog | Company's known decarbonization opportunities | Manual input or database |
| PACK-028 levers | Sector-specific abatement levers from SDA pathways | PACK-028 Bridge |
| Industry benchmarks | Typical initiatives for the sector | Preset-based suggestions |
| Previous plans | Initiatives from past corrective action plans | Database query |
| Supplier proposals | Vendor quotes for equipment/services | Manual input |

### Initiative Categories

| Category | Typical Initiatives | Cost Range ($/tCO2e) |
|----------|--------------------|-----------------------|
| Energy efficiency | LED, HVAC, insulation, controls | -$50 to -$10 |
| Renewable electricity | Solar PV, wind PPA, green tariff | -$10 to +$30 |
| Fuel switching | Gas to electric, heat pumps | +$20 to +$80 |
| Process optimization | Waste reduction, yield improvement | -$30 to +$20 |
| Fleet electrification | EV trucks, vans, cars | +$30 to +$100 |
| Building retrofit | Deep retrofit, net-zero buildings | +$50 to +$150 |
| Supply chain | Supplier engagement, sustainable sourcing | +$10 to +$60 |
| Novel technology | Green hydrogen, CCS, DAC | +$80 to +$300 |

### Initiative Data Entry

```python
initiatives = [
    AbatementInitiative(
        initiative_id="init-001",
        name="LED Lighting Upgrade (all stores)",
        description="Replace all fluorescent lighting with LED across 50 retail stores",
        scope=ScopeType.SCOPE_2,
        category="energy_efficiency",
        abatement_potential=Decimal("2500"),
        capital_cost=Decimal("375000"),
        annual_operating_cost=Decimal("-120000"),  # Negative = savings
        lifetime_years=10,
        implementation_time_months=6,
        confidence_level=Decimal("0.95"),
        co_benefits=["Reduced maintenance", "Better lighting quality", "Lower electricity bills"],
        prerequisites=[],
    ),
    AbatementInitiative(
        initiative_id="init-002",
        name="Refrigerant Leak Detection & Repair",
        description="Install automated leak detection on all refrigeration systems",
        scope=ScopeType.SCOPE_1,
        category="fugitive_emissions",
        abatement_potential=Decimal("1800"),
        capital_cost=Decimal("250000"),
        annual_operating_cost=Decimal("-45000"),
        lifetime_years=8,
        implementation_time_months=4,
        confidence_level=Decimal("0.88"),
        co_benefits=["Regulatory compliance", "Reduced refrigerant costs"],
        prerequisites=[],
    ),
    # ... more initiatives
]
```

---

## MACC Optimization

### MACC Chart Interpretation

```
Cost per tCO2e (USD)
  $200 |
       |
  $150 |                                              +---------+
       |                                              | Green H2|
  $100 |                                    +---------+ ($120)  |
       |                                    |EV Fleet |         |
   $50 |                          +---------+ ($55)   |         |
       |                +---------+ Heat    |         |         |
   $25 |      +---------+ Solar   | Pump    |         |         |
       |      | Refrig  | PV      | ($35)   |         |         |
    $0 +------+---------+---------+---------+---------+---------+-->
       |      | Leak    | ($15)   |                              Cumulative
  -$25 |      | ($-35)  |         |                              Abatement
       |      |         |         |                              (tCO2e)
  -$50 +------+---------+
       | LED  |
  -$75 |(-$48)|
       +------+
       0     2,500    5,000    7,500   10,000  13,000  18,000

       Gap line: 8,000 tCO2e -------->|
```

### Portfolio Selection Result

```
Selected Portfolio (Gap = 8,000 tCO2e, Budget = $2,000,000):

Priority | Initiative          | Abatement | Cost/tCO2e | CapEx    | Cum. Abatement
---------|---------------------|-----------|------------|----------|---------------
   1     | LED Lighting        |   2,500   |   -$48     | $375,000 |   2,500
   2     | Refrigerant L&R     |   1,800   |   -$35     | $250,000 |   4,300
   3     | Solar PV (rooftop)  |   2,200   |   +$15     | $500,000 |   6,500
   4     | Heat Pump (stores)  |   1,500   |   +$35     | $400,000 |   8,000  GAP CLOSED
         |                     |           |            |          |
TOTAL    |                     |   8,000   |    -$6     |$1,525,000|   8,000

Result: Gap fully closed within budget ($1,525,000 of $2,000,000)
Average cost: -$6/tCO2e (net savings portfolio!)
NPV (10-year): +$950,000
Payback: 3.2 years
```

---

## Scenario Analysis

### Three Scenarios

```
Scenario Comparison
===================

| Metric              | Optimistic | Baseline | Pessimistic |
|---------------------|-----------|----------|-------------|
| Total abatement     | 8,800     | 7,350    | 5,145       |
| Total capital       | $1,373,000| $1,525,000| $1,983,000 |
| Gap closed?         | Yes (+800)| Yes (partial)| No (-2,855)|
| Catch-up year       | Q3 2026   | Q1 2027  | --          |
| Net annual savings  | $198,000  | $165,000 | $75,000     |
| NPV                 | $1,200,000| $950,000 | $350,000    |

Note: Baseline scenario achieves 91.9% gap closure.
Pessimistic scenario requires additional initiatives to fully close.
```

### Sensitivity Analysis

```
Key Sensitivity Factors:
1. LED savings confidence (95%): 5% impact on total abatement
2. Refrigerant leak rate estimate: 12% impact on abatement
3. Solar PV capacity factor: 15% impact on renewable abatement
4. Heat pump COP assumption: 8% impact on fuel switching benefit
5. Electricity grid factor: 10% impact on Scope 2 reductions
```

---

## Action Plan Approval Process

### Approval Workflow

```
Step 1: Sustainability Manager drafts plan
    |
    v
Step 2: Operations Director reviews feasibility
    |
    v
Step 3: CFO reviews financials (NPV, payback, budget fit)
    |
    v
Step 4: Board ESG Committee endorses plan
    |
    v
Step 5: Implementation begins (Phase 1 quick wins)
```

### Board Presentation Template

```
CORRECTIVE ACTION PLAN -- BOARD SUMMARY
========================================

SITUATION:
- Q1 2025 emissions 12.3% above target (RED status)
- On current trajectory, will miss 2030 target by ~18,000 tCO2e

PLAN:
- 4 initiatives totaling 8,000 tCO2e abatement
- Total investment: $1,525,000
- Net annual savings: $165,000/year
- Payback: 3.2 years
- NPV (10-year): $950,000

TIMELINE:
- Phase 1 (Q2 2025): LED + Refrigerant (quick wins, $625,000)
- Phase 2 (Q3-Q4 2025): Solar PV + Heat Pumps ($900,000)
- Full abatement achieved: Q1 2027

SCENARIOS:
- Best case: Gap closed by Q3 2026
- Expected: Gap closed by Q1 2027
- Worst case: 64% of gap closed; additional measures needed

REQUEST:
- Approve $1,525,000 CapEx for FY2025-2026
- Authorize implementation of Phase 1 immediately

RISK:
- Without action, cumulative carbon budget will be exhausted by 2028
- SBTi annual disclosure will show off-track status
- CDP score likely downgraded from B to C
```

---

## Implementation Tracking

### Tracking Dashboard

```
Initiative Tracking Dashboard (Updated Monthly)
=================================================

| Initiative       | Status    | Impl %  | Abatement | Actual vs Plan |
|------------------|-----------|---------|-----------|----------------|
| LED Lighting     | On track  | 75%     | 1,875/2,500| Ahead (+5%)   |
| Refrigerant L&R  | Delayed   | 40%     | 720/1,800 | Behind (-15%)  |
| Solar PV         | On track  | 30%     | 0/2,200   | On plan        |
| Heat Pump        | Not started| 0%     | 0/1,500   | On plan        |

Total abatement to date: 2,595 / 8,000 tCO2e (32.4%)
Status: AMBER (Refrigerant initiative delayed)
```

### Monthly Check-In Process

```python
# Track implementation progress
tracking_input = ImplementationTrackingInput(
    entity_id="entity-001",
    reporting_month="2025-07",
    initiative_updates=[
        {"initiative_id": "init-001", "completion_pct": 75, "actual_abatement": Decimal("1875")},
        {"initiative_id": "init-002", "completion_pct": 40, "actual_abatement": Decimal("720")},
        {"initiative_id": "init-003", "completion_pct": 30, "actual_abatement": Decimal("0")},
        {"initiative_id": "init-004", "completion_pct": 0, "actual_abatement": Decimal("0")},
    ],
)

tracking_result = engine.track_implementation(tracking_input)
```

---

## Worked Example: Retail Company

### Company Profile

| Attribute | Value |
|-----------|-------|
| Company | GreenMart Retail Ltd |
| Sector | Retail (50 stores, 3 warehouses) |
| Base year (2021) | 120,000 tCO2e (Scope 1+2) |
| Target (2030) | 64,200 tCO2e (-46.5%) |
| 2024 actual | 105,000 tCO2e |
| 2025 target | 95,400 tCO2e |
| Current gap | 9,600 tCO2e (10.1% above target) |
| Available budget | $3,000,000 |

### Initiative Portfolio

```
Rank | Initiative                | Abatement | Cost/tCO2e | CapEx     | Time
-----|---------------------------|-----------|------------|-----------|------
  1  | Refrigerant management    | 3,200     | -$40       | $280,000  | 4 mo
  2  | LED retrofit (all stores) | 2,800     | -$30       | $420,000  | 8 mo
  3  | Building management system| 1,500     | -$15       | $200,000  | 3 mo
  4  | Solar PV (3 warehouses)   | 2,500     | +$20       | $750,000  | 12 mo
  5  | EV delivery fleet (phase 1)| 1,800    | +$45       | $600,000  | 6 mo
  6  | Green electricity PPA     | 4,000     | +$25       | $50,000   | 2 mo
  7  | Cold chain optimization   | 1,200     | +$60       | $350,000  | 9 mo

Total available: 17,000 tCO2e, $2,650,000
Gap: 9,600 tCO2e
```

### Optimized Selection

```
Selected (gap = 9,600, budget = $3,000,000):

1. Refrigerant management:     3,200 tCO2e,  $280,000   (cum: 3,200)
2. LED retrofit:               2,800 tCO2e,  $420,000   (cum: 6,000)
3. Building management:        1,500 tCO2e,  $200,000   (cum: 7,500)
4. Green electricity PPA:      4,000 tCO2e,   $50,000   (cum: 11,500)  GAP CLOSED

Total: 11,500 tCO2e, $950,000 capital
Surplus: 1,900 tCO2e (buffer for pessimistic scenario)
Budget used: 31.7% of available

Remaining budget ($2,050,000) allocated to:
5. Solar PV:                   2,500 tCO2e,  $750,000
6. EV delivery fleet:          1,800 tCO2e,  $600,000
These provide additional reduction and future-proofing.
```

---

## Worked Example: Financial Services

### Company Profile

| Attribute | Value |
|-----------|-------|
| Company | CleanFin Capital |
| Sector | Financial services (5 offices) |
| Base year (2021) | 8,500 tCO2e (Scope 1+2) |
| Target (2030) | 4,548 tCO2e (-46.5%) |
| 2024 actual | 7,800 tCO2e |
| 2025 target | 7,060 tCO2e |
| Current gap | 740 tCO2e (10.5% above target) |
| Available budget | $250,000 |

### Context

Financial services companies typically have smaller absolute emissions but face challenges with Scope 2 (office electricity) and Scope 1 (heating, fleet). The gap is modest in absolute terms but significant relative to total emissions.

### Selected Initiatives

```
1. Green electricity tariff:     500 tCO2e,  $8,000    (immediate)
2. Smart HVAC controls:          150 tCO2e,  $35,000   (3 months)
3. LED retrofit (HQ):            120 tCO2e,  $18,000   (2 months)

Total: 770 tCO2e, $61,000
Gap closed with 30 tCO2e surplus
Budget used: 24.4% of available
```

### Key Insight

For office-based companies, switching to a green electricity tariff is often the single most impactful and fastest initiative. In this case, it closes 67.6% of the gap at a capital cost of only $8,000.

---

## Integration with Other Workflows

### Corrective Action triggers from other workflows:

```
Quarterly Monitoring
    -> RED status
    -> Triggers Corrective Action Planning

Annual Progress Review
    -> Off-track assessment
    -> Triggers Corrective Action Planning

Variance Investigation
    -> LMDI shows structural issue
    -> Informs initiative selection

Target Recalibration
    -> Acquisition increases baseline
    -> May require corrective action if new baseline creates gap

Corrective Action Planning
    -> Produces action plan
    -> Feeds back into Quarterly Monitoring (expected improvement)
    -> Updates Annual Progress Review (planned trajectory)
    -> Informs CDP/TCFD disclosure (management response)
```

### Feedback Loop

```
Monitor -> Detect Gap -> Analyze Root Cause -> Plan Actions -> Implement -> Monitor
   ^                                                                         |
   |_________________________________________________________________________|
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Resolution |
|-------|-------|-----------|
| No initiatives close the gap | Insufficient abatement options | Expand initiative catalog; consider novel technologies |
| Budget insufficient | CapEx exceeds available funds | Phase implementation; prioritize negative-cost initiatives |
| Implementation delays | Resource constraints or supply chain | Adjust schedule; consider parallel implementation |
| Abatement lower than expected | Confidence level too optimistic | Use pessimistic scenario; add buffer initiatives |
| Initiatives have prerequisites | Dependency chains | Respect ordering in schedule; plan prerequisites first |
| Cost/tCO2e seems unreasonable | Incorrect cost or abatement data | Verify input data with operations/facilities teams |

### Data Quality for Corrective Action

```python
# Validate initiative data before optimization
for init in initiatives:
    assert init.abatement_potential > Decimal("0"), f"{init.name}: zero abatement"
    assert init.capital_cost >= Decimal("0"), f"{init.name}: negative capital cost"
    assert 0 < init.confidence_level <= 1, f"{init.name}: invalid confidence"
    assert init.implementation_time_months > 0, f"{init.name}: zero impl time"
    assert init.lifetime_years > 0, f"{init.name}: zero lifetime"
```

---

## FAQ

**Q: How often should corrective action plans be updated?**
A: Review and update quarterly when off-track. If the plan is working (RED moving to AMBER/GREEN), continue monitoring. If new gaps emerge, re-run the optimization.

**Q: Can I include Scope 3 initiatives?**
A: Yes. Set `scopes=["scope_1", "scope_2", "scope_3"]` when scanning initiatives. Scope 3 initiatives often include supplier engagement, logistics optimization, and product design changes. Note that Scope 3 abatement typically has lower confidence levels.

**Q: What if my gap is too large for available initiatives?**
A: PACK-029 will report a residual gap. Options include: (1) expanding the initiative catalog with novel technologies, (2) increasing budget allocation, (3) requesting target recalibration if a structural change justifies it, (4) accepting the gap and disclosing it transparently.

**Q: How does PACK-029 handle initiatives with variable costs?**
A: v1.0.0 uses fixed cost assumptions. Variable cost modeling is planned for v1.1.0. For now, use the confidence level to account for cost uncertainty (pessimistic scenario applies 1.3x cost multiplier).

**Q: Can I integrate external MACC data (e.g., McKinsey, IEA)?**
A: Yes. Import initiatives as `AbatementInitiative` objects with cost and abatement data from any source. PACK-029's MACC optimization is source-agnostic.

**Q: What if the gap closes itself due to business contraction?**
A: If emissions decrease due to reduced activity rather than efficiency improvements, PACK-029's LMDI decomposition will show this as an "activity effect" rather than an "intensity effect." While the gap may close, SBTi expects absolute reductions not dependent on business contraction. The corrective action plan should focus on intensity improvements.

**Q: How do I report corrective actions to CDP?**
A: PACK-029's CDP Export Template maps corrective action data to CDP C4.3 (emission reduction initiatives) and C4.4 (details of initiatives). The Annual Reporting Workflow includes corrective action data in the CDP export package.

---

**End of Corrective Action Planning Use Case Guide**
