# Calculation Guide: Corrective Actions & MACC Optimization

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Engine:** Corrective Action Engine

---

## Table of Contents

1. [Overview](#overview)
2. [Gap Quantification](#gap-quantification)
3. [Marginal Abatement Cost Curve (MACC)](#marginal-abatement-cost-curve-macc)
4. [Initiative Portfolio Optimization](#initiative-portfolio-optimization)
5. [Catch-Up Timeline Calculation](#catch-up-timeline-calculation)
6. [Three-Scenario Analysis](#three-scenario-analysis)
7. [Investment Requirements](#investment-requirements)
8. [Cost-Effectiveness Ranking](#cost-effectiveness-ranking)
9. [Implementation Scheduling](#implementation-scheduling)
10. [Worked Examples](#worked-examples)

---

## Overview

When quarterly or annual monitoring reveals that emissions are above the interim target pathway, the Corrective Action Engine quantifies the gap and builds an optimized portfolio of abatement initiatives to close it. The engine uses Marginal Abatement Cost Curve (MACC) analysis to select the most cost-effective combination of initiatives under budget and timeline constraints.

### Key Principles

- Gap calculation uses the same Decimal arithmetic as all PACK-029 engines
- MACC optimization follows McKinsey/IEA methodology adapted for corporate use
- Initiatives are ranked by cost-effectiveness (USD per tCO2e abated)
- Negative-cost initiatives (those that save money) are prioritized first
- Three scenarios (optimistic, baseline, pessimistic) provide a range of outcomes
- All outputs include SHA-256 provenance hashing

---

## Gap Quantification

### Basic Gap Calculation

```
Gap = E_actual - E_target

Where:
    E_actual = Most recent actual emissions (tCO2e)
    E_target = Target emissions for the same period (tCO2e)
```

If `Gap > 0`: Emissions are above target (corrective action needed).
If `Gap <= 0`: Emissions are at or below target (no corrective action needed).

### Annualized Gap

For quarterly monitoring, the gap is annualized:

```
Gap_annual = Gap_quarterly * 4

Or more precisely, using the annualization from quarterly monitoring:

Gap_annual = E_actual_annualized - E_target_annual

Where:
    E_actual_annualized = E_actual_YTD * (4 / quarters_elapsed)
```

### Cumulative Gap (Carbon Budget Perspective)

The cumulative gap accounts for the total excess emissions over the target pathway:

```
Cumulative_Gap = SUM(E_actual_t - E_target_t, t = base_year..current_year)
```

This is the total amount of emissions that must be "repaid" through deeper cuts in future years.

### Forward Gap (Projected)

Using trend extrapolation (from the Trend Extrapolation Engine), the forward gap projects the total additional abatement needed:

```
Forward_Gap = SUM(E_projected_t - E_target_t, t = current_year..target_year)

Where:
    E_projected_t = Forecast emissions under current trajectory
    E_target_t = Target pathway emissions
```

### Gap Data Structure

```json
{
  "gap_analysis": {
    "current_year_gap": 15000,
    "annualized_gap": 15000,
    "cumulative_gap": 28000,
    "forward_gap": 95000,
    "gap_as_percentage_of_target": 8.5,
    "years_behind_pathway": 1.2,
    "required_annual_reduction_rate": 5.8,
    "original_annual_reduction_rate": 4.2,
    "acceleration_needed": 1.6
  }
}
```

### Years Behind Pathway

```
Years_behind = Gap / Annual_reduction_rate_original

Where:
    Annual_reduction_rate_original = (E_baseline - E_target_final) / (target_year - base_year)
```

This tells the organization how many years they have "fallen behind" on their pathway.

### Required Acceleration

```
Required_rate = (E_actual - E_target_final) / (target_year - current_year)
Acceleration = Required_rate - Original_rate
```

---

## Marginal Abatement Cost Curve (MACC)

### Concept

A MACC ranks abatement initiatives from lowest cost to highest cost (per tCO2e abated), creating a step-function chart where:
- X-axis = Cumulative abatement potential (tCO2e)
- Y-axis = Cost per tCO2e abated (USD/tCO2e)

Initiatives below the x-axis are "negative cost" -- they save money while reducing emissions.

### Initiative Data Model

```python
class AbatementInitiative(BaseModel):
    """A single abatement initiative."""
    initiative_id: str
    name: str
    description: str
    scope: ScopeType                          # Which scope this initiative addresses
    category: str                             # e.g., "energy_efficiency", "fuel_switching", "renewables"
    abatement_potential: Decimal               # tCO2e per year
    capital_cost: Decimal                      # Total upfront cost (USD)
    annual_operating_cost: Decimal             # Annual cost/savings (USD, negative = savings)
    lifetime_years: int                        # Expected lifetime of the initiative
    implementation_time_months: int            # Time to implement and achieve full abatement
    confidence_level: Decimal                  # 0.0 to 1.0 (probability of achieving stated abatement)
    co_benefits: list[str]                     # Non-emission benefits
    prerequisites: list[str]                   # Other initiative IDs that must be completed first
    scope_coverage: list[str]                  # Scopes addressed by this initiative
```

### Cost-Effectiveness Calculation

The cost per tCO2e is calculated using the annualized cost method:

```
Annualized_cost = (Capital_cost * CRF) + Annual_operating_cost

Where:
    CRF = r * (1 + r)^n / ((1 + r)^n - 1)    [Capital Recovery Factor]
    r = discount rate (default: 5%)
    n = lifetime in years

Cost_per_tCO2e = Annualized_cost / Abatement_potential
```

### Negative-Cost Initiatives

Some initiatives generate net savings:

```
If Annual_operating_cost < 0 and |Annual_operating_cost| > Capital_cost * CRF:
    Then Cost_per_tCO2e < 0  (net savings per tonne abated)
```

Common negative-cost initiatives:
- LED lighting upgrades (reduced electricity cost)
- HVAC optimization (reduced energy consumption)
- Compressed air leak repair (reduced energy waste)
- Insulation improvements (reduced heating/cooling cost)

### MACC Construction

```
Step 1: Calculate Cost_per_tCO2e for each initiative
Step 2: Sort initiatives by Cost_per_tCO2e (ascending)
Step 3: Compute cumulative abatement (running total of abatement_potential)
Step 4: Create step function chart data

MACC_data = [
    {
        "initiative": "LED lighting",
        "cost_per_tco2e": -45,
        "abatement_potential": 500,
        "cumulative_abatement": 500
    },
    {
        "initiative": "HVAC optimization",
        "cost_per_tco2e": -20,
        "abatement_potential": 800,
        "cumulative_abatement": 1300
    },
    {
        "initiative": "Solar PV installation",
        "cost_per_tco2e": 15,
        "abatement_potential": 2000,
        "cumulative_abatement": 3300
    },
    ...
]
```

### MACC Chart Data Structure

```json
{
  "macc_chart": {
    "initiatives": [
      {
        "name": "LED Lighting",
        "x_start": 0,
        "x_end": 500,
        "y": -45,
        "color": "green"
      },
      {
        "name": "HVAC Optimization",
        "x_start": 500,
        "x_end": 1300,
        "y": -20,
        "color": "green"
      },
      {
        "name": "Solar PV",
        "x_start": 1300,
        "x_end": 3300,
        "y": 15,
        "color": "orange"
      }
    ],
    "gap_line": {
      "x": 15000,
      "label": "Target gap (15,000 tCO2e)"
    }
  }
}
```

---

## Initiative Portfolio Optimization

### Optimization Problem

```
Maximize:  Total abatement = SUM( x_i * A_i )

Subject to:
    SUM( x_i * C_i ) <= Budget                     [budget constraint]
    Implementation within timeline                   [timeline constraint]
    x_i in {0, 1}                                    [binary: include or exclude]
    Prerequisite ordering respected                  [dependency constraint]

Where:
    x_i = Binary decision variable (1 = selected, 0 = not selected)
    A_i = Abatement potential of initiative i (tCO2e)
    C_i = Capital cost of initiative i (USD)
```

### Greedy Algorithm (Default)

PACK-029 uses a greedy knapsack approach for efficiency:

```
Algorithm: MACC Greedy Optimization

Input:
    initiatives: list of AbatementInitiative
    gap: target abatement needed (tCO2e)
    budget: maximum capital expenditure (USD)

Step 1: Sort initiatives by Cost_per_tCO2e (ascending)

Step 2: Initialize
    selected = []
    remaining_gap = gap
    remaining_budget = budget
    cumulative_abatement = 0

Step 3: Iterate through sorted initiatives
    For each initiative i:
        If C_i <= remaining_budget AND prerequisites met:
            selected.append(i)
            remaining_gap -= A_i
            remaining_budget -= C_i
            cumulative_abatement += A_i
        If remaining_gap <= 0:
            break  # Gap closed

Step 4: Return selected portfolio
    total_cost = SUM(C_i for i in selected)
    total_abatement = SUM(A_i for i in selected)
    gap_closed = total_abatement >= gap
```

### Budget-Constrained Optimization

When the budget is insufficient to close the entire gap:

```
If total_available_abatement < gap:
    result.gap_fully_closed = False
    result.residual_gap = gap - total_available_abatement
    result.recommendations.append(
        f"Available initiatives can close {total_available_abatement:,.0f} tCO2e "
        f"of the {gap:,.0f} tCO2e gap. Additional measures or budget needed."
    )
```

### Timeline-Constrained Optimization

Initiatives that cannot be implemented before the next milestone are deprioritized:

```
For each initiative i:
    If implementation_time_months > months_to_milestone:
        i.timeline_eligible = False
        # Include in portfolio only if shorter-term options are insufficient
```

### Portfolio Metrics

```python
class PortfolioResult(BaseModel):
    selected_initiatives: list[AbatementInitiative]
    total_abatement: Decimal          # tCO2e
    total_capital_cost: Decimal       # USD
    total_annual_cost: Decimal        # USD/year (can be negative = savings)
    net_present_value: Decimal        # NPV of the portfolio (USD)
    average_cost_per_tco2e: Decimal   # USD/tCO2e
    gap_closed: bool                  # True if total_abatement >= gap
    residual_gap: Decimal             # tCO2e remaining (0 if closed)
    payback_period_years: Decimal     # Weighted average payback
    roi_percentage: Decimal           # Return on investment
```

---

## Catch-Up Timeline Calculation

### Concept

The catch-up timeline determines how many years it will take to get back on the target pathway, given the selected corrective actions.

### Calculation

```
E_corrected(t) = E_projected(t) - Abatement(t)

Where:
    Abatement(t) = SUM of abatement from all initiatives active at time t
    (accounting for implementation delays and ramp-up periods)

Catch_up_year = min(t) such that E_corrected(t) <= E_target(t)
```

### Ramp-Up Modeling

Most initiatives do not deliver full abatement immediately:

```
Abatement_i(t) = A_i * ramp_factor(t - t_start_i)

Where:
    ramp_factor(months) =
        0                        if months < 0 (not yet started)
        months / impl_time       if 0 <= months < impl_time (ramping up)
        1                        if months >= impl_time (full abatement)
```

### Catch-Up Timeline Data

```json
{
  "catch_up_timeline": {
    "current_year": 2024,
    "catch_up_year": 2026,
    "years_to_catch_up": 2,
    "annual_trajectory": [
      {"year": 2024, "projected": 185000, "target": 176000, "corrected": 182000, "gap": 6000},
      {"year": 2025, "projected": 180000, "target": 168000, "corrected": 170000, "gap": 2000},
      {"year": 2026, "projected": 175000, "target": 160000, "corrected": 158000, "gap": -2000}
    ]
  }
}
```

---

## Three-Scenario Analysis

### Scenario Definitions

| Scenario | Abatement Factor | Cost Factor | Description |
|----------|-------------------|-------------|-------------|
| Optimistic | confidence_level * 1.1 | capital_cost * 0.9 | Best-case: higher abatement, lower cost |
| Baseline | confidence_level * 1.0 | capital_cost * 1.0 | Expected case: as planned |
| Pessimistic | confidence_level * 0.7 | capital_cost * 1.3 | Worst-case: lower abatement, higher cost |

### Scenario-Adjusted Abatement

```
Abatement_optimistic = A_i * confidence_level * 1.1    (capped at A_i)
Abatement_baseline   = A_i * confidence_level
Abatement_pessimistic = A_i * confidence_level * 0.7
```

### Scenario Output

```json
{
  "scenarios": {
    "optimistic": {
      "total_abatement": 18500,
      "total_cost": 2700000,
      "gap_closed": true,
      "catch_up_year": 2025
    },
    "baseline": {
      "total_abatement": 15000,
      "total_cost": 3000000,
      "gap_closed": true,
      "catch_up_year": 2026
    },
    "pessimistic": {
      "total_abatement": 10500,
      "total_cost": 3900000,
      "gap_closed": false,
      "residual_gap": 4500,
      "catch_up_year": null
    }
  }
}
```

### Scenario Narratives

```python
narratives = {
    "optimistic": (
        f"Best case: {optimistic.total_abatement:,.0f} tCO2e abated at "
        f"${optimistic.total_cost:,.0f}. Gap fully closed by {optimistic.catch_up_year}."
    ),
    "baseline": (
        f"Expected case: {baseline.total_abatement:,.0f} tCO2e abated at "
        f"${baseline.total_cost:,.0f}. Gap {'fully closed' if baseline.gap_closed else 'partially closed'} "
        f"by {baseline.catch_up_year or 'beyond planning horizon'}."
    ),
    "pessimistic": (
        f"Worst case: {pessimistic.total_abatement:,.0f} tCO2e abated at "
        f"${pessimistic.total_cost:,.0f}. "
        f"{'Gap closed' if pessimistic.gap_closed else f'Residual gap of {pessimistic.residual_gap:,.0f} tCO2e remains'}."
    ),
}
```

---

## Investment Requirements

### Net Present Value (NPV)

```
NPV = -Capital_cost + SUM( Annual_savings_t / (1 + r)^t, t = 1..n )

Where:
    Annual_savings_t = -Annual_operating_cost  (positive if cost reduction)
    r = discount rate (default: 5%)
    n = initiative lifetime (years)
```

### Internal Rate of Return (IRR)

The IRR is the discount rate at which NPV = 0:

```
0 = -Capital_cost + SUM( Annual_savings_t / (1 + IRR)^t, t = 1..n )

Solved numerically using Newton-Raphson method.
```

### Simple Payback Period

```
Payback = Capital_cost / Annual_savings

Where Annual_savings = -Annual_operating_cost (must be positive for payback to exist)
```

### Portfolio-Level Investment Summary

```json
{
  "investment_summary": {
    "total_capital_required": 3000000,
    "total_annual_operating_impact": -450000,
    "net_present_value": 1250000,
    "weighted_average_irr": 12.5,
    "weighted_average_payback_years": 4.2,
    "negative_cost_initiatives": {
      "count": 3,
      "total_abatement": 2500,
      "total_savings_year1": 180000
    },
    "positive_cost_initiatives": {
      "count": 5,
      "total_abatement": 12500,
      "total_capital": 2800000
    },
    "funding_recommendation": "Phase 1: Implement 3 negative-cost initiatives immediately (self-funding). Phase 2: Use savings to partially fund positive-cost initiatives."
  }
}
```

---

## Cost-Effectiveness Ranking

### Ranking Criteria

Primary: Cost per tCO2e (ascending)
Secondary: Implementation time (ascending)
Tertiary: Confidence level (descending)

### Ranking Table Format

```
Rank | Initiative          | Abatement  | Cost/tCO2e | CapEx     | Impl Time | Confidence
-----|---------------------|-----------|------------|-----------|-----------|----------
  1  | LED Lighting        |    500    |    -$45    |  $22,500  |  3 months |    95%
  2  | HVAC Optimization   |    800    |    -$20    |  $48,000  |  4 months |    90%
  3  | Compressed Air Fix  |    300    |    -$15    |  $12,000  |  1 month  |    98%
  4  | Solar PV (rooftop)  |  2,000    |    +$15    | $300,000  | 12 months |    85%
  5  | Heat Pump Install   |  1,500    |    +$35    | $250,000  |  9 months |    80%
  6  | EV Fleet Transition |  3,000    |    +$55    | $800,000  | 18 months |    75%
  7  | Green Hydrogen      |  5,000    |   +$120    |$2,000,000 | 24 months |    60%
  8  | Carbon Capture      |  4,000    |   +$180    |$3,500,000 | 36 months |    50%
```

### Category Breakdown

```json
{
  "by_category": {
    "energy_efficiency": {
      "count": 3,
      "total_abatement": 1600,
      "average_cost_per_tco2e": -26.7
    },
    "renewables": {
      "count": 1,
      "total_abatement": 2000,
      "average_cost_per_tco2e": 15.0
    },
    "electrification": {
      "count": 2,
      "total_abatement": 4500,
      "average_cost_per_tco2e": 47.5
    },
    "novel_technology": {
      "count": 2,
      "total_abatement": 9000,
      "average_cost_per_tco2e": 146.7
    }
  }
}
```

---

## Implementation Scheduling

### Gantt-Style Schedule

PACK-029 generates a phased implementation schedule that respects:
1. Prerequisites (dependency ordering)
2. Budget phasing (annual capital allocation)
3. Implementation capacity (parallel initiative limit)
4. Quick wins first (negative-cost initiatives in Phase 1)

### Phase Allocation

```
Phase 1 (Months 1-6):   Negative-cost initiatives + quick wins
Phase 2 (Months 4-12):  Medium-cost, medium-impact initiatives
Phase 3 (Months 10-24): High-cost, high-impact initiatives
Phase 4 (Months 18-36): Transformational initiatives

Note: Phases may overlap for parallel implementation.
```

### Schedule Data Structure

```json
{
  "implementation_schedule": {
    "phases": [
      {
        "phase": 1,
        "name": "Quick Wins",
        "start_month": 1,
        "end_month": 6,
        "initiatives": [
          {
            "name": "LED Lighting",
            "start_month": 1,
            "end_month": 3,
            "capital_cost": 22500,
            "abatement_by_year_end": 500
          },
          {
            "name": "HVAC Optimization",
            "start_month": 1,
            "end_month": 4,
            "capital_cost": 48000,
            "abatement_by_year_end": 800
          }
        ],
        "phase_capital": 82500,
        "phase_abatement": 1600
      },
      {
        "phase": 2,
        "name": "Core Decarbonization",
        "start_month": 4,
        "end_month": 15,
        "initiatives": [
          {
            "name": "Solar PV Installation",
            "start_month": 4,
            "end_month": 15,
            "capital_cost": 300000,
            "abatement_by_year_end": 1500
          }
        ],
        "phase_capital": 300000,
        "phase_abatement": 1500
      }
    ],
    "total_months": 36,
    "total_capital": 3000000,
    "total_abatement_at_completion": 15000,
    "cumulative_abatement_schedule": [
      {"month": 3, "cumulative_abatement": 500},
      {"month": 6, "cumulative_abatement": 1600},
      {"month": 12, "cumulative_abatement": 5100},
      {"month": 24, "cumulative_abatement": 12000},
      {"month": 36, "cumulative_abatement": 15000}
    ]
  }
}
```

---

## Worked Examples

### Example 1: Manufacturing Company (Gap Closure)

**Situation:**
- Company: Mid-size manufacturer
- 2024 actual emissions: 185,000 tCO2e
- 2024 target emissions: 170,000 tCO2e
- Gap: 15,000 tCO2e
- Available budget: $2,000,000
- Deadline: Close gap by end of 2026

**Available Initiatives:**

| Initiative | Abatement (tCO2e) | Capital ($) | Annual Cost ($) | Impl. Time | Confidence |
|------------|-------------------|-------------|-----------------|------------|------------|
| LED upgrade (all facilities) | 1,200 | 180,000 | -54,000 | 4 months | 95% |
| HVAC modernization | 2,500 | 350,000 | -80,000 | 6 months | 90% |
| Compressed air repair | 600 | 30,000 | -18,000 | 2 months | 98% |
| Solar PV (2 MW rooftop) | 3,500 | 800,000 | -25,000 | 12 months | 85% |
| Process heat recovery | 4,000 | 500,000 | -30,000 | 8 months | 80% |
| EV forklift fleet | 1,800 | 450,000 | +20,000 | 6 months | 90% |
| Biogas boiler conversion | 5,000 | 1,200,000 | +50,000 | 18 months | 70% |
| Green electricity PPA | 8,000 | 50,000 | +400,000 | 3 months | 95% |

**Step 1: Cost-Effectiveness Ranking**

```
Initiative            | Cost/tCO2e | Cumulative Abatement
Compressed air repair |    -$52    |     600
LED upgrade           |    -$35    |   1,800
HVAC modernization    |    -$24    |   4,300
Process heat recovery |     -$4    |   8,300
Solar PV              |    +$20    |  11,800
EV forklift fleet     |    +$40    |  13,600
Green electricity PPA |    +$51    |  21,600
Biogas boiler         |    +$62    |  26,600
```

**Step 2: Greedy Selection (Budget = $2,000,000)**

```
1. Compressed air repair:    600 tCO2e,   $30,000  (budget remaining: $1,970,000)
2. LED upgrade:            1,200 tCO2e,  $180,000  (budget remaining: $1,790,000)
3. HVAC modernization:    2,500 tCO2e,  $350,000  (budget remaining: $1,440,000)
4. Process heat recovery:  4,000 tCO2e,  $500,000  (budget remaining: $940,000)
5. Solar PV:              3,500 tCO2e,  $800,000  (budget remaining: $140,000)
6. Green electricity PPA: 8,000 tCO2e,   $50,000  (budget remaining: $90,000)

Cumulative abatement: 19,800 tCO2e  (gap of 15,000 fully closed)
Total capital: $1,910,000 (within $2,000,000 budget)
Annual operating savings: -$207,000 + $400,000 = +$193,000/year
```

**Step 3: Three-Scenario Analysis**

```
Scenario     | Abatement | Cost      | Gap Closed? | Catch-Up Year
Optimistic   | 21,340    | $1,719,000 | Yes        | 2025
Baseline     | 17,820    | $1,910,000 | Yes        | 2026
Pessimistic  | 12,474    | $2,483,000 | Partial    | 2027 (with additional measures)
```

**Step 4: Implementation Schedule**

```
Phase 1 (Q1 2025): Compressed air + LED + HVAC start
    Abatement by Q2 2025: ~2,800 tCO2e annualized
    Capital: $560,000

Phase 2 (Q2 2025): Process heat recovery + Solar PV + PPA start
    Abatement by Q4 2025: ~8,500 tCO2e annualized
    Capital: $1,350,000

Phase 3 (Q1 2026): Solar PV completes, all initiatives at full capacity
    Abatement achieved: ~19,800 tCO2e
    Capital: $1,910,000

Catch-up timeline: Back on target pathway by mid-2026
```

---

### Example 2: Office-Based Services Company (Budget-Constrained)

**Situation:**
- Company: Professional services firm
- 2024 actual emissions: 12,000 tCO2e (Scope 1+2)
- 2024 target emissions: 10,500 tCO2e
- Gap: 1,500 tCO2e
- Available budget: $200,000
- Deadline: Close gap by end of 2025

**Available Initiatives:**

| Initiative | Abatement (tCO2e) | Capital ($) | Annual Cost ($) | Impl. Time | Confidence |
|------------|-------------------|-------------|-----------------|------------|------------|
| Smart building controls | 400 | 45,000 | -15,000 | 3 months | 92% |
| LED retrofit (HQ) | 200 | 25,000 | -8,000 | 2 months | 96% |
| Green electricity tariff | 600 | 5,000 | +30,000 | 1 month | 98% |
| Heat pump (HQ HVAC) | 350 | 120,000 | -10,000 | 6 months | 85% |
| EV pool cars | 150 | 180,000 | +5,000 | 3 months | 90% |
| Remote work policy | 200 | 10,000 | -5,000 | 1 month | 75% |

**MACC Ranking:**

```
Initiative            | Cost/tCO2e | Cumulative
LED retrofit          |    -$40    |    200
Smart building        |    -$28    |    600
Remote work policy    |    -$40    |    800
Heat pump             |    -$17    |  1,150
Green electricity     |    +$52    |  1,750
EV pool cars          |    +$78    |  1,900
```

**Greedy Selection (Budget = $200,000):**

```
1. LED retrofit:          200 tCO2e,  $25,000  (remaining: $175,000)
2. Remote work policy:    200 tCO2e,  $10,000  (remaining: $165,000)
3. Smart building:        400 tCO2e,  $45,000  (remaining: $120,000)
4. Heat pump:             350 tCO2e, $120,000  (remaining: $0)

Total: 1,150 tCO2e, $200,000
Gap remaining: 350 tCO2e (gap not fully closed within budget)
```

**Budget Augmentation Option:**

```
To close remaining 350 tCO2e gap:
    Add Green electricity tariff: 600 tCO2e, $5,000 capital + $30,000/year
    New total: 1,750 tCO2e (gap fully closed with surplus)
    Additional budget needed: $5,000 capital + $30,000 annual

Recommendation: Secure additional $35,000 for green electricity tariff
to fully close the gap. This is the most cost-effective remaining option.
```

**Result Summary:**

```json
{
  "baseline_scenario": {
    "total_abatement": 1150,
    "total_capital": 200000,
    "net_annual_savings": 38000,
    "gap_closed": false,
    "residual_gap": 350,
    "payback_period_years": 2.6,
    "recommendation": "Close 77% of gap within budget. Add green electricity tariff ($35K) to fully close."
  },
  "with_green_tariff": {
    "total_abatement": 1750,
    "total_capital": 205000,
    "net_annual_savings": 8000,
    "gap_closed": true,
    "surplus_abatement": 250,
    "payback_period_years": 3.1
  }
}
```

---

## Appendix: Key Formulas Summary

| Formula | Description |
|---------|-------------|
| Gap = E_actual - E_target | Basic gap quantification |
| CRF = r(1+r)^n / ((1+r)^n - 1) | Capital Recovery Factor |
| Cost/tCO2e = Annualized_cost / Abatement | Cost-effectiveness |
| NPV = -CapEx + SUM(Savings / (1+r)^t) | Net Present Value |
| Payback = CapEx / Annual_savings | Simple payback period |
| Ramp(t) = min(1, t / impl_time) | Implementation ramp-up |
| Abatement_adjusted = A * confidence | Confidence-adjusted abatement |

---

**End of Corrective Actions Calculation Guide**
