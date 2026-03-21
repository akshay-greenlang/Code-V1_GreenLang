# Calculation Guide: Interim Target Setting

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Engine:** Interim Target Engine

---

## Table of Contents

1. [Overview](#overview)
2. [SBTi Interim Target Methodology](#sbti-interim-target-methodology)
3. [5-Year Target Formula](#5-year-target-formula)
4. [10-Year Target Interpolation](#10-year-target-interpolation)
5. [Linear Pathway](#linear-pathway)
6. [Front-Loaded Pathway](#front-loaded-pathway)
7. [Back-Loaded Pathway](#back-loaded-pathway)
8. [Constant Rate Pathway](#constant-rate-pathway)
9. [Milestone-Based Pathway](#milestone-based-pathway)
10. [Scope-Specific Timelines](#scope-specific-timelines)
11. [Temperature Score Calculation](#temperature-score-calculation)
12. [Cumulative Carbon Budget](#cumulative-carbon-budget)
13. [Worked Examples](#worked-examples)

---

## Overview

Interim targets decompose a long-term net-zero commitment into actionable short- and medium-term milestones. PACK-029 generates interim targets at 5-year and 10-year intervals using SBTi-validated methodologies.

### Key Equations

| Quantity | Formula |
|----------|---------|
| Target at year t | `target(t) = baseline * (1 - reduction_pct(t) / 100)` |
| Annual rate (compound) | `rate = 1 - (1 - total_reduction/100)^(1/years)` |
| Temperature score | `temp = 1.5 + max(0, (4.2 - rate) / 4.2) * 2.0` |
| Cumulative budget | `budget = SUM[ (E_prev + E_curr) / 2 * delta_years ]` |

---

## SBTi Interim Target Methodology

### Source: SBTi Corporate Net-Zero Standard v1.2 (2024)

The SBTi defines minimum ambition levels for interim targets:

| Ambition Level | Annual Rate | Near-Term Reduction | Near-Term Year | Long-Term Reduction |
|---------------|-------------|--------------------|-----------------|--------------------|
| 1.5C Aligned | >= 4.2%/yr | >= 42% | By 2030 | >= 90% |
| Well-Below 2C | >= 2.5%/yr | >= 25% | By 2030 | >= 80% |
| 2C Aligned | >= 1.5%/yr | >= 15% | By 2035 | >= 72% |
| Race to Zero | >= 7.0%/yr | >= 50% | By 2030 | >= 90% |

### Scope Requirements

| Scope | Coverage | Lag Allowance |
|-------|----------|--------------|
| Scope 1+2 | >= 95% of total | None |
| Scope 3 | >= 67% of total (if >= 40% of overall) | Up to 5 years |
| FLAG | Separate target required | SBTi FLAG Guidance v1.1 |

---

## 5-Year Target Formula

### General Formula

Given a baseline and long-term target, the 5-year target is calculated by interpolating the reduction pathway at 5-year intervals from the base year.

```
For each milestone year y = base_year + 5, base_year + 10, ..., target_year:

    reduction_pct(y) = f(y, base_year, target_year, total_reduction_pct, shape)

    target_tco2e(y) = baseline_tco2e * (1 - reduction_pct(y) / 100)
```

Where `f()` depends on the selected pathway shape (see below).

### Example

```
Baseline: 200,000 tCO2e (2021)
Long-term target: 90% reduction by 2050

5-year milestones (linear pathway):
    2026: reduction = 90% * (5/29) = 15.5%  -> 169,000 tCO2e
    2031: reduction = 90% * (10/29) = 31.0% -> 138,000 tCO2e
    2036: reduction = 90% * (15/29) = 46.6% -> 106,900 tCO2e
    2041: reduction = 90% * (20/29) = 62.1% ->  75,900 tCO2e
    2046: reduction = 90% * (25/29) = 77.6% ->  44,800 tCO2e
    2050: reduction = 90% * (29/29) = 90.0% ->  20,000 tCO2e
```

---

## 10-Year Target Interpolation

Same methodology as 5-year targets but at 10-year intervals:

```
For each milestone year y = base_year + 10, base_year + 20, ..., target_year:

    reduction_pct(y) = f(y, base_year, target_year, total_reduction_pct, shape)

    target_tco2e(y) = baseline_tco2e * (1 - reduction_pct(y) / 100)
```

---

## Linear Pathway

### Formula

```
reduction_pct(t) = total_reduction_pct * (t - base_year) / (target_year - base_year)
```

### Properties

- Equal absolute reduction each year
- Simplest to communicate and track
- Recommended by SBTi for most organizations
- Annual reduction amount is constant: `delta = baseline * total_reduction / years`

### Example

```python
# Linear pathway: 200,000 tCO2e, 90% reduction over 29 years
baseline = 200_000
annual_reduction = 200_000 * 0.90 / 29  # = 6,207 tCO2e/year

# Year 2025: reduction = 90% * 4/29 = 12.4%
target_2025 = 200_000 * (1 - 0.124) = 175,172 tCO2e

# Year 2030: reduction = 90% * 9/29 = 27.9%
target_2030 = 200_000 * (1 - 0.279) = 144,138 tCO2e
```

---

## Front-Loaded Pathway

### Formula

```
reduction_pct(t) = total_reduction_pct * sqrt((t - base_year) / (target_year - base_year))
```

### Properties

- Faster reductions in early years (square root curve)
- Larger percentage of total reduction achieved by midpoint (~71%)
- Suitable when low-cost abatement options are available now
- Demonstrates early action commitment

### Example

```python
# Front-loaded: 200,000 tCO2e, 90% reduction over 29 years
# Year 2030 (9 years): progress = sqrt(9/29) = 0.557
target_2030 = 200_000 * (1 - 0.90 * 0.557) = 200_000 * 0.499 = 99,780 tCO2e
# Compare to linear 2030: 144,138 tCO2e -- front-loaded is much more aggressive early
```

---

## Back-Loaded Pathway

### Formula

```
reduction_pct(t) = total_reduction_pct * ((t - base_year) / (target_year - base_year))^2
```

### Properties

- Slower reductions in early years (quadratic curve)
- Only ~25% of total reduction achieved by midpoint
- **Warning**: May not meet SBTi near-term minimum requirements
- Only appropriate when technology transitions take time to scale

### Example

```python
# Back-loaded: 200,000 tCO2e, 90% reduction over 29 years
# Year 2030 (9 years): progress = (9/29)^2 = 0.096
target_2030 = 200_000 * (1 - 0.90 * 0.096) = 200_000 * 0.914 = 182,793 tCO2e
# Only 8.6% reduction by 2030 -- likely fails SBTi 1.5C minimum of 42%
```

---

## Constant Rate Pathway

### Formula

```
remaining_fraction = 1 - total_reduction_pct / 100
annual_rate = 1 - remaining_fraction^(1 / total_years)
reduction_pct(t) = (1 - (1 - annual_rate)^(t - base_year)) * 100
```

### Properties

- Same percentage reduction each year (compound/exponential decay)
- Larger absolute reductions early, smaller later
- Equivalent to constant CAGR (compound annual growth rate of reduction)
- Natural fit for organizations with steady improvement programs

### Example

```python
# Constant rate: 200,000 tCO2e, 90% reduction over 29 years
remaining = 0.10
annual_rate = 1 - 0.10^(1/29) = 1 - 0.923 = 0.077 = 7.7%/year

# Year 2025 (4 years): 200,000 * (1-0.077)^4 = 200,000 * 0.726 = 145,200 tCO2e
# Year 2030 (9 years): 200,000 * (1-0.077)^9 = 200,000 * 0.482 = 96,400 tCO2e
```

---

## Milestone-Based Pathway

### Methodology

Custom milestones define reduction percentages at specific years. Between milestones, linear interpolation is used.

```
Given milestones: [(y1, r1), (y2, r2), ..., (yn, rn)]

For year t between yi and y(i+1):
    reduction_pct(t) = ri + (r(i+1) - ri) * (t - yi) / (y(i+1) - yi)
```

### Example

```python
# Milestone-based with custom targets
milestones = [
    (2025, 15),   # 15% reduction by 2025
    (2030, 45),   # 45% reduction by 2030 (SBTi-aligned)
    (2035, 60),   # 60% reduction by 2035
    (2040, 75),   # 75% reduction by 2040
    (2050, 90),   # 90% reduction by 2050
]

# Year 2028 (between 2025 and 2030):
# reduction = 15 + (45-15) * (2028-2025) / (2030-2025) = 15 + 30 * 0.6 = 33%
target_2028 = 200_000 * (1 - 0.33) = 134,000 tCO2e
```

---

## Scope-Specific Timelines

### Scope 1+2 Timeline

Standard SBTi requirements apply directly:

```
Near-term target: 42% reduction by 2030 (1.5C)
Long-term target: 90% reduction by 2050
Coverage: >= 95% of total Scope 1+2
```

### Scope 3 Timeline (with Lag)

SBTi allows up to 5 years of lag for Scope 3:

```
Effective base year = base_year + lag_years
Effective period = target_year - effective_base_year

Example with 3-year lag:
    Base year: 2021
    Effective base: 2024
    Target year: 2050
    Effective period: 26 years (instead of 29)
    Annual rate: higher (to achieve same reduction in fewer years)
```

### FLAG Sector Timeline

Separate FLAG targets per SBTi FLAG Guidance v1.1:

```
FLAG near-term minimum: 30% reduction
FLAG near-term latest year: 2032 (base_year + 10, capped at 2032)
FLAG annual rate: >= 3.0%/yr
```

---

## Temperature Score Calculation

### Simplified SBTi-Aligned Formula

```
If annual_rate >= 4.2%:
    temperature = 1.5C

Else:
    gap = max(0, 4.2 - annual_rate)
    temperature = 1.5 + (gap / 4.2) * 2.0
    temperature = min(temperature, 4.0)
```

### Interpretation

| Annual Rate | Temperature Score | Alignment |
|------------|------------------|-----------|
| >= 7.0% | 1.5C | Race to Zero |
| >= 4.2% | 1.5C | SBTi 1.5C |
| 2.5% | 2.31C | WB2C |
| 1.5% | 2.79C | 2C |
| 0.0% | 3.50C | No reduction |

---

## Cumulative Carbon Budget

### Trapezoidal Integration

The cumulative carbon budget represents the total emissions allowed over the entire pathway period.

```
Budget = SUM over consecutive milestone pairs (yi, y(i+1)):
    segment_budget = (E(yi) + E(y(i+1))) / 2 * (y(i+1) - yi)

Total Budget = SUM of all segment budgets
```

### Example

```
Milestones: [(2021, 200000), (2025, 170000), (2030, 116000), (2050, 20000)]

Budget segments:
    2021-2025: (200000 + 170000) / 2 * 4 = 740,000 tCO2e
    2025-2030: (170000 + 116000) / 2 * 5 = 715,000 tCO2e
    2030-2050: (116000 + 20000) / 2 * 20 = 1,360,000 tCO2e

Total Budget = 740,000 + 715,000 + 1,360,000 = 2,815,000 tCO2e
```

---

## Worked Examples

### Example 1: Manufacturing Company (1.5C Aligned)

**Input:**
- Entity: SteelCorp International
- Base year: 2021
- Scope 1: 150,000 tCO2e
- Scope 2: 80,000 tCO2e
- Scope 3: 450,000 tCO2e
- Total: 680,000 tCO2e
- Long-term target: 90% reduction by 2050
- Ambition: 1.5C
- Pathway: Linear

**Calculation:**

```
Total years: 2050 - 2021 = 29

Annual rate (compound):
    remaining = 0.10
    rate = 1 - 0.10^(1/29) = 7.77%/yr

Temperature score: 1.5C (rate >= 4.2%)

5-year milestones (linear):
    2026: 680,000 * (1 - 0.90 * 5/29)  = 680,000 * 0.845 = 574,483 tCO2e
    2031: 680,000 * (1 - 0.90 * 10/29) = 680,000 * 0.690 = 468,966 tCO2e
    2036: 680,000 * (1 - 0.90 * 15/29) = 680,000 * 0.534 = 363,448 tCO2e
    2041: 680,000 * (1 - 0.90 * 20/29) = 680,000 * 0.379 = 257,931 tCO2e
    2046: 680,000 * (1 - 0.90 * 25/29) = 680,000 * 0.224 = 152,414 tCO2e
    2050: 680,000 * (1 - 0.90 * 29/29) = 680,000 * 0.100 =  68,000 tCO2e

SBTi validation:
    Near-term (2030): reduction = 90% * 9/29 = 27.9% -- BELOW 42% minimum
    --> SBTi check FAILS for linear pathway with 2021 base year
    --> Recommendation: Use front-loaded pathway or earlier base year
```

### Example 2: Services Company (Race to Zero)

**Input:**
- Entity: ConsultCo Global
- Base year: 2022
- Scope 1: 5,000 tCO2e
- Scope 2: 15,000 tCO2e
- Scope 3: 80,000 tCO2e
- Total: 100,000 tCO2e
- Long-term target: 90% reduction by 2050
- Ambition: Race to Zero
- Pathway: Front-loaded

**Calculation:**

```
Total years: 2050 - 2022 = 28

5-year milestones (front-loaded):
    2027: sqrt(5/28) = 0.423 -> 100,000 * (1 - 0.90 * 0.423) = 61,970 tCO2e
    2032: sqrt(10/28) = 0.598 -> 100,000 * (1 - 0.90 * 0.598) = 46,221 tCO2e
    2037: sqrt(15/28) = 0.732 -> 100,000 * (1 - 0.90 * 0.732) = 34,120 tCO2e
    2042: sqrt(20/28) = 0.845 -> 100,000 * (1 - 0.90 * 0.845) = 23,937 tCO2e
    2047: sqrt(25/28) = 0.945 -> 100,000 * (1 - 0.90 * 0.945) = 14,953 tCO2e
    2050: sqrt(28/28) = 1.000 -> 100,000 * (1 - 0.90 * 1.000) = 10,000 tCO2e

Near-term (2030): sqrt(8/28) = 0.535 -> reduction = 48.1%
    Race to Zero minimum: 50% -- CLOSE but slightly below
    --> Adjust to milestone-based with 50% at 2030
```

### Example 3: SME (Well-Below 2C, Scope 1+2 Only)

**Input:**
- Entity: LocalBakery Ltd
- Base year: 2023
- Scope 1: 500 tCO2e
- Scope 2: 300 tCO2e
- Scope 3: 0 (not measured)
- Total: 800 tCO2e
- Long-term target: 80% reduction by 2050
- Ambition: WB2C
- Pathway: Linear

**Calculation:**

```
Total years: 2050 - 2023 = 27

Annual rate (compound):
    remaining = 0.20
    rate = 1 - 0.20^(1/27) = 5.79%/yr

Temperature score: 1.5C (rate >= 4.2%)

5-year milestones (linear):
    2028: 800 * (1 - 0.80 * 5/27) = 800 * 0.852 = 681 tCO2e
    2033: 800 * (1 - 0.80 * 10/27) = 800 * 0.704 = 563 tCO2e
    2038: 800 * (1 - 0.80 * 15/27) = 800 * 0.556 = 444 tCO2e
    2043: 800 * (1 - 0.80 * 20/27) = 800 * 0.407 = 326 tCO2e
    2048: 800 * (1 - 0.80 * 25/27) = 800 * 0.259 = 207 tCO2e
    2050: 800 * (1 - 0.80 * 27/27) = 800 * 0.200 = 160 tCO2e

SBTi validation:
    Near-term (2030): reduction = 80% * 7/27 = 20.7% -- BELOW 25% WB2C minimum
    --> Need steeper near-term reduction or front-loaded pathway
```

---

**End of Interim Targets Calculation Guide**
