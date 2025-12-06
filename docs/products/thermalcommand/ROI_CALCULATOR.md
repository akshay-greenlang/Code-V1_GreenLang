# ThermalCommand ROI Calculator

**Product Code:** GL-001 | **Version:** 2.0 | **Last Updated:** 2025-12-04

---

## Executive Summary

This document provides a comprehensive methodology for calculating the Return on Investment (ROI) for ThermalCommand implementations. The model incorporates conservative assumptions based on verified customer results across multiple industries.

---

## 1. ROI Framework Overview

### Value Drivers

ThermalCommand delivers value through five primary mechanisms:

| Value Driver | Typical Range | Conservative Estimate |
|--------------|---------------|----------------------|
| Energy Cost Reduction | 15-25% | 15% |
| Maintenance Cost Reduction | 15-25% | 15% |
| Downtime Reduction | 30-50% | 30% |
| Emissions Reduction | 10-20% | 10% |
| Productivity Improvement | 20-40% | 20% |

### ROI Formula

```
ROI = (Total Annual Benefits - Total Annual Costs) / Total Investment x 100%

Payback Period = Total Investment / (Total Annual Benefits - Total Annual Costs)
```

---

## 2. Cost Components

### 2.1 One-Time Costs

| Cost Category | Standard | Professional | Enterprise |
|---------------|----------|--------------|------------|
| Implementation Services | $45,000 | $140,000 | $445,000 |
| Hardware (Edge Gateways) | $10,000 | $25,000 | $75,000 |
| Network Infrastructure | $5,000 | $15,000 | $50,000 |
| Internal Project Resources | $20,000 | $50,000 | $150,000 |
| Training (Extended) | $5,000 | $15,000 | $40,000 |
| **Total One-Time** | **$85,000** | **$245,000** | **$760,000** |

### 2.2 Annual Recurring Costs

| Cost Category | Standard | Professional | Enterprise |
|---------------|----------|--------------|------------|
| Subscription (Base) | $60,000 | $180,000 | $540,000 |
| Per-Asset Fees (100 assets) | $60,000 | $90,000 | $120,000 |
| Support Premium | $0 | $0 | $60,000 |
| Internal Administration | $10,000 | $25,000 | $50,000 |
| **Total Annual** | **$130,000** | **$295,000** | **$770,000** |

---

## 3. Benefit Components

### 3.1 Energy Cost Reduction

**Calculation Methodology:**

```
Energy Savings = Annual Thermal Energy Cost x Reduction Percentage

Where:
- Annual Thermal Energy Cost = Annual Consumption (MMBtu) x Fuel Price ($/MMBtu)
- Reduction Percentage = 15% (conservative) to 25% (optimistic)
```

**Industry Benchmarks:**

| Industry | Avg. Energy Cost/Site | Conservative Savings (15%) | Optimistic Savings (25%) |
|----------|----------------------|---------------------------|--------------------------|
| Oil & Gas Refinery | $25,000,000 | $3,750,000 | $6,250,000 |
| Chemical Plant | $15,000,000 | $2,250,000 | $3,750,000 |
| Steel Mill | $20,000,000 | $3,000,000 | $5,000,000 |
| Food & Beverage | $5,000,000 | $750,000 | $1,250,000 |
| Pulp & Paper | $10,000,000 | $1,500,000 | $2,500,000 |
| Cement Plant | $12,000,000 | $1,800,000 | $3,000,000 |

### 3.2 Maintenance Cost Reduction

**Calculation Methodology:**

```
Maintenance Savings = Annual Maintenance Cost x Reduction Percentage

Where:
- Annual Maintenance Cost = Labor + Parts + Contractors + Lost Production
- Reduction Percentage = 15% (conservative) to 25% (optimistic)
```

**Typical Maintenance Cost Breakdown:**

| Component | % of Total | Reduction with ThermalCommand |
|-----------|------------|------------------------------|
| Preventive Maintenance | 30% | 10% reduction (optimized scheduling) |
| Corrective Maintenance | 40% | 25% reduction (predictive capability) |
| Emergency Repairs | 20% | 40% reduction (early warning) |
| External Contractors | 10% | 20% reduction (fewer emergencies) |

**Industry Benchmarks:**

| Industry | Avg. Maintenance Cost/Site | Conservative Savings (15%) | Optimistic Savings (25%) |
|----------|---------------------------|---------------------------|--------------------------|
| Oil & Gas Refinery | $8,000,000 | $1,200,000 | $2,000,000 |
| Chemical Plant | $5,000,000 | $750,000 | $1,250,000 |
| Steel Mill | $6,000,000 | $900,000 | $1,500,000 |
| Food & Beverage | $2,000,000 | $300,000 | $500,000 |
| Pulp & Paper | $4,000,000 | $600,000 | $1,000,000 |
| Cement Plant | $5,000,000 | $750,000 | $1,250,000 |

### 3.3 Downtime Reduction

**Calculation Methodology:**

```
Downtime Savings = Unplanned Downtime Hours x Hourly Cost x Reduction Percentage

Where:
- Unplanned Downtime Hours = Historical hours/year
- Hourly Cost = Lost Production + Lost Revenue + Restart Costs
- Reduction Percentage = 30% (conservative) to 50% (optimistic)
```

**Industry Benchmarks:**

| Industry | Avg. Downtime Cost/Year | Conservative Savings (30%) | Optimistic Savings (50%) |
|----------|------------------------|---------------------------|--------------------------|
| Oil & Gas Refinery | $5,000,000 | $1,500,000 | $2,500,000 |
| Chemical Plant | $3,000,000 | $900,000 | $1,500,000 |
| Steel Mill | $4,000,000 | $1,200,000 | $2,000,000 |
| Food & Beverage | $1,500,000 | $450,000 | $750,000 |
| Pulp & Paper | $2,500,000 | $750,000 | $1,250,000 |
| Cement Plant | $2,000,000 | $600,000 | $1,000,000 |

### 3.4 Emissions Cost Avoidance

**Calculation Methodology:**

```
Emissions Savings = CO2 Emissions (tons) x Carbon Price ($/ton) x Reduction Percentage

Plus: Avoided Penalties + Avoided Compliance Costs + Carbon Credit Revenue
```

**Carbon Price Assumptions:**

| Region | Current Price | 2025 Projection | 2030 Projection |
|--------|---------------|-----------------|-----------------|
| EU ETS | $85/ton | $100/ton | $150/ton |
| California | $35/ton | $50/ton | $80/ton |
| US Federal (Future) | N/A | $50/ton | $100/ton |
| Internal Carbon Price | $50/ton | $75/ton | $100/ton |

### 3.5 Productivity Improvement

**Calculation Methodology:**

```
Productivity Savings = Operator Hours x Hourly Cost x Efficiency Improvement

Where:
- Operator Hours = FTEs x 2,080 hours/year
- Hourly Cost = Fully loaded labor cost
- Efficiency Improvement = 20% (conservative) to 40% (optimistic)
```

**Productivity Improvements:**

| Activity | Manual Time | With ThermalCommand | Savings |
|----------|-------------|---------------------|---------|
| Daily monitoring rounds | 4 hours/day | 1 hour/day | 75% |
| Report generation | 8 hours/week | 1 hour/week | 87% |
| Root cause analysis | 20 hours/incident | 4 hours/incident | 80% |
| Optimization analysis | 40 hours/month | 8 hours/month | 80% |
| Compliance reporting | 40 hours/quarter | 4 hours/quarter | 90% |

---

## 4. ROI Calculator Template

### Input Variables

| Variable | Your Value | Default |
|----------|------------|---------|
| **Site Information** | | |
| Number of thermal assets | _____ | 100 |
| Annual thermal energy cost | $_____ | $10,000,000 |
| Annual maintenance cost | $_____ | $4,000,000 |
| Annual unplanned downtime cost | $_____ | $2,000,000 |
| Annual CO2 emissions (tons) | _____ | 50,000 |
| Carbon price ($/ton) | $_____ | $50 |
| Operator FTEs for thermal | _____ | 10 |
| Avg. operator cost (loaded) | $_____ | $100,000 |
| **ThermalCommand Selection** | | |
| Edition | _____ | Professional |
| Modules | _____ | None |
| Contract term (years) | _____ | 3 |

### Calculated Results

```
YEAR 1 COSTS
============
One-Time Costs:
  Implementation Services:         $__________
  Hardware:                        $__________
  Network:                         $__________
  Internal Resources:              $__________
  Training:                        $__________
  ----------------------------------------
  Total One-Time:                  $__________

Annual Recurring Costs:
  Subscription:                    $__________
  Per-Asset Fees:                  $__________
  Support:                         $__________
  Internal Admin:                  $__________
  ----------------------------------------
  Total Annual:                    $__________

YEAR 1 TOTAL COST:                 $__________


ANNUAL BENEFITS (Conservative)
==============================
Energy Savings (15%):              $__________
Maintenance Savings (15%):         $__________
Downtime Savings (30%):            $__________
Emissions Savings (10%):           $__________
Productivity Savings (20%):        $__________
----------------------------------------
TOTAL ANNUAL BENEFITS:             $__________


ROI CALCULATIONS
================
Year 1 Net Benefit:                $__________
Year 1 ROI:                        ___________%

3-Year Total Investment:           $__________
3-Year Total Benefits:             $__________
3-Year Net Benefit:                $__________
3-Year ROI:                        ___________%

Payback Period:                    __________ months
```

---

## 5. Sample ROI Calculations

### Example 1: Mid-Size Chemical Plant (Professional Edition)

**Inputs:**
- 150 thermal assets
- $12M annual thermal energy cost
- $4M annual maintenance cost
- $2M annual downtime cost
- 40,000 tons CO2/year at $50/ton
- 12 operator FTEs at $95,000 loaded

**Costs:**
- One-Time: $245,000
- Annual Recurring: $340,000 (base + 150 assets)
- Year 1 Total: $585,000
- 3-Year Total: $1,265,000

**Benefits (Conservative):**
- Energy Savings: $1,800,000 (15% x $12M)
- Maintenance Savings: $600,000 (15% x $4M)
- Downtime Savings: $600,000 (30% x $2M)
- Emissions Savings: $200,000 (10% x 40K tons x $50)
- Productivity: $228,000 (20% x 12 FTEs x $95K)
- **Total Annual: $3,428,000**

**ROI Results:**
- Year 1 Net Benefit: $2,843,000
- Year 1 ROI: 486%
- 3-Year ROI: 713%
- **Payback Period: 2.5 months**

---

### Example 2: Large Oil Refinery (Enterprise Edition)

**Inputs:**
- 800 thermal assets
- $30M annual thermal energy cost
- $10M annual maintenance cost
- $6M annual downtime cost
- 150,000 tons CO2/year at $85/ton (EU ETS)
- 25 operator FTEs at $110,000 loaded

**Costs:**
- One-Time: $760,000
- Annual Recurring: $1,070,000 (base + 800 assets + modules)
- Year 1 Total: $1,830,000
- 3-Year Total: $2,970,000

**Benefits (Conservative):**
- Energy Savings: $4,500,000 (15% x $30M)
- Maintenance Savings: $1,500,000 (15% x $10M)
- Downtime Savings: $1,800,000 (30% x $6M)
- Emissions Savings: $1,275,000 (10% x 150K tons x $85)
- Productivity: $550,000 (20% x 25 FTEs x $110K)
- **Total Annual: $9,625,000**

**ROI Results:**
- Year 1 Net Benefit: $7,795,000
- Year 1 ROI: 426%
- 3-Year ROI: 873%
- **Payback Period: 2.3 months**

---

### Example 3: Food & Beverage Plant (Standard Edition)

**Inputs:**
- 40 thermal assets
- $4M annual thermal energy cost
- $1.5M annual maintenance cost
- $800K annual downtime cost
- 15,000 tons CO2/year at $50/ton
- 5 operator FTEs at $85,000 loaded

**Costs:**
- One-Time: $85,000
- Annual Recurring: $84,000 (base + 40 assets)
- Year 1 Total: $169,000
- 3-Year Total: $337,000

**Benefits (Conservative):**
- Energy Savings: $600,000 (15% x $4M)
- Maintenance Savings: $225,000 (15% x $1.5M)
- Downtime Savings: $240,000 (30% x $800K)
- Emissions Savings: $75,000 (10% x 15K tons x $50)
- Productivity: $85,000 (20% x 5 FTEs x $85K)
- **Total Annual: $1,225,000**

**ROI Results:**
- Year 1 Net Benefit: $1,056,000
- Year 1 ROI: 625%
- 3-Year ROI: 991%
- **Payback Period: 1.7 months**

---

## 6. Sensitivity Analysis

### Impact of Key Variables

| Variable | -20% Change | Base Case | +20% Change |
|----------|-------------|-----------|-------------|
| Energy Savings % | 12% | 15% | 18% |
| Impact on ROI | -15% | Baseline | +15% |
| | | | |
| Fuel Price | $6/MMBtu | $8/MMBtu | $10/MMBtu |
| Impact on ROI | -18% | Baseline | +18% |
| | | | |
| Implementation Cost | -20% | Baseline | +20% |
| Impact on ROI | +5% | Baseline | -4% |
| | | | |
| Carbon Price | $40/ton | $50/ton | $60/ton |
| Impact on ROI | -2% | Baseline | +2% |

### Break-Even Analysis

**Minimum savings required to break even:**

| Edition | Annual Cost | Required Savings | As % of Avg. Site |
|---------|-------------|------------------|-------------------|
| Standard | $130,000 | $130,000 | 0.9% |
| Professional | $295,000 | $295,000 | 1.7% |
| Enterprise | $770,000 | $770,000 | 1.8% |

*Average site thermal spend: $15,000,000*

---

## 7. Risk-Adjusted ROI

### Monte Carlo Simulation Inputs

| Variable | Distribution | Min | Most Likely | Max |
|----------|--------------|-----|-------------|-----|
| Energy Savings | Triangular | 10% | 15% | 25% |
| Maintenance Savings | Triangular | 10% | 15% | 25% |
| Downtime Savings | Triangular | 20% | 30% | 50% |
| Implementation Cost | Triangular | -10% | 0% | +25% |
| Timeline Delay | Triangular | 0 mo | 1 mo | 3 mo |

### Probability Distribution (1,000 simulations)

| Percentile | Payback Period | 3-Year ROI |
|------------|----------------|------------|
| 10th (Conservative) | 4.2 months | 385% |
| 25th | 3.5 months | 485% |
| 50th (Median) | 2.8 months | 620% |
| 75th | 2.2 months | 780% |
| 90th (Optimistic) | 1.8 months | 950% |

**Probability of achieving:**
- ROI > 100%: 99.5%
- ROI > 300%: 95.2%
- ROI > 500%: 78.4%
- Payback < 6 months: 98.7%
- Payback < 12 months: 99.9%

---

## 8. TCO Comparison

### ThermalCommand vs. Status Quo (3-Year)

| Cost Category | Status Quo | ThermalCommand | Difference |
|---------------|------------|----------------|------------|
| Energy Costs | $36,000,000 | $30,600,000 | -$5,400,000 |
| Maintenance | $12,000,000 | $10,200,000 | -$1,800,000 |
| Downtime | $6,000,000 | $4,200,000 | -$1,800,000 |
| Carbon Costs | $7,500,000 | $6,750,000 | -$750,000 |
| Labor Costs | $3,420,000 | $2,736,000 | -$684,000 |
| ThermalCommand | $0 | $1,265,000 | +$1,265,000 |
| **Total** | **$64,920,000** | **$55,751,000** | **-$9,169,000** |

**3-Year TCO Savings: $9,169,000 (14.1%)**

---

## 9. Appendices

### Appendix A: Data Collection Worksheet

To generate an accurate ROI estimate, collect the following data:

**Energy Data:**
- [ ] Annual natural gas consumption (therms or MMBtu)
- [ ] Annual fuel oil consumption (gallons)
- [ ] Annual electricity for thermal (kWh)
- [ ] Current fuel prices
- [ ] Historical energy cost trends

**Maintenance Data:**
- [ ] Annual maintenance budget (thermal systems)
- [ ] Preventive maintenance costs
- [ ] Corrective maintenance costs
- [ ] Emergency repair costs
- [ ] Contractor costs

**Downtime Data:**
- [ ] Hours of unplanned downtime (last 12 months)
- [ ] Cost per hour of downtime
- [ ] Root causes of major incidents
- [ ] Production loss values

**Emissions Data:**
- [ ] Annual CO2 emissions (tons)
- [ ] Current carbon price or internal price
- [ ] Emission reduction targets
- [ ] Compliance requirements

**Labor Data:**
- [ ] Number of operators (thermal systems)
- [ ] Fully loaded labor cost
- [ ] Time spent on manual monitoring
- [ ] Time spent on reporting

### Appendix B: ROI Validation Process

GreenLang offers independent ROI validation through our Customer Success team:

1. **Baseline Establishment** (Month 1-3)
   - Install monitoring infrastructure
   - Collect baseline data (energy, maintenance, downtime)
   - Establish measurement methodology

2. **Performance Period** (Month 4-12)
   - Implement optimizations
   - Track savings continuously
   - Adjust for external factors (weather, production)

3. **ROI Validation** (Month 12)
   - Calculate verified savings
   - Compare to baseline
   - Document methodology
   - Third-party verification (optional)

---

**Contact Information:**

For a customized ROI analysis, contact:
- **Email:** roi@greenlang.com
- **Phone:** +1 (888) 555-GLNG
- **Web:** www.greenlang.com/roi-calculator

---

*Document Version: 2.0 | Last Updated: 2025-12-04*
