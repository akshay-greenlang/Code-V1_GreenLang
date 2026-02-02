# GL-077: Incentive Hunter Agent (INCENTIVEHUNTER)

## Overview

The IncentiveHunterAgent identifies and evaluates energy efficiency incentives, rebates, and tax credits available for projects. It provides comprehensive analysis of utility rebate programs, federal and state tax incentives, and grant opportunities.

## Features

- **Incentive Database**: Built-in database of common incentives (179D, ITC, SGIP, utility rebates)
- **Eligibility Assessment**: Automated eligibility evaluation with confidence scoring
- **Value Calculation**: Deterministic value estimation using program-specific formulas
- **Stacking Analysis**: Identifies stackable incentives and calculates combined value
- **Application Timeline**: Generates recommended application sequence
- **Provenance Tracking**: Complete SHA-256 audit trail for compliance

## Standards Compliance

- DSIRE (Database of State Incentives for Renewables & Efficiency)
- IRS Section 179D Energy Efficient Commercial Building Deduction
- IRA 2022 (Inflation Reduction Act)
- California SGIP (Self-Generation Incentive Program)
- Utility-specific rebate programs

## Installation

```python
from backend.agents.gl_077_incentive_hunter import (
    IncentiveHunterAgent,
    IncentiveHunterInput,
    LocationInfo,
    EquipmentInfo,
    ProjectScope,
    UtilityProvider,
    ProjectType,
)
```

## Quick Start

```python
from backend.agents.gl_077_incentive_hunter import (
    IncentiveHunterAgent,
    IncentiveHunterInput,
    LocationInfo,
    ProjectScope,
    ProjectType,
)

# Initialize agent
agent = IncentiveHunterAgent()

# Prepare input data
input_data = IncentiveHunterInput(
    location=LocationInfo(
        state="CA",
        zip_code="94102",
        utility_territory="PG&E",
    ),
    equipment_types=["LED_LIGHTING", "HVAC", "VFD"],
    project_scope=ProjectScope(
        project_type=ProjectType.RETROFIT,
        project_cost_usd=150000,
        building_size_sqft=50000,
        sector="COMMERCIAL",
    ),
)

# Run analysis
result = agent.run(input_data)

# Access results
print(f"Analysis ID: {result.analysis_id}")
print(f"Total Estimated Value: ${result.total_estimated_value_usd:,.2f}")
print(f"Eligible Incentives: {result.eligible_count}")

# Review available incentives
for incentive in result.available_incentives:
    print(f"\n{incentive.name}")
    print(f"  Type: {incentive.incentive_type.value}")
    print(f"  Value: ${incentive.estimated_value_usd:,.2f}")
    print(f"  Eligibility: {incentive.eligibility.state.value}")
    print(f"  Confidence: {incentive.eligibility.confidence_score:.0%}")

# Review recommendations
print("\nTop Recommendations:")
for rec in result.top_recommendations:
    print(f"  {rec}")
```

## Input Schema

### LocationInfo
| Field | Type | Description |
|-------|------|-------------|
| state | str | US state code (2 letters) |
| zip_code | str | ZIP code (optional) |
| utility_territory | str | Utility service territory (optional) |
| is_disadvantaged_community | bool | Located in DAC for bonus incentives |

### EquipmentInfo
| Field | Type | Description |
|-------|------|-------------|
| equipment_type | str | Type of equipment |
| manufacturer | str | Equipment manufacturer (optional) |
| quantity | int | Number of units |
| capacity_kw | float | Equipment capacity in kW |
| energy_star_certified | bool | ENERGY STAR certification status |

### ProjectScope
| Field | Type | Description |
|-------|------|-------------|
| project_type | ProjectType | NEW_CONSTRUCTION, RETROFIT, etc. |
| project_cost_usd | float | Total project cost |
| building_size_sqft | float | Building size in square feet |
| sector | str | COMMERCIAL, INDUSTRIAL, etc. |

### UtilityProvider
| Field | Type | Description |
|-------|------|-------------|
| electric_utility | str | Electric utility name |
| gas_utility | str | Gas utility name |
| annual_electric_usage_kwh | float | Annual electric consumption |
| peak_demand_kw | float | Peak demand |

## Output Schema

### IncentiveHunterOutput
| Field | Type | Description |
|-------|------|-------------|
| analysis_id | str | Unique analysis identifier |
| available_incentives | List[AvailableIncentive] | All identified incentives |
| total_estimated_value_usd | float | Sum of all eligible incentive values |
| eligible_count | int | Number of eligible incentives |
| conditional_count | int | Number of conditional incentives |
| top_recommendations | List[str] | Prioritized recommendations |
| application_timeline | List[Dict] | Recommended application sequence |
| provenance_hash | str | SHA-256 audit trail hash |

### AvailableIncentive
| Field | Type | Description |
|-------|------|-------------|
| incentive_id | str | Unique identifier |
| name | str | Program name |
| incentive_type | IncentiveType | UTILITY_REBATE, FEDERAL_TAX_CREDIT, etc. |
| estimated_value_usd | float | Estimated incentive value |
| value_basis | str | Calculation basis ($/sqft, % of cost, etc.) |
| eligibility | EligibilityStatus | Detailed eligibility assessment |

## Calculation Methods

### Section 179D Deduction
```
Base Rate: $0.50 - $1.00/sqft (without prevailing wage)
Enhanced Rate: $2.50 - $5.00/sqft (with prevailing wage)
Value = Building_SQFT * Rate
```

### Investment Tax Credit (ITC)
```
Base Rate: 6% (no prevailing wage) or 30% (with prevailing wage)
Domestic Content Bonus: +10%
Energy Community Bonus: +10%
Maximum: 50%
Value = Project_Cost * Total_Rate
```

### SGIP (California)
```
Standard Rate: $200/kWh
Equity Rate (DAC): $400/kWh (non-residential)
Value = Capacity_kWh * Rate
```

### Utility Rebates
```
LED: $50/fixture or $0.08/kWh saved
HVAC: $100/ton
VFD: $80/HP or $0.08/kWh saved
```

## Supported Equipment Types

| Equipment Type | Category | Typical Incentives |
|----------------|----------|-------------------|
| LED_LIGHTING | Lighting | Utility rebates, 179D |
| HVAC | HVAC | Utility rebates, 179D |
| VFD | Motors/Drives | Utility rebates |
| SOLAR | Renewable | ITC, state credits |
| BATTERY | Energy Storage | ITC, SGIP |
| CHILLER | HVAC | Utility rebates |
| BOILER | Process Heat | Custom rebates |
| EMS | Controls | Custom rebates |

## Zero-Hallucination Guarantee

All calculations use deterministic formulas from documented sources:
- No LLM inference in value calculations
- Program-specific rates from official documentation
- Complete audit trail with SHA-256 hashing
- Reproducible results for compliance verification

## Formula Module

The `formulas.py` module provides standalone calculation functions:

```python
from backend.agents.gl_077_incentive_hunter.formulas import (
    calculate_incentive_value,
    calculate_payback_impact,
    calculate_stacking_limit,
    estimate_application_success,
    calculate_npv_with_incentives,
)

# Calculate 179D value
result = calculate_incentive_value(
    "179D",
    building_sqft=50000,
    is_prevailing_wage=True,
)
print(f"179D Value: ${result.total_value:,.2f}")

# Calculate payback impact
payback = calculate_payback_impact(
    project_cost=100000,
    annual_savings=20000,
    incentive_value=30000,
)
print(f"Payback reduced from {payback.original_payback_years} to {payback.adjusted_payback_years} years")

# Analyze incentive stacking
stacking = calculate_stacking_limit(
    project_cost=100000,
    federal_incentives=30000,
    state_incentives=10000,
    utility_incentives=5000,
)
print(f"Total stackable: ${stacking.total_stackable_value:,.2f}")
```

## Testing

Run the test suite:
```bash
pytest backend/agents/gl_077_incentive_hunter/test_agent.py -v
```

Test coverage target: 85%+

## API Reference

### IncentiveHunterAgent

#### Methods

**run(input_data: IncentiveHunterInput) -> IncentiveHunterOutput**

Execute comprehensive incentive analysis.

**Arguments:**
- `input_data`: Complete input including location, equipment, and project scope

**Returns:**
- `IncentiveHunterOutput`: Complete analysis with available incentives and recommendations

### Incentive Types

| Type | Description |
|------|-------------|
| UTILITY_REBATE | Cash rebate from utility company |
| FEDERAL_TAX_CREDIT | Federal income tax credit (ITC, PTC) |
| STATE_TAX_CREDIT | State income tax credit |
| GRANT | Direct cash grant |
| LOW_INTEREST_LOAN | Subsidized financing |
| ACCELERATED_DEPRECIATION | MACRS bonus depreciation |
| PERFORMANCE_INCENTIVE | Performance-based payment |

### Eligibility States

| State | Description |
|-------|-------------|
| ELIGIBLE | Meets all requirements |
| LIKELY_ELIGIBLE | Meets most requirements |
| CONDITIONAL | Some requirements need verification |
| NOT_ELIGIBLE | Does not meet requirements |
| REQUIRES_REVIEW | Manual review needed |

## Version History

- **1.0.0** (2024): Initial release with DSIRE, IRA, and utility program support

## License

Proprietary - GreenLang Platform
