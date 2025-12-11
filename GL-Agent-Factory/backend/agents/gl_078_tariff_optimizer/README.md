# GL-078: Tariff Optimizer Agent (TARIFFOPTIMIZER)

## Overview

The TariffOptimizerAgent analyzes utility rate structures and provides recommendations for rate selection and load management to minimize electricity costs. It evaluates time-of-use rates, demand charges, and load shifting opportunities.

## Features

- **Rate Comparison**: Compare multiple utility rate schedules
- **TOU Analysis**: Time-of-use period optimization
- **Demand Management**: Demand charge analysis and peak shaving recommendations
- **Load Shifting**: Identify load shifting opportunities for cost reduction
- **Battery Economics**: Evaluate battery storage for rate arbitrage
- **Provenance Tracking**: Complete SHA-256 audit trail

## Standards/References

- OpenEI USURDB (Utility Rate Database)
- FERC Form 1 rate data
- Utility-specific tariff books

## Installation

```python
from backend.agents.gl_078_tariff_optimizer import (
    TariffOptimizerAgent,
    TariffOptimizerInput,
    UsageProfile,
    TariffOption,
    RateSchedule,
    RateType,
)
```

## Quick Start

```python
from backend.agents.gl_078_tariff_optimizer import (
    TariffOptimizerAgent,
    TariffOptimizerInput,
    UsageProfile,
    TariffOption,
    RateSchedule,
    RateType,
    SeasonType,
    LoadType,
)

# Initialize agent
agent = TariffOptimizerAgent()

# Define usage profile
usage_profile = UsageProfile(
    utility="PG&E",
    hourly_kwh=[50, 45, 42, 40, 42, 50, 80, 120, 150, 160, 165, 170,
                175, 180, 185, 190, 200, 195, 190, 180, 160, 130, 90, 70],
    peak_demand_kw=200,
    load_factor=0.65,
    shiftable_load_kw=30,
    shiftable_load_types=[LoadType.HVAC, LoadType.EV_CHARGING],
)

# Define available tariffs
tariffs = [
    TariffOption(
        rate_schedule=RateSchedule(
            rate_id="E-19",
            name="Medium General Demand TOU",
            rate_type=RateType.TOU,
            energy_charge_on_peak=0.28,
            energy_charge_mid_peak=0.18,
            energy_charge_off_peak=0.10,
            demand_charge_facility=15.50,
            demand_charge_on_peak=8.25,
            customer_charge_monthly=85.00,
        ),
        is_current=True,
    ),
    TariffOption(
        rate_schedule=RateSchedule(
            rate_id="B-19",
            name="Medium General Demand",
            rate_type=RateType.TOU,
            energy_charge_on_peak=0.25,
            energy_charge_mid_peak=0.15,
            energy_charge_off_peak=0.08,
            demand_charge_facility=12.00,
            customer_charge_monthly=65.00,
        ),
    ),
]

# Create input
input_data = TariffOptimizerInput(
    usage_profile=usage_profile,
    available_tariffs=tariffs,
    include_demand_management=True,
    include_load_shifting=True,
)

# Run analysis
result = agent.run(input_data)

# Access results
print(f"Current Annual Cost: ${result.current_tariff_cost_usd:,.2f}")
print(f"Optimal Tariff: {result.optimal_tariff.rate_name}")
print(f"Potential Savings: ${result.optimal_tariff.savings_vs_current_usd:,.2f}")
print(f"Savings Percentage: {result.optimal_tariff.savings_percent}%")

# Demand analysis
print(f"\nDemand Charges: ${result.demand_analysis.total_demand_charge_usd:,.2f}")
print(f"Peak Shaving Potential: {result.demand_analysis.peak_shaving_potential_kw} kW")

# Load shift opportunities
for opp in result.load_shift_opportunities:
    print(f"\nLoad Shift: {opp.load_type.value}")
    print(f"  Annual Savings: ${opp.annual_savings_usd:,.2f}")
    print(f"  Payback: {opp.payback_months} months")
```

## Input Schema

### UsageProfile
| Field | Type | Description |
|-------|------|-------------|
| hourly_kwh | List[float] | Hourly kWh values (24 or 8760 hours) |
| peak_demand_kw | float | Maximum demand (kW) |
| load_factor | float | Load factor (0-1) |
| shiftable_load_kw | float | Amount of shiftable load |
| shiftable_load_types | List[LoadType] | Types of loads that can shift |

### RateSchedule
| Field | Type | Description |
|-------|------|-------------|
| rate_id | str | Rate schedule identifier |
| name | str | Rate schedule name |
| rate_type | RateType | FLAT, TOU, TIERED, etc. |
| energy_charge_on_peak | float | $/kWh on-peak |
| energy_charge_mid_peak | float | $/kWh mid-peak |
| energy_charge_off_peak | float | $/kWh off-peak |
| demand_charge_facility | float | $/kW facility demand |
| demand_charge_on_peak | float | $/kW on-peak demand |

## Output Schema

### TariffOptimizerOutput
| Field | Type | Description |
|-------|------|-------------|
| current_tariff_cost_usd | float | Current annual cost |
| optimal_tariff | TariffRecommendation | Best tariff option |
| tariff_recommendations | List[TariffRecommendation] | Ranked options |
| demand_analysis | DemandChargeAnalysis | Demand charge analysis |
| load_shift_opportunities | List[LoadShiftOpportunity] | Load shifting options |
| savings_analysis | SavingsAnalysis | Complete savings breakdown |

### TariffRecommendation
| Field | Type | Description |
|-------|------|-------------|
| rank | int | Recommendation rank (1=best) |
| rate_id | str | Rate schedule ID |
| estimated_annual_cost_usd | float | Estimated annual cost |
| savings_vs_current_usd | float | Savings vs current tariff |
| savings_percent | float | Savings percentage |

## Calculation Methods

### TOU Energy Cost
```
Energy_Cost = ON_PEAK_kWh * ON_PEAK_RATE
            + MID_PEAK_kWh * MID_PEAK_RATE
            + OFF_PEAK_kWh * OFF_PEAK_RATE
```

### Demand Charges
```
Demand_Cost = Peak_kW * Facility_Rate + OnPeak_kW * OnPeak_Rate
```

### Load Factor
```
Load_Factor = Total_kWh / (Peak_kW * Hours_in_Period)
```

### Load Shift Savings
```
Savings = Shifted_kWh * (OnPeak_Rate - OffPeak_Rate)
```

### Battery Arbitrage Value
```
Round_Trip_Efficiency = Charge_Eff * Discharge_Eff
Effective_Spread = OnPeak_Rate - OffPeak_Rate / RTE
Annual_Value = Capacity * Spread * Cycles * Days
```

## Rate Types

| Type | Description |
|------|-------------|
| FLAT | Single rate for all hours |
| TOU | Time-of-use with peak periods |
| TIERED | Usage-based tiers |
| TOU_TIERED | Combined TOU and tiered |
| DEMAND | Demand-based pricing |

## Load Types

| Type | Description |
|------|-------------|
| HVAC | Heating, ventilation, cooling |
| LIGHTING | Lighting systems |
| PROCESS | Industrial process loads |
| EV_CHARGING | Electric vehicle charging |
| REFRIGERATION | Refrigeration systems |
| COMPRESSED_AIR | Air compressors |

## Zero-Hallucination Guarantee

All calculations use deterministic formulas:
- Standard utility rate calculation methods
- No LLM inference in cost calculations
- Complete reproducibility for audit compliance
- Rate structures from utility tariff books

## Formula Module

The `formulas.py` module provides standalone calculation functions:

```python
from backend.agents.gl_078_tariff_optimizer.formulas import (
    calculate_tou_cost,
    calculate_demand_charge,
    calculate_optimal_shift,
    calculate_load_factor,
)

# Calculate TOU costs
tou_result = calculate_tou_cost(
    hourly_kwh=hourly_data,
    on_peak_rate=0.25,
    mid_peak_rate=0.15,
    off_peak_rate=0.08,
)

# Calculate demand charges
demand_result = calculate_demand_charge(
    peak_demand_kw=200,
    facility_demand_rate=15.00,
    on_peak_demand_rate=8.00,
)

# Analyze load shifting
shift_result = calculate_optimal_shift(
    on_peak_kwh=1000,
    mid_peak_kwh=800,
    off_peak_kwh=1200,
    on_peak_rate=0.25,
    mid_peak_rate=0.15,
    off_peak_rate=0.08,
)
```

## Testing

Run the test suite:
```bash
pytest backend/agents/gl_078_tariff_optimizer/test_agent.py -v
```

Test coverage target: 85%+

## Version History

- **1.0.0** (2024): Initial release with TOU, demand charge, and load shifting analysis

## License

Proprietary - GreenLang Platform
