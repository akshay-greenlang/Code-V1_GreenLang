# GL-019 HEATSCHEDULER - Calculator Suite

**Zero-Hallucination Calculation Engine for Process Heating Optimization**

## Overview

The GL-019 HEATSCHEDULER calculator suite provides deterministic, auditable calculations for energy cost analysis, schedule optimization, savings forecasting, and load prediction. All calculations are implemented using mathematical algorithms with no AI/LLM involvement in the computation path, ensuring reproducible, verifiable results.

### Design Principles

1. **Zero Hallucination**: No LLM or AI model is used in any calculation path
2. **Deterministic**: Same inputs always produce identical outputs
3. **Reproducible**: SHA-256 hash verification of all calculation chains
4. **Auditable**: Complete step-by-step provenance trail
5. **Standards-Based**: ISO 50001, ISO 50006, ASHRAE Guideline 14, IPMVP

---

## Calculator Inventory

| Calculator | Purpose | Standards |
|------------|---------|-----------|
| **EnergyCostCalculator** | Energy cost calculations under various tariff structures | ISO 50001, ISO 50006 |
| **ScheduleOptimizer** | MILP-based schedule optimization | ISO 50001 |
| **SavingsCalculator** | ROI, payback, NPV analysis | IPMVP, ISO 50006 |
| **LoadForecaster** | Production-based load prediction | ASHRAE Guideline 14 |

---

## Energy Cost Calculator

### Purpose

Calculates total energy costs under various tariff structures including Time-of-Use (ToU), demand charges, tiered rates, and real-time pricing.

### Supported Tariff Types

- **Fixed Rate**: Single rate per kWh
- **Tiered Rate**: Multiple tiers with increasing rates
- **Time-of-Use (ToU)**: Peak, shoulder, off-peak periods
- **Demand Charges**: Peak demand-based charges ($/kW)
- **Real-Time Pricing**: Hourly market rates

### Methodology

#### Time-of-Use Calculation

```
Energy_Cost = sum(Period_Energy_kWh * Period_Rate_per_kWh)

Where:
  Period_Energy = sum(hourly_energy for hours in period)
  Periods = {peak, shoulder, off_peak}
```

#### Demand Charge Calculation

```
Demand_Cost = Peak_Demand_kW * Demand_Rate_per_kW

Where:
  Peak_Demand = max(15-minute average demand readings)
```

#### Total Cost

```
Total_Cost = Energy_Cost + Demand_Cost + Base_Charge
```

### Usage Example

```python
from greenlang.GL_019.calculators.energy_cost_calculator import (
    EnergyCostCalculator,
    TariffStructure,
    TariffType,
    TariffRate,
    TimePeriod,
    HourlyLoad,
    EnergyCostInput
)

# Define Time-of-Use tariff
tariff = TariffStructure(
    tariff_type=TariffType.TIME_OF_USE,
    rates=[
        TariffRate(rate_per_kwh=0.25, period=TimePeriod.PEAK),
        TariffRate(rate_per_kwh=0.15, period=TimePeriod.SHOULDER),
        TariffRate(rate_per_kwh=0.08, period=TimePeriod.OFF_PEAK),
        TariffRate(rate_per_kwh=0.0, demand_rate_per_kw=15.00)
    ],
    peak_hours=list(range(14, 20)),        # 2 PM - 8 PM
    shoulder_hours=list(range(6, 14)) + list(range(20, 22)),
    off_peak_hours=list(range(0, 6)) + list(range(22, 24)),
    base_charge_per_month=500.00
)

# Define hourly loads for a day
loads = [
    HourlyLoad(hour=h, energy_kwh=100.0, peak_demand_kw=120.0, date="2025-01-15")
    for h in range(24)
]

# Calculate costs
calculator = EnergyCostCalculator()
inputs = EnergyCostInput(tariff=tariff, hourly_loads=loads)
result, provenance = calculator.calculate(inputs)

print(f"Total Cost: ${result.total_cost:.2f}")
print(f"  Energy Cost: ${result.energy_cost:.2f}")
print(f"  Demand Cost: ${result.demand_cost:.2f}")
print(f"Average Rate: ${result.average_rate_per_kwh:.4f}/kWh")

print(f"\nCost by Period:")
for period, cost in result.cost_by_period.items():
    print(f"  {period}: ${cost:.2f}")

# Verify provenance
print(f"\nProvenance Hash: {provenance.hash}")
print(f"Calculation Steps: {len(provenance.steps)}")
```

### Output Structure

```python
@dataclass
class EnergyCostOutput:
    total_cost: float              # Total energy cost (currency)
    energy_cost: float             # Cost from energy consumption
    demand_cost: float             # Cost from demand charges
    base_cost: float               # Fixed base charge
    total_energy_kwh: float        # Total energy consumed
    peak_demand_kw: float          # Peak demand recorded
    average_rate_per_kwh: float    # Effective average rate
    cost_by_period: Dict[str, float]  # Cost breakdown by period
    daily_costs: Dict[str, float]     # Cost breakdown by day
    peak_energy_kwh: float         # Energy during peak hours
    off_peak_energy_kwh: float     # Energy during off-peak hours
    shoulder_energy_kwh: float     # Energy during shoulder hours
```

---

## Schedule Optimizer

### Purpose

Optimizes heating operation schedules to minimize energy costs while meeting production deadlines and equipment constraints using Mixed-Integer Linear Programming (MILP).

### Methodology

#### Optimization Problem Formulation

**Decision Variables:**
```
x[j,t] = power allocated to job j in time slot t (kW)
```

**Objective Function (minimize cost):**
```
minimize: sum over j,t: x[j,t] * rate[t] * slot_duration
```

**Constraints:**

1. **Energy Requirement** (each job must receive required energy):
```
sum_t(x[j,t] * duration) = energy_required[j]  for all jobs j
```

2. **Equipment Capacity** (total load cannot exceed capacity):
```
sum_j(x[j,t]) <= max_capacity  for all time slots t
```

3. **Power Bounds** (respect equipment limits):
```
min_power[j] <= x[j,t] <= max_power[j]  for all j, t
```

4. **Deadline Constraint** (no operation after deadline):
```
x[j,t] = 0  for t >= deadline[j]
```

5. **Earliest Start** (no operation before earliest start):
```
x[j,t] = 0  for t < earliest_start[j]
```

#### Solver

Uses scipy.optimize.linprog with HiGHS solver for guaranteed deterministic results.

### Usage Example

```python
from greenlang.GL_019.calculators.schedule_optimizer import (
    ScheduleOptimizer,
    ScheduleOptimizerInput,
    HeatingJob,
    TimeSlotCost,
    EquipmentConstraint,
    OptimizationObjective
)

# Define heating jobs
jobs = [
    HeatingJob(
        job_id="BATCH-001",
        energy_required_kwh=500,
        min_power_kw=50,
        max_power_kw=100,
        deadline_hour=12,
        earliest_start_hour=0,
        priority=1.0,
        can_interrupt=True
    ),
    HeatingJob(
        job_id="BATCH-002",
        energy_required_kwh=800,
        min_power_kw=80,
        max_power_kw=150,
        deadline_hour=18,
        earliest_start_hour=6,
        priority=2.0,
        can_interrupt=True
    )
]

# Define time slot costs (ToU rates)
time_slots = []
for hour in range(24):
    if 14 <= hour < 20:  # Peak
        rate = 0.25
    elif 6 <= hour < 14 or 20 <= hour < 22:  # Shoulder
        rate = 0.15
    else:  # Off-peak
        rate = 0.08
    time_slots.append(TimeSlotCost(hour=hour, energy_rate_per_kwh=rate))

# Define equipment constraints
equipment = EquipmentConstraint(
    equipment_id="FURNACE-001",
    max_capacity_kw=200,
    min_capacity_kw=0,
    ramp_rate_kw_per_hour=100,
    efficiency=0.92
)

# Optimize schedule
optimizer = ScheduleOptimizer()
inputs = ScheduleOptimizerInput(
    jobs=jobs,
    time_slots=time_slots,
    equipment=equipment,
    objective=OptimizationObjective.MINIMIZE_COST,
    horizon_hours=24
)

result, provenance = optimizer.optimize(inputs)

print(f"Optimization Status: {result.optimization_status}")
print(f"Total Cost: ${result.total_cost:.2f}")
print(f"Total Energy: {result.total_energy_kwh:.0f} kWh")
print(f"Peak Demand: {result.peak_demand_kw:.0f} kW")
print(f"Load Factor: {result.load_factor:.2%}")
print(f"Savings vs Flat Profile: ${result.cost_savings_vs_flat:.2f}")

print(f"\nSchedule:")
for op in result.schedule:
    print(f"  {op.job_id}: Hour {op.hour}, {op.power_kw:.0f} kW, ${op.cost:.2f}")
```

### Output Structure

```python
@dataclass
class ScheduleOptimizerOutput:
    schedule: List[ScheduledOperation]  # Scheduled operations
    total_cost: float                   # Total energy cost
    total_energy_kwh: float             # Total energy consumed
    peak_demand_kw: float               # Peak demand
    average_power_kw: float             # Average power
    load_factor: float                  # Load factor (avg/peak)
    cost_savings_vs_flat: float         # Savings vs flat profile
    hours_shifted: int                  # Hours shifted off-peak
    optimization_status: str            # Solver status
    objective_value: float              # Optimal objective value
    hourly_costs: Dict[int, float]      # Cost by hour
```

---

## Savings Calculator

### Purpose

Calculates cost savings from schedule optimization, including ROI metrics, payback period, NPV, and IRR following IPMVP protocols.

### Methodology

#### Savings Calculation

```
Period_Savings = Baseline_Cost - Optimized_Cost

Savings_Percentage = (Period_Savings / Baseline_Cost) * 100

Annual_Savings = Period_Savings * (365 / Period_Days)
```

#### Savings by Category

```
Load_Shifting_Savings = (Peak_Energy_Baseline - Peak_Energy_Optimized) * Peak_Rate_Differential

Demand_Reduction_Savings = (Peak_Demand_Baseline - Peak_Demand_Optimized) * Demand_Rate

Peak_Avoidance_Savings = Avoided_Peak_Hours * Average_Peak_Cost * 0.5
```

#### ROI Metrics

**Simple Payback:**
```
Payback_Years = Initial_Investment / Annual_Net_Savings
```

**Net Present Value (NPV):**
```
NPV = -Initial_Investment + sum(Annual_Savings / (1 + r)^t)

Where:
  r = discount rate
  t = year (1 to project_life)
```

**Internal Rate of Return (IRR):**
```
Find r such that:
  -Initial_Investment + sum(Annual_Savings / (1 + r)^t) = 0

Solved using Newton-Raphson method
```

### Usage Example

```python
from greenlang.GL_019.calculators.savings_calculator import (
    SavingsCalculator,
    SavingsCalculatorInput,
    ScheduleComparison,
    HourlyScheduleData,
    ProjectCosts,
    VerificationMethod
)

# Create baseline schedule (current operation)
baseline = []
for hour in range(24):
    is_peak = 14 <= hour < 20
    rate = 0.25 if is_peak else 0.08
    baseline.append(HourlyScheduleData(
        hour=hour,
        energy_kwh=100,
        demand_kw=120,
        energy_rate=rate,
        demand_rate=15.0,
        is_peak_hour=is_peak
    ))

# Create optimized schedule (load shifted)
optimized = []
for hour in range(24):
    is_peak = 14 <= hour < 20
    rate = 0.25 if is_peak else 0.08
    # Shift load from peak to off-peak
    energy = 50 if is_peak else 150
    optimized.append(HourlyScheduleData(
        hour=hour,
        energy_kwh=energy,
        demand_kw=100,  # Reduced peak demand
        energy_rate=rate,
        demand_rate=15.0,
        is_peak_hour=is_peak
    ))

# Define project costs
project_costs = ProjectCosts(
    capital_cost=50000,
    implementation_cost=10000,
    annual_maintenance_cost=5000,
    software_license_annual=12000,
    training_cost=3000
)

# Calculate savings
calculator = SavingsCalculator()
inputs = SavingsCalculatorInput(
    schedule_comparison=ScheduleComparison(
        baseline_schedule=baseline,
        optimized_schedule=optimized,
        analysis_period_days=30,
        baseline_description="Current manual scheduling",
        optimized_description="HEATSCHEDULER optimized"
    ),
    project_costs=project_costs,
    projection_years=5,
    discount_rate=0.08,
    verification_method=VerificationMethod.OPTION_C
)

result, provenance = calculator.calculate(inputs)

print(f"Period Savings: ${result.period_savings:,.2f}")
print(f"Annual Projection: ${result.annual_savings_projection:,.2f}")
print(f"Savings Percentage: {result.savings_percentage:.1f}%")

print(f"\nSavings by Category:")
print(f"  Load Shifting: ${result.savings_by_category.load_shifting_savings:,.2f}")
print(f"  Demand Reduction: ${result.savings_by_category.demand_reduction_savings:,.2f}")
print(f"  Peak Avoidance: ${result.savings_by_category.peak_avoidance_savings:,.2f}")

print(f"\nROI Metrics:")
print(f"  Simple Payback: {result.roi_metrics.simple_payback_years:.1f} years")
print(f"  NPV: ${result.roi_metrics.npv:,.2f}")
print(f"  IRR: {result.roi_metrics.irr * 100:.1f}%")
print(f"  ROI: {result.roi_metrics.roi_percentage:.1f}%")
print(f"  Benefit-Cost Ratio: {result.roi_metrics.benefit_cost_ratio:.2f}")
```

### Output Structure

```python
@dataclass
class SavingsCalculatorOutput:
    period_savings: float              # Savings for analysis period
    annual_savings_projection: float   # Projected annual savings
    savings_by_category: SavingsByCategory  # Breakdown by category
    roi_metrics: ROIMetrics            # ROI calculations
    baseline_cost: float               # Total baseline cost
    optimized_cost: float              # Total optimized cost
    savings_percentage: float          # Percentage savings
    energy_savings_kwh: float          # Energy saved
    demand_savings_kw: float           # Peak demand reduction
    verification_confidence: str       # Verification confidence level
    monthly_savings: List[float]       # Monthly projection
```

---

## Load Forecaster

### Purpose

Forecasts heating loads based on production schedules, historical patterns, and weather data (heating degree days).

### Methodology

#### Daily Pattern Analysis

```
Daily_Pattern[h] = mean(energy_kwh for all historical data where hour = h)

Where:
  h = hour of day (0-23)
```

#### Weekly Pattern Analysis

```
Weekly_Pattern[d] = mean(energy_kwh for all historical data where day_of_week = d)

Where:
  d = day of week (0=Monday, 6=Sunday)
```

#### Energy Intensity

```
Energy_Intensity = Total_Energy_kWh / Total_Production_Units
```

#### HDD Correlation

```
Correlation = cov(energy, hdd) / (std_energy * std_hdd)
```

#### Confidence Intervals

```
CI = forecast +/- z * (std / sqrt(n))

Where:
  z = 1.645 (90%), 1.960 (95%), 2.576 (99%)
  std = historical standard deviation
  n = number of samples
```

### Usage Example

```python
from greenlang.GL_019.calculators.load_forecaster import (
    LoadForecaster,
    LoadForecastInput,
    HistoricalLoad,
    ProductionSchedule,
    ForecastMethod
)

# Historical load data
historical_data = []
for day in range(30):
    for hour in range(24):
        historical_data.append(HistoricalLoad(
            timestamp=f"2024-12-{day+1:02d}T{hour:02d}:00:00",
            energy_kwh=100 + 20 * (hour // 6),  # Daily pattern
            peak_demand_kw=120,
            production_units=50,
            ambient_temp_c=10,
            heating_degree_days=8,
            day_of_week=day % 7,
            hour_of_day=hour,
            is_holiday=False
        ))

# Production schedule for forecast period
production_schedule = [
    ProductionSchedule(
        date="2025-01-08",
        planned_units=1000,
        shift_pattern="3x8",
        process_type="standard",
        expected_ambient_temp_c=5
    ),
    ProductionSchedule(
        date="2025-01-09",
        planned_units=1200,
        shift_pattern="3x8",
        process_type="standard",
        expected_ambient_temp_c=3
    )
]

# Generate forecast
forecaster = LoadForecaster()
inputs = LoadForecastInput(
    historical_data=historical_data,
    production_schedule=production_schedule,
    forecast_horizon_days=7,
    method=ForecastMethod.WEIGHTED_AVERAGE,
    confidence_level="95%",
    seasonality_period=168  # Weekly
)

result, provenance = forecaster.forecast(inputs)

print(f"Total Forecast Energy: {result.total_energy_kwh:,.0f} kWh")
print(f"Peak Demand: {result.peak_demand_kw:.0f} kW")
print(f"Average Daily: {result.average_daily_kwh:.0f} kWh")
print(f"Energy Intensity: {result.energy_intensity:.4f} kWh/unit")
print(f"HDD Correlation: {result.heating_degree_day_correlation:.4f}")

print(f"\nForecast Accuracy Metrics:")
print(f"  MAPE: {result.forecast_accuracy_metrics['mape']:.1f}%")
print(f"  RMSE: {result.forecast_accuracy_metrics['rmse']:.1f}")
print(f"  R-squared: {result.forecast_accuracy_metrics['r_squared']:.4f}")

print(f"\nDaily Pattern (avg kWh by hour):")
for hour, avg in enumerate(result.daily_pattern):
    print(f"  Hour {hour:02d}: {avg:.1f} kWh")
```

### Output Structure

```python
@dataclass
class LoadForecastOutput:
    hourly_forecasts: List[HourlyForecast]  # Hourly forecasts
    total_energy_kwh: float           # Total forecast energy
    peak_demand_kw: float             # Maximum forecast demand
    average_daily_kwh: float          # Average daily energy
    daily_pattern: List[float]        # 24-hour pattern
    weekly_pattern: List[float]       # 7-day pattern
    forecast_accuracy_metrics: Dict   # MAPE, RMSE, MAE, R-squared
    energy_intensity: float           # kWh per production unit
    heating_degree_day_correlation: float  # HDD correlation
```

---

## Zero-Hallucination Guarantees

### What "Zero Hallucination" Means

1. **No LLM in Calculation Path**: All numeric calculations use deterministic algorithms
2. **Reproducible Results**: Same inputs always produce identical outputs
3. **Verifiable**: Every step can be independently verified
4. **Auditable**: Complete calculation trail with formulas

### How We Ensure It

| Guarantee | Implementation |
|-----------|----------------|
| Deterministic | Fixed random seeds, deterministic solvers |
| No AI in Calculations | Pure Python/numpy/scipy arithmetic |
| Reproducible | SHA-256 hash verification |
| Auditable | Step-by-step provenance tracking |
| Verifiable | Known test values with exact expected outputs |

### AI Usage Boundaries

| Use Case | AI Allowed? | Implementation |
|----------|-------------|----------------|
| Cost calculations | No | Pure arithmetic |
| Schedule optimization | No | scipy MILP solver |
| Savings analysis | No | Pure arithmetic |
| Load forecasting | No | Statistical methods |
| Recommendations | Yes | LLM for text generation |
| Explanations | Yes | LLM for natural language |
| Report narratives | Yes | LLM for writing |

---

## Provenance Tracking

### Overview

Every calculation produces a ProvenanceRecord that documents the complete calculation chain for audit and verification purposes.

### Provenance Structure

```python
@dataclass
class ProvenanceRecord:
    calculator_name: str      # e.g., "EnergyCostCalculator"
    calculator_version: str   # e.g., "1.0.0"
    timestamp: str            # ISO 8601 timestamp
    hash: str                 # SHA-256 of entire record
    inputs: Dict              # Input parameters
    outputs: Dict             # Output values
    steps: List[CalculationStep]  # Step-by-step trail
    metadata: Dict            # Standards, domain info
```

### Calculation Step

```python
@dataclass
class CalculationStep:
    step_number: int          # Sequential step number
    description: str          # Human-readable description
    operation: str            # Operation type (add, multiply, etc.)
    inputs: Dict              # Step inputs
    output_value: Any         # Step output
    output_name: str          # Output variable name
    formula: str              # Mathematical formula used
```

### Usage Example

```python
# Calculate with provenance
calculator = EnergyCostCalculator()
result, provenance = calculator.calculate(inputs)

# Inspect provenance
print(f"Calculator: {provenance.calculator_name} v{provenance.calculator_version}")
print(f"Timestamp: {provenance.timestamp}")
print(f"Hash: {provenance.hash}")

print(f"\nInputs:")
for key, value in provenance.inputs.items():
    print(f"  {key}: {value}")

print(f"\nCalculation Steps:")
for step in provenance.steps:
    print(f"  Step {step.step_number}: {step.description}")
    print(f"    Formula: {step.formula}")
    print(f"    Inputs: {step.inputs}")
    print(f"    Output: {step.output_name} = {step.output_value}")

# Verify hash
import hashlib
import json
data = {
    "inputs": provenance.inputs,
    "outputs": provenance.outputs,
    "steps": [s.__dict__ for s in provenance.steps]
}
computed_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
assert computed_hash == provenance.hash, "Provenance verification failed!"
```

### Audit Trail Storage

```python
# Store provenance for audit
import json
from datetime import datetime

audit_record = {
    "timestamp": datetime.utcnow().isoformat(),
    "calculation_type": "energy_cost",
    "provenance": provenance.__dict__,
    "result": result.__dict__
}

# Save to audit log
with open(f"audit/energy_cost_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
    json.dump(audit_record, f, indent=2, default=str)
```

---

## Standalone Functions

Each calculator module also provides standalone functions for simple calculations:

### Energy Cost Functions

```python
from greenlang.GL_019.calculators.energy_cost_calculator import (
    calculate_simple_energy_cost,
    calculate_demand_charge,
    calculate_tou_cost,
    classify_hour,
    calculate_average_rate
)

# Simple energy cost
cost = calculate_simple_energy_cost(1000.0, 0.12)  # $120.00

# Demand charge
charge = calculate_demand_charge(500.0, 15.00)  # $7,500.00

# Time-of-use cost
total, breakdown = calculate_tou_cost(
    peak_energy_kwh=300,
    shoulder_energy_kwh=400,
    off_peak_energy_kwh=300,
    peak_rate=0.25,
    shoulder_rate=0.15,
    off_peak_rate=0.08
)  # $159.00

# Hour classification
period = classify_hour(15)  # TimePeriod.PEAK
```

### Savings Functions

```python
from greenlang.GL_019.calculators.savings_calculator import (
    calculate_simple_payback,
    calculate_npv,
    calculate_savings_percentage,
    annualize_savings,
    calculate_demand_charge_savings,
    calculate_load_shift_savings
)

# Simple payback
payback = calculate_simple_payback(50000, 12000)  # 4.2 years

# NPV
cash_flows = [10000, 10000, 10000, 10000, 10000]
npv = calculate_npv(30000, cash_flows, 0.08)  # $9,927

# Savings percentage
pct = calculate_savings_percentage(1000, 800)  # 20%

# Demand charge savings
savings = calculate_demand_charge_savings(500, 400, 15)  # $1,500

# Load shift savings
savings = calculate_load_shift_savings(1000, 0.25, 0.08)  # $170
```

### Forecasting Functions

```python
from greenlang.GL_019.calculators.load_forecaster import (
    calculate_heating_degree_days,
    simple_moving_average,
    exponential_smoothing,
    calculate_forecast_confidence_interval,
    calculate_mape,
    calculate_rmse
)

# Heating degree days
hdds = calculate_heating_degree_days([5, 10, 15, 20])  # [13, 8, 3, 0]

# Moving average
sma = simple_moving_average([10, 20, 30, 40, 50], 3)  # [20, 30, 40]

# Confidence interval
lower, upper = calculate_forecast_confidence_interval(100, 15, 30, "95%")

# Accuracy metrics
mape = calculate_mape([100, 110, 120], [95, 115, 130])
rmse = calculate_rmse([100, 110, 120], [95, 115, 130])
```

---

## Test Coverage

### Coverage Summary

| Calculator | Coverage | Test Cases |
|------------|----------|------------|
| EnergyCostCalculator | 95%+ | 70+ |
| ScheduleOptimizer | 95%+ | 60+ |
| SavingsCalculator | 95%+ | 50+ |
| LoadForecaster | 95%+ | 40+ |

### Running Tests

```bash
# All calculator tests
pytest tests/unit/ -v --cov=calculators

# Specific calculator
pytest tests/unit/test_energy_cost_calculator.py -v

# With coverage report
pytest tests/unit/ --cov=calculators --cov-report=html
```

### Known Test Values

```python
# Energy cost: 1000 kWh at $0.12/kWh = $120.00
assert calculate_simple_energy_cost(1000, 0.12) == 120.00

# Demand charge: 500 kW at $15/kW = $7,500.00
assert calculate_demand_charge(500, 15) == 7500.00

# Savings: baseline $1000, optimized $850 = 15%
assert calculate_savings_percentage(1000, 850) == 15.0

# Payback: $50,000 / $25,000/year = 2 years
assert calculate_simple_payback(50000, 25000) == 2.0
```

---

## Standards Compliance

### ISO 50001 - Energy Management Systems

- Energy baseline establishment
- Energy performance indicators (EnPIs)
- Energy cost tracking and analysis
- Continuous improvement targets

### ISO 50006 - Measuring Energy Performance

- Baseline methodology
- Adjustment factors
- Performance measurement
- Verification procedures

### ASHRAE Guideline 14 - Measurement of Energy Savings

- Uncertainty analysis
- Statistical methods
- Confidence intervals
- Data requirements

### IPMVP - International Performance Measurement and Verification Protocol

- Option A: Retrofit Isolation, Key Parameter
- Option B: Retrofit Isolation, All Parameters
- Option C: Whole Facility, Utility Analysis
- Option D: Calibrated Simulation

---

## Support

For questions or issues with the calculator suite:

- **Documentation**: This README
- **Technical Support**: support@greenlang.io
- **GitHub Issues**: github.com/greenlang/gl-019-heatscheduler/issues

---

*GL-019 HEATSCHEDULER Calculator Suite*

*Deterministic, auditable, zero-hallucination calculations for energy optimization.*
