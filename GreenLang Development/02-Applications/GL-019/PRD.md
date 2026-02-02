# Product Requirements Document: GL-019 HEATSCHEDULER

**Version:** 1.0
**Date:** 2025-12-03
**Product Manager:** GreenLang Product Team
**Status:** Approved for Development

---

## 1. Executive Summary

### Problem Statement

Industrial process heating operations (furnaces, kilns, dryers, heat treatment systems) consume 15-25% of total manufacturing energy costs, with significant waste occurring due to suboptimal scheduling. Plant operators schedule heating operations based on production demands without considering time-of-use energy tariffs, demand charges, or grid signals, resulting in peak-hour energy consumption that can cost 2-4x more than off-peak rates.

Current challenges:
- **Peak energy costs:** 40-60% of heating operations occur during peak tariff periods (10 AM - 6 PM), incurring premium energy rates of $0.15-$0.35/kWh vs. $0.05-$0.10/kWh off-peak
- **Demand charge penalties:** Uncoordinated heating startups create demand spikes of 20-50 MW, triggering demand charges of $10-$25 per kW/month
- **Manual scheduling:** Production planners schedule heating based on production deadlines only, with no energy cost visibility or optimization
- **Missed demand response revenue:** Facilities miss $50,000-$500,000/year in demand response program payments by not curtailing heating during grid emergencies
- **Thermal storage underutilization:** Available thermal storage capacity (molten salt, hot water, steam accumulators) is rarely leveraged for load shifting
- **Equipment conflicts:** Multiple heating systems competing for limited electrical capacity cause production bottlenecks

### Solution Overview

GL-019 HEATSCHEDULER is an AI-powered process heating scheduler that optimizes heating operation timing to minimize energy costs while meeting production deadlines. The agent integrates with production planning systems, energy management platforms, and real-time energy markets to deliver:

1. **Production Schedule Integration:** Real-time sync with ERP/MES systems to understand production batches, deadlines, and equipment dependencies
2. **Energy Tariff Optimization:** Time-of-use rate awareness, demand charge minimization, and real-time pricing integration
3. **Equipment Coordination:** Multi-asset scheduling to prevent demand spikes and equipment conflicts
4. **Load Shifting Recommendations:** Intelligent recommendations to shift heating loads to off-peak hours while meeting production requirements
5. **Demand Response Integration:** Automatic participation in utility demand response programs for additional revenue
6. **Cost Savings Forecasting:** Predictive analytics showing expected savings from optimized schedules
7. **What-If Analysis:** Simulation capabilities to test schedule changes before implementation

Zero-Hallucination Guarantee: All schedule optimization uses deterministic mathematical optimization algorithms (mixed-integer linear programming, constraint satisfaction). No AI is used for numeric scheduling decisions, ensuring 100% predictable and auditable results.

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Energy Cost Reduction | 15% average | Monthly energy bill reduction vs. baseline |
| Production Deadline Compliance | 99% | % of batches completed on time |
| Demand Charge Reduction | 25% average | Monthly demand charge reduction |
| Peak Load Reduction | 30% | MW reduction during peak tariff hours |
| Demand Response Revenue | $100K/year avg | Annual DR program payments per facility |
| Schedule Optimization Time | <5 minutes | Time to generate optimized weekly schedule |
| Equipment Utilization | +10% | Improvement in furnace/kiln utilization |
| Customer Payback Period | <12 months | Time to recover implementation cost |
| Customer Satisfaction (NPS) | >50 | Net Promoter Score survey |
| Revenue (Year 1) | $80M ARR | Annual recurring revenue |

---

## 2. Market & Competitive Analysis

### Market Opportunity

**Total Addressable Market (TAM):** $7B
- 1.2 million industrial facilities with significant process heating operations globally
- Average energy cost: $2-10M per facility per year
- Potential savings: 10-20% ($200K-$2M per facility)

**Breakdown by Industry:**
- Industrial Manufacturing: $2.5B (steel, aluminum, glass, ceramics)
- Chemical Processing: $1.8B (petrochemicals, specialty chemicals)
- Oil & Gas Refineries: $1.2B (crude distillation, cracking, reforming)
- Food & Beverage Processing: $0.9B (pasteurization, drying, sterilization)
- Pulp & Paper: $0.6B (drying, pulping, bleaching)

**Serviceable Addressable Market (SAM):** $2.1B
- Facilities with >$1M annual energy spend
- Facilities with flexible production schedules (not continuous 24/7 operations)
- Facilities in markets with time-of-use tariffs or demand charges
- North America, EU, and developed Asia-Pacific markets

**Serviceable Obtainable Market (SOM):** $80M (Year 1)
- Initial focus: North America industrial manufacturers and food processors
- Target: 200 facilities in Year 1
- Average deal size: $400K/year

### Target Customers

**Primary Segments:**

1. **Industrial Manufacturing Plants (Steel, Aluminum, Glass)**
   - Company size: 500-10,000 employees
   - Heating equipment: 10-50 furnaces, kilns, heat treatment systems
   - Annual energy cost: $5M-$50M
   - Pain points: High demand charges, peak energy costs, production schedule conflicts
   - Decision makers: VP Operations, Energy Manager, Plant Manager

2. **Chemical and Petrochemical Plants**
   - Company size: 1,000-20,000 employees
   - Heating equipment: 20-100 fired heaters, reactors, distillation columns
   - Annual energy cost: $10M-$100M
   - Pain points: Batch scheduling complexity, energy intensity, carbon footprint
   - Decision makers: VP Manufacturing, Process Engineering Manager

3. **Oil & Gas Refineries**
   - Company size: 500-5,000 employees
   - Heating equipment: 10-50 crude heaters, reformer furnaces, cracking units
   - Annual energy cost: $20M-$200M
   - Pain points: Fuel gas optimization, demand charges, maintenance coordination
   - Decision makers: Refinery Manager, Energy Optimization Manager

4. **Food & Beverage Processing**
   - Company size: 200-5,000 employees
   - Heating equipment: 5-30 pasteurizers, dryers, ovens, sterilizers
   - Annual energy cost: $1M-$20M
   - Pain points: Production schedule variability, seasonal demand, food safety compliance
   - Decision makers: Plant Manager, Operations Director, Sustainability Manager

**Secondary Segments:**
- Pulp and paper mills
- Pharmaceutical manufacturing
- Textile processing
- Cement and concrete production
- Data centers (cooling as inverse heating problem)

### Competitive Landscape

| Solution | Strengths | Weaknesses | HEATSCHEDULER Advantage |
|----------|-----------|------------|------------------------|
| **Manual Scheduling (Excel)** | No cost, familiar | No optimization, time-consuming, error-prone | 15% cost savings, <5 min scheduling |
| **ERP Production Scheduling (SAP, Oracle)** | Integrated with production | Energy-blind, no tariff optimization | Energy-aware optimization, tariff integration |
| **Energy Management Systems (Schneider, Siemens)** | Real-time monitoring | No production integration, reactive not predictive | Production + energy co-optimization |
| **Demand Response Aggregators (Enel X, CPower)** | DR program access | No production awareness, manual curtailment | Automated DR with production protection |
| **Custom In-House Solutions** | Tailored to facility | $500K-$2M development, maintenance burden | SaaS model, $200K-$600K/year, no maintenance |
| **Energy Consultants** | Expertise | $200K-$500K per project, one-time analysis | Continuous optimization, 24/7 automation |

**Key Differentiators:**
1. **Production-Energy Co-Optimization:** Only solution that integrates production schedules with energy tariffs for holistic optimization
2. **Zero-Hallucination Scheduling:** Deterministic MILP optimization ensures predictable, auditable schedules
3. **Real-Time Tariff Integration:** Connects to utility APIs, ISO markets, and real-time pricing feeds
4. **Demand Response Automation:** Automatic curtailment bids with production deadline protection
5. **What-If Simulation:** Test schedule changes before committing to production
6. **Rapid ROI:** <12 month payback via 15% energy cost reduction

---

## 3. Features & Requirements

### Must-Have Features (P0 - Launch Blockers)

---

#### Feature 1: Production Schedule Integration

**User Story:**
```
As a production planner,
I want HEATSCHEDULER to automatically import my production schedule from SAP/Oracle/MES,
So that heating operations are optimized around actual production requirements without manual data entry.
```

**Acceptance Criteria:**
- [ ] Integrates with SAP PP/DS, Oracle SCM, and Rockwell FactoryTalk MES via REST API
- [ ] Imports production batches with: batch ID, product type, start time, end time, heating equipment required
- [ ] Imports maintenance windows and planned downtime from CMMS (SAP PM, Maximo)
- [ ] Syncs production schedule changes in real-time (<5 minute latency)
- [ ] Handles batch dependencies (e.g., Batch B cannot start until Batch A completes)
- [ ] Supports multi-shift schedules (8/12/24-hour shifts)
- [ ] Validates schedule feasibility (no equipment double-booking)
- [ ] Stores 90 days of historical schedule data for pattern analysis

**Non-Functional Requirements:**
- Performance: <5 second sync time for 1,000 batches
- Reliability: 99.9% API uptime, automatic retry with exponential backoff
- Security: OAuth 2.0 authentication, TLS 1.3 encryption
- Usability: Manual override capability via web UI

**Dependencies:**
- ERP connector libraries (SAP RFC, Oracle REST)
- MES integration adapters
- Schedule validation engine

**Estimated Effort:** 4 weeks (1 integration engineer, 1 backend engineer)

**Business Justification:** Eliminates 4+ hours/week of manual schedule data entry and ensures optimization is based on real-time production requirements.

---

#### Feature 2: Energy Tariff Management

**User Story:**
```
As an energy manager,
I want HEATSCHEDULER to understand my complex energy tariff structure including time-of-use rates, demand charges, and real-time pricing,
So that heating schedules minimize total energy cost, not just energy consumption.
```

**Acceptance Criteria:**
- [ ] Supports time-of-use (TOU) rate structures with up to 8 periods per day
- [ ] Supports seasonal rate variations (summer/winter, weekday/weekend)
- [ ] Calculates demand charges based on peak 15-minute/30-minute demand
- [ ] Integrates with real-time pricing feeds (PJM, ERCOT, CAISO, NYISO)
- [ ] Supports day-ahead and real-time locational marginal pricing (LMP)
- [ ] Handles ratchet clauses (peak demand sets rate for 12 months)
- [ ] Imports utility rate schedules from OpenEI or manual configuration
- [ ] Calculates blended energy cost considering all tariff components
- [ ] Forecasts next-day energy prices using historical patterns
- [ ] Alerts when prices exceed configurable thresholds (e.g., >$100/MWh)

**Tariff Calculation Methodology:**

```
Total Energy Cost = Energy Charge + Demand Charge + Other Charges

Energy Charge = SUM(kWh_period × Rate_period) for all TOU periods
  Where:
  - kWh_period = energy consumption in each TOU period
  - Rate_period = $/kWh rate for that period (on-peak, mid-peak, off-peak)

Demand Charge = Peak_kW × Demand_Rate
  Where:
  - Peak_kW = highest 15-min average demand in billing period
  - Demand_Rate = $/kW/month (may vary by season, on-peak vs. off-peak)

Ratchet Adjustment = MAX(Current_Peak, 0.8 × Historical_Peak_12mo) × Ratchet_Rate

Real-Time Pricing Integration:
- Poll ISO API every 5 minutes for real-time LMP
- Recalculate optimal schedule when price changes >20%
```

**Edge Cases:**
- Holiday schedules (different rates) -> Import utility holiday calendar
- Meter reading misalignment -> Use normalized billing period
- Multiple meters -> Aggregate demand across all meters
- Rate schedule changes -> Version rate schedules with effective dates

**Non-Functional Requirements:**
- Accuracy: <1% variance from actual utility bill
- Performance: <100 ms tariff calculation per schedule
- Data freshness: <5 minute latency for real-time pricing

**Estimated Effort:** 3 weeks (1 energy domain expert, 1 backend engineer)

**Business Justification:** Demand charges often represent 30-50% of industrial energy bills. Accurate tariff modeling is essential for meaningful optimization.

---

#### Feature 3: Equipment Availability Tracking

**User Story:**
```
As a maintenance engineer,
I want HEATSCHEDULER to know the real-time status and capacity of all heating equipment,
So that schedules are optimized around actual equipment availability, not theoretical capacity.
```

**Acceptance Criteria:**
- [ ] Tracks equipment status: AVAILABLE, RUNNING, MAINTENANCE, FAULTED, WARMING_UP
- [ ] Integrates with SCADA/DCS for real-time equipment status via OPC UA
- [ ] Tracks equipment capacity (MW thermal) and efficiency curves
- [ ] Monitors warm-up and cool-down times for each equipment type
- [ ] Tracks thermal storage status (molten salt tanks, steam accumulators, hot water storage)
- [ ] Calculates available thermal storage capacity in MWh
- [ ] Tracks equipment runtime and schedules predictive maintenance windows
- [ ] Supports equipment grouping (e.g., Furnace Bank A = Furnaces 1-5)
- [ ] Handles equipment constraints (max cycles per day, minimum runtime)
- [ ] Forecasts equipment availability based on maintenance schedule

**Equipment Data Model:**

```python
class HeatingEquipment:
    equipment_id: str              # e.g., "FURNACE-001"
    equipment_type: EquipmentType  # FURNACE, KILN, DRYER, OVEN, BOILER
    capacity_mw: float             # Thermal capacity in MW
    efficiency_curve: dict         # Efficiency vs. load curve
    warm_up_time_min: int          # Time to reach operating temperature
    cool_down_time_min: int        # Time to safe shutdown temperature
    min_runtime_min: int           # Minimum runtime once started
    max_cycles_per_day: int        # Maximum start-stop cycles
    current_status: EquipmentStatus
    current_load_pct: float        # Current load as % of capacity
    current_temperature_c: float   # Current operating temperature
    maintenance_windows: list      # Scheduled maintenance periods

class ThermalStorage:
    storage_id: str                # e.g., "MOLTEN-SALT-001"
    storage_type: StorageType      # MOLTEN_SALT, HOT_WATER, STEAM_ACCUMULATOR
    capacity_mwh: float            # Total storage capacity
    current_charge_pct: float      # Current state of charge (0-100%)
    charge_rate_mw: float          # Maximum charge rate
    discharge_rate_mw: float       # Maximum discharge rate
    thermal_losses_pct_hr: float   # Heat loss per hour (% of stored energy)
    min_charge_pct: float          # Minimum operating charge level
```

**Non-Functional Requirements:**
- Performance: <2 second status update latency from SCADA
- Reliability: Graceful degradation if SCADA connection lost (use last known status)
- Accuracy: Equipment capacity within 5% of actual measured capacity

**Estimated Effort:** 3 weeks (1 industrial controls engineer, 1 backend engineer)

**Business Justification:** Accurate equipment tracking prevents schedule conflicts, reduces unplanned downtime, and enables thermal storage optimization.

---

#### Feature 4: Heating Schedule Optimization Engine

**User Story:**
```
As a plant manager,
I want HEATSCHEDULER to generate an optimized heating schedule that minimizes energy cost while meeting all production deadlines,
So that I can reduce energy spend by 15%+ without impacting production output.
```

**Acceptance Criteria:**
- [ ] Generates optimal heating schedule for planning horizon (1-7 days)
- [ ] Minimizes total energy cost (energy charges + demand charges)
- [ ] Respects all production deadline constraints (no late batches)
- [ ] Respects equipment capacity and availability constraints
- [ ] Coordinates multiple heating systems to avoid demand spikes
- [ ] Leverages thermal storage for load shifting when available
- [ ] Considers warm-up/cool-down times and sequencing constraints
- [ ] Supports multiple optimization objectives (cost, emissions, load factor)
- [ ] Generates schedule in <5 minutes for 100+ equipment and 1,000+ batches
- [ ] Provides optimization summary: expected savings, peak demand reduction, schedule changes

**Optimization Algorithm:**

```
Mixed-Integer Linear Programming (MILP) Formulation:

MINIMIZE:
  Total_Cost = SUM(Energy_Rate[t] × Load[t]) + Demand_Charge × Peak_Load
                + SUM(Startup_Cost[e] × Startup[e,t])

SUBJECT TO:
  # Production deadline constraints
  Completion_Time[b] <= Deadline[b]  for all batches b

  # Equipment capacity constraints
  Load[e,t] <= Capacity[e] × Status[e,t]  for all equipment e, time t

  # Minimum runtime constraints
  Runtime[e] >= Min_Runtime[e]  if Started[e]

  # Demand limit constraints (prevent spikes)
  SUM(Load[e,t]) <= Demand_Limit  for all time t

  # Thermal storage constraints
  Storage_Level[s,t+1] = Storage_Level[s,t] × (1 - Loss_Rate)
                       + Charge[s,t] - Discharge[s,t]
  Storage_Level[s,t] >= Min_Level[s]
  Storage_Level[s,t] <= Max_Level[s]

  # Sequencing constraints
  Start_Time[batch_B] >= End_Time[batch_A]  if A precedes B

Variables:
  - Load[e,t]: Load on equipment e at time t (continuous, 0 to Capacity)
  - Start[e,t]: Binary, 1 if equipment e starts at time t
  - Status[e,t]: Binary, 1 if equipment e is running at time t
  - Charge[s,t]: Thermal storage charge rate (continuous)
  - Discharge[s,t]: Thermal storage discharge rate (continuous)

Solver: Gurobi, CPLEX, or OR-Tools (open source)
```

**Edge Cases:**
- Infeasible schedule (deadlines too tight) -> Return infeasibility report with constraint violations
- Multiple optimal solutions -> Choose schedule with least change from current
- Rolling horizon -> Maintain schedule continuity across planning windows
- Emergency production changes -> Re-optimize with 5-minute turnaround

**Non-Functional Requirements:**
- Performance: <5 minutes for weekly schedule (100 equipment, 1,000 batches)
- Accuracy: Actual savings within 5% of predicted savings
- Determinism: Same input always produces same output (reproducible)
- Auditability: Full optimization log with constraint explanations

**Estimated Effort:** 6 weeks (1 optimization engineer, 1 backend engineer, 1 domain expert)

**Business Justification:** Core value proposition. 15% energy cost reduction on $10M annual energy spend = $1.5M savings/year.

---

#### Feature 5: Load Shifting Recommendations

**User Story:**
```
As a production planner,
I want HEATSCHEDULER to recommend specific heating operations that can be shifted to off-peak hours,
So that I can make informed decisions about schedule changes that reduce energy costs.
```

**Acceptance Criteria:**
- [ ] Identifies heating operations with schedule flexibility (slack time before deadline)
- [ ] Calculates potential savings for shifting each operation to off-peak hours
- [ ] Ranks recommendations by savings potential ($/hour shifted)
- [ ] Considers warm-up time and efficiency impacts of shifting
- [ ] Visualizes current vs. recommended schedule on timeline chart
- [ ] Provides one-click acceptance of recommendations
- [ ] Tracks recommendation acceptance rate and actual savings achieved
- [ ] Learns from accepted/rejected recommendations to improve future suggestions
- [ ] Explains reasoning for each recommendation ("Why shift this operation?")

**Recommendation Logic:**

```python
def generate_load_shift_recommendations(schedule, tariffs, equipment):
    recommendations = []

    for operation in schedule.operations:
        # Calculate current cost
        current_cost = calculate_energy_cost(operation, tariffs)

        # Find feasible off-peak windows
        off_peak_windows = find_off_peak_windows(
            earliest_start=operation.earliest_start,
            latest_end=operation.deadline,
            duration=operation.duration + equipment.warm_up_time,
            tariffs=tariffs
        )

        for window in off_peak_windows:
            # Calculate cost if shifted
            shifted_cost = calculate_energy_cost_in_window(operation, window, tariffs)
            savings = current_cost - shifted_cost

            if savings > MIN_SAVINGS_THRESHOLD:
                recommendations.append(LoadShiftRecommendation(
                    operation=operation,
                    current_window=(operation.start, operation.end),
                    recommended_window=window,
                    savings_usd=savings,
                    confidence=calculate_confidence(operation, window),
                    explanation=generate_explanation(operation, savings, tariffs)
                ))

    return sorted(recommendations, key=lambda r: r.savings_usd, reverse=True)
```

**Non-Functional Requirements:**
- Performance: <30 seconds to generate recommendations for weekly schedule
- Usability: Clear, actionable recommendations with savings quantified
- Accuracy: 90% of implemented recommendations achieve predicted savings

**Estimated Effort:** 3 weeks (1 optimization engineer, 1 frontend engineer)

**Business Justification:** Provides transparency into optimization decisions and enables gradual adoption (recommendations before automation).

---

#### Feature 6: Demand Response Integration

**User Story:**
```
As an energy manager,
I want HEATSCHEDULER to automatically participate in utility demand response programs,
So that I can earn $50K-$500K per year in DR payments while protecting production deadlines.
```

**Acceptance Criteria:**
- [ ] Integrates with utility DR program APIs (OpenADR 2.0b standard)
- [ ] Receives DR event notifications (day-ahead, hour-ahead, real-time)
- [ ] Calculates available curtailment capacity based on schedule flexibility
- [ ] Automatically generates curtailment bids that protect production deadlines
- [ ] Implements curtailment during DR events (reduce/shift heating loads)
- [ ] Tracks curtailment performance and baseline calculations
- [ ] Submits performance data to utility for settlement
- [ ] Calculates DR revenue earned (per event and cumulative)
- [ ] Supports multiple DR program types:
  - Capacity programs (availability payments + performance payments)
  - Energy programs (payment per kWh curtailed)
  - Ancillary services (frequency regulation, spinning reserve)
- [ ] Provides DR event calendar and forecast

**DR Event Response Flow:**

```
1. Receive DR Event Notification (OpenADR)
   - Event type: LOAD_CURTAILMENT
   - Start time: 2025-07-15 14:00
   - Duration: 4 hours
   - Target curtailment: 10 MW

2. Calculate Available Curtailment
   - Review current heating schedule
   - Identify flexible loads (not deadline-critical)
   - Calculate max curtailment without production impact: 12 MW

3. Submit Curtailment Bid
   - Offered curtailment: 10 MW (matches target)
   - Duration: 4 hours
   - Protected operations: Batch XYZ (deadline 16:00, cannot be interrupted)

4. Execute Curtailment
   - At 14:00, automatically reduce/shift heating loads
   - Monitor actual curtailment vs. committed
   - Maintain production-critical operations

5. Report Performance
   - Submit metered data to utility
   - Calculate baseline and actual load
   - Achieved curtailment: 9.5 MW (95% of committed)

6. Settlement
   - Revenue earned: $4,750 (9.5 MW × 4 hrs × $125/MWh)
```

**Non-Functional Requirements:**
- Latency: <1 minute response to real-time DR events
- Accuracy: Achieve 95%+ of committed curtailment
- Compliance: Full OpenADR 2.0b certification
- Reliability: 99.9% availability for DR events

**Estimated Effort:** 4 weeks (1 DR specialist, 1 integration engineer)

**Business Justification:** DR revenue of $50-$500K/year per facility provides additional ROI beyond energy cost savings.

---

#### Feature 7: Cost Savings Forecasting

**User Story:**
```
As a plant manager,
I want HEATSCHEDULER to forecast expected cost savings from optimized schedules,
So that I can justify the investment to leadership and track ROI over time.
```

**Acceptance Criteria:**
- [ ] Calculates baseline energy cost (current/historical scheduling approach)
- [ ] Calculates optimized energy cost (HEATSCHEDULER schedule)
- [ ] Shows savings breakdown: energy charges, demand charges, DR revenue
- [ ] Forecasts weekly, monthly, quarterly, and annual savings
- [ ] Tracks actual savings achieved vs. forecast (variance analysis)
- [ ] Adjusts forecast based on actual performance (learning)
- [ ] Provides confidence intervals for savings forecasts
- [ ] Generates executive summary reports (PDF, Excel)
- [ ] Calculates ROI and payback period for HEATSCHEDULER investment
- [ ] Compares facility performance to industry benchmarks

**Savings Calculation:**

```
Baseline Cost (without HEATSCHEDULER):
- Use historical energy consumption patterns
- Apply current tariff structure
- No load shifting optimization

Optimized Cost (with HEATSCHEDULER):
- Apply optimized schedule
- Calculate reduced peak demand
- Include DR revenue

Savings = Baseline_Cost - Optimized_Cost + DR_Revenue

Savings Breakdown:
- Energy charge savings: $(X) (shifted to off-peak hours)
- Demand charge savings: $(Y) (reduced peak demand by Z MW)
- DR revenue: $(W) (N events, M MW average curtailment)

Annualized Savings = Monthly_Savings × 12 × Confidence_Factor

ROI = (Annual_Savings - Annual_License_Cost) / Implementation_Cost × 100%
Payback_Months = Implementation_Cost / (Monthly_Savings - Monthly_License_Cost)
```

**Non-Functional Requirements:**
- Accuracy: Forecast within 10% of actual savings (after 3-month calibration)
- Reporting: Automated monthly reports to stakeholders
- Visualization: Executive dashboards with trend charts

**Estimated Effort:** 3 weeks (1 data analyst, 1 frontend engineer)

**Business Justification:** Clear ROI tracking is critical for customer retention and upselling to additional facilities.

---

### Should-Have Features (P1 - High Priority, Post-MVP)

---

#### Feature 8: What-If Scenario Analysis

**User Story:**
```
As a production planner,
I want to simulate different schedule scenarios and see the energy cost impact,
So that I can make informed decisions about production planning trade-offs.
```

**Acceptance Criteria:**
- [ ] Creates scenario copies of current schedule for modification
- [ ] Allows adding, removing, or moving production batches in scenario
- [ ] Allows modifying equipment availability in scenario
- [ ] Allows testing different tariff structures (rate change impact)
- [ ] Calculates energy cost for each scenario in real-time
- [ ] Compares multiple scenarios side-by-side
- [ ] Shows delta (cost difference, peak demand change) between scenarios
- [ ] Allows promoting scenario to production schedule (one-click)
- [ ] Saves scenarios for future reference
- [ ] Supports scenario templates (e.g., "Holiday schedule", "Maintenance outage")

**Estimated Effort:** 4 weeks (1 backend engineer, 1 frontend engineer)

**Business Justification:** Empowers production planners to make energy-informed decisions without waiting for formal optimization runs.

---

#### Feature 9: Carbon Emissions Tracking and Optimization

**User Story:**
```
As a sustainability manager,
I want HEATSCHEDULER to track carbon emissions from heating operations and optimize schedules for emissions reduction,
So that I can meet corporate carbon reduction targets and report Scope 1/2 emissions.
```

**Acceptance Criteria:**
- [ ] Calculates CO2 emissions from heating operations (tonnes CO2e)
- [ ] Uses grid carbon intensity data (real-time where available)
- [ ] Supports multi-objective optimization (cost + emissions)
- [ ] Generates emissions reports for sustainability reporting (GHG Protocol)
- [ ] Tracks emissions reduction achieved through schedule optimization
- [ ] Forecasts carbon cost impact (carbon tax, ETS)

**Estimated Effort:** 3 weeks (1 sustainability domain expert, 1 backend engineer)

---

#### Feature 10: Mobile Alerts and Approval Workflow

**User Story:**
```
As a plant manager,
I want to receive mobile alerts for schedule changes and approve recommendations on my phone,
So that I can stay informed and make decisions without being at my desk.
```

**Acceptance Criteria:**
- [ ] Mobile app (iOS, Android) for schedule monitoring
- [ ] Push notifications for schedule changes, DR events, optimization opportunities
- [ ] Approval workflow for schedule modifications above threshold
- [ ] Dashboard widgets for key metrics (energy cost, savings, compliance)
- [ ] Offline mode with sync when connectivity restored

**Estimated Effort:** 6 weeks (1 mobile developer, 1 backend engineer)

---

### Could-Have Features (P2 - Nice to Have, Future Roadmap)

#### Feature 11: Predictive Production Scheduling

**User Story:**
```
As a production planner,
I want HEATSCHEDULER to predict production needs based on historical patterns and automatically generate optimized schedules,
So that I can move from reactive to proactive planning.
```

**Scope:** ML-based production demand forecasting with automated schedule generation

**Target:** Version 2.0 (Q4 2026)

---

#### Feature 12: Multi-Site Fleet Optimization

**User Story:**
```
As a corporate energy manager,
I want to optimize heating schedules across all facilities in my portfolio,
So that I can balance production across sites based on energy costs and maximize corporate-wide savings.
```

**Scope:** Cross-facility load balancing and corporate energy procurement optimization

**Target:** Version 2.0 (Q4 2026)

---

#### Feature 13: Renewable Energy Integration

**User Story:**
```
As a sustainability manager,
I want HEATSCHEDULER to schedule heating operations to coincide with on-site solar/wind generation,
So that I can maximize self-consumption and minimize grid purchases.
```

**Scope:** On-site generation forecasting and schedule optimization for renewable self-consumption

**Target:** Version 1.2 (Q3 2026)

---

### Won't-Have Features (P3 - Out of Scope)

- HVAC optimization for building comfort -> Covered by building management systems
- Process control and temperature setpoint optimization -> Covered by DCS/PLC
- Energy procurement and hedging -> Covered by energy trading platforms
- Equipment maintenance optimization -> Covered by GL-013 PREDICTMAINT
- Steam system optimization -> Covered by GL-020 STEAMBALANCE
- Mobile-first design -> Web-first for MVP, mobile as P1 feature

---

## 4. Regulatory & Compliance Requirements

### Regulatory Drivers

| Regulation | Effective Date | Scope | Penalty for Non-Compliance |
|------------|---------------|-------|----------------------------|
| ISO 50001 (Energy Management) | Voluntary | Global energy management certification | N/A (certification requirement) |
| EPA Energy Star | Voluntary | US industrial facilities | N/A (benchmark and recognition) |
| EU Energy Efficiency Directive | 2024 | Large enterprises in EU | Audit requirements, fines vary by country |
| FERC Order 2222 | 2022 | US wholesale markets | N/A (enables DER participation) |
| California Title 24 | Ongoing | California commercial/industrial | Building permit denial |
| OSHA 29 CFR 1910 | Ongoing | Occupational safety (US) | $15,625 per violation |

### Compliance Mapping

| Regulatory Requirement | HEATSCHEDULER Feature | Validation Method |
|------------------------|----------------------|-------------------|
| ISO 50001 energy performance monitoring | Cost Savings Forecasting, Equipment Tracking | ISO 50001 audit checklist validation |
| ISO 50001 energy baseline establishment | Baseline Cost Calculation | Compare to certified baseline methodology |
| FERC 2222 demand response participation | Demand Response Integration | OpenADR 2.0b certification |
| Energy audit documentation | Schedule Optimization Logs | 7-year data retention, export to PDF/Excel |
| Production schedule traceability | Production Schedule Integration | Full audit trail with timestamps |
| Occupational safety (no schedule conflicts) | Equipment Availability Tracking | Conflict detection with safety margins |

### Data Retention Requirements

- **Production Schedules:** 7 years (ISO 50001, SOX compliance)
- **Energy Consumption Data:** 7 years (regulatory audits)
- **Optimization Decisions:** 5 years (auditability)
- **DR Event Records:** 5 years (utility settlement disputes)
- **Equipment Status History:** 3 years (maintenance analysis)

---

## 5. User Experience & Workflows

### Primary User Personas

1. **Production Planner (Primary User)**
   - Technical expertise: Medium (production systems, scheduling)
   - Frequency: Daily use (1-2 hours)
   - Key workflows: Import production schedule, review optimization recommendations, approve/modify schedule changes
   - Top needs: Integration with ERP/MES, clear schedule visualization, minimal disruption to production
   - Pain points: Schedule changes are disruptive, energy costs not visible, conflicting priorities

2. **Energy Manager (Power User)**
   - Technical expertise: High (energy systems, tariffs, demand response)
   - Frequency: Daily monitoring, weekly deep dives
   - Key workflows: Configure tariffs, monitor energy costs, manage DR participation, report savings
   - Top needs: Accurate tariff modeling, real-time cost visibility, DR program integration
   - Pain points: Complex tariff structures, manual DR participation, difficulty proving savings

3. **Plant Manager (Oversight User)**
   - Technical expertise: Medium (operations management)
   - Frequency: Weekly reviews, monthly reports
   - Key workflows: Review KPIs, approve major schedule changes, report to leadership
   - Top needs: Executive dashboards, ROI tracking, exception alerts
   - Pain points: Limited visibility into energy costs, difficulty justifying investments

4. **Maintenance Engineer (Occasional User)**
   - Technical expertise: High (equipment, maintenance)
   - Frequency: As needed (equipment status updates, maintenance scheduling)
   - Key workflows: Update equipment availability, review maintenance windows, coordinate with production
   - Top needs: Clear maintenance windows, equipment status visibility
   - Pain points: Production pressure to minimize maintenance downtime

### Key User Flows

#### Flow 1: Weekly Schedule Optimization (Production Planner)

```
User logs into HEATSCHEDULER web app (Monday 7:00 AM)
  |
Dashboard shows last week's performance: 12% savings achieved, 100% on-time
  |
User clicks "Generate Weekly Schedule" for Week of July 21
  |
System imports production schedule from SAP (1,247 batches)
  |
System imports equipment availability from SCADA (45 units)
  |
System imports current tariff structure and DR forecast
  |
Optimization runs (<5 minutes)
  |
User reviews optimization summary:
  - Baseline cost: $892,450
  - Optimized cost: $758,583
  - Savings: $133,867 (15.0%)
  - Peak demand reduced: 12 MW (18%)
  - All 1,247 batches on schedule: YES
  |
User reviews load shift recommendations (top 10)
  |
User accepts 8 recommendations, modifies 2
  |
User clicks "Publish Schedule"
  |
Schedule is sent to MES and displayed on shop floor dashboards
```

**Time to complete:** <15 minutes

---

#### Flow 2: Demand Response Event (Energy Manager)

```
HEATSCHEDULER receives DR event notification (2:00 PM, Thursday)
  - Event: Peak shaving, 3:00 PM - 7:00 PM
  - Target curtailment: 8 MW
  |
System automatically calculates available curtailment
  - Current heating load: 35 MW
  - Flexible load: 12 MW
  - Protected load (deadline-critical): 23 MW
  |
System generates curtailment plan:
  - Furnace Bank B: Reduce 5 MW (batches can shift 2 hours)
  - Dryers 7-10: Reduce 3 MW (batches have 4-hour slack)
  - Total offered: 8 MW (matches target)
  |
Energy Manager receives mobile notification
  |
Energy Manager reviews curtailment plan in app
  |
Energy Manager approves plan (or modifies if needed)
  |
At 3:00 PM, HEATSCHEDULER automatically reduces loads
  |
System monitors actual curtailment: 7.8 MW achieved (98%)
  |
At 7:00 PM, loads are restored automatically
  |
System reports performance to utility via OpenADR
  |
Dashboard shows DR revenue earned: $3,120 (7.8 MW × 4 hrs × $100/MWh)
```

**Time to respond:** <10 minutes

---

#### Flow 3: What-If Analysis (Production Planner)

```
User navigates to "Scenario Analysis" section
  |
User creates new scenario: "Rush Order - 50 additional batches"
  |
User adds 50 high-priority batches with 48-hour deadline
  |
System checks feasibility:
  - Equipment capacity: SUFFICIENT
  - Production conflicts: 3 batches need rescheduling
  - Energy impact: +$42,000 this week (+5.5%)
  |
User reviews conflict resolution options:
  Option A: Shift 3 existing batches to next week (-$8,000 savings)
  Option B: Run overtime shifts (-$15,000 labor + energy)
  Option C: Outsource 15 batches (-$25,000 outsource cost)
  |
User selects Option A
  |
User clicks "Promote to Production Schedule"
  |
Schedule is updated and published to MES
```

**Time to complete:** <10 minutes

---

### Wireframes

*[Wireframes would be created by UX designer - key screens include:]*

1. **Dashboard:** Real-time energy cost, schedule compliance, savings tracker
2. **Schedule View:** Gantt chart of heating operations with equipment lanes, color-coded by production batch and tariff period
3. **Optimization Summary:** Before/after comparison, savings breakdown, peak demand chart
4. **Load Shift Recommendations:** Ranked list with savings potential, accept/reject actions
5. **Demand Response:** DR event calendar, curtailment capacity, revenue tracking
6. **Equipment Status:** Real-time equipment availability, maintenance windows
7. **Tariff Configuration:** Time-of-use periods, demand charges, real-time pricing
8. **Scenario Analysis:** Side-by-side scenario comparison, cost impact
9. **Reports:** Executive summary, detailed schedule, savings report

---

## 6. Success Criteria & KPIs

### Launch Criteria (Go/No-Go Decision)

- [ ] All P0 features implemented and tested
- [ ] 85%+ unit test coverage achieved
- [ ] Integration tested with at least 2 ERP systems (SAP, Oracle)
- [ ] Integration tested with at least 2 SCADA/DCS platforms (Rockwell, Siemens)
- [ ] Optimization accuracy validated (savings within 5% of actual)
- [ ] Security audit passed (Grade A score, no critical vulnerabilities)
- [ ] Performance targets met:
  - <5 minute schedule optimization for 100 equipment, 1,000 batches
  - <5 second production schedule sync
  - <1 minute DR event response
  - 99.9% uptime (30-day beta test)
- [ ] Regulatory compliance validated:
  - OpenADR 2.0b certification for DR integration
  - ISO 50001 audit compatibility verified
- [ ] Documentation complete:
  - API documentation (OpenAPI spec)
  - User guide (production planners, energy managers)
  - Administrator manual (tariff configuration, integrations)
- [ ] 10 beta customers successfully using the product (>=30 days each)
- [ ] No critical or high-severity bugs in backlog
- [ ] Customer satisfaction (beta) NPS >40

### Post-Launch Metrics

**30 Days:**
- 25 facilities live
- 50,000 schedule optimizations performed
- Average energy cost reduction: 10%
- 99% production deadline compliance
- <10 support tickets per customer
- 99.9% uptime achieved
- NPS >40

**60 Days:**
- 75 facilities live
- 150,000 schedule optimizations performed
- Average energy cost reduction: 12%
- 99.5% production deadline compliance
- Average demand charge reduction: 20%
- <5 support tickets per customer
- NPS >50

**90 Days:**
- 150 facilities live
- 400,000 schedule optimizations performed
- Average energy cost reduction: 15%
- 99% production deadline compliance
- Average demand charge reduction: 25%
- $50K average DR revenue per facility (annualized)
- 99.95% uptime
- NPS >55
- $15M ARR achieved

**Year 1 (12 Months):**
- 200 facilities live
- 2M+ schedule optimizations performed
- Average energy cost reduction: 15% sustained
- Average annual savings: $450K per facility
- Total customer savings: $90M/year
- Revenue: $80M ARR ($400K per facility/year)
- Customer retention: >95%
- NPS >60

---

## 7. Roadmap & Milestones

### Phase 1: MVP (Weeks 1-20, Q3 2025 - Q1 2026)

**Week 1-4: Requirements & Architecture**
- Finalize PRD (this document)
- Design system architecture (microservices, event-driven)
- Set up development environment (K8s, CI/CD)
- Establish development standards and coding guidelines
- Begin ERP integration design (SAP, Oracle)

**Week 5-10: Core Development - Data Integration**
- Implement production schedule integration (SAP PP/DS, Oracle SCM)
- Build energy tariff management module (TOU, demand charges, RTP)
- Develop equipment availability tracking (SCADA/OPC UA integration)
- Create data validation and quality scoring engine

**Week 11-16: Core Development - Optimization Engine**
- Build heating schedule optimization engine (MILP solver)
- Implement load shifting recommendation engine
- Develop demand response integration (OpenADR 2.0b)
- Create cost savings forecasting module

**Week 17-18: Integration & Testing**
- End-to-end integration testing
- Performance testing (1,000+ batches, 100+ equipment)
- Security audit and penetration testing
- Load testing for concurrent users

**Week 19-20: Beta & Launch Prep**
- Beta customer onboarding (10 facilities)
- User training and documentation
- Beta feedback collection and bug fixes
- Production deployment preparation
- Marketing and sales enablement

**Milestone: MVP Launch (Week 20, Q1 2026)**

---

### Phase 2: Enhancements (Weeks 21-36, Q2-Q3 2026)

**Q2 2026:**
- What-if scenario analysis
- Carbon emissions tracking and optimization
- Mobile alerts and approval workflow
- Advanced analytics dashboards
- Multi-fuel support (electricity + natural gas co-optimization)

**Q3 2026:**
- Renewable energy integration (solar/wind self-consumption)
- Thermal storage optimization (advanced algorithms)
- Weather-based load forecasting
- Integration with additional ERP/MES systems
- Enhanced reporting (custom report builder)

**Milestone: Phase 2 Launch (Week 36, Q3 2026) - Target: Q2 2026 deadline met**

---

### Phase 3: Scale & Intelligence (Weeks 37-52, Q4 2026 - Q1 2027)

**Q4 2026:**
- Multi-site fleet optimization
- Predictive production scheduling (ML-based)
- Energy market participation (wholesale markets)
- Digital twin integration
- Edge computing deployment option

**Q1 2027:**
- Autonomous scheduling (minimal human intervention)
- Blockchain-based energy tracking
- Carbon credit integration
- Global regulatory compliance (EU, Asia-Pacific)
- IoT sensor mesh integration

**Milestone: Enterprise Ready (Q1 2027)**

---

## 8. Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **ERP integration complexity** | High | High | - Prioritize SAP (60% market share) for MVP<br>- Build generic REST/CSV fallback<br>- Partner with ERP implementation firms<br>- Offer professional services for custom integrations |
| **Production planner resistance to change** | Medium | High | - Advisory mode first (recommendations only)<br>- Clear savings visualization<br>- Gradual automation with human approval<br>- Training and change management support |
| **Inaccurate tariff modeling** | Medium | High | - Validate against actual utility bills<br>- Partner with utility for rate verification<br>- Monthly tariff accuracy audits |
| **Optimization solver scalability** | Low | High | - Commercial solver (Gurobi) with guaranteed performance<br>- Problem decomposition for large instances<br>- Cloud-based compute scaling |
| **DR program rule complexity** | Medium | Medium | - Start with major ISOs (PJM, CAISO, ERCOT)<br>- Partner with DR aggregators for smaller programs<br>- OpenADR certification for compliance |
| **Customer cybersecurity requirements** | High | Medium | - SOC 2 Type II certification<br>- On-premise deployment option<br>- Air-gapped integration option for sensitive facilities |
| **Regulatory changes (tariff structures)** | Medium | Medium | - Modular tariff configuration<br>- Automated tariff import from OpenEI<br>- Quarterly tariff review process |
| **Competitive response (EMS vendors)** | Medium | Medium | - Production integration as key differentiator<br>- Focus on manufacturing verticals (not buildings)<br>- Partner with complementary EMS vendors |
| **Low adoption of DR participation** | Medium | Medium | - DR revenue sharing model (60/40 split)<br>- Turnkey DR enrollment assistance<br>- Case studies showing DR revenue potential |
| **Equipment data quality issues** | High | Medium | - Data quality scoring with user alerts<br>- Manual override capability<br>- Historical pattern detection for outliers |

---

## 9. Go-to-Market Strategy

### Pricing Model

**Tiered Subscription (Annual):**

| Tier | Facilities | Price per Facility/Year | Total Annual Price | Target Customer |
|------|------------|------------------------|-------------------|-----------------|
| **Starter** | 1-2 facilities | $300,000 | $300K-$600K | Single-site manufacturers, pilot projects |
| **Professional** | 3-10 facilities | $250,000 | $750K-$2.5M | Multi-site manufacturers |
| **Enterprise** | 11-50 facilities | $200,000 | $2.2M-$10M | Large industrial corporations |
| **Enterprise Plus** | 50+ facilities | Custom (volume discount) | $10M+ | Global manufacturers, refineries |

**Pricing Justification:**
- Average energy cost reduction: 15% = $450K/year on $3M annual energy spend
- Average DR revenue: $100K/year
- Total customer value: $550K/year
- ROI: 1.4:1 to 1.8:1 (depending on tier)
- Payback period: 7-12 months

**Additional Revenue Streams:**
- Professional services (integration, training): $50K-$150K per project
- Premium support (24/7, <1 hr response): +25% of subscription
- Custom development (multi-site optimization, etc.): $100K-$300K per feature
- DR revenue sharing: 20% of DR payments (for managed DR service)
- Managed services (GreenLang operates the system): +50% of subscription

### Sales Strategy

**Year 1 Target: $80M ARR (200 facilities)**

**Phase 1 (Months 1-6): Early Adopters**
- Target: 10 beta customers -> 50 paying customers
- Focus: Food processing, automotive manufacturing (high energy costs, flexible schedules)
- Sales motion: Direct sales (GreenLang sales team)
- Deal size: $300K-$600K (1-2 facilities)
- Key message: "15% energy cost reduction with 99% production compliance"

**Phase 2 (Months 7-12): Expansion**
- Target: 50 -> 200 facilities
- Focus: Chemical plants, steel mills, glass manufacturers
- Sales motion: Direct + channel partners (energy consultants, ERP implementers)
- Deal size: $750K-$2.5M (3-10 facilities)
- Key message: "Proven ROI at [Customer X], now available enterprise-wide"

**Phase 3 (Year 2+): Scale**
- Target: 200 -> 1,000 facilities
- Focus: Global expansion (EU, Asia-Pacific), oil & gas refineries
- Sales motion: Channel-led (70% of deals)
- Deal size: $2M-$10M (multi-site enterprises)
- Key message: "Corporate-wide energy optimization platform"

### Marketing Channels

1. **Trade Shows & Conferences:**
   - Hannover Messe (April 2026) - Industrial automation
   - DistribuTECH (February 2026) - Energy management
   - Food Processing Suppliers Association (October 2025)
   - AIChE Annual Meeting (November 2025) - Chemical engineering

2. **Content Marketing:**
   - White paper: "The Hidden Cost of Peak Demand: How Manufacturers Leave $500K on the Table"
   - Case studies: "15% Energy Cost Reduction at [Food Processor Y]"
   - Webinars: "Demand Response for Manufacturers 101"
   - Blog: Weekly posts on energy optimization best practices

3. **Partnerships:**
   - ERP vendors (SAP, Oracle): OEM agreements, marketplace listings
   - Energy consultants: Referral partnerships (10% commission)
   - Utility DR programs: Preferred vendor status
   - MES vendors (Rockwell, Siemens): Integration partnerships

4. **Digital:**
   - SEO: "industrial energy optimization", "demand charge reduction", "manufacturing energy management"
   - LinkedIn ads targeting plant managers, energy managers
   - YouTube: Product demos, customer testimonials, thought leadership
   - Targeted account-based marketing (ABM) for Fortune 500 manufacturers

---

## 10. Technical Architecture (High-Level)

### System Architecture

```
+------------------------------------------------------------------+
|                      HEATSCHEDULER GL-019                        |
+------------------------------------------------------------------+
                                |
                +---------------+---------------+
                |                               |
          +-----v-----+                   +-----v-----+
          | Data      |                   | Data      |
          | Sources   |                   | Sinks     |
          +-----------+                   +-----------+
                |                               |
    +-----------+-----------+         +---------+---------+
    |           |           |         |         |         |
+---v---+  +----v----+  +---v---+  +--v--+  +--v--+  +---v---+
|ERP/MES|  |SCADA/   |  |Utility|  |MES  |  |CMMS |  |BI/    |
|(SAP,  |  |DCS      |  |APIs   |  |     |  |     |  |Reports|
|Oracle)|  |(Rockwell|  |(DR,   |  |     |  |     |  |       |
|       |  |Siemens) |  |Tariff)|  |     |  |     |  |       |
+-------+  +---------+  +-------+  +-----+  +-----+  +-------+
  (REST)    (OPC UA)     (REST)    (REST)   (REST)    (SQL)

                +---------------------------+
                |   Core Engine (Python)    |
                +---------------------------+
                | - Schedule Optimizer      |
                |   (MILP: Gurobi/OR-Tools) |
                | - Tariff Calculator       |
                | - Equipment Tracker       |
                | - DR Manager              |
                | - Savings Forecaster      |
                | - Provenance Tracker      |
                +---------------------------+
                           |
                +----------+----------+
                |                     |
          +-----v-----+         +-----v-----+
          | FastAPI   |         | Redis     |
          | Web Server|         | Cache     |
          +-----------+         +-----------+
                |
          +-----v-----+
          | React     |
          | Frontend  |
          +-----------+
```

### Tech Stack

- **Language:** Python 3.11
- **Web Framework:** FastAPI (async, high performance)
- **Frontend:** React 18, TypeScript, TailwindCSS
- **Optimization Engine:** Gurobi (commercial) or OR-Tools (open source), PuLP
- **Industrial Protocols:** asyncua (OPC UA), pymodbus (Modbus TCP)
- **Database:** PostgreSQL (relational), TimescaleDB (time-series)
- **Cache:** Redis (schedule cache, real-time pricing)
- **Messaging:** RabbitMQ (event-driven architecture)
- **DR Protocol:** OpenADR 2.0b (OpenLeadr library)
- **Monitoring:** Prometheus + Grafana
- **Logging:** Structured logging (JSON), centralized (ELK stack)
- **Deployment:** Kubernetes, Docker
- **CI/CD:** GitHub Actions, ArgoCD

### Integration Architecture

**Inbound Integrations (Data Sources):**
- SAP PP/DS: RFC/BAPI connector
- Oracle SCM: REST API
- Rockwell FactoryTalk: OPC UA
- Siemens SIMATIC: OPC UA
- Utility DR Programs: OpenADR 2.0b
- ISO Markets: REST APIs (PJM, CAISO, ERCOT)

**Outbound Integrations (Data Sinks):**
- MES: REST API (schedule publishing)
- CMMS: REST API (maintenance coordination)
- BI Tools: SQL/ODBC (reporting)
- Email/SMS: Notification service

---

## 11. Appendices

### Appendix A: Glossary

- **Demand Charge:** Utility charge based on peak kW demand during billing period (typically 15-min average)
- **DR (Demand Response):** Programs where customers reduce load in response to grid signals for payment
- **ERP:** Enterprise Resource Planning (SAP, Oracle)
- **ISO:** Independent System Operator (manages wholesale electricity markets)
- **LMP:** Locational Marginal Price (real-time electricity price at specific grid location)
- **MES:** Manufacturing Execution System (shop floor control)
- **MILP:** Mixed-Integer Linear Programming (optimization technique)
- **OPC UA:** Open Platform Communications Unified Architecture (industrial protocol)
- **OpenADR:** Open Automated Demand Response (DR communication standard)
- **RTP:** Real-Time Pricing (electricity rates that vary hourly)
- **SCADA:** Supervisory Control and Data Acquisition
- **TOU:** Time-of-Use (electricity rates that vary by time period)

### Appendix B: References

- ISO 50001:2018 Energy Management Systems: https://www.iso.org/iso-50001-energy-management.html
- OpenADR 2.0b Specification: https://www.openadr.org/specification
- FERC Order 2222: https://www.ferc.gov/media/ferc-order-no-2222-fact-sheet
- PJM Demand Response: https://www.pjm.com/markets-and-operations/demand-response
- CAISO Demand Response: http://www.caiso.com/participate/Pages/DemandResponse/Default.aspx
- ERCOT Demand Response: https://www.ercot.com/services/programs/load

### Appendix C: Competitive Product Comparison

| Feature | HEATSCHEDULER | Schneider EcoStruxure | Siemens Navigator | Manual (Excel) |
|---------|--------------|---------------------|------------------|----------------|
| Production schedule integration | Native ERP/MES | Limited | Limited | Manual entry |
| Time-of-use optimization | Full TOU + demand | Basic TOU | Basic TOU | Manual |
| Demand charge optimization | Advanced MILP | Rule-based | Rule-based | N/A |
| Real-time pricing | Native ISO integration | Add-on | Add-on | N/A |
| Demand response automation | OpenADR certified | Manual | Manual | N/A |
| Thermal storage optimization | Native | N/A | Limited | N/A |
| What-if analysis | Unlimited scenarios | Limited | Limited | Manual |
| Multi-site optimization | P2 feature | Available | Available | N/A |
| Industry focus | Manufacturing | Buildings + Industrial | Buildings + Industrial | Any |
| Typical savings | 15% | 8-10% | 8-10% | 0-5% |
| Price (per facility/year) | $200K-$300K | $150K-$250K | $150K-$250K | Labor cost |
| Payback period | <12 months | 18-24 months | 18-24 months | N/A |

---

## Approval Signatures

**Product Manager:** ___________________  Date: __________

**Engineering Lead:** ___________________  Date: __________

**Energy Domain Expert:** ___________________  Date: __________

**CEO:** ___________________  Date: __________

---

**Document Control:**
- **Version:** 1.0
- **Last Updated:** 2025-12-03
- **Next Review:** 2026-03-03 (quarterly)
- **Owner:** GreenLang Product Team
- **Classification:** Internal - Product Development