# Steam Header Pressure Control

## GL-003 UnifiedSteam - Pressure Control Documentation

**Document Version:** 1.0.0
**Last Updated:** 2025-12-27
**Author:** GreenLang Control Systems Team
**Reference Standards:** ASME B31.1, ISA-75.01, IEC 61511, API 520/521

---

## Table of Contents

1. [Overview](#overview)
2. [Header Pressure Architecture](#header-pressure-architecture)
3. [Control Strategies](#control-strategies)
4. [Optimization Logic](#optimization-logic)
5. [Safety Considerations](#safety-considerations)
6. [Integration with GL-003 Agent](#integration-with-gl-003-agent)
7. [Reference Values](#reference-values)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The GL-003 UnifiedSteam agent provides intelligent pressure control optimization for multi-header steam systems. This document describes the pressure control architecture, control strategies, optimization algorithms, and safety mechanisms implemented within the agent.

**Key Capabilities:**

- Multi-pressure header monitoring and optimization
- PRV (Pressure Reducing Valve) setpoint management
- Backpressure turbine integration and prioritization
- Dynamic setpoint adjustment based on demand forecasting
- Complete safety envelope enforcement with audit trail

**Important Safety Notice:**

GL-003 operates as an advisory and optimization system. It has **read-only access** to Safety Instrumented Systems (SIS) and **cannot modify** Safety Instrumented Functions (SIF) or bypass safety interlocks. All recommendations require operator confirmation before implementation.

---

## Header Pressure Architecture

### Multi-Pressure Header System

Industrial steam systems typically operate with multiple pressure levels to serve different process requirements efficiently. GL-003 supports the standard three-tier configuration:

```
+------------------------------------------------------------------+
|                     STEAM GENERATION                              |
|  +----------+  +----------+  +----------+                         |
|  | Boiler 1 |  | Boiler 2 |  | Boiler 3 |                         |
|  +----+-----+  +----+-----+  +----+-----+                         |
|       |             |             |                               |
|       +-------------+-------------+                               |
|                     |                                             |
|                     v                                             |
+------------------------------------------------------------------+
|              HIGH PRESSURE (HP) HEADER                            |
|              600-900 psig (4.1-6.2 MPa)                           |
|   +---------------------------------------------------------+     |
|   |                                                         |     |
|   |  Typical: 650 psig @ 750 F (4.5 MPa @ 400 C)            |     |
|   |                                                         |     |
|   +---------+------------------+------------------+---------+     |
|             |                  |                  |               |
|             v                  v                  v               |
|      +------+------+    +------+------+    +------+------+        |
|      | PRV-HP-MP-1 |    | PRV-HP-MP-2 |    | Backpressure|        |
|      | (Letdown)   |    | (Letdown)   |    | Turbine     |        |
|      +------+------+    +------+------+    +------+------+        |
|             |                  |                  |               |
|             +------------------+------------------+               |
|                                |                                  |
+------------------------------------------------------------------+
|             MEDIUM PRESSURE (MP) HEADER                           |
|             125-175 psig (0.86-1.2 MPa)                           |
|   +---------------------------------------------------------+     |
|   |                                                         |     |
|   |  Typical: 150 psig @ 400 F (1.0 MPa @ 205 C)            |     |
|   |                                                         |     |
|   +---------+------------------+------------------+---------+     |
|             |                  |                  |               |
|             v                  v                  v               |
|      +------+------+    +------+------+    +------+------+        |
|      | PRV-MP-LP-1 |    | PRV-MP-LP-2 |    | Process     |        |
|      | (Letdown)   |    | (Letdown)   |    | Users       |        |
|      +------+------+    +------+------+    +-------------+        |
|             |                  |                                  |
|             +------------------+                                  |
|                                |                                  |
+------------------------------------------------------------------+
|              LOW PRESSURE (LP) HEADER                             |
|              10-50 psig (0.07-0.34 MPa)                           |
|   +---------------------------------------------------------+     |
|   |                                                         |     |
|   |  Typical: 15 psig @ 250 F (0.1 MPa @ 121 C)             |     |
|   |                                                         |     |
|   +------------------+------------------+-------------------+     |
|                      |                  |                         |
|                      v                  v                         |
|               +------+------+    +------+------+                  |
|               | Process     |    | Space       |                  |
|               | Heating     |    | Heating     |                  |
|               +-------------+    +-------------+                  |
+------------------------------------------------------------------+
```

### Header Type Specifications

| Header | Pressure Range | Typical Setpoint | Temperature | Primary Users |
|--------|---------------|------------------|-------------|---------------|
| HP | 600-900 psig | 650 psig | 700-800 F | Turbine drives, high-temp processes |
| MP | 125-175 psig | 150 psig | 350-450 F | Process heating, steam ejectors |
| LP | 10-50 psig | 15 psig | 240-275 F | Space heating, deaerators |

### Pressure Reduction Stations

Pressure Reducing Valves (PRVs) and letdown stations manage steam flow between headers. GL-003 monitors and optimizes these critical control points.

**PRV Configuration Parameters (from prv_controller.py):**

```python
class PRVConfiguration:
    prv_id: str                          # PRV equipment ID
    upstream_header_id: str              # Source header
    downstream_header_id: str            # Destination header
    design_inlet_pressure_kpa: float     # Design inlet pressure
    design_outlet_pressure_kpa: float    # Design outlet pressure
    max_flow_capacity_kg_s: float        # Maximum flow capacity
    cv_rating: float                     # Valve Cv rating
    response_time_s: float               # Valve response time (default 5.0s)
```

**PRV Operating States:**

| State | Description |
|-------|-------------|
| NORMAL | Operating within design parameters |
| REDUCING | Actively reducing pressure |
| RELIEVING | Passing excess flow (high demand) |
| ISOLATED | Valve isolated for maintenance |
| FAULT | Fault condition detected |

### Backpressure Turbine Integration

Backpressure turbines extract work from steam while reducing pressure, offering significant efficiency advantages over PRVs:

```
                 HP Steam In
                     |
                     v
              +------+------+
              |             |
              | Backpressure|
              | Turbine     |---> Electrical Power (kW)
              |             |
              +------+------+
                     |
                     v
                 MP Steam Out
```

**Efficiency Comparison:**

| Method | Enthalpy Recovery | Power Generation | Typical Application |
|--------|------------------|------------------|---------------------|
| PRV (Isenthalpic) | 0% | 0 kW | Backup, low flow |
| Backpressure Turbine | 70-85% | Proportional to flow | Base load, high flow |
| Condensing Turbine | 85-90% | Maximum | Power generation focus |

GL-003 prioritizes backpressure turbines over PRVs when:
- Steam flow is sufficient for stable turbine operation
- Turbine is available and not in maintenance
- Grid or process power demand exists

---

## Control Strategies

### Boiler Master Pressure Control

The boiler master controller maintains HP header pressure by modulating boiler firing rates. GL-003 optimizes load allocation across multiple boilers.

```
                    +-------------------+
                    |                   |
HP Header --------->| Pressure          |
Pressure (PV)       | Transmitter       |
                    |                   |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |                   |
Setpoint (SP) ----->| Master Pressure   |
                    | Controller        |
                    | (PID)             |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |                   |
                    | Load Allocation   |<---- GL-003 Optimization
                    | (Multi-Boiler)    |
                    |                   |
                    +---+--------+------+
                        |        |
                        v        v
                   Boiler 1  Boiler 2  ... Boiler N
                   Firing    Firing       Firing
                   Rate      Rate         Rate
```

**Load Allocation Objectives (from steam_network_optimizer.py):**

| Objective | Description | Use Case |
|-----------|-------------|----------|
| cost | Minimize total operating cost | Normal operations |
| efficiency | Maximize weighted average efficiency | Energy audits |
| emissions | Minimize total CO2 emissions | Carbon reduction |
| balanced | Multi-objective with equal weights | Balanced operations |

### Header Pressure Setpoint Management

GL-003 manages pressure setpoints through the SetpointManager class, which provides:

- **Centralized Registration:** All setpoints tracked in unified system
- **Authorization Enforcement:** Multi-level approval for safety-critical changes
- **Constraint Validation:** Range, rate-of-change, and safety checks
- **Rollback Capability:** Automatic recovery to previous values
- **Complete Audit Trail:** Regulatory compliance documentation

**Setpoint Categories:**

```python
class SetpointCategory(str, Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    QUALITY = "quality"
    VALVE_POSITION = "valve_position"
    LEVEL = "level"
```

**Authorization Levels:**

| Level | Role | Typical Authority |
|-------|------|-------------------|
| OPERATOR | Control room operator | Normal setpoint adjustments |
| SUPERVISOR | Shift supervisor | Extended range adjustments |
| ENGINEER | Process engineer | Safety-critical changes |
| SAFETY_SYSTEM | SIS/SIF | Emergency actions only |

### Load Following vs Pressure Following Modes

GL-003 supports two primary control modes:

**Load Following Mode:**

```
Steam Demand ----> Boiler Loading ----> Pressure Maintained
     ^                                        |
     |                                        |
     +---------------- Feedback ---------------+
```

- Header pressure is maintained by adjusting generation
- Boilers modulate firing rates to match demand
- Appropriate for variable demand processes
- Provides stable pressure at the expense of efficiency swings

**Pressure Following Mode:**

```
Header Pressure ----> User Demand Adjusted ----> Balance Maintained
       ^                                               |
       |                                               |
       +------------------- Feedback ------------------+
```

- Demand is managed to maintain stable pressure
- Users may be shed during high-demand periods
- Appropriate for limited generation capacity
- Provides generation stability at the expense of user supply

### Cascade Control for Letdown Valves

GL-003 implements cascade control for PRV optimization:

```
+----------------------------------------------------------------+
|                    CASCADE CONTROL STRUCTURE                    |
+----------------------------------------------------------------+
|                                                                 |
|   OUTER LOOP (Header Pressure)                                  |
|   +-----------------------------------------------------------+ |
|   |                                                           | |
|   |  MP Header      +----------+        PRV Setpoint          | |
|   |  Pressure  ---->| Primary  |----->  (Output to            | |
|   |  Setpoint       | PID      |        Inner Loop)           | |
|   |                 +----------+                              | |
|   |                      ^                                    | |
|   |                      |                                    | |
|   |             MP Header Pressure (PV)                       | |
|   |                                                           | |
|   +-----------------------------------------------------------+ |
|                              |                                  |
|                              v                                  |
|   INNER LOOP (Valve Position)                                   |
|   +-----------------------------------------------------------+ |
|   |                                                           | |
|   |  PRV Setpoint   +----------+        Valve                 | |
|   |  (from outer) ->| Secondary|----->  Position              | |
|   |                 | PID      |        Command               | |
|   |                 +----------+                              | |
|   |                      ^                                    | |
|   |                      |                                    | |
|   |             PRV Outlet Pressure (PV)                      | |
|   |                                                           | |
|   +-----------------------------------------------------------+ |
|                                                                 |
+----------------------------------------------------------------+
```

**Cascade Control Benefits:**

1. Faster disturbance rejection
2. Improved stability during demand changes
3. Better valve positioning accuracy
4. Reduced pressure oscillations

---

## Optimization Logic

### Minimizing Letdown Losses

When steam is reduced through a PRV (isenthalpic expansion), significant exergy is lost. GL-003 calculates and minimizes these losses:

**Letdown Loss Calculation:**

```python
def _calculate_letdown_loss(
    flow_klb_hr: float,
    upstream_psig: float,
    downstream_psig: float,
) -> float:
    """Calculate energy loss from pressure letdown (BTU/hr)."""
    # Isenthalpic expansion - no heat loss but exergy loss
    pressure_drop = upstream_psig - downstream_psig
    # Approximate: 1 BTU/lb per psi for saturated steam
    exergy_loss_per_lb = pressure_drop * 1.0
    return flow_klb_hr * 1000 * exergy_loss_per_lb
```

**Loss Minimization Strategies:**

| Strategy | Description | Typical Savings |
|----------|-------------|-----------------|
| Backpressure Turbine | Extract work during pressure reduction | 70-85% exergy recovery |
| Direct Generation | Generate steam at required pressure level | Eliminates letdown entirely |
| Setpoint Optimization | Lower HP pressure when demand allows | 1% fuel per 10 psi reduction |
| Load Shifting | Schedule high-pressure users together | 5-15% reduction in letdown |

### Backpressure Turbine Prioritization

GL-003 implements intelligent prioritization of backpressure turbines over PRVs:

```
+------------------------------------------------------------------+
|              BACKPRESSURE TURBINE PRIORITIZATION                  |
+------------------------------------------------------------------+
|                                                                   |
|  1. Check Turbine Availability                                    |
|     +-- Is turbine online?                                        |
|     +-- Is turbine in maintenance?                                |
|     +-- Are permissives satisfied?                                |
|                                                                   |
|  2. Evaluate Flow Requirements                                    |
|     +-- Is demand > minimum stable flow?                          |
|     +-- Is demand < maximum turbine capacity?                     |
|                                                                   |
|  3. Check Power Demand                                            |
|     +-- Is electrical power needed?                               |
|     +-- What is current grid price?                               |
|                                                                   |
|  4. Calculate Economics                                           |
|     +-- Turbine power value ($/MWh)                               |
|     +-- PRV opportunity cost                                      |
|     +-- Maintenance considerations                                |
|                                                                   |
|  5. Route Steam Flow                                              |
|     +-- IF turbine favorable: Route through turbine               |
|     +-- ELSE: Route through PRV                                   |
|     +-- Balance remaining flow as needed                          |
|                                                                   |
+------------------------------------------------------------------+
```

### Dynamic Setpoint Adjustment

GL-003 adjusts pressure setpoints based on demand forecasting:

**Demand-Based Setpoint Calculation (from prv_controller.py):**

```python
def _calculate_demand_based_setpoint(
    self,
    demand_forecast: DemandForecast,
    header_pressure_kpa: float
) -> float:
    # Base setpoint from design value
    base_setpoint = self.config.design_outlet_pressure_kpa

    # Adjust based on demand ratio
    design_flow = self.config.max_flow_capacity_kg_s
    demand_ratio = demand_forecast.peak_demand_kg_s / design_flow

    # Higher demand may require slightly higher setpoint for margin
    demand_adjustment = 0.0
    if demand_ratio > 0.8:
        # High demand - increase setpoint for headroom
        demand_adjustment = (demand_ratio - 0.8) * 20.0
    elif demand_ratio < 0.3:
        # Low demand - can operate at lower setpoint
        demand_adjustment = -(0.3 - demand_ratio) * 15.0

    return base_setpoint + demand_adjustment
```

**Demand Forecast Integration:**

```
+------------------------------------------------------------------+
|                    DEMAND FORECASTING                             |
+------------------------------------------------------------------+
|                                                                   |
|  Inputs:                                                          |
|  +-- Historical demand patterns                                   |
|  +-- Production schedules                                         |
|  +-- Weather data (for heating loads)                             |
|  +-- Real-time consumption data                                   |
|                                                                   |
|  Forecast Horizon: 24 hours                                       |
|  Time Step: 5 minutes                                             |
|  Confidence Level: 85-95%                                         |
|                                                                   |
|  Outputs:                                                         |
|  +-- HP demand forecast (klb/hr)                                  |
|  +-- MP demand forecast (klb/hr)                                  |
|  +-- LP demand forecast (klb/hr)                                  |
|  +-- Peak demand prediction                                       |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Safety Considerations

### Pressure Relief Settings

GL-003 monitors pressure relief valve (PRV/PSV) setpoints and ensures operating pressures maintain adequate margin:

**Typical Pressure Relief Settings:**

| Header | MAWP (psig) | Relief Setpoint | Operating Limit | GL-003 Alarm |
|--------|-------------|-----------------|-----------------|--------------|
| HP | 900 | 900 | 850 | 800 |
| MP | 200 | 200 | 175 | 160 |
| LP | 75 | 75 | 50 | 45 |

**Safety Envelope Definition (from safety_envelope.py):**

```python
class PressureLimits(BaseModel):
    equipment_id: str
    min_kpa: float              # Minimum pressure limit
    max_kpa: float              # Maximum pressure limit
    design_pressure_kpa: float  # Design pressure
    test_pressure_kpa: float    # Hydrostatic test pressure
    alarm_margins: AlarmMargins # Warning/Alarm/Trip margins
```

**Alarm Margin Structure:**

```python
class AlarmMargins(BaseModel):
    warning_pct: float = 10.0   # Warning at 10% from limit
    alarm_pct: float = 5.0      # Alarm at 5% from limit
    trip_pct: float = 0.0       # Trip at limit
```

### High/Low Pressure Interlocks

GL-003 monitors interlock status but **cannot modify SIF logic**:

```
+------------------------------------------------------------------+
|                  PRESSURE INTERLOCK MATRIX                        |
+------------------------------------------------------------------+
|                                                                   |
|  CONDITION              ACTION              GL-003 RESPONSE       |
|  -----------------------------------------------------------------|
|  HP Header > HH         Trip boilers        Read-only monitoring  |
|  HP Header > H          Alarm               Advisory to operator  |
|  HP Header < L          Alarm               Advisory to operator  |
|  HP Header < LL         Trip PRVs closed    Read-only monitoring  |
|                                                                   |
|  MP Header > HH         Trip HP-MP PRVs     Read-only monitoring  |
|  MP Header > H          Alarm               Advisory to operator  |
|  MP Header < L          Alarm               Advisory to operator  |
|  MP Header < LL         Trip processes      Read-only monitoring  |
|                                                                   |
|  PRV Diff < Min         Close PRV           Read-only monitoring  |
|  PRV Position > Max     Alarm               Advisory to operator  |
|                                                                   |
+------------------------------------------------------------------+
```

**Interlock Manager Safety Philosophy (from interlock_manager.py):**

```
IMPORTANT: GL-003 has NO authority to modify SIF logic or bypass interlocks.
This module provides monitoring and permissive checking only.

Safety Philosophy:
    - GL-003 respects ALL existing safety envelopes
    - No control authority over Safety Instrumented Systems
    - Permissive checking before any optimization action
    - Complete interlock status monitoring and logging
```

### Emergency Venting Procedures

In overpressure situations, emergency venting may be required. GL-003 monitors these conditions:

```
+------------------------------------------------------------------+
|              EMERGENCY VENTING DECISION TREE                      |
+------------------------------------------------------------------+
|                                                                   |
|  1. Pressure Approaching Relief Setting                           |
|     |                                                             |
|     +-- GL-003 Action: Generate CRITICAL alert                    |
|     +-- GL-003 Action: Display pressure trend to operator         |
|     +-- GL-003 Action: Recommend load reduction                   |
|                                                                   |
|  2. Relief Valve Lifting                                          |
|     |                                                             |
|     +-- SIS Action: Relief valve opens automatically              |
|     +-- GL-003 Action: Log event with timestamp                   |
|     +-- GL-003 Action: Calculate steam loss rate                  |
|     +-- GL-003 Action: Generate EMERGENCY alert                   |
|                                                                   |
|  3. Sustained Overpressure                                        |
|     |                                                             |
|     +-- SIS Action: Trip boilers (automatic)                      |
|     +-- GL-003 Action: Read-only monitoring                       |
|     +-- GL-003 Action: Document for audit trail                   |
|                                                                   |
|  4. Recovery                                                      |
|     |                                                             |
|     +-- GL-003 Action: Monitor pressure normalization             |
|     +-- GL-003 Action: Calculate total steam loss                 |
|     +-- GL-003 Action: Generate incident report                   |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Integration with GL-003 Agent

### How the Agent Monitors Pressure

GL-003 continuously monitors pressure through multiple integration points:

```
+------------------------------------------------------------------+
|                  GL-003 PRESSURE MONITORING                       |
+------------------------------------------------------------------+
|                                                                   |
|  DATA SOURCES                                                     |
|  +-- OPC-UA Connector: Real-time pressure values                  |
|  +-- Historian Connector: Historical trends                       |
|  +-- Sensor Transformer: Data quality validation                  |
|  +-- Tag Mapper: Standardized naming conventions                  |
|                                                                   |
|  MONITORING FREQUENCY                                             |
|  +-- Real-time: 1-second updates for critical parameters          |
|  +-- Trend analysis: 1-minute aggregations                        |
|  +-- Optimization cycles: 5-minute intervals                      |
|                                                                   |
|  DATA QUALITY CHECKS                                              |
|  +-- Sensor uncertainty quantification                            |
|  +-- Outlier detection                                            |
|  +-- Frozen value detection                                       |
|  +-- Rate-of-change validation                                    |
|                                                                   |
+------------------------------------------------------------------+
```

**Pressure Monitoring Tags (typical):**

| Tag Pattern | Description | Engineering Unit |
|-------------|-------------|------------------|
| PT-HP-001 | HP Header Pressure | psig |
| PT-MP-001 | MP Header Pressure | psig |
| PT-LP-001 | LP Header Pressure | psig |
| PT-PRV-001-IN | PRV Inlet Pressure | psig |
| PT-PRV-001-OUT | PRV Outlet Pressure | psig |
| PDT-PRV-001 | PRV Differential Pressure | psi |

### Pressure-Related KPIs

GL-003 calculates and tracks the following pressure-related Key Performance Indicators:

| KPI | Description | Target | Unit |
|-----|-------------|--------|------|
| Header Pressure Stability | Standard deviation of pressure | < 2% of setpoint | % |
| PRV Utilization | Percentage of time PRVs are active | < 30% (prefer turbines) | % |
| Letdown Loss | Energy lost through PRV letdown | < 5% of generation | BTU/hr |
| Pressure Margin | Distance from relief setpoint | > 10% | % |
| Setpoint Tracking Error | Deviation from pressure setpoint | < 1% | psi |
| Demand Forecast Accuracy | Accuracy of demand predictions | > 90% | % |

**KPI Dashboard Metrics (from monitoring/metrics.py):**

```python
# Prometheus-style metrics exported by GL-003

# Gauge: Current header pressure
steam_header_pressure_psig{header="hp"}
steam_header_pressure_psig{header="mp"}
steam_header_pressure_psig{header="lp"}

# Gauge: Pressure setpoint
steam_header_setpoint_psig{header="hp"}
steam_header_setpoint_psig{header="mp"}
steam_header_setpoint_psig{header="lp"}

# Counter: PRV operations
steam_prv_operations_total{prv_id="PRV-001", operation="open"}
steam_prv_operations_total{prv_id="PRV-001", operation="close"}

# Histogram: Pressure deviations
steam_pressure_deviation_psig{header="hp"}

# Gauge: Letdown loss rate
steam_letdown_loss_btu_hr{prv_id="PRV-001"}
```

### Automated Recommendations for Pressure Optimization

GL-003 generates actionable recommendations through the advisory system:

**Advisory Types (from prv_controller.py):**

| Advisory Type | Description | Priority |
|---------------|-------------|----------|
| SETPOINT_CHANGE | Recommended pressure setpoint adjustment | 2-4 |
| CONSTRAINT_APPROACH | Approaching operational constraint | 2-3 |
| DEMAND_FORECAST | Demand-based proactive adjustment | 3-4 |
| MAINTENANCE_RECOMMENDATION | Equipment maintenance needed | 3-5 |
| OPTIMIZATION_OPPORTUNITY | Efficiency improvement available | 4-5 |

**Example Advisory Generation:**

```python
advisory = PRVAdvisory(
    advisory_id="ADV_abc123",
    advisory_type=AdvisoryType.SETPOINT_CHANGE,
    prv_id="PRV-HP-MP-1",
    title="PRV Setpoint Change: 650 -> 620 kPa",
    description="""
        Proposed decrease of 30 kPa from 650 to 620 kPa.
        Reason: Demand forecast shows 25% reduction in next 4 hours.
        Expected benefit: 2.1% reduction in letdown losses.
    """,
    current_value=650.0,
    recommended_value=620.0,
    unit="kPa",
    priority=3,
    requires_confirmation=True,
    expected_benefit="$45/hr savings from reduced letdown",
    safety_validated=True
)
```

**Recommendation Workflow:**

```
+------------------------------------------------------------------+
|              PRESSURE OPTIMIZATION WORKFLOW                       |
+------------------------------------------------------------------+
|                                                                   |
|  1. Data Collection                                               |
|     +-- Current pressure readings                                 |
|     +-- Demand forecast                                           |
|     +-- Equipment status                                          |
|     +-- Active interlocks                                         |
|                                                                   |
|  2. Analysis                                                      |
|     +-- Calculate optimal setpoints                               |
|     +-- Evaluate load allocation                                  |
|     +-- Identify loss reduction opportunities                     |
|                                                                   |
|  3. Validation                                                    |
|     +-- Check against safety constraints                          |
|     +-- Verify permissives satisfied                              |
|     +-- Rate-limit check for changes                              |
|                                                                   |
|  4. Advisory Generation                                           |
|     +-- Create structured recommendation                          |
|     +-- Calculate expected benefits                               |
|     +-- Assign priority level                                     |
|                                                                   |
|  5. Operator Interface                                            |
|     +-- Display recommendation to operator                        |
|     +-- Await confirmation or rejection                           |
|     +-- Log decision with audit trail                             |
|                                                                   |
|  6. Implementation (if approved)                                  |
|     +-- Apply setpoint change via DCS                             |
|     +-- Monitor response                                          |
|     +-- Document results                                          |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Reference Values

### Standard Pressure Constraints

**HP Header Constraints:**

```python
# From steam_network_optimizer.py
hp_header_min_psig = 550.0    # Minimum HP pressure
hp_header_max_psig = 900.0    # Maximum HP pressure (MAWP)
hp_typical_setpoint = 650.0   # Normal operating setpoint
```

**MP Header Constraints:**

```python
mp_header_min_psig = 100.0    # Minimum MP pressure
mp_header_max_psig = 200.0    # Maximum MP pressure (MAWP)
mp_typical_setpoint = 150.0   # Normal operating setpoint
```

**LP Header Constraints:**

```python
lp_header_min_psig = 5.0      # Minimum LP pressure
lp_header_max_psig = 75.0     # Maximum LP pressure (MAWP)
lp_typical_setpoint = 15.0    # Normal operating setpoint
```

### PRV Constraints

```python
# From prv_controller.py
class PressureConstraints(BaseModel):
    min_downstream_pressure_kpa: float   # Minimum outlet pressure
    max_downstream_pressure_kpa: float   # Maximum outlet pressure
    max_pressure_rate_kpa_per_min: float = 50.0  # Max rate of change
    min_differential_pressure_kpa: float = 50.0  # Min PRV differential
    alarm_low_kpa: float                 # Low pressure alarm
    alarm_high_kpa: float                # High pressure alarm
```

### Rate of Change Limits

| Parameter | Limit | Reason |
|-----------|-------|--------|
| HP Header Pressure | 50 psig/min | Thermal stress prevention |
| MP Header Pressure | 30 psig/min | Equipment protection |
| LP Header Pressure | 20 psig/min | Process stability |
| PRV Outlet | 10 psig/min | Downstream stability |
| Boiler Pressure | 25 psig/min | Drum stress limits |

---

## Troubleshooting

### Common Pressure Control Issues

**Issue: HP Header Pressure Oscillating**

| Possible Cause | Diagnostic | Resolution |
|----------------|------------|------------|
| PID tuning | Check controller gains | Retune with reduced proportional gain |
| PRV hunting | Check valve positioner | Service or replace positioner |
| Demand swings | Review process schedules | Coordinate load changes |
| Sensor noise | Check PT output | Add damping or replace sensor |

**Issue: Unable to Maintain MP Header Pressure**

| Possible Cause | Diagnostic | Resolution |
|----------------|------------|------------|
| Insufficient HP generation | Check boiler loading | Increase boiler output or start standby |
| PRV capacity exceeded | Check PRV position (>90%) | Parallel additional PRV or use turbine |
| Downstream leak | Check mass balance | Locate and repair leak |
| Demand exceeds capacity | Compare demand vs capacity | Shed non-critical loads |

**Issue: PRV Differential Too Low**

| Possible Cause | Diagnostic | Resolution |
|----------------|------------|------------|
| HP pressure dropping | Check HP header trend | Investigate HP pressure loss |
| MP pressure rising | Check MP header loads | Increase MP demand or vent |
| PRV stuck open | Check valve position feedback | Service or replace valve |
| Control malfunction | Check setpoint vs actual | Verify controller output |

### Diagnostic Commands

GL-003 provides CLI commands for pressure troubleshooting:

```bash
# Check current pressure status
unifiedsteam pressure status

# View pressure trends (last 60 minutes)
unifiedsteam pressure trends --header hp --duration 60m

# Analyze PRV performance
unifiedsteam prv analyze --prv-id PRV-HP-MP-1

# Generate pressure optimization report
unifiedsteam optimize pressure --generate-report

# View active interlocks
unifiedsteam safety interlocks --active-only

# Check setpoint history
unifiedsteam setpoints history --tag PT-HP-001.SP --duration 24h
```

### Alert Types for Pressure Issues

| Alert Type | Severity | Description |
|------------|----------|-------------|
| SAFETY_ENVELOPE | EMERGENCY/CRITICAL | Pressure approaching safety limits |
| BALANCE_DEVIATION | WARNING/CRITICAL | Mass/energy balance discrepancy |
| OPTIMIZATION_OPPORTUNITY | INFO/WARNING | Pressure optimization available |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-27 | GreenLang Control Systems Team | Initial release |

---

## Related Documentation

- [API Reference](./API_REFERENCE.md) - GL-003 API documentation
- [Architecture Overview](./architecture/) - System architecture details
- [Safety Envelope Configuration](../safety/safety_envelope.py) - Safety limit definitions
- [PRV Controller](../control/prv_controller.py) - PRV control implementation
- [Steam Network Optimizer](../optimization/steam_network_optimizer.py) - Optimization algorithms

---

**Disclaimer:** This documentation is for informational purposes. Always consult site-specific procedures and qualified engineering personnel before making changes to steam system pressure controls. GL-003 provides recommendations only; final implementation decisions rest with authorized plant personnel.
