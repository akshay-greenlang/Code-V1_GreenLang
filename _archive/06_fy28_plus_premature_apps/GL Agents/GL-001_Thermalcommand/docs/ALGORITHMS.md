# GL-001 ThermalCommand - Algorithm Reference

This document provides a comprehensive reference for all algorithms implemented
in the GL-001 ThermalCommand agent. It includes mathematical formulations,
tuning guidelines, and references to applicable standards.

## Table of Contents

1. [MILP Load Allocation Optimizer](#1-milp-load-allocation-optimizer)
2. [Cascade PID Controller](#2-cascade-pid-controller)
3. [Safety Boundary Engine](#3-safety-boundary-engine)
4. [References and Standards](#4-references-and-standards)

---

## 1. MILP Load Allocation Optimizer

### 1.1 Overview

The Mixed Integer Linear Programming (MILP) optimizer allocates thermal load
across multiple equipment units to minimize total operating cost while
respecting operational constraints.

### 1.2 Mathematical Formulation

#### Objective Function

Minimize total operating cost:

```
min Z = SUM_{i=1}^{n} [ C_fuel,i * x_i / eta_i + C_fixed,i * y_i + C_CO2 * E_CO2,i * x_i / eta_i ]
```

Where:
| Symbol | Description | Units |
|--------|-------------|-------|
| Z | Total operating cost | $/hr |
| x_i | Thermal output of equipment i | MMBtu/hr |
| y_i | On/off status of equipment i | Binary {0,1} |
| C_fuel,i | Fuel cost for equipment i | $/MMBtu |
| eta_i | Efficiency at operating point | Dimensionless |
| C_fixed,i | Fixed operating cost | $/hr |
| C_CO2 | Carbon price | $/kg CO2 |
| E_CO2,i | CO2 emission factor | kg CO2/MMBtu |
| n | Number of equipment units | Integer |

#### Decision Variables

| Variable | Type | Bounds | Description |
|----------|------|--------|-------------|
| x_i | Continuous | [0, Q_max,i] | Thermal output (MMBtu/hr) |
| y_i | Binary | {0, 1} | Equipment on (1) or off (0) |
| s | Continuous | [0, D] | Unmet demand slack (MMBtu/hr) |

#### Constraints

**1. Demand Balance (Equality):**
```
SUM_{i=1}^{n} x_i + s = D
```
Where D is total thermal demand (MMBtu/hr)

**2. Capacity Upper Bound (Big-M):**
```
x_i <= Q_max,i * y_i    for all i
```
Equipment cannot produce output unless turned on.

**3. Turndown Lower Bound (Big-M):**
```
x_i >= Q_min,i * y_i    for all i
```
If equipment is on, it must operate above minimum turndown.

**4. Ramp Rate Limits (Optional):**
```
|x_i - x_i,prev| <= R_i * dt    for all i
```
Where R_i is ramp rate (MMBtu/hr per minute) and dt is time step.

### 1.3 Solver Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| gap_tolerance | 0.01 (1%) | 0.001-0.10 | Relative optimality gap |
| time_limit | 60 seconds | 5-300 | Maximum solve time |
| presolve | enabled | - | Problem reduction |

#### Termination Criteria

The solver terminates when ANY condition is met:
1. Gap tolerance achieved (proven optimal within tolerance)
2. Time limit reached (best feasible solution returned)
3. Problem proven infeasible or unbounded

### 1.4 Warm-Start Strategy

When `use_warm_start=True`, the previous solution provides:
1. Initial bounds on binary variables
2. Initial LP basis for faster first iteration
3. Incumbent solution for branch-and-bound pruning

Expected solve time reduction: 50-80% for small demand changes.

### 1.5 Linearization Approach

Efficiency eta_i is load-dependent and nonlinear. For MILP compatibility:

**Simple Linearization:**
- Use efficiency at 70% load point (typical operating range)
- Valid assumption: Most equipment operates 50-90% capacity

**Piecewise Linear (Advanced):**
- Define breakpoints: L = [L_1, L_2, ..., L_k]
- Define efficiencies: E = [E_1, E_2, ..., E_k]
- Interpolate: eta(L) = E_i + (E_{i+1} - E_i) * (L - L_i) / (L_{i+1} - L_i)

### 1.6 Parameter Tuning Guidelines

#### Gap Tolerance Selection

| Application | Gap Tolerance | Rationale |
|-------------|---------------|-----------|
| Real-time control | 0.05 (5%) | Speed over optimality |
| 15-minute dispatch | 0.02 (2%) | Balanced |
| Day-ahead planning | 0.005 (0.5%) | Optimality over speed |
| Detailed analysis | 0.001 (0.1%) | Near-optimal |

#### Time Limit Selection

| Application | Time Limit | Rationale |
|-------------|------------|-----------|
| Real-time (1 min cycle) | 5-10 sec | Must complete within cycle |
| 15-minute dispatch | 60 sec | Standard planning |
| Day-ahead planning | 300 sec | Full optimization |

### 1.7 Heuristic Fallback

When MILP is unavailable, merit-order dispatch is used:

**Algorithm:**
1. Calculate marginal cost: MC_i = (C_fuel + C_CO2 * e_CO2) / eta_i
2. Sort equipment by MC (ascending)
3. Dispatch in order until demand met
4. Respect turndown constraints

**Limitations:**
- Does not guarantee global optimum
- Does not optimize fixed costs
- May miss better equipment combinations

---

## 2. Cascade PID Controller

### 2.1 Overview

The cascade controller implements a master-slave PID hierarchy for process
heat applications, providing improved disturbance rejection and control
performance.

### 2.2 PID Algorithm

#### Standard Form (ISA)

```
u(t) = K_p * [ e(t) + (1/T_i) * INTEGRAL(e) dt + T_d * de/dt ]
```

#### Parallel Form (Implementation)

```
u(t) = K_p * e(t) + K_i * INTEGRAL(e) dt + K_d * de/dt
```

Where:
| Symbol | Description | Units |
|--------|-------------|-------|
| e(t) | Error = SP - PV | Engineering units |
| K_p | Proportional gain | %/unit or dimensionless |
| K_i | Integral gain = K_p / T_i | 1/seconds |
| K_d | Derivative gain = K_p * T_d | seconds |
| T_i | Integral time constant | seconds |
| T_d | Derivative time constant | seconds |

### 2.3 Discrete-Time Implementation

For sample time dt:

**Proportional Term:**
```
P(k) = K_p * e(k)
```

**Integral Term (with anti-windup):**
```
I(k) = I(k-1) + K_i * e(k) * dt

If output clamped:
    I(k) = I(k) + K_aw * (u_clamped - u_raw)
```

**Derivative Term (on PV, filtered):**
```
d_pv = (PV(k) - PV(k-1)) / dt
D_raw(k) = -K_d * d_pv
D(k) = alpha * D_raw(k) + (1-alpha) * D(k-1)
```

**Output:**
```
u_raw(k) = P(k) + I(k) + D(k) + FF
u(k) = clamp(u_raw, u_min, u_max)
```

### 2.4 Anti-Windup Mechanism

**Problem:** Integral windup occurs when output saturates but integral
continues accumulating, causing overshoot.

**Solution: Back-Calculation**

When output is clamped:
```
windup_correction = K_aw * (u_clamped - u_raw)
I(k) = I(k) + windup_correction
```

This "back-calculates" the integral to prevent accumulation while saturated.

### 2.5 Gain Scheduling

Process dynamics change with operating point. Gain scheduling adjusts tuning:

**Operating Regions:**

| Region | Typical PV Range | Tuning Characteristics |
|--------|------------------|------------------------|
| STARTUP | 0-30% | Aggressive, high K_p |
| LOW_LOAD | 30-50% | Conservative |
| NORMAL | 50-80% | Balanced |
| HIGH_LOAD | 80-100% | Fast response |
| SHUTDOWN | Decreasing | Safe, slow |

**Transition Formula:**
```
K_new = (1 - rate) * K_current + rate * K_target
```

### 2.6 Bumpless Transfer

**Problem:** Mode switching causes output discontinuities ("bumps").

**Solution: Back-Calculate Integral**

Manual to Auto:
```
I(k) = u_manual - P(k) - D(k)
```

This ensures: u_auto = P + I + D = P + (u_manual - P) + 0 = u_manual

### 2.7 Cascade Structure

**Configuration:**
```
     Master PID          Slave PID         Process
    [Temperature] ----> [Flow] ----> [Valve] ----> [Plant]
         ^                                           |
         +-------------------------------------------+
                    Primary Variable (Temp)
```

**Design Guidelines:**
1. Slave loop should be 3-10x faster than master
2. Tune slave first with master in manual
3. Then tune master with slave in cascade
4. Master tuning should be more conservative

### 2.8 Tuning Guidelines

#### Ziegler-Nichols Ultimate Gain Method

From sustained oscillation at ultimate gain K_u and period T_u:

| Controller | K_p | T_i | T_d |
|------------|-----|-----|-----|
| P only | 0.5 * K_u | - | - |
| PI | 0.45 * K_u | T_u / 1.2 | - |
| PID | 0.6 * K_u | T_u / 2 | T_u / 8 |

#### Lambda Tuning (Stability-Focused)

For FOPDT process (gain K, time constant tau, dead time theta):
```
K_p = tau / (K * (lambda_cl + theta))
T_i = tau
T_d = 0  (PI only recommended)
```

Where lambda_cl is desired closed-loop time constant.

#### Typical Values for Process Heat

| Application | K_p | T_i (sec) | T_d (sec) |
|-------------|-----|-----------|-----------|
| Temperature | 1-5 %/degF | 60-300 | 10-60 |
| Pressure | 5-20 %/psi | 30-120 | 0-30 |
| Flow | 0.5-2 %/% | 10-60 | 0 |

---

## 3. Safety Boundary Engine

### 3.1 Overview

The Safety Boundary Engine implements IEC 61511-compliant safety logic for
pre-actuation validation. It enforces safety boundaries through multiple
layers of protection.

### 3.2 IEC 61511 Compliance

| IEC 61511 Clause | Implementation |
|------------------|----------------|
| 11.2.1 | Demand mode operation with deterministic response |
| 11.3.1 | SIF independence via separate validation path |
| 11.4.2 | Voting logic (2oo3) for critical decisions |
| 11.5.1 | Proof testing via audit chain verification |
| 11.6.1 | Manual shutdown capability preserved |
| 11.7.1 | Fault tolerance via policy redundancy |

### 3.3 Safety Integrity Levels

| SIL | PFD Range | Risk Reduction | Implementation |
|-----|-----------|----------------|----------------|
| SIL 1 | 0.1 - 0.01 | 10-100 | Single redundancy |
| SIL 2 | 0.01 - 0.001 | 100-1,000 | Dual redundancy |
| SIL 3 | 0.001 - 0.0001 | 1,000-10,000 | 2oo3 voting |
| SIL 4 | 0.0001 - 1E-5 | 10,000-100,000 | Special measures |

GL-001 targets:
- SIL-2 for process control outputs
- SIL-3 capability for emergency shutdown functions

### 3.4 2oo3 Voting Logic

**Truth Table:**

| A | B | C | Vote | Action |
|---|---|---|------|--------|
| Safe | Safe | Safe | 3/3 Safe | Allow |
| Safe | Safe | Trip | 2/3 Safe | Allow |
| Safe | Trip | Safe | 2/3 Safe | Allow |
| Trip | Safe | Safe | 2/3 Safe | Allow |
| Safe | Trip | Trip | 2/3 Trip | Block |
| Trip | Safe | Trip | 2/3 Trip | Block |
| Trip | Trip | Safe | 2/3 Trip | Block |
| Trip | Trip | Trip | 3/3 Trip | Block |

**Boolean Formula:**
```
Vote = (A AND B) OR (B AND C) OR (A AND C)
```

Properties:
- Tolerates single sensor failure
- Fails safe (to trip) on 2+ faults
- Detects discrepancy for diagnostics

### 3.5 Safety Margin Calculations

**Absolute Margin:**
```
Margin = SafetyLimit - OperatingLimit
```

**Percentage Margin:**
```
Margin% = (SafetyLimit - OperatingLimit) / SafetyLimit * 100
```

**Time to Limit:**
```
TimeToLimit = Margin / RateOfChange
```

**Pre-Action Thresholds:**

| Condition | Action |
|-----------|--------|
| Margin < 20% | WARNING alarm |
| Margin < 10% | CRITICAL alarm |
| TimeToLimit < 5 min | WARNING alarm |
| TimeToLimit < 2 min | CRITICAL alarm |

### 3.6 Rate Limiting

**Cooldown Period:**
```
if (time_now - last_write_time) < cooldown_seconds:
    BLOCK request
```

**Rate of Change:**
```
rate = |value_new - value_prev| / elapsed_time

if rate > max_rate_per_second:
    BLOCK request
```

**Write Count:**
```
count = writes_in_last_60_seconds

if count >= max_writes_per_minute:
    BLOCK request
```

### 3.7 Decision Priority

1. **BLOCK (Emergency):**
   - Blacklisted tag
   - Interlock active
   - SIS violation
   - Emergency alarm

2. **BLOCK (Critical):**
   - Rate limit exceeded
   - Time restricted
   - Unauthorized tag

3. **BLOCK (Value):**
   - Over maximum
   - Under minimum

4. **CLAMP (if allowed):**
   - Value adjusted to nearest limit

5. **ALLOW:**
   - All checks passed

### 3.8 Audit Trail

**Blockchain-Style Chain:**
```
hash(i) = SHA256(timestamp | event_type | action | hash(i-1))
```

**Verification:**
```
for each record i:
    if i == 0:
        assert previous_hash == ""
    else:
        assert previous_hash == hash(i-1)
```

---

## 4. References and Standards

### 4.1 Control Systems

| Standard | Title | Application |
|----------|-------|-------------|
| ISA-5.1 | Instrumentation Symbols | P&ID notation |
| ISA-88 | Batch Control | Sequence control |
| IEC 61131-3 | PLC Programming | Controller implementation |

### 4.2 Safety Systems

| Standard | Title | Application |
|----------|-------|-------------|
| IEC 61511 | Safety Instrumented Systems | SIS design |
| IEC 61508 | Functional Safety | Safety lifecycle |
| ISA-84 | Safety Instrumented Systems | US implementation |
| ISA-18.2 | Alarm Management | Alarm design |

### 4.3 Process Heat

| Standard | Title | Application |
|----------|-------|-------------|
| ASME PTC 4 | Fired Steam Generators | Boiler efficiency |
| ASHRAE 90.1 | Energy Efficiency | Building systems |
| IEEE 519 | Harmonics | VFD equipment |

### 4.4 Emissions

| Source | Title | Application |
|--------|-------|-------------|
| EPA AP-42 | Emission Factors | CO2, NOx factors |
| IPCC Guidelines | GHG Inventories | Carbon accounting |

### 4.5 Optimization

| Reference | Title | Application |
|-----------|-------|-------------|
| Boyd & Vandenberghe | Convex Optimization | MILP theory |
| Williams | Model Building in Mathematical Programming | Problem formulation |

### 4.6 Control Theory

| Reference | Title | Application |
|-----------|-------|-------------|
| Astrom & Hagglund | PID Controllers: Theory, Design, Tuning | Anti-windup, tuning |
| Seborg et al. | Process Dynamics and Control | Cascade control |

---

## Appendix A: Quick Reference Cards

### A.1 MILP Optimizer Quick Reference

```
OBJECTIVE: min Z = SUM [ (C_fuel/eta + C_CO2*e_CO2/eta) * x + C_fixed * y ]

CONSTRAINTS:
  Demand:   SUM(x) + slack = D
  Capacity: x <= Q_max * y
  Turndown: x >= Q_min * y

DEFAULTS:
  gap_tolerance = 0.01 (1%)
  time_limit = 60 seconds
```

### A.2 PID Controller Quick Reference

```
OUTPUT: u = K_p*e + K_i*INTEGRAL(e) + K_d*de/dt

ANTI-WINDUP: I += K_aw * (u_clamped - u_raw)

TUNING (Z-N):
  P:   K_p = 0.5 * K_u
  PI:  K_p = 0.45 * K_u, T_i = T_u/1.2
  PID: K_p = 0.6 * K_u, T_i = T_u/2, T_d = T_u/8
```

### A.3 Safety Engine Quick Reference

```
VALIDATION SEQUENCE:
  1. Whitelist check
  2. Absolute limits
  3. Rate limits
  4. Conditional policies
  5. Time restrictions
  6. Interlock status
  7. Alarm status

2oo3 VOTING: Vote = (A AND B) OR (B AND C) OR (A AND C)

DECISION PRIORITY: Emergency > Critical > Value > Clamp > Allow
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2024-01 | GreenLang Team | Initial release |

---

*This document is maintained by the GreenLang Technical Documentation Team.*
