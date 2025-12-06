# GL-018 UnifiedCombustion Optimizer HAZOP Study

## Hazard and Operability Analysis per IEC 61882

**Document ID:** GL-HAZOP-018
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61882:2016, NFPA 85, NFPA 86

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Study Scope and Methodology](#2-study-scope-and-methodology)
3. [System Description](#3-system-description)
4. [HAZOP Team and Responsibilities](#4-hazop-team-and-responsibilities)
5. [Node 1: Air-Fuel Ratio Control](#5-node-1-air-fuel-ratio-control)
6. [Node 2: O2 Trim Control](#6-node-2-o2-trim-control)
7. [Node 3: Emissions Monitoring](#7-node-3-emissions-monitoring)
8. [Node 4: Burner Management Interface](#8-node-4-burner-management-interface)
9. [Node 5: Flue Gas Analysis](#9-node-5-flue-gas-analysis)
10. [Risk Ranking Matrix](#10-risk-ranking-matrix)
11. [Action Items Summary](#11-action-items-summary)
12. [Appendices](#12-appendices)

---

## 1. Introduction

### 1.1 Purpose

This document presents the Hazard and Operability (HAZOP) study for the GL-018 UnifiedCombustion Optimizer, a consolidated combustion optimization agent that provides air-fuel ratio control, O2 trim, emissions optimization, and Burner Management System (BMS) coordination.

### 1.2 Objectives

- Identify combustion-related hazards and operability issues
- Evaluate consequences of air-fuel ratio deviations
- Assess flame stability and burner safety implications
- Verify NFPA 85/86 compliance for BMS interface
- Recommend safeguards for combustion safety

### 1.3 References

| Document | Title |
|----------|-------|
| NFPA 85 | Boiler and Combustion Systems Hazards Code |
| NFPA 86 | Standard for Ovens and Furnaces |
| API 560 | Fired Heaters for General Refinery Service |
| ASME PTC 4.1 | Steam Generating Units Performance Test Code |
| EPA Method 19 | Determination of Sulfur Dioxide Removal Efficiency |
| IEC 61511-1:2016 | Functional Safety - Safety Instrumented Systems |

### 1.4 Combustion Safety Background

Combustion systems present inherent hazards including:

- **Explosion/Deflagration:** Accumulation of unburned fuel
- **Fire:** External fire from leaks or hot surfaces
- **Toxic Releases:** CO, NOx, and other combustion products
- **Thermal Damage:** Overtemperature of equipment
- **Efficiency Loss:** Environmental and economic impact

---

## 2. Study Scope and Methodology

### 2.1 Scope

This HAZOP covers the GL-018 UnifiedCombustion Optimizer including:

- Air-fuel ratio control and cross-limiting logic
- O2 trim control and excess air adjustment
- NOx/CO emissions monitoring and optimization
- Burner Management System (BMS) interface
- Flue gas analysis and analyzer integration

**Excluded from scope:**
- Mechanical burner design (covered in equipment HAZOP)
- Fuel supply system upstream of burner valve (covered in fuel system HAZOP)
- Electrical supply to burners (covered in electrical assessment)

### 2.2 Methodology

The study follows IEC 61882:2016 with emphasis on NFPA 85/86 requirements for combustion safety.

### 2.3 Guide Words Applied

| Guide Word | Application to Combustion |
|------------|---------------------------|
| **NO** | No air, no fuel, no flame |
| **MORE** | Excess air, excess fuel, rich mixture |
| **LESS** | Insufficient air, lean mixture |
| **AS WELL AS** | Contaminated air/fuel, additional flow |
| **PART OF** | Partial combustion, incomplete flow |
| **REVERSE** | Backflow, reverse signal |
| **OTHER THAN** | Wrong fuel type, wrong burner |

---

## 3. System Description

### 3.1 GL-018 UnifiedCombustion Optimizer Overview

The GL-018 agent consolidates functionality from:
- GL-002 FLAMEGUARD (Boiler flame monitoring)
- GL-004 BURNMASTER (Burner optimization)
- GL-018 FLUEFLOW (Flue gas analysis)

**Key Features:**
- ASME PTC 4.1 efficiency calculations
- API 560 combustion analysis
- Air-fuel ratio optimization with O2 trim per NFPA 85
- Flame Stability Index (FSI) calculation
- NOx/CO emissions control (LNB, FGR, SCR)
- BMS sequence coordination per NFPA 85

### 3.2 Process Flow Diagram

```
       AIR SUPPLY                        FUEL SUPPLY
          |                                   |
          v                                   v
    +----------+                        +----------+
    |Air Flow  |                        |Fuel Flow |
    |Controller|                        |Controller|
    +----------+                        +----------+
          |                                   |
          v                                   v
    +----------+                        +----------+
    |Air Damper|                        |Fuel Valve|
    |Actuator  |                        |Actuator  |
    +----------+                        +----------+
          |                                   |
          +---------+               +---------+
                    |               |
                    v               v
                +-------------------+
                |     BURNER        |
                |   (Combustion)    |
                +-------------------+
                         |
                         v
                +-------------------+
                |   FURNACE/        |
                |   BOILER          |
                +-------------------+
                         |
                         v
                +-------------------+
                |   FLUE GAS        |<------ Flue Gas Analyzers
                |   STACK           |        (O2, CO, NOx)
                +-------------------+

                    GL-018 AGENT
         +---------------------------+
         |  - Air-Fuel Ratio Control |
         |  - O2 Trim                |
         |  - FSI Calculation        |
         |  - Emissions Monitoring   |
         |  - BMS Interface          |
         +---------------------------+
```

### 3.3 Key Parameters

| Parameter | Normal Range | Low Alarm | High Alarm | Trip |
|-----------|-------------|-----------|------------|------|
| Air-Fuel Ratio | 10.5-12.5:1 | 9.5:1 | 14:1 | 8:1 or 16:1 |
| Flue Gas O2 | 2-4% | 1.5% | 6% | 1% or 8% |
| CO | <50 ppm | - | 100 ppm | 400 ppm |
| NOx | <100 ppm | - | 150 ppm | 200 ppm |
| Flame Signal | 70-90% | 50% | - | 30% |
| Flue Gas Temp | 350-500F | 250F | 600F | 700F |

### 3.4 Safety Functions per NFPA 85

| Safety Function | Response Time | SIL Requirement |
|-----------------|---------------|-----------------|
| Flame Failure Trip | <4 seconds | SIL 2 |
| Low Fuel Pressure Trip | <3 seconds | SIL 2 |
| Low Air Flow Trip | <3 seconds | SIL 2 |
| High Flue Gas Temp Trip | <5 seconds | SIL 1 |
| Pre-Purge Timing | 4 volume changes | SIL 2 |

---

## 4. HAZOP Team and Responsibilities

### 4.1 Study Team

| Role | Name | Responsibility |
|------|------|----------------|
| **HAZOP Leader** | [Facilitator] | Study facilitation, documentation |
| **Combustion Engineer** | [Combustion SME] | Combustion process knowledge |
| **Safety Engineer** | [Safety SME] | NFPA 85/86 compliance |
| **Operations Rep** | [Ops SME] | Burner operations, procedures |
| **Control Systems** | [Controls SME] | BMS configuration |
| **Environmental** | [Env SME] | Emissions compliance |

---

## 5. Node 1: Air-Fuel Ratio Control

### 5.1 Node Description

**Design Intent:** Maintain optimal air-fuel ratio for efficient, safe combustion. Cross-limiting logic ensures fuel-rich conditions cannot occur during transients.

**Parameters:**
- Air flow setpoint
- Fuel flow setpoint
- Air-fuel ratio
- Cross-limiting signals
- Load demand

### 5.2 HAZOP Worksheet - Node 1

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 1.1 | NO | No air flow signal | Air flow transmitter failure, wiring fault | Loss of air flow control, potential rich mixture | Low air flow alarm, flame scanner trip, BMS lockout | R1.1 | H |
| 1.2 | NO | No fuel flow signal | Fuel flow transmitter failure, wiring fault | Loss of fuel control, potential lean/flameout | Low fuel pressure trip, flame scanner | R1.2 | H |
| 1.3 | MORE | Higher air flow than fuel | Air damper stuck open, fuel valve stuck partially closed | Lean combustion, potential flameout | Flame stability monitoring, FSI alarm | R1.3 | M |
| 1.4 | MORE | Higher fuel flow than air | Fuel valve stuck open, air damper stuck partially closed | Rich combustion, CO/soot, explosion risk | Cross-limiting logic, CO alarm, O2 low trip | - | VH |
| 1.5 | LESS | Lower air-fuel ratio (rich) | Cross-limiting failure, setpoint error | Incomplete combustion, CO, explosion hazard | O2 low trip, CO high alarm, flame scanner | - | VH |
| 1.6 | LESS | Lower air flow than required | Air damper drift, fan degradation | Rich combustion, efficiency loss | O2 trim correction, low O2 alarm | R1.4 | H |
| 1.7 | REVERSE | Inverted air-fuel signal | Wiring error, configuration error | Completely wrong ratio control | Commissioning verification, ratio monitoring | R1.5 | VH |
| 1.8 | AS WELL AS | Air with excess humidity | Weather conditions, air intake issue | Combustion efficiency loss, possible flameout | Humidity compensation, efficiency monitoring | R1.6 | L |
| 1.9 | AS WELL AS | Fuel with contaminants | Fuel supply contamination | Burner fouling, flame instability, emissions | Fuel analysis, FSI monitoring | R1.7 | M |
| 1.10 | PART OF | Partial air flow (blocked intake) | Inlet blockage, damper fault | Localized rich combustion, burner damage | Differential pressure monitoring, inspection | R1.8 | M |
| 1.11 | OTHER THAN | Wrong fuel type introduced | Operator error, supply change | Incorrect ratio, flame instability, damage | Fuel type verification, Wobbe index check | R1.9 | H |

### 5.3 Node 1 - Cross-Limiting Logic Detail

Per NFPA 85, cross-limiting logic must ensure:
- **Lead Air / Lag Fuel on Load Increase:** Air increases before fuel
- **Lead Fuel / Lag Air on Load Decrease:** Fuel decreases before air
- **Minimum Air Floor:** Air flow never below safe minimum

```
Cross-Limiting Logic Diagram:

    Load Demand
        |
        v
   +--------+      +--------+
   |  Air   |      |  Fuel  |
   | Demand |      | Demand |
   +--------+      +--------+
        |              |
        v              v
   +--------+      +--------+
   |  High  |      |  Low   |
   | Select |<---->| Select |
   +--------+      +--------+
        |              |
        v              v
   +--------+      +--------+
   |  Air   |      |  Fuel  |
   | Output |      | Output |
   +--------+      +--------+

   Ensures: Air >= Fuel (always lean or stoichiometric)
```

### 5.4 Node 1 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R1.1 | Implement redundant air flow measurement (2oo3) | Critical | Process | Open |
| R1.2 | Implement redundant fuel flow measurement (2oo3) | Critical | Process | Open |
| R1.3 | Add FSI threshold alarm for flame instability | High | Controls | Open |
| R1.4 | Add air damper position verification with feedback | High | Controls | Open |
| R1.5 | Conduct commissioning verification of ratio control polarity | Critical | Commissioning | Open |
| R1.6 | Add combustion air humidity compensation | Medium | Software | Open |
| R1.7 | Implement fuel quality monitoring with Wobbe index tracking | Medium | Process | Open |
| R1.8 | Add air inlet differential pressure monitoring | High | Process | Open |
| R1.9 | Implement fuel type verification with operator confirmation | High | Operations | Open |

---

## 6. Node 2: O2 Trim Control

### 6.1 Node Description

**Design Intent:** Optimize excess air by trimming air flow based on flue gas O2 measurement to maximize efficiency while maintaining safe combustion.

**Parameters:**
- Flue gas O2 setpoint (target: 2-3%)
- O2 trim bias signal
- Maximum trim range (+/- 5%)
- Analyzer validation status

### 6.2 HAZOP Worksheet - Node 2

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 2.1 | NO | No O2 signal | Analyzer failure, sample line blockage | O2 trim disabled, static operation | Analyzer health monitoring, failsafe to default | R2.1 | M |
| 2.2 | MORE | Higher O2 reading than actual | Analyzer drift high, air leak in sample line | Trim reduces air, potential rich condition | Analyzer calibration schedule, plausibility check | R2.2 | H |
| 2.3 | MORE | Higher O2 trim than allowed | Trim limit failure, setpoint error | Excessive air, efficiency loss | Trim range limits, efficiency monitoring | R2.3 | L |
| 2.4 | LESS | Lower O2 reading than actual | Analyzer drift low, sample line contamination | Trim increases air unnecessarily | Analyzer redundancy, cross-check with CO | R2.4 | M |
| 2.5 | LESS | Lower O2 than safe | Trim drives air too low, constraint failure | Rich combustion, CO, explosion risk | O2 low hard limit (1.5%), CO trip | - | VH |
| 2.6 | REVERSE | Inverted O2 trim direction | Software error, configuration error | Trim drives opposite direction | Trim direction validation, response monitoring | R2.5 | VH |
| 2.7 | AS WELL AS | O2 with CO interference | Analyzer cross-sensitivity | Incorrect O2 reading | CO-corrected O2 measurement | R2.6 | M |
| 2.8 | AS WELL AS | O2 trim with rapid load changes | Load transient during trim | Trim action inappropriate for conditions | Load-dependent trim enable/disable | R2.7 | M |
| 2.9 | PART OF | Partial sample (blocked probe) | Sample probe fouling | Unrepresentative O2 reading | Sample flow monitoring, probe inspection | R2.8 | M |
| 2.10 | OTHER THAN | O2 trim on wrong burner | Configuration error, multi-burner addressing | Wrong burner affected | Burner ID validation | R2.9 | H |

### 6.3 O2 Trim Safety Limits

| Parameter | Value | Basis |
|-----------|-------|-------|
| O2 Setpoint | 2.5-3.0% | Efficiency optimization |
| O2 Low Alarm | 1.5% | Approaching lean flammability |
| O2 Low-Low Trip | 1.0% | NFPA 85 safety limit |
| O2 High Alarm | 6.0% | Excessive excess air |
| O2 High-High Trip | 8.0% | Flame stability concern |
| Trim Bias Maximum | +/- 5% | Prevent excessive trim |

### 6.4 Node 2 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R2.1 | Implement analyzer health monitoring with automatic failover | High | Controls | Open |
| R2.2 | Establish analyzer calibration schedule per manufacturer specs | High | Instrument | Open |
| R2.3 | Add O2 reading plausibility check against CO and load | High | Software | Open |
| R2.4 | Install redundant O2 analyzer for safety-critical applications | High | Process | Open |
| R2.5 | Add trim direction validation with response verification | Critical | Software | Open |
| R2.6 | Use CO-corrected O2 algorithm | Medium | Software | Open |
| R2.7 | Disable O2 trim during load transients (>5%/min) | High | Controls | Open |
| R2.8 | Add sample flow monitoring with low flow alarm | High | Instrument | Open |
| R2.9 | Implement burner-specific trim with ID validation | High | Controls | Open |

---

## 7. Node 3: Emissions Monitoring

### 7.1 Node Description

**Design Intent:** Monitor NOx and CO emissions for environmental compliance and combustion optimization. Provide recommendations for Low NOx Burner (LNB), Flue Gas Recirculation (FGR), and SCR optimization.

**Parameters:**
- NOx concentration (ppm)
- CO concentration (ppm)
- Permit limits (lb/MMBTU)
- FGR rate
- SCR efficiency

### 7.2 HAZOP Worksheet - Node 3

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 3.1 | NO | No NOx reading | NOx analyzer failure | Loss of compliance monitoring | Backup analyzer, manual sampling | R3.1 | M |
| 3.2 | NO | No CO reading | CO analyzer failure | Loss of combustion quality indicator | CO analyzer redundancy | R3.2 | M |
| 3.3 | MORE | Higher NOx than limit | Combustion too hot, low FGR | Permit violation, penalties | NOx alarm, FGR increase recommendation | R3.3 | M |
| 3.4 | MORE | Higher CO than safe | Rich combustion, burner fouling | Toxic release, permit violation, CO poisoning risk | CO high alarm, CO trip at 400 ppm | - | H |
| 3.5 | LESS | Lower NOx reading than actual | Analyzer drift, sample dilution | Compliance complacency, actual violation | Analyzer calibration, RATA testing | R3.4 | M |
| 3.6 | LESS | FGR rate too low | FGR fan failure, damper stuck | NOx above target | FGR status monitoring, NOx alarm | R3.5 | M |
| 3.7 | REVERSE | FGR increases NOx instead | Wrong injection point, excessive FGR | Worse emissions, efficiency loss | FGR effectiveness monitoring | R3.6 | M |
| 3.8 | AS WELL AS | Emissions with ammonia slip | SCR over-injection | Secondary pollution, health hazard | Ammonia slip monitoring, injection optimization | R3.7 | M |
| 3.9 | PART OF | Partial emission monitoring | Some analyzers offline | Incomplete compliance picture | Analyzer status dashboard, redundancy | R3.8 | L |
| 3.10 | OTHER THAN | Wrong emission factor applied | Fuel change not detected | Incorrect lb/MMBTU calculation | Fuel type verification, factor lookup | R3.9 | L |

### 7.3 Emission Limits and Alarms

| Parameter | Permit Limit | Warning | Alarm | Trip |
|-----------|-------------|---------|-------|------|
| NOx | 0.03 lb/MMBTU | 80% | 95% | - |
| CO | 0.08 lb/MMBTU | 50 ppm | 100 ppm | 400 ppm |
| Opacity | 20% | 15% | 18% | 20% |
| Ammonia Slip | 5 ppm | 3 ppm | 4 ppm | 5 ppm |

### 7.4 Node 3 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R3.1 | Install backup NOx analyzer or enable manual sampling protocol | Medium | Process | Open |
| R3.2 | Install redundant CO analyzer for combustion safety | High | Process | Open |
| R3.3 | Implement automatic FGR adjustment based on NOx feedback | Medium | Controls | Open |
| R3.4 | Conduct quarterly RATA testing per 40 CFR Part 60 | High | Environmental | Open |
| R3.5 | Add FGR fan and damper status monitoring | High | Controls | Open |
| R3.6 | Add FGR effectiveness trending and optimization | Medium | Software | Open |
| R3.7 | Implement SCR ammonia injection optimization algorithm | Medium | Software | Open |
| R3.8 | Add emissions analyzer availability KPI dashboard | Medium | Operations | Open |
| R3.9 | Implement automatic fuel type detection and factor selection | Medium | Software | Open |

---

## 8. Node 4: Burner Management Interface

### 8.1 Node Description

**Design Intent:** Interface with Burner Management System (BMS) for safe startup, shutdown, and flame failure response per NFPA 85/86. Coordinate permissive verification, purge sequences, and interlock management.

**Parameters:**
- BMS sequence state
- Flame scanner signals
- Permissive status
- Purge timing
- Trip status

### 8.2 NFPA 85 Sequence Requirements

```
BMS Startup Sequence (NFPA 85):

1. INITIAL CONDITIONS
   - All interlocks satisfied
   - Fuel valves proven closed
   - Burner controls in low-fire position

2. PRE-PURGE
   - Air flow verified (>=25% capacity)
   - Purge timer: 4 furnace volume changes
   - Typical: 30-60 seconds minimum

3. PILOT TRIAL FOR IGNITION
   - Pilot fuel valve opens
   - Ignition source activated
   - Pilot flame proven (10-15 seconds max)

4. MAIN FLAME TRIAL FOR IGNITION
   - Main fuel valve opens
   - Main flame proven (10-15 seconds max)
   - Pilot may be shut off (interrupted pilot)

5. RUNNING
   - Normal modulating operation
   - Flame monitoring continuous
   - Safety interlocks active
```

### 8.3 HAZOP Worksheet - Node 4

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 4.1 | NO | No flame signal | Flame failure, scanner failure | BMS trip (correct response) or false trip | Redundant scanners (1oo2 or 2oo2) | - | - |
| 4.2 | NO | No purge completion signal | Timer failure, air flow not verified | BMS remains in purge, no startup | Purge timeout alarm, manual override with authorization | R4.1 | L |
| 4.3 | NO | No response to flame failure | BMS communication loss, logic failure | Fuel continues to unlit furnace, explosion | Hardwired flame failure trip, independent BMS | - | VH |
| 4.4 | MORE | Longer purge than required | Timer error, conservative setting | Startup delay, efficiency loss | Purge timing optimization | R4.2 | L |
| 4.5 | MORE | More permissives required than actual | Overly conservative interlocks | Startup prevented unnecessarily | Permissive review, proper classification | R4.3 | L |
| 4.6 | LESS | Shorter purge than required | Timer failure, bypass | Insufficient purge, explosion risk | Hardware timer, bypass prohibition | - | VH |
| 4.7 | LESS | Fewer permissives checked | Bypass, configuration error | Unsafe startup | Permissive tracking, bypass logging | R4.4 | H |
| 4.8 | REVERSE | False flame signal (flame indicated when none) | Scanner contamination, ambient light | Fuel admitted without ignition, explosion | Scanner verification, startup trial logic | - | VH |
| 4.9 | AS WELL AS | BMS with spurious trips | Noise, loose wiring | Nuisance trips, production loss | Wiring integrity, shielding | R4.5 | M |
| 4.10 | PART OF | Partial shutdown (some burners not tripped) | Multi-burner coordination failure | Uneven shutdown, thermal stress | All-burner trip verification | R4.6 | H |
| 4.11 | OTHER THAN | Wrong BMS sequence executed | State machine error | Inappropriate sequence for conditions | State validation, operator confirmation | R4.7 | H |

### 8.4 Flame Scanner Requirements

| Parameter | Requirement | NFPA Reference |
|-----------|-------------|----------------|
| Flame Failure Response Time | <4 seconds | NFPA 85 8.5.2.2 |
| Scanner Self-Check | Continuous or periodic | NFPA 85 8.5.3.2 |
| Scanner Redundancy | 1oo2 or 2oo2 as appropriate | NFPA 85 8.5.3.1 |
| Discrimination | Must discriminate against adjacent flames | NFPA 85 8.5.3.1 |

### 8.5 Node 4 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R4.1 | Add purge sequence monitoring with diagnostic messaging | Medium | Controls | Open |
| R4.2 | Optimize purge timing based on actual furnace volume calculation | Low | Process | Open |
| R4.3 | Review permissive classification per NFPA 85 guidelines | Medium | Safety | Open |
| R4.4 | Implement comprehensive bypass tracking and reporting | Critical | Controls | Open |
| R4.5 | Verify flame scanner wiring integrity and shielding | High | Instrument | Open |
| R4.6 | Add multi-burner trip verification with all-confirmed logic | High | Controls | Open |
| R4.7 | Add BMS state validation with operator confirmation for manual sequences | High | Controls | Open |

---

## 9. Node 5: Flue Gas Analysis

### 9.1 Node Description

**Design Intent:** Analyze flue gas composition to support combustion optimization, efficiency calculations, and emissions compliance. Integration with O2, CO, CO2, and NOx analyzers.

**Parameters:**
- Flue gas temperature
- O2 concentration
- CO concentration
- CO2 concentration (calculated or measured)
- NOx concentration
- Excess air percentage

### 9.2 HAZOP Worksheet - Node 5

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 5.1 | NO | No flue gas sample | Sample line blocked, pump failure | Loss of analyzer data | Sample flow monitoring, line blowback | R5.1 | M |
| 5.2 | NO | No analyzer output | Analyzer failure, power loss | Loss of combustion feedback | Analyzer redundancy, UPS | R5.2 | M |
| 5.3 | MORE | Higher flue gas temp than actual | Thermowell fouling, calibration error | Wrong efficiency calculation | Thermowell inspection, calibration | R5.3 | L |
| 5.4 | MORE | Higher calculated losses | Wrong input data, formula error | Pessimistic efficiency, over-tuning | Calculation validation, cross-check | R5.4 | L |
| 5.5 | LESS | Lower flue gas temp indication | Thermowell damage, sensor drift | Wrong stack loss calculation | Sensor redundancy, comparison | R5.5 | L |
| 5.6 | LESS | Lower excess air calculated | O2 reading low, algorithm error | Underestimate air margin, risk | O2/CO cross-check | R5.6 | M |
| 5.7 | REVERSE | Inverted analyzer polarity | Wiring error | Completely wrong reading | Commissioning verification | R5.7 | H |
| 5.8 | AS WELL AS | Sample with condensation | Cooler failure, dew point | Analyzer damage, wrong readings | Sample conditioning monitoring | R5.8 | M |
| 5.9 | AS WELL AS | Flue gas with air dilution | Stack leak, negative pressure | Low apparent concentrations | Stack pressure monitoring | R5.9 | M |
| 5.10 | PART OF | Partial analysis (some species missing) | Selective analyzer failure | Incomplete combustion picture | Analyzer status aggregation | R5.10 | L |
| 5.11 | OTHER THAN | Analysis of wrong stream | Sample point error | Unrepresentative data | Sample point verification | R5.11 | M |

### 9.3 Analyzer Requirements

| Analyzer | Range | Accuracy | Response Time | Maintenance |
|----------|-------|----------|---------------|-------------|
| O2 | 0-25% | +/- 0.1% | <30 seconds | Daily zero, weekly span |
| CO | 0-2000 ppm | +/- 10 ppm | <60 seconds | Weekly calibration |
| CO2 | 0-20% | +/- 0.5% | <60 seconds | Weekly calibration |
| NOx | 0-500 ppm | +/- 5 ppm | <120 seconds | Weekly calibration |

### 9.4 Node 5 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R5.1 | Add sample flow monitoring with automatic line blowback | High | Instrument | Open |
| R5.2 | Install UPS for analyzer cabinet | Medium | Electrical | Open |
| R5.3 | Establish thermowell inspection schedule | Medium | Maintenance | Open |
| R5.4 | Add efficiency calculation validation routine | Medium | Software | Open |
| R5.5 | Install redundant flue gas temperature measurement | Medium | Process | Open |
| R5.6 | Implement O2/CO cross-check for combustion validation | High | Software | Open |
| R5.7 | Conduct commissioning verification of all analyzer polarities | Critical | Commissioning | Open |
| R5.8 | Add sample conditioning monitoring with alarm | High | Instrument | Open |
| R5.9 | Monitor stack pressure for air infiltration | Medium | Process | Open |
| R5.10 | Add analyzer status aggregation dashboard | Medium | Operations | Open |
| R5.11 | Verify sample point locations during commissioning | High | Commissioning | Open |

---

## 10. Risk Ranking Matrix

### 10.1 Risk Summary for GL-018

| Risk Level | Count | Key Deviations |
|------------|-------|----------------|
| Very High (VH) | 6 | Rich combustion (1.4, 2.5, 2.6), flame failure response (4.3, 4.6, 4.8) |
| High (H) | 10 | Air-fuel signal loss, O2 drift, CO high, BMS permissives |
| Medium (M) | 18 | Analyzer failures, trim issues, emissions monitoring |
| Low (L) | 8 | Efficiency calculations, minor operational impacts |

### 10.2 Combustion-Specific Risk Factors

| Hazard Category | Risk Multiplier | Justification |
|-----------------|-----------------|---------------|
| Explosion/Deflagration | x2 | Potential for catastrophic event |
| Flame Instability | x1.5 | Leads to multiple secondary hazards |
| Emissions Violation | x1 | Regulatory and financial impact |
| Efficiency Loss | x0.5 | Economic impact only |

---

## 11. Action Items Summary

### 11.1 Critical Priority Actions (VH Risk)

| Action ID | Description | Owner | Due Date | Status |
|-----------|-------------|-------|----------|--------|
| R1.5 | Commissioning verification of ratio control polarity | Commissioning | 2025-12-31 | Open |
| R2.5 | Trim direction validation with response verification | Software | 2025-12-31 | Open |
| R4.4 | Comprehensive bypass tracking and reporting | Controls | 2025-12-31 | Open |
| R5.7 | Commissioning verification of analyzer polarities | Commissioning | 2025-12-31 | Open |

### 11.2 High Priority Actions

| Action ID | Description | Owner | Due Date | Status |
|-----------|-------------|-------|----------|--------|
| R1.1 | Redundant air flow measurement (2oo3) | Process | 2026-01-31 | Open |
| R1.2 | Redundant fuel flow measurement (2oo3) | Process | 2026-01-31 | Open |
| R1.4 | Air damper position verification | Controls | 2026-01-31 | Open |
| R1.8 | Air inlet differential pressure monitoring | Process | 2026-01-31 | Open |
| R1.9 | Fuel type verification | Operations | 2026-01-31 | Open |
| R2.1 | Analyzer health monitoring with failover | Controls | 2026-01-31 | Open |
| R2.2 | Analyzer calibration schedule | Instrument | 2026-01-31 | Open |
| R2.3 | O2 plausibility check | Software | 2026-01-31 | Open |
| R2.4 | Redundant O2 analyzer | Process | 2026-01-31 | Open |
| R2.7 | Disable O2 trim during transients | Controls | 2026-01-31 | Open |
| R2.8 | Sample flow monitoring | Instrument | 2026-01-31 | Open |
| R2.9 | Burner-specific trim validation | Controls | 2026-01-31 | Open |
| R3.2 | Redundant CO analyzer | Process | 2026-01-31 | Open |
| R3.4 | RATA testing schedule | Environmental | 2026-01-31 | Open |
| R3.5 | FGR status monitoring | Controls | 2026-01-31 | Open |
| R4.5 | Flame scanner wiring verification | Instrument | 2026-01-31 | Open |
| R4.6 | Multi-burner trip verification | Controls | 2026-01-31 | Open |
| R4.7 | BMS state validation | Controls | 2026-01-31 | Open |
| R5.1 | Sample flow monitoring with blowback | Instrument | 2026-01-31 | Open |
| R5.6 | O2/CO cross-check | Software | 2026-01-31 | Open |
| R5.8 | Sample conditioning monitoring | Instrument | 2026-01-31 | Open |
| R5.11 | Sample point verification | Commissioning | 2026-01-31 | Open |

---

## 12. Appendices

### Appendix A: NFPA 85 Cross-Reference

| NFPA 85 Section | Requirement | HAZOP Coverage |
|-----------------|-------------|----------------|
| 8.4.2 | Purge Requirements | Node 4 (4.2, 4.6) |
| 8.5.2 | Flame Detection | Node 4 (4.1, 4.8) |
| 8.6 | Fuel Safety Shutoff | Node 4 (4.3) |
| 8.7 | Combustion Air | Node 1 (1.1, 1.6) |
| 8.8 | Fuel/Air Ratio | Node 1 (1.3-1.7) |

### Appendix B: Abbreviations

| Abbreviation | Definition |
|--------------|------------|
| BMS | Burner Management System |
| CEMS | Continuous Emissions Monitoring System |
| FGR | Flue Gas Recirculation |
| FSI | Flame Stability Index |
| LNB | Low NOx Burner |
| RATA | Relative Accuracy Test Audit |
| SCR | Selective Catalytic Reduction |
| SNCR | Selective Non-Catalytic Reduction |

### Appendix C: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang Process Heat Safety Documentation Package.*
