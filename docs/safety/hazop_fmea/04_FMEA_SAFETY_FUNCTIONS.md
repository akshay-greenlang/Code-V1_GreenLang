# Failure Mode and Effects Analysis (FMEA) for Safety Functions

## Safety-Related Functions per IEC 60812

**Document ID:** GL-FMEA-SAFETY-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 60812:2018, IEC 61508, IEC 61511

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [FMEA Methodology](#2-fmea-methodology)
3. [Safety Function 1: High Temperature Trip (GL-001)](#3-safety-function-1-high-temperature-trip-gl-001)
4. [Safety Function 2: Flame Failure Response (GL-018)](#4-safety-function-2-flame-failure-response-gl-018)
5. [Safety Function 3: High Pressure Interlock (GL-001)](#5-safety-function-3-high-pressure-interlock-gl-001)
6. [Safety Function 4: Low Fuel Pressure Cutoff (GL-018)](#6-safety-function-4-low-fuel-pressure-cutoff-gl-018)
7. [Safety Function 5: TMT High Alarm (GL-007)](#7-safety-function-5-tmt-high-alarm-gl-007)
8. [Safety Function 6: Loss of Air Flow Response (GL-018)](#8-safety-function-6-loss-of-air-flow-response-gl-018)
9. [Safety Function 7: Emergency Shutdown Interface (GL-001)](#9-safety-function-7-emergency-shutdown-interface-gl-001)
10. [Diagnostic Coverage Calculations](#10-diagnostic-coverage-calculations)
11. [Common Cause Failure Analysis](#11-common-cause-failure-analysis)
12. [Safe Failure Fraction Calculations](#12-safe-failure-fraction-calculations)
13. [Summary and Conclusions](#13-summary-and-conclusions)

---

## 1. Introduction

### 1.1 Purpose

This document presents the Failure Mode and Effects Analysis (FMEA) for safety-related functions in the GreenLang Process Heat agents (GL-001, GL-018/GL-002, GL-007). The analysis identifies failure modes, evaluates their effects, and calculates risk metrics per IEC 60812 and IEC 61508/61511.

### 1.2 Scope

The FMEA covers seven critical safety functions:

1. High Temperature Trip (GL-001)
2. Flame Failure Response (GL-018)
3. High Pressure Interlock (GL-001)
4. Low Fuel Pressure Cutoff (GL-018)
5. TMT High Alarm (GL-007)
6. Loss of Air Flow Response (GL-018)
7. Emergency Shutdown Interface (GL-001)

### 1.3 References

| Document | Title |
|----------|-------|
| IEC 60812:2018 | Failure Mode and Effects Analysis (FMEA and FMECA) |
| IEC 61508-6:2010 | Functional Safety - Part 6: Guidelines |
| IEC 61511-1:2016 | Functional Safety - SIS for Process Industry |
| GL-HAZOP-001 | GL-001 HAZOP Study |
| GL-HAZOP-018 | GL-018 HAZOP Study |
| GL-HAZOP-007 | GL-007 HAZOP Study |

### 1.4 Definitions

| Term | Definition |
|------|------------|
| **Dangerous Failure** | Failure that prevents safety function from operating when demanded |
| **Safe Failure** | Failure that causes safety function to operate without demand (spurious trip) |
| **Detected Failure** | Failure detected by diagnostics before demand |
| **Undetected Failure** | Failure not detected until proof test or demand |
| **RPN** | Risk Priority Number (Severity x Occurrence x Detection) |
| **SFF** | Safe Failure Fraction |
| **DC** | Diagnostic Coverage |

---

## 2. FMEA Methodology

### 2.1 Approach

The FMEA follows IEC 60812:2018 methodology enhanced for SIS applications per IEC 61508:

1. Define safety function and its components
2. Identify failure modes for each component
3. Determine effects (local, system, plant)
4. Classify failures (safe/dangerous, detected/undetected)
5. Calculate severity, occurrence, and detection ratings
6. Calculate RPN and prioritize actions
7. Recommend mitigation measures

### 2.2 Rating Scales

#### 2.2.1 Severity Rating (S)

| Rating | Description | Effect |
|--------|-------------|--------|
| 10 | Hazardous | Fatality potential, no warning |
| 9 | Serious | Major injury potential, no warning |
| 8 | Very High | Major injury with warning |
| 7 | High | Minor injury, equipment damage |
| 6 | Moderate | Equipment damage, production loss |
| 5 | Low | Minor equipment impact |
| 4 | Very Low | Minor operational impact |
| 3 | Minor | Slight inconvenience |
| 2 | Very Minor | Noticed by trained personnel only |
| 1 | None | No effect |

#### 2.2.2 Occurrence Rating (O)

| Rating | Description | Failure Rate |
|--------|-------------|--------------|
| 10 | Very High | >1/day |
| 9 | | 1/week |
| 8 | High | 1/month |
| 7 | | 1/3 months |
| 6 | Moderate | 1/6 months |
| 5 | | 1/year |
| 4 | Low | 1/2 years |
| 3 | | 1/5 years |
| 2 | Remote | 1/10 years |
| 1 | Nearly Impossible | <1/10 years |

#### 2.2.3 Detection Rating (D)

| Rating | Description | Detection Method |
|--------|-------------|------------------|
| 10 | Almost Impossible | No detection possible |
| 9 | Very Remote | Random detection only |
| 8 | Remote | Detection during proof test only |
| 7 | Very Low | Detection during operation possible |
| 6 | Low | Operator may detect |
| 5 | Moderate | Periodic diagnostic test |
| 4 | Moderately High | Process indication change |
| 3 | High | Continuous online diagnostic |
| 2 | Very High | Immediate alarm on failure |
| 1 | Almost Certain | Fail-safe design, self-announcing |

### 2.3 RPN Calculation

```
RPN = Severity (S) x Occurrence (O) x Detection (D)

RPN Ranges:
- 1-50: Low priority (monitor)
- 51-100: Medium priority (review)
- 101-200: High priority (action required)
- >200: Critical priority (immediate action)
```

### 2.4 Failure Classification per IEC 61508

```
                    +----------------+
                    |  All Failures  |
                    +----------------+
                           |
              +------------+------------+
              |                         |
       +------v------+           +------v------+
       |    Safe     |           |  Dangerous  |
       |  Failures   |           |  Failures   |
       +-------------+           +-------------+
              |                         |
      +-------+-------+         +-------+-------+
      |               |         |               |
+-----v-----+   +-----v-----+   +-----v-----+   +-----v-----+
|   Safe    |   |   Safe    |   | Dangerous |   | Dangerous |
| Detected  |   | Undetected|   | Detected  |   | Undetected|
|   (SD)    |   |   (SU)    |   |   (DD)    |   |   (DU)    |
+-----------+   +-----------+   +-----------+   +-----------+
```

---

## 3. Safety Function 1: High Temperature Trip (GL-001)

### 3.1 Function Description

**Safety Function:** SIF-001 High Temperature Shutdown
**Agent:** GL-001 ThermalCommand Orchestrator
**SIL:** SIL 2
**Response Time:** <500ms
**Voting Logic:** 2oo3

### 3.2 Functional Block Diagram

```
Temperature    Temperature    Temperature
Sensor A       Sensor B       Sensor C
  (TT-101A)     (TT-101B)     (TT-101C)
    |               |               |
    v               v               v
+-------+       +-------+       +-------+
| Input |       | Input |       | Input |
| Card A|       | Card B|       | Card C|
+-------+       +-------+       +-------+
    |               |               |
    +-------+-------+-------+-------+
            |               |
            v               v
      +-----------+   +-----------+
      | GL-001    |   | SIS Logic |
      | Agent     |   | Solver    |
      +-----------+   +-----------+
            |               |
            v               v
      +-----------+   +-----------+
      | Output    |   | Hardwired |
      | Command   |   | Trip      |
      +-----------+   +-----------+
            |               |
            +-------+-------+
                    |
                    v
            +---------------+
            | Fuel Shutoff  |
            | Valve         |
            +---------------+
```

### 3.3 FMEA Worksheet - High Temperature Trip

| ID | Component | Failure Mode | Effect (Local) | Effect (System) | Effect (Plant) | Class | S | O | D | RPN | Action |
|----|-----------|--------------|----------------|-----------------|----------------|-------|---|---|---|-----|--------|
| 1.1 | Sensor TT-101A | Drift high | False high reading | May trigger trip | Spurious shutdown | SD | 4 | 4 | 2 | 32 | Calibration schedule |
| 1.2 | Sensor TT-101A | Drift low | False low reading | Under-protection (1 of 3) | None (2oo3 intact) | DU | 7 | 4 | 8 | 224 | Add comparison diagnostics |
| 1.3 | Sensor TT-101A | Open circuit | Loss of signal | Input goes to failsafe | May cause trip | SD | 4 | 3 | 2 | 24 | Open-wire detection |
| 1.4 | Sensor TT-101A | Short circuit | Fixed low reading | Under-protection (1 of 3) | None (2oo3 intact) | DD | 7 | 3 | 3 | 63 | Short detection enabled |
| 1.5 | Sensor TT-101B | Same as A | Same as A | Same as A | Same as A | - | - | - | - | - | - |
| 1.6 | Sensor TT-101C | Same as A | Same as A | Same as A | Same as A | - | - | - | - | - | - |
| 1.7 | Input Card A | Channel failure | Loss of one input | Degraded voting (1oo2) | Reduced reliability | DD | 6 | 3 | 2 | 36 | Card diagnostics |
| 1.8 | Input Card A | All channels fail | Loss of all A inputs | Degraded voting | Reduced reliability | DD | 7 | 2 | 2 | 28 | Watchdog timer |
| 1.9 | GL-001 Agent | Software crash | No processing | Fallback to SIS | Protection intact | SD | 5 | 3 | 2 | 30 | Watchdog, heartbeat |
| 1.10 | GL-001 Agent | Calculation error | Wrong comparison | May miss high temp | Potential damage | DU | 8 | 2 | 7 | 112 | Validation testing |
| 1.11 | GL-001 Agent | Communication loss | No output to DCS | Fallback to SIS | Protection intact | SD | 4 | 4 | 2 | 32 | Redundant comms |
| 1.12 | SIS Logic Solver | Stuck in state | No trip output | Protection lost | Potential hazard | DU | 9 | 2 | 7 | 126 | Proof testing |
| 1.13 | SIS Logic Solver | Spurious output | False trip | Spurious shutdown | Production loss | SD | 4 | 3 | 2 | 24 | Voting logic |
| 1.14 | Output Relay | Fail to de-energize | No trip signal | Valve not commanded | Potential hazard | DU | 9 | 2 | 8 | 144 | Proof testing |
| 1.15 | Output Relay | Welded contacts | Stuck energized | Cannot trip | Potential hazard | DU | 9 | 2 | 8 | 144 | Proof testing |
| 1.16 | Fuel Valve | Fail to close | Fuel continues | Hazard continues | Fire/explosion risk | DU | 10 | 2 | 8 | 160 | Proof testing, redundant valve |
| 1.17 | Fuel Valve | Spurious closure | Fuel stops | Furnace trips | Production loss | SD | 4 | 3 | 1 | 12 | Fail-safe (desirable) |

### 3.4 Summary - High Temperature Trip

| Failure Category | Count | Total Failure Rate |
|------------------|-------|-------------------|
| Safe Detected (SD) | 7 | 1.2 x 10^-5 /hr |
| Safe Undetected (SU) | 0 | 0 |
| Dangerous Detected (DD) | 3 | 4.5 x 10^-6 /hr |
| Dangerous Undetected (DU) | 7 | 8.2 x 10^-6 /hr |

**Top RPN Failure Modes:**
1. Sensor drift low (RPN 224) - Add comparison diagnostics
2. Fuel valve fail to close (RPN 160) - Add redundant valve
3. Output relay failures (RPN 144) - Proof testing program

---

## 4. Safety Function 2: Flame Failure Response (GL-018)

### 4.1 Function Description

**Safety Function:** SIF-004 Flame Failure Detection
**Agent:** GL-018 UnifiedCombustion Optimizer
**SIL:** SIL 2
**Response Time:** <4 seconds per NFPA 85
**Voting Logic:** 1oo2 or 2oo2 (configurable)

### 4.2 FMEA Worksheet - Flame Failure Response

| ID | Component | Failure Mode | Effect (Local) | Effect (System) | Effect (Plant) | Class | S | O | D | RPN | Action |
|----|-----------|--------------|----------------|-----------------|----------------|-------|---|---|---|-----|--------|
| 2.1 | UV Scanner A | Blind (no output) | No flame detection | Trips if 1oo2 | Spurious shutdown | SD | 4 | 4 | 2 | 32 | Self-check feature |
| 2.2 | UV Scanner A | False flame (always sees flame) | Ignores flame loss | Fuel continues to unlit furnace | Explosion risk | DU | 10 | 2 | 9 | 180 | Scanner self-check |
| 2.3 | UV Scanner A | Slow response | Delayed detection | Late trip | Extended hazard | DU | 8 | 3 | 7 | 168 | Response time testing |
| 2.4 | UV Scanner A | Drift in sensitivity | Marginal detection | Intermittent detection | Nuisance trips or misses | DD | 7 | 4 | 5 | 140 | Sensitivity trending |
| 2.5 | UV Scanner B | Same as A | Same as A | Same as A | Same as A | - | - | - | - | - | - |
| 2.6 | Scanner Amplifier | No output | Loss of scanner signal | Trips | Spurious shutdown | SD | 4 | 3 | 2 | 24 | Output monitoring |
| 2.7 | Scanner Amplifier | Fixed output (stuck high) | False flame | Explosion risk | Explosion risk | DU | 10 | 2 | 8 | 160 | Comparison logic |
| 2.8 | GL-018 Agent | FSI calculation error | Wrong stability assessment | May miss instability | Flame loss undetected | DU | 8 | 2 | 7 | 112 | FSI validation |
| 2.9 | GL-018 Agent | Timeout error | Late response | Trip delayed | Extended hazard | DU | 8 | 3 | 6 | 144 | Watchdog monitoring |
| 2.10 | BMS Logic | Stuck in running state | No trip on flame loss | Fuel continues | Explosion risk | DU | 10 | 2 | 8 | 160 | Independent trip |
| 2.11 | BMS Logic | Lockout fails | Cannot restart | No production | Production loss | SD | 3 | 3 | 2 | 18 | Reset verification |
| 2.12 | Fuel Valve (SV) | Fail to close | Fuel continues | Hazard continues | Fire/explosion | DU | 10 | 2 | 8 | 160 | Double block + bleed |
| 2.13 | Fuel Valve (SV) | Leak through | Some fuel passes | Potential ignition | Fire risk | DU | 9 | 3 | 8 | 216 | Leak testing |
| 2.14 | Ignition System | Fail to spark | No pilot light | Startup failure | Delay only | SD | 3 | 4 | 3 | 36 | Spark detection |
| 2.15 | Ignition System | Continuous spark | Arc damage | Equipment damage | Fire risk | DD | 6 | 3 | 4 | 72 | Spark timing |

### 4.3 Summary - Flame Failure Response

| Failure Category | Count | Total Failure Rate |
|------------------|-------|-------------------|
| Safe Detected (SD) | 4 | 9.5 x 10^-6 /hr |
| Safe Undetected (SU) | 0 | 0 |
| Dangerous Detected (DD) | 2 | 3.2 x 10^-6 /hr |
| Dangerous Undetected (DU) | 9 | 1.1 x 10^-5 /hr |

**Top RPN Failure Modes:**
1. Fuel valve leak through (RPN 216) - Leak testing program
2. UV scanner false flame (RPN 180) - Self-check diagnostics
3. Scanner slow response (RPN 168) - Response time verification

---

## 5. Safety Function 3: High Pressure Interlock (GL-001)

### 5.1 Function Description

**Safety Function:** SIF-003 High Pressure Protection
**Agent:** GL-001 ThermalCommand Orchestrator
**SIL:** SIL 2
**Response Time:** <500ms
**Voting Logic:** 2oo3

### 5.2 FMEA Worksheet - High Pressure Interlock

| ID | Component | Failure Mode | Effect (Local) | Effect (System) | Effect (Plant) | Class | S | O | D | RPN | Action |
|----|-----------|--------------|----------------|-----------------|----------------|-------|---|---|---|-----|--------|
| 3.1 | PT-201A | Drift high | False high reading | May trigger trip | Spurious shutdown | SD | 4 | 4 | 2 | 32 | Calibration |
| 3.2 | PT-201A | Drift low | False low reading | Under-protection (1 of 3) | None (2oo3 intact) | DU | 8 | 4 | 8 | 256 | Comparison diagnostics |
| 3.3 | PT-201A | Plugged impulse line | Stuck reading | Under-protection | May not detect rise | DU | 8 | 4 | 7 | 224 | Line blowdown |
| 3.4 | PT-201A | Diaphragm failure | Loss of signal | Failsafe action | May cause trip | SD | 4 | 3 | 2 | 24 | Diagnostics |
| 3.5 | PT-201B | Same as A | Same as A | Same as A | Same as A | - | - | - | - | - | - |
| 3.6 | PT-201C | Same as A | Same as A | Same as A | Same as A | - | - | - | - | - | - |
| 3.7 | GL-001 Agent | Setpoint error | Wrong trip point | Early or late trip | Protection compromised | DU | 8 | 2 | 6 | 96 | Setpoint validation |
| 3.8 | GL-001 Agent | Voting logic error | Wrong decision | Protection compromised | Potential hazard | DU | 9 | 2 | 7 | 126 | Logic testing |
| 3.9 | SIS Relay | Fail to operate | No trip signal | Protection lost | Potential hazard | DU | 9 | 2 | 8 | 144 | Proof testing |
| 3.10 | Relief Valve | Fail to open | No relief | Vessel overpressure | Rupture risk | DU | 10 | 2 | 8 | 160 | PSV testing |
| 3.11 | Relief Valve | Premature lift | Relief at low pressure | Lost inventory | Production impact | SD | 4 | 3 | 3 | 36 | Setting verification |
| 3.12 | Relief Valve | Stuck open | Continuous relief | Lost inventory | Production loss | DD | 5 | 3 | 3 | 45 | Position indicator |

### 5.3 Summary - High Pressure Interlock

| Failure Category | Count | Total Failure Rate |
|------------------|-------|-------------------|
| Safe Detected (SD) | 3 | 8.2 x 10^-6 /hr |
| Safe Undetected (SU) | 0 | 0 |
| Dangerous Detected (DD) | 1 | 1.5 x 10^-6 /hr |
| Dangerous Undetected (DU) | 8 | 9.8 x 10^-6 /hr |

**Top RPN Failure Modes:**
1. Sensor drift low (RPN 256) - Comparison diagnostics
2. Plugged impulse line (RPN 224) - Line blowdown schedule
3. Relief valve fail to open (RPN 160) - PSV testing program

---

## 6. Safety Function 4: Low Fuel Pressure Cutoff (GL-018)

### 6.1 Function Description

**Safety Function:** Low Fuel Pressure Trip
**Agent:** GL-018 UnifiedCombustion Optimizer
**SIL:** SIL 2
**Response Time:** <3 seconds
**Voting Logic:** 1oo2

### 6.2 FMEA Worksheet - Low Fuel Pressure Cutoff

| ID | Component | Failure Mode | Effect (Local) | Effect (System) | Effect (Plant) | Class | S | O | D | RPN | Action |
|----|-----------|--------------|----------------|-----------------|----------------|-------|---|---|---|-----|--------|
| 4.1 | PS-301A | Drift high | Misses low pressure | No trip on low fuel | Flame instability | DU | 8 | 4 | 7 | 224 | Comparison logic |
| 4.2 | PS-301A | Drift low | False low indication | Spurious trip | Production loss | SD | 4 | 4 | 2 | 32 | Calibration |
| 4.3 | PS-301A | Stuck contacts | No change of state | Protection lost | Flame loss possible | DU | 8 | 3 | 8 | 192 | Proof testing |
| 4.4 | PS-301B | Same as A | Same as A | Same as A | Same as A | - | - | - | - | - | - |
| 4.5 | GL-018 Agent | Threshold error | Wrong setpoint | Early or late trip | Protection compromised | DU | 7 | 2 | 6 | 84 | Setpoint validation |
| 4.6 | BMS Logic | Missing input | No detection | Protection lost | Flame instability | DU | 8 | 3 | 7 | 168 | Input monitoring |
| 4.7 | Fuel Valve | Fail to close | Fuel continues | Flame instability | Fire risk | DU | 9 | 2 | 8 | 144 | Proof testing |
| 4.8 | Fuel Regulator | Stuck position | Cannot control | Pressure varies | Flame instability | DD | 6 | 4 | 4 | 96 | Position feedback |

### 6.3 Summary - Low Fuel Pressure Cutoff

| Failure Category | Count | Total Failure Rate |
|------------------|-------|-------------------|
| Safe Detected (SD) | 1 | 3.2 x 10^-6 /hr |
| Safe Undetected (SU) | 0 | 0 |
| Dangerous Detected (DD) | 1 | 2.8 x 10^-6 /hr |
| Dangerous Undetected (DU) | 6 | 8.5 x 10^-6 /hr |

---

## 7. Safety Function 5: TMT High Alarm (GL-007)

### 7.1 Function Description

**Safety Function:** TMT High Temperature Alarm/Trip
**Agent:** GL-007 Furnace Performance Monitor
**SIL:** SIL 1 (Alarm), SIL 2 (Trip)
**Response Time:** <10 seconds (Alarm), <5 seconds (Trip)

### 7.2 FMEA Worksheet - TMT High Alarm

| ID | Component | Failure Mode | Effect (Local) | Effect (System) | Effect (Plant) | Class | S | O | D | RPN | Action |
|----|-----------|--------------|----------------|-----------------|----------------|-------|---|---|---|-----|--------|
| 5.1 | TC-401 | Open circuit | Loss of reading | Bad quality flag | Alarm, degraded | SD | 5 | 4 | 2 | 40 | Open-wire detection |
| 5.2 | TC-401 | Drift low | Under-reads TMT | Misses high temp | Tube damage | DU | 9 | 4 | 8 | 288 | Comparison, IR backup |
| 5.3 | TC-401 | Detached from tube | Wrong reading | Reads furnace temp | Misses tube overheat | DU | 9 | 3 | 9 | 243 | Periodic IR scan |
| 5.4 | TC-401 | Extension wire fault | Wrong reading | Incorrect TMT | Possible damage | DU | 8 | 3 | 7 | 168 | Wiring inspection |
| 5.5 | Cold Junction | Reference error | All TCs offset | Systematic error | Protection offset | DU | 7 | 3 | 6 | 126 | CJ compensation check |
| 5.6 | GL-007 Agent | Averaging error | Wrong aggregate | Masks hot spots | Localized damage | DU | 8 | 2 | 7 | 112 | Per-TC alarming |
| 5.7 | GL-007 Agent | Alarm suppressed | No notification | Operator unaware | Potential damage | DU | 8 | 3 | 6 | 144 | Suppression tracking |
| 5.8 | Alarm System | Flood condition | Alarm buried | Delayed response | Potential damage | DD | 7 | 4 | 4 | 112 | Alarm rationalization |
| 5.9 | Alarm System | Display failure | No visualization | Delayed awareness | Potential damage | DD | 6 | 3 | 3 | 54 | Redundant display |
| 5.10 | SIS Trip | Fails to operate | No trip | Tube damage | Rupture risk | DU | 10 | 2 | 8 | 160 | Proof testing |

### 7.3 Summary - TMT High Alarm

| Failure Category | Count | Total Failure Rate |
|------------------|-------|-------------------|
| Safe Detected (SD) | 1 | 2.5 x 10^-6 /hr |
| Safe Undetected (SU) | 0 | 0 |
| Dangerous Detected (DD) | 2 | 4.2 x 10^-6 /hr |
| Dangerous Undetected (DU) | 7 | 1.2 x 10^-5 /hr |

**Top RPN Failure Modes:**
1. TC drift low (RPN 288) - Add comparison logic and IR backup
2. TC detached from tube (RPN 243) - Periodic IR scanning
3. Extension wire fault (RPN 168) - Wiring inspection program

---

## 8. Safety Function 6: Loss of Air Flow Response (GL-018)

### 8.1 Function Description

**Safety Function:** Loss of Combustion Air Trip
**Agent:** GL-018 UnifiedCombustion Optimizer
**SIL:** SIL 2
**Response Time:** <3 seconds
**Voting Logic:** 1oo2

### 8.2 FMEA Worksheet - Loss of Air Flow Response

| ID | Component | Failure Mode | Effect (Local) | Effect (System) | Effect (Plant) | Class | S | O | D | RPN | Action |
|----|-----------|--------------|----------------|-----------------|----------------|-------|---|---|---|-----|--------|
| 6.1 | FT-501A | Drift high | Over-reports flow | Misses low flow | Rich combustion | DU | 8 | 4 | 7 | 224 | Comparison logic |
| 6.2 | FT-501A | Drift low | Under-reports flow | Spurious trip | Production loss | SD | 4 | 4 | 2 | 32 | Calibration |
| 6.3 | FT-501A | Plugged sensing | Stuck low | Spurious trip | Production loss | SD | 4 | 4 | 3 | 48 | Sensing line check |
| 6.4 | FT-501B | Same as A | Same as A | Same as A | Same as A | - | - | - | - | - | - |
| 6.5 | Air Damper | Stuck closed | No air | Rich combustion, flame out | Explosion risk | DU | 10 | 3 | 6 | 180 | Position feedback |
| 6.6 | Air Damper | Stuck open | High air | Lean, flame lift | Flameout risk | DD | 6 | 3 | 4 | 72 | Position feedback |
| 6.7 | FD Fan | Motor failure | No air flow | Trip required | Production loss | SD | 5 | 3 | 2 | 30 | Vibration monitoring |
| 6.8 | FD Fan | Bearing failure | Degraded flow | Reduced capacity | Efficiency loss | DD | 5 | 4 | 4 | 80 | Vibration monitoring |
| 6.9 | GL-018 Agent | Cross-limit failure | No lead-air | Rich combustion | Explosion risk | DU | 9 | 2 | 7 | 126 | Logic testing |
| 6.10 | BMS Interlock | Bypass active | No protection | Risk of rich operation | Explosion risk | DU | 10 | 3 | 5 | 150 | Bypass tracking |

### 8.3 Summary - Loss of Air Flow Response

| Failure Category | Count | Total Failure Rate |
|------------------|-------|-------------------|
| Safe Detected (SD) | 3 | 7.5 x 10^-6 /hr |
| Safe Undetected (SU) | 0 | 0 |
| Dangerous Detected (DD) | 2 | 4.8 x 10^-6 /hr |
| Dangerous Undetected (DU) | 5 | 9.2 x 10^-6 /hr |

---

## 9. Safety Function 7: Emergency Shutdown Interface (GL-001)

### 9.1 Function Description

**Safety Function:** SIF-005 Emergency Shutdown System
**Agent:** GL-001 ThermalCommand Orchestrator
**SIL:** SIL 2
**Response Time:** <500ms
**Architecture:** Hardwired + Software (1oo2)

### 9.2 FMEA Worksheet - Emergency Shutdown Interface

| ID | Component | Failure Mode | Effect (Local) | Effect (System) | Effect (Plant) | Class | S | O | D | RPN | Action |
|----|-----------|--------------|----------------|-----------------|----------------|-------|---|---|---|-----|--------|
| 7.1 | ESD Push Button | Stuck | Cannot initiate ESD | Manual ESD lost | Manual backup lost | DU | 7 | 2 | 5 | 70 | Periodic testing |
| 7.2 | ESD Push Button | Spurious activation | False ESD | Plant shutdown | Production loss | SD | 4 | 3 | 1 | 12 | Covered, labeled |
| 7.3 | GL-001 Agent | ESD message lost | No software shutdown | Hardwired intact | Protection intact | SD | 4 | 3 | 2 | 24 | Redundant path |
| 7.4 | GL-001 Agent | Broadcast failure | Agents not notified | Local trips only | Partial shutdown | DD | 6 | 3 | 3 | 54 | Message confirmation |
| 7.5 | GL-001 Agent | ESD logic error | Wrong equipment | Incomplete ESD | Partial hazard | DU | 8 | 2 | 7 | 112 | Logic validation |
| 7.6 | ESD Relay | Fail to de-energize | No trip output | Protection lost | Hazard continues | DU | 10 | 2 | 8 | 160 | Proof testing |
| 7.7 | ESD Relay | Welded contacts | Cannot trip | Protection lost | Hazard continues | DU | 10 | 2 | 8 | 160 | Proof testing |
| 7.8 | Hardwired Bus | Open circuit | Loss of hardwired | Software only | Degraded reliability | DD | 6 | 3 | 2 | 36 | Line monitoring |
| 7.9 | Hardwired Bus | Short circuit | Spurious trip | Plant shutdown | Production loss | SD | 4 | 3 | 2 | 24 | Fusing/isolation |
| 7.10 | Final Elements | Fail to operate | Equipment continues | Hazard continues | Fire/explosion | DU | 10 | 2 | 8 | 160 | Proof testing |
| 7.11 | Reset Circuit | Stuck | Cannot restart | Extended downtime | Production loss | SD | 3 | 3 | 3 | 27 | Reset verification |
| 7.12 | Reset Circuit | Premature reset | Early restart | Hazard may persist | Reoccurrence | DU | 8 | 3 | 5 | 120 | Reset sequencing |

### 9.3 Summary - Emergency Shutdown Interface

| Failure Category | Count | Total Failure Rate |
|------------------|-------|-------------------|
| Safe Detected (SD) | 4 | 1.1 x 10^-5 /hr |
| Safe Undetected (SU) | 0 | 0 |
| Dangerous Detected (DD) | 2 | 3.8 x 10^-6 /hr |
| Dangerous Undetected (DU) | 6 | 9.5 x 10^-6 /hr |

---

## 10. Diagnostic Coverage Calculations

### 10.1 Diagnostic Coverage per IEC 61508

```
DC = lambda_DD / (lambda_DD + lambda_DU)

Where:
  lambda_DD = Dangerous Detected failure rate
  lambda_DU = Dangerous Undetected failure rate
```

### 10.2 DC Summary by Safety Function

| Safety Function | lambda_DD | lambda_DU | DC | Target |
|-----------------|-----------|-----------|----|----|
| SIF-001 High Temp Trip | 4.5E-06 | 8.2E-06 | 35% | 60% |
| SIF-004 Flame Failure | 3.2E-06 | 1.1E-05 | 23% | 60% |
| SIF-003 High Pressure | 1.5E-06 | 9.8E-06 | 13% | 60% |
| Low Fuel Pressure | 2.8E-06 | 8.5E-06 | 25% | 60% |
| TMT High Alarm | 4.2E-06 | 1.2E-05 | 26% | 60% |
| Loss of Air Flow | 4.8E-06 | 9.2E-06 | 34% | 60% |
| SIF-005 ESD Interface | 3.8E-06 | 9.5E-06 | 29% | 60% |

### 10.3 DC Improvement Recommendations

| Safety Function | Current DC | Required Actions to Achieve 60% DC |
|-----------------|------------|-----------------------------------|
| All Functions | 13-35% | Add sensor comparison diagnostics |
| All Functions | | Implement continuous self-test |
| Flame Failure | 23% | Add scanner self-check with internal shutter |
| Pressure | 13% | Add impulse line block/bleed diagnostics |
| TMT | 26% | Add TC comparison and IR verification |

---

## 11. Common Cause Failure Analysis

### 11.1 Beta Factor Methodology

Per IEC 61508-6, common cause failures are modeled using the beta factor:

```
lambda_CCF = beta x lambda_DU

Where beta is determined by assessment of:
- Separation/segregation
- Diversity
- Complexity
- Assessment/analysis
- Procedures
- Competence
- Environmental control
- Environmental testing
```

### 11.2 Beta Factor Assessment

| Factor | Score | Weight | Contribution |
|--------|-------|--------|--------------|
| Separation/Segregation | 1.0 | 2.5 | 2.5 |
| Diversity | 1.0 | 2.0 | 2.0 |
| Complexity | 0.5 | 1.0 | 0.5 |
| Assessment | 1.0 | 1.5 | 1.5 |
| Procedures | 0.5 | 1.0 | 0.5 |
| Competence | 1.0 | 1.5 | 1.5 |
| Environmental Control | 1.0 | 1.5 | 1.5 |
| Environmental Testing | 0.5 | 1.0 | 0.5 |
| **Total** | | | **10.5** |

```
Beta = 10% (for score <= 20)
Beta = 5% (for score 20-35)
Beta = 2% (for score 35-45)
Beta = 1% (for score > 45)

Result: Beta = 5% (based on score = 10.5)
```

### 11.3 CCF Scenarios Identified

| CCF ID | Description | Affected Functions | Mitigation |
|--------|-------------|-------------------|------------|
| CCF-01 | Common power supply failure | All SIS functions | UPS, redundant supplies |
| CCF-02 | EMI affecting all sensors | Temperature, pressure functions | Shielding, filtering |
| CCF-03 | Network failure | Software-based functions | Hardwired backup |
| CCF-04 | Environmental (heat/humidity) | All field devices | Environmental control |
| CCF-05 | Calibration error (same tech) | Similar sensor types | Diverse calibration sources |
| CCF-06 | Software bug in GL agents | All GL functions | Diverse coding, validation |

---

## 12. Safe Failure Fraction Calculations

### 12.1 SFF Calculation per IEC 61508

```
SFF = (lambda_SD + lambda_SU + lambda_DD) / (lambda_SD + lambda_SU + lambda_DD + lambda_DU)

SFF Requirement by SIL (Type B devices):
- SIL 1: SFF >= 60%
- SIL 2: SFF >= 90% (or 60% with HFT=1)
- SIL 3: SFF >= 99% (or 90% with HFT=1, 60% with HFT=2)
```

### 12.2 SFF Summary by Safety Function

| Safety Function | lambda_S | lambda_DD | lambda_DU | SFF | Required | Gap |
|-----------------|----------|-----------|-----------|-----|----------|-----|
| SIF-001 High Temp | 1.2E-05 | 4.5E-06 | 8.2E-06 | 67% | 60% (HFT=1) | OK |
| SIF-004 Flame | 9.5E-06 | 3.2E-06 | 1.1E-05 | 54% | 60% (HFT=1) | -6% |
| SIF-003 Pressure | 8.2E-06 | 1.5E-06 | 9.8E-06 | 50% | 60% (HFT=1) | -10% |
| Low Fuel Press. | 3.2E-06 | 2.8E-06 | 8.5E-06 | 41% | 60% (HFT=1) | -19% |
| TMT Alarm | 2.5E-06 | 4.2E-06 | 1.2E-05 | 36% | 60% (HFT=0) | -24% |
| Air Flow | 7.5E-06 | 4.8E-06 | 9.2E-06 | 57% | 60% (HFT=1) | -3% |
| ESD Interface | 1.1E-05 | 3.8E-06 | 9.5E-06 | 62% | 60% (HFT=1) | OK |

### 12.3 SFF Improvement Actions

| Safety Function | Current SFF | Action Required |
|-----------------|-------------|-----------------|
| SIF-004 Flame Failure | 54% | Add scanner self-diagnostics (target: 65%) |
| SIF-003 Pressure | 50% | Add impulse line diagnostics (target: 65%) |
| Low Fuel Pressure | 41% | Add pressure switch diagnostics (target: 65%) |
| TMT Alarm | 36% | Add TC diagnostics, IR backup (target: 65%) |
| Air Flow | 57% | Minor improvements (target: 62%) |

---

## 13. Summary and Conclusions

### 13.1 Overall Risk Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total Safety Functions | 7 | - | - |
| Average DC | 26% | 60% | Gap |
| Average SFF | 52% | 60% | Gap |
| Total DU Failure Rate | 6.5E-05 /hr | <1E-05 /hr | Gap |
| Critical RPNs (>200) | 4 | 0 | Action |

### 13.2 Top 10 Failure Modes by RPN

| Rank | Function | Failure Mode | RPN | Action |
|------|----------|--------------|-----|--------|
| 1 | TMT Alarm | TC drift low | 288 | Comparison logic, IR |
| 2 | Pressure | Sensor drift low | 256 | Comparison diagnostics |
| 3 | TMT Alarm | TC detached | 243 | IR scanning program |
| 4 | Air Flow | Sensor drift high | 224 | Comparison logic |
| 5 | Temp Trip | Sensor drift low | 224 | Comparison diagnostics |
| 6 | Pressure | Plugged impulse line | 224 | Line blowdown |
| 7 | Low Fuel | Sensor drift high | 224 | Comparison logic |
| 8 | Flame | Valve leak through | 216 | Leak testing |
| 9 | Air Flow | Stuck damper | 180 | Position feedback |
| 10 | Flame | Scanner false flame | 180 | Self-check |

### 13.3 Key Recommendations

1. **Implement Sensor Comparison Diagnostics**
   - Add cross-channel comparison for all redundant sensors
   - Alert on deviation >2% between channels
   - Expected DC improvement: +20-30%

2. **Establish Proof Testing Program**
   - Annual proof test for all SIF devices
   - Partial proof test quarterly
   - Document all test results

3. **Add Self-Check Features**
   - Flame scanner internal shutter test
   - Pressure transmitter diagnostics
   - Thermocouple open-wire detection

4. **Implement Diverse Backup**
   - IR scanning as backup to thermocouples
   - Hardwired trip as backup to software
   - Manual override capability

5. **Address High RPN Items**
   - TC drift: Add comparison and trending
   - Sensor plugging: Implement blowdown schedule
   - Valve leakage: Regular leak testing

### 13.4 Next Steps

| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| Implement sensor comparison | Controls | Q1 2026 | Critical |
| Establish proof test program | Operations | Q1 2026 | Critical |
| Add scanner self-check | Instrumentation | Q1 2026 | High |
| IR scanning program | Inspection | Q2 2026 | High |
| Valve leak testing | Maintenance | Q1 2026 | High |
| Update PFD calculations | Safety | Q2 2026 | High |

---

**Document End**

*This document is part of the GreenLang Process Heat Safety Documentation Package.*
