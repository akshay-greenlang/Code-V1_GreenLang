# Safety Requirements Specification (SRS)

## GreenLang Process Heat Agents - IEC 61511 Compliant SRS

**Document ID:** GL-SIL-SRS-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-1:2016, Clause 10

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Safety Instrumented Functions](#2-safety-instrumented-functions)
3. [SIL Target Determination](#3-sil-target-determination)
4. [Response Time Requirements](#4-response-time-requirements)
5. [Proof Test Intervals](#5-proof-test-intervals)
6. [Safe State Definitions](#6-safe-state-definitions)
7. [Failure Modes and Actions](#7-failure-modes-and-actions)
8. [Performance Requirements](#8-performance-requirements)
9. [Interface Requirements](#9-interface-requirements)
10. [Testing Requirements](#10-testing-requirements)

---

## 1. Introduction

### 1.1 Purpose

This Safety Requirements Specification (SRS) defines the functional and integrity requirements for Safety Instrumented Functions (SIFs) implemented within the GreenLang Process Heat agent system. This document serves as the basis for design, implementation, validation, and verification activities per IEC 61511-1:2016.

### 1.2 Scope

This SRS covers all SIFs associated with:

- **GL-001:** Thermal Command Agent (SIL 2)
- **GL-005:** Building Energy Agent (SIL 1)
- **GL-007:** EU Taxonomy Agent (SIL 1)

### 1.3 Document References

| Document | Title |
|----------|-------|
| GL-SIL-OV-001 | SIL Certification Overview |
| GL-SIL-LOPA-001 | LOPA Analysis |
| GL-HAZOP-001 | Process Heat System HAZOP |
| IEC 61511-1:2016 | Functional Safety - SIS Part 1 |

### 1.4 Definitions

| Term | Definition |
|------|------------|
| Process Safety Time | Time between hazardous event initiation and consequence occurrence |
| SIF Response Time | Time from demand detection to safe state achievement |
| Spurious Trip Rate | Frequency of unwarranted SIF activations |
| Diagnostic Coverage | Fraction of dangerous failures detected by diagnostics |

---

## 2. Safety Instrumented Functions

### 2.1 SIF Summary Table

| SIF ID | SIF Name | Agent | SIL | Description |
|--------|----------|-------|-----|-------------|
| SIF-001 | High Temperature Shutdown | GL-001 | SIL 2 | Initiates shutdown on high process temperature |
| SIF-002 | Low Flow Protection | GL-001 | SIL 2 | Protects against low cooling flow conditions |
| SIF-003 | Pressure Relief Monitoring | GL-001 | SIL 2 | Monitors pressure relief system status |
| SIF-004 | Flame Failure Detection | GL-001 | SIL 2 | Detects loss of flame in combustion systems |
| SIF-005 | Emergency Shutdown | GL-001 | SIL 2 | Master emergency shutdown function |
| SIF-006 | Ventilation Fault Detection | GL-005 | SIL 1 | Monitors building ventilation adequacy |
| SIF-007 | CO/CO2 High Alert | GL-005 | SIL 1 | Alerts on high CO/CO2 concentrations |
| SIF-008 | Emission Threshold Violation | GL-007 | SIL 1 | Detects regulatory emission exceedances |

---

### 2.2 SIF-001: High Temperature Shutdown (HTS)

#### 2.2.1 Function Description

**Purpose:** Prevent equipment damage and personnel injury by initiating automatic shutdown when process temperature exceeds safe operating limits.

**Initiating Cause:** Process temperature exceeds High-High setpoint (THH).

**Safe State:** Fuel shutoff valves closed, process isolation valves closed, cooling systems activated.

#### 2.2.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-001-FR-01 | The SIF shall monitor process temperature continuously | Design review, Testing |
| SIF-001-FR-02 | The SIF shall initiate shutdown when T >= THH setpoint | FAT, SAT |
| SIF-001-FR-03 | The SIF shall close fuel shutoff valve(s) upon activation | FAT, SAT |
| SIF-001-FR-04 | The SIF shall close process isolation valve(s) upon activation | FAT, SAT |
| SIF-001-FR-05 | The SIF shall activate cooling system upon shutdown | FAT, SAT |
| SIF-001-FR-06 | The SIF shall annunciate HTS condition on HMI | FAT, SAT |
| SIF-001-FR-07 | The SIF shall log all activations with timestamp | Testing |
| SIF-001-FR-08 | Manual reset shall be required to restart | FAT, SAT |

#### 2.2.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis |
| PFDavg | < 1 x 10^-2 | SIL 2 requirement |
| Response Time | < 500 ms | Process Safety Time analysis |
| Setpoint Accuracy | +/- 1% of span | Measurement accuracy requirement |
| Spurious Trip Rate | < 0.1 per year | Operational availability |

#### 2.2.4 Architecture

```
                          +-------------------+
                          |   GL-001 Agent    |
                          |  Thermal Command  |
                          +-------------------+
                                   |
                          +--------v--------+
     +----------------+   |   Temperature   |   +----------------+
     | Sensor 1 (TE-1)|-->|   Comparison    |<--| Sensor 2 (TE-2)|
     +----------------+   |   (2oo2 Vote)   |   +----------------+
                          +-----------------+
                                   |
                          +--------v--------+
                          |  Shutdown Logic |
                          |  (T >= THH)     |
                          +-----------------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
+---------v---------+  +-----------v----------+  +----------v----------+
| Fuel Shutoff      |  | Process Isolation    |  | Cooling Activation  |
| Valve (XV-001)    |  | Valve (XV-002)       |  | System              |
+-------------------+  +----------------------+  +---------------------+
```

#### 2.2.5 Input/Output Specification

**Inputs:**

| Tag | Description | Signal Type | Range | Fail State |
|-----|-------------|-------------|-------|------------|
| TE-001A | Process Temperature Sensor A | 4-20 mA | 0-500 C | Low (fail-safe) |
| TE-001B | Process Temperature Sensor B | 4-20 mA | 0-500 C | Low (fail-safe) |
| THH_SP | High-High Temperature Setpoint | Internal | 0-500 C | N/A |

**Outputs:**

| Tag | Description | Signal Type | Action | Fail State |
|-----|-------------|-------------|--------|------------|
| XV-001 | Fuel Shutoff Valve | 24 VDC | De-energize to close | Closed (fail-safe) |
| XV-002 | Process Isolation Valve | 24 VDC | De-energize to close | Closed (fail-safe) |
| YA-001 | HTS Alarm | Digital | Energize on trip | Alarmed |

---

### 2.3 SIF-002: Low Flow Protection (LFP)

#### 2.3.1 Function Description

**Purpose:** Prevent thermal damage from inadequate cooling/heat transfer fluid flow.

**Initiating Cause:** Process flow falls below Low-Low setpoint (FLL).

**Safe State:** Heat source isolated, process in safe cooldown mode.

#### 2.3.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-002-FR-01 | The SIF shall monitor process flow continuously | Design review, Testing |
| SIF-002-FR-02 | The SIF shall initiate protection when F <= FLL setpoint | FAT, SAT |
| SIF-002-FR-03 | The SIF shall isolate heat source upon activation | FAT, SAT |
| SIF-002-FR-04 | Time delay (5 sec) shall be applied to avoid spurious trips | Design review |
| SIF-002-FR-05 | The SIF shall annunciate LFP condition on HMI | FAT, SAT |
| SIF-002-FR-06 | Manual reset shall be required to restart | FAT, SAT |

#### 2.3.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis |
| PFDavg | < 1 x 10^-2 | SIL 2 requirement |
| Response Time | < 500 ms (excluding time delay) | Process Safety Time analysis |
| Setpoint Accuracy | +/- 2% of span | Measurement accuracy requirement |
| Time Delay | 5 +/- 0.5 seconds | Anti-spurious trip requirement |

#### 2.3.4 Architecture

```
                          +-------------------+
                          |   GL-001 Agent    |
                          |  Thermal Command  |
                          +-------------------+
                                   |
                          +--------v--------+
     +----------------+   |     Flow        |   +----------------+
     | Sensor 1 (FT-1)|-->|   Comparison    |<--| Sensor 2 (FT-2)|
     +----------------+   |   (2oo2 Vote)   |   +----------------+
                          +-----------------+
                                   |
                          +--------v--------+
                          |  Time Delay     |
                          |  (5 seconds)    |
                          +-----------------+
                                   |
                          +--------v--------+
                          | Protection Logic|
                          |  (F <= FLL)     |
                          +-----------------+
                                   |
                          +--------v--------+
                          | Heat Source     |
                          | Isolation       |
                          +-----------------+
```

---

### 2.4 SIF-003: Pressure Relief Monitoring (PRM)

#### 2.4.1 Function Description

**Purpose:** Monitor pressure relief device status and initiate protective action on pressure relief activation.

**Initiating Cause:** Pressure relief device discharge detected or high pressure condition.

**Safe State:** Process depressurized, upstream isolation, alarm activated.

#### 2.4.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-003-FR-01 | The SIF shall monitor relief device position/flow | Design review, Testing |
| SIF-003-FR-02 | The SIF shall detect pressure relief activation | FAT, SAT |
| SIF-003-FR-03 | The SIF shall initiate controlled depressurization | FAT, SAT |
| SIF-003-FR-04 | The SIF shall close upstream isolation valves | FAT, SAT |
| SIF-003-FR-05 | The SIF shall annunciate relief condition on HMI | FAT, SAT |

#### 2.4.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis |
| PFDavg | < 1 x 10^-2 | SIL 2 requirement |
| Response Time | < 500 ms | Process Safety Time analysis |
| Detection Accuracy | 100% of relief activations | Safety requirement |

---

### 2.5 SIF-004: Flame Failure Detection (FFD)

#### 2.5.1 Function Description

**Purpose:** Detect loss of flame in combustion systems and initiate fuel shutoff to prevent unburned fuel accumulation.

**Initiating Cause:** Flame scanner detects loss of flame signal.

**Safe State:** Fuel supply isolated, purge sequence initiated.

#### 2.5.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-004-FR-01 | The SIF shall monitor flame presence continuously | Design review, Testing |
| SIF-004-FR-02 | The SIF shall detect flame failure within 1 second | FAT, SAT |
| SIF-004-FR-03 | The SIF shall close fuel shutoff valve on flame loss | FAT, SAT |
| SIF-004-FR-04 | The SIF shall initiate combustion air purge | FAT, SAT |
| SIF-004-FR-05 | Flame proving shall be required before restart | FAT, SAT |
| SIF-004-FR-06 | The SIF shall comply with NFPA 85/86 requirements | Design review |

#### 2.5.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis, NFPA 85/86 |
| PFDavg | < 1 x 10^-2 | SIL 2 requirement |
| Flame Failure Response Time | < 1 second | NFPA 85 requirement |
| Maximum Fuel Valve Closure Time | < 3 seconds | NFPA 86 requirement |

#### 2.5.4 NFPA Compliance Matrix

| NFPA Requirement | Section | Implementation |
|------------------|---------|----------------|
| Flame supervision | NFPA 85 5.3.6 | UV/IR flame scanner with self-check |
| Safety shutoff valves | NFPA 86 8.5.1 | Dual block and bleed arrangement |
| Purge requirements | NFPA 85 5.5 | 4 volume air changes minimum |
| Trial for ignition | NFPA 85 5.3.7 | 15 seconds maximum |

---

### 2.6 SIF-005: Emergency Shutdown (ESD)

#### 2.6.1 Function Description

**Purpose:** Provide master emergency shutdown capability for the entire process heat system.

**Initiating Cause:** Manual activation (pushbutton), automatic trip from other SIFs, or external ESD signal.

**Safe State:** All energy sources isolated, all process isolation valves closed, system in de-energized state.

#### 2.6.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-005-FR-01 | The SIF shall accept manual ESD pushbutton input | FAT, SAT |
| SIF-005-FR-02 | The SIF shall accept automatic trip inputs from SIF-001 to SIF-004 | FAT, SAT |
| SIF-005-FR-03 | The SIF shall accept external ESD input signal | FAT, SAT |
| SIF-005-FR-04 | The SIF shall isolate all fuel/energy sources | FAT, SAT |
| SIF-005-FR-05 | The SIF shall close all process isolation valves | FAT, SAT |
| SIF-005-FR-06 | The SIF shall maintain safe state until manual reset | FAT, SAT |
| SIF-005-FR-07 | The SIF shall provide trip status to supervisory system | Testing |
| SIF-005-FR-08 | Reset shall require physical key switch | FAT, SAT |

#### 2.6.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis |
| PFDavg | < 1 x 10^-2 | SIL 2 requirement |
| Response Time | < 500 ms | Process Safety Time analysis |
| Availability | > 99.9% | Operational requirement |

#### 2.6.4 ESD Matrix

| Input | SIF-001 | SIF-002 | SIF-003 | SIF-004 | Manual | External |
|-------|---------|---------|---------|---------|--------|----------|
| Fuel Isolation | X | X | X | X | X | X |
| Process Isolation | X | X | X | | X | X |
| Cooling Activation | X | | | | X | |
| Purge Sequence | | | | X | X | |
| Full Shutdown | X | X | X | X | X | X |

---

### 2.7 SIF-006: Ventilation Fault Detection (VFD)

#### 2.7.1 Function Description

**Purpose:** Detect ventilation system faults in building applications and activate emergency ventilation.

**Initiating Cause:** Ventilation system fault, air flow below minimum, or differential pressure anomaly.

**Safe State:** Emergency ventilation activated, building management system alerted.

#### 2.7.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-006-FR-01 | The SIF shall monitor ventilation system status | Design review, Testing |
| SIF-006-FR-02 | The SIF shall detect air flow below minimum setpoint | FAT, SAT |
| SIF-006-FR-03 | The SIF shall activate emergency ventilation on fault | FAT, SAT |
| SIF-006-FR-04 | The SIF shall alert building management system | Testing |
| SIF-006-FR-05 | The SIF shall log all fault conditions | Testing |

#### 2.7.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 1 | LOPA Analysis |
| PFDavg | < 1 x 10^-1 | SIL 1 requirement |
| Response Time | < 2000 ms | Building safety analysis |
| Detection Accuracy | 95% of faults | Performance requirement |

---

### 2.8 SIF-007: CO/CO2 High Alert

#### 2.8.1 Function Description

**Purpose:** Alert on high carbon monoxide or carbon dioxide concentrations in occupied spaces.

**Initiating Cause:** CO or CO2 concentration exceeds alarm setpoint.

**Safe State:** Alarm activated, emergency ventilation triggered, evacuation signal (if applicable).

#### 2.8.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-007-FR-01 | The SIF shall monitor CO concentration continuously | Design review, Testing |
| SIF-007-FR-02 | The SIF shall monitor CO2 concentration continuously | Design review, Testing |
| SIF-007-FR-03 | The SIF shall alarm at CO >= 35 ppm | FAT, SAT |
| SIF-007-FR-04 | The SIF shall alarm at CO2 >= 5000 ppm | FAT, SAT |
| SIF-007-FR-05 | The SIF shall activate emergency ventilation at high-high levels | FAT, SAT |
| SIF-007-FR-06 | The SIF shall log all alarm events | Testing |

#### 2.8.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 1 | LOPA Analysis |
| PFDavg | < 1 x 10^-1 | SIL 1 requirement |
| Response Time | < 2000 ms | Occupant safety analysis |
| CO Alarm Setpoint | 35 ppm (High), 100 ppm (High-High) | OSHA/NIOSH limits |
| CO2 Alarm Setpoint | 5000 ppm (High), 30000 ppm (High-High) | OSHA limits |

---

### 2.9 SIF-008: Emission Threshold Violation Detection

#### 2.9.1 Function Description

**Purpose:** Detect exceedances of regulatory emission thresholds and alert compliance personnel.

**Initiating Cause:** Calculated or measured emissions exceed regulatory limits.

**Safe State:** Alert generated, data logging activated, compliance notification sent.

#### 2.9.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-008-FR-01 | The SIF shall calculate emissions from process data | Design review, Testing |
| SIF-008-FR-02 | The SIF shall compare emissions to regulatory limits | Testing |
| SIF-008-FR-03 | The SIF shall alert on threshold exceedance | FAT, SAT |
| SIF-008-FR-04 | The SIF shall activate enhanced data logging | Testing |
| SIF-008-FR-05 | The SIF shall notify compliance personnel | Testing |
| SIF-008-FR-06 | The SIF shall maintain audit trail | Testing |

#### 2.9.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 1 | LOPA Analysis |
| PFDavg | < 1 x 10^-1 | SIL 1 requirement |
| Response Time | < 5000 ms | Compliance reporting requirement |
| Data Integrity | 99.9% accuracy | Regulatory requirement |

---

## 3. SIL Target Determination

### 3.1 Methodology

SIL targets were determined using the Layer of Protection Analysis (LOPA) method per IEC 61511-3:2016, Annex F.

### 3.2 SIL Determination Summary

| SIF | Initiating Event | IE Frequency | Target TMEL | IPL Credits | Required PFD | SIL |
|-----|------------------|--------------|-------------|-------------|--------------|-----|
| SIF-001 | High temperature | 0.1/yr | 1E-05/yr | 2 (BPCS, Alarm) | 1E-02 | SIL 2 |
| SIF-002 | Low flow | 0.1/yr | 1E-05/yr | 2 (BPCS, Alarm) | 1E-02 | SIL 2 |
| SIF-003 | Over-pressure | 0.05/yr | 1E-05/yr | 2 (PRV, Alarm) | 1E-02 | SIL 2 |
| SIF-004 | Flame failure | 0.5/yr | 1E-05/yr | 1 (Alarm) | 2E-03 | SIL 2 |
| SIF-005 | Multiple causes | N/A | 1E-05/yr | Per SIF | 1E-02 | SIL 2 |
| SIF-006 | Vent fault | 0.2/yr | 1E-04/yr | 1 (Alarm) | 5E-02 | SIL 1 |
| SIF-007 | High CO/CO2 | 0.1/yr | 1E-04/yr | 1 (Alarm) | 5E-02 | SIL 1 |
| SIF-008 | Emission exceed | 0.2/yr | 1E-04/yr | 1 (Manual) | 5E-02 | SIL 1 |

See [02_LOPA_ANALYSIS.md](02_LOPA_ANALYSIS.md) for detailed LOPA calculations.

---

## 4. Response Time Requirements

### 4.1 Process Safety Time Analysis

Process Safety Time (PST) is the time from hazard initiation to consequence occurrence without protective action.

| SIF | Hazard Scenario | PST | Required Response Time | Allocated Response Time |
|-----|-----------------|-----|------------------------|------------------------|
| SIF-001 | Thermal runaway | 5 seconds | < 2 seconds | 500 ms |
| SIF-002 | Dry running | 10 seconds | < 5 seconds | 500 ms |
| SIF-003 | Over-pressure | 3 seconds | < 1 second | 500 ms |
| SIF-004 | Unburned fuel accumulation | 3 seconds | < 1 second | 500 ms |
| SIF-005 | Various | 3 seconds min | < 1 second | 500 ms |
| SIF-006 | Air quality degradation | 60 seconds | < 30 seconds | 2000 ms |
| SIF-007 | CO/CO2 accumulation | 60 seconds | < 30 seconds | 2000 ms |
| SIF-008 | Regulatory violation | 5 minutes | < 2 minutes | 5000 ms |

### 4.2 Response Time Budget

For SIL 2 functions (GL-001):

| Component | Allocation | Justification |
|-----------|------------|---------------|
| Sensor response | 100 ms | Sensor specification |
| Signal transmission | 50 ms | Communication latency |
| Agent processing | 100 ms | Software execution time |
| Output command | 50 ms | Communication latency |
| Final element response | 200 ms | Valve stroke time |
| **Total** | **500 ms** | Within 2 second requirement |

### 4.3 SIL 2 Response Time Verification

```
+-------------+     +-------------+     +-------------+     +-------------+     +-------------+
|   Sensor    |---->| Transmission|---->|    Agent    |---->|  Output     |---->|    Valve    |
|   100 ms    |     |   50 ms     |     |   100 ms    |     |   50 ms     |     |   200 ms    |
+-------------+     +-------------+     +-------------+     +-------------+     +-------------+
                                                                                        |
                                    Total: 500 ms                                       v
                                    Requirement: < 500 ms                          ACHIEVED
```

---

## 5. Proof Test Intervals

### 5.1 Proof Test Overview

Proof tests are periodic tests to detect dangerous undetected failures and verify SIF functionality.

### 5.2 Proof Test Interval Requirements

| SIF | SIL | PFDavg Target | DC Assumed | Ti (Proof Test Interval) |
|-----|-----|---------------|------------|--------------------------|
| SIF-001 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-002 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-003 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-004 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-005 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-006 | SIL 1 | < 1E-01 | 60% | 24 months |
| SIF-007 | SIL 1 | < 1E-01 | 60% | 12 months |
| SIF-008 | SIL 1 | < 1E-01 | 60% | 24 months |

### 5.3 PFDavg Calculation Basis

PFDavg for 1oo1 architecture:

```
PFDavg = (lambda_DU x Ti) / 2

Where:
  lambda_DU = Dangerous undetected failure rate
  Ti = Proof test interval
```

For SIL 2 with Ti = 12 months (8760 hours):
- Required: PFDavg < 1E-02
- Architecture: 1oo2 voting provides additional risk reduction

See [03_VOTING_LOGIC_SPECIFICATION.md](03_VOTING_LOGIC_SPECIFICATION.md) for detailed PFD calculations.

---

## 6. Safe State Definitions

### 6.1 Safe State Matrix

| SIF | Safe State Description | Energy State | Valve State | System State |
|-----|------------------------|--------------|-------------|--------------|
| SIF-001 | Thermal isolation | De-energized | Closed | Shutdown |
| SIF-002 | Heat source isolated | De-energized | Closed | Cooldown |
| SIF-003 | Depressurized | De-energized | Open (vent), Closed (isolation) | Depressurized |
| SIF-004 | Fuel isolated, purged | De-energized | Closed | Purged |
| SIF-005 | Full shutdown | De-energized | All closed | Shutdown |
| SIF-006 | Emergency vent active | Energized (vent) | N/A | Ventilating |
| SIF-007 | Alert, emergency vent | Energized (vent) | N/A | Alert |
| SIF-008 | Alert, logging active | Normal | N/A | Alert |

### 6.2 Fail-Safe Design Principles

| Principle | Implementation |
|-----------|----------------|
| Fail-safe on loss of power | Valves close on de-energize (spring return) |
| Fail-safe on signal loss | Low signal interpreted as trip condition |
| Fail-safe on communication loss | Watchdog timer triggers safe state |
| Fail-safe on software fault | Watchdog, memory checks, safe defaults |

---

## 7. Failure Modes and Actions

### 7.1 Input Failure Handling

| Failure Mode | Detection Method | Action |
|--------------|------------------|--------|
| Sensor open circuit | Signal < 4 mA | Trip (fail-safe) |
| Sensor short circuit | Signal > 20.5 mA | Trip (fail-safe) |
| Sensor drift | Cross-check with redundant sensor | Alarm, operator action |
| Communication loss | Watchdog timeout | Trip after timeout |
| Signal stuck | Rate of change check | Alarm, diagnostics |

### 7.2 Output Failure Handling

| Failure Mode | Detection Method | Action |
|--------------|------------------|--------|
| Valve stuck | Position feedback mismatch | Alarm, manual intervention |
| Partial stroke | Position feedback analysis | Alarm, maintenance |
| Coil failure | Current monitoring | Alarm, redundant output |
| Communication loss | Watchdog timeout | Maintain last safe state |

### 7.3 Processing Failure Handling

| Failure Mode | Detection Method | Action |
|--------------|------------------|--------|
| Watchdog timeout | Hardware watchdog | Reboot, trip if persistent |
| Memory corruption | CRC checks | Trip, diagnostic mode |
| CPU overload | Execution time monitoring | Alarm, load shedding |
| Configuration error | Startup validation | Block startup |

---

## 8. Performance Requirements

### 8.1 Reliability Requirements

| Parameter | SIL 2 Requirement | SIL 1 Requirement |
|-----------|-------------------|-------------------|
| PFDavg | < 1 x 10^-2 | < 1 x 10^-1 |
| Spurious trip rate (STR) | < 0.1/year | < 0.5/year |
| MTTR (Mean Time to Repair) | < 8 hours | < 24 hours |
| Availability | > 99.9% | > 99% |

### 8.2 Diagnostic Coverage Requirements

| Component | Required DC (SIL 2) | Diagnostic Method |
|-----------|---------------------|-------------------|
| Input sensors | >= 90% | Range check, comparison, rate of change |
| Logic solver | >= 90% | Watchdog, memory check, self-test |
| Output devices | >= 90% | Position feedback, current monitoring |
| Communication | >= 90% | CRC, heartbeat, timeout detection |

### 8.3 Common Cause Failure Requirements

| Factor | SIL 2 Requirement | Implementation |
|--------|-------------------|----------------|
| Beta factor | < 10% | Physical separation, diverse sensors |
| Diversity | Required | Different sensor principles where practical |
| Separation | Required | Separate cable routes, enclosures |
| Independence | Required | Independent power supplies |

---

## 9. Interface Requirements

### 9.1 Process Control System Interface

| Parameter | Requirement |
|-----------|-------------|
| Communication protocol | Modbus TCP/IP, OPC UA |
| Update rate | 100 ms minimum |
| Data integrity | CRC-32 error detection |
| Timeout handling | 2 second watchdog, trip on timeout |

### 9.2 Human-Machine Interface Requirements

| Function | Requirement |
|----------|-------------|
| SIF status display | Real-time, < 1 second update |
| Alarm display | Dedicated safety alarm area |
| Trip history | Accessible, timestamped log |
| Manual bypass indication | Flashing indicator, access-controlled |
| Reset interface | Key-switch or password protected |

### 9.3 External System Interfaces

| System | Interface | Purpose |
|--------|-----------|---------|
| Emergency Response | Hardwired contact | External ESD input |
| Fire & Gas System | Hardwired contact | Fire detection input |
| Building Management | Modbus TCP | Status exchange |
| Historian | OPC UA | Data logging |

---

## 10. Testing Requirements

### 10.1 Factory Acceptance Test (FAT)

All SIFs shall undergo FAT including:

| Test | Acceptance Criteria |
|------|---------------------|
| Functional test | All requirements in Section 2 verified |
| Response time test | < allocated time (see Section 4) |
| Failure mode test | Correct fail-safe behavior |
| Communication test | All interfaces operational |
| Documentation review | All documents complete and approved |

### 10.2 Site Acceptance Test (SAT)

All SIFs shall undergo SAT including:

| Test | Acceptance Criteria |
|------|---------------------|
| Loop check | All I/O points verified |
| Functional test | Trip at correct setpoints |
| Response time test | < allocated time with actual I/O |
| Integrated test | Correct interaction with BPCS, SIS |
| Documentation | Signed test records |

### 10.3 Proof Test Requirements

See [04_PROOF_TEST_PROCEDURES.md](04_PROOF_TEST_PROCEDURES.md) for detailed test procedures.

| SIF | Test Frequency | Test Coverage |
|-----|----------------|---------------|
| SIF-001 to SIF-005 | 12 months | 100% of function |
| SIF-006 to SIF-008 | 24 months (SIF-007: 12 months) | 100% of function |

---

## Appendix A: SRS Checklist per IEC 61511-1 Clause 10

| Clause | Requirement | Document Section | Status |
|--------|-------------|------------------|--------|
| 10.3.1 a) | Description of SIF including inputs, logic, outputs | Section 2 | Complete |
| 10.3.1 b) | Safe state definition | Section 6 | Complete |
| 10.3.1 c) | SIL for each SIF | Section 3 | Complete |
| 10.3.1 d) | Demand mode | Low demand (all) | Complete |
| 10.3.1 e) | Constraints on MTTR | Section 8.1 | Complete |
| 10.3.1 f) | Maximum spurious trip rate | Section 8.1 | Complete |
| 10.3.1 g) | Response time requirements | Section 4 | Complete |
| 10.3.1 h) | Process interface requirements | Section 9 | Complete |
| 10.3.1 i) | Modes of operation | Section 2.6 | Complete |
| 10.3.1 j) | Bypass requirements | Section 9.2 | Complete |
| 10.3.1 k) | Maintenance requirements | Section 5 | Complete |
| 10.3.1 l) | Test requirements | Section 10 | Complete |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
