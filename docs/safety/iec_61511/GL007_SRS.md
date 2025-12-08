# Safety Requirements Specification (SRS) - GL-007 Furnace Monitoring

## GreenLang Process Heat Agents - Furnace Performance Monitor

**Document ID:** GL-SIL-SRS-007
**Version:** 1.0
**Effective Date:** 2025-12-07
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-1:2016, Clause 10, API 560, NFPA 86

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Safety Function Definitions](#2-safety-function-definitions)
3. [Performance Requirements](#3-performance-requirements)
4. [Failure Mode Specifications](#4-failure-mode-specifications)
5. [Diagnostic Coverage Requirements](#5-diagnostic-coverage-requirements)
6. [Proof Test Requirements](#6-proof-test-requirements)
7. [Integration Requirements with GL-001](#7-integration-requirements-with-gl-001)
8. [Safe State Definitions](#8-safe-state-definitions)
9. [Verification and Validation](#9-verification-and-validation)
10. [Appendices](#10-appendices)

---

## 1. Introduction

### 1.1 Purpose

This Safety Requirements Specification (SRS) defines the functional and integrity requirements for Safety Instrumented Functions (SIFs) implemented within the GL-007 Furnace Performance Monitor agent. This document serves as the basis for design, implementation, validation, and verification activities per IEC 61511-1:2016.

### 1.2 Scope

This SRS covers all SIFs associated with:

- **Tube Metal Temperature (TMT) Monitoring** - Protection against overtemperature and tube failure
- **Flame Detection and Supervision** - Combustion safety per NFPA 85/86
- **Process Flow Protection** - Low flow protection for tube integrity
- **Fouling Detection** - Hot spot identification and alerting
- **Integration with GL-001** - Thermal Command Agent coordination

### 1.3 Document References

| Document | Title |
|----------|-------|
| GL-SIL-LOPA-007 | GL-007 LOPA Analysis |
| GL-HAZOP-007 | GL-007 Furnace HAZOP Study |
| GL-SIL-SRS-001 | GL-001 Safety Requirements Specification |
| IEC 61511-1:2016 | Functional Safety - SIS Part 1 |
| API 560 | Fired Heaters for General Refinery Service |
| API 530 | Calculation of Heater-Tube Thickness |
| NFPA 85 | Boiler and Combustion Systems Hazards Code |
| NFPA 86 | Standard for Ovens and Furnaces |

### 1.4 Definitions

| Term | Definition |
|------|------------|
| TMT | Tube Metal Temperature - measured at tube surface |
| Process Safety Time | Time from hazard initiation to consequence occurrence |
| Flame Failure Response Time | Time from flame loss to fuel shutoff |
| Bridgewall Temperature | Temperature between radiant and convection sections |
| Creep Rupture | Time-dependent deformation leading to tube failure |

---

## 2. Safety Function Definitions

### 2.1 SIF Summary Table

| SIF ID | SIF Name | SIL | Description |
|--------|----------|-----|-------------|
| SIF-007-01 | TMT High Shutdown | SIL 2 | Initiates shutdown on high tube metal temperature |
| SIF-007-02 | Flame Failure Detection | SIL 2 | Detects loss of flame and initiates fuel shutoff |
| SIF-007-03 | Low Flow Protection | SIL 2 | Protects against low process flow conditions |
| SIF-007-04 | Fouling Hot Spot Alert | SIL 1 | Detects localized overtemperature from fouling |
| SIF-007-05 | Emergency Shutdown | SIL 2 | Master emergency shutdown for furnace system |

---

### 2.2 SIF-007-01: TMT High Shutdown

#### 2.2.1 Function Description

**Purpose:** Prevent tube creep rupture and fire by initiating automatic shutdown when any monitored tube metal temperature exceeds the safe operating limit per API 530.

**Initiating Cause:** Tube metal temperature exceeds High-High setpoint (TMT_HH).

**Safe State:** Fuel supply isolated, process flow maintained for controlled cooldown, alarm activated.

#### 2.2.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-007-01-FR-01 | The SIF shall monitor TMT continuously at all critical tube locations | Design review, Testing |
| SIF-007-01-FR-02 | The SIF shall initiate shutdown when any TMT >= TMT_HH setpoint | FAT, SAT |
| SIF-007-01-FR-03 | The SIF shall close fuel shutoff valve(s) upon activation | FAT, SAT |
| SIF-007-01-FR-04 | The SIF shall maintain process flow for controlled cooldown | FAT, SAT |
| SIF-007-01-FR-05 | The SIF shall annunciate TMT High-High condition on HMI | FAT, SAT |
| SIF-007-01-FR-06 | The SIF shall log all activations with timestamp and TMT values | Testing |
| SIF-007-01-FR-07 | Manual reset shall be required to restart after trip | FAT, SAT |
| SIF-007-01-FR-08 | The SIF shall implement 2oo3 voting for redundant TMT sensors | Design review |

#### 2.2.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis |
| PFDavg | < 1E-02 | SIL 2 requirement |
| Response Time | < 3000 ms | Process Safety Time analysis |
| Setpoint Accuracy | +/- 1% of span | API 560 requirement |
| Spurious Trip Rate | < 0.1 per year | Operational availability |
| TMT Measurement Accuracy | +/- 2.2 degC or 0.75% | Thermocouple specification |

#### 2.2.4 Architecture

```
                          +---------------------+
                          |   GL-007 Agent      |
                          |  Furnace Monitor    |
                          +---------------------+
                                    |
                          +---------v---------+
     +----------------+   |     TMT           |   +----------------+
     | TMT Sensor A   |-->|   Comparison      |<--| TMT Sensor C   |
     | (TE-007-A)     |   |   (2oo3 Vote)     |   | (TE-007-C)     |
     +----------------+   +-------------------+   +----------------+
              |                   |                      |
              |           +-------+-------+              |
              |           |               |              |
              +-------+---|               |---+----------+
                      |   |   TMT Logic   |   |
     +----------------+   | (Any >= HH)   |   +----------------+
     | TMT Sensor B   |-->|               |
     | (TE-007-B)     |   +---------------+
     +----------------+           |
                          +-------v-------+
                          |   Shutdown    |
                          |   Sequence    |
                          +---------------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
   +--------v--------+  +---------v---------+  +-------v-------+
   | Fuel Shutoff    |  | Alarm Generation  |  | GL-001 Notify |
   | Valve (XV-007)  |  | (High-High TMT)   |  | (Coordination)|
   +-----------------+  +-------------------+  +---------------+
```

#### 2.2.5 Input/Output Specification

**Inputs:**

| Tag | Description | Signal Type | Range | Fail State |
|-----|-------------|-------------|-------|------------|
| TE-007-A | TMT Sensor A (Radiant Section) | 4-20 mA | 0-1600 degF | Low (fail-safe) |
| TE-007-B | TMT Sensor B (Radiant Section) | 4-20 mA | 0-1600 degF | Low (fail-safe) |
| TE-007-C | TMT Sensor C (Radiant Section) | 4-20 mA | 0-1600 degF | Low (fail-safe) |
| TMT_HH_SP | High-High TMT Setpoint | Internal | Per API 530 | N/A |

**Outputs:**

| Tag | Description | Signal Type | Action | Fail State |
|-----|-------------|-------------|--------|------------|
| XV-007-A | Fuel Shutoff Valve A | 24 VDC | De-energize to close | Closed (fail-safe) |
| XV-007-B | Fuel Shutoff Valve B | 24 VDC | De-energize to close | Closed (fail-safe) |
| YA-007-01 | TMT High-High Alarm | Digital | Energize on trip | Alarmed |

---

### 2.3 SIF-007-02: Flame Failure Detection

#### 2.3.1 Function Description

**Purpose:** Detect loss of flame in furnace burners and initiate fuel shutoff to prevent unburned fuel accumulation and explosion risk per NFPA 85/86.

**Initiating Cause:** Flame scanner detects loss of flame signal from any burner.

**Safe State:** Fuel supply isolated via dual block and bleed, purge sequence initiated.

#### 2.3.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-007-02-FR-01 | The SIF shall monitor flame presence continuously for all burners | Design review, Testing |
| SIF-007-02-FR-02 | The SIF shall detect flame failure within 1 second | FAT, SAT |
| SIF-007-02-FR-03 | The SIF shall close fuel shutoff valves within 3 seconds of detection | FAT, SAT |
| SIF-007-02-FR-04 | The SIF shall implement dual block and bleed fuel isolation | Design review, FAT |
| SIF-007-02-FR-05 | The SIF shall initiate combustion air purge sequence | FAT, SAT |
| SIF-007-02-FR-06 | Flame proving shall be required before restart | FAT, SAT |
| SIF-007-02-FR-07 | The SIF shall comply with NFPA 85/86 requirements | Design review |
| SIF-007-02-FR-08 | The SIF shall implement 2oo3 flame scanner voting | Design review |

#### 2.3.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis, NFPA 85/86 |
| PFDavg | < 1E-02 | SIL 2 requirement |
| Flame Failure Response Time | < 1000 ms | NFPA 85 Section 5.3.6 |
| Maximum Fuel Valve Closure Time | < 3000 ms | NFPA 86 Section 8.5.2.2 |
| Total Response Time | < 4000 ms | NFPA 86 requirement |
| Spurious Trip Rate | < 0.5 per year | Operational availability |

#### 2.3.4 NFPA Compliance Matrix

| NFPA Requirement | Section | Implementation |
|------------------|---------|----------------|
| Flame supervision | NFPA 85 5.3.6 | UV/IR flame scanner with self-check |
| Safety shutoff valves | NFPA 86 8.5.1 | Dual block and bleed arrangement |
| Purge requirements | NFPA 85 5.5 | 4 volume air changes minimum |
| Trial for ignition | NFPA 85 5.3.7 | 10 seconds maximum pilot, 10 seconds main |
| Flame failure response | NFPA 86 8.5.2.2 | < 4 seconds total |

#### 2.3.5 Architecture

```
                          +---------------------+
                          |   GL-007 Agent      |
                          |  Flame Supervision  |
                          +---------------------+
                                    |
                          +---------v---------+
     +----------------+   |     Flame         |   +----------------+
     | Flame Scanner A|-->|   Comparison      |<--| Flame Scanner C|
     | (BE-007-A)     |   |   (2oo3 Vote)     |   | (BE-007-C)     |
     +----------------+   +-------------------+   +----------------+
              |                   |                      |
              |           +-------+                      |
     +----------------+   |   Flame Loss Logic  |        |
     | Flame Scanner B|-->|   (Any = FALSE)     |--------+
     | (BE-007-B)     |   +---------------------+
     +----------------+           |
                          +-------v-------+
                          |   Fuel        |
                          |   Isolation   |
                          +---------------+
                                  |
       +----------+---------------+--------------+----------+
       |          |               |              |          |
   +---v----+ +---v----+    +-----v-----+   +----v---+ +----v----+
   | Block  | | Block  |    | Bleed     |   | Alarm  | | Purge   |
   | Valve 1| | Valve 2|    | Valve     |   | Output | | Sequence|
   +--------+ +--------+    +-----------+   +--------+ +---------+
```

---

### 2.4 SIF-007-03: Low Flow Protection

#### 2.4.1 Function Description

**Purpose:** Prevent tube overtemperature and damage from inadequate process flow through furnace tubes.

**Initiating Cause:** Process flow falls below Low-Low setpoint (FLL).

**Safe State:** Fuel supply reduced or isolated, alarm activated.

#### 2.4.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-007-03-FR-01 | The SIF shall monitor process flow continuously | Design review, Testing |
| SIF-007-03-FR-02 | The SIF shall initiate protection when F <= FLL setpoint | FAT, SAT |
| SIF-007-03-FR-03 | The SIF shall reduce/isolate fuel on low flow | FAT, SAT |
| SIF-007-03-FR-04 | Time delay (5 sec) shall be applied to avoid spurious trips | Design review |
| SIF-007-03-FR-05 | The SIF shall annunciate Low Flow condition on HMI | FAT, SAT |
| SIF-007-03-FR-06 | The SIF shall implement 1oo2 flow sensor voting | Design review |

#### 2.4.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis |
| PFDavg | < 1E-02 | SIL 2 requirement |
| Response Time | < 3000 ms (excluding time delay) | Process Safety Time |
| Setpoint Accuracy | +/- 2% of span | Measurement requirement |
| Time Delay | 5 +/- 0.5 seconds | Anti-spurious trip |

---

### 2.5 SIF-007-04: Fouling Hot Spot Alert

#### 2.5.1 Function Description

**Purpose:** Detect localized overtemperature conditions caused by fouling or coking through TMT trending and comparison analysis.

**Initiating Cause:** TMT rate of change exceeds threshold or TMT deviation between sensors exceeds limit.

**Safe State:** Alert generated, enhanced monitoring activated, maintenance notification.

#### 2.5.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-007-04-FR-01 | The SIF shall calculate TMT rate of change continuously | Design review, Testing |
| SIF-007-04-FR-02 | The SIF shall detect TMT deviation between redundant sensors | Testing |
| SIF-007-04-FR-03 | The SIF shall alert when rate > 5 degF/minute sustained | FAT, SAT |
| SIF-007-04-FR-04 | The SIF shall alert when sensor deviation > 50 degF | FAT, SAT |
| SIF-007-04-FR-05 | The SIF shall log fouling indicators for trending | Testing |

#### 2.5.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 1 | LOPA Analysis |
| PFDavg | < 1E-01 | SIL 1 requirement |
| Detection Time | < 60 seconds | Fouling detection window |
| Rate Threshold | 5 degF/minute | Engineering analysis |
| Deviation Threshold | 50 degF | Hot spot indication |

---

### 2.6 SIF-007-05: Emergency Shutdown

#### 2.6.1 Function Description

**Purpose:** Provide master emergency shutdown capability for the furnace system, triggered manually or automatically from other SIFs.

**Initiating Cause:** Manual ESD pushbutton, automatic trip from SIF-007-01 to SIF-007-04, or external ESD signal.

**Safe State:** All fuel isolated, all burners de-energized, alarm activated.

#### 2.6.2 Functional Requirements

| Requirement ID | Requirement | Verification Method |
|----------------|-------------|---------------------|
| SIF-007-05-FR-01 | The SIF shall accept manual ESD pushbutton input | FAT, SAT |
| SIF-007-05-FR-02 | The SIF shall accept automatic trip inputs from SIF-007-01 to 04 | FAT, SAT |
| SIF-007-05-FR-03 | The SIF shall accept external ESD input from GL-001 | FAT, SAT |
| SIF-007-05-FR-04 | The SIF shall isolate all fuel sources | FAT, SAT |
| SIF-007-05-FR-05 | The SIF shall de-energize all burner controls | FAT, SAT |
| SIF-007-05-FR-06 | The SIF shall maintain safe state until manual reset | FAT, SAT |
| SIF-007-05-FR-07 | Reset shall require physical key switch | FAT, SAT |

#### 2.6.3 Performance Requirements

| Parameter | Requirement | Basis |
|-----------|-------------|-------|
| SIL Target | SIL 2 | LOPA Analysis |
| PFDavg | < 1E-02 | SIL 2 requirement |
| Response Time | < 3000 ms | Process Safety Time |
| Availability | > 99.9% | Operational requirement |

---

## 3. Performance Requirements

### 3.1 Response Time Budget

**TMT High Shutdown (SIF-007-01):**

| Component | Allocation | Justification |
|-----------|------------|---------------|
| TMT sensor response | 500 ms | Thermocouple time constant |
| Signal transmission | 100 ms | Communication latency |
| GL-007 processing | 200 ms | Software execution time |
| Output command | 100 ms | Communication latency |
| Fuel valve response | 2000 ms | Valve stroke time |
| **Total** | **2900 ms** | Within 3000 ms requirement |

**Flame Failure Detection (SIF-007-02):**

| Component | Allocation | Justification |
|-----------|------------|---------------|
| Flame scanner detection | 500 ms | Scanner response time |
| Signal transmission | 50 ms | Hardwired connection |
| GL-007 processing | 100 ms | Software execution |
| Output command | 50 ms | Hardwired connection |
| Fuel valve response | 2000 ms | Valve stroke time |
| **Total** | **2700 ms** | Within 4000 ms per NFPA |

### 3.2 Reliability Requirements

| Parameter | SIL 2 Requirement | SIL 1 Requirement |
|-----------|-------------------|-------------------|
| PFDavg | < 1E-02 | < 1E-01 |
| Spurious Trip Rate | < 0.1/year | < 0.5/year |
| MTTR | < 8 hours | < 24 hours |
| Availability | > 99.9% | > 99% |

### 3.3 Accuracy Requirements

| Measurement | Required Accuracy | Standard Reference |
|-------------|-------------------|-------------------|
| TMT | +/- 2.2 degC or 0.75% | IEC 60584 (Type K) |
| Process Flow | +/- 1% of span | SIL 2 requirement |
| Flame Detection | 100% detection | NFPA 85/86 |
| Fuel Valve Position | Discrete (open/closed) | Position feedback |

---

## 4. Failure Mode Specifications

### 4.1 TMT Sensor Failure Modes

| Failure Mode | Detection Method | Action | Fail State |
|--------------|------------------|--------|------------|
| Open circuit | Signal < 4 mA | Trip (fail-safe) | Safe |
| Short circuit | Signal > 20.5 mA | Trip (fail-safe) | Safe |
| Drift high | Comparison with redundant | Alarm, continue with voting | Detected |
| Drift low | Comparison with redundant | Trip (fail-safe) | Safe |
| Slow response | Rate of change check | Alarm, diagnostics | Detected |

### 4.2 Flame Scanner Failure Modes

| Failure Mode | Detection Method | Action | Fail State |
|--------------|------------------|--------|------------|
| Loss of signal | Signal monitoring | Trip (fail-safe) | Safe |
| False flame indication | Self-check circuit | Alarm, manual override | Detected |
| Scanner fouling | Signal degradation | Alarm, maintenance | Detected |
| Wiring fault | Continuity check | Trip (fail-safe) | Safe |

### 4.3 Fuel Valve Failure Modes

| Failure Mode | Detection Method | Action | Fail State |
|--------------|------------------|--------|------------|
| Fail to close | Position feedback | Alarm, backup valve | Dangerous detected |
| Partial stroke | Position analysis | Alarm, maintenance | Detected |
| Leak-through | Pressure monitoring | Alarm, block & bleed | Detected |
| Coil failure | Current monitoring | Alarm, redundant valve | Detected |

### 4.4 GL-007 Processing Failure Modes

| Failure Mode | Detection Method | Action | Fail State |
|--------------|------------------|--------|------------|
| Watchdog timeout | Hardware watchdog | Reboot, trip if persistent | Safe |
| Memory corruption | CRC checks | Trip, diagnostic mode | Safe |
| CPU overload | Execution time monitoring | Alarm, load shedding | Detected |
| Communication loss | Heartbeat monitoring | Trip after timeout | Safe |
| Configuration error | Startup validation | Block startup | Safe |

---

## 5. Diagnostic Coverage Requirements

### 5.1 Required Diagnostic Coverage

| Component Type | Required DC (SIL 2) | Diagnostic Method |
|----------------|---------------------|-------------------|
| TMT sensors | >= 90% | Range check, comparison, rate of change |
| Flame scanners | >= 90% | Self-check, comparison |
| Logic solver | >= 90% | Watchdog, memory check, self-test |
| Fuel valves | >= 90% | Position feedback, partial stroke test |
| Communication | >= 90% | CRC, heartbeat, timeout detection |

### 5.2 Diagnostic Test Coverage Matrix

| Diagnostic Test | Coverage Claim | Test Interval | Failure Modes Covered |
|-----------------|----------------|---------------|----------------------|
| TMT range check | 70% | Continuous | Open, short circuit |
| TMT comparison | 85% | Continuous | Drift, stuck |
| TMT rate of change | 60% | Continuous | Slow response |
| Flame scanner self-check | 90% | Continuous | Scanner fault |
| Valve position feedback | 90% | Continuous | Fail to close |
| Partial stroke test | 90% | Monthly | Stuck valve |
| Watchdog timer | 99% | Continuous | CPU fault |
| Memory CRC | 90% | Startup + periodic | Corruption |

### 5.3 Combined Diagnostic Coverage Calculation

For TMT monitoring (parallel combination):
```
DC_combined = 1 - (1 - DC_range) x (1 - DC_comparison) x (1 - DC_rate)
DC_combined = 1 - (1 - 0.70) x (1 - 0.85) x (1 - 0.60)
DC_combined = 1 - 0.30 x 0.15 x 0.40
DC_combined = 1 - 0.018
DC_combined = 98.2%
```

**DC Achieved: 98.2% (High DC category per IEC 61508)**

---

## 6. Proof Test Requirements

### 6.1 Proof Test Intervals

| SIF | SIL | PFDavg Target | DC Assumed | Proof Test Interval |
|-----|-----|---------------|------------|---------------------|
| SIF-007-01 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-007-02 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-007-03 | SIL 2 | < 1E-02 | 90% | 12 months |
| SIF-007-04 | SIL 1 | < 1E-01 | 60% | 24 months |
| SIF-007-05 | SIL 2 | < 1E-02 | 90% | 12 months |

### 6.2 Proof Test Procedures

#### 6.2.1 TMT Sensor Proof Test

| Step | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 1 | Isolate TMT sensor from SIF logic | Bypass alarmed |
| 2 | Apply calibrated temperature source | N/A |
| 3 | Verify sensor reading at low point | +/- 2.2 degC of source |
| 4 | Verify sensor reading at midpoint | +/- 2.2 degC of source |
| 5 | Verify sensor reading at high point | +/- 2.2 degC of source |
| 6 | Verify trip setpoint activation | Trip at TMT_HH +/- 1% |
| 7 | Return sensor to service | Normal operation |

#### 6.2.2 Flame Scanner Proof Test

| Step | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 1 | Verify scanner self-check operation | Self-check passes |
| 2 | Block flame signal (shutter test) | Flame loss detected < 1 sec |
| 3 | Verify fuel valve command issued | Valve closure commanded |
| 4 | Verify valve closure | Position feedback = closed |
| 5 | Verify total response time | < 4 seconds |
| 6 | Return scanner to service | Normal operation |

#### 6.2.3 Fuel Valve Proof Test

| Step | Description | Acceptance Criteria |
|------|-------------|---------------------|
| 1 | Perform partial stroke test | 10-30% travel achieved |
| 2 | Verify position feedback | Feedback matches position |
| 3 | Full stroke test (shutdown) | Full closure < 3 seconds |
| 4 | Verify leak-tight closure | No leak-through detected |
| 5 | Return valve to service | Normal operation |

---

## 7. Integration Requirements with GL-001

### 7.1 Communication Interface

| Parameter | Requirement |
|-----------|-------------|
| Communication protocol | Modbus TCP/IP, OPC UA |
| Update rate | 100 ms minimum |
| Data integrity | CRC-32 error detection |
| Timeout handling | 2 second watchdog, notify on timeout |

### 7.2 Data Exchange with GL-001

**GL-007 to GL-001:**

| Data Point | Description | Update Rate |
|------------|-------------|-------------|
| TMT_MAX | Maximum TMT reading | 100 ms |
| TMT_AVG | Average TMT reading | 100 ms |
| SIF_STATUS | Status of all SIFs | 100 ms |
| TRIP_ACTIVE | Trip condition active | Immediate |
| FURNACE_EFFICIENCY | Calculated efficiency | 1 second |

**GL-001 to GL-007:**

| Data Point | Description | Update Rate |
|------------|-------------|-------------|
| SETPOINT_CMD | Temperature setpoint | On change |
| ESD_CMD | Emergency shutdown command | Immediate |
| MODE_CMD | Operating mode command | On change |
| BYPASS_CMD | Bypass command (authorized) | On change |

### 7.3 Coordinated Shutdown Sequence

```
GL-007 Trip Initiated
        |
        v
+-------+-------+
|    GL-007     |
| Fuel Isolation|
+-------+-------+
        |
        v
+-------+-------+
|  Notify GL-001|
| Trip Active   |
+-------+-------+
        |
        v
+-------+-------+
|    GL-001     |
| Process       |
| Isolation     |
+-------+-------+
        |
        v
+-------+-------+
| Coordinated   |
| Safe State    |
+---------------+
```

### 7.4 Independence Requirements

| Aspect | Requirement | Implementation |
|--------|-------------|----------------|
| Power supply | Independent power for GL-007 SIS functions | Separate UPS |
| Communication | Failure of communication shall not prevent SIF | Hardwired backup |
| Logic solver | GL-007 SIS logic independent of GL-001 | Separate controller |
| Sensors | Dedicated SIS sensors for safety functions | Not shared with BPCS |

---

## 8. Safe State Definitions

### 8.1 Safe State Matrix

| SIF | Safe State Description | Energy State | Valve State | System State |
|-----|------------------------|--------------|-------------|--------------|
| SIF-007-01 | TMT High Shutdown | De-energized | Fuel closed, process open | Cooldown |
| SIF-007-02 | Flame Failure | De-energized | Block & bleed closed | Purge ready |
| SIF-007-03 | Low Flow Protection | De-energized | Fuel reduced/closed | Standby |
| SIF-007-04 | Fouling Alert | Normal | No change | Alerting |
| SIF-007-05 | Emergency Shutdown | De-energized | All fuel closed | Full shutdown |

### 8.2 Fail-Safe Design Principles

| Principle | Implementation |
|-----------|----------------|
| Fail-safe on loss of power | Fuel valves close on de-energize (spring return) |
| Fail-safe on signal loss | Low signal interpreted as trip condition |
| Fail-safe on communication loss | Watchdog timer triggers safe state |
| Fail-safe on software fault | Watchdog, memory checks, safe defaults |
| Dual block and bleed | Two valves in series plus bleed to atmosphere |

---

## 9. Verification and Validation

### 9.1 Factory Acceptance Test (FAT)

| Test Category | Tests Required |
|---------------|----------------|
| Functional | All functional requirements verified |
| Response time | < allocated time for each SIF |
| Failure mode | Correct fail-safe behavior |
| Communication | All interfaces operational |
| Documentation | All documents complete and approved |

### 9.2 Site Acceptance Test (SAT)

| Test Category | Tests Required |
|---------------|----------------|
| Loop check | All I/O points verified |
| Functional | Trip at correct setpoints |
| Response time | < allocated time with actual I/O |
| Integrated | Correct interaction with GL-001 |
| Documentation | Signed test records |

### 9.3 Validation Matrix

| Requirement ID | Test Type | Test Procedure | Acceptance Criteria |
|----------------|-----------|----------------|---------------------|
| SIF-007-01-FR-01 | FAT | TMT-01 | All sensors operational |
| SIF-007-01-FR-02 | SAT | TMT-02 | Trip at setpoint +/- 1% |
| SIF-007-02-FR-02 | FAT | FLM-01 | Detection < 1 second |
| SIF-007-02-FR-03 | SAT | FLM-02 | Valve closure < 3 seconds |

---

## 10. Appendices

### Appendix A: SRS Checklist per IEC 61511-1 Clause 10

| Clause | Requirement | Section | Status |
|--------|-------------|---------|--------|
| 10.3.1 a) | Description of SIF | Section 2 | Complete |
| 10.3.1 b) | Safe state definition | Section 8 | Complete |
| 10.3.1 c) | SIL for each SIF | Section 2 | Complete |
| 10.3.1 d) | Demand mode | Low demand (all) | Complete |
| 10.3.1 e) | Constraints on MTTR | Section 3.2 | Complete |
| 10.3.1 f) | Maximum spurious trip rate | Section 3.2 | Complete |
| 10.3.1 g) | Response time requirements | Section 3.1 | Complete |
| 10.3.1 h) | Process interface requirements | Section 7 | Complete |
| 10.3.1 i) | Modes of operation | Section 2 | Complete |
| 10.3.1 j) | Bypass requirements | Section 7.2 | Complete |
| 10.3.1 k) | Maintenance requirements | Section 6 | Complete |
| 10.3.1 l) | Test requirements | Section 9 | Complete |

### Appendix B: Provenance Information

| Parameter | Value |
|-----------|-------|
| Document ID | GL-SIL-SRS-007 |
| Version | 1.0 |
| Created Date | 2025-12-07 |
| Created By | GL-RegulatoryIntelligence |
| Provenance Hash | SHA-256: (calculated at approval) |

### Appendix C: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-07 | GL-RegulatoryIntelligence | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
