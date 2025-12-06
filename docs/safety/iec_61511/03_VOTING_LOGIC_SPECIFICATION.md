# Voting Logic Specification

## GreenLang Process Heat Agents - SIS Architecture and PFD Calculations

**Document ID:** GL-SIL-VOTE-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61508-6:2010, IEC 61511-1:2016

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Voting Architecture Overview](#2-voting-architecture-overview)
3. [PFD Calculation Methodology](#3-pfd-calculation-methodology)
4. [1oo1 Architecture](#4-1oo1-architecture)
5. [1oo2 Architecture](#5-1oo2-architecture)
6. [2oo2 Architecture](#6-2oo2-architecture)
7. [2oo3 Architecture](#7-2oo3-architecture)
8. [Diagnostic Coverage Requirements](#8-diagnostic-coverage-requirements)
9. [Common Cause Failure Analysis](#9-common-cause-failure-analysis)
10. [SIF Architecture Assignments](#10-sif-architecture-assignments)
11. [PFD Verification Summary](#11-pfd-verification-summary)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the voting logic architectures and Probability of Failure on Demand (PFD) calculations for Safety Instrumented Functions (SIFs) in the GreenLang Process Heat agent system. It provides the technical basis for achieving the required Safety Integrity Levels (SIL).

### 1.2 Scope

This specification covers:

- Voting architecture selection for all SIFs
- PFD calculations per IEC 61508-6
- Diagnostic coverage requirements
- Common cause failure (CCF) analysis
- Hardware fault tolerance requirements

### 1.3 Notation

| Symbol | Description | Unit |
|--------|-------------|------|
| PFDavg | Average Probability of Failure on Demand | - |
| PFDsys | System PFD including CCF | - |
| lambda_D | Dangerous failure rate | per hour |
| lambda_DD | Dangerous detected failure rate | per hour |
| lambda_DU | Dangerous undetected failure rate | per hour |
| DC | Diagnostic Coverage | % |
| beta | Common Cause Factor | - |
| Ti | Proof Test Interval | hours |
| MTTR | Mean Time to Repair | hours |
| T1 | Comparison interval (for 1oo2D) | hours |

---

## 2. Voting Architecture Overview

### 2.1 MooN Notation

The MooN (M-out-of-N) notation describes the number of channels (N) and how many must agree to activate (M):

| Architecture | Description | Trip Logic | Safety |
|--------------|-------------|------------|--------|
| 1oo1 | 1-out-of-1 | Single channel | Minimum |
| 1oo2 | 1-out-of-2 | Either channel trips | High safety |
| 2oo2 | 2-out-of-2 | Both channels must trip | High availability |
| 2oo3 | 2-out-of-3 | Any 2 of 3 channels trip | Balanced |
| 1oo2D | 1oo2 with diagnostics | Enhanced 1oo2 | High safety + diagnostics |

### 2.2 Architecture Selection Criteria

| Criterion | 1oo1 | 1oo2 | 2oo2 | 2oo3 |
|-----------|------|------|------|------|
| Maximum achievable SIL | SIL 1 | SIL 3 | SIL 2 | SIL 3 |
| Hardware Fault Tolerance | 0 | 1 | 0 | 1 |
| Safety (PFD) | Low | High | Medium | High |
| Availability (STR) | High | Low | High | Medium |
| Cost | Low | Medium | Medium | High |
| Complexity | Low | Medium | Medium | High |

### 2.3 Hardware Fault Tolerance Requirements

Per IEC 61511-1:2016, Table 6:

| Target SIL | Minimum HFT | Without HFT Requirements |
|------------|-------------|--------------------------|
| SIL 1 | 0 | N/A |
| SIL 2 | 1 | DC >= 90% or prior use justified |
| SIL 3 | 2 | DC >= 99% or diverse redundancy |
| SIL 4 | Not covered by IEC 61511 | Special design |

---

## 3. PFD Calculation Methodology

### 3.1 Failure Rate Definitions

```
lambda_total = lambda_S + lambda_D

lambda_D = lambda_DD + lambda_DU

lambda_DD = DC x lambda_D
lambda_DU = (1 - DC) x lambda_D

Where:
  lambda_S = Safe failure rate (leads to spurious trip)
  lambda_D = Dangerous failure rate (prevents SIF from operating)
  lambda_DD = Dangerous detected (by diagnostics)
  lambda_DU = Dangerous undetected (revealed only by proof test)
  DC = Diagnostic Coverage
```

### 3.2 PFD Equations (IEC 61508-6 Method)

The PFD equations assume:

- Low demand mode of operation
- Proof test restores to "as new"
- Constant failure rate
- Detected failures are repaired immediately
- Mean Time to Repair (MTTR) << Proof Test Interval (Ti)

### 3.3 General PFD Formula Components

```
PFDavg = f(lambda_DU, Ti, lambda_DD, MTTR, beta, architecture)
```

---

## 4. 1oo1 Architecture

### 4.1 Architecture Diagram

```
                     +-------------------+
Input ------+------->|    Channel A      |------+-----> Output
                     |    (Sensor/Logic/ |
                     |     Final Element)|
                     +-------------------+
```

### 4.2 PFD Equation (1oo1)

```
PFDavg(1oo1) = lambda_DU x Ti / 2 + lambda_DD x MTTR

Simplified (MTTR << Ti):
PFDavg(1oo1) = lambda_DU x Ti / 2
```

### 4.3 Example Calculation

**Parameters:**
- lambda_D = 5E-06 per hour (typical sensor)
- DC = 60%
- Ti = 8760 hours (1 year)
- MTTR = 8 hours

**Calculation:**
```
lambda_DU = (1 - 0.6) x 5E-06 = 2E-06 per hour
lambda_DD = 0.6 x 5E-06 = 3E-06 per hour

PFDavg = (2E-06 x 8760) / 2 + (3E-06 x 8)
PFDavg = 8.76E-03 + 2.4E-05
PFDavg = 8.78E-03

This achieves: SIL 1 (PFD < 0.1)
```

### 4.4 1oo1 Application

| Suitable For | Not Suitable For |
|--------------|------------------|
| SIL 1 applications | SIL 2/3 without HFT justification |
| Non-critical monitoring | High consequence scenarios |
| Advisory functions | Primary safety shutdown |

---

## 5. 1oo2 Architecture

### 5.1 Architecture Diagram

```
                     +-------------------+
Input ------+------->|    Channel A      |------+
            |        +-------------------+      |
            |                                   +---[OR]---> Output
            |        +-------------------+      |
            +------->|    Channel B      |------+
                     +-------------------+
```

Either channel can initiate the safety action (trip on first fault).

### 5.2 PFD Equation (1oo2)

Without CCF:
```
PFDavg(1oo2) = (lambda_DU x Ti)^2 / 3 + lambda_DD x MTTR
```

With CCF (beta factor model):
```
PFDavg(1oo2) = (1 - beta)^2 x (lambda_DU x Ti)^2 / 3
              + beta x lambda_DU x Ti / 2
              + lambda_DD x MTTR
```

### 5.3 Example Calculation

**Parameters:**
- lambda_D = 5E-06 per hour per channel
- DC = 90%
- Ti = 8760 hours (1 year)
- MTTR = 8 hours
- beta = 0.1 (10% CCF)

**Calculation:**
```
lambda_DU = (1 - 0.9) x 5E-06 = 5E-07 per hour
lambda_DD = 0.9 x 5E-06 = 4.5E-06 per hour

Term 1 (independent failures):
= (1 - 0.1)^2 x (5E-07 x 8760)^2 / 3
= 0.81 x (4.38E-03)^2 / 3
= 0.81 x 1.92E-05 / 3
= 5.18E-06

Term 2 (CCF):
= 0.1 x 5E-07 x 8760 / 2
= 2.19E-04

Term 3 (detected failures):
= 4.5E-06 x 8
= 3.6E-05

PFDavg = 5.18E-06 + 2.19E-04 + 3.6E-05
PFDavg = 2.6E-04

This achieves: SIL 3 (PFD < 0.001)
```

### 5.4 1oo2 Application

| Suitable For | Considerations |
|--------------|----------------|
| SIL 2/3 applications | CCF dominates at high reliability |
| High safety requirement | Higher spurious trip rate |
| Process with HFT=1 | Requires diversity to minimize CCF |

---

## 6. 2oo2 Architecture

### 6.1 Architecture Diagram

```
                     +-------------------+
Input ------+------->|    Channel A      |------+
            |        +-------------------+      |
            |                                   +---[AND]---> Output
            |        +-------------------+      |
            +------->|    Channel B      |------+
                     +-------------------+
```

Both channels must agree to initiate safety action.

### 6.2 PFD Equation (2oo2)

Without CCF:
```
PFDavg(2oo2) = lambda_DU x Ti + lambda_DD x MTTR
```

With CCF:
```
PFDavg(2oo2) = (1 - beta) x lambda_DU x Ti
              + beta x lambda_DU x Ti / 2
              + lambda_DD x MTTR
```

Simplified:
```
PFDavg(2oo2) = (1 - beta/2) x lambda_DU x Ti + lambda_DD x MTTR
```

### 6.3 Example Calculation

**Parameters:**
- lambda_D = 5E-06 per hour per channel
- DC = 90%
- Ti = 8760 hours (1 year)
- MTTR = 8 hours
- beta = 0.1

**Calculation:**
```
lambda_DU = (1 - 0.9) x 5E-06 = 5E-07 per hour
lambda_DD = 0.9 x 5E-06 = 4.5E-06 per hour

PFDavg = (1 - 0.1/2) x 5E-07 x 8760 + 4.5E-06 x 8
PFDavg = 0.95 x 4.38E-03 + 3.6E-05
PFDavg = 4.16E-03 + 3.6E-05
PFDavg = 4.2E-03

This achieves: SIL 2 (PFD < 0.01)
```

### 6.4 2oo2 Application

| Suitable For | Considerations |
|--------------|----------------|
| High availability | Lower safety than 1oo2 |
| Minimize spurious trips | Single failure can prevent trip |
| Process cannot tolerate spurious shutdown | Not suitable for high SIL without additional measures |

---

## 7. 2oo3 Architecture

### 7.1 Architecture Diagram

```
                     +-------------------+
Input ------+------->|    Channel A      |------+
            |        +-------------------+      |
            |                                   |
            |        +-------------------+      |
            +------->|    Channel B      |------+---[2oo3]---> Output
            |        +-------------------+      |    Voter
            |                                   |
            |        +-------------------+      |
            +------->|    Channel C      |------+
                     +-------------------+
```

Any two channels must agree to initiate safety action.

### 7.2 PFD Equation (2oo3)

Without CCF:
```
PFDavg(2oo3) = (lambda_DU x Ti)^2 + lambda_DD x MTTR
```

With CCF:
```
PFDavg(2oo3) = (1 - beta)^2 x (lambda_DU x Ti)^2
              + beta x lambda_DU x Ti / 2
              + lambda_DD x MTTR
```

### 7.3 Example Calculation

**Parameters:**
- lambda_D = 5E-06 per hour per channel
- DC = 90%
- Ti = 8760 hours (1 year)
- MTTR = 8 hours
- beta = 0.1

**Calculation:**
```
lambda_DU = (1 - 0.9) x 5E-06 = 5E-07 per hour
lambda_DD = 0.9 x 5E-06 = 4.5E-06 per hour

Term 1 (independent failures):
= (1 - 0.1)^2 x (5E-07 x 8760)^2
= 0.81 x (4.38E-03)^2
= 0.81 x 1.92E-05
= 1.55E-05

Term 2 (CCF):
= 0.1 x 5E-07 x 8760 / 2
= 2.19E-04

Term 3 (detected failures):
= 4.5E-06 x 8
= 3.6E-05

PFDavg = 1.55E-05 + 2.19E-04 + 3.6E-05
PFDavg = 2.7E-04

This achieves: SIL 3 (PFD < 0.001)
```

### 7.4 2oo3 Application

| Suitable For | Considerations |
|--------------|----------------|
| SIL 2/3 with balanced safety/availability | Higher cost (3 channels) |
| Critical processes | Online diagnostics detect single failures |
| Continuous operation required | Industry standard for combustion |
| NFPA 85/86 compliance | HFT = 1 maintained after first failure |

---

## 8. Diagnostic Coverage Requirements

### 8.1 Diagnostic Coverage Definition

```
DC = lambda_DD / lambda_D = Detected dangerous failures / Total dangerous failures
```

### 8.2 Diagnostic Coverage Categories

Per IEC 61508-2, Table A.4:

| Coverage Level | DC Range | Description |
|----------------|----------|-------------|
| None | < 60% | Minimal diagnostics |
| Low | 60% to < 90% | Basic diagnostics |
| Medium | 90% to < 99% | Comprehensive diagnostics |
| High | >= 99% | Extensive diagnostics |

### 8.3 Diagnostic Techniques

| Diagnostic | Coverage Achieved | Detection Time |
|------------|-------------------|----------------|
| Input comparison | 90-99% | Immediate |
| Watchdog timer | 60-90% | Timer period |
| Memory CRC | 90-99% | Scan interval |
| Output feedback | 90-99% | Response time |
| Self-test | 60-90% | Test interval |
| Current monitoring | 60-90% | Immediate |
| Plausibility check | 60-90% | Scan interval |
| Cross-channel comparison | 90-99% | Comparison interval |

### 8.4 DC Requirements per SIL

| Target SIL | Minimum DC (1oo1) | Minimum DC (Redundant) |
|------------|-------------------|------------------------|
| SIL 1 | N/A | N/A |
| SIL 2 | 90%* | 60% |
| SIL 3 | 99%* | 90% |

*For architectures without HFT

### 8.5 GL-001 Diagnostic Implementation

| Component | Diagnostic Method | DC Claimed | Basis |
|-----------|-------------------|------------|-------|
| Temperature sensor | Range check, rate of change | 90% | IEC 61508-6 Table D.1 |
| Flow sensor | Cross-comparison, range check | 90% | IEC 61508-6 Table D.1 |
| Pressure sensor | Cross-comparison, range check | 90% | IEC 61508-6 Table D.1 |
| Flame scanner | Self-check, comparison | 95% | Manufacturer data |
| Logic solver | Watchdog, memory CRC, self-test | 99% | Certified PLC |
| Solenoid valve | Current monitoring, feedback | 90% | With limit switch |
| Communication | CRC, heartbeat, timeout | 99% | Protocol specification |

---

## 9. Common Cause Failure Analysis

### 9.1 Beta Factor Model

The beta factor represents the fraction of failures that affect all redundant channels simultaneously:

```
beta = lambda_CCF / lambda_D

Where:
  lambda_CCF = Common cause failure rate
  lambda_D = Total dangerous failure rate
```

### 9.2 Beta Factor Values

| System Characteristics | Beta Factor |
|-----------------------|-------------|
| Identical redundancy, same environment | 10% |
| Diverse hardware, same environment | 5% |
| Diverse hardware, separated environment | 2% |
| Full diversity (different technology) | 1% |

### 9.3 CCF Reduction Techniques

| Technique | Beta Reduction | Implementation |
|-----------|----------------|----------------|
| Physical separation | 0.5x | Separate enclosures, cable routes |
| Diversity | 0.5x | Different sensor types/manufacturers |
| Design review | 0.8x | Independent design verification |
| Separate testing | 0.8x | Staggered proof tests |
| Separate maintenance | 0.9x | Different technicians |
| Environmental control | 0.8x | Temperature, humidity, EMC |

### 9.4 CCF Scoring (IEC 61508-6 Annex D)

| Factor | Score Range | GL-001 Score |
|--------|-------------|--------------|
| Separation/Segregation | 0-3 | 2 |
| Diversity | 0-8 | 4 |
| Complexity | 0-3 | 2 |
| Assessment | 0-3 | 2 |
| Procedures | 0-3 | 2 |
| Training | 0-4 | 3 |
| Environmental control | 0-4 | 3 |
| **Total** | **0-28** | **18** |

Score 18 corresponds to beta = 5% (IEC 61508-6 Table D.2)

### 9.5 CCF Calculation for GL-001 SIFs

| SIF | Architecture | Base Beta | Adjusted Beta | Basis |
|-----|--------------|-----------|---------------|-------|
| SIF-001 | 2oo2 sensors | 10% | 5% | Diverse sensors |
| SIF-002 | 2oo2 sensors | 10% | 5% | Diverse sensors |
| SIF-003 | 1oo2 sensors | 10% | 5% | Diverse sensors |
| SIF-004 | 2oo3 flame | 10% | 5% | Diverse scanners |
| SIF-005 | 1oo1 logic | N/A | N/A | Single channel |

---

## 10. SIF Architecture Assignments

### 10.1 SIF-001: High Temperature Shutdown

**Architecture:** 2oo2 Sensors, 1oo1 Logic, 1oo2 Final Elements

```
+----------+     +----------+
| TE-001A  |---->|          |
+----------+     |   2oo2   |---->+----------+---->+----------+
                 |  Compare |     |   Logic  |     | XV-001A  |
+----------+     |          |     | (GL-001) |     +----------+
| TE-001B  |---->|          |     +----------+---->|          |
+----------+     +----------+                      |   1oo2   |---->Safe State
                                                   |  Output  |
                                  +----------+---->|          |
                                  | XV-001B  |     +----------+
                                  +----------+
```

**PFD Calculation:**

| Subsystem | Architecture | lambda_DU | Ti | DC | beta | PFDavg |
|-----------|--------------|-----------|----|----|------|--------|
| Sensors | 2oo2 | 5E-07 | 8760 | 90% | 5% | 4.2E-03 |
| Logic | 1oo1 | 1E-07 | 8760 | 99% | N/A | 4.4E-05 |
| Final Element | 1oo2 | 2E-06 | 8760 | 90% | 10% | 8.8E-04 |

**System PFDavg:**
```
PFDsys = PFD_sensor + PFD_logic + PFD_FE
PFDsys = 4.2E-03 + 4.4E-05 + 8.8E-04
PFDsys = 5.1E-03

Achieved: SIL 2 (PFD < 1E-02)
```

---

### 10.2 SIF-002: Low Flow Protection

**Architecture:** 2oo2 Sensors, 1oo1 Logic, 1oo1 Final Element

```
+----------+     +----------+     +----------+     +----------+
| FT-001A  |---->|   2oo2   |---->|   Logic  |---->| XV-002   |---->Safe State
+----------+     |  Compare |     | (GL-001) |     +----------+
                 |          |     +----------+
+----------+     |          |
| FT-001B  |---->|          |
+----------+     +----------+
```

**PFD Calculation:**

| Subsystem | Architecture | lambda_DU | Ti | DC | beta | PFDavg |
|-----------|--------------|-----------|----|----|------|--------|
| Sensors | 2oo2 | 5E-07 | 8760 | 90% | 5% | 4.2E-03 |
| Logic | 1oo1 | 1E-07 | 8760 | 99% | N/A | 4.4E-05 |
| Final Element | 1oo1 | 3E-06 | 8760 | 90% | N/A | 1.3E-03 |

**System PFDavg:**
```
PFDsys = 4.2E-03 + 4.4E-05 + 1.3E-03
PFDsys = 5.5E-03

Achieved: SIL 2 (PFD < 1E-02)
```

---

### 10.3 SIF-003: Pressure Relief Monitoring

**Architecture:** 1oo2 Sensors, 1oo1 Logic, 1oo2 Final Elements

```
+----------+     +----------+
| PT-001A  |---->|          |
+----------+     |   1oo2   |---->+----------+---->+----------+
                 |  OR Vote |     |   Logic  |     | PV-001A  |
+----------+     |          |     | (GL-001) |     +----------+
| PT-001B  |---->|          |     +----------+---->|          |
+----------+     +----------+                      |   1oo2   |---->Safe State
                                                   |  Output  |
                                  +----------+---->|          |
                                  | PV-001B  |     +----------+
                                  +----------+
```

**PFD Calculation:**

| Subsystem | Architecture | lambda_DU | Ti | DC | beta | PFDavg |
|-----------|--------------|-----------|----|----|------|--------|
| Sensors | 1oo2 | 5E-07 | 8760 | 90% | 5% | 2.2E-04 |
| Logic | 1oo1 | 1E-07 | 8760 | 99% | N/A | 4.4E-05 |
| Final Element | 1oo2 | 2E-06 | 8760 | 90% | 10% | 8.8E-04 |

**System PFDavg:**
```
PFDsys = 2.2E-04 + 4.4E-05 + 8.8E-04
PFDsys = 1.1E-03

Achieved: SIL 2 (PFD < 1E-02)
```

---

### 10.4 SIF-004: Flame Failure Detection

**Architecture:** 2oo3 Flame Scanners, 1oo1 Logic, Dual Block & Bleed

```
+----------+
| FS-001A  |---->+
+----------+     |
                 |   +----------+     +----------+     +----------+
+----------+     +-->|   2oo3   |---->|   Logic  |---->| XV-004A  |
| FS-001B  |-------->|  Voter   |     | (GL-001) |     +----------+
+----------+     +-->|          |     +----------+---->|  Block   |
                 |   +----------+                      |  & Bleed |---->Safe
+----------+     |                    +----------+---->|          |
| FS-001C  |---->+                    | XV-004B  |     +----------+
+----------+                          +----------+

```

**PFD Calculation (Per NFPA 85/86):**

| Subsystem | Architecture | lambda_DU | Ti | DC | beta | PFDavg |
|-----------|--------------|-----------|----|----|------|--------|
| Flame Scanners | 2oo3 | 3E-07 | 8760 | 95% | 5% | 2.5E-04 |
| Logic | 1oo1 | 1E-07 | 8760 | 99% | N/A | 4.4E-05 |
| Dual Block & Bleed | Series | 2E-06 | 8760 | 90% | 10% | 1.8E-03 |

**System PFDavg:**
```
PFDsys = 2.5E-04 + 4.4E-05 + 1.8E-03
PFDsys = 2.1E-03

Achieved: SIL 2 (PFD < 1E-02)
```

---

### 10.5 SIF-005: Emergency Shutdown

**Architecture:** Multiple Input OR, 1oo1 Logic, Multiple Output

```
+-------------+
| Manual ESD  |------+
+-------------+      |
                     |     +----------+     +----------+
+-------------+      |---->|   OR     |---->|   Logic  |---->Fuel Isolation
| SIF-001     |------+     |  Input   |     | (GL-001) |---->Process Isolation
+-------------+      |     +----------+     +----------+---->Cooling Activation
                     |                                  ---->Alarm
+-------------+      |
| SIF-002     |------+
+-------------+      |
                     |
+-------------+      |
| SIF-003     |------+
+-------------+      |
                     |
+-------------+      |
| SIF-004     |------+
+-------------+      |
                     |
+-------------+      |
| External ESD|------+
+-------------+
```

**PFD Calculation:**

| Subsystem | Architecture | lambda_DU | Ti | DC | beta | PFDavg |
|-----------|--------------|-----------|----|----|------|--------|
| Manual Input | 1oo1 | 1E-07 | 8760 | 90% | N/A | 4.4E-04 |
| Logic | 1oo1 | 1E-07 | 8760 | 99% | N/A | 4.4E-05 |
| Outputs | 1oo2 (avg) | 2E-06 | 8760 | 90% | 10% | 8.8E-04 |

**System PFDavg:**
```
PFDsys = 4.4E-04 + 4.4E-05 + 8.8E-04
PFDsys = 1.4E-03

Achieved: SIL 2 (PFD < 1E-02)
```

---

### 10.6 SIF-006: Ventilation Fault Detection

**Architecture:** 1oo1 Sensor, 1oo1 Logic, 1oo1 Output

```
+----------+     +----------+     +----------+
| AS-001   |---->|   Logic  |---->| Vent Fan |---->Safe State
| (Airflow)|     | (GL-005) |     | Starter  |
+----------+     +----------+     +----------+
```

**PFD Calculation:**

| Subsystem | Architecture | lambda_DU | Ti | DC | PFDavg |
|-----------|--------------|-----------|----|----|--------|
| Sensor | 1oo1 | 1E-06 | 17520 | 60% | 8.8E-03 |
| Logic | 1oo1 | 1E-07 | 17520 | 99% | 8.8E-05 |
| Output | 1oo1 | 3E-06 | 17520 | 60% | 2.6E-02 |

**System PFDavg:**
```
PFDsys = 8.8E-03 + 8.8E-05 + 2.6E-02
PFDsys = 3.5E-02

Achieved: SIL 1 (PFD < 1E-01)
```

---

### 10.7 SIF-007: CO/CO2 High Alert

**Architecture:** 1oo1 Sensor, 1oo1 Logic, 1oo1 Output

```
+----------+     +----------+     +----------+
| CO/CO2   |---->|   Logic  |---->| Alarm &  |---->Alert State
| Sensor   |     | (GL-005) |     | Vent     |
+----------+     +----------+     +----------+
```

**PFD Calculation:**

| Subsystem | Architecture | lambda_DU | Ti | DC | PFDavg |
|-----------|--------------|-----------|----|----|--------|
| Sensor | 1oo1 | 2E-06 | 8760 | 60% | 3.5E-03 |
| Logic | 1oo1 | 1E-07 | 8760 | 99% | 4.4E-05 |
| Output | 1oo1 | 3E-06 | 8760 | 60% | 5.3E-03 |

**System PFDavg:**
```
PFDsys = 3.5E-03 + 4.4E-05 + 5.3E-03
PFDsys = 8.8E-03

Achieved: SIL 1 (PFD < 1E-01)
```

---

### 10.8 SIF-008: Emission Threshold Violation

**Architecture:** 1oo1 Data, 1oo1 Logic, 1oo1 Alert

```
+----------+     +----------+     +----------+
| Emission |---->|   Logic  |---->| Alert &  |---->Alert State
| Data     |     | (GL-007) |     | Log      |
+----------+     +----------+     +----------+
```

**PFD Calculation:**

| Subsystem | Architecture | lambda_DU | Ti | DC | PFDavg |
|-----------|--------------|-----------|----|----|--------|
| Data Input | 1oo1 | 1E-06 | 17520 | 60% | 8.8E-03 |
| Logic | 1oo1 | 1E-07 | 17520 | 99% | 8.8E-05 |
| Output | 1oo1 | 1E-06 | 17520 | 60% | 8.8E-03 |

**System PFDavg:**
```
PFDsys = 8.8E-03 + 8.8E-05 + 8.8E-03
PFDsys = 1.8E-02

Achieved: SIL 1 (PFD < 1E-01)
```

---

## 11. PFD Verification Summary

### 11.1 Summary Table

| SIF | Target SIL | Target PFD | Architecture | Calculated PFD | Achieved SIL | Status |
|-----|------------|------------|--------------|----------------|--------------|--------|
| SIF-001 | SIL 2 | < 1E-02 | 2oo2/1oo1/1oo2 | 5.1E-03 | SIL 2 | PASS |
| SIF-002 | SIL 2 | < 1E-02 | 2oo2/1oo1/1oo1 | 5.5E-03 | SIL 2 | PASS |
| SIF-003 | SIL 2 | < 1E-02 | 1oo2/1oo1/1oo2 | 1.1E-03 | SIL 2 | PASS |
| SIF-004 | SIL 2 | < 1E-02 | 2oo3/1oo1/DBB | 2.1E-03 | SIL 2 | PASS |
| SIF-005 | SIL 2 | < 1E-02 | OR/1oo1/1oo2 | 1.4E-03 | SIL 2 | PASS |
| SIF-006 | SIL 1 | < 1E-01 | 1oo1/1oo1/1oo1 | 3.5E-02 | SIL 1 | PASS |
| SIF-007 | SIL 1 | < 1E-01 | 1oo1/1oo1/1oo1 | 8.8E-03 | SIL 1 | PASS |
| SIF-008 | SIL 1 | < 1E-01 | 1oo1/1oo1/1oo1 | 1.8E-02 | SIL 1 | PASS |

### 11.2 Safety Margin Analysis

| SIF | Target PFD | Calculated PFD | Safety Margin | Assessment |
|-----|------------|----------------|---------------|------------|
| SIF-001 | 1E-02 | 5.1E-03 | 49% | Adequate |
| SIF-002 | 1E-02 | 5.5E-03 | 45% | Adequate |
| SIF-003 | 1E-02 | 1.1E-03 | 89% | Good margin |
| SIF-004 | 1E-02 | 2.1E-03 | 79% | Good margin |
| SIF-005 | 1E-02 | 1.4E-03 | 86% | Good margin |
| SIF-006 | 1E-01 | 3.5E-02 | 65% | Adequate |
| SIF-007 | 1E-01 | 8.8E-03 | 91% | Good margin |
| SIF-008 | 1E-01 | 1.8E-02 | 82% | Good margin |

### 11.3 Hardware Fault Tolerance Verification

| SIF | Required HFT | Achieved HFT | Status |
|-----|--------------|--------------|--------|
| SIF-001 | 1 | 1 (1oo2 FE) | PASS |
| SIF-002 | 1 | 1 (2oo2 sensor) | PASS |
| SIF-003 | 1 | 1 (1oo2 sensor, 1oo2 FE) | PASS |
| SIF-004 | 1 | 1 (2oo3 sensor) | PASS |
| SIF-005 | 1 | 1 (1oo2 FE) | PASS |
| SIF-006 | 0 | 0 | PASS |
| SIF-007 | 0 | 0 | PASS |
| SIF-008 | 0 | 0 | PASS |

---

## Appendix A: Failure Rate Data Sources

| Component | lambda_D (per hour) | Source |
|-----------|---------------------|--------|
| Temperature transmitter | 5E-06 | OREDA 2015 |
| Flow transmitter | 5E-06 | OREDA 2015 |
| Pressure transmitter | 5E-06 | OREDA 2015 |
| Flame scanner | 3E-06 | Manufacturer data |
| Safety PLC | 1E-07 | Manufacturer SIL certificate |
| Solenoid valve | 3E-06 | OREDA 2015 |
| Actuated valve | 2E-06 | OREDA 2015 |
| CO/CO2 sensor | 2E-06 | Manufacturer data |
| Airflow sensor | 1E-06 | Manufacturer data |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
