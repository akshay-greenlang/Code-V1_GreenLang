# GL-007 Furnace Performance Monitor HAZOP Study

## Hazard and Operability Analysis per IEC 61882

**Document ID:** GL-HAZOP-007
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61882:2016, API 560, API 530

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Study Scope and Methodology](#2-study-scope-and-methodology)
3. [System Description](#3-system-description)
4. [HAZOP Team and Responsibilities](#4-hazop-team-and-responsibilities)
5. [Node 1: Tube Metal Temperature Monitoring](#5-node-1-tube-metal-temperature-monitoring)
6. [Node 2: Efficiency Calculations](#6-node-2-efficiency-calculations)
7. [Node 3: Fouling Detection](#7-node-3-fouling-detection)
8. [Node 4: Alarm Generation](#8-node-4-alarm-generation)
9. [Risk Ranking Matrix](#9-risk-ranking-matrix)
10. [Action Items Summary](#10-action-items-summary)
11. [Appendices](#11-appendices)

---

## 1. Introduction

### 1.1 Purpose

This document presents the Hazard and Operability (HAZOP) study for the GL-007 Furnace Performance Monitor, a real-time monitoring agent for fired heater and furnace performance including Tube Metal Temperature (TMT) monitoring, efficiency calculations, and fouling detection.

### 1.2 Objectives

- Identify hazards related to furnace tube overtemperature
- Evaluate consequences of monitoring failures
- Assess alarm system effectiveness
- Verify protection adequacy per API standards
- Recommend improvements for furnace safety

### 1.3 References

| Document | Title |
|----------|-------|
| API 560 | Fired Heaters for General Refinery Service |
| API 530 | Calculation of Heater-Tube Thickness in Petroleum Refineries |
| API 579-1/ASME FFS-1 | Fitness-For-Service |
| API 573 | Inspection of Fired Boilers and Heaters |
| IEC 61882:2016 | Hazard and Operability Studies |

### 1.4 Furnace Safety Background

Fired heaters present significant hazards:

- **Tube Rupture:** Overtemperature leading to creep or stress rupture
- **Fire/Explosion:** Tube leak with hot process fluid ignition
- **Coking/Fouling:** Reduced heat transfer leading to hot spots
- **Flame Impingement:** Localized overheating from misdirected flames
- **Refractory Failure:** Hot spots, structural damage

---

## 2. Study Scope and Methodology

### 2.1 Scope

This HAZOP covers the GL-007 Furnace Performance Monitor including:

- Tube Metal Temperature (TMT) monitoring and protection
- Thermal efficiency calculation and tracking
- Fouling detection and heat transfer degradation analysis
- Alarm generation and operator notification

**Excluded from scope:**
- Burner operation (covered in GL-018 HAZOP)
- Mechanical design (covered in equipment HAZOP)
- Process fluid chemistry (covered in process HAZOP)

### 2.2 Methodology

The study follows IEC 61882:2016 with emphasis on API 560/530 requirements for fired heater safety.

### 2.3 Critical Temperatures

| Temperature Point | Description |
|-------------------|-------------|
| **TMT Design** | Maximum allowable metal temperature per API 530 |
| **TMT Alarm** | Warning threshold (typically TMT Design - 50F) |
| **TMT Trip** | Safety shutdown threshold (typically TMT Design) |
| **Bridgewall** | Temperature between radiant and convection sections |
| **Flue Gas** | Stack temperature for efficiency calculations |

---

## 3. System Description

### 3.1 GL-007 Furnace Performance Monitor Overview

The GL-007 agent provides real-time furnace monitoring with:

- **TMT Monitoring:** Skin thermocouple data processing
- **Efficiency Calculation:** Stack loss, radiation loss methods
- **Fouling Detection:** Heat transfer coefficient trending
- **Alarm Generation:** Multi-level alerts for operators

### 3.2 Furnace Zones

```
                    STACK
                      |
           +------------------+
           |    CONVECTION    |
           |    SECTION       |
           |  (Lower Temp)    |
           +------------------+
                    |
           +------------------+
           |   BRIDGEWALL     |
           +------------------+
                    |
           +------------------+
           |    RADIANT       |
           |    SECTION       |
           |  (Highest Temp)  |
           |                  |
           |   [BURNERS]      |
           +------------------+
                PROCESS
                IN/OUT
```

### 3.3 Key Parameters

| Parameter | Normal Range | Alarm | Trip |
|-----------|-------------|-------|------|
| TMT (Radiant) | 800-1200 F | 1400 F | 1500 F |
| TMT (Convection) | 600-900 F | 1000 F | 1100 F |
| Bridgewall Temp | 1400-1800 F | 1900 F | 2000 F |
| Flue Gas Temp | 350-500 F | 600 F | 700 F |
| Process Outlet Temp | Per design | Design + 25F | Design + 50F |
| Heat Flux | <20,000 BTU/hr-ft2 | Design | Design + 10% |

### 3.4 TMT Measurement System

```
Tube Surface
     |
     v
+-------------+
| Skin        |     Typical: Type K or N thermocouple
| Thermocouple|     welded/attached to tube surface
+-------------+
     |
     v
+-------------+
| Extension   |
| Wire        |     Routed through furnace wall
+-------------+
     |
     v
+-------------+
| Junction    |
| Box         |     Temperature reference
+-------------+
     |
     v
+-------------+
| I/O Card    |     DCS/SIS input
+-------------+
     |
     v
+-------------+
| GL-007      |     Monitoring, alarming, trending
| Agent       |
+-------------+
```

---

## 4. HAZOP Team and Responsibilities

### 4.1 Study Team

| Role | Name | Responsibility |
|------|------|----------------|
| **HAZOP Leader** | [Facilitator] | Study facilitation |
| **Furnace Engineer** | [Furnace SME] | Fired heater design |
| **Materials Engineer** | [Materials SME] | Metallurgy, creep |
| **Safety Engineer** | [Safety SME] | Safety systems |
| **Operations Rep** | [Ops SME] | Operating procedures |
| **Reliability** | [Reliability SME] | Failure modes, RBI |

---

## 5. Node 1: Tube Metal Temperature Monitoring

### 5.1 Node Description

**Design Intent:** Continuously monitor tube metal temperatures to prevent overtemperature damage including creep rupture, oxidation, and carburization. Provide early warning and automatic protection.

**Parameters:**
- TMT readings from skin thermocouples
- TMT design limit per API 530
- TMT margin (design - actual)
- Hot spot detection
- Temperature rate of change

### 5.2 API 530 Background

Tube design temperature per API 530 considers:
- Creep rupture life (typically 100,000 hours)
- Corrosion/erosion allowance
- Material degradation factors
- Operating margin

**Larson-Miller Parameter:** Used to predict remaining tube life based on temperature history.

### 5.3 HAZOP Worksheet - Node 1

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 1.1 | NO | No TMT reading | Thermocouple failure, wiring break | Blind to tube condition, no protection | TMT loss alarm, redundant TCs, manual inspection | R1.1 | H |
| 1.2 | NO | No TMT alarm generated | Alarm system failure, setpoint error | No operator notification, potential damage | Independent alarm channel, SIS trip backup | R1.2 | H |
| 1.3 | MORE | Higher TMT than actual (false high) | TC drift high, EMI, extension wire fault | Unnecessary trip, production loss | TC validation, comparison logic | R1.3 | M |
| 1.4 | MORE | Higher actual TMT than indicated (true high) | Fouling, flame impingement, coking | Tube damage, rupture, fire | SIS trip, operator action, maintenance | - | VH |
| 1.5 | MORE | Faster temperature rise than expected | Runaway reaction, fouling, flow loss | Rapid approach to limit, inadequate response time | Rate-of-change alarm, anticipatory shutdown | R1.4 | H |
| 1.6 | LESS | Lower TMT than actual (false low) | TC drift low, cold junction error | Underestimate severity, damage undetected | TC calibration, comparison, infrared backup | R1.5 | VH |
| 1.7 | LESS | Lower TMT margin than design | Hotter operation, capacity push | Reduced safety factor, accelerated creep | TMT margin monitoring, capacity limits | R1.6 | H |
| 1.8 | REVERSE | Inverted TMT signal | Wiring error, scaling error | Completely wrong indication | Commissioning verification, range checks | R1.7 | VH |
| 1.9 | AS WELL AS | TMT with noise | EMI, grounding issue, loose connections | Erratic readings, nuisance alarms | Shielding, filtering, wiring inspection | R1.8 | M |
| 1.10 | AS WELL AS | TMT with hot spot | Flame impingement, local fouling | Localized damage not representative | Sufficient TC coverage, infrared scanning | R1.9 | H |
| 1.11 | PART OF | Partial TMT coverage | TC failures, insufficient instrumentation | Hot spots undetected | Adequate TC density, portable IR scanning | R1.10 | H |
| 1.12 | OTHER THAN | TMT from wrong tube | Wiring error, tag error | Incorrect assessment | Tag verification, TC location documentation | R1.11 | H |

### 5.4 TMT Protection Hierarchy

| Level | Description | Response Time | Action |
|-------|-------------|---------------|--------|
| Level 1 | Advisory | Continuous | Operator trending |
| Level 2 | Alarm | <5 minutes | Operator investigation |
| Level 3 | High Alarm | <1 minute | Operator action required |
| Level 4 | Trip | <10 seconds | Automatic shutdown |

### 5.5 Node 1 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R1.1 | Install redundant TMT measurement at critical locations | Critical | Process | Open |
| R1.2 | Implement independent alarm channel in SIS | Critical | Safety | Open |
| R1.3 | Add TMT comparison logic to detect failed TCs | High | Software | Open |
| R1.4 | Implement rate-of-change alarm for rapid temperature rise | High | Controls | Open |
| R1.5 | Establish TC calibration program with cold junction verification | High | Instrument | Open |
| R1.6 | Add TMT margin KPI with capacity limit enforcement | High | Operations | Open |
| R1.7 | Verify TC wiring and scaling during commissioning | Critical | Commissioning | Open |
| R1.8 | Verify proper shielding and grounding of TC wiring | High | Instrument | Open |
| R1.9 | Implement periodic infrared scanning program | High | Inspection | Open |
| R1.10 | Review TC density against API 560 recommendations | High | Process | Open |
| R1.11 | Implement TC tag verification during startup | High | Operations | Open |

---

## 6. Node 2: Efficiency Calculations

### 6.1 Node Description

**Design Intent:** Calculate furnace thermal efficiency to track performance, identify degradation, and optimize operation. Use stack loss method and indirect efficiency calculation.

**Parameters:**
- Flue gas temperature
- Flue gas O2/CO2
- Ambient temperature
- Fuel higher heating value
- Stack losses
- Radiation losses

### 6.2 Efficiency Calculation Methods

**Indirect Method (Loss Method):**
```
Efficiency = 100% - Sum of Losses

Losses:
- Stack loss (sensible heat in flue gas)
- Radiation/convection loss (casing)
- Unburned combustibles (CO loss)
- Moisture loss
- Blowdown loss (boilers)
```

**Stack Loss Calculation:**
```
Stack Loss % = (Tstack - Tambient) x K / (CO2% or O2%)

Where K is fuel-specific constant
```

### 6.3 HAZOP Worksheet - Node 2

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 2.1 | NO | No efficiency calculation | Input data missing, calculation failure | Loss of performance monitoring | Calculation status monitoring, backup method | R2.1 | L |
| 2.2 | MORE | Higher calculated efficiency than actual | Input errors, formula error | False confidence, actual degradation hidden | Efficiency trending, heat balance validation | R2.2 | M |
| 2.3 | MORE | Higher indicated stack loss | Flue gas temp sensor drift high | Over-estimate losses, wrong conclusions | Sensor validation, multiple measurements | R2.3 | L |
| 2.4 | LESS | Lower calculated efficiency (pessimistic) | Conservative assumptions, input errors | Unnecessary maintenance, costs | Comparison with design, validation | R2.4 | L |
| 2.5 | LESS | Lower actual efficiency undetected | Algorithm insensitive, fouled sensors | Fuel waste, excess emissions | Efficiency KPIs, regular audits | R2.5 | M |
| 2.6 | REVERSE | Inverted efficiency trend | Data processing error | Wrong optimization direction | Trend validation, physical reasonableness | R2.6 | M |
| 2.7 | AS WELL AS | Efficiency with transient effects | Startup/shutdown, load changes | Unstable readings, wrong conclusions | Steady-state filter, transient exclusion | R2.7 | L |
| 2.8 | PART OF | Partial efficiency (missing losses) | Some inputs unavailable | Underestimate total losses | Complete loss accounting | R2.8 | L |
| 2.9 | OTHER THAN | Efficiency for wrong fuel | Fuel change not detected | Wrong calculation basis | Fuel type verification | R2.9 | M |

### 6.4 Node 2 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R2.1 | Add efficiency calculation status monitoring | Medium | Software | Open |
| R2.2 | Implement efficiency trending with deviation alerts | Medium | Software | Open |
| R2.3 | Add multi-sensor validation for flue gas temperature | Medium | Process | Open |
| R2.4 | Compare calculated efficiency with design periodically | Low | Engineering | Open |
| R2.5 | Establish efficiency KPIs with periodic audits | Medium | Operations | Open |
| R2.6 | Add physical reasonableness checks on efficiency | Medium | Software | Open |
| R2.7 | Implement steady-state detection for valid efficiency | Medium | Software | Open |
| R2.8 | Ensure complete loss accounting in algorithm | Low | Software | Open |
| R2.9 | Implement fuel type verification in calculation | Medium | Software | Open |

---

## 7. Node 3: Fouling Detection

### 7.1 Node Description

**Design Intent:** Detect heat transfer surface fouling through trending of tube skin temperatures, process outlet temperatures, and calculated heat transfer coefficients. Provide early warning for maintenance planning.

**Parameters:**
- Overall heat transfer coefficient (U)
- Fouling factor (Rf)
- Process outlet temperature deviation
- TMT-to-process temperature differential
- Cleaning/soot blowing effectiveness

### 7.2 Fouling Mechanisms

| Mechanism | Description | Detection Method |
|-----------|-------------|------------------|
| **Coking** | Carbon deposition on tube ID | High TMT at constant duty |
| **Scaling** | Mineral deposition | Increased pressure drop |
| **Soot** | Combustion product on tube OD | High bridgewall temp |
| **Ash** | Solid fuel residue | Reduced heat absorption |
| **Corrosion Products** | Oxide buildup | Gradual U decrease |

### 7.3 HAZOP Worksheet - Node 3

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 3.1 | NO | No fouling detection | Algorithm failure, insufficient data | Fouling undetected, surprise failure | Manual inspection program, TMT trends | R3.1 | M |
| 3.2 | MORE | Higher fouling rate than expected | Process upset, fuel change, poor combustion | Rapid degradation, unplanned shutdown | Fouling rate monitoring, process review | R3.2 | M |
| 3.3 | MORE | More severe fouling than indicated | Detection insensitive, wrong baseline | Actual fouling worse, tube damage | Periodic baseline reset, inspection | R3.3 | H |
| 3.4 | LESS | Lower fouling indication than actual | Compensating factors, sensor drift | False confidence, delayed maintenance | TMT trending, inspection validation | R3.4 | H |
| 3.5 | LESS | Less cleaning effectiveness than expected | Incomplete cleaning, wrong method | Fouling persists, performance gap | Post-cleaning verification | R3.5 | M |
| 3.6 | REVERSE | Fouling indication after cleaning | Wrong baseline, detection error | Confusion, wrong decisions | Pre/post cleaning comparison | R3.6 | L |
| 3.7 | AS WELL AS | Fouling with hot spot | Localized severe fouling | Localized overtemperature | TMT distribution analysis | R3.7 | H |
| 3.8 | PART OF | Partial fouling (localized) | Non-uniform flow, flame pattern | Some tubes worse than others | Per-tube TMT analysis, flow verification | R3.8 | M |
| 3.9 | OTHER THAN | Indication from other cause | Flame change, process upset | Misdiagnosis, wrong action | Multi-factor analysis | R3.9 | M |

### 7.4 Fouling Detection Algorithm

```
Fouling Factor Calculation:

Rf = (1/U_actual) - (1/U_clean)

Where:
  U = Q / (A x LMTD)
  Q = Heat duty (BTU/hr)
  A = Heat transfer area (ft2)
  LMTD = Log mean temperature difference

Fouling Indicator:
  - Rf > 0.002 hr-ft2-F/BTU: Moderate fouling
  - Rf > 0.005: Significant fouling
  - Rf > 0.01: Severe fouling, cleaning required
```

### 7.5 Node 3 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R3.1 | Implement backup manual inspection program | High | Inspection | Open |
| R3.2 | Add fouling rate trending with alerts | Medium | Software | Open |
| R3.3 | Establish periodic baseline reset schedule | Medium | Engineering | Open |
| R3.4 | Cross-check fouling detection with TMT trends | High | Software | Open |
| R3.5 | Implement post-cleaning verification routine | Medium | Maintenance | Open |
| R3.6 | Add pre/post cleaning comparison logic | Low | Software | Open |
| R3.7 | Implement TMT distribution analysis for hot spots | High | Software | Open |
| R3.8 | Add per-pass/per-tube analysis capability | Medium | Software | Open |
| R3.9 | Implement multi-factor fouling diagnosis | Medium | Software | Open |

---

## 8. Node 4: Alarm Generation

### 8.1 Node Description

**Design Intent:** Generate appropriate alarms for operator notification based on TMT, efficiency, and fouling status. Support alarm rationalization per ISA-18.2 and IEC 62682.

**Parameters:**
- Alarm setpoints (low, high, high-high)
- Alarm priorities (advisory, warning, critical)
- Alarm shelving/suppression status
- First-out indication
- Alarm rate

### 8.2 Alarm Hierarchy

| Priority | Description | Response | Color |
|----------|-------------|----------|-------|
| **Emergency** | Immediate threat to life/equipment | < 5 seconds | Red + Audio |
| **Critical** | Equipment damage imminent | < 1 minute | Red |
| **Warning** | Abnormal condition | < 10 minutes | Yellow |
| **Advisory** | Information for awareness | Acknowledge | Blue |

### 8.3 HAZOP Worksheet - Node 4

| Dev ID | Guide Word | Deviation | Cause | Consequence | Safeguard | Rec | Risk |
|--------|------------|-----------|-------|-------------|-----------|-----|------|
| 4.1 | NO | No alarm generated when required | Setpoint error, logic failure | Operator not notified, damage occurs | Independent trip, periodic alarm test | R4.1 | H |
| 4.2 | NO | No alarm acknowledgment path | HMI failure, network issue | Alarms not cleared, standing alarms | Redundant HMI, alarm summary | R4.2 | M |
| 4.3 | MORE | More alarms than operator can handle | Alarm flood, cascade | Important alarms missed | Alarm rationalization, suppression | R4.3 | H |
| 4.4 | MORE | Higher priority than appropriate | Conservative classification | Alarm fatigue, desensitization | Alarm priority review | R4.4 | M |
| 4.5 | LESS | Lower alarm than actual severity | Wrong classification, setpoint error | Inadequate response | Setpoint review, consequence analysis | R4.5 | H |
| 4.6 | LESS | Fewer alarms than needed | Insufficient coverage | Conditions undetected | Alarm coverage review | R4.6 | M |
| 4.7 | REVERSE | Alarm clears when should be active | Logic error, signal bounce | False sense of security | Alarm deadband, state machine | R4.7 | H |
| 4.8 | AS WELL AS | Alarm with chattering | Noise, process oscillation | Nuisance alarms, fatigue | Deadband, filtering, process fix | R4.8 | M |
| 4.9 | AS WELL AS | Alarm with wrong message | Configuration error | Operator confusion, wrong action | Message review, validation | R4.9 | M |
| 4.10 | PART OF | Partial alarm system | Some alarms disabled/shelved | Incomplete protection | Shelving limits, review | R4.10 | H |
| 4.11 | OTHER THAN | Alarm for wrong parameter | Tag error, display error | Wrong corrective action | Tag verification, display review | R4.11 | M |

### 8.4 Alarm Rationalization Requirements

Per ISA-18.2 / IEC 62682:

| Metric | Target | Maximum |
|--------|--------|---------|
| Standing Alarms | 0 | 10 |
| Alarms per 10 minutes (normal) | <1 | 2 |
| Alarms per 10 minutes (upset) | <10 | 15 |
| Chattering Alarms | 0 | None |
| Stale Alarms (>24 hr) | 0 | 5 |

### 8.5 Node 4 - Recommendations

| Rec ID | Description | Priority | Owner | Status |
|--------|-------------|----------|-------|--------|
| R4.1 | Implement periodic alarm testing program | Critical | Controls | Open |
| R4.2 | Install redundant alarm annunciator path | High | Controls | Open |
| R4.3 | Conduct alarm rationalization per ISA-18.2 | High | Operations | Open |
| R4.4 | Review and validate alarm priorities | High | Operations | Open |
| R4.5 | Verify alarm setpoints against consequence | High | Safety | Open |
| R4.6 | Conduct alarm coverage review vs. HAZOP | High | Safety | Open |
| R4.7 | Implement proper alarm deadband settings | High | Controls | Open |
| R4.8 | Address chattering alarms systematically | High | Controls | Open |
| R4.9 | Review alarm messages for clarity | Medium | Operations | Open |
| R4.10 | Implement shelving controls and review | High | Operations | Open |
| R4.11 | Verify alarm tag-to-parameter mapping | High | Controls | Open |

---

## 9. Risk Ranking Matrix

### 9.1 Risk Summary for GL-007

| Risk Level | Count | Key Deviations |
|------------|-------|----------------|
| Very High (VH) | 3 | True high TMT (1.4), False low TMT (1.6), Inverted TMT (1.8) |
| High (H) | 14 | TMT signal loss, alarm failures, fouling detection gaps |
| Medium (M) | 12 | Efficiency calculation issues, false alarms |
| Low (L) | 7 | Minor efficiency impacts, transient effects |

### 9.2 Furnace-Specific Risk Factors

| Hazard | Risk Multiplier | Basis |
|--------|-----------------|-------|
| Tube Rupture | x2.5 | Potential fatality, major fire |
| Hot Spot | x2 | Localized damage, escalation potential |
| Fouling Undetected | x1.5 | Leads to other hazards |
| Efficiency Loss | x0.5 | Economic impact only |

---

## 10. Action Items Summary

### 10.1 Critical Priority Actions (VH Risk)

| Action ID | Description | Owner | Due Date | Status |
|-----------|-------------|-------|----------|--------|
| R1.1 | Redundant TMT measurement at critical locations | Process | 2025-12-31 | Open |
| R1.2 | Independent alarm channel in SIS | Safety | 2025-12-31 | Open |
| R1.7 | TC wiring and scaling verification | Commissioning | 2025-12-31 | Open |
| R4.1 | Periodic alarm testing program | Controls | 2025-12-31 | Open |

### 10.2 High Priority Actions

| Action ID | Description | Owner | Due Date | Status |
|-----------|-------------|-------|----------|--------|
| R1.3 | TMT comparison logic for failed TCs | Software | 2026-01-31 | Open |
| R1.4 | Rate-of-change alarm implementation | Controls | 2026-01-31 | Open |
| R1.5 | TC calibration program | Instrument | 2026-01-31 | Open |
| R1.6 | TMT margin KPI with capacity limits | Operations | 2026-01-31 | Open |
| R1.8 | TC wiring shielding verification | Instrument | 2026-01-31 | Open |
| R1.9 | Periodic infrared scanning program | Inspection | 2026-01-31 | Open |
| R1.10 | TC density review vs. API 560 | Process | 2026-01-31 | Open |
| R1.11 | TC tag verification during startup | Operations | 2026-01-31 | Open |
| R3.1 | Manual inspection backup program | Inspection | 2026-01-31 | Open |
| R3.4 | Fouling cross-check with TMT | Software | 2026-01-31 | Open |
| R3.7 | TMT distribution analysis | Software | 2026-01-31 | Open |
| R4.2 | Redundant alarm annunciator | Controls | 2026-01-31 | Open |
| R4.3 | Alarm rationalization | Operations | 2026-01-31 | Open |
| R4.4 | Alarm priority review | Operations | 2026-01-31 | Open |
| R4.5 | Alarm setpoint verification | Safety | 2026-01-31 | Open |
| R4.6 | Alarm coverage review | Safety | 2026-01-31 | Open |
| R4.7 | Alarm deadband implementation | Controls | 2026-01-31 | Open |
| R4.8 | Chattering alarm resolution | Controls | 2026-01-31 | Open |
| R4.10 | Shelving controls and review | Operations | 2026-01-31 | Open |
| R4.11 | Alarm tag verification | Controls | 2026-01-31 | Open |

---

## 11. Appendices

### Appendix A: API 560 TMT Requirements

| Requirement | API 560 Reference | Implementation |
|-------------|-------------------|----------------|
| TMT Monitoring | 8.1.3.2 | Skin TCs required |
| TMT Alarms | 8.1.3.3 | High alarm at design - 50F |
| TMT Trip | 8.1.3.4 | Trip at design temperature |
| TC Quantity | 8.1.3.5 | Per-pass minimum, additional for long tubes |
| TC Location | 8.1.3.6 | Maximum flux zone, outlet end |

### Appendix B: Thermocouple Specifications

| Parameter | Specification |
|-----------|---------------|
| Type | Type K (Chromel-Alumel) or Type N |
| Range | 0-2000 F |
| Accuracy | +/- 2.2 C or 0.75% |
| Attachment | Welded pad or mechanical clamp |
| Extension Wire | Same type as TC, shielded |
| Response Time | <10 seconds |

### Appendix C: Abbreviations

| Abbreviation | Definition |
|--------------|------------|
| API | American Petroleum Institute |
| FFS | Fitness-For-Service |
| HMI | Human-Machine Interface |
| ISA | International Society of Automation |
| LMTD | Log Mean Temperature Difference |
| RBI | Risk-Based Inspection |
| TC | Thermocouple |
| TMT | Tube Metal Temperature |

### Appendix D: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang Process Heat Safety Documentation Package.*
