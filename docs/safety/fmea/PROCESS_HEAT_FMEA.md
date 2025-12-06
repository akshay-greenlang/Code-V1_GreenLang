# Failure Mode and Effects Analysis (FMEA) for Process Heat Safety Functions

## GreenLang Process Heat Agents GL-001 through GL-020

---

## Document Control

| Attribute | Value |
|-----------|-------|
| **Document ID** | GL-FMEA-001-REV1 |
| **Version** | 1.0 |
| **Effective Date** | 2025-12-06 |
| **Classification** | Safety Critical Documentation |
| **Author** | Safety Engineering Team |
| **Reviewed By** | Process Safety Lead |
| **Approved By** | Plant Safety Manager |

### Applicable Standards

| Standard | Title | Application |
|----------|-------|-------------|
| **IEC 60812:2018** | Failure Mode and Effects Analysis (FMEA and FMECA) | Primary methodology |
| **SAE J1739** | Potential Failure Mode and Effects Analysis (FMEA) | FMEA worksheet format |
| **AIAG FMEA 4th Edition** | Potential Failure Mode and Effects Analysis | Rating scales and action priority |
| **IEC 61508** | Functional Safety of E/E/PE Systems | Safety function classification |
| **IEC 61511** | Functional Safety - Process Industry | SIS design and validation |
| **NFPA 85** | Boiler and Combustion Systems Hazards Code | Combustion safety requirements |
| **NFPA 86** | Standard for Ovens and Furnaces | Furnace safety requirements |
| **API 560** | Fired Heaters for General Refinery Service | Fired heater requirements |

### Scope

This FMEA covers all safety functions implemented across the GreenLang Process Heat Agent suite:

- **GL-001 THERMOSYNC**: Process Heat Orchestrator - Master safety coordination
- **GL-002 FLAMEGUARD**: Boiler Combustion Optimizer - Combustion safety
- **GL-003 STEAMWISE**: Steam System Monitor - Pressure protection
- **GL-004 BURNMASTER**: Burner Control Optimizer - Fuel control safety
- **GL-005 COMBUSENSE**: Combustion Analyzer - Air/fuel ratio protection
- **GL-006 HEATRECLAIM**: Heat Recovery Optimizer - Heat exchanger protection
- **GL-007 FURNACEPULSE**: Furnace Performance Monitor - TMT protection
- **GL-008 TRAPCATCHER**: Steam Trap Monitor - Condensate safety
- **GL-009 THERMALIQ**: Thermal Intelligence - System-wide monitoring
- **GL-010 EMISSIONWATCH**: Emissions Monitor - Environmental compliance
- **GL-011 FUELCRAFT**: Fuel Management - Fuel quality protection
- **GL-012 STEAMQUAL**: Steam Quality Monitor - Steam purity safety
- **GL-013 PREDICTMAINT**: Predictive Maintenance - Equipment integrity
- **GL-014 EXCHANGER-PRO**: Heat Exchanger Optimizer - Thermal protection
- **GL-015 INSULSCAN**: Insulation Monitor - Surface temperature safety
- **GL-016 WATERGUARD**: Feedwater Monitor - Water quality protection
- **GL-017 CONDENSYNC**: Condensate System - Return water safety
- **GL-018 FLUEFLOW**: Flue Gas Optimizer - Stack safety
- **GL-019 HEATSCHEDULER**: Load Scheduler - Demand coordination
- **GL-020 ECONOPULSE**: Economics Optimizer - Cost-optimized safety

---

## Table of Contents

1. [Document Control](#document-control)
2. [Safety Function Inventory](#1-safety-function-inventory)
3. [FMEA Methodology](#2-fmea-methodology)
4. [FMEA Worksheets](#3-fmea-worksheets)
5. [RPN Summary and Prioritization](#4-rpn-summary-and-prioritization)
6. [Action Tracking Matrix](#5-action-tracking-matrix)
7. [Verification and Validation](#6-verification-and-validation)
8. [Appendices](#7-appendices)

---

## 1. Safety Function Inventory

### 1.1 Complete Safety Function Registry

| SF-ID | Safety Function | Agent | SIL | Response Time | Reference Standard |
|-------|-----------------|-------|-----|---------------|-------------------|
| **SF-001** | Emergency Shutdown Initiation | GL-001 | SIL 2 | <500ms | IEC 61511 |
| **SF-002** | High Temperature Trip | GL-007 | SIL 2 | <500ms | API 560 |
| **SF-003** | High Pressure Trip | GL-003 | SIL 2 | <500ms | ASME B31.1 |
| **SF-004** | Low Water Level Trip | GL-003 | SIL 3 | <300ms | NFPA 85 |
| **SF-005** | Flame Failure Detection | GL-018 | SIL 2 | <4s | NFPA 85 |
| **SF-006** | Excess Air Monitoring | GL-010 | SIL 1 | <10s | EPA regulations |
| **SF-007** | CO High Alarm | GL-010 | SIL 2 | <5s | OSHA/NIOSH |
| **SF-008** | Furnace Purge Sequence | GL-007 | SIL 2 | N/A | NFPA 86 |
| **SF-009** | Fuel Shutoff Valve Control | GL-018 | SIL 2 | <1s | NFPA 85 |
| **SF-010** | Stack Temperature Monitoring | GL-020 | SIL 1 | <30s | EPA/NFPA 85 |
| **SF-011** | Combustion Air Flow Interlock | GL-005 | SIL 2 | <3s | NFPA 85 |
| **SF-012** | Fuel Pressure Low Trip | GL-004 | SIL 2 | <3s | NFPA 85 |
| **SF-013** | High Fuel Gas Pressure Trip | GL-011 | SIL 2 | <1s | NFPA 54 |
| **SF-014** | Burner Management Sequence | GL-002 | SIL 2 | N/A | NFPA 85 |
| **SF-015** | Tube Metal Temperature (TMT) Alarm | GL-007 | SIL 1 | <10s | API 560 |
| **SF-016** | LEL Monitoring (Furnace Atmosphere) | GL-007 | SIL 2 | <5s | NFPA 86 |
| **SF-017** | Feedwater Flow Low Trip | GL-016 | SIL 2 | <2s | NFPA 85 |
| **SF-018** | Steam Drum Level Control | GL-003 | SIL 2 | Continuous | NFPA 85 |
| **SF-019** | Atomizing Media Pressure Trip | GL-004 | SIL 2 | <3s | NFPA 85 |
| **SF-020** | Safety Relief Valve Monitoring | GL-003 | SIL 1 | <1s | ASME VIII |

### 1.2 Safety Function to Agent Mapping

```
+------------------+     +------------------+     +------------------+
|   GL-001         |     |   GL-003         |     |   GL-007         |
|   THERMOSYNC     |     |   STEAMWISE      |     |   FURNACEPULSE   |
+------------------+     +------------------+     +------------------+
| SF-001: ESD Init |     | SF-003: Hi Press |     | SF-002: Hi Temp  |
|                  |     | SF-004: Lo Water |     | SF-008: Purge    |
|                  |     | SF-018: Drum Lvl |     | SF-015: TMT Alarm|
|                  |     | SF-020: PSV Mon  |     | SF-016: LEL Mon  |
+------------------+     +------------------+     +------------------+

+------------------+     +------------------+     +------------------+
|   GL-002/GL-018  |     |   GL-004         |     |   GL-005         |
|   FLAMEGUARD     |     |   BURNMASTER     |     |   COMBUSENSE     |
+------------------+     +------------------+     +------------------+
| SF-005: Flame    |     | SF-012: Lo Fuel  |     | SF-011: Air Flow |
| SF-009: Fuel SOV |     | SF-019: Atomize  |                        |
| SF-014: BMS Seq  |     |                  |                        |
+------------------+     +------------------+     +------------------+

+------------------+     +------------------+     +------------------+
|   GL-010         |     |   GL-011         |     |   GL-016         |
|   EMISSIONWATCH  |     |   FUELCRAFT      |     |   WATERGUARD     |
+------------------+     +------------------+     +------------------+
| SF-006: Excess O2|     | SF-013: Hi Fuel  |     | SF-017: Lo FW    |
| SF-007: CO High  |     |                  |                        |
+------------------+     +------------------+     +------------------+

+------------------+
|   GL-020         |
|   ECONOPULSE     |
+------------------+
| SF-010: Stack T  |
+------------------+
```

---

## 2. FMEA Methodology

### 2.1 Analysis Approach

This FMEA follows the IEC 60812:2018 methodology combined with the AIAG FMEA 4th Edition worksheet format. The analysis proceeds through the following phases:

1. **System Definition**: Define safety function boundaries and interfaces
2. **Failure Mode Identification**: Identify potential failure modes for each component
3. **Effect Analysis**: Determine local, system, and end effects of failures
4. **Cause Analysis**: Identify potential causes/mechanisms of failure
5. **Current Controls Review**: Evaluate existing prevention and detection measures
6. **Risk Assessment**: Assign Severity, Occurrence, and Detection ratings
7. **RPN Calculation**: Calculate Risk Priority Number (S x O x D)
8. **Action Planning**: Define recommended actions for high-RPN items

### 2.2 Rating Scales

#### 2.2.1 Severity Rating (S) - IEC 60812 / AIAG Aligned

| Rating | Category | Effect Description | Safety Impact |
|--------|----------|-------------------|---------------|
| **10** | Hazardous without warning | Potential fatality, no warning | Catastrophic |
| **9** | Hazardous with warning | Potential fatality, warning present | Major injury |
| **8** | Very High | Major injury or significant damage | Serious injury |
| **7** | High | Minor injury or major equipment damage | Lost time injury |
| **6** | Moderate | Equipment damage requiring repair | Medical treatment |
| **5** | Low | Equipment damage, no safety impact | First aid only |
| **4** | Very Low | Minor equipment effect | No injury |
| **3** | Minor | Slight performance degradation | No safety effect |
| **2** | Very Minor | Noticed by trained personnel | Negligible |
| **1** | None | No discernible effect | None |

#### 2.2.2 Occurrence Rating (O) - Based on Industry Data

| Rating | Category | Failure Rate | Annual Probability |
|--------|----------|--------------|-------------------|
| **10** | Very High | >100/1000 devices | >0.5 |
| **9** | | 50/1000 devices | 0.2 - 0.5 |
| **8** | High | 20/1000 devices | 0.1 - 0.2 |
| **7** | | 10/1000 devices | 0.05 - 0.1 |
| **6** | Moderate | 5/1000 devices | 0.02 - 0.05 |
| **5** | | 2/1000 devices | 0.01 - 0.02 |
| **4** | Low | 1/1000 devices | 0.005 - 0.01 |
| **3** | | 0.5/1000 devices | 0.002 - 0.005 |
| **2** | Remote | 0.1/1000 devices | 0.001 - 0.002 |
| **1** | Nearly Impossible | <0.01/1000 devices | <0.001 |

#### 2.2.3 Detection Rating (D) - Process Control Specific

| Rating | Category | Detection Method | Detection Probability |
|--------|----------|------------------|----------------------|
| **10** | Almost Impossible | No detection method exists | <1% |
| **9** | Very Remote | Random inspection only | 1-10% |
| **8** | Remote | Proof test detection only | 10-20% |
| **7** | Very Low | Infrequent diagnostic | 20-30% |
| **6** | Low | Periodic inspection | 30-50% |
| **5** | Moderate | Scheduled diagnostic test | 50-60% |
| **4** | Moderately High | Process indication | 60-70% |
| **3** | High | Continuous diagnostic monitoring | 70-80% |
| **2** | Very High | Automatic alarm on failure | 80-95% |
| **1** | Almost Certain | Fail-safe design, self-announcing | >95% |

### 2.3 RPN Calculation and Thresholds

```
RPN = Severity (S) x Occurrence (O) x Detection (D)

Maximum RPN = 10 x 10 x 10 = 1000
Minimum RPN = 1 x 1 x 1 = 1
```

#### RPN Action Thresholds

| RPN Range | Priority | Required Action |
|-----------|----------|-----------------|
| **>200** | **CRITICAL** | Immediate corrective action required |
| **101-200** | **HIGH** | Action required, track to completion |
| **51-100** | **MEDIUM** | Action recommended, review in next cycle |
| **1-50** | **LOW** | Monitor, no immediate action |

---

## 3. FMEA Worksheets

### 3.1 SF-001: Emergency Shutdown Initiation (GL-001)

**Function Description:** Initiate system-wide emergency shutdown upon detection of critical safety conditions. Coordinates shutdown sequence across all agents.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 001-01 | ESD Push Button | Stuck/Seized | Cannot initiate manual ESD, operator must use alternate | 8 | Mechanical wear, corrosion | 3 | Quarterly inspection | None - requires test | 6 | 144 | Add monthly function test, position switch | Controls | Q1 2026 | 8 | 2 | 3 | 48 |
| 001-02 | ESD Push Button | Spurious activation | Unplanned plant shutdown | 4 | Vibration, accidental contact | 3 | Protective cover, labeled | Immediate alarm | 2 | 24 | None required | - | - | - | - | - | - |
| 001-03 | GL-001 ESD Logic | Software hang/crash | ESD command not processed | 9 | Memory leak, infinite loop | 2 | Watchdog timer, heartbeat | Watchdog alarm | 2 | 36 | Add redundant logic solver | Safety | Q2 2026 | 9 | 2 | 2 | 36 |
| 001-04 | GL-001 ESD Logic | Incorrect trip sequence | Equipment damaged during shutdown | 7 | Logic error, configuration | 2 | FAT/SAT testing | Sequence verification | 4 | 56 | Enhanced sequence validation | Controls | Q1 2026 | 7 | 2 | 2 | 28 |
| 001-05 | ESD Communication Bus | Network failure | Agents not notified of ESD | 8 | Hardware failure, EMI | 3 | Redundant network path | Network monitoring | 3 | 72 | Hardwired backup path | Controls | Q1 2026 | 8 | 2 | 2 | 32 |
| 001-06 | ESD Output Relay | Fail to de-energize | Final element not commanded | 10 | Welded contacts, coil failure | 2 | De-energize to trip design | Proof testing | 7 | 140 | Add redundant relay (1oo2) | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 001-07 | ESD Output Relay | Spurious de-energize | Unplanned equipment trip | 4 | Loose connection, EMI | 3 | Shielded wiring | Trip alarm | 2 | 24 | None required | - | - | - | - | - | - |
| 001-08 | Hardwired Trip Bus | Open circuit | Loss of hardwired trip capability | 8 | Cable damage, termination | 2 | Protected cable routing | Line monitoring (EOL) | 3 | 48 | None required | - | - | - | - | - | - |
| 001-09 | Reset Circuit | Stuck - cannot reset | Extended downtime | 3 | Contact failure | 3 | Annual maintenance | Operator feedback | 4 | 36 | None required | - | - | - | - | - | - |
| 001-10 | Reset Circuit | Premature reset | Restart before safe | 8 | Operator error, bypass | 3 | Reset sequencing, time delay | Reset verification | 4 | 96 | Add process permissive checks | Controls | Q2 2026 | 8 | 2 | 2 | 32 |

---

### 3.2 SF-002: High Temperature Trip (GL-007)

**Function Description:** Trip fired heater on high tube metal temperature (TMT) to prevent tube damage and potential rupture.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 002-01 | TC-401A (TMT) | Drift low | Under-reads temperature, miss high | 9 | Aging, contamination | 4 | Calibration schedule | Cross-comparison | 5 | 180 | Add IR backup measurement | Instrumentation | Q1 2026 | 9 | 4 | 3 | 108 |
| 002-02 | TC-401A (TMT) | Drift high | False high reading, spurious trip | 4 | Grounding issue | 3 | Isolated installation | Comparison alarm | 3 | 36 | None required | - | - | - | - | - | - |
| 002-03 | TC-401A (TMT) | Open circuit | Loss of reading | 6 | Vibration, thermal stress | 4 | Strain relief | Open-wire detection | 2 | 48 | None required | - | - | - | - | - | - |
| 002-04 | TC-401A (TMT) | Detached from tube | Reads furnace ambient, not tube | 9 | Thermal cycling, poor weld | 3 | Proper attachment method | IR verification scan | 7 | 189 | Periodic IR thermography | Inspection | Q1 2026 | 9 | 2 | 3 | 54 |
| 002-05 | TC-401B (TMT) | Same failure modes as A | 2oo3 voting degraded | 7 | Same causes | 3 | Redundant sensors (2oo3) | Voting degradation alarm | 3 | 63 | Enhanced maintenance | Instrumentation | Q2 2026 | 7 | 2 | 3 | 42 |
| 002-06 | TC-401C (TMT) | Same failure modes as A | 2oo3 voting degraded | 7 | Same causes | 3 | Redundant sensors (2oo3) | Voting degradation alarm | 3 | 63 | Enhanced maintenance | Instrumentation | Q2 2026 | 7 | 2 | 3 | 42 |
| 002-07 | Input Module | Channel failure | Loss of one input | 6 | Component aging | 3 | Quality components | Channel diagnostics | 2 | 36 | None required | - | - | - | - | - | - |
| 002-08 | GL-007 Logic | Setpoint error | Trip at wrong temperature | 8 | Configuration error | 2 | MOC procedure | Setpoint audit | 4 | 64 | Auto setpoint validation | Software | Q2 2026 | 8 | 2 | 2 | 32 |
| 002-09 | GL-007 Logic | Comparison algorithm fault | Hot spots not detected | 8 | Software bug | 2 | Software testing | Regression testing | 5 | 80 | Add individual TC alarming | Software | Q1 2026 | 8 | 2 | 3 | 48 |
| 002-10 | Output to SIS | Communication failure | Trip signal not transmitted | 9 | Network failure | 3 | Hardwired backup | Communication monitoring | 3 | 81 | Add direct hardwired trip | Controls | Q1 2026 | 9 | 2 | 2 | 36 |
| 002-11 | Fuel Valve (XV) | Fail to close | Fuel continues, damage | 10 | Stuck stem, debris | 2 | Filtered fuel, valve design | Proof testing | 7 | 140 | Add redundant valve | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 002-12 | Fuel Valve (XV) | Leak through (closed) | Residual fuel, potential ignition | 8 | Seat damage, erosion | 3 | Double block & bleed | Leak testing | 6 | 144 | Implement leak testing program | Maintenance | Q2 2026 | 8 | 2 | 3 | 48 |

---

### 3.3 SF-003: High Pressure Trip (GL-003)

**Function Description:** Trip boiler/HRSG on high pressure to prevent vessel overpressure and potential rupture.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 003-01 | PT-201A | Drift low | Under-reads pressure, miss high | 9 | Diaphragm degradation | 4 | Calibration schedule | Cross-comparison | 6 | 216 | Add comparison diagnostics | Instrumentation | Q1 2026 | 9 | 3 | 3 | 81 |
| 003-02 | PT-201A | Drift high | False high, spurious trip | 4 | Process buildup | 3 | Blowdown schedule | Comparison alarm | 3 | 36 | None required | - | - | - | - | - | - |
| 003-03 | PT-201A | Plugged impulse line | Stuck reading, miss rise | 9 | Process solidification | 4 | Heat tracing, blowdown | Rate-of-change alarm | 7 | 252 | Implement line blowdown schedule | Operations | Q1 2026 | 9 | 2 | 4 | 72 |
| 003-04 | PT-201A | Diaphragm rupture | Loss of signal | 6 | Pressure spike, corrosion | 2 | Material selection | Failsafe to trip | 2 | 24 | None required | - | - | - | - | - | - |
| 003-05 | PT-201B | Same failure modes | 2oo3 voting degraded | 7 | Same causes | 3 | Redundant sensors | Voting degradation alarm | 3 | 63 | Enhanced maintenance | Instrumentation | Q2 2026 | 7 | 2 | 3 | 42 |
| 003-06 | PT-201C | Same failure modes | 2oo3 voting degraded | 7 | Same causes | 3 | Redundant sensors | Voting degradation alarm | 3 | 63 | Enhanced maintenance | Instrumentation | Q2 2026 | 7 | 2 | 3 | 42 |
| 003-07 | GL-003 Logic | Setpoint error | Trip at wrong pressure | 9 | Configuration error | 2 | MOC procedure | Setpoint audit | 4 | 72 | Auto setpoint validation | Software | Q2 2026 | 9 | 2 | 2 | 36 |
| 003-08 | GL-003 Logic | Voting logic error | Wrong trip decision | 9 | Software bug | 2 | Software testing | Logic verification | 5 | 90 | Enhanced logic testing | Software | Q2 2026 | 9 | 2 | 3 | 54 |
| 003-09 | SIS Output | Fail to command | Trip not initiated | 10 | Relay failure | 2 | De-energize to trip | Proof testing | 7 | 140 | Add redundant output | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 003-10 | PSV | Fail to lift | No pressure relief | 10 | Stuck, corrosion, setpoint drift | 2 | Annual PSV testing | Pop test records | 8 | 160 | Semi-annual testing | Maintenance | Q1 2026 | 10 | 2 | 4 | 80 |
| 003-11 | PSV | Lift too early | Lost inventory | 4 | Set too low | 3 | Setting verification | Pressure monitoring | 3 | 36 | None required | - | - | - | - | - | - |
| 003-12 | PSV | Fail to reseat | Continuous relief | 5 | Seat damage | 3 | Annual testing | Relief indication | 3 | 45 | None required | - | - | - | - | - | - |

---

### 3.4 SF-004: Low Water Level Trip (GL-003)

**Function Description:** Trip boiler on low water level to prevent dry firing and tube damage/rupture.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 004-01 | LT-301A | Drift high | Over-reads level, miss low | 10 | Reference leg degradation | 3 | Reference leg maintenance | Cross-comparison | 5 | 150 | Add DP transmitter diagnostics | Instrumentation | Q1 2026 | 10 | 2 | 3 | 60 |
| 004-02 | LT-301A | Drift low | False low, spurious trip | 4 | Wet leg evaporation | 4 | Filled reference leg | Comparison alarm | 3 | 48 | None required | - | - | - | - | - | - |
| 004-03 | LT-301A | Plugged sensing line | Stuck reading | 10 | Solids buildup | 4 | Blowdown schedule | Rate-of-change alarm | 7 | 280 | Daily blowdown procedure | Operations | Q1 2026 | 10 | 2 | 4 | 80 |
| 004-04 | LT-301B | Same failure modes | Degraded protection (1oo2) | 8 | Same causes | 3 | Diverse measurement (DP + guided wave) | Comparison alarm | 4 | 96 | Add third measurement | Safety | Q2 2026 | 8 | 2 | 3 | 48 |
| 004-05 | Level Switch (LS) | Stuck contacts | No trip on low level | 10 | Corrosion, mechanical | 2 | Quality switches | Proof testing | 8 | 160 | Monthly proof test | Operations | Q1 2026 | 10 | 2 | 4 | 80 |
| 004-06 | Level Switch (LS) | False trip | Spurious shutdown | 4 | Vibration, condensation | 4 | Protected mounting | Trip alarm | 2 | 32 | None required | - | - | - | - | - | - |
| 004-07 | GL-003 Logic | Low level setpoint error | Trip at wrong level | 10 | Configuration error | 2 | MOC procedure | Setpoint audit | 4 | 80 | Auto setpoint validation | Software | Q2 2026 | 10 | 2 | 2 | 40 |
| 004-08 | Feedwater Pump | Fail to start | Cannot recover level | 8 | Motor/starter failure | 3 | Redundant pump | Start failure alarm | 2 | 48 | None required | - | - | - | - | - | - |
| 004-09 | Feedwater Valve | Fail closed | Cannot recover level | 8 | Actuator failure | 2 | Air-to-open design | Position feedback | 3 | 48 | None required | - | - | - | - | - | - |
| 004-10 | Fuel Cutoff | Fail to activate | Dry firing continues | 10 | Relay failure | 2 | De-energize to trip | Proof testing | 7 | 140 | Add redundant cutoff | Safety | Q1 2026 | 10 | 2 | 3 | 60 |

---

### 3.5 SF-005: Flame Failure Detection (GL-018)

**Function Description:** Detect loss of flame and initiate fuel shutoff within 4 seconds per NFPA 85.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 005-01 | UV Scanner A | Blind (no flame signal) | Trips on good flame (1oo2) | 4 | Lens fouling | 5 | Scanner self-check | Flame loss alarm | 2 | 40 | None required | - | - | - | - | - | - |
| 005-02 | UV Scanner A | False flame (always sees) | No trip on actual flame loss | 10 | Internal fault | 2 | Self-check feature | Scanner diagnostics | 7 | 140 | Add internal shutter test | Instrumentation | Q1 2026 | 10 | 2 | 3 | 60 |
| 005-03 | UV Scanner A | Slow response (>4s) | Late trip, fuel accumulation | 9 | Component aging | 3 | Scanner specification | Response time test | 7 | 189 | Annual response testing | Instrumentation | Q1 2026 | 9 | 2 | 3 | 54 |
| 005-04 | UV Scanner A | Sensitivity drift | Marginal detection | 7 | Environmental degradation | 4 | Sensitivity adjustment | Flame quality trending | 5 | 140 | Implement trending | Controls | Q2 2026 | 7 | 3 | 3 | 63 |
| 005-05 | UV Scanner B | Same failure modes | Degraded protection | 7 | Same causes | 3 | Redundant scanner | Comparison logic | 4 | 84 | Add third scanner for 2oo3 | Safety | Q3 2026 | 7 | 2 | 3 | 42 |
| 005-06 | Scanner Amplifier | No output | Trips immediately | 4 | Component failure | 2 | Quality components | Output monitoring | 2 | 16 | None required | - | - | - | - | - | - |
| 005-07 | Scanner Amplifier | Fixed high output | False flame indicated | 10 | Internal short | 2 | Failsafe design | Comparison logic | 6 | 120 | Add amplifier diagnostics | Instrumentation | Q2 2026 | 10 | 2 | 3 | 60 |
| 005-08 | GL-018 FSI Logic | Calculation error | Flame instability not detected | 8 | Algorithm error | 2 | Software validation | FSI trending | 5 | 80 | Enhanced validation | Software | Q2 2026 | 8 | 2 | 3 | 48 |
| 005-09 | BMS Logic | Stuck in running state | No trip on flame loss | 10 | Logic solver failure | 2 | Watchdog monitoring | Proof testing | 7 | 140 | Add independent trip | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 005-10 | Main Gas Valve | Fail to close | Fuel continues to furnace | 10 | Stuck stem, debris | 2 | Filtered fuel, maintenance | Proof testing | 7 | 140 | Add double block & bleed | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 005-11 | Main Gas Valve | Leak through | Fuel accumulation | 9 | Seat damage | 3 | Double block design | Leak testing | 6 | 162 | Implement leak testing | Maintenance | Q1 2026 | 9 | 2 | 3 | 54 |
| 005-12 | Pilot Gas Valve | Fail to close | Pilot remains lit | 6 | Same as main | 3 | Same controls | Proof testing | 6 | 108 | Include in proof test | Maintenance | Q2 2026 | 6 | 2 | 3 | 36 |

---

### 3.6 SF-006: Excess Air Monitoring (GL-010)

**Function Description:** Monitor excess O2 in flue gas and alarm on high levels indicating combustion inefficiency or air leak.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 006-01 | O2 Analyzer (ZAT) | Drift low | Under-reads O2, miss excess air | 6 | Cell degradation | 4 | Calibration schedule | Cross-reference | 5 | 120 | Add auto-validation | Instrumentation | Q2 2026 | 6 | 3 | 3 | 54 |
| 006-02 | O2 Analyzer (ZAT) | Drift high | Over-reads O2, false alarm | 3 | Reference gas issue | 4 | Calibration | Operator awareness | 4 | 48 | None required | - | - | - | - | - | - |
| 006-03 | O2 Analyzer (ZAT) | Cell failure | Loss of reading | 5 | Thermal shock | 3 | Proper installation | Cell diagnostics | 2 | 30 | None required | - | - | - | - | - | - |
| 006-04 | Sample System | Plugged probe | Stale reading | 6 | Particulate | 4 | Filter maintenance | Sample flow monitor | 3 | 72 | Add flow alarm | Instrumentation | Q2 2026 | 6 | 3 | 2 | 36 |
| 006-05 | GL-010 Logic | Threshold error | Alarm at wrong level | 5 | Configuration error | 2 | MOC procedure | Setpoint audit | 4 | 40 | None required | - | - | - | - | - | - |
| 006-06 | GL-010 Logic | Data processing error | Incorrect calculations | 5 | Software bug | 2 | Software testing | Data validation | 4 | 40 | None required | - | - | - | - | - | - |
| 006-07 | Alarm System | Alarm suppressed | Operator not notified | 6 | Operator action | 3 | Alarm management | Suppression tracking | 4 | 72 | ISA 18.2 implementation | Controls | Q2 2026 | 6 | 2 | 3 | 36 |

---

### 3.7 SF-007: CO High Alarm (GL-010)

**Function Description:** Alarm on high CO indicating incomplete combustion or hazardous condition.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 007-01 | CO Analyzer | Drift low | Under-reads CO, miss high | 8 | Sensor poisoning | 4 | Sensor protection | Cross-reference | 6 | 192 | Add redundant analyzer | Instrumentation | Q1 2026 | 8 | 3 | 3 | 72 |
| 007-02 | CO Analyzer | Drift high | False high alarm | 4 | Interference gas | 3 | Selective sensor | Calibration | 3 | 36 | None required | - | - | - | - | - | - |
| 007-03 | CO Analyzer | Complete failure | Loss of protection | 8 | Cell end-of-life | 3 | Preventive replacement | Cell diagnostics | 3 | 72 | Add backup analyzer | Instrumentation | Q2 2026 | 8 | 2 | 2 | 32 |
| 007-04 | Sample Conditioning | Moisture interference | Incorrect reading | 6 | Cooler failure | 4 | Sample conditioning | Dew point monitor | 4 | 96 | Add automatic validation | Instrumentation | Q2 2026 | 6 | 3 | 3 | 54 |
| 007-05 | GL-010 Logic | Alarm threshold error | Trip at wrong level | 8 | Configuration error | 2 | MOC procedure | Setpoint audit | 4 | 64 | Auto setpoint validation | Software | Q2 2026 | 8 | 2 | 2 | 32 |
| 007-06 | GL-010 Logic | Averaging error | Peaks not detected | 7 | Algorithm design | 2 | Peak detection logic | Data review | 5 | 70 | Enhanced peak detection | Software | Q2 2026 | 7 | 2 | 3 | 42 |
| 007-07 | Alarm Interface | Alarm not annunciated | Operator unaware | 8 | System failure | 2 | Redundant HMI | Alarm system monitor | 3 | 48 | None required | - | - | - | - | - | - |
| 007-08 | Trip Interface | Trip not initiated | High CO continues | 9 | Communication failure | 2 | Hardwired backup | Trip confirmation | 4 | 72 | Add direct hardwired trip | Controls | Q1 2026 | 9 | 2 | 2 | 36 |

---

### 3.8 SF-008: Furnace Purge Sequence (GL-007)

**Function Description:** Execute timed furnace purge sequence before lightoff per NFPA 86 requirements.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 008-01 | Air Flow Transmitter | Reading low | Insufficient purge, explosion risk | 10 | Calibration drift | 3 | Calibration schedule | Cross-reference | 5 | 150 | Add redundant flow measurement | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 008-02 | Air Flow Transmitter | Reading high | Over-purge, time delay | 3 | Fouling | 3 | Cleaning schedule | Comparison | 4 | 36 | None required | - | - | - | - | - | - |
| 008-03 | Purge Timer | Timer too short | Incomplete purge | 10 | Configuration error | 2 | MOC procedure, interlocks | Timer verification | 4 | 80 | Add airflow accumulator | Controls | Q2 2026 | 10 | 2 | 2 | 40 |
| 008-04 | Purge Timer | Timer stuck | Cannot complete purge | 4 | Logic solver fault | 2 | Watchdog | Sequence monitoring | 3 | 24 | None required | - | - | - | - | - | - |
| 008-05 | GL-007 Sequence Logic | Skip purge step | Incomplete purge | 10 | Software bug | 2 | Software validation | Sequence verification | 5 | 100 | Enhanced testing | Software | Q2 2026 | 10 | 2 | 3 | 60 |
| 008-06 | GL-007 Sequence Logic | Wrong purge rate | Insufficient air changes | 9 | Configuration error | 2 | Design review | Airflow calculation | 5 | 90 | Add airflow totalization | Controls | Q2 2026 | 9 | 2 | 3 | 54 |
| 008-07 | Damper Position | Stuck partially open | Reduced purge flow | 9 | Mechanical binding | 3 | Preventive maintenance | Position feedback | 4 | 108 | Add position verification | Controls | Q1 2026 | 9 | 2 | 3 | 54 |
| 008-08 | Damper Position | Fails closed | No purge flow | 9 | Actuator failure | 2 | Air-to-open design | Position interlock | 3 | 54 | Enhanced interlock | Controls | Q2 2026 | 9 | 2 | 2 | 36 |
| 008-09 | Fan Operation | Fan trips during purge | Purge interrupted | 6 | Motor overload | 3 | Motor protection | Fan status interlock | 2 | 36 | None required | - | - | - | - | - | - |
| 008-10 | Interlock Logic | Bypassed | Purge not enforced | 10 | Operator bypass | 3 | Bypass tracking | Bypass alarm | 3 | 90 | Implement bypass management | Safety | Q1 2026 | 10 | 2 | 2 | 40 |

---

### 3.9 SF-009: Fuel Shutoff Valve Control (GL-018)

**Function Description:** Control main fuel shutoff valves for emergency and normal shutdown.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 009-01 | Main Fuel SOV-A | Fail to close | Fuel continues, fire/explosion | 10 | Stuck stem, debris | 2 | Filtered fuel, maintenance | Proof testing | 7 | 140 | Add redundant valve (2x) | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 009-02 | Main Fuel SOV-A | Leak through | Fuel accumulation in furnace | 9 | Seat damage, erosion | 3 | Hard seat design | Leak testing | 6 | 162 | Implement leak testing | Maintenance | Q1 2026 | 9 | 2 | 3 | 54 |
| 009-03 | Main Fuel SOV-A | Slow closure (>1s) | Extended hazard | 8 | Low actuator supply | 3 | Pressure regulation | Stroke timing | 5 | 120 | Add stroke timing test | Maintenance | Q2 2026 | 8 | 2 | 3 | 48 |
| 009-04 | Main Fuel SOV-B | Same failure modes | Redundancy compromised | 7 | Same causes | 3 | Double block design | Comparison | 4 | 84 | Enhanced testing | Maintenance | Q2 2026 | 7 | 2 | 3 | 42 |
| 009-05 | Vent Valve | Fail to open | No depressurization | 7 | Stuck | 2 | Maintenance | Position feedback | 5 | 70 | Add vent valve testing | Maintenance | Q2 2026 | 7 | 2 | 3 | 42 |
| 009-06 | Vent Valve | Leak through | Fuel release to vent | 6 | Seat damage | 3 | Quality valves | Vent monitoring | 4 | 72 | Add vent monitoring | Controls | Q2 2026 | 6 | 2 | 3 | 36 |
| 009-07 | Solenoid Pilot | Fail to energize | Cannot open valve | 4 | Coil failure | 2 | Quality components | Circuit monitoring | 3 | 24 | None required | - | - | - | - | - | - |
| 009-08 | Solenoid Pilot | Fail to de-energize | Cannot close valve | 10 | Stuck plunger | 2 | De-energize to trip | Proof testing | 7 | 140 | Add redundant solenoid | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 009-09 | GL-018 Logic | Wrong valve selected | Wrong valve operates | 7 | Configuration error | 2 | Design review | Sequence verification | 4 | 56 | Enhanced validation | Software | Q2 2026 | 7 | 2 | 2 | 28 |
| 009-10 | Position Feedback | False closed indication | Valve thought closed | 9 | Switch failure | 3 | Quality switches | Dual switches | 5 | 135 | Add redundant feedback | Controls | Q1 2026 | 9 | 2 | 3 | 54 |

---

### 3.10 SF-010: Stack Temperature Monitoring (GL-020)

**Function Description:** Monitor stack temperature for efficiency optimization and safety limits.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 010-01 | Stack TC | Drift low | Under-reads, miss high temp | 6 | Degradation | 4 | Calibration | Cross-reference | 5 | 120 | Add redundant TC | Instrumentation | Q2 2026 | 6 | 3 | 3 | 54 |
| 010-02 | Stack TC | Drift high | False high alarm | 3 | Grounding | 3 | Isolated installation | Comparison | 3 | 27 | None required | - | - | - | - | - | - |
| 010-03 | Stack TC | Open circuit | Loss of reading | 5 | Thermal fatigue | 4 | Quality TC | Open-wire detection | 2 | 40 | None required | - | - | - | - | - | - |
| 010-04 | GL-020 Logic | Efficiency calculation error | Wrong optimization | 5 | Algorithm error | 2 | Software testing | Energy balance | 4 | 40 | None required | - | - | - | - | - | - |
| 010-05 | GL-020 Logic | High limit threshold error | Alarm at wrong temp | 6 | Configuration | 2 | MOC procedure | Setpoint audit | 4 | 48 | None required | - | - | - | - | - | - |
| 010-06 | Alarm System | High temp alarm suppressed | Operator not notified | 6 | Operator action | 3 | Alarm management | Suppression tracking | 4 | 72 | ISA 18.2 implementation | Controls | Q2 2026 | 6 | 2 | 3 | 36 |

---

### 3.11 SF-011: Combustion Air Flow Interlock (GL-005)

**Function Description:** Trip on loss of combustion air to prevent rich combustion and explosion.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 011-01 | FT-501A | Drift high | Over-reads, miss low flow | 9 | Fouling | 4 | Calibration | Cross-reference | 5 | 180 | Add redundant transmitter | Safety | Q1 2026 | 9 | 3 | 3 | 81 |
| 011-02 | FT-501A | Drift low | Under-reads, spurious trip | 4 | Plugging | 4 | Sensing line maintenance | Comparison | 3 | 48 | None required | - | - | - | - | - | - |
| 011-03 | FT-501A | Plugged sensing | Stuck reading | 9 | Particulate | 4 | Filter/purge | Rate-of-change | 6 | 216 | Add sensing line purge | Instrumentation | Q1 2026 | 9 | 2 | 3 | 54 |
| 011-04 | FT-501B | Same failure modes | Degraded (1oo2) | 7 | Same causes | 3 | Redundancy | Comparison | 4 | 84 | Add third transmitter | Safety | Q2 2026 | 7 | 2 | 3 | 42 |
| 011-05 | Air Damper | Stuck closed | No air flow | 10 | Mechanical binding | 3 | Preventive maintenance | Position feedback | 4 | 120 | Add position interlock | Controls | Q1 2026 | 10 | 2 | 3 | 60 |
| 011-06 | Air Damper | Stuck open | Cannot reduce air | 4 | Same causes | 3 | Same controls | Position feedback | 4 | 48 | None required | - | - | - | - | - | - |
| 011-07 | FD Fan | Motor failure | No air flow | 9 | Overload, bearing | 3 | Vibration monitoring | Current/speed | 2 | 54 | Enhanced monitoring | Maintenance | Q2 2026 | 9 | 2 | 2 | 36 |
| 011-08 | GL-005 Cross-limit | Logic failure | No lead-air protection | 9 | Software bug | 2 | Software testing | Logic verification | 5 | 90 | Enhanced validation | Software | Q2 2026 | 9 | 2 | 3 | 54 |
| 011-09 | GL-005 Air-fuel ratio | Calculation error | Wrong ratio | 8 | Algorithm error | 2 | Validation | O2/CO monitoring | 4 | 64 | Add ratio verification | Controls | Q2 2026 | 8 | 2 | 3 | 48 |
| 011-10 | Trip Output | Fail to command | No trip on low air | 10 | Relay failure | 2 | De-energize to trip | Proof testing | 7 | 140 | Add redundant output | Safety | Q1 2026 | 10 | 2 | 3 | 60 |

---

### 3.12 SF-012: Fuel Pressure Low Trip (GL-004)

**Function Description:** Trip burner on low fuel pressure to prevent flame instability.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 012-01 | PS-301 | Drift high | Miss low pressure | 8 | Calibration drift | 4 | Calibration | Cross-reference | 6 | 192 | Add redundant switch | Instrumentation | Q1 2026 | 8 | 3 | 3 | 72 |
| 012-02 | PS-301 | Drift low | Spurious trip | 4 | Reference error | 3 | Calibration | Comparison | 3 | 36 | None required | - | - | - | - | - | - |
| 012-03 | PS-301 | Stuck contacts | No state change | 8 | Corrosion | 3 | Quality switches | Proof testing | 7 | 168 | Monthly testing | Maintenance | Q1 2026 | 8 | 2 | 3 | 48 |
| 012-04 | GL-004 Logic | Threshold error | Wrong setpoint | 7 | Configuration | 2 | MOC procedure | Setpoint audit | 4 | 56 | Auto validation | Software | Q2 2026 | 7 | 2 | 2 | 28 |
| 012-05 | Fuel Valve | Fail to close | Fuel continues | 9 | Stuck stem | 2 | Maintenance | Proof testing | 7 | 126 | Add redundant valve | Safety | Q2 2026 | 9 | 2 | 3 | 54 |
| 012-06 | Fuel Regulator | Stuck position | Cannot regulate | 6 | Mechanical | 4 | Maintenance | Position feedback | 4 | 96 | Enhanced monitoring | Controls | Q2 2026 | 6 | 3 | 3 | 54 |

---

### 3.13 SF-013: High Fuel Gas Pressure Trip (GL-011)

**Function Description:** Trip on high fuel gas pressure to prevent equipment damage and leaks.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 013-01 | PT-401 | Drift low | Miss high pressure | 8 | Diaphragm degradation | 3 | Calibration | Cross-reference | 5 | 120 | Add redundant transmitter | Safety | Q1 2026 | 8 | 2 | 3 | 48 |
| 013-02 | PT-401 | Drift high | Spurious trip | 4 | Impulse line | 3 | Line maintenance | Comparison | 3 | 36 | None required | - | - | - | - | - | - |
| 013-03 | Regulator | Fail open | High downstream pressure | 9 | Diaphragm rupture | 2 | Dual regulators | Pressure monitoring | 3 | 54 | Monitor regulator status | Controls | Q2 2026 | 9 | 2 | 2 | 36 |
| 013-04 | Relief Valve | Fail to lift | No relief | 9 | Corrosion, stuck | 2 | Annual testing | Pop test records | 7 | 126 | Semi-annual testing | Maintenance | Q1 2026 | 9 | 2 | 4 | 72 |
| 013-05 | GL-011 Logic | Setpoint error | Wrong trip level | 8 | Configuration | 2 | MOC procedure | Setpoint audit | 4 | 64 | Auto validation | Software | Q2 2026 | 8 | 2 | 2 | 32 |
| 013-06 | SOV | Fail to close | High pressure continues | 9 | Stuck stem | 2 | Maintenance | Proof testing | 7 | 126 | Add redundant valve | Safety | Q2 2026 | 9 | 2 | 3 | 54 |

---

### 3.14 SF-014: Burner Management Sequence (GL-002)

**Function Description:** Execute safe startup, operation, and shutdown sequences per NFPA 85.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 014-01 | BMS Logic Solver | Processor failure | Sequence stops, safe state | 4 | Hardware fault | 2 | Watchdog, redundancy | Diagnostics | 2 | 16 | None required | - | - | - | - | - | - |
| 014-02 | BMS Logic Solver | Wrong sequence executed | Hazardous startup | 9 | Configuration error | 2 | FAT/SAT testing | Sequence verification | 5 | 90 | Enhanced testing | Controls | Q2 2026 | 9 | 2 | 3 | 54 |
| 014-03 | Timing Circuits | Timer too short | Incomplete step | 9 | Configuration | 2 | MOC procedure | Timer verification | 4 | 72 | Add timing validation | Controls | Q2 2026 | 9 | 2 | 2 | 36 |
| 014-04 | Timing Circuits | Timer stuck | Sequence hangs | 4 | Hardware fault | 2 | Watchdog | Sequence monitoring | 3 | 24 | None required | - | - | - | - | - | - |
| 014-05 | GL-002 Interface | Communication loss | BMS standalone | 5 | Network failure | 3 | Standalone operation | Heartbeat monitor | 2 | 30 | None required | - | - | - | - | - | - |
| 014-06 | Permissive Logic | Bypassed permissive | Unsafe startup | 10 | Operator bypass | 3 | Bypass tracking | Bypass alarm | 3 | 90 | Implement bypass management | Safety | Q1 2026 | 10 | 2 | 2 | 40 |
| 014-07 | Ignition System | Fail to spark | No pilot light | 4 | Electrode wear | 4 | Preventive maintenance | Spark detection | 3 | 48 | None required | - | - | - | - | - | - |
| 014-08 | Ignition System | Continuous spark | Arc damage | 6 | Timer failure | 2 | Timing control | Timer monitoring | 4 | 48 | None required | - | - | - | - | - | - |
| 014-09 | Pilot Valve | Fail to open | No pilot gas | 4 | Solenoid failure | 3 | Quality valves | Position feedback | 3 | 36 | None required | - | - | - | - | - | - |
| 014-10 | Pilot Valve | Fail to close | Pilot gas continues | 7 | Stuck open | 2 | De-energize to close | Proof testing | 6 | 84 | Add to proof test | Maintenance | Q2 2026 | 7 | 2 | 3 | 42 |

---

### 3.15 SF-015: TMT High Alarm (GL-007)

**Function Description:** Alarm operator on high tube metal temperature before trip level.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 015-01 | TMT TC Array | Multiple drift low | Average under-reads | 8 | Systematic error | 3 | Diverse installation | Cross-check | 5 | 120 | Add IR backup | Inspection | Q1 2026 | 8 | 2 | 3 | 48 |
| 015-02 | TMT TC Array | Hot spot missed | Localized overheating | 8 | Inadequate coverage | 3 | Coverage analysis | IR scanning | 6 | 144 | Enhanced TC placement | Safety | Q2 2026 | 8 | 2 | 3 | 48 |
| 015-03 | GL-007 Averaging | Masks hot spots | Max not detected | 8 | Algorithm design | 2 | Per-TC evaluation | Max value tracking | 4 | 64 | Add individual alarming | Software | Q2 2026 | 8 | 2 | 2 | 32 |
| 015-04 | Alarm System | Alarm suppressed | Operator unaware | 7 | Operator action | 3 | Alarm management | Suppression tracking | 4 | 84 | ISA 18.2 implementation | Controls | Q2 2026 | 7 | 2 | 3 | 42 |
| 015-05 | Alarm Display | Display failure | No visualization | 6 | Hardware fault | 2 | Redundant display | Display monitoring | 3 | 36 | None required | - | - | - | - | - | - |
| 015-06 | HMI | Response slow | Delayed awareness | 6 | System load | 3 | Performance design | Performance monitoring | 4 | 72 | Optimize HMI | Software | Q2 2026 | 6 | 2 | 3 | 36 |

---

### 3.16 SF-016: LEL Monitoring (GL-007)

**Function Description:** Monitor LEL in furnace atmosphere and alarm/trip on high levels.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 016-01 | LEL Analyzer | Drift low | Under-reads, miss high | 10 | Sensor poisoning | 3 | Sensor protection | Calibration check | 6 | 180 | Add redundant analyzer | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 016-02 | LEL Analyzer | Drift high | False alarm | 4 | Interference | 3 | Selective sensor | Calibration | 3 | 36 | None required | - | - | - | - | - | - |
| 016-03 | LEL Analyzer | Response slow | Late detection | 9 | Fouling | 3 | Maintenance | Response testing | 6 | 162 | Add response testing | Instrumentation | Q1 2026 | 9 | 2 | 3 | 54 |
| 016-04 | Sample System | Plugged | No sample | 9 | Particulate | 4 | Filtering | Flow monitoring | 3 | 108 | Add sample flow alarm | Instrumentation | Q2 2026 | 9 | 3 | 2 | 54 |
| 016-05 | GL-007 Logic | Threshold error | Wrong alarm level | 9 | Configuration | 2 | MOC procedure | Setpoint audit | 4 | 72 | Auto validation | Software | Q2 2026 | 9 | 2 | 2 | 36 |
| 016-06 | Trip Output | Fail to command | No trip on high LEL | 10 | Relay failure | 2 | De-energize to trip | Proof testing | 7 | 140 | Add redundant output | Safety | Q1 2026 | 10 | 2 | 3 | 60 |

---

### 3.17 SF-017: Feedwater Flow Low Trip (GL-016)

**Function Description:** Trip boiler on loss of feedwater flow to prevent dry firing.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 017-01 | FT-601 | Drift high | Miss low flow | 9 | Fouling | 4 | Calibration | Cross-reference | 5 | 180 | Add redundant transmitter | Safety | Q1 2026 | 9 | 3 | 3 | 81 |
| 017-02 | FT-601 | Drift low | Spurious trip | 4 | Plugging | 4 | Sensing maintenance | Comparison | 3 | 48 | None required | - | - | - | - | - | - |
| 017-03 | FS-601 (Switch) | Stuck | No state change | 9 | Mechanical | 3 | Quality switches | Proof testing | 7 | 189 | Monthly testing | Maintenance | Q1 2026 | 9 | 2 | 3 | 54 |
| 017-04 | GL-016 Logic | Threshold error | Wrong trip level | 9 | Configuration | 2 | MOC procedure | Setpoint audit | 4 | 72 | Auto validation | Software | Q2 2026 | 9 | 2 | 2 | 36 |
| 017-05 | Trip Output | Fail to command | No trip | 10 | Relay failure | 2 | De-energize to trip | Proof testing | 7 | 140 | Add redundant output | Safety | Q1 2026 | 10 | 2 | 3 | 60 |
| 017-06 | FW Pump | Fail to start | Cannot recover | 8 | Motor failure | 3 | Redundant pump | Start failure alarm | 2 | 48 | None required | - | - | - | - | - | - |

---

### 3.18 SF-018: Steam Drum Level Control (GL-003)

**Function Description:** Maintain drum level within safe limits using 3-element control.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 018-01 | LT-701 | Drift | Wrong level indication | 7 | Reference leg | 4 | Maintenance | Cross-reference | 4 | 112 | Add diagnostics | Instrumentation | Q2 2026 | 7 | 3 | 3 | 63 |
| 018-02 | Steam Flow FT | Drift | Incorrect feedforward | 5 | Fouling | 4 | Calibration | Cross-check | 4 | 80 | Enhanced calibration | Instrumentation | Q2 2026 | 5 | 3 | 3 | 45 |
| 018-03 | FW Flow FT | Drift | Incorrect feedback | 5 | Same causes | 4 | Calibration | Cross-check | 4 | 80 | Enhanced calibration | Instrumentation | Q2 2026 | 5 | 3 | 3 | 45 |
| 018-04 | GL-003 Control Logic | Algorithm error | Level instability | 6 | Software bug | 2 | Testing | Level monitoring | 3 | 36 | None required | - | - | - | - | - | - |
| 018-05 | FW Control Valve | Fail closed | Level drops | 8 | Actuator failure | 2 | Air-to-open | Position feedback | 3 | 48 | None required | - | - | - | - | - | - |
| 018-06 | FW Control Valve | Fail open | Level rises | 6 | Same causes | 2 | Level trip | High level alarm | 2 | 24 | None required | - | - | - | - | - | - |
| 018-07 | 3-Element Selector | Wrong mode | Manual override | 5 | Operator error | 3 | Training | Mode indication | 4 | 60 | Add mode confirmation | Controls | Q2 2026 | 5 | 2 | 3 | 30 |

---

### 3.19 SF-019: Atomizing Media Pressure Trip (GL-004)

**Function Description:** Trip oil burner on loss of atomizing steam/air pressure.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 019-01 | PT-801 | Drift high | Miss low pressure | 8 | Calibration drift | 4 | Calibration | Cross-reference | 5 | 160 | Add redundant switch | Instrumentation | Q1 2026 | 8 | 3 | 3 | 72 |
| 019-02 | PT-801 | Drift low | Spurious trip | 4 | Line issue | 3 | Line maintenance | Comparison | 3 | 36 | None required | - | - | - | - | - | - |
| 019-03 | PS-801 | Stuck | No state change | 8 | Corrosion | 3 | Quality switches | Proof testing | 7 | 168 | Monthly testing | Maintenance | Q1 2026 | 8 | 2 | 3 | 48 |
| 019-04 | GL-004 Logic | Threshold error | Wrong trip level | 7 | Configuration | 2 | MOC procedure | Setpoint audit | 4 | 56 | Auto validation | Software | Q2 2026 | 7 | 2 | 2 | 28 |
| 019-05 | Trip Output | Fail to command | Oil continues | 9 | Relay failure | 2 | De-energize to trip | Proof testing | 7 | 126 | Add redundant output | Safety | Q2 2026 | 9 | 2 | 3 | 54 |
| 019-06 | Oil Valve | Fail to close | Poor atomization | 8 | Stuck stem | 2 | Maintenance | Proof testing | 6 | 96 | Enhanced testing | Maintenance | Q2 2026 | 8 | 2 | 3 | 48 |

---

### 3.20 SF-020: Safety Relief Valve Monitoring (GL-003)

**Function Description:** Monitor PSV status and alert on lift events.

| FM-ID | Item/Function | Potential Failure Mode | Potential Effect(s) of Failure | S | Potential Cause(s) | O | Current Design Controls (Prevention) | Current Design Controls (Detection) | D | RPN | Recommended Action | Responsibility | Target Date | S' | O' | D' | RPN' |
|-------|---------------|----------------------|-------------------------------|---|-------------------|---|-------------------------------------|-------------------------------------|---|-----|-------------------|----------------|-------------|----|----|----|----|
| 020-01 | Acoustic Monitor | Fail to detect | Miss lift event | 6 | Sensor failure | 3 | Quality sensors | Self-test | 4 | 72 | Add redundant monitor | Instrumentation | Q2 2026 | 6 | 2 | 3 | 36 |
| 020-02 | Acoustic Monitor | False detection | False alarm | 3 | Noise interference | 4 | Signal processing | Event verification | 3 | 36 | None required | - | - | - | - | - | - |
| 020-03 | Temperature Monitor | Drift | Miss lift indication | 6 | Calibration | 3 | Calibration | Cross-reference | 4 | 72 | Enhanced calibration | Instrumentation | Q2 2026 | 6 | 2 | 3 | 36 |
| 020-04 | GL-003 Logic | Event processing error | Miss recorded event | 5 | Software bug | 2 | Testing | Event logging | 4 | 40 | None required | - | - | - | - | - | - |
| 020-05 | Alert System | Alert not generated | Operator unaware | 6 | System failure | 2 | Redundant alerting | Alert confirmation | 3 | 36 | None required | - | - | - | - | - | - |

---

## 4. RPN Summary and Prioritization

### 4.1 Critical Items (RPN > 200) - IMMEDIATE ACTION REQUIRED

| Rank | FM-ID | Safety Function | Failure Mode | RPN | Primary Effect | Recommended Action |
|------|-------|-----------------|--------------|-----|----------------|-------------------|
| **1** | 004-03 | SF-004 Low Water Level | Plugged sensing line | **280** | Miss low level, dry firing | Daily blowdown procedure |
| **2** | 003-03 | SF-003 High Pressure | Plugged impulse line | **252** | Miss pressure rise | Implement line blowdown schedule |
| **3** | 011-03 | SF-011 Combustion Air | Plugged sensing | **216** | Miss low air flow | Add sensing line purge |
| **4** | 003-01 | SF-003 High Pressure | Sensor drift low | **216** | Miss high pressure | Add comparison diagnostics |

### 4.2 High Priority Items (RPN 101-200) - ACTION REQUIRED

| Rank | FM-ID | Safety Function | Failure Mode | RPN | Recommended Action |
|------|-------|-----------------|--------------|-----|-------------------|
| 5 | 007-01 | SF-007 CO High | Analyzer drift low | 192 | Add redundant analyzer |
| 6 | 012-01 | SF-012 Fuel Low | Switch drift high | 192 | Add redundant switch |
| 7 | 002-04 | SF-002 High Temp | TC detached | 189 | Periodic IR thermography |
| 8 | 005-03 | SF-005 Flame | Scanner slow response | 189 | Annual response testing |
| 9 | 017-03 | SF-017 FW Flow Low | Switch stuck | 189 | Monthly testing |
| 10 | 011-01 | SF-011 Air Flow | Transmitter drift high | 180 | Add redundant transmitter |
| 11 | 016-01 | SF-016 LEL | Analyzer drift low | 180 | Add redundant analyzer |
| 12 | 002-01 | SF-002 High Temp | TC drift low | 180 | Add IR backup measurement |
| 13 | 017-01 | SF-017 FW Flow | Transmitter drift high | 180 | Add redundant transmitter |
| 14 | 012-03 | SF-012 Fuel Low | Switch stuck contacts | 168 | Monthly testing |
| 15 | 019-03 | SF-019 Atomizing | Switch stuck | 168 | Monthly testing |
| 16 | 009-02 | SF-009 Fuel SOV | Valve leak through | 162 | Implement leak testing |
| 17 | 016-03 | SF-016 LEL | Slow response | 162 | Add response testing |
| 18 | 003-10 | SF-003 High Pressure | PSV fail to lift | 160 | Semi-annual testing |
| 19 | 019-01 | SF-019 Atomizing | Transmitter drift high | 160 | Add redundant switch |
| 20 | 004-05 | SF-004 Low Water | Level switch stuck | 160 | Monthly proof test |
| 21 | 008-01 | SF-008 Purge | Air flow transmitter low | 150 | Add redundant flow measurement |
| 22 | 004-01 | SF-004 Low Water | Transmitter drift high | 150 | Add DP diagnostics |
| 23 | 015-02 | SF-015 TMT Alarm | Hot spot missed | 144 | Enhanced TC placement |
| 24 | 002-12 | SF-002 High Temp | Valve leak through | 144 | Implement leak testing |
| 25 | 001-01 | SF-001 ESD | Push button stuck | 144 | Add monthly function test |
| 26 | Multiple | Various | Output relay failures | 140 | Add redundant relays |
| 27 | Multiple | Various | Valve fail to close | 140 | Add redundant valves |
| 28 | 009-10 | SF-009 Fuel SOV | False closed indication | 135 | Add redundant feedback |
| 29 | Multiple | Various | Various logic/software | 100-135 | Enhanced validation |

### 4.3 Medium Priority Items (RPN 51-100)

| Category | Count | Typical Actions |
|----------|-------|-----------------|
| Sensor calibration/maintenance | 18 | Enhanced calibration schedules |
| Software/logic validation | 12 | Enhanced testing programs |
| Alarm management | 6 | ISA 18.2 implementation |
| Proof testing | 8 | Monthly/quarterly proof tests |
| Diagnostics | 7 | Add comparison/diagnostics |

### 4.4 Low Priority Items (RPN 1-50)

| Category | Count | Action |
|----------|-------|--------|
| Fail-safe failures (spurious trip) | 24 | Monitor only |
| Redundant element failures | 16 | Monitor only |
| Minor equipment effects | 12 | Monitor only |

### 4.5 RPN Distribution Summary

```
RPN DISTRIBUTION (Total Items: 186)

>200 (CRITICAL):      4 items  (2.2%)   [====                    ]
101-200 (HIGH):      28 items (15.0%)   [========                 ]
51-100 (MEDIUM):     51 items (27.4%)   [=============            ]
1-50 (LOW):         103 items (55.4%)   [=======================  ]

OVERALL RISK PROFILE:
- Critical items requiring immediate action: 4
- High priority items requiring tracking: 28
- Items with recommended actions: 83
- Items requiring monitoring only: 103
```

---

## 5. Action Tracking Matrix

### 5.1 Immediate Actions (Q1 2026) - Critical/High Priority

| Action ID | FM-ID(s) | Action Description | Responsible | Target Date | Status | Verification |
|-----------|----------|-------------------|-------------|-------------|--------|--------------|
| ACT-001 | 004-03 | Implement daily sensing line blowdown for LT-301 | Operations | 2026-01-15 | Open | Procedure review |
| ACT-002 | 003-03 | Implement impulse line blowdown schedule for PT-201 | Operations | 2026-01-15 | Open | Procedure review |
| ACT-003 | 011-03 | Install sensing line purge system for FT-501 | Instrumentation | 2026-02-28 | Open | Installation verification |
| ACT-004 | 003-01 | Add sensor comparison diagnostics for PT-201 A/B/C | Controls | 2026-02-28 | Open | FAT/SAT |
| ACT-005 | Multiple | Add redundant trip outputs for all SIF functions | Safety | 2026-03-31 | Open | SIL verification |
| ACT-006 | 005-02,005-09 | Add UV scanner internal shutter test | Instrumentation | 2026-02-28 | Open | FAT/SAT |
| ACT-007 | 002-04 | Implement periodic IR thermography program | Inspection | 2026-01-31 | Open | Procedure approval |
| ACT-008 | 007-01 | Install redundant CO analyzer | Instrumentation | 2026-03-31 | Open | Installation verification |
| ACT-009 | 016-01 | Install redundant LEL analyzer | Instrumentation | 2026-03-31 | Open | Installation verification |
| ACT-010 | 009-02,002-12 | Implement fuel valve leak testing program | Maintenance | 2026-02-28 | Open | Procedure approval |
| ACT-011 | Multiple | Implement monthly proof test program | Maintenance | 2026-01-31 | Open | Procedure approval |
| ACT-012 | 005-03 | Add annual flame scanner response testing | Instrumentation | 2026-01-31 | Open | Procedure approval |
| ACT-013 | 014-06,008-10 | Implement bypass management system | Safety | 2026-02-28 | Open | System commissioning |
| ACT-014 | 001-06 | Add redundant ESD output relay (1oo2) | Safety | 2026-03-31 | Open | FAT/SAT |

### 5.2 Short-Term Actions (Q2 2026) - High/Medium Priority

| Action ID | FM-ID(s) | Action Description | Responsible | Target Date | Status |
|-----------|----------|-------------------|-------------|-------------|--------|
| ACT-015 | Multiple | Implement auto setpoint validation | Software | 2026-04-30 | Open |
| ACT-016 | Multiple | Enhanced software validation testing | Software | 2026-05-31 | Open |
| ACT-017 | Multiple | Implement ISA 18.2 alarm management | Controls | 2026-06-30 | Open |
| ACT-018 | 005-04 | Implement flame scanner sensitivity trending | Controls | 2026-04-30 | Open |
| ACT-019 | Multiple | Add position verification for dampers | Controls | 2026-05-31 | Open |
| ACT-020 | 008-06 | Add airflow totalization for purge | Controls | 2026-04-30 | Open |
| ACT-021 | 015-03 | Add individual TC alarming | Software | 2026-05-31 | Open |
| ACT-022 | Multiple | Enhanced proof testing schedule | Maintenance | 2026-04-30 | Open |
| ACT-023 | 003-10,013-04 | Implement semi-annual PSV testing | Maintenance | 2026-04-30 | Open |
| ACT-024 | 005-05 | Add third flame scanner for 2oo3 voting | Safety | 2026-06-30 | Open |

### 5.3 Medium-Term Actions (Q3-Q4 2026)

| Action ID | FM-ID(s) | Action Description | Responsible | Target Date | Status |
|-----------|----------|-------------------|-------------|-------------|--------|
| ACT-025 | Multiple | Add 2oo3 voting for all SIF inputs | Safety | 2026-09-30 | Open |
| ACT-026 | 001-03 | Add redundant logic solver | Safety | 2026-09-30 | Open |
| ACT-027 | Multiple | Complete FMEA RPN reduction verification | Safety | 2026-12-31 | Open |

### 5.4 Action Status Summary

```
ACTION STATUS (Total: 27)

Completed:     0 (0%)    [                        ]
In Progress:   0 (0%)    [                        ]
Open:         27 (100%)  [========================]

BY QUARTER:
Q1 2026: 14 actions (52%)
Q2 2026: 10 actions (37%)
Q3-Q4 2026: 3 actions (11%)

BY RESPONSIBILITY:
Safety:          8 actions
Instrumentation: 7 actions
Controls:        5 actions
Maintenance:     4 actions
Software:        3 actions
Operations:      2 actions
Inspection:      1 action
```

---

## 6. Verification and Validation

### 6.1 FMEA Review Schedule

| Review Type | Frequency | Participants | Documentation |
|-------------|-----------|--------------|---------------|
| Action Status Review | Monthly | FMEA Lead, Action Owners | Status report |
| RPN Verification | Quarterly | Safety Team | RPN recalculation |
| Full FMEA Review | Annual | Full HAZOP Team | Updated FMEA document |
| Post-Incident Review | As needed | Investigation Team | FMEA addendum |

### 6.2 Effectiveness Verification

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Critical RPNs (>200) | 0 | 4 | Action required |
| High RPNs (101-200) | <10 | 28 | Action required |
| Average RPN | <50 | 72 | Improvement needed |
| Action completion rate | 100% | 0% | In progress |
| SFF (all functions) | >60% | 52% | Improvement needed |
| DC (all functions) | >60% | 26% | Improvement needed |

### 6.3 Post-Action RPN Targets

| Category | Current | Target After Actions | Reduction |
|----------|---------|---------------------|-----------|
| Critical (>200) | 4 | 0 | 100% |
| High (101-200) | 28 | 8 | 71% |
| Medium (51-100) | 51 | 35 | 31% |
| Total RPN Sum | 13,392 | 7,500 | 44% |

---

## 7. Appendices

### Appendix A: Acronyms and Abbreviations

| Acronym | Definition |
|---------|------------|
| BMS | Burner Management System |
| CCF | Common Cause Failure |
| DC | Diagnostic Coverage |
| DD | Dangerous Detected |
| DU | Dangerous Undetected |
| ESD | Emergency Shutdown |
| FAT | Factory Acceptance Test |
| FD | Forced Draft |
| FMEA | Failure Mode and Effects Analysis |
| FSI | Flame Stability Index |
| FT | Flow Transmitter |
| HMI | Human Machine Interface |
| IR | Infrared |
| LEL | Lower Explosive Limit |
| LT | Level Transmitter |
| MOC | Management of Change |
| NFPA | National Fire Protection Association |
| O2 | Oxygen |
| OPC-UA | Open Platform Communications Unified Architecture |
| PFD | Probability of Failure on Demand |
| PS | Pressure Switch |
| PSV | Pressure Safety Valve |
| PT | Pressure Transmitter |
| RPN | Risk Priority Number |
| SAT | Site Acceptance Test |
| SD | Safe Detected |
| SFF | Safe Failure Fraction |
| SIF | Safety Instrumented Function |
| SIL | Safety Integrity Level |
| SIS | Safety Instrumented System |
| SOV | Shutoff Valve |
| SU | Safe Undetected |
| TC | Thermocouple |
| TMT | Tube Metal Temperature |
| UV | Ultraviolet |

### Appendix B: Reference Standards

| Standard | Title | Application |
|----------|-------|-------------|
| IEC 60812:2018 | Failure Mode and Effects Analysis | FMEA methodology |
| IEC 61508 | Functional Safety | SIL determination |
| IEC 61511 | Process Industry SIS | SIS lifecycle |
| SAE J1739 | Potential FMEA | Worksheet format |
| AIAG FMEA 4th Ed | FMEA Reference Manual | Rating scales |
| NFPA 85 | Boiler and Combustion | BMS requirements |
| NFPA 86 | Ovens and Furnaces | Furnace safety |
| API 560 | Fired Heaters | TMT requirements |
| ISA 18.2 | Alarm Management | Alarm optimization |
| ASME B31.1 | Power Piping | Pressure limits |

### Appendix C: Related Documents

| Document ID | Title | Relationship |
|-------------|-------|--------------|
| GL-HAZOP-001 | GL-001 HAZOP Study | Input scenarios |
| GL-HAZOP-018 | GL-018 HAZOP Study | Input scenarios |
| GL-HAZOP-007 | GL-007 HAZOP Study | Input scenarios |
| GL-SRS-001 | Safety Requirements Specification | SIF requirements |
| GL-LOPA-001 | LOPA Analysis | SIL targets |
| GL-RISK-001 | Risk Matrix | Risk acceptance criteria |
| GL-BYPASS-001 | Bypass Management Procedure | Interlock bypass |
| GL-PROOF-001 | Proof Test Procedures | Testing requirements |

### Appendix D: Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2025-12-06 | Safety Engineering Team | Initial release |

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Prepared By | Safety Engineering Team | | 2025-12-06 |
| Reviewed By | Process Safety Lead | | |
| Approved By | Plant Safety Manager | | |

---

**Document End**

*This document is part of the GreenLang Process Heat Safety Documentation Package and implements TASK-204 per the Phase 5 Safety and Compliance Gap Analysis.*

*GL-FMEA-001-REV1 | IEC 60812:2018 Compliant | GreenLang Process Heat Agents*
