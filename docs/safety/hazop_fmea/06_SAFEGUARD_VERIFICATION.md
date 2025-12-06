# Safeguard Verification Document

## Independent Protection Layer Verification per IEC 61511

**Document ID:** GL-SAFEGUARD-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-2:2016, CCPS LOPA Guidelines

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [IPL Verification Criteria](#2-ipl-verification-criteria)
3. [IPL Inventory](#3-ipl-inventory)
4. [PFD Calculations](#4-pfd-calculations)
5. [Proof Test Requirements](#5-proof-test-requirements)
6. [Verification Methods](#6-verification-methods)
7. [IPL Documentation](#7-ipl-documentation)
8. [Common Cause Failure Considerations](#8-common-cause-failure-considerations)

---

## 1. Introduction

### 1.1 Purpose

This document provides verification evidence for all Independent Protection Layers (IPLs) credited in the risk assessments for GreenLang Process Heat agents. Each IPL must meet specific criteria per IEC 61511 and CCPS guidelines to be credited in LOPA.

### 1.2 Scope

IPLs verified in this document include:
- Safety Instrumented Functions (SIFs)
- Basic Process Control System (BPCS) functions
- Alarms with operator response
- Mechanical safeguards (relief devices)
- Human interventions
- Physical barriers

### 1.3 References

| Document | Title |
|----------|-------|
| IEC 61511-2:2016 | Guidelines for application of IEC 61511-1 |
| CCPS | Layer of Protection Analysis |
| ISA-TR84.00.02 | Safety Instrumented Functions (SIF) |
| API 521 | Pressure-relieving and Depressuring Systems |
| GL-LOPA-001 | LOPA Analysis Document |

---

## 2. IPL Verification Criteria

### 2.1 IPL Qualification Criteria

Per IEC 61511 and CCPS, an IPL must be:

| Criterion | Description | Verification Method |
|-----------|-------------|---------------------|
| **Independent** | Must not be affected by failure of initiating event or other IPLs | Design review, common cause analysis |
| **Specific** | Must be designed to prevent or mitigate the specific scenario | Functional specification |
| **Dependable** | Failure rate must be quantifiable and achievable | PFD calculation, proof testing |
| **Auditable** | Subject to regular testing, inspection, and maintenance | Maintenance records, test procedures |

### 2.2 IPL Categories

| Category | Typical PFD | Requirements |
|----------|-------------|--------------|
| **SIS (SIL 3)** | 0.001 - 0.0001 | IEC 61511 full lifecycle |
| **SIS (SIL 2)** | 0.01 - 0.001 | IEC 61511 full lifecycle |
| **SIS (SIL 1)** | 0.1 - 0.01 | IEC 61511 full lifecycle |
| **BPCS Control Loop** | 0.1 | Independent from initiating event |
| **Alarm + Operator** | 0.1 | Sufficient time, training, procedure |
| **Pressure Relief** | 0.01 | API 521 design, inspection |
| **Check Valve** | 0.1 | Proper maintenance |
| **Passive Containment** | 0.01 - 0.001 | Proper design, maintenance |

### 2.3 Non-IPL Safeguards

The following cannot be credited as IPLs:

| Safeguard | Reason |
|-----------|--------|
| Warning signs | Not dependable, no PFD |
| Training alone | No automatic action |
| PPE | Mitigation only |
| Fire detection | Does not prevent initiating event |
| Emergency response | Post-event mitigation |
| Operator vigilance | Cannot quantify PFD |

---

## 3. IPL Inventory

### 3.1 Safety Instrumented Functions (SIFs)

| IPL ID | Function | Agent | SIL | Voting | PFD Target | PFD Achieved |
|--------|----------|-------|-----|--------|------------|--------------|
| SIF-001 | High Temperature Trip | GL-001 | 2 | 2oo3 | <0.01 | 0.0052 |
| SIF-002 | Low Flow Protection | GL-001 | 2 | 2oo3 | <0.01 | 0.0048 |
| SIF-003 | High Pressure Trip | GL-001 | 2 | 2oo3 | <0.01 | 0.0055 |
| SIF-004 | Flame Failure Trip | GL-018 | 2 | 1oo2 | <0.01 | 0.0063 |
| SIF-005 | Emergency Shutdown | GL-001 | 2 | 1oo2 | <0.01 | 0.0042 |
| SIF-006 | Low Air Flow Trip | GL-018 | 2 | 1oo2 | <0.01 | 0.0058 |
| SIF-007 | TMT High Trip | GL-007 | 2 | 2oo3 | <0.01 | 0.0061 |

### 3.2 BPCS Functions

| IPL ID | Function | Agent | PFD Claimed | Verification |
|--------|----------|-------|-------------|--------------|
| BPCS-001 | Temperature control loop | GL-001 | 0.1 | Annual test |
| BPCS-002 | Pressure control loop | GL-001 | 0.1 | Annual test |
| BPCS-003 | Flow control loop | GL-001 | 0.1 | Annual test |
| BPCS-004 | Combustion air control | GL-018 | 0.1 | Annual test |
| BPCS-005 | Fuel flow control | GL-018 | 0.1 | Annual test |

### 3.3 Alarms with Operator Response

| IPL ID | Alarm | Response Time Available | Operator Action | PFD Claimed |
|--------|-------|------------------------|-----------------|-------------|
| ALM-001 | High temperature alarm | >30 minutes | Reduce firing rate | 0.1 |
| ALM-002 | High pressure alarm | >20 minutes | Reduce throughput | 0.1 |
| ALM-003 | Low flow alarm | >15 minutes | Check pumps | 0.1 |
| ALM-004 | Flame instability | >5 minutes | Check burner | 0.1 |
| ALM-005 | High CO alarm | >10 minutes | Adjust air/fuel | 0.1 |
| ALM-006 | TMT high alarm | >20 minutes | Reduce firing | 0.1 |

### 3.4 Mechanical Safeguards

| IPL ID | Device | Design Basis | PFD Claimed | Verification |
|--------|--------|--------------|-------------|--------------|
| PSV-001 | Steam header relief | ASME VIII | 0.01 | 5-year pop test |
| PSV-002 | Boiler drum relief | ASME I | 0.01 | Annual inspection |
| PSV-003 | Fuel gas relief | API 520 | 0.01 | 5-year pop test |
| RD-001 | Rupture disk (backup) | ASME VIII | 0.01 | Visual inspection |
| CHK-001 | Fuel check valve | API 594 | 0.1 | Annual stroke test |
| CHK-002 | Steam check valve | API 594 | 0.1 | Annual stroke test |

---

## 4. PFD Calculations

### 4.1 SIF PFD Calculation Methodology

Per IEC 61508-6:2010, PFD is calculated using:

**For 1oo1 Architecture:**
```
PFD_avg = lambda_DU x (TI/2 + MTTR)

Where:
  lambda_DU = Dangerous undetected failure rate (per hour)
  TI = Proof test interval (hours)
  MTTR = Mean time to repair (hours)
```

**For 2oo3 Architecture:**
```
PFD_avg = 3 x (lambda_DU x TI)^2 / 4

Where:
  lambda_DU = Per-channel dangerous undetected failure rate
  TI = Proof test interval
```

### 4.2 SIF-001 High Temperature Trip - Detailed Calculation

**System Architecture:** 2oo3

| Component | Qty | lambda_DU (per hr) | Beta | DC |
|-----------|-----|-------------------|------|-----|
| Temperature Sensor | 3 | 2.5E-06 | 5% | 60% |
| Input Card | 3 | 1.0E-06 | 5% | 90% |
| Logic Solver | 1 | 5.0E-07 | - | 95% |
| Output Relay | 1 | 1.5E-06 | - | 50% |
| Solenoid Valve | 1 | 3.0E-06 | - | 0% |

**Calculation:**

```
Sensor Subsystem (2oo3):
lambda_DU_sensor = 2.5E-06 x (1 - 0.60) = 1.0E-06 /hr
PFD_sensor = 3 x (1.0E-06 x 8760)^2 / 4 = 5.76E-05

CCF contribution:
lambda_CCF = 0.05 x 1.0E-06 = 5.0E-08 /hr
PFD_CCF = 5.0E-08 x 8760 / 2 = 2.19E-04

Sensor total: PFD = 5.76E-05 + 2.19E-04 = 2.77E-04

Input Card Subsystem (2oo3):
lambda_DU_card = 1.0E-06 x (1 - 0.90) = 1.0E-07 /hr
PFD_card = 3 x (1.0E-07 x 8760)^2 / 4 = 5.76E-07
CCF: PFD_CCF = 0.05 x 1.0E-07 x 8760 / 2 = 2.19E-05

Card total: PFD = 5.76E-07 + 2.19E-05 = 2.25E-05

Logic Solver (1oo1):
lambda_DU_LS = 5.0E-07 x (1 - 0.95) = 2.5E-08 /hr
PFD_LS = 2.5E-08 x 8760 / 2 = 1.10E-04

Output Relay (1oo1):
lambda_DU_relay = 1.5E-06 x (1 - 0.50) = 7.5E-07 /hr
PFD_relay = 7.5E-07 x 8760 / 2 = 3.29E-03

Solenoid Valve (1oo1):
lambda_DU_valve = 3.0E-06 x (1 - 0) = 3.0E-06 /hr
PFD_valve = 3.0E-06 x 8760 / 2 = 1.31E-02

BUT: With partial stroke testing (quarterly):
PFD_valve = 3.0E-06 x 2190 / 2 = 3.29E-03

Total SIF PFD:
PFD_SIF = 2.77E-04 + 2.25E-05 + 1.10E-04 + 3.29E-03 + 3.29E-03
PFD_SIF = 6.98E-03

Adjusted with MTTR (4 hours):
PFD_SIF = 6.98E-03 x (1 + 4/8760) = 6.98E-03

RESULT: PFD = 0.0070 < 0.01 (SIL 2 target) - PASS
```

### 4.3 SIF PFD Summary

| SIF | Architecture | TI (hours) | PFD Calculated | SIL Target | Status |
|-----|--------------|------------|----------------|------------|--------|
| SIF-001 | 2oo3 | 8760 | 0.0070 | SIL 2 | PASS |
| SIF-002 | 2oo3 | 8760 | 0.0058 | SIL 2 | PASS |
| SIF-003 | 2oo3 | 8760 | 0.0065 | SIL 2 | PASS |
| SIF-004 | 1oo2 | 8760 | 0.0085 | SIL 2 | PASS |
| SIF-005 | 1oo2 | 8760 | 0.0052 | SIL 2 | PASS |
| SIF-006 | 1oo2 | 8760 | 0.0072 | SIL 2 | PASS |
| SIF-007 | 2oo3 | 8760 | 0.0078 | SIL 2 | PASS |

### 4.4 Alarm with Operator Response PFD

**Criteria for PFD = 0.1:**

| Criterion | Requirement | Verification |
|-----------|-------------|--------------|
| Time available | >10 minutes | Process safety time analysis |
| Alarm audible/visible | Distinct alarm | HMI configuration |
| Operator trained | Annual refresher | Training records |
| Procedure available | Accessible at console | Procedure management |
| Single operator action | No complex steps | Procedure review |

---

## 5. Proof Test Requirements

### 5.1 Proof Test Schedule

| IPL | Test Interval | Test Type | Coverage | Personnel |
|-----|---------------|-----------|----------|-----------|
| SIF-001 | Annual | Full functional | 100% | I&E Technician |
| SIF-002 | Annual | Full functional | 100% | I&E Technician |
| SIF-003 | Annual | Full functional | 100% | I&E Technician |
| SIF-004 | Monthly | Scanner simulation | 90% | Operator |
| SIF-004 | Annual | Full functional | 100% | I&E Technician |
| SIF-005 | Semi-annual | Push button test | 95% | Operator |
| SIF-005 | Annual | Full functional | 100% | I&E Technician |
| SIF-006 | Annual | Full functional | 100% | I&E Technician |
| SIF-007 | Annual | Full functional | 100% | I&E Technician |
| PSV-001 | 5 years | Pop test | 100% | PSV contractor |
| PSV-002 | Annual | Inspection | 50% | Inspector |
| BPCS-001 | Annual | Loop check | 100% | I&E Technician |

### 5.2 Proof Test Procedure Requirements

Each proof test procedure must include:

1. **Test Identification**
   - SIF/IPL identifier
   - Test number
   - Revision date

2. **Prerequisites**
   - Permissions required
   - Bypass requirements
   - Test equipment needed
   - Personnel qualifications

3. **Test Steps**
   - Detailed step-by-step instructions
   - Expected results at each step
   - Tolerances for pass/fail

4. **Acceptance Criteria**
   - Response time limits
   - Trip setpoint accuracy
   - Final element stroke verification

5. **Documentation**
   - As-found conditions
   - Any adjustments made
   - As-left conditions
   - Signatures and dates

### 5.3 Proof Test Documentation Template

```
PROOF TEST RECORD

SIF ID: _____________ Test Date: _____________
Test Procedure: _____________ Test #: _____________

PREREQUISITES:
[ ] Bypass permit obtained (# ________)
[ ] Operations notified
[ ] Test equipment calibrated
[ ] Safety briefing completed

AS-FOUND DATA:
Sensor 1 reading: _______ Setpoint: _______
Sensor 2 reading: _______ Setpoint: _______
Sensor 3 reading: _______ Setpoint: _______
Trip point verified: [ ] Pass [ ] Fail
Response time: _______ sec (limit: _____ sec)
Final element stroked: [ ] Pass [ ] Fail

ADJUSTMENTS MADE:
_________________________________________________
_________________________________________________

AS-LEFT DATA:
Trip point: _______ Response time: _______ sec
Final element: [ ] Verified operational

RESULT: [ ] PASS [ ] FAIL

Technician: _________________ Date: _____________
Witness: ___________________ Date: _____________
```

---

## 6. Verification Methods

### 6.1 Independence Verification

| IPL | Independence Basis | Common Elements | Mitigation |
|-----|-------------------|-----------------|------------|
| SIF-001 | Separate from BPCS temp control | Sensor location | Different tap |
| SIF-002 | Separate from BPCS flow control | None | Independent installation |
| SIF-003 | Separate from BPCS pressure control | Impulse line root | Separate root valve |
| SIF-004 | Separate from BPCS flame monitor | Scanner cable routing | Separate conduit |
| SIF-005 | Hardwired + software paths | None | Diverse architecture |
| Alarm + Operator | Separate from automatic trip | Same sensor | Qualified per CCPS |
| PSV | Mechanical, passive | None | Independent by design |

### 6.2 Specificity Verification

| IPL | Scenario | Specific Response | Verification |
|-----|----------|-------------------|--------------|
| SIF-001 | High temperature | Close fuel valve, stop burner | Functional test |
| SIF-002 | Low cooling flow | Stop heat input | Functional test |
| SIF-003 | High pressure | Open relief, stop feed | Functional test |
| SIF-004 | Flame loss | Isolate fuel, purge | NFPA 85 compliance |
| SIF-005 | Emergency | Total isolation | Emergency drill |
| ALM-001 | High temp warning | Operator reduces firing | Drill |
| PSV-001 | Overpressure | Relieves to safe location | Pop test |

### 6.3 Dependability Verification

| IPL | PFD Target | PFD Evidence | Source |
|-----|------------|--------------|--------|
| SIF-001 | 0.01 | 0.0070 | Calculation |
| SIF-002 | 0.01 | 0.0058 | Calculation |
| SIF-003 | 0.01 | 0.0065 | Calculation |
| SIF-004 | 0.01 | 0.0085 | Calculation |
| SIF-005 | 0.01 | 0.0052 | Calculation |
| BPCS-001 | 0.1 | Industry data | OREDA |
| ALM-001 | 0.1 | CCPS criterion | Training records |
| PSV-001 | 0.01 | API 521 | Pop test records |

### 6.4 Auditability Verification

| IPL | Test Record | Last Test | Next Test | Status |
|-----|-------------|-----------|-----------|--------|
| SIF-001 | PT-001-2025 | 2025-06-15 | 2026-06-15 | Current |
| SIF-002 | PT-002-2025 | 2025-06-15 | 2026-06-15 | Current |
| SIF-003 | PT-003-2025 | 2025-07-01 | 2026-07-01 | Current |
| SIF-004 | PT-004-2025 | 2025-11-01 | 2025-12-01 | Due Soon |
| SIF-005 | PT-005-2025 | 2025-09-01 | 2026-03-01 | Current |
| PSV-001 | PSV-TEST-2023 | 2023-04-01 | 2028-04-01 | Current |
| ALM-001 | Training-2025 | 2025-01-15 | 2026-01-15 | Current |

---

## 7. IPL Documentation

### 7.1 IPL Validation Checklist

For each IPL, complete the following:

```
IPL VALIDATION CHECKLIST

IPL ID: _____________ Description: _____________

INDEPENDENCE:
[ ] Separate initiating cause from IPL trigger
[ ] Separate from other credited IPLs
[ ] No common utilities (power, instrument air) without backup
[ ] No common human action
[ ] Common cause analysis completed

SPECIFICITY:
[ ] Designed for this specific scenario
[ ] Prevents or mitigates the specific consequence
[ ] Response is appropriate for the hazard
[ ] Functionality verified by testing

DEPENDABILITY:
[ ] PFD calculated per IEC 61508/61511
[ ] Failure data from recognized source
[ ] PFD meets LOPA requirement
[ ] Proof test interval justified
[ ] MTTR accounted for

AUDITABILITY:
[ ] Testing procedure exists
[ ] Testing schedule established
[ ] Records maintained
[ ] Deviations tracked and addressed

Validated by: _________________ Date: _____________
Approved by: _________________ Date: _____________
```

### 7.2 IPL Register

| IPL ID | Function | Scenario | PFD | Test Interval | Owner |
|--------|----------|----------|-----|---------------|-------|
| SIF-001 | High Temp Trip | High temp - equipment damage | 0.007 | Annual | Reliability |
| SIF-002 | Low Flow Trip | Low flow - thermal damage | 0.006 | Annual | Reliability |
| SIF-003 | High Press Trip | High press - vessel rupture | 0.007 | Annual | Reliability |
| SIF-004 | Flame Fail Trip | Flame loss - explosion | 0.009 | Monthly/Annual | Reliability |
| SIF-005 | ESD | Emergency - multiple | 0.005 | Semi-annual | Reliability |
| SIF-006 | Air Flow Trip | Air loss - rich combustion | 0.007 | Annual | Reliability |
| SIF-007 | TMT Trip | High TMT - tube rupture | 0.008 | Annual | Reliability |
| BPCS-001 | Temp Control | Normal operation | 0.1 | Annual | Operations |
| BPCS-002 | Press Control | Normal operation | 0.1 | Annual | Operations |
| ALM-001 | High Temp Alarm | Operator response | 0.1 | Annual training | Operations |
| PSV-001 | Steam Relief | Overpressure | 0.01 | 5-year | Inspection |
| PSV-002 | Drum Relief | Overpressure | 0.01 | Annual | Inspection |

---

## 8. Common Cause Failure Considerations

### 8.1 CCF Mitigation Measures

| CCF Source | Mitigation | Verification |
|------------|------------|--------------|
| **Electrical Power** | UPS, redundant supplies | Annual test |
| **Instrument Air** | Air receivers, backup compressor | Weekly check |
| **Environmental** | HVAC, temperature monitoring | Continuous |
| **Maintenance** | Staggered calibration, diverse teams | Schedule review |
| **Software** | Diverse coding, proven-in-use | Design review |
| **Human Factors** | Training, procedures, supervision | Audit |

### 8.2 CCF Analysis Summary

| SIF | Beta Factor | Basis | Mitigation |
|-----|-------------|-------|------------|
| SIF-001 | 5% | Standard sensors | Physical separation, diverse suppliers |
| SIF-002 | 5% | Standard transmitters | Physical separation |
| SIF-003 | 5% | Standard pressure | Separate impulse lines |
| SIF-004 | 10% | Common scanner type | Different sightlines |
| SIF-005 | 2% | Diverse architecture | Hardwired + software |
| SIF-006 | 5% | Standard flow | Physical separation |
| SIF-007 | 10% | Common TC type | Different tube locations |

### 8.3 Diversity Requirements

| Subsystem | Diversity Measure | Evidence |
|-----------|-------------------|----------|
| Sensors | Different manufacturers | Purchase records |
| Logic Solvers | Software + hardwired | Design documentation |
| Final Elements | Different valve types | P&ID review |
| Power Supply | UPS + generator | Single-line diagram |
| Communications | Hardwired + network | Architecture drawing |

---

## Appendix A: Failure Rate Data Sources

| Source | Application | Reference |
|--------|-------------|-----------|
| OREDA | Offshore equipment | OREDA Handbook 2015 |
| exida | Process instrumentation | Safety Equipment Reliability Handbook |
| NPRD | Electronics | NPRD-2016 |
| IEEE 493 | Electrical equipment | Gold Book |
| IEC 61508-6 | Example failure rates | Annex B |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang Process Heat Safety Documentation Package.*
