# Proof Test Procedures

## GreenLang Process Heat Agents - SIF Verification Testing

**Document ID:** GL-SIL-PT-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-1:2016, Clause 16.3

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [General Testing Requirements](#2-general-testing-requirements)
3. [SIF-001 Proof Test: High Temperature Shutdown](#3-sif-001-proof-test-high-temperature-shutdown)
4. [SIF-002 Proof Test: Low Flow Protection](#4-sif-002-proof-test-low-flow-protection)
5. [SIF-003 Proof Test: Pressure Relief Monitoring](#5-sif-003-proof-test-pressure-relief-monitoring)
6. [SIF-004 Proof Test: Flame Failure Detection](#6-sif-004-proof-test-flame-failure-detection)
7. [SIF-005 Proof Test: Emergency Shutdown](#7-sif-005-proof-test-emergency-shutdown)
8. [SIF-006 Proof Test: Ventilation Fault Detection](#8-sif-006-proof-test-ventilation-fault-detection)
9. [SIF-007 Proof Test: CO/CO2 High Alert](#9-sif-007-proof-test-coco2-high-alert)
10. [SIF-008 Proof Test: Emission Threshold Violation](#10-sif-008-proof-test-emission-threshold-violation)
11. [Test Equipment Requirements](#11-test-equipment-requirements)
12. [Documentation Requirements](#12-documentation-requirements)

---

## 1. Introduction

### 1.1 Purpose

This document defines the proof test procedures for Safety Instrumented Functions (SIFs) in the GreenLang Process Heat agent system. Proof tests verify that SIFs can perform their intended safety function and detect dangerous undetected failures.

### 1.2 Scope

These procedures cover:

- All SIFs listed in the Safety Requirements Specification (GL-SIL-SRS-001)
- Sensors, logic solvers, and final elements
- End-to-end functional testing
- Partial stroke testing where applicable

### 1.3 Test Interval Summary

| SIF | Test Interval (Ti) | Basis |
|-----|-------------------|-------|
| SIF-001 | 12 months | SIL 2 PFD requirement |
| SIF-002 | 12 months | SIL 2 PFD requirement |
| SIF-003 | 12 months | SIL 2 PFD requirement |
| SIF-004 | 12 months | SIL 2 PFD requirement, NFPA 85/86 |
| SIF-005 | 12 months | SIL 2 PFD requirement |
| SIF-006 | 24 months | SIL 1 PFD requirement |
| SIF-007 | 12 months | CO sensor calibration requirement |
| SIF-008 | 24 months | SIL 1 PFD requirement |

### 1.4 Definitions

| Term | Definition |
|------|------------|
| Proof Test | Periodic test to detect dangerous undetected failures |
| Functional Test | Complete end-to-end test of SIF operation |
| Partial Stroke Test | Partial actuation of final element to verify operation |
| Test Coverage | Percentage of dangerous failures detected by test |
| As-Found | Condition when test begins (before any adjustment) |
| As-Left | Condition when test ends (after adjustment if needed) |

---

## 2. General Testing Requirements

### 2.1 Pre-Test Requirements

Before conducting any proof test:

1. **Authorization**
   - Obtain work permit for safety system testing
   - Notify operations and affected personnel
   - Verify safe conditions for testing

2. **Preparation**
   - Review SIF functional requirements
   - Gather test equipment (calibrated)
   - Prepare test documentation
   - Verify bypass/override capability

3. **Safety Measures**
   - Implement compensating measures if SIF is bypassed
   - Post warning signs at HMI and field locations
   - Ensure emergency procedures are available

### 2.2 Test Execution Principles

| Principle | Requirement |
|-----------|-------------|
| Independence | Tester independent of routine maintenance personnel |
| Documentation | Record all as-found and as-left conditions |
| Traceability | Reference calibration certificates for test equipment |
| Systematic | Follow procedure steps in order |
| Verification | Witness verification for SIL 2/3 tests |

### 2.3 Post-Test Requirements

After completing proof test:

1. **Documentation**
   - Complete test record form
   - Document any deviations or anomalies
   - Record pass/fail status

2. **Restoration**
   - Return SIF to normal operating mode
   - Remove bypasses and overrides
   - Verify SIF is fully operational

3. **Follow-up**
   - Report failures to responsible engineer
   - Initiate corrective action if needed
   - Update maintenance records

### 2.4 Pass/Fail Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| Trip setpoint | Within +/- accuracy spec | Outside accuracy spec |
| Response time | <= allocated time | > allocated time |
| Valve stroke | Achieves full travel | Incomplete travel |
| Alarm annunciation | Correct indication | Missing or incorrect |
| Logic function | Correct action | Incorrect action |

---

## 3. SIF-001 Proof Test: High Temperature Shutdown

### 3.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-001 |
| Function | High Temperature Shutdown (HTS) |
| Agent | GL-001 |
| SIL | SIL 2 |
| Test Interval | 12 months |
| Test Duration | Approximately 2 hours |
| Process Impact | Full shutdown test recommended during turnaround |

### 3.2 Test Equipment

| Equipment | Specification | Calibration |
|-----------|---------------|-------------|
| Temperature calibrator | 0-500 C range, accuracy +/- 0.1 C | Annual |
| Stopwatch | 0.01 s resolution | N/A |
| Multimeter | 4-20 mA range, accuracy +/- 0.01 mA | Annual |
| Position indicator | Visual/mechanical | N/A |

### 3.3 Pre-Test Conditions

- [ ] Process is in safe condition for test
- [ ] Work permit obtained
- [ ] Operations notified
- [ ] Test equipment available and calibrated
- [ ] Bypass procedure ready (if not doing full trip test)

### 3.4 Test Procedure

#### Step 1: Sensor A (TE-001A) Calibration Check

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Record as-found TE-001A reading at ambient | | | |
| 1.2 | Connect calibrator to TE-001A | Stable connection | | |
| 1.3 | Apply 0% of range (0 C equivalent) | 4.00 mA +/- 0.08 mA | | |
| 1.4 | Apply 50% of range (250 C equivalent) | 12.00 mA +/- 0.08 mA | | |
| 1.5 | Apply 100% of range (500 C equivalent) | 20.00 mA +/- 0.08 mA | | |
| 1.6 | Apply THH setpoint (380 C) | Verify GL-001 alarm | | |
| 1.7 | Record as-left TE-001A values | Within specification | | |

#### Step 2: Sensor B (TE-001B) Calibration Check

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Record as-found TE-001B reading at ambient | | | |
| 2.2 | Connect calibrator to TE-001B | Stable connection | | |
| 2.3 | Apply 0% of range (0 C equivalent) | 4.00 mA +/- 0.08 mA | | |
| 2.4 | Apply 50% of range (250 C equivalent) | 12.00 mA +/- 0.08 mA | | |
| 2.5 | Apply 100% of range (500 C equivalent) | 20.00 mA +/- 0.08 mA | | |
| 2.6 | Apply THH setpoint (380 C) | Verify GL-001 alarm | | |
| 2.7 | Record as-left TE-001B values | Within specification | | |

#### Step 3: 2oo2 Voting Logic Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 3.1 | Apply THH to TE-001A only | NO trip (2oo2 requires both) | | |
| 3.2 | Apply THH to TE-001B only | NO trip (2oo2 requires both) | | |
| 3.3 | Apply THH to both sensors | TRIP initiated | | |
| 3.4 | Record response time (start timer when second sensor reaches THH) | < 500 ms | | |

#### Step 4: Final Element Test - XV-001A

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 4.1 | Record as-found XV-001A position | Open (normal) | | |
| 4.2 | Initiate trip signal | Valve closes | | |
| 4.3 | Record stroke time | < 2 seconds | | |
| 4.4 | Verify limit switch feedback | Closed indication | | |
| 4.5 | Reset and verify valve opens | Valve opens, open indication | | |

#### Step 5: Final Element Test - XV-001B

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 5.1 | Record as-found XV-001B position | Open (normal) | | |
| 5.2 | Initiate trip signal | Valve closes | | |
| 5.3 | Record stroke time | < 2 seconds | | |
| 5.4 | Verify limit switch feedback | Closed indication | | |
| 5.5 | Reset and verify valve opens | Valve opens, open indication | | |

#### Step 6: End-to-End Functional Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 6.1 | Apply THH to both sensors simultaneously | | | |
| 6.2 | Verify GL-001 HTS alarm activates | HTS alarm on HMI | | |
| 6.3 | Verify XV-001A closes | Closed position | | |
| 6.4 | Verify XV-001B closes | Closed position | | |
| 6.5 | Verify cooling system activates | Cooling pumps start | | |
| 6.6 | Record total response time | < 500 ms | | |
| 6.7 | Verify event logging | Time-stamped log entry | | |
| 6.8 | Reset SIF, verify all returns to normal | Normal operation | | |

### 3.5 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| Sensor accuracy | +/- 1% of span | |
| Trip setpoint | 380 C +/- 4 C | |
| Response time | < 500 ms | |
| Valve stroke time | < 2 seconds | |
| Logic function | 2oo2 correct | |

### 3.6 Test Record

| Field | Entry |
|-------|-------|
| Test Date | |
| Tester Name | |
| Witness Name | |
| Overall Result | PASS / FAIL |
| Next Test Due | |
| Comments | |

---

## 4. SIF-002 Proof Test: Low Flow Protection

### 4.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-002 |
| Function | Low Flow Protection (LFP) |
| Agent | GL-001 |
| SIL | SIL 2 |
| Test Interval | 12 months |
| Test Duration | Approximately 1.5 hours |

### 4.2 Test Equipment

| Equipment | Specification | Calibration |
|-----------|---------------|-------------|
| Flow calibrator | 0-100% range, accuracy +/- 0.5% | Annual |
| Stopwatch | 0.01 s resolution | N/A |
| Multimeter | 4-20 mA range | Annual |

### 4.3 Test Procedure

#### Step 1: Sensor Calibration Check (FT-001A, FT-001B)

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Check FT-001A at 0%, 50%, 100% | Within +/- 2% span | | |
| 1.2 | Check FT-001B at 0%, 50%, 100% | Within +/- 2% span | | |
| 1.3 | Apply FLL setpoint to both | GL-001 alarm after 5s delay | | |

#### Step 2: Time Delay Verification

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Apply FLL to both sensors | Start stopwatch | | |
| 2.2 | Verify no trip for 4.5 seconds | No trip action | | |
| 2.3 | Verify trip occurs at ~5 seconds | Trip at 5.0 +/- 0.5 s | | |

#### Step 3: Final Element Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 3.1 | Test heat source isolation valve | Closes on trip | | |
| 3.2 | Verify stroke time | < 2 seconds | | |
| 3.3 | Verify reset function | Returns to normal | | |

### 4.4 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| Sensor accuracy | +/- 2% of span | |
| Time delay | 5.0 +/- 0.5 seconds | |
| Response time (after delay) | < 500 ms | |

---

## 5. SIF-003 Proof Test: Pressure Relief Monitoring

### 5.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-003 |
| Function | Pressure Relief Monitoring (PRM) |
| Agent | GL-001 |
| SIL | SIL 2 |
| Test Interval | 12 months |

### 5.2 Test Procedure

#### Step 1: Pressure Sensor Calibration

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Check PT-001A at 0%, 50%, 100% | Within +/- 1% span | | |
| 1.2 | Check PT-001B at 0%, 50%, 100% | Within +/- 1% span | | |
| 1.3 | Apply PHH setpoint to either sensor | GL-001 alarm (1oo2) | | |

#### Step 2: 1oo2 Voting Verification

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Apply PHH to PT-001A only | TRIP (1oo2) | | |
| 2.2 | Reset, apply PHH to PT-001B only | TRIP (1oo2) | | |

#### Step 3: Final Element Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 3.1 | Verify PV-001A opens on trip | Depressurization valve opens | | |
| 3.2 | Verify PV-001B opens on trip | Isolation valve closes | | |
| 3.3 | Record response time | < 500 ms | | |

### 5.3 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| Sensor accuracy | +/- 1% of span | |
| Response time | < 500 ms | |
| Voting logic | 1oo2 confirmed | |

---

## 6. SIF-004 Proof Test: Flame Failure Detection

### 6.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-004 |
| Function | Flame Failure Detection (FFD) |
| Agent | GL-001 |
| SIL | SIL 2 |
| Test Interval | 12 months (scanners tested monthly) |
| Special Requirements | Per NFPA 85/86 |

### 6.2 Test Equipment

| Equipment | Specification | Calibration |
|-----------|---------------|-------------|
| Flame simulator (if applicable) | UV/IR source | Annual |
| Stopwatch | 0.01 s resolution | N/A |
| Scanner test lamp | Per manufacturer | Annual |

### 6.3 Test Procedure

#### Step 1: Flame Scanner Self-Test (Monthly)

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Initiate FS-001A self-test | Self-test passes | | |
| 1.2 | Initiate FS-001B self-test | Self-test passes | | |
| 1.3 | Initiate FS-001C self-test | Self-test passes | | |
| 1.4 | Verify diagnostic outputs | No faults | | |

#### Step 2: Flame Loss Simulation (Annual)

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Block FS-001A view (with burner off) | Loss of flame signal | | |
| 2.2 | Verify single scanner does not trip | No trip (2oo3) | | |
| 2.3 | Block FS-001B view | Loss of flame signal | | |
| 2.4 | Verify 2oo3 trip occurs | TRIP within 1 second | | |
| 2.5 | Record flame failure response time | < 1 second | | |

#### Step 3: Fuel Shutoff Valve Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 3.1 | Initiate trip signal | XV-004A closes | | |
| 3.2 | Verify XV-004A closure time | < 3 seconds (NFPA) | | |
| 3.3 | Verify XV-004B closes | < 3 seconds | | |
| 3.4 | Verify block and bleed opens | Vent valve opens | | |

#### Step 4: Purge Sequence Verification

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 4.1 | Verify purge timer starts on trip | Timer running | | |
| 4.2 | Verify minimum 4 volume changes | Purge complete | | |
| 4.3 | Verify restart lockout until purge complete | Restart blocked | | |

### 6.4 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| Flame failure response | < 1 second | |
| Fuel valve closure | < 3 seconds | |
| Voting logic | 2oo3 confirmed | |
| Purge sequence | 4 volume changes | |

---

## 7. SIF-005 Proof Test: Emergency Shutdown

### 7.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-005 |
| Function | Emergency Shutdown (ESD) |
| Agent | GL-001 |
| SIL | SIL 2 |
| Test Interval | 12 months |

### 7.2 Test Procedure

#### Step 1: Manual ESD Pushbutton Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Press manual ESD pushbutton | Full shutdown initiated | | |
| 1.2 | Verify all fuel isolation valves close | All closed | | |
| 1.3 | Verify all process isolation valves close | All closed | | |
| 1.4 | Verify cooling systems activate | Pumps running | | |
| 1.5 | Verify ESD alarm annunciates | Alarm on HMI | | |
| 1.6 | Record response time | < 500 ms | | |

#### Step 2: Automatic Trip Input Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Simulate SIF-001 trip output | ESD activates | | |
| 2.2 | Reset, simulate SIF-002 trip output | ESD activates | | |
| 2.3 | Reset, simulate SIF-003 trip output | ESD activates | | |
| 2.4 | Reset, simulate SIF-004 trip output | ESD activates | | |

#### Step 3: External ESD Input Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 3.1 | Apply external ESD signal | Full shutdown | | |
| 3.2 | Verify response time | < 500 ms | | |

#### Step 4: Reset Function Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 4.1 | Attempt reset without key switch | Reset blocked | | |
| 4.2 | Insert key and reset | System resets | | |
| 4.3 | Verify all systems return to normal | Normal operation | | |

### 7.3 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| Manual ESD response | < 500 ms | |
| Automatic trip response | < 500 ms | |
| Reset security | Key switch required | |

---

## 8. SIF-006 Proof Test: Ventilation Fault Detection

### 8.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-006 |
| Function | Ventilation Fault Detection (VFD) |
| Agent | GL-005 |
| SIL | SIL 1 |
| Test Interval | 24 months |

### 8.2 Test Procedure

#### Step 1: Airflow Sensor Calibration

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Check AS-001 at 0%, 50%, 100% | Within +/- 5% span | | |
| 1.2 | Apply low flow setpoint | GL-005 alarm | | |

#### Step 2: Emergency Ventilation Activation

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Simulate ventilation fault | Emergency vent starts | | |
| 2.2 | Verify BMS notification | Alert received | | |
| 2.3 | Record response time | < 2000 ms | | |

### 8.3 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| Sensor accuracy | +/- 5% of span | |
| Response time | < 2000 ms | |

---

## 9. SIF-007 Proof Test: CO/CO2 High Alert

### 9.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-007 |
| Function | CO/CO2 High Alert |
| Agent | GL-005 |
| SIL | SIL 1 |
| Test Interval | 12 months (sensor bump test quarterly) |

### 9.2 Test Equipment

| Equipment | Specification | Calibration |
|-----------|---------------|-------------|
| CO calibration gas | 35 ppm, 100 ppm | Certificate |
| CO2 calibration gas | 5000 ppm | Certificate |
| Flow regulator | 0.5 LPM | N/A |

### 9.3 Test Procedure

#### Step 1: CO Sensor Calibration

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Apply zero gas | 0 ppm +/- 2 ppm | | |
| 1.2 | Apply 35 ppm span gas | High alarm | | |
| 1.3 | Apply 100 ppm span gas | High-High alarm | | |
| 1.4 | Record response time | < 30 seconds | | |

#### Step 2: CO2 Sensor Calibration

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Apply zero gas | 0 ppm +/- 100 ppm | | |
| 2.2 | Apply 5000 ppm span gas | High alarm | | |
| 2.3 | Record response time | < 60 seconds | | |

#### Step 3: Alert and Ventilation Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 3.1 | Apply CO at High-High level | Alarm + vent activation | | |
| 3.2 | Verify GL-005 alert | Alert generated | | |
| 3.3 | Verify event logging | Log entry created | | |

### 9.4 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| CO accuracy | +/- 5 ppm or 10% | |
| CO2 accuracy | +/- 200 ppm or 5% | |
| Alarm setpoints | CO: 35/100 ppm, CO2: 5000 ppm | |

---

## 10. SIF-008 Proof Test: Emission Threshold Violation

### 10.1 Test Information

| Item | Specification |
|------|---------------|
| SIF ID | SIF-008 |
| Function | Emission Threshold Violation Detection |
| Agent | GL-007 |
| SIL | SIL 1 |
| Test Interval | 24 months |

### 10.2 Test Procedure

#### Step 1: Data Input Validation

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 1.1 | Input test emission data below threshold | No alert | | |
| 1.2 | Input test emission data at threshold | GL-007 alert | | |
| 1.3 | Input test emission data above threshold | GL-007 high alert | | |

#### Step 2: Alert and Logging Test

| Step | Action | Expected Result | Actual | Pass/Fail |
|------|--------|-----------------|--------|-----------|
| 2.1 | Trigger threshold violation | Alert generated | | |
| 2.2 | Verify compliance notification | Notification sent | | |
| 2.3 | Verify enhanced logging activates | Additional data logged | | |
| 2.4 | Verify audit trail | Audit record created | | |

### 10.3 Acceptance Criteria

| Parameter | Requirement | Achieved |
|-----------|-------------|----------|
| Detection accuracy | 100% of violations | |
| Response time | < 5000 ms | |
| Data integrity | 99.9% | |

---

## 11. Test Equipment Requirements

### 11.1 Calibration Requirements

| Equipment | Calibration Interval | Traceability |
|-----------|---------------------|--------------|
| Temperature calibrator | 12 months | NIST |
| Pressure calibrator | 12 months | NIST |
| Multimeter | 12 months | NIST |
| Calibration gases | Per certificate | ISO 17025 |
| Stopwatch | N/A (verified) | N/A |

### 11.2 Equipment List

| Item | Quantity | Specification |
|------|----------|---------------|
| Temperature calibrator | 1 | 0-500 C, +/- 0.1 C |
| Pressure calibrator | 1 | 0-100 bar, +/- 0.05% |
| Flow calibrator | 1 | 0-100%, +/- 0.5% |
| Multimeter | 1 | 4-20 mA, +/- 0.01 mA |
| CO cal gas (35 ppm) | 1 cylinder | Certificate |
| CO cal gas (100 ppm) | 1 cylinder | Certificate |
| CO2 cal gas (5000 ppm) | 1 cylinder | Certificate |
| Flame test lamp | 1 | Per scanner type |
| Stopwatch | 1 | 0.01 s resolution |

---

## 12. Documentation Requirements

### 12.1 Test Record Form Template

```
==============================================================================
                        PROOF TEST RECORD
==============================================================================

SIF ID: ____________          SIF Name: _________________________________
Agent: _____________          SIL: _____________________________________
Test Date: __________         Next Test Due: ___________________________

Plant/Unit: _________________________________________________________
Equipment Tag Numbers: ______________________________________________

TESTER INFORMATION
Tester Name: ________________________  Signature: ____________________
Witness Name: _______________________  Signature: ____________________
(Required for SIL 2)

TEST EQUIPMENT USED
Equipment                    Serial No.        Cal Due Date
_________________________   ______________    ______________
_________________________   ______________    ______________
_________________________   ______________    ______________

PRE-TEST CONDITIONS
[ ] Work permit obtained (Permit #: ____________)
[ ] Operations notified
[ ] Test equipment calibration verified
[ ] Bypass/compensating measures in place

TEST RESULTS SUMMARY
Component          As-Found        As-Left         Pass/Fail
____________       ___________     ___________     _________
____________       ___________     ___________     _________
____________       ___________     ___________     _________

Overall Response Time: _______________ (Requirement: _______________)

OVERALL TEST RESULT: [ ] PASS  [ ] FAIL

ANOMALIES/DEVIATIONS
___________________________________________________________________
___________________________________________________________________
___________________________________________________________________

CORRECTIVE ACTIONS REQUIRED
___________________________________________________________________
___________________________________________________________________

POST-TEST VERIFICATION
[ ] SIF returned to normal operation
[ ] All bypasses removed
[ ] HMI indications normal
[ ] Test documented in maintenance system

Reviewed by: _______________________ Date: ______________
(Safety Systems Engineer)

==============================================================================
```

### 12.2 Record Retention

| Record Type | Retention Period | Location |
|-------------|------------------|----------|
| Proof test records | Life of system + 5 years | Safety files |
| Calibration certificates | 5 years | Calibration files |
| Anomaly reports | Life of system | Safety files |
| Corrective action records | Life of system | Maintenance files |

### 12.3 Trend Analysis

Proof test results shall be trended to identify:

- Degrading components requiring replacement
- Systematic failures requiring design modification
- Effectiveness of test procedures
- Compliance with PFD assumptions

---

## Appendix A: Partial Stroke Testing

### A.1 PST Application

Partial Stroke Testing (PST) may be used between full proof tests to:

- Reduce proof test interval credit
- Detect stuck valves early
- Maintain availability during operation

### A.2 PST Parameters

| Parameter | Setting |
|-----------|---------|
| Stroke percentage | 10-30% of travel |
| Test frequency | Monthly to quarterly |
| PFD credit | Up to 80% detection of stroke failures |

### A.3 PST Procedure (Valves with Positioner)

1. Initiate PST from HMI or locally
2. Valve moves to PST position (e.g., 30% closed)
3. Verify position feedback matches command
4. Return valve to normal position
5. Record result

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
