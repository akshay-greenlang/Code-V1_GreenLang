# Phase 5: Safety and Compliance Gap Analysis

## GreenLang Process Heat Agents - Regulatory Requirements Analysis

**Document ID:** GL-REG-GAP-PHASE5-001
**Version:** 1.0
**Created:** 2025-12-05
**Classification:** Regulatory Intelligence Analysis
**Current Status:** Phase 5 at 48% Complete (24/50 tasks done, 26 remaining)

---

## Executive Summary

This document provides a comprehensive regulatory gap analysis for Phase 5 (Safety and Compliance) of the GreenLang Process Heat Agents engineering roadmap. The analysis maps remaining tasks to specific regulatory requirements, establishes priority rankings based on compliance deadlines and risk severity, and provides implementation templates and sequences.

### Key Findings

| Category | Status | Critical Gaps | Priority |
|----------|--------|---------------|----------|
| SIL Assessment (5.1) | 80% | GL-007 LOPA/SRS missing | HIGH |
| Fail-Safe Design (5.2) | 80% | CCF mitigation, diversity | MEDIUM |
| ESD Integration (5.3) | 30% | 7 tasks incomplete | CRITICAL |
| Regulatory Compliance (5.4) | 30% | EPA/NFPA/OSHA gaps | CRITICAL |
| HAZOP/FMEA (5.5) | 0% | All 10 tasks pending | CRITICAL |

---

## Table of Contents

1. [Regulatory Framework Overview](#1-regulatory-framework-overview)
2. [Task-by-Task Regulatory Mapping](#2-task-by-task-regulatory-mapping)
3. [Priority Ranking Matrix](#3-priority-ranking-matrix)
4. [Implementation Sequence](#4-implementation-sequence)
5. [Documentation Templates](#5-documentation-templates)
6. [Specific Standard Requirements](#6-specific-standard-requirements)
7. [Compliance Deadlines and Penalties](#7-compliance-deadlines-and-penalties)
8. [Recommended Action Plan](#8-recommended-action-plan)

---

## 1. Regulatory Framework Overview

### 1.1 Applicable Standards and Regulations

| Standard/Regulation | Jurisdiction | Applicability | Enforcement |
|---------------------|--------------|---------------|-------------|
| IEC 61511 | International | Safety Instrumented Systems | Mandatory for SIS |
| IEC 61508 | International | Functional Safety | Foundation standard |
| NFPA 85 | US/International | Boiler/Combustion Safety | Insurance/Local AHJ |
| NFPA 86 | US/International | Furnace Safety | Insurance/Local AHJ |
| EPA 40 CFR Part 60 | US | NSPS - New Source Performance | EPA Enforcement |
| EPA 40 CFR Part 98 | US | GHG Mandatory Reporting | EPA Enforcement |
| OSHA 1910.119 | US | Process Safety Management | OSHA Enforcement |
| EU IED 2010/75/EU | EU | Industrial Emissions | Member State |
| ISA 18.2 / IEC 62682 | International | Alarm Management | Industry Practice |

### 1.2 Current Implementation Status

```
PHASE 5: SAFETY & COMPLIANCE
===============================

5.1 SIL Assessment & SRS     [========--] 80%  (8/10 tasks)
5.2 Fail-Safe Design         [========--] 80%  (8/10 tasks)
5.3 ESD Integration          [===-------] 30%  (3/10 tasks)
5.4 Regulatory Compliance    [===-------] 30%  (3/10 tasks)
5.5 HAZOP & FMEA             [----------]  0%  (0/10 tasks)

OVERALL PHASE 5              [=====-----] 48%  (24/50 tasks)
```

### 1.3 Gap Severity Classification

| Severity | Definition | Compliance Impact |
|----------|------------|-------------------|
| CRITICAL | Missing mandatory requirement | Operation not permitted |
| HIGH | Missing recommended practice | Increased liability risk |
| MEDIUM | Incomplete implementation | Reduced certification confidence |
| LOW | Enhancement opportunity | Best practice gap |

---

## 2. Task-by-Task Regulatory Mapping

### 2.1 Section 5.1: SIL Assessment and SRS (2 Tasks Remaining)

#### TASK-163: Conduct LOPA Analysis for GL-007

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-3:2016 Clause 8 |
| **Standard Reference** | CCPS LOPA Guidelines, ISA TR84.00.04 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | HIGH |
| **Dependencies** | Requires GL-007 HAZOP study (TASK-203) |
| **Estimated Effort** | 24-40 hours (engineering study) |

**Specific Requirements:**
- Identify all hazard scenarios from GL-007 HAZOP
- Assign initiating event frequencies per OREDA/industry data
- Identify Independent Protection Layers (IPLs)
- Calculate required SIF PFD
- Determine SIL for TMT monitoring functions

**Regulatory Citation:**
> IEC 61511-3:2016, Clause 8.2: "The required SIL for each safety instrumented function shall be determined using one of the following methods: risk graph, LOPA, quantitative risk assessment, or other acceptable methods."

**Implementation Notes:**
- Reference existing LOPA document: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\iec_61511\02_LOPA_ANALYSIS.md`
- Follow same worksheet format for GL-007 specific scenarios
- Key scenarios: TMT overtemperature, tube rupture, fouling-induced hot spots

---

#### TASK-166: Create SRS Document for GL-007

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 10 |
| **Standard Reference** | ISA 84.00.01 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | HIGH |
| **Dependencies** | TASK-163 (LOPA analysis) |
| **Estimated Effort** | 16-24 hours |

**Specific Requirements per IEC 61511-1 Clause 10.3:**
- Description of each SIF (TMT alarm, TMT trip)
- Safe state definition
- Assumed sources of demand
- SIL requirement (from LOPA)
- Process safety time
- Response time requirements
- Spurious trip requirements
- Functional test requirements

**Template Reference:**
- Use format from: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\iec_61511\01_SAFETY_REQUIREMENTS_SPECIFICATION.md`

---

### 2.2 Section 5.2: Fail-Safe Design (2 Tasks Remaining)

#### TASK-179: Create Common Cause Failure Mitigation

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 11.4 |
| **Standard Reference** | IEC 61508-6 Annex D (Beta Factor) |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | MEDIUM |
| **Dependencies** | FMEA completion (TASK-204) |
| **Estimated Effort** | 16-24 hours |

**Specific Requirements:**
Per IEC 61511-1 Clause 11.4.4, CCF mitigation shall address:
- Physical separation of redundant channels
- Electrical isolation
- Different manufacturers for diverse equipment
- Environmental protection
- Calibration procedures from different sources

**Beta Factor Requirements:**
```
Required Assessment per IEC 61508-6 Annex D:
- Separation/segregation score
- Diversity score
- Complexity score
- Assessment/analysis score
- Procedures/human interface score
- Competence/training score
- Environmental control score
- Environmental testing score

Target: Beta <= 5% for SIL 2 applications
```

**Implementation:**
- Code exists: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\sil\hardware_fault_tolerance.py`
- Documentation exists: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\04_FMEA_SAFETY_FUNCTIONS.md` (Section 11)
- Gap: Need programmatic CCF assessment and mitigation tracking

---

#### TASK-180: Implement Diversity Requirements

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 11.4.5 |
| **Standard Reference** | ISA TR84.00.04 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | MEDIUM |
| **Dependencies** | TASK-179 |
| **Estimated Effort** | 16-24 hours |

**Specific Requirements:**
Diversity shall be considered for:
- Sensor technology (e.g., RTD vs thermocouple)
- Logic solver technology
- Final element types
- Software/firmware versions
- Installation methods

**IEC 61511-1 Clause 11.4.5:**
> "Where redundancy is used to achieve the required SIL, diversity shall be considered to reduce the impact of common cause failures."

---

### 2.3 Section 5.3: Emergency Shutdown Integration (7 Tasks Remaining)

#### TASK-182: Implement Hardwired Interlock Integration

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 11.2 |
| **Standard Reference** | NFPA 85-2019 Chapter 8 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | CRITICAL |
| **Dependencies** | Hardware design, OPC-UA integration |
| **Estimated Effort** | 40-60 hours |

**Specific Requirements:**
Per NFPA 85-2019 Section 8.5.1:
- Hardwired trips shall be independent of software-based systems
- Manual emergency stops shall be hardwired
- Fuel safety shutoff valves shall have hardwired trip capability
- Hardwired and software trips shall be in 1oo2 voting configuration

**Interface Requirements:**
```
GL-001 Agent Interface:
- Read hardwired trip status via OPC-UA
- Coordinate software ESD with hardwired system
- Priority: Hardwired > Software
- Implement status monitoring and alarming
```

**Code Reference:**
- Existing ESD interface: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\esd\esd_interface.py`
- Priority manager: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\esd\priority_manager.py`

---

#### TASK-185: Implement <1s Response Time Validation

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 12.4 |
| **Standard Reference** | ISA TR84.00.04 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | HIGH |
| **Dependencies** | ESD interface complete |
| **Estimated Effort** | 16-24 hours |

**Specific Requirements:**
Per IEC 61511-1 Clause 12.4:
- SIF response time shall be verified during FAT/SAT
- Response time shall be less than process safety time
- All elements (sensor, logic, actuator) shall be included

**NFPA 85 Requirements:**
- Flame failure response: <4 seconds
- Fuel valve closure: <1 second typical
- Main burner safety shutoff: Per Table 8.5.3.4

**Implementation:**
```python
# Response time validation requirements:
class ResponseTimeValidator:
    REQUIREMENTS = {
        "high_temperature_trip": 0.5,    # seconds
        "flame_failure_response": 4.0,   # per NFPA 85
        "fuel_valve_closure": 1.0,       # seconds
        "pressure_relief_trip": 0.5,     # seconds
        "esd_total_response": 1.0,       # seconds
    }
```

**Code Reference:**
- Response validator: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\esd\response_validator.py`

---

#### TASK-186: Build ESD Test Procedures

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 16.2 |
| **Standard Reference** | ISA TR84.00.03 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | HIGH |
| **Dependencies** | ESD system design complete |
| **Estimated Effort** | 24-40 hours |

**Specific Requirements per IEC 61511-1 Clause 16.2:**
- Proof test procedures for each SIF
- Partial stroke testing procedures (where applicable)
- Full trip testing procedures
- Test frequency based on SIL and PFD requirements
- Documentation and record-keeping requirements

**Test Categories:**
| Test Type | Frequency | Coverage |
|-----------|-----------|----------|
| Full proof test | Annual | 100% of function |
| Partial stroke test | Monthly | 60-70% of valve |
| Logic solver test | Quarterly | Logic execution |
| Sensor calibration | Semi-annual | Input accuracy |

**Template Required:**
See Section 5.3 below for ESD Test Procedure Template.

---

#### TASK-187: Create Bypass Management System

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 11.7 |
| **Standard Reference** | ISA TR84.00.09 |
| **Current Status** | PARTIAL - Code exists, needs integration |
| **Gap Severity** | HIGH |
| **Dependencies** | None |
| **Estimated Effort** | 24 hours |

**Specific Requirements per IEC 61511-1 Clause 11.7:**
- Bypass shall be time-limited
- Bypass shall generate alarm
- Bypass shall be logged
- Compensating measures shall be documented
- Maximum bypass duration: 24 hours (typical)

**Code Reference:**
- Bypass manager: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\esd\bypass_manager.py` (implemented)
- Gap: Integration with GL-001 orchestrator and SIS interface

---

#### TASK-188: Implement Bypass Logging

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 11.7.4 |
| **Standard Reference** | ISA TR84.00.09 |
| **Current Status** | PARTIAL - Audit trail in bypass_manager.py |
| **Gap Severity** | MEDIUM |
| **Dependencies** | TASK-187 |
| **Estimated Effort** | 8-16 hours |

**Specific Requirements:**
Bypass log shall include:
- Date/time of bypass activation
- SIF/equipment bypassed
- Person requesting and approving bypass
- Reason for bypass
- Compensating measures implemented
- Duration and expiration
- Date/time of bypass removal

**Code Status:**
- Implemented in `BypassRecord.audit_trail` in bypass_manager.py
- Gap: Persistent storage, reporting, regulatory export

---

#### TASK-189: Build ESD Simulation Mode

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 14.2 (FAT) |
| **Standard Reference** | ISA 84.00.01 |
| **Current Status** | PARTIAL - esd_simulator.py exists |
| **Gap Severity** | MEDIUM |
| **Dependencies** | ESD interface complete |
| **Estimated Effort** | 16-24 hours |

**Specific Requirements:**
- Simulate all trip scenarios without process impact
- Verify logic execution
- Test response times
- Validate alarm generation
- Support FAT/SAT procedures

**Code Reference:**
- ESD simulator: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\esd\esd_simulator.py` (exists)
- Gap: Complete scenario library, integration testing

---

#### TASK-190: Create ESD Audit Reports

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1:2016 Clause 5.2.6 |
| **Standard Reference** | OSHA 1910.119(o) |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | MEDIUM |
| **Dependencies** | TASK-187, TASK-188 |
| **Estimated Effort** | 16 hours |

**Specific Requirements:**
- Trip event history
- Bypass history with justifications
- Proof test results
- Demand rate tracking
- PFD verification
- Compliance status summary

---

### 2.4 Section 5.4: Regulatory Compliance (7 Tasks Remaining)

#### TASK-191: Implement EPA Part 60 NSPS Compliance

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | 40 CFR Part 60 |
| **Enforcement Agency** | US EPA |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | CRITICAL (for US facilities) |
| **Penalties** | Up to $127,500/day per violation |
| **Estimated Effort** | 40-60 hours |

**Specific Requirements:**
40 CFR Part 60 establishes New Source Performance Standards for:
- Subpart D: Fossil fuel-fired steam generators
- Subpart Da: Electric utility steam generating units
- Subpart Db: Industrial-commercial-institutional steam generating units
- Subpart Dc: Small ICI steam generating units

**Emission Limits (Subpart Db - 100+ MMBtu/hr):**
| Pollutant | Fuel | Limit | Unit |
|-----------|------|-------|------|
| PM | Gas | 0.03 | lb/MMBtu |
| PM | Oil | 0.03 | lb/MMBtu |
| SO2 | Gas | Exempt | - |
| SO2 | Oil | 0.50 | lb/MMBtu |
| NOx | Gas | 0.20 | lb/MMBtu |
| NOx | Oil | 0.30 | lb/MMBtu |

**Code Reference:**
- EPA reporter: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\epa_reporter.py` (partial)
- Gap: Part 60 specific subpart implementation

---

#### TASK-193: Create EPA Part 98 GHG Reporting

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | 40 CFR Part 98 |
| **Enforcement Agency** | US EPA |
| **Current Status** | PARTIAL (GL-010 has some support) |
| **Gap Severity** | CRITICAL |
| **Deadlines** | Annual report due March 31 |
| **Penalties** | Up to $127,500/day |
| **Estimated Effort** | 40 hours |

**Applicability Threshold:**
- Facilities emitting >= 25,000 metric tons CO2e/year
- Subpart C: General stationary fuel combustion sources

**Calculation Methods per Subpart C:**
| Tier | Description | Applicability |
|------|-------------|---------------|
| 1 | Fuel-based with default emission factors | < 250 MMBtu/hr |
| 2 | Fuel-based with supplier HHV | Most units |
| 3 | Fuel-based with measured HHV/carbon | High accuracy |
| 4 | CEMS-based | Required if CEMS installed |

**Implementation Requirements:**
```python
# Part 98 Subpart C calculations
class Part98Calculator:
    # Tier 2 CO2 calculation
    def calculate_co2_tier2(self, fuel_qty, fuel_hhv, emission_factor):
        """
        CO2 = Fuel x HHV x EF x 0.001 (metric tons)
        """
        return fuel_qty * fuel_hhv * emission_factor * 0.001

    # Required data elements
    REQUIRED_DATA = [
        "fuel_type",
        "fuel_quantity",
        "fuel_hhv",
        "emission_factor_co2",
        "emission_factor_ch4",
        "emission_factor_n2o",
    ]
```

---

#### TASK-194: Implement NFPA 85 Combustion Safeguards

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | NFPA 85-2019 |
| **Enforcement** | Insurance, Local AHJ |
| **Current Status** | PARTIAL - nfpa_85_checker.py exists |
| **Gap Severity** | CRITICAL |
| **Estimated Effort** | 40 hours |

**Specific Requirements per NFPA 85:**

**Chapter 8 - Burner Management Systems:**
| Requirement | NFPA Reference | Implementation Status |
|-------------|----------------|----------------------|
| Prepurge timing | 8.5.1.1 | Partial |
| Pilot trial timing | 8.5.2.1 | Partial |
| Main flame trial | 8.5.3.1 | Partial |
| Flame failure response | 8.5.3.4 | NOT IMPLEMENTED |
| Postpurge | 8.5.4 | NOT IMPLEMENTED |
| Safety interlocks | 8.6 | Partial |

**Timing Requirements (per NFPA 85 Table 8.5.3.4):**
| Function | Single Burner | Multiple Burner |
|----------|---------------|-----------------|
| Pilot trial | 10 sec max | 10 sec max |
| Main flame trial | 10 sec max | 10 sec max |
| Flame failure response | 4 sec max | 4 sec max |

**Code Reference:**
- NFPA 85 checker: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\nfpa_85_checker.py`
- Gap: Flame failure response validation, postpurge implementation

---

#### TASK-195: Build NFPA 86 Furnace Compliance

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | NFPA 86-2019 |
| **Enforcement** | Insurance, Local AHJ |
| **Current Status** | PARTIAL - nfpa_86_checker.py exists |
| **Gap Severity** | HIGH |
| **Estimated Effort** | 24 hours |

**Specific Requirements by Furnace Class:**

**Class A (Ovens with flammable volatiles):**
- LEL monitoring required
- Ventilation rate: minimum 10 CFM/sq ft
- Temperature limit monitoring
- Safety relief requirements

**Class C (Atmosphere furnaces):**
- Atmosphere monitoring
- Purge capability
- Burn-off system
- Pressure relief

**Code Reference:**
- NFPA 86 checker: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\nfpa_86_checker.py`

---

#### TASK-196: Create OSHA 1910.119 PSM Support

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | 29 CFR 1910.119 |
| **Enforcement Agency** | OSHA |
| **Current Status** | PARTIAL - osha_psm.py exists |
| **Gap Severity** | CRITICAL (for covered processes) |
| **Penalties** | Up to $161,323/serious violation |
| **Estimated Effort** | 40 hours |

**14 PSM Elements:**
| Element | Status | GL Agent Support |
|---------|--------|------------------|
| Employee Participation | Framework | Audit module |
| Process Safety Information | Partial | Data management |
| Process Hazard Analysis | Framework | HAZOP analyzer |
| Operating Procedures | NOT IMPL | Documentation |
| Training | NOT IMPL | Training tracking |
| Contractors | NOT IMPL | Contractor management |
| Pre-startup Safety Review | NOT IMPL | Checklist module |
| Mechanical Integrity | Partial | GL-013 integration |
| Hot Work | NOT IMPL | Permit system |
| Management of Change | Framework | MOC workflow |
| Incident Investigation | NOT IMPL | Incident tracking |
| Emergency Planning | NOT IMPL | Emergency procedures |
| Compliance Audits | Partial | Audit framework |
| Trade Secrets | NOT IMPL | Access control |

**Code Reference:**
- OSHA PSM: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\osha_psm.py`

---

#### TASK-197: Implement EU IED Compliance

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | Directive 2010/75/EU |
| **Enforcement** | EU Member States |
| **Current Status** | PARTIAL - eu_ied.py exists |
| **Gap Severity** | CRITICAL (for EU operations) |
| **Estimated Effort** | 40 hours |

**Key Requirements:**
- Best Available Techniques (BAT) compliance
- BAT-Associated Emission Levels (BAT-AELs)
- Integrated permit requirements
- Monitoring and reporting

**BAT-AELs for Large Combustion Plants (>50 MWth):**
| Pollutant | Natural Gas | Fuel Oil | Unit |
|-----------|-------------|----------|------|
| NOx | 50-85 | 50-150 | mg/Nm3 |
| SO2 | N/A | 35-200 | mg/Nm3 |
| Dust | 2-5 | 2-20 | mg/Nm3 |
| CO | 10-30 | 10-30 | mg/Nm3 |

**Code Reference:**
- EU IED: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\eu_ied.py`

---

#### TASK-200: Implement ISA 18.2 Alarm Management

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | ISA 18.2 / IEC 62682 |
| **Enforcement** | Industry best practice |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | HIGH |
| **Estimated Effort** | 32 hours |

**Specific Requirements:**

**Alarm System Performance Targets (per ISA 18.2):**
| Metric | Target | Maximum |
|--------|--------|---------|
| Standing alarms | 0 | 10 |
| Alarms/10 min (normal) | <1 | 2 |
| Alarms/10 min (upset) | <10 | 15 |
| Chattering alarms | 0 | None |
| Stale alarms (>24 hr) | 0 | 5 |

**Alarm Lifecycle Management:**
1. Alarm philosophy
2. Alarm identification
3. Alarm rationalization
4. Detailed design
5. Implementation
6. Operation
7. Maintenance
8. Monitoring and assessment
9. Management of change
10. Audit

**Code Reference:**
- ISA 18.2: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\isa_18_2.py`

---

### 2.5 Section 5.5: HAZOP and FMEA (10 Tasks - All Remaining)

#### TASK-201: Conduct HAZOP for GL-001 Orchestrator

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61882:2016 |
| **Standard Reference** | OSHA 1910.119(e) |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | CRITICAL |
| **Dependencies** | Process design documentation |
| **Estimated Effort** | 40-60 hours (team study) |

**HAZOP Study Requirements per IEC 61882:**

**Team Composition:**
| Role | Responsibility |
|------|----------------|
| HAZOP Leader | Facilitation |
| Process Engineer | Design intent |
| Safety Engineer | Safeguard analysis |
| Operations Rep | Operational knowledge |
| Instrumentation | Control systems |

**Nodes for GL-001:**
1. Temperature Control Loop
2. Load Allocation System
3. Cascade Control Interface
4. SIS Integration
5. Agent Communication

**Guide Words:**
- NO, MORE, LESS, REVERSE, AS WELL AS, PART OF, OTHER THAN

**Code Reference:**
- HAZOP analyzer: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\risk\hazop_analyzer.py`
- Existing study: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\01_GL001_HAZOP_STUDY.md` (partial)

---

#### TASK-202: Conduct HAZOP for GL-002 Boiler Optimization

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61882:2016, NFPA 85 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | CRITICAL |
| **Estimated Effort** | 40-60 hours |

**Nodes for GL-002/GL-018:**
1. Combustion Air Control
2. Fuel Control System
3. Flame Monitoring
4. Flue Gas Analysis
5. Efficiency Optimization
6. Emissions Control

**Reference:**
- Partial study: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\02_GL018_HAZOP_STUDY.md`

---

#### TASK-203: Conduct HAZOP for GL-007 Furnace Monitoring

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61882:2016, API 560 |
| **Current Status** | PARTIAL - Document exists |
| **Gap Severity** | HIGH |
| **Estimated Effort** | 24-40 hours |

**Nodes for GL-007:**
1. TMT Monitoring
2. Efficiency Calculations
3. Fouling Detection
4. Alarm Generation

**Reference:**
- Existing study: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\03_GL007_HAZOP_STUDY.md`
- Gap: Formal sign-off, action tracking

---

#### TASK-204: Create FMEA for All Safety Functions

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 60812:2018 |
| **Standard Reference** | IEC 61508-2 Annex C |
| **Current Status** | PARTIAL - Document exists |
| **Gap Severity** | CRITICAL |
| **Estimated Effort** | 40-60 hours |

**Safety Functions Requiring FMEA:**
1. SIF-001: High Temperature Trip
2. SIF-002: Low Flow Protection
3. SIF-003: High Pressure Interlock
4. SIF-004: Flame Failure Detection
5. SIF-005: Emergency Shutdown
6. TMT High Alarm
7. Loss of Air Flow Response

**Reference:**
- Existing FMEA: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\04_FMEA_SAFETY_FUNCTIONS.md`
- FMEA analyzer: `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\risk\fmea_analyzer.py`
- Gap: Formal verification, action tracking, RPN reduction follow-up

---

#### TASK-205: Build Risk Matrix with Severity/Likelihood

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-3:2016 |
| **Current Status** | PARTIAL - Document exists |
| **Gap Severity** | MEDIUM |
| **Estimated Effort** | 8-16 hours |

**Reference:**
- Risk matrix: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\05_RISK_MATRIX.md`
- Gap: Programmatic risk calculation, integration with agents

---

#### TASK-206: Document Safeguard Verification

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | IEC 61511-1 Clause 11.5 |
| **Current Status** | PARTIAL - Document exists |
| **Gap Severity** | HIGH |
| **Estimated Effort** | 16-24 hours |

**Reference:**
- Safeguard verification: `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\06_SAFEGUARD_VERIFICATION.md`

---

#### TASK-207: Create Action Item Tracking

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | OSHA 1910.119(e)(5) |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | HIGH |
| **Estimated Effort** | 16-24 hours |

**OSHA Requirement:**
> "The employer shall establish a system to promptly address the team's findings and recommendations; assure that the recommendations are resolved in a timely manner..."

---

#### TASK-208: Implement Risk Register

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | ISO 31000:2018 |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | MEDIUM |
| **Estimated Effort** | 24 hours |

**Risk Register Contents:**
- Risk ID
- Risk description
- Cause(s)
- Consequence(s)
- Inherent risk rating
- Safeguards
- Residual risk rating
- Action owner
- Target date
- Status

---

#### TASK-209: Build Risk Monitoring Dashboard

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | Best Practice |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | LOW |
| **Estimated Effort** | 24-40 hours |

**Dashboard KPIs:**
- Open action items by severity
- Risk reduction trends
- HAZOP/FMEA completion status
- Overdue actions
- High-RPN items

---

#### TASK-210: Create Periodic Risk Review Process

| Attribute | Value |
|-----------|-------|
| **Regulatory Requirement** | OSHA 1910.119(e)(6) |
| **Current Status** | NOT IMPLEMENTED |
| **Gap Severity** | MEDIUM |
| **Estimated Effort** | 16 hours |

**OSHA Requirement:**
> "The employer shall update and revalidate the process hazard analysis at least every five (5) years."

---

## 3. Priority Ranking Matrix

### 3.1 Priority Scoring Methodology

| Factor | Weight | Description |
|--------|--------|-------------|
| Regulatory Mandatory | 40% | Required by regulation |
| Safety Impact | 30% | Impact on personnel safety |
| Deadline Urgency | 15% | Compliance deadline proximity |
| Dependencies | 15% | Blocking other tasks |

### 3.2 Priority Rankings

| Rank | Task ID | Description | Score | Priority |
|------|---------|-------------|-------|----------|
| 1 | TASK-194 | NFPA 85 Combustion Safeguards | 95 | CRITICAL |
| 2 | TASK-182 | Hardwired Interlock Integration | 93 | CRITICAL |
| 3 | TASK-204 | FMEA for Safety Functions | 90 | CRITICAL |
| 4 | TASK-191 | EPA Part 60 NSPS Compliance | 88 | CRITICAL |
| 5 | TASK-193 | EPA Part 98 GHG Reporting | 88 | CRITICAL |
| 6 | TASK-196 | OSHA 1910.119 PSM Support | 87 | CRITICAL |
| 7 | TASK-201 | HAZOP for GL-001 | 85 | CRITICAL |
| 8 | TASK-202 | HAZOP for GL-002 | 85 | CRITICAL |
| 9 | TASK-163 | LOPA for GL-007 | 82 | HIGH |
| 10 | TASK-186 | ESD Test Procedures | 80 | HIGH |
| 11 | TASK-185 | Response Time Validation | 78 | HIGH |
| 12 | TASK-166 | SRS for GL-007 | 78 | HIGH |
| 13 | TASK-187 | Bypass Management System | 75 | HIGH |
| 14 | TASK-197 | EU IED Compliance | 73 | HIGH |
| 15 | TASK-200 | ISA 18.2 Alarm Management | 72 | HIGH |
| 16 | TASK-203 | HAZOP for GL-007 | 70 | HIGH |
| 17 | TASK-195 | NFPA 86 Furnace Compliance | 68 | MEDIUM |
| 18 | TASK-179 | CCF Mitigation | 65 | MEDIUM |
| 19 | TASK-180 | Diversity Requirements | 63 | MEDIUM |
| 20 | TASK-205 | Risk Matrix | 60 | MEDIUM |
| 21 | TASK-206 | Safeguard Verification | 58 | MEDIUM |
| 22 | TASK-207 | Action Item Tracking | 55 | MEDIUM |
| 23 | TASK-188 | Bypass Logging | 52 | MEDIUM |
| 24 | TASK-208 | Risk Register | 50 | MEDIUM |
| 25 | TASK-189 | ESD Simulation Mode | 45 | LOW |
| 26 | TASK-190 | ESD Audit Reports | 42 | LOW |
| 27 | TASK-209 | Risk Monitoring Dashboard | 40 | LOW |
| 28 | TASK-210 | Periodic Risk Review | 38 | LOW |

---

## 4. Implementation Sequence

### 4.1 Dependency Graph

```
                    PHASE 5 IMPLEMENTATION DEPENDENCIES
                    =====================================

WEEK 1-2: Foundation
+------------------+     +------------------+     +------------------+
| TASK-201         |     | TASK-202         |     | TASK-203         |
| HAZOP GL-001     |     | HAZOP GL-002     |     | HAZOP GL-007     |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
| TASK-163         |<----+ TASK-204         +---->| TASK-205         |
| LOPA GL-007      |     | FMEA Safety Func |     | Risk Matrix      |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
| TASK-166         |     | TASK-179         |     | TASK-206         |
| SRS GL-007       |     | CCF Mitigation   |     | Safeguard Verif  |
+------------------+     +--------+---------+     +------------------+
                                  |
                                  v
                         +------------------+
                         | TASK-180         |
                         | Diversity Req    |
                         +------------------+

WEEK 3-4: ESD Integration
+------------------+     +------------------+
| TASK-182         |     | TASK-194         |
| Hardwired Intrlk |     | NFPA 85 Combust  |
+--------+---------+     +--------+---------+
         |                        |
         +------------+-----------+
                      |
                      v
              +------------------+
              | TASK-185         |
              | Response Time    |
              +--------+---------+
                       |
         +-------------+-------------+
         |                           |
         v                           v
+------------------+        +------------------+
| TASK-186         |        | TASK-187         |
| ESD Test Proced  |        | Bypass Mgmt      |
+--------+---------+        +--------+---------+
         |                           |
         v                           v
+------------------+        +------------------+
| TASK-189         |        | TASK-188         |
| ESD Simulation   |        | Bypass Logging   |
+--------+---------+        +--------+---------+
         |                           |
         +-------------+-------------+
                       |
                       v
              +------------------+
              | TASK-190         |
              | ESD Audit Rpts   |
              +------------------+

WEEK 5-6: Regulatory Compliance
+------------------+     +------------------+     +------------------+
| TASK-191         |     | TASK-193         |     | TASK-196         |
| EPA Part 60      |     | EPA Part 98      |     | OSHA PSM         |
+------------------+     +------------------+     +------------------+

+------------------+     +------------------+     +------------------+
| TASK-195         |     | TASK-197         |     | TASK-200         |
| NFPA 86          |     | EU IED           |     | ISA 18.2         |
+------------------+     +------------------+     +------------------+

WEEK 7-8: Risk Management
+------------------+     +------------------+     +------------------+
| TASK-207         |     | TASK-208         |     | TASK-209         |
| Action Tracking  |     | Risk Register    |     | Risk Dashboard   |
+------------------+     +------------------+     +------------------+
                                  |
                                  v
                         +------------------+
                         | TASK-210         |
                         | Periodic Review  |
                         +------------------+
```

### 4.2 Recommended Implementation Timeline

| Week | Tasks | Focus Area | Deliverables |
|------|-------|------------|--------------|
| 1-2 | 201, 202, 203 | HAZOP Studies | 3 HAZOP study reports |
| 2-3 | 163, 204, 205 | Risk Analysis | LOPA, FMEA, Risk Matrix |
| 3-4 | 166, 179, 180 | SRS & Fail-Safe | SRS document, CCF analysis |
| 4-5 | 182, 194, 185 | ESD & NFPA | Hardwired interface, NFPA 85 |
| 5-6 | 186, 187, 188 | ESD Complete | Test procedures, bypass system |
| 6-7 | 191, 193, 196 | US Regulatory | EPA Part 60/98, OSHA PSM |
| 7-8 | 195, 197, 200 | Other Regulatory | NFPA 86, EU IED, ISA 18.2 |
| 8-9 | 206, 207, 208 | Risk Management | Verification, tracking, register |
| 9-10 | 189, 190, 209, 210 | Finalization | Simulation, audit, dashboard |

---

## 5. Documentation Templates

### 5.1 HAZOP Worksheet Template

```markdown
# HAZOP WORKSHEET

**Study ID:** HAZOP-XXX
**Node:** [Node Name]
**Design Intent:** [Description of intended operation]
**P&ID Reference:** [Drawing number]

| ID | Guide Word | Parameter | Deviation | Causes | Consequences | Safeguards | Severity | Likelihood | Risk | Recommendations |
|----|------------|-----------|-----------|--------|--------------|------------|----------|------------|------|-----------------|
| 1.1 | NO | Flow | No flow | Pump failure | Overheat | Low flow alarm | 4 | 3 | 12 | Add redundant pump |
| 1.2 | MORE | Flow | High flow | Control valve fail open | Flooding | High flow trip | 3 | 2 | 6 | None |

**Team Members:**
- Leader: [Name]
- Process: [Name]
- Safety: [Name]
- Operations: [Name]

**Date:** YYYY-MM-DD
**Status:** Draft | Review | Approved
```

### 5.2 FMEA Worksheet Template

```markdown
# FMEA WORKSHEET

**Study ID:** FMEA-XXX
**System:** [System Name]
**Prepared By:** [Name]
**Date:** YYYY-MM-DD

| FM ID | Component | Function | Failure Mode | Local Effect | System Effect | End Effect | Cause | S | O | D | RPN | Action | Resp | Status |
|-------|-----------|----------|--------------|--------------|---------------|------------|-------|---|---|---|-----|--------|------|--------|
| FM-001 | Sensor A | Measure temp | Drift low | Wrong reading | Under-protection | Damage | Aging | 8 | 4 | 7 | 224 | Add comparison | Controls | Open |

**Rating Scales:**
- Severity (S): 1-10 (10 = most severe)
- Occurrence (O): 1-10 (10 = most frequent)
- Detection (D): 1-10 (10 = hardest to detect)

**RPN Threshold:** 100 (actions required above threshold)
```

### 5.3 LOPA Worksheet Template

```markdown
# LOPA WORKSHEET

**Scenario ID:** LOPA-XXX
**Description:** [Scenario description]
**Consequence Category:** [Cat 1-5]
**TMEL:** [Target Mitigated Event Likelihood]

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | [Description] | [freq/yr] | - |
| Enabling Condition | [Description] | [probability] | - |
| **Unmitigated Frequency** | | [freq/yr] | - |
| IPL 1 | [Description] | [PFD] | [credit] |
| IPL 2 | [Description] | [PFD] | [credit] |
| **IPL Total** | | [PFD] | [total] |
| **Intermediate Frequency** | | [freq/yr] | - |
| **Required SIF PFD** | TMEL / Intermediate | [PFD] | - |
| **Required SIL** | | [SIL 1/2/3] | - |

**Assumptions:**
1. [Assumption 1]
2. [Assumption 2]

**References:**
- [Reference 1]
- [Reference 2]
```

### 5.4 ESD Test Procedure Template

```markdown
# ESD TEST PROCEDURE

**Procedure ID:** ESD-TEST-XXX
**SIF ID:** [SIF-001, etc.]
**Test Type:** Full Proof Test | Partial Stroke Test
**Revision:** X.X
**Effective Date:** YYYY-MM-DD

## 1. PURPOSE
[Description of test purpose]

## 2. SCOPE
[Equipment and functions covered]

## 3. PREREQUISITES
- [ ] Work permit obtained
- [ ] Operations notified
- [ ] Bypass activated (if applicable)
- [ ] Test equipment calibrated

## 4. TEST PROCEDURE

| Step | Action | Expected Result | Actual Result | Pass/Fail |
|------|--------|-----------------|---------------|-----------|
| 1 | [Action] | [Expected] | | |
| 2 | [Action] | [Expected] | | |

## 5. ACCEPTANCE CRITERIA
- Response time: < [X] seconds
- Trip verified: Yes/No
- Valve stroke: [X]%

## 6. SIGN-OFF

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Tester | | | |
| Witness | | | |
| Supervisor | | | |
```

---

## 6. Specific Standard Requirements

### 6.1 IEC 61511 Key Requirements Summary

| Clause | Requirement | Task Mapping |
|--------|-------------|--------------|
| 8 | SIL determination methods | TASK-163 |
| 10 | Safety Requirements Specification | TASK-166 |
| 11.4 | Common cause failure | TASK-179, TASK-180 |
| 11.7 | Bypass management | TASK-187, TASK-188 |
| 12.4 | Response time verification | TASK-185 |
| 16.2 | Proof testing | TASK-186 |

### 6.2 NFPA 85/86 Key Requirements Summary

| Section | Requirement | Task Mapping |
|---------|-------------|--------------|
| NFPA 85 Ch. 8 | BMS requirements | TASK-194 |
| NFPA 85 8.5.3.4 | Flame failure timing | TASK-194 |
| NFPA 85 8.6 | Safety interlocks | TASK-194 |
| NFPA 86 Ch. 8 | Furnace safety | TASK-195 |
| NFPA 86 8.3 | Class C furnaces | TASK-195 |

### 6.3 EPA Part 60/98 Key Requirements Summary

| Regulation | Requirement | Task Mapping |
|------------|-------------|--------------|
| 40 CFR 60.44b | NOx limits | TASK-191 |
| 40 CFR 60.45b | SO2 limits | TASK-191 |
| 40 CFR 98.30-38 | GHG calculation | TASK-193 |
| 40 CFR 98.36 | Data requirements | TASK-193 |

---

## 7. Compliance Deadlines and Penalties

### 7.1 US Federal Requirements

| Regulation | Deadline | Penalty |
|------------|----------|---------|
| EPA Part 60 | Continuous | Up to $127,500/day |
| EPA Part 98 | March 31 annually | Up to $127,500/day |
| OSHA 1910.119 | Ongoing | Up to $161,323/serious |

### 7.2 Insurance/Certification Requirements

| Standard | Requirement | Impact |
|----------|-------------|--------|
| NFPA 85/86 | Compliance for insurance | Coverage denial risk |
| IEC 61511 | SIL certification | Liability risk |
| ISA 18.2 | Best practice | Audit findings |

### 7.3 EU Requirements

| Regulation | Deadline | Penalty |
|------------|----------|---------|
| EU IED | Ongoing permit | Operation suspension |
| BAT Conclusions | Within 4 years of publication | Permit review |

---

## 8. Recommended Action Plan

### 8.1 Immediate Actions (Week 1-2)

| Priority | Task | Owner | Due Date |
|----------|------|-------|----------|
| 1 | Begin HAZOP studies (201, 202, 203) | Safety Team | Week 2 |
| 2 | Complete NFPA 85 implementation (194) | Controls | Week 2 |
| 3 | Start hardwired interlock design (182) | I&C | Week 2 |

### 8.2 Short-Term Actions (Week 3-6)

| Priority | Task | Owner | Due Date |
|----------|------|-------|----------|
| 1 | Complete LOPA for GL-007 (163) | Safety | Week 3 |
| 2 | Finalize FMEA for safety functions (204) | Safety | Week 4 |
| 3 | Implement ESD test procedures (186) | Controls | Week 5 |
| 4 | Complete EPA Part 60/98 (191, 193) | Environmental | Week 6 |

### 8.3 Medium-Term Actions (Week 7-10)

| Priority | Task | Owner | Due Date |
|----------|------|-------|----------|
| 1 | Complete OSHA PSM support (196) | Safety | Week 7 |
| 2 | Implement ISA 18.2 alarm management (200) | Controls | Week 8 |
| 3 | Build risk register and dashboard (208, 209) | Safety | Week 9 |
| 4 | Establish periodic review process (210) | Safety | Week 10 |

### 8.4 Resource Requirements

| Resource Type | Quantity | Duration |
|---------------|----------|----------|
| Safety Engineers | 2 FTE | 10 weeks |
| Controls Engineers | 2 FTE | 8 weeks |
| Process Engineers | 1 FTE | 6 weeks |
| External HAZOP Facilitator | 1 | 2 weeks |
| Regulatory Specialist | 1 FTE | 4 weeks |

### 8.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Phase 5 Completion | 100% | Task completion |
| HAZOP Studies | 3 completed | Documents signed |
| FMEA RPN > 200 | 0 items | RPN reduction |
| ESD Response Time | <1 second | Validated tests |
| Regulatory Compliance | 100% | Audit findings |

---

## Appendix A: Reference Documents

| Document | Location |
|----------|----------|
| LOPA Analysis | `C:\Users\aksha\Code-V1_GreenLang\docs\safety\iec_61511\02_LOPA_ANALYSIS.md` |
| SRS Template | `C:\Users\aksha\Code-V1_GreenLang\docs\safety\iec_61511\01_SAFETY_REQUIREMENTS_SPECIFICATION.md` |
| GL-007 HAZOP | `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\03_GL007_HAZOP_STUDY.md` |
| FMEA Study | `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\04_FMEA_SAFETY_FUNCTIONS.md` |
| Risk Matrix | `C:\Users\aksha\Code-V1_GreenLang\docs\safety\hazop_fmea\05_RISK_MATRIX.md` |
| NFPA 85 Checker | `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\nfpa_85_checker.py` |
| EPA Reporter | `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\compliance\epa_reporter.py` |
| Bypass Manager | `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\esd\bypass_manager.py` |
| HAZOP Analyzer | `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\risk\hazop_analyzer.py` |
| FMEA Analyzer | `C:\Users\aksha\Code-V1_GreenLang\greenlang\safety\risk\fmea_analyzer.py` |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-RegulatoryIntelligence | Initial release |

---

**Document End**

*This document is part of the GreenLang Regulatory Intelligence documentation package.*
