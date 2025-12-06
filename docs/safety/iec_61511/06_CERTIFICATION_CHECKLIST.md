# IEC 61511 Certification Checklist

## GreenLang Process Heat Agents - Pre-Certification Assessment

**Document ID:** GL-SIL-CHK-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-1:2016, IEC 61511-2:2016

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Documentation Requirements](#2-documentation-requirements)
3. [Safety Lifecycle Compliance](#3-safety-lifecycle-compliance)
4. [Hardware Requirements](#4-hardware-requirements)
5. [Software Requirements](#5-software-requirements)
6. [Testing Requirements](#6-testing-requirements)
7. [Operational Requirements](#7-operational-requirements)
8. [Third-Party Assessment Requirements](#8-third-party-assessment-requirements)
9. [Certification Readiness Summary](#9-certification-readiness-summary)

---

## 1. Introduction

### 1.1 Purpose

This checklist provides a comprehensive assessment of readiness for IEC 61511 SIL certification of the GreenLang Process Heat agents. It ensures all requirements are addressed before third-party assessment.

### 1.2 Scope

This checklist covers:

- GL-001 Thermal Command Agent (SIL 2)
- GL-005 Building Energy Agent (SIL 1)
- GL-007 EU Taxonomy Agent (SIL 1)

### 1.3 Checklist Usage

| Symbol | Meaning |
|--------|---------|
| [ ] | Not started / Not verified |
| [P] | Partially complete |
| [X] | Complete / Verified |
| [N/A] | Not applicable |

### 1.4 Reference Documents

| ID | Document | Status |
|----|----------|--------|
| GL-SIL-OV-001 | SIL Certification Overview | [ ] |
| GL-SIL-SRS-001 | Safety Requirements Specification | [ ] |
| GL-SIL-LOPA-001 | LOPA Analysis | [ ] |
| GL-SIL-VOTE-001 | Voting Logic Specification | [ ] |
| GL-SIL-PT-001 | Proof Test Procedures | [ ] |
| GL-SIL-INT-001 | GL-001 SIS Integration Specification | [ ] |

---

## 2. Documentation Requirements

### 2.1 Safety Lifecycle Documentation

#### 2.1.1 Phase 1: Analysis

| Ref | Requirement | IEC 61511 Clause | Document | Status | Comments |
|-----|-------------|------------------|----------|--------|----------|
| 2.1.1.1 | Process Hazard Analysis (PHA/HAZOP) documented | 8.2 | GL-HAZOP-001 | [ ] | |
| 2.1.1.2 | Risk assessment methodology defined | 8.2.3 | GL-SIL-OV-001 | [ ] | |
| 2.1.1.3 | SIL determination documented (LOPA or Risk Graph) | 9 | GL-SIL-LOPA-001 | [ ] | |
| 2.1.1.4 | Target SIL for each SIF defined | 9.3 | GL-SIL-SRS-001 | [ ] | |
| 2.1.1.5 | Safety function allocation documented | 9.4 | GL-SIL-SRS-001 | [ ] | |

#### 2.1.2 Phase 2: Realization

| Ref | Requirement | IEC 61511 Clause | Document | Status | Comments |
|-----|-------------|------------------|----------|--------|----------|
| 2.1.2.1 | Safety Requirements Specification (SRS) | 10 | GL-SIL-SRS-001 | [ ] | |
| 2.1.2.2 | SIF inputs, logic, outputs defined | 10.3.1 | GL-SIL-SRS-001 | [ ] | |
| 2.1.2.3 | Safe state definition for each SIF | 10.3.1 | GL-SIL-SRS-001 | [ ] | |
| 2.1.2.4 | Response time requirements | 10.3.1 | GL-SIL-SRS-001 | [ ] | |
| 2.1.2.5 | Proof test requirements | 10.3.1 | GL-SIL-PT-001 | [ ] | |
| 2.1.2.6 | SIS architecture design | 11 | GL-SIL-VOTE-001 | [ ] | |
| 2.1.2.7 | PFD calculations documented | 11.9 | GL-SIL-VOTE-001 | [ ] | |
| 2.1.2.8 | Hardware selection justified | 11.4 | GL-SIL-VOTE-001 | [ ] | |
| 2.1.2.9 | Software requirements specification | 12.4 | GL-SIL-INT-001 | [ ] | |

#### 2.1.3 Phase 3: Operation

| Ref | Requirement | IEC 61511 Clause | Document | Status | Comments |
|-----|-------------|------------------|----------|--------|----------|
| 2.1.3.1 | Operation and maintenance procedures | 16 | GL-SIL-PT-001 | [ ] | |
| 2.1.3.2 | Proof test procedures | 16.3 | GL-SIL-PT-001 | [ ] | |
| 2.1.3.3 | Bypass management procedures | 16.2.2 | GL-OPS-001 | [ ] | |
| 2.1.3.4 | Modification management procedures | 17 | GL-MOC-001 | [ ] | |

### 2.2 Document Control

| Ref | Requirement | Status | Comments |
|-----|-------------|--------|----------|
| 2.2.1 | Document revision control established | [ ] | |
| 2.2.2 | Document approval process defined | [ ] | |
| 2.2.3 | Distribution list maintained | [ ] | |
| 2.2.4 | Master document register available | [ ] | |
| 2.2.5 | Cross-reference matrix complete | [ ] | |

---

## 3. Safety Lifecycle Compliance

### 3.1 Management and Organization (Clause 5)

| Ref | Requirement | IEC 61511 Clause | Status | Evidence | Comments |
|-----|-------------|------------------|--------|----------|----------|
| 3.1.1 | Functional Safety Management System defined | 5.2.1 | [ ] | | |
| 3.1.2 | Roles and responsibilities documented | 5.2.2 | [ ] | | |
| 3.1.3 | Competency requirements defined | 5.2.5 | [ ] | | |
| 3.1.4 | Training records maintained | 5.2.5 | [ ] | | |
| 3.1.5 | FSA independence requirements met | 5.2.6 | [ ] | | |
| 3.1.6 | Planning and scheduling documented | 5.2.3 | [ ] | | |

### 3.2 Hazard and Risk Assessment (Clauses 8-9)

| Ref | Requirement | IEC 61511 Clause | Status | Evidence | Comments |
|-----|-------------|------------------|--------|----------|----------|
| 3.2.1 | HAZOP or equivalent PHA completed | 8.2 | [ ] | | |
| 3.2.2 | Risk criteria defined and approved | 9.2.3 | [ ] | | |
| 3.2.3 | SIL determination methodology documented | 9.2 | [ ] | | |
| 3.2.4 | LOPA worksheets completed for each SIF | 9.2.4 | [ ] | | |
| 3.2.5 | IPLs validated as independent | 9.2.4 | [ ] | | |
| 3.2.6 | Initiating event frequencies documented | 9.2.4 | [ ] | | |
| 3.2.7 | SIL targets assigned to all SIFs | 9.3 | [ ] | | |

### 3.3 Safety Requirements Specification (Clause 10)

| Ref | Requirement | IEC 61511 Clause | Status | Evidence | Comments |
|-----|-------------|------------------|--------|----------|----------|
| 3.3.1 | Each SIF fully described | 10.3.1(a) | [ ] | | |
| 3.3.2 | Safe states defined | 10.3.1(b) | [ ] | | |
| 3.3.3 | SIL assigned to each SIF | 10.3.1(c) | [ ] | | |
| 3.3.4 | Demand mode specified | 10.3.1(d) | [ ] | | |
| 3.3.5 | MTTR constraints defined | 10.3.1(e) | [ ] | | |
| 3.3.6 | Spurious trip rate specified | 10.3.1(f) | [ ] | | |
| 3.3.7 | Response time requirements | 10.3.1(g) | [ ] | | |
| 3.3.8 | Process interface requirements | 10.3.1(h) | [ ] | | |
| 3.3.9 | Operating modes documented | 10.3.1(i) | [ ] | | |
| 3.3.10 | Bypass requirements defined | 10.3.1(j) | [ ] | | |
| 3.3.11 | Maintenance requirements | 10.3.1(k) | [ ] | | |
| 3.3.12 | Test requirements defined | 10.3.1(l) | [ ] | | |

### 3.4 SIS Design and Engineering (Clause 11)

| Ref | Requirement | IEC 61511 Clause | Status | Evidence | Comments |
|-----|-------------|------------------|--------|----------|----------|
| 3.4.1 | Architecture meets HFT requirements | 11.4 | [ ] | | |
| 3.4.2 | Redundancy requirements met | 11.4 | [ ] | | |
| 3.4.3 | Diagnostic coverage adequate | 11.4 | [ ] | | |
| 3.4.4 | Common cause failures addressed | 11.4.4 | [ ] | | |
| 3.4.5 | PFD calculations verified | 11.9 | [ ] | | |
| 3.4.6 | Response time budget documented | 11.5 | [ ] | | |
| 3.4.7 | Environmental conditions specified | 11.4.2 | [ ] | | |
| 3.4.8 | Process interface designed | 11.6 | [ ] | | |
| 3.4.9 | Final element selection justified | 11.4.3 | [ ] | | |

---

## 4. Hardware Requirements

### 4.1 Sensor Subsystem

| Ref | Requirement | SIF | Status | Evidence | Comments |
|-----|-------------|-----|--------|----------|----------|
| 4.1.1 | Sensors selected per SIL requirement | All | [ ] | | |
| 4.1.2 | Sensor data sheets available | All | [ ] | | |
| 4.1.3 | SIL certificates for sensors (if claiming) | All | [ ] | | |
| 4.1.4 | Failure mode analysis completed | All | [ ] | | |
| 4.1.5 | Proof test coverage documented | All | [ ] | | |
| 4.1.6 | Voting architecture implemented | All | [ ] | | |

**Temperature Sensors (SIF-001):**

| Item | Requirement | Status |
|------|-------------|--------|
| TE-001A | Range 0-500C, 4-20mA, SIL capable | [ ] |
| TE-001B | Range 0-500C, 4-20mA, SIL capable | [ ] |
| Installation | Per manufacturer spec, diverse routing | [ ] |

**Flow Sensors (SIF-002):**

| Item | Requirement | Status |
|------|-------------|--------|
| FT-001A | Range 0-100%, 4-20mA, SIL capable | [ ] |
| FT-001B | Range 0-100%, 4-20mA, SIL capable | [ ] |
| Installation | Per manufacturer spec | [ ] |

**Pressure Sensors (SIF-003):**

| Item | Requirement | Status |
|------|-------------|--------|
| PT-001A | Range 0-100 bar, 4-20mA, SIL capable | [ ] |
| PT-001B | Range 0-100 bar, 4-20mA, SIL capable | [ ] |
| Installation | Per manufacturer spec | [ ] |

**Flame Scanners (SIF-004):**

| Item | Requirement | Status |
|------|-------------|--------|
| FS-001A | UV/IR scanner, self-checking | [ ] |
| FS-001B | UV/IR scanner, self-checking | [ ] |
| FS-001C | UV/IR scanner, self-checking | [ ] |
| Installation | Per NFPA 85/86 and manufacturer | [ ] |

### 4.2 Logic Solver

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 4.2.1 | Safety PLC selected | [ ] | | |
| 4.2.2 | PLC SIL certificate available | [ ] | | |
| 4.2.3 | PLC certified to >= target SIL | [ ] | | |
| 4.2.4 | Safety manual obtained | [ ] | | |
| 4.2.5 | Architecture constraints documented | [ ] | | |
| 4.2.6 | Failure rate data available | [ ] | | |
| 4.2.7 | Proof test requirements documented | [ ] | | |
| 4.2.8 | Watchdog function configured | [ ] | | |

**Logic Solver Selection:**

| Requirement | Specification | Actual | Status |
|-------------|---------------|--------|--------|
| Manufacturer | TUV certified vendor | | [ ] |
| Model | | | [ ] |
| SIL capability | SIL 3 | | [ ] |
| Certificate number | | | [ ] |
| Certificate expiry | | | [ ] |

### 4.3 Final Elements

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 4.3.1 | Valves selected per SIL requirement | [ ] | | |
| 4.3.2 | Valve data sheets available | [ ] | | |
| 4.3.3 | Fail-safe action verified | [ ] | | |
| 4.3.4 | Stroke time documented | [ ] | | |
| 4.3.5 | Partial stroke testing capability | [ ] | | |
| 4.3.6 | Position feedback available | [ ] | | |

**Fuel Isolation Valves (SIF-001, SIF-004):**

| Item | Requirement | Status |
|------|-------------|--------|
| XV-001A | Fail-close, spring return, position FB | [ ] |
| XV-001B | Fail-close, spring return, position FB | [ ] |
| XV-004A | Fail-close, spring return, position FB | [ ] |
| XV-004B | Fail-close, spring return, position FB | [ ] |
| Stroke time | < 2 seconds | [ ] |

### 4.4 Power Supply and Wiring

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 4.4.1 | Power supply redundancy adequate | [ ] | | |
| 4.4.2 | UPS capacity documented | [ ] | | |
| 4.4.3 | Cable segregation from non-safety | [ ] | | |
| 4.4.4 | Grounding scheme documented | [ ] | | |
| 4.4.5 | EMC requirements addressed | [ ] | | |

---

## 5. Software Requirements

### 5.1 Application Software (Clause 12)

| Ref | Requirement | IEC 61511 Clause | Status | Evidence | Comments |
|-----|-------------|------------------|--------|----------|----------|
| 5.1.1 | Software development lifecycle defined | 12.2 | [ ] | | |
| 5.1.2 | Software requirements specification | 12.4 | [ ] | | |
| 5.1.3 | Software design specification | 12.4.2 | [ ] | | |
| 5.1.4 | Coding standards defined | 12.4.2 | [ ] | | |
| 5.1.5 | Limited Variability Language used | 12.4.2 | [ ] | | |
| 5.1.6 | Module testing completed | 12.5 | [ ] | | |
| 5.1.7 | Integration testing completed | 12.5 | [ ] | | |
| 5.1.8 | Software review completed | 12.6 | [ ] | | |
| 5.1.9 | Version control implemented | 12.4.3 | [ ] | | |
| 5.1.10 | Configuration management | 12.4.3 | [ ] | | |

### 5.2 GL-001 Software Requirements

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 5.2.1 | GL-001 is NOT in certified safety path | [ ] | | |
| 5.2.2 | GL-001 provides monitoring only | [ ] | | |
| 5.2.3 | Trip request is non-certified path | [ ] | | |
| 5.2.4 | Communication watchdog implemented | [ ] | | |
| 5.2.5 | Fail-safe on communication loss | [ ] | | |
| 5.2.6 | Version control established | [ ] | | |
| 5.2.7 | Software testing documented | [ ] | | |

### 5.3 Embedded Software (Safety PLC)

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 5.3.1 | Safety PLC program reviewed | [ ] | | |
| 5.3.2 | Logic matches P&ID/SRS | [ ] | | |
| 5.3.3 | Voting logic correctly implemented | [ ] | | |
| 5.3.4 | Time delays correctly configured | [ ] | | |
| 5.3.5 | Setpoints verified | [ ] | | |
| 5.3.6 | Bypass logic reviewed | [ ] | | |
| 5.3.7 | Reset logic reviewed | [ ] | | |
| 5.3.8 | Program backup maintained | [ ] | | |
| 5.3.9 | Access control implemented | [ ] | | |

---

## 6. Testing Requirements

### 6.1 Factory Acceptance Test (FAT)

| Ref | Requirement | Status | Date | Witness | Comments |
|-----|-------------|--------|------|---------|----------|
| 6.1.1 | FAT procedure documented | [ ] | | | |
| 6.1.2 | FAT acceptance criteria defined | [ ] | | | |

**FAT Test Items:**

| Test | SIF | Status | Date | Pass/Fail |
|------|-----|--------|------|-----------|
| Sensor simulation test | All | [ ] | | |
| Logic function test | All | [ ] | | |
| Voting logic test | All | [ ] | | |
| Response time test | All | [ ] | | |
| Final element test | All | [ ] | | |
| Failure mode test | All | [ ] | | |
| Communication test | All | [ ] | | |
| Alarm test | All | [ ] | | |
| HMI test | All | [ ] | | |
| Documentation review | All | [ ] | | |

### 6.2 Site Acceptance Test (SAT)

| Ref | Requirement | Status | Date | Witness | Comments |
|-----|-------------|--------|------|---------|----------|
| 6.2.1 | SAT procedure documented | [ ] | | | |
| 6.2.2 | SAT acceptance criteria defined | [ ] | | | |

**SAT Test Items:**

| Test | SIF | Status | Date | Pass/Fail |
|------|-----|--------|------|-----------|
| Loop check (each I/O) | All | [ ] | | |
| End-to-end trip test | All | [ ] | | |
| Response time with actual I/O | All | [ ] | | |
| Process interface verification | All | [ ] | | |
| Alarm verification | All | [ ] | | |
| BPCS/SIS interface test | All | [ ] | | |
| Full system integration | All | [ ] | | |

### 6.3 Pre-Startup Safety Review (PSSR)

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 6.3.1 | PSSR checklist completed | [ ] | | |
| 6.3.2 | All test records reviewed | [ ] | | |
| 6.3.3 | Outstanding items closed | [ ] | | |
| 6.3.4 | Operating procedures approved | [ ] | | |
| 6.3.5 | Training completed | [ ] | | |
| 6.3.6 | Management of Change complete | [ ] | | |
| 6.3.7 | Authorization to start signed | [ ] | | |

### 6.4 Proof Test Readiness

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 6.4.1 | Proof test procedures documented | [ ] | GL-SIL-PT-001 | |
| 6.4.2 | Test equipment identified | [ ] | | |
| 6.4.3 | Calibration requirements defined | [ ] | | |
| 6.4.4 | Test schedule established | [ ] | | |
| 6.4.5 | Record keeping system ready | [ ] | | |
| 6.4.6 | Personnel competency verified | [ ] | | |

---

## 7. Operational Requirements

### 7.1 Operation Procedures

| Ref | Requirement | Status | Document | Comments |
|-----|-------------|--------|----------|----------|
| 7.1.1 | Normal operating procedures | [ ] | | |
| 7.1.2 | Emergency operating procedures | [ ] | | |
| 7.1.3 | Startup procedures | [ ] | | |
| 7.1.4 | Shutdown procedures | [ ] | | |
| 7.1.5 | Alarm response procedures | [ ] | | |
| 7.1.6 | Trip response procedures | [ ] | | |

### 7.2 Maintenance Procedures

| Ref | Requirement | Status | Document | Comments |
|-----|-------------|--------|----------|----------|
| 7.2.1 | Preventive maintenance schedule | [ ] | | |
| 7.2.2 | Proof test procedures | [ ] | GL-SIL-PT-001 | |
| 7.2.3 | Calibration procedures | [ ] | | |
| 7.2.4 | Repair procedures | [ ] | | |
| 7.2.5 | Spare parts list | [ ] | | |

### 7.3 Bypass Management (Clause 16.2.2)

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 7.3.1 | Bypass procedure documented | [ ] | | |
| 7.3.2 | Bypass authorization requirements | [ ] | | |
| 7.3.3 | Bypass time limits defined | [ ] | | |
| 7.3.4 | Compensating measures identified | [ ] | | |
| 7.3.5 | Bypass indication on HMI | [ ] | | |
| 7.3.6 | Bypass logging/recording | [ ] | | |

### 7.4 Management of Change (Clause 17)

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 7.4.1 | MOC procedure documented | [ ] | | |
| 7.4.2 | Impact assessment requirements | [ ] | | |
| 7.4.3 | Approval levels defined | [ ] | | |
| 7.4.4 | Documentation update requirements | [ ] | | |
| 7.4.5 | Re-verification requirements | [ ] | | |

### 7.5 Training Requirements

| Ref | Requirement | Status | Evidence | Comments |
|-----|-------------|--------|----------|----------|
| 7.5.1 | Operator training program | [ ] | | |
| 7.5.2 | Maintenance technician training | [ ] | | |
| 7.5.3 | Engineering training | [ ] | | |
| 7.5.4 | Training records maintained | [ ] | | |
| 7.5.5 | Competency assessment documented | [ ] | | |

---

## 8. Third-Party Assessment Requirements

### 8.1 Assessment Scope

| Ref | Requirement | Status | Comments |
|-----|-------------|--------|----------|
| 8.1.1 | Assessment scope defined and agreed | [ ] | |
| 8.1.2 | Certification body selected | [ ] | |
| 8.1.3 | Assessor qualifications verified | [ ] | |
| 8.1.4 | Assessment schedule agreed | [ ] | |

**Certification Body Information:**

| Item | Details |
|------|---------|
| Certification Body | |
| Accreditation | |
| Lead Assessor | |
| Contact | |
| Assessment Date | |

### 8.2 Independence Requirements

| SIL | Requirement | Status | Evidence |
|-----|-------------|--------|----------|
| SIL 1 | Independent person | [ ] | |
| SIL 2 | Independent department or external | [ ] | |
| SIL 3 | Independent organization | [ ] | |

### 8.3 Documentation Package for Assessment

| Ref | Document | Required For | Status | Comments |
|-----|----------|--------------|--------|----------|
| 8.3.1 | SIL Certification Overview | Assessment | [ ] | |
| 8.3.2 | HAZOP/PHA Report | SIL determination | [ ] | |
| 8.3.3 | LOPA Worksheets | SIL determination | [ ] | |
| 8.3.4 | Safety Requirements Specification | Design review | [ ] | |
| 8.3.5 | Architecture Design | Design review | [ ] | |
| 8.3.6 | PFD Calculations | SIL verification | [ ] | |
| 8.3.7 | Hardware Selection Justification | Hardware review | [ ] | |
| 8.3.8 | Software Design Documentation | Software review | [ ] | |
| 8.3.9 | FAT Records | Testing review | [ ] | |
| 8.3.10 | SAT Records | Testing review | [ ] | |
| 8.3.11 | Proof Test Procedures | O&M review | [ ] | |
| 8.3.12 | Training Records | Competency review | [ ] | |
| 8.3.13 | FSM Documentation | Management review | [ ] | |

### 8.4 Assessment Activities

| Ref | Activity | Planned Date | Status | Findings |
|-----|----------|--------------|--------|----------|
| 8.4.1 | Document review | | [ ] | |
| 8.4.2 | Design review meeting | | [ ] | |
| 8.4.3 | FAT witness | | [ ] | |
| 8.4.4 | SAT witness | | [ ] | |
| 8.4.5 | Operational readiness review | | [ ] | |
| 8.4.6 | Final assessment meeting | | [ ] | |

### 8.5 Certificate Requirements

| Ref | Requirement | Status | Comments |
|-----|-------------|--------|----------|
| 8.5.1 | Certificate scope matches project | [ ] | |
| 8.5.2 | Certificate validity period adequate | [ ] | |
| 8.5.3 | Certificate constraints documented | [ ] | |
| 8.5.4 | Surveillance requirements understood | [ ] | |

---

## 9. Certification Readiness Summary

### 9.1 Readiness Assessment by SIF

| SIF | SIL | Documentation | Hardware | Software | Testing | Overall |
|-----|-----|---------------|----------|----------|---------|---------|
| SIF-001 | SIL 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SIF-002 | SIL 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SIF-003 | SIL 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SIF-004 | SIL 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SIF-005 | SIL 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SIF-006 | SIL 1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SIF-007 | SIL 1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| SIF-008 | SIL 1 | [ ] | [ ] | [ ] | [ ] | [ ] |

### 9.2 Open Items Summary

| Item # | Description | Owner | Due Date | Status |
|--------|-------------|-------|----------|--------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### 9.3 Readiness Declaration

| Section | Complete | Partial | Not Started | N/A |
|---------|----------|---------|-------------|-----|
| Documentation | | | | |
| Safety Lifecycle | | | | |
| Hardware | | | | |
| Software | | | | |
| Testing | | | | |
| Operations | | | | |
| Third-Party | | | | |

### 9.4 Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Manager | | | |
| Safety Engineer | | | |
| Functional Safety Manager | | | |
| Quality Assurance | | | |
| Operations Manager | | | |

### 9.5 Certification Recommendation

Based on the assessment documented in this checklist:

[ ] Ready for third-party certification assessment
[ ] Not ready - open items must be closed first

**Comments:**

---

## Appendix A: IEC 61511 Clause Cross-Reference

| Clause | Title | Document Reference |
|--------|-------|-------------------|
| 5 | Management of functional safety | GL-SIL-OV-001 |
| 6 | Safety lifecycle requirements | GL-SIL-OV-001 |
| 7 | Verification | All documents |
| 8 | Process hazard and risk assessment | GL-HAZOP-001 |
| 9 | Allocation of safety functions | GL-SIL-LOPA-001 |
| 10 | Safety requirements specification | GL-SIL-SRS-001 |
| 11 | SIS design and engineering | GL-SIL-VOTE-001 |
| 12 | Application software | GL-SIL-INT-001 |
| 13 | Factory acceptance testing | This checklist |
| 14 | SIS installation and commissioning | This checklist |
| 15 | SIS safety validation | This checklist |
| 16 | SIS operation and maintenance | GL-SIL-PT-001 |
| 17 | SIS modification | GL-MOC-001 |
| 18 | SIS decommissioning | N/A |

---

## Appendix B: Abbreviations

| Abbreviation | Definition |
|--------------|------------|
| BPCS | Basic Process Control System |
| CCF | Common Cause Failure |
| DC | Diagnostic Coverage |
| FAT | Factory Acceptance Test |
| FMEA | Failure Mode and Effects Analysis |
| FSM | Functional Safety Management |
| HAZOP | Hazard and Operability Study |
| HFT | Hardware Fault Tolerance |
| IPL | Independent Protection Layer |
| LOPA | Layer of Protection Analysis |
| MOC | Management of Change |
| MTTR | Mean Time to Repair |
| PFD | Probability of Failure on Demand |
| PHA | Process Hazard Analysis |
| PSSR | Pre-Startup Safety Review |
| SAT | Site Acceptance Test |
| SIF | Safety Instrumented Function |
| SIL | Safety Integrity Level |
| SIS | Safety Instrumented System |
| SRS | Safety Requirements Specification |

---

## Appendix C: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
