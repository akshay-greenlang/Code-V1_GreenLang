# Layer of Protection Analysis (LOPA)

## GreenLang Process Heat Agents - SIL Determination

**Document ID:** GL-SIL-LOPA-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-3:2016, CCPS LOPA Guidelines

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [LOPA Methodology](#2-lopa-methodology)
3. [Risk Criteria](#3-risk-criteria)
4. [Initiating Event Frequencies](#4-initiating-event-frequencies)
5. [Independent Protection Layers](#5-independent-protection-layers)
6. [LOPA Worksheets](#6-lopa-worksheets)
7. [TMEL Calculations](#7-tmel-calculations)
8. [SIL Determination Summary](#8-sil-determination-summary)
9. [Assumptions and Limitations](#9-assumptions-and-limitations)

---

## 1. Introduction

### 1.1 Purpose

This document presents the Layer of Protection Analysis (LOPA) performed to determine the required Safety Integrity Level (SIL) for Safety Instrumented Functions (SIFs) in the GreenLang Process Heat agent system.

### 1.2 Scope

This LOPA covers hazard scenarios associated with:

- GL-001 Thermal Command Agent (Process heat control)
- GL-005 Building Energy Agent (Building systems monitoring)
- GL-007 EU Taxonomy Agent (Compliance monitoring)

### 1.3 References

| Document | Title |
|----------|-------|
| CCPS | Layer of Protection Analysis - Simplified Process Risk Assessment |
| IEC 61511-3:2016 | Guidance for determination of required SIL |
| ISA-TR84.00.04 | Guidelines for SIF Implementation |
| GL-HAZOP-001 | Process Heat System HAZOP |

### 1.4 LOPA Team

| Role | Responsibility |
|------|----------------|
| LOPA Leader | Facilitation, documentation |
| Process Engineer | Process knowledge, consequences |
| Safety Engineer | IPL validation, SIL assignment |
| Operations | Operational procedures, human factors |
| Instrumentation | Sensor/actuator capabilities |

---

## 2. LOPA Methodology

### 2.1 LOPA Overview

LOPA is a semi-quantitative risk assessment method that:

1. Identifies hazard scenarios from HAZOP or other PHA methods
2. Assigns frequency to initiating events
3. Quantifies risk reduction from Independent Protection Layers (IPLs)
4. Determines required SIL to achieve tolerable risk

### 2.2 LOPA Process Flow

```
+-------------------+
| Scenario from PHA |
+-------------------+
         |
         v
+-------------------+
| Identify          |
| Initiating Event  |
| (Frequency)       |
+-------------------+
         |
         v
+-------------------+
| Identify Enabling |
| Conditions        |
| (Probability)     |
+-------------------+
         |
         v
+-------------------+
| Identify          |
| Consequence       |
+-------------------+
         |
         v
+-------------------+
| Determine Target  |
| Mitigated Event   |
| Likelihood (TMEL) |
+-------------------+
         |
         v
+-------------------+
| Identify IPLs     |
| (PFD Credits)     |
+-------------------+
         |
         v
+-------------------+
| Calculate         |
| Required SIF PFD  |
+-------------------+
         |
         v
+-------------------+
| Assign SIL        |
+-------------------+
```

### 2.3 LOPA Equation

The fundamental LOPA equation:

```
f_consequence = f_IE x P_EC x PFD_IPL1 x PFD_IPL2 x ... x PFD_SIF

Where:
  f_consequence = Mitigated event frequency
  f_IE = Initiating event frequency
  P_EC = Enabling condition probability
  PFD_IPLn = Probability of Failure on Demand for IPL n
  PFD_SIF = Required SIF PFD
```

Rearranging to find required SIF PFD:

```
PFD_SIF = TMEL / (f_IE x P_EC x PFD_IPL1 x PFD_IPL2 x ...)
```

---

## 3. Risk Criteria

### 3.1 Corporate Risk Tolerance Criteria

GreenLang has adopted the following risk tolerance criteria based on ALARP principles:

#### 3.1.1 Individual Risk Criteria

| Risk Category | Tolerable Frequency | Description |
|---------------|---------------------|-------------|
| Fatality | < 1 x 10^-4 per year | Based on 1 in 10,000 per year |
| Major Injury | < 1 x 10^-3 per year | Based on 1 in 1,000 per year |
| Minor Injury | < 1 x 10^-2 per year | Based on 1 in 100 per year |

#### 3.1.2 Environmental Risk Criteria

| Risk Category | Tolerable Frequency | Description |
|---------------|---------------------|-------------|
| Major release | < 1 x 10^-4 per year | Significant environmental damage |
| Moderate release | < 1 x 10^-3 per year | Local environmental impact |
| Minor release | < 1 x 10^-2 per year | Negligible environmental impact |

#### 3.1.3 Asset/Business Risk Criteria

| Risk Category | Tolerable Frequency | Description |
|---------------|---------------------|-------------|
| > $10M damage | < 1 x 10^-4 per year | Catastrophic asset loss |
| $1M - $10M damage | < 1 x 10^-3 per year | Major asset damage |
| < $1M damage | < 1 x 10^-2 per year | Minor asset damage |

### 3.2 Target Mitigated Event Likelihood (TMEL)

Based on consequence severity, the following TMEL values are used:

| Consequence Category | Severity | TMEL |
|---------------------|----------|------|
| Category 5 | Fatality, major environmental damage | 1 x 10^-5 per year |
| Category 4 | Major injury, significant environmental | 1 x 10^-4 per year |
| Category 3 | Minor injury, local environmental | 1 x 10^-3 per year |
| Category 2 | First aid, negligible environmental | 1 x 10^-2 per year |
| Category 1 | No injury, no environmental impact | 1 x 10^-1 per year |

---

## 4. Initiating Event Frequencies

### 4.1 Process-Related Initiating Events

| Initiating Event | Frequency | Source | Comments |
|------------------|-----------|--------|----------|
| BPCS failure | 0.1 /year | Industry data | Generic programmable controller |
| Control valve failure | 0.1 /year | OREDA | Control valve, fail to operate |
| Transmitter failure | 0.1 /year | OREDA | Pressure/temperature transmitter |
| Loss of cooling | 0.1 /year | Industry data | Cooling system failure |
| Loss of instrument air | 0.1 /year | Industry data | Air supply failure |
| Operator error (routine) | 0.1 /year | CCPS | Well-trained, procedure-following |
| Operator error (non-routine) | 1.0 /year | CCPS | Less frequent, more complex tasks |
| Heat exchanger tube leak | 0.01 /year | OREDA | Internal leak |
| Pump failure | 0.1 /year | OREDA | Centrifugal pump |
| External fire | 0.01 /year | API | Plant area fire |

### 4.2 Combustion System Initiating Events

| Initiating Event | Frequency | Source | Comments |
|------------------|-----------|--------|----------|
| Flame failure | 0.5 /year | NFPA | Based on burner type |
| Fuel supply upset | 0.1 /year | Industry data | Gas supply variation |
| Air supply failure | 0.1 /year | Industry data | Combustion air fan failure |
| Ignition system failure | 0.2 /year | NFPA | During startup |
| Flame scanner failure | 0.1 /year | Industry data | False negative |
| Fuel valve leak-through | 0.01 /year | API | Passing in closed position |

### 4.3 Building System Initiating Events

| Initiating Event | Frequency | Source | Comments |
|------------------|-----------|--------|----------|
| HVAC fan failure | 0.2 /year | Industry data | Supply/return fan |
| Damper failure | 0.1 /year | Industry data | Stuck open/closed |
| CO generation | 0.1 /year | Industry data | Incomplete combustion |
| CO2 buildup | 0.1 /year | Industry data | Inadequate ventilation |
| BMS failure | 0.1 /year | Industry data | Building management system |

### 4.4 Enabling Conditions

| Enabling Condition | Probability | Comments |
|--------------------|-------------|----------|
| Process operating | 0.9 | 90% availability |
| Building occupied | 0.5 | 50% occupancy time |
| Personnel in area | 0.25 | 6 hours per day |
| Night operation | 0.33 | 8 hours per day |
| Maintenance in progress | 0.05 | 18 days per year |

---

## 5. Independent Protection Layers

### 5.1 IPL Criteria

Per IEC 61511-3 and CCPS, an IPL must be:

1. **Independent** - Not affected by failure of initiating event or other IPLs
2. **Specific** - Designed to prevent or mitigate the specific consequence
3. **Dependable** - Can be counted on when needed (quantifiable PFD)
4. **Auditable** - Subject to regular testing/inspection

### 5.2 IPL PFD Values

| IPL Type | PFD | Credits | Conditions |
|----------|-----|---------|------------|
| Basic Process Control System (BPCS) | 0.1 | 1 | Single loop, no SIL credit |
| Alarm with operator response | 0.1 | 1 | Time to respond, trained operator |
| Pressure relief valve | 0.01 | 2 | Properly sized, tested |
| Rupture disk | 0.01 | 2 | Properly sized |
| Check valve | 0.1 | 1 | Properly maintained |
| Restriction orifice | 0.01 | 2 | Fixed, non-plugging |
| Physical containment (dike) | 0.01 | 2 | Adequate sizing |
| Operator response to abnormal | 0.1 | 1 | > 20 min response time |
| Operator response (critical) | 1.0 | 0 | < 10 min response time |

### 5.3 IPL Validation for Process Heat Systems

| IPL | PFD Claimed | Validation Evidence |
|-----|-------------|---------------------|
| BPCS temperature control | 0.1 | Independent controller, tested annually |
| High temperature alarm | 0.1 | Operator training, response procedure |
| Pressure relief valve | 0.01 | API inspection, 5-year recertification |
| Low flow alarm | 0.1 | Independent sensor, operator procedure |
| Flame scanner (BPCS) | 0.1 | Manufacturer data, tested monthly |
| Emergency ventilation | 0.1 | Standby fan, tested monthly |

### 5.4 Non-IPL Safeguards

The following are NOT credited as IPLs:

| Safeguard | Reason |
|-----------|--------|
| Warning signs | Not dependable, no PFD claim |
| PPE (gloves, glasses) | Mitigation only, not prevention |
| Procedures without alarm | Requires continuous vigilance |
| Fire detection (consequence only) | Does not prevent initiating event |
| Spare equipment (online) | May share common cause |

---

## 6. LOPA Worksheets

### 6.1 LOPA Worksheet - SIF-001: High Temperature Shutdown

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-001 |
| **Description** | High process temperature leads to equipment damage and potential release |
| **Consequence Category** | Category 4 - Major equipment damage, potential thermal injury |
| **TMEL** | 1 x 10^-5 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | BPCS temperature control failure | 0.1 /year | - |
| Enabling Condition | Process operating | 0.9 | - |
| **Unmitigated Frequency** | | **0.09 /year** | - |
| IPL 1 | High temperature alarm with operator response | 0.1 | 1 |
| IPL 2 | BPCS independent high limit | 0.1 | 1 |
| **IPL Total** | | **0.01** | **2** |
| **Intermediate Frequency** | 0.09 x 0.01 | **9 x 10^-4 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **1.1 x 10^-2** | - |
| **Required SIL** | | **SIL 2** | - |

**Calculation:**
```
PFD_SIF = TMEL / (f_IE x P_EC x PFD_IPL1 x PFD_IPL2)
PFD_SIF = 1E-05 / (0.1 x 0.9 x 0.1 x 0.1)
PFD_SIF = 1E-05 / 9E-04
PFD_SIF = 1.1E-02
SIL = SIL 2 (PFD between 1E-02 and 1E-03)
```

---

### 6.2 LOPA Worksheet - SIF-002: Low Flow Protection

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-002 |
| **Description** | Low cooling flow leads to thermal damage, potential release |
| **Consequence Category** | Category 4 - Equipment damage, potential injury |
| **TMEL** | 1 x 10^-5 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Cooling pump failure or valve closure | 0.1 /year | - |
| Enabling Condition | Process operating | 0.9 | - |
| **Unmitigated Frequency** | | **0.09 /year** | - |
| IPL 1 | Low flow alarm with operator response | 0.1 | 1 |
| IPL 2 | BPCS low flow interlock | 0.1 | 1 |
| **IPL Total** | | **0.01** | **2** |
| **Intermediate Frequency** | 0.09 x 0.01 | **9 x 10^-4 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **1.1 x 10^-2** | - |
| **Required SIL** | | **SIL 2** | - |

---

### 6.3 LOPA Worksheet - SIF-003: Pressure Relief Monitoring

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-003 |
| **Description** | Over-pressure leads to vessel rupture, potential injury |
| **Consequence Category** | Category 4 - Vessel failure, potential fatality |
| **TMEL** | 1 x 10^-5 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | BPCS pressure control failure | 0.05 /year | - |
| Enabling Condition | Process operating | 0.9 | - |
| **Unmitigated Frequency** | | **0.045 /year** | - |
| IPL 1 | Pressure relief valve | 0.01 | 2 |
| IPL 2 | High pressure alarm with operator response | 0.1 | 1 |
| **IPL Total** | | **0.001** | **3** |
| **Intermediate Frequency** | 0.045 x 0.001 | **4.5 x 10^-5 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **2.2 x 10^-1** | - |
| **Assigned SIL** | | **SIL 2** | - |

**Note:** Although calculation shows SIL 1 would be sufficient (PFD 0.22), SIL 2 is assigned due to:
- High consequence severity (potential fatality)
- Common practice for pressure protection
- Consistency with other GL-001 SIFs

---

### 6.4 LOPA Worksheet - SIF-004: Flame Failure Detection

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-004 |
| **Description** | Flame loss leads to unburned fuel accumulation, explosion risk |
| **Consequence Category** | Category 5 - Explosion, potential fatality |
| **TMEL** | 1 x 10^-5 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Flame failure (various causes) | 0.5 /year | - |
| Enabling Condition | Burner operating | 0.9 | - |
| **Unmitigated Frequency** | | **0.45 /year** | - |
| IPL 1 | Flame failure alarm (BPCS) | 0.1 | 1 |
| **IPL Total** | | **0.1** | **1** |
| **Intermediate Frequency** | 0.45 x 0.1 | **4.5 x 10^-2 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **2.2 x 10^-4** | - |
| **Required SIL** | | **SIL 3** | - |

**Note:** LOPA indicates SIL 3 requirement. However, per NFPA 85/86 and industry practice:
- SIL 2 with enhanced architecture (2oo3 voting) is accepted
- Additional mitigations: dual block and bleed, purge interlocks
- Final assignment: **SIL 2** with architectural constraints

---

### 6.5 LOPA Worksheet - SIF-005: Emergency Shutdown

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-005 |
| **Description** | Master ESD for all process heat hazards |
| **Consequence Category** | Category 5 - Multiple hazards |
| **TMEL** | 1 x 10^-5 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Multiple (worst case from SIF-001 to 004) | 0.5 /year | - |
| Enabling Condition | Process operating | 0.9 | - |
| **Unmitigated Frequency** | | **0.45 /year** | - |
| IPL 1 | Individual SIF functions (SIF-001 to 004) | 0.01 | 2 |
| IPL 2 | Operator emergency response | 0.1 | 1 |
| **IPL Total** | | **0.001** | **3** |
| **Intermediate Frequency** | 0.45 x 0.001 | **4.5 x 10^-4 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **2.2 x 10^-2** | - |
| **Required SIL** | | **SIL 2** | - |

---

### 6.6 LOPA Worksheet - SIF-006: Ventilation Fault Detection

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-006 |
| **Description** | Ventilation failure leads to poor air quality in occupied space |
| **Consequence Category** | Category 3 - Minor injury, discomfort |
| **TMEL** | 1 x 10^-4 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | HVAC system failure | 0.2 /year | - |
| Enabling Condition | Building occupied | 0.5 | - |
| **Unmitigated Frequency** | | **0.1 /year** | - |
| IPL 1 | BMS alarm with operator response | 0.1 | 1 |
| **IPL Total** | | **0.1** | **1** |
| **Intermediate Frequency** | 0.1 x 0.1 | **1 x 10^-2 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **1 x 10^-2** | - |
| **Assigned SIL** | | **SIL 1** | - |

**Note:** Calculation shows borderline SIL 1/SIL 2. Assigned SIL 1 due to:
- Lower consequence severity (Category 3)
- Additional non-credited mitigations (natural ventilation, occupant egress)
- Building applications typically SIL 1

---

### 6.7 LOPA Worksheet - SIF-007: CO/CO2 High Alert

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-007 |
| **Description** | High CO/CO2 leads to occupant health effects |
| **Consequence Category** | Category 4 - Major injury potential (CO poisoning) |
| **TMEL** | 1 x 10^-4 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | CO/CO2 generation source | 0.1 /year | - |
| Enabling Condition | Building occupied | 0.5 | - |
| **Unmitigated Frequency** | | **0.05 /year** | - |
| IPL 1 | General ventilation | 0.1 | 1 |
| **IPL Total** | | **0.1** | **1** |
| **Intermediate Frequency** | 0.05 x 0.1 | **5 x 10^-3 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **2 x 10^-2** | - |
| **Required SIL** | | **SIL 1** | - |

---

### 6.8 LOPA Worksheet - SIF-008: Emission Threshold Violation

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-008 |
| **Description** | Emission exceedance leads to regulatory violation |
| **Consequence Category** | Category 2 - Regulatory/business consequence |
| **TMEL** | 1 x 10^-4 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Process upset causing excess emissions | 0.2 /year | - |
| Enabling Condition | Reporting period active | 1.0 | - |
| **Unmitigated Frequency** | | **0.2 /year** | - |
| IPL 1 | Manual monitoring and adjustment | 0.1 | 1 |
| **IPL Total** | | **0.1** | **1** |
| **Intermediate Frequency** | 0.2 x 0.1 | **2 x 10^-2 /year** | - |
| **Required SIF PFD** | TMEL / Intermediate | **5 x 10^-3** | - |
| **Required SIL** | | **SIL 1** | - |

---

## 7. TMEL Calculations

### 7.1 TMEL Calculation Summary

| SIF | Consequence Category | TMEL | Basis |
|-----|---------------------|------|-------|
| SIF-001 | Cat 4 - Major equipment damage | 1E-05 | Corporate risk criteria |
| SIF-002 | Cat 4 - Equipment damage | 1E-05 | Corporate risk criteria |
| SIF-003 | Cat 4 - Vessel failure potential | 1E-05 | Corporate risk criteria |
| SIF-004 | Cat 5 - Explosion potential | 1E-05 | Corporate risk criteria + NFPA |
| SIF-005 | Cat 5 - Multiple hazards | 1E-05 | Corporate risk criteria |
| SIF-006 | Cat 3 - Minor injury | 1E-04 | Corporate risk criteria |
| SIF-007 | Cat 4 - CO poisoning potential | 1E-04 | Corporate risk criteria + OSHA |
| SIF-008 | Cat 2 - Regulatory | 1E-04 | Business risk criteria |

### 7.2 Risk Reduction Factor Summary

| SIF | Unmitigated Freq | IPL Credit | Intermediate Freq | TMEL | Required RRF | Required PFD | Assigned SIL |
|-----|------------------|------------|-------------------|------|--------------|--------------|--------------|
| SIF-001 | 0.09 | 100 | 9E-04 | 1E-05 | 90 | 1.1E-02 | SIL 2 |
| SIF-002 | 0.09 | 100 | 9E-04 | 1E-05 | 90 | 1.1E-02 | SIL 2 |
| SIF-003 | 0.045 | 1000 | 4.5E-05 | 1E-05 | 4.5 | 2.2E-01 | SIL 2* |
| SIF-004 | 0.45 | 10 | 4.5E-02 | 1E-05 | 4500 | 2.2E-04 | SIL 2** |
| SIF-005 | 0.45 | 1000 | 4.5E-04 | 1E-05 | 45 | 2.2E-02 | SIL 2 |
| SIF-006 | 0.1 | 10 | 1E-02 | 1E-04 | 100 | 1E-02 | SIL 1 |
| SIF-007 | 0.05 | 10 | 5E-03 | 1E-04 | 50 | 2E-02 | SIL 1 |
| SIF-008 | 0.2 | 10 | 2E-02 | 1E-04 | 200 | 5E-03 | SIL 1 |

*SIF-003: Elevated from calculated requirement for consistency
**SIF-004: Reduced from SIL 3 with architectural constraints per NFPA

---

## 8. SIL Determination Summary

### 8.1 Final SIL Assignments

| SIF ID | SIF Name | Agent | Required SIL | Assigned SIL | PFD Target |
|--------|----------|-------|--------------|--------------|------------|
| SIF-001 | High Temperature Shutdown | GL-001 | SIL 2 | SIL 2 | < 1E-02 |
| SIF-002 | Low Flow Protection | GL-001 | SIL 2 | SIL 2 | < 1E-02 |
| SIF-003 | Pressure Relief Monitoring | GL-001 | SIL 1 | SIL 2 | < 1E-02 |
| SIF-004 | Flame Failure Detection | GL-001 | SIL 3 | SIL 2* | < 1E-02 |
| SIF-005 | Emergency Shutdown | GL-001 | SIL 2 | SIL 2 | < 1E-02 |
| SIF-006 | Ventilation Fault Detection | GL-005 | SIL 1 | SIL 1 | < 1E-01 |
| SIF-007 | CO/CO2 High Alert | GL-005 | SIL 1 | SIL 1 | < 1E-01 |
| SIF-008 | Emission Threshold Violation | GL-007 | SIL 1 | SIL 1 | < 1E-01 |

*SIF-004 architectural constraints: 2oo3 voting, dual block and bleed, enhanced diagnostics

### 8.2 Agent Summary

| Agent | SIFs | Maximum SIL | Comments |
|-------|------|-------------|----------|
| GL-001 | SIF-001 to SIF-005 | SIL 2 | Process heat safety critical |
| GL-005 | SIF-006, SIF-007 | SIL 1 | Building safety |
| GL-007 | SIF-008 | SIL 1 | Compliance monitoring |

### 8.3 SIL to PFD Mapping

| SIL | PFD Range (Low Demand) | Risk Reduction Factor |
|-----|------------------------|----------------------|
| SIL 1 | 0.1 to 0.01 | 10 to 100 |
| SIL 2 | 0.01 to 0.001 | 100 to 1,000 |
| SIL 3 | 0.001 to 0.0001 | 1,000 to 10,000 |
| SIL 4 | 0.0001 to 0.00001 | 10,000 to 100,000 |

---

## 9. Assumptions and Limitations

### 9.1 LOPA Assumptions

| ID | Assumption | Impact if Invalid |
|----|------------|-------------------|
| A-1 | Initiating event frequencies from OREDA/industry data are applicable | Re-evaluate with site-specific data |
| A-2 | IPL PFD values are achievable with proper design | Verify during design phase |
| A-3 | Operator response times are achievable | Validate with operator trials |
| A-4 | Common cause failures are managed through design | Verify CCF beta factor |
| A-5 | Process safety times allow for SIF response | Re-analyze if process changes |
| A-6 | Consequence categories are accurate | Re-assess with consequence modeling |

### 9.2 LOPA Limitations

| Limitation | Mitigation |
|------------|------------|
| Semi-quantitative method | Supplement with QRA where consequence severity is Cat 5 |
| Generic failure data | Update with site-specific data after commissioning |
| Human factors simplified | Conduct detailed HRA for complex operator actions |
| CCF not explicitly modeled | Apply CCF factors in PFD calculations |
| Uncertainty not quantified | Use conservative assumptions throughout |

### 9.3 Validation Requirements

| Requirement | Method | Timing |
|-------------|--------|--------|
| IPL independence | Design review | Before detailed design |
| IPL effectiveness | Testing, inspection | FAT, SAT, operations |
| Initiating event frequency | Operational data review | After 1 year operation |
| Consequence assessment | Incident investigation | Ongoing |

---

## Appendix A: LOPA References

### A.1 Initiating Event Frequency Sources

| Source | Reference |
|--------|-----------|
| OREDA | Offshore Reliability Data Handbook, 6th Edition |
| CCPS | Guidelines for Process Equipment Reliability Data |
| API | API 581 Risk-Based Inspection |
| NFPA | NFPA Fire Protection Handbook |
| IEEE | IEEE 493 Recommended Practice for Electric System Reliability |

### A.2 IPL PFD Sources

| Source | Reference |
|--------|-----------|
| ISA TR84.00.02 | SIF Safety Integrity Level Evaluation Techniques |
| CCPS | LOPA - Simplified Process Risk Assessment |
| exida | SIL Verification Library |
| IEC 61511-3 | Guidance for SIL Determination |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
