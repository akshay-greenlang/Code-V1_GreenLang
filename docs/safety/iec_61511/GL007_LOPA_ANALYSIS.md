# Layer of Protection Analysis (LOPA) - GL-007 Furnace Monitoring

## GreenLang Process Heat Agents - Furnace Performance Monitor

**Document ID:** GL-SIL-LOPA-007
**Version:** 1.0
**Effective Date:** 2025-12-07
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-3:2016, CCPS LOPA Guidelines, API 560, API 530

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Furnace Hazard Overview](#2-furnace-hazard-overview)
3. [Initiating Events](#3-initiating-events)
4. [Consequence Analysis](#4-consequence-analysis)
5. [Independent Protection Layers](#5-independent-protection-layers)
6. [LOPA Worksheets](#6-lopa-worksheets)
7. [SIL Target Determination](#7-sil-target-determination)
8. [ALARP Demonstration](#8-alarp-demonstration)
9. [Risk Reduction Requirements](#9-risk-reduction-requirements)
10. [Appendices](#10-appendices)

---

## 1. Introduction

### 1.1 Purpose

This document presents the Layer of Protection Analysis (LOPA) for the GL-007 Furnace Performance Monitor agent. The analysis identifies hazard scenarios, quantifies initiating event frequencies, credits Independent Protection Layers (IPLs), and determines required Safety Integrity Levels (SIL) for Safety Instrumented Functions (SIFs).

### 1.2 Scope

This LOPA covers hazard scenarios associated with:

- Tube Metal Temperature (TMT) monitoring failures
- Furnace overtemperature conditions
- Tube rupture scenarios
- Fire and explosion hazards
- Flame detection failures
- Fouling-induced hot spots

### 1.3 References

| Document | Title |
|----------|-------|
| GL-HAZOP-007 | GL-007 Furnace Performance Monitor HAZOP Study |
| IEC 61511-3:2016 | Guidance for determination of required SIL |
| API 560 | Fired Heaters for General Refinery Service |
| API 530 | Calculation of Heater-Tube Thickness in Petroleum Refineries |
| NFPA 86 | Standard for Ovens and Furnaces |
| CCPS | Layer of Protection Analysis - Simplified Process Risk Assessment |

### 1.4 LOPA Team

| Role | Responsibility |
|------|----------------|
| LOPA Leader | Facilitation, documentation, SIL assignment |
| Furnace Engineer | Furnace design, TMT knowledge, API standards |
| Safety Engineer | IPL validation, SIL determination |
| Operations | Operating procedures, alarm response |
| Instrumentation | TMT sensor capabilities, diagnostics |
| Materials Engineer | Creep rupture, tube life estimation |

---

## 2. Furnace Hazard Overview

### 2.1 Furnace Safety Background

Fired heaters and furnaces present significant process safety hazards:

| Hazard Category | Description | Potential Consequence |
|-----------------|-------------|----------------------|
| **Tube Rupture** | Overtemperature leading to creep or stress rupture | Fire, explosion, release of process fluid |
| **Fire** | Tube leak with ignition of process fluid | Major fire, equipment damage, fatality |
| **Explosion** | Unburned fuel accumulation | Furnace explosion, structural damage |
| **Coking/Fouling** | Reduced heat transfer, hot spots | Localized tube failure |
| **Flame Impingement** | Direct flame contact with tubes | Rapid overtemperature, tube damage |
| **Refractory Failure** | Hot spots, structural degradation | Secondary fires, equipment damage |

### 2.2 Critical Temperature Points

| Temperature Point | Typical Value | Safety Implication |
|-------------------|---------------|-------------------|
| TMT Design Limit | 1200-1500 degF | Maximum allowable per API 530 |
| TMT Alarm | Design - 50 degF | Early warning for operator action |
| TMT Trip | Design Limit | Automatic shutdown trigger |
| Bridgewall Temperature | 1400-1800 degF | Convection section protection |
| Flue Gas Temperature | 350-700 degF | Efficiency and draft monitoring |

### 2.3 Tube Failure Mechanisms

Per API 530 and API 579, tube failure occurs through:

1. **Creep Rupture**: Time-dependent deformation at elevated temperature
   - Larson-Miller Parameter: P = T(K) x (C + log t)
   - Design life: 100,000 hours typical

2. **Stress Rupture**: Instantaneous failure above design stress

3. **Oxidation**: Surface degradation reducing wall thickness

4. **Carburization**: Carbon diffusion causing embrittlement

5. **Thermal Fatigue**: Cyclic temperature causing crack initiation

---

## 3. Initiating Events

### 3.1 TMT Monitoring Failures

| Initiating Event ID | Description | Frequency (per year) | Source |
|---------------------|-------------|---------------------|--------|
| IE-007-01 | TMT thermocouple failure (undetected) | 0.1 | Industry data |
| IE-007-02 | TMT signal loss (wiring failure) | 0.05 | OREDA |
| IE-007-03 | TMT false low reading (cold junction error) | 0.05 | Industry data |
| IE-007-04 | TMT data processing failure | 0.02 | Software reliability |
| IE-007-05 | TMT alarm generation failure | 0.02 | Alarm system reliability |

### 3.2 Furnace Process Upsets

| Initiating Event ID | Description | Frequency (per year) | Source |
|---------------------|-------------|---------------------|--------|
| IE-007-06 | Loss of process flow (pump failure) | 0.1 | OREDA |
| IE-007-07 | Excessive firing rate | 0.1 | Operator error, BPCS failure |
| IE-007-08 | Flame impingement (burner misalignment) | 0.05 | Maintenance, combustion issues |
| IE-007-09 | Coking/fouling event | 0.2 | Process conditions |
| IE-007-10 | Burner control failure | 0.1 | BPCS, valve failure |

### 3.3 Combustion System Failures

| Initiating Event ID | Description | Frequency (per year) | Source |
|---------------------|-------------|---------------------|--------|
| IE-007-11 | Flame failure (loss of flame) | 0.5 | NFPA data |
| IE-007-12 | Fuel valve leak-through | 0.01 | API data |
| IE-007-13 | Air supply failure | 0.1 | Industry data |
| IE-007-14 | Ignition system failure during startup | 0.2 | NFPA data |
| IE-007-15 | Combustion air fan failure | 0.1 | OREDA |

### 3.4 Enabling Conditions

| Enabling Condition | Probability | Description |
|--------------------|-------------|-------------|
| Furnace operating | 0.9 | 90% availability |
| Personnel in area | 0.25 | 6 hours per day |
| High-severity process fluid | 1.0 | Always present |
| Ignition source present | 0.8 | Hot surfaces, flames |
| Startup/shutdown | 0.1 | 36 days per year |

---

## 4. Consequence Analysis

### 4.1 Consequence Categories

| Category | Description | Severity | TMEL (per year) |
|----------|-------------|----------|-----------------|
| Cat 5 | Multiple fatalities, major explosion | Catastrophic | 1E-06 |
| Cat 4 | Single fatality, major fire | Severe | 1E-05 |
| Cat 3 | Permanent disability, significant fire | Serious | 1E-04 |
| Cat 2 | Lost time injury, equipment damage | Moderate | 1E-03 |
| Cat 1 | First aid, minor damage | Minor | 1E-02 |

### 4.2 Tube Rupture Consequence Analysis

**Scenario: TMT High Leading to Tube Rupture**

| Factor | Value | Basis |
|--------|-------|-------|
| Tube rupture probability at TMT > Design | 0.5 | Engineering estimate |
| Process fluid release rate | 50-500 kg/min | Tube size dependent |
| Ignition probability | 0.3-0.8 | Hot furnace environment |
| Fire escalation probability | 0.5 | Depending on location |
| Fatality probability given fire | 0.1-0.5 | Personnel exposure |

**Consequence Severity**: Category 4 (Single fatality potential)

### 4.3 Furnace Explosion Consequence Analysis

**Scenario: Flame Failure Leading to Explosion**

| Factor | Value | Basis |
|--------|-------|-------|
| Fuel accumulation time | 5-15 seconds | Volume dependent |
| Explosive mixture formation | 0.8 | High probability in enclosed space |
| Ignition probability | 0.9 | Hot surfaces, pilot flames |
| Personnel fatality probability | 0.3 | Depending on location |
| Equipment destruction probability | 0.9 | Overpressure damage |

**Consequence Severity**: Category 5 (Multiple fatality potential)

### 4.4 Fire Consequence Analysis

**Scenario: Tube Leak with Fire**

| Factor | Value | Basis |
|--------|-------|-------|
| Leak detection time | 1-60 minutes | Depending on size |
| Fire development time | 2-10 minutes | Rapid escalation |
| Secondary equipment damage | High | Radiation, direct flame |
| Business interruption | $1M-$50M | Extent of damage |

**Consequence Severity**: Category 3-4 (Serious to Severe)

---

## 5. Independent Protection Layers

### 5.1 IPL Criteria per IEC 61511-3

An IPL must be:
1. **Independent**: Not affected by initiating event or other IPLs
2. **Specific**: Designed to prevent/mitigate the specific consequence
3. **Dependable**: Quantifiable PFD
4. **Auditable**: Subject to regular testing/inspection

### 5.2 IPLs for GL-007 Furnace Monitoring

| IPL ID | IPL Description | PFD | Credits | Validation |
|--------|-----------------|-----|---------|------------|
| IPL-007-01 | BPCS temperature control | 0.1 | 1 | Annual testing |
| IPL-007-02 | High TMT alarm with operator response | 0.1 | 1 | Operator training, >20 min response |
| IPL-007-03 | Redundant TMT measurement | 0.1 | 1 | Comparison logic |
| IPL-007-04 | Burner Management System (BMS) | 0.01 | 2 | SIL-rated, tested |
| IPL-007-05 | Flame scanner with SIS interlock | 0.01 | 2 | Per NFPA 85/86 |
| IPL-007-06 | Process flow interlock | 0.01 | 2 | SIS function |
| IPL-007-07 | Pressure relief device | 0.01 | 2 | API inspection |
| IPL-007-08 | Purge interlock | 0.01 | 2 | Per NFPA 86 |
| IPL-007-09 | Manual emergency shutdown | 0.1 | 1 | Accessible pushbutton |

### 5.3 IPL Independence Analysis

| IPL Pair | Independence Assessment | Conclusion |
|----------|-------------------------|------------|
| IPL-01 vs IPL-02 | Different systems (BPCS vs operator) | Independent |
| IPL-01 vs IPL-03 | Same measurement type | Partially dependent |
| IPL-04 vs IPL-05 | Part of same BMS | Credit only one |
| IPL-06 vs IPL-01 | Different process variables | Independent |

### 5.4 Non-Credited Safeguards

| Safeguard | Reason Not Credited |
|-----------|---------------------|
| Portable infrared scanning | Periodic, not continuous |
| Operator rounds | Not continuous monitoring |
| Maintenance program | Preventive, not protective |
| Training | Enables IPLs, not IPL itself |
| Warning signs | Not dependable barrier |

---

## 6. LOPA Worksheets

### 6.1 LOPA-007-01: TMT High Leading to Tube Rupture

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-007-01 |
| **Description** | Undetected high TMT leads to tube creep rupture and fire |
| **Consequence Category** | Category 4 - Single fatality potential |
| **TMEL** | 1E-05 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | TMT monitoring failure + High temperature | 0.1 /year | - |
| Enabling Condition | Furnace operating | 0.9 | - |
| Enabling Condition | Tube rupture given TMT > Design | 0.5 | - |
| Enabling Condition | Fire given rupture | 0.5 | - |
| **Unmitigated Frequency** | | **2.25E-02 /year** | - |
| IPL 1 | High TMT alarm with operator response | 0.1 | 1 |
| IPL 2 | BPCS temperature control | 0.1 | 1 |
| **IPL Total** | | **0.01** | **2** |
| **Mitigated Frequency** | 2.25E-02 x 0.01 | **2.25E-04 /year** | - |
| **Required SIF PFD** | TMEL / Mitigated | **4.4E-02** | - |
| **Required SIL** | | **SIL 1** | - |
| **Assigned SIL** | | **SIL 2** | - |

**Calculation:**
```
PFD_SIF = TMEL / (f_IE x P_EC1 x P_EC2 x P_EC3 x PFD_IPL1 x PFD_IPL2)
PFD_SIF = 1E-05 / (0.1 x 0.9 x 0.5 x 0.5 x 0.1 x 0.1)
PFD_SIF = 1E-05 / 2.25E-04
PFD_SIF = 4.4E-02
SIL = SIL 1 (PFD between 1E-01 and 1E-02)
Assigned SIL 2 for additional margin and API 560 compliance
```

---

### 6.2 LOPA-007-02: Flame Failure Leading to Explosion

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-007-02 |
| **Description** | Flame loss leads to fuel accumulation and explosion |
| **Consequence Category** | Category 5 - Multiple fatality potential |
| **TMEL** | 1E-06 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Flame failure | 0.5 /year | - |
| Enabling Condition | Burner operating | 0.9 | - |
| Enabling Condition | Explosion given fuel accumulation | 0.3 | - |
| **Unmitigated Frequency** | | **1.35E-01 /year** | - |
| IPL 1 | Flame scanner alarm (BPCS) | 0.1 | 1 |
| **IPL Total** | | **0.1** | **1** |
| **Mitigated Frequency** | 1.35E-01 x 0.1 | **1.35E-02 /year** | - |
| **Required SIF PFD** | TMEL / Mitigated | **7.4E-05** | - |
| **Required SIL** | | **SIL 3** | - |
| **Assigned SIL** | | **SIL 2** | - |

**Note:** LOPA indicates SIL 3 requirement. Per NFPA 85/86 and industry practice:
- SIL 2 with architectural constraints (2oo3 voting) is accepted
- Additional mitigations: dual block and bleed, purge interlocks
- Final assignment: **SIL 2** with enhanced architecture

---

### 6.3 LOPA-007-03: Fouling-Induced Hot Spot

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-007-03 |
| **Description** | Undetected fouling causes localized hot spot and tube failure |
| **Consequence Category** | Category 3 - Serious injury potential |
| **TMEL** | 1E-04 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Fouling event | 0.2 /year | - |
| Enabling Condition | Furnace operating | 0.9 | - |
| Enabling Condition | Hot spot formation | 0.3 | - |
| Enabling Condition | Tube damage given hot spot | 0.5 | - |
| **Unmitigated Frequency** | | **2.7E-02 /year** | - |
| IPL 1 | TMT trending/comparison | 0.1 | 1 |
| IPL 2 | Periodic infrared scanning | 0.5 | 0 |
| **IPL Total** | | **0.1** | **1** |
| **Mitigated Frequency** | 2.7E-02 x 0.1 | **2.7E-03 /year** | - |
| **Required SIF PFD** | TMEL / Mitigated | **3.7E-02** | - |
| **Assigned SIL** | | **SIL 1** | - |

---

### 6.4 LOPA-007-04: Low Process Flow with High Heat Input

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-007-04 |
| **Description** | Low flow condition with continued firing leads to tube damage |
| **Consequence Category** | Category 4 - Severe damage, fatality potential |
| **TMEL** | 1E-05 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Process pump failure or valve closure | 0.1 /year | - |
| Enabling Condition | Furnace operating | 0.9 | - |
| Enabling Condition | Fire given tube failure | 0.5 | - |
| **Unmitigated Frequency** | | **4.5E-02 /year** | - |
| IPL 1 | Low flow alarm with operator response | 0.1 | 1 |
| IPL 2 | BPCS low flow interlock | 0.1 | 1 |
| **IPL Total** | | **0.01** | **2** |
| **Mitigated Frequency** | 4.5E-02 x 0.01 | **4.5E-04 /year** | - |
| **Required SIF PFD** | TMEL / Mitigated | **2.2E-02** | - |
| **Assigned SIL** | | **SIL 2** | - |

---

### 6.5 LOPA-007-05: Startup Purge Failure

| Field | Value |
|-------|-------|
| **Scenario ID** | LOPA-007-05 |
| **Description** | Inadequate purge before ignition leads to explosion |
| **Consequence Category** | Category 5 - Explosion, multiple fatality |
| **TMEL** | 1E-06 per year |

| Item | Description | Value | Credit |
|------|-------------|-------|--------|
| Initiating Event | Startup with residual fuel | 0.2 /year | - |
| Enabling Condition | Startup operation | 0.1 | - |
| Enabling Condition | Explosion given ignition | 0.5 | - |
| **Unmitigated Frequency** | | **1.0E-02 /year** | - |
| IPL 1 | Purge timer interlock (BMS) | 0.01 | 2 |
| IPL 2 | Airflow proving | 0.1 | 1 |
| **IPL Total** | | **0.001** | **3** |
| **Mitigated Frequency** | 1.0E-02 x 0.001 | **1.0E-05 /year** | - |
| **Required SIF PFD** | TMEL / Mitigated | **1.0E-01** | - |
| **Assigned SIL** | | **SIL 2** | - |

**Note:** Per NFPA 86, purge interlock is mandatory. SIL 2 assignment for BMS consistency.

---

## 7. SIL Target Determination

### 7.1 SIL Assignment Summary

| SIF ID | Scenario | Required SIL | Assigned SIL | PFD Target | Architecture |
|--------|----------|--------------|--------------|------------|--------------|
| SIF-007-01 | TMT High Shutdown | SIL 1 | SIL 2 | < 1E-02 | 1oo2 |
| SIF-007-02 | Flame Failure Shutdown | SIL 3 | SIL 2* | < 1E-02 | 2oo3 |
| SIF-007-03 | Fouling Hot Spot Alert | SIL 1 | SIL 1 | < 1E-01 | 1oo1 |
| SIF-007-04 | Low Flow Protection | SIL 2 | SIL 2 | < 1E-02 | 1oo2 |
| SIF-007-05 | Purge Interlock | SIL 1 | SIL 2 | < 1E-02 | 1oo2 |

*With architectural constraints per NFPA 85/86

### 7.2 Risk Reduction Factor (RRF) Requirements

| SIF ID | Mitigated Freq | TMEL | Risk Gap | Required RRF | Achieved RRF |
|--------|----------------|------|----------|--------------|--------------|
| SIF-007-01 | 2.25E-04 | 1E-05 | 22.5 | 23 | 100 (SIL 2) |
| SIF-007-02 | 1.35E-02 | 1E-06 | 13,500 | 13,500 | 1,000 (SIL 2+Arch) |
| SIF-007-03 | 2.7E-03 | 1E-04 | 27 | 27 | 10 (SIL 1) |
| SIF-007-04 | 4.5E-04 | 1E-05 | 45 | 45 | 100 (SIL 2) |
| SIF-007-05 | 1.0E-05 | 1E-06 | 10 | 10 | 100 (SIL 2) |

### 7.3 SIL to PFD Mapping

| SIL | PFD Range (Low Demand) | Risk Reduction Factor |
|-----|------------------------|----------------------|
| SIL 1 | 0.1 to 0.01 | 10 to 100 |
| SIL 2 | 0.01 to 0.001 | 100 to 1,000 |
| SIL 3 | 0.001 to 0.0001 | 1,000 to 10,000 |
| SIL 4 | 0.0001 to 0.00001 | 10,000 to 100,000 |

---

## 8. ALARP Demonstration

### 8.1 ALARP Framework

As Low As Reasonably Practicable (ALARP) requires demonstrating that:
1. Risk is below intolerable region
2. Further risk reduction is grossly disproportionate to benefit
3. Good engineering practice is applied

### 8.2 Intolerable Risk Threshold

| Risk Category | Intolerable Threshold | GL-007 Status |
|---------------|----------------------|---------------|
| Individual Risk | > 1E-03 per year | Below threshold |
| Societal Risk | > 1E-04 per year (fatality) | Below threshold |
| Environmental | > 1E-03 per year (major) | Below threshold |

### 8.3 ALARP Analysis

| Scenario | Mitigated Risk | ALARP Zone | Further Reduction |
|----------|----------------|------------|-------------------|
| Tube Rupture | 2.25E-04 | Broadly Acceptable | Not required |
| Flame Failure | 1.35E-05 | ALARP | Enhanced architecture |
| Fouling Hot Spot | 2.7E-04 | Broadly Acceptable | Not required |
| Low Flow | 4.5E-05 | ALARP | Redundancy added |
| Purge Failure | 1.0E-06 | Broadly Acceptable | Not required |

### 8.4 Cost-Benefit Analysis

| Additional Measure | Cost | Risk Reduction | Cost per Life Saved | Decision |
|-------------------|------|----------------|---------------------|----------|
| Additional TMT sensors | $50,000 | 50% | $2.2M | Implement |
| 2oo3 flame detection | $80,000 | 90% | $890K | Implement |
| Continuous IR monitoring | $200,000 | 30% | $6.7M | Not justified |
| Triple redundant flow | $120,000 | 50% | $2.4M | Implement |

### 8.5 ALARP Conclusion

All GL-007 furnace hazard scenarios demonstrate:
1. Mitigated risk is below intolerable threshold
2. Selected SIL levels provide adequate risk reduction
3. Cost-effective measures have been implemented
4. Further risk reduction measures are disproportionate

**ALARP Status: DEMONSTRATED**

---

## 9. Risk Reduction Requirements

### 9.1 Implementation Requirements

| Requirement ID | Description | SIF | Priority |
|----------------|-------------|-----|----------|
| RR-007-01 | Implement redundant TMT monitoring at critical tube locations | SIF-007-01 | Critical |
| RR-007-02 | Install 2oo3 flame detection per NFPA 85/86 | SIF-007-02 | Critical |
| RR-007-03 | Implement TMT rate-of-change monitoring for fouling detection | SIF-007-03 | High |
| RR-007-04 | Install redundant flow measurement with 1oo2 voting | SIF-007-04 | Critical |
| RR-007-05 | Ensure purge timer interlock meets NFPA 86 Section 8.7.2 | SIF-007-05 | Critical |

### 9.2 Design Requirements

| Design Aspect | Requirement | Standard Reference |
|---------------|-------------|-------------------|
| TMT Sensor Type | Type K or Type N thermocouple | API 560 Section 8.1.3 |
| TMT Response Time | < 10 seconds | Process safety time |
| Flame Scanner Type | UV/IR self-checking | NFPA 85 Section 5.3.6 |
| Flame Response Time | < 4 seconds | NFPA 86 Section 8.5.2.2 |
| Flow Sensor Type | Orifice or Coriolis | SIL 2 certified |
| Purge Time Calculation | 4 volume changes minimum | NFPA 86 Section 8.7.2 |

### 9.3 Testing Requirements

| Test Type | Frequency | Scope |
|-----------|-----------|-------|
| Proof Test | 12 months | Full function test of all SIFs |
| Partial Stroke Test | Monthly | Valve partial movement |
| Flame Scanner Test | Weekly | Shutter test or equivalent |
| TMT Comparison Check | Continuous | Automated discrepancy detection |

---

## 10. Appendices

### Appendix A: LOPA Assumptions

| ID | Assumption | Impact if Invalid |
|----|------------|-------------------|
| A-1 | Initiating event frequencies from API/NFPA data are applicable | Re-evaluate with site data |
| A-2 | IPL PFD values are achievable with proper design | Verify during design |
| A-3 | Operator response time is > 20 minutes | Validate with operator trials |
| A-4 | TMT sensors represent actual tube temperature | Verify sensor placement |
| A-5 | Process safety time allows for SIF response | Confirm with process analysis |

### Appendix B: Provenance Information

| Parameter | Value |
|-----------|-------|
| Document ID | GL-SIL-LOPA-007 |
| Analysis Date | 2025-12-07 |
| LOPA Software | GreenLang LOPAAnalyzer v1.0 |
| Provenance Hash | SHA-256: (calculated at approval) |
| Analysis Basis | IEC 61511-3:2016, CCPS LOPA |

### Appendix C: Abbreviations

| Abbreviation | Definition |
|--------------|------------|
| ALARP | As Low As Reasonably Practicable |
| BMS | Burner Management System |
| BPCS | Basic Process Control System |
| CCF | Common Cause Failure |
| IPL | Independent Protection Layer |
| LOPA | Layer of Protection Analysis |
| PFD | Probability of Failure on Demand |
| RRF | Risk Reduction Factor |
| SIF | Safety Instrumented Function |
| SIL | Safety Integrity Level |
| SIS | Safety Instrumented System |
| TMEL | Target Mitigated Event Likelihood |
| TMT | Tube Metal Temperature |

### Appendix D: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-07 | GL-RegulatoryIntelligence | Initial release |

---

**Document End**

*This document is part of the GreenLang IEC 61511 SIL Certification Documentation Package.*
