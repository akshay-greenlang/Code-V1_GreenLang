# IEC 61511 Safety Integrity Level (SIL) Rating Documentation

## GL-001 ThermalCommand - Functional Safety Assessment

**Document Version:** 1.0.0
**Last Updated:** 2024-01-15
**Standard:** IEC 61511:2016 - Functional Safety: Safety Instrumented Systems for the Process Industry Sector

---

## 1. Executive Summary

This document defines the Safety Integrity Level (SIL) ratings for safety-related functions within GL-001 ThermalCommand and its coordinated agents. All safety instrumented functions (SIFs) are designed to meet IEC 61511 requirements.

## 2. Applicable Standards

| Standard | Title | Application |
|----------|-------|-------------|
| IEC 61511-1 | Framework, Definitions, System/Hardware/Software Requirements | Primary standard |
| IEC 61511-2 | Guidelines for the Application of IEC 61511-1 | Implementation guidance |
| IEC 61511-3 | Guidance for Determination of Safety Integrity Levels | SIL determination |
| IEC 61508 | Functional Safety of E/E/PE Systems | Referenced for components |

## 3. SIL Definitions

| SIL | Probability of Failure on Demand (PFD_avg) | Risk Reduction Factor |
|-----|---------------------------------------------|----------------------|
| SIL 4 | ≥10⁻⁵ to <10⁻⁴ | 100,000 to 10,000 |
| SIL 3 | ≥10⁻⁴ to <10⁻³ | 10,000 to 1,000 |
| SIL 2 | ≥10⁻³ to <10⁻² | 1,000 to 100 |
| SIL 1 | ≥10⁻² to <10⁻¹ | 100 to 10 |

## 4. Safety Instrumented Functions (SIFs) in GL-001

### 4.1 Boiler High Pressure Trip (SIF-001)

**Description:** Trip fuel supply when drum pressure exceeds high-high limit

| Parameter | Value |
|-----------|-------|
| **SIF ID** | SIF-001 |
| **SIL Target** | SIL 2 |
| **PFD_avg Target** | <10⁻² |
| **Demand Mode** | Low demand |
| **Response Time** | ≤1 second |
| **Coordinating Agent** | GL-002 FlameGuard |

**Architecture:** 1oo2 (1 out of 2) pressure transmitters

**Calculation:**
```
PFD_sys = 2 × PFD_sensor × PFD_logic × PFD_final_element
PFD_sys = 2 × (0.005) × (0.001) × (0.01) = 1×10⁻⁴ (SIL 2 achieved)
```

### 4.2 Flame Failure Shutdown (SIF-002)

**Description:** Close fuel valves on loss of flame detection

| Parameter | Value |
|-----------|-------|
| **SIF ID** | SIF-002 |
| **SIL Target** | SIL 3 |
| **PFD_avg Target** | <10⁻³ |
| **Demand Mode** | Low demand |
| **Response Time** | ≤4 seconds (per NFPA 85) |
| **Coordinating Agent** | GL-002 FlameGuard |

**Architecture:** 2oo3 (2 out of 3) flame detectors

**NFPA 85 Alignment:**
- Section 5.3.7.1: Flame failure response ≤4 seconds
- Section 5.3.3: Minimum of 2 flame detectors required

### 4.3 High Furnace Pressure Trip (SIF-003)

**Description:** Trip forced draft fans on high furnace pressure

| Parameter | Value |
|-----------|-------|
| **SIF ID** | SIF-003 |
| **SIL Target** | SIL 2 |
| **PFD_avg Target** | <10⁻² |
| **Coordinating Agent** | GL-007 FurnacePulse |

**Architecture:** 1oo2 pressure transmitters with 2oo2 logic for trip

### 4.4 Low Combustion Air Flow Trip (SIF-004)

**Description:** Trip fuel on loss of combustion air

| Parameter | Value |
|-----------|-------|
| **SIF ID** | SIF-004 |
| **SIL Target** | SIL 2 |
| **PFD_avg Target** | <10⁻² |
| **Response Time** | ≤2 seconds |
| **Coordinating Agent** | GL-005 CombustionSense |

### 4.5 High O₂ Limit Trip (SIF-005)

**Description:** Reduce firing on high excess oxygen (fuel-rich protection)

| Parameter | Value |
|-----------|-------|
| **SIF ID** | SIF-005 |
| **SIL Target** | SIL 1 |
| **PFD_avg Target** | <10⁻¹ |
| **Coordinating Agent** | GL-005 CombustionSense |

### 4.6 Steam Drum Low Level Trip (SIF-006)

**Description:** Trip boiler on low-low drum level

| Parameter | Value |
|-----------|-------|
| **SIF ID** | SIF-006 |
| **SIL Target** | SIL 3 |
| **PFD_avg Target** | <10⁻³ |
| **Coordinating Agent** | GL-003 UnifiedSteam |

**Architecture:** 2oo3 level transmitters (ASME requirement)

### 4.7 Emission Limit Exceedance Trip (SIF-007)

**Description:** Reduce load on emission limit approach

| Parameter | Value |
|-----------|-------|
| **SIF ID** | SIF-007 |
| **SIL Target** | SIL 1 |
| **PFD_avg Target** | <10⁻¹ |
| **Coordinating Agent** | GL-010 EmissionGuardian |

**Note:** Regulatory compliance function, not safety critical

## 5. SIL Verification Methods

### 5.1 Probability of Failure on Demand (PFD) Calculation

For each SIF, PFD is calculated using:

```
PFD_avg = (λ_DU × T_proof) / 2

Where:
- λ_DU = Dangerous undetected failure rate
- T_proof = Proof test interval
```

### 5.2 Architectural Constraints

| SIL | Max Hardware Fault Tolerance | Min SFF Required |
|-----|------------------------------|------------------|
| SIL 1 | HFT 0 | 60% |
| SIL 2 | HFT 1 | 90% |
| SIL 3 | HFT 1 | 99% |
| SIL 4 | HFT 2 | 99% |

**SFF** = Safe Failure Fraction

### 5.3 Proof Test Requirements

| SIF ID | Proof Test Interval | Test Procedure |
|--------|---------------------|----------------|
| SIF-001 | 12 months | Pressure transmitter calibration, logic test |
| SIF-002 | 6 months | Flame detector sensitivity, response time |
| SIF-003 | 12 months | Pressure transmitter, damper stroke test |
| SIF-004 | 12 months | Flow transmitter, fan interlock test |
| SIF-005 | 12 months | O₂ analyzer calibration |
| SIF-006 | 6 months | Level transmitter comparison, trip test |

## 6. Software Safety Requirements

### 6.1 GL-001 Software Classification

Per IEC 61511-1 Clause 11.5:

| Aspect | Requirement | GL-001 Implementation |
|--------|-------------|----------------------|
| Software lifecycle | Defined phases | Yes - documented |
| Verification | Independent review | Yes - code review |
| Configuration management | Version control | Yes - Git |
| Modification procedures | Documented | Yes - MOC process |

### 6.2 Safety-Critical Code Isolation

Safety-related calculations are isolated in the `safety/` module:
- `safety/boundary_engine.py` - Operating limit enforcement
- `safety/sis_integration.py` - SIS communication interface
- `safety/interlock_logic.py` - Software interlock implementation

### 6.3 Deterministic Execution

All safety calculations use:
- `Decimal` arithmetic (no floating-point errors)
- Defined precision (4 decimal places)
- Documented rounding (`ROUND_HALF_UP`)

## 7. Agent-Specific SIL Assignments

| Agent | Safety Functions | Max SIL |
|-------|-----------------|---------|
| GL-001 ThermalCommand | Orchestration, SIF coordination | SIL 2 |
| GL-002 FlameGuard | Burner safety, flame failure | SIL 3 |
| GL-003 UnifiedSteam | Drum level protection | SIL 3 |
| GL-004 BurnMaster | Air/fuel ratio limits | SIL 2 |
| GL-005 CombustionSense | O₂/CO limits, flame stability | SIL 2 |
| GL-006 HeatReclaim | Temperature limits | SIL 1 |
| GL-007 FurnacePulse | Furnace pressure, atmosphere | SIL 2 |
| GL-008 TrapCatcher | None (diagnostic only) | N/A |
| GL-009 ThermalIQ | None (analytical only) | N/A |
| GL-010 EmissionGuardian | Emission limits (regulatory) | SIL 1 |
| GL-011 FuelCraft | Fuel quality limits | SIL 1 |
| GL-012 SteamQual | Moisture alarms | SIL 1 |
| GL-013 PredictiveMaint | None (advisory only) | N/A |
| GL-014 ExchangerPro | Temperature limits | SIL 1 |
| GL-015 InsulScan | None (survey only) | N/A |
| GL-016 WaterGuard | Chemistry limits | SIL 1 |

## 8. Change Management

All changes to safety functions require:

1. **Hazard Analysis Update** - HAZOP/LOPA review
2. **SIL Verification** - Recalculate PFD
3. **Design Review** - Independent assessment
4. **Validation Testing** - Factory and site acceptance
5. **Documentation Update** - This document and related SRS

## 9. References

1. IEC 61511-1:2016 - Functional Safety: SIS for Process Industry
2. IEC 61508:2010 - Functional Safety of E/E/PE Systems
3. NFPA 85 - Boiler and Combustion Systems Hazards Code
4. NFPA 86 - Standard for Ovens and Furnaces
5. ASME Boiler and Pressure Vessel Code
6. API RP 556 - Instrumentation, Control, and Protective Systems

## 10. Document Control

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0 | 2024-01-15 | GreenLang Framework | Initial release |

---

**DISCLAIMER:** This document provides guidance for SIL assignment within the GreenLang framework. Actual SIL determination for specific installations must be performed by qualified functional safety engineers using site-specific hazard analysis (HAZOP, LOPA, etc.).
