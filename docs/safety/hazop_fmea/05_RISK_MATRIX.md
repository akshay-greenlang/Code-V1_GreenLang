# Risk Assessment Matrix

## Comprehensive Risk Ranking System per IEC 61511

**Document ID:** GL-RISK-MATRIX-001
**Version:** 1.0
**Effective Date:** 2025-12-05
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61511-3:2016, IEC 61508-5:2010

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Risk Matrix Structure](#2-risk-matrix-structure)
3. [Severity Categories](#3-severity-categories)
4. [Likelihood Categories](#4-likelihood-categories)
5. [Risk Matrix](#5-risk-matrix)
6. [Risk Acceptance Criteria](#6-risk-acceptance-criteria)
7. [ALARP Demonstration](#7-alarp-demonstration)
8. [Risk Category Definitions](#8-risk-category-definitions)
9. [Application Guidelines](#9-application-guidelines)
10. [Examples](#10-examples)

---

## 1. Introduction

### 1.1 Purpose

This document defines the comprehensive risk matrix used for evaluating hazards identified in HAZOP studies and other Process Hazard Analyses (PHA) for GreenLang Process Heat agents. The matrix provides consistent criteria for risk ranking and acceptance decisions.

### 1.2 Scope

The risk matrix applies to:
- GL-001 ThermalCommand Orchestrator
- GL-018 UnifiedCombustion Optimizer (including GL-002)
- GL-007 Furnace Performance Monitor
- All associated safety instrumented functions

### 1.3 References

| Document | Title |
|----------|-------|
| IEC 61511-3:2016 | Guidance for determination of required SIL |
| IEC 61508-5:2010 | Examples of methods for determination of SIL |
| CCPS | Guidelines for Hazard Evaluation Procedures |
| ISO 31000:2018 | Risk Management |

---

## 2. Risk Matrix Structure

### 2.1 Matrix Dimensions

The risk matrix is a **5x5** matrix with:
- **5 Severity levels** (Columns): Negligible to Catastrophic
- **5 Likelihood levels** (Rows): Improbable to Frequent
- **25 Risk cells**: Each with defined risk category

### 2.2 Risk Calculation

```
Risk Level = f(Severity, Likelihood)

Where:
  Severity = Consequence magnitude
  Likelihood = Probability of occurrence
```

### 2.3 Visual Structure

```
              S E V E R I T Y

        1         2         3         4         5
    Negligible  Minor   Moderate   Major   Catastrophic

5   +----------------------------------------------------+
F   |                                                    |
r   |    LOW    | MEDIUM  |  HIGH   |VERY HIGH|VERY HIGH|
e   |           |         |         |         |         |
q   +----------------------------------------------------+
u 4 |                                                    |
e   |    LOW    | MEDIUM  | MEDIUM  |  HIGH   |VERY HIGH|
n   |           |         |         |         |         |
t   +----------------------------------------------------+
  3 |                                                    |
L   |    LOW    |   LOW   | MEDIUM  | MEDIUM  |  HIGH   |
I   |           |         |         |         |         |
K   +----------------------------------------------------+
E 2 |                                                    |
L   |    LOW    |   LOW   |   LOW   | MEDIUM  | MEDIUM  |
I   |           |         |         |         |         |
H   +----------------------------------------------------+
O 1 |                                                    |
O   |    LOW    |   LOW   |   LOW   |   LOW   | MEDIUM  |
D   |           |         |         |         |         |
    +----------------------------------------------------+
```

---

## 3. Severity Categories

### 3.1 Severity Rating Scale

| Level | Category | Safety Impact | Operational Impact | Environmental Impact | Financial Impact |
|-------|----------|---------------|-------------------|---------------------|------------------|
| **5** | **Catastrophic** | Multiple fatalities | Total plant loss | Major off-site release, long-term contamination | >$100M |
| **4** | **Major** | Single fatality or multiple serious injuries | Major equipment damage, extended shutdown | Significant off-site release, reportable | $10M-$100M |
| **3** | **Moderate** | Serious injury, hospitalization | Equipment damage, days of downtime | On-site release requiring cleanup | $1M-$10M |
| **2** | **Minor** | Minor injury, first aid | Minor equipment damage, hours of downtime | Minor on-site release, contained | $100K-$1M |
| **1** | **Negligible** | No injury | No equipment damage, brief interruption | No release | <$100K |

### 3.2 Severity Examples - Process Heat Systems

| Level | GL-001 Example | GL-018 Example | GL-007 Example |
|-------|----------------|----------------|----------------|
| **5** | Boiler explosion, multiple casualties | Furnace explosion | Multiple tube ruptures, fire |
| **4** | Steam release with fatality | Combustion chamber breach | Major tube rupture, personnel injury |
| **3** | Equipment overpressure, serious burn | Flame rollback with injury | Tube leak with localized fire |
| **2** | Steam leak, minor burn | Burner trip, refractory damage | Tube thinning requiring repair |
| **1** | Control upset, no damage | Flame instability, nuisance trip | Minor efficiency loss |

### 3.3 Consequence Hierarchy

When multiple consequence types occur, use the **highest** severity:

```
Priority Order:
1. Safety (people)
2. Environmental
3. Asset/Financial
4. Operational/Business
```

---

## 4. Likelihood Categories

### 4.1 Likelihood Rating Scale

| Level | Category | Frequency | Probability per Year | Industry Equivalent |
|-------|----------|-----------|---------------------|---------------------|
| **5** | **Frequent** | Expected to occur multiple times per year | >1 | Very likely |
| **4** | **Likely** | Expected to occur once per year | 0.1 - 1 | Probable |
| **3** | **Occasional** | May occur once in 1-10 years | 0.01 - 0.1 | Possible |
| **2** | **Remote** | May occur once in 10-100 years | 0.001 - 0.01 | Unlikely |
| **1** | **Improbable** | May occur once in >100 years | <0.001 | Rare |

### 4.2 Likelihood Estimation Factors

Consider the following when estimating likelihood:

| Factor | Increases Likelihood | Decreases Likelihood |
|--------|---------------------|---------------------|
| **Complexity** | Complex systems | Simple, proven designs |
| **Experience** | New technology | Well-established |
| **Maintenance** | Deferred maintenance | Proactive maintenance |
| **Human Factors** | Manual operations | Automated systems |
| **Environment** | Harsh conditions | Controlled environment |
| **History** | Previous incidents | Clean track record |

### 4.3 Likelihood with Safeguards

When evaluating **residual risk**, include credit for safeguards:

```
Likelihood_residual = Likelihood_inherent x PFD_safeguards

Example:
  Inherent likelihood = Level 4 (0.5/yr)
  SIL 2 SIF PFD = 0.01
  Residual likelihood = 0.5 x 0.01 = 0.005/yr = Level 2
```

---

## 5. Risk Matrix

### 5.1 Complete Risk Matrix

```
                           S E V E R I T Y

                1           2           3           4           5
            Negligible    Minor     Moderate     Major    Catastrophic

       5    +-----------+-----------+-----------+-----------+-----------+
  F    F    |           |           |           |           |           |
  r    r    |    LOW    |  MEDIUM   |   HIGH    | VERY HIGH | VERY HIGH |
  e    e    |   (5)     |   (10)    |   (15)    |   (20)    |   (25)    |
  q    q    +-----------+-----------+-----------+-----------+-----------+
  u    u  4 |           |           |           |           |           |
  e    e    |    LOW    |  MEDIUM   |  MEDIUM   |   HIGH    | VERY HIGH |
  n    n    |   (4)     |   (8)     |   (12)    |   (16)    |   (20)    |
  c    t    +-----------+-----------+-----------+-----------+-----------+
  y      3  |           |           |           |           |           |
L          |    LOW    |   LOW     |  MEDIUM   |  MEDIUM   |   HIGH    |
I          |   (3)     |   (6)     |   (9)     |   (12)    |   (15)    |
K          +-----------+-----------+-----------+-----------+-----------+
E        2  |           |           |           |           |           |
L          |    LOW    |   LOW     |   LOW     |  MEDIUM   |  MEDIUM   |
I          |   (2)     |   (4)     |   (6)     |   (8)     |   (10)    |
H          +-----------+-----------+-----------+-----------+-----------+
O        1  |           |           |           |           |           |
O          |    LOW    |   LOW     |   LOW     |   LOW     |  MEDIUM   |
D          |   (1)     |   (2)     |   (3)     |   (4)     |   (5)     |
           +-----------+-----------+-----------+-----------+-----------+

Risk Score = Severity x Likelihood
```

### 5.2 Color-Coded Matrix

| Risk Category | Color | Score Range | Cell Locations |
|---------------|-------|-------------|----------------|
| **LOW** | Green | 1-6 | (1,1)(1,2)(1,3)(1,4)(2,1)(2,2)(2,3)(3,1)(3,2)(4,1)(5,1) |
| **MEDIUM** | Yellow | 7-12 | (1,5)(2,4)(2,5)(3,3)(3,4)(4,2)(4,3)(5,2) |
| **HIGH** | Orange | 13-16 | (3,5)(4,4)(5,3) |
| **VERY HIGH** | Red | 17-25 | (4,5)(5,4)(5,5) |

### 5.3 Numerical Risk Score

```
Risk Score = Severity Level x Likelihood Level

Score Ranges:
  1-6:   LOW
  7-12:  MEDIUM
  13-16: HIGH
  17-25: VERY HIGH
```

---

## 6. Risk Acceptance Criteria

### 6.1 Risk Category Actions

| Risk Category | Acceptance | Required Action |
|---------------|------------|-----------------|
| **LOW** | Acceptable | Monitor, document, no further action required |
| **MEDIUM** | Tolerable (ALARP) | Reduce if reasonably practicable, implement additional safeguards |
| **HIGH** | Unacceptable | Risk reduction required before operation, additional safeguards mandatory |
| **VERY HIGH** | Intolerable | Operation not permitted, fundamental design changes required |

### 6.2 IEC 61511 Alignment

| Risk Category | IEC 61511 Requirement | Typical SIL |
|---------------|----------------------|-------------|
| **LOW** | No SIF required | - |
| **MEDIUM** | Consider SIF, may use other safeguards | SIL 1 |
| **HIGH** | SIF required | SIL 2 |
| **VERY HIGH** | SIF with high SIL required | SIL 3 |

### 6.3 Numerical Criteria

| Individual Risk | Tolerable Frequency | Application |
|-----------------|---------------------|-------------|
| Fatality risk to worker | < 1 x 10^-4 per year | Onsite personnel |
| Fatality risk to public | < 1 x 10^-5 per year | Offsite public |
| Major injury | < 1 x 10^-3 per year | All personnel |
| Environmental release | < 1 x 10^-4 per year | Major release |

---

## 7. ALARP Demonstration

### 7.1 ALARP Principle

```
                    Intolerable
                    Region
                        |
                        |   Risk cannot be justified
                        |   (except in extraordinary
                        |    circumstances)
                        |
    +--------------+----+----+--------------+
    |              |         |              |
    |   ALARP Zone |  <----->|  Risk is     |
    |              |         |  reduced     |
    |   (Tolerable |         |  until       |
    |    if ALARP) |         |  further     |
    |              |         |  reduction   |
    +--------------+----+----+--------------+
                        |    is impractical
                        |
                    Broadly
                    Acceptable
                    Region
```

### 7.2 ALARP Assessment Criteria

For risks in the ALARP region (MEDIUM), demonstrate:

| Criterion | Evidence Required |
|-----------|-------------------|
| **Cost-Benefit Analysis** | Risk reduction cost vs. expected loss |
| **Industry Standards** | Compliance with NFPA, API, IEC standards |
| **Good Practice** | Comparison with similar facilities |
| **Practicability** | Technical feasibility of further reduction |

### 7.3 ALARP Cost-Benefit Formula

```
ALARP Ratio = Cost of Safeguard / (Risk Reduction x Value of Life)

Where:
  Cost of Safeguard = Capital + NPV of operating costs
  Risk Reduction = Delta frequency x Consequence probability
  Value of Life = $5M-$10M (industry standard)

ALARP Accepted if Ratio < 10 (typically)
```

### 7.4 ALARP Documentation

For each MEDIUM risk, document:

1. **Current Risk Level:** Score and justification
2. **Proposed Safeguards:** Description and estimated PFD
3. **Cost Estimate:** Capital and operating costs
4. **Residual Risk:** Post-implementation risk level
5. **ALARP Justification:** Why further reduction is impractical

---

## 8. Risk Category Definitions

### 8.1 LOW Risk (Green)

**Definition:** Risk is acceptable with current controls. No additional action required beyond routine monitoring.

**Characteristics:**
- Consequence is minor or likelihood is very low
- Existing safeguards are adequate
- Industry good practice is met

**Actions:**
- Document in risk register
- Monitor periodically
- Include in Management of Change reviews

### 8.2 MEDIUM Risk (Yellow)

**Definition:** Risk is tolerable if reduced to As Low As Reasonably Practicable (ALARP). Additional safeguards should be considered.

**Characteristics:**
- Moderate consequence or occasional likelihood
- Further risk reduction may be practicable
- May require SIL 1 safety function

**Actions:**
- Conduct ALARP assessment
- Consider additional safeguards
- Document justification if no further action
- Review annually

### 8.3 HIGH Risk (Orange)

**Definition:** Risk is unacceptable with current controls. Additional safeguards are required before operation.

**Characteristics:**
- Significant consequence with reasonable likelihood
- Additional protection layers needed
- Typically requires SIL 2 safety function

**Actions:**
- Mandatory risk reduction
- Design additional safeguards
- Verify PFD meets requirements
- Management approval before operation
- Review quarterly

### 8.4 VERY HIGH Risk (Red)

**Definition:** Risk is intolerable. Operation is not permitted until risk is reduced.

**Characteristics:**
- Severe consequence with probable likelihood
- Fundamental design change required
- May require SIL 3 safety function or redesign

**Actions:**
- Stop operation (if operating)
- Do not proceed with design (if new)
- Fundamental risk reduction required
- Multiple independent safeguards
- Executive management approval
- Independent verification

---

## 9. Application Guidelines

### 9.1 Risk Assessment Process

```
+-------------------+
| 1. Identify       |
|    Hazard         |
+-------------------+
         |
         v
+-------------------+
| 2. Assess         |
|    Consequence    |
|    (Severity)     |
+-------------------+
         |
         v
+-------------------+
| 3. Assess         |
|    Likelihood     |
|    (Frequency)    |
+-------------------+
         |
         v
+-------------------+
| 4. Determine      |
|    Risk Level     |
+-------------------+
         |
         v
+-------------------+
| 5. Identify       |
|    Safeguards     |
+-------------------+
         |
         v
+-------------------+
| 6. Assess         |
|    Residual Risk  |
+-------------------+
         |
         v
+-------------------+
| 7. Document       |
|    & Track        |
+-------------------+
```

### 9.2 Consequence Assessment Guidelines

| Question | Consideration |
|----------|---------------|
| What is the worst credible consequence? | Consider escalation potential |
| Who is affected? | Workers, contractors, public |
| What is the duration of impact? | Short-term vs. long-term |
| What environmental media are affected? | Air, water, soil |
| What is the financial impact? | Direct + indirect costs |

### 9.3 Likelihood Assessment Guidelines

| Question | Consideration |
|----------|---------------|
| What is the initiating event frequency? | Use industry data |
| What enabling conditions exist? | Simultaneous requirements |
| What is the exposure time? | Continuous vs. intermittent |
| What is the probability of escalation? | Conditional probabilities |
| What safeguards provide credit? | Only independent, validated safeguards |

### 9.4 Multi-Consequence Scenarios

When a single event can lead to multiple consequence types:

1. **Assess each consequence type separately**
2. **Use highest severity for risk ranking**
3. **Document all consequence types in risk register**
4. **Address safeguards for each consequence type**

### 9.5 Uncertainty Handling

| Uncertainty Level | Approach |
|-------------------|----------|
| **High** | Use conservative (higher) estimates |
| **Moderate** | Use best estimate with sensitivity analysis |
| **Low** | Use best estimate |

Document uncertainty assumptions and revisit when better data available.

---

## 10. Examples

### 10.1 Example 1: High Temperature Trip Failure

**Scenario:** GL-001 High Temperature Trip fails to function on demand, leading to equipment overpressure.

| Assessment | Rating | Justification |
|------------|--------|---------------|
| **Severity** | 4 (Major) | Single equipment overpressure with potential for fatality |
| **Likelihood (Inherent)** | 3 (Occasional) | Based on BPCS failure rate (0.1/yr) x Probability of demand (0.1/yr) = 0.01/yr |
| **Risk Score (Inherent)** | 12 (MEDIUM) | Requires ALARP assessment |
| **Safeguards** | SIL 2 SIF (PFD 0.01) | 2oo3 temperature trip |
| **Likelihood (Residual)** | 1 (Improbable) | 0.01 x 0.01 = 0.0001/yr |
| **Risk Score (Residual)** | 4 (LOW) | Acceptable |

### 10.2 Example 2: Flame Failure Leading to Explosion

**Scenario:** GL-018 Flame failure undetected, fuel accumulates, explosion on re-ignition.

| Assessment | Rating | Justification |
|------------|--------|---------------|
| **Severity** | 5 (Catastrophic) | Furnace explosion, multiple casualties |
| **Likelihood (Inherent)** | 3 (Occasional) | Flame failure (0.5/yr) x Failure to detect (0.1) x Ignition (0.1) = 0.005/yr |
| **Risk Score (Inherent)** | 15 (HIGH) | Risk reduction required |
| **Safeguards** | BMS with 4-second detection | Per NFPA 85 |
| **Additional** | Double block and bleed | Fuel isolation |
| **Likelihood (Residual)** | 1 (Improbable) | With safeguards: 0.00005/yr |
| **Risk Score (Residual)** | 5 (LOW) | Acceptable with safeguards |

### 10.3 Example 3: TMT Sensor Drift

**Scenario:** GL-007 TMT sensor drifts low, actual tube temperature exceeds design.

| Assessment | Rating | Justification |
|------------|--------|---------------|
| **Severity** | 4 (Major) | Tube rupture with fire potential |
| **Likelihood (Inherent)** | 4 (Likely) | Sensor drift (0.2/yr) x Undetected (0.5) = 0.1/yr |
| **Risk Score (Inherent)** | 16 (HIGH) | Risk reduction required |
| **Safeguards** | Redundant sensors (2oo3) | Comparison logic |
| **Additional** | IR scanning monthly | Verification |
| **Likelihood (Residual)** | 2 (Remote) | With redundancy: 0.005/yr |
| **Risk Score (Residual)** | 8 (MEDIUM) | ALARP - monitor and maintain |

### 10.4 Risk Decision Tree

```
Start
  |
  v
Assess Severity --> 5? --> Very High likely --> VERY HIGH (RED)
  |
  v
Assess Likelihood --> 5? --> High severity likely --> HIGH/VERY HIGH
  |
  v
Calculate Score
  |
  v
Score > 16? --> VERY HIGH --> Intolerable, redesign required
  |
  v
Score 13-16? --> HIGH --> Risk reduction required
  |
  v
Score 7-12? --> MEDIUM --> ALARP assessment required
  |
  v
Score 1-6? --> LOW --> Acceptable, document and monitor
```

---

## Appendix A: Quick Reference Card

### Risk Matrix (One Page)

```
+-------------+-------------+-------------+-------------+-------------+
|    1-Neg    |    2-Min    |    3-Mod    |    4-Maj    |    5-Cat    |
+-------------+-------------+-------------+-------------+-------------+
| 5-Freq  LOW |   MEDIUM    |    HIGH     | VERY HIGH   | VERY HIGH   |
+-------------+-------------+-------------+-------------+-------------+
| 4-Like  LOW |   MEDIUM    |   MEDIUM    |    HIGH     | VERY HIGH   |
+-------------+-------------+-------------+-------------+-------------+
| 3-Occa  LOW |    LOW      |   MEDIUM    |   MEDIUM    |    HIGH     |
+-------------+-------------+-------------+-------------+-------------+
| 2-Remo  LOW |    LOW      |    LOW      |   MEDIUM    |   MEDIUM    |
+-------------+-------------+-------------+-------------+-------------+
| 1-Impr  LOW |    LOW      |    LOW      |    LOW      |   MEDIUM    |
+-------------+-------------+-------------+-------------+-------------+

LOW (Green):     Accept, monitor
MEDIUM (Yellow): ALARP, consider safeguards
HIGH (Orange):   Reduce risk before operation
VERY HIGH (Red): Intolerable, redesign
```

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-TechWriter | Initial release |

---

**Document End**

*This document is part of the GreenLang Process Heat Safety Documentation Package.*
