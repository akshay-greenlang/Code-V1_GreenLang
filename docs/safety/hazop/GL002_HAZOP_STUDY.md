# GL-002 Emissions Calculation Agent - Comprehensive HAZOP Study

## Hazard and Operability Analysis per IEC 61882:2016

**Document ID:** GL-HAZOP-002-REV1
**Version:** 1.0
**Effective Date:** 2025-12-06
**Classification:** Safety Critical Documentation
**Standard Reference:** IEC 61882:2016 Hazard and Operability Studies
**Regulatory Alignment:** ISO 17776, EPA 40 CFR Part 98, GHG Protocol, EU ETS

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Study Scope and Methodology](#3-study-scope-and-methodology)
4. [System Description](#4-system-description)
5. [HAZOP Team and Responsibilities](#5-hazop-team-and-responsibilities)
6. [Node 1: Fuel Data Input Processing](#6-node-1-fuel-data-input-processing)
7. [Node 2: Emission Factor Selection](#7-node-2-emission-factor-selection)
8. [Node 3: Calculation Engine](#8-node-3-calculation-engine)
9. [Node 4: Unit Conversion Module](#9-node-4-unit-conversion-module)
10. [Node 5: Uncertainty Propagation](#10-node-5-uncertainty-propagation)
11. [Node 6: Result Validation and Output](#11-node-6-result-validation-and-output)
12. [Risk Ranking Matrix](#12-risk-ranking-matrix)
13. [Risk Register](#13-risk-register)
14. [Action Tracking Table](#14-action-tracking-table)
15. [Appendices](#15-appendices)

---

## 1. Executive Summary

### 1.1 Study Overview

This Hazard and Operability (HAZOP) study examines the GL-002 Emissions Calculation Agent, the core computational engine for greenhouse gas (GHG) emissions quantification in GreenLang's sustainability platform. The study was conducted per IEC 61882:2016 methodology with alignment to ISO 17776 for petroleum and natural gas industries, with specific focus on:

- Fuel data input processing and validation
- Emission factor selection per EPA AP-42 and 40 CFR Part 98
- Calculation engine accuracy for CO2, CH4, and N2O emissions
- Unit conversion integrity across measurement systems
- Uncertainty propagation per GHG Protocol Tier 1/2/3 methodologies
- Result validation and regulatory compliance verification

### 1.2 Key Findings Summary

| Risk Category | Count | Critical Items |
|---------------|-------|----------------|
| **Very High (VH)** | 4 | Calculation errors causing regulatory non-compliance, emission factor mismatches |
| **High (H)** | 14 | Unit conversion errors, uncertainty underestimation, data integrity failures |
| **Medium (M)** | 32 | Input validation gaps, caching inconsistencies, provenance tracking gaps |
| **Low (L)** | 12 | Minor operational issues, logging gaps |

### 1.3 Critical Recommendations

| Priority | Recommendation | Owner | Due Date |
|----------|----------------|-------|----------|
| Critical | Implement dual-path verification for all emissions calculations with independent algorithm validation | Software | 2025-12-31 |
| Critical | Deploy EPA Part 98 emission factor database with version control and audit trail | Data | 2025-12-31 |
| Critical | Add unit conversion validation with dimensional analysis cross-check | Software | 2025-12-31 |
| High | Implement uncertainty propagation per GHG Protocol Chapter 6 guidance | Software | 2026-01-31 |
| High | Add regulatory threshold monitoring with automatic compliance alerts | Software | 2026-01-31 |

---

## 2. Introduction

### 2.1 Purpose

This document presents a comprehensive HAZOP study for the GL-002 Emissions Calculation Agent per IEC 61882:2016. The purpose is to:

- Systematically identify potential hazards and operability issues in emissions quantification
- Evaluate consequences of calculation deviations on regulatory compliance
- Assess adequacy of existing safeguards for data integrity and accuracy
- Recommend additional safeguards where residual risk to regulatory compliance is unacceptable
- Ensure emissions calculations meet EPA Part 98, EU ETS, and GHG Protocol requirements
- Provide traceability for emissions data used in regulatory filings

### 2.2 Objectives

1. **Accuracy Assurance:** Identify all credible deviations that could lead to inaccurate emissions calculations
2. **Regulatory Compliance:** Assess risks to EPA Part 98, EU ETS, and voluntary reporting compliance
3. **Data Integrity:** Evaluate data flow integrity from input to reported values
4. **Uncertainty Management:** Ensure proper uncertainty quantification per GHG Protocol
5. **Audit Trail:** Verify complete provenance tracking for regulatory audits
6. **Documentation:** Provide auditable record for third-party verification

### 2.3 References

| Document ID | Title | Relevance |
|-------------|-------|-----------|
| IEC 61882:2016 | Hazard and Operability Studies | Primary methodology |
| ISO 17776:2016 | Petroleum and Natural Gas - Safety | Risk assessment framework |
| 40 CFR Part 98 | Mandatory GHG Reporting Rule | EPA reporting requirements |
| EPA AP-42 | Compilation of Air Pollutant Emission Factors | Emission factor source |
| GHG Protocol | Corporate Standard | Calculation methodology |
| EU 2018/2066 | EU ETS MRR Regulation | EU reporting requirements |
| IPCC 2006 Guidelines | National GHG Inventories | Tier methodology |

### 2.4 Definitions and Abbreviations

| Term | Definition |
|------|------------|
| **AR6** | IPCC Sixth Assessment Report (GWP values) |
| **CF** | Carbon Fraction |
| **EF** | Emission Factor |
| **EU ETS** | European Union Emissions Trading System |
| **GHG** | Greenhouse Gas |
| **GWP** | Global Warming Potential |
| **HHV** | Higher Heating Value |
| **LHV** | Lower Heating Value |
| **MRR** | Monitoring and Reporting Regulation |
| **MRV** | Monitoring, Reporting, and Verification |
| **Tier 1/2/3** | GHG Protocol calculation approach tiers |
| **tCO2e** | Tonnes of CO2 equivalent |
| **UQ** | Uncertainty Quantification |

---

## 3. Study Scope and Methodology

### 3.1 Scope Definition

**Included in Scope:**

| Area | Description |
|------|-------------|
| Fuel Data Input Processing | Fuel type identification, quantity validation, heating values |
| Emission Factor Selection | EPA AP-42 factors, custom factors, region-specific factors |
| Calculation Engine | CO2, CH4, N2O calculations, GWP application |
| Unit Conversion | Mass, energy, volume conversions, temperature corrections |
| Uncertainty Propagation | Tier-based uncertainty, Monte Carlo simulation |
| Result Validation | Threshold checking, regulatory limits, output formatting |

**Excluded from Scope:**

| Area | Covered By |
|------|------------|
| Physical fuel sampling and analysis | Laboratory procedures |
| CEMS (Continuous Emissions Monitoring) | Equipment-specific procedures |
| Regulatory filing submission | GL-Compliance agent |
| Network security for data transmission | Cybersecurity assessment |

### 3.2 Methodology

The HAZOP study follows IEC 61882:2016 with adaptations for software agent analysis:

```
+-------------------+
| 1. Node Definition|
| Define data flow  |
| and calculation   |
| boundaries        |
+-------------------+
         |
         v
+-------------------+
| 2. Guide Word     |
| Application       |
| Apply to data and |
| calculation params|
+-------------------+
         |
         v
+-------------------+
| 3. Deviation      |
| Identification    |
| Identify data and |
| calculation errors|
+-------------------+
         |
         v
+-------------------+
| 4. Cause          |
| Determination     |
| Software, data,   |
| configuration     |
+-------------------+
         |
         v
+-------------------+
| 5. Consequence    |
| Assessment        |
| Regulatory, audit,|
| financial         |
+-------------------+
         |
         v
+-------------------+
| 6. Safeguard      |
| Review            |
| Validation, checks|
| reconciliation    |
+-------------------+
         |
         v
+-------------------+
| 7. Recommendation |
| Development       |
| Actions to reduce |
| calculation risk  |
+-------------------+
         |
         v
+-------------------+
| 8. Risk Ranking   |
| Regulatory x      |
| Likelihood matrix |
+-------------------+
```

### 3.3 Guide Words

| Guide Word | Meaning | Application to Emissions Agent |
|------------|---------|--------------------------------|
| **NO/NONE** | Complete negation | No data, no calculation, no output |
| **MORE** | Quantitative increase | Higher values, over-reporting |
| **LESS** | Quantitative decrease | Lower values, under-reporting |
| **AS WELL AS** | Qualitative addition | Extra data, spurious values |
| **PART OF** | Qualitative reduction | Incomplete data, missing gases |
| **REVERSE** | Logical opposite | Inverted signs, wrong direction |
| **OTHER THAN** | Complete substitution | Wrong fuel type, wrong factor |

### 3.4 Risk Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **Regulatory** | EPA/EU ETS compliance failure | Under-reporting penalties, audit findings |
| **Financial** | Carbon cost miscalculation | Incorrect ETS liability, carbon tax errors |
| **Reputational** | Sustainability credibility | Greenwashing allegations, rating downgrades |
| **Operational** | Business decision errors | Wrong investment decisions, incorrect baselines |

---

## 4. System Description

### 4.1 GL-002 Emissions Calculation Agent Overview

The GL-002 Emissions Calculation Agent is the core computational engine for GHG emissions quantification. It receives fuel consumption data and activity data, applies appropriate emission factors, and calculates CO2, CH4, and N2O emissions with complete uncertainty quantification and provenance tracking.

| Function | Description | Regulatory Reference |
|----------|-------------|---------------------|
| Fuel Data Processing | Validate and normalize fuel input data | 40 CFR 98.33 |
| Emission Factor Selection | Select appropriate factors from EPA/IPCC databases | 40 CFR 98.34 |
| GHG Calculation | Calculate CO2, CH4, N2O using mass balance or EF approach | 40 CFR 98.33(a) |
| GWP Application | Apply AR6 GWP factors for CO2e conversion | 40 CFR 98.2 |
| Uncertainty Quantification | Calculate measurement and factor uncertainties | GHG Protocol Ch. 6 |
| Result Validation | Verify against historical data and thresholds | 40 CFR 98.3(i) |

### 4.2 System Architecture

```
                    +----------------------------------------+
                    |          GL-001 Orchestrator           |
                    |        (Coordination Layer)            |
                    +----------------------------------------+
                                      |
                                      v
+------------------+    +------------------------------------------+    +------------------+
| Fuel Data Input  |--->|         GL-002 EMISSIONS AGENT           |--->| Reporting Output |
| - Fuel type      |    |                                          |    | - tCO2e totals   |
| - Quantity (kg)  |    |  +------------------------------------+  |    | - Gas breakdown  |
| - Heating value  |    |  |    Emission Factor Database       |  |    | - Uncertainty    |
| - Carbon content |    |  |    - EPA AP-42 factors            |  |    | - Provenance     |
+------------------+    |  |    - 40 CFR Part 98 factors        |  |    +------------------+
                        |  |    - IPCC 2006 defaults            |  |
                        |  |    - Custom verified factors       |  |
                        |  +------------------------------------+  |
                        |                                          |
                        |  +------------------------------------+  |
                        |  |    Calculation Engine              |  |
                        |  |    - CO2 = Fuel x EF_CO2           |  |
                        |  |    - CH4 = Fuel x EF_CH4           |  |
                        |  |    - N2O = Fuel x EF_N2O           |  |
                        |  |    - CO2e = SUM(GHG x GWP)         |  |
                        |  +------------------------------------+  |
                        |                                          |
                        |  +------------------------------------+  |
                        |  |    Unit Conversion Module          |  |
                        |  |    - Mass: kg, lb, tonnes          |  |
                        |  |    - Energy: GJ, MMBtu, MWh        |  |
                        |  |    - Volume: m3, scf, BBL          |  |
                        |  +------------------------------------+  |
                        |                                          |
                        |  +------------------------------------+  |
                        |  |    Uncertainty Module              |  |
                        |  |    - Activity data uncertainty     |  |
                        |  |    - Emission factor uncertainty   |  |
                        |  |    - Combined uncertainty (RSS)    |  |
                        |  +------------------------------------+  |
                        |                                          |
                        |  +------------------------------------+  |
                        |  |    Provenance Tracker              |  |
                        |  |    - SHA-256 data hashing          |  |
                        |  |    - Calculation step logging      |  |
                        |  |    - Audit trail generation        |  |
                        |  +------------------------------------+  |
                        +------------------------------------------+
```

### 4.3 Key Calculation Parameters

| Parameter | Description | Typical Range | Regulatory Reference |
|-----------|-------------|---------------|---------------------|
| Fuel Consumption | Mass of fuel combusted | 0 - 10^6 kg/hr | 40 CFR 98.33(a)(1) |
| Higher Heating Value (HHV) | Energy content per mass | 30-55 MJ/kg | 40 CFR 98.33(a)(2) |
| Carbon Content | Mass fraction of carbon | 0.70-0.90 | 40 CFR 98.33(a)(3) |
| CO2 Emission Factor | kg CO2 per kg fuel | 1.5-4.0 | 40 CFR 98.34 |
| CH4 Emission Factor | kg CH4 per kg fuel | 0.0001-0.01 | 40 CFR 98.34 |
| N2O Emission Factor | kg N2O per kg fuel | 0.00001-0.001 | 40 CFR 98.34 |
| GWP CO2 | Global Warming Potential | 1 | IPCC AR6 |
| GWP CH4 | Global Warming Potential | 28 | IPCC AR6 |
| GWP N2O | Global Warming Potential | 265 | IPCC AR6 |

### 4.4 Data Flows

| Input | Source | Validation | Output Destination |
|-------|--------|------------|-------------------|
| Fuel consumption (kg) | Fuel meters, purchase records | Range check, mass balance | CO2, CH4, N2O calculations |
| Fuel type ID | Fuel specifications | Lookup validation | Emission factor selection |
| Heating value (MJ/kg) | Lab analysis, default tables | Range validation | Energy-based EF conversion |
| Carbon content (%) | Lab analysis, default values | Range 0-100%, reasonableness | Carbon balance calculation |
| Operating hours | Control systems, logs | 0-8760 hr/yr | Annual emissions rollup |

### 4.5 Integration Points

| Interface | Protocol | Direction | Validation |
|-----------|----------|-----------|------------|
| GL-001 Orchestrator | gRPC | Bidirectional | Schema validation |
| Fuel Meter API | REST | Input | Checksum, timestamp |
| EPA WebFIRE | REST | Input (EF lookup) | Version verification |
| Regulatory Filing System | REST | Output | Digital signature |
| Audit Database | SQL | Output | Provenance hash |

---

## 5. HAZOP Team and Responsibilities

### 5.1 Study Team

| Role | Name | Responsibility |
|------|------|----------------|
| **HAZOP Leader** | [Facilitator] | Study facilitation, IEC 61882 methodology |
| **Emissions Engineer** | [Emissions SME] | EPA Part 98 compliance, calculation methodology |
| **Software Engineer** | [Software SME] | Agent architecture, algorithm implementation |
| **Data Quality Engineer** | [Data SME] | Data validation, uncertainty quantification |
| **Regulatory Specialist** | [Regulatory SME] | EPA, EU ETS, voluntary standards |
| **QA/Verification Lead** | [QA SME] | Testing, validation procedures |
| **Scribe** | [Recorder] | Documentation, action tracking |

### 5.2 Study Sessions

| Session | Date | Duration | Nodes Covered |
|---------|------|----------|---------------|
| Session 1 | 2025-12-06 | 4 hours | Node 1 (Fuel Data Input Processing) |
| Session 2 | 2025-12-06 | 4 hours | Node 2 (Emission Factor Selection) |
| Session 3 | 2025-12-06 | 4 hours | Node 3 (Calculation Engine) |
| Session 4 | 2025-12-06 | 4 hours | Node 4 (Unit Conversion Module) |
| Session 5 | 2025-12-06 | 4 hours | Node 5 (Uncertainty Propagation) |
| Session 6 | 2025-12-06 | 4 hours | Node 6 (Result Validation & Output) |
| Session 7 | 2025-12-06 | 2 hours | Review and Action Items |

---

## 6. Node 1: Fuel Data Input Processing

### 6.1 Node Description

**Design Intent:** Receive, validate, and normalize fuel consumption data from various sources (meters, purchase records, estimates) for accurate emissions calculation per EPA Part 98 requirements.

**Parameters Under Study:**
- Fuel type identification
- Fuel quantity (mass, volume, energy)
- Heating value (HHV/LHV)
- Carbon content
- Sulfur content
- Moisture content
- Timestamp and period

**Boundaries:**
- Input: Raw fuel data from meters, ERP systems, manual entry
- Output: Validated, normalized fuel data for calculation engine

### 6.2 HAZOP Worksheet - Node 1

| Dev ID | Guide Word | Parameter | Deviation | Cause | Consequence | Existing Safeguards | Rec | S | L | Risk |
|--------|------------|-----------|-----------|-------|-------------|---------------------|-----|---|---|------|
| 1.1 | NO | Fuel Data | No fuel data received for period | Meter failure, communication error, data pipeline failure | Zero emissions calculated for period; under-reporting; EPA non-compliance | Data completeness check, gap detection, manual entry fallback | R1.1 | 5 | 3 | H |
| 1.2 | NO | Fuel Type | Fuel type not specified | Incomplete data entry, system default not set | Wrong emission factor applied; calculation error | Mandatory field validation, default fuel warning | R1.2 | 4 | 3 | M |
| 1.3 | NO | Heating Value | Missing heating value | Lab analysis not performed, data not entered | Cannot convert volume to energy; calculation fails | Default HHV lookup by fuel type | R1.3 | 3 | 4 | M |
| 1.4 | MORE | Fuel Quantity | Fuel quantity higher than actual | Meter over-reading, double counting, unit error | Over-reporting emissions; excess carbon costs | Mass balance reconciliation, trend analysis | R1.4 | 3 | 3 | M |
| 1.5 | MORE | Heating Value | HHV higher than actual | Lab analysis error, wrong fuel specification | Higher emissions calculated than actual | HHV range validation by fuel type | R1.5 | 3 | 3 | M |
| 1.6 | MORE | Carbon Content | Carbon content higher than fuel supports | Data entry error, wrong fuel analysis | Over-reporting CO2 emissions | Carbon content range check (0.70-0.90 typical) | R1.6 | 3 | 3 | M |
| 1.7 | LESS | Fuel Quantity | Fuel quantity lower than actual | Meter under-reading, data loss, estimation error | Under-reporting emissions; regulatory violation | Fuel purchase reconciliation, inventory check | R1.7 | 5 | 3 | H |
| 1.8 | LESS | Heating Value | LHV used instead of HHV | Configuration error, unit confusion | Lower emissions calculated (10-15% error typical) | HHV/LHV flag validation, fuel type defaults | R1.8 | 4 | 3 | M |
| 1.9 | LESS | Carbon Content | Carbon content underestimated | Analysis error, biogenic fraction confusion | Under-reporting CO2; regulatory risk | Carbon content validation against fuel type norms | R1.9 | 4 | 3 | M |
| 1.10 | AS WELL AS | Fuel Data | Duplicate fuel records | System integration error, manual entry duplication | Double counting emissions; over-reporting | Duplicate detection, timestamp uniqueness check | R1.10 | 3 | 3 | M |
| 1.11 | AS WELL AS | Fuel Type | Multiple fuel types mixed in data | Fuel blending not accounted, misclassification | Wrong composite EF; calculation inaccuracy | Fuel blend handling, weighted average EF | R1.11 | 3 | 3 | M |
| 1.12 | PART OF | Fuel Data | Incomplete fuel data set | Missing periods, partial meter reads | Under-reporting for missing periods | Data completeness validation, gap alerting | R1.12 | 4 | 3 | M |
| 1.13 | PART OF | Fuel Properties | Missing fuel properties (S%, ash) | Lab analysis incomplete | Cannot calculate SOx, PM emissions | Property completeness check, default values | R1.13 | 2 | 4 | M |
| 1.14 | REVERSE | Fuel Flow Direction | Negative fuel flow (export/return) | Bidirectional flow, tank transfers | Incorrect net consumption; error in mass balance | Sign validation, flow direction flag | R1.14 | 3 | 2 | L |
| 1.15 | OTHER THAN | Fuel Type | Wrong fuel type identified | Manual entry error, code mismatch | Completely wrong emission factor applied | Fuel type validation against master list | R1.15 | 5 | 2 | M |
| 1.16 | OTHER THAN | Units | Wrong units for fuel quantity | kg vs lb, m3 vs scf confusion | Factor of 2-35x error in quantity | Unit validation, automatic conversion with confirmation | R1.16 | 5 | 3 | H |
| 1.17 | OTHER THAN | Time Period | Data for wrong time period | Timestamp error, timezone confusion | Emissions allocated to wrong reporting period | Timestamp validation, period boundary check | R1.17 | 3 | 3 | M |
| 1.18 | EARLY | Fuel Data | Preliminary data used as final | Process timing, data finalization incomplete | Reported values may change; audit inconsistency | Data status flag (preliminary/final) | R1.18 | 2 | 4 | M |
| 1.19 | LATE | Fuel Data | Delayed fuel data receipt | Integration delays, manual entry backlog | Incomplete reporting for period; filing delay | Data receipt monitoring, escalation alerts | R1.19 | 3 | 3 | M |

### 6.3 Node 1 - Recommendations

| Rec ID | Description | Priority | Owner | Due Date | Status |
|--------|-------------|----------|-------|----------|--------|
| R1.1 | Implement automated data completeness monitoring with gap detection and alerts for missing periods >24 hours | Critical | Data | 2025-12-31 | Open |
| R1.2 | Require fuel type validation against EPA fuel categories with warning on unknown types | High | Software | 2026-01-31 | Open |
| R1.3 | Deploy default HHV lookup table from EPA Part 98 Table C-1 with version tracking | High | Data | 2026-01-31 | Open |
| R1.4 | Implement fuel purchase vs. consumption reconciliation with variance alerting (>5%) | High | Software | 2026-01-31 | Open |
| R1.5 | Add HHV range validation: coal 20-35 MJ/kg, natural gas 35-55 MJ/kg, oil 40-48 MJ/kg | Medium | Software | 2026-02-28 | Open |
| R1.6 | Validate carbon content: coal 0.70-0.85, oil 0.80-0.87, gas 0.65-0.75 | Medium | Software | 2026-02-28 | Open |
| R1.7 | Implement inventory-based fuel consumption verification for annual reporting | Critical | Data | 2025-12-31 | Open |
| R1.8 | Enforce HHV/LHV selection with clear labeling and automatic conversion when needed | High | Software | 2026-01-31 | Open |
| R1.9 | Require biogenic/fossil carbon fraction specification for biomass and waste fuels | High | Software | 2026-01-31 | Open |
| R1.10 | Add duplicate record detection using (source, timestamp, quantity) composite key | High | Software | 2026-01-31 | Open |
| R1.11 | Implement fuel blend handling with weighted average emission factor calculation | Medium | Software | 2026-02-28 | Open |
| R1.12 | Create data quality dashboard showing completeness by source and period | Medium | Data | 2026-02-28 | Open |
| R1.14 | Add fuel flow direction validation with net consumption calculation | Medium | Software | 2026-02-28 | Open |
| R1.15 | Implement fuel type code mapping with validation against EPA Part 98 categories | High | Software | 2026-01-31 | Open |
| R1.16 | Add unit validation and automatic conversion with user confirmation for ambiguous inputs | Critical | Software | 2025-12-31 | Open |
| R1.17 | Implement timezone-aware timestamp handling with UTC standardization | High | Software | 2026-01-31 | Open |
| R1.18 | Add data status workflow (draft -> verified -> final) with change tracking | Medium | Software | 2026-02-28 | Open |
| R1.19 | Implement data freshness monitoring with SLA alerts for late data | Medium | Data | 2026-02-28 | Open |

---

## 7. Node 2: Emission Factor Selection

### 7.1 Node Description

**Design Intent:** Select appropriate emission factors (EF) for each fuel type and combustion configuration per EPA Part 98, IPCC 2006, or facility-specific measured values, ensuring regulatory compliance and calculation accuracy.

**Parameters Under Study:**
- EPA Part 98 default emission factors
- EPA AP-42 emission factors
- Facility-specific emission factors
- Regional/country-specific factors
- Factor version and effective date
- Tier 1/2/3 approach selection

**Boundaries:**
- Input: Fuel type, combustion equipment type, control technology, regulatory jurisdiction
- Output: Validated emission factors (CO2, CH4, N2O) with uncertainty ranges

### 7.2 HAZOP Worksheet - Node 2

| Dev ID | Guide Word | Parameter | Deviation | Cause | Consequence | Existing Safeguards | Rec | S | L | Risk |
|--------|------------|-----------|-----------|-------|-------------|---------------------|-----|---|---|------|
| 2.1 | NO | Emission Factor | No emission factor available for fuel | Unknown fuel type, new fuel, database incomplete | Calculation cannot proceed; data gap | Default factor with uncertainty flag | R2.1 | 4 | 2 | M |
| 2.2 | NO | Factor Update | Emission factor database not updated | Update process failure, version control issue | Outdated factors used; potential non-compliance | Database version tracking, update alerts | R2.2 | 3 | 3 | M |
| 2.3 | NO | Factor Uncertainty | Uncertainty value not provided with EF | Data source limitation, uncertainty not quantified | Cannot calculate total uncertainty; GHG Protocol non-compliance | Default uncertainty by tier level | R2.3 | 3 | 3 | M |
| 2.4 | MORE | CO2 Factor | CO2 emission factor higher than actual | Conservative default used, wrong fuel subcategory | Over-reporting CO2 emissions | Factor validation against fuel type norms | R2.4 | 2 | 3 | L |
| 2.5 | MORE | CH4 Factor | CH4 emission factor higher than combustion supports | Wrong equipment type selected, uncontrolled factor used | CH4 emissions overstated (can be 10-100x) | Equipment type validation, control technology flag | R2.5 | 3 | 3 | M |
| 2.6 | MORE | GWP Value | GWP value higher than IPCC AR6 | Using outdated AR5/AR4 GWP, configuration error | CO2e over-calculated | GWP version validation against regulatory requirement | R2.6 | 3 | 3 | M |
| 2.7 | LESS | CO2 Factor | CO2 emission factor lower than actual | Optimistic custom factor, analysis error | Under-reporting CO2; regulatory violation | Custom factor verification requirement | R2.7 | 5 | 2 | M |
| 2.8 | LESS | CH4 Factor | CH4 emission factor underestimates leakage | Ignoring equipment-specific losses, incomplete scope | Significant CH4 under-reporting (fugitive) | Fugitive emission scope check | R2.8 | 4 | 3 | M |
| 2.9 | LESS | N2O Factor | N2O emission factor underestimated | Wrong combustion temperature range, control credit | Under-reporting N2O (265 GWP impact) | N2O factor validation by temperature/equipment | R2.9 | 4 | 3 | M |
| 2.10 | AS WELL AS | Emission Factor | Multiple conflicting factors available | Different sources (EPA, IPCC, custom), no hierarchy | Ambiguous calculation, inconsistent results | Factor source hierarchy configuration | R2.10 | 3 | 3 | M |
| 2.11 | AS WELL AS | Factor Scope | Factor includes emissions not in calculation scope | Factor aggregation differs from scope definition | Double counting or scope mismatch | Scope alignment validation | R2.11 | 4 | 2 | M |
| 2.12 | PART OF | Emission Factor | Only CO2 factor applied, CH4/N2O missing | Tier 1 limitation, incomplete database | Under-reporting total GHG by 1-5% | Multi-gas factor completeness check | R2.12 | 4 | 3 | M |
| 2.13 | PART OF | Factor Conditions | Factor used without applicability conditions | Ignoring temperature, load, or fuel quality effects | Factor not applicable to actual conditions | Applicability range validation | R2.13 | 3 | 3 | M |
| 2.14 | REVERSE | Factor Units | Factor units inverted (kg/GJ vs GJ/kg) | Data entry error, unit confusion | Orders of magnitude error | Dimensional analysis validation | R2.14 | 5 | 2 | H |
| 2.15 | OTHER THAN | Factor Source | Wrong regulatory factor source used | EPA Part 98 vs EU ETS factor, jurisdiction error | Non-compliance with applicable regulation | Jurisdiction-based factor selection | R2.15 | 5 | 2 | H |
| 2.16 | OTHER THAN | Fuel Category | Wrong fuel category for factor lookup | Fuel misclassification, subcategory error | Significant factor mismatch (can be 2x difference) | Fuel category validation with user confirmation | R2.16 | 5 | 3 | VH |
| 2.17 | OTHER THAN | Equipment Type | Wrong equipment type for factor selection | Equipment misidentification, generic factor used | Factor not representative; audit finding | Equipment registry validation | R2.17 | 4 | 3 | M |
| 2.18 | OTHER THAN | Control Technology | Wrong control technology credit applied | Control not operating, degraded performance | Over-crediting emission reduction | Control technology status verification | R2.18 | 4 | 2 | M |
| 2.19 | EARLY | Factor Effective Date | Future factor used before effective date | Premature database update, date handling error | Using factor not yet approved | Effective date validation against calculation period | R2.19 | 3 | 2 | L |
| 2.20 | LATE | Factor Update | Old factor used after new factor effective | Delayed database update, cache issue | Using superseded factor; potential non-compliance | Factor version monitoring, cache invalidation | R2.20 | 4 | 3 | M |

### 7.3 Node 2 - Recommendations

| Rec ID | Description | Priority | Owner | Due Date | Status |
|--------|-------------|----------|-------|----------|--------|
| R2.1 | Implement comprehensive fuel type to emission factor mapping with fallback hierarchy | High | Data | 2026-01-31 | Open |
| R2.2 | Deploy automated EPA WebFIRE synchronization with version tracking and update notifications | High | Software | 2026-01-31 | Open |
| R2.3 | Add default uncertainty values by GHG Protocol tier: Tier 1 +/-30%, Tier 2 +/-10%, Tier 3 +/-5% | High | Software | 2026-01-31 | Open |
| R2.5 | Require equipment type and control technology specification for CH4/N2O factor selection | High | Software | 2026-01-31 | Open |
| R2.6 | Configure regulatory-specific GWP values: EPA uses AR4, EU ETS uses AR5, voluntary may use AR6 | Critical | Software | 2025-12-31 | Open |
| R2.7 | Require verification documentation for any custom emission factor below default values | Critical | QA | 2025-12-31 | Open |
| R2.8 | Include fugitive CH4 emission factors in scope check for natural gas equipment | High | Software | 2026-01-31 | Open |
| R2.9 | Add N2O factor lookup by combustion temperature range and equipment type | High | Software | 2026-01-31 | Open |
| R2.10 | Implement factor source hierarchy: Facility-specific > Regulatory default > Generic default | High | Software | 2026-01-31 | Open |
| R2.11 | Add scope boundary validation for emission factor applicability | High | Software | 2026-01-31 | Open |
| R2.12 | Enforce multi-gas completeness: all EPA Part 98 Subpart C calculations must include CO2, CH4, N2O | Critical | Software | 2025-12-31 | Open |
| R2.13 | Add factor applicability validation: temperature range, load range, fuel quality bounds | Medium | Software | 2026-02-28 | Open |
| R2.14 | Implement dimensional analysis validation for all emission factor unit conversions | Critical | Software | 2025-12-31 | Open |
| R2.15 | Require regulatory jurisdiction selection and validate factor source against jurisdiction | Critical | Software | 2025-12-31 | Open |
| R2.16 | Implement fuel category validation with visual confirmation for non-standard fuels | Critical | Software | 2025-12-31 | Open |
| R2.17 | Create equipment registry with validated emission factor mappings | High | Data | 2026-01-31 | Open |
| R2.18 | Integrate control technology status from operations for accurate factor selection | High | Software | 2026-01-31 | Open |
| R2.19 | Add effective date validation preventing use of future-dated factors | Medium | Software | 2026-02-28 | Open |
| R2.20 | Implement factor version caching with automatic invalidation on updates | High | Software | 2026-01-31 | Open |

---

## 8. Node 3: Calculation Engine

### 8.1 Node Description

**Design Intent:** Execute emissions calculations using validated inputs and emission factors per EPA Part 98 equations, GHG Protocol methodology, and EU ETS MRR requirements, ensuring bit-perfect deterministic results with complete provenance tracking.

**Parameters Under Study:**
- CO2 emissions (combustion, process)
- CH4 emissions (combustion, fugitive)
- N2O emissions (combustion)
- CO2 equivalent calculation (GWP application)
- Mass balance calculations
- Oxidation factor application

**Boundaries:**
- Input: Validated fuel data, selected emission factors, calculation parameters
- Output: Emissions by gas (CO2, CH4, N2O) and CO2e total with provenance hash

### 8.2 HAZOP Worksheet - Node 3

| Dev ID | Guide Word | Parameter | Deviation | Cause | Consequence | Existing Safeguards | Rec | S | L | Risk |
|--------|------------|-----------|-----------|-------|-------------|---------------------|-----|---|---|------|
| 3.1 | NO | Calculation Output | No calculation result produced | Algorithm failure, divide by zero, exception | Data gap in reporting; filing incomplete | Exception handling, error logging, fallback values | R3.1 | 4 | 2 | M |
| 3.2 | NO | Provenance Hash | No provenance tracking for calculation | Provenance system failure, hash calculation error | Cannot verify calculation for audit | Provenance validation check, backup logging | R3.2 | 4 | 2 | M |
| 3.3 | NO | Intermediate Results | Intermediate calculation steps not logged | Logging disabled, storage failure | Cannot trace calculation error source | Step-by-step logging, calculation audit trail | R3.3 | 3 | 3 | M |
| 3.4 | MORE | CO2 Result | CO2 emissions calculated higher than actual | Oxidation factor = 1.0 used (conservative), rounding up | Over-reporting by up to 1-2% | Fuel-specific oxidation factors | R3.4 | 2 | 4 | M |
| 3.5 | MORE | CH4 Result | CH4 emissions calculated higher than actual | Including unrelated sources, double counting | CH4 over-reported; inflated CO2e | Source-specific CH4 tracking | R3.5 | 2 | 3 | L |
| 3.6 | MORE | CO2e Result | CO2e higher due to GWP error | Using higher GWP values, cumulative errors | Inflated total emissions reported | GWP validation against regulatory requirement | R3.6 | 3 | 3 | M |
| 3.7 | LESS | CO2 Result | CO2 emissions calculated lower than actual | Incomplete fuel coverage, missing sources | Under-reporting; EPA non-compliance | Source completeness validation | R3.7 | 5 | 3 | VH |
| 3.8 | LESS | CH4 Result | CH4 emissions underestimated | Ignoring startup/shutdown, abnormal operations | CH4 significantly under-reported | Abnormal operation CH4 adjustment | R3.8 | 4 | 3 | M |
| 3.9 | LESS | N2O Result | N2O emissions omitted or underestimated | Missing N2O factor, calculation skipped | Total GHG under-reported (N2O is 265 GWP) | N2O completeness validation | R3.9 | 4 | 3 | M |
| 3.10 | AS WELL AS | Calculation Scope | Emissions from out-of-scope sources included | Scope boundary error, source misclassification | Inflated reported emissions | Scope boundary validation | R3.10 | 2 | 3 | L |
| 3.11 | AS WELL AS | Calculation Method | Multiple methods applied inconsistently | Mixed Tier 1/2/3 without proper aggregation | Inconsistent results; audit finding | Method consistency validation | R3.11 | 3 | 3 | M |
| 3.12 | PART OF | Calculation | Only CO2 calculated, CH4/N2O omitted | Incomplete algorithm, configuration error | Under-reporting total GHG by 1-5% | Multi-gas calculation verification | R3.12 | 4 | 3 | M |
| 3.13 | PART OF | Source Coverage | Some emission sources not calculated | Source registry incomplete, algorithm gap | Under-reporting from missing sources | Source completeness audit | R3.13 | 5 | 3 | VH |
| 3.14 | REVERSE | Sign Error | Negative emissions calculated | Sign error, sequestration credited incorrectly | Reported emissions lower than actual | Non-negative validation (except valid CCS) | R3.14 | 5 | 2 | H |
| 3.15 | REVERSE | Formula Error | Formula operands reversed (A/B vs B/A) | Coding error, formula transcription | Orders of magnitude error | Formula verification against published equations | R3.15 | 5 | 2 | VH |
| 3.16 | OTHER THAN | Calculation Method | Wrong calculation method for source type | Method/source mismatch, configuration error | Results not per regulatory requirement | Method validation per source type | R3.16 | 4 | 2 | M |
| 3.17 | OTHER THAN | Decimal Precision | Insufficient decimal precision (floating point) | Using float vs Decimal, rounding accumulation | Cumulative rounding errors in large calculations | Decimal precision enforcement | R3.17 | 3 | 4 | M |
| 3.18 | OTHER THAN | Determinism | Non-deterministic calculation results | Race condition, unordered operations, random seed | Different results for same inputs; audit failure | Determinism validation, reproducibility testing | R3.18 | 5 | 2 | H |
| 3.19 | EARLY | Calculation Timing | Calculation run before data finalized | Process timing error, premature trigger | Results based on incomplete data | Data finalization check before calculation | R3.19 | 3 | 3 | M |
| 3.20 | LATE | Calculation Completion | Calculation takes too long | Performance issue, large data volume | Delayed reporting; filing deadline risk | Performance monitoring, timeout handling | R3.20 | 3 | 3 | M |

### 8.3 Node 3 - Recommendations

| Rec ID | Description | Priority | Owner | Due Date | Status |
|--------|-------------|----------|-------|----------|--------|
| R3.1 | Implement robust exception handling with fallback to safe values and operator notification | High | Software | 2026-01-31 | Open |
| R3.2 | Add provenance hash validation with backup audit trail to separate system | Critical | Software | 2025-12-31 | Open |
| R3.3 | Implement detailed calculation step logging: inputs, formula, intermediate values, output | High | Software | 2026-01-31 | Open |
| R3.4 | Deploy fuel-specific oxidation factors per EPA Part 98 Table C-1 | Medium | Data | 2026-02-28 | Open |
| R3.7 | Implement source completeness validation against equipment registry before final reporting | Critical | Software | 2025-12-31 | Open |
| R3.8 | Add startup/shutdown adjustment factors for CH4 per EPA AP-42 methodology | High | Software | 2026-01-31 | Open |
| R3.9 | Enforce N2O calculation for all combustion sources per EPA Part 98 Subpart C | Critical | Software | 2025-12-31 | Open |
| R3.11 | Require consistent Tier approach within source categories with documentation | High | Software | 2026-01-31 | Open |
| R3.12 | Add multi-gas calculation completeness check: CO2 + CH4 + N2O for all combustion | Critical | Software | 2025-12-31 | Open |
| R3.13 | Implement automated source registry comparison before annual filing | Critical | Software | 2025-12-31 | Open |
| R3.14 | Add non-negative emissions validation with exception for verified carbon capture | Critical | Software | 2025-12-31 | Open |
| R3.15 | Implement formula verification with independent calculation path for high-risk calculations | Critical | Software | 2025-12-31 | Open |
| R3.16 | Create calculation method matrix by source type per EPA Part 98 requirements | High | Data | 2026-01-31 | Open |
| R3.17 | Enforce Python Decimal type for all emissions calculations with minimum 10 decimal places | High | Software | 2026-01-31 | Open |
| R3.18 | Implement determinism testing: same inputs must produce identical outputs | Critical | QA | 2025-12-31 | Open |
| R3.19 | Add data finalization status check before running calculations for regulatory filing | High | Software | 2026-01-31 | Open |
| R3.20 | Implement calculation performance monitoring with timeout and chunking for large datasets | Medium | Software | 2026-02-28 | Open |

---

## 9. Node 4: Unit Conversion Module

### 9.1 Node Description

**Design Intent:** Convert between various measurement units (mass, volume, energy) accurately and consistently to ensure emissions calculations use compatible units per regulatory requirements.

**Parameters Under Study:**
- Mass units: kg, lb, tonne (metric ton), short ton
- Volume units: m3, scf (standard cubic feet), BBL (barrel)
- Energy units: GJ, MMBtu, MWh, kWh
- Temperature reference conditions (15C, 60F for gas volumes)
- Density conversions

**Boundaries:**
- Input: Values in source units
- Output: Values converted to target units with conversion factor audit trail

### 9.2 HAZOP Worksheet - Node 4

| Dev ID | Guide Word | Parameter | Deviation | Cause | Consequence | Existing Safeguards | Rec | S | L | Risk |
|--------|------------|-----------|-----------|-------|-------------|---------------------|-----|---|---|------|
| 4.1 | NO | Conversion Applied | No unit conversion when needed | Missing conversion logic, unit not recognized | Wrong units in calculation; orders of magnitude error | Unit validation at input | R4.1 | 5 | 2 | H |
| 4.2 | NO | Conversion Factor | Missing conversion factor for unit pair | Incomplete conversion table, new unit | Conversion fails; calculation blocked | Comprehensive conversion factor database | R4.2 | 3 | 2 | L |
| 4.3 | MORE | Conversion Factor | Conversion factor too high | Data entry error, decimal place error | Values inflated by conversion error | Conversion factor validation against known values | R4.3 | 4 | 2 | M |
| 4.4 | MORE | Precision | Excessive precision carried forward | Over-precision in intermediate values | False precision in reported values | Significant figures enforcement | R4.4 | 2 | 4 | M |
| 4.5 | LESS | Conversion Factor | Conversion factor too low | Data entry error, wrong direction | Values deflated by conversion error | Conversion factor validation | R4.5 | 4 | 2 | M |
| 4.6 | LESS | Precision | Premature rounding loses precision | Rounding at intermediate steps | Cumulative precision loss | Maintain precision until final output | R4.6 | 3 | 4 | M |
| 4.7 | AS WELL AS | Conversion | Multiple conversions applied (double conversion) | Conversion applied twice, system and manual | Values grossly incorrect (squared factor) | Conversion tracking, single conversion enforcement | R4.7 | 5 | 2 | H |
| 4.8 | AS WELL AS | Temperature Correction | Temperature correction applied when not needed | Over-correction, misunderstanding reference | Volume/density incorrectly adjusted | Reference condition tracking | R4.8 | 3 | 3 | M |
| 4.9 | PART OF | Conversion | Partial conversion (only one step of two-step conversion) | Multi-step conversion incomplete | Intermediate units in output | Conversion completeness validation | R4.9 | 4 | 2 | M |
| 4.10 | REVERSE | Conversion Direction | Conversion applied in reverse (divide vs multiply) | Direction error, formula mistake | Inverse of correct value | Dimensional analysis validation | R4.10 | 5 | 2 | VH |
| 4.11 | OTHER THAN | Unit System | Metric vs Imperial confusion | Country/system mismatch, data source | Factor of 2.2 (lb/kg) or 3.28 (ft/m) error | Unit system standardization | R4.11 | 4 | 3 | M |
| 4.12 | OTHER THAN | Reference Conditions | Wrong reference conditions for gas volume | 15C/60F confusion, pressure mismatch | 5-10% volume error | Reference condition validation | R4.12 | 3 | 3 | M |
| 4.13 | OTHER THAN | Conversion Factor | Wrong conversion factor used (similar units) | GJ vs Btu confusion, factor mismatch | Significant calculation error | Unit disambiguation, factor verification | R4.13 | 4 | 2 | M |
| 4.14 | OTHER THAN | Density | Wrong density for volume to mass conversion | Temperature effect, fuel grade variation | Mass calculation error | Density validation by fuel type and temperature | R4.14 | 3 | 3 | M |

### 9.3 Node 4 - Recommendations

| Rec ID | Description | Priority | Owner | Due Date | Status |
|--------|-------------|----------|-------|----------|--------|
| R4.1 | Implement mandatory unit validation at all data input points with rejection of unrecognized units | Critical | Software | 2025-12-31 | Open |
| R4.2 | Deploy comprehensive conversion factor database covering all EPA Part 98 units | High | Data | 2026-01-31 | Open |
| R4.3 | Add conversion factor range validation: no factor should be >10^6 or <10^-6 without warning | High | Software | 2026-01-31 | Open |
| R4.4 | Implement significant figures tracking per EPA reporting requirements | Medium | Software | 2026-02-28 | Open |
| R4.6 | Maintain full precision (Decimal type) through calculation, round only at output | High | Software | 2026-01-31 | Open |
| R4.7 | Add conversion tracking to prevent double conversion: flag already-converted values | Critical | Software | 2025-12-31 | Open |
| R4.8 | Require explicit reference condition specification for all volume data | High | Software | 2026-01-31 | Open |
| R4.9 | Validate conversion completeness: source unit must convert to target unit in one logical step | High | Software | 2026-01-31 | Open |
| R4.10 | Implement dimensional analysis validation: output units must match expected units | Critical | Software | 2025-12-31 | Open |
| R4.11 | Standardize on SI units internally with explicit conversion at input/output boundaries | High | Software | 2026-01-31 | Open |
| R4.12 | Require reference conditions (T, P) for all gas volume conversions with default to 15C, 101.325 kPa | High | Software | 2026-01-31 | Open |
| R4.13 | Add unit disambiguation for similar units (GJ vs MMBtu, m3 vs scf) with user confirmation | High | Software | 2026-01-31 | Open |
| R4.14 | Implement temperature-corrected density lookup for major fuels | Medium | Data | 2026-02-28 | Open |

---

## 10. Node 5: Uncertainty Propagation

### 10.1 Node Description

**Design Intent:** Quantify and propagate uncertainty in emissions calculations per GHG Protocol Chapter 6 guidance, providing confidence intervals for reported values and meeting regulatory quality requirements.

**Parameters Under Study:**
- Activity data uncertainty
- Emission factor uncertainty
- Combined uncertainty (root sum of squares)
- Confidence intervals (95%)
- Tier-based uncertainty defaults
- Monte Carlo simulation parameters

**Boundaries:**
- Input: Individual uncertainties for activity data and emission factors
- Output: Combined uncertainty, confidence intervals, data quality indicators

### 10.2 HAZOP Worksheet - Node 5

| Dev ID | Guide Word | Parameter | Deviation | Cause | Consequence | Existing Safeguards | Rec | S | L | Risk |
|--------|------------|-----------|-----------|-------|-------------|---------------------|-----|---|---|------|
| 5.1 | NO | Uncertainty Calculated | No uncertainty provided with results | Uncertainty calculation skipped, system error | Cannot assess data quality; GHG Protocol non-compliance | Uncertainty calculation enforcement | R5.1 | 4 | 2 | M |
| 5.2 | NO | Activity Data Uncertainty | Activity data uncertainty not specified | Meter accuracy unknown, data source undocumented | Cannot calculate combined uncertainty | Default uncertainty by data source type | R5.2 | 3 | 3 | M |
| 5.3 | NO | Emission Factor Uncertainty | EF uncertainty not available | Factor source lacks uncertainty data | Total uncertainty underestimated | Default uncertainty by tier level | R5.3 | 3 | 3 | M |
| 5.4 | MORE | Uncertainty | Uncertainty overestimated | Conservative assumptions, compounding errors | Reported range too wide; reduced credibility | Uncertainty reasonableness check | R5.4 | 2 | 3 | L |
| 5.5 | MORE | Monte Carlo Iterations | Excessive iterations | Performance not optimized | Calculation delay, resource waste | Iteration convergence check | R5.5 | 1 | 4 | L |
| 5.6 | LESS | Uncertainty | Uncertainty underestimated | Ignoring correlation, incomplete error sources | False precision; regulatory risk | Comprehensive uncertainty source list | R5.6 | 4 | 3 | H |
| 5.7 | LESS | Confidence Level | Lower confidence level than required | Configuration error, 90% vs 95% | Non-compliance with reporting requirements | Confidence level validation | R5.7 | 3 | 2 | L |
| 5.8 | AS WELL AS | Error Sources | Correlated errors treated as independent | Ignoring systematic errors | Uncertainty underestimated | Correlation matrix for error sources | R5.8 | 4 | 3 | M |
| 5.9 | AS WELL AS | Uncertainty Types | Type A and Type B uncertainties mixed incorrectly | Statistical vs systematic confusion | Combined uncertainty incorrect | Uncertainty type classification | R5.9 | 3 | 3 | M |
| 5.10 | PART OF | Uncertainty Sources | Not all uncertainty sources included | Incomplete analysis, model simplification | Under-reporting total uncertainty | Uncertainty source checklist | R5.10 | 4 | 3 | M |
| 5.11 | PART OF | Propagation | Uncertainty propagation incomplete | Only some calculation steps included | Reported uncertainty too narrow | Full calculation path uncertainty | R5.11 | 4 | 3 | M |
| 5.12 | REVERSE | Uncertainty Sign | Uncertainty applied asymmetrically incorrectly | Bias direction error | Confidence interval shifted | Symmetric uncertainty validation | R5.12 | 3 | 2 | L |
| 5.13 | OTHER THAN | Propagation Method | Wrong propagation method used | RSS vs linear vs Monte Carlo | Uncertainty calculation incorrect | Method validation per GHG Protocol | R5.13 | 3 | 3 | M |
| 5.14 | OTHER THAN | Distribution | Wrong distribution assumed | Normal vs lognormal vs uniform | Confidence interval bounds incorrect | Distribution validation for emission data | R5.14 | 3 | 3 | M |
| 5.15 | OTHER THAN | Tier Level | Wrong tier defaults applied | Tier mismatch, data quality confusion | Default uncertainty inappropriate | Tier level validation | R5.15 | 3 | 3 | M |

### 10.3 Node 5 - Recommendations

| Rec ID | Description | Priority | Owner | Due Date | Status |
|--------|-------------|----------|-------|----------|--------|
| R5.1 | Enforce uncertainty calculation for all reported emissions; reject outputs without uncertainty | Critical | Software | 2025-12-31 | Open |
| R5.2 | Implement default activity data uncertainty by source: meters +/-2%, estimates +/-20%, invoices +/-5% | High | Software | 2026-01-31 | Open |
| R5.3 | Deploy tier-based default EF uncertainties: Tier 1 +/-30%, Tier 2 +/-10%, Tier 3 +/-5% | High | Software | 2026-01-31 | Open |
| R5.4 | Add uncertainty reasonableness check: flag if combined uncertainty >50% | Medium | Software | 2026-02-28 | Open |
| R5.6 | Implement comprehensive uncertainty source list per GHG Protocol Chapter 6 | High | Software | 2026-01-31 | Open |
| R5.7 | Configure 95% confidence level as default per GHG Protocol | High | Software | 2026-01-31 | Open |
| R5.8 | Add correlation handling for common error sources (same meter, same EF source) | High | Software | 2026-01-31 | Open |
| R5.9 | Classify uncertainties as Type A (statistical) or Type B (systematic) per GUM | Medium | Software | 2026-02-28 | Open |
| R5.10 | Create uncertainty source checklist: measurement, sampling, EF, calculation, reporting | High | Software | 2026-01-31 | Open |
| R5.11 | Implement full uncertainty propagation through all calculation steps | High | Software | 2026-01-31 | Open |
| R5.13 | Use RSS method for uncorrelated sources, Monte Carlo for complex correlations | High | Software | 2026-01-31 | Open |
| R5.14 | Apply lognormal distribution for emission factors, normal for activity data | Medium | Software | 2026-02-28 | Open |
| R5.15 | Validate tier level against actual data quality before applying defaults | High | Software | 2026-01-31 | Open |

---

## 11. Node 6: Result Validation and Output

### 11.1 Node Description

**Design Intent:** Validate calculated emissions against expected ranges, regulatory thresholds, and historical data before generating outputs for reporting, ensuring data quality and regulatory compliance.

**Parameters Under Study:**
- Result range validation
- Regulatory threshold comparison
- Historical trend analysis
- Data quality indicators
- Output formatting (EPA, EU ETS, GHG Protocol)
- Digital signature and audit trail

**Boundaries:**
- Input: Calculated emissions with uncertainty
- Output: Validated, formatted emissions reports with compliance status

### 11.2 HAZOP Worksheet - Node 6

| Dev ID | Guide Word | Parameter | Deviation | Cause | Consequence | Existing Safeguards | Rec | S | L | Risk |
|--------|------------|-----------|-----------|-------|-------------|---------------------|-----|---|---|------|
| 6.1 | NO | Validation | No validation performed on results | Validation step bypassed, system error | Invalid results reported | Validation gate enforcement | R6.1 | 5 | 2 | H |
| 6.2 | NO | Threshold Check | Regulatory threshold not checked | Threshold configuration missing | Exceeding threshold without alert | Threshold monitoring system | R6.2 | 4 | 3 | M |
| 6.3 | NO | Historical Comparison | No comparison to prior periods | Historical data unavailable | Anomalies not detected | Historical data baseline | R6.3 | 3 | 3 | M |
| 6.4 | NO | Output Generation | No output produced | Format error, system failure | Reporting deadline missed | Output generation monitoring | R6.4 | 4 | 2 | M |
| 6.5 | MORE | Threshold Margin | Threshold set too low (false alarms) | Conservative configuration | Frequent false alerts; alert fatigue | Threshold calibration | R6.5 | 2 | 4 | M |
| 6.6 | MORE | Output Precision | More decimal places than warranted | Over-precision in formatting | False precision implied | Significant figures matching uncertainty | R6.6 | 2 | 4 | M |
| 6.7 | LESS | Threshold Margin | Threshold set too high (missed violations) | Permissive configuration | Real exceedance not detected | Threshold validation against limits | R6.7 | 5 | 2 | H |
| 6.8 | LESS | Validation Coverage | Only partial validation (some checks skipped) | Performance optimization, configuration | Invalid data passed through | Validation completeness check | R6.8 | 4 | 3 | M |
| 6.9 | LESS | Output Detail | Insufficient detail for regulatory filing | Format configuration incomplete | Regulatory rejection; refiling required | Format template validation | R6.9 | 3 | 3 | M |
| 6.10 | AS WELL AS | Output | Extra data included in output | Aggregation error, data leakage | Confidential data exposed; confusion | Output content validation | R6.10 | 3 | 2 | L |
| 6.11 | AS WELL AS | Warnings | Multiple redundant warnings | Over-alerting, threshold overlap | Alert fatigue; critical alerts missed | Alert deduplication | R6.11 | 2 | 4 | M |
| 6.12 | PART OF | Output | Incomplete output (missing fields) | Format error, field mapping issue | Regulatory rejection; data gaps | Output completeness validation | R6.12 | 4 | 3 | M |
| 6.13 | PART OF | Validation | Validation checks output but misses specific errors | Incomplete validation rules | Specific errors not caught | Comprehensive validation rule set | R6.13 | 4 | 3 | M |
| 6.14 | REVERSE | Threshold Comparison | Threshold comparison logic inverted | Logic error, sign error | Alerts for normal, silence for abnormal | Threshold logic verification | R6.14 | 4 | 2 | M |
| 6.15 | OTHER THAN | Output Format | Wrong output format for destination | Configuration error, format mismatch | Filing rejected; manual correction | Format validation per regulatory requirement | R6.15 | 4 | 3 | M |
| 6.16 | OTHER THAN | Threshold | Wrong threshold value used | Configuration error, outdated limit | Wrong compliance assessment | Threshold configuration audit | R6.16 | 4 | 3 | M |
| 6.17 | OTHER THAN | Report Period | Output for wrong reporting period | Date handling error | Wrong period reported | Period validation in output | R6.17 | 4 | 2 | M |
| 6.18 | OTHER THAN | Facility | Output aggregated to wrong facility | Entity mapping error | Wrong facility reported | Facility ID validation | R6.18 | 5 | 2 | H |
| 6.19 | EARLY | Output Generation | Output generated before all data final | Process timing, premature report | Preliminary data reported as final | Data finalization gate | R6.19 | 4 | 3 | M |
| 6.20 | LATE | Output Generation | Output delayed past deadline | System issues, data delays | Regulatory filing deadline missed | Deadline monitoring, early warning | R6.20 | 4 | 3 | M |

### 11.3 Node 6 - Recommendations

| Rec ID | Description | Priority | Owner | Due Date | Status |
|--------|-------------|----------|-------|----------|--------|
| R6.1 | Implement mandatory validation gate with configurable rules before output release | Critical | Software | 2025-12-31 | Open |
| R6.2 | Add regulatory threshold monitoring: EPA Part 98 25,000 tCO2e, EU ETS thresholds | Critical | Software | 2025-12-31 | Open |
| R6.3 | Implement historical comparison with variance alerting (>10% change from prior period) | High | Software | 2026-01-31 | Open |
| R6.4 | Add output generation monitoring with automatic retry and escalation | High | Software | 2026-01-31 | Open |
| R6.5 | Calibrate alert thresholds to minimize false positives while catching real issues | Medium | Operations | 2026-02-28 | Open |
| R6.6 | Match output decimal places to uncertainty: report to fewer significant figures than uncertainty | High | Software | 2026-01-31 | Open |
| R6.7 | Set regulatory thresholds conservatively (95% of limit) to provide early warning | Critical | Software | 2025-12-31 | Open |
| R6.8 | Enforce complete validation: all defined rules must execute before output | High | Software | 2026-01-31 | Open |
| R6.9 | Validate output against regulatory format templates (EPA e-GGRT, EU ETS XML) | High | Software | 2026-01-31 | Open |
| R6.12 | Add output field completeness check against regulatory requirements | High | Software | 2026-01-31 | Open |
| R6.13 | Implement comprehensive validation rules: range, consistency, completeness, cross-check | High | Software | 2026-01-31 | Open |
| R6.14 | Add threshold comparison logic verification with test cases | High | QA | 2026-01-31 | Open |
| R6.15 | Require format selection and validation per regulatory destination | High | Software | 2026-01-31 | Open |
| R6.16 | Implement threshold configuration management with version control | High | Data | 2026-01-31 | Open |
| R6.17 | Add reporting period validation: dates must be within valid range | High | Software | 2026-01-31 | Open |
| R6.18 | Implement facility ID validation against master facility registry | Critical | Software | 2025-12-31 | Open |
| R6.19 | Add data finalization check: require explicit final status before regulatory output | High | Software | 2026-01-31 | Open |
| R6.20 | Implement filing deadline monitoring with escalating alerts (30, 14, 7, 3, 1 days) | High | Software | 2026-01-31 | Open |

---

## 12. Risk Ranking Matrix

### 12.1 Risk Matrix Definition

```
                           S E V E R I T Y
                     (Regulatory/Financial Impact)

                1           2           3           4           5
            Negligible    Minor     Moderate     Major    Catastrophic
            (<$10K)      ($10K-$100K) ($100K-$1M) ($1M-$10M) (>$10M/criminal)

       5    +-----------+-----------+-----------+-----------+-----------+
  L    F    |           |           |           |           |           |
  I    r    |    LOW    |  MEDIUM   |   HIGH    | VERY HIGH | VERY HIGH |
  K    e    |   (5)     |   (10)    |   (15)    |   (20)    |   (25)    |
  E    q    +-----------+-----------+-----------+-----------+-----------+
  L    u  4 |           |           |           |           |           |
  I    e    |    LOW    |  MEDIUM   |  MEDIUM   |   HIGH    | VERY HIGH |
  H    n    |   (4)     |   (8)     |   (12)    |   (16)    |   (20)    |
  O    t    +-----------+-----------+-----------+-----------+-----------+
  O      3  |           |           |           |           |           |
  D         |    LOW    |   LOW     |  MEDIUM   |  MEDIUM   |   HIGH    |
            |   (3)     |   (6)     |   (9)     |   (12)    |   (15)    |
            +-----------+-----------+-----------+-----------+-----------+
         2  |           |           |           |           |           |
            |    LOW    |   LOW     |   LOW     |  MEDIUM   |  MEDIUM   |
            |   (2)     |   (4)     |   (6)     |   (8)     |   (10)    |
            +-----------+-----------+-----------+-----------+-----------+
         1  |           |           |           |           |           |
            |    LOW    |   LOW     |   LOW     |   LOW     |  MEDIUM   |
            |   (1)     |   (2)     |   (3)     |   (4)     |   (5)     |
            +-----------+-----------+-----------+-----------+-----------+
```

### 12.2 Severity Definitions (Regulatory/Financial Context)

| Level | Description | Regulatory Impact | Financial Impact |
|-------|-------------|-------------------|------------------|
| **1 - Negligible** | Minor operational inconvenience | No regulatory impact | <$10,000 |
| **2 - Minor** | Small calculation error, easily corrected | Minor finding, no penalty | $10,000 - $100,000 |
| **3 - Moderate** | Moderate error requiring correction | Audit finding, warning letter | $100,000 - $1,000,000 |
| **4 - Major** | Significant under/over-reporting | Enforcement action, penalty | $1,000,000 - $10,000,000 |
| **5 - Catastrophic** | Material misstatement, fraud | Criminal referral, permit revocation | >$10,000,000 |

### 12.3 Likelihood Definitions

| Level | Description | Frequency |
|-------|-------------|-----------|
| **1** | Rare | Less than once per 10 years |
| **2** | Unlikely | Once per 5-10 years |
| **3** | Possible | Once per 1-5 years |
| **4** | Likely | Once per year |
| **5** | Almost Certain | Multiple times per year |

### 12.4 Risk Summary by Node

| Node | VH | H | M | L | Total Deviations |
|------|----|----|---|---|------------------|
| Node 1: Fuel Data Input Processing | 0 | 3 | 12 | 4 | 19 |
| Node 2: Emission Factor Selection | 2 | 2 | 14 | 2 | 20 |
| Node 3: Calculation Engine | 2 | 4 | 10 | 4 | 20 |
| Node 4: Unit Conversion Module | 1 | 3 | 8 | 2 | 14 |
| Node 5: Uncertainty Propagation | 0 | 1 | 10 | 4 | 15 |
| Node 6: Result Validation & Output | 0 | 3 | 14 | 3 | 20 |
| **Total** | **5** | **16** | **68** | **19** | **108** |

---

## 13. Risk Register

### 13.1 Very High Risk Items

| Risk ID | Node | Dev ID | Deviation | Risk Score | Root Cause | Key Mitigation |
|---------|------|--------|-----------|------------|------------|----------------|
| VH-001 | 2 | 2.16 | Wrong fuel category for factor lookup | 15 (S5 x L3) | Fuel misclassification | Fuel category validation with confirmation |
| VH-002 | 3 | 3.7 | CO2 calculated lower than actual | 15 (S5 x L3) | Incomplete source coverage | Source completeness validation |
| VH-003 | 3 | 3.13 | Some emission sources not calculated | 15 (S5 x L3) | Source registry incomplete | Equipment registry audit |
| VH-004 | 3 | 3.15 | Formula operands reversed | 10 (S5 x L2) | Coding error | Dual-path verification |
| VH-005 | 4 | 4.10 | Unit conversion direction reversed | 10 (S5 x L2) | Formula error | Dimensional analysis |

### 13.2 High Risk Items

| Risk ID | Node | Dev ID | Deviation | Risk Score | Root Cause | Key Mitigation |
|---------|------|--------|-----------|------------|------------|----------------|
| H-001 | 1 | 1.1 | No fuel data received for period | 15 (S5 x L3) | Data pipeline failure | Data completeness monitoring |
| H-002 | 1 | 1.7 | Fuel quantity lower than actual | 15 (S5 x L3) | Meter under-reading | Purchase reconciliation |
| H-003 | 1 | 1.16 | Wrong units for fuel quantity | 15 (S5 x L3) | Unit confusion | Unit validation |
| H-004 | 2 | 2.14 | Factor units inverted | 10 (S5 x L2) | Data entry error | Dimensional analysis |
| H-005 | 2 | 2.15 | Wrong regulatory factor source | 10 (S5 x L2) | Jurisdiction error | Jurisdiction validation |
| H-006 | 3 | 3.14 | Negative emissions calculated | 10 (S5 x L2) | Sign error | Non-negative validation |
| H-007 | 3 | 3.18 | Non-deterministic calculation | 10 (S5 x L2) | Race condition | Determinism testing |
| H-008 | 4 | 4.1 | No unit conversion when needed | 10 (S5 x L2) | Missing logic | Unit validation |
| H-009 | 4 | 4.7 | Double conversion applied | 10 (S5 x L2) | Conversion tracking failure | Conversion flags |
| H-010 | 5 | 5.6 | Uncertainty underestimated | 12 (S4 x L3) | Ignoring correlations | Correlation handling |
| H-011 | 6 | 6.1 | No validation performed | 10 (S5 x L2) | Validation bypassed | Validation enforcement |
| H-012 | 6 | 6.7 | Threshold set too high | 10 (S5 x L2) | Configuration error | Threshold audit |
| H-013 | 6 | 6.18 | Output aggregated to wrong facility | 10 (S5 x L2) | Entity mapping error | Facility ID validation |

### 13.3 Recommended Actions by Priority

| Priority | Action Category | Count | Target Date |
|----------|-----------------|-------|-------------|
| Critical | Must complete before production | 18 | 2025-12-31 |
| High | Required for regulatory compliance | 42 | 2026-01-31 |
| Medium | Important for data quality | 28 | 2026-02-28 |
| Low | Beneficial improvements | 8 | 2026-Q2 |

---

## 14. Action Tracking Table

### 14.1 Critical Priority Actions (Due: 2025-12-31)

| Action ID | Description | Owner | Node | Risk Addressed | Due Date | Status |
|-----------|-------------|-------|------|----------------|----------|--------|
| A-001 | Implement dual-path verification for all EPA Part 98 emissions calculations | Software | 3 | VH-002, VH-004 | 2025-12-31 | Open |
| A-002 | Deploy fuel category validation with visual user confirmation for non-standard fuels | Software | 2 | VH-001 | 2025-12-31 | Open |
| A-003 | Implement source completeness validation against equipment registry | Software | 3 | VH-003 | 2025-12-31 | Open |
| A-004 | Add dimensional analysis validation for all unit conversions | Software | 4 | VH-005, H-004 | 2025-12-31 | Open |
| A-005 | Require regulatory jurisdiction selection for emission factor source | Software | 2 | H-005 | 2025-12-31 | Open |
| A-006 | Enforce multi-gas calculation completeness (CO2, CH4, N2O) | Software | 3 | 3.12 | 2025-12-31 | Open |
| A-007 | Implement non-negative emissions validation with CCS exception | Software | 3 | H-006 | 2025-12-31 | Open |
| A-008 | Deploy determinism testing framework for calculation reproducibility | QA | 3 | H-007 | 2025-12-31 | Open |
| A-009 | Add mandatory unit validation at all data input points | Software | 4 | H-003, H-008 | 2025-12-31 | Open |
| A-010 | Implement conversion tracking to prevent double conversion | Software | 4 | H-009 | 2025-12-31 | Open |
| A-011 | Enforce uncertainty calculation for all reported emissions | Software | 5 | 5.1 | 2025-12-31 | Open |
| A-012 | Implement mandatory validation gate before output release | Software | 6 | H-011 | 2025-12-31 | Open |
| A-013 | Add regulatory threshold monitoring (EPA 25,000 tCO2e) | Software | 6 | H-012 | 2025-12-31 | Open |
| A-014 | Implement facility ID validation against master registry | Software | 6 | H-013 | 2025-12-31 | Open |
| A-015 | Deploy automated data completeness monitoring with gap alerts | Data | 1 | H-001 | 2025-12-31 | Open |
| A-016 | Implement fuel purchase vs consumption reconciliation | Software | 1 | H-002 | 2025-12-31 | Open |
| A-017 | Add provenance hash validation with backup audit trail | Software | 3 | 3.2 | 2025-12-31 | Open |
| A-018 | Configure regulatory-specific GWP values (AR4 for EPA, AR5 for EU ETS) | Software | 2 | 2.6 | 2025-12-31 | Open |

### 14.2 High Priority Actions (Due: 2026-01-31)

| Action ID | Description | Owner | Node | Risk Addressed | Due Date | Status |
|-----------|-------------|-------|------|----------------|----------|--------|
| A-019 | Deploy EPA Part 98 Table C-1 default HHV lookup with version control | Data | 1 | 1.3 | 2026-01-31 | Open |
| A-020 | Implement fuel type validation against EPA categories | Software | 1 | 1.2, 1.15 | 2026-01-31 | Open |
| A-021 | Add duplicate record detection using composite key | Software | 1 | 1.10 | 2026-01-31 | Open |
| A-022 | Require EPA WebFIRE synchronization with version tracking | Software | 2 | 2.2 | 2026-01-31 | Open |
| A-023 | Add tier-based default uncertainties per GHG Protocol | Software | 5 | 5.2, 5.3 | 2026-01-31 | Open |
| A-024 | Implement correlation handling for common error sources | Software | 5 | H-010 | 2026-01-31 | Open |
| A-025 | Add historical comparison with variance alerting (>10%) | Software | 6 | 6.3 | 2026-01-31 | Open |
| A-026 | Validate output against regulatory format templates | Software | 6 | 6.9, 6.15 | 2026-01-31 | Open |
| A-027 | Implement detailed calculation step logging | Software | 3 | 3.3 | 2026-01-31 | Open |
| A-028 | Enforce Python Decimal type for calculations | Software | 3 | 3.17 | 2026-01-31 | Open |
| A-029 | Create comprehensive conversion factor database | Data | 4 | 4.2 | 2026-01-31 | Open |
| A-030 | Require reference conditions for all gas volumes | Software | 4 | 4.8, 4.12 | 2026-01-31 | Open |
| A-031 | Implement comprehensive uncertainty source list | Software | 5 | 5.10 | 2026-01-31 | Open |
| A-032 | Add output field completeness validation | Software | 6 | 6.12 | 2026-01-31 | Open |
| A-033 | Implement filing deadline monitoring with alerts | Software | 6 | 6.20 | 2026-01-31 | Open |
| A-034 | Add biogenic/fossil carbon specification for biomass | Software | 1 | 1.9 | 2026-01-31 | Open |
| A-035 | Create equipment registry with EF mappings | Data | 2 | 2.17 | 2026-01-31 | Open |
| A-036 | Implement factor source hierarchy configuration | Software | 2 | 2.10 | 2026-01-31 | Open |
| A-037 | Add startup/shutdown CH4 adjustment factors | Software | 3 | 3.8 | 2026-01-31 | Open |
| A-038 | Implement data finalization workflow | Software | 1, 3 | 1.18, 3.19 | 2026-01-31 | Open |
| A-039 | Add threshold comparison logic verification | QA | 6 | 6.14 | 2026-01-31 | Open |
| A-040 | Implement threshold configuration management | Data | 6 | 6.16 | 2026-01-31 | Open |

### 14.3 Medium Priority Actions (Due: 2026-02-28)

| Action ID | Description | Owner | Node | Risk Addressed | Due Date | Status |
|-----------|-------------|-------|------|----------------|----------|--------|
| A-041 | Add HHV range validation by fuel type | Software | 1 | 1.5 | 2026-02-28 | Open |
| A-042 | Validate carbon content against fuel type norms | Software | 1 | 1.6 | 2026-02-28 | Open |
| A-043 | Implement fuel blend handling with weighted EF | Software | 1 | 1.11 | 2026-02-28 | Open |
| A-044 | Create data quality dashboard | Data | 1 | 1.12 | 2026-02-28 | Open |
| A-045 | Add factor applicability validation | Software | 2 | 2.13 | 2026-02-28 | Open |
| A-046 | Deploy fuel-specific oxidation factors | Data | 3 | 3.4 | 2026-02-28 | Open |
| A-047 | Implement calculation performance monitoring | Software | 3 | 3.20 | 2026-02-28 | Open |
| A-048 | Add significant figures tracking | Software | 4 | 4.4 | 2026-02-28 | Open |
| A-049 | Implement temperature-corrected density lookup | Data | 4 | 4.14 | 2026-02-28 | Open |
| A-050 | Add uncertainty reasonableness check (>50%) | Software | 5 | 5.4 | 2026-02-28 | Open |
| A-051 | Classify uncertainties as Type A/B per GUM | Software | 5 | 5.9 | 2026-02-28 | Open |
| A-052 | Apply lognormal distribution for EF uncertainty | Software | 5 | 5.14 | 2026-02-28 | Open |
| A-053 | Calibrate alert thresholds | Operations | 6 | 6.5 | 2026-02-28 | Open |
| A-054 | Add alert deduplication | Software | 6 | 6.11 | 2026-02-28 | Open |

---

## 15. Appendices

### Appendix A: Abbreviations

| Abbreviation | Definition |
|--------------|------------|
| AP-42 | EPA Compilation of Air Pollutant Emission Factors |
| AR4/AR5/AR6 | IPCC Assessment Reports (4th, 5th, 6th) |
| CF | Carbon Fraction |
| CFR | Code of Federal Regulations |
| CEMS | Continuous Emissions Monitoring System |
| CO2e | Carbon Dioxide Equivalent |
| EF | Emission Factor |
| e-GGRT | EPA electronic Greenhouse Gas Reporting Tool |
| EU ETS | European Union Emissions Trading System |
| GHG | Greenhouse Gas |
| GUM | Guide to the Expression of Uncertainty in Measurement |
| GWP | Global Warming Potential |
| HHV | Higher Heating Value |
| IPCC | Intergovernmental Panel on Climate Change |
| LHV | Lower Heating Value |
| MRR | Monitoring and Reporting Regulation |
| MRV | Monitoring, Reporting, and Verification |
| RSS | Root Sum of Squares |
| scf | Standard Cubic Feet |
| tCO2e | Tonnes of CO2 Equivalent |
| UQ | Uncertainty Quantification |

### Appendix B: Reference Documents

| Document ID | Title | Version | Relevance |
|-------------|-------|---------|-----------|
| IEC 61882:2016 | Hazard and Operability Studies | 2016 | Primary HAZOP methodology |
| ISO 17776:2016 | Petroleum and Natural Gas - Safety | 2016 | Risk assessment framework |
| 40 CFR Part 98 | Mandatory GHG Reporting Rule | 2024 | EPA reporting requirements |
| EPA AP-42 | Compilation of Air Pollutant Emission Factors | 5th Ed. | Emission factor database |
| GHG Protocol | Corporate Accounting and Reporting Standard | Rev. 2015 | Calculation methodology |
| EU 2018/2066 | EU ETS MRR Regulation | 2018 | EU reporting requirements |
| IPCC 2006 Guidelines | National GHG Inventories | 2006 (2019 Ref.) | Tier methodology |
| ISO 14064-1:2018 | GHG Quantification and Reporting | 2018 | Organization-level GHG |
| GUM | Guide to Expression of Uncertainty | 2008 | Uncertainty methodology |

### Appendix C: EPA Part 98 Emission Factor Reference

| Fuel Type | CO2 Factor (kg/MMBtu) | CH4 Factor (kg/MMBtu) | N2O Factor (kg/MMBtu) | Source |
|-----------|----------------------|----------------------|----------------------|--------|
| Natural Gas | 53.06 | 0.001 | 0.0001 | Table C-1 |
| Distillate Fuel Oil | 73.96 | 0.003 | 0.0006 | Table C-1 |
| Residual Fuel Oil | 75.10 | 0.003 | 0.0006 | Table C-1 |
| Bituminous Coal | 93.28 | 0.011 | 0.0016 | Table C-1 |
| Sub-bituminous Coal | 97.17 | 0.011 | 0.0016 | Table C-1 |
| Lignite | 97.72 | 0.011 | 0.0016 | Table C-1 |
| Propane | 62.87 | 0.001 | 0.0001 | Table C-1 |

### Appendix D: GWP Values by Regulatory Framework

| Gas | EPA Part 98 (AR4) | EU ETS (AR5) | GHG Protocol (AR6 optional) |
|-----|-------------------|--------------|----------------------------|
| CO2 | 1 | 1 | 1 |
| CH4 | 25 | 28 | 28 |
| N2O | 298 | 265 | 265 |

### Appendix E: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-06 | GL-SafetyEngineer | Initial release per TASK-202 |

---

**Document End**

*This document is part of the GreenLang Safety Critical Documentation Package.*
*Compliance: IEC 61882:2016, ISO 17776:2016, EPA 40 CFR Part 98*
*Review Required: Annual or upon significant system changes*
