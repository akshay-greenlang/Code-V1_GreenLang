# PACK-030: Framework Compliance Matrices

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [SBTi Compliance Matrix](#1-sbti-compliance-matrix)
2. [CDP Compliance Matrix](#2-cdp-compliance-matrix)
3. [TCFD Compliance Matrix](#3-tcfd-compliance-matrix)
4. [GRI 305 Compliance Matrix](#4-gri-305-compliance-matrix)
5. [ISSB IFRS S2 Compliance Matrix](#5-issb-ifrs-s2-compliance-matrix)
6. [SEC Climate Disclosure Compliance Matrix](#6-sec-climate-disclosure-compliance-matrix)
7. [CSRD ESRS E1 Compliance Matrix](#7-csrd-esrs-e1-compliance-matrix)
8. [Cross-Framework Mapping](#8-cross-framework-mapping)

---

## 1. SBTi Compliance Matrix

**Standard:** SBTi Corporate Net-Zero Standard v1.1 (2024)

| Requirement | Description | PACK-030 Implementation | Status |
|-------------|-------------|------------------------|--------|
| Target scope | Cover Scope 1 + 2 (minimum 95%) | Data Aggregation Engine validates scope coverage | Compliant |
| Scope 3 coverage | Minimum 67% of total Scope 3 | Scope 3 category completeness check | Compliant |
| Near-term target | 5-10 year reduction target | PACK-029 interim target integration | Compliant |
| Long-term target | At least 90% by 2050 | GL-SBTi-APP target data integration | Compliant |
| Base year | Fixed base year with recalculation policy | PACK-021 baseline integration | Compliant |
| Annual progress | Annual disclosure of progress | SBTi Progress Report Workflow | Compliant |
| 1.5C alignment | Aligned with 1.5C pathway (42% by 2030) | Validation Engine checks pathway alignment | Compliant |
| Methodology | GHG Protocol Corporate Standard | Methodology documentation in evidence bundle | Compliant |
| Verification | Independent verification recommended | Assurance Packaging Engine | Supported |

---

## 2. CDP Compliance Matrix

**Standard:** CDP Climate Change Questionnaire (2025)

| Module | Title | Questions | PACK-030 Coverage | Status |
|--------|-------|-----------|-------------------|--------|
| C0 | Introduction | ~15 | Organization metadata from config | Full |
| C1 | Governance | ~20 | GL-TCFD-APP governance data | Full |
| C2 | Risks & Opportunities | ~40 | GL-TCFD-APP risk assessments | Full |
| C3 | Business Strategy | ~25 | GL-TCFD-APP strategy data | Full |
| C4 | Targets & Performance | ~30 | PACK-029 + GL-SBTi-APP targets | Full |
| C5 | Emissions Methodology | ~15 | Methodology documentation | Full |
| C6 | Emissions (Scope 1/2) | ~20 | GL-GHG-APP inventory data | Full |
| C7 | Emissions (Scope 3) | ~25 | GL-GHG-APP Scope 3 categories | Full |
| C8 | Energy | ~15 | GL-GHG-APP energy data | Full |
| C9 | Additional Metrics | ~10 | Intensity metrics calculation | Full |
| C10 | Verification | ~10 | Assurance Packaging Engine | Full |
| C11 | Carbon Pricing | ~10 | Internal carbon pricing data | Full |
| C12 | Engagement | ~15 | Value chain engagement data | Partial |
| **Total** | | **~250** | | **95%+ coverage** |

### CDP Scoring Alignment

| Scoring Level | Requirement | PACK-030 Support |
|---------------|-------------|-----------------|
| D- (Disclosure) | Basic data provided | Automatic data population |
| C (Awareness) | Risk/opportunity identification | GL-TCFD-APP integration |
| B (Management) | Management processes documented | Workflow audit trail |
| A- (Leadership) | Targets, verification, engagement | Full target/assurance integration |
| A (Leadership) | Best practice in all areas | Completeness scoring + recommendations |

---

## 3. TCFD Compliance Matrix

**Standard:** TCFD Recommendations (2023)

| Pillar | Recommended Disclosure | PACK-030 Component | Data Source | Status |
|--------|----------------------|-------------------|------------|--------|
| **Governance** | a) Board oversight | TCFD Governance Template | GL-TCFD-APP | Compliant |
| | b) Management's role | TCFD Governance Template | GL-TCFD-APP | Compliant |
| **Strategy** | a) Climate risks/opportunities | TCFD Strategy Template | GL-TCFD-APP | Compliant |
| | b) Impact on business/strategy | TCFD Strategy Template | GL-TCFD-APP | Compliant |
| | c) Resilience (scenario analysis) | TCFD Strategy Template | GL-TCFD-APP | Compliant |
| **Risk Mgmt** | a) Identification/assessment | TCFD Risk Template | GL-TCFD-APP | Compliant |
| | b) Management processes | TCFD Risk Template | GL-TCFD-APP | Compliant |
| | c) Integration with ERM | TCFD Risk Template | GL-TCFD-APP | Compliant |
| **Metrics** | a) Metrics used | TCFD Metrics Template | GL-GHG-APP | Compliant |
| | b) Scope 1, 2, 3 emissions | TCFD Metrics Template | GL-GHG-APP | Compliant |
| | c) Targets and progress | TCFD Metrics Template | PACK-029 | Compliant |

---

## 4. GRI 305 Compliance Matrix

**Standard:** GRI 305: Emissions (2016)

| Disclosure | Title | Required Metrics | PACK-030 Source | Status |
|-----------|-------|-----------------|----------------|--------|
| 305-1 | Direct (Scope 1) | Gross Scope 1, gases included, biogenic CO2, base year, methodology | GL-GHG-APP | Compliant |
| 305-2 | Energy indirect (Scope 2) | Location-based, market-based, gases, methodology | GL-GHG-APP | Compliant |
| 305-3 | Other indirect (Scope 3) | Categories, gases, biogenic CO2, methodology | GL-GHG-APP | Compliant |
| 305-4 | GHG emissions intensity | Intensity ratio, included scopes, gases | Calculated | Compliant |
| 305-5 | Reduction of GHG emissions | Reductions achieved, gases, base year, methodology | PACK-022/029 | Compliant |
| 305-6 | Emissions of ODS | Production, imports, exports of ODS | GL-GHG-APP | Compliant |
| 305-7 | NOx, SOx, air emissions | Significant air emissions by type | GL-GHG-APP | Compliant |

---

## 5. ISSB IFRS S2 Compliance Matrix

**Standard:** IFRS S2 Climate-related Disclosures (2023)

| Paragraph | Requirement | PACK-030 Component | Status |
|-----------|-------------|-------------------|--------|
| 5-7 | Governance: Board oversight, management role | ISSB Governance section | Compliant |
| 8-22 | Strategy: Risks, opportunities, transition plans, resilience | ISSB Strategy section | Compliant |
| 23-27 | Risk Management: Processes for identifying, assessing, managing | ISSB Risk section | Compliant |
| 28 | Cross-industry metrics | Scope 1/2/3 emissions from GL-GHG-APP | Compliant |
| 29 | Industry-specific metrics (SASB) | SASB metric mapping by industry | Compliant |
| 30-37 | Targets and progress | PACK-029 + GL-SBTi-APP | Compliant |
| B1-B71 | Application guidance | Followed in template design | Compliant |

### XBRL Compliance

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Digital reporting taxonomy | IFRS S2 XBRL taxonomy applied | Compliant |
| Machine-readable format | XBRL output with validated tags | Compliant |
| Taxonomy validation | Automated validation against official taxonomy | Compliant |

---

## 6. SEC Climate Disclosure Compliance Matrix

**Standard:** SEC Climate Disclosure Rule (2024)

| Regulation | Requirement | PACK-030 Component | Status |
|-----------|-------------|-------------------|--------|
| Reg S-K 1500 | Definitions | Aligned with SEC definitions | Compliant |
| Reg S-K 1501 | Applicability | Filer type configuration | Compliant |
| Reg S-K 1502 | Governance | SEC Governance section | Compliant |
| Reg S-K 1503 | Strategy | SEC Strategy section | Compliant |
| Reg S-K 1504 | Risk management | SEC Risk Management section | Compliant |
| Reg S-K 1505 | GHG emissions metrics | Scope 1 + 2 from GL-GHG-APP | Compliant |
| Reg S-K 1506 | Targets and goals | PACK-029 + GL-SBTi-APP | Compliant |

### XBRL/iXBRL Compliance

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| XBRL tagging of emissions | XBRL Tagging Engine applies SEC taxonomy | Compliant |
| iXBRL inline filing | iXBRL rendering with embedded tags | Compliant |
| Taxonomy version | SEC-2024 taxonomy version | Compliant |
| Context references | Filing context with entity and period | Compliant |
| Unit references | tCO2e, USD as defined by taxonomy | Compliant |

### Attestation

| Filer Type | Attestation Requirement | PACK-030 Support |
|-----------|------------------------|-----------------|
| Large accelerated | Limited assurance (phased) | Attestation template + evidence bundle |
| Accelerated | Limited assurance (phased) | Attestation template + evidence bundle |
| Non-accelerated | Not required initially | Optional attestation template |

---

## 7. CSRD ESRS E1 Compliance Matrix

**Standard:** ESRS E1 Climate Change (2024)

| Disclosure | Title | Data Points | PACK-030 Source | Status |
|-----------|-------|------------|----------------|--------|
| E1-1 | Transition plan | Transition plan details, implementation status | Organization config + PACK-022 | Compliant |
| E1-2 | Policies | Climate policies, integration with strategy | GL-TCFD-APP | Compliant |
| E1-3 | Actions and resources | Actions taken, resources committed, outcomes | PACK-022 initiatives | Compliant |
| E1-4 | Targets | GHG reduction targets, progress | PACK-029 + GL-SBTi-APP | Compliant |
| E1-5 | Energy consumption | Total energy, renewable share, intensity | GL-GHG-APP energy data | Compliant |
| E1-6 | Scope 1/2/3 | Disaggregated emissions by scope | GL-GHG-APP inventory | Compliant |
| E1-7 | Removals & credits | GHG removals, carbon credit quality | Organization data | Compliant |
| E1-8 | Carbon pricing | Internal carbon pricing mechanisms | Organization data | Compliant |
| E1-9 | Financial effects | Anticipated physical/transition risk impacts | GL-TCFD-APP scenarios | Compliant |

### Digital Taxonomy

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| ESRS digital taxonomy tags | XBRL Tagging Engine applies CSRD taxonomy | Compliant |
| Machine-readable format | Digital taxonomy XML output | Compliant |
| Taxonomy validation | Automated validation against EFRAG taxonomy | Compliant |

### Double Materiality

| Aspect | PACK-030 Support |
|--------|-----------------|
| Impact materiality | Organization's climate impact (E1-6 emissions data) |
| Financial materiality | Climate's financial impact (E1-9 from GL-TCFD-APP) |
| Cross-reference | Links E1 disclosures to double materiality assessment |

---

## 8. Cross-Framework Mapping

### Metric Equivalence Table

| Metric | SBTi | CDP | TCFD | GRI | ISSB | SEC | CSRD |
|--------|------|-----|------|-----|------|-----|------|
| Scope 1 total | Target metric | C6.1 | M&T-a/b | 305-1 | S2.29(a) | 1505 | E1-6 |
| Scope 2 (location) | Target metric | C6.3 | M&T-a/b | 305-2 | S2.29(a) | 1505 | E1-6 |
| Scope 2 (market) | Target metric | C6.3 | M&T-a/b | 305-2 | S2.29(a) | 1505 | E1-6 |
| Scope 3 total | Target metric | C6.5 | M&T-a/b | 305-3 | S2.29(a) | N/A* | E1-6 |
| Reduction target | Progress table | C4.1 | M&T-c | 305-5 | S2.33 | 1506 | E1-4 |
| Base year | Progress table | C5.2 | M&T | 305-1 | S2.29 | 1505 | E1-6 |
| Intensity metric | N/A | C6.10 | M&T-a | 305-4 | S2.29(b) | N/A | E1-6 |
| Energy consumption | N/A | C8.2 | N/A | 302-1 | S2.29(d) | N/A | E1-5 |

\* SEC Scope 3 disclosure is voluntary under current rules

### Section Equivalence Table

| Section | SBTi | CDP | TCFD | ISSB | SEC | CSRD |
|---------|------|-----|------|------|-----|------|
| Governance | N/A | C1 | Governance | S2.5-7 | 1502 | E1-2 |
| Strategy | N/A | C3 | Strategy | S2.8-22 | 1503 | E1-1 |
| Risk Management | N/A | C2 | Risk Mgmt | S2.23-27 | 1504 | E1-3 |
| Metrics | Progress | C6-C7 | Metrics | S2.28-29 | 1505 | E1-6 |
| Targets | Progress | C4 | Targets | S2.33-37 | 1506 | E1-4 |

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
