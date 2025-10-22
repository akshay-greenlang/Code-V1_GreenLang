# CSRD Reporting Platform - Compliance Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-18
**Target Audience:** Compliance officers, auditors, sustainability managers

---

## Table of Contents

1. [Overview](#overview)
2. [CSRD/ESRS Regulatory Framework](#csrdesrs-regulatory-framework)
3. [Coverage and Scope](#coverage-and-scope)
4. [Materiality Assessment Requirements](#materiality-assessment-requirements)
5. [Data Quality Requirements](#data-quality-requirements)
6. [Disclosure Requirements](#disclosure-requirements)
7. [Audit Trail and Documentation](#audit-trail-and-documentation)
8. [Regulatory Citations](#regulatory-citations)
9. [Compliance Checklist](#compliance-checklist)

---

## Overview

### What is CSRD?

The **Corporate Sustainability Reporting Directive (CSRD)** is an EU regulation requiring companies to disclose information about:
- Environmental matters (climate, pollution, biodiversity, water, circular economy)
- Social matters (workforce, workers in value chain, affected communities, consumers)
- Governance matters (business conduct, corporate culture, political influence, management)

### Key Requirements

- **Double Materiality Assessment**: Identify both impact materiality (inside-out) and financial materiality (outside-in)
- **ESRS Standards**: Report according to European Sustainability Reporting Standards
- **ESEF Format**: Electronic submission in iXBRL format
- **Third-Party Assurance**: External audit required (limited initially, reasonable later)
- **Digital Tagging**: XBRL tags for machine-readable disclosures

### Who Must Comply?

**Phase 1 (2025 reporting on 2024)**: Large public-interest entities (>500 employees) already subject to NFRD
**Phase 2 (2026 reporting on 2025)**: Large companies (≥250 employees OR ≥€50M revenue OR ≥€25M assets)
**Phase 3 (2027 reporting on 2026)**: Listed SMEs (with opt-out until 2028)
**Phase 4 (2029 reporting on 2028)**: Non-EU companies with significant EU activity (>€150M revenue)

### Platform Compliance Features

Our CSRD Reporting Platform ensures compliance through:

1. **IntakeAgent**: Data validation against ESRS requirements
2. **MaterialityAgent**: AI-assisted double materiality assessment (with human review)
3. **CalculatorAgent**: Zero-hallucination metric calculations
4. **AggregatorAgent**: Multi-framework alignment (TCFD, GRI, SASB)
5. **ReportingAgent**: ESEF-compliant XBRL generation
6. **AuditAgent**: 215+ compliance rule validation

---

## CSRD/ESRS Regulatory Framework

### Regulatory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 1: CSRD Directive (EU 2022/2464)                          │
│ - Legal framework and requirements                              │
│ - Mandatory disclosure areas                                     │
│ - Assurance requirements                                         │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ Level 2: ESRS (European Sustainability Reporting Standards)     │
│ - ESRS 1: General Requirements                                  │
│ - ESRS 2: General Disclosures                                   │
│ - E1-E5: Environmental Standards                                │
│ - S1-S4: Social Standards                                       │
│ - G1: Governance Standard                                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ Level 3: ESEF Taxonomy & Technical Standards                    │
│ - XBRL taxonomy for ESRS                                        │
│ - iXBRL formatting requirements                                 │
│ - Digital tagging specifications                                │
└─────────────────────────────────────────────────────────────────┘
```

### ESRS Standards Overview

#### Cross-Cutting Standards

**ESRS 1: General Requirements**
- Purpose: Sets out foundational principles for sustainability reporting
- Key concepts: Double materiality, value chain, time horizons
- Our implementation: Embedded in all 6 agents

**ESRS 2: General Disclosures**
- Purpose: General information about company and sustainability reporting
- Requirements: Basis of preparation, governance, strategy, impact/risk/opportunity management
- Our implementation: Company profile structure, materiality assessment process

#### Environmental Standards (E1-E5)

**E1: Climate Change**
- Scope: GHG emissions (Scope 1, 2, 3), transition plans, climate risks
- Key metrics:
  - E1-1: GHG emissions Scope 1 (tCO2e)
  - E1-2: GHG emissions Scope 2 (tCO2e)
  - E1-3: GHG emissions Scope 3 (tCO2e)
  - E1-4: Total GHG emissions
  - E1-5: GHG intensity
  - E1-6: Energy consumption and mix
- Our implementation: CalculatorAgent with GHG Protocol-aligned formulas

**E2: Pollution**
- Scope: Air, water, soil pollution; substances of concern
- Key metrics: Emissions to air/water, microplastics, hazardous substances
- Our implementation: Pollution metric calculations in CalculatorAgent

**E3: Water and Marine Resources**
- Scope: Water consumption, discharge, water stress
- Key metrics: Water withdrawal, consumption, discharge by source
- Our implementation: Water metric calculations

**E4: Biodiversity and Ecosystems**
- Scope: Impact on biodiversity, land use, ecosystem services
- Key metrics: Sites in/near protected areas, land degradation
- Our implementation: Biodiversity assessments in MaterialityAgent

**E5: Resource Use and Circular Economy**
- Scope: Resource inflows/outflows, waste, circular practices
- Key metrics: Material footprint, waste generation, recycling rates
- Our implementation: Circular economy metrics in CalculatorAgent

#### Social Standards (S1-S4)

**S1: Own Workforce**
- Scope: Working conditions, equal treatment, training, health & safety
- Key metrics:
  - S1-1: Total workforce by contract type
  - S1-2: Employee turnover rate
  - S1-3: Gender pay gap
  - S1-4: Work-related accidents and fatalities
  - S1-5: Training hours per employee
- Our implementation: Social metric calculations in CalculatorAgent

**S2: Workers in the Value Chain**
- Scope: Working conditions in supply chain, forced labor, child labor
- Key metrics: Suppliers screened for social criteria
- Our implementation: Value chain assessments

**S3: Affected Communities**
- Scope: Impact on local communities, indigenous peoples
- Key metrics: Community engagement, grievance mechanisms
- Our implementation: Community impact assessments

**S4: Consumers and End-Users**
- Scope: Product safety, data privacy, marketing practices
- Key metrics: Customer complaints, data breaches
- Our implementation: Consumer impact metrics

#### Governance Standard (G1)

**G1: Business Conduct**
- Scope: Corporate culture, whistleblower protection, anti-corruption, political influence
- Key metrics:
  - G1-1: Board diversity by gender/age
  - G1-2: Ethics and compliance violations
  - G1-3: Anti-corruption training
- Our implementation: Governance metrics in CalculatorAgent

### Sector-Specific Standards

Currently in development by EFRAG for high-impact sectors:
- Agriculture, forestry and fishing
- Mining, quarrying and coal
- Manufacturing
- Electricity, gas, steam and air conditioning
- Road transport
- Food and beverage sector

---

## Coverage and Scope

### Data Points Coverage

Our platform supports **500+ ESRS data points** across all standards:

| Standard | Data Points | Mandatory | Conditional |
|----------|-------------|-----------|-------------|
| ESRS 2 | 58 | 58 | 0 |
| E1 | 86 | 24 | 62 |
| E2 | 42 | 8 | 34 |
| E3 | 38 | 6 | 32 |
| E4 | 44 | 7 | 37 |
| E5 | 52 | 9 | 43 |
| S1 | 67 | 18 | 49 |
| S2 | 34 | 5 | 29 |
| S3 | 28 | 4 | 24 |
| S4 | 26 | 3 | 23 |
| G1 | 31 | 12 | 19 |
| **Total** | **506** | **154** | **352** |

### Materiality-Based Reporting

**Key Principle**: Not all companies must report on all standards. Materiality assessment determines which standards apply.

**Always Mandatory** (regardless of materiality):
- ESRS 2: General Disclosures (all)
- E1: Climate Change (minimum disclosure requirements)

**Conditional** (based on materiality assessment):
- E2, E3, E4, E5: Environmental standards
- S1, S2, S3, S4: Social standards
- G1: Business conduct

**Platform Support**:
- MaterialityAgent conducts double materiality assessment
- Identifies material ESRS standards
- CalculatorAgent calculates only material metrics (plus ESRS 2 & E1 minimums)
- AuditAgent validates materiality-disclosure alignment

### Time Horizons

CSRD requires reporting across three time horizons:

| Horizon | Definition | Examples |
|---------|------------|----------|
| Short-term | ≤ 1 year | Annual targets, immediate risks |
| Medium-term | 1-5 years | Strategic plans, medium-term risks |
| Long-term | > 5 years | Net-zero commitments, long-term trends |

**Platform Support**: Metadata fields for time horizon classification

### Value Chain Scope

CSRD requires consideration of:
- **Upstream**: Suppliers, raw material extraction, transportation
- **Own Operations**: Direct activities and employees
- **Downstream**: Product use, end-of-life, customers

**Platform Support**:
- Value chain classification in data intake
- Scope 3 GHG emissions for upstream/downstream
- Value chain impact assessment in MaterialityAgent

---

## Materiality Assessment Requirements

### Double Materiality Concept

Companies must assess sustainability matters from two perspectives:

#### 1. Impact Materiality (Inside-Out)

**Definition**: Material if the company has significant positive or negative impacts on people or the environment.

**Assessment Criteria**:
- Scale: How grave/beneficial is the impact?
- Scope: How widespread is the impact?
- Irremediable character: How hard to counteract/remedy?

**Rating Scale**: 0-10 (our platform uses this)
- 0-3: Not material
- 4-6: Potentially material (requires deeper assessment)
- 7-10: Material

**Examples**:
- High impact materiality: Manufacturing company with significant water consumption in water-stressed area
- Low impact materiality: Office-based service company with minimal environmental footprint

#### 2. Financial Materiality (Outside-In)

**Definition**: Material if the sustainability matter triggers financial effects on the company's cash flows, development, position, or performance.

**Assessment Criteria**:
- Risks: Potential negative effects on company value
- Opportunities: Potential positive effects on company value
- Magnitude: Size of potential financial effect
- Likelihood: Probability of occurrence

**Rating Scale**: 0-10 (our platform uses this)
- 0-3: Not material
- 4-6: Potentially material
- 7-10: Material

**Examples**:
- High financial materiality: Carbon-intensive company facing carbon pricing
- Low financial materiality: IT company with low exposure to climate transition risks

#### 3. Double Materiality

**Definition**: Material if EITHER impact materiality OR financial materiality thresholds are met.

```
Impact Material + Financial Material = DOUBLE MATERIAL (highest priority)
Impact Material OR Financial Material = MATERIAL (must disclose)
Neither = NOT MATERIAL (may omit, with explanation)
```

### Materiality Assessment Process

**ESRS 1 requires a structured process:**

1. **Understand Context** (MaterialityAgent step 1)
   - Company's business model, value chain, sector
   - Stakeholder perspectives
   - Geographic and regulatory context

2. **Identify Actual/Potential Impacts, Risks, Opportunities** (MaterialityAgent step 2)
   - List all sustainability matters from ESRS
   - Identify actual impacts (already occurring)
   - Identify potential impacts (could occur)
   - Identify risks and opportunities

3. **Assess Materiality** (MaterialityAgent step 3)
   - Rate impact materiality (scale, scope, irremediability)
   - Rate financial materiality (magnitude, likelihood)
   - Apply thresholds

4. **Determine Material Matters** (MaterialityAgent step 4)
   - List material matters
   - Map to ESRS standards
   - Document rationale

5. **Review and Validate** (Human review required)
   - Board/senior management approval
   - Stakeholder consultation
   - Update annually

### Platform Implementation

**MaterialityAgent Capabilities**:
- AI-powered assessment using LLM (GPT-4 or Claude)
- Analyzes company profile, sector, geographic context
- Rates all 82 ESRS topics for impact and financial materiality
- Provides rationale for each rating
- Flags items requiring human review
- Generates documentation for audit trail

**Human Review Requirements**:
```python
result = csrd_assess_materiality(
    company_context="company.json",
    llm_provider="openai",
    impact_threshold=5.0,
    financial_threshold=5.0
)

# IMPORTANT: Human review required!
print(f"Material topics: {result['summary_statistics']['material_topics_count']}")
print(f"Review flags: {len(result['review_flags'])} items need human review")

for flag in result['review_flags']:
    print(f"  - {flag['topic']}: {flag['reason']}")
    # Assign to compliance officer for review
```

**Materiality Matrix Output**:

```
Impact Materiality (0-10) →
10 │     E1      │  E2, S1  │ E4, S2
9  │            │          │
8  │            │    E3    │
7  │            │          │
6  │     E5     │          │
5  │──────┼──────┼──────│  ← Financial Materiality Threshold
4  │            │          │
3  │     G1     │          │
2  │            │          │
1  │   S3, S4   │          │
0  │──────┼──────┼──────│
   0      5      7      10
          ↑
   Impact Materiality Threshold
```

### Materiality Documentation Requirements

**ESRS 2 requires disclosure of**:
1. Process for identifying impacts, risks, opportunities
2. Methodology for materiality assessment
3. Thresholds used (impact and financial)
4. List of material sustainability matters
5. Changes from prior period
6. Stakeholder engagement process

**Platform Support**:
- Auto-generated materiality assessment report
- Structured documentation of process
- Version control for year-over-year comparison
- Stakeholder input fields

---

## Data Quality Requirements

### ESRS Data Quality Dimensions

ESRS 1 defines six data quality characteristics:

#### 1. Relevance
**Definition**: Information enables users to assess company's impacts, risks, opportunities.
**Requirements**:
- Aligned with decision-making needs
- Focused on material matters
- Specific to company's situation

**Platform Check**:
- IntakeAgent validates metric relevance to company profile
- Flags irrelevant data points
- Maps metrics to material ESRS standards

#### 2. Faithful Representation
**Definition**: Information accurately depicts the reality it purports to represent.
**Requirements**:
- Complete (all material info)
- Neutral (unbiased)
- Free from error

**Platform Check**:
- Validation against ESRS definitions
- Cross-checks for consistency
- Outlier detection
- Accuracy scoring (0-100)

#### 3. Verifiability
**Definition**: Independent parties can verify information is faithfully represented.
**Requirements**:
- Clear data sources documented
- Calculation methodologies transparent
- Assumptions explicit
- Complete audit trail

**Platform Support**:
- Complete provenance tracking for all metrics
- Calculation formulas documented
- Data lineage from source to report
- AuditAgent generates auditor packages

#### 4. Comparability
**Definition**: Users can compare over time and across entities.
**Requirements**:
- Consistent methodologies year-over-year
- Aligned with industry practices
- Restatements clearly disclosed

**Platform Support**:
- Standardized ESRS metric definitions
- Version-controlled formulas
- Multi-framework alignment (TCFD, GRI, SASB)

#### 5. Understandability
**Definition**: Information is clear and concise.
**Requirements**:
- Plain language
- Well-structured
- Defined technical terms
- Appropriate detail level

**Platform Support**:
- Human-readable summary reports
- Clear metric descriptions
- Glossary of terms

#### 6. Timeliness
**Definition**: Information available when needed for decision-making.
**Requirements**:
- Published within regulatory deadlines
- Updated when material changes occur
- Aligned with financial reporting cycle

**Platform Support**:
- Fast processing (<30 minutes for full pipeline)
- Automated validation and calculation
- Real-time compliance checking

### Data Quality Scoring

**Platform Quality Assessment**:

IntakeAgent calculates an overall data quality score (0-100) based on:

```python
Quality Score = (
    0.30 * Completeness +
    0.25 * Accuracy +
    0.20 * Consistency +
    0.15 * Timeliness +
    0.10 * Validity
)
```

**Dimension Calculations**:

1. **Completeness** (0-100):
   ```
   Completeness = (Required fields populated / Total required fields) * 100
   ```

2. **Accuracy** (0-100):
   ```
   Accuracy = (Values within expected range / Total values) * 100
   ```

3. **Consistency** (0-100):
   ```
   Consistency = (Cross-checks passed / Total cross-checks) * 100
   ```
   Examples: GHG Scope 1+2+3 = Total GHG, Workforce sum matches by category

4. **Timeliness** (0-100):
   ```
   Timeliness = (Data points < 12 months old / Total data points) * 100
   ```

5. **Validity** (0-100):
   ```
   Validity = (Valid metric codes / Total metric codes) * 100
   ```

**Quality Thresholds**:
- **Excellent** (90-100): Ready for submission
- **Good** (80-89): Minor issues, acceptable
- **Fair** (70-79): Review recommended
- **Poor** (<70): Significant data quality issues, must remediate

**Platform Configuration**:
```python
config = CSRDConfig(
    quality_threshold=0.80,  # Minimum 80% for acceptance
    ...
)

result = csrd_validate_data(
    esg_data="data.csv",
    config=config
)

if result['metadata']['data_quality_score'] < 80:
    print("WARNING: Data quality below threshold")
    for issue in result['quality_metrics']:
        print(f"{issue['dimension']}: {issue['score']:.1f}/100")
```

---

## Disclosure Requirements

### Mandatory Disclosures (ESRS 2)

All companies must disclose:

**Basis of Preparation (BP)**
- BP-1: General basis of preparation
- BP-2: Disclosures in relation to specific circumstances (e.g., time constraints, insufficient data)

**Governance (GOV)**
- GOV-1: Role of administrative, management and supervisory bodies
- GOV-2: Information provided to and sustainability matters addressed by bodies
- GOV-3: Integration of sustainability-related performance in incentive schemes
- GOV-4: Statement on due diligence
- GOV-5: Risk management and internal controls over sustainability reporting

**Strategy (SBM)**
- SBM-1: Strategy, business model and value chain
- SBM-2: Interests and views of stakeholders
- SBM-3: Material impacts, risks and opportunities and their interaction with strategy

**Impact, Risk and Opportunity Management (IRO)**
- IRO-1: Description of processes to identify and assess material impacts, risks and opportunities
- IRO-2: Disclosure requirements in ESRS covered by the undertaking's sustainability statement

**Platform Support**:
```python
# ESRS 2 disclosure fields auto-populated from company profile
report = csrd_build_report(
    esg_data="data.csv",
    company_profile={
        "company_info": {...},
        "governance": {
            "board_composition": [...],
            "sustainability_oversight": "...",
            "incentive_schemes": "..."
        },
        "strategy": {
            "business_model": "...",
            "value_chain": "...",
            "stakeholder_engagement": "..."
        },
        "risk_management": {
            "process_description": "...",
            "internal_controls": "..."
        }
    }
)
```

### Climate Disclosures (E1 - Always Mandatory)

**Minimum requirements even if E1 not material:**
- E1-1: Transition plan for climate change mitigation (qualitative)
- E1-2: Policies related to climate change mitigation and adaptation (qualitative)
- E1-3: Actions and resources in relation to climate change policies (qualitative)

**If E1 is material, additional requirements:**
- E1-4: Targets related to climate change mitigation and adaptation
- E1-5: Energy consumption and mix
- E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions
- E1-7: GHG removals and GHG mitigation projects financed through carbon credits
- E1-8: Internal carbon pricing
- E1-9: Anticipated financial effects from material physical and transition risks

**Platform Support**:
```python
# E1 metrics always calculated
result = csrd_calculate_metrics(
    validated_data=validated,
    materiality=materiality  # E1 minimum even if not material
)

# GHG emissions with full provenance
for metric in result['calculated_metrics']:
    if metric['metric_code'].startswith('E1'):
        print(f"{metric['metric_code']}: {metric['value']} {metric['unit']}")
        print(f"  Formula: {metric['formula_used']}")
        print(f"  Source data: {metric['input_data']}")
```

### Conditional Disclosures

**Disclosure Requirements (DRs)** in E2-E5, S1-S4, G1 apply only if:
1. Topic assessed as material, AND
2. Company has policies/actions/targets related to the topic

**Phased Implementation** (for companies applying ESRS for first time):
- **Year 1**: Omit Scope 3 GHG emissions, own workforce metrics for value chain workers
- **Year 2**: Omit only Scope 3 if facing undue cost/effort
- **Year 3**: Full compliance

**Platform Support**:
```python
# Materiality-driven disclosure
materiality = csrd_assess_materiality(company_context="company.json")

# Only calculates metrics for material standards
metrics = csrd_calculate_metrics(
    validated_data=validated,
    materiality=materiality  # Determines which metrics to calculate
)

print(f"Material standards: {materiality['summary_statistics']['esrs_standards_triggered']}")
print(f"Metrics calculated: {metrics['metadata']['metrics_calculated']}")
```

---

## Audit Trail and Documentation

### Provenance Tracking

**Complete Data Lineage**:

Every calculated metric includes:
```json
{
  "metric_code": "E1-1",
  "value": 12345.67,
  "unit": "tCO2e",
  "calculation_provenance": {
    "formula_used": "sum(scope1_emissions)",
    "formula_source": "GHG Protocol",
    "input_data": {
      "scope1_emissions": [
        {"source": "ERP", "value": 10000, "timestamp": "2024-12-31"},
        {"source": "Manual", "value": 2345.67, "timestamp": "2024-12-31"}
      ]
    },
    "calculation_timestamp": "2025-01-15T10:30:00Z",
    "calculated_by": "CalculatorAgent v1.0.0",
    "deterministic": true,
    "zero_hallucination_guarantee": true
  }
}
```

**Audit Trail Components**:

1. **Data Intake Log**
   - Original source files (with checksums)
   - Validation results
   - Data transformations applied
   - Quality scores

2. **Materiality Assessment Log**
   - LLM provider and model used
   - Input prompts and responses
   - Human review decisions
   - Approval timestamps

3. **Calculation Log**
   - All formulas used
   - Input values
   - Calculation steps
   - Output values
   - Verification checksums

4. **Compliance Validation Log**
   - All rules checked
   - Pass/fail results
   - Timestamps
   - Validator version

### Documentation Requirements

**ESRS requires documentation of:**

1. **Methodology Documentation**
   - Data collection processes
   - Calculation methodologies
   - Estimation techniques
   - Assumptions made
   - Changes from prior period

2. **Quality Assurance**
   - Internal controls
   - Review process
   - Approval workflow
   - External verification

3. **Materiality Assessment**
   - Process description
   - Stakeholder engagement
   - Thresholds used
   - Material matters identified
   - Rationale for determinations

4. **Limitations and Uncertainties**
   - Data gaps
   - Estimation uncertainties
   - Methodological limitations
   - Plans for improvement

**Platform-Generated Documentation**:

```python
# Full audit package generation
audit_package = agent.generate_audit_package(
    company_name="Acme Manufacturing GmbH",
    reporting_year=2024,
    compliance_report=compliance_result,
    calculation_audit_trail=calculation_trail,
    output_dir="audit_package"
)

# Generates:
# - compliance_report.json
# - calculation_audit_trail.json
# - data_lineage_report.json
# - methodology_documentation.pdf
# - materiality_assessment_report.pdf
```

### Retention Requirements

**CSRD requires retaining documentation for minimum 10 years.**

**Platform Support**:
- All outputs timestamped and versioned
- Complete reports saved to output directory
- JSON format for long-term preservation
- Deterministic calculations allow recreation

---

## Regulatory Citations

### Primary Legislation

**Directive (EU) 2022/2464** - Corporate Sustainability Reporting Directive (CSRD)
- Amends Directive 2013/34/EU (Accounting Directive)
- Published: 14 December 2022
- Entry into force: 5 January 2023
- Key articles:
  - Article 19a: Sustainability reporting
  - Article 29a: Consolidated sustainability reporting
  - Article 34aa: Assurance of sustainability reporting

**Regulation (EU) 2020/852** - Taxonomy Regulation
- Defines environmentally sustainable activities
- Referenced in ESRS for alignment disclosures

**Regulation (EU) 2019/2088** - Sustainable Finance Disclosure Regulation (SFDR)
- Financial market participants' sustainability disclosures
- Aligned with CSRD requirements

### ESRS Standards (Delegated Regulation)

**Commission Delegated Regulation (EU) 2023/2772** - European Sustainability Reporting Standards
- Adopted: 31 July 2023
- Published: 22 December 2023
- Entry into application: Phased (2024-2028)

**Standards Included**:
- ESRS 1: General requirements
- ESRS 2: General disclosures
- ESRS E1: Climate change
- ESRS E2: Pollution
- ESRS E3: Water and marine resources
- ESRS E4: Biodiversity and ecosystems
- ESRS E5: Resource use and circular economy
- ESRS S1: Own workforce
- ESRS S2: Workers in the value chain
- ESRS S3: Affected communities
- ESRS S4: Consumers and end-users
- ESRS G1: Business conduct

### Technical Standards

**ESEF Regulation** - Commission Delegated Regulation (EU) 2019/815
- European Single Electronic Format
- XBRL taxonomy requirements
- iXBRL formatting specifications

**ESRS XBRL Taxonomy** (in development)
- Published by EFRAG
- Updates periodically
- Platform uses latest version

### Key EFRAG Publications

**Implementation Guidance**:
- ESRS 1 Appendix B: Application Guidance
- ESRS 1 Appendix C: List of ESRS Disclosure Requirements and data points
- ESRS 1 Appendix D: Structure of ESRS sustainability statements
- ESRS 1 Appendix E: Disclosure/Application Requirements applicable for the year of first application

**Q&A Documents**:
- EFRAG publishes periodic Q&A on ESRS application
- Check https://www.efrag.org/ for latest updates

---

## Compliance Checklist

### Pre-Reporting Checklist

- [ ] **Scope Determination**
  - [ ] Confirmed company is in scope of CSRD
  - [ ] Identified applicable reporting year
  - [ ] Determined consolidation perimeter
  - [ ] Identified value chain boundaries

- [ ] **Governance Setup**
  - [ ] Board/management approval for CSRD project
  - [ ] Sustainability reporting team established
  - [ ] Roles and responsibilities defined
  - [ ] Budget and resources allocated

- [ ] **Data Infrastructure**
  - [ ] Data collection processes established
  - [ ] IT systems configured for ESG data
  - [ ] Data quality controls implemented
  - [ ] Historical data gathered (if available)

### Materiality Assessment Checklist

- [ ] **Process Documentation**
  - [ ] Materiality assessment process documented
  - [ ] Stakeholder engagement conducted
  - [ ] Thresholds defined and approved

- [ ] **Impact Materiality Assessment**
  - [ ] All ESRS topics assessed for actual impacts
  - [ ] All ESRS topics assessed for potential impacts
  - [ ] Impacts rated for scale, scope, irremediability
  - [ ] Material impacts identified and documented

- [ ] **Financial Materiality Assessment**
  - [ ] All ESRS topics assessed for risks
  - [ ] All ESRS topics assessed for opportunities
  - [ ] Risks/opportunities rated for magnitude and likelihood
  - [ ] Material risks/opportunities identified and documented

- [ ] **Materiality Determination**
  - [ ] Material sustainability matters listed
  - [ ] Mapped to ESRS standards
  - [ ] Rationale documented for each determination
  - [ ] Changes from prior period explained (if applicable)
  - [ ] Board/senior management approval obtained

### Data Collection Checklist

- [ ] **Required Data Gathered**
  - [ ] ESRS 2 general disclosures
  - [ ] E1 climate data (minimum requirements)
  - [ ] Material ESRS standard data
  - [ ] Company profile information
  - [ ] Governance information

- [ ] **Data Quality Verified**
  - [ ] Completeness check passed (≥80%)
  - [ ] Accuracy check passed (≥80%)
  - [ ] Consistency check passed (≥80%)
  - [ ] Timeliness check passed (≥80%)
  - [ ] Overall quality score ≥80%

- [ ] **Data Sources Documented**
  - [ ] Source systems identified for each data point
  - [ ] Data collection methods documented
  - [ ] Responsible persons identified
  - [ ] Collection frequency defined

### Calculation Checklist

- [ ] **Methodology Defined**
  - [ ] Calculation formulas documented
  - [ ] Aligned with ESRS requirements
  - [ ] Aligned with established standards (e.g., GHG Protocol)
  - [ ] Assumptions documented
  - [ ] Limitations identified

- [ ] **Calculations Performed**
  - [ ] All mandatory metrics calculated (ESRS 2, E1 minimum)
  - [ ] All material metrics calculated
  - [ ] Calculations verified for accuracy
  - [ ] Audit trail complete
  - [ ] Peer review completed

- [ ] **GHG Emissions** (if E1 material)
  - [ ] Scope 1 emissions calculated
  - [ ] Scope 2 emissions calculated (location-based and market-based)
  - [ ] Scope 3 emissions calculated (or phase-in applied)
  - [ ] Emission factors documented
  - [ ] Consolidation approach documented

### Compliance Validation Checklist

- [ ] **Mandatory Disclosures Complete**
  - [ ] ESRS 2 general disclosures complete
  - [ ] E1 minimum climate disclosures complete
  - [ ] Material ESRS disclosures complete
  - [ ] No omissions without explanation

- [ ] **Disclosure Requirements Met**
  - [ ] All applicable DRs addressed
  - [ ] Phased implementation correctly applied (if applicable)
  - [ ] Omissions justified per ESRS 1

- [ ] **Data Quality Thresholds Met**
  - [ ] Overall quality score ≥80%
  - [ ] Critical data points 100% complete
  - [ ] No critical compliance failures

- [ ] **Cross-Checks Passed**
  - [ ] Materiality assessment aligns with disclosures
  - [ ] Material topics disclosed in sustainability statement
  - [ ] Non-material topics explained if disclosed
  - [ ] Calculations verified

### Reporting Checklist

- [ ] **ESEF Format**
  - [ ] iXBRL format generated
  - [ ] ESRS XBRL taxonomy applied
  - [ ] All required tags applied
  - [ ] XBRL validation passed

- [ ] **Sustainability Statement Structure**
  - [ ] Follows ESRS structure (ESRS 1 Appendix D)
  - [ ] Cross-references to financial statements included
  - [ ] Index of ESRS disclosures included
  - [ ] Readable format (PDF + iXBRL)

- [ ] **Review and Approval**
  - [ ] Internal review completed
  - [ ] Management review completed
  - [ ] Board approval obtained
  - [ ] Declaration signed

### Assurance Checklist

- [ ] **Assurance Scope Defined**
  - [ ] Limited assurance scope agreed (initially)
  - [ ] Assurance provider selected
  - [ ] Timeline agreed
  - [ ] Materiality for assurance defined

- [ ] **Documentation Prepared**
  - [ ] Audit package generated
  - [ ] Calculation audit trail provided
  - [ ] Data lineage documented
  - [ ] Controls documentation provided
  - [ ] Prior year reports (if applicable)

- [ ] **Assurance Report Obtained**
  - [ ] Assurance report received
  - [ ] No material misstatements identified
  - [ ] Recommendations reviewed
  - [ ] Report published with sustainability statement

### Submission Checklist

- [ ] **Files Prepared**
  - [ ] Sustainability statement (PDF)
  - [ ] iXBRL file
  - [ ] Assurance report
  - [ ] Accompanying documentation

- [ ] **Submission Completed**
  - [ ] Filed with national regulator
  - [ ] Published on company website
  - [ ] Incorporated in management report
  - [ ] Submitted within deadline

- [ ] **Post-Submission**
  - [ ] Confirmations received
  - [ ] Public availability verified
  - [ ] Feedback documented
  - [ ] Lessons learned captured for next year

### Platform-Assisted Compliance

**Our platform automates key checklist items:**

```python
# 1. Materiality assessment
materiality = csrd_assess_materiality(company_context="company.json")

# 2. Data validation
validated = csrd_validate_data(esg_data="data.csv")
assert validated['metadata']['data_quality_score'] >= 80, "Quality threshold not met"

# 3. Metric calculation
metrics = csrd_calculate_metrics(validated_data=validated, materiality=materiality)
assert metrics['metadata']['zero_hallucination_guarantee'], "Calculation integrity compromised"

# 4. Compliance validation
compliance = csrd_audit_compliance(report_package=complete_report)
assert compliance['compliance_report']['compliance_status'] == "PASS", "Compliance check failed"

# 5. Report generation
report = csrd_build_report(esg_data="data.csv", company_profile="company.json")

# 6. Audit package
audit_pkg = agent.generate_audit_package(
    company_name="Acme",
    reporting_year=2024,
    compliance_report=compliance,
    calculation_audit_trail=trail,
    output_dir="audit"
)

print("✅ All compliance checks passed!")
```

---

**End of Compliance Guide**

For more information, see:
- [User Guide](USER_GUIDE.md) - Step-by-step tutorials
- [API Reference](API_REFERENCE.md) - Technical documentation
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Installation and setup
- [Operations Manual](OPERATIONS_MANUAL.md) - Running in production
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
