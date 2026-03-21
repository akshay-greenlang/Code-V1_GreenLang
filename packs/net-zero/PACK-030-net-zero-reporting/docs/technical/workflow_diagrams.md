# PACK-030: Workflow Diagrams

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Workflow Overview](#workflow-overview)
2. [Workflow 1: SBTi Progress Report](#workflow-1-sbti-progress-report)
3. [Workflow 2: CDP Questionnaire](#workflow-2-cdp-questionnaire)
4. [Workflow 3: TCFD Disclosure](#workflow-3-tcfd-disclosure)
5. [Workflow 4: GRI 305 Disclosure](#workflow-4-gri-305-disclosure)
6. [Workflow 5: ISSB IFRS S2](#workflow-5-issb-ifrs-s2)
7. [Workflow 6: SEC Climate Disclosure](#workflow-6-sec-climate-disclosure)
8. [Workflow 7: CSRD ESRS E1](#workflow-7-csrd-esrs-e1)
9. [Workflow 8: Multi-Framework Full Report](#workflow-8-multi-framework-full-report)
10. [DAG Orchestration](#dag-orchestration)

---

## 1. Workflow Overview

| # | Workflow | Phases | Lines | Duration |
|---|----------|--------|-------|----------|
| 1 | SBTi Progress Report | 8 | 1,400 | <5s |
| 2 | CDP Questionnaire | 8 | 1,600 | <8s |
| 3 | TCFD Disclosure | 8 | 1,500 | <6s |
| 4 | GRI 305 Disclosure | 8 | 1,200 | <4s |
| 5 | ISSB IFRS S2 | 7 | 1,300 | <5s |
| 6 | SEC Climate Disclosure | 8 | 1,400 | <6s |
| 7 | CSRD ESRS E1 | 12 | 1,500 | <7s |
| 8 | Multi-Framework Full | 7 | 1,800 | <10s |
| **Total** | | **66** | **11,700** | |

---

## 2. Workflow 1: SBTi Progress Report

**File:** `workflows/sbti_progress_workflow.py` (~1,400 lines)

```
START
  |
  v
[1. Aggregate Targets]-----> GL-SBTi-APP
  | Target data: base year, target year, ambition level
  v
[2. Aggregate Emissions]---> PACK-021, PACK-029
  | Current & historical emissions data
  v
[3. Calculate Progress]
  | Progress % = (base_year - current_year) / (base_year - target_year)
  | RAG status: Green <= 5%, Amber <= 15%, Red > 15%
  v
[4. Generate Variance]
  | Explain deviation from linear trajectory
  | Attribute to activity, intensity, structural effects
  v
[5. Compile Report]-------> SBTi Progress Template
  | Sections: target description, progress table, variance explanation
  v
[6. Validate Schema]------> SBTi Schema v1.1
  | Check required fields, scope coverage, base year rules
  v
[7. Render Formats]
  | PDF: Executive-ready progress report
  | JSON: Machine-readable for API consumers
  v
[8. Package Submission]
  | Create submission-ready package with evidence
  v
END -> SBTi Progress Report (PDF, JSON)
```

---

## 3. Workflow 2: CDP Questionnaire

**File:** `workflows/cdp_questionnaire_workflow.py` (~1,600 lines)

```
START
  |
  v
[1. Aggregate Emissions]---> GL-GHG-APP
  | C6: Scope 1/2 emissions
  | C7: Scope 3 emissions by category
  v
[2. Pull Targets]----------> PACK-029, GL-SBTi-APP
  | C4: Emission reduction targets, progress
  v
[3. Pull Governance]-------> GL-CDP-APP (historical), GL-TCFD-APP
  | C1: Board oversight, management roles
  v
[4. Pull Risks]------------> GL-TCFD-APP
  | C2: Climate-related risks identification
  | C3: Business strategy integration
  v
[5. Pull Opportunities]----> GL-TCFD-APP
  | C2: Climate-related opportunities
  v
[6. Generate Narratives]--> Narrative Generation Engine
  | Text responses for C1-C4 qualitative questions
  | Citation management for all quantitative claims
  v
[7. Validate Completeness]
  | Score: (questions_answered / questions_total) * 100
  | Flag missing required questions
  | A-list scoring guidance
  v
[8. Export Excel]----------> CDP Excel Template
  | Module-by-module Excel workbook
  | Ready for CDP online upload
  v
END -> CDP Questionnaire (Excel, JSON)
```

---

## 4. Workflow 3: TCFD Disclosure

**File:** `workflows/tcfd_disclosure_workflow.py` (~1,500 lines)

```
START
  |
  +---> [1. Governance Pillar]
  |       | Board oversight of climate risks/opportunities
  |       | Management's role in climate assessment
  |
  +---> [2. Strategy Pillar]
  |       | Climate risks and opportunities identified
  |       | Impact on business, strategy, financial planning
  |       | Resilience under different scenarios
  |
  +---> [3. Risk Management Pillar]
  |       | Identification and assessment processes
  |       | Management processes
  |       | Integration with enterprise risk management
  |
  +---> [4. Metrics & Targets Pillar]
          | Scope 1, 2, 3 emissions
          | Targets and progress against targets
          | Climate-related financial metrics
  |
  v
[5. Compile Report]-------> TCFD Report Template
  | Assemble 4 pillars into executive report
  v
[6. Add Scenarios]---------> GL-TCFD-APP
  | Scenario analysis (1.5C, 2C, 4C)
  | Physical and transition risk assessment
  v
[7. Render PDF]
  | Executive-ready PDF with charts and tables
  | Interactive HTML with drill-down
  v
[8. Generate Evidence]
  | Assurance evidence bundle for TCFD sections
  v
END -> TCFD Disclosure (PDF, HTML, Evidence ZIP)
```

---

## 5. Workflow 4: GRI 305 Disclosure

**File:** `workflows/gri_305_workflow.py` (~1,200 lines)

```
START
  |
  +---> [1. 305-1: Direct Scope 1]
  |       | Gross Scope 1 by GHG type (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
  |       | Biogenic CO2 emissions separately
  |
  +---> [2. 305-2: Energy Indirect Scope 2]
  |       | Location-based and market-based approaches
  |       | Contractual instruments used
  |
  +---> [3. 305-3: Other Indirect Scope 3]
  |       | Categories 1-15 breakdown
  |       | Methodology per category
  |
  +---> [4. 305-4: Emissions Intensity]
  |       | Organization-specific intensity ratio
  |       | Revenue intensity, headcount intensity
  |
  +---> [5. 305-5: Emission Reductions]
  |       | Reduction initiatives and quantified impact
  |       | Year-over-year comparison
  |
  +---> [6. 305-6: ODS Emissions]
  |       | Ozone-depleting substance emissions
  |
  +---> [7. 305-7: Air Emissions]
          | NOx, SOx, persistent organic pollutants
          | Volatile organic compounds, HAPs
  |
  v
[8. Generate Content Index]
  | GRI Content Index table
  | Disclosure number, title, page reference
  v
END -> GRI 305 Disclosure (PDF, HTML, Content Index)
```

---

## 6. Workflow 5: ISSB IFRS S2

**File:** `workflows/issb_ifrs_s2_workflow.py` (~1,300 lines)

```
START
  |
  +---> [1. Governance (S2 para 5-7)]
  |       | Board body/committee oversight
  |       | Management role and expertise
  |
  +---> [2. Strategy (S2 para 8-22)]
  |       | Climate risks and opportunities
  |       | Transition plans
  |       | Resilience assessment
  |
  +---> [3. Risk Management (S2 para 23-27)]
  |       | Identification and assessment
  |       | Prioritization processes
  |       | Integration with overall risk
  |
  +---> [4. Metrics & Targets (S2 para 28-37)]
          | Cross-industry metrics (Scope 1/2/3)
          | Industry-specific metrics (SASB)
          | Targets and progress
  |
  v
[5. XBRL Tagging]---------> XBRL Tagging Engine
  | Apply IFRS S2 taxonomy tags to all metrics
  v
[6. Validation]
  | Validate against IFRS S2 requirements
  | Check industry-specific completeness
  v
[7. Render Output]
  | PDF: Narrative report
  | XBRL: Machine-readable digital report
  v
END -> IFRS S2 Report (PDF, XBRL)
```

---

## 7. Workflow 6: SEC Climate Disclosure

**File:** `workflows/sec_climate_workflow.py` (~1,400 lines)

```
START
  |
  +---> [1. Item 1: Business Description]
  |       | Climate risks in business model
  |
  +---> [2. Item 1A: Risk Factors]
  |       | Material climate-related risks
  |       | Physical risks, transition risks
  |
  +---> [3. Item 7: MD&A]
  |       | Climate impacts on financial condition
  |       | Climate expenditures
  |
  +---> [4. Reg S-K 1502-1506]
          | 1502: Governance
          | 1503: Strategy, financial impacts
          | 1504: Risk management processes
          | 1505: Scope 1 and 2 emissions
          | 1506: Targets and progress
  |
  v
[5. XBRL/iXBRL Tagging]---> XBRL Tagging Engine
  | Apply SEC taxonomy tags to all metrics
  | Generate XBRL file + iXBRL HTML
  v
[6. Validate SEC Schema]
  | Element validation
  | Context and unit validation
  | Cross-reference with financial data
  v
[7. Generate Attestation]
  | Attestation report template (limited assurance)
  | For large accelerated filers
  v
[8. Package for 10-K]
  | SEC filing package
  | PDF narrative + XBRL + iXBRL + attestation
  v
END -> SEC Disclosure (PDF, XBRL, iXBRL, Attestation)
```

---

## 8. Workflow 7: CSRD ESRS E1

**File:** `workflows/csrd_esrs_e1_workflow.py` (~1,500 lines)

```
START
  |
  +---> [1. E1-1: Transition Plan]
  |       | Climate change mitigation transition plan
  |       | Implementation status and timeline
  |
  +---> [2. E1-2: Policies]
  |       | Policies related to climate change
  |       | Integration with business strategy
  |
  +---> [3. E1-3: Actions & Resources]
  |       | Actions taken, resources committed
  |       | Expected outcomes and timeline
  |
  +---> [4. E1-4: Targets]
  |       | GHG emission reduction targets
  |       | Progress tracking against targets
  |
  +---> [5. E1-5: Energy]
  |       | Total energy consumption
  |       | Energy mix (renewable vs non-renewable)
  |       | Energy intensity
  |
  +---> [6. E1-6: Scope 1/2/3]
  |       | Gross Scope 1 emissions (disaggregated)
  |       | Gross Scope 2 emissions (location + market)
  |       | Gross Scope 3 emissions (categories)
  |
  +---> [7. E1-7: Removals & Credits]
  |       | GHG removal activities
  |       | Carbon credit quality and retirement
  |
  +---> [8. E1-8: Carbon Pricing]
  |       | Internal carbon pricing mechanisms
  |       | Price levels and coverage
  |
  +---> [9. E1-9: Financial Effects]
          | Anticipated financial effects of physical risks
          | Anticipated financial effects of transition risks
  |
  v
[10. Digital Taxonomy]-----> XBRL Tagging Engine
  | Apply CSRD ESRS E1 digital taxonomy tags
  v
[11. Validate ESRS]
  | Validate against ESRS E1 requirements
  | Check all 9 disclosure requirements present
  v
[12. Render Digital Report]
  | PDF: Full ESRS E1 disclosure
  | Digital taxonomy: Machine-readable XML
  v
END -> ESRS E1 Disclosure (PDF, Digital Taxonomy)
```

---

## 9. Workflow 8: Multi-Framework Full Report

**File:** `workflows/multi_framework_workflow.py` (~1,800 lines)

```
START
  |
  v
[1. Aggregate Data Once]
  | Single data aggregation from all 8 sources
  | Reconciliation and gap detection
  | Data lineage tracking
  |
  v
[2. Generate Shared Narratives]
  | Create master narratives for common sections
  | Framework-specific adaptations from master
  | Cross-framework consistency validation
  |
  v
[3. Execute 7 Workflows in Parallel]
  |
  +---> SBTi Workflow --------> SBTi Report
  +---> CDP Workflow ----------> CDP Questionnaire
  +---> TCFD Workflow ---------> TCFD Report
  +---> GRI Workflow ----------> GRI 305 Disclosure
  +---> ISSB Workflow ---------> IFRS S2 Report
  +---> SEC Workflow ----------> SEC 10-K Section
  +---> CSRD Workflow ---------> ESRS E1 Disclosure
  |
  v
[4. Validate Cross-Framework Consistency]
  | Check all 7 reports for consistency
  | Scope 1/2/3 totals match
  | Target descriptions match
  | Methodologies match
  | Score: 0-100%
  |
  v
[5. Generate Executive Dashboard]
  | Framework coverage heatmap
  | Deadline countdown timers
  | Consistency score gauge
  | Drill-down capabilities
  |
  v
[6. Create Master Evidence Bundle]
  | Combined evidence across all frameworks
  | Unified provenance hash chain
  | Cross-framework control matrix
  |
  v
[7. Package All Reports]
  | Deliverable package with all outputs
  | Index page with links to all reports
  | Summary statistics
  |
  v
END -> Multi-Framework Package
  |
  +-- SBTi Report (PDF, JSON)
  +-- CDP Questionnaire (Excel, JSON)
  +-- TCFD Report (PDF, HTML)
  +-- GRI 305 Disclosure (PDF, HTML)
  +-- ISSB IFRS S2 (PDF, XBRL)
  +-- SEC 10-K Section (PDF, XBRL, iXBRL)
  +-- CSRD ESRS E1 (PDF, Digital Taxonomy)
  +-- Executive Dashboard (HTML)
  +-- Evidence Bundle (ZIP)
  +-- Consistency Report (JSON)
```

---

## 10. DAG Orchestration

### Dependency Graph

All workflows are orchestrated as Directed Acyclic Graphs (DAGs) with explicit dependencies:

```
                    Data Aggregation
                         |
              +----------+----------+
              |                     |
    Shared Narratives    Framework Mapping
              |                     |
              +----------+----------+
                         |
         +------+---+---+---+---+------+
         |      |   |   |   |   |      |
       SBTi  CDP TCFD GRI ISSB SEC  CSRD
         |      |   |   |   |   |      |
         +------+---+---+---+---+------+
                         |
              Cross-Framework Validation
                         |
              +----------+----------+
              |          |          |
          Dashboard  Evidence  Package
```

### Phase Execution Model

Each workflow phase follows this execution model:

1. **Pre-check**: Validate inputs, check dependencies
2. **Execute**: Run the phase logic (async)
3. **Validate**: Verify outputs meet quality requirements
4. **Provenance**: Generate SHA-256 hash of phase output
5. **Checkpoint**: Save state for recovery
6. **Post-hook**: Log metrics, update status

### Error Recovery

- **Phase failure**: Retry up to 3 times with exponential backoff
- **Partial completion**: Resume from last checkpoint
- **Data unavailability**: Proceed with available data, flag gaps
- **Timeout**: Configurable per-phase timeout (default 30s)

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
