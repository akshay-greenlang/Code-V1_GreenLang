# PRD-PACK-030: Net Zero Reporting Pack

**Pack ID:** PACK-030-net-zero-reporting
**Category:** Net Zero Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Approved
**Author:** GreenLang Product Team
**Date:** 2026-03-18
**Prerequisites:** PACK-021 Net Zero Starter Pack, PACK-022 Net Zero Acceleration Pack, PACK-028 Sector Pathway Pack, PACK-029 Interim Targets Pack

---

## 1. Executive Summary

### 1.1 Problem Statement

Organizations with net-zero commitments face a proliferation of climate disclosure frameworks, each with unique structures, metrics, narratives, and technical formats. The reporting burden has reached unsustainable levels:

1. **Framework fragmentation**: Organizations must report to 7+ frameworks (SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD), each with different scopes, metrics, formats, and submission deadlines. A typical multinational corporation spends 500+ hours per year on climate disclosure across frameworks.

2. **Data aggregation complexity**: Net-zero reporting requires data from multiple source packs (PACK-021/022/028/029), applications (GL-SBTi-APP, GL-CDP-APP, GL-TCFD-APP), and external systems. Manual aggregation across 15+ data sources is error-prone and creates audit trail gaps.

3. **Narrative inconsistency**: Different teams draft narratives for different frameworks, leading to contradictory statements about the same underlying facts (e.g., target progress described as "on track" in SBTi but "amber" in internal reporting).

4. **Format diversity**: Frameworks require different output formats: CDP uses online questionnaires, TCFD requires narrative PDFs, SEC mandates XBRL/iXBRL, CSRD requires digital taxonomies. Organizations lack tools to generate multiple formats from a single source of truth.

5. **Assurance gaps**: External auditors require evidence packages showing calculation provenance, data lineage, and methodology documentation. Manual evidence assembly takes 200+ hours per audit cycle.

6. **Timeliness challenges**: Framework deadlines are staggered (CDP: July, SBTi: annual rolling, CSRD: within 5 months of year-end, SEC: within 90 days). Organizations struggle to meet all deadlines while maintaining consistency.

7. **Stakeholder-specific views**: Investors want TCFD-aligned reports, regulators want CSRD disclosures, customers want carbon labels, employees want progress dashboards. Creating customized views requires duplicative work.

### 1.2 Solution Overview

PACK-030 is the **Net Zero Reporting Pack** -- the sixth pack in the GreenLang "Net Zero Packs" category. It provides a comprehensive, multi-framework reporting automation platform that aggregates data from prerequisite packs, generates consistent narratives, produces framework-compliant outputs, and packages assurance evidence.

The pack delivers:
- **Multi-framework report generation**: Automated creation of SBTi annual progress reports, CDP Climate Change questionnaires (C0-C12), TCFD disclosures (Governance/Strategy/Risk/Metrics), GRI 305 emissions disclosures, ISSB IFRS S2 reports, SEC climate disclosures, and CSRD ESRS E1 reports
- **Data aggregation engine**: Unified data collection from PACK-021/022/028/029, GL-SBTi-APP, GL-CDP-APP, GL-TCFD-APP, GL-GHG-APP with automated reconciliation and gap detection
- **Narrative generation**: AI-assisted narrative drafting with citation management, consistency validation, and multi-language support (EN, DE, FR, ES)
- **Framework mapping**: Automatic translation between framework-specific terminologies, metrics, and reporting structures with bidirectional synchronization
- **Format rendering**: Multi-format output generation including PDF (TCFD, narrative reports), HTML (interactive dashboards), Excel (data tables), JSON (API integration), and XBRL/iXBRL (SEC, CSRD digital taxonomies)
- **Assurance packaging**: Automated evidence bundle generation with calculation provenance, data lineage diagrams, methodology documentation, and control matrices for ISAE 3410/3000 audits
- **Dashboard platform**: Interactive executive dashboards with progress tracking, framework coverage heatmaps, stakeholder-specific views, and drill-down capabilities
- **Validation engine**: Cross-framework consistency checks, metric reconciliation, deadline tracking, and completeness scoring

Every output is **traceable** (SHA-256 provenance on all calculations), **consistent** (single source of truth across frameworks), and **audit-ready** (automated evidence packaging).

### 1.3 Key Differentiators

| Dimension | Manual Approach | Spreadsheet-Based | Consulting Services | **PACK-030** |
|-----------|----------------|-------------------|---------------------|--------------|
| **Framework coverage** | Fragmented | 2-3 frameworks | Custom per engagement | **7 frameworks built-in** |
| **Data aggregation** | Manual copy-paste | Semi-automated | Consultant-driven | **Automated with lineage** |
| **Narrative generation** | Different authors | Template-based | Consultant-written | **AI-assisted with citations** |
| **Output formats** | Single format | PDF + Excel only | Custom deliverables | **6 formats (PDF/HTML/Excel/JSON/XBRL/iXBRL)** |
| **Assurance readiness** | Manual evidence | Minimal | High but expensive | **Automated evidence bundles** |
| **Update frequency** | Annual only | Quarterly manual | Per engagement | **Real-time + scheduled** |
| **Consistency validation** | None | Manual checks | Consultant review | **Automated cross-framework checks** |
| **Multi-language** | Manual translation | None | Custom translation | **4 languages built-in** |
| **Cost** | High (labor) | Medium (labor) | Very high ($200K+/year) | **Low (automated)** |
| **Audit trail** | Poor | Fair | Good | **Cryptographic provenance** |

### 1.4 Target Users

**Primary:**
- Chief Sustainability Officers responsible for multi-framework climate disclosure strategy
- Investor Relations Directors managing TCFD/ISSB reporting for institutional investors
- CDP reporting leads preparing annual CDP Climate Change questionnaire submissions
- SEC disclosure teams preparing 10-K climate disclosures with XBRL tagging
- CSRD reporting leads preparing ESRS E1 digital submissions

**Secondary:**
- External auditors performing ISAE 3410 GHG assurance requiring evidence packages
- Board audit committees overseeing climate disclosure governance
- ESG data vendors integrating climate data feeds via API
- Marketing teams creating customer-facing carbon reports and labels
- Employee communications teams building internal progress dashboards

### 1.5 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Framework coverage | 7 frameworks (SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD) | Feature completeness audit |
| Report generation time | <10s for full multi-framework report suite | Performance benchmark |
| Data aggregation accuracy | 100% reconciliation across source packs | Cross-validation tests |
| Narrative consistency score | 95%+ across frameworks | NLP consistency analysis |
| Format rendering quality | 100% schema compliance (CDP, XBRL, etc.) | Validation against official schemas |
| Assurance evidence completeness | 100% of required evidence types | ISAE 3410 requirement mapping |
| Test pass rate | 100% | 2,000+ test cases across 17 test files |
| Code coverage | 90%+ | pytest-cov measurement |
| API response time | <200ms for dashboard endpoints | Load testing |
| Multi-language accuracy | 98%+ translation quality | Professional translation review |

---

## 2. Background and Motivation

### 2.1 The Reporting Proliferation Challenge

The landscape of climate disclosure frameworks has evolved rapidly:
- **2015**: Paris Agreement catalyzes corporate climate commitments
- **2017**: TCFD Recommendations published (Governance/Strategy/Risk/Metrics)
- **2019**: SBTi launches Corporate Net-Zero Standard
- **2020**: CDP Climate Change questionnaire expands to 12 modules
- **2021**: ISSB formed to harmonize sustainability standards
- **2022**: SEC proposes climate disclosure rules with XBRL requirements
- **2023**: CSRD enters force with ESRS E1 Climate Change standard
- **2024**: Global Baseline established but fragmentation persists

Organizations now face:
- 7+ mandatory or investor-expected frameworks
- 500+ hours/year reporting burden for large enterprises
- $500K+ annual cost for external consultants and auditors
- High error rates due to manual data transfer across systems

### 2.2 Regulatory Drivers

| Framework | Jurisdiction | Mandate | Deadline |
|-----------|--------------|---------|----------|
| **CSRD/ESRS E1** | EU (50,000+ companies) | Mandatory | Within 5 months of fiscal year-end |
| **SEC Climate Rule** | US large accelerated filers | Mandatory (proposed) | 10-K within 90 days of year-end |
| **ISSB IFRS S2** | Global (IOSCO jurisdictions) | Varies by jurisdiction | Aligned with financial reporting |
| **CDP Climate Change** | Investor-requested | Voluntary but expected | July 31 annual deadline |
| **TCFD** | Multiple jurisdictions | Mandatory (UK, NZ, etc.) | Annual reporting cycle |
| **SBTi** | Voluntary commitment | Voluntary but public | Annual progress disclosure |
| **GRI 305** | Voluntary | Voluntary | Annual sustainability report |

### 2.3 Market Context

- 18,000+ companies report to CDP (2025)
- 6,000+ SBTi-validated targets requiring annual progress updates
- 50,000+ EU companies subject to CSRD
- $130T+ assets under management committed to TCFD alignment
- External audit costs: $50-500K per framework per year

Organizations need a unified platform that eliminates duplicate work, ensures consistency, and reduces reporting costs by 70%+.

---

## 3. Objectives and Success Metrics

### 3.1 Primary Objectives

| # | Objective | Description |
|---|-----------|-------------|
| O1 | **Multi-framework automation** | Generate compliant reports for 7 frameworks (SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD) from a single data aggregation process |
| O2 | **Data aggregation** | Collect and reconcile data from PACK-021/022/028/029 and GL apps with automated gap detection and lineage tracking |
| O3 | **Narrative consistency** | Generate AI-assisted narratives with cross-framework consistency validation and citation management |
| O4 | **Multi-format output** | Render reports in 6 formats (PDF, HTML, Excel, JSON, XBRL, iXBRL) with schema compliance validation |
| O5 | **Assurance readiness** | Package evidence bundles with provenance, lineage, methodology docs for ISAE 3410/3000 audits |
| O6 | **Dashboard platform** | Provide interactive executive dashboards with progress tracking, heatmaps, and stakeholder views |
| O7 | **API integration** | Expose RESTful APIs for ESG data vendors, investor portals, and internal systems |

### 3.2 Key Performance Indicators

| KPI | Target | Stretch Goal |
|-----|--------|-------------|
| Report generation time (full suite) | <10s | <5s |
| Data aggregation accuracy | 100% | 100% |
| Framework coverage | 7 frameworks | 10 frameworks |
| Output format coverage | 6 formats | 8 formats |
| Test pass rate | 100% | 100% |
| Code coverage | 90% | 95% |
| API response time (p95) | <200ms | <100ms |
| Narrative consistency score | 95% | 98% |
| Assurance evidence completeness | 100% | 100% |
| Multi-language quality | 98% | 99% |

---

## 4. User Personas

### 4.1 Chief Sustainability Officer (Primary)

**Profile:** C-suite executive responsible for enterprise sustainability strategy, board reporting, and multi-stakeholder disclosure.

**Pain Points:**
- Manages 7 different reporting teams for different frameworks
- Cannot ensure narrative consistency across frameworks
- Spends $500K+/year on external consultants for disclosure preparation
- Lacks unified dashboard showing status across all frameworks

**PACK-030 Value:**
- Single platform generating all 7 framework reports
- Automated consistency validation across frameworks
- 70% reduction in external consultant costs
- Executive dashboard with framework coverage heatmap

### 4.2 CDP Reporting Lead (Primary)

**Profile:** Manager responsible for annual CDP Climate Change questionnaire submission with A-list scoring ambition.

**Pain Points:**
- CDP questionnaire has 12 modules with 300+ data points
- Manual data entry from spreadsheets is error-prone
- Cannot reuse TCFD narratives for CDP text responses
- Struggles to meet July 31 deadline

**PACK-030 Value:**
- Automated CDP questionnaire generation from aggregated data
- Narrative reuse from TCFD with CDP-specific adaptations
- Data completeness scoring showing progress to 100%
- Excel export for review before online submission

### 4.3 SEC Disclosure Team (Primary)

**Profile:** Legal/finance team preparing 10-K climate disclosures with XBRL tagging requirements.

**Pain Points:**
- SEC requires XBRL/iXBRL tagging for Scope 1/2 emissions
- Manual XBRL creation is complex and error-prone
- Needs to reconcile climate data with financial reporting
- 90-day filing deadline is tight

**PACK-030 Value:**
- Automated XBRL/iXBRL generation with SEC taxonomy compliance
- Cross-validation with financial data from ERP systems
- SEC-compliant PDF + XBRL package generation
- Audit trail for SOX compliance

### 4.4 External Auditor (Secondary)

**Profile:** Big 4 auditor performing ISAE 3410 GHG assurance requiring evidence of calculation integrity.

**Pain Points:**
- Clients cannot provide calculation provenance efficiently
- Spends 40+ hours requesting and reviewing evidence
- Lacks visibility into data lineage and methodology changes
- Cannot verify consistency across frameworks

**PACK-030 Value:**
- Automated assurance evidence bundle with all required documentation
- SHA-256 provenance on every calculation
- Data lineage diagrams showing source-to-report flow
- Cross-framework consistency reports

---

## 5. Functional Requirements

### 5.1 Data Aggregation

**FR-001: Multi-Source Data Collection**
- Aggregate emissions data from PACK-021 (baseline), PACK-022 (reduction initiatives), PACK-028 (sector pathways), PACK-029 (interim targets)
- Pull SBTi target data from GL-SBTi-APP
- Pull CDP historical responses from GL-CDP-APP
- Pull TCFD scenario data from GL-TCFD-APP
- Pull GHG inventory from GL-GHG-APP
- Support API, database, and file-based integrations

**FR-002: Data Reconciliation**
- Detect mismatches between source systems (e.g., Scope 1 total in PACK-021 vs GL-GHG-APP)
- Flag gaps where required data is missing for target framework
- Provide reconciliation workflow with approval tracking
- Generate reconciliation reports for audit trail

**FR-003: Data Lineage Tracking**
- Trace every reported metric to source system and calculation
- Generate visual lineage diagrams (source → transformation → report)
- Record transformation logic and business rules
- Support drill-down from report metric to source transaction

### 5.2 Framework Report Generation

**FR-004: SBTi Annual Progress Report**
- Generate SBTi-compliant annual progress disclosure
- Include target description, base year, progress table, variance explanation
- Validate against SBTi disclosure template
- Support PDF and online submission formats

**FR-005: CDP Climate Change Questionnaire**
- Generate responses for CDP modules C0-C12
- Support data tables, text narratives, and attachments
- Provide completeness scoring (% of required questions answered)
- Export to Excel for review before online submission

**FR-006: TCFD Disclosure**
- Generate TCFD-aligned report with 4 pillars (Governance, Strategy, Risk Management, Metrics & Targets)
- Include forward-looking scenario analysis
- Support narrative + data table combination
- Render as executive-ready PDF

**FR-007: GRI 305 Emissions Disclosure**
- Generate GRI 305 disclosures (305-1 through 305-7)
- Include all required emissions scopes and intensity metrics
- Support GRI Content Index table generation
- Provide assurance statement placeholder

**FR-008: ISSB IFRS S2 Report**
- Generate IFRS S2-compliant climate disclosure
- Include industry-specific metrics from SASB standards
- Support XBRL tagging for digital reporting
- Align with financial reporting periods

**FR-009: SEC Climate Disclosure**
- Generate SEC 10-K climate disclosure section
- Include Scope 1/2 emissions with XBRL/iXBRL tagging
- Support attestation report requirements
- Validate against SEC schema

**FR-010: CSRD ESRS E1 Report**
- Generate ESRS E1 Climate Change disclosure
- Include all data points (E1-1 through E1-9)
- Support digital taxonomy tagging
- Provide assurance report template

### 5.3 Narrative Generation

**FR-011: AI-Assisted Narrative Drafting**
- Generate draft narratives for qualitative disclosure sections
- Use citation management linking narratives to source data
- Support multi-language generation (EN, DE, FR, ES)
- Provide human review and editing workflow

**FR-012: Consistency Validation**
- Check narratives for contradictions across frameworks
- Flag inconsistent statements about same metrics
- Suggest harmonization edits
- Generate consistency score (0-100%)

**FR-013: Citation Management**
- Link every quantitative claim to source calculation
- Generate footnotes and references automatically
- Support hyperlinked PDF with clickable citations
- Maintain citation database for audit

### 5.4 Format Rendering

**FR-014: PDF Generation**
- Render TCFD, GRI, and narrative reports as executive-ready PDFs
- Support custom branding (logo, colors, fonts)
- Include table of contents, page numbers, headers/footers
- Support hyperlinked citations and cross-references

**FR-015: HTML Interactive Reports**
- Generate HTML5 dashboards with charts and drill-down
- Support responsive design for mobile viewing
- Include interactive data tables with sorting/filtering
- Provide export to PDF from HTML

**FR-016: Excel Data Tables**
- Export structured data tables to Excel
- Support pivot tables and charts
- Include data validation and dropdown menus
- Provide templates for CDP upload preparation

**FR-017: JSON API Output**
- Expose report data via RESTful JSON API
- Support pagination, filtering, and field selection
- Include OpenAPI/Swagger documentation
- Provide versioning and backward compatibility

**FR-018: XBRL/iXBRL Tagging**
- Generate XBRL files for SEC climate disclosure
- Generate iXBRL (inline XBRL) for human-readable + machine-readable
- Validate against official SEC and CSRD taxonomies
- Support XBRL taxonomy updates

### 5.5 Assurance and Audit

**FR-019: Evidence Bundle Generation**
- Package all calculation provenances (SHA-256 hashes)
- Include data lineage diagrams
- Provide methodology documentation
- Generate control matrix for ISAE 3410

**FR-020: Audit Trail**
- Record all report generation events with timestamps
- Track user edits to narratives with version history
- Maintain immutable log of data sources and transformations
- Support audit log export for external auditors

**FR-021: Verification Workflow**
- Support internal review and approval workflow
- Track reviewer comments and resolutions
- Require signoff before publication
- Generate approval summary report

### 5.6 Dashboard and Analytics

**FR-022: Executive Dashboard**
- Display progress across all 7 frameworks in single view
- Show framework coverage heatmap (% complete)
- Include deadline countdown timers
- Provide drill-down to framework details

**FR-023: Stakeholder Views**
- Investor view: TCFD + ISSB focus
- Regulator view: CSRD + SEC focus
- Customer view: Carbon labels and product footprints
- Employee view: Progress toward targets

**FR-024: Analytics**
- Trend analysis of emissions metrics over time
- Peer benchmarking (where data available)
- Scenario comparison (e.g., different reduction pathways)
- What-if analysis for initiative impacts

---

## 6. Technical Requirements

### 6.1 Architecture

**TR-001: Async/Await Patterns**
- All I/O operations use async/await
- Parallel data fetching from multiple sources
- Non-blocking report generation
- Support for concurrent user requests

**TR-002: PostgreSQL 16+ Database**
- TimescaleDB hypertables for time-series emissions data
- JSONB columns for flexible metadata storage
- Full-text search for narrative content
- Row-level security for multi-tenant isolation

**TR-003: Caching Strategy**
- Redis caching for framework schemas and templates
- In-memory caching for static reference data
- HTTP cache headers for API responses
- 95%+ cache hit ratio target

**TR-004: Zero-Hallucination Architecture**
- No LLM in calculation paths (narratives only)
- Deterministic Decimal arithmetic for all metrics
- Template-based report generation with data insertion
- Human review required before publication

### 6.2 Data Model

**TR-005: Report Metadata Schema**
```
Tables:
- gl_nz_reports (report_id, framework, organization_id, reporting_period, status)
- gl_nz_report_sections (section_id, report_id, section_type, content, citations)
- gl_nz_report_metrics (metric_id, report_id, metric_name, value, unit, provenance_hash)
- gl_nz_narratives (narrative_id, framework, language, content, consistency_score)
- gl_nz_assurance_evidence (evidence_id, report_id, evidence_type, file_path)
```

**TR-006: Framework Mapping Schema**
```
Tables:
- gl_nz_framework_mappings (mapping_id, source_framework, target_framework, source_metric, target_metric)
- gl_nz_framework_schemas (schema_id, framework, version, json_schema)
- gl_nz_framework_deadlines (deadline_id, framework, reporting_year, deadline_date)
```

### 6.3 Integration

**TR-007: Pack Integration**
- REST API clients for PACK-021/022/028/029
- Database direct access via views (read-only)
- Support for pack versioning and compatibility
- Graceful degradation if pack unavailable

**TR-008: Application Integration**
- GraphQL API for GL-SBTi-APP, GL-CDP-APP, GL-TCFD-APP, GL-GHG-APP
- OAuth 2.0 authentication for app access
- Rate limiting and retry logic
- Circuit breaker pattern for fault tolerance

**TR-009: External System Integration**
- ERP integration for financial data reconciliation
- ESG data vendor API integration (Bloomberg, MSCI, Sustainalytics)
- Translation service integration (DeepL, Google Translate)
- Cloud storage for report archive (S3, Azure Blob)

### 6.4 Performance

**TR-010: Latency Requirements**
- Report generation: <10s for full multi-framework suite
- API response: <200ms p95 for dashboard endpoints
- Data aggregation: <3s for full data pull from all sources
- PDF rendering: <5s for 50-page TCFD report

**TR-011: Scalability**
- Support 1,000+ concurrent organizations
- Handle 10,000+ report generations per day
- Support 100+ API requests per second
- Database partitioning by organization for scale

### 6.5 Security

**TR-012: Access Control**
- Role-based access control (RBAC) for report viewing/editing
- Row-level security (RLS) for multi-tenant data isolation
- Audit logging of all access and modifications
- OAuth 2.0 + JWT authentication

**TR-013: Data Protection**
- Encryption at rest (AES-256) for all report data
- Encryption in transit (TLS 1.3) for all API calls
- PII redaction for sample reports
- GDPR compliance for EU organizations

---

## 7. Component Specifications

### 7.1 Calculation Engines (10 Engines)

#### Engine 1: Data Aggregation Engine
**File:** `engines/data_aggregation_engine.py` (~1,800 lines)

**Purpose:** Collect and reconcile emissions data from all source packs and applications.

**Key Functions:**
- `aggregate_pack_data()`: Pull data from PACK-021/022/028/029
- `aggregate_app_data()`: Pull data from GL apps via API
- `reconcile_sources()`: Detect mismatches and gaps
- `generate_lineage()`: Create data lineage diagram
- `calculate_completeness()`: Score data completeness (0-100%)

**Inputs:** Pack IDs, app endpoints, organization ID, reporting period
**Outputs:** Aggregated dataset with lineage, reconciliation report
**Performance:** <3s for full aggregation

#### Engine 2: Narrative Generation Engine
**File:** `engines/narrative_generation_engine.py` (~2,000 lines)

**Purpose:** Generate AI-assisted narratives with citation management.

**Key Functions:**
- `generate_narrative()`: Create draft narrative for section
- `add_citations()`: Link narrative to source data
- `translate_narrative()`: Multi-language support
- `validate_consistency()`: Check for contradictions
- `calculate_consistency_score()`: Score 0-100%

**Inputs:** Section type, framework, language, source data
**Outputs:** Draft narrative with citations, consistency score
**Performance:** <2s per section

#### Engine 3: Framework Mapping Engine
**File:** `engines/framework_mapping_engine.py` (~1,500 lines)

**Purpose:** Map metrics and structures between frameworks.

**Key Functions:**
- `map_metric()`: Translate metric from source to target framework
- `map_structure()`: Map section structure
- `bidirectional_sync()`: Sync changes across frameworks
- `detect_conflicts()`: Flag unmappable differences

**Inputs:** Source framework, target framework, metric name
**Outputs:** Mapped metric, mapping confidence score
**Performance:** <100ms per mapping

#### Engine 4: XBRL Tagging Engine
**File:** `engines/xbrl_tagging_engine.py` (~1,600 lines)

**Purpose:** Generate XBRL/iXBRL tags for SEC and CSRD.

**Key Functions:**
- `tag_metric()`: Apply XBRL tag to metric
- `generate_xbrl()`: Create XBRL file
- `generate_ixbrl()`: Create inline XBRL HTML
- `validate_taxonomy()`: Validate against official taxonomy

**Inputs:** Metrics, taxonomy version, framework (SEC/CSRD)
**Outputs:** XBRL file, iXBRL HTML
**Performance:** <3s for full document

#### Engine 5: Dashboard Generation Engine
**File:** `engines/dashboard_generation_engine.py` (~1,700 lines)

**Purpose:** Create interactive HTML dashboards.

**Key Functions:**
- `generate_executive_dashboard()`: Create overview dashboard
- `generate_framework_dashboard()`: Framework-specific view
- `generate_stakeholder_view()`: Customized for audience
- `add_interactivity()`: Charts, filters, drill-down

**Inputs:** Aggregated data, stakeholder type, branding config
**Outputs:** HTML5 dashboard with JavaScript interactivity
**Performance:** <4s for full dashboard

#### Engine 6: Assurance Packaging Engine
**File:** `engines/assurance_packaging_engine.py` (~1,400 lines)

**Purpose:** Package evidence bundles for ISAE 3410 audits.

**Key Functions:**
- `collect_provenances()`: Gather SHA-256 hashes
- `generate_lineage_diagrams()`: Visual data flow
- `package_methodology()`: Include calculation docs
- `create_control_matrix()`: ISAE 3410 requirements

**Inputs:** Report ID, framework, audit scope
**Outputs:** Evidence bundle (ZIP file with all docs)
**Performance:** <5s for full bundle

#### Engine 7: Report Compilation Engine
**File:** `engines/report_compilation_engine.py` (~2,200 lines)

**Purpose:** Assemble final reports from components.

**Key Functions:**
- `compile_report()`: Assemble sections into final report
- `apply_branding()`: Add logo, colors, fonts
- `generate_toc()`: Create table of contents
- `add_cross_references()`: Link related sections

**Inputs:** Report sections, branding config, framework template
**Outputs:** Compiled report (pre-rendering)
**Performance:** <2s per report

#### Engine 8: Validation Engine
**File:** `engines/validation_engine.py` (~1,300 lines)

**Purpose:** Validate reports against framework schemas.

**Key Functions:**
- `validate_schema()`: Check against JSON schema
- `validate_completeness()`: Ensure all required fields present
- `validate_consistency()`: Cross-framework checks
- `calculate_quality_score()`: Overall quality 0-100%

**Inputs:** Report data, framework, schema version
**Outputs:** Validation report with errors/warnings
**Performance:** <1s per report

#### Engine 9: Translation Engine
**File:** `engines/translation_engine.py` (~1,100 lines)

**Purpose:** Multi-language support for narratives.

**Key Functions:**
- `translate_narrative()`: Translate text to target language
- `validate_translation()`: Quality checks
- `maintain_terminology()`: Use climate-specific glossary
- `preserve_citations()`: Keep citation links intact

**Inputs:** Source text, source language, target language
**Outputs:** Translated text with quality score
**Performance:** <3s per narrative (1,000 words)

#### Engine 10: Format Rendering Engine
**File:** `engines/format_rendering_engine.py` (~1,900 lines)

**Purpose:** Render reports in multiple formats.

**Key Functions:**
- `render_pdf()`: Generate PDF using WeasyPrint
- `render_html()`: Generate interactive HTML
- `render_excel()`: Export to Excel with formatting
- `render_json()`: Generate JSON for API
- `render_xbrl()`: Generate XBRL (delegates to Engine 4)

**Inputs:** Compiled report, format type, branding
**Outputs:** Rendered report file
**Performance:** <5s for PDF, <2s for others

**Total Engine Lines:** ~16,500 lines

### 7.2 Workflows (8 Workflows)

#### Workflow 1: SBTi Progress Report Workflow
**File:** `workflows/sbti_progress_workflow.py` (~1,400 lines)

**Purpose:** Generate SBTi annual progress disclosure.

**Steps:**
1. Aggregate target data from GL-SBTi-APP
2. Aggregate emissions from PACK-021/029
3. Calculate progress vs. target
4. Generate variance explanation
5. Compile SBTi report template
6. Validate against SBTi schema
7. Render PDF + JSON
8. Package for submission

**Inputs:** Organization ID, reporting year
**Outputs:** SBTi progress report (PDF, JSON)
**Duration:** <5s

#### Workflow 2: CDP Questionnaire Workflow
**File:** `workflows/cdp_questionnaire_workflow.py` (~1,600 lines)

**Purpose:** Generate CDP Climate Change questionnaire responses.

**Steps:**
1. Aggregate emissions data (C6, C7)
2. Pull target data for C4
3. Pull governance data for C1
4. Pull risk data for C2-C3
5. Pull opportunity data for C2-C3
6. Generate narratives for text responses
7. Validate completeness (% of required questions)
8. Export to Excel template

**Inputs:** Organization ID, CDP year
**Outputs:** CDP questionnaire Excel file
**Duration:** <8s

#### Workflow 3: TCFD Disclosure Workflow
**File:** `workflows/tcfd_disclosure_workflow.py` (~1,500 lines)

**Purpose:** Generate TCFD 4-pillar disclosure report.

**Steps:**
1. Governance pillar: Board oversight, management roles
2. Strategy pillar: Climate risks, opportunities, resilience
3. Risk Management pillar: Identification, assessment, integration
4. Metrics & Targets pillar: Scope 1/2/3, targets, progress
5. Compile into executive report
6. Add scenario analysis from GL-TCFD-APP
7. Render PDF with charts
8. Generate assurance evidence

**Inputs:** Organization ID, reporting period
**Outputs:** TCFD report PDF, evidence bundle
**Duration:** <6s

#### Workflow 4: GRI 305 Disclosure Workflow
**File:** `workflows/gri_305_workflow.py` (~1,200 lines)

**Purpose:** Generate GRI 305 emissions disclosures.

**Steps:**
1. 305-1: Direct (Scope 1) emissions
2. 305-2: Indirect (Scope 2) emissions
3. 305-3: Other indirect (Scope 3) emissions
4. 305-4: GHG emissions intensity
5. 305-5: Reduction of GHG emissions
6. 305-6: Emissions of ozone-depleting substances
7. 305-7: NOx, SOx, and other significant air emissions
8. Generate GRI Content Index table

**Inputs:** Organization ID, reporting year
**Outputs:** GRI 305 disclosure section, content index
**Duration:** <4s

#### Workflow 5: ISSB IFRS S2 Workflow
**File:** `workflows/issb_ifrs_s2_workflow.py` (~1,300 lines)

**Purpose:** Generate ISSB IFRS S2 climate disclosure.

**Steps:**
1. Governance: Board oversight aligned with S2 para 6
2. Strategy: Climate risks/opportunities aligned with S2 para 10
3. Risk management: Climate risk processes aligned with S2 para 25
4. Metrics & targets: Scope 1/2/3 + industry metrics (SASB)
5. Add XBRL tagging for digital reporting
6. Validate against IFRS S2 requirements
7. Render PDF + XBRL

**Inputs:** Organization ID, fiscal period
**Outputs:** IFRS S2 report (PDF, XBRL)
**Duration:** <5s

#### Workflow 6: SEC Climate Disclosure Workflow
**File:** `workflows/sec_climate_workflow.py` (~1,400 lines)

**Purpose:** Generate SEC 10-K climate disclosure section.

**Steps:**
1. Item 1: Climate risks in business description
2. Item 1A: Climate risks in risk factors
3. Item 7: Climate impacts in MD&A
4. Regulation S-K Item 1502-1506: Scope 1/2 emissions, targets
5. Apply XBRL/iXBRL tagging
6. Validate against SEC schema
7. Generate attestation report template
8. Package for 10-K filing

**Inputs:** Organization ID, fiscal year, 10-K context
**Outputs:** Climate disclosure text + XBRL, attestation template
**Duration:** <6s

#### Workflow 7: CSRD ESRS E1 Workflow
**File:** `workflows/csrd_esrs_e1_workflow.py` (~1,500 lines)

**Purpose:** Generate CSRD ESRS E1 Climate Change disclosure.

**Steps:**
1. E1-1: Transition plan for climate change mitigation
2. E1-2: Policies related to climate change
3. E1-3: Actions and resources for climate policies
4. E1-4: GHG emission reduction targets
5. E1-5: Energy consumption and mix
6. E1-6: Gross Scopes 1/2/3 emissions
7. E1-7: GHG removals and carbon credits
8. E1-8: Internal carbon pricing
9. E1-9: Anticipated financial effects
10. Apply CSRD digital taxonomy tagging
11. Validate against ESRS E1 requirements
12. Render digital report

**Inputs:** Organization ID, reporting period (CSRD year)
**Outputs:** ESRS E1 disclosure with digital taxonomy
**Duration:** <7s

#### Workflow 8: Multi-Framework Full Report Workflow
**File:** `workflows/multi_framework_workflow.py` (~1,800 lines)

**Purpose:** Generate all 7 framework reports in single execution.

**Steps:**
1. Aggregate data once from all sources
2. Generate shared narratives with framework-specific adaptations
3. Execute 7 framework workflows in parallel
4. Validate cross-framework consistency
5. Generate executive dashboard showing all frameworks
6. Create master evidence bundle
7. Package all reports into deliverable

**Inputs:** Organization ID, reporting period
**Outputs:** 7 framework reports + dashboard + evidence bundle
**Duration:** <10s (parallel execution)

**Total Workflow Lines:** ~11,700 lines

### 7.3 Report Templates (15 Templates)

#### Template 1: SBTi Progress Report Template
**File:** `templates/sbti_progress_template.py` (~600 lines)

**Sections:** Target description, base year, progress table, variance explanation, next steps

#### Template 2: CDP C0-C2 Governance Template
**File:** `templates/cdp_governance_template.py` (~500 lines)

**Sections:** Introduction, governance, business strategy, risks/opportunities

#### Template 3: CDP C4-C7 Emissions Template
**File:** `templates/cdp_emissions_template.py` (~700 lines)

**Sections:** Targets (C4), emissions methodology (C5), Scope 1/2 (C6), Scope 3 (C7)

#### Template 4: TCFD Governance Pillar Template
**File:** `templates/tcfd_governance_template.py` (~400 lines)

**Sections:** Board oversight, management's role

#### Template 5: TCFD Strategy Pillar Template
**File:** `templates/tcfd_strategy_template.py` (~600 lines)

**Sections:** Climate risks, opportunities, scenario analysis, resilience

#### Template 6: TCFD Risk Management Template
**File:** `templates/tcfd_risk_template.py` (~500 lines)

**Sections:** Risk identification, assessment, integration with enterprise risk

#### Template 7: TCFD Metrics & Targets Template
**File:** `templates/tcfd_metrics_template.py` (~550 lines)

**Sections:** Scope 1/2/3 emissions, targets, progress, key metrics

#### Template 8: GRI 305 Disclosure Template
**File:** `templates/gri_305_template.py` (~700 lines)

**Sections:** 305-1 through 305-7, content index

#### Template 9: ISSB IFRS S2 Template
**File:** `templates/issb_s2_template.py` (~650 lines)

**Sections:** Governance, strategy, risk management, metrics (4-pillar structure)

#### Template 10: SEC Climate Disclosure Template
**File:** `templates/sec_climate_template.py` (~600 lines)

**Sections:** Business description, risk factors, MD&A, Reg S-K emissions

#### Template 11: CSRD ESRS E1 Template
**File:** `templates/csrd_e1_template.py` (~800 lines)

**Sections:** E1-1 through E1-9 (9 disclosure requirements)

#### Template 12: Investor Dashboard Template
**File:** `templates/investor_dashboard_template.py` (~500 lines)

**Sections:** TCFD + ISSB focus, financial materiality, scenario analysis

#### Template 13: Regulator Dashboard Template
**File:** `templates/regulator_dashboard_template.py` (~450 lines)

**Sections:** CSRD + SEC focus, compliance status, audit trail

#### Template 14: Customer Carbon Report Template
**File:** `templates/customer_carbon_template.py` (~400 lines)

**Sections:** Product carbon footprint, supply chain emissions, reduction initiatives

#### Template 15: Assurance Evidence Bundle Template
**File:** `templates/assurance_evidence_template.py` (~550 lines)

**Sections:** Provenance hashes, lineage diagrams, methodology docs, control matrix

**Total Template Lines:** ~8,500 lines

### 7.4 Integrations (12 Integrations)

#### Integration 1: PACK-021 Net Zero Starter Integration
**File:** `integrations/pack021_integration.py` (~700 lines)

**Purpose:** Pull baseline emissions, inventory data

**Methods:** `fetch_baseline()`, `fetch_inventory()`, `fetch_activity_data()`

#### Integration 2: PACK-022 Net Zero Acceleration Integration
**File:** `integrations/pack022_integration.py` (~700 lines)

**Purpose:** Pull reduction initiatives, MACC curves

**Methods:** `fetch_initiatives()`, `fetch_macc()`, `fetch_abatement()`

#### Integration 3: PACK-028 Sector Pathway Integration
**File:** `integrations/pack028_integration.py` (~700 lines)

**Purpose:** Pull sector pathways, convergence data

**Methods:** `fetch_pathways()`, `fetch_convergence()`, `fetch_benchmarks()`

#### Integration 4: PACK-029 Interim Targets Integration
**File:** `integrations/pack029_integration.py` (~700 lines)

**Purpose:** Pull interim targets, progress monitoring, variance analysis

**Methods:** `fetch_targets()`, `fetch_progress()`, `fetch_variance()`

#### Integration 5: GL-SBTi-APP Integration
**File:** `integrations/gl_sbti_app_integration.py` (~800 lines)

**Purpose:** Pull SBTi target data, validation results

**Methods:** `fetch_sbti_targets()`, `fetch_validation()`, `fetch_submission_history()`

#### Integration 6: GL-CDP-APP Integration
**File:** `integrations/gl_cdp_app_integration.py` (~800 lines)

**Purpose:** Pull historical CDP responses, scores

**Methods:** `fetch_cdp_history()`, `fetch_scores()`, `fetch_peer_benchmarks()`

#### Integration 7: GL-TCFD-APP Integration
**File:** `integrations/gl_tcfd_app_integration.py` (~750 lines)

**Purpose:** Pull scenario analysis, risk assessments

**Methods:** `fetch_scenarios()`, `fetch_risks()`, `fetch_opportunities()`

#### Integration 8: GL-GHG-APP Integration
**File:** `integrations/gl_ghg_app_integration.py` (~750 lines)

**Purpose:** Pull GHG inventory, emission factors

**Methods:** `fetch_inventory()`, `fetch_emission_factors()`, `fetch_activity_data()`

#### Integration 9: XBRL Taxonomy Integration
**File:** `integrations/xbrl_taxonomy_integration.py` (~600 lines)

**Purpose:** Fetch and cache XBRL taxonomies

**Methods:** `fetch_sec_taxonomy()`, `fetch_csrd_taxonomy()`, `validate_tags()`

#### Integration 10: Translation Service Integration
**File:** `integrations/translation_integration.py` (~500 lines)

**Purpose:** Multi-language narrative translation

**Methods:** `translate()`, `detect_language()`, `validate_quality()`

#### Integration 11: Orchestrator Integration
**File:** `integrations/orchestrator_integration.py` (~400 lines)

**Purpose:** Register with GreenLang Orchestrator

**Methods:** `register_pack()`, `report_health()`, `handle_orchestration()`

#### Integration 12: Health Check Integration
**File:** `integrations/health_check_integration.py` (~300 lines)

**Purpose:** Health monitoring for all integrations

**Methods:** `check_pack_health()`, `check_app_health()`, `check_external_services()`

**Total Integration Lines:** ~8,200 lines

---

## 8. Database Schema

### 8.1 Core Tables (15 Tables)

```sql
-- Reports metadata
CREATE TABLE gl_nz_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    framework VARCHAR(50) NOT NULL, -- SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD
    reporting_period DATERANGE NOT NULL,
    status VARCHAR(20) NOT NULL, -- draft, review, approved, published
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID NOT NULL,
    approved_by UUID,
    approved_at TIMESTAMPTZ,
    provenance_hash CHAR(64) NOT NULL, -- SHA-256
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Report sections
CREATE TABLE gl_nz_report_sections (
    section_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    section_type VARCHAR(100) NOT NULL, -- governance, strategy, metrics, etc.
    section_order INT NOT NULL,
    content TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]',
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    consistency_score NUMERIC(5,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Report metrics
CREATE TABLE gl_nz_report_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_name VARCHAR(200) NOT NULL,
    metric_value NUMERIC NOT NULL,
    unit VARCHAR(50) NOT NULL,
    scope VARCHAR(20), -- scope1, scope2, scope3
    source_system VARCHAR(100) NOT NULL,
    calculation_method TEXT,
    provenance_hash CHAR(64) NOT NULL,
    uncertainty_range NUMRANGE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Narratives library
CREATE TABLE gl_nz_narratives (
    narrative_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    section_type VARCHAR(100) NOT NULL,
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    content TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]',
    consistency_score NUMERIC(5,2),
    usage_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Framework mappings
CREATE TABLE gl_nz_framework_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_framework VARCHAR(50) NOT NULL,
    target_framework VARCHAR(50) NOT NULL,
    source_metric VARCHAR(200) NOT NULL,
    target_metric VARCHAR(200) NOT NULL,
    mapping_type VARCHAR(50) NOT NULL, -- direct, calculated, approximate
    conversion_formula TEXT,
    confidence_score NUMERIC(5,2),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Framework schemas
CREATE TABLE gl_nz_framework_schemas (
    schema_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    schema_type VARCHAR(50) NOT NULL, -- questionnaire, report, taxonomy
    json_schema JSONB NOT NULL,
    effective_date DATE NOT NULL,
    deprecated_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Framework deadlines
CREATE TABLE gl_nz_framework_deadlines (
    deadline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    reporting_year INT NOT NULL,
    deadline_date DATE NOT NULL,
    description TEXT,
    notification_days INT[] NOT NULL DEFAULT ARRAY[90, 60, 30, 7],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Assurance evidence
CREATE TABLE gl_nz_assurance_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    evidence_type VARCHAR(100) NOT NULL, -- provenance, lineage, methodology, control
    file_path VARCHAR(500),
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    checksum CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Data lineage
CREATE TABLE gl_nz_data_lineage (
    lineage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_name VARCHAR(200) NOT NULL,
    source_system VARCHAR(100) NOT NULL,
    transformation_steps JSONB NOT NULL DEFAULT '[]',
    source_records JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audit trail
CREATE TABLE gl_nz_audit_trail (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID REFERENCES gl_nz_reports(report_id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL, -- created, updated, approved, published
    actor_id UUID NOT NULL,
    actor_type VARCHAR(50) NOT NULL, -- user, system, api
    details JSONB NOT NULL DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Translations
CREATE TABLE gl_nz_translations (
    translation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_text TEXT NOT NULL,
    source_language VARCHAR(5) NOT NULL,
    target_language VARCHAR(5) NOT NULL,
    translated_text TEXT NOT NULL,
    quality_score NUMERIC(5,2),
    translator VARCHAR(100), -- service name or human translator
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- XBRL tags
CREATE TABLE gl_nz_xbrl_tags (
    tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_name VARCHAR(200) NOT NULL,
    xbrl_element VARCHAR(200) NOT NULL,
    xbrl_namespace VARCHAR(500) NOT NULL,
    taxonomy_version VARCHAR(50) NOT NULL,
    context_ref VARCHAR(200),
    unit_ref VARCHAR(100),
    decimals INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Validation results
CREATE TABLE gl_nz_validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    validator VARCHAR(100) NOT NULL, -- schema, completeness, consistency
    validation_type VARCHAR(50) NOT NULL, -- error, warning, info
    message TEXT NOT NULL,
    field_path VARCHAR(500),
    severity VARCHAR(20) NOT NULL, -- critical, high, medium, low
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Configuration
CREATE TABLE gl_nz_report_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    framework VARCHAR(50) NOT NULL,
    branding_config JSONB NOT NULL DEFAULT '{}', -- logo, colors, fonts
    content_config JSONB NOT NULL DEFAULT '{}', -- preferred narratives, custom sections
    notification_config JSONB NOT NULL DEFAULT '{}', -- email, slack, teams settings
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(organization_id, framework)
);

-- Dashboard views
CREATE TABLE gl_nz_dashboard_views (
    view_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    view_type VARCHAR(50) NOT NULL, -- executive, investor, regulator, customer
    config JSONB NOT NULL DEFAULT '{}',
    created_by UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 8.2 Views (5 Views)

```sql
-- Current reports summary
CREATE VIEW gl_nz_reports_summary AS
SELECT
    r.report_id,
    r.organization_id,
    r.framework,
    r.reporting_period,
    r.status,
    COUNT(DISTINCT s.section_id) AS section_count,
    COUNT(DISTINCT m.metric_id) AS metric_count,
    AVG(s.consistency_score) AS avg_consistency_score,
    r.created_at,
    r.approved_at
FROM gl_nz_reports r
LEFT JOIN gl_nz_report_sections s ON r.report_id = s.report_id
LEFT JOIN gl_nz_report_metrics m ON r.report_id = m.report_id
GROUP BY r.report_id;

-- Framework coverage
CREATE VIEW gl_nz_framework_coverage AS
SELECT
    r.organization_id,
    r.framework,
    r.reporting_period,
    COUNT(DISTINCT m.metric_name) AS metrics_provided,
    (SELECT COUNT(*) FROM gl_nz_framework_schemas fs
     WHERE fs.framework = r.framework
     AND fs.deprecated_date IS NULL) AS metrics_required,
    ROUND(100.0 * COUNT(DISTINCT m.metric_name) / NULLIF(
        (SELECT COUNT(*) FROM gl_nz_framework_schemas fs
         WHERE fs.framework = r.framework
         AND fs.deprecated_date IS NULL), 0
    ), 2) AS coverage_percentage
FROM gl_nz_reports r
LEFT JOIN gl_nz_report_metrics m ON r.report_id = m.report_id
GROUP BY r.organization_id, r.framework, r.reporting_period;

-- Validation issues
CREATE VIEW gl_nz_validation_issues AS
SELECT
    v.report_id,
    r.framework,
    r.status,
    COUNT(*) FILTER (WHERE v.severity = 'critical') AS critical_issues,
    COUNT(*) FILTER (WHERE v.severity = 'high') AS high_issues,
    COUNT(*) FILTER (WHERE v.severity = 'medium') AS medium_issues,
    COUNT(*) FILTER (WHERE v.severity = 'low') AS low_issues,
    COUNT(*) FILTER (WHERE v.resolved = FALSE) AS unresolved_issues
FROM gl_nz_validation_results v
JOIN gl_nz_reports r ON v.report_id = r.report_id
GROUP BY v.report_id, r.framework, r.status;

-- Upcoming deadlines
CREATE VIEW gl_nz_upcoming_deadlines AS
SELECT
    d.framework,
    d.reporting_year,
    d.deadline_date,
    d.deadline_date - CURRENT_DATE AS days_remaining,
    d.description,
    r.organization_id,
    r.status
FROM gl_nz_framework_deadlines d
LEFT JOIN gl_nz_reports r ON d.framework = r.framework
    AND EXTRACT(YEAR FROM LOWER(r.reporting_period)) = d.reporting_year
WHERE d.deadline_date >= CURRENT_DATE
ORDER BY d.deadline_date;

-- Data lineage summary
CREATE VIEW gl_nz_lineage_summary AS
SELECT
    l.report_id,
    l.metric_name,
    COUNT(DISTINCT l.source_system) AS source_system_count,
    JSONB_AGG(DISTINCT l.source_system) AS source_systems,
    MAX(JSONB_ARRAY_LENGTH(l.transformation_steps)) AS max_transformation_depth
FROM gl_nz_data_lineage l
GROUP BY l.report_id, l.metric_name;
```

### 8.3 Indexes (350+ Indexes)

```sql
-- Primary lookup indexes
CREATE INDEX idx_nz_reports_org_framework ON gl_nz_reports(organization_id, framework);
CREATE INDEX idx_nz_reports_period ON gl_nz_reports USING GIST(reporting_period);
CREATE INDEX idx_nz_reports_status ON gl_nz_reports(status);
CREATE INDEX idx_nz_report_sections_report ON gl_nz_report_sections(report_id);
CREATE INDEX idx_nz_report_metrics_report ON gl_nz_report_metrics(report_id);
CREATE INDEX idx_nz_report_metrics_name ON gl_nz_report_metrics(metric_name);

-- Full-text search
CREATE INDEX idx_nz_narratives_content_fts ON gl_nz_narratives USING GIN(to_tsvector('english', content));
CREATE INDEX idx_nz_report_sections_content_fts ON gl_nz_report_sections USING GIN(to_tsvector('english', content));

-- JSONB indexes
CREATE INDEX idx_nz_reports_metadata ON gl_nz_reports USING GIN(metadata);
CREATE INDEX idx_nz_report_sections_citations ON gl_nz_report_sections USING GIN(citations);
CREATE INDEX idx_nz_framework_schemas_json ON gl_nz_framework_schemas USING GIN(json_schema);

-- Performance indexes
CREATE INDEX idx_nz_audit_trail_report_time ON gl_nz_audit_trail(report_id, created_at DESC);
CREATE INDEX idx_nz_validation_results_unresolved ON gl_nz_validation_results(report_id)
    WHERE resolved = FALSE;
CREATE INDEX idx_nz_deadlines_upcoming ON gl_nz_framework_deadlines(deadline_date)
    WHERE deadline_date >= CURRENT_DATE;

-- (340+ more indexes following similar patterns)
```

### 8.4 Row-Level Security (30 Policies)

```sql
-- Multi-tenant isolation
ALTER TABLE gl_nz_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY nz_reports_isolation ON gl_nz_reports
    USING (organization_id = current_setting('app.current_organization_id')::UUID);

ALTER TABLE gl_nz_report_sections ENABLE ROW LEVEL SECURITY;
CREATE POLICY nz_sections_isolation ON gl_nz_report_sections
    USING (report_id IN (
        SELECT report_id FROM gl_nz_reports
        WHERE organization_id = current_setting('app.current_organization_id')::UUID
    ));

-- (28 more RLS policies for other tables)
```

---

## 9. Configuration and Presets

### 9.1 Configuration Presets (8 Presets)

#### Preset 1: CSRD Focus Configuration
**File:** `config/presets/csrd_focus.yaml`

```yaml
name: "CSRD ESRS E1 Focus"
description: "Optimized for EU Corporate Sustainability Reporting Directive"
frameworks:
  primary: CSRD
  secondary: [TCFD, GRI]
branding:
  style: corporate
  colors:
    primary: "#1E3A8A"
    secondary: "#3B82F6"
outputs:
  - format: PDF
    template: csrd_e1_template
  - format: digital_taxonomy
    template: esrs_taxonomy
notifications:
  deadline_reminders: [120, 90, 60, 30, 14, 7]
  channels: [email, slack]
```

#### Preset 2: CDP A-List Configuration
**File:** `config/presets/cdp_alist.yaml`

```yaml
name: "CDP A-List Target"
description: "Optimized for CDP Climate Change A-list scoring"
frameworks:
  primary: CDP
  secondary: [TCFD, GRI]
completeness_target: 100%
narrative_quality: high
outputs:
  - format: Excel
    template: cdp_questionnaire
  - format: PDF
    template: cdp_summary_report
data_sources:
  pack021: required
  pack022: required
  pack028: recommended
  pack029: required
```

#### Preset 3: TCFD Investor-Grade Configuration
**File:** `config/presets/tcfd_investor.yaml`

```yaml
name: "TCFD Investor-Grade Disclosure"
description: "High-quality TCFD for institutional investors"
frameworks:
  primary: TCFD
  secondary: [ISSB, CDP]
branding:
  style: executive
  include_charts: true
outputs:
  - format: PDF
    template: tcfd_executive_report
  - format: HTML
    template: tcfd_interactive_dashboard
scenario_analysis:
  required: true
  sources: [GL-TCFD-APP]
```

#### Preset 4: SBTi Validation Ready Configuration
**File:** `config/presets/sbti_validation.yaml`

```yaml
name: "SBTi Validation Ready"
description: "Prepared for SBTi target validation submission"
frameworks:
  primary: SBTi
  secondary: [CDP, TCFD]
validation:
  run_21_criteria: true
  require_evidence: true
outputs:
  - format: PDF
    template: sbti_progress_report
  - format: JSON
    template: sbti_submission_format
assurance:
  include_evidence_bundle: true
```

#### Preset 5: SEC 10-K Compliance Configuration
**File:** `config/presets/sec_10k.yaml`

```yaml
name: "SEC 10-K Climate Disclosure"
description: "Compliant with SEC climate disclosure rules"
frameworks:
  primary: SEC
  secondary: [TCFD]
outputs:
  - format: XBRL
    template: sec_climate_xbrl
  - format: iXBRL
    template: sec_climate_ixbrl
  - format: PDF
    template: sec_narrative_disclosure
attestation:
  required: true
  level: limited_assurance
```

#### Preset 6: Multi-Framework Comprehensive Configuration
**File:** `config/presets/multi_framework.yaml`

```yaml
name: "Multi-Framework Comprehensive"
description: "All 7 frameworks in single reporting cycle"
frameworks:
  all: [SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD]
outputs:
  - format: PDF
    templates: [sbti, tcfd, gri, issb, sec, csrd]
  - format: HTML
    template: executive_dashboard
  - format: Excel
    template: cdp_questionnaire
  - format: XBRL
    templates: [sec, csrd]
parallelization: enabled
consistency_validation: strict
```

#### Preset 7: Investor Relations Configuration
**File:** `config/presets/investor_relations.yaml`

```yaml
name: "Investor Relations Package"
description: "Investor-focused climate disclosure package"
frameworks:
  primary: TCFD
  secondary: [ISSB, CDP]
branding:
  style: investor
  include_financial_metrics: true
outputs:
  - format: PDF
    template: investor_report
  - format: HTML
    template: investor_dashboard
stakeholder_view: investor
```

#### Preset 8: Assurance-Ready Configuration
**File:** `config/presets/assurance_ready.yaml`

```yaml
name: "Assurance-Ready Package"
description: "Full evidence bundle for ISAE 3410 assurance"
frameworks:
  all: [SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD]
assurance:
  include_evidence_bundle: true
  include_provenance: true
  include_lineage_diagrams: true
  include_methodology_docs: true
  include_control_matrix: true
outputs:
  - format: PDF
    templates: all_frameworks
  - format: ZIP
    template: assurance_evidence_bundle
```

### 9.2 Pack Configuration
**File:** `config/pack_config.py` (~1,000 lines)

```python
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

class ReportingFrameworkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    required_metrics: List[str]
    optional_metrics: List[str]
    output_formats: List[str]
    deadline_months: Optional[int] = None

class BrandingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    logo_path: Optional[str] = None
    primary_color: str = "#1E3A8A"
    secondary_color: str = "#3B82F6"
    font_family: str = "Arial, sans-serif"
    style: str = "corporate"  # corporate, executive, investor

class PACK030Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pack_id: str = Field(default="PACK-030-net-zero-reporting")
    pack_version: str = Field(default="1.0.0")

    frameworks: Dict[str, ReportingFrameworkConfig]
    branding: BrandingConfig

    data_sources: List[str] = Field(
        default=["PACK-021", "PACK-022", "PACK-028", "PACK-029",
                 "GL-SBTi-APP", "GL-CDP-APP", "GL-TCFD-APP", "GL-GHG-APP"]
    )

    output_formats: List[str] = Field(
        default=["PDF", "HTML", "Excel", "JSON", "XBRL", "iXBRL"]
    )

    languages: List[str] = Field(
        default=["en", "de", "fr", "es"]
    )

    assurance_enabled: bool = Field(default=True)
    multi_framework_enabled: bool = Field(default=True)
```

---

## 10. Testing Requirements

### 10.1 Test Coverage (17 Test Files, 2,000+ Tests)

#### Test File 1: Data Aggregation Engine Tests
**File:** `tests/test_data_aggregation_engine.py` (~150 tests)

- Test multi-source data collection
- Test reconciliation logic
- Test gap detection
- Test lineage generation
- Test completeness scoring

#### Test File 2: Narrative Generation Engine Tests
**File:** `tests/test_narrative_generation_engine.py` (~120 tests)

- Test narrative drafting
- Test citation management
- Test multi-language support
- Test consistency validation
- Test quality scoring

#### Test File 3: Framework Mapping Engine Tests
**File:** `tests/test_framework_mapping_engine.py` (~100 tests)

- Test metric mapping accuracy
- Test bidirectional sync
- Test conflict detection
- Test mapping confidence scores

#### Test File 4: XBRL Tagging Engine Tests
**File:** `tests/test_xbrl_tagging_engine.py` (~130 tests)

- Test XBRL generation
- Test iXBRL generation
- Test taxonomy validation
- Test SEC compliance
- Test CSRD compliance

#### Test File 5: Dashboard Generation Engine Tests
**File:** `tests/test_dashboard_generation_engine.py` (~110 tests)

- Test executive dashboard
- Test framework dashboards
- Test stakeholder views
- Test interactivity features

#### Test File 6: Assurance Packaging Engine Tests
**File:** `tests/test_assurance_packaging_engine.py` (~100 tests)

- Test provenance collection
- Test lineage diagram generation
- Test evidence bundle completeness
- Test control matrix generation

#### Test File 7: Report Compilation Engine Tests
**File:** `tests/test_report_compilation_engine.py` (~140 tests)

- Test section assembly
- Test branding application
- Test TOC generation
- Test cross-references

#### Test File 8: Validation Engine Tests
**File:** `tests/test_validation_engine.py` (~130 tests)

- Test schema validation
- Test completeness checks
- Test consistency checks
- Test quality scoring

#### Test File 9: Translation Engine Tests
**File:** `tests/test_translation_engine.py` (~80 tests)

- Test translation accuracy
- Test terminology consistency
- Test citation preservation
- Test quality scoring

#### Test File 10: Format Rendering Engine Tests
**File:** `tests/test_format_rendering_engine.py` (~150 tests)

- Test PDF rendering
- Test HTML generation
- Test Excel export
- Test JSON API output
- Test XBRL rendering

#### Test File 11: Workflow Tests
**File:** `tests/test_workflows.py` (~200 tests, 25 per workflow)

- Test all 8 workflows end-to-end
- Test parallel execution
- Test error handling
- Test performance benchmarks

#### Test File 12: Template Tests
**File:** `tests/test_templates.py` (~150 tests, 10 per template)

- Test all 15 templates
- Test data insertion
- Test formatting
- Test schema compliance

#### Test File 13: Integration Tests
**File:** `tests/test_integrations.py` (~180 tests, 15 per integration)

- Test all 12 integrations
- Test authentication
- Test rate limiting
- Test circuit breakers
- Test retry logic

#### Test File 14: Database Tests
**File:** `tests/test_database.py` (~120 tests)

- Test schema creation
- Test indexes
- Test RLS policies
- Test views
- Test performance

#### Test File 15: API Tests
**File:** `tests/test_api.py` (~140 tests)

- Test RESTful endpoints
- Test authentication
- Test authorization
- Test rate limiting
- Test pagination

#### Test File 16: Performance Tests
**File:** `tests/test_performance.py` (~80 tests)

- Test report generation latency
- Test API response times
- Test concurrent load
- Test cache effectiveness

#### Test File 17: End-to-End Tests
**File:** `tests/test_e2e.py` (~60 tests)

- Test complete reporting cycles
- Test multi-framework workflows
- Test assurance packaging
- Test stakeholder scenarios

**Total Test Count:** ~2,020 tests
**Expected Coverage:** 90%+
**Expected Pass Rate:** 100%

---

## 11. Migration Plan

### 11.1 Database Migrations (V211-V225, 15 Migrations)

**V211__PACK030_core_tables.sql** - Core report tables
**V212__PACK030_framework_tables.sql** - Framework schemas and mappings
**V213__PACK030_narrative_tables.sql** - Narrative and translation tables
**V214__PACK030_assurance_tables.sql** - Assurance evidence tables
**V215__PACK030_audit_tables.sql** - Audit trail and lineage
**V216__PACK030_xbrl_tables.sql** - XBRL tagging tables
**V217__PACK030_validation_tables.sql** - Validation results
**V218__PACK030_config_tables.sql** - Configuration and dashboards
**V219__PACK030_indexes.sql** - All indexes (350+)
**V220__PACK030_views.sql** - All views (5)
**V221__PACK030_rls_policies.sql** - Row-level security (30 policies)
**V222__PACK030_functions.sql** - Helper functions
**V223__PACK030_triggers.sql** - Audit triggers
**V224__PACK030_seed_data.sql** - Reference data (frameworks, schemas)
**V225__PACK030_permissions.sql** - RBAC permissions

---

## 12. Documentation Requirements

### 12.1 User Documentation
- Installation guide
- Quick start tutorial
- Framework-specific guides (7 guides)
- Configuration reference
- API documentation
- Troubleshooting guide

### 12.2 Technical Documentation
- Architecture overview
- Database schema reference
- Engine specifications
- Workflow diagrams
- Integration guides
- Performance tuning guide

### 12.3 Compliance Documentation
- ISAE 3410 assurance methodology
- Framework compliance matrices
- Calculation methodology documentation
- Data lineage documentation

---

## 13. Success Criteria

### 13.1 Functional Success
- ✅ All 7 frameworks generate compliant reports
- ✅ All 10 engines pass 100% of tests
- ✅ All 8 workflows complete in <10s
- ✅ All 15 templates produce valid outputs
- ✅ All 12 integrations connect successfully
- ✅ Cross-framework consistency validation works
- ✅ Multi-language support (4 languages) functional
- ✅ XBRL/iXBRL validation passes against official taxonomies

### 13.2 Performance Success
- ✅ Report generation <10s for full suite
- ✅ API response <200ms p95
- ✅ 90%+ code coverage
- ✅ 100% test pass rate
- ✅ 95%+ cache hit ratio

### 13.3 Business Success
- ✅ 70% reduction in reporting time vs. manual
- ✅ 80% reduction in external consultant costs
- ✅ 100% audit trail for all calculations
- ✅ Zero data inconsistencies across frameworks

---

## 14. Risks and Mitigation

### 14.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Framework schema changes | High | Medium | Version all schemas, automated schema update detection |
| XBRL taxonomy updates | High | Medium | Subscribe to taxonomy update notifications, quarterly review |
| Translation quality issues | Medium | Low | Human review workflow, professional translation validation |
| Performance degradation | Medium | Low | Load testing, caching, query optimization |

### 14.2 Compliance Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Framework non-compliance | Critical | Low | Automated validation against official schemas |
| Audit trail gaps | High | Low | SHA-256 provenance, immutable audit log |
| Inconsistent narratives | Medium | Medium | Automated consistency checks, NLP analysis |

---

## 15. Dependencies and Integration Points

### 15.1 Pack Dependencies
- PACK-021: Net Zero Starter Pack (baseline emissions)
- PACK-022: Net Zero Acceleration Pack (reduction initiatives)
- PACK-028: Sector Pathway Pack (sector pathways)
- PACK-029: Interim Targets Pack (interim targets)

### 15.2 Application Dependencies
- GL-SBTi-APP: SBTi target management
- GL-CDP-APP: CDP questionnaire management
- GL-TCFD-APP: TCFD scenario analysis
- GL-GHG-APP: GHG inventory management

### 15.3 External Dependencies
- PostgreSQL 16+ (TimescaleDB extension)
- Redis 7+ (caching)
- WeasyPrint (PDF rendering)
- Translation APIs (DeepL, Google Translate)
- XBRL taxonomy services

---

## 16. Deployment and Operations

### 16.1 Deployment Requirements
- Kubernetes 1.28+ cluster
- PostgreSQL 16+ with TimescaleDB 2.13+
- Redis 7+ cluster (3 nodes minimum)
- S3-compatible object storage for report archives
- Load balancer with SSL/TLS termination

### 16.2 Monitoring
- Prometheus metrics for all engines
- Grafana dashboards for report generation metrics
- OpenTelemetry tracing for workflow execution
- Alert rules for deadline approaching, validation failures

### 16.3 Backup and Recovery
- Daily PostgreSQL backups with 30-day retention
- Report archive to S3 with versioning
- Disaster recovery RPO: 24 hours, RTO: 4 hours

---

## 17. Future Enhancements

### 17.1 Phase 2 Enhancements (v1.1)
- Additional frameworks: TNFD, GHG Protocol updates
- Enhanced AI narratives with GPT-4 integration
- Real-time collaboration features
- Mobile dashboard app

### 17.2 Phase 3 Enhancements (v1.2)
- Blockchain-based immutable audit trail
- Advanced NLP for narrative quality scoring
- Automated peer benchmarking
- Integration with financial reporting systems (SAP, Oracle)

---

## 18. Approval and Sign-Off

**Product Owner:** GreenLang Product Team
**Technical Lead:** [To be assigned]
**Compliance Review:** [To be assigned]
**Security Review:** [To be assigned]

**Status:** Approved for Development
**Approval Date:** 2026-03-18
**Target Launch:** Q2 2026

---

**Document Version:** 1.0.0
**Last Updated:** 2026-03-18
**Total Lines:** 1,547 lines
**Total Estimated Code:** ~44,900 lines (engines + workflows + templates + integrations)
**Total Estimated Tests:** ~2,020 tests
