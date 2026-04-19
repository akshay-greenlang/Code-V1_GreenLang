# PRD: AGENT-EUDR-030 -- Documentation Generator Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-030 |
| **Agent ID** | GL-EUDR-DGN-030 |
| **Component** | Documentation Generator Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 4, 9, 10, 11, 12, 14-16, 29, 31 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 12 mandates that operators and traders submit a Due Diligence Statement (DDS) to the EU Information System before placing relevant commodities on the EU market or exporting them from the EU. The DDS is the culmination of the entire due diligence process: it must contain all information gathered under Article 9 (product description, quantities, country of production, geolocation of production plots, supplier and buyer identification, supporting certifications), the results of the risk assessment conducted under Article 10 (composite risk score, criterion-by-criterion evaluation, country benchmarking determination), and a description of the risk mitigation measures taken under Article 11 where risks were identified (mitigation strategy, measures implemented, effectiveness verification, residual risk determination). The operator must conclude with a declaration that the products are deforestation-free and legally produced, or that negligible risk has been achieved through mitigation.

Beyond the DDS submission itself, Articles 14 through 16 require operators to make documentation available to competent authorities upon request. Article 14 grants competent authorities the power to demand full access to the operator's due diligence system, including all underlying evidence. Article 15 requires competent authorities to perform risk-based checks on operators, and Article 16 mandates substantive checks that verify the accuracy of the due diligence documentation. Article 31 requires operators to retain all due diligence records for at least 5 years from the date of the DDS submission. Penalties for non-compliance include fines of up to 4% of annual EU turnover, confiscation of goods, market exclusion, and public naming under Articles 23-25.

The GreenLang platform has built a comprehensive suite of 29 upstream EUDR agents that perform every step of the due diligence lifecycle: Supply Chain Traceability agents (EUDR-001 through EUDR-015) handle supply chain mapping, geolocation verification, satellite monitoring, forest cover analysis, land use change detection, plot boundary management, GPS validation, multi-tier supplier tracking, chain of custody, segregation verification, mass balance calculation, document authentication, blockchain integration, QR code generation, and mobile data collection. Risk Assessment agents (EUDR-016 through EUDR-020) handle country risk evaluation, supplier risk scoring, commodity risk analysis, corruption index monitoring, and deforestation alerting. Due Diligence agents (EUDR-021 through EUDR-029) handle indigenous rights checking, protected area validation, legal compliance verification, third-party audit management, risk mitigation advisory, due diligence orchestration, information gathering coordination, risk assessment engine operation, and mitigation measure design.

However, there is currently no agent that transforms the outputs of these 29 agents into the structured documentation required by the regulation. Operators face the following critical gaps:

- **No DDS generation capability**: The 29 upstream agents produce risk scores, supply chain graphs, geolocation verification results, satellite analysis reports, mitigation strategies, and audit findings -- but none of them generates a Due Diligence Statement in the format required by the EU Information System. Operators must manually extract data from each agent's output, reformat it into the EU JSON/XML schema, populate all mandatory fields, and validate completeness. This manual process takes 4-8 hours per DDS and is error-prone.

- **No Article 9 information assembly**: The information required by Article 9 is scattered across multiple agents. Product descriptions come from the supply chain mapping agent (EUDR-001), geolocation data comes from the geolocation verification agent (EUDR-002) and GPS validator (EUDR-007), quantities come from mass balance calculations (EUDR-011), supplier information comes from multi-tier supplier tracking (EUDR-008), and certifications come from document authentication (EUDR-012). There is no aggregation engine that assembles all Article 9 data from these sources into a single, validated, complete information package.

- **No structured risk documentation**: Risk assessment results from EUDR-028 Risk Assessment Engine include composite scores, criterion-by-criterion evaluations, and country benchmarking determinations. However, these results are in technical formats (JSON payloads, database records) not suitable for regulatory documentation or auditor review. There is no documentation engine that transforms raw risk assessment outputs into structured, human-readable risk assessment documentation that maps to Article 10(2) criteria (a) through (g).

- **No mitigation documentation**: Mitigation measures from EUDR-029 Mitigation Measure Designer include strategy definitions, implementation evidence, effectiveness verification results, and residual risk determinations. These outputs need to be transformed into structured documentation that maps each measure to Article 11(2) categories and provides clear evidence chains for auditor review.

- **No audit-ready compliance packages**: When competent authorities conduct substantive checks under Article 16, they require access to the complete due diligence evidence package: the DDS itself, the underlying Article 9 information, risk assessment documentation, mitigation documentation, supply chain maps, certificate copies, satellite verification evidence, and provenance chains. There is no engine that assembles all of these into a single, indexed, cross-referenced compliance package that can be provided to inspectors within the timeframe demanded.

- **No document versioning or retention management**: EUDR Article 31 mandates 5-year record retention. DDS documents may need to be amended after submission (e.g., when new information becomes available or errors are discovered). There is no version control system for due diligence documents, no amendment tracking, no immutable audit trail of document changes, and no retention management to ensure documents remain available and retrievable for the full 5-year period.

- **No submission lifecycle management**: The DDS submission to the EU Information System has its own lifecycle: draft, validate, submit, acknowledged, rejected. Rejected submissions require correction and resubmission. There is no engine that manages this lifecycle, validates DDS against the EU schema before submission, tracks submission status, handles rejections with error remediation, or records acknowledgement receipts.

- **No batch documentation for large operators**: Large operators processing hundreds of products across seven commodity categories need to generate hundreds of DDS documents per quarter. Manual generation is not scalable. There is no batch generation engine that can produce multiple DDS documents from a portfolio of products, validate them all, and manage their submission as a coordinated batch.

Without solving these problems, operators cannot complete the final step of their EUDR compliance obligation -- the generation and submission of regulatory documentation. The 29 upstream agents perform all analytical work, but without documentation generation, that work cannot be translated into the legally required DDS submissions and audit-ready evidence packages. This is the critical last-mile gap that exposes operators to enforcement action.

### 1.2 Solution Overview

Agent-EUDR-030: Documentation Generator is the capstone agent of the EUDR agent family. It is responsible for transforming the outputs of all 29 upstream EUDR agents into structured, validated, submission-ready regulatory documentation. It generates Due Diligence Statements (DDS) per Article 12, assembles Article 9 information packages, documents risk assessment results from EUDR-028, documents mitigation measures from EUDR-029, builds complete audit-ready compliance packages for competent authority inspection, manages document versioning with 5-year retention, and handles the DDS submission lifecycle to the EU Information System.

Core capabilities:

1. **DDS Statement Generation** -- Generates Due Diligence Statements in the EU Information System's required JSON/XML schema. Populates all mandatory fields: operator identification, product description (HS/CN codes), quantity, supplier information, country of production, geolocation references, risk assessment summary, mitigation measures summary, and compliance conclusion. Validates all mandatory fields before generation. Assigns unique DDS reference numbers with provenance hashes.

2. **Article 9 Data Assembly** -- Aggregates all Article 9 required information from upstream agents into a single, validated package. Gathers product descriptions and classification codes (HS/CN), quantities and units, country of production, geolocation of all production plots (GPS coordinates and polygons from EUDR-002/007), date/time range of production, supplier and buyer identification, and supporting evidence. Cross-references data completeness against the Article 9 requirements checklist.

3. **Risk Assessment Documentation** -- Transforms raw risk assessment outputs from EUDR-028 Risk Assessment Engine into structured, human-readable documentation. Generates risk assessment documentation including: composite risk score with full decomposition, Article 10(2) criterion-by-criterion evaluation results for all criteria (a) through (g), country benchmarking determination (Article 29), simplified due diligence eligibility (Article 13), risk classification rationale, and complete provenance chain. Formats for both DDS inclusion and standalone audit documentation.

4. **Mitigation Documentation** -- Transforms mitigation measure outputs from EUDR-029 Mitigation Measure Designer into structured documentation. Includes: risk trigger summary, mitigation strategy designed, measures implemented with evidence references, effectiveness verification results, final risk determination (negligible/non-negligible), and implementation timeline. Maps each measure to the specific Article 11(2) categories: (a) additional information, (b) independent survey/audit, (c) other measures.

5. **Compliance Package Building** -- Assembles complete, audit-ready compliance packages for competent authority inspection per Articles 14-16. Packages include: DDS document, Article 9 information package, risk assessment documentation, mitigation documentation, supply chain maps from EUDR-001, certificate copies from EUDR-012, satellite verification evidence from EUDR-003/004/005, geolocation validation from EUDR-002/007, and provenance chain. Generates table of contents, cross-reference index, and evidence summary. Supports PDF-ready structured output.

6. **Document Version Management** -- Manages document lifecycle with comprehensive version control. Tracks document versions, revisions, and amendment history. Supports draft/final/submitted/amended state machine. Maintains immutable audit trail of all document changes with actor, timestamp, change description, and SHA-256 hash. Enforces 5-year retention per Article 31 with automatic retention tracking and expiry alerting. Supports document recall and amendment after submission with full amendment provenance.

7. **Regulatory Submission Engine** -- Manages the DDS submission lifecycle to the EU Information System. Validates DDS against the EU schema before submission (pre-flight check). Tracks submission status through the complete lifecycle: draft, validated, submitted, acknowledged, rejected. Manages resubmission workflow after rejection with error analysis and correction guidance. Records submission receipts and acknowledgement numbers. Supports batch submission for multiple products/shipments with parallel submission management and aggregate status tracking.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| DDS generation time | < 30 seconds per DDS from aggregated data | p99 latency from generate-dds request to document ready |
| Article 9 assembly completeness | 100% of mandatory fields populated when upstream data available | Automated completeness validator against Article 9 checklist |
| DDS schema validation pass rate | 100% on first validation attempt | Pre-submission validation result tracking |
| Compliance package build time | < 2 minutes for full package with all evidence | p99 latency from build-package request to package ready |
| Document version integrity | 100% of versions with valid SHA-256 provenance | Automated hash verification on every version |
| EU Information System acceptance rate | >= 98% on first submission attempt | Submission result tracking (accepted vs. rejected) |
| 5-year retention compliance | 100% of documents retrievable within retention period | Automated retention validation and retrieval tests |
| Batch DDS generation throughput | >= 100 DDS documents per minute | Batch generation load test |
| Audit package inspection readiness | < 5 minutes from request to delivery | Time from competent authority request to package download |
| Regulatory compliance coverage | 100% of Article 12 DDS requirements | Regulatory mapping validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ EUDR-affected operators and traders across the EU must submit DDS documents for every relevant product placed on the EU market, representing a regulatory documentation market of 2-4 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers processing the 7 regulated commodities requiring automated DDS generation and submission management, estimated at 500M-1B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 30-50M EUR in documentation generation module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities needing automated DDS generation at scale
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with complex, multi-commodity portfolios
- Timber and paper industry operators with hundreds of product lines
- Compliance departments managing quarterly DDS submission deadlines across multiple business units

**Secondary:**
- Customs brokers and freight forwarders handling DDS submission on behalf of importers
- Compliance consultants preparing DDS documentation for multiple clients
- Certification bodies requiring standardized documentation formats for their audit processes
- SME importers (1,000-10,000 shipments/year) ahead of June 30, 2026 enforcement

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual DDS preparation (Word/Excel) | No cost; familiar tools | 4-8 hours per DDS; error-prone; no schema validation; no version control | < 30 seconds per DDS; automated validation; full version control |
| Generic document management (SharePoint, Confluence) | Multi-purpose; collaboration | Not EUDR-specific; no DDS schema; no EU IS integration; no retention enforcement | Purpose-built for EUDR DDS; native EU IS schema; automated 5-year retention |
| Compliance consulting firms | EUDR expertise; regulatory interpretation | EUR 5K-20K per DDS batch; slow turnaround; no automation; no real-time updates | Fully automated; EUR 50-200 per DDS; instant generation; continuous updates |
| Niche EUDR tools (early movers) | First to market | Limited to DDS form filling; no upstream agent integration; no provenance; no audit packages | Full integration with 29 upstream agents; complete provenance chain; audit-ready packages |
| In-house custom builds | Tailored to org | 6-12 months to build; no regulatory updates; schema changes break system | Ready now; automatic regulatory updates; schema-adaptive |

### 2.4 Differentiation Strategy

1. **Full-stack integration** -- Not a standalone DDS form filler. Natively integrated with 29 upstream EUDR agents that provide the actual due diligence data. No manual data entry required.
2. **Regulatory fidelity** -- Every DDS field maps to a specific EUDR Article requirement. Every document section references the underlying regulation with article-level precision.
3. **Complete provenance** -- SHA-256 hash chains from raw data through calculation to final document. Every claim in the DDS is traceable to its evidence source. Zero-hallucination (no LLM in the documentation generation critical path).
4. **Audit-ready by design** -- Compliance packages are structured for competent authority inspection from day one, not retrofitted after the fact.
5. **Submission lifecycle management** -- End-to-end management from draft through submission to acknowledgement, including rejection handling and resubmission.
6. **Scale** -- Batch generation of 100+ DDS documents per minute for large operators with multi-commodity portfolios.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable operators to submit fully compliant DDS to the EU Information System | 100% of generated DDS documents pass EU IS schema validation | Q2 2026 |
| BG-2 | Reduce DDS preparation time from hours to seconds | 99% reduction in per-DDS documentation time (4-8 hours to < 30 seconds) | Q2 2026 |
| BG-3 | Ensure zero EUDR documentation penalties for active customers | Zero penalties attributable to documentation gaps for GreenLang customers | Ongoing |
| BG-4 | Become the reference EUDR documentation solution | 500+ enterprise customers using documentation generation | Q4 2026 |
| BG-5 | Achieve 100% audit inspection readiness | All customers can produce compliance packages within 5 minutes of request | Q3 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Automated DDS generation | Generate complete, schema-valid DDS documents from upstream agent outputs with zero manual intervention |
| PG-2 | Article 9 completeness | Assemble all Article 9 required information from 15+ upstream agents with automated completeness validation |
| PG-3 | Risk documentation | Transform technical risk assessment outputs into structured, auditor-readable documentation per Article 10 |
| PG-4 | Mitigation documentation | Transform mitigation measure outputs into structured documentation per Article 11 with evidence mapping |
| PG-5 | Audit-ready packages | Build complete compliance packages that satisfy Articles 14-16 competent authority inspection requirements |
| PG-6 | Document lifecycle | Manage document versioning, amendment tracking, and 5-year retention per Article 31 |
| PG-7 | Submission management | Handle the complete DDS submission lifecycle including validation, submission, tracking, and resubmission |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | DDS generation performance | < 30 seconds p99 for single DDS generation |
| TG-2 | Batch generation throughput | >= 100 DDS documents per minute |
| TG-3 | Package build performance | < 2 minutes p99 for complete compliance package |
| TG-4 | API response time | < 200ms p95 for standard queries (GET endpoints) |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic document generation; no LLM in critical path |
| TG-7 | Schema compliance | 100% validation pass rate against EU IS DDS schema |
| TG-8 | Data integrity | SHA-256 provenance hash on every document and evidence artifact |

### 3.4 Non-Goals

1. Due diligence workflow orchestration (EUDR-026 Due Diligence Orchestrator handles this)
2. Information gathering from raw data sources (EUDR-027 Information Gathering Agent handles this)
3. Risk assessment calculation (EUDR-028 Risk Assessment Engine handles this)
4. Mitigation measure design (EUDR-029 Mitigation Measure Designer handles this)
5. Supply chain graph visualization (EUDR-001 handles graph topology and rendering)
6. Satellite data acquisition or analysis (EUDR-003/004/005 handle this)
7. Direct integration with external certification databases (EUDR-012 handles document authentication)
8. Carbon footprint or GHG reporting (GL-GHG-APP handles this)
9. Mobile data collection (EUDR-015 handles this)
10. Blockchain-based tamper proofing of documents (EUDR-013 handles blockchain integration; this agent uses SHA-256 hashes)

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, importing cocoa, palm oil, and soya from 15+ countries |
| **EUDR Pressure** | Must submit 200+ DDS documents per quarter across 3 commodity categories; board-level mandate for zero compliance failures |
| **Pain Points** | Currently spends 2 FTEs (full-time equivalents) on manual DDS preparation; each DDS takes 4-8 hours to compile from multiple data sources; frequent submission rejections due to schema errors; no version control on submitted documents; cannot produce audit packages quickly when competent authority requests arrive |
| **Goals** | Automated DDS generation from existing agent outputs; one-click compliance package generation; zero submission rejections; full document version history for audit trail |
| **Technical Skill** | Moderate -- comfortable with web applications and regulatory portals but not a developer |

### Persona 2: Supply Chain Analyst -- Lukas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Analyst at an EU timber importer |
| **Company** | 800 employees, importing tropical and temperate wood from 20+ countries through complex multi-tier chains |
| **EUDR Pressure** | Must ensure DDS documentation accurately reflects the complete supply chain with all Article 9 geolocation data and Article 10 risk assessment results; responsible for data accuracy in submitted DDS |
| **Pain Points** | Article 9 data scattered across 15+ upstream agents; risk of incomplete or inconsistent DDS content; no way to verify that DDS reflects latest upstream agent outputs; manual cross-referencing between supply chain data, risk scores, and DDS content |
| **Goals** | Automated assembly of Article 9 data from all upstream agents with completeness validation; ability to review and verify DDS content before submission; clear traceability from DDS fields to source data |
| **Technical Skill** | High -- comfortable with data tools, APIs, and basic scripting |

### Persona 3: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm conducting Article 16 substantive checks |
| **EUDR Pressure** | Must verify that operator DDS documents are accurate, complete, and supported by evidence; must assess due diligence system adequacy |
| **Pain Points** | Operators provide inconsistent, poorly organized documentation; evidence is scattered across emails, spreadsheets, and databases; no standardized audit package format; cannot verify provenance of claims made in DDS |
| **Goals** | Access to structured, indexed compliance packages with cross-references; ability to trace every DDS claim back to source evidence; standardized documentation format across all audited operators |
| **Technical Skill** | Moderate -- comfortable with audit software and document review |

### Persona 4: Regulatory Inspector -- Inspector Jansen (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | EUDR Enforcement Inspector at a national competent authority |
| **Company** | Government customs and market surveillance agency |
| **EUDR Pressure** | Must conduct risk-based checks (Article 15) and substantive checks (Article 16) on operators; must verify DDS accuracy and underlying due diligence system adequacy within constrained timeframes |
| **Pain Points** | Operators take days or weeks to assemble documentation when requested; documentation quality varies widely; no standardized format for evidence review; difficult to verify provenance and completeness of claims |
| **Goals** | Operators can produce complete compliance packages within minutes of request; standardized, indexed format that enables efficient inspection; clear provenance chain from DDS claims to source evidence |
| **Technical Skill** | Low-moderate -- uses government enforcement portals and document review tools |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(1-2)** | Operators shall exercise due diligence and shall not place/export products unless due diligence has been exercised | DDS generation is the formal output demonstrating due diligence was exercised |
| **Art. 9(1)(a)** | Description of the product, trade name, type, common and full scientific name of species | DDSStatementGenerator populates product description fields from supply chain data |
| **Art. 9(1)(b)** | Quantity of the product | DDSStatementGenerator populates quantity and unit from mass balance data (EUDR-011) |
| **Art. 9(1)(c)** | Country of production, and where applicable, parts thereof | Article9DataAssembler gathers country of production from supply chain mapping (EUDR-001) |
| **Art. 9(1)(d)** | Geolocation of all plots of land where commodities were produced | Article9DataAssembler gathers geolocation data from EUDR-002 and EUDR-007; enforces polygon for > 4 ha |
| **Art. 9(1)(e)** | Date or time range of production | Article9DataAssembler gathers production date ranges from upstream data |
| **Art. 9(1)(f)** | Name, postal address, email of supplier | Article9DataAssembler gathers supplier identification from EUDR-008 multi-tier tracker |
| **Art. 9(1)(g)** | Name, postal address, email of buyer | Article9DataAssembler gathers buyer identification from operator records |
| **Art. 9(2)** | Adequately conclusive and verifiable information | Article9DataAssembler validates evidence quality and references provenance hashes |
| **Art. 10(1)** | Risk assessment to determine whether there is a risk of non-compliance | RiskAssessmentDocumenter documents risk assessment results from EUDR-028 |
| **Art. 10(2)(a)** | Risk of deforestation in the country of production | RiskAssessmentDocumenter documents country-level deforestation risk assessment |
| **Art. 10(2)(b)** | Presence of forests in the country of production | RiskAssessmentDocumenter documents forest presence analysis |
| **Art. 10(2)(c)** | Presence of indigenous peoples | RiskAssessmentDocumenter documents indigenous rights assessment from EUDR-021 |
| **Art. 10(2)(d)** | Consultation of indigenous peoples | RiskAssessmentDocumenter documents consultation evidence |
| **Art. 10(2)(e)** | Concerns about the country of production | RiskAssessmentDocumenter documents country risk concerns from EUDR-016 |
| **Art. 10(2)(f)** | Risk of circumvention or mixing | RiskAssessmentDocumenter documents chain of custody verification from EUDR-009/010/011 |
| **Art. 10(2)(g)** | History of compliance | RiskAssessmentDocumenter documents compliance history from EUDR-017 |
| **Art. 11(1)** | Adopt risk mitigation measures adequate to reach negligible risk | MitigationDocumenter documents mitigation measures from EUDR-029 |
| **Art. 11(2)(a)** | Request additional information, data, or documents | MitigationDocumenter documents additional information gathering measures |
| **Art. 11(2)(b)** | Carry out independent surveys or audits | MitigationDocumenter documents independent audit measures from EUDR-024 |
| **Art. 11(2)(c)** | Any other measure adequate to reach negligible risk | MitigationDocumenter documents supplementary mitigation measures |
| **Art. 12(1)** | Submit DDS to the Information System before placing/making available on market | RegulatorySubmissionEngine manages DDS submission to EU IS |
| **Art. 12(2)** | DDS shall contain information listed in Annex II | DDSStatementGenerator populates all Annex II fields |
| **Art. 13** | Simplified due diligence for low-risk country products | DDSStatementGenerator supports simplified DDS variant for Article 29 low-risk countries |
| **Art. 14(1)** | Competent authorities may require operators to make DDS available | CompliancePackageBuilder generates inspection-ready packages on demand |
| **Art. 15** | Risk-based checks by competent authorities | CompliancePackageBuilder supports risk-based check documentation requirements |
| **Art. 16** | Substantive checks verifying due diligence system adequacy | CompliancePackageBuilder generates complete evidence packages for substantive checks |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | RiskAssessmentDocumenter documents country benchmarking classification and its impact on due diligence level |
| **Art. 31(1)** | Retain records of DDS for at least 5 years | DocumentVersionManager enforces 5-year retention with automated tracking |

### 5.2 DDS Content Requirements (EUDR Annex II)

The Due Diligence Statement must contain the following information, all of which this agent generates:

| # | DDS Field (Annex II) | Source Agent(s) | Documentation Engine |
|---|---------------------|-----------------|---------------------|
| 1 | Operator/trader identification (name, address, EORI) | Operator registration, EUDR-001 | DDSStatementGenerator |
| 2 | Product description (trade name, type, common name, scientific name) | EUDR-001, EUDR-009 | DDSStatementGenerator, Article9DataAssembler |
| 3 | HS/CN code(s) | EUDR-001, EUDR-009 | DDSStatementGenerator |
| 4 | Quantity (net mass in kg, volume in m3, or items) | EUDR-011, EUDR-010 | DDSStatementGenerator, Article9DataAssembler |
| 5 | Country of production | EUDR-001, EUDR-008 | Article9DataAssembler |
| 6 | Geolocation of production plots | EUDR-002, EUDR-006, EUDR-007 | Article9DataAssembler |
| 7 | Date or time range of production | EUDR-001, EUDR-008 | Article9DataAssembler |
| 8 | Supplier name, address, email | EUDR-008 | Article9DataAssembler |
| 9 | Buyer name, address, email | Operator records | Article9DataAssembler |
| 10 | Adequately conclusive and verifiable information (deforestation-free) | EUDR-003, EUDR-004, EUDR-005 | Article9DataAssembler |
| 11 | Adequately conclusive and verifiable information (legal production) | EUDR-023 | Article9DataAssembler |
| 12 | Risk assessment conclusion | EUDR-028 | RiskAssessmentDocumenter |
| 13 | Risk mitigation measures (if applicable) | EUDR-029 | MitigationDocumenter |
| 14 | Declaration of compliance | Compliance conclusion logic | DDSStatementGenerator |

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date referenced in all DDS deforestation-free declarations |
| June 29, 2023 | Regulation entered into force | Legal basis for all DDS requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Operators must have DDS submission capability operational NOW |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; batch DDS generation critical |
| Ongoing (quarterly) | Country benchmarking updates by EC | DDS must reflect current country classifications (Article 29) |
| 5 years post-submission | Record retention deadline | Document retention management per Article 31 |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-4 form the core document generation engine; Features 5-7 form the lifecycle and submission management layer; Features 8-9 form the integration and validation layer.

**P0 Features 1-4: Core Document Generation Engine**

---

#### Feature 1: DDS Statement Generation

**User Story:**
```
As a compliance officer,
I want to generate a complete Due Diligence Statement that contains all information required by EUDR Article 12 and Annex II,
So that I can submit a valid, schema-compliant DDS to the EU Information System before placing products on the EU market.
```

**Acceptance Criteria:**
- [ ] Generates DDS documents containing all 14 Annex II required fields (operator ID, product description, HS/CN codes, quantity, country of production, geolocation references, production dates, supplier info, buyer info, deforestation-free evidence, legal production evidence, risk assessment conclusion, mitigation measures, compliance declaration)
- [ ] Produces DDS in EU Information System expected JSON schema with all mandatory fields populated
- [ ] Produces DDS in EU Information System expected XML schema as an alternative format
- [ ] Validates all mandatory fields before generation; rejects generation with detailed error list if any mandatory field is missing
- [ ] Assigns unique DDS reference numbers using format: GL-DDS-{operator_id}-{commodity}-{YYYYMMDD}-{sequence}
- [ ] Calculates and embeds SHA-256 provenance hash covering all DDS content fields
- [ ] Supports standard DDS for standard/high-risk country products and simplified DDS for low-risk country products (Article 13)
- [ ] Generates compliance conclusion based on: (a) information gathering completeness, (b) risk assessment result, (c) mitigation adequacy -- deterministic logic, no LLM
- [ ] Supports multi-product DDS (one DDS covering multiple product lines from the same supply chain)
- [ ] Generates human-readable DDS summary (PDF format) alongside the machine-readable JSON/XML

**Non-Functional Requirements:**
- Performance: DDS generation < 30 seconds p99 for single DDS
- Determinism: Same inputs produce bit-identical DDS output
- Auditability: Complete generation audit trail with timestamps and input references
- Schema compliance: 100% validation pass rate against EU IS schema

**Dependencies:**
- Article9DataAssembler (Feature 2) for information gathering data
- RiskAssessmentDocumenter (Feature 3) for risk assessment data
- MitigationDocumenter (Feature 4) for mitigation data
- EUDR-001 supply chain data for product/supplier/buyer information
- EUDR-028 risk assessment output for risk conclusion

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Missing optional fields (e.g., no mitigation required for low-risk products) -- Generate DDS without mitigation section; include Article 13 reference
- Multiple countries of production for single product -- Include all countries; generate separate geolocation sections per country
- Operator acts as both importer and trader -- Generate appropriate DDS variant per operator role

---

#### Feature 2: Article 9 Information Assembly

**User Story:**
```
As a supply chain analyst,
I want all Article 9 required information automatically assembled from the upstream EUDR agents,
So that I can verify the data is complete and accurate before it flows into the DDS.
```

**Acceptance Criteria:**
- [ ] Retrieves product description and HS/CN codes from supply chain mapping (EUDR-001) and chain of custody (EUDR-009)
- [ ] Retrieves quantity and unit from mass balance calculator (EUDR-011) and segregation verifier (EUDR-010)
- [ ] Retrieves country of production from supply chain mapping (EUDR-001) and multi-tier supplier tracker (EUDR-008)
- [ ] Retrieves geolocation of all production plots (GPS coordinates and polygons) from geolocation verification (EUDR-002), plot boundary manager (EUDR-006), and GPS coordinator validator (EUDR-007)
- [ ] Retrieves date/time range of production from upstream supply chain data
- [ ] Retrieves supplier name, address, and email from multi-tier supplier tracker (EUDR-008)
- [ ] Retrieves buyer name, address, and email from operator records
- [ ] Retrieves deforestation-free evidence from satellite monitoring (EUDR-003), forest cover analysis (EUDR-004), and land use change detection (EUDR-005)
- [ ] Retrieves legal production evidence from legal compliance verifier (EUDR-023)
- [ ] Validates data completeness against Article 9 requirements checklist (all mandatory fields present)
- [ ] Generates completeness score (0-100) with detailed breakdown by Article 9 sub-requirement
- [ ] Flags missing or inconsistent data with specific remediation guidance (e.g., "Geolocation missing for Plot P-2341 -- request from supplier via EUDR-015 mobile collector")
- [ ] Supports polygon enforcement for plots > 4 hectares per Article 9(1)(d)
- [ ] Timestamps all assembled data with source agent reference and retrieval time

**Non-Functional Requirements:**
- Completeness: 100% of mandatory Article 9 fields populated when upstream data is available
- Performance: Assembly completes in < 60 seconds for supply chains with 1,000+ plots
- Data Freshness: Alerts when assembled data is older than configurable threshold (default 30 days)

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master (supply chain graph, product data)
- EUDR-002 Geolocation Verification Agent (verified plot coordinates)
- EUDR-003 Satellite Monitoring Agent (deforestation-free evidence)
- EUDR-004 Forest Cover Analysis Agent (forest cover verification)
- EUDR-005 Land Use Change Detector Agent (land use change evidence)
- EUDR-006 Plot Boundary Manager Agent (polygon boundaries)
- EUDR-007 GPS Coordinate Validator Agent (validated GPS coordinates)
- EUDR-008 Multi-Tier Supplier Tracker (supplier information)
- EUDR-009 Chain of Custody Agent (product classification, custody chain)
- EUDR-010 Segregation Verifier Agent (quantity verification)
- EUDR-011 Mass Balance Calculator Agent (quantity data)
- EUDR-012 Document Authentication Agent (supporting certificates)
- EUDR-023 Legal Compliance Verifier (legal production evidence)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 integration engineer)

---

#### Feature 3: Risk Assessment Documentation

**User Story:**
```
As a compliance officer,
I want the risk assessment results from EUDR-028 automatically transformed into structured, auditor-readable documentation,
So that I can include accurate risk assessment documentation in the DDS and provide it to competent authorities during inspections.
```

**Acceptance Criteria:**
- [ ] Retrieves composite risk score from EUDR-028 Risk Assessment Engine with full decomposition (country risk, commodity risk, supplier risk, deforestation risk, corruption risk, indigenous rights risk, protected area risk, legal compliance risk)
- [ ] Documents Article 10(2) criterion-by-criterion evaluation results for all 7 criteria (a through g)
- [ ] Documents country benchmarking determination (Article 29): Low, Standard, or High risk classification with underlying data
- [ ] Documents simplified due diligence eligibility assessment (Article 13) with rationale
- [ ] Documents risk classification rationale: why the product was classified as low/standard/high risk
- [ ] Documents the complete provenance chain from raw risk data through calculation to final score
- [ ] Generates risk assessment summary suitable for DDS inclusion (structured JSON section)
- [ ] Generates standalone risk assessment report suitable for auditor review (detailed PDF with methodology, data sources, calculations, and conclusions)
- [ ] Cross-references risk factors with specific EUDR article requirements
- [ ] Documents enhanced due diligence triggers and the evidence that triggered them
- [ ] All risk documentation is deterministic -- same risk assessment input produces identical documentation

**Non-Functional Requirements:**
- Accuracy: 100% faithful representation of EUDR-028 outputs (no transformation errors)
- Determinism: Bit-perfect reproducibility of documentation from same inputs
- Auditability: Complete provenance from DDS risk section back to EUDR-028 raw outputs

**Dependencies:**
- EUDR-028 Risk Assessment Engine (composite risk scores, criterion evaluations)
- EUDR-016 Country Risk Evaluator (country risk data)
- EUDR-017 Supplier Risk Scorer (supplier risk data)
- EUDR-018 Commodity Risk Analyzer (commodity risk data)
- EUDR-019 Corruption Index Monitor (corruption risk data)
- EUDR-020 Deforestation Alert System (deforestation alert data)
- EUDR-021 Indigenous Rights Checker (indigenous rights assessment)
- EUDR-022 Protected Area Validator (protected area assessment)
- EUDR-023 Legal Compliance Verifier (legal compliance assessment)

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 4: Mitigation Measure Documentation

**User Story:**
```
As a compliance officer,
I want the mitigation measures from EUDR-029 automatically transformed into structured documentation,
So that I can demonstrate to competent authorities that adequate measures were taken to reduce risk to a negligible level per Article 11.
```

**Acceptance Criteria:**
- [ ] Retrieves mitigation strategy and individual measures from EUDR-029 Mitigation Measure Designer
- [ ] Documents the risk trigger that necessitated mitigation (which risk criteria exceeded threshold, what score was observed)
- [ ] Documents the mitigation strategy designed (overall approach, target risk reduction, expected timeline)
- [ ] Documents each individual measure implemented with: description, category per Article 11(2) (a/b/c), responsible party, implementation date, evidence references
- [ ] Maps each measure to the specific Article 11(2) category: (a) additional information/data/documents, (b) independent survey/audit, (c) other adequate measures
- [ ] Documents effectiveness verification results for each measure (pre-mitigation risk, post-mitigation risk, risk reduction achieved)
- [ ] Documents final risk determination after mitigation: negligible (compliant) or non-negligible (non-compliant) with supporting rationale
- [ ] Documents implementation timeline showing when each measure was initiated and completed
- [ ] Generates mitigation summary suitable for DDS inclusion (structured JSON section)
- [ ] Generates standalone mitigation report suitable for auditor review (detailed PDF)
- [ ] Handles the case where no mitigation was required (low-risk assessment) -- documents the rationale for no-mitigation conclusion

**Non-Functional Requirements:**
- Completeness: 100% of EUDR-029 outputs documented with evidence references
- Traceability: Every mitigation claim traceable to EUDR-029 output and underlying evidence
- Determinism: Same EUDR-029 inputs produce identical mitigation documentation

**Dependencies:**
- EUDR-029 Mitigation Measure Designer (mitigation strategies, measures, effectiveness)
- EUDR-024 Third-Party Audit Manager (audit evidence for Article 11(2)(b) measures)
- EUDR-025 Risk Mitigation Advisor (risk mitigation recommendations)

**Estimated Effort:** 2 weeks (1 backend engineer)

---

**P0 Features 5-7: Lifecycle and Submission Management**

> Features 5, 6, and 7 are P0 launch blockers. Without document versioning, compliance package building, and submission management, the core document generation engine cannot deliver the complete regulatory workflow. These features ensure documents are managed through their full lifecycle from creation through submission to 5-year retention.

---

#### Feature 5: Compliance Package Building

**User Story:**
```
As a compliance officer,
I want to build a complete, indexed, cross-referenced compliance package from all due diligence documentation,
So that I can provide competent authorities with a comprehensive evidence package during Article 14-16 inspections within minutes of their request.
```

**Acceptance Criteria:**
- [ ] Assembles DDS document (from Feature 1) as the primary document in the package
- [ ] Includes Article 9 information package (from Feature 2) with all supporting data
- [ ] Includes risk assessment documentation (from Feature 3) with methodology and results
- [ ] Includes mitigation documentation (from Feature 4) with measures and effectiveness evidence
- [ ] Includes supply chain maps from EUDR-001 (graph exports, Sankey diagrams)
- [ ] Includes certificate copies from EUDR-012 Document Authentication (FSC, RSPO, etc.)
- [ ] Includes satellite verification evidence from EUDR-003/004/005 (satellite imagery analysis results, NDVI calculations, land use change detection)
- [ ] Includes geolocation validation evidence from EUDR-002/007 (coordinate verification, polygon validation)
- [ ] Includes blockchain provenance records from EUDR-013 (if applicable)
- [ ] Generates table of contents with section numbering and page references
- [ ] Generates cross-reference index linking DDS fields to supporting evidence sections
- [ ] Generates evidence summary listing all artifacts with SHA-256 hashes
- [ ] Supports PDF-ready structured output (generates PDF compilation of all documents)
- [ ] Supports ZIP archive output (all documents in organized folder structure)
- [ ] Package generation completes in < 2 minutes for full compliance package
- [ ] Package includes metadata header with: operator ID, product ID, DDS reference, generation timestamp, version number

**Non-Functional Requirements:**
- Completeness: Package contains all evidence required by Articles 14-16
- Performance: Package build < 2 minutes p99 for complete package
- Integrity: SHA-256 hash on every evidence artifact; package-level integrity hash
- Size: Handles packages up to 500 MB (large supply chains with many satellite images)

**Dependencies:**
- Features 1-4 (DDS, Article 9, risk docs, mitigation docs)
- EUDR-001 Supply Chain Mapping Master (graph exports)
- EUDR-002 Geolocation Verification Agent (verification reports)
- EUDR-003 Satellite Monitoring Agent (satellite evidence)
- EUDR-004 Forest Cover Analysis Agent (analysis reports)
- EUDR-005 Land Use Change Detector Agent (change detection reports)
- EUDR-007 GPS Coordinate Validator Agent (validation reports)
- EUDR-012 Document Authentication Agent (certificate copies)
- EUDR-013 Blockchain Integration Agent (provenance records)
- S3 Object Storage for package assembly and storage

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 6: Document Version Management

**User Story:**
```
As a compliance officer,
I want complete version control on all due diligence documents with amendment tracking and 5-year retention,
So that I can demonstrate an immutable audit trail to competent authorities and ensure Article 31 retention compliance.
```

**Acceptance Criteria:**
- [ ] Assigns version numbers to all documents using semantic versioning (major.minor.patch)
- [ ] Tracks document state machine: DRAFT -> FINAL -> SUBMITTED -> ACKNOWLEDGED -> AMENDED
- [ ] Records every state transition with: actor, timestamp, reason, change description, SHA-256 hash
- [ ] Supports document amendment after submission with full amendment provenance (original + amendment chain)
- [ ] Maintains immutable audit trail -- no state transitions can be deleted or modified
- [ ] Enforces 5-year retention from DDS submission date per Article 31
- [ ] Generates retention alerts: 60 days before expiry, 30 days before expiry, 7 days before expiry
- [ ] Prevents deletion of documents within retention period (hard lock)
- [ ] Supports document recall: mark a submitted DDS as recalled with reason code and replacement DDS reference
- [ ] Tracks document lineage: which DDS replaced which, amendment chains, and related packages
- [ ] Generates version comparison reports (diff between document versions)
- [ ] Archives expired documents (past 5-year retention) with configurable archive policy

**Non-Functional Requirements:**
- Integrity: Every version change produces a new SHA-256 hash; hash chain is verifiable
- Immutability: Audit trail entries are append-only; no modification or deletion
- Retention: 100% of documents within retention period must be retrievable in < 5 seconds
- Scale: Support 100,000+ document versions across all operators

**Dependencies:**
- PostgreSQL for document metadata and version tracking
- S3 Object Storage for document content storage
- TimescaleDB hypertable for audit log (time-series optimized)

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 7: Regulatory Submission Engine

**User Story:**
```
As a compliance officer,
I want the DDS submission to the EU Information System fully managed -- validated before submission, tracked through the lifecycle, and automatically handling rejections,
So that I can ensure timely, successful DDS submissions without manual intervention.
```

**Acceptance Criteria:**
- [ ] Pre-flight validation: validates DDS against EU IS JSON schema before submission attempt
- [ ] Returns detailed validation errors with field-level error descriptions and remediation guidance
- [ ] Tracks submission lifecycle: DRAFT -> VALIDATED -> SUBMITTED -> ACKNOWLEDGED -> REJECTED
- [ ] Records submission receipts: timestamp, submission ID, EU IS acknowledgement number
- [ ] Handles rejection workflow: captures rejection reason from EU IS, generates correction guidance, supports resubmission
- [ ] Supports batch submission: submit multiple DDS documents in a single batch operation
- [ ] Manages batch submission status: tracks individual DDS status within batch (partial success handling)
- [ ] Supports parallel submission: configurable concurrency for batch submissions (default 5 concurrent)
- [ ] Implements retry logic: automatic retry with exponential backoff on transient EU IS errors (timeout, 503)
- [ ] Generates submission summary report: total submitted, accepted, rejected, pending
- [ ] Records all submission attempts for audit trail (including failed attempts)
- [ ] Supports submission scheduling: schedule DDS submission for a future date/time (e.g., before quarterly deadline)

**Non-Functional Requirements:**
- Reliability: Zero lost submissions (every submission attempt is durably recorded)
- Performance: Batch submission of 100 DDS documents completes in < 10 minutes
- Resilience: Handles EU IS outages with queuing and automatic retry
- Auditability: Complete submission audit trail with receipts

**Dependencies:**
- Feature 1 (DDSStatementGenerator) for DDS documents to submit
- EU Information System API specification (JSON schema, endpoints)
- Network connectivity to EU IS endpoints
- AGENT-DATA-005 EUSystemConnector for EU IS API integration

**Estimated Effort:** 3 weeks (1 backend engineer, 1 integration engineer)

---

**P0 Features 8-9: Integration and Validation Layer**

---

#### Feature 8: DDS Completeness Validation

**User Story:**
```
As a compliance officer,
I want to validate any DDS document against the complete set of EUDR requirements before submission,
So that I can identify and fix all issues before they cause submission rejection or regulatory non-compliance.
```

**Acceptance Criteria:**
- [ ] Validates DDS against EU IS JSON schema (structural validation)
- [ ] Validates all Annex II mandatory fields are present and non-empty
- [ ] Validates field-level data quality: HS/CN codes match valid code lists, country codes are ISO 3166, coordinates are valid WGS84, dates are in expected range
- [ ] Validates geolocation completeness: all production plots have GPS coordinates; plots > 4 ha have polygon data
- [ ] Validates risk assessment completeness: all Article 10(2) criteria have evaluation results
- [ ] Validates mitigation adequacy: if risk was non-negligible, mitigation measures are documented; final risk determination is present
- [ ] Validates compliance conclusion consistency: conclusion matches risk assessment and mitigation results (no "compliant" conclusion with high residual risk)
- [ ] Generates validation report with: pass/fail per requirement, severity (error/warning/info), remediation guidance
- [ ] Calculates overall DDS readiness score (0-100) with breakdown by section
- [ ] Supports configurable validation rules (operator can add custom validation checks)
- [ ] Tracks validation history for each DDS (when validated, what passed, what failed)

**Non-Functional Requirements:**
- Performance: Validation completes in < 5 seconds for single DDS
- Accuracy: Zero false negatives (never passes an invalid DDS as valid)
- Coverage: 100% of EU IS validation rules replicated locally for pre-flight check

**Dependencies:**
- EU IS JSON schema specification (for structural validation rules)
- CN/HS code reference database (for code list validation)
- ISO 3166 country code reference (for country validation)

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 9: Batch Documentation Generation

**User Story:**
```
As a compliance officer at a large importer,
I want to generate DDS documents and compliance packages for my entire quarterly portfolio in a single batch operation,
So that I can meet quarterly submission deadlines for 200+ products without manual per-product processing.
```

**Acceptance Criteria:**
- [ ] Accepts batch generation request with list of product IDs, operator IDs, or supply chain IDs
- [ ] Generates DDS documents for all products in the batch using parallel processing
- [ ] Validates each generated DDS in the batch (batch validation)
- [ ] Generates compliance packages for each product in the batch (optional, configurable)
- [ ] Tracks batch job progress: queued, in_progress, completed, failed, partially_completed
- [ ] Reports per-product status within batch: success, failed (with error detail), skipped (missing data)
- [ ] Supports batch size up to 1,000 products per batch job
- [ ] Configurable concurrency: number of parallel DDS generations (default 10, max 50)
- [ ] Supports batch cancellation: cancel remaining items in a running batch
- [ ] Generates batch summary report: total processed, succeeded, failed, skipped, total time
- [ ] Supports scheduled batch execution: schedule batch for future date/time
- [ ] Handles partial failures gracefully: completed DDS are preserved even if some fail

**Non-Functional Requirements:**
- Throughput: >= 100 DDS documents generated per minute
- Reliability: Completed DDS never lost on partial batch failure
- Progress: Real-time progress tracking with estimated time to completion

**Dependencies:**
- Features 1-8 (all documentation generation, validation, and versioning features)
- Redis for batch job queue management
- Background task processing (Celery or equivalent)

**Estimated Effort:** 2 weeks (1 backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Multi-Language DDS Output
- Generate DDS in all EU official languages (24 languages)
- Start with EN, FR, DE, ES, PT, IT, NL
- Translation of standardized regulatory text sections (not free-form content)
- Language selection per DDS or per operator preference

#### Feature 11: DDS Analytics Dashboard
- Statistics on DDS generation volume over time
- Submission success/rejection rate trends
- Average generation time trends
- Most common validation failures
- Compliance readiness scores across portfolio

#### Feature 12: Automated DDS Monitoring and Re-generation
- Monitor upstream agent data changes that may affect submitted DDS accuracy
- Alert when DDS content may be outdated (e.g., new deforestation alert on a production plot)
- Suggest DDS amendment when material changes detected
- Auto-generate amendment DDS with change summary

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Direct EU Information System portal integration (API-based submission; portal UI automation is out of scope)
- DDS content generation using LLMs or AI text generation (all content is deterministic, template-based)
- Carbon footprint or GHG reporting within DDS (defer to GL-GHG-APP)
- Financial impact analysis of DDS rejections (defer to analytics platform)
- Mobile-native DDS review app (web responsive design only)
- Third-party DDS review/approval workflow (external to GreenLang; operators manage their own approval)
- Competitor DDS benchmarking

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                                    +---------------------------+
                                    |     GL-EUDR-APP v1.0      |
                                    |   Frontend (React/TS)     |
                                    +-------------+-------------+
                                                  |
                                    +-------------v-------------+
                                    |     Unified API Layer      |
                                    |       (FastAPI)            |
                                    +-------------+-------------+
                                                  |
            +-------------------------------------+-------------------------------------+
            |                                     |                                     |
+-----------v-----------+           +-------------v-------------+           +-----------v-----------+
| AGENT-EUDR-030        |           | AGENT-EUDR-028            |           | AGENT-EUDR-029        |
| Documentation         |<--------->| Risk Assessment           |<--------->| Mitigation Measure    |
| Generator             |           | Engine                    |           | Designer              |
|                       |           |                           |           |                       |
| - DDSStatementGen     |           | - CompositeRiskScorer     |           | - StrategyDesigner    |
| - Article9Assembler   |           | - CriterionEvaluator      |           | - MeasureImplementor  |
| - RiskAssessDocumenter|           | - CountryBenchmarker      |           | - EffectivenessVerify |
| - MitigationDocumenter|           | - SimplifiedDDEvaluator   |           | - ResidualRiskCalc    |
| - PackageBuilder      |           +---------------------------+           +-----------------------+
| - VersionManager      |
| - SubmissionEngine    |
+-----------+-----------+
            |
            +------- Reads from ALL upstream EUDR agents (001-029) -------+
            |                                                              |
+-----------v-----------+   +-----------v-----------+   +-----------v-----------+
| EUDR-001 thru 007     |   | EUDR-008 thru 015     |   | EUDR-016 thru 027     |
| Supply Chain +        |   | Supply Chain           |   | Risk Assessment +     |
| Geolocation Agents    |   | Traceability Agents    |   | Due Diligence Agents  |
+-----------------------+   +-----------------------+   +-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/documentation_generator/
    __init__.py                          # Package exports (100+ symbols)
    config.py                            # DocumentationGeneratorConfig with GL_EUDR_DGN_ env prefix
    models.py                            # Pydantic v2 models for DDS, packages, versions
    provenance.py                        # SHA-256 hash chains for document integrity
    metrics.py                           # Prometheus metrics (18 metrics, gl_eudr_dgn_ prefix)
    dds_statement_generator.py           # Engine 1: DDS generation per Article 12
    article9_data_assembler.py           # Engine 2: Article 9 data assembly
    risk_assessment_documenter.py        # Engine 3: Risk assessment documentation
    mitigation_documenter.py             # Engine 4: Mitigation documentation
    compliance_package_builder.py        # Engine 5: Audit-ready package building
    document_version_manager.py          # Engine 6: Version control and retention
    regulatory_submission_engine.py      # Engine 7: EU IS submission lifecycle
    setup.py                             # DocumentationGeneratorService facade
    reference_data/
        __init__.py
        eu_dds_schema.py                 # EU IS DDS JSON/XML schema definitions
        annex_ii_fields.py               # EUDR Annex II field definitions and validation rules
        hs_cn_codes.py                   # HS/CN code reference data
        country_codes.py                 # ISO 3166 country codes with Article 29 classifications
        dds_templates.py                 # DDS templates (standard, simplified)
    api/
        __init__.py
        router.py                        # FastAPI router (12 endpoints)
        schemas.py                       # API request/response Pydantic models
        dependencies.py                  # FastAPI dependencies (auth, db, services)
        dds_routes.py                    # DDS generation and retrieval endpoints
        assembly_routes.py               # Article 9 assembly endpoints
        documentation_routes.py          # Risk and mitigation documentation endpoints
        package_routes.py                # Compliance package endpoints
        submission_routes.py             # Submission lifecycle endpoints
        validation_routes.py             # Validation endpoints
        version_routes.py                # Version management endpoints
```

### 7.3 Data Models (Key Entities)

```python
# DDS Document Status
class DDSStatus(str, Enum):
    DRAFT = "draft"                      # Initial creation
    FINAL = "final"                      # Finalized, ready for validation
    VALIDATED = "validated"              # Passed schema validation
    SUBMITTED = "submitted"              # Submitted to EU IS
    ACKNOWLEDGED = "acknowledged"        # Acknowledged by EU IS
    REJECTED = "rejected"                # Rejected by EU IS
    AMENDED = "amended"                  # Amended after submission
    RECALLED = "recalled"                # Recalled by operator

# Document Version State
class DocumentState(str, Enum):
    DRAFT = "draft"
    FINAL = "final"
    SUBMITTED = "submitted"
    AMENDED = "amended"
    ARCHIVED = "archived"

# DDS Document
class DDSDocument(BaseModel):
    dds_id: str                          # Unique DDS reference: GL-DDS-{operator}-{commodity}-{date}-{seq}
    operator_id: str                     # Operator UUID
    operator_name: str                   # Operator legal name
    operator_address: str                # Operator address
    operator_eori: Optional[str]         # EORI number
    product_description: str             # Product trade name and type
    product_scientific_name: Optional[str]  # Scientific name of species
    hs_code: str                         # HS code
    cn_code: str                         # CN code
    quantity: Decimal                    # Net mass/volume/items
    quantity_unit: str                   # kg, m3, items
    country_of_production: List[str]     # ISO 3166-1 alpha-2 codes
    geolocation_references: List[GeolocationReference]  # Plot GPS/polygon refs
    production_date_start: date          # Production period start
    production_date_end: date            # Production period end
    supplier_info: SupplierInfo          # Supplier identification
    buyer_info: BuyerInfo               # Buyer identification
    deforestation_free_evidence: List[EvidenceReference]
    legal_production_evidence: List[EvidenceReference]
    risk_assessment_summary: RiskAssessmentSummary
    mitigation_summary: Optional[MitigationSummary]
    compliance_conclusion: ComplianceConclusion
    dds_type: str                        # "standard" or "simplified"
    commodity: str                       # EUDR commodity category
    status: DDSStatus
    version: int
    provenance_hash: str                 # SHA-256 of all content fields
    created_at: datetime
    updated_at: datetime
    submitted_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    eu_is_reference: Optional[str]       # EU IS acknowledgement number

# Article 9 Information Package
class Article9Package(BaseModel):
    package_id: str
    operator_id: str
    dds_id: Optional[str]               # Linked DDS (if generated)
    product_description: ProductDescription
    quantity: QuantityInfo
    country_of_production: List[CountryOfProduction]
    geolocation_data: List[PlotGeolocation]
    production_dates: ProductionDateRange
    supplier_info: SupplierInfo
    buyer_info: BuyerInfo
    deforestation_evidence: DeforestationEvidence
    legal_production_evidence: LegalProductionEvidence
    completeness_score: float            # 0-100
    completeness_details: List[CompletenessCheck]
    missing_fields: List[MissingFieldDetail]
    assembled_at: datetime
    source_agents: Dict[str, str]        # agent_id -> data retrieval timestamp

# Compliance Package
class CompliancePackage(BaseModel):
    package_id: str
    operator_id: str
    dds_id: str
    dds_document: DDSDocument
    article9_package: Article9Package
    risk_assessment_doc: RiskAssessmentDoc
    mitigation_doc: Optional[MitigationDoc]
    supply_chain_maps: List[SupplyChainMapExport]
    certificate_copies: List[CertificateCopy]
    satellite_evidence: List[SatelliteEvidence]
    geolocation_evidence: List[GeolocationEvidence]
    blockchain_records: List[BlockchainRecord]
    table_of_contents: TableOfContents
    cross_reference_index: CrossReferenceIndex
    evidence_summary: EvidenceSummary
    package_hash: str                    # SHA-256 of entire package
    generated_at: datetime
    total_artifacts: int
    total_size_bytes: int

# Document Version
class DocumentVersion(BaseModel):
    version_id: str
    document_id: str                     # DDS ID or package ID
    document_type: str                   # "dds", "article9", "risk_doc", "mitigation_doc", "package"
    version_number: str                  # Semantic version: "1.0.0"
    state: DocumentState
    content_hash: str                    # SHA-256 of document content
    previous_version_id: Optional[str]   # Link to prior version
    change_description: str
    changed_by: str                      # Actor who made the change
    changed_at: datetime
    retention_expiry: date               # 5 years from submission date
    is_retained: bool                    # Within retention period

# Submission Record
class SubmissionRecord(BaseModel):
    submission_id: str
    dds_id: str
    operator_id: str
    submission_type: str                 # "initial", "amendment", "resubmission"
    status: str                          # "submitted", "acknowledged", "rejected"
    submitted_at: datetime
    eu_is_acknowledgement: Optional[str]
    eu_is_response: Optional[Dict[str, Any]]
    rejection_reason: Optional[str]
    rejection_details: Optional[List[ValidationError]]
    retry_count: int
    batch_id: Optional[str]             # If part of batch submission
    receipt_hash: str                    # SHA-256 of submission receipt
```

### 7.4 Database Schema (New Migration: V118)

```sql
CREATE SCHEMA IF NOT EXISTS gl_eudr_dgn;

-- ============================================================
-- Table 1: DDS Documents
-- ============================================================
CREATE TABLE gl_eudr_dgn.dds_documents (
    dds_id VARCHAR(100) PRIMARY KEY,
    operator_id UUID NOT NULL,
    operator_name VARCHAR(500) NOT NULL,
    operator_address TEXT,
    operator_eori VARCHAR(50),
    product_description TEXT NOT NULL,
    product_scientific_name VARCHAR(500),
    hs_code VARCHAR(20) NOT NULL,
    cn_code VARCHAR(20) NOT NULL,
    quantity NUMERIC(18,4) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL DEFAULT 'kg',
    country_of_production JSONB NOT NULL DEFAULT '[]',
    geolocation_references JSONB NOT NULL DEFAULT '[]',
    production_date_start DATE,
    production_date_end DATE,
    supplier_info JSONB NOT NULL DEFAULT '{}',
    buyer_info JSONB NOT NULL DEFAULT '{}',
    deforestation_free_evidence JSONB DEFAULT '[]',
    legal_production_evidence JSONB DEFAULT '[]',
    risk_assessment_summary JSONB DEFAULT '{}',
    mitigation_summary JSONB,
    compliance_conclusion JSONB NOT NULL DEFAULT '{}',
    dds_type VARCHAR(20) NOT NULL DEFAULT 'standard',
    commodity VARCHAR(50) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'draft',
    version INTEGER NOT NULL DEFAULT 1,
    provenance_hash VARCHAR(64) NOT NULL,
    content_json JSONB,
    content_xml TEXT,
    eu_is_reference VARCHAR(100),
    submitted_at TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    CONSTRAINT fk_dgn_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

CREATE INDEX idx_dgn_dds_operator ON gl_eudr_dgn.dds_documents(operator_id);
CREATE INDEX idx_dgn_dds_status ON gl_eudr_dgn.dds_documents(status);
CREATE INDEX idx_dgn_dds_commodity ON gl_eudr_dgn.dds_documents(commodity);
CREATE INDEX idx_dgn_dds_submitted ON gl_eudr_dgn.dds_documents(submitted_at);

-- ============================================================
-- Table 2: Article 9 Packages
-- ============================================================
CREATE TABLE gl_eudr_dgn.article9_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    dds_id VARCHAR(100) REFERENCES gl_eudr_dgn.dds_documents(dds_id),
    product_description JSONB NOT NULL DEFAULT '{}',
    quantity_info JSONB NOT NULL DEFAULT '{}',
    country_of_production JSONB NOT NULL DEFAULT '[]',
    geolocation_data JSONB NOT NULL DEFAULT '[]',
    production_dates JSONB DEFAULT '{}',
    supplier_info JSONB NOT NULL DEFAULT '{}',
    buyer_info JSONB NOT NULL DEFAULT '{}',
    deforestation_evidence JSONB DEFAULT '{}',
    legal_production_evidence JSONB DEFAULT '{}',
    completeness_score NUMERIC(5,2) DEFAULT 0.0,
    completeness_details JSONB DEFAULT '[]',
    missing_fields JSONB DEFAULT '[]',
    source_agents JSONB DEFAULT '{}',
    assembled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dgn_a9_operator ON gl_eudr_dgn.article9_packages(operator_id);
CREATE INDEX idx_dgn_a9_dds ON gl_eudr_dgn.article9_packages(dds_id);

-- ============================================================
-- Table 3: Risk Assessment Documentation
-- ============================================================
CREATE TABLE gl_eudr_dgn.risk_assessment_docs (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(100) REFERENCES gl_eudr_dgn.dds_documents(dds_id),
    assessment_id UUID NOT NULL,
    composite_risk_score NUMERIC(5,2),
    risk_level VARCHAR(20),
    criterion_evaluations JSONB NOT NULL DEFAULT '{}',
    country_benchmarking JSONB DEFAULT '{}',
    simplified_dd_eligible BOOLEAN DEFAULT FALSE,
    risk_classification_rationale TEXT,
    provenance_chain JSONB DEFAULT '[]',
    enhanced_dd_triggers JSONB DEFAULT '[]',
    documentation_json JSONB NOT NULL DEFAULT '{}',
    documentation_pdf_path VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dgn_risk_dds ON gl_eudr_dgn.risk_assessment_docs(dds_id);
CREATE INDEX idx_dgn_risk_assessment ON gl_eudr_dgn.risk_assessment_docs(assessment_id);

-- ============================================================
-- Table 4: Mitigation Documentation
-- ============================================================
CREATE TABLE gl_eudr_dgn.mitigation_docs (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(100) REFERENCES gl_eudr_dgn.dds_documents(dds_id),
    strategy_id UUID NOT NULL,
    risk_trigger_summary JSONB NOT NULL DEFAULT '{}',
    mitigation_strategy JSONB NOT NULL DEFAULT '{}',
    measures_implemented JSONB NOT NULL DEFAULT '[]',
    article_11_2_mapping JSONB NOT NULL DEFAULT '{}',
    effectiveness_results JSONB DEFAULT '{}',
    final_risk_determination VARCHAR(30),
    implementation_timeline JSONB DEFAULT '[]',
    documentation_json JSONB NOT NULL DEFAULT '{}',
    documentation_pdf_path VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dgn_mitigation_dds ON gl_eudr_dgn.mitigation_docs(dds_id);
CREATE INDEX idx_dgn_mitigation_strategy ON gl_eudr_dgn.mitigation_docs(strategy_id);

-- ============================================================
-- Table 5: Compliance Packages
-- ============================================================
CREATE TABLE gl_eudr_dgn.compliance_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    dds_id VARCHAR(100) NOT NULL REFERENCES gl_eudr_dgn.dds_documents(dds_id),
    article9_package_id UUID REFERENCES gl_eudr_dgn.article9_packages(package_id),
    risk_doc_id UUID REFERENCES gl_eudr_dgn.risk_assessment_docs(doc_id),
    mitigation_doc_id UUID REFERENCES gl_eudr_dgn.mitigation_docs(doc_id),
    table_of_contents JSONB NOT NULL DEFAULT '{}',
    cross_reference_index JSONB NOT NULL DEFAULT '{}',
    evidence_summary JSONB NOT NULL DEFAULT '{}',
    total_artifacts INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    package_hash VARCHAR(64) NOT NULL,
    storage_path VARCHAR(500),
    pdf_path VARCHAR(500),
    zip_path VARCHAR(500),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dgn_pkg_operator ON gl_eudr_dgn.compliance_packages(operator_id);
CREATE INDEX idx_dgn_pkg_dds ON gl_eudr_dgn.compliance_packages(dds_id);

-- ============================================================
-- Table 6: Document Versions
-- ============================================================
CREATE TABLE gl_eudr_dgn.document_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(200) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    version_number VARCHAR(20) NOT NULL,
    state VARCHAR(30) NOT NULL DEFAULT 'draft',
    content_hash VARCHAR(64) NOT NULL,
    previous_version_id UUID REFERENCES gl_eudr_dgn.document_versions(version_id),
    change_description TEXT,
    changed_by VARCHAR(100),
    retention_expiry DATE,
    is_retained BOOLEAN DEFAULT TRUE,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dgn_ver_document ON gl_eudr_dgn.document_versions(document_id);
CREATE INDEX idx_dgn_ver_type ON gl_eudr_dgn.document_versions(document_type);
CREATE INDEX idx_dgn_ver_retention ON gl_eudr_dgn.document_versions(retention_expiry);
CREATE INDEX idx_dgn_ver_state ON gl_eudr_dgn.document_versions(state);

-- ============================================================
-- Table 7: Submission Records
-- ============================================================
CREATE TABLE gl_eudr_dgn.submission_records (
    submission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(100) NOT NULL REFERENCES gl_eudr_dgn.dds_documents(dds_id),
    operator_id UUID NOT NULL,
    submission_type VARCHAR(30) NOT NULL DEFAULT 'initial',
    status VARCHAR(30) NOT NULL DEFAULT 'submitted',
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    eu_is_acknowledgement VARCHAR(200),
    eu_is_response JSONB,
    rejection_reason TEXT,
    rejection_details JSONB,
    retry_count INTEGER DEFAULT 0,
    batch_id UUID,
    receipt_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dgn_sub_dds ON gl_eudr_dgn.submission_records(dds_id);
CREATE INDEX idx_dgn_sub_operator ON gl_eudr_dgn.submission_records(operator_id);
CREATE INDEX idx_dgn_sub_status ON gl_eudr_dgn.submission_records(status);
CREATE INDEX idx_dgn_sub_batch ON gl_eudr_dgn.submission_records(batch_id);

-- ============================================================
-- Table 8: Validation Results
-- ============================================================
CREATE TABLE gl_eudr_dgn.validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(100) NOT NULL REFERENCES gl_eudr_dgn.dds_documents(dds_id),
    validation_type VARCHAR(50) NOT NULL,
    is_valid BOOLEAN NOT NULL,
    readiness_score NUMERIC(5,2) DEFAULT 0.0,
    errors JSONB DEFAULT '[]',
    warnings JSONB DEFAULT '[]',
    info JSONB DEFAULT '[]',
    validated_by VARCHAR(100),
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_dgn_val_dds ON gl_eudr_dgn.validation_results(dds_id);
CREATE INDEX idx_dgn_val_valid ON gl_eudr_dgn.validation_results(is_valid);

-- ============================================================
-- Table 9: Audit Log (Hypertable)
-- ============================================================
CREATE TABLE gl_eudr_dgn.audit_log (
    log_id UUID DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(200) NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(100),
    details JSONB DEFAULT '{}',
    previous_state JSONB,
    new_state JSONB,
    provenance_hash VARCHAR(64),
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_dgn.audit_log', 'logged_at');

CREATE INDEX idx_dgn_audit_entity ON gl_eudr_dgn.audit_log(entity_type, entity_id);
CREATE INDEX idx_dgn_audit_action ON gl_eudr_dgn.audit_log(action);
CREATE INDEX idx_dgn_audit_actor ON gl_eudr_dgn.audit_log(actor);

-- ============================================================
-- Retention policy: compress audit log after 90 days, retain for 7 years
-- ============================================================
SELECT add_compression_policy('gl_eudr_dgn.audit_log', INTERVAL '90 days');
SELECT add_retention_policy('gl_eudr_dgn.audit_log', INTERVAL '7 years');
```

### 7.5 API Endpoints (12)

| # | Method | Path | Description | Auth Required |
|---|--------|------|-------------|---------------|
| 1 | POST | `/api/v1/eudr/documentation-generator/generate-dds` | Generate a new DDS from upstream agent data | Yes (eudr-dgn:dds:write) |
| 2 | GET | `/api/v1/eudr/documentation-generator/dds/{dds_id}` | Get DDS document details | Yes (eudr-dgn:dds:read) |
| 3 | GET | `/api/v1/eudr/documentation-generator/dds` | List DDS documents with filters (status, commodity, date range) | Yes (eudr-dgn:dds:read) |
| 4 | POST | `/api/v1/eudr/documentation-generator/assemble-article9/{operator_id}` | Assemble Article 9 information package | Yes (eudr-dgn:article9:assemble) |
| 5 | POST | `/api/v1/eudr/documentation-generator/document-risk/{assessment_id}` | Generate risk assessment documentation | Yes (eudr-dgn:risk-docs:generate) |
| 6 | POST | `/api/v1/eudr/documentation-generator/document-mitigation/{strategy_id}` | Generate mitigation measure documentation | Yes (eudr-dgn:mitigation-docs:generate) |
| 7 | POST | `/api/v1/eudr/documentation-generator/build-package/{dds_id}` | Build complete compliance package | Yes (eudr-dgn:packages:build) |
| 8 | POST | `/api/v1/eudr/documentation-generator/submit/{dds_id}` | Submit DDS to EU Information System | Yes (eudr-dgn:dds:submit) |
| 9 | GET | `/api/v1/eudr/documentation-generator/submissions/{submission_id}/status` | Get submission status | Yes (eudr-dgn:submissions:read) |
| 10 | POST | `/api/v1/eudr/documentation-generator/validate/{dds_id}` | Validate DDS completeness and schema compliance | Yes (eudr-dgn:dds:read) |
| 11 | GET | `/api/v1/eudr/documentation-generator/versions/{document_id}` | Get document version history | Yes (eudr-dgn:versions:read) |
| 12 | GET | `/api/v1/eudr/documentation-generator/health` | Service health check | No |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_dgn_dds_generated_total` | Counter | DDS documents generated (labels: commodity, dds_type) |
| 2 | `gl_eudr_dgn_article9_assemblies_total` | Counter | Article 9 packages assembled |
| 3 | `gl_eudr_dgn_risk_docs_total` | Counter | Risk assessment documents generated |
| 4 | `gl_eudr_dgn_mitigation_docs_total` | Counter | Mitigation documents generated |
| 5 | `gl_eudr_dgn_compliance_packages_total` | Counter | Compliance packages built |
| 6 | `gl_eudr_dgn_submissions_total` | Counter | DDS submissions to EU IS (labels: status) |
| 7 | `gl_eudr_dgn_validations_total` | Counter | Document validations (labels: result) |
| 8 | `gl_eudr_dgn_api_errors_total` | Counter | API errors (labels: endpoint, error_type) |
| 9 | `gl_eudr_dgn_dds_generation_duration_seconds` | Histogram | DDS generation latency |
| 10 | `gl_eudr_dgn_article9_assembly_duration_seconds` | Histogram | Article 9 assembly latency |
| 11 | `gl_eudr_dgn_package_build_duration_seconds` | Histogram | Compliance package build latency |
| 12 | `gl_eudr_dgn_submission_duration_seconds` | Histogram | DDS submission round-trip latency |
| 13 | `gl_eudr_dgn_validation_duration_seconds` | Histogram | DDS validation latency |
| 14 | `gl_eudr_dgn_active_drafts` | Gauge | DDS documents currently in draft state |
| 15 | `gl_eudr_dgn_pending_submissions` | Gauge | DDS documents awaiting submission |
| 16 | `gl_eudr_dgn_rejected_submissions` | Gauge | Rejected submissions needing resubmission |
| 17 | `gl_eudr_dgn_document_versions` | Gauge | Total document versions managed across all documents |
| 18 | `gl_eudr_dgn_retention_documents` | Gauge | Documents currently under 5-year retention |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for audit log |
| Cache | Redis | DDS template caching, validation rule caching, batch job queuing |
| Object Storage | S3 | Compliance packages (PDF, ZIP), document content storage |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| PDF Generation | ReportLab + WeasyPrint | PDF compliance packages, human-readable DDS reports |
| XML Generation | lxml | EU IS XML schema DDS output |
| JSON Schema Validation | jsonschema + fastjsonschema | Pre-flight DDS validation against EU IS schema |
| Hashing | hashlib (SHA-256) | Document integrity, provenance chains |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control with 16 permissions |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| Background Tasks | Celery + Redis | Batch DDS generation, package building |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following 16 permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-dgn:dds:read` | View DDS documents and their content | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-dgn:dds:write` | Generate new DDS documents | Compliance Officer, Admin |
| `eudr-dgn:dds:submit` | Submit DDS to EU Information System | Compliance Officer, Admin |
| `eudr-dgn:article9:read` | View Article 9 information packages | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-dgn:article9:assemble` | Trigger Article 9 data assembly | Analyst, Compliance Officer, Admin |
| `eudr-dgn:risk-docs:read` | View risk assessment documentation | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-dgn:risk-docs:generate` | Generate risk assessment documentation | Analyst, Compliance Officer, Admin |
| `eudr-dgn:mitigation-docs:read` | View mitigation documentation | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-dgn:mitigation-docs:generate` | Generate mitigation documentation | Analyst, Compliance Officer, Admin |
| `eudr-dgn:packages:read` | View compliance packages | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-dgn:packages:build` | Build compliance packages | Compliance Officer, Admin |
| `eudr-dgn:submissions:read` | View submission status and history | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-dgn:submissions:manage` | Manage submission lifecycle (retry, cancel, resubmit) | Compliance Officer, Admin |
| `eudr-dgn:versions:read` | View document version history | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-dgn:versions:manage` | Create document versions, manage amendments, control retention | Compliance Officer, Admin |
| `eudr-dgn:audit:read` | View immutable audit trail | Auditor, Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| EUDR-001 Supply Chain Mapping Master | Graph data, product info, supplier/buyer IDs | Supply chain topology -> DDS supply chain section |
| EUDR-002 Geolocation Verification Agent | Verified plot coordinates | GPS coordinates/polygons -> Article 9 geolocation |
| EUDR-003 Satellite Monitoring Agent | Satellite analysis results | Deforestation-free evidence -> compliance package |
| EUDR-004 Forest Cover Analysis Agent | Forest cover verification | Forest status evidence -> compliance package |
| EUDR-005 Land Use Change Detector Agent | Land use change reports | Change detection evidence -> compliance package |
| EUDR-006 Plot Boundary Manager Agent | Plot polygon boundaries | Polygon data -> Article 9 geolocation |
| EUDR-007 GPS Coordinate Validator Agent | Validated GPS coordinates | Validated coords -> Article 9 geolocation |
| EUDR-008 Multi-Tier Supplier Tracker | Supplier profiles, hierarchy | Supplier info -> Article 9 supplier section |
| EUDR-009 Chain of Custody Agent | Custody chain records | Product classification, custody model -> DDS |
| EUDR-010 Segregation Verifier Agent | Segregation verification | Quantity verification -> Article 9 quantity |
| EUDR-011 Mass Balance Calculator Agent | Mass balance results | Quantity data -> Article 9 quantity |
| EUDR-012 Document Authentication Agent | Authenticated certificates | Certificate copies -> compliance package |
| EUDR-013 Blockchain Integration Agent | Blockchain provenance records | Provenance evidence -> compliance package |
| EUDR-016 Country Risk Evaluator | Country risk classifications | Country risk data -> risk documentation |
| EUDR-017 Supplier Risk Scorer | Supplier risk scores | Supplier risk -> risk documentation |
| EUDR-018 Commodity Risk Analyzer | Commodity risk analysis | Commodity risk -> risk documentation |
| EUDR-019 Corruption Index Monitor | Corruption risk data | Corruption risk -> risk documentation |
| EUDR-020 Deforestation Alert System | Deforestation alerts | Alert data -> risk documentation |
| EUDR-021 Indigenous Rights Checker | Indigenous rights assessment | Rights assessment -> risk documentation |
| EUDR-022 Protected Area Validator | Protected area assessment | PA assessment -> risk documentation |
| EUDR-023 Legal Compliance Verifier | Legal compliance assessment | Legal evidence -> Article 9 + risk documentation |
| EUDR-024 Third-Party Audit Manager | Audit results | Audit evidence -> mitigation documentation |
| EUDR-025 Risk Mitigation Advisor | Mitigation recommendations | Advisory data -> mitigation documentation |
| EUDR-027 Information Gathering Agent | Aggregated information | Gathered data -> Article 9 assembly |
| EUDR-028 Risk Assessment Engine | Composite risk scores, criterion evaluations | Risk results -> risk documentation + DDS |
| EUDR-029 Mitigation Measure Designer | Mitigation strategies, measures, effectiveness | Mitigation data -> mitigation documentation + DDS |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | DDS, packages -> frontend display, download, submission UI |
| EU Information System | Submission API | DDS JSON/XML -> EU IS portal |
| AGENT-DATA-005 EUSystemConnector | EU IS API | Submission management, acknowledgement tracking |
| External Auditors | Read-only API + package downloads | Compliance packages for third-party verification |
| Competent Authorities | Package delivery | Audit-ready evidence packages for inspection |
| AGENT-FOUND-010 Observability Agent | Metrics stream | 18 Prometheus metrics for monitoring dashboard |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Generate and Submit DDS (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Documentation" -> "Generate DDS"
3. Selects operator, commodity, and product from dropdown
4. Clicks "Assemble Article 9 Data"
   -> System calls Article9DataAssembler
   -> System retrieves data from EUDR-001 through EUDR-023
   -> Displays completeness score (e.g., 96%) with any missing fields highlighted
5. Officer reviews assembled data, addresses any flagged gaps
6. Clicks "Generate DDS"
   -> System calls DDSStatementGenerator
   -> System retrieves risk assessment from EUDR-028
   -> System retrieves mitigation from EUDR-029 (if applicable)
   -> DDS generated in < 30 seconds
7. System displays DDS preview with all sections
8. Officer clicks "Validate"
   -> System validates against EU IS schema
   -> All checks pass (green checkmarks)
9. Officer clicks "Submit to EU Information System"
   -> System calls RegulatorySubmissionEngine
   -> Submission status: SUBMITTED
10. EU IS acknowledges receipt
    -> Status updates to ACKNOWLEDGED
    -> Acknowledgement number recorded
11. Officer downloads DDS as PDF and JSON for records
```

#### Flow 2: Build Audit Package (Competent Authority Request)

```
1. Compliance officer receives request from competent authority for DDS evidence
2. Navigates to "Documentation" -> "Compliance Packages"
3. Searches for the DDS by reference number
4. Clicks "Build Compliance Package"
   -> System calls CompliancePackageBuilder
   -> System assembles DDS + Article 9 + risk docs + mitigation docs
   -> System includes supply chain maps, certificates, satellite evidence
   -> Package generated in < 2 minutes
5. System displays package contents with table of contents
6. Officer downloads ZIP archive or PDF compilation
7. Officer provides package to competent authority
8. Audit trail records: package generated, downloaded, reason = "CA inspection request"
```

#### Flow 3: Batch Quarterly DDS Generation (Large Operator)

```
1. Compliance officer navigates to "Documentation" -> "Batch Generation"
2. Selects "Q1 2026 Portfolio" (pre-configured product list)
3. System displays: 250 products, 3 commodities, 12 countries of production
4. Officer clicks "Generate All DDS"
   -> System creates batch job
   -> Parallel processing: 10 DDS at a time
   -> Progress bar shows: 50/250 completed... 100/250... 200/250...
5. Batch completes in ~3 minutes
   -> 247 DDS generated successfully
   -> 3 DDS failed (missing Article 9 data -- details provided)
6. Officer reviews 3 failures, resolves data gaps, regenerates
7. Officer clicks "Validate All"
   -> All 250 DDS pass validation
8. Officer clicks "Submit All"
   -> Batch submission to EU IS
   -> Progress tracking per DDS
9. Batch summary: 250 submitted, 248 acknowledged, 2 pending
```

#### Flow 4: DDS Amendment (Post-Submission Update)

```
1. New deforestation alert detected for Plot P-5678 (EUDR-020)
2. Compliance officer receives alert that DDS GL-DDS-OP1-COCOA-20260215-001 may need amendment
3. Navigates to DDS detail page
4. System shows: "Risk assessment may be outdated -- new deforestation alert on Plot P-5678"
5. Officer clicks "Generate Amendment"
   -> System creates new version of DDS with updated risk assessment
   -> Previous version marked as AMENDED
   -> Amendment provenance chain recorded
6. Officer reviews updated DDS
7. Validates and submits amended DDS
   -> EU IS receives amended DDS with reference to original
8. Document version history shows: v1.0 (original) -> v1.1 (amendment, reason: deforestation alert)
```

### 8.2 Key Screen Descriptions

**DDS Generation Dashboard:**
- Left panel: operator and product selector with commodity filter
- Center panel: DDS preview with expandable sections (Article 9 data, risk assessment, mitigation, compliance conclusion)
- Right panel: validation status, completeness score, action buttons (Generate, Validate, Submit)
- Bottom bar: version history timeline

**Compliance Package Builder:**
- Header: DDS reference number, operator name, commodity
- Content list: checklist of all evidence artifacts to include (DDS, Article 9, risk docs, mitigation docs, supply chain maps, certificates, satellite evidence)
- Package preview: table of contents with section sizes
- Action buttons: Build Package, Download PDF, Download ZIP
- Status indicator: building progress, package size, artifact count

**Batch Generation Manager:**
- Top: batch configuration (product list, commodity filter, generation options)
- Center: progress table with per-product status (queued, generating, validating, completed, failed)
- Bottom: batch summary (total, succeeded, failed, skipped) with estimated time to completion
- Action buttons: Start Batch, Pause, Cancel, Retry Failed

**Document Version History:**
- Timeline view of all document versions with state transitions
- Version detail panel: version number, state, change description, actor, timestamp, hash
- Diff viewer: side-by-side comparison between two versions
- Retention indicator: days remaining in 5-year retention period

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: DDSStatementGenerator -- generates schema-valid DDS in JSON/XML
  - [ ] Feature 2: Article9DataAssembler -- assembles from 15+ upstream agents with completeness validation
  - [ ] Feature 3: RiskAssessmentDocumenter -- transforms EUDR-028 outputs into structured documentation
  - [ ] Feature 4: MitigationDocumenter -- transforms EUDR-029 outputs into Article 11 documentation
  - [ ] Feature 5: CompliancePackageBuilder -- builds audit-ready packages with table of contents and cross-references
  - [ ] Feature 6: DocumentVersionManager -- version control with 5-year retention enforcement
  - [ ] Feature 7: RegulatorySubmissionEngine -- DDS submission lifecycle with rejection handling
  - [ ] Feature 8: DDS Completeness Validation -- pre-flight validation against EU IS schema
  - [ ] Feature 9: Batch Documentation Generation -- batch DDS generation at >= 100/minute throughput
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated with 16 permissions)
- [ ] Performance targets met: DDS generation < 30s p99, package build < 2 min p99, batch >= 100 DDS/min
- [ ] EU IS DDS schema validation: 100% pass rate on generated DDS
- [ ] All 7 commodity types tested with golden test fixtures
- [ ] Document version integrity: 100% SHA-256 hash verification passing
- [ ] 5-year retention enforcement validated
- [ ] API documentation complete (OpenAPI spec for all 12 endpoints)
- [ ] Database migration V118 tested and validated
- [ ] Integration with EUDR-028 and EUDR-029 verified end-to-end
- [ ] 5 beta customers successfully generated and submitted DDS
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ DDS documents generated by customers
- EU IS acceptance rate >= 95% on first submission
- Average DDS generation time < 20 seconds
- < 3 support tickets per customer
- Zero data integrity failures (all provenance hashes valid)

**60 Days:**
- 500+ DDS documents generated
- 50+ compliance packages built for audit/inspection
- EU IS acceptance rate >= 98% on first submission
- Average batch generation throughput >= 100 DDS/minute
- NPS > 40 from compliance officer persona

**90 Days:**
- 2,000+ DDS documents generated
- 200+ compliance packages delivered to auditors/inspectors
- EU IS acceptance rate >= 99% on first submission
- Zero EUDR documentation penalties for active customers
- 100% Article 31 retention compliance
- NPS > 50 from compliance officer persona

---

## 10. Timeline and Milestones

### Phase 1: Core Document Generation (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Article9DataAssembler (Feature 2): integration with EUDR-001 through EUDR-023, completeness validation | Senior Backend Engineer |
| 2-3 | RiskAssessmentDocumenter (Feature 3): EUDR-028 integration, Article 10 documentation | Backend Engineer |
| 3-4 | MitigationDocumenter (Feature 4): EUDR-029 integration, Article 11 documentation | Backend Engineer |
| 4-6 | DDSStatementGenerator (Feature 1): DDS generation in JSON/XML, schema validation, all Annex II fields | Senior Backend Engineer |

**Milestone: Core document generation operational with DDS, Article 9, risk, and mitigation documentation (Week 6)**

### Phase 2: Lifecycle and Submission (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | DocumentVersionManager (Feature 6): version control, state machine, retention enforcement | Backend Engineer |
| 8-9 | CompliancePackageBuilder (Feature 5): package assembly, table of contents, cross-references, PDF/ZIP output | Senior Backend Engineer |
| 9-10 | RegulatorySubmissionEngine (Feature 7): EU IS integration, submission lifecycle, rejection handling | Backend + Integration Engineer |

**Milestone: Full document lifecycle and submission management operational (Week 10)**

### Phase 3: Validation, Batch, and API (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11 | DDS Completeness Validation (Feature 8): schema validation, field-level checks, readiness scoring | Backend Engineer |
| 12 | Batch Documentation Generation (Feature 9): parallel generation, batch management, progress tracking | Backend Engineer |
| 12-13 | REST API Layer: 12 endpoints, authentication, rate limiting, OpenAPI docs | Backend Engineer |
| 13-14 | RBAC integration (16 permissions), end-to-end integration testing with upstream agents | Backend Engineer |

**Milestone: All 9 P0 features implemented with full API and RBAC (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 500+ tests, golden tests for all 7 commodities, EU IS schema compliance tests | Test Engineer |
| 16-17 | Performance testing, security audit, load testing (batch generation at scale) | DevOps + Security |
| 17 | Database migration V118 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers), DDS generation and submission validation | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Multi-language DDS output (Feature 10)
- DDS analytics dashboard (Feature 11)
- Automated DDS monitoring and re-generation (Feature 12)
- Performance optimization for very large supply chains
- Additional DDS template variants for evolving EU IS requirements

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable, production-ready |
| EUDR-002 Geolocation Verification Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-003 Satellite Monitoring Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-004 Forest Cover Analysis Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-005 Land Use Change Detector Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-006 Plot Boundary Manager Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-007 GPS Coordinate Validator Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-008 Multi-Tier Supplier Tracker | BUILT (100%) | Low | Stable, production-ready |
| EUDR-009 Chain of Custody Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-010 Segregation Verifier Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-011 Mass Balance Calculator Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-012 Document Authentication Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-013 Blockchain Integration Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-016 through EUDR-020 Risk Assessment Agents | BUILT (100%) | Low | Stable, production-ready |
| EUDR-027 Information Gathering Agent | In Development | Medium | Fallback: direct agent-to-agent integration |
| EUDR-028 Risk Assessment Engine | In Development | Medium | Fallback: mock risk assessment data for testing |
| EUDR-029 Mitigation Measure Designer | In Development | Medium | Fallback: mock mitigation data for testing |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration points defined |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| S3 Object Storage | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| AGENT-DATA-005 EUSystemConnector | BUILT (100%) | Low | EU IS API integration layer |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System DDS schema specification | Published (v1.x) | Medium | Adapter pattern for schema version changes; schema versioning in reference_data |
| EU Information System submission API | Available | Medium | Retry logic with exponential backoff; queue-based submission; fallback to manual upload |
| EC country benchmarking list (Article 29) | Published; updated periodically | Medium | Database-driven; hot-reloadable country classifications |
| EU EUDR implementing regulations | Evolving | Medium | Configuration-driven compliance rules; modular template system |
| CN/HS code reference databases | Available (EU TARIC) | Low | Cached locally; periodic refresh |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EU Information System schema changes before or after launch | Medium | High | Adapter pattern isolates EU schema layer from core generation engine; schema definitions in configuration (not hardcoded); automated schema diff detection on EU IS updates |
| R2 | EU IS submission API downtime during critical quarterly deadlines | Medium | High | Queue-based submission with automatic retry; local DDS storage for manual upload fallback; submission scheduling ahead of deadlines |
| R3 | Upstream agent data quality insufficient for complete DDS | Medium | High | Article 9 completeness validation with detailed gap reporting; partial DDS generation with explicit missing-data flagging; integration with EUDR-027 Information Gathering for gap remediation |
| R4 | DDS rejection by EU IS due to undocumented validation rules | Medium | Medium | Pre-flight validation replicating all known EU IS rules; rejection analysis engine learning from rejection patterns; feedback loop from rejected submissions to validation rules |
| R5 | Regulatory changes to EUDR Annex II DDS content requirements | Low | High | Template-based DDS generation with configurable field mappings; modular documentation sections; rapid template updates without code changes |
| R6 | Performance degradation for large supply chains with 10,000+ plots | Low | Medium | Lazy data retrieval from upstream agents; pagination for large datasets; async package building with progress tracking |
| R7 | Document integrity compromise during storage or transmission | Low | High | SHA-256 hash on every document; hash verification on every retrieval; S3 versioning with encryption at rest (AES-256) |
| R8 | 5-year retention storage costs for large operators | Medium | Low | S3 lifecycle policies with intelligent tiering; compressed archive for older documents; configurable retention beyond 5-year minimum |
| R9 | Batch generation bottleneck during quarterly submission peaks | Medium | Medium | Horizontal scaling via Kubernetes; configurable concurrency; priority queuing for deadline-critical submissions |
| R10 | Integration complexity with 29 upstream agents | Medium | Medium | Well-defined interfaces per agent; mock adapters for testing; circuit breaker pattern; graceful degradation when individual agents unavailable |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| DDS Generation Unit Tests | 100+ | DDS field population, schema generation (JSON/XML), compliance conclusion logic, reference number generation, provenance hashing |
| Article 9 Assembly Tests | 80+ | Data retrieval from each upstream agent, completeness validation, missing field detection, polygon enforcement, data freshness checks |
| Risk Documentation Tests | 60+ | Risk score documentation, criterion-by-criterion output, country benchmarking, simplified DD eligibility, provenance chain |
| Mitigation Documentation Tests | 50+ | Mitigation strategy documentation, Article 11(2) mapping, effectiveness verification, final risk determination |
| Compliance Package Tests | 70+ | Package assembly, table of contents generation, cross-reference indexing, evidence summary, PDF/ZIP output, hash integrity |
| Document Version Tests | 50+ | Version creation, state machine transitions, amendment tracking, retention enforcement, immutable audit trail |
| Submission Lifecycle Tests | 60+ | Pre-flight validation, submission flow, acknowledgement handling, rejection handling, retry logic, batch submission |
| Validation Tests | 40+ | Schema validation, field-level validation, consistency checks, readiness scoring |
| Batch Generation Tests | 30+ | Batch job creation, parallel processing, partial failure handling, progress tracking, cancellation |
| API Tests | 50+ | All 12 endpoints, auth, error handling, pagination, rate limiting |
| Golden Tests | 49+ | All 7 commodities x 7 scenarios: complete DDS, missing data, high-risk, low-risk (simplified), multi-country, batch, amendment |
| Integration Tests | 30+ | Cross-agent integration with EUDR-028, EUDR-029, EUDR-001 through EUDR-023 |
| Performance Tests | 20+ | DDS generation latency, batch throughput, package build time, concurrent submission |
| **Total** | **500+** | |

### 13.2 Golden Test Scenarios

Each of the 7 commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood) will have dedicated golden test scenarios:

| # | Scenario | Expected Outcome |
|---|----------|-----------------|
| 1 | Complete supply chain, standard risk, no mitigation needed | DDS generated with all fields; validated; ready for submission |
| 2 | Complete supply chain, high risk, mitigation required | DDS generated with risk assessment + mitigation sections populated |
| 3 | Low-risk country origin, simplified due diligence | Simplified DDS generated per Article 13; reduced field requirements |
| 4 | Missing Article 9 data (no geolocation for some plots) | DDS generation blocked; completeness score < 100; gaps flagged with remediation |
| 5 | Multi-country supply chain (3+ countries of production) | DDS generated with multiple country sections; geolocation per country |
| 6 | Batch generation (10 products, same commodity) | All 10 DDS generated in parallel; batch summary report |
| 7 | DDS amendment (post-submission update) | New version created; amendment provenance recorded; original preserved |

Total: 7 commodities x 7 scenarios = 49 golden test scenarios

### 13.3 EU IS Schema Compliance Tests

| Test | Description |
|------|-------------|
| SC-001 | Generated JSON validates against EU IS JSON schema (all mandatory fields present) |
| SC-002 | Generated XML validates against EU IS XML schema (all mandatory elements present) |
| SC-003 | HS/CN codes in DDS match valid EU TARIC code list |
| SC-004 | Country codes in DDS are valid ISO 3166-1 alpha-2 |
| SC-005 | GPS coordinates are valid WGS84 (latitude -90 to 90, longitude -180 to 180) |
| SC-006 | Polygon data present for all plots > 4 hectares |
| SC-007 | Quantities are positive decimals with valid units |
| SC-008 | Dates are within valid range (not before 2020-12-31 cutoff, not in future) |
| SC-009 | Compliance conclusion consistent with risk assessment and mitigation results |
| SC-010 | Provenance hash is valid SHA-256 covering all content fields |

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 12 submitted to the EU Information System |
| **EU IS** | EU Information System -- the central portal where operators submit DDS documents |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |
| **EORI** | Economic Operators Registration and Identification -- EU customs identification number |
| **Article 9** | EUDR Article 9 -- specifies the information operators must gather (product, quantity, geolocation, suppliers) |
| **Article 10** | EUDR Article 10 -- specifies the risk assessment requirements and criteria |
| **Article 11** | EUDR Article 11 -- specifies the risk mitigation requirements |
| **Article 12** | EUDR Article 12 -- specifies the DDS submission obligation |
| **Article 13** | EUDR Article 13 -- provides for simplified due diligence for low-risk country products |
| **Article 14** | EUDR Article 14 -- competent authority powers to request documentation |
| **Article 15** | EUDR Article 15 -- risk-based checks by competent authorities |
| **Article 16** | EUDR Article 16 -- substantive checks by competent authorities |
| **Article 29** | EUDR Article 29 -- country benchmarking (Low/Standard/High risk) |
| **Article 31** | EUDR Article 31 -- 5-year record retention obligation |
| **Annex II** | EUDR Annex II -- specifies the mandatory content of the DDS |
| **Competent Authority** | National authority responsible for EUDR enforcement in each EU Member State |
| **Negligible Risk** | Risk level determined after mitigation that is adequate for compliance (Article 11) |
| **Provenance Hash** | SHA-256 hash providing tamper-evident integrity verification for documents and data |

### Appendix B: EUDR DDS Content Requirements (Annex II Summary)

The DDS must contain:
1. Name and contact details of the operator or trader
2. EORI number (where applicable)
3. Description of the relevant product, including trade name, type, common name, and scientific name
4. HS code of the product
5. Quantity of the product (net mass in kg, volume, or number of items)
6. Country of production (and sub-national unit where applicable)
7. Geolocation of all plots of land where the relevant commodities were produced
8. Date or time range of production
9. Name, postal address, and email address of suppliers
10. Name, postal address, and email address of buyers
11. Adequately conclusive and verifiable information that the products are deforestation-free
12. Adequately conclusive and verifiable information that production was legal
13. Where applicable, information on risk assessment and risk mitigation measures
14. Date and place of the due diligence statement

### Appendix C: DDS Reference Number Format

```
GL-DDS-{operator_id_short}-{commodity_code}-{YYYYMMDD}-{sequence_4digit}

Examples:
  GL-DDS-OP12345-COCOA-20260301-0001    (first cocoa DDS for operator OP12345 on March 1, 2026)
  GL-DDS-OP12345-PALMOIL-20260301-0001  (first palm oil DDS for same operator, same date)
  GL-DDS-OP12345-COCOA-20260301-0002    (second cocoa DDS for same operator, same date)

Commodity codes:
  CATTLE, COCOA, COFFEE, PALMOIL, RUBBER, SOYA, WOOD
```

### Appendix D: Document State Machine

```
                 +-------+
                 | DRAFT |
                 +---+---+
                     |
                     | finalize()
                     v
                 +-------+
                 | FINAL |
                 +---+---+
                     |
                     | validate() -> passes
                     v
              +-----------+
              | VALIDATED |
              +-----+-----+
                    |
                    | submit()
                    v
              +-----------+
              | SUBMITTED |
              +-----+-----+
                    |
           +--------+--------+
           |                 |
           | acknowledge()   | reject()
           v                 v
    +-------------+    +----------+
    | ACKNOWLEDGED|    | REJECTED |
    +------+------+    +-----+----+
           |                 |
           | amend()         | correct_and_resubmit()
           v                 v
      +---------+      +-----------+
      | AMENDED |      | SUBMITTED | (new version)
      +---------+      +-----------+

Special transitions:
  - Any state -> RECALLED (operator recall with reason)
  - DRAFT -> (deleted) -- only drafts can be permanently deleted
```

### Appendix E: Article 11(2) Mitigation Measure Categories

| Category | Article Reference | Description | Examples |
|----------|------------------|-------------|----------|
| (a) | Art. 11(2)(a) | Request additional information, data, or documents from suppliers | Request GPS coordinates from supplier; request certification copies; request origin declarations |
| (b) | Art. 11(2)(b) | Carry out independent surveys or audits | Commission third-party field audit; satellite imagery independent verification; on-site inspection |
| (c) | Art. 11(2)(c) | Any other measures adequate to reach negligible risk | Switch to certified supply chain; implement segregation; increase monitoring frequency; supplier replacement |

### Appendix F: Upstream Agent Output Format References

| Agent | Output Format | Key Fields Used by EUDR-030 |
|-------|--------------|----------------------------|
| EUDR-001 | SupplyChainGraph JSON | nodes (actors), edges (transfers), graph_id, commodity, traceability_score |
| EUDR-002 | GeolocationVerification JSON | plot_id, coordinates, polygon, verification_status, accuracy |
| EUDR-003 | SatelliteAnalysis JSON | plot_id, ndvi_score, deforestation_detected, analysis_date, imagery_source |
| EUDR-004 | ForestCoverReport JSON | plot_id, forest_cover_pct, baseline_2020, current, change_detected |
| EUDR-005 | LandUseChangeReport JSON | plot_id, change_type, change_date, area_affected, confidence |
| EUDR-007 | GPSValidation JSON | plot_id, lat, lon, validation_status, accuracy_meters |
| EUDR-008 | SupplierProfile JSON | supplier_id, legal_name, address, country, tier_depth, risk_score |
| EUDR-011 | MassBalanceResult JSON | product_id, input_qty, output_qty, balance_status, origin_plots |
| EUDR-012 | DocumentAuthentication JSON | cert_id, cert_type, issuer, valid_from, valid_to, hash |
| EUDR-028 | RiskAssessment JSON | composite_score, risk_level, criterion_scores, country_benchmark, provenance |
| EUDR-029 | MitigationPlan JSON | strategy_id, measures[], effectiveness[], residual_risk, timeline |

### Appendix G: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 on the making available on the Union market and the export from the Union of certain commodities and products associated with deforestation and forest degradation
2. EU Deforestation Regulation Guidance Document (European Commission, DG Environment)
3. EUDR Technical Specifications for the Information System (DG SANTE)
4. EUDR Annex II -- Content of the Due Diligence Statement
5. Commission Implementing Regulation on the EU Information System (expected 2025/2026)
6. ISO 22095:2020 -- Chain of Custody -- General Terminology and Models
7. ISO 19011:2018 -- Guidelines for Auditing Management Systems
8. EU TARIC database -- Combined Nomenclature and HS Code reference
9. ISO 3166-1:2020 -- Country codes

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-11 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-11 | GL-ProductManager | Initial draft created: 9 P0 features, 7 engines, 18 metrics, 12 API endpoints, 16 RBAC permissions, 9 DB tables (V118), regulatory mapping for Articles 4/9/10/11/12/13/14/15/16/29/31, 4 personas, integration with 29 upstream agents |
