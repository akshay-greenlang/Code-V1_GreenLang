# PRD: AGENT-EUDR-037 -- Due Diligence Statement Creator

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-037 |
| **Agent ID** | GL-EUDR-DDSC-037 |
| **Component** | Due Diligence Statement Creator Agent |
| **Category** | EUDR Regulatory Agent -- Reporting (Category 6) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-13 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 4 (Due Diligence Obligations), 9 (Information Requirements and Geolocation), 10 (Risk Assessment), 11 (Risk Mitigation), 12 (Due Diligence Statements), 13 (Simplified Due Diligence), 14 (Operator Registration), 29 (Country Benchmarking), 31 (Record Keeping), 33 (EU Information System); Annex II (DDS Content Requirements); eIDAS Regulation (EU) No 910/2014 (Electronic Identification and Qualified Electronic Signatures) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Prerequisite Agents** | AGENT-EUDR-001 through AGENT-EUDR-015 (Supply Chain Traceability), AGENT-EUDR-016 through AGENT-EUDR-025 (Risk Assessment and Due Diligence), AGENT-EUDR-030 (Documentation Generator), AGENT-EUDR-036 (EU Information System Interface) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 4 places a binding obligation on every operator and trader: before any regulated commodity or derived product may be placed on the EU market, made available on the EU market, or exported from the EU, the operator must submit a Due Diligence Statement (DDS) to the EU Information System established under Article 33. The DDS is not merely a form -- it is the formal, legally binding declaration that the operator has exercised due diligence in accordance with Articles 8 through 11, has collected all information required by Article 9, has conducted a risk assessment per Article 10, has applied adequate risk mitigation measures per Article 11 where risks were identified, and has concluded that the commodity is deforestation-free and was produced in accordance with the relevant legislation of the country of production. The DDS content requirements are specified in Annex II and include operator identification, product description with HS/CN codes, quantities, country of production, geolocation of all production plots per Article 9(1)(d), production date ranges, supplier and buyer identification, deforestation-free evidence, legal production evidence, risk assessment conclusions, mitigation measures where applicable, and a compliance declaration. Each DDS must be submitted before the product enters the EU market, and the EU Information System assigns a unique reference number that must accompany the product through the entire downstream supply chain.

The GreenLang platform has built a comprehensive suite of 36 EUDR agents spanning the full compliance lifecycle. Supply Chain Traceability agents (EUDR-001 through EUDR-015) provide supply chain mapping, geolocation verification, satellite monitoring, forest cover analysis, land use change detection, plot boundary management, GPS coordinate validation, multi-tier supplier tracking, chain of custody management, segregation verification, mass balance calculation, document authentication, blockchain integration, QR code generation, and mobile data collection. Risk Assessment agents (EUDR-016 through EUDR-020) provide country risk evaluation, supplier risk scoring, commodity risk analysis, corruption index monitoring, and deforestation alerting. Due Diligence agents (EUDR-021 through EUDR-035) provide indigenous rights checking, protected area validation, legal compliance verification, third-party audit management, risk mitigation advisory, due diligence orchestration, information gathering coordination, risk assessment engine operation, mitigation measure design, documentation generation, stakeholder engagement, grievance mechanism management, continuous monitoring, annual review scheduling, and improvement plan creation. The EU Information System Interface agent (EUDR-036) handles external transmission to the EU IS, operator registration, submission status tracking, and audit trail recording.

However, there is a critical gap between the documentation generation capabilities of EUDR-030 and the transmission capabilities of EUDR-036. EUDR-030 generates raw DDS document content from upstream agent outputs. EUDR-036 transmits finalized documents to the EU Information System. Between these two agents, there is no specialized engine that performs the critical task of assembling a complete, regulation-compliant Due Diligence Statement by pulling together all required data elements from 36 upstream agents, formatting geolocation data to exact Article 9 specifications, integrating risk assessment scores from all 10 risk evaluation agents (EUDR-016 through EUDR-025), compiling supply chain traceability data from all 15 traceability agents (EUDR-001 through EUDR-015), validating that every mandatory Article 4 field is present and correct, packaging the DDS with all supporting evidence and certificates, preparing the document for qualified electronic signature per eIDAS requirements, and managing DDS versions including amendments and withdrawals. Operators face the following gaps:

- **No unified DDS assembly from all 36 agents**: EUDR-030 generates document components, but the actual assembly of a complete DDS that draws data from all 36 upstream agents -- reconciling potentially conflicting data from different agent outputs, resolving data freshness issues, and ensuring cross-agent data consistency -- requires a dedicated assembler that understands the full Article 4 data model.

- **No Article 9 geolocation formatting to EU IS specifications**: Plot geolocation data exists in various formats across EUDR-002, EUDR-006, and EUDR-007. The EU Information System requires geolocation in a precise format: WGS 84 coordinate reference system, minimum 6 decimal places, single-point coordinates for plots under 4 hectares, polygon coordinates with closed rings for plots over 4 hectares, counter-clockwise winding order for exterior rings. There is no engine that takes raw geolocation data from multiple upstream agents and produces the exact formatted output required by the EU IS DDS submission schema.

- **No integrated risk score aggregation across all 10 risk agents**: Risk assessment data comes from EUDR-016 (country risk), EUDR-017 (supplier risk), EUDR-018 (commodity risk), EUDR-019 (corruption index), EUDR-020 (deforestation alerts), EUDR-021 (indigenous rights), EUDR-022 (protected areas), EUDR-023 (legal compliance), EUDR-024 (third-party audits), and EUDR-025 (risk mitigation). There is no engine that pulls risk scores from all 10 agents, reconciles them into the Article 10 risk assessment structure, and formats the result for DDS inclusion.

- **No supply chain data compilation from all 15 traceability agents**: Supply chain traceability data is distributed across EUDR-001 through EUDR-015. A complete DDS requires supplier information from EUDR-008, chain of custody data from EUDR-009, segregation verification from EUDR-010, mass balance data from EUDR-011, document authentication from EUDR-012, and blockchain provenance from EUDR-013. There is no compiler that aggregates all traceability data into the supply chain section of the DDS.

- **No commodity-specific traceability aggregation**: Each of the 7 EUDR-regulated commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood) has unique traceability requirements, different supply chain archetypes, different derived product classifications, and different CN code mappings. There is no aggregator that handles commodity-specific data assembly with the correct product classifications, species names, and HS/CN code mappings.

- **No comprehensive Article 4 compliance validation**: Before a DDS can be submitted, it must be validated against every mandatory requirement of Article 4 and Annex II. This validation spans 15 mandatory DDS fields, geolocation format requirements, risk assessment completeness, mitigation adequacy, and compliance conclusion consistency. There is no validator that performs this comprehensive pre-submission check across all data elements.

- **No evidence and certificate packaging**: A DDS submission to the EU Information System must be accompanied by supporting evidence: certificates (FSC, RSPO, Rainforest Alliance), satellite verification reports, chain of custody records, risk assessment reports, and mitigation documentation. There is no packager that bundles all evidence from upstream agents into a structured, indexed package that accompanies the DDS.

- **No qualified electronic signature preparation**: Under eIDAS Regulation (EU) No 910/2014, DDS documents submitted to the EU Information System may require or benefit from qualified electronic signatures to establish legal validity and non-repudiation. There is no engine that prepares DDS documents for qualified electronic signature by generating signature-ready document hashes, managing signature metadata, and formatting signed documents for EU IS acceptance.

- **No DDS version control with amendment and withdrawal**: DDS documents have a lifecycle that extends beyond initial submission. Operators may need to amend a DDS when new information becomes available, correct errors discovered after submission, or withdraw a DDS entirely when products are not placed on the market. There is no version controller that manages DDS amendments, tracks version lineage, records withdrawal reasons, and ensures that amended DDS documents maintain provenance chains back to the original.

Without solving these problems, operators cannot produce legally complete Due Diligence Statements that satisfy Article 4 requirements. The gap between raw documentation generation (EUDR-030) and EU IS transmission (EUDR-036) leaves operators with DDS documents that may be incomplete, inconsistently formatted, missing required evidence, or lacking the version control needed for regulatory compliance. This is the critical assembly and validation gap that exposes operators to fines of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, and public naming under Articles 23-25.

### 1.2 Solution Overview

Agent-EUDR-037: Due Diligence Statement Creator is the specialized assembly, validation, and packaging agent that sits between EUDR-030 (Documentation Generator) and EUDR-036 (EU Information System Interface) in the EUDR reporting pipeline. It receives raw document components from EUDR-030, pulls supplementary data directly from all 36 upstream EUDR agents, assembles complete Article 4-compliant Due Diligence Statements, validates them against every mandatory requirement, packages them with supporting evidence, prepares them for qualified electronic signature, manages their version lifecycle, and delivers finalized DDS packages to EUDR-036 for EU Information System submission. It is the 37th agent in the EUDR agent family and the second agent in the Reporting sub-category (Category 6).

Core capabilities:

1. **DDS Document Assembler** -- Compiles all required EUDR Article 4 and Annex II data elements into a complete, structured Due Diligence Statement. Pulls operator identification, product descriptions, HS/CN codes, quantities, country of production, geolocation references, production dates, supplier and buyer information, deforestation-free evidence references, legal production evidence references, risk assessment conclusions, mitigation measures summaries, and compliance declarations from EUDR-030 document components and directly from upstream agents where needed. Reconciles data from multiple sources when conflicts exist. Generates both JSON and XML format DDS documents conforming to the EU IS submission schema. Supports standard DDS (Articles 4-11) and simplified DDS (Article 13 for low-risk country products).

2. **Geolocation Formatter** -- Formats all production plot geolocation data to the exact specifications required by EUDR Article 9(1)(d) and the EU Information System. Normalizes coordinates to WGS 84 (EPSG:4326), enforces minimum 6 decimal places, formats single-point coordinates for plots under 4 hectares, formats polygon coordinates with closed rings for plots over 4 hectares, validates and corrects polygon winding order to counter-clockwise, handles coordinate reference system transformations from non-WGS84 sources, groups geolocations by country of production, and generates geolocation quality reports.

3. **Risk Assessment Integrator** -- Pulls risk assessment scores from all 10 risk evaluation agents (EUDR-016 through EUDR-025), reconciles them into a unified risk assessment structure per Article 10, documents Article 10(2) criterion-by-criterion evaluations, includes Article 29 country benchmarking determinations, evaluates Article 13 simplified due diligence eligibility, and formats the composite risk profile for DDS inclusion. All risk integration is deterministic with no LLM involvement.

4. **Supply Chain Data Compiler** -- Aggregates supply chain traceability data from all 15 Supply Chain Traceability agents (EUDR-001 through EUDR-015) into the supply chain section of the DDS. Compiles supply chain mapping data, geolocation verification results, satellite monitoring evidence, forest cover analysis, land use change detection, plot boundary data, GPS validation results, multi-tier supplier information, chain of custody records, segregation verification, mass balance calculations, document authentication certificates, blockchain provenance records, QR code references, and mobile data collection results.

5. **Commodity Traceability Aggregator** -- Consolidates commodity-specific traceability data for each of the 7 EUDR-regulated commodities. Handles commodity-specific supply chain archetypes (cattle ranch-to-slaughterhouse, cocoa smallholder-to-cooperative, coffee farm-to-mill, palm oil plantation-to-refinery, rubber smallholder-to-processor, soya farm-to-crusher, wood forest-to-sawmill). Maps commodity-specific product classifications, scientific names, HS/CN codes, and derived product categories. Ensures commodity-specific Article 9 requirements are met.

6. **Compliance Validator** -- Verifies that assembled DDS documents meet every mandatory requirement of Article 4 and Annex II before the DDS proceeds to packaging and submission. Validates all 15 Annex II mandatory fields, geolocation format compliance, risk assessment completeness across all Article 10(2) criteria, mitigation adequacy per Article 11, compliance conclusion consistency, cross-field validation (quantities match mass balance, countries match geolocation, dates are within range), and generates a comprehensive validation report with readiness score and remediation guidance for any failures.

7. **Document Packager** -- Bundles the validated DDS document with all supporting evidence, certificates, and documentation into a structured, indexed submission package. Attaches risk assessment reports, mitigation documentation, supply chain maps, certificate copies (FSC, RSPO, Rainforest Alliance, UTZ), satellite verification evidence, geolocation validation reports, blockchain provenance records, and chain of custody documentation. Generates package manifests with SHA-256 hashes, table of contents, and cross-reference indices. Compresses packages and manages size limits for EU IS transmission.

8. **Digital Signature Preparer** -- Prepares finalized DDS documents and packages for qualified electronic signature (QES) per eIDAS Regulation requirements. Generates signature-ready document hashes (SHA-256), creates signature metadata including signing time, signer identity, and certificate references, formats documents for PAdES (PDF Advanced Electronic Signatures) or XAdES (XML Advanced Electronic Signatures) standards, validates that signed documents maintain integrity after signature application, and records signature provenance in the audit trail.

9. **Version Controller** -- Manages the complete DDS document lifecycle including amendments, corrections, and withdrawals. Assigns version numbers, tracks document state machine (DRAFT, ASSEMBLED, VALIDATED, PACKAGED, SIGNED, SUBMITTED, ACKNOWLEDGED, AMENDED, WITHDRAWN), records every state transition with actor, timestamp, and reason, supports DDS amendment with change tracking and provenance chains, supports DDS withdrawal with reason codes and replacement references, enforces 5-year retention per Article 31, and generates version comparison reports.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| DDS assembly time | < 30 seconds per DDS from all upstream agent data | p99 latency from assemble-dds request to document ready |
| Article 4 mandatory field completeness | 100% of mandatory fields populated when upstream data available | Automated completeness validator against Article 4 and Annex II checklist |
| Geolocation format compliance | 100% of coordinates pass EU IS geolocation validation | Geolocation validation pass rate tracked per DDS |
| Risk assessment integration completeness | 100% of Article 10(2) criteria evaluated and documented | Cross-validation against all 10 risk agents |
| Supply chain data compilation | 100% of available traceability data included | Coverage check against 15 traceability agents |
| Compliance validation pass rate | >= 98% of assembled DDS pass validation on first attempt | Validation result tracking (pass/fail ratio) |
| Evidence package completeness | 100% of referenced evidence included in package | Package manifest completeness audit |
| Digital signature preparation success | 100% of signed documents maintain integrity | Post-signature integrity verification |
| Version control integrity | 100% of versions with valid SHA-256 provenance chain | Automated hash chain verification |
| EU IS acceptance rate | >= 98% of submitted DDS accepted on first attempt | Tracking via EUDR-036 submission status |
| Amendment turnaround time | < 15 minutes from amendment request to resubmission-ready DDS | Time from amendment initiation to validated amended DDS |
| 5-year retention compliance | 100% of DDS documents retrievable within retention period | Automated retention validation tests |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ EUDR-affected operators and traders across the EU must submit Due Diligence Statements for every regulated commodity placed on the EU market. The regulatory compliance technology market for DDS creation and management is estimated at 2-5 billion EUR, as each operator may need to generate dozens to thousands of DDS documents per year depending on their product portfolio and import volume.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring automated DDS assembly, validation, and packaging tools. These operators collectively process millions of commodity shipments annually, each requiring a compliant DDS. Estimated market of 600M-1.2B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 30-50M EUR in DDS creation module ARR. Enterprise customers are the primary target because they generate the highest volume of DDS documents and face the greatest complexity in assembling data from multi-tier, multi-commodity supply chains.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities requiring automated DDS creation at scale
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with complex, multi-commodity portfolios spanning dozens of source countries
- Timber and paper industry operators with hundreds of product lines and multi-step processing chains (forest to finished furniture)
- Compliance departments managing quarterly DDS submission deadlines across multiple business units and Member States

**Secondary:**
- Customs brokers and freight forwarders preparing DDS on behalf of importer clients
- Compliance consultants managing DDS creation for multiple operator clients
- Certification bodies requiring standardized DDS formats for their audit processes
- SME importers (1,000-10,000 shipments/year) ahead of June 30, 2026 enforcement deadline

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual DDS preparation (spreadsheets, forms) | No cost; familiar tools | 4-8 hours per DDS; error-prone; no data integration; no geolocation formatting; no validation | < 30 seconds per DDS; automated data integration from 36 agents; zero-error geolocation formatting; comprehensive validation |
| Generic compliance platforms (SAP GRC, MetricStream) | Enterprise integration; multi-regulation | Not EUDR-specific; no Article 9 geolocation handling; no risk agent integration; no DDS schema compliance | Purpose-built for EUDR Article 4; native integration with 36 EUDR agents; full Article 9 compliance |
| Compliance consulting firms | EUDR regulatory expertise; manual DDS preparation | EUR 5K-20K per DDS batch; 2-5 day turnaround; cannot scale; no version control | Fully automated; EUR 50-200 per DDS; < 30 seconds; complete version control |
| Niche EUDR tools (early movers) | First to market; DDS form filling | Single-agent DDS filling; no cross-agent data integration; no evidence packaging; no signature preparation | Full 36-agent integration; comprehensive evidence packaging; eIDAS signature preparation |
| In-house custom development | Tailored to organization | 6-12 months to build; fragile to regulatory changes; no upstream agent integration; maintenance burden | Ready now; continuous regulatory updates; native 36-agent integration; managed updates |

### 2.4 Differentiation Strategy

1. **36-agent integration depth** -- Not a standalone DDS form filler. Natively integrates with all 36 upstream EUDR agents. Data flows from supply chain mapping, geolocation verification, satellite monitoring, risk assessment, due diligence, and documentation generation directly into the DDS assembly engine without manual intervention.
2. **Regulatory precision** -- Every DDS field maps to a specific EUDR Article and Annex II requirement. Geolocation formatting follows Article 9(1)(d) to the decimal place. Risk assessment integration covers all Article 10(2) criteria. Compliance validation checks every mandatory requirement.
3. **Zero-hallucination assembly** -- All DDS assembly, formatting, and validation is deterministic. No LLM in the critical path. Same inputs produce bit-identical DDS output. Complete SHA-256 provenance chain from raw data through assembly to final document.
4. **Evidence packaging** -- DDS documents are not submitted in isolation. The Document Packager bundles every referenced certificate, satellite report, risk assessment, and chain of custody record into an indexed, cross-referenced evidence package that satisfies Articles 14-16 inspection requirements.
5. **eIDAS signature readiness** -- Built-in preparation for qualified electronic signatures per eIDAS standards, providing legal validity and non-repudiation for DDS submissions.
6. **Complete lifecycle management** -- From initial assembly through validation, packaging, signing, submission, acknowledgement, amendment, and withdrawal, with 5-year retention and full version provenance.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable operators to create fully Article 4-compliant DDS documents from all upstream agent data | 100% of generated DDS pass EU IS schema validation and Article 4 completeness checks | Q2 2026 |
| BG-2 | Reduce DDS creation time from hours of manual assembly to seconds of automated assembly | 99.5% reduction in per-DDS creation time (4-8 hours to < 30 seconds) | Q2 2026 |
| BG-3 | Eliminate DDS submission rejections caused by geolocation formatting or data completeness errors | >= 98% first-attempt EU IS acceptance rate for DDS created by this agent | Q2 2026 |
| BG-4 | Ensure zero EUDR penalties for active customers due to DDS deficiencies | Zero Article 23-25 penalties attributable to DDS quality for GreenLang customers | Ongoing |
| BG-5 | Become the reference EUDR DDS creation solution for enterprise operators | 500+ enterprise customers using automated DDS creation | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Automated DDS assembly | Assemble complete Article 4-compliant DDS from all 36 upstream agent outputs with zero manual intervention |
| PG-2 | Geolocation precision | Format all Article 9 geolocation data to exact EU IS specifications (WGS 84, 6 decimal places, polygon requirements) |
| PG-3 | Risk integration | Integrate risk scores from all 10 risk agents (EUDR-016 through EUDR-025) into unified Article 10 assessment |
| PG-4 | Supply chain compilation | Compile traceability data from all 15 supply chain agents (EUDR-001 through EUDR-015) |
| PG-5 | Commodity specificity | Handle all 7 EUDR commodity types with correct product classifications and supply chain archetypes |
| PG-6 | Compliance validation | Validate every Article 4 and Annex II mandatory requirement before packaging |
| PG-7 | Evidence packaging | Bundle DDS with all supporting evidence in structured, indexed packages |
| PG-8 | Signature readiness | Prepare DDS for qualified electronic signature per eIDAS standards |
| PG-9 | Version lifecycle | Manage DDS amendments, corrections, and withdrawals with full provenance |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | DDS assembly performance | < 30 seconds p99 for single DDS assembly from all upstream data |
| TG-2 | Batch assembly throughput | >= 100 DDS documents assembled per minute |
| TG-3 | Geolocation formatting | < 5 seconds for 10,000 plots; 100% WGS 84 compliance |
| TG-4 | Risk integration latency | < 10 seconds to pull and reconcile scores from all 10 risk agents |
| TG-5 | Validation performance | < 5 seconds for comprehensive Article 4 validation |
| TG-6 | API response time | < 200ms p95 for standard queries (GET endpoints) |
| TG-7 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-8 | Zero-hallucination | 100% deterministic assembly, formatting, and validation; no LLM in critical path |
| TG-9 | Data integrity | SHA-256 provenance hash on every DDS document, version, and evidence artifact |
| TG-10 | Availability | 99.5% uptime SLA for DDS creation service |

### 3.4 Non-Goals

1. Raw DDS content generation from individual agent outputs (EUDR-030 Documentation Generator handles initial content generation)
2. External transmission to the EU Information System (EUDR-036 EU Information System Interface handles API submission)
3. Operator registration management with competent authorities (EUDR-036 handles registration lifecycle)
4. Due diligence workflow orchestration (EUDR-026 Due Diligence Orchestrator handles workflow)
5. Risk assessment calculation (EUDR-028 Risk Assessment Engine handles scoring)
6. Mitigation measure design (EUDR-029 Mitigation Measure Designer handles mitigation)
7. Satellite data acquisition or analysis (EUDR-003/004/005 handle satellite operations)
8. Supply chain graph visualization (EUDR-001 handles graph topology and rendering)
9. Carbon footprint or GHG reporting (GL-GHG-APP handles this)
10. Mobile data collection (EUDR-015 handles field data capture)

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, importing cocoa, palm oil, and soya from 15+ countries; registered in 4 EU Member States |
| **EUDR Pressure** | Must create and submit 200+ DDS documents per quarter across 3 commodity categories; each DDS must pull data from supply chain mapping, geolocation verification, risk assessment, and mitigation -- all produced by different upstream agents |
| **Pain Points** | EUDR-030 generates raw document components but they must be manually assembled, geolocation data must be reformatted, risk scores from 10 different agents must be reconciled, supporting evidence must be collected and attached, and the complete DDS must be validated before submission. This manual assembly takes 2-4 hours per DDS and is prone to errors that cause EU IS rejection. |
| **Goals** | One-click DDS creation that automatically pulls all data from upstream agents, formats geolocation, integrates risk scores, validates completeness, packages evidence, and delivers a submission-ready DDS to EUDR-036; zero submission rejections; full version control for amendments |
| **Technical Skill** | Moderate -- comfortable with web applications and regulatory portals but not a developer |

### Persona 2: Supply Chain Analyst -- Lukas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Analyst at an EU timber importer |
| **Company** | 800 employees, importing tropical and temperate wood from 20+ countries through complex multi-tier processing chains |
| **EUDR Pressure** | Must ensure DDS documents accurately reflect the complete supply chain for wood products that pass through 5-8 intermediaries (forest, sawmill, veneer, plywood, furniture manufacturer, trader, importer). Each intermediary's data must be compiled into the DDS supply chain section. Wood-specific HS/CN codes and scientific species names must be correct. |
| **Pain Points** | Supply chain data is distributed across EUDR-001 (mapping), EUDR-008 (multi-tier tracking), EUDR-009 (chain of custody), EUDR-010 (segregation), EUDR-011 (mass balance), and EUDR-012 (certificates). Manually compiling this data into a DDS takes hours and risks missing intermediaries or misattributing quantities. Geolocation data from forest concessions requires polygon formatting for plots over 4 hectares. |
| **Goals** | Automated supply chain data compilation from all 15 traceability agents; correct commodity-specific formatting for wood products (scientific species names, concession boundaries); mass balance verification in the DDS; complete chain of custody documentation in the evidence package |
| **Technical Skill** | High -- comfortable with data tools, APIs, and basic scripting |

### Persona 3: Procurement Manager -- Ana (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Procurement Director at a palm oil refinery |
| **Company** | 3,000 employees, sourcing from 200+ plantations across Indonesia and Malaysia |
| **EUDR Pressure** | Must create DDS for refined palm oil products that trace to hundreds of production plots across multiple countries. Each DDS must include geolocation for every contributing plantation. Palm oil-specific traceability (RSPO mass balance) must be documented. |
| **Pain Points** | A single DDS for refined palm oil may reference 50-200 production plots across 2-3 countries. Manually collecting geolocation for all plots, verifying coordinates, and formatting them to EU IS specifications is a multi-day effort. Risk scores vary significantly across plantations in different regions. |
| **Goals** | Automated geolocation aggregation from all contributing plots; commodity-specific RSPO mass balance integration; multi-country geolocation grouping; clear risk differentiation by production origin |
| **Technical Skill** | Low-moderate -- uses ERP and web applications |

### Persona 4: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm conducting Article 16 substantive checks |
| **EUDR Pressure** | Must verify that operator DDS documents contain all required data elements, that geolocation data is correctly formatted, that risk assessments are complete, and that evidence packages are comprehensive and verifiable |
| **Pain Points** | DDS documents vary widely in quality and completeness; geolocation data is often incorrectly formatted; evidence packages are disorganized; version history is unclear; it is difficult to verify that a DDS was not modified after submission |
| **Goals** | Access to structured, validated DDS documents with clear version history; evidence packages with SHA-256 integrity hashes; signature provenance for non-repudiation; ability to trace every DDS field back to its source agent data |
| **Technical Skill** | Moderate -- comfortable with audit software and document review |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(1)** | No relevant commodity or product shall be placed on, made available on, or exported from the EU market unless a DDS has been submitted | DDSDocumentAssembler creates the DDS; ComplianceValidator ensures it meets all requirements before submission via EUDR-036 |
| **Art. 4(2)** | Due diligence system shall include information collection, risk assessment, risk mitigation | DDSDocumentAssembler compiles information from EUDR-027; RiskAssessmentIntegrator compiles risk data from EUDR-028; supply chain and mitigation data from EUDR-029 |
| **Art. 4(2)(a)** | Collection of information required by Article 9 | SupplyChainDataCompiler and CommodityTraceabilityAggregator gather all Article 9 data from upstream agents |
| **Art. 4(2)(b)** | Risk assessment in accordance with Article 10 | RiskAssessmentIntegrator pulls scores from EUDR-016 through EUDR-025 and structures per Article 10 |
| **Art. 4(2)(c)** | Risk mitigation in accordance with Article 11 | DDSDocumentAssembler includes mitigation data from EUDR-029 Mitigation Measure Designer |
| **Art. 9(1)(a)** | Description of the product, including trade name, type, common and scientific name of species | DDSDocumentAssembler populates from EUDR-001 supply chain data; CommodityTraceabilityAggregator provides commodity-specific names |
| **Art. 9(1)(b)** | Quantity of the relevant commodity or product (net mass in kg, volume in m3, or number of items) | DDSDocumentAssembler populates from EUDR-011 mass balance and EUDR-010 segregation data |
| **Art. 9(1)(c)** | Country of production, and where applicable, parts thereof | DDSDocumentAssembler populates from EUDR-001 and EUDR-008; GeolocationFormatter groups by country |
| **Art. 9(1)(d)** | Geolocation of all plots of land where the commodity was produced: GPS coordinates (single point for plots <= 4 ha) or polygon (for plots > 4 ha) in WGS 84 | GeolocationFormatter normalizes all coordinates to WGS 84, 6 decimal places, enforces polygon for > 4 ha, validates closure and winding |
| **Art. 9(1)(e)** | Date or time range of production | DDSDocumentAssembler populates from upstream supply chain temporal data |
| **Art. 9(1)(f)** | Name, postal address, and email of the supplier | DDSDocumentAssembler populates from EUDR-008 multi-tier supplier tracker |
| **Art. 9(1)(g)** | Name, postal address, and email of the buyer | DDSDocumentAssembler populates from operator records |
| **Art. 9(2)** | Adequately conclusive and verifiable information | DocumentPackager bundles all evidence with SHA-256 provenance hashes for verifiability |
| **Art. 10(1)** | Operators shall identify and assess the risk that relevant commodities were not produced in accordance with the provisions of this Regulation | RiskAssessmentIntegrator compiles risk assessment from EUDR-028 and all 10 risk agents |
| **Art. 10(2)(a)** | Risk of deforestation or forest degradation in the country of production | RiskAssessmentIntegrator includes country deforestation risk from EUDR-016 and EUDR-020 |
| **Art. 10(2)(b)** | Presence of forests including primary forests in the country or region | RiskAssessmentIntegrator includes forest presence data from EUDR-004 |
| **Art. 10(2)(c)** | Presence of indigenous peoples in the sourcing area | RiskAssessmentIntegrator includes indigenous rights assessment from EUDR-021 |
| **Art. 10(2)(d)** | Consultation with indigenous peoples where relevant | RiskAssessmentIntegrator includes consultation evidence from EUDR-021 |
| **Art. 10(2)(e)** | Concerns in relation to the country of production or parts thereof | RiskAssessmentIntegrator includes country risk concerns from EUDR-016 and EUDR-019 |
| **Art. 10(2)(f)** | Risk of circumvention, mixing with products of unknown origin | RiskAssessmentIntegrator includes chain of custody verification from EUDR-009/010/011 |
| **Art. 10(2)(g)** | History of compliance of the supplier | RiskAssessmentIntegrator includes supplier compliance history from EUDR-017 |
| **Art. 11(1)** | Adopt risk mitigation measures adequate to reach a negligible level of risk | DDSDocumentAssembler includes mitigation measures from EUDR-029 in the DDS mitigation section |
| **Art. 11(2)(a-c)** | Additional information, independent surveys/audits, other adequate measures | DDSDocumentAssembler documents each mitigation category per Article 11(2) sub-paragraphs |
| **Art. 12(1)** | Submit DDS to the Information System before placing/making available/exporting | ComplianceValidator ensures DDS is complete; VersionController tracks submission state; EUDR-036 handles actual transmission |
| **Art. 12(2)** | DDS shall contain information listed in Annex II | ComplianceValidator validates all 15 Annex II fields are present and correctly formatted |
| **Art. 12(3)** | Information System assigns reference number to each DDS | VersionController records EU IS reference numbers received via EUDR-036 |
| **Art. 13** | Simplified due diligence for products from low-risk countries (Article 29) | DDSDocumentAssembler supports simplified DDS variant; ComplianceValidator applies simplified validation rules |
| **Art. 14-16** | Competent authority checks and inspections | DocumentPackager creates inspection-ready evidence packages; VersionController provides complete version history |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | RiskAssessmentIntegrator includes country benchmarking classification from EUDR-016 in the risk assessment section |
| **Art. 31(1)** | Retain records of DDS and supporting information for at least 5 years | VersionController enforces 5-year retention with automated tracking and expiry alerting |
| **Art. 33** | EU Information System for DDS submission | DDSDocumentAssembler produces DDS conforming to EU IS schema; handoff to EUDR-036 for transmission |
| **eIDAS Art. 3** | Electronic identification definitions | DigitalSignaturePreparer uses eIDAS-compliant identification for signature preparation |
| **eIDAS Art. 25** | Qualified electronic signatures have legal effect equivalent to handwritten signatures | DigitalSignaturePreparer supports QES standards (PAdES, XAdES) for DDS signing |
| **eIDAS Art. 26** | Requirements for qualified electronic signatures | DigitalSignaturePreparer validates signer identity, certificate validity, and signature integrity |

### 5.2 DDS Content Requirements (EUDR Annex II)

| # | DDS Field (Annex II) | Source Agent(s) | Assembly Engine |
|---|---------------------|-----------------|-----------------|
| 1 | Operator/trader identification (name, address, EORI) | Operator records, EUDR-001 | DDSDocumentAssembler |
| 2 | Registration number (per Article 14) | EUDR-036 OperatorRegistrationManager | DDSDocumentAssembler |
| 3 | Product description (trade name, type, common name, scientific name) | EUDR-001, EUDR-009, CommodityTraceabilityAggregator | DDSDocumentAssembler |
| 4 | HS/CN code(s) per Annex I | EUDR-001, EUDR-009, CommodityTraceabilityAggregator | DDSDocumentAssembler |
| 5 | Quantity (net mass in kg, volume in m3, or number of items) | EUDR-011, EUDR-010 | DDSDocumentAssembler |
| 6 | Country of production (ISO 3166-1 alpha-2) | EUDR-001, EUDR-008 | DDSDocumentAssembler |
| 7 | Geolocation of all production plots | EUDR-002, EUDR-006, EUDR-007 | GeolocationFormatter |
| 8 | Date or time range of production | EUDR-001, EUDR-008 | DDSDocumentAssembler |
| 9 | Supplier name, postal address, email | EUDR-008 | DDSDocumentAssembler |
| 10 | Buyer name, postal address, email | Operator records | DDSDocumentAssembler |
| 11 | Deforestation-free evidence | EUDR-003, EUDR-004, EUDR-005 | DDSDocumentAssembler |
| 12 | Legal production evidence | EUDR-023 | DDSDocumentAssembler |
| 13 | Risk assessment conclusion | EUDR-016 through EUDR-025, EUDR-028 | RiskAssessmentIntegrator |
| 14 | Risk mitigation measures (if applicable) | EUDR-029 | DDSDocumentAssembler |
| 15 | Compliance declaration | ComplianceValidator conclusion logic | DDSDocumentAssembler |

### 5.3 Covered Commodities and DDS Specifics

| Commodity | Scientific Names Required | HS Code Range | DDS Complexity | Typical Plot Count per DDS |
|-----------|--------------------------|---------------|----------------|---------------------------|
| **Cattle** | Bos taurus, Bos indicus | 0102, 0201-0202, 4101-4104 | High (animal movement tracking) | 5-50 (pasture/feedlot plots) |
| **Cocoa** | Theobroma cacao | 1801-1806 | Very High (thousands of smallholders) | 50-500 (cooperative plots) |
| **Coffee** | Coffea arabica, Coffea canephora | 0901 | High (altitude/origin segregation) | 20-200 (farm plots) |
| **Palm Oil** | Elaeis guineensis | 1511, 1513, 3823 | High (mass balance challenges) | 50-300 (plantation plots) |
| **Rubber** | Hevea brasiliensis | 4001, 4005-4017 | High (smallholder aggregation) | 30-200 (smallholder plots) |
| **Soya** | Glycine max | 1201, 1507, 2304 | Medium-High (large volumes, co-mingling) | 10-100 (farm plots) |
| **Wood** | Varies (hundreds of species) | 4401-4421, 9401-9403 | Very High (species identification, multi-step processing) | 5-50 (forest concession plots) |

### 5.4 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date referenced in all DDS deforestation-free declarations assembled by this agent |
| June 29, 2023 | Regulation entered into force | Legal basis for all DDS requirements; defines Annex II field specifications |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Operators must have DDS creation capability operational NOW; agent must be production-ready |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; batch DDS assembly critical for high-volume processing |
| Ongoing (quarterly) | Country benchmarking updates by EC | RiskAssessmentIntegrator must reflect current Article 29 classifications in every DDS |
| Ongoing (as needed) | DDS amendments when new information available | VersionController must support rapid amendment workflow with full provenance |
| 5 years post-submission | Record retention deadline | VersionController must maintain all DDS versions and evidence for 5 years per Article 31 |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core DDS assembly and data integration engine; Features 6-7 form the validation and packaging layer; Features 8-9 form the signature and lifecycle management layer.

**P0 Features 1-5: Core DDS Assembly and Data Integration Engine**

---

#### Feature 1: DDS Document Assembler

**User Story:**
```
As a compliance officer,
I want a complete Due Diligence Statement automatically assembled from all upstream EUDR agent data,
So that I can create a fully Article 4-compliant DDS without manually collecting and reconciling data from 36 different agents.
```

**Acceptance Criteria:**
- [ ] Receives DDS generation request with operator ID, product ID, commodity type, and supply chain reference
- [ ] Pulls operator identification data (legal name, registered address, EORI number) from operator records and EUDR-036 registration data
- [ ] Pulls product description (trade name, product type, common name, scientific name of species) from EUDR-001 supply chain data and EUDR-009 chain of custody records
- [ ] Pulls HS/CN codes from EUDR-001 and CommodityTraceabilityAggregator (Feature 5) with validation against Annex I product list
- [ ] Pulls quantity data (net mass, volume, or item count with unit) from EUDR-011 mass balance calculator and EUDR-010 segregation verifier
- [ ] Pulls country of production (ISO 3166-1 alpha-2 codes) from EUDR-001 supply chain mapping and EUDR-008 multi-tier supplier tracker
- [ ] Invokes GeolocationFormatter (Feature 2) to produce formatted geolocation data for all production plots
- [ ] Pulls production date ranges from upstream supply chain temporal data
- [ ] Pulls supplier identification (name, postal address, email) from EUDR-008 multi-tier supplier tracker
- [ ] Pulls buyer identification (name, postal address, email) from operator records
- [ ] Pulls deforestation-free evidence references from EUDR-003 satellite monitoring, EUDR-004 forest cover analysis, and EUDR-005 land use change detection
- [ ] Pulls legal production evidence references from EUDR-023 legal compliance verifier
- [ ] Invokes RiskAssessmentIntegrator (Feature 3) to produce the risk assessment conclusion section
- [ ] Pulls mitigation measures summary from EUDR-029 mitigation measure designer (when applicable)
- [ ] Generates compliance declaration based on deterministic logic: all information gathered (Article 9 complete), risk assessment conducted (Article 10 complete), mitigation applied where needed (Article 11 complete or not required), conclusion is deforestation-free and legally produced
- [ ] Reconciles data conflicts when multiple sources provide different values for the same field (priority order: most recent, most authoritative, manual override)
- [ ] Generates DDS in EU IS JSON schema format with all mandatory fields populated
- [ ] Generates DDS in EU IS XML schema format as alternative output
- [ ] Generates human-readable DDS summary in PDF format alongside machine-readable formats
- [ ] Supports standard DDS (Articles 4-11) for standard and high-risk country products
- [ ] Supports simplified DDS (Article 13) for products exclusively from low-risk countries per Article 29
- [ ] Assigns unique DDS reference number: GL-DDS-{operator_id}-{commodity}-{YYYYMMDD}-{sequence}
- [ ] Calculates and embeds SHA-256 provenance hash covering all DDS content fields
- [ ] Records complete assembly audit trail: which agent provided which data, retrieval timestamp, data version

**Non-Functional Requirements:**
- Performance: DDS assembly < 30 seconds p99 for single DDS with up to 500 plots
- Determinism: Same inputs produce bit-identical DDS output
- Auditability: Complete assembly provenance with agent-level attribution
- Schema compliance: 100% validation pass rate against EU IS schema

**Dependencies:**
- EUDR-030 Documentation Generator (raw document components)
- EUDR-001 Supply Chain Mapping Master (supply chain data, product info)
- EUDR-008 Multi-Tier Supplier Tracker (supplier information)
- EUDR-009 Chain of Custody Agent (product classification)
- EUDR-010 Segregation Verifier (quantity verification)
- EUDR-011 Mass Balance Calculator (quantity data)
- EUDR-023 Legal Compliance Verifier (legal evidence)
- EUDR-036 Operator Registration Manager (registration numbers)
- Feature 2: GeolocationFormatter (geolocation data)
- Feature 3: RiskAssessmentIntegrator (risk assessment)
- Feature 5: CommodityTraceabilityAggregator (commodity-specific data)

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 integration engineer)

**Edge Cases:**
- Multiple suppliers provide conflicting product descriptions -- Use most recent data with conflict flagged in audit trail
- Operator has no EORI number (non-EU exporter acting through representative) -- Allow representative EORI with delegation reference
- Product from both low-risk and standard-risk countries -- Generate standard DDS (simplified DDS requires ALL plots in low-risk countries)
- Upstream agent data is stale (> 30 days old) -- Generate freshness warning; allow assembly with stale data flagged
- Quantity data from mass balance differs from segregation verifier -- Flag discrepancy; use mass balance as authoritative with explanation

---

#### Feature 2: Geolocation Formatter

**User Story:**
```
As a compliance officer,
I want all production plot geolocation data automatically formatted to the exact specifications required by EUDR Article 9(1)(d),
So that I can ensure zero DDS submission rejections caused by geolocation format errors.
```

**Acceptance Criteria:**
- [ ] Retrieves raw geolocation data from EUDR-002 (Geolocation Verification Agent), EUDR-006 (Plot Boundary Manager), and EUDR-007 (GPS Coordinate Validator)
- [ ] Normalizes all coordinates to WGS 84 coordinate reference system (EPSG:4326)
- [ ] Transforms coordinates from non-WGS84 reference systems (UTM zones, national grids, local CRS) using pyproj geodetic library
- [ ] Enforces minimum 6 decimal places for all latitude and longitude values; pads with trailing zeros if source has fewer decimals
- [ ] Validates latitude range: -90.000000 to +90.000000
- [ ] Validates longitude range: -180.000000 to +180.000000
- [ ] Formats single-point coordinates (plots <= 4 ha) as JSON array: [latitude, longitude]
- [ ] Formats polygon coordinates (plots > 4 ha) as JSON array of coordinate arrays forming a closed ring: [[lat1,lon1],[lat2,lon2],...,[lat1,lon1]]
- [ ] Validates polygon closure: first vertex must equal last vertex (within tolerance of 1e-10 degrees)
- [ ] Automatically closes unclosed polygons by appending the first vertex as the last vertex
- [ ] Validates polygon winding order: counter-clockwise for exterior rings per GeoJSON convention
- [ ] Corrects clockwise winding to counter-clockwise using vertex reversal
- [ ] Validates polygon geometry: no self-intersections, no degenerate polygons (< 3 unique vertices), minimum area threshold
- [ ] Handles multi-plot commodities: formats geolocation data for products tracing to dozens or hundreds of production plots
- [ ] Groups geolocation data by country of production per EU IS structure requirements
- [ ] Generates geolocation quality report: total plots processed, plots with sufficient precision, plots requiring polygon upgrade, CRS transformations performed, validation errors found
- [ ] Flags plots with insufficient source precision for manual review
- [ ] Supports batch formatting: format geolocation for all plots in a DDS in a single operation

**Geolocation Formatting Rules:**

```
Single Point (plot <= 4 ha):
  Input:  lat=5.123, lon=-3.456 (from EUDR-007)
  Output: [5.123000, -3.456000]

Polygon (plot > 4 ha):
  Input:  [(5.123, -3.456), (5.124, -3.455), (5.123, -3.454), (5.122, -3.455)]
  Output: [[5.123000, -3.456000], [5.124000, -3.455000], [5.123000, -3.454000],
           [5.122000, -3.455000], [5.123000, -3.456000]]
  Note: Last vertex = first vertex (auto-closed); counter-clockwise winding

Multi-Plot by Country:
  Output: {
    "BR": [{"type": "point", "coordinates": [...]}, ...],
    "ID": [{"type": "polygon", "coordinates": [...]}, ...],
    "GH": [{"type": "point", "coordinates": [...]}, ...]
  }
```

**Non-Functional Requirements:**
- Precision: 100% of output coordinates have 6+ decimal places in WGS 84
- Accuracy: CRS transformations accurate to < 1 meter
- Performance: Format 10,000 plots in < 5 seconds
- Validation: 100% of output geometries pass EU IS geolocation validation

**Dependencies:**
- EUDR-002 Geolocation Verification Agent (verified coordinates)
- EUDR-006 Plot Boundary Manager Agent (polygon boundaries)
- EUDR-007 GPS Coordinate Validator Agent (validated GPS coordinates)
- pyproj for coordinate reference system transformations
- Shapely for polygon geometry validation and correction

**Estimated Effort:** 2 weeks (1 backend engineer with GIS expertise)

**Edge Cases:**
- Source coordinates in UTM Zone 37N (East Africa) -- Transform to WGS 84 using EPSG code lookup
- Polygon with self-intersection (bowtie shape) -- Flag as invalid with remediation guidance
- Plot straddles the antimeridian (180 degrees longitude) -- Split polygon at antimeridian per GeoJSON convention
- Very small plot (< 0.01 ha) with coordinates that round to same point at 6 decimals -- Flag for manual precision verification
- Plot area is 3.9 ha (borderline for polygon requirement) -- Format as point; include area for auditor reference
- Polygon with duplicate consecutive vertices -- Deduplicate while maintaining geometry integrity

---

#### Feature 3: Risk Assessment Integrator

**User Story:**
```
As a compliance officer,
I want risk assessment scores from all 10 risk evaluation agents automatically integrated into a unified risk profile for the DDS,
So that I can include a complete, structured Article 10 risk assessment in every Due Diligence Statement.
```

**Acceptance Criteria:**
- [ ] Pulls country risk scores and country benchmarking classification (Low/Standard/High per Article 29) from EUDR-016 Country Risk Evaluator
- [ ] Pulls supplier risk scores and compliance history from EUDR-017 Supplier Risk Scorer
- [ ] Pulls commodity risk scores and deforestation association ratings from EUDR-018 Commodity Risk Analyzer
- [ ] Pulls corruption perception index data from EUDR-019 Corruption Index Monitor
- [ ] Pulls deforestation alert data and satellite-verified deforestation status from EUDR-020 Deforestation Alert System
- [ ] Pulls indigenous rights assessment results from EUDR-021 Indigenous Rights Checker
- [ ] Pulls protected area proximity and overlap status from EUDR-022 Protected Area Validator
- [ ] Pulls legal compliance verification results from EUDR-023 Legal Compliance Verifier
- [ ] Pulls third-party audit findings and scores from EUDR-024 Third-Party Audit Manager
- [ ] Pulls risk mitigation effectiveness scores from EUDR-025 Risk Mitigation Advisor
- [ ] Pulls composite risk score and criterion evaluations from EUDR-028 Risk Assessment Engine (if available; otherwise computes from individual agents)
- [ ] Structures risk data into Article 10(2) criterion-by-criterion format covering all 7 criteria (a through g)
- [ ] Documents Article 29 country benchmarking determination with supporting data
- [ ] Evaluates Article 13 simplified due diligence eligibility (all plots in low-risk countries AND no substantiated concerns)
- [ ] Generates risk assessment summary for DDS inclusion (structured JSON section)
- [ ] Generates detailed risk assessment documentation for evidence package (human-readable report)
- [ ] Calculates overall risk classification: NEGLIGIBLE (compliant), LOW, STANDARD, HIGH
- [ ] Documents enhanced due diligence triggers when risk exceeds configurable threshold
- [ ] All risk integration is deterministic: same input scores produce identical output
- [ ] Records complete integration provenance: which agent provided which score, retrieval timestamp, score version

**Risk Integration Formula:**

```
Composite Risk = weighted_max(
    Country_Risk * W_country,           # W_country = 0.25 (from EUDR-016)
    Supplier_Risk * W_supplier,         # W_supplier = 0.15 (from EUDR-017)
    Commodity_Risk * W_commodity,       # W_commodity = 0.15 (from EUDR-018)
    Corruption_Risk * W_corruption,     # W_corruption = 0.10 (from EUDR-019)
    Deforestation_Risk * W_deforestation, # W_deforestation = 0.15 (from EUDR-020)
    Indigenous_Rights_Risk * W_indigenous, # W_indigenous = 0.05 (from EUDR-021)
    Protected_Area_Risk * W_protected,  # W_protected = 0.05 (from EUDR-022)
    Legal_Risk * W_legal,               # W_legal = 0.05 (from EUDR-023)
    Audit_Risk * W_audit,               # W_audit = 0.025 (from EUDR-024)
    Mitigation_Risk * W_mitigation      # W_mitigation = 0.025 (from EUDR-025)
)

Classification:
- NEGLIGIBLE: Composite Risk < 20 AND no HIGH individual scores
- LOW: Composite Risk < 40 AND no HIGH individual scores
- STANDARD: Composite Risk < 70 OR any individual score >= 60
- HIGH: Composite Risk >= 70 OR any critical individual score >= 80
```

**Non-Functional Requirements:**
- Accuracy: 100% faithful representation of upstream agent risk scores
- Determinism: Bit-perfect reproducibility of risk integration from same inputs
- Performance: Risk integration < 10 seconds for all 10 agents
- Auditability: Complete provenance from DDS risk section back to individual agent outputs

**Dependencies:**
- EUDR-016 Country Risk Evaluator
- EUDR-017 Supplier Risk Scorer
- EUDR-018 Commodity Risk Analyzer
- EUDR-019 Corruption Index Monitor
- EUDR-020 Deforestation Alert System
- EUDR-021 Indigenous Rights Checker
- EUDR-022 Protected Area Validator
- EUDR-023 Legal Compliance Verifier
- EUDR-024 Third-Party Audit Manager
- EUDR-025 Risk Mitigation Advisor
- EUDR-028 Risk Assessment Engine (optional, for pre-computed composite scores)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- One risk agent is unavailable or returns error -- Assemble with available scores; flag missing agent with impact assessment; block DDS if critical agent missing (EUDR-016 country risk is mandatory)
- Risk scores from different agents are inconsistent (country says LOW, deforestation says HIGH) -- Apply highest-risk-wins principle per Article 10; document the discrepancy
- All plots in low-risk country but supplier has HIGH compliance risk -- Cannot use simplified DDS; must use standard DDS with full risk documentation
- Risk agent returns score of 0 (no data) vs. score of 0 (assessed as zero risk) -- Distinguish missing data from assessed-zero; flag missing data

---

#### Feature 4: Supply Chain Data Compiler

**User Story:**
```
As a supply chain analyst,
I want all supply chain traceability data from the 15 traceability agents automatically compiled into the supply chain section of the DDS,
So that I can ensure the DDS contains complete, accurate traceability information for every product.
```

**Acceptance Criteria:**
- [ ] Pulls supply chain graph data from EUDR-001 Supply Chain Mapping Master (nodes, edges, tier depth, graph topology)
- [ ] Pulls geolocation verification results from EUDR-002 Geolocation Verification Agent (coordinate verification status per plot)
- [ ] Pulls satellite monitoring evidence from EUDR-003 Satellite Monitoring Agent (satellite imagery analysis timestamps and results)
- [ ] Pulls forest cover analysis results from EUDR-004 Forest Cover Analysis Agent (forest cover percentage, change detection)
- [ ] Pulls land use change detection from EUDR-005 Land Use Change Detector Agent (land use change status since Dec 31, 2020 cutoff)
- [ ] Pulls plot boundary data from EUDR-006 Plot Boundary Manager Agent (registered boundaries, area calculations)
- [ ] Pulls GPS validation results from EUDR-007 GPS Coordinate Validator Agent (coordinate quality scores)
- [ ] Pulls multi-tier supplier data from EUDR-008 Multi-Tier Supplier Tracker (supplier hierarchy, tier depth, onboarding status)
- [ ] Pulls chain of custody records from EUDR-009 Chain of Custody Agent (custody model, transfer records)
- [ ] Pulls segregation verification from EUDR-010 Segregation Verifier Agent (segregation status, verification results)
- [ ] Pulls mass balance calculations from EUDR-011 Mass Balance Calculator Agent (input/output balance, mass balance credits)
- [ ] Pulls document authentication results from EUDR-012 Document Authentication Agent (certificate verification, authenticity scores)
- [ ] Pulls blockchain provenance records from EUDR-013 Blockchain Integration Agent (blockchain hashes, transaction references)
- [ ] Pulls QR code references from EUDR-014 QR Code Generator Agent (QR codes linked to products/batches)
- [ ] Pulls mobile data collection results from EUDR-015 Mobile Data Collector Agent (field verification data, GPS captures)
- [ ] Compiles supply chain summary: total tiers mapped, total nodes, total plots, traceability completeness score (0-100)
- [ ] Generates supply chain traceability evidence list for evidence package inclusion
- [ ] Flags supply chain gaps (missing tiers, unverified actors, broken custody chains) with remediation guidance
- [ ] Cross-validates supply chain data consistency (quantities match across tiers, dates are sequential, locations are plausible)

**Non-Functional Requirements:**
- Completeness: 100% of available traceability data from all 15 agents included
- Performance: Supply chain compilation < 15 seconds for supply chains with 1,000+ nodes
- Consistency: Cross-validation catches 95%+ of data inconsistencies across agents
- Auditability: Agent-level attribution for every data element

**Dependencies:**
- EUDR-001 through EUDR-015 (all 15 Supply Chain Traceability agents)

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 integration engineer)

**Edge Cases:**
- Supply chain has gaps (missing Tier 3 intermediary) -- Compile available data; flag gap in traceability score; include in compliance report
- EUDR-013 blockchain records unavailable (operator does not use blockchain) -- Skip blockchain section; note as not applicable
- Mobile data collection (EUDR-015) has newer coordinates than EUDR-007 validation -- Use most recent validated data; flag discrepancy
- Supply chain has 10,000+ nodes (very complex palm oil chain) -- Paginate data retrieval; summarize in DDS; full details in evidence package

---

#### Feature 5: Commodity Traceability Aggregator

**User Story:**
```
As a compliance officer,
I want commodity-specific traceability data correctly aggregated with the right product classifications, scientific names, and HS/CN codes,
So that each DDS contains accurate commodity-specific information required by Article 9 and Annex I.
```

**Acceptance Criteria:**
- [ ] Handles all 7 EUDR-regulated commodities: cattle, cocoa, coffee, palm oil, rubber, soya, wood
- [ ] Maps each commodity to its correct HS/CN code range per EUDR Annex I product list
- [ ] Includes correct scientific species names per commodity: Bos taurus/indicus (cattle), Theobroma cacao (cocoa), Coffea arabica/canephora (coffee), Elaeis guineensis (palm oil), Hevea brasiliensis (rubber), Glycine max (soya), species-specific for wood (hundreds of species)
- [ ] Handles derived product mapping: chocolate products from cocoa (CN 1806), refined oil from palm oil (CN 1511), leather from cattle (CN 4101-4104), furniture from wood (CN 9401-9403), tires from rubber (CN 4011-4013)
- [ ] Applies commodity-specific supply chain archetype logic:
  - Cattle: ranch -> feedlot -> slaughterhouse -> packer -> trader -> importer (animal movement tracking)
  - Cocoa: smallholder -> cooperative -> collector -> processor -> trader -> importer (cooperative aggregation)
  - Coffee: farm -> wet mill -> dry mill -> exporter -> trader -> importer (altitude/origin segregation)
  - Palm oil: plantation -> mill -> refinery -> trader -> importer (RSPO mass balance)
  - Rubber: smallholder -> collector -> processor -> trader -> importer (latex aggregation)
  - Soya: farm -> silo -> crusher -> trader -> importer (volume co-mingling)
  - Wood: forest -> sawmill -> veneer/plywood -> furniture -> trader -> importer (multi-step processing, species mixing)
- [ ] Validates commodity-product consistency: CN code matches commodity type, scientific name matches commodity
- [ ] Handles multi-commodity products (e.g., chocolate with cocoa + soya lecithin + palm oil) by generating separate commodity sections per DDS or multiple linked DDS documents
- [ ] Handles commodity-specific certification mapping: FSC/PEFC (wood), RSPO (palm oil), Rainforest Alliance/UTZ (cocoa, coffee), RTRS (soya)
- [ ] Generates commodity-specific traceability summary with archetype visualization

**Non-Functional Requirements:**
- Accuracy: 100% correct HS/CN code mapping for all 7 commodities and their derived products
- Completeness: All commodity-specific Annex I products covered
- Performance: Commodity aggregation < 5 seconds per commodity per DDS

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master (supply chain topology per commodity)
- EUDR-009 Chain of Custody Agent (commodity-specific custody models)
- EUDR-011 Mass Balance Calculator (commodity-specific mass balance)
- Reference data: Annex I product list, HS/CN code database, scientific name database

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Wood product made from multiple species (tropical hardwood mix) -- List all species with proportions if available
- Palm oil in a product below de minimis threshold -- Include in DDS if EUDR-regulated product; exclude if exempt
- Product cannot be classified under a single commodity (palm kernel oil vs. palm oil) -- Apply correct CN code per product type
- Unknown species (wood imported without species declaration) -- Flag as critical data gap; block DDS assembly

---

**P0 Features 6-7: Validation and Packaging Layer**

> Features 6 and 7 are P0 launch blockers. Without compliance validation and evidence packaging, assembled DDS documents cannot be verified for completeness or accompanied by the supporting evidence required for EU IS submission and competent authority inspections. These features ensure that every DDS leaving this agent is submission-ready.

---

#### Feature 6: Compliance Validator

**User Story:**
```
As a compliance officer,
I want every assembled DDS validated against the complete set of Article 4 and Annex II requirements before it proceeds to packaging and submission,
So that I can identify and resolve all compliance gaps before the DDS reaches the EU Information System.
```

**Acceptance Criteria:**
- [ ] Validates all 15 Annex II mandatory fields are present and non-empty
- [ ] Validates operator identification: name is non-empty, address is structured, EORI format matches pattern (2-letter country code + up to 15 digits)
- [ ] Validates registration number: cross-references with EUDR-036 OperatorRegistrationManager to confirm valid, non-expired registration in target Member State
- [ ] Validates product description: trade name non-empty, scientific name matches commodity type, common name present
- [ ] Validates HS/CN codes: matches valid code from EUDR Annex I product list; code is consistent with declared commodity
- [ ] Validates quantity: positive numeric value with valid unit (KGM for kg, MTQ for m3, NAR for items)
- [ ] Validates country of production: valid ISO 3166-1 alpha-2 codes; consistent with geolocation coordinates (country of coordinate matches declared country)
- [ ] Validates geolocation: all coordinates in WGS 84 with 6+ decimal places; plots > 4 ha have polygon data; polygons are closed with correct winding order
- [ ] Validates production dates: start date before end date; within reasonable range (not future, not before EUDR cutoff date where relevant)
- [ ] Validates supplier information: name, postal address, and email all present and non-empty; email format valid
- [ ] Validates buyer information: name, postal address, and email all present and non-empty; email format valid
- [ ] Validates deforestation-free evidence: at least one evidence reference present; evidence IDs resolvable to upstream agent outputs
- [ ] Validates legal production evidence: at least one evidence reference present; evidence IDs resolvable
- [ ] Validates risk assessment conclusion: all Article 10(2) criteria evaluated; risk level classification present; country benchmarking documented
- [ ] Validates mitigation measures: if risk was non-negligible, mitigation section present with measures mapped to Article 11(2) categories; if risk was negligible, mitigation section may be absent with documented rationale
- [ ] Validates compliance conclusion consistency: conclusion of "compliant" is only possible if risk is negligible (directly or after mitigation); "non-compliant" conclusion blocks submission
- [ ] Validates cross-field consistency: quantities match mass balance data; countries match geolocation; supply chain depth matches declared tiers
- [ ] Generates comprehensive validation report with: pass/fail per requirement, severity classification (CRITICAL/HIGH/MEDIUM/LOW), remediation guidance for each failure
- [ ] Calculates overall DDS readiness score (0-100) with breakdown by validation category
- [ ] Supports configurable validation strictness: STRICT (production, blocks on any failure), STANDARD (allows warnings), LENIENT (development, flags but does not block)
- [ ] Records validation history for each DDS: when validated, what passed, what failed, what was remediated

**Non-Functional Requirements:**
- Performance: Validation completes in < 5 seconds for single DDS
- Accuracy: Zero false negatives (never passes an invalid DDS as valid)
- Coverage: 100% of Annex II mandatory fields validated; 100% of Article 4 requirements checked
- Determinism: Same DDS input produces identical validation result

**Dependencies:**
- EU IS JSON schema specification (structural validation)
- EUDR Annex I product list (HS/CN code validation)
- ISO 3166-1 country code reference (country validation)
- EUDR-036 OperatorRegistrationManager (registration validation)
- Feature 1 DDSDocumentAssembler (assembled DDS for validation)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- DDS for product with zero deforestation risk (plantation established before Dec 31, 2020 cutoff) -- Validate that temporal evidence supports the claim
- DDS with mitigation but residual risk still above negligible -- Compliance conclusion must be "non-compliant"; validation blocks submission with guidance
- Geolocation country does not match declared country (GPS in Cameroon but country declared as Ghana) -- Critical validation failure; block with location mismatch error
- Multiple products in single DDS with different risk levels -- Apply highest risk level to overall DDS conclusion
- Simplified DDS (Article 13) validation -- Verify ALL plots in low-risk countries AND no substantiated concerns

---

#### Feature 7: Document Packager

**User Story:**
```
As a compliance officer,
I want the validated DDS automatically bundled with all supporting evidence, certificates, and documentation into a structured submission package,
So that I can submit a complete evidence package to the EU Information System and provide audit-ready documentation to competent authorities.
```

**Acceptance Criteria:**
- [ ] Includes the validated DDS document (JSON and PDF formats) as the primary document
- [ ] Attaches risk assessment documentation from RiskAssessmentIntegrator (Feature 3) including detailed criterion evaluations
- [ ] Attaches mitigation documentation from EUDR-029 (when applicable) including measure descriptions and effectiveness evidence
- [ ] Attaches supply chain maps from EUDR-001 (graph exports, Sankey diagrams)
- [ ] Attaches certificate copies from EUDR-012 Document Authentication: FSC, RSPO, Rainforest Alliance, UTZ, PEFC, and other relevant certifications
- [ ] Attaches satellite verification evidence from EUDR-003/004/005 (satellite analysis results, NDVI calculations, land use change reports)
- [ ] Attaches geolocation validation evidence from EUDR-002/006/007 (coordinate verification, polygon validation, GPS quality reports)
- [ ] Attaches chain of custody records from EUDR-009 (custody transfer documentation)
- [ ] Attaches blockchain provenance records from EUDR-013 (if applicable)
- [ ] Attaches third-party audit reports from EUDR-024 (if applicable)
- [ ] Generates package manifest listing all included documents with: document ID, document type, file size, MIME type, SHA-256 hash
- [ ] Generates table of contents with section numbering and document references
- [ ] Generates cross-reference index linking DDS fields to supporting evidence sections
- [ ] Generates evidence summary listing total artifacts, total size, evidence categories
- [ ] Compresses package using gzip compression to meet EU IS size requirements
- [ ] Splits packages exceeding EU IS size limit (10 MB) into appropriately sized segments with segment manifests
- [ ] Validates package completeness: every evidence reference in the DDS has a corresponding document in the package
- [ ] Generates package-level SHA-256 integrity hash covering all included documents
- [ ] Supports ZIP archive output for auditor distribution (organized folder structure)
- [ ] Supports PDF compilation output (single PDF with all documents concatenated with table of contents)

**Non-Functional Requirements:**
- Completeness: 100% of referenced evidence included; zero missing references
- Performance: Package assembly < 60 seconds for packages up to 100 MB uncompressed
- Compression: >= 40% size reduction on average for text-heavy packages
- Integrity: SHA-256 hash on every artifact; package-level integrity hash; verifiable chain
- Size: Handles packages up to 500 MB (complex supply chains with satellite imagery)

**Dependencies:**
- Feature 1 DDSDocumentAssembler (validated DDS document)
- Feature 3 RiskAssessmentIntegrator (risk documentation)
- Feature 4 SupplyChainDataCompiler (supply chain evidence list)
- EUDR-001 through EUDR-015 (evidence artifacts)
- EUDR-024 Third-Party Audit Manager (audit reports)
- EUDR-029 Mitigation Measure Designer (mitigation documentation)
- S3 Object Storage for package assembly and staging

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Package exceeds 500 MB (complex supply chain with many satellite images) -- Segment into parts; generate multi-part manifest
- Evidence document in unsupported format (DOCX, BMP) -- Convert to PDF/PNG or reject with format guidance
- Same certificate referenced by multiple DDS in a batch -- Deduplicate within batch; include once; reference in all manifests
- Evidence artifact no longer available from upstream agent (deleted or archived) -- Flag missing evidence; include placeholder with explanation
- Auditor requests package years after submission (within 5-year retention) -- Reconstruct from versioned storage

---

**P0 Features 8-9: Signature and Lifecycle Management Layer**

> Features 8 and 9 are P0 launch blockers. Without digital signature preparation, DDS documents lack the legal validity and non-repudiation required for regulatory acceptance. Without version control, operators cannot manage DDS amendments, corrections, and withdrawals required by the ongoing compliance lifecycle. These features complete the DDS creation pipeline from assembly to submission-ready status.

---

#### Feature 8: Digital Signature Preparer

**User Story:**
```
As a compliance officer,
I want my DDS documents prepared for qualified electronic signature per eIDAS standards,
So that my submitted DDS has legal validity and non-repudiation equivalent to a handwritten signature.
```

**Acceptance Criteria:**
- [ ] Generates signature-ready document hash (SHA-256) from the finalized, validated DDS content
- [ ] Creates signature metadata block containing: signing time (UTC), signer identity (operator name, role), certificate reference (X.509 certificate serial number), signature purpose ("EUDR Due Diligence Statement"), regulatory reference ("Article 4, Regulation (EU) 2023/1115")
- [ ] Formats DDS for PAdES signature (PDF Advanced Electronic Signatures per ETSI EN 319 142) when DDS is in PDF format
- [ ] Formats DDS for XAdES signature (XML Advanced Electronic Signatures per ETSI EN 319 132) when DDS is in XML format
- [ ] Formats DDS for JAdES signature (JSON Advanced Electronic Signatures per ETSI EN 319 182) when DDS is in JSON format
- [ ] Supports signature delegation: compliance officer delegates signing authority to authorized representative
- [ ] Validates signer identity against operator registration records (signer must be authorized for the operator)
- [ ] Validates certificate validity: not expired, not revoked, issued by trusted CA
- [ ] Verifies document integrity after signature application: signed document hash matches pre-signature hash
- [ ] Records signature provenance in audit trail: who signed, when, with which certificate, hash before and after
- [ ] Supports countersignature: second authorized signer can add countersignature for dual-authorization workflows
- [ ] Generates signature verification report: confirms signature validity, signer identity, signing time, certificate chain
- [ ] Handles signature format negotiation with EU IS (adapts to EU IS signature requirements)

**Non-Functional Requirements:**
- Security: Private keys never exposed outside of secure enclave (SEC-006 Vault integration)
- Integrity: 100% of signed documents pass post-signature integrity verification
- Compliance: Signature formats comply with eIDAS qualified electronic signature requirements
- Performance: Signature preparation < 5 seconds per DDS document
- Auditability: Complete signature provenance chain with certificate references

**Dependencies:**
- SEC-003 Encryption at Rest (document encryption during signature process)
- SEC-006 Secrets Management (certificate and private key storage in Vault)
- eIDAS trust service provider integration (qualified certificate issuance)
- EUDR-036 API Integration Manager (eIDAS credential management)

**Estimated Effort:** 3 weeks (1 backend engineer with security/cryptography expertise)

**Edge Cases:**
- Signing certificate expires between DDS creation and submission -- Alert; require re-signing with valid certificate
- Operator uses multiple signing certificates (different authorized signers) -- Track which certificate signed which DDS
- EU IS requires specific signature format not yet supported -- Extensible signature format adapter pattern
- DDS amended after signing -- Previous signature invalidated; require re-signing of amended version
- Countersignature timeout (second signer does not sign within deadline) -- Configurable timeout with escalation

---

#### Feature 9: Version Controller

**User Story:**
```
As a compliance officer,
I want complete version control on all DDS documents with amendment tracking, withdrawal management, and 5-year retention,
So that I can manage the DDS lifecycle, demonstrate version provenance to auditors, and comply with Article 31 retention requirements.
```

**Acceptance Criteria:**
- [ ] Assigns version numbers to all DDS documents using semantic versioning (major.minor.patch): major = new DDS, minor = amendment, patch = correction
- [ ] Tracks DDS state machine: DRAFT -> ASSEMBLED -> VALIDATED -> PACKAGED -> SIGNED -> SUBMITTED -> ACKNOWLEDGED -> AMENDED -> WITHDRAWN
- [ ] Records every state transition with: actor (user or system), timestamp (UTC), reason code, change description, SHA-256 hash of document at transition
- [ ] Supports DDS amendment workflow: create amended version from existing DDS, track changes between versions, link amendment to original with reason code
- [ ] Supports amendment reason codes: NEW_INFORMATION (new data from upstream agents), ERROR_CORRECTION (fix data errors), REGULATORY_UPDATE (respond to regulatory change), RISK_CHANGE (risk profile changed), SCOPE_CHANGE (product or supply chain changed)
- [ ] Supports DDS withdrawal workflow: mark submitted DDS as withdrawn with reason code, record withdrawal timestamp, link to replacement DDS if applicable
- [ ] Supports withdrawal reason codes: PRODUCT_NOT_PLACED (product not placed on EU market), ERROR_IN_DDS (fundamental error requiring replacement), DUPLICATE_SUBMISSION (accidental duplicate), REGULATORY_DIRECTION (withdrawn at competent authority direction)
- [ ] Maintains immutable audit trail: no state transitions can be deleted or modified after recording
- [ ] Enforces 5-year retention from DDS submission date per Article 31
- [ ] Generates retention alerts: 90 days, 60 days, 30 days, 7 days before documents exit retention period
- [ ] Prevents deletion of documents within retention period (database-level hard lock)
- [ ] Generates version comparison reports: diff between any two versions showing field-level changes
- [ ] Tracks DDS lineage: which DDS replaced which, amendment chains, withdrawal-replacement links
- [ ] Records EU IS reference numbers (received from EUDR-036) alongside GreenLang DDS references
- [ ] Archives documents past retention period to cold storage (S3) with configurable archive policy
- [ ] Supports bulk version queries: list all DDS versions for an operator, commodity, or time period

**Non-Functional Requirements:**
- Integrity: Every version change produces a new SHA-256 hash; hash chain is verifiable
- Immutability: Audit trail entries are append-only; zero modification or deletion within retention period
- Retention: 100% of documents within retention period retrievable in < 5 seconds
- Scale: Support 500,000+ DDS document versions across all operators
- Performance: Version lookup < 100ms; version comparison < 5 seconds

**Dependencies:**
- PostgreSQL for version metadata and state tracking
- S3 Object Storage for document content storage and archival
- TimescaleDB hypertable for version audit log (time-series optimized)
- SEC-005 Centralized Audit Logging (platform audit integration)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Amendment to already-amended DDS (chain of 3+ versions) -- Track full amendment chain; latest version is authoritative
- Withdrawal of DDS that has already been used for market placement -- Record withdrawal with note that product was already placed; regulatory implications flagged
- Retention period extended by regulatory update (5 years to 7 years) -- Configuration-driven retention; update without data migration
- Concurrent amendments by different users -- Optimistic locking; reject second amendment with version conflict error
- DDS acknowledged by EU IS then subsequently withdrawn -- Track post-acknowledgement withdrawal as distinct from pre-submission withdrawal

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Multi-Language DDS Output
- Generate DDS in EU official languages beyond English
- Start with EN, FR, DE, ES, PT, IT, NL (top 7 by operator population)
- Translation of standardized regulatory text sections and field labels (not free-form evidence content)
- Language selection per DDS or per operator preference
- Compliance declaration text in the language of the target Member State

#### Feature 11: DDS Analytics Dashboard
- Statistics on DDS creation volume over time (daily, weekly, monthly, quarterly)
- Validation pass/fail rate trends with most common failure categories
- Average assembly time trends by commodity and supply chain complexity
- Risk profile distribution across DDS portfolio
- Geolocation quality metrics (precision, CRS transformations, polygon corrections)
- Commodity breakdown: DDS count, plot count, and evidence volume per commodity

#### Feature 12: Automated DDS Refresh and Re-Assembly
- Monitor upstream agent data changes that may affect assembled DDS accuracy
- Alert when assembled DDS content may be outdated (e.g., new risk score, new satellite alert)
- Suggest DDS amendment when material changes detected in upstream agent data
- Auto-generate amendment DDS with change summary and provenance chain to new data
- Configurable refresh triggers: new risk score above threshold, new deforestation alert, supplier status change

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Direct EU Information System transmission (EUDR-036 handles API submission; this agent prepares the DDS)
- Operator registration management (EUDR-036 handles registration lifecycle)
- Raw DDS content generation from scratch (EUDR-030 handles initial document content generation)
- Due diligence workflow orchestration (EUDR-026 handles workflow management)
- Risk assessment calculation (EUDR-028 handles scoring; this agent integrates scores)
- Mitigation measure design (EUDR-029 handles mitigation; this agent documents it)
- Satellite data acquisition or analysis (EUDR-003/004/005 handle satellite operations)
- Carbon footprint or GHG reporting (GL-GHG-APP handles this)
- Mobile-native DDS creation app (web responsive design only for v1.0)
- AI-generated DDS content (all content is deterministic, template-based, zero-LLM)
- Blockchain-based DDS immutability (SHA-256 hash chains provide sufficient integrity)
- Financial impact analysis of DDS rejections (defer to analytics platform)

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
| AGENT-EUDR-037        |           | AGENT-EUDR-030            |           | AGENT-EUDR-036        |
| Due Diligence Statement|<-------->| Documentation             |---------->| EU Information System |
| Creator               |           | Generator                 |           | Interface             |
|                       |           |                           |           |                       |
| - DDSDocumentAssembler|           | - DDSStatementGenerator   |           | - DDSSubmissionGW     |
| - GeolocationFormatter|           | - Article9DataAssembler   |           | - OperatorRegMgr      |
| - RiskAssessIntegrator|           | - RiskAssessDocumenter    |           | - GeoDataFormatter    |
| - SupplyChainCompiler |           | - MitigationDocumenter    |           | - DocPkgAssembler     |
| - CommodityAggregator |           | - CompliancePackageBlder  |           | - StatusTracker       |
| - ComplianceValidator |           | - DocumentVersionMgr      |           | - APIIntegrationMgr   |
| - DocumentPackager    |           | - SubmissionEngine        |           | - AuditTrailRecorder  |
| - DigitalSigPreparer  |           +---------------------------+           +-----------------------+
| - VersionController   |
+-----------+-----------+
            |
            +--- Reads from ALL upstream EUDR agents (001-036) ---+
            |                                                      |
+-----------v-----------+   +-----------v-----------+   +-----------v-----------+
| EUDR-001 thru 007     |   | EUDR-008 thru 015     |   | EUDR-016 thru 029     |
| Supply Chain +        |   | Supply Chain           |   | Risk Assessment +     |
| Geolocation Agents    |   | Traceability Agents    |   | Due Diligence Agents  |
+-----------------------+   +-----------------------+   +-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/due_diligence_statement_creator/
    __init__.py                          # Package exports (100+ symbols)
    config.py                            # DDSCreatorConfig with GL_EUDR_DDSC_ env prefix
    models.py                            # Pydantic v2 models for DDS, packages, versions, geolocations
    provenance.py                        # SHA-256 hash chains for document and version integrity
    metrics.py                           # Prometheus metrics (18 metrics, gl_eudr_ddsc_ prefix)
    dds_document_assembler.py            # Engine 1: DDS assembly from all upstream agents
    geolocation_formatter.py             # Engine 2: Article 9 geolocation formatting
    risk_assessment_integrator.py        # Engine 3: Risk score integration from 10 agents
    supply_chain_data_compiler.py        # Engine 4: Traceability data compilation from 15 agents
    commodity_traceability_aggregator.py # Engine 5: Commodity-specific data aggregation
    compliance_validator.py              # Engine 6: Article 4 and Annex II validation
    document_packager.py                 # Engine 7: Evidence packaging and bundling
    digital_signature_preparer.py        # Engine 8 (within Engine 7 scope): eIDAS signature preparation
    version_controller.py               # Engine 9 (within lifecycle scope): DDS version management
    setup.py                             # DDSCreatorService facade
    reference_data/
        __init__.py
        eu_dds_schema.py                 # EU IS DDS JSON/XML schema definitions (versioned)
        annex_i_products.py              # EUDR Annex I product list with HS/CN codes
        annex_ii_fields.py               # EUDR Annex II DDS field definitions and validation rules
        commodity_species.py             # Scientific species names per commodity
        hs_cn_codes.py                   # HS/CN code reference database
        country_codes.py                 # ISO 3166 country codes with Article 29 classifications
        dds_templates.py                 # DDS templates (standard, simplified)
        validation_rules.py              # Comprehensive validation rule definitions
    api/
        __init__.py
        router.py                        # FastAPI router (15+ endpoints)
        schemas.py                       # API request/response Pydantic models
        dependencies.py                  # FastAPI dependencies (auth, db, services)
        assembly_routes.py               # DDS assembly endpoints
        geolocation_routes.py            # Geolocation formatting endpoints
        risk_routes.py                   # Risk integration endpoints
        supply_chain_routes.py           # Supply chain compilation endpoints
        validation_routes.py             # Compliance validation endpoints
        package_routes.py                # Document packaging endpoints
        signature_routes.py              # Digital signature preparation endpoints
        version_routes.py                # Version management endpoints
```

### 7.3 Data Models (Key Entities)

```python
# ============================================================
# Enumerations
# ============================================================

class DDSDocumentStatus(str, Enum):
    DRAFT = "draft"                      # Initial creation / assembly in progress
    ASSEMBLED = "assembled"              # All data pulled from upstream agents
    VALIDATED = "validated"              # Passed compliance validation
    PACKAGED = "packaged"                # Evidence package assembled
    SIGNED = "signed"                    # Digital signature applied
    SUBMITTED = "submitted"              # Handed to EUDR-036 for EU IS submission
    ACKNOWLEDGED = "acknowledged"        # EU IS accepted and assigned reference number
    AMENDED = "amended"                  # Amended version created
    WITHDRAWN = "withdrawn"              # Withdrawn by operator

class DDSType(str, Enum):
    STANDARD = "standard"                # Standard DDS per Articles 4-11
    SIMPLIFIED = "simplified"            # Simplified DDS per Article 13

class EUDRCommodity(str, Enum):
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

class RiskClassification(str, Enum):
    NEGLIGIBLE = "negligible"            # Composite risk < 20, no HIGH scores
    LOW = "low"                          # Composite risk < 40, no HIGH scores
    STANDARD = "standard"               # Composite risk < 70 or any score >= 60
    HIGH = "high"                        # Composite risk >= 70 or critical score >= 80

class CoordinateFormat(str, Enum):
    POINT = "point"                      # Single GPS coordinate (plots <= 4 ha)
    POLYGON = "polygon"                  # Polygon boundary (plots > 4 ha)

class AmendmentReason(str, Enum):
    NEW_INFORMATION = "new_information"
    ERROR_CORRECTION = "error_correction"
    REGULATORY_UPDATE = "regulatory_update"
    RISK_CHANGE = "risk_change"
    SCOPE_CHANGE = "scope_change"

class WithdrawalReason(str, Enum):
    PRODUCT_NOT_PLACED = "product_not_placed"
    ERROR_IN_DDS = "error_in_dds"
    DUPLICATE_SUBMISSION = "duplicate_submission"
    REGULATORY_DIRECTION = "regulatory_direction"

class ValidationSeverity(str, Enum):
    CRITICAL = "critical"                # Blocks DDS progression
    HIGH = "high"                        # Should block; configurable
    MEDIUM = "medium"                    # Warning; does not block
    LOW = "low"                          # Informational

class SignatureFormat(str, Enum):
    PADES = "PAdES"                      # PDF Advanced Electronic Signatures
    XADES = "XAdES"                      # XML Advanced Electronic Signatures
    JADES = "JAdES"                      # JSON Advanced Electronic Signatures

# ============================================================
# Core Models
# ============================================================

# Assembled DDS Document
class AssembledDDS(BaseModel):
    dds_id: str                          # GL-DDS-{operator_id}-{commodity}-{YYYYMMDD}-{seq}
    operator_id: str                     # Operator UUID
    operator_name: str                   # Legal name
    operator_address: str                # Registered address
    operator_eori: Optional[str]         # EORI number
    registration_number: str             # Per Article 14
    product_description: str             # Trade name and type
    product_scientific_name: Optional[str]  # Scientific name of species
    product_common_name: Optional[str]   # Common name
    hs_code: str                         # HS code per Annex I
    cn_code: str                         # CN code per Annex I
    quantity: Decimal                    # Net mass/volume/items
    quantity_unit: str                   # KGM, MTQ, NAR
    country_of_production: List[str]     # ISO 3166-1 alpha-2 codes
    geolocation_data: List[FormattedGeolocation]  # Formatted plot coordinates
    production_date_start: date          # Production period start
    production_date_end: date            # Production period end
    supplier_info: SupplierInfo          # Name, address, email
    buyer_info: BuyerInfo               # Name, address, email
    deforestation_free_evidence: List[EvidenceReference]
    legal_production_evidence: List[EvidenceReference]
    risk_assessment: IntegratedRiskAssessment  # From RiskAssessmentIntegrator
    mitigation_summary: Optional[MitigationSummary]  # From EUDR-029 (if applicable)
    compliance_conclusion: ComplianceConclusion
    supply_chain_summary: SupplyChainSummary  # From SupplyChainDataCompiler
    commodity_details: CommodityDetails  # From CommodityTraceabilityAggregator
    dds_type: DDSType                    # STANDARD or SIMPLIFIED
    commodity: EUDRCommodity             # Primary commodity
    status: DDSDocumentStatus
    version: str                         # Semantic version: "1.0.0"
    provenance_hash: str                 # SHA-256 of all content fields
    content_json: Optional[Dict]         # EU IS JSON format
    content_xml: Optional[str]           # EU IS XML format
    assembly_provenance: AssemblyProvenance  # Which agent provided what
    created_at: datetime
    updated_at: datetime

# Formatted Geolocation (per plot)
class FormattedGeolocation(BaseModel):
    plot_id: str                         # Reference to upstream plot ID
    country_code: str                    # ISO 3166-1 alpha-2
    coordinate_format: CoordinateFormat  # POINT or POLYGON
    coordinates: Union[List[float], List[List[float]]]
    source_crs: str                      # Source CRS before transformation
    target_crs: str = "EPSG:4326"       # Always WGS 84
    precision_decimal_places: int = 6    # Minimum 6 per Article 9
    area_hectares: Optional[float]       # Plot area
    polygon_closed: bool = True          # Polygon closure validated
    winding_order: str = "counter_clockwise"
    transformation_applied: bool = False
    validation_passed: bool = True
    quality_score: float                 # 0-100

# Integrated Risk Assessment
class IntegratedRiskAssessment(BaseModel):
    composite_risk_score: float          # 0-100
    risk_classification: RiskClassification
    country_risk: RiskComponentScore     # From EUDR-016
    supplier_risk: RiskComponentScore    # From EUDR-017
    commodity_risk: RiskComponentScore   # From EUDR-018
    corruption_risk: RiskComponentScore  # From EUDR-019
    deforestation_risk: RiskComponentScore  # From EUDR-020
    indigenous_rights_risk: RiskComponentScore  # From EUDR-021
    protected_area_risk: RiskComponentScore  # From EUDR-022
    legal_compliance_risk: RiskComponentScore  # From EUDR-023
    audit_risk: RiskComponentScore       # From EUDR-024
    mitigation_effectiveness: RiskComponentScore  # From EUDR-025
    article_10_2_criteria: Dict[str, CriterionEvaluation]  # a through g
    country_benchmarking: CountryBenchmarking  # Article 29
    simplified_dd_eligible: bool         # Article 13 eligibility
    enhanced_dd_required: bool           # Enhanced due diligence triggered
    integration_provenance: Dict[str, AgentDataReference]  # Agent attribution

# Compliance Validation Result
class ValidationResult(BaseModel):
    validation_id: str
    dds_id: str
    is_valid: bool
    readiness_score: float               # 0-100
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    results: List[ValidationCheckResult]
    validated_at: datetime

# DDS Version Record
class DDSVersion(BaseModel):
    version_id: str
    dds_id: str
    version_number: str                  # Semantic version
    status: DDSDocumentStatus
    content_hash: str                    # SHA-256 of document at this version
    previous_version_id: Optional[str]
    amendment_reason: Optional[AmendmentReason]
    withdrawal_reason: Optional[WithdrawalReason]
    change_description: str
    changed_by: str
    eu_is_reference: Optional[str]       # EU IS assigned reference
    retention_expiry: date               # 5 years from submission
    is_retained: bool = True
    changed_at: datetime

# Evidence Package
class EvidencePackage(BaseModel):
    package_id: str
    dds_id: str
    operator_id: str
    dds_document: Dict                   # DDS JSON
    dds_pdf: Optional[str]               # DDS PDF storage path
    risk_assessment_doc: Dict
    mitigation_doc: Optional[Dict]
    supply_chain_maps: List[Dict]
    certificates: List[CertificateReference]
    satellite_evidence: List[SatelliteEvidenceRef]
    geolocation_evidence: List[GeolocationEvidenceRef]
    chain_of_custody_records: List[CustodyRecord]
    blockchain_records: List[BlockchainRecord]
    audit_reports: List[AuditReportRef]
    manifest: PackageManifest
    table_of_contents: TableOfContents
    cross_reference_index: CrossReferenceIndex
    total_artifacts: int
    total_size_bytes: int
    compressed_size_bytes: int
    package_hash: str                    # SHA-256 of entire package
    assembled_at: datetime
```

### 7.4 Database Schema (New Migration: V125)

```sql
CREATE SCHEMA IF NOT EXISTS gl_eudr_ddsc;

-- ============================================================
-- Table 1: Assembled DDS Documents
-- ============================================================
CREATE TABLE gl_eudr_ddsc.assembled_dds (
    dds_id VARCHAR(150) PRIMARY KEY,
    operator_id UUID NOT NULL,
    operator_name VARCHAR(500) NOT NULL,
    operator_address TEXT,
    operator_eori VARCHAR(50),
    registration_number VARCHAR(100),
    product_description TEXT NOT NULL,
    product_scientific_name VARCHAR(500),
    product_common_name VARCHAR(500),
    hs_code VARCHAR(20) NOT NULL,
    cn_code VARCHAR(20) NOT NULL,
    quantity NUMERIC(18,4) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL DEFAULT 'KGM',
    country_of_production JSONB NOT NULL DEFAULT '[]',
    geolocation_data JSONB NOT NULL DEFAULT '[]',
    production_date_start DATE,
    production_date_end DATE,
    supplier_info JSONB NOT NULL DEFAULT '{}',
    buyer_info JSONB NOT NULL DEFAULT '{}',
    deforestation_free_evidence JSONB DEFAULT '[]',
    legal_production_evidence JSONB DEFAULT '[]',
    risk_assessment JSONB NOT NULL DEFAULT '{}',
    mitigation_summary JSONB,
    compliance_conclusion JSONB NOT NULL DEFAULT '{}',
    supply_chain_summary JSONB DEFAULT '{}',
    commodity_details JSONB DEFAULT '{}',
    dds_type VARCHAR(20) NOT NULL DEFAULT 'standard',
    commodity VARCHAR(50) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'draft',
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    provenance_hash VARCHAR(64) NOT NULL,
    content_json JSONB,
    content_xml TEXT,
    assembly_provenance JSONB DEFAULT '{}',
    eu_is_reference VARCHAR(200),
    submitted_at TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    CONSTRAINT fk_ddsc_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

CREATE INDEX idx_ddsc_dds_operator ON gl_eudr_ddsc.assembled_dds(operator_id);
CREATE INDEX idx_ddsc_dds_status ON gl_eudr_ddsc.assembled_dds(status);
CREATE INDEX idx_ddsc_dds_commodity ON gl_eudr_ddsc.assembled_dds(commodity);
CREATE INDEX idx_ddsc_dds_type ON gl_eudr_ddsc.assembled_dds(dds_type);
CREATE INDEX idx_ddsc_dds_submitted ON gl_eudr_ddsc.assembled_dds(submitted_at);
CREATE INDEX idx_ddsc_dds_country ON gl_eudr_ddsc.assembled_dds USING GIN (country_of_production);

-- ============================================================
-- Table 2: Formatted Geolocations
-- ============================================================
CREATE TABLE gl_eudr_ddsc.formatted_geolocations (
    geolocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(150) NOT NULL REFERENCES gl_eudr_ddsc.assembled_dds(dds_id),
    plot_id VARCHAR(100) NOT NULL,
    country_code CHAR(2) NOT NULL,
    coordinate_format VARCHAR(10) NOT NULL,
    coordinates JSONB NOT NULL,
    source_crs VARCHAR(30) NOT NULL DEFAULT 'EPSG:4326',
    target_crs VARCHAR(30) NOT NULL DEFAULT 'EPSG:4326',
    precision_decimal_places INTEGER NOT NULL DEFAULT 6,
    area_hectares NUMERIC(12,4),
    polygon_closed BOOLEAN NOT NULL DEFAULT TRUE,
    winding_order VARCHAR(20) DEFAULT 'counter_clockwise',
    transformation_applied BOOLEAN NOT NULL DEFAULT FALSE,
    validation_passed BOOLEAN NOT NULL DEFAULT TRUE,
    quality_score NUMERIC(5,2) DEFAULT 100.0,
    formatted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ddsc_geo_dds ON gl_eudr_ddsc.formatted_geolocations(dds_id);
CREATE INDEX idx_ddsc_geo_plot ON gl_eudr_ddsc.formatted_geolocations(plot_id);
CREATE INDEX idx_ddsc_geo_country ON gl_eudr_ddsc.formatted_geolocations(country_code);

-- ============================================================
-- Table 3: Integrated Risk Assessments
-- ============================================================
CREATE TABLE gl_eudr_ddsc.integrated_risk_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(150) NOT NULL REFERENCES gl_eudr_ddsc.assembled_dds(dds_id),
    composite_risk_score NUMERIC(5,2) NOT NULL,
    risk_classification VARCHAR(20) NOT NULL,
    country_risk JSONB NOT NULL DEFAULT '{}',
    supplier_risk JSONB NOT NULL DEFAULT '{}',
    commodity_risk JSONB NOT NULL DEFAULT '{}',
    corruption_risk JSONB NOT NULL DEFAULT '{}',
    deforestation_risk JSONB NOT NULL DEFAULT '{}',
    indigenous_rights_risk JSONB NOT NULL DEFAULT '{}',
    protected_area_risk JSONB NOT NULL DEFAULT '{}',
    legal_compliance_risk JSONB NOT NULL DEFAULT '{}',
    audit_risk JSONB NOT NULL DEFAULT '{}',
    mitigation_effectiveness JSONB NOT NULL DEFAULT '{}',
    article_10_2_criteria JSONB NOT NULL DEFAULT '{}',
    country_benchmarking JSONB DEFAULT '{}',
    simplified_dd_eligible BOOLEAN DEFAULT FALSE,
    enhanced_dd_required BOOLEAN DEFAULT FALSE,
    integration_provenance JSONB DEFAULT '{}',
    integrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ddsc_risk_dds ON gl_eudr_ddsc.integrated_risk_assessments(dds_id);
CREATE INDEX idx_ddsc_risk_classification ON gl_eudr_ddsc.integrated_risk_assessments(risk_classification);

-- ============================================================
-- Table 4: Compliance Validation Results
-- ============================================================
CREATE TABLE gl_eudr_ddsc.validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(150) NOT NULL REFERENCES gl_eudr_ddsc.assembled_dds(dds_id),
    is_valid BOOLEAN NOT NULL,
    readiness_score NUMERIC(5,2) DEFAULT 0.0,
    total_checks INTEGER NOT NULL DEFAULT 0,
    passed_checks INTEGER NOT NULL DEFAULT 0,
    failed_checks INTEGER NOT NULL DEFAULT 0,
    warning_checks INTEGER NOT NULL DEFAULT 0,
    results JSONB NOT NULL DEFAULT '[]',
    validation_mode VARCHAR(20) NOT NULL DEFAULT 'strict',
    validated_by VARCHAR(100),
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ddsc_val_dds ON gl_eudr_ddsc.validation_results(dds_id);
CREATE INDEX idx_ddsc_val_valid ON gl_eudr_ddsc.validation_results(is_valid);

-- ============================================================
-- Table 5: Evidence Packages
-- ============================================================
CREATE TABLE gl_eudr_ddsc.evidence_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(150) NOT NULL REFERENCES gl_eudr_ddsc.assembled_dds(dds_id),
    operator_id UUID NOT NULL,
    manifest JSONB NOT NULL DEFAULT '{}',
    table_of_contents JSONB NOT NULL DEFAULT '{}',
    cross_reference_index JSONB NOT NULL DEFAULT '{}',
    total_artifacts INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    compressed_size_bytes BIGINT DEFAULT 0,
    package_hash VARCHAR(64) NOT NULL,
    storage_path VARCHAR(500),
    pdf_path VARCHAR(500),
    zip_path VARCHAR(500),
    assembled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ddsc_pkg_dds ON gl_eudr_ddsc.evidence_packages(dds_id);
CREATE INDEX idx_ddsc_pkg_operator ON gl_eudr_ddsc.evidence_packages(operator_id);

-- ============================================================
-- Table 6: DDS Versions
-- ============================================================
CREATE TABLE gl_eudr_ddsc.dds_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(150) NOT NULL,
    version_number VARCHAR(20) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'draft',
    content_hash VARCHAR(64) NOT NULL,
    previous_version_id UUID REFERENCES gl_eudr_ddsc.dds_versions(version_id),
    amendment_reason VARCHAR(50),
    withdrawal_reason VARCHAR(50),
    change_description TEXT,
    changed_by VARCHAR(100),
    eu_is_reference VARCHAR(200),
    retention_expiry DATE,
    is_retained BOOLEAN DEFAULT TRUE,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ddsc_ver_dds ON gl_eudr_ddsc.dds_versions(dds_id);
CREATE INDEX idx_ddsc_ver_status ON gl_eudr_ddsc.dds_versions(status);
CREATE INDEX idx_ddsc_ver_retention ON gl_eudr_ddsc.dds_versions(retention_expiry);

-- ============================================================
-- Table 7: Digital Signature Records
-- ============================================================
CREATE TABLE gl_eudr_ddsc.signature_records (
    signature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(150) NOT NULL REFERENCES gl_eudr_ddsc.assembled_dds(dds_id),
    version_id UUID REFERENCES gl_eudr_ddsc.dds_versions(version_id),
    signer_name VARCHAR(200) NOT NULL,
    signer_role VARCHAR(100),
    certificate_serial VARCHAR(200),
    certificate_issuer VARCHAR(500),
    signature_format VARCHAR(10) NOT NULL,
    document_hash_before VARCHAR(64) NOT NULL,
    document_hash_after VARCHAR(64) NOT NULL,
    signature_valid BOOLEAN NOT NULL DEFAULT TRUE,
    is_countersignature BOOLEAN NOT NULL DEFAULT FALSE,
    signed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ddsc_sig_dds ON gl_eudr_ddsc.signature_records(dds_id);

-- ============================================================
-- Table 8: Assembly Audit Log (Hypertable)
-- ============================================================
CREATE TABLE gl_eudr_ddsc.assembly_audit_log (
    log_id UUID DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(200) NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(100),
    details JSONB DEFAULT '{}',
    source_agent VARCHAR(50),
    data_version VARCHAR(50),
    previous_state JSONB,
    new_state JSONB,
    provenance_hash VARCHAR(64),
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_ddsc.assembly_audit_log', 'logged_at');

CREATE INDEX idx_ddsc_audit_entity ON gl_eudr_ddsc.assembly_audit_log(entity_type, entity_id);
CREATE INDEX idx_ddsc_audit_action ON gl_eudr_ddsc.assembly_audit_log(action);
CREATE INDEX idx_ddsc_audit_source ON gl_eudr_ddsc.assembly_audit_log(source_agent);

-- ============================================================
-- Retention and compression policies
-- ============================================================
SELECT add_compression_policy('gl_eudr_ddsc.assembly_audit_log', INTERVAL '90 days');
SELECT add_retention_policy('gl_eudr_ddsc.assembly_audit_log', INTERVAL '7 years');
```

### 7.5 API Endpoints (15+)

| # | Method | Path | Description | Auth Required |
|---|--------|------|-------------|---------------|
| **DDS Assembly** | | | | |
| 1 | POST | `/api/v1/eudr/dds-creator/assemble` | Assemble a new DDS from upstream agent data | Yes (eudr-ddsc:dds:assemble) |
| 2 | GET | `/api/v1/eudr/dds-creator/dds/{dds_id}` | Get assembled DDS document details | Yes (eudr-ddsc:dds:read) |
| 3 | GET | `/api/v1/eudr/dds-creator/dds` | List assembled DDS documents with filters | Yes (eudr-ddsc:dds:read) |
| **Geolocation** | | | | |
| 4 | POST | `/api/v1/eudr/dds-creator/format-geolocation/{dds_id}` | Format geolocation data for a DDS | Yes (eudr-ddsc:geolocation:format) |
| 5 | GET | `/api/v1/eudr/dds-creator/geolocation/{dds_id}` | Get formatted geolocation data | Yes (eudr-ddsc:geolocation:read) |
| **Risk Integration** | | | | |
| 6 | POST | `/api/v1/eudr/dds-creator/integrate-risk/{dds_id}` | Integrate risk scores from all 10 agents | Yes (eudr-ddsc:risk:integrate) |
| 7 | GET | `/api/v1/eudr/dds-creator/risk/{dds_id}` | Get integrated risk assessment | Yes (eudr-ddsc:risk:read) |
| **Validation** | | | | |
| 8 | POST | `/api/v1/eudr/dds-creator/validate/{dds_id}` | Validate DDS against Article 4 and Annex II | Yes (eudr-ddsc:validation:execute) |
| 9 | GET | `/api/v1/eudr/dds-creator/validation/{dds_id}` | Get validation results | Yes (eudr-ddsc:validation:read) |
| **Packaging** | | | | |
| 10 | POST | `/api/v1/eudr/dds-creator/package/{dds_id}` | Build evidence package for DDS | Yes (eudr-ddsc:packages:build) |
| 11 | GET | `/api/v1/eudr/dds-creator/package/{dds_id}` | Get evidence package details and download | Yes (eudr-ddsc:packages:read) |
| **Signature** | | | | |
| 12 | POST | `/api/v1/eudr/dds-creator/prepare-signature/{dds_id}` | Prepare DDS for qualified electronic signature | Yes (eudr-ddsc:signature:prepare) |
| 13 | GET | `/api/v1/eudr/dds-creator/signature/{dds_id}` | Get signature records | Yes (eudr-ddsc:signature:read) |
| **Version Management** | | | | |
| 14 | POST | `/api/v1/eudr/dds-creator/amend/{dds_id}` | Create amended version of DDS | Yes (eudr-ddsc:versions:amend) |
| 15 | POST | `/api/v1/eudr/dds-creator/withdraw/{dds_id}` | Withdraw a submitted DDS | Yes (eudr-ddsc:versions:withdraw) |
| 16 | GET | `/api/v1/eudr/dds-creator/versions/{dds_id}` | Get version history for DDS | Yes (eudr-ddsc:versions:read) |
| **Batch Operations** | | | | |
| 17 | POST | `/api/v1/eudr/dds-creator/batch/assemble` | Batch assemble multiple DDS | Yes (eudr-ddsc:batch:assemble) |
| 18 | GET | `/api/v1/eudr/dds-creator/batch/{batch_id}/status` | Get batch assembly status | Yes (eudr-ddsc:batch:read) |
| **Health** | | | | |
| 19 | GET | `/api/v1/eudr/dds-creator/health` | Service health check | No |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_ddsc_dds_assembled_total` | Counter | DDS documents assembled by commodity |
| 2 | `gl_eudr_ddsc_geolocations_formatted_total` | Counter | Plot geolocations formatted by format type |
| 3 | `gl_eudr_ddsc_risk_integrations_total` | Counter | Risk assessment integrations completed |
| 4 | `gl_eudr_ddsc_supply_chain_compilations_total` | Counter | Supply chain compilations completed |
| 5 | `gl_eudr_ddsc_validations_total` | Counter | Compliance validations by result (pass/fail) |
| 6 | `gl_eudr_ddsc_packages_built_total` | Counter | Evidence packages assembled |
| 7 | `gl_eudr_ddsc_signatures_prepared_total` | Counter | Digital signatures prepared by format |
| 8 | `gl_eudr_ddsc_versions_created_total` | Counter | DDS versions created by type (initial/amendment/withdrawal) |
| 9 | `gl_eudr_ddsc_batch_operations_total` | Counter | Batch assembly operations |
| 10 | `gl_eudr_ddsc_assembly_duration_seconds` | Histogram | DDS assembly latency |
| 11 | `gl_eudr_ddsc_geolocation_duration_seconds` | Histogram | Geolocation formatting latency |
| 12 | `gl_eudr_ddsc_risk_integration_duration_seconds` | Histogram | Risk integration latency |
| 13 | `gl_eudr_ddsc_validation_duration_seconds` | Histogram | Compliance validation latency |
| 14 | `gl_eudr_ddsc_packaging_duration_seconds` | Histogram | Evidence packaging latency |
| 15 | `gl_eudr_ddsc_errors_total` | Counter | Errors by operation type and error category |
| 16 | `gl_eudr_ddsc_active_dds_count` | Gauge | Currently active DDS documents by status |
| 17 | `gl_eudr_ddsc_validation_readiness_avg` | Gauge | Average validation readiness score |
| 18 | `gl_eudr_ddsc_upstream_agent_latency_seconds` | Histogram | Latency pulling data from upstream agents |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Geospatial | pyproj + Shapely | CRS transformations, polygon validation and geometry |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for audit log |
| Cache | Redis | Assembly caching, upstream agent data caching |
| Object Storage | S3 | Evidence packages, DDS documents, PDF storage |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible models |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based DDS creation and management permissions |
| Encryption | AES-256-GCM via SEC-003 | Document encryption, signature key protection |
| Secrets | HashiCorp Vault via SEC-006 | Certificate and private key storage |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent data retrieval |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-ddsc:dds:assemble` | Assemble new DDS from upstream data | Compliance Officer, Admin |
| `eudr-ddsc:dds:read` | View assembled DDS documents | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ddsc:geolocation:format` | Trigger geolocation formatting | Analyst, Compliance Officer, Admin |
| `eudr-ddsc:geolocation:read` | View formatted geolocation data | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ddsc:risk:integrate` | Trigger risk assessment integration | Compliance Officer, Admin |
| `eudr-ddsc:risk:read` | View integrated risk assessments | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ddsc:validation:execute` | Run compliance validation | Analyst, Compliance Officer, Admin |
| `eudr-ddsc:validation:read` | View validation results | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ddsc:packages:build` | Build evidence packages | Compliance Officer, Admin |
| `eudr-ddsc:packages:read` | View and download evidence packages | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ddsc:signature:prepare` | Prepare DDS for digital signature | Compliance Officer, Admin |
| `eudr-ddsc:signature:read` | View signature records | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ddsc:versions:amend` | Create amended DDS versions | Compliance Officer, Admin |
| `eudr-ddsc:versions:withdraw` | Withdraw submitted DDS | Compliance Officer, Admin |
| `eudr-ddsc:versions:read` | View version history | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-ddsc:batch:assemble` | Execute batch DDS assembly | Compliance Officer, Admin |
| `eudr-ddsc:batch:read` | View batch assembly status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ddsc:audit:read` | View assembly audit trail | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| EUDR-001 Supply Chain Mapping Master | Supply chain graph, product descriptions | Nodes, edges, commodities, product info -> DDS supply chain section |
| EUDR-002 Geolocation Verification Agent | Verified plot coordinates | Verified GPS/polygon data -> GeolocationFormatter |
| EUDR-003 Satellite Monitoring Agent | Satellite analysis results | Deforestation-free evidence -> DDS evidence section |
| EUDR-004 Forest Cover Analysis Agent | Forest cover verification | Forest cover status -> DDS evidence section |
| EUDR-005 Land Use Change Detector Agent | Land use change since cutoff | Land use change evidence -> DDS evidence section |
| EUDR-006 Plot Boundary Manager Agent | Registered boundaries | Polygon boundaries -> GeolocationFormatter |
| EUDR-007 GPS Coordinate Validator Agent | Validated GPS coordinates | Coordinate quality data -> GeolocationFormatter |
| EUDR-008 Multi-Tier Supplier Tracker | Supplier hierarchy | Supplier info, tier data -> DDS supplier section |
| EUDR-009 Chain of Custody Agent | Custody model, transfer records | Custody documentation -> DDS supply chain section |
| EUDR-010 Segregation Verifier Agent | Segregation verification | Quantity verification -> DDS quantity section |
| EUDR-011 Mass Balance Calculator Agent | Mass balance data | Quantities, balance credits -> DDS quantity section |
| EUDR-012 Document Authentication Agent | Certificate verification | Authenticated certificates -> Evidence package |
| EUDR-013 Blockchain Integration Agent | Blockchain provenance | Blockchain records -> Evidence package |
| EUDR-014 QR Code Generator Agent | QR code references | QR codes -> Evidence package |
| EUDR-015 Mobile Data Collector Agent | Field verification data | Mobile captures -> Evidence package |
| EUDR-016 Country Risk Evaluator | Country risk scores | Country risk + Article 29 -> RiskAssessmentIntegrator |
| EUDR-017 Supplier Risk Scorer | Supplier risk scores | Supplier risk -> RiskAssessmentIntegrator |
| EUDR-018 Commodity Risk Analyzer | Commodity risk scores | Commodity risk -> RiskAssessmentIntegrator |
| EUDR-019 Corruption Index Monitor | Corruption index data | Corruption risk -> RiskAssessmentIntegrator |
| EUDR-020 Deforestation Alert System | Deforestation alerts | Deforestation risk -> RiskAssessmentIntegrator |
| EUDR-021 Indigenous Rights Checker | Indigenous rights assessment | Indigenous risk -> RiskAssessmentIntegrator |
| EUDR-022 Protected Area Validator | Protected area analysis | Protected area risk -> RiskAssessmentIntegrator |
| EUDR-023 Legal Compliance Verifier | Legal compliance status | Legal risk + evidence -> RiskAssessmentIntegrator + DDS |
| EUDR-024 Third-Party Audit Manager | Audit findings | Audit risk + reports -> RiskAssessmentIntegrator + Evidence |
| EUDR-025 Risk Mitigation Advisor | Mitigation effectiveness | Mitigation risk -> RiskAssessmentIntegrator |
| EUDR-028 Risk Assessment Engine | Composite risk scores | Pre-computed risk (if available) -> RiskAssessmentIntegrator |
| EUDR-029 Mitigation Measure Designer | Mitigation strategies | Mitigation documentation -> DDS mitigation section |
| EUDR-030 Documentation Generator | Raw DDS components | Document templates, content components -> DDSDocumentAssembler |
| EUDR-036 EU IS Interface | Registration numbers, submission status | Registration data -> DDS; submission acknowledgement -> VersionController |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| EUDR-036 EU Information System Interface | DDS submission | Assembled, validated, packaged, signed DDS -> EU IS submission pipeline |
| GL-EUDR-APP v1.0 | Frontend display | DDS data, validation results, version history -> UI dashboards |
| External Auditors | Read-only API + package downloads | Evidence packages, version history -> third-party verification |
| Competent Authorities | On-demand package delivery | Compliance packages -> Article 14-16 inspection response |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Standard DDS Creation (Compliance Officer -- Maria)

```
1. Maria logs in to GL-EUDR-APP
2. Navigates to "DDS Creator" module
3. Clicks "New DDS" -> selects commodity (Cocoa), product (Chocolate bars)
4. System automatically identifies upstream data sources:
   - Supply chain from EUDR-001 (300 plots, 5 tiers, 12 countries)
   - Geolocation from EUDR-002/006/007 (300 plots, 45 polygons)
   - Risk scores from EUDR-016 through EUDR-025 (10 agents)
   - Mitigation from EUDR-029 (3 measures applied)
5. System assembles DDS in < 30 seconds:
   - All 15 Annex II fields populated
   - Geolocation formatted to EU IS specification (WGS 84, 6 decimals)
   - Risk assessment integrated (Article 10, all criteria a-g)
   - Compliance conclusion: COMPLIANT (risk reduced to negligible)
6. System runs compliance validation -> DDS readiness score: 98/100
   - 2 warnings: 2 plots have 5 decimal places (padded to 6)
7. Maria reviews validation report -> approves DDS
8. System packages DDS with evidence (certificates, satellite reports, risk docs)
9. System prepares DDS for digital signature
10. Maria signs DDS with qualified electronic signature
11. DDS delivered to EUDR-036 for EU IS submission
12. EUDR-036 submits; EU IS accepts; reference number stored
```

#### Flow 2: Commodity-Specific DDS for Wood (Supply Chain Analyst -- Lukas)

```
1. Lukas initiates DDS for tropical hardwood furniture shipment
2. CommodityTraceabilityAggregator identifies:
   - Commodity: Wood
   - Species: Tectona grandis (teak), Swietenia macrophylla (mahogany)
   - CN codes: 9403 (wooden furniture), 4407 (sawn wood)
   - Supply chain: Forest (MY) -> Sawmill (MY) -> Furniture factory (VN) -> Trader (NL) -> Importer (DE)
3. SupplyChainDataCompiler pulls data from all 15 traceability agents:
   - EUDR-001: 6-tier supply chain graph
   - EUDR-006: 12 forest concession polygons (all > 4 ha)
   - EUDR-009: Chain of custody (Identity Preserved model)
   - EUDR-012: FSC certification for 8 of 12 concessions
4. GeolocationFormatter formats 12 polygon boundaries:
   - All transformed from UTM Zone 47N to WGS 84
   - All polygons closed and counter-clockwise
5. RiskAssessmentIntegrator pulls from 10 risk agents:
   - Country risk (MY): STANDARD (Article 29)
   - Deforestation risk: LOW (satellite verified deforestation-free)
   - Supplier risk: MEDIUM (4 concessions without FSC)
   - Composite: STANDARD -> Enhanced due diligence required
6. DDS assembled with complete Article 10 documentation
7. Validated: 100/100 readiness score
8. Packaged with FSC certificates, satellite imagery, CoC records
9. Signed and delivered to EUDR-036
```

#### Flow 3: DDS Amendment (Compliance Officer -- Maria)

```
1. Maria receives notification: new deforestation alert on Plot P-2341 in Ghana
2. EUDR-020 Deforestation Alert System flagged the plot
3. Maria opens the affected DDS (GL-DDS-OP123-COCOA-20260301-001 v1.0.0)
4. Clicks "Amend DDS" -> selects reason: RISK_CHANGE
5. RiskAssessmentIntegrator re-pulls scores with updated deforestation data
   - Deforestation risk: LOW -> HIGH for Plot P-2341
   - Composite risk changes from NEGLIGIBLE to STANDARD
6. System re-assembles DDS sections affected by the change:
   - Risk assessment section updated
   - Mitigation section added (EUDR-029 provides new measures)
   - Compliance conclusion updated
7. New version: GL-DDS-OP123-COCOA-20260301-001 v1.1.0
8. Amendment provenance: links v1.0.0 to v1.1.0 with change description
9. Re-validated, re-packaged, re-signed
10. Delivered to EUDR-036 for submission as amendment to original
```

### 8.2 Key Screen Descriptions

**DDS Assembly Dashboard:**
- Assembly form: select operator, commodity, product, supply chain reference
- Progress indicator: data retrieval (from each agent), assembly, formatting, integration
- Real-time assembly status: which agents responded, data freshness indicators
- Preview panel: assembled DDS content with field-level source attribution

**Validation Report View:**
- Summary cards: readiness score (0-100), passed/failed/warning counts
- Grouped check list organized by validation category (Annex II fields, geolocation, risk, mitigation)
- Severity badges (CRITICAL/HIGH/MEDIUM/LOW) with color coding
- Remediation guidance for each failed check
- One-click "Fix" buttons for automatically remediable issues (e.g., geolocation padding)

**Evidence Package View:**
- Table of contents with expandable sections
- Document list with: type, source agent, size, SHA-256 hash, preview link
- Cross-reference view: click any DDS field to see linked evidence documents
- Package integrity status: hash verification result
- Download options: full package (ZIP), individual documents (PDF)

**Version History View:**
- Timeline view of all DDS versions with state transitions
- Version comparison: side-by-side diff between any two versions
- Amendment chain: visual lineage from original through amendments
- Withdrawal records with reason codes
- EU IS reference numbers for submitted versions

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: DDS Document Assembler -- complete assembly from all 36 upstream agents
  - [ ] Feature 2: Geolocation Formatter -- WGS 84, 6 decimals, polygon closure, winding order
  - [ ] Feature 3: Risk Assessment Integrator -- 10 risk agents integrated, Article 10(2) criteria
  - [ ] Feature 4: Supply Chain Data Compiler -- 15 traceability agents compiled
  - [ ] Feature 5: Commodity Traceability Aggregator -- all 7 commodities handled
  - [ ] Feature 6: Compliance Validator -- all Annex II fields, cross-field validation
  - [ ] Feature 7: Document Packager -- evidence bundling, manifest, compression
  - [ ] Feature 8: Digital Signature Preparer -- PAdES/XAdES/JAdES, signature verification
  - [ ] Feature 9: Version Controller -- amendment, withdrawal, 5-year retention
- [ ] >= 85% test coverage achieved (line coverage), >= 90% branch coverage
- [ ] Security audit passed (JWT + RBAC integrated, signature key management verified)
- [ ] Performance targets met:
  - [ ] DDS assembly < 30 seconds p99 for single DDS with 500 plots
  - [ ] Geolocation formatting < 5 seconds for 10,000 plots
  - [ ] Risk integration < 10 seconds for all 10 agents
  - [ ] Validation < 5 seconds for comprehensive Article 4 check
  - [ ] Package assembly < 60 seconds for 100 MB package
- [ ] All 7 commodity archetypes tested with golden test fixtures
- [ ] DDS assembly verified deterministic (bit-perfect reproducibility)
- [ ] Geolocation formatting verified against EU IS specification (100% pass rate)
- [ ] Risk integration verified against known baseline scores (100% accuracy)
- [ ] Compliance validation verified against manually audited DDS (zero false negatives)
- [ ] API documentation complete (OpenAPI spec with 19 endpoints)
- [ ] Database migration V125 tested and validated
- [ ] Integration with all 36 upstream EUDR agents verified
- [ ] Integration with EUDR-036 EU IS Interface for DDS handoff verified
- [ ] 5 beta customers successfully created and submitted DDS documents
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ DDS documents assembled by customers
- Average assembly time < 25 seconds
- Compliance validation pass rate >= 95%
- EU IS first-attempt acceptance rate >= 96%
- < 5 support tickets per customer related to DDS creation
- All 7 commodities used in production DDS

**60 Days:**
- 500+ DDS documents assembled
- Average assembly time < 20 seconds
- Compliance validation pass rate >= 97%
- EU IS first-attempt acceptance rate >= 98%
- < 3 support tickets per customer
- Batch assembly used by 30%+ of enterprise customers
- 50+ DDS amendments processed successfully

**90 Days:**
- 2,000+ DDS documents assembled
- Compliance validation pass rate >= 99%
- EU IS first-attempt acceptance rate >= 99%
- Zero EUDR penalties for active customers attributable to DDS quality
- NPS > 50 from compliance officer persona
- 99.5% uptime achieved
- < 2 support tickets per customer

---

## 10. Timeline and Milestones

### Phase 1: Core Assembly Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | DDS Document Assembler (Feature 1): upstream agent integration, data reconciliation, JSON/XML output | Senior Backend Engineer |
| 2-3 | Geolocation Formatter (Feature 2): WGS 84 normalization, polygon handling, country grouping | GIS Specialist |
| 3-4 | Risk Assessment Integrator (Feature 3): 10-agent score retrieval, Article 10 structuring | Backend Engineer |
| 4-5 | Supply Chain Data Compiler (Feature 4): 15-agent traceability compilation | Integration Engineer |
| 5-6 | Commodity Traceability Aggregator (Feature 5): 7-commodity handling, HS/CN codes, species names | Backend Engineer |

**Milestone: Core assembly engine operational with all data integration (Week 6)**

### Phase 2: Validation, Packaging, and API (Weeks 7-11)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Compliance Validator (Feature 6): Annex II validation, cross-field checks, readiness scoring | Senior Backend Engineer |
| 8-9 | Document Packager (Feature 7): evidence bundling, manifests, compression, size management | Backend Engineer |
| 9-10 | REST API Layer: 19 endpoints, authentication, RBAC, rate limiting | Backend Engineer |
| 10-11 | Digital Signature Preparer (Feature 8): PAdES/XAdES/JAdES, eIDAS compliance | Security Engineer |

**Milestone: Full API operational with validation, packaging, and signature preparation (Week 11)**

### Phase 3: Lifecycle Management and Integration (Weeks 12-15)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 12-13 | Version Controller (Feature 9): amendment, withdrawal, retention, version comparison | Backend Engineer |
| 13-14 | EUDR-036 integration: DDS handoff pipeline, submission status feedback loop | Integration Engineer |
| 14-15 | Batch assembly: parallel DDS creation, batch status tracking, batch validation | Backend Engineer |

**Milestone: Complete lifecycle management with EUDR-036 integration (Week 15)**

### Phase 4: Testing and Launch (Weeks 16-20)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 16-17 | Complete test suite: 500+ tests, golden tests for all 7 commodities | Test Engineer |
| 17-18 | Performance testing, security audit, load testing | DevOps + Security |
| 18-19 | Database migration V125 finalized, RBAC permissions registered | DevOps |
| 19 | Beta customer onboarding (5 customers) | Product + Engineering |
| 20 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 20)**

### Phase 5: Enhancements (Weeks 21-30)

- Multi-language DDS output (Feature 10)
- DDS analytics dashboard (Feature 11)
- Automated DDS refresh and re-assembly (Feature 12)
- Performance optimization for batch assembly of 1,000+ DDS
- Additional commodity-specific templates and validations

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
| EUDR-014 QR Code Generator Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-015 Mobile Data Collector Agent | BUILT (100%) | Low | Stable, production-ready |
| EUDR-016 through EUDR-025 (Risk Agents) | BUILT (100%) | Low | All 10 risk agents stable |
| EUDR-028 Risk Assessment Engine | BUILT (100%) | Low | Stable, production-ready |
| EUDR-029 Mitigation Measure Designer | BUILT (100%) | Low | Stable, production-ready |
| EUDR-030 Documentation Generator | BUILT (100%) | Low | Primary data source for DDS components |
| EUDR-036 EU Information System Interface | IN PROGRESS | Medium | Critical downstream; coordinate development timeline |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | Document encryption support |
| SEC-006 Secrets Management | BUILT (100%) | Low | Certificate storage in Vault |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System DDS schema | Published (v1.x) | Medium | Adapter pattern for schema version changes; versioned schema in reference_data |
| EU Annex I product list | Published; stable | Low | Database-driven; updateable without code changes |
| EC Article 29 country benchmarking list | Published; updated periodically | Medium | Hot-reloadable country risk classification in reference_data |
| eIDAS trust service providers | Available | Medium | Support multiple TSP providers; certificate management via SEC-006 |
| HS/CN code reference database | Published annually | Low | Reference data updated in deployment cycle |
| ISO 3166-1 country code standard | Stable | Low | Standard reference data |
| pyproj geodetic library | Stable open source | Low | Pinned version; comprehensive CRS database |
| Shapely geometry library | Stable open source | Low | Pinned version; well-tested polygon operations |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EU IS DDS schema updated with new mandatory fields | Medium | High | Adapter pattern isolates schema layer; versioned schema in reference_data; can update without touching assembly engine |
| R2 | Upstream agent data inconsistencies across 36 agents | High | Medium | Data reconciliation logic with priority ordering; conflict resolution rules; flagging of discrepancies in assembly provenance |
| R3 | Geolocation formatting rejected by EU IS despite local validation | Medium | High | Mirror EU IS validation rules locally; maintain test suite against known EU IS rejection patterns; rapid correction loop via EUDR-036 feedback |
| R4 | Risk agent scores change between DDS assembly and submission | Medium | Medium | Timestamp all risk scores; alert if scores change materially before submission; support rapid re-assembly |
| R5 | eIDAS qualified certificate not available or expired | Low | High | Multi-TSP support; certificate expiry alerting (90/60/30/7 days); fallback to advanced electronic signature |
| R6 | Batch assembly performance degrades at scale (1,000+ DDS) | Medium | Medium | Parallel processing with configurable concurrency; Redis-based job queue; horizontal scaling via K8s |
| R7 | Commodity-specific HS/CN code mapping errors | Low | High | Comprehensive reference database from EU Annex I; golden test fixtures for all 7 commodities; regulatory review of mappings |
| R8 | EUDR regulation amended (new commodities, new Annex II fields) | Medium | Medium | Modular design allows adding commodities and fields without architectural changes; configuration-driven field definitions |
| R9 | Low customer adoption of digital signature feature | Medium | Low | Signature is recommended but optional in v1.0; value demonstrated through non-repudiation benefits; gradual adoption supported |
| R10 | EUDR-036 EU IS Interface not ready when EUDR-037 launches | Medium | High | Design clean handoff interface; support DDS export for manual upload as fallback; coordinate development timelines closely |
| R11 | Evidence package sizes exceed EU IS limits for complex supply chains | Medium | Medium | Intelligent segmentation; compression; manifest-based multi-part upload; satellite imagery resolution reduction option |
| R12 | Competent authority demands documentation faster than 5-minute target | Low | Medium | Pre-built compliance packages cached; incremental package building; instant delivery of last-generated package |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| DDS Assembly Unit Tests | 100+ | Assembly from upstream data, field population, conflict reconciliation, schema generation |
| Geolocation Formatting Tests | 80+ | WGS 84 normalization, polygon closure, winding correction, CRS transformation, precision enforcement |
| Risk Integration Tests | 70+ | 10-agent score retrieval, composite calculation, Article 10 structuring, classification logic |
| Supply Chain Compilation Tests | 60+ | 15-agent data retrieval, summary generation, gap detection, cross-validation |
| Commodity Aggregation Tests | 50+ | All 7 commodities, HS/CN mapping, species names, derived products, multi-commodity |
| Compliance Validation Tests | 80+ | All 15 Annex II fields, cross-field validation, geolocation checks, risk completeness |
| Document Packaging Tests | 50+ | Evidence bundling, manifest generation, compression, segmentation, integrity hashes |
| Digital Signature Tests | 40+ | PAdES/XAdES/JAdES formatting, signature metadata, integrity verification, certificate validation |
| Version Control Tests | 50+ | State machine transitions, amendment chains, withdrawal, retention, version comparison |
| API Tests | 60+ | All 19 endpoints, authentication, authorization, error handling, pagination |
| Golden Tests | 49+ | All 7 commodities x 7 scenarios (complete/partial/broken/multi-plot/batch/high-risk/multi-tier) |
| Integration Tests | 30+ | Cross-agent integration with EUDR-001 through EUDR-036 |
| Performance Tests | 20+ | Assembly timing, batch throughput, geolocation at scale, concurrent access |
| **Total** | **750+** | |

### 13.2 Golden Test Commodities

Each of the 7 commodities will have dedicated golden test DDS scenarios:

1. **Complete DDS**: All 15 Annex II fields populated, all plots geolocated, risk assessed, compliance conclusion COMPLIANT -- expect 100% validation pass
2. **Partial DDS**: Some optional fields missing, some plots without polygon -- expect validation pass with warnings
3. **Broken DDS**: Missing mandatory fields (no geolocation, no risk assessment) -- expect validation FAIL with CRITICAL errors
4. **Multi-Plot DDS**: Product from 100+ plots across 3+ countries -- expect correct geolocation grouping and multi-country handling
5. **Batch Split/Merge DDS**: Commodity processed through splits and merges -- expect correct mass balance and origin preservation
6. **High-Risk DDS**: Product from high-risk country with deforestation alerts -- expect STANDARD or HIGH risk classification, mitigation required
7. **Multi-Tier DDS**: Supply chain with 6+ tiers -- expect complete tier documentation and supply chain summary

Total: 7 commodities x 7 scenarios = 49 golden test scenarios

### 13.3 Determinism Tests

Critical tests verifying that the agent produces bit-identical output from identical inputs:
- Same upstream agent data -> identical DDS JSON (byte-for-byte comparison)
- Same geolocation input -> identical formatted coordinates
- Same risk scores -> identical integrated risk assessment
- Same DDS -> identical SHA-256 provenance hash
- 100 sequential runs with same input -> zero variance in any output field

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 and submitted to the EU Information System |
| **EU IS** | EU Information System -- the digital platform established under Article 33 for DDS submission and operator registration |
| **eIDAS** | Electronic Identification, Authentication and trust Services -- EU Regulation No 910/2014 governing electronic signatures |
| **QES** | Qualified Electronic Signature -- highest level of electronic signature under eIDAS, legally equivalent to handwritten signature |
| **PAdES** | PDF Advanced Electronic Signatures -- signature format for PDF documents per ETSI EN 319 142 |
| **XAdES** | XML Advanced Electronic Signatures -- signature format for XML documents per ETSI EN 319 132 |
| **JAdES** | JSON Advanced Electronic Signatures -- signature format for JSON documents per ETSI EN 319 182 |
| **CN Code** | Combined Nomenclature -- EU product classification code used in customs declarations |
| **HS Code** | Harmonized System -- international product classification code |
| **EORI** | Economic Operators Registration and Identification -- unique EU customs identification number |
| **WGS 84** | World Geodetic System 1984 -- standard coordinate reference system (EPSG:4326) required by EUDR Article 9 |
| **CRS** | Coordinate Reference System -- mathematical model for mapping coordinates to locations on Earth |
| **Annex I** | EUDR Annex I -- list of regulated products with HS/CN codes |
| **Annex II** | EUDR Annex II -- list of information required in a Due Diligence Statement |
| **Article 29** | EUDR article establishing country benchmarking system (Low/Standard/High risk) |
| **Article 13** | EUDR article providing simplified due diligence for products exclusively from low-risk countries |
| **Mass Balance** | Chain of custody model where compliant and non-compliant material may be mixed but quantities are tracked |
| **Identity Preserved** | Chain of custody model where compliant material is physically separated throughout the supply chain |
| **Segregated** | Chain of custody model where compliant material is kept separate from non-compliant material |

### Appendix B: EUDR Annex II Field Mapping

Complete mapping of every Annex II DDS field to the specific engine within this agent that populates it:

| Annex II Field | Engine | Source Agent(s) | Validation Rule |
|----------------|--------|-----------------|-----------------|
| Operator name | DDSDocumentAssembler | Operator records | Non-empty, max 500 chars |
| Operator address | DDSDocumentAssembler | Operator records | Structured address, non-empty |
| Operator EORI | DDSDocumentAssembler | EUDR-036 | 2-letter country + up to 15 digits |
| Registration number | DDSDocumentAssembler | EUDR-036 | Valid, non-expired per target MS |
| Product trade name | DDSDocumentAssembler | EUDR-001, EUDR-009 | Non-empty |
| Product type | DDSDocumentAssembler | EUDR-009 | Valid product type classification |
| Common name | CommodityTraceabilityAggregator | Reference data | Matches commodity |
| Scientific name | CommodityTraceabilityAggregator | Reference data | Valid species per commodity |
| HS/CN code | CommodityTraceabilityAggregator | Annex I reference | Valid code from Annex I list |
| Quantity | DDSDocumentAssembler | EUDR-011, EUDR-010 | Positive numeric, valid unit |
| Country of production | DDSDocumentAssembler | EUDR-001, EUDR-008 | Valid ISO 3166-1 alpha-2 |
| Geolocation | GeolocationFormatter | EUDR-002, EUDR-006, EUDR-007 | WGS 84, 6 decimals, polygon rules |
| Production dates | DDSDocumentAssembler | EUDR-001, EUDR-008 | Start before end, reasonable range |
| Supplier info | DDSDocumentAssembler | EUDR-008 | Name + address + valid email |
| Buyer info | DDSDocumentAssembler | Operator records | Name + address + valid email |
| Deforestation evidence | DDSDocumentAssembler | EUDR-003, EUDR-004, EUDR-005 | At least 1 evidence reference |
| Legal production evidence | DDSDocumentAssembler | EUDR-023 | At least 1 evidence reference |
| Risk assessment | RiskAssessmentIntegrator | EUDR-016 through EUDR-025 | All Art 10(2) criteria evaluated |
| Mitigation measures | DDSDocumentAssembler | EUDR-029 | Required if risk non-negligible |
| Compliance declaration | DDSDocumentAssembler | ComplianceValidator | Consistent with risk + mitigation |

### Appendix C: Article 10(2) Criterion Mapping to Risk Agents

| Criterion | Article 10(2) Text | Primary Agent | Supporting Agents |
|-----------|-------------------|---------------|-------------------|
| (a) | Risk of deforestation in country of production | EUDR-016 Country Risk | EUDR-020 Deforestation Alert |
| (b) | Presence of forests in country of production | EUDR-004 Forest Cover | EUDR-003 Satellite Monitoring |
| (c) | Presence of indigenous peoples | EUDR-021 Indigenous Rights | -- |
| (d) | Consultation with indigenous peoples | EUDR-021 Indigenous Rights | EUDR-024 Third-Party Audit |
| (e) | Concerns about country of production | EUDR-016 Country Risk | EUDR-019 Corruption Index |
| (f) | Risk of circumvention or mixing | EUDR-009 Chain of Custody | EUDR-010 Segregation, EUDR-011 Mass Balance |
| (g) | History of compliance of supplier | EUDR-017 Supplier Risk | EUDR-024 Third-Party Audit |

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EU Deforestation Regulation)
2. Regulation (EU) No 910/2014 (eIDAS -- Electronic Identification and Trust Services)
3. ETSI EN 319 142 -- PAdES (PDF Advanced Electronic Signatures)
4. ETSI EN 319 132 -- XAdES (XML Advanced Electronic Signatures)
5. ETSI EN 319 182 -- JAdES (JSON Advanced Electronic Signatures)
6. EU EUDR Information System Technical Specifications (Commission Implementing Regulation)
7. ISO 22095:2020 -- Chain of Custody -- General Terminology and Models
8. ISO 3166-1:2020 -- Codes for the representation of names of countries
9. EU Combined Nomenclature (Council Regulation (EEC) No 2658/87, updated annually)
10. FSC Chain of Custody Standard (FSC-STD-40-004)
11. RSPO Supply Chain Certification Standard
12. Global Forest Watch Technical Documentation
13. EUDR Guidance Document (European Commission)

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-13 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Security Lead | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-13 | GL-ProductManager | Initial draft created: 9 P0 features defined, regulatory mapping to Articles 4/9/10/11/12/13/14/29/31/33 and eIDAS completed, 36-agent integration architecture specified, 7-commodity coverage verified, technical architecture with 7 engines designed, V125 migration schema defined, 750+ test target established |
