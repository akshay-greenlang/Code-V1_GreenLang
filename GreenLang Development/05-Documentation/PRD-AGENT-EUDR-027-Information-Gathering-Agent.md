# PRD: AGENT-EUDR-027 -- Information Gathering Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-027 |
| **Agent ID** | GL-EUDR-IGA-027 |
| **Component** | Information Gathering Agent |
| **Category** | EUDR Regulatory Agent -- Information Gathering (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-11 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 4, 9, 10, 12, 13, 29, 31 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 9 mandates that operators collect comprehensive information about their products and supply chains before placing regulated commodities on the EU market. Article 4(2) requires operators to gather: product descriptions with CN/HS codes, quantities, countries of production, geolocation of all plots of land, date or date range of production, supplier and buyer identification, and substantiated evidence that products are deforestation-free and legally produced. This information must be assembled into a coherent package supporting the Due Diligence Statement (DDS) submitted per Articles 12 and 13.

The GreenLang platform has built 26 EUDR agents spanning supply chain traceability (EUDR-001 through EUDR-015), risk assessment (EUDR-016 through EUDR-020), environmental and social rights (EUDR-021 through EUDR-022), and due diligence workflow management (EUDR-023 through EUDR-026). These agents provide deep analytical capabilities, but they depend on information that must first be **gathered** from diverse, distributed, and often incompatible external sources. Today, operators face the following critical gaps in the information gathering phase:

- **No automated external database integration**: EUDR compliance requires cross-referencing data from EU TRACES (trade control system), CITES (Convention on International Trade in Endangered Species), FLEGT/VPA licensing databases (Forest Law Enforcement, Governance and Trade), national customs registries, and phytosanitary certificate databases. Operators must manually query each system, download results in incompatible formats (CSV, XML, PDF, HTML), and manually reconcile records. This process takes 40-80 hours per commodity per country of origin.

- **No certification database aggregation**: Evidence of sustainable sourcing requires querying certification body databases (FSC for forest products, RSPO for palm oil, Rainforest Alliance for cocoa/coffee, UTZ for cocoa, PEFC for timber). Each certification body has its own API, database format, license validation process, and certificate lifecycle management. There is no unified connector that queries all relevant certification databases, validates certificate authenticity and currency, and compiles certification evidence for the DDS.

- **No public data mining capability**: Operators must collect publicly available data from FAO (Food and Agriculture Organization), UN COMTRADE (trade statistics), Global Forest Watch, World Bank governance indicators, Transparency International corruption indices, and national land registry databases. This data is critical for Article 10 risk assessment and Article 9 contextual information, but must currently be gathered manually from dozens of web portals with different access methods, data formats, and update schedules.

- **No supplier information normalization**: Supplier self-declared information arrives in heterogeneous formats -- questionnaire responses (AGENT-DATA-008), email attachments, portal uploads, ERP exports, spreadsheets, scanned documents, and verbal declarations. There is no standardized normalization engine that ingests these diverse inputs, validates them against EUDR requirements, resolves inconsistencies, and produces a unified, structured supplier information record.

- **No information completeness validation**: Before risk assessment can begin (Article 10), operators must verify that all Article 9 information elements are present, valid, and internally consistent. Without automated completeness validation, operators cannot determine if their information package is adequate for DDS submission, leading to incomplete filings and regulatory penalties.

- **No regulatory reference data management**: EUDR compliance requires maintaining current reference data: EU country benchmarking lists (Article 29 classifications updated quarterly by the EC), CN/HS code product classifications, commodity-derived product mappings, and regulatory thresholds. This reference data changes frequently, and operators have no automated mechanism to track, validate, and apply updates.

- **No evidence package assembly**: The information gathered from external databases, certification bodies, public sources, and suppliers must be assembled into a structured, provenance-tracked, hash-verified evidence package. Without automated assembly, compliance officers spend 20-30 hours per product manually compiling and cross-referencing evidence documents.

Without solving these problems, the downstream EUDR agents (risk assessment, mitigation, DDS generation) operate on incomplete or outdated information, compromising the entire due diligence process. Operators face penalties of up to 4% of annual EU turnover, goods confiscation, temporary exclusion from public procurement, and public naming under Articles 23-25.

### 1.2 Solution Overview

Agent-EUDR-027: Information Gathering Agent is a specialized data collection and normalization agent that automates the comprehensive gathering of EUDR Article 9 required information from multiple external and internal data sources. It operates as the primary data acquisition layer for the entire due diligence process, feeding validated, normalized, and complete information packages to downstream risk assessment (EUDR-016 through EUDR-025) and due diligence orchestration (EUDR-026) agents.

Core capabilities:

1. **External Database Connector Engine** -- Unified connector framework for querying EU TRACES, CITES trade databases, FLEGT/VPA licensing systems, national customs registries, phytosanitary certificate databases, and sanction lists. Each connector implements adapter pattern with standardized query/response interfaces, rate limiting, retry logic, and response caching.

2. **Certification Verification Engine** -- Integrated connector for FSC, RSPO, PEFC, Rainforest Alliance, UTZ, SAN, and EU organic certification databases. Validates certificate authenticity, checks current validity status, verifies chain-of-custody scope, and detects expired or suspended certificates. Maintains a local cache with configurable TTL for high-frequency lookups.

3. **Public Data Mining Engine** -- Automated collection of publicly available data from FAO STAT, UN COMTRADE, Global Forest Watch, World Bank WGI, Transparency International CPI, national land registries, and academic deforestation monitoring databases. Implements scheduled harvesting with incremental updates and data freshness monitoring.

4. **Supplier Information Aggregator** -- Normalizes and validates supplier self-declared information from questionnaire responses (AGENT-DATA-008), ERP exports (AGENT-DATA-003), document extracts (AGENT-DATA-001), spreadsheet imports (AGENT-DATA-002), and direct API integrations. Resolves entity conflicts, deduplicates supplier records, and produces unified supplier information profiles.

5. **Information Completeness Validator** -- Validates gathered information against the complete Article 9 checklist (product description, quantity, country of production, geolocation, production date range, supplier identification, deforestation-free evidence, legal production evidence). Computes element-level and aggregate completeness scores. Generates remediation action lists for missing or insufficient information elements.

6. **Data Normalization Engine** -- Normalizes heterogeneous data formats, measurement units, date formats, geographic coordinate systems, product classification codes, and currency values into EUDR-compliant standardized formats. Applies configurable transformation rules per commodity type, country of origin, and data source.

7. **Information Package Assembler** -- Compiles all gathered, validated, and normalized information into a structured, SHA-256 hash-verified evidence package. Generates Article 9 compliance summaries, produces machine-readable JSON packages for downstream agent consumption, and creates human-readable evidence indices for auditor review. Maintains full provenance chain from original source to assembled package.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| External database query coverage | 10+ external databases connected | Integration test matrix |
| Certification verification accuracy | >= 99.5% correct validity status | Cross-verification against certification body portals |
| Public data freshness | Data updated within 24 hours of source publication | Data freshness monitoring |
| Supplier information normalization | Support for 15+ input formats | Format coverage matrix |
| Information completeness scoring | 100% deterministic, reproducible | Bit-perfect reproducibility tests |
| Article 9 element coverage | All 10 mandatory elements validated | Regulatory compliance matrix |
| Evidence package assembly time | < 60 seconds for standard package | Package generation benchmarks |
| Data normalization accuracy | >= 99.9% correct unit/format conversions | Golden test validation |
| External API response time | < 5 seconds p95 per database query | Latency monitoring |
| Error recovery from API failures | >= 95% automatic recovery | Retry success rate tracking |
| Package provenance integrity | 100% SHA-256 hash verification | Hash chain validation |
| Concurrent gathering operations | 100+ simultaneous per operator | Load test benchmarks |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated information gathering automation market of 2-4 billion EUR as the volume of required evidence collection grows with each reporting period.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring automated information gathering, estimated at 600M-1B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 40-60M EUR in information gathering module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) processing multiple EUDR commodities
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya)
- Timber and paper industry operators with complex multi-country sourcing
- Compliance teams at rubber and cattle product importers

**Secondary:**
- Customs brokers and freight forwarders handling EUDR-regulated goods
- Certification bodies verifying operator compliance claims
- Third-party audit firms conducting EUDR verification
- Government agencies monitoring EUDR compliance (competent authorities)
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual collection | No cost; flexibility | 40-80 hrs/commodity; error-prone; incomplete | Automated; < 60 seconds per package; 99.5% accuracy |
| Generic data aggregators | Broad data access | Not EUDR-specific; no Article 9 validation; no certification verification | Purpose-built for EUDR Article 9; full regulatory validation |
| Certification body portals | Authoritative data | Single-body only; no cross-platform aggregation; manual query | All major certification bodies unified; batch verification |
| Custom ERP integrations | Tailored to org | 6-12 month build; no external sources; no regulatory updates | Ready now; 10+ external sources; continuous regulatory updates |
| Niche EUDR tools | Commodity expertise | Limited external database coverage; manual evidence compilation | Comprehensive external coverage; automated package assembly |

### 2.4 Differentiation Strategy

1. **Unified external data layer** -- Single agent that connects to all relevant external databases, eliminating manual multi-portal data collection.
2. **Certification aggregation** -- First platform to unify FSC, RSPO, PEFC, Rainforest Alliance, UTZ, and EU organic verification through a single API.
3. **Article 9 compliance scoring** -- Deterministic completeness validation against every Article 9 element with gap-specific remediation guidance.
4. **Deep GreenLang integration** -- Pre-built data handoff to 20+ downstream EUDR agents for seamless risk assessment and DDS generation.
5. **Zero-hallucination information** -- All collected data traceable to authoritative sources with SHA-256 provenance chains; no LLM-generated evidence.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Reduce information gathering time from weeks to minutes | 95% reduction in time-to-complete Article 9 information | Q2 2026 |
| BG-2 | Eliminate manual external database queries | 100% automated for all connected databases | Q2 2026 |
| BG-3 | Ensure information completeness for DDS submission | >= 95% Article 9 completeness before risk assessment | Q3 2026 |
| BG-4 | Reduce compliance documentation errors | < 1% error rate in assembled evidence packages | Ongoing |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | External database coverage | Connect to 10+ external databases relevant to EUDR compliance |
| PG-2 | Certification verification | Verify certificates from 6+ major certification bodies |
| PG-3 | Public data aggregation | Harvest data from 8+ public data sources on scheduled basis |
| PG-4 | Supplier normalization | Normalize supplier information from 15+ input formats |
| PG-5 | Completeness validation | Validate all 10 Article 9 mandatory information elements |
| PG-6 | Evidence assembly | Generate structured, hash-verified evidence packages |
| PG-7 | Regulatory reference data | Maintain current EU country benchmarking and product classifications |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | External database query latency | < 5 seconds p95 per query |
| TG-2 | Package assembly throughput | < 60 seconds for standard package |
| TG-3 | Concurrent operations | 100+ simultaneous gathering operations |
| TG-4 | API response time | < 200ms p95 for internal API calls |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-7 | Data normalization accuracy | >= 99.9% correct transformations |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries |
| **EUDR Pressure** | Must gather comprehensive Article 9 information for 200+ supply chain paths |
| **Pain Points** | Spends 3 weeks manually querying certification databases, customs records, and government registries; information arrives in incompatible formats; cannot verify completeness against Article 9 requirements |
| **Goals** | Automated information gathering from all relevant sources; real-time completeness tracking; audit-ready evidence packages |
| **Technical Skill** | Moderate -- comfortable with web applications but not APIs or databases |

### Persona 2: Data Analyst -- Thomas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Supply Chain Data Analyst at a palm oil trading company |
| **Company** | 1,200 employees, trading palm oil from Indonesia and Malaysia |
| **EUDR Pressure** | Must reconcile supplier declarations with external database records for 500+ plantations |
| **Pain Points** | Supplier-provided information contradicts public database records; certificate numbers cannot be verified quickly; no automated way to cross-reference FAO data with customs records |
| **Goals** | Automated cross-referencing of supplier data against external sources; discrepancy detection; normalized data feeds for risk assessment |
| **Technical Skill** | High -- comfortable with data tools, APIs, and scripting |

### Persona 3: Certification Manager -- Sophie (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Certification and Standards Manager at a timber importer |
| **Company** | 800 employees, importing FSC/PEFC certified wood from 15 countries |
| **EUDR Pressure** | Must verify currency and scope of 2,000+ supplier certificates across FSC, PEFC, and national forestry schemes |
| **Pain Points** | Manual certificate verification takes 2 minutes per certificate; certificates expire without notification; suspended certificates discovered too late |
| **Goals** | Batch certificate verification; proactive expiry alerts; automated re-verification on schedule |
| **Technical Skill** | Moderate -- uses web portals and spreadsheets |

### Persona 4: External Auditor -- Dr. Weber (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify that operators have gathered sufficient information per Article 9 |
| **Pain Points** | Operators provide ad-hoc evidence bundles; no standardized evidence package format; difficult to trace data provenance back to source |
| **Goals** | Structured evidence packages with provenance chains; verifiable data source attribution; standardized completeness reporting |
| **Technical Skill** | Moderate -- comfortable with audit software and structured data |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(2)** | Due diligence obligation -- collect, assess, mitigate | Information Gathering Agent automates the "collect" phase for all required data elements |
| **Art. 4(2)(a)** | Description of the product, including trade name, product type, and common/scientific name | Product description collector with CN/HS code resolution and commodity classification |
| **Art. 4(2)(b)** | Quantity of the product | Quantity extraction and normalization from trade documents and customs records |
| **Art. 4(2)(c)** | Country of production; where applicable, parts thereof | Country and sub-national region identification from trade records and supplier declarations |
| **Art. 4(2)(d)** | Geolocation of all plots of land | Geolocation data collection from supplier declarations, land registries, and GPS data integration |
| **Art. 4(2)(e)** | Date or time range of production | Production date extraction from phytosanitary certificates, bills of lading, and supplier records |
| **Art. 4(2)(f)** | Information on supply chain | Supply chain data aggregation from ERP, questionnaire, and trade document sources |
| **Art. 4(2)(g)** | Adequately conclusive information that products are deforestation-free | Deforestation-free evidence compilation from satellite data, certification records, and government attestations |
| **Art. 4(2)(h)** | Adequately conclusive information that products were produced in accordance with relevant legislation | Legal production evidence from FLEGT licenses, phytosanitary certificates, export permits, and legal compliance databases |
| **Art. 9(1)(a-d)** | Geolocation specifications (GPS, polygon > 4 ha) | Geolocation data validation and format normalization per Article 9 specifications |
| **Art. 10(1)** | Risk assessment requires complete information | Completeness Validator ensures information sufficiency before risk assessment phase |
| **Art. 12** | DDS content requirements | Evidence Package Assembler formats output for DDS compatibility |
| **Art. 13** | Simplified due diligence for low-risk countries | Reduced information requirements for Article 29 low-risk origin countries |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | Regulatory reference data engine maintains current EC country classifications |
| **Art. 31** | Record keeping for 5 years | All gathered information and evidence packages retained with immutable provenance |

### 5.2 Article 9 Information Elements Checklist

| # | Element | Source(s) | Agent Engine |
|---|---------|-----------|-------------|
| 1 | Product description (CN/HS code, trade name, type, common/scientific name) | Trade documents, customs records, supplier declarations | External Database Connector, Supplier Information Aggregator |
| 2 | Quantity (net mass, volume, number of items) | Bills of lading, customs declarations, invoices | External Database Connector, Data Normalization Engine |
| 3 | Country of production (sub-national region if applicable) | Certificates of origin, supplier declarations, customs records | External Database Connector, Public Data Mining Engine |
| 4 | Geolocation of all plots (GPS, polygon > 4 ha) | Supplier GPS submissions, land registries, satellite data | Supplier Information Aggregator, Public Data Mining Engine |
| 5 | Date or date range of production | Phytosanitary certificates, harvest records, processing records | External Database Connector, Supplier Information Aggregator |
| 6 | Supplier name, postal address, email | Supplier questionnaires, trade documents, business registries | External Database Connector, Supplier Information Aggregator |
| 7 | Buyer/recipient identification | Trade documents, ERP records | External Database Connector, Supplier Information Aggregator |
| 8 | Deforestation-free evidence | Satellite analysis, certification records, government attestations | Certification Verification Engine, Public Data Mining Engine |
| 9 | Legal production evidence | FLEGT licenses, phytosanitary certificates, export permits | External Database Connector, Certification Verification Engine |
| 10 | Supply chain information (chain of custody) | ERP, questionnaires, trade documents | Supplier Information Aggregator, External Database Connector |

### 5.3 External Database Sources

| # | Database | Data Type | Access Method | Update Frequency |
|---|----------|-----------|---------------|-----------------|
| 1 | EU TRACES | Trade control, phytosanitary certificates | REST API / SOAP | Real-time |
| 2 | CITES Trade Database | Endangered species trade permits | REST API | Monthly |
| 3 | EU FLEGT/VPA System | Forest law enforcement licenses | REST API | Weekly |
| 4 | UN COMTRADE | International trade statistics | REST API | Monthly |
| 5 | FAO STAT | Agriculture and forestry statistics | REST API | Quarterly |
| 6 | Global Forest Watch | Deforestation alerts, forest cover | REST API | Weekly |
| 7 | World Bank WGI | Governance indicators | REST API | Annually |
| 8 | Transparency Intl CPI | Corruption perception index | REST API | Annually |
| 9 | FSC Certificate Database | Forest certification records | REST API | Real-time |
| 10 | RSPO PalmTrace | Palm oil certification tracking | REST API | Real-time |
| 11 | PEFC Certificate Database | Forest certification records | REST API | Real-time |
| 12 | Rainforest Alliance | Cocoa/coffee certification | REST API | Real-time |
| 13 | EU Sanctions Lists | Restricted entities | REST API | Daily |
| 14 | National Land Registries | Plot ownership, land use rights | Varies by country | Varies |

### 5.4 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Reference date for all deforestation-free evidence |
| June 29, 2023 | Regulation entered into force | Legal basis for information requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Full information gathering automation required |
| June 30, 2026 | Enforcement for SMEs | SME-tier information gathering templates |
| Ongoing (quarterly) | Country benchmarking updates by EC | Agent must consume and apply updated country classifications |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 7 features below are P0 launch blockers. The agent cannot ship without all 7 features operational.

---

#### Feature 1: External Database Connector Engine

**User Story:**
```
As a compliance officer,
I want the system to automatically query EU trade databases, customs registries, and government systems for product and supplier information,
So that I can gather all required Article 9 data without manually logging into multiple portals.
```

**Acceptance Criteria:**
- [ ] Implements adapter pattern with standardized ExternalDatabaseConnector interface
- [ ] Connects to EU TRACES for trade control and phytosanitary certificate queries
- [ ] Connects to CITES trade database for endangered species permit verification
- [ ] Connects to FLEGT/VPA licensing system for forest law enforcement license checks
- [ ] Connects to UN COMTRADE for international trade statistics
- [ ] Connects to EU sanctions lists for restricted entity screening
- [ ] Connects to national customs registries (configurable per country)
- [ ] Implements per-connector rate limiting with configurable request quotas
- [ ] Implements retry logic with exponential backoff for transient failures
- [ ] Caches responses in Redis with configurable TTL per data source
- [ ] Normalizes responses from all connectors into unified QueryResult format
- [ ] Logs every external query with source, parameters, response status, and latency
- [ ] Supports both synchronous and asynchronous query modes
- [ ] Implements circuit breaker for each external database connection
- [ ] Provides fallback to cached data when external source is unavailable

**Non-Functional Requirements:**
- Latency: < 5 seconds p95 per individual database query
- Throughput: 50+ concurrent queries across all databases
- Availability: Graceful degradation with cached fallback
- Security: All credentials stored in Vault (SEC-006); no secrets in code or config

**Dependencies:**
- Redis cache (INFRA-003)
- Vault secrets management (SEC-006)
- Circuit breaker pattern from error recovery framework

**Estimated Effort:** 4 weeks (1 senior backend engineer)

**Edge Cases:**
- External database returns partial results -- mark as partial and flag for retry
- External database changes API version -- adapter pattern isolates change
- Rate limiting exceeded -- exponential backoff with jitter
- Credential rotation -- Vault integration handles automatic refresh

---

#### Feature 2: Certification Verification Engine

**User Story:**
```
As a certification manager,
I want the system to automatically verify the validity and scope of supplier certificates from FSC, RSPO, PEFC, Rainforest Alliance, and other bodies,
So that I can ensure all certification claims are current and applicable without manually checking each certificate.
```

**Acceptance Criteria:**
- [ ] Verifies FSC certificates: license code, scope, validity dates, suspension status
- [ ] Verifies RSPO PalmTrace certificates: membership, supply chain model, valid until
- [ ] Verifies PEFC certificates: certificate number, standard version, scope, validity
- [ ] Verifies Rainforest Alliance certificates: certificate ID, commodity scope, expiry date
- [ ] Verifies UTZ/SAN certificates: registration number, program, validity
- [ ] Verifies EU organic certification: control body, scope, certification status
- [ ] Supports batch verification: up to 1,000 certificates per request
- [ ] Detects expired certificates with configurable advance warning (default 90 days)
- [ ] Detects suspended or withdrawn certificates with alert generation
- [ ] Verifies chain-of-custody scope matches claimed activities (processing, trading, etc.)
- [ ] Returns structured CertificateVerificationResult with status, details, and evidence hash
- [ ] Caches verification results with configurable TTL (default 24 hours)
- [ ] Tracks certificate lifecycle: first seen, last verified, status history
- [ ] Generates certificate compliance matrix per supplier

**Non-Functional Requirements:**
- Accuracy: >= 99.5% correct verification status against certification body portals
- Throughput: 1,000 certificate verifications in < 5 minutes
- Freshness: Cached results refreshed within 24 hours

**Dependencies:**
- FSC Public Certificate Database API
- RSPO PalmTrace API
- PEFC Certificate Search API
- Rainforest Alliance Certification Portal API
- Redis cache (INFRA-003)

**Estimated Effort:** 3 weeks (1 backend engineer)

---

#### Feature 3: Public Data Mining Engine

**User Story:**
```
As a data analyst,
I want the system to automatically harvest relevant publicly available data from FAO, Global Forest Watch, World Bank, and other sources,
So that I can access up-to-date contextual information for risk assessment without manual data collection.
```

**Acceptance Criteria:**
- [ ] Harvests FAO STAT data: production volumes, yields, land use by country/commodity
- [ ] Harvests UN COMTRADE data: bilateral trade flows, HS code volumes, trade partners
- [ ] Harvests Global Forest Watch data: deforestation alerts, tree cover loss, fire alerts
- [ ] Harvests World Bank WGI data: governance indicators by country
- [ ] Harvests Transparency International CPI data: corruption perception scores
- [ ] Harvests national land registry data (where available via API)
- [ ] Harvests EU country benchmarking classifications (Article 29)
- [ ] Harvests EU sanctions and restricted entity lists
- [ ] Implements scheduled harvesting with configurable intervals per source
- [ ] Supports incremental updates: only fetch data changed since last harvest
- [ ] Normalizes harvested data into standardized PublicDataRecord format
- [ ] Tracks data freshness with source-level timestamp tracking
- [ ] Alerts when data exceeds configurable freshness threshold
- [ ] Stores harvested data in PostgreSQL with TimescaleDB time-series optimization
- [ ] Generates data freshness report across all sources

**Non-Functional Requirements:**
- Freshness: Data updated within 24 hours of source publication
- Storage: Efficient storage with JSONB for flexible schema
- Incremental: Only changed data transferred on update
- Resilience: Harvest failure for one source does not block others

**Dependencies:**
- PostgreSQL + TimescaleDB (INFRA-002)
- Scheduled task execution framework
- External API clients

**Estimated Effort:** 3 weeks (1 data engineer)

---

#### Feature 4: Supplier Information Aggregator

**User Story:**
```
As a compliance officer,
I want the system to automatically collect, normalize, and validate supplier self-declared information from multiple sources,
So that I can build a complete supplier profile meeting EUDR Article 9 requirements.
```

**Acceptance Criteria:**
- [ ] Ingests supplier questionnaire responses from AGENT-DATA-008
- [ ] Ingests ERP procurement records from AGENT-DATA-003
- [ ] Ingests document extracts from AGENT-DATA-001 (PDF invoices, certificates)
- [ ] Ingests spreadsheet imports from AGENT-DATA-002 (Excel/CSV supplier lists)
- [ ] Ingests mobile data collections from AGENT-EUDR-015
- [ ] Resolves entity conflicts: same supplier appearing under different names/IDs
- [ ] Deduplicates supplier records using fuzzy matching on name, address, registration number
- [ ] Produces unified SupplierProfile with all EUDR-required fields
- [ ] Validates mandatory fields per Article 9: name, postal address, email address
- [ ] Tracks information source priority: government registry > certification body > supplier self-declared
- [ ] Flags discrepancies between sources (e.g., declared country vs. customs record)
- [ ] Generates supplier information completeness score per supplier
- [ ] Supports bulk ingestion of 10,000+ supplier records per operation
- [ ] Maintains full audit trail of all data source contributions per field

**Non-Functional Requirements:**
- Accuracy: >= 99% correct entity resolution
- Throughput: 10,000 supplier records processed in < 5 minutes
- Deduplication: False positive rate < 1%, false negative rate < 5%

**Dependencies:**
- AGENT-DATA-001 PDF & Invoice Extractor
- AGENT-DATA-002 Excel/CSV Normalizer
- AGENT-DATA-003 ERP/Finance Connector
- AGENT-DATA-008 Supplier Questionnaire Processor
- AGENT-EUDR-015 Mobile Data Collector

**Estimated Effort:** 3 weeks (1 backend engineer)

---

#### Feature 5: Information Completeness Validator

**User Story:**
```
As a compliance officer,
I want the system to automatically validate whether my gathered information meets all EUDR Article 9 requirements,
So that I can identify gaps before submitting my Due Diligence Statement and avoid regulatory penalties.
```

**Acceptance Criteria:**
- [ ] Validates all 10 Article 9 information elements (see Section 5.2)
- [ ] Computes element-level completeness: present/absent/partial for each element
- [ ] Computes aggregate completeness score (0-100) using configurable element weights
- [ ] Classifies completeness: Insufficient (< 60), Partial (60-89), Complete (>= 90)
- [ ] Applies commodity-specific validation rules (cattle geolocation vs. cocoa geolocation differ)
- [ ] Applies country-specific validation rules (FLEGT required for VPA partner countries)
- [ ] Applies simplified due diligence rules per Article 13 for low-risk countries
- [ ] Generates gap report listing missing elements with specific remediation actions
- [ ] Validates cross-element consistency (e.g., declared country matches geolocation coordinates)
- [ ] Validates data freshness: flags elements older than configurable threshold (default 12 months)
- [ ] Validates data source authority: higher weight for government/certification sources
- [ ] All completeness calculations are deterministic (no LLM involvement)
- [ ] Returns CompletenessReport with element-by-element breakdown

**Completeness Scoring Formula:**
```
Completeness = sum(element_weight_i * element_status_i) / sum(element_weight_i)

Where:
- element_weight_i = configured weight for element i (default equal weights)
- element_status_i = 1.0 (complete), 0.5 (partial), 0.0 (missing)
```

**Non-Functional Requirements:**
- Determinism: Bit-perfect reproducibility across runs
- Performance: Full validation < 5 seconds for standard product
- Coverage: 100% of Article 9 elements validated

**Dependencies:**
- Features 1-4 (data sources)
- EUDR Article 9 regulatory specification

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 6: Data Normalization Engine

**User Story:**
```
As a data analyst,
I want the system to automatically normalize data from different sources into consistent formats, units, and classifications,
So that I can reliably compare and analyze information from heterogeneous sources.
```

**Acceptance Criteria:**
- [ ] Normalizes measurement units: kg, tonnes, cubic meters, liters, pieces, heads (cattle)
- [ ] Normalizes date formats: ISO 8601, EU format, US format, UNIX timestamps
- [ ] Normalizes geographic coordinates: DD, DMS, UTM to WGS84 decimal degrees
- [ ] Normalizes product classification codes: CN codes, HS codes, TARIC codes
- [ ] Normalizes currency values to EUR using ECB reference exchange rates
- [ ] Normalizes country identifiers: ISO 3166-1 alpha-2, alpha-3, numeric, common names
- [ ] Normalizes certificate identifiers: standardized per certification body format
- [ ] Normalizes address formats: street, city, postal code, country
- [ ] Applies commodity-specific normalization rules (cattle weights, timber volumes)
- [ ] Handles missing values with configurable strategies: skip, default, flag
- [ ] Validates normalization results against expected ranges per field type
- [ ] Maintains original value and normalized value for audit trail
- [ ] Supports configurable transformation rules via rule definition files
- [ ] Logs all normalization transformations with source value and result

**Non-Functional Requirements:**
- Accuracy: >= 99.9% correct transformations
- Performance: 10,000 normalizations per second
- Auditability: Every transformation logged with before/after values

**Dependencies:**
- ECB exchange rate API for currency conversion
- ISO code reference databases

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 7: Information Package Assembler

**User Story:**
```
As a compliance officer,
I want the system to compile all gathered information into a structured, hash-verified evidence package,
So that I can submit a complete Due Diligence Statement and provide audit-ready documentation.
```

**Acceptance Criteria:**
- [ ] Compiles gathered data from all sources into unified InformationPackage
- [ ] Generates Article 9 compliance summary with element-by-element status
- [ ] Produces machine-readable JSON package for downstream agent consumption
- [ ] Produces human-readable evidence index (PDF) for auditor review
- [ ] Includes SHA-256 hash on every evidence artifact for integrity verification
- [ ] Includes full provenance chain: source -> collection timestamp -> normalization -> assembly
- [ ] Links evidence to specific Article 9 elements (evidence-to-requirement mapping)
- [ ] Supports package versioning with immutable snapshots
- [ ] Generates package comparison (diff) between versions
- [ ] Exports package in multiple formats: JSON, XML (EU Information System), PDF
- [ ] Validates assembled package against DDS schema before export
- [ ] Supports multi-commodity packages (single operator, multiple commodities)
- [ ] Generates package completeness certificate with digital signature

**Non-Functional Requirements:**
- Performance: Package assembly < 60 seconds for standard package (up to 100 suppliers)
- Integrity: 100% SHA-256 hash verification on all artifacts
- Storage: Packages stored in S3 with 5-year retention per Article 31

**Dependencies:**
- Features 1-6 (all data collection engines)
- S3 Object Storage (INFRA-004)
- PDF generation library

**Estimated Effort:** 3 weeks (1 backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 8: Intelligent Data Source Recommendation
- Recommend additional data sources based on commodity and country of origin
- Suggest alternative sources when primary sources are unavailable
- Rank data sources by authority and reliability

#### Feature 9: Historical Information Trending
- Track information changes over time per supplier/product
- Detect significant changes in supplier declarations
- Generate information drift alerts

#### Feature 10: Multi-Language Document Processing
- Process supplier documents in 10+ languages
- Extract structured data from non-English certificates and permits
- Translate key fields to operator's preferred language

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Real-time streaming data ingestion (batch/scheduled model sufficient for v1.0)
- Blockchain-based evidence immutability (SHA-256 provenance chains provide sufficient integrity)
- AI-powered data extraction from unstructured documents (defer to AGENT-DATA-001)
- Direct integration with national customs systems via X.400/EDIFACT (REST API adapters only)
- Automated supplier negotiation or engagement workflows
- Invoice payment or financial transaction processing

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
+-------v-----------+           +-------------v-------------+           +-----------v-----------+
| AGENT-EUDR-027    |           | Downstream EUDR Agents    |           | External Databases    |
| Information       |---------->| EUDR-016..025 Risk Assess |           |                       |
| Gathering Agent   |           | EUDR-026 DD Orchestrator  |           | EU TRACES, CITES,     |
|                   |           +---------------------------+           | FLEGT, COMTRADE,      |
| Engines:          |                                                   | FSC, RSPO, PEFC,      |
| 1. ExtDB Connect  |<---------------------------------------------------------| GFW, FAO, WGI, CPI   |
| 2. Cert Verify    |                                                   +-----------------------+
| 3. Public Data    |
| 4. Supplier Agg   |           +---------------------------+
| 5. Completeness   |<----------| Upstream AGENT-DATA       |
| 6. Normalization  |           | DATA-001 PDF Extractor    |
| 7. Package Asm    |           | DATA-002 Excel/CSV        |
+-------------------+           | DATA-003 ERP Connector    |
                                | DATA-008 Questionnaire    |
                                +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/information_gathering/
    __init__.py                              # Public API exports
    config.py                                # InformationGatheringConfig with GL_EUDR_IGA_ env prefix
    models.py                                # Pydantic v2 models for queries, results, packages
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains
    metrics.py                               # 18 Prometheus self-monitoring metrics
    external_database_connector.py           # ExternalDatabaseConnectorEngine (Feature 1)
    certification_verification_engine.py     # CertificationVerificationEngine (Feature 2)
    public_data_mining_engine.py             # PublicDataMiningEngine (Feature 3)
    supplier_information_aggregator.py       # SupplierInformationAggregator (Feature 4)
    information_completeness_validator.py    # InformationCompletenessValidator (Feature 5)
    data_normalization_engine.py             # DataNormalizationEngine (Feature 6)
    information_package_assembler.py         # InformationPackageAssembler (Feature 7)
    setup.py                                 # InformationGatheringService facade + get_service()
    api/
        __init__.py
        router.py                            # FastAPI router (25+ endpoints)
```

### 7.3 Data Models (Key Entities)

```python
# External Database Query
class ExternalDatabaseSource(str, Enum):
    EU_TRACES = "eu_traces"
    CITES = "cites"
    FLEGT_VPA = "flegt_vpa"
    UN_COMTRADE = "un_comtrade"
    FAO_STAT = "fao_stat"
    GLOBAL_FOREST_WATCH = "global_forest_watch"
    WORLD_BANK_WGI = "world_bank_wgi"
    TRANSPARENCY_CPI = "transparency_cpi"
    EU_SANCTIONS = "eu_sanctions"
    NATIONAL_CUSTOMS = "national_customs"
    NATIONAL_LAND_REGISTRY = "national_land_registry"

class QueryResult(BaseModel):
    query_id: str
    source: ExternalDatabaseSource
    query_parameters: Dict[str, Any]
    status: QueryStatus              # SUCCESS, PARTIAL, FAILED, CACHED
    records: List[Dict[str, Any]]
    record_count: int
    query_timestamp: datetime
    response_time_ms: int
    cached: bool
    cache_age_seconds: Optional[int]
    provenance_hash: str             # SHA-256

# Certification Verification
class CertificationBody(str, Enum):
    FSC = "fsc"
    RSPO = "rspo"
    PEFC = "pefc"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    UTZ = "utz"
    EU_ORGANIC = "eu_organic"

class CertificateVerificationResult(BaseModel):
    certificate_id: str
    certification_body: CertificationBody
    holder_name: str
    verification_status: CertVerificationStatus  # VALID, EXPIRED, SUSPENDED, NOT_FOUND, ERROR
    valid_from: Optional[datetime]
    valid_until: Optional[datetime]
    scope: List[str]                  # e.g., ["manufacturing", "trading"]
    commodity_scope: List[EUDRCommodity]
    chain_of_custody_model: Optional[str]
    days_until_expiry: Optional[int]
    last_verified: datetime
    provenance_hash: str

# Supplier Profile
class SupplierProfile(BaseModel):
    supplier_id: str
    name: str
    alternative_names: List[str]
    postal_address: str
    country_code: str
    email: Optional[str]
    registration_number: Optional[str]
    commodities: List[EUDRCommodity]
    certifications: List[CertificateVerificationResult]
    plot_ids: List[str]
    tier_depth: int
    data_sources: List[str]           # Which systems contributed data
    completeness_score: Decimal
    confidence_score: Decimal
    discrepancies: List[DataDiscrepancy]
    last_updated: datetime
    provenance_hash: str

# Information Package
class InformationPackage(BaseModel):
    package_id: str
    operator_id: str
    commodity: EUDRCommodity
    version: int
    article_9_elements: Dict[str, Article9ElementStatus]
    completeness_score: Decimal
    completeness_classification: str   # INSUFFICIENT, PARTIAL, COMPLETE
    supplier_profiles: List[SupplierProfile]
    external_data: Dict[str, List[QueryResult]]
    certification_results: List[CertificateVerificationResult]
    public_data: Dict[str, Any]
    normalization_log: List[NormalizationRecord]
    gap_report: GapReport
    evidence_artifacts: List[EvidenceArtifact]
    provenance_chain: List[ProvenanceEntry]
    package_hash: str                  # SHA-256 of entire package
    assembled_at: datetime
    valid_until: datetime              # Data freshness expiry
```

### 7.4 Database Schema (New Migration: V114)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_information_gathering;

-- Information gathering operations
CREATE TABLE eudr_information_gathering.gathering_operations (
    operation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    workflow_id UUID,
    status VARCHAR(30) NOT NULL DEFAULT 'initiated',
    sources_queried JSONB DEFAULT '[]',
    sources_completed JSONB DEFAULT '[]',
    sources_failed JSONB DEFAULT '[]',
    completeness_score NUMERIC(5,2) DEFAULT 0.0,
    completeness_classification VARCHAR(30),
    total_records_collected INTEGER DEFAULT 0,
    total_suppliers_resolved INTEGER DEFAULT 0,
    total_certificates_verified INTEGER DEFAULT 0,
    package_id UUID,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- External database query log (hypertable)
CREATE TABLE eudr_information_gathering.external_query_log (
    query_id UUID DEFAULT gen_random_uuid(),
    operation_id UUID NOT NULL,
    source VARCHAR(50) NOT NULL,
    query_parameters JSONB NOT NULL,
    status VARCHAR(20) NOT NULL,
    record_count INTEGER DEFAULT 0,
    response_time_ms INTEGER,
    cached BOOLEAN DEFAULT FALSE,
    cache_age_seconds INTEGER,
    error_message TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    queried_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_information_gathering.external_query_log', 'queried_at');

-- Certification verification records (hypertable)
CREATE TABLE eudr_information_gathering.certification_verifications (
    verification_id UUID DEFAULT gen_random_uuid(),
    operation_id UUID,
    certificate_id VARCHAR(200) NOT NULL,
    certification_body VARCHAR(50) NOT NULL,
    holder_name VARCHAR(500),
    verification_status VARCHAR(30) NOT NULL,
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    scope JSONB DEFAULT '[]',
    commodity_scope JSONB DEFAULT '[]',
    chain_of_custody_model VARCHAR(50),
    days_until_expiry INTEGER,
    provenance_hash VARCHAR(64) NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_information_gathering.certification_verifications', 'verified_at');

-- Public data harvest records (hypertable)
CREATE TABLE eudr_information_gathering.public_data_harvests (
    harvest_id UUID DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    country_code CHAR(2),
    commodity VARCHAR(50),
    records_harvested INTEGER DEFAULT 0,
    data_timestamp TIMESTAMPTZ,
    is_incremental BOOLEAN DEFAULT FALSE,
    freshness_status VARCHAR(20) DEFAULT 'fresh',
    provenance_hash VARCHAR(64) NOT NULL,
    harvested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_information_gathering.public_data_harvests', 'harvested_at');

-- Supplier profiles (resolved)
CREATE TABLE eudr_information_gathering.supplier_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id VARCHAR(200) NOT NULL,
    name VARCHAR(500) NOT NULL,
    alternative_names JSONB DEFAULT '[]',
    postal_address TEXT,
    country_code CHAR(2),
    email VARCHAR(500),
    registration_number VARCHAR(200),
    commodities JSONB DEFAULT '[]',
    certifications JSONB DEFAULT '[]',
    plot_ids JSONB DEFAULT '[]',
    tier_depth INTEGER DEFAULT 0,
    data_sources JSONB DEFAULT '[]',
    completeness_score NUMERIC(5,2) DEFAULT 0.0,
    confidence_score NUMERIC(5,2) DEFAULT 0.0,
    discrepancies JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(supplier_id)
);

-- Information packages
CREATE TABLE eudr_information_gathering.information_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    version INTEGER DEFAULT 1,
    article_9_elements JSONB NOT NULL DEFAULT '{}',
    completeness_score NUMERIC(5,2) NOT NULL,
    completeness_classification VARCHAR(30) NOT NULL,
    supplier_count INTEGER DEFAULT 0,
    evidence_count INTEGER DEFAULT 0,
    gap_count INTEGER DEFAULT 0,
    s3_path VARCHAR(1000),
    package_hash VARCHAR(64) NOT NULL,
    valid_until TIMESTAMPTZ,
    assembled_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Normalization audit log (hypertable)
CREATE TABLE eudr_information_gathering.normalization_log (
    log_id UUID DEFAULT gen_random_uuid(),
    operation_id UUID,
    field_name VARCHAR(200) NOT NULL,
    source_value TEXT NOT NULL,
    normalized_value TEXT NOT NULL,
    normalization_type VARCHAR(50) NOT NULL,
    confidence NUMERIC(5,4) DEFAULT 1.0,
    normalized_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_information_gathering.normalization_log', 'normalized_at');

-- Data freshness tracking
CREATE TABLE eudr_information_gathering.data_freshness (
    freshness_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL,
    next_expected_update TIMESTAMPTZ,
    freshness_status VARCHAR(20) NOT NULL DEFAULT 'fresh',
    max_age_hours INTEGER NOT NULL DEFAULT 24,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(source, data_type)
);

-- Indexes
CREATE INDEX idx_operations_operator ON eudr_information_gathering.gathering_operations(operator_id);
CREATE INDEX idx_operations_commodity ON eudr_information_gathering.gathering_operations(commodity);
CREATE INDEX idx_operations_status ON eudr_information_gathering.gathering_operations(status);
CREATE INDEX idx_queries_operation ON eudr_information_gathering.external_query_log(operation_id);
CREATE INDEX idx_queries_source ON eudr_information_gathering.external_query_log(source);
CREATE INDEX idx_certs_body ON eudr_information_gathering.certification_verifications(certification_body);
CREATE INDEX idx_certs_status ON eudr_information_gathering.certification_verifications(verification_status);
CREATE INDEX idx_certs_holder ON eudr_information_gathering.certification_verifications(holder_name);
CREATE INDEX idx_harvests_source ON eudr_information_gathering.public_data_harvests(source);
CREATE INDEX idx_harvests_country ON eudr_information_gathering.public_data_harvests(country_code);
CREATE INDEX idx_profiles_supplier ON eudr_information_gathering.supplier_profiles(supplier_id);
CREATE INDEX idx_profiles_country ON eudr_information_gathering.supplier_profiles(country_code);
CREATE INDEX idx_packages_operator ON eudr_information_gathering.information_packages(operator_id);
CREATE INDEX idx_packages_commodity ON eudr_information_gathering.information_packages(commodity);
CREATE INDEX idx_freshness_source ON eudr_information_gathering.data_freshness(source);
```

### 7.5 API Endpoints (25+)

| Method | Path | Description |
|--------|------|-------------|
| **Gathering Operations** | | |
| POST | `/v1/gathering/operations` | Initiate a new information gathering operation |
| GET | `/v1/gathering/operations` | List gathering operations (with filters) |
| GET | `/v1/gathering/operations/{operation_id}` | Get operation status and progress |
| POST | `/v1/gathering/operations/{operation_id}/execute` | Execute gathering for all configured sources |
| **External Database Queries** | | |
| POST | `/v1/gathering/external/query` | Query a specific external database |
| POST | `/v1/gathering/external/batch-query` | Batch query multiple databases |
| GET | `/v1/gathering/external/sources` | List available external database sources |
| GET | `/v1/gathering/external/sources/{source}/status` | Get connection status for a source |
| **Certification Verification** | | |
| POST | `/v1/gathering/certifications/verify` | Verify a single certificate |
| POST | `/v1/gathering/certifications/batch-verify` | Batch verify up to 1,000 certificates |
| GET | `/v1/gathering/certifications/expiring` | List certificates expiring within N days |
| GET | `/v1/gathering/certifications/supplier/{supplier_id}` | Get all certifications for a supplier |
| **Public Data** | | |
| POST | `/v1/gathering/public-data/harvest` | Trigger data harvest from configured sources |
| GET | `/v1/gathering/public-data/freshness` | Get data freshness status across all sources |
| GET | `/v1/gathering/public-data/{source}/latest` | Get latest harvested data from a source |
| **Supplier Aggregation** | | |
| POST | `/v1/gathering/suppliers/aggregate` | Aggregate supplier information from all sources |
| GET | `/v1/gathering/suppliers/{supplier_id}/profile` | Get unified supplier profile |
| GET | `/v1/gathering/suppliers/discrepancies` | List supplier data discrepancies |
| POST | `/v1/gathering/suppliers/resolve/{supplier_id}` | Resolve entity conflict for supplier |
| **Completeness Validation** | | |
| POST | `/v1/gathering/completeness/validate` | Validate information completeness |
| GET | `/v1/gathering/completeness/{operation_id}/report` | Get completeness report |
| GET | `/v1/gathering/completeness/{operation_id}/gaps` | Get gap report with remediation actions |
| **Information Packages** | | |
| POST | `/v1/gathering/packages/assemble` | Assemble information package |
| GET | `/v1/gathering/packages/{package_id}` | Get package details |
| GET | `/v1/gathering/packages/{package_id}/download` | Download package (JSON/PDF) |
| GET | `/v1/gathering/packages/{package_id}/verify` | Verify package integrity (hash check) |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_iga_gathering_operations_total` | Counter | Information gathering operations initiated |
| 2 | `gl_eudr_iga_external_queries_total` | Counter | External database queries executed by source |
| 3 | `gl_eudr_iga_certifications_verified_total` | Counter | Certificates verified by body and status |
| 4 | `gl_eudr_iga_public_data_harvests_total` | Counter | Public data harvests completed by source |
| 5 | `gl_eudr_iga_suppliers_aggregated_total` | Counter | Supplier profiles aggregated |
| 6 | `gl_eudr_iga_completeness_validations_total` | Counter | Completeness validations performed |
| 7 | `gl_eudr_iga_packages_assembled_total` | Counter | Information packages assembled |
| 8 | `gl_eudr_iga_api_errors_total` | Counter | API errors by operation type |
| 9 | `gl_eudr_iga_external_query_duration_seconds` | Histogram | External database query latency by source |
| 10 | `gl_eudr_iga_certification_verification_duration_seconds` | Histogram | Certificate verification latency |
| 11 | `gl_eudr_iga_harvest_duration_seconds` | Histogram | Public data harvest latency by source |
| 12 | `gl_eudr_iga_aggregation_duration_seconds` | Histogram | Supplier aggregation latency |
| 13 | `gl_eudr_iga_package_assembly_duration_seconds` | Histogram | Package assembly latency |
| 14 | `gl_eudr_iga_active_operations` | Gauge | Currently active gathering operations |
| 15 | `gl_eudr_iga_stale_data_sources` | Gauge | Data sources exceeding freshness threshold |
| 16 | `gl_eudr_iga_expiring_certificates` | Gauge | Certificates expiring within 90 days |
| 17 | `gl_eudr_iga_cache_hit_ratio` | Gauge | External query cache hit ratio |
| 18 | `gl_eudr_iga_normalization_errors` | Counter | Data normalization errors by type |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables |
| Cache | Redis | External query caching, certificate cache |
| Object Storage | S3 | Information package storage (5-year retention) |
| HTTP Client | httpx | Async HTTP for external database queries |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| Secrets | HashiCorp Vault via SEC-006 | External API credentials management |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-iga:operations:read` | View gathering operations | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-iga:operations:write` | Create and execute gathering operations | Analyst, Compliance Officer, Admin |
| `eudr-iga:external:query` | Execute external database queries | Analyst, Compliance Officer, Admin |
| `eudr-iga:external:status` | View external source connection status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-iga:certifications:verify` | Verify certificates | Analyst, Compliance Officer, Admin |
| `eudr-iga:certifications:read` | View certification verification results | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-iga:public-data:harvest` | Trigger public data harvests | Analyst, Compliance Officer, Admin |
| `eudr-iga:public-data:read` | View harvested public data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-iga:suppliers:aggregate` | Execute supplier aggregation | Analyst, Compliance Officer, Admin |
| `eudr-iga:suppliers:read` | View supplier profiles | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-iga:suppliers:resolve` | Resolve entity conflicts | Compliance Officer, Admin |
| `eudr-iga:completeness:validate` | Execute completeness validation | Analyst, Compliance Officer, Admin |
| `eudr-iga:completeness:read` | View completeness reports | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-iga:packages:assemble` | Assemble information packages | Compliance Officer, Admin |
| `eudr-iga:packages:read` | View and download packages | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-iga:packages:verify` | Verify package integrity | Auditor, Compliance Officer, Admin |
| `eudr-iga:normalization:read` | View normalization audit logs | Analyst, Compliance Officer, Admin |
| `eudr-iga:audit:read` | View full audit trail | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent/System | Integration | Data Flow |
|--------------|-------------|-----------|
| AGENT-DATA-001 PDF & Invoice Extractor | Document extract ingestion | Extracted invoice/certificate fields -> supplier aggregation |
| AGENT-DATA-002 Excel/CSV Normalizer | Spreadsheet import | Normalized supplier lists -> supplier aggregation |
| AGENT-DATA-003 ERP/Finance Connector | ERP procurement records | Procurement data, supplier records -> supplier aggregation |
| AGENT-DATA-008 Supplier Questionnaire | Questionnaire responses | Supplier self-declared data -> supplier aggregation |
| AGENT-EUDR-015 Mobile Data Collector | Mobile field data | GPS, photos, field observations -> supplier profiles |
| EU TRACES API | Trade control system | Phytosanitary certificates, import permits |
| CITES Trade Database | Endangered species permits | Trade permits, species verification |
| FLEGT/VPA System | Forest law enforcement | FLEGT licenses, VPA partner status |
| FSC/RSPO/PEFC/RA APIs | Certification databases | Certificate verification results |
| FAO/COMTRADE/GFW | Public statistics | Production, trade, deforestation data |
| SEC-006 Vault | Secrets management | API credentials for external databases |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-016 Country Risk Evaluator | Information feed | Country-level data, governance indices -> risk scoring |
| AGENT-EUDR-017 Supplier Risk Scorer | Information feed | Supplier profiles, certification status -> supplier risk |
| AGENT-EUDR-018 Commodity Risk Analyzer | Information feed | Trade statistics, production data -> commodity risk |
| AGENT-EUDR-019 Corruption Index Monitor | Information feed | CPI scores, governance data -> corruption monitoring |
| AGENT-EUDR-020 Deforestation Alert System | Information feed | GFW alerts, satellite data references -> deforestation monitoring |
| AGENT-EUDR-026 DD Orchestrator | Package handoff | Complete information packages -> orchestrated due diligence |
| GL-EUDR-APP v1.0 | API integration | Information status, packages -> frontend display |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Automated Information Gathering (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Information Gathering" module
3. Clicks "New Gathering Operation" -> selects commodity (e.g., Cocoa)
4. System auto-detects relevant external databases for cocoa
5. System queries EU TRACES, CITES, COMTRADE for operator's import records
6. System verifies all supplier certificates (Rainforest Alliance, UTZ)
7. System harvests latest FAO production data for sourcing countries
8. System aggregates supplier information from ERP + questionnaires
9. Progress bar shows real-time status per data source
10. System normalizes all collected data into standard formats
11. System validates completeness against Article 9 checklist
12. Gap report shows: "2 suppliers missing geolocation data"
13. Officer sends remediation requests via linked supplier onboarding
14. System assembles evidence package when completeness >= 90%
15. Officer downloads package for DDS submission
```

#### Flow 2: Certificate Batch Verification (Certification Manager)

```
1. Certification manager uploads CSV with 500 supplier certificates
2. System parses CSV, identifies certification body per certificate
3. System launches batch verification across FSC, RSPO, PEFC, RA
4. Progress shows: 450 valid, 30 expired, 15 suspended, 5 not found
5. System generates certificate compliance matrix per supplier
6. Manager exports expired certificate list for supplier follow-up
7. System schedules automatic re-verification in 24 hours
```

#### Flow 3: Cross-Reference Check (Data Analyst)

```
1. Analyst selects supplier "PT ABC Trading" for cross-reference
2. System queries EU TRACES for supplier's import history
3. System queries sanctions lists for entity screening
4. System retrieves COMTRADE trade flow data for supplier's country
5. System compares supplier self-declared volumes with customs records
6. Discrepancy detected: declared production volume 20% higher than COMTRADE data
7. System flags discrepancy with recommendation for investigation
8. Analyst marks finding for compliance officer review
```

### 8.2 Key Screen Descriptions

**Information Gathering Dashboard:**
- Source status cards: green/yellow/red indicators per external database
- Active operations table with progress bars
- Certificate verification summary: valid/expired/suspended counts
- Data freshness matrix: last update time per source
- Completeness score trend chart

**Evidence Package Viewer:**
- Article 9 checklist with per-element status (complete/partial/missing)
- Expandable evidence sections per element
- Provenance chain visualization: source -> normalization -> assembly
- Package integrity verification badge (SHA-256)
- Download buttons: JSON, PDF, XML (EU Information System format)

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 7 P0 features (Features 1-7) implemented and tested
  - [ ] Feature 1: External Database Connector Engine -- 6+ databases connected
  - [ ] Feature 2: Certification Verification Engine -- 6 certification bodies verified
  - [ ] Feature 3: Public Data Mining Engine -- 8+ public sources harvested
  - [ ] Feature 4: Supplier Information Aggregator -- 5+ input formats supported
  - [ ] Feature 5: Information Completeness Validator -- all 10 Article 9 elements
  - [ ] Feature 6: Data Normalization Engine -- units, dates, coordinates, currencies
  - [ ] Feature 7: Information Package Assembler -- JSON, PDF, XML export
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated)
- [ ] Performance targets met (< 5s external query, < 60s package assembly)
- [ ] All 7 commodity types tested with golden test fixtures
- [ ] Completeness scoring verified deterministic (bit-perfect reproducibility)
- [ ] API documentation complete (OpenAPI spec)
- [ ] Database migration V114 tested and validated
- [ ] Integration with AGENT-DATA-001/002/003/008 verified
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ gathering operations completed by customers
- 5,000+ certificates verified
- < 5% external database query failures
- Average completeness score >= 80% per operation

**60 Days:**
- 200+ gathering operations active
- 20,000+ certificates verified
- All 10+ external databases operational with < 2% failure rate
- Average completeness score >= 85%

**90 Days:**
- 500+ information packages assembled
- 50,000+ certificates verified
- Average package assembly time < 45 seconds
- Zero compliance penalties attributable to incomplete information gathering

---

## 10. Timeline and Milestones

### Phase 1: Core Collection Engines (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | External Database Connector Engine (Feature 1): adapter pattern, 6+ connectors | Backend Engineer |
| 2-3 | Certification Verification Engine (Feature 2): 6 certification body connectors | Backend Engineer |
| 3-4 | Public Data Mining Engine (Feature 3): 8+ source harvesters | Data Engineer |
| 4-5 | Supplier Information Aggregator (Feature 4): multi-source normalization | Backend Engineer |
| 5-6 | Data Normalization Engine (Feature 6): units, dates, coordinates | Backend Engineer |

### Phase 2: Validation, Assembly, and API (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Information Completeness Validator (Feature 5): Article 9 validation | Backend Engineer |
| 8-9 | Information Package Assembler (Feature 7): JSON, PDF, XML export | Backend Engineer |
| 9-10 | REST API Layer: 25+ endpoints, authentication, rate limiting | Backend Engineer |

### Phase 3: Integration and Testing (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | AGENT-DATA integration: DATA-001/002/003/008 data flow testing | Integration Engineer |
| 12-13 | Downstream EUDR agent integration: EUDR-016..026 data handoff | Integration Engineer |
| 13-14 | Complete test suite: 500+ tests, golden tests for all 7 commodities | Test Engineer |

### Phase 4: Security, Performance, and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Security audit, RBAC integration, Vault credential management | Security Engineer |
| 16-17 | Performance testing, load testing, external API resilience testing | DevOps |
| 17 | Database migration V114 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding and launch readiness review | All |

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-DATA-001 PDF & Invoice Extractor | BUILT (100%) | Low | Stable, production-ready |
| AGENT-DATA-002 Excel/CSV Normalizer | BUILT (100%) | Low | Stable, production-ready |
| AGENT-DATA-003 ERP/Finance Connector | BUILT (100%) | Low | Stable, production-ready |
| AGENT-DATA-008 Supplier Questionnaire Processor | BUILT (100%) | Low | Stable, production-ready |
| AGENT-EUDR-015 Mobile Data Collector | BUILT (100%) | Low | Stable, production-ready |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| SEC-006 Vault Secrets Management | BUILT (100%) | Low | Credential storage for external APIs |
| INFRA-002 PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| INFRA-003 Redis Caching | Production Ready | Low | Standard infrastructure |
| INFRA-004 S3 Object Storage | Production Ready | Low | Standard infrastructure |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU TRACES API | Available | Medium | Adapter pattern; cached fallback |
| CITES Trade Database API | Available | Medium | Adapter pattern; cached fallback |
| FLEGT/VPA Licensing System | Available | Medium | Adapter pattern; cached fallback |
| FSC Certificate Database | Available | Low | Well-documented REST API |
| RSPO PalmTrace API | Available | Low | Well-documented REST API |
| PEFC Certificate Search | Available | Low | Web scraping adapter as fallback |
| Rainforest Alliance API | Available | Medium | Adapter pattern; cached fallback |
| UN COMTRADE API | Available | Low | Well-documented REST API |
| FAO STAT API | Available | Low | Well-documented REST API |
| Global Forest Watch API | Available | Low | Well-documented REST API |
| ECB Exchange Rate API | Available | Low | Free, reliable, well-documented |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | External API downtime or rate limiting | High | Medium | Circuit breaker, cached fallback, exponential backoff |
| R2 | External API specification changes | Medium | High | Adapter pattern isolates changes; version-pinned connectors |
| R3 | Certification body API access restrictions | Medium | Medium | Pre-negotiate API access; web scraping fallback |
| R4 | Data quality from external sources | Medium | High | Validation rules, confidence scoring, source authority ranking |
| R5 | Entity resolution false positives/negatives | Medium | Medium | Configurable matching thresholds, manual override capability |
| R6 | EU regulatory changes to Article 9 requirements | Low | High | Configuration-driven validation rules; modular element checklist |
| R7 | Large data volumes for batch operations | Medium | Medium | Pagination, streaming, configurable batch sizes |
| R8 | Cross-border data privacy regulations | Medium | Medium | Data minimization, GDPR compliance, regional data processing |
| R9 | Credential compromise for external APIs | Low | High | Vault rotation, audit logging, principle of least privilege |
| R10 | Performance degradation under concurrent load | Medium | Medium | Connection pooling, caching, horizontal scaling |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| External Connector Unit Tests | 80+ | Adapter pattern, query construction, response parsing, error handling |
| Certification Engine Tests | 70+ | All 6 certification bodies, batch verification, lifecycle tracking |
| Public Data Mining Tests | 60+ | All 8+ sources, incremental harvesting, freshness monitoring |
| Supplier Aggregation Tests | 80+ | Multi-source normalization, entity resolution, deduplication |
| Completeness Validation Tests | 60+ | All 10 elements, commodity-specific rules, scoring determinism |
| Data Normalization Tests | 70+ | Units, dates, coordinates, currencies, edge cases |
| Package Assembly Tests | 50+ | JSON, PDF, XML export, provenance chains, hash verification |
| API Tests | 80+ | All 25+ endpoints, auth, error handling, pagination |
| Integration Tests | 30+ | Cross-agent data flow with DATA-001/002/003/008, EUDR-016..026 |
| Golden Tests | 50+ | All 7 commodities, complete/partial/missing scenarios |
| Performance Tests | 20+ | Concurrent operations, large batch processing, external API latency |
| **Total** | **650+** | |

### 13.2 Golden Test Commodities

Each of the 7 commodities will have golden test fixtures:
1. Complete information (all Article 9 elements present) -- expect completeness = 100%
2. Partial information (some elements missing) -- expect correct gap identification
3. Missing critical elements -- expect INSUFFICIENT classification
4. Contradictory data sources -- expect discrepancy detection
5. Expired certificates -- expect expiry flagging
6. Low-risk country (simplified DD) -- expect reduced element requirements
7. Multi-country sourcing -- expect per-country validation

Total: 7 commodities x 7 scenarios = 49 golden test scenarios

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **Article 9** | EUDR article specifying the information that operators must collect |
| **EU TRACES** | Trade Control and Expert System -- EU tool for sanitary/phytosanitary trade certification |
| **CITES** | Convention on International Trade in Endangered Species |
| **FLEGT** | Forest Law Enforcement, Governance and Trade |
| **VPA** | Voluntary Partnership Agreement under FLEGT |
| **FSC** | Forest Stewardship Council |
| **RSPO** | Roundtable on Sustainable Palm Oil |
| **PEFC** | Programme for the Endorsement of Forest Certification |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |
| **WGI** | Worldwide Governance Indicators (World Bank) |
| **CPI** | Corruption Perceptions Index (Transparency International) |

### Appendix B: Article 9 Cross-Reference Matrix

| Article 9 Element | External Sources | Internal Sources | Validation Rules |
|-------------------|-----------------|-----------------|-----------------|
| Product description | EU TRACES, customs | ERP, invoices | CN/HS code format, commodity match |
| Quantity | Customs, COMTRADE | Bills of lading, invoices | Positive numeric, unit conversion |
| Country of production | Certificates of origin, customs | Supplier declarations | Valid ISO 3166-1, not sanctioned |
| Geolocation | Land registries, satellite | Supplier GPS submissions | WGS84, valid ranges, polygon > 4 ha |
| Production date | Phytosanitary certs | Harvest records, processing logs | Not future, within 5 years |
| Supplier info | Business registries, customs | Questionnaires, ERP | Name + address + email required |
| Buyer info | Trade documents | ERP records | Name + address required |
| Deforestation-free | GFW, satellite data | Certification records | Post-2020 cutoff verification |
| Legal production | FLEGT, phytosanitary | Export permits, certificates | Valid license, not expired |
| Supply chain | COMTRADE, customs | ERP, questionnaires | Traceability chain complete |

### Appendix C: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the Information System
4. EU TRACES Technical Documentation (DG SANTE)
5. CITES Trade Database Technical Guide
6. FLEGT/VPA Implementation Framework
7. FSC Certificate Database API Documentation
8. RSPO PalmTrace API Specification
9. UN COMTRADE API Documentation
10. FAO STAT API Documentation
11. Global Forest Watch API Documentation

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-11 | APPROVED |
| Engineering Lead | GL-EngineeringLead | 2026-03-11 | APPROVED |
| EUDR Regulatory Advisor | GL-RegulatoryAdvisor | 2026-03-11 | APPROVED |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-03-11 | GL-ProductManager | Initial version: 7 P0 features, 10 Article 9 elements, 14 external sources, 18 metrics, V114 migration, 18 RBAC permissions |