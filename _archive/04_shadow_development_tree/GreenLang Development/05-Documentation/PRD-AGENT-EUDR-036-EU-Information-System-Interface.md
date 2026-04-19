# PRD: AGENT-EUDR-036 -- EU Information System Interface Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-036 |
| **Agent ID** | GL-EUDR-EUIS-036 |
| **Component** | EU Information System Interface Agent |
| **Category** | EUDR Regulatory Agent -- Reporting (Category 6) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-13 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 4 (Due Diligence Obligations), 9 (Information/Geolocation), 12 (Due Diligence Statements), 14 (Operator Registration), 31 (Record Keeping), 33 (EU Information System); Commission Implementing Regulation (EU) 2024/XXX (Technical Specifications for the Information System); eIDAS Regulation (EU) No 910/2014 (Electronic Identification) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Prerequisite Agents** | AGENT-EUDR-001 (Supply Chain Mapping Master), AGENT-EUDR-006 (Plot Boundary Manager), AGENT-EUDR-007 (GPS Coordinate Validator), AGENT-EUDR-030 (Documentation Generator) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 33 mandates the establishment of an EU Information System through which operators and traders must submit Due Diligence Statements (DDS), register as operators with competent authorities, and provide geolocation data for all production plots. Article 4 requires that no relevant commodity or product shall be placed on the EU market unless a DDS has been submitted to this Information System. Article 12 specifies the submission requirements for DDS documents, including format specifications, mandatory data fields, and the assignment of reference numbers by the system. Article 14 requires operators to register with the competent authority in the Member State where they are established, obtaining a registration number that must be referenced in all DDS submissions. Article 9(1)(d) mandates that geolocation data for production plots be provided in a specific format: WGS 84 coordinate reference system with sufficient decimal precision for regulatory purposes.

The GreenLang platform has built a comprehensive suite of 35 EUDR agents spanning the full due diligence lifecycle. Supply Chain Traceability agents (EUDR-001 through EUDR-015) handle supply chain mapping, geolocation verification, satellite monitoring, forest cover analysis, land use change detection, plot boundary management, GPS coordinate validation, multi-tier supplier tracking, chain of custody, segregation verification, mass balance calculation, document authentication, blockchain integration, QR code generation, and mobile data collection. Risk Assessment agents (EUDR-016 through EUDR-020) handle country risk evaluation, supplier risk scoring, commodity risk analysis, corruption index monitoring, and deforestation alerting. Due Diligence agents (EUDR-021 through EUDR-035) handle indigenous rights checking, protected area validation, legal compliance verification, third-party audit management, risk mitigation advisory, due diligence orchestration, information gathering coordination, risk assessment engine operation, mitigation measure design, documentation generation, stakeholder engagement, grievance mechanism management, continuous monitoring, annual review scheduling, and improvement plan creation.

Critically, EUDR-030 (Documentation Generator) produces submission-ready DDS documents in JSON and XML formats, assembles Article 9 information packages, builds compliance packages, and manages document versioning. However, EUDR-030 generates these documents internally within the GreenLang platform. There is currently no agent that handles the external interface to the EU Information System itself -- the actual transmission, registration, format compliance, status tracking, and audit logging required by Articles 4, 12, 14, and 33. Operators face the following critical gaps:

- **No direct EU Information System API integration**: The EU Information System exposes a set of APIs for DDS submission, operator registration, status queries, and geolocation data upload. These APIs require specific authentication (eIDAS-compatible electronic identification), rate limiting compliance, payload formatting, error handling, and retry logic. EUDR-030 produces the documents but cannot transmit them. Operators must manually log into the EU portal, upload each document individually, and record acknowledgement numbers -- a process that takes 15-30 minutes per DDS and does not scale for large operators submitting hundreds of DDS documents per quarter.

- **No operator registration lifecycle management**: Article 14 requires operators to register with the competent authority of the Member State where they are established. Registration involves submitting operator identification data (legal name, address, EORI number, economic activity codes), receiving a registration number, and maintaining that registration through annual renewals, updates when company details change, and re-registration when expanding to new Member States. There is no engine that manages the operator registration lifecycle, tracks registration status across multiple Member States, alerts on expiring registrations, or ensures that all DDS submissions reference valid, current registration numbers.

- **No geolocation data formatting to EU specifications**: Article 9(1)(d) requires geolocation data in the WGS 84 coordinate reference system with at least 6 decimal places of precision (approximately 0.11 meter accuracy). The EU Information System expects geolocation data in specific JSON structures: single-point coordinates as [latitude, longitude] arrays and polygon coordinates as arrays of coordinate pairs forming closed rings. Plots from different upstream agents (EUDR-002, EUDR-006, EUDR-007) may use varying coordinate formats, precisions, and reference systems. There is no formatting engine that normalizes all geolocation data to the exact EU specification, validates coordinate precision, handles coordinate reference system transformations, and structures multi-plot commodities correctly.

- **No document package assembly for EU transmission**: The EU Information System imposes size limits on individual submissions (typically 10-50 MB depending on the endpoint). Large compliance packages from EUDR-030 (which can reach 500 MB for complex supply chains with satellite imagery evidence) cannot be submitted as-is. They must be decomposed into appropriately sized segments, compressed, manifested, and transmitted in the correct order. There is no engine that handles document package decomposition, compression, manifest generation, multi-part upload, and reassembly confirmation.

- **No DDS submission status tracking**: After a DDS is submitted to the EU Information System, it enters a lifecycle managed by the system: received, under review, accepted, rejected with reasons, or requiring amendment. The EU Information System assigns a unique reference number to each accepted DDS. Currently, operators must manually log into the EU portal to check submission status, a process that provides no programmatic notification, no batch status monitoring, and no integration with the operator's compliance workflow. There is no engine that polls the EU Information System for status updates, records state transitions, triggers automated workflows on rejection (correction and resubmission), or maintains a synchronized view of all submission statuses.

- **No eIDAS-compliant authentication management**: The EU Information System uses eIDAS (electronic IDentification, Authentication and trust Services) for operator authentication. This involves digital certificates, qualified electronic signatures, and potentially multi-factor authentication flows. Managing eIDAS credentials, certificate renewals, signature delegation, and authentication session management across multiple Member State endpoints requires specialized handling that does not exist in the platform today.

- **No regulatory audit trail for EU submissions**: Article 31 requires operators to retain all due diligence records for at least 5 years. For EU Information System interactions, this means every submission attempt, every acknowledgement receipt, every rejection with reasons, every resubmission, every status change, and every registration event must be recorded in a tamper-proof audit log that can be presented to competent authorities during Article 14-16 inspections. There is no engine that maintains this submission-specific audit trail with the immutability, completeness, and 5-year retention guarantees required by the regulation.

- **No error recovery for transient EU IS failures**: The EU Information System, like any external government system, experiences outages, rate limiting, timeout errors, and intermittent failures. Large operators submitting hundreds of DDS documents need resilient submission pipelines that can queue submissions, retry on transient failures, handle partial batch failures, and provide clear status reporting. There is no resilience layer between the GreenLang platform and the EU Information System.

Without solving these problems, the entire due diligence workflow from EUDR-001 through EUDR-035 culminates in documents that cannot be officially submitted to the regulatory authority. The last-mile gap between document generation (EUDR-030) and regulatory acceptance is the single point of failure that renders all upstream compliance work legally incomplete. Operators who cannot submit DDS documents to the EU Information System cannot legally place products on the EU market, regardless of how thorough their due diligence process is.

### 1.2 Solution Overview

Agent-EUDR-036: EU Information System Interface is the external gateway agent that connects the GreenLang platform to the EU Information System mandated by EUDR Article 33. It receives submission-ready documents from EUDR-030 (Documentation Generator), formats them to exact EU specifications, authenticates with the EU Information System using eIDAS-compliant credentials, transmits DDS submissions, manages operator registrations, tracks submission lifecycles, handles rejections with automated resubmission workflows, and maintains a tamper-proof audit trail of all interactions with the EU Information System. It is the 36th agent in the EUDR agent family and establishes the Reporting sub-category (Category 6).

Core capabilities:

1. **DDS Submission Gateway** -- Receives finalized DDS documents from EUDR-030, validates them against the EU Information System's current schema version, packages them for transmission, authenticates with the EU IS API, submits them via the official API endpoints, captures acknowledgement reference numbers, and handles synchronous and asynchronous submission flows. Supports individual and batch submission with configurable concurrency. Implements pre-submission validation that mirrors the EU IS server-side validation to minimize rejection rates. Manages submission queuing for large operators with hundreds of quarterly DDS documents.

2. **Operator Registration Manager** -- Manages the complete operator registration lifecycle with EU Member State competent authorities per Article 14. Handles initial registration (submitting operator identification, receiving registration numbers), registration renewal (tracking expiry dates, submitting renewal applications), registration updates (notifying competent authorities of changes to operator details), multi-Member-State registration (operators established in or submitting from multiple EU countries), and registration validation (ensuring every DDS references a valid, current registration number). Tracks registration status across all applicable Member States with alerting for expiring or invalid registrations.

3. **Geolocation Data Formatter** -- Normalizes and formats all geolocation data from upstream agents (EUDR-002, EUDR-006, EUDR-007) to the exact specifications required by the EU Information System. Ensures WGS 84 coordinate reference system, enforces minimum 6 decimal places per Article 9, validates latitude/longitude ranges, structures single-point coordinates and polygon coordinates in the EU-specified JSON format, handles coordinate reference system transformations from non-WGS84 sources, validates polygon closure (first point equals last point), validates polygon winding order, and handles multi-plot commodities where a single product traces to dozens or hundreds of production plots across multiple countries.

4. **Document Package Assembler** -- Assembles and prepares document packages for EU Information System transmission. Handles document compression (gzip, deflate) to meet EU IS size limits, splits large packages into appropriately sized segments with manifests, generates package manifests listing all included documents with SHA-256 hashes, attaches supporting evidence documents (risk assessments, certificates, chain of custody records, contracts) in EU-accepted formats, validates package completeness against submission requirements, and manages multi-part upload workflows for large submissions.

5. **Submission Status Tracker** -- Tracks the complete lifecycle of every DDS submission from draft through final acceptance or rejection. Polls the EU Information System API for status updates at configurable intervals. Records every state transition: submitted, received, under_review, accepted, rejected, amendment_required. Captures EU IS reference numbers for accepted submissions. Handles rejection workflows: parses rejection reasons, generates correction guidance, triggers resubmission through EUDR-030 for document correction and back through the submission gateway. Provides real-time submission dashboards and batch status monitoring. Supports webhook notifications for status changes.

6. **API Integration Manager** -- Manages all technical aspects of the EU Information System API integration. Handles eIDAS-compatible authentication (digital certificates, qualified electronic signatures, session management). Manages API rate limiting (respecting EU IS rate limits, implementing backoff). Implements circuit breaker patterns for EU IS outages. Manages connection pooling and keepalive for high-throughput submission periods. Handles API versioning (adapting to EU IS API version updates). Implements request/response logging for debugging and audit. Manages endpoint discovery for different Member State EU IS instances.

7. **Audit Trail Recorder** -- Records every interaction with the EU Information System in a tamper-proof, immutable audit log per Article 31 requirements. Logs all submission attempts (successful and failed), acknowledgement receipts, rejection notifications, resubmission events, registration actions, status queries, and authentication events. Each log entry includes timestamp, actor, action, payload hash, response hash, and provenance chain. Enforces 5-year retention with automatic retention tracking. Supports competent authority inspection by generating audit reports showing the complete submission history for any operator, product, or time period. Implements SHA-256 hash chains linking consecutive log entries to prevent tampering.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| DDS submission success rate | >= 98% on first attempt | Accepted submissions / total submissions |
| Submission turnaround time | < 60 seconds per individual DDS submission | Time from submission initiation to EU IS acknowledgement |
| Batch submission throughput | >= 50 DDS documents per hour | Sustained batch submission rate |
| Operator registration success rate | 100% of valid registrations accepted | Registration acceptance / total registration attempts |
| Geolocation format compliance | 100% of coordinates pass EU IS validation | EU IS geolocation validation pass rate |
| Status tracking latency | < 5 minutes from EU IS state change to platform notification | Time delta between EU IS update and platform update |
| Audit trail completeness | 100% of EU IS interactions logged | Audit log entries / total API interactions |
| 5-year retention compliance | 100% of records retrievable within retention period | Automated retention validation test |
| Error recovery rate | >= 99% of transient failures recovered automatically | Auto-recovered submissions / total transient failures |
| eIDAS authentication uptime | >= 99.9% session availability | Authentication success rate during operating hours |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ EUDR-affected operators and traders across the EU must interface with the EU Information System for DDS submission and operator registration, representing a regulatory submission technology market of 1.5-3 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers processing the 7 regulated commodities requiring automated EU IS integration for DDS submission, status tracking, and compliance management, estimated at 400-800M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 25-40M EUR in EU IS interface module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities submitting hundreds of DDS documents per quarter
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with multi-Member-State registration requirements
- Timber and paper industry operators with complex product portfolios requiring batch DDS submission
- Compliance departments managing quarterly DDS submission deadlines across multiple business units and Member States

**Secondary:**
- Customs brokers and freight forwarders handling DDS submission on behalf of importers
- Compliance consultants managing EU IS interactions for multiple client operators
- Certification bodies requiring submission status tracking for their audit processes
- SME importers (1,000-10,000 shipments/year) ahead of June 30, 2026 enforcement deadline

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual EU portal submission | No cost; direct control | 15-30 min per DDS; no batch capability; no automation; no audit trail; does not scale | < 60 seconds per DDS; batch submission; full automation; complete audit trail |
| Generic API integration platforms (MuleSoft, Boomi) | Multi-purpose; enterprise integration | Not EUDR-specific; no eIDAS handling; no DDS validation; no submission lifecycle; high configuration cost | Purpose-built for EU IS; native eIDAS support; built-in DDS validation; full lifecycle management |
| Compliance consulting firms | EUDR expertise; manual submission service | EUR 50-100 per DDS submission; slow turnaround; limited scale; no programmatic tracking | Fully automated; EUR 2-5 per DDS; instant submission; real-time tracking |
| Niche EUDR portal tools (early movers) | First to market | DDS form filling only; no upstream agent integration; no batch submission; no registration management | Full integration with 35 upstream agents; batch submission; multi-MS registration; audit trail |
| In-house custom API clients | Tailored to org | 3-6 months to build; fragile to EU IS API changes; no eIDAS management; no error recovery | Ready now; auto-adapts to API changes; managed eIDAS; production-grade resilience |

### 2.4 Differentiation Strategy

1. **Full-stack integration** -- Not a standalone submission tool. Natively integrated with EUDR-030 (Documentation Generator) and all 35 upstream EUDR agents. Documents flow seamlessly from generation to submission with zero manual handoff.
2. **Production-grade resilience** -- Circuit breakers, exponential backoff, submission queuing, batch management, and automatic retry ensure reliable submission even during EU IS outages or rate limiting periods.
3. **Multi-Member-State registration** -- Unified management of operator registrations across all 27 EU Member States, with centralized status tracking, renewal alerting, and cross-reference validation.
4. **Complete audit trail** -- Every EU IS interaction is immutably logged with SHA-256 hash chains, meeting Article 31 five-year retention requirements and supporting competent authority inspections under Articles 14-16.
5. **eIDAS-native authentication** -- Built-in management of eIDAS digital certificates, qualified electronic signatures, and authentication sessions, eliminating the need for operators to manage complex cryptographic credentials manually.
6. **Geolocation precision guarantee** -- Automated normalization of all geolocation data to EU IS specifications (WGS 84, 6 decimal places, polygon closure validation), preventing the most common cause of DDS submission rejection.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable operators to submit DDS to the EU Information System without manual portal interaction | 100% of DDS submissions via automated API | Q2 2026 |
| BG-2 | Eliminate DDS submission rejections caused by formatting or technical errors | >= 98% first-attempt acceptance rate | Q2 2026 |
| BG-3 | Manage operator registrations across all EU Member States from a single platform | 100% of operator registrations tracked and managed | Q2 2026 |
| BG-4 | Provide competent authorities with complete submission audit trails on demand | < 5 minutes from request to audit report delivery | Q3 2026 |
| BG-5 | Become the reference EU IS integration solution for EUDR compliance | 500+ enterprise customers using automated submission | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Automated DDS submission | Submit DDS documents to EU IS via API with authentication, validation, and acknowledgement tracking |
| PG-2 | Operator registration management | Full lifecycle management of operator registrations across EU Member States |
| PG-3 | Geolocation formatting | Normalize all geolocation data to exact EU IS specifications (WGS 84, 6 decimal places) |
| PG-4 | Document package assembly | Prepare, compress, and segment document packages for EU IS size limits |
| PG-5 | Submission lifecycle tracking | Track DDS status from submission through acceptance/rejection with automated workflows |
| PG-6 | API resilience | Handle EU IS outages, rate limiting, and transient failures with production-grade resilience |
| PG-7 | Regulatory audit trail | Immutable, tamper-proof log of all EU IS interactions with 5-year retention |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Submission latency | < 60 seconds p99 for individual DDS submission end-to-end |
| TG-2 | Batch submission throughput | >= 50 DDS documents per hour sustained |
| TG-3 | API response time | < 200ms p95 for internal platform API endpoints |
| TG-4 | Error recovery | >= 99% automatic recovery from transient EU IS failures |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic formatting, validation, and submission; no LLM in critical path |
| TG-7 | Geolocation precision | 100% of coordinates formatted to 6+ decimal places in WGS 84 |
| TG-8 | Audit integrity | SHA-256 hash chain on all audit log entries; zero gaps |

### 3.4 Non-Goals

1. DDS document content generation (EUDR-030 Documentation Generator handles this)
2. Due diligence workflow orchestration (EUDR-026 Due Diligence Orchestrator handles this)
3. Risk assessment calculation (EUDR-028 Risk Assessment Engine handles this)
4. Mitigation measure design (EUDR-029 Mitigation Measure Designer handles this)
5. Supply chain graph visualization (EUDR-001 handles graph topology and rendering)
6. Satellite data acquisition or analysis (EUDR-003/004/005 handle this)
7. Direct manipulation of EU portal UI through browser automation (API-only integration)
8. Carbon footprint or GHG reporting (GL-GHG-APP handles this)
9. Competitor submission benchmarking
10. Financial impact analysis of submission rejections (defer to analytics platform)

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, importing cocoa, palm oil, and soya from 15+ countries; registered in 4 EU Member States |
| **EUDR Pressure** | Must submit 200+ DDS documents per quarter across 3 commodity categories to the EU Information System; must maintain valid operator registrations in Germany, Netherlands, Belgium, and France |
| **Pain Points** | Currently submits DDS manually through the EU portal -- each submission takes 15-30 minutes; no batch submission capability; registration renewals are tracked in a spreadsheet and occasionally missed; no programmatic way to check submission status; audit trail is a collection of screenshots and email confirmations |
| **Goals** | Automated batch DDS submission; unified registration management across 4 Member States; real-time submission status dashboard; complete audit trail for competent authority inspections; zero submission rejections |
| **Technical Skill** | Moderate -- comfortable with web applications and regulatory portals but not a developer |

### Persona 2: IT Integration Engineer -- Stefan (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Integration Engineer at a large EU timber importer |
| **Company** | 800 employees, importing tropical and temperate wood from 20+ countries; complex ERP landscape (SAP) |
| **EUDR Pressure** | Must integrate GreenLang platform with the EU Information System API; responsible for managing eIDAS digital certificates and API credentials; must ensure submission pipeline reliability during peak quarterly submission periods |
| **Pain Points** | EU IS API documentation is complex; eIDAS certificate management is cryptographically challenging; rate limiting causes submission failures during peak periods; no standard error recovery framework; must handle API version changes when EU IS updates |
| **Goals** | Production-grade API integration with automatic error recovery; managed eIDAS credential lifecycle; clear API versioning strategy; comprehensive request/response logging for debugging; SLA monitoring for submission pipeline |
| **Technical Skill** | High -- experienced with enterprise API integration, certificate management, and distributed systems |

### Persona 3: Customs Broker -- Jean-Pierre (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Customs Broker at an EU freight forwarding company |
| **Company** | 200 employees, handling EUDR submissions on behalf of 50+ importer clients |
| **EUDR Pressure** | Must manage DDS submissions for multiple operators across multiple Member States; each client has different registration numbers, commodities, and submission schedules; must track submission status for all clients simultaneously |
| **Pain Points** | Managing 50+ operator registrations manually is overwhelming; submitting DDS for multiple clients requires switching between EU portal accounts; cannot provide clients with real-time submission status; audit trail is fragmented across client accounts |
| **Goals** | Multi-operator submission management from a single platform; unified submission status dashboard across all clients; automated registration tracking for all clients; client-facing submission reports |
| **Technical Skill** | Low-moderate -- uses customs declaration software and web portals |

### Persona 4: Regulatory Inspector -- Inspector Jansen (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | EUDR Enforcement Inspector at a national competent authority |
| **Company** | Government customs and market surveillance agency |
| **EUDR Pressure** | Must verify that operators have submitted valid DDS documents to the EU Information System; must check submission timestamps against market placement dates; must verify that registration numbers are current |
| **Pain Points** | Operators provide inconsistent submission records; cannot independently verify submission claims without accessing the EU IS directly; registration status verification requires manual cross-referencing |
| **Goals** | Operators can produce complete submission audit trails instantly; audit reports show timestamp-verified submission records with EU IS reference numbers; registration status history is available on demand |
| **Technical Skill** | Low-moderate -- uses government enforcement portals and document review tools |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(1)** | Operators shall exercise due diligence and submit DDS before placing products on market | DDSSubmissionGateway transmits DDS to EU IS; SubmissionStatusTracker confirms acceptance before market placement |
| **Art. 4(2)** | Due diligence system shall include information collection, risk assessment, risk mitigation | Entire upstream agent pipeline (EUDR-001 through EUDR-035) feeds into EUDR-036 for final submission |
| **Art. 9(1)(a-g)** | Information requirements: product description, quantity, country, geolocation, dates, supplier, buyer | GeolocationDataFormatter ensures all Article 9 geolocation data meets EU IS format specifications |
| **Art. 9(1)(d)** | Geolocation of all plots: GPS coordinates (single point) or polygon (> 4 ha) in WGS 84 | GeolocationDataFormatter enforces WGS 84, 6 decimal places, polygon closure, winding order validation |
| **Art. 12(1)** | Submit DDS to the Information System before placing/making available on market or exporting | DDSSubmissionGateway handles API-based submission with pre-flight validation |
| **Art. 12(2)** | DDS shall contain information listed in Annex II | DDSSubmissionGateway validates Annex II completeness before submission |
| **Art. 12(3)** | Information System assigns reference number to each DDS | SubmissionStatusTracker captures and records EU IS reference numbers |
| **Art. 13** | Simplified due diligence for low-risk country products | DDSSubmissionGateway supports simplified DDS submission variant |
| **Art. 14(1)** | Operators shall register with competent authority of establishment Member State | OperatorRegistrationManager handles initial registration submission |
| **Art. 14(2)** | Competent authority assigns registration number to operator | OperatorRegistrationManager captures and tracks registration numbers |
| **Art. 14(3)** | Operators shall notify competent authority of changes to registration details | OperatorRegistrationManager submits registration updates on detail changes |
| **Art. 15** | Competent authorities perform risk-based checks | AuditTrailRecorder provides submission records for competent authority inspection |
| **Art. 16** | Substantive checks verifying due diligence system adequacy | AuditTrailRecorder generates comprehensive audit reports for substantive checks |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) affects due diligence level | DDSSubmissionGateway includes country benchmarking data in submissions |
| **Art. 31(1)** | Retain records of DDS and due diligence for at least 5 years | AuditTrailRecorder enforces 5-year retention on all submission records |
| **Art. 33(1)** | EU Information System established for submission and verification of DDS | Entire agent interfaces with the EU IS per Article 33 mandate |
| **Art. 33(2)** | Information System shall enable electronic submission of DDS | DDSSubmissionGateway implements electronic API-based submission |
| **Art. 33(3)** | Information System accessible to competent authorities for verification | AuditTrailRecorder maintains records synchronized with EU IS for inspection support |
| **eIDAS Art. 3** | Electronic identification means accepted across EU Member States | APIIntegrationManager implements eIDAS-compliant authentication |
| **eIDAS Art. 25** | Qualified electronic signatures have legal effect equivalent to handwritten signatures | APIIntegrationManager supports qualified electronic signature for DDS signing |

### 5.2 EU Information System Technical Specifications

| Specification | Requirement | Agent Implementation |
|---------------|-------------|---------------------|
| **API Protocol** | RESTful HTTPS with JSON payloads | APIIntegrationManager implements REST client with JSON serialization |
| **Authentication** | eIDAS Level of Assurance: Substantial or High | APIIntegrationManager manages digital certificates and authentication flows |
| **DDS Format** | JSON per EU IS schema specification (versioned) | DDSSubmissionGateway validates against schema version; adapts to updates |
| **Geolocation Format** | WGS 84, coordinates as [lat, lon] arrays, polygons as closed rings | GeolocationDataFormatter normalizes all coordinates to specification |
| **Coordinate Precision** | Minimum 6 decimal places (approximately 0.11m accuracy) | GeolocationDataFormatter enforces minimum precision; pads if necessary |
| **Polygon Requirements** | Closed ring (first vertex = last vertex); counter-clockwise winding | GeolocationDataFormatter validates and corrects polygon geometry |
| **Payload Size Limit** | Maximum 10 MB per individual API request | DocumentPackageAssembler segments large payloads; manages multi-part upload |
| **Rate Limiting** | Maximum requests per minute varies by endpoint (typically 60-120 RPM) | APIIntegrationManager implements rate limiter with per-endpoint configuration |
| **API Versioning** | Semantic versioning; breaking changes with deprecation notice | APIIntegrationManager implements version negotiation and adapter pattern |
| **Response Format** | JSON with standard error codes and descriptions | APIIntegrationManager parses all responses with structured error handling |
| **Timeout** | 30-second request timeout; 5-minute upload timeout for large payloads | APIIntegrationManager implements configurable timeout with retry on timeout |
| **TLS** | TLS 1.2 minimum; TLS 1.3 preferred | APIIntegrationManager enforces TLS 1.2+ via SEC-004 TLS configuration |

### 5.3 Operator Registration Requirements (Article 14)

| Requirement | Data Fields | Agent Handling |
|-------------|------------|----------------|
| Initial registration | Legal name, registered address, EORI number, economic activity code (NACE), contact person, email, phone | OperatorRegistrationManager assembles and submits registration payload |
| Registration number | Issued by competent authority upon successful registration | OperatorRegistrationManager captures and stores registration number per Member State |
| Registration updates | Changes to any registration field require notification | OperatorRegistrationManager detects changes and submits updates |
| Multi-Member-State | Operators established in multiple MS must register in each | OperatorRegistrationManager tracks registrations per MS with unified dashboard |
| SME classification | SME operators have different obligations and enforcement dates | OperatorRegistrationManager records SME status for appropriate handling |
| Annual renewal | Registration validity period with renewal requirement | OperatorRegistrationManager tracks expiry and initiates renewal workflow |

### 5.4 DDS Submission Requirements (Article 12 and Annex II)

| # | DDS Field (Annex II) | EU IS Format | Validation |
|---|---------------------|--------------|------------|
| 1 | Operator identification (name, address, EORI) | JSON object with name, address, eori fields | EORI format validation (2-letter country + up to 15 digits) |
| 2 | Registration number | String matching MS-specific format | Cross-reference with OperatorRegistrationManager records |
| 3 | Product description (trade name, type, common name, scientific name) | JSON object with structured fields | Non-empty validation; scientific name against accepted species list |
| 4 | HS/CN code(s) | String matching EU CN code format | Validation against EUDR Annex I product list |
| 5 | Quantity (net mass in kg, volume in m3, or number of items) | Numeric with unit code | Positive value; unit from accepted list (KGM, MTQ, NAR) |
| 6 | Country of production | ISO 3166-1 alpha-2 code(s) | Valid country code; cross-reference with Article 29 benchmarking |
| 7 | Geolocation of production plots | Array of coordinate objects (point or polygon) | WGS 84; 6 decimal places; polygon closure; > 4 ha requires polygon |
| 8 | Date or time range of production | ISO 8601 date range | Start date before end date; within reasonable range |
| 9 | Supplier identification (name, address, email) | JSON object | Non-empty; valid email format |
| 10 | Buyer identification (name, address, email) | JSON object | Non-empty; valid email format |
| 11 | Deforestation-free evidence reference | Array of evidence IDs | At least one evidence reference required |
| 12 | Legal production evidence reference | Array of evidence IDs | At least one evidence reference required |
| 13 | Risk assessment conclusion | Structured risk summary JSON | Required risk level classification |
| 14 | Risk mitigation measures (if applicable) | Structured mitigation summary JSON | Required if risk was non-negligible |
| 15 | Compliance declaration | Boolean with declaration text | Must be true for submission |

### 5.5 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Referenced in all DDS deforestation-free declarations submitted through the agent |
| June 29, 2023 | Regulation entered into force | Legal basis for EU IS establishment and all submission requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Operators must have EU IS submission capability operational NOW |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; batch submission capacity critical |
| Ongoing (quarterly) | Country benchmarking updates by EC | DDS submissions must reflect current country classifications |
| Ongoing (annual) | Operator registration renewal | OperatorRegistrationManager must track and initiate renewals |
| 5 years post-submission | Record retention deadline | AuditTrailRecorder must maintain all records for 5 years |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-3 form the core submission engine; Features 4-5 form the data preparation layer; Features 6-7 form the lifecycle and resilience layer; Features 8-9 form the compliance and audit layer.

**P0 Features 1-3: Core Submission Engine**

---

#### Feature 1: DDS Submission Gateway

**User Story:**
```
As a compliance officer,
I want to submit finalized DDS documents to the EU Information System through an automated API gateway,
So that I can complete my EUDR compliance obligation without manually uploading documents through the EU portal.
```

**Acceptance Criteria:**
- [ ] Receives finalized DDS documents from EUDR-030 Documentation Generator (JSON and XML formats)
- [ ] Validates DDS against the current EU Information System schema version before submission (pre-flight check)
- [ ] Returns detailed pre-flight validation errors with field-level descriptions and remediation guidance
- [ ] Authenticates with the EU IS API using eIDAS credentials managed by the API Integration Manager (Feature 6)
- [ ] Submits DDS via the official EU IS API endpoint with proper headers, payload formatting, and authentication tokens
- [ ] Handles synchronous submission flow: submit -> immediate acceptance/rejection response
- [ ] Handles asynchronous submission flow: submit -> receipt -> poll for result
- [ ] Captures EU IS acknowledgement reference number for accepted submissions
- [ ] Records EU IS rejection reason codes and human-readable descriptions for rejected submissions
- [ ] Supports individual DDS submission (single document per request)
- [ ] Supports batch DDS submission: queues multiple DDS documents and submits them sequentially or with configurable concurrency (default 5 concurrent, max 20)
- [ ] Manages submission queue with priority levels: URGENT (deadline approaching), NORMAL, LOW
- [ ] Generates submission receipt for each attempt: timestamp, DDS reference, EU IS response, submission hash
- [ ] Supports submission scheduling: schedule DDS for submission at a future date/time
- [ ] Supports dry-run mode: validate and format the submission without actually transmitting to EU IS
- [ ] Validates that the DDS references a valid, current operator registration number before submission

**Non-Functional Requirements:**
- Performance: Individual submission < 60 seconds p99 end-to-end
- Throughput: >= 50 DDS submissions per hour sustained (batch mode)
- Reliability: Zero lost submissions (every attempt durably recorded before transmission)
- Idempotency: Duplicate submission detection prevents accidental double-submission

**Dependencies:**
- EUDR-030 Documentation Generator (source of DDS documents)
- Feature 6: API Integration Manager (authentication and connection management)
- Feature 2: Operator Registration Manager (registration number validation)
- EU Information System API specification

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 integration engineer)

**Edge Cases:**
- EU IS returns HTTP 503 (service unavailable) -- Queue submission for retry with exponential backoff
- EU IS returns HTTP 429 (rate limited) -- Back off per Retry-After header; re-queue
- DDS references an expired operator registration -- Block submission; alert compliance officer
- Duplicate DDS ID detected by EU IS -- Parse duplicate error; link to existing submission record
- EU IS API version mismatch -- Fall back to previous API version if supported; alert on deprecation
- Network timeout during submission -- Record timeout event; retry with idempotency key

---

#### Feature 2: Operator Registration Manager

**User Story:**
```
As a compliance officer,
I want to manage my company's operator registration with EU Member State competent authorities from a single platform,
So that I can maintain valid registrations across all applicable Member States and ensure every DDS references a current registration number.
```

**Acceptance Criteria:**
- [ ] Supports initial operator registration: assembles registration payload (legal name, address, EORI, NACE code, contact person, email, phone) and submits to competent authority via EU IS API
- [ ] Captures registration numbers issued by competent authorities upon successful registration
- [ ] Tracks registration status per Member State: PENDING, ACTIVE, EXPIRED, SUSPENDED, REVOKED
- [ ] Supports multi-Member-State registration: operators established in or submitting from multiple EU countries can manage all registrations from a unified view
- [ ] Detects changes to operator details (name, address, EORI) and generates registration update notifications for submission to relevant competent authorities
- [ ] Tracks registration expiry dates and generates renewal alerts: 90 days, 60 days, 30 days, 7 days before expiry
- [ ] Initiates automated renewal workflow: pre-populates renewal application with current data; submits upon compliance officer approval
- [ ] Validates that every DDS submission references a valid, non-expired registration number for the applicable Member State
- [ ] Blocks DDS submission if registration is expired, suspended, or revoked; provides clear error with remediation steps
- [ ] Records SME classification status per operator for appropriate handling of enforcement timelines
- [ ] Supports delegation: one operator can authorize another entity (e.g., customs broker) to submit DDS on their behalf using delegated registration credentials
- [ ] Maintains complete registration history: all registration events (initial, update, renewal, suspension, revocation) with timestamps and evidence

**Non-Functional Requirements:**
- Completeness: 100% of operator registrations tracked with current status
- Alerting: Zero missed renewal deadlines (alerts generated for 100% of expiring registrations)
- Latency: Registration status lookup < 100ms (cached)
- Audit: Complete registration lifecycle audit trail per Article 31

**Dependencies:**
- Feature 6: API Integration Manager (authentication for competent authority endpoints)
- Feature 7: Audit Trail Recorder (logging all registration events)
- SEC-001 JWT Authentication (operator identity verification)
- SEC-002 RBAC Authorization (registration management permissions)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Operator registered in one MS opens operations in another -- Trigger new registration workflow for additional MS
- Competent authority rejects registration -- Parse rejection reason; provide remediation guidance; support resubmission
- EORI number changes (rare but possible during corporate restructuring) -- Update all active registrations across all MS
- Registration suspended pending investigation -- Block all DDS submissions; alert compliance officer with urgency
- Delegated submission authority revoked -- Immediately invalidate delegation; block further delegated submissions

---

#### Feature 3: Geolocation Data Formatter

**User Story:**
```
As a compliance officer,
I want all production plot geolocation data automatically formatted to the exact specifications required by the EU Information System,
So that I can ensure zero DDS submission rejections caused by geolocation format errors.
```

**Acceptance Criteria:**
- [ ] Normalizes all coordinates to WGS 84 coordinate reference system (EPSG:4326)
- [ ] Transforms coordinates from non-WGS84 reference systems (UTM zones, national grids) using pyproj or equivalent geodetic library
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
- [ ] Handles multi-plot commodities: structures geolocation data for products tracing to dozens or hundreds of production plots across multiple countries
- [ ] Groups geolocation data by country of production per EU IS structure requirements
- [ ] Retrieves source geolocation data from EUDR-002 (Geolocation Verification), EUDR-006 (Plot Boundary Manager), EUDR-007 (GPS Coordinate Validator)
- [ ] Generates geolocation quality report: total plots, plots with sufficient precision, plots requiring polygon upgrade, coordinate system transformations performed
- [ ] Flags plots with insufficient precision (fewer than 6 decimal places in source data that cannot be reliably padded) for manual review

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

Multi-Plot (commodity from 50 plots across 3 countries):
  Output: {
    "BR": [{"type": "point", "coordinates": [...]}, ...],
    "ID": [{"type": "polygon", "coordinates": [...]}, ...],
    "GH": [{"type": "point", "coordinates": [...]}, ...]
  }
```

**Non-Functional Requirements:**
- Precision: 100% of output coordinates have 6+ decimal places
- Accuracy: Coordinate system transformations accurate to < 1 meter
- Performance: Format 10,000 plots in < 5 seconds
- Validation: 100% of output geometries pass EU IS geolocation validation

**Dependencies:**
- EUDR-002 Geolocation Verification Agent (verified coordinates)
- EUDR-006 Plot Boundary Manager Agent (polygon boundaries)
- EUDR-007 GPS Coordinate Validator Agent (validated GPS coordinates)
- pyproj or equivalent geodetic transformation library
- Shapely for polygon geometry validation

**Estimated Effort:** 2 weeks (1 backend engineer with GIS expertise)

**Edge Cases:**
- Source coordinates in UTM Zone 37N (East Africa) -- Transform to WGS 84 using EPSG code lookup
- Polygon with self-intersection (bowtie shape) -- Flag as invalid; generate remediation guidance
- Plot straddles the antimeridian (180 degrees longitude) -- Split polygon at antimeridian per GeoJSON convention
- Very small plot (< 0.01 ha) with coordinates that round to same point at 6 decimals -- Flag for manual precision verification
- Polygon with duplicate consecutive vertices -- Deduplicate while maintaining geometry

---

**P0 Features 4-5: Data Preparation Layer**

> Features 4 and 5 are P0 launch blockers. Without document package assembly and submission status tracking, the core submission engine cannot handle large submissions or provide operators with visibility into the EU IS processing of their DDS documents. These features ensure that the submission pipeline is complete from preparation through confirmation.

---

#### Feature 4: Document Package Assembler

**User Story:**
```
As a compliance officer,
I want my DDS documents and supporting evidence automatically assembled, compressed, and segmented for EU Information System transmission,
So that I can submit complete compliance packages without worrying about EU IS size limits or format requirements.
```

**Acceptance Criteria:**
- [ ] Receives DDS documents and compliance packages from EUDR-030 Documentation Generator
- [ ] Attaches supporting evidence documents: risk assessment reports, certificates (FSC, RSPO, Rainforest Alliance), chain of custody records, supplier contracts, satellite verification reports
- [ ] Validates all attached documents against EU IS accepted formats (PDF, JSON, XML, PNG, JPEG, TIFF for satellite imagery)
- [ ] Compresses document packages using gzip compression to reduce transmission size
- [ ] Enforces EU IS payload size limit (10 MB per request); splits larger packages into appropriately sized segments
- [ ] Generates package manifest listing all included documents with: document ID, document type, file size, SHA-256 hash, segment assignment
- [ ] Manages multi-part upload workflow for segmented packages: upload segments in order, confirm assembly, verify integrity
- [ ] Calculates and embeds package-level SHA-256 integrity hash covering all segments
- [ ] Validates package completeness against DDS submission requirements: every referenced evidence document must be included
- [ ] Supports package preview: list all documents that will be included before assembly
- [ ] Handles duplicate evidence documents (same certificate referenced by multiple DDS) by deduplication within the package
- [ ] Generates package summary: total documents, total compressed size, number of segments, estimated upload time

**Non-Functional Requirements:**
- Performance: Package assembly < 30 seconds for packages up to 50 MB uncompressed
- Compression: Achieve >= 40% size reduction on average for text-heavy packages
- Integrity: 100% of packages pass integrity verification after transmission
- Reliability: Multi-part upload resumes from last successful segment on failure

**Dependencies:**
- EUDR-030 Documentation Generator (DDS documents and compliance packages)
- EUDR-012 Document Authentication Agent (authenticated certificates)
- S3 Object Storage for temporary package staging
- Feature 6: API Integration Manager (upload endpoint management)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Package exceeds 500 MB (complex supply chain with many satellite images) -- Segment into 50+ parts; manage upload state across extended upload session
- Evidence document in unsupported format (e.g., DOCX) -- Convert to PDF or reject with guidance
- Upload interrupted at segment 15 of 20 -- Resume from segment 16 without re-uploading prior segments
- EU IS returns checksum mismatch for a segment -- Re-upload the specific failed segment
- Same certificate file attached to 200 DDS documents in a batch -- Deduplicate; upload once; reference in all DDS manifests

---

#### Feature 5: Submission Status Tracker

**User Story:**
```
As a compliance officer,
I want to track the status of every DDS submission through its complete lifecycle in the EU Information System,
So that I can monitor acceptance rates, respond to rejections immediately, and provide audit-ready submission records.
```

**Acceptance Criteria:**
- [ ] Tracks submission lifecycle states: QUEUED, SUBMITTING, SUBMITTED, RECEIVED, UNDER_REVIEW, ACCEPTED, REJECTED, AMENDMENT_REQUIRED, RESUBMITTED, WITHDRAWN
- [ ] Polls EU IS API for status updates at configurable intervals (default: every 5 minutes during business hours, every 30 minutes outside business hours)
- [ ] Records every state transition with: timestamp, previous state, new state, EU IS response data, transition reason
- [ ] Captures EU IS reference numbers for accepted submissions and stores alongside GreenLang DDS reference
- [ ] Parses rejection reasons from EU IS response into structured error objects with: error code, affected field, error description, suggested remediation
- [ ] Triggers automated rejection workflow: notification to compliance officer, correction guidance generation, resubmission pathway via EUDR-030
- [ ] Supports manual rejection override: compliance officer can review rejection, make manual corrections, and trigger resubmission
- [ ] Tracks resubmission chain: links original submission, rejection, corrected version, and resubmission with full provenance
- [ ] Provides real-time submission dashboard: total submitted, pending, accepted, rejected, resubmitted -- with drill-down per operator, commodity, Member State
- [ ] Supports batch status monitoring: aggregate status view for batch submissions with per-DDS detail
- [ ] Generates submission status reports: daily, weekly, quarterly summary reports with acceptance rates, rejection reasons breakdown, turnaround times
- [ ] Supports webhook notifications for status changes: configurable webhook URL receives POST notifications on every state transition
- [ ] Calculates key metrics: average acceptance rate, average turnaround time, most common rejection reasons, resubmission success rate

**Non-Functional Requirements:**
- Latency: Status updates reflected in platform within 5 minutes of EU IS state change
- Completeness: 100% of submissions tracked through complete lifecycle (no orphaned submissions)
- Availability: Status dashboard available 99.9% of time
- History: Complete status history retained per Article 31 (5 years)

**Dependencies:**
- Feature 1: DDS Submission Gateway (submission records)
- Feature 6: API Integration Manager (EU IS status query API)
- Feature 7: Audit Trail Recorder (lifecycle event logging)
- Redis for real-time status caching
- WebSocket or SSE for real-time dashboard updates

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- EU IS does not provide status update for 7+ days -- Escalate to compliance officer with stale submission alert
- EU IS returns unknown status code -- Log raw response; classify as UNKNOWN; alert for manual review
- Submission accepted then subsequently revoked by competent authority -- Handle post-acceptance state change; alert immediately
- Batch of 200 DDS: 195 accepted, 5 rejected -- Track individually; generate focused rejection report for the 5 failed DDS
- Status poll fails due to EU IS outage -- Continue polling with exponential backoff; report EU IS availability issue

---

**P0 Features 6-7: Lifecycle and Resilience Layer**

> Features 6 and 7 are P0 launch blockers. Without API integration management and audit trail recording, the agent cannot authenticate with the EU Information System or maintain the regulatory compliance records required by Article 31. These features provide the technical foundation and compliance backbone for all submission operations.

---

#### Feature 6: API Integration Manager

**User Story:**
```
As an IT integration engineer,
I want the EU Information System API integration fully managed -- authentication, rate limiting, error recovery, and versioning,
So that I can ensure reliable, production-grade connectivity to the EU IS without building custom API infrastructure.
```

**Acceptance Criteria:**
- [ ] Implements eIDAS-compliant authentication: digital certificate management (X.509), qualified electronic signature support, authentication session management
- [ ] Manages digital certificate lifecycle: storage (encrypted at rest via SEC-003), expiry tracking, renewal alerts (90, 60, 30, 7 days before expiry), automatic rotation when new certificate is provisioned
- [ ] Supports delegated authentication: operator delegates submission authority to a third party (customs broker) using certificate delegation
- [ ] Implements OAuth 2.0 client credentials flow as fallback authentication method (if supported by EU IS)
- [ ] Manages API rate limiting per endpoint: tracks current request count, respects EU IS rate limits, implements token bucket algorithm for smooth request distribution
- [ ] Returns rate limit status to callers: remaining requests, reset time, recommended wait period
- [ ] Implements circuit breaker pattern: opens after configurable failure threshold (default 5 consecutive failures), half-open probe after configurable timeout (default 60 seconds), closes on successful probe
- [ ] Implements exponential backoff with jitter: base delay 1 second, max delay 300 seconds, jitter factor 0.5, max retries 5
- [ ] Manages connection pooling: configurable pool size (default 10 connections), keepalive interval (30 seconds), connection timeout (10 seconds)
- [ ] Handles API version negotiation: detects EU IS API version from response headers; adapts request format to match
- [ ] Implements adapter pattern for API version changes: version-specific request/response transformers that can be hot-swapped without agent restart
- [ ] Logs all API requests and responses: method, URL, headers (redacted), request body hash, response status, response body hash, latency, correlation ID
- [ ] Manages endpoint discovery: configurable EU IS base URLs per Member State; health check on endpoint availability
- [ ] Supports mutual TLS (mTLS) for enhanced security where required by specific Member State endpoints
- [ ] Provides health check endpoint reporting: EU IS connectivity status, authentication status, rate limit headroom, circuit breaker state

**Non-Functional Requirements:**
- Availability: >= 99.9% authentication session availability during operating hours
- Resilience: Automatic recovery from 99%+ of transient EU IS failures
- Security: All credentials encrypted at rest (AES-256-GCM via SEC-003); zero plaintext credential exposure
- Observability: Complete request/response logging with correlation IDs for debugging

**Dependencies:**
- SEC-001 JWT Authentication (internal platform authentication)
- SEC-003 Encryption at Rest (credential encryption)
- SEC-004 TLS 1.3 Configuration (transport security)
- SEC-006 Secrets Management (certificate storage in Vault)
- OBS-003 OpenTelemetry Tracing (distributed tracing across API calls)

**Estimated Effort:** 4 weeks (1 senior backend engineer with security expertise)

**Edge Cases:**
- Digital certificate expires during a batch submission -- Gracefully pause submissions; use cached authentication token for in-flight requests; alert for certificate renewal
- EU IS migrates to new API version mid-quarter -- Version adapter pattern allows immediate adaptation; log deprecation warnings
- EU IS endpoint for one Member State is unavailable while others work -- Per-endpoint circuit breakers; route submissions to available endpoints; queue submissions for unavailable endpoints
- Rate limit exceeded despite local tracking (concurrent requests from multiple platform instances) -- Implement distributed rate limiter using Redis; handle 429 responses gracefully
- mTLS handshake fails due to intermediate CA not trusted by EU IS -- Provide detailed certificate chain diagnosis; suggest CA bundle update

---

#### Feature 7: Audit Trail Recorder

**User Story:**
```
As a compliance officer,
I want every interaction with the EU Information System recorded in a tamper-proof, immutable audit log,
So that I can provide competent authorities with complete, verifiable submission records during Article 14-16 inspections.
```

**Acceptance Criteria:**
- [ ] Records all DDS submission events: submission attempt, EU IS response (acceptance/rejection), acknowledgement receipt, reference number assignment
- [ ] Records all registration events: initial registration, registration number issued, registration update, renewal, suspension, revocation
- [ ] Records all status tracking events: status poll, state transition detected, notification sent
- [ ] Records all authentication events: certificate used, session established, session expired, authentication failure
- [ ] Records all error and recovery events: transient failure, retry attempt, circuit breaker state change, recovery success/failure
- [ ] Each audit log entry contains: entry_id, timestamp (UTC), event_type, operator_id, dds_id (if applicable), registration_id (if applicable), action, request_hash (SHA-256 of outgoing payload), response_hash (SHA-256 of incoming response), actor, correlation_id, metadata
- [ ] Implements SHA-256 hash chain: each log entry includes the hash of the previous entry, creating a tamper-evident chain
- [ ] Hash chain verification: provides endpoint to verify hash chain integrity from any entry back to genesis
- [ ] Enforces immutability: audit log entries cannot be modified or deleted (append-only log in TimescaleDB hypertable)
- [ ] Enforces 5-year retention per Article 31: retention_expiry calculated as submission_date + 5 years
- [ ] Prevents deletion of entries within retention period (database-level constraint)
- [ ] Generates retention alerts: 60 days, 30 days, 7 days before entries exit retention period
- [ ] Archives expired entries (past 5-year retention) to cold storage (S3) with configurable archive policy
- [ ] Generates audit reports for competent authority inspection: filter by operator, time period, event type, DDS reference; export as PDF or CSV
- [ ] Supports forensic queries: "Show all EU IS interactions for DDS GL-DDS-OP123-COCOA-20260315-001 from submission to acceptance"
- [ ] Calculates audit metrics: total interactions, success rate, average latency, entries per operator

**Non-Functional Requirements:**
- Immutability: Zero modification or deletion of audit entries within retention period
- Integrity: SHA-256 hash chain verifiable from any entry to genesis
- Performance: Audit log write < 5ms (async, non-blocking to submission pipeline)
- Retention: 100% of entries retrievable within 5-year retention period
- Query: Audit report generation < 10 seconds for 100,000-entry date range

**Dependencies:**
- PostgreSQL + TimescaleDB (hypertable for time-series audit log)
- S3 Object Storage (cold archive for expired entries)
- SEC-005 Centralized Audit Logging (integration with platform audit system)
- Feature 1-6 (all features emit audit events)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- Hash chain broken by database corruption -- Detect during periodic integrity check; alert immediately; reconstruct from replicated log
- Audit query spans 5 years of data (millions of entries) -- Paginated query with streaming export; TimescaleDB chunk-based scanning
- Clock skew between platform instances producing out-of-order timestamps -- Use monotonic sequence numbers in addition to timestamps
- Retention period for a single DDS crosses regulatory update (retention extended from 5 to 7 years) -- Configuration-driven retention; can be updated without data migration

---

**P0 Features 8-9: Compliance and Integration Layer**

---

#### Feature 8: Rejection Analysis and Resubmission Engine

**User Story:**
```
As a compliance officer,
I want automated analysis of DDS submission rejections with correction guidance and streamlined resubmission,
So that I can resolve rejections quickly and minimize the delay to market placement.
```

**Acceptance Criteria:**
- [ ] Parses EU IS rejection responses into structured error objects: error_code, affected_field, error_description, severity (blocking/warning)
- [ ] Classifies rejection reasons into categories: FORMAT_ERROR (geolocation format, schema violation), DATA_ERROR (invalid CN code, missing field), REGISTRATION_ERROR (invalid/expired registration), DUPLICATE_ERROR (DDS already submitted), SYSTEM_ERROR (EU IS internal error)
- [ ] Generates field-level correction guidance for each rejection reason: what was wrong, what the correct value should be, which upstream agent needs to regenerate data
- [ ] Routes format errors to GeolocationDataFormatter (Feature 3) for automatic correction and reformatting
- [ ] Routes data errors to EUDR-030 Documentation Generator for DDS content correction
- [ ] Routes registration errors to OperatorRegistrationManager (Feature 2) for registration remediation
- [ ] Handles automatic resubmission: when correction is automated (format fix), resubmits without human intervention
- [ ] Handles manual resubmission: when correction requires human review (data error), queues for compliance officer review with correction guidance
- [ ] Tracks resubmission attempts: maximum configurable resubmissions per DDS (default 3) before escalation
- [ ] Generates rejection analytics: most common rejection reasons, rejection rate by commodity, rejection rate by Member State, trend over time
- [ ] Links rejection to root cause agent: if geolocation format caused rejection, trace to EUDR-007 GPS Coordinator Validator as root source
- [ ] Supports bulk rejection handling: when a batch submission has multiple rejections, group by rejection reason for efficient correction

**Non-Functional Requirements:**
- Speed: Rejection analysis < 5 seconds per rejection
- Accuracy: 95%+ of format errors correctable automatically without human intervention
- Traceability: Complete chain from rejection to correction to resubmission with provenance

**Dependencies:**
- Feature 1: DDS Submission Gateway (rejection records)
- Feature 5: Submission Status Tracker (rejection detection)
- EUDR-030 Documentation Generator (DDS regeneration for data corrections)
- Feature 3: GeolocationDataFormatter (format corrections)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- EU IS rejection reason is vague ("validation failed") with no field-level detail -- Log raw response; flag for manual investigation; submit support request to EU IS helpdesk
- Automatic correction introduces new error -- Detect correction loop (same DDS rejected twice after correction); escalate to manual review
- Rejection reason changes between resubmissions (first rejection: format error; second rejection: data error) -- Track each rejection independently; apply corrections incrementally
- Maximum resubmission attempts reached -- Block further automatic resubmission; escalate to compliance officer with full rejection history

---

#### Feature 9: Multi-Member-State Submission Router

**User Story:**
```
As a compliance officer at a multinational company,
I want DDS submissions automatically routed to the correct Member State EU IS endpoint based on operator registration and product destination,
So that I can manage submissions across multiple EU countries from a single platform without manual endpoint selection.
```

**Acceptance Criteria:**
- [ ] Maintains registry of EU IS API endpoints per Member State (27 EU Member States + EEA countries if applicable)
- [ ] Routes DDS submissions to the correct Member State endpoint based on: operator registration Member State, product destination Member State, or explicit routing rule
- [ ] Supports routing rules: operator-level default MS, product-level MS override, manual MS selection per DDS
- [ ] Validates that operator has active registration in the target Member State before routing
- [ ] Handles Member State-specific submission requirements: some MS may require additional fields, different formats, or supplementary documentation
- [ ] Manages endpoint health per Member State: monitors availability, tracks latency, detects outages
- [ ] Fails over gracefully when a Member State endpoint is unavailable: queues submissions for the unavailable MS; processes submissions for other MS uninterrupted
- [ ] Generates routing report: submissions per Member State, acceptance rate per MS, average latency per MS
- [ ] Supports routing dry-run: shows where each DDS in a batch would be routed without actually submitting
- [ ] Handles EU IS federation: if EU IS uses a centralized endpoint with MS-level routing headers, adapts request format accordingly

**Non-Functional Requirements:**
- Routing accuracy: 100% of submissions routed to correct MS endpoint
- Resilience: MS endpoint failure does not block submissions to other MS
- Latency: Routing decision < 10ms per DDS

**Dependencies:**
- Feature 2: Operator Registration Manager (registration-to-MS mapping)
- Feature 6: API Integration Manager (per-endpoint connection management)
- Configuration: EU IS endpoint registry (base URLs, API versions, auth requirements per MS)

**Estimated Effort:** 2 weeks (1 backend engineer)

**Edge Cases:**
- New EU Member State added (e.g., EU enlargement) -- Configuration-driven endpoint registry; add new MS without code change
- MS endpoint uses different API version than other MS -- Per-endpoint version adapter from Feature 6
- Operator submits DDS for a product destined for MS where they are not registered -- Block with clear error; suggest registration in target MS
- EU IS transitions from distributed MS endpoints to centralized endpoint -- Adapter pattern allows seamless migration

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Submission Analytics Dashboard
- Real-time visualization of submission volumes, acceptance rates, and turnaround times
- Trend analysis: submission patterns over time, seasonal peaks (quarterly deadlines)
- Rejection reason heatmap: most common issues by commodity, by MS, by time period
- Benchmark: operator's submission metrics compared to anonymized platform averages
- Forecast: predicted submission volume for upcoming quarter based on pipeline data

#### Feature 11: Proactive Schema Compliance Monitor
- Monitor EU IS for schema updates and new API versions
- Automatically download and validate against new schema versions
- Generate compliance gap report when new schema introduces new required fields
- Alert integration engineers to breaking changes with migration guidance
- Test all current DDS templates against new schema in sandbox mode

#### Feature 12: Submission Scheduling and Calendar Integration
- Calendar view of submission deadlines per MS per commodity
- Automated scheduling: submit DDS N days before regulatory deadline
- Integration with corporate calendars (Outlook, Google Calendar) for compliance officer reminders
- Historical deadline compliance tracking: percentage of submissions made before deadline

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- EU portal UI automation (browser-based submission via Selenium or equivalent) -- API-only integration
- DDS content generation or modification (EUDR-030 handles document content; this agent handles transmission only)
- Direct competent authority communication for inspections (out of scope; handled by compliance team)
- Financial impact analysis of submission delays (defer to analytics platform)
- Mobile-native submission app (web responsive design only for v1.0)
- Integration with non-EU regulatory systems (US Lacey Act, UK EUDR equivalent) -- defer to future international modules
- Blockchain-based submission receipts (SHA-256 hash chain provides sufficient integrity)
- AI-based rejection prediction (defer to Phase 2; use rejection analytics for pattern identification first)

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
| AGENT-EUDR-036        |           | AGENT-EUDR-030            |           | AGENT-EUDR-001        |
| EU Information System |<--------->| Documentation             |<--------->| Supply Chain Mapping  |
| Interface             |           | Generator                 |           | Master                |
|                       |           |                           |           |                       |
| - DDSSubmissionGW     |           | - DDSStatementGenerator   |           | - GraphEngine         |
| - OperatorRegMgr      |           | - Article9DataAssembler   |           | - MultiTierMapper     |
| - GeoDataFormatter    |           | - RiskAssessDocumenter    |           | - GeolocationLinker   |
| - DocPkgAssembler     |           | - MitigationDocumenter    |           | - RiskPropagation     |
| - StatusTracker       |           | - CompliancePackageBlder  |           +-----------------------+
| - APIIntegrationMgr   |           | - DocumentVersionMgr      |
| - AuditTrailRecorder  |           | - SubmissionEngine        |
+-----------+-----------+           +---------------------------+
            |
            |  eIDAS / TLS 1.3 / HTTPS
            |
+-----------v-----------+           +-----------v-----------+
| EU Information System |           | EUDR-006 / EUDR-007   |
| (Article 33)          |           | Plot Boundary Manager  |
|                       |           | GPS Coordinate Validator|
| - DDS Submission API  |           |                       |
| - Registration API    |           | Source geolocation data|
| - Status Query API    |           | for formatting         |
| - Geolocation Upload  |           +-----------------------+
+-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/eu_information_system_interface/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # EUISInterfaceConfig with GL_EUDR_EUIS_ env prefix
    models.py                            # Pydantic v2 models for submissions, registrations, audit entries
    provenance.py                        # SHA-256 hash chains for audit trail integrity
    metrics.py                           # Prometheus metrics (18 metrics, gl_eudr_euis_ prefix)
    dds_submission_gateway.py            # Engine 1: DDS submission to EU IS
    operator_registration_manager.py     # Engine 2: Operator registration lifecycle
    geolocation_data_formatter.py        # Engine 3: Geolocation normalization and formatting
    document_package_assembler.py        # Engine 4: Package assembly and compression
    submission_status_tracker.py         # Engine 5: Submission lifecycle tracking
    api_integration_manager.py           # Engine 6: EU IS API client, auth, resilience
    audit_trail_recorder.py              # Engine 7: Immutable audit logging
    setup.py                             # EUISInterfaceService facade
    api.py                               # FastAPI router (30+ endpoints)
    reference_data/
        __init__.py
        eu_is_endpoints.py               # EU IS API endpoint registry per Member State
        eu_is_schema.py                  # EU IS DDS submission schema definitions (versioned)
        member_state_codes.py            # EU Member State codes and competent authority references
        rejection_codes.py               # EU IS rejection reason code catalog
        eidas_config.py                  # eIDAS authentication configuration
```

### 7.3 Data Models (Key Entities)

```python
# ============================================================
# Enumerations
# ============================================================

class SubmissionStatus(str, Enum):
    QUEUED = "queued"                    # In local queue awaiting submission
    SUBMITTING = "submitting"            # Transmission in progress
    SUBMITTED = "submitted"              # Transmitted to EU IS
    RECEIVED = "received"                # Acknowledged by EU IS as received
    UNDER_REVIEW = "under_review"        # EU IS processing/reviewing
    ACCEPTED = "accepted"                # Accepted by EU IS
    REJECTED = "rejected"                # Rejected by EU IS with reasons
    AMENDMENT_REQUIRED = "amendment_required"  # EU IS requests amendment
    RESUBMITTED = "resubmitted"          # Corrected version resubmitted
    WITHDRAWN = "withdrawn"              # Withdrawn by operator

class RegistrationStatus(str, Enum):
    PENDING = "pending"                  # Registration submitted, awaiting confirmation
    ACTIVE = "active"                    # Registration active and valid
    EXPIRED = "expired"                  # Registration expired (renewal needed)
    SUSPENDED = "suspended"              # Suspended by competent authority
    REVOKED = "revoked"                  # Permanently revoked
    RENEWAL_PENDING = "renewal_pending"  # Renewal submitted, awaiting confirmation

class RejectionCategory(str, Enum):
    FORMAT_ERROR = "format_error"        # Geolocation format, schema violation
    DATA_ERROR = "data_error"            # Invalid CN code, missing field
    REGISTRATION_ERROR = "registration_error"  # Invalid/expired registration
    DUPLICATE_ERROR = "duplicate_error"  # DDS already submitted
    SYSTEM_ERROR = "system_error"        # EU IS internal error
    UNKNOWN = "unknown"                  # Unclassified rejection

class AuditEventType(str, Enum):
    SUBMISSION_ATTEMPT = "submission_attempt"
    SUBMISSION_SUCCESS = "submission_success"
    SUBMISSION_FAILURE = "submission_failure"
    SUBMISSION_REJECTED = "submission_rejected"
    SUBMISSION_ACCEPTED = "submission_accepted"
    RESUBMISSION = "resubmission"
    REGISTRATION_SUBMIT = "registration_submit"
    REGISTRATION_CONFIRMED = "registration_confirmed"
    REGISTRATION_RENEWAL = "registration_renewal"
    REGISTRATION_UPDATE = "registration_update"
    STATUS_POLL = "status_poll"
    STATUS_CHANGE = "status_change"
    AUTH_SESSION_START = "auth_session_start"
    AUTH_SESSION_END = "auth_session_end"
    AUTH_FAILURE = "auth_failure"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    RATE_LIMIT_HIT = "rate_limit_hit"
    ERROR_RECOVERY = "error_recovery"

class CoordinateFormat(str, Enum):
    POINT = "point"                      # Single GPS coordinate (plots <= 4 ha)
    POLYGON = "polygon"                  # Polygon boundary (plots > 4 ha)

class MemberState(str, Enum):
    AT = "AT"   # Austria
    BE = "BE"   # Belgium
    BG = "BG"   # Bulgaria
    HR = "HR"   # Croatia
    CY = "CY"   # Cyprus
    CZ = "CZ"   # Czech Republic
    DK = "DK"   # Denmark
    EE = "EE"   # Estonia
    FI = "FI"   # Finland
    FR = "FR"   # France
    DE = "DE"   # Germany
    GR = "GR"   # Greece
    HU = "HU"   # Hungary
    IE = "IE"   # Ireland
    IT = "IT"   # Italy
    LV = "LV"   # Latvia
    LT = "LT"   # Lithuania
    LU = "LU"   # Luxembourg
    MT = "MT"   # Malta
    NL = "NL"   # Netherlands
    PL = "PL"   # Poland
    PT = "PT"   # Portugal
    RO = "RO"   # Romania
    SK = "SK"   # Slovakia
    SI = "SI"   # Slovenia
    ES = "ES"   # Spain
    SE = "SE"   # Sweden

# ============================================================
# Core Models
# ============================================================

# DDS Submission Record
class DDSSubmission(BaseModel):
    submission_id: str                   # Unique submission ID: GL-SUB-{operator}-{seq}
    dds_id: str                          # Reference to EUDR-030 DDS document
    operator_id: str                     # Operator UUID
    member_state: MemberState            # Target Member State for submission
    registration_number: str             # Operator registration number in target MS
    status: SubmissionStatus             # Current submission status
    priority: str                        # URGENT, NORMAL, LOW
    eu_is_reference: Optional[str]       # EU IS assigned reference number
    submitted_at: Optional[datetime]     # Timestamp of transmission to EU IS
    received_at: Optional[datetime]      # Timestamp of EU IS receipt acknowledgement
    accepted_at: Optional[datetime]      # Timestamp of acceptance
    rejected_at: Optional[datetime]      # Timestamp of rejection
    rejection_reasons: Optional[List[RejectionDetail]]
    resubmission_of: Optional[str]       # Previous submission ID if this is a resubmission
    resubmission_count: int              # Number of resubmission attempts
    batch_id: Optional[str]              # Batch submission ID if part of batch
    request_hash: str                    # SHA-256 of outgoing payload
    response_hash: Optional[str]         # SHA-256 of EU IS response
    provenance_hash: str                 # SHA-256 of complete submission record
    created_at: datetime
    updated_at: datetime

# Operator Registration
class OperatorRegistration(BaseModel):
    registration_id: str                 # Internal registration tracking ID
    operator_id: str                     # Operator UUID
    operator_name: str                   # Legal name
    operator_address: str                # Registered address
    eori_number: str                     # EORI number
    nace_code: str                       # Economic activity code
    contact_person: str                  # Contact person name
    contact_email: str                   # Contact email
    contact_phone: str                   # Contact phone
    member_state: MemberState            # Registration Member State
    registration_number: Optional[str]   # Issued by competent authority
    status: RegistrationStatus           # Current registration status
    registered_at: Optional[datetime]    # Date of successful registration
    expires_at: Optional[datetime]       # Registration expiry date
    last_renewed_at: Optional[datetime]  # Last renewal date
    is_sme: bool                         # SME classification
    delegation_active: bool              # Whether delegation is active
    delegated_to: Optional[str]          # Delegated entity ID
    created_at: datetime
    updated_at: datetime

# Formatted Geolocation Data
class FormattedGeolocation(BaseModel):
    plot_id: str                         # Reference to upstream plot ID
    country_code: str                    # ISO 3166-1 alpha-2
    coordinate_format: CoordinateFormat  # POINT or POLYGON
    coordinates: Union[List[float], List[List[float]]]  # [lat, lon] or [[lat,lon],...]
    source_crs: str                      # Source coordinate reference system
    precision_decimal_places: int        # Number of decimal places in output
    area_hectares: Optional[float]       # Plot area (determines point vs polygon)
    polygon_closed: bool                 # Whether polygon is properly closed
    winding_order: Optional[str]         # "counter_clockwise" or "clockwise" (before correction)
    transformation_applied: bool         # Whether CRS transformation was performed
    validation_passed: bool              # Whether all EU IS validations pass
    validation_errors: List[str]         # List of any validation failures

# Document Package
class DocumentPackage(BaseModel):
    package_id: str                      # Unique package ID
    dds_id: str                          # Associated DDS
    operator_id: str                     # Operator UUID
    documents: List[PackageDocument]     # All documents in the package
    manifest: PackageManifest            # Package manifest with hashes
    total_size_bytes: int                # Total uncompressed size
    compressed_size_bytes: int           # Total compressed size
    segment_count: int                   # Number of upload segments
    package_hash: str                    # SHA-256 of complete package
    assembled_at: datetime

# Audit Trail Entry
class AuditTrailEntry(BaseModel):
    entry_id: str                        # Unique entry ID
    sequence_number: int                 # Monotonic sequence for ordering
    timestamp: datetime                  # UTC timestamp
    event_type: AuditEventType           # Type of event
    operator_id: str                     # Operator involved
    dds_id: Optional[str]                # DDS involved (if applicable)
    registration_id: Optional[str]       # Registration involved (if applicable)
    member_state: Optional[MemberState]  # Target Member State
    action: str                          # Human-readable action description
    request_hash: Optional[str]          # SHA-256 of outgoing payload
    response_hash: Optional[str]         # SHA-256 of incoming response
    actor: str                           # User or system that triggered the event
    correlation_id: str                  # Correlation ID for tracing
    previous_entry_hash: str             # Hash of previous audit entry (chain)
    entry_hash: str                      # SHA-256 of this entry (including previous_entry_hash)
    metadata: Dict[str, Any]             # Additional event-specific data
    retention_expiry: date               # 5 years from event date
```

### 7.4 Database Schema (New Migration: V124)

```sql
CREATE SCHEMA IF NOT EXISTS gl_eudr_euis;

-- ============================================================
-- Table 1: DDS Submissions
-- ============================================================
CREATE TABLE gl_eudr_euis.dds_submissions (
    submission_id VARCHAR(100) PRIMARY KEY,
    dds_id VARCHAR(100) NOT NULL,
    operator_id UUID NOT NULL,
    member_state CHAR(2) NOT NULL,
    registration_number VARCHAR(100),
    status VARCHAR(30) NOT NULL DEFAULT 'queued',
    priority VARCHAR(10) NOT NULL DEFAULT 'normal',
    eu_is_reference VARCHAR(200),
    submitted_at TIMESTAMPTZ,
    received_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,
    rejected_at TIMESTAMPTZ,
    rejection_reasons JSONB,
    resubmission_of VARCHAR(100),
    resubmission_count INTEGER NOT NULL DEFAULT 0,
    batch_id VARCHAR(100),
    request_hash VARCHAR(64) NOT NULL,
    response_hash VARCHAR(64),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    CONSTRAINT fk_euis_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

CREATE INDEX idx_euis_sub_operator ON gl_eudr_euis.dds_submissions(operator_id);
CREATE INDEX idx_euis_sub_dds ON gl_eudr_euis.dds_submissions(dds_id);
CREATE INDEX idx_euis_sub_status ON gl_eudr_euis.dds_submissions(status);
CREATE INDEX idx_euis_sub_ms ON gl_eudr_euis.dds_submissions(member_state);
CREATE INDEX idx_euis_sub_batch ON gl_eudr_euis.dds_submissions(batch_id);
CREATE INDEX idx_euis_sub_submitted ON gl_eudr_euis.dds_submissions(submitted_at);

-- ============================================================
-- Table 2: Operator Registrations
-- ============================================================
CREATE TABLE gl_eudr_euis.operator_registrations (
    registration_id VARCHAR(100) PRIMARY KEY,
    operator_id UUID NOT NULL,
    operator_name VARCHAR(500) NOT NULL,
    operator_address TEXT,
    eori_number VARCHAR(50),
    nace_code VARCHAR(20),
    contact_person VARCHAR(200),
    contact_email VARCHAR(200),
    contact_phone VARCHAR(50),
    member_state CHAR(2) NOT NULL,
    registration_number VARCHAR(100),
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    registered_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    last_renewed_at TIMESTAMPTZ,
    is_sme BOOLEAN NOT NULL DEFAULT FALSE,
    delegation_active BOOLEAN NOT NULL DEFAULT FALSE,
    delegated_to VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_euis_reg_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

CREATE INDEX idx_euis_reg_operator ON gl_eudr_euis.operator_registrations(operator_id);
CREATE INDEX idx_euis_reg_ms ON gl_eudr_euis.operator_registrations(member_state);
CREATE INDEX idx_euis_reg_status ON gl_eudr_euis.operator_registrations(status);
CREATE INDEX idx_euis_reg_expires ON gl_eudr_euis.operator_registrations(expires_at);
CREATE UNIQUE INDEX idx_euis_reg_op_ms ON gl_eudr_euis.operator_registrations(operator_id, member_state);

-- ============================================================
-- Table 3: Formatted Geolocation Records
-- ============================================================
CREATE TABLE gl_eudr_euis.formatted_geolocations (
    geolocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dds_id VARCHAR(100),
    plot_id VARCHAR(100) NOT NULL,
    country_code CHAR(2) NOT NULL,
    coordinate_format VARCHAR(10) NOT NULL,
    coordinates JSONB NOT NULL,
    source_crs VARCHAR(20) NOT NULL DEFAULT 'EPSG:4326',
    precision_decimal_places INTEGER NOT NULL DEFAULT 6,
    area_hectares NUMERIC(12,4),
    polygon_closed BOOLEAN NOT NULL DEFAULT TRUE,
    winding_order VARCHAR(20),
    transformation_applied BOOLEAN NOT NULL DEFAULT FALSE,
    validation_passed BOOLEAN NOT NULL DEFAULT TRUE,
    validation_errors JSONB DEFAULT '[]',
    formatted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_euis_geo_dds ON gl_eudr_euis.formatted_geolocations(dds_id);
CREATE INDEX idx_euis_geo_plot ON gl_eudr_euis.formatted_geolocations(plot_id);
CREATE INDEX idx_euis_geo_country ON gl_eudr_euis.formatted_geolocations(country_code);

-- ============================================================
-- Table 4: Document Packages
-- ============================================================
CREATE TABLE gl_eudr_euis.document_packages (
    package_id VARCHAR(100) PRIMARY KEY,
    dds_id VARCHAR(100) NOT NULL,
    operator_id UUID NOT NULL,
    manifest JSONB NOT NULL DEFAULT '{}',
    total_size_bytes BIGINT NOT NULL DEFAULT 0,
    compressed_size_bytes BIGINT NOT NULL DEFAULT 0,
    segment_count INTEGER NOT NULL DEFAULT 1,
    package_hash VARCHAR(64) NOT NULL,
    s3_location VARCHAR(500),
    assembled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_euis_pkg_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

CREATE INDEX idx_euis_pkg_dds ON gl_eudr_euis.document_packages(dds_id);
CREATE INDEX idx_euis_pkg_operator ON gl_eudr_euis.document_packages(operator_id);

-- ============================================================
-- Table 5: Submission Status History (hypertable)
-- ============================================================
CREATE TABLE gl_eudr_euis.submission_status_history (
    history_id UUID DEFAULT gen_random_uuid(),
    submission_id VARCHAR(100) NOT NULL,
    previous_status VARCHAR(30),
    new_status VARCHAR(30) NOT NULL,
    eu_is_response JSONB,
    transition_reason TEXT,
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_euis.submission_status_history', 'transitioned_at');
CREATE INDEX idx_euis_hist_sub ON gl_eudr_euis.submission_status_history(submission_id, transitioned_at DESC);

-- ============================================================
-- Table 6: Audit Trail (hypertable -- immutable, append-only)
-- ============================================================
CREATE TABLE gl_eudr_euis.audit_trail (
    entry_id VARCHAR(100) NOT NULL,
    sequence_number BIGINT NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    operator_id UUID NOT NULL,
    dds_id VARCHAR(100),
    registration_id VARCHAR(100),
    member_state CHAR(2),
    action TEXT NOT NULL,
    request_hash VARCHAR(64),
    response_hash VARCHAR(64),
    actor VARCHAR(100) NOT NULL,
    correlation_id VARCHAR(100) NOT NULL,
    previous_entry_hash VARCHAR(64) NOT NULL,
    entry_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    retention_expiry DATE NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_euis.audit_trail', 'recorded_at');
CREATE INDEX idx_euis_audit_operator ON gl_eudr_euis.audit_trail(operator_id, recorded_at DESC);
CREATE INDEX idx_euis_audit_dds ON gl_eudr_euis.audit_trail(dds_id, recorded_at DESC);
CREATE INDEX idx_euis_audit_event ON gl_eudr_euis.audit_trail(event_type, recorded_at DESC);
CREATE INDEX idx_euis_audit_corr ON gl_eudr_euis.audit_trail(correlation_id);
CREATE INDEX idx_euis_audit_retention ON gl_eudr_euis.audit_trail(retention_expiry);

-- ============================================================
-- Table 7: Batch Submissions
-- ============================================================
CREATE TABLE gl_eudr_euis.batch_submissions (
    batch_id VARCHAR(100) PRIMARY KEY,
    operator_id UUID NOT NULL,
    total_dds INTEGER NOT NULL DEFAULT 0,
    submitted_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    pending_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(30) NOT NULL DEFAULT 'queued',
    priority VARCHAR(10) NOT NULL DEFAULT 'normal',
    concurrency INTEGER NOT NULL DEFAULT 5,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_euis_batch_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

CREATE INDEX idx_euis_batch_operator ON gl_eudr_euis.batch_submissions(operator_id);
CREATE INDEX idx_euis_batch_status ON gl_eudr_euis.batch_submissions(status);

-- ============================================================
-- Table 8: Registration Events (hypertable)
-- ============================================================
CREATE TABLE gl_eudr_euis.registration_events (
    event_id UUID DEFAULT gen_random_uuid(),
    registration_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    previous_status VARCHAR(30),
    new_status VARCHAR(30) NOT NULL,
    event_data JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_euis.registration_events', 'recorded_at');
CREATE INDEX idx_euis_reg_evt_reg ON gl_eudr_euis.registration_events(registration_id, recorded_at DESC);

-- ============================================================
-- Retention Policy: Prevent deletion within 5-year retention
-- ============================================================
-- Note: Application-level enforcement; database trigger as safety net
CREATE OR REPLACE FUNCTION gl_eudr_euis.prevent_audit_deletion()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.retention_expiry > CURRENT_DATE THEN
        RAISE EXCEPTION 'Cannot delete audit entry within 5-year retention period (expires: %)', OLD.retention_expiry;
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_prevent_audit_deletion
BEFORE DELETE ON gl_eudr_euis.audit_trail
FOR EACH ROW EXECUTE FUNCTION gl_eudr_euis.prevent_audit_deletion();
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **DDS Submission** | | |
| POST | `/v1/submissions` | Submit a single DDS to EU IS |
| POST | `/v1/submissions/batch` | Submit a batch of DDS documents |
| POST | `/v1/submissions/dry-run` | Validate and format without transmitting |
| GET | `/v1/submissions` | List all submissions (with filters: status, MS, date range) |
| GET | `/v1/submissions/{submission_id}` | Get submission details with status history |
| POST | `/v1/submissions/{submission_id}/resubmit` | Resubmit a rejected DDS |
| DELETE | `/v1/submissions/{submission_id}/withdraw` | Withdraw a pending submission |
| GET | `/v1/submissions/batch/{batch_id}` | Get batch submission status |
| POST | `/v1/submissions/batch/{batch_id}/cancel` | Cancel remaining items in batch |
| **Operator Registration** | | |
| POST | `/v1/registrations` | Submit new operator registration |
| GET | `/v1/registrations` | List all registrations (with filters: MS, status) |
| GET | `/v1/registrations/{registration_id}` | Get registration details |
| PUT | `/v1/registrations/{registration_id}` | Update registration details |
| POST | `/v1/registrations/{registration_id}/renew` | Initiate registration renewal |
| POST | `/v1/registrations/{registration_id}/delegate` | Delegate submission authority |
| DELETE | `/v1/registrations/{registration_id}/delegate` | Revoke delegation |
| GET | `/v1/registrations/validate/{member_state}` | Validate registration for a MS |
| **Geolocation Formatting** | | |
| POST | `/v1/geolocation/format` | Format geolocation data for EU IS |
| POST | `/v1/geolocation/format/batch` | Format batch of geolocations |
| POST | `/v1/geolocation/validate` | Validate geolocation data without formatting |
| GET | `/v1/geolocation/{dds_id}` | Get formatted geolocations for a DDS |
| **Document Packages** | | |
| POST | `/v1/packages/assemble` | Assemble document package for submission |
| GET | `/v1/packages/{package_id}` | Get package details and manifest |
| GET | `/v1/packages/{package_id}/download` | Download assembled package |
| **Status Tracking** | | |
| GET | `/v1/status/dashboard` | Get real-time submission dashboard data |
| GET | `/v1/status/{submission_id}/history` | Get full status history for a submission |
| POST | `/v1/status/poll` | Trigger manual status poll for specific submissions |
| **Audit Trail** | | |
| GET | `/v1/audit` | Query audit trail (with filters: operator, date range, event type) |
| GET | `/v1/audit/report` | Generate audit report (PDF/CSV export) |
| GET | `/v1/audit/verify` | Verify hash chain integrity |
| GET | `/v1/audit/{entry_id}` | Get specific audit entry with chain context |
| **Health and Diagnostics** | | |
| GET | `/health` | Service health check |
| GET | `/v1/diagnostics/eu-is` | EU IS connectivity and health status |
| GET | `/v1/diagnostics/rate-limits` | Current rate limit status per endpoint |
| GET | `/v1/diagnostics/circuit-breakers` | Circuit breaker states |

### 7.6 Prometheus Self-Monitoring Metrics (18)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_euis_submissions_total` | Counter | DDS submissions by status (accepted/rejected/error) |
| 2 | `gl_eudr_euis_submissions_latency_seconds` | Histogram | End-to-end submission latency |
| 3 | `gl_eudr_euis_batch_submissions_total` | Counter | Batch submission operations |
| 4 | `gl_eudr_euis_registrations_total` | Counter | Registration operations by type (initial/renewal/update) |
| 5 | `gl_eudr_euis_registrations_active` | Gauge | Currently active registrations per MS |
| 6 | `gl_eudr_euis_geolocations_formatted_total` | Counter | Geolocation formatting operations |
| 7 | `gl_eudr_euis_geolocation_errors_total` | Counter | Geolocation validation errors by type |
| 8 | `gl_eudr_euis_packages_assembled_total` | Counter | Document packages assembled |
| 9 | `gl_eudr_euis_packages_size_bytes` | Histogram | Package size distribution |
| 10 | `gl_eudr_euis_status_polls_total` | Counter | EU IS status poll operations |
| 11 | `gl_eudr_euis_rejections_total` | Counter | DDS rejections by reason category |
| 12 | `gl_eudr_euis_resubmissions_total` | Counter | Resubmission operations |
| 13 | `gl_eudr_euis_api_requests_total` | Counter | EU IS API requests by endpoint and status code |
| 14 | `gl_eudr_euis_api_latency_seconds` | Histogram | EU IS API request latency |
| 15 | `gl_eudr_euis_circuit_breaker_state` | Gauge | Circuit breaker state per endpoint (0=closed, 1=half-open, 2=open) |
| 16 | `gl_eudr_euis_rate_limit_remaining` | Gauge | Remaining rate limit per endpoint |
| 17 | `gl_eudr_euis_audit_entries_total` | Counter | Audit trail entries by event type |
| 18 | `gl_eudr_euis_errors_total` | Counter | Errors by operation type and severity |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| HTTP Client | httpx (async) | Async HTTP/2 support, connection pooling, timeout management |
| Geodetic Library | pyproj + Shapely | CRS transformations, polygon validation, geometry operations |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for audit trail |
| Cache | Redis | Submission status caching, rate limiter state, registration lookup |
| Object Storage | S3 | Document packages, audit trail archives |
| Queue | Redis (or Celery) | Submission queue management, batch job processing |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication (internal) | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authentication (EU IS) | eIDAS digital certificates | X.509 certificates, qualified electronic signatures |
| Authorization | RBAC via SEC-002 | Role-based submission and registration permissions |
| Encryption | AES-256-GCM via SEC-003 | Credential and certificate encryption at rest |
| Transport | TLS 1.3 via SEC-004 | Secure transport to EU IS endpoints |
| Secrets | HashiCorp Vault via SEC-006 | eIDAS certificate storage |
| Monitoring | Prometheus + Grafana | 18 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across submission pipeline |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-euis:submissions:read` | View DDS submission records and status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-euis:submissions:submit` | Submit DDS documents to EU IS | Compliance Officer, Admin |
| `eudr-euis:submissions:batch` | Submit batch DDS to EU IS | Compliance Officer, Admin |
| `eudr-euis:submissions:resubmit` | Resubmit rejected DDS | Compliance Officer, Admin |
| `eudr-euis:submissions:withdraw` | Withdraw pending submissions | Compliance Officer, Admin |
| `eudr-euis:registrations:read` | View operator registration status | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-euis:registrations:manage` | Create, update, renew registrations | Compliance Officer, Admin |
| `eudr-euis:registrations:delegate` | Manage submission delegation | Admin |
| `eudr-euis:geolocation:read` | View formatted geolocation data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-euis:geolocation:format` | Format geolocation data for EU IS | Analyst, Compliance Officer, Admin |
| `eudr-euis:packages:read` | View document packages | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-euis:packages:assemble` | Assemble document packages | Compliance Officer, Admin |
| `eudr-euis:status:read` | View submission status dashboard | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-euis:audit:read` | View audit trail records | Auditor (read-only), Compliance Officer, Admin |
| `eudr-euis:audit:export` | Export audit reports (PDF/CSV) | Auditor (read-only), Compliance Officer, Admin |
| `eudr-euis:diagnostics:read` | View EU IS connectivity diagnostics | IT Engineer, Admin |
| `eudr-euis:config:manage` | Manage EU IS endpoint configuration | Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-EUDR-030 Documentation Generator | DDSStatementGenerator, CompliancePackageBuilder | DDS documents (JSON/XML) and compliance packages -> submission pipeline |
| AGENT-EUDR-001 Supply Chain Mapping Master | SupplyChainGraphEngine | Supply chain data referenced in DDS geolocation sections |
| AGENT-EUDR-002 Geolocation Verification Agent | Verified plot coordinates | Raw verified coordinates -> GeolocationDataFormatter |
| AGENT-EUDR-006 Plot Boundary Manager Agent | Polygon boundaries | Plot polygons -> GeolocationDataFormatter |
| AGENT-EUDR-007 GPS Coordinate Validator Agent | Validated GPS coordinates | Validated coordinates -> GeolocationDataFormatter |
| AGENT-EUDR-012 Document Authentication Agent | Authenticated certificates | Certificate copies -> DocumentPackageAssembler |
| AGENT-DATA-005 EUDR Traceability Connector | EUSystemConnector | EU IS API specifications and endpoint configuration |
| SEC-006 Secrets Management (Vault) | eIDAS certificate storage | Digital certificates -> APIIntegrationManager |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | Submission status, registration status -> frontend dashboard |
| EUDR-030 Documentation Generator | Rejection feedback | Rejection reasons -> DDS correction guidance -> regenerated DDS |
| EUDR-033 Continuous Monitoring Agent | Submission metrics | Acceptance rates, rejection trends -> monitoring alerts |
| EUDR-034 Annual Review Scheduler | Registration expiry data | Registration renewal dates -> annual review calendar |
| External Auditors | Audit trail export | Audit reports for third-party verification |
| Competent Authorities | Audit trail inspection | Submission records for Article 14-16 inspections |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Single DDS Submission (Compliance Officer)

```
1. Compliance officer opens GL-EUDR-APP -> "EU Submissions" module
2. Selects a finalized DDS document from EUDR-030 output list
3. System displays pre-submission checklist:
   - DDS schema validation: PASSED
   - Operator registration (DE): ACTIVE (expires 2027-01-15)
   - Geolocation formatting: 47 plots formatted, 100% valid
   - Document package: 3.2 MB (within EU IS limit)
4. Officer clicks "Submit to EU Information System"
5. System authenticates with EU IS (eIDAS certificate)
6. System transmits DDS and waits for acknowledgement
7. EU IS returns: RECEIVED (reference: EU-DDS-2026-DE-00045678)
8. System records submission in status tracker
9. Officer sees real-time status: RECEIVED -> UNDER_REVIEW
10. 30 minutes later, status updates to ACCEPTED
11. System records acceptance; notifies officer
12. Audit trail shows complete submission timeline
```

#### Flow 2: Batch DDS Submission (Large Operator)

```
1. Compliance officer opens "Batch Submissions" view
2. Selects 150 finalized DDS documents from Q1 2026 portfolio
3. System performs pre-flight validation on all 150 DDS:
   - 148 pass all checks
   - 2 have geolocation precision warnings (4 decimal places)
4. System auto-fixes geolocation precision (pads to 6 decimals)
5. Officer reviews fixes and approves batch submission
6. System creates batch job with priority: NORMAL, concurrency: 5
7. Batch submission begins: real-time progress bar shows 0/150
8. Dashboard updates as each DDS is submitted:
   - Accepted: 143 | Pending: 4 | Rejected: 3
9. For 3 rejected DDS, system shows rejection analysis:
   - DDS-047: Invalid CN code (8523.11.00 not in EUDR Annex I)
   - DDS-089: Registration expired for NL (expired 2026-03-01)
   - DDS-112: Duplicate submission (already submitted 2026-03-10)
10. Officer corrects CN code for DDS-047 (routes to EUDR-030)
11. Officer initiates NL registration renewal (routes to Feature 2)
12. DDS-112 flagged as duplicate -- no action needed
13. Corrected DDS-047 resubmitted -> ACCEPTED
14. Batch summary report generated: 149 accepted, 1 duplicate
```

#### Flow 3: Operator Registration (New Member State)

```
1. Compliance officer opens "Registrations" view
2. Current registrations shown: DE (ACTIVE), NL (ACTIVE), BE (ACTIVE)
3. Officer clicks "Register in New Member State" -> selects FR
4. System pre-populates registration form with operator data
5. Officer reviews and confirms: legal name, address, EORI, NACE code
6. System submits registration to French competent authority via EU IS API
7. Status: PENDING (submitted 2026-03-13 10:24 UTC)
8. 5 business days later, status updates: ACTIVE
9. Registration number issued: FR-EUDR-2026-REG-00789
10. System validates all future FR-bound DDS reference this registration
11. Renewal alert scheduled: 90 days before expiry
12. Registration dashboard shows: DE, NL, BE, FR -- all ACTIVE
```

#### Flow 4: Rejection Handling and Resubmission

```
1. Status tracker detects: DDS GL-DDS-OP456-PALM-20260315-003 REJECTED
2. System parses rejection: FORMAT_ERROR -- geolocation polygon not closed
3. Rejection analysis identifies: Plot P-7823 has open polygon (first != last vertex)
4. System routes to GeolocationDataFormatter for automatic correction
5. Formatter auto-closes polygon by appending first vertex as last vertex
6. System regenerates geolocation section (coordinates EUDR-030 for DDS update)
7. Corrected DDS available for resubmission
8. System auto-resubmits (format errors are auto-correctable)
9. Resubmission ACCEPTED by EU IS
10. Audit trail shows: original -> rejected (reason) -> corrected -> resubmitted -> accepted
11. Compliance officer receives notification: "Rejection auto-resolved"
```

### 8.2 Key Screen Descriptions

**Submission Dashboard:**
- Top bar: Summary cards -- Total Submitted (150), Accepted (143), Pending (4), Rejected (3)
- Central area: Submission timeline chart showing daily submission volume and acceptance rate
- Left sidebar: Filters -- Member State, commodity, date range, status
- Table view: All submissions with columns: DDS ID, Commodity, MS, Status, EU IS Ref, Submitted At, Updated At
- Drill-down: Click any submission to see full status history and audit trail

**Registration Management:**
- Map view: EU map with colored pins per Member State (green = active, yellow = expiring, red = expired)
- Table view: All registrations with: MS, Registration #, Status, Expires, Last Renewed
- Alert panel: Upcoming renewals with countdown timers
- Action buttons: Register New MS, Renew, Update Details, Delegate

**Audit Trail Viewer:**
- Search bar: Filter by operator, DDS ID, event type, date range
- Timeline view: Chronological list of all EU IS interactions
- Detail panel: Click any entry to see full event data, request/response hashes, chain verification
- Export buttons: PDF Report, CSV Export
- Integrity check: "Verify Hash Chain" button with visual pass/fail result

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: DDS Submission Gateway -- individual and batch submission to EU IS
  - [ ] Feature 2: Operator Registration Manager -- registration lifecycle across Member States
  - [ ] Feature 3: Geolocation Data Formatter -- WGS 84, 6 decimals, polygon validation
  - [ ] Feature 4: Document Package Assembler -- compression, segmentation, manifest generation
  - [ ] Feature 5: Submission Status Tracker -- lifecycle tracking with polling and notifications
  - [ ] Feature 6: API Integration Manager -- eIDAS auth, rate limiting, circuit breaker, versioning
  - [ ] Feature 7: Audit Trail Recorder -- immutable log with SHA-256 hash chain, 5-year retention
  - [ ] Feature 8: Rejection Analysis and Resubmission Engine -- automated correction and resubmission
  - [ ] Feature 9: Multi-Member-State Submission Router -- routing with per-MS endpoint management
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (eIDAS certificate handling, credential encryption, JWT + RBAC integrated)
- [ ] Performance targets met (< 60 seconds submission p99, >= 50 DDS/hour batch throughput)
- [ ] EU IS API integration tested against EU IS sandbox/test environment (if available)
- [ ] Geolocation formatting validated against EU IS schema for all coordinate formats (point, polygon, multi-plot)
- [ ] Circuit breaker and retry logic tested under simulated EU IS failure conditions
- [ ] Audit trail hash chain integrity verified across 10,000+ entries
- [ ] Database migration V124 tested and validated
- [ ] Integration with EUDR-030 Documentation Generator verified end-to-end
- [ ] Integration with EUDR-006, EUDR-007 for geolocation data verified
- [ ] 5 beta customers successfully submitted DDS to EU IS through the platform
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ DDS submissions processed through the platform
- >= 95% first-attempt acceptance rate
- Average submission latency < 60 seconds
- 20+ operator registrations managed across EU Member States
- < 5 support tickets per customer related to EU IS submission
- 100% audit trail integrity (zero hash chain breaks)

**60 Days:**
- 500+ DDS submissions processed
- >= 97% first-attempt acceptance rate
- Batch submission throughput >= 50 DDS/hour sustained
- 50+ operator registrations active across 10+ Member States
- Automated format error correction resolving 90%+ of format rejections
- < 3 support tickets per customer

**90 Days:**
- 2,000+ DDS submissions processed
- >= 98% first-attempt acceptance rate
- Zero EUDR penalties for active customers attributable to submission failures
- 100+ operator registrations active across 15+ Member States
- Complete audit trails available for all customers (5-year retention verified)
- NPS > 50 from compliance officer persona

---

## 10. Timeline and Milestones

### Phase 1: Core Submission Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | API Integration Manager (Feature 6): eIDAS auth, connection pooling, rate limiting, circuit breaker | Senior Backend Engineer |
| 2-3 | DDS Submission Gateway (Feature 1): individual submission, pre-flight validation, acknowledgement capture | Senior Backend Engineer + Integration Engineer |
| 3-4 | Geolocation Data Formatter (Feature 3): WGS 84 normalization, polygon validation, multi-plot formatting | Backend Engineer (GIS) |
| 4-5 | Operator Registration Manager (Feature 2): registration lifecycle, multi-MS support, renewal tracking | Senior Backend Engineer |
| 5-6 | Document Package Assembler (Feature 4): compression, segmentation, manifest generation | Backend Engineer |

**Milestone: Core submission pipeline operational (Week 6)**

### Phase 2: Lifecycle, Resilience, and Audit (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Submission Status Tracker (Feature 5): polling, state transitions, dashboard data, webhook notifications | Senior Backend Engineer |
| 8-9 | Audit Trail Recorder (Feature 7): immutable log, SHA-256 hash chain, 5-year retention, audit reports | Backend Engineer |
| 9-10 | Rejection Analysis Engine (Feature 8): rejection parsing, correction routing, auto-resubmission | Backend Engineer |

**Milestone: Full lifecycle and audit capability operational (Week 10)**

### Phase 3: Multi-MS Routing, API, and Integration (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Multi-MS Submission Router (Feature 9): endpoint registry, routing rules, MS-specific handling | Backend Engineer |
| 12-13 | REST API Layer: 30+ endpoints, authentication, rate limiting, OpenAPI documentation | Backend Engineer |
| 13-14 | End-to-end integration with EUDR-030, EUDR-006, EUDR-007; RBAC integration; GL-EUDR-APP frontend integration | Integration Engineer + Frontend Engineer |

**Milestone: All 9 P0 features implemented with full integration (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 500+ tests, EU IS simulation, geolocation golden tests, audit chain verification | Test Engineer |
| 16-17 | Performance testing (submission latency, batch throughput), security audit (eIDAS, credential handling) | DevOps + Security |
| 17 | Database migration V124 finalized; EU IS sandbox integration testing | DevOps + Integration |
| 17-18 | Beta customer onboarding (5 customers); production launch readiness review | Product + Engineering |
| 18 | Go-live: all 9 P0 features verified, beta customers submitting successfully | All |

**Milestone: Production launch (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Submission Analytics Dashboard (Feature 10)
- Proactive Schema Compliance Monitor (Feature 11)
- Submission Scheduling and Calendar Integration (Feature 12)
- Performance optimization for 500+ DDS/hour batch throughput
- Additional Member State-specific adaptations

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-030 Documentation Generator | BUILT (100%) | Low | Stable, production-ready; DDS generation verified |
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable, production-ready |
| AGENT-EUDR-002 Geolocation Verification Agent | BUILT (100%) | Low | Stable; source of verified coordinates |
| AGENT-EUDR-006 Plot Boundary Manager Agent | BUILT (100%) | Low | Stable; source of polygon boundaries |
| AGENT-EUDR-007 GPS Coordinate Validator Agent | BUILT (100%) | Low | Stable; source of validated GPS |
| AGENT-EUDR-012 Document Authentication Agent | BUILT (100%) | Low | Stable; source of authenticated certificates |
| AGENT-DATA-005 EUDR Traceability Connector | BUILT (100%) | Low | EUSystemConnector provides EU IS integration foundation |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard internal auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration (17 permissions) |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | AES-256-GCM for credential encryption |
| SEC-004 TLS 1.3 Configuration | BUILT (100%) | Low | Transport security for EU IS communication |
| SEC-006 Secrets Management (Vault) | BUILT (100%) | Low | eIDAS certificate storage |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| Redis | Production Ready | Low | Caching and queue management |
| S3 Object Storage | Production Ready | Low | Package staging and archive storage |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System API | Published (v1.x) | High | Adapter pattern for API version changes; mock API for development; sandbox for testing |
| EU IS DDS schema specification | Published | Medium | Schema-driven validation; hot-reloadable schema definitions |
| eIDAS authentication infrastructure | Available | Medium | Certificate delegation support; fallback to OAuth 2.0 if supported |
| Member State competent authority endpoints | Variable per MS | Medium | Per-MS endpoint configuration; health monitoring; graceful degradation |
| EC country benchmarking list | Published; updated periodically | Low | Database-driven; hot-reloadable |
| CN/HS code reference database | Stable | Low | Versioned reference data; annual updates |
| pyproj coordinate transformation database | Stable | Low | Well-established geodetic library; tested CRS transformations |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EU Information System API specification changes before or after launch | High | High | Adapter pattern isolates EU IS API layer; version-specific transformers hot-swappable; monitor EU IS release notes |
| R2 | EU IS experiences extended outage during critical submission period (end of quarter) | Medium | Critical | Submission queue with persistent storage; automatic retry on recovery; alert compliance officers of delays; batch resubmission when service restored |
| R3 | eIDAS certificate management complexity exceeds initial estimates | Medium | High | Start with simplest supported auth method; escalate to full eIDAS only if required; partner with eIDAS certificate provider for managed service |
| R4 | Member State EU IS endpoints have inconsistent API implementations | Medium | High | Per-MS configuration and adapter layer; MS-specific test suites; community intelligence on MS variations |
| R5 | Geolocation data quality from upstream agents insufficient for EU IS validation | Medium | Medium | GeolocationDataFormatter applies corrections (precision padding, polygon closure, winding correction); flag uncorrectable issues for manual review |
| R6 | EU IS rate limits more restrictive than anticipated during peak periods | Medium | Medium | Distributed rate limiter; submission scheduling to spread load; priority queue for urgent submissions |
| R7 | 5-year audit trail storage costs exceed projections | Low | Medium | Tiered storage: hot (TimescaleDB, 1 year), warm (S3 Standard, 2 years), cold (S3 Glacier, 2 years); compression |
| R8 | Batch submission failures cause data consistency issues | Low | High | Transactional submission recording; per-DDS idempotency keys; batch partial failure handling |
| R9 | EU IS sandbox/test environment not available for pre-launch testing | Medium | Medium | Build comprehensive EU IS mock service; test against published API specification; coordinate with EU IS team for sandbox access |
| R10 | Operator registration requirements vary significantly across Member States | Medium | Medium | Configuration-driven MS-specific fields; modular registration form builder; MS-specific validation rules |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| DDS Submission Gateway Tests | 80+ | Individual submission, batch submission, pre-flight validation, acknowledgement capture, queue management |
| Operator Registration Tests | 60+ | Registration lifecycle, multi-MS, renewal, delegation, validation |
| Geolocation Formatter Tests | 70+ | WGS 84 normalization, CRS transformation, polygon validation, closure, winding, precision enforcement |
| Document Package Tests | 40+ | Assembly, compression, segmentation, manifest, multi-part upload |
| Status Tracker Tests | 50+ | Lifecycle tracking, polling, state transitions, rejection detection, webhook notifications |
| API Integration Tests | 60+ | eIDAS auth, rate limiting, circuit breaker, connection pooling, version negotiation, error recovery |
| Audit Trail Tests | 50+ | Immutable logging, hash chain integrity, retention enforcement, audit report generation, forensic queries |
| Rejection Analysis Tests | 40+ | Rejection parsing, correction routing, auto-resubmission, resubmission limits, bulk rejection handling |
| Multi-MS Router Tests | 30+ | Endpoint routing, MS health monitoring, failover, routing validation |
| API Endpoint Tests | 60+ | All 30+ endpoints, auth, error handling, pagination, filters |
| Integration Tests | 30+ | End-to-end with EUDR-030, EUDR-006, EUDR-007; EU IS mock service |
| Performance Tests | 20+ | Submission latency, batch throughput, concurrent submissions, status poll frequency |
| Geolocation Golden Tests | 35+ | All coordinate formats (point, polygon), all CRS transformations, all 7 commodities with multi-plot scenarios |
| **Total** | **625+** | |

### 13.2 EU IS Mock Service

A comprehensive EU IS mock service will be built for testing that simulates:

1. **DDS submission endpoint** -- Accepts valid DDS; rejects invalid DDS with realistic rejection reasons
2. **Registration endpoint** -- Simulates registration lifecycle per Member State
3. **Status query endpoint** -- Returns configurable status transitions with realistic timing
4. **Rate limiting** -- Simulates EU IS rate limits with 429 responses
5. **Outage simulation** -- Configurable failure injection (503, timeout, connection refused)
6. **Authentication** -- Simulates eIDAS certificate validation
7. **Schema validation** -- Validates submissions against published EU IS schema

### 13.3 Geolocation Golden Tests

Each of the 7 EUDR commodities will have dedicated geolocation golden tests covering:

1. **Single point** (plot <= 4 ha) -- WGS 84 coordinates with 6 decimal places
2. **Polygon** (plot > 4 ha) -- Closed ring, counter-clockwise winding, 6 decimal places
3. **Multi-plot single country** -- 50 plots in Brazil for soya
4. **Multi-plot multi-country** -- Plots across Indonesia, Malaysia, Ghana for palm oil
5. **CRS transformation** -- UTM Zone source coordinates transformed to WGS 84

Total: 7 commodities x 5 scenarios = 35 golden test scenarios

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration submitted to EU IS per Article 4/12 |
| **EU IS** | EU Information System -- the digital platform mandated by Article 33 for DDS submission |
| **eIDAS** | Electronic IDentification, Authentication and trust Services (Regulation (EU) No 910/2014) |
| **WGS 84** | World Geodetic System 1984 -- the coordinate reference system required by EUDR Article 9 |
| **EPSG:4326** | EPSG code for WGS 84 geographic coordinate system |
| **EORI** | Economic Operators Registration and Identification -- EU customs registration number |
| **NACE** | Nomenclature of Economic Activities -- EU economic activity classification code |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |
| **Member State (MS)** | An EU Member State (27 countries) |
| **Competent Authority** | National authority designated to enforce EUDR in each Member State |
| **Circuit Breaker** | Resilience pattern that prevents cascading failures by stopping requests to a failing service |
| **mTLS** | Mutual Transport Layer Security -- both client and server authenticate via certificates |
| **Qualified Electronic Signature** | eIDAS-defined signature with legal equivalence to handwritten signature |

### Appendix B: EU Information System API Reference (Summary)

| Endpoint Category | Base Path | Authentication | Rate Limit |
|-------------------|-----------|----------------|------------|
| DDS Submission | `/api/v1/dds` | eIDAS certificate | 60 RPM |
| DDS Status | `/api/v1/dds/{ref}/status` | eIDAS certificate | 120 RPM |
| Operator Registration | `/api/v1/registration` | eIDAS certificate | 30 RPM |
| Geolocation Upload | `/api/v1/geolocation` | eIDAS certificate | 60 RPM |
| Reference Data | `/api/v1/reference` | API key | 300 RPM |

Note: Actual EU IS API endpoints and rate limits will be confirmed from official EU IS technical documentation. The above represents estimated values based on published specifications and comparable government APIs.

### Appendix C: eIDAS Authentication Flow

```
1. Operator obtains qualified certificate from EU-accredited certificate provider
2. Certificate stored in HashiCorp Vault (SEC-006) with AES-256-GCM encryption
3. APIIntegrationManager retrieves certificate from Vault at session start
4. Mutual TLS handshake with EU IS endpoint:
   a. Platform presents operator's client certificate
   b. EU IS presents its server certificate
   c. Both certificates validated against trusted CA chain
5. TLS session established
6. DDS submission request sent over encrypted channel
7. EU IS validates operator identity from certificate CN/SAN fields
8. EU IS cross-references operator identity with registration number
9. Submission processed; response returned over same TLS session
10. Session cached for subsequent requests (configurable keepalive)
```

### Appendix D: Geolocation Format Specification

```json
// Single Point (plot <= 4 ha)
{
  "plot_id": "P-7823",
  "country": "BR",
  "type": "point",
  "coordinates": [-3.456000, -42.789000]
}

// Polygon (plot > 4 ha)
{
  "plot_id": "P-4501",
  "country": "ID",
  "type": "polygon",
  "coordinates": [
    [-2.345000, 104.567000],
    [-2.346000, 104.568000],
    [-2.347000, 104.567500],
    [-2.346500, 104.566000],
    [-2.345000, 104.567000]
  ]
}

// Multi-Plot Multi-Country (commodity from 3 countries)
{
  "dds_id": "GL-DDS-OP789-COCOA-20260315-001",
  "geolocation": {
    "GH": [
      {"plot_id": "P-001", "type": "point", "coordinates": [6.789000, -1.234000]},
      {"plot_id": "P-002", "type": "polygon", "coordinates": [[6.790000, -1.235000], ...]}
    ],
    "CI": [
      {"plot_id": "P-003", "type": "point", "coordinates": [5.678000, -4.567000]}
    ],
    "CM": [
      {"plot_id": "P-004", "type": "point", "coordinates": [4.567000, 9.876000]}
    ]
  }
}
```

### Appendix E: Submission State Machine

```
                    +--------+
                    | QUEUED |
                    +----+---+
                         |
                    +----v-------+
                    | SUBMITTING |
                    +----+-------+
                         |
                    +----v------+
              +---->| SUBMITTED |<----+
              |     +----+------+     |
              |          |            |
              |     +----v-----+     |
              |     | RECEIVED |     |
              |     +----+-----+     |
              |          |           |
              |   +------v--------+  |
              |   | UNDER_REVIEW  |  |
              |   +--+--------+---+  |
              |      |        |      |
         +----v---+  |  +-----v---+  |
         |ACCEPTED|  |  |REJECTED |  |
         +--------+  |  +----+----+  |
                      |       |      |
              +-------v---+   |      |
              |AMENDMENT  |   |      |
              |REQUIRED   |   +------+
              +-------+---+  (via RESUBMITTED)
                      |
                      +-------> (correction -> RESUBMITTED -> SUBMITTED)

         +----------+
         | WITHDRAWN| (can occur from QUEUED, SUBMITTED, RECEIVED, UNDER_REVIEW)
         +----------+
```

### Appendix F: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. Regulation (EU) No 910/2014 (eIDAS) -- Electronic Identification and Trust Services
3. Commission Implementing Regulation on the technical specifications for the EU Information System (Article 33)
4. EU EUDR Information System Technical Specifications (API documentation)
5. EPSG Geodetic Parameter Dataset -- EPSG:4326 (WGS 84)
6. RFC 7946 -- The GeoJSON Format
7. ISO 3166-1:2020 -- Country codes
8. ISO 8601 -- Date and time format
9. NIST SP 800-186 -- Recommendations for Discrete Logarithm-Based Cryptography (eIDAS signature algorithms)
10. GreenLang SEC-001 JWT Authentication Service Documentation
11. GreenLang SEC-003 Encryption at Rest (AES-256-GCM) Documentation
12. GreenLang SEC-004 TLS 1.3 Configuration Documentation
13. GreenLang SEC-006 Secrets Management (Vault) Documentation

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
| 1.0.0-draft | 2026-03-13 | GL-ProductManager | Initial draft created: 14 sections, 9 P0 features (DDS Submission Gateway, Operator Registration Manager, Geolocation Data Formatter, Document Package Assembler, Submission Status Tracker, API Integration Manager, Audit Trail Recorder, Rejection Analysis Engine, Multi-MS Router), regulatory coverage verified (Articles 4/9/12/14/15/16/29/31/33 + eIDAS), 30+ API endpoints, V124 migration, 18 Prometheus metrics, 17 RBAC permissions, 625+ test target |
