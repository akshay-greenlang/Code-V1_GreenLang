# PRD: AGENT-EUDR-040 -- Authority Communication Manager

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-040 |
| **Agent ID** | GL-EUDR-ACM-040 |
| **Component** | Authority Communication Manager Agent |
| **Category** | EUDR Regulatory Agent -- Reporting (Category 6) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-13 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 15 (Competent Authorities' Powers and Controls), 16 (Penalties and Enforcement Measures), 17 (Information Exchange Between Competent Authorities), 18 (Information Exchange with Third Countries), 19 (Administrative Procedures and Right to Appeal), 20 (Substantive Checks), 22 (Competent Authorities Designation), 31 (Record Keeping); General Data Protection Regulation (EU) 2016/679 (GDPR); eIDAS Regulation (EU) No 910/2014 (Electronic Identification); Regulation (EU) 2019/1020 (Market Surveillance and Compliance of Products) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **Prerequisite Agents** | AGENT-EUDR-001 through AGENT-EUDR-015 (Supply Chain Traceability), AGENT-EUDR-016 through AGENT-EUDR-025 (Risk Assessment and Due Diligence), AGENT-EUDR-026 (Due Diligence Orchestrator), AGENT-EUDR-030 (Documentation Generator), AGENT-EUDR-036 (EU Information System Interface), AGENT-EUDR-037 (Due Diligence Statement Creator), AGENT-EUDR-038 (Reference Number Generator), AGENT-EUDR-039 (Customs Declaration Support) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) establishes a comprehensive enforcement framework through Articles 15 through 19 that grants competent authorities in each of the 27 EU Member States extensive powers to communicate with, investigate, and penalize operators and traders. Article 15 empowers competent authorities to carry out checks and controls on operators and traders, including the power to request any information relevant to the operator's due diligence system, conduct on-the-spot inspections of business premises and production sites, take samples for analysis, and access documents, data, and information held by the operator in any form including electronic records. Article 16 mandates that Member States establish effective, proportionate, and dissuasive penalties for non-compliance, including fines proportionate to the environmental damage, the market value of the commodities concerned, and the operator's turnover -- with maximum fines of at least 4% of the operator's annual EU-wide turnover. Article 17 requires competent authorities to exchange information with competent authorities of other Member States and with the Commission where they have reasonable grounds to believe that an operator is not in compliance, establishing a cross-border information sharing obligation. Article 18 extends this information exchange to third countries, enabling competent authorities to share information with authorities in countries of production to verify legal production and deforestation-free status. Article 19 establishes the fundamental right of operators to administrative review and judicial appeal of any decision taken by competent authorities under the Regulation, requiring Member States to provide effective remedies including the right to be heard before adverse decisions are made, the right to access the file, the right to a reasoned decision, and the right to judicial review.

Beyond these specific articles, Article 20 requires competent authorities to perform substantive checks on at least a specified percentage of operators and quantities, with enhanced checking rates for products from high-risk countries. Article 22 requires each Member State to designate one or more competent authorities responsible for EUDR enforcement, creating a landscape of potentially 27+ different authority endpoints with varying communication protocols, languages, and procedural requirements. Article 31 requires operators to retain all records of communications with competent authorities for at least 5 years from the date of the due diligence statement, establishing an immutable record-keeping obligation for every interaction.

The GreenLang platform has built a comprehensive suite of 39 EUDR agents spanning the full compliance lifecycle. Supply Chain Traceability agents (EUDR-001 through EUDR-015) handle supply chain mapping, geolocation verification, satellite monitoring, forest cover analysis, land use change detection, plot boundary management, GPS coordinate validation, multi-tier supplier tracking, chain of custody management, segregation verification, mass balance calculation, document authentication, blockchain integration, QR code generation, and mobile data collection. Risk Assessment agents (EUDR-016 through EUDR-020) handle country risk evaluation, supplier risk scoring, commodity risk analysis, corruption index monitoring, and deforestation alerting. Due Diligence agents (EUDR-021 through EUDR-035) handle indigenous rights checking, protected area validation, legal compliance verification, third-party audit management, risk mitigation advisory, due diligence orchestration, information gathering coordination, risk assessment engine operation, mitigation measure design, documentation generation, stakeholder engagement, grievance mechanism management, continuous monitoring, annual review scheduling, and improvement plan creation. Reporting agents (EUDR-036 through EUDR-039) handle EU Information System interface, due diligence statement creation, reference number generation, and customs declaration support.

However, there is currently no agent that manages the critical two-way communication channel between operators and EU competent authorities. Operators face the following gaps:

- **No centralized authority communication management**: When competent authorities in any of 27 Member States send information requests, inspection notifications, non-compliance warnings, or penalty notices, these communications arrive through multiple channels (email, registered post, authority portals, EU Information System notifications) in different languages, formats, and with varying response deadlines. Operators have no centralized system to receive, classify, route, track, and respond to these communications. Critical deadlines are missed because communications are handled by individual staff members without shared visibility, leading to enforcement escalation.

- **No information request response automation**: Article 15 grants competent authorities the power to request "any information relevant to the operator's due diligence system." Responding to such requests requires assembling data from multiple upstream agents -- supply chain graphs from EUDR-001, geolocation data from EUDR-002/007, risk assessments from EUDR-016 through EUDR-025, mitigation measures from EUDR-029, DDS documents from EUDR-037, and evidence packages from EUDR-030. Currently, operators must manually locate, extract, compile, and transmit this information, a process that takes days or weeks and risks incomplete or inconsistent responses.

- **No DDS submission notification management**: When operators submit Due Diligence Statements through EUDR-036, competent authorities may request additional information, issue queries about specific data points, or acknowledge receipt with conditions. There is no engine that tracks the post-submission communication lifecycle between operators and authorities regarding specific DDS documents.

- **No inspection coordination capability**: Article 15 authorizes competent authorities to conduct on-the-spot inspections of operator premises. Inspections require coordination of logistics (scheduling, access provisioning, document preparation), assembly of all relevant compliance evidence, designation of responsible personnel, and post-inspection follow-up. There is no engine that manages the complete inspection lifecycle from notification receipt through inspection execution to findings response.

- **No non-compliance response workflow**: When competent authorities issue non-compliance findings under Article 16, operators must respond within strict deadlines with corrective action plans, evidence of remediation, and requests for deadline extensions where needed. Non-compliance notifications carry severe consequences (fines up to 4% of EU turnover, goods confiscation, market exclusion, public naming). There is no structured workflow to receive non-compliance notifications, parse the specific findings, generate corrective action plans by leveraging upstream agent capabilities, track remediation progress, and submit formal responses to the authority.

- **No appeal process management**: Article 19 grants operators the right to administrative review and judicial appeal of any adverse decision. Appeal processes have strict procedural requirements including filing deadlines, required documentation, evidence submission formats, and hearing scheduling. There is no engine that manages appeal timelines, assembles appeal documentation from platform data, tracks appeal status through administrative and judicial proceedings, and ensures that no deadline is missed.

- **No secure document exchange with authorities**: Competent authorities frequently request specific documents -- certificates, satellite imagery, supply chain maps, risk assessment reports, contracts, and invoices. These documents contain commercially sensitive and personally identifiable information that must be transmitted securely, with access controls, encryption, and audit trails. There is no secure document exchange portal that allows operators to share specific documents with specific authorities while maintaining confidentiality, integrity, and non-repudiation.

- **No multi-language communication support**: The 27 EU Member States have 24 official languages. Competent authority communications arrive in the official language of the Member State concerned. Operators with EU-wide operations receive communications in German, French, Italian, Spanish, Portuguese, Dutch, Polish, Romanian, and numerous other languages. There is no engine that detects the language of incoming communications, provides translation support for response preparation, and formats outgoing communications in the required language of the receiving authority.

- **No notification routing to correct authorities**: Different Member States have designated different competent authorities for EUDR enforcement under Article 22. Some Member States have designated environment ministries, others have designated customs authorities, and others have designated dedicated agencies. The correct authority endpoint varies by Member State, by commodity, and by the type of communication. There is no routing engine that maintains the registry of all 27+ competent authorities with their contact endpoints, jurisdictions, communication protocols, and language requirements.

- **No comprehensive communication audit trail**: Article 31 requires 5-year retention of all records, which explicitly includes all communications with competent authorities. Every incoming communication, every outgoing response, every document exchange, every inspection record, every non-compliance finding, every corrective action, and every appeal filing must be recorded in an immutable, tamper-proof audit trail. There is no engine that maintains this complete communication history with the integrity guarantees required for regulatory inspection.

Without solving these problems, operators cannot effectively manage their relationship with 27 EU competent authorities. Missed response deadlines trigger enforcement escalation. Incomplete information responses lead to adverse findings. Uncoordinated inspection preparation exposes compliance gaps. Failed appeal deadlines result in irrevocable penalties. The absence of a communication management layer between operators and competent authorities is the single most dangerous gap in the EUDR compliance lifecycle, because it is through this communication channel that penalties are imposed, goods are confiscated, and market access is revoked.

### 1.2 Solution Overview

Agent-EUDR-040: Authority Communication Manager is the specialized two-way communication management agent that handles all interactions between operators/traders and the competent authorities of the 27 EU Member States under EUDR Articles 15 through 19. It receives and classifies incoming communications from competent authorities across all channels, routes them to the correct internal stakeholders, assembles responses by pulling data from all 39 upstream EUDR agents, manages inspection coordination, handles non-compliance response workflows, manages appeal processes, provides a secure document exchange portal, supports 24 EU official languages, maintains the complete registry of EU competent authorities, and records every communication in an immutable 5-year audit trail per Article 31.

Core capabilities:

1. **Information Request Handler** -- Receives, classifies, and responds to information requests from competent authorities under Article 15 and Article 17. Parses the scope of each request (which DDS, which commodity, which time period, which specific data elements), identifies the relevant upstream agents that hold the requested data, assembles response packages by pulling data from EUDR-001 through EUDR-039, formats responses in the format and language required by the requesting authority, tracks response deadlines with escalation alerts, and records all request-response pairs in the audit trail. Supports bulk information requests where authorities request data for multiple products or time periods simultaneously.

2. **DDS Submission Notification Engine** -- Manages the post-submission communication lifecycle between operators and competent authorities regarding Due Diligence Statements. Tracks DDS acknowledgements from EUDR-036, processes authority queries about specific DDS data points, prepares supplementary information responses, handles conditional acceptance notices requiring additional evidence, manages amendment requests triggered by authority feedback, and coordinates with EUDR-037 for DDS amendments and EUDR-036 for resubmissions. Provides a unified view of all DDS-related authority communications per product and per Member State.

3. **Inspection Coordination Engine** -- Manages the complete lifecycle of competent authority inspections under Article 15 and Article 20. Receives inspection notification from authorities, parses the inspection scope (which facilities, which commodities, which compliance areas), coordinates scheduling with internal stakeholders, assembles all relevant compliance documentation from upstream agents, prepares facility-specific compliance packages, tracks inspector assignments and access provisioning, manages real-time inspection support (document requests during inspection), captures inspection findings, generates post-inspection response plans, and tracks finding remediation to closure. Supports both announced and unannounced inspection scenarios.

4. **Non-Compliance Response Manager** -- Handles the complete non-compliance workflow from initial notification through remediation and closure. Receives non-compliance findings from competent authorities under Article 16, parses specific non-compliance items and their severity (warning, corrective action required, fine, confiscation, market exclusion), maps each finding to the relevant upstream agent and compliance gap, generates corrective action plans leveraging EUDR-025 Risk Mitigation Advisor and EUDR-029 Mitigation Measure Designer, tracks remediation implementation with evidence collection, prepares formal response submissions to the authority, monitors authority review of corrective actions, and manages escalation scenarios including fine negotiations and payment tracking.

5. **Appeal Process Manager** -- Manages the complete administrative and judicial appeal lifecycle under Article 19. Calculates appeal filing deadlines from the date of the adverse decision, assembles appeal documentation by compiling evidence from all relevant upstream agents, formats appeal submissions according to the procedural requirements of the specific Member State and authority, tracks appeal proceedings through administrative review and judicial stages, manages hearing preparation including evidence packages and legal argument summaries, records appeal outcomes, and implements required actions following appeal decisions (compliance adjustments, fine payments, or continued operations if appeal succeeds).

6. **Document Exchange Portal** -- Provides a secure, encrypted file-sharing channel between operators and competent authorities. Enables operators to share specific documents (certificates, satellite imagery, supply chain maps, risk assessments, contracts, invoices) with designated authority recipients. Implements end-to-end encryption (AES-256-GCM), access control lists per document, time-limited download links, watermarking for confidential documents, read receipts and download tracking, virus scanning for incoming documents, file format validation, and size limit management. Supports both operator-initiated document sharing and authority-initiated document requests.

7. **Multi-Language Communication Engine** -- Supports communication in all 24 official EU languages across the 27 Member States. Detects the language of incoming communications using NLP-based language identification, provides machine translation for internal review using EU-approved translation services, prepares outgoing communications in the required language of the receiving authority, maintains terminology databases for EUDR-specific legal and technical terms in all 24 languages, supports human review and approval workflows for critical translated communications, and handles character encoding for all European scripts including Latin, Greek, and Cyrillic alphabets.

8. **Notification Routing Engine** -- Maintains the complete registry of competent authorities designated under Article 22 across all 27 EU Member States and routes communications to the correct authority. Maps each Member State to its designated competent authority or authorities (some Member States designate multiple authorities for different commodities or functions), maintains authority contact endpoints (portal URLs, API endpoints, email addresses, postal addresses), tracks authority personnel and department assignments, determines the correct authority for each communication based on Member State, commodity, communication type, and procedural context, and handles cross-border routing for Article 17 inter-authority information exchange and Article 18 third-country information exchange.

9. **Communication Audit Trail Manager** -- Maintains a comprehensive, immutable, tamper-proof audit trail of every communication between operators and competent authorities per Article 31. Records every incoming communication (with full content, metadata, timestamp, source authority, channel), every outgoing response (with full content, metadata, timestamp, destination authority, channel), every document exchange (with document hashes, access logs, download confirmations), every inspection event (notification, execution, findings, responses), every non-compliance interaction (finding, corrective action, remediation evidence, closure), and every appeal event (filing, hearing, decision, implementation). Implements SHA-256 hash chains linking consecutive records to prevent tampering. Enforces 5-year minimum retention with configurable extended retention. Supports competent authority audit by generating complete communication histories for any DDS, product, commodity, operator, or time period on demand.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Information request response time | < 24 hours for standard requests, < 2 hours for urgent requests | Time from request receipt to response submission |
| Response completeness rate | >= 98% of information requests answered completely on first response | Authority follow-up request rate (lower = better) |
| Inspection readiness time | < 4 hours from notification to full documentation package ready | Time from inspection notification to package assembly complete |
| Non-compliance response deadline compliance | 100% of responses submitted before authority deadlines | Deadline miss rate (target: zero) |
| Appeal filing deadline compliance | 100% of appeals filed before procedural deadlines | Deadline miss rate (target: zero) |
| Document exchange security | Zero unauthorized access incidents | Security audit findings + penetration test results |
| Language detection accuracy | >= 97% correct language identification | Automated language detection accuracy testing |
| Authority routing accuracy | 100% of communications routed to correct competent authority | Routing error rate (target: zero) |
| Audit trail completeness | 100% of communications recorded with full metadata | Audit trail gap detection automated testing |
| Audit trail integrity | 100% of hash chains valid across 5-year retention | Automated hash chain verification |
| Platform availability | 99.9% uptime | Infrastructure monitoring |
| Communication processing throughput | >= 500 communications per hour | Load test sustained throughput |
| GDPR compliance | 100% PII handling per GDPR requirements | Data protection audit results |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ EUDR-affected operators and traders across the EU must communicate with competent authorities during checks, inspections, and enforcement actions. The regulatory communication management market for EUDR is estimated at 1-3 billion EUR, driven by the complexity of managing communications across 27 Member States with 24 official languages.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring systematic authority communication management, estimated at 300M-700M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 25-40M EUR in authority communication module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) operating across multiple Member States and subject to enhanced checking rates under Article 20
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya) with complex authority relationships across multiple EU jurisdictions
- Timber and paper industry operators subject to frequent EUDR inspections due to high-risk country sourcing
- Compliance teams managing authority interactions across multiple EU subsidiaries

**Secondary:**
- Legal and compliance departments handling EUDR appeals and non-compliance responses
- Customs brokers and freight forwarders interfacing with customs authorities on EUDR matters
- Compliance consultants managing authority communications on behalf of operator clients
- SME importers (1,000-10,000 shipments/year) needing structured authority communication workflows (enforcement from June 30, 2026)

**Tertiary:**
- External law firms handling EUDR administrative and judicial appeals
- Industry associations coordinating collective responses to authority inquiries
- Third-country exporters communicating with EU authorities under Article 18

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual Process (email, phone, postal) | No cost; familiar to staff | No tracking; missed deadlines; no audit trail; inconsistent responses; no multi-language support | Centralized, tracked, auditable, automated, multi-lingual |
| Generic CRM/Ticketing Systems (Salesforce, ServiceNow) | Enterprise workflow; case management | Not EUDR-specific; no regulatory deadline tracking; no upstream agent integration; no authority registry | Purpose-built for EUDR Articles 15-19; deep integration with 39 upstream agents |
| Legal Case Management (Clio, Thomson Reuters) | Legal workflow; appeal tracking | Not EUDR-specific; no environmental regulation support; no supply chain data integration | EUDR-specific appeal management; automated evidence assembly from supply chain agents |
| Government Communication Platforms (EU portal, Member State portals) | Official channels; authority-mandated | Fragmented across 27 Member States; no unified view; no automated response assembly; no internal routing | Unified hub connecting all 27 authority endpoints with automated response capabilities |
| Compliance Management Suites (SAP GRC, MetricStream) | Enterprise governance; multi-regulation | Generic compliance; no EUDR-specific communication management; no authority registry; no multi-language EUDR support | Deep EUDR regulatory knowledge; 27 Member State authority registry; 24-language support |
| In-house Custom Builds | Tailored to organization | 12-18 month build time; no regulatory updates; no authority registry maintenance; no multi-language | Ready now; continuously updated authority registry; automated multi-language support |

### 2.4 Differentiation Strategy

1. **Deep upstream integration** -- Pulls compliance data from all 39 EUDR agents to assemble complete, accurate authority responses in minutes instead of days.
2. **27 Member State authority registry** -- Maintained, verified authority contact database with routing rules for every communication type, commodity, and jurisdiction.
3. **24-language communication support** -- Automatic language detection and translation support for all official EU languages.
4. **EUDR-specific workflow automation** -- Purpose-built workflows for Article 15 information requests, Article 16 non-compliance responses, Article 19 appeals, and Article 20 inspection coordination.
5. **Immutable audit trail** -- SHA-256 hash-chained communication records satisfying Article 31 for 5+ years.
6. **Zero-deadline-miss guarantee** -- Automated deadline tracking with multi-level escalation ensures no authority deadline is ever missed.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to manage all competent authority communications from a single platform | 100% of authority communications processed through the platform | Q2 2026 |
| BG-2 | Reduce information request response time from days/weeks to hours | 90% reduction in average response time | Q2 2026 |
| BG-3 | Achieve zero missed authority deadlines for active customers | 0% deadline miss rate | Ongoing |
| BG-4 | Reduce non-compliance escalation rate through proactive response management | 50% reduction in escalated non-compliance cases | Q4 2026 |
| BG-5 | Support operators in successful EUDR appeals with comprehensive evidence assembly | 80%+ appeal success rate for substantively justified appeals | Q4 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Centralized communication hub | Unified inbox for all authority communications across 27 Member States |
| PG-2 | Automated response assembly | Pull data from 39 upstream agents to compile complete responses |
| PG-3 | Inspection readiness | Full compliance documentation package assembled within 4 hours of notification |
| PG-4 | Non-compliance workflow | Structured workflow from finding to corrective action to closure |
| PG-5 | Appeal lifecycle management | End-to-end appeal management with deadline tracking and evidence assembly |
| PG-6 | Secure document exchange | Encrypted, auditable document sharing with competent authorities |
| PG-7 | Multi-language support | 24 EU official language support for incoming and outgoing communications |
| PG-8 | Authority registry | Maintained database of all 27 Member State competent authorities |
| PG-9 | Complete audit trail | Immutable 5-year communication history per Article 31 |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Communication processing latency | < 500ms p99 for incoming communication classification and routing |
| TG-2 | Response assembly throughput | Assembly of information request response packages in < 30 seconds |
| TG-3 | Document exchange transfer rate | < 10 seconds per 100 MB document upload/download |
| TG-4 | Language detection accuracy | >= 97% correct language identification across 24 EU languages |
| TG-5 | API response time | < 200ms p95 for standard queries |
| TG-6 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-7 | Audit trail integrity | 100% SHA-256 hash chain validity verified continuously |
| TG-8 | Concurrent communication handling | 1,000+ simultaneous communications without degradation |

---

## 4. User Personas

### Persona 1: Compliance Director -- Dr. Elena Vasquez (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Director of Regulatory Compliance at a large EU food manufacturer |
| **Company** | 8,000 employees, importing cocoa, palm oil, and soya from 15 countries, operating subsidiaries in 6 EU Member States |
| **EUDR Pressure** | Receiving information requests and inspection notifications from competent authorities in Germany, France, Netherlands, and Belgium simultaneously |
| **Pain Points** | Communications arrive in 4 different languages through 4 different channels; no unified tracking; response deadlines vary by Member State; has missed one deadline resulting in a formal warning; cannot assemble comprehensive responses because data is spread across multiple systems; legal team spends 40+ hours per authority interaction |
| **Goals** | Single dashboard showing all active authority communications across all Member States; automated deadline tracking with escalation; one-click response assembly pulling data from all compliance agents; multi-language support for communication preparation |
| **Technical Skill** | Moderate -- comfortable with enterprise software but not a developer |

### Persona 2: Legal Counsel -- Maitre Pierre Fontaine (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Legal Counsel, EU Regulatory Affairs at a multinational timber importer |
| **Company** | 3,000 employees, importing tropical wood from Southeast Asia and Central Africa, headquarters in France with offices in 4 EU Member States |
| **EUDR Pressure** | Currently managing two non-compliance proceedings and one pending appeal across Germany and France; facing potential fine of EUR 2.5M |
| **Pain Points** | Appeal deadlines differ by Member State (14 days in Germany, 30 days in France); evidence assembly for appeals requires data from 15+ agents; translation of appeal documents between French and German is costly and slow; no centralized view of all proceedings; risk of procedural errors that invalidate appeals |
| **Goals** | Automated appeal deadline calculation per Member State procedural law; one-click evidence assembly for appeal documentation; professional translation support; unified view of all non-compliance and appeal proceedings; compliance with Article 19 procedural requirements |
| **Technical Skill** | Moderate -- experienced with legal technology platforms |

### Persona 3: Inspection Coordinator -- Jan van der Berg (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Quality and Compliance Manager at a palm oil refinery |
| **Company** | 1,200 employees, single site in Rotterdam, Netherlands, processing palm oil from Indonesia and Malaysia |
| **EUDR Pressure** | Subject to enhanced checking under Article 20 due to high-risk country sourcing; expecting 4-6 inspections per year |
| **Pain Points** | Last inspection required 3 weeks to assemble all requested documentation; inspectors requested satellite imagery, geolocation data, chain of custody records, and risk assessments that were stored in different systems; post-inspection findings response took 6 weeks instead of the 30-day deadline |
| **Goals** | Pre-built inspection readiness packages that can be assembled in hours; real-time document provision during inspections; structured findings response with automatic evidence linking; inspection scheduling and logistics coordination |
| **Technical Skill** | High -- comfortable with technical systems and data management |

### Persona 4: Compliance Analyst -- Sofia Papadopoulos (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | EUDR Compliance Analyst at a coffee trading company |
| **Company** | 500 employees, trading coffee from 8 origin countries, operating from Greece |
| **EUDR Pressure** | Receives regular information requests from Greek competent authority in Greek language; must respond with technical compliance data in Greek |
| **Pain Points** | Platform data is in English; must translate all technical terms to Greek; no standardized EUDR terminology in Greek; response formatting must follow Greek administrative law procedures; small team cannot handle volume of authority communications efficiently |
| **Goals** | Automated translation of compliance data to Greek; pre-built response templates in Greek; EUDR terminology database in Greek; efficient communication workflow for small team |
| **Technical Skill** | Moderate-High -- comfortable with compliance software and data analysis |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 15(1)** | Competent authorities shall carry out checks on operators and traders to verify compliance | Inspection Coordination Engine manages the complete inspection lifecycle including scheduling, preparation, execution support, and findings response |
| **Art. 15(2)(a)** | Power to request any information relevant to the operator's due diligence system | Information Request Handler receives, classifies, and assembles responses by pulling data from all 39 upstream agents |
| **Art. 15(2)(b)** | Power to carry out on-the-spot inspections of business premises | Inspection Coordination Engine handles announced and unannounced inspection logistics, documentation assembly, and access provisioning |
| **Art. 15(2)(c)** | Power to take samples of relevant commodities or products | Inspection Coordination Engine coordinates sample collection logistics and tracks sample analysis results and chain of custody |
| **Art. 15(2)(d)** | Power to require any person to provide information relevant to compliance | Information Request Handler handles third-party information requests routed through the operator, with document exchange for evidence provision |
| **Art. 16(1)** | Member States shall lay down rules on penalties for infringements | Non-Compliance Response Manager receives penalty notifications, parses severity and fine amounts, generates corrective action plans, and manages payment workflows |
| **Art. 16(1)(a)** | Fines proportionate to environmental damage and market value | Non-Compliance Response Manager tracks fine calculations, supports proportionality challenges, and manages fine payment or appeal |
| **Art. 16(1)(b)** | Confiscation of the relevant commodities or products | Non-Compliance Response Manager handles goods confiscation notifications, coordinates with customs (EUDR-039), and manages release requests |
| **Art. 16(1)(c)** | Confiscation of revenues gained by the operator | Non-Compliance Response Manager tracks revenue confiscation orders and coordinates with finance departments |
| **Art. 16(1)(d)** | Temporary exclusion from public procurement processes | Non-Compliance Response Manager tracks exclusion orders, manages reinstatement applications, and monitors exclusion status |
| **Art. 16(1)(e)** | Temporary prohibition of placing products on the market | Non-Compliance Response Manager handles market prohibition orders, tracks affected products, and manages prohibition lifting requests |
| **Art. 16(2)** | Maximum fine shall be at least 4% of annual EU-wide turnover | Non-Compliance Response Manager calculates potential fine exposure, tracks actual fines against this threshold, and supports proportionality arguments in appeals |
| **Art. 17(1)** | Competent authorities shall cooperate with each other and with the Commission | Notification Routing Engine routes inter-authority information exchange requests, manages operator obligations to facilitate information sharing between authorities |
| **Art. 17(2)** | Exchange of information between competent authorities of different Member States | Notification Routing Engine handles cross-border communication routing when one Member State authority requests information about operations in another Member State |
| **Art. 18(1)** | Competent authorities may exchange information with third-country authorities | Document Exchange Portal supports secure information sharing with third-country authorities under formal cooperation agreements, with GDPR-compliant data handling |
| **Art. 18(2)** | Information exchange with countries of production | Information Request Handler assembles country-of-production verification data for third-country authority requests, coordinating with EUDR-001 supply chain data |
| **Art. 19(1)** | Right to an effective remedy and administrative review | Appeal Process Manager manages the complete appeal lifecycle from decision receipt through administrative review to judicial appeal |
| **Art. 19(2)** | Right to be heard before adverse decisions | Appeal Process Manager tracks pre-decision hearing rights, manages hearing preparation, and ensures operators exercise their right to be heard before penalties are imposed |
| **Art. 20(1)** | Risk-based approach to substantive checks | Inspection Coordination Engine adjusts inspection readiness posture based on operator risk profile and expected checking frequency |
| **Art. 20(2)** | Enhanced checking rates for high-risk country products | Inspection Coordination Engine implements elevated inspection preparedness for high-risk commodity flows |
| **Art. 22** | Designation of competent authorities by Member States | Notification Routing Engine maintains the complete registry of designated competent authorities across all 27 Member States |
| **Art. 31(1)** | Record keeping for at least 5 years | Communication Audit Trail Manager records all communications with SHA-256 hash chains and enforces 5-year minimum retention |
| **Art. 31(2)** | Records shall be made available to competent authorities on request | Communication Audit Trail Manager generates on-demand communication histories for authority inspection in the format and language required |

### 5.2 EU Member State Competent Authority Landscape

| Member State | Primary Competent Authority (Typical) | Language(s) | Communication Protocol |
|-------------|---------------------------------------|-------------|----------------------|
| **Austria** | Federal Ministry for Climate Action, Environment, Energy | German | Portal + email |
| **Belgium** | Federal Public Service Health, Food Chain Safety and Environment | French, Dutch, German | Portal + registered post |
| **Bulgaria** | Ministry of Environment and Water | Bulgarian | Portal + email |
| **Croatia** | Ministry of Economy and Sustainable Development | Croatian | Portal + registered post |
| **Cyprus** | Department of Forests, Ministry of Agriculture | Greek | Email + postal |
| **Czech Republic** | Czech Environmental Inspectorate | Czech | Portal + data box |
| **Denmark** | Danish Environmental Protection Agency | Danish | Digital post (e-Boks) |
| **Estonia** | Environmental Board | Estonian | Portal + email |
| **Finland** | Finnish Customs / Finnish Food Authority | Finnish, Swedish | Portal + email |
| **France** | Ministry of Ecological Transition / DGCCRF | French | Teleservice portal |
| **Germany** | Federal Agency for Agriculture and Food (BLE) | German | Portal + De-Mail |
| **Greece** | Ministry of Environment and Energy | Greek | Portal + email |
| **Hungary** | National Food Chain Safety Office | Hungarian | Portal + email |
| **Ireland** | Department of Agriculture, Food and the Marine | English, Irish | Portal + email |
| **Italy** | Carabinieri Forestali / Ministry of Agriculture | Italian | PEC (certified email) |
| **Latvia** | State Forest Service | Latvian | Portal + email |
| **Lithuania** | State Forest Service | Lithuanian | Portal + email |
| **Luxembourg** | Administration of Nature and Forests | French, German, Luxembourgish | Portal + email |
| **Malta** | Environment and Resources Authority | Maltese, English | Portal + email |
| **Netherlands** | Netherlands Food and Consumer Product Safety Authority (NVWA) | Dutch | Portal + email |
| **Poland** | General Directorate of Environmental Protection | Polish | Portal + ePUAP |
| **Portugal** | Institute for Nature Conservation and Forests | Portuguese | Portal + email |
| **Romania** | National Environmental Guard | Romanian | Portal + registered post |
| **Slovakia** | Slovak Environment Inspectorate | Slovak | Portal + email |
| **Slovenia** | Inspectorate for the Environment and Spatial Planning | Slovenian | Portal + email |
| **Spain** | Ministry for the Ecological Transition | Spanish | Sede electronica portal |
| **Sweden** | Swedish Board of Agriculture | Swedish | Portal + email |

### 5.3 Communication Types and Regulatory Basis

| Communication Type | Direction | EUDR Article | Urgency | Typical Deadline | Agent Handler |
|-------------------|-----------|-------------|---------|-----------------|---------------|
| Information request (standard) | Authority -> Operator | Art. 15(2)(a) | Medium | 30 days | Information Request Handler |
| Information request (urgent) | Authority -> Operator | Art. 15(2)(a) | Critical | 48-72 hours | Information Request Handler |
| DDS query/clarification | Authority -> Operator | Art. 4, 12 | Medium | 14-30 days | DDS Submission Notification Engine |
| DDS conditional acceptance | Authority -> Operator | Art. 12 | High | Varies | DDS Submission Notification Engine |
| Inspection notification (announced) | Authority -> Operator | Art. 15(2)(b) | High | 5-15 days notice | Inspection Coordination Engine |
| Inspection notification (unannounced) | Authority -> Operator | Art. 15(2)(b) | Critical | Immediate | Inspection Coordination Engine |
| Sample collection notice | Authority -> Operator | Art. 15(2)(c) | High | At inspection | Inspection Coordination Engine |
| Non-compliance warning | Authority -> Operator | Art. 16 | High | 30 days to respond | Non-Compliance Response Manager |
| Corrective action order | Authority -> Operator | Art. 16 | Critical | 14-60 days | Non-Compliance Response Manager |
| Fine notification | Authority -> Operator | Art. 16(1)(a) | Critical | 30 days to pay/appeal | Non-Compliance Response Manager |
| Goods confiscation order | Authority -> Operator | Art. 16(1)(b) | Critical | Immediate | Non-Compliance Response Manager |
| Market exclusion order | Authority -> Operator | Art. 16(1)(e) | Critical | Immediate | Non-Compliance Response Manager |
| Pre-decision hearing invitation | Authority -> Operator | Art. 19(2) | Critical | 7-14 days | Appeal Process Manager |
| Adverse decision notification | Authority -> Operator | Art. 19 | Critical | 14-60 day appeal window | Appeal Process Manager |
| Inter-authority information request | Authority -> Operator | Art. 17 | Medium | 30 days | Notification Routing Engine |
| Third-country information exchange | Authority -> Operator | Art. 18 | Low-Medium | 60 days | Information Request Handler |
| Information response | Operator -> Authority | Art. 15(2)(a) | Per deadline | Per request | Information Request Handler |
| Corrective action plan | Operator -> Authority | Art. 16 | Per deadline | Per order | Non-Compliance Response Manager |
| Appeal filing | Operator -> Authority | Art. 19 | Critical | Per Member State law | Appeal Process Manager |
| Document submission | Operator -> Authority | Art. 15 | Per deadline | Per request | Document Exchange Portal |
| General inquiry | Operator -> Authority | General | Low | N/A | Notification Routing Engine |

### 5.4 Key Regulatory Dates and Enforcement Timeline

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date referenced in all competent authority deforestation inquiries |
| June 29, 2023 | Regulation entered into force | Legal basis for all competent authority powers and operator obligations |
| December 30, 2024 | Member States designate competent authorities | Authority registry must be populated with all designated authorities |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Full communication management required; inspection and enforcement actions begin |
| June 30, 2026 | Enforcement for SMEs | SME communication management; expanded authority interaction volume |
| Ongoing | Article 20 risk-based checks | Continuous inspection coordination and response management |
| Ongoing | Article 17 inter-authority cooperation | Cross-border communication routing as authorities share information |
| Per event | Article 19 appeal deadlines | Appeal deadline calculation per Member State procedural law upon adverse decision |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core communication management engine handling the five primary communication workflows (information requests, DDS notifications, inspections, non-compliance, appeals). Features 6-9 form the infrastructure layer providing secure document exchange, multi-language support, authority routing, and audit trail management.

**P0 Features 1-5: Core Communication Workflow Engines**

---

#### Feature 1: Information Request Handler

**User Story:**
```
As a compliance director,
I want to receive, classify, and respond to information requests from competent authorities through a single centralized system,
So that I can ensure every request is answered completely, accurately, and within the required deadline using data from my EUDR compliance platform.
```

**Acceptance Criteria:**
- [ ] Receives information requests from all channels: EU IS notifications (via EUDR-036), authority portal messages, email (POP3/IMAP), and manual entry for postal communications
- [ ] Classifies each request by type (standard, urgent, bulk), scope (specific DDS, commodity, time period, general), requesting authority (Member State, department, officer), and regulatory basis (Article 15, 17, 18)
- [ ] Parses request content to identify specific data elements requested (supply chain data, geolocation, risk assessments, certificates, DDS documents, financial records)
- [ ] Maps requested data elements to upstream EUDR agents: supply chain data from EUDR-001, geolocation from EUDR-002/006/007, risk assessments from EUDR-016 through EUDR-025, mitigation from EUDR-029, DDS from EUDR-037, evidence from EUDR-030
- [ ] Assembles response packages by pulling data from identified upstream agents automatically, without manual data extraction
- [ ] Formats response packages in the format required by the requesting authority (PDF, JSON, XML, CSV, or structured portal submission)
- [ ] Calculates response deadline from request receipt date based on Member State procedural rules (14, 30, or 60 days) and urgency classification
- [ ] Sends deadline reminders at configurable intervals (7 days, 3 days, 1 day, same day) with escalation to management
- [ ] Supports partial responses when not all requested data is immediately available, with acknowledgement and timeline for completion
- [ ] Tracks response status through the complete lifecycle: received, acknowledged, in_progress, response_assembled, response_reviewed, response_submitted, authority_confirmed
- [ ] Handles bulk information requests where authorities request data for multiple products, shipments, or time periods in a single request
- [ ] Generates response quality score based on completeness, accuracy, and timeliness metrics

**Non-Functional Requirements:**
- Performance: Response package assembly from upstream agents in < 30 seconds for standard requests
- Reliability: Zero lost requests; every incoming request must be recorded and tracked
- Scalability: Handle 100+ concurrent open information requests across multiple Member States
- Auditability: Complete audit trail of every request, response, and intermediate action

**Dependencies:**
- EUDR-036 EU Information System Interface (for EU IS notification ingestion)
- EUDR-001 through EUDR-039 (for response data assembly)
- Email integration service (POP3/IMAP for email channel ingestion)
- SEC-001 JWT Authentication and SEC-002 RBAC (for access control)

**Edge Cases:**
- Authority requests data that the operator does not possess -- Generate "data not available" response with explanation and offer to assist authority in locating data through alternative channels
- Request references a DDS that was withdrawn -- Include withdrawal documentation and reference any replacement DDS
- Request arrives in a language not yet configured -- Route to multi-language engine for translation before processing; do not delay acknowledgement
- Multiple authorities request the same data simultaneously (Article 17 cross-border scenario) -- Detect duplicate requests, prepare consistent responses, route to correct authorities

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 integration engineer)

---

#### Feature 2: DDS Submission Notification Engine

**User Story:**
```
As a compliance officer,
I want to manage all communications with competent authorities regarding my Due Diligence Statement submissions,
So that I can respond promptly to authority queries, provide supplementary information when requested, and maintain a complete record of all DDS-related authority interactions.
```

**Acceptance Criteria:**
- [ ] Receives DDS-related notifications from competent authorities via EUDR-036 EU Information System Interface (acknowledgements, queries, conditional acceptances, rejection notices)
- [ ] Links every authority communication to the specific DDS document(s) it references, using DDS reference numbers from EUDR-038
- [ ] Classifies DDS notifications by type: acknowledgement, query/clarification, conditional acceptance, amendment request, rejection with reasons, compliance verification request
- [ ] For DDS queries: parses the specific data points questioned, identifies the relevant upstream agent data, and prepares clarification responses
- [ ] For conditional acceptances: tracks conditions, assembles required supplementary evidence from upstream agents, and coordinates submission of additional materials
- [ ] For amendment requests: triggers DDS amendment workflow through EUDR-037 (DDS Creator) and EUDR-036 (EU IS Interface), tracking the amendment lifecycle
- [ ] For rejection notices: parses rejection reasons, maps to specific DDS data elements, generates correction guidance, and coordinates resubmission
- [ ] Provides a unified DDS communication dashboard showing all authority interactions per DDS, per product, per Member State
- [ ] Tracks notification response deadlines and sends escalation alerts
- [ ] Maintains association between DDS documents and their complete communication histories for Article 31 compliance

**Non-Functional Requirements:**
- Performance: DDS notification processing and classification in < 5 seconds
- Integration: Bidirectional real-time sync with EUDR-036, EUDR-037, and EUDR-038
- Completeness: 100% of DDS notifications linked to their source DDS documents
- Traceability: Full provenance chain from DDS creation through every authority interaction

**Dependencies:**
- EUDR-036 EU Information System Interface (for notification ingestion and response submission)
- EUDR-037 Due Diligence Statement Creator (for amendment coordination)
- EUDR-038 Reference Number Generator (for DDS reference number resolution)
- EUDR-030 Documentation Generator (for supplementary evidence assembly)

**Edge Cases:**
- Authority queries a DDS that has already been amended -- Link query to the current version and include amendment history
- Rejection of a DDS for a shipment already cleared by customs (EUDR-039) -- Coordinate with customs module for potential post-clearance audit and corrective action
- Simultaneous queries from multiple Member State authorities about the same DDS (Article 17 scenario) -- Ensure consistent responses to all authorities

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 3: Inspection Coordination Engine

**User Story:**
```
As an inspection coordinator,
I want to manage the complete lifecycle of competent authority inspections from notification through findings response,
So that my organization is always prepared for inspections, can provide requested documentation instantly, and can respond to findings within required deadlines.
```

**Acceptance Criteria:**
- [ ] Receives and processes inspection notifications from competent authorities, extracting inspection date, scope, inspector details, and facility/commodity focus
- [ ] Distinguishes between announced inspections (advance notice) and unannounced inspections (immediate or short notice)
- [ ] For announced inspections: generates pre-inspection preparation checklist, identifies required documentation, assigns preparation tasks to responsible personnel, and tracks preparation progress
- [ ] Assembles inspection readiness package by pulling data from upstream agents: supply chain maps (EUDR-001), geolocation evidence (EUDR-002/006/007), satellite imagery (EUDR-003/004/005), chain of custody records (EUDR-009), segregation verification (EUDR-010), mass balance data (EUDR-011), certificates (EUDR-012), risk assessments (EUDR-016-025), DDS documents (EUDR-037), and all evidence packages (EUDR-030)
- [ ] Manages inspector access provisioning: generates read-only access credentials for inspectors to view relevant platform data during inspection, with scope limited to the inspection focus area
- [ ] Supports real-time document requests during inspection: inspectors can request additional documents through a secure portal, and the system assembles and delivers them in < 5 minutes
- [ ] Captures inspection findings, observations, and preliminary conclusions as structured data
- [ ] Generates post-inspection response plans mapping each finding to corrective actions, responsible parties, deadlines, and evidence requirements
- [ ] Tracks finding remediation progress with evidence collection and deadline management
- [ ] Produces inspection summary reports for internal management and external authority communication
- [ ] Maintains inspection history per facility, per commodity, per authority for trend analysis and inspection preparedness optimization

**Non-Functional Requirements:**
- Performance: Inspection readiness package assembly in < 4 hours for announced inspections; < 30 minutes for pre-assembled standard packages
- Availability: 99.99% during active inspection windows (no downtime during scheduled inspections)
- Security: Inspector access scoped to specific data; no access to unrelated operator data; access automatically revoked after inspection period
- Scalability: Handle 10+ concurrent inspections across multiple facilities and Member States

**Dependencies:**
- EUDR-001 through EUDR-039 (for comprehensive documentation assembly)
- SEC-001 JWT Authentication (for inspector temporary access provisioning)
- SEC-002 RBAC (for scoped read-only inspector roles)
- AGENT-FOUND-006 Access & Policy Guard (for inspection-scoped access policies)

**Edge Cases:**
- Unannounced inspection at a facility where compliance staff are not present -- Auto-generate and deliver standard inspection readiness package to nearest available compliance staff; provide inspector portal access immediately
- Inspector requests data outside the stated inspection scope -- Flag as scope expansion, log the request, provide data if available while noting the scope change
- Inspection findings contradict data in the platform (e.g., inspector observes different conditions than reported) -- Create discrepancy record, trigger investigation workflow, prevent automatic overwrite of inspector findings
- Multiple concurrent inspections at different facilities by different Member State authorities -- Manage each inspection independently with dedicated resource allocation

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 frontend engineer)

---

#### Feature 4: Non-Compliance Response Manager

**User Story:**
```
As a legal counsel,
I want a structured workflow to manage non-compliance findings from competent authorities, from initial notification through corrective action and closure,
So that my organization can respond within required deadlines, implement effective remediation, and minimize the risk of escalation to fines, confiscation, or market exclusion.
```

**Acceptance Criteria:**
- [ ] Receives non-compliance notifications from competent authorities through all channels (EU IS, portal, email, postal)
- [ ] Parses non-compliance notifications to extract: finding reference number, specific non-compliance items, regulatory basis (which EUDR article violated), severity classification (warning, corrective action required, preliminary fine, final fine, confiscation order, market exclusion order), response deadline, and required remediation actions
- [ ] Classifies non-compliance severity using a 5-level system: Level 1 (advisory), Level 2 (formal warning), Level 3 (corrective action required), Level 4 (financial penalty), Level 5 (goods confiscation/market exclusion)
- [ ] Maps each non-compliance finding to the relevant upstream EUDR agent and specific compliance gap (e.g., "missing geolocation for Plot X" maps to EUDR-002/007; "inadequate risk assessment for Supplier Y" maps to EUDR-016-025)
- [ ] Generates corrective action plans for each finding by leveraging: EUDR-025 Risk Mitigation Advisor for risk-based remediation strategies, EUDR-029 Mitigation Measure Designer for specific measure design, EUDR-035 Improvement Plan Creator for structured improvement plans
- [ ] Tracks corrective action implementation with evidence collection from upstream agents
- [ ] Prepares formal response submissions to the authority including: acknowledgement of findings, corrective action plan with timeline, evidence of remediation steps taken, and request for extensions where justified
- [ ] Manages fine payment workflows including: fine amount verification, payment deadline tracking, installment plan management where available, and payment confirmation
- [ ] Manages goods confiscation workflows including: notification to customs (EUDR-039), release request preparation, evidence assembly for goods release, and tracking of confiscated goods status
- [ ] Manages market exclusion workflows including: affected product identification, exclusion period tracking, reinstatement application preparation, and evidence assembly for reinstatement
- [ ] Tracks overall non-compliance exposure: total open findings, total pending fines, affected products, and compliance improvement trajectory
- [ ] Generates non-compliance trend reports for management and board reporting

**Non-Functional Requirements:**
- Responsiveness: Non-compliance notification processing and internal alert generation in < 1 minute
- Reliability: Zero missed response deadlines (automated escalation to C-level if deadline is within 48 hours)
- Auditability: Complete audit trail of every non-compliance interaction, corrective action, and evidence submission
- Confidentiality: Non-compliance data restricted to authorized personnel (legal, compliance, management) with separate RBAC controls

**Dependencies:**
- EUDR-025 Risk Mitigation Advisor (for remediation strategy)
- EUDR-029 Mitigation Measure Designer (for specific measure design)
- EUDR-035 Improvement Plan Creator (for structured improvement plans)
- EUDR-039 Customs Declaration Support (for confiscation/release coordination)
- SEC-002 RBAC (for restricted access to non-compliance data)

**Edge Cases:**
- Authority imposes a fine that exceeds the 4% turnover threshold of Article 16(2) -- Flag as potentially disproportionate, generate proportionality analysis, recommend appeal
- Multiple authorities issue non-compliance findings for the same underlying issue (cross-border operations) -- Detect duplicate findings, coordinate unified response, inform each authority of parallel proceedings
- Corrective action deadline is impossible to meet (e.g., requiring supply chain restructuring in 14 days) -- Auto-generate extension request with justification and interim remediation plan
- Non-compliance finding is factually incorrect (authority error) -- Prepare factual correction submission with supporting evidence from upstream agents

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 legal workflow specialist)

---

#### Feature 5: Appeal Process Manager

**User Story:**
```
As a legal counsel,
I want to manage the complete appeal lifecycle for EUDR adverse decisions, from calculating filing deadlines to assembling evidence packages to tracking proceedings,
So that my organization can exercise its Article 19 rights effectively and protect itself from unjustified penalties.
```

**Acceptance Criteria:**
- [ ] Detects adverse decisions from competent authorities that are eligible for appeal (fines, confiscation orders, market exclusion orders, corrective action orders, registration rejections)
- [ ] Calculates appeal filing deadlines based on the specific Member State's administrative procedural law (varies from 14 days to 60 days depending on jurisdiction and decision type)
- [ ] Maintains a database of Member State appeal procedures: filing requirements, required documents, competent appeal bodies (administrative tribunal, court), procedural languages, format requirements, and fee schedules
- [ ] Generates appeal submission packages including: formal appeal letter, statement of grounds, evidence compilation from all relevant upstream agents, legal argument summary, procedural compliance checklist
- [ ] Assembles evidence for appeal by pulling comprehensive data from upstream agents: supply chain traceability evidence (EUDR-001-015), risk assessment documentation (EUDR-016-025), due diligence evidence (EUDR-026-035), DDS submission records (EUDR-036-039), and all communication history (this agent)
- [ ] Manages pre-decision hearing preparation under Article 19(2): hearing scheduling, evidence preparation, presentation materials, and hearing minutes recording
- [ ] Tracks appeal proceedings through stages: filing_submitted, acknowledgement_received, hearing_scheduled, hearing_completed, decision_pending, appeal_granted, appeal_denied, judicial_review_initiated
- [ ] Manages judicial review (second-level appeal to courts) when administrative appeal is denied
- [ ] Calculates potential financial impact of appeal (fine amount, legal costs, opportunity cost of market exclusion) to support go/no-go appeal decisions
- [ ] Tracks appeal outcomes and lessons learned for future compliance improvement
- [ ] Generates appeal status reports for management and board, including success probability assessment based on evidence strength

**Non-Functional Requirements:**
- Deadline compliance: Zero missed appeal filing deadlines (critical -- a missed deadline forfeits the right to appeal)
- Completeness: Appeal evidence packages include all available data from upstream agents
- Confidentiality: Appeal data restricted to legal team with separate RBAC controls
- Multi-jurisdiction: Support appeal procedures for all 27 EU Member States

**Dependencies:**
- EUDR-001 through EUDR-039 (for comprehensive evidence assembly)
- SEC-002 RBAC (for restricted access to appeal data)
- Multi-Language Communication Engine (Feature 7) for translations of appeal documents

**Edge Cases:**
- Appeal deadline falls on a weekend or public holiday -- Apply Member State-specific rules for deadline extension (most Member States extend to next business day)
- Operator wants to appeal in one Member State while complying with similar finding in another -- Support parallel strategies per Member State
- Appeal body requests additional information during proceedings -- Handle supplementary evidence requests through Information Request Handler integration
- Appeal is partially successful (some findings overturned, others upheld) -- Parse partial decision, update non-compliance records accordingly, manage remaining remediation

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 legal workflow specialist)

---

**P0 Features 6-9: Communication Infrastructure Layer**

> Features 6, 7, 8, and 9 are P0 launch blockers. Without secure document exchange, multi-language support, authority routing, and audit trail management, the core communication workflow engines (Features 1-5) cannot function effectively across 27 EU Member States. These features provide the foundational infrastructure on which all communication workflows depend.

---

#### Feature 6: Document Exchange Portal

**User Story:**
```
As a compliance director,
I want a secure, encrypted channel to exchange documents with competent authorities,
So that I can share commercially sensitive compliance evidence with confidence that it will only be accessed by authorized authority personnel.
```

**Acceptance Criteria:**
- [ ] Enables operators to upload documents for sharing with specific competent authorities, with recipient designation per document
- [ ] Implements end-to-end encryption using AES-256-GCM for documents at rest and TLS 1.3 for documents in transit
- [ ] Generates time-limited download links (configurable expiry: 24 hours to 30 days) for authority document access
- [ ] Supports access control lists (ACLs) per document: which authority, which department, which officer can access
- [ ] Applies document watermarking with authority name, access timestamp, and confidentiality classification for shared documents
- [ ] Tracks document access: records every download with timestamp, IP address, user agent, and authority identity
- [ ] Implements read receipts: notifies operator when authority has accessed/downloaded a shared document
- [ ] Supports inbound documents from authorities: receives uploaded documents with virus scanning (ClamAV or equivalent), file format validation, and size limit enforcement (max 500 MB per file, 2 GB per package)
- [ ] Validates file formats against allowed list: PDF, DOCX, XLSX, CSV, JSON, XML, GeoJSON, PNG, JPG, TIFF, ZIP
- [ ] Implements document versioning: when a document is replaced, previous versions are retained with version history
- [ ] Supports batch document sharing: upload multiple documents in a single operation with shared metadata and recipient list
- [ ] Provides document search: full-text search across shared document metadata, titles, and descriptions
- [ ] Generates document exchange reports showing all sharing activity for a given authority, time period, or DDS reference

**Non-Functional Requirements:**
- Security: AES-256-GCM encryption at rest; TLS 1.3 in transit; no unencrypted storage of shared documents
- Performance: Upload/download transfer rate > 10 MB/s for standard connections
- Storage: S3-backed with automatic tiering (hot -> warm -> cold) based on access frequency
- GDPR: Full compliance with GDPR requirements for data handling, including right to erasure where not conflicting with Article 31 retention
- Availability: 99.9% uptime for document exchange operations

**Dependencies:**
- SEC-003 Encryption at Rest (AES-256-GCM)
- SEC-004 TLS 1.3 Configuration
- INFRA-004 S3/Object Storage (for document storage)
- SEC-007 Security Scanning Pipeline (for virus scanning)
- SEC-011 PII Detection/Redaction (for PII handling in shared documents)

**Edge Cases:**
- Authority attempts to access a document after the download link has expired -- Return clear message with instructions to request a new link; log the access attempt
- Document contains PII that should be redacted before sharing with certain authority types -- Integrate with SEC-011 PII Detection/Redaction for automated redaction options
- Authority uploads a malicious file -- Virus scanning blocks the upload; alert operator; quarantine file for analysis
- Document exchange link is shared with unauthorized personnel -- Watermarking identifies the leak source; access revocation capability

**Estimated Effort:** 3 weeks (1 backend engineer, 1 security engineer)

---

#### Feature 7: Multi-Language Communication Engine

**User Story:**
```
As a compliance analyst operating across multiple EU Member States,
I want the system to detect the language of incoming authority communications and support response preparation in the required language,
So that I can communicate effectively with competent authorities in any EU Member State without relying on external translation services for routine communications.
```

**Acceptance Criteria:**
- [ ] Detects the language of incoming communications using NLP-based language identification with >= 97% accuracy across all 24 official EU languages
- [ ] Supports the 24 official EU languages: Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Irish, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish
- [ ] Provides machine translation of incoming communications for internal review, clearly marked as machine translation with confidence scores
- [ ] Maintains EUDR-specific terminology databases in all 24 languages, covering: regulatory terms (due diligence, deforestation-free, operator, trader, competent authority), technical terms (geolocation, polygon, emission factor, mass balance), commodity terms (all 7 commodities and derived products), and procedural terms (inspection, non-compliance, appeal, corrective action)
- [ ] Formats outgoing communications in the required language of the receiving authority, using: machine translation for draft preparation, EUDR terminology database for technical accuracy, and human review workflow for critical communications (non-compliance responses, appeals)
- [ ] Supports human review and approval workflow for translated communications: translator queue, side-by-side original/translation view, term suggestion from EUDR glossary, approval by authorized reviewer
- [ ] Handles character encoding for all European scripts: Latin (most EU languages), Greek (Greek), Cyrillic (Bulgarian), and special characters (accented letters, umlauts, cedillas)
- [ ] Generates bilingual communications when required by authority protocol (e.g., Belgium requires Dutch/French, Luxembourg requires French/German)
- [ ] Provides real-time language switching in the user interface for multilingual compliance teams
- [ ] Tracks translation quality metrics: post-edit distance, terminology accuracy, human correction rate

**Non-Functional Requirements:**
- Accuracy: >= 97% language detection accuracy; >= 90% machine translation adequacy for routine communications
- Performance: Language detection in < 500ms; machine translation in < 5 seconds per page
- Terminology: EUDR glossary maintained with >= 500 terms per language, updated quarterly
- Privacy: Translation processing compliant with GDPR; no training on operator data

**Dependencies:**
- Machine translation service (EU-hosted translation API, DeepL, or equivalent GDPR-compliant service)
- NLP language detection library (fastText, langdetect, or equivalent)
- EUDR terminology database (custom-built, maintained by GL-ProductManager)

**Edge Cases:**
- Communication contains mixed languages (e.g., French letter with German annexes) -- Detect language per section, translate each section separately
- Authority uses regional dialect or informal language not well-supported by machine translation -- Flag for human review, provide glossary suggestions
- EUDR-specific term has no official translation in a particular language (new regulation, evolving terminology) -- Use the English term with a explanatory note in the target language until official translation is established
- Communication is in a non-EU language (e.g., Chinese from third-country authority under Article 18) -- Detect and flag; route to manual translation workflow; acknowledge receipt in English

**Estimated Effort:** 3 weeks (1 NLP engineer, 1 backend engineer)

---

#### Feature 8: Notification Routing Engine

**User Story:**
```
As a compliance director managing operations across multiple EU Member States,
I want all communications to be automatically routed to and from the correct competent authority based on Member State, commodity, and communication type,
So that no communication is misdirected and every response reaches the correct regulatory endpoint.
```

**Acceptance Criteria:**
- [ ] Maintains a comprehensive registry of competent authorities designated under Article 22 for all 27 EU Member States, including: authority name, department, jurisdiction (commodity scope, geographic scope, functional scope), contact endpoints (portal URL, API endpoint, email, postal address), communication protocols (portal submission, certified email, registered post, digital post), language requirements, and key personnel contacts
- [ ] Determines the correct authority for each outgoing communication based on: Member State of the operator's establishment, Member State where the product is placed on the market, commodity category, type of communication (DDS-related, inspection-related, non-compliance-related, appeal-related), and any special routing rules (e.g., customs-related communications routed to customs authority rather than environment authority)
- [ ] Handles cross-border routing for Article 17 inter-authority information exchange: when Authority A in Member State X requests information about operations in Member State Y, routes through the correct Authority B in Member State Y
- [ ] Handles third-country routing for Article 18: routes information exchange requests to/from third-country authorities through the appropriate EU competent authority intermediary
- [ ] Tracks authority personnel changes and organizational restructuring, updating the registry when authorities change contact points or jurisdictions
- [ ] Supports operator registration with multiple Member State authorities per Article 14, tracking registration status per authority
- [ ] Validates outgoing communications against authority-specific format requirements before submission
- [ ] Manages communication channel selection: when an authority supports multiple channels (portal + email), selects the preferred channel based on authority guidelines and communication urgency
- [ ] Generates authority relationship map showing all active interactions per authority with communication volume and status metrics
- [ ] Implements failover routing: if the primary communication channel to an authority is unavailable, automatically routes through the secondary channel with appropriate notification

**Non-Functional Requirements:**
- Accuracy: 100% routing accuracy to correct authority (zero misdirected communications)
- Currency: Authority registry updated within 30 days of any published authority change
- Availability: Routing engine available 99.9% of the time
- Scalability: Handle routing decisions for 1,000+ communications per day across all 27 Member States

**Dependencies:**
- European Commission published list of designated competent authorities (Article 22)
- EUDR-036 EU Information System Interface (for EU IS-based authority communication)
- Authority portal integration modules (per Member State)

**Edge Cases:**
- Member State has not yet designated a competent authority (transition period) -- Route to the Ministry of Environment as default, flag for manual review
- Member State redesignates its competent authority (authority changes) -- Update registry, redirect pending communications, notify operator of the change
- Communication concerns operations in a Member State where the operator is not registered -- Flag as potential Article 14 compliance issue, recommend registration before communication
- Authority endpoint is temporarily unavailable -- Queue communication, retry with backoff, alert operator if delay exceeds 24 hours, attempt alternative channel

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 9: Communication Audit Trail Manager

**User Story:**
```
As a compliance officer subject to Article 31 record-keeping requirements,
I want every communication between my organization and competent authorities to be recorded in an immutable, tamper-proof audit trail,
So that I can demonstrate complete communication compliance during authority inspections and retain all records for the required 5-year period.
```

**Acceptance Criteria:**
- [ ] Records every incoming communication with: full content (text, attachments), source authority (Member State, department, officer), communication channel (EU IS, portal, email, postal), receipt timestamp (UTC and local), classification (type, urgency, scope), SHA-256 content hash, and linked DDS reference numbers
- [ ] Records every outgoing communication with: full content (text, attachments), destination authority (Member State, department, officer), communication channel, submission timestamp, response deadline, SHA-256 content hash, linked DDS reference numbers, and approval chain (who reviewed and approved)
- [ ] Records every document exchange event: document ID, document hash, sharing direction (operator->authority or authority->operator), access events (downloads, views), timestamps, and recipient identity
- [ ] Records every inspection event: notification receipt, preparation activities, inspector access grants, documents provided during inspection, findings received, corrective actions planned, remediation evidence submitted, and closure confirmation
- [ ] Records every non-compliance event: finding receipt, severity classification, corrective action plan submission, remediation progress, evidence submissions, authority response, and finding closure
- [ ] Records every appeal event: adverse decision receipt, deadline calculation, appeal filing, evidence submission, hearing events, appeal decision, and implementation actions
- [ ] Implements SHA-256 hash chains linking consecutive audit trail entries to prevent tampering or deletion
- [ ] Enforces 5-year minimum retention from the date of the associated DDS submission per Article 31
- [ ] Supports configurable extended retention (7 years, 10 years) for operators subject to additional regulatory requirements
- [ ] Generates comprehensive communication history reports on demand: per DDS, per product, per commodity, per authority, per Member State, per time period
- [ ] Supports competent authority inspection of communication records: generates formatted audit trail exports in the language and format requested by the inspecting authority
- [ ] Implements automated integrity verification: scheduled daily hash chain validation with alert on any integrity failure
- [ ] Provides search and filtering across the entire audit trail: by date range, authority, communication type, DDS reference, commodity, and free text

**Non-Functional Requirements:**
- Immutability: Audit trail entries cannot be modified or deleted once written (append-only storage)
- Integrity: SHA-256 hash chain verified daily; any break in the chain triggers immediate alert
- Retention: 5-year minimum with automated retention tracking and expiry alerting
- Performance: Audit trail query response < 2 seconds for typical queries; < 30 seconds for full history export
- Storage: Compressed audit trail storage using TimescaleDB continuous aggregates for efficient querying
- Compliance: GDPR-compliant PII handling within audit trail records

**Dependencies:**
- INFRA-002 PostgreSQL + TimescaleDB (for time-series audit trail storage)
- SEC-005 Centralized Audit Logging (for platform-wide audit trail integration)
- SEC-003 Encryption at Rest (for encrypted audit trail storage)
- AGENT-FOUND-008 Reproducibility Agent (for hash chain integrity verification)

**Edge Cases:**
- Audit trail entry contains PII that is subject to GDPR erasure request -- Per GDPR Article 17(3)(b), retain data required for compliance with legal obligations (EUDR Article 31 overrides erasure for regulatory records); document the legal basis for retention
- Hash chain integrity check fails (data corruption or tampering) -- Immediately alert security team, isolate affected entries, restore from backup if available, generate incident report
- Audit trail storage approaches capacity limits -- Automated archival to cold storage (S3 Glacier) with retrieval capability within 24 hours
- Communication from 5+ years ago requested by authority in a new investigation -- Verify retention status; if within retention period, provide; if beyond retention period, provide notice that records have been purged per retention policy

**Estimated Effort:** 3 weeks (1 backend engineer, 1 security engineer)

---

### 6.2 Should-Have Features (P1 -- High Priority)

#### Feature 10: Communication Analytics Dashboard

- Real-time dashboard showing all active authority communications across Member States
- Communication volume metrics by authority, type, commodity, and time period
- Response time analytics with SLA compliance tracking
- Non-compliance trend analysis showing improvement or deterioration over time
- Inspection frequency analysis and preparedness scoring
- Appeal success rate tracking and lessons learned repository

#### Feature 11: Automated Response Templates

- Pre-built response templates for common information request types
- Template library per Member State and per language
- Variable substitution from upstream agent data
- Template versioning and approval workflows
- Compliance team template customization and sharing

---

### 6.3 Could-Have Features (P2 -- Nice to Have)

#### Feature 12: Predictive Communication Intelligence

- Predict likely authority communication based on operator risk profile and checking patterns
- Pre-assemble response packages for predicted information requests
- Identify operators likely to be targeted for enhanced Article 20 checks
- Suggest proactive communications to authorities to demonstrate compliance posture

#### Feature 13: Authority Relationship Management

- Track authority personnel assignments and preferences
- Record communication style and format preferences per authority officer
- Manage authority meeting scheduling and preparation
- Track authority satisfaction with operator responsiveness

---

### 6.4 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Direct integration with all 27 Member State authority portals (Phase 2 -- start with EU IS + email + manual channels)
- AI-powered legal argument generation for appeals (human legal review required for v1.0)
- Real-time video conferencing integration for authority meetings
- Automated fine negotiation
- Integration with external legal case management platforms (Clio, Thomson Reuters)
- Mobile native application for inspection coordination (web responsive only for v1.0)

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
| AGENT-EUDR-040        |           | AGENT-EUDR-036            |           | AGENT-EUDR-037        |
| Authority Comms       |<--------->| EU Information System      |<--------->| DDS Creator           |
| Manager               |           | Interface                 |           |                       |
|                       |           |                           |           |                       |
| - Info Request Handler|           | - DDS Submission Gateway  |           | - DDS Assembler       |
| - DDS Notification    |           | - Operator Registration   |           | - Geolocation Fmt     |
| - Inspection Coord.   |           | - Status Tracker          |           | - Risk Integrator     |
| - Non-Compliance Mgr  |           | - Audit Trail Recorder    |           | - Compliance Valid.   |
| - Appeal Manager      |           |                           |           |                       |
| - Doc Exchange Portal |           +---------------------------+           +-----------------------+
| - Multi-Language Eng. |
| - Notification Router |           +---------------------------+           +---------------------------+
| - Audit Trail Manager |           | AGENT-EUDR-038            |           | AGENT-EUDR-039            |
+-----------+-----------+           | Reference Number          |           | Customs Declaration       |
            |                       | Generator                 |           | Support                   |
            |                       +---------------------------+           +---------------------------+
            |
+-----------v-----------+           +---------------------------+           +---------------------------+
| Upstream Agents       |           | Security Layer             |           | Infrastructure Layer      |
| EUDR-001 to EUDR-035  |           |                           |           |                           |
|                       |           | - SEC-001 JWT Auth        |           | - PostgreSQL/TimescaleDB  |
| - Supply Chain (1-15) |           | - SEC-002 RBAC            |           | - Redis Cache             |
| - Risk Assess. (16-20)|           | - SEC-003 AES-256 Encrypt |           | - S3 Object Storage       |
| - Due Diligence (21-35)|          | - SEC-005 Audit Logging   |           | - Kubernetes (EKS)        |
|                       |           | - SEC-011 PII Detection   |           |                           |
+-----------------------+           +---------------------------+           +---------------------------+
```

```
Communication Flow:

Competent Authority (27 EU Member States)
    |
    | (EU IS API, Portal, Email, Postal)
    |
    v
+-------------------------------------------+
| AGENT-EUDR-040 Communication Channels     |
|                                           |
| +-------+ +-------+ +-------+ +-------+  |
| |EU IS  | |Portal | |Email  | |Manual |  |
| |Channel| |Channel| |Channel| |Entry  |  |
| +---+---+ +---+---+ +---+---+ +---+---+  |
|     |         |         |         |       |
|     +----+----+----+----+----+----+       |
|          |              |                 |
|     +----v----+    +----v----+            |
|     |Language | -->|Classify |            |
|     |Detect   |    |& Route  |            |
|     +---------+    +----+----+            |
|                         |                 |
|    +----------+---------+--------+        |
|    |          |         |        |        |
| +--v--+  +---v--+ +----v--+ +---v---+    |
| |Info | |DDS   | |Inspect| |Non-   |    |
| |Req. | |Notif.| |Coord. | |Comply |    |
| |Hndlr| |Eng.  | |Engine | |Resp.  |    |
| +--+--+ +--+---+ +---+---+ +---+---+    |
|    |        |         |         |        |
|    +--------+---------+---------+        |
|                   |                      |
|           +-------v--------+             |
|           |Audit Trail Mgr |             |
|           +----------------+             |
+-------------------------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/authority_communication_manager/
    __init__.py                              # Public API exports
    config.py                                # AuthorityCommunicationConfig with GL_EUDR_ACM_ env prefix
    models.py                                # Pydantic v2 models for communications, requests, responses, appeals
    info_request_handler.py                  # InformationRequestHandler: receive, classify, assemble, respond
    dds_notification_engine.py               # DDSNotificationEngine: DDS post-submission communication lifecycle
    inspection_coordinator.py                # InspectionCoordinationEngine: inspection lifecycle management
    non_compliance_manager.py                # NonComplianceResponseManager: findings, corrective actions, fines
    appeal_manager.py                        # AppealProcessManager: appeal lifecycle, evidence assembly, tracking
    document_exchange.py                     # DocumentExchangePortal: encrypted file sharing with authorities
    multi_language_engine.py                 # MultiLanguageCommunicationEngine: 24-language support
    notification_router.py                   # NotificationRoutingEngine: 27 Member State authority registry
    audit_trail_manager.py                   # CommunicationAuditTrailManager: immutable SHA-256 hash chain
    authority_registry.py                    # AuthorityRegistry: competent authority database management
    deadline_tracker.py                      # DeadlineTracker: communication deadline management with escalation
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains for all communications
    metrics.py                               # 15 Prometheus self-monitoring metrics
    setup.py                                 # AuthorityCommunicationService facade
    api/
        __init__.py
        router.py                            # FastAPI router (30+ endpoints)
        info_request_routes.py               # Information request CRUD and response endpoints
        dds_notification_routes.py           # DDS notification management endpoints
        inspection_routes.py                 # Inspection lifecycle endpoints
        non_compliance_routes.py             # Non-compliance management endpoints
        appeal_routes.py                     # Appeal lifecycle endpoints
        document_exchange_routes.py          # Secure document exchange endpoints
        language_routes.py                   # Language detection and translation endpoints
        routing_routes.py                    # Authority routing and registry endpoints
        audit_trail_routes.py                # Audit trail query and export endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Communication Types
class CommunicationType(str, Enum):
    INFORMATION_REQUEST = "information_request"
    INFORMATION_RESPONSE = "information_response"
    DDS_ACKNOWLEDGEMENT = "dds_acknowledgement"
    DDS_QUERY = "dds_query"
    DDS_CONDITIONAL_ACCEPTANCE = "dds_conditional_acceptance"
    DDS_REJECTION = "dds_rejection"
    DDS_AMENDMENT_REQUEST = "dds_amendment_request"
    INSPECTION_NOTIFICATION = "inspection_notification"
    INSPECTION_FINDING = "inspection_finding"
    NON_COMPLIANCE_WARNING = "non_compliance_warning"
    CORRECTIVE_ACTION_ORDER = "corrective_action_order"
    FINE_NOTIFICATION = "fine_notification"
    CONFISCATION_ORDER = "confiscation_order"
    MARKET_EXCLUSION_ORDER = "market_exclusion_order"
    PRE_DECISION_HEARING = "pre_decision_hearing"
    ADVERSE_DECISION = "adverse_decision"
    APPEAL_FILING = "appeal_filing"
    APPEAL_DECISION = "appeal_decision"
    INTER_AUTHORITY_REQUEST = "inter_authority_request"
    THIRD_COUNTRY_EXCHANGE = "third_country_exchange"
    DOCUMENT_SUBMISSION = "document_submission"
    GENERAL_INQUIRY = "general_inquiry"
    CORRECTIVE_ACTION_PLAN = "corrective_action_plan"
    REMEDIATION_EVIDENCE = "remediation_evidence"

# Communication Direction
class CommunicationDirection(str, Enum):
    INCOMING = "incoming"       # Authority -> Operator
    OUTGOING = "outgoing"       # Operator -> Authority

# Communication Urgency
class CommunicationUrgency(str, Enum):
    LOW = "low"                 # No specific deadline; general inquiry
    MEDIUM = "medium"           # Standard deadline (30-60 days)
    HIGH = "high"               # Short deadline (7-30 days)
    CRITICAL = "critical"       # Urgent deadline (< 7 days) or immediate action required

# Communication Status
class CommunicationStatus(str, Enum):
    RECEIVED = "received"
    ACKNOWLEDGED = "acknowledged"
    CLASSIFIED = "classified"
    IN_PROGRESS = "in_progress"
    RESPONSE_ASSEMBLED = "response_assembled"
    RESPONSE_REVIEWED = "response_reviewed"
    RESPONSE_SUBMITTED = "response_submitted"
    AUTHORITY_CONFIRMED = "authority_confirmed"
    CLOSED = "closed"
    ESCALATED = "escalated"
    OVERDUE = "overdue"

# Communication Channel
class CommunicationChannel(str, Enum):
    EU_INFORMATION_SYSTEM = "eu_information_system"
    AUTHORITY_PORTAL = "authority_portal"
    EMAIL = "email"
    CERTIFIED_EMAIL = "certified_email"     # PEC (Italy), De-Mail (Germany)
    REGISTERED_POST = "registered_post"
    DIGITAL_POST = "digital_post"           # e-Boks (Denmark), ePUAP (Poland)
    MANUAL_ENTRY = "manual_entry"           # For postal/fax communications

# Non-Compliance Severity
class NonComplianceSeverity(str, Enum):
    LEVEL_1_ADVISORY = "advisory"
    LEVEL_2_WARNING = "formal_warning"
    LEVEL_3_CORRECTIVE = "corrective_action_required"
    LEVEL_4_PENALTY = "financial_penalty"
    LEVEL_5_EXCLUSION = "confiscation_or_market_exclusion"

# Appeal Status
class AppealStatus(str, Enum):
    DECISION_RECEIVED = "decision_received"
    APPEAL_ASSESSED = "appeal_assessed"
    APPEAL_DRAFTED = "appeal_drafted"
    APPEAL_REVIEWED = "appeal_reviewed"
    APPEAL_FILED = "appeal_filed"
    ACKNOWLEDGEMENT_RECEIVED = "acknowledgement_received"
    HEARING_SCHEDULED = "hearing_scheduled"
    HEARING_COMPLETED = "hearing_completed"
    DECISION_PENDING = "decision_pending"
    APPEAL_GRANTED = "appeal_granted"
    APPEAL_PARTIALLY_GRANTED = "appeal_partially_granted"
    APPEAL_DENIED = "appeal_denied"
    JUDICIAL_REVIEW_INITIATED = "judicial_review_initiated"
    JUDICIAL_DECISION_RENDERED = "judicial_decision_rendered"
    CLOSED = "closed"

# Inspection Status
class InspectionStatus(str, Enum):
    NOTIFIED = "notified"
    PREPARATION_IN_PROGRESS = "preparation_in_progress"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    FINDINGS_RECEIVED = "findings_received"
    RESPONSE_IN_PROGRESS = "response_in_progress"
    RESPONSE_SUBMITTED = "response_submitted"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    CLOSED = "closed"

# EU Member State
class EUMemberState(str, Enum):
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

# Competent Authority
class CompetentAuthority(BaseModel):
    authority_id: str                       # Unique identifier
    member_state: EUMemberState             # EU Member State
    authority_name: str                     # Official authority name
    authority_name_en: str                  # English name for reference
    department: Optional[str]               # Specific department/division
    jurisdiction_commodities: List[str]     # Which commodities this authority handles
    jurisdiction_functions: List[str]       # Functions: inspection, enforcement, registration
    portal_url: Optional[str]              # Authority portal URL
    api_endpoint: Optional[str]            # API endpoint if available
    email: Optional[str]                   # Official email address
    certified_email: Optional[str]         # PEC/De-Mail/certified email
    postal_address: str                    # Official postal address
    phone: Optional[str]                   # Contact phone number
    language: str                          # Primary communication language
    secondary_languages: List[str]         # Additional accepted languages
    communication_protocol: CommunicationChannel  # Preferred communication method
    response_deadlines: Dict[str, int]     # Default deadlines by communication type (days)
    appeal_body: Optional[str]             # Administrative appeal authority
    appeal_deadline_days: int              # Days to file appeal
    registration_required: bool            # Whether operator registration is required
    active: bool                           # Whether authority is currently active
    last_verified: datetime                # Date of last registry verification
    metadata: Dict[str, Any]               # Additional authority-specific metadata

# Communication Record
class CommunicationRecord(BaseModel):
    communication_id: str                  # Unique identifier (UUID)
    operator_id: str                       # GreenLang operator ID
    authority_id: str                      # Competent authority reference
    member_state: EUMemberState            # Member State
    direction: CommunicationDirection      # Incoming or outgoing
    communication_type: CommunicationType  # Type of communication
    channel: CommunicationChannel          # Communication channel used
    urgency: CommunicationUrgency          # Urgency classification
    status: CommunicationStatus            # Current status
    subject: str                           # Communication subject
    content: str                           # Full communication content
    content_language: str                  # ISO 639-1 language code
    translated_content: Optional[str]      # Machine-translated content (if applicable)
    translation_language: Optional[str]    # Target translation language
    dds_reference_numbers: List[str]       # Associated DDS reference numbers
    commodity: Optional[str]               # Associated commodity
    regulatory_basis: List[str]            # EUDR articles cited
    response_deadline: Optional[datetime]  # Response deadline
    attachments: List[str]                 # Attachment file IDs
    parent_communication_id: Optional[str] # Parent communication (for threads)
    assigned_to: Optional[str]             # Assigned internal handler
    reviewed_by: Optional[str]             # Internal reviewer
    approved_by: Optional[str]             # Internal approver
    content_hash: str                      # SHA-256 hash of content
    provenance_hash: str                   # SHA-256 chain hash
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

# Inspection Record
class InspectionRecord(BaseModel):
    inspection_id: str                     # Unique identifier
    operator_id: str                       # GreenLang operator ID
    authority_id: str                      # Inspecting authority
    member_state: EUMemberState            # Member State
    inspection_type: str                   # "announced" or "unannounced"
    status: InspectionStatus               # Current status
    notification_date: datetime            # Date notification received
    scheduled_date: Optional[datetime]     # Scheduled inspection date
    actual_date: Optional[datetime]        # Actual inspection date
    facility_id: Optional[str]            # Facility being inspected
    scope_commodities: List[str]           # Commodities in scope
    scope_description: str                 # Detailed scope description
    inspector_names: List[str]             # Inspector names
    inspector_access_credentials: List[str]  # Temporary access IDs
    documents_requested: List[str]         # Documents requested during inspection
    documents_provided: List[str]          # Document IDs provided
    findings: List[Dict[str, Any]]         # Inspection findings
    corrective_actions: List[Dict[str, Any]]  # Post-inspection corrective actions
    response_deadline: Optional[datetime]  # Deadline for findings response
    closed_date: Optional[datetime]        # Date inspection closed
    communication_ids: List[str]           # All related communication IDs
    content_hash: str                      # SHA-256 hash
    created_at: datetime
    updated_at: datetime

# Non-Compliance Record
class NonComplianceRecord(BaseModel):
    non_compliance_id: str                 # Unique identifier
    operator_id: str                       # GreenLang operator ID
    authority_id: str                      # Issuing authority
    member_state: EUMemberState            # Member State
    severity: NonComplianceSeverity        # Severity level
    finding_reference: str                 # Authority finding reference number
    eudr_articles_violated: List[str]      # Specific EUDR articles violated
    finding_details: List[Dict[str, Any]]  # Detailed findings
    affected_products: List[str]           # Affected product/DDS IDs
    affected_commodities: List[str]        # Affected commodity categories
    fine_amount: Optional[Decimal]         # Fine amount if applicable
    fine_currency: str                     # Currency (default EUR)
    fine_paid: bool                        # Whether fine has been paid
    corrective_action_plan: Optional[Dict[str, Any]]  # CAP details
    corrective_action_deadline: Optional[datetime]
    remediation_status: str                # pending, in_progress, completed, verified
    remediation_evidence: List[str]        # Evidence document IDs
    appeal_filed: bool                     # Whether appeal has been filed
    appeal_id: Optional[str]              # Link to appeal record
    status: str                            # open, corrective_action, remediation, appealed, closed
    response_deadline: Optional[datetime]
    communication_ids: List[str]           # All related communication IDs
    content_hash: str                      # SHA-256 hash
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]

# Appeal Record
class AppealRecord(BaseModel):
    appeal_id: str                         # Unique identifier
    operator_id: str                       # GreenLang operator ID
    authority_id: str                      # Authority whose decision is appealed
    appeal_body_id: str                    # Appeal body (tribunal, court)
    member_state: EUMemberState            # Member State
    status: AppealStatus                   # Current appeal status
    adverse_decision_id: str               # Reference to the adverse decision
    adverse_decision_date: datetime        # Date of adverse decision
    adverse_decision_type: str             # Type: fine, confiscation, exclusion
    adverse_decision_amount: Optional[Decimal]  # Financial amount if applicable
    filing_deadline: datetime              # Calculated appeal filing deadline
    filing_date: Optional[datetime]        # Actual filing date
    grounds: List[str]                     # Grounds for appeal
    evidence_document_ids: List[str]       # Evidence package document IDs
    hearing_date: Optional[datetime]       # Hearing date if scheduled
    hearing_outcome: Optional[str]         # Hearing outcome notes
    appeal_decision: Optional[str]         # granted, partially_granted, denied
    appeal_decision_date: Optional[datetime]
    judicial_review_initiated: bool        # Whether second-level review initiated
    financial_impact: Optional[Dict[str, Decimal]]  # Fine amount, legal costs, opportunity cost
    communication_ids: List[str]           # All related communication IDs
    content_hash: str                      # SHA-256 hash
    created_at: datetime
    updated_at: datetime

# Audit Trail Entry
class AuditTrailEntry(BaseModel):
    entry_id: str                          # Unique identifier
    operator_id: str                       # GreenLang operator ID
    communication_id: Optional[str]        # Related communication ID
    inspection_id: Optional[str]           # Related inspection ID
    non_compliance_id: Optional[str]       # Related non-compliance ID
    appeal_id: Optional[str]              # Related appeal ID
    event_type: str                        # Event type classification
    event_description: str                 # Human-readable event description
    actor: str                             # Who performed the action
    actor_type: str                        # "operator", "authority", "system"
    authority_id: Optional[str]            # Authority involved
    member_state: Optional[EUMemberState]  # Member State
    dds_reference_numbers: List[str]       # Associated DDS references
    data_hash: str                         # SHA-256 hash of event data
    previous_entry_hash: str               # Hash of previous entry (chain)
    chain_hash: str                        # Combined chain hash
    event_timestamp: datetime              # Event timestamp (UTC)
    retention_expiry: datetime             # Retention period expiry date
    metadata: Dict[str, Any]
```

### 7.4 Database Schema (New Migration: V119)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_authority_communication;

-- Competent authority registry (27+ Member State authorities)
CREATE TABLE eudr_authority_communication.competent_authorities (
    authority_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    member_state CHAR(2) NOT NULL,
    authority_name VARCHAR(500) NOT NULL,
    authority_name_en VARCHAR(500) NOT NULL,
    department VARCHAR(300),
    jurisdiction_commodities JSONB DEFAULT '[]',
    jurisdiction_functions JSONB DEFAULT '[]',
    portal_url VARCHAR(1000),
    api_endpoint VARCHAR(1000),
    email VARCHAR(300),
    certified_email VARCHAR(300),
    postal_address TEXT NOT NULL,
    phone VARCHAR(50),
    language VARCHAR(5) NOT NULL,
    secondary_languages JSONB DEFAULT '[]',
    communication_protocol VARCHAR(50) DEFAULT 'email',
    response_deadlines JSONB DEFAULT '{}',
    appeal_body VARCHAR(500),
    appeal_deadline_days INTEGER DEFAULT 30,
    registration_required BOOLEAN DEFAULT TRUE,
    active BOOLEAN DEFAULT TRUE,
    last_verified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_authorities_member_state ON eudr_authority_communication.competent_authorities(member_state);
CREATE INDEX idx_authorities_active ON eudr_authority_communication.competent_authorities(active);

-- Communication records (all authority interactions)
CREATE TABLE eudr_authority_communication.communications (
    communication_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    authority_id UUID REFERENCES eudr_authority_communication.competent_authorities(authority_id),
    member_state CHAR(2) NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('incoming', 'outgoing')),
    communication_type VARCHAR(50) NOT NULL,
    channel VARCHAR(30) NOT NULL,
    urgency VARCHAR(10) NOT NULL DEFAULT 'medium',
    status VARCHAR(30) NOT NULL DEFAULT 'received',
    subject VARCHAR(1000) NOT NULL,
    content TEXT NOT NULL,
    content_language VARCHAR(5) NOT NULL DEFAULT 'en',
    translated_content TEXT,
    translation_language VARCHAR(5),
    dds_reference_numbers JSONB DEFAULT '[]',
    commodity VARCHAR(50),
    regulatory_basis JSONB DEFAULT '[]',
    response_deadline TIMESTAMPTZ,
    attachments JSONB DEFAULT '[]',
    parent_communication_id UUID REFERENCES eudr_authority_communication.communications(communication_id),
    assigned_to VARCHAR(200),
    reviewed_by VARCHAR(200),
    approved_by VARCHAR(200),
    content_hash VARCHAR(64) NOT NULL,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_comms_operator ON eudr_authority_communication.communications(operator_id);
CREATE INDEX idx_comms_authority ON eudr_authority_communication.communications(authority_id);
CREATE INDEX idx_comms_member_state ON eudr_authority_communication.communications(member_state);
CREATE INDEX idx_comms_direction ON eudr_authority_communication.communications(direction);
CREATE INDEX idx_comms_type ON eudr_authority_communication.communications(communication_type);
CREATE INDEX idx_comms_status ON eudr_authority_communication.communications(status);
CREATE INDEX idx_comms_urgency ON eudr_authority_communication.communications(urgency);
CREATE INDEX idx_comms_deadline ON eudr_authority_communication.communications(response_deadline);
CREATE INDEX idx_comms_parent ON eudr_authority_communication.communications(parent_communication_id);
CREATE INDEX idx_comms_dds_refs ON eudr_authority_communication.communications USING GIN (dds_reference_numbers);

-- Inspection records
CREATE TABLE eudr_authority_communication.inspections (
    inspection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    authority_id UUID REFERENCES eudr_authority_communication.competent_authorities(authority_id),
    member_state CHAR(2) NOT NULL,
    inspection_type VARCHAR(20) NOT NULL CHECK (inspection_type IN ('announced', 'unannounced')),
    status VARCHAR(30) NOT NULL DEFAULT 'notified',
    notification_date TIMESTAMPTZ NOT NULL,
    scheduled_date TIMESTAMPTZ,
    actual_date TIMESTAMPTZ,
    facility_id VARCHAR(100),
    scope_commodities JSONB DEFAULT '[]',
    scope_description TEXT NOT NULL,
    inspector_names JSONB DEFAULT '[]',
    inspector_access_credentials JSONB DEFAULT '[]',
    documents_requested JSONB DEFAULT '[]',
    documents_provided JSONB DEFAULT '[]',
    findings JSONB DEFAULT '[]',
    corrective_actions JSONB DEFAULT '[]',
    response_deadline TIMESTAMPTZ,
    closed_date TIMESTAMPTZ,
    communication_ids JSONB DEFAULT '[]',
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inspections_operator ON eudr_authority_communication.inspections(operator_id);
CREATE INDEX idx_inspections_authority ON eudr_authority_communication.inspections(authority_id);
CREATE INDEX idx_inspections_status ON eudr_authority_communication.inspections(status);
CREATE INDEX idx_inspections_scheduled ON eudr_authority_communication.inspections(scheduled_date);

-- Non-compliance records
CREATE TABLE eudr_authority_communication.non_compliance_records (
    non_compliance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    authority_id UUID REFERENCES eudr_authority_communication.competent_authorities(authority_id),
    member_state CHAR(2) NOT NULL,
    severity VARCHAR(40) NOT NULL,
    finding_reference VARCHAR(200) NOT NULL,
    eudr_articles_violated JSONB DEFAULT '[]',
    finding_details JSONB DEFAULT '[]',
    affected_products JSONB DEFAULT '[]',
    affected_commodities JSONB DEFAULT '[]',
    fine_amount NUMERIC(18,2),
    fine_currency VARCHAR(3) DEFAULT 'EUR',
    fine_paid BOOLEAN DEFAULT FALSE,
    corrective_action_plan JSONB,
    corrective_action_deadline TIMESTAMPTZ,
    remediation_status VARCHAR(30) DEFAULT 'pending',
    remediation_evidence JSONB DEFAULT '[]',
    appeal_filed BOOLEAN DEFAULT FALSE,
    appeal_id UUID,
    status VARCHAR(30) NOT NULL DEFAULT 'open',
    response_deadline TIMESTAMPTZ,
    communication_ids JSONB DEFAULT '[]',
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

CREATE INDEX idx_noncompliance_operator ON eudr_authority_communication.non_compliance_records(operator_id);
CREATE INDEX idx_noncompliance_authority ON eudr_authority_communication.non_compliance_records(authority_id);
CREATE INDEX idx_noncompliance_severity ON eudr_authority_communication.non_compliance_records(severity);
CREATE INDEX idx_noncompliance_status ON eudr_authority_communication.non_compliance_records(status);
CREATE INDEX idx_noncompliance_deadline ON eudr_authority_communication.non_compliance_records(response_deadline);

-- Appeal records
CREATE TABLE eudr_authority_communication.appeals (
    appeal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    authority_id UUID REFERENCES eudr_authority_communication.competent_authorities(authority_id),
    appeal_body_id VARCHAR(200),
    member_state CHAR(2) NOT NULL,
    status VARCHAR(40) NOT NULL DEFAULT 'decision_received',
    adverse_decision_id VARCHAR(200) NOT NULL,
    adverse_decision_date TIMESTAMPTZ NOT NULL,
    adverse_decision_type VARCHAR(50) NOT NULL,
    adverse_decision_amount NUMERIC(18,2),
    filing_deadline TIMESTAMPTZ NOT NULL,
    filing_date TIMESTAMPTZ,
    grounds JSONB DEFAULT '[]',
    evidence_document_ids JSONB DEFAULT '[]',
    hearing_date TIMESTAMPTZ,
    hearing_outcome TEXT,
    appeal_decision VARCHAR(30),
    appeal_decision_date TIMESTAMPTZ,
    judicial_review_initiated BOOLEAN DEFAULT FALSE,
    financial_impact JSONB DEFAULT '{}',
    communication_ids JSONB DEFAULT '[]',
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_appeals_operator ON eudr_authority_communication.appeals(operator_id);
CREATE INDEX idx_appeals_authority ON eudr_authority_communication.appeals(authority_id);
CREATE INDEX idx_appeals_status ON eudr_authority_communication.appeals(status);
CREATE INDEX idx_appeals_filing_deadline ON eudr_authority_communication.appeals(filing_deadline);

-- Communication audit trail (hypertable for time-series)
CREATE TABLE eudr_authority_communication.audit_trail (
    entry_id UUID DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    communication_id UUID,
    inspection_id UUID,
    non_compliance_id UUID,
    appeal_id UUID,
    event_type VARCHAR(100) NOT NULL,
    event_description TEXT NOT NULL,
    actor VARCHAR(200) NOT NULL,
    actor_type VARCHAR(20) NOT NULL CHECK (actor_type IN ('operator', 'authority', 'system')),
    authority_id UUID,
    member_state CHAR(2),
    dds_reference_numbers JSONB DEFAULT '[]',
    data_hash VARCHAR(64) NOT NULL,
    previous_entry_hash VARCHAR(64) NOT NULL,
    chain_hash VARCHAR(64) NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    retention_expiry TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('eudr_authority_communication.audit_trail', 'event_timestamp');

CREATE INDEX idx_audit_operator ON eudr_authority_communication.audit_trail(operator_id, event_timestamp DESC);
CREATE INDEX idx_audit_communication ON eudr_authority_communication.audit_trail(communication_id);
CREATE INDEX idx_audit_inspection ON eudr_authority_communication.audit_trail(inspection_id);
CREATE INDEX idx_audit_noncompliance ON eudr_authority_communication.audit_trail(non_compliance_id);
CREATE INDEX idx_audit_appeal ON eudr_authority_communication.audit_trail(appeal_id);
CREATE INDEX idx_audit_event_type ON eudr_authority_communication.audit_trail(event_type);
CREATE INDEX idx_audit_retention ON eudr_authority_communication.audit_trail(retention_expiry);

-- Document exchange records
CREATE TABLE eudr_authority_communication.document_exchanges (
    exchange_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    authority_id UUID REFERENCES eudr_authority_communication.competent_authorities(authority_id),
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('outgoing', 'incoming')),
    document_id VARCHAR(200) NOT NULL,
    document_name VARCHAR(500) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    document_hash VARCHAR(64) NOT NULL,
    encryption_key_id VARCHAR(200),
    download_link VARCHAR(2000),
    link_expiry TIMESTAMPTZ,
    watermark_applied BOOLEAN DEFAULT FALSE,
    access_control_list JSONB DEFAULT '[]',
    download_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    virus_scan_status VARCHAR(20) DEFAULT 'pending',
    communication_id UUID REFERENCES eudr_authority_communication.communications(communication_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_docexchange_operator ON eudr_authority_communication.document_exchanges(operator_id);
CREATE INDEX idx_docexchange_authority ON eudr_authority_communication.document_exchanges(authority_id);
CREATE INDEX idx_docexchange_communication ON eudr_authority_communication.document_exchanges(communication_id);

-- Deadline tracking
CREATE TABLE eudr_authority_communication.deadlines (
    deadline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    communication_id UUID REFERENCES eudr_authority_communication.communications(communication_id),
    inspection_id UUID REFERENCES eudr_authority_communication.inspections(inspection_id),
    non_compliance_id UUID REFERENCES eudr_authority_communication.non_compliance_records(non_compliance_id),
    appeal_id UUID REFERENCES eudr_authority_communication.appeals(appeal_id),
    deadline_type VARCHAR(50) NOT NULL,
    deadline_date TIMESTAMPTZ NOT NULL,
    reminder_sent_7d BOOLEAN DEFAULT FALSE,
    reminder_sent_3d BOOLEAN DEFAULT FALSE,
    reminder_sent_1d BOOLEAN DEFAULT FALSE,
    reminder_sent_same_day BOOLEAN DEFAULT FALSE,
    escalated BOOLEAN DEFAULT FALSE,
    escalation_date TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    met BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_deadlines_operator ON eudr_authority_communication.deadlines(operator_id);
CREATE INDEX idx_deadlines_date ON eudr_authority_communication.deadlines(deadline_date);
CREATE INDEX idx_deadlines_status ON eudr_authority_communication.deadlines(status);

-- Translation cache
CREATE TABLE eudr_authority_communication.translations (
    translation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_text_hash VARCHAR(64) NOT NULL,
    source_language VARCHAR(5) NOT NULL,
    target_language VARCHAR(5) NOT NULL,
    translated_text TEXT NOT NULL,
    translation_method VARCHAR(20) NOT NULL DEFAULT 'machine',
    confidence_score NUMERIC(3,2),
    reviewed_by VARCHAR(200),
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_translations_lookup ON eudr_authority_communication.translations(source_text_hash, source_language, target_language);

-- EUDR terminology glossary
CREATE TABLE eudr_authority_communication.terminology (
    term_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    term_key VARCHAR(200) NOT NULL,
    language VARCHAR(5) NOT NULL,
    term_text VARCHAR(500) NOT NULL,
    definition TEXT,
    eudr_article_reference VARCHAR(50),
    category VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_terminology_lookup ON eudr_authority_communication.terminology(term_key, language);
CREATE INDEX idx_terminology_category ON eudr_authority_communication.terminology(category);
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Information Requests** | | |
| POST | `/v1/communications/info-requests` | Register an incoming information request from a competent authority |
| GET | `/v1/communications/info-requests` | List information requests (with filters: authority, status, deadline, urgency) |
| GET | `/v1/communications/info-requests/{request_id}` | Get information request details with response timeline |
| POST | `/v1/communications/info-requests/{request_id}/assemble` | Trigger automated response assembly from upstream agents |
| POST | `/v1/communications/info-requests/{request_id}/respond` | Submit assembled response to the requesting authority |
| **DDS Notifications** | | |
| GET | `/v1/communications/dds-notifications` | List DDS-related authority notifications (with filters) |
| GET | `/v1/communications/dds-notifications/{notification_id}` | Get DDS notification details |
| POST | `/v1/communications/dds-notifications/{notification_id}/respond` | Submit response to DDS query or amendment request |
| **Inspections** | | |
| POST | `/v1/inspections` | Register an incoming inspection notification |
| GET | `/v1/inspections` | List inspections (with filters: status, authority, date range) |
| GET | `/v1/inspections/{inspection_id}` | Get inspection details with preparation status |
| POST | `/v1/inspections/{inspection_id}/prepare` | Trigger inspection readiness package assembly |
| PUT | `/v1/inspections/{inspection_id}/findings` | Record inspection findings |
| POST | `/v1/inspections/{inspection_id}/respond` | Submit findings response to authority |
| POST | `/v1/inspections/{inspection_id}/access` | Generate inspector temporary access credentials |
| **Non-Compliance** | | |
| POST | `/v1/non-compliance` | Register an incoming non-compliance notification |
| GET | `/v1/non-compliance` | List non-compliance records (with filters: severity, status, authority) |
| GET | `/v1/non-compliance/{nc_id}` | Get non-compliance record with corrective action status |
| POST | `/v1/non-compliance/{nc_id}/corrective-action` | Submit corrective action plan |
| POST | `/v1/non-compliance/{nc_id}/evidence` | Submit remediation evidence |
| POST | `/v1/non-compliance/{nc_id}/respond` | Submit formal response to authority |
| **Appeals** | | |
| POST | `/v1/appeals` | Initiate an appeal for an adverse decision |
| GET | `/v1/appeals` | List appeals (with filters: status, authority, type) |
| GET | `/v1/appeals/{appeal_id}` | Get appeal details with timeline and evidence |
| POST | `/v1/appeals/{appeal_id}/evidence` | Assemble evidence package from upstream agents |
| POST | `/v1/appeals/{appeal_id}/file` | Submit appeal filing to the appeal body |
| PUT | `/v1/appeals/{appeal_id}/status` | Update appeal status (hearing, decision) |
| **Document Exchange** | | |
| POST | `/v1/documents/share` | Share documents with a competent authority |
| GET | `/v1/documents/shared` | List shared documents (with filters: authority, date range) |
| GET | `/v1/documents/shared/{exchange_id}` | Get document exchange details with access log |
| POST | `/v1/documents/receive` | Receive documents from a competent authority |
| **Authority Registry** | | |
| GET | `/v1/authorities` | List all registered competent authorities |
| GET | `/v1/authorities/{authority_id}` | Get authority details with contact information |
| GET | `/v1/authorities/route` | Determine correct authority for a given Member State/commodity/type |
| **Communications (General)** | | |
| GET | `/v1/communications` | List all communications (universal inbox with filters) |
| GET | `/v1/communications/{communication_id}` | Get communication details |
| GET | `/v1/communications/{communication_id}/thread` | Get full communication thread |
| **Audit Trail** | | |
| GET | `/v1/audit-trail` | Query audit trail (with filters: date range, type, authority, DDS ref) |
| GET | `/v1/audit-trail/export` | Export audit trail for authority inspection |
| GET | `/v1/audit-trail/integrity` | Verify audit trail hash chain integrity |
| **Deadlines** | | |
| GET | `/v1/deadlines` | List all active deadlines with countdown |
| GET | `/v1/deadlines/overdue` | List overdue deadlines requiring immediate attention |
| **Language** | | |
| POST | `/v1/language/detect` | Detect language of text content |
| POST | `/v1/language/translate` | Translate text between EU languages |
| GET | `/v1/language/glossary/{language}` | Get EUDR terminology glossary for a language |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_acm_communications_received_total` | Counter | Communications received by type, direction, authority, urgency |
| 2 | `gl_eudr_acm_communications_responded_total` | Counter | Communications responded to by type and authority |
| 3 | `gl_eudr_acm_response_assembly_duration_seconds` | Histogram | Time to assemble response packages from upstream agents |
| 4 | `gl_eudr_acm_inspections_total` | Counter | Inspections by type (announced/unannounced) and authority |
| 5 | `gl_eudr_acm_non_compliance_total` | Counter | Non-compliance records by severity and authority |
| 6 | `gl_eudr_acm_appeals_total` | Counter | Appeals by status and authority |
| 7 | `gl_eudr_acm_deadlines_met_total` | Counter | Deadlines met on time |
| 8 | `gl_eudr_acm_deadlines_missed_total` | Counter | Deadlines missed (critical -- should always be zero) |
| 9 | `gl_eudr_acm_documents_exchanged_total` | Counter | Documents exchanged by direction and authority |
| 10 | `gl_eudr_acm_translations_total` | Counter | Translations performed by language pair |
| 11 | `gl_eudr_acm_language_detection_accuracy` | Gauge | Rolling average language detection accuracy |
| 12 | `gl_eudr_acm_processing_duration_seconds` | Histogram | Communication processing latency by operation type |
| 13 | `gl_eudr_acm_errors_total` | Counter | Errors by operation type and severity |
| 14 | `gl_eudr_acm_active_communications` | Gauge | Currently active (open) communications across all types |
| 15 | `gl_eudr_acm_audit_trail_integrity` | Gauge | Audit trail hash chain integrity status (1 = valid, 0 = broken) |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for audit trail |
| Cache | Redis | Communication state caching, deadline countdown, authority registry cache |
| Object Storage | S3 | Document exchange storage, audit trail archival, evidence packages |
| Email Integration | aiosmtplib + aioimaplib | Async email send/receive for authority email channel |
| NLP | fastText + langdetect | Language detection for incoming communications |
| Translation | DeepL API / EU eTranslation | Machine translation for 24 EU languages (GDPR-compliant) |
| Encryption | AES-256-GCM via SEC-003 | Document encryption at rest |
| TLS | TLS 1.3 via SEC-004 | Document encryption in transit |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access control with enhanced restrictions for non-compliance and appeal data |
| PII Handling | SEC-011 PII Detection/Redaction | PII handling in authority communications |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-acm:communications:read` | View all authority communications | Viewer, Analyst, Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:communications:write` | Create and respond to communications | Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:communications:approve` | Approve outgoing communications before submission | Compliance Director, Legal Counsel, Admin |
| `eudr-acm:info-requests:read` | View information requests | Viewer, Analyst, Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:info-requests:respond` | Assemble and submit information request responses | Compliance Officer, Admin |
| `eudr-acm:dds-notifications:read` | View DDS-related authority notifications | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-acm:dds-notifications:respond` | Respond to DDS queries and amendment requests | Compliance Officer, Admin |
| `eudr-acm:inspections:read` | View inspection records | Viewer, Analyst, Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:inspections:manage` | Manage inspection lifecycle (prepare, respond, close) | Compliance Officer, Admin |
| `eudr-acm:inspections:access` | Generate inspector temporary access credentials | Compliance Director, Admin |
| `eudr-acm:non-compliance:read` | View non-compliance records (restricted) | Compliance Officer, Legal Counsel, Compliance Director, Admin |
| `eudr-acm:non-compliance:manage` | Manage non-compliance responses and corrective actions | Legal Counsel, Compliance Director, Admin |
| `eudr-acm:appeals:read` | View appeal records (restricted) | Legal Counsel, Compliance Director, Admin |
| `eudr-acm:appeals:manage` | Manage appeal lifecycle (file, evidence, track) | Legal Counsel, Admin |
| `eudr-acm:documents:read` | View document exchange records | Viewer, Analyst, Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:documents:share` | Share documents with authorities | Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:documents:receive` | Receive and process documents from authorities | Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:authorities:read` | View competent authority registry | Viewer, Analyst, Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:authorities:manage` | Manage authority registry (add, update, verify) | Admin |
| `eudr-acm:audit-trail:read` | View communication audit trail | Compliance Officer, Legal Counsel, Compliance Director, Auditor, Admin |
| `eudr-acm:audit-trail:export` | Export audit trail for authority inspection | Compliance Director, Admin |
| `eudr-acm:language:translate` | Use translation services | Analyst, Compliance Officer, Legal Counsel, Admin |
| `eudr-acm:deadlines:read` | View active deadlines | Viewer, Analyst, Compliance Officer, Legal Counsel, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources for Response Assembly)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping | Supply chain graph data | Supply chain maps, node/edge data for information request responses and inspection packages |
| AGENT-EUDR-002 Geolocation Verification | Geolocation evidence | Plot coordinates, verification results for authority data requests |
| AGENT-EUDR-003/004/005 Satellite/Forest/Land Use | Environmental evidence | Satellite imagery, forest cover analysis, land use change data for inspection and appeal evidence |
| AGENT-EUDR-006/007 Plot Boundary/GPS Validator | Geospatial data | Plot boundaries, GPS validation for geolocation-related authority queries |
| AGENT-EUDR-008-015 Supply Chain Traceability | Chain of custody data | Multi-tier supplier data, segregation, mass balance, certificates for information requests |
| AGENT-EUDR-016-020 Risk Assessment | Risk scores and evidence | Country risk, supplier risk, commodity risk, deforestation alerts for risk-related authority queries |
| AGENT-EUDR-021-025 Due Diligence (Rights/Legal/Audit) | Compliance evidence | Indigenous rights checks, protected area validation, legal compliance, audit results, risk mitigation |
| AGENT-EUDR-026 Due Diligence Orchestrator | Orchestration status | Overall due diligence status and completion for authority reporting |
| AGENT-EUDR-029 Mitigation Measure Designer | Mitigation plans | Mitigation measures and effectiveness data for corrective action plan generation |
| AGENT-EUDR-030 Documentation Generator | Document packages | Complete compliance packages, evidence assemblies for authority submission |
| AGENT-EUDR-035 Improvement Plan Creator | Improvement plans | Structured improvement plans for corrective action responses |
| AGENT-EUDR-036 EU Information System Interface | EU IS notifications | DDS submission status, authority notifications via EU IS channel |
| AGENT-EUDR-037 Due Diligence Statement Creator | DDS documents | Complete DDS documents for authority reference and amendment coordination |
| AGENT-EUDR-038 Reference Number Generator | DDS reference numbers | Reference number resolution for DDS-linked communications |
| AGENT-EUDR-039 Customs Declaration Support | Customs data | Customs declaration data for goods confiscation/release coordination |
| AGENT-FOUND-005 Citations & Evidence | Regulatory citations | EUDR article citations for legal argument assembly in appeals |
| AGENT-FOUND-008 Reproducibility | Integrity verification | Hash chain verification for audit trail integrity |
| SEC-005 Centralized Audit Logging | Platform audit integration | Integration with platform-wide audit logging |
| SEC-011 PII Detection/Redaction | PII handling | PII detection and redaction in authority communications |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | Communication dashboards, deadline tracking, authority interaction views |
| EUDR-036 EU Information System Interface | Bidirectional | Send responses via EU IS channel; receive EU IS notifications |
| EUDR-037 DDS Creator | Amendment coordination | Trigger DDS amendments based on authority feedback |
| Management Dashboard | Reporting API | Non-compliance exposure, appeal status, inspection frequency, communication metrics |
| External Legal Counsel | Export API | Appeal documentation packages, evidence exports for legal review |
| Board Reporting | Summary API | Regulatory risk exposure, fine exposure, compliance posture summaries |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Responding to an Information Request (Compliance Director)

```
1. Competent authority in Germany (BLE) sends information request via EU IS
   regarding cocoa supply chain for DDS-2026-DE-00142
2. EUDR-040 receives notification from EUDR-036, classifies as:
   Type: information_request, Urgency: medium, Deadline: 30 days
3. System detects language (German), provides English translation for internal review
4. System parses request scope: supply chain map, geolocation data for all
   production plots, risk assessment results, and certificate copies
5. Compliance Director reviews in unified inbox, clicks "Assemble Response"
6. System pulls data from upstream agents:
   - Supply chain graph from EUDR-001
   - Geolocation data from EUDR-002/007
   - Risk assessment from EUDR-016-025
   - Certificates from EUDR-012
7. Response package assembled in 25 seconds, compliance score: 98%
8. Director reviews package, adds a cover letter, assigns for legal review
9. Legal Counsel reviews and approves
10. Director clicks "Submit Response" -- system formats in German,
    transmits via EU IS through EUDR-036
11. System records response in audit trail with SHA-256 hash
12. Status updated to "response_submitted"; awaiting authority confirmation
```

#### Flow 2: Managing an Inspection (Inspection Coordinator)

```
1. Netherlands authority (NVWA) sends announced inspection notification:
   Date: March 25, 2026; Scope: palm oil chain of custody, risk assessment
2. EUDR-040 receives, classifies, calculates preparation deadline (10 days)
3. System generates preparation checklist:
   - Palm oil supply chain map (EUDR-001)
   - Geolocation evidence for all Malaysian/Indonesian plots (EUDR-002/006/007)
   - Satellite imagery and forest cover analysis (EUDR-003/004)
   - Chain of custody records (EUDR-009)
   - Mass balance calculations (EUDR-011)
   - Certificates (EUDR-012)
   - Risk assessments (EUDR-016-020)
   - DDS documents (EUDR-037)
4. Inspection Coordinator reviews checklist, clicks "Assemble Package"
5. System assembles complete inspection readiness package in 3 hours
6. On inspection day: system generates read-only inspector access credentials
7. During inspection: inspector requests additional document (specific contract)
   System locates and delivers via secure portal in 4 minutes
8. Post-inspection: findings received -- 2 medium findings
9. System maps findings to relevant agents, generates corrective action plan
10. Coordinator submits response within 25-day deadline
11. Complete inspection record archived in audit trail
```

#### Flow 3: Filing an Appeal (Legal Counsel)

```
1. French authority (DGCCRF) issues adverse decision:
   Fine of EUR 150,000 for inadequate risk assessment on soya imports
2. EUDR-040 receives, classifies as adverse_decision, Severity: Level 4
3. System calculates appeal deadline: 30 days (French administrative law)
4. Legal Counsel reviews, assesses grounds for appeal:
   - Platform data shows comprehensive risk assessment was performed (EUDR-028)
   - Mitigation measures were designed and implemented (EUDR-029)
   - DDS was submitted with complete risk documentation (EUDR-037)
5. Counsel clicks "Prepare Appeal" -- system assembles evidence:
   - Complete risk assessment output from EUDR-028
   - Mitigation measure documentation from EUDR-029
   - DDS submission records from EUDR-036
   - Communication history proving timely compliance
6. System formats appeal in French using EUDR terminology glossary
7. Counsel reviews, edits, and approves the translated appeal
8. Appeal filed via French Teleservice portal within deadline
9. System tracks appeal status: filed -> acknowledged -> hearing scheduled
10. System prepares hearing evidence package
11. Appeal decision: fine reduced to EUR 50,000 (partial success)
12. System records outcome, updates non-compliance record, tracks payment
```

### 8.2 Key Screen Descriptions

**Authority Communication Inbox:**
- Universal inbox showing all active communications across all Member States
- Left sidebar: filter panel (Member State, authority, type, urgency, status, deadline)
- Main panel: communication list sorted by deadline urgency (overdue first, then soonest deadline)
- Right sidebar: communication detail panel (appears on selection) with thread view, attachments, and response actions
- Top bar: summary statistics (total active, overdue, due this week, by urgency)
- Color coding: red = overdue, orange = due within 3 days, yellow = due within 7 days, green = on track

**Inspection Dashboard:**
- Active inspections panel with status cards (upcoming, in_progress, findings_response_due)
- Inspection calendar view showing scheduled inspections by facility
- Preparation checklist with progress tracking per inspection
- Inspector access management panel
- Post-inspection findings tracker with corrective action progress

**Non-Compliance Dashboard:**
- Exposure summary: total open findings, total pending fines, total affected products
- Severity distribution chart (Level 1 through Level 5)
- Timeline view showing all non-compliance events with deadlines
- Corrective action progress tracker with evidence collection status
- Trend analysis: non-compliance findings over time by authority and category

**Appeal Management Panel:**
- Active appeals list with status, deadline, and financial impact
- Appeal timeline visualization showing stages from decision through resolution
- Evidence package builder with upstream agent data selection
- Deadline countdown with escalation status
- Appeal success tracking and lessons learned repository

**Authority Registry Browser:**
- Interactive map of EU showing all 27 Member States with competent authority pins
- Click Member State to see authority details, contact information, communication protocols
- Authority relationship view showing all interactions with a specific authority
- Communication volume metrics per authority

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Information Request Handler -- receive, classify, assemble, respond to authority information requests
  - [ ] Feature 2: DDS Submission Notification Engine -- manage post-submission DDS authority communications
  - [ ] Feature 3: Inspection Coordination Engine -- complete inspection lifecycle management
  - [ ] Feature 4: Non-Compliance Response Manager -- structured non-compliance workflow with corrective actions
  - [ ] Feature 5: Appeal Process Manager -- appeal lifecycle with deadline tracking and evidence assembly
  - [ ] Feature 6: Document Exchange Portal -- encrypted, auditable document sharing with authorities
  - [ ] Feature 7: Multi-Language Communication Engine -- 24 EU language detection and translation support
  - [ ] Feature 8: Notification Routing Engine -- 27 Member State authority registry and routing
  - [ ] Feature 9: Communication Audit Trail Manager -- immutable 5-year audit trail with SHA-256 hash chains
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC + encryption integrated, PII handling verified)
- [ ] Performance targets met (< 30 seconds response assembly, < 500ms processing latency)
- [ ] Authority registry populated and verified for all 27 EU Member States
- [ ] Multi-language support validated for all 24 EU official languages (language detection >= 97% accuracy)
- [ ] Audit trail integrity verified (SHA-256 hash chain validation passing)
- [ ] GDPR compliance validated (PII handling, data retention, erasure procedures)
- [ ] API documentation complete (OpenAPI spec for all 30+ endpoints)
- [ ] Database migration V119 tested and validated
- [ ] Integration with EUDR-036, EUDR-037, EUDR-038, EUDR-039 verified
- [ ] Integration with at least 10 upstream agents (EUDR-001, 002, 007, 016-020, 028, 029, 030) verified for response assembly
- [ ] 5 beta customers successfully processing authority communications through the platform
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ operators processing authority communications through the platform
- 100+ communications processed across all types
- Zero missed deadlines for active customers
- Average response assembly time < 30 seconds
- Language detection accuracy >= 97%
- Audit trail integrity at 100%
- < 5 support tickets per customer

**60 Days:**
- 200+ operators active on the platform
- 500+ communications processed
- Zero missed deadlines maintained
- Authority registry verified and updated for any changes
- 10+ inspections coordinated through the platform
- 5+ non-compliance responses managed through the platform
- Response completeness rate >= 98%

**90 Days:**
- 500+ operators active on the platform
- 2,000+ communications processed
- Zero missed deadlines maintained
- First successful appeal managed through the platform
- Non-compliance escalation rate reduced by 50% for active customers
- NPS > 50 from compliance director and legal counsel personas
- 99.9% platform uptime achieved

---

## 10. Timeline and Milestones

### Phase 1: Core Communication Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Authority Registry and Notification Routing (Feature 8): 27 Member State authority database, routing rules, channel selection | Backend Engineer |
| 2-3 | Communication Audit Trail (Feature 9): immutable audit trail, SHA-256 hash chains, retention management | Backend + Security Engineer |
| 3-4 | Information Request Handler (Feature 1): receive, classify, upstream agent data assembly, response formatting | Senior Backend Engineer |
| 4-5 | DDS Notification Engine (Feature 2): post-submission communication lifecycle, EUDR-036/037/038 integration | Backend Engineer |
| 5-6 | Deadline Tracker: deadline calculation, reminder engine, escalation workflows | Backend Engineer |

**Milestone: Core communication engine operational with information request handling and audit trail (Week 6)**

### Phase 2: Enforcement Response Engines (Weeks 7-12)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Inspection Coordination Engine (Feature 3): inspection lifecycle, readiness package assembly, inspector access | Senior Backend Engineer |
| 8-10 | Non-Compliance Response Manager (Feature 4): findings parsing, corrective action generation, remediation tracking, fine management | Senior Backend + Legal Workflow |
| 10-12 | Appeal Process Manager (Feature 5): appeal lifecycle, deadline calculation per Member State, evidence assembly, hearing preparation | Senior Backend + Legal Workflow |

**Milestone: Full enforcement response capability operational (Week 12)**

### Phase 3: Infrastructure and Integration (Weeks 13-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 13-14 | Document Exchange Portal (Feature 6): encrypted upload/download, ACLs, watermarking, virus scanning | Backend + Security Engineer |
| 14-16 | Multi-Language Communication Engine (Feature 7): language detection, translation integration, EUDR glossary in 24 languages | NLP Engineer + Backend Engineer |
| 16-17 | REST API Layer: 30+ endpoints, authentication, rate limiting, OpenAPI documentation | Backend Engineer |
| 17-18 | Frontend integration: communication inbox, inspection dashboard, non-compliance dashboard, appeal panel, authority registry | Frontend Engineer |

**Milestone: All 9 P0 features implemented with full UI (Week 18)**

### Phase 4: Testing and Launch (Weeks 19-22)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 19-20 | Complete test suite: 900+ tests covering all features, edge cases, and integrations | Test Engineer |
| 20-21 | Performance testing, security audit, GDPR compliance review, load testing | DevOps + Security + Legal |
| 21 | Database migration V119 finalized and tested | DevOps |
| 21-22 | Beta customer onboarding (5 customers across at least 3 Member States) | Product + Engineering |
| 22 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 22)**

### Phase 5: Enhancements (Weeks 23-30)

- Communication analytics dashboard (Feature 10)
- Automated response templates (Feature 11)
- Predictive communication intelligence (Feature 12)
- Authority relationship management (Feature 13)
- Additional Member State portal integrations
- Enhanced appeal analytics and success prediction

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-001 through EUDR-015 (Supply Chain Traceability) | BUILT (100%) | Low | Stable, production-ready; standard API integration |
| AGENT-EUDR-016 through EUDR-020 (Risk Assessment) | BUILT (100%) | Low | Stable, production-ready |
| AGENT-EUDR-021 through EUDR-035 (Due Diligence) | BUILT (100%) | Low | Stable, production-ready |
| AGENT-EUDR-036 EU Information System Interface | BUILT (100%) | Low | Bidirectional integration defined |
| AGENT-EUDR-037 Due Diligence Statement Creator | BUILT (100%) | Low | Amendment coordination interface defined |
| AGENT-EUDR-038 Reference Number Generator | BUILT (100%) | Low | Reference number lookup interface defined |
| AGENT-EUDR-039 Customs Declaration Support | BUILT (100%) | Low | Confiscation/release coordination interface defined |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC with enhanced non-compliance/appeal restrictions |
| SEC-003 Encryption at Rest (AES-256) | BUILT (100%) | Low | Document encryption integration |
| SEC-005 Centralized Audit Logging | BUILT (100%) | Low | Platform audit trail integration |
| SEC-011 PII Detection/Redaction | BUILT (100%) | Low | PII handling in communications |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System API | Published (v1.x) | Medium | Adapter pattern for API version changes; EUDR-036 abstracts EU IS details |
| EC published list of designated competent authorities | Published; periodically updated | Medium | Database-driven; manual verification quarterly; automated detection of authority changes |
| Machine translation service (DeepL/eTranslation) | Available | Medium | Multi-provider fallback; cached translations; human review for critical communications |
| Member State authority portal specifications | Varies by Member State | High | Phase 1: EU IS + email + manual; Phase 2: direct portal integration per Member State |
| eIDAS electronic identification | Available | Medium | Managed through EUDR-036 eIDAS integration |
| GDPR regulatory framework | Active | Low | GDPR compliance built into all data handling from design phase |
| Member State administrative procedural laws (appeal deadlines) | Published | Medium | Appeal deadline database maintained and verified per Member State; legal review annually |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Member State authority communication protocols change without notice | Medium | High | Multi-channel approach (EU IS + email + manual entry); failover routing; quarterly authority registry verification |
| R2 | Machine translation quality insufficient for legal communications (appeals, non-compliance responses) | Medium | High | Human review mandatory for all Level 3+ communications and all appeals; translation confidence scoring; EUDR terminology database for technical accuracy |
| R3 | Authority registry becomes stale (new authorities designated, contact info changes) | High | Medium | Quarterly automated verification against EC published lists; manual spot checks; operator feedback loop for detected changes |
| R4 | Volume of authority communications exceeds capacity during enforcement ramp-up (June 2026 SME deadline) | Medium | Medium | Horizontal scaling with Kubernetes; Redis caching for frequently accessed data; batch processing for high-volume periods |
| R5 | Appeal deadline calculation error due to incorrect Member State procedural law data | Low | Critical | Double verification of appeal deadline database; automated cross-check against published legal sources; human confirmation before appeal filing |
| R6 | Document exchange security breach (unauthorized access to confidential compliance data) | Low | Critical | AES-256-GCM encryption; time-limited links; access logging; watermarking; regular penetration testing; SEC-007 security scanning |
| R7 | GDPR conflict with Article 31 retention requirements (erasure request vs. 5-year retention) | Medium | Medium | Clear legal basis documentation (GDPR Article 17(3)(b) exemption for legal compliance obligations); automated GDPR response workflow |
| R8 | Authority communication in a language not yet supported | Low | Medium | Graceful degradation to English with manual translation workflow; prioritized language addition based on demand |
| R9 | Integration complexity with 39 upstream agents for response assembly | Medium | Medium | Well-defined agent API interfaces; mock adapters for testing; circuit breaker pattern; graceful degradation if specific agent unavailable |
| R10 | Non-compliance fine exposure exceeds customer expectations | Medium | High | Proactive non-compliance risk monitoring; early warning alerts; preemptive corrective action recommendations |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Information Request Handler Tests | 120+ | Request ingestion, classification, upstream data assembly, response formatting, deadline tracking |
| DDS Notification Engine Tests | 80+ | DDS query processing, amendment coordination, conditional acceptance handling |
| Inspection Coordination Tests | 100+ | Lifecycle management, readiness package assembly, inspector access, findings response |
| Non-Compliance Manager Tests | 120+ | Finding parsing, severity classification, corrective action generation, fine management, remediation tracking |
| Appeal Process Manager Tests | 100+ | Deadline calculation per 27 Member States, evidence assembly, filing, status tracking, hearing management |
| Document Exchange Portal Tests | 80+ | Encryption, ACLs, watermarking, virus scanning, download tracking, size limits |
| Multi-Language Engine Tests | 80+ | Language detection for all 24 EU languages, translation accuracy, terminology database, character encoding |
| Notification Routing Tests | 60+ | Authority registry CRUD, routing accuracy for all 27 Member States, cross-border routing, failover |
| Audit Trail Manager Tests | 100+ | SHA-256 hash chain integrity, retention enforcement, export generation, search and filtering |
| API Tests | 90+ | All 30+ endpoints, auth, error handling, pagination, rate limiting |
| Integration Tests | 40+ | Cross-agent data assembly from 10+ upstream agents, EUDR-036/037/038/039 bidirectional integration |
| Performance Tests | 25+ | Response assembly timing, concurrent communication handling, audit trail query performance |
| Security Tests | 30+ | Encryption validation, RBAC enforcement, PII handling, document exchange security, GDPR compliance |
| **Total** | **900+** | |

### 13.2 Golden Test Scenarios

Each major communication type will have dedicated golden test scenarios:

1. **Information Request Scenarios** (7 scenarios)
   - Standard information request about cocoa supply chain -- complete response assembly
   - Urgent information request (48-hour deadline) -- expedited response workflow
   - Bulk information request for all DDS in a quarter -- batch response generation
   - Cross-border information request (Article 17) -- multi-authority routing
   - Third-country information exchange (Article 18) -- third-country cooperation workflow
   - Request for data not held by operator -- appropriate "data not available" response
   - Request referencing a withdrawn DDS -- withdrawal documentation included

2. **Inspection Scenarios** (5 scenarios)
   - Announced inspection with 10-day notice -- full preparation workflow
   - Unannounced inspection -- immediate response with pre-assembled package
   - Inspection with real-time document requests -- on-demand document delivery
   - Inspection with findings requiring corrective action -- findings response workflow
   - Multi-facility concurrent inspections -- parallel preparation management

3. **Non-Compliance Scenarios** (6 scenarios)
   - Level 2 formal warning -- acknowledgement and improvement plan
   - Level 3 corrective action order -- corrective action plan generation and submission
   - Level 4 fine notification -- fine management and payment/appeal decision
   - Level 5 goods confiscation -- customs coordination and release request
   - Level 5 market exclusion -- affected product tracking and reinstatement
   - Duplicate findings from multiple authorities -- coordination workflow

4. **Appeal Scenarios** (5 scenarios)
   - Administrative appeal against fine (Germany, 30-day deadline) -- evidence assembly and filing
   - Administrative appeal against confiscation (France, 60-day deadline) -- hearing preparation
   - Partially successful appeal -- partial decision handling
   - Judicial review after failed administrative appeal -- second-level appeal workflow
   - Appeal deadline on weekend/holiday -- deadline extension calculation

5. **Multi-Language Scenarios** (5 scenarios)
   - German-language information request with English platform data -- translation workflow
   - French-language appeal filing -- EUDR terminology in French
   - Mixed-language communication (Dutch/French Belgian authority) -- bilingual handling
   - Greek-language DDS query with technical EUDR terms -- terminology database validation
   - Communication in non-EU language (Article 18 third-country) -- manual translation routing

Total: 28 golden test scenarios covering all communication types, Member States, and edge cases.

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **Competent Authority** | National authority designated by each EU Member State under Article 22 to enforce EUDR |
| **Article 15** | Powers of competent authorities to carry out checks, request information, and conduct inspections |
| **Article 16** | Penalties framework including fines (up to 4% EU turnover), confiscation, and market exclusion |
| **Article 17** | Information exchange between competent authorities of different EU Member States |
| **Article 18** | Information exchange between EU competent authorities and third-country authorities |
| **Article 19** | Right to effective remedy, administrative review, and judicial appeal |
| **Article 20** | Risk-based approach to substantive checks with enhanced rates for high-risk products |
| **Article 22** | Designation of competent authorities by EU Member States |
| **Article 31** | Record-keeping obligation (minimum 5 years from DDS submission) |
| **eIDAS** | EU Regulation on electronic IDentification, Authentication and trust Services (Regulation (EU) No 910/2014) |
| **GDPR** | General Data Protection Regulation (Regulation (EU) 2016/679) |
| **PEC** | Posta Elettronica Certificata -- Italian certified email system |
| **De-Mail** | German secure electronic communication system |
| **ePUAP** | Polish electronic Platform of Public Administration Services |
| **e-Boks** | Danish digital post system |
| **SAD** | Single Administrative Document -- EU customs declaration form |
| **BLE** | Bundesanstalt fuer Landwirtschaft und Ernaehrung -- German Federal Agency for Agriculture and Food |
| **DGCCRF** | Direction Generale de la Concurrence, de la Consommation et de la Repression des Fraudes -- French consumer protection authority |
| **NVWA** | Nederlandse Voedsel- en Warenautoriteit -- Netherlands Food and Consumer Product Safety Authority |
| **SHA-256** | Secure Hash Algorithm producing 256-bit hash values for data integrity verification |
| **AES-256-GCM** | Advanced Encryption Standard with 256-bit key in Galois/Counter Mode -- authenticated encryption |
| **ACL** | Access Control List -- specifies which users or authorities can access specific resources |
| **NLP** | Natural Language Processing -- computational linguistics techniques for language detection and translation |

### Appendix B: EUDR Article 15 -- Competent Authority Powers (Summary)

Article 15(2) grants competent authorities the following powers:

- **(a)** Request any information relevant to the operator's or trader's compliance with the Regulation, including access to relevant documents, data, procedures, and any other material relevant to the checks, in any form or format, including electronic.
- **(b)** Carry out on-the-spot inspections, including in the form of unannounced checks, of business premises of operators and traders.
- **(c)** Take samples of relevant commodities or products for examination or testing.
- **(d)** Initiate interim measures, including provisional seizure of relevant commodities or products, where there are sufficient grounds to believe that an operator or trader has failed to comply.
- **(e)** Take any necessary temporary or permanent measure to ensure compliance.

The agent implements comprehensive support for all five sub-paragraphs through the Information Request Handler (a), Inspection Coordination Engine (b, c), and Non-Compliance Response Manager (d, e).

### Appendix C: EUDR Article 16 -- Penalties Framework (Summary)

Article 16(1) requires Member States to establish penalties that include:

- **(a)** Fines proportionate to environmental damage and market value of relevant commodities
- **(b)** Confiscation of the relevant commodities or products
- **(c)** Confiscation of revenues gained by the operator or trader
- **(d)** Temporary exclusion from public procurement processes (maximum 12 months)
- **(e)** Temporary prohibition from placing relevant products on the EU market or exporting them

Article 16(2) specifies that maximum fines shall be at least 4% of the operator's total annual Union-wide turnover in the financial year preceding the decision to impose the fine.

### Appendix D: EUDR Article 19 -- Right to Appeal (Summary)

Article 19 establishes:

- **(1)** Operators and traders affected by decisions of competent authorities under this Regulation shall have access to an effective remedy, including the right to appeal such decisions before a court or tribunal.
- **(2)** Member States shall ensure that the operator or trader is given an opportunity to be heard before any decision adversely affecting them is taken.

The agent implements full appeal lifecycle management including deadline calculation per Member State administrative procedural law, evidence assembly, hearing preparation, and outcome tracking.

### Appendix E: Member State Appeal Deadline Reference

| Member State | Administrative Appeal Deadline | Judicial Review Deadline | Appeal Body |
|-------------|-------------------------------|-------------------------|-------------|
| Austria | 4 weeks from notification | 6 weeks from administrative decision | Verwaltungsgericht |
| Belgium | 30 days from notification | 60 days from administrative decision | Conseil d'Etat / Raad van State |
| Bulgaria | 14 days from notification | 30 days from administrative decision | Administrative Court |
| Croatia | 30 days from notification | 30 days from administrative decision | High Administrative Court |
| Cyprus | 75 days from notification | N/A (direct judicial review) | Supreme Court |
| Czech Republic | 30 days from notification | 2 months from administrative decision | Administrative Court |
| Denmark | 4 weeks from notification | 6 months from administrative decision | Environment and Food Board of Appeal |
| Estonia | 30 days from notification | 30 days from administrative decision | Administrative Court |
| Finland | 30 days from notification | 30 days from administrative decision | Administrative Court |
| France | 2 months from notification | 2 months from administrative decision | Tribunal administratif |
| Germany | 1 month from notification | 1 month from administrative decision | Verwaltungsgericht |
| Greece | 60 days from notification | 60 days from administrative decision | Administrative Court |
| Hungary | 30 days from notification | 30 days from administrative decision | Administrative Court |
| Ireland | 21 days from notification | 3 months from administrative decision | High Court |
| Italy | 60 days from notification | 60 days from administrative decision | Tribunale Amministrativo Regionale |
| Latvia | 30 days from notification | 1 month from administrative decision | Administrative District Court |
| Lithuania | 30 days from notification | 1 month from administrative decision | Regional Administrative Court |
| Luxembourg | 3 months from notification | 3 months from administrative decision | Tribunal administratif |
| Malta | 30 days from notification | 20 days from administrative decision | Administrative Review Tribunal |
| Netherlands | 6 weeks from notification | 6 weeks from administrative decision | Rechtbank (Administrative Court) |
| Poland | 14 days from notification | 30 days from administrative decision | Wojewodzki Sad Administracyjny |
| Portugal | 30 days from notification | 3 months from administrative decision | Tribunal Administrativo |
| Romania | 30 days from notification | 6 months from administrative decision | Court of Appeal |
| Slovakia | 30 days from notification | 2 months from administrative decision | Administrative Court |
| Slovenia | 15 days from notification | 30 days from administrative decision | Administrative Court |
| Spain | 1 month from notification | 2 months from administrative decision | Juzgado de lo Contencioso-Administrativo |
| Sweden | 3 weeks from notification | 3 weeks from administrative decision | Administrative Court |

*Note: Deadlines are indicative and subject to specific Member State procedural rules. The agent maintains a verified database with actual deadlines per communication type and authority.*

### Appendix F: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 on the making available on the Union market and the export from the Union of certain commodities and products associated with deforestation and forest degradation (EUDR)
2. General Data Protection Regulation (EU) 2016/679 (GDPR)
3. Regulation (EU) No 910/2014 on electronic identification and trust services for electronic transactions in the internal market (eIDAS)
4. Union Customs Code (Regulation (EU) No 952/2013)
5. Regulation (EU) 2019/1020 on market surveillance and compliance of products
6. European Commission -- EUDR Implementation Guidance Documents
7. European Commission -- Designated Competent Authorities for EUDR (Article 22 published list)
8. Council of Europe -- Administrative Law Handbook (appeal procedures by Member State)
9. European e-Justice Portal -- Administrative Procedural Law by Member State
10. ISO 639-1:2002 -- Codes for the representation of names of languages
11. ISO 3166-1:2020 -- Codes for the representation of names of countries

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-13 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Legal Counsel | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-13 | GL-ProductManager | Initial draft created: 9 P0 features (Information Request Handler, DDS Submission Notification, Inspection Coordination, Non-Compliance Response, Appeal Process Management, Document Exchange Portal, Multi-Language Communication, Notification Routing, Communication Audit Trail), 7 engines, full regulatory coverage (Articles 15-19, 20, 22, 31), 27 Member State authority registry, 24 EU language support, V119 database migration, 30+ API endpoints, 23 RBAC permissions, 900+ test targets, 28 golden test scenarios |
