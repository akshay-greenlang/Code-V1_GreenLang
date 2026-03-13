# PRD: AGENT-EUDR-031 -- Stakeholder Engagement Tool

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-031 |
| **Agent ID** | GL-EUDR-SET-031 |
| **Component** | Stakeholder Engagement Tool Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence (Category 5) |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-12 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 2, 4, 8, 9, 10, 11, 12, 29, 31; ILO Convention 169; UN Declaration on the Rights of Indigenous Peoples (UNDRIP); EU Corporate Sustainability Due Diligence Directive (CSDDD) Articles 7, 8, 9 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) embeds stakeholder engagement as a structural requirement throughout the due diligence process. Article 10(2)(e) requires operators to consider "information from consultations with indigenous peoples, local communities, and other stakeholders" as part of their risk assessment. Article 11(2) mandates that risk mitigation measures include consultation with affected communities when supply chains intersect their territories. Article 29(3)(c) conditions the European Commission's country benchmarking on "respect for the rights of indigenous peoples, local communities, and other customary tenure rights holders, including their right to Free, Prior and Informed Consent (FPIC)." Recital 32 explicitly references the United Nations Declaration on the Rights of Indigenous Peoples (UNDRIP) and ILO Convention 169, establishing that EUDR-compliant supply chains must be built on meaningful, documented, and ongoing engagement with affected stakeholders.

The scope of stakeholder engagement under the EUDR extends well beyond indigenous peoples alone. Operators must engage with local communities affected by commodity production, smallholder farmer cooperatives supplying raw materials, civil society organizations monitoring deforestation and human rights, workers in production and processing facilities, government authorities in producing countries, certification bodies verifying compliance, and the operators' own supply chain partners. Globally, approximately 1.6 billion people depend on forests for their livelihoods, 370 million indigenous peoples occupy 28% of the world's land surface, and an estimated 500 million smallholder farmers produce 80% of the food consumed in developing countries -- many of whom are directly involved in the supply chains of EUDR-regulated commodities (cocoa, coffee, palm oil, rubber, soya, cattle, wood).

The EU Corporate Sustainability Due Diligence Directive (CSDDD), adopted in 2024 with phased enforcement beginning in 2027, further elevates stakeholder engagement from a compliance consideration to a core legal obligation. CSDDD Article 7 requires companies to carry out meaningful engagement with affected stakeholders as part of their due diligence process, including indigenous peoples and local communities. CSDDD Article 8 mandates the establishment of grievance mechanisms accessible to affected persons and stakeholders. CSDDD Article 9 requires companies to remediate adverse impacts identified through stakeholder engagement. The convergence of EUDR and CSDDD creates an urgent need for integrated stakeholder engagement infrastructure that serves both regulatory frameworks.

The GreenLang platform has built 30 EUDR agents spanning supply chain traceability (EUDR-001 through EUDR-015), risk assessment (EUDR-016 through EUDR-020), and due diligence (EUDR-021 through EUDR-030). Agent EUDR-021 (Indigenous Rights Checker) provides territory overlap detection and FPIC verification, and Agent EUDR-025 (Risk Mitigation Advisor) includes a stakeholder collaboration hub. However, the platform currently has no dedicated agent that manages the full lifecycle of stakeholder engagement across all stakeholder categories, provides structured FPIC workflow management, operates a compliant grievance mechanism, maintains comprehensive consultation records, manages multi-channel communications, and generates the engagement documentation required for DDS submission and competent authority inspection. Today, operators face the following critical gaps:

- **No unified stakeholder registry**: Operators interact with hundreds of stakeholders across their EUDR supply chains -- indigenous communities, local villages, cooperatives, NGOs, workers' unions, government agencies, certification bodies, and civil society monitors. These stakeholders are tracked in disconnected systems: procurement databases, sustainability team spreadsheets, legal department case files, and personal email archives. There is no centralized, categorized, rights-aware stakeholder registry that maps each stakeholder to their role, rights, geographic location, relevant supply chain nodes, engagement history, and applicable legal protections.

- **No structured FPIC workflow management**: Free, Prior and Informed Consent is a multi-stage process defined by ILO Convention 169 and UNDRIP, requiring: identification of affected communities, provision of information in accessible formats and languages, adequate time for community deliberation (typically 90-180 days), genuine consultation through culturally appropriate mechanisms, documentation of consent or objection, formal agreement recording, and ongoing monitoring of consent conditions. There is no workflow engine that manages these stages with configurable timelines, SLA enforcement, evidence collection, and audit trail. EUDR-021 provides FPIC documentation verification, but it does not manage the FPIC process itself -- it only checks whether completed FPIC documentation is adequate.

- **No grievance mechanism**: The EU requires operators to establish accessible, transparent, and effective complaint mechanisms through which affected stakeholders can raise concerns about supply chain impacts. CSDDD Article 8 makes this an explicit legal requirement. The UN Guiding Principles on Business and Human Rights (Principle 31) define the effectiveness criteria for grievance mechanisms: legitimate, accessible, predictable, equitable, transparent, rights-compatible, and a source of continuous learning. There is no structured grievance intake, triage, investigation, resolution, and appeal system integrated with the EUDR compliance infrastructure.

- **No consultation record management**: EUDR Article 10(2)(e) requires operators to consider "information from consultations." This implies that consultations must be documented in a manner that can be reviewed by competent authorities under Articles 14-16. There is no structured system to record consultation objectives, participants, methodology, discussions, outcomes, commitments, and follow-up actions in an audit-ready format with provenance tracking.

- **No stakeholder communication management**: Engaging hundreds of stakeholders across multiple countries, languages, and communication channels (in-person meetings, community radio, mobile SMS, email, portal notifications, printed correspondence) requires a communications management platform. There is no system to plan, schedule, execute, and track stakeholder communications with delivery confirmation, response tracking, and multi-language support.

- **No indigenous rights engagement verification**: While EUDR-021 checks whether indigenous rights are respected at the data level (territory overlap, FPIC documentation), it does not verify whether the operator has actually engaged with indigenous communities in a meaningful way. Operators need a system that tracks direct engagement activities: community meetings, information sessions, negotiation processes, benefit-sharing discussions, and ongoing relationship management.

- **No compliance reporting for stakeholder engagement**: DDS submissions must demonstrate that stakeholder engagement was conducted as part of due diligence. Competent authorities under Articles 14-16 may request evidence of engagement processes. There is no documentation engine that generates audit-ready stakeholder engagement reports covering all engagement activities, FPIC processes, grievance records, consultation outcomes, and communication logs, formatted for regulatory submission and inspection.

Without solving these problems, EU operators cannot demonstrate the stakeholder engagement component of their EUDR due diligence, cannot satisfy CSDDD grievance mechanism requirements, cannot produce evidence of meaningful consultation with indigenous peoples and local communities, and cannot respond to competent authority requests for engagement documentation. This exposes operators to penalties of up to 4% of annual EU turnover under EUDR and additional liability under CSDDD, plus reputational damage, certification scheme exclusion (FSC, RSPO, Rainforest Alliance all require stakeholder engagement), and litigation risk from affected communities.

### 1.2 Solution Overview

Agent-EUDR-031: Stakeholder Engagement Tool is a dedicated compliance agent that manages the full lifecycle of stakeholder engagement for EUDR due diligence. It provides a centralized stakeholder registry, structured FPIC workflow management, a compliant grievance mechanism, comprehensive consultation record management, multi-channel communication capabilities, indigenous rights engagement verification, and audit-ready compliance reporting. It is the 31st agent in the EUDR agent family and extends the Due Diligence sub-category (Category 5) with dedicated stakeholder engagement infrastructure.

The agent integrates with the existing EUDR agent ecosystem: EUDR-001 (Supply Chain Mapping Master) for supply chain graph data to identify which stakeholders are relevant to which supply chain nodes, EUDR-002 (Geolocation Verification) for plot coordinates to map stakeholder geographic proximity, EUDR-021 (Indigenous Rights Checker) for indigenous territory data and FPIC documentation verification, EUDR-025 (Risk Mitigation Advisor) for stakeholder collaboration on mitigation measures, EUDR-026 (Due Diligence Orchestrator) for workflow integration into the end-to-end due diligence process, and EUDR-030 (Documentation Generator) for inclusion of engagement evidence in DDS packages.

Core capabilities:

1. **Stakeholder Mapper** -- Centralized registry of all stakeholders across the EUDR supply chain, categorized by type (indigenous peoples, local communities, smallholder cooperatives, NGOs, workers/unions, government authorities, certification bodies, civil society monitors, academic institutions, media). Each stakeholder record includes identity, geographic location, relevant supply chain nodes, rights classification, legal protections, engagement history, preferred communication channels, language preferences, and relationship status. Supports automated stakeholder discovery from supply chain graph (EUDR-001), territory overlap data (EUDR-021), and protected area data (EUDR-022). Maps stakeholders to EUDR Articles and applicable legal frameworks (ILO 169, UNDRIP, national legislation).

2. **FPIC Workflow Engine** -- Multi-stage workflow for managing Free, Prior and Informed Consent processes per ILO Convention 169 and UNDRIP. Stages: (1) Identification of affected communities, (2) Information provision in accessible formats and local languages, (3) Community deliberation period with configurable timeline (default 90 days, extendable to 180 days), (4) Consultation through culturally appropriate mechanisms, (5) Consent or objection recording with community representative verification, (6) Formal agreement documentation with benefit-sharing terms, (7) Ongoing monitoring of consent conditions and agreement compliance. Each stage has configurable SLAs, evidence requirements, approval gates, and escalation rules. Integrates with EUDR-021 for FPIC documentation verification.

3. **Grievance Mechanism** -- Structured complaint management system compliant with UN Guiding Principles Principle 31 effectiveness criteria. Intake channels: web portal, mobile app, SMS hotline, email, community-based submission points. Triage engine classifies complaints by severity (critical/high/medium/low), category (environmental, human rights, labor, land rights, community impact, process), and urgency. Investigation workflow with evidence collection, stakeholder interview tracking, and root cause analysis. Resolution management with remediation tracking, satisfaction assessment, and appeal process. Full transparency: complainants receive acknowledgement within 48 hours, status updates at configurable intervals, and resolution notification. Anonymous reporting supported. Multi-language support for 12+ languages.

4. **Consultation Record Manager** -- Structured documentation system for all consultations with indigenous peoples, local communities, and other stakeholders per EUDR Article 10(2)(e). Each consultation record captures: objectives, date/time/location, participants (with role and affiliation), methodology (community meeting, focus group, survey, bilateral negotiation), topics discussed, outcomes and decisions, commitments made by each party, follow-up actions with deadlines and responsible parties, and supporting evidence (meeting minutes, photos, audio recordings with consent, signed attendance sheets). Records are immutable after finalization, with SHA-256 provenance hashes. Supports offline data entry for field consultations with sync-on-connect capability.

5. **Communication Hub** -- Multi-channel stakeholder communication platform supporting email, SMS, WhatsApp Business API, community portal notifications, printed correspondence generation, and community radio script generation. Communication planning with stakeholder segmentation, message scheduling, delivery tracking, and response management. Template library with 100+ pre-built communication templates for common engagement scenarios (FPIC notification, grievance acknowledgement, consultation invitation, agreement renewal, compliance update) in 12+ languages. Campaign management for coordinated multi-stakeholder communications. Delivery confirmation and read receipt tracking where channel supports it.

6. **Indigenous Rights Engagement Verifier** -- Verification engine that assesses whether operator engagement with indigenous communities meets the substantive requirements of ILO Convention 169, UNDRIP, and applicable national legislation. Evaluates engagement quality across 6 dimensions: (a) Cultural appropriateness of engagement methods, (b) Language accessibility of information provided, (c) Adequacy of deliberation time, (d) Inclusiveness of community representation, (e) Genuineness of consultation (not merely notification), (f) Respect for community decision-making processes. Generates engagement quality score (0-100) with dimension-level decomposition. Integrates with EUDR-021 territory data for geographic context.

7. **Compliance Reporter** -- Generates audit-ready stakeholder engagement documentation for DDS submission (Article 12), competent authority inspection (Articles 14-16), certification scheme audits (FSC, RSPO, Rainforest Alliance), and third-party verification. Report types: (a) Stakeholder Engagement Summary for DDS, (b) FPIC Compliance Report, (c) Grievance Mechanism Annual Report, (d) Consultation Register, (e) Indigenous Rights Engagement Assessment, (f) Communication Activity Log, (g) Stakeholder Engagement Effectiveness Report. All reports include SHA-256 provenance hashes, regulatory article mapping, and evidence cross-references. Formats: PDF, JSON, HTML, XLSX. Multi-language support (EN, FR, DE, ES, PT, ID, SW).

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Stakeholder registry coverage | All stakeholders mapped for 100% of active supply chains | % of supply chains with complete stakeholder mapping |
| Stakeholder categorization accuracy | >= 99% correct stakeholder type classification | Cross-validation against manual stakeholder assessments |
| FPIC workflow coverage | 100% of supply chains intersecting indigenous territories with active FPIC workflows | % of indigenous territory overlaps with FPIC workflow |
| FPIC stage compliance | 100% of FPIC workflows following all 7 mandatory stages | Stage completion audit |
| Grievance mechanism accessibility | Available in 12+ languages, 5+ intake channels | Language and channel coverage matrix |
| Grievance resolution time | < 30 days for standard complaints; < 7 days for critical | Median resolution time by severity |
| Consultation record completeness | 100% of consultations with all mandatory fields documented | Completeness scoring per consultation record |
| Communication delivery rate | >= 95% successful delivery across all channels | Delivery confirmation tracking |
| Engagement quality score accuracy | >= 95% agreement with expert assessment | Blind comparison with indigenous rights expert evaluations |
| Compliance report generation time | < 15 seconds per report (JSON); < 30 seconds (PDF) | Report generation benchmarks |
| DDS integration completeness | 100% of required stakeholder engagement fields in DDS populated | Schema validation against EU Information System spec |
| Zero-hallucination guarantee | 100% deterministic calculations, no LLM in critical path | Bit-perfect reproducibility tests |
| EUDR regulatory coverage | Full coverage of Articles 2, 4, 8, 9, 10, 11, 12, 29 stakeholder requirements | Regulatory compliance matrix |
| 5-year record retention | 100% of records retrievable within retention period | Automated retention validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, combined with the 13,000+ companies that will fall under CSDDD stakeholder engagement obligations by 2029, representing an estimated stakeholder engagement technology market of 2-4 billion EUR. The convergence of EUDR, CSDDD, and voluntary sustainability standards (FSC, RSPO, Rainforest Alliance) creates compounding demand for integrated engagement platforms.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of EUDR-regulated commodities sourcing from regions with indigenous populations, local community dependencies, and smallholder farmer supply bases, estimated at 400-700M EUR. Primary demand concentrators: cocoa (West Africa, Latin America), palm oil (Southeast Asia), coffee (Latin America, East Africa), rubber (Southeast Asia), soya (South America), cattle (South America), and wood (tropical forests globally).
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1 using stakeholder engagement capabilities, representing 25-45M EUR in stakeholder engagement module ARR. Additional revenue from CSDDD cross-sell as enforcement approaches in 2027.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) with supply chains intersecting indigenous territories and local community lands
- Multinational food and beverage companies sourcing cocoa, coffee, palm oil, and soya from smallholder farmer networks
- Timber and paper industry operators with tropical wood sourcing from indigenous forest regions requiring FPIC
- Automotive and tire manufacturers (rubber supply chain) sourcing from Southeast Asian communities
- Meat and leather importers (cattle supply chain) sourcing from South American indigenous and traditional communities

**Secondary:**
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance) requiring evidence of stakeholder engagement for certification audits
- Compliance consulting firms delivering stakeholder engagement programs on behalf of operators
- NGOs and indigenous rights organizations partnering with operators on community engagement
- Financial institutions requiring evidence of stakeholder engagement for ESG due diligence under CSDDD
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026
- Government agencies in producing countries collaborating on EUDR implementation

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet tracking | No cost; familiar tools | Cannot manage multi-stage FPIC; no grievance workflow; no audit trail; error-prone; no multi-language; no communication tracking | Full lifecycle automation; 7-stage FPIC; compliant grievance mechanism; complete audit trail |
| Generic stakeholder management (Borealis, Darzin, Tractivity) | Multi-purpose engagement tracking; mature UI | Not EUDR-specific; no FPIC workflow; no indigenous territory integration; no regulatory compliance reporting; no supply chain linkage | Purpose-built for EUDR/CSDDD; FPIC-native; indigenous territory integration via EUDR-021; supply chain-aware |
| Generic GRC platforms (OneTrust, MetricStream, SAP GRC) | Enterprise risk management; policy management | No FPIC workflow; no indigenous rights compliance; no community-level engagement; no commodity-specific features | FPIC workflow engine; indigenous rights verification; community-level engagement; commodity-specific templates |
| Sustainability consultancies (ERM, BSR, Shift) | Deep FPIC expertise; community relationships | Project-based (EUR 50-200K per engagement); no technology platform; no real-time tracking; not scalable | Always-on platform; 10x more cost-effective; real-time tracking; scales to thousands of stakeholders |
| Certification scheme platforms (FSC Marketplace, RSPO eTrace) | Scheme-specific engagement requirements | Siloed to single scheme; no cross-scheme integration; no EUDR DDS integration; limited to certified operators | Cross-scheme; EUDR DDS-native; available to all operators; multi-certification alignment |
| In-house custom builds | Tailored to organization | 12-18 month build; no regulatory updates; no indigenous territory data; no multi-language support | Ready now; continuous regulatory updates; territory data via EUDR-021; 12+ languages |

### 2.4 Differentiation Strategy

1. **EUDR-native stakeholder engagement** -- Not a generic stakeholder management tool. Every feature maps to specific EUDR Articles and CSDDD requirements, ensuring regulatory defensibility.
2. **30-agent ecosystem integration** -- Pre-built integration with 30 existing EUDR agents. Stakeholder data flows bidirectionally with supply chain mapping (EUDR-001), indigenous territory checks (EUDR-021), risk mitigation (EUDR-025), due diligence orchestration (EUDR-026), and documentation generation (EUDR-030).
3. **FPIC-native architecture** -- The only platform with a dedicated 7-stage FPIC workflow engine that enforces ILO 169 and UNDRIP procedural requirements with configurable timelines, evidence requirements, and approval gates.
4. **Compliant grievance mechanism** -- Built from the ground up to satisfy UN Guiding Principles Principle 31 effectiveness criteria and CSDDD Article 8 requirements, with multi-channel intake, investigation workflow, and remediation tracking.
5. **Community-centered design** -- Multi-language support (12+ languages), offline-capable field tools, culturally appropriate communication templates, and accessibility features for low-literacy stakeholders (audio, visual, community radio).
6. **Zero-hallucination compliance** -- All scoring, classification, and reporting is deterministic with no LLM in the critical path. Every output is reproducible and provenance-tracked with SHA-256 hashes.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to demonstrate EUDR-compliant stakeholder engagement in their DDS | 100% of customers pass Articles 10/11 audits for stakeholder engagement component | Q2 2026 |
| BG-2 | Reduce time-to-document stakeholder engagement from weeks to minutes | 95% reduction in engagement documentation time | Q2 2026 |
| BG-3 | Become the reference stakeholder engagement platform for EUDR and CSDDD compliance | 500+ enterprise customers using stakeholder engagement module | Q4 2026 |
| BG-4 | Prevent EUDR penalties attributable to inadequate stakeholder engagement | Zero EUDR penalties for active customers related to engagement gaps | Ongoing |
| BG-5 | Build CSDDD-ready stakeholder engagement infrastructure ahead of 2027 enforcement | Stakeholder engagement module satisfies CSDDD Articles 7, 8, 9 requirements | Q4 2026 |
| BG-6 | Support certification scheme compliance (FSC, RSPO, RA) for stakeholder requirements | 100% of certification scheme stakeholder requirements covered | Q3 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Unified stakeholder registry | Map and categorize all supply chain stakeholders with rights classification, engagement history, and supply chain linkage |
| PG-2 | FPIC process management | Manage complete 7-stage FPIC workflows with timeline enforcement, evidence collection, and audit trail |
| PG-3 | Compliant grievance mechanism | Operate a multi-channel, multi-language grievance system satisfying UNGP Principle 31 and CSDDD Article 8 |
| PG-4 | Consultation documentation | Record all stakeholder consultations with full audit trail for Article 10(2)(e) compliance |
| PG-5 | Multi-channel communications | Manage stakeholder communications across 6+ channels with delivery tracking and response management |
| PG-6 | Engagement quality verification | Assess and score the quality of indigenous rights engagement against ILO 169 and UNDRIP standards |
| PG-7 | Regulatory reporting | Generate audit-ready engagement reports for DDS, competent authorities, and certification bodies |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Stakeholder lookup performance | < 100ms p99 for single stakeholder lookup; < 2 seconds for full supply chain stakeholder query |
| TG-2 | FPIC workflow processing | < 500ms per workflow state transition |
| TG-3 | Grievance intake processing | < 3 seconds from submission to acknowledgement generation |
| TG-4 | Communication dispatch throughput | 10,000 messages per minute across all channels |
| TG-5 | Report generation performance | < 15 seconds per compliance report (JSON); < 30 seconds (PDF) |
| TG-6 | API response time | < 200ms p95 for standard queries |
| TG-7 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-8 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-9 | Data freshness | Stakeholder records updated within 24 hours of engagement activity |
| TG-10 | Offline sync latency | Field data synced within 60 seconds of connectivity restoration |

### 3.4 Non-Goals

1. Indigenous territory spatial database management (EUDR-021 handles territory data)
2. Territory overlap detection and FPIC documentation verification (EUDR-021 handles these)
3. Risk assessment calculation (EUDR-028 Risk Assessment Engine handles this)
4. Mitigation measure design and effectiveness tracking (EUDR-029 handles this)
5. Due diligence workflow orchestration (EUDR-026 handles this)
6. DDS document generation (EUDR-030 Documentation Generator handles this)
7. Supply chain graph topology management (EUDR-001 handles this)
8. Satellite monitoring of deforestation (EUDR-003/004/005 handle this)
9. Direct payment or financial transaction processing with stakeholders
10. Legal representation or mediation services for grievance disputes

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries, 200+ cooperatives |
| **EUDR Pressure** | Must demonstrate stakeholder engagement in DDS submissions; competent authority audit imminent; board mandate for zero compliance gaps |
| **Pain Points** | Cannot identify which communities are affected by sourcing activities; FPIC processes managed through email with no structured workflow; no grievance mechanism in place; consultation records scattered across sustainability team files; cannot produce engagement evidence for auditors within required timeframe |
| **Goals** | Centralized stakeholder registry linked to supply chain; structured FPIC workflows with SLA tracking; compliant grievance mechanism; one-click engagement compliance reports for DDS and auditors |
| **Technical Skill** | Moderate -- comfortable with web applications and compliance platforms |

### Persona 2: Indigenous Rights Specialist -- Sofia (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Indigenous Rights & Community Relations Manager at a European palm oil refinery |
| **Company** | 6,000 employees, sourcing from Indonesia (Kalimantan, Sumatra, Papua) and Colombia |
| **EUDR Pressure** | Multiple NGO reports of indigenous rights violations in supply chain; FPIC compliance required for 50+ plantation concessions overlapping indigenous territories |
| **Pain Points** | No structured FPIC process across 50+ concessions; each concession managed independently with different approaches; cannot track FPIC stage progression across portfolio; community representatives change and contact information is lost; benefit-sharing agreements not systematically managed; no way to verify engagement quality |
| **Goals** | Portfolio-wide FPIC workflow management; engagement quality scoring; community representative registry; benefit-sharing agreement tracking; indigenous rights engagement reports for FSC/RSPO audits |
| **Technical Skill** | High -- comfortable with GIS, stakeholder management tools, and field data collection |

### Persona 3: Grievance Officer -- Klaus (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Stakeholder Grievance & Remediation at an EU timber importer |
| **Company** | 2,500 employees, importing tropical wood from Brazil, DRC, Myanmar, Indonesia |
| **EUDR Pressure** | CSDDD requires establishment of grievance mechanism by 2027; EUDR auditors requesting evidence of community complaint handling; FSC certification requires operational grievance mechanism |
| **Pain Points** | No formal grievance mechanism exists; complaints arrive through ad-hoc channels (email, social media, NGO reports) with no tracking; cannot demonstrate complaint resolution process to auditors; no anonymous reporting capability; community trust is low due to perceived lack of responsiveness |
| **Goals** | Formal multi-channel grievance mechanism with intake, triage, investigation, and resolution workflows; anonymous reporting; stakeholder-facing transparency dashboard; resolution time SLA tracking; annual grievance report for auditors and board |
| **Technical Skill** | Moderate -- comfortable with case management systems and reporting tools |

### Persona 4: Community Liaison Officer -- Amara (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Field Community Liaison at a cocoa cooperative federation in Cote d'Ivoire |
| **Company** | Local cooperative federation working with EU importers on EUDR compliance |
| **EUDR Pressure** | Must facilitate engagement between EU buyers and 150+ smallholder farming communities; must document consultations for EUDR compliance evidence |
| **Pain Points** | Conducts community meetings in remote locations with no internet; paper-based meeting records are lost or damaged; cannot communicate consultation outcomes back to EU compliance teams in real time; language barriers between French-speaking compliance teams and local-language communities; no standardized consultation documentation format |
| **Goals** | Offline-capable mobile tool for consultation documentation; multi-language support (French, local languages); photo and audio evidence capture; automatic sync to compliance platform when connectivity available; standardized consultation templates |
| **Technical Skill** | Low-moderate -- comfortable with mobile phones and basic applications |

### Persona 5: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm specializing in EUDR and sustainability certification audits |
| **EUDR Pressure** | Must verify operator stakeholder engagement as part of EUDR due diligence audits; must assess FPIC compliance and grievance mechanism effectiveness |
| **Pain Points** | Operators provide inconsistent engagement evidence; FPIC documentation incomplete; no standardized way to verify consultation quality; grievance mechanism evidence scattered or non-existent; cannot assess engagement effectiveness from available records |
| **Goals** | Access read-only stakeholder engagement data with full provenance; verify FPIC workflow compliance; audit grievance mechanism effectiveness; validate consultation record completeness; review engagement quality scores with methodology transparency |
| **Technical Skill** | Moderate -- comfortable with audit software, document review, and compliance assessment tools |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(28)** | Definition of "legally produced" -- in accordance with relevant legislation including laws on land tenure and indigenous peoples' rights | Stakeholder Mapper (F1) identifies communities with legal protections; FPIC Workflow Engine (F2) manages consent processes required by national legislation; Indigenous Rights Engagement Verifier (F6) validates engagement meets legal standards |
| **Art. 4(2)** | Operators shall exercise due diligence with regard to all relevant products, including collecting information and assessing risk | Stakeholder engagement data feeds into Article 9 information gathering and Article 10 risk assessment through integration with EUDR-027 and EUDR-028 |
| **Art. 8(1)** | Operators shall establish a due diligence system comprising information gathering, risk assessment, and risk mitigation | Stakeholder engagement is cross-cutting across all three due diligence phases: information from consultations (gathering), stakeholder concerns as risk factor (assessment), community engagement as mitigation measure (mitigation) |
| **Art. 8(2)** | Operators shall make available evidence of due diligence system upon request | Compliance Reporter (F7) generates inspection-ready engagement evidence; all records retained for 5+ years per Article 31 |
| **Art. 9(1)** | Information to be collected includes supporting evidence for deforestation-free and legal production | Consultation Record Manager (F4) captures community-provided evidence; FPIC Workflow Engine (F2) documents consent as legal production evidence |
| **Art. 10(1)** | Operators shall assess and identify risk of non-compliance | Stakeholder engagement data (consultation outcomes, grievance patterns, FPIC status) feeds into EUDR-028 risk assessment as mandatory input per Article 10(2)(e) |
| **Art. 10(2)(d)** | Risk factor: risk linked to the country of production, including respect for indigenous peoples' rights | Indigenous Rights Engagement Verifier (F6) assesses engagement quality for indigenous rights compliance; engagement quality scores feed EUDR-016 country risk evaluation |
| **Art. 10(2)(e)** | Risk factor: information from consultations with indigenous peoples, local communities, and other stakeholders | **Primary article for this agent**. Consultation Record Manager (F4) captures all consultation data; Compliance Reporter (F7) generates Article 10(2)(e) evidence packages |
| **Art. 11(1)** | Adopt adequate and proportionate risk mitigation measures | Stakeholder engagement serves as a risk mitigation measure: community consultation, FPIC processes, and grievance mechanisms reduce supply chain risk. Integration with EUDR-029 Mitigation Measure Designer |
| **Art. 11(2)(a)** | Mitigation: additional information from suppliers | Consultation records provide additional supply chain information; community knowledge contributes to supply chain verification |
| **Art. 11(2)(c)** | Mitigation: other measures to manage and mitigate non-negligible risk | FPIC processes, community partnerships, benefit-sharing agreements, and grievance mechanisms are recognized mitigation measures |
| **Art. 12(2)** | DDS content requirements | Compliance Reporter (F7) generates stakeholder engagement sections for DDS; integration with EUDR-030 for DDS assembly |
| **Art. 29(1-3)** | Country benchmarking criteria including respect for indigenous peoples' rights | Engagement data (FPIC status, consultation outcomes, grievance patterns) contributes to country-level indigenous rights assessment via EUDR-016 |
| **Art. 31(1)** | Record keeping for 5 years | All stakeholder engagement records (consultation records, FPIC workflows, grievance cases, communications) retained for minimum 5 years with immutable audit trail |

### 5.2 ILO Convention 169 Requirements

| ILO 169 Article | Requirement | Agent Implementation |
|-----------------|-------------|---------------------|
| **Art. 6(1)(a)** | Consult indigenous peoples through appropriate procedures and representative institutions | Consultation Record Manager (F4) records consultation methodology and participant representation; Indigenous Rights Engagement Verifier (F6) validates cultural appropriateness |
| **Art. 6(2)** | Consultations shall be undertaken in good faith with the objective of achieving agreement or consent | FPIC Workflow Engine (F2) enforces good-faith consultation stages; Indigenous Rights Engagement Verifier (F6) assesses consultation genuineness |
| **Art. 7(1)** | Indigenous peoples shall participate in the formulation of development plans affecting them | Stakeholder Mapper (F1) identifies affected communities; FPIC Workflow Engine (F2) manages participation processes |
| **Art. 15(2)** | Consultation before exploration or exploitation of resources on indigenous lands | FPIC Workflow Engine (F2) ensures consultation occurs prior to commodity production activities |
| **Art. 16** | Indigenous peoples shall not be removed from lands they occupy; where relocation is necessary, FPIC is required | FPIC Workflow Engine (F2) manages consent for any land-affecting activities; Grievance Mechanism (F3) provides remedy channel for displacement concerns |

### 5.3 UN Declaration on the Rights of Indigenous Peoples (UNDRIP)

| UNDRIP Article | Requirement | Agent Implementation |
|----------------|-------------|---------------------|
| **Art. 10** | Indigenous peoples shall not be forcibly removed from their lands or territories; no relocation without FPIC | FPIC Workflow Engine (F2) with mandatory consent stage for territory-affecting activities |
| **Art. 19** | States shall consult and cooperate with indigenous peoples to obtain FPIC before adopting measures that may affect them | FPIC Workflow Engine (F2) manages the multi-stage consent process; Consultation Record Manager (F4) documents the process |
| **Art. 26** | Indigenous peoples have the right to lands, territories, and resources they have traditionally owned, occupied, or used | Stakeholder Mapper (F1) maps indigenous land rights; integration with EUDR-021 territory overlap data |
| **Art. 29** | Indigenous peoples have the right to conservation and protection of the environment and productive capacity of their lands | Indigenous Rights Engagement Verifier (F6) evaluates environmental impact engagement; Consultation Record Manager (F4) records environmental concerns |
| **Art. 32** | Indigenous peoples have the right to FPIC in relation to projects affecting their lands or territories | **Core FPIC requirement**. FPIC Workflow Engine (F2) implements complete Article 32 workflow |
| **Art. 40** | Indigenous peoples have the right to access effective mechanisms for prompt decision and remedies for violations | Grievance Mechanism (F3) provides accessible remedy channel per UNDRIP Article 40 |

### 5.4 CSDDD Requirements (Forward-Looking)

| CSDDD Article | Requirement | Agent Implementation |
|---------------|-------------|---------------------|
| **Art. 7** | Meaningful engagement with affected stakeholders in due diligence process | Full agent capability: Stakeholder Mapper, FPIC Workflow, Consultation Record Manager, Communication Hub |
| **Art. 8(1)** | Companies shall establish and maintain a complaints procedure | Grievance Mechanism (F3) provides CSDDD-compliant complaints procedure |
| **Art. 8(2)** | Complaints procedure accessible to persons and stakeholder groups who may be affected | Multi-channel intake (web, mobile, SMS, email, community-based); multi-language (12+); anonymous reporting |
| **Art. 8(3)** | Companies shall respond to complainants and seek resolution | Grievance Mechanism (F3) with acknowledgement SLA, investigation workflow, and resolution tracking |
| **Art. 9** | Remediation of adverse impacts | Grievance Mechanism (F3) remediation tracking; integration with EUDR-025 for remediation plan design |

### 5.5 UN Guiding Principles on Business and Human Rights

| UNGP Principle | Requirement | Agent Implementation |
|----------------|-------------|---------------------|
| **Principle 18** | Human rights due diligence should involve meaningful consultation with potentially affected groups | Consultation Record Manager (F4) and FPIC Workflow Engine (F2) provide structured consultation processes |
| **Principle 21** | Communication about how impacts are addressed | Communication Hub (F5) manages transparent communication about impact responses; Compliance Reporter (F7) generates public-facing engagement reports |
| **Principle 29** | Operational-level grievance mechanisms | Grievance Mechanism (F3) implements operational-level grievance mechanism |
| **Principle 31** | Effectiveness criteria: legitimate, accessible, predictable, equitable, transparent, rights-compatible, source of continuous learning | Grievance Mechanism (F3) designed against all 8 Principle 31 effectiveness criteria |

### 5.6 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Stakeholder engagement must verify community impact since cutoff date |
| June 29, 2023 | EUDR entered into force | Legal basis for stakeholder engagement obligations |
| December 30, 2025 | EUDR enforcement for large operators (ACTIVE) | Stakeholder engagement evidence required in DDS submissions |
| June 30, 2026 | EUDR enforcement for SMEs | SME onboarding wave; simplified engagement templates required |
| 2027 (phased) | CSDDD enforcement begins | Grievance mechanism mandatory; stakeholder engagement legally required |
| Ongoing (quarterly) | EC country benchmarking updates | Engagement data contributes to benchmarking evidence per Article 29(3)(c) |
| Ongoing (annually) | Article 8(3) due diligence system review | Annual stakeholder engagement effectiveness review |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 7 features below are P0 launch blockers. The agent cannot ship without all 7 features operational. Features 1-4 form the core stakeholder engagement engine; Features 5-7 form the communication, verification, and reporting layer.

**P0 Features 1-4: Core Stakeholder Engagement Engine**

---

#### Feature 1: Stakeholder Mapper

**User Story:**
```
As a compliance officer,
I want to map and categorize all stakeholders across my EUDR supply chain,
So that I can identify who is affected by my sourcing activities, what rights they hold,
and what engagement obligations I must fulfill per EUDR and CSDDD.
```

**Acceptance Criteria:**
- [ ] Registers stakeholders with typed categories: Indigenous Peoples, Local Communities, Smallholder Cooperatives, NGOs/CSOs, Workers/Unions, Government Authorities, Certification Bodies, Civil Society Monitors, Academic Institutions, Media
- [ ] Links stakeholders to supply chain nodes from EUDR-001 graph (producer, collector, processor, trader, importer)
- [ ] Links stakeholders to geographic locations with GPS coordinates and country/region classification
- [ ] Tracks rights classification per stakeholder: FPIC rights (ILO 169), customary land tenure, labor rights, environmental rights, community rights
- [ ] Records applicable legal protections per stakeholder: national legislation, ILO conventions, UNDRIP articles, certification scheme requirements
- [ ] Maintains engagement history timeline with all interactions, consultations, grievances, and communications
- [ ] Records preferred communication channels and language preferences per stakeholder
- [ ] Tracks relationship status: New, Active, Paused, Disputed, Resolved, Archived
- [ ] Supports automated stakeholder discovery from EUDR-001 supply chain graph (identify communities near production plots) and EUDR-021 territory overlap data (identify indigenous communities)
- [ ] Supports manual stakeholder registration with validation against existing records (deduplication)
- [ ] Generates stakeholder maps overlaid on supply chain visualization (geographic and graph views)
- [ ] Supports bulk import of stakeholder data from CSV, Excel, and JSON
- [ ] Handles 100,000+ stakeholder records without performance degradation

**Non-Functional Requirements:**
- Performance: Single stakeholder lookup < 100ms; full supply chain stakeholder query < 2 seconds for 10,000 stakeholders
- Data Quality: Automatic deduplication using fuzzy matching on name, location, and organization
- Security: Stakeholder personal data encrypted at rest (AES-256) per SEC-003; access controlled by RBAC per SEC-002
- Privacy: Supports data minimization and consent-based data collection per GDPR where applicable

**Dependencies:**
- EUDR-001 Supply Chain Mapping Master for supply chain graph data
- EUDR-021 Indigenous Rights Checker for indigenous territory data
- EUDR-022 Protected Area Validator for protected area community data
- SEC-002 RBAC for access control

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Stakeholder represents multiple categories (e.g., indigenous community that is also a smallholder cooperative) -- support multi-category assignment
- Stakeholder contact information changes -- maintain history of previous contact details
- Same community known by different names in different sources -- fuzzy match and merge with conflict resolution

---

#### Feature 2: FPIC Workflow Engine

**User Story:**
```
As an indigenous rights specialist,
I want a structured workflow to manage Free, Prior and Informed Consent processes
across all supply chain locations that intersect indigenous territories,
So that I can ensure FPIC compliance per ILO 169 and UNDRIP with documented evidence
at every stage.
```

**Acceptance Criteria:**
- [ ] Implements 7 mandatory FPIC stages with configurable transitions:
  - Stage 1: **Identification** -- Identify affected indigenous communities and their representative institutions
  - Stage 2: **Information Provision** -- Provide project information in accessible formats and local languages
  - Stage 3: **Deliberation Period** -- Allow community deliberation time (configurable: default 90 days, extendable to 180 days)
  - Stage 4: **Consultation** -- Conduct consultation through culturally appropriate mechanisms
  - Stage 5: **Consent/Objection** -- Record community decision with representative verification
  - Stage 6: **Agreement** -- Document formal agreement including benefit-sharing terms
  - Stage 7: **Monitoring** -- Ongoing monitoring of consent conditions and agreement compliance
- [ ] Enforces stage sequencing: each stage must be completed before the next can begin (with configurable override for authorized users)
- [ ] Configurable SLA timelines per stage with escalation rules and deadline notifications
- [ ] Evidence requirements per stage: mandatory document types, minimum evidence count, quality validation
- [ ] Approval gates between stages: configurable approver roles (Compliance Officer, Indigenous Rights Specialist, Legal Counsel)
- [ ] Community representative verification: validates that signatories are legitimate community representatives
- [ ] Benefit-sharing agreement management: tracks terms, payment schedules, compliance monitoring
- [ ] FPIC renewal management: tracks consent expiry dates and triggers renewal workflows
- [ ] Multi-community FPIC: supports workflows involving multiple communities for a single project area
- [ ] Integrates with EUDR-021 for FPIC documentation verification after workflow completion
- [ ] Generates FPIC compliance certificates upon successful workflow completion
- [ ] Complete audit trail of all stage transitions with timestamps, actors, evidence hashes

**FPIC Stage State Machine:**
```
IDENTIFICATION --> INFORMATION_PROVISION --> DELIBERATION --> CONSULTATION
    --> CONSENT_GRANTED --> AGREEMENT --> MONITORING --> [RENEWAL or CLOSED]
    --> CONSENT_WITHHELD --> OBJECTION_RECORDED --> [RENEGOTIATION or SUSPENDED]
```

**Non-Functional Requirements:**
- Performance: Stage transition < 500ms; workflow status query < 200ms
- Audit: Every state transition immutably recorded with SHA-256 hash
- Availability: FPIC deadlines enforced even during partial system outages (deadline stored in database, not dependent on scheduler uptime)
- Compliance: 100% of completed FPIC workflows produce documentation that passes EUDR-021 verification

**Dependencies:**
- EUDR-021 Indigenous Rights Checker for territory data and FPIC documentation verification
- Feature 1 (Stakeholder Mapper) for community identification
- Feature 4 (Consultation Record Manager) for consultation documentation

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 domain expert)

**Edge Cases:**
- Community withholds consent -- workflow transitions to OBJECTION_RECORDED with documented rationale; operator cannot proceed with activities in that area
- Community representative disputed by sub-group -- flag for mediation; pause workflow until representation resolved
- FPIC obtained but conditions violated -- trigger re-consent workflow with evidence of violation
- Multiple overlapping communities for same production area -- parallel FPIC workflows with aggregated consent requirement (all communities must consent)

---

#### Feature 3: Grievance Mechanism

**User Story:**
```
As a grievance officer,
I want a structured complaint management system accessible to all affected stakeholders,
So that I can receive, investigate, and resolve grievances transparently
while meeting UNGP Principle 31 effectiveness criteria and CSDDD Article 8 requirements.
```

**Acceptance Criteria:**
- [ ] Multi-channel intake: web portal form, mobile app submission, SMS hotline (short code), email inbox, community-based physical submission points (scanned paper forms)
- [ ] Anonymous reporting: complainants may submit without identifying themselves; system generates anonymous case reference
- [ ] Multi-language support: intake forms and communications available in 12+ languages (EN, FR, DE, ES, PT, ID, SW, AR, ZH, HI, TH, VI)
- [ ] Automatic acknowledgement: complainant receives acknowledgement within 48 hours of submission via their preferred channel
- [ ] Triage engine classifies complaints:
  - Severity: Critical (immediate human rights/safety risk), High (ongoing harm), Medium (potential harm), Low (inquiry/feedback)
  - Category: Environmental, Human Rights, Labor Rights, Land Rights, Community Impact, Process Complaint, Information Request
  - Urgency: Immediate (< 24 hours), Urgent (< 72 hours), Standard (< 30 days), Low (< 90 days)
- [ ] Investigation workflow: assign investigator, collect evidence, interview stakeholders, document findings, root cause analysis
- [ ] Resolution management: define remediation actions, track implementation, verify effectiveness, confirm with complainant
- [ ] Appeal process: complainant may appeal resolution within 30 days; appeal reviewed by independent reviewer
- [ ] Satisfaction assessment: complainant rates resolution satisfaction (1-5 scale) with optional comments
- [ ] Transparency dashboard: public-facing (anonymized) dashboard showing complaint volumes, categories, resolution rates, and average resolution times
- [ ] Escalation rules: automatic escalation if SLA breached (e.g., critical complaint not acknowledged in 24 hours escalates to VP Sustainability)
- [ ] Integration with EUDR-025 Risk Mitigation Advisor for grievance-informed mitigation measures
- [ ] Periodic reporting: monthly summary, quarterly trend analysis, annual grievance report for board and auditors

**Grievance Lifecycle State Machine:**
```
SUBMITTED --> ACKNOWLEDGED --> TRIAGED --> INVESTIGATION_ASSIGNED
    --> UNDER_INVESTIGATION --> FINDINGS_DOCUMENTED --> RESOLUTION_PROPOSED
    --> RESOLUTION_ACCEPTED --> REMEDIATION_IN_PROGRESS --> REMEDIATION_VERIFIED
    --> CLOSED
    --> RESOLUTION_REJECTED --> APPEAL_SUBMITTED --> APPEAL_REVIEW --> [REVISED_RESOLUTION or CLOSED]
```

**UNGP Principle 31 Effectiveness Criteria Mapping:**
| Criterion | Implementation |
|-----------|----------------|
| (a) Legitimate | Independent oversight option; transparent governance of grievance process |
| (b) Accessible | Multi-channel, multi-language, anonymous reporting, no cost to complainant |
| (c) Predictable | Published SLA timelines; predictable process stages; regular status updates |
| (d) Equitable | Impartial investigation; complainant has right to be heard and to appeal |
| (e) Transparent | Anonymized public dashboard; complainant receives all findings; annual public report |
| (f) Rights-compatible | Outcomes aligned with international human rights standards; does not preclude legal remedy |
| (g) Source of continuous learning | Trend analysis feeds into risk assessment; root cause patterns inform mitigation strategy |
| (h) Based on engagement and dialogue | Resolution process involves complainant participation; satisfaction assessment |

**Non-Functional Requirements:**
- Availability: Grievance intake channels available 99.9% uptime
- Performance: Complaint submission < 3 seconds to acknowledgement generation
- Security: Complainant identity protected with encryption; anonymous complaints cannot be de-anonymized
- Scalability: Handle 10,000+ active grievances simultaneously
- Retention: All grievance records retained for minimum 5 years per EUDR Article 31

**Dependencies:**
- Communication Hub (Feature 5) for multi-channel acknowledgement and status updates
- Feature 1 (Stakeholder Mapper) for complainant identification (where not anonymous)
- EUDR-025 Risk Mitigation Advisor for grievance-informed mitigation

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 frontend engineer)

---

#### Feature 4: Consultation Record Manager

**User Story:**
```
As a community liaison officer,
I want to document all consultations with indigenous peoples and local communities
using a structured format that captures objectives, participants, discussions, and outcomes,
So that I can provide audit-ready evidence of stakeholder consultation per EUDR Article 10(2)(e).
```

**Acceptance Criteria:**
- [ ] Structured consultation record with mandatory fields: consultation ID, date/time, location (GPS coordinates), type (community meeting, focus group, bilateral negotiation, public hearing, field visit, survey), objectives, participants (name, role, affiliation, community represented), methodology description
- [ ] Discussion topics tracking with structured topic-outcome mapping
- [ ] Outcome documentation: decisions made, commitments by each party, conditions and caveats, dissenting views
- [ ] Follow-up action tracking: action description, responsible party, deadline, status (pending/in-progress/completed/overdue)
- [ ] Evidence attachment: meeting minutes (text), photos (with GPS metadata), audio recordings (with participant consent), signed attendance sheets, translated materials
- [ ] Participant consent management: record whether participants consent to documentation, audio recording, and photo capture
- [ ] Offline data entry: mobile-optimized form that works without internet connectivity; local storage with automatic sync when connectivity restored
- [ ] Multi-language support: consultation forms and templates available in 12+ languages
- [ ] Template library: 20+ pre-built consultation templates for common scenarios (FPIC consultation, community needs assessment, benefit-sharing negotiation, grievance mediation, annual review meeting, environmental impact discussion)
- [ ] Immutable finalization: once a consultation record is finalized, it cannot be edited; amendments create linked addendum records
- [ ] SHA-256 provenance hash on every finalized consultation record
- [ ] Search and filter: find consultations by date range, location, community, topic, outcome, stakeholder
- [ ] Cross-reference with FPIC workflows: consultations automatically linked to relevant FPIC workflow stages

**Non-Functional Requirements:**
- Offline: Full functionality on mobile devices without internet for up to 7 days; data integrity maintained through conflict-free sync
- Performance: Record creation < 500ms; search results < 1 second for 100,000+ records
- Storage: Support for attachments up to 50 MB per consultation record (photos, audio)
- Auditability: Complete immutable audit trail of all record creation, finalization, and addendum activities

**Dependencies:**
- Feature 1 (Stakeholder Mapper) for participant identification
- Feature 2 (FPIC Workflow Engine) for FPIC-linked consultations
- S3/Object Storage (INFRA-004) for evidence file storage

**Estimated Effort:** 3 weeks (1 backend engineer, 1 mobile/frontend engineer)

---

**P0 Features 5-7: Communication, Verification, and Reporting Layer**

> Features 5, 6, and 7 are P0 launch blockers. Without communications management, engagement quality verification, and compliance reporting, the core engagement engine cannot deliver auditable value. These features are the delivery mechanism through which compliance officers, auditors, and regulators interact with the engagement data.

---

#### Feature 5: Communication Hub

**User Story:**
```
As a compliance officer,
I want to manage all stakeholder communications through a centralized platform
that supports multiple channels and languages,
So that I can ensure consistent, trackable, and inclusive communication
with all affected stakeholders across my supply chain.
```

**Acceptance Criteria:**
- [ ] Multi-channel dispatch: email (SMTP/API), SMS (Twilio/API), WhatsApp Business API, portal notifications, printed correspondence generation (PDF), community radio script generation
- [ ] Stakeholder segmentation: create communication groups based on stakeholder type, location, language, engagement status, supply chain node, or custom criteria
- [ ] Template library: 100+ pre-built templates for common engagement scenarios:
  - FPIC notifications (initial contact, information provision, consultation invitation, consent request, agreement terms, renewal notice)
  - Grievance communications (acknowledgement, investigation update, resolution proposal, appeal notice, closure confirmation)
  - Consultation invitations and follow-ups
  - Compliance updates and reporting
  - Benefit-sharing communications
- [ ] Multi-language support: templates available in 12+ languages; automatic language selection based on stakeholder preference
- [ ] Communication scheduling: plan and schedule communications with date/time targeting
- [ ] Delivery tracking: confirmation of delivery, open/read receipts (where channel supports), bounce/failure handling
- [ ] Response management: capture and route stakeholder responses to appropriate workflows (grievance, consultation, FPIC)
- [ ] Campaign management: coordinate multi-stakeholder, multi-channel communication campaigns with progress tracking
- [ ] Communication audit trail: every communication logged with sender, recipient, channel, content hash, delivery status, and timestamp
- [ ] Rate limiting and compliance: respect communication frequency preferences; unsubscribe management where applicable

**Non-Functional Requirements:**
- Throughput: 10,000 messages per minute across all channels
- Delivery: >= 95% successful delivery rate across all channels
- Latency: Message dispatch < 5 seconds from send request
- Security: Communication content encrypted in transit (TLS 1.3) and at rest (AES-256)

**Dependencies:**
- Feature 1 (Stakeholder Mapper) for recipient identification and preferences
- Email notification service (existing GreenLang infrastructure)
- SMS gateway (Twilio or equivalent)
- WhatsApp Business API (optional, degraded gracefully if unavailable)

**Estimated Effort:** 3 weeks (1 backend engineer, 1 integration engineer)

---

#### Feature 6: Indigenous Rights Engagement Verifier

**User Story:**
```
As an indigenous rights specialist,
I want to verify that our engagement with indigenous communities meets
the substantive requirements of ILO Convention 169 and UNDRIP,
So that I can demonstrate to auditors and regulators that our engagement
is meaningful, culturally appropriate, and rights-respecting.
```

**Acceptance Criteria:**
- [ ] Evaluates engagement quality across 6 dimensions with deterministic scoring (0-100 per dimension, 0-100 aggregate):
  - (a) **Cultural Appropriateness** (0-100): Engagement methods respect community customs, protocols, and governance structures. Scored based on: use of community-designated meeting spaces, respect for community calendar/schedule, involvement of community-recognized leaders, use of culturally appropriate communication materials.
  - (b) **Language Accessibility** (0-100): Information provided in languages understood by the community. Scored based on: materials translated to local languages, interpreters provided during consultations, information conveyed through multiple formats (written, oral, visual).
  - (c) **Deliberation Time Adequacy** (0-100): Sufficient time provided for community decision-making. Scored based on: actual deliberation time vs. minimum required by ILO 169 and national legislation; time proportionate to complexity of decision.
  - (d) **Representation Inclusiveness** (0-100): All relevant community sub-groups represented in engagement. Scored based on: women's participation rate, youth representation, elder participation, sub-clan/sub-community coverage, representation of marginalized groups within community.
  - (e) **Consultation Genuineness** (0-100): Consultation was substantive, not merely informational. Scored based on: evidence of two-way dialogue, community input incorporated into decisions, modifications made based on community feedback, community veto right respected.
  - (f) **Decision Respect** (0-100): Community decision-making processes respected. Scored based on: consent/objection recorded through community-recognized decision mechanisms, no evidence of coercion or undue influence, community timeline for decision respected.
- [ ] Aggregate engagement quality score: weighted average of 6 dimensions (configurable weights; default equal weighting)
- [ ] Quality classification: Exemplary (90-100), Good (75-89), Adequate (60-74), Insufficient (40-59), Non-Compliant (0-39)
- [ ] Generates dimension-level improvement recommendations for scores below "Good" threshold
- [ ] Compares engagement quality across supply chain locations for portfolio-level assessment
- [ ] Integrates with EUDR-021 for indigenous rights context (territory data, legal framework)
- [ ] All scoring deterministic and reproducible (no LLM in scoring path)
- [ ] Provenance hash on every assessment

**Scoring Formula:**
```
Engagement_Quality_Score = sum(
    Dimension_Score[i] * Dimension_Weight[i]
    for i in [cultural, language, deliberation, representation, genuineness, decision]
)

Where:
- Default weights: all 1/6 (equal weighting)
- Configurable per operator based on commodity and country context
- Dimension scores calculated from objective evidence metrics using Decimal arithmetic
```

**Non-Functional Requirements:**
- Performance: Assessment calculation < 200ms per engagement
- Determinism: Bit-perfect reproducibility across runs
- Auditability: Scoring methodology transparent and documented; every score traceable to input evidence

**Dependencies:**
- Feature 2 (FPIC Workflow Engine) for FPIC process data
- Feature 4 (Consultation Record Manager) for consultation evidence
- EUDR-021 Indigenous Rights Checker for territory and rights framework data

**Estimated Effort:** 2 weeks (1 senior backend engineer)

---

#### Feature 7: Compliance Reporter

**User Story:**
```
As a compliance officer,
I want to generate audit-ready stakeholder engagement reports
for DDS submission, competent authority inspection, and certification audits,
So that I can demonstrate comprehensive, documented stakeholder engagement
per EUDR Articles 10, 11, and 12 requirements.
```

**Acceptance Criteria:**
- [ ] Generates 7 report types:
  - (a) **Stakeholder Engagement Summary for DDS** -- Overview of all engagement activities for inclusion in Due Diligence Statement; maps to Article 12(2) content requirements
  - (b) **FPIC Compliance Report** -- Detailed report on all FPIC workflows including stage progression, evidence summaries, consent status, and agreement terms; for Article 10(2)(d-e) compliance
  - (c) **Grievance Mechanism Annual Report** -- Annual summary of grievance intake, processing, resolution, and trends; for CSDDD Article 8 and UNGP Principle 31 compliance
  - (d) **Consultation Register** -- Complete log of all stakeholder consultations with outcomes and follow-up status; for Article 10(2)(e) evidence
  - (e) **Indigenous Rights Engagement Assessment** -- Engagement quality scores with dimension-level analysis and improvement recommendations; for Article 29(3)(c) evidence
  - (f) **Communication Activity Log** -- Complete record of all stakeholder communications with delivery status; for Article 8(2) evidence of due diligence system
  - (g) **Stakeholder Engagement Effectiveness Report** -- Trend analysis of engagement metrics, relationship health, and impact on risk reduction; for Article 8(3) annual review
- [ ] Each report includes: executive summary, detailed findings, evidence references, regulatory article mapping, provenance chain, generation metadata
- [ ] SHA-256 provenance hash on every generated report
- [ ] Regulatory cross-reference: every data point in the report linked to its source record and applicable EUDR Article
- [ ] Supports filtering by: commodity, country, supply chain segment, date range, stakeholder type, engagement type
- [ ] Formats: PDF (human-readable), JSON (machine-readable for EUDR-030 integration), HTML (web display), XLSX (tabular data export)
- [ ] Multi-language report generation: EN, FR, DE, ES, PT, ID, SW
- [ ] Batch report generation for portfolio-level reporting (multiple commodities/countries)
- [ ] Integrates with EUDR-030 Documentation Generator for DDS package inclusion

**Non-Functional Requirements:**
- Performance: < 15 seconds per report (JSON); < 30 seconds (PDF); < 60 seconds for portfolio-level batch
- Completeness: All mandatory fields populated when upstream data available; missing fields clearly flagged
- Integrity: Reports immutable after generation; amendments create new versions

**Dependencies:**
- Features 1-6 (all core engagement features provide data for reports)
- EUDR-030 Documentation Generator for DDS integration
- S3/Object Storage (INFRA-004) for report storage and retrieval

**Estimated Effort:** 3 weeks (1 backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 8: Stakeholder Sentiment Analysis
- Track stakeholder sentiment trends over time using structured feedback data (not NLP/LLM-based for determinism)
- Aggregate sentiment scores from consultation outcomes, grievance patterns, and satisfaction ratings
- Generate sentiment dashboards per stakeholder group, region, and commodity
- Trigger alerts when sentiment drops below configurable threshold

#### Feature 9: Benefit-Sharing Agreement Manager
- Structured management of benefit-sharing agreements with indigenous communities and local communities
- Track payment schedules, in-kind contributions, capacity building commitments, and monitoring obligations
- Generate compliance reports on agreement fulfillment
- Integration with financial systems for payment verification

#### Feature 10: Community Impact Assessment
- Structured framework for assessing positive and negative impacts of commodity production on communities
- Track impact indicators (economic, social, environmental, cultural) over time
- Generate impact assessment reports for sustainability reporting and CSDDD compliance
- Compare impact across supply chain locations

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- AI/NLP-based sentiment analysis from unstructured text (defer to v2.0; v1.0 uses structured feedback only)
- Direct payment processing for benefit-sharing agreements (integrate with external payment systems)
- Legal mediation or arbitration services for disputes
- Real-time video conferencing for remote consultations (integrate with external platforms)
- Mobile native application (responsive web with offline PWA for v1.0)
- Social media monitoring for stakeholder sentiment (defer to v2.0)
- Predictive analytics for stakeholder relationship risk (defer to v2.0)

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
| AGENT-EUDR-031        |           | AGENT-EUDR-021            |           | AGENT-EUDR-030        |
| Stakeholder Engagement|<--------->| Indigenous Rights          |<--------->| Documentation         |
| Tool                  |           | Checker                   |           | Generator             |
|                       |           |                           |           |                       |
| - Stakeholder Mapper  |           | - Territory Database      |           | - DDS Generation      |
| - FPIC Workflow Engine|           | - FPIC Doc Verification   |           | - Package Builder     |
| - Grievance Mechanism |           | - Overlap Detector        |           | - Submission Engine    |
| - Consultation Mgr    |           | - Community Registry      |           |                       |
| - Communication Hub   |           | - Violation Alerts        |           |                       |
| - Engagement Verifier |           |                           |           |                       |
| - Compliance Reporter |           |                           |           |                       |
+-----------+-----------+           +---------------------------+           +-----------------------+
            |
+-----------v-----------+           +---------------------------+           +---------------------------+
| AGENT-EUDR-001        |           | AGENT-EUDR-025            |           | AGENT-EUDR-026            |
| Supply Chain Mapping  |           | Risk Mitigation           |           | Due Diligence             |
| Master                |           | Advisor                   |           | Orchestrator              |
|                       |           |                           |           |                           |
| - Graph Engine        |           | - Stakeholder Collab Hub  |           | - Workflow Coordinator    |
| - Node/Edge data      |           | - Remediation Plans       |           | - Quality Gates           |
+-----------------------+           +---------------------------+           +---------------------------+
            |
+-----------v-----------+           +---------------------------+
| AGENT-EUDR-022        |           | AGENT-EUDR-028            |
| Protected Area        |           | Risk Assessment           |
| Validator             |           | Engine                    |
|                       |           |                           |
| - Community data      |           | - Engagement data input   |
+-----------------------+           +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/stakeholder_engagement/
    __init__.py                          # Public API exports
    config.py                            # StakeholderEngagementConfig with GL_EUDR_SET_ env prefix
    models.py                            # Pydantic v2 models for stakeholders, FPIC, grievances, consultations
    stakeholder_mapper.py                # StakeholderMapperEngine: registry management and discovery
    fpic_workflow.py                     # FPICWorkflowEngine: 7-stage FPIC process management
    grievance_mechanism.py               # GrievanceMechanismEngine: complaint lifecycle management
    consultation_manager.py              # ConsultationRecordManager: consultation documentation
    communication_hub.py                 # CommunicationHub: multi-channel communication dispatch
    engagement_verifier.py               # EngagementVerifier: indigenous rights engagement scoring
    compliance_reporter.py               # ComplianceReporter: audit-ready report generation
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains
    metrics.py                           # 15 Prometheus self-monitoring metrics
    setup.py                             # StakeholderEngagementService facade
    api/
        __init__.py
        router.py                        # FastAPI router (30+ endpoints)
        stakeholder_routes.py            # Stakeholder registry CRUD and query endpoints
        fpic_routes.py                   # FPIC workflow management endpoints
        grievance_routes.py              # Grievance mechanism endpoints
        consultation_routes.py           # Consultation record endpoints
        communication_routes.py          # Communication dispatch and tracking endpoints
        engagement_routes.py             # Engagement verification endpoints
        report_routes.py                 # Compliance report generation endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Stakeholder Types
class StakeholderType(str, Enum):
    INDIGENOUS_PEOPLES = "indigenous_peoples"
    LOCAL_COMMUNITY = "local_community"
    SMALLHOLDER_COOPERATIVE = "smallholder_cooperative"
    NGO_CSO = "ngo_cso"
    WORKERS_UNION = "workers_union"
    GOVERNMENT_AUTHORITY = "government_authority"
    CERTIFICATION_BODY = "certification_body"
    CIVIL_SOCIETY_MONITOR = "civil_society_monitor"
    ACADEMIC_INSTITUTION = "academic_institution"
    MEDIA = "media"

# Stakeholder Rights Classification
class RightsClassification(str, Enum):
    FPIC_RIGHTS = "fpic_rights"           # ILO 169, UNDRIP
    CUSTOMARY_LAND_TENURE = "customary_land_tenure"
    LABOR_RIGHTS = "labor_rights"
    ENVIRONMENTAL_RIGHTS = "environmental_rights"
    COMMUNITY_RIGHTS = "community_rights"

# Relationship Status
class RelationshipStatus(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    PAUSED = "paused"
    DISPUTED = "disputed"
    RESOLVED = "resolved"
    ARCHIVED = "archived"

# Stakeholder Record
class StakeholderRecord(BaseModel):
    stakeholder_id: str                   # UUID
    stakeholder_type: StakeholderType
    name: str
    organization: Optional[str]
    country_code: str                     # ISO 3166-1 alpha-2
    region: Optional[str]
    coordinates: Optional[Tuple[float, float]]  # (lat, lon)
    supply_chain_node_ids: List[str]      # Linked EUDR-001 graph nodes
    territory_ids: List[str]              # Linked EUDR-021 territory IDs
    rights_classifications: List[RightsClassification]
    legal_protections: List[str]          # e.g., "ILO_169", "UNDRIP_Art32", "BR_Law_6040"
    preferred_language: str               # ISO 639-1
    preferred_channels: List[str]         # email, sms, whatsapp, portal, in_person
    relationship_status: RelationshipStatus
    engagement_history_count: int
    last_engagement_date: Optional[datetime]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# FPIC Workflow Stage
class FPICStage(str, Enum):
    IDENTIFICATION = "identification"
    INFORMATION_PROVISION = "information_provision"
    DELIBERATION = "deliberation"
    CONSULTATION = "consultation"
    CONSENT_GRANTED = "consent_granted"
    CONSENT_WITHHELD = "consent_withheld"
    AGREEMENT = "agreement"
    MONITORING = "monitoring"
    RENEWAL = "renewal"
    SUSPENDED = "suspended"
    CLOSED = "closed"

# FPIC Workflow Record
class FPICWorkflow(BaseModel):
    workflow_id: str                       # UUID
    operator_id: str
    stakeholder_ids: List[str]            # Affected communities
    supply_chain_node_ids: List[str]      # Relevant supply chain nodes
    territory_id: Optional[str]           # EUDR-021 territory reference
    commodity: str                        # EUDR commodity
    current_stage: FPICStage
    stage_history: List[Dict[str, Any]]   # All stage transitions
    deliberation_days_allowed: int         # Default 90
    deliberation_start_date: Optional[datetime]
    deliberation_end_date: Optional[datetime]
    consent_status: Optional[str]          # granted/withheld/conditional
    agreement_terms: Optional[Dict[str, Any]]
    benefit_sharing: Optional[Dict[str, Any]]
    evidence_ids: List[str]               # Document/evidence references
    provenance_hash: str                  # SHA-256
    created_at: datetime
    updated_at: datetime

# Grievance Severity
class GrievanceSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Grievance Category
class GrievanceCategory(str, Enum):
    ENVIRONMENTAL = "environmental"
    HUMAN_RIGHTS = "human_rights"
    LABOR_RIGHTS = "labor_rights"
    LAND_RIGHTS = "land_rights"
    COMMUNITY_IMPACT = "community_impact"
    PROCESS_COMPLAINT = "process_complaint"
    INFORMATION_REQUEST = "information_request"

# Grievance Status
class GrievanceStatus(str, Enum):
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    TRIAGED = "triaged"
    INVESTIGATION_ASSIGNED = "investigation_assigned"
    UNDER_INVESTIGATION = "under_investigation"
    FINDINGS_DOCUMENTED = "findings_documented"
    RESOLUTION_PROPOSED = "resolution_proposed"
    RESOLUTION_ACCEPTED = "resolution_accepted"
    RESOLUTION_REJECTED = "resolution_rejected"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    REMEDIATION_VERIFIED = "remediation_verified"
    APPEAL_SUBMITTED = "appeal_submitted"
    APPEAL_REVIEW = "appeal_review"
    CLOSED = "closed"

# Grievance Record
class GrievanceRecord(BaseModel):
    grievance_id: str                      # UUID
    case_reference: str                    # Human-readable reference (e.g., GRV-2026-00123)
    is_anonymous: bool
    complainant_id: Optional[str]          # Stakeholder ID (null if anonymous)
    complainant_channel: str               # Intake channel used
    language: str                          # Submission language
    severity: GrievanceSeverity
    category: GrievanceCategory
    description: str
    location_coordinates: Optional[Tuple[float, float]]
    supply_chain_node_ids: List[str]
    current_status: GrievanceStatus
    status_history: List[Dict[str, Any]]
    assigned_investigator: Optional[str]
    investigation_findings: Optional[str]
    resolution_description: Optional[str]
    remediation_actions: List[Dict[str, Any]]
    satisfaction_rating: Optional[int]     # 1-5
    appeal_record: Optional[Dict[str, Any]]
    sla_deadline: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]
    evidence_ids: List[str]
    provenance_hash: str
    created_at: datetime
    updated_at: datetime

# Consultation Record
class ConsultationRecord(BaseModel):
    consultation_id: str                   # UUID
    consultation_type: str                 # community_meeting, focus_group, bilateral, public_hearing, field_visit, survey
    fpic_workflow_id: Optional[str]        # Link to FPIC workflow if applicable
    date: datetime
    location: str
    location_coordinates: Optional[Tuple[float, float]]
    objectives: List[str]
    participants: List[Dict[str, Any]]     # name, role, affiliation, community
    methodology: str
    topics_discussed: List[Dict[str, Any]]
    outcomes: List[Dict[str, Any]]
    commitments: List[Dict[str, Any]]      # party, commitment, deadline
    follow_up_actions: List[Dict[str, Any]]
    evidence_ids: List[str]                # Attachments: minutes, photos, audio
    participant_consent_recorded: bool
    is_finalized: bool
    finalized_at: Optional[datetime]
    provenance_hash: str
    created_at: datetime
    updated_at: datetime
```

### 7.4 Database Schema (New Migration: V119)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_stakeholder_engagement;

-- Stakeholder registry
CREATE TABLE eudr_stakeholder_engagement.stakeholders (
    stakeholder_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stakeholder_type VARCHAR(50) NOT NULL,
    name VARCHAR(500) NOT NULL,
    organization VARCHAR(500),
    country_code CHAR(2) NOT NULL,
    region VARCHAR(200),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    supply_chain_node_ids JSONB DEFAULT '[]',
    territory_ids JSONB DEFAULT '[]',
    rights_classifications JSONB DEFAULT '[]',
    legal_protections JSONB DEFAULT '[]',
    preferred_language VARCHAR(10) DEFAULT 'en',
    preferred_channels JSONB DEFAULT '["email"]',
    relationship_status VARCHAR(30) DEFAULT 'new',
    engagement_history_count INTEGER DEFAULT 0,
    last_engagement_date TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- FPIC workflows
CREATE TABLE eudr_stakeholder_engagement.fpic_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    stakeholder_ids JSONB NOT NULL DEFAULT '[]',
    supply_chain_node_ids JSONB DEFAULT '[]',
    territory_id UUID,
    commodity VARCHAR(50) NOT NULL,
    current_stage VARCHAR(50) NOT NULL DEFAULT 'identification',
    stage_history JSONB DEFAULT '[]',
    deliberation_days_allowed INTEGER DEFAULT 90,
    deliberation_start_date TIMESTAMPTZ,
    deliberation_end_date TIMESTAMPTZ,
    consent_status VARCHAR(30),
    agreement_terms JSONB,
    benefit_sharing JSONB,
    evidence_ids JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

-- FPIC stage transitions (hypertable for audit trail)
CREATE TABLE eudr_stakeholder_engagement.fpic_stage_transitions (
    transition_id UUID DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    from_stage VARCHAR(50),
    to_stage VARCHAR(50) NOT NULL,
    transitioned_by VARCHAR(100) NOT NULL,
    evidence_ids JSONB DEFAULT '[]',
    notes TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_stakeholder_engagement.fpic_stage_transitions', 'transitioned_at');

-- Grievances
CREATE TABLE eudr_stakeholder_engagement.grievances (
    grievance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_reference VARCHAR(50) NOT NULL UNIQUE,
    is_anonymous BOOLEAN DEFAULT FALSE,
    complainant_id UUID,
    complainant_channel VARCHAR(50) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    location_latitude DOUBLE PRECISION,
    location_longitude DOUBLE PRECISION,
    supply_chain_node_ids JSONB DEFAULT '[]',
    current_status VARCHAR(50) NOT NULL DEFAULT 'submitted',
    assigned_investigator VARCHAR(100),
    investigation_findings TEXT,
    resolution_description TEXT,
    remediation_actions JSONB DEFAULT '[]',
    satisfaction_rating SMALLINT,
    appeal_record JSONB,
    sla_deadline TIMESTAMPTZ NOT NULL,
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    evidence_ids JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Grievance status transitions (hypertable)
CREATE TABLE eudr_stakeholder_engagement.grievance_status_transitions (
    transition_id UUID DEFAULT gen_random_uuid(),
    grievance_id UUID NOT NULL,
    from_status VARCHAR(50),
    to_status VARCHAR(50) NOT NULL,
    transitioned_by VARCHAR(100) NOT NULL,
    notes TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_stakeholder_engagement.grievance_status_transitions', 'transitioned_at');

-- Consultation records
CREATE TABLE eudr_stakeholder_engagement.consultations (
    consultation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    consultation_type VARCHAR(50) NOT NULL,
    fpic_workflow_id UUID,
    consultation_date TIMESTAMPTZ NOT NULL,
    location VARCHAR(500) NOT NULL,
    location_latitude DOUBLE PRECISION,
    location_longitude DOUBLE PRECISION,
    objectives JSONB DEFAULT '[]',
    participants JSONB DEFAULT '[]',
    methodology TEXT,
    topics_discussed JSONB DEFAULT '[]',
    outcomes JSONB DEFAULT '[]',
    commitments JSONB DEFAULT '[]',
    follow_up_actions JSONB DEFAULT '[]',
    evidence_ids JSONB DEFAULT '[]',
    participant_consent_recorded BOOLEAN DEFAULT FALSE,
    is_finalized BOOLEAN DEFAULT FALSE,
    finalized_at TIMESTAMPTZ,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Communication log (hypertable)
CREATE TABLE eudr_stakeholder_engagement.communications (
    communication_id UUID DEFAULT gen_random_uuid(),
    stakeholder_id UUID NOT NULL,
    channel VARCHAR(30) NOT NULL,
    direction VARCHAR(10) NOT NULL DEFAULT 'outbound',
    template_id VARCHAR(100),
    subject VARCHAR(500),
    content_hash VARCHAR(64) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    delivery_status VARCHAR(30) DEFAULT 'pending',
    delivered_at TIMESTAMPTZ,
    read_at TIMESTAMPTZ,
    response_received BOOLEAN DEFAULT FALSE,
    campaign_id UUID,
    metadata JSONB DEFAULT '{}',
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_stakeholder_engagement.communications', 'sent_at');

-- Engagement quality assessments
CREATE TABLE eudr_stakeholder_engagement.engagement_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stakeholder_id UUID NOT NULL,
    supply_chain_node_id UUID,
    fpic_workflow_id UUID,
    cultural_score NUMERIC(5,2) DEFAULT 0.0,
    language_score NUMERIC(5,2) DEFAULT 0.0,
    deliberation_score NUMERIC(5,2) DEFAULT 0.0,
    representation_score NUMERIC(5,2) DEFAULT 0.0,
    genuineness_score NUMERIC(5,2) DEFAULT 0.0,
    decision_respect_score NUMERIC(5,2) DEFAULT 0.0,
    aggregate_score NUMERIC(5,2) DEFAULT 0.0,
    quality_classification VARCHAR(30),
    recommendations JSONB DEFAULT '[]',
    evidence_ids JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Compliance reports generated (hypertable)
CREATE TABLE eudr_stakeholder_engagement.compliance_reports (
    report_id UUID DEFAULT gen_random_uuid(),
    report_type VARCHAR(100) NOT NULL,
    operator_id UUID NOT NULL,
    commodity VARCHAR(50),
    country_code CHAR(2),
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,
    format VARCHAR(10) NOT NULL DEFAULT 'json',
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_stakeholder_engagement.compliance_reports', 'generated_at');

-- Indexes
CREATE INDEX idx_stakeholders_type ON eudr_stakeholder_engagement.stakeholders(stakeholder_type);
CREATE INDEX idx_stakeholders_country ON eudr_stakeholder_engagement.stakeholders(country_code);
CREATE INDEX idx_stakeholders_status ON eudr_stakeholder_engagement.stakeholders(relationship_status);
CREATE INDEX idx_stakeholders_name_trgm ON eudr_stakeholder_engagement.stakeholders USING gin(name gin_trgm_ops);
CREATE INDEX idx_fpic_operator ON eudr_stakeholder_engagement.fpic_workflows(operator_id);
CREATE INDEX idx_fpic_stage ON eudr_stakeholder_engagement.fpic_workflows(current_stage);
CREATE INDEX idx_fpic_commodity ON eudr_stakeholder_engagement.fpic_workflows(commodity);
CREATE INDEX idx_grievances_status ON eudr_stakeholder_engagement.grievances(current_status);
CREATE INDEX idx_grievances_severity ON eudr_stakeholder_engagement.grievances(severity);
CREATE INDEX idx_grievances_category ON eudr_stakeholder_engagement.grievances(category);
CREATE INDEX idx_grievances_case_ref ON eudr_stakeholder_engagement.grievances(case_reference);
CREATE INDEX idx_consultations_type ON eudr_stakeholder_engagement.consultations(consultation_type);
CREATE INDEX idx_consultations_date ON eudr_stakeholder_engagement.consultations(consultation_date);
CREATE INDEX idx_consultations_fpic ON eudr_stakeholder_engagement.consultations(fpic_workflow_id);
CREATE INDEX idx_assessments_stakeholder ON eudr_stakeholder_engagement.engagement_assessments(stakeholder_id);
CREATE INDEX idx_assessments_quality ON eudr_stakeholder_engagement.engagement_assessments(quality_classification);
```

### 7.5 API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Stakeholder Registry** | | |
| POST | `/v1/stakeholders` | Register a new stakeholder |
| GET | `/v1/stakeholders` | List stakeholders (with filters: type, country, status, supply chain node) |
| GET | `/v1/stakeholders/{stakeholder_id}` | Get stakeholder details with engagement history |
| PUT | `/v1/stakeholders/{stakeholder_id}` | Update stakeholder record |
| DELETE | `/v1/stakeholders/{stakeholder_id}` | Archive a stakeholder record |
| POST | `/v1/stakeholders/discover` | Trigger automated stakeholder discovery from supply chain graph |
| POST | `/v1/stakeholders/import` | Bulk import stakeholders from CSV/Excel/JSON |
| GET | `/v1/stakeholders/{stakeholder_id}/timeline` | Get engagement timeline for a stakeholder |
| **FPIC Workflows** | | |
| POST | `/v1/fpic/workflows` | Create a new FPIC workflow |
| GET | `/v1/fpic/workflows` | List FPIC workflows (with filters: stage, commodity, stakeholder) |
| GET | `/v1/fpic/workflows/{workflow_id}` | Get FPIC workflow details with stage history |
| POST | `/v1/fpic/workflows/{workflow_id}/transition` | Advance FPIC workflow to next stage |
| POST | `/v1/fpic/workflows/{workflow_id}/evidence` | Upload evidence for current FPIC stage |
| GET | `/v1/fpic/workflows/{workflow_id}/certificate` | Generate FPIC compliance certificate |
| **Grievance Mechanism** | | |
| POST | `/v1/grievances` | Submit a new grievance (supports anonymous) |
| GET | `/v1/grievances` | List grievances (with filters: status, severity, category) |
| GET | `/v1/grievances/{grievance_id}` | Get grievance details with status history |
| PUT | `/v1/grievances/{grievance_id}/triage` | Triage a grievance (assign severity, category, urgency) |
| PUT | `/v1/grievances/{grievance_id}/assign` | Assign investigator to grievance |
| PUT | `/v1/grievances/{grievance_id}/findings` | Record investigation findings |
| PUT | `/v1/grievances/{grievance_id}/resolution` | Propose resolution |
| PUT | `/v1/grievances/{grievance_id}/close` | Close a grievance |
| POST | `/v1/grievances/{grievance_id}/appeal` | Submit an appeal |
| GET | `/v1/grievances/dashboard` | Get anonymized grievance dashboard data |
| **Consultation Records** | | |
| POST | `/v1/consultations` | Create a new consultation record |
| GET | `/v1/consultations` | List consultations (with filters: type, date range, community, topic) |
| GET | `/v1/consultations/{consultation_id}` | Get consultation details |
| PUT | `/v1/consultations/{consultation_id}` | Update consultation record (before finalization) |
| POST | `/v1/consultations/{consultation_id}/finalize` | Finalize and lock consultation record |
| POST | `/v1/consultations/{consultation_id}/addendum` | Add addendum to finalized record |
| POST | `/v1/consultations/{consultation_id}/evidence` | Upload evidence (photos, audio, documents) |
| **Communications** | | |
| POST | `/v1/communications/send` | Send communication to stakeholder(s) |
| POST | `/v1/communications/campaign` | Create communication campaign |
| GET | `/v1/communications` | List communications (with filters: stakeholder, channel, status) |
| GET | `/v1/communications/{communication_id}/status` | Get delivery status |
| **Engagement Verification** | | |
| POST | `/v1/engagement/assess` | Assess engagement quality for a stakeholder/FPIC workflow |
| GET | `/v1/engagement/assessments` | List engagement assessments |
| GET | `/v1/engagement/assessments/{assessment_id}` | Get assessment details with dimension scores |
| **Compliance Reports** | | |
| POST | `/v1/reports/generate` | Generate a compliance report (specify type, filters, format) |
| GET | `/v1/reports` | List generated reports |
| GET | `/v1/reports/{report_id}` | Download a generated report |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_set_stakeholders_registered_total` | Counter | Stakeholders registered by type |
| 2 | `gl_eudr_set_fpic_workflows_created_total` | Counter | FPIC workflows created by commodity |
| 3 | `gl_eudr_set_fpic_transitions_total` | Counter | FPIC stage transitions by stage |
| 4 | `gl_eudr_set_grievances_submitted_total` | Counter | Grievances submitted by severity and category |
| 5 | `gl_eudr_set_grievances_resolved_total` | Counter | Grievances resolved by resolution type |
| 6 | `gl_eudr_set_consultations_created_total` | Counter | Consultation records created by type |
| 7 | `gl_eudr_set_communications_sent_total` | Counter | Communications sent by channel |
| 8 | `gl_eudr_set_communications_delivered_total` | Counter | Communications successfully delivered |
| 9 | `gl_eudr_set_assessments_completed_total` | Counter | Engagement quality assessments completed |
| 10 | `gl_eudr_set_reports_generated_total` | Counter | Compliance reports generated by type |
| 11 | `gl_eudr_set_processing_duration_seconds` | Histogram | Processing latency by operation type |
| 12 | `gl_eudr_set_grievance_resolution_duration_days` | Histogram | Grievance resolution time in days |
| 13 | `gl_eudr_set_errors_total` | Counter | Errors by operation type |
| 14 | `gl_eudr_set_active_fpic_workflows` | Gauge | Currently active FPIC workflows |
| 15 | `gl_eudr_set_active_grievances` | Gauge | Currently open grievances by severity |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for audit trail |
| Cache | Redis | Stakeholder lookup caching, communication rate limiting |
| Object Storage | S3 | Evidence files (photos, audio, documents), generated reports |
| Communication | Twilio (SMS), SMTP (email), WhatsApp Business API | Multi-channel stakeholder communication |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based access to stakeholder data |
| Encryption | AES-256-GCM via SEC-003 | Stakeholder PII encryption at rest |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-set:stakeholders:read` | View stakeholder registry | Viewer, Analyst, Compliance Officer, Indigenous Rights Specialist, Grievance Officer, Admin |
| `eudr-set:stakeholders:write` | Create, update, archive stakeholders | Analyst, Compliance Officer, Indigenous Rights Specialist, Admin |
| `eudr-set:stakeholders:discover` | Trigger automated stakeholder discovery | Compliance Officer, Indigenous Rights Specialist, Admin |
| `eudr-set:fpic:read` | View FPIC workflows and stage history | Viewer, Analyst, Compliance Officer, Indigenous Rights Specialist, Admin |
| `eudr-set:fpic:write` | Create FPIC workflows | Compliance Officer, Indigenous Rights Specialist, Admin |
| `eudr-set:fpic:transition` | Advance FPIC workflow stages | Indigenous Rights Specialist, Compliance Officer, Admin |
| `eudr-set:fpic:approve` | Approve FPIC stage gates | Compliance Officer, Legal Counsel, Admin |
| `eudr-set:grievances:read` | View grievances (non-anonymous details) | Grievance Officer, Compliance Officer, Admin |
| `eudr-set:grievances:submit` | Submit new grievances | All authenticated users, Public (anonymous) |
| `eudr-set:grievances:manage` | Triage, investigate, resolve grievances | Grievance Officer, Compliance Officer, Admin |
| `eudr-set:grievances:appeal` | Review and decide appeals | Compliance Officer (senior), Admin |
| `eudr-set:consultations:read` | View consultation records | Viewer, Analyst, Compliance Officer, Community Liaison, Admin |
| `eudr-set:consultations:write` | Create and update consultation records | Community Liaison, Compliance Officer, Indigenous Rights Specialist, Admin |
| `eudr-set:consultations:finalize` | Finalize and lock consultation records | Compliance Officer, Indigenous Rights Specialist, Admin |
| `eudr-set:communications:read` | View communication logs | Viewer, Compliance Officer, Admin |
| `eudr-set:communications:send` | Send stakeholder communications | Compliance Officer, Community Liaison, Indigenous Rights Specialist, Admin |
| `eudr-set:engagement:read` | View engagement quality assessments | Viewer, Analyst, Compliance Officer, Indigenous Rights Specialist, Admin |
| `eudr-set:engagement:assess` | Perform engagement quality assessments | Indigenous Rights Specialist, Compliance Officer, Admin |
| `eudr-set:reports:read` | View and download compliance reports | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| `eudr-set:reports:generate` | Generate compliance reports | Compliance Officer, Admin |
| `eudr-set:audit:read` | View audit trails and provenance data | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping Master | Supply chain graph API | Supply chain nodes/edges -> stakeholder-to-node linkage; geographic proximity for stakeholder discovery |
| AGENT-EUDR-002 Geolocation Verification | Plot coordinates API | Production plot locations -> community proximity analysis |
| AGENT-EUDR-021 Indigenous Rights Checker | Territory database API, Community registry | Indigenous territory boundaries and community data -> stakeholder discovery, FPIC workflow initialization |
| AGENT-EUDR-022 Protected Area Validator | Protected area community data | Community data from protected areas -> stakeholder registry enrichment |
| AGENT-EUDR-016 Country Risk Evaluator | Country governance data | Indigenous rights scores per country -> engagement requirement classification |
| AGENT-FOUND-005 Citations & Evidence | Regulatory references | EUDR article citations for compliance checks and report generation |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-025 Risk Mitigation Advisor | Stakeholder engagement as mitigation measure | Engagement status, FPIC status, grievance data -> mitigation strategy inputs |
| AGENT-EUDR-026 Due Diligence Orchestrator | Workflow step integration | Stakeholder engagement as due diligence phase element; quality gate data |
| AGENT-EUDR-028 Risk Assessment Engine | Engagement data as risk input | Consultation outcomes, grievance patterns, FPIC status -> Article 10(2)(e) risk factor |
| AGENT-EUDR-030 Documentation Generator | Engagement evidence for DDS | Stakeholder engagement reports -> DDS package assembly |
| GL-EUDR-APP v1.0 | API integration | Stakeholder data, FPIC dashboards, grievance dashboards -> frontend display |
| External Auditors | Read-only API + exports | Engagement reports, FPIC certificates, consultation records for third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Stakeholder Discovery and Mapping (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Stakeholder Engagement" module
3. Clicks "Discover Stakeholders" for a selected supply chain (e.g., Cocoa from Cote d'Ivoire)
4. System queries EUDR-001 supply chain graph for all production plot locations
5. System queries EUDR-021 for indigenous territories overlapping production plots
6. System queries EUDR-022 for protected area communities near plots
7. System presents discovered stakeholders: 12 indigenous communities, 45 local villages,
   8 cooperatives, 3 NGOs monitoring the region
8. Compliance officer reviews, confirms, and registers stakeholders
9. System auto-classifies rights and engagement obligations per stakeholder type
10. Stakeholder map displayed overlaid on supply chain visualization
```

#### Flow 2: FPIC Workflow Management (Indigenous Rights Specialist)

```
1. Indigenous rights specialist opens FPIC Workflows dashboard
2. System shows 5 active FPIC workflows for palm oil concessions in Kalimantan
3. Specialist selects workflow for "Community X - Concession Area 7"
4. Current stage: CONSULTATION (Stage 4)
5. Specialist records consultation meeting: uploads meeting minutes, photo evidence,
   signed attendance sheet
6. Specialist advances workflow to CONSENT stage
7. System validates: all previous stage evidence complete, SLA met, approval obtained
8. Community representative signs digital consent form
9. Specialist records consent as "GRANTED" with conditions
10. System generates FPIC compliance certificate
11. FPIC status automatically updates EUDR-021 records and EUDR-028 risk inputs
```

#### Flow 3: Grievance Processing (Grievance Officer)

```
1. Community member submits complaint via SMS: "Clearing activity near our village
   water source. Reference: Plot Area 15, Kalimantan"
2. System generates grievance GRV-2026-00456, classifies as Environmental/High
3. Complainant receives SMS acknowledgement with case reference within 48 hours
4. Grievance officer receives alert, opens case in grievance dashboard
5. Officer triages: confirms severity HIGH, assigns 7-day SLA
6. Officer assigns investigation to field team
7. Field team conducts site visit, documents findings with photos and GPS coordinates
8. Findings uploaded: clearing activity confirmed, within 2km of village water source
9. Resolution proposed: halt clearing, conduct environmental impact assessment,
   engage community on water source protection plan
10. Complainant notified of proposed resolution via SMS
11. Complainant accepts resolution
12. Remediation tracked: clearing halted (verified), EIA commissioned (in progress)
13. Case closed after remediation verified; satisfaction rating: 4/5
14. Grievance data feeds into EUDR-028 risk reassessment for the area
```

#### Flow 4: Consultation Documentation (Community Liaison - Offline)

```
1. Community liaison officer travels to remote cocoa cooperative in Cote d'Ivoire
2. Opens mobile consultation form (offline mode)
3. Selects template: "Community Needs Assessment"
4. Records: date, GPS location, objectives, participant names and roles
5. During meeting, captures: topics discussed, community concerns, outcomes
6. Takes photos of meeting (with participant consent recorded in app)
7. Records commitments: cooperative requests training on sustainable practices;
   EU buyer commits to premium price for deforestation-free cocoa
8. Saves consultation record locally (encrypted on device)
9. Returns to town with connectivity; app auto-syncs to platform
10. Consultation record appears in compliance officer's dashboard within 60 seconds
11. Compliance officer reviews and finalizes the record
12. Record immutably locked with SHA-256 provenance hash
```

### 8.2 Key Screen Descriptions

**Stakeholder Registry Dashboard:**
- Map view: stakeholders plotted on geographic map overlaid with supply chain nodes
- List view: filterable table with stakeholder type, country, status, engagement history
- Detail panel: full stakeholder profile with engagement timeline, linked FPIC workflows, grievances, and communications
- Discovery panel: automated stakeholder discovery with source attribution

**FPIC Workflow Dashboard:**
- Portfolio view: all active FPIC workflows with stage progress indicators (7-stage pipeline visualization)
- Workflow detail: stage-by-stage view with evidence, approvals, SLA status, and audit trail
- Calendar view: upcoming FPIC deadlines, deliberation period end dates, renewal dates
- Certificate view: completed FPIC certificates with download option

**Grievance Dashboard:**
- Summary cards: Open grievances by severity (Critical: red, High: orange, Medium: yellow, Low: green)
- Status pipeline: grievances flowing through lifecycle stages
- SLA tracker: overdue grievances highlighted with escalation alerts
- Trend charts: monthly grievance intake, resolution rates, average resolution time
- Public dashboard: anonymized version for external stakeholder transparency

**Consultation Register:**
- Calendar/timeline view of all consultations
- Map view: consultation locations plotted geographically
- Search: find consultations by community, topic, date, outcome
- Offline sync status: indicator showing pending syncs from field tools

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 7 P0 features (Features 1-7) implemented and tested
  - [ ] Feature 1: Stakeholder Mapper -- registry CRUD, discovery, supply chain linkage
  - [ ] Feature 2: FPIC Workflow Engine -- 7-stage workflow, SLA enforcement, evidence management
  - [ ] Feature 3: Grievance Mechanism -- multi-channel intake, lifecycle management, UNGP compliance
  - [ ] Feature 4: Consultation Record Manager -- structured documentation, offline support, finalization
  - [ ] Feature 5: Communication Hub -- multi-channel dispatch, templates, delivery tracking
  - [ ] Feature 6: Indigenous Rights Engagement Verifier -- 6-dimension scoring, deterministic
  - [ ] Feature 7: Compliance Reporter -- 7 report types, multi-format, regulatory mapping
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC + PII encryption integrated)
- [ ] Performance targets met (< 200ms API p95; < 30 seconds report generation)
- [ ] All 7 EUDR commodities tested with engagement golden test fixtures
- [ ] Engagement quality scoring verified deterministic (bit-perfect reproducibility)
- [ ] Grievance mechanism tested against all UNGP Principle 31 effectiveness criteria
- [ ] FPIC workflow tested with all stage transitions including edge cases (consent withheld, appeal, renewal)
- [ ] API documentation complete (OpenAPI spec)
- [ ] Database migration V119 tested and validated
- [ ] Integration with EUDR-001, EUDR-021, EUDR-025, EUDR-026, EUDR-028, EUDR-030 verified
- [ ] Multi-language support verified for 12+ languages
- [ ] Offline consultation recording tested with sync verification
- [ ] 5 beta customers successfully using stakeholder engagement module
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ stakeholder registries created by customers
- 25+ FPIC workflows initiated
- 50+ grievances processed through full lifecycle
- 200+ consultations documented
- Average consultation record completeness >= 90%
- p99 API latency < 200ms in production
- < 3 support tickets per customer

**60 Days:**
- 500+ stakeholder registries active
- 100+ FPIC workflows in progress across 5+ commodities
- Grievance resolution rate >= 80% within SLA
- Average engagement quality score >= 65 (Adequate+)
- 1,000+ consultation records with provenance hashes
- Multi-language communication deployed in 8+ languages

**90 Days:**
- 2,000+ stakeholder registries active across 500+ customers
- FPIC workflow completion rate >= 70% within planned timeline
- Grievance mechanism meets all 8 UNGP Principle 31 criteria (validated by third-party assessment)
- Zero EUDR penalties for active customers related to stakeholder engagement gaps
- Engagement quality scores trending upward (month-over-month improvement)
- NPS > 50 from compliance officer and indigenous rights specialist personas

---

## 10. Timeline and Milestones

### Phase 1: Core Registry and FPIC (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Stakeholder Mapper (Feature 1): registry CRUD, stakeholder types, supply chain linkage | Backend Engineer |
| 2-3 | Stakeholder discovery engine: integration with EUDR-001, EUDR-021, EUDR-022 | Backend Engineer |
| 3-5 | FPIC Workflow Engine (Feature 2): 7-stage state machine, SLA enforcement, evidence management | Senior Backend Engineer |
| 5-6 | FPIC integration with EUDR-021 for documentation verification | Backend Engineer |

**Milestone: Stakeholder registry and FPIC workflows operational (Week 6)**

### Phase 2: Grievance and Consultation (Weeks 7-11)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-9 | Grievance Mechanism (Feature 3): multi-channel intake, lifecycle, UNGP compliance | Senior Backend Engineer + Frontend |
| 9-11 | Consultation Record Manager (Feature 4): structured docs, offline PWA, sync engine | Backend + Mobile Engineer |

**Milestone: Grievance mechanism and consultation management operational (Week 11)**

### Phase 3: Communications, Verification, and Reporting (Weeks 12-16)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 12-13 | Communication Hub (Feature 5): multi-channel dispatch, templates, delivery tracking | Backend + Integration Engineer |
| 13-14 | Indigenous Rights Engagement Verifier (Feature 6): 6-dimension scoring | Senior Backend Engineer |
| 14-16 | Compliance Reporter (Feature 7): 7 report types, multi-format, multi-language | Backend Engineer |

**Milestone: All 7 P0 features implemented (Week 16)**

### Phase 4: Integration, Testing, and Launch (Weeks 17-20)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 17 | Full integration with EUDR-025, EUDR-026, EUDR-028, EUDR-030 | Backend Engineer |
| 17-18 | REST API Layer: 30+ endpoints, authentication, RBAC, rate limiting | Backend Engineer |
| 18-19 | Complete test suite: 500+ tests, golden tests for all 7 commodities | Test Engineer |
| 19 | Database migration V119 finalized, performance testing, security audit | DevOps + Security |
| 19-20 | Beta customer onboarding (5 customers), launch readiness review | Product + Engineering |
| 20 | Production launch | All |

**Milestone: Production launch with all 7 P0 features (Week 20)**

### Phase 5: Enhancements (Weeks 21-28)

- Sentiment analysis (Feature 8)
- Benefit-sharing agreement manager (Feature 9)
- Community impact assessment (Feature 10)
- CSDDD Article 7/8/9 specific workflow templates
- Enhanced offline capabilities with native mobile PWA
- Additional language support expansion

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable, production-ready; graph API available |
| AGENT-EUDR-002 Geolocation Verification | BUILT (100%) | Low | Stable, production-ready |
| AGENT-EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | Territory data and FPIC verification available |
| AGENT-EUDR-022 Protected Area Validator | BUILT (100%) | Low | Community data from protected areas available |
| AGENT-EUDR-025 Risk Mitigation Advisor | BUILT (100%) | Low | Stakeholder collaboration hub integration defined |
| AGENT-EUDR-026 Due Diligence Orchestrator | BUILT (100%) | Low | Workflow step integration defined |
| AGENT-EUDR-028 Risk Assessment Engine | BUILT (100%) | Low | Article 10(2)(e) input interface defined |
| AGENT-EUDR-030 Documentation Generator | BUILT (100%) | Low | DDS package integration defined |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Frontend integration points defined |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | PII encryption for stakeholder data |
| INFRA-004 S3 Object Storage | Production Ready | Low | Evidence file storage |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| SMS gateway (Twilio or equivalent) | Available | Low | Multiple provider fallback; SMS is optional channel |
| WhatsApp Business API | Available | Medium | Optional channel; degrades gracefully; alternative SMS fallback |
| EU Information System DDS schema | Published (v1.x) | Medium | Adapter pattern for schema version changes |
| ILO Convention 169 ratification status | Published | Low | Static reference data; updated annually |
| UNDRIP implementation guidance | Published | Low | Static reference framework |
| CSDDD implementing regulations | Evolving (2026-2027) | Medium | Modular design allows adaptation; core principles stable |
| National indigenous rights legislation (Brazil, Indonesia, Colombia, etc.) | Published; varies | Medium | Configurable per-country legal requirements; regular review cycle |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Stakeholders lack digital access for grievance submission or communication | High | High | Multi-channel including SMS, paper forms, community-based submission points; lowest-common-denominator design |
| R2 | FPIC processes take longer than planned timelines (90-180 days) | High | Medium | Configurable timeline extension; workflow supports indefinite deliberation; SLA alerts prevent silent delays |
| R3 | Community representatives disputed or lack legitimacy | Medium | High | Representative verification process; support for multiple representatives per community; mediation workflow for disputes |
| R4 | Offline consultation data lost or corrupted before sync | Medium | High | Encrypted local storage; conflict-free replication on sync; automatic backup to device storage; checksums on every record |
| R5 | Grievance mechanism overwhelmed by volume in high-risk sourcing regions | Medium | Medium | Auto-triage engine; configurable escalation; batch processing for similar complaints; capacity scaling |
| R6 | Language barriers prevent meaningful engagement | High | High | 12+ language support; interpreter coordination tracking; visual and audio communication templates; community radio scripts |
| R7 | Sensitive stakeholder PII data breach | Low | Critical | AES-256 encryption at rest (SEC-003); TLS 1.3 in transit (SEC-004); RBAC access control (SEC-002); data minimization; anonymous grievance option |
| R8 | CSDDD requirements change during development | Medium | Medium | Modular design with configurable compliance rules; CSDDD features built as extensible modules |
| R9 | Integration complexity with 6+ upstream EUDR agents | Medium | Medium | Well-defined API contracts; mock adapters for testing; circuit breaker pattern; integration tested per phase |
| R10 | Low stakeholder adoption of digital engagement tools | Medium | Medium | Training materials in local languages; community liaison onboarding program; paper-to-digital bridge workflows; gradual digitization approach |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Stakeholder Mapper Unit Tests | 80+ | Registry CRUD, type classification, discovery, deduplication, search |
| FPIC Workflow Unit Tests | 100+ | 7-stage state machine, transitions, SLA enforcement, evidence validation, edge cases (consent withheld, renewal, multi-community) |
| Grievance Mechanism Unit Tests | 90+ | Lifecycle states, triage, SLA enforcement, appeal, anonymous handling, UNGP criteria |
| Consultation Record Tests | 70+ | CRUD, finalization, addendum, offline sync, evidence attachment, provenance |
| Communication Hub Tests | 60+ | Multi-channel dispatch, template rendering, delivery tracking, rate limiting, multi-language |
| Engagement Verifier Tests | 50+ | 6-dimension scoring, determinism, classification, recommendations |
| Compliance Reporter Tests | 60+ | All 7 report types, multi-format, regulatory mapping, provenance |
| API Tests | 80+ | All 30+ endpoints, auth, error handling, pagination, RBAC |
| Golden Tests | 50+ | All 7 commodities with complete engagement scenarios |
| Integration Tests | 30+ | Cross-agent integration with EUDR-001/021/025/026/028/030 |
| Performance Tests | 20+ | Concurrent workflows, communication throughput, report generation timing |
| Security Tests | 15+ | PII encryption, anonymous grievance, RBAC enforcement, data isolation |
| **Total** | **700+** | |

### 13.2 Golden Test Scenarios

Each of the 7 commodities will have a dedicated golden test with the following stakeholder engagement scenarios:

1. **Complete engagement** -- Full stakeholder mapping, FPIC completed with consent, consultations documented, no grievances -- expect 100% compliance
2. **FPIC consent withheld** -- Community withholds consent; operator cannot proceed -- expect workflow correctly blocks and documents objection
3. **Grievance lifecycle** -- Complaint submitted, investigated, resolved with satisfaction -- expect full lifecycle tracked
4. **Multi-community FPIC** -- Multiple communities with overlapping interests -- expect parallel workflows with aggregated consent
5. **Offline consultation** -- Consultation documented offline with delayed sync -- expect data integrity preserved
6. **Anonymous grievance** -- Anonymous complaint through SMS -- expect complainant identity protected
7. **Engagement quality failure** -- Engagement with insufficient cultural appropriateness -- expect low quality score with recommendations

Total: 7 commodities x 7 scenarios = 49 golden test scenarios

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **CSDDD** | EU Corporate Sustainability Due Diligence Directive |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **FPIC** | Free, Prior and Informed Consent -- indigenous peoples' right to give or withhold consent for projects affecting their lands |
| **ILO 169** | ILO Convention 169 on Indigenous and Tribal Peoples -- international convention on indigenous rights, ratified by 24 countries |
| **UNDRIP** | United Nations Declaration on the Rights of Indigenous Peoples -- adopted 2007, 46 articles on indigenous rights |
| **UNGP** | UN Guiding Principles on Business and Human Rights -- framework for corporate human rights responsibility |
| **Stakeholder** | Any individual, group, or organization affected by or able to affect an operator's supply chain activities |
| **Grievance Mechanism** | Structured process for receiving, investigating, and resolving complaints from affected stakeholders |
| **Consultation** | Structured engagement process with stakeholders to gather information, obtain views, or seek consent |
| **Benefit-Sharing** | Arrangement whereby communities affected by commodity production receive a fair share of benefits |
| **Competent Authority** | National authority designated by EU Member States to enforce EUDR (Articles 14-16) |

### Appendix B: ILO Convention 169 Ratification Status (EUDR-Relevant Countries)

| Country | EUDR Commodities | ILO 169 Ratified | Year | FPIC Mandatory |
|---------|-----------------|-------------------|------|---------------|
| Brazil | Cattle, Cocoa, Coffee, Soya, Wood | Yes | 2002 | Yes |
| Colombia | Cocoa, Coffee, Palm Oil | Yes | 1991 | Yes |
| Peru | Cocoa, Coffee, Wood | Yes | 1994 | Yes |
| Guatemala | Cocoa, Coffee, Palm Oil | Yes | 1996 | Yes |
| Honduras | Cocoa, Coffee, Palm Oil | Yes | 1995 | Yes |
| Ecuador | Cocoa, Coffee | Yes | 1998 | Yes |
| Paraguay | Cattle, Soya | Yes | 1993 | Yes |
| Bolivia | Cattle, Cocoa, Coffee, Soya, Wood | Yes | 1991 | Yes |
| Indonesia | Palm Oil, Rubber, Cocoa, Coffee, Wood | No | -- | National legislation varies |
| Cote d'Ivoire | Cocoa, Coffee, Rubber | No | -- | Customary rights recognized |
| Ghana | Cocoa, Coffee, Wood | No | -- | Constitutional protections |
| DRC | Cocoa, Coffee, Wood | No | -- | National legislation developing |
| Malaysia | Palm Oil, Rubber, Wood | No | -- | National legislation varies by state |
| Vietnam | Coffee, Rubber, Wood | No | -- | Ethnic minority protections |
| Thailand | Rubber, Wood | No | -- | National protections limited |

### Appendix C: UNGP Principle 31 Effectiveness Criteria Reference

The eight effectiveness criteria for non-judicial grievance mechanisms (Principle 31):

1. **Legitimate**: Enabling trust from stakeholder groups; accountable for fair conduct
2. **Accessible**: Known to all stakeholder groups; providing adequate assistance for those who face barriers
3. **Predictable**: Providing a clear and known procedure with indicative timeframes; clarity on types of outcomes
4. **Equitable**: Seeking to ensure aggrieved parties have reasonable access to information, advice, and expertise
5. **Transparent**: Keeping parties informed about progress; providing sufficient information about mechanism performance
6. **Rights-compatible**: Ensuring outcomes and remedies accord with internationally recognized human rights
7. **Source of continuous learning**: Drawing on relevant measures to identify lessons for improving mechanism and preventing future grievances
8. **Based on engagement and dialogue**: Consulting stakeholder groups on design and performance; focusing on dialogue as means of addressing and resolving grievances

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EU Deforestation Regulation)
2. Directive (EU) 2024/1760 of the European Parliament and of the Council (Corporate Sustainability Due Diligence Directive)
3. ILO Convention 169 on Indigenous and Tribal Peoples in Independent Countries (1989)
4. United Nations Declaration on the Rights of Indigenous Peoples (2007)
5. UN Guiding Principles on Business and Human Rights (2011)
6. FAO Voluntary Guidelines on the Responsible Governance of Tenure (2012)
7. FSC Policy for the Association of Organizations with FSC (FSC-POL-01-004)
8. RSPO Principles and Criteria for the Production of Sustainable Palm Oil (2018)
9. Rainforest Alliance Sustainable Agriculture Standard (2020)
10. EU Guidance on FPIC in the Context of EUDR (European Commission, 2025)
11. LandMark Global Platform of Indigenous and Community Lands
12. RAISG Amazonian Network of Geo-referenced Socio-Environmental Information

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-12 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Indigenous Rights Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-12 | GL-ProductManager | Initial draft created: 7 P0 features (Stakeholder Mapper, FPIC Workflow Engine, Grievance Mechanism, Consultation Record Manager, Communication Hub, Indigenous Rights Engagement Verifier, Compliance Reporter), regulatory coverage (EUDR Articles 2/4/8/9/10/11/12/29/31 + ILO 169 + UNDRIP + CSDDD), 30+ API endpoints, V119 migration, 700+ test targets, 20-week timeline |
