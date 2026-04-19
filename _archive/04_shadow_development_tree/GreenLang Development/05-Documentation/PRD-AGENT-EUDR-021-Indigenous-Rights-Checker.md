# PRD: AGENT-EUDR-021 -- Indigenous Rights Checker Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-021 |
| **Agent ID** | GL-EUDR-IRC-021 |
| **Component** | Indigenous Rights Checker Agent |
| **Category** | EUDR Regulatory Agent -- Human Rights & Indigenous Peoples Compliance |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 2, 8, 10, 11, 29; ILO Convention 169; UNDRIP |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) explicitly conditions the legality of commodity production on compliance with the laws of the country of production, including laws governing indigenous peoples' rights, land tenure, and customary use (Article 2(28), Article 3, Article 10(2)(d)). Article 29(3)(c) further mandates that the European Commission's country benchmarking must consider the "respect for the rights of indigenous peoples, local communities, and other customary tenure rights holders." The regulation's Recital 32 explicitly references the United Nations Declaration on the Rights of Indigenous Peoples (UNDRIP) and ILO Convention 169, establishing that EUDR-compliant supply chains must not violate indigenous peoples' rights to their lands, territories, and resources.

In practice, approximately 370 million indigenous people across 90+ countries occupy or depend upon 28% of the world's land surface, including some of the most critical tropical forest regions from which EUDR-regulated commodities are sourced. The Amazon Basin, Congo Basin, and Southeast Asian archipelago -- the three primary sourcing regions for cocoa, palm oil, rubber, soya, cattle, coffee, and wood -- are home to thousands of indigenous communities with legally recognized or customary land rights. Commodity production that encroaches on indigenous territories without Free, Prior and Informed Consent (FPIC) is not only a human rights violation but renders the resulting products non-compliant with EUDR, exposing EU operators to penalties of up to 4% of annual EU turnover.

Today, EU operators and compliance teams face the following critical gaps when verifying indigenous rights compliance:

- **No indigenous territory database integration**: Operators have no systematic access to georeferenced indigenous territory boundaries. Data is scattered across national registries (FUNAI in Brazil, BPN in Indonesia), regional databases (RAISG for Amazonia), and global platforms (LandMark). No existing compliance tool consolidates these sources into a queryable, verified spatial database.
- **No FPIC verification engine**: Free, Prior and Informed Consent is required by ILO Convention 169 (ratified by 24 countries) and by national legislation in major commodity-producing countries (Brazil, Colombia, Peru, Philippines). There is no automated system to verify whether FPIC has been obtained for commodity production activities on or near indigenous territories, track consent status, or validate FPIC documentation.
- **No land rights overlap detection**: When production plots are located on, adjacent to, or overlapping with indigenous territories, operators must perform enhanced verification. There is no spatial analysis tool that automatically detects these overlaps and quantifies the degree of encroachment.
- **No community consultation tracking**: EUDR due diligence requires engagement with indigenous communities when supply chains intersect their territories. There is no structured system to track consultation processes, record community responses, manage grievance mechanisms, or document consultation outcomes for audit purposes.
- **No rights violation alert system**: When indigenous rights violations are reported by NGOs, media, judicial authorities, or community organizations in regions where operators source commodities, there is no automated mechanism to correlate these reports with active supply chains and trigger enhanced due diligence.
- **No indigenous community registry**: Operators cannot systematically identify which indigenous communities are present in their sourcing regions, what rights those communities hold, what legal protections apply, and what consultation obligations exist.
- **No FPIC workflow management**: The FPIC process involves multiple stages (identification, information sharing, consultation, consent/objection, agreement, monitoring). There is no structured workflow to manage this multi-stage process, enforce timelines, or ensure procedural compliance.
- **No compliance documentation for auditors**: Auditors reviewing EUDR due diligence statements need structured evidence that indigenous rights have been assessed and respected. There is no standardized reporting framework that generates audit-ready indigenous rights compliance documentation.

Without solving these problems, EU operators face EUDR penalties, reputational damage from association with indigenous rights violations, legal liability under the EU Corporate Sustainability Due Diligence Directive (CSDDD), and exclusion from responsible sourcing certification schemes (FSC, RSPO) that require FPIC compliance.

### 1.2 Solution Overview

Agent-EUDR-021: Indigenous Rights Checker is a specialized compliance agent that provides comprehensive indigenous peoples' rights verification for EUDR-regulated commodity supply chains. It is the 21st agent in the EUDR agent family and extends the Risk Assessment sub-category (EUDR-016 through EUDR-020) with dedicated human rights compliance capabilities. The agent integrates authoritative indigenous territory databases, automates FPIC verification, performs spatial overlap analysis between production plots and indigenous lands, manages community consultation workflows, monitors rights violation alerts, and generates audit-ready compliance documentation.

The agent builds on and integrates with the existing EUDR agent ecosystem: EUDR-001 (Supply Chain Mapping Master) for supply chain graph data, EUDR-002 (Geolocation Verification) for plot coordinate validation, EUDR-006 (Plot Boundary Manager) for spatial plot data, EUDR-016 (Country Risk Evaluator) for governance and indigenous rights scoring, EUDR-017 (Supplier Risk Scorer) for supplier-level risk assessment, and EUDR-020 (Deforestation Alert System) for spatial monitoring alerts.

Core capabilities:

1. **Indigenous territory database integration** -- Consolidated, georeferenced database of indigenous and tribal peoples' territories from 6 authoritative sources (LandMark, RAISG, FUNAI, BPN/AMAN, ACHPR, national registries) covering 100+ countries and 50,000+ territories. Spatial indexing for sub-second overlap queries. Territory legal status tracking (titled, claimed, customary, pending).
2. **FPIC documentation verification engine** -- Automated verification of FPIC documentation completeness, procedural compliance, and temporal validity. Validates that consent was obtained prior to project commencement, that information disclosure was adequate, that community representation was legitimate, and that consent was freely given without coercion. Deterministic scoring (0-100) with provenance tracking.
3. **Land rights overlap detector** -- Spatial analysis engine that detects overlaps between supply chain production plots and indigenous territories using PostGIS intersection, buffer zone analysis, and proximity scoring. Classifies overlaps as direct (plot inside territory), partial (polygon intersection), adjacent (within configurable buffer, default 5km), and proximate (within 25km). Calculates overlap area, percentage, and affected community identification.
4. **Community consultation tracker** -- Structured workflow for tracking consultation processes with indigenous communities, from initial identification through engagement, information sharing, consultation, consent/objection recording, agreement management, and ongoing monitoring. Supports multi-stakeholder engagement records, grievance mechanisms, and benefit-sharing agreement tracking.
5. **Rights violation alert system** -- Monitors indigenous rights violation reports from 10+ sources (IWGIA, Cultural Survival, Forest Peoples Programme, Amnesty International, national human rights commissions, judicial databases, media monitoring). Correlates violation reports with operator supply chain locations. Generates alerts with severity classification and recommended due diligence response.
6. **Indigenous community registry** -- Structured database of indigenous communities present in EUDR commodity-producing regions, including community name, population, language, territory boundaries, legal recognition status, representative organizations, contact channels, and applicable legal protections (ILO 169 ratification, national indigenous rights legislation).
7. **FPIC workflow management** -- Multi-stage FPIC process management: identification, information disclosure, consultation, consent/objection, agreement, monitoring, and renewal. Stage-gate enforcement with configurable timelines, SLA tracking, escalation rules, and audit trail. Templates for FPIC documentation in 8 languages.
8. **Compliance reporting and documentation** -- Automated generation of indigenous rights compliance reports for EUDR due diligence statements, third-party audits, certification scheme verification (FSC FPIC requirements, RSPO FPIC standard), and regulatory submission. Reports include territory overlap analysis, FPIC status, consultation records, and violation screening results. Formats: PDF, JSON, HTML. Multi-language support (EN, FR, DE, ES, PT).
9. **Integration with existing EUDR agents** -- Bidirectional integration with EUDR-001 (supply chain graph nodes enriched with indigenous rights data), EUDR-002 (plot coordinates validated against territory boundaries), EUDR-006 (plot boundaries for overlap analysis), EUDR-016 (indigenous rights scores feed country risk), EUDR-017 (supplier risk adjusted for FPIC compliance), and EUDR-020 (deforestation alerts correlated with indigenous territory proximity).

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Indigenous territory coverage | 50,000+ territories across 100+ countries | Count of territories in spatial database |
| Territory data source integration | 6 authoritative sources (LandMark, RAISG, FUNAI, BPN/AMAN, ACHPR, national registries) | Source count and coverage validation |
| Overlap detection accuracy | >= 99% spatial precision (validated against manual GIS analysis) | Cross-validation with expert GIS review |
| Overlap detection performance | < 500ms per plot-territory overlap query | p99 latency under load |
| FPIC verification coverage | All 7 EUDR commodities in all producing countries with indigenous presence | Commodity-country coverage matrix |
| FPIC documentation completeness scoring | 100% deterministic, reproducible | Bit-perfect reproducibility tests |
| Rights violation source coverage | 10+ monitoring sources integrated | Count of active monitoring feeds |
| Violation-to-alert latency | < 48 hours from report publication to operator alert | Time from source publication to alert dispatch |
| Community consultation tracking | 100% of consultations with complete audit trail | % of consultations with full lifecycle records |
| Compliance report generation | < 10 seconds per indigenous rights compliance report | Time from request to PDF/JSON delivery |
| EUDR regulatory coverage | 100% of Articles 2, 8, 10, 11, 29 indigenous rights requirements | Regulatory compliance matrix |
| Zero-hallucination guarantee | 100% deterministic calculations, no LLM in critical path | Bit-perfect reproducibility tests |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, plus the broader indigenous rights compliance technology market estimated at 1-2 billion EUR as CSDDD enforcement approaches (2027).
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of EUDR-regulated commodities requiring indigenous rights verification for sourcing regions with indigenous presence (Amazon, Congo Basin, Southeast Asia, Central America), estimated at 300-500M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 20-35M EUR in indigenous rights compliance module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) sourcing from regions with indigenous populations (Brazil, Indonesia, Colombia, DRC, Peru)
- Multinational food and beverage companies with cocoa, coffee, palm oil, and soya supply chains intersecting indigenous territories
- Timber and paper industry operators with tropical wood sourcing from indigenous forest regions
- Compliance officers responsible for EUDR due diligence with indigenous rights verification obligations

**Secondary:**
- Certification bodies (FSC, RSPO, Rainforest Alliance) requiring FPIC verification as part of certification audits
- Commodity traders and intermediaries operating in regions with indigenous land rights conflicts
- Financial institutions with exposure to EUDR-regulated commodity supply chains requiring human rights due diligence under CSDDD
- NGOs and indigenous rights organizations monitoring corporate compliance
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 User Personas

#### Persona 1: Indigenous Rights Officer -- Sofia (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Indigenous Rights & Community Relations at a European palm oil refinery |
| **Company** | 6,000 employees, sourcing from Indonesia (Kalimantan, Sumatra, Papua) and Colombia |
| **EUDR Pressure** | Board mandate to verify FPIC compliance for all sourcing regions; multiple NGO reports of indigenous rights violations in supply chain |
| **Pain Points** | Cannot identify which plantations overlap indigenous territories; no system to track FPIC status across 200+ supplier plantations; manual FPIC verification takes months; no centralized community engagement records; audit requests for indigenous rights documentation take weeks to compile |
| **Goals** | Automated territory overlap detection for all supplier plots; structured FPIC workflow management; real-time violation monitoring; audit-ready compliance reports |
| **Technical Skill** | Moderate -- comfortable with web applications, GIS basics, and stakeholder engagement platforms |

#### Persona 2: Compliance Officer -- Marco (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior EUDR Compliance Manager at a European chocolate manufacturer |
| **Company** | 4,000 employees, sourcing cocoa from Cote d'Ivoire, Ghana, Ecuador, Peru, Colombia |
| **EUDR Pressure** | Must demonstrate indigenous rights compliance in DDS; cocoa sourcing regions in Ecuador and Peru overlap indigenous territories |
| **Pain Points** | No visibility into which cocoa cooperatives operate on or near indigenous lands; FPIC documentation scattered across supplier files; cannot correlate rights violation reports with supply chain; regulatory auditors requesting indigenous rights evidence |
| **Goals** | Territory overlap screening for entire supply chain; FPIC compliance dashboard; integrated indigenous rights section in DDS reports; automated violation monitoring for sourcing regions |
| **Technical Skill** | Moderate -- comfortable with compliance platforms and reporting tools |

#### Persona 3: Sustainability Director -- Ingrid (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | VP Sustainability at a European timber importer |
| **Company** | 2,500 employees, importing tropical wood from Brazil, DRC, Myanmar, Indonesia |
| **EUDR Pressure** | FSC certification requires FPIC verification; EUDR DDS requires indigenous rights assessment; investor ESG reporting demands indigenous rights metrics |
| **Pain Points** | Cannot demonstrate positive impact on indigenous communities; benefit-sharing agreements not systematically tracked; FSC audit findings on FPIC gaps; no trend reporting on indigenous rights performance |
| **Goals** | Comprehensive indigenous rights compliance dashboard; benefit-sharing agreement management; certification scheme alignment (FSC, PEFC); investor-ready indigenous rights reporting |
| **Technical Skill** | Low-moderate -- uses dashboards and reports; relies on team for technical analysis |

#### Persona 4: External Auditor -- Dr. Reyes (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body specializing in human rights due diligence |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify operator indigenous rights assessment methodology, FPIC documentation, and community engagement records |
| **Pain Points** | Operators provide inconsistent indigenous rights evidence; no standardized FPIC verification framework; territory data from multiple sources difficult to cross-reference; consultation records incomplete or missing |
| **Goals** | Access read-only indigenous rights data with full provenance; verify FPIC documentation completeness; validate territory overlap analysis methodology; audit community consultation records |
| **Technical Skill** | Moderate -- comfortable with audit software, GIS review, and document verification |

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to achieve EUDR compliance for indigenous rights requirements | 100% of customers pass Article 10/11 audits for indigenous rights component | Q2 2026 |
| BG-2 | Reduce time-to-verify indigenous rights compliance from weeks to minutes | 95% reduction in verification time (weeks to < 10 minutes per supply chain) | Q2 2026 |
| BG-3 | Become the reference indigenous rights compliance platform for EUDR | 500+ enterprise customers using indigenous rights module | Q4 2026 |
| BG-4 | Prevent EUDR penalties attributable to indigenous rights non-compliance | Zero EUDR penalties for active customers related to indigenous rights | Ongoing |
| BG-5 | Support CSDDD readiness by building indigenous rights due diligence infrastructure | Indigenous rights module reusable for CSDDD Article 7 compliance | Q1 2027 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Comprehensive territory database | Integrate 50,000+ indigenous territories from 6 authoritative sources with verified spatial data |
| PG-2 | Automated FPIC verification | Score FPIC documentation completeness and procedural compliance deterministically |
| PG-3 | Spatial overlap detection | Detect and classify plot-territory overlaps with sub-second performance |
| PG-4 | Community engagement tracking | Manage full lifecycle of community consultation processes with audit trail |
| PG-5 | Violation monitoring | Monitor 10+ sources for rights violation reports correlated with supply chains |
| PG-6 | Audit-ready reporting | Generate compliant indigenous rights documentation for DDS, auditors, and certifiers |
| PG-7 | Agent ecosystem integration | Enrich EUDR-001 graph, EUDR-016 country risk, and EUDR-017 supplier risk with indigenous rights data |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Overlap query performance | < 500ms p99 per plot-territory overlap query |
| TG-2 | Batch overlap screening | 10,000 plots in < 5 minutes |
| TG-3 | FPIC scoring performance | < 100ms per FPIC documentation assessment |
| TG-4 | Report generation | < 10 seconds per indigenous rights compliance report PDF |
| TG-5 | API response time | < 200ms p95 for standard queries |
| TG-6 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-7 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-8 | Data freshness | Territory database updated within 30 days of source publication |

---

## 4. Regulatory Requirements

### 4.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(28)** | Definition of "legally produced" -- produced in accordance with the relevant legislation of the country of production, including on indigenous peoples' rights and land tenure | FPIC verification engine (F2) validates that production complied with national indigenous rights legislation; territory overlap detector (F3) verifies land tenure compliance |
| **Art. 2(30-32)** | Definitions of "plot of land" and "geolocation" | Land rights overlap detector (F3) uses plot geolocation data to perform spatial intersection with indigenous territory boundaries |
| **Art. 3** | Prohibition on placing non-compliant products on the EU market | Compliance reporting (F8) flags products sourced from plots with unresolved indigenous territory overlaps or missing FPIC as non-compliant |
| **Art. 8** | Due diligence obligations requiring operators to collect information, assess risk, and mitigate risk including with respect to legality of production | Full agent pipeline: territory database (F1) for information collection, overlap detection (F3) and FPIC verification (F2) for risk assessment, consultation tracker (F4) and FPIC workflow (F7) for risk mitigation |
| **Art. 10(1)** | Risk assessment -- operators shall assess and identify risk of non-compliance | Indigenous rights risk scoring across all 9 features contributes to Article 10 risk assessment |
| **Art. 10(2)(a)** | Risk assessment criterion: complexity of relevant supply chain | Multi-tier territory screening through supply chain graph integration (F9) assesses indigenous rights risk at every tier |
| **Art. 10(2)(d)** | Risk assessment criterion: prevalence of deforestation or forest degradation linked to violations of indigenous peoples' rights | Rights violation alert system (F5) correlates deforestation events with indigenous rights violations; historical violation data feeds risk scoring |
| **Art. 10(2)(e)** | Risk assessment criterion: concerns about the country of production | Country-level indigenous rights scoring (ILO 169 ratification, FPIC legal requirements, land tenure security) feeds EUDR-016 country risk evaluator |
| **Art. 10(2)(f)** | Risk assessment criterion: risk of circumvention or mixing with products of unknown origin | Territory overlap screening for all identified origin plots; flags products from unscreened origins as elevated indigenous rights risk |
| **Art. 11** | Risk mitigation measures | Community consultation tracker (F4), FPIC workflow management (F7), and benefit-sharing agreement tracking provide structured risk mitigation tools |
| **Art. 29(3)(c)** | Country benchmarking criterion: respect for the rights of indigenous peoples, local communities, and other customary tenure right holders | Indigenous rights country scoring engine provides per-country indigenous rights protection score to EUDR-016 for Article 29 benchmarking |
| **Art. 31** | Record keeping for 5 years | All territory data, FPIC records, consultation records, violation alerts, and compliance reports retained for minimum 5 years with immutable audit trail |

### 4.2 International Legal Framework

| Legal Instrument | Status | Agent Relevance |
|-----------------|--------|----------------|
| **ILO Convention 169** (Indigenous and Tribal Peoples Convention, 1989) | Ratified by 24 countries including major producers (Brazil, Colombia, Peru, Guatemala, Honduras, Paraguay, Bolivia) | FPIC verification engine validates compliance with ILO 169 requirements; country scoring reflects ratification status |
| **UNDRIP** (UN Declaration on the Rights of Indigenous Peoples, 2007) | Endorsed by 148 UN member states; referenced in EUDR Recital 32 | Community consultation requirements aligned with UNDRIP Articles 10, 19, 26, 29, 32 |
| **VGGT** (Voluntary Guidelines on Governance of Tenure, 2012) | FAO-endorsed; referenced in EU due diligence guidance | Land tenure verification aligned with VGGT principles on legitimate tenure rights |
| **EU CSDDD** (Corporate Sustainability Due Diligence Directive, 2024) | Effective 2027 for large companies | Agent infrastructure designed for CSDDD Article 7 human rights due diligence reuse |
| **National indigenous rights legislation** | Varies by country | Country-specific legal requirements database covers 50+ producing countries |

### 4.3 FPIC Legal Requirements by Producing Country

| Country | FPIC Legal Basis | Applicable Commodities | Agent Feature |
|---------|-----------------|----------------------|---------------|
| **Brazil** | Federal Constitution Art. 231; ILO 169 (ratified 2002); FUNAI regulatory framework | Cattle, soya, cocoa, coffee, wood | FUNAI territory database integration; FPIC verification per Brazilian Constitutional requirements |
| **Indonesia** | Constitutional Court Decision 35/2012; AMAN customary territory recognition; FPIC in RSPO P&C | Palm oil, rubber, wood, cocoa, coffee | BPN/AMAN territory database; FPIC verification per Indonesian customary law recognition |
| **Colombia** | Constitution Art. 330; ILO 169 (ratified 1991); Constitutional Court T-129/2011 | Coffee, palm oil, cocoa, wood | Resguardo territory database; Prior Consultation per Decree 1320/1998 |
| **Peru** | ILO 169 (ratified 1994); Prior Consultation Law 29785 (2011) | Coffee, cocoa, wood, palm oil | AIDESEP territory maps; Prior Consultation verification per Law 29785 |
| **DRC** | Forest Code Art. 7; Community Forest Concession framework | Wood, cocoa, coffee | ICCN territory data; community forest verification |
| **Cote d'Ivoire** | Land Law 98-750; Rural Land Code 2019 | Cocoa, coffee, rubber, palm oil | Customary land rights verification; community forest boundary recognition |
| **Ghana** | Constitution Art. 267; Stool land management | Cocoa, wood | Stool land boundary database; traditional authority consultation tracking |
| **Malaysia** | Native Customary Rights (NCR) under Sarawak Land Code | Palm oil, rubber, wood | NCR territory database; MSPO FPIC requirements |

### 4.4 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for territory encroachment assessment |
| June 29, 2023 | EUDR entered into force | Legal basis for indigenous rights compliance verification |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | All indigenous rights verification must be operational |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; agent must handle 10x scale |
| 2027 | CSDDD enforcement begins | Indigenous rights module reused for CSDDD human rights due diligence |
| Ongoing (quarterly) | DDS submission deadlines | Indigenous rights compliance reports must be current for quarterly filing |
| Ongoing (periodic) | EC benchmarking updates | Indigenous rights country scores updated when EC revises benchmarks |

---

## 5. Scope and Zero-Hallucination Principles

### 5.1 Scope -- In and Out

**In Scope (v1.0):**
- Indigenous territory database with 50,000+ territories from 6 authoritative sources
- FPIC documentation verification with deterministic completeness scoring
- Spatial overlap detection between production plots and indigenous territories
- Community consultation lifecycle tracking with audit trail
- Rights violation monitoring from 10+ sources with supply chain correlation
- Indigenous community registry for EUDR commodity-producing regions
- FPIC workflow management with stage-gate enforcement
- Compliance reporting for DDS, auditors, and certification schemes
- Integration with EUDR-001, EUDR-002, EUDR-006, EUDR-016, EUDR-017, EUDR-020

**Out of Scope (v1.0):**
- Direct community engagement platform (operators manage their own engagement; agent tracks records)
- Legal advisory on specific FPIC disputes (agent provides data and process; not legal counsel)
- Benefit-sharing payment processing (agent tracks agreements; does not handle funds)
- Real-time satellite monitoring of territory encroachment (defer to EUDR-020 integration)
- Mobile native application for field-level community engagement (web responsive only)
- Predictive ML models for rights violation forecasting (defer to Phase 2)
- Direct submission to EU Information System (handled by GL-EUDR-APP DDS module)
- Blockchain-based consent records (SHA-256 provenance hashes provide sufficient integrity)

### 5.2 Zero-Hallucination Principles

All 9 features in this agent operate under strict zero-hallucination guarantees:

| Principle | Implementation |
|-----------|---------------|
| **Deterministic calculations** | Same inputs always produce the same FPIC scores, overlap results, and risk assessments (bit-perfect reproducibility) |
| **No LLM in critical path** | All FPIC scoring, overlap detection, violation correlation, and compliance assessment use deterministic algorithms only |
| **Authoritative data sources only** | All territory boundaries sourced from LandMark, RAISG, FUNAI, BPN/AMAN, ACHPR, national registries; no synthetic boundaries |
| **Full provenance tracking** | Every FPIC score, overlap result, and compliance assessment includes SHA-256 hash, data source citations, and calculation timestamps |
| **Immutable audit trail** | All indigenous rights data changes recorded in `gl_eudr_irc_audit_log` with before/after values |
| **Decimal arithmetic** | FPIC scores and overlap percentages use Decimal type to prevent floating-point drift |
| **Version-controlled data** | Territory databases are versioned; any boundary update creates a new version with timestamp and source attribution |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core indigenous rights verification engine; Features 6-9 form the management, reporting, and integration layer.

**P0 Features 1-5: Core Indigenous Rights Verification Engine**

---

#### Feature 1: Indigenous Territory Database Integration

**User Story:**
```
As a compliance officer,
I want a comprehensive, georeferenced database of indigenous territories worldwide,
So that I can determine whether my supply chain production plots are located on or near indigenous lands.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F1.1: Integrates LandMark (Global Platform of Indigenous and Community Lands) territory boundaries covering 100+ countries, with spatial data in WGS84 coordinate reference system and GeoJSON/Shapefile format
- [ ] F1.2: Integrates RAISG (Amazon Georeferenced Socio-Environmental Information Network) indigenous territory data for the 9 Amazon Basin countries (Brazil, Bolivia, Colombia, Ecuador, French Guiana, Guyana, Peru, Suriname, Venezuela) with demarcation status tracking
- [ ] F1.3: Integrates FUNAI (Fundacao Nacional dos Povos Indigenas) officially demarcated indigenous land (Terras Indigenas) boundaries for Brazil, covering 700+ territories with legal status classification (homologated, declared, identified, under study)
- [ ] F1.4: Integrates BPN/AMAN (Aliansi Masyarakat Adat Nusantara) indigenous territory maps for Indonesia, covering 17,000+ Masyarakat Adat community territories across Kalimantan, Sumatra, Papua, and Sulawesi
- [ ] F1.5: Integrates ACHPR (African Commission on Human and Peoples' Rights) indigenous peoples' territory data for Sub-Saharan Africa, supplemented by national forest community databases (DRC, Cameroon, Ghana, Cote d'Ivoire)
- [ ] F1.6: Integrates national indigenous territory registries for Latin American producing countries (Colombia Resguardos, Peru Comunidades Nativas, Bolivia TCOs, Guatemala, Honduras, Paraguay)
- [ ] F1.7: Stores each territory with structured metadata: territory name, indigenous community/people name, country, administrative region, total area (hectares), legal recognition status (titled/claimed/customary/pending/disputed), date of legal recognition, governing authority, and data source with provenance hash
- [ ] F1.8: Maintains spatial index (PostGIS GIST) enabling sub-second bounding box queries across 50,000+ territory polygons; supports point-in-polygon and polygon-polygon intersection queries
- [ ] F1.9: Implements territory version control: every boundary update creates a new version with effective date, source attribution, change description, and previous boundary preserved for audit trail
- [ ] F1.10: Provides territory data freshness tracking with source update schedules, staleness alerts when data exceeds configurable age threshold (default: 12 months), and automated refresh triggers

**Non-Functional Requirements:**
- Coverage: 50,000+ territories across 100+ countries
- Spatial Precision: Territory boundaries stored with coordinates at 6+ decimal places
- Performance: Point-in-polygon query < 100ms; batch screening of 10,000 plots < 5 minutes
- Data Quality: Source attribution for every territory polygon; confidence scoring for boundary accuracy

**Dependencies:**
- LandMark API / data download for global coverage
- RAISG data portal for Amazonia
- FUNAI Geoserver for Brazil Terras Indigenas
- BPN/AMAN data sharing agreement for Indonesia
- AGENT-DATA-006 GIS/Mapping Connector for spatial operations
- PostGIS extension for spatial indexing and queries

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 GIS specialist)

**Edge Cases:**
- Overlapping territory claims (multiple communities claiming same area) -> Record all claims; flag as disputed; do not merge
- Territory without polygon boundary (point-only reference) -> Create circular buffer (configurable radius, default 10km); flag as approximate
- Territory spanning multiple countries -> Store with primary country; cross-reference in all affected countries
- Territory boundary dispute in progress -> Track dispute status; use most conservative (largest) boundary for compliance purposes

---

#### Feature 2: FPIC Documentation Verification Engine

**User Story:**
```
As an indigenous rights officer,
I want to verify that Free, Prior and Informed Consent documentation meets legal and procedural requirements,
So that I can confirm that commodity production activities on or near indigenous territories have obtained valid consent.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F2.1: Scores FPIC documentation completeness (0-100) using a deterministic 10-element checklist: community identification (10%), information disclosure adequacy (15%), prior timing verification (10%), consultation process documentation (15%), community representation verification (10%), consent/objection record (15%), absence of coercion evidence (10%), agreement documentation (5%), benefit-sharing terms (5%), ongoing monitoring provisions (5%)
- [ ] F2.2: Validates temporal compliance: consent must be obtained PRIOR to project commencement; system verifies that FPIC documentation date precedes production activity start date with configurable minimum lead time (default: 90 days)
- [ ] F2.3: Validates information disclosure adequacy: checks that disclosure documents cover project scope, environmental impact, social impact, economic impact, alternative options, and right to withhold consent, in a language accessible to the community
- [ ] F2.4: Validates community representation: verifies that consent was obtained from legitimate community representatives as recognized by the community's own governance structures, not imposed external representatives
- [ ] F2.5: Detects coercion indicators using rule-based analysis: consent obtained under time pressure (< 30 days from disclosure to consent), consent obtained during active conflict/protest, consent obtained without legal representation, consent obtained with conditional benefits
- [ ] F2.6: Classifies FPIC status for each supply chain plot: CONSENT_OBTAINED (score >= 80), CONSENT_PARTIAL (score 50-79), CONSENT_MISSING (score < 50), CONSENT_WITHDRAWN, CONSENT_DISPUTED, NOT_APPLICABLE (no indigenous territory overlap)
- [ ] F2.7: Supports country-specific FPIC validation rules: Brazil (FUNAI consultation protocol), Indonesia (PADIATAPA framework), Colombia (Decreto 1320/1998), Peru (Ley 29785), with configurable rule sets per jurisdiction
- [ ] F2.8: Tracks FPIC validity period and renewal requirements: consent agreements have configurable validity periods (default: 5 years); system generates renewal alerts at configurable lead times (default: 180, 90, 30 days before expiry)
- [ ] F2.9: Generates FPIC verification certificate with deterministic score, assessment methodology, data sources reviewed, compliance status, and SHA-256 provenance hash for audit trail
- [ ] F2.10: Maintains immutable history of all FPIC assessments per plot, enabling auditors to review score evolution, remediation actions, and consent status changes over time

**FPIC Scoring Formula:**
```
FPIC_Score = (
    community_identification * 0.10
    + information_disclosure * 0.15
    + prior_timing * 0.10
    + consultation_process * 0.15
    + community_representation * 0.10
    + consent_record * 0.15
    + absence_of_coercion * 0.10
    + agreement_documentation * 0.05
    + benefit_sharing * 0.05
    + monitoring_provisions * 0.05
)

Classification:
  CONSENT_OBTAINED:   FPIC_Score >= 80
  CONSENT_PARTIAL:    50 <= FPIC_Score < 80
  CONSENT_MISSING:    FPIC_Score < 50
```

**Non-Functional Requirements:**
- Accuracy: 100% deterministic scoring, bit-perfect reproducibility
- Performance: < 100ms per FPIC assessment
- Auditability: SHA-256 provenance hash on every assessment; full calculation breakdown available
- Jurisdiction Coverage: Country-specific rules for 8+ major producing countries

**Dependencies:**
- Feature 1 (Territory Database) for territory-plot linkage
- AGENT-DATA-001 PDF Invoice Extractor for FPIC document parsing
- AGENT-FOUND-005 Citations & Evidence Agent for source attribution

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- FPIC obtained from only a subset of affected communities -> Score proportionally; flag incomplete coverage
- Community governance structure disputed -> Flag as CONSENT_DISPUTED; require manual review
- Historical consent obtained before FPIC legal requirement enacted -> Accept with reduced confidence; flag for review
- Consent withdrawn after initial granting -> Immediately update status to CONSENT_WITHDRAWN; trigger alert

---

#### Feature 3: Land Rights Overlap Detector

**User Story:**
```
As a compliance officer,
I want to automatically detect when my supply chain production plots overlap with or are near indigenous territories,
So that I can identify plots requiring FPIC verification and enhanced due diligence.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F3.1: Performs spatial intersection analysis between production plot polygons/points and indigenous territory polygons using PostGIS ST_Intersects, ST_Within, and ST_DWithin functions
- [ ] F3.2: Classifies overlap type into 4 categories: DIRECT (plot centroid inside territory), PARTIAL (polygon intersection but centroid outside), ADJACENT (plot within configurable buffer, default 5km), PROXIMATE (plot within 25km of territory boundary)
- [ ] F3.3: Calculates overlap metrics: overlap area (hectares), overlap percentage of plot area, overlap percentage of territory area, minimum distance from plot to territory boundary (meters), and bearing from plot centroid to nearest territory boundary point
- [ ] F3.4: Identifies all affected indigenous communities for each overlap, returning community name, people/ethnic group, population (where available), legal recognition status, and representative organization
- [ ] F3.5: Supports batch overlap screening: upload 10,000+ plot coordinates/polygons and screen against entire territory database in single operation with progress tracking and partial results delivery
- [ ] F3.6: Supports configurable buffer zones for proximity analysis: operator-defined alert radius around indigenous territories (default: 5km inner buffer, 25km outer buffer) with configurable resolution (default: 64-point polygon approximation)
- [ ] F3.7: Generates overlap risk score (0-100) per plot using weighted factors: overlap type (40%), territory legal status (20%), community population (10%), historical conflict reports (15%), country indigenous rights framework strength (15%)
- [ ] F3.8: Cross-references overlap results with deforestation alerts from EUDR-020, flagging plots where both territory overlap AND post-cutoff deforestation are detected as CRITICAL compliance risk
- [ ] F3.9: Produces overlap analysis report per plot with map visualization (GeoJSON export), affected territory details, community information, FPIC requirement assessment, and recommended due diligence actions
- [ ] F3.10: Tracks overlap status changes over time: new territories recognized, boundary updates, overlap reclassification due to plot boundary changes, and generates change alerts for operators

**Overlap Risk Scoring Formula:**
```
Overlap_Risk_Score = (
    overlap_type_score * 0.40          # DIRECT=100, PARTIAL=80, ADJACENT=50, PROXIMATE=25
    + territory_legal_status * 0.20    # titled=100, declared=80, claimed=60, customary=50, pending=40
    + community_population_factor * 0.10  # normalized 0-100 based on population
    + conflict_history * 0.15          # 0-100 based on reported violations
    + country_rights_framework * 0.15  # 0-100 from EUDR-016 indigenous rights score
)

Classification:
  CRITICAL:  Overlap_Risk_Score >= 80
  HIGH:      60 <= Overlap_Risk_Score < 80
  MEDIUM:    40 <= Overlap_Risk_Score < 60
  LOW:       Overlap_Risk_Score < 40
  NONE:      No overlap detected
```

**Non-Functional Requirements:**
- Spatial Precision: Overlap calculations accurate to 1-meter resolution
- Performance: Single-plot query < 500ms; batch 10,000 plots < 5 minutes
- Determinism: Same plot-territory combination always produces identical overlap result
- Auditability: SHA-256 provenance hash on every overlap analysis

**Dependencies:**
- Feature 1 (Territory Database) for territory polygons
- AGENT-EUDR-002 Geolocation Verification Agent for plot coordinate validation
- AGENT-EUDR-006 Plot Boundary Manager Agent for plot polygon data
- PostGIS extension for spatial operations
- AGENT-DATA-006 GIS/Mapping Connector for spatial utilities

**Estimated Effort:** 3 weeks (1 backend engineer, 1 GIS specialist)

---

#### Feature 4: Community Consultation Tracker

**User Story:**
```
As an indigenous rights officer,
I want to track all community consultation activities with indigenous peoples in my supply chain regions,
So that I can demonstrate to auditors and regulators that proper engagement has taken place.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F4.1: Tracks consultation lifecycle through 7 stages: IDENTIFIED (community identified in supply chain region), NOTIFIED (initial contact and notification), INFORMATION_SHARED (project information disclosed), CONSULTATION_HELD (formal consultation conducted), RESPONSE_RECORDED (community response documented), AGREEMENT_REACHED (terms agreed or objection recorded), MONITORING_ACTIVE (ongoing compliance monitoring)
- [ ] F4.2: Records consultation meeting details: date, location, attendees (with roles -- community representatives, operator representatives, observers, government representatives), agenda, minutes, outcomes, and follow-up actions
- [ ] F4.3: Tracks community grievance mechanisms: grievance submission, acknowledgment, investigation, response, resolution, and appeal; with configurable SLA timelines (default: acknowledge 5 days, investigate 30 days, resolve 90 days)
- [ ] F4.4: Manages benefit-sharing agreements: agreement terms, monetary and non-monetary benefits, payment schedules, delivery tracking, community satisfaction assessments, and agreement renewal
- [ ] F4.5: Records community representative verification: name, role, mandate (elected/hereditary/appointed), community endorsement evidence, and legitimacy assessment
- [ ] F4.6: Tracks information disclosure completeness: which documents were shared (project description, environmental impact assessment, social impact assessment, economic analysis, alternatives analysis), in which languages, and community acknowledgment of receipt
- [ ] F4.7: Supports multi-stakeholder engagement: tracks parallel consultation with multiple communities affected by the same production activity, managing distinct consent processes per community
- [ ] F4.8: Generates consultation timeline visualization: Gantt-chart style view of consultation stages per community, highlighting overdue milestones and pending actions
- [ ] F4.9: Provides notification and reminder system: automated alerts for upcoming consultation milestones, overdue responses, agreement renewals, and grievance SLA deadlines
- [ ] F4.10: Exports consultation records in structured audit format: complete consultation dossier per community with chronological activity log, document inventory, and compliance assessment summary

**Non-Functional Requirements:**
- Completeness: 100% of consultation activities tracked with timestamped audit trail
- Multi-language: Support consultation records in EN, FR, DE, ES, PT, ID, and local languages (free-text fields)
- SLA Tracking: Minute-level precision for grievance and consultation deadlines
- Retention: All consultation records retained for minimum 5 years per EUDR Article 31

**Dependencies:**
- Feature 1 (Territory Database) for community identification
- Feature 3 (Overlap Detector) for identifying communities requiring consultation
- GL-EUDR-APP notification service for alerts and reminders

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

---

#### Feature 5: Rights Violation Alert System

**User Story:**
```
As a compliance officer,
I want to be automatically alerted when indigenous rights violations are reported in my supply chain sourcing regions,
So that I can trigger enhanced due diligence and assess whether my supply chain is affected.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F5.1: Monitors 10+ authoritative sources for indigenous rights violation reports: IWGIA (International Work Group for Indigenous Affairs), Cultural Survival, Forest Peoples Programme, Amnesty International, Human Rights Watch, national human rights commissions, IACHR (Inter-American Court of Human Rights), ACHPR, OHCHR (UN Office of the High Commissioner for Human Rights), and judicial/court databases
- [ ] F5.2: Ingests violation reports with structured metadata: report source, publication date, violation type (from controlled vocabulary: land_seizure, forced_displacement, fpic_violation, environmental_damage, physical_violence, cultural_destruction, restriction_of_access, benefit_sharing_breach, consultation_denial, discriminatory_policy), affected country, affected region, affected communities, and severity assessment
- [ ] F5.3: Correlates violation reports with operator supply chain locations using spatial proximity matching: violation location (point or region) compared against operator's plot locations and territory overlap records
- [ ] F5.4: Classifies violation alert severity using weighted scoring: violation type severity (30%), spatial proximity to supply chain (25%), community population affected (15%), legal framework strength in country (15%), media coverage intensity (15%). Severity levels: CRITICAL (>= 80), HIGH (>= 60), MEDIUM (>= 40), LOW (< 40)
- [ ] F5.5: Generates supply chain impact assessment for each violation alert: lists affected suppliers, plots, commodities, products, and estimated supply chain exposure (% of volume from affected region)
- [ ] F5.6: Triggers enhanced due diligence requirements when violation alert severity reaches HIGH or CRITICAL and supply chain correlation is confirmed (proximity <= 25km of operator's plot)
- [ ] F5.7: Maintains violation history database with trend analysis: violation frequency by country, region, violation type, and commodity; trend direction (improving/stable/deteriorating) calculated on 3-year rolling window
- [ ] F5.8: Supports operator-configurable alert rules: custom severity thresholds, geographic focus areas, commodity filters, and notification channel preferences (email, webhook, in-app notification)
- [ ] F5.9: Implements 7-day deduplication window: multiple reports of the same violation event from different sources are consolidated into a single alert with source cross-references
- [ ] F5.10: Provides violation report provenance tracking: every alert includes source URLs, publication dates, original language, and SHA-256 hash of source document for audit verification

**Violation Severity Scoring Formula:**
```
Violation_Severity = (
    violation_type_score * 0.30        # land_seizure=100, forced_displacement=100, physical_violence=100, fpic_violation=80, etc.
    + spatial_proximity * 0.25         # within_plot=100, adjacent_5km=80, proximate_25km=50, region=30, country=10
    + community_population * 0.15      # normalized 0-100
    + legal_framework_gap * 0.15       # 100 - country indigenous rights score
    + media_coverage * 0.15            # based on source count and prominence
)

Classification:
  CRITICAL:  Violation_Severity >= 80
  HIGH:      60 <= Violation_Severity < 80
  MEDIUM:    40 <= Violation_Severity < 60
  LOW:       Violation_Severity < 40
```

**Non-Functional Requirements:**
- Latency: Violation reports ingested within 48 hours of publication
- Coverage: 10+ authoritative sources monitored continuously
- Deduplication: > 90% duplicate elimination within 7-day window
- Determinism: Severity scoring is deterministic, bit-perfect reproducible
- Retention: All violation records retained for minimum 5 years

**Dependencies:**
- Feature 1 (Territory Database) for territory location correlation
- Feature 3 (Overlap Detector) for supply chain spatial correlation
- AGENT-EUDR-020 Deforestation Alert System for deforestation-violation cross-referencing
- AGENT-DATA-004 API Gateway Agent for external source data ingestion

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

**P0 Features 6-9: Management, Reporting, and Integration Layer**

> Features 6, 7, 8, and 9 are P0 launch blockers. Without community registry, FPIC workflow management, compliance reporting, and agent ecosystem integration, the core verification engine cannot deliver end-user value. These features are the delivery mechanism through which indigenous rights officers, compliance officers, auditors, and the broader EUDR platform interact with the verification engine.

---

#### Feature 6: Indigenous Community Registry

**User Story:**
```
As a compliance officer,
I want a structured registry of indigenous communities present in my sourcing regions,
So that I can identify which communities may be affected by my supply chain and understand their rights and protections.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F6.1: Maintains structured records for indigenous communities in EUDR commodity-producing regions: community name (in official and indigenous language), people/ethnic group, language(s), estimated population, country, administrative region, territory reference (linked to F1), and data source citation
- [ ] F6.2: Tracks legal recognition status per community: constitutionally recognized, statutory recognition, customary recognition only, recognition pending, recognition denied/disputed
- [ ] F6.3: Records applicable legal protections per community: ILO 169 coverage (country ratified), national indigenous rights legislation, FPIC legal requirement, land tenure security level, and judicial protection mechanisms
- [ ] F6.4: Lists representative organizations per community: indigenous peoples' organization (IPO) name, leadership contacts (with privacy controls), mandate scope, and external representation capacity
- [ ] F6.5: Links communities to EUDR commodity production relevance: which commodities are produced on or near the community's territory, which supply chain operators source from the region, and estimated production overlap
- [ ] F6.6: Tracks community engagement history: past consultations, FPIC processes, grievances filed, agreements in force, and satisfaction assessments
- [ ] F6.7: Supports community data sovereignty: communities can request data corrections, updates, or access restrictions through designated contacts; all changes tracked with requester attribution
- [ ] F6.8: Provides community profile reports: exportable community profiles with territory map, legal status, commodity relevance, engagement history, and compliance requirements
- [ ] F6.9: Implements privacy controls: personally identifiable community contact information is encrypted at rest (AES-256) and accessible only to users with explicit `eudr-irc:community:contacts` permission
- [ ] F6.10: Supports search and filtering: find communities by country, region, commodity relevance, legal status, population range, and engagement status

**Non-Functional Requirements:**
- Coverage: Communities in all EUDR commodity-producing regions with indigenous presence
- Privacy: PII encrypted at rest (AES-256); access logged in audit trail
- Performance: Community search < 200ms with filtering
- Data Sovereignty: Community correction requests processed within 30 days

**Dependencies:**
- Feature 1 (Territory Database) for territory-community linkage
- SEC-003 Encryption at Rest for PII protection
- SEC-011 PII Detection/Redaction for privacy compliance

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 7: FPIC Workflow Management

**User Story:**
```
As an indigenous rights officer,
I want a structured, stage-gated workflow to manage FPIC processes with indigenous communities,
So that I can ensure procedural compliance, meet timelines, and maintain audit-ready records.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F7.1: Implements 7-stage FPIC workflow with stage-gate enforcement: IDENTIFICATION -> INFORMATION_DISCLOSURE -> CONSULTATION -> CONSENT_DECISION -> AGREEMENT -> IMPLEMENTATION -> MONITORING; each stage must be completed before proceeding to next
- [ ] F7.2: Defines configurable SLA timelines per stage: Identification (14 days), Information Disclosure (30 days), Consultation (60 days), Consent Decision (30 days), Agreement (30 days), Implementation (ongoing), Monitoring (ongoing with annual review)
- [ ] F7.3: Tracks SLA compliance per FPIC process: percentage of stages completed within SLA, average stage duration, overdue stage alerts, and escalation triggers (Level 1: 7 days overdue, Level 2: 14 days, Level 3: 30 days)
- [ ] F7.4: Provides FPIC documentation templates in 8 languages (English, French, German, Spanish, Portuguese, Indonesian, Quechua, Swahili): project information sheet, community notification letter, consultation agenda template, consent form, objection form, benefit-sharing agreement template, and monitoring checklist
- [ ] F7.5: Manages parallel FPIC processes: when a production activity affects multiple communities, each community has an independent FPIC workflow with its own timeline, consent decision, and agreement
- [ ] F7.6: Supports consent withdrawal: at any stage, a community can withdraw previously granted consent; system immediately updates FPIC status to CONSENT_WITHDRAWN, triggers operator alert, and initiates remediation workflow
- [ ] F7.7: Records all FPIC workflow state transitions with timestamp, actor (operator representative, community representative, system), reason, supporting evidence, and SHA-256 provenance hash
- [ ] F7.8: Generates FPIC progress dashboard: visual pipeline view showing all active FPIC processes, their current stage, SLA status (on-track/at-risk/overdue), and next required action
- [ ] F7.9: Integrates with Feature 2 (FPIC Verification Engine): when FPIC workflow reaches CONSENT_DECISION stage, automatically triggers FPIC documentation verification and generates completeness score
- [ ] F7.10: Supports FPIC renewal: tracks consent validity periods (configurable, default 5 years) and triggers renewal workflow 180 days before expiry

**FPIC Workflow State Machine:**
```
IDENTIFICATION -> INFORMATION_DISCLOSURE -> CONSULTATION -> CONSENT_DECISION
                                                              |
                                              +---------------+--------------+
                                              |               |              |
                                        CONSENT_GRANTED  CONSENT_DENIED  CONSENT_DEFERRED
                                              |               |              |
                                          AGREEMENT    REMEDIATION_REQUIRED  CONSULTATION
                                              |                              (return)
                                        IMPLEMENTATION
                                              |
                                        MONITORING
                                              |
                                    (RENEWAL at expiry)

At any stage: -> CONSENT_WITHDRAWN (community-initiated)
```

**Non-Functional Requirements:**
- State Machine: Invalid transitions rejected with descriptive error
- SLA Tracking: Minute-level precision for all deadlines
- Concurrency: Support 1,000+ concurrent FPIC processes
- Auditability: Complete state transition history per FPIC process

**Dependencies:**
- Feature 2 (FPIC Verification Engine) for automated scoring at consent stage
- Feature 4 (Community Consultation Tracker) for consultation records
- Feature 6 (Community Registry) for community identification
- GL-EUDR-APP notification service for SLA alerts

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 8: Compliance Reporting and Documentation

**User Story:**
```
As a compliance officer,
I want automated, audit-ready indigenous rights compliance reports,
So that I can include them in my Due Diligence Statement and present them to auditors, certifiers, and regulators.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F8.1: Generates comprehensive indigenous rights compliance reports in PDF, JSON, and HTML formats, including territory overlap analysis, FPIC status summary, community consultation records, violation screening results, and compliance risk assessment
- [ ] F8.2: Generates DDS-integrated indigenous rights section: structured risk assessment narrative for EUDR Article 4(2) submission, covering indigenous rights verification methodology, territory screening results, FPIC compliance status, and mitigation measures
- [ ] F8.3: Produces certification scheme compliance reports: FSC FPIC compliance evidence (Principle 3 and 4), RSPO FPIC standard compliance (Principle 7.5/7.6), Rainforest Alliance SAN Standard indigenous rights requirements, formatted per scheme-specific audit checklists
- [ ] F8.4: Generates supplier-level indigenous rights scorecards: per-supplier summary of territory overlap count, FPIC compliance rate, violation exposure, community engagement score, and overall indigenous rights risk rating
- [ ] F8.5: Produces trend reports showing indigenous rights compliance evolution over time per supply chain, commodity, or country, with annotated events (new territory recognition, violation reports, FPIC completions)
- [ ] F8.6: Creates executive summary reports with key indicators: total plots screened, territory overlap count by severity, FPIC compliance rate, active violations, community engagement coverage, and overall indigenous rights readiness score
- [ ] F8.7: Exports dashboard data packages for BI integration (Grafana, Tableau, Power BI) in CSV, JSON, and XLSX formats with standardized schema for indigenous rights metrics
- [ ] F8.8: Includes complete audit trail in reports: every overlap analysis, FPIC score, and violation correlation linked to source data, calculation method, and SHA-256 provenance hash
- [ ] F8.9: Supports multi-language report generation in English (EN), French (FR), German (DE), Spanish (ES), and Portuguese (PT) using translated templates with jurisdiction-appropriate legal terminology
- [ ] F8.10: Maintains report generation history with version control: reports are immutable once generated; updated assessments produce new report versions, not overwrites

**Non-Functional Requirements:**
- Performance: PDF generation < 10 seconds per supply chain compliance report
- Quality: Reports pass WCAG 2.1 AA accessibility standards
- Size: PDF reports optimized to < 10 MB each
- Formats: PDF (audit), JSON (API integration), HTML (web display), CSV/XLSX (BI export)

**Dependencies:**
- Features 1-7 for all indigenous rights assessment data
- Report generation library (WeasyPrint for PDF)
- Jinja2 templates for multi-format output
- i18n framework for multi-language support
- S3 for report storage

**Estimated Effort:** 3 weeks (1 backend engineer, 1 template engineer)

---

#### Feature 9: Integration with Existing EUDR Agents

**User Story:**
```
As the GreenLang platform,
I want the Indigenous Rights Checker to enrich existing EUDR agents with indigenous rights data,
So that indigenous rights compliance is embedded throughout the supply chain compliance workflow.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F9.1: Integrates with EUDR-001 (Supply Chain Mapping Master): enriches supply chain graph nodes with indigenous rights risk attributes (territory overlap status, FPIC status, violation exposure) for risk propagation through the supply chain
- [ ] F9.2: Integrates with EUDR-002 (Geolocation Verification Agent): validates plot coordinates against indigenous territory boundaries as part of geolocation verification; flags plots requiring indigenous rights review
- [ ] F9.3: Integrates with EUDR-006 (Plot Boundary Manager Agent): consumes plot polygon boundaries for precise overlap analysis; receives plot boundary update notifications for re-screening
- [ ] F9.4: Integrates with EUDR-016 (Country Risk Evaluator): provides per-country indigenous rights protection scores that feed the governance index engine (Feature 4, criterion F4.7 indigenous peoples' rights recognition); receives country risk classification for indigenous rights scoring context
- [ ] F9.5: Integrates with EUDR-017 (Supplier Risk Scorer): provides per-supplier indigenous rights risk scores; suppliers with unresolved territory overlaps or missing FPIC receive elevated risk scores
- [ ] F9.6: Integrates with EUDR-020 (Deforestation Alert System): receives deforestation alerts and correlates with indigenous territory proximity; deforestation alerts within indigenous territories generate CRITICAL indigenous rights alerts
- [ ] F9.7: Provides standardized indigenous rights data API consumed by GL-EUDR-APP v1.0: territory overlay for map visualization, FPIC status dashboard, violation alert feed, and compliance report download
- [ ] F9.8: Publishes indigenous rights events to the GreenLang event bus for consumption by other agents: territory_overlap_detected, fpic_status_changed, violation_alert_generated, consultation_milestone_reached, compliance_report_generated
- [ ] F9.9: Supports webhook notifications for external system integration: configurable webhooks triggered on indigenous rights events (overlap detection, FPIC status change, violation alert) with retry logic and delivery confirmation
- [ ] F9.10: Maintains integration health monitoring: tracks message delivery rates, API response times, and error rates for all integration points; exposes integration health metrics via Prometheus

**Non-Functional Requirements:**
- Latency: Integration API responses < 200ms p95
- Reliability: 99.9% message delivery for event bus publications
- Compatibility: RESTful API with OpenAPI 3.0 specification
- Versioning: API versioned (v1) with backward compatibility guarantee

**Dependencies:**
- AGENT-EUDR-001 Supply Chain Mapping Master (BUILT 100%)
- AGENT-EUDR-002 Geolocation Verification Agent (BUILT 100%)
- AGENT-EUDR-006 Plot Boundary Manager Agent (BUILT 100%)
- AGENT-EUDR-016 Country Risk Evaluator (BUILT 100%)
- AGENT-EUDR-017 Supplier Risk Scorer (BUILT 100%)
- AGENT-EUDR-020 Deforestation Alert System (BUILT 100%)
- GL-EUDR-APP v1.0 Platform (BUILT 100%)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Indigenous Rights Impact Assessment Dashboard
- Interactive map showing indigenous territories overlaid with supply chain plots
- Heat map visualization of indigenous rights risk across sourcing regions
- Community engagement coverage indicators per territory
- Benefit-sharing agreement compliance tracking with financial metrics

#### Feature 11: Community Voice Integration
- Direct feedback channel for indigenous communities to report compliance concerns
- Community satisfaction surveys integrated into monitoring stage
- Anonymous grievance submission portal with case tracking
- Community-verified territory boundary updates

#### Feature 12: Predictive Encroachment Analysis
- Machine learning model to predict areas at risk of indigenous territory encroachment
- Correlation analysis between commodity price changes and encroachment events
- Early warning system for emerging indigenous rights conflicts
- Climate-driven displacement risk modeling

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Direct community engagement platform (operators manage engagement; agent tracks records)
- Legal advisory or dispute resolution services (agent provides data; not legal counsel)
- Benefit-sharing payment processing (agent tracks agreements; does not handle funds)
- Mobile native application for field community engagement (web responsive only)
- Predictive ML models for rights violation forecasting (defer to Phase 2)
- Real-time satellite monitoring of territory encroachment (defer to EUDR-020 integration)
- Blockchain-based consent ledger (SHA-256 provenance hashes sufficient for v1.0)
- Automated supplier deselection recommendations (legal liability concern)

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
| AGENT-EUDR-021        |           | AGENT-EUDR-001            |           | AGENT-EUDR-016        |
| Indigenous Rights     |<--------->| Supply Chain Mapping      |<--------->| Country Risk          |
| Checker               |           | Master                    |           | Evaluator             |
|                       |           |                           |           |                       |
| - Territory Database  |           | - Graph Engine            |           | - Risk Scoring Engine |
| - FPIC Verifier       |           | - Risk Propagation        |           | - Governance Engine   |
| - Overlap Detector    |           | - Gap Analysis            |           | - Hotspot Detector    |
| - Consultation Track. |           |                           |           |                       |
| - Violation Alerts    |           +---------------------------+           +-----------------------+
| - Community Registry  |
| - FPIC Workflow       |           +---------------------------+           +---------------------------+
| - Compliance Reports  |           | AGENT-EUDR-006            |           | AGENT-EUDR-020            |
| - Agent Integration   |           | Plot Boundary Manager     |           | Deforestation Alert       |
+-----------+-----------+           |                           |           | System                    |
            |                       | - Plot Polygons           |           | - Satellite Detection     |
            |                       | - Boundary Validation     |           | - Buffer Monitoring       |
+-----------v-----------+           +---------------------------+           +---------------------------+
| Indigenous Rights     |
| Data Sources          |           +---------------------------+           +---------------------------+
|                       |           | AGENT-EUDR-002            |           | AGENT-EUDR-017            |
| - LandMark            |           | Geolocation Verification  |           | Supplier Risk Scorer      |
| - RAISG               |           |                           |           |                           |
| - FUNAI               |           | - Coordinate Validation   |           | - Supplier Risk Scoring   |
| - BPN/AMAN            |           +---------------------------+           +---------------------------+
| - ACHPR               |
| - National Registries |
+-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/indigenous_rights_checker/
    __init__.py                              # Public API exports
    config.py                                # IndigenousRightsCheckerConfig with GL_EUDR_IRC_ env prefix
    models.py                                # Pydantic v2 models for indigenous rights data
    territory_database.py                    # TerritoryDatabaseEngine: territory data management (F1)
    fpic_verifier.py                         # FPICVerificationEngine: FPIC documentation scoring (F2)
    overlap_detector.py                      # LandRightsOverlapDetector: spatial overlap analysis (F3)
    consultation_tracker.py                  # CommunityConsultationTracker: consultation lifecycle (F4)
    violation_alert_engine.py                # ViolationAlertEngine: rights violation monitoring (F5)
    community_registry.py                    # IndigenousCommunityRegistry: community data management (F6)
    fpic_workflow_engine.py                  # FPICWorkflowEngine: stage-gated FPIC process (F7)
    compliance_reporter.py                   # ComplianceReporter: report generation (F8)
    agent_integrator.py                      # AgentIntegrator: cross-agent integration (F9)
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains
    metrics.py                               # 20 Prometheus self-monitoring metrics
    setup.py                                 # IndigenousRightsCheckerService facade (singleton, 9 engines)
    reference_data/
        __init__.py
        territory_sources.py                 # Data source specifications and ingestion configs
        fpic_legal_requirements.py           # Per-country FPIC legal framework database
        violation_sources.py                 # Violation monitoring source configurations
        indigenous_rights_scores.py          # Per-country indigenous rights protection scores
        community_database.py                # Seed data for indigenous community registry
    api/
        __init__.py
        router.py                            # FastAPI router (~35 endpoints)
        schemas.py                           # API request/response Pydantic schemas
        dependencies.py                      # FastAPI dependency injection
        territory_routes.py                  # Territory database endpoints
        fpic_routes.py                       # FPIC verification endpoints
        overlap_routes.py                    # Overlap detection endpoints
        consultation_routes.py               # Consultation tracking endpoints
        violation_routes.py                  # Violation alert endpoints
        community_routes.py                  # Community registry endpoints
        workflow_routes.py                   # FPIC workflow endpoints
        report_routes.py                     # Compliance reporting endpoints
        integration_routes.py                # Agent integration endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Territory Legal Status
class TerritoryLegalStatus(str, Enum):
    TITLED = "titled"                # Formally titled/homologated
    DECLARED = "declared"            # Officially declared but not yet titled
    CLAIMED = "claimed"              # Claimed by community, under review
    CUSTOMARY = "customary"          # Customary use without formal recognition
    PENDING = "pending"              # Recognition process in progress
    DISPUTED = "disputed"            # Boundary or ownership disputed

# FPIC Status
class FPICStatus(str, Enum):
    CONSENT_OBTAINED = "consent_obtained"      # Score >= 80
    CONSENT_PARTIAL = "consent_partial"        # Score 50-79
    CONSENT_MISSING = "consent_missing"        # Score < 50
    CONSENT_WITHDRAWN = "consent_withdrawn"    # Community withdrew consent
    CONSENT_DISPUTED = "consent_disputed"      # Consent validity contested
    NOT_APPLICABLE = "not_applicable"          # No indigenous territory overlap

# Overlap Type
class OverlapType(str, Enum):
    DIRECT = "direct"                # Plot centroid inside territory
    PARTIAL = "partial"              # Polygon intersection, centroid outside
    ADJACENT = "adjacent"            # Within 5km buffer
    PROXIMATE = "proximate"          # Within 25km buffer
    NONE = "none"                    # No overlap detected

# FPIC Workflow Stage
class FPICWorkflowStage(str, Enum):
    IDENTIFICATION = "identification"
    INFORMATION_DISCLOSURE = "information_disclosure"
    CONSULTATION = "consultation"
    CONSENT_DECISION = "consent_decision"
    AGREEMENT = "agreement"
    IMPLEMENTATION = "implementation"
    MONITORING = "monitoring"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    CONSENT_DENIED = "consent_denied"

# Violation Type
class ViolationType(str, Enum):
    LAND_SEIZURE = "land_seizure"
    FORCED_DISPLACEMENT = "forced_displacement"
    FPIC_VIOLATION = "fpic_violation"
    ENVIRONMENTAL_DAMAGE = "environmental_damage"
    PHYSICAL_VIOLENCE = "physical_violence"
    CULTURAL_DESTRUCTION = "cultural_destruction"
    RESTRICTION_OF_ACCESS = "restriction_of_access"
    BENEFIT_SHARING_BREACH = "benefit_sharing_breach"
    CONSULTATION_DENIAL = "consultation_denial"
    DISCRIMINATORY_POLICY = "discriminatory_policy"

# Indigenous Territory
class IndigenousTerritory(BaseModel):
    territory_id: str                        # UUID
    territory_name: str                      # Official name
    indigenous_name: Optional[str]           # Name in indigenous language
    people_name: str                         # Indigenous people/ethnic group
    country_code: str                        # ISO 3166-1 alpha-2
    region: str                              # Administrative region
    area_hectares: Decimal                   # Total area
    legal_status: TerritoryLegalStatus       # Legal recognition status
    recognition_date: Optional[date]         # Date of formal recognition
    governing_authority: Optional[str]       # National authority responsible
    boundary_geojson: Dict                   # GeoJSON polygon/multipolygon
    data_source: str                         # LandMark/RAISG/FUNAI/BPN/ACHPR/National
    source_url: Optional[str]               # Source data URL
    confidence: ConfidenceLevel              # Boundary accuracy confidence
    version: int                             # Data version
    provenance_hash: str                     # SHA-256
    last_verified: datetime
    created_at: datetime

# FPIC Assessment
class FPICAssessment(BaseModel):
    assessment_id: str
    plot_id: str                             # Production plot assessed
    territory_id: str                        # Overlapping territory
    community_id: str                        # Affected community
    fpic_score: Decimal                      # 0-100 composite score
    fpic_status: FPICStatus
    element_scores: Dict[str, Decimal]       # 10-element breakdown
    country_specific_rules: str              # Jurisdiction applied
    temporal_compliance: bool                # Consent prior to production
    coercion_flags: List[str]                # Detected coercion indicators
    validity_start: Optional[date]           # Consent validity start
    validity_end: Optional[date]             # Consent expiry
    decision_rationale: str
    provenance_hash: str
    assessed_at: datetime
    version: int

# Territory Overlap Result
class TerritoryOverlap(BaseModel):
    overlap_id: str
    plot_id: str                             # Production plot
    territory_id: str                        # Overlapping territory
    overlap_type: OverlapType
    overlap_area_hectares: Optional[Decimal] # For DIRECT/PARTIAL
    overlap_pct_of_plot: Optional[Decimal]   # % of plot area
    overlap_pct_of_territory: Optional[Decimal]
    distance_meters: Decimal                 # Min distance to territory
    bearing_degrees: Optional[Decimal]       # Direction to nearest boundary
    affected_communities: List[str]          # Community IDs
    risk_score: Decimal                      # 0-100 overlap risk
    risk_level: str                          # CRITICAL/HIGH/MEDIUM/LOW/NONE
    deforestation_correlation: bool          # Cross-ref with EUDR-020
    provenance_hash: str
    detected_at: datetime

# FPIC Workflow Instance
class FPICWorkflow(BaseModel):
    workflow_id: str
    plot_id: str
    territory_id: str
    community_id: str
    current_stage: FPICWorkflowStage
    stage_history: List[Dict]               # [{stage, entered_at, completed_at, actor, notes}]
    sla_status: str                         # on_track/at_risk/overdue
    next_deadline: Optional[datetime]
    consent_decision: Optional[str]         # granted/denied/deferred/withdrawn
    agreement_id: Optional[str]
    validity_end: Optional[date]
    provenance_hash: str
    created_at: datetime
    updated_at: datetime

# Violation Alert
class ViolationAlert(BaseModel):
    alert_id: str
    source: str                             # IWGIA/Cultural Survival/etc.
    source_url: str
    publication_date: date
    violation_type: ViolationType
    country_code: str
    region: Optional[str]
    location_lat: Optional[float]
    location_lon: Optional[float]
    affected_communities: List[str]
    severity_score: Decimal                 # 0-100
    severity_level: str                     # CRITICAL/HIGH/MEDIUM/LOW
    supply_chain_correlation: bool          # Matches operator supply chain
    affected_plots: List[str]              # Correlated plot IDs
    affected_suppliers: List[str]          # Correlated supplier IDs
    impact_assessment: Optional[Dict]
    deduplication_group: Optional[str]     # Group ID for duplicate reports
    provenance_hash: str
    detected_at: datetime
```

### 7.4 Database Schema (New Migration: V109)

```sql
-- =========================================================================
-- V109: AGENT-EUDR-021 Indigenous Rights Checker Schema
-- Agent: GL-EUDR-IRC-021
-- Tables: 14 (4 hypertables)
-- Prefix: gl_eudr_irc_
-- =========================================================================

CREATE SCHEMA IF NOT EXISTS eudr_indigenous_rights_checker;

-- 1. Indigenous Territories (spatial data with PostGIS)
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_territories (
    territory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_name VARCHAR(500) NOT NULL,
    indigenous_name VARCHAR(500),
    people_name VARCHAR(500) NOT NULL,
    country_code CHAR(2) NOT NULL,
    region VARCHAR(200),
    area_hectares NUMERIC(14,2),
    legal_status VARCHAR(30) NOT NULL CHECK (legal_status IN ('titled', 'declared', 'claimed', 'customary', 'pending', 'disputed')),
    recognition_date DATE,
    governing_authority VARCHAR(500),
    boundary_geom GEOMETRY(MultiPolygon, 4326),
    boundary_geojson JSONB,
    data_source VARCHAR(200) NOT NULL,
    source_url VARCHAR(2000),
    confidence VARCHAR(10) NOT NULL DEFAULT 'medium' CHECK (confidence IN ('high', 'medium', 'low')),
    version INTEGER DEFAULT 1,
    provenance_hash VARCHAR(64) NOT NULL,
    last_verified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_irc_territories_country ON eudr_indigenous_rights_checker.gl_eudr_irc_territories(country_code);
CREATE INDEX idx_irc_territories_status ON eudr_indigenous_rights_checker.gl_eudr_irc_territories(legal_status);
CREATE INDEX idx_irc_territories_people ON eudr_indigenous_rights_checker.gl_eudr_irc_territories(people_name);
CREATE INDEX idx_irc_territories_source ON eudr_indigenous_rights_checker.gl_eudr_irc_territories(data_source);
CREATE INDEX idx_irc_territories_geom ON eudr_indigenous_rights_checker.gl_eudr_irc_territories USING GIST (boundary_geom);

-- 2. FPIC Assessments (hypertable on assessed_at)
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_fpic_assessments (
    assessment_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    territory_id UUID NOT NULL,
    community_id UUID,
    fpic_score NUMERIC(5,2) NOT NULL CHECK (fpic_score >= 0 AND fpic_score <= 100),
    fpic_status VARCHAR(30) NOT NULL CHECK (fpic_status IN ('consent_obtained', 'consent_partial', 'consent_missing', 'consent_withdrawn', 'consent_disputed', 'not_applicable')),
    community_identification_score NUMERIC(5,2) DEFAULT 0,
    information_disclosure_score NUMERIC(5,2) DEFAULT 0,
    prior_timing_score NUMERIC(5,2) DEFAULT 0,
    consultation_process_score NUMERIC(5,2) DEFAULT 0,
    community_representation_score NUMERIC(5,2) DEFAULT 0,
    consent_record_score NUMERIC(5,2) DEFAULT 0,
    absence_of_coercion_score NUMERIC(5,2) DEFAULT 0,
    agreement_documentation_score NUMERIC(5,2) DEFAULT 0,
    benefit_sharing_score NUMERIC(5,2) DEFAULT 0,
    monitoring_provisions_score NUMERIC(5,2) DEFAULT 0,
    country_specific_rules VARCHAR(100),
    temporal_compliance BOOLEAN DEFAULT FALSE,
    coercion_flags JSONB DEFAULT '[]',
    validity_start DATE,
    validity_end DATE,
    decision_rationale TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    version INTEGER DEFAULT 1,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (assessment_id, assessed_at)
);

SELECT create_hypertable('eudr_indigenous_rights_checker.gl_eudr_irc_fpic_assessments', 'assessed_at');

-- 3. Territory Overlap Results
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_overlaps (
    overlap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    territory_id UUID NOT NULL REFERENCES eudr_indigenous_rights_checker.gl_eudr_irc_territories(territory_id),
    overlap_type VARCHAR(20) NOT NULL CHECK (overlap_type IN ('direct', 'partial', 'adjacent', 'proximate', 'none')),
    overlap_area_hectares NUMERIC(14,2),
    overlap_pct_of_plot NUMERIC(5,2),
    overlap_pct_of_territory NUMERIC(7,4),
    distance_meters NUMERIC(12,2) NOT NULL,
    bearing_degrees NUMERIC(5,2),
    affected_communities JSONB DEFAULT '[]',
    risk_score NUMERIC(5,2) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('critical', 'high', 'medium', 'low', 'none')),
    deforestation_correlation BOOLEAN DEFAULT FALSE,
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (plot_id, territory_id)
);

CREATE INDEX idx_irc_overlaps_plot ON eudr_indigenous_rights_checker.gl_eudr_irc_overlaps(plot_id);
CREATE INDEX idx_irc_overlaps_territory ON eudr_indigenous_rights_checker.gl_eudr_irc_overlaps(territory_id);
CREATE INDEX idx_irc_overlaps_type ON eudr_indigenous_rights_checker.gl_eudr_irc_overlaps(overlap_type);
CREATE INDEX idx_irc_overlaps_risk ON eudr_indigenous_rights_checker.gl_eudr_irc_overlaps(risk_level);
CREATE INDEX idx_irc_overlaps_deforestation ON eudr_indigenous_rights_checker.gl_eudr_irc_overlaps(deforestation_correlation) WHERE deforestation_correlation = TRUE;

-- 4. Indigenous Communities
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_communities (
    community_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_name VARCHAR(500) NOT NULL,
    indigenous_name VARCHAR(500),
    people_name VARCHAR(500) NOT NULL,
    language VARCHAR(200),
    estimated_population INTEGER,
    country_code CHAR(2) NOT NULL,
    region VARCHAR(200),
    territory_ids JSONB DEFAULT '[]',
    legal_recognition_status VARCHAR(50),
    applicable_legal_protections JSONB DEFAULT '[]',
    ilo_169_coverage BOOLEAN DEFAULT FALSE,
    fpic_legal_requirement BOOLEAN DEFAULT FALSE,
    representative_organizations JSONB DEFAULT '[]',
    contact_channels JSONB DEFAULT '[]',
    commodity_relevance JSONB DEFAULT '[]',
    engagement_history_summary JSONB DEFAULT '{}',
    data_source VARCHAR(200),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_irc_communities_country ON eudr_indigenous_rights_checker.gl_eudr_irc_communities(country_code);
CREATE INDEX idx_irc_communities_people ON eudr_indigenous_rights_checker.gl_eudr_irc_communities(people_name);
CREATE INDEX idx_irc_communities_ilo ON eudr_indigenous_rights_checker.gl_eudr_irc_communities(ilo_169_coverage);

-- 5. Community Consultation Records
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_consultations (
    consultation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_id UUID NOT NULL REFERENCES eudr_indigenous_rights_checker.gl_eudr_irc_communities(community_id),
    plot_id UUID,
    territory_id UUID,
    consultation_stage VARCHAR(30) NOT NULL,
    meeting_date DATE,
    meeting_location VARCHAR(500),
    attendees JSONB DEFAULT '[]',
    agenda TEXT,
    minutes TEXT,
    outcomes TEXT,
    follow_up_actions JSONB DEFAULT '[]',
    documents_shared JSONB DEFAULT '[]',
    community_response VARCHAR(100),
    grievance_id UUID,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_irc_consultations_community ON eudr_indigenous_rights_checker.gl_eudr_irc_consultations(community_id);
CREATE INDEX idx_irc_consultations_stage ON eudr_indigenous_rights_checker.gl_eudr_irc_consultations(consultation_stage);

-- 6. Grievance Records
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_grievances (
    grievance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_id UUID NOT NULL,
    territory_id UUID,
    grievance_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    status VARCHAR(30) NOT NULL DEFAULT 'submitted' CHECK (status IN ('submitted', 'acknowledged', 'investigating', 'responded', 'resolved', 'appealed', 'closed')),
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    investigation_deadline TIMESTAMPTZ,
    resolution_deadline TIMESTAMPTZ,
    response TEXT,
    resolution TEXT,
    resolved_at TIMESTAMPTZ,
    sla_compliant BOOLEAN,
    provenance_hash VARCHAR(64) NOT NULL
);

CREATE INDEX idx_irc_grievances_community ON eudr_indigenous_rights_checker.gl_eudr_irc_grievances(community_id);
CREATE INDEX idx_irc_grievances_status ON eudr_indigenous_rights_checker.gl_eudr_irc_grievances(status);
CREATE INDEX idx_irc_grievances_severity ON eudr_indigenous_rights_checker.gl_eudr_irc_grievances(severity);

-- 7. Benefit-Sharing Agreements
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_agreements (
    agreement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    community_id UUID NOT NULL,
    territory_id UUID,
    operator_id UUID NOT NULL,
    agreement_type VARCHAR(100) NOT NULL,
    terms_summary TEXT NOT NULL,
    monetary_benefits JSONB DEFAULT '{}',
    non_monetary_benefits JSONB DEFAULT '[]',
    effective_date DATE NOT NULL,
    expiry_date DATE,
    renewal_required BOOLEAN DEFAULT TRUE,
    status VARCHAR(30) NOT NULL DEFAULT 'active' CHECK (status IN ('draft', 'active', 'expired', 'terminated', 'renewed')),
    compliance_status VARCHAR(30) DEFAULT 'compliant',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_irc_agreements_community ON eudr_indigenous_rights_checker.gl_eudr_irc_agreements(community_id);
CREATE INDEX idx_irc_agreements_status ON eudr_indigenous_rights_checker.gl_eudr_irc_agreements(status);

-- 8. FPIC Workflow Instances
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_fpic_workflows (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    territory_id UUID NOT NULL,
    community_id UUID NOT NULL,
    current_stage VARCHAR(30) NOT NULL,
    stage_entered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stage_deadline TIMESTAMPTZ,
    sla_status VARCHAR(20) DEFAULT 'on_track' CHECK (sla_status IN ('on_track', 'at_risk', 'overdue')),
    escalation_level INTEGER DEFAULT 0,
    consent_decision VARCHAR(30),
    agreement_id UUID,
    validity_end DATE,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_irc_workflows_stage ON eudr_indigenous_rights_checker.gl_eudr_irc_fpic_workflows(current_stage);
CREATE INDEX idx_irc_workflows_sla ON eudr_indigenous_rights_checker.gl_eudr_irc_fpic_workflows(sla_status);
CREATE INDEX idx_irc_workflows_community ON eudr_indigenous_rights_checker.gl_eudr_irc_fpic_workflows(community_id);

-- 9. FPIC Workflow State Transitions (hypertable)
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_workflow_transitions (
    transition_id UUID DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    from_stage VARCHAR(30) NOT NULL,
    to_stage VARCHAR(30) NOT NULL,
    actor VARCHAR(200) NOT NULL,
    reason TEXT,
    supporting_evidence JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (transition_id, transitioned_at)
);

SELECT create_hypertable('eudr_indigenous_rights_checker.gl_eudr_irc_workflow_transitions', 'transitioned_at');

-- 10. Violation Alerts (hypertable on detected_at)
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts (
    alert_id UUID DEFAULT gen_random_uuid(),
    source VARCHAR(200) NOT NULL,
    source_url VARCHAR(2000),
    source_document_hash VARCHAR(64),
    publication_date DATE NOT NULL,
    violation_type VARCHAR(50) NOT NULL,
    country_code CHAR(2) NOT NULL,
    region VARCHAR(200),
    location_lat DOUBLE PRECISION,
    location_lon DOUBLE PRECISION,
    affected_communities JSONB DEFAULT '[]',
    severity_score NUMERIC(5,2) NOT NULL CHECK (severity_score >= 0 AND severity_score <= 100),
    severity_level VARCHAR(20) NOT NULL CHECK (severity_level IN ('critical', 'high', 'medium', 'low')),
    supply_chain_correlation BOOLEAN DEFAULT FALSE,
    affected_plots JSONB DEFAULT '[]',
    affected_suppliers JSONB DEFAULT '[]',
    impact_assessment JSONB DEFAULT '{}',
    deduplication_group VARCHAR(100),
    status VARCHAR(30) DEFAULT 'active' CHECK (status IN ('active', 'investigating', 'resolved', 'false_positive', 'archived')),
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (alert_id, detected_at)
);

SELECT create_hypertable('eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts', 'detected_at');

CREATE INDEX idx_irc_violations_country ON eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts(country_code);
CREATE INDEX idx_irc_violations_type ON eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts(violation_type);
CREATE INDEX idx_irc_violations_severity ON eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts(severity_level);
CREATE INDEX idx_irc_violations_correlation ON eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts(supply_chain_correlation) WHERE supply_chain_correlation = TRUE;
CREATE INDEX idx_irc_violations_source ON eudr_indigenous_rights_checker.gl_eudr_irc_violation_alerts(source);

-- 11. Compliance Reports
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_compliance_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    format VARCHAR(10) NOT NULL CHECK (format IN ('pdf', 'json', 'html', 'csv', 'xlsx')),
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    scope_type VARCHAR(30) NOT NULL,
    scope_ids JSONB DEFAULT '[]',
    parameters JSONB DEFAULT '{}',
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    provenance_hash VARCHAR(64),
    generated_by VARCHAR(200),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_irc_reports_type ON eudr_indigenous_rights_checker.gl_eudr_irc_compliance_reports(report_type);

-- 12. Country Indigenous Rights Scores
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_country_scores (
    score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    ilo_169_ratified BOOLEAN DEFAULT FALSE,
    ilo_169_ratification_date DATE,
    fpic_legal_requirement BOOLEAN DEFAULT FALSE,
    land_tenure_security_score NUMERIC(5,2) NOT NULL CHECK (land_tenure_security_score >= 0 AND land_tenure_security_score <= 100),
    indigenous_rights_recognition_score NUMERIC(5,2) NOT NULL,
    judicial_protection_score NUMERIC(5,2) NOT NULL,
    territory_demarcation_pct NUMERIC(5,2),
    active_land_conflicts INTEGER DEFAULT 0,
    composite_indigenous_rights_score NUMERIC(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'standard', 'high')),
    data_sources JSONB NOT NULL DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code)
);

CREATE INDEX idx_irc_country_scores_level ON eudr_indigenous_rights_checker.gl_eudr_irc_country_scores(risk_level);

-- 13. Territory Version History
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_territory_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    territory_id UUID NOT NULL REFERENCES eudr_indigenous_rights_checker.gl_eudr_irc_territories(territory_id),
    version_number INTEGER NOT NULL,
    previous_boundary_geojson JSONB,
    new_boundary_geojson JSONB,
    change_description TEXT NOT NULL,
    change_source VARCHAR(200) NOT NULL,
    effective_date DATE NOT NULL,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_irc_territory_versions_territory ON eudr_indigenous_rights_checker.gl_eudr_irc_territory_versions(territory_id);

-- 14. Immutable Audit Log (hypertable)
CREATE TABLE eudr_indigenous_rights_checker.gl_eudr_irc_audit_log (
    log_id UUID DEFAULT gen_random_uuid(),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    actor VARCHAR(200) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    previous_state JSONB,
    new_state JSONB,
    ip_address VARCHAR(45),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (log_id, created_at)
);

SELECT create_hypertable('eudr_indigenous_rights_checker.gl_eudr_irc_audit_log', 'created_at');

CREATE INDEX idx_irc_audit_entity ON eudr_indigenous_rights_checker.gl_eudr_irc_audit_log(entity_type, entity_id);
CREATE INDEX idx_irc_audit_actor ON eudr_indigenous_rights_checker.gl_eudr_irc_audit_log(actor);
CREATE INDEX idx_irc_audit_action ON eudr_indigenous_rights_checker.gl_eudr_irc_audit_log(action);
```

### 7.5 API Endpoints (~35)

| Method | Path | Description |
|--------|------|-------------|
| **Territory Database** | | |
| GET | `/v1/eudr-irc/territories` | List indigenous territories (with filters: country, status, source) |
| GET | `/v1/eudr-irc/territories/{territory_id}` | Get territory details with boundary GeoJSON |
| POST | `/v1/eudr-irc/territories/search` | Search territories by name, people, or country |
| GET | `/v1/eudr-irc/territories/{territory_id}/history` | Get territory version history |
| GET | `/v1/eudr-irc/territories/coverage` | Get territory database coverage statistics |
| **FPIC Verification** | | |
| POST | `/v1/eudr-irc/fpic/verify` | Verify FPIC documentation for a plot-territory pair |
| GET | `/v1/eudr-irc/fpic/{assessment_id}` | Get FPIC assessment details |
| GET | `/v1/eudr-irc/fpic/plot/{plot_id}` | Get all FPIC assessments for a plot |
| GET | `/v1/eudr-irc/fpic/summary` | Get FPIC compliance summary statistics |
| **Overlap Detection** | | |
| POST | `/v1/eudr-irc/overlaps/detect` | Detect territory overlaps for a single plot |
| POST | `/v1/eudr-irc/overlaps/batch` | Batch overlap screening for multiple plots |
| GET | `/v1/eudr-irc/overlaps/{overlap_id}` | Get overlap details with map data |
| GET | `/v1/eudr-irc/overlaps` | List all detected overlaps (with filters) |
| GET | `/v1/eudr-irc/overlaps/statistics` | Get overlap statistics by country, commodity, severity |
| **Community Consultation** | | |
| POST | `/v1/eudr-irc/consultations` | Record a consultation activity |
| GET | `/v1/eudr-irc/consultations/{consultation_id}` | Get consultation record |
| GET | `/v1/eudr-irc/consultations` | List consultations (with filters: community, stage, date) |
| POST | `/v1/eudr-irc/grievances` | Submit a grievance |
| GET | `/v1/eudr-irc/grievances/{grievance_id}` | Get grievance details |
| PUT | `/v1/eudr-irc/grievances/{grievance_id}` | Update grievance status |
| **Violation Alerts** | | |
| GET | `/v1/eudr-irc/violations` | List violation alerts (with filters: country, type, severity) |
| GET | `/v1/eudr-irc/violations/{alert_id}` | Get violation alert details |
| POST | `/v1/eudr-irc/violations/correlate` | Correlate violations with supply chain |
| GET | `/v1/eudr-irc/violations/trends` | Get violation trend analysis |
| **Community Registry** | | |
| GET | `/v1/eudr-irc/communities` | List communities (with filters: country, people, commodity) |
| GET | `/v1/eudr-irc/communities/{community_id}` | Get community profile |
| GET | `/v1/eudr-irc/communities/{community_id}/engagements` | Get community engagement history |
| **FPIC Workflow** | | |
| POST | `/v1/eudr-irc/workflows` | Create new FPIC workflow |
| GET | `/v1/eudr-irc/workflows/{workflow_id}` | Get workflow status |
| POST | `/v1/eudr-irc/workflows/{workflow_id}/advance` | Advance workflow to next stage |
| POST | `/v1/eudr-irc/workflows/{workflow_id}/withdraw` | Record consent withdrawal |
| GET | `/v1/eudr-irc/workflows/dashboard` | Get FPIC workflow pipeline dashboard data |
| GET | `/v1/eudr-irc/workflows/sla` | Get SLA compliance statistics |
| **Compliance Reporting** | | |
| POST | `/v1/eudr-irc/reports/generate` | Generate indigenous rights compliance report |
| GET | `/v1/eudr-irc/reports/{report_id}` | Get report metadata |
| GET | `/v1/eudr-irc/reports/{report_id}/download` | Download report file |
| GET | `/v1/eudr-irc/reports` | List generated reports |
| **Country Scores** | | |
| GET | `/v1/eudr-irc/countries/{country_code}` | Get country indigenous rights score |
| GET | `/v1/eudr-irc/countries` | List all country indigenous rights scores |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (20)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_irc_territory_queries_total` | Counter | Territory database queries by type |
| 2 | `gl_eudr_irc_fpic_assessments_total` | Counter | FPIC assessments performed |
| 3 | `gl_eudr_irc_overlaps_detected_total` | Counter | Territory overlaps detected by type |
| 4 | `gl_eudr_irc_consultations_recorded_total` | Counter | Consultation records created |
| 5 | `gl_eudr_irc_violations_ingested_total` | Counter | Violation reports ingested by source |
| 6 | `gl_eudr_irc_violations_correlated_total` | Counter | Violations correlated with supply chain |
| 7 | `gl_eudr_irc_workflows_created_total` | Counter | FPIC workflows initiated |
| 8 | `gl_eudr_irc_workflow_transitions_total` | Counter | Workflow stage transitions |
| 9 | `gl_eudr_irc_reports_generated_total` | Counter | Compliance reports generated by type |
| 10 | `gl_eudr_irc_grievances_submitted_total` | Counter | Grievances submitted |
| 11 | `gl_eudr_irc_overlap_query_duration_seconds` | Histogram | Overlap detection query latency |
| 12 | `gl_eudr_irc_fpic_assessment_duration_seconds` | Histogram | FPIC assessment processing time |
| 13 | `gl_eudr_irc_batch_screening_duration_seconds` | Histogram | Batch overlap screening latency |
| 14 | `gl_eudr_irc_report_generation_duration_seconds` | Histogram | Report generation time |
| 15 | `gl_eudr_irc_api_errors_total` | Counter | API errors by endpoint and status code |
| 16 | `gl_eudr_irc_active_territories` | Gauge | Total territories in database |
| 17 | `gl_eudr_irc_active_overlaps` | Gauge | Currently active territory overlaps |
| 18 | `gl_eudr_irc_active_workflows` | Gauge | Currently active FPIC workflows |
| 19 | `gl_eudr_irc_workflow_sla_breaches_total` | Counter | FPIC workflow SLA breaches by stage |
| 20 | `gl_eudr_irc_grievance_sla_breaches_total` | Counter | Grievance SLA breaches by deadline type |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for assessment/violation history |
| Spatial | PostGIS + Shapely + GeoJSON | Territory boundary storage, spatial indexing, overlap calculations |
| Cache | Redis | Territory query caching, overlap result caching |
| Object Storage | S3 | Generated reports, FPIC document storage, territory boundary snapshots |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Arithmetic | Python Decimal | Precision for FPIC scores and overlap percentages |
| PDF Generation | WeasyPrint | HTML-to-PDF for compliance report generation |
| Templates | Jinja2 | Multi-format report templates (HTML/PDF) |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based indigenous rights data access control |
| Encryption | AES-256 via SEC-003 | Community PII encryption at rest |
| Monitoring | Prometheus + Grafana | 20 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-irc:territories:read` | View indigenous territory data | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:territories:write` | Update territory database records | Indigenous Rights Officer, Admin |
| `eudr-irc:fpic:read` | View FPIC assessments | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:fpic:verify` | Trigger FPIC verification assessments | Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:overlaps:read` | View territory overlap results | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:overlaps:detect` | Trigger overlap detection analysis | Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:consultations:read` | View consultation records | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:consultations:write` | Create and update consultation records | Indigenous Rights Officer, Compliance Officer, Admin |
| `eudr-irc:grievances:read` | View grievance records | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:grievances:write` | Manage grievance lifecycle | Indigenous Rights Officer, Admin |
| `eudr-irc:violations:read` | View violation alerts | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:violations:correlate` | Correlate violations with supply chain | Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:communities:read` | View community registry (non-PII) | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:communities:contacts` | View community contact PII | Indigenous Rights Officer, Admin |
| `eudr-irc:communities:write` | Update community registry | Indigenous Rights Officer, Admin |
| `eudr-irc:workflows:read` | View FPIC workflow status | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:workflows:manage` | Manage FPIC workflows (create, advance, withdraw) | Indigenous Rights Officer, Compliance Officer, Admin |
| `eudr-irc:reports:read` | View generated reports | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:reports:generate` | Generate compliance reports | Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:reports:download` | Download report files | Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:countries:read` | View country indigenous rights scores | Viewer, Analyst, Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:audit:read` | View audit trail | Auditor (read-only), Compliance Officer, Indigenous Rights Officer, Admin |
| `eudr-irc:config:manage` | Manage configuration (buffer zones, SLA timelines, scoring weights) | Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent/Source | Integration | Data Flow |
|-------------|-------------|-----------|
| LandMark Global Platform | Data download/API | Indigenous territory polygons for 100+ countries -> territory database (F1) |
| RAISG Data Portal | Data download | Amazon Basin indigenous territory polygons -> territory database (F1) |
| FUNAI Geoserver | GIS data feed | Brazil Terras Indigenas boundaries -> territory database (F1) |
| BPN/AMAN | Data sharing agreement | Indonesia Masyarakat Adat territories -> territory database (F1) |
| AGENT-EUDR-002 Geolocation Verification | Plot coordinates | Validated plot coordinates -> overlap detection (F3) |
| AGENT-EUDR-006 Plot Boundary Manager | Plot polygons | Plot boundary polygons -> overlap detection (F3) |
| AGENT-EUDR-016 Country Risk Evaluator | Country governance data | Indigenous rights governance scores -> country scoring (F6) |
| AGENT-EUDR-020 Deforestation Alert System | Deforestation alerts | Deforestation events near territories -> cross-reference (F5, F9) |
| AGENT-DATA-006 GIS/Mapping Connector | Spatial operations | PostGIS utilities, protected area data -> overlap analysis (F3) |
| AGENT-FOUND-005 Citations & Evidence | Source attribution | Citation generation for all data sources -> provenance tracking |
| AGENT-FOUND-008 Reproducibility | Determinism verification | Bit-perfect verification of FPIC scoring and overlap calculations |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping Master | Graph enrichment | Indigenous rights risk attributes -> supply chain graph node properties |
| AGENT-EUDR-016 Country Risk Evaluator | Governance input | Country indigenous rights scores -> governance index engine F4.7 |
| AGENT-EUDR-017 Supplier Risk Scorer | Supplier risk input | Per-supplier indigenous rights risk -> supplier composite risk score |
| GL-EUDR-APP v1.0 Platform | API integration | Territory map overlay, FPIC dashboard, violation feed, report downloads |
| GL-EUDR-APP DDS Reporting Engine | DDS section | Indigenous rights risk assessment for Article 4(2) DDS submission |
| External Auditors | Read-only API + reports | Indigenous rights compliance exports for third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Territory Overlap Screening (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Indigenous Rights" module -> "Territory Screening" tab
3. Selects commodity (e.g., Palm Oil) and clicks "Screen All Plots"
4. System performs batch overlap detection for all palm oil supply chain plots
   -> Progress bar: "Screening 3,200 plots against 50,000+ territories..."
5. Results dashboard displays:
   - Total plots screened: 3,200
   - Direct overlaps: 12 (CRITICAL)
   - Partial overlaps: 28 (HIGH)
   - Adjacent (< 5km): 45 (MEDIUM)
   - Proximate (< 25km): 89 (LOW)
   - No overlap: 3,026 (CLEAR)
6. Officer clicks on a CRITICAL overlap -> sees plot on map with territory overlay
   - Plot: Plantation PLT-ID-0342 (West Kalimantan, Indonesia)
   - Territory: Dayak Iban community territory (titled, 12,500 ha)
   - Overlap: 340 ha (27% of plot area)
   - FPIC status: CONSENT_MISSING
7. Officer clicks "Initiate FPIC Process" -> FPIC workflow created
8. System generates indigenous rights screening report for DDS
```

#### Flow 2: FPIC Workflow Management (Indigenous Rights Officer)

```
1. Indigenous rights officer opens "FPIC Workflows" dashboard
2. Pipeline view shows:
   - IDENTIFICATION: 5 workflows
   - INFORMATION_DISCLOSURE: 8 workflows (2 at-risk SLA)
   - CONSULTATION: 3 workflows
   - CONSENT_DECISION: 1 workflow
   - AGREEMENT: 2 workflows
   - MONITORING: 15 workflows (1 renewal due in 60 days)
3. Officer clicks on at-risk workflow WF-2026-0087
   - Community: Dayak Iban (West Kalimantan)
   - Current stage: INFORMATION_DISCLOSURE
   - SLA: 5 days overdue (30-day deadline exceeded)
   - Action required: Share environmental impact assessment in Bahasa Indonesia
4. Officer uploads translated EIA document and marks information shared
5. System advances workflow to CONSULTATION stage
6. System sends notification to community representative contact
7. System sets 60-day consultation SLA deadline
```

#### Flow 3: Violation Alert Investigation (Compliance Officer)

```
1. System receives violation report from Forest Peoples Programme:
   "Forced displacement of Baka communities in eastern Cameroon for palm oil plantation expansion"
2. System correlates with operator supply chain:
   - Operator sources palm oil from Cameroon (2 supplier mills)
   - Mill CM-MILL-014 is 18km from reported violation location
   - Severity: HIGH (forced_displacement=100 * 0.30 + proximate_25km=50 * 0.25 + ...)
3. Compliance officer receives alert notification
4. Opens violation alert detail view:
   - Source: Forest Peoples Programme (2026-03-05)
   - Type: Forced displacement
   - Location: Eastern Province, Cameroon
   - Affected community: Baka people
   - Supply chain correlation: CONFIRMED (Mill CM-MILL-014, 18km distance)
5. Officer triggers enhanced due diligence for Mill CM-MILL-014
6. System generates violation impact assessment report
7. Officer records remediation actions in consultation tracker
```

### 8.2 Key Screen Descriptions

**Territory Overlap Dashboard:**
- Interactive map: production plots (circles) overlaid with indigenous territories (shaded polygons)
- Color-coded by overlap severity: red (CRITICAL/DIRECT), orange (HIGH/PARTIAL), yellow (MEDIUM/ADJACENT), green (CLEAR)
- Left sidebar: filter panel (commodity, country, overlap type, FPIC status)
- Right sidebar: selected plot/territory detail panel
- Top bar: summary statistics (plots screened, overlap counts by severity)
- Bottom panel: territory database source attribution

**FPIC Workflow Pipeline:**
- Kanban-style board with 7 stage columns
- Each workflow card shows: community name, plot reference, SLA status indicator (green/yellow/red)
- SLA countdown timer on each card
- Drag-and-drop disabled (stage-gate enforcement; must use "Advance" action)
- Filter bar: by country, commodity, SLA status, community

**Violation Alert Feed:**
- Timeline view: violation alerts sorted by date with severity badges
- Map view: violation locations with operator supply chain plot overlay
- Correlation indicator: supply chain match (checkmark) or no match (dash)
- Detail panel: source citation, affected communities, impact assessment, recommended actions

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Indigenous Territory Database -- 50,000+ territories, 6 sources, PostGIS spatial indexing
  - [ ] Feature 2: FPIC Verification Engine -- 10-element scoring, country-specific rules, temporal validation
  - [ ] Feature 3: Land Rights Overlap Detector -- 4-type classification, batch screening, risk scoring
  - [ ] Feature 4: Community Consultation Tracker -- 7-stage lifecycle, grievance management, audit trail
  - [ ] Feature 5: Rights Violation Alert System -- 10+ sources, severity classification, supply chain correlation
  - [ ] Feature 6: Indigenous Community Registry -- community profiles, legal protections, commodity relevance
  - [ ] Feature 7: FPIC Workflow Management -- 7-stage state machine, SLA enforcement, template library
  - [ ] Feature 8: Compliance Reporting -- PDF/JSON/HTML, multi-language, DDS integration, certification alignment
  - [ ] Feature 9: Agent Integration -- EUDR-001/002/006/016/017/020 bidirectional integration verified
- [ ] >= 85% test coverage achieved (650+ tests)
- [ ] Security audit passed (JWT + RBAC integrated, 23 permissions)
- [ ] Performance targets met (< 500ms overlap query p99, < 10s report generation)
- [ ] Territory database validated against manual GIS review (>= 99% spatial precision)
- [ ] FPIC scoring verified deterministic (bit-perfect reproducibility)
- [ ] Overlap detection validated against 100+ known plot-territory pairs
- [ ] API documentation complete (OpenAPI spec, ~35 endpoints)
- [ ] Database migration V109 tested and validated (14 tables, 4 hypertables)
- [ ] Integration with EUDR-001, EUDR-002, EUDR-006, EUDR-016, EUDR-017, EUDR-020 verified
- [ ] 5 beta customers successfully screening their supply chains
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ supply chains screened for indigenous territory overlaps
- 1,000+ plots overlap-screened
- 10+ FPIC workflows initiated
- Average overlap query latency < 500ms (p99)
- Territory database freshness within 30 days
- < 5 support tickets per customer

**60 Days:**
- 200+ supply chains actively monitored
- 10,000+ plots screened
- 50+ FPIC workflows active across all stages
- 20+ violation alerts correlated with supply chains
- Compliance reports generated for 3+ certification schemes
- NPS > 45 from indigenous rights officer persona

**90 Days:**
- 500+ supply chains actively monitored
- 50,000+ plots screened (full customer portfolio)
- 100+ FPIC workflows in production
- Zero EUDR penalties attributable to indigenous rights non-compliance for active customers
- Full integration with GL-EUDR-APP DDS workflow operational
- NPS > 55

---

## 10. Timeline and Milestones

### Phase 1: Core Verification Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Indigenous Territory Database (Feature 1): 6-source integration, PostGIS spatial indexing, version control | Senior Backend Engineer + GIS Specialist |
| 2-3 | Land Rights Overlap Detector (Feature 3): spatial intersection, 4-type classification, batch screening | Backend Engineer + GIS Specialist |
| 3-4 | FPIC Verification Engine (Feature 2): 10-element scoring, country-specific rules, temporal validation | Senior Backend Engineer |
| 4-5 | Rights Violation Alert System (Feature 5): 10+ source integration, severity scoring, supply chain correlation | Backend Engineer + Data Engineer |
| 5-6 | Indigenous Community Registry (Feature 6): community profiles, legal protections, privacy controls | Backend Engineer |

**Milestone: Core verification engine operational with 5 core features (Week 6)**

### Phase 2: Workflow, API, and Reporting (Weeks 7-11)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Community Consultation Tracker (Feature 4): 7-stage lifecycle, grievance management, SLA tracking | Backend Engineer |
| 8-9 | FPIC Workflow Management (Feature 7): state machine, SLA enforcement, template library | Senior Backend Engineer |
| 9-10 | REST API Layer: ~35 endpoints, authentication, rate limiting | Backend Engineer |
| 10-11 | Compliance Reporting (Feature 8): PDF/JSON/HTML, multi-language, certification alignment | Backend + Template Engineer |

**Milestone: Full API operational with workflow management and compliance reporting (Week 11)**

### Phase 3: Integration, RBAC, and Observability (Weeks 12-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 12 | Agent Integration (Feature 9): EUDR-001/002/006/016/017/020 bidirectional integration | Senior Backend Engineer |
| 12-13 | RBAC integration (23 permissions), PII encryption, Prometheus metrics (20) | Backend + DevOps |
| 13-14 | Grafana dashboard, OpenTelemetry tracing, end-to-end integration testing | DevOps + Backend |

**Milestone: All 9 P0 features implemented with full integration and observability (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 650+ tests, golden tests for all 7 commodities, spatial validation | Test Engineer |
| 16-17 | Performance testing, security audit (PII encryption), load testing | DevOps + Security |
| 17 | Database migration V109 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers) | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Interactive indigenous rights impact dashboard (Feature 10)
- Community voice integration (Feature 11)
- Predictive encroachment analysis (Feature 12)
- Additional territory data sources (national registries for 20+ countries)
- CSDDD Article 7 alignment and reporting

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable; graph enrichment API defined |
| AGENT-EUDR-002 Geolocation Verification Agent | BUILT (100%) | Low | Plot coordinates validated and available |
| AGENT-EUDR-006 Plot Boundary Manager Agent | BUILT (100%) | Low | Plot polygons available for overlap analysis |
| AGENT-EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Governance F4.7 integration point defined |
| AGENT-EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Supplier risk enrichment API defined |
| AGENT-EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Deforestation-territory correlation API defined |
| AGENT-DATA-006 GIS/Mapping Connector | BUILT (100%) | Low | PostGIS utilities available |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration target available |
| PostgreSQL + TimescaleDB + PostGIS | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | AES-256 for community PII |
| SEC-011 PII Detection/Redaction | BUILT (100%) | Low | Privacy compliance for community data |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| LandMark Global Platform | Available (open data) | Medium | Data download with offline cache; periodic refresh |
| RAISG Data Portal | Available (open data) | Low | Data download with offline cache; annual refresh |
| FUNAI Geoserver | Available (government data) | Medium | GIS data download; government API may change format |
| BPN/AMAN Territory Data | Requires data sharing agreement | High | Establish partnership; use LandMark Indonesia data as fallback |
| ACHPR Indigenous Peoples Data | Limited availability | Medium | Supplement with national sources; use LandMark Africa data |
| Violation monitoring sources (IWGIA, Cultural Survival, etc.) | Available (public) | Medium | Multi-source redundancy; RSS/API/scraping with adapter pattern |
| ILO Convention 169 ratification database | Available (ILO) | Low | Static data; updated on ratification events (rare) |
| EC EUDR benchmarking updates | Published (evolving) | Medium | Hot-reloadable country scores; adapter pattern |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Indigenous territory boundary data is incomplete or inaccurate for some regions | High | High | Multi-source data fusion (6 sources); confidence scoring per territory; conservative (largest) boundary when disputed; continuous data improvement program |
| R2 | BPN/AMAN data sharing agreement for Indonesia not established in time | Medium | High | Use LandMark Indonesia data as primary source; engage AMAN through GreenLang partnership program; supplement with RSPO concession maps |
| R3 | FPIC legal requirements vary significantly across jurisdictions, creating complexity | High | Medium | Country-specific rule engine with configurable validation rules per jurisdiction; start with 8 major producing countries; extend coverage iteratively |
| R4 | Community PII data creates privacy liability under GDPR and national data protection laws | Medium | High | AES-256 encryption at rest; RBAC-controlled access; PII minimization; community data sovereignty provisions; privacy impact assessment |
| R5 | Indigenous communities lack digital access for engagement tracking | High | Medium | System tracks operator-side records; does not require community digital access; paper-based engagement records can be digitized by operator |
| R6 | Violation monitoring sources provide inconsistent, incomplete, or delayed reports | Medium | Medium | Multi-source redundancy (10+ sources); deduplication; confidence scoring; supplement with operator self-reporting |
| R7 | Overlap detection produces false positives due to territory boundary inaccuracies | Medium | Medium | Confidence scoring on boundary data; operator review workflow for flagged overlaps; manual verification option for CRITICAL results |
| R8 | Regulatory interpretation of indigenous rights requirements evolves | Medium | Medium | Configuration-driven compliance rules; version-controlled legal framework database; modular architecture for quick adaptation |
| R9 | Integration complexity with 6 upstream EUDR agents | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern; integration health monitoring |
| R10 | Customer resistance to indigenous rights compliance (seen as cost burden) | Medium | Medium | Demonstrate regulatory penalty risk (4% turnover); show CSDDD dual-use value; provide ROI analysis for compliance vs. penalty |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Territory Database Tests | 80+ | Data ingestion, spatial indexing, version control, multi-source merge, search, coverage |
| FPIC Verification Tests | 70+ | 10-element scoring, country-specific rules, temporal validation, coercion detection, edge cases |
| Overlap Detection Tests | 80+ | Spatial intersection, 4-type classification, batch screening, buffer zones, risk scoring |
| Community Consultation Tests | 50+ | 7-stage lifecycle, grievance management, SLA tracking, multi-community engagement |
| Violation Alert Tests | 60+ | Source ingestion, severity scoring, supply chain correlation, deduplication, trend analysis |
| Community Registry Tests | 40+ | CRUD operations, privacy controls, search, filtering, data sovereignty |
| FPIC Workflow Tests | 60+ | State machine transitions, SLA enforcement, consent withdrawal, parallel workflows |
| Compliance Reporting Tests | 40+ | All formats (PDF/JSON/HTML), multi-language, template rendering, provenance |
| Agent Integration Tests | 40+ | Cross-agent data flow with EUDR-001/002/006/016/017/020 |
| API Tests | 50+ | All ~35 endpoints, auth, error handling, pagination, rate limiting |
| Golden Tests | 50+ | 7 commodity supply chains x 7 scenarios (overlap, no overlap, FPIC complete, FPIC missing, violation, multi-community, cross-border) |
| Performance Tests | 20+ | Single-plot overlap, batch 10K plots, concurrent queries, report generation |
| Determinism Tests | 10+ | Bit-perfect reproducibility for FPIC scoring and overlap detection |
| **Total** | **650+** | |

### 13.2 Golden Test Scenarios

Each of the 7 EUDR commodities will have 7 golden test scenarios:

1. **Clear plot** -- Plot with no indigenous territory overlap -> expect NONE classification, NOT_APPLICABLE FPIC status
2. **Direct overlap** -- Plot centroid inside titled territory -> expect DIRECT/CRITICAL, CONSENT_MISSING until FPIC workflow completes
3. **Partial overlap** -- Plot polygon intersects territory boundary -> expect PARTIAL/HIGH, overlap metrics calculated correctly
4. **Adjacent plot** -- Plot within 5km of territory -> expect ADJACENT/MEDIUM, community identification completed
5. **FPIC complete** -- Plot with full FPIC documentation -> expect CONSENT_OBTAINED (score >= 80), valid certificate
6. **Violation correlation** -- Plot in region with reported violation -> expect supply chain correlation, enhanced DD triggered
7. **Multi-community** -- Plot overlapping multiple territories -> expect independent FPIC workflows per community

Total: 7 commodities x 7 scenarios = 49 golden test scenarios (+ 1 cross-border scenario = 50)

### 13.3 Determinism Tests

Every scoring and calculation engine will include determinism tests that:
1. Run the same calculation 100 times with identical inputs
2. Verify bit-perfect identical outputs (SHA-256 hash match)
3. Test across Python versions (3.11, 3.12) to ensure no platform-dependent behavior
4. Verify Decimal arithmetic produces identical results to reference calculations
5. Verify spatial calculations produce identical overlap metrics across PostGIS versions

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **FPIC** | Free, Prior and Informed Consent -- principle requiring that indigenous peoples give consent before projects affecting their lands, freely, before the project starts, based on adequate information |
| **ILO Convention 169** | International Labour Organization Convention on Indigenous and Tribal Peoples (1989) -- binding treaty ratified by 24 countries requiring FPIC and land rights protection |
| **UNDRIP** | United Nations Declaration on the Rights of Indigenous Peoples (2007) -- non-binding declaration endorsed by 148 states establishing indigenous rights standards |
| **VGGT** | Voluntary Guidelines on the Governance of Tenure (2012) -- FAO guidelines on legitimate tenure rights |
| **CSDDD** | Corporate Sustainability Due Diligence Directive (2024) -- EU directive requiring human rights and environmental due diligence |
| **LandMark** | Global Platform of Indigenous and Community Lands -- georeferenced database of indigenous territories |
| **RAISG** | Red Amazonica de Informacion Socioambiental Georeferenciada -- Amazon Basin indigenous territory network |
| **FUNAI** | Fundacao Nacional dos Povos Indigenas -- Brazilian federal agency for indigenous peoples |
| **BPN** | Badan Pertanahan Nasional -- Indonesian National Land Agency |
| **AMAN** | Aliansi Masyarakat Adat Nusantara -- Alliance of Indigenous Peoples of the Archipelago (Indonesia) |
| **ACHPR** | African Commission on Human and Peoples' Rights |
| **Terras Indigenas** | Indigenous Lands (Brazil) -- constitutionally protected indigenous territories demarcated by FUNAI |
| **Masyarakat Adat** | Indigenous/customary communities (Indonesia) recognized under Constitutional Court Decision 35/2012 |
| **Resguardo** | Indigenous reserve (Colombia) -- collectively titled indigenous territory |
| **NCR** | Native Customary Rights (Malaysia) -- customary land rights under Sarawak Land Code |
| **PADIATAPA** | Indonesian FPIC framework for indigenous communities |
| **PostGIS** | Spatial database extension for PostgreSQL enabling geographic queries |
| **GIST Index** | Generalized Search Tree -- PostGIS index type for spatial queries |

### Appendix B: Indigenous Territory Data Sources

| Source | Coverage | Records | Format | Access | Update Frequency |
|--------|----------|---------|--------|--------|-----------------|
| LandMark | Global (100+ countries) | 30,000+ territories | GeoJSON/Shapefile | Open data download | Annual |
| RAISG | Amazon Basin (9 countries) | 5,000+ territories | GeoJSON/Shapefile | Open data portal | Biennial |
| FUNAI | Brazil | 700+ Terras Indigenas | WFS/Shapefile | Government GIS portal | Monthly |
| BPN/AMAN | Indonesia | 17,000+ territories | GeoJSON/KML | Data sharing agreement | Annual |
| ACHPR | Sub-Saharan Africa | 1,000+ territories | Various | Research partnership | Biennial |
| National Registries | Latin America (6 countries) | 3,000+ territories | Various | Government portals | Annual |

### Appendix C: ILO Convention 169 Ratification Status (Major Producing Countries)

| Country | EUDR Commodities | ILO 169 Ratified | Ratification Year | FPIC Legal Requirement |
|---------|-----------------|------------------|-------------------|----------------------|
| Brazil | Cattle, soya, cocoa, coffee, wood | Yes | 2002 | Yes (Constitution Art. 231) |
| Indonesia | Palm oil, rubber, wood, cocoa, coffee | No | -- | Partial (Constitutional Court 35/2012) |
| Colombia | Coffee, palm oil, cocoa, wood | Yes | 1991 | Yes (Constitution Art. 330) |
| Peru | Coffee, cocoa, wood, palm oil | Yes | 1994 | Yes (Ley 29785/2011) |
| Bolivia | Soya, wood, cattle | Yes | 1991 | Yes (Constitution Art. 30) |
| Paraguay | Soya, cattle, wood | Yes | 1993 | Yes (Ley 904/1981) |
| Guatemala | Coffee, palm oil, rubber | Yes | 1996 | Yes (ILO 169 direct application) |
| Honduras | Coffee, palm oil, wood | Yes | 1995 | Yes (ILO 169 direct application) |
| Cote d'Ivoire | Cocoa, coffee, rubber, palm oil | No | -- | Partial (Land Law 98-750) |
| Ghana | Cocoa, wood | No | -- | No (customary consultation only) |
| DRC | Wood, cocoa, coffee | No | -- | Partial (Forest Code Art. 7) |
| Malaysia | Palm oil, rubber, wood | No | -- | No (NCR under state law) |
| Cameroon | Cocoa, wood, palm oil | No | -- | No (limited legal framework) |

### Appendix D: EUDR Recital 32 Reference

> "The Union is committed to respecting, protecting and promoting human rights, including the rights of indigenous peoples, as enshrined in the Charter of Fundamental Rights of the European Union and in international human rights instruments, including the United Nations Declaration on the Rights of Indigenous Peoples."

This recital establishes the regulatory basis for indigenous rights compliance within EUDR due diligence, referenced in the agent's FPIC verification engine and community consultation tracker.

### Appendix E: Integration API Contracts

**Provided to EUDR-001 (Supply Chain Mapping Master):**
```python
# Indigenous rights risk data for graph node enrichment
def get_indigenous_rights_risk(plot_id: str) -> Dict:
    """Returns: {
        territory_overlap: bool,
        overlap_type: str,  # direct/partial/adjacent/proximate/none
        overlap_risk_score: Decimal,
        fpic_status: str,  # consent_obtained/partial/missing/withdrawn/not_applicable
        fpic_score: Decimal,
        violation_exposure: bool,
        indigenous_rights_risk_level: str,  # critical/high/medium/low/none
        provenance_hash: str
    }"""
```

**Provided to EUDR-016 (Country Risk Evaluator):**
```python
# Country indigenous rights score for governance index F4.7
def get_country_indigenous_rights_score(country_code: str) -> Dict:
    """Returns: {
        country_code: str,
        ilo_169_ratified: bool,
        fpic_legal_requirement: bool,
        land_tenure_security_score: Decimal,  # 0-100
        indigenous_rights_recognition_score: Decimal,  # 0-100
        composite_indigenous_rights_score: Decimal,  # 0-100
        risk_level: str,  # low/standard/high
        provenance_hash: str
    }"""
```

**Provided to EUDR-017 (Supplier Risk Scorer):**
```python
# Supplier indigenous rights risk for composite supplier scoring
def get_supplier_indigenous_rights_risk(supplier_id: str) -> Dict:
    """Returns: {
        supplier_id: str,
        total_plots: int,
        plots_with_overlap: int,
        plots_with_fpic_obtained: int,
        plots_with_fpic_missing: int,
        active_violations: int,
        indigenous_rights_risk_score: Decimal,  # 0-100
        provenance_hash: str
    }"""
```

### Appendix F: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. European Commission EUDR Guidance Document -- Risk Assessment and Due Diligence
3. ILO Convention 169 -- Indigenous and Tribal Peoples Convention (1989)
4. United Nations Declaration on the Rights of Indigenous Peoples (UNDRIP, 2007)
5. FAO Voluntary Guidelines on the Governance of Tenure (VGGT, 2012)
6. Directive (EU) 2024/1760 -- Corporate Sustainability Due Diligence Directive (CSDDD)
7. LandMark: Global Platform of Indigenous and Community Lands -- Technical Documentation
8. RAISG -- Amazon Georeferenced Socio-Environmental Information Network Data Methodology
9. FUNAI -- Sistema de Informacoes Fundiarias (SIFUND) Technical Documentation
10. AMAN -- Peta Wilayah Adat (Ancestral Domain Maps) Methodology
11. FSC Principle 3: Indigenous Peoples' Rights (FSC-STD-01-001)
12. RSPO Principles and Criteria 7.5/7.6: Free, Prior and Informed Consent
13. Rainforest Alliance SAN Standard -- Indigenous Rights Requirements
14. ACHPR Resolution on the Rights of Indigenous Populations/Communities in Africa
15. Brazilian Federal Constitution, Article 231 -- Indigenous Lands
16. Colombian Constitution, Article 330 -- Prior Consultation
17. Peru Prior Consultation Law 29785 (2011)
18. Indonesian Constitutional Court Decision 35/PUU-X/2012 -- Customary Forest Rights
19. ISO 3166-1 -- Country Codes Standard

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-10 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Indigenous Rights Specialist | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-10 | GL-ProductManager | Initial draft created: all 9 P0 features specified (90 sub-requirements), regulatory coverage verified (EUDR Articles 2/3/8/10/11/29/31, ILO 169, UNDRIP, VGGT, CSDDD), 14-table DB schema V109 designed, ~35 API endpoints specified, 20 Prometheus metrics defined, 23 RBAC permissions registered, 650+ test target set, 18-week timeline established |
