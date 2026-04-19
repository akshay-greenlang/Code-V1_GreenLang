# PRD: AGENT-EUDR-022 -- Protected Area Validator Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-022 |
| **Agent ID** | GL-EUDR-PAV-022 |
| **Component** | Protected Area Validator Agent |
| **Category** | EUDR Regulatory Agent -- Protected Area & Biodiversity Compliance |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 2, 9, 10, 11, 29, 31; Convention on Biological Diversity (CBD); CITES; World Heritage Convention; Ramsar Convention |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) requires that commodities placed on the EU market are not only deforestation-free but also "legally produced" in accordance with all applicable laws of the country of production (Article 2(28), Article 3). This legality requirement encompasses compliance with national laws governing protected areas, nature reserves, national parks, and biodiversity conservation zones. Article 10(2)(d) specifically requires operators to assess the risk of non-compliance by considering "the prevalence of deforestation or forest degradation" in production regions -- a metric inextricably linked to the proximity of commodity production to legally protected ecosystems. Article 29(3) mandates the European Commission to consider protected area governance and enforcement in its country benchmarking, directly tying protected area management effectiveness to EUDR risk classification.

Globally, there are over 270,000 designated protected areas covering approximately 16.6% of terrestrial land area and 8.2% of marine and coastal areas, as recorded in the World Database on Protected Areas (WDPA) maintained by UN Environment Programme's World Conservation Monitoring Centre (UNEP-WCMC). These protected areas span the full spectrum of IUCN management categories, from Strict Nature Reserves (Category Ia) where no commodity production is permitted, to Protected Landscapes (Category V) where sustainable resource extraction may be conditionally allowed. Many of the world's most critical protected areas -- including UNESCO World Heritage Sites, Ramsar Wetlands of International Importance, and Key Biodiversity Areas (KBAs) -- are located in the tropical regions from which EUDR-regulated commodities are predominantly sourced. The Amazon, Congo Basin, and Southeast Asian tropical forests contain thousands of protected areas that directly overlap with or are adjacent to agricultural frontier zones for cattle, cocoa, coffee, palm oil, rubber, soya, and wood production.

Today, EU operators and compliance teams face the following critical gaps when verifying protected area compliance for their EUDR-regulated supply chains:

- **No protected area database integration**: Operators have no systematic access to georeferenced protected area boundaries from the WDPA, Protected Planet, or national protected area networks. Data is scattered across international databases (WDPA/Protected Planet), regional networks (BIOPAMA, RAPAC), and national registries (ICMBio in Brazil, KLHK in Indonesia, ICCN in DRC). No existing EUDR compliance tool consolidates these sources into a queryable, verified spatial database with IUCN category classification.
- **No spatial overlap detection with IUCN category awareness**: When production plots are located within, adjacent to, or overlapping with protected areas, the compliance risk varies dramatically based on the IUCN management category. A plot inside a Category Ia Strict Nature Reserve presents an absolute compliance failure, while a plot adjacent to a Category VI Sustainable Use area may require only enhanced monitoring. There is no automated system that performs spatial intersection analysis with IUCN-category-weighted risk scoring.
- **No buffer zone monitoring**: Protected area buffer zones -- the transitional areas surrounding formal protected area boundaries -- are critical ecological corridors subject to national legislation in many jurisdictions. There is no configurable buffer zone monitoring system that detects commodity production activities within these zones, tracks encroachment trends, and generates proximity alerts at customizable radii (1km, 5km, 10km, 25km, 50km).
- **No protected area designation validator**: Protected area legal status varies from internationally designated (UNESCO World Heritage, Ramsar) through nationally gazetted to locally proposed. Operators cannot verify the legal designation status, management authority, governance effectiveness, or enforcement capacity of protected areas near their supply chains.
- **No high-risk proximity alert system**: When supply chain plots are located near protected areas with high biodiversity value (Key Biodiversity Areas, Alliance for Zero Extinction sites, IUCN Red List critical habitats), there is no automated risk scoring and alert mechanism to flag these as requiring enhanced due diligence.
- **No protected area compliance tracking**: Operators cannot systematically track violations of protected area boundaries, monitor remediation of encroachment incidents, or maintain compliance records for audit purposes.
- **No conservation status assessment**: There is no integration with IUCN Red List data to assess the conservation significance of biodiversity present in protected areas near supply chain operations, which is essential for risk assessment under Article 10.
- **No audit-ready compliance reporting**: Auditors reviewing EUDR due diligence statements need structured evidence that protected area compliance has been assessed. There is no standardized reporting framework that generates audit-ready protected area compliance documentation with spatial analysis, IUCN category assessment, and buffer zone monitoring results.

Without solving these problems, EU operators face EUDR penalties of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, reputational damage from association with protected area violations, and exclusion from responsible sourcing certification schemes (FSC, RSPO, Rainforest Alliance) that require protected area compliance.

### 1.2 Solution Overview

Agent-EUDR-022: Protected Area Validator is a specialized compliance agent that provides comprehensive protected area compliance verification for EUDR-regulated commodity supply chains. It is the 22nd agent in the EUDR agent family and extends the Risk Assessment sub-category (EUDR-016 through EUDR-021) with dedicated protected area and biodiversity compliance capabilities. The agent integrates the authoritative World Database on Protected Areas (WDPA) containing 270,000+ protected areas, automates IUCN-category-aware spatial overlap detection, provides configurable buffer zone monitoring, validates protected area legal designations, and generates audit-ready compliance documentation.

The agent builds on and integrates with the existing EUDR agent ecosystem: EUDR-001 (Supply Chain Mapping Master) for supply chain graph data, EUDR-002 (Geolocation Verification) for plot coordinate validation, EUDR-006 (Plot Boundary Manager) for spatial plot data, EUDR-016 (Country Risk Evaluator) for governance and environmental enforcement scoring, EUDR-020 (Deforestation Alert System) for spatial monitoring alerts, and EUDR-021 (Indigenous Rights Checker) for complementary rights-based risk assessment.

Core capabilities:

1. **Protected area database integration** -- Consolidated, georeferenced database of protected areas from 4 authoritative sources (WDPA/Protected Planet, national registries, regional conservation networks, Key Biodiversity Areas database) covering 270,000+ protected areas across 245 countries and territories. IUCN management category classification (Ia, Ib, II, III, IV, V, VI, Not Reported). Spatial indexing for sub-second overlap queries. Legal designation status tracking (international, national, local, proposed).
2. **Spatial overlap detection engine** -- PostGIS-powered spatial analysis engine that detects overlaps between supply chain production plots and protected area boundaries using polygon intersection, point-in-polygon, and distance calculations. IUCN-category-aware risk scoring where Category Ia/Ib violations are scored as CRITICAL (100) while Category VI proximity is scored as LOW (25). Supports batch screening of 10,000+ plots against the full WDPA database.
3. **Buffer zone monitoring** -- Configurable buffer zone analysis around protected areas with operator-defined radii (1km, 5km, 10km, 25km, 50km). Detects production activities within buffer zones, tracks encroachment trends over time, generates proximity alerts with severity classification, and monitors buffer zone expansion/contraction events.
4. **Protected area designation validator** -- Validates the legal designation status of protected areas including international designations (UNESCO World Heritage, Ramsar Wetlands, Biosphere Reserves), national designations (national parks, wildlife reserves, forest reserves), management effectiveness (METT/PAME scores), governance type (government, shared, private, indigenous), and enforcement level assessment.
5. **High-risk proximity alert system** -- Automated risk scoring and alert generation when supply chain plots are near protected areas of high biodiversity significance. Integrates Key Biodiversity Areas (KBAs), Alliance for Zero Extinction (AZE) sites, IUCN Red List critical habitat data, and UNESCO World Heritage Sites for enhanced risk classification.
6. **Protected area compliance tracker** -- Systematic tracking of protected area boundary violations, encroachment incidents, remediation workflows, and compliance status per supply chain plot. Historical compliance records with trend analysis for audit purposes.
7. **Conservation status assessment** -- Integration with IUCN Red List data to assess biodiversity significance of protected areas near supply chain operations. Habitat fragmentation analysis for protected areas experiencing agricultural pressure. Species presence assessment for protected areas containing EUDR-relevant habitats.
8. **Compliance reporting engine** -- Automated generation of 8 report types (full compliance, executive summary, per-supplier scorecard, per-commodity analysis, DDS section, certification audit, trend analysis, BI export) in 5 formats (PDF, JSON, HTML, CSV, XLSX) with 5 language support (EN, FR, DE, ES, PT). SHA-256 provenance hashes on all reports.
9. **Integration with existing EUDR agents** -- Bidirectional integration with EUDR-001 (supply chain graph nodes enriched with protected area risk), EUDR-002 (plot coordinates validated against protected area boundaries), EUDR-006 (plot boundaries for overlap analysis), EUDR-016 (protected area governance scores feed country risk), EUDR-020 (deforestation alerts correlated with protected area proximity), and EUDR-021 (indigenous territory overlap cross-referenced with protected area overlap for comprehensive spatial risk).

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Protected area coverage | 270,000+ protected areas across 245 countries | Count of protected areas in spatial database |
| WDPA data source integration | 100% WDPA October 2025 release coverage | WDPA record count verification |
| IUCN category classification | 100% of protected areas classified by IUCN category | % of records with valid IUCN category |
| Overlap detection accuracy | >= 99% spatial precision (validated against manual GIS analysis) | Cross-validation with expert GIS review |
| Overlap detection performance | < 500ms per plot-protected area overlap query | p99 latency under load |
| Buffer zone monitoring | Configurable radii 1-50km with < 1s query time | p99 latency for buffer analysis |
| IUCN category risk scoring | 100% deterministic, reproducible | Bit-perfect reproducibility tests |
| Batch screening throughput | 10,000 plots in < 5 minutes against full WDPA | Batch processing benchmark |
| Compliance report generation | < 10 seconds per protected area compliance report | Time from request to PDF/JSON delivery |
| EUDR regulatory coverage | 100% of Articles 2, 9, 10, 11, 29, 31 protected area requirements | Regulatory compliance matrix |
| Agent integration health | 99.9% message delivery for cross-agent events | Integration health monitoring |
| Zero-hallucination guarantee | 100% deterministic calculations, no LLM in critical path | Bit-perfect reproducibility tests |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, plus the broader environmental compliance technology market estimated at 2-4 billion EUR as enforcement scales.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of EUDR-regulated commodities requiring protected area verification for sourcing regions adjacent to or overlapping protected areas (Amazon, Congo Basin, Southeast Asia, West Africa), estimated at 400-700M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 25-40M EUR in protected area compliance module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) sourcing from regions with high protected area density (Brazil, Indonesia, DRC, Colombia, Peru, Cameroon)
- Multinational food and beverage companies with cocoa, coffee, palm oil, and soya supply chains near protected tropical forests
- Timber and paper industry operators with tropical wood sourcing from production landscapes adjacent to national parks and reserves
- Compliance officers responsible for EUDR due diligence with protected area risk assessment obligations

**Secondary:**
- Certification bodies (FSC, RSPO, Rainforest Alliance) requiring protected area compliance as part of certification audits
- Commodity traders and intermediaries operating in regions with contested protected area boundaries
- Financial institutions with exposure to EUDR-regulated commodity supply chains requiring environmental due diligence under EU Taxonomy Regulation
- Conservation NGOs monitoring corporate compliance with protected area legislation
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / GIS Analysis | No license cost; flexible | Takes days per assessment; requires GIS expertise; not scalable; no IUCN-weighted scoring | Automated, sub-second queries, IUCN-category-aware risk scoring, batch screening |
| Generic Environmental Platforms (Ecometrica, Trase) | Broad environmental scope; deforestation monitoring | Not EUDR-specific; no IUCN category risk weighting; no buffer zone monitoring; no compliance workflow | Purpose-built for EUDR Article 10; IUCN category I-VI risk scoring; configurable buffer zones |
| Protected Planet / WDPA Direct Access | Authoritative data; free access | Raw data only; no compliance workflow; no batch screening; no risk scoring; no EUDR integration | WDPA data + EUDR compliance engine + batch screening + IUCN risk scoring + audit reporting |
| Certification Scheme Tools (FSC GIS Portal, RSPO RADD) | Scheme-specific compliance | Single-scheme; limited protected area categories; no cross-scheme reporting | All IUCN categories; multi-scheme reporting; comprehensive EUDR coverage |
| In-house Custom Builds | Tailored to organization | 12-18 month build; no regulatory updates; no WDPA integration updates | Ready now; continuous WDPA updates; production-grade; regulatory update pipeline |

### 2.4 Differentiation Strategy

1. **IUCN-category-aware risk scoring** -- Not a binary "in/out protected area" check; risk is weighted by IUCN management category (Ia/Ib = CRITICAL through VI = LOW).
2. **Regulatory fidelity** -- Every data field maps to a specific EUDR Article requirement; compliance reporting aligned with DDS submission format.
3. **Configurable buffer zones** -- Operators define monitoring radii per protected area type, enabling proportionate risk management.
4. **Integration depth** -- Pre-built connectors to EUDR-001 (supply chain), EUDR-002 (geolocation), EUDR-006 (boundaries), EUDR-016 (country risk), EUDR-020 (deforestation), EUDR-021 (indigenous rights), and GL-EUDR-APP.
5. **Zero-hallucination spatial analysis** -- Deterministic PostGIS calculations with SHA-256 provenance hashes; no LLM in the critical path.
6. **Scale** -- Tested for batch screening of 10,000+ plots against 270,000+ protected areas with sub-second per-plot query performance.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to achieve EUDR compliance for protected area requirements | 100% of customers pass Article 10/11 audits for protected area component | Q2 2026 |
| BG-2 | Reduce time-to-verify protected area compliance from days to minutes | 95% reduction in verification time (days to < 10 minutes per supply chain) | Q2 2026 |
| BG-3 | Become the reference protected area compliance platform for EUDR | 500+ enterprise customers using protected area module | Q4 2026 |
| BG-4 | Prevent EUDR penalties attributable to protected area non-compliance | Zero EUDR penalties for active customers related to protected area violations | Ongoing |
| BG-5 | Support EU Taxonomy Regulation alignment by providing protected area biodiversity metrics | Protected area module reusable for EU Taxonomy DNSH (Do No Significant Harm) criteria | Q1 2027 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Comprehensive protected area database | Integrate 270,000+ protected areas from WDPA with IUCN category classification and spatial indexing |
| PG-2 | IUCN-category-aware risk scoring | Score overlap risk differentially based on IUCN categories Ia through VI |
| PG-3 | Configurable buffer zone monitoring | Monitor production activities within configurable buffer zones (1-50km) around protected areas |
| PG-4 | Protected area designation validation | Validate legal status, management effectiveness, and enforcement level of protected areas |
| PG-5 | High-risk proximity alerting | Automatic alerts for supply chain plots near KBAs, World Heritage Sites, and AZE sites |
| PG-6 | Audit-ready compliance reporting | Generate compliant protected area documentation for DDS, auditors, and certifiers |
| PG-7 | Agent ecosystem integration | Enrich EUDR-001 graph, EUDR-016 country risk, and EUDR-020 deforestation alerts with protected area data |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Overlap query performance | < 500ms p99 per plot-protected area overlap query |
| TG-2 | Batch overlap screening | 10,000 plots in < 5 minutes against full WDPA |
| TG-3 | Buffer zone analysis | < 1 second per multi-radius buffer analysis (1/5/10/25/50km) |
| TG-4 | Report generation | < 10 seconds per protected area compliance report PDF |
| TG-5 | API response time | < 200ms p95 for standard queries |
| TG-6 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-7 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-8 | Data freshness | WDPA database updated within 30 days of UNEP-WCMC publication |

---

## 4. Regulatory Requirements

### 4.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(28)** | Definition of "legally produced" -- produced in accordance with the relevant legislation of the country of production, including on land use, environmental protection, and protected areas | Protected area overlap detector (F2) verifies that production does not violate protected area legislation; designation validator (F4) confirms legal status of relevant protected areas |
| **Art. 2(30-32)** | Definitions of "plot of land" and "geolocation" | Spatial overlap detection (F2) uses plot geolocation data to perform spatial intersection with protected area boundaries |
| **Art. 3** | Prohibition on placing non-compliant products on the EU market | Compliance reporting (F8) flags products sourced from plots inside or overlapping protected areas as non-compliant; severity based on IUCN category |
| **Art. 9(1)** | Geolocation of all plots of land including polygon data for plots > 4 hectares | Full agent pipeline uses Article 9 geolocation data from EUDR-002/006 as input for protected area spatial analysis |
| **Art. 10(1)** | Risk assessment -- operators shall assess and identify risk of non-compliance | Protected area risk scoring across all 9 features contributes to Article 10 risk assessment |
| **Art. 10(2)(a)** | Risk assessment criterion: complexity of relevant supply chain | Multi-tier protected area screening through supply chain graph integration (F9) assesses protected area risk at every tier |
| **Art. 10(2)(d)** | Risk assessment criterion: prevalence of deforestation or forest degradation | Protected area proximity analysis (F2, F3) correlates deforestation events with protected area boundaries; deforestation inside/near protected areas indicates elevated risk |
| **Art. 10(2)(e)** | Risk assessment criterion: concerns about the country of production | Country-level protected area governance scoring (management effectiveness, enforcement capacity) feeds EUDR-016 country risk evaluator |
| **Art. 10(2)(f)** | Risk assessment criterion: risk of circumvention or mixing with products of unknown origin | Protected area screening for all identified origin plots; flags products from unscreened origins as elevated protected area risk |
| **Art. 11** | Risk mitigation measures | Buffer zone monitoring (F3), compliance tracker (F6), and remediation workflows provide structured risk mitigation tools |
| **Art. 29(3)** | Country benchmarking -- Commission shall consider rate of deforestation, expansion of agriculture, governance of protected areas | Protected area governance scoring engine provides per-country protected area enforcement effectiveness score to EUDR-016 for Article 29 benchmarking |
| **Art. 31** | Record keeping for 5 years | All protected area data, overlap analyses, compliance records, violation alerts, and reports retained for minimum 5 years with immutable audit trail |

### 4.2 International Legal Framework

| Legal Instrument | Status | Agent Relevance |
|-----------------|--------|----------------|
| **Convention on Biological Diversity (CBD)** | 196 parties; Kunming-Montreal Global Biodiversity Framework (GBF) Target 3: 30% of land protected by 2030 | Protected area database covers all CBD-designated areas; agent supports 30x30 target monitoring |
| **CITES** (Convention on International Trade in Endangered Species, 1975) | 184 parties; governs trade in specimens of wild species | Species presence assessment (F7) cross-references CITES-listed species with protected areas near supply chains |
| **World Heritage Convention** (UNESCO, 1972) | 195 parties; 1,199 World Heritage Sites | Designation validator (F4) flags World Heritage Sites as highest-designation protected areas; overlap triggers CRITICAL alert |
| **Ramsar Convention** (1971) | 172 parties; 2,500+ Wetlands of International Importance | Ramsar sites integrated into protected area database (F1); overlap detection flags Ramsar wetlands |
| **EU Biodiversity Strategy 2030** | EU policy; 30% land/sea protected, 10% strictly | Agent supports EU policy tracking for protected area coverage assessment |
| **EU CSDDD** (Corporate Sustainability Due Diligence Directive, 2024) | Effective 2027 for large companies | Agent infrastructure designed for CSDDD Article 7 environmental due diligence reuse |
| **EU Taxonomy Regulation** (2020/852) | Active; DNSH criteria for biodiversity | Protected area data supports EU Taxonomy Article 17 DNSH assessment |

### 4.3 IUCN Protected Area Categories

| IUCN Category | Name | Description | Commodity Production Allowed | Agent Risk Classification |
|--------------|------|-------------|------------------------------|---------------------------|
| **Ia** | Strict Nature Reserve | Strictly protected for biodiversity; minimal human visitation | Absolutely prohibited | CRITICAL (100) |
| **Ib** | Wilderness Area | Large unmodified area without permanent human habitation | Absolutely prohibited | CRITICAL (100) |
| **II** | National Park | Large natural area for ecosystem protection and recreation | Prohibited (except limited traditional use) | CRITICAL (95) |
| **III** | Natural Monument | Specific natural feature of outstanding value | Prohibited | HIGH (85) |
| **IV** | Habitat/Species Management Area | Area for active management intervention for conservation | Generally prohibited; limited exceptions | HIGH (80) |
| **V** | Protected Landscape/Seascape | Landscape shaped by human interaction; conservation of nature and culture | Conditionally allowed (sustainable practices) | MEDIUM (50) |
| **VI** | Protected Area with Sustainable Use | Conservation of ecosystems with sustainable natural resource management | Allowed under management plan | LOW (25) |
| **Not Reported** | Unclassified | IUCN category not assigned or not reported to WDPA | Uncertain -- treated as HIGH by default | HIGH (75) |

### 4.4 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for protected area encroachment assessment |
| June 29, 2023 | EUDR entered into force | Legal basis for protected area compliance verification |
| December 2022 | Kunming-Montreal GBF adopted | 30x30 target drives protected area expansion; agent must handle dynamic boundaries |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | All protected area verification must be operational |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; agent must handle 10x scale |
| 2027 | CSDDD enforcement begins | Protected area module reused for CSDDD environmental due diligence |
| 2030 | CBD 30x30 target deadline | Protected area database must scale to accommodate 30% land coverage |
| Ongoing (quarterly) | DDS submission deadlines | Protected area compliance reports must be current for quarterly filing |
| Ongoing (monthly) | WDPA database updates by UNEP-WCMC | Agent ingests updated protected area boundaries within 30 days |

---

## 5. Scope and Zero-Hallucination Principles

### 5.1 Scope -- In and Out

**In Scope (v1.0):**
- Protected area database with 270,000+ areas from WDPA/Protected Planet, national registries, regional networks, and KBA database
- IUCN-category-aware spatial overlap detection between production plots and protected areas using PostGIS
- Configurable buffer zone monitoring (1-50km radii) with encroachment detection and proximity alerts
- Protected area designation validation (international, national, local) with management effectiveness and enforcement assessment
- High-risk proximity alert system for KBAs, World Heritage Sites, AZE sites, and Ramsar Wetlands
- Protected area compliance tracking with violation detection and remediation workflow
- Conservation status assessment with IUCN Red List integration and habitat fragmentation analysis
- Compliance reporting for DDS, auditors, and certification schemes (8 report types, 5 formats, 5 languages)
- Integration with EUDR-001, EUDR-002, EUDR-006, EUDR-016, EUDR-020, EUDR-021

**Out of Scope (v1.0):**
- Real-time satellite monitoring of protected area boundaries (defer to EUDR-003/EUDR-020 integration)
- Protected area management plan advisory (agent provides data and compliance assessment; not conservation counsel)
- Payment for Ecosystem Services (PES) management (agent tracks protected areas; does not manage PES payments)
- Marine protected area monitoring for fisheries (terrestrial focus for EUDR v1.0)
- Mobile native application for field-level protected area inspection (web responsive only)
- Predictive ML models for future protected area designation forecasting (defer to Phase 2)
- Direct submission to EU Information System (handled by GL-EUDR-APP DDS module)
- Blockchain-based protected area records (SHA-256 provenance hashes provide sufficient integrity)

### 5.2 Zero-Hallucination Principles

All 9 features in this agent operate under strict zero-hallucination guarantees:

| Principle | Implementation |
|-----------|---------------|
| **Deterministic calculations** | Same inputs always produce the same overlap results, risk scores, and buffer zone assessments (bit-perfect reproducibility) |
| **No LLM in critical path** | All overlap detection, IUCN risk scoring, buffer zone monitoring, and compliance assessment use deterministic PostGIS and arithmetic operations only |
| **Authoritative data sources only** | All protected area boundaries sourced from WDPA/Protected Planet, national registries (ICMBio, KLHK, ICCN), and the KBA database; no synthetic boundaries |
| **Full provenance tracking** | Every overlap result, risk score, and compliance assessment includes SHA-256 hash, data source citations, WDPA ID reference, and calculation timestamps |
| **Immutable audit trail** | All protected area data changes recorded in `gl_eudr_pav_audit_log` with before/after values |
| **Decimal arithmetic** | Risk scores, overlap percentages, and buffer distances use Decimal type to prevent floating-point drift |
| **Version-controlled data** | Protected area databases are versioned by WDPA release month; any boundary update creates a new version with timestamp and source attribution |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core protected area verification engine; Features 6-9 form the compliance, reporting, and integration layer.

**P0 Features 1-5: Core Protected Area Verification Engine**

---

#### Feature 1: Protected Area Database Integration

**User Story:**
```
As a compliance officer,
I want a comprehensive, georeferenced database of protected areas worldwide with IUCN category classification,
So that I can determine whether my supply chain production plots are located in, adjacent to, or near legally protected ecosystems.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F1.1: Integrates the full World Database on Protected Areas (WDPA) from UNEP-WCMC/Protected Planet, containing 270,000+ protected areas across 245 countries and territories, with polygon boundaries in WGS84 coordinate reference system and GeoJSON/Shapefile format
- [ ] F1.2: Classifies each protected area by IUCN management category (Ia, Ib, II, III, IV, V, VI, Not Reported) with validated category assignment from WDPA source data; tracks category changes over time
- [ ] F1.3: Integrates national protected area registries for major EUDR commodity-producing countries: ICMBio/CNUC (Brazil) for Conservation Units, KLHK (Indonesia) for Kawasan Konservasi, ICCN (DRC) for National Parks and Reserves, SERNANP (Peru) for Natural Protected Areas, SINAP (Colombia) for Protected Areas System
- [ ] F1.4: Integrates Key Biodiversity Areas (KBA) database from BirdLife International/IUCN, covering 16,000+ sites identified as globally significant for biodiversity, with spatial boundaries and trigger species data
- [ ] F1.5: Integrates UNESCO World Heritage Sites (natural and mixed), Ramsar Wetlands of International Importance, and UNESCO Man and Biosphere (MAB) reserves as highest-designation protected areas with separate tracking and alert thresholds
- [ ] F1.6: Stores each protected area with structured metadata: WDPA ID, name, designation type (international/national/regional/local), designation name, IUCN category, country code, total area (hectares), marine area (hectares), status (designated/proposed/inscribed), status year, governance type (government/shared/private/indigenous), management authority, parent ISO3 code, and data source with provenance hash
- [ ] F1.7: Maintains spatial index (PostGIS GIST) enabling sub-second bounding box queries across 270,000+ protected area polygons; supports point-in-polygon, polygon-polygon intersection, and distance queries with spatial indexing
- [ ] F1.8: Implements protected area version control: WDPA releases are versioned by month (e.g., WDPA_Oct2025); each update creates a new data version with effective date, record additions/removals/modifications, and previous boundaries preserved for audit trail
- [ ] F1.9: Provides data freshness tracking with WDPA release schedule monitoring, staleness alerts when data exceeds configurable age threshold (default: 60 days since last WDPA release), and automated refresh triggers when new WDPA release is detected
- [ ] F1.10: Supports protected area search and filtering by: country, IUCN category, designation type, governance type, area range, status, bounding box, and proximity to a given coordinate

**Non-Functional Requirements:**
- Coverage: 270,000+ protected areas across 245 countries and territories
- Spatial Precision: Protected area boundaries stored with coordinates at 6+ decimal places (sub-meter accuracy)
- Performance: Point-in-polygon query < 100ms; batch screening of 10,000 plots < 5 minutes
- Data Quality: WDPA ID reference for every protected area record; confidence scoring for boundary accuracy based on WDPA GIS quality field

**Dependencies:**
- WDPA/Protected Planet data download (monthly releases from UNEP-WCMC)
- KBA database from BirdLife International
- National protected area registries (ICMBio, KLHK, ICCN, SERNANP, SINAP)
- AGENT-DATA-006 GIS/Mapping Connector for spatial operations
- PostGIS extension for spatial indexing and queries

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 GIS specialist)

**Edge Cases:**
- Protected areas without polygon boundary (point-only reference in WDPA) -> Create circular buffer using WDPA reported area (radius = sqrt(area/pi)); flag as approximate
- Overlapping protected areas (e.g., national park inside biosphere reserve) -> Record all designations; use highest-risk IUCN category for scoring
- Transboundary protected areas spanning multiple countries -> Store with primary country; cross-reference in all affected countries; unified boundary polygon
- Protected area recently de-gazetted or boundary reduced -> Track status changes; retain historical boundary for audit; flag as status_changed

---

#### Feature 2: Spatial Overlap Detection Engine

**User Story:**
```
As a compliance officer,
I want to automatically detect when my supply chain production plots overlap with or are located inside protected areas,
So that I can identify plots violating protected area legislation and assess EUDR compliance risk with IUCN-category-appropriate severity.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F2.1: Performs spatial intersection analysis between production plot polygons/points and protected area polygons using PostGIS ST_Intersects, ST_Within, ST_Contains, and ST_DWithin functions
- [ ] F2.2: Classifies overlap type into 5 categories: INSIDE (plot entirely within protected area), PARTIAL (polygon intersection with partial overlap), BOUNDARY (plot boundary touches protected area boundary), BUFFER (plot within configurable buffer zone, not overlapping), CLEAR (plot outside all buffer zones)
- [ ] F2.3: Calculates overlap metrics: overlap area (hectares), overlap percentage of plot area, overlap percentage of protected area, minimum distance from plot centroid to nearest protected area boundary (meters), minimum distance from plot boundary to protected area boundary (meters), and bearing from plot centroid to nearest protected area
- [ ] F2.4: Applies IUCN-category-weighted risk scoring: Category Ia/Ib = 100 (CRITICAL), Category II = 95 (CRITICAL), Category III = 85 (HIGH), Category IV = 80 (HIGH), Category V = 50 (MEDIUM), Category VI = 25 (LOW), Not Reported = 75 (HIGH default). Risk score further weighted by overlap type (INSIDE = 1.0x, PARTIAL = 0.8x, BOUNDARY = 0.6x, BUFFER = 0.3x)
- [ ] F2.5: Returns all affected protected areas for each overlap, including: WDPA ID, protected area name, IUCN category, designation type (national park, wildlife reserve, etc.), governance type, management authority, and country
- [ ] F2.6: Supports batch overlap screening: upload 10,000+ plot coordinates/polygons and screen against entire WDPA database in single operation with progress tracking and partial results delivery
- [ ] F2.7: Detects multi-designation overlaps: when a plot overlaps with multiple protected areas (e.g., national park inside a World Heritage Site inside a biosphere reserve), reports all overlaps and applies highest-risk scoring
- [ ] F2.8: Cross-references overlap results with deforestation alerts from EUDR-020, flagging plots where both protected area overlap AND post-cutoff deforestation are detected as MAXIMUM CRITICAL compliance risk
- [ ] F2.9: Produces overlap analysis report per plot with map visualization (GeoJSON export), affected protected area details, IUCN category risk assessment, and recommended compliance actions
- [ ] F2.10: Tracks overlap status changes over time: new protected areas designated, boundary updates from WDPA releases, overlap reclassification due to plot boundary changes, and generates change alerts for operators

**Overlap Risk Scoring Formula:**
```
Protected_Area_Risk = (
    iucn_category_score * overlap_type_multiplier * 0.50
    + designation_level_score * 0.20       # UNESCO=100, National=70, Regional=50, Local=30, Proposed=20
    + management_effectiveness_gap * 0.15  # 100 - METT score (higher gap = higher risk)
    + country_enforcement_gap * 0.15       # 100 - enforcement score from EUDR-016
)

Where:
    iucn_category_score:   Ia/Ib=100, II=95, III=85, IV=80, V=50, VI=25, NR=75
    overlap_type_multiplier: INSIDE=1.0, PARTIAL=0.8, BOUNDARY=0.6, BUFFER=0.3

Classification:
  CRITICAL:  Protected_Area_Risk >= 80
  HIGH:      60 <= Protected_Area_Risk < 80
  MEDIUM:    40 <= Protected_Area_Risk < 60
  LOW:       20 <= Protected_Area_Risk < 40
  CLEAR:     Protected_Area_Risk < 20 (no overlap or buffer-only with Category VI)
```

**Non-Functional Requirements:**
- Spatial Precision: Overlap calculations accurate to 1-meter resolution
- Performance: Single-plot query < 500ms; batch 10,000 plots < 5 minutes
- Determinism: Same plot-protected area combination always produces identical overlap result
- Auditability: SHA-256 provenance hash on every overlap analysis

**Dependencies:**
- Feature 1 (Protected Area Database) for protected area polygons
- AGENT-EUDR-002 Geolocation Verification Agent for plot coordinate validation
- AGENT-EUDR-006 Plot Boundary Manager Agent for plot polygon data
- PostGIS extension for spatial operations
- AGENT-DATA-006 GIS/Mapping Connector for spatial utilities

**Estimated Effort:** 3 weeks (1 backend engineer, 1 GIS specialist)

**Edge Cases:**
- Plot overlaps a protected area polygon that has known boundary inaccuracies (low WDPA GIS quality) -> Flag overlap as UNCERTAIN; include confidence level in result; recommend manual verification
- Plot is located in a corridor between two protected areas -> Report proximity to both; apply buffer zone rules for both; flag as ecological corridor risk
- Protected area has been proposed but not yet formally gazetted -> Report as PROPOSED designation; apply reduced risk multiplier (0.5x) but still flag for monitoring

---

#### Feature 3: Buffer Zone Monitoring

**User Story:**
```
As a compliance officer,
I want to monitor production activities within configurable buffer zones around protected areas,
So that I can detect encroachment trends, enforce national buffer zone regulations, and proactively manage protected area proximity risk.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F3.1: Supports configurable buffer zone radii around protected areas: 1km, 2km, 5km, 10km, 25km, and 50km, with operator-customizable radii per IUCN category and per country (e.g., Brazil 10km buffer for Category II, Indonesia 5km for Category IV)
- [ ] F3.2: Generates buffer zone polygons around protected area boundaries using PostGIS ST_Buffer with configurable segment resolution (default: 64-point polygon approximation) and stores computed buffers in spatial index for efficient querying
- [ ] F3.3: Detects all production plots within each buffer zone tier and classifies proximity: IMMEDIATE (0-1km), CLOSE (1-5km), MODERATE (5-10km), DISTANT (10-25km), PERIPHERAL (25-50km)
- [ ] F3.4: Tracks encroachment trends over time: monitors whether plots are moving closer to or further from protected area boundaries across successive assessments; calculates encroachment velocity (distance change per quarter)
- [ ] F3.5: Generates proximity alerts when new plots are detected within buffer zones, when existing plots expand toward protected area boundaries, or when new protected areas are designated near existing plots
- [ ] F3.6: Supports national buffer zone regulations: Brazil (10km buffer for IUCN Category II+ per Forest Code), Indonesia (buffer zones per PP 28/2011), Colombia (buffer zones per Decree 2372/2010), with configurable country-specific buffer enforcement rules
- [ ] F3.7: Calculates buffer zone density metrics: number of production plots per km2 within each buffer tier, total production area within buffer, and commodity concentration by buffer tier
- [ ] F3.8: Cross-references buffer zone plots with deforestation alerts from EUDR-020: deforestation detected within protected area buffer zones triggers ELEVATED risk classification
- [ ] F3.9: Generates buffer zone monitoring reports per protected area: list of plots within each tier, trend analysis (expanding/stable/contracting encroachment), and compliance recommendations
- [ ] F3.10: Supports buffer zone exemption management: operator can record exemptions for plots within buffer zones that have valid legal permits (e.g., pre-existing land titles, environmental licenses) with documentation upload and expiry tracking

**Non-Functional Requirements:**
- Performance: Multi-radius buffer analysis < 1 second per protected area
- Coverage: Buffer zone monitoring for all 270,000+ protected areas
- Precision: Buffer distances accurate to 10-meter resolution
- Configurability: Buffer radii customizable per operator, per IUCN category, per country without code changes

**Dependencies:**
- Feature 1 (Protected Area Database) for protected area boundaries
- Feature 2 (Overlap Detection) for spatial analysis infrastructure
- AGENT-EUDR-020 Deforestation Alert System for deforestation-buffer correlation
- PostGIS ST_Buffer for buffer polygon generation

**Estimated Effort:** 3 weeks (1 backend engineer, 1 GIS specialist)

**Edge Cases:**
- Buffer zones overlap with adjacent protected areas -> Merge overlapping buffers; attribute to nearest protected area
- Plot with valid pre-existing permit within buffer zone -> Exemption recorded; reduced risk scoring; exemption expiry monitoring
- National law mandates larger buffer than operator's configured radius -> Apply the larger (more restrictive) buffer; flag regulatory override

---

#### Feature 4: Protected Area Designation Validator

**User Story:**
```
As a compliance officer,
I want to verify the legal designation status, management effectiveness, and enforcement capacity of protected areas near my supply chain,
So that I can accurately assess the regulatory and reputational risk of protected area proximity.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F4.1: Classifies protected area designation level into 4 tiers: INTERNATIONAL (UNESCO World Heritage, Ramsar, MAB Biosphere Reserve), NATIONAL (national park, national reserve, wildlife sanctuary), REGIONAL (state/provincial protected area, regional park), LOCAL (municipal reserve, community conserved area, private reserve)
- [ ] F4.2: Validates protected area legal status from WDPA status field: DESIGNATED (formally gazetted under national law), PROPOSED (under consideration, not yet gazetted), INSCRIBED (internationally inscribed, e.g., World Heritage), ADOPTED (adopted but not yet enforced), ESTABLISHED (established without formal gazettement)
- [ ] F4.3: Tracks protected area governance type from WDPA: Federal/National Ministry (governance_type = "Federal or national ministry or agency"), Sub-national (state, regional), Collaborative (shared governance), Private (for-profit or non-profit), Indigenous and Local Communities (community conserved), Not Reported
- [ ] F4.4: Integrates Management Effectiveness Tracking Tool (METT) scores where available from GD-PAME (Global Database on Protected Areas Management Effectiveness): overall effectiveness score (0-100), threat assessment, resource adequacy, and governance quality
- [ ] F4.5: Assesses enforcement level using proxy indicators: country environmental enforcement score from EUDR-016, recent violation/penalty records for the protected area, ranger density (where available from SMART data), and budget allocation adequacy
- [ ] F4.6: Flags UNESCO World Heritage Sites, Ramsar Wetlands, and Alliance for Zero Extinction (AZE) sites with special HIGH-DESIGNATION alert tier that triggers mandatory enhanced due diligence regardless of IUCN category
- [ ] F4.7: Tracks designation history: changes in protected area status (expansion, reduction, de-gazettement, re-designation, IUCN category change) with effective dates and source documentation
- [ ] F4.8: Generates designation validation report per protected area: legal basis, management authority, IUCN category rationale, governance assessment, management effectiveness score, enforcement assessment, and overall designation strength score (0-100)
- [ ] F4.9: Monitors upcoming designation changes: proposed protected areas, pending boundary revisions, UNESCO/Ramsar nominations, and generates proactive alerts for operators with supply chains in affected regions
- [ ] F4.10: Provides per-country protected area system assessment: total protected area coverage (% of land), IUCN category distribution, management effectiveness average, and protected area system governance score for EUDR-016 country risk input

**Designation Strength Score:**
```
Designation_Strength = (
    designation_level_score * 0.30     # International=100, National=75, Regional=50, Local=25
    + legal_status_score * 0.25        # Designated=100, Inscribed=100, Established=70, Adopted=50, Proposed=30
    + management_effectiveness * 0.25  # METT score (0-100); default 50 if unavailable
    + enforcement_assessment * 0.20    # Country enforcement score + PA-specific indicators (0-100)
)
```

**Non-Functional Requirements:**
- Coverage: Designation validation for all protected areas in WDPA
- Data Sources: WDPA, GD-PAME, UNESCO WH List, Ramsar Sites Database, AZE database
- Performance: Designation lookup < 100ms per protected area
- Auditability: SHA-256 provenance hash on every designation assessment

**Dependencies:**
- Feature 1 (Protected Area Database) for protected area base data
- GD-PAME database for management effectiveness scores
- EUDR-016 Country Risk Evaluator for enforcement scoring
- UNESCO World Heritage List, Ramsar Sites Database, AZE database

**Estimated Effort:** 2 weeks (1 senior backend engineer)

---

#### Feature 5: High-Risk Proximity Alert System

**User Story:**
```
As a compliance officer,
I want to be automatically alerted when my supply chain plots are near protected areas of exceptional biodiversity significance,
So that I can trigger enhanced due diligence and manage the highest-risk protected area exposures.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F5.1: Defines high-risk protected area categories: UNESCO World Heritage Sites (natural/mixed), Ramsar Wetlands of International Importance, Alliance for Zero Extinction (AZE) sites, Key Biodiversity Areas (KBAs) with confirmed global trigger species, IUCN Category Ia/Ib Strict Nature Reserves, and nationally designated critical habitats
- [ ] F5.2: Generates HIGH-PRIORITY alerts when supply chain plots are detected within 25km of any high-risk protected area, with alert severity escalating based on proximity: CRITICAL (inside or < 1km), SEVERE (1-5km), HIGH (5-10km), ELEVATED (10-25km)
- [ ] F5.3: Calculates composite proximity risk score incorporating: distance to protected area (40%), IUCN category severity (25%), designation level (15%), number of overlapping high-risk designations (10%), and deforestation trend in vicinity (10%)
- [ ] F5.4: Generates supply chain impact assessment for each high-risk alert: lists affected suppliers, plots, commodities, products, and estimated supply chain exposure (% of volume from affected region)
- [ ] F5.5: Triggers mandatory enhanced due diligence workflow when alert severity reaches CRITICAL or SEVERE for high-risk protected areas, requiring operator acknowledgment and documented response within configurable SLA (default: 14 days for CRITICAL, 30 days for SEVERE)
- [ ] F5.6: Supports operator-configurable alert rules: custom proximity thresholds per protected area type, commodity-specific alert sensitivity, notification channel preferences (email, webhook, in-app), and alert suppression for acknowledged/mitigated risks
- [ ] F5.7: Maintains alert history with trend analysis: alert frequency by country, region, protected area type, and commodity; trend direction (improving/stable/deteriorating) calculated on 12-month rolling window
- [ ] F5.8: Cross-references high-risk proximity alerts with EUDR-021 indigenous rights alerts: plots that are near both high-risk protected areas AND indigenous territories receive COMPOUND CRITICAL classification
- [ ] F5.9: Generates alert digest reports: daily/weekly/monthly summary of new alerts, resolved alerts, pending alerts, and overall protected area risk exposure for operator dashboard
- [ ] F5.10: Implements alert deduplication: when multiple protected area designations overlap at the same location (e.g., national park + World Heritage Site + KBA), consolidates into a single alert with all designations listed rather than generating duplicate alerts

**Proximity Risk Score:**
```
Proximity_Risk = (
    distance_score * 0.40              # inside=100, <1km=95, 1-5km=80, 5-10km=60, 10-25km=40, >25km=0
    + iucn_category_score * 0.25       # Ia/Ib=100, II=95, III=85, IV=80, V=50, VI=25, NR=75
    + designation_level_score * 0.15   # UNESCO_WH=100, Ramsar=90, AZE=95, KBA=80, National=70
    + multi_designation_bonus * 0.10   # +20 per additional high-risk designation (max 100)
    + deforestation_trend * 0.10       # active_deforestation=100, recent=60, historical=30, none=0
)

Classification:
  CRITICAL:  Proximity_Risk >= 85
  SEVERE:    70 <= Proximity_Risk < 85
  HIGH:      55 <= Proximity_Risk < 70
  ELEVATED:  40 <= Proximity_Risk < 55
  STANDARD:  Proximity_Risk < 40
```

**Non-Functional Requirements:**
- Latency: Alert generation < 2 seconds from overlap detection completion
- Coverage: All high-risk protected area categories monitored
- Determinism: Proximity risk scoring is deterministic, bit-perfect reproducible
- SLA Tracking: Minute-level precision for enhanced DD response deadlines

**Dependencies:**
- Feature 1 (Protected Area Database) for protected area data
- Feature 2 (Overlap Detection) for spatial analysis
- Feature 4 (Designation Validator) for designation-level scoring
- EUDR-020 Deforestation Alert System for deforestation trend data
- EUDR-021 Indigenous Rights Checker for compound risk cross-reference

**Estimated Effort:** 2 weeks (1 backend engineer)

---

**P0 Features 6-9: Compliance, Reporting, and Integration Layer**

> Features 6, 7, 8, and 9 are P0 launch blockers. Without compliance tracking, conservation assessment, compliance reporting, and agent ecosystem integration, the core verification engine cannot deliver end-user value. These features are the delivery mechanism through which compliance officers, auditors, and the broader EUDR platform interact with the verification engine.

---

#### Feature 6: Protected Area Compliance Tracker

**User Story:**
```
As a compliance officer,
I want to systematically track protected area boundary violations, encroachment incidents, and remediation progress,
So that I can demonstrate to auditors and regulators that protected area risks are actively managed.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F6.1: Tracks compliance status per plot-protected area pair through lifecycle stages: DETECTED (overlap identified), INVESTIGATING (operator reviewing), VIOLATION_CONFIRMED (protected area encroachment confirmed), REMEDIATION_PLANNED (remediation actions defined), REMEDIATION_IN_PROGRESS (actions underway), REMEDIATED (encroachment resolved), EXEMPTION_GRANTED (valid legal permit confirmed), FALSE_POSITIVE (overlap was data error)
- [ ] F6.2: Records violation details: violation type (encroachment, illegal clearing, unauthorized construction, resource extraction), area affected (hectares), detection method (WDPA overlap, satellite alert, field inspection), detection date, evidence documentation, and severity classification (CRITICAL/HIGH/MEDIUM/LOW)
- [ ] F6.3: Manages remediation workflows with configurable SLA timelines: investigation (14 days), remediation plan (30 days), remediation execution (90 days), verification (30 days); with escalation triggers at 50%, 75%, and 100% of SLA
- [ ] F6.4: Tracks exemption records for plots with valid legal permits within protected area buffer zones: permit type, issuing authority, permit number, effective date, expiry date, conditions, and document upload with SHA-256 hash
- [ ] F6.5: Generates compliance timeline per plot: chronological view of all compliance events (detection, investigation, remediation, resolution) with document links and actor attribution
- [ ] F6.6: Calculates per-operator protected area compliance score (0-100) based on: percentage of plots fully compliant, number of active violations, remediation SLA compliance rate, and exemption coverage
- [ ] F6.7: Supports multi-operator compliance view: when multiple operators source from the same region, aggregated compliance view shows regional protected area pressure and operator-specific contribution
- [ ] F6.8: Tracks compliance status changes with immutable audit trail: every status transition recorded with timestamp, actor, reason, supporting evidence, and SHA-256 provenance hash
- [ ] F6.9: Generates compliance SLA dashboard: total violations, violations by stage, SLA compliance rate per stage, overdue items with escalation status, and trend analysis
- [ ] F6.10: Supports compliance data export for external audit: complete compliance dossier per plot, per protected area, or per operator with chronological activity log, document inventory, and status history

**Non-Functional Requirements:**
- Completeness: 100% of compliance events tracked with timestamped audit trail
- SLA Tracking: Minute-level precision for all deadlines
- Retention: All compliance records retained for minimum 5 years per EUDR Article 31
- Performance: Compliance dashboard loads in < 2 seconds for operators with 10,000+ plots

**Dependencies:**
- Feature 2 (Overlap Detection) for overlap identification
- Feature 3 (Buffer Zone Monitoring) for buffer zone compliance
- GL-EUDR-APP notification service for SLA alerts and escalation

**Estimated Effort:** 3 weeks (1 backend engineer)

---

#### Feature 7: Conservation Status Assessment

**User Story:**
```
As a compliance officer,
I want to understand the biodiversity significance and conservation status of protected areas near my supply chain,
So that I can make informed risk decisions and demonstrate environmental due diligence to stakeholders.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F7.1: Integrates IUCN Red List data to identify threatened species present in protected areas near supply chain operations: species name, Red List category (CR, EN, VU, NT, LC), population trend, and habitat requirements
- [ ] F7.2: Calculates biodiversity significance score per protected area: based on number of globally threatened species (CR/EN/VU), number of CITES-listed species, presence of AZE trigger species, and habitat type rarity using IUCN Habitat Classification Scheme
- [ ] F7.3: Assesses habitat fragmentation for protected areas experiencing agricultural pressure: fragmentation index based on protected area perimeter-to-area ratio, edge effects from adjacent agricultural plots, and corridor connectivity to other protected areas
- [ ] F7.4: Tracks ecosystem service value indicators for protected areas: carbon storage (tCO2e from biomass), watershed protection function, and biodiversity intactness index (where available from PREDICTS database)
- [ ] F7.5: Generates conservation context report per protected area: biome classification, dominant habitat types, species richness indicators, threat assessment (agricultural expansion, logging, mining, infrastructure), and conservation priority ranking
- [ ] F7.6: Cross-references supply chain commodity types with habitat threat profiles: e.g., palm oil production is primary driver of lowland rainforest loss in Southeast Asia; soya expansion drives Cerrado savanna conversion in Brazil
- [ ] F7.7: Provides species-commodity impact matrix: which EUDR commodities pose the greatest threat to which species groups in each sourcing region, enabling operators to prioritize mitigation efforts
- [ ] F7.8: Monitors conservation status changes: new IUCN Red List assessments, species uplisting/downlisting events, new KBA identifications, and generates alerts when conservation significance of supply chain-adjacent protected areas increases
- [ ] F7.9: Calculates composite conservation risk score per supply chain combining: protected area proximity risk (Feature 2/5), biodiversity significance (this feature), deforestation trend (EUDR-020), and indigenous rights risk (EUDR-021)
- [ ] F7.10: Supports conservation data in compliance reports: biodiversity significance summary for DDS Article 10 risk assessment section, enabling operators to demonstrate awareness of conservation context

**Biodiversity Significance Score:**
```
Biodiversity_Score = (
    threatened_species_count * 0.35     # Normalized: 0 = 0, 1-5 = 40, 6-20 = 70, 21+ = 100
    + cites_species_count * 0.15        # Normalized: 0 = 0, 1-3 = 40, 4-10 = 70, 11+ = 100
    + aze_trigger_presence * 0.20       # AZE trigger species present = 100, absent = 0
    + habitat_rarity * 0.15            # Based on global extent: <1% = 100, 1-5% = 70, >5% = 40
    + connectivity_index * 0.15         # Connected to corridor = 100, isolated = 30
)
```

**Non-Functional Requirements:**
- Data Currency: IUCN Red List data updated within 60 days of new assessment publication
- Coverage: Species data for protected areas in all EUDR commodity-producing regions
- Determinism: All scores deterministic, bit-perfect reproducible
- Performance: Conservation assessment < 500ms per protected area

**Dependencies:**
- Feature 1 (Protected Area Database) for protected area location
- IUCN Red List API for species data
- CITES Species Database for trade-regulated species
- KBA database for trigger species
- EUDR-020 Deforestation Alert System for deforestation trend context
- EUDR-021 Indigenous Rights Checker for compound risk assessment

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data engineer)

---

#### Feature 8: Compliance Reporting Engine

**User Story:**
```
As a compliance officer,
I want automated, audit-ready protected area compliance reports,
So that I can include them in my Due Diligence Statement and present them to auditors, certifiers, and regulators.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F8.1: Generates comprehensive protected area compliance reports in PDF, JSON, and HTML formats, including spatial overlap analysis, IUCN category risk assessment, buffer zone monitoring results, designation validation, conservation significance, and compliance tracking status
- [ ] F8.2: Generates DDS-integrated protected area section: structured risk assessment narrative for EUDR Article 4(2) submission, covering protected area screening methodology, overlap results, IUCN category analysis, buffer zone assessment, and mitigation measures
- [ ] F8.3: Produces certification scheme compliance reports: FSC High Conservation Value (HCV) protected area evidence (Principle 9), RSPO High Conservation Value Area compliance (RSPO P&C 7.12), Rainforest Alliance Critical Ecosystem requirements, formatted per scheme-specific audit checklists
- [ ] F8.4: Generates supplier-level protected area scorecards: per-supplier summary of protected area overlap count by IUCN category, buffer zone plot count by proximity tier, compliance status distribution, active violations, and overall protected area risk rating
- [ ] F8.5: Produces trend reports showing protected area compliance evolution over time per supply chain, commodity, or country, with annotated events (new protected areas, boundary changes, violation detections, remediation completions)
- [ ] F8.6: Creates executive summary reports with key indicators: total plots screened, overlap count by IUCN category, buffer zone plot distribution by tier, compliance score, active violations, high-risk alerts, and overall protected area readiness score
- [ ] F8.7: Exports dashboard data packages for BI integration (Grafana, Tableau, Power BI) in CSV, JSON, and XLSX formats with standardized schema for protected area metrics
- [ ] F8.8: Includes complete audit trail in reports: every overlap analysis, risk score, and compliance event linked to source data, WDPA version, calculation method, and SHA-256 provenance hash
- [ ] F8.9: Supports multi-language report generation in English (EN), French (FR), German (DE), Spanish (ES), and Portuguese (PT) using translated templates with jurisdiction-appropriate legal terminology
- [ ] F8.10: Maintains report generation history with version control: reports are immutable once generated; updated assessments produce new report versions, not overwrites

**Non-Functional Requirements:**
- Performance: PDF generation < 10 seconds per supply chain compliance report
- Quality: Reports pass WCAG 2.1 AA accessibility standards
- Size: PDF reports optimized to < 10 MB each
- Formats: PDF (audit), JSON (API integration), HTML (web display), CSV/XLSX (BI export)

**Dependencies:**
- Features 1-7 for all protected area assessment data
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
I want the Protected Area Validator to enrich existing EUDR agents with protected area compliance data,
So that protected area risk is embedded throughout the supply chain compliance workflow.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F9.1: Integrates with EUDR-001 (Supply Chain Mapping Master): enriches supply chain graph nodes with protected area risk attributes (overlap status, IUCN category exposure, buffer zone proximity, conservation significance) for risk propagation through the supply chain
- [ ] F9.2: Integrates with EUDR-002 (Geolocation Verification Agent): validates plot coordinates against protected area boundaries as part of geolocation verification; flags plots requiring protected area review
- [ ] F9.3: Integrates with EUDR-006 (Plot Boundary Manager Agent): consumes plot polygon boundaries for precise overlap analysis; receives plot boundary update notifications for re-screening
- [ ] F9.4: Integrates with EUDR-016 (Country Risk Evaluator): provides per-country protected area governance scores (management effectiveness, enforcement capacity, % land protected) that feed the environmental governance index; receives country risk classification for enforcement scoring context
- [ ] F9.5: Integrates with EUDR-017 (Supplier Risk Scorer): provides per-supplier protected area risk scores; suppliers with unresolved protected area overlaps or buffer zone violations receive elevated risk scores
- [ ] F9.6: Integrates with EUDR-020 (Deforestation Alert System): receives deforestation alerts and correlates with protected area proximity; deforestation alerts inside protected areas or their buffer zones generate MAXIMUM CRITICAL protected area alerts
- [ ] F9.7: Integrates with EUDR-021 (Indigenous Rights Checker): cross-references protected area overlaps with indigenous territory overlaps; plots with compound protected area + indigenous rights risk receive elevated compound risk classification
- [ ] F9.8: Provides standardized protected area data API consumed by GL-EUDR-APP v1.0: protected area map overlay for visualization, overlap status dashboard, alert feed, compliance tracker, and report download
- [ ] F9.9: Publishes protected area events to the GreenLang event bus for consumption by other agents: protected_area_overlap_detected, buffer_zone_encroachment_detected, high_risk_alert_generated, compliance_status_changed, compliance_report_generated, protected_area_boundary_updated
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
- AGENT-EUDR-021 Indigenous Rights Checker (BUILT 100%)
- GL-EUDR-APP v1.0 Platform (BUILT 100%)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Protected Area Impact Dashboard
- Interactive map showing protected areas overlaid with supply chain plots with IUCN category color coding
- Heat map visualization of protected area risk across sourcing regions
- Buffer zone density visualization showing agricultural pressure patterns
- Time-lapse view of encroachment trends over quarters

#### Feature 11: Ecosystem Connectivity Assessment
- Corridor analysis between protected areas to assess landscape connectivity
- Fragmentation impact modeling from agricultural expansion
- Wildlife movement pathway analysis for protected areas near supply chains
- Connectivity-weighted risk scoring for isolated protected areas

#### Feature 12: Predictive Protected Area Expansion Analysis
- Identify areas likely to be designated as protected under 30x30 target
- Model impact of upcoming protected area expansions on existing supply chains
- Early warning system for operators in potential future protection zones
- Cost-benefit analysis for proactive supply chain adjustment

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Marine protected area monitoring (terrestrial EUDR focus for v1.0)
- Protected area management plan advisory (agent provides data; not conservation counsel)
- Payment for Ecosystem Services management (agent tracks protected areas; does not handle PES)
- Mobile native application for field inspection (web responsive only)
- Predictive ML models for protected area violation forecasting (defer to Phase 2)
- Real-time satellite monitoring of protected area boundaries (defer to EUDR-003/020 integration)
- Blockchain-based protected area compliance ledger (SHA-256 provenance hashes sufficient for v1.0)
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
| AGENT-EUDR-022        |           | AGENT-EUDR-001            |           | AGENT-EUDR-016        |
| Protected Area        |<--------->| Supply Chain Mapping      |<--------->| Country Risk          |
| Validator             |           | Master                    |           | Evaluator             |
|                       |           |                           |           |                       |
| - PA Database (F1)    |           | - Graph Engine            |           | - Risk Scoring Engine |
| - Overlap Detector(F2)|           | - Risk Propagation        |           | - Governance Engine   |
| - Buffer Monitor (F3) |           | - Gap Analysis            |           | - Env. Enforcement    |
| - Designation Val.(F4)|           |                           |           |                       |
| - Proximity Alert(F5) |           +---------------------------+           +-----------------------+
| - Compliance Track(F6)|
| - Conservation (F7)   |           +---------------------------+           +---------------------------+
| - Reporting (F8)      |           | AGENT-EUDR-006            |           | AGENT-EUDR-020            |
| - Integration (F9)    |           | Plot Boundary Manager     |           | Deforestation Alert       |
+-----------+-----------+           |                           |           | System                    |
            |                       | - Plot Polygons           |           | - Satellite Detection     |
            |                       | - Boundary Validation     |           | - Buffer Monitoring       |
+-----------v-----------+           +---------------------------+           +---------------------------+
| Protected Area        |
| Data Sources          |           +---------------------------+           +---------------------------+
|                       |           | AGENT-EUDR-002            |           | AGENT-EUDR-021            |
| - WDPA/Protected      |           | Geolocation Verification  |           | Indigenous Rights         |
|   Planet              |           |                           |           | Checker                   |
| - KBA Database        |           | - Coordinate Validation   |           | - Territory Overlap       |
| - ICMBio/KLHK/ICCN   |           +---------------------------+           | - Compound Risk           |
| - UNESCO/Ramsar       |                                                   +---------------------------+
| - IUCN Red List       |
+-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/protected_area_validator/
    __init__.py                              # Public API exports
    config.py                                # ProtectedAreaValidatorConfig with GL_EUDR_PAV_ env prefix
    models.py                                # Pydantic v2 models for protected area data
    protected_area_database.py               # ProtectedAreaDatabaseEngine: PA data management (F1)
    overlap_detector.py                      # SpatialOverlapDetector: PostGIS overlap analysis (F2)
    buffer_monitor.py                        # BufferZoneMonitor: buffer zone analysis and tracking (F3)
    designation_validator.py                 # DesignationValidator: legal status and effectiveness (F4)
    proximity_alert_engine.py                # ProximityAlertEngine: high-risk proximity alerts (F5)
    compliance_tracker.py                    # ComplianceTracker: violation and remediation tracking (F6)
    conservation_assessor.py                 # ConservationAssessor: biodiversity significance (F7)
    compliance_reporter.py                   # ComplianceReporter: report generation (F8)
    agent_integrator.py                      # AgentIntegrator: cross-agent integration (F9)
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains
    metrics.py                               # 20 Prometheus self-monitoring metrics
    setup.py                                 # ProtectedAreaValidatorService facade (singleton, 9 engines)
    reference_data/
        __init__.py
        wdpa_sources.py                      # WDPA data source specifications and ingestion configs
        iucn_categories.py                   # IUCN category definitions, risk scores, buffer defaults
        designation_types.py                 # Protected area designation type classifications
        national_registries.py               # Per-country national PA registry configurations
        conservation_data.py                 # Biodiversity significance reference data
    api/
        __init__.py
        router.py                            # FastAPI router (~30 endpoints)
        schemas.py                           # API request/response Pydantic schemas
        dependencies.py                      # FastAPI dependency injection
        protected_area_routes.py             # Protected area database endpoints
        overlap_routes.py                    # Overlap detection endpoints
        buffer_routes.py                     # Buffer zone monitoring endpoints
        designation_routes.py                # Designation validation endpoints
        alert_routes.py                      # Proximity alert endpoints
        compliance_routes.py                 # Compliance tracking endpoints
        conservation_routes.py               # Conservation assessment endpoints
        report_routes.py                     # Compliance reporting endpoints
        integration_routes.py                # Agent integration endpoints
```

### 7.3 Data Models (Key Entities)

```python
# IUCN Management Category
class IUCNCategory(str, Enum):
    IA = "Ia"                # Strict Nature Reserve
    IB = "Ib"                # Wilderness Area
    II = "II"                # National Park
    III = "III"              # Natural Monument
    IV = "IV"                # Habitat/Species Management Area
    V = "V"                  # Protected Landscape/Seascape
    VI = "VI"                # Protected Area with Sustainable Use
    NOT_REPORTED = "NR"      # Not Reported

# Protected Area Overlap Type
class PAOverlapType(str, Enum):
    INSIDE = "inside"        # Plot entirely within protected area
    PARTIAL = "partial"      # Polygon intersection with partial overlap
    BOUNDARY = "boundary"    # Plot boundary touches PA boundary
    BUFFER = "buffer"        # Plot within buffer zone only
    CLEAR = "clear"          # Plot outside all buffer zones

# Protected Area Designation Level
class DesignationLevel(str, Enum):
    INTERNATIONAL = "international"  # UNESCO WH, Ramsar, MAB
    NATIONAL = "national"            # National park, national reserve
    REGIONAL = "regional"            # State/provincial protected area
    LOCAL = "local"                  # Municipal, community, private reserve
    PROPOSED = "proposed"            # Under consideration

# Protected Area Legal Status
class PALegalStatus(str, Enum):
    DESIGNATED = "designated"
    PROPOSED = "proposed"
    INSCRIBED = "inscribed"
    ADOPTED = "adopted"
    ESTABLISHED = "established"

# Governance Type
class GovernanceType(str, Enum):
    FEDERAL = "federal"
    SUBNATIONAL = "subnational"
    COLLABORATIVE = "collaborative"
    PRIVATE = "private"
    INDIGENOUS = "indigenous"
    NOT_REPORTED = "not_reported"

# Buffer Proximity Tier
class BufferProximityTier(str, Enum):
    IMMEDIATE = "immediate"      # 0-1km
    CLOSE = "close"              # 1-5km
    MODERATE = "moderate"        # 5-10km
    DISTANT = "distant"          # 10-25km
    PERIPHERAL = "peripheral"    # 25-50km

# Compliance Status
class PAComplianceStatus(str, Enum):
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    VIOLATION_CONFIRMED = "violation_confirmed"
    REMEDIATION_PLANNED = "remediation_planned"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    REMEDIATED = "remediated"
    EXEMPTION_GRANTED = "exemption_granted"
    FALSE_POSITIVE = "false_positive"

# Alert Severity
class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    SEVERE = "severe"
    HIGH = "high"
    ELEVATED = "elevated"
    STANDARD = "standard"

# Protected Area Record
class ProtectedArea(BaseModel):
    wdpa_id: int                             # WDPA unique identifier
    name: str                                # Protected area name
    orig_name: Optional[str]                 # Name in local language
    designation: str                         # Designation type name
    designation_level: DesignationLevel      # International/National/Regional/Local
    iucn_category: IUCNCategory              # IUCN management category
    country_code: str                        # ISO 3166-1 alpha-3
    iso3: str                                # ISO3 code
    area_hectares: Decimal                   # Total reported area
    marine_area_hectares: Decimal            # Marine component area
    legal_status: PALegalStatus              # Legal designation status
    status_year: int                         # Year of current status
    governance_type: GovernanceType          # Governance classification
    management_authority: Optional[str]      # Managing body name
    management_plan: Optional[str]           # Management plan status
    mett_score: Optional[Decimal]            # METT effectiveness score (0-100)
    boundary_geom: Optional[Dict]            # PostGIS geometry reference
    boundary_geojson: Optional[Dict]         # GeoJSON polygon/multipolygon
    is_world_heritage: bool                  # UNESCO WH designation
    is_ramsar: bool                          # Ramsar Wetland designation
    is_biosphere: bool                       # MAB Biosphere Reserve
    is_kba: bool                             # Key Biodiversity Area
    is_aze: bool                             # Alliance for Zero Extinction site
    wdpa_version: str                        # WDPA release version (e.g., Oct2025)
    data_source: str                         # WDPA/National/Regional
    confidence: str                          # Boundary accuracy (high/medium/low)
    provenance_hash: str                     # SHA-256
    last_verified: datetime
    created_at: datetime

# Overlap Analysis Result
class ProtectedAreaOverlap(BaseModel):
    overlap_id: str                          # UUID
    plot_id: str                             # Production plot assessed
    wdpa_id: int                             # WDPA protected area ID
    protected_area_name: str
    iucn_category: IUCNCategory
    designation_level: DesignationLevel
    overlap_type: PAOverlapType              # INSIDE/PARTIAL/BOUNDARY/BUFFER/CLEAR
    overlap_area_hectares: Optional[Decimal] # For INSIDE/PARTIAL
    overlap_pct_of_plot: Optional[Decimal]   # % of plot area
    overlap_pct_of_pa: Optional[Decimal]     # % of PA area
    distance_meters: Decimal                 # Min distance to PA boundary
    bearing_degrees: Optional[Decimal]       # Direction to nearest PA boundary
    risk_score: Decimal                      # 0-100 composite risk
    risk_level: str                          # CRITICAL/HIGH/MEDIUM/LOW/CLEAR
    designation_strength: Decimal            # 0-100
    deforestation_correlation: bool          # Cross-ref with EUDR-020
    indigenous_rights_correlation: bool      # Cross-ref with EUDR-021
    provenance_hash: str                     # SHA-256
    detected_at: datetime

# Buffer Zone Analysis Result
class BufferZoneResult(BaseModel):
    buffer_id: str                           # UUID
    plot_id: str
    wdpa_id: int
    proximity_tier: BufferProximityTier      # IMMEDIATE/CLOSE/MODERATE/DISTANT/PERIPHERAL
    distance_meters: Decimal                 # Exact distance
    buffer_radius_km: Decimal                # Configured buffer radius
    iucn_category: IUCNCategory
    encroachment_trend: str                  # approaching/stable/retreating
    national_buffer_required: bool           # National law mandates buffer
    national_buffer_km: Optional[Decimal]    # Required buffer distance
    compliant_with_national: bool            # Plot complies with national buffer
    provenance_hash: str
    detected_at: datetime

# High-Risk Proximity Alert
class ProximityAlert(BaseModel):
    alert_id: str                            # UUID
    plot_id: str
    wdpa_id: int
    protected_area_name: str
    iucn_category: IUCNCategory
    high_risk_designations: List[str]        # UNESCO_WH, Ramsar, AZE, KBA
    distance_meters: Decimal
    proximity_risk_score: Decimal            # 0-100
    alert_severity: AlertSeverity            # CRITICAL/SEVERE/HIGH/ELEVATED/STANDARD
    supply_chain_impact: Optional[Dict]
    enhanced_dd_required: bool
    enhanced_dd_deadline: Optional[datetime]
    compound_risk_indigenous: bool           # Also near indigenous territory
    deforestation_trend: Optional[str]
    provenance_hash: str
    created_at: datetime

# Compliance Record
class PAComplianceRecord(BaseModel):
    compliance_id: str                       # UUID
    plot_id: str
    wdpa_id: int
    overlap_id: str
    compliance_status: PAComplianceStatus
    violation_type: Optional[str]            # encroachment/clearing/construction/extraction
    affected_area_hectares: Optional[Decimal]
    detection_method: str
    remediation_plan: Optional[str]
    remediation_deadline: Optional[datetime]
    exemption_permit: Optional[Dict]
    compliance_score: Optional[Decimal]      # 0-100
    provenance_hash: str
    created_at: datetime
    updated_at: datetime
```

### 7.4 Database Schema (New Migration: V110)

```sql
-- =========================================================================
-- V110: AGENT-EUDR-022 Protected Area Validator Schema
-- Agent: GL-EUDR-PAV-022
-- Tables: 15 (4 hypertables)
-- Prefix: gl_eudr_pav_
-- =========================================================================

CREATE SCHEMA IF NOT EXISTS eudr_protected_area_validator;

-- 1. Protected Areas (spatial data with PostGIS)
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_protected_areas (
    wdpa_id INTEGER PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    orig_name VARCHAR(500),
    designation VARCHAR(500) NOT NULL,
    designation_level VARCHAR(20) NOT NULL CHECK (designation_level IN ('international', 'national', 'regional', 'local', 'proposed')),
    iucn_category VARCHAR(5) NOT NULL CHECK (iucn_category IN ('Ia', 'Ib', 'II', 'III', 'IV', 'V', 'VI', 'NR')),
    country_code CHAR(3) NOT NULL,
    iso3 CHAR(3) NOT NULL,
    area_hectares NUMERIC(16,2),
    marine_area_hectares NUMERIC(16,2) DEFAULT 0,
    legal_status VARCHAR(20) NOT NULL CHECK (legal_status IN ('designated', 'proposed', 'inscribed', 'adopted', 'established')),
    status_year INTEGER,
    governance_type VARCHAR(20) NOT NULL DEFAULT 'not_reported' CHECK (governance_type IN ('federal', 'subnational', 'collaborative', 'private', 'indigenous', 'not_reported')),
    management_authority VARCHAR(500),
    management_plan VARCHAR(50),
    mett_score NUMERIC(5,2),
    boundary_geom GEOMETRY(MultiPolygon, 4326),
    boundary_geojson JSONB,
    is_world_heritage BOOLEAN DEFAULT FALSE,
    is_ramsar BOOLEAN DEFAULT FALSE,
    is_biosphere BOOLEAN DEFAULT FALSE,
    is_kba BOOLEAN DEFAULT FALSE,
    is_aze BOOLEAN DEFAULT FALSE,
    wdpa_version VARCHAR(20) NOT NULL,
    data_source VARCHAR(200) NOT NULL DEFAULT 'WDPA',
    gis_quality VARCHAR(10) DEFAULT 'medium' CHECK (gis_quality IN ('high', 'medium', 'low')),
    confidence VARCHAR(10) NOT NULL DEFAULT 'medium' CHECK (confidence IN ('high', 'medium', 'low')),
    provenance_hash VARCHAR(64) NOT NULL,
    last_verified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pav_pa_country ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(country_code);
CREATE INDEX idx_pav_pa_iso3 ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(iso3);
CREATE INDEX idx_pav_pa_iucn ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(iucn_category);
CREATE INDEX idx_pav_pa_designation ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(designation_level);
CREATE INDEX idx_pav_pa_status ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(legal_status);
CREATE INDEX idx_pav_pa_wh ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(is_world_heritage) WHERE is_world_heritage = TRUE;
CREATE INDEX idx_pav_pa_ramsar ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(is_ramsar) WHERE is_ramsar = TRUE;
CREATE INDEX idx_pav_pa_kba ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(is_kba) WHERE is_kba = TRUE;
CREATE INDEX idx_pav_pa_aze ON eudr_protected_area_validator.gl_eudr_pav_protected_areas(is_aze) WHERE is_aze = TRUE;
CREATE INDEX idx_pav_pa_geom ON eudr_protected_area_validator.gl_eudr_pav_protected_areas USING GIST (boundary_geom);

-- 2. Overlap Analysis Results
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_overlaps (
    overlap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    wdpa_id INTEGER NOT NULL REFERENCES eudr_protected_area_validator.gl_eudr_pav_protected_areas(wdpa_id),
    iucn_category VARCHAR(5) NOT NULL,
    designation_level VARCHAR(20) NOT NULL,
    overlap_type VARCHAR(20) NOT NULL CHECK (overlap_type IN ('inside', 'partial', 'boundary', 'buffer', 'clear')),
    overlap_area_hectares NUMERIC(14,2),
    overlap_pct_of_plot NUMERIC(7,4),
    overlap_pct_of_pa NUMERIC(7,4),
    distance_meters NUMERIC(12,2) NOT NULL,
    bearing_degrees NUMERIC(5,2),
    risk_score NUMERIC(5,2) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('critical', 'high', 'medium', 'low', 'clear')),
    designation_strength NUMERIC(5,2),
    deforestation_correlation BOOLEAN DEFAULT FALSE,
    indigenous_rights_correlation BOOLEAN DEFAULT FALSE,
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (plot_id, wdpa_id)
);

CREATE INDEX idx_pav_overlaps_plot ON eudr_protected_area_validator.gl_eudr_pav_overlaps(plot_id);
CREATE INDEX idx_pav_overlaps_wdpa ON eudr_protected_area_validator.gl_eudr_pav_overlaps(wdpa_id);
CREATE INDEX idx_pav_overlaps_type ON eudr_protected_area_validator.gl_eudr_pav_overlaps(overlap_type);
CREATE INDEX idx_pav_overlaps_risk ON eudr_protected_area_validator.gl_eudr_pav_overlaps(risk_level);
CREATE INDEX idx_pav_overlaps_iucn ON eudr_protected_area_validator.gl_eudr_pav_overlaps(iucn_category);
CREATE INDEX idx_pav_overlaps_deforestation ON eudr_protected_area_validator.gl_eudr_pav_overlaps(deforestation_correlation) WHERE deforestation_correlation = TRUE;

-- 3. Buffer Zone Analysis Results
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_buffer_zones (
    buffer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    wdpa_id INTEGER NOT NULL REFERENCES eudr_protected_area_validator.gl_eudr_pav_protected_areas(wdpa_id),
    proximity_tier VARCHAR(20) NOT NULL CHECK (proximity_tier IN ('immediate', 'close', 'moderate', 'distant', 'peripheral')),
    distance_meters NUMERIC(12,2) NOT NULL,
    buffer_radius_km NUMERIC(6,2) NOT NULL,
    iucn_category VARCHAR(5) NOT NULL,
    encroachment_trend VARCHAR(20) DEFAULT 'stable' CHECK (encroachment_trend IN ('approaching', 'stable', 'retreating')),
    national_buffer_required BOOLEAN DEFAULT FALSE,
    national_buffer_km NUMERIC(6,2),
    compliant_with_national BOOLEAN DEFAULT TRUE,
    exemption_id UUID,
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (plot_id, wdpa_id, buffer_radius_km)
);

CREATE INDEX idx_pav_buffers_plot ON eudr_protected_area_validator.gl_eudr_pav_buffer_zones(plot_id);
CREATE INDEX idx_pav_buffers_wdpa ON eudr_protected_area_validator.gl_eudr_pav_buffer_zones(wdpa_id);
CREATE INDEX idx_pav_buffers_tier ON eudr_protected_area_validator.gl_eudr_pav_buffer_zones(proximity_tier);
CREATE INDEX idx_pav_buffers_national ON eudr_protected_area_validator.gl_eudr_pav_buffer_zones(national_buffer_required) WHERE national_buffer_required = TRUE;

-- 4. High-Risk Proximity Alerts (hypertable on created_at)
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_proximity_alerts (
    alert_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    wdpa_id INTEGER NOT NULL,
    protected_area_name VARCHAR(500),
    iucn_category VARCHAR(5) NOT NULL,
    high_risk_designations JSONB DEFAULT '[]',
    distance_meters NUMERIC(12,2) NOT NULL,
    proximity_risk_score NUMERIC(5,2) NOT NULL CHECK (proximity_risk_score >= 0 AND proximity_risk_score <= 100),
    alert_severity VARCHAR(20) NOT NULL CHECK (alert_severity IN ('critical', 'severe', 'high', 'elevated', 'standard')),
    supply_chain_impact JSONB DEFAULT '{}',
    enhanced_dd_required BOOLEAN DEFAULT FALSE,
    enhanced_dd_deadline TIMESTAMPTZ,
    enhanced_dd_acknowledged BOOLEAN DEFAULT FALSE,
    compound_risk_indigenous BOOLEAN DEFAULT FALSE,
    deforestation_trend VARCHAR(20),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'mitigated', 'resolved', 'suppressed')),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (alert_id, created_at)
);

SELECT create_hypertable('eudr_protected_area_validator.gl_eudr_pav_proximity_alerts', 'created_at');

CREATE INDEX idx_pav_alerts_plot ON eudr_protected_area_validator.gl_eudr_pav_proximity_alerts(plot_id);
CREATE INDEX idx_pav_alerts_severity ON eudr_protected_area_validator.gl_eudr_pav_proximity_alerts(alert_severity);
CREATE INDEX idx_pav_alerts_status ON eudr_protected_area_validator.gl_eudr_pav_proximity_alerts(status);
CREATE INDEX idx_pav_alerts_enhanced ON eudr_protected_area_validator.gl_eudr_pav_proximity_alerts(enhanced_dd_required) WHERE enhanced_dd_required = TRUE;

-- 5. Compliance Records
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_compliance_records (
    compliance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    wdpa_id INTEGER NOT NULL,
    overlap_id UUID REFERENCES eudr_protected_area_validator.gl_eudr_pav_overlaps(overlap_id),
    compliance_status VARCHAR(30) NOT NULL CHECK (compliance_status IN ('detected', 'investigating', 'violation_confirmed', 'remediation_planned', 'remediation_in_progress', 'remediated', 'exemption_granted', 'false_positive')),
    violation_type VARCHAR(50),
    affected_area_hectares NUMERIC(14,2),
    detection_method VARCHAR(50) NOT NULL,
    evidence_documents JSONB DEFAULT '[]',
    severity VARCHAR(20) CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    remediation_plan TEXT,
    remediation_deadline TIMESTAMPTZ,
    remediation_sla_status VARCHAR(20) DEFAULT 'on_track' CHECK (remediation_sla_status IN ('on_track', 'at_risk', 'overdue')),
    exemption_permit JSONB,
    compliance_score NUMERIC(5,2),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pav_compliance_plot ON eudr_protected_area_validator.gl_eudr_pav_compliance_records(plot_id);
CREATE INDEX idx_pav_compliance_wdpa ON eudr_protected_area_validator.gl_eudr_pav_compliance_records(wdpa_id);
CREATE INDEX idx_pav_compliance_status ON eudr_protected_area_validator.gl_eudr_pav_compliance_records(compliance_status);
CREATE INDEX idx_pav_compliance_sla ON eudr_protected_area_validator.gl_eudr_pav_compliance_records(remediation_sla_status);

-- 6. Compliance Status Transitions (hypertable)
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_compliance_transitions (
    transition_id UUID DEFAULT gen_random_uuid(),
    compliance_id UUID NOT NULL,
    from_status VARCHAR(30) NOT NULL,
    to_status VARCHAR(30) NOT NULL,
    actor VARCHAR(200) NOT NULL,
    reason TEXT,
    supporting_evidence JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    transitioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (transition_id, transitioned_at)
);

SELECT create_hypertable('eudr_protected_area_validator.gl_eudr_pav_compliance_transitions', 'transitioned_at');

-- 7. Conservation Assessments
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_conservation_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wdpa_id INTEGER NOT NULL REFERENCES eudr_protected_area_validator.gl_eudr_pav_protected_areas(wdpa_id),
    biodiversity_score NUMERIC(5,2) NOT NULL CHECK (biodiversity_score >= 0 AND biodiversity_score <= 100),
    threatened_species_count INTEGER DEFAULT 0,
    cites_species_count INTEGER DEFAULT 0,
    aze_trigger_present BOOLEAN DEFAULT FALSE,
    habitat_rarity_score NUMERIC(5,2),
    connectivity_score NUMERIC(5,2),
    fragmentation_index NUMERIC(5,2),
    carbon_storage_tco2e NUMERIC(14,2),
    biome VARCHAR(200),
    dominant_habitats JSONB DEFAULT '[]',
    threat_assessment JSONB DEFAULT '{}',
    species_data JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pav_conservation_wdpa ON eudr_protected_area_validator.gl_eudr_pav_conservation_assessments(wdpa_id);
CREATE INDEX idx_pav_conservation_score ON eudr_protected_area_validator.gl_eudr_pav_conservation_assessments(biodiversity_score);

-- 8. Designation Validations
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_designations (
    designation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wdpa_id INTEGER NOT NULL REFERENCES eudr_protected_area_validator.gl_eudr_pav_protected_areas(wdpa_id),
    designation_strength_score NUMERIC(5,2) NOT NULL CHECK (designation_strength_score >= 0 AND designation_strength_score <= 100),
    designation_level_score NUMERIC(5,2),
    legal_status_score NUMERIC(5,2),
    management_effectiveness_score NUMERIC(5,2),
    enforcement_score NUMERIC(5,2),
    governance_assessment JSONB DEFAULT '{}',
    designation_history JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pav_designations_wdpa ON eudr_protected_area_validator.gl_eudr_pav_designations(wdpa_id);

-- 9. Country Protected Area Scores
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_country_scores (
    score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(3) NOT NULL,
    total_protected_areas INTEGER DEFAULT 0,
    total_protected_area_hectares NUMERIC(16,2) DEFAULT 0,
    pct_land_protected NUMERIC(5,2) DEFAULT 0,
    iucn_category_distribution JSONB DEFAULT '{}',
    avg_management_effectiveness NUMERIC(5,2),
    enforcement_score NUMERIC(5,2) NOT NULL,
    governance_quality_score NUMERIC(5,2) NOT NULL,
    composite_pa_governance_score NUMERIC(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'standard', 'high')),
    data_sources JSONB NOT NULL DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code)
);

CREATE INDEX idx_pav_country_scores_level ON eudr_protected_area_validator.gl_eudr_pav_country_scores(risk_level);

-- 10. Buffer Zone Exemptions
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_exemptions (
    exemption_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    wdpa_id INTEGER NOT NULL,
    permit_type VARCHAR(100) NOT NULL,
    issuing_authority VARCHAR(500) NOT NULL,
    permit_number VARCHAR(200),
    effective_date DATE NOT NULL,
    expiry_date DATE,
    conditions TEXT,
    document_hash VARCHAR(64),
    document_path VARCHAR(1000),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'expired', 'revoked', 'pending_renewal')),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pav_exemptions_plot ON eudr_protected_area_validator.gl_eudr_pav_exemptions(plot_id);
CREATE INDEX idx_pav_exemptions_status ON eudr_protected_area_validator.gl_eudr_pav_exemptions(status);
CREATE INDEX idx_pav_exemptions_expiry ON eudr_protected_area_validator.gl_eudr_pav_exemptions(expiry_date);

-- 11. WDPA Version History
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_wdpa_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wdpa_version VARCHAR(20) NOT NULL UNIQUE,
    release_date DATE NOT NULL,
    total_records INTEGER NOT NULL,
    records_added INTEGER DEFAULT 0,
    records_modified INTEGER DEFAULT 0,
    records_removed INTEGER DEFAULT 0,
    ingestion_started_at TIMESTAMPTZ,
    ingestion_completed_at TIMESTAMPTZ,
    ingestion_status VARCHAR(20) DEFAULT 'pending' CHECK (ingestion_status IN ('pending', 'in_progress', 'completed', 'failed')),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 12. Protected Area Boundary History
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_boundary_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wdpa_id INTEGER NOT NULL REFERENCES eudr_protected_area_validator.gl_eudr_pav_protected_areas(wdpa_id),
    change_type VARCHAR(30) NOT NULL CHECK (change_type IN ('created', 'boundary_expanded', 'boundary_reduced', 'category_changed', 'status_changed', 'degazetted', 'redesignated')),
    previous_area_hectares NUMERIC(16,2),
    new_area_hectares NUMERIC(16,2),
    previous_iucn_category VARCHAR(5),
    new_iucn_category VARCHAR(5),
    previous_boundary_geojson JSONB,
    change_description TEXT NOT NULL,
    wdpa_version_from VARCHAR(20),
    wdpa_version_to VARCHAR(20),
    effective_date DATE,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pav_boundary_history_wdpa ON eudr_protected_area_validator.gl_eudr_pav_boundary_history(wdpa_id);
CREATE INDEX idx_pav_boundary_history_type ON eudr_protected_area_validator.gl_eudr_pav_boundary_history(change_type);

-- 13. Compliance Reports
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_compliance_reports (
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

CREATE INDEX idx_pav_reports_type ON eudr_protected_area_validator.gl_eudr_pav_compliance_reports(report_type);

-- 14. Encroachment Trend Data (hypertable)
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_encroachment_trends (
    trend_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    wdpa_id INTEGER NOT NULL,
    distance_meters NUMERIC(12,2) NOT NULL,
    previous_distance_meters NUMERIC(12,2),
    distance_change_meters NUMERIC(12,2),
    trend_direction VARCHAR(20) CHECK (trend_direction IN ('approaching', 'stable', 'retreating')),
    measurement_period VARCHAR(20),
    provenance_hash VARCHAR(64) NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (trend_id, measured_at)
);

SELECT create_hypertable('eudr_protected_area_validator.gl_eudr_pav_encroachment_trends', 'measured_at');

-- 15. Immutable Audit Log (hypertable)
CREATE TABLE eudr_protected_area_validator.gl_eudr_pav_audit_log (
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

SELECT create_hypertable('eudr_protected_area_validator.gl_eudr_pav_audit_log', 'created_at');

CREATE INDEX idx_pav_audit_entity ON eudr_protected_area_validator.gl_eudr_pav_audit_log(entity_type, entity_id);
CREATE INDEX idx_pav_audit_actor ON eudr_protected_area_validator.gl_eudr_pav_audit_log(actor);
CREATE INDEX idx_pav_audit_action ON eudr_protected_area_validator.gl_eudr_pav_audit_log(action);
```

### 7.5 API Endpoints (~30)

| Method | Path | Description |
|--------|------|-------------|
| **Protected Area Database** | | |
| GET | `/v1/eudr-pav/protected-areas` | List protected areas (with filters: country, IUCN category, designation, bbox) |
| GET | `/v1/eudr-pav/protected-areas/{wdpa_id}` | Get protected area details with boundary GeoJSON |
| POST | `/v1/eudr-pav/protected-areas/search` | Search protected areas by name, country, designation, or proximity |
| GET | `/v1/eudr-pav/protected-areas/{wdpa_id}/history` | Get protected area boundary change history |
| GET | `/v1/eudr-pav/protected-areas/coverage` | Get WDPA database coverage statistics |
| **Overlap Detection** | | |
| POST | `/v1/eudr-pav/overlaps/detect` | Detect protected area overlaps for a single plot |
| POST | `/v1/eudr-pav/overlaps/batch` | Batch overlap screening for multiple plots |
| GET | `/v1/eudr-pav/overlaps/{overlap_id}` | Get overlap details with map data |
| GET | `/v1/eudr-pav/overlaps` | List all detected overlaps (with filters) |
| GET | `/v1/eudr-pav/overlaps/statistics` | Get overlap statistics by IUCN category, country, commodity |
| **Buffer Zone Monitoring** | | |
| POST | `/v1/eudr-pav/buffers/analyze` | Analyze buffer zone for a plot against nearby protected areas |
| GET | `/v1/eudr-pav/buffers` | List buffer zone results (with filters: tier, compliance) |
| GET | `/v1/eudr-pav/buffers/trends` | Get encroachment trend data |
| PUT | `/v1/eudr-pav/buffers/config` | Update buffer zone configuration per operator |
| **Designation Validation** | | |
| GET | `/v1/eudr-pav/designations/{wdpa_id}` | Get designation validation for a protected area |
| GET | `/v1/eudr-pav/designations/upcoming` | Get upcoming designation changes and proposals |
| **Proximity Alerts** | | |
| GET | `/v1/eudr-pav/alerts` | List proximity alerts (with filters: severity, status, date) |
| GET | `/v1/eudr-pav/alerts/{alert_id}` | Get alert details with impact assessment |
| POST | `/v1/eudr-pav/alerts/{alert_id}/acknowledge` | Acknowledge enhanced DD requirement |
| GET | `/v1/eudr-pav/alerts/digest` | Get alert digest (daily/weekly/monthly summary) |
| **Compliance Tracking** | | |
| GET | `/v1/eudr-pav/compliance` | List compliance records (with filters: status, violation type) |
| GET | `/v1/eudr-pav/compliance/{compliance_id}` | Get compliance record details |
| POST | `/v1/eudr-pav/compliance/{compliance_id}/transition` | Transition compliance status |
| POST | `/v1/eudr-pav/exemptions` | Record a buffer zone exemption |
| GET | `/v1/eudr-pav/compliance/dashboard` | Get compliance SLA dashboard data |
| **Conservation Assessment** | | |
| GET | `/v1/eudr-pav/conservation/{wdpa_id}` | Get conservation assessment for a protected area |
| GET | `/v1/eudr-pav/conservation/species/{wdpa_id}` | Get species data for a protected area |
| **Reporting** | | |
| POST | `/v1/eudr-pav/reports/generate` | Generate protected area compliance report |
| GET | `/v1/eudr-pav/reports/{report_id}` | Get report metadata |
| GET | `/v1/eudr-pav/reports/{report_id}/download` | Download report file |
| GET | `/v1/eudr-pav/reports` | List generated reports |
| **Country Scores** | | |
| GET | `/v1/eudr-pav/countries/{country_code}` | Get country protected area governance score |
| GET | `/v1/eudr-pav/countries` | List all country protected area scores |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (20)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_pav_pa_queries_total` | Counter | Protected area database queries by type |
| 2 | `gl_eudr_pav_overlaps_detected_total` | Counter | Overlaps detected by IUCN category and type |
| 3 | `gl_eudr_pav_batch_screenings_total` | Counter | Batch overlap screening operations |
| 4 | `gl_eudr_pav_buffer_analyses_total` | Counter | Buffer zone analyses performed |
| 5 | `gl_eudr_pav_proximity_alerts_total` | Counter | Proximity alerts generated by severity |
| 6 | `gl_eudr_pav_compliance_transitions_total` | Counter | Compliance status transitions by type |
| 7 | `gl_eudr_pav_reports_generated_total` | Counter | Compliance reports generated by type |
| 8 | `gl_eudr_pav_exemptions_recorded_total` | Counter | Buffer zone exemptions recorded |
| 9 | `gl_eudr_pav_conservation_assessments_total` | Counter | Conservation assessments performed |
| 10 | `gl_eudr_pav_wdpa_ingestions_total` | Counter | WDPA data ingestion operations |
| 11 | `gl_eudr_pav_overlap_query_duration_seconds` | Histogram | Overlap detection query latency |
| 12 | `gl_eudr_pav_batch_screening_duration_seconds` | Histogram | Batch overlap screening latency |
| 13 | `gl_eudr_pav_buffer_analysis_duration_seconds` | Histogram | Buffer zone analysis latency |
| 14 | `gl_eudr_pav_report_generation_duration_seconds` | Histogram | Report generation time |
| 15 | `gl_eudr_pav_api_errors_total` | Counter | API errors by endpoint and status code |
| 16 | `gl_eudr_pav_active_protected_areas` | Gauge | Total protected areas in database |
| 17 | `gl_eudr_pav_active_overlaps` | Gauge | Currently active protected area overlaps |
| 18 | `gl_eudr_pav_active_alerts` | Gauge | Currently active proximity alerts |
| 19 | `gl_eudr_pav_compliance_sla_breaches_total` | Counter | Compliance SLA breaches by stage |
| 20 | `gl_eudr_pav_wdpa_data_age_days` | Gauge | Days since last WDPA data ingestion |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for alerts/trends/audit |
| Spatial | PostGIS + Shapely + GeoJSON | Protected area boundary storage, spatial indexing, overlap calculations, buffer generation |
| Cache | Redis | Protected area query caching, overlap result caching, buffer computation caching |
| Object Storage | S3 | Generated reports, WDPA snapshots, boundary history |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Arithmetic | Python Decimal | Precision for risk scores, overlap percentages, and distances |
| PDF Generation | WeasyPrint | HTML-to-PDF for compliance report generation |
| Templates | Jinja2 | Multi-format report templates (HTML/PDF) |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based protected area data access control |
| Monitoring | Prometheus + Grafana | 20 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-pav:protected-areas:read` | View protected area data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:overlaps:read` | View overlap results | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:overlaps:detect` | Trigger overlap detection analysis | Analyst, Compliance Officer, Admin |
| `eudr-pav:buffers:read` | View buffer zone results | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:buffers:config` | Configure buffer zone parameters | Compliance Officer, Admin |
| `eudr-pav:designations:read` | View designation validations | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:alerts:read` | View proximity alerts | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:alerts:manage` | Acknowledge and manage alerts | Compliance Officer, Admin |
| `eudr-pav:compliance:read` | View compliance records | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:compliance:manage` | Manage compliance lifecycle (transition, record violations) | Compliance Officer, Admin |
| `eudr-pav:exemptions:manage` | Record and manage buffer zone exemptions | Compliance Officer, Admin |
| `eudr-pav:conservation:read` | View conservation assessments | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:reports:read` | View generated reports | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:reports:generate` | Generate compliance reports | Analyst, Compliance Officer, Admin |
| `eudr-pav:reports:download` | Download report files | Analyst, Compliance Officer, Admin |
| `eudr-pav:countries:read` | View country protected area scores | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-pav:audit:read` | View audit trail | Auditor (read-only), Compliance Officer, Admin |
| `eudr-pav:config:manage` | Manage configuration (buffer defaults, scoring weights, alert thresholds) | Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent/Source | Integration | Data Flow |
|-------------|-------------|-----------|
| WDPA/Protected Planet | Monthly data download | Protected area polygons for 245 countries -> PA database (F1) |
| KBA Database (BirdLife/IUCN) | Data download/API | Key Biodiversity Area boundaries and trigger species -> PA database (F1) |
| ICMBio/CNUC (Brazil) | Government GIS portal | Brazil Conservation Units boundaries -> PA database (F1) |
| KLHK (Indonesia) | Government GIS portal | Indonesia Kawasan Konservasi boundaries -> PA database (F1) |
| IUCN Red List API | REST API | Threatened species data per protected area -> conservation assessment (F7) |
| GD-PAME | Data download | Management effectiveness (METT) scores -> designation validation (F4) |
| AGENT-EUDR-002 Geolocation Verification | Plot coordinates | Validated plot coordinates -> overlap detection (F2) |
| AGENT-EUDR-006 Plot Boundary Manager | Plot polygons | Plot boundary polygons -> overlap detection (F2) |
| AGENT-EUDR-016 Country Risk Evaluator | Country governance data | Environmental enforcement scores -> enforcement assessment (F4) |
| AGENT-EUDR-020 Deforestation Alert System | Deforestation alerts | Deforestation events near PAs -> cross-reference (F2, F5, F9) |
| AGENT-EUDR-021 Indigenous Rights Checker | Territory overlaps | Indigenous territory proximity -> compound risk (F5, F9) |
| AGENT-DATA-006 GIS/Mapping Connector | Spatial operations | PostGIS utilities -> overlap analysis (F2) |
| AGENT-FOUND-005 Citations & Evidence | Source attribution | Citation generation for all data sources -> provenance tracking |
| AGENT-FOUND-008 Reproducibility | Determinism verification | Bit-perfect verification of scoring and overlap calculations |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-001 Supply Chain Mapping Master | Graph enrichment | Protected area risk attributes -> supply chain graph node properties |
| AGENT-EUDR-016 Country Risk Evaluator | Governance input | Country PA governance scores -> environmental governance index |
| AGENT-EUDR-017 Supplier Risk Scorer | Supplier risk input | Per-supplier protected area risk -> supplier composite risk score |
| GL-EUDR-APP v1.0 Platform | API integration | PA map overlay, overlap dashboard, alert feed, compliance tracker, report downloads |
| GL-EUDR-APP DDS Reporting Engine | DDS section | Protected area risk assessment for Article 4(2) DDS submission |
| External Auditors | Read-only API + reports | Protected area compliance exports for third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Protected Area Overlap Screening (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Protected Areas" module -> "Overlap Screening" tab
3. Selects commodity (e.g., Palm Oil) and clicks "Screen All Plots"
4. System performs batch overlap detection for all palm oil supply chain plots
   -> Progress bar: "Screening 4,500 plots against 270,000+ protected areas..."
5. Results dashboard displays:
   - Total plots screened: 4,500
   - Inside protected area: 3 (CRITICAL -- IUCN Ia/II)
   - Partial overlap: 8 (HIGH -- IUCN III/IV)
   - Boundary touching: 15 (MEDIUM -- mixed categories)
   - Buffer zone (< 5km): 42 (ELEVATED)
   - Buffer zone (5-25km): 187 (STANDARD)
   - Clear (> 25km): 4,245 (CLEAR)
6. Officer clicks on a CRITICAL overlap -> sees plot on map with PA overlay
   - Plot: Plantation PLT-ID-0891 (West Kalimantan, Indonesia)
   - Protected area: Gunung Palung National Park (WDPA ID 6823)
   - IUCN Category: II (National Park) -- risk score 95
   - Overlap: 28 ha of plot inside park boundary (12% of plot area)
   - Additional flags: KBA site, UNESCO Tentative List
   - Deforestation correlation: YES (EUDR-020 alert within PA boundary)
7. System displays: "CRITICAL: Plot inside IUCN Category II National Park.
   Commodity production inside this protected area is prohibited.
   Recommended action: Immediate supplier engagement for plot relocation."
8. Officer clicks "Initiate Compliance Investigation" -> Compliance record created
9. System generates protected area screening report for DDS
```

#### Flow 2: Buffer Zone Monitoring (Protected Area Officer)

```
1. Protected area officer opens "Buffer Zone Monitoring" dashboard
2. Map view shows all protected areas in sourcing regions with buffer zones:
   - Red zones: 0-1km (IMMEDIATE) -- 5 plots
   - Orange zones: 1-5km (CLOSE) -- 18 plots
   - Yellow zones: 5-10km (MODERATE) -- 34 plots
   - Blue zones: 10-25km (DISTANT) -- 89 plots
3. Officer selects Tesso Nilo National Park (Riau, Indonesia)
   - IUCN Category II, WDPA ID 341458
   - National buffer zone regulation: 10km (PP 28/2011)
   - Plots within national buffer: 12 (3 non-compliant with national regulation)
4. Officer reviews non-compliant Plot PLT-ID-0342:
   - Distance: 4.2km from park boundary (inside 10km national buffer)
   - Trend: APPROACHING (was 5.1km in Q3 2025, now 4.2km -- plot expanded)
   - National compliance: NON-COMPLIANT (no buffer exemption on file)
5. Officer clicks "Request Exemption Documentation" from supplier
6. System sends notification to supplier requesting environmental permit upload
7. Officer configures alert rule: "Notify me when any plot moves within 3km
   of this protected area"
8. System saves custom alert configuration
```

#### Flow 3: High-Risk Alert Investigation (Compliance Officer)

```
1. System detects new WDPA boundary update: Virunga National Park (DRC)
   boundary expanded by 2,300 hectares in latest WDPA release
2. System re-screens all plots in DRC supply chain against new boundary
3. System generates SEVERE alert:
   - Plot PLT-DRC-0127 (North Kivu) now within 3.8km of expanded boundary
   - Previously 6.1km (was in MODERATE tier, now CLOSE tier)
   - Protected area: Virunga National Park (IUCN II, UNESCO World Heritage)
   - Additional flags: AZE site (Mountain Gorilla critical habitat)
   - Compound risk: Also within 8km of Batwa indigenous territory (EUDR-021)
4. Compliance officer receives email alert with severity SEVERE
5. Opens alert detail in GL-EUDR-APP:
   - Impact assessment: 2 suppliers, 5 plots, ~8% of DRC coffee volume
   - Enhanced due diligence required within 14 days (SEVERE SLA)
   - Recommended actions: (1) Verify supplier permits, (2) Request field
     inspection report, (3) Assess alternative sourcing options
6. Officer acknowledges alert and initiates enhanced DD workflow
7. System tracks DD response against 14-day SLA
8. Officer uploads supplier response and field inspection report
9. System generates compliance documentation for DDS
```

### 8.2 Key Screen Descriptions

**Protected Area Overlap Dashboard:**
- Interactive map: production plots (circles) overlaid with protected areas (shaded polygons, color-coded by IUCN category: dark red = Ia/Ib, red = II, orange = III/IV, yellow = V, green = VI)
- Buffer zone rings visible on map (concentric zones around PAs)
- Left sidebar: filter panel (commodity, country, IUCN category, overlap type, risk level)
- Right sidebar: selected plot/PA detail panel with overlap metrics
- Top bar: summary statistics (plots screened, overlap counts by IUCN category, compliance score)
- Bottom panel: WDPA data version and freshness indicator

**Buffer Zone Monitoring View:**
- Map-centric view with protected areas and configurable buffer zone visualization
- Heat map overlay showing plot density within buffer tiers
- Encroachment trend arrows on plots (approaching/stable/retreating)
- National buffer regulation indicator per country
- Filter bar: by IUCN category, buffer tier, compliance status, encroachment trend
- Timeline slider: view buffer zone status at historical dates

**High-Risk Alert Feed:**
- Timeline view: proximity alerts sorted by date with severity badges (color-coded)
- Map view: alert locations with protected area and supply chain overlay
- Compound risk indicator: protected area + indigenous rights overlap (double-flag icon)
- Detail panel: WDPA reference, IUCN category, designation details, conservation significance, impact assessment, enhanced DD SLA countdown
- Action buttons: Acknowledge, Initiate DD, Record Mitigation, Suppress (with reason)

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Protected Area Database -- 270,000+ PAs, WDPA integration, PostGIS spatial indexing, IUCN categories
  - [ ] Feature 2: Spatial Overlap Detection -- 5-type classification, IUCN-weighted scoring, batch screening, PostGIS operations
  - [ ] Feature 3: Buffer Zone Monitoring -- configurable radii 1-50km, encroachment trends, national regulation support
  - [ ] Feature 4: Designation Validator -- 4-tier designation levels, METT scores, enforcement assessment, WH/Ramsar/AZE flags
  - [ ] Feature 5: High-Risk Proximity Alert System -- severity classification, enhanced DD workflow, compound risk cross-referencing
  - [ ] Feature 6: Compliance Tracker -- 8-status lifecycle, remediation SLA, exemption management, audit trail
  - [ ] Feature 7: Conservation Assessment -- IUCN Red List, biodiversity scoring, habitat fragmentation, species-commodity matrix
  - [ ] Feature 8: Compliance Reporting -- 8 report types, 5 formats, 5 languages, DDS integration, certification alignment
  - [ ] Feature 9: Agent Integration -- EUDR-001/002/006/016/020/021 bidirectional integration verified
- [ ] >= 85% test coverage achieved (700+ tests)
- [ ] Security audit passed (JWT + RBAC integrated, 18 permissions)
- [ ] Performance targets met (< 500ms overlap query p99, < 5 min batch 10K plots, < 10s report generation)
- [ ] WDPA database validated against Protected Planet reference (>= 99% record match)
- [ ] IUCN-category risk scoring verified deterministic (bit-perfect reproducibility)
- [ ] Overlap detection validated against 100+ known plot-PA pairs (manual GIS cross-validation)
- [ ] API documentation complete (OpenAPI spec, ~30 endpoints)
- [ ] Database migration V110 tested and validated (15 tables, 4 hypertables)
- [ ] Integration with EUDR-001, EUDR-002, EUDR-006, EUDR-016, EUDR-020, EUDR-021 verified
- [ ] 5 beta customers successfully screening their supply chains
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ supply chains screened for protected area overlaps
- 5,000+ plots overlap-screened against full WDPA database
- Average overlap query latency < 500ms (p99)
- WDPA database freshness within 30 days of latest release
- High-risk alerts generated for all known WH/Ramsar/AZE proximity events
- < 5 support tickets per customer

**60 Days:**
- 200+ supply chains actively monitored for protected area compliance
- 25,000+ plots screened
- Buffer zone monitoring active for top 100 protected areas in sourcing regions
- 50+ compliance records with full lifecycle tracking
- Compliance reports generated for 3+ certification schemes (FSC, RSPO, RA)
- NPS > 45 from compliance officer persona

**90 Days:**
- 500+ supply chains actively monitored
- 100,000+ plots screened (full customer portfolio)
- Zero EUDR penalties attributable to protected area non-compliance for active customers
- Full integration with GL-EUDR-APP DDS workflow operational
- Protected area governance scores feeding EUDR-016 country risk for all producing countries
- NPS > 55

---

## 10. Timeline and Milestones

### Phase 1: Core Verification Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Protected Area Database (Feature 1): WDPA ingestion, IUCN classification, PostGIS spatial indexing, KBA/WH/Ramsar integration | Senior Backend Engineer + GIS Specialist |
| 2-3 | Spatial Overlap Detection (Feature 2): PostGIS intersection, 5-type classification, IUCN-weighted scoring, batch screening | Backend Engineer + GIS Specialist |
| 3-4 | Buffer Zone Monitoring (Feature 3): configurable radii, encroachment detection, national regulation rules, trend tracking | Backend Engineer + GIS Specialist |
| 4-5 | Designation Validator (Feature 4): 4-tier designation, METT integration, enforcement assessment, WH/Ramsar/AZE flags | Senior Backend Engineer |
| 5-6 | High-Risk Proximity Alert System (Feature 5): severity classification, enhanced DD workflow, compound risk cross-referencing | Backend Engineer |

**Milestone: Core verification engine operational with 5 core features (Week 6)**

### Phase 2: Compliance, API, and Reporting (Weeks 7-11)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Protected Area Compliance Tracker (Feature 6): 8-status lifecycle, remediation SLA, exemption management | Backend Engineer |
| 8-9 | Conservation Assessment (Feature 7): IUCN Red List integration, biodiversity scoring, habitat fragmentation | Backend Engineer + Data Engineer |
| 9-10 | REST API Layer: ~30 endpoints, authentication, rate limiting | Backend Engineer |
| 10-11 | Compliance Reporting (Feature 8): 8 report types, 5 formats, 5 languages, DDS section, certification alignment | Backend + Template Engineer |

**Milestone: Full API operational with compliance tracking, conservation assessment, and reporting (Week 11)**

### Phase 3: Integration, RBAC, and Observability (Weeks 12-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 12 | Agent Integration (Feature 9): EUDR-001/002/006/016/020/021 bidirectional integration | Senior Backend Engineer |
| 12-13 | RBAC integration (18 permissions), Prometheus metrics (20), Grafana dashboard | Backend + DevOps |
| 13-14 | OpenTelemetry tracing, event bus integration, end-to-end integration testing | DevOps + Backend |

**Milestone: All 9 P0 features implemented with full integration and observability (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 700+ tests, golden tests for all 7 commodities, spatial validation against known PA boundaries | Test Engineer |
| 16-17 | Performance testing (batch 10K plots), security audit, load testing (concurrent queries) | DevOps + Security |
| 17 | Database migration V110 finalized and tested (15 tables, 4 hypertables) | DevOps |
| 17-18 | Beta customer onboarding (5 customers) | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Protected area impact dashboard (Feature 10)
- Ecosystem connectivity assessment (Feature 11)
- Predictive protected area expansion analysis (Feature 12)
- Additional national PA registries (20+ countries)
- EU Taxonomy DNSH alignment and reporting

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable; graph enrichment API defined |
| AGENT-EUDR-002 Geolocation Verification Agent | BUILT (100%) | Low | Plot coordinates validated and available |
| AGENT-EUDR-006 Plot Boundary Manager Agent | BUILT (100%) | Low | Plot polygons available for overlap analysis |
| AGENT-EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Environmental governance integration point defined |
| AGENT-EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Supplier risk enrichment API defined |
| AGENT-EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Deforestation-PA correlation API defined |
| AGENT-EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | Compound risk cross-reference API defined |
| AGENT-DATA-006 GIS/Mapping Connector | BUILT (100%) | Low | PostGIS utilities available |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration target available |
| PostgreSQL + TimescaleDB + PostGIS | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| WDPA/Protected Planet (UNEP-WCMC) | Available (open data, monthly releases) | Low | Data download with offline cache; monthly refresh pipeline |
| Key Biodiversity Areas Database (BirdLife/IUCN) | Available (partnership access) | Medium | Establish data partnership; use WDPA KBA flag as fallback |
| IUCN Red List API | Available (API key required) | Low | API key obtained; rate-limited queries with local caching |
| GD-PAME (Management Effectiveness) | Available (open data) | Medium | Limited coverage; default scores for PAs without METT assessment |
| UNESCO World Heritage List | Available (open data) | Low | Static data; updated on inscription events |
| Ramsar Sites Database | Available (open data) | Low | Static data; updated on designation events |
| ICMBio/CNUC Brazil PA Registry | Available (government portal) | Medium | Government API may change format; adapter pattern for ingestion |
| KLHK Indonesia PA Registry | Available (government portal) | Medium | Data format varies; adapter pattern; WDPA Indonesia data as fallback |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | WDPA protected area boundary data has accuracy limitations for some regions (point-only records, low GIS quality) | High | Medium | Confidence scoring per PA record based on WDPA GIS quality field; flag low-confidence overlaps for manual review; supplement with national registry data where available |
| R2 | WDPA release schedule changes or data format updates break ingestion pipeline | Low | High | Adapter pattern isolates WDPA ingestion layer; version-controlled schema mapping; automated format detection; manual fallback procedure documented |
| R3 | Batch screening of 10,000+ plots against 270,000+ PAs may not meet < 5 minute SLA under peak load | Medium | High | PostGIS spatial indexing (GIST), R-tree pre-filtering, bounding box elimination, parallel batch processing with worker pool; load testing with 50K plots |
| R4 | Protected area boundary changes (expansions, reductions) between WDPA releases create temporary data gaps | Medium | Medium | Boundary history tracking; monitoring for out-of-cycle boundary changes; national registry cross-referencing for major PAs; 30-day freshness target |
| R5 | IUCN category "Not Reported" for many PAs (estimated 30% of WDPA) creates scoring uncertainty | High | Medium | Default HIGH risk score (75) for NR category; encourage operators to request category clarification from national authorities; track category updates |
| R6 | National buffer zone regulations vary significantly across jurisdictions | High | Medium | Country-specific buffer rule engine with configurable parameters per country; start with 5 major producing countries; extend coverage iteratively |
| R7 | Conservation assessment (IUCN Red List) data is incomplete for some regions and species | Medium | Low | Biodiversity scoring uses available data; missing data scored conservatively; species count normalized to comparable regional baselines |
| R8 | Integration complexity with 7 upstream EUDR agents and multiple external data sources | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern; integration health monitoring; retry logic for external APIs |
| R9 | 30x30 target (CBD GBF) drives rapid expansion of protected areas, requiring frequent database updates | Medium | Medium | Monthly WDPA refresh pipeline; boundary change detection and re-screening automation; proactive alerting for proposed PAs near supply chains |
| R10 | Customer resistance to buffer zone compliance (seen as exceeding legal requirements) | Medium | Medium | Configurable buffer radii (operator choice); demonstrate national legal requirements; show certification scheme alignment; provide cost-benefit analysis |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Protected Area Database Tests | 80+ | WDPA ingestion, spatial indexing, IUCN classification, version control, search, coverage statistics |
| Overlap Detection Tests | 100+ | PostGIS intersection, 5-type classification, IUCN-weighted scoring, batch screening, multi-designation overlaps |
| Buffer Zone Tests | 70+ | Multi-radius buffer generation, proximity tier classification, national regulation compliance, encroachment trends |
| Designation Validation Tests | 50+ | 4-tier designation, METT scoring, enforcement assessment, WH/Ramsar/AZE/KBA flagging, history tracking |
| Proximity Alert Tests | 60+ | Severity classification, enhanced DD workflow, compound risk, alert deduplication, alert lifecycle |
| Compliance Tracking Tests | 50+ | 8-status lifecycle, remediation SLA, exemption management, compliance scoring, audit trail |
| Conservation Assessment Tests | 40+ | Red List integration, biodiversity scoring, habitat fragmentation, species-commodity matrix |
| Compliance Reporting Tests | 40+ | All 8 report types, 5 formats, 5 languages, template rendering, provenance |
| Agent Integration Tests | 50+ | Cross-agent data flow with EUDR-001/002/006/016/020/021, event bus, webhook |
| API Tests | 50+ | All ~30 endpoints, auth, error handling, pagination, rate limiting |
| Golden Tests | 50+ | 7 commodities x 7 scenarios (see 13.2), cross-validated against manual GIS analysis |
| Performance Tests | 25+ | Single-plot overlap, batch 10K/50K plots, concurrent queries, buffer generation, report generation |
| Determinism Tests | 15+ | Bit-perfect reproducibility for IUCN scoring, overlap detection, buffer analysis |
| Data Ingestion Tests | 20+ | WDPA format parsing, incremental update, boundary change detection, version tracking |
| **Total** | **700+** | |

### 13.2 Golden Test Scenarios

Each of the 7 EUDR commodities will have 7 golden test scenarios:

1. **Clear plot** -- Plot with no protected area within 50km -> expect CLEAR classification, no overlap, no buffer alert
2. **Inside IUCN Ia/II** -- Plot centroid inside Strict Nature Reserve or National Park -> expect INSIDE/CRITICAL, maximum risk score, mandatory investigation
3. **Partial overlap IUCN III/IV** -- Plot polygon intersects protected area boundary -> expect PARTIAL/HIGH, overlap metrics calculated correctly, IUCN-weighted score
4. **Buffer zone IUCN II** -- Plot within 5km of National Park -> expect BUFFER/ELEVATED, buffer tier CLOSE, national regulation check
5. **UNESCO World Heritage proximity** -- Plot within 10km of WH site -> expect HIGH-DESIGNATION alert, enhanced DD triggered, compound designation scoring
6. **Deforestation correlation** -- Plot overlapping IUCN IV PA with active deforestation alert -> expect MAXIMUM CRITICAL, deforestation_correlation = TRUE
7. **Multi-designation overlap** -- Plot inside PA that is National Park + World Heritage + KBA + AZE -> expect all designations reported, highest risk score applied, alert deduplication

Total: 7 commodities x 7 scenarios = 49 golden test scenarios (+ 1 compound risk scenario with indigenous territory = 50)

### 13.3 Determinism Tests

Every scoring and calculation engine will include determinism tests that:
1. Run the same calculation 100 times with identical inputs
2. Verify bit-perfect identical outputs (SHA-256 hash match)
3. Test across Python versions (3.11, 3.12) to ensure no platform-dependent behavior
4. Verify Decimal arithmetic produces identical results to reference calculations
5. Verify PostGIS spatial calculations produce identical overlap metrics across PostGIS versions
6. Verify IUCN category scoring is deterministic regardless of query order
7. Verify buffer zone generation produces identical polygons across runs

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **WDPA** | World Database on Protected Areas -- the most comprehensive global database of marine and terrestrial protected areas, maintained by UNEP-WCMC |
| **Protected Planet** | The online interface for the WDPA, managed by UNEP-WCMC and IUCN |
| **UNEP-WCMC** | United Nations Environment Programme World Conservation Monitoring Centre |
| **IUCN** | International Union for Conservation of Nature |
| **IUCN Category** | Management category system (Ia-VI) classifying protected areas by management objective |
| **KBA** | Key Biodiversity Area -- sites identified as globally significant for biodiversity by BirdLife/IUCN |
| **AZE** | Alliance for Zero Extinction -- sites holding the last remaining populations of highly threatened species |
| **CBD** | Convention on Biological Diversity (1992) -- international treaty for biodiversity conservation |
| **GBF** | Kunming-Montreal Global Biodiversity Framework (2022) -- CBD framework including 30x30 target |
| **30x30** | Global target to protect 30% of land and sea areas by 2030 under CBD GBF Target 3 |
| **CITES** | Convention on International Trade in Endangered Species of Wild Fauna and Flora |
| **Ramsar** | Convention on Wetlands of International Importance (Ramsar Convention, 1971) |
| **UNESCO WH** | United Nations Educational, Scientific and Cultural Organization World Heritage Convention |
| **MAB** | Man and the Biosphere Programme -- UNESCO programme with Biosphere Reserves network |
| **METT** | Management Effectiveness Tracking Tool -- standardized assessment of PA management effectiveness |
| **GD-PAME** | Global Database on Protected Areas Management Effectiveness |
| **PAME** | Protected Area Management Effectiveness |
| **ICMBio** | Instituto Chico Mendes de Conservacao da Biodiversidade (Brazil PA management agency) |
| **CNUC** | Cadastro Nacional de Unidades de Conservacao (Brazil National Registry of Conservation Units) |
| **KLHK** | Kementerian Lingkungan Hidup dan Kehutanan (Indonesia Ministry of Environment and Forestry) |
| **ICCN** | Institut Congolais pour la Conservation de la Nature (DRC conservation agency) |
| **SERNANP** | Servicio Nacional de Areas Naturales Protegidas por el Estado (Peru PA agency) |
| **SINAP** | Sistema Nacional de Areas Protegidas (Colombia National Protected Areas System) |
| **HCV** | High Conservation Value -- areas of outstanding biological, ecological, social, or cultural significance |
| **PostGIS** | Spatial database extension for PostgreSQL enabling geographic queries |
| **GIST Index** | Generalized Search Tree -- PostGIS index type for spatial queries |
| **DNSH** | Do No Significant Harm -- EU Taxonomy principle for environmental objectives |
| **CSDDD** | Corporate Sustainability Due Diligence Directive (EU, 2024) |
| **SMART** | Spatial Monitoring and Reporting Tool -- conservation area management tool |

### Appendix B: IUCN Protected Area Categories -- Detailed Reference

| Category | Name | Primary Objective | Allowed Activities | % of Global PAs |
|----------|------|-------------------|-------------------|-----------------|
| **Ia** | Strict Nature Reserve | Biodiversity preservation; scientific research | Scientific research only; no resource extraction | ~5% |
| **Ib** | Wilderness Area | Wilderness protection; ecological processes | Low-impact recreation; no infrastructure | ~3% |
| **II** | National Park | Ecosystem conservation; recreation | Tourism, education; no extractive use | ~12% |
| **III** | Natural Monument | Conservation of specific natural features | Managed visitation; no extractive use | ~4% |
| **IV** | Habitat/Species Management | Active conservation management | Managed intervention; limited resource use | ~8% |
| **V** | Protected Landscape | Conservation with traditional land management | Sustainable agriculture; cultural practices | ~10% |
| **VI** | Sustainable Use | Long-term conservation with sustainable use | Managed resource extraction under management plan | ~18% |
| **NR** | Not Reported | Category not assigned to WDPA | Unknown -- conservative assessment required | ~40% |

### Appendix C: WDPA Metadata Fields Used

| WDPA Field | Agent Field | Usage |
|-----------|-------------|-------|
| WDPAID | wdpa_id | Primary key for protected area records |
| NAME | name | Protected area name (English) |
| ORIG_NAME | orig_name | Protected area name in local language |
| DESIG | designation | Designation type (e.g., "National Park") |
| DESIG_TYPE | designation_level | International/National/Regional/Local mapping |
| IUCN_CAT | iucn_category | IUCN management category (Ia-VI, NR) |
| ISO3 | iso3 | ISO 3166-1 alpha-3 country code |
| REP_AREA | area_hectares | Reported total area in hectares |
| REP_M_AREA | marine_area_hectares | Reported marine area in hectares |
| STATUS | legal_status | Designated/Proposed/Inscribed/Adopted/Established |
| STATUS_YR | status_year | Year of current legal status |
| GOV_TYPE | governance_type | Federal/Sub-national/Collaborative/Private/Indigenous |
| MANG_AUTH | management_authority | Managing authority name |
| MANG_PLAN | management_plan | Management plan status |
| GIS_AREA | (calculated) | GIS-calculated area for validation |
| GIS_M_AREA | (calculated) | GIS-calculated marine area |
| geometry | boundary_geom | PostGIS MultiPolygon geometry |

### Appendix D: National Protected Area Buffer Zone Regulations

| Country | Regulation | Buffer Requirement | IUCN Categories Affected | Agent Implementation |
|---------|-----------|-------------------|------------------------|---------------------|
| **Brazil** | Forest Code (Lei 12.651/2012), CONAMA 428/2010 | 3-10km buffer zone around Conservation Units; environmental licensing required for activities within buffer | All categories; stricter for IUCN I-IV | Configurable per-category buffer; default 10km for Cat I-IV, 5km for Cat V-VI |
| **Indonesia** | PP 28/2011 (Spatial Planning), PP 26/2008 (Spatial Plan for Sumatra) | Buffer zones around kawasan konservasi; width varies by protected area type and local regulation | All national PAs; stricter for Taman Nasional | Configurable per-PA type; default 5km; Taman Nasional 10km |
| **Colombia** | Decree 2372/2010 (SINAP) | Buffer zones (zonas amortiguadoras) required for all national PAs; defined in each PA's management plan | All SINAP protected areas | Use PA-specific buffer from management plan when available; default 5km |
| **Peru** | Law 26834 (Natural Protected Areas), DS 038-2001 | Buffer zones (zonas de amortiguamiento) defined per protected area | All SERNANP-managed PAs | Use PA-specific buffer from management plan; default 5km |
| **DRC** | Law 14/003 (Nature Conservation) | Buffer zones around protected areas; width defined per PA | National parks and reserves | Default 10km for national parks; 5km for reserves |

### Appendix E: Integration API Contracts

**Provided to EUDR-001 (Supply Chain Mapping Master):**
```python
# Protected area risk data for graph node enrichment
def get_protected_area_risk(plot_id: str) -> Dict:
    """Returns: {
        protected_area_overlap: bool,
        overlap_type: str,           # inside/partial/boundary/buffer/clear
        highest_iucn_category: str,  # Ia/Ib/II/III/IV/V/VI/NR
        overlap_risk_score: Decimal, # 0-100
        buffer_zone_proximity: str,  # immediate/close/moderate/distant/peripheral/none
        high_risk_designations: List[str],  # UNESCO_WH/Ramsar/AZE/KBA
        active_alerts: int,
        compliance_status: str,
        protected_area_risk_level: str,  # critical/high/medium/low/clear
        provenance_hash: str
    }"""
```

**Provided to EUDR-016 (Country Risk Evaluator):**
```python
# Country protected area governance score for environmental governance index
def get_country_pa_governance_score(country_code: str) -> Dict:
    """Returns: {
        country_code: str,
        total_protected_areas: int,
        pct_land_protected: Decimal,    # % of land area
        iucn_category_distribution: Dict,
        avg_management_effectiveness: Decimal,  # 0-100
        enforcement_score: Decimal,     # 0-100
        composite_pa_governance_score: Decimal,  # 0-100
        risk_level: str,                # low/standard/high
        provenance_hash: str
    }"""
```

**Provided to EUDR-017 (Supplier Risk Scorer):**
```python
# Supplier protected area risk for composite supplier scoring
def get_supplier_pa_risk(supplier_id: str) -> Dict:
    """Returns: {
        supplier_id: str,
        total_plots: int,
        plots_inside_pa: int,
        plots_partial_overlap: int,
        plots_in_buffer: int,
        plots_clear: int,
        highest_iucn_exposure: str,     # Highest IUCN category overlap
        active_violations: int,
        active_high_risk_alerts: int,
        protected_area_risk_score: Decimal,  # 0-100
        provenance_hash: str
    }"""
```

### Appendix F: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. European Commission EUDR Guidance Document -- Risk Assessment and Due Diligence
3. UNEP-WCMC and IUCN (2025) -- Protected Planet: The World Database on Protected Areas (WDPA), Cambridge, UK
4. IUCN (2008) -- Guidelines for Applying Protected Area Management Categories
5. Dudley, N. (ed.) (2008) -- Guidelines for Applying Protected Area Management Categories, IUCN, Gland, Switzerland
6. Convention on Biological Diversity (1992) -- Text and Annexes
7. Kunming-Montreal Global Biodiversity Framework (2022) -- CBD/COP/DEC/15/4
8. Convention on International Trade in Endangered Species (CITES, 1973) -- Text and Appendices
9. UNESCO World Heritage Convention (1972) -- Operational Guidelines for the Implementation
10. Ramsar Convention on Wetlands (1971) -- Strategic Plan 2016-2024
11. EU Biodiversity Strategy for 2030 -- COM(2020) 380
12. Directive (EU) 2024/1760 -- Corporate Sustainability Due Diligence Directive (CSDDD)
13. Regulation (EU) 2020/852 -- EU Taxonomy Regulation
14. BirdLife International / IUCN -- Key Biodiversity Areas: Identification and Gap Analysis
15. Alliance for Zero Extinction (AZE) -- Site identification methodology
16. Leverington, F. et al. -- Management effectiveness evaluation in protected areas (GD-PAME)
17. IUCN Red List of Threatened Species -- Assessment methodology and categories
18. Brazil Forest Code (Lei 12.651/2012) -- Buffer zone provisions
19. Indonesia PP 28/2011 -- Spatial Planning for Protected Areas
20. Colombia Decree 2372/2010 -- SINAP Protected Areas System
21. Peru Law 26834 -- Natural Protected Areas
22. ISO 3166-1 -- Country Codes Standard

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-10 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Conservation Specialist | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-10 | GL-ProductManager | Initial draft created: all 9 P0 features specified (90 sub-requirements), regulatory coverage verified (EUDR Articles 2/3/9/10/11/29/31, CBD, CITES, World Heritage, Ramsar), IUCN categories Ia-VI risk scoring defined, 15-table DB schema V110 designed, ~30 API endpoints specified, 20 Prometheus metrics defined, 18 RBAC permissions registered, 700+ test target set, 18-week timeline established |
