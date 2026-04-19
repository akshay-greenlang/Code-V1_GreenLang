# PRD: AGENT-EUDR-023 -- Legal Compliance Verifier Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-023 |
| **Agent ID** | GL-EUDR-LCV-023 |
| **Component** | Legal Compliance Verifier Agent |
| **Category** | EUDR Regulatory Agent -- Legal Framework & Legislation Compliance |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 2(40), 3, 4, 8, 9, 10, 11, 29, 31; ILO Conventions; UDHR; UNGP on Business and Human Rights |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) requires that commodities placed on the EU market are not only deforestation-free but also "legally produced" in accordance with all "relevant legislation" of the country of production. Article 2(40) defines "relevant legislation" as encompassing eight distinct categories of law applicable in the country of production: (i) land-use rights, (ii) environmental protection, (iii) forest-related rules including forest management and biodiversity conservation, (iv) third parties' rights, (v) labour rights, (vi) human rights protected under international law, (vii) the principle of free, prior and informed consent (FPIC) including as set out in UNDRIP, and (viii) tax, anti-corruption, trade, and customs regulations. Article 3 conditions market access on legality, and Articles 10(2)(d)-(e) mandate operators to assess the risk of non-compliance by evaluating the prevalence of violations linked to the legal framework of producing countries.

This legality requirement is not merely a reference to environmental law; it encompasses the full breadth of the legal framework governing commodity production in each origin country. A cocoa shipment from Ghana must comply with Ghanaian forestry law, land tenure law, labour law (including child labour prohibitions), environmental impact assessment requirements, tax obligations, and anti-corruption provisions. A timber shipment from Indonesia must demonstrate compliance with Indonesian forestry licensing (SVLK), spatial planning regulations, indigenous customary rights recognition, provincial environmental permits, and export customs requirements. The legal complexity multiplies across the 80+ countries from which EUDR-regulated commodities are sourced.

Today, EU operators and compliance teams face the following critical gaps when verifying legal compliance:

- **No comprehensive legal framework database**: Operators have no systematic access to a structured, queryable database of the legislation applicable to commodity production in each producing country. Legal requirements are scattered across national statutes, provincial regulations, ministerial decrees, and local ordinances in multiple languages. No existing compliance tool consolidates the 8 Article 2(40) legislation categories into a queryable framework covering 20+ producing countries.
- **No document verification engine for legal permits**: Commodity production in most countries requires multiple permits, licenses, and certificates (forestry concession licenses, environmental impact assessments, land title certificates, tax clearance certificates, export permits). There is no automated system to verify the validity, authenticity, and completeness of these documents against country-specific requirements.
- **No certification scheme validation**: Certification schemes (FSC, RSPO, PEFC, Rainforest Alliance, ISCC, UTZ) provide evidence of legal compliance across multiple Article 2(40) categories, but there is no automated mechanism to verify certification validity, scope alignment with EUDR requirements, chain-of-custody certificate status, and certification-to-legislation mapping.
- **No red flag detection for legal non-compliance**: Indicators of illegal production -- corruption risk in permitting authorities, known illegal logging hotspots, forced labour indicators, tax haven routing, permit fraud patterns -- are not systematically identified, correlated with supply chains, or used to trigger enhanced due diligence.
- **No country-specific compliance checker**: Legal requirements vary drastically across jurisdictions. What constitutes legal timber in Brazil (DOF system, IBAMA licensing) is entirely different from legal timber in Indonesia (SVLK/FLEGT), DRC (Forest Code permits), or Myanmar (export ban verification). There is no system that applies country-specific compliance rules to evaluate legality per jurisdiction.
- **No third-party audit integration**: Independent auditors assess legal compliance through field inspections, document reviews, and stakeholder consultations. There is no structured mechanism to ingest audit reports, extract compliance findings, track corrective actions, and integrate audit results into the compliance record.
- **No legal opinion management**: Complex supply chains may require legal opinions from qualified attorneys in producing countries to confirm the legality of production under national law. There is no system to manage legal opinion requirements, track opinion validity, store opinions with provenance, or link opinions to specific compliance determinations.
- **No multi-category compliance reporting**: Auditors and regulators reviewing EUDR due diligence statements need structured evidence that all 8 Article 2(40) legislation categories have been assessed. There is no standardized reporting framework that generates audit-ready legal compliance documentation covering all categories with per-country, per-supplier, and per-commodity analysis.

Without solving these problems, EU operators face penalties of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, public naming, and criminal liability under national implementing legislation for knowingly placing illegally produced goods on the EU market.

### 1.2 Solution Overview

Agent-EUDR-023: Legal Compliance Verifier is a specialized compliance agent that provides comprehensive legal compliance verification across all 8 Article 2(40) legislation categories for EUDR-regulated commodity supply chains. It is the 23rd agent in the EUDR agent family and extends the Risk Assessment sub-category (EUDR-016 through EUDR-022) with dedicated legal framework compliance capabilities. The agent maintains a structured database of legal requirements for 20+ commodity-producing countries, automates permit and license verification, validates certification scheme compliance, detects legal non-compliance red flags, applies country-specific compliance rules, integrates third-party audit findings, manages legal opinions, and generates audit-ready compliance documentation.

The agent builds on and integrates with the existing EUDR agent ecosystem: EUDR-001 (Supply Chain Mapping Master) for supply chain graph data, EUDR-002 (Geolocation Verification) for plot coordinate validation, EUDR-008 (Multi-Tier Supplier Tracker) for supplier chain information, EUDR-012 (Document Authentication) for document integrity verification, EUDR-016 (Country Risk Evaluator) for governance and corruption scoring, EUDR-017 (Supplier Risk Scorer) for supplier-level risk assessment, EUDR-019 (Corruption Index Monitor) for corruption risk data, EUDR-021 (Indigenous Rights Checker) for FPIC and indigenous rights compliance, and EUDR-022 (Protected Area Validator) for environmental protected area compliance.

Core capabilities:

1. **Legal framework database integration** -- Structured, queryable database of legislation applicable to EUDR-regulated commodity production across 8 Article 2(40) categories for 20+ producing countries. Per-country legislation profiles covering land-use rights, environmental protection, forest-related rules, third parties' rights, labour rights, human rights, FPIC, and tax/anti-corruption/trade/customs. Legislation version tracking, amendment monitoring, and enforceability assessment.
2. **Document verification engine** -- Automated verification of permits, licenses, certificates, and legal documents required for legal commodity production. Country-specific document requirement matrices (e.g., Brazil DOF + IBAMA license + CAR registration + ITR tax clearance; Indonesia SVLK + HGU land title + AMDAL EIA + PNBP royalty receipt). Document validity checking (expiry dates, issuing authority verification, scope alignment). Deterministic completeness scoring (0-100) with provenance tracking.
3. **Certification scheme validator** -- Integration with 6 major certification scheme databases (FSC, RSPO, PEFC, Rainforest Alliance, ISCC, UTZ/Rainforest Alliance merged). Certificate validity verification (active/suspended/withdrawn/expired), scope alignment with EUDR requirements (which Article 2(40) categories does the certification cover), chain-of-custody status, and certification-to-legislation equivalence mapping.
4. **Red flag detection engine** -- Rule-based detection of legal non-compliance indicators across all 8 Article 2(40) categories. Corruption risk scoring for permitting authorities (CPI-based), illegal logging probability indicators (timber legality verification gaps, known illegal logging regions), forced labour risk indicators (ILO indicators, US DOL ILAB lists), tax evasion patterns (transfer pricing anomalies, tax haven routing), and permit fraud indicators (duplicate permit numbers, expired permits presented as valid, issuing authority mismatch).
5. **Country-specific compliance checker** -- Per-country legal compliance rule engine covering the specific permits, licenses, registrations, and legal requirements for each EUDR commodity in each producing country. Supports 20+ countries with commodity-specific legal requirement matrices. Evaluates compliance status per legislation category and generates per-country compliance gap analysis.
6. **Third-party audit integration** -- Structured ingestion of audit reports from accredited verification bodies, certification auditors, and independent assessors. Extracts audit findings (major non-conformities, minor non-conformities, observations), tracks corrective action plans (CAPs) with SLA deadlines, and integrates audit results into the supplier compliance record.
7. **Legal opinion manager** -- Manages legal opinion requirements for complex compliance determinations. Identifies when legal opinions are required (conflicting legal requirements, novel commodity classifications, disputed land tenure), tracks opinion status (requested/received/valid/expired), stores opinions with SHA-256 provenance hashes, and links opinions to specific compliance determinations.
8. **Compliance reporting engine** -- Automated generation of 8 report types (full legal compliance, per-category breakdown, per-supplier scorecard, per-commodity analysis, DDS legal compliance section, certification audit evidence, trend analysis, BI export) in 5 formats (PDF, JSON, HTML, CSV, XLSX) with 5 language support (EN, FR, DE, ES, PT). SHA-256 provenance hashes on all reports.
9. **Integration with existing EUDR agents** -- Bidirectional integration with EUDR-001 (supply chain graph nodes enriched with legal compliance data), EUDR-002 (plot coordinates linked to country legal requirements), EUDR-008 (supplier chain legal compliance propagation), EUDR-012 (document authentication for permit verification), EUDR-016 (country governance scores inform legal risk), EUDR-017 (supplier risk adjusted for legal compliance), EUDR-019 (corruption data feeds red flag detection), EUDR-021 (indigenous rights compliance feeds FPIC category), and EUDR-022 (protected area compliance feeds environmental category).

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Legal framework coverage | 20+ countries across all 8 Article 2(40) categories | Count of country-category combinations in legal database |
| Document type coverage | 150+ document types across 20+ countries | Count of recognized document types in verification engine |
| Certification scheme integration | 6 schemes (FSC, RSPO, PEFC, RA, ISCC, UTZ) | Scheme count with active API/database integration |
| Document verification accuracy | >= 98% correct validity determination (validated against manual review) | Cross-validation with legal expert review |
| Document verification performance | < 2 seconds per document verification | p99 latency under load |
| Red flag detection precision | >= 90% precision (validated against known violation cases) | Precision/recall against test corpus of 500+ cases |
| Country compliance check performance | < 500ms per country-commodity compliance evaluation | p99 latency under load |
| Audit report ingestion | < 5 minutes per structured audit report | Time from upload to findings extracted |
| Compliance report generation | < 10 seconds per legal compliance report | Time from request to PDF/JSON delivery |
| EUDR regulatory coverage | 100% of Article 2(40) eight categories assessed | Regulatory compliance matrix |
| Agent integration health | 99.9% message delivery for cross-agent events | Integration health monitoring |
| Zero-hallucination guarantee | 100% deterministic calculations, no LLM in critical path | Bit-perfect reproducibility tests |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, plus the broader legal compliance technology market for commodity supply chains estimated at 3-5 billion EUR as EUDR, CSDDD, and FLEGT enforcement scales.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of EUDR-regulated commodities requiring legal compliance verification across 8 legislation categories for multi-country supply chains, estimated at 500-800M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 30-50M EUR in legal compliance module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) sourcing from countries with complex legal frameworks (Brazil, Indonesia, DRC, Colombia, Peru, Cameroon, Ghana, Cote d'Ivoire)
- Multinational food and beverage companies with cocoa, coffee, palm oil, and soya supply chains requiring multi-country legal compliance verification
- Timber and paper industry operators with tropical wood sourcing requiring timber legality verification (FLEGT, SVLK, DOF compliance)
- Compliance officers and legal departments responsible for EUDR due diligence with Article 2(40) legal compliance obligations

**Secondary:**
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance) requiring legal compliance evidence as part of certification audits
- Commodity traders and intermediaries operating in jurisdictions with high corruption or weak rule of law
- Financial institutions with exposure to EUDR-regulated commodity supply chains requiring legal due diligence under CSDDD and EU Taxonomy
- Law firms and compliance consultants advising EU operators on EUDR legal compliance
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Legal Counsel | Deep legal expertise; jurisdiction-specific | Costs EUR 50-200K per assessment; takes months; not scalable; no systematic tracking | Automated, < 15 minutes per supply chain, deterministic, scalable to 500+ supply chains |
| Generic Compliance Platforms (SAP GRC, MetricStream) | Enterprise integration; broad compliance scope | Not EUDR-specific; no Article 2(40) category structure; no country-specific permit verification; no certification mapping | Purpose-built for EUDR Article 2(40); 8-category coverage; country-specific rule engines |
| Timber Legality Tools (Timber Risk Tool, Open Timber Portal) | Deep forestry expertise; timber-specific | Single-commodity; no multi-category coverage; no certification validation; limited country coverage | All 7 commodities; all 8 categories; certification integration; 20+ countries |
| Certification Scheme Databases (FSC Info, RSPO RT) | Authoritative certificate data; free access | Single-scheme; no cross-scheme coverage mapping; no compliance workflow; no document verification | Multi-scheme integration; Article 2(40) coverage mapping; compliance workflow; document verification |
| In-house Custom Builds | Tailored to organization | 12-18 month build; no regulatory updates; no legal database maintenance | Ready now; continuous legal database updates; production-grade; regulatory update pipeline |

### 2.4 Differentiation Strategy

1. **8-category legal framework structure** -- Not a generic compliance check; systematically addresses all 8 Article 2(40) categories with per-country rule engines.
2. **Regulatory fidelity** -- Every data field maps to a specific EUDR Article requirement; compliance reporting aligned with DDS submission format.
3. **Certification-to-legislation mapping** -- Unique capability to map certification scheme coverage to Article 2(40) categories, identifying precisely which gaps remain after certification.
4. **Integration depth** -- Pre-built connectors to 9 EUDR agents (001, 002, 008, 012, 016, 017, 019, 021, 022) and GL-EUDR-APP.
5. **Zero-hallucination legal assessment** -- Deterministic rule-based compliance evaluation with SHA-256 provenance hashes; no LLM in the critical path.
6. **Country-specific rule engines** -- Not one-size-fits-all; 20+ country-specific compliance rule sets with commodity-specific permit requirement matrices.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to achieve EUDR compliance for all 8 Article 2(40) legal requirements | 100% of customers pass Article 10/11 audits for legal compliance component | Q2 2026 |
| BG-2 | Reduce time-to-verify legal compliance from weeks to minutes per supply chain | 95% reduction in verification time (weeks to < 15 minutes per supply chain) | Q2 2026 |
| BG-3 | Become the reference legal compliance verification platform for EUDR | 500+ enterprise customers using legal compliance module | Q4 2026 |
| BG-4 | Prevent EUDR penalties attributable to legal non-compliance | Zero EUDR penalties for active customers related to legal compliance gaps | Ongoing |
| BG-5 | Support CSDDD readiness by building legal due diligence infrastructure | Legal compliance module reusable for CSDDD Article 7 environmental and human rights due diligence | Q1 2027 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Comprehensive legal framework database | Structured legislation profiles for 20+ producing countries across all 8 Article 2(40) categories with version tracking |
| PG-2 | Automated document verification | Verify permits, licenses, and certificates for validity, completeness, and scope alignment per country |
| PG-3 | Certification scheme validation | Validate certification status and map certification coverage to Article 2(40) categories |
| PG-4 | Red flag detection | Identify indicators of illegal production across corruption, illegal logging, forced labour, tax evasion, and permit fraud |
| PG-5 | Country-specific compliance | Apply per-country legal requirement matrices for all 7 EUDR commodities |
| PG-6 | Audit integration | Ingest third-party audit reports and track corrective action compliance |
| PG-7 | Audit-ready reporting | Generate compliant legal compliance documentation for DDS, auditors, and certifiers across all 8 categories |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Document verification performance | < 2 seconds p99 per document verification |
| TG-2 | Country compliance evaluation | < 500ms p99 per country-commodity compliance check |
| TG-3 | Batch compliance screening | 1,000 suppliers in < 10 minutes |
| TG-4 | Report generation | < 10 seconds per legal compliance report PDF |
| TG-5 | API response time | < 200ms p95 for standard queries |
| TG-6 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-7 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |
| TG-8 | Data freshness | Legal framework database updated within 30 days of legislative amendment publication |

---

## 4. Regulatory Requirements

### 4.1 EUDR Article 2(40) -- Eight Categories of "Relevant Legislation"

Article 2(40) of the EUDR defines "relevant legislation" as the legislation applicable in the country of production covering the following:

| Category | Article 2(40) Text | Scope | Agent Feature |
|----------|-------------------|-------|---------------|
| **Cat 1: Land-Use Rights** | "land-use rights" | Land title, land concession, land lease, land registration, customary tenure recognition, spatial planning compliance | Country-specific compliance checker (F5): land title validation, concession verification, spatial planning compliance per country |
| **Cat 2: Environmental Protection** | "environmental protection" | Environmental impact assessments (EIA/AMDAL), environmental permits, pollution control, waste management, water use permits | Document verification engine (F2): EIA/AMDAL validation, environmental permit verification; integration with EUDR-022 protected area compliance |
| **Cat 3: Forest-Related Rules** | "forest-related rules, including forest management and biodiversity conservation, where directly related to forest harvesting" | Forestry concession licenses, annual allowable cut permits, forest management plans, selective logging permits, reforestation obligations, timber legality verification (FLEGT/SVLK/DOF) | Document verification (F2) + country compliance checker (F5): forestry license verification per country legal framework |
| **Cat 4: Third Parties' Rights** | "third parties' rights" | Indigenous peoples' rights, community land rights, benefit-sharing agreements, consultation obligations, customary use rights | Integration with EUDR-021 (Indigenous Rights Checker): FPIC verification, land rights overlap, community consultation records |
| **Cat 5: Labour Rights** | "labour rights" | ILO core conventions (child labour, forced labour, discrimination, freedom of association), minimum wage, occupational health and safety, working hours, employment contracts | Red flag detection engine (F4): forced labour indicators, child labour risk, ILO core convention compliance assessment per country |
| **Cat 6: Human Rights** | "human rights protected under international law" | UDHR, ICCPR, ICESCR, rights to life, liberty, security, fair trial, freedom from torture; UN Guiding Principles on Business and Human Rights | Red flag detection (F4): human rights violation indicators, UNGP due diligence framework alignment; integration with EUDR-021 |
| **Cat 7: FPIC** | "the principle of free, prior and informed consent (FPIC) including as set out in the United Nations Declaration on the Rights of Indigenous Peoples" | FPIC processes, consent documentation, community engagement, UNDRIP compliance | Integration with EUDR-021 (Indigenous Rights Checker): FPIC verification engine, FPIC workflow management |
| **Cat 8: Tax, Anti-Corruption, Trade, Customs** | "tax, anti-corruption, trade and customs regulations" | Corporate tax compliance, royalty payments, anti-bribery compliance, export permits, customs declarations, CITES permits, transfer pricing | Document verification (F2): tax clearance certificates, export permits; red flag detection (F4): corruption indicators, tax evasion patterns |

### 4.2 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(28)** | Definition of "legally produced" -- produced in accordance with the relevant legislation of the country of production | Full 8-category legal compliance verification across all agent features; country-specific compliance checker (F5) evaluates legality per jurisdiction |
| **Art. 2(40)** | Definition of "relevant legislation" -- 8 categories of applicable law | Legal framework database (F1) structures all 8 categories per country; compliance reporting (F8) generates per-category compliance evidence |
| **Art. 3** | Prohibition on placing non-compliant products on the EU market | Compliance reporting (F8) flags products with unresolved legal compliance gaps across any of the 8 categories as non-compliant |
| **Art. 4(2)** | Due diligence -- collect information, assess risk, mitigate risk | Full agent pipeline: legal database (F1) for information, document verification (F2) and red flag detection (F4) for risk assessment, audit integration (F6) and legal opinions (F7) for risk mitigation |
| **Art. 8** | Due diligence obligations requiring operators to collect supply chain information and assess legality | Legal framework database (F1) + document verification (F2) + country compliance checker (F5) provide systematic legality assessment |
| **Art. 9(1)(h)** | Information on the commodities being in compliance with the relevant legislation | All 9 features collectively produce the evidence base for Article 9(1)(h) compliance confirmation |
| **Art. 10(1)** | Risk assessment -- operators shall assess and identify risk of non-compliance | Legal compliance risk scoring across all 9 features contributes to Article 10 risk assessment |
| **Art. 10(2)(d)** | Risk assessment criterion: prevalence of deforestation or forest degradation linked to violations | Red flag detection engine (F4) identifies legal violation patterns in supply chain regions; legal non-compliance prevalence feeds risk scoring |
| **Art. 10(2)(e)** | Risk assessment criterion: concerns about the country of production, including level of corruption and prevalence of document fraud | Country legal compliance scoring (F5) + corruption-informed red flag detection (F4) + integration with EUDR-019 corruption data |
| **Art. 11** | Risk mitigation measures | Third-party audit integration (F6), legal opinion management (F7), and compliance tracking provide structured risk mitigation tools |
| **Art. 29(3)** | Country benchmarking -- Commission shall consider rate of deforestation, governance quality, legal enforcement | Per-country legal compliance scoring feeds EUDR-016 country risk evaluator for Article 29 benchmarking input |
| **Art. 31** | Record keeping for 5 years | All legal compliance data, document verification records, audit findings, legal opinions, and compliance reports retained for minimum 5 years with immutable audit trail |

### 4.3 International Legal Framework

| Legal Instrument | Status | Agent Relevance |
|-----------------|--------|----------------|
| **ILO Core Conventions** (C029 Forced Labour, C087 Freedom of Association, C098 Right to Organise, C100 Equal Remuneration, C105 Abolition of Forced Labour, C111 Discrimination, C138 Minimum Age, C182 Worst Forms of Child Labour) | Ratified by all major producing countries (with exceptions) | Red flag detection engine (F4) assesses labour rights compliance per ILO convention ratification status; country compliance checker (F5) evaluates labour law alignment |
| **UDHR** (Universal Declaration of Human Rights, 1948) | Universal (non-binding) | Human rights category (Cat 6) assessment framework; red flag detection (F4) uses UDHR articles as reference standard |
| **UNGP** (UN Guiding Principles on Business and Human Rights, 2011) | Endorsed by UN Human Rights Council; referenced in CSDDD | Due diligence framework for human rights assessment (Cat 6); risk assessment methodology aligned with UNGP Pillar II |
| **ICCPR / ICESCR** | International Covenants on Civil and Political Rights / Economic, Social and Cultural Rights | Human rights (Cat 6) compliance framework; country-specific human rights assessment |
| **UNDRIP** (UN Declaration on the Rights of Indigenous Peoples, 2007) | Referenced explicitly in Article 2(40) Cat 7 | FPIC compliance assessment; integration with EUDR-021 for UNDRIP alignment |
| **ILO Convention 169** (Indigenous and Tribal Peoples, 1989) | Ratified by 24 countries | FPIC (Cat 7) legal requirement assessment per ratifying country; integration with EUDR-021 |
| **CITES** (Convention on International Trade in Endangered Species, 1975) | 184 parties | Trade/customs (Cat 8) compliance: CITES permit verification for timber species |
| **FLEGT** (Forest Law Enforcement, Governance and Trade) | EU Regulation 2173/2005; VPAs with 7 countries | Forest-related rules (Cat 3) compliance for VPA partner countries; FLEGT license verification |
| **EU CSDDD** (Corporate Sustainability Due Diligence Directive, 2024) | Effective 2027 | Agent infrastructure designed for CSDDD Article 7 legal due diligence reuse |
| **EU Taxonomy Regulation** (2020/852) | Active; DNSH criteria | Legal compliance data supports EU Taxonomy Article 17 DNSH assessment |
| **OECD Due Diligence Guidance** (Responsible Supply Chains, 2018) | Widely adopted by EU companies | Risk assessment methodology aligned with OECD 5-step framework |

### 4.4 Country-Specific Legal Framework Summary

| Country | Key Legislation | Commodities | Required Permits/Documents | Agent Feature |
|---------|----------------|-------------|---------------------------|---------------|
| **Brazil** | Forest Code (Lei 12.651/2012), Environmental Crimes Law (Lei 9.605/1998), CAR Registration, IBAMA Licensing | Cattle, soya, cocoa, coffee, wood | DOF (Document of Forest Origin), GF (Forest Management Authorization), CAR (Rural Environmental Registry), ITR (Rural Land Tax), IBAMA License, LAR (Environmental License) | F2 + F5: Brazil compliance rule set |
| **Indonesia** | Forestry Law 41/1999, SVLK (Timber Legality Verification), HGU Land Title, AMDAL EIA | Palm oil, rubber, wood, cocoa, coffee | SVLK Certificate, HGU/HGB Land Title, AMDAL EIA Approval, IPPKH Forest Permit, PNBP Royalty Receipt, STLHK Legality Certificate | F2 + F5: Indonesia compliance rule set |
| **DRC** | Forest Code (Lei 011/2002), Mining Code, Land Law | Wood, cocoa, coffee | Forest Concession Contract, Annual Coupe Permit, EIA Certificate, Tax Clearance, FLEGT License (pending) | F2 + F5: DRC compliance rule set |
| **Colombia** | Forest Law (Decreto 1076/2015), Environmental License Decree | Coffee, palm oil, cocoa, wood | Salvoconducto (Timber Transport Permit), Environmental License, Prior Consultation Certificate, ICA Phytosanitary Certificate | F2 + F5: Colombia compliance rule set |
| **Peru** | Forest Law (Ley 29763), Environmental Law | Coffee, cocoa, wood, palm oil | OSINFOR Forest Inspection Certificate, EIA Approval, SERFOR Permit, SUNAT Tax Registration | F2 + F5: Peru compliance rule set |
| **Cote d'Ivoire** | Forest Code (2019), Land Law 98-750 | Cocoa, coffee, rubber, palm oil | Forest Exploitation Permit, Land Certificate, Tax Clearance, ANADER Agricultural Permit, Export License | F2 + F5: CDI compliance rule set |
| **Ghana** | Forestry Commission Act (1999), Timber Resource Management Act | Cocoa, wood | TUC (Timber Utilization Contract), SFC (Salvage Felling Certificate), VLTP, Cocoa Board License, GRA Tax Clearance | F2 + F5: Ghana compliance rule set |
| **Malaysia** | National Forestry Act 1984, MTCS/PEFC, Sarawak Forests Ordinance | Palm oil, rubber, wood | Timber License, MTCS Certificate, State Land Title, Environmental Quality Act Permit, MPOB License | F2 + F5: Malaysia compliance rule set |

### 4.5 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for legal compliance assessment (permits must cover production from this date forward) |
| June 29, 2023 | EUDR entered into force | Legal basis for legal compliance verification |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | All legal compliance verification must be operational |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; agent must handle 10x scale |
| 2027 | CSDDD enforcement begins | Legal compliance module reused for CSDDD Article 7 due diligence |
| Ongoing (quarterly) | DDS submission deadlines | Legal compliance reports must be current for quarterly filing |
| Ongoing (periodic) | EC benchmarking updates | Legal compliance country scores updated when EC revises benchmarks |
| Ongoing | National legislation amendments | Legal framework database updated within 30 days of amendment publication |

---

## 5. Scope and Zero-Hallucination Principles

### 5.1 Scope -- In and Out

**In Scope (v1.0):**
- Legal framework database covering 20+ producing countries across all 8 Article 2(40) categories
- Document verification engine for 150+ permit/license/certificate types with validity, authenticity, and completeness checking
- Certification scheme validation for 6 schemes (FSC, RSPO, PEFC, Rainforest Alliance, ISCC, UTZ)
- Red flag detection across corruption, illegal logging, forced labour, tax evasion, and permit fraud indicators
- Country-specific compliance checker with per-country legal requirement matrices for all 7 EUDR commodities
- Third-party audit report integration with findings extraction and corrective action tracking
- Legal opinion management with requirement identification, tracking, and provenance
- Compliance reporting for DDS, auditors, and certification schemes (8 report types, 5 formats, 5 languages)
- Integration with EUDR-001, EUDR-002, EUDR-008, EUDR-012, EUDR-016, EUDR-017, EUDR-019, EUDR-021, EUDR-022

**Out of Scope (v1.0):**
- Legal advisory on specific compliance disputes (agent provides data and assessment; not legal counsel)
- Automated translation of national legislation texts (agent stores structured metadata; not full legal text translation)
- Direct permit application or renewal processing (agent verifies permits; does not apply for them)
- Real-time court database monitoring for legal proceedings against suppliers (defer to Phase 2)
- Mobile native application for field-level document collection (web responsive only)
- Predictive ML models for legal risk forecasting (defer to Phase 2)
- Direct submission to EU Information System (handled by GL-EUDR-APP DDS module)
- Blockchain-based legal compliance records (SHA-256 provenance hashes provide sufficient integrity)

### 5.2 Zero-Hallucination Principles

All 9 features in this agent operate under strict zero-hallucination guarantees:

| Principle | Implementation |
|-----------|---------------|
| **Deterministic calculations** | Same inputs always produce the same compliance scores, document verification results, and red flag assessments (bit-perfect reproducibility) |
| **No LLM in critical path** | All compliance scoring, document verification, red flag detection, and certification validation use deterministic rule-based algorithms only |
| **Authoritative data sources only** | All legal framework data sourced from official government gazettes, ILO databases, certification scheme registries, and published legal databases; no synthesized or inferred legal requirements |
| **Full provenance tracking** | Every compliance score, document verification, and red flag assessment includes SHA-256 hash, data source citations, legal reference, and calculation timestamps |
| **Immutable audit trail** | All legal compliance data changes recorded in `gl_eudr_lcv_audit_log` with before/after values |
| **Decimal arithmetic** | Compliance scores and risk percentages use Decimal type to prevent floating-point drift |
| **Version-controlled legal data** | Legal framework database is versioned; any legislation amendment creates a new version with effective date and source attribution |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-5 form the core legal compliance verification engine; Features 6-9 form the management, reporting, and integration layer.

**P0 Features 1-5: Core Legal Compliance Verification Engine**

---

#### Feature 1: Legal Framework Database Integration

**User Story:**
```
As a compliance officer,
I want a comprehensive, structured database of the legislation applicable to commodity production in each producing country across all 8 EUDR Article 2(40) categories,
So that I can determine what legal requirements apply to my supply chain in each jurisdiction.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F1.1: Maintains structured legislation profiles for 20+ EUDR commodity-producing countries (Brazil, Indonesia, Colombia, Peru, DRC, Cote d'Ivoire, Ghana, Malaysia, Cameroon, Myanmar, Guatemala, Honduras, Paraguay, Bolivia, Ecuador, Nigeria, Liberia, Sierra Leone, Papua New Guinea, Thailand) covering all 8 Article 2(40) categories
- [ ] F1.2: Each country profile contains per-category legislation records: statute name, statute number, enactment date, last amendment date, enforcing authority, applicable commodities, required permits/documents, penalty provisions, and English summary of key obligations
- [ ] F1.3: Maps each country's legislation to the 8 Article 2(40) categories with coverage assessment: FULL (legislation exists and is enforced), PARTIAL (legislation exists but enforcement is weak), GAP (no applicable legislation), CONFLICTING (multiple conflicting provisions)
- [ ] F1.4: Tracks legislation amendments and effective dates: when a country amends forestry law, environmental law, or land tenure legislation, the database is updated within 30 days with old version preserved for historical compliance assessment
- [ ] F1.5: Stores per-country document requirement matrices: for each commodity in each country, lists the specific permits, licenses, registrations, and certificates required for legal production, with issuing authority, validity period, and renewal requirements
- [ ] F1.6: Assesses enforceability per legislation: enforceability score (0-100) based on regulatory capacity, corruption perception index, judicial independence, and known enforcement gaps, sourced from World Justice Project Rule of Law Index, Transparency International CPI, and EUDR-016/019 data
- [ ] F1.7: Provides commodity-specific legal requirement summaries: for each of the 7 EUDR commodities, generates a per-country compliance checklist covering all applicable legislation across all 8 categories
- [ ] F1.8: Supports legal framework search and filtering by: country, Article 2(40) category, commodity, enforcement level, amendment date range, and issuing authority
- [ ] F1.9: Integrates ILO ratification database for labour rights (Cat 5): tracks which ILO core conventions each country has ratified, implementation status, and ILO Committee of Experts observations
- [ ] F1.10: Maintains provenance for all legal data: each legislation record includes source URLs (official gazette, ILO database, FAO FAOLEX), retrieval date, and SHA-256 hash of source document

**Non-Functional Requirements:**
- Coverage: 20+ countries across all 8 Article 2(40) categories
- Data Quality: Source attribution for every legislation record; legal expert review for initial dataset
- Performance: Country legal framework lookup < 100ms; compliance checklist generation < 500ms
- Freshness: Legislative amendments tracked within 30 days of publication

**Dependencies:**
- FAO FAOLEX database for national environmental and forestry legislation
- ILO NATLEX/NORMLEX for labour law and convention ratification
- National official gazettes for legislation text
- EUDR-016 Country Risk Evaluator for governance scoring
- EUDR-019 Corruption Index Monitor for CPI data

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 legal data specialist)

**Edge Cases:**
- Country with no forestry-specific legislation (some small producing countries) -> Map to general environmental law; flag as GAP in forest-related category; recommend enhanced due diligence
- Conflicting national and provincial legislation (e.g., Indonesian national vs. provincial forestry regulations) -> Record both; flag as CONFLICTING; apply more restrictive provision for compliance purposes
- Country with recent regime change and legal framework in transition -> Flag as TRANSITIONAL; apply previous legislation until new framework is published; monitor for updates
- Legislation published only in local language without English translation -> Store original language reference; provide structured metadata summary in English; flag for legal expert review

---

#### Feature 2: Document Verification Engine

**User Story:**
```
As a compliance officer,
I want to verify that the permits, licenses, and certificates presented by my suppliers are valid, authentic, and complete for legal commodity production,
So that I can confirm that production activities comply with the required legal framework in each jurisdiction.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F2.1: Recognizes and validates 150+ document types across 20+ countries, categorized by Article 2(40) category: land titles (Cat 1), environmental permits (Cat 2), forestry licenses (Cat 3), consultation certificates (Cat 4), labour compliance certificates (Cat 5), tax clearance certificates (Cat 8), export permits (Cat 8), and CITES permits (Cat 8)
- [ ] F2.2: Verifies document validity: checks expiry date, issuing authority match against known authorities database, permit scope (geographic area, commodity type, volume limits), and document status (active/suspended/revoked/expired)
- [ ] F2.3: Scores document completeness per supply chain node (0-100) using weighted assessment: required documents present (40%), document validity confirmed (30%), document scope alignment (20%), document authenticity indicators (10%). Classification: COMPLIANT (>= 80), PARTIAL (50-79), NON-COMPLIANT (< 50)
- [ ] F2.4: Cross-references document details with country-specific compliance checker (F5): verifies that the set of documents presented covers all legal requirements for the commodity in the producing country
- [ ] F2.5: Detects document anomalies using rule-based checks: expired documents presented as valid, issuing authority mismatch (permit from wrong jurisdiction), duplicate permit numbers across different suppliers, volume limits exceeded (permit allows 1,000 m3 but supplier ships 2,000 m3), and geographic scope mismatch (permit for Province A but production in Province B)
- [ ] F2.6: Tracks document lifecycle: records document upload, verification, expiry, renewal, and replacement with full audit trail; generates expiry alerts at configurable lead times (default: 90, 60, 30 days before expiry)
- [ ] F2.7: Integrates with EUDR-012 (Document Authentication Agent) for document integrity verification: digital signature validation, hash-based tamper detection, and format compliance checking
- [ ] F2.8: Supports batch document verification: upload document packages per supplier or per shipment for bulk verification against country requirements
- [ ] F2.9: Generates document verification certificate with deterministic score, verification methodology, documents reviewed, compliance status per Article 2(40) category, and SHA-256 provenance hash
- [ ] F2.10: Maintains immutable document verification history per supplier: auditors can review all document submissions, verification results, and compliance status changes over time

**Document Completeness Scoring Formula:**
```
Document_Score = (
    documents_present_score * 0.40       # % of required documents submitted (0-100)
    + document_validity_score * 0.30     # % of submitted documents currently valid (0-100)
    + scope_alignment_score * 0.20       # % of documents with correct scope (geographic, commodity, volume) (0-100)
    + authenticity_score * 0.10          # % of documents passing integrity checks (0-100)
)

Classification:
  COMPLIANT:       Document_Score >= 80
  PARTIAL:         50 <= Document_Score < 80
  NON-COMPLIANT:   Document_Score < 50
```

**Non-Functional Requirements:**
- Accuracy: 98% correct validity determination (validated against manual legal review)
- Performance: < 2 seconds per document verification; batch 100 documents < 3 minutes
- Auditability: SHA-256 provenance hash on every verification; full calculation breakdown
- Document Types: 150+ recognized document types across 20+ countries

**Dependencies:**
- Feature 1 (Legal Framework Database) for per-country document requirements
- EUDR-012 Document Authentication Agent for integrity checks
- AGENT-DATA-001 PDF Invoice Extractor for document parsing
- AGENT-FOUND-005 Citations & Evidence Agent for source attribution

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 document processing specialist)

**Edge Cases:**
- Document in local language without English translation -> Accept; extract structured metadata (dates, numbers, issuing authority); flag for legal expert review if critical
- Document with expired validity but renewal application pending -> Mark as RENEWAL_PENDING; apply reduced compliance score; track renewal deadline
- Multiple versions of same permit (amendments) -> Track version history; verify latest version is current; retain all versions for audit
- Document from issuing authority not in known authorities database -> Flag as UNVERIFIED_AUTHORITY; require manual verification; accept provisionally with reduced authenticity score

---

#### Feature 3: Certification Scheme Validator

**User Story:**
```
As a compliance officer,
I want to verify that certification scheme certificates presented by my suppliers are valid and cover the relevant EUDR legal compliance requirements,
So that I can determine which Article 2(40) categories are addressed by certification and which require additional verification.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F3.1: Integrates with 6 major certification scheme databases: FSC (Forest Stewardship Council) certificate database, RSPO (Roundtable on Sustainable Palm Oil) membership and certificate database, PEFC (Programme for the Endorsement of Forest Certification) certificate database, Rainforest Alliance certification database, ISCC (International Sustainability & Carbon Certification) certificate database, and UTZ (now merged with Rainforest Alliance) legacy certificate database
- [ ] F3.2: Verifies certificate validity: checks certificate status (active/suspended/withdrawn/expired/terminated), certificate holder match against supplier, certificate scope (commodity, geographic area, supply chain model), validity period, and certifying body accreditation status
- [ ] F3.3: Maps each certification scheme to Article 2(40) categories with coverage assessment: for each scheme, documents which of the 8 legislation categories are FULLY covered (scheme standard addresses the category), PARTIALLY covered (scheme addresses some aspects), or NOT covered (category outside scheme scope)
- [ ] F3.4: Generates certification coverage gap analysis per supplier: identifies which Article 2(40) categories are covered by the supplier's certifications and which categories have gaps requiring additional evidence (legal documents, audit reports, or legal opinions)
- [ ] F3.5: Tracks chain-of-custody (CoC) certificate validity for supply chain intermediaries: verifies that all actors in the supply chain between the certified source and the EU importer hold valid CoC certificates in the correct supply chain model (identity preserved, segregated, mass balance, controlled sources)
- [ ] F3.6: Monitors certification scheme news for suspensions, withdrawals, and scope changes: when a supplier's certificate is suspended or withdrawn, the system immediately updates compliance status and generates an alert
- [ ] F3.7: Supports multi-certification scenarios: when a supplier holds multiple certifications (e.g., FSC + Rainforest Alliance), aggregates coverage across schemes to determine total Article 2(40) coverage
- [ ] F3.8: Validates certification body accreditation: verifies that the certifying body (e.g., SGS, Control Union, SCS Global) is accredited by the certification scheme (e.g., FSC-accredited, RSPO-approved) and that accreditation is current
- [ ] F3.9: Generates certification compliance report per supplier: certificate details, validity status, Article 2(40) coverage matrix, CoC chain verification, gap identification, and recommendations for additional compliance evidence
- [ ] F3.10: Maintains certification verification history with immutable audit trail: every certificate check recorded with verification date, result, certificate status at time of check, and SHA-256 provenance hash

**Certification-to-Article 2(40) Coverage Matrix:**
```
| Scheme              | Cat 1   | Cat 2   | Cat 3   | Cat 4   | Cat 5   | Cat 6   | Cat 7   | Cat 8   |
|                     | Land    | Enviro  | Forest  | 3rd Pty | Labour  | Human   | FPIC    | Tax/    |
|                     | Use     | Protect | Rules   | Rights  | Rights  | Rights  |         | Customs |
|---------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| FSC                 | FULL    | FULL    | FULL    | FULL    | FULL    | PARTIAL | FULL    | PARTIAL |
| RSPO                | FULL    | FULL    | PARTIAL | FULL    | FULL    | PARTIAL | FULL    | PARTIAL |
| PEFC                | FULL    | FULL    | FULL    | PARTIAL | PARTIAL | NONE    | PARTIAL | NONE    |
| Rainforest Alliance | PARTIAL | FULL    | FULL    | PARTIAL | FULL    | PARTIAL | PARTIAL | NONE    |
| ISCC                | PARTIAL | FULL    | PARTIAL | NONE    | PARTIAL | NONE    | NONE    | PARTIAL |
| UTZ (legacy)        | PARTIAL | PARTIAL | NONE    | PARTIAL | FULL    | PARTIAL | PARTIAL | NONE    |
```

**Non-Functional Requirements:**
- Coverage: 6 certification scheme databases integrated
- Performance: Certificate verification < 1 second per certificate; batch 500 certificates < 5 minutes
- Freshness: Certificate status refreshed daily from scheme databases
- Determinism: Coverage mapping is deterministic and auditable

**Dependencies:**
- FSC Certificate Database API (info.fsc.org)
- RSPO Certificate Database (rspo.org/members)
- PEFC Certificate Search (pefc.org/find-certified)
- Rainforest Alliance Certificate Search
- ISCC Certificate Database
- Feature 1 (Legal Framework Database) for Article 2(40) category definitions

**Estimated Effort:** 3 weeks (1 backend engineer, 1 integration engineer)

**Edge Cases:**
- Supplier's certificate recently expired but renewal in process -> Mark as RENEWAL_PENDING; apply reduced coverage; track renewal
- Certification scheme changes standard (e.g., FSC updates Principles and Criteria) -> Update coverage matrix; flag affected suppliers for re-assessment
- Supplier holds certificate in one supply chain model (e.g., mass balance) but claims identity preserved -> Flag scope mismatch; apply mass balance coverage level
- Certificate issued by non-accredited body -> Flag as INVALID_CERTIFIER; do not accept for compliance coverage; alert operator

---

#### Feature 4: Red Flag Detection Engine

**User Story:**
```
As a risk manager,
I want the system to automatically detect indicators of illegal commodity production across all 8 Article 2(40) categories,
So that I can identify high-risk supply chain nodes requiring enhanced due diligence and prevent illegally produced goods from reaching the EU market.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F4.1: Detects corruption risk red flags for permitting processes: country CPI score (from EUDR-019) below threshold (default: 40), permitting authority on known corruption risk list, permit issued unusually quickly (below country average processing time), and multiple permits from same issuing officer within short period
- [ ] F4.2: Detects illegal logging indicators: production from regions with known illegal logging prevalence (FAO/INTERPOL hotspot data), timber volume exceeding sustainable yield (annual allowable cut verification), species mismatch between harvesting permit and exported product, lack of timber legality verification (FLEGT/SVLK/DOF gap), and transport documentation gaps
- [ ] F4.3: Detects forced labour risk indicators using ILO forced labour indicator framework: presence of all 11 ILO indicators (abuse of vulnerability, deception, restriction of movement, isolation, physical/sexual violence, intimidation/threats, retention of identity documents, withholding of wages, debt bondage, abusive working/living conditions, excessive overtime); country-commodity combination on US DOL ILAB List of Goods Produced by Forced Labor or Child Labor
- [ ] F4.4: Detects tax evasion and financial irregularity indicators: transfer pricing between related entities at non-arm's-length values, commodity routing through tax haven jurisdictions, missing tax clearance certificates, significant discrepancy between declared production value and market value
- [ ] F4.5: Detects permit fraud indicators: duplicate permit numbers used by different suppliers, permits referencing non-existent or dissolved issuing authorities, permit dates preceding establishment of issuing authority, forged document format characteristics (detected via EUDR-012 authentication), and permits exceeding maximum allowable scope
- [ ] F4.6: Classifies red flag severity using weighted scoring: red flag type severity (30%), number of concurrent red flags (25%), country governance score (20%), and supplier compliance history (25%). Severity levels: CRITICAL (>= 80), HIGH (>= 60), MEDIUM (>= 40), LOW (< 40)
- [ ] F4.7: Generates supply chain impact assessment for each red flag: lists affected suppliers, plots, commodities, products, and estimated supply chain exposure (% of volume from flagged source)
- [ ] F4.8: Triggers enhanced due diligence requirements when red flag severity reaches HIGH or CRITICAL: requires operator acknowledgment and documented response within configurable SLA (default: 14 days for CRITICAL, 30 days for HIGH)
- [ ] F4.9: Maintains red flag history database with trend analysis: red flag frequency by country, region, flag type, and commodity; trend direction (improving/stable/deteriorating) calculated on 12-month rolling window
- [ ] F4.10: Implements red flag deduplication: multiple indicators pointing to the same underlying issue are consolidated into a single red flag with cross-references to all contributing indicators

**Red Flag Severity Scoring Formula:**
```
Red_Flag_Severity = (
    flag_type_severity * 0.30            # corruption=80, illegal_logging=100, forced_labour=100, tax_evasion=60, permit_fraud=90
    + concurrent_flag_count * 0.25       # 1 flag=20, 2=40, 3=60, 4=80, 5+=100
    + governance_gap * 0.20              # 100 - country governance score (from EUDR-016)
    + supplier_history_gap * 0.25        # 100 - supplier compliance history score (from EUDR-017)
)

Classification:
  CRITICAL:  Red_Flag_Severity >= 80
  HIGH:      60 <= Red_Flag_Severity < 80
  MEDIUM:    40 <= Red_Flag_Severity < 60
  LOW:       Red_Flag_Severity < 40
```

**Non-Functional Requirements:**
- Precision: >= 90% (validated against known violation cases)
- Coverage: All 8 Article 2(40) categories screened for red flags
- Determinism: Severity scoring is deterministic, bit-perfect reproducible
- Latency: Red flag assessment < 500ms per supply chain node

**Dependencies:**
- Feature 1 (Legal Framework Database) for country legal context
- Feature 2 (Document Verification) for permit anomaly data
- EUDR-016 Country Risk Evaluator for governance scoring
- EUDR-017 Supplier Risk Scorer for supplier history
- EUDR-019 Corruption Index Monitor for CPI and corruption data
- US DOL ILAB database for forced/child labour lists
- FAO/INTERPOL illegal logging hotspot data

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 data engineer)

**Edge Cases:**
- Red flag triggered by data error (e.g., incorrect CPI score) -> Allow operator to dispute with evidence; maintain disputed status until resolved; do not auto-dismiss
- Multiple red flags from different categories for same supplier -> Aggregate into compound red flag; apply escalated severity; require holistic response
- Red flag in country undergoing governance improvement (rising CPI) -> Use most recent CPI data; note trend direction in flag context; do not retroactively dismiss historical flags

---

#### Feature 5: Country-Specific Compliance Checker

**User Story:**
```
As a compliance officer,
I want the system to evaluate my suppliers' legal compliance against the specific legal requirements of each producing country for each EUDR commodity,
So that I can identify compliance gaps per legislation category and take targeted remediation action.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F5.1: Implements per-country compliance rule engines for 20+ producing countries, each containing the specific legal requirements for all 7 EUDR commodities applicable in that country across all 8 Article 2(40) categories
- [ ] F5.2: Evaluates compliance per Article 2(40) category per supplier-country combination: for each supplier in each producing country, assesses compliance status for all 8 categories as COMPLIANT (all requirements met with evidence), PARTIAL (some requirements met, gaps identified), NON-COMPLIANT (critical requirements not met), or UNABLE_TO_ASSESS (insufficient information to evaluate)
- [ ] F5.3: Generates compliance gap analysis per supplier: identifies specific missing documents, unmet legal requirements, and recommended remediation actions per Article 2(40) category, prioritized by regulatory risk and penalty severity
- [ ] F5.4: Calculates composite legal compliance score per supplier (0-100) using weighted assessment across all 8 categories: Cat 1 Land Use (15%), Cat 2 Environmental (15%), Cat 3 Forest Rules (15%), Cat 4 Third Party Rights (10%), Cat 5 Labour Rights (10%), Cat 6 Human Rights (10%), Cat 7 FPIC (10%), Cat 8 Tax/Customs (15%)
- [ ] F5.5: Supports country-specific compliance pathways: for example, Indonesia SVLK-certified timber receives automatic COMPLIANT status for Cat 3 (Forest Rules) and PARTIAL for Cat 1 (Land Use); Brazil CAR-registered properties receive PARTIAL for Cat 1 pending full compliance verification
- [ ] F5.6: Incorporates FLEGT license recognition: for VPA partner countries with operational FLEGT licensing (currently Indonesia), a valid FLEGT license provides strong evidence of Cat 3 compliance; the checker recognizes FLEGT as a compliance pathway
- [ ] F5.7: Generates per-country compliance dashboard: visual matrix showing compliance status across all 8 categories for all suppliers in a given country, highlighting critical gaps and compliance trends
- [ ] F5.8: Supports configurable compliance thresholds per category: operators can set minimum acceptable compliance levels per Article 2(40) category based on their risk appetite (default: 60 minimum for all categories)
- [ ] F5.9: Tracks compliance status changes over time per supplier: records when compliance improves (new documents submitted, audit findings closed) or deteriorates (permit expiry, new red flags, certification suspension)
- [ ] F5.10: Exports compliance evaluation results in structured format for DDS Article 4(2) risk assessment section, providing per-category compliance evidence and gap documentation

**Composite Legal Compliance Score:**
```
Legal_Compliance_Score = (
    cat1_land_use * 0.15
    + cat2_environmental * 0.15
    + cat3_forest_rules * 0.15
    + cat4_third_party_rights * 0.10
    + cat5_labour_rights * 0.10
    + cat6_human_rights * 0.10
    + cat7_fpic * 0.10
    + cat8_tax_customs * 0.15
)

Per-category score: COMPLIANT=100, PARTIAL=60, NON-COMPLIANT=20, UNABLE_TO_ASSESS=0

Classification:
  COMPLIANT:                Legal_Compliance_Score >= 80
  SUBSTANTIALLY_COMPLIANT:  60 <= Legal_Compliance_Score < 80
  PARTIALLY_COMPLIANT:      40 <= Legal_Compliance_Score < 60
  NON_COMPLIANT:            Legal_Compliance_Score < 40
```

**Non-Functional Requirements:**
- Coverage: 20+ countries with commodity-specific compliance rules for all 7 EUDR commodities
- Performance: Per-supplier compliance evaluation < 500ms; batch 1,000 suppliers < 10 minutes
- Determinism: Same supplier data always produces identical compliance score
- Auditability: SHA-256 provenance hash on every compliance evaluation

**Dependencies:**
- Feature 1 (Legal Framework Database) for per-country legal requirements
- Feature 2 (Document Verification) for document compliance evidence
- Feature 3 (Certification Validator) for certification coverage
- EUDR-016 Country Risk Evaluator for governance context
- EUDR-021 Indigenous Rights Checker for Cat 4/7 compliance input
- EUDR-022 Protected Area Validator for Cat 2 compliance input

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 legal data specialist)

**Edge Cases:**
- Supplier operates in country not yet in database (e.g., new sourcing country) -> Apply UNABLE_TO_ASSESS for all categories; trigger priority legal framework data collection; recommend enhanced due diligence
- Supplier has compliance evidence for some categories but UNABLE_TO_ASSESS for others -> Calculate partial score using assessed categories only; clearly flag unassessed categories; require operator acknowledgment
- Country changes legal requirements mid-assessment period -> Apply legislation effective at time of production; flag transitional period; offer re-assessment against new requirements

---

**P0 Features 6-9: Management, Reporting, and Integration Layer**

> Features 6, 7, 8, and 9 are P0 launch blockers. Without audit integration, legal opinion management, compliance reporting, and agent ecosystem integration, the core verification engine cannot deliver end-user value. These features are the delivery mechanism through which compliance officers, legal departments, auditors, and the broader EUDR platform interact with the verification engine.

---

#### Feature 6: Third-Party Audit Integration

**User Story:**
```
As a compliance officer,
I want to integrate third-party audit reports into the legal compliance assessment,
So that independent verification findings strengthen or qualify the compliance determination for my suppliers.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F6.1: Ingests structured audit reports from accredited verification bodies in standardized formats: PDF (parsed via AGENT-DATA-001), JSON (direct import), and CSV (tabular findings import). Extracts audit metadata: auditor, audit firm, accreditation, audit date, audit scope, and supplier audited
- [ ] F6.2: Classifies audit findings into 4 severity levels: MAJOR_NC (major non-conformity -- immediate compliance failure), MINOR_NC (minor non-conformity -- compliance at risk), OBSERVATION (area for improvement -- no compliance impact), POSITIVE (positive finding -- compliance strengthened)
- [ ] F6.3: Maps audit findings to Article 2(40) categories: each finding is tagged with the legislation category it addresses (e.g., finding about missing EIA -> Cat 2 Environmental Protection; finding about child labour -> Cat 5 Labour Rights)
- [ ] F6.4: Tracks corrective action plans (CAPs) with SLA deadlines: MAJOR_NC requires CAP within 30 days; MINOR_NC requires CAP within 90 days; system tracks CAP submission, verification, and closure with escalation at 50%, 75%, 100% of SLA
- [ ] F6.5: Integrates audit results into supplier compliance score: MAJOR_NC findings reduce the affected category score by 40 points; MINOR_NC findings reduce by 20 points; POSITIVE findings add 5 points (capped at 100)
- [ ] F6.6: Validates auditor accreditation: checks that the audit firm and lead auditor are accredited by the relevant certification scheme or recognized by the competent authority in the producing country
- [ ] F6.7: Supports audit report comparison across time periods: tracks compliance improvement or deterioration between successive audits for the same supplier
- [ ] F6.8: Generates audit summary dashboard per supplier: list of all audits, findings by severity and category, CAP status, trend analysis, and next scheduled audit date
- [ ] F6.9: Exports audit evidence packages for DDS submission: structured compilation of audit findings, CAP status, and compliance impact assessment per Article 2(40) category
- [ ] F6.10: Maintains immutable audit record history: all audit reports, findings, and CAP actions recorded with timestamps, actor attribution, and SHA-256 provenance hashes

**Non-Functional Requirements:**
- Ingestion: < 5 minutes per structured audit report (PDF parsing + finding extraction)
- Coverage: Support for audit reports from 20+ accredited verification bodies
- SLA Tracking: Minute-level precision for all CAP deadlines
- Retention: All audit records retained for minimum 5 years per EUDR Article 31

**Dependencies:**
- Feature 1 (Legal Framework Database) for Article 2(40) category mapping
- Feature 5 (Country Compliance Checker) for compliance score integration
- AGENT-DATA-001 PDF Invoice Extractor for PDF audit report parsing
- GL-EUDR-APP notification service for CAP deadline alerts

**Estimated Effort:** 3 weeks (1 backend engineer, 1 data processing engineer)

---

#### Feature 7: Legal Opinion Manager

**User Story:**
```
As a legal compliance officer,
I want a structured system to manage legal opinions required for complex compliance determinations,
So that I can track when legal opinions are needed, manage their lifecycle, and use them as evidence in compliance assessments.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F7.1: Identifies situations requiring legal opinions using rule-based triggers: conflicting legal requirements across jurisdictions, disputed land tenure status, novel commodity classification under national law, FPIC process challenges, regulatory interpretation uncertainty, and customs classification disputes
- [ ] F7.2: Tracks legal opinion lifecycle through 5 stages: IDENTIFIED (need detected), REQUESTED (opinion solicited from legal counsel), RECEIVED (opinion received and stored), VALID (opinion assessed as applicable and current), EXPIRED (opinion no longer current due to legal changes or time elapsed)
- [ ] F7.3: Stores legal opinions with structured metadata: opinion ID, issuing law firm/attorney, jurisdiction, legal question, opinion summary, applicable legislation, conclusion, confidence level (high/medium/low), validity period, and associated supply chain elements (country, commodity, supplier, plot)
- [ ] F7.4: Links legal opinions to specific compliance determinations: when a compliance evaluation relies on a legal opinion, the opinion is referenced with provenance in the compliance record
- [ ] F7.5: Tracks opinion validity: opinions have configurable validity periods (default: 24 months); system generates renewal alerts at configurable lead times (default: 90, 60, 30 days before expiry); expired opinions reduce compliance confidence
- [ ] F7.6: Supports multiple opinions on the same legal question: when different counsel provide different opinions, records all opinions; flags as CONFLICTING; applies most conservative interpretation for compliance purposes
- [ ] F7.7: Generates legal opinion requirement reports per supply chain: identifies all compliance determinations that require or would benefit from legal opinion, organized by country and Article 2(40) category
- [ ] F7.8: Implements access control for legal opinions: opinions may contain privileged legal advice; access restricted to users with explicit `eudr-lcv:legal-opinions:read` permission; all access logged in audit trail
- [ ] F7.9: Exports legal opinion inventory for audit: structured listing of all legal opinions with status, validity, associated compliance determinations, and provenance
- [ ] F7.10: Stores opinion documents with SHA-256 hash for integrity verification; original documents encrypted at rest (AES-256) via SEC-003

**Non-Functional Requirements:**
- Privacy: Legal opinions encrypted at rest (AES-256); access logged
- Lifecycle Tracking: All state transitions recorded with timestamps
- Retention: Legal opinions retained for minimum 5 years per EUDR Article 31
- Performance: Opinion lookup < 100ms; requirement analysis < 500ms

**Dependencies:**
- Feature 1 (Legal Framework Database) for trigger rule definitions
- Feature 5 (Country Compliance Checker) for compliance determination linkage
- SEC-003 Encryption at Rest for opinion document protection
- S3 for opinion document storage

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 8: Compliance Reporting Engine

**User Story:**
```
As a compliance officer,
I want automated, audit-ready legal compliance reports covering all 8 Article 2(40) categories,
So that I can include them in my Due Diligence Statement and present them to auditors, certifiers, and regulators.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F8.1: Generates comprehensive legal compliance reports in PDF, JSON, and HTML formats, including per-category compliance assessment across all 8 Article 2(40) categories, document verification results, certification coverage, red flag analysis, audit findings, legal opinion status, and compliance risk assessment
- [ ] F8.2: Generates DDS-integrated legal compliance section: structured risk assessment narrative for EUDR Article 4(2)/9(1)(h) submission, covering legal compliance verification methodology, per-category results, gap analysis, and mitigation measures
- [ ] F8.3: Produces certification scheme compliance evidence reports: mapping of certification coverage to Article 2(40) categories, gap identification, and additional evidence requirements, formatted per scheme-specific audit checklists (FSC Principle 1, RSPO P&C 2.1)
- [ ] F8.4: Generates supplier-level legal compliance scorecards: per-supplier summary across all 8 categories with compliance score, document completeness, certification coverage, red flag count, audit finding status, and overall legal compliance rating
- [ ] F8.5: Produces per-commodity legal compliance reports: for each EUDR commodity, aggregated compliance view across all sourcing countries with country-specific legal requirement matrix, compliance rate, and risk distribution
- [ ] F8.6: Creates executive summary reports with key indicators: total suppliers assessed, compliance score distribution by category, critical gaps count, red flag distribution, certification coverage rate, and overall legal compliance readiness score
- [ ] F8.7: Exports dashboard data packages for BI integration (Grafana, Tableau, Power BI) in CSV, JSON, and XLSX formats with standardized schema for legal compliance metrics
- [ ] F8.8: Includes complete audit trail in reports: every compliance score, document verification, and red flag assessment linked to source data, calculation method, legal reference, and SHA-256 provenance hash
- [ ] F8.9: Supports multi-language report generation in English (EN), French (FR), German (DE), Spanish (ES), and Portuguese (PT) using translated templates with jurisdiction-appropriate legal terminology
- [ ] F8.10: Maintains report generation history with version control: reports are immutable once generated; updated assessments produce new report versions, not overwrites

**Non-Functional Requirements:**
- Performance: PDF generation < 10 seconds per supply chain compliance report
- Quality: Reports pass WCAG 2.1 AA accessibility standards
- Size: PDF reports optimized to < 10 MB each
- Formats: PDF (audit), JSON (API integration), HTML (web display), CSV/XLSX (BI export)

**Dependencies:**
- Features 1-7 for all legal compliance assessment data
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
I want the Legal Compliance Verifier to enrich existing EUDR agents with legal compliance data,
So that legal compliance status is embedded throughout the supply chain compliance workflow.
```

**Acceptance Criteria (10 sub-requirements):**
- [ ] F9.1: Integrates with EUDR-001 (Supply Chain Mapping Master): enriches supply chain graph nodes with legal compliance risk attributes (compliance score per Article 2(40) category, red flag count, document completeness, certification coverage) for risk propagation through the supply chain
- [ ] F9.2: Integrates with EUDR-002 (Geolocation Verification Agent): links plot coordinates to country-specific legal requirements; flags plots in jurisdictions with identified compliance gaps or transitional legal frameworks
- [ ] F9.3: Integrates with EUDR-008 (Multi-Tier Supplier Tracker): propagates legal compliance status through multi-tier supply chains; identifies tiers with compliance gaps; generates per-tier compliance heat maps
- [ ] F9.4: Integrates with EUDR-012 (Document Authentication Agent): sends documents for integrity verification; receives authentication results for incorporation into document verification scores
- [ ] F9.5: Integrates with EUDR-016 (Country Risk Evaluator): provides per-country legal compliance scores across all 8 Article 2(40) categories that feed the governance and legal framework index; receives country risk classification for enforcement and governance scoring context
- [ ] F9.6: Integrates with EUDR-017 (Supplier Risk Scorer): provides per-supplier legal compliance scores; suppliers with unresolved compliance gaps, active red flags, or major audit non-conformities receive elevated risk scores
- [ ] F9.7: Integrates with EUDR-019 (Corruption Index Monitor): receives corruption perception index data and corruption event feeds for red flag detection engine (F4); provides legal compliance findings that may indicate corruption
- [ ] F9.8: Integrates with EUDR-021 (Indigenous Rights Checker): receives FPIC compliance status and indigenous rights data for Article 2(40) Cat 4 (Third Party Rights) and Cat 7 (FPIC); provides legal framework context for FPIC requirements per country
- [ ] F9.9: Integrates with EUDR-022 (Protected Area Validator): receives protected area compliance status for Article 2(40) Cat 2 (Environmental Protection) and Cat 3 (Forest-Related Rules); provides legal framework context for protected area legislation per country
- [ ] F9.10: Publishes legal compliance events to the GreenLang event bus for consumption by other agents: legal_compliance_evaluated, document_verified, red_flag_detected, certification_validated, audit_finding_recorded, compliance_status_changed, legal_opinion_received, compliance_report_generated. Maintains integration health monitoring via Prometheus

**Non-Functional Requirements:**
- Latency: Integration API responses < 200ms p95
- Reliability: 99.9% message delivery for event bus publications
- Compatibility: RESTful API with OpenAPI 3.0 specification
- Versioning: API versioned (v1) with backward compatibility guarantee

**Dependencies:**
- AGENT-EUDR-001 Supply Chain Mapping Master (BUILT 100%)
- AGENT-EUDR-002 Geolocation Verification Agent (BUILT 100%)
- AGENT-EUDR-008 Multi-Tier Supplier Tracker (BUILT 100%)
- AGENT-EUDR-012 Document Authentication Agent (BUILT 100%)
- AGENT-EUDR-016 Country Risk Evaluator (BUILT 100%)
- AGENT-EUDR-017 Supplier Risk Scorer (BUILT 100%)
- AGENT-EUDR-019 Corruption Index Monitor (BUILT 100%)
- AGENT-EUDR-021 Indigenous Rights Checker (BUILT 100%)
- AGENT-EUDR-022 Protected Area Validator (BUILT 100%)
- GL-EUDR-APP v1.0 Platform (BUILT 100%)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Legal Compliance Analytics Dashboard
- Interactive heat map showing legal compliance by country and Article 2(40) category
- Drill-down from country to supplier to individual compliance gaps
- Trend analysis showing compliance improvement/deterioration over quarters
- Certification coverage visualization with gap highlighting

#### Feature 11: Legislative Change Monitoring
- Automated monitoring of legislative amendments in producing countries
- Impact assessment of legal changes on existing compliance determinations
- Proactive alerts to operators when legislation changes affect their supply chains
- Legal framework version comparison (before/after amendment analysis)

#### Feature 12: Compliance Benchmarking
- Anonymous cross-industry compliance benchmarking per country and commodity
- Best-practice sharing for high-performing compliance programs
- Gap analysis against industry average compliance scores
- Regulatory expectation alignment scoring

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Legal advisory or dispute resolution services (agent provides data; not legal counsel)
- Automated translation of full national legislation texts (structured metadata only)
- Direct permit application or renewal processing (agent verifies; does not apply)
- Real-time court database monitoring for legal proceedings (defer to Phase 2)
- Mobile native application for field document collection (web responsive only)
- Predictive ML models for legal risk forecasting (defer to Phase 2)
- Blockchain-based legal compliance ledger (SHA-256 provenance hashes sufficient for v1.0)
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
| AGENT-EUDR-023        |           | AGENT-EUDR-001            |           | AGENT-EUDR-016        |
| Legal Compliance      |<--------->| Supply Chain Mapping      |<--------->| Country Risk          |
| Verifier              |           | Master                    |           | Evaluator             |
|                       |           |                           |           |                       |
| - Legal Database (F1) |           | - Graph Engine            |           | - Risk Scoring Engine |
| - Doc Verifier (F2)   |           | - Risk Propagation        |           | - Governance Engine   |
| - Cert Validator (F3) |           | - Gap Analysis            |           | - Enforcement Score   |
| - Red Flag Engine(F4) |           |                           |           |                       |
| - Country Checker(F5) |           +---------------------------+           +-----------------------+
| - Audit Integr. (F6)  |
| - Legal Opinions (F7) |           +---------------------------+           +---------------------------+
| - Reporting (F8)      |           | AGENT-EUDR-012            |           | AGENT-EUDR-019            |
| - Integration (F9)    |           | Document Authentication   |           | Corruption Index          |
+-----------+-----------+           |                           |           | Monitor                   |
            |                       | - Digital Signatures      |           | - CPI Tracking            |
            |                       | - Hash Verification       |           | - Corruption Alerts       |
+-----------v-----------+           +---------------------------+           +---------------------------+
| Legal Compliance      |
| Data Sources          |           +---------------------------+           +---------------------------+
|                       |           | AGENT-EUDR-021            |           | AGENT-EUDR-022            |
| - FAO FAOLEX          |           | Indigenous Rights         |           | Protected Area            |
| - ILO NATLEX/NORMLEX  |           | Checker                   |           | Validator                 |
| - National Gazettes   |           |                           |           |                           |
| - Cert. Scheme DBs    |           | - FPIC Status (Cat 4/7)   |           | - PA Compliance (Cat 2/3) |
| - US DOL ILAB         |           +---------------------------+           +---------------------------+
| - INTERPOL Data       |
+-----------------------+           +---------------------------+
                                    | AGENT-EUDR-008/017        |
                                    | Multi-Tier Supplier /     |
                                    | Supplier Risk Scorer      |
                                    +---------------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/legal_compliance_verifier/
    __init__.py                              # Public API exports
    config.py                                # LegalComplianceVerifierConfig with GL_EUDR_LCV_ env prefix
    models.py                                # Pydantic v2 models for legal compliance data
    legal_framework_database.py              # LegalFrameworkDatabaseEngine: legislation data management (F1)
    document_verifier.py                     # DocumentVerificationEngine: permit/license verification (F2)
    certification_validator.py               # CertificationSchemeValidator: cert scheme integration (F3)
    red_flag_detector.py                     # RedFlagDetectionEngine: illegal production indicators (F4)
    country_compliance_checker.py            # CountryComplianceChecker: per-country legal evaluation (F5)
    audit_integrator.py                      # AuditIntegrationEngine: third-party audit management (F6)
    legal_opinion_manager.py                 # LegalOpinionManager: legal opinion lifecycle (F7)
    compliance_reporter.py                   # ComplianceReporter: report generation (F8)
    agent_integrator.py                      # AgentIntegrator: cross-agent integration (F9)
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains
    metrics.py                               # 20 Prometheus self-monitoring metrics
    setup.py                                 # LegalComplianceVerifierService facade (singleton, 9 engines)
    reference_data/
        __init__.py
        legal_frameworks.py                  # Per-country legislation database (20+ countries)
        document_requirements.py             # Per-country document requirement matrices
        certification_coverage.py            # Certification-to-Article 2(40) coverage mapping
        red_flag_rules.py                    # Red flag detection rule definitions
        ilo_conventions.py                   # ILO convention ratification database
        issuing_authorities.py               # Known issuing authority registry per country
        forced_labour_lists.py               # US DOL ILAB forced/child labour lists
    api/
        __init__.py
        router.py                            # FastAPI router (~35 endpoints)
        schemas.py                           # API request/response Pydantic schemas
        dependencies.py                      # FastAPI dependency injection
        legal_framework_routes.py            # Legal framework database endpoints
        document_routes.py                   # Document verification endpoints
        certification_routes.py              # Certification validation endpoints
        red_flag_routes.py                   # Red flag detection endpoints
        compliance_routes.py                 # Country compliance checker endpoints
        audit_routes.py                      # Audit integration endpoints
        opinion_routes.py                    # Legal opinion management endpoints
        report_routes.py                     # Compliance reporting endpoints
        integration_routes.py                # Agent integration endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Legislation Coverage Status
class LegislationCoverage(str, Enum):
    FULL = "full"                    # Legislation exists and is enforced
    PARTIAL = "partial"              # Legislation exists, enforcement weak
    GAP = "gap"                      # No applicable legislation
    CONFLICTING = "conflicting"      # Multiple conflicting provisions
    TRANSITIONAL = "transitional"    # Legal framework in transition

# Document Compliance Status
class DocumentComplianceStatus(str, Enum):
    COMPLIANT = "compliant"          # Score >= 80
    PARTIAL = "partial"              # Score 50-79
    NON_COMPLIANT = "non_compliant"  # Score < 50

# Certificate Status
class CertificateStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    RENEWAL_PENDING = "renewal_pending"

# Red Flag Type
class RedFlagType(str, Enum):
    CORRUPTION = "corruption"
    ILLEGAL_LOGGING = "illegal_logging"
    FORCED_LABOUR = "forced_labour"
    CHILD_LABOUR = "child_labour"
    TAX_EVASION = "tax_evasion"
    PERMIT_FRAUD = "permit_fraud"
    HUMAN_RIGHTS_VIOLATION = "human_rights_violation"
    ENVIRONMENTAL_VIOLATION = "environmental_violation"

# Legal Compliance Classification
class ComplianceClassification(str, Enum):
    COMPLIANT = "compliant"                              # Score >= 80
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"  # Score 60-79
    PARTIALLY_COMPLIANT = "partially_compliant"          # Score 40-59
    NON_COMPLIANT = "non_compliant"                      # Score < 40

# Legal Opinion Stage
class LegalOpinionStage(str, Enum):
    IDENTIFIED = "identified"
    REQUESTED = "requested"
    RECEIVED = "received"
    VALID = "valid"
    EXPIRED = "expired"

# Audit Finding Severity
class AuditFindingSeverity(str, Enum):
    MAJOR_NC = "major_nc"
    MINOR_NC = "minor_nc"
    OBSERVATION = "observation"
    POSITIVE = "positive"

# Article 2(40) Category
class Article240Category(str, Enum):
    LAND_USE = "cat1_land_use"
    ENVIRONMENTAL = "cat2_environmental"
    FOREST_RULES = "cat3_forest_rules"
    THIRD_PARTY_RIGHTS = "cat4_third_party_rights"
    LABOUR_RIGHTS = "cat5_labour_rights"
    HUMAN_RIGHTS = "cat6_human_rights"
    FPIC = "cat7_fpic"
    TAX_CUSTOMS = "cat8_tax_customs"

# Country Legal Framework
class CountryLegalFramework(BaseModel):
    framework_id: str                        # UUID
    country_code: str                        # ISO 3166-1 alpha-2
    country_name: str
    category: Article240Category
    statute_name: str
    statute_number: Optional[str]
    enactment_date: Optional[date]
    last_amendment_date: Optional[date]
    enforcing_authority: str
    applicable_commodities: List[str]        # EUDR commodity codes
    required_documents: List[str]            # Document type codes
    penalty_provisions: Optional[str]
    obligations_summary: str                 # English summary
    coverage_status: LegislationCoverage
    enforceability_score: Decimal            # 0-100
    source_url: Optional[str]
    source_language: str
    provenance_hash: str
    version: int
    created_at: datetime
    updated_at: datetime

# Document Verification Result
class DocumentVerification(BaseModel):
    verification_id: str
    document_id: str
    supplier_id: str
    document_type: str                       # Permit type code
    country_code: str
    article_240_category: Article240Category
    issuing_authority: str
    document_number: Optional[str]
    issue_date: Optional[date]
    expiry_date: Optional[date]
    is_valid: bool
    is_expired: bool
    scope_aligned: bool
    authority_verified: bool
    document_score: Decimal                  # 0-100
    compliance_status: DocumentComplianceStatus
    anomalies_detected: List[str]
    provenance_hash: str
    verified_at: datetime

# Certification Verification
class CertificationVerification(BaseModel):
    verification_id: str
    supplier_id: str
    scheme: str                              # FSC/RSPO/PEFC/RA/ISCC/UTZ
    certificate_number: str
    certificate_status: CertificateStatus
    certificate_holder: str
    scope_commodity: str
    scope_geography: str
    supply_chain_model: str                  # IP/Segregated/MB/Controlled
    valid_from: Optional[date]
    valid_to: Optional[date]
    certifying_body: str
    certifier_accredited: bool
    article_240_coverage: Dict[str, str]     # category -> FULL/PARTIAL/NONE
    coverage_gaps: List[str]                 # Uncovered categories
    provenance_hash: str
    verified_at: datetime

# Red Flag Alert
class RedFlagAlert(BaseModel):
    alert_id: str
    supplier_id: str
    country_code: str
    flag_type: RedFlagType
    article_240_category: Article240Category
    severity_score: Decimal                  # 0-100
    severity_level: str                      # CRITICAL/HIGH/MEDIUM/LOW
    description: str
    indicators: List[Dict]                   # Contributing indicators
    supply_chain_impact: Optional[Dict]
    enhanced_dd_required: bool
    enhanced_dd_deadline: Optional[datetime]
    status: str                              # active/investigating/mitigated/resolved/disputed
    deduplication_group: Optional[str]
    provenance_hash: str
    detected_at: datetime

# Supplier Legal Compliance Evaluation
class LegalComplianceEvaluation(BaseModel):
    evaluation_id: str
    supplier_id: str
    country_code: str
    commodity: str
    cat1_score: Decimal
    cat2_score: Decimal
    cat3_score: Decimal
    cat4_score: Decimal
    cat5_score: Decimal
    cat6_score: Decimal
    cat7_score: Decimal
    cat8_score: Decimal
    composite_score: Decimal                 # 0-100
    classification: ComplianceClassification
    gap_analysis: Dict[str, List[str]]       # category -> [missing requirements]
    document_completeness: Decimal
    certification_coverage: Decimal
    red_flag_count: int
    audit_finding_impact: Decimal
    legal_opinion_count: int
    provenance_hash: str
    evaluated_at: datetime
    version: int
```

### 7.4 Database Schema (New Migration: V111)

```sql
-- =========================================================================
-- V111: AGENT-EUDR-023 Legal Compliance Verifier Schema
-- Agent: GL-EUDR-LCV-023
-- Tables: 15 (4 hypertables)
-- Prefix: gl_eudr_lcv_
-- =========================================================================

CREATE SCHEMA IF NOT EXISTS eudr_legal_compliance_verifier;

-- 1. Country Legal Frameworks
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_legal_frameworks (
    framework_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(200) NOT NULL,
    category VARCHAR(30) NOT NULL CHECK (category IN ('cat1_land_use', 'cat2_environmental', 'cat3_forest_rules', 'cat4_third_party_rights', 'cat5_labour_rights', 'cat6_human_rights', 'cat7_fpic', 'cat8_tax_customs')),
    statute_name VARCHAR(500) NOT NULL,
    statute_number VARCHAR(200),
    enactment_date DATE,
    last_amendment_date DATE,
    enforcing_authority VARCHAR(500) NOT NULL,
    applicable_commodities JSONB DEFAULT '[]',
    required_documents JSONB DEFAULT '[]',
    penalty_provisions TEXT,
    obligations_summary TEXT NOT NULL,
    coverage_status VARCHAR(20) NOT NULL CHECK (coverage_status IN ('full', 'partial', 'gap', 'conflicting', 'transitional')),
    enforceability_score NUMERIC(5,2) NOT NULL CHECK (enforceability_score >= 0 AND enforceability_score <= 100),
    source_url VARCHAR(2000),
    source_language VARCHAR(50) NOT NULL DEFAULT 'en',
    provenance_hash VARCHAR(64) NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_frameworks_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_legal_frameworks(country_code);
CREATE INDEX idx_lcv_frameworks_category ON eudr_legal_compliance_verifier.gl_eudr_lcv_legal_frameworks(category);
CREATE INDEX idx_lcv_frameworks_coverage ON eudr_legal_compliance_verifier.gl_eudr_lcv_legal_frameworks(coverage_status);

-- 2. Document Verification Records (hypertable on verified_at)
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_document_verifications (
    verification_id UUID DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    supplier_id UUID NOT NULL,
    document_type VARCHAR(100) NOT NULL,
    country_code CHAR(2) NOT NULL,
    article_240_category VARCHAR(30) NOT NULL,
    issuing_authority VARCHAR(500),
    document_number VARCHAR(200),
    issue_date DATE,
    expiry_date DATE,
    is_valid BOOLEAN NOT NULL,
    is_expired BOOLEAN NOT NULL DEFAULT FALSE,
    scope_aligned BOOLEAN NOT NULL DEFAULT TRUE,
    authority_verified BOOLEAN NOT NULL DEFAULT FALSE,
    document_score NUMERIC(5,2) NOT NULL CHECK (document_score >= 0 AND document_score <= 100),
    compliance_status VARCHAR(20) NOT NULL CHECK (compliance_status IN ('compliant', 'partial', 'non_compliant')),
    anomalies_detected JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (verification_id, verified_at)
);

SELECT create_hypertable('eudr_legal_compliance_verifier.gl_eudr_lcv_document_verifications', 'verified_at');

CREATE INDEX idx_lcv_docs_supplier ON eudr_legal_compliance_verifier.gl_eudr_lcv_document_verifications(supplier_id);
CREATE INDEX idx_lcv_docs_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_document_verifications(country_code);
CREATE INDEX idx_lcv_docs_type ON eudr_legal_compliance_verifier.gl_eudr_lcv_document_verifications(document_type);
CREATE INDEX idx_lcv_docs_status ON eudr_legal_compliance_verifier.gl_eudr_lcv_document_verifications(compliance_status);

-- 3. Certification Verifications
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_certification_verifications (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    scheme VARCHAR(50) NOT NULL,
    certificate_number VARCHAR(200) NOT NULL,
    certificate_status VARCHAR(20) NOT NULL CHECK (certificate_status IN ('active', 'suspended', 'withdrawn', 'expired', 'terminated', 'renewal_pending')),
    certificate_holder VARCHAR(500),
    scope_commodity VARCHAR(200),
    scope_geography VARCHAR(500),
    supply_chain_model VARCHAR(30),
    valid_from DATE,
    valid_to DATE,
    certifying_body VARCHAR(500),
    certifier_accredited BOOLEAN DEFAULT FALSE,
    article_240_coverage JSONB NOT NULL DEFAULT '{}',
    coverage_gaps JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_certs_supplier ON eudr_legal_compliance_verifier.gl_eudr_lcv_certification_verifications(supplier_id);
CREATE INDEX idx_lcv_certs_scheme ON eudr_legal_compliance_verifier.gl_eudr_lcv_certification_verifications(scheme);
CREATE INDEX idx_lcv_certs_status ON eudr_legal_compliance_verifier.gl_eudr_lcv_certification_verifications(certificate_status);

-- 4. Red Flag Alerts (hypertable on detected_at)
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_red_flags (
    alert_id UUID DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    country_code CHAR(2) NOT NULL,
    flag_type VARCHAR(50) NOT NULL CHECK (flag_type IN ('corruption', 'illegal_logging', 'forced_labour', 'child_labour', 'tax_evasion', 'permit_fraud', 'human_rights_violation', 'environmental_violation')),
    article_240_category VARCHAR(30) NOT NULL,
    severity_score NUMERIC(5,2) NOT NULL CHECK (severity_score >= 0 AND severity_score <= 100),
    severity_level VARCHAR(20) NOT NULL CHECK (severity_level IN ('critical', 'high', 'medium', 'low')),
    description TEXT NOT NULL,
    indicators JSONB NOT NULL DEFAULT '[]',
    supply_chain_impact JSONB DEFAULT '{}',
    enhanced_dd_required BOOLEAN DEFAULT FALSE,
    enhanced_dd_deadline TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'investigating', 'mitigated', 'resolved', 'disputed')),
    deduplication_group VARCHAR(100),
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (alert_id, detected_at)
);

SELECT create_hypertable('eudr_legal_compliance_verifier.gl_eudr_lcv_red_flags', 'detected_at');

CREATE INDEX idx_lcv_flags_supplier ON eudr_legal_compliance_verifier.gl_eudr_lcv_red_flags(supplier_id);
CREATE INDEX idx_lcv_flags_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_red_flags(country_code);
CREATE INDEX idx_lcv_flags_type ON eudr_legal_compliance_verifier.gl_eudr_lcv_red_flags(flag_type);
CREATE INDEX idx_lcv_flags_severity ON eudr_legal_compliance_verifier.gl_eudr_lcv_red_flags(severity_level);
CREATE INDEX idx_lcv_flags_status ON eudr_legal_compliance_verifier.gl_eudr_lcv_red_flags(status);

-- 5. Supplier Compliance Evaluations (hypertable on evaluated_at)
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_evaluations (
    evaluation_id UUID DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    country_code CHAR(2) NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    cat1_score NUMERIC(5,2) DEFAULT 0,
    cat2_score NUMERIC(5,2) DEFAULT 0,
    cat3_score NUMERIC(5,2) DEFAULT 0,
    cat4_score NUMERIC(5,2) DEFAULT 0,
    cat5_score NUMERIC(5,2) DEFAULT 0,
    cat6_score NUMERIC(5,2) DEFAULT 0,
    cat7_score NUMERIC(5,2) DEFAULT 0,
    cat8_score NUMERIC(5,2) DEFAULT 0,
    composite_score NUMERIC(5,2) NOT NULL CHECK (composite_score >= 0 AND composite_score <= 100),
    classification VARCHAR(30) NOT NULL CHECK (classification IN ('compliant', 'substantially_compliant', 'partially_compliant', 'non_compliant')),
    gap_analysis JSONB DEFAULT '{}',
    document_completeness NUMERIC(5,2) DEFAULT 0,
    certification_coverage NUMERIC(5,2) DEFAULT 0,
    red_flag_count INTEGER DEFAULT 0,
    audit_finding_impact NUMERIC(5,2) DEFAULT 0,
    legal_opinion_count INTEGER DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    version INTEGER DEFAULT 1,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (evaluation_id, evaluated_at)
);

SELECT create_hypertable('eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_evaluations', 'evaluated_at');

CREATE INDEX idx_lcv_evals_supplier ON eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_evaluations(supplier_id);
CREATE INDEX idx_lcv_evals_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_evaluations(country_code);
CREATE INDEX idx_lcv_evals_commodity ON eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_evaluations(commodity);
CREATE INDEX idx_lcv_evals_classification ON eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_evaluations(classification);

-- 6. Audit Reports
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_audit_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    audit_firm VARCHAR(500) NOT NULL,
    lead_auditor VARCHAR(300),
    accreditation VARCHAR(200),
    audit_date DATE NOT NULL,
    audit_scope TEXT,
    report_format VARCHAR(10) NOT NULL CHECK (report_format IN ('pdf', 'json', 'csv')),
    file_path VARCHAR(1000),
    file_hash VARCHAR(64),
    findings_count INTEGER DEFAULT 0,
    major_nc_count INTEGER DEFAULT 0,
    minor_nc_count INTEGER DEFAULT 0,
    observation_count INTEGER DEFAULT 0,
    positive_count INTEGER DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_audit_reports_supplier ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_reports(supplier_id);

-- 7. Audit Findings
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_audit_findings (
    finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES eudr_legal_compliance_verifier.gl_eudr_lcv_audit_reports(report_id),
    supplier_id UUID NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('major_nc', 'minor_nc', 'observation', 'positive')),
    article_240_category VARCHAR(30) NOT NULL,
    description TEXT NOT NULL,
    evidence TEXT,
    cap_required BOOLEAN DEFAULT FALSE,
    cap_deadline TIMESTAMPTZ,
    cap_status VARCHAR(20) DEFAULT 'pending' CHECK (cap_status IN ('pending', 'submitted', 'verified', 'closed', 'overdue')),
    cap_description TEXT,
    cap_submitted_at TIMESTAMPTZ,
    cap_verified_at TIMESTAMPTZ,
    cap_closed_at TIMESTAMPTZ,
    score_impact NUMERIC(5,2) DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_findings_report ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_findings(report_id);
CREATE INDEX idx_lcv_findings_supplier ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_findings(supplier_id);
CREATE INDEX idx_lcv_findings_severity ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_findings(severity);
CREATE INDEX idx_lcv_findings_cap ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_findings(cap_status) WHERE cap_required = TRUE;

-- 8. Legal Opinions
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_legal_opinions (
    opinion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issuing_firm VARCHAR(500) NOT NULL,
    issuing_attorney VARCHAR(300),
    jurisdiction VARCHAR(200) NOT NULL,
    legal_question TEXT NOT NULL,
    opinion_summary TEXT NOT NULL,
    applicable_legislation JSONB DEFAULT '[]',
    conclusion TEXT NOT NULL,
    confidence_level VARCHAR(10) NOT NULL CHECK (confidence_level IN ('high', 'medium', 'low')),
    stage VARCHAR(20) NOT NULL CHECK (stage IN ('identified', 'requested', 'received', 'valid', 'expired')),
    validity_start DATE,
    validity_end DATE,
    associated_country CHAR(2),
    associated_commodity VARCHAR(50),
    associated_suppliers JSONB DEFAULT '[]',
    document_path VARCHAR(1000),
    document_hash VARCHAR(64),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_opinions_stage ON eudr_legal_compliance_verifier.gl_eudr_lcv_legal_opinions(stage);
CREATE INDEX idx_lcv_opinions_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_legal_opinions(associated_country);
CREATE INDEX idx_lcv_opinions_validity ON eudr_legal_compliance_verifier.gl_eudr_lcv_legal_opinions(validity_end);

-- 9. Document Requirement Matrices
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_document_requirements (
    requirement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    article_240_category VARCHAR(30) NOT NULL,
    document_type VARCHAR(100) NOT NULL,
    document_name VARCHAR(500) NOT NULL,
    issuing_authority VARCHAR(500) NOT NULL,
    is_mandatory BOOLEAN DEFAULT TRUE,
    validity_period_months INTEGER,
    renewal_required BOOLEAN DEFAULT TRUE,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_doc_reqs_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_document_requirements(country_code, commodity);

-- 10. ILO Convention Ratifications
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_ilo_ratifications (
    ratification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    convention_number VARCHAR(10) NOT NULL,
    convention_name VARCHAR(500) NOT NULL,
    ratification_date DATE,
    is_ratified BOOLEAN NOT NULL DEFAULT FALSE,
    implementation_status VARCHAR(20) CHECK (implementation_status IN ('full', 'partial', 'non_compliant', 'not_ratified')),
    latest_ceacr_observation TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code, convention_number)
);

CREATE INDEX idx_lcv_ilo_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_ilo_ratifications(country_code);

-- 11. Issuing Authorities Registry
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_issuing_authorities (
    authority_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    authority_name VARCHAR(500) NOT NULL,
    authority_type VARCHAR(100) NOT NULL,
    jurisdiction_level VARCHAR(20) CHECK (jurisdiction_level IN ('national', 'provincial', 'district', 'local')),
    document_types_issued JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    established_date DATE,
    dissolved_date DATE,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_authorities_country ON eudr_legal_compliance_verifier.gl_eudr_lcv_issuing_authorities(country_code);

-- 12. Country Legal Compliance Scores
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_country_scores (
    score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    cat1_framework_score NUMERIC(5,2) DEFAULT 0,
    cat2_framework_score NUMERIC(5,2) DEFAULT 0,
    cat3_framework_score NUMERIC(5,2) DEFAULT 0,
    cat4_framework_score NUMERIC(5,2) DEFAULT 0,
    cat5_framework_score NUMERIC(5,2) DEFAULT 0,
    cat6_framework_score NUMERIC(5,2) DEFAULT 0,
    cat7_framework_score NUMERIC(5,2) DEFAULT 0,
    cat8_framework_score NUMERIC(5,2) DEFAULT 0,
    composite_legal_framework_score NUMERIC(5,2) NOT NULL,
    enforceability_score NUMERIC(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'standard', 'high')),
    data_sources JSONB NOT NULL DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (country_code)
);

CREATE INDEX idx_lcv_country_scores_level ON eudr_legal_compliance_verifier.gl_eudr_lcv_country_scores(risk_level);

-- 13. Compliance Reports
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_reports (
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

CREATE INDEX idx_lcv_reports_type ON eudr_legal_compliance_verifier.gl_eudr_lcv_compliance_reports(report_type);

-- 14. Legal Framework Version History
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_framework_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework_id UUID NOT NULL REFERENCES eudr_legal_compliance_verifier.gl_eudr_lcv_legal_frameworks(framework_id),
    version_number INTEGER NOT NULL,
    change_type VARCHAR(30) NOT NULL CHECK (change_type IN ('enacted', 'amended', 'repealed', 'superseded', 'corrected')),
    change_description TEXT NOT NULL,
    previous_text_summary TEXT,
    new_text_summary TEXT,
    effective_date DATE NOT NULL,
    source_url VARCHAR(2000),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_lcv_framework_versions ON eudr_legal_compliance_verifier.gl_eudr_lcv_framework_versions(framework_id);

-- 15. Immutable Audit Log (hypertable)
CREATE TABLE eudr_legal_compliance_verifier.gl_eudr_lcv_audit_log (
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

SELECT create_hypertable('eudr_legal_compliance_verifier.gl_eudr_lcv_audit_log', 'created_at');

CREATE INDEX idx_lcv_audit_entity ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_log(entity_type, entity_id);
CREATE INDEX idx_lcv_audit_actor ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_log(actor);
CREATE INDEX idx_lcv_audit_action ON eudr_legal_compliance_verifier.gl_eudr_lcv_audit_log(action);
```

### 7.5 API Endpoints (~35)

| Method | Path | Description |
|--------|------|-------------|
| **Legal Framework Database** | | |
| GET | `/v1/eudr-lcv/frameworks` | List legal frameworks (with filters: country, category, commodity, coverage) |
| GET | `/v1/eudr-lcv/frameworks/{framework_id}` | Get legal framework details |
| POST | `/v1/eudr-lcv/frameworks/search` | Search legal frameworks by statute, authority, or keyword |
| GET | `/v1/eudr-lcv/frameworks/checklist/{country_code}/{commodity}` | Get compliance checklist for country-commodity |
| GET | `/v1/eudr-lcv/frameworks/{framework_id}/history` | Get framework version history |
| **Document Verification** | | |
| POST | `/v1/eudr-lcv/documents/verify` | Verify a single document |
| POST | `/v1/eudr-lcv/documents/batch` | Batch document verification |
| GET | `/v1/eudr-lcv/documents/{verification_id}` | Get document verification result |
| GET | `/v1/eudr-lcv/documents/supplier/{supplier_id}` | Get all document verifications for a supplier |
| GET | `/v1/eudr-lcv/documents/requirements/{country_code}/{commodity}` | Get document requirements for country-commodity |
| **Certification Validation** | | |
| POST | `/v1/eudr-lcv/certifications/validate` | Validate a certification certificate |
| GET | `/v1/eudr-lcv/certifications/{verification_id}` | Get certification verification result |
| GET | `/v1/eudr-lcv/certifications/supplier/{supplier_id}` | Get all certification verifications for a supplier |
| GET | `/v1/eudr-lcv/certifications/coverage-matrix` | Get certification-to-Article 2(40) coverage matrix |
| **Red Flag Detection** | | |
| POST | `/v1/eudr-lcv/red-flags/scan` | Scan a supplier for red flags |
| GET | `/v1/eudr-lcv/red-flags` | List red flag alerts (with filters: country, type, severity, status) |
| GET | `/v1/eudr-lcv/red-flags/{alert_id}` | Get red flag alert details |
| POST | `/v1/eudr-lcv/red-flags/{alert_id}/acknowledge` | Acknowledge enhanced DD requirement |
| GET | `/v1/eudr-lcv/red-flags/trends` | Get red flag trend analysis |
| **Country Compliance** | | |
| POST | `/v1/eudr-lcv/compliance/evaluate` | Evaluate supplier legal compliance |
| POST | `/v1/eudr-lcv/compliance/batch` | Batch compliance evaluation for multiple suppliers |
| GET | `/v1/eudr-lcv/compliance/{evaluation_id}` | Get compliance evaluation result |
| GET | `/v1/eudr-lcv/compliance/supplier/{supplier_id}` | Get latest compliance evaluation for a supplier |
| GET | `/v1/eudr-lcv/compliance/dashboard/{country_code}` | Get per-country compliance dashboard data |
| **Audit Integration** | | |
| POST | `/v1/eudr-lcv/audits/ingest` | Ingest audit report |
| GET | `/v1/eudr-lcv/audits/{report_id}` | Get audit report details and findings |
| GET | `/v1/eudr-lcv/audits/supplier/{supplier_id}` | Get all audits for a supplier |
| POST | `/v1/eudr-lcv/audits/findings/{finding_id}/cap` | Update CAP status |
| **Legal Opinions** | | |
| POST | `/v1/eudr-lcv/opinions` | Create legal opinion record |
| GET | `/v1/eudr-lcv/opinions/{opinion_id}` | Get legal opinion details |
| PUT | `/v1/eudr-lcv/opinions/{opinion_id}` | Update legal opinion stage/status |
| GET | `/v1/eudr-lcv/opinions/requirements` | Get legal opinion requirements analysis |
| **Reporting** | | |
| POST | `/v1/eudr-lcv/reports/generate` | Generate legal compliance report |
| GET | `/v1/eudr-lcv/reports/{report_id}` | Get report metadata |
| GET | `/v1/eudr-lcv/reports/{report_id}/download` | Download report file |
| GET | `/v1/eudr-lcv/reports` | List generated reports |
| **Country Scores** | | |
| GET | `/v1/eudr-lcv/countries/{country_code}` | Get country legal compliance score |
| GET | `/v1/eudr-lcv/countries` | List all country legal compliance scores |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (20)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_lcv_framework_queries_total` | Counter | Legal framework database queries by type |
| 2 | `gl_eudr_lcv_document_verifications_total` | Counter | Document verifications performed by type |
| 3 | `gl_eudr_lcv_certification_validations_total` | Counter | Certification validations by scheme |
| 4 | `gl_eudr_lcv_red_flags_detected_total` | Counter | Red flags detected by type and severity |
| 5 | `gl_eudr_lcv_compliance_evaluations_total` | Counter | Compliance evaluations performed |
| 6 | `gl_eudr_lcv_audit_reports_ingested_total` | Counter | Audit reports ingested by format |
| 7 | `gl_eudr_lcv_audit_findings_recorded_total` | Counter | Audit findings recorded by severity |
| 8 | `gl_eudr_lcv_legal_opinions_created_total` | Counter | Legal opinions created by stage |
| 9 | `gl_eudr_lcv_reports_generated_total` | Counter | Compliance reports generated by type |
| 10 | `gl_eudr_lcv_cap_sla_breaches_total` | Counter | Corrective action plan SLA breaches |
| 11 | `gl_eudr_lcv_document_verify_duration_seconds` | Histogram | Document verification latency |
| 12 | `gl_eudr_lcv_compliance_eval_duration_seconds` | Histogram | Compliance evaluation processing time |
| 13 | `gl_eudr_lcv_red_flag_scan_duration_seconds` | Histogram | Red flag scanning latency |
| 14 | `gl_eudr_lcv_report_generation_duration_seconds` | Histogram | Report generation time |
| 15 | `gl_eudr_lcv_api_errors_total` | Counter | API errors by endpoint and status code |
| 16 | `gl_eudr_lcv_active_frameworks` | Gauge | Total legal framework records in database |
| 17 | `gl_eudr_lcv_active_red_flags` | Gauge | Currently active red flag alerts |
| 18 | `gl_eudr_lcv_pending_caps` | Gauge | Pending corrective action plans |
| 19 | `gl_eudr_lcv_expiring_documents` | Gauge | Documents expiring within 90 days |
| 20 | `gl_eudr_lcv_expiring_opinions` | Gauge | Legal opinions expiring within 90 days |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables for evaluations/red flags/audit |
| Cache | Redis | Legal framework caching, compliance evaluation caching, certification status caching |
| Object Storage | S3 | Generated reports, audit report storage, legal opinion documents |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Arithmetic | Python Decimal | Precision for compliance scores and risk percentages |
| PDF Generation | WeasyPrint | HTML-to-PDF for compliance report generation |
| Templates | Jinja2 | Multi-format report templates (HTML/PDF) |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based legal compliance data access control |
| Encryption | AES-256 via SEC-003 | Legal opinion document encryption at rest |
| Monitoring | Prometheus + Grafana | 20 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-lcv:frameworks:read` | View legal framework data | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:frameworks:write` | Update legal framework database | Legal Officer, Admin |
| `eudr-lcv:documents:read` | View document verification results | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:documents:verify` | Trigger document verification | Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:certifications:read` | View certification verifications | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:certifications:validate` | Trigger certification validation | Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:red-flags:read` | View red flag alerts | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:red-flags:manage` | Manage red flag lifecycle (acknowledge, dispute) | Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:compliance:read` | View compliance evaluations | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:compliance:evaluate` | Trigger compliance evaluation | Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:audits:read` | View audit reports and findings | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:audits:ingest` | Ingest audit reports | Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:audits:manage-cap` | Manage corrective action plans | Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:legal-opinions:read` | View legal opinions (privileged) | Legal Officer, Admin |
| `eudr-lcv:legal-opinions:manage` | Create and manage legal opinions | Legal Officer, Admin |
| `eudr-lcv:reports:read` | View generated reports | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:reports:generate` | Generate compliance reports | Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:reports:download` | Download report files | Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:countries:read` | View country legal compliance scores | Viewer, Analyst, Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:audit-log:read` | View audit trail | Auditor (read-only), Compliance Officer, Legal Officer, Admin |
| `eudr-lcv:config:manage` | Manage configuration (scoring weights, thresholds, SLA timelines) | Admin |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Legal Compliance Evaluation (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Legal Compliance" module -> "Supplier Assessment" tab
3. Selects supplier (e.g., PT Sawit Lestari, Indonesia) and commodity (Palm Oil)
4. Clicks "Evaluate Legal Compliance"
5. System evaluates across all 8 Article 2(40) categories:
   - Cat 1 Land Use: COMPLIANT (HGU land title verified, spatial plan aligned)
   - Cat 2 Environmental: PARTIAL (AMDAL EIA on file but expired 6 months ago)
   - Cat 3 Forest Rules: COMPLIANT (SVLK certificate active, IPPKH permit valid)
   - Cat 4 Third Party Rights: PARTIAL (Indigenous territory adjacent, FPIC in progress -- from EUDR-021)
   - Cat 5 Labour Rights: COMPLIANT (ILO conventions ratified, no forced labour flags)
   - Cat 6 Human Rights: COMPLIANT (No human rights violations reported)
   - Cat 7 FPIC: PARTIAL (FPIC workflow at CONSULTATION stage -- from EUDR-021)
   - Cat 8 Tax/Customs: COMPLIANT (PNBP royalties paid, export permit valid)
   - Composite Score: 76 (SUBSTANTIALLY COMPLIANT)
6. System highlights gaps:
   - EXPIRED: AMDAL Environmental Impact Assessment (expired 2025-09-15)
   - IN PROGRESS: FPIC consultation with Dayak Iban community
7. Officer clicks "Generate Gap Report" -> PDF with remediation recommendations
8. Officer sends AMDAL renewal request to supplier via platform notification
```

#### Flow 2: Red Flag Investigation (Risk Manager)

```
1. System detects red flag for Supplier CM-WOOD-047 (Cameroon, Timber):
   - Type: PERMIT_FRAUD (severity: HIGH)
   - Indicators: Forestry concession permit number duplicated with another supplier;
     permit volume (50,000 m3) exceeds national annual allowable cut for concession area
2. Risk manager receives alert notification
3. Opens red flag detail view:
   - Supplier: CM-WOOD-047 (Eastern Province, Cameroon)
   - Red flag type: Permit fraud + Illegal logging (compound)
   - Contributing indicators:
     a) Duplicate permit #CMR-FC-2024-0891 also held by CM-WOOD-112
     b) Permit volume 50,000 m3 exceeds AAC of 32,000 m3 for concession #RC-088
     c) Country CPI: 26 (high corruption risk)
   - Severity: CRITICAL (compound flags + low governance)
   - Supply chain impact: 12% of Cameroon timber volume
4. System triggers enhanced due diligence with 14-day SLA
5. Risk manager acknowledges and initiates investigation
6. Risk manager requests third-party field audit for concession RC-088
7. System tracks investigation with SLA countdown
```

#### Flow 3: Certification Coverage Analysis (Legal Officer)

```
1. Legal officer opens "Certification Mapping" view
2. Selects all palm oil suppliers in Indonesia
3. System displays certification coverage across Article 2(40):
   - 85% of suppliers hold RSPO certification (covers Cat 1, 2, 4, 5, 7 FULLY)
   - Coverage gaps for RSPO-certified: Cat 3 PARTIAL, Cat 6 PARTIAL, Cat 8 NONE
   - 15% of suppliers uncertified: ALL categories require document-based evidence
4. Officer clicks "Gap Analysis" for supplier with RSPO only
   - Cat 3 (Forest Rules): RSPO covers plantation management but not natural forest;
     need SVLK or IPPKH for forest-related compliance
   - Cat 8 (Tax/Customs): RSPO does not cover tax compliance;
     need PNBP receipt and export permit verification
5. Officer generates "Certification Coverage Report" for audit preparation
6. Report maps each supplier's certification to Article 2(40) with gap identification
```

### 8.2 Key Screen Descriptions

**Legal Compliance Dashboard:**
- 8-column compliance matrix: rows = suppliers, columns = Article 2(40) categories, cells color-coded (green = COMPLIANT, yellow = PARTIAL, red = NON-COMPLIANT, grey = UNABLE_TO_ASSESS)
- Left sidebar: filter panel (country, commodity, classification, red flag status)
- Top bar: summary statistics (total suppliers assessed, compliance distribution, critical gaps count)
- Right sidebar: selected supplier detail panel with composite score and gap list
- Bottom panel: legal framework data version and freshness indicator

**Red Flag Alert Feed:**
- Timeline view: red flag alerts sorted by date with severity badges (color-coded by severity)
- Map view: red flag locations with supply chain overlay
- Compound flag indicator: multiple flag types for same supplier (stacked flag icons)
- Detail panel: indicators list, supply chain impact, enhanced DD SLA countdown, investigation status
- Action buttons: Acknowledge, Investigate, Dispute, Record Mitigation

**Certification Coverage Matrix:**
- Grid view: rows = certification schemes, columns = Article 2(40) categories, cells = FULL/PARTIAL/NONE
- Per-supplier view: selected supplier's certifications overlaid on coverage matrix with gap highlighting
- CoC chain visualization: supply chain path with certification status at each node
- Gap report: list of uncovered categories with recommended evidence for each gap

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Legal Framework Database -- 20+ countries, 8 categories, version tracking, enforceability scoring
  - [ ] Feature 2: Document Verification Engine -- 150+ document types, validity checking, anomaly detection, batch processing
  - [ ] Feature 3: Certification Scheme Validator -- 6 schemes integrated, Article 2(40) coverage mapping, CoC verification
  - [ ] Feature 4: Red Flag Detection Engine -- 8 flag types, severity scoring, supply chain impact, enhanced DD workflow
  - [ ] Feature 5: Country-Specific Compliance Checker -- 20+ countries, 7 commodities, 8-category scoring, gap analysis
  - [ ] Feature 6: Third-Party Audit Integration -- report ingestion, finding classification, CAP tracking, score integration
  - [ ] Feature 7: Legal Opinion Manager -- lifecycle tracking, validity monitoring, access control, provenance
  - [ ] Feature 8: Compliance Reporting -- 8 report types, 5 formats, 5 languages, DDS integration
  - [ ] Feature 9: Agent Integration -- EUDR-001/002/008/012/016/017/019/021/022 bidirectional integration verified
- [ ] >= 85% test coverage achieved (750+ tests)
- [ ] Security audit passed (JWT + RBAC integrated, 21 permissions)
- [ ] Performance targets met (< 2s document verification p99, < 500ms compliance check p99, < 10s report generation)
- [ ] Legal framework database validated against legal expert review (>= 95% accuracy for initial 20 countries)
- [ ] Compliance scoring verified deterministic (bit-perfect reproducibility)
- [ ] Document verification validated against 200+ known document-validity pairs
- [ ] API documentation complete (OpenAPI spec, ~35 endpoints)
- [ ] Database migration V111 tested and validated (15 tables, 4 hypertables)
- [ ] Integration with all 9 dependent EUDR agents verified
- [ ] 5 beta customers successfully evaluating their supply chains
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ supply chains evaluated for legal compliance
- 500+ document verifications completed
- 200+ certification certificates validated
- Average compliance evaluation latency < 500ms (p99)
- Legal framework database freshness within 30 days
- < 5 support tickets per customer

**60 Days:**
- 200+ supply chains actively monitored for legal compliance
- 2,000+ documents verified
- 50+ red flags detected and triaged
- 20+ audit reports ingested with findings tracked
- Compliance reports generated for 3+ certification scheme audits
- NPS > 45 from legal compliance officer persona

**90 Days:**
- 500+ supply chains actively monitored
- 10,000+ documents verified across all 20+ countries
- Zero EUDR penalties attributable to legal compliance gaps for active customers
- Full integration with GL-EUDR-APP DDS workflow operational
- Legal compliance scores feeding EUDR-016 country risk for all producing countries
- NPS > 55

---

## 10. Timeline and Milestones

### Phase 1: Core Verification Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Legal Framework Database (Feature 1): 20+ country profiles, 8-category structure, enforceability scoring, version tracking | Senior Backend Engineer + Legal Data Specialist |
| 2-3 | Document Verification Engine (Feature 2): 150+ document types, validity checking, anomaly detection, batch processing | Senior Backend Engineer + Document Processing Specialist |
| 3-4 | Certification Scheme Validator (Feature 3): 6 scheme integrations, coverage matrix, CoC verification | Backend Engineer + Integration Engineer |
| 4-5 | Red Flag Detection Engine (Feature 4): 8 flag types, severity scoring, supply chain impact, deduplication | Senior Backend Engineer + Data Engineer |
| 5-6 | Country-Specific Compliance Checker (Feature 5): 20+ country rule engines, composite scoring, gap analysis | Senior Backend Engineer + Legal Data Specialist |

**Milestone: Core verification engine operational with 5 core features (Week 6)**

### Phase 2: Management, API, and Reporting (Weeks 7-11)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Third-Party Audit Integration (Feature 6): report ingestion, finding classification, CAP tracking | Backend Engineer + Data Processing Engineer |
| 8-9 | Legal Opinion Manager (Feature 7): lifecycle tracking, access control, provenance | Backend Engineer |
| 9-10 | REST API Layer: ~35 endpoints, authentication, rate limiting | Backend Engineer |
| 10-11 | Compliance Reporting (Feature 8): 8 report types, 5 formats, 5 languages, DDS section | Backend + Template Engineer |

**Milestone: Full API operational with audit integration, legal opinions, and compliance reporting (Week 11)**

### Phase 3: Integration, RBAC, and Observability (Weeks 12-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 12 | Agent Integration (Feature 9): EUDR-001/002/008/012/016/017/019/021/022 bidirectional integration | Senior Backend Engineer |
| 12-13 | RBAC integration (21 permissions), AES-256 encryption for legal opinions, Prometheus metrics (20) | Backend + DevOps |
| 13-14 | Grafana dashboard, OpenTelemetry tracing, event bus integration, end-to-end integration testing | DevOps + Backend |

**Milestone: All 9 P0 features implemented with full integration and observability (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 750+ tests, golden tests for all 7 commodities, legal framework validation | Test Engineer |
| 16-17 | Performance testing, security audit (legal opinion encryption), load testing (batch evaluations) | DevOps + Security |
| 17 | Database migration V111 finalized and tested (15 tables, 4 hypertables) | DevOps |
| 17-18 | Beta customer onboarding (5 customers) | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Legal compliance analytics dashboard (Feature 10)
- Legislative change monitoring (Feature 11)
- Compliance benchmarking (Feature 12)
- Additional country legal frameworks (30+ countries)
- CSDDD Article 7 alignment and reporting

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable; graph enrichment API defined |
| AGENT-EUDR-002 Geolocation Verification Agent | BUILT (100%) | Low | Plot coordinates validated and available |
| AGENT-EUDR-008 Multi-Tier Supplier Tracker | BUILT (100%) | Low | Multi-tier supply chain data available |
| AGENT-EUDR-012 Document Authentication Agent | BUILT (100%) | Low | Document integrity verification API defined |
| AGENT-EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Governance and enforcement scores available |
| AGENT-EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Supplier risk enrichment API defined |
| AGENT-EUDR-019 Corruption Index Monitor | BUILT (100%) | Low | CPI and corruption event feeds available |
| AGENT-EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | FPIC and indigenous rights compliance data available (Cat 4/7) |
| AGENT-EUDR-022 Protected Area Validator | BUILT (100%) | Low | Protected area compliance data available (Cat 2/3) |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration target available |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | AES-256 for legal opinion documents |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| FAO FAOLEX (Legal Database) | Available (open data) | Low | Data download with offline cache; periodic refresh |
| ILO NATLEX/NORMLEX | Available (open data) | Low | Convention ratification data; static with infrequent updates |
| FSC Certificate Database | Available (API) | Medium | API rate limits; local cache with daily sync |
| RSPO Certificate Database | Available (web scrape/API) | Medium | Data format may change; adapter pattern |
| PEFC Certificate Search | Available (web) | Medium | No formal API; adapter pattern with web extraction |
| Rainforest Alliance Database | Available (partnership) | Medium | Establish data partnership; manual import as fallback |
| US DOL ILAB Lists | Available (open data) | Low | Static data; updated annually |
| National official gazettes (20+ countries) | Variable availability | High | Multi-source approach; legal data specialist for manual curation; prioritize major producing countries |
| Transparency International CPI | Available (annual) | Low | Sourced via EUDR-019; annual refresh |
| INTERPOL illegal logging data | Limited availability | Medium | Supplement with FAO, EIA, and academic research data |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Legal framework data is incomplete or inaccurate for some countries, particularly smaller producing countries with limited online legal databases | High | High | Prioritize 8 major producing countries for initial launch; engage legal data specialists for manual curation; confidence scoring per country; continuous improvement program |
| R2 | National legislation is amended frequently, creating data freshness challenges | Medium | High | 30-day update SLA; automated gazette monitoring for major countries; legal data specialist for ongoing maintenance; version-controlled framework database |
| R3 | Certification scheme APIs change or become unavailable, breaking integration | Medium | Medium | Adapter pattern isolates scheme integration; local certificate cache; manual import fallback; multi-scheme redundancy |
| R4 | Red flag detection produces false positives, causing alert fatigue | High | Medium | Conservative initial thresholds; operator feedback loop for threshold tuning; deduplication; severity-based prioritization; precision target >= 90% |
| R5 | Country-specific compliance rules are complex and require legal expertise to define correctly | High | High | Engage local legal experts for rule validation; start with well-documented countries (Brazil, Indonesia); peer review of rule engines; iterative refinement |
| R6 | Legal opinions contain privileged information creating liability risk if improperly disclosed | Medium | High | AES-256 encryption; RBAC-controlled access (Legal Officer + Admin only); access logging; privacy impact assessment |
| R7 | Audit report formats vary significantly across verification bodies, complicating ingestion | High | Medium | Standardized ingestion templates; JSON direct import for structured reports; PDF parser with configurable extraction rules; manual import fallback |
| R8 | Forced labour and child labour risk assessment may trigger reputational sensitivity | Medium | Medium | Clear disclaimer that agent provides risk indicators, not accusations; graduated severity levels; operator controls for response workflow |
| R9 | Integration complexity with 9 upstream EUDR agents | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern; integration health monitoring; retry logic |
| R10 | Customers may resist comprehensive legal compliance assessment (seen as excessive burden) | Medium | Medium | Demonstrate regulatory penalty risk (4% turnover); show CSDDD dual-use value; start with critical categories; provide ROI analysis |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Legal Framework Database Tests | 80+ | Data integrity, 8-category coverage, version control, enforceability scoring, search, country coverage |
| Document Verification Tests | 80+ | 150+ document types, validity checking, anomaly detection, batch processing, scoring formula, edge cases |
| Certification Validation Tests | 60+ | 6 schemes, certificate status, coverage mapping, CoC verification, multi-certification, accreditation |
| Red Flag Detection Tests | 70+ | 8 flag types, severity scoring, compound flags, deduplication, supply chain impact, trend analysis |
| Country Compliance Tests | 80+ | 20+ country rule engines, 8-category scoring, gap analysis, composite score, FLEGT recognition |
| Audit Integration Tests | 40+ | Report ingestion (PDF/JSON/CSV), finding classification, CAP tracking, score integration, audit comparison |
| Legal Opinion Tests | 30+ | Lifecycle tracking, validity monitoring, access control, encryption, requirement analysis |
| Compliance Reporting Tests | 40+ | All 8 report types, 5 formats, 5 languages, template rendering, provenance |
| Agent Integration Tests | 50+ | Cross-agent data flow with 9 EUDR agents, event bus, webhook, API contracts |
| API Tests | 50+ | All ~35 endpoints, auth, error handling, pagination, rate limiting |
| Golden Tests | 50 | 7 commodities x 7 scenarios (see 13.2) + 1 multi-country scenario |
| Performance Tests | 25+ | Document verification, compliance evaluation, batch processing, report generation, concurrent queries |
| Determinism Tests | 15+ | Bit-perfect reproducibility for compliance scoring, red flag severity, document scores |
| Security Tests | 10+ | Legal opinion encryption, RBAC enforcement, audit trail integrity |
| **Total** | **750+** | |

### 13.2 Golden Test Scenarios

Each of the 7 EUDR commodities will have 7 golden test scenarios:

1. **Fully compliant supplier** -- All 8 categories COMPLIANT, valid documents, active certification -> expect COMPLIANT classification (score >= 80)
2. **Document gaps** -- Missing critical documents (e.g., no EIA, no forestry license) -> expect NON_COMPLIANT for affected categories, composite score < 40
3. **Certification coverage** -- Supplier with FSC only -> expect FULL coverage for Cat 1/2/3/4/5/7, PARTIAL for Cat 6, gap for Cat 8
4. **Red flag detection** -- Supplier with corruption indicators + permit anomalies -> expect compound red flag, CRITICAL severity, enhanced DD triggered
5. **Country with legal gap** -- Supplier in country with GAP in labour rights legislation -> expect UNABLE_TO_ASSESS for Cat 5, enhanced DD recommended
6. **Audit integration** -- Supplier with major NC finding + closed CAP -> expect initial score reduction, then restoration after CAP closure
7. **Multi-certification** -- Supplier with RSPO + Rainforest Alliance -> expect aggregated coverage across schemes; identify remaining gaps

Total: 7 commodities x 7 scenarios = 49 golden test scenarios (+ 1 multi-country supply chain scenario = 50)

### 13.3 Determinism Tests

Every scoring and calculation engine will include determinism tests that:
1. Run the same calculation 100 times with identical inputs
2. Verify bit-perfect identical outputs (SHA-256 hash match)
3. Test across Python versions (3.11, 3.12) to ensure no platform-dependent behavior
4. Verify Decimal arithmetic produces identical results to reference calculations
5. Verify compliance scoring is deterministic regardless of feature evaluation order
6. Verify red flag severity is deterministic with identical indicator inputs
7. Verify document completeness scoring is deterministic with identical document sets

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **Article 2(40)** | EUDR article defining "relevant legislation" as 8 categories of law applicable in the country of production |
| **FLEGT** | Forest Law Enforcement, Governance and Trade -- EU initiative to combat illegal logging through VPAs |
| **VPA** | Voluntary Partnership Agreement -- bilateral agreement between EU and timber-producing country under FLEGT |
| **SVLK** | Sistem Verifikasi Legalitas Kayu -- Indonesian Timber Legality Verification System |
| **DOF** | Documento de Origem Florestal -- Brazilian Document of Forest Origin for timber tracking |
| **CAR** | Cadastro Ambiental Rural -- Brazilian Rural Environmental Registry |
| **AMDAL** | Analisis Mengenai Dampak Lingkungan -- Indonesian Environmental Impact Assessment |
| **HGU** | Hak Guna Usaha -- Indonesian Right to Cultivate (land title for plantations) |
| **IPPKH** | Izin Pinjam Pakai Kawasan Hutan -- Indonesian Forest Area Borrowing-Use Permit |
| **PNBP** | Penerimaan Negara Bukan Pajak -- Indonesian Non-Tax State Revenue (royalty payments) |
| **STLHK** | Sertifikat Legalitas Hasil Hutan Kayu -- Indonesian Timber Legality Certificate |
| **ILO** | International Labour Organization |
| **UDHR** | Universal Declaration of Human Rights (1948) |
| **UNGP** | UN Guiding Principles on Business and Human Rights (2011) |
| **ICCPR** | International Covenant on Civil and Political Rights |
| **ICESCR** | International Covenant on Economic, Social and Cultural Rights |
| **UNDRIP** | United Nations Declaration on the Rights of Indigenous Peoples (2007) |
| **FPIC** | Free, Prior and Informed Consent |
| **CPI** | Corruption Perceptions Index (Transparency International) |
| **CITES** | Convention on International Trade in Endangered Species |
| **CSDDD** | Corporate Sustainability Due Diligence Directive (EU, 2024) |
| **FSC** | Forest Stewardship Council |
| **RSPO** | Roundtable on Sustainable Palm Oil |
| **PEFC** | Programme for the Endorsement of Forest Certification |
| **ISCC** | International Sustainability & Carbon Certification |
| **UTZ** | UTZ Certified (now merged with Rainforest Alliance) |
| **CoC** | Chain of Custody -- tracking certified material through the supply chain |
| **NC** | Non-Conformity -- audit finding of non-compliance with standard |
| **CAP** | Corrective Action Plan -- documented plan to address audit findings |
| **FAOLEX** | FAO legislative database for national environmental and natural resource legislation |
| **NATLEX** | ILO database of national labour, social security, and related human rights legislation |
| **NORMLEX** | ILO Information System on International Labour Standards |
| **ILAB** | Bureau of International Labor Affairs (US Department of Labor) |

### Appendix B: Article 2(40) Detailed Category Breakdown

| Category | Sub-Categories | Key Compliance Questions | Typical Evidence |
|----------|---------------|------------------------|------------------|
| **Cat 1: Land Use** | Land title validity, concession legality, spatial planning alignment, tenure registration | Does the operator hold legal title/concession? Is the land use consistent with spatial plans? | Land title certificate, concession agreement, spatial plan extract |
| **Cat 2: Environmental** | EIA completion, environmental permits, pollution control, water rights | Has an EIA been conducted and approved? Are environmental permits current? | EIA approval, environmental permit, water use license |
| **Cat 3: Forest Rules** | Forestry license, AAC compliance, FMP approval, reforestation obligations | Is the forestry concession legally granted? Does harvesting comply with AAC? | Forestry license, FMP, timber transport documents (DOF/SVLK) |
| **Cat 4: Third Party** | Indigenous rights, community land rights, consultation obligations | Have third-party land rights been assessed? Has consultation occurred? | FPIC documentation, consultation records, benefit-sharing agreements |
| **Cat 5: Labour** | Child labour prohibition, forced labour prohibition, minimum wage, OHS | Are ILO core conventions respected? Is there evidence of child/forced labour risk? | Labour inspection certificates, ILO compliance records, worker contracts |
| **Cat 6: Human Rights** | UDHR compliance, freedom from violence, right to security | Are human rights respected in production? Are there reported violations? | Human rights impact assessments, UNGP self-assessment, violation screening |
| **Cat 7: FPIC** | FPIC process, consent documentation, UNDRIP alignment | Has FPIC been obtained where required? Is consent documentation complete? | FPIC certificates, consultation records, consent agreements |
| **Cat 8: Tax/Customs** | Corporate tax, royalties, export permits, anti-bribery, CITES | Are taxes and royalties paid? Are export permits valid? Are CITES permits obtained? | Tax clearance certificates, royalty receipts, export permits, CITES permits |

### Appendix C: Integration API Contracts

**Provided to EUDR-001 (Supply Chain Mapping Master):**
```python
# Legal compliance risk data for graph node enrichment
def get_legal_compliance_risk(supplier_id: str) -> Dict:
    """Returns: {
        supplier_id: str,
        country_code: str,
        composite_score: Decimal,          # 0-100
        classification: str,               # compliant/substantially/partially/non_compliant
        per_category_scores: Dict[str, Decimal],  # cat1..cat8 -> score
        document_completeness: Decimal,    # 0-100
        certification_coverage: Decimal,   # 0-100
        active_red_flags: int,
        active_major_ncs: int,
        legal_compliance_risk_level: str,  # critical/high/medium/low
        provenance_hash: str
    }"""
```

**Provided to EUDR-016 (Country Risk Evaluator):**
```python
# Country legal framework score for governance index
def get_country_legal_framework_score(country_code: str) -> Dict:
    """Returns: {
        country_code: str,
        per_category_framework_scores: Dict[str, Decimal],  # cat1..cat8 -> 0-100
        composite_legal_framework_score: Decimal,  # 0-100
        enforceability_score: Decimal,     # 0-100
        risk_level: str,                   # low/standard/high
        ilo_ratification_count: int,       # of 8 core conventions
        provenance_hash: str
    }"""
```

**Provided to EUDR-017 (Supplier Risk Scorer):**
```python
# Supplier legal compliance risk for composite supplier scoring
def get_supplier_legal_risk(supplier_id: str) -> Dict:
    """Returns: {
        supplier_id: str,
        composite_score: Decimal,          # 0-100
        classification: str,
        non_compliant_categories: List[str],
        active_red_flags: int,
        active_major_ncs: int,
        document_completeness: Decimal,
        certification_coverage: Decimal,
        legal_compliance_risk_score: Decimal,  # 0-100
        provenance_hash: str
    }"""
```

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. European Commission EUDR Guidance Document -- Risk Assessment and Due Diligence
3. EUDR Article 2(40) -- Definition of "relevant legislation"
4. ILO Declaration on Fundamental Principles and Rights at Work (1998, amended 2022)
5. ILO Conventions C029, C087, C098, C100, C105, C111, C138, C182
6. Universal Declaration of Human Rights (UDHR, 1948)
7. UN Guiding Principles on Business and Human Rights (2011)
8. International Covenant on Civil and Political Rights (ICCPR)
9. International Covenant on Economic, Social and Cultural Rights (ICESCR)
10. United Nations Declaration on the Rights of Indigenous Peoples (UNDRIP, 2007)
11. Directive (EU) 2024/1760 -- Corporate Sustainability Due Diligence Directive (CSDDD)
12. Regulation (EU) 2020/852 -- EU Taxonomy Regulation
13. EU FLEGT Regulation 2173/2005
14. CITES Convention (1975) -- Text and Appendices
15. OECD Due Diligence Guidance for Responsible Supply Chains (2018)
16. Transparency International Corruption Perceptions Index Methodology
17. World Justice Project Rule of Law Index
18. FAO FAOLEX Database -- Technical Documentation
19. ILO NATLEX/NORMLEX -- Technical Documentation
20. FSC Principles and Criteria (FSC-STD-01-001 V5-3)
21. RSPO Principles and Criteria (2018)
22. PEFC Sustainable Forest Management Standard (PEFC ST 1003:2018)
23. Rainforest Alliance Sustainable Agriculture Standard (2020)
24. ISCC EU/PLUS Certification Requirements
25. US Department of Labor -- List of Goods Produced by Child Labor or Forced Labor
26. Brazil Forest Code (Lei 12.651/2012)
27. Indonesia Forestry Law 41/1999 and SVLK Regulation
28. DRC Forest Code (Lei 011/2002)
29. ISO 3166-1 -- Country Codes Standard

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-10 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| Legal Compliance Specialist | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-10 | GL-ProductManager | Initial draft created: all 9 P0 features specified (90 sub-requirements), regulatory coverage verified (EUDR Articles 2(40)/3/4/8/9/10/11/29/31, ILO Conventions, UDHR, UNGP, CITES, FLEGT, CSDDD), 8-category Article 2(40) compliance framework defined, 15-table DB schema V111 designed, ~35 API endpoints specified, 20 Prometheus metrics defined, 21 RBAC permissions registered, 750+ test target set, 18-week timeline established |
