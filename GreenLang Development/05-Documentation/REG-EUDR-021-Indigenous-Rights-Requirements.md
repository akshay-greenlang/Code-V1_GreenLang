# Regulatory Requirements Document: EUDR Indigenous Rights Compliance

## REG-EUDR-021 -- Indigenous Rights Checker Agent

| Field | Value |
|-------|-------|
| **Document ID** | REG-EUDR-021 |
| **Agent ID** | GL-EUDR-IRC-021 |
| **Component** | Indigenous Rights Checker Agent |
| **Category** | EUDR Regulatory Agent -- Human Rights & Indigenous Peoples Compliance |
| **Author** | GL-RegulatoryIntelligence |
| **Date** | 2026-03-10 |
| **Status** | Complete |
| **Classification** | Regulatory Intelligence -- Internal |
| **Primary Regulation** | Regulation (EU) 2023/1115 (EUDR) |
| **Secondary Frameworks** | UNDRIP, ILO Convention 169, CSDDD (EU) 2024/1760, EU Timber Regulation 995/2010, EU Conflict Minerals Regulation 2017/821 |

---

## Table of Contents

1. [EUDR Articles Addressing Indigenous Peoples' Rights](#1-eudr-articles-addressing-indigenous-peoples-rights)
2. [FPIC Requirements Under EUDR and International Frameworks](#2-fpic-requirements-under-eudr-and-international-frameworks)
3. [Indigenous Land Rights Verification and Geolocation Overlap](#3-indigenous-land-rights-verification-and-geolocation-overlap)
4. [International Human Rights Frameworks Referenced by EUDR](#4-international-human-rights-frameworks-referenced-by-eudr)
5. [Indigenous Community Engagement and Consultation Requirements](#5-indigenous-community-engagement-and-consultation-requirements)
6. [Documentation and Evidence Requirements for FPIC Compliance](#6-documentation-and-evidence-requirements-for-fpic-compliance)
7. [Penalties and Enforcement for Indigenous Rights Violations](#7-penalties-and-enforcement-for-indigenous-rights-violations)
8. [Integration with Country Risk Benchmarking (Article 29)](#8-integration-with-country-risk-benchmarking-article-29)
9. [Regulatory Precedents from EUTR and Conflict Minerals Regulation](#9-regulatory-precedents-from-eutr-and-conflict-minerals-regulation)
10. [Best Practices from Certification Schemes](#10-best-practices-from-certification-schemes)
11. [Technical Specifications for Indigenous Rights Checker Agent](#11-technical-specifications-for-indigenous-rights-checker-agent)
12. [Data Sources and Integration Architecture](#12-data-sources-and-integration-architecture)
13. [Regulatory Compliance Matrix](#13-regulatory-compliance-matrix)

---

## 1. EUDR Articles Addressing Indigenous Peoples' Rights

### 1.1 Article 2 -- Definitions (Point 40: "Relevant Legislation")

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 2, paragraph 40.

Article 2(40) defines "relevant legislation" as the laws applicable in the country of production concerning the legal status of the area of production in terms of eight categories:

| Category | Scope | Indigenous Rights Relevance |
|----------|-------|----------------------------|
| **(a)** Land use rights | Rules on land tenure, right of use and ownership | **PRIMARY** -- Directly governs indigenous land title, customary use rights, and territorial claims |
| **(b)** Environmental protection | Conservation, biodiversity, environmental laws | SECONDARY -- Protected areas often overlap indigenous territories |
| **(c)** Forest-related rules | Forest management, logging, forest conversion | SECONDARY -- Forest concessions may conflict with indigenous forest use rights |
| **(d)** Third-party rights | Rights of third parties to property and resources | **PRIMARY** -- Indigenous peoples are third parties with pre-existing property rights |
| **(e)** Labour rights | Employment, working conditions | INDIRECT -- Forced or exploitative labor affecting indigenous workers |
| **(f)** Human rights protected under international law | International human rights obligations | **PRIMARY** -- UNDRIP, ILO 169, ICCPR, ICESCR protections for indigenous peoples |
| **(g)** The principle of free, prior and informed consent (FPIC) | Including as set out in UNDRIP | **PRIMARY** -- Core mechanism for protecting indigenous decision-making authority |
| **(h)** Tax, anti-corruption, trade and customs regulations | Fiscal and trade compliance | INDIRECT -- Corruption in land titling affects indigenous rights |

**Critical Implementation Note:** The phrase "provided that these standards are incorporated in national law" qualifies categories (f) and (g). This means FPIC obligations under EUDR apply only where the country of production has enacted national legislation incorporating FPIC requirements. However, 24 countries have ratified ILO Convention 169 (which contains binding FPIC requirements), and numerous additional countries have constitutional or legislative provisions recognizing indigenous land rights and consent requirements.

**Agent Implementation:** The Indigenous Rights Checker must maintain a per-country legal framework database that tracks:
- Whether the country has ratified ILO Convention 169
- Whether national legislation incorporates FPIC requirements
- Constitutional provisions recognizing indigenous peoples' rights
- Specific statutes governing indigenous land tenure
- Customary law recognition provisions
- Judicial precedents affirming indigenous rights

### 1.2 Article 3 -- Prohibition (Legality Requirement)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 3.

Article 3 establishes that relevant commodities and products may only be placed on the EU market or exported if they are:
- **(a)** Deforestation-free
- **(b)** Produced in accordance with the relevant legislation of the country of production
- **(c)** Covered by a due diligence statement

The legality requirement in Article 3(b) is the anchor for indigenous rights compliance. Because "relevant legislation" under Article 2(40) includes land use rights, third-party rights, human rights under international law, and FPIC, any commodity production that violates these provisions in the country of production renders the product non-compliant with Article 3(b), regardless of whether deforestation occurred.

**Agent Implementation:** The agent must verify legality across all eight categories of relevant legislation for each plot of land, with particular attention to:
- Land title verification against indigenous territorial claims
- FPIC documentation for production on or near indigenous territories
- Compliance with national indigenous rights statutes
- Absence of pending litigation or disputed land claims by indigenous communities

### 1.3 Article 8 -- Due Diligence System

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 8.

Article 8 requires operators to exercise due diligence prior to placing commodities on the EU market or exporting them. The due diligence system must include:
- **(a)** Information collection (Article 9)
- **(b)** Risk assessment (Article 10)
- **(c)** Risk mitigation (Article 11)

The three-step process forms the procedural framework within which indigenous rights verification must occur. The due diligence obligation is continuous -- operators must maintain and regularly update their due diligence system.

**Agent Implementation:** The Indigenous Rights Checker implements a parallel three-step verification process:
1. **Information collection** (mapped to Article 9): Gather indigenous territory data, FPIC documentation, community information, rights violation reports
2. **Risk assessment** (mapped to Article 10): Evaluate plot-territory overlaps, FPIC compliance gaps, violation severity, consultation adequacy
3. **Risk mitigation** (mapped to Article 11): Recommend enhanced verification, require additional documentation, escalate to human review, block non-compliant submissions

### 1.4 Article 9 -- Information Requirements

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 9.

Article 9 specifies the information operators must collect as part of due diligence. The following requirements are directly relevant to indigenous rights verification:

| Article 9 Provision | Indigenous Rights Application |
|---------------------|------------------------------|
| Art. 9(1)(d): Geolocation of all plots of land | Enables spatial overlap analysis with indigenous territory boundaries |
| Art. 9(1)(e): Date or time range of production | Enables temporal verification of FPIC validity (was consent obtained before production commenced?) |
| Art. 9(1)(f): Verification that relevant legislation was complied with | Requires evidence of compliance with indigenous rights legislation |
| Art. 9(1)(g): Information enabling risk assessment | Includes indigenous rights risk factors per Article 10(2) |

**Agent Implementation:** The agent must collect and validate the following information elements:
- Plot geolocation (coordinates or polygon) for overlap analysis
- Production date range for FPIC temporal verification
- Supplier attestation of relevant legislation compliance
- Country-specific indigenous rights legislation compliance evidence
- FPIC documentation (where applicable)

### 1.5 Article 10 -- Risk Assessment (Indigenous-Specific Criteria)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 10, paragraphs 1-3.

Article 10(2) enumerates the criteria that operators must consider in their risk assessment. The following criteria directly address indigenous peoples:

| Article 10(2) Criterion | Text Summary | Agent Implementation |
|--------------------------|-------------|---------------------|
| **Art. 10(2)(a)** | Assignment of risk to the country of production per Article 29 | Country risk score incorporating indigenous rights governance indicators |
| **Art. 10(2)(c)** | Presence of indigenous peoples in the country of production or parts thereof | Indigenous territory presence detection per country and sub-national region |
| **Art. 10(2)(d)** | Consultation and cooperation in good faith with indigenous peoples | FPIC process verification, consultation tracking, and good faith assessment |
| **Art. 10(2)(e)** | Duly reasoned claims by indigenous peoples regarding the use or ownership of the area used for producing the relevant commodity | Land rights claim database, dispute tracking, and claim-plot correlation |
| **Art. 10(2)(f)** | Prevalence of deforestation or forest degradation in the country of production | Correlated with indigenous territory encroachment patterns |
| **Art. 10(2)(i)** | Concerns about the country of production or origin related to the level of corruption, prevalence of document and data falsification, lack of law enforcement, human rights violations | Indigenous rights violation monitoring, law enforcement gaps for indigenous land rights |
| **Art. 10(2)(j)** | Whether relevant products involve circumvention, mixing, or substitution | Supply chain opacity obscuring indigenous rights non-compliance |

**Critical Criteria for Indigenous Rights (10(2)(c), (d), and (e)):**

These three criteria form the core indigenous rights risk assessment mandate:

1. **Presence of indigenous peoples** (10(2)(c)): The agent must determine whether indigenous peoples are present in the country of production or the specific sub-national region from which the commodity is sourced. This requires a comprehensive database of indigenous territorial presence at country and sub-national levels.

2. **Consultation and cooperation in good faith** (10(2)(d)): When indigenous peoples are present, the agent must verify that consultation and cooperation occurred in good faith. "Good faith" implies:
   - Consultation before commencement of production activities
   - Adequate time for community decision-making processes
   - Information provided in accessible language and format
   - Respect for community decision-making structures
   - Genuine willingness to accommodate community concerns
   - Documentation of consultation process and outcomes

3. **Duly reasoned claims** (10(2)(e)): The agent must check for existing claims by indigenous peoples regarding the use or ownership of production areas. "Duly reasoned" means claims supported by "objective and verifiable information," which may include:
   - Historical occupation evidence
   - Customary use documentation
   - Legal proceedings (past or pending)
   - Community testimony and oral history records
   - Anthropological or ethnographic evidence
   - Government recognition documents
   - Certification body field assessment records

**Agent Implementation:** The agent must implement:
- Indigenous presence database covering 100+ countries at sub-national granularity
- Good faith consultation verification engine with multi-factor scoring
- Land claims registry integration with dispute status tracking
- Automated risk flagging when any of the three criteria indicate indigenous rights risk

### 1.6 Article 11 -- Risk Mitigation

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 11.

When risk assessment under Article 10 identifies a non-negligible risk, Article 11 requires operators to take adequate and proportionate risk mitigation measures. For indigenous rights, risk mitigation measures include:

| Risk Level | Required Mitigation Measures |
|------------|------------------------------|
| Non-negligible indigenous territory overlap | Enhanced spatial verification; field-level boundary confirmation; community engagement |
| FPIC documentation gaps | Request FPIC documentation from supplier; commission independent FPIC verification; engage with community representatives |
| Active land rights claims | Legal review of claims; suspension of sourcing until claims resolved; engagement with national land authority |
| Reported rights violations | Independent investigation; source verification; consideration of supply chain modification |
| Inadequate consultation evidence | Commission independent consultation verification; engage with community leaders; require supplier to conduct proper consultation |

Article 11 also specifies that operators must:
- Request additional information or documentation from suppliers
- Carry out independent surveys or audits
- Undertake other measures to reach a conclusion of no or merely negligible risk
- Not place the product on the market if risk cannot be mitigated to negligible

**Agent Implementation:** The agent must:
- Classify indigenous rights risk as negligible, non-negligible, or high
- Generate specific mitigation recommendations based on the nature of the identified risk
- Track mitigation actions and their outcomes
- Enforce a "no-go" decision when risk cannot be adequately mitigated
- Maintain an audit trail of risk mitigation decisions

### 1.7 Article 29 -- Benchmarking (Indigenous Rights Criteria)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 29, paragraphs 2-4.

Article 29 mandates the European Commission to classify countries or parts thereof as low, standard, or high risk. The benchmarking criteria relevant to indigenous rights are:

**Article 29(3) -- Primary Criteria:**
| Criterion | Text | Relevance |
|-----------|------|-----------|
| Art. 29(3)(a) | Rate of deforestation and forest degradation | Correlated with indigenous territory encroachment |
| Art. 29(3)(b) | Rate of expansion of agricultural land for relevant commodities | Direct pressure on indigenous lands |
| Art. 29(3)(c) | Production trends of relevant commodities | Increasing production may drive land conflicts |

**Article 29(4) -- Additional Assessment Criteria:**
| Criterion | Text | Relevance |
|-----------|------|-----------|
| Art. 29(4)(a) | Information submitted by countries, indigenous peoples, or civil society | Indigenous communities' own submissions on rights violations |
| Art. 29(4)(b) | Agreements between the country and the EU on deforestation | May include indigenous rights safeguards |
| Art. 29(4)(c) | Existence of laws aimed at combating deforestation and their enforcement | Includes indigenous land protection laws |
| **Art. 29(4)(d)** | **Existence, compliance with, or effective enforcement of laws protecting human rights, the rights of indigenous peoples, local communities, and other customary tenure rights holders** | **Direct and explicit indigenous rights criterion** |
| Art. 29(4)(e) | Accessibility of transparent data regarding land rights | Transparency of indigenous territory registries |

**Critical Analysis of Article 29(4)(d):**

This is the most explicit indigenous rights provision in the EUDR benchmarking framework. It requires the Commission to consider three dimensions:
1. **Existence** of laws protecting indigenous rights
2. **Compliance** with those laws (whether they are observed in practice)
3. **Effective enforcement** of those laws (whether violations are prosecuted and remedied)

However, a significant regulatory gap exists: Article 29(4) uses the word "may" rather than "shall," giving the Commission discretion over whether to consider these criteria. Civil society organizations have criticized this approach, noting that the Commission's initial benchmarking methodology did not directly and conclusively consider indigenous rights protections, treating them as secondary to deforestation rate data.

**Agent Implementation:** The agent must:
- Track Article 29(4)(d) indicators for each country: existence, compliance, and enforcement of indigenous rights laws
- Maintain a country-level indigenous rights governance score
- Feed this score into EUDR-016 (Country Risk Evaluator) for composite risk calculation
- Monitor Commission benchmarking updates for reclassifications that affect indigenous rights risk levels
- Flag countries where formal laws exist but enforcement is weak (de jure vs. de facto gap)

---

## 2. FPIC Requirements Under EUDR and International Frameworks

### 2.1 FPIC Under EUDR

EUDR incorporates FPIC through Article 2(40)(g), which includes "the principle of free, prior and informed consent, including as set out in the United Nations Declaration on the Rights of Indigenous Peoples" within the definition of "relevant legislation." This incorporation is conditional on FPIC being part of the national legislation of the country of production.

**FPIC Applicability Matrix:**

| Country Category | FPIC Status Under EUDR | Examples | Agent Action |
|-----------------|------------------------|----------|-------------|
| ILO 169 ratified + national FPIC legislation | FPIC is part of relevant legislation; full verification required | Brazil, Colombia, Peru, Bolivia, Mexico, Guatemala, Honduras, Paraguay, Chile, Argentina, Ecuador, Venezuela, Nicaragua, Costa Rica, Central African Republic, Nepal, Denmark, Netherlands, Norway, Spain, Luxembourg, Fiji, Dominica | Full FPIC verification engine activation |
| No ILO 169 ratification + national FPIC legislation | FPIC may be part of relevant legislation through other statutes | Philippines (IPRA 1997), Australia (Native Title Act), Canada (Duty to Consult), India (PESA/FRA) | Verify national FPIC provisions; apply verification |
| No ILO 169 + constitutional indigenous recognition | Partial FPIC obligations may exist | Indonesia (constitutional recognition), DRC (Forest Code 2002), Cameroon (constitutional provisions) | Assess constitutional and statutory protections; conditional FPIC check |
| No ILO 169 + minimal indigenous rights legislation | FPIC may not be legally required under EUDR | Some African and Asian producer countries | Flag gap in legal protection; recommend voluntary FPIC; apply enhanced risk |
| No indigenous presence | N/A | Some EU member states sourcing domestically | No FPIC check required |

### 2.2 FPIC Under UNDRIP

The United Nations Declaration on the Rights of Indigenous Peoples (UNDRIP), adopted by the UN General Assembly on September 13, 2007, establishes the international standard for FPIC. While UNDRIP is not legally binding, it represents the consensus of 144 states and is referenced in EUDR Article 2(40)(g).

**Key UNDRIP Articles on FPIC:**

| UNDRIP Article | Requirement | Agent Verification Element |
|----------------|-------------|---------------------------|
| **Art. 10** | No forcible removal from lands or territories; no relocation without FPIC and just compensation | Verify no forced relocation occurred; check for relocation agreements and compensation records |
| **Art. 11(2)** | Redress for cultural, intellectual, religious, and spiritual property taken without FPIC | Check for cultural heritage impact assessments |
| **Art. 19** | States shall consult with indigenous peoples to obtain FPIC before adopting legislation or administrative measures affecting them | Verify that applicable national legislation was enacted with indigenous consultation |
| **Art. 26** | Indigenous peoples have the right to lands, territories, and resources they have traditionally owned, occupied, or used | Verify that production does not encroach on traditionally occupied territories |
| **Art. 28** | Right to redress for lands confiscated, taken, occupied, or used without FPIC; restitution or just compensation | Check for outstanding restitution claims or compensation proceedings |
| **Art. 29(2)** | States shall take measures to prevent storage or disposal of hazardous materials on indigenous lands without FPIC | Environmental contamination checks for production plots on/near indigenous lands |
| **Art. 32(2)** | States shall consult and cooperate in good faith with indigenous peoples through their representative institutions to obtain FPIC prior to approval of any project affecting their lands, territories, and resources | Core FPIC verification: was consent obtained through legitimate representative institutions before project approval? |

### 2.3 FPIC Under ILO Convention 169

ILO Convention 169 (Indigenous and Tribal Peoples Convention, 1989) is the primary binding international legal instrument on indigenous rights. Ratified by 24 countries as of 2026, it contains mandatory FPIC and consultation requirements.

**Key ILO 169 Articles:**

| ILO 169 Article | Requirement | Binding Status | Agent Verification |
|-----------------|-------------|----------------|-------------------|
| **Art. 6(1)(a)** | Governments shall consult indigenous peoples through appropriate procedures and genuine representative institutions whenever legislative or administrative measures may affect them directly | Binding on ratifying states | Verify consultation occurred through recognized representative institutions |
| **Art. 6(2)** | Consultations shall be undertaken in good faith and in a form appropriate to the circumstances, with the objective of achieving agreement or consent | Binding on ratifying states | Assess good faith indicators (timing, information, process, outcomes) |
| **Art. 7(1)** | Indigenous peoples shall have the right to decide their own priorities for development as it affects their lives, lands, and livelihoods | Binding on ratifying states | Verify development priorities alignment and community decision-making authority |
| **Art. 14(1)** | The rights of ownership and possession of the peoples concerned over the lands which they traditionally occupy shall be recognized | Binding on ratifying states | Verify land ownership recognition and possession rights |
| **Art. 15(1)** | The rights of indigenous peoples to natural resources pertaining to their lands shall be specially safeguarded | Binding on ratifying states | Verify natural resource rights protection for affected communities |
| **Art. 15(2)** | Governments shall consult with indigenous peoples before undertaking or permitting exploration or exploitation of resources on their lands | Binding on ratifying states | Verify consultation occurred before resource exploitation |
| **Art. 16(1)** | Indigenous peoples shall not be removed from the lands they occupy | Binding on ratifying states | Verify no displacement occurred |
| **Art. 16(2)** | Where relocation is necessary as exceptional measure, shall take place only with free and informed consent | Binding on ratifying states | Verify FPIC for any relocation |

**ILO 169 Ratification Status for Major EUDR Commodity Producers:**

| Country | EUDR Commodities | ILO 169 Ratification | Year | Status |
|---------|-----------------|---------------------|------|--------|
| Brazil | Soya, Cattle, Coffee, Wood | Yes | 2002 | In force |
| Colombia | Coffee, Palm Oil, Cocoa | Yes | 1991 | In force |
| Peru | Coffee, Cocoa, Wood | Yes | 1994 | In force |
| Bolivia | Soya, Coffee, Wood | Yes | 1991 | In force |
| Ecuador | Cocoa, Coffee, Palm Oil | Yes | 1998 | In force |
| Guatemala | Coffee, Palm Oil, Rubber | Yes | 1996 | In force |
| Honduras | Coffee, Palm Oil | Yes | 1995 | In force |
| Mexico | Coffee | Yes | 1990 | In force |
| Paraguay | Soya, Cattle | Yes | 1993 | In force |
| Argentina | Soya | Yes | 2000 | In force |
| Chile | Wood | Yes | 2008 | In force |
| Central African Republic | Wood, Cocoa | Yes | 2010 | In force |
| Nepal | -- | Yes | 2007 | In force |
| Indonesia | Palm Oil, Rubber, Wood, Cocoa, Coffee | **No** | N/A | Not ratified |
| Cote d'Ivoire | Cocoa, Coffee, Rubber | **No** | N/A | Not ratified |
| Ghana | Cocoa | **No** | N/A | Not ratified |
| Malaysia | Palm Oil, Rubber, Wood | **No** | N/A | Not ratified |
| DRC | Wood, Cocoa, Coffee | **No** | N/A | Not ratified |
| Cameroon | Cocoa, Wood, Rubber | **No** | N/A | Not ratified |
| Thailand | Rubber | **No** | N/A | Not ratified |
| Vietnam | Coffee, Rubber | **No** | N/A | Not ratified |

### 2.4 FPIC Process Requirements (Consolidated)

The following table consolidates FPIC process requirements from UNDRIP, ILO 169, and certification scheme best practices into a unified verification framework:

| FPIC Element | Requirement | Verification Criteria | Score Weight |
|-------------|-------------|----------------------|-------------|
| **FREE** | Consent given voluntarily, absent coercion, intimidation, or manipulation | No evidence of threats, bribes, or pressure; no imposed timelines; community self-directed process | 25% |
| **PRIOR** | Consent sought sufficiently in advance of any authorization or commencement of activities | Consent obtained before production commenced; reasonable advance notice period; no retrospective consent | 25% |
| **INFORMED** | Adequate information provided in accessible format and language before and during consent process | Information in local language(s); technical information explained; independent legal advice available; environmental and social impact assessments shared; alternative options presented | 25% |
| **CONSENT** | Collective decision made through customary decision-making processes | Decision made by legitimate community representatives; customary governance processes followed; community assembly records; consent documented in writing; right to withhold consent respected | 25% |

---

## 3. Indigenous Land Rights Verification and Geolocation Overlap

### 3.1 Spatial Overlap Detection Requirements

EUDR Article 9(1)(d) requires geolocation data for all production plots, and Article 10(2)(c)-(e) requires assessment of indigenous territorial presence and claims. The intersection of these requirements mandates a spatial analysis capability.

**Overlap Classification Framework:**

| Overlap Category | Definition | Spatial Criteria | Risk Level | Required Action |
|-----------------|------------|-----------------|------------|----------------|
| **DIRECT** | Production plot is entirely within an indigenous territory | Plot polygon completely contained within territory boundary | CRITICAL | FPIC verification mandatory; enhanced DD required; production suspension if FPIC absent |
| **PARTIAL** | Production plot partially overlaps with an indigenous territory | Non-zero intersection area between plot and territory polygons | HIGH | FPIC verification for overlapping area; determine if production occurs in overlap zone |
| **ADJACENT** | Production plot is within immediate proximity of an indigenous territory | Plot centroid or boundary within configurable buffer (default: 5 km) | MEDIUM | Enhanced monitoring; community engagement assessment; buffer zone encroachment check |
| **PROXIMATE** | Production plot is within broader proximity of an indigenous territory | Plot within extended buffer (default: 25 km) | LOW | Awareness flagging; quarterly monitoring; inclusion in indigenous rights risk report |
| **NONE** | No indigenous territory within detection radius | No intersection and distance exceeds maximum buffer | MINIMAL | Standard due diligence only; no indigenous rights verification required |

### 3.2 Data Sources for Indigenous Territory Boundaries

| Source | Coverage | Data Type | Update Frequency | API/Format | Reliability |
|--------|----------|-----------|-------------------|------------|-------------|
| **LandMark** (landmarkmap.org) | Global (100+ countries, 1.15 million areas) | Georeferenced polygons, legal status, community data | Continuous | GeoJSON, WMS, WFS | HIGH -- 70+ institutional contributors |
| **RAISG** (raisg.org) | Amazon Basin (9 countries) | Indigenous territories, protected areas, deforestation | Annual | Shapefile, GeoJSON | HIGH -- 9 regional organizations |
| **FUNAI** (funai.gov.br) | Brazil | Terras Indigenas (official indigenous territories) | Continuous | Shapefile, GeoJSON | AUTHORITATIVE -- Brazilian federal agency |
| **BPN/AMAN** | Indonesia | Adat (customary) territories; government concession maps | Periodic | Shapefile | MEDIUM -- ongoing mapping effort; incomplete coverage |
| **ACHPR** (achpr.org) | Africa (54 countries) | Indigenous and tribal peoples' territories | Periodic | Reports, some GIS data | MEDIUM -- varies by country |
| **National Land Registries** | Per country | Official land title, cadastral data, indigenous demarcations | Continuous | Varies (API, bulk download, manual request) | HIGH -- official government data |
| **WDPA** (protectedplanet.net) | Global | Protected areas (often overlap with indigenous territories) | Monthly | Shapefile, GeoJSON, API | HIGH -- UNEP-WCMC maintained |
| **Global Forest Watch** | Global | Forest cover, tree cover loss, fire alerts | Daily to annual | API, raster, vector | HIGH -- WRI maintained |
| **Native Land Digital** (native-land.ca) | Global (focus on Americas, Australia, NZ) | Indigenous territories, languages, treaties | Community-contributed | GeoJSON, API | MEDIUM -- community-sourced; educational |

### 3.3 Geospatial Technical Requirements

| Requirement | Specification | Justification |
|-------------|--------------|---------------|
| Coordinate reference system | WGS84 (EPSG:4326) | EUDR standard; alignment with GPS and satellite data |
| Spatial precision | < 10 meters | Sufficient for plot-territory boundary discrimination |
| Spatial index | R-tree or equivalent (PostGIS GIST) | Sub-second overlap detection across millions of polygons |
| Overlap area calculation | Geodetic (WGS84 ellipsoid) using Karney's algorithm | Accurate area calculation for tropical regions |
| Buffer zone computation | Geodesic buffer with configurable radius | Accurate distance computation on curved earth surface |
| Topological operations | ST_Intersects, ST_Contains, ST_Within, ST_Buffer, ST_Distance, ST_Area | Full PostGIS spatial analysis capability |
| Polygon validation | OGC Simple Features compliance | Prevent topological errors in overlap calculations |
| Multi-polygon support | Handle territories with non-contiguous areas | Many indigenous territories are fragmented |
| Temporal versioning | Point-in-time boundary queries | Territory boundaries change over time (demarcation, recognition, encroachment) |
| Data retention | 5 years per EUDR Article 31 | Regulatory requirement for due diligence records |

### 3.4 Integration with Existing EUDR Agents

| Agent | Integration Point | Data Flow |
|-------|-------------------|-----------|
| **EUDR-001** Supply Chain Mapping Master | Supply chain node enrichment with indigenous rights risk flags | Bidirectional: IRC provides risk flags; SCM provides supply chain graph context |
| **EUDR-002** Geolocation Verification | Validated plot coordinates for overlap analysis | Inbound: Verified coordinates feed into overlap detection |
| **EUDR-006** Plot Boundary Manager | Plot polygon data for spatial analysis | Inbound: Validated, versioned plot polygons used for intersection queries |
| **EUDR-016** Country Risk Evaluator | Indigenous rights governance score for country risk composite | Outbound: Indigenous rights score contributes to governance_quality factor |
| **EUDR-017** Supplier Risk Scorer | Supplier-level indigenous rights risk adjustment | Outbound: Supplier risk adjusted by FPIC compliance status and territory overlap |
| **EUDR-020** Deforestation Alert System | Deforestation alerts correlated with indigenous territory proximity | Inbound: Deforestation alerts near indigenous territories trigger enhanced investigation |

---

## 4. International Human Rights Frameworks Referenced by EUDR

### 4.1 Framework Hierarchy

The EUDR references and builds upon a hierarchy of international human rights instruments. The following table maps each framework to its EUDR relevance:

| Framework | Full Name | Year | Binding Status | EUDR Reference | Key Indigenous Rights Provisions |
|-----------|-----------|------|---------------|----------------|----------------------------------|
| **UNDRIP** | United Nations Declaration on the Rights of Indigenous Peoples | 2007 | Non-binding (but referenced in Art. 2(40)) | Article 2(40)(g) explicit reference | FPIC (Arts. 10, 19, 29, 32); land rights (Art. 26); self-determination (Art. 3); cultural rights (Art. 11) |
| **ILO 169** | Indigenous and Tribal Peoples Convention | 1989 | Binding on ratifying states | Implied through Art. 2(40)(f)-(g) | Consultation (Art. 6); land ownership (Art. 14); natural resources (Art. 15); non-removal (Art. 16) |
| **ICCPR** | International Covenant on Civil and Political Rights | 1966 | Binding | Art. 2(40)(f) "human rights protected under international law" | Self-determination (Art. 1); minority rights (Art. 27); equality (Art. 26) |
| **ICESCR** | International Covenant on Economic, Social and Cultural Rights | 1966 | Binding | Art. 2(40)(f) | Self-determination (Art. 1); right to adequate standard of living (Art. 11) |
| **CERD** | International Convention on the Elimination of All Forms of Racial Discrimination | 1965 | Binding | Art. 2(40)(f) | Non-discrimination in property rights (Art. 5(d)(v)); General Recommendation 23 on indigenous peoples |
| **UNGPs** | UN Guiding Principles on Business and Human Rights | 2011 | Soft law (framework) | Implied through CSDDD alignment | Corporate responsibility to respect rights; due diligence obligation; remedy provision |
| **CSDDD** | EU Corporate Sustainability Due Diligence Directive | 2024 | Binding (EU) | Cross-regulatory linkage | Mandatory human rights due diligence for EU companies; adverse impact identification; remediation |

### 4.2 EUDR-CSDDD Alignment

The EU Corporate Sustainability Due Diligence Directive (CSDDD, Directive (EU) 2024/1760) entered into force in 2024 with phased implementation starting 2027 (large companies) through 2029 (smaller in-scope companies). CSDDD creates a broader human rights due diligence obligation that reinforces and extends EUDR's indigenous rights requirements.

| CSDDD Provision | EUDR Alignment | Agent Implementation |
|----------------|----------------|---------------------|
| Art. 5: Identification of adverse human rights impacts | Art. 10: Risk assessment including indigenous rights criteria | Indigenous territory overlap detection; rights violation monitoring |
| Art. 7: Prevention and mitigation of adverse impacts | Art. 11: Risk mitigation for non-negligible risk | FPIC remediation workflow; community engagement tracking |
| Art. 8: Bringing actual adverse impacts to an end | Art. 11: Enhanced due diligence measures | Supply chain modification recommendations; remediation action tracking |
| Art. 9: Complaints mechanism | Not explicit in EUDR | Grievance mechanism integration; community complaint tracking |
| Art. 10: Monitoring of due diligence | Art. 8: Due diligence system maintenance | Continuous monitoring of indigenous rights compliance status |

**Agent Implementation:** The Indigenous Rights Checker should be designed for dual-use: EUDR compliance (current mandate) and CSDDD readiness (2027 mandate). This future-proofs the investment and positions the agent as the indigenous rights compliance backbone for both regulations.

---

## 5. Indigenous Community Engagement and Consultation Requirements

### 5.1 Good Faith Consultation Criteria (Article 10(2)(d))

EUDR Article 10(2)(d) requires operators to assess "consultation and cooperation in good faith with indigenous peoples." The following criteria define "good faith" based on UNDRIP Article 32(2), ILO 169 Article 6(2), and international jurisprudence:

| Good Faith Criterion | Description | Verification Indicators | Score (0-100) |
|---------------------|-------------|------------------------|---------------|
| **Timeliness** | Consultation initiated sufficiently in advance of activities | Evidence of consultation before project approval/commencement; minimum 90-day advance consultation period | 0-15 |
| **Adequate Information** | Complete, accurate, accessible information provided | Information in local language(s); technical summaries available; independent explanation provided; environmental/social impact data shared | 0-15 |
| **Representative Participation** | Legitimate community representatives involved | Recognized community leaders or governance structures participated; community-designated representatives; no by-passing of traditional governance | 0-15 |
| **Genuine Dialogue** | Meaningful two-way communication, not mere notification | Evidence of iterative discussions; community questions answered; modifications made in response to community input; meeting minutes/records | 0-15 |
| **Accommodation of Concerns** | Demonstrable effort to address community concerns | Written responses to community objections; modifications to project design; benefit-sharing arrangements; mitigation measures adopted | 0-15 |
| **Absence of Coercion** | No intimidation, threats, or improper inducements | No evidence of threats, bribery, or pressure; independent observer presence; community attestation of voluntary participation | 0-10 |
| **Cultural Appropriateness** | Process respects customary decision-making | Consultation format aligned with community traditions; adequate time for community deliberation; spiritual/cultural considerations respected | 0-10 |
| **Documentation** | Complete records of consultation process | Meeting minutes, attendance records, information materials, community responses, agreements documented | 0-5 |

**Total Score Interpretation:**
- 80-100: Adequate consultation (COMPLIANT)
- 60-79: Partial consultation (REQUIRES ENHANCEMENT)
- 40-59: Insufficient consultation (NON-COMPLIANT -- risk mitigation required)
- 0-39: No meaningful consultation (CRITICAL VIOLATION)

### 5.2 Community Engagement Workflow

The agent must support a structured multi-stage community engagement process:

```
Stage 1: IDENTIFICATION
  |-- Identify indigenous communities in production area
  |-- Verify community recognition status (legal, customary, pending)
  |-- Identify representative organizations and governance structures
  |-- Document community demographics and territories
  |
Stage 2: INFORMATION DISCLOSURE
  |-- Prepare project information in accessible formats
  |-- Translate materials into community language(s)
  |-- Provide environmental and social impact assessment
  |-- Present alternatives and potential impacts
  |-- Allow independent advisory support
  |
Stage 3: CONSULTATION
  |-- Engage through community-designated process
  |-- Record all consultation meetings and interactions
  |-- Document community questions and concerns
  |-- Provide written responses to concerns
  |-- Allow adequate deliberation time (minimum 90 days)
  |
Stage 4: CONSENT/OBJECTION
  |-- Record community decision (consent, conditional consent, or objection)
  |-- If consent: document conditions and commitments
  |-- If objection: document grounds; respect decision; explore alternatives
  |-- Verify decision was made through legitimate community process
  |
Stage 5: AGREEMENT
  |-- Formalize consent in written agreement
  |-- Include benefit-sharing provisions
  |-- Define monitoring and review mechanisms
  |-- Establish grievance mechanism
  |-- Both parties sign and retain copies
  |
Stage 6: MONITORING
  |-- Regular compliance reviews (minimum annual)
  |-- Community feedback collection
  |-- Grievance resolution tracking
  |-- Agreement amendment process as needed
  |-- Renewal process at defined intervals
```

### 5.3 Grievance Mechanism Requirements

Per CSDDD Article 9 and certification scheme best practices (FSC Principle 2, RSPO Principle 6), operators must establish or participate in grievance mechanisms accessible to affected indigenous communities.

| Requirement | Specification | Agent Implementation |
|-------------|--------------|---------------------|
| Accessibility | Available in local language(s); accessible to remote communities; no internet dependency | Multi-channel intake (phone, SMS, in-person, written, digital) |
| Timeliness | Acknowledgment within 7 days; investigation within 30 days; resolution within 90 days | SLA tracking and escalation engine |
| Independence | Process free from interference by the operator | Independent reviewer integration; community observer provisions |
| Transparency | Complainants informed of process and outcomes | Status tracking and notification system |
| Documentation | Complete records of complaints, investigations, and resolutions | Immutable audit trail with provenance tracking |
| Non-retaliation | Protection against reprisals for complainants | Anonymous submission capability; retaliation monitoring |

---

## 6. Documentation and Evidence Requirements for FPIC Compliance

### 6.1 FPIC Documentation Checklist

The following documentation must be collected, verified, and retained for each production plot or supply chain node where indigenous rights verification is required:

| Document Category | Required Documents | Verification Criteria | Retention |
|-------------------|-------------------|----------------------|-----------|
| **Territory Identification** | Indigenous territory boundary map (georeferenced); community identification records; territory legal status documentation | Spatial data matches authoritative sources; community recognized by government or community organizations | 5 years (Art. 31) |
| **Community Information** | Community demographics; governance structure documentation; representative organization identification; contact information | Government recognition documents or anthropological evidence; community attestation of governance structure | 5 years (Art. 31) |
| **Information Disclosure** | Project information package (in local language); environmental and social impact assessment; technical summaries; alternative options presented | Translated to community language(s); independent verification of accuracy; receipt acknowledged by community | 5 years (Art. 31) |
| **Consultation Records** | Meeting minutes (signed); attendance records; consultation schedule; questions raised and responses provided; community deliberation records | Signed by community representatives; adequate number of consultation sessions; time between sessions for deliberation | 5 years (Art. 31) |
| **Consent Documentation** | Written consent agreement (signed by authorized community representatives); conditions and commitments; benefit-sharing provisions; grievance mechanism description | Signed by legitimate representatives; decision made through customary process; conditions clearly articulated; language accessible | 5 years (Art. 31) |
| **Objection Documentation** | Written record of objection (if consent withheld); grounds for objection; operator response; alternative actions taken | Community decision respected; no coercion; alternative sourcing documented | 5 years (Art. 31) |
| **Benefit-Sharing Records** | Benefit-sharing agreement; payment/transfer records; community development program documentation; monitoring reports | Payments made as agreed; benefits reaching intended recipients; community satisfaction assessment | 5 years (Art. 31) |
| **Ongoing Monitoring** | Annual compliance review reports; community feedback records; grievance logs; agreement amendment records; renewal documentation | Regular reviews conducted; community feedback incorporated; grievances resolved within SLA | 5 years (Art. 31) |

### 6.2 FPIC Documentation Scoring Engine

The FPIC Documentation Verification Engine assigns a deterministic completeness and quality score (0-100) based on the presence and quality of required documentation:

```
FPIC_Score = (
    territory_identification_score * 0.10
    + community_information_score * 0.10
    + information_disclosure_score * 0.15
    + consultation_records_score * 0.20
    + consent_documentation_score * 0.25
    + benefit_sharing_score * 0.10
    + monitoring_records_score * 0.10
)

Where each component_score is calculated as:
    component_score = (documents_present / documents_required) * quality_factor

quality_factor = {
    1.0  if all quality criteria met
    0.75 if most quality criteria met (>= 75%)
    0.50 if some quality criteria met (>= 50%)
    0.25 if minimal quality criteria met (>= 25%)
    0.10 if documents present but quality criteria largely unmet
    0.00 if documents absent
}

FPIC Score Interpretation:
    >= 80: COMPLIANT -- FPIC process adequately documented
    60-79: PARTIAL -- Documentation gaps requiring supplementation
    40-59: INSUFFICIENT -- Significant documentation deficiencies
    20-39: CRITICAL -- Major FPIC process failures
    0-19:  ABSENT -- No meaningful FPIC documentation
```

### 6.3 Evidence Authentication

All FPIC documents must be verified for authenticity. The agent integrates with EUDR-012 (Document Authentication Agent) for:

| Authentication Check | Method | Threshold |
|---------------------|--------|-----------|
| Document integrity | SHA-256 hash comparison | Bit-perfect match |
| Signature verification | Digital signature or witnessed physical signature | Valid signature from authorized representative |
| Temporal verification | Document timestamp within expected process timeline | Consent date prior to production commencement date |
| Language verification | Document in community language(s) or with certified translation | Translation certificate or community attestation |
| Source verification | Document traceable to legitimate consultation process | Provenance chain from community engagement to digital record |

---

## 7. Penalties and Enforcement for Indigenous Rights Violations

### 7.1 EUDR Penalty Framework (Article 25)

Article 25 of the EUDR establishes the penalty framework for non-compliance. While penalties are determined by Member States (Article 25(1)), the regulation sets minimum standards:

| Penalty Category | EUDR Provision | Application to Indigenous Rights Violations |
|-----------------|----------------|---------------------------------------------|
| **Financial penalties** | At least 4% of EU-wide annual turnover for the preceding financial year (Art. 25(2)) | Applied when operator places products on EU market that were produced in violation of indigenous rights (Art. 3(b) via Art. 2(40)) |
| **Confiscation of products** | Confiscation of relevant commodities and products (Art. 25(2)) | Products linked to indigenous rights violations subject to seizure at EU borders or within the market |
| **Confiscation of revenues** | Confiscation of revenues earned from transactions involving non-compliant products (Art. 25(2)) | Revenue from sales of products linked to indigenous rights violations may be confiscated |
| **Market exclusion** | Temporary exclusion from placing products on EU market (Art. 25(2)) | Operators with systematic indigenous rights violations may be barred from EU market |
| **Public procurement ban** | Temporary exclusion from public procurement and access to public funding (Art. 25(2)) | Operators with indigenous rights violation history excluded from government contracts |
| **Public naming** | Publication of names of operators found in violation (Art. 25(2)) | Public disclosure of companies linked to indigenous rights violations (reputational impact) |

### 7.2 Enforcement Mechanisms

| Mechanism | Description | Indigenous Rights Application |
|-----------|-------------|------------------------------|
| Competent authority checks | Art. 14-19: Member States designate competent authorities for compliance checks | Check rate: minimum 9% of operators for high-risk countries (Art. 16(8)); checks must cover indigenous rights compliance |
| Substantiated concerns | Art. 31: Any natural or legal person can submit substantiated concerns about non-compliance | Indigenous communities, NGOs, and human rights organizations can submit concerns about violations |
| EU Information System | Art. 33: Centralized system for due diligence statement submission and monitoring | Indigenous rights compliance data included in DDS; system enables cross-referencing with territory databases |
| International cooperation | Art. 30-31: Cooperation with producer countries on enforcement | Cooperation with indigenous rights bodies and national indigenous affairs agencies |

### 7.3 Enhanced Penalties for High-Risk Country Sourcing

Under Article 16, competent authorities must perform enhanced checks on operators sourcing from high-risk countries (as classified under Article 29). If a country is classified as high-risk partly due to inadequate indigenous rights protections (Article 29(4)(d)), operators sourcing from that country face:

- **Higher check frequency**: Minimum 9% of operators checked (vs. lower thresholds for standard/low risk)
- **Enhanced DD requirement**: Must demonstrate indigenous rights verification beyond standard requirements
- **Mandatory documentation**: FPIC documentation required for all production plots, not just those with identified overlap
- **Increased scrutiny**: Competent authorities specifically review indigenous rights compliance as part of enhanced checks

### 7.4 Cross-Regulatory Penalty Exposure

| Regulation | Penalty for Indigenous Rights Violations | Cumulative Risk |
|------------|------------------------------------------|-----------------|
| EUDR (2023/1115) | Up to 4% of EU-wide annual turnover; market exclusion; product confiscation | PRIMARY |
| CSDDD (2024/1760) | Civil liability for failure to prevent adverse human rights impacts; fines up to 5% of net worldwide turnover | ADDITIONAL (from 2027) |
| National criminal law | Criminal prosecution in producer countries for indigenous land violations | PARALLEL -- in jurisdiction of production |
| Civil litigation | Community lawsuits for damages; injunctions against operations | PARALLEL -- in multiple jurisdictions |
| Certification revocation | FSC/RSPO/RA certification withdrawal for FPIC violations | REPUTATIONAL AND COMMERCIAL |

---

## 8. Integration with Country Risk Benchmarking (Article 29)

### 8.1 Indigenous Rights Governance Score

The Indigenous Rights Checker agent contributes an indigenous rights governance score to the EUDR-016 Country Risk Evaluator composite risk calculation. This score is derived from multiple indicators:

**Indigenous Rights Governance Score Calculation:**

```
Indigenous_Rights_Score = (
    legal_framework_score * 0.25
    + enforcement_effectiveness_score * 0.20
    + territory_recognition_score * 0.20
    + fpic_implementation_score * 0.20
    + judicial_access_score * 0.15
)

Where:
    legal_framework_score (0-100):
        - ILO 169 ratification (25 points)
        - Constitutional recognition of indigenous rights (20 points)
        - Specific indigenous rights legislation (20 points)
        - FPIC codified in national law (20 points)
        - Customary law recognition provisions (15 points)

    enforcement_effectiveness_score (0-100):
        - Land title enforcement rate for indigenous territories (30 points)
        - Prosecution rate for indigenous land encroachment (25 points)
        - Indigenous affairs agency capacity and budget (20 points)
        - Response time to rights violation reports (15 points)
        - Remediation effectiveness (10 points)

    territory_recognition_score (0-100):
        - % of indigenous territories with formal legal title (40 points)
        - Demarcation completeness (30 points)
        - Territory mapping coverage (20 points)
        - Registration in national land registry (10 points)

    fpic_implementation_score (0-100):
        - FPIC legal requirements exist (25 points)
        - FPIC process guidelines published (20 points)
        - FPIC implementation track record (positive outcomes) (25 points)
        - Independent FPIC verification mechanisms exist (15 points)
        - Community capacity to participate in FPIC (15 points)

    judicial_access_score (0-100):
        - Indigenous peoples can access courts (25 points)
        - Legal aid available for land rights cases (20 points)
        - Court recognition of customary tenure (25 points)
        - Timeliness of judicial proceedings (15 points)
        - Enforcement of court decisions favoring indigenous rights (15 points)
```

**Score Interpretation and Risk Classification:**

| Score Range | Classification | EUDR Risk Implication | Agent Action |
|-------------|---------------|----------------------|--------------|
| 80-100 | STRONG indigenous rights protection | Supports LOW risk classification | Standard FPIC checks; simplified verification where appropriate |
| 60-79 | MODERATE indigenous rights protection | Supports STANDARD risk classification | Standard FPIC verification; enhanced monitoring for specific regions |
| 40-59 | WEAK indigenous rights protection | May support HIGH risk classification | Enhanced FPIC verification; field-level checks recommended; heightened monitoring |
| 0-39 | INADEQUATE indigenous rights protection | Supports HIGH risk classification | Mandatory enhanced due diligence; independent verification required; supply chain modification consideration |

### 8.2 Data Flow to Country Risk Evaluator (EUDR-016)

The Indigenous Rights Score feeds into EUDR-016's governance_quality factor, which carries a 20% weight in the composite country risk score. Within the governance_quality factor, indigenous rights is one of four sub-components:

```
governance_quality = (
    wgi_score * 0.40           -- World Bank Worldwide Governance Indicators
    + cpi_score * 0.30         -- Transparency International CPI
    + forest_governance * 0.20 -- FAO/ITTO forest governance indicators
    + enforcement * 0.10       -- Enforcement effectiveness
)

Enhanced governance_quality (with indigenous rights integration):
governance_quality_enhanced = (
    wgi_score * 0.30
    + cpi_score * 0.25
    + forest_governance * 0.15
    + enforcement * 0.10
    + indigenous_rights_score * 0.20  -- NEW: from IRC agent
)
```

### 8.3 Country-Specific Indigenous Rights Profiles

The agent maintains detailed profiles for each EUDR-relevant commodity-producing country. Each profile includes:

| Profile Element | Data Points | Update Frequency |
|----------------|-------------|------------------|
| Legal framework summary | ILO 169 status; constitutional provisions; indigenous rights laws; FPIC legislation | Annual or on legislative change |
| Indigenous population data | Population size; number of distinct peoples; % of national population; geographic distribution | Annual |
| Territory data | Number of territories; total area; % of national territory; legal status breakdown (titled/claimed/pending) | Semi-annual |
| Active conflicts | Land conflicts; ongoing litigation; displacement incidents; rights violation reports | Continuous monitoring |
| FPIC track record | FPIC processes conducted; consent rate; objection rate; dispute frequency | Quarterly |
| Institutional capacity | Indigenous affairs agency name, budget, staffing; judicial capacity for land rights | Annual |
| NGO assessments | IWGIA Indigenous World report; Forest Peoples Programme assessments; Cultural Survival reports | Annual |

---

## 9. Regulatory Precedents from EUTR and Conflict Minerals Regulation

### 9.1 EU Timber Regulation (Regulation (EU) No 995/2010)

The EUTR was the direct predecessor to the EUDR (which repealed it). Key lessons for indigenous rights compliance:

| EUTR Provision | Lesson Learned | EUDR Application |
|---------------|----------------|-------------------|
| Legality definition limited to timber harvesting laws | Narrow legality definition excluded human rights considerations; operators focused only on logging permits | EUDR Article 2(40) explicitly expanded legality to include human rights, FPIC, and land use rights |
| Due diligence obligation (Art. 6) | Due diligence was often superficial; operators relied on paper compliance | EUDR Article 10 specifies detailed risk assessment criteria including indigenous presence |
| Monitoring Organization recognition (Art. 8) | Third-party verification provided assurance but coverage was limited | EUDR does not use Monitoring Organizations but requires operator-level due diligence |
| FLEGT VPA licensing | VPAs in 15 countries promoted governance reforms and stakeholder engagement including indigenous communities | EUDR Article 29 benchmarking incorporates governance quality; VPA experience informs country assessment |
| Enforcement gaps | EUTR enforcement varied significantly across Member States; some states had minimal checking capacity | EUDR Article 16 sets minimum check rates; Article 25 sets minimum penalty thresholds |
| Indonesia VPA success | Only VPA that produced operational licensing (SVLK); included indigenous rights safeguards | Demonstrates that country-level governance reform can address indigenous rights through supply chain requirements |

**Key Lesson:** The EUTR's narrow scope (timber only, legality of harvesting only) failed to address the broader human rights dimensions of forestry. The EUDR's expanded scope (7 commodities, broader legality definition including human rights and FPIC) directly addresses this gap, but creates significantly greater compliance complexity for operators.

### 9.2 EU Conflict Minerals Regulation (Regulation (EU) 2017/821)

The EU Conflict Minerals Regulation addresses supply chain due diligence for tin, tantalum, tungsten, and gold from conflict-affected and high-risk areas. Key parallels with EUDR indigenous rights requirements:

| Conflict Minerals Provision | Parallel with EUDR Indigenous Rights | Agent Implementation |
|----------------------------|--------------------------------------|---------------------|
| Five-step due diligence framework (aligned with OECD DDG) | Similar three-step due diligence in EUDR (Art. 8-11) | FPIC verification as part of due diligence step 2 (risk assessment) and step 3 (mitigation) |
| Conflict-affected and high-risk areas (CAHRA) identification | Country risk benchmarking (Art. 29) including indigenous rights criteria | Indigenous rights governance score feeds into country risk classification |
| Human rights adverse impacts in supply chain | Indigenous rights violations as adverse human rights impacts | Rights violation monitoring and alert system |
| Third-party audit requirement (Art. 6) | No explicit third-party audit requirement in EUDR, but Art. 11 allows independent surveys | Independent FPIC verification recommendation for high-risk scenarios |
| OECD alignment | EUDR is not explicitly OECD-aligned but follows similar principles | Agent implements OECD DDG-compatible process for indigenous rights |
| Responsible sourcing | Deforestation-free AND legally compliant sourcing | Indigenous rights compliance as component of legal compliance |

**Key Lesson:** The Conflict Minerals Regulation's OECD-aligned five-step due diligence framework provides a more structured and internationally recognized approach than the EUDR's three-step process. The Indigenous Rights Checker agent should implement a process that is compatible with both the EUDR three-step model and the OECD five-step model, enabling future alignment with emerging international due diligence standards.

### 9.3 EU Corporate Sustainability Due Diligence Directive (CSDDD, 2024/1760)

While not a predecessor, the CSDDD creates overlapping obligations that the agent must prepare for:

| CSDDD Requirement | EUDR Overlap | Agent Future-Proofing |
|-------------------|-------------|----------------------|
| Art. 5: Identify actual and potential adverse human rights impacts | Art. 10: Risk assessment including indigenous rights | Territory overlap detection and rights violation monitoring are dual-use |
| Art. 7: Prevent potential adverse impacts | Art. 11: Risk mitigation | FPIC verification and community engagement are dual-use |
| Art. 8: Bring actual adverse impacts to an end | Not explicit in EUDR | Remediation tracking module (future) |
| Art. 9: Complaints mechanism | Not explicit in EUDR | Grievance mechanism already included in agent design |
| Art. 22: Civil liability | Not in EUDR | Documentation and evidence preservation critical for legal defense |

---

## 10. Best Practices from Certification Schemes

### 10.1 FSC (Forest Stewardship Council)

FSC Principle 3 (Indigenous Peoples' Rights) and FSC Principle 4 (Community Relations) provide the most detailed FPIC requirements in voluntary certification:

| FSC Requirement | Reference | Agent Implementation |
|----------------|-----------|---------------------|
| Identify indigenous peoples affected by management activities | Criterion 3.1 | Indigenous community registry; territory overlap detection |
| Recognize and uphold indigenous peoples' legal and customary rights of ownership, use, and management | Criterion 3.2 | Land rights database; customary rights tracking; ownership verification |
| Identify sites of special cultural, ecological, economic, or religious significance | Criterion 3.3 | Sacred site database; cultural heritage overlay in spatial analysis |
| Obtain FPIC through binding agreements before management activities | Criterion 3.6 | FPIC workflow management; agreement tracking; consent documentation |
| Protect indigenous peoples' rights to protect and utilize traditional knowledge | Criterion 3.4 | Traditional knowledge protection provisions in agreements |
| Compensate indigenous peoples for use of traditional knowledge | Criterion 3.5 | Benefit-sharing agreement tracking; payment records |
| Establish permanent indigenous peoples' committee | FSC PIPC | Community governance structure recognition in consultation tracking |

**FSC FPIC Guidelines (FSC-GUI-30-003 V2.0):**

FSC has published detailed FPIC implementation guidelines that provide operational specificity beyond the EUDR text. Key elements adoptable by the agent:

1. **Identification Phase:** Map all potentially affected indigenous peoples using government registries, community self-identification, anthropological studies, and stakeholder engagement.
2. **Assessment Phase:** Determine the nature and extent of impacts on indigenous rights, territories, resources, and cultural heritage.
3. **Negotiation Phase:** Engage in good faith negotiations with legitimate community representatives, providing full information and allowing adequate time.
4. **Agreement Phase:** Document consent (or refusal) in written agreements specifying rights, obligations, benefit-sharing, grievance mechanisms, and review periods.
5. **Monitoring Phase:** Ongoing monitoring of agreement compliance, community satisfaction, and changed circumstances.

### 10.2 RSPO (Roundtable on Sustainable Palm Oil)

RSPO Principle 3 (Respect for Community and Human Rights and Delivery of Benefits) and the RSPO FPIC Guide (2015) establish requirements for palm oil operations:

| RSPO Requirement | Reference | Agent Implementation |
|-----------------|-----------|---------------------|
| Conduct social and environmental impact assessment before new planting | Criterion 7.1 | Impact assessment verification as part of FPIC documentation |
| Negotiate with affected communities for compensation and consent | Criterion 7.5 | FPIC negotiation tracking; compensation records |
| Document FPIC process including consent or refusal | Criterion 7.6 | FPIC documentation verification engine |
| Maintain up-to-date maps of community land use | Criterion 7.3 | Community land use mapping integration with territory database |
| Establish participatory mapping with communities | Criterion 2.3 | Community mapping records in territory database |
| Implement complaints and grievance mechanism | Criterion 1.3 | Grievance mechanism module |
| No new planting on indigenous lands without FPIC | Criterion 7.5 | Strict enforcement of FPIC requirement for palm oil plots overlapping indigenous territories |

### 10.3 Rainforest Alliance (2020 Sustainable Agriculture Standard)

Rainforest Alliance Requirements 5.8.1 and 5.8.2 mandate FPIC implementation:

| RA Requirement | Reference | Agent Implementation |
|---------------|-----------|---------------------|
| Implement FPIC process for activities diminishing land or resource use rights of indigenous peoples and local communities | Requirement 5.8.1 | FPIC workflow management; triggering based on territory overlap detection |
| Identify indigenous peoples and local communities with potential interests, rights, claims | Guidance T, Step 1 | Community registry; stakeholder identification |
| Identify rights, claims, or interests to land or resources | Guidance T, Step 2 | Land rights database; claims registry |
| Identify sites and resources of cultural, archaeological, historical significance | Guidance T, Step 3 | Cultural heritage site database; spatial overlay |
| Disclose full information about proposed activities and potential impacts | Guidance T, Step 4 | Information disclosure verification in FPIC documentation |
| Conduct participatory mapping and land use assessment | Guidance T, Step 5 | Community mapping records; land use documentation |
| Carry out social and environmental impact assessment | Guidance T, Step 6 | Impact assessment documentation verification |
| Negotiate agreements including mitigation measures, benefits, and conflict resolution | Guidance T, Step 7 | Agreement tracking; benefit-sharing records; grievance mechanism |
| Document FPIC process and outcomes | Guidance T, Step 8 | Full documentation audit trail |
| Monitor implementation and resolve grievances | Guidance T, Step 9 | Ongoing monitoring module; grievance tracking |

### 10.4 Certification Alignment Matrix

The agent should verify alignment with certification scheme requirements to support dual compliance (EUDR + certification):

| Requirement Area | EUDR | FSC | RSPO | Rainforest Alliance | Agent Coverage |
|-----------------|------|-----|------|---------------------|---------------|
| Indigenous territory identification | Art. 10(2)(c) | Criterion 3.1 | Criterion 2.3 | Requirement 5.8.1 | Territory database + overlap detection |
| FPIC before operations | Art. 2(40)(g) | Criterion 3.6 | Criterion 7.5 | Requirement 5.8.2 | FPIC workflow management |
| Good faith consultation | Art. 10(2)(d) | Criterion 3.6 | Criterion 7.6 | Guidance T | Consultation scoring engine |
| Land rights recognition | Art. 2(40)(a),(d) | Criterion 3.2 | Criterion 7.3 | Requirement 5.8.1 | Land rights verification |
| Benefit sharing | Not explicit | Criterion 3.5 | Criterion 7.5 | Guidance T, Step 7 | Benefit-sharing tracking |
| Grievance mechanism | Not explicit (CSDDD Art. 9) | Criterion 1.6 | Criterion 1.3 | Requirement 5.8.2 | Grievance module |
| Cultural heritage protection | Art. 2(40)(f) implied | Criterion 3.3 | Criterion 7.4 | Guidance T, Step 3 | Cultural site database |
| Documentation and records | Art. 31 (5-year retention) | Annual audit | Annual audit | Annual audit | Complete audit trail |

---

## 11. Technical Specifications for Indigenous Rights Checker Agent

### 11.1 Engine Architecture (9 Engines)

| Engine | Name | Core Function | Key Operations |
|--------|------|---------------|----------------|
| **F1** | Indigenous Territory Database Engine | Manage consolidated spatial database of indigenous territories from 6+ authoritative sources | Territory CRUD; source ingestion; spatial indexing; version management; legal status tracking |
| **F2** | FPIC Documentation Verification Engine | Score and verify FPIC documentation completeness and quality | Document checklist validation; quality scoring; temporal verification; authentication integration |
| **F3** | Land Rights Overlap Detector | Detect and classify spatial overlaps between production plots and indigenous territories | PostGIS intersection; buffer zone analysis; proximity scoring; multi-polygon handling; batch screening |
| **F4** | Community Consultation Tracker | Manage multi-stage community engagement workflows | Stage-gate enforcement; timeline tracking; SLA monitoring; record management; escalation rules |
| **F5** | Rights Violation Alert System | Monitor external sources for indigenous rights violation reports and correlate with supply chains | Source monitoring (10+); NLP-based report parsing; spatial correlation; severity classification; alert dispatch |
| **F6** | Indigenous Community Registry | Maintain structured database of indigenous communities in commodity-producing regions | Community CRUD; governance structure records; representative organizations; population demographics; language data |
| **F7** | FPIC Workflow Manager | Manage the complete FPIC lifecycle from identification through monitoring | 6-stage workflow; template management; deadline tracking; approval workflows; renewal scheduling |
| **F8** | Compliance Report Generator | Generate audit-ready indigenous rights compliance documentation | PDF/JSON/HTML report generation; multi-language (EN, FR, DE, ES, PT); DDS integration; certification alignment |
| **F9** | Country Indigenous Rights Profiler | Maintain and score country-level indigenous rights governance profiles | Legal framework database; enforcement scoring; territory recognition tracking; integration with EUDR-016 |

### 11.2 Database Schema Requirements

| Table | Purpose | Key Columns | Relations |
|-------|---------|-------------|-----------|
| `gl_irc_indigenous_territories` | Store georeferenced indigenous territory boundaries | id, territory_name, community_id, country_code, legal_status, boundary_geom (PostGIS), area_hectares, source, source_id, recognition_date, version | FK to communities |
| `gl_irc_indigenous_communities` | Indigenous community registry | id, community_name, country_code, population, language_primary, governance_structure, representative_org, legal_recognition_status, contact_channel | FK to territories |
| `gl_irc_plot_territory_overlaps` | Cached results of plot-territory overlap analysis | id, plot_id, territory_id, overlap_category (DIRECT/PARTIAL/ADJACENT/PROXIMATE/NONE), overlap_area_hectares, overlap_percentage, distance_km, detection_date | FK to territories, FK to plots |
| `gl_irc_fpic_processes` | FPIC workflow instances | id, plot_id, territory_id, community_id, current_stage, stage_started_at, consent_status, consent_date, agreement_id, assignee, created_at | FK to territories, communities |
| `gl_irc_fpic_documents` | FPIC documentation records | id, fpic_process_id, document_type, document_hash, upload_date, language, verified, verification_date, quality_score | FK to fpic_processes |
| `gl_irc_consultation_records` | Individual consultation meeting/event records | id, fpic_process_id, consultation_date, location, attendees_count, representative_names, minutes_hash, community_response, follow_up_actions | FK to fpic_processes |
| `gl_irc_rights_violations` | Rights violation reports from monitoring sources | id, source_name, source_url, report_date, country_code, region, violation_type, severity, affected_community_id, coordinates, correlated_supply_chain | FK to communities |
| `gl_irc_grievances` | Community grievance records | id, community_id, complainant_type, complaint_date, complaint_text, status, investigation_date, resolution_date, resolution_description | FK to communities |
| `gl_irc_benefit_sharing` | Benefit-sharing agreement tracking | id, fpic_process_id, agreement_type, agreement_date, value_committed, value_disbursed, disbursement_schedule, last_disbursement, compliance_status | FK to fpic_processes |
| `gl_irc_country_profiles` | Country-level indigenous rights governance profiles | id, country_code, ilo_169_ratified, ilo_169_year, constitutional_recognition, fpic_legislation, indigenous_affairs_agency, legal_framework_score, enforcement_score, territory_recognition_score, fpic_implementation_score, judicial_access_score, composite_score, assessment_date | Unique per country |
| `gl_irc_legal_frameworks` | Per-country indigenous rights legal framework details | id, country_code, law_name, law_type, enactment_date, fpic_provisions, land_rights_provisions, enforcement_mechanism, judicial_access, notes | FK to country_profiles |
| `gl_irc_audit_log` | Immutable audit trail for all operations | id, timestamp, operation, entity_type, entity_id, actor, action_detail, provenance_hash | Append-only |

### 11.3 API Endpoint Specifications

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/api/v1/eudr/irc/territories/overlap` | POST | Detect territory overlaps for a plot | Plot coordinates or polygon | Overlap results with category, area, affected communities |
| `/api/v1/eudr/irc/territories/batch-overlap` | POST | Batch overlap screening for multiple plots | List of plot coordinates/polygons (max 10,000) | Batch overlap results |
| `/api/v1/eudr/irc/territories/{territory_id}` | GET | Retrieve territory details | Territory ID | Territory boundary, community info, legal status |
| `/api/v1/eudr/irc/communities/{community_id}` | GET | Retrieve community information | Community ID | Community demographics, governance, contact |
| `/api/v1/eudr/irc/fpic/verify` | POST | Verify FPIC documentation completeness | FPIC documentation package | FPIC score (0-100), gaps, recommendations |
| `/api/v1/eudr/irc/fpic/process` | POST | Create new FPIC workflow instance | Plot ID, territory ID, community ID | FPIC process ID, workflow stages, deadlines |
| `/api/v1/eudr/irc/fpic/process/{process_id}` | GET/PUT | Get or advance FPIC workflow | Process ID, stage transition data | Current stage, next steps, SLA status |
| `/api/v1/eudr/irc/consultation/good-faith` | POST | Assess good faith consultation quality | Consultation records and evidence | Good faith score (0-100), criterion breakdown |
| `/api/v1/eudr/irc/violations/screen` | POST | Screen for rights violations in a region | Country code, coordinates, radius | Violation reports, severity, affected communities |
| `/api/v1/eudr/irc/violations/alerts` | GET | Get active violation alerts for operator | Operator ID, filters | Alert list with severity, correlation, recommendations |
| `/api/v1/eudr/irc/country/{country_code}/profile` | GET | Get country indigenous rights profile | Country code | Legal framework, scores, territory data, active conflicts |
| `/api/v1/eudr/irc/country/{country_code}/score` | GET | Get country indigenous rights governance score | Country code | Composite score, factor breakdown, classification |
| `/api/v1/eudr/irc/report/compliance` | POST | Generate indigenous rights compliance report | Scope (supply chain, country, supplier) | PDF/JSON/HTML compliance report |
| `/api/v1/eudr/irc/grievance` | POST/GET | Submit or query grievance records | Grievance data or query filters | Grievance ID, status, investigation progress |
| `/api/v1/eudr/irc/benefit-sharing/{agreement_id}` | GET | Get benefit-sharing agreement status | Agreement ID | Agreement details, disbursement status, compliance |
| `/api/v1/eudr/irc/health` | GET | Agent health and readiness check | None | Service status, data freshness, source connectivity |

### 11.4 Performance Requirements

| Requirement | Target | Measurement |
|-------------|--------|-------------|
| Single plot overlap query | < 500ms p99 | Latency under load with 50,000+ territories indexed |
| Batch overlap screening (10,000 plots) | < 5 minutes | Total processing time for batch job |
| FPIC documentation scoring | < 100ms p99 | Per-document-set scoring latency |
| Compliance report generation | < 10 seconds | Time from request to PDF delivery |
| Standard API response | < 200ms p95 | GET endpoint latency |
| Territory database ingestion | < 30 minutes for full refresh | Time to ingest and index all territory data from all sources |
| Violation alert correlation | < 5 seconds per alert | Time to correlate violation report with supply chain locations |
| Concurrent users | 100+ simultaneous | Without degradation beyond p95 targets |

### 11.5 Zero-Hallucination Requirements

All calculations and scores produced by the Indigenous Rights Checker agent must be:

1. **Deterministic**: Same inputs always produce same outputs, bit-perfect reproducible
2. **Database-sourced**: All data comes from verified databases and authoritative sources, not generated
3. **Auditable**: Every score traceable to specific data points and calculation steps
4. **No LLM in critical path**: Core scoring, overlap detection, and classification use deterministic algorithms only
5. **Provenance tracked**: SHA-256 hashes for all data inputs and calculation outputs
6. **Version controlled**: All territory data, legal framework data, and scoring parameters versioned
7. **Immutable audit log**: All operations logged with timestamp, actor, and provenance hash

---

## 12. Data Sources and Integration Architecture

### 12.1 External Data Sources

| Source | Type | Data Elements | Update Frequency | Access Method | License |
|--------|------|---------------|-------------------|---------------|---------|
| LandMark | Territory boundaries | Polygons, legal status, community data for 1.15M+ areas | Continuous | WMS/WFS/GeoJSON API | Open (CC BY-SA) |
| RAISG | Amazon territories | Indigenous territories, protected areas in 9 countries | Annual | Shapefile download | Open |
| FUNAI (Brazil) | Brazilian territories | Terras Indigenas boundaries and status | Continuous | Shapefile/API | Government open data |
| BPN/AMAN (Indonesia) | Indonesian territories | Adat territory claims and government maps | Periodic | Shapefile | Research/advocacy |
| ACHPR (Africa) | African territories | Indigenous peoples and tribal territories | Periodic | Reports + GIS | Open |
| WDPA | Protected areas | Global protected areas (overlap with indigenous territories) | Monthly | API/Download | Open |
| ILO NORMLEX | Convention ratifications | ILO 169 ratification status by country | Continuous | Web/API | Open |
| World Bank WGI | Governance indicators | 6 governance dimensions for 200+ countries | Annual | API/Download | Open |
| TI CPI | Corruption index | CPI scores for 180+ countries | Annual | API/Download | Open |
| IWGIA | Indigenous rights reports | Annual Indigenous World report; country assessments | Annual | Publication | Open (reports) |
| Forest Peoples Programme | Rights monitoring | Rights violation reports, case studies | Continuous | Web | Open (reports) |
| Cultural Survival | Indigenous news | Rights violation reports, advocacy alerts | Continuous | RSS/Web | Open |
| Amnesty International | Human rights | Country-specific human rights reports | Continuous | API/Web | Open (reports) |
| National legislation databases | Legal frameworks | Indigenous rights laws, FPIC legislation | Varies | Web/Manual | Open |

### 12.2 Internal Integration Points

| GreenLang Component | Integration | Data Flow Direction |
|--------------------|-------------|---------------------|
| EUDR-001 Supply Chain Mapping Master | Supply chain graph enrichment with indigenous rights flags | Bidirectional |
| EUDR-002 Geolocation Verification | Validated plot coordinates for overlap analysis | IRC <-- EUDR-002 |
| EUDR-006 Plot Boundary Manager | Plot polygon data for spatial intersection queries | IRC <-- EUDR-006 |
| EUDR-012 Document Authentication | FPIC document authenticity verification | IRC --> EUDR-012 |
| EUDR-016 Country Risk Evaluator | Indigenous rights governance score for composite risk | IRC --> EUDR-016 |
| EUDR-017 Supplier Risk Scorer | FPIC compliance status for supplier risk adjustment | IRC --> EUDR-017 |
| EUDR-020 Deforestation Alert System | Correlation of deforestation alerts with indigenous territories | IRC <-- EUDR-020 |
| GL-EUDR-APP | Indigenous rights section in DDS; compliance dashboard | IRC --> APP |
| SEC-005 Audit Logging | Immutable audit trail for all indigenous rights operations | IRC --> SEC-005 |
| SEC-002 RBAC | Permission control for sensitive community data | IRC <-- SEC-002 |
| SEC-011 PII Detection | Protection of indigenous community personal data | IRC --> SEC-011 |

---

## 13. Regulatory Compliance Matrix

### 13.1 Complete EUDR Article Coverage

| EUDR Article | Provision | Agent Feature | Coverage |
|-------------|-----------|---------------|----------|
| Art. 2(28) | Definition of "relevant legislation" -- land use rights | Territory database; land rights verification | FULL |
| Art. 2(28) | Definition of "relevant legislation" -- third-party rights | Indigenous peoples as third parties with property rights | FULL |
| Art. 2(28) | Definition of "relevant legislation" -- human rights under international law | UNDRIP, ILO 169, ICCPR integration | FULL |
| Art. 2(28) | Definition of "relevant legislation" -- FPIC (UNDRIP) | FPIC verification engine; workflow management | FULL |
| Art. 3(b) | Legality requirement -- compliance with relevant legislation | Comprehensive legality check including indigenous rights | FULL |
| Art. 8 | Due diligence system | Three-step indigenous rights verification process | FULL |
| Art. 9(1)(d) | Geolocation of production plots | Spatial overlap analysis with territory boundaries | FULL |
| Art. 9(1)(f) | Verification of relevant legislation compliance | Evidence collection and verification for indigenous rights laws | FULL |
| Art. 10(2)(a) | Country risk assignment per Art. 29 | Indigenous rights governance score | FULL |
| Art. 10(2)(c) | Presence of indigenous peoples | Indigenous territory database; presence detection | FULL |
| Art. 10(2)(d) | Good faith consultation with indigenous peoples | Consultation scoring engine; workflow tracking | FULL |
| Art. 10(2)(e) | Duly reasoned claims by indigenous peoples | Land claims registry; dispute tracking | FULL |
| Art. 10(2)(i) | Human rights violations; lack of law enforcement | Rights violation monitoring; enforcement gap assessment | FULL |
| Art. 11 | Risk mitigation measures | Mitigation recommendations; FPIC remediation workflow | FULL |
| Art. 13 | Simplified due diligence for low-risk | Reduced verification for countries with strong indigenous rights protection | FULL |
| Art. 25 | Penalties for non-compliance | Risk flagging and violation severity classification | FULL |
| Art. 29(4)(a) | Information from indigenous peoples and civil society | Community and NGO report integration | FULL |
| Art. 29(4)(d) | Laws protecting indigenous rights -- existence, compliance, enforcement | Country indigenous rights governance score (3 dimensions) | FULL |
| Art. 31 | 5-year record retention | Immutable audit trail; document retention management | FULL |

### 13.2 International Framework Coverage

| Framework | Key Articles/Provisions | Agent Implementation | Coverage |
|-----------|------------------------|---------------------|----------|
| UNDRIP Art. 10 | No forcible removal; FPIC for relocation | Displacement check; relocation agreement verification | FULL |
| UNDRIP Art. 19 | FPIC before legislation affecting indigenous peoples | National legislative consultation verification | PARTIAL (dependent on country data) |
| UNDRIP Art. 26 | Rights to traditionally occupied lands | Territory database; historical occupation evidence | FULL |
| UNDRIP Art. 28 | Right to redress for lands taken without FPIC | Restitution claim tracking; compensation records | FULL |
| UNDRIP Art. 32(2) | FPIC for projects affecting lands and resources | Core FPIC verification engine | FULL |
| ILO 169 Art. 6 | Consultation through appropriate procedures | Consultation process verification; representative participation | FULL |
| ILO 169 Art. 14 | Land ownership and possession recognition | Territory legal status tracking; title verification | FULL |
| ILO 169 Art. 15 | Natural resource rights safeguarding | Resource rights assessment; exploitation consent verification | FULL |
| ILO 169 Art. 16 | Non-removal from occupied lands | Displacement monitoring; FPIC for relocation | FULL |

### 13.3 Certification Scheme Alignment

| Certification | FPIC Standard | Principle/Criterion | Agent Alignment |
|--------------|---------------|---------------------|-----------------|
| FSC | FSC-STD-01-001 V5-2 | Principle 3 (Indigenous Peoples' Rights) | FULL -- territory, FPIC, cultural heritage, benefits |
| FSC | FSC-GUI-30-003 V2.0 | FPIC Guidelines | FULL -- 5-phase FPIC process mapped to workflow |
| RSPO | RSPO P&C 2018/2024 | Principle 3, Criterion 7.5-7.6 | FULL -- FPIC, community consent, grievance |
| Rainforest Alliance | SA-S-SD-1-V1.3 | Requirements 5.8.1-5.8.2 | FULL -- FPIC process, Guidance T steps 1-9 |
| PEFC | PEFC ST 1003:2018 | Requirement 5.6 (Indigenous rights) | PARTIAL -- basic FPIC verification |
| ISCC | ISCC EU 202 | Principle 1 (Biomass legality) | PARTIAL -- legality check including land rights |

---

## Appendices

### Appendix A: ILO Convention 169 Ratification Status (Complete)

As of 2026, the following 24 countries have ratified ILO Convention 169:

| # | Country | Ratification Date | In Force Since | Major EUDR Commodities |
|---|---------|-------------------|---------------|----------------------|
| 1 | Argentina | 2000-07-03 | 2001-07-03 | Soya |
| 2 | Bolivia | 1991-12-11 | 1992-12-11 | Soya, Coffee, Wood |
| 3 | Brazil | 2002-07-25 | 2003-07-25 | Soya, Cattle, Coffee, Wood |
| 4 | Central African Republic | 2010-08-30 | 2011-08-30 | Wood, Cocoa |
| 5 | Chile | 2008-09-15 | 2009-09-15 | Wood |
| 6 | Colombia | 1991-03-07 | 1992-03-07 | Coffee, Palm Oil, Cocoa |
| 7 | Costa Rica | 1993-04-02 | 1994-04-02 | Coffee |
| 8 | Denmark | 1996-02-22 | 1997-02-22 | (Greenland governance) |
| 9 | Dominica | 2002-06-25 | 2003-06-25 | -- |
| 10 | Ecuador | 1998-05-15 | 1999-05-15 | Cocoa, Coffee, Palm Oil |
| 11 | Fiji | 1998-03-03 | 1999-03-03 | -- |
| 12 | Guatemala | 1996-06-05 | 1997-06-05 | Coffee, Palm Oil, Rubber |
| 13 | Honduras | 1995-03-28 | 1996-03-28 | Coffee, Palm Oil |
| 14 | Luxembourg | 2018-06-05 | 2019-06-05 | (EU governance) |
| 15 | Mexico | 1990-09-05 | 1991-09-05 | Coffee |
| 16 | Nepal | 2007-09-14 | 2008-09-14 | -- |
| 17 | Netherlands | 1998-02-02 | 1999-02-02 | (Suriname governance) |
| 18 | Nicaragua | 2010-08-25 | 2011-08-25 | Coffee |
| 19 | Norway | 1990-06-19 | 1991-06-19 | (Sami governance) |
| 20 | Paraguay | 1993-08-10 | 1994-08-10 | Soya, Cattle |
| 21 | Peru | 1994-02-02 | 1995-02-02 | Coffee, Cocoa, Wood |
| 22 | Spain | 2007-02-15 | 2008-02-15 | (EU governance) |
| 23 | Venezuela | 2002-05-22 | 2003-05-22 | Cocoa, Coffee |
| 24 | Germany | 2021-06-15 | 2022-06-15 | (EU governance) |

### Appendix B: Key EUDR Recitals on Indigenous Rights

| Recital | Content Summary | Agent Relevance |
|---------|----------------|-----------------|
| **Recital 17** | References the EU Charter of Fundamental Rights and the right to an effective remedy | Supports community grievance mechanism requirement |
| **Recital 29** | Defines the scope of "relevant legislation" including land tenure, human rights, and FPIC | Foundational definition for indigenous rights compliance |
| **Recital 30** | References UNDRIP and the rights of indigenous peoples as framework for FPIC | Anchors FPIC obligations to UNDRIP standard |
| **Recital 31** | Emphasizes the importance of respecting indigenous peoples' rights in the context of deforestation | Establishes policy intent for indigenous rights protection |
| **Recital 32** | Recognizes that indigenous peoples and local communities play a critical role in forest conservation; references UNDRIP and ILO Convention 169 | Key recital establishing the full international framework context |
| **Recital 34** | Discusses partnerships with producer countries that strengthen indigenous and community rights | Supports capacity building and institutional strengthening requirements |
| **Recital 56** | References the Commission's assessment of human rights impacts in the context of country benchmarking | Supports indigenous rights criteria in Article 29 benchmarking |

### Appendix C: Glossary of Key Terms

| Term | Definition | Source |
|------|-----------|--------|
| FPIC | Free, Prior and Informed Consent -- the right of indigenous peoples to give or withhold consent to actions that affect their lands, territories, or resources | UNDRIP Art. 32(2) |
| Indigenous Peoples | Peoples who have a historical continuity with pre-invasion and pre-colonial societies, consider themselves distinct, and are determined to preserve their territories | ILO 169 Art. 1 |
| Customary Tenure | Rights to land and resources arising from long-standing customary practice, regardless of formal legal title | Various (VGGT, ILO 169) |
| Relevant Legislation | Laws applicable in the country of production, including land rights, human rights, FPIC, and environmental laws | EUDR Art. 2(40) |
| Due Diligence | The process of information gathering, risk assessment, and risk mitigation required before placing commodities on the EU market | EUDR Art. 8 |
| Good Faith | Genuine intention to negotiate and reach agreement; absence of coercion or bad faith tactics | ILO 169 Art. 6(2) |
| Duly Reasoned Claims | Claims supported by objective and verifiable information | EUDR Art. 10(2)(e) |
| Territory | Geographic area traditionally occupied, used, or claimed by an indigenous people | UNDRIP Art. 26 |
| Demarcation | The process of officially delineating the boundaries of an indigenous territory | National law (varies) |
| Competent Authority | Member State authority designated to enforce EUDR compliance | EUDR Art. 14 |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-10 | GL-RegulatoryIntelligence | Initial comprehensive regulatory requirements document |

---

## Sources and References

1. [Regulation (EU) 2023/1115 -- Official Text (EUR-Lex)](https://eur-lex.europa.eu/eli/reg/2023/1115/oj/eng)
2. [EUDR Consolidated Text (December 2024)](https://eur-lex.europa.eu/eli/reg/2023/1115/2024-12-26/eng)
3. [EUDR: How companies can ensure Indigenous rights are protected (FoodNavigator)](https://www.foodnavigator.com/Article/2025/09/24/eudr-navigating-indigenous-rights/)
4. [Out of the Woods? Human Rights and the New EU Regulation on Deforestation (Verfassungsblog)](https://verfassungsblog.de/out-of-the-woods/)
5. [Meeting the EUDR legality requirement: Human rights (ISEAL Alliance)](https://isealalliance.org/sustainability-news/meeting-eudr-legality-requirement-human-rights-and-role-voluntary)
6. [Open Letter: Ensuring EUDR benchmarking reflects human rights (Coffee Watch)](https://coffeewatch.org/updates/open-letter-ensuring-eudr-benchmarking/)
7. [OHCHR: Consultation and FPIC](https://www.ohchr.org/en/indigenous-peoples/consultation-and-free-prior-and-informed-consent-fpic)
8. [ILO Convention 169 Text](https://www.ilo.org/media/458651/download)
9. [FAO: Free Prior and Informed Consent](https://www.fao.org/indigenous-peoples/pillars-of-work/free--prior-and-informed-consent/en)
10. [LandMark Global Platform](https://www.landmarkmap.org/data/)
11. [RAISG Maps](https://www.raisg.org/en/maps/)
12. [FSC: Indigenous Peoples](https://fsc.org/en/with-indigenous-peoples)
13. [FSC FPIC Guidelines](https://fsc.org/en/document-centre/documents/retrieve/e3adfb1d-f2ed-4e36-a171-6864c96f0d76)
14. [RSPO FPIC Standards 2024](https://rspo.org/rspo-reiterates-fpic-and-components-to-address-deforestation-remain-key-criteria-in-2024-rspo-standards/)
15. [Rainforest Alliance FPIC Guidance](https://knowledge.rainforest-alliance.org/docs/guidance-t-free-prior-and-informed-consent-fpic-processes)
16. [Discussion Paper on EUDR and Land Rights (Zero Deforestation Hub)](https://zerodeforestationhub.eu/wp-content/uploads/2025/01/SAFE_Land-Rights-Paper.pdf)
17. [EU Timber Regulation (Wikipedia)](https://en.wikipedia.org/wiki/European_Union_Timber_Regulation)
18. [EU Conflict Minerals Regulation 2017/821 (EUR-Lex)](https://eur-lex.europa.eu/eli/reg/2017/821/oj/eng)
19. [Rainforest Foundation Norway: EUDR Risk Benchmarking Input](https://dv719tqmsuwvb.cloudfront.net/documents/EUDR-Risk-Benchmarking-input-to-Commission-1.pdf)
20. [EUDR Penalties: What Happens If You Don't Comply](https://eudr.co/eudr-penalties/)
21. [Understanding EUDR: Global Alliance of Impact Lawyers](https://gailnet.org/understanding-the-eu-deforestation-regulation-eudr-what-legally-savvy-audiences-worldwide-need-to-know/)
22. [Preferred by Nature EUDR Indicators](https://www.preferredbynature.org/sites/default/files/Sustainability-Framework-EUDR-Indicators(V1.0).pdf)
23. [EC Country Benchmarks Publication (Preferred by Nature)](https://www.preferredbynature.org/news/european-commission-publishes-first-list-country-benchmarks-under-eu-deforestation-regulation)
24. [IWGIA: EU Engagement with Indigenous Issues](https://iwgia.org/en/european-union-engagement-with-indigenous-issues/5408-iw-2024-eu.html)
25. [Earthsight: 40 Organisations Call for Robust EUDR Benchmarking](https://www.earthsight.org.uk/news/benchmarking-open-letter)
