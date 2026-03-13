# Regulatory Requirements Document: EUDR Protected Area Compliance

## REG-EUDR-022 -- Protected Area Validator Agent

| Field | Value |
|-------|-------|
| **Document ID** | REG-EUDR-022 |
| **Agent ID** | GL-EUDR-PAV-022 |
| **Component** | Protected Area Validator Agent |
| **Category** | EUDR Regulatory Agent -- Environmental Compliance & Biodiversity Conservation |
| **Author** | GL-RegulatoryIntelligence |
| **Date** | 2026-03-10 |
| **Status** | Complete |
| **Classification** | Regulatory Intelligence -- Internal |
| **Primary Regulation** | Regulation (EU) 2023/1115 (EUDR) |
| **Secondary Frameworks** | CBD Kunming-Montreal GBF, CITES, World Heritage Convention, Ramsar Convention, EU Habitats Directive 92/43/EEC, EU Birds Directive 2009/147/EC, EU Biodiversity Strategy 2030, National Protected Area Laws (Brazil SNUC, Indonesia Forest Law, Malaysia National Forestry Act, DRC Forest Code) |

---

## Table of Contents

1. [EUDR Articles on Protected Areas](#1-eudr-articles-on-protected-areas)
2. [Protected Area Categories (IUCN I-VI)](#2-protected-area-categories-iucn-i-vi)
3. [Key Protected Area Databases and Data Sources](#3-key-protected-area-databases-and-data-sources)
4. [Spatial Validation Requirements](#4-spatial-validation-requirements)
5. [International Conservation Frameworks](#5-international-conservation-frameworks)
6. [Deforestation-Protected Area Correlation](#6-deforestation-protected-area-correlation)
7. [Country-Specific Protected Area Laws](#7-country-specific-protected-area-laws)
8. [Certification Scheme Requirements](#8-certification-scheme-requirements)
9. [EU-Specific Protected Area Regulations](#9-eu-specific-protected-area-regulations)
10. [Enforcement and Penalties](#10-enforcement-and-penalties)
11. [Technical Specifications for Protected Area Validator Agent](#11-technical-specifications-for-protected-area-validator-agent)
12. [Data Sources and Integration Architecture](#12-data-sources-and-integration-architecture)
13. [Regulatory Compliance Matrix](#13-regulatory-compliance-matrix)

---

## 1. EUDR Articles on Protected Areas

### 1.1 Article 2 -- Definitions (Points Relevant to Protected Areas)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 2.

Protected areas are not defined as a standalone concept in the EUDR. Instead, they are incorporated through several intersecting definitions that collectively create binding compliance obligations when production plots overlap with or adjoin protected areas.

**Article 2(1) -- "Deforestation":** Defined as the conversion of forest to agricultural use, whether human-induced or not, after 31 December 2020. Protected forests -- those within national parks, nature reserves, and other legally designated conservation areas -- are forests under this definition. Any conversion of a protected forest to agricultural use constitutes deforestation for EUDR purposes, regardless of whether the national government authorized the conversion.

**Article 2(2) -- "Forest degradation":** Defined as structural changes to forest cover, taking the form of the conversion of (a) primary forests or (b) naturally regenerating forests into plantation forests or into other wooded land, and (c) the conversion of primary forests into planted forests. Degradation of forests within protected areas -- such as selective logging in a national park buffer zone that converts primary forest to plantation -- triggers this definition.

**Article 2(3) -- "Forest":** Land spanning more than 0.5 hectares with trees higher than 5 metres and a canopy cover of more than 10%, or trees able to reach those thresholds in situ, excluding land that is predominantly under agricultural or urban land use. All IUCN Category I-VI protected areas that meet this threshold are "forests" under the EUDR, including those managed for sustainable use (IUCN Category VI).

**Article 2(40) -- "Relevant legislation":** Defined as the laws applicable in the country of production concerning the legal status of the area of production in terms of eight categories. The categories directly relevant to protected areas are:

| Category | EUDR Art. 2(40) | Protected Area Relevance |
|----------|-----------------|--------------------------|
| **(a)** Land use rights | Rules on land tenure, right of use and ownership | Protected area designation restricts land use rights; production within protected areas may be illegal under national law |
| **(b)** Environmental protection | Conservation, biodiversity, environmental laws | **PRIMARY** -- National protected area laws are environmental protection legislation; production in or near protected areas must comply with these laws |
| **(c)** Forest-related rules | Forest management, biodiversity conservation, where directly related to wood harvesting | Protected forest designations (forest reserves, protection forests) create binding harvesting restrictions |
| **(d)** Third-party rights | Rights of third parties to property and resources | Conservation agencies and local communities may hold rights over protected area resources |
| **(f)** Human rights protected under international law | International human rights obligations | Environmental rights are increasingly recognized as human rights (UNHRC Resolution 48/13) |

**Critical Implementation Note:** Because Article 2(40)(b) includes "environmental protection" laws, and virtually every country has national legislation designating and protecting conservation areas, the EUDR effectively requires operators to verify that commodity production complies with protected area legislation in every country of production. Production within a legally designated protected area, in violation of that area's management plan or legal restrictions, renders the product non-compliant with Article 3(b) ("produced in accordance with the relevant legislation of the country of production"), regardless of whether deforestation occurred.

**Agent Implementation:** The Protected Area Validator must maintain a per-country database of protected area legal frameworks, tracking:
- National protected area legislation and amendments
- Protected area categories and their legal restrictions
- Permitted and prohibited activities per protected area designation
- Buffer zone regulations and management plans
- Enforcement agency jurisdiction and contact information
- Recent judicial decisions interpreting protected area boundaries

### 1.2 Article 3 -- Prohibition (Legality Requirement Applied to Protected Areas)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 3.

Article 3 establishes that relevant commodities and products may only be placed on the EU market or exported if they are:
- **(a)** Deforestation-free
- **(b)** Produced in accordance with the relevant legislation of the country of production
- **(c)** Covered by a due diligence statement

For protected areas, Article 3(b) is the primary compliance anchor. Production of EUDR-regulated commodities within a protected area in violation of that area's legal restrictions renders the product non-compliant, even if no deforestation occurred. For example:

| Scenario | Deforestation-Free? | Legal Production? | EUDR Compliant? |
|----------|--------------------|--------------------|-----------------|
| Cattle ranching inside a Brazilian Biological Reserve (SNUC Full Protection) | Yes (no forest conversion) | **No** (prohibited activity) | **NON-COMPLIANT** |
| Palm oil plantation inside an Indonesian Protection Forest (Hutan Lindung) | No (forest converted) | **No** (prohibited land use) | **NON-COMPLIANT** |
| Coffee cultivation in a buffer zone of a Guatemalan Biosphere Reserve, authorized under management plan | Yes | Yes (authorized by management plan) | COMPLIANT |
| Cocoa farming inside a Ghanaian Forest Reserve without permit | Yes | **No** (unauthorized encroachment) | **NON-COMPLIANT** |
| Soya production adjacent to (but outside) a Brazilian National Park | Yes | Yes | COMPLIANT (monitor for encroachment) |

**Agent Implementation:** The agent must:
- Perform spatial overlap analysis between production plots and protected area boundaries
- Determine whether the detected overlap constitutes a legal violation based on the protected area's management category and applicable national law
- Distinguish between absolute prohibitions (no production allowed, e.g., IUCN Ia/Ib) and conditional permissions (production may be allowed under management plan, e.g., IUCN V/VI)
- Generate a compliance determination: COMPLIANT, NON-COMPLIANT, or REQUIRES_VERIFICATION

### 1.3 Article 9 -- Information Requirements (Protected Area Data Collection)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 9.

Article 9 specifies the information operators must collect as part of due diligence. The following requirements are relevant to protected area validation:

| Article 9 Provision | Protected Area Application |
|---------------------|---------------------------|
| Art. 9(1)(d): Geolocation of all plots of land | Enables spatial overlap analysis between production plots and protected area boundaries |
| Art. 9(1)(e): Date or time range of production | Enables temporal verification: was production occurring before or after protected area designation? |
| Art. 9(1)(f): Verification that relevant legislation was complied with | Requires evidence of compliance with protected area legislation in the country of production |
| Art. 9(1)(g): Information enabling risk assessment under Article 10 | Must include protected area proximity data as part of risk factor assessment |

**Agent Implementation:** The agent must collect and validate:
- Plot geolocation (coordinates or polygon) for protected area overlap analysis
- Production date range for temporal verification against protected area designation dates
- Protected area designation status for all areas within configurable detection radius (default: 50 km)
- Management plan documentation where production occurs within or adjacent to protected areas
- Permits or authorizations for any production activities within protected area buffer zones

### 1.4 Article 10 -- Risk Assessment (Protected Area as Risk Factor)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 10, paragraphs 1-2.

Article 10(2) enumerates the criteria that operators must consider in their risk assessment. While "protected areas" are not explicitly listed as a standalone criterion, multiple criteria directly encompass protected area considerations:

| Article 10(2) Criterion | Text Summary | Protected Area Relevance |
|--------------------------|-------------|--------------------------|
| **Art. 10(2)(a)** | Assignment of risk to the country of production per Article 29 | Country risk scores should incorporate protected area coverage, enforcement effectiveness, and encroachment rates |
| **Art. 10(2)(b)** | Presence of forests in the country of production or parts thereof | Protected forests constitute the highest-value forest assets; their presence elevates risk |
| **Art. 10(2)(f)** | Prevalence of deforestation or forest degradation in the country of production or parts thereof, including the concentration thereof on specific relevant commodities | Deforestation within or adjacent to protected areas is an indicator of governance failure and elevated risk for all nearby production |
| **Art. 10(2)(g)** | Concerns about the country of production related to corruption, prevalence of document falsification, lack of law enforcement | Weak enforcement of protected area legislation is a key risk indicator |
| **Art. 10(2)(h)** | Complexity of the supply chain, including risk of circumvention or mixing with unknown-origin commodities | Products from protected area encroachment may be mixed into legitimate supply chains |

**Recital 36 (Contextual Interpretation):** Recital 36 of the EUDR states that the risk assessment should take into account, inter alia, "the rate of deforestation or forest degradation" and "environmental protection." While recitals are not legally binding, they provide interpretive guidance that reinforces the inclusion of protected area status as a material risk factor.

**Agent Implementation:** The agent must:
- Integrate protected area proximity as a weighted factor in risk assessment scoring
- Apply a risk multiplier for production plots within or adjacent to protected areas (recommended: 1.5x for adjacent, 2.0x for within)
- Consider protected area enforcement effectiveness as a governance indicator
- Track deforestation within protected areas as an independent risk signal
- Flag supply chain nodes where products may have originated from protected area encroachment

### 1.5 Article 11 -- Risk Mitigation (Enhanced Due Diligence for Protected Area Proximity)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 11.

When risk assessment under Article 10 identifies a non-negligible risk (including risks related to protected area proximity), Article 11 requires operators to take adequate and proportionate risk mitigation measures.

| Risk Level | Protected Area Scenario | Required Mitigation Measures |
|------------|------------------------|------------------------------|
| **CRITICAL** | Production plot overlaps with IUCN Ia/Ib/II protected area | Immediate sourcing suspension; independent field verification; legal compliance review; regulatory notification consideration |
| **HIGH** | Production plot overlaps with IUCN III-VI protected area | Request management plan authorization documentation; commission independent verification; verify permitted activities under management plan |
| **HIGH** | Production plot within buffer zone of any protected area | Request buffer zone permit documentation; verify compliance with buffer zone management plan; enhanced satellite monitoring |
| **MEDIUM** | Production plot within 5 km of protected area boundary | Enhanced monitoring; periodic satellite review; supplier attestation of boundary compliance; annual audit |
| **LOW** | Production plot within 25 km of protected area | Standard monitoring; inclusion in quarterly protected area compliance report; flag for awareness |
| **NEGLIGIBLE** | No protected area within 50 km | Standard due diligence; no additional protected area measures required |

Article 11 also specifies that operators must:
- Request additional information or documentation from suppliers regarding protected area compliance
- Carry out independent surveys or audits to verify boundaries are not being encroached
- Undertake satellite monitoring to detect boundary encroachment over time
- Not place the product on the market if protected area non-compliance risk cannot be mitigated to negligible

**Agent Implementation:** The agent must:
- Classify protected area risk as negligible, low, medium, high, or critical based on spatial analysis
- Generate specific mitigation recommendations based on IUCN category and national legal framework
- Track mitigation actions and their outcomes with SLA enforcement
- Enforce a "no-go" decision for confirmed protected area violations
- Maintain an immutable audit trail of all protected area compliance decisions

### 1.6 Article 29 -- Country Benchmarking (Protected Area Governance Criteria)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 29, paragraphs 2-4.

Article 29 mandates the European Commission to classify countries or parts thereof as low, standard, or high risk. Protected area governance factors into multiple benchmarking criteria:

**Article 29(3) -- Primary Criteria:**

| Criterion | Text | Protected Area Relevance |
|-----------|------|--------------------------|
| Art. 29(3)(a) | Rate of deforestation and forest degradation | Deforestation rates within protected areas are a stronger governance failure indicator than overall national rates |
| Art. 29(3)(b) | Rate of expansion of agricultural land for relevant commodities | Agricultural expansion into protected areas represents the most severe form of land conversion |
| Art. 29(3)(c) | Production trends of relevant commodities | Increasing production in regions with high protected area coverage indicates potential encroachment pressure |

**Article 29(4) -- Additional Assessment Criteria:**

| Criterion | Text | Protected Area Relevance |
|-----------|------|--------------------------|
| Art. 29(4)(a) | Information submitted by countries, indigenous peoples, or civil society | NGO reports on protected area violations feed into country classification |
| Art. 29(4)(c) | Existence of laws aimed at combating deforestation and forest degradation, and their enforcement | **PRIMARY** -- Protected area legislation is the primary legal instrument for combating deforestation; enforcement effectiveness is a key differentiator |
| Art. 29(4)(d) | Existence, compliance with, and enforcement of laws protecting human rights, indigenous peoples' rights, local communities' rights, and customary tenure rights holders | Protected areas often overlap with indigenous territories; intersection of protections |
| Art. 29(4)(e) | Accessibility of transparent data regarding land rights and forest cover | Availability of protected area boundary data, management plans, and monitoring reports indicates governance transparency |

**Country Benchmarking Implications for Protected Areas:**

The Commission's benchmarking methodology should (and industry practice does) consider:
- **Protected area coverage percentage** relative to forest area (indicator of conservation commitment)
- **Protected area deforestation rate** relative to overall deforestation rate (indicator of enforcement effectiveness)
- **Protected area downgrading, downsizing, and degazettement (PADDD) trends** (indicator of political will erosion)
- **Management plan compliance rates** (indicator of institutional capacity)
- **Budget allocation for protected area management** (indicator of resource commitment)

**Agent Implementation:** The agent must:
- Track protected area governance indicators for each country
- Maintain a protected area deforestation index per country
- Monitor PADDD events (downgrading, downsizing, degazettement)
- Feed protected area governance scores into EUDR-016 (Country Risk Evaluator)
- Alert when country benchmarking reclassifications affect protected area risk levels

### 1.7 Article 31 -- Data Retention (Protected Area Records)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 31.

Article 31 requires operators to retain due diligence information for at least five years. For protected area compliance, this includes:
- All protected area overlap analysis results
- Buffer zone compliance determinations
- Management plan authorizations and permits
- Satellite monitoring imagery and change detection results
- Protected area boundary data versions used in analysis (temporal snapshots)
- Compliance determination rationale and supporting evidence

**Agent Implementation:** All protected area validation data, analysis results, and compliance determinations must be stored with 5-year minimum retention, with immutable audit trails and provenance tracking.

---

## 2. Protected Area Categories (IUCN I-VI)

### 2.1 IUCN Protected Area Management Categories Framework

The International Union for Conservation of Nature (IUCN) has developed a globally recognized system of six protected area management categories, adopted by the Convention on Biological Diversity (CBD), the United Nations Environment Programme (UNEP), and national governments worldwide. The WDPA (World Database on Protected Areas) uses IUCN categories as the primary classification system for all protected areas globally.

For EUDR compliance, the IUCN category of a protected area determines:
- The severity of a compliance violation if production overlaps with the protected area
- The permitted activities (if any) under the management plan
- The risk multiplier to apply in deforestation risk scoring
- The buffer zone radius appropriate for monitoring

### 2.2 Category Definitions and EUDR Compliance Implications

| IUCN Category | Name | Definition | Permitted Human Activity | EUDR Production Compatibility | Risk Level if Overlap | Recommended Buffer Zone |
|---------------|------|------------|--------------------------|-------------------------------|----------------------|------------------------|
| **Ia** | Strict Nature Reserve | Areas set aside to protect biodiversity and geological/geomorphological features, where human visitation, use and impacts are strictly controlled and limited | Virtually none; scientific research only with strict controls | **INCOMPATIBLE** -- No commodity production permitted | **CRITICAL** (risk score 100) | 10 km |
| **Ib** | Wilderness Area | Usually large unmodified or slightly modified areas, retaining their natural character, without permanent or significant human habitation | Minimal; traditional indigenous use may be permitted | **INCOMPATIBLE** -- No commodity production permitted | **CRITICAL** (risk score 100) | 10 km |
| **II** | National Park | Large natural or near-natural areas protecting large-scale ecological processes with compatible visitor opportunities | Ecotourism, education, research; no extractive use | **INCOMPATIBLE** -- No commodity production permitted | **CRITICAL** (risk score 95) | 10 km |
| **III** | Natural Monument or Feature | Areas protecting a specific outstanding natural feature (landform, sea mount, cave, living feature) | Limited visitation and research | **INCOMPATIBLE** -- No commodity production on the feature | **HIGH** (risk score 85) | 5 km |
| **IV** | Habitat/Species Management Area | Areas targeting protection of particular species or habitats, with active management intervention | Active management for conservation; some sustainable use may be permitted | **CONDITIONAL** -- Production may be permitted if compatible with conservation objectives and authorized under management plan | **HIGH** (risk score 75) | 5 km |
| **V** | Protected Landscape/Seascape | Areas where long-term interaction between people and nature has produced a distinct landscape with significant ecological, biological, cultural, and scenic value | Agriculture, forestry, and other traditional land uses explicitly anticipated | **CONDITIONAL** -- Production permitted where consistent with landscape character and management plan | **MEDIUM** (risk score 50) | 2 km |
| **VI** | Protected Area with Sustainable Use of Natural Resources | Areas conserving ecosystems together with associated cultural values and traditional natural resource management systems | Low-level, non-industrial sustainable natural resource use is a principal management objective | **CONDITIONAL** -- Sustainable, low-impact commodity production may be compatible where authorized under management plan | **MEDIUM** (risk score 40) | 2 km |

### 2.3 Additional International Protected Area Designations

Beyond IUCN categories, several international designation systems create additional compliance obligations:

| Designation | Governing Convention | Number of Sites (approx.) | Legal Status | EUDR Relevance |
|-------------|---------------------|---------------------------|-------------|----------------|
| **UNESCO World Heritage Site (Natural)** | World Heritage Convention (1972) | 227 natural + 39 mixed sites (2025) | Binding obligation under international law for state parties | Production within a natural World Heritage site boundary constitutes the highest-severity protected area violation; state parties have treaty obligations to protect Outstanding Universal Value |
| **Ramsar Wetland of International Importance** | Ramsar Convention (1971) | 2,531+ sites covering 2.6+ million km2 (2025) | Binding obligation to maintain ecological character | Production that damages the ecological character of a Ramsar site (drainage, pollution, encroachment) violates treaty obligations; wetland conversion for agriculture is a primary threat |
| **UNESCO Biosphere Reserve** | UNESCO MAB Programme (1971) | 759 sites in 136 countries (2025) | Not legally binding under international law, but often recognized in national legislation | Biosphere reserves have three zones: core (no production), buffer (limited use), transition (sustainable development including agriculture); zone-specific compliance required |
| **Key Biodiversity Area (KBA)** | IUCN KBA Standard | 16,500+ sites globally | Not legally binding; scientific identification | KBAs identify areas of global significance for biodiversity; production in unprotected KBAs is not legally prohibited but represents elevated risk for future protected area designation and stakeholder scrutiny |
| **Alliance for Zero Extinction (AZE) Site** | AZE Partnership | 853 sites globally | Not legally binding; scientific identification | AZE sites contain the last remaining habitat for one or more Critically Endangered or Endangered species; production here poses extreme reputational and future regulatory risk |
| **Important Bird and Biodiversity Area (IBA)** | BirdLife International | 13,000+ sites globally | Not legally binding; scientific identification | IBAs are identified using standardized criteria; many overlap with protected areas; production in IBAs triggers enhanced scrutiny from conservation stakeholders |

### 2.4 Risk Scoring Matrix by Protected Area Type

The agent must apply a deterministic risk score based on the protected area designation type:

| Protected Area Type | Base Risk Score (0-100) | Overlap Multiplier | Buffer Zone Risk Decay | No-Go Threshold |
|---------------------|------------------------|--------------------|-----------------------|----------------|
| IUCN Ia (Strict Nature Reserve) | 100 | 3.0x | Linear decay over 10 km | YES -- absolute no-go |
| IUCN Ib (Wilderness Area) | 100 | 3.0x | Linear decay over 10 km | YES -- absolute no-go |
| IUCN II (National Park) | 95 | 2.5x | Linear decay over 10 km | YES -- absolute no-go |
| IUCN III (Natural Monument) | 85 | 2.0x | Linear decay over 5 km | NO -- conditional |
| IUCN IV (Habitat/Species Management) | 75 | 1.8x | Linear decay over 5 km | NO -- conditional |
| IUCN V (Protected Landscape) | 50 | 1.3x | Linear decay over 2 km | NO -- management plan review |
| IUCN VI (Sustainable Use) | 40 | 1.2x | Linear decay over 2 km | NO -- management plan review |
| UNESCO World Heritage (Natural) | 100 | 3.0x | Linear decay over 10 km | YES -- absolute no-go |
| UNESCO World Heritage (Mixed) | 90 | 2.5x | Linear decay over 5 km | YES -- conditional |
| Ramsar Wetland | 85 | 2.0x | Linear decay over 5 km | NO -- conditional |
| UNESCO Biosphere Reserve (Core) | 95 | 2.5x | Linear decay over 10 km | YES -- absolute no-go |
| UNESCO Biosphere Reserve (Buffer) | 60 | 1.5x | Linear decay over 5 km | NO -- conditional |
| UNESCO Biosphere Reserve (Transition) | 30 | 1.1x | Linear decay over 2 km | NO -- management plan review |
| Key Biodiversity Area (unprotected) | 40 | 1.3x | Linear decay over 2 km | NO -- enhanced monitoring |
| AZE Site | 85 | 2.0x | Linear decay over 5 km | NO -- conditional |
| Natura 2000 SAC | 80 | 2.0x | Linear decay over 5 km | NO -- Appropriate Assessment required |
| Natura 2000 SPA | 75 | 1.8x | Linear decay over 5 km | NO -- Appropriate Assessment required |

---

## 3. Key Protected Area Databases and Data Sources

### 3.1 Primary Data Sources

| Source | Provider | Coverage | Data Type | Update Frequency | Access Method | Reliability | License |
|--------|----------|----------|-----------|-------------------|---------------|-------------|---------|
| **WDPA** (World Database on Protected Areas) | UNEP-WCMC / IUCN | Global -- 270,000+ protected areas in 245 countries | Polygons (91%) and points (9%); IUCN category, management authority, designation type, status, area | Monthly updates | Protected Planet API (api.protectedplanet.net); bulk download (Shapefile, GeoJSON); Google Earth Engine (WCMC/WDPA/current/polygons) | **AUTHORITATIVE** -- The globally recognized reference database for protected areas; used by CBD, UN, and national governments | Non-commercial use free; commercial use requires written permission from UNEP-WCMC |
| **Protected Planet** (protectedplanet.net) | UNEP-WCMC | Global | Web interface to WDPA; statistics and reporting tools | Monthly | Web viewer; API; download | **AUTHORITATIVE** -- Official WDPA web platform | Same as WDPA |
| **Natura 2000** | European Environment Agency (EEA) | EU-27 member states | Polygons for SACs and SPAs; species and habitat data; standard data forms | Annual update | EEA Data Hub; Shapefile download; WMS/WFS services | **AUTHORITATIVE** -- Official EU protected area network database | Open data (EU) |
| **Ramsar Sites Information Service (RSIS)** | Ramsar Convention Secretariat | Global -- 2,531+ wetlands | Site boundaries (polygons where available), ecological character descriptions, management information | Continuous (site-by-site updates) | rsis.ramsar.org; bulk download for authorized users | **AUTHORITATIVE** -- Official Ramsar Convention database | Varies; contact Secretariat |
| **UNESCO World Heritage Centre** | UNESCO | Global -- 1,223 properties (2025) | Site boundaries, descriptions, conservation status, Outstanding Universal Value statements | Annual state of conservation reports | whc.unesco.org; spatial data through UNEP-WCMC | **AUTHORITATIVE** -- Official UNESCO database | Open data |
| **MAB Biosphere Reserves Directory** | UNESCO MAB Programme | Global -- 759 sites in 136 countries | Site descriptions, zonation maps, management information | Annual | unesco.org/new/en/natural-sciences/environment/ecological-sciences/biosphere-reserves/ | **AUTHORITATIVE** -- Official UNESCO MAB database | Open data |
| **Key Biodiversity Areas (KBA)** | KBA Partnership (IUCN, BirdLife, et al.) | Global -- 16,500+ sites | Polygons; species trigger information; IUCN Red List status | Continuous | keybiodiversityareas.org; download with registration | **HIGH** -- Standardized scientific criteria | Registration required; citation obligations |
| **Alliance for Zero Extinction (AZE)** | AZE Partnership | Global -- 853 sites | Site polygons; trigger species | Periodic | zeroextinction.org; download available | **HIGH** -- Peer-reviewed scientific criteria | Open data with citation |
| **Global Forest Watch (GFW)** | World Resources Institute | Global | Tree cover loss, fire alerts, protected area overlays | Daily to annual depending on product | globalforestwatch.org; GFW API; Google Earth Engine | **HIGH** -- Multi-source satellite monitoring | Open data (CC BY 4.0) |
| **Important Bird and Biodiversity Areas (IBA)** | BirdLife International | Global -- 13,000+ sites | Site polygons; trigger species and criteria | Periodic | birdlife.org; datazone.birdlife.org | **HIGH** -- Standardized scientific criteria | Registration required |

### 3.2 National Protected Area Data Sources

| Country | Data Source | Agency | Coverage | Format | Update Frequency | Access |
|---------|-----------|---------|----------|--------|-------------------|--------|
| **Brazil** | Cadastro Nacional de Unidades de Conservacao (CNUC) | ICMBio / MMA | All federal, state, and municipal conservation units under SNUC | Shapefile, GeoJSON | Continuous | cnuc.mma.gov.br; SNUC geodata portal |
| **Brazil** | PRODES (deforestation monitoring) | INPE | Legal Amazon deforestation polygons | GeoTIFF, Shapefile | Annual + near-real-time alerts | terrabrasilis.dpi.inpe.br |
| **Indonesia** | One Map Indonesia (Satu Peta) | Geospatial Information Agency (BIG) | National parks, protection forests, conservation forests | Shapefile | Periodic (harmonization ongoing) | tanahair.indonesia.go.id |
| **Indonesia** | Forest Area Moratorium Map | KLHK | Moratorium areas for primary forests and peatlands | Shapefile | Semi-annual | webgis.menlhk.go.id |
| **Malaysia** | Sabah Biodiversity Centre | State agency | Sabah protected areas (parks, wildlife sanctuaries, reserves) | Shapefile | Periodic | sabahbiodiversity.org.my |
| **Malaysia** | Sarawak Forest Department | State agency | Sarawak national parks, nature reserves, wildlife sanctuaries | Shapefile | Periodic | forestry.sarawak.gov.my |
| **DRC** | Institut Congolais pour la Conservation de la Nature (ICCN) | ICCN | National parks, nature reserves, hunting reserves | Shapefile (limited availability) | Irregular | iccnrdc.org |
| **Ghana** | Forestry Commission | FC Ghana | Forest reserves, national parks, wildlife sanctuaries | Shapefile | Periodic | fcghana.org |
| **Cote d'Ivoire** | OIPR (Office Ivoirien des Parcs et Reserves) | OIPR | National parks, nature reserves, classified forests | Shapefile | Periodic | oipr.ci |
| **Colombia** | RUNAP (Registro Unico Nacional de Areas Protegidas) | Parques Nacionales Naturales | All SINAP registered protected areas | Shapefile, GeoJSON | Continuous | runap.parquesnacionales.gov.co |
| **Peru** | SERNANP (Servicio Nacional de Areas Naturales Protegidas) | SERNANP | National protected areas | Shapefile | Continuous | sernanp.gob.pe |

### 3.3 Data Quality and Completeness Assessment

| Data Source | Spatial Completeness | Attribute Completeness | Temporal Currency | Boundary Accuracy | Overall Reliability Score |
|-------------|---------------------|-----------------------|-------------------|-------------------|--------------------------|
| WDPA (Global) | 91% polygon coverage | IUCN category available for ~65% of sites | Monthly updates | Varies (some boundaries approximate) | 85/100 |
| Natura 2000 (EU) | 100% polygon coverage for EU-27 | Complete standard data forms | Annual updates | High (digitized from official maps) | 95/100 |
| CNUC Brazil | 100% for federal; 85%+ for state/municipal | Complete for federal units | Continuous updates | High (official surveyed boundaries) | 90/100 |
| One Map Indonesia | Improving; some spatial conflicts | Ongoing harmonization | Semi-annual | Medium (known discrepancies between datasets) | 70/100 |
| WDPA + National | Combined: best available global + national detail | Varies by country | Best of monthly global + continuous national | Enhanced by national data | 90/100 |

**Agent Implementation:** The agent must:
- Use WDPA as the baseline global protected area layer
- Overlay national protected area datasets where available for enhanced boundary precision
- Maintain data provenance tracking for all protected area boundary sources
- Flag spatial conflicts between global and national datasets for manual resolution
- Track data currency and alert when datasets exceed staleness thresholds (e.g., > 6 months since last update)

---

## 4. Spatial Validation Requirements

### 4.1 Plot-Protected Area Overlap Detection

The core spatial validation requirement is to determine whether a production plot overlaps with, is adjacent to, or is proximate to any protected area. The agent must perform the following spatial operations:

**Overlap Classification Framework:**

| Overlap Category | Definition | Spatial Criteria | Risk Level | Required Action |
|-----------------|------------|-----------------|------------|----------------|
| **DIRECT** | Production plot is entirely within a protected area | Plot polygon completely contained within protected area boundary (ST_Contains) | CRITICAL | Immediate flag; determine IUCN category; check management plan authorization; if unauthorized, flag as non-compliant |
| **PARTIAL** | Production plot partially overlaps with a protected area | Non-zero intersection area between plot and protected area (ST_Intersects AND NOT ST_Contains) | HIGH | Calculate overlap area and percentage; determine IUCN category of overlapping portion; check authorization for overlapping activities |
| **BUFFER** | Production plot is within the regulated buffer zone of a protected area | Plot within protected area buffer zone as defined by national legislation (typically 500m-10km) | HIGH | Verify buffer zone permit; check compliance with buffer zone management plan; monitor for encroachment |
| **ADJACENT** | Production plot is within immediate proximity of a protected area boundary | Plot centroid or nearest boundary point within configurable distance (default: 5 km) from protected area boundary (ST_DWithin) | MEDIUM | Enhanced monitoring; annual satellite review; supplier attestation of boundary compliance |
| **PROXIMATE** | Production plot is within broader detection radius | Plot within extended monitoring radius (default: 25 km) | LOW | Quarterly monitoring; inclusion in protected area compliance report; awareness flagging |
| **NONE** | No protected area within maximum detection radius | No intersection and distance exceeds maximum radius (default: 50 km) | NEGLIGIBLE | Standard due diligence only; no additional protected area measures |

### 4.2 Buffer Zone Monitoring Requirements

Buffer zones around protected areas serve as transition areas that reduce edge effects and protect the ecological integrity of the core protected area. National legislation defines buffer zone widths, permitted activities, and management requirements.

**Buffer Zone Analysis Specifications:**

| Parameter | Specification | Justification |
|-----------|--------------|---------------|
| Buffer zone computation method | Geodesic buffer using ST_Buffer with geography type | Accurate distance computation on curved earth surface for tropical latitudes |
| Buffer zone widths | Configurable per protected area: 500m, 1km, 2km, 5km, 10km, 25km, 50km | National legislation specifies different buffer widths; default configuration based on IUCN category |
| Default buffer widths by IUCN category | Ia/Ib: 10km; II: 10km; III: 5km; IV: 5km; V: 2km; VI: 2km | Higher-category protected areas require larger monitoring buffers |
| Buffer zone resolution | 64-point polygon approximation for circular buffers; exact polygon for legislated buffer zones | Balance between spatial precision and computational performance |
| Multi-ring buffer analysis | Concentric buffers at 1km, 5km, 10km, 25km, 50km from protected area boundary | Enables graduated risk scoring based on proximity |
| Encroachment detection | Compare buffer zone boundary with production plot expansion over time | Detect gradual encroachment through time-series analysis |

### 4.3 No-Go Zone Enforcement

Certain protected area categories constitute absolute no-go zones for commodity production:

| No-Go Category | Protected Areas Included | Legal Basis | Agent Response |
|----------------|------------------------|------------|----------------|
| **ABSOLUTE NO-GO** | IUCN Ia (Strict Nature Reserve); IUCN Ib (Wilderness Area); IUCN II (National Park); UNESCO World Heritage Site (Natural); Biosphere Reserve Core Zone | IUCN guidelines; national legislation; international conventions | Production plot overlap triggers automatic CRITICAL risk; compliance status: NON-COMPLIANT; recommendation: immediate sourcing suspension |
| **CONDITIONAL NO-GO** | IUCN III-IV; Ramsar Wetland; IUCN V-VI without management plan authorization | IUCN guidelines; Ramsar Convention; national legislation | Production plot overlap triggers HIGH risk; requires management plan review and authorization evidence; compliance status: REQUIRES_VERIFICATION |
| **MANAGED ACCESS** | IUCN V-VI with authorized management plan; Biosphere Reserve Transition Zone; nationally designated sustainable use areas | National legislation; management plans | Production plot overlap triggers MEDIUM risk; requires documentation of authorized activity; compliance status: COMPLIANT if authorized |

### 4.4 Geospatial Technical Requirements

| Requirement | Specification | Justification |
|-------------|--------------|---------------|
| Coordinate reference system | WGS84 (EPSG:4326) | EUDR standard; alignment with GPS, WDPA, and satellite data |
| Spatial precision | < 10 meters | Sufficient for plot-protected area boundary discrimination |
| Spatial database | PostGIS 3.4+ with Geography type support | Production-grade spatial operations on WGS84 ellipsoid |
| Spatial index | GiST R-tree index on all geometry/geography columns | Sub-second overlap detection across 270,000+ protected areas |
| Overlap area calculation | Geodetic calculation using ST_Area on geography type | Accurate area calculation for tropical regions |
| Distance calculation | ST_Distance on geography type (geodesic distance) | Accurate distance for buffer zone proximity analysis |
| Topological operations | ST_Intersects, ST_Contains, ST_Within, ST_Buffer, ST_Distance, ST_Area, ST_Intersection, ST_DWithin | Full PostGIS spatial analysis capability |
| Polygon validation | ST_IsValid, ST_MakeValid | Prevent topological errors in overlap calculations |
| Multi-polygon support | MULTIPOLYGON geometry type | Protected areas may have non-contiguous zones |
| Temporal versioning | Point-in-time boundary queries using validity ranges | Protected area boundaries change over time (expansion, degazettement, PADDD) |
| Data retention | 5 years per EUDR Article 31 | Regulatory requirement for due diligence records |
| Query performance | < 500ms per single-plot protected area overlap query | Sub-second response for interactive compliance checking |
| Batch performance | < 2 seconds per 1000-plot batch analysis | Efficient bulk analysis for portfolio screening |

### 4.5 Spatial Analysis Workflow

```
Step 1: PLOT INGESTION
  |-- Receive production plot geometry (point or polygon)
  |-- Validate geometry (WGS84, valid topology, reasonable bounds)
  |-- Convert point to minimum area polygon if needed (4-hectare default)
  |
Step 2: PROTECTED AREA INTERSECTION
  |-- Query WDPA spatial index for all protected areas within 50 km of plot centroid
  |-- Query national protected area overlays for the country of production
  |-- Perform ST_Intersects with each candidate protected area
  |-- Calculate overlap area, percentage, and centroid for each intersection
  |
Step 3: BUFFER ZONE ANALYSIS
  |-- For each non-intersecting protected area within 50 km:
  |     |-- Calculate geodesic distance from plot boundary to protected area boundary
  |     |-- Classify by buffer ring (1km, 5km, 10km, 25km, 50km)
  |     |-- Check against legislated buffer zone width for specific protected area
  |
Step 4: DESIGNATION CLASSIFICATION
  |-- For each intersecting or proximate protected area:
  |     |-- Determine IUCN category (Ia, Ib, II, III, IV, V, VI)
  |     |-- Determine international designations (World Heritage, Ramsar, Biosphere)
  |     |-- Determine national designation (national park, reserve, etc.)
  |     |-- Look up permitted activities under management plan
  |
Step 5: RISK SCORING
  |-- Apply base risk score per protected area type (Section 2.4)
  |-- Apply overlap multiplier (direct: full, partial: proportional, buffer: decayed)
  |-- Apply cumulative risk if multiple protected areas are proximate
  |-- Combine with country governance factor from EUDR-016
  |-- Generate composite protected area risk score (0-100)
  |
Step 6: COMPLIANCE DETERMINATION
  |-- COMPLIANT: No overlap; no buffer zone violation; adequate distance
  |-- NON-COMPLIANT: Overlap with no-go zone; unauthorized activity in protected area
  |-- REQUIRES_VERIFICATION: Overlap with conditional zone; buffer zone activity
  |-- MONITORING_REQUIRED: Proximate to protected area; enhanced surveillance needed
  |
Step 7: REPORT GENERATION
  |-- Generate protected area compliance report with:
  |     |-- All protected areas analyzed (name, IUCN category, designation, distance)
  |     |-- Overlap analysis results (area, percentage, type)
  |     |-- Risk scores and compliance determination
  |     |-- Mitigation recommendations
  |     |-- Data provenance and methodology
```

---

## 5. International Conservation Frameworks

### 5.1 Convention on Biological Diversity (CBD)

**Overview:** The Convention on Biological Diversity (CBD), signed at the 1992 Rio Earth Summit and entered into force on 29 December 1993, is the principal international treaty for biodiversity conservation. As of 2025, 196 parties (195 countries + EU) have ratified the convention.

**Kunming-Montreal Global Biodiversity Framework (GBF):** Adopted at CBD COP15 on 19 December 2022, the GBF establishes the "30x30" target -- Target 3 commits parties to conserve at least 30% of terrestrial, inland water, and of coastal and marine areas by 2030 through ecologically representative, well-connected, and equitably governed systems of protected areas and other effective area-based conservation measures (OECMs).

**EUDR Relevance:** The CBD 30x30 target has direct implications for EUDR compliance:
- Countries expanding protected area networks to meet 30x30 will designate new protected areas in commodity-producing regions
- Operators must monitor for new protected area designations that may affect existing supply chain plots
- Production plots within newly designated protected areas may transition from compliant to non-compliant
- The agent must track CBD National Biodiversity Strategies and Action Plans (NBSAPs) for protected area expansion plans

| CBD Provision | EUDR Connection | Agent Requirement |
|---------------|----------------|-------------------|
| Article 8 (In-situ conservation) | Countries must establish protected area systems | Monitor national protected area expansions |
| Article 8(a) (Protected area systems) | Countries must designate and manage protected areas | Track IUCN categorization and management effectiveness |
| Article 8(j) (Traditional knowledge) | Respect indigenous knowledge in conservation | Integrate with EUDR-021 Indigenous Rights Checker |
| Target 3 (30x30) | 30% terrestrial conservation by 2030 | Monitor upcoming protected area designations in commodity regions |
| Target 7 (Pollution reduction) | Reduce pollution impacts on biodiversity | Monitor agrochemical use near protected areas |

### 5.2 CITES (Convention on International Trade in Endangered Species)

**Overview:** CITES (entered into force 1 July 1975) regulates international trade in specimens of wild animals and plants. 184 parties. CITES controls trade through three appendices: Appendix I (trade prohibited), Appendix II (trade regulated), Appendix III (trade monitored at request of a country).

**EUDR Relevance:** CITES intersects with EUDR protected area compliance where:
- Timber species listed on CITES Appendix I or II are harvested from protected areas or their buffer zones
- Protected areas serve as critical habitat for CITES-listed species
- Trade in commodities from areas critical for CITES-listed species faces enhanced scrutiny

| CITES Connection | Agent Requirement |
|-----------------|-------------------|
| Appendix I species habitat overlap | Flag production plots overlapping with critical habitat for Appendix I species |
| Appendix II timber species | Verify CITES export permits for timber from protected area buffer zones |
| Species distribution data | Integrate IUCN Red List range maps with protected area boundaries |

### 5.3 World Heritage Convention

**Overview:** The Convention Concerning the Protection of the World Cultural and Natural Heritage (1972) obliges state parties to ensure the identification, protection, conservation, presentation, and transmission to future generations of cultural and natural heritage. 195 state parties.

**Natural World Heritage Site criteria (vii-x):**
- **(vii)** Superlative natural phenomena or areas of exceptional natural beauty
- **(viii)** Outstanding examples of major stages of earth's history
- **(ix)** Outstanding examples of ecological and biological processes
- **(x)** Most important natural habitats for in-situ conservation of biological diversity, including threatened species of Outstanding Universal Value

**EUDR Relevance:** Production within a Natural World Heritage Site boundary constitutes the most severe protected area compliance violation:
- State parties have treaty obligations to protect Outstanding Universal Value (OUV)
- UNESCO can place sites on the "World Heritage in Danger" list if OUV is threatened
- Removal from the World Heritage List is possible in extreme cases
- Commodity production within or threatening a World Heritage Site attracts intense international scrutiny

| World Heritage Provision | EUDR Connection | Agent Requirement |
|-------------------------|----------------|-------------------|
| Article 4 (Duty to protect) | Art. 2(40)(b) environmental protection | Absolute no-go zone for all natural World Heritage Sites |
| Article 5 (Conservation measures) | Art. 10(2)(g) environmental law enforcement | Monitor state of conservation reports for supply chain regions |
| Danger Listing | Art. 29(4)(a) information from civil society | Flag supply chains near World Heritage in Danger sites |

### 5.4 Ramsar Convention

**Overview:** The Convention on Wetlands of International Importance (Ramsar, 1971) is the intergovernmental treaty for the conservation and wise use of wetlands. 172 contracting parties. 2,531+ designated Ramsar Sites covering over 2.6 million km2.

**EUDR Relevance:** Ramsar wetlands are particularly relevant to EUDR commodities because:
- Palm oil plantations have historically driven peatland drainage (Indonesia, Malaysia)
- Soya and cattle expansion have destroyed Pantanal and Amazon wetlands (Brazil)
- Wetland conversion is both deforestation (if meeting forest definition) and environmental law violation

| Ramsar Provision | EUDR Connection | Agent Requirement |
|-----------------|----------------|-------------------|
| Article 3(1) (Wise use obligation) | Art. 2(40)(b) environmental protection | Monitor for wetland conversion or degradation near Ramsar sites |
| Article 3(2) (Notification of change) | Art. 10(2)(f) prevalence of degradation | Track Ramsar site condition change notifications |
| Article 4(1) (Conservation of wetlands) | Art. 11 risk mitigation | Enhanced due diligence for production near Ramsar sites |

### 5.5 Framework Integration Matrix

| Framework | Spatial Data Available | API Access | Update Frequency | Integration Priority |
|-----------|----------------------|------------|-------------------|---------------------|
| CBD (Protected Areas) | Via WDPA | Via Protected Planet API | Monthly | P0 -- Core requirement |
| CBD (30x30 tracking) | Via UN Biodiversity Lab | REST API | Periodic | P1 -- Forward monitoring |
| CITES | Species range maps via IUCN Red List | IUCN Red List API | Annual | P2 -- Enhancement |
| World Heritage | Via UNEP-WCMC / WDPA | UNESCO World Heritage API | Annual | P0 -- Core requirement |
| Ramsar | Via RSIS / WDPA | Ramsar RSIS database | Continuous | P0 -- Core requirement |
| Biosphere Reserves | Via UNESCO MAB / WDPA | UNESCO API | Annual | P1 -- Important |

---

## 6. Deforestation-Protected Area Correlation

### 6.1 Protected Area Status as Deforestation Risk Indicator

The relationship between protected area status and deforestation risk is complex and bidirectional:

**Protected areas as risk attenuator:** Effectively managed protected areas have lower deforestation rates than surrounding unprotected forests. Globally, protected forests experience 1.5-5x lower deforestation rates than comparable unprotected forests. This means production plots within well-managed protected area landscapes have lower deforestation risk.

**Protected areas as risk amplifier:** However, deforestation within or adjacent to protected areas is a strong indicator of governance failure, corruption, and weak enforcement. When deforestation does occur in a protected area, it signals:
- Breakdown of law enforcement in the region
- Possible corruption enabling illegal land conversion
- Insufficient management resources
- Political pressure to downgrade or degazette the protected area
- All nearby production is at elevated risk of non-compliance

### 6.2 Deforestation Risk Scoring Adjustments for Protected Areas

| Scenario | Base Deforestation Risk | Protected Area Adjustment | Final Risk Assessment |
|----------|------------------------|--------------------------|----------------------|
| Plot in region with NO deforestation, NO protected areas nearby | Low | No adjustment | Low |
| Plot in region with NO deforestation, NEAR protected area | Low | +15 points (monitoring) | Low-Medium |
| Plot in region with ACTIVE deforestation, NO protected areas nearby | High | No adjustment | High |
| Plot in region with ACTIVE deforestation, NEAR protected area | High | +25 points (elevated concern) | Very High |
| Plot in region with deforestation WITHIN protected area | Very High | +40 points (governance failure) | Critical |
| Plot WITHIN protected area with deforestation occurring | Critical | Maximum score (100) | Critical -- Non-Compliant |

### 6.3 Protected Area Deforestation Monitoring Sources

| Source | Coverage | Metric | Resolution | Frequency | Agent Integration |
|--------|----------|--------|-----------|-----------|-------------------|
| Global Forest Watch (GFW) | Global | Tree cover loss within protected areas | 30m (Landsat-derived) | Annual (Hansen) + Near-real-time (GLAD) | API integration; overlay with WDPA boundaries |
| RADD (Radar Alerts for Detecting Deforestation) | Tropics | SAR-based deforestation alerts within protected areas | 10m (Sentinel-1) | Near-real-time (daily) | API integration; cloud-penetrating detection |
| PRODES / DETER (Brazil) | Legal Amazon | Deforestation polygons within conservation units | 6.25 ha minimum (PRODES); 25 ha minimum (DETER) | Annual (PRODES) + Near-real-time (DETER) | INPE API; intersection with CNUC boundaries |
| GLAD (Global Land Analysis and Discovery) | Global tropics | Tree cover loss alerts | 10m (Sentinel-2) + 30m (Landsat) | Weekly | GFW API; intersection with WDPA |
| VIIRS Fire Alerts | Global | Active fire detections within protected areas | 375m | Near-real-time (< 12 hours) | FIRMS API; fire-protected area intersection |

### 6.4 PADDD (Protected Area Downgrading, Downsizing, and Degazettement) Monitoring

PADDD events -- where protected areas are legally weakened, shrunk, or completely degazetted -- are critical risk indicators for EUDR compliance because they may signal political pressure to open conservation lands for commodity production.

| PADDD Event Type | Definition | EUDR Risk Implication | Agent Response |
|-----------------|-----------|----------------------|----------------|
| **Downgrading** | Decrease in legal restrictions on human activities within a protected area | Formerly prohibited activities (farming, ranching) may become legal; compliance status of existing production may change | Re-assess all supply chain plots within downgraded area; update permitted activity database; flag for compliance team review |
| **Downsizing** | Decrease in the legal boundary of a protected area | Areas previously protected become available for legal production; but may indicate governance weakness | Re-assess plots formerly inside now-excised areas; flag potential governance risk; update spatial boundaries |
| **Degazettement** | Complete legal revocation of protected area status | Former protected area loses all legal protections; but indicates severe governance weakness | Critical alert for all supply chains in region; maximum governance risk adjustment; update spatial database |

**Data Source for PADDD:** PADDDtracker.org (Conservation International) maintains a global database of PADDD events. The agent must integrate this data source and monitor for new events.

---

## 7. Country-Specific Protected Area Laws

### 7.1 Brazil -- SNUC (National System of Conservation Units)

**Legal Framework:** Law No. 9.985/2000 (Sistema Nacional de Unidades de Conservacao -- SNUC) establishes the Brazilian protected area system. This is the most directly relevant national framework for EUDR compliance given Brazil's position as the world's largest exporter of soya, coffee, and beef -- all EUDR-regulated commodities.

**SNUC Categories:**

| Group | Category | Brazilian Portuguese | Permitted Commodity Production | EUDR Risk Level |
|-------|----------|---------------------|-------------------------------|----------------|
| **Full Protection (Protecao Integral)** | Ecological Station (ESEC) | Estacao Ecologica | **NONE** -- Scientific research only | CRITICAL |
| | Biological Reserve (REBIO) | Reserva Biologica | **NONE** -- No human visitation except educational | CRITICAL |
| | National Park (PARNA) | Parque Nacional | **NONE** -- Ecotourism and education only | CRITICAL |
| | Natural Monument (MONA) | Monumento Natural | **NONE** on public land; limited on private within boundaries | HIGH |
| | Wildlife Refuge (REVIS) | Refugio de Vida Silvestre | **NONE** on public land; limited on private within boundaries | HIGH |
| **Sustainable Use (Uso Sustentavel)** | Environmental Protection Area (APA) | Area de Protecao Ambiental | **YES** -- Subject to zoning and management plan | MEDIUM |
| | Area of Relevant Ecological Interest (ARIE) | Area de Relevante Interesse Ecologico | **LIMITED** -- Small areas; usage restrictions apply | MEDIUM |
| | National Forest (FLONA) | Floresta Nacional | **YES** -- Sustainable forest management; research | LOW-MEDIUM |
| | Extractive Reserve (RESEX) | Reserva Extrativista | **YES** -- Traditional extractivism by resident communities | LOW |
| | Sustainable Development Reserve (RDS) | Reserva de Desenvolvimento Sustentavel | **YES** -- Traditional resource use by resident communities | LOW |
| | Private Natural Heritage Reserve (RPPN) | Reserva Particular do Patrimonio Natural | **NONE** -- Conservation, research, ecotourism only | HIGH |

**Additional Brazilian Protected Area Types:**
- **Indigenous Lands (Terras Indigenas):** Not SNUC conservation units but legally protected; managed by FUNAI; production prohibited without FPIC (see EUDR-021)
- **Quilombola Territories:** Protected under Article 68 ADCT (Constitution); land use restrictions apply
- **Permanent Preservation Areas (APP):** Not conservation units; protected under Forest Code (Law 12.651/2012); include riparian buffers, steep slopes, hilltops, mangroves
- **Legal Reserves (RL):** Not conservation units; rural property must maintain 20-80% native vegetation under Forest Code

**Agent Implementation for Brazil:**
- Integrate CNUC spatial data for all federal, state, and municipal conservation units
- Integrate FUNAI indigenous territory boundaries (cross-reference with EUDR-021)
- Classify each conservation unit by SNUC category and determine production compatibility
- Check CAR (Cadastro Ambiental Rural) for Legal Reserve and APP compliance
- Monitor PRODES/DETER for deforestation within and adjacent to conservation units

### 7.2 Indonesia -- Forest Zone Classification

**Legal Framework:** Forestry Law No. 41/1999, Government Regulation No. 6/2007, and Ministerial Decree No. 327/2024 govern Indonesian forest zone classification. Indonesia is the world's largest palm oil producer and a major EUDR commodity source.

**Forest Zone Categories:**

| Zone | Indonesian Name | Description | Permitted Commodity Production | EUDR Risk Level |
|------|----------------|-------------|-------------------------------|----------------|
| **Conservation Forest** | Hutan Konservasi | National parks, nature reserves, wildlife sanctuaries, hunting parks, nature recreation parks | **NONE** -- All extractive use prohibited | CRITICAL |
| **Protection Forest** | Hutan Lindung | Forests designated for watershed protection, flood prevention, erosion control | **NONE** -- No conversion or harvesting | HIGH |
| **Production Forest (Limited)** | Hutan Produksi Terbatas (HPT) | Forests where limited logging is permitted under concession | **LIMITED** -- Sustainable timber harvesting with concession permit | MEDIUM |
| **Production Forest (Regular)** | Hutan Produksi (HP) | Forests allocated for timber and other forest product extraction | **YES** -- With concession permit; conversion possible through reclassification | LOW-MEDIUM |
| **Production Forest (Convertible)** | Hutan Produksi Konversi (HPK) | Forests that may be legally converted to other land uses | **YES** -- Conversion legally authorized (but may still constitute deforestation under EUDR if post-2020) | CONDITIONAL |
| **Other Land Use (APL)** | Areal Penggunaan Lain | Non-forest state land; private land | **YES** -- Subject to land use permits | LOW |

**Key Indonesian Regulatory Considerations:**
- **Moratorium on Primary Forest and Peatland Clearing:** Presidential Instruction (INPRES) provides ongoing moratorium on new permits in primary forests and peatlands; agent must check moratorium map overlay
- **One Map Policy:** Indonesia's effort to harmonize conflicting forest maps; the agent must track the latest harmonized map version
- **EUDR Discrepancy:** The Indonesian government has identified discrepancies between its forest maps and those used by the EU for EUDR reference; the agent must flag parcels where Indonesian and EU baseline data conflict

**Agent Implementation for Indonesia:**
- Integrate One Map Indonesia spatial data for all forest zone classifications
- Apply moratorium map overlay for primary forest and peatland protections
- Cross-reference with KLHK (Ministry of Environment and Forestry) concession maps
- Flag discrepancies between Indonesian and EU forest baseline definitions
- Monitor for reclassification of HPK (convertible production forest) to non-forest use

### 7.3 Malaysia -- Federal and State Protected Area Systems

**Legal Framework:** Malaysia's protected area system is divided between federal and state jurisdiction, which creates compliance complexity:
- **Peninsular Malaysia:** National Parks Act 1980; Wildlife Conservation Act 2010; National Forestry Act 1984
- **Sabah:** Forest Enactment 1968; Wildlife Conservation Enactment 1997; Parks Enactment 1984
- **Sarawak:** National Parks and Nature Reserves Ordinance 1998; Wild Life Protection Ordinance 1998; Forests Ordinance 2015

**Protected Area Categories:**

| Jurisdiction | Category | Permitted Production | EUDR Risk Level |
|-------------|----------|---------------------|----------------|
| Federal | National Park (Taman Negara) | **NONE** | CRITICAL |
| Federal | Wildlife Reserve | **NONE** | CRITICAL |
| Sabah | Class I Forest Reserve (Protection) | **NONE** | HIGH |
| Sabah | Class VI Forest Reserve (Virgin Jungle) | **NONE** | CRITICAL |
| Sabah | Class VII Forest Reserve (Wildlife) | **NONE** | CRITICAL |
| Sabah | Wildlife Sanctuary | **NONE** | CRITICAL |
| Sabah | Wildlife Conservation Area | **LIMITED** -- Managed conservation | HIGH |
| Sarawak | National Park | **NONE** | CRITICAL |
| Sarawak | Nature Reserve | **NONE** | CRITICAL |
| Sarawak | Wildlife Sanctuary | **NONE** | CRITICAL |
| All | Permanent Reserved Forest (Production) | **YES** -- Sustainable timber with license | MEDIUM |

**Heart of Borneo (HoB):** The 2007 Heart of Borneo Declaration commits Indonesia, Malaysia, and Brunei to conserve 22 million hectares of tropical forest. While not a legally binding protected area designation, HoB areas overlap with many formal protected areas and represent elevated scrutiny zones for EUDR compliance.

### 7.4 Democratic Republic of Congo (DRC) -- Protected Area Network

**Legal Framework:** Forest Code (Law 011/2002); Nature Conservation Law (Law 14/003 of 2014); various presidential and ministerial decrees establishing individual protected areas.

**Protected Area Categories:**

| Category | French Name | Permitted Production | EUDR Risk Level |
|----------|------------|---------------------|----------------|
| National Park | Parc National | **NONE** | CRITICAL |
| Nature Reserve | Reserve Naturelle | **NONE** | CRITICAL |
| Biosphere Reserve | Reserve de Biosphere | **CONDITIONAL** -- Zone dependent | MEDIUM-HIGH |
| Hunting Reserve | Domaine de Chasse | **LIMITED** -- No deforestation | MEDIUM |
| Community Reserve | Reserve Communautaire | **CONDITIONAL** -- Community management plan | MEDIUM |

**Key DRC Considerations:**
- The DRC hosts five UNESCO Natural World Heritage Sites (Virunga, Kahuzi-Biega, Garamba, Salonga, Okapi Wildlife Reserve) -- all on the World Heritage in Danger list as of 2025
- Protected areas cover approximately 13.83% of national territory (goal: 17%)
- Enforcement capacity is extremely limited; protected area encroachment is widespread
- The DRC is classified as HIGH risk under EUDR benchmarking for wood and other commodities
- In January 2025, DRC announced plans for the Kivu-Kinshasa Green Corridor -- the world's largest tropical forest reserve

### 7.5 Ghana and Cote d'Ivoire -- West African Cocoa Producing Nations

**Ghana:**
- Forestry Commission manages Forest Reserves (approximately 280 reserves)
- Wildlife Division manages National Parks and Wildlife Sanctuaries
- Key concern: cocoa farming encroachment into forest reserves (illegal under Concessions Act 1962 and Timber Resources Management Act 1998)
- An estimated 7% of Ghana's remaining closed-canopy forest is within designated forest reserves

**Cote d'Ivoire:**
- OIPR (Office Ivoirien des Parcs et Reserves) manages national parks and reserves
- SODEFOR manages classified forests (forets classees)
- Key concern: massive cocoa farming encroachment into classified forests and national parks (especially Tai National Park -- UNESCO World Heritage Site)
- An estimated 40% of Cote d'Ivoire's cocoa is produced in protected areas (2024 estimates)

**Agent Implementation for Ghana/Cote d'Ivoire:**
- Integrate Ghana Forest Reserve boundaries from Forestry Commission
- Integrate Cote d'Ivoire classified forest and national park boundaries from OIPR/SODEFOR
- Apply HIGH risk multiplier for cocoa supply chains in these countries due to known encroachment prevalence
- Flag any production plots within or adjacent to classified forests or forest reserves

---

## 8. Certification Scheme Requirements

### 8.1 FSC (Forest Stewardship Council) Protected Area Requirements

FSC certification (relevant to wood and timber products under EUDR) contains specific protected area requirements:

**FSC Principle 9 -- High Conservation Values (HCV):**

| HCV Category | Definition | Protected Area Relevance | Agent Integration |
|-------------|-----------|--------------------------|-------------------|
| **HCV 1** | Species diversity -- Concentrations of biological diversity including endemic and rare, threatened or endangered species | Protected areas designated for species conservation (IUCN IV, wildlife sanctuaries) | Cross-reference species data with protected area designations |
| **HCV 2** | Landscape-level ecosystems -- Large landscape-level ecosystems and ecosystem mosaics significant at global, regional or national levels | National parks, biosphere reserves, intact forest landscapes | Verify production does not fragment landscape-level ecosystems |
| **HCV 3** | Ecosystems and habitats -- Rare, threatened, or endangered ecosystems, habitats, or refugia | Ramsar wetlands, unique habitats within protected areas | Map production plots against rare ecosystem databases |
| **HCV 4** | Critical ecosystem services -- Areas providing basic services of nature in critical situations (watershed protection, erosion control, fire barriers) | Protection forests, watershed reserves | Verify production does not compromise critical ecosystem services |
| **HCV 5** | Community needs -- Areas fundamental to meeting basic needs of local communities (subsistence, health) | Community-managed protected areas, extractive reserves | Cross-reference with EUDR-021 community rights |
| **HCV 6** | Cultural values -- Areas critical for traditional cultural identity of communities (sacred sites, ancestral lands) | Cultural heritage sites within or adjacent to protected areas | Cross-reference with EUDR-021 indigenous rights |

**FSC and EUDR Alignment:** FSC has developed "FSC EUDR Aligned" provisions that align HCV assessments with EUDR requirements, including protected area mapping and verification.

### 8.2 RSPO (Roundtable on Sustainable Palm Oil) Requirements

RSPO certification (relevant to palm oil under EUDR) has stringent protected area requirements:

**RSPO Principle 7 -- Protection of Natural Ecosystems:**
- **Criterion 7.7:** New plantings since November 2005 shall not replace primary forest or any area required to maintain or enhance one or more High Conservation Values
- **Criterion 7.8:** No new plantings on peat, regardless of depth
- **RSPO HCV/HCS Assessment:** Required before any new plantation development; must identify and map all HCVs including protected areas

**RSPO High Carbon Stock (HCS) Approach:**
- The HCS approach identifies areas of forest land for conservation by classifying vegetation into six strata from High Density Forest to Open Land
- All protected areas are automatically classified as "Go" areas for conservation (no development)
- Buffer zones around HCS conservation areas must be maintained

**Agent Integration:** Cross-reference RSPO certified concession boundaries with protected area databases to verify compliance.

### 8.3 Rainforest Alliance Requirements

Rainforest Alliance certification (relevant to cocoa, coffee, and other EUDR commodities):

**Rainforest Alliance 2020 Standard (updated 2023):**
- **Critical Criterion 4.1.1:** No deforestation or destruction of natural ecosystems
- **Criterion 4.2:** Protection of natural ecosystems including protected areas, HCV areas, and riparian buffers
- **Criterion 4.2.3:** Buffer zones around water bodies, steep slopes, and protected areas must be maintained
- **Strengthened HCV Protection:** Rainforest Alliance has enhanced requirements for HCV identification, management, and monitoring, aligning with FSC and RSPO approaches

**Agent Integration:** Cross-reference Rainforest Alliance certified farm boundaries with protected area databases.

### 8.4 Certification-Protected Area Compliance Matrix

| Certification | Protected Area Mapping Required? | HCV Assessment Required? | Buffer Zone Standard | No-Go Zones Defined? | EUDR Alignment Status |
|--------------|--------------------------------|--------------------------|---------------------|---------------------|----------------------|
| FSC | Yes (Principle 9) | Yes (HCV 1-6) | Varies by national standard | Yes (HCV areas) | FSC EUDR Aligned (2025) |
| RSPO | Yes (P&C 7) | Yes (HCV/HCS) | Yes (riparian, peat, HCS) | Yes (primary forest, peat) | RSPO Due Diligence Guidance (2024) |
| Rainforest Alliance | Yes (Standard 4.2) | Yes (critical and management criteria) | Yes (water, slopes, PA) | Yes (natural ecosystems post-2014) | RA EUDR Ready Program (2025) |
| PEFC | Partial (sustainable forest management) | Recommended | National standards vary | Varies | Under development |
| ISCC (bioenergy) | Partial (HCV screening) | Yes for new areas | Varies | Yes (no-go areas) | Partial alignment |

---

## 9. EU-Specific Protected Area Regulations

### 9.1 Natura 2000 Network

**Legal Basis:** EU Habitats Directive (92/43/EEC) and EU Birds Directive (2009/147/EC).

**Overview:** Natura 2000 is the largest coordinated network of protected areas in the world, covering approximately 18% of the EU's land area and more than 8% of its marine territory. It comprises:
- **Special Areas of Conservation (SACs):** Designated under the Habitats Directive for habitats and species listed in Annexes I and II
- **Special Protection Areas (SPAs):** Designated under the Birds Directive for rare, vulnerable, or regularly occurring migratory bird species

**EUDR Relevance for EU-Origin Commodities:**

While the EUDR is primarily focused on tropical commodity imports, it also applies to commodities produced within the EU (particularly wood and cattle). For EU-origin commodities, Natura 2000 compliance is a critical component of the "relevant legislation" requirement under Article 2(40)(b).

| Natura 2000 Requirement | Directive | EUDR Connection | Agent Implementation |
|------------------------|-----------|-----------------|---------------------|
| Article 6(2) Habitats Directive: Avoid deterioration of habitats in SACs | 92/43/EEC | Art. 2(40)(b) environmental protection | Flag EU production plots within or adjacent to SACs |
| Article 6(3) Habitats Directive: Appropriate Assessment for plans/projects likely to have significant effects | 92/43/EEC | Art. 10 risk assessment | Require evidence of Appropriate Assessment for production activities near SACs/SPAs |
| Article 6(4) Habitats Directive: Imperative reasons of overriding public interest (derogation) | 92/43/EEC | Art. 11 risk mitigation | Verify derogation authorization if production occurs within SAC |
| Article 4 Birds Directive: Classification and protection of SPAs | 2009/147/EC | Art. 2(40)(b) environmental protection | Flag EU production plots within or adjacent to SPAs |

**Spatial Data Access:**
- EEA Natura 2000 dataset: Publicly available as Shapefile, WMS, and WFS from the European Environment Agency Data Hub
- Updated annually; includes all designated SACs and SPAs across EU-27
- Includes standard data forms with species and habitat information

### 9.2 EU Biodiversity Strategy 2030

**Overview:** The EU Biodiversity Strategy for 2030, adopted on 20 May 2020 as part of the European Green Deal, commits the EU to:
- Legally protect at least 30% of EU land area and 30% of EU sea area (extending Natura 2000 and national designations)
- Strictly protect at least one third of protected areas (10% of land, 10% of sea)
- Restore degraded ecosystems through the EU Nature Restoration Law (Regulation (EU) 2024/1674)

**EUDR Connection:** The Biodiversity Strategy creates an expanding protected area landscape within the EU. For EU-origin wood and cattle commodities, operators must anticipate:
- New Natura 2000 site designations reaching 30% terrestrial coverage
- Strict protection zones within existing Natura 2000 where commodity production may be prohibited
- Nature restoration obligations that may restrict production activities in degraded areas

### 9.3 EU Nature Restoration Law

**Legal Basis:** Regulation (EU) 2024/1674 on nature restoration (entered into force August 2024).

**EUDR Connection:**
- Member States must submit National Restoration Plans identifying restoration areas
- Restoration areas may overlap with or be adjacent to commodity production plots
- Production activities incompatible with restoration objectives may be restricted
- Agent must monitor National Restoration Plan publications and spatial designations

### 9.4 EU Forest Strategy for 2030

**Overview:** The New EU Forest Strategy for 2030, adopted on 16 July 2021, includes commitments to:
- Protect EU primary and old-growth forests
- Promote sustainable forest management
- Enhance forest monitoring through EU Forest Observatory

**EUDR Connection:** EU-origin wood products must be verified against:
- Primary and old-growth forest inventories
- Sustainable forest management requirements
- EU Forest Observatory monitoring data (when available)

---

## 10. Enforcement and Penalties

### 10.1 EUDR Penalty Framework (Article 25)

**Regulatory Text Reference:** Regulation (EU) 2023/1115, Article 25.

Article 25 requires Member States to establish rules on penalties for infringements. The penalty framework applies with enhanced severity when violations involve protected areas:

| Penalty Type | Maximum Level | Protected Area Enhancement |
|-------------|--------------|---------------------------|
| **Financial fines** | Up to 4% of the operator's total annual turnover in the EU | Protected area violations likely to attract maximum fines due to aggravating circumstances (environmental harm to conservation areas) |
| **Revenue confiscation** | Confiscation of revenue from non-compliant products | All revenue from products sourced from protected area encroachment subject to confiscation |
| **Product seizure** | Confiscation of non-compliant commodities and products | Products from protected area sources subject to immediate seizure at EU borders |
| **Public procurement exclusion** | Temporary exclusion (max 12 months) from EU public procurement and funding | Exclusion triggered by protected area violations |
| **Market ban** | Temporary prohibition from placing products on EU market | Possible for repeated or severe protected area violations |
| **Public disclosure** | Publication of identity of infringers on Commission website | Name-and-shame for protected area compliance failures |

### 10.2 Enhanced Due Diligence Failure Penalties

When operators fail to conduct adequate due diligence regarding protected areas, additional penalties may apply:

| Due Diligence Failure | Penalty Implication |
|----------------------|---------------------|
| Failure to check production plot against protected area databases | Negligent non-compliance; financial penalty likely |
| Knowledge of protected area overlap without disclosure | Intentional non-compliance; maximum penalty likely |
| Failure to implement enhanced due diligence for high-risk country protected areas | Non-compliance with Article 11; financial penalty plus possible market restriction |
| Failure to maintain protected area compliance records for 5 years | Article 31 violation; administrative penalty |
| Providing false information about protected area compliance in DDS | Fraud; criminal penalties possible under national law |

### 10.3 National Protected Area Violation Penalties

In addition to EUDR penalties, violations of national protected area laws carry their own penalties in the country of production:

| Country | Protected Area Violation | Maximum Penalty | Criminal Liability |
|---------|-------------------------|-----------------|--------------------|
| Brazil | Unauthorized activity in Full Protection conservation unit | BRL 10,000-50,000,000 fine; imprisonment 1-5 years (Environmental Crimes Law 9.605/1998) | Yes |
| Indonesia | Encroachment in conservation forest | IDR 10 billion fine; imprisonment up to 15 years (Forestry Law 41/1999 Art. 78) | Yes |
| Malaysia (Sabah) | Unauthorized entry/activity in forest reserve | MYR 100,000 fine; imprisonment up to 5 years | Yes |
| DRC | Unauthorized activity in national park | Fine + imprisonment (Nature Conservation Law 14/003) | Yes |
| Ghana | Farming in forest reserve without permit | Fine + imprisonment (Concessions Act; Timber Resources Management Act) | Yes |
| Cote d'Ivoire | Encroachment in classified forest | Fine + imprisonment (Forest Code) | Yes |

**Agent Implementation:** The agent must:
- Maintain a per-country penalty database for protected area violations
- Include penalty exposure estimates in compliance reports
- Factor penalty severity into risk scoring (higher penalties = higher risk of enforcement action = higher risk score)

### 10.4 Reputational and Financial Risk

Beyond regulatory penalties, protected area compliance failures carry significant reputational and financial consequences:

| Risk Type | Description | Potential Impact |
|-----------|-------------|-----------------|
| **Investor divestment** | ESG-focused investors divest from companies linked to protected area destruction | Share price decline; increased cost of capital |
| **Consumer boycott** | Consumer campaigns targeting brands sourcing from protected area regions | Revenue decline; brand damage |
| **NGO campaigns** | Environmental organizations (WWF, Greenpeace, Global Witness) naming companies sourcing from protected areas | Sustained media pressure; regulatory scrutiny |
| **Certification loss** | FSC, RSPO, or Rainforest Alliance decertification for protected area violations | Loss of market access to sustainability-conscious buyers |
| **Insurance implications** | Environmental liability insurance exclusions for protected area damage | Uninsured liability exposure |
| **Litigation risk** | Civil society organizations, indigenous communities, or state prosecutors filing suit | Legal costs; injunctions; damages |

---

## 11. Technical Specifications for Protected Area Validator Agent

### 11.1 Agent Architecture Overview

The Protected Area Validator Agent (GL-EUDR-PAV-022) must implement the following seven processing engines:

| Engine | Name | Purpose | Key Functionality |
|--------|------|---------|-------------------|
| **Engine 1** | Protected Area Database Engine | Manage the consolidated protected area spatial database | WDPA integration; national data overlay; IUCN category mapping; designation tracking; temporal versioning |
| **Engine 2** | Spatial Overlap Detection Engine | Detect plot-protected area overlaps and proximity | PostGIS spatial queries; overlap classification (direct/partial/buffer/adjacent/proximate/none); area calculation |
| **Engine 3** | Buffer Zone Monitoring Engine | Monitor compliance with protected area buffer zones | Multi-ring buffer analysis; encroachment detection; buffer zone permit verification; temporal encroachment tracking |
| **Engine 4** | Risk Scoring Engine | Calculate protected area risk scores | Weighted scoring by IUCN category; overlap multiplier; country governance adjustment; deforestation correlation; cumulative risk |
| **Engine 5** | Compliance Determination Engine | Generate compliance determinations | No-go zone enforcement; management plan authorization verification; legal compliance assessment; determination classification |
| **Engine 6** | PADDD and Designation Monitoring Engine | Track changes in protected area designations | PADDD event monitoring; new designation tracking; boundary change detection; 30x30 expansion monitoring |
| **Engine 7** | Report Generation Engine | Generate protected area compliance reports | PDF/JSON/HTML reports; audit-ready documentation; multi-language support; regulatory submission formatting |

### 11.2 Data Model (High-Level)

```
gl_eudr_pav_protected_areas
  -- protected_area_id (UUID, PK)
  -- wdpa_id (INTEGER, UNIQUE)
  -- name (VARCHAR)
  -- designation_type (VARCHAR)
  -- iucn_category (VARCHAR) -- Ia, Ib, II, III, IV, V, VI, Not_Reported, Not_Applicable
  -- international_designation (VARCHAR) -- WORLD_HERITAGE, RAMSAR, BIOSPHERE, KBA, AZE, NONE
  -- country_iso3 (CHAR(3))
  -- sub_national_region (VARCHAR)
  -- boundary (GEOGRAPHY(MULTIPOLYGON, 4326))
  -- area_km2 (DECIMAL)
  -- management_authority (VARCHAR)
  -- management_plan_status (VARCHAR) -- EXISTS, NOT_EXISTS, UNKNOWN
  -- no_go_status (VARCHAR) -- ABSOLUTE, CONDITIONAL, MANAGED_ACCESS
  -- designation_date (DATE)
  -- last_boundary_update (TIMESTAMPTZ)
  -- data_source (VARCHAR)
  -- metadata (JSONB)
  -- created_at (TIMESTAMPTZ)
  -- updated_at (TIMESTAMPTZ)

gl_eudr_pav_overlap_analyses (hypertable, 30d chunks)
  -- analysis_id (UUID)
  -- analysis_time (TIMESTAMPTZ, PK dimension)
  -- plot_id (UUID, FK)
  -- protected_area_id (UUID, FK)
  -- overlap_category (VARCHAR) -- DIRECT, PARTIAL, BUFFER, ADJACENT, PROXIMATE, NONE
  -- overlap_area_ha (DECIMAL)
  -- overlap_percentage (DECIMAL)
  -- distance_km (DECIMAL)
  -- buffer_ring (VARCHAR) -- 1KM, 5KM, 10KM, 25KM, 50KM
  -- iucn_category (VARCHAR)
  -- base_risk_score (DECIMAL)
  -- adjusted_risk_score (DECIMAL)
  -- compliance_status (VARCHAR) -- COMPLIANT, NON_COMPLIANT, REQUIRES_VERIFICATION, MONITORING_REQUIRED
  -- determination_rationale (TEXT)
  -- data_provenance (JSONB)
  -- tenant_id (UUID)

gl_eudr_pav_buffer_zones
  -- buffer_zone_id (UUID, PK)
  -- protected_area_id (UUID, FK)
  -- buffer_type (VARCHAR) -- LEGISLATED, IUCN_DEFAULT, CUSTOM
  -- buffer_width_km (DECIMAL)
  -- buffer_geometry (GEOGRAPHY(MULTIPOLYGON, 4326))
  -- permitted_activities (JSONB)
  -- legal_basis (VARCHAR)
  -- created_at (TIMESTAMPTZ)

gl_eudr_pav_paddd_events (hypertable, 90d chunks)
  -- event_id (UUID)
  -- event_time (TIMESTAMPTZ, PK dimension)
  -- protected_area_id (UUID, FK)
  -- event_type (VARCHAR) -- DOWNGRADING, DOWNSIZING, DEGAZETTEMENT, UPGRADE, EXPANSION, NEW_DESIGNATION
  -- event_description (TEXT)
  -- area_affected_km2 (DECIMAL)
  -- previous_iucn_category (VARCHAR)
  -- new_iucn_category (VARCHAR)
  -- legal_instrument (VARCHAR)
  -- data_source (VARCHAR)
  -- tenant_id (UUID)

gl_eudr_pav_deforestation_in_pa (hypertable, 30d chunks)
  -- detection_id (UUID)
  -- detection_time (TIMESTAMPTZ, PK dimension)
  -- protected_area_id (UUID, FK)
  -- area_ha (DECIMAL)
  -- source (VARCHAR) -- GFW, RADD, PRODES, GLAD, VIIRS
  -- latitude (DOUBLE PRECISION)
  -- longitude (DOUBLE PRECISION)
  -- confidence (DECIMAL)
  -- country_iso3 (CHAR(3))
  -- tenant_id (UUID)

gl_eudr_pav_compliance_reports
  -- report_id (UUID, PK)
  -- report_time (TIMESTAMPTZ)
  -- plot_id (UUID)
  -- supplier_id (UUID)
  -- commodity_type (VARCHAR)
  -- total_protected_areas_analyzed (INTEGER)
  -- highest_risk_score (DECIMAL)
  -- overall_compliance_status (VARCHAR)
  -- overlap_count (INTEGER)
  -- buffer_violation_count (INTEGER)
  -- no_go_violation (BOOLEAN)
  -- report_format (VARCHAR) -- PDF, JSON, HTML
  -- report_content (JSONB)
  -- provenance_hash (VARCHAR(64))
  -- tenant_id (UUID)
  -- created_at (TIMESTAMPTZ)

gl_eudr_pav_audit_log (hypertable, 30d chunks)
  -- log_id (UUID)
  -- log_time (TIMESTAMPTZ, PK dimension)
  -- operation (VARCHAR)
  -- entity_type (VARCHAR)
  -- entity_id (UUID)
  -- actor_id (UUID)
  -- details (JSONB)
  -- provenance_hash (VARCHAR(64))
  -- tenant_id (UUID)
```

### 11.3 Integration with Existing EUDR Agents

| Agent | Integration Point | Data Flow |
|-------|-------------------|-----------|
| **EUDR-001** Supply Chain Mapping Master | Supply chain node enrichment with protected area risk flags | Outbound: PAV provides protected area risk per supply chain node |
| **EUDR-002** Geolocation Verification | Validated plot coordinates for overlap analysis | Inbound: Verified coordinates feed into spatial analysis |
| **EUDR-004** Forest Cover Analysis | Forest cover data within and adjacent to protected areas | Inbound: Forest cover status enriches protected area monitoring |
| **EUDR-005** Land Use Change Detector | Land use change detection within protected area buffers | Inbound: Change events trigger encroachment alerts |
| **EUDR-006** Plot Boundary Manager | Plot polygon data for spatial intersection analysis | Inbound: Validated plot polygons for overlap detection |
| **EUDR-016** Country Risk Evaluator | Protected area governance scores feed country risk | Outbound: Protected area enforcement index contributes to governance_quality |
| **EUDR-017** Supplier Risk Scorer | Supplier-level protected area risk adjustment | Outbound: Supplier risk adjusted by protected area overlap status |
| **EUDR-018** Commodity Risk Analyzer | Commodity-protected area encroachment patterns | Outbound: Commodity-specific protected area risk profiles |
| **EUDR-020** Deforestation Alert System | Deforestation alerts correlated with protected area boundaries | Bidirectional: Deforestation in protected areas triggers critical alerts; protected area overlay in severity scoring |
| **EUDR-021** Indigenous Rights Checker | Protected area-indigenous territory overlap identification | Bidirectional: Many protected areas overlap with indigenous territories; shared spatial analysis |

### 11.4 Performance Requirements

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Protected area database coverage | 270,000+ areas from WDPA + national overlays | Count of areas in spatial database |
| Single-plot overlap query | < 500ms (p99 latency) | Benchmark against production load |
| Batch plot analysis (1000 plots) | < 2 seconds (p99 latency) | Benchmark against batch processing |
| Protected area database refresh | < 4 hours for full WDPA refresh | Data pipeline SLA monitoring |
| Report generation | < 5 seconds per compliance report | Time from request to delivery |
| Spatial precision | < 10 meters | Cross-validation with surveyed boundaries |
| Determinism | 100% reproducible (zero-hallucination) | Bit-perfect reproducibility tests |
| Data retention | 5 years (per EUDR Article 31) | Retention policy enforcement |

---

## 12. Data Sources and Integration Architecture

### 12.1 Data Source Priority and Refresh Schedule

| Priority | Data Source | Refresh Schedule | Staleness Alert Threshold | Integration Method |
|----------|-----------|------------------|--------------------------|-------------------|
| P0 | WDPA (Protected Planet) | Monthly | 60 days | Protected Planet API + bulk download |
| P0 | National protected area datasets (Brazil CNUC, Indonesia One Map) | As available (minimum quarterly) | 90 days | Agency API + bulk download |
| P0 | Natura 2000 (EEA) | Annual | 18 months | EEA Data Hub download |
| P1 | UNESCO World Heritage boundaries | Annual | 24 months | UNEP-WCMC / WDPA overlay |
| P1 | Ramsar site boundaries | Continuous (per site) | 12 months | RSIS download + WDPA overlay |
| P1 | Key Biodiversity Areas | Continuous | 12 months | KBA Partnership download |
| P2 | PADDDtracker events | Continuous | 30 days | PADDDtracker database + manual monitoring |
| P2 | GFW deforestation in protected areas | Daily to weekly | 14 days | GFW API |
| P2 | VIIRS fire alerts in protected areas | Near-real-time | 24 hours | NASA FIRMS API |

### 12.2 API Endpoints Required

| External API | Purpose | Authentication | Rate Limit | Fallback |
|-------------|---------|---------------|------------|----------|
| Protected Planet API (api.protectedplanet.net) | WDPA protected area boundaries and attributes | API key (free for non-commercial) | 100 requests/minute | Bulk download cache |
| GFW API (data-api.globalforestwatch.org) | Deforestation alerts within protected areas | API key | Rate-limited | Bulk download |
| NASA FIRMS API (firms.modaps.eosdis.nasa.gov) | Fire alerts within protected areas | MAP key | 100 requests/minute | Bulk download |
| EEA API (discomap.eea.europa.eu) | Natura 2000 WMS/WFS | None (public) | Standard | Bulk download cache |
| IUCN Red List API (apiv3.iucnredlist.org) | Species data for CITES/AZE cross-reference | API key | Rate-limited | Bulk download |

### 12.3 Internal API Integration

| Internal Service | Endpoint Pattern | Purpose |
|-----------------|-----------------|---------|
| EUDR-002 Geolocation Verification | /api/v1/eudr/geolocation/validate | Validated plot coordinates for analysis |
| EUDR-006 Plot Boundary Manager | /api/v1/eudr/plots/{plot_id}/boundary | Plot polygon data for intersection |
| EUDR-016 Country Risk Evaluator | /api/v1/eudr/country-risk/{country_iso3} | Country governance scores |
| EUDR-017 Supplier Risk Scorer | /api/v1/eudr/supplier-risk/{supplier_id}/adjust | Supplier risk adjustment |
| EUDR-020 Deforestation Alert System | /api/v1/eudr/deforestation/alerts | Deforestation alerts for correlation |
| EUDR-021 Indigenous Rights Checker | /api/v1/eudr/indigenous/territory-overlap | Indigenous territory data for cross-reference |

---

## 13. Regulatory Compliance Matrix

### 13.1 EUDR Article-to-Feature Mapping

| EUDR Article | Requirement | Agent Feature | Implementation |
|-------------|-------------|---------------|----------------|
| Art. 2(1) | Deforestation definition | Engine 5 (Compliance Determination) | Forest conversion detection within protected areas |
| Art. 2(2) | Forest degradation definition | Engine 4 (Risk Scoring) | Degradation monitoring in protected forests |
| Art. 2(3) | Forest definition (0.5ha, 5m, 10%) | Engine 1 (PA Database) | Forest-classified protected areas identified |
| Art. 2(40)(b) | Relevant legislation: environmental protection | Engine 5 (Compliance Determination) | Protected area law compliance verification |
| Art. 2(40)(c) | Relevant legislation: forest-related rules | Engine 5 (Compliance Determination) | Forest reserve and protection forest compliance |
| Art. 3(b) | Legal production requirement | Engine 5 (Compliance Determination) | Overall protected area legality determination |
| Art. 9(1)(d) | Geolocation of all plots | Engine 2 (Spatial Overlap) | Plot-protected area intersection analysis |
| Art. 9(1)(f) | Verification of legislation compliance | Engine 5 + Engine 7 (Reports) | Documentation of protected area compliance |
| Art. 10(2)(a) | Country risk classification | Engine 4 (Risk Scoring) | Protected area governance in country risk |
| Art. 10(2)(b) | Presence of forests | Engine 1 + Engine 2 | Protected forest identification and mapping |
| Art. 10(2)(f) | Prevalence of deforestation | Engine 4 (Risk Scoring) | Deforestation within protected areas as risk factor |
| Art. 10(2)(g) | Corruption, lack of enforcement | Engine 4 + Engine 6 (PADDD) | Protected area enforcement effectiveness |
| Art. 11 | Risk mitigation measures | Engine 5 (Compliance Determination) | Mitigation recommendations by risk level |
| Art. 25 | Penalties for non-compliance | Engine 7 (Reports) | Penalty exposure estimates in compliance reports |
| Art. 29(3)(a) | Rate of deforestation | Engine 4 (Risk Scoring) | Protected area deforestation rates in benchmarking |
| Art. 29(4)(c) | Laws combating deforestation | Engine 1 + Engine 6 | Protected area legislation tracking and enforcement |
| Art. 31 | Data retention (5 years) | All Engines | 5-year retention for all analysis data |

### 13.2 International Convention Compliance

| Convention | Key Provisions | Agent Coverage | Integration Method |
|-----------|---------------|----------------|-------------------|
| CBD | Article 8 (protected areas); Target 3 (30x30) | WDPA integration; 30x30 expansion monitoring | Engine 1 + Engine 6 |
| CITES | Appendix I-III species habitat | Species range overlay with protected areas | Engine 1 + IUCN Red List API |
| World Heritage Convention | Article 4 (duty to protect OUV) | World Heritage Site no-go enforcement | Engine 1 + Engine 5 |
| Ramsar Convention | Article 3 (wise use); Article 4 (conservation) | Ramsar site overlap detection and monitoring | Engine 1 + Engine 2 |
| Habitats Directive | Article 6 (SAC management) | Natura 2000 SAC compliance | Engine 1 + Engine 5 |
| Birds Directive | Article 4 (SPA protection) | Natura 2000 SPA compliance | Engine 1 + Engine 5 |
| EU Biodiversity Strategy | 30% terrestrial protection by 2030 | Protected area expansion monitoring | Engine 6 |
| EU Nature Restoration Law | National Restoration Plans | Restoration area overlap monitoring | Engine 6 |

### 13.3 Certification Scheme Alignment

| Scheme | Protected Area Requirement | Agent Coverage | Cross-Reference |
|--------|---------------------------|----------------|----------------|
| FSC Principle 9 | HCV 1-6 identification and protection | HCV area overlay with protected areas | Engine 1 + Engine 2 |
| RSPO P&C 7 | No planting in primary forest; HCV/HCS assessment | Protected area no-go enforcement | Engine 5 |
| Rainforest Alliance 4.2 | Protection of natural ecosystems | Ecosystem buffer zone monitoring | Engine 3 |

---

## Sources

- [Regulation (EU) 2023/1115 - EUR-Lex](https://eur-lex.europa.eu/eli/reg/2023/1115/oj/eng)
- [Regulation on Deforestation-free Products - European Commission](https://environment.ec.europa.eu/topics/forests/deforestation/regulation-deforestation-free-products_en)
- [What Is the EU Deforestation Regulation (EUDR)? - World Resources Institute](https://www.wri.org/insights/explain-eu-deforestation-regulation)
- [Unpacking the EUDR Legal Production Requirement - WRI](https://www.wri.org/technical-perspectives/eu-deforestation-regulation-legal-production-requirement)
- [EUDR Explained - FSC](https://fsc.org/en/eudr-explained)
- [IUCN Protected Area Categories - Wikipedia](https://en.wikipedia.org/wiki/IUCN_protected_area_categories)
- [Guidelines for Applying Protected Area Management Categories - IUCN](https://portals.iucn.org/library/sites/library/files/documents/pag-021.pdf)
- [WDPA: World Database on Protected Areas - Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/WCMC_WDPA_current_polygons)
- [Protected Planet API - UNEP-WCMC](https://api.protectedplanet.net)
- [pywdpa - Python WDPA Interface](https://ecology.ghislainv.fr/pywdpa/notebooks/get_started.html)
- [Kunming-Montreal Global Biodiversity Framework - CBD](https://www.cbd.int/gbf)
- [30x30 Implementation Guide - IUCN/WWF](https://iucn.org/sites/default/files/2023-09/30x30-target-framework.pdf)
- [Natura 2000 Spatial Data - European Environment Agency](https://www.eea.europa.eu/data-and-maps/data/natura-14/natura-2000-spatial-data)
- [EU Biodiversity Strategy 2030 - EEA](https://www.eea.europa.eu/policy-documents/eu-biodiversity-strategy-for-2030-1)
- [Protected Areas of Brazil - Wikipedia](https://en.wikipedia.org/wiki/Protected_areas_of_Brazil)
- [SNUC Law No. 9.985 - ECOLEX](https://www.ecolex.org/details/legislation/law-no-9985-establishing-the-national-system-of-protected-areas-management-snuc-lex-faoc024591/)
- [Indonesia Deforestation and Protected Areas - Springer](https://link.springer.com/article/10.1007/s10531-023-02679-8)
- [Indonesia EUDR Risk Benchmarking Profile](https://dv719tqmsuwvb.cloudfront.net/documents/EUDR-Risk-Benchmarking-Indonesia.pdf)
- [Malaysia Legal Forest Classes - Sahabat Alam Malaysia](https://foe-malaysia.org/articles/legal-classes-of-forests-and-conservation-areas-in-malaysia-2/)
- [Heart of Borneo Declaration - WWF](https://wwf.panda.org/discover/knowledge_hub/where_we_work/borneo_forests/about_borneo_forests/declaration/)
- [Protected Areas in the DRC - WWF](https://www.worldwildlife.org/projects/protected-areas-a-pathway-to-sustainable-growth-in-the-democratic-republic-of-congo)
- [DRC Protected Areas - Protected Planet](https://www.protectedplanet.net/country/COD)
- [HCV Network - Forests Workstream](https://www.hcvnetwork.org/workstreams/forests)
- [Rainforest Alliance HCV Standard Revision](https://www.hcvnetwork.org/posts/rainforest-alliances-revamped-standard-to-strengthen-protection-of-high-conservation-value-areas)
- [EUDR Penalties of Non-Compliance - FoodNavigator](https://www.foodnavigator.com/Article/2025/07/22/eudr-penalties-of-non-compliance/)
- [EUDR Penalties - TracexTech](https://tracextech.com/eudr-penalties/)
- [EU Classifies Countries by Deforestation Risk Under EUDR - National Law Review](https://natlawreview.com/article/ec-announces-low-and-high-risk-countries-under-eudr)
- [White & Case - 10 Key Things About EUDR](https://www.whitecase.com/insight-alert/10-key-things-you-still-need-know-about-new-eu-deforestation-regulation)
- [Ramsar Convention - Wikipedia](https://en.wikipedia.org/wiki/Ramsar_Convention)
- [World Heritage and Ramsar Convention - UNESCO](https://whc.unesco.org/en/ramsar/)
