# EU Deforestation Regulation (EUDR) - Comprehensive Regulatory Requirements
**Regulation (EU) 2023/1115, as amended by Regulation (EU) 2025/2650**
**Document Version:** 3.0
**Last Updated:** 2026-02-09
**Purpose:** Production-grade traceability connector technical reference for PRD development
**Maintained by:** GL-RegulatoryIntelligence

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Key Articles and Technical Requirements](#2-key-articles-and-technical-requirements)
3. [Covered Commodities and CN/HS Codes](#3-covered-commodities-and-cnhs-codes)
4. [Geolocation Requirements](#4-geolocation-requirements)
5. [Due Diligence Statement Requirements](#5-due-diligence-statement-requirements)
6. [Risk Assessment Methodology](#6-risk-assessment-methodology)
7. [Timeline and Enforcement Dates](#7-timeline-and-enforcement-dates)
8. [Penalties and Compliance Obligations](#8-penalties-and-compliance-obligations)
9. [EU Information System Technical Integration](#9-eu-information-system-technical-integration)
10. [Mass Balance and Segregation Requirements](#10-mass-balance-and-segregation-requirements)
11. [2025-2026 Amendments (Regulation 2025/2650)](#11-2025-2026-amendments-regulation-20252650)
12. [Data Model Reference](#12-data-model-reference)
13. [Feature Mapping for GreenLang Traceability Connector](#13-feature-mapping-for-greenlang-traceability-connector)

---

## 1. Executive Summary

The EU Deforestation Regulation (Regulation (EU) 2023/1115) entered into force on 29 June 2023. It
prohibits placing on the EU market, making available on the EU market, or exporting from the EU
certain commodities and derived products that are linked to deforestation or forest degradation. The
regulation was significantly amended by Regulation (EU) 2025/2650, published on 23 December 2025,
which postponed application dates and introduced targeted simplifications.

**Core Compliance Principle:** Products must be (a) deforestation-free with respect to the cutoff date
of 31 December 2020, (b) produced in accordance with the relevant legislation of the country of
production, and (c) covered by a due diligence statement or reference number thereof.

**Enforcement Start:** 30 December 2026 for large/medium enterprises; 30 June 2027 for micro/small
enterprises.

---

## 2. Key Articles and Technical Requirements

### Article 2 - Definitions

Critical definitions for the traceability connector:

| Term | Definition | Technical Implication |
|------|-----------|----------------------|
| **Deforestation** | Conversion of forest to agricultural use, whether human-induced or not | Must detect land-use change from forest to agriculture |
| **Forest degradation** | Conversion of primary forests or naturally regenerating forests into plantation forests or other wooded land | Applies specifically to wood commodity |
| **Forest** | Land spanning >0.5 hectares with trees >5m and canopy cover >10%, or trees able to reach these thresholds in situ | FAO definition; must be encoded in detection algorithms |
| **Relevant commodities** | Cattle, cocoa, coffee, oil palm, rubber, soya, wood | Annex I enumeration |
| **Operator** | Any natural or legal person who places relevant products on the EU market or exports them | Bears primary due diligence responsibility |
| **Trader** | Any natural or legal person in the supply chain other than the operator who makes relevant products available on the market | Reduced obligations under 2025/2650 amendment |
| **Downstream operator** | An operator who places products on the market that have been manufactured by processing products placed on the Union market under a due diligence declaration or simplified declaration | New category introduced by 2025/2650 amendment |
| **Plot of land** | The land within a single real-estate property, as recognized by the law of the country of production, used for the production of the relevant commodity | Geolocation unit; determines point vs polygon |
| **Geolocation** | The geographic location of a plot of land described by latitude and longitude coordinates | WGS-84, EPSG-4326, minimum 6 decimal digits |

### Article 3 - Prohibitions

**Technical requirement:** The system must enforce three cumulative conditions before any product can
be cleared for EU market placement or export:

1. **Deforestation-free** - The relevant commodities were produced on land that has NOT been
   subject to deforestation after 31 December 2020. For wood: no forest degradation after 31
   December 2020.
2. **Legal production** - Produced in accordance with the relevant legislation of the country of
   production, including laws on land use rights, environmental protection, forest-related rules,
   third-party rights, labor laws, human rights obligations, free prior and informed consent (FPIC)
   of indigenous peoples, tax and anti-corruption rules, and trade and customs regulations.
3. **Covered by due diligence statement** - A due diligence statement has been submitted to the
   EU Information System.

**Connector implication:** All three conditions must be independently verified and their status tracked
as separate Boolean flags with supporting evidence chains.

### Article 4 - Obligations of Operators

Operators placing relevant products on the Union market or exporting them shall:

1. Exercise due diligence in accordance with Articles 8-12 (information collection, risk assessment,
   risk mitigation, due diligence system).
2. Submit a due diligence statement to the competent authority through the EU Information System
   (per Article 4(2)) prior to placing products on the market or exporting.
3. NOT place on or export from the market products that are not deforestation-free or were not
   produced in compliance with the relevant legislation of the country of production.

**Connector implication:** The system must block product clearance if any compliance check fails. The
due diligence statement submission must occur BEFORE the product enters the EU market.

### Article 9 - Information Collection Requirements

Article 9 specifies the mandatory data that operators must collect for each relevant product. This
is the primary data schema for the traceability connector.

**Required data fields:**

| # | Data Element | Article Ref | Technical Specification |
|---|-------------|-------------|----------------------|
| 1 | Product description | 9(1)(a) | Free text with CN code mapping |
| 2 | Quantity | 9(1)(a) | Numeric with unit (kg, m3, head, etc.) |
| 3 | Country of production | 9(1)(b) | ISO 3166-1 alpha-2 country code |
| 4 | Geolocation of all plots of land | 9(1)(c) | Lat/lon with 6 decimal digits minimum; polygon for plots >4 ha |
| 5 | Date or time range of production | 9(1)(d) | ISO 8601 date/date range (harvest/slaughter date) |
| 6 | Name, postal address, email of supplier | 9(1)(e) | Structured contact record |
| 7 | Name, postal address, email of buyer | 9(1)(f) | Structured contact record (for exports) |
| 8 | Adequately conclusive and verifiable information that products are deforestation-free | 9(1)(g) | Evidence package: satellite imagery, certificates, audit reports |
| 9 | Adequately conclusive and verifiable information that products comply with relevant legislation | 9(1)(h) | Legal compliance documentation |

**For cattle specifically:** Geolocation also requires identification of the establishment or
holding/farm where the cattle were kept during the last relevant period.

### Article 10 - Risk Assessment

The risk assessment must evaluate whether there is a risk that the relevant products intended to be
placed on the market or exported are non-compliant. The operator must consider AT MINIMUM:

| # | Risk Factor | Article Ref | Data Source | System Implementation |
|---|------------|-------------|-------------|----------------------|
| 1 | Country/region risk classification | 10(2)(a) | EU country benchmarking (Art 29) | Lookup table; auto-populated |
| 2 | Presence of forests in the country/region of production | 10(2)(b) | FAO FRA, Global Forest Watch | GIS overlay analysis |
| 3 | Presence of indigenous peoples | 10(2)(c) | National registries, FPIC databases | Flag-based alert |
| 4 | Consultation and cooperation with indigenous peoples | 10(2)(d) | FPIC documentation | Document verification |
| 5 | Prevalence of deforestation/degradation in the country/region/area | 10(2)(e) | Hansen GFC, PRODES, TMF | Satellite change detection |
| 6 | Concerns related to the country of production or origin (sanctions, armed conflict) | 10(2)(f) | EU sanctions lists, conflict databases | Automated screening |
| 7 | Risk of circumvention or mixing with products of unknown origin | 10(2)(g) | Supply chain complexity analysis | Graph-based risk scoring |
| 8 | Complexity of the supply chain | 10(2)(g) | Supplier tier count, intermediaries | Calculated metric |
| 9 | Reliability and triangulation of information | 10(2)(h) | Cross-reference validation | Multi-source verification engine |

**Risk assessment output:** Risk level determination (negligible / non-negligible) that determines
whether mitigation is required.

### Article 11 - Risk Mitigation

If the risk assessment identifies a NON-NEGLIGIBLE risk, the operator MUST take adequate and
proportionate mitigation measures before placing products on the market. These include:

1. Gathering additional information, documentation, or data.
2. Undertaking independent surveys or audits, including field inspections.
3. Requiring additional third-party verification of compliance.
4. Supporting the supplier in building capacity to comply.
5. Any other risk-proportionate measures.

**Connector implication:** The system must support a risk mitigation workflow with status tracking,
evidence attachment, approval gates, and re-assessment triggers.

### Article 12 - Due Diligence System

Operators must establish, implement, and maintain a due diligence SYSTEM (not just individual
assessments) that includes:

| Requirement | Article Ref | System Component |
|------------|-------------|-----------------|
| Written policies and control procedures | 12(1)(a) | Policy document storage and versioning |
| Adequate risk management | 12(1)(b) | Risk engine with configurable rules |
| Compliance officer appointment | 12(1)(c) | User role: EUDR Compliance Officer |
| Internal annual review | 12(2) | Scheduled audit workflow |
| Annual public report | 12(3) | Report generation module |
| Record retention for 5 years | 12(4) | Data retention policy enforcement |

### Article 29 - Country Benchmarking System

The European Commission classifies countries (or sub-national regions) into three risk tiers:

| Tier | Risk Level | Due Diligence | Inspection Rate | Simplified DD |
|------|-----------|---------------|-----------------|---------------|
| **Low** | Low risk of deforestation | Simplified: collect info only, no risk assessment/mitigation required | 1% of operators | Yes |
| **Standard** | Standard risk (default for all countries at entry into force) | Full due diligence required (Arts 9-11) | 3% of operators | No |
| **High** | High risk of deforestation | Enhanced due diligence; enhanced checks on 9% of operators AND 9% of product volume | 9% of operators + 9% volume | No |

**Assessment criteria for benchmarking (Article 29(3)):**

Quantitative:
- Rate of deforestation and forest degradation
- Rate of expansion of agricultural land for relevant commodities
- Production trends for relevant commodities and derived products

Qualitative:
- National/sub-national legal framework and enforcement capacity
- Paris Agreement commitments (NDCs) related to LULUCF
- Cooperation agreements with the EU
- Human rights protections, including FPIC for indigenous peoples
- Information provided by the country itself

**Current classifications (as of December 2025 benchmarking exercise):**
- **Low risk:** 140 countries (includes all EU Member States, UK, US, Canada, Australia, Japan, etc.)
- **Standard risk:** Majority of tropical commodity-producing countries (Brazil, Indonesia, Colombia, etc.)
- **High risk:** Russia, Belarus, Myanmar, North Korea (4 countries)

**Connector implication:** The system must maintain an updatable country/sub-national benchmarking
lookup table. The benchmarking list is dynamic and will be reviewed periodically (first review
scheduled for 2026 using updated FAO FRA data).

---

## 3. Covered Commodities and CN/HS Codes

### The 7 Regulated Commodities

| # | Commodity | HS Chapter | Primary CN Codes | Key Derived Products |
|---|----------|------------|------------------|---------------------|
| 1 | **Cattle** | Ch. 01, 02, 41, 43 | 0102, 0201, 0202, ex 0206, ex 0210, ex 4101, ex 4104, ex 4107, ex 4113, ex 4114 | Live bovine animals, beef, veal, hides, leather, tallow |
| 2 | **Cocoa** | Ch. 18 | 1801, 1802, 1803, 1804, 1805, ex 1806 | Cocoa beans, butter, paste, powder, chocolate, cocoa preparations |
| 3 | **Coffee** | Ch. 09, 21 | 0901, ex 2101 | Green beans, roasted, decaffeinated, extracts, concentrates |
| 4 | **Oil palm** | Ch. 15, 23, 33, 34, 38 | 1511, 1513 11, 1513 19, ex 1513 21, ex 1513 29, ex 2306 60, ex 3823, ex 3824 | Palm oil, palm kernel oil, fractions, oilcake, glycerol, biodiesel feedstock |
| 5 | **Rubber** | Ch. 40 | 4001, ex 4005, ex 4006, ex 4007, 4008, 4009, 4010, 4011, 4012, 4013, ex 4015, 4016, 4017 | Natural rubber, compounded rubber, tyres, tubes, rubber articles |
| 6 | **Soya** | Ch. 12, 15, 23 | 1201, 1207 40, ex 1208, 1507, 2304 | Soybeans, soybean oil, soya flour, oilcake, lecithin |
| 7 | **Wood** | Ch. 44, 47, 48, 49, 94 | 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, ex 4501, ex 4503, ex 4504, 4701-4713, 4801-4823, ex 9401, ex 9403, ex 9404, ex 9406 | Fuel wood, sawn timber, plywood, particle board, paper, pulp, furniture, cork, charcoal |

**Note on "ex" prefix:** The "ex" prefix on CN codes indicates that only a SUBSET of products under
that CN heading are covered -- specifically those derived from EUDR-regulated commodities. The system
must verify not only the CN code but also whether the product actually derives from a relevant
commodity.

**Important 2025/2650 amendment change:** Printed products (books, newspapers, images, and other
products of the printing industry -- CN codes in Chapter 49 such as manuscripts, typescripts, and
plans on paper) have been REMOVED from scope.

### CN Code Matching Rules for the Connector

1. **Exact match:** Product CN code matches a code in Annex I exactly -- product is in scope.
2. **"ex" match:** Product CN code falls under an "ex" heading -- additional verification needed to
   confirm the product derives from a relevant commodity.
3. **Multi-commodity products:** If a product contains multiple EUDR commodities, only the commodity
   that determines the product's CN classification is subject to due diligence.
4. **De minimis:** No de minimis threshold exists. Even trace amounts of a relevant commodity in a
   product that falls under a listed CN code trigger the regulation.

---

## 4. Geolocation Requirements

### Core Specification

| Parameter | Requirement | Source |
|-----------|------------|--------|
| Coordinate system | WGS-84 (World Geodetic System 1984) | Art 9, Commission FAQ |
| Projection | EPSG:4326 | EU Information System specification |
| Precision | Minimum 6 decimal digits for latitude AND longitude | Art 9(1)(c), Recital 44 |
| Format | GeoJSON (for Information System submission) | EU IS technical specification |
| Plot threshold | 4 hectares | Art 9(1)(c) |

### Point vs. Polygon Rules

| Plot Size | Geolocation Type | Specification |
|-----------|-----------------|---------------|
| **< 4 hectares** | Single point (latitude, longitude) | One coordinate pair with >= 6 decimal digits |
| **>= 4 hectares** | Polygon boundary | Latitude/longitude pairs defining the perimeter of each plot |

**Polygon technical requirements:**
- Sufficient points to accurately describe the plot boundary (typically 6-12 points minimum)
- Coordinates listed in sequential order tracing the perimeter
- First and last coordinate must be identical (closed polygon)
- No self-intersections allowed
- Holes in polygons supported (outer ring + inner ring(s) in GeoJSON)
- Multi-polygon supported for non-contiguous plots

**Special case -- cattle:** For cattle, geolocation must identify the establishment or holding where
the cattle were kept. The 4-hectare polygon rule applies to the grazing/farming land associated with
the establishment.

### Precision Implications

6 decimal digits of latitude/longitude correspond to approximately 0.11 meters (11 centimeters) of
precision at the equator. This is sufficient to identify individual trees or field boundaries.

| Decimal Places | Approximate Precision | Sufficient for EUDR? |
|---------------|----------------------|---------------------|
| 1 | 11.1 km | No |
| 2 | 1.1 km | No |
| 3 | 111 m | No |
| 4 | 11.1 m | No |
| 5 | 1.1 m | Marginal |
| **6** | **0.11 m** | **Yes (minimum)** |
| 7 | 0.011 m | Yes |

### Validation Rules for the Connector

```
GEOLOCATION_VALIDATION_RULES:
  1. coordinate_system: WGS-84 (EPSG:4326)
  2. latitude_range: -90.000000 <= lat <= 90.000000
  3. longitude_range: -180.000000 <= lon <= 180.000000
  4. decimal_precision: >= 6 decimal digits for BOTH lat and lon
  5. plot_size_check:
     - if area_hectares < 4: accept single point
     - if area_hectares >= 4: require polygon boundary
  6. polygon_closure: first_point == last_point
  7. polygon_validity: no self-intersections (use ST_IsValid)
  8. country_boundary_check: coordinates must fall within declared country of production
  9. land_type_check: coordinates must not fall in ocean, lake, or urban area (unless cattle)
  10. format: GeoJSON (FeatureCollection with Point or Polygon geometry)
  11. crs_transformation: accept input in any CRS, transform to EPSG:4326 for storage
```

---

## 5. Due Diligence Statement Requirements

### DDS Data Schema

The Due Diligence Statement (DDS) is the formal declaration submitted through the EU Information
System. Under the 2025/2650 amendment, only the FIRST OPERATOR placing the product on the EU market
is required to submit a full DDS.

**Mandatory DDS fields:**

| # | Field | Data Type | Validation | Notes |
|---|-------|-----------|-----------|-------|
| 1 | Operator identification | Structured object | EORI number validation | Name, address, EORI |
| 2 | Product description | Text + CN code | Annex I CN code lookup | Must match listed CN codes |
| 3 | Quantity | Numeric + unit | Positive number | kg, m3, head, litres, etc. |
| 4 | Country of production | ISO code | ISO 3166-1 alpha-2 | Multi-country if applicable |
| 5 | Geolocation data | GeoJSON | See Section 4 validation rules | Point or Polygon per plot size |
| 6 | Production date/range | ISO 8601 | Must be after 31 Dec 2020 | Date or date range |
| 7 | Supplier details | Structured object | Required fields: name, address, email | Full chain if available |
| 8 | Buyer details | Structured object | Required for exports | Name, address, email |
| 9 | Deforestation-free declaration | Boolean + evidence | Must be TRUE with supporting evidence | Satellite data, certificates |
| 10 | Legal compliance declaration | Boolean + evidence | Must be TRUE with supporting evidence | Legal documentation |
| 11 | Risk assessment result | Enum | negligible / non-negligible | Determines mitigation need |
| 12 | Risk mitigation measures | Structured list | Required if risk is non-negligible | Actions taken with evidence |
| 13 | Compliance officer attestation | Signature | Digital signature of appointed officer | Legal declaration |

### DDS Reference Number System

Each DDS receives a unique reference number from the EU Information System upon submission. This
reference number is the key traceability artifact throughout the supply chain.

```
DDS Reference Number Format: [to be confirmed by EU IS -- expected alphanumeric]
Retention period: Minimum 5 years from date of submission
Accessibility: Must be retrievable by competent authorities on request
```

### DDS Lifecycle States

```
DRAFT -> SUBMITTED -> VALIDATED -> ACTIVE -> [AMENDED | WITHDRAWN]
                   -> REJECTED (with error codes)
```

**Amendment:** Operators can amend a submitted DDS if information changes. The amended statement
replaces the original but retains the original reference number with a version suffix.

**Withdrawal:** Operators can withdraw a DDS if the product is not ultimately placed on the market.

### Simplified Declaration (New in 2025/2650 Amendment)

Micro and small primary operators in LOW-RISK countries may submit a simplified declaration instead
of a full DDS. The simplified declaration requires:
- Basic operator information
- Product and quantity
- Country of production
- A declaration that the product is deforestation-free and legally produced
- Declaration identification number (distinct from DDS reference number)

---

## 6. Risk Assessment Methodology

### Three-Tier Country Benchmarking Framework

```
RISK_CLASSIFICATION:
  LOW_RISK:
    description: "Low or negligible level of deforestation risk"
    due_diligence: "Simplified - collect information only (Art 9)"
    risk_assessment: "Not required"
    risk_mitigation: "Not required"
    inspection_rate: "1% of operators"
    current_count: "140 countries"

  STANDARD_RISK:
    description: "Default classification for all countries"
    due_diligence: "Full (Arts 9, 10, 11)"
    risk_assessment: "Required"
    risk_mitigation: "Required if non-negligible risk found"
    inspection_rate: "3% of operators"
    current_count: "Majority of tropical producing countries"

  HIGH_RISK:
    description: "High level of deforestation risk"
    due_diligence: "Enhanced (Arts 9, 10, 11 + additional checks)"
    risk_assessment: "Required with enhanced scrutiny"
    risk_mitigation: "Required with enhanced measures"
    inspection_rate: "9% of operators + 9% of product volume"
    current_count: "4 countries (Russia, Belarus, Myanmar, North Korea)"
```

### Risk Assessment Scoring Model

The following weighted scoring model is recommended for the GreenLang connector based on Article 10
requirements:

```
TOTAL_RISK_SCORE =
    (COUNTRY_RISK       * 0.30) +   // Art 10(2)(a) - country benchmark tier
    (DEFORESTATION_RISK * 0.25) +   // Art 10(2)(b,e) - forest presence, deforestation rates
    (SUPPLY_CHAIN_RISK  * 0.20) +   // Art 10(2)(g) - complexity, intermediaries
    (LEGAL_RISK         * 0.10) +   // Art 10(2)(f) - sanctions, conflict, governance
    (INDIGENOUS_RISK    * 0.05) +   // Art 10(2)(c,d) - indigenous peoples presence/FPIC
    (DATA_QUALITY_RISK  * 0.10)     // Art 10(2)(h) - reliability of information

Where each factor is scored 0.0 (no risk) to 1.0 (maximum risk)

RISK_THRESHOLD:
  negligible:     total_score < 0.30
  non_negligible: total_score >= 0.30
```

### Risk Factor Data Sources

| Risk Factor | Primary Data Source | Update Frequency | API Available |
|------------|-------------------|------------------|---------------|
| Country benchmark | EU Commission classification | Annual review | EU IS lookup |
| Deforestation rate | Hansen Global Forest Change | Annual | GFW API |
| Forest cover | FAO Forest Resources Assessment | 5-year cycle | FAO STAT |
| Satellite monitoring | Copernicus Sentinel-2 | 5-day revisit | Copernicus API |
| Sanctions | EU Consolidated Sanctions List | Continuous | EU API |
| Corruption index | Transparency International CPI | Annual | Public dataset |
| Indigenous territories | LandMark / RAISG | Periodic | GIS layers |
| Supply chain complexity | Internal calculation | Per transaction | N/A |

---

## 7. Timeline and Enforcement Dates

### Original Timeline (Regulation 2023/1115)

| Date | Event | Status |
|------|-------|--------|
| 29 Jun 2023 | Regulation enters into force | COMPLETED |
| 30 Dec 2024 | Original application date for large operators | POSTPONED |
| 30 Jun 2025 | Original application date for SMEs | POSTPONED |

### Amended Timeline (Regulation 2025/2650, published 23 Dec 2025)

| Date | Event | Status |
|------|-------|--------|
| 23 Dec 2025 | Amending Regulation (EU) 2025/2650 published in Official Journal | COMPLETED |
| 26 Dec 2025 | Regulation 2025/2650 enters into force (3 days after publication) | COMPLETED |
| **30 Apr 2026** | **Commission must present simplification review report** | UPCOMING |
| **30 Dec 2026** | **APPLICATION DATE: Large and medium operators/traders** | UPCOMING |
| **30 Jun 2027** | **APPLICATION DATE: Micro and small operators** | UPCOMING |
| 2026-2027 | First country benchmarking review (using updated FAO FRA data) | PLANNED |

### Key Milestones for GreenLang Connector Development

| Milestone | Target Date | Dependency |
|-----------|------------|------------|
| EU IS API specification finalized | Expected Q1-Q2 2026 | EU Commission |
| EU IS production environment available | Expected H2 2026 | EU Commission |
| Commission simplification review | 30 Apr 2026 | May trigger further amendments |
| Country benchmarking update | H2 2026 | FAO FRA 2025 data |
| Large operator compliance deadline | 30 Dec 2026 | Hard deadline |
| SME compliance deadline | 30 Jun 2027 | Hard deadline |

---

## 8. Penalties and Compliance Obligations

### Penalty Framework (Articles 23-25)

Member States are responsible for defining specific penalties, but the Regulation sets MINIMUM
requirements that penalties must meet:

| Penalty Type | Description | Maximum |
|-------------|------------|---------|
| **Financial fines** | Proportional to environmental damage and product value | Up to **4% of annual EU turnover** |
| **Confiscation** | Seizure of non-compliant products | Full product value |
| **Revenue confiscation** | Recovery of profits from non-compliant transactions | Full profit from infringement |
| **Market ban** | Temporary prohibition from placing products on market | Up to 12 months |
| **Procurement exclusion** | Temporary exclusion from public procurement | Duration set by Member State |
| **Registration suspension** | Temporary suspension from EUDR Information System | Prevents all EU market activity |

**Escalation:** Fines must increase for repeated infringements. Penalties must exceed any economic
benefit gained from non-compliance.

### Operator Obligations (Primary Responsibility)

| Obligation | Description | Evidence Required |
|-----------|------------|------------------|
| Full due diligence | Exercise due diligence per Arts 8-12 | DDS submission |
| DDS submission | Submit before market placement | EU IS confirmation |
| Information collection | Collect all Art 9 data | Supplier records, geolocation data |
| Risk assessment | Evaluate all Art 10 factors | Risk assessment report |
| Risk mitigation | Mitigate non-negligible risks per Art 11 | Mitigation evidence |
| Due diligence system | Maintain system per Art 12 | Policies, procedures, annual review |
| Record retention | Retain all records for 5 years | Digital archive |
| Annual review | Review DD system annually | Review report |
| Annual reporting | Publish annual report on DD steps taken | Public report |
| Compliance officer | Appoint a compliance officer | Appointment record |

### Trader Obligations (Post-2025/2650 Amendment)

**Critical change:** Under the 2025/2650 amendment, downstream operators and traders further down
the supply chain are NO LONGER required to submit their own DDS or conduct due diligence.

| Trader Type | Obligations |
|------------|------------|
| **First downstream operator/trader** | Collect and retain the DDS reference number from the upstream operator |
| **Subsequent downstream operators/traders** | Record-keeping, notification of risks to competent authorities, cooperation with authorities |
| **Non-SME downstream traders** | Must register in the EU Information System |
| **All traders** | Must not make available non-compliant products; must cooperate with competent authorities |

### Competent Authority Inspection Framework

| Risk Tier | Operator Inspection Rate | Volume Inspection Rate | Frequency |
|-----------|------------------------|----------------------|-----------|
| Low risk | 1% of relevant operators | N/A | Annual |
| Standard risk | 3% of relevant operators | N/A | Annual |
| High risk | 9% of relevant operators | 9% of relevant product volume | Annual |

---

## 9. EU Information System Technical Integration

### System Overview

The EU Information System is built on the **TRACES NT** (Trade Control and Expert System - New
Technology) platform, operated by the European Commission's Directorate-General for Health and
Food Safety (DG SANTE).

### Access Methods

| Method | Target Users | Use Case |
|--------|-------------|----------|
| **Web portal** | Individual operators, small traders | Manual DDS submission, single statements |
| **API (M2M)** | Large operators, system integrators | Bulk DDS submission, automated workflows |
| **Bulk upload** | Medium operators | CSV/file-based batch submission |

### API Integration Requirements

| Parameter | Specification |
|-----------|--------------|
| Protocol | HTTPS REST API |
| Authentication | EU Login (ECAS) + OAuth 2.0 token |
| Data format | JSON (GeoJSON for geolocation data) |
| Coordinate system | WGS-84 (EPSG:4326) |
| Character encoding | UTF-8 |
| Rate limiting | To be specified by EU (expected per-operator limits) |
| Conformance testing | **Required** - operators must pass technical conformance test before production access |
| Sandbox environment | Available for integration testing |

### API Operations (Expected)

```
POST   /api/v1/dds                    # Submit new DDS
GET    /api/v1/dds/{reference}        # Retrieve DDS by reference number
PUT    /api/v1/dds/{reference}        # Amend existing DDS
DELETE /api/v1/dds/{reference}        # Withdraw DDS
GET    /api/v1/dds/status/{reference} # Check DDS processing status
POST   /api/v1/dds/bulk               # Bulk submit multiple DDS
GET    /api/v1/benchmarking/{country}  # Get country risk classification
GET    /api/v1/cn-codes               # Get current list of covered CN codes
```

### GeoJSON Submission Format

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-47.929200, -15.780100]
      },
      "properties": {
        "plot_id": "PLOT-BR-2024-00001",
        "area_hectares": 2.5,
        "commodity": "coffee",
        "production_date_start": "2024-03-01",
        "production_date_end": "2024-09-30"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-47.929200, -15.780100],
          [-47.928100, -15.780100],
          [-47.928100, -15.779000],
          [-47.929200, -15.779000],
          [-47.929200, -15.780100]
        ]]
      },
      "properties": {
        "plot_id": "PLOT-BR-2024-00002",
        "area_hectares": 12.3,
        "commodity": "soya",
        "production_date_start": "2024-01-15",
        "production_date_end": "2024-05-20"
      }
    }
  ]
}
```

**Note on coordinate order:** GeoJSON uses [longitude, latitude] order (not [latitude, longitude]).
This is a common source of errors in implementations.

---

## 10. Mass Balance and Segregation Requirements

### EUDR Position on Chain of Custody

The EUDR takes a **strict physical traceability** approach. This is fundamentally different from
mass balance systems used by certification schemes.

### What Is Prohibited

| Practice | Allowed? | Rationale |
|----------|---------|-----------|
| Mixing compliant + non-compliant products | **NO** | Products lose traceability to plot of origin |
| Traditional mass balance (certified/non-certified blending) | **NO** | Cannot guarantee individual product is deforestation-free |
| Book-and-claim credits | **NO** | No physical link to deforestation-free source |
| Batch dilution | **NO** | Undermines per-product compliance |

### What Is Allowed

| Practice | Allowed? | Conditions |
|----------|---------|------------|
| **Physical segregation** | **YES** | Compliant products kept physically separate from non-compliant |
| **Declaration in excess** | **YES** | Declare more plots than directly linked to a shipment, provided ALL declared plots meet EUDR requirements |
| **Controlled mass balance** | **Conditional** | Only if ALL sources in the balance are EUDR-compliant (mixing two compliant sources is acceptable) |
| **Mixing compliant sources** | **YES** | Multiple EUDR-compliant sources can be combined as long as all originating plots are declared |

### Segregation Implementation Requirements

```
SEGREGATION_RULES:
  1. identity_preserved:
     description: "Single origin, single plot, no mixing"
     traceability: "1:1 mapping from product to plot"
     use_case: "Premium single-origin products"

  2. segregated:
     description: "Multiple compliant sources, kept separate from non-compliant"
     traceability: "Product linked to set of compliant plots"
     use_case: "Standard EUDR compliance"

  3. controlled_mass_balance:
     description: "Mixing of multiple EUDR-compliant sources"
     traceability: "All input sources verified compliant; total declared plots >= total volume"
     use_case: "Commodity processing (refining, milling)"
     condition: "ALL inputs must be EUDR-compliant with valid DDS"

  4. NOT_ALLOWED - traditional_mass_balance:
     description: "Mixing certified + non-certified"
     reason: "Cannot guarantee individual product compliance"
```

### Connector Implementation

The traceability connector must:

1. Track lot/batch identity through all supply chain nodes (farm -> collection -> processing ->
   shipping -> port -> distribution).
2. Enforce segregation rules at each transformation point (milling, refining, manufacturing).
3. Maintain a "transformation matrix" that links output products to input plots.
4. Calculate yield/conversion ratios to ensure volume consistency.
5. Flag any point where compliant and non-compliant products could have been mixed.
6. Support "declaration in excess" by linking a DDS to a superset of qualifying plots.

---

## 11. 2025-2026 Amendments (Regulation 2025/2650)

### Summary of Key Changes

Regulation (EU) 2025/2650 was published on 23 December 2025 in the Official Journal and entered
into force on 26 December 2025. It amends Regulation (EU) 2023/1115 with the following changes:

| Change | Original (2023/1115) | Amended (2025/2650) | Impact on Connector |
|--------|---------------------|---------------------|-------------------|
| **Application date (large/medium)** | 30 Dec 2024 | **30 Dec 2026** | 2 additional years for development |
| **Application date (micro/small)** | 30 Jun 2025 | **30 Jun 2027** | Extended timeline for SME features |
| **Downstream operator DDS** | Required to submit own DDS | **Not required** - only collect/retain reference number | Simplifies downstream workflow |
| **Subsequent traders** | Required to submit DDS | **Not required** - record-keeping and risk notification only | Reduces system complexity |
| **First downstream operator** | N/A (new concept) | Must collect and retain DDS reference number | New role in data model |
| **Micro/small primary operators (low-risk)** | Full DDS | **Simplified declaration** (one-off) | New simplified submission path |
| **Printed products** | In scope (Ch. 49) | **Removed from scope** | Remove CN codes from Annex I lookup |
| **Simplification review** | N/A | Commission report by **30 Apr 2026** | Possible further changes |
| **New definitions** | N/A | "Downstream operator", "first downstream operator", "first downstream trader" | Update data model roles |

### Implications for the GreenLang Connector

1. **Role-based workflow differentiation:** The system must distinguish between (a) first operators
   (full DDS), (b) first downstream operators (reference number retention), (c) subsequent
   downstream operators/traders (record-keeping only), and (d) micro/small primary operators in
   low-risk countries (simplified declaration).

2. **Printed product exclusion:** CN code lookup table must be updated to exclude Chapter 49
   products removed from scope.

3. **Simplification review monitoring:** The Commission's report due 30 April 2026 may trigger
   further legislative changes. The connector architecture should be designed for adaptability.

4. **Country benchmarking dynamic updates:** The benchmarking list will be reviewed periodically,
   with the first review expected in 2026.

---

## 12. Data Model Reference

### Core Entities for the Traceability Connector

```
DueDiligenceStatement:
  reference_number: string          # EU IS assigned
  version: integer                  # Amendment tracking
  status: enum [DRAFT, SUBMITTED, VALIDATED, ACTIVE, AMENDED, WITHDRAWN, REJECTED]
  operator_id: FK -> Operator
  submission_date: datetime
  products: FK[] -> Product
  plots: FK[] -> Plot
  risk_assessment: FK -> RiskAssessment
  risk_mitigation: FK -> RiskMitigation (nullable)
  compliance_officer_id: FK -> User
  evidence_package: FK -> EvidencePackage
  retention_expiry: date            # submission_date + 5 years

Operator:
  id: uuid
  name: string
  eori_number: string               # Economic Operators Registration and Identification
  address: StructuredAddress
  email: string
  operator_type: enum [FIRST_OPERATOR, DOWNSTREAM_OPERATOR, EXPORTER]
  size_category: enum [LARGE, MEDIUM, SMALL, MICRO]
  eu_is_registration_id: string
  compliance_officer_id: FK -> User

Product:
  id: uuid
  description: string
  commodity: enum [CATTLE, COCOA, COFFEE, OIL_PALM, RUBBER, SOYA, WOOD]
  cn_code: string                   # 8-digit Combined Nomenclature code
  hs_code: string                   # 6-digit Harmonized System code
  quantity: decimal
  unit: enum [KG, M3, HEAD, LITRE, TONNE, PIECE]
  production_date_start: date
  production_date_end: date
  country_of_production: string     # ISO 3166-1 alpha-2

Plot:
  id: uuid
  plot_reference: string            # External plot identifier
  country: string                   # ISO 3166-1 alpha-2
  geolocation_type: enum [POINT, POLYGON]
  point_latitude: decimal(9,6)      # For plots < 4 ha
  point_longitude: decimal(9,6)     # For plots < 4 ha
  polygon_geojson: jsonb            # For plots >= 4 ha
  area_hectares: decimal
  commodity: enum [CATTLE, COCOA, COFFEE, OIL_PALM, RUBBER, SOYA, WOOD]
  establishment_id: string          # For cattle: establishment/holding ID
  forest_cover_2020: decimal        # Baseline forest cover as of 31 Dec 2020
  current_forest_cover: decimal     # Latest satellite assessment
  deforestation_detected: boolean
  last_satellite_check: datetime

RiskAssessment:
  id: uuid
  dds_id: FK -> DueDiligenceStatement
  assessment_date: datetime
  country_risk_tier: enum [LOW, STANDARD, HIGH]
  country_risk_score: decimal(3,2)
  deforestation_risk_score: decimal(3,2)
  supply_chain_risk_score: decimal(3,2)
  legal_risk_score: decimal(3,2)
  indigenous_risk_score: decimal(3,2)
  data_quality_risk_score: decimal(3,2)
  total_risk_score: decimal(3,2)
  risk_level: enum [NEGLIGIBLE, NON_NEGLIGIBLE]
  assessor_id: FK -> User
  methodology_version: string

Supplier:
  id: uuid
  name: string
  address: StructuredAddress
  email: string
  country: string
  supplier_tier: integer            # 1 = direct, 2 = indirect, etc.
  plots: FK[] -> Plot
  certifications: jsonb             # FSC, RSPO, PEFC, etc.
  compliance_history: jsonb

CountryBenchmark:
  country_code: string              # ISO 3166-1 alpha-2
  sub_national_region: string       # nullable; for sub-national classification
  risk_tier: enum [LOW, STANDARD, HIGH]
  effective_date: date
  review_date: date
  assessment_criteria: jsonb
  source: string                    # "EU Commission Implementing Regulation"
```

---

## 13. Feature Mapping for GreenLang Traceability Connector

### Mapping Regulatory Requirements to GreenLang EUDR Agent PRDs

| Regulatory Requirement | Article | GreenLang PRD | Agent |
|----------------------|---------|---------------|-------|
| Supply chain mapping | Art 9(1)(e,f) | GL-EUDR-001 | Supply Chain Mapper Agent |
| Geolocation collection and validation | Art 9(1)(c) | GL-EUDR-002 | Geolocation Collector Agent |
| Commodity traceability | Art 9(1)(a,b,d) | GL-EUDR-003 | Commodity Traceability Agent |
| Supplier verification | Art 9, 10 | GL-EUDR-004 | Supplier Verification Agent |
| Plot registry management | Art 9(1)(c) | GL-EUDR-005 | Plot Registry Agent |
| Origin declaration | Art 4, 9 | GL-EUDR-006 | Origin Declaration Agent |
| Chain of custody | Art 3, segregation | GL-EUDR-007 | Chain of Custody Agent |
| Batch tracking | Art 9, segregation | GL-EUDR-008 | Batch Tracking Agent |
| Transportation tracking | Art 9 | GL-EUDR-009 | Transportation Tracking Agent |
| Warehouse tracking | Art 9, segregation | GL-EUDR-010 | Warehouse Tracking Agent |
| Processing facility | Art 9, mass balance | GL-EUDR-011 | Processing Facility Agent |
| Mass balance / declaration in excess | Art 3, segregation | GL-EUDR-012 | Mass Balance Agent |
| Segregation compliance | Art 3 | GL-EUDR-013 | Segregation Compliance Agent |
| Traceability audit | Art 12, 23 | GL-EUDR-014 | Traceability Audit Agent |
| Supply chain risk assessment | Art 10, 11, 29 | GL-EUDR-015 | Supply Chain Risk Agent |

### GreenLang 5-Agent Pipeline Mapping

```
Pipeline Stage              | Regulatory Articles Covered
-----------------------------------------------------------------
[1] SupplierDataIntakeAgent | Art 9 (information collection)
[2] GeoValidationAgent      | Art 9(1)(c) (geolocation validation)
[3] DeforestationRiskAgent  | Art 10, 29 (risk assessment, country benchmarking)
[4] DocumentVerificationAgent| Art 9(1)(g,h), Art 11 (evidence, mitigation)
[5] DDSReportingAgent       | Art 4(2), Art 12 (DDS submission, system compliance)
```

### Key GreenLang Implementation Files

| Component | File Path | Lines |
|-----------|----------|-------|
| Commodity Database | `greenlang/data/eudr_commodities.py` | 1,221 |
| Country Risk Database | `greenlang/data/eudr_country_risk.py` | 1,953 |
| GeoJSON Parser | `greenlang/governance/validation/geolocation/geojson_parser.py` | 876 |
| Supply Chain Traceability | `greenlang/data/supply_chain/eudr/traceability.py` | 1,180 |
| Deforestation Baseline | `greenlang/governance/validation/geolocation/deforestation_baseline.py` | N/A |
| Satellite Alert | `greenlang/extensions/satellite/alerts/deforestation_alert.py` | N/A |
| Core EUDR Tools | `greenlang/utilities/tools/eudr.py` | N/A |

---

## Appendix A: Key Regulatory References

| Document | Identifier | URL |
|----------|-----------|-----|
| Original Regulation | Regulation (EU) 2023/1115 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R1115 |
| Amending Regulation | Regulation (EU) 2025/2650 | https://eur-lex.europa.eu/eli/reg/2025/2650/oj/eng |
| EU Information System | TRACES NT | https://green-forum.ec.europa.eu/nature-and-biodiversity/deforestation-regulation-implementation/information-system-deforestation-regulation_en |
| Country Benchmarking | Commission Assessment | https://green-forum.ec.europa.eu/nature-and-biodiversity/deforestation-regulation-implementation/eudr-cooperation-and-partnerships_en |
| Traceability & Geolocation FAQ | 4th Edition | https://www.global-traceability.com/en/faqs-explained-eudr-traceability-geolocation-4th-ed/ |

## Appendix B: Glossary

| Acronym | Full Name |
|---------|----------|
| CBAM | Carbon Border Adjustment Mechanism |
| CN | Combined Nomenclature (EU 8-digit product classification) |
| CPI | Corruption Perceptions Index |
| DDS | Due Diligence Statement |
| ECAS | European Commission Authentication Service |
| EORI | Economic Operators Registration and Identification |
| EPSG | European Petroleum Survey Group (coordinate reference codes) |
| EUDR | EU Deforestation Regulation |
| FAO FRA | FAO Forest Resources Assessment |
| FPIC | Free, Prior and Informed Consent |
| GFC | Global Forest Change (Hansen dataset) |
| GFW | Global Forest Watch |
| HS | Harmonized System (international 6-digit product classification) |
| LULUCF | Land Use, Land-Use Change and Forestry |
| M2M | Machine-to-Machine |
| NDC | Nationally Determined Contribution (Paris Agreement) |
| NDVI | Normalized Difference Vegetation Index |
| SME | Small and Medium-sized Enterprise |
| TRACES NT | Trade Control and Expert System - New Technology |
| WGS-84 | World Geodetic System 1984 |

---

*This document is maintained by GL-RegulatoryIntelligence and should be reviewed monthly or upon*
*any regulatory update. Next scheduled review: March 2026.*
*Last updated: 2026-02-09*
