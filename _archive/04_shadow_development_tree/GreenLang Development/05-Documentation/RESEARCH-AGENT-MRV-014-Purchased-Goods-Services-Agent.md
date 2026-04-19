# Research: AGENT-MRV-014 -- Scope 3 Category 1 Purchased Goods and Services Agent

## Executive Summary

This document provides the comprehensive technical research foundation for
AGENT-MRV-014, the Scope 3 Category 1 Purchased Goods and Services calculation
agent. Category 1 is typically the single largest Scope 3 category for most
organizations, often representing 20-70% of total value chain emissions
depending on industry sector. This research covers the GHG Protocol Corporate
Value Chain (Scope 3) Standard requirements, all four calculation methods
(supplier-specific, hybrid, average-data, spend-based), emission factor
databases, spend classification systems, data quality scoring, regulatory
obligations, category boundaries, materiality analysis, and industry benchmarks.

**Key finding**: Scope 3 Category 1 requires a multi-method calculation engine
that can simultaneously operate at different tiers of data quality -- from
high-accuracy supplier-specific cradle-to-gate data down to broad spend-based
EEIO estimates -- and produce a unified inventory with transparent data quality
scoring. The agent must integrate with GreenLang's existing AGENT-DATA-009
(Spend Data Categorizer) for spend classification and the Emission Factor
Library for factor lookup, while providing its own specialized calculation
engines for each of the four GHG Protocol methods.

**Regulatory urgency**: Multiple regulatory frameworks now mandate or strongly
encourage Scope 3 Category 1 reporting:
- EU CSRD/ESRS E1: Scope 3 disclosure mandatory for large companies (FY2025+)
- California SB 253: Scope 3 mandatory for entities >$1B revenue (by 2027)
- SBTi: Scope 3 target required if >40% of total emissions
- CDP: Category 1 is the most commonly reported Scope 3 category

---

## Table of Contents

1. [GHG Protocol Scope 3 Standard -- Category 1 Requirements](#1-ghg-protocol-scope-3-standard----category-1-requirements)
2. [Four Calculation Methods](#2-four-calculation-methods)
3. [Emission Factor Sources and Databases](#3-emission-factor-sources-and-databases)
4. [Spend Classification Systems](#4-spend-classification-systems)
5. [Key Formulas and Calculations](#5-key-formulas-and-calculations)
6. [Data Quality Scoring](#6-data-quality-scoring)
7. [Regulatory Context](#7-regulatory-context)
8. [Category Boundaries and Double-Counting Avoidance](#8-category-boundaries-and-double-counting-avoidance)
9. [Materiality and Prioritization](#9-materiality-and-prioritization)
10. [Industry Benchmarks](#10-industry-benchmarks)
11. [Suggested 7-Engine Architecture](#11-suggested-7-engine-architecture)
12. [Suggested Enums, Models, and Constants](#12-suggested-enums-models-and-constants)
13. [Integration with Existing GreenLang Agents](#13-integration-with-existing-greenlang-agents)
14. [References](#14-references)

---

## 1. GHG Protocol Scope 3 Standard -- Category 1 Requirements

### 1.1 Standard Overview

The GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting
Standard (2011) defines 15 categories of Scope 3 emissions. Category 1,
"Purchased Goods and Services," covers all upstream (cradle-to-gate) emissions
from the production of products purchased or acquired by the reporting company
in the reporting year. This includes both goods (tangible products) and
services (intangible products).

The Scope 3 Standard is supplemented by the Technical Guidance for Calculating
Scope 3 Emissions (2013), which provides detailed calculation methods,
decision trees, and worked examples for each category. Chapter 1 of the
Technical Guidance is dedicated entirely to Category 1.

**Source documents:**
- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011), Chapter 5
- Technical Guidance for Calculating Scope 3 Emissions (2013), Chapter 1
- Scope 3 Calculation Guidance (online, updated periodically)

### 1.2 Category 1 Definition

Category 1 includes all upstream emissions from the production of products
purchased or acquired by the reporting company in the reporting year. Products
include both goods (tangible) and services (intangible).

**Purchased Goods** include:
- Raw materials (metals, minerals, chemicals, agricultural commodities)
- Components and sub-assemblies
- Manufactured parts
- Packaging materials (primary, secondary, tertiary)
- Office supplies and consumables
- Food and beverages (for consumption or resale)
- Clothing and textiles
- Electronics and IT equipment (non-capital, i.e., items below capitalization threshold)

**Purchased Services** include:
- Professional services (consulting, legal, accounting, auditing)
- IT services (cloud computing, SaaS, data hosting, software licenses)
- Cleaning and janitorial services
- Security services
- Marketing and advertising services
- Temporary staffing and recruitment
- Insurance services
- Telecommunications
- Financial services (banking fees, transaction processing)
- Maintenance and repair services (for non-capital assets)
- Training and education services
- Research and development services (outsourced)
- Printing and publishing services
- Catering and food services

### 1.3 Minimum Boundary

Per Table 5.4 of the Scope 3 Standard, the minimum boundary for Category 1
encompasses:

| Activity | Minimum Boundary | Optional |
|----------|-----------------|----------|
| Extraction of raw materials | Required | -- |
| Agricultural/forestry activities | Required | -- |
| Processing/manufacturing (all tiers) | Required | -- |
| Generation of electricity consumed upstream | Required | -- |
| Land use change upstream | Optional | Yes |
| Transportation between tiers | Optional | Yes |
| Disposal of waste in upstream operations | Optional | Yes |

The boundary is "cradle-to-gate" -- meaning all emissions from the extraction
of raw materials through to the point where the product is delivered to the
reporting company's gate. This includes:

1. **Tier 1 suppliers**: Direct (first-tier) suppliers of the reporting company
2. **Tier 2+ suppliers**: Suppliers of Tier 1 suppliers, and so on back to
   raw material extraction
3. **Embedded energy**: Emissions from energy consumed during production at
   all tiers
4. **Process emissions**: Non-combustion emissions from manufacturing processes

### 1.4 Temporal Scope

Emissions are accounted for in the year of purchase, not the year of use or
consumption. If a company purchases $10M of steel in FY2025, all cradle-to-gate
emissions for that steel are reported in the FY2025 Scope 3 inventory,
regardless of when the steel is actually used in production.

This temporal rule has implications:
- Inventory changes do not affect the calculation
- Prepayments and accruals should follow accounting treatment
- Multi-year service contracts may be allocated proportionally
- Goods purchased for resale are included (retail sector)

### 1.5 Organizational Boundary Considerations

Category 1 applies regardless of the organizational boundary approach chosen
(equity share, financial control, or operational control). However, the choice
of approach affects which purchases are included:

- **Equity share**: Include purchases proportional to equity stake
- **Financial control**: Include all purchases by entities over which the
  company has financial control
- **Operational control**: Include all purchases by entities over which the
  company has operational control

### 1.6 Reporting Requirements

Per Chapter 9 of the Scope 3 Standard, companies shall report:

1. Total Category 1 emissions in metric tons CO2e
2. A description of the types of purchased goods and services included
3. A description of the calculation methodologies used
4. A description of the data sources used
5. Data quality assessment results
6. A description of any exclusions and their justification
7. Year-over-year change and explanation of trends

Companies should also report (recommended but not required):
- Emissions broken down by product/service category
- Emissions broken down by calculation method used
- Percentage of spend or quantity covered by each method
- Emissions intensity metrics (e.g., tCO2e per unit revenue)

---

## 2. Four Calculation Methods

### 2.1 Method Hierarchy and Selection

The GHG Protocol Technical Guidance defines four calculation methods for
Category 1, listed here from most to least accurate:

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | Supplier-specific | Supplier cradle-to-gate data | Highest (+/-10-30%) | Lowest (key suppliers only) |
| 2 | Hybrid | Mix of supplier + secondary data | High (+/-20-50%) | Medium |
| 3 | Average-data | Quantities + physical EFs | Medium (+/-30-60%) | Medium-High |
| 4 | Spend-based | Spend amounts + EEIO factors | Lowest (+/-50-100%) | Highest (all spend) |

The GHG Protocol recommends:
- Use supplier-specific data wherever possible for high-impact purchases
- Use hybrid or average-data methods for medium-impact purchases
- Use spend-based methods as a screening tool and for low-impact categories
- Progressively improve data quality over time (move up the hierarchy)

**Decision tree for method selection:**

```
Is supplier-specific cradle-to-gate data available?
  YES --> Use Supplier-Specific Method
  NO  --> Is physical quantity data available?
            YES --> Are process/product-level EFs available?
                      YES --> Use Average-Data Method
                      NO  --> Is spend data available?
                                YES --> Use Hybrid Method (quantity + spend)
                                NO  --> Estimate and document
            NO  --> Is spend data available?
                      YES --> Use Spend-Based Method
                      NO  --> Estimate using industry benchmarks
```

### 2.2 Supplier-Specific Method

#### 2.2.1 Description

The supplier-specific method uses primary data from suppliers on the
cradle-to-gate GHG emissions of their products or services. This is the
most accurate method and is the goal state for high-impact procurement
categories.

#### 2.2.2 Data Requirements

| Data Point | Source | Format |
|-----------|--------|--------|
| Product-level GHG emissions (cradle-to-gate) | Supplier EPD, LCA, or direct disclosure | kgCO2e per unit |
| Quantity of each product purchased | Procurement system, purchase orders | Physical units (kg, m3, pieces, etc.) |
| Allocation method (if shared production) | Supplier disclosure | Revenue, mass, or economic allocation |
| Boundary of supplier data | Supplier methodology description | Cradle-to-gate vs. gate-to-gate |
| GHG gases included | Supplier methodology | CO2, CH4, N2O, HFCs, PFCs, SF6, NF3 |
| GWP values used | Supplier methodology | IPCC AR5 or AR6 |
| Third-party verification status | Verification statement | Verified / Not verified |

#### 2.2.3 Data Sources for Supplier-Specific Data

1. **Environmental Product Declarations (EPDs)**: Standardized LCA-based
   documents per ISO 14025, published by program operators (EPD International,
   UL Environment, IBU, INIES). EPDs provide cradle-to-gate (or cradle-to-grave)
   GWP data per declared unit.

2. **CDP Supply Chain Program**: Major companies request Scope 1, 2, and 3
   data from suppliers through CDP's platform. Over 35,000 suppliers reported
   through CDP in 2024. The CDP allocation methodology allows suppliers to
   allocate their emissions to specific customers.

3. **EcoVadis**: Provides sustainability ratings for over 130,000 companies
   across 220+ industries. While EcoVadis does not provide product-level
   emission factors, it provides supplier-level carbon intensity data that
   can be used as a proxy.

4. **Direct supplier engagement**: Companies send questionnaires to suppliers
   requesting GHG data. This is common for Tier 1 suppliers of raw materials
   and key components.

5. **Supplier sustainability reports**: Published GHG inventories in annual
   sustainability or ESG reports. Allocation to specific products requires
   estimation.

6. **Product Carbon Footprints (PCFs)**: ISO 14067-compliant carbon footprints
   calculated by suppliers for specific products. Growing in use, especially in
   automotive, electronics, and chemicals industries.

#### 2.2.4 Calculation Steps

```
Step 1: Identify all purchased goods/services with supplier-specific data
Step 2: Collect cradle-to-gate emission data from each supplier
Step 3: Verify boundary completeness (all upstream tiers, all GHGs)
Step 4: Apply allocation if supplier data covers shared production
Step 5: Multiply allocated EF by quantity purchased
Step 6: Sum across all supplier-specific items
Step 7: Assess and document data quality
```

#### 2.2.5 Allocation Methods

When supplier data represents total facility or product-line emissions
rather than product-specific data, allocation is required:

| Allocation Method | Formula | Use Case |
|------------------|---------|----------|
| Revenue-based | Emissions_allocated = Emissions_total * (Revenue_customer / Revenue_total) | Services, diverse product mix |
| Mass-based | Emissions_allocated = Emissions_total * (Mass_customer / Mass_total) | Commodities, bulk materials |
| Economic allocation | Emissions_allocated = Emissions_total * (Economic_value_product / Economic_value_total) | Multi-product facilities |
| Physical allocation | Emissions_allocated = Emissions_total * (Units_customer / Units_total) | Single-product or similar products |

### 2.3 Hybrid Method

#### 2.3.1 Description

The hybrid method combines supplier-specific data for some inputs with
secondary data (average-data or spend-based) for the remaining inputs. This
is the most practical approach for organizations transitioning from spend-based
to supplier-specific methods.

#### 2.3.2 Application Strategy

```
Tier 1: Top 20 suppliers by spend --> Supplier-specific data (aim for 60-80% of spend)
Tier 2: Next 30-50 suppliers --> Average-data method (physical quantities)
Tier 3: Remaining suppliers (long tail) --> Spend-based method
```

#### 2.3.3 Data Requirements

- Supplier-specific emission data for Tier 1 suppliers
- Physical quantity data (kg, liters, units) for Tier 2 suppliers
- Spend data (USD, EUR) for Tier 3 suppliers
- Process-based emission factors (ecoinvent, GaBi) for Tier 2
- EEIO emission factors (EPA USEEIO, EXIOBASE) for Tier 3

#### 2.3.4 Calculation Steps

```
Step 1: Classify all purchases into supplier-specific, average-data,
        or spend-based buckets
Step 2: Calculate emissions for each bucket using the appropriate method
Step 3: Sum across all buckets
Step 4: Apply data quality scores reflecting the mix of methods
Step 5: Track percentage of total by each method for trend reporting
```

#### 2.3.5 Weighting and Aggregation

The hybrid result is simply the sum of emissions calculated by each method:

```
Emissions_hybrid = Emissions_supplier_specific + Emissions_average_data + Emissions_spend_based
```

Data quality is the weighted average of DQI scores:

```
DQI_hybrid = (Spend_SS * DQI_SS + Spend_AD * DQI_AD + Spend_SB * DQI_SB) / Spend_total
```

Where:
- SS = Supplier-specific
- AD = Average-data
- SB = Spend-based

### 2.4 Average-Data Method

#### 2.4.1 Description

The average-data method uses mass, volume, or other physical units of
purchased goods/services multiplied by industry-average cradle-to-gate
emission factors. This method produces more accurate results than spend-based
because it uses physical quantities rather than monetary values.

#### 2.4.2 Data Requirements

| Data Point | Source | Example |
|-----------|--------|---------|
| Mass/volume/units of each product | Procurement system, BOM | 500 tonnes of steel |
| Industry-average emission factor | LCA database | 1.85 kgCO2e/kg (hot-rolled steel) |
| Product specification | Purchase orders | Grade, alloy, form factor |
| Country of origin | Supply chain data | China, Germany, USA |

#### 2.4.3 Key Emission Factor Sources for Average-Data

| Database | Coverage | Units | Cost | Update Cycle |
|----------|---------|-------|------|--------------|
| ecoinvent v3.11 | 21,000+ datasets, global | kgCO2e/unit | Commercial license | Annual |
| GaBi (Sphera) | 15,000+ datasets, global | kgCO2e/unit | Commercial license | Annual |
| DEFRA/DESNZ | UK focus, 200+ materials | kgCO2e/unit | Free | Annual (June) |
| ICE (Bath) | 200+ construction materials | kgCO2e/kg | Free | Periodic |
| World Steel Association | Steel products | kgCO2e/tonne | Free | Annual |
| Aluminum Association | Aluminum products | kgCO2e/tonne | Free | Biennial |
| PlasticsEurope | Plastic resins and products | kgCO2e/kg | Free | Periodic |
| ELCD/ILCD | European reference data | kgCO2e/unit | Free | Periodic |

#### 2.4.4 Example Physical Emission Factors

| Material | Emission Factor | Unit | Source |
|----------|----------------|------|--------|
| Primary steel (BOF) | 2.33 | kgCO2e/kg | World Steel 2023 |
| Secondary steel (EAF) | 0.67 | kgCO2e/kg | World Steel 2023 |
| Primary aluminum (global avg) | 16.7 | kgCO2e/kg | IAI 2023 |
| Secondary aluminum | 0.5 | kgCO2e/kg | IAI 2023 |
| Cement (Portland, global avg) | 0.63 | kgCO2e/kg | GCCA 2023 |
| Float glass | 1.20 | kgCO2e/kg | Glass Alliance Europe |
| HDPE (high-density polyethylene) | 1.80 | kgCO2e/kg | PlasticsEurope 2022 |
| LDPE (low-density polyethylene) | 2.08 | kgCO2e/kg | PlasticsEurope 2022 |
| PP (polypropylene) | 1.63 | kgCO2e/kg | PlasticsEurope 2022 |
| PET (polyethylene terephthalate) | 2.15 | kgCO2e/kg | PlasticsEurope 2022 |
| Corrugated cardboard | 0.79 | kgCO2e/kg | FEFCO 2022 |
| Kraft paper | 1.06 | kgCO2e/kg | CEPI 2022 |
| Recycled paper | 0.61 | kgCO2e/kg | CEPI 2022 |
| Cotton fiber (conventional) | 5.90 | kgCO2e/kg | Textile Exchange 2023 |
| Cotton fiber (organic) | 3.80 | kgCO2e/kg | Textile Exchange 2023 |
| Polyester fiber | 3.40 | kgCO2e/kg | Textile Exchange 2023 |
| Sawn timber (softwood) | 0.31 | kgCO2e/kg | ICE v3.0 |
| Sawn timber (hardwood) | 0.42 | kgCO2e/kg | ICE v3.0 |
| Copper (primary) | 4.10 | kgCO2e/kg | ICA 2022 |
| Lithium carbonate | 7.30 | kgCO2e/kg | IEA 2023 |
| Silicon wafer (solar grade) | 50.0 | kgCO2e/kg | IEA 2023 |
| Concrete (ready-mix, 30MPa) | 0.13 | kgCO2e/kg | GCCA 2023 |

### 2.5 Spend-Based Method

#### 2.5.1 Description

The spend-based method estimates emissions for purchased goods and services
by multiplying the economic value of each procurement category by an
environmentally-extended input-output (EEIO) emission factor. This method has
the broadest coverage (applies to all spend data) but the lowest accuracy.

#### 2.5.2 Data Requirements

| Data Point | Source | Format |
|-----------|--------|--------|
| Spend by procurement category | ERP/Accounts Payable | USD/EUR/GBP |
| Industry classification code | Procurement taxonomy | NAICS, NACE, ISIC, UNSPSC |
| EEIO emission factor | EPA USEEIO, EXIOBASE | kgCO2e per USD/EUR |
| Year of spend | Financial records | Calendar or fiscal year |
| Currency of spend | Financial records | ISO 4217 code |
| Base year of EEIO factors | EEIO database metadata | e.g., 2021 USD |

#### 2.5.3 Key EEIO Databases

**EPA USEEIO (US Environmentally-Extended Input-Output)**
- Current version: v2.0.1-411 (released 2024)
- Commodity emission factors: v1.2 (2023) with 1,016 commodities by NAICS-6
- Latest: v1.3 (2024) further updates
- Base year: 2019 emissions in 2021 USD
- Coverage: US economy, 389 industry sectors
- Includes: direct + indirect (supply chain) emissions
- GHG gases: CO2, CH4, N2O, HFCs, PFCs, SF6 (in CO2e using GWP)
- Download: data.gov and EPA Science Inventory

**EXIOBASE 3**
- Current version: 3.8.2
- Coverage: 49 regions (44 countries + 5 rest-of-world), 163 product groups,
  200 industry sectors
- Approximately 9,800 emission factors (200 products x 49 regions)
- Base year: Various (latest monetary tables in EUR)
- Includes: Production-based and consumption-based accounts
- GHG gases: CO2, CH4, N2O, and F-gases
- Download: exiobase.eu (open access)

**WIOD (World Input-Output Database)**
- Current version: 2016 release
- Coverage: 43 countries, 56 sectors
- Time series: 2000-2014
- Limitation: Older data, less granular than EXIOBASE
- Download: wiod.org

**GTAP (Global Trade Analysis Project)**
- Current version: GTAP 11
- Coverage: 141 countries/regions, 65 sectors
- Base year: 2017
- Focus: Trade-linked emissions
- Access: Requires subscription from Purdue University

#### 2.5.4 EPA USEEIO Emission Factor Examples (kgCO2e per USD)

| NAICS Code | Sector | Factor (kgCO2e/$) | Notes |
|-----------|--------|-------------------|-------|
| 1111 | Oilseed and grain farming | 0.88 | High due to fertilizer, land use |
| 1121 | Cattle ranching | 1.20 | Highest agriculture factor (methane) |
| 2111 | Oil and gas extraction | 0.95 | Includes fugitive emissions |
| 2122 | Metal ore mining | 0.72 | Varies by ore type |
| 2211 | Electric power generation | 0.85 | National average grid mix |
| 3111 | Animal food manufacturing | 0.65 | Feed ingredients contribution |
| 3221 | Pulp, paper, and paperboard | 0.52 | Energy-intensive process |
| 3241 | Petroleum and coal products | 1.45 | Highest manufacturing factor |
| 3251 | Basic chemical manufacturing | 0.78 | Process + energy emissions |
| 3311 | Iron and steel mills | 0.82 | Coke and blast furnace emissions |
| 3341 | Computer and peripheral equipment | 0.25 | Low relative to value |
| 3361 | Motor vehicle manufacturing | 0.32 | Assembly + supply chain |
| 4411 | Automobile dealers | 0.08 | Primarily retail markup |
| 4841 | General freight trucking | 0.75 | High fuel intensity |
| 5112 | Software publishers | 0.03 | Very low physical intensity |
| 5182 | Data processing, hosting | 0.10 | Electricity-dominated |
| 5191 | Other information services | 0.05 | Low physical intensity |
| 5411 | Legal services | 0.06 | Office-based services |
| 5412 | Accounting, tax, bookkeeping | 0.05 | Office-based services |
| 5413 | Architectural, engineering | 0.07 | Office-based services |
| 5415 | Computer systems design | 0.04 | Office-based services |
| 5416 | Management consulting | 0.06 | Office + travel |
| 5511 | Management of companies | 0.08 | Holding company overhead |
| 5611 | Office administrative services | 0.05 | Office-based |
| 5613 | Employment services | 0.04 | Temporary staffing |
| 5617 | Services to buildings | 0.12 | Cleaning, janitorial |
| 6111 | Elementary and secondary schools | 0.09 | Buildings + transport |
| 7211 | Traveler accommodation | 0.18 | Hotels, energy use |
| 7221 | Full-service restaurants | 0.22 | Food + energy |
| 8111 | Automotive repair | 0.15 | Parts + energy |

**Note**: These are representative values. Actual factors vary by version,
geography, and methodology. The agent must always reference the specific
version and vintage of the EEIO database used.

---

## 3. Emission Factor Sources and Databases

### 3.1 EEIO Databases (Spend-Based Factors)

#### 3.1.1 EPA USEEIO Detailed Structure

The EPA Supply Chain Greenhouse Gas Emission Factors dataset is the primary
US source for spend-based Scope 3 calculations. Key characteristics:

**Model Architecture:**
- Based on the US Bureau of Economic Analysis (BEA) Make and Use tables
- Links economic transactions between 389 industry sectors with environmental
  satellite accounts
- Produces "supply chain emission factors" that capture total upstream
  (cradle-to-gate) emissions per dollar of output
- Factors include: direct emissions at the producer + all upstream supply
  chain emissions

**Factor Types (per commodity):**
- `Supply Chain Emission Factors with Margins`: Include wholesale, retail,
  and transport margins (use when spend includes these costs)
- `Supply Chain Emission Factors without Margins`: Producer price only
  (use when spend reflects producer prices)

**Version History:**
| Version | Release | Base Year | Commodities | Price Year |
|---------|---------|-----------|-------------|------------|
| v1.0 | 2020 | 2012 | 1,016 | 2018 USD |
| v1.1 | 2022 | 2016 | 1,016 | 2018 USD |
| v1.2 | 2023 | 2019 | 1,016 | 2021 USD |
| v1.3 | 2024 | 2019 | 1,016 | 2021 USD |

**Sector Coverage:**
- Agriculture: NAICS 111-115 (45 sectors)
- Mining: NAICS 211-213 (18 sectors)
- Utilities: NAICS 221 (8 sectors)
- Construction: NAICS 236-238 (17 sectors)
- Manufacturing: NAICS 311-339 (290 sectors)
- Wholesale/Retail: NAICS 420-454 (30 sectors)
- Transportation: NAICS 481-493 (25 sectors)
- Information: NAICS 511-519 (18 sectors)
- Finance/Insurance: NAICS 521-525 (20 sectors)
- Professional Services: NAICS 541-561 (30+ sectors)
- Healthcare: NAICS 621-624 (20 sectors)
- Other Services: NAICS 711-814 (40+ sectors)

#### 3.1.2 EXIOBASE Detailed Structure

EXIOBASE is the primary multi-regional EEIO database for non-US spend:

**Model Architecture:**
- Multi-Regional Environmentally-Extended Supply-Use / Input-Output
  (MR EE SUT/IOT) database
- Harmonizes national supply-use tables for 44 countries + 5 rest-of-world
  regions into a global model
- Links economic flows with environmental extensions (emissions, resources,
  land use, water)

**Product Coverage (163 product groups):**
- Primary products: crops, forestry, fishing, mining (40+ groups)
- Manufactured products: food, textiles, wood, paper, chemicals, metals,
  machinery, electronics, vehicles (80+ groups)
- Services: construction, trade, transport, business services, public
  services (40+ groups)

**Regional Coverage (49 regions):**
- 28 EU member states (pre-Brexit)
- 16 major economies (US, China, Japan, South Korea, Brazil, India, etc.)
- 5 rest-of-world aggregates (Asia, Americas, Africa, Middle East, Europe)

**Emission Types:**
- CO2 combustion, CO2 non-combustion (cement, etc.)
- CH4, N2O, SF6, HFCs, PFCs
- Results in kgCO2e per EUR of output

#### 3.1.3 EEIO Factor Comparison

| Feature | EPA USEEIO | EXIOBASE | WIOD | GTAP |
|---------|-----------|----------|------|------|
| Regional scope | US only | 49 regions | 43 countries | 141 regions |
| Sector detail | 1,016 (NAICS-6) | 163 products | 56 sectors | 65 sectors |
| Currency | USD | EUR | USD | USD |
| Latest base year | 2019 | Varies | 2014 | 2017 |
| Cost | Free | Free | Free | Subscription |
| Best for | US procurement | EU/global | Time series | Trade flows |
| Update frequency | Annual | Periodic | Ceased | Periodic |

### 3.2 Physical Emission Factor Databases (Average-Data Factors)

#### 3.2.1 ecoinvent

The ecoinvent database is the gold standard for process-based LCA data:

**Coverage:**
- 21,000+ Life Cycle Inventory datasets
- Covers: energy, transport, building materials, chemicals, metals,
  agriculture, waste treatment, electronics
- Geography: Global, with regional datasets for 50+ countries
- System models: Allocation at Point of Substitution (APOS), Consequential,
  Cut-off by Classification

**Scope 3 Application:**
- Provides cradle-to-gate emission factors for thousands of products
- Results available in multiple impact assessment methods (IPCC GWP100,
  ReCiPe, etc.)
- Can be used directly as average-data emission factors

**Access:**
- Commercial license required (academic and commercial tiers)
- Integration via API (ecoQuery) or exported files
- Version 3.11 (2024) is the latest release

#### 3.2.2 GaBi (Sphera)

**Coverage:**
- 15,000+ LCI plans and processes
- Strong coverage of automotive, chemicals, electronics, packaging
- Regional data for 100+ countries
- Regularly updated with industry data

**Scope 3 Application:**
- Product-level cradle-to-gate GHG factors
- Integrated with Sphera's corporate sustainability platform
- Used by many automotive and electronics OEMs for supply chain calculations

#### 3.2.3 DEFRA/DESNZ Emission Factors

The UK government publishes annual conversion factors:

**Scope 3 Upstream Factors (Table 13):**
- Material-based factors for common goods (kgCO2e per unit)
- Categories: construction, food, drink, clothing, electronics, paper,
  organic materials, plastics, metals, glass, chemicals
- Updated annually (typically June release)
- Free to download from GOV.UK

**Example DEFRA Material Factors:**

| Material | Factor | Unit | Year |
|----------|--------|------|------|
| Aggregates | 7.76 | kgCO2e/tonne | 2025 |
| Average construction | 79.22 | kgCO2e/tonne | 2025 |
| Asphalt | 39.21 | kgCO2e/tonne | 2025 |
| Bricks | 241.00 | kgCO2e/tonne | 2025 |
| Concrete | 131.76 | kgCO2e/tonne | 2025 |
| Insulation | 1,861.00 | kgCO2e/tonne | 2025 |
| Metals (average) | 3,637.00 | kgCO2e/tonne | 2025 |
| Soils | 21.00 | kgCO2e/tonne | 2025 |
| Mineral oil | 676.00 | kgCO2e/tonne | 2025 |
| Plasterboard | 120.05 | kgCO2e/tonne | 2025 |
| Tyres | 3,335.00 | kgCO2e/tonne | 2025 |
| Wood | 312.68 | kgCO2e/tonne | 2025 |
| Books (paper) | 946.00 | kgCO2e/tonne | 2025 |
| Glass (average) | 1,402.00 | kgCO2e/tonne | 2025 |
| Clothing | 22,107.00 | kgCO2e/tonne | 2025 |
| Food and drink (average) | 3,700.00 | kgCO2e/tonne | 2025 |
| Organic (garden) | 578.00 | kgCO2e/tonne | 2025 |
| Electrical items | 5,502.00 | kgCO2e/tonne | 2025 |
| Plastics (average) | 3,116.00 | kgCO2e/tonne | 2025 |

#### 3.2.4 ICE (Inventory of Carbon and Energy)

Developed by the University of Bath, the ICE database focuses on construction
and building materials:

**Coverage:**
- 200+ construction materials across 30+ categories
- Includes embodied carbon (kgCO2e/kg) for each material
- Originally included embodied energy; since 2019, carbon-only
- Free to access

**Key Material Factors:**

| Material | Embodied Carbon | Unit |
|----------|----------------|------|
| General steel (world avg, recycled content) | 1.37 | kgCO2e/kg |
| Virgin steel (100% primary) | 2.89 | kgCO2e/kg |
| Recycled steel (100% EAF) | 0.47 | kgCO2e/kg |
| General aluminum (33% recycled) | 6.67 | kgCO2e/kg |
| Virgin aluminum | 12.79 | kgCO2e/kg |
| Recycled aluminum | 0.84 | kgCO2e/kg |
| Concrete (general, 1:2:4 mix) | 0.13 | kgCO2e/kg |
| Concrete (high strength, 50MPa) | 0.17 | kgCO2e/kg |
| Cement (Portland CEM I) | 0.91 | kgCO2e/kg |
| Timber (general sawn) | 0.31 | kgCO2e/kg |
| Timber (glulam) | 0.51 | kgCO2e/kg |
| Bricks (general) | 0.24 | kgCO2e/kg |
| Glass (general float) | 1.22 | kgCO2e/kg |
| Copper (average) | 3.81 | kgCO2e/kg |
| Lead | 1.57 | kgCO2e/kg |
| Zinc | 3.86 | kgCO2e/kg |
| Plasterboard | 0.39 | kgCO2e/kg |

#### 3.2.5 Industry-Specific Sources

| Source | Coverage | Access |
|--------|---------|--------|
| World Steel Association | Steel products worldwide | Free annual data |
| International Aluminium Institute (IAI) | Primary and recycled aluminum | Free |
| Global Cement & Concrete Assoc. (GCCA) | Cement and concrete | Free |
| PlasticsEurope | Plastic resins (Eco-profiles) | Free |
| CEPI (Confederation of European Paper Industries) | Paper and pulp | Free |
| Textile Exchange | Cotton, polyester, nylon, wool | Free (Preferred Fiber & Materials report) |
| International Copper Association (ICA) | Copper products | Free |
| Glass Alliance Europe | Flat glass, container glass | Free |
| FEFCO (European Corrugated Packaging) | Corrugated board | Free |
| International Energy Agency (IEA) | Energy system materials | Subscription |

### 3.3 Supplier Data Platforms

| Platform | Type | Coverage | Data Provided |
|----------|------|---------|---------------|
| CDP Supply Chain | Questionnaire | 35,000+ suppliers | Scope 1/2/3, targets, actions |
| EcoVadis | Rating | 130,000+ companies | Sustainability score, carbon rating |
| Ecodesk | Platform | 50,000+ products | Product carbon footprints |
| Higg Index (SAC) | Index | Apparel/textiles | Facility and product scores |
| SEDEX/Smeta | Audit | 85,000+ sites | Social and environmental audits |
| Open LCA Nexus | Database | 100,000+ datasets | LCA data, emission factors |
| PACT Network (WBCSD) | Exchange | Growing | Product-level PCF exchange |

### 3.4 Emission Factor Selection Hierarchy

The agent should implement a hierarchical emission factor selection:

```
Priority 1: Supplier-specific EPD or PCF data (ISO 14025/14067 verified)
Priority 2: Supplier-specific CDP/direct disclosure data (unverified)
Priority 3: Product-specific LCA data (ecoinvent, GaBi)
Priority 4: Material-specific average EF (ICE, World Steel, DEFRA)
Priority 5: Industry-average physical EF (literature, industry reports)
Priority 6: Regional EEIO factor (EXIOBASE for non-US)
Priority 7: National EEIO factor (EPA USEEIO for US)
Priority 8: Global average EEIO factor (fallback)
```

---

## 4. Spend Classification Systems

### 4.1 Overview of Classification Systems

Spend classification is critical for the spend-based method because EEIO
emission factors are mapped to industry classification codes. The agent must
support multiple classification systems and provide cross-mapping capabilities.

### 4.2 NAICS (North American Industry Classification System)

**Structure:**
- 6-digit hierarchical code
- Levels: Sector (2-digit) > Subsector (3-digit) > Industry Group (4-digit) >
  Industry (5-digit) > National Industry (6-digit)
- Current version: NAICS 2022 (updated every 5 years)
- Coverage: US, Canada, Mexico

**Sector Summary (2-digit):**

| Code | Sector | Example Subcategories |
|------|--------|--------------------|
| 11 | Agriculture, Forestry, Fishing, Hunting | Crop production, animal production |
| 21 | Mining, Quarrying, Oil/Gas | Oil/gas, metal mining, coal |
| 22 | Utilities | Electric power, natural gas, water |
| 23 | Construction | Building, heavy civil, specialty |
| 31-33 | Manufacturing | Food, chemicals, metals, electronics, vehicles |
| 42 | Wholesale Trade | Durable and nondurable goods |
| 44-45 | Retail Trade | Motor vehicles, food, general merchandise |
| 48-49 | Transportation & Warehousing | Air, rail, truck, pipeline, postal |
| 51 | Information | Publishing, telecom, data processing |
| 52 | Finance & Insurance | Banking, securities, insurance |
| 53 | Real Estate & Rental | Real estate, rental/leasing |
| 54 | Professional, Scientific, Technical | Legal, accounting, architecture, consulting |
| 55 | Management of Companies | Holding companies |
| 56 | Administrative & Support | Office admin, employment services, security |
| 61 | Educational Services | Schools, universities, training |
| 62 | Healthcare & Social Assistance | Hospitals, physicians, nursing |
| 71 | Arts, Entertainment, Recreation | Performing arts, museums, sports |
| 72 | Accommodation & Food Services | Hotels, restaurants |
| 81 | Other Services | Repair, personal care, organizations |
| 92 | Public Administration | Government agencies |

**Application to Scope 3:**
- EPA USEEIO factors are indexed by NAICS-6 code
- Most US procurement systems use NAICS for vendor classification
- 1,016 commodity-level factors available at NAICS-6 level

### 4.3 NACE (Statistical Classification of Economic Activities in the European Community)

**Structure:**
- 4-digit hierarchical code (+ 2-digit national extensions in some countries)
- Levels: Section (letter) > Division (2-digit) > Group (3-digit) > Class (4-digit)
- Current version: NACE Rev. 2.1 (2023 update)
- Coverage: EU member states, widely used in Europe

**Section Summary:**

| Code | Section | Examples |
|------|---------|---------|
| A | Agriculture, Forestry, Fishing | Crop and animal production |
| B | Mining and Quarrying | Metal ores, petroleum, gas |
| C | Manufacturing | Food, textiles, chemicals, metals |
| D | Electricity, Gas, Steam, AC | Power generation and distribution |
| E | Water Supply, Sewerage, Waste | Water collection, waste management |
| F | Construction | Building, civil engineering |
| G | Wholesale and Retail Trade | Motor vehicles, food, household |
| H | Transportation and Storage | Land, water, air transport |
| I | Accommodation and Food Service | Hotels, restaurants, catering |
| J | Information and Communication | Publishing, telecom, IT, media |
| K | Financial and Insurance | Banking, insurance, pensions |
| L | Real Estate | Buying, selling, renting property |
| M | Professional, Scientific, Technical | Legal, accounting, R&D, advertising |
| N | Administrative and Support | Employment, security, cleaning |
| O | Public Administration and Defence | Government, military |
| P | Education | All levels of education |
| Q | Human Health and Social Work | Hospitals, social care |
| R | Arts, Entertainment, Recreation | Creative, sports, amusement |
| S | Other Service Activities | Repair, personal services |

**Application to Scope 3:**
- EXIOBASE factors map to NACE divisions/groups
- European companies typically classify vendors by NACE code
- CSRD reporting uses NACE for activity classification

### 4.4 ISIC (International Standard Industrial Classification)

**Structure:**
- 4-digit hierarchical code
- Levels: Section (letter) > Division (2-digit) > Group (3-digit) > Class (4-digit)
- Current version: ISIC Rev. 4.1 (2024)
- Coverage: United Nations standard, used worldwide

**Application to Scope 3:**
- ISIC serves as the common bridge between NAICS and NACE
- Many international EEIO databases reference ISIC codes
- UN correspondence tables link ISIC to NAICS and NACE

### 4.5 UNSPSC (United Nations Standard Products and Services Code)

**Structure:**
- 8-digit hierarchical code
- Levels: Segment (2-digit) > Family (4-digit) > Class (6-digit) >
  Commodity (8-digit)
- Current version: v28.0 (2024)
- Coverage: 55 segments, 400+ families, 5,000+ classes, 60,000+ commodities

**Segment Summary:**

| Code | Segment | Scope 3 Relevance |
|------|---------|-------------------|
| 10 | Live plant and animal material | Category 1 (agriculture) |
| 11 | Mineral and textile and chemical material | Category 1 (raw materials) |
| 12 | Chemicals including bio-chemicals | Category 1 (chemicals) |
| 13 | Resin, rosin, rubber, foam, film | Category 1 (polymers) |
| 14 | Paper materials and products | Category 1 (paper goods) |
| 15 | Fuels, additives, lubricants | Category 3 (fuel/energy) |
| 20 | Mining, well drilling machinery | Category 2 (capital goods) |
| 21 | Agriculture, fishing, forestry equipment | Category 2 (capital goods) |
| 23 | Industrial manufacturing equipment | Category 2 (capital goods) |
| 24 | Material handling equipment | Category 2 (capital goods) |
| 25 | Commercial vehicles and transport | Category 2 or 6 |
| 26 | Power generation/distribution equipment | Category 2 (capital goods) |
| 30 | Structures, building, construction | Category 2 (capital goods) |
| 31 | Manufacturing components and supplies | Category 1 (components) |
| 39 | Lighting, electrical, electronic | Category 1 or 2 |
| 40 | Distribution, conditioning systems | Category 2 (HVAC) |
| 41 | Laboratory, measuring equipment | Category 1 or 2 |
| 42 | Medical equipment and supplies | Category 1 (healthcare) |
| 43 | Information technology equipment | Category 1 or 2 |
| 44 | Office equipment and supplies | Category 1 (supplies) |
| 45 | Printing and photographic equipment | Category 1 or 2 |
| 46 | Defense and law enforcement | Category 1 or 2 |
| 47 | Cleaning equipment and supplies | Category 1 |
| 48 | Service industry equipment | Category 1 or 2 |
| 49 | Sports and recreation equipment | Category 1 |
| 50 | Food, beverage, tobacco | Category 1 (F&B) |
| 51 | Drugs and pharmaceutical products | Category 1 (pharma) |
| 52 | Domestic appliances, supplies | Category 1 (consumer goods) |
| 53 | Apparel, luggage, personal care | Category 1 (textiles) |
| 55 | Published products | Category 1 (publishing) |
| 56 | Furniture and furnishings | Category 1 or 2 |
| 60 | Musical instruments, games, toys | Category 1 |
| 70 | Farming, fishing, forestry services | Category 1 (services) |
| 71 | Mining and oil/gas services | Category 1 (services) |
| 72 | Building/construction services | Category 1 (services) |
| 73 | Industrial production services | Category 1 (services) |
| 76 | Industrial cleaning services | Category 1 (services) |
| 77 | Environmental services | Category 1 (services) |
| 78 | Transportation and storage services | Category 4/9 (transport) |
| 80 | Management and business services | Category 1 (services) |
| 81 | Engineering and R&D services | Category 1 (services) |
| 82 | Editorial and design services | Category 1 (services) |
| 83 | Public utilities and public services | Category 3 |
| 84 | Financial and insurance services | Category 1 (services) |
| 85 | Healthcare services | Category 1 (services) |
| 86 | Education and training services | Category 1 (services) |
| 90 | Travel, food, lodging services | Category 6 (travel) |
| 91 | Personal and domestic services | Category 1 (services) |
| 92 | National defense, public order | Category 1 (services) |
| 93 | Political and civic affairs | Category 1 (services) |
| 94 | Organizations and clubs | Category 1 (services) |
| 95 | Land, buildings, structures | Category 2 or 8 |

**Application to Scope 3:**
- UNSPSC is the most granular product classification system
- Many procurement platforms (SAP Ariba, Coupa, Jaggaer) use UNSPSC
- Mapping UNSPSC to NAICS or NACE codes enables EEIO factor lookup
- UNSPSC can distinguish Cat 1 (goods/services) from Cat 2 (capital goods)

### 4.6 Cross-Mapping Between Systems

The agent must support mapping between classification systems:

| From | To | Mapping Method | Source |
|------|----|---------------|--------|
| NAICS -> ISIC | Correspondence table | UN Statistics Division |
| NACE -> ISIC | Correspondence table | Eurostat |
| ISIC -> NAICS | Correspondence table | US Census Bureau |
| ISIC -> NACE | Correspondence table | Eurostat |
| UNSPSC -> NAICS | Fuzzy mapping | Custom + isdata-org/what-links-to-what |
| UNSPSC -> NACE | Via ISIC intermediate | Custom chain mapping |

**Mapping Challenges:**
- Many-to-many relationships between systems
- Different granularity levels (NAICS-6 vs. NACE-4 vs. ISIC-4)
- Service sectors often poorly mapped between systems
- Custom products may not fit any standard code
- Requires confidence scoring for ambiguous mappings

### 4.7 Top Spending Categories by Industry

Understanding typical procurement patterns helps prioritize:

**Manufacturing (automotive, electronics, machinery):**

| Rank | Category | % of Spend | Typical NAICS | EF Intensity |
|------|---------|-----------|---------------|-------------|
| 1 | Metals and alloys | 15-25% | 3311-3316 | High |
| 2 | Electronic components | 10-20% | 3344 | Medium |
| 3 | Plastic and rubber parts | 8-15% | 3261-3262 | Medium-High |
| 4 | Chemicals and coatings | 5-10% | 3251-3259 | High |
| 5 | Packaging materials | 3-8% | 3221-3222 | Medium |
| 6 | Contract manufacturing | 5-15% | 3329 | Medium |
| 7 | IT and software | 3-7% | 5112, 5415 | Low |
| 8 | Professional services | 3-5% | 5411-5419 | Low |
| 9 | Logistics (outsourced) | 3-8% | 4841, 4931 | High |
| 10 | Facilities management | 2-5% | 5617 | Low |

**Retail (consumer goods, apparel, grocery):**

| Rank | Category | % of Spend | Typical NAICS | EF Intensity |
|------|---------|-----------|---------------|-------------|
| 1 | Goods for resale (COGS) | 50-70% | Various 311-339 | High |
| 2 | Packaging | 5-10% | 3221-3222 | Medium |
| 3 | Store operations (cleaning, security) | 3-7% | 5611-5619 | Low |
| 4 | Marketing and advertising | 3-5% | 5418 | Low |
| 5 | IT systems | 2-5% | 5112, 5182 | Low |

**Financial Services (banking, insurance):**

| Rank | Category | % of Spend | Typical NAICS | EF Intensity |
|------|---------|-----------|---------------|-------------|
| 1 | IT and data services | 20-35% | 5112, 5182, 5415 | Low |
| 2 | Professional services | 15-25% | 5411-5419 | Low |
| 3 | Real estate / facilities | 10-15% | 5311 | Medium |
| 4 | Marketing | 5-10% | 5418 | Low |
| 5 | Printing and stationery | 2-5% | 3231 | Medium |

---

## 5. Key Formulas and Calculations

### 5.1 Spend-Based Method Formula

**Core Formula:**

```
Emissions_cat1_spend = SUM over all categories c:
    Spend_c (in base currency) * EEIO_factor_c (kgCO2e per base currency unit)
```

**With currency conversion and inflation adjustment:**

```
Emissions_cat1_spend = SUM over all categories c:
    (Spend_c_local / FX_rate_to_base) * (Price_index_base_year / Price_index_spend_year) * EEIO_factor_c
```

Where:
- `Spend_c_local`: Spend in local currency for category c
- `FX_rate_to_base`: Exchange rate from local currency to EEIO base currency
  (e.g., USD for EPA USEEIO, EUR for EXIOBASE)
- `Price_index_base_year`: CPI or PPI for the EEIO base year
- `Price_index_spend_year`: CPI or PPI for the year of actual spend
- `EEIO_factor_c`: EEIO emission factor for category c

**Margin Adjustment:**

When spend includes retail/wholesale margins but the EEIO factor is at
producer prices:

```
Spend_producer = Spend_purchaser * (1 - margin_rate_c)
```

Where `margin_rate_c` is the average wholesale + retail + transport margin
for category c (typically 15-40% for goods, 5-15% for services).

### 5.2 Average-Data Method Formula

**Core Formula:**

```
Emissions_cat1_avg = SUM over all products p:
    Quantity_p * EF_p
```

Where:
- `Quantity_p`: Physical quantity of product p purchased (kg, m3, liters, units)
- `EF_p`: Cradle-to-gate emission factor for product p (kgCO2e per physical unit)

**With transportation adder:**

```
Emissions_cat1_avg = SUM over all products p:
    (Quantity_p * EF_production_p) + (Quantity_p * Distance_p * EF_transport_mode)
```

Where:
- `EF_production_p`: Cradle-to-gate production emission factor
- `Distance_p`: Transport distance from supplier to reporting company (km)
- `EF_transport_mode`: Transport emission factor (kgCO2e per tonne-km)

**Note**: If the EF already includes transportation to the customer (common in
cradle-to-gate LCA data), do not add transport separately to avoid double counting.

### 5.3 Supplier-Specific Method Formula

**Core Formula (product-level data):**

```
Emissions_cat1_ss = SUM over all suppliers s, products p:
    Quantity_s_p * EF_supplier_s_p
```

Where:
- `Quantity_s_p`: Quantity of product p purchased from supplier s
- `EF_supplier_s_p`: Supplier-specific cradle-to-gate emission factor for
  product p from supplier s (kgCO2e per unit)

**With allocation (facility-level data):**

```
Emissions_cat1_ss = SUM over all suppliers s:
    Emissions_total_s * Allocation_factor_s
```

**Revenue-based allocation:**

```
Allocation_factor_s = Revenue_from_reporting_company / Revenue_total_s
```

**Mass-based allocation:**

```
Allocation_factor_s = Mass_purchased_from_s / Mass_total_output_s
```

**Economic allocation:**

```
Allocation_factor_s = Economic_value_purchased / Economic_value_total_output_s
```

### 5.4 Hybrid Method Formula

**Core Formula:**

```
Emissions_cat1_hybrid = Emissions_SS + Emissions_AD + Emissions_SB
```

Where:
- `Emissions_SS`: Emissions from supplier-specific items
- `Emissions_AD`: Emissions from average-data items
- `Emissions_SB`: Emissions from spend-based items

**Coverage tracking:**

```
Coverage_SS = Spend_SS / Spend_total
Coverage_AD = Spend_AD / Spend_total
Coverage_SB = Spend_SB / Spend_total
Coverage_total = Coverage_SS + Coverage_AD + Coverage_SB (should equal 1.0)
```

### 5.5 Currency Adjustment Formulas

#### 5.5.1 Exchange Rate Conversion

```
Spend_base = Spend_local * (1 / FX_rate)
```

Where `FX_rate` is the exchange rate expressed as units of base currency per
unit of local currency. Use the average annual exchange rate for the reporting
year.

#### 5.5.2 Inflation Deflation

```
Spend_real = Spend_nominal * (GDP_deflator_base / GDP_deflator_current)
```

Or using CPI:

```
Spend_real = Spend_nominal * (CPI_base_year / CPI_current_year)
```

**Example**: EPA USEEIO v1.2 uses 2021 USD. If actual spend is 2025 USD:

```
Spend_2021_USD = Spend_2025_USD * (CPI_2021 / CPI_2025)
```

Using approximate US CPI values:
- CPI_2021 = 270.97
- CPI_2025 = 316.40 (estimated)

```
Spend_2021_USD = Spend_2025_USD * (270.97 / 316.40) = Spend_2025_USD * 0.8564
```

#### 5.5.3 Purchasing Power Parity (PPP) Adjustment

For multinational companies with spend in multiple countries, PPP adjustment
can improve accuracy when using a single country's EEIO factors:

```
Spend_PPP_adjusted = Spend_local * (PPP_factor_local / PPP_factor_base)
```

Where PPP factors are from the World Bank International Comparison Program.

**Note**: PPP adjustment is recommended but not required by the GHG Protocol.
It is most important when applying US EEIO factors to spend in countries with
very different price levels (e.g., applying EPA USEEIO factors to spend in
India or China).

### 5.6 Emissions Intensity Metrics

**Per revenue:**

```
Intensity_revenue = Emissions_cat1 / Revenue_total
```

Units: tCO2e per million USD revenue

**Per employee:**

```
Intensity_employee = Emissions_cat1 / FTE_count
```

**Per unit produced:**

```
Intensity_product = Emissions_cat1 / Units_produced
```

**Per unit spend:**

```
Intensity_spend = Emissions_cat1 / Total_procurement_spend
```

### 5.7 Year-over-Year Change Decomposition

Understanding drivers of change is required for CSRD/CDP reporting:

```
Delta_emissions = Delta_activity + Delta_EF + Delta_method + Delta_scope
```

Where:
- `Delta_activity`: Change due to procurement volume changes
- `Delta_EF`: Change due to updated emission factors
- `Delta_method`: Change due to method improvements (e.g., moving from
  spend-based to supplier-specific)
- `Delta_scope`: Change due to boundary or category scope changes

**Calculation approach:**

```
Delta_activity = (Spend_current - Spend_prior) * EF_prior
Delta_EF = Spend_current * (EF_current - EF_prior)
Delta_method = Emissions_current_new_method - Emissions_current_old_method
```

---

## 6. Data Quality Scoring

### 6.1 GHG Protocol Data Quality Framework

The GHG Protocol Scope 3 Standard (Chapter 7) defines five data quality
indicators (DQI) that should be assessed for all Scope 3 calculations:

| Indicator | Description | Scale |
|-----------|------------|-------|
| Temporal representativeness | How recent is the data? | 1 (very good) to 5 (very poor) |
| Geographical representativeness | Does the data match the geographical location? | 1 to 5 |
| Technological representativeness | Does the data match the technology used? | 1 to 5 |
| Completeness | Does the data cover all relevant emissions? | 1 to 5 |
| Reliability | Is the data verified and from authoritative sources? | 1 to 5 |

### 6.2 DQI Scoring Criteria

#### 6.2.1 Temporal Representativeness

| Score | Description | Example |
|-------|------------|---------|
| 1 | Data from the reporting year | 2025 data for FY2025 report |
| 2 | Data within 3 years | 2023-2025 data for FY2025 |
| 3 | Data within 6 years | 2020-2025 data |
| 4 | Data within 10 years | 2016-2025 data |
| 5 | Data older than 10 years | Pre-2016 data |

#### 6.2.2 Geographical Representativeness

| Score | Description | Example |
|-------|------------|---------|
| 1 | Data from the same country/region | US data for US supplier |
| 2 | Data from similar region | EU data for UK supplier |
| 3 | Data from same continent | Asia data for China supplier |
| 4 | Data from different continent | US data for Asia supplier |
| 5 | Global average or unknown geography | Global average EF |

#### 6.2.3 Technological Representativeness

| Score | Description | Example |
|-------|------------|---------|
| 1 | Data from identical technology/process | BOF steel EF for BOF steel |
| 2 | Data from similar technology | Blast furnace EF for BOF |
| 3 | Data from related technology/material | General steel EF for BOF |
| 4 | Data from broader category | Metal manufacturing EF for steel |
| 5 | Data from unrelated category | General manufacturing EF |

#### 6.2.4 Completeness

| Score | Description | Example |
|-------|------------|---------|
| 1 | All relevant sources/sinks included | Full cradle-to-gate LCA |
| 2 | More than 80% of emissions covered | LCA missing minor processes |
| 3 | 50-80% of emissions covered | Partial LCA, major gaps |
| 4 | 20-50% of emissions covered | Only direct production |
| 5 | Less than 20% of emissions covered | Single process step only |

#### 6.2.5 Reliability

| Score | Description | Example |
|-------|------------|---------|
| 1 | Third-party verified data | EPD verified by independent auditor |
| 2 | Data from peer-reviewed source | Published LCA study |
| 3 | Data from established database | ecoinvent, DEFRA factors |
| 4 | Data from non-verified source | Supplier self-reported |
| 5 | Estimate or assumption | Industry average or proxy |

### 6.3 Composite DQI Score

The composite DQI score is the arithmetic mean of the five indicators:

```
DQI_composite = (DQI_temporal + DQI_geographical + DQI_technological +
                 DQI_completeness + DQI_reliability) / 5
```

**Quality classification:**

| DQI Range | Classification | Recommended Action |
|-----------|---------------|-------------------|
| 1.0 - 1.5 | Very Good | Maintain current data quality |
| 1.6 - 2.5 | Good | Monitor for improvements |
| 2.6 - 3.5 | Fair | Prioritize improvement plan |
| 3.6 - 4.5 | Poor | Active improvement required |
| 4.6 - 5.0 | Very Poor | Urgent data quality intervention |

### 6.4 Uncertainty Ranges by Calculation Method

| Method | Typical DQI Range | Uncertainty Range | Confidence Level |
|--------|------------------|------------------|-----------------|
| Supplier-specific (verified EPD) | 1.0 - 1.5 | +/- 10-15% | Very High |
| Supplier-specific (unverified) | 1.5 - 2.5 | +/- 15-30% | High |
| Average-data (process LCA) | 2.0 - 3.0 | +/- 30-50% | Medium-High |
| Average-data (industry avg) | 2.5 - 3.5 | +/- 40-60% | Medium |
| Spend-based (regional EEIO) | 3.0 - 4.0 | +/- 50-80% | Low-Medium |
| Spend-based (national EEIO) | 3.5 - 4.5 | +/- 60-100% | Low |
| Spend-based (global average) | 4.0 - 5.0 | +/- 80-150% | Very Low |

### 6.5 Pedigree Matrix Approach

The GHG Protocol references the pedigree matrix approach (originally from
ecoinvent) for quantitative uncertainty assessment. Each DQI score maps to
an uncertainty factor:

| DQI Score | Uncertainty Factor (sigma) |
|-----------|--------------------------|
| 1 | 1.00 (no additional uncertainty) |
| 2 | 1.05 (+/- 5% additional) |
| 3 | 1.10 (+/- 10% additional) |
| 4 | 1.20 (+/- 20% additional) |
| 5 | 1.50 (+/- 50% additional) |

**Combined uncertainty:**

```
Sigma_combined = sqrt(sigma_base^2 + sigma_temporal^2 + sigma_geo^2 +
                      sigma_tech^2 + sigma_completeness^2 + sigma_reliability^2)
```

Where `sigma_base` is the inherent uncertainty of the emission factor itself.

### 6.6 Weighted Data Quality Score for Total Inventory

When multiple methods are used across the inventory:

```
DQI_weighted = SUM over all line items i:
    (Emissions_i / Emissions_total) * DQI_i
```

This produces a single weighted DQI score for the entire Category 1 inventory,
weighted by the emission contribution of each line item. Items with higher
emissions (and thus more material impact) have proportionally greater influence
on the overall quality score.

---

## 7. Regulatory Context

### 7.1 EU CSRD / ESRS E1

#### 7.1.1 Overview

The Corporate Sustainability Reporting Directive (CSRD) requires companies to
report under the European Sustainability Reporting Standards (ESRS). ESRS E1
("Climate Change") mandates disclosure of GHG emissions including Scope 3.

#### 7.1.2 Scope 3 Requirements under ESRS E1

**Mandatory disclosures (when climate change is material):**
- E1-6: Gross GHG emissions by Scope 1, 2, and 3
- All material Scope 3 categories must be disclosed
- Category 1 is almost always material for companies with significant
  procurement
- Must disclose: total emissions, methodology, data sources, significant
  changes

**Reporting timeline:**
- Wave 1 (FY2025, reported 2026): Large listed companies >500 employees
- Wave 2 (FY2026, reported 2027): Other large companies meeting 2 of 3
  criteria (>250 employees, >EUR 50M revenue, >EUR 25M total assets)
- Wave 3 (FY2028, reported 2029): Listed SMEs (option to defer to 2028)

**2025 Omnibus Package Changes:**
- Mandatory datapoints reduced by 61%
- Scope 3 remains required when material (no exemption)
- Topic disclosures depend on materiality assessment
- Value chain data collection burden reduced but not eliminated

#### 7.1.3 ESRS E1 Data Points for Category 1

| Data Point | Requirement Level | Description |
|-----------|------------------|-------------|
| E1-6 para 44(a) | Mandatory (if material) | Gross Scope 3 GHG emissions by category |
| E1-6 para 44(b) | Mandatory (if material) | Percentage of Scope 3 from measured/estimated data |
| E1-6 para 44(c) | Mandatory (if material) | Description of methodologies and assumptions |
| E1-6 para 46 | Recommended | Scope 3 intensity metrics |
| E1-6 para 48 | Recommended | Engagement with value chain on GHG data |

### 7.2 California SB 253 (Climate Corporate Data Accountability Act)

#### 7.2.1 Overview

SB 253, signed into law on October 7, 2023, requires annual GHG emissions
disclosure for US entities with revenues exceeding $1 billion that do business
in California. CARB (California Air Resources Board) is responsible for
implementing regulations.

#### 7.2.2 Key Requirements

| Requirement | Detail |
|------------|--------|
| Scope | US entities with >$1B annual revenue doing business in CA |
| Scopes covered | Scope 1, 2, and 3 (all 15 categories) |
| Standard | GHG Protocol Corporate Standard and Scope 3 Standard |
| Scope 1+2 reporting start | FY2025 (first reports due 2026) |
| Scope 3 reporting start | FY2026 (first reports due 2027) |
| Assurance (Scope 1+2) | Limited assurance starting 2027, reasonable by 2030 |
| Assurance (Scope 3) | Limited assurance starting 2030 |
| Administrator | California Air Resources Board (CARB) |
| Penalties | Up to $500,000 per entity per year |
| Scope 3 safe harbor | 2027-2030: no penalties for good-faith misstatements |

#### 7.2.3 SB 253 Implementation Timeline

| Year | Milestone |
|------|-----------|
| 2024 | CARB begins rulemaking process |
| 2025 | Draft regulations released (December 2025) |
| 2026 | First Scope 1+2 reports due; CARB finalizes initial regulations; public comment period through Feb 2026; board vote Feb 26, 2026 |
| 2027 | First Scope 3 reports due; safe harbor begins for Scope 3 |
| 2030 | Safe harbor for Scope 3 good-faith misstatements expires |
| 2030 | Limited assurance required for Scope 3 |

#### 7.2.4 Implications for Category 1

- Companies must report Category 1 if it is a relevant Scope 3 category
  (it almost always is)
- During 2027-2030, good-faith efforts at calculation are protected from
  penalties, even if the numbers contain errors
- Spend-based methods are expected to be acceptable initially, with
  expectation of improvement over time
- CARB has indicated it will issue further guidance on acceptable methodologies

### 7.3 SEC Climate Disclosure Rule

#### 7.3.1 Current Status (as of February 2026)

The SEC adopted its climate disclosure rule on March 6, 2024. However:
- The rule was voluntarily stayed in April 2024 pending judicial review
- On March 27, 2025, the SEC voted to end its defense of the rules
- On September 12, 2025, the Eighth Circuit held the litigation in abeyance
- The rule remains technically in effect but unenforced
- Scope 3 was excluded from the final rule (only Scope 1 and 2 were included)

#### 7.3.2 Scope 3 Treatment

The SEC's final rule did NOT require Scope 3 disclosure, making it less
extensive than the original proposal. The proposed rule (March 2022) would
have required Scope 3 disclosure if material or if the registrant had set
a Scope 3 target. The final rule dropped this requirement entirely.

**Implications**: Even if the SEC rule takes effect, it does not create a
federal US Scope 3 mandate. However, SB 253 fills this gap for large companies.

### 7.4 CDP (Carbon Disclosure Project)

#### 7.4.1 Scope 3 Category 1 Reporting

CDP is the most widely used voluntary reporting framework for Scope 3:

- **Coverage**: Over 23,000 companies report to CDP annually
- **Category 1 prominence**: Category 1 is the most commonly reported Scope 3
  category across all sectors
- **Relevance question**: CDP asks companies to classify each Scope 3 category
  as "Relevant, calculated," "Relevant, not yet calculated," or "Not relevant"
- **Statistics**: Category 1 is reported as "Relevant, calculated" by 57-79%
  of companies depending on sector (CDP 2023 data)

#### 7.4.2 CDP Supply Chain Program

CDP operates a dedicated supply chain program:
- Over 35,000 suppliers reported through CDP in 2024
- Corporates can request emission data from their suppliers via CDP
- Data includes Scope 1, 2, and 3 emissions, plus allocated emissions
- CDP provides allocation methodologies for companies to allocate their
  emissions to specific customers

#### 7.4.3 CDP Scoring Criteria for Category 1

CDP assesses:
- Whether the company has identified Category 1 as relevant
- Whether the company has calculated Category 1 emissions
- The methodology used (supplier-specific > hybrid > average > spend)
- The percentage of spend or activity covered
- Year-over-year trend and explanation
- Engagement initiatives with suppliers

### 7.5 SBTi (Science Based Targets initiative)

#### 7.5.1 Scope 3 Target Requirements

| Criterion | Requirement |
|-----------|------------|
| Threshold | Scope 3 target required if Scope 3 >= 40% of total S1+S2+S3 |
| Coverage | Target must cover >= 67% of total Scope 3 emissions |
| Target type | Absolute reduction or intensity target or supplier engagement |
| Timeframe | Near-term: 5-10 years from submission date |
| Ambition | Must be consistent with well-below 2C or 1.5C pathway |

#### 7.5.2 SBTi Category 1 Implications

- For most companies, Category 1 is a significant portion of Scope 3
- SBTi requires companies to include Category 1 in their inventory
- Supplier engagement targets are a common approach for Category 1:
  e.g., "67% of suppliers by spend will have SBTs by 2030"
- Absolute emission reduction targets require accurate baseline and
  tracking methodology
- SBTi's FLAG (Forest, Land, and Agriculture) guidance provides specific
  requirements for land-intensive supply chains

#### 7.5.3 SBTi v5.3 Criteria (September 2025)

- Near-term targets: Cover 67% of Scope 3 (up from 67% in v5.0)
- Supplier engagement targets: At least 67% of suppliers by emissions
  or spend set their own SBTs within 5 years
- Category 1 is almost always included in the 67% coverage requirement
- Net-Zero Standard v2 (draft): Will require near-term + long-term targets

### 7.6 GRI Standards

GRI 305 (Emissions) requires disclosure of Scope 3 emissions where they are
significant. GRI aligns with the GHG Protocol methodology. Category 1 is
typically identified as significant for most manufacturing and retail companies.

### 7.7 ISO 14064-1:2018

ISO 14064-1 provides the framework for organizational GHG inventories. It
requires reporting of indirect GHG emissions from "imported energy" (equivalent
to Scope 2) and allows reporting of other indirect emissions (equivalent to
Scope 3). The standard is methodology-neutral but references the GHG Protocol
as a common calculation approach.

### 7.8 Regulatory Comparison Matrix

| Regulation | Scope 3 Required? | Category 1 Required? | Methodology | Assurance | Penalties |
|-----------|-------------------|---------------------|-------------|-----------|-----------|
| CSRD/ESRS E1 | Yes (if material) | Yes (if material) | GHG Protocol | Limited (phase-in) | Member state enforcement |
| SB 253 | Yes (all categories) | Yes | GHG Protocol | Limited by 2030 | Up to $500K/year |
| SEC Rule | No (stayed/dropped) | No | N/A | N/A | N/A |
| CDP | Voluntary | Voluntary | GHG Protocol | Not required | No direct penalty |
| SBTi | If >40% of total | Yes (if material) | GHG Protocol | SBTi validation | Target removal |
| GRI 305 | If significant | If significant | GHG Protocol | External assurance recommended | No direct penalty |
| ISO 14064-1 | Optional | Optional | Flexible | Third-party verification available | No direct penalty |

---

## 8. Category Boundaries and Double-Counting Avoidance

### 8.1 What IS Included in Category 1

Category 1 encompasses all upstream cradle-to-gate emissions from the
production of goods and services purchased in the reporting year:

**Goods:**
- Raw materials (ores, agricultural commodities, timber, petroleum feedstocks)
- Intermediate goods (chemicals, plastics, fabrics, components)
- Finished goods purchased for resale (retail sector)
- Packaging materials (primary, secondary, tertiary packaging)
- Office supplies and consumables
- Maintenance materials and spare parts (for non-capital assets)
- Food and beverages (for cafeterias, vending, etc.)
- Non-capital IT equipment (below capitalization threshold)

**Services:**
- Professional services (legal, accounting, consulting, engineering)
- IT services (cloud, SaaS, managed services, data hosting)
- Outsourced operations (cleaning, security, catering, landscaping)
- Marketing and advertising services
- Telecommunications services
- Insurance and financial services
- Training and education services
- Temporary staffing and recruitment
- Research and development (outsourced)
- Printing, design, and creative services
- Logistics management services (but NOT the transport itself -- see below)

### 8.2 What is NOT Included in Category 1

The GHG Protocol explicitly excludes certain items from Category 1 to avoid
double counting with other Scope 3 categories:

| Exclusion | Belongs To | Rationale |
|-----------|-----------|-----------|
| Capital goods (above capitalization threshold) | Category 2 | Reported separately due to different amortization treatment |
| Fuel and energy purchases (for own operations) | Category 3 | Upstream fuel/energy activities reported in Cat 3 |
| Upstream transportation (paid by reporting company) | Category 4 | Transport of purchased goods where company pays freight |
| Downstream transportation (paid by reporting company) | Category 9 | Distribution of sold products |
| Business travel | Category 6 | Reported as separate category |
| Employee commuting | Category 7 | Reported as separate category |
| Waste generated in operations | Category 5 | Waste from own operations |
| Upstream leased assets | Category 8 | Operating leased assets |
| Investments | Category 15 | Financial portfolio emissions |

### 8.3 Capital Goods Boundary (Category 1 vs. Category 2)

The distinction between Category 1 and Category 2 is based on the reporting
company's capitalization policy:

```
If asset_cost > capitalization_threshold AND useful_life > 1 year:
    --> Category 2 (Capital Goods)
Else:
    --> Category 1 (Purchased Goods and Services)
```

**Common capitalization thresholds:**
- Small companies: $1,000 - $5,000
- Medium companies: $5,000 - $10,000
- Large companies: $5,000 - $50,000

**Ambiguous items:**
- IT equipment: Laptop ($1,500) may be Cat 1 or Cat 2 depending on policy
- Furniture: Desk ($800) typically Cat 1; office fit-out ($50,000) typically Cat 2
- Vehicles: Fleet vehicles are Cat 2; rental vehicles are Cat 1 or Cat 6
- Software: SaaS subscriptions are Cat 1; perpetual licenses may be Cat 2

**GHG Protocol guidance**: The reporting company should apply its own
financial capitalization policy consistently and disclose the policy used.

### 8.4 Fuel and Energy Boundary (Category 1 vs. Category 3)

Purchases of fuel and energy for the reporting company's own operations are
excluded from Category 1 and reported under Category 3 (Fuel- and
energy-related activities not included in Scope 1 or Scope 2).

**Category 3 includes:**
- Upstream emissions of purchased fuels (extraction, refining, transport)
- Upstream emissions of purchased electricity (fuel extraction, generation)
- Transmission and distribution (T&D) losses
- Generation of purchased steam, heat, or cooling (upstream)

**Rule**: If the purchase is for direct consumption by the reporting company's
operations, it belongs in Category 3, not Category 1.

### 8.5 Transportation Boundary (Category 1 vs. Category 4/9)

The boundary depends on who pays for the transportation:

| Scenario | Category | Rationale |
|----------|---------|-----------|
| Transport included in product price (FOB/CIF) | Category 1 | Embedded in cradle-to-gate EF |
| Transport paid separately by reporting company (inbound) | Category 4 | Upstream T&D |
| Transport arranged and paid by supplier | Category 1 | Part of supplier's operations |
| Transport of products to customers (outbound) | Category 9 | Downstream T&D |

**Key principle**: If using EEIO emission factors (spend-based method), the
factors already include an average transport component embedded in the supply
chain. Do NOT add separate transport emissions to avoid double counting.

If using supplier-specific or average-data methods with gate-to-gate or
cradle-to-gate production factors that exclude transport, then inbound
transport may be added -- but should be reported under Category 4 unless
it is explicitly included in the cradle-to-gate boundary.

### 8.6 Services Boundary

Services present unique boundary challenges:

**What counts in Category 1 (services):**
- The service provider's Scope 1 + 2 + upstream Scope 3 emissions during
  the delivery of the service
- Example: Emissions from a consulting firm's office operations and business
  travel attributable to the engagement

**What does NOT count:**
- The service provider's Scope 3 Cat 11 (use of sold products) -- this would
  be the reporting company's own Scope 1 or 2
- Travel by the reporting company's own employees to receive the service
  (this is Cat 6, Business Travel)

**Practical challenge**: Most service providers do not yet report emissions
at an engagement level. The spend-based method using EEIO factors for service
sectors (NAICS 54xx, 56xx) is the most practical approach.

### 8.7 Double-Counting Prevention Checklist

The agent should implement checks to prevent these common double-counting errors:

| Check | Description | Action |
|-------|------------|--------|
| Cap/OpEx split | Ensure capital goods are not in Cat 1 | Flag items above capitalization threshold |
| Fuel/energy exclusion | Exclude fuel/energy for own use | Filter out utility and fuel POs |
| Transport separation | Avoid double-counting transport | Check if EF includes transport |
| Scope 1/2 overlap | Ensure no Scope 1/2 items in Cat 1 | Filter direct energy purchases |
| Cat 6/7 overlap | Travel and commuting separate | Filter travel-related spend |
| Intra-company transactions | Eliminate intercompany spend | Filter by vendor type |
| Returns and credits | Net out returns | Apply credit memos |
| Tax and duties | Decide whether to include | Configure per policy |

---

## 9. Materiality and Prioritization

### 9.1 Hot-Spot Analysis

The Pareto principle (80/20 rule) typically applies to procurement emissions:

- Approximately 80% of Category 1 emissions come from 20% of procurement
  categories
- Often, the top 5-10 procurement categories represent 60-80% of emissions
- Long-tail categories (hundreds or thousands) contribute the remaining 20-40%

**Approach:**
1. Calculate spend-based emissions for ALL procurement categories
2. Rank categories by emission contribution (descending)
3. Identify the top categories that cumulatively represent 80% of emissions
4. Prioritize these categories for data quality improvement

### 9.2 Materiality Matrix

The GHG Protocol recommends a materiality matrix combining spend volume and
emission factor intensity:

```
                    High EF Intensity
                    |
        Quadrant 2: |  Quadrant 1:
        MONITOR     |  PRIORITIZE
        (Low spend, |  (High spend,
         high EF)   |   high EF)
                    |
   Low Spend -------+------- High Spend
                    |
        Quadrant 4: |  Quadrant 3:
        LOW PRIORITY|  IMPROVE DATA
        (Low spend, |  (High spend,
         low EF)    |   low EF)
                    |
                    Low EF Intensity
```

**Quadrant definitions:**

| Quadrant | Spend | EF Intensity | Strategy |
|----------|-------|-------------|----------|
| Q1: Prioritize | High (>5% of total) | High (>0.5 kgCO2e/$) | Supplier-specific data, engagement programs |
| Q2: Monitor | Low (<5%) | High (>0.5 kgCO2e/$) | Average-data method, track growth |
| Q3: Improve Data | High (>5%) | Low (<0.5 kgCO2e/$) | Average-data or supplier data, refine EFs |
| Q4: Low Priority | Low (<5%) | Low (<0.5 kgCO2e/$) | Spend-based method acceptable |

### 9.3 Coverage Thresholds

Best practices for coverage:

| Level | Coverage Target | Description |
|-------|---------------|-------------|
| Minimum viable | >= 80% of procurement spend | Required for credible reporting |
| Good practice | >= 90% of procurement spend | Recommended by CDP/SBTi |
| Best practice | >= 95% of procurement spend | Leading practice, minimal gaps |
| Complete | 100% of procurement spend | Aspirational, may include estimates |

**Note**: Coverage can be measured by:
- Percentage of total procurement spend included
- Percentage of total line items included
- Percentage of total suppliers included
- The most common metric is spend coverage

### 9.4 Prioritization Framework

**Step 1: Screening** (Week 1-2)
- Gather total procurement spend by category
- Apply spend-based EEIO factors to all categories
- Identify top 20 categories by emissions

**Step 2: Hot-spot analysis** (Week 2-4)
- Rank categories by: (1) absolute emissions, (2) EF intensity, (3) data
  availability, (4) reduction potential
- Assign priority tier (Q1-Q4)

**Step 3: Data improvement plan** (Month 2-6)
- Q1 categories: Engage top 20 suppliers for primary data
- Q2 categories: Research average-data emission factors
- Q3 categories: Refine EEIO category mapping
- Q4 categories: Maintain spend-based approach

**Step 4: Progressive improvement** (Year 2+)
- Increase supplier-specific coverage by 10-20% annually
- Upgrade average-data factors to more specific sources
- Implement supplier engagement programs aligned with SBTi

### 9.5 Spend Threshold for Different Methods

| Annual Spend per Category | Recommended Method | Rationale |
|--------------------------|-------------------|-----------|
| > $10M | Supplier-specific | Material impact justifies effort |
| $1M - $10M | Average-data or hybrid | Good balance of effort and accuracy |
| $100K - $1M | Average-data or spend-based | Moderate effort appropriate |
| < $100K | Spend-based | Low materiality, spend-based sufficient |

---

## 10. Industry Benchmarks

### 10.1 Category 1 as Percentage of Total Scope 3

Category 1 significance varies dramatically by sector:

| Industry Sector | Cat 1 as % of Total Scope 3 | Cat 1 as % of Total S1+S2+S3 | Typical Cat 1 Absolute Range |
|----------------|---------------------------|-------------------------------|---------------------------|
| Retail (apparel) | 60-80% | 55-75% | 500K - 50M tCO2e |
| Retail (grocery) | 50-70% | 45-65% | 1M - 100M tCO2e |
| Automotive manufacturing | 40-60% | 35-55% | 10M - 200M tCO2e |
| Electronics manufacturing | 35-55% | 30-50% | 5M - 100M tCO2e |
| Chemicals | 30-50% | 25-45% | 5M - 50M tCO2e |
| Construction | 40-60% | 35-55% | 1M - 50M tCO2e |
| Pharmaceuticals | 25-45% | 20-40% | 1M - 20M tCO2e |
| Banking/Financial services | 15-30% | 10-25% | 100K - 5M tCO2e |
| Technology/Software | 20-40% | 15-35% | 100K - 10M tCO2e |
| Telecommunications | 20-35% | 15-30% | 500K - 10M tCO2e |
| Food and beverage manufacturing | 50-70% | 45-65% | 5M - 100M tCO2e |
| Mining and metals | 20-40% | 15-30% | 1M - 20M tCO2e |
| Oil and gas | 5-15% | 3-10% | 1M - 50M tCO2e |
| Airlines | 10-20% | 5-15% | 1M - 20M tCO2e |
| Healthcare | 30-50% | 25-40% | 500K - 10M tCO2e |
| Agricultural commodities | 60-80% | 55-70% | 1M - 50M tCO2e |

### 10.2 CDP Sector-Specific Data

Based on CDP 2023 reporting data:

| Sector | % Reporting Cat 1 as "Relevant, Calculated" | Cat 1 % of Total S3 | Cat 1 % of S1+S2+S3 |
|--------|---------------------------------------------|---------------------|---------------------|
| Agricultural Commodities | 79% | 69% | 63% |
| Food, Beverage, Tobacco | 74% | 55% | 49% |
| Capital Goods | 57% | 5.7% | 5.6% |
| Automobiles & Components | 65% | 45% | 40% |
| Household & Personal | 68% | 52% | 47% |
| Materials (Chemicals) | 62% | 35% | 28% |
| Technology Hardware | 70% | 48% | 42% |
| Software & Services | 58% | 28% | 22% |
| Banks | 45% | 12% | 8% |
| Insurance | 42% | 15% | 10% |
| Healthcare Equipment | 55% | 38% | 32% |
| Retailing | 72% | 62% | 58% |

### 10.3 Emission Intensity Benchmarks

**Emission intensity per unit of revenue (tCO2e per $M revenue):**

| Sector | Cat 1 Intensity (tCO2e/$M) | Range |
|--------|---------------------------|-------|
| Retail (apparel) | 200-600 | Wide variation by product mix |
| Retail (grocery) | 300-800 | Driven by food supply chain |
| Automotive | 150-400 | Materials-intensive |
| Electronics | 50-200 | High-value, lower mass |
| Chemicals | 200-500 | Energy-intensive supply chain |
| Construction | 150-400 | Materials-dominated |
| Pharmaceuticals | 30-100 | High-value, R&D-heavy |
| Financial services | 5-30 | Service-dominated |
| Technology/SaaS | 10-50 | IT infrastructure + services |
| Food manufacturing | 300-1000 | Agricultural supply chain |
| Mining | 100-300 | Equipment and chemicals |

### 10.4 Year-over-Year Trend Benchmarks

Companies actively managing Category 1 emissions typically achieve:

| Strategy | Annual Reduction Rate | Timeline |
|----------|---------------------|----------|
| Supplier engagement (SBT-aligned) | 2-5% per year | 5-10 years |
| Material substitution (recycled content) | 5-15% one-time | 1-3 years |
| Procurement shifting (low-carbon suppliers) | 3-8% per year | 3-5 years |
| Energy efficiency in supply chain | 1-3% per year | Ongoing |
| Renewable energy adoption by suppliers | 2-10% per year | 5-10 years |
| Data quality improvement (method upgrade) | Appears as 10-30% change | 1-2 years |

**Important note**: When companies transition from spend-based to average-data
or supplier-specific methods, reported emissions often change by 20-50%. This
is a methodological change, not a real emission change, and must be disclosed
as such. Re-baselining may be required.

### 10.5 Regional Variation in Emission Factors

The same product can have very different emission factors depending on the
country of production:

| Product | China | India | EU Average | US | Japan |
|---------|-------|-------|-----------|-----|-------|
| Steel (BOF, kgCO2e/kg) | 2.50 | 2.80 | 1.85 | 1.95 | 2.10 |
| Aluminum (primary, kgCO2e/kg) | 20.0 | 22.5 | 8.5 | 12.0 | 10.5 |
| Cement (kgCO2e/kg) | 0.73 | 0.82 | 0.63 | 0.65 | 0.55 |
| Electricity grid (kgCO2e/kWh) | 0.58 | 0.71 | 0.26 | 0.39 | 0.47 |
| Paper (virgin, kgCO2e/kg) | 1.30 | 1.50 | 0.85 | 0.95 | 0.90 |

These variations mean that supply chain geography is a critical input for
accurate Category 1 calculations. A company switching steel sourcing from
China to Europe could reduce Category 1 emissions by 25-30% for that material,
purely based on the production country's energy mix and process efficiency.

---

## 11. Suggested 7-Engine Architecture

Based on this research, the AGENT-MRV-014 should implement seven engines
following the established GreenLang MRV agent pattern:

### 11.1 Engine Overview

| Engine | Name | Responsibility |
|--------|------|---------------|
| 1 | ProcurementDatabaseEngine | Manage procurement records, supplier data, classification codes |
| 2 | SpendBasedCalculatorEngine | Calculate emissions using EEIO factors (EPA USEEIO, EXIOBASE) |
| 3 | AverageDataCalculatorEngine | Calculate emissions using physical EFs (ecoinvent, DEFRA, ICE) |
| 4 | SupplierSpecificCalculatorEngine | Calculate emissions using supplier-provided cradle-to-gate data |
| 5 | HybridAggregatorEngine | Combine results from engines 2-4 into unified inventory |
| 6 | DataQualityScorerEngine | Assess and score data quality per GHG Protocol DQI framework |
| 7 | PurchasedGoodsPipelineEngine | Orchestrate end-to-end calculation, reporting, and compliance |

### 11.2 Engine 1: ProcurementDatabaseEngine

**Responsibilities:**
- Ingest procurement data from ERP systems, purchase orders, AP records
- Classify spend by NAICS, NACE, ISIC, and/or UNSPSC codes
- Manage supplier profiles (location, industry, SBTi status, CDP score)
- Cross-map between classification systems
- Store and retrieve historical procurement data
- Interface with AGENT-DATA-009 (Spend Data Categorizer)

**Key methods:**
- `ingest_procurement_data(records)` -- Import spend records
- `classify_spend(record)` -- Assign industry classification codes
- `get_supplier_profile(supplier_id)` -- Retrieve supplier metadata
- `cross_map_classification(code, from_system, to_system)` -- Map between systems
- `get_procurement_summary(period, groupby)` -- Aggregate spend data

### 11.3 Engine 2: SpendBasedCalculatorEngine

**Responsibilities:**
- Apply EEIO emission factors to spend data
- Support multiple EEIO databases (EPA USEEIO, EXIOBASE, WIOD, GTAP)
- Handle currency conversion and inflation adjustment
- Handle margin adjustments (producer vs. purchaser prices)
- Calculate emissions with uncertainty ranges

**Key methods:**
- `calculate_spend_based(spend_records, eeio_database, base_currency)` -- Core calculation
- `adjust_currency(amount, from_currency, to_currency, year)` -- FX conversion
- `adjust_inflation(amount, from_year, to_year, currency)` -- Deflation
- `get_eeio_factor(classification_code, database, region)` -- Factor lookup
- `calculate_with_uncertainty(spend, factor, dqi_scores)` -- Result with bounds

### 11.4 Engine 3: AverageDataCalculatorEngine

**Responsibilities:**
- Apply physical emission factors to quantity-based procurement data
- Support multiple EF databases (ecoinvent, DEFRA, ICE, industry sources)
- Handle unit conversions (kg, tonnes, m3, liters, pieces)
- Apply transport adders when cradle-to-gate EFs exclude transport
- Implement EF hierarchy (product-specific > material > industry average)

**Key methods:**
- `calculate_average_data(quantity_records, ef_database)` -- Core calculation
- `select_emission_factor(product, region, technology)` -- Hierarchical EF selection
- `convert_units(value, from_unit, to_unit)` -- Unit conversion
- `add_transport_emissions(quantity, distance, mode)` -- Transport component
- `calculate_with_uncertainty(quantity, factor, dqi_scores)` -- Result with bounds

### 11.5 Engine 4: SupplierSpecificCalculatorEngine

**Responsibilities:**
- Process supplier-provided cradle-to-gate emission data
- Validate supplier data completeness and boundary
- Apply allocation methods (revenue, mass, economic, physical)
- Cross-reference with EPD databases and CDP Supply Chain data
- Score supplier data reliability

**Key methods:**
- `calculate_supplier_specific(supplier_data, allocation_method)` -- Core calculation
- `validate_supplier_data(data, required_boundary)` -- Completeness check
- `allocate_emissions(total_emissions, allocation_basis, allocation_data)` -- Allocation
- `score_supplier_reliability(data_source, verification_status)` -- Reliability score
- `aggregate_supplier_results(results)` -- Sum across suppliers

### 11.6 Engine 5: HybridAggregatorEngine

**Responsibilities:**
- Combine results from spend-based, average-data, and supplier-specific engines
- Ensure no double-counting between methods
- Track method coverage (% of spend by each method)
- Produce unified Category 1 total with weighted DQI
- Support year-over-year comparisons and trend analysis

**Key methods:**
- `aggregate_methods(spend_results, avg_results, supplier_results)` -- Combine
- `check_double_counting(all_results)` -- Overlap detection
- `calculate_coverage(results, total_spend)` -- Method coverage %
- `calculate_weighted_dqi(results, emissions_weights)` -- Weighted quality score
- `compare_year_over_year(current, prior, decompose=True)` -- Trend analysis

### 11.7 Engine 6: DataQualityScorerEngine

**Responsibilities:**
- Score data quality across all five DQI dimensions
- Calculate composite DQI scores
- Estimate uncertainty ranges using pedigree matrix
- Generate data quality improvement recommendations
- Track DQI trends over time

**Key methods:**
- `score_temporal(data_year, reporting_year)` -- Temporal DQI
- `score_geographical(data_region, activity_region)` -- Geographical DQI
- `score_technological(data_tech, activity_tech)` -- Technological DQI
- `score_completeness(boundary_coverage)` -- Completeness DQI
- `score_reliability(source_type, verification)` -- Reliability DQI
- `calculate_composite_dqi(scores)` -- Weighted composite
- `estimate_uncertainty(dqi_scores, method)` -- Uncertainty range
- `recommend_improvements(dqi_scores, materiality)` -- Improvement plan

### 11.8 Engine 7: PurchasedGoodsPipelineEngine

**Responsibilities:**
- Orchestrate the full Category 1 calculation pipeline
- Run all engines in correct sequence with error handling
- Apply category boundary rules (exclude Cat 2, Cat 3, Cat 4, etc.)
- Generate compliance-ready outputs (CDP, CSRD, SBTi, GRI formats)
- Produce audit trail with full provenance

**Key methods:**
- `run_pipeline(config)` -- Full end-to-end calculation
- `apply_boundary_rules(records)` -- Filter out non-Cat-1 items
- `generate_report(results, framework)` -- Format for CDP/CSRD/GRI
- `generate_audit_trail(results)` -- Full provenance documentation
- `validate_completeness(results, total_spend)` -- Coverage check

---

## 12. Suggested Enums, Models, and Constants

### 12.1 Enums

```python
class CalculationMethod(str, Enum):
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"

class EEIODatabase(str, Enum):
    EPA_USEEIO_V12 = "epa_useeio_v1.2"
    EPA_USEEIO_V13 = "epa_useeio_v1.3"
    EXIOBASE_V38 = "exiobase_v3.8"
    WIOD_2016 = "wiod_2016"
    GTAP_11 = "gtap_11"

class PhysicalEFDatabase(str, Enum):
    ECOINVENT_V311 = "ecoinvent_v3.11"
    GABI = "gabi"
    DEFRA_2025 = "defra_2025"
    ICE_V3 = "ice_v3.0"
    WORLD_STEEL_2023 = "world_steel_2023"
    IAI_2023 = "iai_2023"
    PLASTICS_EUROPE_2022 = "plastics_europe_2022"

class ClassificationSystem(str, Enum):
    NAICS_2022 = "naics_2022"
    NACE_REV2 = "nace_rev2"
    NACE_REV21 = "nace_rev2.1"
    ISIC_REV4 = "isic_rev4"
    UNSPSC_V28 = "unspsc_v28"

class AllocationMethod(str, Enum):
    REVENUE_BASED = "revenue_based"
    MASS_BASED = "mass_based"
    ECONOMIC = "economic"
    PHYSICAL = "physical"
    ENERGY_BASED = "energy_based"

class DQILevel(str, Enum):
    VERY_GOOD = "very_good"      # 1.0-1.5
    GOOD = "good"                # 1.6-2.5
    FAIR = "fair"                # 2.6-3.5
    POOR = "poor"                # 3.6-4.5
    VERY_POOR = "very_poor"      # 4.6-5.0

class SupplierDataSource(str, Enum):
    EPD = "epd"                  # Environmental Product Declaration
    PCF = "pcf"                  # Product Carbon Footprint (ISO 14067)
    CDP_SUPPLY_CHAIN = "cdp_supply_chain"
    ECOVADIS = "ecovadis"
    DIRECT_DISCLOSURE = "direct_disclosure"
    SUSTAINABILITY_REPORT = "sustainability_report"
    PACT_NETWORK = "pact_network"

class ComplianceFramework(str, Enum):
    GHG_PROTOCOL = "ghg_protocol"
    CSRD_ESRS_E1 = "csrd_esrs_e1"
    SB_253 = "sb_253"
    CDP = "cdp"
    SBTI = "sbti"
    GRI_305 = "gri_305"
    ISO_14064 = "iso_14064"

class ProcurementType(str, Enum):
    RAW_MATERIAL = "raw_material"
    COMPONENT = "component"
    FINISHED_GOOD = "finished_good"
    PACKAGING = "packaging"
    CONSUMABLE = "consumable"
    PROFESSIONAL_SERVICE = "professional_service"
    IT_SERVICE = "it_service"
    OUTSOURCED_OPERATION = "outsourced_operation"
    MAINTENANCE = "maintenance"
    MARKETING = "marketing"
    INSURANCE = "insurance"
    TELECOM = "telecom"
    TRAINING = "training"
    OTHER_GOOD = "other_good"
    OTHER_SERVICE = "other_service"

class PriceType(str, Enum):
    BASIC_PRICE = "basic_price"        # Producer/factory gate price
    PURCHASER_PRICE = "purchaser_price" # Including margins, tax, transport
```

### 12.2 Key Constants

```python
# Database table prefix
GL_PGS_PREFIX = "gl_pgs_"

# Default DQI scores by method
DEFAULT_DQI_BY_METHOD = {
    CalculationMethod.SUPPLIER_SPECIFIC: {
        "temporal": 1.5, "geographical": 1.5,
        "technological": 1.5, "completeness": 2.0, "reliability": 1.5
    },
    CalculationMethod.AVERAGE_DATA: {
        "temporal": 2.5, "geographical": 2.5,
        "technological": 2.5, "completeness": 2.5, "reliability": 2.5
    },
    CalculationMethod.SPEND_BASED: {
        "temporal": 3.5, "geographical": 3.5,
        "technological": 4.0, "completeness": 3.0, "reliability": 3.5
    },
}

# Uncertainty ranges by method (as fraction, e.g., 0.15 = +/-15%)
UNCERTAINTY_BY_METHOD = {
    CalculationMethod.SUPPLIER_SPECIFIC: {"low": 0.10, "high": 0.30},
    CalculationMethod.HYBRID: {"low": 0.20, "high": 0.50},
    CalculationMethod.AVERAGE_DATA: {"low": 0.30, "high": 0.60},
    CalculationMethod.SPEND_BASED: {"low": 0.50, "high": 1.00},
}

# Coverage thresholds
COVERAGE_MINIMUM = 0.80   # 80% of spend
COVERAGE_GOOD = 0.90      # 90% of spend
COVERAGE_BEST = 0.95      # 95% of spend

# Materiality thresholds
MATERIALITY_HIGH_SPEND = 0.05     # >5% of total spend
MATERIALITY_HIGH_EF = 0.50        # >0.50 kgCO2e per USD
MATERIALITY_SUPPLIER_THRESHOLD = 10_000_000  # $10M for supplier-specific

# Supported currencies
SUPPORTED_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CNY", "INR", "KRW",
    "BRL", "CAD", "AUD", "CHF", "SEK", "NOK", "DKK",
    "SGD", "HKD", "TWD", "THB", "MXN", "ZAR"
]

# EEIO base currencies
EEIO_BASE_CURRENCIES = {
    EEIODatabase.EPA_USEEIO_V12: "USD",
    EEIODatabase.EPA_USEEIO_V13: "USD",
    EEIODatabase.EXIOBASE_V38: "EUR",
    EEIODatabase.WIOD_2016: "USD",
    EEIODatabase.GTAP_11: "USD",
}

# EEIO base years (for inflation adjustment)
EEIO_BASE_YEARS = {
    EEIODatabase.EPA_USEEIO_V12: 2021,
    EEIODatabase.EPA_USEEIO_V13: 2021,
    EEIODatabase.EXIOBASE_V38: 2022,
    EEIODatabase.WIOD_2016: 2014,
    EEIODatabase.GTAP_11: 2017,
}
```

### 12.3 Key Models

```python
class ProcurementRecord(BaseModel):
    record_id: str
    tenant_id: str
    reporting_year: int
    supplier_id: Optional[str]
    supplier_name: str
    classification_code: str
    classification_system: ClassificationSystem
    description: str
    procurement_type: ProcurementType
    spend_amount: Decimal
    spend_currency: str
    quantity: Optional[Decimal]
    quantity_unit: Optional[str]
    country_of_origin: Optional[str]
    is_capital_good: bool = False

class EmissionResult(BaseModel):
    record_id: str
    calculation_method: CalculationMethod
    emissions_kgco2e: Decimal
    emissions_co2_kg: Optional[Decimal]
    emissions_ch4_kg: Optional[Decimal]
    emissions_n2o_kg: Optional[Decimal]
    emission_factor_value: Decimal
    emission_factor_unit: str
    emission_factor_source: str
    dqi_composite: float
    uncertainty_lower: Decimal
    uncertainty_upper: Decimal
    provenance_hash: str

class CategoryOneInventory(BaseModel):
    tenant_id: str
    reporting_year: int
    total_emissions_tco2e: Decimal
    method_breakdown: Dict[CalculationMethod, Decimal]
    coverage_by_method: Dict[CalculationMethod, float]
    total_spend_covered: Decimal
    total_spend_universe: Decimal
    spend_coverage_pct: float
    weighted_dqi: float
    top_categories: List[Dict]
    top_suppliers: List[Dict]
    year_over_year_change_pct: Optional[float]
    compliance_flags: Dict[ComplianceFramework, bool]
```

---

## 13. Integration with Existing GreenLang Agents

### 13.1 Upstream Dependencies

| Agent | Integration Point | Data Flow |
|-------|------------------|-----------|
| AGENT-DATA-009 (Spend Categorizer) | Spend classification, NAICS/UNSPSC mapping, EEIO factors | Spend records with classification codes |
| AGENT-DATA-003 (ERP Connector) | Procurement data extraction from SAP, Oracle, etc. | Raw purchase orders, AP data |
| AGENT-DATA-008 (Supplier Questionnaire) | Supplier emission data from CDP, EcoVadis, etc. | Supplier-specific emission factors |
| AGENT-DATA-002 (Excel/CSV Normalizer) | Standardize procurement spreadsheets | Normalized procurement records |
| AGENT-DATA-001 (PDF Extractor) | Extract data from supplier EPDs and invoices | Emission factor data from PDFs |
| AGENT-DATA-010 (Data Quality Profiler) | Quality scoring of input data | Data quality assessments |
| AGENT-FOUND-003 (Unit Normalizer) | Unit conversion for physical quantities | Normalized quantities |
| AGENT-FOUND-005 (Citations & Evidence) | Citation management for EF sources | Auditable citations |

### 13.2 Downstream Dependencies

| Agent | Integration Point | Data Flow |
|-------|------------------|-----------|
| AGENT-FOUND-001 (Orchestrator) | DAG execution for Category 1 pipeline | Pipeline execution control |
| AGENT-FOUND-008 (Reproducibility) | Artifact hashing, drift detection | Provenance verification |
| AGENT-FOUND-009 (QA Test Harness) | Golden file testing for calculations | Test validation |
| AGENT-FOUND-010 (Observability) | Metrics, traces, SLO tracking | Operational monitoring |
| MRV Scope 3 Category Mapper | Overall Scope 3 inventory assembly | Category 1 contribution to total |

### 13.3 Database Integration

Following the GreenLang pattern, AGENT-MRV-014 should use:
- PostgreSQL + TimescaleDB for time-series procurement and emission data
- Table prefix: `gl_pgs_` (purchased goods and services)
- Hypertables for: procurement_records, emission_results, dqi_scores
- Continuous aggregates for: monthly/quarterly/annual summaries
- pgvector for: supplier similarity matching (optional)

### 13.4 API Integration

Following the GreenLang pattern, expose REST endpoints:
- `POST /api/v1/mrv/purchased-goods/calculate` -- Run calculation
- `GET /api/v1/mrv/purchased-goods/results/{period}` -- Retrieve results
- `GET /api/v1/mrv/purchased-goods/summary` -- Dashboard summary
- `POST /api/v1/mrv/purchased-goods/supplier-data` -- Upload supplier data
- `GET /api/v1/mrv/purchased-goods/dqi/{period}` -- Data quality report
- `GET /api/v1/mrv/purchased-goods/report/{framework}` -- Compliance report

---

## 14. References

### 14.1 Primary Standards

1. GHG Protocol. "Corporate Value Chain (Scope 3) Accounting and Reporting
   Standard." World Resources Institute and WBCSD, 2011.
   https://ghgprotocol.org/sites/default/files/standards/Corporate-Value-Chain-Accounting-Reporing-Standard_041613_2.pdf

2. GHG Protocol. "Technical Guidance for Calculating Scope 3 Emissions."
   Chapter 1: Category 1 -- Purchased Goods and Services, 2013.
   https://ghgprotocol.org/sites/default/files/2022-12/Chapter1.pdf

3. GHG Protocol. "Scope 3 Calculation Guidance."
   https://ghgprotocol.org/scope-3-calculation-guidance-2

4. GHG Protocol. "Quantitative Uncertainty Guidance."
   https://ghgprotocol.org/sites/default/files/2022-12/Quantitative%20Uncertainty%20Guidance.pdf

5. GHG Protocol. "Scope 3, Discussion Paper A.1 -- Inventory Quality."
   Working Draft 1, October 2024.
   https://ghgprotocol.org/sites/default/files/2024-10/S3-DiscussionPaper-20241024.pdf

### 14.2 EEIO Databases

6. US EPA. "Supply Chain Greenhouse Gas Emission Factors for US Industries
   and Commodities." v1.2, April 2023.
   https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=358530

7. US EPA. "US Environmentally-Extended Input-Output (USEEIO) Models."
   https://www.epa.gov/land-research/us-environmentally-extended-input-output-useeio-models

8. EXIOBASE Consortium. "EXIOBASE 3." https://www.exiobase.eu/

9. Timmer, M.P., et al. "An Illustrated User Guide to the World Input-Output
   Database: the Case of Global Automotive Production." WIOD, 2015.

10. Aguiar, A., et al. "The GTAP Data Base: Version 11." GTAP, 2022.

### 14.3 Physical Emission Factor Sources

11. ecoinvent Association. "ecoinvent Database v3.11." 2024.
    https://ecoinvent.org/database/

12. DESNZ/DEFRA. "UK Government GHG Conversion Factors for Company Reporting."
    2025. https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting

13. Hammond, G.P. and Jones, C.I. "Inventory of Carbon and Energy (ICE)."
    University of Bath. https://circularecology.com/embodied-carbon-footprint-database.html

14. World Steel Association. "Steel Statistical Yearbook 2023."
    https://worldsteel.org/

15. International Aluminium Institute. "Life Cycle Inventory Data." 2023.
    https://international-aluminium.org/

16. PlasticsEurope. "Eco-profiles and Environmental Declarations." 2022.
    https://plasticseurope.org/

### 14.4 Regulatory Sources

17. European Commission. "Corporate Sustainability Reporting Directive (CSRD)."
    Directive (EU) 2022/2464.
    https://finance.ec.europa.eu/capital-markets-union-and-financial-markets/company-reporting-and-auditing/company-reporting/corporate-sustainability-reporting_en

18. EFRAG. "European Sustainability Reporting Standards: ESRS E1 Climate Change."
    2023.

19. California State Legislature. "SB 253: Climate Corporate Data
    Accountability Act." 2023.
    https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB253

20. US Securities and Exchange Commission. "The Enhancement and Standardization
    of Climate-Related Disclosures for Investors." March 2024.
    https://www.sec.gov/rules-regulations/2024/03/s7-10-22

21. Science Based Targets initiative. "SBTi Corporate Near-Term Criteria."
    Version 5.3, September 2025.
    https://files.sciencebasedtargets.org/production/files/SBTi-criteria.pdf

22. CDP. "Scope 3 Upstream Report: Big Challenges, Simple Remedies."
    June 2024.
    https://cdn.cdp.net/cdp-production/cms/reports/documents/000/007/834/original/Scope-3-Upstream-Report.pdf

23. CDP. "Technical Note: Relevance of Scope 3 Categories by Sector."
    https://cdn.cdp.net/cdp-production/cms/guidance_docs/pdfs/000/003/504/original/CDP-technical-note-scope-3-relevance-by-sector.pdf

### 14.5 Classification Systems

24. US Census Bureau. "North American Industry Classification System (NAICS)
    2022." https://www.census.gov/naics/

25. Eurostat. "NACE Rev. 2.1 -- Statistical Classification of Economic
    Activities." 2023.

26. United Nations. "International Standard Industrial Classification of All
    Economic Activities (ISIC) Rev. 4.1." 2024.

27. GS1 US / UNDP. "United Nations Standard Products and Services Code
    (UNSPSC) v28.0." 2024. https://www.unspsc.org/

### 14.6 Additional Technical References

28. Climatiq. "The Science Behind Spend-Based Emission Factors."
    https://www.climatiq.io/blog/science-behind-spend-based-emission-factors

29. Climatiq. "Guide: How to Calculate Emissions Using Spend-Based Emission
    Factors from EXIOBASE."
    https://www.climatiq.io/blog/guide-how-to-calculate-spend-based-emissions

30. Watershed. "Guidance on the Use of EEIO Models for Spend-Based Carbon
    Accounting." https://watershed.com/blog/eeio-guidance

31. isdata-org. "what-links-to-what: Mapping Interlinkages Between Industrial
    and Product Classification Systems."
    https://github.com/isdata-org/what-links-to-what

32. GHG Protocol. "Scope 3 Standard Development Plan." December 2024.
    https://ghgprotocol.org/sites/default/files/2025-01/S3-SDP-20241220.pdf

---

## Appendix A: GreenLang Agent Naming Convention

Following the established naming pattern:
- Agent ID: AGENT-MRV-014
- Internal Code: GL-MRV-S3-001 (first Scope 3 calculation agent)
- Database prefix: `gl_pgs_`
- API prefix: `/api/v1/mrv/purchased-goods/`
- Metric prefix: `gl_pgs_`
- Log prefix: `[GL-PGS]`

## Appendix B: Migration Script Reference

Following the established pattern (V058 as next available):
- V058: AGENT-MRV-014 Purchased Goods and Services schema
- Expected tables: ~12-15 tables + 3 hypertables + 2 continuous aggregates
- Key tables:
  - `gl_pgs_procurement_records` (hypertable)
  - `gl_pgs_emission_results` (hypertable)
  - `gl_pgs_supplier_profiles`
  - `gl_pgs_emission_factors`
  - `gl_pgs_classification_mappings`
  - `gl_pgs_dqi_scores` (hypertable)
  - `gl_pgs_allocation_records`
  - `gl_pgs_coverage_tracking`
  - `gl_pgs_audit_trail`
  - `gl_pgs_compliance_reports`
  - `gl_pgs_eeio_factors`
  - `gl_pgs_physical_factors`

## Appendix C: Test Strategy

Following the GreenLang QA pattern:
- Target: 1,000+ tests
- Unit tests per engine: ~120-150 each (7 engines x ~140 = ~980)
- Integration tests: ~50
- Golden file tests: ~20 (known-good calculation scenarios)
- Property-based tests: ~30 (invariant checking)
- Benchmark tests: ~10 (performance regression)

Key test scenarios:
1. Spend-based calculation with currency conversion and inflation
2. Average-data calculation with multiple physical EFs
3. Supplier-specific calculation with various allocation methods
4. Hybrid aggregation with no double counting
5. DQI scoring across all five dimensions
6. Boundary rule enforcement (Cat 1 vs. Cat 2/3/4/6)
7. Year-over-year change decomposition
8. Multi-framework report generation
9. Edge cases: zero spend, negative credits, unknown classifications
10. Regression: Known calculation results from GHG Protocol examples

---

*Document version: 1.0.0*
*Last updated: 2026-02-24*
*Author: GL-RegulatoryIntelligence*
*Status: Research Complete -- Ready for PRD Development*
