# PRD: AGENT-MRV-015 -- Scope 3 Category 2 Capital Goods Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-002 |
| **Internal Label** | AGENT-MRV-015 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/capital_goods/` |
| **DB Migration** | V066 |
| **Metrics Prefix** | `gl_cg_` |
| **Table Prefix** | `cg_` |
| **API** | `/api/v1/capital-goods` |
| **Env Prefix** | `GL_CG_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The Capital Goods Agent implements **GHG Protocol Scope 3 Category 2**,
which covers all upstream (cradle-to-gate) emissions from the production of
capital goods purchased or acquired by the reporting company in the
reporting year. Capital goods are defined as final products that have an
extended useful life and are treated as fixed assets or property, plant, and
equipment (PP&E) under applicable accounting standards (IFRS/US GAAP).

The critical accounting rule for Category 2 is that **100% of cradle-to-gate
emissions are reported in the year of acquisition** -- there is no
depreciation or amortization of emissions over the asset's useful life.
This differs fundamentally from financial accounting treatment and means
Category 2 emissions exhibit significant year-over-year fluctuations driven
by capital expenditure cycles (e.g., new factory builds, fleet renewals,
IT infrastructure refreshes).

The agent provides four calculation methods -- supplier-specific,
average-data, spend-based, and hybrid -- with capital-asset-specific
classification taxonomies, ICE Database integration for construction
materials, capitalization threshold enforcement, and double-counting
prevention against both Category 1 (purchased goods) AND Scope 1/2
(use-phase operational emissions).

### Justification for Dedicated Agent

1. **Distinct accounting treatment** -- 100% of emissions in acquisition year
   requires asset-aware tracking entirely different from Category 1 expensed goods
2. **Asset classification complexity** -- 8 asset categories with useful life
   ranges, capitalization thresholds, and PP&E classification rules that vary by
   accounting standard (IFRS, US GAAP, local GAAP)
3. **ICE Database integration** -- Construction and building materials require
   specialized Inventory of Carbon and Energy (ICE) factors distinct from the
   product-level LCA databases used in Category 1
4. **CapEx volatility context** -- Year-over-year fluctuations of 50-500% are
   normal due to infrequent large purchases; requires contextual reporting to
   prevent misinterpretation
5. **Double-counting against multiple categories** -- Must prevent overlap with
   Category 1 (non-capital purchases), Category 8 (leased assets), and Scope 1/2
   (use-phase emissions from the capital goods in operation)
6. **Regulatory urgency** -- CSRD (FY2025+), SB 253 (FY2027), SBTi, CDP all
   require or strongly encourage Category 2 reporting as a separate line item
7. **Materiality for capital-intensive sectors** -- Category 2 represents 10-40%
   of total Scope 3 for construction, manufacturing, utilities, and
   telecommunications sectors

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011) -- Chapter 5
- GHG Protocol Scope 3 Technical Guidance (2013) -- Chapter 2: Category 2
- GHG Protocol Scope 3 Calculation Guidance (online)
- GHG Protocol Quantitative Uncertainty Guidance
- CSRD/ESRS E1 -- Scope 3 disclosure (E1-6 para 44)
- California SB 253 -- Mandatory Scope 3 by FY2027 for entities >$1B revenue
- CDP Climate Change Questionnaire -- C6.5 Scope 3 Category 2
- SBTi Corporate Manual v5.3 -- Scope 3 target required if >40% of total
- GRI 305 -- Emissions (Scope 3 disclosure)
- ISO 14064-1:2018 -- Category 4 indirect GHG emissions
- ISO 14025 -- Environmental Product Declarations (EPDs)
- ISO 14067 -- Product Carbon Footprints (PCFs)
- IAS 16 -- Property, Plant and Equipment (IFRS)
- ASC 360 -- Property, Plant and Equipment (US GAAP)
- ICE Database v3.0 -- Inventory of Carbon and Energy (Bath/Circular Ecology)

---

## 2. Methodology

### 2.1 Capital Goods vs Purchased Goods (Category 1 vs Category 2)

The distinction between Category 1 and Category 2 is determined by the
company's financial accounting treatment:

| Criterion | Category 1 (Purchased Goods) | Category 2 (Capital Goods) |
|-----------|------------------------------|---------------------------|
| Accounting treatment | Expensed in period | Capitalized as PP&E/fixed asset |
| Useful life | Consumed/used within 1 year | Multi-year useful life (>1 year) |
| Financial threshold | Below capitalization threshold | At or above capitalization threshold |
| Emission timing | Year of purchase | 100% in year of acquisition (no depreciation) |
| Example items | Raw materials, consumables, services | Buildings, machinery, vehicles, IT servers |

**Key rule: The company's own capitalization policy determines the boundary.**

| Organization Size | Typical Capitalization Threshold | Common Policy |
|------------------|--------------------------------|---------------|
| Small (<$10M revenue) | $1,000 - $2,500 | Conservative |
| Medium ($10M - $500M) | $2,500 - $5,000 | Moderate |
| Large ($500M - $5B) | $5,000 - $10,000 | Standard |
| Enterprise (>$5B) | $10,000 - $25,000+ | Materiality-based |

### 2.2 Four Calculation Methods

The GHG Protocol Technical Guidance defines four methods for Category 2,
listed from most to least accurate:

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | Supplier-Specific | Supplier cradle-to-gate data for capital assets | Highest (+/-10-30%) | Lowest (key assets) |
| 2 | Hybrid | Mix of supplier + secondary data | High (+/-20-50%) | Medium |
| 3 | Average-Data | Physical quantities + capital-asset EFs | Medium (+/-30-60%) | Medium-High |
| 4 | Spend-Based | Capital expenditure + EEIO factors | Lowest (+/-50-100%) | Highest (all CapEx) |

**Decision tree for method selection:**

```
Is supplier-specific cradle-to-gate data available for the capital asset?
  YES --> Use Supplier-Specific Method
  NO  --> Is physical quantity/material data available?
            YES --> Are capital-asset-specific EFs available?
                      YES --> Use Average-Data Method
                      NO  --> Is CapEx data available?
                                YES --> Use Hybrid Method (quantity + spend)
                                NO  --> Estimate and document
            NO  --> Is CapEx data available?
                      YES --> Use Spend-Based Method
                      NO  --> Estimate using industry benchmarks
```

#### 2.2.1 Spend-Based Method (Lowest Accuracy, Broadest Coverage)

The spend-based method estimates emissions by multiplying capital
expenditure amounts by capital-goods-specific EEIO emission factors.
This is the typical starting point for organizations beginning Category 2
reporting.

**Core formula:**

```
Emissions_cat2_spend = SUM over all capital asset categories c:
    CapEx_c (in base currency) * EEIO_factor_c (kgCO2e per base currency unit)
```

**With currency conversion and inflation adjustment:**

```
Emissions_cat2_spend = SUM over all capital asset categories c:
    (CapEx_c_local / FX_rate_to_base) * (CPI_base_year / CPI_spend_year) * EEIO_factor_c
```

**Margin removal (producer price vs. purchaser price):**

```
CapEx_producer = CapEx_purchaser * (1 - margin_rate_c)
```

Where `margin_rate_c` is the average wholesale + retail + transport margin for
capital goods category c (typically 10-30% for equipment, 5-15% for
construction, 15-35% for vehicles).

**EEIO sources (capital-goods-specific sectors):**
- EPA USEEIO v1.2/v1.3 -- Capital goods sectors from 1,016 commodities by
  NAICS-6 (construction, machinery, computer/electronic, motor vehicle, etc.)
- EXIOBASE 3.8 -- 163 product groups x 49 regions (~9,800 factors) filtered
  for capital asset categories
- WIOD 2016 -- 43 countries, 56 sectors (older, less granular)
- GTAP 11 -- 141 regions, 65 sectors (subscription required)

**Capital-specific EEIO sectors (EPA USEEIO top 30):**

| NAICS Code | Description | EF (kgCO2e/USD 2021) |
|-----------|-------------|---------------------|
| 236210 | Industrial building construction | 0.38 |
| 236220 | Commercial building construction | 0.35 |
| 237110 | Water/sewer line construction | 0.42 |
| 237310 | Highway/street construction | 0.45 |
| 333111 | Farm machinery manufacturing | 0.31 |
| 333120 | Construction machinery manufacturing | 0.33 |
| 333241 | Food product machinery manufacturing | 0.28 |
| 333249 | Other industrial machinery manufacturing | 0.29 |
| 333310 | Commercial/service industry machinery | 0.25 |
| 333414 | HVAC equipment manufacturing | 0.27 |
| 333511 | Industrial mold manufacturing | 0.30 |
| 333517 | Machine tool manufacturing | 0.32 |
| 333611 | Turbine manufacturing | 0.26 |
| 333911 | Pump manufacturing | 0.28 |
| 333921 | Elevator/escalator manufacturing | 0.24 |
| 334111 | Electronic computer manufacturing | 0.18 |
| 334112 | Computer storage device manufacturing | 0.16 |
| 334210 | Telephone apparatus manufacturing | 0.15 |
| 334413 | Semiconductor manufacturing | 0.22 |
| 335311 | Power transformer manufacturing | 0.30 |
| 335911 | Storage battery manufacturing | 0.35 |
| 336111 | Automobile manufacturing | 0.28 |
| 336112 | Light truck manufacturing | 0.30 |
| 336120 | Heavy duty truck manufacturing | 0.34 |
| 336411 | Aircraft manufacturing | 0.21 |
| 336510 | Railroad rolling stock manufacturing | 0.32 |
| 336611 | Ship building | 0.36 |
| 337211 | Office furniture manufacturing | 0.19 |
| 337214 | Office furniture (non-wood) manufacturing | 0.17 |
| 811310 | Machinery/equipment repair | 0.12 |

#### 2.2.2 Average-Data Method (Medium Accuracy)

Uses physical quantities (mass, area, units) multiplied by industry-average
cradle-to-gate emission factors from LCA databases and the ICE Database.
Particularly effective for construction/building assets where material
quantities are well documented.

**Core formula:**

```
Emissions_cat2_avg = SUM over all capital assets a:
    Quantity_a (physical units) * EF_a (kgCO2e per physical unit)
```

**For multi-material assets (e.g., buildings):**

```
Emissions_cat2_avg = SUM over all assets a, materials m:
    (Quantity_a_m * EF_m) + Transport_emissions_a
```

**With transportation adder (when EF excludes transport to site):**

```
Emissions_cat2_avg = SUM over all assets a:
    (Quantity_a * EF_production_a) + (Mass_a * Distance_a * EF_transport_mode)
```

**EF sources for capital goods:** ecoinvent v3.11, ICE v3.0 (Inventory of
Carbon and Energy -- 200+ construction materials), DEFRA/DESNZ, World Steel
Association, International Aluminium Institute, GaBi/Sphera.

**Representative physical emission factors for capital asset materials (35+ materials):**

| Material | EF (kgCO2e/kg) | Source | Typical Asset Category |
|----------|----------------|--------|----------------------|
| Primary steel (BOF) | 2.33 | World Steel 2023 | Machinery, Buildings |
| Secondary steel (EAF) | 0.67 | World Steel 2023 | Machinery, Buildings |
| General steel (world avg) | 1.37 | ICE v3.0 | All metal structures |
| Primary aluminum (global avg) | 16.70 | IAI 2023 | Equipment, Vehicles |
| Secondary aluminum | 0.50 | IAI 2023 | Equipment, Vehicles |
| Cement (Portland CEM I) | 0.91 | ICE v3.0 | Buildings |
| Concrete (ready-mix, 30MPa) | 0.13 | GCCA 2023 | Buildings |
| Concrete (high strength, 50MPa) | 0.17 | ICE v3.0 | Buildings |
| Bricks (general) | 0.24 | ICE v3.0 | Buildings |
| Float glass | 1.22 | ICE v3.0 | Buildings |
| Sawn timber (softwood) | 0.31 | ICE v3.0 | Buildings |
| Sawn timber (hardwood) | 0.42 | ICE v3.0 | Buildings |
| Timber (glulam) | 0.51 | ICE v3.0 | Buildings |
| Copper (primary) | 4.10 | ICA 2022 | IT Infra, Equipment |
| Lead | 1.57 | ICE v3.0 | Equipment |
| Zinc | 3.86 | ICE v3.0 | Buildings |
| HDPE | 1.80 | PlasticsEurope 2022 | Equipment |
| PVC | 2.41 | PlasticsEurope 2022 | Buildings |
| Polycarbonate | 5.50 | PlasticsEurope 2022 | IT Infrastructure |
| Epoxy resin | 6.20 | ecoinvent 3.11 | Equipment |
| Stainless steel (304) | 4.30 | World Steel 2023 | Equipment |
| Cast iron | 1.51 | ICE v3.0 | Machinery |
| Reinforcing bar (rebar) | 1.40 | ICE v3.0 | Buildings |
| Structural steel (sections) | 1.53 | ICE v3.0 | Buildings |
| Asphalt | 0.047 | ICE v3.0 | Land Improvements |
| Mineral wool insulation | 1.28 | ICE v3.0 | Buildings |
| EPS insulation | 3.29 | ICE v3.0 | Buildings |
| Plasterboard | 0.39 | ICE v3.0 | Buildings |
| Ceramic tiles | 0.78 | ICE v3.0 | Buildings |
| Lithium-ion battery cells | 73.00 | IEA 2023 | Vehicles, IT Infra |
| Silicon wafer (solar grade) | 50.00 | IEA 2023 | Equipment |
| Printed circuit board | 28.50 | ecoinvent 3.11 | IT Infrastructure |
| Optical fiber cable | 3.40 | ecoinvent 3.11 | IT Infrastructure |
| Electric motor (per kW) | 15.00 | ecoinvent 3.11 | Machinery |
| Photovoltaic panel (per m2) | 250.00 | IEA 2023 | Equipment |

**Asset-level embodied carbon benchmarks:**

| Asset Type | Embodied Carbon | Unit | Source |
|-----------|----------------|------|--------|
| Office building (steel frame) | 500-800 | kgCO2e/m2 | RICS 2023 |
| Office building (timber frame) | 300-500 | kgCO2e/m2 | RICS 2023 |
| Warehouse/industrial | 300-600 | kgCO2e/m2 | RICS 2023 |
| Data center | 800-1200 | kgCO2e/m2 | Whitehead 2023 |
| CNC machining center | 15,000-25,000 | kgCO2e/unit | ecoinvent |
| Forklift (electric) | 5,000-8,000 | kgCO2e/unit | ecoinvent |
| Heavy truck (diesel) | 15,000-25,000 | kgCO2e/unit | ecoinvent |
| Server rack (populated) | 2,000-4,000 | kgCO2e/unit | Dell/HP ESG |
| Desktop computer | 300-500 | kgCO2e/unit | Apple/Dell ESG |
| Laptop | 200-400 | kgCO2e/unit | Apple/Dell ESG |
| Network switch (enterprise) | 100-300 | kgCO2e/unit | Cisco ESG |
| Solar PV system (per kWp) | 1,000-1,500 | kgCO2e/kWp | IEA 2023 |
| Wind turbine (onshore, per MW) | 350,000-500,000 | kgCO2e/MW | Vestas LCA |

#### 2.2.3 Supplier-Specific Method (Highest Accuracy)

Uses primary data from suppliers on cradle-to-gate GHG emissions of their
capital goods products. This is the gold standard and the goal state for
high-value capital acquisitions.

**Core formula (product-level data):**

```
Emissions_cat2_ss = SUM over all suppliers s, assets a:
    Quantity_s_a * EF_supplier_s_a
```

**With allocation (facility-level data):**

```
Emissions_cat2_ss = SUM over all suppliers s:
    Emissions_total_s * Allocation_factor_s
```

**Allocation methods:**

| Method | Formula | Use Case |
|--------|---------|----------|
| Economic | `Allocation = CapEx_customer / Revenue_total_s` | Mixed-product manufacturers |
| Mass-based | `Allocation = Mass_customer / Mass_total_s` | Heavy equipment, bulk materials |
| Physical | `Allocation = Units_customer / Units_total_s` | Standard product lines |
| Energy-based | `Allocation = Energy_customer / Energy_total_s` | Energy-intensive production |
| Hybrid | Weighted combination of above methods | Complex multi-product facilities |

**Supplier data sources:**
- Environmental Product Declarations (EPDs) -- ISO 14025 (common for
  construction products and building materials)
- Product Carbon Footprints (PCFs) -- ISO 14067
- CDP Supply Chain Program -- 35,000+ suppliers (many large equipment OEMs)
- EcoVadis ratings -- 130,000+ companies
- Direct manufacturer sustainability reports (common for vehicles, IT equipment)
- Industry-specific programs (e.g., Steel EPD, Concrete EPD via EPDItaly,
  IBU, EPD International)

#### 2.2.4 Hybrid Method (Recommended)

Combines all three methods, using the most accurate data available for each
capital asset or asset category:

```
Emissions_cat2_hybrid = Emissions_SS + Emissions_AD + Emissions_SB
```

**Coverage strategy for capital goods:**

| Tier | Assets | Target Coverage | Method |
|------|--------|----------------|--------|
| Tier 1 | Top 5-10 highest CapEx assets | 50-80% of total CapEx | Supplier-specific (EPD/PCF) |
| Tier 2 | Next 10-20 assets | 15-30% of total CapEx | Average-data (ICE/ecoinvent) |
| Tier 3 | Remaining small CapEx items | 5-15% of total CapEx | Spend-based (EEIO) |

**Note on capital goods specifics:** Because capital purchases are typically
fewer in number but higher in value than Category 1 purchases, achieving
high supplier-specific coverage (Tier 1 > 60%) is often more feasible for
Category 2 than Category 1.

**Weighted DQI for hybrid:**

```
DQI_hybrid = (CapEx_SS * DQI_SS + CapEx_AD * DQI_AD + CapEx_SB * DQI_SB) / CapEx_total
```

### 2.3 Asset Classification Taxonomy

The agent classifies capital goods into 8 primary asset categories with
detailed subcategories:

| Category | Subcategories | Typical Useful Life | Example Items |
|----------|--------------|--------------------|--------------  |
| Buildings | Office, warehouse, manufacturing, retail, data center, laboratory | 20-50 years | New HQ, production facility |
| Machinery | Production, process, material handling, packaging, testing | 7-20 years | CNC machines, presses, conveyors |
| Equipment | Laboratory, medical, telecommunications, power generation, HVAC | 5-15 years | Generators, transformers, lab instruments |
| Vehicles | Passenger, light commercial, heavy commercial, specialized, electric | 3-10 years | Fleet trucks, forklifts, company cars |
| IT Infrastructure | Servers, storage, networking, end-user devices, peripherals | 3-7 years | Server racks, switches, PCs, monitors |
| Furniture | Office, laboratory, warehouse, retail fixtures | 5-15 years | Desks, chairs, shelving, display cases |
| Land Improvements | Paving, landscaping, fencing, drainage, lighting, signage | 10-30 years | Parking lots, site works, outdoor lighting |
| Leasehold Improvements | Interior buildout, HVAC mods, electrical upgrades, fixtures | Lease term (3-15 years) | Office buildouts, tenant improvements |

### 2.4 Spend Classification Systems

The agent supports four industry classification systems with cross-mapping,
filtered for capital-goods-relevant sectors:

| System | Structure | Granularity | Coverage | Primary Use |
|--------|-----------|-------------|----------|-------------|
| NAICS 2022 | 6-digit, 20 sectors | 1,057 industries | US, Canada, Mexico | EPA USEEIO mapping |
| NACE Rev 2.1 | Letter + 4-digit, 21 sections | 615 activities | EU member states | EXIOBASE mapping, CSRD |
| ISIC Rev 4.1 | Letter + 4-digit | ~450 classes | UN global standard | Bridge between NAICS/NACE |
| UNSPSC v28.0 | 8-digit, 55 segments | 60,000+ commodities | Global procurement | Product-level classification |

**Cross-mapping:**

| From | To | Method | Source |
|------|----|--------|--------|
| NAICS --> ISIC | Correspondence table | UN Statistics Division |
| NACE --> ISIC | Correspondence table | Eurostat |
| ISIC --> NAICS | Correspondence table | US Census Bureau |
| ISIC --> NACE | Correspondence table | Eurostat |
| UNSPSC --> NAICS | Fuzzy mapping + custom | isdata-org/what-links-to-what |
| UNSPSC --> NACE | Via ISIC intermediate | Custom chain mapping |

### 2.5 Data Quality Indicator (DQI)

Per GHG Protocol Scope 3 Standard Chapter 7, five data quality indicators
are assessed on a 1-5 scale:

| Indicator | Score 1 (Very Good) | Score 3 (Fair) | Score 5 (Very Poor) |
|-----------|--------------------|-----------------|--------------------|
| Temporal | Data from reporting year | Data within 6 years | Data older than 10 years |
| Geographical | Same country/region | Same continent | Global average |
| Technological | Identical technology | Related technology | Unrelated category |
| Completeness | All sources included | 50-80% covered | Less than 20% covered |
| Reliability | Third-party verified | Established database | Estimate or assumption |

**Composite DQI:**

```
DQI_composite = (DQI_temporal + DQI_geographical + DQI_technological
                + DQI_completeness + DQI_reliability) / 5
```

**Quality classification:**

| DQI Range | Classification | Recommended Action |
|-----------|---------------|-------------------|
| 1.0 - 1.5 | Very Good | Maintain current data quality |
| 1.6 - 2.5 | Good | Monitor for improvements |
| 2.6 - 3.5 | Fair | Prioritize improvement plan |
| 3.6 - 4.5 | Poor | Active improvement required |
| 4.6 - 5.0 | Very Poor | Urgent data quality intervention |

### 2.6 Uncertainty Ranges

| Method | Typical DQI Range | Uncertainty Range | Confidence Level |
|--------|------------------|------------------|-----------------|
| Supplier-Specific (verified EPD) | 1.0 - 1.5 | +/- 10-15% | Very High |
| Supplier-Specific (unverified) | 1.5 - 2.5 | +/- 15-30% | High |
| Average-Data (process LCA) | 2.0 - 3.0 | +/- 30-50% | Medium-High |
| Average-Data (industry avg/ICE) | 2.5 - 3.5 | +/- 40-60% | Medium |
| Spend-Based (regional EEIO) | 3.0 - 4.0 | +/- 50-80% | Low-Medium |
| Spend-Based (national EEIO) | 3.5 - 4.5 | +/- 60-100% | Low |
| Hybrid | 2.0 - 3.5 | +/- 20-50% | Medium-High |

**Pedigree matrix uncertainty factors:**

| DQI Score | Uncertainty Factor (sigma) |
|-----------|--------------------------|
| 1 | 1.00 (no additional uncertainty) |
| 2 | 1.05 (+/- 5% additional) |
| 3 | 1.10 (+/- 10% additional) |
| 4 | 1.20 (+/- 20% additional) |
| 5 | 1.50 (+/- 50% additional) |

**Combined uncertainty:**

```
Sigma_combined = sqrt(sigma_base^2 + sigma_temporal^2 + sigma_geo^2
                    + sigma_tech^2 + sigma_completeness^2 + sigma_reliability^2)
```

### 2.7 Materiality & Hot-Spot Analysis

**Capital goods Pareto analysis:** Because capital purchases are fewer in
number but larger in value than Category 1, the top 5-10 capital assets
often represent 70-90% of total Category 2 emissions.

**Materiality matrix:**

| Quadrant | CapEx | EF Intensity | Strategy |
|----------|-------|-------------|----------|
| Q1: Prioritize | High (>10% of total CapEx) | High (>0.3 kgCO2e/$) | Supplier-specific data, EPDs |
| Q2: Monitor | Low (<10%) | High (>0.3 kgCO2e/$) | Average-data, track acquisitions |
| Q3: Improve Data | High (>10%) | Low (<0.3 kgCO2e/$) | Average-data, refine EFs |
| Q4: Low Priority | Low (<10%) | Low (<0.3 kgCO2e/$) | Spend-based acceptable |

**Coverage thresholds:**

| Level | Target | Description |
|-------|--------|-------------|
| Minimum viable | >= 80% of CapEx | Required for credible reporting |
| Good practice | >= 90% of CapEx | Recommended by CDP/SBTi |
| Best practice | >= 95% of CapEx | Leading practice |

**CapEx threshold recommendations:**

| CapEx per Asset/Category | Recommended Method |
|--------------------------|-------------------|
| > $5M | Supplier-specific (EPD/PCF) |
| $500K - $5M | Average-data or hybrid |
| $50K - $500K | Average-data or spend-based |
| < $50K | Spend-based |

### 2.8 Key Formulas

**Currency conversion:**

```
CapEx_base = CapEx_local * (1 / FX_rate_to_base)
```

**Inflation deflation (to EEIO base year):**

```
CapEx_real = CapEx_nominal * (CPI_base_year / CPI_current_year)
```

**PPP adjustment (optional, for cross-country EEIO application):**

```
CapEx_PPP = CapEx_local * (PPP_factor_local / PPP_factor_base)
```

**Coverage tracking:**

```
Coverage_SS = CapEx_SS / CapEx_total
Coverage_AD = CapEx_AD / CapEx_total
Coverage_SB = CapEx_SB / CapEx_total
Coverage_total = Coverage_SS + Coverage_AD + Coverage_SB  (should = 1.0)
```

**Emissions intensity:**

```
Intensity_revenue = Emissions_cat2 / Revenue_total       (tCO2e per $M)
Intensity_employee = Emissions_cat2 / FTE_count           (tCO2e per FTE)
Intensity_capex = Emissions_cat2 / Total_CapEx            (tCO2e per $M CapEx)
Intensity_asset = Emissions_cat2 / Asset_count            (tCO2e per asset)
```

**Year-over-year change decomposition (capital-goods-specific):**

```
Delta_emissions = Delta_activity + Delta_EF + Delta_method + Delta_scope

Where:
  Delta_activity = (CapEx_current - CapEx_prior) * EF_prior
  Delta_EF       = CapEx_current * (EF_current - EF_prior)
  Delta_method   = Emissions_current_new_method - Emissions_current_old_method
  Delta_scope    = Change from boundary or scope changes
```

**CapEx volatility context (critical for Category 2 reporting):**

```
CapEx_rolling_avg = SUM(CapEx_year[i] for i in range(-2, 1)) / 3
Volatility_ratio = CapEx_current / CapEx_rolling_avg
Emission_normalized = Emissions_cat2_current / Volatility_ratio
```

If `Volatility_ratio > 2.0`, the agent flags the result with a contextual
note explaining that the increase is driven by CapEx activity rather than
emission intensity changes.

**Weighted DQI for total inventory:**

```
DQI_weighted = SUM over all line items i:
    (Emissions_i / Emissions_total) * DQI_i
```

### 2.9 Category Boundaries & Double-Counting Prevention

**Included in Category 2 (capital goods):** Any asset that meets ALL of:
(a) above the company's capitalization threshold, (b) has a useful life
greater than 1 year, AND (c) is treated as PP&E or fixed asset on the
balance sheet.

**Capital goods by asset category:**
- Buildings: New construction, major renovations (if capitalized)
- Machinery: Production equipment, manufacturing tools, industrial systems
- Equipment: Laboratory, telecom, HVAC, power generation, medical
- Vehicles: Fleet vehicles, forklifts, heavy machinery, company cars
- IT Infrastructure: Servers, storage arrays, networking equipment,
  end-user devices (if above threshold)
- Furniture: Office furnishings, fixtures (if above threshold)
- Land Improvements: Paving, fencing, drainage, lighting
- Leasehold Improvements: Tenant buildouts, interior modifications

**Excluded (to prevent double counting):**

| Exclusion | Belongs To | Enforcement Rule |
|-----------|-----------|-----------------|
| Items below capitalization threshold | Category 1 | Check cost < threshold OR useful_life <= 1yr |
| Operational emissions from capital assets | Scope 1/2 | Only cradle-to-gate; exclude use-phase energy, fuel |
| Maintenance and repair (non-capital) | Category 1 | Check if expensed vs capitalized |
| Operating leases (lessee) | Category 8 | Check lease classification (IFRS 16 / ASC 842) |
| Finance leases (lessor) | Category 13 | Check lease classification for lessor reporting |
| Land (no depreciation) | Excluded | Land itself has no cradle-to-gate emissions |
| Fuel/energy for own operations | Category 3 | Filter utility and fuel purchase orders |
| Business travel vehicles (rentals) | Category 6 | Filter travel-related spend codes |
| Intercompany transfers | None | Filter by vendor type (intercompany) |
| Donated or government-granted assets | Excluded | Zero-cost acquisition; no CapEx basis |
| Fully depreciated asset re-acquisitions | Depends | If re-measured/revalued, treat as new acquisition |

### 2.10 Industry Benchmarks

**Category 2 as percentage of total Scope 3 by sector:**

| Industry Sector | Cat 2 as % of Total S3 | Cat 2 as % of S1+S2+S3 |
|----------------|----------------------|----------------------|
| Construction | 25-40% | 20-35% |
| Utilities (power generation) | 20-35% | 15-30% |
| Telecommunications | 15-30% | 10-25% |
| Mining & metals | 15-25% | 10-20% |
| Manufacturing (heavy) | 10-25% | 8-20% |
| Oil & gas | 10-20% | 5-15% |
| Transportation & logistics | 10-20% | 8-15% |
| Healthcare | 5-15% | 3-10% |
| Retail | 3-10% | 2-8% |
| Technology/Software | 3-10% | 2-7% |
| Financial services | 2-8% | 1-5% |
| Food & beverage | 2-8% | 1-5% |
| Pharmaceuticals | 5-15% | 3-10% |

**Emission intensity benchmarks (tCO2e per $M CapEx):**

| Asset Category | Emission Intensity | Range |
|---------------|-------------------|-------|
| Buildings (new construction) | 200-500 | Material-intensive |
| Machinery (heavy industrial) | 250-450 | Steel/metal-intensive |
| Equipment (general industrial) | 150-350 | Mixed materials |
| Vehicles (commercial fleet) | 200-400 | Manufacturing-intensive |
| IT Infrastructure (servers) | 100-250 | Electronics-intensive |
| Furniture & fixtures | 100-200 | Lower intensity |
| Land improvements | 250-450 | Concrete/asphalt-heavy |
| Leasehold improvements | 150-350 | Interior fitout materials |

### 2.11 Emission Factor Selection Hierarchy

The agent implements an 8-level EF priority hierarchy for capital goods:

| Priority | Source | DQI Score |
|----------|--------|-----------|
| 1 | Supplier-specific EPD or PCF (ISO 14025/14067, verified) | 1.0-1.5 |
| 2 | Supplier-specific CDP/direct disclosure (unverified) | 1.5-2.5 |
| 3 | Product-specific LCA data (ecoinvent, GaBi) | 2.0-3.0 |
| 4 | Material-specific average EF (ICE, World Steel, DEFRA) | 2.5-3.5 |
| 5 | Asset-level embodied carbon benchmark (RICS, industry) | 3.0-3.5 |
| 6 | Regional EEIO factor (EXIOBASE for non-US) | 3.0-4.0 |
| 7 | National EEIO factor (EPA USEEIO for US) | 3.5-4.5 |
| 8 | Global average EEIO factor (fallback) | 4.0-5.0 |

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
+---------------------------------------------------------+
|                   AGENT-MRV-015                          |
|            Capital Goods Agent                           |
|                                                          |
|  +----------------------------------------------------+ |
|  | Engine 1: CapitalAssetDatabaseEngine                | |
|  |   - Asset taxonomy (8 categories, 40+ subcategories)| |
|  |   - EEIO emission factors (EPA USEEIO, EXIOBASE)   | |
|  |   - Physical EFs (35+ materials from ICE/ecoinvent) | |
|  |   - NAICS/NACE/ISIC/UNSPSC classification mapping   | |
|  |   - Currency exchange rates and PPP factors         | |
|  |   - Industry margin percentages for capital goods   | |
|  |   - Capitalization thresholds by org size           | |
|  |   - Useful life ranges by asset category            | |
|  |   - Supplier emission factor registry               | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 2: SpendBasedCalculatorEngine                 | |
|  |   - CapEx * EEIO factor calculation                 | |
|  |   - Currency conversion (20+ currencies)            | |
|  |   - Inflation deflation (CPI base year adjustment)  | |
|  |   - Margin removal (producer vs purchaser price)    | |
|  |   - NAICS-to-EEIO mapping (capital sectors)         | |
|  |   - Progressive NAICS matching (6->4->3->2 digit)   | |
|  |   - Batch capital expenditure processing            | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 3: AverageDataCalculatorEngine                | |
|  |   - Quantity * physical EF calculation              | |
|  |   - ICE database embodied carbon (construction)     | |
|  |   - ecoinvent LCA factors (equipment/machinery)     | |
|  |   - Multi-material asset decomposition              | |
|  |   - Unit conversion (kg, tonnes, m2, m3, units)     | |
|  |   - Transport-to-site emission inclusion            | |
|  |   - Waste and loss factor application              | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 4: SupplierSpecificCalculatorEngine           | |
|  |   - EPD/PCF/CDP supplier data integration           | |
|  |   - Product-level emission factors                 | |
|  |   - Facility-level allocation (5 methods)          | |
|  |   - Supplier data quality validation               | |
|  |   - Cradle-to-gate boundary verification           | |
|  |   - Construction product EPD processing            | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 5: HybridAggregatorEngine                    | |
|  |   - Method prioritization and selection            | |
|  |   - Coverage analysis (% CapEx per method)         | |
|  |   - Hot-spot analysis (Pareto 80/20 ranking)       | |
|  |   - Double-counting prevention vs Cat 1 & S1/S2    | |
|  |   - YoY variance context for CapEx fluctuations    | |
|  |   - CapEx rolling average normalization             | |
|  |   - Gap filling (fallback to lower-tier methods)   | |
|  |   - Total Category 2 emission aggregation          | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 6: ComplianceCheckerEngine                   | |
|  |   - 7 frameworks: GHG Protocol, CSRD/ESRS, CDP,   | |
|  |     SBTi, California SB 253, GRI 305, ISO 14064   | |
|  |   - Coverage thresholds (>=80% CapEx)              | |
|  |   - DQI scoring validation                         | |
|  |   - Cat 1 vs Cat 2 boundary verification           | |
|  |   - No-depreciation rule enforcement               | |
|  |   - Methodology documentation checks              | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 7: CapitalGoodsPipelineEngine                 | |
|  |   - 10-stage pipeline orchestration                | |
|  |   - Batch multi-period processing                  | |
|  |   - Multi-facility aggregation                     | |
|  |   - Category boundary enforcement                  | |
|  |   - CapEx volatility flagging                      | |
|  |   - Export (JSON/CSV/Excel/PDF)                    | |
|  |   - Compliance-ready outputs (CDP, CSRD, SBTi)    | |
|  |   - Provenance chain assembly                     | |
|  +----------------------------------------------------+ |
+---------------------------------------------------------+
```

### 3.2 Ten-Stage Pipeline

```
Stage 1: VALIDATE
  - Schema validation of input records
  - Required field checks (CapEx amount, date, vendor)
  - Data type enforcement (Decimal for monetary values)
  - Duplicate detection

Stage 2: CLASSIFY
  - Asset category classification (8 categories)
  - NAICS/NACE/ISIC/UNSPSC code assignment
  - Capitalization threshold enforcement
  - PP&E vs expense determination

Stage 3: RESOLVE_EFS
  - 8-level EF hierarchy resolution
  - Best-available EF selection per asset
  - EF source and version tracking
  - Missing EF flagging with fallback

Stage 4: SPEND_CALC
  - CapEx * EEIO factor calculation
  - Currency conversion
  - Inflation deflation
  - Margin removal

Stage 5: AVERAGE_CALC
  - Physical quantity * EF calculation
  - ICE/ecoinvent factor application
  - Transport-to-site adder
  - Multi-material decomposition

Stage 6: SUPPLIER_CALC
  - Supplier EPD/PCF data application
  - Allocation factor calculation
  - Boundary verification (cradle-to-gate only)
  - Data quality validation

Stage 7: HYBRID
  - Method prioritization
  - Coverage analysis
  - Double-counting prevention
  - CapEx volatility context

Stage 8: COMPLIANCE
  - 7-framework compliance check
  - Coverage threshold validation
  - No-depreciation rule verification
  - Gap identification

Stage 9: AGGREGATE
  - Total Category 2 emissions
  - By asset category
  - By method
  - By facility/entity
  - Intensity metrics

Stage 10: SEAL
  - SHA-256 provenance hash
  - Audit trail assembly
  - Export generation
  - Result persistence
```

### 3.3 File Structure

```
greenlang/capital_goods/
+-- __init__.py                          # Lazy imports, module exports
+-- models.py                            # Pydantic v2 models, enums, constants
+-- config.py                            # GL_CG_ prefixed configuration
+-- metrics.py                           # Prometheus metrics (gl_cg_*)
+-- provenance.py                        # SHA-256 provenance chain
+-- capital_asset_database.py            # Engine 1: Asset taxonomy & EF database
+-- spend_based_calculator.py            # Engine 2: Spend-based calculation
+-- average_data_calculator.py           # Engine 3: Average-data calculation
+-- supplier_specific_calculator.py      # Engine 4: Supplier-specific calculation
+-- hybrid_aggregator.py                 # Engine 5: Hybrid aggregation
+-- compliance_checker.py                # Engine 6: Compliance checking
+-- capital_goods_pipeline.py            # Engine 7: Pipeline orchestration
+-- setup.py                             # Service facade
+-- api/
    +-- __init__.py                      # API package
    +-- router.py                        # FastAPI REST endpoints

tests/unit/mrv/test_capital_goods/
+-- __init__.py
+-- conftest.py
+-- test_models.py
+-- test_config.py
+-- test_metrics.py
+-- test_provenance.py
+-- test_capital_asset_database.py
+-- test_spend_based_calculator.py
+-- test_average_data_calculator.py
+-- test_supplier_specific_calculator.py
+-- test_hybrid_aggregator.py
+-- test_compliance_checker.py
+-- test_capital_goods_pipeline.py
+-- test_setup.py
+-- test_api.py

deployment/database/migrations/sql/
+-- V066__capital_goods_service.sql
```

### 3.4 Database Schema (V066)

16 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `cg_eeio_factors` | EPA USEEIO + EXIOBASE EEIO factors for capital goods sectors | Seed (30+ capital NAICS sectors) |
| `cg_physical_efs` | Physical emission factors for capital asset materials (35+ materials) | Seed (35+ rows) |
| `cg_supplier_efs` | Supplier-specific emission factors (EPD, CDP, PCF) | Dimension |
| `cg_asset_categories` | 8-category asset taxonomy with subcategories | Seed (8 categories, 40+ subcategories) |
| `cg_useful_life_ranges` | Default useful life ranges by asset category | Seed (40+ rows) |
| `cg_capitalization_thresholds` | Capitalization threshold policies by org size | Seed (4 org sizes x 3 GAAP standards) |
| `cg_naics_codes` | NAICS 2022 classification codes (capital-relevant subset) | Seed (200+ capital industries) |
| `cg_classification_mapping` | Cross-mapping between NAICS/NACE/ISIC/UNSPSC for capital sectors | Seed |
| `cg_currency_rates` | Annual exchange rates (20+ currencies) | Dimension |
| `cg_margin_factors` | Wholesale/retail/transport margin by capital goods sector | Dimension |
| `cg_inflation_indices` | CPI/GDP deflator by country and year | Dimension |
| `cg_calculations` | Calculation results (spend/avg/supplier/hybrid) | Hypertable |
| `cg_calculation_details` | Line-item detail per calculation | Regular |
| `cg_asset_records` | Capital asset inventory and classification | Regular |
| `cg_dqi_scores` | Data quality scores per line item | Regular |
| `cg_compliance_checks` | Compliance check results (7 frameworks) | Regular |
| `cg_batch_jobs` | Batch processing jobs | Regular |
| `cg_hourly_stats` | Hourly calculation statistics | Continuous Aggregate |
| `cg_daily_stats` | Daily calculation statistics | Continuous Aggregate |

**Key seed data:**

- `cg_eeio_factors`: 30+ rows from EPA USEEIO v1.2 capital goods sectors (NAICS-6 code,
  factor kgCO2e/USD, database version, base year, margin type)
- `cg_physical_efs`: 35+ rows of material-level emission factors from ICE v3.0, World
  Steel, IAI, GCCA, PlasticsEurope, ecoinvent
- `cg_asset_categories`: 8 primary categories with 40+ subcategories, NAICS mapping,
  useful life ranges, typical materials
- `cg_useful_life_ranges`: Default useful life min/max/typical by asset subcategory
  per IAS 16 / ASC 360 guidance
- `cg_capitalization_thresholds`: Threshold amounts by organization size (small/
  medium/large/enterprise) and GAAP standard (IFRS/US GAAP/local)
- `cg_classification_mapping`: NAICS-ISIC, NACE-ISIC, UNSPSC-NAICS correspondence
  entries filtered for capital goods sectors

**Schema design principles:**

- Row-Level Security (RLS) on all tenant-facing tables via `tenant_id`
- TimescaleDB hypertables on `cg_calculations` (partitioned by `calculated_at`)
- Continuous aggregates for hourly and daily statistics
- Foreign key relationships from `cg_calculation_details` to `cg_calculations`
- GIN indexes on JSONB columns for metadata queries
- B-tree indexes on all classification codes and foreign keys
- Partial indexes for active/current records

### 3.5 API Endpoints (20)

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/calculate` | Run single calculation (any method) |
| 2 | POST | `/calculate/batch` | Batch multi-period calculation |
| 3 | GET | `/calculations` | List calculations with filtering |
| 4 | GET | `/calculations/{calculation_id}` | Get calculation result |
| 5 | DELETE | `/calculations/{calculation_id}` | Delete a calculation |
| 6 | POST | `/assets` | Register or update capital asset record |
| 7 | GET | `/assets` | List capital asset records with filtering |
| 8 | PUT | `/assets/{asset_id}` | Update a capital asset record |
| 9 | GET | `/emission-factors` | List emission factors (EEIO + physical) with filtering |
| 10 | GET | `/emission-factors/{ef_id}` | Get a specific emission factor |
| 11 | POST | `/emission-factors/custom` | Register custom emission factor |
| 12 | POST | `/classify` | Classify asset into category + NAICS/NACE |
| 13 | POST | `/compliance/check` | Run compliance check (all 7 frameworks) |
| 14 | GET | `/compliance/{calculation_id}` | Get compliance results for a calculation |
| 15 | POST | `/uncertainty` | Run uncertainty analysis on a calculation |
| 16 | GET | `/aggregations` | Get aggregated results (by category, method, period) |
| 17 | GET | `/hot-spots` | Get hot-spot analysis (Pareto ranking) |
| 18 | POST | `/export` | Export results (JSON/CSV/Excel/PDF) |
| 19 | GET | `/health` | Health check |
| 20 | GET | `/stats` | Service statistics and metrics |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees

- All emission calculations use Python `Decimal` (8 decimal places)
- No LLM calls in any calculation path -- deterministic lookups only
- Every calculation step recorded in a provenance trace
- SHA-256 provenance hash for every emission result
- Bit-perfect reproducibility: same input always produces same output
- EEIO factor lookup is exact-match by classification code and database version
- Physical EF lookup is exact-match by material, region, and source year
- Capitalization threshold checks use deterministic comparison (Decimal)
- Asset classification uses rule-based taxonomy, not ML/LLM inference

### 4.2 Enumerations (20)

| Enum | Values | Description |
|------|--------|-------------|
| `CalculationMethod` | SPEND_BASED, AVERAGE_DATA, SUPPLIER_SPECIFIC, HYBRID | 4 GHG Protocol methods |
| `AssetCategory` | BUILDINGS, MACHINERY, EQUIPMENT, VEHICLES, IT_INFRASTRUCTURE, FURNITURE, LAND_IMPROVEMENTS, LEASEHOLD_IMPROVEMENTS | 8 asset categories |
| `AssetSubCategory` | OFFICE_BUILDING, WAREHOUSE, MANUFACTURING_FACILITY, DATA_CENTER, LABORATORY, RETAIL_SPACE, CNC_MACHINE, PRESS, CONVEYOR, FORKLIFT, CRANE, LAB_INSTRUMENT, HVAC_SYSTEM, GENERATOR, TRANSFORMER, PASSENGER_VEHICLE, LIGHT_COMMERCIAL, HEAVY_TRUCK, SPECIALIZED_VEHICLE, ELECTRIC_VEHICLE, SERVER, STORAGE_ARRAY, NETWORK_SWITCH, DESKTOP, LAPTOP, MONITOR, OFFICE_DESK, OFFICE_CHAIR, SHELVING, DISPLAY_CASE, PAVING, FENCING, DRAINAGE, EXTERIOR_LIGHTING, SIGNAGE, INTERIOR_BUILDOUT, HVAC_MODIFICATION, ELECTRICAL_UPGRADE, FIXTURE_INSTALLATION | 39 subcategories |
| `SpendClassificationSystem` | NAICS_2022, NACE_REV2, NACE_REV21, ISIC_REV4, UNSPSC_V28 | 5 classification systems |
| `EEIODatabase` | EPA_USEEIO_V12, EPA_USEEIO_V13, EXIOBASE_V38, WIOD_2016, GTAP_11 | 5 EEIO databases |
| `PhysicalEFSource` | ECOINVENT_V311, DEFRA_2025, ICE_V3, WORLD_STEEL_2023, IAI_2023, PLASTICS_EUROPE_2022, GCCA_2023, CUSTOM | 8 physical EF sources |
| `SupplierDataSource` | EPD, PCF, CDP_SUPPLY_CHAIN, ECOVADIS, DIRECT_MEASUREMENT, SUSTAINABILITY_REPORT, ESTIMATED | 7 supplier data sources |
| `AllocationMethod` | ECONOMIC, PHYSICAL, MASS, ENERGY, HYBRID | 5 allocation methods |
| `CurrencyCode` | USD, EUR, GBP, JPY, CNY, CHF, CAD, AUD, KRW, INR, BRL, MXN, SEK, NOK, DKK, SGD, HKD, NZD, ZAR, AED | 20 currencies |
| `DQIDimension` | TEMPORAL, GEOGRAPHICAL, TECHNOLOGICAL, COMPLETENESS, RELIABILITY | 5 DQI dimensions |
| `DQIScore` | VERY_GOOD (1), GOOD (2), FAIR (3), POOR (4), VERY_POOR (5) | 5 quality scores |
| `UncertaintyMethod` | MONTE_CARLO, ANALYTICAL, TIER_DEFAULT | 3 uncertainty methods |
| `ComplianceFramework` | GHG_PROTOCOL, CSRD_ESRS_E1, CDP, SBTI, SB_253, GRI_305, ISO_14064 | 7 regulatory frameworks |
| `ComplianceStatus` | COMPLIANT, PARTIAL, NON_COMPLIANT | 3 compliance statuses |
| `PipelineStage` | VALIDATE, CLASSIFY, RESOLVE_EFS, SPEND_CALC, AVERAGE_CALC, SUPPLIER_CALC, HYBRID, COMPLIANCE, AGGREGATE, SEAL | 10 pipeline stages |
| `ExportFormat` | JSON, CSV, XLSX, PDF | 4 export formats |
| `BatchStatus` | PENDING, RUNNING, COMPLETED, FAILED | 4 batch statuses |
| `GWPSource` | AR4, AR5, AR6, AR6_20YR | 4 GWP assessment report versions |
| `EmissionGas` | CO2, CH4, N2O, CO2E | 4 emission gases |
| `CapitalizationPolicy` | COMPANY_DEFINED, IFRS, US_GAAP, LOCAL_GAAP | 4 capitalization policy standards |

### 4.3 Models (25)

| Model | Description | Key Fields |
|-------|-------------|------------|
| `CapitalAssetRecord` | Capital asset with classification and lifecycle info | asset_id, name, category, subcategory, acquisition_date, capex_amount, useful_life_years, capitalization_policy, naics_code, vendor |
| `CapExSpendRecord` | Capital expenditure line item for spend-based calculation | record_id, asset_id, spend_amount, currency, naics_code, nace_code, reporting_year, vendor_name, description |
| `PhysicalRecord` | Physical quantity data for average-data calculation | record_id, asset_id, material, quantity, unit, ef_source, transport_distance_km, transport_mode |
| `SupplierRecord` | Supplier-provided emission data | record_id, asset_id, supplier_id, supplier_name, data_source, emission_factor, allocation_method, boundary_type |
| `SpendBasedResult` | Result from spend-based calculation | result_id, capex_base, eeio_factor, emissions_co2e, eeio_database, naics_code, margin_applied, deflation_applied |
| `AverageDataResult` | Result from average-data calculation | result_id, quantity, ef_value, ef_source, emissions_co2e, transport_emissions, total_emissions |
| `SupplierSpecificResult` | Result from supplier-specific calculation | result_id, supplier_emission, allocation_factor, allocated_emissions, data_source, boundary_verified |
| `HybridResult` | Combined multi-method result | result_id, ss_emissions, ad_emissions, sb_emissions, total_emissions, coverage_ss, coverage_ad, coverage_sb, dqi_weighted |
| `EEIOFactor` | EEIO emission factor record | factor_id, naics_code, description, ef_kgco2e_per_usd, database, base_year, currency, margin_type |
| `PhysicalEF` | Physical emission factor record | ef_id, material, ef_kgco2e_per_unit, unit, source, source_year, region, asset_category |
| `SupplierEF` | Supplier-specific emission factor | ef_id, supplier_id, product_id, ef_value, unit, data_source, verification_status, valid_from, valid_to |
| `DQIAssessment` | Data quality assessment across 5 dimensions | assessment_id, temporal, geographical, technological, completeness, reliability, composite_score, classification |
| `AssetClassification` | Asset classification result | asset_id, category, subcategory, naics_code, nace_code, unspsc_code, confidence_score |
| `CapitalizationThreshold` | Capitalization threshold configuration | threshold_id, org_size, gaap_standard, min_amount, currency, policy_description |
| `UsefulLifeRange` | Useful life range by asset type | range_id, asset_category, subcategory, min_years, max_years, typical_years, source |
| `DepreciationContext` | Context for CapEx volatility reporting | context_id, reporting_year, capex_current, capex_prior, capex_rolling_avg, volatility_ratio, context_note |
| `MaterialityItem` | Hot-spot analysis item | item_id, category, capex_amount, capex_pct, emissions, emissions_pct, ef_intensity, quadrant, cumulative_pct |
| `CoverageReport` | Method coverage analysis | report_id, coverage_ss, coverage_ad, coverage_sb, coverage_total, coverage_level, gaps |
| `ComplianceRequirement` | Framework-specific compliance requirement | requirement_id, framework, requirement_code, description, is_mandatory, data_points_required |
| `ComplianceCheckResult` | Compliance check result per framework | result_id, framework, status, score, findings, gaps, recommendations |
| `CalculationRequest` | API request to run a calculation | method, records, reporting_year, organization_id, capitalization_threshold, options |
| `BatchRequest` | Batch calculation request | batch_id, requests, reporting_years, options |
| `CalculationResult` | Complete calculation output | calculation_id, method, emissions_co2e, dqi_score, provenance_hash, processing_time_ms, created_at |
| `AggregationResult` | Aggregated emissions by dimension | aggregation_id, dimension, breakdowns, total_emissions, total_capex, intensity |
| `HotSpotAnalysis` | Pareto analysis of emission sources | analysis_id, items, top_n_coverage, top_n_count, total_emissions, total_capex |

### 4.4 Constant Tables (12)

#### 4.4.1 GWP_VALUES

| Gas | AR4 (100yr) | AR5 (100yr) | AR6 (100yr) | AR6 (20yr) |
|-----|-------------|-------------|-------------|------------|
| CO2 | 1 | 1 | 1 | 1 |
| CH4 | 25 | 28 | 27.9 | 82.5 |
| N2O | 298 | 265 | 273 | 273 |

#### 4.4.2 DQI_SCORE_VALUES

| Score | Label | Numeric | Description |
|-------|-------|---------|-------------|
| 1 | Very Good | 1.0 | Best available; third-party verified |
| 2 | Good | 2.0 | Established databases; same region |
| 3 | Fair | 3.0 | Related technology; within 6 years |
| 4 | Poor | 4.0 | Proxy data; different region |
| 5 | Very Poor | 5.0 | Estimate or assumption; >10 years old |

#### 4.4.3 UNCERTAINTY_RANGES

| Method | Lower Bound (%) | Upper Bound (%) | Default Sigma |
|--------|-----------------|-----------------|---------------|
| Supplier-Specific (verified) | 10 | 15 | 0.12 |
| Supplier-Specific (unverified) | 15 | 30 | 0.22 |
| Average-Data (process LCA) | 30 | 50 | 0.40 |
| Average-Data (industry avg) | 40 | 60 | 0.50 |
| Spend-Based (regional EEIO) | 50 | 80 | 0.65 |
| Spend-Based (national EEIO) | 60 | 100 | 0.80 |
| Hybrid | 20 | 50 | 0.35 |

#### 4.4.4 COVERAGE_THRESHOLDS

| Level | Minimum (%) | Label | Description |
|-------|-------------|-------|-------------|
| MINIMUM | 80 | Minimum Viable | Required for credible reporting |
| GOOD | 90 | Good Practice | Recommended by CDP/SBTi |
| BEST | 95 | Best Practice | Leading practice |
| COMPLETE | 100 | Complete | Full coverage |

#### 4.4.5 EF_HIERARCHY_PRIORITY

| Priority | Source Type | DQI Range | Description |
|----------|-----------|-----------|-------------|
| 1 | SUPPLIER_EPD_VERIFIED | 1.0-1.5 | Verified EPD (ISO 14025) |
| 2 | SUPPLIER_CDP_DIRECT | 1.5-2.5 | CDP/direct disclosure |
| 3 | PRODUCT_LCA | 2.0-3.0 | Product-level LCA (ecoinvent, GaBi) |
| 4 | MATERIAL_AVERAGE | 2.5-3.5 | ICE, World Steel, DEFRA |
| 5 | ASSET_BENCHMARK | 3.0-3.5 | RICS, industry benchmarks |
| 6 | REGIONAL_EEIO | 3.0-4.0 | EXIOBASE regional |
| 7 | NATIONAL_EEIO | 3.5-4.5 | EPA USEEIO national |
| 8 | GLOBAL_AVERAGE | 4.0-5.0 | Global average fallback |

#### 4.4.6 CURRENCY_EXCHANGE_RATES

| Currency | Code | Rate to USD (2024 avg) | Region |
|----------|------|----------------------|--------|
| US Dollar | USD | 1.0000 | North America |
| Euro | EUR | 0.9240 | Europe |
| British Pound | GBP | 0.7920 | Europe |
| Japanese Yen | JPY | 151.35 | Asia-Pacific |
| Chinese Yuan | CNY | 7.2450 | Asia-Pacific |
| Swiss Franc | CHF | 0.8840 | Europe |
| Canadian Dollar | CAD | 1.3620 | North America |
| Australian Dollar | AUD | 1.5310 | Asia-Pacific |
| South Korean Won | KRW | 1360.50 | Asia-Pacific |
| Indian Rupee | INR | 83.4200 | Asia-Pacific |
| Brazilian Real | BRL | 4.9750 | South America |
| Mexican Peso | MXN | 17.0800 | North America |
| Swedish Krona | SEK | 10.4900 | Europe |
| Norwegian Krone | NOK | 10.5600 | Europe |
| Danish Krone | DKK | 6.8940 | Europe |
| Singapore Dollar | SGD | 1.3420 | Asia-Pacific |
| Hong Kong Dollar | HKD | 7.8160 | Asia-Pacific |
| New Zealand Dollar | NZD | 1.6480 | Asia-Pacific |
| South African Rand | ZAR | 18.6800 | Africa |
| UAE Dirham | AED | 3.6730 | Middle East |

#### 4.4.7 INDUSTRY_MARGIN_PERCENTAGES

| Capital Goods Sector | Wholesale Margin (%) | Retail Margin (%) | Transport Margin (%) | Total Margin (%) |
|---------------------|---------------------|-------------------|---------------------|-----------------|
| Construction/Buildings | 5.0 | 0.0 | 3.0 | 8.0 |
| Heavy machinery | 10.0 | 5.0 | 4.0 | 19.0 |
| Light equipment | 12.0 | 8.0 | 3.0 | 23.0 |
| Vehicles (commercial) | 8.0 | 12.0 | 2.0 | 22.0 |
| IT hardware (servers) | 15.0 | 10.0 | 2.0 | 27.0 |
| IT hardware (end-user) | 15.0 | 15.0 | 2.0 | 32.0 |
| Furniture & fixtures | 12.0 | 18.0 | 5.0 | 35.0 |
| Electrical equipment | 10.0 | 8.0 | 3.0 | 21.0 |
| HVAC systems | 10.0 | 6.0 | 4.0 | 20.0 |
| Telecommunications equip | 12.0 | 8.0 | 2.0 | 22.0 |

#### 4.4.8 CAPITAL_EEIO_EMISSION_FACTORS

Top 30 capital-goods-specific NAICS-6 sectors with EEIO factors (see Section 2.2.1 for full table).

#### 4.4.9 CAPITAL_PHYSICAL_EMISSION_FACTORS

35+ material-level emission factors for capital asset materials (see Section 2.2.2 for full table).

#### 4.4.10 ASSET_USEFUL_LIFE_RANGES

| Asset Category | Subcategory | Min (years) | Max (years) | Typical (years) |
|---------------|-------------|-------------|-------------|----------------|
| Buildings | Office | 25 | 50 | 40 |
| Buildings | Warehouse | 20 | 40 | 30 |
| Buildings | Manufacturing | 20 | 40 | 30 |
| Buildings | Data center | 15 | 30 | 20 |
| Buildings | Laboratory | 20 | 40 | 30 |
| Machinery | CNC machines | 10 | 20 | 15 |
| Machinery | Presses | 10 | 25 | 15 |
| Machinery | Conveyors | 7 | 15 | 10 |
| Equipment | HVAC | 10 | 20 | 15 |
| Equipment | Generators | 10 | 25 | 15 |
| Equipment | Transformers | 15 | 30 | 20 |
| Equipment | Lab instruments | 5 | 15 | 10 |
| Vehicles | Passenger | 3 | 7 | 5 |
| Vehicles | Light commercial | 4 | 8 | 6 |
| Vehicles | Heavy truck | 5 | 10 | 7 |
| Vehicles | Specialized | 5 | 15 | 10 |
| Vehicles | Electric | 3 | 8 | 5 |
| IT Infrastructure | Servers | 3 | 5 | 4 |
| IT Infrastructure | Storage | 3 | 5 | 4 |
| IT Infrastructure | Network | 3 | 7 | 5 |
| IT Infrastructure | Desktop | 3 | 5 | 4 |
| IT Infrastructure | Laptop | 3 | 5 | 3 |
| Furniture | Office desk | 7 | 15 | 10 |
| Furniture | Office chair | 5 | 10 | 7 |
| Furniture | Shelving | 10 | 20 | 15 |
| Land Improvements | Paving | 15 | 30 | 20 |
| Land Improvements | Fencing | 10 | 20 | 15 |
| Land Improvements | Drainage | 15 | 30 | 20 |
| Leasehold Improvements | Interior buildout | Lease term | Lease term | Lease term |
| Leasehold Improvements | HVAC mods | Lease term | 15 | Lease term |

#### 4.4.11 CAPITALIZATION_THRESHOLDS

| Organization Size | Revenue Range | IFRS (USD) | US GAAP (USD) | Local GAAP (USD) |
|------------------|--------------|-----------|--------------|-----------------|
| Small | < $10M | 1,000 | 1,000 | 500 |
| Medium | $10M - $500M | 2,500 | 5,000 | 2,000 |
| Large | $500M - $5B | 5,000 | 10,000 | 5,000 |
| Enterprise | > $5B | 10,000 | 25,000 | 10,000 |

#### 4.4.12 FRAMEWORK_REQUIRED_DISCLOSURES

| Framework | Required Disclosures |
|-----------|---------------------|
| GHG Protocol | Total Cat 2 emissions (tCO2e), methodology per asset category, EF sources, data quality assessment, boundary description, base year recalculation policy |
| CSRD/ESRS E1 | E1-6 para 44: Cat 2 gross emissions, methodology, % supplier-specific, engagement strategy; para 46: intensity metrics; para 48: value chain engagement |
| CDP | C6.5: Cat 2 relevance, emissions figure, methodology, % calculated using primary data, emissions intensity, explanation of changes YoY |
| SBTi | Scope 3 screening results, Cat 2 significance (>40% threshold), data quality improvement plan, target coverage (67%+), supplier engagement strategy |
| SB 253 | Total Cat 2 in tCO2e, methodology description, data sources, safe harbor attestation (2027-2030), third-party assurance (limited -> reasonable) |
| GRI 305 | Scope 3 Cat 2 emissions if significant, methodology and EF sources, base year and recalculation, organizational boundary |
| ISO 14064 | Category 4 indirect emissions, methodology selection, uncertainty quantification, verification statement, comparison with prior periods |

### 4.5 Regulatory Frameworks (7)

1. **GHG Protocol Scope 3 Standard** -- Chapter 5 Category 2 definition (capital
   goods are fixed assets with multi-year useful life), Chapter 7 DQI, Chapter 9
   reporting requirements. Key rule: 100% of cradle-to-gate emissions in
   acquisition year, no depreciation of emissions.
2. **CSRD/ESRS E1** -- E1-6 para 44a/44b/44c Scope 3 by category, methodology,
   data sources; para 46 intensity; para 48 value chain engagement. Requires
   separate disclosure of Category 2 with methodology description.
3. **California SB 253** -- Scope 3 mandatory by FY2027 for >$1B revenue entities;
   GHG Protocol methodology; safe harbor 2027-2030; up to $500K penalty. Category 2
   must be reported if material (typically >5% of Scope 3).
4. **CDP Climate Change** -- C6.5 Category 2 relevance assessment and calculation;
   methodology description; percent calculated using primary (supplier-specific)
   data; year-over-year explanation for fluctuations.
5. **SBTi v5.3** -- Scope 3 target required if >40% of total; 67% coverage; supplier
   engagement targets; near-term 5-10 years. Category 2 included in screening
   and target boundary.
6. **GRI 305** -- Scope 3 disclosure if significant; methodology and EF sources;
   base year and recalculation policy. Category 2 reported separately.
7. **ISO 14064-1:2018** -- Category 4 indirect GHG emissions; methodology-neutral;
   uncertainty quantification; third-party verification. Requires clear boundary
   definition and completeness assessment.

### 4.6 Performance Targets

| Metric | Target |
|--------|--------|
| Spend-based calculation (500 line items) | < 300ms |
| Spend-based calculation (5,000 line items) | < 2s |
| Average-data calculation (500 line items) | < 300ms |
| Supplier-specific calculation (50 suppliers) | < 200ms |
| Hybrid aggregation (full inventory) | < 800ms |
| DQI scoring (full inventory) | < 200ms |
| Compliance check (all 7 frameworks) | < 300ms |
| Asset classification (single asset) | < 10ms |
| Classification cross-mapping (single code) | < 10ms |
| Currency + inflation adjustment (single) | < 5ms |
| Full pipeline (5,000 line items, hybrid) | < 8s |
| Capitalization threshold check (single item) | < 1ms |

---

## 5. Acceptance Criteria

### 5.1 Core Calculation

- [ ] Spend-based calculation with capital-goods-specific EEIO factors (30+ NAICS sectors)
- [ ] Spend-based calculation with EXIOBASE factors (capital goods product groups)
- [ ] Currency conversion for 20+ currencies using annual average FX rates
- [ ] Inflation deflation to EEIO base year using CPI/GDP deflator
- [ ] Margin removal (producer price adjustment) by capital goods sector
- [ ] Progressive NAICS matching (6-digit -> 4-digit -> 3-digit -> 2-digit fallback)
- [ ] Average-data calculation with 35+ physical emission factors
- [ ] ICE Database v3.0 integration for construction/building materials
- [ ] Unit conversion (kg, tonnes, m2, m3, pieces, kWp, kW)
- [ ] Transport-to-site emission inclusion for physical quantity method
- [ ] Multi-material asset decomposition (e.g., building = steel + concrete + glass)
- [ ] Supplier-specific calculation with product-level and facility-level data
- [ ] 5 allocation methods (economic, physical, mass, energy, hybrid)
- [ ] EPD/PCF/CDP/EcoVadis data integration and validation
- [ ] Hybrid aggregation combining all three methods with no double counting
- [ ] Coverage tracking (% of CapEx by method) with target thresholds
- [ ] 100% of cradle-to-gate emissions reported in year of acquisition (no depreciation)

### 5.2 Asset Classification & Database

- [ ] 8 asset categories with 40+ subcategories
- [ ] NAICS 2022 classification (capital-relevant 200+ industries)
- [ ] NACE Rev 2.1 classification (capital-relevant activities)
- [ ] UNSPSC v28 classification (capital equipment segments)
- [ ] ISIC Rev 4.1 as bridge standard
- [ ] Cross-mapping between all four systems with confidence scoring
- [ ] 8-level emission factor selection hierarchy
- [ ] EF versioning with source, year, database version tracking
- [ ] Useful life ranges by asset category (IAS 16 / ASC 360)
- [ ] Capitalization threshold enforcement by organization size and GAAP standard

### 5.3 Data Quality

- [ ] 5-dimension DQI scoring (temporal, geographical, technological, completeness, reliability)
- [ ] Composite DQI score (1.0-5.0 scale, arithmetic mean)
- [ ] Quality classification (Very Good through Very Poor)
- [ ] Pedigree matrix uncertainty quantification
- [ ] Weighted DQI for total inventory (emission-weighted)
- [ ] DQI improvement recommendations per asset/line item

### 5.4 Materiality & Analysis

- [ ] Hot-spot analysis with Pareto ranking of asset categories by emission contribution
- [ ] Materiality matrix classification (Q1-Q4)
- [ ] Year-over-year change decomposition (activity, EF, method, scope drivers)
- [ ] CapEx volatility context (rolling average, volatility ratio, contextual notes)
- [ ] Emission intensity metrics (revenue, FTE, CapEx, asset count)
- [ ] Industry benchmark comparison by sector

### 5.5 Double-Counting Prevention

- [ ] Category 1 vs Category 2 boundary enforcement (capitalization threshold + useful life)
- [ ] Scope 1/2 use-phase exclusion (cradle-to-gate boundary only)
- [ ] Category 8 leased asset exclusion (operating/finance lease check)
- [ ] Category 3 fuel/energy exclusion (filter fuel purchase orders)
- [ ] Intercompany transaction filtering
- [ ] Returns and credit memo netting
- [ ] Overlap detection across spend-based, average-data, and supplier-specific within Category 2
- [ ] Donated/granted asset exclusion (zero CapEx basis)

### 5.6 Compliance

- [ ] 7 regulatory framework compliance checks
- [ ] GHG Protocol Chapter 5 Category 2 definition compliance
- [ ] GHG Protocol no-depreciation rule enforcement
- [ ] CSRD/ESRS E1 data point coverage (para 44a/44b/44c)
- [ ] SB 253 methodology and coverage validation
- [ ] CDP C6.5 scoring criteria alignment (including YoY fluctuation explanation)
- [ ] SBTi coverage threshold validation (67% of Scope 3)
- [ ] GRI 305 and ISO 14064 compliance flags

### 5.7 Infrastructure

- [ ] 20 REST API endpoints
- [ ] V066 database migration (16 tables, 3 hypertables, 2 continuous aggregates)
- [ ] SHA-256 provenance on every calculation result
- [ ] Prometheus metrics with `gl_cg_` prefix
- [ ] Auth integration (route_protector.py + auth_setup.py)
- [ ] 1,000+ unit tests
- [ ] All calculations use Python `Decimal` (no floating point in emission path)
- [ ] Export in JSON, CSV, Excel, and PDF formats
- [ ] Row-Level Security (RLS) on all tenant-facing tables

---

## 6. Key Differentiators from Category 1 (AGENT-MRV-014)

This section explicitly documents the differences between Category 2 (this
agent) and Category 1 (AGENT-MRV-014: Purchased Goods & Services) to
prevent confusion during implementation.

| Aspect | Category 1 (AGENT-MRV-014) | Category 2 (AGENT-MRV-015) |
|--------|---------------------------|---------------------------|
| Emission timing | Year of purchase (annual flow) | 100% in year of acquisition (no depreciation) |
| Item classification | Expensed goods and services | Capitalized PP&E / fixed assets |
| Threshold | No capitalization threshold | Capitalization threshold enforced |
| Useful life tracking | Not applicable | Required (useful life ranges by asset type) |
| CapEx volatility | Low (relatively steady annual spend) | High (infrequent large purchases, 50-500% YoY swings) |
| Double-counting checks | vs Cat 2, Cat 3, Cat 4, Cat 6, Cat 7, Cat 8 | vs Cat 1, Cat 8, AND Scope 1/2 use-phase |
| ICE Database | Optional (for construction materials) | Primary source (buildings, land improvements) |
| Asset benchmarks | Not applicable | RICS embodied carbon, OEM sustainability reports |
| Financial accounting link | Purchase orders, invoices | Fixed asset register, depreciation schedule |
| Number of line items | Thousands to millions (all procurement) | Tens to hundreds (large assets) |
| Coverage feasibility | Supplier-specific hard (many vendors) | Supplier-specific easier (fewer, larger vendors) |
| Volatility context | Not required | Required (rolling avg, ratio, contextual notes) |

---

## 7. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic v2 | Data models, validation |
| FastAPI | REST API framework |
| prometheus_client | Prometheus metrics |
| psycopg[binary] | PostgreSQL driver |
| TimescaleDB | Hypertables and continuous aggregates |
| AGENT-MRV-014 | Purchased Goods & Services Agent (Category 1 boundary check, shared EEIO infra) |
| AGENT-DATA-009 | Spend Data Categorizer (NAICS/UNSPSC mapping, EEIO factors) |
| AGENT-DATA-003 | ERP/Finance Connector (fixed asset register extraction, CapEx data) |
| AGENT-DATA-008 | Supplier Questionnaire Processor (CDP, EcoVadis data for equipment OEMs) |
| AGENT-DATA-002 | Excel/CSV Normalizer (capital asset spreadsheets) |
| AGENT-DATA-001 | PDF & Invoice Extractor (supplier EPDs for construction products) |
| AGENT-DATA-010 | Data Quality Profiler (input data quality scoring) |
| AGENT-FOUND-003 | Unit & Reference Normalizer (unit conversion) |
| AGENT-FOUND-005 | Citations & Evidence Agent (EF source citations) |
| AGENT-FOUND-001 | Orchestrator (DAG pipeline execution) |
| AGENT-FOUND-008 | Reproducibility Agent (artifact hashing, drift detection) |
| AGENT-FOUND-009 | QA Test Harness (golden file testing) |
| AGENT-FOUND-010 | Observability Agent (metrics, traces, SLO tracking) |

---

## 8. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-25 | Initial PRD |
