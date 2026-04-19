# PRD: AGENT-MRV-014 -- Scope 3 Category 1 Purchased Goods & Services Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-001 |
| **Internal Label** | AGENT-MRV-014 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/purchased_goods_services/` |
| **DB Migration** | V065 |
| **Metrics Prefix** | `gl_pgs_` |
| **Table Prefix** | `pgs_` |
| **API** | `/api/v1/purchased-goods` |
| **Env Prefix** | `GL_PGS_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The Purchased Goods & Services Agent implements **GHG Protocol Scope 3
Category 1**, which covers all upstream (cradle-to-gate) emissions from the
production of goods and services purchased or acquired by the reporting
company in the reporting year. Category 1 is typically the **single largest
Scope 3 category** for most organizations, representing 20-70% of total
value chain emissions depending on industry sector.

The agent provides four calculation methods of varying accuracy and
coverage -- supplier-specific, average-data, spend-based, and hybrid -- so
that organizations can start with broad spend-based screening and
progressively improve data quality over time. It performs currency
conversion, inflation deflation, margin adjustment, classification
cross-mapping (NAICS/NACE/ISIC/UNSPSC), data quality scoring across five
GHG Protocol dimensions, hot-spot analysis, and compliance checking against
seven regulatory frameworks.

### Justification for Dedicated Agent

1. **Largest Scope 3 category** -- Category 1 dominates 20-70% of total Scope 3
   for most industries (retail, manufacturing, food & beverage, construction)
2. **Multi-method complexity** -- Four distinct calculation methods with different
   data requirements, accuracy levels, and aggregation rules
3. **EEIO integration** -- Requires specialized handling of 1,016+ EPA USEEIO
   commodities and 9,800+ EXIOBASE factors with currency/inflation adjustment
4. **Classification cross-mapping** -- NAICS, NACE, ISIC, and UNSPSC systems with
   many-to-many relationships
5. **Regulatory urgency** -- CSRD (FY2025+), SB 253 (FY2027), SBTi, CDP all
   require or strongly encourage Category 1 reporting
6. **Supplier engagement** -- Needs dedicated EPD/CDP/EcoVadis integration and
   allocation engines distinct from other Scope 3 categories
7. **Double-counting prevention** -- Category boundary enforcement against Cat 2
   (capital goods), Cat 3 (fuel/energy), Cat 4 (transport), Cat 6 (travel)

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011) -- Chapter 5
- GHG Protocol Scope 3 Technical Guidance (2013) -- Chapter 1: Category 1
- GHG Protocol Scope 3 Calculation Guidance (online)
- GHG Protocol Quantitative Uncertainty Guidance
- CSRD/ESRS E1 -- Scope 3 disclosure (E1-6 para 44)
- California SB 253 -- Mandatory Scope 3 by FY2027 for entities >$1B revenue
- CDP Climate Change Questionnaire -- C6.5 Scope 3 Category 1
- SBTi Corporate Manual v5.3 -- Scope 3 target required if >40% of total
- GRI 305 -- Emissions (Scope 3 disclosure)
- ISO 14064-1:2018 -- Category 4 indirect GHG emissions
- ISO 14025 -- Environmental Product Declarations (EPDs)
- ISO 14067 -- Product Carbon Footprints (PCFs)

---

## 2. Methodology

### 2.1 Four Calculation Methods

The GHG Protocol Technical Guidance defines four methods for Category 1,
listed from most to least accurate:

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | Supplier-Specific | Supplier cradle-to-gate data | Highest (+/-10-30%) | Lowest (key suppliers) |
| 2 | Hybrid | Mix of supplier + secondary data | High (+/-20-50%) | Medium |
| 3 | Average-Data | Physical quantities + LCA EFs | Medium (+/-30-60%) | Medium-High |
| 4 | Spend-Based | Spend amounts + EEIO factors | Lowest (+/-50-100%) | Highest (all spend) |

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

#### 2.1.1 Spend-Based Method (Lowest Accuracy, Broadest Coverage)

The spend-based method estimates emissions by multiplying the economic value
of each procurement category by an EEIO (Environmentally-Extended Input-Output)
emission factor. It applies to all spend data but carries the highest uncertainty.

**Core formula:**

```
Emissions_cat1_spend = SUM over all categories c:
    Spend_c (in base currency) * EEIO_factor_c (kgCO2e per base currency unit)
```

**With currency conversion and inflation adjustment:**

```
Emissions_cat1_spend = SUM over all categories c:
    (Spend_c_local / FX_rate_to_base) * (CPI_base_year / CPI_spend_year) * EEIO_factor_c
```

**Margin removal (producer price vs. purchaser price):**

```
Spend_producer = Spend_purchaser * (1 - margin_rate_c)
```

Where `margin_rate_c` is the average wholesale + retail + transport margin for
category c (typically 15-40% for goods, 5-15% for services).

**EEIO sources:**
- EPA USEEIO v1.2/v1.3 -- 1,016 commodities by NAICS-6, base year 2019 in 2021 USD
- EXIOBASE 3.8 -- 163 product groups x 49 regions (~9,800 factors), in EUR
- WIOD 2016 -- 43 countries, 56 sectors (older, less granular)
- GTAP 11 -- 141 regions, 65 sectors (subscription required)

#### 2.1.2 Average-Data Method (Medium Accuracy)

Uses physical quantities (mass, volume, units) multiplied by industry-average
cradle-to-gate emission factors from LCA databases.

**Core formula:**

```
Emissions_cat1_avg = SUM over all products p:
    Quantity_p (physical units) * EF_p (kgCO2e per physical unit)
```

**With transportation adder (when EF excludes transport):**

```
Emissions_cat1_avg = SUM over all products p:
    (Quantity_p * EF_production_p) + (Quantity_p * Distance_p * EF_transport_mode)
```

**EF sources:** ecoinvent v3.11 (21,000+ datasets), GaBi/Sphera (15,000+),
DEFRA/DESNZ (200+ materials), ICE v3.0 (200+ construction materials), World
Steel Association, International Aluminium Institute, PlasticsEurope, CEPI,
Textile Exchange, Glass Alliance Europe, FEFCO, ICA.

**Representative physical emission factors (30+ materials):**

| Material | EF (kgCO2e/kg) | Source |
|----------|----------------|--------|
| Primary steel (BOF) | 2.33 | World Steel 2023 |
| Secondary steel (EAF) | 0.67 | World Steel 2023 |
| Primary aluminum (global avg) | 16.70 | IAI 2023 |
| Secondary aluminum | 0.50 | IAI 2023 |
| Cement (Portland, global avg) | 0.63 | GCCA 2023 |
| Float glass | 1.20 | Glass Alliance Europe |
| HDPE | 1.80 | PlasticsEurope 2022 |
| LDPE | 2.08 | PlasticsEurope 2022 |
| PP (polypropylene) | 1.63 | PlasticsEurope 2022 |
| PET | 2.15 | PlasticsEurope 2022 |
| Corrugated cardboard | 0.79 | FEFCO 2022 |
| Kraft paper | 1.06 | CEPI 2022 |
| Recycled paper | 0.61 | CEPI 2022 |
| Cotton fiber (conventional) | 5.90 | Textile Exchange 2023 |
| Cotton fiber (organic) | 3.80 | Textile Exchange 2023 |
| Polyester fiber | 3.40 | Textile Exchange 2023 |
| Sawn timber (softwood) | 0.31 | ICE v3.0 |
| Sawn timber (hardwood) | 0.42 | ICE v3.0 |
| Copper (primary) | 4.10 | ICA 2022 |
| Lithium carbonate | 7.30 | IEA 2023 |
| Silicon wafer (solar grade) | 50.00 | IEA 2023 |
| Concrete (ready-mix, 30MPa) | 0.13 | GCCA 2023 |
| General steel (world avg, recycled content) | 1.37 | ICE v3.0 |
| Virgin steel (100% primary) | 2.89 | ICE v3.0 |
| General aluminum (33% recycled) | 6.67 | ICE v3.0 |
| Concrete (high strength, 50MPa) | 0.17 | ICE v3.0 |
| Cement (Portland CEM I) | 0.91 | ICE v3.0 |
| Timber (glulam) | 0.51 | ICE v3.0 |
| Bricks (general) | 0.24 | ICE v3.0 |
| Glass (general float) | 1.22 | ICE v3.0 |
| Lead | 1.57 | ICE v3.0 |
| Zinc | 3.86 | ICE v3.0 |

#### 2.1.3 Supplier-Specific Method (Highest Accuracy)

Uses primary data from suppliers on cradle-to-gate GHG emissions of their
products or services. This is the gold standard and the goal state for
high-impact procurement categories.

**Core formula (product-level data):**

```
Emissions_cat1_ss = SUM over all suppliers s, products p:
    Quantity_s_p * EF_supplier_s_p
```

**With allocation (facility-level data):**

```
Emissions_cat1_ss = SUM over all suppliers s:
    Emissions_total_s * Allocation_factor_s
```

**Allocation methods:**

| Method | Formula | Use Case |
|--------|---------|----------|
| Revenue-based | `Allocation = Revenue_customer / Revenue_total_s` | Services, diverse product mix |
| Mass-based | `Allocation = Mass_customer / Mass_total_s` | Commodities, bulk materials |
| Economic | `Allocation = Economic_value / Economic_value_total_s` | Multi-product facilities |
| Physical | `Allocation = Units_customer / Units_total_s` | Single-product lines |

**Supplier data sources:**
- Environmental Product Declarations (EPDs) -- ISO 14025
- Product Carbon Footprints (PCFs) -- ISO 14067
- CDP Supply Chain Program -- 35,000+ suppliers
- EcoVadis ratings -- 130,000+ companies
- PACT Network (WBCSD) -- product-level PCF exchange
- Direct supplier disclosure / sustainability reports

#### 2.1.4 Hybrid Method (Recommended)

Combines all three methods, using the most accurate data available for each
supplier or procurement category:

```
Emissions_cat1_hybrid = Emissions_SS + Emissions_AD + Emissions_SB
```

**Coverage strategy:**

| Tier | Suppliers | Target Coverage | Method |
|------|-----------|----------------|--------|
| Tier 1 | Top 20 by spend | 60-80% of total spend | Supplier-specific |
| Tier 2 | Next 30-50 | 15-25% of total spend | Average-data |
| Tier 3 | Remaining long tail | 5-15% of total spend | Spend-based |

**Weighted DQI for hybrid:**

```
DQI_hybrid = (Spend_SS * DQI_SS + Spend_AD * DQI_AD + Spend_SB * DQI_SB) / Spend_total
```

### 2.2 Spend Classification Systems

The agent supports four industry classification systems with cross-mapping:

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

### 2.3 Data Quality Indicator (DQI)

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

### 2.4 Uncertainty Ranges

| Method | Typical DQI Range | Uncertainty Range | Confidence Level |
|--------|------------------|------------------|-----------------|
| Supplier-Specific (verified EPD) | 1.0 - 1.5 | +/- 10-15% | Very High |
| Supplier-Specific (unverified) | 1.5 - 2.5 | +/- 15-30% | High |
| Average-Data (process LCA) | 2.0 - 3.0 | +/- 30-50% | Medium-High |
| Average-Data (industry avg) | 2.5 - 3.5 | +/- 40-60% | Medium |
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

### 2.5 Materiality & Hot-Spot Analysis

**Pareto 80/20 rule:** Approximately 80% of Category 1 emissions come from
20% of procurement categories. The top 5-10 categories typically represent
60-80% of emissions.

**Materiality matrix:**

| Quadrant | Spend | EF Intensity | Strategy |
|----------|-------|-------------|----------|
| Q1: Prioritize | High (>5% of total) | High (>0.5 kgCO2e/$) | Supplier-specific data, engagement |
| Q2: Monitor | Low (<5%) | High (>0.5 kgCO2e/$) | Average-data, track growth |
| Q3: Improve Data | High (>5%) | Low (<0.5 kgCO2e/$) | Average-data, refine EFs |
| Q4: Low Priority | Low (<5%) | Low (<0.5 kgCO2e/$) | Spend-based acceptable |

**Coverage thresholds:**

| Level | Target | Description |
|-------|--------|-------------|
| Minimum viable | >= 80% of spend | Required for credible reporting |
| Good practice | >= 90% of spend | Recommended by CDP/SBTi |
| Best practice | >= 95% of spend | Leading practice |

**Spend threshold recommendations:**

| Annual Spend per Category | Recommended Method |
|--------------------------|-------------------|
| > $10M | Supplier-specific |
| $1M - $10M | Average-data or hybrid |
| $100K - $1M | Average-data or spend-based |
| < $100K | Spend-based |

### 2.6 Key Formulas

**Currency conversion:**

```
Spend_base = Spend_local * (1 / FX_rate_to_base)
```

**Inflation deflation (to EEIO base year):**

```
Spend_real = Spend_nominal * (CPI_base_year / CPI_current_year)
```

**PPP adjustment (optional, for cross-country EEIO application):**

```
Spend_PPP = Spend_local * (PPP_factor_local / PPP_factor_base)
```

**Coverage tracking:**

```
Coverage_SS = Spend_SS / Spend_total
Coverage_AD = Spend_AD / Spend_total
Coverage_SB = Spend_SB / Spend_total
Coverage_total = Coverage_SS + Coverage_AD + Coverage_SB  (should = 1.0)
```

**Emissions intensity:**

```
Intensity_revenue = Emissions_cat1 / Revenue_total      (tCO2e per $M)
Intensity_employee = Emissions_cat1 / FTE_count          (tCO2e per FTE)
Intensity_product = Emissions_cat1 / Units_produced      (tCO2e per unit)
Intensity_spend = Emissions_cat1 / Total_procurement     (tCO2e per $M spend)
```

**Year-over-year change decomposition:**

```
Delta_emissions = Delta_activity + Delta_EF + Delta_method + Delta_scope

Where:
  Delta_activity = (Spend_current - Spend_prior) * EF_prior
  Delta_EF       = Spend_current * (EF_current - EF_prior)
  Delta_method   = Emissions_current_new_method - Emissions_current_old_method
  Delta_scope    = Change from boundary or scope changes
```

**Weighted DQI for total inventory:**

```
DQI_weighted = SUM over all line items i:
    (Emissions_i / Emissions_total) * DQI_i
```

### 2.7 Category Boundaries & Double-Counting Prevention

**Included in Category 1 (goods):** Raw materials, components, finished goods
for resale, packaging, office supplies, consumables, maintenance materials,
food/beverages, non-capital IT equipment.

**Included in Category 1 (services):** Professional, IT, outsourced operations,
marketing, telecom, insurance, training, staffing, R&D (outsourced), printing.

**Excluded (to prevent double counting):**

| Exclusion | Belongs To | Enforcement Rule |
|-----------|-----------|-----------------|
| Capital goods (above capitalization threshold) | Category 2 | Flag items where cost > threshold AND useful_life > 1yr |
| Fuel/energy for own operations | Category 3 | Filter utility and fuel purchase orders |
| Upstream transport (paid by reporter) | Category 4 | Check if EF includes transport; separate if not |
| Business travel | Category 6 | Filter travel-related spend codes |
| Employee commuting | Category 7 | Filter commuting-related spend |
| Waste from operations | Category 5 | Filter waste service POs |
| Upstream leased assets | Category 8 | Filter operating lease payments |
| Intra-company transactions | None | Filter by vendor type (intercompany) |
| Returns and credits | Adjustment | Net out credit memos |

### 2.8 Industry Benchmarks

**Category 1 as percentage of total Scope 3 by sector:**

| Industry Sector | Cat 1 as % of Total S3 | Cat 1 as % of S1+S2+S3 |
|----------------|----------------------|----------------------|
| Retail (apparel) | 60-80% | 55-75% |
| Retail (grocery) | 50-70% | 45-65% |
| Food & beverage manufacturing | 50-70% | 45-65% |
| Automotive manufacturing | 40-60% | 35-55% |
| Construction | 40-60% | 35-55% |
| Electronics manufacturing | 35-55% | 30-50% |
| Chemicals | 30-50% | 25-45% |
| Healthcare | 30-50% | 25-40% |
| Pharmaceuticals | 25-45% | 20-40% |
| Technology/Software | 20-40% | 15-35% |
| Telecommunications | 20-35% | 15-30% |
| Banking/Financial | 15-30% | 10-25% |
| Oil & Gas | 5-15% | 3-10% |

**Emission intensity benchmarks (tCO2e per $M revenue):**

| Sector | Cat 1 Intensity | Range |
|--------|----------------|-------|
| Food manufacturing | 300-1000 | Agricultural supply chain |
| Retail (grocery) | 300-800 | Food supply chain |
| Retail (apparel) | 200-600 | Textile supply chain |
| Chemicals | 200-500 | Energy-intensive upstream |
| Automotive | 150-400 | Materials-intensive |
| Construction | 150-400 | Materials-dominated |
| Electronics | 50-200 | High-value, lower mass |
| Pharmaceuticals | 30-100 | High-value, R&D-heavy |
| Technology/SaaS | 10-50 | IT infrastructure + services |
| Financial services | 5-30 | Service-dominated |

### 2.9 Emission Factor Selection Hierarchy

The agent implements an 8-level EF priority hierarchy:

| Priority | Source | DQI Score |
|----------|--------|-----------|
| 1 | Supplier-specific EPD or PCF (ISO 14025/14067, verified) | 1.0-1.5 |
| 2 | Supplier-specific CDP/direct disclosure (unverified) | 1.5-2.5 |
| 3 | Product-specific LCA data (ecoinvent, GaBi) | 2.0-3.0 |
| 4 | Material-specific average EF (ICE, World Steel, DEFRA) | 2.5-3.5 |
| 5 | Industry-average physical EF (literature, reports) | 3.0-3.5 |
| 6 | Regional EEIO factor (EXIOBASE for non-US) | 3.0-4.0 |
| 7 | National EEIO factor (EPA USEEIO for US) | 3.5-4.5 |
| 8 | Global average EEIO factor (fallback) | 4.0-5.0 |

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
+---------------------------------------------------------+
|                   AGENT-MRV-014                          |
|       Purchased Goods & Services Agent                   |
|                                                          |
|  +----------------------------------------------------+ |
|  | Engine 1: ProcurementDatabaseEngine                 | |
|  |   - EEIO emission factors (EPA USEEIO 1016, EXIO)  | |
|  |   - Physical emission factors (30+ materials)       | |
|  |   - NAICS/NACE/ISIC/UNSPSC classification mapping   | |
|  |   - Currency exchange rates and PPP factors         | |
|  |   - Industry margin percentages for spend adjust    | |
|  |   - Supplier emission factor registry               | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 2: SpendBasedCalculatorEngine                 | |
|  |   - Spend * EEIO factor calculation                 | |
|  |   - Currency conversion (50+ currencies)            | |
|  |   - Inflation deflation (CPI base year adjustment)  | |
|  |   - Margin removal (producer vs purchaser price)    | |
|  |   - NAICS-to-EEIO mapping                           | |
|  |   - Batch spend category processing                 | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 3: AverageDataCalculatorEngine                | |
|  |   - Quantity * physical EF calculation              | |
|  |   - Material category classification               | |
|  |   - Unit conversion (kg, tonnes, liters, m3, pcs)  | |
|  |   - Waste and loss factor application              | |
|  |   - Transport-to-gate emission inclusion           | |
|  |   - Multi-material product allocation              | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 4: SupplierSpecificCalculatorEngine           | |
|  |   - Supplier emission allocation (revenue/mass/econ)| |
|  |   - EPD data integration                           | |
|  |   - CDP Supply Chain data processing               | |
|  |   - Supplier data quality validation               | |
|  |   - Cradle-to-gate boundary verification           | |
|  |   - Allocation factor calculation                  | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 5: HybridAggregatorEngine                    | |
|  |   - Method prioritization and selection            | |
|  |   - Coverage analysis (% spend per method)         | |
|  |   - Hot-spot analysis (Pareto ranking)             | |
|  |   - Method blending with weighting                 | |
|  |   - Gap filling (fallback to lower-tier methods)   | |
|  |   - Double-counting detection and prevention       | |
|  |   - Total Category 1 emission aggregation          | |
|  |   - Year-over-year trend and decomposition         | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 6: ComplianceCheckerEngine                   | |
|  |   - 7 frameworks: GHG Protocol, CSRD/ESRS, CDP,   | |
|  |     SBTi, California SB 253, GRI 305, ISO 14064   | |
|  |   - Coverage thresholds (>=80% spend)              | |
|  |   - DQI scoring validation                         | |
|  |   - Methodology documentation checks              | |
|  |   - Boundary verification                          | |
|  |   - Double-counting prevention checks              | |
|  +----------------------------------------------------+ |
|                         |                                |
|  +----------------------------------------------------+ |
|  | Engine 7: PurchasedGoodsPipelineEngine               | |
|  |   - 10-stage pipeline orchestration                | |
|  |   - Batch multi-period processing                  | |
|  |   - Multi-facility aggregation                     | |
|  |   - Category boundary enforcement                  | |
|  |   - Export (JSON/CSV/Excel)                        | |
|  |   - Compliance-ready outputs (CDP, CSRD, SBTi)    | |
|  |   - Provenance chain assembly                     | |
|  +----------------------------------------------------+ |
+---------------------------------------------------------+
```

### 3.2 File Structure

```
greenlang/purchased_goods_services/
+-- __init__.py                          # Lazy imports, module exports
+-- models.py                            # Pydantic v2 models, enums, constants
+-- config.py                            # GL_PGS_ prefixed configuration
+-- metrics.py                           # Prometheus metrics (gl_pgs_*)
+-- provenance.py                        # SHA-256 provenance chain
+-- procurement_database.py              # Engine 1: Procurement & EF database
+-- spend_based_calculator.py            # Engine 2: Spend-based calculation
+-- average_data_calculator.py           # Engine 3: Average-data calculation
+-- supplier_specific_calculator.py      # Engine 4: Supplier-specific calculation
+-- hybrid_aggregator.py                 # Engine 5: Hybrid aggregation
+-- compliance_checker.py                # Engine 6: Compliance checking
+-- purchased_goods_pipeline.py          # Engine 7: Pipeline orchestration
+-- setup.py                             # Service facade
+-- api/
    +-- __init__.py                      # API package
    +-- router.py                        # FastAPI REST endpoints

tests/unit/mrv/test_purchased_goods_services/
+-- __init__.py
+-- conftest.py
+-- test_models.py
+-- test_config.py
+-- test_metrics.py
+-- test_provenance.py
+-- test_procurement_database.py
+-- test_spend_based_calculator.py
+-- test_average_data_calculator.py
+-- test_supplier_specific_calculator.py
+-- test_hybrid_aggregator.py
+-- test_compliance_checker.py
+-- test_purchased_goods_pipeline.py
+-- test_setup.py
+-- test_api.py

deployment/database/migrations/sql/
+-- V065__purchased_goods_services_service.sql
```

### 3.3 Database Schema (V065)

16 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `pgs_eeio_factors` | EPA USEEIO + EXIOBASE EEIO factors | Seed (100+ NAICS sectors) |
| `pgs_physical_efs` | Physical emission factors (30+ materials) | Seed (30+ rows) |
| `pgs_supplier_efs` | Supplier-specific emission factors (EPD, CDP, PCF) | Dimension |
| `pgs_naics_codes` | NAICS 2022 classification codes | Seed (1,057 industries) |
| `pgs_nace_codes` | NACE Rev 2.1 classification codes | Seed (615 activities) |
| `pgs_unspsc_codes` | UNSPSC v28 segments and families | Seed (55 segments) |
| `pgs_classification_mapping` | Cross-mapping between NAICS/NACE/ISIC/UNSPSC | Seed |
| `pgs_currency_rates` | Annual exchange rates (50+ currencies) | Dimension |
| `pgs_margin_factors` | Wholesale/retail/transport margin by sector | Dimension |
| `pgs_inflation_indices` | CPI/GDP deflator by country and year | Dimension |
| `pgs_calculations` | Calculation results (spend/avg/supplier/hybrid) | Hypertable |
| `pgs_calculation_details` | Line-item detail per calculation | Regular |
| `pgs_supplier_data` | Supplier profiles and engagement status | Regular |
| `pgs_dqi_scores` | Data quality scores per line item | Regular |
| `pgs_compliance_checks` | Compliance check results (7 frameworks) | Regular |
| `pgs_batch_jobs` | Batch processing jobs | Regular |
| `pgs_hourly_stats` | Hourly calculation statistics | Continuous Aggregate |
| `pgs_daily_stats` | Daily calculation statistics | Continuous Aggregate |

**Key seed data:**

- `pgs_eeio_factors`: 100+ rows from EPA USEEIO v1.2 top sectors (NAICS-6 code, factor kgCO2e/USD, database version, base year, margin type)
- `pgs_physical_efs`: 30+ rows of material-level emission factors from World Steel, IAI, GCCA, PlasticsEurope, ICE, DEFRA
- `pgs_naics_codes`: 1,057 NAICS-6 industries (code, title, sector, subsector)
- `pgs_nace_codes`: 615 NACE Rev 2.1 activities (code, title, section, division)
- `pgs_unspsc_codes`: 55 segments with family-level detail
- `pgs_classification_mapping`: NAICS-ISIC, NACE-ISIC, UNSPSC-NAICS correspondence entries

### 3.4 API Endpoints (20)

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/calculate/spend-based` | Run spend-based calculation |
| 2 | POST | `/calculate/average-data` | Run average-data calculation |
| 3 | POST | `/calculate/supplier-specific` | Run supplier-specific calculation |
| 4 | POST | `/calculate/hybrid` | Run hybrid (multi-method) calculation |
| 5 | POST | `/calculate/batch` | Batch multi-period calculation |
| 6 | GET | `/calculations/{calculation_id}` | Get calculation result |
| 7 | GET | `/calculations/{calculation_id}/details` | Get line-item detail |
| 8 | POST | `/procurement/upload` | Upload procurement records |
| 9 | GET | `/procurement/summary` | Get procurement spend summary |
| 10 | POST | `/suppliers` | Register or update supplier profile |
| 11 | GET | `/suppliers/{supplier_id}` | Get supplier profile and EF data |
| 12 | GET | `/suppliers/{supplier_id}/emissions` | Get supplier emission allocation |
| 13 | GET | `/emission-factors/eeio` | List EEIO factors (with filtering) |
| 14 | GET | `/emission-factors/physical` | List physical EFs (with filtering) |
| 15 | POST | `/dqi/score` | Score data quality for a record set |
| 16 | GET | `/dqi/{calculation_id}` | Get DQI results for a calculation |
| 17 | POST | `/compliance/check` | Run compliance check (all 7 frameworks) |
| 18 | GET | `/compliance/frameworks` | List available compliance frameworks |
| 19 | POST | `/export` | Export results (JSON/CSV/Excel) |
| 20 | GET | `/health` | Health check |

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

### 4.2 Enumerations (20)

| Enum | Values | Description |
|------|--------|-------------|
| `CalculationMethod` | SUPPLIER_SPECIFIC, HYBRID, AVERAGE_DATA, SPEND_BASED | 4 GHG Protocol methods |
| `EEIODatabase` | EPA_USEEIO_V12, EPA_USEEIO_V13, EXIOBASE_V38, WIOD_2016, GTAP_11 | 5 EEIO databases |
| `PhysicalEFDatabase` | ECOINVENT_V311, GABI, DEFRA_2025, ICE_V3, WORLD_STEEL_2023, IAI_2023, PLASTICS_EUROPE_2022 | 7 physical EF databases |
| `ClassificationSystem` | NAICS_2022, NACE_REV2, NACE_REV21, ISIC_REV4, UNSPSC_V28 | 5 classification systems |
| `AllocationMethod` | REVENUE_BASED, MASS_BASED, ECONOMIC, PHYSICAL, ENERGY_BASED | 5 allocation methods |
| `DQILevel` | VERY_GOOD, GOOD, FAIR, POOR, VERY_POOR | 5 quality tiers |
| `DQIDimension` | TEMPORAL, GEOGRAPHICAL, TECHNOLOGICAL, COMPLETENESS, RELIABILITY | 5 DQI dimensions |
| `SupplierDataSource` | EPD, PCF, CDP_SUPPLY_CHAIN, ECOVADIS, DIRECT_DISCLOSURE, SUSTAINABILITY_REPORT, PACT_NETWORK | 7 supplier data sources |
| `ComplianceFramework` | GHG_PROTOCOL, CSRD_ESRS_E1, SB_253, CDP, SBTI, GRI_305, ISO_14064 | 7 regulatory frameworks |
| `ProcurementType` | RAW_MATERIAL, COMPONENT, FINISHED_GOOD, PACKAGING, CONSUMABLE, PROFESSIONAL_SERVICE, IT_SERVICE, OUTSOURCED_OPERATION, MAINTENANCE, MARKETING, INSURANCE, TELECOM, TRAINING, OTHER_GOOD, OTHER_SERVICE | 15 procurement types |
| `PriceType` | BASIC_PRICE, PURCHASER_PRICE | 2 price types (producer vs. purchaser) |
| `MaterialityQuadrant` | PRIORITIZE, MONITOR, IMPROVE_DATA, LOW_PRIORITY | 4 hot-spot quadrants |
| `CoverageLevel` | MINIMUM, GOOD, BEST, COMPLETE | 4 coverage tiers |
| `PipelineStage` | INGEST, CLASSIFY, BOUNDARY_CHECK, SPEND_CALC, AVGDATA_CALC, SUPPLIER_CALC, AGGREGATE, DQI_SCORE, COMPLIANCE_CHECK, EXPORT | 10 pipeline stages |
| `ExportFormat` | JSON, CSV, EXCEL | 3 export formats |
| `ComplianceStatus` | COMPLIANT, NON_COMPLIANT, PARTIAL, NOT_APPLICABLE | 4 compliance statuses |
| `BatchStatus` | PENDING, RUNNING, COMPLETED, FAILED, PARTIAL | 5 batch statuses |
| `TrendDirection` | INCREASING, DECREASING, STABLE | 3 trend directions |
| `ChangeDriver` | ACTIVITY_CHANGE, EF_UPDATE, METHOD_UPGRADE, SCOPE_CHANGE | 4 YoY change drivers |
| `EmissionGas` | CO2, CH4, N2O, CO2E | 4 emission gases |

### 4.3 Regulatory Frameworks (7)

1. **GHG Protocol Scope 3 Standard** -- Chapter 5 Category 1 definition, Chapter 7 DQI, Chapter 9 reporting requirements
2. **CSRD/ESRS E1** -- E1-6 para 44a/44b/44c Scope 3 by category, methodology, data sources; para 46 intensity; para 48 value chain engagement
3. **California SB 253** -- Scope 3 mandatory by FY2027 for >$1B revenue entities; GHG Protocol methodology; safe harbor 2027-2030; up to $500K penalty
4. **CDP Climate Change** -- C6.5 Category 1 relevance and calculation; methodology assessment; spend coverage; supplier engagement scoring
5. **SBTi v5.3** -- Scope 3 target if >40% of total; 67% coverage; supplier engagement targets; near-term 5-10 years
6. **GRI 305** -- Scope 3 disclosure if significant; methodology and EF sources; base year and recalculation policy
7. **ISO 14064-1:2018** -- Category 4 indirect emissions; methodology-neutral; uncertainty quantification; third-party verification

### 4.4 Performance Targets

| Metric | Target |
|--------|--------|
| Spend-based calculation (1,000 line items) | < 500ms |
| Spend-based calculation (10,000 line items) | < 3s |
| Average-data calculation (1,000 line items) | < 500ms |
| Supplier-specific calculation (100 suppliers) | < 300ms |
| Hybrid aggregation (full inventory) | < 1s |
| DQI scoring (full inventory) | < 200ms |
| Compliance check (all 7 frameworks) | < 300ms |
| Classification cross-mapping (single code) | < 10ms |
| Currency + inflation adjustment (single) | < 5ms |
| Full pipeline (10,000 line items, hybrid) | < 10s |

---

## 5. Acceptance Criteria

### 5.1 Core Calculation

- [ ] Spend-based calculation with EPA USEEIO factors (1,016 commodity mapping)
- [ ] Spend-based calculation with EXIOBASE factors (163 product groups x 49 regions)
- [ ] Currency conversion for 50+ currencies using annual average FX rates
- [ ] Inflation deflation to EEIO base year using CPI/GDP deflator
- [ ] Margin removal (producer price adjustment) by sector
- [ ] Average-data calculation with 30+ physical emission factors
- [ ] Unit conversion (kg, tonnes, liters, m3, pieces, kWh)
- [ ] Transport adder when cradle-to-gate EF excludes inbound transport
- [ ] Supplier-specific calculation with product-level and facility-level data
- [ ] 5 allocation methods (revenue, mass, economic, physical, energy)
- [ ] EPD/PCF/CDP data integration and validation
- [ ] Hybrid aggregation combining all three methods with no double counting
- [ ] Coverage tracking (% of spend by method) with target thresholds

### 5.2 Classification & Database

- [ ] NAICS 2022 classification (1,057 industries, 20 sectors)
- [ ] NACE Rev 2.1 classification (615 activities, 21 sections)
- [ ] UNSPSC v28 classification (55 segments, 60,000+ commodities)
- [ ] ISIC Rev 4.1 as bridge standard
- [ ] Cross-mapping between all four systems with confidence scoring
- [ ] 8-level emission factor selection hierarchy (supplier EPD down to global EEIO)
- [ ] EF versioning with source, year, database version tracking

### 5.3 Data Quality

- [ ] 5-dimension DQI scoring (temporal, geographical, technological, completeness, reliability)
- [ ] Composite DQI score (1.0-5.0 scale, arithmetic mean)
- [ ] Quality classification (Very Good through Very Poor)
- [ ] Pedigree matrix uncertainty quantification
- [ ] Weighted DQI for total inventory (emission-weighted)
- [ ] DQI improvement recommendations per line item

### 5.4 Materiality & Analysis

- [ ] Hot-spot analysis with Pareto ranking of categories by emission contribution
- [ ] Materiality matrix classification (Q1-Q4)
- [ ] Year-over-year change decomposition (activity, EF, method, scope drivers)
- [ ] Emission intensity metrics (revenue, FTE, production unit, spend)
- [ ] Industry benchmark comparison

### 5.5 Double-Counting Prevention

- [ ] Capital goods boundary enforcement (Category 1 vs Category 2)
- [ ] Fuel/energy exclusion (Category 1 vs Category 3)
- [ ] Transport boundary checking (Category 1 vs Category 4)
- [ ] Business travel exclusion (Category 1 vs Category 6)
- [ ] Intercompany transaction filtering
- [ ] Returns and credit memo netting
- [ ] Overlap detection across spend-based, average-data, and supplier-specific

### 5.6 Compliance

- [ ] 7 regulatory framework compliance checks
- [ ] GHG Protocol Chapter 9 reporting completeness
- [ ] CSRD/ESRS E1 data point coverage (para 44a/44b/44c)
- [ ] SB 253 methodology and coverage validation
- [ ] CDP C6.5 scoring criteria alignment
- [ ] SBTi coverage threshold validation (67% of Scope 3)
- [ ] GRI 305 and ISO 14064 compliance flags

### 5.7 Infrastructure

- [ ] 20 REST API endpoints
- [ ] V065 database migration (16 tables, 3 hypertables, 2 continuous aggregates)
- [ ] SHA-256 provenance on every calculation result
- [ ] Prometheus metrics with `gl_pgs_` prefix
- [ ] Auth integration (route_protector.py + auth_setup.py)
- [ ] 1,000+ unit tests
- [ ] All calculations use Python `Decimal` (no floating point in emission path)
- [ ] Export in JSON, CSV, and Excel formats

---

## 6. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic v2 | Data models, validation |
| FastAPI | REST API framework |
| prometheus_client | Prometheus metrics |
| psycopg[binary] | PostgreSQL driver |
| TimescaleDB | Hypertables and continuous aggregates |
| AGENT-DATA-009 | Spend Data Categorizer (NAICS/UNSPSC mapping, EEIO factors) |
| AGENT-DATA-003 | ERP/Finance Connector (procurement data extraction) |
| AGENT-DATA-008 | Supplier Questionnaire Processor (CDP, EcoVadis data) |
| AGENT-DATA-002 | Excel/CSV Normalizer (procurement spreadsheets) |
| AGENT-DATA-001 | PDF & Invoice Extractor (supplier EPDs) |
| AGENT-DATA-010 | Data Quality Profiler (input data quality scoring) |
| AGENT-FOUND-003 | Unit & Reference Normalizer (unit conversion) |
| AGENT-FOUND-005 | Citations & Evidence Agent (EF source citations) |
| AGENT-FOUND-001 | Orchestrator (DAG pipeline execution) |
| AGENT-FOUND-008 | Reproducibility Agent (artifact hashing, drift detection) |
| AGENT-FOUND-009 | QA Test Harness (golden file testing) |
| AGENT-FOUND-010 | Observability Agent (metrics, traces, SLO tracking) |

---

## 7. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-24 | Initial PRD |
