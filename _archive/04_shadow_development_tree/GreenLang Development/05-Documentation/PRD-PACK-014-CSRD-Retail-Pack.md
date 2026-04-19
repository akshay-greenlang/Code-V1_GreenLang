# PRD-PACK-014: CSRD Retail & Consumer Goods Pack

**Pack ID:** PACK-014-csrd-retail
**Category:** EU Compliance / CSRD
**Tier:** Sector-Specific (Retail & Consumer Goods)
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-16

---

## 1. Executive Summary

PACK-014 is a **sector-specific CSRD Solution Pack** purpose-built for retail and consumer goods companies (NACE Division G: G46-G47) required to comply with the EU Corporate Sustainability Reporting Directive. While PACK-001/002/003 provide general CSRD compliance, retail companies face a fundamentally different sustainability profile that demands specialized engines:

- **Scope 3 Dominance**: Retail emissions are 80-95% Scope 3 (vs 30-60% for manufacturing), with Category 1 (purchased goods and services) alone representing 50-75% of total emissions. This inverts the usual compliance priority -- store-level Scope 1+2 is comparatively minor.
- **Product Regulation Convergence**: Unlike any other sector, retailers face 5-6 simultaneous product-level regulations: PPWR (packaging), ESPR (ecodesign/DPP), EUDR (deforestation-free commodities), EU Textile Strategy (EPR/microplastics), ECGT (green claims), and the Forced Labour Regulation. Each applies at the individual product or commodity level, creating exponential complexity.
- **Consumer-Facing Greenwashing Risk**: The EU Empowering Consumers for the Green Transition Directive (ECGT 2024/825) directly prohibits generic environmental claims ("eco", "green", "sustainable") from September 27, 2026, with penalties up to 4% of turnover. Retailers are the primary enforcement target.
- **Multi-Location Operations**: Large retailers operate hundreds to thousands of stores, each a separate emissions source (electricity, heating, refrigeration F-gas leakage), requiring granular store-level MRV with group-level consolidation.
- **Supply Chain Breadth and Depth**: Retailers source from Tier 1-4+ suppliers across garments, food, electronics, and household goods, requiring deep supply chain due diligence under CSDDD and EUDR simultaneously.
- **Circular Economy Centrality**: ESRS E5 (Resource Use and Circular Economy) is almost always material for retail, covering packaging waste, product take-back, textile EPR, WEEE obligations, and material circularity indices.

PACK-014 reuses approximately 70-75% of existing platform components (all 30 MRV agents, 20 DATA agents, 10 FOUND agents) and builds 25-30% net-new retail-specific functionality through 8 specialized engines, 8 workflows, 8 templates, and 10 integrations.

**Target Users**: Retail and consumer goods companies with >1,000 employees AND >EUR 450M net turnover (Omnibus I threshold), subject to CSRD reporting under ESRS Set 1. Secondary users include wholesale trade companies (G46) and non-store retailers (G47.9).

**Key Differentiators vs PACK-013 (Manufacturing)**:

| Dimension | PACK-013 Manufacturing | PACK-014 Retail |
|-----------|----------------------|-----------------|
| Dominant emissions | Scope 1 process emissions | Scope 3 purchased goods |
| Key regulation | EU ETS, IED/BAT | PPWR, ECGT, EUDR |
| Supply chain focus | BOM-based, Tier 1-2 | Category-based, Tier 1-4+ |
| Circular economy | Waste diversion, recycled content | Take-back, EPR schemes, packaging |
| Location model | Facilities (1-50) | Stores (100-5,000+) |
| Product complexity | Industrial goods, PCF | Consumer goods, DPP, green claims |

---

## 2. Regulatory Scope

### 2.1 Primary Regulations

| Regulation | Reference | Effective | Retail Relevance | Penalty |
|------------|-----------|-----------|------------------|---------|
| CSRD | Directive (EU) 2022/2464 | FY2025/2026+ | Core sustainability reporting | Market access |
| ESRS Set 1 | Delegated Reg (EU) 2023/2772 | With CSRD | 12 cross-sector standards; E1, E5, S2, S4 most material | N/A (part of CSRD) |
| Omnibus I | Directive (EU) 2026/470 | 2026 | >1,000 employees AND >EUR 450M turnover; 61% datapoint reduction | N/A (threshold change) |
| PPWR | Regulation (EU) 2025/40 | Aug 2026 labeling; 2030 targets | Packaging: 30% recycled PET by 2030, 50-65% by 2040, EPR eco-modulation | Varies by MS |
| ECGT | Directive (EU) 2024/825 | Sep 27, 2026 | Anti-greenwashing: prohibits generic eco/green claims | Up to 4% turnover |
| EUDR | Regulation (EU) 2023/1115 | Dec 30, 2026 | 7 commodities: palm oil, soy, cocoa, coffee, rubber, timber, cattle | Up to 4% turnover |
| CSDDD | Directive (EU) 2024/1760 | Jul 2028 Phase 1 | Supply chain human rights + environmental due diligence | Up to 5% worldwide turnover |
| ESPR | Regulation (EU) 2024/1781 | Textiles 2027; electronics 2027-28 | Digital Product Passports, ecodesign requirements | Product withdrawal |
| EU Textile Strategy | Proposed regulation | Mid-2028 (EPR) | Mandatory EPR for textiles, microplastic 30% reduction by 2030 | Varies |
| Forced Labour Reg | Regulation (EU) 2024/3015 | Dec 14, 2027 | Product-level enforcement, no size threshold | Product ban |
| EED | Directive (EU) 2023/1791 | 2026-2027 | Energy audits for stores (10-85 TJ), ISO 50001 (>85 TJ) | Varies by MS |
| F-Gas Regulation | Regulation (EU) 2024/573 | 2025+ | Refrigeration: GWP <150 for new systems, HFC phase-down | Product ban |
| Food Waste | EC Decision 2019/1597 + binding target | 2030 | 30% food waste reduction (binding), measurement methodology | Varies by MS |

### 2.2 Standards and Frameworks

| Standard | Scope | Retail Application |
|----------|-------|-------------------|
| GHG Protocol | Corporate + Scope 3 Standard | Full Scope 1-3 accounting, 15 categories |
| SBTi | FLAG + Corporate Net-Zero Standard | Sector pathway for retail, FLAG for food retailers |
| CDP | Climate, Forests, Water questionnaires | Supply chain program, supplier scoring |
| TNFD | LEAP approach for nature | Nature-related dependencies (food, textiles) |
| ISO 14064-1 | Organization GHG | Store + group level |
| ISO 14040/14044 | Life Cycle Assessment | Product environmental footprint |
| PEF/PEFCR | Product Environmental Footprint | EU methodology for product claims |
| Ellen MacArthur Foundation | Material Circularity Index | Circular economy measurement |

### 2.3 ESRS Materiality for Retail

The following ESRS topics are typically material for retail companies (based on double materiality assessment):

| ESRS | Topic | Material? | Retail Rationale |
|------|-------|-----------|-----------------|
| E1 | Climate Change | Always | Scope 3 dominance, store energy, refrigeration |
| E2 | Pollution | Sometimes | Packaging chemicals, textile dyes (sub-sector dependent) |
| E3 | Water and Marine Resources | Sometimes | Food supply chain, cotton/textile supply chain |
| E4 | Biodiversity and Ecosystems | Often | EUDR commodities, agricultural supply chains |
| E5 | Circular Economy | Always | Packaging, product end-of-life, take-back, EPR |
| S1 | Own Workforce | Always | Large retail workforce, store employees |
| S2 | Workers in the Value Chain | Always | CSDDD, forced labour risk, garment workers |
| S3 | Affected Communities | Sometimes | EUDR deforestation communities |
| S4 | Consumers and End-Users | Always | Product safety, green claims, accessibility |
| G1 | Business Conduct | Always | Anti-corruption, supplier payments |

---

## 3. Architecture

### 3.1 Pack Structure

```
PACK-014-csrd-retail/
+-- __init__.py
+-- pack.yaml
+-- config/
|   +-- __init__.py
|   +-- pack_config.py
|   +-- presets/
|   |   +-- __init__.py
|   |   +-- grocery_retail.yaml          # Food/grocery: refrigeration + food waste + EUDR
|   |   +-- apparel_retail.yaml          # Fashion: textile EPR + EUDR cotton + supply chain DD
|   |   +-- electronics_retail.yaml      # Electronics: WEEE + use-phase + DPP
|   |   +-- general_retail.yaml          # Department stores: balanced profile
|   |   +-- online_retail.yaml           # E-commerce: last-mile + packaging + returns
|   |   +-- sme_retailer.yaml            # Simplified for smaller retailers
|   +-- demo/
|       +-- __init__.py
|       +-- demo_config.yaml
+-- engines/
|   +-- __init__.py
|   +-- store_emissions_engine.py         # Engine 1: Store-level Scope 1+2
|   +-- retail_scope3_engine.py           # Engine 2: Retail Scope 3 (all 15 categories)
|   +-- packaging_compliance_engine.py    # Engine 3: PPWR packaging compliance
|   +-- product_sustainability_engine.py  # Engine 4: ESPR/DPP/PEF/green claims
|   +-- food_waste_engine.py             # Engine 5: Food waste tracking
|   +-- supply_chain_due_diligence_engine.py  # Engine 6: CSDDD/EUDR/forced labour
|   +-- retail_circular_economy_engine.py     # Engine 7: EPR/take-back/MCI
|   +-- retail_benchmark_engine.py        # Engine 8: Sector benchmarking
+-- workflows/
|   +-- __init__.py
|   +-- store_emissions_workflow.py              # Workflow 1: Store emissions assessment
|   +-- supply_chain_assessment_workflow.py      # Workflow 2: Supply chain emissions + DD
|   +-- packaging_compliance_workflow.py         # Workflow 3: PPWR compliance
|   +-- product_sustainability_workflow.py       # Workflow 4: Product DPP/PEF/claims
|   +-- food_waste_tracking_workflow.py          # Workflow 5: Food waste measurement
|   +-- circular_economy_workflow.py             # Workflow 6: EPR/take-back/circularity
|   +-- esrs_retail_disclosure_workflow.py       # Workflow 7: ESRS disclosure generation
|   +-- regulatory_compliance_workflow.py        # Workflow 8: Multi-regulation compliance
+-- templates/
|   +-- __init__.py
|   +-- store_emissions_report.py                # Template 1: Store + consolidated emissions
|   +-- supply_chain_report.py                   # Template 2: Scope 3 hotspot + supplier scorecard
|   +-- packaging_compliance_report.py           # Template 3: PPWR compliance dashboard
|   +-- product_sustainability_report.py         # Template 4: DPP summary + PEF results
|   +-- food_waste_report.py                     # Template 5: Food waste reduction progress
|   +-- circular_economy_report.py               # Template 6: Take-back + EPR + MCI metrics
|   +-- retail_esg_scorecard.py                  # Template 7: Executive KPI dashboard
|   +-- esrs_retail_disclosure.py                # Template 8: Full ESRS chapter generation
+-- integrations/
|   +-- __init__.py
|   +-- pack_orchestrator.py                     # Master orchestrator (11-phase retail pipeline)
|   +-- csrd_pack_bridge.py                      # Bridge to PACK-001/002/003
|   +-- mrv_retail_bridge.py                     # Bridge to MRV agents (Scope 1/2/3)
|   +-- data_retail_bridge.py                    # Bridge to DATA agents + retail ERP
|   +-- eudr_retail_bridge.py                    # Bridge to EUDR agents for commodity tracing
|   +-- circular_economy_bridge.py               # Bridge to EPR schemes + waste agents
|   +-- supply_chain_bridge.py                   # Bridge to CSDDD/forced labour agents
|   +-- taxonomy_bridge.py                       # EU Taxonomy alignment for retail activities
|   +-- health_check.py                          # 22-category system health verification
|   +-- setup_wizard.py                          # 8-step retail-specific configuration wizard
+-- tests/
    +-- __init__.py
    +-- conftest.py                              # Shared fixtures (retail company, store data)
    +-- test_manifest.py                         # Pack YAML validation
    +-- test_config.py                           # Config system tests
    +-- test_demo.py                             # Demo smoke tests
    +-- test_store_emissions.py                  # Engine 1 tests
    +-- test_retail_scope3.py                    # Engine 2 tests
    +-- test_packaging_compliance.py             # Engine 3 tests
    +-- test_product_sustainability.py           # Engine 4 tests
    +-- test_food_waste.py                       # Engine 5 tests
    +-- test_supply_chain_dd.py                  # Engine 6 tests
    +-- test_circular_economy.py                 # Engine 7 tests
    +-- test_retail_benchmark.py                 # Engine 8 tests
    +-- test_workflows.py                        # All 8 workflows
    +-- test_templates.py                        # All 8 templates + registry
    +-- test_integrations.py                     # All 10 integrations
    +-- test_e2e.py                              # End-to-end flows
    +-- test_agent_integration.py                # Agent wiring verification
```

### 3.2 Components Summary

| Category | Count | Description |
|----------|-------|-------------|
| Engines | 8 | Store emissions, Scope 3, packaging, product sustainability, food waste, supply chain DD, circular economy, benchmarking |
| Workflows | 8 | Store emissions, supply chain, packaging, product, food waste, circular economy, ESRS disclosure, regulatory compliance |
| Templates | 8 | Store emissions, supply chain, packaging, product, food waste, circular economy, ESG scorecard, ESRS disclosure |
| Integrations | 10 | Orchestrator, CSRD bridge, MRV bridge, DATA bridge, EUDR bridge, circular economy bridge, supply chain bridge, taxonomy bridge, health check, setup wizard |
| Presets | 6 | Grocery, apparel, electronics, general, online, SME |
| Tests | 18 | conftest + manifest + config + demo + 8 engines + workflows + templates + integrations + e2e + agent integration |

### 3.3 Key Architectural Decisions

1. **Store-centric data model**: Unlike manufacturing (facility-centric), retail uses a store-centric model where each physical location is a separate emissions boundary. Stores are grouped into regions, banners, and the corporate entity for consolidation.

2. **Category-based Scope 3**: Manufacturing uses BOM-based Scope 3 calculation. Retail uses category-based calculation where purchased goods are classified by product category (food, apparel, electronics, household) and mapped to emission factors per category rather than per component.

3. **Multi-regulation product overlay**: A single product (e.g., a cotton t-shirt) may simultaneously trigger EUDR (cotton origin), ESPR (DPP), EU Textile Strategy (EPR), ECGT (green claims), Forced Labour Regulation (supply chain), and PPWR (packaging). The product sustainability engine handles this convergence.

4. **Preset-driven configuration**: Each retail sub-sector has fundamentally different materiality profiles. The preset system activates only the relevant engines and regulations per sub-sector (e.g., grocery preset enables food waste + refrigerant F-gas; apparel preset enables textile EPR + EUDR cotton).

---

## 4. Engine Specifications

### 4.1 Engine 1: Store Emissions Engine

**File**: `engines/store_emissions_engine.py`
**Purpose**: Calculate store-level Scope 1 and Scope 2 emissions for individual retail locations and consolidate across the entire store portfolio.

**Key Features**:
- **Scope 1 sources**: Refrigerant leakage (F-gas: R404A, R407C, R410A, R744/CO2, R290/propane), on-site heating (natural gas, heating oil), backup generators (diesel), company fleet vehicles
- **Scope 2 sources**: Purchased electricity (kWh per store), district heating/cooling, on-site renewable generation (solar PV, wind) and green tariff/PPA tracking
- **F-Gas Regulation compliance**: GWP tracking per refrigerant type, GWP <150 requirement for new commercial refrigeration systems (2025+), HFC phase-down schedule alignment, leak rate monitoring
- **Multi-location consolidation**: Roll up from individual store to region to banner to corporate group, with allocation rules for shared facilities (distribution centers, offices, data centers)
- **Store typology**: Hypermarket (>2,500 sqm), supermarket (400-2,500 sqm), convenience (<400 sqm), department store, specialty store, warehouse club, discount store, e-commerce fulfillment center
- **Energy benchmarking**: kWh/sqm, kWh/revenue, emissions/sqm by store type and climate zone
- **Renewable energy tracking**: On-site generation, PPAs, green tariffs, RECs/GOs with market-based vs location-based differentiation

**Core Calculations**:
```
Store Scope 1 = Refrigerant_Leakage + Heating_Combustion + Fleet_Vehicles + Backup_Generators
  where:
    Refrigerant_Leakage = SUM(charge_kg * annual_leak_rate * GWP)
    Heating_Combustion  = SUM(fuel_qty * NCV * emission_factor)

Store Scope 2 (location) = Electricity_kWh * Grid_Factor_Country + District_Heat * Heat_Factor
Store Scope 2 (market)   = Electricity_kWh * Supplier_Factor (or residual mix if no contract)

Portfolio Total = SUM(all stores) + Distribution_Centers + Offices + Data_Centers
```

**Regulatory References**:
- F-Gas Regulation (EU) 2024/573 -- GWP limits, HFC phase-down, leak checks
- ESRS E1 -- Scope 1 and Scope 2 GHG emissions
- GHG Protocol -- Corporate Standard Chapter 4/6
- EED (EU) 2023/1791 -- Energy audit obligations

**Models**: `StoreEmissionsConfig`, `StoreData`, `RefrigerantData`, `EnergyConsumptionData`, `StoreEmissionsResult`, `PortfolioConsolidation`, `StoreTypology`, `EnergyBenchmark`

**Edge Cases**:
- Store with no sub-metered electricity (use area-based allocation from building meter)
- Refrigerant charge unknown (use system capacity default by equipment type)
- Mid-year store opening/closure (pro-rate by operational months)
- Franchise stores (Scope 1+2 excluded from corporate boundary unless operational control)

### 4.2 Engine 2: Retail Scope 3 Engine

**File**: `engines/retail_scope3_engine.py`
**Purpose**: Calculate all 15 Scope 3 categories with retail-specific prioritization, where Category 1 (purchased goods and services) dominates at 50-75% of total emissions.

**Key Features**:
- **Category prioritization for retail**: Cat 1 >> Cat 4/9 >> Cat 11/12 >> Cat 7 >> remainder. Engine allocates calculation effort proportional to materiality.
- **4 calculation methods**: Supplier-specific (highest quality), hybrid (partial supplier data + industry average), average-data (emission factors per product category), spend-based (economic input-output, lowest quality)
- **Product category emission factors**: Pre-loaded for 200+ retail product categories (fresh produce, frozen food, dairy, meat, bakery, beverages, apparel by fiber type, electronics by product type, household goods, personal care, toys, pet products)
- **Data quality scoring**: 5-level retail data quality ladder:
  - Level 1: Supplier-specific verified data (CDP, PACT)
  - Level 2: Supplier-reported unverified
  - Level 3: Product category average (e.g., DEFRA, Exiobase)
  - Level 4: Spend-based EEIO models
  - Level 5: Revenue proxy estimation
- **Hotspot identification**: Automatic identification of top 20 emission-contributing product categories and top 50 suppliers by emissions
- **Supplier engagement tracking**: Integration with CDP Supply Chain, EcoVadis, SEDEX, and proprietary supplier questionnaires
- **Transport emissions (Cat 4/9)**: Mode-specific (road, rail, sea, air), route optimization, last-mile delivery (Cat 9) including customer travel for in-store retail
- **Use-phase emissions (Cat 11)**: Relevant for electronics retailers -- energy consumption of sold products over lifetime
- **End-of-life (Cat 12)**: Packaging waste, product disposal, textile waste, food waste at consumer level
- **Employee commuting (Cat 7)**: Large retail workforce, shift patterns, store location accessibility

**Core Calculations**:
```
Cat 1 (Purchased Goods) = SUM(product_category_quantity * category_emission_factor)
  OR = SUM(supplier_reported_emissions)  [if supplier-specific]
  OR = SUM(spend_EUR * EEIO_factor)      [if spend-based]

Cat 4 (Upstream Transport) = SUM(tonne_km * mode_factor) for all inbound logistics
Cat 9 (Downstream Transport) = Last_mile_delivery + Customer_travel_to_store

Total Scope 3 = SUM(Cat 1..15)
Data Quality Score = Weighted average of category-level DQ scores (weighted by emission share)
```

**Models**: `RetailScope3Config`, `ProductCategoryData`, `SupplierEmissionsData`, `TransportData`, `Scope3Result`, `CategoryBreakdown`, `HotspotAnalysis`, `SupplierScorecard`, `DataQualityAssessment`

### 4.3 Engine 3: Packaging Compliance Engine

**File**: `engines/packaging_compliance_engine.py`
**Purpose**: Track and validate compliance with PPWR (Regulation (EU) 2025/40) packaging requirements including recycled content, reuse targets, EPR eco-modulation, and labeling.

**Key Features**:
- **Recycled content tracking by material type**:
  - PET: 30% by 2030, 50% by 2040
  - Other contact-sensitive plastics: 10% by 2030, 50% by 2040
  - Non-contact-sensitive plastics: 35% by 2030, 65% by 2040
  - Single-use plastic beverage bottles: 30% by 2030, 65% by 2040
  - Glass, metal, paper: voluntary targets with eco-modulation incentives
- **EPR eco-modulation grading**: Grade A (best) to Grade E (worst) based on recyclability, recycled content, reusability, hazardous substances. Fee modulation up to +/- 50% of base EPR fee.
- **Packaging inventory management**: Track all packaging types (primary, secondary, tertiary/transport) by material (PET, HDPE, PP, PS, PVC, glass, aluminum, tinplate, paper/board, wood, multi-material composite), weight, and volume
- **Reuse targets**: 2030/2040 reuse percentage targets for transport packaging (40%/70%), e-commerce packaging (10%/50%), grouped packaging (10%/40%)
- **Labeling compliance**: Mandatory harmonized labeling from August 2026 -- material identification, sorting instructions, QR code linking to detailed disposal info, "reusable" or "recyclable" marks
- **E-commerce packaging**: Specific PPWR rules for e-commerce -- void ratio (<40%), over-packaging prohibition, right-sizing requirements
- **Packaging reduction targets**: 5% by 2030, 10% by 2035, 15% by 2040 vs 2018 baseline (per capita weight)

**Core Calculations**:
```
Recycled Content % = (post_consumer_recycled_weight + pre_consumer_recycled_weight)
                     / total_material_weight * 100
  Must meet: PET >= 30% (by 2030), other contact >= 10%, non-contact >= 35%

EPR Eco-Modulation Grade = f(recyclability_score, recycled_content_%,
                             reusability, hazardous_substances)
  Grade A: >= 90 points -> -50% fee
  Grade E: < 20 points -> +50% fee

Void Ratio (e-commerce) = (package_internal_volume - product_volume)
                          / package_internal_volume * 100
  Must be: < 40%

Packaging Intensity = total_packaging_weight_kg / units_sold
```

**Models**: `PackagingComplianceConfig`, `PackagingItem`, `MaterialBreakdown`, `RecycledContentData`, `EPREcoModulation`, `LabelingCompliance`, `PackagingComplianceResult`, `ReuseTarget`, `VoidRatioAssessment`

### 4.4 Engine 4: Product Sustainability Engine

**File**: `engines/product_sustainability_engine.py`
**Purpose**: Manage product-level sustainability data for ESPR Digital Product Passports, PEF calculations, green claims substantiation (ECGT), and textile-specific requirements.

**Key Features**:
- **Digital Product Passport (DPP) data management**:
  - Textiles (2027): Fiber composition, country of manufacturing, durability rating, recyclability, microplastic release
  - Electronics (2027-2028): Energy efficiency, reparability score, recycled content, hazardous substances, expected lifetime
  - Furniture (2028): Material composition, recyclability, formaldehyde emissions, durability
  - DPP data model: 100+ fields per product category, QR code generation, blockchain-ready hash
- **Product Environmental Footprint (PEF)**:
  - 16 impact categories (carbon footprint, water use, land use, ecotoxicity, etc.)
  - PEFCR (Product Environmental Footprint Category Rules) implementation for food, apparel, electronics
  - Lifecycle stages: raw materials, manufacturing, distribution, use, end-of-life
  - Data quality rating per lifecycle stage
- **Green Claims Substantiation (ECGT)**:
  - Claim registry: Track all environmental claims made on products, packaging, and marketing
  - Substantiation evidence: Link each claim to PEF/LCA data, certifications, or verified measurements
  - Prohibited claims detection: Flag generic claims ("eco-friendly", "green", "sustainable", "climate-neutral" without offset disclosure)
  - Certification validation: Verify third-party labels (EU Ecolabel, GOTS, OEKO-TEX, FSC, MSC, Fairtrade)
  - Comparative claims: Requirements for like-for-like comparison, methodology disclosure
- **Textile-specific requirements**:
  - Microplastic release: Grams per wash cycle by fiber type (polyester, nylon, acrylic), 30% reduction target by 2030
  - Durability testing: Minimum wash cycle count by garment type
  - Fiber composition accuracy: Labeling compliance per Regulation (EU) No 1007/2011
- **Repairability scoring**: French repairability index methodology adaptation for EU-wide deployment (electronics, appliances)

**Models**: `ProductSustainabilityConfig`, `ProductData`, `DPPRecord`, `PEFResult`, `GreenClaim`, `ClaimSubstantiation`, `TextileData`, `RepairabilityScore`, `ProductSustainabilityResult`

### 4.5 Engine 5: Food Waste Engine

**File**: `engines/food_waste_engine.py`
**Purpose**: Measure, track, and report food waste across the retail value chain per EC Decision 2019/1597, targeting the binding 30% reduction by 2030.

**Key Features**:
- **Waste stream categories**: Bakery, produce (fruit/vegetables), dairy, meat/fish, deli/prepared foods, frozen foods, ambient/dry goods, beverages, non-food organic waste (flowers, plants)
- **Measurement methodology (EC 2019/1597)**: Weight-based measurement at each stage -- receiving (damaged goods), storage (expiry/spoilage), display (shrinkage), preparation (trimming/processing), customer returns
- **Waste hierarchy tracking**:
  - Prevention: demand forecasting accuracy, dynamic markdown/pricing
  - Redistribution: food bank donation (kg), charity partnerships, employee sales
  - Animal feed: qualifying waste streams routed to registered feed operators
  - Composting/anaerobic digestion: on-site and third-party organic processing
  - Energy recovery: waste-to-energy facilities
  - Landfill: residual waste (target: minimize to <5% of total food waste)
- **30% reduction target tracking**: Baseline year establishment, annual progress measurement, reduction trajectory modeling, remediation actions when off-track
- **Store-level waste benchmarking**: Waste rate (%) by department (bakery typically 8-15%, produce 5-10%, dairy 2-5%, meat 3-8%), waste per sqm, waste per revenue
- **Financial impact**: Cost of waste (purchase cost of wasted goods), donation tax benefits, EPR savings from prevention, disposal cost avoidance
- **Scope 3 Cat 12 linkage**: Food waste at consumer level (downstream) estimated from product shelf-life data and consumption patterns

**Core Calculations**:
```
Food Waste Rate (%) = food_waste_weight_kg / food_purchased_weight_kg * 100

Waste Hierarchy Distribution:
  Prevention %     = prevented_waste / total_potential_waste * 100
  Redistribution % = donated_weight / total_waste * 100
  Recycling %      = (composting + AD + animal_feed) / total_waste * 100
  Recovery %       = energy_recovery / total_waste * 100
  Disposal %       = landfill / total_waste * 100
  MUST SUM TO 100%

Reduction vs Baseline = (baseline_waste - current_waste) / baseline_waste * 100
  Target: >= 30% by 2030

GHG Avoidance = redistributed_kg * emission_factor_avoided_production
```

**Models**: `FoodWasteConfig`, `WasteStreamData`, `WasteMeasurement`, `WasteHierarchyBreakdown`, `FoodWasteResult`, `ReductionTrajectory`, `RedistributionRecord`, `WasteBenchmark`

### 4.6 Engine 6: Supply Chain Due Diligence Engine

**File**: `engines/supply_chain_due_diligence_engine.py`
**Purpose**: Comprehensive supply chain due diligence covering CSDDD, EUDR commodity tracing, forced labour risk scoring, and human rights impact assessment for retail supply chains.

**Key Features**:
- **CSDDD compliance (Directive 2024/1760)**:
  - Scope: Direct and indirect business partners in the chain of activities
  - Adverse impacts: Human rights (ILO core conventions, UNGP) + environmental (Paris Agreement, CBD)
  - Due diligence steps: Identify, assess, prevent, mitigate, remediate, monitor
  - Remediation tracking: Corrective action plans per supplier, timeline, verification
  - Climate transition plan: Integration with SBTi targets and decarbonization pathway
- **EUDR commodity tracing (Regulation 2023/1115)**:
  - 7 commodity groups: Palm oil, soy, cocoa, coffee, rubber, timber, cattle (+ derived products)
  - Geolocation data: GPS coordinates of production plots (for all non-negligible risk)
  - Deforestation-free verification: Satellite imagery cross-reference (post-Dec 31, 2020 cutoff)
  - Due diligence statement: Per-commodity DDS generation for EU customs submission
  - Risk classification: Country + commodity risk level (standard/high/low)
  - Retail-specific: Private label products with EUDR commodities as ingredients
- **Forced Labour Regulation (2024/3015)**:
  - Product-level risk assessment: Region of origin, commodity type, supplier audit history
  - Risk indicators: ILO forced labour indicators (11 categories)
  - Country risk scoring: US DOL List of Goods, UK Modern Slavery, Walk Free Global Slavery Index
  - Remediation: Immediate cessation, worker compensation, supplier development
- **Human rights impact assessment**:
  - Salient issue identification per UNGPs
  - Stakeholder engagement records
  - Grievance mechanism tracking
  - Living wage gap analysis for value chain workers (ESRS S2)
- **Supplier tier mapping**:
  - Tier 1: Direct suppliers (retailers typically 500-5,000)
  - Tier 2: Component/ingredient suppliers
  - Tier 3: Raw material processors
  - Tier 4+: Primary producers (farms, mines, forests)
  - Coverage tracking: % of spend mapped to each tier level
- **Country risk factors**: Governance (WGI), corruption (CPI), labour rights (ITUC), environmental risk, conflict zones

**Models**: `SupplyChainDDConfig`, `SupplierProfile`, `CSDDDAssessment`, `EUDRCommodityTrace`, `ForcedLabourRisk`, `HumanRightsImpact`, `SupplierTierMap`, `CountryRiskScore`, `RemediationPlan`, `DueDiligenceResult`

### 4.7 Engine 7: Retail Circular Economy Engine

**File**: `engines/retail_circular_economy_engine.py`
**Purpose**: Track circular economy performance across all EPR schemes, take-back programs, material circularity, and product end-of-life pathways relevant to retail.

**Key Features**:
- **EPR scheme compliance tracking**:
  - Packaging EPR: All EU Member State schemes (e.g., Gruner Punkt/DE, CITEO/FR, Conai/IT, Ecoembes/ES, Valpak/UK)
  - WEEE EPR: Collection targets (65% of average weight placed on market), producer registration, financing
  - Battery EPR: Collection (45% by 2023, 63% by 2027, 73% by 2030), recycled content
  - Textile EPR: Mandatory from mid-2028, collection infrastructure, sorting, recycling
  - Vehicle EPR: Relevant for automotive parts retailers
  - Per-scheme: Registration status, fee calculation, declaration filing, target achievement
- **Take-back programs**:
  - In-store collection: Electronics (WEEE), batteries, textiles, packaging
  - Program metrics: Collection volume (kg), collection rate (% of sold), cost per kg, consumer participation rate
  - Logistics: Reverse logistics tracking, collection point coverage, partner management
- **Material Circularity Index (MCI)**:
  - Ellen MacArthur Foundation methodology
  - Product-level MCI: Virgin input fraction, recycled input fraction, utility factor, end-of-life recycling rate
  - Portfolio-level MCI: Weighted average across product range
- **Product end-of-life pathway tracking**:
  - By product category: Recyclable %, reusable %, compostable %, energy recoverable %, landfill %
  - Actual vs designed: Track real-world recycling rates vs design intent
  - Infrastructure availability: Local recycling infrastructure assessment by geography
- **Textile circularity specifics**:
  - Fiber-to-fiber recycling rate
  - Mono-material vs multi-material garment ratio
  - Design for recyclability scoring
  - Second-hand/resale program metrics

**Core Calculations**:
```
Material Circularity Index (MCI) = 1 - LFI * F(X)
  where:
    LFI = Linear Flow Index = (V + W) / (2M + Wf - Wc)
    V   = Virgin material input
    W   = Unrecoverable waste
    M   = Product mass
    Wf  = Waste from recycling process
    Wc  = Waste collected for recycling
    F(X) = Utility factor (product lifetime / industry average lifetime)

EPR Fee = base_fee * eco_modulation_factor * declared_weight_tonnes
  eco_modulation_factor: 0.5 (Grade A) to 1.5 (Grade E)

Collection Rate = collected_weight / (average_3yr_placed_on_market_weight) * 100
  WEEE target: >= 65%
```

**Models**: `CircularEconomyConfig`, `EPRSchemeData`, `TakeBackProgram`, `MCICalculation`, `EndOfLifePathway`, `TextileCircularity`, `CircularEconomyResult`, `EPRFeeCalculation`, `CollectionRateData`

### 4.8 Engine 8: Retail Benchmark Engine

**File**: `engines/retail_benchmark_engine.py`
**Purpose**: Benchmark retail sustainability performance against sector peers using standardized KPIs, enabling relative positioning and target-setting.

**Key Features**:
- **Emissions intensity metrics**:
  - tCO2e per sqm of selling space (Scope 1+2)
  - tCO2e per million EUR revenue (Scope 1+2+3)
  - tCO2e per employee (Scope 1+2)
  - tCO2e per transaction (Scope 1+2+3)
  - Scope 3 data quality score (0-100)
- **Energy metrics**:
  - kWh per sqm per year (by store type and climate zone)
  - Renewable electricity % (on-site + contracted)
  - Energy Use Intensity (EUI) vs sub-sector median
- **Circular economy metrics**:
  - Waste diversion rate (% diverted from landfill)
  - Packaging recycled content % (weighted average)
  - Food waste rate (% for grocery retailers)
  - Take-back collection rate
  - Material Circularity Index (portfolio average)
- **Supply chain metrics**:
  - Scope 3 data coverage (% of spend with supplier-specific data)
  - Supplier SBTi adoption rate
  - EUDR compliance rate (% of EUDR-relevant products with DDS)
  - CSDDD due diligence coverage (% of high-risk suppliers assessed)
- **SBTi alignment**:
  - 1.5C pathway alignment check (retail sector pathway)
  - Near-term target progress (42% reduction Scope 1+2 by 2030)
  - FLAG target for food retailers (deforestation + land use)
  - Long-term net-zero target gap analysis
- **Peer comparison**:
  - Ranking within retail sub-sector (grocery, apparel, electronics, general, online)
  - Percentile position (top quartile, median, bottom quartile, laggard)
  - Year-over-year improvement trajectory
  - CDP score comparison (A-list, management, awareness, disclosure)

**Benchmark Database Sources**:
- CDP Climate Change responses (retail sector)
- SBTi target database
- Eurostat energy statistics (NACE G)
- Industry reports: BRC, EuroCommerce, NRF sustainability benchmarks
- Company sustainability reports (top 50 EU retailers)

**Models**: `BenchmarkConfig`, `RetailKPIs`, `SectorBenchmark`, `BenchmarkResult`, `PeerComparison`, `SBTiAlignment`, `TrajectoryAnalysis`, `PercentileRanking`

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Store Emissions Workflow

**File**: `workflows/store_emissions_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | DataCollection | Store portfolio data, energy bills, refrigerant logs, fleet records | Validate store data, normalize energy units, verify refrigerant charges | Validated store dataset |
| 2 | Scope1Calculation | Validated store data | Calculate refrigerant leakage (F-gas GWP), heating combustion, fleet, generators per store | Store-level Scope 1 results |
| 3 | Scope2Calculation | Energy consumption data, grid factors, green tariffs | Calculate location-based and market-based Scope 2 per store | Store-level Scope 2 results (dual reporting) |
| 4 | Consolidation | All store results | Aggregate by region, banner, corporate; calculate intensity KPIs; generate portfolio view | Consolidated portfolio emissions report |

**Trigger**: Quarterly or annually, upon energy data availability.
**Duration**: <5 minutes for 500 stores, <15 minutes for 5,000 stores.

### 5.2 Workflow 2: Supply Chain Assessment Workflow

**File**: `workflows/supply_chain_assessment_workflow.py`
**Phases**: 5

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | SupplierMapping | Procurement data, supplier master | Map suppliers to tiers (1-4+), categorize by product category and geography | Supplier tier map with coverage metrics |
| 2 | DataCollection | Supplier questionnaires, CDP data, EcoVadis, certifications | Collect and validate supplier-specific emission data; apply data quality scoring | Scored supplier emissions dataset |
| 3 | EmissionCalculation | Supplier data + product category emission factors | Calculate Scope 3 Cat 1 using best-available method per supplier; calculate Cat 4/9 transport | Scope 3 category results with DQ scores |
| 4 | HotspotAnalysis | Scope 3 results | Identify top 20 product categories and top 50 suppliers by emissions; flag high-risk geographies | Hotspot report with prioritized action list |
| 5 | EngagementPlanning | Hotspot results, supplier profiles | Generate supplier engagement plan: SBTi adoption requests, data improvement targets, switch-to-renewable encouragement | Supplier engagement plan + scorecards |

**Trigger**: Annually for full assessment; quarterly for top-50 supplier updates.
**Duration**: <30 minutes for 5,000 suppliers.

### 5.3 Workflow 3: Packaging Compliance Workflow

**File**: `workflows/packaging_compliance_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | PackagingInventory | Product packaging specifications, material data sheets | Catalog all packaging by type (primary/secondary/tertiary), material, weight, recycled content | Complete packaging inventory |
| 2 | RecycledContentAssessment | Packaging inventory, supplier certificates | Calculate recycled content % per material vs PPWR targets; identify gaps | Recycled content gap analysis |
| 3 | EPRCompliance | Packaging inventory, EPR scheme rules per MS | Calculate EPR fees with eco-modulation; verify registration status per country; check declaration accuracy | EPR compliance dashboard with fee projections |
| 4 | LabelingAudit | Packaging designs, labeling requirements | Audit packaging labels against PPWR harmonized labeling rules; check QR codes, sorting instructions, material marks | Labeling compliance report with non-conformances |

**Trigger**: Annually for full assessment; on-demand for new product launches.
**Duration**: <10 minutes for 10,000 SKUs.

### 5.4 Workflow 4: Product Sustainability Workflow

**File**: `workflows/product_sustainability_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | ProductCatalog | Product master data, material composition, supplier data | Classify products by ESPR category (textiles, electronics, furniture); identify DPP-relevant products | Product sustainability catalog |
| 2 | DPPGeneration | Product data, LCA data, certifications | Generate Digital Product Passport data for applicable products; calculate PEF scores | DPP records with QR codes |
| 3 | GreenClaimsAudit | Marketing materials, product labels, website claims | Scan all environmental claims against ECGT prohibited list; verify substantiation evidence; flag unsupported claims | Green claims audit report |
| 4 | ComplianceReport | DPP records, claims audit, regulatory timelines | Assess overall product sustainability compliance; identify upcoming deadlines; generate remediation roadmap | Product sustainability compliance report |

**Trigger**: Quarterly for claims audit; annually for DPP preparation; on-demand for new products.
**Duration**: <15 minutes for 50,000 SKUs.

### 5.5 Workflow 5: Food Waste Tracking Workflow

**File**: `workflows/food_waste_tracking_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | WasteBaseline | Historical waste data, purchase volumes by category | Establish baseline year waste rates by department (bakery, produce, dairy, meat, deli, frozen, ambient) | Baseline waste profile |
| 2 | CategoryAnalysis | Current period waste measurements | Analyze waste by cause (expiry, damage, shrinkage, preparation) and department; identify highest-waste categories | Waste cause analysis by department |
| 3 | ReductionTargeting | Waste analysis, industry benchmarks | Set reduction targets per department to achieve 30% overall by 2030; model intervention scenarios (markdown, donation, ordering) | Reduction targets with intervention plan |
| 4 | ProgressReporting | Current vs baseline, redistribution records | Calculate reduction progress; report waste hierarchy distribution; monetize avoidance; generate EC 2019/1597 report | Food waste progress report |

**Trigger**: Monthly for progress tracking; annually for formal reporting.
**Duration**: <5 minutes for 500 stores.

### 5.6 Workflow 6: Circular Economy Workflow

**File**: `workflows/circular_economy_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | TakeBackPrograms | Collection records, reverse logistics data | Track take-back volumes (WEEE, batteries, textiles, packaging) per store; calculate collection rates | Take-back program performance |
| 2 | MaterialRecovery | Collected materials, recycling/reuse records | Track material recovery pathways; calculate recycling yield; monitor contamination rates | Material recovery metrics |
| 3 | EPRSchemeCompliance | EPR registrations, declarations, fee payments | Verify compliance across all applicable EPR schemes per Member State; flag delinquencies | EPR compliance status per scheme per MS |
| 4 | CircularityMetrics | All circular economy data | Calculate Material Circularity Index (portfolio), waste diversion rate, circularity revenue share | Circular economy KPI dashboard |

**Trigger**: Quarterly for EPR declarations; annually for MCI calculation.
**Duration**: <10 minutes.

### 5.7 Workflow 7: ESRS Retail Disclosure Workflow

**File**: `workflows/esrs_retail_disclosure_workflow.py`
**Phases**: 4

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | MaterialityAssessment | Industry materiality profile, stakeholder input, Omnibus I reduced datapoints | Conduct double materiality assessment; identify material topics; map to ESRS disclosure requirements | Materiality matrix with ESRS mapping |
| 2 | DataPointCollection | All engine outputs, HR data, governance data | Collect all required ESRS datapoints (reduced set per Omnibus I); validate completeness | ESRS datapoint register with completeness score |
| 3 | DisclosureGeneration | Validated datapoints, narrative templates | Generate ESRS disclosure chapters: E1 (climate), E5 (circular economy), S2 (value chain workers), S4 (consumers), G1 (conduct) | Draft ESRS disclosure document |
| 4 | AuditPreparation | Disclosure draft, evidence files, provenance hashes | Prepare audit evidence package; cross-reference all quantitative disclosures to source data; generate auditor working papers | Audit-ready disclosure with evidence pack |

**Trigger**: Annually, aligned with financial reporting cycle.
**Duration**: <30 minutes for full disclosure generation.

### 5.8 Workflow 8: Regulatory Compliance Workflow

**File**: `workflows/regulatory_compliance_workflow.py`
**Phases**: 3

| Phase | Name | Inputs | Processing | Outputs |
|-------|------|--------|------------|---------|
| 1 | RegulationMapping | Company profile (sub-sector, geographies, products, revenue, employees) | Map applicable regulations from the regulatory universe to the company; assess applicability per regulation per operating country | Regulatory applicability matrix |
| 2 | ComplianceAssessment | Applicability matrix, current compliance status | Assess compliance level per regulation (fully compliant, partially compliant, non-compliant, not yet applicable); identify gaps | Compliance gap register with risk scores |
| 3 | ActionPlanning | Compliance gaps, regulatory timelines, resource constraints | Generate prioritized action plan with deadlines, responsible parties, estimated cost, and dependencies | Compliance action plan with Gantt-style timeline |

**Trigger**: Quarterly for status updates; on-demand when new regulations are published.
**Duration**: <5 minutes.

---

## 6. Template Specifications

### 6.1 Template 1: Store Emissions Report

**File**: `templates/store_emissions_report.py`
**Outputs**: Per-store emissions card, regional roll-up, portfolio summary, energy benchmark dashboard.

**Key Sections**:
- Executive summary: Total Scope 1+2 with YoY change
- Store-level detail: Table of all stores with Scope 1 (refrigerant, heating, fleet), Scope 2 (location, market), intensity (tCO2e/sqm)
- Regional consolidation: Aggregation by country/region with heatmap
- Refrigerant analysis: F-gas inventory, leak rates, GWP-weighted emissions, phase-out timeline
- Energy performance: kWh/sqm benchmarking, renewable % by store, efficiency trend
- Recommended actions: Top 10 stores for energy efficiency investment, refrigerant retrofit candidates

### 6.2 Template 2: Supply Chain Report

**File**: `templates/supply_chain_report.py`
**Outputs**: Scope 3 breakdown by category, hotspot analysis, supplier scorecards.

**Key Sections**:
- Scope 3 overview: Waterfall chart of all 15 categories with retail-specific annotations
- Category 1 deep-dive: Top 20 product categories by emissions, data quality per category
- Supplier scorecards: Top 50 suppliers with emissions, DQ score, SBTi status, improvement trend
- Data quality improvement plan: Pathway from Level 4/5 to Level 1/2 for material categories
- EUDR commodity overlay: Scope 3 emissions from EUDR-relevant commodities highlighted
- Engagement metrics: % of suppliers contacted, response rate, data provision rate

### 6.3 Template 3: Packaging Compliance Report

**File**: `templates/packaging_compliance_report.py`
**Outputs**: PPWR compliance dashboard with traffic-light indicators.

**Key Sections**:
- Recycled content tracker: Current % vs 2030/2040 targets per material, gap to target
- EPR fee summary: Total fees by Member State, eco-modulation impact, optimization recommendations
- Labeling compliance: % of packaging labels meeting PPWR requirements, non-conformance list
- Reuse targets: Progress toward 2030/2040 reuse % for transport and e-commerce packaging
- Packaging reduction: Weight reduction trajectory vs baseline year
- E-commerce focus: Void ratio analysis, over-packaging incidents, right-sizing recommendations

### 6.4 Template 4: Product Sustainability Report

**File**: `templates/product_sustainability_report.py`
**Outputs**: DPP readiness assessment, PEF summary, green claims compliance.

**Key Sections**:
- DPP readiness: % of applicable products with complete DPP data, gaps by data field
- PEF results: Environmental footprint scores for assessed product categories
- Green claims audit: Number of active claims, substantiated vs unsubstantiated, remediation status
- Textile dashboard: Microplastic scores, durability ratings, fiber composition accuracy
- Regulatory timeline: Upcoming DPP deadlines by product category (textiles 2027, electronics 2027-28, furniture 2028)
- Certification inventory: Active third-party certifications and validity dates

### 6.5 Template 5: Food Waste Report

**File**: `templates/food_waste_report.py`
**Outputs**: Food waste reduction progress dashboard.

**Key Sections**:
- Headline metric: Total food waste (tonnes), waste rate (%), reduction vs baseline year
- Department breakdown: Bakery, produce, dairy, meat, deli, frozen, ambient waste rates with benchmarks
- Waste hierarchy: Sankey diagram data showing prevention/redistribution/recycling/recovery/disposal split
- Redistribution impact: Donations (kg), meals equivalent, charitable partner summary
- Financial impact: Cost of waste, savings from prevention, tax benefits from donation
- 2030 trajectory: On-track/off-track indicator with remediation actions if behind target

### 6.6 Template 6: Circular Economy Report

**File**: `templates/circular_economy_report.py`
**Outputs**: EPR, take-back, and circularity metrics dashboard.

**Key Sections**:
- Portfolio MCI: Material Circularity Index score (0-1) with trend
- EPR compliance matrix: Scheme x Member State grid with status indicators
- Take-back volumes: Collection volumes for WEEE, batteries, textiles, packaging with collection rates
- Waste diversion: Total waste generated, diversion rate (% diverted from landfill), breakdown by destination
- Textile circularity: Fiber-to-fiber recycling rate, mono-material ratio, second-hand program metrics
- Improvement roadmap: Circularity targets with quarterly milestones

### 6.7 Template 7: Retail ESG Scorecard

**File**: `templates/retail_esg_scorecard.py`
**Outputs**: Executive-level KPI dashboard for board and investor reporting.

**Key Sections**:
- Traffic-light summary: 12 key KPIs with green/amber/red status
- Environmental KPIs: Scope 1+2 (tCO2e, intensity), Scope 3 (tCO2e, DQ score), renewable %, waste diversion %, packaging recycled content %, food waste rate
- Social KPIs: CSDDD due diligence coverage %, supplier audit pass rate, living wage coverage, forced labour risk score
- Governance KPIs: Regulatory compliance score, audit readiness %, ECGT claims compliance %
- Benchmark position: Peer ranking within retail sub-sector
- Trend indicators: 3-year trend for each KPI with directional arrows

### 6.8 Template 8: ESRS Retail Disclosure

**File**: `templates/esrs_retail_disclosure.py`
**Outputs**: Full ESRS chapter generation for retail-material topics.

**Key Sections**:
- E1 Climate Change: Scope 1+2+3 emissions (retail format), transition plan, SBTi targets, energy mix
- E5 Circular Economy: Resource inflows (materials), outflows (waste), circular design, EPR
- S2 Workers in Value Chain: Supply chain due diligence, CSDDD disclosure, living wage, forced labour
- S4 Consumers: Product safety, green claims substantiation, accessibility, data privacy
- G1 Business Conduct: Anti-corruption, supplier payment practices, tax transparency
- Cross-cutting: Double materiality assessment, governance, strategy, metrics and targets
- Omnibus I optimized: Only material datapoints included; reduced set per 61% reduction

---

## 7. Integration Specifications

### 7.1 Pack Orchestrator

**File**: `integrations/pack_orchestrator.py`
**Purpose**: Manage the 11-phase retail compliance pipeline from data intake through final ESRS reporting.

**Pipeline Phases**:
```
Phase 1:  INITIALIZATION        - Load config, validate preset, check agent availability
Phase 2:  DATA_INTAKE           - Ingest store data, procurement data, supplier data, waste data
Phase 3:  QUALITY_ASSURANCE     - Data quality profiling, duplicate detection, outlier flagging
Phase 4:  STORE_EMISSIONS       - Store Scope 1+2 calculation (Engine 1)
Phase 5:  SCOPE3_CALCULATION    - Retail Scope 3 all categories (Engine 2)
Phase 6:  PACKAGING             - PPWR compliance assessment (Engine 3)
Phase 7:  PRODUCT_SUSTAINABILITY - DPP/PEF/green claims (Engine 4)
Phase 8:  FOOD_WASTE            - Food waste tracking (Engine 5) [grocery preset only]
Phase 9:  SUPPLY_CHAIN_DD       - CSDDD/EUDR/forced labour (Engine 6)
Phase 10: CIRCULAR_ECONOMY      - EPR/take-back/MCI (Engine 7)
Phase 11: REPORTING             - ESRS disclosure generation, benchmarking, scorecard
```

**Orchestrator Features**:
- Phase-level enable/disable (via preset configuration)
- Retry policy: Configurable max retries (default 3), exponential backoff (1.5x)
- Provenance tracking: SHA-256 hash at every phase boundary (input hash, output hash, agent versions)
- Dry-run mode: Validate pipeline without executing calculations
- Parallel execution: Phases 4-8 can run in parallel (no inter-dependencies)
- Timeout management: Configurable per-phase timeout (default 300s)
- Progress callbacks: Real-time progress updates for UI integration

**Models**: `OrchestratorConfig`, `RetryPolicy`, `PipelinePhase`, `PhaseStatus`, `PhaseResult`, `PhaseProvenance`, `PipelineResult`

### 7.2 CSRD Pack Bridge

**File**: `integrations/csrd_pack_bridge.py`
**Purpose**: Bridge to PACK-001 (CSRD Starter), PACK-002 (CSRD Professional), PACK-003 (CSRD Enterprise) for base CSRD reporting capabilities.

**Bridge Functions**:
- Import cross-cutting ESRS datapoints (governance, strategy, IRO)
- Reuse double materiality assessment framework
- Share common ESRS disclosure templates (ESRS 1, ESRS 2 general disclosures)
- Avoid duplication of metrics already calculated by base CSRD packs
- Version compatibility checking

### 7.3 MRV Retail Bridge

**File**: `integrations/mrv_retail_bridge.py`
**Purpose**: Bridge to all 30 MRV agents with retail-specific routing and configuration.

**Agent Routing**:
- **Scope 1**: Stationary Combustion (MRV-001) for store heating, Refrigerants (MRV-002) for F-gas, Mobile Combustion (MRV-003) for delivery fleet
- **Scope 2**: Location-Based (MRV-009), Market-Based (MRV-010), Steam/Heat (MRV-011) for district heating, Dual Reporting (MRV-013)
- **Scope 3**: Purchased Goods (MRV-014) for Cat 1 (dominant), Upstream Transport (MRV-017) for Cat 4, Business Travel (MRV-019) for Cat 6, Employee Commuting (MRV-020) for Cat 7, Downstream Transport (MRV-022) for Cat 9 (last-mile), Use of Sold Products (MRV-024) for Cat 11, End-of-Life (MRV-025) for Cat 12, Category Mapper (MRV-029) for routing, Audit Trail (MRV-030)

**Retail-Specific Configuration**:
- Refrigerant agent configured for commercial refrigeration (not industrial)
- Scope 3 Cat 1 configured for product-category emission factors (not BOM-based)
- Transport agents configured for retail logistics (hub-and-spoke distribution)

### 7.4 Data Retail Bridge

**File**: `integrations/data_retail_bridge.py`
**Purpose**: Bridge to all 20 DATA agents with retail-specific ERP integration (SAP Retail, Oracle Retail, Microsoft Dynamics 365 Commerce).

**Data Source Routing**:
- **Store energy data**: ERP energy module or utility bill upload (DATA-002 CSV/Excel, DATA-001 PDF invoice)
- **Procurement data**: ERP purchasing module (DATA-003 ERP Connector) for spend, quantities, suppliers
- **Supplier data**: Questionnaire processor (DATA-008) for CDP/custom questionnaires
- **Waste data**: Waste management system integration or manual upload
- **Product data**: PIM/MDM systems for product master, material composition
- **Quality assurance**: Data Quality Profiler (DATA-010), Duplicate Detection (DATA-011), Outlier Detection (DATA-013), Validation Rule Engine (DATA-019)

### 7.5 EUDR Retail Bridge

**File**: `integrations/eudr_retail_bridge.py`
**Purpose**: Bridge to EUDR agents (AGENT-EUDR-001 through 040) for commodity tracing of retail products containing EUDR-relevant commodities.

**Retail-Specific EUDR Context**:
- Private label products: Retailer is the "operator" placing product on EU market
- Third-party brands: Retailer may be "trader" (>SME threshold) with due diligence obligations
- Ingredients: EUDR applies to derived products (e.g., palm oil in chocolate, soy in animal feed for meat)
- Volume: Large grocery retailers may have 5,000-20,000 SKUs with EUDR-relevant ingredients

**Bridge Functions**:
- Route commodity identification to EUDR traceability agents (001-015)
- Connect risk assessment (EUDR 016-020) for country/commodity risk
- Generate due diligence statements via EUDR due diligence agents (021-040)
- Aggregate DDS status across product range

### 7.6 Circular Economy Bridge

**File**: `integrations/circular_economy_bridge.py`
**Purpose**: Bridge to EPR scheme registries, waste management systems, and MRV waste agents.

**External Connections**:
- National EPR scheme registries (data exchange formats per MS)
- WEEE compliance schemes (e.g., EAR Foundation/DE, Ecosystem/FR)
- Battery compliance schemes
- Textile collection partners
- Waste management contractors (weight tickets, processing certificates)
- MRV Waste Treatment Agent (MRV-007) for waste emissions calculation

### 7.7 Supply Chain Bridge

**File**: `integrations/supply_chain_bridge.py`
**Purpose**: Bridge to CSDDD agents, forced labour risk databases, and supplier engagement platforms.

**External Connections**:
- CDP Supply Chain program (supplier questionnaire data)
- EcoVadis ratings (supplier sustainability scores)
- SEDEX/SMETA (ethical trade audit data)
- amfori BSCI (social compliance)
- Walk Free Global Slavery Index (country-level forced labour risk)
- ILO forced labour indicators database
- WBCSD PACT (Partnership for Carbon Transparency) data exchange

### 7.8 Taxonomy Bridge

**File**: `integrations/taxonomy_bridge.py`
**Purpose**: EU Taxonomy alignment assessment for retail economic activities.

**Relevant Taxonomy Activities**:
- 5.1: Construction, extension and operation of water collection (if applicable)
- 5.5: Collection and transport of non-hazardous waste (take-back programs)
- 6.5: Transport by motorbikes, passenger cars, light commercial vehicles (delivery fleet)
- 6.6: Freight transport services by road (logistics)
- 7.1-7.7: Building renovation/construction (store construction/renovation)
- 8.1: Data processing, hosting (e-commerce infrastructure)
- Additional: Taxonomy delegated act updates for retail-specific activities

**Bridge Functions**:
- Classify CapEx/OpEx/Revenue by Taxonomy-eligible activities
- Assess substantial contribution to climate mitigation/adaptation
- Check DNSH (Do No Significant Harm) criteria
- Verify minimum social safeguards
- Calculate Green Asset Ratio equivalents for non-financial companies

### 7.9 Health Check

**File**: `integrations/health_check.py`
**Purpose**: 22-category system health verification covering all pack components.

**Health Check Categories**:
1. Pack manifest integrity
2. Configuration system
3. Preset loading (all 6 presets)
4. Engine 1-8 availability and initialization
5. Workflow 1-8 readiness
6. Template 1-8 rendering
7. MRV agent connectivity (30 agents)
8. DATA agent connectivity (20 agents)
9. FOUND agent connectivity (10 agents)
10. EUDR agent connectivity (40 agents)
11. CSRD pack bridge status
12. Circular economy bridge status
13. Supply chain bridge status
14. Taxonomy bridge status
15. Database connectivity
16. Configuration validation
17. Emission factor database availability
18. Product category emission factor availability
19. EPR scheme data availability
20. Country risk factor database availability
21. Benchmark database availability
22. Overall system health score (0-100)

### 7.10 Setup Wizard

**File**: `integrations/setup_wizard.py`
**Purpose**: 8-step retail-specific guided configuration wizard.

**Setup Steps**:
1. **Company Profile**: Legal entity, NACE code (G46/G47.x), number of stores, countries of operation, annual revenue, employee count
2. **Sub-Sector Selection**: Grocery, apparel, electronics, general, online, mixed; auto-loads preset
3. **Regulation Mapping**: Based on profile, identify applicable regulations with dates
4. **Store Portfolio**: Import store list with addresses, types (hypermarket/supermarket/convenience/etc.), selling areas, climate zones
5. **Data Source Connection**: Connect ERP, energy management, waste management, procurement systems
6. **Baseline Configuration**: Set reporting year, baseline year, currency, units, boundary (operational vs financial control)
7. **Target Setting**: Import or set SBTi targets, food waste reduction targets, packaging targets, CSDDD rollout plan
8. **Go-Live Validation**: Run health check, validate demo data, confirm all engines operational

---

## 8. Configuration System

### 8.1 Pack Configuration

**File**: `config/pack_config.py`

**Configuration Hierarchy** (later overrides earlier):
1. Base `pack.yaml` manifest (defaults)
2. Preset YAML (grocery_retail / apparel_retail / electronics_retail / general_retail / online_retail / sme_retailer)
3. Environment overrides (CSRD_RETAIL_PACK_* environment variables)
4. Explicit runtime overrides

**Top-Level Configuration Model** (`PackConfig`):
```python
class RetailSubSector(str, Enum):
    GROCERY = "grocery"
    APPAREL = "apparel"
    ELECTRONICS = "electronics"
    GENERAL = "general"       # Department stores
    ONLINE = "online"         # E-commerce pure-play
    WHOLESALE = "wholesale"   # B2B wholesale (NACE G46)
    MIXED = "mixed"           # Multi-category retailer

class PackConfig(BaseModel):
    pack_id: str = "PACK-014-csrd-retail"
    version: str = "1.0.0"
    retail_sub_sector: RetailSubSector
    reporting_year: int
    baseline_year: int
    currency: str = "EUR"

    # Store portfolio
    store_count: int
    total_selling_area_sqm: float
    countries_of_operation: List[str]

    # Engine configurations
    store_emissions: StoreEmissionsConfig
    scope3: RetailScope3Config
    packaging: PackagingComplianceConfig
    product_sustainability: ProductSustainabilityConfig
    food_waste: FoodWasteConfig          # Enabled only for grocery/mixed
    supply_chain_dd: SupplyChainDDConfig
    circular_economy: CircularEconomyConfig
    benchmark: BenchmarkConfig

    # Feature flags
    eudr_enabled: bool = True
    csddd_enabled: bool = True
    ecgt_enabled: bool = True
    dpp_enabled: bool = True
    food_waste_enabled: bool = False     # Auto-set by preset
    forced_labour_enabled: bool = True
```

### 8.2 Preset Specifications

#### 8.2.1 Grocery Retail Preset (`grocery_retail.yaml`)

**Target**: Tesco, Carrefour, Lidl, Aldi, REWE, Ahold Delhaize, Migros, Coop

**Configuration Overrides**:
- `food_waste_enabled: true` (Engine 5 active)
- `store_emissions.refrigerant_tracking: detailed` (F-gas is major Scope 1 source)
- `scope3.priority_categories: [1, 4, 9, 12]` (purchased food, transport, end-of-life)
- `packaging.ppwr_focus: [PET, HDPE, PP, glass, paper_board]`
- `eudr_enabled: true` (palm oil, soy, cocoa, coffee, cattle in food products)
- `circular_economy.epr_schemes: [packaging]` (packaging is primary EPR)
- `benchmark.peer_group: grocery`
- `product_sustainability.dpp_categories: []` (food DPP not yet required)

#### 8.2.2 Apparel Retail Preset (`apparel_retail.yaml`)

**Target**: H&M, Zara/Inditex, Primark, C&A, Zalando, ASOS

**Configuration Overrides**:
- `food_waste_enabled: false`
- `store_emissions.refrigerant_tracking: basic` (minimal refrigeration)
- `scope3.priority_categories: [1, 4, 9, 11, 12]` (garments, transport, use-phase washing, end-of-life textile waste)
- `packaging.ppwr_focus: [PP, LDPE, paper_board]` (shipping bags, boxes)
- `eudr_enabled: true` (cotton/rubber supply chains)
- `csddd_enabled: true` (garment worker due diligence is critical)
- `circular_economy.epr_schemes: [packaging, textiles]`
- `product_sustainability.textile_microplastics: true`
- `product_sustainability.dpp_categories: [textiles]` (2027 deadline)
- `benchmark.peer_group: apparel`
- `forced_labour_enabled: true` (cotton, garment manufacturing high-risk)

#### 8.2.3 Electronics Retail Preset (`electronics_retail.yaml`)

**Target**: MediaMarkt, Currys, Fnac Darty, Elkjop

**Configuration Overrides**:
- `food_waste_enabled: false`
- `store_emissions.refrigerant_tracking: basic`
- `scope3.priority_categories: [1, 4, 9, 11, 12]` (electronics, transport, use-phase energy, WEEE)
- `packaging.ppwr_focus: [EPS, paper_board, LDPE]`
- `circular_economy.epr_schemes: [packaging, weee, batteries]`
- `circular_economy.weee_collection: true`
- `product_sustainability.dpp_categories: [electronics]` (2027-2028)
- `product_sustainability.repairability_index: true`
- `benchmark.peer_group: electronics`
- `scope3.use_phase_calculation: detailed` (energy consumption of sold electronics is material)

#### 8.2.4 General Retail Preset (`general_retail.yaml`)

**Target**: El Corte Ingles, Galeries Lafayette, John Lewis, department stores

**Configuration Overrides**:
- `food_waste_enabled: false` (unless food hall present)
- `scope3.priority_categories: [1, 4, 9, 12]`
- All EPR schemes enabled at basic level
- All product sustainability features enabled at basic level
- `benchmark.peer_group: general`

#### 8.2.5 Online Retail Preset (`online_retail.yaml`)

**Target**: Amazon (EU), Zalando, ASOS, AboutYou, Coolblue

**Configuration Overrides**:
- `store_emissions.store_types: [fulfillment_center, office, data_center]` (no physical stores)
- `scope3.priority_categories: [1, 4, 9, 12]` (purchased goods, upstream transport, last-mile delivery, returns/end-of-life)
- `scope3.last_mile_detailed: true` (last-mile delivery is major differentiator)
- `packaging.e_commerce_focus: true` (void ratio, right-sizing critical)
- `packaging.returns_packaging: true` (reverse logistics packaging)
- `circular_economy.returns_tracking: true` (product returns rate and destination)
- `benchmark.peer_group: online`

#### 8.2.6 SME Retailer Preset (`sme_retailer.yaml`)

**Target**: Retailers approaching Omnibus I threshold, voluntary reporters

**Configuration Overrides**:
- Simplified Scope 1+2 (spend-based energy estimation allowed)
- Scope 3 limited to Cat 1 (spend-based only)
- EPR compliance only (no voluntary circular economy metrics)
- No DPP (below threshold)
- Basic ECGT claims check
- Guided data entry with industry-average defaults

---

## 9. Agent Dependencies

### 9.1 Summary

| Agent Layer | Count | Usage in PACK-014 |
|-------------|-------|-------------------|
| AGENT-MRV | 30 | All 30 agents for Scope 1 (001-008), Scope 2 (009-013), Scope 3 (014-030) |
| AGENT-DATA | 20 | All 20 agents for data intake (001-007) and quality (008-019), geo (020) |
| AGENT-FOUND | 10 | All 10 agents for orchestration, schema, units, assumptions, citations, access, registry, reproducibility, QA, observability |
| AGENT-EUDR | 15 bridged | Traceability agents (001-015) for commodity origin verification |
| Other bridged | -- | CSDDD, taxonomy, climate risk connectors |
| **Total** | **75** | **60 direct + 15 bridged** |

### 9.2 MRV Agent Mapping (30 agents)

| MRV Agent | Retail Application | Priority |
|-----------|-------------------|----------|
| MRV-001 Stationary Combustion | Store heating (gas, oil) | High |
| MRV-002 Refrigerants & F-Gas | Commercial refrigeration leakage | Critical (grocery) |
| MRV-003 Mobile Combustion | Delivery fleet vehicles | Medium |
| MRV-004 Process Emissions | Not typically applicable | Low |
| MRV-005 Fugitive Emissions | Refrigerant leaks | Medium |
| MRV-006 Land Use Emissions | Not typically applicable | Low |
| MRV-007 Waste Treatment | Store/warehouse waste | Medium |
| MRV-008 Agricultural Emissions | Not typically applicable (unless farm-to-shelf) | Low |
| MRV-009 Scope 2 Location-Based | Store electricity (grid factor) | Critical |
| MRV-010 Scope 2 Market-Based | Green tariffs, PPAs, RECs/GOs | Critical |
| MRV-011 Steam/Heat Purchase | District heating for stores | Medium |
| MRV-012 Cooling Purchase | District cooling (rare) | Low |
| MRV-013 Dual Reporting Reconciliation | Location vs market Scope 2 | High |
| MRV-014 Purchased Goods (Cat 1) | Dominant: 50-75% of total emissions | Critical |
| MRV-015 Capital Goods (Cat 2) | Store fit-out, equipment | Low |
| MRV-016 Fuel & Energy (Cat 3) | T&D losses, upstream fuel | Medium |
| MRV-017 Upstream Transport (Cat 4) | Supplier to DC, DC to store | High |
| MRV-018 Waste Generated (Cat 5) | Store operational waste | Medium |
| MRV-019 Business Travel (Cat 6) | Corporate travel | Low |
| MRV-020 Employee Commuting (Cat 7) | Large retail workforce | Medium |
| MRV-021 Upstream Leased (Cat 8) | Leased stores (if not Scope 1+2) | Medium |
| MRV-022 Downstream Transport (Cat 9) | Last-mile delivery, customer travel | High (online) |
| MRV-023 Processing of Sold Products (Cat 10) | Not typically applicable | Low |
| MRV-024 Use of Sold Products (Cat 11) | Electronics energy use | High (electronics) |
| MRV-025 End-of-Life (Cat 12) | Product/packaging waste at consumer | High |
| MRV-026 Downstream Leased (Cat 13) | Subleased retail space | Low |
| MRV-027 Franchises (Cat 14) | Franchise store emissions | High (if franchised) |
| MRV-028 Investments (Cat 15) | Not typically applicable | Low |
| MRV-029 Category Mapper | Route products to categories | Critical |
| MRV-030 Audit Trail & Lineage | Provenance for all calculations | Critical |

### 9.3 DATA Agent Mapping (20 agents)

| DATA Agent | Retail Application |
|------------|-------------------|
| DATA-001 PDF Extractor | Utility bills, supplier certificates, audit reports |
| DATA-002 Excel/CSV Normalizer | Store energy data, waste reports, procurement data |
| DATA-003 ERP Connector | SAP Retail, Oracle Retail, Dynamics 365 Commerce |
| DATA-004 API Gateway | External data feeds (grid factors, emission factors) |
| DATA-005 EUDR Traceability | Commodity origin data for EUDR compliance |
| DATA-006 GIS/Mapping | Store locations, supplier geography |
| DATA-007 Satellite Connector | Deforestation monitoring for EUDR |
| DATA-008 Supplier Questionnaire | CDP, custom sustainability questionnaires |
| DATA-009 Spend Categorizer | Procurement spend to Scope 3 categories |
| DATA-010 Data Quality Profiler | Completeness, accuracy, timeliness scoring |
| DATA-011 Duplicate Detection | Deduplicate store/supplier records |
| DATA-012 Missing Value Imputer | Fill gaps in energy/waste data |
| DATA-013 Outlier Detection | Flag anomalous energy readings, waste spikes |
| DATA-014 Time Series Gap Filler | Monthly energy data gap filling |
| DATA-015 Cross-Source Reconciliation | Reconcile ERP vs utility vs meter data |
| DATA-016 Data Freshness Monitor | Alert on stale data sources |
| DATA-017 Schema Migration | Handle data format changes across years |
| DATA-018 Data Lineage Tracker | Track data from source to report |
| DATA-019 Validation Rule Engine | Retail-specific validation rules |
| DATA-020 Climate Hazard Connector | Physical climate risk for store locations |

---

## 10. Testing Strategy

### 10.1 Test Files (18 files, 600+ tests target)

| Test File | Scope | Target Tests |
|-----------|-------|-------------|
| conftest.py | Shared fixtures: retail company, store portfolio, product catalog, supplier data | N/A (fixtures) |
| test_manifest.py | Pack YAML validation, version, structure | 60+ |
| test_config.py | Config system, preset loading, merge hierarchy, validation | 45+ |
| test_demo.py | Demo data smoke tests, demo config loading | 60+ |
| test_store_emissions.py | Engine 1: store Scope 1+2, F-gas, consolidation | 40+ |
| test_retail_scope3.py | Engine 2: all 15 categories, 4 calculation methods, DQ scoring | 50+ |
| test_packaging_compliance.py | Engine 3: PPWR recycled content, EPR, labeling, reuse | 40+ |
| test_product_sustainability.py | Engine 4: DPP, PEF, green claims, textiles | 40+ |
| test_food_waste.py | Engine 5: waste measurement, hierarchy, reduction tracking | 35+ |
| test_supply_chain_dd.py | Engine 6: CSDDD, EUDR, forced labour, supplier tiers | 45+ |
| test_circular_economy.py | Engine 7: EPR, take-back, MCI, end-of-life | 40+ |
| test_retail_benchmark.py | Engine 8: KPIs, peer comparison, SBTi alignment | 35+ |
| test_workflows.py | All 8 workflows end-to-end with demo data | 30+ |
| test_templates.py | All 8 templates + template registry | 28+ |
| test_integrations.py | All 10 integrations | 20+ |
| test_e2e.py | End-to-end flows (5 scenarios) | 15+ |
| test_agent_integration.py | Agent wiring verification (MRV, DATA, FOUND, EUDR) | 15+ |

### 10.2 Key Test Scenarios

**Scenario 1: Grocery Retailer Full Assessment**
- Company: GreenMart Europe GmbH (synthetic), 800 stores across 5 EU countries
- Flow: Load grocery preset -> Import store energy data -> Calculate Scope 1 (refrigerant focus) -> Calculate Scope 2 (dual reporting) -> Calculate Scope 3 Cat 1 (food products) -> Track food waste (5 departments) -> PPWR packaging compliance -> EUDR for palm oil/soy/cocoa -> Benchmark against grocery peers -> Generate ESRS E1 + E5 + S2 disclosure
- Expected outputs: Portfolio emissions report, food waste dashboard, PPWR compliance, EUDR status, ESRS chapters

**Scenario 2: Fashion Retailer Supply Chain Due Diligence**
- Company: EcoThread Retail SE (synthetic), 200 stores + online, sourcing from 12 countries
- Flow: Load apparel preset -> Map supplier tiers (1-4: brand -> factory -> spinner -> farm) -> Scope 3 Cat 1 (garments by fiber) -> CSDDD assessment (garment workers) -> Forced labour risk scoring (cotton origins) -> EUDR (cotton, rubber) -> Textile EPR compliance -> DPP for textiles (2027) -> ECGT green claims audit -> Benchmark against apparel peers
- Expected outputs: Supply chain risk map, forced labour heat map, textile DPP data, claims audit, ESRS S2 disclosure

**Scenario 3: Electronics Retailer Circular Economy**
- Company: TechStore Europa NV (synthetic), 150 stores, WEEE collection program
- Flow: Load electronics preset -> Store Scope 1+2 -> Scope 3 Cat 1 (electronics) + Cat 11 (use-phase energy) -> WEEE take-back volumes -> Battery EPR -> DPP for electronics (2027-28) -> Repairability scoring -> PPWR packaging -> Benchmark
- Expected outputs: Use-phase emissions, WEEE collection report, repairability dashboard, DPP readiness

**Scenario 4: Online Retailer Last-Mile + Packaging**
- Company: GreenShop Online BV (synthetic), 3 fulfillment centers, 15 EU markets
- Flow: Load online preset -> FC emissions (Scope 1+2) -> Scope 3 Cat 9 last-mile (van/bike/drone) + Cat 1 -> E-commerce packaging (void ratio, returns) -> PPWR compliance -> Customer returns tracking -> Benchmark vs online peers
- Expected outputs: Last-mile emissions by mode, packaging optimization report, returns circularity metrics

**Scenario 5: SME Retailer Simplified Compliance**
- Company: BioMarkt KG (synthetic), 15 stores, approaching Omnibus I threshold
- Flow: Load SME preset -> Simplified Scope 1+2 (area-based estimation) -> Spend-based Scope 3 Cat 1 -> Basic ECGT claims check -> Basic EPR compliance -> Simplified ESRS disclosure
- Expected outputs: Simplified emissions summary, compliance checklist, ESRS-light disclosure

### 10.3 Test Infrastructure

- **Dynamic loading**: All tests use `importlib` dynamic loading (no package installation required)
- **Fixtures**: Shared conftest.py provides synthetic retail company data, store portfolios, product catalogs, supplier databases, waste records
- **Determinism**: All tests verify SHA-256 provenance hashes for identical inputs
- **Coverage target**: 85%+ line coverage across all engines, workflows, templates
- **CI integration**: Tests run in GitHub Actions via INFRA-007 CI/CD pipeline

---

## 11. Deployment

### 11.1 Pack Registration

PACK-014 registers in the GreenLang Solution Pack registry with the following metadata:
- Pack ID: `PACK-014-csrd-retail`
- Category: `eu-compliance`
- Sector: `retail-consumer-goods`
- NACE: `G46, G47`
- Tier: `sector-specific`
- Dependencies: `PACK-001, PACK-002, PACK-003` (CSRD base)
- Optional bridges: `PACK-006/007` (EUDR), `PACK-008` (Taxonomy)

### 11.2 Database Migrations

No new database tables required for PACK-014. The pack uses existing tables from:
- MRV agent tables (V051-V081)
- DATA agent tables (V031-V050)
- FOUND agent tables (V021-V030)
- EUDR agent tables (V089-V128)
- APP tables (V082-V088)
- Pack configuration stored in `pack_configurations` table (existing)

If retail-specific reference data tables are needed (e.g., product category emission factors, EPR scheme rules, PPWR targets), they will be added as V129+ migrations.

### 11.3 Infrastructure Requirements

| Resource | Requirement | Notes |
|----------|-------------|-------|
| Compute | 2 vCPU, 4 GB RAM (per pack instance) | Scales horizontally via K8s |
| Storage | 500 MB for emission factor databases + EPR reference data | S3-backed |
| Database | Existing PostgreSQL + TimescaleDB | No additional DB needed |
| Cache | Existing Redis cluster | For benchmark data caching |
| Network | Outbound HTTPS for external data feeds | Grid factors, EPR registries |

### 11.4 Deployment Configuration

```yaml
# Kubernetes deployment snippet
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pack-014-csrd-retail
  labels:
    app: greenlang
    component: solution-pack
    pack: csrd-retail
spec:
  replicas: 2
  selector:
    matchLabels:
      pack: csrd-retail
  template:
    spec:
      containers:
      - name: pack-014
        image: greenlang/pack-014-csrd-retail:1.0.0
        resources:
          requests:
            cpu: "1"
            memory: 2Gi
          limits:
            cpu: "2"
            memory: 4Gi
        env:
        - name: CSRD_RETAIL_PACK_LOG_LEVEL
          value: "INFO"
        - name: CSRD_RETAIL_PACK_PROVENANCE
          value: "true"
```

---

## 12. Timeline

### Phase 1: Foundation (Weeks 1-4)

| Week | Deliverable | Owner |
|------|------------|-------|
| 1 | PRD finalization, architecture review, pack scaffold | Product + Arch |
| 2 | pack_config.py, all 6 presets, demo_config.yaml | Config engineer |
| 3 | Engine 1 (store emissions), Engine 8 (benchmark) | Engine engineer |
| 4 | Engine 2 (Scope 3), Engine 6 (supply chain DD) | Engine engineer |

**Milestone**: Core engines operational with demo data.

### Phase 2: Product Regulation Engines (Weeks 5-8)

| Week | Deliverable | Owner |
|------|------------|-------|
| 5 | Engine 3 (packaging/PPWR), Engine 4 (product sustainability/DPP) | Engine engineer |
| 6 | Engine 5 (food waste), Engine 7 (circular economy/EPR) | Engine engineer |
| 7 | All 8 workflows implemented | Workflow engineer |
| 8 | All 8 templates implemented | Template engineer |

**Milestone**: All engines, workflows, and templates complete.

### Phase 3: Integrations and Testing (Weeks 9-12)

| Week | Deliverable | Owner |
|------|------------|-------|
| 9 | Pack orchestrator, CSRD bridge, MRV bridge, DATA bridge | Integration engineer |
| 10 | EUDR bridge, circular economy bridge, supply chain bridge, taxonomy bridge | Integration engineer |
| 11 | Health check, setup wizard, all unit tests (600+ target) | QA + Integration |
| 12 | E2E testing, performance testing, security review, documentation | QA + DevOps |

**Milestone**: Pack launch-ready with 600+ tests, 100% pass rate.

### Phase 4: Beta and GA (Weeks 13-16)

| Week | Deliverable | Owner |
|------|------------|-------|
| 13 | Beta deployment to 5 pilot customers (1 per sub-sector) | Product + CS |
| 14 | Beta feedback integration, bug fixes | Engineering |
| 15 | GA readiness review, final documentation | Product + QA |
| 16 | General Availability release | All teams |

**Milestone**: GA release.

---

## 13. Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PPWR delegated acts change recycled content targets | Medium | High | Modular target configuration; targets stored as config not code; monitor EU Official Journal |
| ESPR DPP product categories or data fields change before 2027 | High | Medium | DPP data model designed as extensible schema; field registry updateable without code changes |
| ECGT interpretation varies across Member States | Medium | Medium | Conservative claims checking (flag all borderline claims); country-specific rule overrides |
| Scope 3 Cat 1 emission factors unavailable for niche product categories | Medium | High | Fallback to parent category average; spend-based EEIO as last resort; DQ score reflects method |
| EUDR implementation delayed again (beyond Dec 2026) | Medium | Low | EUDR bridge is modular; can be deactivated without affecting other engines |
| Omnibus II further reduces CSRD requirements | Medium | Medium | Materiality-based engine activation; engines can be disabled via preset without code changes |
| Large store portfolio performance issues (>5,000 stores) | Low | High | Parallel phase execution in orchestrator; horizontal scaling; batch processing for store calculations |
| EPR scheme rules differ significantly across 27 Member States | High | Medium | Country-specific EPR rule database; start with top 5 markets (DE, FR, IT, ES, NL), expand iteratively |
| Forced Labour Regulation enforcement mechanism unclear | Medium | Low | Conservative approach: implement risk assessment regardless; product-level tracking ready |
| Competitor releases retail-specific CSRD tool first | Medium | High | Speed to market with 70-75% platform reuse; unique differentiation through multi-regulation convergence |

---

## 14. Appendices

### Appendix A: Retail Sub-Sectors Covered

| NACE Code | Sub-Sector | Key Engines | Dominant Scope 3 | Primary Regulations |
|-----------|-----------|-------------|-------------------|-------------------|
| G46 | Wholesale trade | Scope 3, Supply Chain DD | Cat 1 (60-80%) | CSDDD, EUDR |
| G47.1 | Food/grocery retail | Store Emissions, Food Waste, Scope 3, Packaging | Cat 1 food (50-75%) | PPWR, EUDR, F-Gas, Food Waste |
| G47.2 | General food/tobacco/beverage | Store Emissions, Food Waste | Cat 1 (50-70%) | PPWR, EUDR |
| G47.4 | ICT/electronics retail | Circular Economy (WEEE), Product Sust (DPP) | Cat 1 + Cat 11 (60-80%) | ESPR/DPP, WEEE, Battery Reg |
| G47.5 | Household/furniture retail | Circular Economy, Product Sust | Cat 1 (40-60%) | ESPR/DPP (furniture 2028) |
| G47.7 | Apparel/fashion retail | Supply Chain DD, Circular Economy (textile EPR) | Cat 1 (50-70%) | EU Textile Strategy, CSDDD, EUDR, ECGT |
| G47.8 | Market stalls | Simplified (SME preset) | Cat 1 (60-80%) | Basic EPR |
| G47.9 | Non-store / online retail | Packaging (e-commerce), Scope 3 (last-mile) | Cat 1 + Cat 9 (60-80%) | PPWR (e-commerce), ECGT |

### Appendix B: Glossary

| Term | Definition |
|------|-----------|
| CBAM | Carbon Border Adjustment Mechanism |
| CSDDD | Corporate Sustainability Due Diligence Directive |
| CSRD | Corporate Sustainability Reporting Directive |
| DPP | Digital Product Passport |
| DQ | Data Quality |
| ECGT | Empowering Consumers for the Green Transition (Directive 2024/825) |
| EED | Energy Efficiency Directive |
| EEIO | Environmentally Extended Input-Output (economic modeling for Scope 3) |
| EPR | Extended Producer Responsibility |
| ESPR | Ecodesign for Sustainable Products Regulation |
| ESRS | European Sustainability Reporting Standards |
| EUDR | EU Deforestation Regulation |
| GHG | Greenhouse Gas |
| GO | Guarantee of Origin (renewable energy certificate in EU) |
| GWP | Global Warming Potential |
| MCI | Material Circularity Index |
| MRV | Measurement, Reporting, and Verification |
| MS | Member State (EU) |
| NACE | Statistical classification of economic activities in the European Community |
| NCV | Net Calorific Value |
| PACT | Partnership for Carbon Transparency (WBCSD) |
| PEF | Product Environmental Footprint |
| PEFCR | Product Environmental Footprint Category Rules |
| PIM | Product Information Management |
| PPA | Power Purchase Agreement |
| PPWR | Packaging and Packaging Waste Regulation |
| REC | Renewable Energy Certificate |
| SBTi | Science Based Targets initiative |
| FLAG | Forest, Land and Agriculture (SBTi guidance) |
| SKU | Stock Keeping Unit |
| WEEE | Waste Electrical and Electronic Equipment |

### Appendix C: Regulatory Timeline for Retail

| Date | Regulation | Event | Retail Impact |
|------|-----------|-------|---------------|
| 2026-08-01 | PPWR | Harmonized labeling requirements take effect | All packaging must meet labeling standards |
| 2026-09-27 | ECGT | Anti-greenwashing directive transposition deadline | Generic green claims prohibited |
| 2026-12-30 | EUDR | Full application (operators and large traders) | EUDR commodity DDS required |
| 2027-H1 | ESPR | Textiles DPP delegated act expected | Fashion retailers prepare DPP systems |
| 2027-H2 | ESPR | Electronics DPP delegated act expected | Electronics retailers prepare DPP |
| 2027-12-14 | Forced Labour | Application date | Product-level risk assessment required |
| 2028-H1 | ESPR | Furniture DPP delegated act expected | Furniture retailers prepare DPP |
| 2028-07-26 | CSDDD | Phase 1: >5,000 employees, >EUR 1.5B turnover | Large retailers begin supply chain DD |
| 2028-mid | EU Textile Strategy | Mandatory textile EPR expected | Apparel retailers register for textile EPR |
| 2029-07-26 | CSDDD | Phase 2: >3,000 employees, >EUR 900M turnover | Mid-size retailers begin DD |
| 2030-01-01 | PPWR | 30% recycled PET target; packaging reduction -5% vs 2018 | Packaging material composition change |
| 2030-01-01 | Food Waste | Binding 30% reduction target | Grocery retailers must achieve reduction |
| 2030-01-01 | EU Textile Strategy | Microplastic 30% reduction target | Textile composition changes |
| 2031-07-26 | CSDDD | Phase 3: >1,000 employees, >EUR 450M turnover | All CSRD-scope retailers in CSDDD |
| 2035-01-01 | PPWR | Packaging reduction -10% vs 2018 | Further packaging optimization |
| 2040-01-01 | PPWR | 50-65% recycled content targets | Major material composition shift |

### Appendix D: Comparison with PACK-013 (Manufacturing)

| Dimension | PACK-013 Manufacturing | PACK-014 Retail |
|-----------|----------------------|-----------------|
| NACE codes | C10-C33 | G46-G47 |
| Location unit | Factory/facility | Store/fulfillment center |
| Location count | 1-50 typical | 100-5,000+ typical |
| Scope 1 driver | Process emissions | Refrigerant leakage + heating |
| Scope 2 driver | Industrial electricity | Store electricity |
| Scope 3 share | 30-60% | 80-95% |
| Scope 3 method | BOM-based | Product category-based |
| Key regulation | EU ETS, IED/BAT | PPWR, ECGT, EUDR |
| Circular economy | Industrial waste, recycled content | EPR schemes (packaging/WEEE/textile), take-back |
| Supply chain | Tier 1-2, BOM components | Tier 1-4+, product categories |
| Product regulation | PCF/DPP (industrial) | DPP + PEF + green claims + packaging |
| Water/pollution | Industrial wastewater, IED | Not typically material |
| BAT compliance | BREF/BAT-AEL critical | Not applicable |
| EU ETS | Direct coverage (Scope 1) | Not typically covered |
| Food waste | Minor (food manufacturing only) | Critical (grocery retailers) |
| Worker rights | Factory workers (S1) | Value chain workers (S2), consumers (S4) |
| Consumer facing | B2B mostly | B2C direct: ECGT, product safety |
| Benchmark KPIs | tCO2e/tonne, MJ/tonne, m3/tonne | tCO2e/sqm, tCO2e/EUR revenue, waste rate % |

### Appendix E: References

1. CSRD -- Directive (EU) 2022/2464 of the European Parliament and of the Council
2. ESRS Set 1 -- Commission Delegated Regulation (EU) 2023/2772
3. Omnibus I -- Directive (EU) 2026/470 (published OJ L, 2026-02-26)
4. PPWR -- Regulation (EU) 2025/40 on packaging and packaging waste
5. ECGT -- Directive (EU) 2024/825 on empowering consumers for the green transition
6. EUDR -- Regulation (EU) 2023/1115 on deforestation-free products
7. CSDDD -- Directive (EU) 2024/1760 on corporate sustainability due diligence
8. ESPR -- Regulation (EU) 2024/1781 on ecodesign for sustainable products
9. F-Gas Regulation -- Regulation (EU) 2024/573 on fluorinated greenhouse gases
10. Forced Labour Regulation -- Regulation (EU) 2024/3015
11. EED -- Directive (EU) 2023/1791 on energy efficiency
12. EC Decision 2019/1597 -- Food waste measurement methodology
13. GHG Protocol -- Corporate Standard (revised), Scope 3 Standard
14. SBTi -- Corporate Net-Zero Standard v1.1, FLAG Guidance
15. PCAF -- Global GHG Accounting and Reporting Standard, 3rd Edition (Dec 2025)
16. Ellen MacArthur Foundation -- Material Circularity Indicator methodology
17. ILO -- Indicators of Forced Labour (2012)
18. IPCC -- 2006 Guidelines for National Greenhouse Gas Inventories (2019 Refinement)

---

## Acceptance Criteria

1. All 8 engines implement deterministic calculations with SHA-256 provenance hashing
2. All Pydantic v2 models with field_validator/model_validator (NO `from __future__ import annotations`)
3. All 6 presets load and validate without errors
4. All 8 workflows complete end-to-end with demo data
5. All 8 templates generate valid output
6. All 10 integrations pass health check
7. 600+ unit tests, 100% pass rate
8. Cross-pack bridges verify connectivity to PACK-001/002/003 (CSRD base)
9. Retail-specific KPIs: emissions/sqm, emissions/revenue, Scope 3 DQ score, food waste rate, packaging recycled content %, EPR compliance status, CSDDD coverage %, ECGT claims compliance %
10. Demo mode: GreenMart Europe GmbH synthetic retail company with 800 stores across 5 EU countries, grocery sub-sector, full pipeline execution

---

## Non-Functional Requirements

- **Performance**: All engines complete in <5s for single store, <30s for 500 stores, <2 minutes for 5,000 stores
- **Determinism**: Identical inputs produce identical outputs and provenance hashes across runs
- **Extensibility**: New retail sub-sectors and product categories addable via configuration without code changes
- **Auditability**: Complete calculation chain traceable from raw data to reported ESRS metric with SHA-256 hashes
- **Compatibility**: Python 3.11+, Pydantic v2, no external API calls required for core calculations
- **Zero Hallucination**: All emission factors from published regulatory/scientific sources; no LLM in any calculation path
- **Multi-tenancy**: Multiple retail companies can run simultaneously without data leakage (SEC-002 RBAC enforced)

---

## Dependencies

| Dependency | Component | Version | Required |
|-----------|-----------|---------|----------|
| pydantic | All models | >=2.0 | Yes |
| pyyaml | Config/presets | >=6.0 | Yes |
| PACK-001 | CSRD Starter | 1.0 | Bridge |
| PACK-002 | CSRD Professional | 1.0 | Bridge |
| PACK-003 | CSRD Enterprise | 1.0 | Bridge |
| PACK-006/007 | EUDR Starter/Professional | 1.0 | Bridge (optional) |
| PACK-008 | EU Taxonomy | 1.0 | Bridge (optional) |
| AGENT-MRV-001..030 | MRV Agents | 1.0 | Integration |
| AGENT-DATA-001..020 | Data Agents | 1.0 | Integration |
| AGENT-FOUND-001..010 | Foundation | 1.0 | Integration |
| AGENT-EUDR-001..015 | EUDR Traceability | 1.0 | Bridge |

---

**Approval Signatures:**

- Product Manager: ___________________
- Engineering Lead: ___________________
- CEO: ___________________
