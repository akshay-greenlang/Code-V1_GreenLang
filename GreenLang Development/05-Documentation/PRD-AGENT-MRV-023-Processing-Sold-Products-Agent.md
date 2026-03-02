# PRD: AGENT-MRV-023 — Processing of Sold Products Agent (GL-MRV-S3-010)

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-010 |
| **Component** | AGENT-MRV-023 |
| **Category** | Scope 3, Category 10: Processing of Sold Products |
| **Version** | 1.0.0 |
| **Table Prefix** | `gl_psp_` |
| **Module** | `greenlang/processing_sold_products/` |
| **Tests** | `tests/unit/mrv/test_processing_sold_products/` |
| **Migration** | `V074__processing_sold_products_service.sql` |
| **Standard** | GHG Protocol Scope 3 Standard, Chapter 6 |

## 2. Scope & Applicability

Category 10 covers emissions from downstream processing of **intermediate products** sold by the reporting company to third-party processors/manufacturers, occurring AFTER the sale and BEFORE end use by the final consumer.

**Applicable when:**
- Company sells intermediate products (raw materials, semi-finished goods, components)
- These products require further processing, transformation, or incorporation into another product
- Processing occurs at customer facilities (not the reporting company's operations)

**NOT applicable when:**
- Company sells finished goods directly to end consumers (→ use Cat 11)
- Processing occurs at the reporting company's own facilities (→ Scope 1/2)
- Transportation to the processor (→ Cat 4 or Cat 9)

### 2.1 Intermediate Product Categories (12)
| Code | Category | Examples |
|------|----------|----------|
| `METALS_FERROUS` | Ferrous Metals | Steel coil, sheet, billet, wire rod |
| `METALS_NON_FERROUS` | Non-Ferrous Metals | Aluminum sheet, copper wire, zinc ingot |
| `PLASTICS_THERMOPLASTIC` | Thermoplastics | PE pellets, PP granules, PET resin, ABS |
| `PLASTICS_THERMOSET` | Thermosets | Epoxy resin, polyurethane, phenolic resin |
| `CHEMICALS` | Chemicals | Chemical intermediates, catalysts, solvents |
| `FOOD_INGREDIENTS` | Food Ingredients | Flour, sugar, vegetable oil, starch |
| `TEXTILES` | Textile Materials | Yarn, fiber, fabric roll, dyed material |
| `ELECTRONICS` | Electronic Components | Semiconductors, PCBs, passive components |
| `GLASS_CERAMICS` | Glass & Ceramics | Float glass, ceramic powder, fiberglass |
| `WOOD_PAPER` | Wood & Paper | Lumber, pulp, paperboard, MDF |
| `MINERALS` | Mineral Products | Cement clinker, aggregates, gypsum |
| `AGRICULTURAL` | Agricultural Products | Grain, oilseed, raw cotton, raw rubber |

### 2.2 Processing Types (18)
| Code | Processing Type | Typical Energy (kWh/t) | Applicable Products |
|------|----------------|----------------------|---------------------|
| `MACHINING` | CNC/Conventional Machining | 150-450 | Metals |
| `STAMPING` | Metal Stamping/Pressing | 80-200 | Ferrous/Non-ferrous metals |
| `WELDING` | Welding/Joining | 100-350 | Metals, thermoplastics |
| `HEAT_TREATMENT` | Heat Treatment/Annealing | 200-600 | Metals, glass |
| `INJECTION_MOLDING` | Injection Molding | 300-800 | Thermoplastics |
| `EXTRUSION` | Extrusion | 200-500 | Plastics, metals, food |
| `BLOW_MOLDING` | Blow Molding | 250-600 | Thermoplastics |
| `CASTING` | Casting/Foundry | 400-1200 | Metals |
| `FORGING` | Forging | 300-900 | Ferrous metals |
| `COATING` | Coating/Painting/Plating | 50-200 | Metals, plastics, wood |
| `ASSEMBLY` | Assembly/Integration | 20-80 | Electronics, automotive |
| `CHEMICAL_REACTION` | Chemical Synthesis | 500-2000 | Chemicals |
| `REFINING` | Refining/Purification | 400-1500 | Chemicals, food |
| `MILLING` | Milling/Grinding | 100-300 | Minerals, food, wood |
| `DRYING` | Drying/Curing | 150-500 | Food, textiles, wood, ceramics |
| `SINTERING` | Sintering | 600-2000 | Ceramics, metals (powder) |
| `FERMENTATION` | Fermentation/Bio-processing | 80-250 | Food, chemicals |
| `TEXTILE_FINISHING` | Weaving/Dyeing/Finishing | 200-700 | Textiles |

## 3. Seven-Engine Architecture

### Engine 1: ProcessingDatabaseEngine
- **Purpose**: Emission factor storage and retrieval for processing activities
- **Key data**: 18 processing type EFs, 12 product category EFs, grid EFs (130+ countries), industry sector benchmarks, energy mix profiles
- **Lookups**: By processing type, product category, country, industry sector, year

### Engine 2: SiteSpecificCalculatorEngine
- **Purpose**: Calculate emissions using actual customer-provided processing data
- **Methods**:
  - Direct emissions data: `emissions = Σ(customer_reported_emissions_per_unit × quantity_sold)`
  - Energy-based: `emissions = Σ(energy_per_unit × quantity_sold × grid_EF_customer_location)`
  - Fuel-based: `emissions = Σ(fuel_per_unit × quantity_sold × fuel_EF)`
- **Data sources**: Customer sustainability reports, supplier surveys, EPDs
- **DQI**: Highest quality (score 80-100)

### Engine 3: AverageDataCalculatorEngine
- **Purpose**: Calculate using industry-average processing emission intensities
- **Methods**:
  - Process-specific: `emissions = Σ(quantity_sold × process_EF_per_unit)`
  - Energy intensity: `emissions = Σ(quantity_sold × energy_intensity × grid_EF_avg)`
  - Sector benchmark: `emissions = Σ(quantity_sold × sector_benchmark_EF)`
- **Data sources**: DEFRA, EPA, Ecoinvent, industry associations
- **DQI**: Medium quality (score 40-70)

### Engine 4: SpendBasedCalculatorEngine
- **Purpose**: EEIO-based screening estimate using revenue from intermediate products
- **Methods**:
  - EEIO: `emissions = Σ(revenue × EEIO_sector_factor × margin_adjustment)`
  - Hybrid: Combine spend-based with partial process data
- **Data sources**: USEEIO, EXIOBASE, DEFRA EEIO
- **DQI**: Lowest quality (score 20-40)

### Engine 5: HybridAggregatorEngine
- **Purpose**: Combine multiple calculation methods across product portfolio
- **Features**:
  - Method waterfall: site-specific → average-data → spend-based
  - Proportional allocation for multi-use products
  - Portfolio-level aggregation with method mixing
  - Gap-filling with lower-tier methods
  - Weighted DQI scoring across methods

### Engine 6: ComplianceCheckerEngine
- **Purpose**: Validate against 7 regulatory frameworks
- **Frameworks**: GHG Protocol, ISO 14064, CSRD ESRS E1, CDP, SBTi, SB 253, GRI 305
- **Rules**: ~45 compliance rules, 8 double-counting prevention rules
- **Features**: Framework-specific validation, boundary checks, completeness scoring

### Engine 7: ProcessingPipelineEngine
- **Purpose**: 10-stage orchestration pipeline
- **Stages**: validate → classify → normalize → resolve_efs → calculate → allocate → aggregate → compliance → provenance → seal
- **Features**: Stage-level error handling, partial results, batch processing, portfolio analysis

## 4. Calculation Methods

### 4.1 Site-Specific Method (Highest Accuracy)

**Formula A — Direct Emissions:**
```
E_cat10 = Σᵢ (Q_sold_i × EF_processing_i)
where:
  Q_sold_i  = quantity of intermediate product i sold (tonnes or units)
  EF_processing_i = customer-reported processing emissions per unit (kgCO2e/t or kgCO2e/unit)
```

**Formula B — Energy-Based:**
```
E_cat10 = Σᵢ (Q_sold_i × EP_i × EF_grid_j)
where:
  EP_i      = energy consumed per unit in processing (kWh/t or kWh/unit)
  EF_grid_j = grid emission factor at customer location j (kgCO2e/kWh)
```

**Formula C — Fuel-Based:**
```
E_cat10 = Σᵢ Σⱼ (Q_sold_i × FP_ij × EF_fuel_j)
where:
  FP_ij     = fuel consumed per unit of product i for fuel type j (liters/t)
  EF_fuel_j = emission factor for fuel type j (kgCO2e/liter)
```

### 4.2 Average-Data Method

**Formula D — Process EF:**
```
E_cat10 = Σᵢ (Q_sold_i × EF_process_i)
where:
  EF_process_i = industry-average processing EF for product category i (kgCO2e/t)
```

**Formula E — Energy Intensity:**
```
E_cat10 = Σᵢ (Q_sold_i × EI_i × EF_grid_avg)
where:
  EI_i       = average energy intensity for processing type (kWh/t)
  EF_grid_avg = average grid EF for processing regions (kgCO2e/kWh)
```

### 4.3 Spend-Based Method

**Formula F — EEIO:**
```
E_cat10 = Σᵢ (Rev_i × EF_eeio_sector_i × (1 - margin_i))
where:
  Rev_i       = revenue from intermediate product i (USD)
  EF_eeio_i   = EEIO factor for downstream sector (kgCO2e/USD)
  margin_i    = profit margin adjustment (typically 0.05-0.15)
```

## 5. Emission Factor Tables

### 5.1 Processing Emission Factors by Product Category (kgCO2e/tonne)
| Product Category | Site-Specific Range | Average EF | Uncertainty |
|-----------------|-------------------|-----------|-------------|
| `METALS_FERROUS` | 120-450 | 280 | ±25% |
| `METALS_NON_FERROUS` | 200-600 | 380 | ±25% |
| `PLASTICS_THERMOPLASTIC` | 300-900 | 520 | ±30% |
| `PLASTICS_THERMOSET` | 250-750 | 450 | ±30% |
| `CHEMICALS` | 200-2500 | 680 | ±35% |
| `FOOD_INGREDIENTS` | 50-250 | 130 | ±20% |
| `TEXTILES` | 150-700 | 350 | ±30% |
| `ELECTRONICS` | 500-2000 | 950 | ±35% |
| `GLASS_CERAMICS` | 300-1200 | 580 | ±25% |
| `WOOD_PAPER` | 80-350 | 190 | ±20% |
| `MINERALS` | 100-500 | 250 | ±25% |
| `AGRICULTURAL` | 40-200 | 110 | ±20% |

### 5.2 Processing Energy Intensity (kWh/tonne by Processing Type)
| Processing Type | Low | Mid | High | Default |
|----------------|-----|-----|------|---------|
| `MACHINING` | 150 | 280 | 450 | 280 |
| `STAMPING` | 80 | 140 | 200 | 140 |
| `WELDING` | 100 | 220 | 350 | 220 |
| `HEAT_TREATMENT` | 200 | 380 | 600 | 380 |
| `INJECTION_MOLDING` | 300 | 520 | 800 | 520 |
| `EXTRUSION` | 200 | 340 | 500 | 340 |
| `BLOW_MOLDING` | 250 | 400 | 600 | 400 |
| `CASTING` | 400 | 750 | 1200 | 750 |
| `FORGING` | 300 | 580 | 900 | 580 |
| `COATING` | 50 | 120 | 200 | 120 |
| `ASSEMBLY` | 20 | 45 | 80 | 45 |
| `CHEMICAL_REACTION` | 500 | 1100 | 2000 | 1100 |
| `REFINING` | 400 | 900 | 1500 | 900 |
| `MILLING` | 100 | 190 | 300 | 190 |
| `DRYING` | 150 | 310 | 500 | 310 |
| `SINTERING` | 600 | 1200 | 2000 | 1200 |
| `FERMENTATION` | 80 | 160 | 250 | 160 |
| `TEXTILE_FINISHING` | 200 | 420 | 700 | 420 |

### 5.3 EEIO Factors by Downstream Sector (kgCO2e/USD)
| Sector Code | Sector | EEIO Factor | Margin |
|-------------|--------|-------------|--------|
| `NAICS_331` | Primary Metal Manufacturing | 0.82 | 0.08 |
| `NAICS_332` | Fabricated Metal Products | 0.45 | 0.10 |
| `NAICS_325` | Chemical Manufacturing | 0.65 | 0.12 |
| `NAICS_326` | Plastics & Rubber Products | 0.52 | 0.10 |
| `NAICS_311` | Food Manufacturing | 0.38 | 0.08 |
| `NAICS_313` | Textile Mills | 0.42 | 0.10 |
| `NAICS_334` | Computer & Electronic Products | 0.28 | 0.15 |
| `NAICS_327` | Nonmetallic Mineral Products | 0.72 | 0.08 |
| `NAICS_321` | Wood Product Manufacturing | 0.35 | 0.10 |
| `NAICS_322` | Paper Manufacturing | 0.48 | 0.10 |
| `NAICS_336` | Transportation Equipment | 0.40 | 0.12 |
| `NAICS_335` | Electrical Equipment | 0.32 | 0.12 |

### 5.4 Grid Emission Factors (kgCO2e/kWh by Country/Region)
| Country/Region | Code | Grid EF | Year |
|---------------|------|---------|------|
| United States | US | 0.417 | 2024 |
| United Kingdom | GB | 0.233 | 2024 |
| Germany | DE | 0.348 | 2024 |
| France | FR | 0.052 | 2024 |
| China | CN | 0.555 | 2024 |
| India | IN | 0.708 | 2024 |
| Japan | JP | 0.462 | 2024 |
| South Korea | KR | 0.424 | 2024 |
| Brazil | BR | 0.075 | 2024 |
| Canada | CA | 0.120 | 2024 |
| Australia | AU | 0.656 | 2024 |
| Mexico | MX | 0.431 | 2024 |
| Italy | IT | 0.256 | 2024 |
| Spain | ES | 0.175 | 2024 |
| Poland | PL | 0.635 | 2024 |
| Global Average | GLOBAL | 0.475 | 2024 |

### 5.5 Fuel Emission Factors (kgCO2e/liter)
| Fuel Type | Code | EF | Source |
|-----------|------|-----|--------|
| Natural Gas (m³) | `NATURAL_GAS` | 2.024 | DEFRA 2024 |
| Diesel | `DIESEL` | 2.706 | DEFRA 2024 |
| Heavy Fuel Oil | `HFO` | 3.114 | DEFRA 2024 |
| LPG | `LPG` | 1.557 | DEFRA 2024 |
| Coal (kg) | `COAL` | 2.883 | DEFRA 2024 |
| Biomass (kg) | `BIOMASS` | 0.015 | DEFRA 2024 |

### 5.6 Multi-Step Processing Chains
| Product Category | Typical Chain | Combined EF (kgCO2e/t) |
|-----------------|--------------|----------------------|
| Steel Automotive Parts | Stamping → Welding → Coating | 480 |
| Aluminum Beverage Cans | Stamping → Coating → Assembly | 420 |
| Plastic Packaging | Injection Molding → Assembly | 565 |
| Semiconductor Chips | Chemical → Assembly → Testing | 1800 |
| Food Products | Milling → Drying → Packaging | 350 |
| Textile Garments | Weaving → Dyeing → Assembly | 620 |
| Glass Bottles | Heat Treatment → Coating | 500 |
| Paper Products | Milling → Drying → Coating | 380 |

### 5.7 Currencies & CPI Deflation
| Currency | Code | USD Rate | CPI 2024 Base |
|----------|------|----------|---------------|
| US Dollar | USD | 1.000 | 100.0 |
| Euro | EUR | 1.085 | 100.0 |
| British Pound | GBP | 1.268 | 100.0 |
| Japanese Yen | JPY | 0.0067 | 100.0 |
| Chinese Yuan | CNY | 0.138 | 100.0 |
| Indian Rupee | INR | 0.012 | 100.0 |
| Canadian Dollar | CAD | 0.742 | 100.0 |
| Australian Dollar | AUD | 0.651 | 100.0 |
| Korean Won | KRW | 0.00075 | 100.0 |
| Brazilian Real | BRL | 0.198 | 100.0 |
| Mexican Peso | MXN | 0.058 | 100.0 |
| Swiss Franc | CHF | 1.122 | 100.0 |

| Year | CPI Index |
|------|-----------|
| 2015 | 76.5 |
| 2016 | 77.5 |
| 2017 | 79.1 |
| 2018 | 81.0 |
| 2019 | 82.5 |
| 2020 | 83.5 |
| 2021 | 87.3 |
| 2022 | 94.1 |
| 2023 | 97.8 |
| 2024 | 100.0 |
| 2025 | 102.4 |

### 5.8 Data Quality Indicator Dimensions
| Dimension | Score 1 (Low) | Score 3 (Medium) | Score 5 (High) |
|-----------|--------------|-----------------|---------------|
| Reliability | Estimate/proxy | Published EF | Verified measurement |
| Completeness | <30% products | 30-80% products | >80% products |
| Temporal | >10 years old | 3-10 years old | <3 years old |
| Geographical | Global average | Continental/national | Site-specific |
| Technological | Generic sector | Industry subsector | Process-specific |

### 5.9 Uncertainty Ranges by Method
| Method | Min Uncertainty | Default | Max Uncertainty |
|--------|----------------|---------|----------------|
| Site-Specific (Direct) | ±5% | ±10% | ±20% |
| Site-Specific (Energy) | ±10% | ±15% | ±25% |
| Average-Data (Process) | ±15% | ±25% | ±40% |
| Average-Data (Sector) | ±20% | ±30% | ±50% |
| Spend-Based (EEIO) | ±30% | ±50% | ±100% |

## 6. Double-Counting Prevention Rules (8)

| Rule ID | Rule | Description |
|---------|------|-------------|
| DC-PSP-001 | vs Scope 1 | Exclude emissions from processing at own facilities |
| DC-PSP-002 | vs Scope 2 | Exclude electricity consumed at own facilities |
| DC-PSP-003 | vs Cat 1 | No overlap with purchased goods (upstream) |
| DC-PSP-004 | vs Cat 2 | No overlap with capital goods (own equipment) |
| DC-PSP-005 | vs Cat 4/9 | Exclude transportation emissions (covered by Cat 4/9) |
| DC-PSP-006 | vs Cat 11 | No overlap with use-phase emissions (post-processing) |
| DC-PSP-007 | vs Cat 12 | No overlap with end-of-life (post-use) |
| DC-PSP-008 | Multi-step | Avoid counting same processing step twice in chain |

## 7. Compliance Frameworks (7)

| Framework | Key Requirements |
|-----------|-----------------|
| **GHG Protocol** | Scope 3 Ch. 6, intermediate product boundary, 3 calc methods, DQI |
| **ISO 14064-1** | Clause 5.2.4 indirect GHG, verification evidence |
| **CSRD ESRS E1** | E1-6 Scope 3 disclosure, DNSH assessment, value chain boundary |
| **CDP** | C6.5 Scope 3 Cat 10, method description, data quality |
| **SBTi** | Scope 3 target coverage, base year recalculation triggers |
| **SB 253** | California Climate Corporate Data Accountability Act |
| **GRI 305** | 305-3 Other indirect GHG, calculation methodology |

## 8. API Endpoints (20)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/api/v1/processing-sold-products/calculate` | Full pipeline calculation |
| 2 | POST | `/api/v1/processing-sold-products/calculate/site-specific` | Site-specific method |
| 3 | POST | `/api/v1/processing-sold-products/calculate/site-specific/energy` | Energy-based site-specific |
| 4 | POST | `/api/v1/processing-sold-products/calculate/site-specific/fuel` | Fuel-based site-specific |
| 5 | POST | `/api/v1/processing-sold-products/calculate/average-data` | Average-data method |
| 6 | POST | `/api/v1/processing-sold-products/calculate/average-data/energy-intensity` | Energy intensity method |
| 7 | POST | `/api/v1/processing-sold-products/calculate/spend` | Spend-based EEIO |
| 8 | POST | `/api/v1/processing-sold-products/calculate/hybrid` | Hybrid aggregation |
| 9 | POST | `/api/v1/processing-sold-products/calculate/batch` | Batch calculation |
| 10 | POST | `/api/v1/processing-sold-products/calculate/portfolio` | Portfolio analysis |
| 11 | POST | `/api/v1/processing-sold-products/compliance/check` | Compliance validation |
| 12 | GET | `/api/v1/processing-sold-products/calculations/{id}` | Get calculation by ID |
| 13 | GET | `/api/v1/processing-sold-products/calculations` | List calculations |
| 14 | DELETE | `/api/v1/processing-sold-products/calculations/{id}` | Delete calculation |
| 15 | GET | `/api/v1/processing-sold-products/emission-factors/{category}` | Get EFs by product category |
| 16 | GET | `/api/v1/processing-sold-products/processing-types` | List processing types & EFs |
| 17 | GET | `/api/v1/processing-sold-products/processing-chains` | Get multi-step chains |
| 18 | GET | `/api/v1/processing-sold-products/aggregations` | Get aggregated results |
| 19 | GET | `/api/v1/processing-sold-products/provenance/{id}` | Get provenance chain |
| 20 | GET | `/api/v1/processing-sold-products/health` | Health check |

## 9. Database Schema (19 Tables)

### 9.1 Core Tables
1. `gl_psp_processing_emission_factors` — Processing EFs by product category & type
2. `gl_psp_energy_intensity_factors` — Energy intensity by processing type (kWh/t)
3. `gl_psp_grid_emission_factors` — Grid EFs by country/region (kgCO2e/kWh)
4. `gl_psp_fuel_emission_factors` — Fuel EFs (kgCO2e/liter or kg)
5. `gl_psp_eeio_sector_factors` — EEIO factors by NAICS sector
6. `gl_psp_processing_chains` — Multi-step processing chain definitions
7. `gl_psp_intermediate_products` — Product registry with categories
8. `gl_psp_customer_processing_data` — Customer-provided processing data
9. `gl_psp_currencies` — Currency conversion rates
10. `gl_psp_cpi_deflators` — CPI deflation table

### 9.2 Result Tables
11. `gl_psp_calculations` — Calculation results (TimescaleDB hypertable)
12. `gl_psp_calculation_details` — Per-product calculation breakdown
13. `gl_psp_aggregations` — Aggregated results by period/category
14. `gl_psp_provenance_records` — Provenance chain hashes
15. `gl_psp_compliance_results` — Compliance check results

### 9.3 Operational Tables
16. `gl_psp_data_quality_scores` — DQI scoring per calculation
17. `gl_psp_uncertainty_results` — Uncertainty analysis results
18. `gl_psp_audit_trail` — Change tracking and audit log
19. `gl_psp_batch_jobs` — Batch processing job tracking

## 10. Enumerations (20)

1. `IntermediateProductCategory` — 12 product categories
2. `ProcessingType` — 18 processing types
3. `CalculationMethod` — 5 methods (site_specific_direct, site_specific_energy, site_specific_fuel, average_data, spend_based)
4. `EnergyType` — 6 energy types (electricity, natural_gas, diesel, hfo, lpg, coal)
5. `FuelType` — 6 fuel types
6. `GridRegion` — 16 countries/regions
7. `NAICSSector` — 12 downstream sectors
8. `Currency` — 12 currencies
9. `ProcessingChainType` — 8 chain types (metals_automotive, aluminum_packaging, etc.)
10. `DataQualityTier` — 3 tiers (tier_1, tier_2, tier_3)
11. `DQIDimension` — 5 dimensions (reliability, completeness, temporal, geographical, technological)
12. `ComplianceFramework` — 7 frameworks
13. `ComplianceStatus` — 4 statuses (compliant, non_compliant, partial, not_applicable)
14. `PipelineStage` — 10 stages
15. `ProvenanceStage` — 10 stages
16. `AllocationMethod` — 4 methods (mass, revenue, units, equal)
17. `UncertaintyMethod` — 3 methods (analytical, monte_carlo, bootstrap)
18. `BatchStatus` — 4 statuses (pending, running, completed, failed)
19. `AuditAction` — 6 actions (create, update, delete, calculate, validate, export)
20. `ProductUnit` — 5 units (tonne, kg, unit, m2, m3)

## 11. Pydantic Models (12)

1. `IntermediateProductInput` — Product input with category, processing type, quantity
2. `SiteSpecificInput` — Customer-provided processing data
3. `AverageDataInput` — Average-data method input
4. `SpendBasedInput` — Revenue and sector-based input
5. `ProcessingChainInput` — Multi-step processing chain input
6. `CalculationResult` — Complete calculation result with emissions
7. `ProductBreakdown` — Per-product calculation breakdown
8. `AggregationResult` — Portfolio/period aggregation
9. `ComplianceResult` — Framework compliance check result
10. `ProvenanceRecord` — Provenance chain record
11. `DataQualityScore` — 5-dimension DQI score
12. `UncertaintyResult` — Uncertainty analysis output

## 12. File Manifest

### Source Files (15)
| # | File | Engine | Est. Lines |
|---|------|--------|-----------|
| 1 | `__init__.py` | Package init | ~130 |
| 2 | `models.py` | Data models, enums, constants | ~2,200 |
| 3 | `config.py` | Configuration (GL_PSP_ env prefix) | ~2,400 |
| 4 | `metrics.py` | Prometheus metrics (gl_psp_ prefix) | ~1,200 |
| 5 | `provenance.py` | SHA-256 chain hashing | ~2,100 |
| 6 | `processing_database.py` | Engine 1: EF storage & retrieval | ~2,200 |
| 7 | `site_specific_calculator.py` | Engine 2: Customer data calc | ~2,400 |
| 8 | `average_data_calculator.py` | Engine 3: Industry average calc | ~1,800 |
| 9 | `spend_based_calculator.py` | Engine 4: EEIO screening calc | ~1,500 |
| 10 | `hybrid_aggregator.py` | Engine 5: Multi-method aggregation | ~1,800 |
| 11 | `compliance_checker.py` | Engine 6: 7 frameworks, 8 DC rules | ~3,200 |
| 12 | `processing_pipeline.py` | Engine 7: 10-stage pipeline | ~1,800 |
| 13 | `setup.py` | Service facade wiring 7 engines | ~1,700 |
| 14 | `api/__init__.py` | API subpackage | ~1 |
| 15 | `api/router.py` | 20 REST endpoints | ~2,300 |

### Test Files (14)
| # | File | Tests |
|---|------|-------|
| 1 | `__init__.py` | Package marker |
| 2 | `conftest.py` | Fixtures, factories, singletons |
| 3 | `test_models.py` | Enums, constants, Pydantic models |
| 4 | `test_config.py` | Config loading, env overrides |
| 5 | `test_processing_database.py` | EF lookups, grid factors |
| 6 | `test_site_specific_calculator.py` | Direct/energy/fuel methods |
| 7 | `test_average_data_calculator.py` | Process EF, energy intensity |
| 8 | `test_spend_based_calculator.py` | EEIO, CPI deflation |
| 9 | `test_hybrid_aggregator.py` | Multi-method, portfolio |
| 10 | `test_compliance_checker.py` | 7 frameworks, 8 DC rules |
| 11 | `test_provenance.py` | Hash chains, Merkle trees |
| 12 | `test_processing_pipeline.py` | 10-stage pipeline |
| 13 | `test_api.py` | 20 endpoints |
| 14 | `test_setup.py` | Service wiring |

### Migration (1)
| # | File | Description |
|---|------|-------------|
| 1 | `V074__processing_sold_products_service.sql` | 19 tables, 3 hypertables, 2 cont. aggs |

## 13. Key Architectural Decisions

1. **Intermediate product focus**: Unlike Cat 1 (purchased goods upstream), Cat 10 tracks DOWNSTREAM processing at customer facilities
2. **Multi-step chains**: Support for sequential processing steps (e.g., stamping → welding → coating) with combined emission factors
3. **Customer data hierarchy**: Prioritize site-specific data over averages; method waterfall with automatic fallback
4. **Energy-based calculation**: Convert processing energy consumption to emissions using location-specific grid factors
5. **Proportional allocation**: When same product goes to multiple end-uses, allocate by mass/revenue/units
6. **Boundary clarity**: Clear exclusion of transportation (Cat 4/9), use-phase (Cat 11), and end-of-life (Cat 12)
7. **EEIO margin adjustment**: Apply downstream margin removal before applying EEIO factors
8. **Thread-safe singletons**: All engines use threading.RLock for concurrent access
9. **Decimal precision**: All monetary and emission calculations use Python Decimal with ROUND_HALF_UP
10. **SHA-256 provenance**: Every calculation gets a cryptographic hash chain for audit trail
