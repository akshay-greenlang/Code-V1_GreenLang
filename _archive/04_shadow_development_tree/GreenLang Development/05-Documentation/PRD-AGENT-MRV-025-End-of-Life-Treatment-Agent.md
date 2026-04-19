# PRD: AGENT-MRV-025 — End-of-Life Treatment of Sold Products Agent

## Document Info
| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-MRV-025 |
| **Agent ID** | GL-MRV-S3-012 |
| **Internal Label** | AGENT-MRV-025 |
| **Category** | Layer 3 — MRV / Accounting Agents (Scope 3, Category 12) |
| **Package** | `greenlang/end_of_life_treatment/` |
| **DB Migration** | V076 |
| **API Prefix** | `/api/v1/end-of-life-treatment` |
| **Metrics Prefix** | `gl_eol_` |
| **Table Prefix** | `eol_` |
| **Env Prefix** | `GL_EOL_` |
| **Status** | PLANNED |
| **Date** | 2026-02-28 |
| **Author** | GreenLang AI Engineering |

---

## 1. Executive Summary

AGENT-MRV-025 implements **GHG Protocol Scope 3 Category 12: End-of-Life Treatment of Sold Products** — emissions from waste disposal and treatment of products sold by the reporting organization (in the reporting year) at the end of their useful life.

**Key distinction from Category 5 (Waste Generated in Operations)**:
- **Cat 5** = emissions from treatment of the reporter's OWN operational waste
- **Cat 12** = emissions from treatment of the reporter's SOLD PRODUCTS after customer use

This is a **downstream** category. The reporter must estimate how customers will eventually dispose of sold products, accounting for product material composition, expected treatment pathways by region, and treatment-specific emission factors.

---

## 2. Regulatory & Standard References

| Standard | Section | Requirement |
|----------|---------|-------------|
| GHG Protocol Scope 3 | Chapter 5.12 | Category 12 methodology and boundary |
| GHG Protocol Technical Guidance | Chapter 12 | Detailed calculation methods |
| IPCC 2006 GL | Volume 5 (Waste) | Treatment-specific emission factors |
| IPCC 2019 Refinement | Volume 5 | Updated waste sector methodologies |
| EPA WARM v16 | All modules | Material-specific treatment EFs |
| DEFRA 2024 | Table 13 | UK waste treatment conversion factors |
| ISO 14064-1:2018 | Clause 5.2.4 | Downstream indirect emissions |
| CSRD/ESRS E1 | DR E1-6 | GHG emissions, Scope 3 Cat 12 |
| CSRD/ESRS E5 | DR E5-5 | Resource outflows, waste management |
| CDP | C6.5 | Scope 3 Category 12 reporting |
| SBTi | Scope 3 Standard | Category 12 target setting |
| EU Waste Framework Dir | 2008/98/EC | Waste hierarchy (prevent > reuse > recycle > recover > dispose) |
| EU Packaging & Packaging Waste | 2024/3110 | Extended Producer Responsibility |

---

## 3. Scope & Boundary

### 3.1 Included
- Emissions from **landfilling** of sold products (IPCC FOD model: CH4 from anaerobic decomposition)
- Emissions from **incineration/WtE** (fossil CO2, N2O from combustion)
- Emissions from **recycling** processing (transport + MRF energy, cut-off approach)
- Emissions from **composting** (CH4 + N2O from aerobic decomposition)
- Emissions from **anaerobic digestion** (fugitive CH4 leakage)
- Emissions from **open burning** (developing region default)
- Emissions from **wastewater treatment** of liquid products (CH4 + N2O)
- Product **material decomposition** (multi-material BOM analysis)
- **Regional treatment mix** estimation (EU vs US vs APAC vs Global defaults)
- **Product lifetime distribution** (when EOL occurs after sale)

### 3.2 Excluded
- Emissions from **use phase** of sold products (→ Cat 11)
- Emissions from **processing** of intermediate sold products (→ Cat 10)
- Emissions from the reporter's **own operational waste** (→ Cat 5)
- Emissions from **upstream production** of sold products (→ Cat 1)
- **Transport to treatment facility** (optionally included as secondary emission)
- Revenue from recycled material sales (financial, not GHG)

### 3.3 Double-Counting Prevention (8 Rules)
| Rule ID | Description |
|---------|-------------|
| DC-EOL-001 | vs Cat 5: Reporter's own waste ≠ customer disposal of sold products |
| DC-EOL-002 | vs Cat 1: Upstream cradle-to-gate excludes post-consumer EOL |
| DC-EOL-003 | vs Cat 11: Use-phase ≠ end-of-life treatment |
| DC-EOL-004 | vs Cat 10: Processing of intermediates ≠ final disposal |
| DC-EOL-005 | vs Scope 1: On-site treatment of own products ≠ customer disposal |
| DC-EOL-006 | vs Cat 13: Downstream leased assets have separate boundary |
| DC-EOL-007 | Avoided emissions from recycling reported SEPARATELY (not netted) |
| DC-EOL-008 | Energy recovery credits reported SEPARATELY (not netted) |

---

## 4. Calculation Methods

### 4.1 Method A — Waste-Type-Specific (Recommended)
```
E_total = Σ_products [ Σ_materials [ Σ_treatments [
    units_sold × weight_per_unit × material_fraction
    × treatment_fraction × treatment_EF(material, treatment)
]]]
```

### 4.2 Method B — Average-Data
```
E_total = Σ_products [
    units_sold × weight_per_unit × average_eol_EF(product_type)
]
```
Uses composite EFs that embed typical regional treatment mix.

### 4.3 Method C — Spend-Based (EEIO Screening)
```
E_total = Σ_products [
    revenue_from_product × waste_management_EEIO_EF
]
```
Least accurate, for initial screening only.

### 4.4 Method D — Supplier/Producer-Specific
```
E_total = Σ_products [
    units_sold × weight_per_unit × producer_declared_eol_EF
]
```
Uses EPD/PCF data with verified EOL scenarios.

### 4.5 Method E — Hybrid
Combines methods A-D using best available data per product, with method waterfall:
Producer-specific → Waste-type-specific → Average-data → Spend-based

---

## 5. Architecture — 7-Engine Design

### Engine 1: EOLProductDatabaseEngine (`eol_product_database.py`)
- 15 material types with treatment-specific EFs (landfill/incineration/recycling/composting/AD/open_burn)
- 20 product categories with default material composition (BOM)
- 12 regional treatment mix profiles (US/EU/UK/DE/FR/JP/CN/IN/BR/AU/KR/GLOBAL)
- EPA WARM v16 factors (61 materials)
- DEFRA 2024 factors
- IPCC FOD parameters (DOC, DOCf, MCF, k, F, OX by climate zone)
- Landfill gas collection efficiency factors
- Energy recovery factors for WtE plants

### Engine 2: WasteTypeSpecificCalculatorEngine (`waste_type_specific_calculator.py`)
- Material decomposition from BOM (Bill of Materials)
- Treatment-specific emission calculations per IPCC methodology
- Landfill: IPCC FOD model (DDOCm, methane generation, oxidation, gas collection)
- Incineration: CO2 = mass × dry_matter × carbon_fraction × fossil_carbon × oxidation × 44/12
- Recycling: Cut-off approach (transport + MRF processing only)
- Composting: CH4 = mass × EF_ch4; N2O = mass × EF_n2o
- Anaerobic digestion: CH4_fugitive = biogas × CH4_content × (1 - capture_efficiency)
- Open burning: CO2 + CH4 + N2O per IPCC emission factors
- Biogenic vs fossil CO2 separation

### Engine 3: AverageDataCalculatorEngine (`average_data_calculator.py`)
- Composite EFs by product category (pre-mixed treatment scenarios)
- Regional adjustment factors
- Product weight estimation from unit count
- Uncertainty ranges by data quality tier
- Fallback EF hierarchy: product-specific → category → mixed waste

### Engine 4: ProducerSpecificCalculatorEngine (`producer_specific_calculator.py`)
- EPD/PCF-declared EOL EFs with verification status
- Producer-declared treatment scenarios
- Actual recycled content tracking
- Extended Producer Responsibility (EPR) scheme data
- Take-back program emissions (collection + transport + treatment)
- ISO 14025/EN 15804 conformance validation

### Engine 5: HybridAggregatorEngine (`hybrid_aggregator.py`)
- Method waterfall: producer-specific → waste-type → average-data → spend-based
- Gap-filling with best available method per product
- Avoided emissions calculation (recycling credits, energy recovery)
- Pareto 80/20 hot-spot analysis
- Material substitution benefit estimation
- Circular economy metrics (recycling rate, diversion rate, circularity index)

### Engine 6: ComplianceCheckerEngine (`compliance_checker.py`)
- 7 regulatory frameworks (~50 rules)
- GHG Protocol Scope 3 Cat 12 requirements
- CSRD/ESRS E1 (GHG) + E5 (circular economy/waste)
- EU Waste Framework Directive hierarchy compliance
- EPR scheme compliance validation
- Avoided emissions separate reporting rule
- 8 double-counting prevention rules

### Engine 7: EndOfLifeTreatmentPipelineEngine (`end_of_life_treatment_pipeline.py`)
- 10-stage pipeline: validate → classify → normalize → resolve_efs → calculate → allocate → aggregate → compliance → provenance → seal
- Batch processing (up to 5000 products)
- Portfolio analysis with material flow analysis
- Circularity scoring and waste hierarchy assessment

---

## 6. Data Models (models.py)

### 6.1 Enumerations (22)
1. `MaterialType` — 15: plastic, metal, aluminum, steel, glass, paper, cardboard, wood, textile, electronics, organic, rubber, ceramic, concrete, mixed
2. `TreatmentMethod` — 7: landfill, incineration, recycling, composting, anaerobic_digestion, open_burning, wastewater
3. `ProductCategory` — 20: electronics, appliances, furniture, packaging, clothing, automotive_parts, building_materials, toys, medical_devices, batteries, tires, food_products, beverages, chemicals, cosmetics, office_supplies, sporting_goods, tools, lighting, mixed_products
4. `RegionalTreatmentProfile` — 12: US, EU, UK, DE, FR, JP, CN, IN, BR, AU, KR, GLOBAL
5. `CalculationMethod` — 5: waste_type_specific, average_data, spend_based, producer_specific, hybrid
6. `LandfillType` — 6: managed_anaerobic, managed_semi_aerobic, unmanaged_deep, unmanaged_shallow, engineered_with_gas, engineered_without_gas
7. `ClimateZone` — 4: boreal_temperate_dry, boreal_temperate_wet, tropical_dry, tropical_wet
8. `IncinerationType` — 4: mass_burn, refuse_derived, waste_to_energy, open_burning
9. `RecyclingApproach` — 3: cut_off, closed_loop, substitution
10. `EFSource` — 6: EPA_WARM, DEFRA, IPCC, ECOINVENT, PRODUCER_EPD, CUSTOM
11. `DataQualityTier` — 3: tier_1, tier_2, tier_3
12. `DQIDimension` — 5: temporal, geographical, technological, completeness, reliability
13. `ComplianceFramework` — 7: GHG_PROTOCOL, ISO_14064, CSRD_ESRS, CDP, SBTI, SB_253, GRI
14. `ComplianceStatus` — 4: compliant, non_compliant, partial, not_assessed
15. `PipelineStage` — 10: validate, classify, normalize, resolve_efs, calculate, allocate, aggregate, compliance, provenance, seal
16. `ProvenanceStage` — 10 (mirrors pipeline)
17. `UncertaintyMethod` — 3: monte_carlo, analytical, ipcc_tier2
18. `BatchStatus` — 4: pending, processing, completed, failed
19. `GWPSource` — 2: AR5, AR6
20. `EmissionGas` — 4: co2_fossil, co2_biogenic, ch4, n2o
21. `WasteHierarchyLevel` — 5: prevention, reuse, recycling, recovery, disposal
22. `CircularityMetric` — 4: recycling_rate, diversion_rate, circularity_index, material_recovery_rate

### 6.2 Constant Tables (16)
1. `MATERIAL_TREATMENT_EFS` — 15 materials × 7 treatments = emission factors (kgCO2e/kg)
2. `PRODUCT_MATERIAL_COMPOSITIONS` — 20 product categories with default BOM (material fractions)
3. `REGIONAL_TREATMENT_MIXES` — 12 regions × 7 treatment percentages
4. `LANDFILL_FOD_PARAMETERS` — DOC, DOCf, MCF, k, F, OX by material + climate zone
5. `INCINERATION_PARAMETERS` — dry_matter, carbon_fraction, fossil_fraction, oxidation_factor by material
6. `GAS_COLLECTION_EFFICIENCY` — by landfill type (0.0 to 0.75)
7. `ENERGY_RECOVERY_FACTORS` — WtE efficiency and displaced grid EF by region
8. `RECYCLING_PROCESSING_EFS` — Transport + MRF energy per material (kgCO2e/kg)
9. `AVOIDED_EMISSION_FACTORS` — Material substitution credits (negative EFs)
10. `PRODUCT_WEIGHT_DEFAULTS` — Average weight per unit for 20 product categories
11. `COMPOSTING_EMISSION_FACTORS` — CH4 + N2O per kg organic waste
12. `AD_EMISSION_FACTORS` — Fugitive CH4 per m3 biogas, capture efficiency
13. `DC_RULES` — 8 double-counting prevention rules
14. `COMPLIANCE_FRAMEWORK_RULES` — 7 frameworks with requirements
15. `DQI_SCORING` — 5 dimensions × 3 tiers
16. `UNCERTAINTY_RANGES` — By method and data quality

### 6.3 Pydantic Models (14)
1. `ProductEOLInput` — product_id, category, units_sold, weight_per_unit_kg, material_composition, treatment_scenario, region, lifetime_years
2. `MaterialComposition` — material_type, fraction (0-1), weight_kg
3. `TreatmentScenario` — treatment_method, fraction (0-1), landfill_type, climate_zone, gas_collection, energy_recovery
4. `WasteTypeResult` — material, treatment, weight_kg, emissions_kgco2e, ef_used, ef_source
5. `AverageDataResult` — product_category, units, weight_kg, emissions_kgco2e, composite_ef
6. `ProducerSpecificResult` — product_id, epd_reference, eol_ef, verification_status
7. `CalculationResult` — calc_id, org_id, year, total_tco2e, by_treatment, by_material, by_product, avoided_emissions, dqi, uncertainty, provenance_hash
8. `AvoidedEmissions` — recycling_credit_tco2e, energy_recovery_tco2e, total_avoided_tco2e (reported separately)
9. `CircularityScore` — recycling_rate, diversion_rate, circularity_index, waste_hierarchy_compliance
10. `AggregationResult` — period, total_tco2e, by_treatment, by_material, by_category, by_region
11. `ComplianceResult` — framework, status, rules_checked/passed/failed, findings
12. `ProvenanceRecord` — stage, input_hash, output_hash, timestamp, metadata
13. `DataQualityScore` — 5 dimensions + overall
14. `UncertaintyResult` — method, mean, std_dev, ci_lower, ci_upper

---

## 7. Database Schema (V076)

### 7.1 Tables (21)

**Reference Tables (10):**
1. `gl_eol_material_emission_factors` — material × treatment EFs
2. `gl_eol_product_compositions` — product category → material BOM
3. `gl_eol_regional_treatment_mixes` — region → treatment percentages
4. `gl_eol_landfill_parameters` — FOD model parameters by material/climate
5. `gl_eol_incineration_parameters` — combustion parameters by material
6. `gl_eol_recycling_factors` — processing EFs + avoided emission credits
7. `gl_eol_product_weight_defaults` — average weight per unit by category
8. `gl_eol_gas_collection_factors` — landfill gas collection efficiency
9. `gl_eol_energy_recovery_factors` — WtE efficiency by region
10. `gl_eol_composting_ad_factors` — composting/AD emission parameters

**Operational Tables (8):**
11. `gl_eol_calculations` — **HYPERTABLE** (7-day chunks)
12. `gl_eol_calculation_details` — Extended input/output
13. `gl_eol_material_results` — Per-material treatment results
14. `gl_eol_avoided_emissions` — Recycling/energy recovery credits
15. `gl_eol_compliance_checks` — **HYPERTABLE** (30-day chunks)
16. `gl_eol_aggregations` — **HYPERTABLE** (30-day chunks)
17. `gl_eol_provenance_records` — SHA-256 hash chains
18. `gl_eol_audit_trail` — Operation audit log

**Supporting (3):**
19. `gl_eol_batch_jobs` — Batch processing tracking
20. `gl_eol_data_quality_scores` — DQI scoring
21. `gl_eol_uncertainty_results` — Uncertainty analysis

### 7.2 Infrastructure
- 3 hypertables (calculations, compliance_checks, aggregations)
- 2 continuous aggregates (daily_by_treatment, monthly_by_material)
- RLS policies on all operational tables
- ~100+ indexes, ~90+ check constraints
- ~120 seed records (15 materials × 7 treatments + 20 compositions + 12 regions + FOD params)

---

## 8. REST API (22 Endpoints)

| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | `/calculate` | Full pipeline calculation |
| 2 | POST | `/calculate/waste-type-specific` | Method A — material-level |
| 3 | POST | `/calculate/waste-type-specific/landfill` | Landfill IPCC FOD |
| 4 | POST | `/calculate/waste-type-specific/incineration` | Incineration/WtE |
| 5 | POST | `/calculate/waste-type-specific/recycling` | Recycling cut-off |
| 6 | POST | `/calculate/average-data` | Method B — composite EFs |
| 7 | POST | `/calculate/producer-specific` | Method D — EPD/PCF |
| 8 | POST | `/calculate/hybrid` | Method E — combined |
| 9 | POST | `/calculate/batch` | Batch processing |
| 10 | POST | `/calculate/portfolio` | Portfolio analysis |
| 11 | POST | `/compliance/check` | Compliance validation |
| 12 | GET | `/calculations/{id}` | Get calculation by ID |
| 13 | GET | `/calculations` | List with pagination |
| 14 | DELETE | `/calculations/{id}` | Soft delete |
| 15 | GET | `/emission-factors/{material}` | Material EFs |
| 16 | GET | `/product-compositions` | Default BOM lookup |
| 17 | GET | `/treatment-mixes` | Regional treatment profiles |
| 18 | GET | `/avoided-emissions/{id}` | Recycling/energy credits |
| 19 | GET | `/circularity-score/{id}` | Circularity metrics |
| 20 | GET | `/aggregations` | Aggregated results |
| 21 | GET | `/provenance/{id}` | Provenance chain |
| 22 | GET | `/health` | Health check |

---

## 9. Auth Integration

### 9.1 PERMISSION_MAP (22 entries)
```
end-of-life-treatment:calculate — POST calculate endpoints
end-of-life-treatment:read — GET endpoints
end-of-life-treatment:delete — DELETE endpoints
end-of-life-treatment:compliance — POST compliance/check
```

### 9.2 Router Registration
- Import: `from greenlang.end_of_life_treatment.setup import get_router as get_eol_router`
- Variable: `_eol_router`
- Include: `app.include_router(_eol_router)`

---

## 10. Prometheus Metrics (14)

| # | Metric Name | Type | Labels |
|---|-------------|------|--------|
| 1 | `gl_eol_calculations_total` | Counter | method, category, status |
| 2 | `gl_eol_calculation_duration_seconds` | Histogram | method, category |
| 3 | `gl_eol_emissions_kg_total` | Counter | treatment_method |
| 4 | `gl_eol_products_processed_total` | Counter | category |
| 5 | `gl_eol_landfill_emissions_total` | Counter | material_type |
| 6 | `gl_eol_incineration_emissions_total` | Counter | material_type |
| 7 | `gl_eol_recycling_emissions_total` | Counter | material_type |
| 8 | `gl_eol_avoided_emissions_total` | Counter | type (recycling/energy_recovery) |
| 9 | `gl_eol_compliance_checks_total` | Counter | framework, status |
| 10 | `gl_eol_dc_rule_triggers_total` | Counter | rule_id |
| 11 | `gl_eol_pipeline_stage_duration_seconds` | Histogram | stage |
| 12 | `gl_eol_circularity_score` | Gauge | — |
| 13 | `gl_eol_diversion_rate` | Gauge | — |
| 14 | `gl_eol_active_calculations` | Gauge | — |

---

## 11. Test Plan

### 11.1 Test Files (14)
| File | Focus | Target Tests |
|------|-------|-------------|
| `__init__.py` | Package constants | 5 |
| `conftest.py` | Fixtures & mocks | — |
| `test_models.py` | 22 enums, 16 tables, 14 models | 120+ |
| `test_config.py` | 18 config sections, env vars | 50+ |
| `test_eol_product_database.py` | Material EFs, BOMs, regions | 60+ |
| `test_waste_type_specific_calculator.py` | Landfill/incineration/recycling | 50+ |
| `test_average_data_calculator.py` | Composite EFs, regional | 40+ |
| `test_producer_specific_calculator.py` | EPD/PCF validation | 35+ |
| `test_hybrid_aggregator.py` | Method waterfall, avoided | 40+ |
| `test_compliance_checker.py` | 7 frameworks, 8 DC rules | 45+ |
| `test_provenance.py` | SHA-256, Merkle, chains | 60+ |
| `test_end_of_life_pipeline.py` | 10-stage pipeline | 30+ |
| `test_api.py` | 22 endpoints | 30+ |
| `test_setup.py` | Service facade, singletons | 30+ |

**Total Target: 600+ expanded tests**

---

## 12. Key Technical Differentiators (vs Cat 5)

| Aspect | Cat 5 (Own Waste) | Cat 12 (Sold Product EOL) |
|--------|-------------------|---------------------------|
| Whose waste? | Reporter's operations | Customer disposal |
| Known treatment? | Usually known | Must be ESTIMATED |
| Material data | Waste audit data | Product BOM (Bill of Materials) |
| Regional mix | Single site location | Customer distribution regions |
| Product lifetime | N/A (immediate waste) | Years after sale |
| Avoided emissions | Optional | Must report SEPARATELY |
| Circularity | Diversion rate | Product circularity index |
| EPR schemes | N/A | Extended Producer Responsibility |
| Data quality | Higher (operational) | Lower (estimated downstream) |

---

## 13. Dependencies

| Dependency | Purpose |
|-----------|---------|
| AGENT-FOUND-001 (Orchestrator) | DAG execution |
| AGENT-FOUND-003 (Unit Normalizer) | Mass unit conversions |
| AGENT-FOUND-005 (Citations) | EF source citations |
| AGENT-FOUND-008 (Reproducibility) | Determinism verification |
| AGENT-FOUND-010 (Observability) | Metrics and tracing |
| AGENT-MRV-018 (Waste Generated) | Shared IPCC methodology (FOD model) |
| AGENT-MRV-024 (Use of Sold Products) | Product lifetime → EOL timing |
