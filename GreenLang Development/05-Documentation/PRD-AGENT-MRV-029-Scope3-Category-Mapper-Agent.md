# PRD: AGENT-MRV-029 -- Scope 3 Category Mapper (Cross-Cutting MRV Agent)

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-AGENT-MRV-029 |
| Agent ID | GL-MRV-X-040 |
| Component | AGENT-MRV-029 |
| Category | Cross-Cutting MRV -- Scope 3 Category Mapping & Routing |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-28 |

---

## 1. Overview

### 1.1 Purpose
Build a production-grade **Scope 3 Category Mapper** cross-cutting agent that deterministically classifies organizational data (spend records, purchase orders, bills of materials, activity data, supplier records) into the correct GHG Protocol Scope 3 category (Category 1--15) and routes them to the appropriate category-specific agent (AGENT-MRV-014 through AGENT-MRV-028). This is the **first cross-cutting MRV agent** and serves as the intelligent gateway between raw organizational data and the 15 Scope 3 calculation agents.

### 1.2 Problem Statement
Organizations have diverse data formats (ERP exports, procurement systems, travel booking platforms, fleet management tools) that need to be classified into Scope 3 categories before emissions can be calculated. Without a reliable mapper:
- Data is manually assigned to categories (error-prone, inconsistent)
- Transactions that span multiple categories are misclassified
- Completeness gaps across categories go undetected
- Double-counting between categories occurs silently
- No audit trail of why data was assigned to a particular category

### 1.3 Cross-Cutting Role
Unlike category-specific agents (MRV-014 through MRV-028) that calculate emissions for a single Scope 3 category, this agent operates **across all 15 categories**:

| Aspect | Category-Specific Agents | Scope 3 Category Mapper (This Agent) |
|--------|-------------------------|--------------------------------------|
| Scope | Single category (e.g., Cat 1) | All 15 categories |
| Input | Pre-classified activity data | Raw organizational data |
| Output | tCO2e emissions | Category classification + routing |
| Position | Downstream of mapper | Upstream gateway |
| Count | 15 agents (MRV-014 to 028) | 1 cross-cutting agent |

### 1.4 Key Capabilities
1. **Multi-Code Classification**: NAICS 2022 (1,057 codes), ISIC Rev 4 (419 codes), UNSPSC v26 (55,000+ codes), HS 2022 (5,600+ codes)
2. **Spend Classification**: GL account mapping, procurement category mapping, keyword-based classification
3. **Activity Routing**: Routes classified data to the correct category agent with input transformation
4. **Multi-Category Splitting**: Handles transactions that span multiple categories (e.g., freight + goods)
5. **Boundary Determination**: Upstream vs downstream, operational vs financial control
6. **Completeness Screening**: Identifies which categories are relevant, detects gaps
7. **Double-Counting Prevention**: Cross-category overlap detection (DC-SCM-001 through DC-SCM-010)
8. **Calculation Approach Recommendation**: Recommends supplier-specific, hybrid, average-data, or spend-based approach per record

### 1.5 Supported Input Sources
| Source Type | Description | Primary Categories |
|-------------|-------------|-------------------|
| Spend Data | GL exports, AP ledger, procurement spend | Cat 1, 2, 4, 5, 6 |
| Purchase Orders | PO lines with item descriptions | Cat 1, 2, 3, 4 |
| Bill of Materials | BOM items with material types | Cat 1, 10, 12 |
| Travel Records | Booking systems, expense reports | Cat 6, 7 |
| Fleet Data | Vehicle registrations, fuel cards | Cat 3, 7 |
| Waste Manifests | Waste transfer notes, disposal records | Cat 5 |
| Lease Agreements | Operating/finance leases | Cat 8, 13 |
| Logistics Data | Freight bills, shipping records | Cat 4, 9 |
| Product Sales | Revenue by product, distribution data | Cat 9, 10, 11, 12 |
| Investment Portfolio | Holdings, fund allocations | Cat 15 |
| Franchise Agreements | Franchise fees, royalty records | Cat 14 |
| Energy Invoices | Electricity, gas, steam, cooling bills | Cat 3 |
| Supplier Data | Supplier emissions reports, questionnaires | Cat 1, 2, 4 |

---

## 2. Regulatory Requirements

### 2.1 GHG Protocol Scope 3 Standard
- **Chapter 2**: Companies shall identify and categorize Scope 3 emissions into 15 categories
- **Chapter 4**: Screening to identify relevant categories (materiality threshold)
- **Chapter 5**: Setting Scope 3 boundary -- upstream vs downstream categorization
- **Decision Tree**: Figure 1.2 -- determining which categories apply to a company
- **Appendix A**: Category descriptions and boundaries for all 15 categories

### 2.2 Category Boundary Definitions (GHG Protocol)
| Category | Boundary | Direction | Reporter Role |
|----------|----------|-----------|---------------|
| Cat 1 | Purchased goods & services | Upstream | Buyer |
| Cat 2 | Capital goods | Upstream | Buyer |
| Cat 3 | Fuel & energy (not Scope 1/2) | Upstream | Consumer |
| Cat 4 | Upstream transportation | Upstream | Buyer (pays freight) |
| Cat 5 | Waste generated in operations | Upstream | Generator |
| Cat 6 | Business travel | Upstream | Employer |
| Cat 7 | Employee commuting | Upstream | Employer |
| Cat 8 | Upstream leased assets | Upstream | Lessee |
| Cat 9 | Downstream transportation | Downstream | Seller |
| Cat 10 | Processing of sold products | Downstream | Seller |
| Cat 11 | Use of sold products | Downstream | Seller |
| Cat 12 | End-of-life treatment | Downstream | Seller |
| Cat 13 | Downstream leased assets | Downstream | Lessor |
| Cat 14 | Franchises | Downstream | Franchisor |
| Cat 15 | Investments | Downstream | Investor |

### 2.3 Compliance Frameworks
| Framework | Mapping Requirement |
|-----------|-------------------|
| GHG Protocol Scope 3 | All 15 categories must be screened; material categories reported |
| ISO 14064-1:2018 | Clause 5.2 indirect emissions categorization |
| CSRD / ESRS E1 | E1-6 value chain categorization |
| CDP Climate Change | C6.5 category-level reporting |
| SBTi FLAG / SBTi-FI | Category-specific target requirements |
| SB 253 (California) | All material Scope 3 categories with assurance |
| SEC Climate Rule | Material Scope 3 disclosure |
| ISSB S2 (IFRS) | Value chain categorization |

### 2.4 Double-Counting Rules (DC-SCM-001 through DC-SCM-010)
| Rule ID | Description |
|---------|-------------|
| DC-SCM-001 | Cat 1 vs Cat 2: Purchased goods (opex) vs capital goods (capex) -- use capitalization policy |
| DC-SCM-002 | Cat 1 vs Cat 4: Goods cost vs freight cost -- split by Incoterm |
| DC-SCM-003 | Cat 3 vs Scope 2: WTT and T&D losses must exclude Scope 2 reported amounts |
| DC-SCM-004 | Cat 4 vs Cat 9: Upstream vs downstream transport -- split at point of sale |
| DC-SCM-005 | Cat 6 vs Cat 7: Business travel vs commuting -- do not double-count commute on travel days |
| DC-SCM-006 | Cat 8 vs Scope 1/2: Leased assets -- consolidation approach determines boundary |
| DC-SCM-007 | Cat 10 vs Cat 11: Processing vs use of sold products -- sequential, no overlap |
| DC-SCM-008 | Cat 11 vs Cat 12: Use vs end-of-life -- product lifetime boundary |
| DC-SCM-009 | Cat 13 vs Scope 1/2: Lessor perspective -- exclude consolidated assets |
| DC-SCM-010 | Cat 14 vs Cat 15: Franchise vs investment -- use agreement type as determinant |

---

## 3. Architecture

### 3.1 Seven-Engine Design

| # | Engine | Class | Responsibility |
|---|--------|-------|---------------|
| 1 | Category Database | `CategoryDatabaseEngine` | NAICS/ISIC/UNSPSC/HS code mappings, spend category lookups, keyword dictionaries |
| 2 | Spend Classifier | `SpendClassifierEngine` | Deterministic spend classification using GL codes, procurement categories, industry codes |
| 3 | Activity Router | `ActivityRouterEngine` | Routes classified data to correct category agent (MRV-014 through MRV-028), transforms inputs |
| 4 | Boundary Determiner | `BoundaryDeterminerEngine` | Upstream/downstream determination, consolidation approach analysis, Incoterm routing |
| 5 | Completeness Screener | `CompletenessScreenerEngine` | Category relevance screening, gap analysis, materiality assessment across all 15 categories |
| 6 | Compliance Checker | `ComplianceCheckerEngine` | Multi-framework compliance validation for category mapping completeness |
| 7 | Category Mapper Pipeline | `CategoryMapperPipelineEngine` | End-to-end 10-stage orchestration pipeline |

### 3.2 Engine Details

#### Engine 1: Category Database (`CategoryDatabaseEngine`)
- **NAICS 2022 Mapping**: 1,057 6-digit NAICS codes → primary Scope 3 category + secondary categories
- **ISIC Rev 4 Mapping**: 419 4-digit ISIC codes → Scope 3 categories (international classification)
- **UNSPSC v26 Mapping**: 55,000+ commodity codes → categories (procurement systems)
- **HS 2022 Mapping**: 5,600+ tariff codes → categories (trade/customs)
- **GL Account Mapping**: 200+ standard GL account ranges → categories
- **Spend Keyword Dictionary**: 500+ keywords → categories with confidence weights
- **Cross-reference tables**: NAICS↔ISIC, UNSPSC↔NAICS concordance tables
- **Version tracking**: Each mapping table has version, effective date, source authority

#### Engine 2: Spend Classifier (`SpendClassifierEngine`)
- **Priority hierarchy**: (1) Industry code → (2) GL account → (3) Procurement category → (4) Keyword match → (5) Default
- **Multi-category splitting**: A single PO line can map to multiple categories (e.g., "IT equipment + installation" → Cat 2 + Cat 1)
- **Confidence scoring**: 0.0--1.0 based on mapping method (industry code=0.95, keyword=0.40)
- **Calculation approach recommendation**: Based on data quality, recommends supplier-specific/hybrid/average/spend-based
- **Currency normalization**: 40+ currencies to base currency (USD/EUR) for spend-based calculations
- **Batch processing**: Up to 50,000 records per batch with streaming results

#### Engine 3: Activity Router (`ActivityRouterEngine`)
- **Agent registry**: Maps each Scope 3 category to its agent module and API endpoint
- **Input transformation**: Converts generic classified records into category-specific input models
- **Routing table**: Cat 1→MRV-014 (PGS), Cat 2→MRV-015 (CG), ... Cat 15→MRV-028 (INV)
- **Bulk routing**: Groups classified records by category and dispatches batch requests
- **Fallback routing**: Records that cannot be classified go to a review queue
- **Dry-run mode**: Returns routing plan without executing category agents

#### Engine 4: Boundary Determiner (`BoundaryDeterminerEngine`)
- **Consolidation approach**: Operational control vs financial control vs equity share
- **Upstream/downstream split**: Based on reporter's position in value chain
- **Incoterm routing**: For Cat 4 vs Cat 9 boundary (FOB/CIF/DDP etc.)
- **Capitalization policy**: For Cat 1 vs Cat 2 boundary (opex vs capex threshold)
- **Lease classification**: For Cat 8 vs Scope 1/2 boundary (IFRS 16 / ASC 842)
- **Investment consolidation**: For Cat 15 vs Scope 1/2 boundary (equity share thresholds)

#### Engine 5: Completeness Screener (`CompletenessScreenerEngine`)
- **Category relevance matrix**: Based on company type (manufacturer, services, financial, retailer, etc.)
- **Data availability assessment**: Which categories have data available vs required estimation
- **Materiality screening**: Identifies categories likely to be >1% of total Scope 3
- **Gap analysis**: Missing categories, incomplete data coverage within categories
- **Recommended actions**: Prioritized list of data collection actions to improve completeness
- **Industry benchmarks**: Expected category distribution by industry sector (GICS/NAICS)

#### Engine 6: Compliance Checker (`ComplianceCheckerEngine`)
- **Framework-specific requirements**: Minimum categories required per framework
- **Category-level compliance**: Data quality checks per category per framework
- **Completeness thresholds**: Minimum number of categories reported
- **Disclosure requirements**: Which categories need narrative explanation if excluded
- **Assurance readiness**: Limited vs reasonable assurance requirements
- **Improvement plan**: Year-over-year data quality improvement tracking

#### Engine 7: Category Mapper Pipeline (`CategoryMapperPipelineEngine`)
- **Stage 1**: Input validation and normalization
- **Stage 2**: Data source classification (spend/PO/BOM/travel/etc.)
- **Stage 3**: Industry code lookup (NAICS/ISIC/UNSPSC/HS)
- **Stage 4**: Spend classification (GL/procurement/keyword)
- **Stage 5**: Boundary determination (upstream/downstream)
- **Stage 6**: Double-counting check (DC-SCM rules)
- **Stage 7**: Multi-category splitting
- **Stage 8**: Calculation approach recommendation
- **Stage 9**: Completeness screening
- **Stage 10**: Provenance hashing and output assembly

### 3.3 Package Layout
```
greenlang/scope3_category_mapper/
├── __init__.py                    # Module init, exports, metadata
├── config.py                      # Thread-safe singleton configuration
├── models.py                      # Pydantic models (25+ enums, 30+ models)
├── metrics.py                     # Prometheus metrics (gl_scm_ prefix)
├── provenance.py                  # SHA-256 provenance chain (10-stage)
├── category_database.py           # Engine 1: Classification code mappings
├── spend_classifier.py            # Engine 2: Deterministic spend classification
├── activity_router.py             # Engine 3: Category agent routing
├── boundary_determiner.py         # Engine 4: Upstream/downstream boundaries
├── completeness_screener.py       # Engine 5: Category completeness analysis
├── compliance_checker.py          # Engine 6: Multi-framework compliance
├── category_mapper_pipeline.py    # Engine 7: End-to-end pipeline
└── setup.py                       # FastAPI router + service facade
```

### 3.4 Test Layout
```
tests/unit/mrv/test_scope3_category_mapper/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_models.py                 # ~80 tests
├── test_config.py                 # ~50 tests
├── test_category_database.py      # ~120 tests
├── test_spend_classifier.py       # ~100 tests
├── test_activity_router.py        # ~80 tests
├── test_boundary_determiner.py    # ~80 tests
├── test_completeness_screener.py  # ~70 tests
├── test_compliance_checker.py     # ~60 tests
├── test_category_mapper_pipeline.py # ~80 tests
├── test_provenance.py             # ~40 tests
├── test_setup.py                  # ~30 tests
└── test_api.py                    # ~40 tests
```
Target: **830+ tests**

---

## 4. Data Models

### 4.1 Enumerations (25)
1. `Scope3Category` -- 15 GHG Protocol categories
2. `DataSourceType` -- 13 input source types (spend, PO, BOM, travel, fleet, waste, lease, logistics, product_sales, investment, franchise, energy, supplier)
3. `ClassificationMethod` -- 6 methods (naics, isic, unspsc, hs_code, gl_account, keyword)
4. `CalculationApproach` -- 4 approaches (supplier_specific, hybrid, average_data, spend_based)
5. `ConfidenceLevel` -- 5 levels (very_high, high, medium, low, very_low)
6. `ConsolidationApproach` -- 3 approaches (operational_control, financial_control, equity_share)
7. `ValueChainPosition` -- 2 positions (upstream, downstream)
8. `CategoryRelevance` -- 4 levels (material, relevant, not_relevant, unknown)
9. `CompanyType` -- 8 types (manufacturer, services, financial, retailer, energy, mining, agriculture, transport)
10. `IncotermsRule` -- 11 rules (EXW, FCA, CPT, CIP, DAP, DPU, DDP, FAS, FOB, CFR, CIF)
11. `LeaseClassification` -- 4 types (operating_lease, finance_lease, short_term, low_value)
12. `CapitalizationPolicy` -- 3 policies (capitalize, expense, threshold_based)
13. `CurrencyCode` -- 40+ ISO 4217 currencies
14. `GWPVersion` -- 4 versions (AR4, AR5, AR6, SAR)
15. `ComplianceFramework` -- 8 frameworks
16. `DataQualityTier` -- 5 tiers (1=best to 5=worst)
17. `MappingStatus` -- 5 statuses (mapped, split, unmapped, review_required, excluded)
18. `DoubleCounting Rule` -- 10 rules (DC-SCM-001 to DC-SCM-010)
19. `MaterialityThreshold` -- 3 levels (quantitative_1pct, qualitative_high, de_minimis)
20. `NAICSLevel` -- 4 levels (sector_2, subsector_3, industry_group_4, industry_6)
21. `ISICLevel` -- 4 levels (section_1, division_2, group_3, class_4)
22. `UNSPSCLevel` -- 4 levels (segment_2, family_4, class_6, commodity_8)
23. `RoutingAction` -- 4 actions (route, split_route, queue_review, exclude)
24. `ScreeningResult` -- 3 results (complete, partial, missing)
25. `PipelineStage` -- 10 stages

### 4.2 Core Input Models
- `ClassificationInput` -- Single record for classification
- `BatchClassificationInput` -- Batch of records (up to 50,000)
- `SpendRecord` -- Spend data with GL code, NAICS, amount, currency
- `PurchaseOrderRecord` -- PO with line items
- `BOMRecord` -- Bill of materials item
- `TravelRecord` -- Business travel / commuting record
- `WasteRecord` -- Waste manifest record
- `LeaseRecord` -- Lease agreement details
- `LogisticsRecord` -- Freight / shipping record
- `InvestmentRecord` -- Investment holding
- `EnergyRecord` -- Energy invoice

### 4.3 Core Output Models
- `ClassificationResult` -- Single classification with category, confidence, approach
- `BatchClassificationResult` -- Batch results with aggregation
- `RoutingPlan` -- Routing instructions for category agents
- `BoundaryDetermination` -- Upstream/downstream analysis
- `CompletenessReport` -- Category coverage assessment
- `DoubleCounting Check` -- Cross-category overlap analysis
- `ComplianceAssessment` -- Framework compliance status
- `CategorySummary` -- Aggregated stats per category

---

## 5. Classification Code Databases

### 5.1 NAICS 2022 → Scope 3 Category Mapping
| NAICS Range | Industry | Primary Category | Secondary |
|-------------|----------|-----------------|-----------|
| 11xxxx | Agriculture | Cat 1 | Cat 10 |
| 21xxxx | Mining | Cat 1 | Cat 3 |
| 22xxxx | Utilities | Cat 3 | -- |
| 23xxxx | Construction | Cat 2 | Cat 1 |
| 31-33xxxx | Manufacturing | Cat 1 | Cat 2, 10 |
| 333xxx | Machinery Mfg | Cat 2 | -- |
| 334xxx | Electronics Mfg | Cat 2 | Cat 11 |
| 336xxx | Transport Equipment | Cat 2 | Cat 11 |
| 42xxxx | Wholesale | Cat 1 | Cat 4 |
| 44-45xxxx | Retail | Cat 1 | Cat 9 |
| 481xxx | Air Transport | Cat 6 | Cat 4 |
| 482xxx | Rail Transport | Cat 6/4 | Cat 7 |
| 484xxx | Truck Transport | Cat 4 | Cat 9 |
| 49xxxx | Warehousing | Cat 4 | Cat 9 |
| 51xxxx | Information | Cat 1 | -- |
| 52xxxx | Finance/Insurance | Cat 15 | Cat 1 |
| 53xxxx | Real Estate | Cat 8/13 | Cat 2 |
| 54xxxx | Professional Services | Cat 1 | -- |
| 562xxx | Waste Management | Cat 5 | Cat 12 |
| 72xxxx | Accommodation | Cat 6 | -- |

### 5.2 GL Account Mapping
| GL Range | Account Type | Primary Category |
|----------|-------------|-----------------|
| 5000-5199 | COGS - Materials | Cat 1 |
| 5200-5299 | COGS - Direct Labor (outsourced) | Cat 1 |
| 5300-5399 | Freight In | Cat 4 |
| 5400-5499 | Subcontractor | Cat 1 |
| 6100-6199 | Office Supplies | Cat 1 |
| 6200-6299 | IT & Software | Cat 1 |
| 6300-6399 | Professional Services | Cat 1 |
| 6400-6499 | Travel & Entertainment | Cat 6 |
| 6500-6599 | Vehicle / Fleet | Cat 3/7 |
| 6600-6699 | Utilities (electric, gas) | Cat 3 |
| 6700-6799 | Rent & Leases | Cat 8 |
| 6800-6899 | Insurance | Cat 1 |
| 7000-7999 | Capital Expenditures | Cat 2 |
| 8000-8099 | Waste Disposal | Cat 5 |
| 8100-8199 | Distribution / Outbound | Cat 9 |

---

## 6. API Endpoints

### 6.1 Route Prefix: `/api/v1/scope3-category-mapper`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/classify` | Classify a single record |
| POST | `/classify/batch` | Classify batch of records (up to 50K) |
| POST | `/classify/spend` | Classify spend data specifically |
| POST | `/classify/purchase-orders` | Classify purchase orders |
| POST | `/classify/bom` | Classify bill of materials |
| POST | `/route` | Route classified records to category agents |
| POST | `/route/dry-run` | Preview routing without execution |
| POST | `/boundary/determine` | Determine upstream/downstream boundary |
| POST | `/completeness/screen` | Screen category completeness |
| POST | `/completeness/gap-analysis` | Detailed gap analysis |
| POST | `/double-counting/check` | Check for cross-category double counting |
| POST | `/compliance/assess` | Assess mapping compliance |
| GET | `/categories` | List all 15 categories with descriptions |
| GET | `/categories/{number}` | Get single category details |
| GET | `/codes/naics/{code}` | Look up NAICS code mapping |
| GET | `/codes/isic/{code}` | Look up ISIC code mapping |
| GET | `/codes/unspsc/{code}` | Look up UNSPSC code mapping |
| GET | `/health` | Service health check |
| GET | `/metrics` | Prometheus metrics |

---

## 7. Database Schema (V080)

### 7.1 Tables (prefix: `gl_scm_`)
1. `gl_scm_classification_runs` -- Master run table
2. `gl_scm_classification_results` -- Individual classification results
3. `gl_scm_routing_plans` -- Routing instructions
4. `gl_scm_routing_executions` -- Executed routing batches
5. `gl_scm_boundary_determinations` -- Boundary analysis results
6. `gl_scm_completeness_reports` -- Category completeness assessments
7. `gl_scm_double_counting_checks` -- DC overlap analysis
8. `gl_scm_compliance_assessments` -- Framework compliance results
9. `gl_scm_naics_mappings` -- NAICS code → category reference
10. `gl_scm_isic_mappings` -- ISIC code → category reference
11. `gl_scm_category_summaries` -- Aggregated per-category stats

### 7.2 Hypertables (TimescaleDB)
- `gl_scm_classification_results` (time column: `classified_at`)
- `gl_scm_routing_executions` (time column: `routed_at`)
- `gl_scm_compliance_assessments` (time column: `assessed_at`)

### 7.3 Continuous Aggregates
- `gl_scm_hourly_classification_stats` -- Classifications per hour by category
- `gl_scm_daily_routing_stats` -- Routing volume per day by category

---

## 8. Metrics (Prometheus)

### 8.1 Prefix: `gl_scm_`
| Metric | Type | Description |
|--------|------|-------------|
| `gl_scm_classifications_total` | Counter | Total records classified (by category, method, confidence) |
| `gl_scm_classification_duration_seconds` | Histogram | Classification latency |
| `gl_scm_batch_size` | Histogram | Batch sizes processed |
| `gl_scm_routing_total` | Counter | Records routed (by category, action) |
| `gl_scm_routing_duration_seconds` | Histogram | Routing latency |
| `gl_scm_double_counting_detected_total` | Counter | Double-counting detections (by rule) |
| `gl_scm_unmapped_records_total` | Counter | Records that could not be classified |
| `gl_scm_confidence_score` | Histogram | Distribution of confidence scores |
| `gl_scm_completeness_score` | Gauge | Current completeness score (0-100) |
| `gl_scm_categories_active` | Gauge | Number of active categories |
| `gl_scm_compliance_score` | Gauge | Framework compliance score (by framework) |
| `gl_scm_errors_total` | Counter | Processing errors (by type) |

---

## 9. Provenance Chain

### 9.1 Ten-Stage SHA-256 Pipeline
| Stage | Hash Input | Description |
|-------|-----------|-------------|
| P1 | Raw input record | Input data fingerprint |
| P2 | Code lookup result | NAICS/ISIC/UNSPSC lookup |
| P3 | Classification decision | Category + method + confidence |
| P4 | Boundary determination | Upstream/downstream result |
| P5 | DC check result | Double-counting analysis |
| P6 | Split allocation | Multi-category split ratios |
| P7 | Approach recommendation | Calculation approach |
| P8 | Routing instruction | Target agent + transformed input |
| P9 | Completeness contribution | Category coverage impact |
| P10 | Final chain hash | SHA-256(P1 || P2 || ... || P9) |

---

## 10. Zero-Hallucination Guarantees

1. **All mappings use deterministic lookup tables** -- NO LLM/ML involvement in classification
2. **Industry codes from authoritative sources** -- NAICS (Census Bureau), ISIC (UN), UNSPSC (GS1), HS (WCO)
3. **Complete provenance hash** for every classification decision
4. **Confidence scoring** reflects mapping method quality (code-based > keyword-based)
5. **Double-counting rules** are deterministic, based on GHG Protocol guidance
6. **Boundary determination** uses explicit rules (capitalization policy, Incoterms, lease type)
7. **No probabilistic classification** -- every record gets a deterministic category assignment
8. **Fallback to review queue** rather than guessing when confidence is below threshold

---

## 11. Performance Requirements

| Metric | Target |
|--------|--------|
| Single record classification | < 5ms |
| Batch classification (10K records) | < 2s |
| Batch classification (50K records) | < 10s |
| Routing plan generation | < 1s |
| Completeness screening | < 500ms |
| Double-counting check | < 1s |
| API endpoint p99 latency | < 200ms |
| Throughput | 10,000 records/sec |
| Memory usage (50K batch) | < 512MB |

---

## 12. Integration Points

### 12.1 Upstream Dependencies
| Component | Integration |
|-----------|-------------|
| AGENT-DATA-002 (Excel/CSV) | Normalized spend data input |
| AGENT-DATA-003 (ERP) | ERP-extracted PO and GL data |
| AGENT-DATA-008 (Questionnaire) | Supplier classification data |
| AGENT-DATA-009 (Spend Categorizer) | Pre-categorized spend data |
| AGENT-FOUND-001 (Orchestrator) | DAG-based pipeline orchestration |

### 12.2 Downstream Consumers
| Component | Integration |
|-----------|-------------|
| AGENT-MRV-014 | Cat 1: Purchased Goods & Services |
| AGENT-MRV-015 | Cat 2: Capital Goods |
| AGENT-MRV-016 | Cat 3: Fuel & Energy Activities |
| AGENT-MRV-017 | Cat 4: Upstream Transportation |
| AGENT-MRV-018 | Cat 5: Waste Generated |
| AGENT-MRV-019 | Cat 6: Business Travel |
| AGENT-MRV-020 | Cat 7: Employee Commuting |
| AGENT-MRV-021 | Cat 8: Upstream Leased Assets |
| AGENT-MRV-022 | Cat 9: Downstream Transportation |
| AGENT-MRV-023 | Cat 10: Processing of Sold Products |
| AGENT-MRV-024 | Cat 11: Use of Sold Products |
| AGENT-MRV-025 | Cat 12: End-of-Life Treatment |
| AGENT-MRV-026 | Cat 13: Downstream Leased Assets |
| AGENT-MRV-027 | Cat 14: Franchises |
| AGENT-MRV-028 | Cat 15: Investments |

---

## 13. Acceptance Criteria

### 13.1 Functional
- [ ] Correctly maps all NAICS 2-digit sectors to primary Scope 3 categories
- [ ] Correctly maps NAICS 3-digit subsectors with specialized mappings (e.g., 481→Cat 6)
- [ ] Handles ISIC Rev 4 codes with cross-reference to NAICS
- [ ] GL account ranges correctly map to categories
- [ ] Multi-category splitting produces correct allocation ratios
- [ ] Double-counting rules DC-SCM-001 through DC-SCM-010 all enforced
- [ ] Boundary determination handles all 3 consolidation approaches
- [ ] Completeness screener identifies all relevant categories for 8 company types
- [ ] Compliance checker validates against all 8 frameworks
- [ ] Routing correctly targets all 15 category agents

### 13.2 Non-Functional
- [ ] 830+ passing tests
- [ ] Zero LLM/ML dependencies in classification logic
- [ ] 10-stage provenance chain produces deterministic hashes
- [ ] Thread-safe configuration singleton
- [ ] Prometheus metrics with `gl_scm_` prefix registered
- [ ] API endpoints secured via JWT + RBAC
- [ ] DB migration V080 applies cleanly

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NAICS code coverage gaps | Unmapped records | Keyword fallback + review queue |
| Multi-category splitting accuracy | Incorrect allocation | Conservative default (100% to primary) |
| Category boundary ambiguity | Misclassification | Explicit capitalization/Incoterm/lease rules |
| Performance with 50K batches | Timeout | Streaming results, parallel code lookups |
| Double-counting complexity | Over/under-reporting | 10 explicit DC rules with deterministic checks |

---

## 15. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-28 | Initial release -- 7 engines, 830+ tests, 10 DC rules |
