# PRD-PACK-042: Scope 3 Starter Pack

**Pack ID:** PACK-042-scope-3-starter
**Category:** GHG Accounting Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Production Ready
**Author:** GreenLang Product Team
**Date:** 2026-03-24
**Prerequisite:** None (standalone; enhanced with PACK-041 Scope 1-2 Complete Pack, PACK-031-040 Energy Efficiency Packs, and PACK-021-030 Net Zero Packs when present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Scope 3 (value chain) emissions constitute 70-90% of total corporate GHG footprints for most sectors (CDP, 2023), yet organizations face extraordinary challenges in measuring, managing, and reporting them. The GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard defines 15 categories spanning upstream and downstream activities, but practical implementation remains the single largest barrier to credible climate disclosure. Organizations face eight persistent challenges:

1. **Overwhelming category complexity**: Scope 3 encompasses 15 categories -- Purchased Goods & Services (Cat 1), Capital Goods (Cat 2), Fuel & Energy-Related Activities (Cat 3), Upstream Transportation & Distribution (Cat 4), Waste Generated in Operations (Cat 5), Business Travel (Cat 6), Employee Commuting (Cat 7), Upstream Leased Assets (Cat 8), Downstream Transportation & Distribution (Cat 9), Processing of Sold Products (Cat 10), Use of Sold Products (Cat 11), End-of-Life Treatment of Sold Products (Cat 12), Downstream Leased Assets (Cat 13), Franchises (Cat 14), and Investments (Cat 15). Most organizations attempt 3-5 categories and abandon the effort, resulting in incomplete inventories that fail regulatory and verification standards.

2. **Methodology tier paralysis**: Each Scope 3 category supports three calculation tiers -- spend-based (Tier 1, using economic input-output models and EEIO emission factors), average-data (Tier 2, using industry-average physical emission factors), and supplier-specific (Tier 3, using primary data from suppliers). Organizations struggle to select appropriate tiers for each category based on data availability, materiality, and improvement roadmap. Most default to spend-based for everything, producing low-accuracy estimates (±50-200% uncertainty) that fail to identify true emission hotspots.

3. **Double-counting across categories**: Scope 3 categories have inherent overlaps -- Cat 3 (fuel & energy) partially overlaps with Cat 1 (purchased goods) for energy procurement; Cat 4 (upstream transport) overlaps with Cat 1 when suppliers include logistics in product prices; Cat 9 (downstream transport) may overlap with Cat 4 for intermediary companies. Without systematic double-counting prevention, organizations overstate total Scope 3 by 5-25%, undermining credibility with verifiers and stakeholders.

4. **Spend data classification failures**: Spend-based calculations require mapping every procurement transaction to the correct EEIO sector code (NAICS, ISIC, UNSPSC, HS). A typical mid-sized company processes 10,000-100,000 unique procurement line items annually. Manual classification achieves 60-75% accuracy; unclassified spend defaults to high-emission sectors, systematically inflating estimates. Without deterministic classification using validated code mappings, spend-based inventories are unreliable.

5. **Supplier engagement bottlenecks**: Transitioning from spend-based to supplier-specific data requires engaging hundreds or thousands of suppliers. Response rates for supplier carbon data requests average 15-30%. Without structured engagement workflows, standardized data request templates, response validation, and quality scoring, supplier data programs stall after 1-2 cycles.

6. **Hotspot identification failures**: Organizations need to focus improvement efforts on the 3-5 Scope 3 categories and supplier segments that drive 80% of value chain emissions. Without Pareto analysis, materiality screening (ESRS double materiality), and sector benchmarking, organizations spread resources thinly across all categories instead of targeting the highest-impact opportunities.

7. **Multi-framework Scope 3 requirements**: ESRS E1 (CSRD) requires Scope 3 disclosure with phase-in through 2029; CDP Climate Change requires Scope 3 for A-list scoring; SBTi requires Scope 3 targets for companies where Scope 3 exceeds 40% of total emissions; SEC Climate Rules reference Scope 3 materiality; California SB 253 mandates Scope 3 from 2027. Each framework has different boundary, materiality, and disclosure requirements for Scope 3. Manual reconciliation across frameworks requires 100-300 hours per reporting cycle.

8. **Verification readiness gaps**: Scope 3 verification under ISO 14064-3 and ISAE 3410 requires evidence of data quality, methodology selection rationale, completeness assessment, and uncertainty quantification for each category. Most Scope 3 inventories lack the audit trail depth required for reasonable assurance, limiting organizations to limited assurance at best and increasing verification costs by 30-50%.

### 1.2 Solution Overview

PACK-042 is the **Scope 3 Starter Pack** -- the second pack in the "GHG Accounting Packs" category. It provides a structured, step-by-step approach to Scope 3 emissions measurement by orchestrating all 17 existing Scope 3 MRV agents (15 category agents + Category Mapper + Audit Trail) into an integrated pack with screening workflows, hotspot analysis, supplier engagement tools, and multi-framework reporting.

The pack is designed as a "starter" pack because it:
- Guides organizations through their first complete Scope 3 inventory
- Supports all three methodology tiers with progressive data quality improvement
- Provides screening-level estimates first, then refines to higher accuracy
- Includes supplier engagement workflows to systematically upgrade from spend-based to supplier-specific data
- Focuses on practical hotspot identification to prioritize improvement efforts

The pack orchestrates:
- **15 Scope 3 category agents**: Purchased Goods & Services (MRV-014), Capital Goods (MRV-015), Fuel & Energy-Related Activities (MRV-016), Upstream Transportation (MRV-017), Waste in Operations (MRV-018), Business Travel (MRV-019), Employee Commuting (MRV-020), Upstream Leased Assets (MRV-021), Downstream Transportation (MRV-022), Processing of Sold Products (MRV-023), Use of Sold Products (MRV-024), End-of-Life Treatment (MRV-025), Downstream Leased Assets (MRV-026), Franchises (MRV-027), Investments (MRV-028)
- **2 cross-cutting agents**: Category Mapper (MRV-029), Audit Trail & Lineage (MRV-030)

The pack adds 10 pack-level engines, 8 workflows, 10 templates, 12 integrations, and 8 presets that provide:
- Scope 3 screening and relevance assessment per GHG Protocol Scope 3 Standard Chapter 6
- Category completeness scanning with materiality thresholds
- Multi-tier methodology management (spend-based, average-data, supplier-specific) per category
- Spend data classification using NAICS/ISIC/UNSPSC/HS deterministic mapping
- Double-counting prevention with cross-category reconciliation
- Hotspot analysis with Pareto ranking and sector benchmarking
- Supplier engagement scoring and data quality tracking
- Value chain mapping with upstream/downstream boundary visualization
- Multi-framework Scope 3 compliance mapping (GHG Protocol, ESRS E1, CDP, SBTi, SEC, SB 253)
- Uncertainty quantification at category and total Scope 3 level
- Full audit trail with SHA-256 provenance hashing on every calculation

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet Approach | PACK-042 Scope 3 Starter Pack |
|-----------|-------------------------------|-------------------------------|
| Category coverage | 3-5 of 15 categories (20-33%) | All 15 categories with relevance screening (100% coverage) |
| Time to first Scope 3 inventory | 3-6 months | <2 weeks for screening, <6 weeks for full inventory (4-6x faster) |
| Cost per inventory cycle | EUR 50,000-200,000 (consultant + internal) | EUR 5,000-15,000 per cycle (10x reduction) |
| Methodology tier selection | Ad hoc, often all spend-based | Guided tier selection per category with data maturity roadmap |
| Spend classification accuracy | 60-75% manual mapping | >95% deterministic NAICS/ISIC/UNSPSC mapping |
| Double-counting prevention | Ad hoc checks (5-25% overstatement risk) | Systematic cross-category reconciliation (<1% residual overlap) |
| Hotspot identification | Subjective assessment | Pareto analysis with sector benchmarking and materiality screening |
| Supplier engagement | Unstructured, 15-30% response rate | Structured workflows, standardized templates, quality scoring |
| Multi-framework reporting | Manual reconciliation (100-300 hours) | Automated mapping to 6+ frameworks (<8 hours) |
| Uncertainty quantification | Point estimates only | Monte Carlo + analytical propagation with 95% CI per category |
| Audit trail | Spreadsheet-based | SHA-256 provenance chain, ISAE 3410-ready audit package |
| Data quality improvement | No structured path | Progressive tier upgrade roadmap with ROI quantification |

### 1.4 Scope 3 Category Overview

**Upstream Emissions (Categories 1-8):**

| # | Category | Agent | Key Emission Sources | Typical Share |
|---|----------|-------|---------------------|---------------|
| 1 | Purchased Goods & Services | MRV-014 | Raw materials, components, office supplies, professional services | 30-60% of Scope 3 |
| 2 | Capital Goods | MRV-015 | Machinery, vehicles, buildings, IT equipment | 2-10% of Scope 3 |
| 3 | Fuel & Energy-Related Activities | MRV-016 | T&D losses, upstream fuel extraction, WTT emissions | 3-8% of Scope 3 |
| 4 | Upstream Transportation & Distribution | MRV-017 | Inbound logistics, warehousing (paid by reporter) | 3-10% of Scope 3 |
| 5 | Waste Generated in Operations | MRV-018 | Landfill, recycling, incineration, wastewater | 1-5% of Scope 3 |
| 6 | Business Travel | MRV-019 | Air travel, rail, hotel stays, car rental | 1-5% of Scope 3 |
| 7 | Employee Commuting | MRV-020 | Personal vehicles, public transit, remote work | 1-5% of Scope 3 |
| 8 | Upstream Leased Assets | MRV-021 | Leased offices, equipment, vehicles (not in Scope 1/2) | 0-3% of Scope 3 |

**Downstream Emissions (Categories 9-15):**

| # | Category | Agent | Key Emission Sources | Typical Share |
|---|----------|-------|---------------------|---------------|
| 9 | Downstream Transportation | MRV-022 | Outbound logistics (paid by customer or included in price) | 2-8% of Scope 3 |
| 10 | Processing of Sold Products | MRV-023 | Energy used by intermediaries processing products | 0-15% of Scope 3 |
| 11 | Use of Sold Products | MRV-024 | Energy consumed by products during use phase | 5-60% of Scope 3 |
| 12 | End-of-Life Treatment | MRV-025 | Disposal, recycling, incineration of sold products | 1-5% of Scope 3 |
| 13 | Downstream Leased Assets | MRV-026 | Assets leased to tenants (landlord reporting) | 0-10% of Scope 3 |
| 14 | Franchises | MRV-027 | Franchise operations (franchisor reporting) | 0-20% of Scope 3 |
| 15 | Investments | MRV-028 | Equity, debt, project finance emissions | 0-80% of Scope 3 (financial sector) |

**Cross-Cutting:**

| ID | Agent | Function |
|----|-------|----------|
| MRV-029 | Category Mapper | Routes spend/activity data to correct Scope 3 category |
| MRV-030 | Audit Trail & Lineage | Immutable audit trail, calculation lineage, evidence packaging |

### 1.5 Target Users

**Primary:**
- Sustainability managers conducting their first comprehensive Scope 3 inventory
- GHG reporting specialists needing to expand from Scope 1-2 to full value chain coverage
- Procurement teams responsible for supplier carbon data collection
- Corporate sustainability directors preparing for CSRD/ESRS Scope 3 disclosure requirements

**Secondary:**
- ESG consultants helping clients build Scope 3 measurement programs
- SBTi target-setting teams needing Scope 3 baselines for net-zero commitments
- CDP respondents improving Scope 3 disclosure quality for A-list scoring
- Supply chain managers identifying carbon hotspots in their value chain
- CFOs and investor relations teams preparing for SEC/SB 253 Scope 3 requirements
- Industry associations developing sector-specific Scope 3 benchmarks

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to screening-level Scope 3 estimate | <2 weeks (vs. 3-6 months manual) | Time from data intake to 15-category screening results |
| Category coverage | 15/15 categories assessed | Count of categories with quantified estimates |
| Spend classification accuracy | >95% correct NAICS/ISIC mapping | Validated against manual expert classification sample |
| Double-counting residual | <1% of total Scope 3 | Cross-category reconciliation audit |
| Hotspot identification accuracy | Top 3 categories match detailed audit | Validated against full Tier 3 inventory |
| Supplier data request response rate | >50% (vs. 15-30% baseline) | Response tracking over 2 engagement cycles |
| Multi-framework compliance | 100% mapping to GHG Protocol, ESRS, CDP | Verified against framework requirement checklists |
| Uncertainty quantification | 95% CI for each category and total | Monte Carlo results with documented methodology |
| Verification readiness | Limited assurance achievable on first cycle | Verifier feedback on audit package completeness |
| Customer NPS | >55 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| GHG Protocol Scope 3 Standard | Corporate Value Chain (Scope 3) Accounting and Reporting Standard (WRI/WBCSD, 2011) | Core methodology: 15 category definitions, boundary setting, calculation approaches, data quality guidance |
| GHG Protocol Technical Guidance | Scope 3 Calculation Guidance (2013) | Detailed calculation methods per category, emission factor sources, worked examples |
| ISO 14064-1:2018 | GHG quantification at organization level | Clause 5.2.4 indirect GHG emissions, category significance assessment |
| ESRS E1 (CSRD) | European Sustainability Reporting Standards | E1-6 para 44-46 Scope 3 disclosure with category breakdown, phase-in through 2029 |
| CDP Climate Change 2025 | Carbon Disclosure Project Questionnaire | C6.5 Scope 3 emissions by category, C6.7 methodology, C6.10 data quality |
| SBTi Corporate Net-Zero | Science Based Targets initiative | Scope 3 target required when >40% of total; near-term: 2.5% annual reduction; long-term: 90% absolute by 2050 |

### 2.2 Supporting Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| US SEC Climate Disclosure | Regulation S-K Item 1505 | Scope 3 disclosure when material (safe harbor provisions) |
| California SB 253 | Climate Corporate Data Accountability Act | Scope 3 reporting mandatory from 2027 for companies with >$1B revenue |
| PCAF Global Standard | Partnership for Carbon Accounting Financials | Category 15 (Investments) methodology for financial institutions |
| GLEC Framework | Global Logistics Emissions Council Framework v3.0 | Categories 4 and 9 (upstream/downstream transport) methodology |
| Quantis Scope 3 Evaluator | WRI/Quantis screening tool | Spend-based screening methodology reference |
| EEIO Models | Exiobase 3, USEEIO 2.0, GTAP | Economic input-output emission factors for spend-based calculations |
| ISO 14040/14044 | Life Cycle Assessment | Category 1, 10, 11, 12 product lifecycle methodology |
| GHG Protocol Scope 2 Guidance | Market-based and location-based | Category 3 (fuel & energy-related) upstream energy methodology |

### 2.3 Emission Factor Sources

| Source | Coverage | Categories |
|--------|----------|------------|
| Exiobase 3 (2024) | 200 product sectors × 49 countries | Cat 1, 2 (spend-based) |
| USEEIO 2.0 | 400 sectors (US economy) | Cat 1, 2 (spend-based, US operations) |
| DEFRA 2025 | UK government conversion factors | All categories (activity-based) |
| EPA 2024 | US emission factors for all GHGs | All categories (US operations) |
| ecoinvent 3.10 | 21,000+ LCI datasets | Cat 1, 10, 11, 12 (product LCA) |
| IEA 2024 | Country-level grid emission factors | Cat 3 (T&D losses, WTT) |
| GLEC v3.0 | Transport emission factors by mode | Cat 4, 9 (transport) |
| ICAO Carbon Calculator | Aviation emission factors | Cat 6 (business travel air) |
| UK BEIS Hotel Factors | Per night-stay emission factors | Cat 6 (business travel hotels) |
| National Waste Factors | Country-specific waste treatment EFs | Cat 5 (waste) |

---

## 3. Technical Architecture

### 3.1 System Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PACK-042: Scope 3 Starter Pack                      │
│                                                                         │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │   10 Engines  │  │   8 Workflows     │  │   10 Templates           │  │
│  │   (pack-level)│  │   (orchestration) │  │   (output generation)    │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────────┬─────────────┘  │
│         │                   │                          │                 │
│  ┌──────┴───────────────────┴──────────────────────────┴─────────────┐  │
│  │                    12 Integrations                                 │  │
│  │  (MRV bridges, data intake, supplier mgmt, ERP, health check)     │  │
│  └──────┬────────────────────┬─────────────────────────┬─────────────┘  │
└─────────┼────────────────────┼─────────────────────────┼─────────────────┘
          │                    │                          │
   ┌──────┴──────┐      ┌─────┴──────┐           ┌──────┴──────┐
   │ MRV Scope 3 │      │ AGENT-DATA │           │ AGENT-FOUND │
   │ 014-030     │      │ 001-020    │           │ 001-010     │
   │ (17 agents) │      │ (20 agents)│           │ (10 agents) │
   └─────────────┘      └────────────┘           └─────────────┘
```

### 3.2 Component Summary

| Component | Count | Total Lines (est.) | Purpose |
|-----------|-------|--------------------|---------|
| Engines | 10 | ~15,000 | Pack-level calculation and analysis |
| Workflows | 8 | ~8,500 | Multi-phase orchestration |
| Templates | 10 | ~7,500 | Report and disclosure generation |
| Integrations | 12 | ~8,500 | Agent bridges, data intake, supplier tools |
| Presets | 8 | ~3,000 | Sector-specific configurations |
| Config | 2 | ~1,200 | Pack configuration and enums |
| Tests | ~20 | ~8,000 | Unit, integration, e2e, performance |
| Migrations | 10 | ~4,500 | Database schema V336-V345 |
| **Total** | **~80** | **~56,000** | |

### 3.3 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | All engines, workflows, templates |
| Models | Pydantic v2 | Configuration, data models, validation |
| Arithmetic | `decimal.Decimal` | Financial and emission calculations (no float) |
| Statistics | `numpy` | Monte Carlo simulation, statistical analysis |
| Hash | `hashlib.sha256` | Provenance chain, audit trail |
| Database | PostgreSQL 16 + TimescaleDB | Time-series audit data, hypertables |
| YAML | `pyyaml` | Pack manifest, preset configurations |
| Testing | `pytest` | Unit, integration, e2e test suites |

---

## 4. Engine Specifications

### 4.1 Engine 1: Scope 3 Screening Engine (`scope3_screening_engine.py`)

**Purpose:** Perform rapid screening-level Scope 3 assessment per GHG Protocol Scope 3 Standard Chapter 6. Determines relevance, estimated magnitude, and recommended methodology tier for each of the 15 categories.

**Key Features:**
- 15-category relevance assessment using sector-specific decision trees
- Spend-based screening estimates using EEIO emission factors (Exiobase 3, USEEIO 2.0)
- Revenue-based screening for downstream categories (Cat 10-12)
- Category significance ranking (% of estimated total Scope 3)
- Data availability assessment per category
- Recommended methodology tier per category based on materiality and data maturity
- Sector-specific screening profiles for 25+ NAICS 2-digit sectors

**Inputs:** Organization sector (NAICS/ISIC), revenue, employee count, procurement spend total, facility count, product/service type
**Outputs:** 15-category screening results with estimated tCO2e, % share, relevance flag, recommended tier, data requirements

### 4.2 Engine 2: Spend Classification Engine (`spend_classification_engine.py`)

**Purpose:** Deterministically classify procurement spend data into Scope 3 categories and EEIO sectors for spend-based calculations.

**Key Features:**
- Multi-code classification: NAICS 2022 (1,057 codes), ISIC Rev 4 (419 codes), UNSPSC v26 (55,000+ codes), HS 2022 (5,600+ codes)
- GL account-based classification (200+ standard account ranges)
- Keyword-based fallback classification for uncoded transactions
- Automatic Scope 3 category assignment from sector codes
- Confidence scoring (high/medium/low) for each classification
- Split transaction handling (single invoice → multiple categories)
- Currency normalization (50+ currencies to base currency using ECB/Fed rates)
- Inflation adjustment for multi-year spend analysis (CPI-based deflators)
- Integration with MRV-029 Category Mapper for deterministic routing

**Inputs:** Procurement transaction data (vendor, description, amount, GL code, NAICS/ISIC if available)
**Outputs:** Classified transactions with Scope 3 category, EEIO sector, emission factor, confidence score

### 4.3 Engine 3: Category Consolidation Engine (`category_consolidation_engine.py`)

**Purpose:** Consolidate emissions from all 15 Scope 3 categories into a unified inventory with per-category and aggregate totals.

**Key Features:**
- Per-category total emissions (tCO2e) with gas-level breakdown (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)
- Weighted average methodology tier per category
- Data quality scoring per category (1-5 scale per GHG Protocol Data Quality Indicators)
- Upstream vs. downstream subtotals
- Scope 3 as percentage of total footprint (requires Scope 1+2 from PACK-041 if available)
- Year-over-year comparison with base year reference
- Consolidation across multiple facilities/entities
- Boundary alignment with organizational boundary from PACK-041

**Inputs:** Per-category results from MRV-014 through MRV-028
**Outputs:** Consolidated Scope 3 inventory with category breakdown, data quality metrics, aggregated totals

### 4.4 Engine 4: Double-Counting Prevention Engine (`double_counting_engine.py`)

**Purpose:** Detect and resolve double-counting across Scope 3 categories, between Scope 3 and Scope 1/2, and within upstream/downstream boundaries.

**Key Features:**
- 12 defined overlap rules:
  1. Cat 1 ↔ Cat 3 (energy procurement in goods/services vs. fuel & energy)
  2. Cat 1 ↔ Cat 4 (logistics included in supplier prices)
  3. Cat 1 ↔ Cat 2 (capitalized vs. expensed goods)
  4. Cat 3 ↔ Scope 2 (upstream energy already in market-based Scope 2)
  5. Cat 4 ↔ Cat 9 (transport cost allocation between buyer/seller)
  6. Cat 8 ↔ Scope 1/2 (leased assets vs. operational control boundary)
  7. Cat 13 ↔ Cat 11 (downstream leased ↔ use of sold products)
  8. Cat 14 ↔ Scope 1/2 (franchise ↔ operational control)
  9. Cat 1 ↔ Cat 5 (packaging waste in purchased goods)
  10. Cat 10 ↔ Cat 11 (processing intermediary ↔ end use)
  11. Cat 6 ↔ Cat 7 (business travel car ↔ commuting personal vehicle)
  12. Cat 15 ↔ Cat 13/14 (investment ↔ leased/franchise for financial institutions)
- Automated flagging with resolution recommendations
- Adjustable allocation rules (conservative vs. proportional)
- Net double-counting impact quantification (tCO2e removed)
- Audit trail for every adjustment

**Inputs:** Per-category emission results, Scope 1/2 data (if available), boundary definition
**Outputs:** Adjusted category totals, double-counting flags, resolution log, net adjustment summary

### 4.5 Engine 5: Hotspot Analysis Engine (`hotspot_analysis_engine.py`)

**Purpose:** Identify emission hotspots across Scope 3 categories, supplier segments, and product lines to prioritize reduction efforts.

**Key Features:**
- Pareto analysis (80/20 rule): categories, suppliers, products driving majority of emissions
- Sector benchmarking against CDP, GHG Protocol, and industry-specific databases
- Materiality matrix (emissions magnitude × data quality × improvement potential)
- Supplier segmentation: top N suppliers by emission contribution
- Product carbon intensity ranking (tCO2e per unit, per revenue, per kg)
- Geographic emission mapping (country-level supplier emissions)
- Reduction opportunity quantification (tier upgrade impact, supplier engagement ROI)
- SBTi-aligned reduction pathway modeling per category
- Time-series hotspot tracking (which hotspots are growing/declining)

**Inputs:** Consolidated Scope 3 inventory, supplier data, product data, sector benchmarks
**Outputs:** Ranked hotspot list, Pareto chart data, materiality scores, reduction opportunities

### 4.6 Engine 6: Supplier Engagement Engine (`supplier_engagement_engine.py`)

**Purpose:** Manage supplier carbon data collection, quality scoring, and progressive engagement to upgrade from spend-based to supplier-specific data.

**Key Features:**
- Supplier prioritization: rank suppliers by emission contribution and engagement readiness
- Data request template generation: standardized questionnaires per industry
- Response tracking: sent, opened, in-progress, completed, overdue status
- Data quality scoring: 5-level quality assessment per GHG Protocol data quality indicators
  - Level 1: No data (use EEIO estimate)
  - Level 2: Spend-based with general sector EF
  - Level 3: Average-data with product-specific EF
  - Level 4: Supplier-reported aggregate emissions (allocated by revenue)
  - Level 5: Supplier-specific product-level LCA data
- Progressive engagement roadmap: pathway from Level 1 → Level 5 over 3-5 years
- CDP Supply Chain integration: extract data from CDP supplier responses
- Engagement ROI calculator: emission accuracy improvement per dollar of engagement effort
- Automated reminder scheduling and escalation

**Inputs:** Supplier list, procurement data, historical engagement data, category mapping
**Outputs:** Prioritized supplier engagement plan, data quality dashboard, response tracking

### 4.7 Engine 7: Data Quality Assessment Engine (`data_quality_engine.py`)

**Purpose:** Assess and track data quality across all Scope 3 categories using GHG Protocol Data Quality Indicators (DQI).

**Key Features:**
- 5 data quality indicators per GHG Protocol Technical Guidance:
  1. Technological representativeness (1-5 scale)
  2. Temporal representativeness (1-5 scale)
  3. Geographical representativeness (1-5 scale)
  4. Completeness (% of category covered by data)
  5. Reliability (primary vs. secondary vs. estimated data)
- Weighted Data Quality Rating (DQR) per category and overall
- Data quality improvement roadmap with prioritized actions
- Year-over-year data quality trend tracking
- Minimum quality thresholds per framework (ESRS requires "reasonable" quality)
- Gap analysis: what data is missing and impact on accuracy
- Quality score contribution to overall uncertainty quantification

**Inputs:** Category data sources, emission factor metadata, supplier data quality levels
**Outputs:** DQR scores per category, quality trend data, improvement roadmap, gap list

### 4.8 Engine 8: Scope 3 Uncertainty Engine (`scope3_uncertainty_engine.py`)

**Purpose:** Quantify uncertainty in Scope 3 estimates using Monte Carlo simulation and analytical methods, reflecting the inherently higher uncertainty of value chain emissions.

**Key Features:**
- Per-category uncertainty ranges:
  - Spend-based: ±50-200% (per EEIO model limitations)
  - Average-data: ±20-50% (per DEFRA/EPA guidance)
  - Supplier-specific: ±5-20% (per primary data quality)
- Monte Carlo simulation: 10,000+ iterations per category
- Analytical error propagation: IPCC Approach 1 quadrature for aggregation
- Correlation handling between categories (e.g., Cat 1 and Cat 4 share supply chain)
- 95% confidence interval at category and total Scope 3 level
- Data quality impact: uncertainty reduction from tier upgrades quantified
- Sensitivity analysis: which input parameters drive most uncertainty
- Comparison to sector benchmark uncertainty ranges

**Inputs:** Per-category emission estimates, methodology tiers, data quality scores, correlation matrix
**Outputs:** Uncertainty ranges (95% CI), Monte Carlo distribution, sensitivity rankings, tier upgrade impact

### 4.9 Engine 9: Scope 3 Compliance Mapping Engine (`scope3_compliance_engine.py`)

**Purpose:** Map Scope 3 inventory results to 6+ regulatory frameworks and generate framework-specific disclosure data.

**Key Features:**
- Framework requirement mapping:
  1. **GHG Protocol Scope 3 Standard**: All 15 categories, boundary, methodology, data quality
  2. **ESRS E1 (CSRD)**: E1-6 para 44-46, Scope 3 with phase-in (2025: Cat 1-3; 2029: all)
  3. **CDP Climate Change**: C6.5 per-category disclosure, C6.7 methodology, C6.10 data quality
  4. **SBTi Net-Zero**: Scope 3 screening, target setting (FLAG, absolute, intensity)
  5. **SEC Climate Rules**: Material Scope 3 disclosure (safe harbor), methodology description
  6. **California SB 253**: Mandatory Scope 3 from 2027, per-category reporting
- Gap analysis per framework: what's missing for compliance
- Compliance score per framework (0-100%)
- Action plan generation for closing compliance gaps
- Cross-framework reconciliation: one dataset, multiple output formats

**Inputs:** Consolidated Scope 3 inventory, framework configuration, data quality scores
**Outputs:** Per-framework compliance scores, gap analyses, action plans, disclosure-ready data

### 4.10 Engine 10: Scope 3 Reporting Engine (`scope3_reporting_engine.py`)

**Purpose:** Generate comprehensive Scope 3 reports, disclosures, and verification packages in multiple output formats.

**Key Features:**
- 10 report types:
  1. Full Scope 3 Inventory Report (all 15 categories)
  2. Category Deep-Dive Report (per selected category)
  3. Executive Summary (2-4 pages, C-suite level)
  4. Hotspot Analysis Report (Pareto charts, materiality matrix)
  5. Supplier Engagement Report (engagement status, data quality)
  6. Data Quality Assessment Report (DQR scores, improvement roadmap)
  7. Compliance Dashboard (multi-framework readiness)
  8. Uncertainty Analysis Report (Monte Carlo results, sensitivity)
  9. Year-over-Year Trend Report (emissions trajectory, reduction progress)
  10. Verification Package (ISO 14064-3 / ISAE 3410 evidence bundle)
- 4 output formats: Markdown, HTML (self-contained CSS), JSON, CSV
- SHA-256 provenance hash on every report
- XBRL tagging for ESRS E1 digital taxonomy
- Automated appendix: methodology notes, emission factor sources, assumptions log

**Inputs:** Consolidated inventory, compliance mapping, hotspot analysis, data quality scores
**Outputs:** Reports in selected formats with provenance hashing

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Scope 3 Screening Workflow (`scope3_screening_workflow.py`)

**Purpose:** Rapid screening-level assessment of all 15 Scope 3 categories.

**Phases:**
1. **Organization Profile**: Collect sector, revenue, employee count, product types
2. **Spend Data Intake**: Import procurement data, classify by category
3. **Screening Calculation**: Run spend-based estimates for all 15 categories using EEIO factors
4. **Relevance Assessment**: Rank categories by magnitude, flag relevant categories

**Duration:** 2-4 hours
**Output:** 15-category screening report with relevance flags and recommended next steps

### 5.2 Workflow 2: Category Data Collection Workflow (`category_data_collection_workflow.py`)

**Purpose:** Structured data collection for each relevant Scope 3 category.

**Phases:**
1. **Category Selection**: User selects categories for detailed analysis (guided by screening)
2. **Data Requirements**: Generate data requirements checklist per category and methodology tier
3. **Data Intake**: Collect activity data via forms, file upload, ERP integration, or API
4. **Data Validation**: Validate completeness, units, date ranges, and plausibility

**Duration:** 1-3 weeks per category (depends on data availability)
**Output:** Validated category data ready for calculation

### 5.3 Workflow 3: Category Calculation Workflow (`category_calculation_workflow.py`)

**Purpose:** Execute emission calculations for selected categories using appropriate MRV agents.

**Phases:**
1. **Methodology Selection**: Confirm tier for each category (spend-based, average-data, supplier-specific)
2. **Agent Routing**: Route to correct MRV agent (MRV-014 through MRV-028) via MRV-029
3. **Calculation Execution**: Run per-category calculations with provenance tracking
4. **Result Validation**: Cross-check results against sector benchmarks, flag outliers

**Duration:** 1-4 hours per category
**Output:** Per-category emission results with provenance chain

### 5.4 Workflow 4: Consolidation & Double-Counting Workflow (`consolidation_workflow.py`)

**Purpose:** Consolidate all category results and resolve double-counting.

**Phases:**
1. **Category Aggregation**: Sum per-category results into Scope 3 total
2. **Double-Counting Check**: Run 12-rule double-counting detection engine
3. **Scope Integration**: Integrate with Scope 1+2 (PACK-041) if available for full footprint
4. **Final Reconciliation**: Produce reconciled Scope 3 total with audit trail

**Duration:** 1-2 hours
**Output:** Consolidated, reconciled Scope 3 inventory

### 5.5 Workflow 5: Hotspot & Prioritization Workflow (`hotspot_workflow.py`)

**Purpose:** Identify emission hotspots and prioritize reduction opportunities.

**Phases:**
1. **Pareto Analysis**: Rank categories, suppliers, products by emission contribution
2. **Materiality Assessment**: Score each hotspot on magnitude, data quality, reduction potential
3. **Benchmarking**: Compare against sector averages and best practices
4. **Action Planning**: Generate prioritized reduction roadmap with ROI estimates

**Duration:** 2-4 hours
**Output:** Hotspot report with prioritized reduction plan

### 5.6 Workflow 6: Supplier Engagement Workflow (`supplier_engagement_workflow.py`)

**Purpose:** Manage supplier carbon data collection and engagement.

**Phases:**
1. **Supplier Prioritization**: Rank suppliers by emission contribution (top 80% of procurement spend)
2. **Data Request Generation**: Create standardized questionnaires per supplier segment
3. **Response Collection**: Track responses, send reminders, validate submissions
4. **Quality Assessment**: Score supplier data quality, identify upgrade opportunities

**Duration:** Ongoing (quarterly cycle)
**Output:** Supplier engagement dashboard, updated data quality scores

### 5.7 Workflow 7: Disclosure Generation Workflow (`disclosure_workflow.py`)

**Purpose:** Generate framework-specific Scope 3 disclosures.

**Phases:**
1. **Framework Selection**: Choose target frameworks (GHG Protocol, ESRS, CDP, SBTi, SEC, SB 253)
2. **Compliance Mapping**: Map inventory data to framework requirements
3. **Gap Analysis**: Identify and flag missing requirements per framework
4. **Disclosure Output**: Generate framework-specific reports and data exports

**Duration:** 2-4 hours
**Output:** Framework-specific disclosure documents

### 5.8 Workflow 8: Full Scope 3 Pipeline Workflow (`full_scope3_pipeline_workflow.py`)

**Purpose:** End-to-end Scope 3 inventory from screening through disclosure.

**Phases:**
1. **Screening**: Run scope3_screening_workflow
2. **Data Collection**: Run category_data_collection_workflow for relevant categories
3. **Calculation**: Run category_calculation_workflow for all categories
4. **Consolidation**: Run consolidation_workflow with double-counting resolution
5. **Hotspot Analysis**: Run hotspot_workflow
6. **Data Quality**: Assess data quality and generate improvement plan
7. **Uncertainty**: Run Monte Carlo uncertainty analysis
8. **Disclosure**: Run disclosure_workflow for target frameworks

**Duration:** 2-6 weeks (full first-time Scope 3 inventory)
**Output:** Complete Scope 3 inventory with all reports and disclosures

---

## 6. Template Specifications

### 6.1 Template 1: Scope 3 Inventory Report (`scope3_inventory_report.py`)

Full 15-category Scope 3 inventory with per-category breakdown, methodology, data quality, uncertainty, and compliance mapping. Three render methods: `render_markdown()`, `render_html()`, `render_json()`. SHA-256 provenance hash.

### 6.2 Template 2: Category Deep-Dive Report (`category_deep_dive_report.py`)

Detailed single-category report with sub-category breakdown, emission factor sources, supplier contributions, methodology description, and data quality assessment. Parameterized for any of 15 categories.

### 6.3 Template 3: Executive Summary Report (`scope3_executive_summary.py`)

2-4 page C-suite summary with total Scope 3 tCO2e, top 5 categories chart, year-over-year trend, % of total footprint, SBTi alignment, and 3 priority actions.

### 6.4 Template 4: Hotspot Analysis Report (`hotspot_report.py`)

Pareto analysis with waterfall chart data, materiality matrix, supplier concentration analysis, geographic distribution, and prioritized reduction opportunities with ROI.

### 6.5 Template 5: Supplier Engagement Report (`supplier_engagement_report.py`)

Supplier engagement status dashboard: engagement rates, data quality distribution, tier upgrade progress, top supplier profiles, and engagement ROI metrics.

### 6.6 Template 6: Data Quality Assessment Report (`data_quality_report.py`)

DQR scores per category with spider/radar chart data, quality trend over time, gap analysis, improvement roadmap with effort/impact scoring.

### 6.7 Template 7: Compliance Dashboard (`scope3_compliance_dashboard.py`)

Multi-framework compliance readiness: GHG Protocol, ESRS, CDP, SBTi, SEC, SB 253. Per-framework score, gap list, and action items.

### 6.8 Template 8: Uncertainty Analysis Report (`scope3_uncertainty_report.py`)

Monte Carlo results with probability distribution data, 95% CI per category and total, sensitivity tornado chart, tier upgrade impact quantification.

### 6.9 Template 9: Verification Package (`scope3_verification_package.py`)

ISO 14064-3 / ISAE 3410 evidence bundle: methodology summary, data sources, emission factor registry, calculation log, assumption register, provenance chain, materiality assessment.

### 6.10 Template 10: ESRS E1 Scope 3 Disclosure (`esrs_e1_scope3_disclosure.py`)

ESRS E1-6 para 44-46 formatted Scope 3 disclosure with XBRL taxonomy alignment, category breakdown, methodology description, data quality statement, and phase-in compliance.

---

## 7. Integration Specifications

### 7.1 Integration 1: Pack Orchestrator (`pack_orchestrator.py`)

12-phase DAG pipeline orchestrating all pack components from screening through disclosure. Phase dependencies, parallel execution where possible, checkpoint/resume capability.

### 7.2 Integration 2: MRV Scope 3 Bridge (`mrv_scope3_bridge.py`)

Routes calculations to 15 Scope 3 category agents (MRV-014 through MRV-028). Handles agent discovery, input transformation, result collection, and error handling. Supports parallel execution of independent categories.

### 7.3 Integration 3: Category Mapper Bridge (`category_mapper_bridge.py`)

Integration with MRV-029 (Category Mapper) for deterministic routing of spend/activity data to correct Scope 3 category using NAICS/ISIC/UNSPSC/HS code mappings.

### 7.4 Integration 4: Audit Trail Bridge (`audit_trail_bridge.py`)

Integration with MRV-030 (Audit Trail & Lineage) for immutable audit logging, calculation lineage DAG, evidence packaging, and compliance traceability.

### 7.5 Integration 5: Data Bridge (`data_bridge.py`)

Routes data intake to AGENT-DATA agents: PDF Extractor (DATA-001), Excel Normalizer (DATA-002), ERP Connector (DATA-003), API Gateway (DATA-004), Spend Categorizer (DATA-009), Data Quality Profiler (DATA-010), Data Lineage (DATA-018).

### 7.6 Integration 6: Foundation Bridge (`foundation_bridge.py`)

Routes to AGENT-FOUND agents: Orchestrator (FOUND-001), Schema Validator (FOUND-002), Unit Normalizer (FOUND-003), Assumptions Registry (FOUND-004), Citations (FOUND-005), Access Guard (FOUND-006), Reproducibility (FOUND-008), Observability (FOUND-010).

### 7.7 Integration 7: Scope 1-2 Bridge (`scope12_bridge.py`)

Integration with PACK-041 (Scope 1-2 Complete Pack) for full GHG footprint view. Retrieves Scope 1+2 totals for % of total calculation, Cat 3 alignment, boundary consistency.

### 7.8 Integration 8: EEIO Factor Bridge (`eeio_factor_bridge.py`)

Integration with EEIO emission factor databases (Exiobase 3, USEEIO 2.0) for spend-based calculations. Handles sector mapping, currency conversion, and inflation adjustment.

### 7.9 Integration 9: ERP Connector (`erp_connector.py`)

Integration with enterprise ERP systems (SAP, Oracle, Microsoft Dynamics, NetSuite) for procurement/spend data extraction, GL account mapping, and vendor master data.

### 7.10 Integration 10: Health Check (`health_check.py`)

22-category system verification: agent availability, database connectivity, emission factor freshness, configuration validity, data pipeline status.

### 7.11 Integration 11: Setup Wizard (`setup_wizard.py`)

8-step guided configuration: organization profile, sector selection, category relevance, methodology tiers, data sources, framework targets, notification preferences, initial screening.

### 7.12 Integration 12: Alert Bridge (`alert_bridge.py`)

Multi-channel notifications: data collection reminders, supplier response deadlines, calculation completion, compliance deadline alerts, data quality warnings.

---

## 8. Database Migrations (V336-V345)

### 8.1 V336: Core Scope 3 Schema

Organizations, reporting periods, Scope 3 inventories, category configurations, methodology tier settings. Tenant-isolated with RLS.

### 8.2 V337: Spend Classification Tables

Spend transactions, EEIO sector mappings, classification results, confidence scores, NAICS/ISIC/UNSPSC code tables with seed data.

### 8.3 V338: Category Results Tables

Per-category emission results (15 tables), gas-level breakdown, emission factor references, methodology metadata.

### 8.4 V339: Double-Counting Prevention

Overlap rules, detection results, resolution log, adjusted totals, cross-category reconciliation records.

### 8.5 V340: Hotspot Analysis Tables

Pareto results, materiality scores, sector benchmarks, supplier rankings, reduction opportunities.

### 8.6 V341: Supplier Engagement Tables

Supplier profiles, engagement status, data requests, responses, quality scores, progressive roadmaps.

### 8.7 V342: Data Quality Tables

DQR scores (5 indicators × 15 categories), quality trends, gap analysis, improvement actions.

### 8.8 V343: Uncertainty Analysis Tables

Monte Carlo results, confidence intervals, sensitivity rankings, correlation matrices.

### 8.9 V344: Compliance & Reporting Tables

Framework assessments, compliance scores, gap analysis, report metadata, verification packages, audit trail (hypertable).

### 8.10 V345: Views, Indexes, Seed Data

Materialized views for dashboards, composite indexes for query performance, EEIO seed data, sector benchmark seed data, RBAC policies.

---

## 9. Non-Functional Requirements

### 9.1 Performance

| Metric | Target |
|--------|--------|
| Screening (15 categories) | <30 seconds |
| Single category calculation | <60 seconds |
| Full 15-category calculation | <10 minutes |
| Monte Carlo (10,000 iterations) | <45 seconds |
| Report generation | <15 seconds per report |
| Spend classification (10,000 transactions) | <60 seconds |
| Supplier engagement batch (500 suppliers) | <30 seconds |

### 9.2 Security

- Row-Level Security (RLS) with tenant_id on all tables
- SHA-256 provenance hashing on all calculation outputs
- Encrypted supplier data at rest (AES-256-GCM)
- RBAC: 6 roles (admin, analyst, reviewer, supplier_contact, auditor, viewer)
- Audit logging via MRV-030 integration

### 9.3 Scalability

- Support up to 500,000 procurement transactions per organization
- Support up to 10,000 suppliers per engagement program
- Support up to 100 facilities per inventory
- Horizontal scaling via agent-per-category parallelism

---

## 10. Testing Strategy

### 10.1 Unit Tests (~500+ test functions)

- Per-engine test files validating all calculation logic
- Spend classification accuracy tests against known mappings
- Double-counting detection tests against defined overlap scenarios
- Hotspot Pareto calculation validation
- Data quality scoring validation

### 10.2 Integration Tests

- Workflow phase transition tests
- MRV agent bridge routing tests
- Data pipeline end-to-end tests
- Cross-pack integration with PACK-041

### 10.3 Compliance Formula Tests

- GHG Protocol Scope 3 reference calculations per Technical Guidance worked examples
- EEIO factor multiplication validation
- Transport emission factor validation (GLEC v3.0)
- Activity-based factor validation (DEFRA 2025)

### 10.4 Performance Tests

- Screening performance benchmarks
- Spend classification throughput
- Monte Carlo execution time
- Report generation performance
- Database query performance with large datasets

### 10.5 End-to-End Tests

- Manufacturing company full Scope 3 inventory
- Financial services company (Cat 15 dominant)
- Retail company (Cat 1 dominant)
- Technology company (Cat 11 dominant)
- SME simplified pathway

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| EEIO | Environmentally Extended Input-Output model |
| DQI | Data Quality Indicator (GHG Protocol) |
| DQR | Data Quality Rating (weighted average of DQIs) |
| Exiobase | Multi-regional EEIO database (EU-funded) |
| USEEIO | US Environmentally Extended Input-Output model (EPA) |
| NAICS | North American Industry Classification System |
| ISIC | International Standard Industrial Classification |
| UNSPSC | United Nations Standard Products and Services Code |
| HS | Harmonized System (trade classification) |
| GLEC | Global Logistics Emissions Council |
| PCAF | Partnership for Carbon Accounting Financials |
| FLAG | Forest, Land and Agriculture (SBTi guidance) |
| WTT | Well-to-Tank (upstream fuel emissions) |
| T&D | Transmission and Distribution (electricity losses) |
| LCA | Life Cycle Assessment |
| LCI | Life Cycle Inventory |

---

## 12. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-24 | GreenLang Product Team | Initial release |
