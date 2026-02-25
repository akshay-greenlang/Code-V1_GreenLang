# PRD: AGENT-MRV-013 — Scope 2 Dual Reporting Reconciliation Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-X-024 |
| **Internal Label** | AGENT-MRV-013 |
| **Category** | Layer 3 — MRV / Accounting Agents (Scope 2) |
| **Package** | `greenlang/dual_reporting_reconciliation/` |
| **DB Migration** | V064 |
| **Metrics Prefix** | `gl_drr_` |
| **Table Prefix** | `drr_` |
| **API** | `/api/v1/dual-reporting` |
| **Env Prefix** | `GL_DRR_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The Dual Reporting Reconciliation Agent is the **capstone Scope 2 agent** that
reconciles location-based and market-based Scope 2 results per GHG Protocol
Scope 2 Guidance Chapter 7. It consumes outputs from four upstream agents:

- **MRV-009**: Scope 2 Location-Based (grid EFs, eGRID, IEA, T&D loss)
- **MRV-010**: Scope 2 Market-Based (contractual instruments, RECs, GOs, residual mix)
- **MRV-011**: Steam/Heat Purchase (steam, district heating, CHP allocation)
- **MRV-012**: Cooling Purchase (electric chillers, absorption, district cooling, TES)

This agent does NOT calculate emissions itself — it **collects, reconciles,
scores, and reports** dual Scope 2 results across all energy types.

### Justification for Dedicated Agent

1. **GHG Protocol mandates dual reporting** — Chapter 7 requires organizations in
   markets with contractual instruments to report BOTH location-based and market-based
2. **CSRD requires two total GHG figures** — (S1+S2_loc+S3) and (S1+S2_mkt+S3)
3. **CDP A-list eligibility** requires dual reporting for maximum scoring
4. **SBTi v5.3** requires both methods; draft v2 mandates dual targets
5. **Discrepancy analysis** needs cross-agent coordination impossible within a single agent
6. **Quality scoring** spans completeness, consistency, accuracy, transparency
7. **Trend analysis** requires multi-year comparison across both methods

### Standards & References

- GHG Protocol Scope 2 Guidance (2015) — Chapter 7: Dual Reporting
- GHG Protocol Scope 2 Guidance Table 6.1 — Location vs Market comparison
- CSRD/ESRS E1-6 — Paragraphs 49a-49b, dual Scope 2 disclosure
- CDP Climate Change Questionnaire (2024) — C6.3/C6.4, Route A scoring
- SBTi Corporate Manual v5.3 — Scope 2 dual method targets
- GRI 305-2 — Energy indirect GHG emissions
- ISO 14064-1:2018 — Category 2 quantification
- RE100 Technical Criteria — Renewable electricity tracking

---

## 2. Dual Reporting Methodology

### 2.1 Core Concept

Dual reporting presents **two views of the same physical energy consumption**:
- **Location-based**: Uses grid-average emission factors reflecting the actual
  generation mix at the point of consumption
- **Market-based**: Uses emission factors from contractual instruments (RECs,
  GOs, PPAs, supplier EFs) reflecting purchasing decisions

The two results are NEVER summed. They enable stakeholders to understand both
the physical emissions profile and the impact of renewable energy procurement.

### 2.2 Upstream Agent Mapping

| Energy Type | Location-Based Agent | Market-Based Agent |
|-------------|---------------------|-------------------|
| Electricity | MRV-009 | MRV-010 |
| Steam | MRV-011 (fuel-based EF) | MRV-011 (supplier EF if available) |
| District Heating | MRV-011 (regional EF) | MRV-011 (supplier EF if available) |
| District Cooling | MRV-012 (COP × grid EF) | MRV-012 (supplier EF if available) |

### 2.3 Discrepancy Types (8)

| Type | Description | Direction |
|------|-------------|-----------|
| REC/GO Impact | RECs reduce market-based below location | Market << Location |
| Residual Mix Uplift | Residual mix higher than grid average | Market > Location |
| Supplier-Specific EF | Supplier EF differs from grid average | Either direction |
| Geographic Mismatch | Instruments from different grid region | Either direction |
| Temporal Mismatch | Instruments from different vintage year | Either direction |
| Partial Coverage | Some energy covered by instruments, some not | Partial effect |
| Steam/Heat Method | Different allocation for non-electricity | Either direction |
| Grid EF Update Timing | Location uses newer EFs than market residual | Small difference |

### 2.4 Materiality Thresholds

| Level | Threshold | Action Required |
|-------|-----------|----------------|
| Immaterial | < 5% difference | Note in report |
| Minor | 5-15% difference | Explain in methodology |
| Material | 15-50% difference | Detailed waterfall decomposition |
| Significant | 50-100% difference | Board-level disclosure |
| Extreme | > 100% difference | Verification recommended |

### 2.5 Quality Scoring (4 Dimensions)

| Dimension | Weight | Score Range | Description |
|-----------|--------|-------------|-------------|
| Completeness | 0.30 | 0.0-1.0 | All energy types covered in both methods |
| Consistency | 0.25 | 0.0-1.0 | Same boundaries, periods, GWP sources |
| Accuracy | 0.25 | 0.0-1.0 | EF quality, tier levels, uncertainty bounds |
| Transparency | 0.20 | 0.0-1.0 | Documentation, provenance, audit trail |

Composite Quality = Σ(dimension_score × weight)

| Grade | Score | Label |
|-------|-------|-------|
| A | ≥ 0.90 | Assurance-Ready |
| B | ≥ 0.80 | High Quality |
| C | ≥ 0.65 | Acceptable |
| D | ≥ 0.50 | Needs Improvement |
| F | < 0.50 | Insufficient |

### 2.6 Key Formulas

**Procurement Impact Factor (PIF):**
```
PIF = (Location_tCO2e - Market_tCO2e) / Location_tCO2e × 100
```
Positive PIF = market-based purchasing reduces emissions.

**Discrepancy Percentage:**
```
Discrepancy_Pct = |Location_tCO2e - Market_tCO2e| / Location_tCO2e × 100
```

**RE100 Progress:**
```
RE100_Pct = Renewable_MWh / Total_Electricity_MWh × 100
```

**Composite Quality Score:**
```
Quality = 0.30 × Completeness + 0.25 × Consistency + 0.25 × Accuracy + 0.20 × Transparency
```

**Year-over-Year Change:**
```
YoY_Change_Pct = (Current_tCO2e - Previous_tCO2e) / Previous_tCO2e × 100
```

**SBTi Linear Target Trajectory:**
```
Target_Year_Emissions = Base_Year × (1 - Annual_Reduction_Rate) ^ Years_Elapsed
```

**Emission Intensity (per revenue):**
```
Intensity = Total_Scope2_tCO2e / Revenue_MUSD
```

**Waterfall Decomposition:**
```
Total_Discrepancy = Σ(REC_Impact + Residual_Uplift + Supplier_Delta + Coverage_Gap + Other)
```

### 2.7 Residual Mix Factors (30+ regions)

| Region | Grid Average (kgCO2e/kWh) | Residual Mix (kgCO2e/kWh) | Ratio |
|--------|--------------------------|---------------------------|-------|
| Norway | 0.008 | 0.360 | 45.0x |
| Sweden | 0.012 | 0.350 | 29.2x |
| France | 0.052 | 0.420 | 8.1x |
| Germany | 0.380 | 0.520 | 1.4x |
| UK | 0.210 | 0.350 | 1.7x |
| US PJM | 0.380 | 0.410 | 1.1x |
| US ERCOT | 0.350 | 0.380 | 1.1x |
| US WECC | 0.280 | 0.320 | 1.1x |
| Japan | 0.450 | 0.480 | 1.1x |
| Australia | 0.680 | 0.720 | 1.1x |
| India | 0.720 | 0.750 | 1.0x |
| China | 0.580 | 0.600 | 1.0x |
| Singapore | 0.410 | 0.425 | 1.0x |
| South Korea | 0.420 | 0.450 | 1.1x |
| Global Default | 0.450 | 0.500 | 1.1x |

### 2.8 EF Hierarchy (Market-Based)

| Priority | Source | Quality Score |
|----------|--------|---------------|
| 1 | Supplier-specific EF with certificates | 1.00 |
| 2 | Supplier-specific EF without certificates | 0.85 |
| 3 | Bundled certificates (REC/GO/EAC) | 0.75 |
| 4 | Unbundled certificates | 0.65 |
| 5 | Residual mix factor | 0.40 |
| 6 | Grid-average fallback | 0.20 |

### 2.9 Framework Disclosure Requirements

| Framework | Required Disclosures |
|-----------|---------------------|
| GHG Protocol | Table 6.1 dual comparison, methodology, EF sources, instrument details |
| CSRD/ESRS E1 | Para 49a (location), 49b (market), two total GHG figures, YoY trend |
| CDP C6.3/C6.4 | Location total, market total, per-country breakdown, instruments |
| SBTi | Both methods for target tracking, base year recalculation |
| GRI 305-2 | Location + market if instruments exist, methodology, EF sources |
| ISO 14064 | Category 2, methodology, uncertainty, verification statement |

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AGENT-MRV-013                           │
│       Dual Reporting Reconciliation Agent                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 1: DualResultCollectorEngine               │    │
│  │   - Collect location-based results (MRV-009/011/012) │
│  │   - Collect market-based results (MRV-010/011/012)   │
│  │   - Align boundaries, periods, units              │    │
│  │   - Map energy purchases to both methods          │    │
│  │   - Validate completeness (all energy types)      │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 2: DiscrepancyAnalyzerEngine               │    │
│  │   - Calculate discrepancies at 4 levels           │    │
│  │     (total, energy-type, facility, instrument)    │    │
│  │   - Classify 8 discrepancy types                  │    │
│  │   - Determine materiality level                   │    │
│  │   - Waterfall decomposition of drivers            │    │
│  │   - Flag material discrepancies                   │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 3: QualityScorerEngine                     │    │
│  │   - Completeness scoring (30% weight)             │    │
│  │   - Consistency scoring (25% weight)              │    │
│  │   - Accuracy scoring (25% weight)                 │    │
│  │   - Transparency scoring (20% weight)             │    │
│  │   - Composite quality grade (A-F)                 │    │
│  │   - Scope 2 quality criteria (GHG Protocol)       │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 4: ReportingTableGeneratorEngine           │    │
│  │   - GHG Protocol Table 6.1 format                 │    │
│  │   - CSRD/ESRS E1 dual disclosure                  │    │
│  │   - CDP C6.3/C6.4 format                          │    │
│  │   - SBTi progress table                           │    │
│  │   - GRI 305-2 format                              │    │
│  │   - ISO 14064 Category 2 table                    │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 5: TrendAnalysisEngine                     │    │
│  │   - Year-over-year change (both methods)          │    │
│  │   - CAGR calculation                              │    │
│  │   - Intensity metrics (revenue/FTE/area/unit)     │    │
│  │   - RE100 progress tracking                       │    │
│  │   - SBTi target trajectory comparison             │    │
│  │   - Procurement impact trend                      │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 6: ComplianceCheckerEngine                 │    │
│  │   - GHG Protocol Scope 2 Chapter 7               │    │
│  │   - CSRD/ESRS E1 dual disclosure                  │    │
│  │   - CDP Climate Change C6.3/C6.4                  │    │
│  │   - SBTi Corporate Manual                         │    │
│  │   - GRI 305-2                                     │    │
│  │   - ISO 14064-1                                   │    │
│  │   - RE100 Technical Criteria                      │    │
│  └──────────────────────────────────────────────────┘    │
│                         │                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Engine 7: DualReportingPipelineEngine             │    │
│  │   - 10-stage pipeline orchestration               │    │
│  │   - Batch multi-period reconciliation             │    │
│  │   - Multi-facility aggregation                    │    │
│  │   - Export (JSON/CSV/PDF-ready)                   │    │
│  │   - Provenance chain assembly                     │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
greenlang/dual_reporting_reconciliation/
├── __init__.py                          # Lazy imports, module exports
├── models.py                            # Pydantic v2 models, enums, constants
├── config.py                            # GL_DRR_ prefixed configuration
├── metrics.py                           # Prometheus metrics (gl_drr_*)
├── provenance.py                        # SHA-256 provenance chain
├── dual_result_collector.py             # Engine 1: Result collection
├── discrepancy_analyzer.py              # Engine 2: Discrepancy analysis
├── quality_scorer.py                    # Engine 3: Quality scoring
├── reporting_table_generator.py         # Engine 4: Table generation
├── trend_analysis.py                    # Engine 5: Trend analysis
├── compliance_checker.py                # Engine 6: Compliance
├── dual_reporting_pipeline.py           # Engine 7: Pipeline
├── setup.py                             # Service facade
└── api/
    ├── __init__.py                      # API package
    └── router.py                        # FastAPI REST endpoints

tests/unit/mrv/test_dual_reporting_reconciliation/
├── __init__.py
├── conftest.py
├── test_models.py
├── test_config.py
├── test_metrics.py
├── test_provenance.py
├── test_dual_result_collector.py
├── test_discrepancy_analyzer.py
├── test_quality_scorer.py
├── test_reporting_table_generator.py
├── test_trend_analysis.py
├── test_compliance_checker.py
├── test_dual_reporting_pipeline.py
├── test_setup.py
└── test_api.py

deployment/database/migrations/sql/
└── V064__dual_reporting_reconciliation_service.sql
```

### 3.3 Database Schema (V064)

14 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `drr_upstream_results` | Cached upstream agent results | Regular |
| `drr_energy_type_mapping` | Energy type to agent mapping | Dimension |
| `drr_residual_mix_factors` | 30+ regional residual mix EFs | Dimension |
| `drr_materiality_thresholds` | Configurable thresholds | Dimension |
| `drr_quality_weights` | Quality dimension weights | Dimension |
| `drr_facilities` | Facility registry | Dimension |
| `drr_reconciliations` | Reconciliation results | Hypertable |
| `drr_discrepancies` | Per-discrepancy detail | Regular |
| `drr_waterfall_items` | Waterfall decomposition items | Regular |
| `drr_quality_assessments` | Quality scoring results | Regular |
| `drr_reporting_tables` | Generated framework tables | Regular |
| `drr_trend_results` | Trend analysis results | Hypertable |
| `drr_compliance_checks` | Compliance check results | Regular |
| `drr_batch_jobs` | Batch processing jobs | Regular |
| `drr_hourly_stats` | Hourly reconciliation stats | Continuous Aggregate |
| `drr_daily_stats` | Daily reconciliation stats | Continuous Aggregate |

### 3.4 API Endpoints (20)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reconcile` | Full dual reporting reconciliation |
| POST | `/reconcile/batch` | Batch multi-period reconciliation |
| POST | `/collect` | Collect upstream results |
| GET | `/discrepancies/{reconciliation_id}` | Get discrepancy analysis |
| GET | `/discrepancies/{reconciliation_id}/waterfall` | Get waterfall decomposition |
| POST | `/quality/score` | Score data quality |
| GET | `/quality/{reconciliation_id}` | Get quality assessment |
| POST | `/report/generate` | Generate framework-specific table |
| GET | `/report/{reconciliation_id}/{framework}` | Get generated report |
| GET | `/report/frameworks` | List available frameworks |
| POST | `/trend/analyze` | Run trend analysis |
| GET | `/trend/{tenant_id}` | Get trend results |
| POST | `/compliance/check` | Run compliance check |
| GET | `/compliance/frameworks` | List compliance frameworks |
| GET | `/residual-mix/{region}` | Get residual mix factor |
| GET | `/residual-mix` | List all residual mix factors |
| POST | `/facilities` | Register facility |
| GET | `/facilities/{facility_id}` | Get facility |
| POST | `/export` | Export reconciliation (JSON/CSV) |
| GET | `/health` | Health check |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees
- All reconciliation calculations use Python `Decimal` (8 decimal places)
- No LLM calls in reconciliation path
- Every step recorded in calculation trace
- SHA-256 provenance hash for every result

### 4.2 Enumerations (22)

| Enum | Values | Description |
|------|--------|-------------|
| `EnergyType` | ELECTRICITY, STEAM, DISTRICT_HEATING, DISTRICT_COOLING | 4 Scope 2 energy types |
| `Scope2Method` | LOCATION_BASED, MARKET_BASED | Two Scope 2 methods |
| `UpstreamAgent` | MRV_009, MRV_010, MRV_011, MRV_012 | 4 upstream agents |
| `DiscrepancyType` | 8 values | REC_GO, RESIDUAL_MIX, SUPPLIER_EF, GEOGRAPHIC, TEMPORAL, PARTIAL_COVERAGE, STEAM_HEAT, GRID_UPDATE |
| `DiscrepancyDirection` | MARKET_LOWER, MARKET_HIGHER, EQUAL | Direction of discrepancy |
| `MaterialityLevel` | IMMATERIAL, MINOR, MATERIAL, SIGNIFICANT, EXTREME | 5 levels |
| `QualityDimension` | COMPLETENESS, CONSISTENCY, ACCURACY, TRANSPARENCY | 4 dimensions |
| `QualityGrade` | A, B, C, D, F | Composite grade |
| `EFHierarchyPriority` | 6 values | Supplier cert, supplier no cert, bundled, unbundled, residual, grid |
| `ReportingFramework` | GHG_PROTOCOL, CSRD_ESRS, CDP, SBTI, GRI, ISO_14064, RE100 | 7 frameworks |
| `FlagType` | WARNING, ERROR, INFO, RECOMMENDATION | 4 flag types |
| `FlagSeverity` | LOW, MEDIUM, HIGH, CRITICAL | 4 severities |
| `ReconciliationStatus` | PENDING, IN_PROGRESS, COMPLETED, FAILED | 4 statuses |
| `IntensityMetric` | REVENUE, FTE, FLOOR_AREA, PRODUCTION_UNIT | 4 intensity types |
| `TrendDirection` | INCREASING, DECREASING, STABLE | 3 trend directions |
| `PipelineStage` | 10 values | All pipeline stages |
| `ExportFormat` | JSON, CSV, EXCEL | 3 export formats |
| `ComplianceStatus` | COMPLIANT, NON_COMPLIANT, PARTIAL, NOT_APPLICABLE | 4 statuses |
| `DataQualityTier` | TIER_1, TIER_2, TIER_3 | 3 tiers |
| `GWPSource` | AR4, AR5, AR6, AR6_20YR | 4 GWP sources |
| `EmissionGas` | CO2, CH4, N2O, CO2E | 4 gases |
| `BatchStatus` | PENDING, RUNNING, COMPLETED, FAILED, PARTIAL | 5 statuses |

### 4.3 Regulatory Frameworks (7)

1. **GHG Protocol Scope 2 Guidance** — Chapter 7 dual reporting, Table 6.1
2. **CSRD/ESRS E1** — Paragraphs 49a-49b, dual total GHG figures
3. **CDP Climate Change** — C6.3/C6.4, Route A scoring
4. **SBTi Corporate Manual** — Dual method target tracking
5. **GRI 305-2** — Energy indirect GHG emissions
6. **ISO 14064-1:2018** — Category 2 quantification
7. **RE100 Technical Criteria** — Renewable electricity progress

### 4.4 Performance Targets

| Metric | Target |
|--------|--------|
| Single reconciliation | < 200ms |
| Batch (12 months) | < 2s |
| Quality scoring | < 50ms |
| Table generation (single framework) | < 100ms |
| Trend analysis (5 years) | < 500ms |
| Compliance check (all frameworks) | < 200ms |

---

## 5. Acceptance Criteria

- [ ] Collect and align results from 4 upstream agents (MRV-009/010/011/012)
- [ ] Calculate discrepancies at 4 levels (total, energy-type, facility, instrument)
- [ ] Classify 8 discrepancy types with materiality assessment
- [ ] Waterfall decomposition of discrepancy drivers
- [ ] 4-dimension quality scoring (completeness/consistency/accuracy/transparency)
- [ ] Composite quality grades A through F
- [ ] Generate framework-specific tables for 7 frameworks
- [ ] Year-over-year trend analysis with CAGR
- [ ] 4 intensity metrics (revenue/FTE/area/production)
- [ ] RE100 progress tracking
- [ ] SBTi target trajectory comparison
- [ ] Procurement Impact Factor calculation
- [ ] 30+ regional residual mix factors
- [ ] 6-level EF hierarchy with quality scores
- [ ] 7 regulatory framework compliance checks (84 requirements)
- [ ] 20 REST API endpoints
- [ ] V064 database migration (14 tables, 3 hypertables, 2 CAs)
- [ ] 1,000+ unit tests
- [ ] SHA-256 provenance on every result
- [ ] Auth integration (route_protector.py + auth_setup.py)

---

## 6. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic v2 | Data models |
| FastAPI | REST API |
| prometheus_client | Metrics |
| psycopg[binary] | PostgreSQL |
| TimescaleDB | Hypertables |
| MRV-009 | Location-based Scope 2 results |
| MRV-010 | Market-based Scope 2 results |
| MRV-011 | Steam/Heat Purchase results |
| MRV-012 | Cooling Purchase results |

---

## 7. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-22 | Initial PRD |
