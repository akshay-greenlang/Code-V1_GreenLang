# PRD: AGENT-DATA-012 — GL-DATA-X-015 Missing Value Imputer

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-015 |
| Internal Label | AGENT-DATA-012 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Detect, analyze, and impute missing values across datasets using statistical, ML, and rule-based methods with full provenance |
| Estimated Variants | 200 |
| Status | BUILT (100%) |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| Files | 50+ files, ~15K+ lines |
| DB Migration | V042 |

## 2. Problem Statement
Missing values in sustainability datasets (emission inventories, supplier surveys, energy consumption records, Scope 3 spend data) undermine calculation accuracy, cause regulatory non-compliance, and break downstream agent pipelines. Current Layer 1 handling is limited to simple deletion or zero-fill, which:

1. Introduces systematic bias in emission totals (zero-fill understates, deletion overstates per-unit)
2. Breaks GHG Protocol completeness requirements (must account for all material sources)
3. Causes CSRD/ESRS disclosure gaps (Article 29a requires complete value chains)
4. Invalidates EUDR due diligence statements (missing geolocation = non-compliant)
5. Produces unreliable Scope 3 calculations (missing spend categories = incomplete coverage)
6. Fails SBTi target-setting validation (incomplete base-year data rejected)
7. Triggers audit findings in SOC 2 evidence packages (data quality below threshold)
8. Prevents accurate year-over-year trending (missing periods distort baselines)
9. Breaks AGENT-DATA-010 quality gates (completeness dimension fails)
10. Causes AGENT-DATA-011 duplicate detection false negatives (missing fields reduce similarity)

A dedicated agent is needed for pattern-aware missingness analysis, multiple imputation strategies (statistical, ML, rule-based, time-series), confidence scoring, validation of imputed values, and full audit trail with documented methodology per regulatory requirements.

## 3. Existing Layer 1 Capabilities
- `greenlang.excel_normalizer.transform_engine.TransformEngine` — `fill_missing()` with constant/forward-fill only
- `greenlang.data_quality_profiler.completeness_analyzer.CompletenessAnalyzer` — detects missing values, no imputation
- `greenlang.spend_categorizer.spend_ingestion.SpendIngestionEngine` — drops records with >50% missing fields
- `greenlang.normalizer.entity_resolver.EntityResolver` — fills missing entity fields from alias registry
- `greenlang.data_quality_profiler.models.CompletenessReport` — reports missing percentage per column

## 4. Identified Gaps (12)
| # | Gap | Layer 1 | Layer 2 Needed |
|---|-----|---------|----------------|
| 1 | Missingness pattern analysis | Basic % only | MCAR/MAR/MNAR classification, Little's test, pattern matrix visualization |
| 2 | Statistical imputation | Constant/ffill only | Mean/median/mode, k-NN, regression, hot-deck, LOCF/NOCB |
| 3 | ML-based imputation | None | Random Forest, gradient boosting, iterative (MICE-like), matrix factorization |
| 4 | Rule-based imputation | None | Domain rules (e.g., "if fuel_type=diesel, default EF=2.68"), lookup tables |
| 5 | Time-series imputation | Forward-fill only | Linear/spline interpolation, seasonal decomposition, trend extrapolation |
| 6 | Multiple imputation | None | Generate M imputed datasets, pool estimates, uncertainty quantification |
| 7 | Confidence scoring | None | Per-imputed-value confidence (0-1) based on method, neighbors, variance |
| 8 | Validation of imputations | None | Cross-validation, plausibility checks, distribution preservation |
| 9 | Regulatory documentation | None | Method justification per GHG Protocol, CSRD data quality requirements |
| 10 | Imputation pipeline | None | Multi-strategy orchestration with fallback chains |
| 11 | Batch processing | None | Large dataset support with checkpointing |
| 12 | Imputation provenance | None | SHA-256 chain tracking every imputed value with method + justification |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | MissingnessAnalyzer | Detect missing patterns (MCAR/MAR/MNAR), compute missingness matrix, Little's MCAR test, per-column/per-row analysis, correlation of missingness |
| 2 | StatisticalImputer | Mean/median/mode, k-NN (k=1-20), regression (linear/logistic), hot-deck (random/sequential), LOCF/NOCB, grouped imputation |
| 3 | MLImputer | Random Forest, gradient boosting, iterative multivariate (MICE-like rounds), matrix factorization (SVD), feature importance ranking |
| 4 | RuleBasedImputer | Domain rules engine (if-then), lookup table imputation, default value registry, regulatory default values (GHG Protocol, DEFRA, EPA) |
| 5 | TimeSeriesImputer | Linear/cubic spline interpolation, seasonal decomposition (STL), trend extrapolation, moving average, exponential smoothing |
| 6 | ValidationEngine | Cross-validation of imputed values, distribution preservation tests (KS-test, chi-square), plausibility range checks, before/after statistics |
| 7 | ImputationPipeline | End-to-end orchestration: analyze -> strategize -> impute -> validate -> document, strategy selection, fallback chains, checkpointing |

### 5.2 Data Flow
```
Input Dataset → MissingnessAnalyzer → Pattern Classification
    → Strategy Selection (auto/manual)
    → [StatisticalImputer | MLImputer | RuleBasedImputer | TimeSeriesImputer]
    → ValidationEngine → Quality Check
    → ImputationPipeline → Output Dataset + Provenance Report
```

### 5.3 Database Schema (V042)
- `imputation_jobs` — job tracking (dataset refs, status, strategy config)
- `imputation_analyses` — missingness analysis results per dataset
- `imputation_strategies` — selected strategies per column/field
- `imputation_results` — imputed values with method, confidence, original state
- `imputation_validations` — validation results (distribution tests, plausibility)
- `imputation_rules` — domain rules and lookup tables
- `imputation_rule_sets` — grouped rule configurations
- `imputation_templates` — reusable imputation configurations
- `imputation_reports` — generated compliance reports
- `imputation_audit_log` — all actions with provenance
- 3 hypertables: `imputation_events`, `validation_events`, `pipeline_events` (7-day chunks)
- 2 continuous aggregates: `imputation_hourly_stats`, `validation_hourly_stats`

### 5.4 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_mvi_jobs_processed_total | Counter | Jobs processed by status |
| gl_mvi_values_imputed_total | Counter | Values imputed by method |
| gl_mvi_analyses_completed_total | Counter | Missingness analyses completed |
| gl_mvi_validations_passed_total | Counter | Validations passed/failed |
| gl_mvi_rules_evaluated_total | Counter | Rules evaluated |
| gl_mvi_strategies_selected_total | Counter | Strategies selected by type |
| gl_mvi_confidence_score | Histogram | Imputation confidence distribution |
| gl_mvi_processing_duration_seconds | Histogram | Processing duration by operation |
| gl_mvi_completeness_improvement | Histogram | Completeness improvement (before vs after) |
| gl_mvi_active_jobs | Gauge | Currently active jobs |
| gl_mvi_total_missing_detected | Gauge | Total missing values detected |
| gl_mvi_processing_errors_total | Counter | Errors by type |

### 5.5 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/imputer/jobs | Create imputation job |
| 2 | GET | /api/v1/imputer/jobs | List imputation jobs |
| 3 | GET | /api/v1/imputer/jobs/{id} | Get job details |
| 4 | DELETE | /api/v1/imputer/jobs/{id} | Cancel/delete job |
| 5 | POST | /api/v1/imputer/analyze | Analyze missingness patterns |
| 6 | GET | /api/v1/imputer/analyze/{id} | Get analysis results |
| 7 | POST | /api/v1/imputer/impute | Impute missing values |
| 8 | POST | /api/v1/imputer/impute/batch | Batch imputation |
| 9 | GET | /api/v1/imputer/results/{id} | Get imputation results |
| 10 | POST | /api/v1/imputer/validate | Validate imputed values |
| 11 | GET | /api/v1/imputer/validate/{id} | Get validation results |
| 12 | POST | /api/v1/imputer/rules | Create imputation rule |
| 13 | GET | /api/v1/imputer/rules | List rules |
| 14 | PUT | /api/v1/imputer/rules/{id} | Update rule |
| 15 | DELETE | /api/v1/imputer/rules/{id} | Delete rule |
| 16 | POST | /api/v1/imputer/templates | Create imputation template |
| 17 | GET | /api/v1/imputer/templates | List templates |
| 18 | POST | /api/v1/imputer/pipeline | Run full pipeline |
| 19 | GET | /api/v1/imputer/health | Health check |
| 20 | GET | /api/v1/imputer/stats | Service statistics |

### 5.6 Configuration (GL_MVI_ prefix)
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| GL_MVI_DATABASE_URL | str | "" | PostgreSQL connection string |
| GL_MVI_REDIS_URL | str | "" | Redis connection string |
| GL_MVI_LOG_LEVEL | str | "INFO" | Logging level |
| GL_MVI_BATCH_SIZE | int | 1000 | Records per batch |
| GL_MVI_MAX_RECORDS | int | 100000 | Maximum records per job |
| GL_MVI_DEFAULT_STRATEGY | str | "auto" | Default imputation strategy |
| GL_MVI_KNN_NEIGHBORS | int | 5 | Default k for k-NN imputation |
| GL_MVI_MICE_ITERATIONS | int | 10 | MICE iteration rounds |
| GL_MVI_CONFIDENCE_THRESHOLD | float | 0.7 | Minimum confidence to accept imputation |
| GL_MVI_MAX_MISSING_PCT | float | 0.8 | Max missing % before column is dropped |
| GL_MVI_ENABLE_ML_IMPUTATION | bool | True | Enable ML-based methods |
| GL_MVI_ENABLE_TIMESERIES | bool | True | Enable time-series methods |
| GL_MVI_MULTIPLE_IMPUTATIONS | int | 5 | Number of multiple imputations (M) |
| GL_MVI_VALIDATION_SPLIT | float | 0.2 | Validation holdout ratio |
| GL_MVI_WORKER_COUNT | int | 4 | Parallel worker count |
| GL_MVI_POOL_MIN_SIZE | int | 2 | Connection pool minimum |
| GL_MVI_POOL_MAX_SIZE | int | 10 | Connection pool maximum |
| GL_MVI_CACHE_TTL | int | 3600 | Cache TTL in seconds |
| GL_MVI_RATE_LIMIT_RPM | int | 120 | API rate limit per minute |
| GL_MVI_ENABLE_PROVENANCE | bool | True | Enable SHA-256 provenance chains |

## 6. Layer 1 Re-exports
The SDK will re-export from existing Layer 1:
- `greenlang.data_quality_profiler.completeness_analyzer.CompletenessAnalyzer` — for initial completeness assessment
- `greenlang.data_quality_profiler.models.CompletenessReport, QualityDimension` — for quality dimension models
- `greenlang.excel_normalizer.models.DataType` — for data type detection

## 7. Success Criteria
- 7 engines with full unit test coverage (>=85%)
- 8+ imputation algorithms (statistical + ML + rule + time-series)
- MCAR/MAR/MNAR missingness classification
- Multiple imputation with uncertainty quantification
- Per-value confidence scoring (0-1)
- Cross-validation and distribution preservation checks
- Domain rule engine with regulatory defaults (GHG Protocol, DEFRA, EPA)
- SHA-256 provenance chain on all imputed values
- 12 Prometheus metrics, 20 REST endpoints
- V042 migration with 10+ tables, 3 hypertables, 2 continuous aggregates
- K8s manifests + CI/CD pipeline
- Auth integration with RBAC permissions

## 8. Integration Points
- AGENT-DATA-010 (Data Quality Profiler) — completeness assessment triggers imputation
- AGENT-DATA-011 (Duplicate Detection) — impute before dedup to improve similarity
- AGENT-DATA-002 (Excel/CSV Normalizer) — normalized data as input
- AGENT-DATA-009 (Spend Data Categorizer) — fill missing spend categories
- AGENT-FOUND-003 (Unit Normalizer) — entity resolution for lookup imputation
- AGENT-FOUND-008 (Reproducibility) — deterministic imputation verification
- AGENT-FOUND-005 (Citations) — provenance chain integration

## 9. Test Coverage Strategy

### 9.1 Unit Tests (Target: 85%+ coverage)
| Test File | Coverage Scope | Min Tests |
|-----------|---------------|-----------|
| test_config.py | Config defaults, env vars, singleton, validation | 70 |
| test_models.py | Model creation, field constraints, enum values | 50 |
| test_metrics.py | Metric recording, increment, Prometheus export | 20 |
| test_provenance.py | SHA-256 hashing, chain integrity, audit trail | 15 |
| test_missingness_analyzer.py | MCAR/MAR/MNAR detection, pattern matrix, Little's test | 60 |
| test_statistical_imputer.py | Mean/median/mode, k-NN, regression, hot-deck, LOCF/NOCB | 60 |
| test_ml_imputer.py | Random Forest, gradient boosting, MICE, matrix factorization | 50 |
| test_rule_based_imputer.py | Rule evaluation, lookup tables, regulatory defaults | 50 |
| test_time_series_imputer.py | Interpolation, seasonal, trend, moving average | 50 |
| test_validation_engine.py | KS-test, chi-square, plausibility, distribution preservation | 50 |
| test_imputation_pipeline.py | End-to-end pipeline, checkpointing, strategy selection | 40 |
| test_setup.py | Service facade, all public methods, health | 85 |
| test_router.py | All 20 API endpoints, error handling, validation | 40 |
| **Total** | **All components** | **640+** |

### 9.2 Integration Tests
- E2E pipeline: analyze -> impute -> validate full flow
- Database: V042 tables, hypertables, continuous aggregates, RLS policies
- API: All 20 endpoints with auth, tenant isolation
- Performance: 100K records imputation throughput

## 10. Security & Data Privacy

### 10.1 Authentication & Authorization
- All 20 API endpoints protected via SEC-001 JWT Authentication
- RBAC permissions: `imputer:jobs:create`, `imputer:jobs:read`, `imputer:analyze:execute`, `imputer:impute:execute`, `imputer:rules:manage`, `imputer:admin`
- Tenant isolation enforced via RLS policies on all tables

### 10.2 Data Privacy
- Original values preserved alongside imputed values (never destructive)
- Imputation methods documented per value for regulatory audit
- Data retention: 90 days for events, configurable per tenant
- GDPR: Right to erasure supported via cascade delete with audit

## 11. Performance & Scalability
| Metric | Target | Method |
|--------|--------|--------|
| Statistical imputation | 50K values/sec | Vectorized NumPy operations |
| k-NN imputation | 10K values/sec | KD-tree nearest neighbor search |
| ML imputation | 5K values/sec | Pre-trained model inference |
| Time-series imputation | 20K values/sec | Vectorized interpolation |
| API latency (p95) | < 200ms | Connection pooling, Redis caching |
| API latency (p99) | < 500ms | Background job processing |

## 12. Rollout & Migration Plan

### Phase 1: Core SDK
- [x] 7 engines implemented
- [x] Configuration with GL_MVI_ env prefix
- [x] Models (30+ Pydantic v2 models)
- [x] Provenance tracking (SHA-256)
- [x] Metrics (12 Prometheus metrics)

### Phase 2: Service Layer
- [x] Service facade (MissingValueImputerService)
- [x] API router (20 REST endpoints)
- [x] SDK init with 70+ exports

### Phase 3: Infrastructure
- [x] V042 database migration
- [x] K8s manifests (10 files)
- [x] CI/CD pipeline
- [x] Dockerfile
- [x] Auth integration

### Phase 4: Testing
- [x] Unit tests (640+, 85%+ coverage)
- [x] Integration tests (50+)

## Appendix A: Glossary
| Term | Definition |
|------|-----------|
| MCAR | Missing Completely At Random — missingness unrelated to any variable |
| MAR | Missing At Random — missingness related to observed variables only |
| MNAR | Missing Not At Random — missingness related to the unobserved value itself |
| MICE | Multiple Imputation by Chained Equations — iterative multivariate imputation |
| k-NN | k-Nearest Neighbors — impute from k most similar complete records |
| Hot-deck | Impute from a randomly selected similar record in the dataset |
| LOCF | Last Observation Carried Forward — time-series forward fill |
| NOCB | Next Observation Carried Backward — time-series backward fill |
| STL | Seasonal-Trend decomposition using Loess — time-series decomposition |
| Multiple Imputation | Generate M complete datasets, analyze each, pool results for uncertainty |
