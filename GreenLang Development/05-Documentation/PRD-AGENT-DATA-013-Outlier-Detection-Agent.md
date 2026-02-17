# PRD: AGENT-DATA-013 — GL-DATA-X-016 Outlier Detection Agent

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-016 |
| Internal Label | AGENT-DATA-013 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Detect, classify, and handle statistical outliers across sustainability datasets with full provenance |
| Estimated Variants | 200 |
| Status | BUILT (100%) |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| Files | 53+ files, ~16K+ lines |
| DB Migration | V043 |

## 2. Problem Statement
Outliers in sustainability datasets (emission spikes, anomalous energy readings, erroneous spend data, sensor malfunctions) corrupt GHG calculations, trigger false compliance alerts, and undermine regulatory credibility. Current Layer 1 capabilities (AGENT-DATA-010 AnomalyDetector) provide basic IQR/z-score detection but lack:

1. Multi-method ensemble outlier scoring for robust detection across data distributions
2. Contextual outlier detection (value normal globally but anomalous within its context/group)
3. Temporal outlier detection (sudden changes in time-series that aren't seasonal)
4. Multivariate outlier detection (records normal per-feature but anomalous in combination)
5. Outlier classification (error vs. genuine extreme vs. data entry mistake vs. regime change)
6. Automated treatment strategies (cap, winsorize, flag, remove, replace, investigate)
7. Domain-specific thresholds (emission factors have known valid ranges per GHG Protocol)
8. Regulatory-compliant documentation of outlier handling per CSRD/ESRS data quality requirements
9. Impact analysis (how do outliers affect downstream calculations and compliance reports)
10. Feedback loop integration (confirmed outliers improve detection model over time)

## 3. Existing Layer 1 Capabilities
- `greenlang.data_quality_profiler.anomaly_detector.AnomalyDetector` — IQR, z-score, MAD, Grubbs, modified z-score (5 methods)
- `greenlang.data_quality_profiler.models.AnomalyResult` — basic anomaly result model
- `greenlang.data_quality_profiler.quality_rule_engine.QualityRuleEngine` — range-check rules
- `greenlang.missing_value_imputer.validation_engine.ValidationEngine` — plausibility checks, KS-test

## 4. Identified Gaps (12)
| # | Gap | Layer 1 | Layer 2 Needed |
|---|-----|---------|----------------|
| 1 | Ensemble outlier scoring | Individual methods only | Weighted ensemble of multiple detectors with consensus scoring |
| 2 | Contextual outliers | Global detection only | Group-by contextual detection (e.g., outlier within same facility/region) |
| 3 | Temporal outliers | No time awareness | Change point detection, trend break analysis, seasonal adjustment |
| 4 | Multivariate outliers | Univariate only | Mahalanobis distance, isolation forest, Local Outlier Factor (LOF) |
| 5 | Outlier classification | Binary only | 5-class: error/extreme/data_entry/regime_change/sensor_fault |
| 6 | Treatment strategies | None | Cap/winsorize/flag/remove/replace/investigate with undo |
| 7 | Domain thresholds | Generic ranges | Emission factor valid ranges, energy intensity bounds, spend limits |
| 8 | Impact analysis | None | Quantify effect of outlier treatment on totals, trends, compliance |
| 9 | Batch processing | Single column | Multi-column, multi-dataset batch with parallel processing |
| 10 | Regulatory documentation | None | CSRD/GHG Protocol/ESRS compliant outlier handling justification |
| 11 | Feedback learning | None | Confirmed outlier labels improve thresholds over time |
| 12 | Outlier provenance | None | SHA-256 chain tracking every detection, classification, and treatment |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | StatisticalDetector | Univariate detection: IQR, z-score, modified z-score, MAD, Grubbs, Tukey fences, percentile-based |
| 2 | ContextualDetector | Group-by contextual detection, conditional outliers, peer comparison, facility/region/sector grouping |
| 3 | TemporalDetector | Time-series outliers: change point detection (CUSUM, PELT-like), trend breaks, seasonal-adjusted residuals |
| 4 | MultivariateDetector | Multi-feature detection: Mahalanobis distance, Isolation Forest, Local Outlier Factor (LOF), DBSCAN-based |
| 5 | OutlierClassifier | 5-class classification: error/extreme/data_entry/regime_change/sensor_fault, confidence scoring, evidence collection |
| 6 | TreatmentEngine | 6 strategies: cap/winsorize/flag/remove/replace/investigate, undo support, impact quantification |
| 7 | OutlierPipeline | End-to-end orchestration: detect -> classify -> treat -> validate -> document, ensemble scoring, batch processing |

### 5.2 Data Flow
```
Input Dataset → StatisticalDetector (univariate scores)
             → ContextualDetector (contextual scores)
             → TemporalDetector (temporal scores)
             → MultivariateDetector (multivariate scores)
             → Ensemble Scoring (weighted combination)
             → OutlierClassifier (5-class + confidence)
             → TreatmentEngine (strategy application)
             → OutlierPipeline (orchestration + provenance)
```

### 5.3 Database Schema (V043)
- `outlier_jobs` — job tracking (dataset, status, config, ensemble weights)
- `outlier_detections` — detected outliers (record ref, column, score, method, threshold)
- `outlier_scores` — per-method scores for ensemble (statistical, contextual, temporal, multivariate)
- `outlier_classifications` — classification results (5-class, confidence, evidence)
- `outlier_treatments` — applied treatments (strategy, original value, treated value, justification)
- `outlier_thresholds` — domain-specific thresholds (column, context, min/max, source authority)
- `outlier_feedback` — confirmed labels for model improvement
- `outlier_impact_analyses` — impact of treatment on downstream calculations
- `outlier_reports` — generated compliance reports
- `outlier_audit_log` — all actions with provenance
- 3 hypertables: `outlier_events`, `detection_events`, `treatment_events` (7-day chunks)
- 2 continuous aggregates: `outlier_hourly_stats`, `detection_hourly_stats`

### 5.4 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_od_jobs_processed_total | Counter | Jobs processed by status |
| gl_od_outliers_detected_total | Counter | Outliers detected by method |
| gl_od_outliers_classified_total | Counter | Outliers classified by type |
| gl_od_treatments_applied_total | Counter | Treatments applied by strategy |
| gl_od_thresholds_evaluated_total | Counter | Domain thresholds evaluated |
| gl_od_feedback_received_total | Counter | Feedback labels received |
| gl_od_ensemble_score | Histogram | Ensemble outlier score distribution |
| gl_od_processing_duration_seconds | Histogram | Processing duration by operation |
| gl_od_detection_confidence | Histogram | Detection confidence distribution |
| gl_od_active_jobs | Gauge | Currently active jobs |
| gl_od_total_outliers_flagged | Gauge | Total outliers currently flagged |
| gl_od_processing_errors_total | Counter | Errors by type |

### 5.5 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/outlier/jobs | Create detection job |
| 2 | GET | /api/v1/outlier/jobs | List jobs |
| 3 | GET | /api/v1/outlier/jobs/{id} | Get job details |
| 4 | DELETE | /api/v1/outlier/jobs/{id} | Cancel/delete job |
| 5 | POST | /api/v1/outlier/detect | Detect outliers in dataset |
| 6 | POST | /api/v1/outlier/detect/batch | Batch detection across columns |
| 7 | GET | /api/v1/outlier/detections | List detections |
| 8 | GET | /api/v1/outlier/detections/{id} | Get detection details |
| 9 | POST | /api/v1/outlier/classify | Classify detected outliers |
| 10 | GET | /api/v1/outlier/classifications/{id} | Get classification details |
| 11 | POST | /api/v1/outlier/treat | Apply treatment strategy |
| 12 | GET | /api/v1/outlier/treatments/{id} | Get treatment details |
| 13 | POST | /api/v1/outlier/treatments/{id}/undo | Undo treatment |
| 14 | POST | /api/v1/outlier/thresholds | Create domain threshold |
| 15 | GET | /api/v1/outlier/thresholds | List thresholds |
| 16 | POST | /api/v1/outlier/feedback | Submit feedback label |
| 17 | POST | /api/v1/outlier/impact | Run impact analysis |
| 18 | POST | /api/v1/outlier/pipeline | Run full pipeline |
| 19 | GET | /api/v1/outlier/health | Health check |
| 20 | GET | /api/v1/outlier/stats | Service statistics |

### 5.6 Configuration (GL_OD_ prefix)
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| GL_OD_DATABASE_URL | str | "" | PostgreSQL connection string |
| GL_OD_REDIS_URL | str | "" | Redis connection string |
| GL_OD_LOG_LEVEL | str | "INFO" | Logging level |
| GL_OD_BATCH_SIZE | int | 1000 | Records per batch |
| GL_OD_MAX_RECORDS | int | 100000 | Maximum records per job |
| GL_OD_IQR_MULTIPLIER | float | 1.5 | IQR fence multiplier |
| GL_OD_ZSCORE_THRESHOLD | float | 3.0 | Z-score outlier threshold |
| GL_OD_MAD_THRESHOLD | float | 3.5 | MAD outlier threshold |
| GL_OD_GRUBBS_ALPHA | float | 0.05 | Grubbs test significance level |
| GL_OD_LOF_NEIGHBORS | int | 20 | LOF k-neighbors |
| GL_OD_ISOLATION_TREES | int | 100 | Isolation Forest tree count |
| GL_OD_ENSEMBLE_METHOD | str | "weighted_average" | Ensemble combination method |
| GL_OD_MIN_CONSENSUS | int | 2 | Minimum methods agreeing for outlier |
| GL_OD_ENABLE_CONTEXTUAL | bool | True | Enable contextual detection |
| GL_OD_ENABLE_TEMPORAL | bool | True | Enable temporal detection |
| GL_OD_ENABLE_MULTIVARIATE | bool | True | Enable multivariate detection |
| GL_OD_DEFAULT_TREATMENT | str | "flag" | Default treatment strategy |
| GL_OD_WINSORIZE_PCT | float | 0.05 | Winsorization percentile |
| GL_OD_WORKER_COUNT | int | 4 | Parallel worker count |
| GL_OD_POOL_MIN_SIZE | int | 2 | Connection pool minimum |
| GL_OD_POOL_MAX_SIZE | int | 10 | Connection pool maximum |
| GL_OD_CACHE_TTL | int | 3600 | Cache TTL in seconds |
| GL_OD_RATE_LIMIT_RPM | int | 120 | API rate limit per minute |
| GL_OD_ENABLE_PROVENANCE | bool | True | Enable SHA-256 provenance |

## 6. Layer 1 Re-exports
- `greenlang.data_quality_profiler.anomaly_detector.AnomalyDetector` — for base detection methods
- `greenlang.data_quality_profiler.models.AnomalyResult, QualityDimension` — for result models

## 7. Success Criteria
- 7 engines with full unit test coverage (>=85%)
- 7+ detection methods (IQR, z-score, MAD, Grubbs, LOF, Isolation Forest, Mahalanobis)
- Ensemble scoring with configurable weights and consensus threshold
- 5-class outlier classification with confidence scoring
- 6 treatment strategies with undo support
- Domain threshold registry for emission factors, energy, spend
- Impact analysis quantifying treatment effects on downstream calculations
- SHA-256 provenance chain on all operations
- 12 Prometheus metrics, 20 REST endpoints
- V043 migration with 10+ tables, 3 hypertables, 2 continuous aggregates

## 8. Integration Points
- AGENT-DATA-010 (Data Quality Profiler) — Layer 1 anomaly detection, quality scoring
- AGENT-DATA-012 (Missing Value Imputer) — impute after outlier removal
- AGENT-DATA-011 (Duplicate Detection) — detect outlier duplicates
- AGENT-DATA-009 (Spend Data Categorizer) — spend amount outlier detection
- AGENT-FOUND-008 (Reproducibility) — deterministic detection verification
- AGENT-FOUND-005 (Citations) — provenance chain integration

## 9. Test Coverage Strategy
| Test File | Min Tests |
|-----------|-----------|
| test_config.py | 70 |
| test_models.py | 50 |
| test_metrics.py | 20 |
| test_provenance.py | 15 |
| test_statistical_detector.py | 60 |
| test_contextual_detector.py | 50 |
| test_temporal_detector.py | 50 |
| test_multivariate_detector.py | 50 |
| test_outlier_classifier.py | 50 |
| test_treatment_engine.py | 50 |
| test_outlier_pipeline.py | 40 |
| test_setup.py | 85 |
| test_router.py | 40 |
| **Total** | **630+** |

## 10. Security & Data Privacy
- All 20 API endpoints protected via SEC-001 JWT Authentication
- RBAC: `outlier:jobs:create/read`, `outlier:detect:execute`, `outlier:treat:execute`, `outlier:thresholds:manage`, `outlier:admin`
- Tenant isolation via RLS policies on all tables
- Original values preserved (treatments are non-destructive with undo)
- Audit trail via SHA-256 chain hashing

## 11. Performance & Scalability
| Metric | Target |
|--------|--------|
| Statistical detection | 100K records/sec |
| Contextual detection | 50K records/sec |
| Multivariate detection | 10K records/sec |
| Ensemble scoring | 50K records/sec |
| API latency (p95) | < 200ms |
| API latency (p99) | < 500ms |

## 12. Glossary
| Term | Definition |
|------|-----------|
| IQR | Interquartile Range — Q3-Q1, outliers beyond 1.5*IQR fences |
| MAD | Median Absolute Deviation — robust alternative to standard deviation |
| Grubbs | Statistical test for single outlier in normally-distributed data |
| LOF | Local Outlier Factor — density-based outlier scoring relative to neighbors |
| Isolation Forest | Tree-based anomaly detector, outliers have shorter average path lengths |
| Mahalanobis | Distance accounting for feature correlations in multivariate space |
| CUSUM | Cumulative Sum — change point detection via cumulative deviations |
| Winsorize | Cap extreme values at specified percentile instead of removing |
| Ensemble | Combining multiple methods for robust detection with consensus |
| Contextual Outlier | Value normal globally but anomalous within its group/context |
