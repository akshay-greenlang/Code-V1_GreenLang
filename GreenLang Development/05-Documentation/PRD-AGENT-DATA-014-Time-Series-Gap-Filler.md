# PRD: AGENT-DATA-014 — GL-DATA-X-017 Time Series Gap Filler

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-017 |
| Internal Label | AGENT-DATA-014 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Detect, characterize, and fill gaps in time-series sustainability datasets with frequency-aware interpolation, seasonal pattern matching, cross-series correlation, and full provenance |
| Estimated Variants | 200 |
| Status | BUILT (100%) |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| Files | 53+ files, ~16K+ lines |
| DB Migration | V044 |

## 2. Problem Statement
Time-series gaps in sustainability datasets (missing monthly emissions, skipped energy meter readings, quarterly spend gaps, intermittent sensor data) fundamentally undermine GHG accounting, trend analysis, and regulatory reporting. Current capabilities address general missing values (AGENT-DATA-012) but lack time-series-specific intelligence:

1. No gap detection — cannot identify where timestamps are missing relative to expected frequency
2. No frequency inference — cannot automatically detect hourly/daily/weekly/monthly/quarterly/annual patterns
3. No gap characterization — cannot classify gaps as short/long/periodic/random/systematic
4. No calendar-aware filling — business day calendars, holidays, reporting periods not considered
5. No seasonal pattern matching — cannot leverage repeating seasonal patterns for gap filling
6. No cross-series correlation — cannot use correlated reference series to fill gaps
7. No multi-resolution filling — cannot handle mixed-frequency data (monthly + quarterly + annual)
8. No gap impact assessment — cannot quantify how gaps affect emission totals and trends
9. No fill quality validation — no confidence scoring or distribution preservation checks
10. No regulatory-compliant documentation — GHG Protocol Scope 1/2/3 gap handling requirements not met

## 3. Existing Layer 1 Capabilities
- `greenlang.missing_value_imputer.time_series_imputer.TimeSeriesImputerEngine` — linear/cubic spline interpolation, seasonal decomposition, moving average, exponential smoothing
- `greenlang.missing_value_imputer.models.ImputedValue, ConfidenceLevel, ImputationStrategy` — imputation result models
- `greenlang.data_quality_profiler.timeliness_tracker.TimelinessTracker` — freshness SLA, staleness detection
- `greenlang.data_quality_profiler.completeness_analyzer.CompletenessAnalyzer` — missing % per column
- `greenlang.outlier_detector.temporal_detector.TemporalDetectorEngine` — CUSUM, trend break detection

## 4. Identified Gaps (12)
| # | Gap | Layer 1 | Layer 2 Needed |
|---|-----|---------|----------------|
| 1 | Gap detection | No gap awareness | Detect missing timestamps against expected frequency grid |
| 2 | Frequency inference | None | Auto-detect data frequency (hourly→annual), regularity scoring |
| 3 | Gap characterization | None | Classify: short_gap/long_gap/periodic_gap/random_gap/systematic_gap |
| 4 | Calendar-aware filling | None | Business day calendars, holidays, fiscal periods, reporting windows |
| 5 | Seasonal pattern matching | Basic STL only | Multi-period seasonality detection, pattern library, seasonal fill |
| 6 | Cross-series filling | None | Correlated series identification, regression-based fill, donor matching |
| 7 | Multi-resolution support | Single frequency | Handle mixed-frequency data, temporal aggregation/disaggregation |
| 8 | Gap impact assessment | None | Quantify gap effect on totals, trends, YoY comparisons, compliance |
| 9 | Fill quality validation | None | Per-value confidence, distribution preservation, cross-validation |
| 10 | Regulatory documentation | None | GHG Protocol/CSRD/ESRS compliant gap handling justification |
| 11 | Batch gap processing | None | Multi-series batch processing with parallel execution |
| 12 | Gap provenance | None | SHA-256 chain tracking every gap detection, characterization, and fill |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | GapDetector | Detect gaps against expected frequency grid, gap inventory, gap statistics, consecutive-gap detection, edge-gap handling |
| 2 | FrequencyAnalyzer | Auto-detect frequency (9 levels: sub-hourly → annual), regularity scoring, mixed-frequency detection, dominant frequency extraction |
| 3 | InterpolationEngine | Linear, cubic spline, polynomial, piecewise cubic Hermite, Akima, nearest-neighbor interpolation with time-weighting |
| 4 | SeasonalFiller | Multi-period seasonal decomposition (STL-style), seasonal pattern library, calendar-aware fill (business days, holidays, fiscal periods), day-of-week/month-of-year patterns |
| 5 | TrendExtrapolator | OLS trend fitting, exponential smoothing (single/double/Holt-Winters), moving average extrapolation, regime-aware extrapolation |
| 6 | CrossSeriesFiller | Pairwise correlation analysis, regression-based fill, donor series matching, similarity scoring, multi-series consensus |
| 7 | GapFillerPipeline | End-to-end orchestration: detect → characterize → select strategy → fill → validate → document, strategy auto-selection, fallback chains, batch processing |

### 5.2 Data Flow
```
Input Time Series → FrequencyAnalyzer (detect frequency)
                  → GapDetector (identify gaps against frequency grid)
                  → Gap Characterization (short/long/periodic/random/systematic)
                  → Strategy Selection (auto/manual per gap type)
                  → [InterpolationEngine | SeasonalFiller | TrendExtrapolator | CrossSeriesFiller]
                  → Fill Validation (confidence, distribution, plausibility)
                  → GapFillerPipeline (orchestration + provenance + documentation)
```

### 5.3 Database Schema (V044)
- `gap_filler_jobs` — job tracking (series ref, status, config, frequency detected)
- `gap_detections` — detected gaps (start, end, duration, type, severity)
- `gap_frequencies` — inferred frequencies per series (frequency, regularity, confidence)
- `gap_fills` — filled values (position, original state, filled value, method, confidence)
- `gap_fill_validations` — validation results (distribution tests, plausibility checks)
- `gap_fill_strategies` — selected strategies per gap (auto/manual, fallback chain)
- `gap_calendars` — calendar definitions (business days, holidays, fiscal periods)
- `gap_reference_series` — registered reference/donor series for cross-series filling
- `gap_fill_reports` — generated compliance reports
- `gap_fill_audit_log` — all actions with provenance
- 3 hypertables: `gap_events`, `fill_events`, `validation_events` (7-day chunks)
- 2 continuous aggregates: `gap_hourly_stats`, `fill_hourly_stats`

### 5.4 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_tsgf_jobs_processed_total | Counter | Jobs processed by status |
| gl_tsgf_gaps_detected_total | Counter | Gaps detected by type |
| gl_tsgf_gaps_filled_total | Counter | Gaps filled by method |
| gl_tsgf_validations_passed_total | Counter | Fill validations passed/failed |
| gl_tsgf_frequencies_detected_total | Counter | Frequencies detected by level |
| gl_tsgf_strategies_selected_total | Counter | Strategies selected by type |
| gl_tsgf_fill_confidence | Histogram | Fill confidence score distribution |
| gl_tsgf_processing_duration_seconds | Histogram | Processing duration by operation |
| gl_tsgf_gap_duration_seconds | Histogram | Gap duration distribution |
| gl_tsgf_active_jobs | Gauge | Currently active jobs |
| gl_tsgf_total_gaps_open | Gauge | Total gaps currently unfilled |
| gl_tsgf_processing_errors_total | Counter | Errors by type |

### 5.5 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/gap-filler/jobs | Create gap filling job |
| 2 | GET | /api/v1/gap-filler/jobs | List jobs |
| 3 | GET | /api/v1/gap-filler/jobs/{id} | Get job details |
| 4 | DELETE | /api/v1/gap-filler/jobs/{id} | Cancel/delete job |
| 5 | POST | /api/v1/gap-filler/detect | Detect gaps in time series |
| 6 | POST | /api/v1/gap-filler/detect/batch | Batch gap detection across series |
| 7 | GET | /api/v1/gap-filler/detections | List gap detections |
| 8 | GET | /api/v1/gap-filler/detections/{id} | Get detection details |
| 9 | POST | /api/v1/gap-filler/frequency | Analyze series frequency |
| 10 | GET | /api/v1/gap-filler/frequency/{id} | Get frequency analysis |
| 11 | POST | /api/v1/gap-filler/fill | Fill detected gaps |
| 12 | GET | /api/v1/gap-filler/fills/{id} | Get fill details |
| 13 | POST | /api/v1/gap-filler/fills/{id}/undo | Undo gap fill |
| 14 | POST | /api/v1/gap-filler/validate | Validate filled values |
| 15 | GET | /api/v1/gap-filler/validate/{id} | Get validation results |
| 16 | POST | /api/v1/gap-filler/calendars | Create calendar definition |
| 17 | GET | /api/v1/gap-filler/calendars | List calendars |
| 18 | POST | /api/v1/gap-filler/pipeline | Run full pipeline |
| 19 | GET | /api/v1/gap-filler/health | Health check |
| 20 | GET | /api/v1/gap-filler/stats | Service statistics |

### 5.6 Configuration (GL_TSGF_ prefix)
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| GL_TSGF_DATABASE_URL | str | "" | PostgreSQL connection string |
| GL_TSGF_REDIS_URL | str | "" | Redis connection string |
| GL_TSGF_LOG_LEVEL | str | "INFO" | Logging level |
| GL_TSGF_BATCH_SIZE | int | 1000 | Records per batch |
| GL_TSGF_MAX_RECORDS | int | 100000 | Maximum records per job |
| GL_TSGF_MIN_POINTS | int | 10 | Minimum data points for frequency inference |
| GL_TSGF_MAX_GAP_PCT | float | 0.5 | Maximum gap percentage before series rejected |
| GL_TSGF_DEFAULT_STRATEGY | str | "auto" | Default filling strategy |
| GL_TSGF_INTERPOLATION_METHOD | str | "linear" | Default interpolation method |
| GL_TSGF_SEASONAL_PERIODS | int | 12 | Default seasonal period length |
| GL_TSGF_SMOOTHING_ALPHA | float | 0.3 | Exponential smoothing alpha |
| GL_TSGF_SMOOTHING_BETA | float | 0.1 | Holt trend smoothing beta |
| GL_TSGF_SMOOTHING_GAMMA | float | 0.1 | Holt-Winters seasonal gamma |
| GL_TSGF_CONFIDENCE_THRESHOLD | float | 0.7 | Minimum confidence to accept fill |
| GL_TSGF_CORRELATION_THRESHOLD | float | 0.7 | Minimum correlation for cross-series fill |
| GL_TSGF_SHORT_GAP_LIMIT | int | 3 | Max consecutive missing for short gap |
| GL_TSGF_LONG_GAP_LIMIT | int | 12 | Max consecutive missing for long gap |
| GL_TSGF_ENABLE_SEASONAL | bool | True | Enable seasonal filling |
| GL_TSGF_ENABLE_CROSS_SERIES | bool | True | Enable cross-series filling |
| GL_TSGF_ENABLE_CALENDAR | bool | True | Enable calendar-aware filling |
| GL_TSGF_WORKER_COUNT | int | 4 | Parallel worker count |
| GL_TSGF_POOL_MIN_SIZE | int | 2 | Connection pool minimum |
| GL_TSGF_POOL_MAX_SIZE | int | 10 | Connection pool maximum |
| GL_TSGF_CACHE_TTL | int | 3600 | Cache TTL in seconds |
| GL_TSGF_RATE_LIMIT_RPM | int | 120 | API rate limit per minute |
| GL_TSGF_ENABLE_PROVENANCE | bool | True | Enable SHA-256 provenance chains |

## 6. Layer 1 Re-exports
- `greenlang.missing_value_imputer.time_series_imputer.TimeSeriesImputerEngine` — for basic interpolation
- `greenlang.missing_value_imputer.models.ImputedValue, ConfidenceLevel` — for result models
- `greenlang.data_quality_profiler.timeliness_tracker.TimelinessTracker` — for freshness checking

## 7. Success Criteria
- 7 engines with full unit test coverage (>=85%)
- 6+ interpolation/fill methods (linear, spline, seasonal, trend, cross-series, calendar)
- Automatic frequency detection (9 frequency levels)
- Gap characterization (5 gap types with severity scoring)
- Calendar-aware filling (business days, holidays, fiscal periods)
- Cross-series correlation-based filling
- Per-fill confidence scoring (0-1)
- Fill validation with distribution preservation checks
- SHA-256 provenance chain on all operations
- 12 Prometheus metrics, 20 REST endpoints
- V044 migration with 10+ tables, 3 hypertables, 2 continuous aggregates
- K8s manifests + CI/CD pipeline
- Auth integration with RBAC permissions

## 8. Integration Points
- AGENT-DATA-012 (Missing Value Imputer) — Layer 1 time-series interpolation
- AGENT-DATA-013 (Outlier Detection) — temporal anomaly context
- AGENT-DATA-010 (Data Quality Profiler) — timeliness/completeness checks
- AGENT-DATA-002 (Excel/CSV Normalizer) — normalized time-series input
- AGENT-DATA-009 (Spend Data Categorizer) — quarterly spend gap filling
- AGENT-FOUND-008 (Reproducibility) — deterministic fill verification
- AGENT-FOUND-005 (Citations) — provenance chain integration

## 9. Test Coverage Strategy
| Test File | Min Tests |
|-----------|-----------|
| test_config.py | 70 |
| test_models.py | 50 |
| test_metrics.py | 20 |
| test_provenance.py | 15 |
| test_gap_detector.py | 60 |
| test_frequency_analyzer.py | 50 |
| test_interpolation_engine.py | 60 |
| test_seasonal_filler.py | 50 |
| test_trend_extrapolator.py | 50 |
| test_cross_series_filler.py | 50 |
| test_gap_filler_pipeline.py | 40 |
| test_setup.py | 85 |
| test_router.py | 40 |
| **Total** | **640+** |

## 10. Security & Data Privacy
- All 20 API endpoints protected via SEC-001 JWT Authentication
- RBAC: `gap_filler:jobs:create/read`, `gap_filler:detect:execute`, `gap_filler:fill:execute`, `gap_filler:calendars:manage`, `gap_filler:admin`
- Tenant isolation via RLS policies on all tables
- Original values preserved (fills are non-destructive with undo)
- Audit trail via SHA-256 chain hashing

## 11. Performance & Scalability
| Metric | Target |
|--------|--------|
| Gap detection | 100K records/sec |
| Linear interpolation | 50K fills/sec |
| Seasonal filling | 20K fills/sec |
| Cross-series filling | 10K fills/sec |
| API latency (p95) | < 200ms |
| API latency (p99) | < 500ms |

## 12. Glossary
| Term | Definition |
|------|-----------|
| Gap | Missing data point(s) in a time series relative to expected frequency |
| Frequency | Regular interval between observations (hourly, daily, weekly, monthly, etc.) |
| Regularity | Score (0-1) measuring how consistently a series follows its detected frequency |
| Seasonal Pattern | Repeating pattern at fixed intervals (e.g., 12-month annual cycle) |
| STL | Seasonal-Trend decomposition using Loess — separates trend, seasonal, residual |
| Holt-Winters | Triple exponential smoothing for level, trend, and seasonality |
| Cross-Series Fill | Using correlated reference series to fill gaps via regression |
| Calendar-Aware | Filling that considers business days, holidays, and fiscal periods |
| Akima | Piecewise cubic interpolation that avoids overshooting |
| Donor Series | A complete reference series used to fill gaps in a target series |
