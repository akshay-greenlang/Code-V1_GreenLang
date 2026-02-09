# PRD: AGENT-DATA-010 — GL-DATA-X-013 Data Quality Profiler

## 1. Overview

| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-013 |
| Internal Code | AGENT-DATA-010 |
| Category | Data Quality Agents |
| SDK Package | `greenlang.data_quality_profiler` |
| DB Migration | V040 |
| Priority | Core (Year 1 — 2026) |
| Est. Variants | 500 |

The Data Quality Profiler is the **first** of 10 Data Quality Agents. It provides comprehensive, multi-dimensional data quality assessment covering completeness, validity, consistency, timeliness, uniqueness, and anomaly detection for any dataset flowing through the GreenLang platform.

## 2. Problem Statement

Climate data flowing from Data Intake Agents (AGENT-DATA-001–009) into MRV/Calculation agents must meet strict quality standards. Poor data quality (missing values, format inconsistencies, stale data, outliers) directly impacts emissions calculations and regulatory submissions. Currently, only a basic quality scorer exists within the Excel Normalizer (AGENT-DATA-002), limited to tabular data with 3 dimensions (completeness, accuracy, consistency).

**The Data Quality Profiler** elevates this to a standalone, cross-cutting service that:
- Profiles **any** dataset type (not just Excel/CSV)
- Adds **timeliness**, **validity**, and **uniqueness** dimensions (6 total)
- Detects **statistical anomalies** with multiple methods
- Provides **custom quality rules** and **quality gates**
- **Tracks quality trends** over time with historical comparisons
- Generates **quality scorecards** for compliance auditing

## 3. Existing Layer 1 Implementation

### 3.1 `greenlang/excel_normalizer/data_quality_scorer.py` (691 lines)
- `QualityLevel` enum: EXCELLENT/GOOD/FAIR/POOR/CRITICAL
- `DataQualityReport` Pydantic model: overall_score, completeness/accuracy/consistency scores, column_scores, issues
- `DataQualityScorer` class: score_file, score_completeness, score_accuracy, score_consistency, detect_outliers (IQR/z-score), detect_duplicates, compute_quality_level, generate_issues
- Weighted scoring: completeness(0.4), accuracy(0.35), consistency(0.25)

### 3.2 `greenlang/qa_test_harness/assertion_engine.py`
- TestAssertion model and assertion patterns for zero-hallucination verification
- Deterministic comparison patterns

## 4. Identified Gaps (Layer 2 SDK fills)

| # | Gap | Layer 2 Solution |
|---|-----|-----------------|
| 1 | Only 3 quality dimensions | 6 dimensions: completeness, validity, consistency, timeliness, uniqueness, accuracy |
| 2 | No dataset profiling | Full statistical profiling per column (min/max/mean/median/stddev/percentiles) |
| 3 | No schema inference | Auto-detect column types, cardinality, patterns |
| 4 | No timeliness/freshness | Data age tracking, staleness scoring, SLA monitoring |
| 5 | No validity checking | Format validation (email/phone/date/URL), range checks, regex patterns |
| 6 | No uniqueness analysis | Duplicate detection, cardinality ratios, unique constraint checking |
| 7 | No anomaly detection beyond outliers | Distribution profiling, z-score/IQR/MAD methods, pattern breaking |
| 8 | No custom quality rules | Rule definition, threshold evaluation, quality gates |
| 9 | No quality trending | Historical comparison, degradation alerts, improvement tracking |
| 10 | No cross-dataset consistency | Schema drift, referential integrity, cross-source reconciliation |
| 11 | No quality scorecards | Compliance-grade quality reports for auditors |
| 12 | Limited to Excel/CSV only | Any dict-based dataset (JSON, API, DB results, etc.) |

## 5. Architecture

### 5.1 Seven Engines

| # | Engine | Responsibility |
|---|--------|---------------|
| 1 | `DatasetProfiler` | Schema inference, column statistics (min/max/mean/median/stddev/percentiles), cardinality, data type detection, memory estimation, null pattern analysis |
| 2 | `CompletenessAnalyzer` | Null/empty analysis, coverage rates, required field checks, missing pattern detection (MCAR/MAR/MNAR), record-level completeness |
| 3 | `ValidityChecker` | Type conformance, format validation (20+ patterns: email/phone/date/URL/IP/UUID), range checks, regex pattern matching, domain validation, cross-field constraints |
| 4 | `ConsistencyAnalyzer` | Format uniformity within columns, cross-column referential checks, schema drift detection, value distribution stability, cross-dataset comparison |
| 5 | `TimelinessTracker` | Data freshness scoring, staleness detection, update frequency analysis, SLA compliance, age-based quality degradation curves |
| 6 | `AnomalyDetector` | Statistical outlier detection (IQR/z-score/MAD/Grubbs), distribution profiling (normal/uniform/skewed), pattern breaking, sudden change detection |
| 7 | `QualityRuleEngine` | Custom rule definitions (6 rule types), threshold evaluation, quality gates (pass/warn/fail), scoring aggregation, rule templates, quality SLA tracking |

### 5.2 Database Schema (V040)

```
Schema: data_quality_profiler_service
Tables:
  1. dataset_profiles       - Profile results per dataset
  2. column_profiles        - Per-column statistics
  3. quality_assessments    - Overall quality scores per run
  4. quality_dimensions     - Per-dimension scores (6 dimensions)
  5. quality_rules          - Custom rule definitions
  6. rule_evaluations       - Rule evaluation results
  7. quality_issues         - Detected issues with severity
  8. anomaly_detections     - Anomaly detection results
  9. quality_gates          - Gate definitions and pass/fail
  10. quality_trends         - Historical quality tracking
Hypertables (3):
  - quality_events (7-day chunks) - Time-series quality events
  - profile_events (7-day chunks) - Profiling event tracking
  - anomaly_events (7-day chunks) - Anomaly detection events
Continuous Aggregates (2):
  - quality_hourly_stats - Hourly quality aggregation
  - profile_hourly_stats - Hourly profiling aggregation
```

### 5.3 Prometheus Metrics (12)

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | `gl_dq_datasets_profiled_total` | Counter | source |
| 2 | `gl_dq_columns_profiled_total` | Counter | data_type |
| 3 | `gl_dq_assessments_completed_total` | Counter | quality_level |
| 4 | `gl_dq_rules_evaluated_total` | Counter | result |
| 5 | `gl_dq_anomalies_detected_total` | Counter | method |
| 6 | `gl_dq_gates_evaluated_total` | Counter | outcome |
| 7 | `gl_dq_overall_quality_score` | Histogram | - |
| 8 | `gl_dq_processing_duration_seconds` | Histogram | operation |
| 9 | `gl_dq_active_profiles` | Gauge | - |
| 10 | `gl_dq_total_issues_found` | Gauge | - |
| 11 | `gl_dq_processing_errors_total` | Counter | error_type |
| 12 | `gl_dq_freshness_checks_total` | Counter | status |

### 5.4 REST API Endpoints (20)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/v1/profile` | Profile a dataset |
| 2 | POST | `/v1/profile/batch` | Batch profile multiple datasets |
| 3 | GET | `/v1/profiles` | List profile results |
| 4 | GET | `/v1/profiles/{profile_id}` | Get specific profile |
| 5 | POST | `/v1/assess` | Run quality assessment |
| 6 | POST | `/v1/assess/batch` | Batch quality assessment |
| 7 | GET | `/v1/assessments` | List assessments |
| 8 | GET | `/v1/assessments/{assessment_id}` | Get specific assessment |
| 9 | POST | `/v1/validate` | Validate dataset against rules |
| 10 | POST | `/v1/detect-anomalies` | Detect anomalies |
| 11 | GET | `/v1/anomalies` | List detected anomalies |
| 12 | POST | `/v1/check-freshness` | Check data freshness |
| 13 | POST | `/v1/rules` | Create quality rule |
| 14 | GET | `/v1/rules` | List rules |
| 15 | PUT | `/v1/rules/{rule_id}` | Update rule |
| 16 | DELETE | `/v1/rules/{rule_id}` | Delete rule |
| 17 | POST | `/v1/gates` | Evaluate quality gate |
| 18 | GET | `/v1/trends` | Get quality trends |
| 19 | POST | `/v1/reports` | Generate quality report |
| 20 | GET | `/health` | Health check |

### 5.5 Configuration (GL_DQ_ prefix, ~35 fields)

Connections, profiling limits, dimension weights, freshness thresholds, anomaly settings, rule limits, gate thresholds, cache/pool sizing.

## 6. Layer 1 Re-Exports

From `greenlang.excel_normalizer.data_quality_scorer`:
- `QualityLevel` (enum)
- `DataQualityReport` (Pydantic model)
- `DataQualityScorer` (class)

## 7. Success Criteria

- 7 engines with full test coverage
- 20 REST API endpoints
- 12 Prometheus metrics with graceful fallback
- V040 DB migration with TimescaleDB hypertables
- SHA-256 provenance chain on all mutations
- K8s manifests (deployment, service, configmap, secret, hpa, pdb, networkpolicy, servicemonitor, alerts, grafana dashboard)
- CI/CD pipeline (lint, type-check, unit-test, integration-test, security-scan, migration-validate)
- 85%+ test coverage target
- Zero-hallucination: all quality scores are deterministic arithmetic

## 8. Integration Points

- **Upstream**: All Data Intake Agents (AGENT-DATA-001–009) feed datasets for quality profiling
- **Downstream**: MRV agents consume quality scores; Orchestrator uses quality gates
- **Cross-cutting**: QA Test Harness (AGENT-FOUND-009) for validation; Observability Agent (AGENT-FOUND-010) for metrics
- **Future**: AGENT-DATA-011 (Duplicate Detection), AGENT-DATA-012 (Missing Value Imputer) will leverage profiling results
