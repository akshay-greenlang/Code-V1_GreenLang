# PRD: AGENT-DATA-016 — GL-DATA-X-019 Data Freshness Monitor

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-019 |
| Internal Label | AGENT-DATA-016 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Monitor data freshness across all registered datasets and sources, enforce SLA compliance, detect staleness patterns, predict refresh failures, generate alerts for SLA breaches, and produce compliance reports with full provenance |
| Estimated Variants | 200 |
| Status | To Be Built |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| DB Migration | V046 |

## 2. Problem Statement
Sustainability reporting depends on timely data from dozens of sources — ERP systems, utility providers, IoT meters, supplier questionnaires, government registries, and manual uploads. Organizations face critical challenges:

1. **No centralized freshness tracking** — cannot see which datasets are current vs stale across all sources
2. **No SLA enforcement** — no configurable per-dataset/per-source service-level agreements for data timeliness
3. **No staleness alerting** — no proactive notification when data ages beyond acceptable thresholds
4. **No refresh cadence monitoring** — cannot detect when scheduled refreshes fail or drift from expected frequency
5. **No historical freshness trending** — no burn-rate analytics showing freshness degradation over time
6. **No multi-dataset aggregation** — cannot compute organization-wide freshness scores across source groups
7. **No refresh prediction** — cannot forecast when the next refresh should arrive or detect anomalous delays
8. **No freshness-aware routing** — cannot automatically switch to backup sources when primary goes stale
9. **No compliance documentation** — GHG Protocol/CSRD/ESRS require documented data timeliness processes
10. **No freshness impact analysis** — cannot assess how stale data affects downstream calculation accuracy
11. **No batch monitoring** — cannot efficiently monitor thousands of datasets simultaneously
12. **No provenance trail** — no audit chain for freshness checks, SLA violations, and resolution actions

## 3. Existing Layer 1 Capabilities
- `greenlang.data_quality_profiler.timeliness_tracker.TimelinessTracker` — freshness scoring (5-tier piecewise-linear), SLA compliance checks, stale record detection, update frequency computation, field-level freshness analysis
- `greenlang.data_quality_profiler.models.FreshnessResult` — Pydantic v2 model for freshness check results
- `greenlang.data_quality_profiler.models.FRESHNESS_BOUNDARIES_HOURS` — 4-tier boundary constants (24h/72h/168h/720h)
- `greenlang.data_quality_profiler.models.QualityDimension` — TIMELINESS dimension enum
- `greenlang.data_quality_profiler.models.RuleType` — FRESHNESS rule type enum
- `greenlang.cross_source_reconciliation.source_registry.CADENCE_TIMEDELTAS` — 7 refresh cadence definitions (realtime→annual)
- `greenlang.cross_source_reconciliation.source_registry.SourceRegistryEngine` — timeliness score computation via exponential decay

## 4. Identified Gaps (12)
| # | Gap | Layer 1 | Layer 2 Needed |
|---|-----|---------|----------------|
| 1 | Dataset registry | No centralized dataset catalog | Register datasets with source, owner, refresh cadence, SLA, priority, metadata |
| 2 | SLA definitions | Single default_sla_hours | Per-dataset SLA rules with warning/critical thresholds, escalation policies, business hours awareness |
| 3 | Refresh schedule tracking | Basic update frequency stats | Expected refresh schedules (cron-like), drift detection, missed refresh alerts |
| 4 | Freshness scoring engine | Single-dataset scoring | Multi-dataset aggregated scoring, weighted by priority/criticality, group rollups |
| 5 | Staleness pattern detection | Simple stale record detection | Historical pattern analysis, recurring staleness detection, seasonal patterns, systematic source failures |
| 6 | SLA breach management | Boolean sla_compliant flag | Breach lifecycle (detected→acknowledged→investigating→resolved), severity classification, escalation chains |
| 7 | Refresh prediction | None | Statistical prediction of next expected refresh, anomaly detection for late arrivals |
| 8 | Alert generation | No alerting | Multi-channel alerts (webhook, email, Slack, PagerDuty integration), configurable thresholds, alert deduplication |
| 9 | Freshness dashboard data | No aggregated metrics | Time-series freshness history, heatmaps, SLA compliance trends, source reliability rankings |
| 10 | Impact analysis | No downstream impact | Assess how stale data propagates to calculations, reports, and compliance filings |
| 11 | Compliance reporting | No regulatory documentation | Generate timeliness attestation reports for GHG Protocol, CSRD/ESRS, SOC 2 audits |
| 12 | Provenance | Per-check SHA-256 | End-to-end provenance: check decisions, SLA evaluations, breach lifecycle, resolution actions |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | DatasetRegistryEngine | Register and manage monitored datasets with metadata (name, source, owner, refresh_cadence, sla_config, priority, tags), dataset grouping, bulk registration, health status tracking |
| 2 | SLADefinitionEngine | Define and manage SLA rules per dataset/group: warning threshold (hours), critical threshold (hours), breach severity classification (INFO/LOW/MEDIUM/HIGH/CRITICAL), escalation policies, business hours configuration, SLA templates |
| 3 | FreshnessCheckerEngine | Execute freshness checks: compute age since last update, apply freshness scoring (5-tier), evaluate SLA compliance, batch checking across all registered datasets, incremental checks for changed datasets only |
| 4 | StalenessDetectorEngine | Detect staleness patterns: historical trend analysis, recurring staleness identification, seasonal pattern detection, source reliability scoring, systematic failure detection, refresh drift monitoring |
| 5 | RefreshPredictorEngine | Predict refresh behavior: estimate next expected refresh time based on historical patterns, detect anomalous delays, compute refresh regularity score, identify degrading refresh patterns |
| 6 | AlertManagerEngine | Generate and manage alerts: multi-severity alerts (warning/critical/emergency), alert deduplication and throttling, escalation chain execution, alert lifecycle (open→acknowledged→resolved), notification formatting |
| 7 | FreshnessMonitorPipelineEngine | End-to-end orchestration: register datasets → check freshness → detect staleness → predict refreshes → evaluate SLAs → generate alerts → produce reports, batch processing, checkpoint/resume, scheduled monitoring runs |

### 5.2 Data Flow
```
Registered Datasets → DatasetRegistryEngine (catalog + metadata)
                    → SLADefinitionEngine (SLA rules per dataset)
                    → FreshnessCheckerEngine (compute freshness + SLA compliance)
                    → StalenessDetectorEngine (pattern detection + source reliability)
                    → RefreshPredictorEngine (next refresh prediction + anomaly detection)
                    → AlertManagerEngine (breach alerts + escalation)
                    → FreshnessMonitorPipelineEngine (orchestration + reporting)
```

### 5.3 Database Schema (V046)
- `freshness_datasets` — registered datasets (name, source, owner, cadence, priority, status, metadata)
- `freshness_sla_definitions` — SLA rules per dataset (warning_hours, critical_hours, escalation_policy)
- `freshness_checks` — individual freshness check results (dataset_id, checked_at, age_hours, score, level, sla_status)
- `freshness_refresh_history` — observed refresh timestamps per dataset (dataset_id, refreshed_at, data_size, source)
- `freshness_staleness_patterns` — detected staleness patterns (dataset_id, pattern_type, frequency, severity)
- `freshness_sla_breaches` — SLA breach records (dataset_id, sla_id, breach_severity, detected_at, resolved_at, status)
- `freshness_alerts` — generated alerts (breach_id, alert_type, severity, channel, sent_at, acknowledged_at)
- `freshness_predictions` — refresh predictions (dataset_id, predicted_at, confidence, actual_at, error_hours)
- `freshness_reports` — generated compliance/audit reports
- `freshness_audit_log` — all actions with provenance
- 3 hypertables: `freshness_check_events`, `refresh_events`, `alert_events` (7-day chunks)
- 2 continuous aggregates: `freshness_hourly_stats`, `sla_breach_hourly_stats`

### 5.4 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_dfm_checks_performed_total | Counter | Freshness checks performed by dataset and result |
| gl_dfm_sla_breaches_total | Counter | SLA breaches detected by severity |
| gl_dfm_alerts_sent_total | Counter | Alerts sent by channel and severity |
| gl_dfm_datasets_registered_total | Counter | Datasets registered |
| gl_dfm_refresh_events_total | Counter | Refresh events recorded |
| gl_dfm_predictions_made_total | Counter | Refresh predictions generated |
| gl_dfm_freshness_score | Histogram | Freshness score distribution |
| gl_dfm_data_age_hours | Histogram | Data age distribution in hours |
| gl_dfm_processing_duration_seconds | Histogram | Processing duration by operation |
| gl_dfm_active_breaches | Gauge | Currently active SLA breaches |
| gl_dfm_monitored_datasets | Gauge | Number of actively monitored datasets |
| gl_dfm_processing_errors_total | Counter | Processing errors by type |

### 5.5 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/freshness/datasets | Register dataset for monitoring |
| 2 | GET | /api/v1/freshness/datasets | List monitored datasets |
| 3 | GET | /api/v1/freshness/datasets/{id} | Get dataset details |
| 4 | PUT | /api/v1/freshness/datasets/{id} | Update dataset metadata |
| 5 | DELETE | /api/v1/freshness/datasets/{id} | Remove dataset from monitoring |
| 6 | POST | /api/v1/freshness/sla | Create SLA definition |
| 7 | GET | /api/v1/freshness/sla | List SLA definitions |
| 8 | GET | /api/v1/freshness/sla/{id} | Get SLA details |
| 9 | PUT | /api/v1/freshness/sla/{id} | Update SLA definition |
| 10 | POST | /api/v1/freshness/check | Run freshness check |
| 11 | POST | /api/v1/freshness/check/batch | Run batch freshness check |
| 12 | GET | /api/v1/freshness/checks | List check results |
| 13 | GET | /api/v1/freshness/breaches | List SLA breaches |
| 14 | GET | /api/v1/freshness/breaches/{id} | Get breach details |
| 15 | PUT | /api/v1/freshness/breaches/{id} | Update breach status |
| 16 | GET | /api/v1/freshness/alerts | List alerts |
| 17 | GET | /api/v1/freshness/predictions | Get refresh predictions |
| 18 | POST | /api/v1/freshness/pipeline | Run full monitoring pipeline |
| 19 | GET | /api/v1/freshness/health | Health check |
| 20 | GET | /api/v1/freshness/stats | Service statistics |

### 5.6 Configuration (GL_DFM_ prefix)
| Setting | Default | Description |
|---------|---------|-------------|
| database_url | postgresql://localhost:5432/greenlang | PostgreSQL connection |
| redis_url | redis://localhost:6379/0 | Redis cache connection |
| log_level | INFO | Logging level |
| batch_size | 1000 | Datasets per batch check |
| max_datasets | 50000 | Maximum monitored datasets |
| default_sla_warning_hours | 24.0 | Default warning threshold |
| default_sla_critical_hours | 72.0 | Default critical threshold |
| freshness_excellent_hours | 1.0 | Excellent freshness boundary |
| freshness_good_hours | 6.0 | Good freshness boundary |
| freshness_fair_hours | 24.0 | Fair freshness boundary |
| freshness_poor_hours | 72.0 | Poor freshness boundary |
| check_interval_minutes | 15 | Automatic check interval |
| alert_throttle_minutes | 60 | Minimum time between duplicate alerts |
| alert_dedup_window_hours | 24 | Alert deduplication window |
| prediction_history_days | 90 | Days of refresh history for predictions |
| prediction_min_samples | 5 | Minimum refresh samples for prediction |
| staleness_pattern_window_days | 30 | Window for staleness pattern detection |
| max_workers | 4 | Concurrent worker threads |
| pool_size | 5 | Connection pool size |
| cache_ttl | 300 | Cache TTL in seconds |
| rate_limit | 100 | Max requests per minute |
| enable_provenance | true | Enable SHA-256 provenance chains |
| enable_predictions | true | Enable refresh predictions |
| enable_alerts | true | Enable alert generation |
| escalation_enabled | true | Enable alert escalation |
| genesis_hash | greenlang-data-freshness-monitor-genesis | Provenance chain genesis string |

### 5.7 Layer 1 Re-exports
- `greenlang.data_quality_profiler.timeliness_tracker.TimelinessTracker`
- `greenlang.data_quality_profiler.models.FreshnessResult`
- `greenlang.data_quality_profiler.models.QualityDimension`
- `greenlang.data_quality_profiler.models.RuleType`
- `greenlang.data_quality_profiler.models.FRESHNESS_BOUNDARIES_HOURS`

### 5.8 Provenance Design
- Genesis string: `"greenlang-data-freshness-monitor-genesis"`
- SHA-256 chain: each operation (check, breach, alert, predict, resolve) appends to chain
- Per-dataset freshness history: tracks all checks with timestamps and scores
- SLA breach lifecycle: complete audit trail from detection through resolution
- Deterministic: same inputs + same config = same provenance hash

## 6. Success Criteria
- All 7 engines fully implemented with pure Python (no external ML dependencies)
- 20 REST API endpoints operational
- 12 Prometheus metrics collecting
- V046 database migration applied
- 600+ unit tests passing
- Integration tests with multi-dataset monitoring scenarios
- SHA-256 provenance chains deterministic
- Auth integration complete (PERMISSION_MAP + router)
- K8s manifests, Dockerfile, CI/CD pipeline ready
