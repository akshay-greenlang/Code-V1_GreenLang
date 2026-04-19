# PRD: AGENT-DATA-015 — GL-DATA-X-018 Cross-Source Reconciliation

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-018 |
| Internal Label | AGENT-DATA-015 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Reconcile sustainability data across multiple sources (utility bills, ERP, meters, questionnaires) by matching records, detecting discrepancies, resolving conflicts, and producing auditable golden records with full provenance |
| Estimated Variants | 200 |
| Status | To Be Built |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| DB Migration | V045 |

## 2. Problem Statement
Sustainability datasets are sourced from multiple systems — ERP extracts, utility provider data, IoT meters, supplier questionnaires, government registries, and manual spreadsheets. These sources frequently disagree:

1. **No source matching** — cannot identify which records across sources refer to the same entity/period
2. **No schema alignment** — different column names, units, date formats, currencies across sources
3. **No discrepancy detection** — cannot systematically find where sources disagree on the same metric
4. **No conflict classification** — cannot distinguish critical conflicts (>50% deviation) from minor rounding differences
5. **No resolution rules** — no configurable priority/strategy for which source to trust per field
6. **No golden record generation** — cannot produce a single authoritative record from multiple conflicting sources
7. **No source credibility scoring** — no systematic assessment of source reliability, timeliness, completeness
8. **No temporal alignment** — cannot reconcile monthly utility bills against daily meter readings
9. **No tolerance-aware comparison** — cannot distinguish acceptable variance (rounding, unit conversion) from real discrepancies
10. **No audit trail** — no provenance tracking for which source contributed which field to the reconciled record
11. **No batch reconciliation** — cannot process thousands of entity-period combinations across sources
12. **No regulatory documentation** — GHG Protocol/CSRD/ESRS require documented data quality processes

## 3. Existing Layer 1 Capabilities
- `greenlang.data_quality_profiler.consistency_analyzer.ConsistencyAnalyzer` — cross-dataset comparison, referential integrity, schema drift detection, distribution comparison (KS-test)
- `greenlang.duplicate_detector.similarity_scorer.SimilarityScorer` — 8 similarity algorithms (exact, Levenshtein, Jaro-Winkler, Soundex, n-gram, TF-IDF, numeric proximity, date proximity)
- `greenlang.duplicate_detector.blocking_engine.BlockingEngine` — 3 blocking strategies (standard, sorted neighborhood, canopy)
- `greenlang.duplicate_detector.match_classifier.MatchClassifier` — Fellegi-Sunter probabilistic scoring, auto-threshold, weighted aggregation
- `greenlang.duplicate_detector.cluster_resolver.ClusterResolver` — Union-find clustering, representative selection, cluster quality scoring
- `greenlang.duplicate_detector.merge_engine.MergeEngine` — 6 merge strategies (keep_first, keep_latest, keep_most_complete, merge_fields, golden_record, custom)
- `greenlang.duplicate_detector.record_fingerprinter.RecordFingerprinter` — SHA-256/SimHash/MinHash, field normalization (5 types)
- `greenlang.missing_value_imputer.missingness_analyzer.MissingnessAnalyzerEngine` — MCAR/MAR/MNAR classification, missingness correlation, pattern detection
- `greenlang.outlier_detector.statistical_detector.StatisticalDetectorEngine` — 7 outlier methods (IQR, z-score, MAD, Grubbs, Tukey, percentile, ensemble)
- `greenlang.data.data_engineering.reconciliation.factor_reconciliation.FactorReconciler` — conflict resolution strategies, source priority tiers, conflict severity classification

## 4. Identified Gaps (12)
| # | Gap | Layer 1 | Layer 2 Needed |
|---|-----|---------|----------------|
| 1 | Source registry | No source metadata management | Register sources with schema, priority, credibility, refresh cadence |
| 2 | Schema alignment | Basic schema drift detection | Automatic column mapping, unit conversion, currency normalization, date format alignment |
| 3 | Record matching | Duplicate detection focus | Cross-source entity-period matching with composite keys, temporal alignment, fuzzy join |
| 4 | Field-level comparison | Basic distribution comparison | Tolerance-aware field comparison (absolute, relative, percentage thresholds per field) |
| 5 | Discrepancy detection | No systematic classification | Classify: value_mismatch, missing_in_source, extra_in_source, timing_difference, unit_difference, aggregation_mismatch |
| 6 | Conflict resolution | Factor-specific only | Configurable per-field resolution: priority_wins, most_recent, average, weighted_average, most_complete, consensus, manual_review |
| 7 | Golden record | Merge engine for dedup only | Cross-source golden record: best field from best source, confidence per field, full lineage |
| 8 | Source credibility | Static priority tiers | Dynamic scoring: completeness, timeliness, consistency history, error rate, certification level |
| 9 | Temporal alignment | None | Aggregate/disaggregate to common frequency, interpolate to align periods |
| 10 | Tolerance configuration | None | Per-field tolerance rules (absolute diff, relative %, rounding tolerance, unit conversion epsilon) |
| 11 | Reconciliation reports | No compliance documentation | Generate reconciliation summary, discrepancy log, resolution justification, regulatory attestation |
| 12 | Provenance | Per-module SHA-256 | End-to-end provenance: which source contributed which field, resolution decision chain, confidence inheritance |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | SourceRegistryEngine | Register data sources with metadata (name, type, schema, priority, refresh cadence, credibility score), schema mapping rules, column aliases, unit/currency/date format definitions, source health tracking |
| 2 | MatchingEngine | Cross-source record matching using composite keys (entity+period+metric), fuzzy key matching (Jaro-Winkler, n-gram), temporal alignment (daily↔monthly↔quarterly↔annual), blocking for scalability, match confidence scoring |
| 3 | ComparisonEngine | Field-by-field comparison of matched records with configurable tolerances (absolute, relative, percentage), unit-aware comparison (kg vs tonnes, MWh vs kWh), currency-aware comparison (multi-currency normalization), null handling, aggregation-level comparison |
| 4 | DiscrepancyDetectorEngine | Detect and classify discrepancies: value_mismatch (exceeds tolerance), missing_in_source (present in A but not B), extra_in_source, timing_difference, unit_difference, aggregation_mismatch, severity scoring (CRITICAL/HIGH/MEDIUM/LOW/INFO), pattern detection (systematic bias per source) |
| 5 | ResolutionEngine | Apply configurable resolution strategies per field: priority_wins (highest credibility source), most_recent (latest timestamp), weighted_average (credibility-weighted), most_complete (fewest nulls), consensus (majority vote), manual_review (flag for human), golden record assembly with per-field lineage |
| 6 | AuditTrailEngine | Complete audit trail: every match decision, comparison result, discrepancy classification, resolution choice, golden record field selection with source attribution, regulatory attestation generation, compliance report assembly |
| 7 | ReconciliationPipelineEngine | End-to-end orchestration: register sources → align schemas → match records → compare fields → detect discrepancies → resolve conflicts → assemble golden records → generate audit trail, batch processing, checkpoint/resume, strategy auto-selection |

### 5.2 Data Flow
```
Multiple Data Sources → SourceRegistryEngine (register + schema mapping)
                      → MatchingEngine (cross-source record matching)
                      → ComparisonEngine (field-by-field tolerance-aware comparison)
                      → DiscrepancyDetectorEngine (classify conflicts + severity)
                      → ResolutionEngine (apply rules → golden records)
                      → AuditTrailEngine (provenance + compliance reports)
                      → ReconciliationPipelineEngine (orchestration + batch)
```

### 5.3 Database Schema (V045)
- `reconciliation_jobs` — job tracking (source refs, status, config, match count)
- `reconciliation_sources` — registered data sources (name, type, schema, priority, credibility)
- `reconciliation_schema_maps` — column mapping rules between sources (source_col → canonical_col, transform)
- `reconciliation_matches` — matched record pairs/groups across sources (entity, period, confidence)
- `reconciliation_comparisons` — field-level comparison results (field, source_a_val, source_b_val, diff, tolerance, pass/fail)
- `reconciliation_discrepancies` — detected discrepancies (type, severity, field, sources involved, values)
- `reconciliation_resolutions` — resolution decisions (strategy, winning_source, resolved_value, justification)
- `reconciliation_golden_records` — assembled golden records (entity, period, field values, per-field source attribution)
- `reconciliation_reports` — generated compliance/audit reports
- `reconciliation_audit_log` — all actions with provenance
- 3 hypertables: `reconciliation_events`, `comparison_events`, `resolution_events` (7-day chunks)
- 2 continuous aggregates: `reconciliation_hourly_stats`, `discrepancy_hourly_stats`

### 5.4 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_csr_jobs_processed_total | Counter | Reconciliation jobs processed by status |
| gl_csr_records_matched_total | Counter | Records matched across sources |
| gl_csr_comparisons_total | Counter | Field comparisons performed |
| gl_csr_discrepancies_detected_total | Counter | Discrepancies detected by type and severity |
| gl_csr_resolutions_applied_total | Counter | Resolution strategies applied by type |
| gl_csr_golden_records_created_total | Counter | Golden records assembled |
| gl_csr_match_confidence | Histogram | Match confidence score distribution |
| gl_csr_processing_duration_seconds | Histogram | Processing duration by operation |
| gl_csr_discrepancy_magnitude | Histogram | Discrepancy magnitude distribution (% deviation) |
| gl_csr_active_jobs | Gauge | Currently active reconciliation jobs |
| gl_csr_pending_reviews | Gauge | Discrepancies pending manual review |
| gl_csr_processing_errors_total | Counter | Processing errors by type |

### 5.5 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/reconciliation/jobs | Create reconciliation job |
| 2 | GET | /api/v1/reconciliation/jobs | List jobs |
| 3 | GET | /api/v1/reconciliation/jobs/{id} | Get job details |
| 4 | DELETE | /api/v1/reconciliation/jobs/{id} | Cancel/delete job |
| 5 | POST | /api/v1/reconciliation/sources | Register data source |
| 6 | GET | /api/v1/reconciliation/sources | List registered sources |
| 7 | GET | /api/v1/reconciliation/sources/{id} | Get source details |
| 8 | PUT | /api/v1/reconciliation/sources/{id} | Update source metadata |
| 9 | POST | /api/v1/reconciliation/match | Match records across sources |
| 10 | GET | /api/v1/reconciliation/matches | List match results |
| 11 | GET | /api/v1/reconciliation/matches/{id} | Get match details |
| 12 | POST | /api/v1/reconciliation/compare | Compare matched records |
| 13 | GET | /api/v1/reconciliation/discrepancies | List discrepancies |
| 14 | GET | /api/v1/reconciliation/discrepancies/{id} | Get discrepancy details |
| 15 | POST | /api/v1/reconciliation/resolve | Resolve discrepancies |
| 16 | GET | /api/v1/reconciliation/golden-records | List golden records |
| 17 | GET | /api/v1/reconciliation/golden-records/{id} | Get golden record details |
| 18 | POST | /api/v1/reconciliation/pipeline | Run full reconciliation pipeline |
| 19 | GET | /api/v1/reconciliation/health | Health check |
| 20 | GET | /api/v1/reconciliation/stats | Service statistics |

### 5.6 Configuration (GL_CSR_ prefix)
| Setting | Default | Description |
|---------|---------|-------------|
| database_url | postgresql://localhost:5432/greenlang | PostgreSQL connection |
| redis_url | redis://localhost:6379/0 | Redis cache connection |
| log_level | INFO | Logging level |
| batch_size | 1000 | Records per batch |
| max_records | 100000 | Maximum records per job |
| max_sources | 20 | Maximum sources per reconciliation |
| default_match_threshold | 0.85 | Default match confidence threshold |
| default_tolerance_pct | 5.0 | Default relative tolerance (%) |
| default_tolerance_abs | 0.01 | Default absolute tolerance |
| default_resolution_strategy | priority_wins | Default conflict resolution strategy |
| source_credibility_weight | 0.4 | Weight for credibility in resolution |
| temporal_alignment_enabled | true | Enable temporal period alignment |
| fuzzy_matching_enabled | true | Enable fuzzy key matching |
| max_match_candidates | 100 | Max candidates per blocking key |
| enable_golden_records | true | Enable golden record assembly |
| max_workers | 4 | Concurrent worker threads |
| pool_size | 5 | Connection pool size |
| cache_ttl | 3600 | Cache TTL in seconds |
| rate_limit | 100 | Max requests per minute |
| enable_provenance | true | Enable SHA-256 provenance chains |
| manual_review_threshold | 0.6 | Below this confidence, flag for manual review |
| critical_discrepancy_pct | 50.0 | Deviation % for CRITICAL severity |
| high_discrepancy_pct | 25.0 | Deviation % for HIGH severity |
| medium_discrepancy_pct | 10.0 | Deviation % for MEDIUM severity |
| genesis_hash | greenlang-cross-source-reconciliation-genesis | Provenance chain genesis string |

### 5.7 Layer 1 Re-exports
- `greenlang.data_quality_profiler.consistency_analyzer.ConsistencyAnalyzer`
- `greenlang.duplicate_detector.similarity_scorer.SimilarityScorer`
- `greenlang.duplicate_detector.match_classifier.MatchClassifier`
- `greenlang.data.data_engineering.reconciliation.factor_reconciliation.FactorReconciler`
- `greenlang.data.data_engineering.reconciliation.factor_reconciliation.ConflictResolutionStrategy`

### 5.8 Provenance Design
- Genesis string: `"greenlang-cross-source-reconciliation-genesis"`
- SHA-256 chain: each operation (match, compare, detect, resolve) appends to chain
- Per-field attribution: golden record tracks which source contributed each field
- Resolution justification: logged with strategy name, confidence, source credibility scores
- Deterministic: same inputs + same config = same provenance hash

## 6. Success Criteria
- All 7 engines fully implemented with pure Python (no external ML dependencies)
- 20 REST API endpoints operational
- 12 Prometheus metrics collecting
- V045 database migration applied
- 600+ unit tests passing
- Integration tests with multi-source reconciliation scenarios
- SHA-256 provenance chains deterministic
- Auth integration complete (PERMISSION_MAP + router)
- K8s manifests, Dockerfile, CI/CD pipeline ready
