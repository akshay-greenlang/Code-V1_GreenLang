# PRD: AGENT-DATA-011 — GL-DATA-X-014 Duplicate Detection Agent

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-014 |
| Internal Label | AGENT-DATA-011 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Identify and merge duplicate records across datasets |
| Estimated Variants | 200 |
| Status | TO BUILD |

## 2. Problem Statement
Duplicate records across datasets (supplier lists, emission inventories, facility registries) lead to double-counted emissions, inflated Scope 3 totals, and non-compliant regulatory disclosures. Existing Layer 1 deduplication (TransformEngine.deduplicate, SpendIngestion fuzzy dedup) only supports exact-key or simple Levenshtein matching. A dedicated agent is needed for multi-algorithm similarity scoring, blocking strategies for O(n) performance, configurable match thresholds, cluster resolution via transitive closure, and sophisticated merge strategies with full provenance.

## 3. Existing Layer 1 Capabilities
- `greenlang.excel_normalizer.transform_engine.TransformEngine.deduplicate()` — exact-key deduplication with keep-first/keep-last
- `greenlang.spend_categorizer.spend_ingestion.SpendIngestionEngine` — vendor name fuzzy dedup via Levenshtein
- `greenlang.normalizer.entity_resolver.EntityResolver` — 3-tier matching (exact/alias/fuzzy) for entity resolution
- `greenlang.infrastructure.security_scanning.deduplication.DeduplicationEngine` — CVE/fingerprint dedup pattern

## 4. Identified Gaps (12)
| # | Gap | Layer 1 | Layer 2 Needed |
|---|-----|---------|----------------|
| 1 | Record fingerprinting | None | Field-set based SHA-256 / SimHash fingerprints |
| 2 | Blocking strategies | None | Sorted neighborhood, standard blocking, canopy |
| 3 | Multi-algorithm similarity | Levenshtein only | Jaro-Winkler, Soundex, n-gram, TF-IDF cosine, numeric proximity |
| 4 | Weighted field scoring | None | Configurable per-field weights and algorithms |
| 5 | Match classification | Binary only | Match/non-match/possible with thresholds |
| 6 | Transitive closure clustering | None | Connected components, cluster quality metrics |
| 7 | Merge strategies | Keep-first/last only | Keep-most-complete, merge-fields, golden-record |
| 8 | Cross-dataset dedup | Single dataset only | Multi-dataset comparison with source tracking |
| 9 | Duplicate group management | None | Review, approve, reject, split groups |
| 10 | Performance optimization | O(n²) pairwise | Blocking reduces to O(n·k) |
| 11 | Dedup pipeline orchestration | None | End-to-end pipeline with checkpointing |
| 12 | Merge decision provenance | None | SHA-256 chain tracking every merge decision |

## 5. Architecture

### 5.1 Seven Engines
| Engine | Class | Responsibility |
|--------|-------|----------------|
| 1 | RecordFingerprinter | Generate fingerprints (SHA-256, SimHash) from configurable field sets, normalize inputs before hashing |
| 2 | BlockingEngine | Reduce comparison space via sorted neighborhood (window), standard blocking (key functions), canopy clustering |
| 3 | SimilarityScorer | Multi-algorithm similarity: exact, Levenshtein, Jaro-Winkler, Soundex, n-gram Jaccard, TF-IDF cosine, numeric proximity, date proximity |
| 4 | MatchClassifier | Weighted field scoring, threshold-based classification (match/non-match/possible), Fellegi-Sunter inspired scoring |
| 5 | ClusterResolver | Connected components via union-find, transitive closure, cluster quality metrics (density, diameter), cluster splitting |
| 6 | MergeEngine | 6 merge strategies (keep_first, keep_latest, keep_most_complete, merge_fields, golden_record, custom), conflict resolution |
| 7 | DeduplicationPipeline | End-to-end orchestration: fingerprint -> block -> compare -> classify -> cluster -> merge, with checkpointing and stats |

### 5.2 Database Schema (V041)
- `dedup_jobs` — deduplication job tracking (dataset refs, status, config)
- `dedup_fingerprints` — record fingerprints (SHA-256, SimHash per field set)
- `dedup_blocks` — blocking results (block key, record count)
- `dedup_comparisons` — pairwise comparison results (similarity scores per field)
- `dedup_matches` — classified matches (record pair, score, classification)
- `dedup_clusters` — duplicate clusters (cluster ID, member records, quality)
- `dedup_merge_decisions` — merge decisions (cluster, strategy, merged record, provenance)
- `dedup_merge_conflicts` — field-level conflicts during merge
- `dedup_rules` — configurable dedup rule sets (field weights, algorithms, thresholds)
- `dedup_audit_log` — all dedup actions with provenance
- 3 hypertables: `dedup_events`, `comparison_events`, `merge_events` (7-day chunks)
- 2 continuous aggregates: `dedup_hourly_stats`, `comparison_hourly_stats`

### 5.3 Prometheus Metrics (12)
| Metric | Type | Description |
|--------|------|-------------|
| gl_dd_jobs_processed_total | Counter | Jobs processed by status |
| gl_dd_records_fingerprinted_total | Counter | Records fingerprinted |
| gl_dd_blocks_created_total | Counter | Blocks created by strategy |
| gl_dd_comparisons_performed_total | Counter | Pairwise comparisons |
| gl_dd_matches_found_total | Counter | Matches by classification |
| gl_dd_clusters_formed_total | Counter | Clusters formed |
| gl_dd_merges_completed_total | Counter | Merges by strategy |
| gl_dd_merge_conflicts_total | Counter | Field conflicts during merge |
| gl_dd_processing_duration_seconds | Histogram | Processing duration by operation |
| gl_dd_similarity_score | Histogram | Similarity score distribution |
| gl_dd_active_jobs | Gauge | Currently active jobs |
| gl_dd_processing_errors_total | Counter | Errors by type |

### 5.4 API Endpoints (20)
| # | Method | Path | Purpose |
|---|--------|------|---------|
| 1 | POST | /api/v1/dedup/jobs | Create dedup job |
| 2 | GET | /api/v1/dedup/jobs | List dedup jobs |
| 3 | GET | /api/v1/dedup/jobs/{id} | Get job details |
| 4 | DELETE | /api/v1/dedup/jobs/{id} | Cancel/delete job |
| 5 | POST | /api/v1/dedup/fingerprint | Fingerprint records |
| 6 | POST | /api/v1/dedup/block | Create blocks |
| 7 | POST | /api/v1/dedup/compare | Compare record pairs |
| 8 | POST | /api/v1/dedup/classify | Classify matches |
| 9 | GET | /api/v1/dedup/matches | List matches |
| 10 | GET | /api/v1/dedup/matches/{id} | Get match details |
| 11 | POST | /api/v1/dedup/clusters | Form clusters |
| 12 | GET | /api/v1/dedup/clusters | List clusters |
| 13 | GET | /api/v1/dedup/clusters/{id} | Get cluster details |
| 14 | POST | /api/v1/dedup/merge | Execute merge |
| 15 | GET | /api/v1/dedup/merge/{id} | Get merge result |
| 16 | POST | /api/v1/dedup/pipeline | Run full pipeline |
| 17 | POST | /api/v1/dedup/rules | Create dedup rule set |
| 18 | GET | /api/v1/dedup/rules | List rule sets |
| 19 | GET | /api/v1/dedup/health | Health check |
| 20 | GET | /api/v1/dedup/stats | Service statistics |

## 6. Layer 1 Re-exports
The SDK will re-export from existing Layer 1:
- `greenlang.normalizer.entity_resolver.EntityResolver` — for entity resolution patterns
- `greenlang.normalizer.models.ConfidenceLevel, EntityMatch` — for confidence scoring models

## 7. Success Criteria
- 7 engines with full unit test coverage (≥85%)
- 8+ similarity algorithms with deterministic outputs
- 3 blocking strategies reducing O(n²) to O(n·k)
- Union-find cluster resolution with quality metrics
- 6 merge strategies with conflict tracking
- SHA-256 provenance chain on all operations
- 12 Prometheus metrics, 20 REST endpoints
- V041 migration with 10+ tables, 3 hypertables, 2 continuous aggregates
- K8s manifests + CI/CD pipeline

## 8. Integration Points
- AGENT-DATA-010 (Data Quality Profiler) — quality assessment pre/post dedup
- AGENT-DATA-002 (Excel/CSV Normalizer) — normalized data as input
- AGENT-DATA-009 (Spend Data Categorizer) — vendor deduplication
- AGENT-FOUND-003 (Unit Normalizer) — entity resolution patterns
- AGENT-FOUND-008 (Reproducibility) — deterministic dedup verification
- AGENT-FOUND-005 (Citations) — provenance chain integration
