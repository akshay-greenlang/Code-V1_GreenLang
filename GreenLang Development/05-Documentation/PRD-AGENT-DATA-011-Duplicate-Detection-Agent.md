# PRD: AGENT-DATA-011 — GL-DATA-X-014 Duplicate Detection Agent

## 1. Overview
| Field | Value |
|-------|-------|
| Agent ID | GL-DATA-X-014 |
| Internal Label | AGENT-DATA-011 |
| Category | Layer 2 – Data Quality Agents |
| Purpose | Identify and merge duplicate records across datasets |
| Estimated Variants | 200 |
| Status | BUILT (100%) |
| Author | GreenLang Platform Team |
| Date | February 2026 |
| Files | 55+ files, ~15K+ lines |
| DB Migration | V041 |

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

## 9. Test Coverage Strategy

### 9.1 Unit Tests (Target: 85%+ coverage)
| Test File | Coverage Scope | Min Tests |
|-----------|---------------|-----------|
| test_config.py | Config defaults, env vars, singleton, validation | 70 |
| test_models.py | Model creation, field constraints, enum values | 50 |
| test_metrics.py | Metric recording, increment, Prometheus export | 20 |
| test_provenance.py | SHA-256 hashing, chain integrity, audit trail | 15 |
| test_record_fingerprinter.py | SHA-256, SimHash, MinHash, field normalization | 50 |
| test_blocking_engine.py | Sorted neighborhood, standard, canopy blocking | 50 |
| test_similarity_scorer.py | 8 algorithms, edge cases, boundary conditions | 60 |
| test_match_classifier.py | Threshold-based, Fellegi-Sunter, weighted scoring | 50 |
| test_cluster_resolver.py | Union-find, connected components, quality metrics | 50 |
| test_merge_engine.py | 6 strategies, conflict resolution, golden record | 60 |
| test_deduplication_pipeline.py | End-to-end pipeline, checkpointing, recovery | 40 |
| test_setup.py | Service facade, all public methods, health | 85 |
| test_router.py | All 20 API endpoints, error handling, validation | 40 |
| **Total** | **All components** | **640+** |

### 9.2 Integration Tests
- E2E pipeline: fingerprint -> block -> compare -> classify -> cluster -> merge
- Database: V041 tables, hypertables, continuous aggregates, RLS policies
- API: All 20 endpoints with auth, tenant isolation
- Concurrency: Thread-safe singleton, parallel job execution
- Performance: 100K records fingerprinting, 50K comparisons/sec target

### 9.3 Test Data Patterns
- Synthetic supplier records with known duplicates (varying similarity)
- Multi-dataset cross-references (facility, vendor, product)
- Edge cases: empty fields, Unicode, special characters, numeric-only

## 10. Security & Data Privacy

### 10.1 Authentication & Authorization
- All 20 API endpoints protected via SEC-001 JWT Authentication
- RBAC permissions: `dedup:jobs:create`, `dedup:jobs:read`, `dedup:matches:read`, `dedup:merge:execute`, `dedup:rules:manage`, `dedup:admin`
- Tenant isolation enforced via RLS policies on all 10 tables + 3 hypertables

### 10.2 Data Privacy
- PII fields (names, addresses, emails) hashed before fingerprinting
- Merge decisions logged with full provenance (who, when, what, why)
- Audit trail immutable via SHA-256 chain hashing
- Data retention: 90 days for events, configurable per tenant
- GDPR: Right to erasure supported via cascade delete with audit

### 10.3 Encryption
- All data at rest encrypted via SEC-003 AES-256-GCM
- Fingerprints stored as cryptographic hashes (non-reversible)
- API transport via SEC-004 TLS 1.3

## 11. Performance & Scalability

### 11.1 Benchmark Targets
| Metric | Target | Method |
|--------|--------|--------|
| Fingerprinting throughput | 50K records/sec | Batch SHA-256 with normalization |
| Blocking efficiency | O(n*k) where k << n | Sorted neighborhood, standard blocking |
| Comparison throughput | 100K pairs/sec | Vectorized similarity scoring |
| Cluster resolution | < 1s for 10K nodes | Union-find with path compression |
| Merge execution | 10K records/sec | Batch merge with conflict resolution |
| API latency (p95) | < 200ms | Connection pooling, Redis caching |
| API latency (p99) | < 500ms | Background job processing for large datasets |

### 11.2 Scalability
- Horizontal: K8s HPA scales 2-10 replicas based on CPU/memory
- Vertical: Worker count configurable (default 4, max 16)
- Batch processing: Configurable batch_size (default 1000, max 100K)
- Pipeline checkpointing: Resume from last checkpoint on failure

## 12. Rollout & Migration Plan

### Phase 1: Core SDK (Complete)
- [x] 7 engines implemented
- [x] Configuration with GL_DD_ env prefix
- [x] Models (30+ Pydantic v2 models)
- [x] Provenance tracking (SHA-256)
- [x] Metrics (12 Prometheus metrics)
- [x] Service facade (DuplicateDetectorService)
- [x] API router (20 REST endpoints)

### Phase 2: Infrastructure (Complete)
- [x] V041 database migration (10 tables, 3 hypertables, 2 aggregates)
- [x] K8s manifests (deployment, service, configmap, secret, HPA, PDB, NetworkPolicy)
- [x] ServiceMonitor + PrometheusRule alerts
- [x] Grafana dashboard
- [x] CI/CD pipeline (lint, typecheck, unit test, security scan, migration validate)

### Phase 3: Integration (Complete)
- [x] Auth integration (PERMISSION_MAP entries for all 20 endpoints)
- [x] Integration tests (30+ tests covering E2E, DB, API)
- [x] Dockerfile for container image build
- [x] CI/CD integration test job enabled

### Phase 4: Production Readiness
- [x] Unit tests passing (640+, 85%+ coverage)
- [x] Integration tests passing
- [x] Security review complete
- [x] Performance benchmarks met
- [x] Documentation complete

## Appendix A: Glossary
| Term | Definition |
|------|-----------|
| Fingerprint | A compact hash representation of a record's key fields, used for fast dedup detection |
| Blocking | Grouping records into blocks to reduce O(n^2) pairwise comparisons to O(n*k) |
| Sorted Neighborhood | Blocking strategy that sorts records by a key and compares within a sliding window |
| Transitive Closure | If A=B and B=C, then A=C — used to form duplicate clusters |
| Golden Record | The single "best" record created by merging all duplicates in a cluster |
| Conflict Resolution | Strategy for resolving disagreements when merging field values from duplicates |
| SimHash | Locality-sensitive hash that produces similar hashes for similar inputs |
| MinHash | Probabilistic technique for estimating Jaccard similarity between sets |
| Fellegi-Sunter | Probabilistic record linkage model using match/non-match likelihood ratios |
| Union-Find | Disjoint set data structure for efficient cluster formation via union and find operations |

## Appendix B: Configuration Reference
| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| GL_DD_DATABASE_URL | str | "" | PostgreSQL connection string |
| GL_DD_REDIS_URL | str | "" | Redis connection string |
| GL_DD_LOG_LEVEL | str | "INFO" | Logging level |
| GL_DD_BATCH_SIZE | int | 1000 | Records per batch |
| GL_DD_MAX_RECORDS | int | 100000 | Maximum records per job |
| GL_DD_MATCH_THRESHOLD | float | 0.85 | Minimum similarity for match |
| GL_DD_POSSIBLE_THRESHOLD | float | 0.65 | Minimum similarity for possible match |
| GL_DD_BLOCKING_WINDOW | int | 10 | Sorted neighborhood window size |
| GL_DD_DEFAULT_STRATEGY | str | "keep_most_complete" | Default merge strategy |
| GL_DD_ENABLE_SIMHASH | bool | True | Enable SimHash fingerprinting |
| GL_DD_ENABLE_MINHASH | bool | True | Enable MinHash fingerprinting |
| GL_DD_WORKER_COUNT | int | 4 | Parallel worker count |
| GL_DD_POOL_MIN_SIZE | int | 2 | Connection pool minimum |
| GL_DD_POOL_MAX_SIZE | int | 10 | Connection pool maximum |
| GL_DD_CACHE_TTL | int | 3600 | Cache TTL in seconds |
| GL_DD_RATE_LIMIT_RPM | int | 120 | API rate limit per minute |
| GL_DD_ENABLE_PROVENANCE | bool | True | Enable SHA-256 provenance chains |
