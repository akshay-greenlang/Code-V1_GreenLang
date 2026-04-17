# GreenLang Factors FY27 — Product Development Plan

**Date**: 2026-04-17
**Status**: ACTIVE — Execution Ready
**Owner**: Product + CTO
**FY27 Target**: 100K+ catalog rows | 40-60K certified | 20-40K connector-backed

---

## PART 1: CURRENT STATE ASSESSMENT

### What Has Been Built (CTO Sprint — Complete)

| Layer | Component | Files | LOC | Status |
|-------|-----------|-------|-----|--------|
| Core Runtime | service.py, cli.py, __init__.py, __main__.py | 4 | ~360 | PRODUCTION |
| Repository | catalog_repository.py (SQLite + Memory) | 1 | ~993 | PRODUCTION |
| Governance | source_registry.py, approval_gate.py, backfill.py, edition_manifest.py | 4 | ~350 | PRODUCTION |
| Quality | validators.py, review_queue.py, promotion.py, release_signoff.py, audit_export.py | 5 | ~145 | PRODUCTION |
| ETL | normalize.py (CBAM + DEFRA), ingest.py, qa.py | 3 | ~400 | PRODUCTION |
| Ingestion | fetchers.py, tabular_fetchers.py, artifacts.py, normalizer.py, parser_harness.py, sqlite_metadata.py | 6 | ~250 | PRODUCTION |
| Matching | pipeline.py (token overlap + DQS), semantic_index.py (noop stub) | 2 | ~120 | PARTIAL |
| Ontology | units.py, geography.py, methodology.py | 3 | ~130 | PRODUCTION |
| Watch | source_watch.py, change_classification.py, changelog_draft.py, doc_diff.py, rollback_edition.py | 5 | ~110 | PRODUCTION |
| Ancillary | cache_redis.py, dedupe_rules.py, metering.py, policy_mapping.py, tenant_overlay.py | 5 | ~270 | PRODUCTION |
| Billing | usage_sink.py | 1 | ~48 | PRODUCTION |
| SDK | sdk/__init__.py | 1 | ~60 | STUB |
| Bounded Contexts | bounded_contexts/__init__.py | 1 | ~27 | PRODUCTION |
| DB Migrations | V426 (core), V427 (CTO extensions), V428 (license tags) | 3 SQL | ~129 | PRODUCTION |
| Tests | 5 test files + 1 fixture | 6 | ~384 | GOOD |
| Scripts | load_smoke.py, match_eval.py | 2 | ~81 | PRODUCTION |
| Docs | PRD, commercial matrix, personas, release policy, runbook, etc. | 9 | ~273 | COMPLETE |
| Data Record | EmissionFactorRecord (v2 schema, multi-gas, GWP, DQS, licensing) | 1 | ~1,339 | PRODUCTION |
| Data DB | EmissionFactorDatabase (built-in 327+ factors) | 1 | ~500+ | PRODUCTION |
| API | FastAPI main.py + models.py (8 endpoints, JWT, rate limiting) | 2 | ~350+ | PRODUCTION |
| **TOTAL** | **49 Python modules + 3 SQL + 9 docs + 2 scripts** | **65+ files** | **~5,500+** | **95% PROD** |

### Current Factor Count: ~327 built-in
### Target Factor Count: 100,000+
### Gap: ~99,673 factors need to be ingested, normalized, and QA'd

### Architecture Readiness Score

```
Source Registry ............ DONE (13 sources registered)
Ingestion Pipeline ......... DONE (framework + 2 parsers: CBAM, DEFRA)
ETL Normalization .......... DONE (canonical schema + content hashing)
Quality Gates (Q1-Q6) ...... DONE (validators, promotion, review queue)
Approval Gates (G5-G6) ..... DONE (legal, export, certification)
Catalog Repository ......... DONE (SQLite + Memory, editions, search)
Matching Pipeline .......... PARTIAL (lexical only; semantic stub)
Watch Engine ............... DONE (monitoring, classification, changelog)
API Layer .................. DONE (8 endpoints, auth, rate limiting)
DB Migrations .............. DONE (V426-V428, Postgres + SQLite parity)
SDK ........................ STUB (placeholder client)
Tenant Overlay ............. STUB (enterprise multi-tenancy)
Cron/Scheduler ............. MISSING (manual CLI only)
CI/CD Integration .......... MISSING (tests not in pipeline)
Monitoring/Prometheus ...... MISSING (in-process counters only)
Postgres Repository ........ MISSING (SQLite only; migrations ready)
```

---

## PART 2: DETAILED PRODUCT DEVELOPMENT TO-DO LIST

### Organized: Basics -> Intermediate -> Advanced -> Enterprise -> Scale

---

## PHASE 1: FOUNDATION HARDENING (Weeks 1-3)
> Goal: Make what exists bullet-proof before scaling

### 1.1 Test Suite Expansion
- [ ] **TASK-F001**: Expand unit test coverage from 13 to 80+ test functions
  - [ ] Test every public function in `catalog_repository.py` (10 methods x 3 scenarios each)
  - [ ] Test ETL pipeline end-to-end (CBAM ingestion -> search -> retrieval)
  - [ ] Test DEFRA pipeline end-to-end
  - [ ] Test edition comparison with actual content changes
  - [ ] Test replacement chain walking (3+ levels deep)
  - [ ] Test dedup rules with conflicting factors
  - [ ] Test policy mapping CRUD operations
  - [ ] Test metering counter accuracy under concurrency
  - [ ] Test Redis cache hit/miss/eviction paths
  - [ ] Test CLI subcommands (inventory, manifest, ingest-builtin, watch-dry-run, validate-registry, ingest-paths)
  - [ ] Test billing usage_sink with mock SQLite
  - [ ] Test backfill_missing_governance for all license patterns
  - [ ] Test change_classification for numeric/metadata/docs-only
  - [ ] Test changelog_draft output format
  - [ ] Add negative test cases: malformed JSON, missing fields, invalid dates, overflow values

- [ ] **TASK-F002**: Expand gold evaluation set from 2 to 100+ cases
  - [ ] Add 20 Scope 1 cases (diesel, gasoline, natural gas, coal, propane, kerosene, jet fuel, etc.)
  - [ ] Add 15 Scope 2 cases (electricity grids: US, EU, UK, DE, FR, JP, CN, IN, etc.)
  - [ ] Add 20 Scope 3 cases (business travel, employee commuting, purchased goods, freight, waste)
  - [ ] Add 10 CBAM-specific cases (steel, aluminum, cement, fertilizer, electricity imports)
  - [ ] Add 10 edge cases (ambiguous descriptions, multi-word fuel types, regional vs national)
  - [ ] Add 10 negative cases (nonsense queries, empty strings, SQL injection attempts)
  - [ ] Add geography variants (US vs US-CA vs global vs EU27)
  - [ ] Document expected precision@1 >= 0.85 for starter, >= 0.90 for v1 GA
  - [ ] Add recall@5 metric to eval script

- [ ] **TASK-F003**: Add pytest fixtures and conftest.py
  - [ ] Create `tests/factors/conftest.py` with shared fixtures
  - [ ] Fixture: `tmp_sqlite_catalog` (pre-loaded with built-in + CBAM + DEFRA)
  - [ ] Fixture: `source_registry` (loaded from YAML)
  - [ ] Fixture: `sample_edition_manifest`
  - [ ] Fixture: `sample_factor_record` (certified, preview, connector-only, deprecated)
  - [ ] Fixture: `api_test_client` (FastAPI TestClient with env vars set)

### 1.2 CI/CD Integration
- [ ] **TASK-F004**: Add factors tests to GitHub Actions pipeline
  - [ ] Create `.github/workflows/factors-tests.yml`
  - [ ] Run: `pytest tests/factors/ -v --tb=short`
  - [ ] Run: `python scripts/factors_match_eval.py` (precision gate: fail if < 0.80)
  - [ ] Run: `python -m greenlang.factors.cli validate-registry` (fail if issues found)
  - [ ] Run: `python -m greenlang.factors.cli inventory --out /tmp/coverage.json` (smoke test)
  - [ ] Add badge to README for factors test status
  - [ ] Gate on: all tests pass, eval precision >= threshold, registry valid

- [ ] **TASK-F005**: Add pre-commit hooks for factors
  - [ ] Lint check: all new factors files pass ruff/flake8
  - [ ] Type check: mypy on `greenlang/factors/` (strict mode)
  - [ ] Schema check: `validate-registry` passes
  - [ ] Content hash determinism: manifest fingerprint reproducible

### 1.3 Error Handling & Logging
- [ ] **TASK-F006**: Add structured logging to all factors modules
  - [ ] Replace any remaining f-string logs with %-format (per project standard)
  - [ ] Add `logger = logging.getLogger(__name__)` to every module
  - [ ] Log at appropriate levels: INFO for operations, WARNING for degraded, ERROR for failures
  - [ ] Add correlation IDs to ingestion pipeline (run_id propagation)
  - [ ] Log edition resolution decisions (which source won: env/header/query/default)

- [ ] **TASK-F007**: Add factors-specific exceptions
  - [ ] Create `greenlang/exceptions/factors.py`
  - [ ] Define: `FactorNotFoundError`, `EditionNotFoundError`, `EditionConflictError`
  - [ ] Define: `IngestionError`, `ParserError`, `NormalizationError`
  - [ ] Define: `ApprovalGateError`, `LicenseViolationError`
  - [ ] Define: `MatchingError`, `SemanticIndexError`
  - [ ] Integrate with centralized `greenlang/exceptions/` hierarchy
  - [ ] Replace bare `ValueError`/`RuntimeError` with domain exceptions

---

## PHASE 2: SOURCE PARSERS & DATA INGESTION (Weeks 3-8)
> Goal: Go from 327 factors to 25,000+ via public source parsers

### 2.1 EPA GHG Hub Parser
- [ ] **TASK-F010**: Build EPA GHG Emissions Factors Hub parser
  - [ ] Create `greenlang/factors/ingestion/parsers/epa_ghg_hub.py`
  - [ ] Download & parse EPA Hub JSON/Excel (annual release)
  - [ ] Parse Scope 1 stationary combustion factors (all fuels)
  - [ ] Parse Scope 1 mobile combustion factors (on-road, non-road, rail, aviation, marine)
  - [ ] Parse Scope 2 electricity factors (eGRID subregion-level)
  - [ ] Parse Scope 3 upstream factors where available
  - [ ] Map EPA units -> GreenLang ontology (MMBtu, scf, short tons, gallons)
  - [ ] Set GWP basis (EPA uses AR5 for current reporting)
  - [ ] Generate factor_ids: `EF:EPA:{fuel}:{geography}:{year}:v{N}`
  - [ ] Set license: `public_us_government`, `redistribution_allowed: true`
  - [ ] Set DQS: temporal=5, geographical=4-5, technological=4, representativeness=4, methodological=5
  - [ ] Expected output: ~2,000-5,000 factors
  - [ ] Tests: `tests/factors/test_parser_epa_ghg_hub.py`

### 2.2 eGRID Parser
- [ ] **TASK-F011**: Build EPA eGRID electricity grid parser
  - [ ] Create `greenlang/factors/ingestion/parsers/egrid.py`
  - [ ] Parse eGRID annual dataset (Excel with multiple sheets)
  - [ ] Extract subregion-level grid emission factors (26 subregions)
  - [ ] Extract state-level averages
  - [ ] Extract plant-level factors (for Scope 2 market-based where applicable)
  - [ ] Map: lb CO2/MWh -> kg CO2e/kWh conversion
  - [ ] Include: CO2, CH4, N2O (eGRID reports all three)
  - [ ] Include: generation mix metadata (% coal, gas, nuclear, renewable)
  - [ ] Set license: `public_us_government`
  - [ ] Expected output: ~500-1,000 factors (subregion x fuel x year)
  - [ ] Tests: `tests/factors/test_parser_egrid.py`

### 2.3 DESNZ/DEFRA UK Conversion Factors Parser
- [ ] **TASK-F012**: Expand DEFRA parser for full UK conversion factors
  - [ ] Extend `greenlang/factors/etl/normalize.py` or create `parsers/desnz_uk.py`
  - [ ] Parse full DESNZ spreadsheet (not just Scope 1 fuels):
    - Scope 1: All fuel types (natural gas, LPG, coal, fuel oil, biomass, biogas, etc.)
    - Scope 2: UK grid electricity, UK heat/steam
    - Scope 3: WTT factors, freight, business travel, hotel stays, water, waste
  - [ ] Parse: activity-based factors (km, tonne-km, passenger-km, kWh)
  - [ ] Map DEFRA units -> GreenLang ontology
  - [ ] Set GWP: DEFRA uses AR5 (confirm for current year)
  - [ ] Set license: `uk_open_government`, `redistribution_allowed: true`
  - [ ] Expected output: ~3,000-5,000 factors
  - [ ] Tests: `tests/factors/test_parser_desnz_full.py`

### 2.4 IPCC Default Emission Factors
- [ ] **TASK-F013**: Build IPCC Tier 1 default emission factors parser
  - [ ] Create `greenlang/factors/ingestion/parsers/ipcc_defaults.py`
  - [ ] Source: IPCC 2006 Guidelines + 2019 Refinement (Tables in Annex)
  - [ ] Parse default emission factors by:
    - Energy sector (stationary + mobile combustion)
    - Industrial processes (cement, lime, glass, steel, aluminum, chemicals)
    - Agriculture (enteric fermentation, manure management, rice cultivation, soils)
    - LULUCF (land use change, forest management)
    - Waste (solid waste disposal, wastewater, incineration)
  - [ ] Set DQS: temporal=3 (guideline defaults), geographical=2 (global), technological=3, representativeness=3, methodological=5
  - [ ] Set license: `ipcc_guideline`, `redistribution_allowed: true`, `attribution_required: true`
  - [ ] Expected output: ~5,000-10,000 factors (IPCC has extensive tables)
  - [ ] Tests: `tests/factors/test_parser_ipcc.py`

### 2.5 EU CBAM Full Coverage
- [ ] **TASK-F014**: Expand CBAM parser for all covered products
  - [ ] Extend `greenlang/factors/etl/normalize.py` CBAM section
  - [ ] Parse CBAM default values for:
    - Iron & steel (hot-rolled, cold-rolled, stainless, alloy, scrap)
    - Aluminum (primary, secondary, alloys, wrought, cast)
    - Cement (clinker, Portland, composite)
    - Fertilizers (urea, ammonia, nitric acid, mixed fertilizers)
    - Electricity (country-specific grid factors for EU import origins)
    - Hydrogen (grey, blue, green by production method)
  - [ ] Parse per-installation data where published (EU delegated acts)
  - [ ] Include direct + indirect emissions separately
  - [ ] Tag: `regulatory_tags: ["CBAM_2026"]`, `compliance_frameworks: ["EU_CBAM"]`
  - [ ] Expected output: ~2,000-4,000 factors
  - [ ] Tests: `tests/factors/test_parser_cbam_full.py`

### 2.6 GHG Protocol Cross-Sector Tools
- [ ] **TASK-F015**: Parse GHG Protocol standard factors
  - [ ] Create `greenlang/factors/ingestion/parsers/ghg_protocol.py`
  - [ ] Source: GHG Protocol Scope 3 calculation guidance documents
  - [ ] Parse: Category-specific default emission factors
    - Cat 1: Purchased goods/services (EEIO factors by sector)
    - Cat 4-9: Transportation factors (tonne-km, passenger-km)
    - Cat 6: Business travel (flight distance bands, hotel stays)
    - Cat 7: Employee commuting (mode-specific)
    - Cat 13: Downstream leased assets
  - [ ] Set license: `ghg_protocol_reference`, `attribution_required: true`
  - [ ] Expected output: ~1,000-3,000 factors
  - [ ] Tests: `tests/factors/test_parser_ghg_protocol.py`

### 2.7 TCR/Climate Registry Default Factors
- [ ] **TASK-F016**: Build The Climate Registry (TCR) default factors parser
  - [ ] Create `greenlang/factors/ingestion/parsers/tcr.py`
  - [ ] Parse TCR General Reporting Protocol default tables
  - [ ] Parse US-specific GHG reporting defaults
  - [ ] Set license per source_registry.yaml rules
  - [ ] Expected output: ~500-1,000 factors
  - [ ] Tests: `tests/factors/test_parser_tcr.py`

### 2.8 Green-e Residual Mix Factors
- [ ] **TASK-F017**: Build Green-e residual mix factors parser
  - [ ] Create `greenlang/factors/ingestion/parsers/green_e.py`
  - [ ] Parse annual residual mix emission rates (US + Canada)
  - [ ] Scope 2 market-based methodology factors
  - [ ] Published annually (spring)
  - [ ] Set license per source_registry.yaml rules
  - [ ] Expected output: ~100-500 factors (state-level or grid-zone-level)
  - [ ] Tests: `tests/factors/test_parser_green_e.py`

### 2.9 Parser Framework Enhancements
- [ ] **TASK-F018**: Build parser plugin system
  - [ ] Create `greenlang/factors/ingestion/parsers/__init__.py` with parser registry
  - [ ] Define `BaseSourceParser` abstract class:
    ```python
    class BaseSourceParser(ABC):
        source_id: str
        parser_id: str
        supported_formats: List[str]  # json, csv, xlsx, xml

        @abstractmethod
        def parse(self, raw_bytes: bytes, metadata: Dict) -> List[Dict]:
            """Parse raw source data to intermediate dicts."""

        @abstractmethod
        def validate_schema(self, raw_bytes: bytes) -> Tuple[bool, List[str]]:
            """Validate source file matches expected schema."""
    ```
  - [ ] Implement `ParserRegistry` for dynamic parser lookup by source_id
  - [ ] Add parser versioning (e.g., `epa_parser_v2`, `epa_parser_v3`)
  - [ ] Add parser config (column mappings, unit conversions, etc.) via YAML

### 2.10 Bulk Ingestion Pipeline
- [ ] **TASK-F019**: Build automated bulk ingestion workflow
  - [ ] Create `greenlang/factors/ingestion/bulk_ingest.py`
  - [ ] Accept list of (source_id, file_path) pairs
  - [ ] Route each to correct parser via ParserRegistry
  - [ ] Run Q1-Q2 validation gates on each parsed batch
  - [ ] Aggregate into pending edition with manifest
  - [ ] Generate draft changelog
  - [ ] Report: total ingested, per-source counts, errors, warnings
  - [ ] CLI: `gl factors bulk-ingest --config ingest_config.yaml --edition-id 2026.04.0`

---

## PHASE 3: DATA QUALITY & GOVERNANCE AT SCALE (Weeks 6-10)
> Goal: Ensure 40-60K factors pass QA gates and earn "certified" status

### 3.1 Automated QA Pipeline
- [ ] **TASK-F020**: Build batch QA runner
  - [ ] Create `greenlang/factors/quality/batch_qa.py`
  - [ ] Run Q1-Q6 gates on entire edition (not individual factors)
  - [ ] Parallel validation (thread pool for CPU-bound checks)
  - [ ] Output: QA report JSON with per-factor results
  - [ ] Flag: factors that fail any gate stay in `preview`
  - [ ] Auto-promote: factors that pass all gates to `certified` (if methodology_signed)
  - [ ] CLI: `gl factors qa-batch --edition-id 2026.04.0 --report qa_report.json`

- [ ] **TASK-F021**: Enhanced duplicate detection
  - [ ] Create `greenlang/factors/quality/dedup_engine.py`
  - [ ] Implement `duplicate_fingerprint()` across entire edition
  - [ ] Detect: exact duplicates (same fingerprint), near-duplicates (same fuel+geo+scope, different values)
  - [ ] Resolution strategy per `dedupe_rules.py`: SOURCE_PRIORITY, geography specificity, temporal recency
  - [ ] Report: duplicate pairs with resolution recommendation
  - [ ] Auto-resolve: merge lower-priority source into higher (keep higher DQS)
  - [ ] Human review: flag ambiguous cases for methodology lead

- [ ] **TASK-F022**: Cross-source consistency checks
  - [ ] Create `greenlang/factors/quality/cross_source.py`
  - [ ] Compare factors from different sources for same activity:
    - EPA diesel Scope 1 vs DEFRA diesel Scope 1 vs IPCC diesel default
    - Flag if discrepancy > 20% (log warning)
    - Flag if discrepancy > 50% (require human review)
  - [ ] Output: consistency matrix showing agreement/disagreement across sources
  - [ ] Purpose: identify data quality issues before certification

### 3.2 Methodology Review Workflow
- [ ] **TASK-F023**: Build methodology review UI data layer
  - [ ] Create `greenlang/factors/quality/review_workflow.py`
  - [ ] Implement review assignment (reviewer, due_date, priority)
  - [ ] Implement checklist tracking (10-point checklist per methodology_review_checklist.md)
  - [ ] Implement review decision recording (approved/rejected/needs_revision)
  - [ ] Implement batch review (approve all factors from same source+parser run)
  - [ ] API: `POST /api/v1/factors/reviews` (create), `PATCH /api/v1/factors/reviews/{id}` (update)
  - [ ] API: `GET /api/v1/factors/reviews?status=pending&reviewer=me` (list queue)

- [ ] **TASK-F024**: Build release signoff workflow
  - [ ] Extend `greenlang/factors/quality/release_signoff.py` from 3 items to full checklist:
    - [ ] All Q1-Q6 gates pass for every factor in edition
    - [ ] No unresolved duplicate pairs
    - [ ] Cross-source consistency report reviewed
    - [ ] Changelog draft reviewed and approved
    - [ ] Methodology lead signed off
    - [ ] Legal confirmed license_class for all new sources
    - [ ] Regression test passed (compare_editions shows expected changes)
    - [ ] Load test passed (p95 < 500ms with new data volume)
    - [ ] Gold eval passed (precision@1 >= 0.85)
  - [ ] CLI: `gl factors release-signoff --edition-id 2026.04.0 --approver alice@greenlang.io`

### 3.3 License Compliance Automation
- [ ] **TASK-F025**: Build license compliance scanner
  - [ ] Create `greenlang/factors/quality/license_scanner.py`
  - [ ] Scan all factors in edition for license violations:
    - `connector_only` source but `redistribution_allowed: true` -> ERROR
    - Missing `citation_text` for `attribution_required` sources -> WARNING
    - `factor_status: certified` but source `approval_required_for_certified` and no legal signoff -> BLOCK
  - [ ] Generate license compliance report
  - [ ] Integrate with release signoff (must pass before stable promotion)
  - [ ] CLI: `gl factors license-scan --edition-id 2026.04.0`

---

## PHASE 4: API & SDK COMPLETION (Weeks 8-12)
> Goal: Production-ready API with tier enforcement and SDK for customers

### 4.1 API Enhancements
- [ ] **TASK-F030**: Implement tier-based factor visibility enforcement
  - [ ] Read tier from JWT token or API key metadata
  - [ ] Community: filter out preview + connector_only factors
  - [ ] Pro: allow preview (if `include_preview=true`), filter connector_only
  - [ ] Enterprise: allow all (if `include_connector=true`)
  - [ ] Add middleware for entitlement checking
  - [ ] Return 403 with clear message if tier insufficient for requested visibility

- [ ] **TASK-F031**: Implement Postgres repository
  - [ ] Create `greenlang/factors/catalog_repository_pg.py`
  - [ ] Implement `PostgresFactorCatalogRepository(FactorCatalogRepository)`
  - [ ] Use async `psycopg` + `psycopg_pool` (per project patterns)
  - [ ] All 10 abstract methods: get_default_edition, list_editions, get_factor, search, facets, etc.
  - [ ] Full-text search via `tsvector` + `ts_query` (replace search_blob LIKE)
  - [ ] JSONB operators for payload queries
  - [ ] Connection pooling (min=2, max=20, per env)
  - [ ] Tests: `tests/factors/test_catalog_repository_pg.py` (with testcontainers or mock)
  - [ ] Migration V426-V428 compatibility verified

- [ ] **TASK-F032**: Add audit bundle export endpoint
  - [ ] `GET /api/v1/factors/{factor_id}/audit-bundle`
  - [ ] Returns: raw_artifact_uri, parser_log, normalized record, QA gates, reviewer decision
  - [ ] Enterprise tier only
  - [ ] Include SHA-256 verification chain
  - [ ] Tests: `tests/factors/test_audit_bundle_api.py`

- [ ] **TASK-F033**: Add bulk export endpoint
  - [ ] `GET /api/v1/factors/export?format=json&status=certified`
  - [ ] Streaming JSON Lines response for large datasets
  - [ ] Filter by: status, geography, fuel_type, scope, source_id
  - [ ] Enforce license: exclude `connector_only` unless enterprise tier
  - [ ] Include: `X-Factors-Edition` header and manifest hash
  - [ ] Rate limit: 1 export per 15 minutes per API key

- [ ] **TASK-F034**: Add factor diff endpoint
  - [ ] `GET /api/v1/factors/{factor_id}/diff?left_edition=X&right_edition=Y`
  - [ ] Shows field-by-field changes between editions
  - [ ] Useful for consultants defending factor choice changes

- [ ] **TASK-F035**: Add factor search v2 with advanced filters
  - [ ] `POST /api/v1/factors/search/v2`
  - [ ] Add filters: `source_id`, `license_class`, `dqs_min`, `valid_on_date`, `sector_tags`, `activity_tags`
  - [ ] Add sort: `relevance`, `dqs_score`, `co2e_total`, `source_year`
  - [ ] Add pagination: `offset` + `limit` with total count
  - [ ] Add response fields: `highlights` (matched tokens), `explanation`

- [ ] **TASK-F036**: Implement ETag caching
  - [ ] `GET /api/v1/factors/{id}` returns `ETag: sha256:{content_hash}`
  - [ ] Client sends `If-None-Match: sha256:{hash}` -> 304 Not Modified
  - [ ] Reduces bandwidth for repeat queries
  - [ ] Cache-Control: `max-age=3600` for certified, `max-age=600` for preview

### 4.2 SDK Generation
- [ ] **TASK-F037**: Generate Python SDK from OpenAPI spec
  - [ ] Export OpenAPI 3.1 spec from FastAPI
  - [ ] Generate typed Python client using `openapi-python-client` or `datamodel-code-generator`
  - [ ] SDK methods: `list_editions()`, `get_factor()`, `search()`, `match()`, `calculate()`, `export()`
  - [ ] Auto-set headers: `X-Factors-Edition`, `Authorization: Bearer ...`
  - [ ] Retry logic (exponential backoff on 429/503)
  - [ ] Replace stub `sdk/__init__.py` with generated client
  - [ ] Publish to PyPI as `greenlang-factors-sdk`
  - [ ] Tests: `tests/factors/test_sdk_integration.py`

- [ ] **TASK-F038**: Build JavaScript/TypeScript SDK
  - [ ] Generate from same OpenAPI spec
  - [ ] Publish to npm as `@greenlang/factors-sdk`
  - [ ] Typed responses (TypeScript)
  - [ ] Support: Node.js + browser (fetch-based)

### 4.3 API Documentation
- [ ] **TASK-F039**: Build developer documentation portal
  - [ ] Quickstart guide (5-minute integration)
  - [ ] Authentication guide (API keys vs JWT)
  - [ ] Edition pinning guide (reproducibility)
  - [ ] Factor search cookbook (common queries)
  - [ ] Match API guide (activity description -> factor)
  - [ ] Audit bundle guide (for external auditors)
  - [ ] Rate limiting reference
  - [ ] Error code reference
  - [ ] Changelog format reference

---

## PHASE 5: SEMANTIC MATCHING & AI (Weeks 10-14)
> Goal: Upgrade matching from lexical to hybrid (lexical + semantic + LLM rerank)

### 5.1 Embedding Infrastructure
- [ ] **TASK-F040**: Implement pgvector semantic index
  - [ ] Replace `NoopSemanticIndex` with `PgVectorSemanticIndex`
  - [ ] Create `greenlang/factors/matching/pgvector_index.py`
  - [ ] Schema: `factor_embeddings(edition_id, factor_id, embedding vector(384), search_text)`
  - [ ] Migration V429: Create `factors_catalog.factor_embeddings` table with HNSW index
  - [ ] Embedding model: MiniLM-L6-v2 (384d) for v1, MPNet (768d) for v2
  - [ ] Batch embed all factors on edition publish
  - [ ] Search: `SELECT factor_id FROM factor_embeddings ORDER BY embedding <=> $1 LIMIT 50`
  - [ ] Tests: `tests/factors/test_pgvector_matching.py`

- [ ] **TASK-F041**: Build embedding pipeline
  - [ ] Create `greenlang/factors/matching/embedding.py`
  - [ ] Construct search text for each factor: `{fuel_type} {geography} {scope_description} {boundary} {tags} {notes}`
  - [ ] Batch embedding (process 1000 factors at a time)
  - [ ] Cache embeddings (regenerate only when content_hash changes)
  - [ ] CLI: `gl factors embed-edition --edition-id 2026.04.0`

### 5.2 Hybrid Matching Pipeline
- [ ] **TASK-F042**: Implement RRF hybrid search
  - [ ] Extend `greenlang/factors/matching/pipeline.py`
  - [ ] Stage 1: Facet filter (existing)
  - [ ] Stage 2a: Lexical search (existing token overlap)
  - [ ] Stage 2b: Semantic search (pgvector cosine similarity)
  - [ ] Stage 3: Reciprocal Rank Fusion (RRF) combining lexical + semantic rankings
  - [ ] Stage 4: DQS boost (existing)
  - [ ] Configurable: weights for lexical vs semantic (default 0.4:0.6)
  - [ ] Fallback: if pgvector unavailable, use lexical-only (graceful degradation)

- [ ] **TASK-F043**: Implement LLM-assisted reranking (optional, enterprise)
  - [ ] Create `greenlang/factors/matching/llm_rerank.py`
  - [ ] Input: top-20 candidates from hybrid search + user activity description
  - [ ] Prompt: "Given activity '{desc}', rank these emission factors by relevance..."
  - [ ] Output: reranked list with explanations
  - [ ] Guard: LLM reranking never changes factor VALUES, only ranking
  - [ ] Flag: `X-Factors-Reranked: llm` header when LLM used
  - [ ] Rate limit: LLM reranking max 10 req/min per API key
  - [ ] Tests: mock LLM responses, verify ranking change, verify no value mutation

### 5.3 Matching Quality
- [ ] **TASK-F044**: Build comprehensive evaluation suite
  - [ ] Expand gold eval to 500+ cases (from 100+ in Phase 1)
  - [ ] Add: precision@1, precision@5, recall@5, MRR, NDCG
  - [ ] Add: per-domain breakdown (energy, transport, industry, agriculture)
  - [ ] Add: A/B test framework (lexical vs hybrid vs LLM-reranked)
  - [ ] CI gate: precision@1 >= 0.90 for lexical, >= 0.95 for hybrid
  - [ ] Generate: monthly eval report for methodology team

- [ ] **TASK-F045**: Build factor suggestion agent
  - [ ] Create `greenlang/factors/matching/suggestion_agent.py`
  - [ ] Input: user's GHG inventory data (fuel type, location, activity)
  - [ ] Output: recommended factor with confidence score + alternatives
  - [ ] Logic: Match -> verify scope/boundary alignment -> suggest with explanation
  - [ ] Include: "Did you mean X?" for common mismatches
  - [ ] API: `POST /api/v1/factors/suggest` (enterprise tier)

---

## PHASE 6: WATCH & UPDATE AUTOMATION (Weeks 10-14)
> Goal: Automated source monitoring with event-driven releases

### 6.1 Source Watch Scheduler
- [ ] **TASK-F050**: Build automated source watch cron
  - [ ] Create `greenlang/factors/watch/scheduler.py`
  - [ ] Daily: HTTP HEAD/GET checks for all registry sources
  - [ ] Store results in `watch_results` table (new migration V430)
  - [ ] Detect: file hash changes, 404s, new publications
  - [ ] Notify: Slack/email on detected changes
  - [ ] CLI: `gl factors watch-run` (manual trigger)
  - [ ] Docker: cron container running `watch-run` daily at 06:00 UTC

- [ ] **TASK-F051**: Build change detection pipeline
  - [ ] Create `greenlang/factors/watch/change_detector.py`
  - [ ] On source change detected:
    1. Download new artifact
    2. Store in artifact store with SHA-256
    3. Run parser
    4. Compare parsed factors against current edition
    5. Classify changes (numeric/policy/parser-break)
    6. Generate change report
  - [ ] Automatic: Create pending_edition.json for numeric changes
  - [ ] Manual: Flag policy changes and parser breaks for human review
  - [ ] Notify: methodology lead for policy changes

- [ ] **TASK-F052**: Build changelog automation
  - [ ] Extend `greenlang/factors/watch/changelog_draft.py`
  - [ ] Input: change_report from change_detector
  - [ ] Output: Markdown changelog with:
    - Source changes (which sources updated)
    - Factor changes (added/removed/modified counts)
    - Numeric corrections (factor_id, old_value -> new_value)
    - Policy changes (methodology/boundary reclassifications)
    - Deprecations (factors deprecated with replacement_factor_id)
  - [ ] Draft goes to release_manager for review
  - [ ] Human edits -> approved changelog stored in edition metadata

### 6.2 Release Automation
- [ ] **TASK-F053**: Build release orchestrator
  - [ ] Create `greenlang/factors/watch/release_orchestrator.py`
  - [ ] Monthly release workflow:
    1. Collect all pending changes since last stable
    2. Run batch QA (Q1-Q6) on pending factors
    3. Run duplicate detection
    4. Run cross-source consistency check
    5. Run license compliance scan
    6. Generate draft changelog
    7. Generate release signoff checklist
    8. Notify release_manager + methodology_lead
  - [ ] After human approval:
    9. Promote edition to stable
    10. Tag with semver
    11. Publish changelog
    12. Update default edition
  - [ ] CLI: `gl factors release-prepare --edition-id 2026.05.0`
  - [ ] CLI: `gl factors release-publish --edition-id 2026.05.0 --approved-by alice`

---

## PHASE 7: CONNECTOR SOURCES & LICENSING (Weeks 12-16)
> Goal: Add 20-40K connector-backed factors (licensed, enterprise only)

### 7.1 Connector Framework
- [ ] **TASK-F060**: Build connector source framework
  - [ ] Create `greenlang/factors/connectors/` package
  - [ ] Define `BaseConnector` abstract class:
    ```python
    class BaseConnector(ABC):
        source_id: str
        requires_license: bool = True
        license_class: str = "commercial_connector"

        @abstractmethod
        async def fetch_metadata(self) -> List[Dict]:
            """Fetch factor metadata (IDs, descriptions, not values)."""

        @abstractmethod
        async def fetch_values(self, factor_ids: List[str], license_key: str) -> List[Dict]:
            """Fetch factor values (requires valid license)."""
    ```
  - [ ] Implement connector registry
  - [ ] License key management (per-tenant, encrypted at rest)
  - [ ] Audit logging for all connector calls

### 7.2 IEA Connector
- [ ] **TASK-F061**: Build IEA statistics connector
  - [ ] Create `greenlang/factors/connectors/iea.py`
  - [ ] API: IEA Data Services (subscription required)
  - [ ] Factors: Country-level CO2 emission factors for electricity
  - [ ] Factors: Fuel-specific emission factors by country
  - [ ] Coverage: 150+ countries
  - [ ] Set: `factor_status: connector_only`, `license_class: commercial_connector`
  - [ ] Expected: ~5,000-10,000 factors

### 7.3 Ecoinvent Connector
- [ ] **TASK-F062**: Build ecoinvent connector
  - [ ] Create `greenlang/factors/connectors/ecoinvent.py`
  - [ ] API: ecoinvent database (license required)
  - [ ] Factors: LCA-based emission factors for products and processes
  - [ ] Coverage: 18,000+ processes globally
  - [ ] Set: `factor_status: connector_only`, `license_class: commercial_connector`
  - [ ] Expected: ~10,000-20,000 factors (subset most relevant to GHG reporting)

### 7.4 Electricity Maps Connector
- [ ] **TASK-F063**: Build Electricity Maps connector
  - [ ] Create `greenlang/factors/connectors/electricity_maps.py`
  - [ ] API: Electricity Maps (real-time + historical)
  - [ ] Factors: Real-time grid carbon intensity by zone
  - [ ] Coverage: 200+ electricity zones globally
  - [ ] Set: `factor_status: connector_only`, `cadence: daily`
  - [ ] Unique: real-time + forecasted factors (marginal vs average)
  - [ ] Expected: ~2,000-5,000 factors (zone x timeframe)

### 7.5 Tenant Overlay Implementation
- [ ] **TASK-F064**: Build enterprise tenant overlay system
  - [ ] Extend `greenlang/factors/tenant_overlay.py` from stub to production
  - [ ] Support: customer-supplied factors (internal energy audits, supplier-specific)
  - [ ] Storage: encrypted SQLite per tenant (AES-256-GCM)
  - [ ] Merge logic: tenant overlay > catalog default (for same activity+geography)
  - [ ] Isolation: tenant factors never visible to other tenants
  - [ ] API: `POST /api/v1/factors/overlays` (upload), `GET /api/v1/factors?tenant_overlay=true` (query)
  - [ ] Tests: multi-tenant isolation tests

---

## PHASE 8: OBSERVABILITY & OPERATIONS (Weeks 14-18)
> Goal: Production monitoring, alerting, and operational tooling

### 8.1 Prometheus Metrics
- [ ] **TASK-F070**: Export factors-specific Prometheus metrics
  - [ ] `greenlang_factors_api_requests_total{path, method, status, tier}` (counter)
  - [ ] `greenlang_factors_api_latency_seconds{path}` (histogram, p50/p95/p99)
  - [ ] `greenlang_factors_search_results_count{edition}` (histogram)
  - [ ] `greenlang_factors_match_score_top1{edition}` (histogram)
  - [ ] `greenlang_factors_edition_factor_count{edition, status}` (gauge)
  - [ ] `greenlang_factors_ingestion_rows_total{source_id, status}` (counter)
  - [ ] `greenlang_factors_watch_source_changes{source_id}` (counter)
  - [ ] `greenlang_factors_qa_gate_failures{gate, edition}` (counter)
  - [ ] Integrate with existing OBS-001 Prometheus stack

### 8.2 Grafana Dashboards
- [ ] **TASK-F071**: Build Factors operational dashboard
  - [ ] Panel: API request rate by endpoint
  - [ ] Panel: p95 latency by endpoint (target: <500ms)
  - [ ] Panel: Error rate by endpoint (target: <0.1%)
  - [ ] Panel: Factor count by status (certified/preview/connector/deprecated)
  - [ ] Panel: Source watch status (last check, changes detected)
  - [ ] Panel: Ingestion pipeline status (running/success/failed)
  - [ ] Panel: QA gate pass/fail rates
  - [ ] Panel: Edition comparison (current vs previous)
  - [ ] Integrate with existing OBS-002 Grafana stack

### 8.3 Alerting
- [ ] **TASK-F072**: Configure Factors-specific alerts
  - [ ] CRITICAL: API error rate > 1% for 5 minutes
  - [ ] CRITICAL: p95 latency > 2s for 5 minutes
  - [ ] WARNING: Source watch failure (404 or timeout) for any source
  - [ ] WARNING: Ingestion pipeline failure
  - [ ] INFO: New edition published
  - [ ] INFO: Factor count dropped > 5% between editions (possible data loss)
  - [ ] Integrate with existing OBS-004 alerting stack

### 8.4 Health Checks
- [ ] **TASK-F073**: Implement comprehensive health endpoint
  - [ ] `GET /api/v1/health` returns:
    - API status (up/degraded/down)
    - Database connectivity (SQLite/Postgres)
    - Redis cache connectivity
    - Default edition available
    - Factor count (current edition)
    - Last successful ingestion timestamp
    - Last successful source watch timestamp
  - [ ] K8s: readiness probe on `/api/v1/health`
  - [ ] K8s: liveness probe on `/api/v1/health/live` (lightweight)

---

## PHASE 9: DEPLOYMENT & INFRASTRUCTURE (Weeks 16-20)
> Goal: Production deployment on K8s with full CI/CD

### 9.1 Docker
- [ ] **TASK-F080**: Build Factors service Docker image
  - [ ] Create `deployment/docker/factors/Dockerfile`
  - [ ] Base: Python 3.11-slim
  - [ ] Install: `greenlang[server,security]` + factors dependencies
  - [ ] Embed: source_registry.yaml, built-in factors
  - [ ] Entrypoint: uvicorn with factors API
  - [ ] Health check: `curl -f http://localhost:8000/api/v1/health`
  - [ ] Multi-stage build (builder + runtime)
  - [ ] Size target: <500MB

- [ ] **TASK-F081**: Build Factors ingestion worker Docker image
  - [ ] Separate container for ingestion/watch jobs
  - [ ] Entrypoint: `gl factors watch-run` or `gl factors bulk-ingest`
  - [ ] Cron schedule: daily watch, weekly diff, monthly release prep
  - [ ] Shared: same SQLite/Postgres as API container

### 9.2 Kubernetes
- [ ] **TASK-F082**: Create K8s manifests for Factors service
  - [ ] Deployment: 2-3 replicas, resource limits (256Mi-1Gi, 0.5-2 CPU)
  - [ ] Service: ClusterIP + Ingress
  - [ ] ConfigMap: environment variables
  - [ ] Secret: API keys, JWT secret, database credentials
  - [ ] PVC: SQLite storage (if not using Postgres)
  - [ ] CronJob: daily source watch, weekly diff batch
  - [ ] HPA: scale on CPU > 70%
  - [ ] Integrate with existing INFRA-001 K8s stack

### 9.3 Database Migration
- [ ] **TASK-F083**: Run Flyway migrations for Factors
  - [ ] V426-V428 in Postgres (hosted environment)
  - [ ] V429: factor_embeddings (if pgvector enabled)
  - [ ] V430: watch_results table
  - [ ] Validate: all tables created, indexes present, FK constraints active
  - [ ] Seed: initial builtin edition + factors
  - [ ] Integrate with existing Flyway CI/CD pipeline

---

## PHASE 10: SCALE TO 100K (Weeks 18-24)
> Goal: Hit the 100K factor catalog target with governance

### 10.1 Factor Volume Targets
- [ ] **TASK-F090**: Execute ingestion plan to reach 100K
  - [ ] EPA GHG Hub: ~3,000 factors (certified)
  - [ ] eGRID: ~500 factors (certified)
  - [ ] DESNZ/DEFRA full: ~4,000 factors (certified)
  - [ ] IPCC defaults: ~8,000 factors (certified)
  - [ ] CBAM full: ~3,000 factors (certified)
  - [ ] GHG Protocol: ~2,000 factors (certified)
  - [ ] TCR: ~500 factors (certified)
  - [ ] Green-e: ~200 factors (certified)
  - [ ] GreenLang built-in: ~327 factors (certified)
  - [ ] Subtotal certified: ~21,500
  - [ ] Preview factors (pending deeper QA): ~20,000-30,000
  - [ ] IEA connector: ~10,000 (connector-only)
  - [ ] Ecoinvent connector: ~15,000 (connector-only)
  - [ ] Electricity Maps: ~3,000 (connector-only)
  - [ ] Other connectors: ~5,000 (connector-only)
  - [ ] **Total: ~75,000-100,000+ factors**

### 10.2 Performance Optimization
- [ ] **TASK-F091**: Optimize for 100K+ factor queries
  - [ ] Benchmark: list/search/match/facets at 100K scale
  - [ ] Target: p95 < 500ms for search, p95 < 100ms for get_factor
  - [ ] Optimize: Postgres indexes (GIN on payload_json, GiST on embeddings)
  - [ ] Optimize: Redis caching for hot factors (top 1000 by access count)
  - [ ] Optimize: Batch embedding generation (GPU if available)
  - [ ] Load test: `scripts/factors_load_smoke.py --rounds 30` (simulate 100K)

- [ ] **TASK-F092**: Implement query result caching
  - [ ] Cache: search results by (query_hash, edition_id, tier) -> JSON
  - [ ] TTL: 1h for certified, 10m for preview
  - [ ] Invalidation: on edition publish or hotfix
  - [ ] Redis backend (existing INFRA-003)

### 10.3 Regulatory Tagging at Scale
- [ ] **TASK-F093**: Build regulatory compliance tagger
  - [ ] Create `greenlang/factors/compliance/regulatory_tagger.py`
  - [ ] Auto-tag factors with applicable regulations:
    - CBAM: Iron/steel/aluminum/cement/fertilizer/electricity imports to EU
    - SB 253: All US Scope 1-2 factors
    - CSRD: All EU factors
    - ISO 14064: All factors (universal)
    - GHG Protocol: All factors with methodology mapping
  - [ ] Store in `compliance_frameworks` field of EmissionFactorRecord
  - [ ] Searchable: `GET /api/v1/factors?regulation=CBAM`

---

## PHASE 11: DESIGN PARTNER PILOTS (Weeks 20-26)
> Goal: Onboard 3-6 design partners, collect feedback, iterate

### 11.1 Pilot Onboarding
- [ ] **TASK-F100**: Execute DP1 onboarding for 3 design partners
  - [ ] Provision API keys (pro or enterprise tier)
  - [ ] Configure edition pinning
  - [ ] Deliver audit export sample
  - [ ] NDA and data-rights addendum (if connector sources needed)
  - [ ] Onboarding call: walkthrough of API, SDK, docs

### 11.2 Pilot Instrumentation
- [ ] **TASK-F101**: Build pilot telemetry dashboard
  - [ ] Track: time-to-first-correct-factor (per partner)
  - [ ] Track: mismatch rate vs gold eval (per partner)
  - [ ] Track: citation completeness score (per partner)
  - [ ] Track: most-queried factor IDs (product intelligence)
  - [ ] Track: search queries that returned 0 results (gap analysis)

### 11.3 Feedback Integration
- [ ] **TASK-F102**: Build pilot feedback pipeline
  - [ ] Collect: mismatch reports as qa_reviews rows
  - [ ] Collect: missing factor requests
  - [ ] Collect: methodology questions/disputes
  - [ ] Monthly sync: review feedback with methodology team
  - [ ] Update gold eval set with partner mismatches
  - [ ] Prioritize: new parsers/sources based on partner demand

---

## PHASE 12: GA LAUNCH PREPARATION (Weeks 24-30)
> Goal: General availability with SLA, billing, and support

### 12.1 GA Checklist
- [ ] **TASK-F110**: Complete GA readiness checklist
  - [ ] 100K+ factors in catalog (with status labels)
  - [ ] 40K+ certified factors (default API)
  - [ ] All public source parsers operational
  - [ ] Connector sources: IEA + ecoinvent operational
  - [ ] Semantic matching deployed (pgvector + embeddings)
  - [ ] Gold eval: precision@1 >= 0.90
  - [ ] API: p95 < 500ms at 100K scale
  - [ ] API: rate limiting per tier
  - [ ] Authentication: JWT + API key
  - [ ] Billing: usage metering operational
  - [ ] Monitoring: Prometheus + Grafana dashboards
  - [ ] Alerting: SEV1-SEV4 routing configured
  - [ ] Runbook: updated with all operational procedures
  - [ ] SDK: Python + TypeScript published
  - [ ] Docs: developer portal complete
  - [ ] Support: severity matrix enforced
  - [ ] Legal: all source licenses reviewed and documented
  - [ ] Load test: 1000 req/sec verified

### 12.2 Billing Integration
- [ ] **TASK-F111**: Build billing system integration
  - [ ] Connect api_usage_events to billing provider (Stripe/internal)
  - [ ] Community: free (rate limited)
  - [ ] Pro: $99/month + $0.025/call above 1000
  - [ ] Enterprise: custom pricing
  - [ ] Usage dashboard for customers
  - [ ] Overage alerts

### 12.3 SLA Enforcement
- [ ] **TASK-F112**: Implement SLA monitoring
  - [ ] Track: API uptime (target 99.9%)
  - [ ] Track: hotfix response time (target 1h for SEV1)
  - [ ] Track: monthly release delivery (target 1st of each month)
  - [ ] Report: monthly SLA report for enterprise customers
  - [ ] Escalation: automated PagerDuty/Slack for SLA breaches

---

## SUMMARY: TASK COUNT BY PHASE

| Phase | Scope | Tasks | Weeks |
|-------|-------|-------|-------|
| 1. Foundation Hardening | Tests, CI/CD, logging, exceptions | F001-F007 (7) | 1-3 |
| 2. Source Parsers | EPA, eGRID, DESNZ, IPCC, CBAM, GHG Protocol, TCR, Green-e | F010-F019 (10) | 3-8 |
| 3. Data Quality | Batch QA, dedup, cross-source, methodology review, license scan | F020-F025 (6) | 6-10 |
| 4. API & SDK | Tier enforcement, Postgres, audit bundle, bulk export, SDK | F030-F039 (10) | 8-12 |
| 5. Semantic Matching | pgvector, embeddings, hybrid search, LLM rerank, eval | F040-F045 (6) | 10-14 |
| 6. Watch Automation | Scheduler, change detection, changelog, release orchestrator | F050-F053 (4) | 10-14 |
| 7. Connectors | IEA, ecoinvent, Electricity Maps, tenant overlay | F060-F064 (5) | 12-16 |
| 8. Observability | Prometheus, Grafana, alerting, health checks | F070-F073 (4) | 14-18 |
| 9. Deployment | Docker, K8s, Flyway | F080-F083 (4) | 16-20 |
| 10. Scale to 100K | Volume ingestion, perf optimization, regulatory tagging | F090-F093 (4) | 18-24 |
| 11. Design Partners | Onboarding, telemetry, feedback loop | F100-F102 (3) | 20-26 |
| 12. GA Launch | Checklist, billing, SLA | F110-F112 (3) | 24-30 |
| **TOTAL** | | **66 tasks** | **30 weeks** |

---

## CRITICAL PATH

```
Week 1-3:   Tests + CI/CD (Foundation)
             |
Week 3-8:   Source Parsers (EPA, eGRID, DESNZ, IPCC, CBAM)
             |
Week 6-10:  QA Pipeline (batch QA, dedup, methodology review)
             |
Week 8-12:  API Completion (Postgres, tier enforcement, SDK)
             |                            |
Week 10-14: Semantic Matching        Watch Automation
             |                            |
Week 12-16: Connector Sources (IEA, ecoinvent)
             |
Week 14-18: Observability (Prometheus, Grafana)
             |
Week 16-20: K8s Deployment
             |
Week 18-24: Scale to 100K + Performance
             |
Week 20-26: Design Partner Pilots
             |
Week 24-30: GA Launch
```

---

## TEAM REQUIREMENTS (CTO Proposal Alignment)

| Role | Count | Responsibility |
|------|-------|---------------|
| **Backend Engineer (Senior)** | 2 | Parsers, repository, API, connectors |
| **Data Engineer** | 1 | ETL pipelines, source monitoring, ingestion |
| **ML Engineer** | 1 | Semantic matching, embeddings, pgvector |
| **Methodology Lead** | 1 | QA review, factor certification, regulatory compliance |
| **DevOps Engineer** | 0.5 | Docker, K8s, CI/CD, monitoring |
| **Product Manager** | 0.5 | Pilot coordination, roadmap, commercial model |
| **Technical Writer** | 0.5 | Developer docs, API reference, onboarding guides |

**Total: ~6 FTEs for 30 weeks (7.5 months)**

---

## RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Source format changes (EPA/DESNZ) | Medium | High | Parser versioning + automated detection |
| License dispute (IEA/ecoinvent) | Low | Critical | Legal review before connector launch |
| Factor quality issues at scale | Medium | High | Batch QA + cross-source consistency |
| pgvector performance at 100K | Low | Medium | HNSW tuning + query optimization |
| Design partner churn | Medium | Medium | Weekly syncs + rapid feedback loop |
| SB 253 deadline pressure | High | High | Prioritize US Scope 1-2 factors first |
| CBAM reporting changes | Medium | Medium | Weekly EU regulatory monitoring |

---

## FIRST 30/60/90 DAYS

### Days 1-30 (Weeks 1-4)
- Complete Phase 1 (test expansion, CI/CD, logging)
- Start Phase 2 (EPA + eGRID parsers)
- Ingest first 5,000+ factors
- First eval run with expanded gold set (target: precision@1 >= 0.80)

### Days 31-60 (Weeks 5-8)
- Complete Phase 2 (all public source parsers)
- Start Phase 3 (batch QA pipeline)
- Reach 20,000+ factors in catalog
- First methodology review cycle
- Start Phase 4 (Postgres repository, tier enforcement)

### Days 61-90 (Weeks 9-12)
- Complete Phase 3 (QA pipeline operational)
- Complete Phase 4 (API production-ready, SDK published)
- Start Phase 5 (semantic matching)
- Start Phase 6 (watch automation)
- Reach 30,000+ certified factors
- First design partner outreach
