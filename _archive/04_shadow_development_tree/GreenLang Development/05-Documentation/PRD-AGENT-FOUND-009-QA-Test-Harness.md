# PRD: AGENT-FOUND-009 - QA Test Harness

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-009 |
| **Agent ID** | GL-FOUND-X-009 |
| **Component** | Quality Gate & Test Harness (Test Framework, Snapshot Testing) |
| **Category** | Foundations Agent |
| **Priority** | P0 - Critical (quality assurance backbone for all agents) |
| **Status** | Layer 1 Complete (~1,763 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS runs 47+ agents producing emission calculations, compliance
assessments, and regulatory reports that must be **provably correct** and
**regression-free**. Every agent must pass zero-hallucination verification,
determinism tests, lineage completeness checks, and golden file comparisons
before deployment. Without a production-grade QA test harness:

- **No zero-hallucination verification**: Agents may produce fabricated data
- **No determinism testing**: Same inputs may produce different outputs across runs
- **No lineage verification**: Output data may lack provenance trails
- **No golden file/snapshot testing**: No baseline comparison for regression detection
- **No performance benchmarking**: Agent performance degradation goes undetected
- **No coverage tracking**: Test gaps across agents are invisible
- **No standardized test framework**: Each agent team builds ad-hoc testing
- **No audit trail**: Test executions are not recorded for compliance

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/qa_test_harness.py` (1,763 lines)
- `QATestHarnessAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-008 -- note: ID conflict, should be GL-FOUND-X-009)
- 3 enums: TestStatus(7: pending/running/passed/failed/skipped/error/timeout), TestCategory(9: zero_hallucination/determinism/lineage/golden_file/regression/performance/coverage/integration/unit), SeverityLevel(5: critical/high/medium/low/info)
- 9 Pydantic models: TestAssertion, TestCaseInput, TestCaseResult, TestSuiteInput, TestSuiteResult, GoldenFileSpec, PerformanceBenchmark, CoverageReport, TestFixture(@dataclass)
- Agent registry: register_agent(agent_type, agent_class) for test targets
- Test execution: run_test(TestCaseInput), run_suite(TestSuiteInput) with parallel/sequential modes
- Zero-hallucination testing: test_zero_hallucination() with numeric traceability, provenance checks, output consistency
- Determinism testing: test_determinism() with multi-iteration hash comparison, deep value comparison
- Lineage testing: test_lineage_completeness() with provenance/timestamp/metrics checks
- Golden file testing: test_golden_file(), save_golden_file(), load_golden_file() with JSON comparison
- Regression testing: test_regression() with baseline hash and historical comparison
- Performance benchmarking: benchmark_agent() with warmup, percentiles (p95/p99), std dev, threshold checking
- Coverage tracking: get_coverage_report() with method-level coverage
- Report generation: generate_report() in text/json/markdown formats
- 9 common test fixtures: empty_input, null_values, large_input, special_characters, unicode_input, negative_numbers, zero_values, very_large_numbers, deeply_nested
- In-memory storage (no database persistence)

### 3.2 Layer 1 Tests
None found.

## 4. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/qa_test_harness/` package providing a clean SDK for other agents/services.

### Gap 2: No Prometheus Metrics
No `greenlang/qa_test_harness/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_qa_test_harness(app)` / `get_qa_test_harness(app)` pattern.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 has in-memory storage; doesn't delegate to persistent integration module.

### Gap 5: No REST API Router
No `greenlang/qa_test_harness/api/router.py` with FastAPI endpoints.

### Gap 6: No K8s Deployment Manifests
No `deployment/kubernetes/qa-test-harness-service/` manifests.

### Gap 7: No Database Migration
No `V029__qa_test_harness_service.sql` for persistent test result storage.

### Gap 8: No Monitoring
No Grafana dashboard or alert rules.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/qa-test-harness-ci.yml`.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for QA test harness operations.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/qa_test_harness/
  __init__.py               # Public API exports
  config.py                 # QATestHarnessConfig with GL_QA_TEST_HARNESS_ env prefix
  models.py                 # Pydantic v2 models (re-export + enhance from foundation agent)
  test_runner.py            # TestRunner: execute tests with parallel/sequential, timeout, fail-fast
  assertion_engine.py       # AssertionEngine: zero-hallucination, determinism, lineage assertions
  golden_file_manager.py    # GoldenFileManager: snapshot save/load/compare/update/versioning
  regression_detector.py    # RegressionDetector: baseline comparison, historical trend analysis
  performance_benchmarker.py # PerformanceBenchmarker: timing, percentiles, memory, threshold checks
  coverage_tracker.py       # CoverageTracker: method-level coverage, gap identification
  report_generator.py       # ReportGenerator: text/json/markdown/html reports
  provenance.py             # ProvenanceTracker: SHA-256 hash chain for test execution audit
  metrics.py                # 12 Prometheus metrics
  setup.py                  # QATestHarnessService facade, configure/get
  api/
    __init__.py
    router.py               # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V029)
```sql
CREATE SCHEMA qa_test_harness_service;
-- test_suites (suite definitions and configurations)
-- test_cases (individual test case definitions)
-- test_runs (hypertable - test execution records with results)
-- test_assertions (assertion results linked to runs)
-- golden_files (golden file registry with versioning)
-- performance_baselines (performance benchmark baselines)
-- coverage_snapshots (coverage tracking over time)
-- regression_baselines (regression detection baselines)
-- qa_audit_log (hypertable - test execution audit trail)
```

### 5.3 Prometheus Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_qa_test_runs_total` | Counter | Total test runs by status and category |
| 2 | `gl_qa_test_duration_seconds` | Histogram | Test execution latency |
| 3 | `gl_qa_test_assertions_total` | Counter | Total assertions by result |
| 4 | `gl_qa_test_pass_rate` | Gauge | Current pass rate percentage |
| 5 | `gl_qa_test_failures_total` | Counter | Test failures by severity |
| 6 | `gl_qa_test_regressions_total` | Counter | Regressions detected |
| 7 | `gl_qa_golden_file_mismatches_total` | Counter | Golden file mismatches |
| 8 | `gl_qa_performance_threshold_breaches_total` | Counter | Performance threshold breaches |
| 9 | `gl_qa_coverage_percent` | Gauge | Test coverage percentage by agent |
| 10 | `gl_qa_suites_total` | Counter | Total suite executions |
| 11 | `gl_qa_cache_hits_total` | Counter | Golden file cache hits |
| 12 | `gl_qa_cache_misses_total` | Counter | Golden file cache misses |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/tests/run` | Run a single test case |
| POST | `/v1/suites/run` | Run a test suite |
| GET | `/v1/runs` | List test runs (with filters) |
| GET | `/v1/runs/{run_id}` | Get test run details |
| GET | `/v1/runs/{run_id}/assertions` | Get assertions for a run |
| POST | `/v1/tests/determinism` | Run determinism test |
| POST | `/v1/tests/zero-hallucination` | Run zero-hallucination test |
| POST | `/v1/tests/lineage` | Run lineage completeness test |
| POST | `/v1/tests/regression` | Run regression test |
| POST | `/v1/golden-files` | Save a golden file |
| GET | `/v1/golden-files` | List golden files |
| GET | `/v1/golden-files/{file_id}` | Get golden file details |
| POST | `/v1/golden-files/{file_id}/compare` | Compare against golden file |
| POST | `/v1/benchmarks/run` | Run performance benchmark |
| GET | `/v1/benchmarks/{agent_type}` | Get benchmark history |
| GET | `/v1/coverage/{agent_type}` | Get coverage report |
| GET | `/v1/coverage` | Get coverage summary for all agents |
| POST | `/v1/report` | Generate test report |
| GET | `/v1/statistics` | Get QA statistics |
| GET | `/health` | Service health check |

### 5.5 Key Design Principles
1. **Zero-hallucination verification**: All test assertions use deterministic comparison, no LLM-generated expected values
2. **Determinism guarantee**: Multi-iteration verification with hash comparison
3. **Snapshot/golden file testing**: Versioned golden files with JSON deep comparison
4. **Regression detection**: Baseline hash comparison + historical trend analysis
5. **Performance benchmarking**: Warmup + percentile statistics with configurable thresholds
6. **Coverage tracking**: Method-level coverage with gap identification
7. **Parallel execution**: ThreadPoolExecutor with configurable workers and fail-fast
8. **Complete audit trail**: Every test execution logged with SHA-256 provenance chain
9. **Multi-format reporting**: text/json/markdown/html report generation
10. **Common fixtures**: 9 reusable test fixtures for edge case testing

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/qa_test_harness/__init__.py` - Public API exports
2. Create `greenlang/qa_test_harness/config.py` - QATestHarnessConfig with GL_QA_TEST_HARNESS_ env prefix
3. Create `greenlang/qa_test_harness/models.py` - Pydantic v2 models
4. Create `greenlang/qa_test_harness/test_runner.py` - TestRunner with parallel/sequential execution
5. Create `greenlang/qa_test_harness/assertion_engine.py` - AssertionEngine with category-specific assertions
6. Create `greenlang/qa_test_harness/golden_file_manager.py` - GoldenFileManager with snapshot CRUD
7. Create `greenlang/qa_test_harness/regression_detector.py` - RegressionDetector with baseline comparison
8. Create `greenlang/qa_test_harness/performance_benchmarker.py` - PerformanceBenchmarker with statistics
9. Create `greenlang/qa_test_harness/coverage_tracker.py` - CoverageTracker with method-level tracking
10. Create `greenlang/qa_test_harness/report_generator.py` - ReportGenerator with multi-format output
11. Create `greenlang/qa_test_harness/provenance.py` - ProvenanceTracker
12. Create `greenlang/qa_test_harness/metrics.py` - 12 Prometheus metrics
13. Create `greenlang/qa_test_harness/api/router.py` - FastAPI router with 20 endpoints
14. Create `greenlang/qa_test_harness/setup.py` - QATestHarnessService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V029__qa_test_harness_service.sql`
2. Create K8s manifests in `deployment/kubernetes/qa-test-harness-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-14. Create unit, integration, and load tests (500+ tests target)

## 7. Success Criteria
- Integration module provides clean SDK for all QA test harness operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V029 database migration for persistent test result storage
- 20 REST API endpoints operational
- 500+ tests passing
- Zero-hallucination test framework operational
- Determinism verification with multi-iteration hash comparison
- Golden file/snapshot testing with versioned baselines
- Performance benchmarking with p95/p99 statistics
- Complete audit trail for every test execution

## 8. Integration Points

### 8.1 Upstream Dependencies
- **AGENT-FOUND-001 Orchestrator**: QA gates in DAG execution pipelines
- **AGENT-FOUND-006 Access Guard**: Authorization for test execution
- **AGENT-FOUND-007 Agent Registry**: Discover agents for testing
- **AGENT-FOUND-008 Reproducibility**: Hash verification for determinism tests

### 8.2 Downstream Consumers
- **All agents (001-008+)**: Must pass QA gates before deployment
- **CI/CD pipelines**: Automated test execution on code changes
- **Compliance reporting**: Test results for regulatory audit
- **Admin dashboard**: Test coverage and pass rate visualization

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent test results and audit storage (V029 migration)
- **Redis**: Golden file caching, test result caching
- **Prometheus**: 12 observability metrics
- **Grafana**: QA test harness dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
