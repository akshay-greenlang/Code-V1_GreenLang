# AGENT-FOUND-009: QA Test Harness API Reference

**Agent ID:** AGENT-FOUND-009
**Service:** QA Test Harness
**Status:** Production Ready
**Base Path:** `/api/v1/qa-test-harness`
**Tag:** `qa-test-harness`
**Source:** `greenlang/agents/foundation/qa_test_harness/api/router.py`

The QA Test Harness provides endpoints for running agent tests (individual and
suites), specialized test types (determinism, zero-hallucination, lineage,
regression), golden file management, performance benchmarking, coverage tracking,
and report generation.

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/v1/tests/run` | Run single test case | 200, 400, 503 |
| 2 | POST | `/v1/suites/run` | Run test suite | 200, 400, 503 |
| 3 | GET | `/v1/runs` | List test runs | 200, 503 |
| 4 | GET | `/v1/runs/{run_id}` | Get run details | 200, 404, 503 |
| 5 | GET | `/v1/runs/{run_id}/assertions` | Get assertions for run | 200, 404, 503 |
| 6 | POST | `/v1/tests/determinism` | Run determinism test | 200, 400, 503 |
| 7 | POST | `/v1/tests/zero-hallucination` | Run zero-hallucination test | 200, 400, 503 |
| 8 | POST | `/v1/tests/lineage` | Run lineage test | 200, 400, 503 |
| 9 | POST | `/v1/tests/regression` | Run regression test | 200, 400, 503 |
| 10 | POST | `/v1/golden-files` | Save golden file | 200, 400, 503 |
| 11 | GET | `/v1/golden-files` | List golden files | 200, 503 |
| 12 | GET | `/v1/golden-files/{file_id}` | Get golden file | 200, 404, 503 |
| 13 | POST | `/v1/golden-files/{file_id}/compare` | Compare with golden file | 200, 400, 404, 503 |
| 14 | POST | `/v1/benchmarks/run` | Run performance benchmark | 200, 400, 503 |
| 15 | GET | `/v1/benchmarks/{agent_type}` | Get benchmark baseline | 200, 404, 503 |
| 16 | GET | `/v1/coverage/{agent_type}` | Get coverage report | 200, 503 |
| 17 | GET | `/v1/coverage` | Get all coverage reports | 200, 503 |
| 18 | POST | `/v1/report` | Generate test report | 200, 400, 503 |
| 19 | GET | `/v1/statistics` | Get QA statistics | 200, 503 |
| 20 | GET | `/health` | Health check | 200 |

---

## Detailed Endpoints

### POST /v1/tests/run -- Run Single Test Case

Execute a single test case against a GreenLang agent.

**Request Body:**

```json
{
  "name": "scope1_natural_gas_basic",
  "agent_type": "gl-mrv-scope1-stationary",
  "category": "unit",
  "input_data": {
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3"
  },
  "expected_output": {
    "total_emissions_tCO2e": 3.0
  },
  "golden_file_path": null,
  "timeout_seconds": 60,
  "tags": ["scope1", "natural_gas", "smoke"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Test case name |
| `agent_type` | string | Yes | Agent type to test |
| `category` | string | No | Test category: `unit` (default), `integration`, `regression`, `performance` |
| `input_data` | object | No | Input data for the agent |
| `expected_output` | object | No | Expected output for assertion |
| `golden_file_path` | string | No | Path to golden file for comparison |
| `timeout_seconds` | integer | No | Timeout in seconds (default: 60) |
| `tags` | array | No | Test tags for filtering |

**Response (200):**

```json
{
  "run_id": "trun_abc123",
  "test_name": "scope1_natural_gas_basic",
  "agent_type": "gl-mrv-scope1-stationary",
  "status": "passed",
  "duration_ms": 245.3,
  "assertions": [
    {
      "name": "output_matches_expected",
      "passed": true,
      "message": "Output matches expected within tolerance"
    }
  ],
  "output_data": {
    "total_emissions_tCO2e": 3.0
  },
  "provenance_hash": "sha256:..."
}
```

---

### POST /v1/suites/run -- Run Test Suite

Execute a collection of test cases as a suite, optionally in parallel.

**Request Body:**

```json
{
  "name": "Scope 1 Full Suite",
  "test_cases": [
    {
      "name": "natural_gas_basic",
      "agent_type": "gl-mrv-scope1-stationary",
      "input_data": { "fuel_type": "natural_gas", "quantity": 1500.0, "unit": "m3" },
      "expected_output": { "total_emissions_tCO2e": 3.0 }
    },
    {
      "name": "diesel_mobile",
      "agent_type": "gl-mrv-scope1-mobile",
      "input_data": { "fuel_type": "diesel", "quantity": 200.0, "unit": "liters" },
      "expected_output": { "total_emissions_tCO2e": 0.53 }
    }
  ],
  "parallel": true,
  "max_workers": 4,
  "fail_fast": false,
  "tags_include": ["scope1"],
  "tags_exclude": ["slow"]
}
```

**Response (200):**

```json
{
  "suite_id": "suite_xyz789",
  "name": "Scope 1 Full Suite",
  "status": "passed",
  "total_tests": 2,
  "passed": 2,
  "failed": 0,
  "skipped": 0,
  "duration_ms": 520.7,
  "results": [ ... ]
}
```

---

### POST /v1/tests/determinism -- Determinism Test

Verify that an agent produces identical output across multiple runs with the
same input, ensuring calculation determinism.

**Request Body:**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "input_data": {
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3"
  },
  "iterations": 5
}
```

**Response (200):**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "deterministic": true,
  "iterations": 5,
  "unique_hashes": 1,
  "output_hash": "sha256:a1b2c3..."
}
```

---

### POST /v1/tests/zero-hallucination -- Zero-Hallucination Test

Verify that an agent does not produce fabricated values -- every output value
must be traceable to an input, emission factor, or cited source.

**Request Body:**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "input_data": {
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3"
  },
  "checks": ["output_traceability", "citation_completeness", "no_invented_factors"]
}
```

**Response (200):**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "hallucination_detected": false,
  "checks_passed": 3,
  "checks_failed": 0,
  "details": [ ... ]
}
```

---

### POST /v1/golden-files -- Save Golden File

Save a known-good input/output pair as a golden file for future regression
testing.

**Request Body:**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "name": "natural_gas_1500m3_golden",
  "input_data": {
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3"
  },
  "output_data": {
    "total_emissions_tCO2e": 3.0,
    "emission_factor_used": 0.002
  },
  "description": "Baseline calculation for 1500m3 natural gas"
}
```

**Response (200):**

```json
{
  "file_id": "gf_abc123",
  "agent_type": "gl-mrv-scope1-stationary",
  "name": "natural_gas_1500m3_golden",
  "input_hash": "sha256:...",
  "output_hash": "sha256:...",
  "created_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/golden-files/{file_id}/compare -- Compare With Golden File

Compare an agent's result against a stored golden file.

**Request Body:**

```json
{
  "agent_result": {
    "total_emissions_tCO2e": 3.0,
    "emission_factor_used": 0.002
  }
}
```

**Response (200):**

```json
{
  "assertions": [
    { "name": "hash_match", "passed": true, "message": "Output hash matches golden file" },
    { "name": "value_match", "passed": true, "message": "All values within tolerance" }
  ],
  "count": 2,
  "all_passed": true
}
```

---

### POST /v1/benchmarks/run -- Run Performance Benchmark

**Request Body:**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "input_data": { "fuel_type": "natural_gas", "quantity": 1500.0, "unit": "m3" },
  "iterations": 50,
  "warmup": 5,
  "threshold_ms": 100.0
}
```

**Response (200):**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "iterations": 50,
  "warmup": 5,
  "min_ms": 8.2,
  "max_ms": 42.1,
  "mean_ms": 15.3,
  "median_ms": 13.8,
  "p95_ms": 28.5,
  "p99_ms": 39.2,
  "within_threshold": true,
  "threshold_ms": 100.0
}
```

---

### GET /v1/coverage/{agent_type} -- Coverage Report

**Response (200):**

```json
{
  "agent_type": "gl-mrv-scope1-stationary",
  "total_test_cases": 45,
  "categories_covered": {
    "unit": 20,
    "integration": 10,
    "regression": 10,
    "performance": 5
  },
  "input_coverage_pct": 92.5,
  "edge_cases_covered": 15,
  "last_run_at": "2026-04-04T09:00:00Z"
}
```
