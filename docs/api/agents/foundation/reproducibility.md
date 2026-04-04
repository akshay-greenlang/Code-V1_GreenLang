# AGENT-FOUND-008: Reproducibility Agent API Reference

**Agent ID:** AGENT-FOUND-008
**Service:** Reproducibility Agent
**Status:** Production Ready
**Base Path:** `/api/v1/reproducibility`
**Tag:** `reproducibility`
**Source:** `greenlang/agents/foundation/reproducibility/api/router.py`

The Reproducibility Agent ensures that calculation results can be independently
verified and reproduced. It provides endpoints for deterministic hashing,
verification (input/output), drift detection, replay execution, environment
fingerprinting, version pinning, and reporting.

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/v1/verify` | Full reproducibility verification | 200, 400, 503 |
| 2 | POST | `/v1/verify/input` | Verify input hash only | 200, 503 |
| 3 | POST | `/v1/verify/output` | Verify output hash only | 200, 503 |
| 4 | GET | `/v1/verifications` | List verification runs | 200, 503 |
| 5 | GET | `/v1/verifications/{verification_id}` | Get verification details | 200, 404, 503 |
| 6 | POST | `/v1/hash` | Compute deterministic hash | 200, 503 |
| 7 | GET | `/v1/hashes/{artifact_id}` | Get artifact hash history | 200, 503 |
| 8 | POST | `/v1/drift/detect` | Run drift detection | 200, 400, 404, 503 |
| 9 | GET | `/v1/drift/baselines` | List drift baselines | 200, 503 |
| 10 | POST | `/v1/drift/baselines` | Create drift baseline | 200, 400, 503 |
| 11 | GET | `/v1/drift/baselines/{baseline_id}` | Get baseline | 200, 404, 503 |
| 12 | POST | `/v1/replay` | Execute replay session | 200, 400, 503 |
| 13 | GET | `/v1/replays/{replay_id}` | Get replay session | 200, 404, 503 |
| 14 | GET | `/v1/environment` | Capture current environment | 200, 503 |
| 15 | GET | `/v1/environment/{fingerprint_id}` | Get stored fingerprint | 200, 404, 503 |
| 16 | POST | `/v1/versions/pin` | Pin current versions | 200, 503 |
| 17 | GET | `/v1/versions/manifest/{manifest_id}` | Get version manifest | 200, 404, 503 |
| 18 | POST | `/v1/report` | Generate reproducibility report | 200, 404, 503 |
| 19 | GET | `/v1/statistics` | Get service statistics | 200, 503 |
| 20 | GET | `/health` | Health check | 200 |

---

## Detailed Endpoints

### POST /v1/verify -- Full Reproducibility Verification

Run a complete reproducibility verification comparing input and output hashes
against expected values within specified tolerances.

**Request Body:**

```json
{
  "execution_id": "exec_abc123",
  "input_data": {
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3",
    "emission_factor": 0.002
  },
  "output_data": {
    "total_emissions_tCO2e": 3.0,
    "calculation_method": "tier1"
  },
  "expected_input_hash": "sha256:a1b2c3...",
  "expected_output_hash": "sha256:d4e5f6...",
  "absolute_tolerance": 1e-9,
  "relative_tolerance": 1e-6
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `execution_id` | string | Yes | Unique execution identifier |
| `input_data` | object | Yes | Input data to verify |
| `output_data` | object | No | Output data to verify |
| `expected_input_hash` | string | No | Expected input hash |
| `expected_output_hash` | string | No | Expected output hash |
| `absolute_tolerance` | float | No | Absolute tolerance (default: 1e-9) |
| `relative_tolerance` | float | No | Relative tolerance (default: 1e-6) |

**Response (200):**

```json
{
  "verification_id": "ver_xyz789",
  "execution_id": "exec_abc123",
  "status": "passed",
  "input_check": {
    "hash": "sha256:a1b2c3...",
    "matches_expected": true
  },
  "output_check": {
    "hash": "sha256:d4e5f6...",
    "matches_expected": true,
    "within_tolerance": true
  },
  "verified_at": "2026-04-04T10:00:00Z",
  "provenance_hash": "sha256:..."
}
```

---

### POST /v1/hash -- Compute Deterministic Hash

Compute a deterministic hash of arbitrary data using canonical normalization.

**Request Body:**

```json
{
  "data": {
    "fuel_type": "natural_gas",
    "quantity": 1500.0
  },
  "algorithm": "sha256"
}
```

**Response (200):**

```json
{
  "data_hash": "sha256:a1b2c3d4e5f6...",
  "algorithm": "sha256",
  "normalization_applied": true
}
```

---

### POST /v1/drift/detect -- Run Drift Detection

Compare current data against a baseline to detect calculation drift.

**Request Body:**

```json
{
  "baseline_id": "bl_abc123",
  "current_data": {
    "total_emissions_tCO2e": 3.05,
    "scope1": 1.5,
    "scope2": 1.55
  },
  "soft_threshold": 0.01,
  "hard_threshold": 0.05,
  "tolerance": 1e-9
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `baseline_id` | string | No* | Reference baseline ID |
| `baseline_data` | object | No* | Inline baseline data |
| `current_data` | object | Yes | Current data to compare |
| `soft_threshold` | float | No | Soft drift threshold (default: 0.01 = 1%) |
| `hard_threshold` | float | No | Hard drift threshold (default: 0.05 = 5%) |
| `tolerance` | float | No | Absolute tolerance (default: 1e-9) |

*One of `baseline_id` or `baseline_data` must be provided.

**Response (200):**

```json
{
  "drift_detection": {
    "drift_detected": true,
    "severity": "soft",
    "drifted_fields": [
      {
        "path": "scope2",
        "baseline_value": 1.50,
        "current_value": 1.55,
        "absolute_drift": 0.05,
        "relative_drift": 0.033
      }
    ],
    "summary": {
      "total_fields": 3,
      "drifted_count": 1,
      "max_relative_drift": 0.033
    }
  },
  "baseline_id": "bl_abc123"
}
```

---

### POST /v1/replay -- Execute Replay

Re-execute a past calculation using captured inputs, environment, seeds, and
version pins to verify reproducibility.

**Request Body:**

```json
{
  "original_execution_id": "exec_abc123",
  "captured_inputs": { ... },
  "captured_environment": {
    "python_version": "3.11.8",
    "os": "linux",
    "architecture": "x86_64"
  },
  "captured_seeds": {
    "random_seed": 42,
    "numpy_seed": 42
  },
  "captured_versions": {
    "greenlang": "2.1.0",
    "numpy": "1.26.4"
  },
  "strict_mode": true
}
```

**Response (200):**

```json
{
  "replay_session": {
    "replay_id": "rpl_xyz789",
    "original_execution_id": "exec_abc123",
    "status": "completed",
    "reproducible": true,
    "output_hash_match": true,
    "environment_match": true,
    "version_match": true,
    "replayed_at": "2026-04-04T10:05:00Z"
  }
}
```

---

### GET /v1/environment -- Capture Environment

Capture a fingerprint of the current execution environment.

**Response (200):**

```json
{
  "fingerprint_id": "fp_abc123",
  "python_version": "3.11.8",
  "os": "linux",
  "architecture": "x86_64",
  "cpu_count": 8,
  "hostname": "worker-01",
  "installed_packages": {
    "greenlang": "2.1.0",
    "numpy": "1.26.4",
    "pandas": "2.2.0"
  },
  "environment_hash": "sha256:...",
  "captured_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/versions/pin -- Pin Versions

Pin current library versions as a version manifest for reproducibility.

**Request Body:**

```json
{
  "auto_detect": true
}
```

**Response (200):**

```json
{
  "manifest_id": "mfst_abc123",
  "versions": {
    "greenlang": "2.1.0",
    "numpy": "1.26.4",
    "pandas": "2.2.0"
  },
  "pinned_at": "2026-04-04T10:00:00Z",
  "manifest_hash": "sha256:..."
}
```

---

### POST /v1/report -- Generate Report

Generate a human-readable reproducibility report for an execution.

**Request Body:**

```json
{
  "execution_id": "exec_abc123",
  "verification_id": "ver_xyz789"
}
```

**Response (200):**

```json
{
  "report_id": "rpt_abc123",
  "execution_id": "exec_abc123",
  "verification_status": "passed",
  "sections": [ ... ],
  "generated_at": "2026-04-04T10:10:00Z"
}
```
