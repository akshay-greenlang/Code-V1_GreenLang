# GLIP v1 - GreenLang Invocation Protocol Specification

**Version:** 1.0.0
**Status:** DRAFT
**Document ID:** GL-FOUND-X-001-SPEC-001
**Last Updated:** 2026-01-27
**Authors:** GreenLang Platform Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Terminology](#2-terminology)
3. [Agent Container Packaging](#3-agent-container-packaging)
4. [Input Envelope (RunContext)](#4-input-envelope-runcontext)
5. [Output Format](#5-output-format)
6. [Exit Code Semantics](#6-exit-code-semantics)
7. [Idempotency](#7-idempotency)
8. [Complete Examples](#8-complete-examples)
9. [JSON Schemas](#9-json-schemas)
10. [Security Considerations](#10-security-considerations)
11. [Conformance](#11-conformance)
12. [Changelog](#12-changelog)

---

## 1. Overview

### 1.1 Purpose

The GreenLang Invocation Protocol (GLIP) v1 defines a standardized, artifact-first, Kubernetes Job-native interface for invoking GreenLang agents. This specification ensures that all 402+ agents in the GreenLang ecosystem can be orchestrated uniformly by GL-FOUND-X-001 (GreenLang Orchestrator) regardless of their internal implementation.

### 1.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Artifact-First** | All inputs and outputs are explicit files or URIs with checksums |
| **K8s Job-Native** | Designed for Kubernetes Job execution model (init, run, cleanup) |
| **Deterministic** | Same inputs always produce same outputs (with idempotency keys) |
| **Observable** | Full tracing, metrics, and audit trail support |
| **Secure** | Tenant isolation, permission enforcement, signed artifacts |

### 1.3 Scope

This specification applies to:

- All GreenLang agents (402 agents across 10 application domains)
- GreenLang Orchestrator (GL-FOUND-X-001)
- Pipeline execution runtime
- Agent development SDK

### 1.4 Protocol Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GLIP v1 Execution Flow                            │
└─────────────────────────────────────────────────────────────────────────────┘

  Orchestrator                        Agent Container
       │                                    │
       │  1. Create K8s Job with:           │
       │     - GL_INPUT_URI env var         │
       │     - GL_OUTPUT_URI env var        │
       │     - Container image + labels     │
       ├───────────────────────────────────>│
       │                                    │
       │                              ┌─────┴─────┐
       │                              │ Init Phase│
       │                              │ - Fetch   │
       │                              │   input   │
       │                              │ - Validate│
       │                              │   schema  │
       │                              └─────┬─────┘
       │                                    │
       │                              ┌─────┴─────┐
       │                              │ Run Phase │
       │                              │ - Execute │
       │                              │   agent   │
       │                              │ - Generate│
       │                              │   outputs │
       │                              └─────┬─────┘
       │                                    │
       │                              ┌─────┴─────┐
       │                              │ Finalize  │
       │                              │ - Upload  │
       │                              │   results │
       │                              │ - Write   │
       │                              │   metadata│
       │                              └─────┬─────┘
       │                                    │
       │  2. Job completes with exit code   │
       │<───────────────────────────────────┤
       │                                    │
       │  3. Orchestrator fetches results   │
       │     from GL_OUTPUT_URI             │
       │                                    │
       ▼                                    ▼
```

---

## 2. Terminology

| Term | Definition |
|------|------------|
| **Agent** | A containerized processing unit that implements GLIP v1 |
| **RunContext** | The JSON envelope containing all inputs and metadata for an agent invocation |
| **Artifact** | Any file produced by an agent (data files, reports, intermediate results) |
| **Step** | A single agent invocation within a pipeline |
| **Pipeline** | An ordered sequence of steps with dependencies |
| **Orchestrator** | The system (GL-FOUND-X-001) that schedules and monitors agent execution |
| **Tenant** | An isolated customer environment with its own data and permissions |
| **Idempotency Key** | A unique identifier ensuring repeated invocations produce identical results |

---

## 3. Agent Container Packaging

### 3.1 Required Container Labels

All GLIP v1 compliant agent containers MUST include the following OCI labels:

| Label | Format | Required | Description |
|-------|--------|----------|-------------|
| `gl.agent.id` | `string` | **Yes** | Unique agent identifier (e.g., `REG-001`, `CBAM-CALC-001`) |
| `gl.agent.version` | `semver` | **Yes** | Agent version following SemVer 2.0.0 (e.g., `1.2.3`) |
| `gl.schema.version` | `semver` | **Yes** | GLIP schema version this agent implements (e.g., `1.0.0`) |
| `gl.agent.name` | `string` | No | Human-readable agent name |
| `gl.agent.description` | `string` | No | Brief description of agent functionality |
| `gl.agent.category` | `string` | No | Agent category (e.g., `regulatory`, `calculation`, `reporting`) |
| `gl.agent.capabilities` | `csv` | No | Comma-separated list of capabilities |
| `gl.agent.maintainer` | `string` | No | Maintainer contact information |

**Example Dockerfile Labels:**

```dockerfile
FROM python:3.11-slim

LABEL gl.agent.id="CBAM-CALC-001" \
      gl.agent.version="2.1.0" \
      gl.schema.version="1.0.0" \
      gl.agent.name="CBAM Emissions Calculator" \
      gl.agent.description="Calculates embedded emissions for CBAM-covered goods" \
      gl.agent.category="regulatory" \
      gl.agent.capabilities="cbam,emissions,calculation,eu-regulation" \
      gl.agent.maintainer="platform@greenlang.io"

# ... rest of Dockerfile
```

### 3.2 Entrypoint Requirements

#### 3.2.1 Standard Entrypoint

The container MUST support invocation via the `gl-agent run` command or an equivalent entrypoint that:

1. Reads the `GL_INPUT_URI` environment variable
2. Fetches and parses the RunContext JSON
3. Executes the agent logic
4. Writes outputs to `GL_OUTPUT_URI`
5. Exits with appropriate exit code

**Standard Entrypoint Format:**

```bash
# Primary entrypoint (RECOMMENDED)
gl-agent run

# Alternative: Direct Python invocation
python -m greenlang.agent.main run

# Alternative: Custom entrypoint script
/app/entrypoint.sh run
```

**Dockerfile Example:**

```dockerfile
# Using GreenLang base image with gl-agent CLI
FROM greenlang/agent-base:1.0.0

COPY . /app
WORKDIR /app

# Standard entrypoint
ENTRYPOINT ["gl-agent"]
CMD ["run"]
```

#### 3.2.2 Entrypoint Contract

The entrypoint MUST:

| Requirement | Description |
|-------------|-------------|
| Read `GL_INPUT_URI` | Parse environment variable to locate input envelope |
| Fetch RunContext | Download and validate the input JSON from the URI |
| Verify checksums | Validate all input artifact checksums before processing |
| Execute agent | Run the agent logic with the provided parameters |
| Write outputs | Upload `result.json`, `step_metadata.json`, and artifacts to `GL_OUTPUT_URI` |
| Handle signals | Respond to SIGTERM for graceful cancellation |
| Exit cleanly | Return appropriate exit code (see Section 6) |

#### 3.2.3 Supported URI Schemes

| Scheme | Description | Example |
|--------|-------------|---------|
| `file://` | Local filesystem (testing/development) | `file:///data/input/context.json` |
| `s3://` | AWS S3 | `s3://gl-artifacts/runs/abc123/input.json` |
| `gs://` | Google Cloud Storage | `gs://gl-artifacts/runs/abc123/input.json` |
| `az://` | Azure Blob Storage | `az://glartifacts/runs/abc123/input.json` |
| `https://` | Pre-signed HTTPS URL | `https://storage.greenlang.io/...` |

### 3.3 Resource Requirements

Agents SHOULD declare resource requirements via labels:

```dockerfile
LABEL gl.resources.cpu.request="100m" \
      gl.resources.cpu.limit="2000m" \
      gl.resources.memory.request="256Mi" \
      gl.resources.memory.limit="4Gi" \
      gl.resources.gpu="false"
```

---

## 4. Input Envelope (RunContext)

### 4.1 Delivery Mechanism

The RunContext is delivered to agents via the `GL_INPUT_URI` environment variable. This URI points to a JSON file containing all information needed for agent execution.

**Environment Variable:**

```bash
GL_INPUT_URI=s3://gl-artifacts/tenant-123/runs/run-abc/steps/step-001/input.json
```

### 4.2 RunContext Schema

```json
{
  "$schema": "https://greenlang.io/schema/glip/run-context.v1.json",
  "run_id": "string (required)",
  "step_id": "string (required)",
  "pipeline_id": "string (required)",
  "tenant_id": "string (required)",
  "agent_id": "string (required)",
  "agent_version": "string (required)",
  "schema_version": "string (required)",
  "params": "object (required)",
  "inputs": "array (required)",
  "permissions_context": "object (required)",
  "deadline_ts": "string (required, ISO 8601)",
  "timeout_s": "integer (required)",
  "retry_attempt": "integer (required)",
  "idempotency_key": "string (required)",
  "observability": "object (required)"
}
```

### 4.3 Field Specifications

#### 4.3.1 Identification Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | string | **Yes** | Globally unique identifier for this pipeline run (UUID v4) |
| `step_id` | string | **Yes** | Unique identifier for this step within the pipeline |
| `pipeline_id` | string | **Yes** | Identifier of the pipeline definition being executed |
| `tenant_id` | string | **Yes** | Tenant/organization identifier for data isolation |

**Format Requirements:**

- `run_id`: UUID v4 format (e.g., `550e8400-e29b-41d4-a716-446655440000`)
- `step_id`: Pipeline-scoped identifier (e.g., `calculate_emissions_001`)
- `pipeline_id`: Slug format (e.g., `cbam-quarterly-report-v2`)
- `tenant_id`: Organization identifier (e.g., `org_acme_corp_eu`)

#### 4.3.2 Agent Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | string | **Yes** | Identifier of the agent being invoked (must match container label) |
| `agent_version` | string | **Yes** | SemVer version of the agent to execute |
| `schema_version` | string | **Yes** | GLIP schema version (always `1.0.0` for this spec) |

#### 4.3.3 Parameters (params)

The `params` object contains agent-specific parameters as defined in the pipeline YAML configuration.

**Structure:**

```json
{
  "params": {
    "calculation_method": "activity-based",
    "emission_factor_source": "EU-ETS-2024",
    "precision": "high",
    "include_uncertainty": true,
    "custom_factors": {
      "steel_primary": 1.85,
      "steel_secondary": 0.42
    }
  }
}
```

**Requirements:**

- Agents MUST validate params against their expected schema
- Unknown params SHOULD be logged as warnings but not cause failures
- Required params missing MUST result in exit code 2

#### 4.3.4 Inputs (Upstream Artifacts)

The `inputs` array contains references to artifacts from upstream steps or external sources.

**Structure:**

```json
{
  "inputs": [
    {
      "name": "shipment_data",
      "uri": "s3://gl-artifacts/tenant-123/runs/run-abc/steps/intake/artifacts/shipments.csv",
      "checksum": {
        "algorithm": "sha256",
        "value": "a3f2b8c9d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
      },
      "size_bytes": 1048576,
      "media_type": "text/csv",
      "schema_ref": "https://greenlang.io/schema/cbam/shipments.v1.json",
      "source_step": "data_intake"
    },
    {
      "name": "emission_factors",
      "uri": "s3://gl-reference-data/emission-factors/eu-ets-2024.json",
      "checksum": {
        "algorithm": "sha256",
        "value": "b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3"
      },
      "size_bytes": 524288,
      "media_type": "application/json",
      "schema_ref": "https://greenlang.io/schema/emission-factors.v2.json",
      "source_step": null
    }
  ]
}
```

**Input Object Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Logical name for referencing this input |
| `uri` | string | **Yes** | URI to fetch the artifact |
| `checksum` | object | **Yes** | Cryptographic checksum for integrity verification |
| `checksum.algorithm` | string | **Yes** | Algorithm used (`sha256`, `sha384`, `sha512`) |
| `checksum.value` | string | **Yes** | Hex-encoded checksum value |
| `size_bytes` | integer | No | Expected file size in bytes |
| `media_type` | string | No | MIME type of the artifact |
| `schema_ref` | string | No | URI to JSON Schema for validation |
| `source_step` | string | No | Step ID that produced this artifact (null for external) |

#### 4.3.5 Permissions Context

The `permissions_context` object defines what the agent is authorized to access and perform.

**Structure:**

```json
{
  "permissions_context": {
    "tenant_id": "org_acme_corp_eu",
    "user_id": "user_john_doe_001",
    "roles": ["pipeline_executor", "cbam_operator"],
    "scopes": [
      "read:shipments",
      "read:emission-factors",
      "write:calculations",
      "write:reports"
    ],
    "data_classification": "confidential",
    "allowed_regions": ["eu-west-1", "eu-central-1"],
    "network_policy": {
      "allow_external": false,
      "allowed_hosts": ["api.eu-ets.europa.eu", "storage.greenlang.io"]
    },
    "resource_quotas": {
      "max_memory_mb": 4096,
      "max_cpu_millicores": 2000,
      "max_runtime_seconds": 3600
    }
  }
}
```

#### 4.3.6 Execution Control Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `deadline_ts` | string | **Yes** | ISO 8601 timestamp by which execution must complete |
| `timeout_s` | integer | **Yes** | Maximum execution time in seconds |
| `retry_attempt` | integer | **Yes** | Current retry attempt (0 for first attempt) |
| `idempotency_key` | string | **Yes** | Key for ensuring idempotent execution |

**Example:**

```json
{
  "deadline_ts": "2026-01-27T15:30:00Z",
  "timeout_s": 3600,
  "retry_attempt": 0,
  "idempotency_key": "glip-run-abc123-step-calc-001-v1"
}
```

#### 4.3.7 Observability Context

The `observability` object provides tracing and logging correlation IDs.

**Structure:**

```json
{
  "observability": {
    "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
    "span_id": "00f067aa0ba902b7",
    "parent_span_id": "a3ce929d0e0e4736",
    "trace_flags": "01",
    "trace_state": "gl=1,tenant=acme",
    "log_correlation_id": "log-run-abc123-step-calc-001",
    "metrics_labels": {
      "pipeline": "cbam-quarterly-report",
      "agent": "CBAM-CALC-001",
      "tenant": "org_acme_corp_eu",
      "environment": "production"
    },
    "baggage": {
      "user.id": "user_john_doe_001",
      "request.origin": "api-gateway"
    }
  }
}
```

**Observability Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | string | W3C Trace Context trace ID (32 hex chars) |
| `span_id` | string | W3C Trace Context span ID (16 hex chars) |
| `parent_span_id` | string | Parent span ID for trace correlation |
| `trace_flags` | string | W3C Trace Context flags |
| `trace_state` | string | W3C Trace Context state |
| `log_correlation_id` | string | ID for correlating log entries |
| `metrics_labels` | object | Labels to attach to emitted metrics |
| `baggage` | object | W3C Baggage for cross-service context |

---

## 5. Output Format

### 5.1 Output Location

Agents write outputs to the location specified by the `GL_OUTPUT_URI` environment variable.

**Environment Variable:**

```bash
GL_OUTPUT_URI=s3://gl-artifacts/tenant-123/runs/run-abc/steps/step-001/output/
```

### 5.2 Required Output Structure

```
{output_uri}/
├── result.json           # Required: Structured agent output
├── step_metadata.json    # Required: Execution metadata
├── error.json            # Conditional: Error details (on failure)
└── artifacts/            # Optional: Additional output files
    ├── report.pdf
    ├── calculations.csv
    └── audit_log.json
```

### 5.3 result.json Specification

The `result.json` file contains the structured output of the agent execution.

**Schema:**

```json
{
  "$schema": "https://greenlang.io/schema/glip/result.v1.json",
  "status": "string (required: 'success' | 'partial' | 'failed')",
  "agent_id": "string (required)",
  "agent_version": "string (required)",
  "run_id": "string (required)",
  "step_id": "string (required)",
  "started_at": "string (required, ISO 8601)",
  "completed_at": "string (required, ISO 8601)",
  "outputs": "object (required)",
  "artifacts": "array (required)",
  "warnings": "array (optional)",
  "metrics": "object (optional)"
}
```

**Field Specifications:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | enum | **Yes** | Execution status: `success`, `partial`, or `failed` |
| `agent_id` | string | **Yes** | Agent identifier (echo from input) |
| `agent_version` | string | **Yes** | Agent version (echo from input) |
| `run_id` | string | **Yes** | Run identifier (echo from input) |
| `step_id` | string | **Yes** | Step identifier (echo from input) |
| `started_at` | string | **Yes** | ISO 8601 timestamp of execution start |
| `completed_at` | string | **Yes** | ISO 8601 timestamp of execution completion |
| `outputs` | object | **Yes** | Agent-specific output data |
| `artifacts` | array | **Yes** | List of artifacts produced (can be empty) |
| `warnings` | array | No | Non-fatal warnings encountered |
| `metrics` | object | No | Performance and quality metrics |

### 5.4 step_metadata.json Specification

The `step_metadata.json` file contains execution metadata for audit and provenance.

**Schema:**

```json
{
  "$schema": "https://greenlang.io/schema/glip/step-metadata.v1.json",
  "schema_version": "string (required)",
  "run_id": "string (required)",
  "step_id": "string (required)",
  "idempotency_key": "string (required)",
  "execution": "object (required)",
  "inputs_consumed": "array (required)",
  "outputs_produced": "array (required)",
  "provenance": "object (required)",
  "resource_usage": "object (optional)",
  "audit": "object (required)"
}
```

**Execution Object:**

```json
{
  "execution": {
    "started_at": "2026-01-27T14:00:00.000Z",
    "completed_at": "2026-01-27T14:05:23.456Z",
    "duration_ms": 323456,
    "retry_attempt": 0,
    "exit_code": 0,
    "container_id": "abc123def456",
    "node_id": "k8s-node-eu-west-1a-001",
    "pod_name": "glip-run-abc123-step-calc-001-xyz789"
  }
}
```

**Inputs Consumed:**

```json
{
  "inputs_consumed": [
    {
      "name": "shipment_data",
      "uri": "s3://gl-artifacts/.../shipments.csv",
      "checksum_verified": true,
      "size_bytes": 1048576,
      "records_processed": 15000
    }
  ]
}
```

**Outputs Produced:**

```json
{
  "outputs_produced": [
    {
      "name": "result.json",
      "uri": "s3://gl-artifacts/.../output/result.json",
      "checksum": {
        "algorithm": "sha256",
        "value": "c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
      },
      "size_bytes": 2048,
      "media_type": "application/json"
    },
    {
      "name": "artifacts/calculations.csv",
      "uri": "s3://gl-artifacts/.../output/artifacts/calculations.csv",
      "checksum": {
        "algorithm": "sha256",
        "value": "d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7"
      },
      "size_bytes": 524288,
      "media_type": "text/csv"
    }
  ]
}
```

**Provenance Object:**

```json
{
  "provenance": {
    "agent_image": "greenlang/cbam-calc:2.1.0@sha256:abc123...",
    "agent_id": "CBAM-CALC-001",
    "agent_version": "2.1.0",
    "glip_schema_version": "1.0.0",
    "base_image": "greenlang/agent-base:1.0.0",
    "build_timestamp": "2026-01-15T10:00:00Z",
    "git_commit": "a1b2c3d4e5f6",
    "dependencies_hash": "sha256:def456..."
  }
}
```

**Resource Usage:**

```json
{
  "resource_usage": {
    "cpu_seconds": 245.67,
    "memory_peak_mb": 1536,
    "memory_avg_mb": 1024,
    "network_rx_bytes": 2097152,
    "network_tx_bytes": 524288,
    "disk_read_bytes": 1048576,
    "disk_write_bytes": 2097152
  }
}
```

**Audit Object:**

```json
{
  "audit": {
    "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
    "span_id": "00f067aa0ba902b7",
    "tenant_id": "org_acme_corp_eu",
    "user_id": "user_john_doe_001",
    "pipeline_id": "cbam-quarterly-report-v2",
    "deterministic": true,
    "cached_result": false,
    "signature": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

### 5.5 error.json Specification (Failure Cases)

When an agent exits with a non-zero code, it SHOULD produce an `error.json` file with details.

**Schema:**

```json
{
  "$schema": "https://greenlang.io/schema/glip/error.v1.json",
  "error_code": "string (required)",
  "error_type": "string (required)",
  "message": "string (required)",
  "details": "object (optional)",
  "stack_trace": "string (optional)",
  "retriable": "boolean (required)",
  "suggested_action": "string (optional)",
  "documentation_url": "string (optional)"
}
```

**Error Types:**

| Error Type | Description |
|------------|-------------|
| `INPUT_VALIDATION_ERROR` | Input data failed schema validation |
| `CHECKSUM_MISMATCH` | Input artifact checksum verification failed |
| `PERMISSION_DENIED` | Agent lacks required permissions |
| `TIMEOUT` | Execution exceeded allowed time |
| `RESOURCE_EXHAUSTED` | Memory, CPU, or storage limits exceeded |
| `UPSTREAM_FAILURE` | Required upstream service unavailable |
| `CALCULATION_ERROR` | Agent-specific calculation failure |
| `OUTPUT_VALIDATION_ERROR` | Generated output failed validation |
| `UNKNOWN_ERROR` | Unexpected error condition |

### 5.6 Artifacts Directory

The `artifacts/` directory contains any additional files produced by the agent.

**Requirements:**

- All artifacts MUST be registered in `step_metadata.json` with checksums
- Artifacts SHOULD use descriptive filenames
- Large artifacts (>100MB) SHOULD be compressed
- Sensitive data MUST be encrypted if written to artifacts

**Common Artifact Types:**

| Type | Extension | Media Type | Description |
|------|-----------|------------|-------------|
| Data export | `.csv`, `.json`, `.parquet` | `text/csv`, `application/json` | Processed data files |
| Reports | `.pdf`, `.html`, `.xlsx` | `application/pdf` | Human-readable reports |
| Audit logs | `.json`, `.jsonl` | `application/json` | Detailed audit trails |
| Visualizations | `.png`, `.svg` | `image/png`, `image/svg+xml` | Charts and graphs |
| Archives | `.zip`, `.tar.gz` | `application/zip` | Bundled outputs |

---

## 6. Exit Code Semantics

### 6.1 Exit Code Definitions

| Exit Code | Name | Description | Orchestrator Action |
|-----------|------|-------------|---------------------|
| `0` | SUCCESS | Agent completed successfully | Proceed to next step |
| `1` | GENERAL_FAILURE | Unspecified failure | Check error.json, may retry |
| `2` | INPUT_VALIDATION_FAILURE | Input data invalid | Do not retry, fix input |
| `3` | CHECKSUM_FAILURE | Input checksum mismatch | Retry with fresh fetch |
| `4` | PERMISSION_FAILURE | Authorization denied | Do not retry, check permissions |
| `5` | TIMEOUT_FAILURE | Execution timed out | May retry with longer timeout |
| `6` | RESOURCE_FAILURE | Resource limits exceeded | May retry with more resources |
| `7` | OUTPUT_VALIDATION_FAILURE | Output validation failed | Check agent logic |
| `8` | UPSTREAM_FAILURE | Upstream dependency failed | Retry after delay |
| `125-127` | CONTAINER_ERROR | Container infrastructure error | Retry on different node |
| `128+N` | SIGNAL_TERMINATION | Terminated by signal N | See signal handling |

### 6.2 Signal Handling

Agents MUST handle the following signals:

#### 6.2.1 SIGTERM (Signal 15)

Sent when the orchestrator requests graceful cancellation.

**Required Behavior:**

1. Stop accepting new work
2. Complete or checkpoint in-progress operations (within 30 seconds)
3. Write partial results to `result.json` with `status: "partial"`
4. Write `step_metadata.json` with current progress
5. Exit with code `0` if checkpoint successful, `1` otherwise

**Example Handler (Python):**

```python
import signal
import sys

class GracefulShutdown:
    shutdown_requested = False

    @classmethod
    def handler(cls, signum, frame):
        cls.shutdown_requested = True
        # Complete current operation, then exit

signal.signal(signal.SIGTERM, GracefulShutdown.handler)
```

#### 6.2.2 SIGKILL (Signal 9)

Forced termination after SIGTERM grace period (30 seconds).

**Behavior:**

- Agent is immediately terminated
- No cleanup possible
- Orchestrator treats as failed execution
- Will retry if retry policy allows

### 6.3 Exit Code Examples

```bash
# Success
exit 0

# Input validation failed
echo '{"error_code": "E001", "error_type": "INPUT_VALIDATION_ERROR", ...}' > error.json
exit 2

# Timeout
echo '{"error_code": "E005", "error_type": "TIMEOUT", "retriable": true, ...}' > error.json
exit 5

# Graceful shutdown (partial results saved)
exit 0  # With status: "partial" in result.json
```

---

## 7. Idempotency

### 7.1 Idempotency Key

The `idempotency_key` field ensures that repeated invocations of the same logical operation produce identical results.

**Key Format:**

```
glip-{run_id}-{step_id}-{content_hash}-v{retry_attempt}
```

**Example:**

```
glip-550e8400-e29b-41d4-a716-446655440000-calculate_emissions-sha256:abc123-v0
```

### 7.2 Expected Agent Behavior

#### 7.2.1 First Execution (retry_attempt = 0)

1. Check if results exist for this idempotency key
2. If cached results exist and are valid, return them immediately
3. Otherwise, execute normally and cache results

#### 7.2.2 Retry Execution (retry_attempt > 0)

1. Check if successful results exist for this idempotency key (ignoring retry version)
2. If found, return cached results (idempotent)
3. If previous attempt failed or was partial, re-execute
4. Cache new results with updated retry version

### 7.3 Caching Requirements

**Agents SHOULD implement:**

```python
class IdempotentAgent:
    def run(self, context: RunContext) -> Result:
        # Check cache
        cached = self.cache.get(context.idempotency_key)
        if cached and cached.status == "success":
            return cached

        # Execute
        result = self.execute(context)

        # Cache successful results
        if result.status == "success":
            self.cache.set(
                context.idempotency_key,
                result,
                ttl=86400  # 24 hours
            )

        return result
```

### 7.4 Non-Idempotent Operations

Some operations cannot be made idempotent (e.g., sending emails, external API writes). These agents MUST:

1. Declare `idempotent: false` in their manifest
2. Check retry_attempt and refuse to re-execute if > 0
3. Return error with `retriable: false`

---

## 8. Complete Examples

### 8.1 Complete input.json (RunContext)

```json
{
  "$schema": "https://greenlang.io/schema/glip/run-context.v1.json",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "step_id": "calculate_embedded_emissions",
  "pipeline_id": "cbam-quarterly-report-v2",
  "tenant_id": "org_acme_corp_eu",
  "agent_id": "CBAM-CALC-001",
  "agent_version": "2.1.0",
  "schema_version": "1.0.0",
  "params": {
    "calculation_method": "activity-based",
    "emission_factor_source": "EU-ETS-2024",
    "reporting_period": {
      "start": "2025-10-01",
      "end": "2025-12-31"
    },
    "precision": "high",
    "include_uncertainty": true,
    "product_categories": ["7206", "7207", "7208"],
    "custom_emission_factors": null
  },
  "inputs": [
    {
      "name": "shipment_data",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/data_intake/artifacts/validated_shipments.csv",
      "checksum": {
        "algorithm": "sha256",
        "value": "a3f2b8c9d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
      },
      "size_bytes": 2097152,
      "media_type": "text/csv",
      "schema_ref": "https://greenlang.io/schema/cbam/validated-shipments.v1.json",
      "source_step": "data_intake"
    },
    {
      "name": "supplier_declarations",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/supplier_collection/artifacts/declarations.json",
      "checksum": {
        "algorithm": "sha256",
        "value": "b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3"
      },
      "size_bytes": 524288,
      "media_type": "application/json",
      "schema_ref": "https://greenlang.io/schema/cbam/supplier-declarations.v1.json",
      "source_step": "supplier_collection"
    },
    {
      "name": "emission_factors",
      "uri": "s3://gl-reference-data/emission-factors/eu-ets-2024-q4.json",
      "checksum": {
        "algorithm": "sha256",
        "value": "c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4"
      },
      "size_bytes": 131072,
      "media_type": "application/json",
      "schema_ref": "https://greenlang.io/schema/emission-factors.v2.json",
      "source_step": null
    }
  ],
  "permissions_context": {
    "tenant_id": "org_acme_corp_eu",
    "user_id": "user_maria_schmidt_001",
    "roles": ["pipeline_executor", "cbam_operator", "report_viewer"],
    "scopes": [
      "read:shipments",
      "read:supplier-declarations",
      "read:emission-factors",
      "write:calculations",
      "write:reports"
    ],
    "data_classification": "confidential",
    "allowed_regions": ["eu-west-1", "eu-central-1"],
    "network_policy": {
      "allow_external": true,
      "allowed_hosts": [
        "api.eu-ets.europa.eu",
        "storage.greenlang.io",
        "registry.greenlang.io"
      ]
    },
    "resource_quotas": {
      "max_memory_mb": 4096,
      "max_cpu_millicores": 4000,
      "max_runtime_seconds": 7200
    }
  },
  "deadline_ts": "2026-01-27T16:00:00Z",
  "timeout_s": 7200,
  "retry_attempt": 0,
  "idempotency_key": "glip-550e8400-e29b-41d4-a716-446655440000-calculate_embedded_emissions-sha256:e7f8a9b0-v0",
  "observability": {
    "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
    "span_id": "00f067aa0ba902b7",
    "parent_span_id": "a3ce929d0e0e4736",
    "trace_flags": "01",
    "trace_state": "gl=1,tenant=acme,env=prod",
    "log_correlation_id": "log-550e8400-calculate_embedded_emissions-001",
    "metrics_labels": {
      "pipeline": "cbam-quarterly-report-v2",
      "agent": "CBAM-CALC-001",
      "tenant": "org_acme_corp_eu",
      "environment": "production",
      "region": "eu-west-1"
    },
    "baggage": {
      "user.id": "user_maria_schmidt_001",
      "user.email": "maria.schmidt@acme-corp.eu",
      "request.origin": "scheduled-pipeline",
      "report.quarter": "Q4-2025"
    }
  }
}
```

### 8.2 Complete result.json

```json
{
  "$schema": "https://greenlang.io/schema/glip/result.v1.json",
  "status": "success",
  "agent_id": "CBAM-CALC-001",
  "agent_version": "2.1.0",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "step_id": "calculate_embedded_emissions",
  "started_at": "2026-01-27T14:00:00.000Z",
  "completed_at": "2026-01-27T14:23:45.678Z",
  "outputs": {
    "summary": {
      "total_shipments_processed": 15234,
      "total_goods_mass_tonnes": 45678.90,
      "total_embedded_emissions_tco2e": 84521.34,
      "emissions_intensity_tco2e_per_tonne": 1.85,
      "reporting_period": {
        "start": "2025-10-01",
        "end": "2025-12-31"
      }
    },
    "by_product_category": {
      "7206": {
        "cn_code": "7206",
        "description": "Iron and non-alloy steel in ingots",
        "shipments": 5234,
        "mass_tonnes": 15678.90,
        "embedded_emissions_tco2e": 29000.12,
        "emission_factor_used": 1.85,
        "emission_factor_source": "EU-ETS-2024",
        "data_quality_score": 0.95
      },
      "7207": {
        "cn_code": "7207",
        "description": "Semi-finished products of iron",
        "shipments": 6000,
        "mass_tonnes": 18000.00,
        "embedded_emissions_tco2e": 33300.00,
        "emission_factor_used": 1.85,
        "emission_factor_source": "EU-ETS-2024",
        "data_quality_score": 0.92
      },
      "7208": {
        "cn_code": "7208",
        "description": "Hot-rolled iron and steel",
        "shipments": 4000,
        "mass_tonnes": 12000.00,
        "embedded_emissions_tco2e": 22221.22,
        "emission_factor_used": 1.85,
        "emission_factor_source": "EU-ETS-2024",
        "data_quality_score": 0.89
      }
    },
    "by_origin_country": {
      "CN": {
        "country_code": "CN",
        "country_name": "China",
        "shipments": 8000,
        "mass_tonnes": 24000.00,
        "embedded_emissions_tco2e": 44400.00
      },
      "IN": {
        "country_code": "IN",
        "country_name": "India",
        "shipments": 4234,
        "mass_tonnes": 12678.90,
        "embedded_emissions_tco2e": 23456.78
      },
      "TR": {
        "country_code": "TR",
        "country_name": "Turkey",
        "shipments": 3000,
        "mass_tonnes": 9000.00,
        "embedded_emissions_tco2e": 16664.56
      }
    },
    "data_quality": {
      "overall_score": 0.92,
      "coverage_rate": 0.98,
      "supplier_declaration_rate": 0.75,
      "default_values_used_rate": 0.25,
      "validation_errors": 12,
      "validation_warnings": 45
    },
    "uncertainty": {
      "methodology": "monte_carlo",
      "iterations": 10000,
      "confidence_interval": 0.95,
      "lower_bound_tco2e": 80295.27,
      "upper_bound_tco2e": 88747.41,
      "standard_deviation_tco2e": 2112.04
    },
    "next_steps": {
      "certificate_requirement_tco2e": 84521.34,
      "estimated_certificate_cost_eur": 7184313.90,
      "carbon_price_used_eur_per_tco2e": 85.00
    }
  },
  "artifacts": [
    {
      "name": "detailed_calculations.csv",
      "path": "artifacts/detailed_calculations.csv",
      "description": "Line-by-line emission calculations for all shipments",
      "media_type": "text/csv",
      "size_bytes": 4194304
    },
    {
      "name": "emission_breakdown.json",
      "path": "artifacts/emission_breakdown.json",
      "description": "Structured emission breakdown by category, country, and supplier",
      "media_type": "application/json",
      "size_bytes": 524288
    },
    {
      "name": "audit_trail.jsonl",
      "path": "artifacts/audit_trail.jsonl",
      "description": "Detailed audit log of all calculations and data sources",
      "media_type": "application/jsonl",
      "size_bytes": 1048576
    },
    {
      "name": "data_quality_report.pdf",
      "path": "artifacts/data_quality_report.pdf",
      "description": "Data quality assessment report",
      "media_type": "application/pdf",
      "size_bytes": 262144
    }
  ],
  "warnings": [
    {
      "code": "W001",
      "message": "12 shipments had missing supplier declarations, default emission factors applied",
      "affected_records": 12,
      "severity": "medium"
    },
    {
      "code": "W002",
      "message": "3 suppliers did not provide electricity source data, grid average used",
      "affected_records": 450,
      "severity": "low"
    }
  ],
  "metrics": {
    "processing_rate_records_per_second": 10.7,
    "memory_peak_mb": 2048,
    "cpu_utilization_percent": 75,
    "io_read_mb": 25,
    "io_write_mb": 8
  }
}
```

### 8.3 Complete step_metadata.json

```json
{
  "$schema": "https://greenlang.io/schema/glip/step-metadata.v1.json",
  "schema_version": "1.0.0",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "step_id": "calculate_embedded_emissions",
  "idempotency_key": "glip-550e8400-e29b-41d4-a716-446655440000-calculate_embedded_emissions-sha256:e7f8a9b0-v0",
  "execution": {
    "started_at": "2026-01-27T14:00:00.000Z",
    "completed_at": "2026-01-27T14:23:45.678Z",
    "duration_ms": 1425678,
    "retry_attempt": 0,
    "exit_code": 0,
    "container_id": "glip-cbam-calc-001-xyz789abc",
    "node_id": "k8s-node-eu-west-1a-worker-042",
    "pod_name": "glip-run-550e8400-step-calc-001-x7z9y",
    "namespace": "gl-pipelines-prod"
  },
  "inputs_consumed": [
    {
      "name": "shipment_data",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/data_intake/artifacts/validated_shipments.csv",
      "checksum_verified": true,
      "checksum_algorithm": "sha256",
      "checksum_expected": "a3f2b8c9d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1",
      "checksum_actual": "a3f2b8c9d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1",
      "size_bytes": 2097152,
      "records_processed": 15234,
      "download_duration_ms": 1234
    },
    {
      "name": "supplier_declarations",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/supplier_collection/artifacts/declarations.json",
      "checksum_verified": true,
      "checksum_algorithm": "sha256",
      "checksum_expected": "b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3",
      "checksum_actual": "b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3",
      "size_bytes": 524288,
      "records_processed": 234,
      "download_duration_ms": 567
    },
    {
      "name": "emission_factors",
      "uri": "s3://gl-reference-data/emission-factors/eu-ets-2024-q4.json",
      "checksum_verified": true,
      "checksum_algorithm": "sha256",
      "checksum_expected": "c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4",
      "checksum_actual": "c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4",
      "size_bytes": 131072,
      "records_processed": 450,
      "download_duration_ms": 234
    }
  ],
  "outputs_produced": [
    {
      "name": "result.json",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/calculate_embedded_emissions/output/result.json",
      "checksum": {
        "algorithm": "sha256",
        "value": "e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9"
      },
      "size_bytes": 8192,
      "media_type": "application/json",
      "upload_duration_ms": 456
    },
    {
      "name": "step_metadata.json",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/calculate_embedded_emissions/output/step_metadata.json",
      "checksum": {
        "algorithm": "sha256",
        "value": "f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0"
      },
      "size_bytes": 4096,
      "media_type": "application/json",
      "upload_duration_ms": 234
    },
    {
      "name": "artifacts/detailed_calculations.csv",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/calculate_embedded_emissions/output/artifacts/detailed_calculations.csv",
      "checksum": {
        "algorithm": "sha256",
        "value": "a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1"
      },
      "size_bytes": 4194304,
      "media_type": "text/csv",
      "upload_duration_ms": 2345
    },
    {
      "name": "artifacts/emission_breakdown.json",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/calculate_embedded_emissions/output/artifacts/emission_breakdown.json",
      "checksum": {
        "algorithm": "sha256",
        "value": "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2"
      },
      "size_bytes": 524288,
      "media_type": "application/json",
      "upload_duration_ms": 567
    },
    {
      "name": "artifacts/audit_trail.jsonl",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/calculate_embedded_emissions/output/artifacts/audit_trail.jsonl",
      "checksum": {
        "algorithm": "sha256",
        "value": "c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3"
      },
      "size_bytes": 1048576,
      "media_type": "application/jsonl",
      "upload_duration_ms": 890
    },
    {
      "name": "artifacts/data_quality_report.pdf",
      "uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/calculate_embedded_emissions/output/artifacts/data_quality_report.pdf",
      "checksum": {
        "algorithm": "sha256",
        "value": "d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4"
      },
      "size_bytes": 262144,
      "media_type": "application/pdf",
      "upload_duration_ms": 345
    }
  ],
  "provenance": {
    "agent_image": "greenlang/cbam-calc:2.1.0@sha256:abc123def456789abc123def456789abc123def456789abc123def456789abcd",
    "agent_id": "CBAM-CALC-001",
    "agent_version": "2.1.0",
    "glip_schema_version": "1.0.0",
    "base_image": "greenlang/agent-base:1.0.0@sha256:def456abc789def456abc789def456abc789def456abc789def456abc789defg",
    "build_timestamp": "2026-01-15T10:30:00Z",
    "git_repository": "https://github.com/greenlang/agents-cbam",
    "git_commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
    "git_branch": "release/v2.1.0",
    "ci_build_id": "build-12345",
    "dependencies_hash": "sha256:def456abc789def456abc789def456abc789def456abc789def456abc789defg",
    "sbom_uri": "s3://gl-sbom/cbam-calc-2.1.0.spdx.json"
  },
  "resource_usage": {
    "cpu_seconds": 1068.75,
    "cpu_utilization_avg_percent": 75,
    "cpu_utilization_max_percent": 98,
    "memory_peak_mb": 2048,
    "memory_avg_mb": 1536,
    "memory_limit_mb": 4096,
    "network_rx_bytes": 2752512,
    "network_tx_bytes": 5242880,
    "disk_read_bytes": 2752512,
    "disk_write_bytes": 6029312,
    "disk_iops_read": 1234,
    "disk_iops_write": 567
  },
  "audit": {
    "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
    "span_id": "00f067aa0ba902b7",
    "parent_span_id": "a3ce929d0e0e4736",
    "tenant_id": "org_acme_corp_eu",
    "user_id": "user_maria_schmidt_001",
    "pipeline_id": "cbam-quarterly-report-v2",
    "pipeline_version": "2.3.1",
    "deterministic": true,
    "cached_result": false,
    "idempotency_key_used": "glip-550e8400-e29b-41d4-a716-446655440000-calculate_embedded_emissions-sha256:e7f8a9b0-v0",
    "permissions_validated": true,
    "signature": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImdsLXNpZ24tMjAyNi0wMSJ9.eyJydW5faWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAiLCJzdGVwX2lkIjoiY2FsY3VsYXRlX2VtYmVkZGVkX2VtaXNzaW9ucyIsImhhc2giOiJzaGEyNTY6YWJjMTIzIn0.signature",
    "signature_algorithm": "ES256",
    "signing_key_id": "gl-sign-2026-01"
  }
}
```

### 8.4 Complete error.json

```json
{
  "$schema": "https://greenlang.io/schema/glip/error.v1.json",
  "error_code": "GLIP-E002-001",
  "error_type": "INPUT_VALIDATION_ERROR",
  "message": "Shipment data file failed schema validation: missing required field 'supplier_id' in 45 records",
  "details": {
    "input_name": "shipment_data",
    "input_uri": "s3://gl-artifacts/org_acme_corp_eu/runs/550e8400-e29b-41d4-a716-446655440000/steps/data_intake/artifacts/validated_shipments.csv",
    "schema_ref": "https://greenlang.io/schema/cbam/validated-shipments.v1.json",
    "validation_errors": [
      {
        "row": 1234,
        "field": "supplier_id",
        "error": "required field missing",
        "value": null
      },
      {
        "row": 2345,
        "field": "supplier_id",
        "error": "required field missing",
        "value": null
      }
    ],
    "total_validation_errors": 45,
    "sample_shown": 2
  },
  "stack_trace": "Traceback (most recent call last):\n  File \"/app/agent/main.py\", line 145, in run\n    validated_data = self.validate_inputs(context)\n  File \"/app/agent/validators.py\", line 78, in validate_inputs\n    raise InputValidationError(errors)\ngreenlang.exceptions.InputValidationError: 45 validation errors in shipment_data",
  "retriable": false,
  "suggested_action": "Review the shipment data file and ensure all records have the required 'supplier_id' field populated. Re-run the data_intake step with corrected data.",
  "documentation_url": "https://docs.greenlang.io/agents/cbam-calc/errors/GLIP-E002-001",
  "occurred_at": "2026-01-27T14:00:15.234Z",
  "agent_id": "CBAM-CALC-001",
  "agent_version": "2.1.0",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "step_id": "calculate_embedded_emissions"
}
```

---

## 9. JSON Schemas

### 9.1 Schema Locations

| Schema | Location |
|--------|----------|
| RunContext | `https://greenlang.io/schema/glip/run-context.v1.json` |
| Result | `https://greenlang.io/schema/glip/result.v1.json` |
| Step Metadata | `https://greenlang.io/schema/glip/step-metadata.v1.json` |
| Error | `https://greenlang.io/schema/glip/error.v1.json` |

### 9.2 RunContext JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://greenlang.io/schema/glip/run-context.v1.json",
  "title": "GLIP v1 RunContext Schema",
  "description": "Input envelope schema for GreenLang agent invocation",
  "type": "object",
  "required": [
    "run_id",
    "step_id",
    "pipeline_id",
    "tenant_id",
    "agent_id",
    "agent_version",
    "schema_version",
    "params",
    "inputs",
    "permissions_context",
    "deadline_ts",
    "timeout_s",
    "retry_attempt",
    "idempotency_key",
    "observability"
  ],
  "properties": {
    "run_id": {
      "type": "string",
      "format": "uuid",
      "description": "Globally unique identifier for this pipeline run"
    },
    "step_id": {
      "type": "string",
      "pattern": "^[a-z][a-z0-9_-]{0,62}$",
      "description": "Unique identifier for this step within the pipeline"
    },
    "pipeline_id": {
      "type": "string",
      "pattern": "^[a-z][a-z0-9-]{0,62}$",
      "description": "Identifier of the pipeline definition"
    },
    "tenant_id": {
      "type": "string",
      "pattern": "^[a-z][a-z0-9_-]{0,62}$",
      "description": "Tenant/organization identifier"
    },
    "agent_id": {
      "type": "string",
      "pattern": "^[A-Z][A-Z0-9-]{0,30}$",
      "description": "Agent identifier"
    },
    "agent_version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+(-[a-zA-Z0-9.]+)?$",
      "description": "SemVer agent version"
    },
    "schema_version": {
      "type": "string",
      "const": "1.0.0",
      "description": "GLIP schema version"
    },
    "params": {
      "type": "object",
      "description": "Agent-specific parameters"
    },
    "inputs": {
      "type": "array",
      "items": {
        "$ref": "#/$defs/InputArtifact"
      },
      "description": "Input artifacts from upstream steps"
    },
    "permissions_context": {
      "$ref": "#/$defs/PermissionsContext",
      "description": "Authorization context"
    },
    "deadline_ts": {
      "type": "string",
      "format": "date-time",
      "description": "Deadline for execution completion"
    },
    "timeout_s": {
      "type": "integer",
      "minimum": 1,
      "maximum": 86400,
      "description": "Maximum execution time in seconds"
    },
    "retry_attempt": {
      "type": "integer",
      "minimum": 0,
      "description": "Current retry attempt number"
    },
    "idempotency_key": {
      "type": "string",
      "minLength": 1,
      "maxLength": 256,
      "description": "Key for idempotent execution"
    },
    "observability": {
      "$ref": "#/$defs/ObservabilityContext",
      "description": "Tracing and logging context"
    }
  },
  "$defs": {
    "InputArtifact": {
      "type": "object",
      "required": ["name", "uri", "checksum"],
      "properties": {
        "name": {
          "type": "string",
          "pattern": "^[a-z][a-z0-9_-]{0,62}$"
        },
        "uri": {
          "type": "string",
          "format": "uri"
        },
        "checksum": {
          "$ref": "#/$defs/Checksum"
        },
        "size_bytes": {
          "type": "integer",
          "minimum": 0
        },
        "media_type": {
          "type": "string"
        },
        "schema_ref": {
          "type": "string",
          "format": "uri"
        },
        "source_step": {
          "type": ["string", "null"]
        }
      }
    },
    "Checksum": {
      "type": "object",
      "required": ["algorithm", "value"],
      "properties": {
        "algorithm": {
          "type": "string",
          "enum": ["sha256", "sha384", "sha512"]
        },
        "value": {
          "type": "string",
          "pattern": "^[a-f0-9]{64,128}$"
        }
      }
    },
    "PermissionsContext": {
      "type": "object",
      "required": ["tenant_id", "scopes"],
      "properties": {
        "tenant_id": { "type": "string" },
        "user_id": { "type": "string" },
        "roles": {
          "type": "array",
          "items": { "type": "string" }
        },
        "scopes": {
          "type": "array",
          "items": { "type": "string" }
        },
        "data_classification": { "type": "string" },
        "allowed_regions": {
          "type": "array",
          "items": { "type": "string" }
        },
        "network_policy": {
          "type": "object",
          "properties": {
            "allow_external": { "type": "boolean" },
            "allowed_hosts": {
              "type": "array",
              "items": { "type": "string" }
            }
          }
        },
        "resource_quotas": {
          "type": "object",
          "properties": {
            "max_memory_mb": { "type": "integer" },
            "max_cpu_millicores": { "type": "integer" },
            "max_runtime_seconds": { "type": "integer" }
          }
        }
      }
    },
    "ObservabilityContext": {
      "type": "object",
      "required": ["trace_id", "span_id", "log_correlation_id"],
      "properties": {
        "trace_id": {
          "type": "string",
          "pattern": "^[a-f0-9]{32}$"
        },
        "span_id": {
          "type": "string",
          "pattern": "^[a-f0-9]{16}$"
        },
        "parent_span_id": {
          "type": "string",
          "pattern": "^[a-f0-9]{16}$"
        },
        "trace_flags": { "type": "string" },
        "trace_state": { "type": "string" },
        "log_correlation_id": { "type": "string" },
        "metrics_labels": {
          "type": "object",
          "additionalProperties": { "type": "string" }
        },
        "baggage": {
          "type": "object",
          "additionalProperties": { "type": "string" }
        }
      }
    }
  }
}
```

---

## 10. Security Considerations

### 10.1 Authentication and Authorization

- All artifact URIs MUST use signed URLs or authenticated storage access
- Agents MUST validate `permissions_context` before accessing resources
- Network access MUST be restricted to `allowed_hosts` in permissions

### 10.2 Data Protection

- Input artifacts MUST be verified against provided checksums
- Output artifacts MUST include checksums in `step_metadata.json`
- Sensitive data MUST be encrypted at rest and in transit
- Agents MUST respect `data_classification` levels

### 10.3 Container Security

- Agent containers SHOULD run as non-root users
- Agent containers MUST NOT require privileged mode
- Agent containers SHOULD use read-only root filesystems
- Agent images MUST be scanned for vulnerabilities

### 10.4 Audit Trail

- All executions MUST generate signed `step_metadata.json`
- All inputs and outputs MUST be logged with checksums
- Access to audit logs MUST be restricted and immutable

---

## 11. Conformance

### 11.1 Conformance Levels

| Level | Requirements |
|-------|--------------|
| **GLIP v1 Basic** | Sections 3, 4, 5, 6 |
| **GLIP v1 Standard** | Basic + Section 7 (Idempotency) |
| **GLIP v1 Full** | Standard + Section 10 (Security) |

### 11.2 Compliance Testing

Agents can be tested for GLIP v1 compliance using:

```bash
# Test GLIP v1 compliance
gl-agent test --compliance glip-v1 --level full

# Validate RunContext
gl-agent validate --schema run-context input.json

# Validate outputs
gl-agent validate --schema result result.json
gl-agent validate --schema step-metadata step_metadata.json
```

### 11.3 Certification

GreenLang provides certification for GLIP v1 compliant agents:

1. Submit agent for automated compliance testing
2. Pass all required tests for desired conformance level
3. Receive GLIP v1 certification badge
4. Agent listed in GreenLang Hub as certified

---

## 12. Changelog

### Version 1.0.0 (2026-01-27)

- Initial release of GLIP v1 specification
- Defined agent container packaging requirements
- Specified RunContext input envelope schema
- Defined output format (result.json, step_metadata.json, artifacts/)
- Established exit code semantics
- Defined idempotency requirements
- Added complete JSON schemas
- Included security considerations

---

## Appendix A: Quick Reference

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GL_INPUT_URI` | URI to fetch RunContext JSON |
| `GL_OUTPUT_URI` | Base URI to write outputs |

### Required Container Labels

| Label | Example |
|-------|---------|
| `gl.agent.id` | `CBAM-CALC-001` |
| `gl.agent.version` | `2.1.0` |
| `gl.schema.version` | `1.0.0` |

### Required Output Files

| File | Description |
|------|-------------|
| `result.json` | Agent execution results |
| `step_metadata.json` | Execution metadata |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General failure |
| 2 | Input validation failure |
| 3 | Checksum failure |
| 4 | Permission failure |
| 5 | Timeout |
| 6 | Resource exhaustion |
| 7 | Output validation failure |
| 8 | Upstream failure |

---

**Document Control:**

| Item | Value |
|------|-------|
| Document ID | GL-FOUND-X-001-SPEC-001 |
| Version | 1.0.0 |
| Status | DRAFT |
| Classification | Public |
| Owner | GreenLang Platform Team |
| Review Cycle | Quarterly |

---

*End of GLIP v1 Specification*
