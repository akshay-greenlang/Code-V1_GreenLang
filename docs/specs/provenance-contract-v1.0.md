# GreenLang Provenance Contract v1.0

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-02-03

## Overview

The Provenance Contract specifies what information MUST be recorded for audit purposes in GreenLang pipeline executions. This ensures that every climate calculation can be traced back to its inputs, methods, and data sources.

**Core Principle:**
> Every number must have a story. Auditors should be able to trace any output back to its origins.

## Provenance Record Structure

### Complete Provenance Record

```json
{
  "provenance_version": "1.0",
  "record_id": "prov_abc123",
  "run_id": "run_xyz789",

  "run_metadata": {
    "pipeline_name": "emissions-calculation",
    "pipeline_version": "1.0.0",
    "pipeline_hash": "sha256:abc...",
    "started_at": "2026-02-03T12:00:00Z",
    "completed_at": "2026-02-03T12:05:00Z",
    "executor": "k8s",
    "executor_version": "1.2.0",
    "environment": {
      "python_version": "3.10.12",
      "greenlang_version": "0.3.0",
      "os": "linux",
      "locale": "en_US.UTF-8"
    }
  },

  "input_lineage": {
    "sources": [
      {
        "source_id": "src_001",
        "type": "file",
        "path": "/data/input.json",
        "hash": "sha256:def...",
        "size_bytes": 1024,
        "created_at": "2026-02-01T10:00:00Z"
      }
    ],
    "parameters": {
      "param1": "value1",
      "param2": 42
    },
    "parameter_hash": "sha256:ghi..."
  },

  "execution_trace": {
    "steps": [
      {
        "step_id": "normalize",
        "agent": "data-normalizer",
        "agent_version": "1.0.0",
        "started_at": "2026-02-03T12:00:01Z",
        "completed_at": "2026-02-03T12:00:05Z",
        "duration_ms": 4000,
        "inputs_hash": "sha256:jkl...",
        "outputs_hash": "sha256:mno...",
        "tool_calls": []
      }
    ]
  },

  "factor_citations": [
    {
      "factor_id": "DEFRA-2024-diesel",
      "source": "DEFRA",
      "vintage": 2024,
      "value": "2.6572",
      "unit": "kg_CO2e/liter",
      "methodology": "GHG Protocol Scope 1",
      "uncertainty": "5%",
      "citation": "DEFRA GHG Conversion Factors 2024, Table 1"
    }
  ],

  "llm_interactions": [],

  "approval_records": [],

  "output_artifacts": {
    "primary": {
      "path": "outputs/result.json",
      "hash": "sha256:pqr...",
      "size_bytes": 512
    },
    "all": [
      {
        "name": "result.json",
        "path": "outputs/result.json",
        "hash": "sha256:pqr...",
        "type": "application/json"
      }
    ]
  },

  "integrity": {
    "record_hash": "sha256:stu...",
    "chain_hash": "sha256:vwx...",
    "signature": null
  }
}
```

## Required Provenance Elements

### 1. Run Metadata (REQUIRED)

Every run MUST record:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique run identifier |
| `pipeline_name` | string | Pipeline identifier |
| `pipeline_version` | string | Pipeline semver |
| `pipeline_hash` | string | SHA-256 of pipeline definition |
| `started_at` | ISO8601 | UTC start timestamp |
| `completed_at` | ISO8601 | UTC completion timestamp |
| `executor` | string | Execution backend (local, k8s) |
| `greenlang_version` | string | GreenLang version used |

**Implementation:**

```python
def create_run_metadata(pipeline, executor):
    return {
        "run_id": generate_run_id(),
        "pipeline_name": pipeline.name,
        "pipeline_version": pipeline.version,
        "pipeline_hash": compute_pipeline_hash(pipeline),
        "started_at": datetime.utcnow().isoformat() + "Z",
        "completed_at": None,  # Set on completion
        "executor": executor.name,
        "executor_version": executor.version,
        "environment": {
            "python_version": sys.version.split()[0],
            "greenlang_version": greenlang.__version__,
            "os": platform.system().lower(),
            "locale": locale.getlocale()[0]
        }
    }
```

### 2. Input Lineage (REQUIRED)

Every input MUST be traceable:

| Field | Type | Description |
|-------|------|-------------|
| `sources` | array | Input data sources |
| `sources[].type` | string | file, api, database, user_input |
| `sources[].path` | string | Location identifier |
| `sources[].hash` | string | SHA-256 of content |
| `sources[].created_at` | ISO8601 | Source creation time |
| `parameters` | object | Run parameters |
| `parameter_hash` | string | Hash of parameters |

**Implementation:**

```python
def record_input_lineage(inputs: dict, sources: list) -> dict:
    return {
        "sources": [
            {
                "source_id": f"src_{i:03d}",
                "type": source["type"],
                "path": source["path"],
                "hash": compute_file_hash(source["path"]) if source["type"] == "file" else None,
                "size_bytes": get_size(source["path"]) if source["type"] == "file" else None,
                "created_at": get_created_time(source["path"]) if source["type"] == "file" else None
            }
            for i, source in enumerate(sources)
        ],
        "parameters": sanitize_parameters(inputs),
        "parameter_hash": compute_hash(inputs)
    }
```

### 3. Execution Trace (REQUIRED)

Every step MUST record:

| Field | Type | Description |
|-------|------|-------------|
| `step_id` | string | Step identifier |
| `agent` | string | Agent identifier |
| `agent_version` | string | Agent version |
| `started_at` | ISO8601 | Step start time |
| `completed_at` | ISO8601 | Step end time |
| `duration_ms` | number | Execution duration |
| `inputs_hash` | string | Hash of step inputs |
| `outputs_hash` | string | Hash of step outputs |
| `tool_calls` | array | External tool invocations |

**Implementation:**

```python
def record_step_execution(step, inputs, outputs, duration):
    return {
        "step_id": step.name,
        "agent": step.agent_id,
        "agent_version": get_agent_version(step.agent_id),
        "started_at": step.started_at.isoformat() + "Z",
        "completed_at": step.completed_at.isoformat() + "Z",
        "duration_ms": int(duration * 1000),
        "inputs_hash": compute_hash(inputs),
        "outputs_hash": compute_hash(outputs),
        "tool_calls": step.tool_calls or []
    }
```

### 4. Factor Citations (REQUIRED for calculations)

Every emission factor MUST cite:

| Field | Type | Description |
|-------|------|-------------|
| `factor_id` | string | Unique factor identifier |
| `source` | string | Data source (DEFRA, EPA, IPCC) |
| `vintage` | number | Year of publication |
| `value` | string | Factor value (as string for precision) |
| `unit` | string | Factor unit |
| `methodology` | string | Calculation methodology |
| `uncertainty` | string | Uncertainty range |
| `citation` | string | Full citation |

**Implementation:**

```python
def record_factor_citation(factor):
    return {
        "factor_id": factor.id,
        "source": factor.source,
        "vintage": factor.year,
        "value": str(factor.value),  # String for precision
        "unit": factor.unit,
        "methodology": factor.methodology,
        "uncertainty": factor.uncertainty,
        "citation": factor.full_citation
    }
```

### 5. LLM Interactions (REQUIRED when AI is used)

If LLM is invoked, record:

| Field | Type | Description |
|-------|------|-------------|
| `interaction_id` | string | Unique interaction ID |
| `step_id` | string | Step that invoked LLM |
| `timestamp` | ISO8601 | Invocation time |
| `model.provider` | string | OpenAI, Anthropic, etc. |
| `model.model_id` | string | Specific model used |
| `model.api_version` | string | API version |
| `request.prompt_hash` | string | Hash of prompt |
| `request.temperature` | number | Temperature setting |
| `request.seed` | number | Seed if used |
| `response.output_hash` | string | Hash of response |
| `response.token_count` | number | Tokens used |
| `purpose` | string | Why LLM was used |

**Implementation:**

```python
def record_llm_interaction(step_id, request, response, model_info):
    return {
        "interaction_id": generate_interaction_id(),
        "step_id": step_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": {
            "provider": model_info["provider"],
            "model_id": model_info["model_id"],
            "api_version": model_info.get("api_version")
        },
        "request": {
            "prompt_hash": compute_hash(request["prompt"]),
            "temperature": request.get("temperature", 0),
            "seed": request.get("seed"),
            "max_tokens": request.get("max_tokens")
        },
        "response": {
            "output_hash": compute_hash(response["text"]),
            "token_count": response["usage"]["total_tokens"],
            "finish_reason": response.get("finish_reason")
        },
        "purpose": request.get("purpose", "unspecified")
    }
```

### 6. Approval Records (REQUIRED for human-in-the-loop)

If human approval was required, record:

| Field | Type | Description |
|-------|------|-------------|
| `approval_id` | string | Unique approval ID |
| `step_id` | string | Step requiring approval |
| `requested_at` | ISO8601 | When approval requested |
| `approved_at` | ISO8601 | When approved |
| `approver` | string | Who approved |
| `decision` | string | approved, rejected |
| `reason` | string | Approval reason |

### 7. Output Artifacts (REQUIRED)

All outputs MUST be recorded:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Artifact name |
| `path` | string | Storage path |
| `hash` | string | SHA-256 of content |
| `type` | string | MIME type |
| `size_bytes` | number | File size |

## Provenance Integrity

### Hash Chain

Provenance records form a hash chain for tamper detection:

```python
def compute_chain_hash(current_record, previous_hash):
    """Compute hash chain linking provenance records."""
    data = {
        "previous_hash": previous_hash,
        "record_hash": current_record["integrity"]["record_hash"],
        "run_id": current_record["run_id"],
        "timestamp": current_record["run_metadata"]["completed_at"]
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
```

### Record Hash

Each record has its own integrity hash:

```python
def compute_record_hash(record):
    """Compute hash of provenance record (excluding integrity section)."""
    record_copy = copy.deepcopy(record)
    record_copy.pop("integrity", None)
    return hashlib.sha256(
        json.dumps(record_copy, sort_keys=True).encode()
    ).hexdigest()
```

### Signature (Optional)

Records can be signed for non-repudiation:

```python
def sign_provenance_record(record, private_key):
    """Sign provenance record with private key."""
    record_hash = record["integrity"]["record_hash"]
    signature = private_key.sign(record_hash.encode())
    return base64.b64encode(signature).decode()
```

## Retention Requirements

| Record Type | Minimum Retention | Regulation |
|-------------|-------------------|------------|
| Run metadata | 7 years | SOX, CSRD |
| Input lineage | 7 years | Audit trail |
| Factor citations | 10 years | IPCC methodology |
| LLM interactions | 7 years | AI audit |
| Output artifacts | 7 years | Compliance |

## Verification Commands

```bash
# Verify provenance chain
gl verify --provenance run.json

# Audit factor citations
gl verify --factors run.json

# Check LLM usage compliance
gl verify --llm-audit run.json

# Export audit report
gl verify --export-audit run.json > audit_report.json
```

## Implementation Example

```python
class ProvenanceRecorder:
    """Record provenance for pipeline execution."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.record = {
            "provenance_version": "1.0",
            "record_id": f"prov_{uuid.uuid4().hex[:12]}",
            "run_id": run_id,
            "run_metadata": {},
            "input_lineage": {},
            "execution_trace": {"steps": []},
            "factor_citations": [],
            "llm_interactions": [],
            "approval_records": [],
            "output_artifacts": {"all": []},
            "integrity": {}
        }

    def start_run(self, pipeline, executor):
        self.record["run_metadata"] = create_run_metadata(pipeline, executor)

    def record_inputs(self, inputs, sources):
        self.record["input_lineage"] = record_input_lineage(inputs, sources)

    def record_step(self, step, inputs, outputs, duration):
        self.record["execution_trace"]["steps"].append(
            record_step_execution(step, inputs, outputs, duration)
        )

    def cite_factor(self, factor):
        self.record["factor_citations"].append(
            record_factor_citation(factor)
        )

    def record_llm(self, step_id, request, response, model_info):
        self.record["llm_interactions"].append(
            record_llm_interaction(step_id, request, response, model_info)
        )

    def finalize(self, previous_hash: str = None):
        self.record["run_metadata"]["completed_at"] = (
            datetime.utcnow().isoformat() + "Z"
        )
        self.record["integrity"]["record_hash"] = compute_record_hash(self.record)
        self.record["integrity"]["chain_hash"] = compute_chain_hash(
            self.record, previous_hash or "genesis"
        )
        return self.record
```

## Appendix A: Provenance Query Examples

```python
# Find all runs using a specific factor
runs = query_provenance(
    filter={"factor_citations.factor_id": "DEFRA-2024-diesel"}
)

# Find runs with LLM usage
runs = query_provenance(
    filter={"llm_interactions": {"$exists": True, "$ne": []}}
)

# Find runs by time range
runs = query_provenance(
    filter={
        "run_metadata.started_at": {
            "$gte": "2026-01-01T00:00:00Z",
            "$lt": "2026-02-01T00:00:00Z"
        }
    }
)
```

## Appendix B: Audit Report Format

```json
{
  "audit_report_version": "1.0",
  "generated_at": "2026-02-03T12:00:00Z",
  "run_id": "run_xyz789",
  "summary": {
    "total_steps": 5,
    "factors_used": 3,
    "llm_invocations": 1,
    "human_approvals": 0,
    "integrity_verified": true
  },
  "factor_audit": [
    {
      "factor_id": "DEFRA-2024-diesel",
      "used_in_steps": ["calculate-emissions"],
      "citation_complete": true,
      "vintage_valid": true
    }
  ],
  "llm_audit": [
    {
      "interaction_id": "llm_001",
      "purpose": "summarization",
      "used_for_calculation": false,
      "audit_status": "compliant"
    }
  ],
  "chain_verification": {
    "verified": true,
    "chain_length": 15,
    "first_record": "prov_abc123",
    "last_record": "prov_xyz789"
  }
}
```
