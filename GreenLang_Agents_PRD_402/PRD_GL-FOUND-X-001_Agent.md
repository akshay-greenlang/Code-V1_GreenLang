# PRD: GreenLang Orchestrator (GL-FOUND-X-001)

**Agent family:** OrchestrationFamily  
**Layer:** Foundation & Governance  
**Primary domains:** Platform runtime, pipeline orchestration (cross-cutting)  
**Priority:** P0 (highest)  
**Doc version:** 1.0  
**Last updated:** 2026-01-27 (Asia/Kolkata)

---

## 1. Executive summary

**GreenLang Orchestrator (GL-FOUND-X-001)** is the **foundation & governance layer agent** responsible for **planning and executing multi-agent pipelines** in GreenLang. It compiles a Pipeline YAML into a dependency graph (DAG), produces a deterministic execution plan, executes steps with policy enforcement, and emits **auditable run metadata**, **run logs**, **step artifacts**, and **end-to-end lineage**.

This agent is the “control plane” for GreenLang pipelines:
- Turns pipeline intent into an **execution plan** with stable, deterministic identifiers.
- Executes the plan with **retries, timeouts, cancellation, and handoffs**.
- Enforces governance: **permissions, policy checks, and audit trail**.
- Provides platform-grade observability: **logs, metrics, traces**, and step-level artifacts.

---

## 2. Problem statement

As GreenLang expands across multiple sectors/domains, we need a single orchestration layer that:
1. **Standardizes** how multi-agent pipelines are defined and executed.
2. Provides **reliable execution** (retries/timeouts/handoffs) and **safe operations** (policy checks).
3. Guarantees **deterministic run metadata** and **auditability** for governance and compliance.

Without a dedicated foundation orchestrator:
- Pipelines become fragmented across teams and tools.
- Governance becomes inconsistent, causing security/compliance risk.
- Runs are difficult to reproduce, troubleshoot, or audit.

---

## 3. Goals and non-goals

### 3.1 Goals (must deliver)

1. **Pipeline compilation and planning**
   - Parse and validate Pipeline YAML.
   - Resolve agents/steps from an agent registry.
   - Produce a **deterministic execution plan** and step graph.

2. **Execution and resilience**
   - Execute DAG steps with dependency management.
   - Support **retries, exponential backoff, timeouts**, and **cancellation**.
   - Manage **handoffs** to OPS+DATA agents and other domain agents.

3. **Governance and auditability**
   - Enforce permissions/credentials/least-privilege.
   - Run **policy checks** pre-run and at step boundaries.
   - Produce deterministic run metadata suitable for **audit trails**.

4. **Observability and lineage**
   - Emit structured run logs, metrics, traces.
   - Capture step-level artifacts and establish input->output lineage.

### 3.2 Non-goals (explicitly out of scope for GL-FOUND-X-001)

- Performing heavy data processing itself (delegated to DATA agents).
- Implementing model training/fine-tuning.
- Being the UI/portal (but must expose APIs for UI).
- Replacing existing infra schedulers; instead provide **pluggable execution backends**.

---

## 4. Stakeholders and users

### 4.1 Primary stakeholders
- **Platform Engineering:** owns runtime, orchestration reliability, scaling.
- **Governance / Risk / Compliance:** audit trail, policy enforcement, reproducibility.
- **SRE / Ops:** on-call, incident response, run health, capacity.
- **Data Engineering & MLOps:** pipelines that invoke OPS+DATA agents.
- **Security:** RBAC/ABAC, secrets, boundary enforcement.

### 4.2 Primary user personas
1. **Pipeline Author (Engineer):** writes Pipeline YAML, triggers runs, inspects outcomes.
2. **Operator (SRE/Ops):** watches run health, diagnoses failures, retries/cancels runs.
3. **Auditor / Governance analyst:** validates that a run is reproducible and policy-compliant.
4. **Agent Developer:** registers agents, declares inputs/outputs/permissions/capabilities.

---

## 5. Definitions and glossary

| Term | Definition |
|---|---|
| **Pipeline YAML** | Declarative pipeline specification in YAML (steps, dependencies, configs). |
| **Step** | A node in the DAG. Executes an agent task or control-flow primitive. |
| **DAG** | Directed acyclic graph representing step dependencies. |
| **Execution Plan** | Compiled, deterministic plan derived from pipeline YAML + run config + registry snapshot. |
| **Run Metadata** | Immutable record of run inputs, versions, environment, plan hash, and emitted artifacts. |
| **Lineage** | Traceable mapping from inputs -> intermediate artifacts -> outputs. |
| **Policy Check** | A governance rule evaluated pre-run or at step boundaries (e.g., access, residency, cost). |
| **Artifact** | Step outputs (files, JSON, datasets, model outputs, reports) stored with metadata. |

---

## 6. Assumptions and constraints

### 6.1 Assumptions
- GreenLang maintains an **Agent Registry** that can provide:
  - agent identity/version, capabilities, interfaces, permissions needed, resource hints.
- A **central artifact store** exists (object storage or equivalent) plus a metadata DB.
- Authentication exists (OIDC/JWT/service tokens) for user and service identities.

### 6.2 Constraints
- Must support **cross-cutting pipelines** across sectors/domains.
- Must be able to operate in **multi-tenant** setups (namespaces/projects).
- Auditability requires an **append-only event model** and stable identifiers.

---

## 7. High-level requirements

### 7.1 Inputs (required)
- **Pipeline YAML** (pipeline definition)
- **Agent registry snapshot** (agent versions, endpoints, contracts)
- **Run configuration** (params, environment, run mode, budgets, priority)
- **Credentials/permissions** (identity, secrets references, scoped tokens)

### 7.2 Outputs
- **Execution plan** (DAG + ordered schedule + resolved agent bindings)
- **Run logs** (structured, queryable)
- **Step-level artifacts** (object-store + metadata)
- **Status & lineage** (run + step status, dependency relationships, provenance)

### 7.3 Key methods/tools
- DAG orchestration engine
- Policy checks (OPA/Rego or equivalent policy engine)
- Observability hooks (OpenTelemetry compatible)

### 7.4 Dependencies
- OPS+DATA agents (execution targets)
- Central audit trail / event log system

---

## 8. Pipeline YAML specification (v1)

### 8.1 Design principles
- Declarative and human-readable.
- Explicit dependencies.
- Deterministic compilation (stable ordering rules).
- Extensible (versioned schema).

### 8.2 Minimal schema (conceptual)

```yaml
apiVersion: greenlang/v1
kind: Pipeline
metadata:
  name: example-pipeline
  namespace: demo
spec:
  parameters:
    input_uri:
      type: string
      required: true
  defaults:
    retries: 2
    timeoutSeconds: 900
  steps:
    - id: ingest
      agent: OPS.DATA.Ingest
      with:
        uri: "{{ params.input_uri }}"
      outputs:
        dataset: "$.artifact.dataset_uri"

    - id: validate
      dependsOn: [ingest]
      agent: OPS.DATA.QualityCheck
      with:
        dataset_uri: "{{ steps.ingest.outputs.dataset }}"
      policy:
        - name: no_pii_export
          severity: error

    - id: publish
      dependsOn: [validate]
      agent: OPS.DATA.Publish
      with:
        dataset_uri: "{{ steps.ingest.outputs.dataset }}"
```

### 8.3 Required validations
- YAML schema validation (apiVersion/kind/required fields).
- Unique step IDs.
- DAG acyclicity.
- Agent existence and version resolution.
- Parameter template validation (no unknown refs).
- Policy references exist and are allowed for the namespace.

---

## 9. Deterministic plan and run metadata

### 9.1 Determinism requirements (core)
Given the same:
- normalized Pipeline YAML (whitespace/ordering-insensitive normalization),
- run configuration,
- agent registry snapshot (versions + contracts),
- policy bundle version,
- execution backend version,

GL-FOUND-X-001 must produce:
- the same **plan hash**,
- the same **step identifiers**,
- the same execution graph topology,
- and reproducible, auditable metadata.

### 9.2 Deterministic identifiers
- `run_id` MUST be content-addressable (e.g., hash of normalized inputs) OR include a random nonce but then also store a stable `plan_id`.
- `step_id` MUST be stable within a run and derived from (pipeline_step_id + plan_hash).

### 9.3 Metadata to capture (minimum)
- pipeline: original YAML, normalized YAML hash, schema version
- run config: params, triggers, budgets, priority
- identity: submitter, service account, approvals (if any)
- agent registry snapshot hash
- policy bundle hash and evaluation results
- execution backend details
- timestamps for every state transition
- artifacts: URIs + checksums + producers + consumers
- logs: pointers to structured logs, traces

---

## 10. Functional requirements

> **Priority legend:** P0 = must-have (MVP/GA critical), P1 = should-have, P2 = nice-to-have.

### 10.1 Pipeline intake and compilation
- **FR-001 (P0):** Accept Pipeline YAML via API/CLI and store immutable original.
- **FR-002 (P0):** Validate YAML against versioned schema; reject invalid pipelines with actionable errors.
- **FR-003 (P0):** Compile steps into a DAG and verify acyclicity.
- **FR-004 (P0):** Resolve agent bindings from registry (including version pinning rules).
- **FR-005 (P1):** Support reusable pipeline templates/modules (import/include) with deterministic expansion.

### 10.2 Planning
- **FR-010 (P0):** Produce an execution plan containing:
  - resolved DAG, stable step ordering rules, resource hints, policies, and handoff contracts.
- **FR-011 (P0):** Produce deterministic plan hash and record it in metadata.
- **FR-012 (P1):** Support “dry-run” plan output without execution.

### 10.3 Execution engine (DAG orchestration)
- **FR-020 (P0):** Execute steps when dependencies are satisfied.
- **FR-021 (P0):** Support step retries with policy-defined retry budget and backoff.
- **FR-022 (P0):** Support per-step and per-run timeouts; mark terminal states correctly.
- **FR-023 (P0):** Support cancellation; ensure downstream steps do not start after cancel.
- **FR-024 (P1):** Support concurrency controls (max parallel steps, per-namespace quotas).
- **FR-025 (P1):** Support “map” style dynamic fan-out with deterministic child step IDs (bounded, policy-governed).

### 10.4 Handoffs and agent execution contracts
- **FR-030 (P0):** Invoke agents via a standardized “agent execution contract”:
  - inputs, expected outputs, permissions, resource hints.
- **FR-031 (P0):** Enforce step isolation boundaries (at least logical, ideally runtime sandbox).
- **FR-032 (P1):** Support synchronous and asynchronous agent execution modes.
- **FR-033 (P1):** Support human-in-the-loop handoff steps (approval gates) if enabled by policy.

### 10.5 Governance: credentials, permissions, policies
- **FR-040 (P0):** Evaluate permissions for pipeline submission and for each step.
- **FR-041 (P0):** Integrate with secrets management (never store secrets in plaintext).
- **FR-042 (P0):** Run policy checks:
  - pre-run (plan validation),
  - pre-step (before invocation),
  - post-step (artifact and output policy).
- **FR-043 (P1):** Support signed approvals/attestations for sensitive actions.

### 10.6 Audit trail, lineage, artifacts
- **FR-050 (P0):** Emit append-only run events (state transitions, policy decisions, agent invocations).
- **FR-051 (P0):** Store step artifacts with checksums and metadata.
- **FR-052 (P0):** Maintain lineage links:
  - step inputs -> outputs,
  - producer/consumer steps,
  - agent versions and environment.
- **FR-053 (P1):** Provide “exportable audit package” (plan + event log + artifact manifest).

### 10.7 Observability
- **FR-060 (P0):** Structured logs with run_id/step_id correlation.
- **FR-061 (P0):** Metrics (success rate, duration, retries, queue time).
- **FR-062 (P0):** Traces/spans per step (OpenTelemetry-friendly).
- **FR-063 (P1):** Alert hooks (webhooks, pager integrations) for failure/SLO breach.

### 10.8 Interfaces (API/CLI)
- **FR-070 (P0):** Start run (submit pipeline + config), return run_id/plan_id.
- **FR-071 (P0):** Get run status and per-step status.
- **FR-072 (P0):** Fetch logs pointers and artifact manifests.
- **FR-073 (P1):** List runs by namespace, time range, status; pagination.
- **FR-074 (P1):** Cancel run; retry from failed step (if idempotent and policy allows).

---

## 11. Non-functional requirements (NFRs)

### 11.1 Reliability and correctness
- **NFR-001 (P0):** Orchestrator must be crash-resilient; state stored durably.
- **NFR-002 (P0):** Step execution must be at-least-once; provide idempotency keys for agents.
- **NFR-003 (P0):** Exactly-once semantics are not required but should be approximated via idempotency.
- **NFR-004 (P0):** Deterministic planning (see Section 9).

### 11.2 Performance and scale
- **NFR-010 (P0):** Handle pipelines with at least 1,000 steps (planning) and 200 concurrent runnable steps (execution) in production targets (tunable).
- **NFR-011 (P1):** Planning latency under 2 seconds for typical pipelines (<100 steps).
- **NFR-012 (P1):** Horizontal scaling of schedulers/executors.

### 11.3 Security
- **NFR-020 (P0):** Least privilege for all tokens and agent calls.
- **NFR-021 (P0):** Encrypt data in transit and at rest (metadata + artifacts).
- **NFR-022 (P0):** Tenant isolation (namespace scoping, quotas, access checks).
- **NFR-023 (P1):** Support key rotation and short-lived credentials.

### 11.4 Compliance and audit
- **NFR-030 (P0):** Append-only event log and immutable metadata snapshots for runs.
- **NFR-031 (P1):** Support tamper-evident logging (hash chaining / signing).
- **NFR-032 (P1):** Data residency and PII handling policies enforceable.

### 11.5 Operability
- **NFR-040 (P0):** Health endpoints, readiness/liveness checks.
- **NFR-041 (P0):** Runbook-ready error codes and recommended remediation actions.
- **NFR-042 (P1):** Backpressure and queue controls to prevent overload.

---

## 12. System architecture

### 12.1 Logical components
1. **API Gateway / Control API**
2. **Pipeline Compiler**
3. **Planner**
4. **Policy Engine**
5. **Execution Scheduler**
6. **Agent Invocation Layer (Adapters)**
7. **State Store (Run/Step state)**
8. **Artifact Store + Artifact Metadata Service**
9. **Audit/Event Log**
10. **Observability (logs/metrics/traces)**

### 12.2 Suggested deployment model
- Control plane services run as highly-available pods/services.
- Executors can be:
  - local worker pool,
  - Kubernetes jobs,
  - serverless tasks,
  - external scheduler integration (pluggable).

### 12.3 Data stores
- **Metadata DB:** PostgreSQL (or compatible) for run/step state and indexes.
- **Object store:** S3/GCS/MinIO for artifacts.
- **Event log:** Kafka/Pulsar or append-only DB table + WORM backups.
- **Secrets:** Vault/KMS integrations.

---

## 13. Data model (conceptual)

### 13.1 Core entities
- `PipelineDefinition` (immutable YAML + schema version)
- `Plan` (compiled DAG + resolved agents + policy bundle references)
- `Run` (run_id, plan_id, status, timing, identity, config)
- `StepRun` (step_id, status, retries, timings, agent binding)
- `Artifact` (artifact_id, uri, checksum, media type, produced_by step, consumed_by steps)
- `RunEvent` (append-only; state transitions and decisions)

### 13.2 State machine
**Run status:** `QUEUED -> PLANNED -> RUNNING -> (SUCCEEDED | FAILED | CANCELED | TIMED_OUT)`  
**Step status:** `PENDING -> READY -> RUNNING -> (SUCCEEDED | FAILED | CANCELED | SKIPPED | TIMED_OUT)`

---

## 14. Policy model

### 14.1 Policy sources
- Namespace policy bundle (versioned)
- Organization-wide baseline policies
- Step-level policies declared in YAML (must be allowed)

### 14.2 Policy evaluation points
1. Pre-run (pipeline + plan)
2. Pre-step (permissions, cost, residency, dataset classification)
3. Post-step (artifact classification, export controls, lineage completeness)

### 14.3 Example policies
- No export of PII outside region.
- Require approval for publishing to production.
- Cost budget limits per run and per namespace.
- Model/tool allowlist by domain.

---

## 15. Observability specification

### 15.1 Logging
- JSON structured logs
- Mandatory fields: `timestamp`, `run_id`, `plan_id`, `step_id`, `event_type`, `severity`, `message`, `namespace`

### 15.2 Metrics (examples)
- `run_success_rate`, `run_duration_seconds`
- `step_duration_seconds`, `step_retry_count`
- `queue_time_seconds`
- `policy_denial_count`
- `artifact_write_failures`

### 15.3 Tracing
- Trace per run with spans per step invocation.
- Propagate trace context across agent calls.

---

## 16. Acceptance criteria (P0)

1. Given the same pipeline YAML + config + registry snapshot + policy bundle, plan hash is identical.
2. Can execute a DAG with dependencies, retries, and timeouts; correct terminal statuses.
3. Policy denials block execution and are auditable with reason codes.
4. Step artifacts are stored with checksums and retrievable via API.
5. Run event log shows complete state transitions for run and steps.
6. Logs/metrics/traces correlate via run_id/step_id.

---

## 17. Testing strategy

### 17.1 Automated tests
- Schema validation tests for Pipeline YAML.
- DAG compilation correctness (cycle detection, dependency validation).
- Determinism tests (same inputs -> same plan hash).
- Execution tests with fake agents (success/failure/timeout).
- Policy engine tests (allow/deny, reason codes).
- Artifact integrity tests (checksum validation).

### 17.2 Resilience / chaos tests
- Kill orchestrator pods mid-run; ensure recovery without losing state.
- Inject network failures to agents; verify retry/backoff.
- Corrupt artifact upload; verify detection and failure.

---

## 18. Rollout plan and milestones

### 18.1 Milestones
- **M1 (MVP, P0):** YAML intake + DAG planning + sequential execution + logs + artifacts + basic audit events.
- **M2 (Beta, P0/P1):** retries/timeouts/cancel + RBAC + policy checks + lineage model.
- **M3 (GA, P0/P1):** HA scaling + pluggable execution backend + tamper-evident audit + quota/concurrency controls.

### 18.2 SLO targets (initial)
- Orchestrator API availability: 99.9%
- Scheduler recovery time after crash: < 60 seconds
- Event/log delivery: at least once; eventual consistency < 2 minutes

---

## 19. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Determinism conflicts with dynamic scheduling | Audit/repro issues | Separate stable `plan_id` from runtime `run_id`; normalize inputs |
| Exactly-once is hard | Duplicate side effects | Enforce idempotency keys and agent-side idempotency |
| Policy complexity grows | Slower planning, more denials | Policy bundle versioning, caching, progressive enforcement |
| Multi-tenant isolation mistakes | Security incident | Strong RBAC, namespace scoping, automated security tests |

---

## 20. Open questions (to resolve during design)

1. Which execution backend is primary for GreenLang (Kubernetes jobs, worker pool, or hybrid)?
2. What is the canonical artifact store and retention policy per namespace?
3. Will policies be authored in OPA/Rego, or a GreenLang-native DSL?
4. What level of “human approval” is required for high-risk steps at GA?
5. What is the standard agent invocation protocol (HTTP/gRPC), payload schema, and auth?

---

## Appendix A: Minimal API sketch (conceptual)

- `POST /v1/runs` (submit pipeline + run config) -> `{run_id, plan_id}`
- `GET /v1/runs/{run_id}` (status)
- `GET /v1/runs/{run_id}/steps` (step statuses)
- `GET /v1/runs/{run_id}/artifacts` (artifact manifest)
- `GET /v1/runs/{run_id}/events` (audit events)
- `POST /v1/runs/{run_id}:cancel`

---

## Appendix B: Execution plan (conceptual JSON)

```json
{
  "plan_id": "pln_...",
  "plan_hash": "sha256:...",
  "pipeline": {
    "name": "example-pipeline",
    "namespace": "demo",
    "apiVersion": "greenlang/v1"
  },
  "steps": [
    {
      "step_id": "stp_ingest_...",
      "pipeline_step_id": "ingest",
      "agent": {
        "name": "OPS.DATA.Ingest",
        "version": "1.2.3"
      },
      "depends_on": [],
      "timeouts": {
        "step_seconds": 900
      },
      "retries": {
        "max_attempts": 3,
        "backoff": "exponential"
      },
      "policy": ["baseline", "no_pii_export"]
    }
  ]
}
```

---

## 21. Orchestration semantics (execution model)

### 21.1 Step types (v1)
GL-FOUND-X-001 must support at minimum these step types:

| Step type | Description | Determinism considerations |
|---|---|---|
| `agentTask` | Invoke a registered agent with inputs. | Step ID stable; agent version pinned at plan time. |
| `condition` | Branch based on an expression evaluated from prior outputs. | Expression evaluation must be pure and recorded. |
| `join` | Wait for multiple upstream steps. | Join semantics deterministic by DAG edges. |
| `noop` | Marker / documentation / group boundary. | Must not affect plan hash beyond explicit inclusion. |
| `approvalGate` (P1) | Human or policy approval required to proceed. | Attestation must be captured and signed. |

> P1/P2: `map` (dynamic fan-out), `foreach`, `waitUntil`, `sleep`, `subPipeline` (template expansion).

### 21.2 Execution semantics
- A step becomes **READY** when *all* dependencies are in a terminal success state (or explicitly allowed terminal states for that dependency).
- Scheduler MUST select runnable steps using a deterministic ordering rule:
  1. primary: topological order (Kahn’s algorithm),
  2. tie-breaker: lexical sort by `pipeline_step_id`,
  3. tie-breaker: stable hash of the step definition.
- When concurrency is enabled, ordering should still be deterministic in selection, even if wall-clock completion differs.

### 21.3 Idempotency and side effects
- Orchestrator MUST generate an `idempotency_key` per step attempt:
  - derived from `(plan_hash, step_id, attempt_number)`.
- Agents MUST receive the idempotency key and are expected to implement idempotent behavior for retried attempts.
- If an agent cannot be idempotent, the registry MUST declare the step as `non_idempotent: true` and orchestrator must restrict retries by policy.

---

## 22. Agent registry contract (v1)

### 22.1 Required registry fields
Each agent entry should minimally provide:

- `agent_name` (global unique)
- `agent_version` (semver or content hash)
- `interface` (HTTP/gRPC, endpoints)
- `input_schema` / `output_schema` (JSON schema or equivalent)
- `required_permissions` (scopes, datasets, actions)
- `resource_hints` (cpu/mem/gpu, expected runtime)
- `idempotency_support` (true/false)
- `observability` (log/trace propagation capabilities)

### 22.2 Version pinning rules
- Default rule: **pin latest compatible version at plan time**, then store it in the plan.
- Optional: allow explicit version pinning in YAML (`agent: OPS.DATA.Ingest@1.2.3`).
- Compatibility must be determined by:
  - major version changes breaking,
  - declared contract versions.

### 22.3 Registry snapshotting
At submission time, orchestrator must record:
- registry snapshot hash,
- resolved agent versions,
- agent manifests (or immutable references).

---

## 23. Error taxonomy, retries, and timeouts

### 23.1 Error classes
Orchestrator should classify failures into:

| Class | Examples | Default retry? |
|---|---|---|
| `TRANSIENT` | network timeouts, 5xx, rate limit | yes |
| `RESOURCE` | OOM, node eviction, quota exceeded | yes (bounded) |
| `USER_CONFIG` | invalid params, missing dataset | no |
| `POLICY_DENIED` | permission/policy fails | no |
| `AGENT_BUG` | deterministic crash, 4xx non-retry | no (unless agent declares safe retry) |

Classification sources:
- agent response codes,
- execution backend signals,
- policy engine output.

### 23.2 Retry policy (defaults)
- exponential backoff with jitter
- max attempts: pipeline default or step override
- circuit breaker: if failure rate exceeds threshold, pause and alert (P1)

### 23.3 Timeouts
- **Soft timeout:** orchestrator marks step as timed out and requests cancellation.
- **Hard timeout:** backend-enforced kill (K8s job deadline, worker kill).
- Both timeouts must be recorded in audit events.

---

## 24. Security and permissions model

### 24.1 Identity model
- Every run has:
  - a **submitter identity** (human or service),
  - an **execution identity** (service account) scoped to namespace and policies.
- All agent calls must be authenticated with short-lived credentials.

### 24.2 Authorization model
- Prefer ABAC over pure RBAC for data-centric rules:
  - subject: user/service identity,
  - resource: dataset/artifact/classification,
  - action: read/write/publish/export,
  - environment: prod/dev, region.

### 24.3 Example roles (suggested)
| Role | Permissions |
|---|---|
| `PipelineAuthor` | submit runs, view own runs, read allowed artifacts |
| `Operator` | cancel runs, retry runs, view all runs in namespace |
| `GovernanceAuditor` | read audit packages and policy decisions (no execution) |
| `RegistryAdmin` | publish agent versions, deprecate agents |

### 24.4 Secrets handling
- Pipeline YAML must never contain raw secrets.
- Only references are allowed, e.g. `secretRef: vault://path/to/secret`.
- Orchestrator must request secrets on-demand and inject them ephemerally into execution context.

---

## 25. Artifact, lineage, and retention specification

### 25.1 Artifact naming and addressing
- Artifact URIs should include: namespace, run_id, step_id, artifact_id.
- Every artifact stored with:
  - `sha256` checksum,
  - media type,
  - size,
  - producer step,
  - logical name (e.g., `dataset`, `report`, `model_card`).

### 25.2 Lineage rules
- For every step:
  - record input artifact references,
  - record output artifacts,
  - record transformation agent + version.

### 25.3 Retention and deletion
- Retention should be policy-driven per namespace (e.g., 30/90/365 days).
- For compliance, some runs may be “hold” (legal hold) and not deletable.
- Deletion should remove artifacts but keep minimal audit metadata (depending on policy).

### 25.4 Example artifact manifest
```json
{
  "run_id": "run_...",
  "artifacts": [
    {
      "artifact_id": "art_...",
      "name": "dataset",
      "uri": "s3://gl-artifacts/demo/run_.../ingest/art_....parquet",
      "sha256": "....",
      "produced_by_step_id": "stp_ingest_...",
      "consumed_by_step_ids": ["stp_validate_..."],
      "media_type": "application/x-parquet"
    }
  ]
}
```

---

## 26. Audit trail and tamper-evidence

### 26.1 Event model (append-only)
All significant actions must emit a `RunEvent`, e.g.:
- `RUN_SUBMITTED`, `PLAN_COMPILED`, `POLICY_EVALUATED`
- `STEP_READY`, `STEP_STARTED`, `STEP_RETRIED`, `STEP_SUCCEEDED`, `STEP_FAILED`
- `ARTIFACT_WRITTEN`, `RUN_SUCCEEDED`, `RUN_FAILED`, `RUN_CANCELED`

### 26.2 Tamper evidence (P1)
- Hash-chain events per run:
  - each event stores `prev_event_hash` and `event_hash`.
- Optional signing:
  - sign final event hash with platform key.
- Store audit packages in WORM storage.

---

## 27. Detailed API examples (conceptual)

### 27.1 Submit run
Request:
```json
{
  "namespace": "demo",
  "pipeline_yaml": "...",
  "run_config": {
    "params": {"input_uri": "s3://..."},
    "priority": "normal",
    "budgets": {"max_cost_usd": 10}
  }
}
```

Response:
```json
{
  "run_id": "run_...",
  "plan_id": "pln_...",
  "plan_hash": "sha256:..."
}
```

### 27.2 Step status response snippet
```json
{
  "run_id": "run_...",
  "steps": [
    {"pipeline_step_id": "ingest", "step_id": "stp_ingest_...", "status": "SUCCEEDED", "duration_ms": 120034},
    {"pipeline_step_id": "validate", "step_id": "stp_validate_...", "status": "RUNNING"}
  ]
}
```

---

## 28. Operational runbook (starter)

### 28.1 Common failure modes
- **Policy denial spike:** check policy bundle rollout, validate evaluation latency.
- **Artifact store outage:** orchestrator should fail fast with `ARTIFACT_WRITE_FAILURE`, pause new runs.
- **Agent timeouts:** inspect backend saturation, raise per-namespace quotas, validate agent SLAs.

### 28.2 On-call actions
- Cancel runaway runs (budget/policy).
- Re-run from a safe checkpoint (idempotency required).
- Export audit package for incidents.

---

## 29. Success metrics

- % runs producing complete lineage (target: > 99% after GA)
- mean time to diagnose run failure (MTTD) reduced by correlated logs/traces
- policy compliance rate (target: 100% enforced; any exceptions recorded)
- orchestrator control plane availability and p95 planning latency

---

## Appendix C: Advanced pipeline example (fan-out + join, conceptual)

```yaml
apiVersion: greenlang/v1
kind: Pipeline
metadata:
  name: sector-batch-analysis
  namespace: sector-x
spec:
  parameters:
    uris:
      type: list[string]
  steps:
    - id: ingest_all
      type: map
      items: "{{{{ params.uris }}}}"
      itemVar: uri
      agent: OPS.DATA.Ingest
      with:
        uri: "{{{{ item.uri }}}}"

    - id: join_ingest
      type: join
      dependsOn: [ingest_all]

    - id: publish
      dependsOn: [join_ingest]
      agent: OPS.DATA.Publish
```
