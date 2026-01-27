# ADR-001: GLIP v1 Protocol Choice for Agent Invocation

**Date:** 2026-01-27
**Status:** Accepted
**Deciders:** Platform Engineering, Architecture Team, Security Team
**Consulted:** Agent Development Teams, SRE/Ops, Governance/Compliance

---

## Context

### Problem Statement
GL-FOUND-X-001 (GreenLang Orchestrator) must invoke 402 agents across multiple domains and sectors. We need a standardized agent invocation protocol that ensures determinism, auditability, and language-agnostic execution.

### Current Situation
- **Agent Landscape:** 402 agents written in Python, Go, Rust, and Node.js
- **Execution Requirements:** Deterministic, reproducible, auditable invocations
- **Compliance Needs:** Regulatory audit trails require complete invocation records
- **Scale:** Support 200+ concurrent step executions with reliable handoffs

### Business Impact
- **Compliance:** Regulatory frameworks (CBAM, GHG Protocol, CSRD) require reproducible calculations
- **Audit:** External auditors must verify that identical inputs produce identical outputs
- **Multi-tenant:** Enterprise customers require isolation guarantees
- **Time-to-Market:** New agents must integrate without protocol negotiation overhead

---

## Decision

### What We're Implementing
**GLIP v1 (GreenLang Invocation Protocol v1)**: An artifact-first, Kubernetes Job-native protocol for agent invocation.

### Core Protocol Principles

1. **Artifact-First Design**
   - All inputs are artifacts (files, JSON blobs, references)
   - All outputs are artifacts stored with checksums
   - No in-memory state transfer between agents

2. **Declarative Invocation**
   ```yaml
   apiVersion: glip/v1
   kind: Invocation
   metadata:
     invocation_id: "inv_abc123"
     idempotency_key: "sha256:plan_hash+step_id+attempt"
   spec:
     agent:
       name: "OPS.DATA.Ingest"
       version: "1.2.3"
       image: "ghcr.io/greenlang/agents/ingest:1.2.3"
     inputs:
       - name: "source_data"
         artifact_uri: "s3://gl-artifacts/run_xyz/input.csv"
         sha256: "abc123..."
     outputs:
       - name: "dataset"
         artifact_prefix: "s3://gl-artifacts/run_xyz/ingest/"
     resources:
       cpu: "2"
       memory: "4Gi"
       timeout_seconds: 900
     observability:
       trace_context: "00-abc123-def456-01"
   ```

3. **K8s Job Native**
   - Each agent invocation maps to a Kubernetes Job
   - Resource isolation via pod security contexts
   - Automatic cleanup via Job TTL

### Technology Stack
- **Protocol Format:** YAML/JSON (OpenAPI 3.1 schema)
- **Transport:** Kubernetes Job API (primary), HTTP REST (legacy adapter)
- **Serialization:** JSON for metadata, binary for artifacts
- **Authentication:** ServiceAccount tokens, short-lived JWTs

### Code Location
- `greenlang/orchestrator/protocols/glip/v1/`
  - `schema.py` - Protocol schema definitions
  - `invocation.py` - Invocation builder and validator
  - `executor.py` - K8s Job executor
  - `artifacts.py` - Artifact resolution and storage

---

## Rationale

### Why GLIP v1 (Artifact-First, K8s Job-Native)

**1. Determinism**
- Artifact URIs with SHA256 checksums guarantee reproducibility
- No hidden state in memory or environment variables
- Same artifacts + same agent version = same outputs

**2. Language-Agnostic**
- Agents are containers; any language can implement the protocol
- No SDK dependency required; just read inputs, write outputs
- Polyglot teams can use their preferred stack

**3. Audit-Friendly**
- Complete invocation records stored as artifacts
- Every input/output is addressable and verifiable
- Lineage is automatic: inputs -> agent -> outputs

**4. Cloud-Native**
- Kubernetes Jobs provide resource isolation and scheduling
- Built-in retry, timeout, and cleanup mechanisms
- Horizontal scaling via K8s node autoscaling

**5. Security**
- Pod security contexts enforce isolation
- ServiceAccount tokens provide least-privilege access
- Network policies restrict agent communication

---

## Alternatives Considered

### Alternative 1: HTTP/REST Direct Invocation
**Pros:**
- Simple, widely understood
- Low latency for small payloads
- Existing tooling (OpenAPI, Swagger)

**Cons:**
- Requires agents to run as long-lived services
- Memory-based state transfer (not artifact-first)
- Harder to guarantee determinism
- Resource isolation requires additional orchestration

**Why Rejected:** Does not meet determinism and audit requirements. Would require significant additional infrastructure for isolation.

### Alternative 2: gRPC with Protobuf
**Pros:**
- Strong typing via Protocol Buffers
- Streaming support for large payloads
- High performance binary serialization

**Cons:**
- Requires agents to implement gRPC servers
- SDK/code generation adds complexity
- Still memory-based, not artifact-first
- Steeper learning curve for some teams

**Why Rejected:** Adds SDK complexity without solving the fundamental artifact-first requirement. Performance benefits not critical for batch agent workloads.

### Alternative 3: Message Queue (Kafka/RabbitMQ)
**Pros:**
- Decoupled producer/consumer model
- Built-in retry and dead-letter queues
- High throughput for event streaming

**Cons:**
- Adds operational complexity (queue management)
- Messages are not naturally artifact-addressed
- Ordering guarantees complex for DAG execution
- Not K8s-native; requires additional infrastructure

**Why Rejected:** Adds infrastructure dependency without clear benefit over K8s Jobs. Queue semantics don't map well to DAG step execution.

### Alternative 4: Hybrid (HTTP for small, Jobs for large)
**Pros:**
- Flexibility based on payload size
- Lower latency for simple agents

**Cons:**
- Two execution paths to maintain
- Inconsistent audit trail format
- Complexity in routing decisions

**Why Rejected:** Consistency is more valuable than marginal performance gains. Single protocol simplifies operations and auditing.

---

## Consequences

### Positive
- **Determinism:** Reproducible agent invocations for audit and compliance
- **Isolation:** Strong resource isolation via K8s pod boundaries
- **Polyglot:** Any language can implement agents (just containers)
- **Audit Trail:** Complete, verifiable invocation records
- **Lineage:** Automatic input/output tracking via artifact URIs
- **Scalability:** K8s horizontal scaling handles load spikes
- **Simplicity:** One protocol for all 402 agents

### Negative
- **Containerization Required:** All agents must be packaged as containers
- **Cold Start Latency:** Job creation adds ~5-15 seconds overhead
- **Storage Costs:** Artifact-first means more S3/object storage usage
- **K8s Expertise:** Operations team must be proficient in Kubernetes
- **Legacy Migration:** 119 existing HTTP agents need adapters (see ADR-007)

### Neutral
- **Learning Curve:** New protocol requires documentation and training
- **Tooling:** New CLI/SDK tools needed for agent development

---

## Implementation Plan

### Phase 1: Protocol Specification (Week 1-2)
1. Finalize GLIP v1 OpenAPI schema
2. Document protocol semantics and error codes
3. Create validation library
4. Write protocol compliance test suite

### Phase 2: Executor Implementation (Week 3-4)
1. Implement K8s Job executor
2. Integrate with artifact store (S3)
3. Add observability (traces, logs, metrics)
4. Implement idempotency key handling

### Phase 3: Agent SDK (Week 5-6)
1. Create Python SDK for GLIP v1
2. Create Go SDK for GLIP v1
3. Write agent development guide
4. Build example agents

### Phase 4: Migration (Week 7-12)
1. Deploy legacy HTTP adapter (ADR-007)
2. Migrate agents in batches
3. Validate audit trail completeness
4. Deprecate legacy invocation path

---

## Compliance & Security

### Security Considerations
- **Authentication:** ServiceAccount tokens with RBAC
- **Authorization:** Namespace-scoped permissions
- **Encryption:** TLS 1.3 for all API calls, AES-256 for artifacts at rest
- **Isolation:** Pod security contexts, network policies
- **Secrets:** Injected via Kubernetes Secrets, never in invocation spec

### Audit Considerations
- **Invocation Records:** Stored as immutable artifacts
- **Checksums:** SHA256 for all inputs and outputs
- **Lineage:** Producer/consumer relationships tracked
- **Retention:** Configurable per namespace (30/90/365 days)

---

## Migration Plan

### Short-term (0-6 months)
- Deploy GLIP v1 executor alongside legacy HTTP path
- New agents must use GLIP v1
- Existing agents continue on HTTP with adapter

### Medium-term (6-12 months)
- Migrate 50% of agents to native GLIP v1
- Deprecation notices for HTTP-only agents
- Performance benchmarking and optimization

### Long-term (12+ months)
- Complete migration to GLIP v1
- Remove legacy HTTP adapter
- GLIP v2 planning based on learnings

---

## Links & References

- **PRD:** GL-FOUND-X-001 GreenLang Orchestrator
- **Related ADRs:** ADR-002 (K8s Jobs), ADR-004 (S3 Artifacts), ADR-007 (Legacy Adapter)
- **OpenAPI Schema:** `schemas/glip-v1.yaml`
- **Agent Development Guide:** `docs/agents/glip-v1-guide.md`

---

## Updates

### 2026-01-27 - Status: Accepted
ADR approved by Platform Engineering and Architecture Team. Implementation scheduled for Q1 2026.

---

**Template Version:** 1.0
**Last Updated:** 2026-01-27
**ADR Author:** Platform Architecture Team
**Reviewers:** Security Team, SRE, Agent Development Leads
