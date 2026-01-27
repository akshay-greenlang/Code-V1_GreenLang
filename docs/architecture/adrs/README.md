# Architecture Decision Records (ADRs) - GL-FOUND-X-001

This directory contains Architecture Decision Records for **GL-FOUND-X-001 (GreenLang Orchestrator)**, the foundation and governance layer agent responsible for planning and executing multi-agent pipelines.

## Overview

The GreenLang Orchestrator is the control plane for GreenLang pipelines, providing:
- Pipeline compilation and deterministic execution planning
- Multi-agent DAG orchestration with retries, timeouts, and handoffs
- Policy enforcement and governance
- Auditable run metadata and lineage tracking

## ADR Index

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [ADR-001](ADR-001-GLIP-v1-Protocol-Choice.md) | GLIP v1 Protocol Choice | Accepted | Artifact-first, K8s Job-native protocol for agent invocation |
| [ADR-002](ADR-002-Kubernetes-Jobs-Primary-Executor.md) | K8s Jobs as Primary Executor | Accepted | Kubernetes Jobs for reliable, isolated agent execution |
| [ADR-003](ADR-003-Hybrid-Policy-Engine-OPA-YAML.md) | Hybrid Policy Engine (OPA + YAML) | Accepted | OPA/Rego for complex rules, YAML for simple rules |
| [ADR-004](ADR-004-S3-Artifact-Store.md) | S3 Artifact Store | Accepted | AWS S3 with S3-compatible API support for artifact storage |
| [ADR-005](ADR-005-Hash-Chained-Audit-Events.md) | Hash-Chained Audit Events | Accepted | Tamper-evident audit trail using hash chains |
| [ADR-006](ADR-006-Deterministic-Plan-Generation.md) | Deterministic Plan Generation | Accepted | Content-addressable plan_id with normalized YAML |
| [ADR-007](ADR-007-Legacy-Adapter-Strategy.md) | Legacy Adapter Strategy | Accepted | Gradual migration for 119 HTTP agents |

## Decision Summary

### ADR-001: GLIP v1 Protocol Choice
- **Context:** Need standard agent invocation protocol for 402 agents
- **Decision:** GLIP v1 (GreenLang Invocation Protocol v1) - artifact-first, K8s Job-native
- **Rationale:** Determinism, language-agnostic, audit-friendly
- **Alternatives Rejected:** HTTP/REST, gRPC, Message Queue
- **Consequences:** All agents must be containerized

### ADR-002: K8s Jobs as Primary Executor
- **Context:** Need reliable, isolated execution backend
- **Decision:** Kubernetes Jobs as the primary execution mechanism
- **Rationale:** Cloud-native, auto-scaling, resource isolation, mature ecosystem
- **Alternatives Rejected:** Worker pool, Serverless functions, Hybrid approach
- **Consequences:** K8s expertise required, cold start latency

### ADR-003: Hybrid Policy Engine (OPA + YAML)
- **Context:** Need governance without steep learning curve
- **Decision:** OPA/Rego for complex rules, YAML for simple declarative rules
- **Rationale:** Balance of power and usability for different personas
- **Alternatives Rejected:** Pure OPA, Custom DSL, Python functions
- **Consequences:** Two policy languages to maintain and document

### ADR-004: S3 Artifact Store
- **Context:** Need durable artifact storage with lineage tracking
- **Decision:** AWS S3 with S3-compatible API support (MinIO, GCS, Azure Blob)
- **Rationale:** Mature, multi-region capable, auditable, industry standard API
- **Alternatives Rejected:** GCS only, Azure Blob only, MinIO only, HDFS
- **Consequences:** AWS S3 API as the interface, metadata DB required

### ADR-005: Hash-Chained Audit Events
- **Context:** Need tamper-evident audit trail for compliance
- **Decision:** Hash-chain events at MVP (no full PKI signing initially)
- **Rationale:** Basic tamper-evidence without complex signing infrastructure
- **Alternatives Rejected:** No signing, Full PKI, WORM storage only, Blockchain
- **Consequences:** Performance overhead for hash computation, foundation for future signing

### ADR-006: Deterministic Plan Generation
- **Context:** Same inputs must produce same plan for reproducibility
- **Decision:** Content-addressable plan_id, normalized YAML processing
- **Rationale:** Audit reproducibility, governance verification, zero-hallucination
- **Alternatives Rejected:** Random UUIDs, Timestamp-based IDs, AI-assisted planning
- **Consequences:** Careful YAML normalization required, registry snapshotting

### ADR-007: Legacy Adapter Strategy
- **Context:** 119 existing agents use HTTP invocation, cannot migrate immediately
- **Decision:** Gradual migration with shared HTTP adapter library
- **Rationale:** Low-risk transition, reuse existing agents, team autonomy
- **Alternatives Rejected:** Big bang migration, No migration, Per-agent custom adapters
- **Consequences:** Two invocation paths during 18-month transition period

## Key Architectural Principles

1. **Artifact-First Design:** All inputs and outputs are artifacts with checksums
2. **Determinism:** Same inputs always produce same execution plan
3. **Audit-Friendly:** Complete, tamper-evident records of all operations
4. **Language-Agnostic:** Agents can be written in any language (containerized)
5. **Cloud-Native:** Kubernetes-native execution with portability
6. **Progressive Governance:** Simple YAML policies to complex Rego as needed
7. **Graceful Migration:** Legacy support with clear migration path

## Related Documentation

- [PRD: GL-FOUND-X-001 GreenLang Orchestrator](../../../GreenLang_Agents_PRD_402/PRD_GL-FOUND-X-001_Agent.md)
- [Platform Architecture](../platform-architecture.md)
- [Agent Factory Architecture](../AGENT_FACTORY_ARCHITECTURE.md)

## ADR Template

New ADRs should follow the template in [.greenlang/adrs/TEMPLATE.md](../../../.greenlang/adrs/TEMPLATE.md).

## Updating ADRs

When updating an ADR:
1. Add an entry to the "Updates" section at the bottom
2. Change status if applicable (Proposed -> Accepted, Accepted -> Superseded)
3. If superseding, create new ADR and link both documents

---

**Last Updated:** 2026-01-27
**Maintainer:** Platform Architecture Team
