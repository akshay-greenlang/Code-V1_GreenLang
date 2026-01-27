# ADR-007: Legacy Adapter Strategy for HTTP Agents

**Date:** 2026-01-27
**Status:** Accepted
**Deciders:** Platform Engineering, Architecture Team, Agent Development Teams
**Consulted:** SRE/Ops, Security Team, Individual Agent Owners

---

## Context

### Problem Statement
GL-FOUND-X-001 (GreenLang Orchestrator) adopts GLIP v1 (artifact-first, K8s Job-native) as the standard agent invocation protocol. However, 119 existing agents currently use HTTP-based invocation and cannot be immediately migrated. We need a strategy that:
- Allows existing HTTP agents to work with the new orchestrator
- Provides a migration path to native GLIP v1
- Minimizes disruption to agent owners
- Maintains audit and governance requirements

### Current Situation
- **Total Agents:** 402 agents in the GreenLang ecosystem
- **HTTP Agents:** 119 agents using HTTP/REST invocation (29.6%)
- **GLIP v1 Native:** 283 agents to be built with GLIP v1 (70.4%)
- **Migration Timeline:** 12-18 months for full migration
- **Agent Ownership:** Distributed across 15+ teams

### Business Impact
- **Continuity:** Cannot halt operations during migration
- **Team Velocity:** Agent teams have competing priorities
- **Risk Mitigation:** Gradual migration reduces deployment risk
- **Cost Efficiency:** Reuse existing agents rather than rewrite

---

## Decision

### What We're Implementing
**Gradual migration with a shared adapter library** that wraps HTTP agents in GLIP v1-compatible containers.

### Core Strategy

1. **Shared Adapter Library**
   - Generic HTTP-to-GLIP adapter container
   - Configurable for any HTTP agent
   - Handles artifact download/upload
   - Provides GLIP v1 compliance layer

2. **Adapter Architecture**
   ```
   K8s Job (GLIP v1 Container)
   +----------------------------------+
   |  GLIP v1 Adapter                 |
   |  +----------------------------+  |
   |  | 1. Read input artifacts    |  |
   |  | 2. Call HTTP agent         |  |
   |  | 3. Capture response        |  |
   |  | 4. Write output artifacts  |  |
   |  +----------------------------+  |
   |           |                      |
   |           v                      |
   |  +----------------------------+  |
   |  | HTTP Agent (sidecar/remote)|  |
   |  +----------------------------+  |
   +----------------------------------+
   ```

3. **Adapter Configuration**
   ```yaml
   apiVersion: adapter.glip.greenlang.io/v1
   kind: HTTPAdapter
   metadata:
     name: legacy-ingest-adapter
   spec:
     agent:
       name: "LEGACY.DATA.Ingest"
       version: "2.1.0"
     http:
       endpoint: "http://legacy-ingest-service:8080/invoke"
       method: POST
       timeout_seconds: 300
       retry:
         max_attempts: 3
         backoff_multiplier: 2
     input_mapping:
       - glip_input: "source_data"
         http_field: "input_file_url"
         transform: "presigned_url"
       - glip_input: "config"
         http_field: "configuration"
         transform: "json_embed"
     output_mapping:
       - http_field: "result_url"
         glip_output: "dataset"
         transform: "download_artifact"
       - http_field: "metadata"
         glip_output: "processing_metadata"
         transform: "json_artifact"
     health_check:
       endpoint: "/health"
       interval_seconds: 30
   ```

4. **Two Invocation Paths (Transition Period)**
   ```
   Orchestrator
        |
        +---> GLIP v1 Native Path (new agents)
        |          |
        |          v
        |     K8s Job Executor
        |
        +---> Legacy Adapter Path (HTTP agents)
                   |
                   v
              K8s Job with Adapter Container
                   |
                   v
              HTTP Agent (sidecar or external)
   ```

### Technology Stack
- **Adapter Image:** `ghcr.io/greenlang/glip-http-adapter:1.0`
- **HTTP Client:** httpx (async, retry support)
- **Artifact Handling:** boto3 for S3 operations
- **Configuration:** Pydantic models for validation

### Code Location
- `greenlang/orchestrator/adapters/`
  - `http_adapter.py` - HTTP adapter implementation
  - `config.py` - Adapter configuration models
  - `mapping.py` - Input/output transformation
  - `health.py` - Health check monitoring
- `containers/glip-http-adapter/`
  - `Dockerfile` - Adapter container image
  - `entrypoint.py` - Container entrypoint

---

## Rationale

### Why Gradual Migration with Shared Adapter

**1. Low-Risk Transition**
- Existing HTTP agents continue working unchanged
- No forced migration deadline
- Issues isolated to adapter layer

**2. Reuse Existing Agents**
- 119 agents represent significant investment
- Avoid rewriting working code
- Preserve domain expertise in agents

**3. Team Autonomy**
- Teams migrate on their own schedule
- No blocking dependencies
- Parallel migration across teams

**4. Single Adapter Codebase**
- One adapter serves all HTTP agents
- Bug fixes benefit all wrapped agents
- Centralized observability

**5. Clear Migration Path**
- Adapter provides GLIP v1 compatibility today
- Native migration when ready
- Incentives to migrate (performance, simplicity)

---

## Alternatives Considered

### Alternative 1: Big Bang Migration
**Pros:**
- Single cutover, no transition period
- No two-path complexity
- Cleaner architecture

**Cons:**
- High risk of failures
- All 119 agents must migrate simultaneously
- Blocks all teams until migration complete
- Rollback difficult

**Why Rejected:** Risk too high. A single failure could halt all agent execution. Migration complexity multiplied across 15+ teams.

### Alternative 2: No Migration (Keep HTTP Forever)
**Pros:**
- No migration effort
- Existing agents work as-is
- No team disruption

**Cons:**
- Lose GLIP v1 benefits (determinism, artifacts, isolation)
- Inconsistent invocation model
- Audit trail gaps for HTTP agents
- Two permanent execution paths

**Why Rejected:** Does not achieve GLIP v1 goals. Perpetual complexity and audit inconsistency.

### Alternative 3: Per-Agent Custom Adapters
**Pros:**
- Optimized for each agent
- No generic overhead

**Cons:**
- 119 custom adapters to build and maintain
- Duplicated effort across teams
- Inconsistent adapter quality
- No shared improvements

**Why Rejected:** Massive duplication of effort. Shared adapter is more efficient and maintainable.

### Alternative 4: Orchestrator-Level HTTP Support
**Pros:**
- No adapter containers needed
- Lower latency
- Simpler deployment

**Cons:**
- Orchestrator becomes complex (two protocols)
- HTTP path lacks artifact-first benefits
- Inconsistent execution model
- Harder to deprecate HTTP path

**Why Rejected:** Complicates orchestrator core. Adapter approach keeps orchestrator clean and focused.

---

## Consequences

### Positive
- **Continuity:** Existing HTTP agents work immediately
- **Low Risk:** Gradual migration reduces deployment risk
- **Reuse:** 119 agents preserved, no rewrites
- **Flexibility:** Teams migrate on their schedule
- **Centralization:** Single adapter codebase to maintain
- **Audit Consistency:** Adapter provides GLIP v1 audit events

### Negative
- **Two Paths:** Temporary complexity with two invocation paths
- **Adapter Overhead:** Additional container layer for HTTP agents
- **Latency:** HTTP round-trip adds latency vs. native GLIP
- **Maintenance:** Adapter must be maintained until migration complete
- **Configuration:** Each HTTP agent needs adapter config

### Neutral
- **Incentives:** Native GLIP v1 is faster, simpler (natural migration driver)
- **Deprecation Timeline:** HTTP path deprecated after 18 months (planned)

---

## Implementation Plan

### Phase 1: Adapter Core (Week 1-2)
1. Implement HTTP adapter container
2. Build input/output mapping framework
3. Add retry and timeout handling
4. Create health check monitoring

### Phase 2: Configuration (Week 3)
1. Define adapter configuration schema
2. Build configuration validation
3. Create adapter registry in orchestrator
4. Implement dynamic adapter loading

### Phase 3: Integration (Week 4-5)
1. Integrate adapter path in orchestrator executor
2. Add observability (logs, metrics, traces)
3. Implement audit event generation
4. Test with 5 pilot HTTP agents

### Phase 4: Rollout (Week 6-8)
1. Generate adapter configs for all 119 HTTP agents
2. Validate each agent with adapter
3. Deploy to production
4. Monitor and resolve issues

### Phase 5: Migration Support (Ongoing)
1. Create native GLIP v1 migration guide
2. Build migration tooling
3. Track migration progress per agent
4. Deprecation notices at 12 months

---

## Adapter Configuration Examples

### Simple HTTP Agent
```yaml
apiVersion: adapter.glip.greenlang.io/v1
kind: HTTPAdapter
metadata:
  name: simple-transform-adapter
spec:
  agent:
    name: "LEGACY.DATA.Transform"
    version: "1.0.0"
  http:
    endpoint: "http://transform-svc:8080/transform"
    method: POST
  input_mapping:
    - glip_input: "input_data"
      http_field: "data"
      transform: "json_embed"
  output_mapping:
    - http_field: "result"
      glip_output: "output_data"
      transform: "json_artifact"
```

### Complex HTTP Agent with Auth
```yaml
apiVersion: adapter.glip.greenlang.io/v1
kind: HTTPAdapter
metadata:
  name: erp-connector-adapter
spec:
  agent:
    name: "LEGACY.ERP.SAPConnector"
    version: "3.2.1"
  http:
    endpoint: "https://sap-connector.internal:443/sync"
    method: POST
    timeout_seconds: 600
    headers:
      X-API-Version: "2.0"
    auth:
      type: oauth2
      token_url: "https://auth.internal/token"
      client_id_secret: "vault://secrets/sap/client_id"
      client_secret_secret: "vault://secrets/sap/client_secret"
  input_mapping:
    - glip_input: "erp_query"
      http_field: "query"
      transform: "json_embed"
    - glip_input: "date_range"
      http_field: "dateRange"
      transform: "json_embed"
  output_mapping:
    - http_field: "data"
      glip_output: "erp_data"
      transform: "download_artifact"
    - http_field: "sync_metadata"
      glip_output: "sync_report"
      transform: "json_artifact"
```

---

## Migration Path

### From HTTP to Native GLIP v1

1. **Assessment**
   - Review HTTP agent code
   - Identify input/output patterns
   - Plan artifact storage

2. **Refactor**
   - Read inputs from GLIP manifest
   - Write outputs to GLIP output directory
   - Remove HTTP server code

3. **Containerize**
   - Create Dockerfile
   - Test container locally
   - Push to registry

4. **Register**
   - Update agent registry with GLIP v1 manifest
   - Remove adapter configuration
   - Test in staging

5. **Deploy**
   - Update production registry
   - Monitor for issues
   - Remove legacy HTTP service

### Migration Incentives
- **Performance:** Native GLIP v1 faster (no adapter overhead)
- **Simplicity:** Single container vs. adapter + agent
- **Features:** Full GLIP v1 observability and lineage
- **Support:** HTTP adapter deprecated after 18 months

---

## Compliance & Security

### Security Considerations
- **Secrets Handling:** Adapter uses vault for auth secrets
- **Network Isolation:** HTTP calls within K8s network
- **TLS:** All HTTP calls use TLS
- **Timeout Enforcement:** Hard timeouts prevent runaway calls

### Audit Considerations
- **Event Generation:** Adapter generates GLIP v1 audit events
- **Artifact Storage:** Adapter stores outputs as artifacts
- **Lineage Tracking:** Input/output mapping recorded
- **Transparency:** Adapter config part of audit record

---

## Migration Tracking

### Dashboard Metrics
- Total HTTP agents: 119
- Migrated to native GLIP v1: 0 (initial)
- Using adapter: 119 (initial)
- Migration velocity: agents/month

### Milestones
| Milestone | Target Date | HTTP Agents | Native GLIP v1 |
|-----------|-------------|-------------|----------------|
| Adapter Launch | Q1 2026 | 119 | 0 |
| 25% Migrated | Q2 2026 | 89 | 30 |
| 50% Migrated | Q3 2026 | 60 | 59 |
| 75% Migrated | Q4 2026 | 30 | 89 |
| 100% Migrated | Q2 2027 | 0 | 119 |
| Adapter Deprecated | Q3 2027 | N/A | N/A |

---

## Links & References

- **PRD:** GL-FOUND-X-001 GreenLang Orchestrator
- **Related ADRs:** ADR-001 (GLIP v1), ADR-002 (K8s Jobs)
- **Adapter Image:** `ghcr.io/greenlang/glip-http-adapter`
- **Migration Guide:** `docs/agents/http-to-glip-migration.md`

---

## Updates

### 2026-01-27 - Status: Accepted
ADR approved by Platform Engineering and Agent Development Teams. Adapter development starting Q1 2026.

---

**Template Version:** 1.0
**Last Updated:** 2026-01-27
**ADR Author:** Platform Architecture Team
**Reviewers:** Agent Development Teams, SRE/Ops, Security Team
