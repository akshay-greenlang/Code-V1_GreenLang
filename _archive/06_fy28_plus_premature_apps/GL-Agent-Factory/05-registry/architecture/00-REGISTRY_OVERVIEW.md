# Agent Registry Architecture

**Version:** 1.0.0
**Status:** PRODUCTION
**Owner:** GL-DevOpsEngineer
**Last Updated:** 2025-12-03

---

## Executive Summary

The Agent Registry is the centralized metadata repository that treats agents as first-class, versioned, governed assets within the GreenLang Agent Factory. It provides comprehensive lifecycle management, governance controls, discovery capabilities, and integration with the Runtime execution environment.

**Key Capabilities:**
- Immutable agent versioning with semantic versioning (SemVer)
- Multi-tenant governance and access control
- Lifecycle state management (draft → experimental → certified → deprecated)
- Rich metadata and capability indexing
- Integration with evaluation results and quality metrics
- API-first design for programmatic access
- Multi-region replication for data residency compliance

---

## Architecture Principles

### 1. Agents as First-Class Assets

Agents are treated as versioned, immutable artifacts similar to container images or code packages:

```yaml
agent_as_asset:
  identity: "Unique agent ID + version"
  immutability: "Published versions cannot be modified"
  versioning: "Semantic versioning (major.minor.patch)"
  metadata: "Rich descriptive metadata"
  provenance: "Full audit trail of creation and modification"
  dependencies: "Explicit declaration of runtime requirements"
```

### 2. Registry Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Registry API                        │
│  (RESTful + gRPC for publish, search, promote, deprecate)   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                  Registry Core Services                      │
├──────────────────────────────────────────────────────────────┤
│  • Metadata Service       • Version Management               │
│  • Search & Discovery     • Lifecycle State Machine          │
│  • Governance Engine      • Promotion Pipeline               │
│  • Audit Logging          • Dependency Resolution            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Storage Backend                            │
├──────────────────────────────────────────────────────────────┤
│  • PostgreSQL (metadata, versions, audit logs)               │
│  • S3/Blob Storage (agent artifacts, evaluation results)     │
│  • Redis (search index, hot metadata cache)                  │
│  • Vector DB (semantic search on capabilities)               │
└──────────────────────────────────────────────────────────────┘
```

---

## What Gets Registered

### 1. Agent Core Metadata

```yaml
agent_metadata:
  identity:
    agent_id: "gl-cbam-calculator-v2"
    name: "CBAM Carbon Calculator"
    version: "2.3.1"
    semantic_version:
      major: 2  # Breaking changes
      minor: 3  # New features, backward compatible
      patch: 1  # Bug fixes

  ownership:
    creator: "user@greenlang.ai"
    team: "greenlang/cbam-team"
    tenant_id: "customer-abc-123"
    created_at: "2025-11-15T10:30:00Z"

  description:
    short: "Calculates embedded carbon for CBAM shipments"
    long: |
      Full-featured CBAM carbon intensity calculator supporting
      all CBAM product categories (cement, steel, aluminum, etc.)
      with real-time emission factor resolution and uncertainty
      quantification.

  classification:
    domain: "sustainability.cbam"
    type: "calculator"
    category: "regulatory_compliance"
    tags: ["cbam", "carbon", "eu-regulation", "emissions"]

  capabilities:
    - name: "calculate_carbon_intensity"
      description: "Calculate embedded carbon per unit"
      input_schema: "schemas/cbam_input_v2.json"
      output_schema: "schemas/cbam_output_v2.json"
    - name: "resolve_emission_factors"
      description: "Resolve emission factors for materials"
      input_schema: "schemas/factor_resolution_input.json"
      output_schema: "schemas/factor_resolution_output.json"
```

### 2. Agent Artifacts

```yaml
agent_artifacts:
  code:
    source_repo: "https://github.com/greenlang/agents/cbam-calculator"
    commit_sha: "a1b2c3d4e5f6g7h8i9j0"
    dockerfile: "s3://greenlang-registry/agents/cbam-calculator/2.3.1/Dockerfile"

  container_image:
    registry: "gcr.io/greenlang"
    image: "gcr.io/greenlang/cbam-calculator:2.3.1"
    image_digest: "sha256:abc123..."
    size_mb: 450
    layers: 12
    base_image: "gcr.io/greenlang/agent-base:1.2.0"

  schemas:
    input_schema: "s3://greenlang-registry/schemas/cbam_input_v2.json"
    output_schema: "s3://greenlang-registry/schemas/cbam_output_v2.json"
    config_schema: "s3://greenlang-registry/schemas/cbam_config_v2.json"

  documentation:
    readme: "s3://greenlang-registry/agents/cbam-calculator/2.3.1/README.md"
    api_docs: "s3://greenlang-registry/agents/cbam-calculator/2.3.1/api-docs/"
    user_guide: "s3://greenlang-registry/agents/cbam-calculator/2.3.1/user-guide.pdf"
    runbook: "s3://greenlang-registry/agents/cbam-calculator/2.3.1/runbook.md"
```

### 3. Runtime Requirements

```yaml
runtime_requirements:
  compute:
    cpu_request: "500m"
    cpu_limit: "2000m"
    memory_request: "512Mi"
    memory_limit: "2Gi"
    gpu_required: false

  dependencies:
    services:
      - name: "greenlang.db"
        version: ">=14.0"
        required: true
      - name: "greenlang.cache"
        version: ">=7.0"
        required: true
      - name: "factor_broker"
        version: ">=2.1"
        required: true

    llm_providers:
      - provider: "anthropic"
        models: ["claude-sonnet-4"]
        required: true
      - provider: "openai"
        models: ["gpt-4"]
        required: false

    python_packages:
      - "greenlang-sdk>=2.3.0"
      - "pydantic>=2.0"
      - "pandas>=2.0"

  environment:
    required_env_vars:
      - "DATABASE_URL"
      - "REDIS_URL"
      - "ANTHROPIC_API_KEY"
    optional_env_vars:
      - "OPENAI_API_KEY"
      - "LOG_LEVEL"

  networking:
    ingress_required: true
    egress_required: true
    allowed_egress:
      - "api.anthropic.com"
      - "api.openai.com"
      - "factor-db.greenlang.ai"
```

### 4. Evaluation Results

```yaml
evaluation_results:
  evaluation_run_id: "eval-run-2025-11-15-001"
  evaluated_at: "2025-11-15T12:00:00Z"
  evaluator_version: "1.5.0"

  performance_metrics:
    latency_p50_ms: 120
    latency_p95_ms: 450
    latency_p99_ms: 850
    throughput_per_sec: 1200
    error_rate: 0.003
    memory_peak_mb: 450

  quality_metrics:
    accuracy: 0.98
    precision: 0.96
    recall: 0.94
    f1_score: 0.95

  compliance_checks:
    schema_validation: "PASS"
    security_scan: "PASS"
    license_check: "PASS"
    dependency_audit: "PASS"

  test_results:
    unit_tests:
      passed: 245
      failed: 0
      coverage: 0.92
    integration_tests:
      passed: 38
      failed: 0
    e2e_tests:
      passed: 12
      failed: 0

  certification_status:
    certified: true
    certification_level: "production"
    certification_date: "2025-11-15"
    certified_by: "qa-team@greenlang.ai"
    expires_at: "2026-11-15"
```

### 5. Lifecycle State

```yaml
lifecycle_state:
  current_state: "certified"
  state_history:
    - state: "draft"
      entered_at: "2025-11-01T08:00:00Z"
      duration_hours: 120
    - state: "experimental"
      entered_at: "2025-11-06T08:00:00Z"
      duration_hours: 72
      experimental_users: 5
      experimental_requests: 50000
    - state: "certified"
      entered_at: "2025-11-15T12:00:00Z"
      certified_by: "qa-team@greenlang.ai"

  promotion_criteria_met:
    evaluation_passed: true
    security_scan_passed: true
    min_experimental_requests: true  # >10k requests
    min_experimental_users: true     # >3 users
    error_rate_below_threshold: true # <1%
    performance_slos_met: true

  deprecation_info:
    deprecated: false
    deprecation_date: null
    replacement_version: null
    sunset_date: null
```

### 6. Usage Analytics

```yaml
usage_analytics:
  total_deployments: 42
  active_deployments: 38
  total_requests_30d: 15000000
  unique_tenants_30d: 25

  request_distribution:
    last_24h: 500000
    last_7d: 3500000
    last_30d: 15000000

  error_statistics:
    error_rate_30d: 0.004
    top_errors:
      - error_type: "FactorNotFound"
        count: 45000
        percentage: 0.003
      - error_type: "ValidationError"
        count: 15000
        percentage: 0.001

  performance_metrics_30d:
    p50_latency_ms: 125
    p95_latency_ms: 460
    p99_latency_ms: 890

  top_tenants:
    - tenant_id: "customer-xyz-456"
      requests_30d: 5000000
      error_rate: 0.002
    - tenant_id: "customer-abc-123"
      requests_30d: 3000000
      error_rate: 0.003
```

---

## Storage Backend Architecture

### 1. PostgreSQL Schema

```sql
-- Agent metadata and versions
CREATE TABLE agents (
    agent_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    domain VARCHAR(100),
    type VARCHAR(50),
    created_by VARCHAR(255),
    team VARCHAR(255),
    tenant_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_tenant (tenant_id),
    INDEX idx_domain (domain),
    INDEX idx_type (type)
);

CREATE TABLE agent_versions (
    version_id VARCHAR(255) PRIMARY KEY,
    agent_id VARCHAR(255) REFERENCES agents(agent_id),
    version VARCHAR(50) NOT NULL,
    semantic_version JSONB,
    lifecycle_state VARCHAR(50),
    container_image VARCHAR(500),
    image_digest VARCHAR(100),
    metadata JSONB,
    runtime_requirements JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    published_at TIMESTAMP,
    deprecated_at TIMESTAMP,
    UNIQUE (agent_id, version),
    INDEX idx_agent (agent_id),
    INDEX idx_state (lifecycle_state),
    INDEX idx_version (agent_id, version)
);

-- Evaluation results
CREATE TABLE evaluation_results (
    evaluation_id VARCHAR(255) PRIMARY KEY,
    version_id VARCHAR(255) REFERENCES agent_versions(version_id),
    evaluated_at TIMESTAMP,
    evaluator_version VARCHAR(50),
    performance_metrics JSONB,
    quality_metrics JSONB,
    compliance_checks JSONB,
    test_results JSONB,
    certification_status JSONB,
    INDEX idx_version (version_id),
    INDEX idx_evaluated_at (evaluated_at)
);

-- Lifecycle state transitions
CREATE TABLE state_transitions (
    transition_id SERIAL PRIMARY KEY,
    version_id VARCHAR(255) REFERENCES agent_versions(version_id),
    from_state VARCHAR(50),
    to_state VARCHAR(50),
    transitioned_at TIMESTAMP DEFAULT NOW(),
    transitioned_by VARCHAR(255),
    reason TEXT,
    metadata JSONB,
    INDEX idx_version (version_id),
    INDEX idx_transition (transitioned_at)
);

-- Usage analytics
CREATE TABLE usage_metrics (
    metric_id SERIAL PRIMARY KEY,
    version_id VARCHAR(255) REFERENCES agent_versions(version_id),
    tenant_id VARCHAR(255),
    timestamp TIMESTAMP,
    request_count INTEGER,
    error_count INTEGER,
    latency_p50_ms INTEGER,
    latency_p95_ms INTEGER,
    latency_p99_ms INTEGER,
    INDEX idx_version_time (version_id, timestamp),
    INDEX idx_tenant_time (tenant_id, timestamp)
);

-- Audit logs
CREATE TABLE audit_logs (
    log_id SERIAL PRIMARY KEY,
    version_id VARCHAR(255),
    action VARCHAR(100),
    performed_by VARCHAR(255),
    tenant_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW(),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    INDEX idx_version (version_id),
    INDEX idx_tenant (tenant_id),
    INDEX idx_timestamp (timestamp)
);

-- Governance policies
CREATE TABLE governance_policies (
    policy_id VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(255),
    policy_type VARCHAR(50),
    policy_rules JSONB,
    active BOOLEAN DEFAULT true,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_tenant (tenant_id),
    INDEX idx_type (policy_type)
);
```

### 2. S3 Bucket Structure

```
s3://greenlang-agent-registry/
├── agents/
│   ├── {agent_id}/
│   │   ├── {version}/
│   │   │   ├── Dockerfile
│   │   │   ├── README.md
│   │   │   ├── api-docs/
│   │   │   ├── schemas/
│   │   │   │   ├── input.json
│   │   │   │   ├── output.json
│   │   │   │   └── config.json
│   │   │   ├── evaluation/
│   │   │   │   ├── test-results.json
│   │   │   │   ├── performance-report.json
│   │   │   │   └── security-scan.json
│   │   │   └── artifacts/
│   │   │       ├── source.tar.gz
│   │   │       └── dependencies.txt
├── schemas/
│   ├── global/
│   │   ├── cbam_input_v2.json
│   │   └── cbam_output_v2.json
├── evaluations/
│   ├── {evaluation_run_id}/
│   │   ├── results.json
│   │   ├── logs/
│   │   └── reports/
└── backups/
    └── {date}/
```

### 3. Redis Cache Structure

```
# Hot metadata cache (TTL: 1 hour)
registry:agent:{agent_id}:latest -> agent_metadata_json
registry:agent:{agent_id}:versions -> [list of versions]
registry:version:{version_id} -> version_metadata_json

# Search index cache (TTL: 5 minutes)
registry:search:{query_hash} -> [list of agent_ids]
registry:popular:agents -> sorted_set(agent_id, request_count)

# Lock management (TTL: 30 seconds)
registry:lock:publish:{agent_id}:{version} -> publisher_id
registry:lock:promote:{version_id} -> promoter_id
```

### 4. Vector Database (Semantic Search)

```yaml
vector_db_schema:
  collection: "agent_capabilities"
  vectors:
    - id: "{agent_id}:{version}"
      embedding: [1536-dimensional vector from description + capabilities]
      metadata:
        agent_id: "gl-cbam-calculator-v2"
        version: "2.3.1"
        name: "CBAM Carbon Calculator"
        domain: "sustainability.cbam"
        tags: ["cbam", "carbon"]

  search_queries:
    - query: "Calculate embedded carbon for aluminum shipments"
      similarity_threshold: 0.85
      results:
        - agent_id: "gl-cbam-calculator-v2"
          similarity: 0.92
        - agent_id: "gl-carbon-intensity-v1"
          similarity: 0.87
```

---

## API Surface

The Registry exposes a comprehensive API for all operations:

### 1. Core APIs

```yaml
registry_apis:
  publish:
    endpoint: "POST /api/v1/registry/agents"
    description: "Publish a new agent version"

  search:
    endpoint: "GET /api/v1/registry/agents?query={query}"
    description: "Search for agents"

  get_agent:
    endpoint: "GET /api/v1/registry/agents/{agent_id}"
    description: "Get agent metadata"

  get_version:
    endpoint: "GET /api/v1/registry/agents/{agent_id}/versions/{version}"
    description: "Get specific version"

  promote:
    endpoint: "POST /api/v1/registry/agents/{agent_id}/versions/{version}/promote"
    description: "Promote agent to next lifecycle state"

  deprecate:
    endpoint: "POST /api/v1/registry/agents/{agent_id}/versions/{version}/deprecate"
    description: "Deprecate an agent version"

  list_versions:
    endpoint: "GET /api/v1/registry/agents/{agent_id}/versions"
    description: "List all versions of an agent"
```

### 2. gRPC Service Definition

```protobuf
service AgentRegistry {
  // Agent lifecycle
  rpc PublishAgent(PublishAgentRequest) returns (PublishAgentResponse);
  rpc GetAgent(GetAgentRequest) returns (Agent);
  rpc ListAgents(ListAgentsRequest) returns (ListAgentsResponse);
  rpc SearchAgents(SearchAgentsRequest) returns (SearchAgentsResponse);

  // Version management
  rpc GetVersion(GetVersionRequest) returns (AgentVersion);
  rpc ListVersions(ListVersionsRequest) returns (ListVersionsResponse);
  rpc PromoteVersion(PromoteVersionRequest) returns (PromoteVersionResponse);
  rpc DeprecateVersion(DeprecateVersionRequest) returns (DeprecateVersionResponse);

  // Governance
  rpc CheckPolicy(CheckPolicyRequest) returns (CheckPolicyResponse);
  rpc ListPolicies(ListPoliciesRequest) returns (ListPoliciesResponse);

  // Analytics
  rpc GetUsageMetrics(GetUsageMetricsRequest) returns (UsageMetrics);
  rpc GetPopularAgents(GetPopularAgentsRequest) returns (PopularAgentsResponse);
}
```

---

## Multi-Region Replication

For data residency compliance, the Registry supports multi-region deployment:

```yaml
multi_region_strategy:
  regions:
    - region: "eu-central-1"
      primary: true
      serves: ["EU tenants"]

    - region: "us-east-1"
      primary: false
      serves: ["US tenants"]
      replication: "async from eu-central-1"

    - region: "cn-north-1"
      primary: false
      serves: ["China tenants"]
      isolated: true  # No replication

  replication_policy:
    metadata: "Cross-region async replication"
    artifacts: "Regional S3 buckets with on-demand replication"
    audit_logs: "Region-specific, no cross-border transfer"
```

---

## Security Considerations

### 1. Access Control

```yaml
security_model:
  authentication:
    - API keys with scopes
    - OAuth 2.0 / OIDC
    - mTLS for service-to-service

  authorization:
    - Tenant isolation (row-level security)
    - RBAC for registry operations
    - Policy-based access control

  encryption:
    - TLS 1.3 for all API calls
    - Encrypted S3 buckets (AES-256)
    - Encrypted database (at rest and in transit)
```

### 2. Audit Logging

All registry operations are logged:
- Agent publish/update/delete
- Version promotion/deprecation
- Policy changes
- Access attempts (successful and failed)

---

## Performance Targets

```yaml
registry_slos:
  api_latency:
    get_agent: "< 50ms (p95)"
    search: "< 200ms (p95)"
    publish: "< 2 seconds (p95)"

  availability: "99.99%"

  throughput:
    read_ops: "> 10,000 req/sec"
    write_ops: "> 500 req/sec"

  consistency:
    metadata: "Strong consistency within region"
    cross_region: "Eventual consistency (< 5 seconds)"
```

---

## Integration Points

The Registry integrates with:

1. **Agent Factory Build System** - Receives newly built agents
2. **Evaluation Framework** - Stores evaluation results
3. **Runtime Environment** - Provides agent metadata for deployment
4. **Governance System** - Enforces policies on agent usage
5. **Monitoring System** - Collects usage analytics
6. **CI/CD Pipeline** - Automates publish and promotion

---

## Related Documentation

- [Registry API Specification](../api-specs/00-REGISTRY_API.md)
- [Agent Lifecycle Management](../lifecycle/00-AGENT_LIFECYCLE.md)
- [Governance Controls](../governance/00-GOVERNANCE_CONTROLS.md)
- [Runtime Architecture](01-RUNTIME_ARCHITECTURE.md)

---

**Questions or feedback?**
- Slack: #agent-registry
- Email: devops@greenlang.ai
- Wiki: https://wiki.greenlang.ai/registry
