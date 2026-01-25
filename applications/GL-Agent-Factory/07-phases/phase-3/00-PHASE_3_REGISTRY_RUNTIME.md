# Phase 3: Agent Registry and Runtime Governance

**Version:** 1.0
**Date:** 2025-12-03
**Product Manager:** GL-ProductManager
**Phase Duration:** 12 weeks (May 24 - Aug 15, 2026)
**Status:** Planned

---

## Executive Summary

Phase 3 transforms agents from generated artifacts into first-class, governed assets. This phase delivers the Agent Registry (discovery and versioning), Lifecycle Management (CRUD and deployment), and Runtime Governance (policy enforcement and monitoring).

**Phase Goal:** Treat agents as first-class, governed assets with full lifecycle management and runtime governance.

**Key Outcome:** 50+ agents deployed via registry with governance policies enforced, 99.9% uptime, and complete audit trails.

---

## Objectives

### Primary Objectives

1. **Build Agent Registry** - Central repository for agent discovery, versioning, and metadata
2. **Implement Lifecycle Management** - Create, update, deprecate, and retire agents
3. **Create Runtime Governance** - Policy enforcement, access control, and compliance monitoring
4. **Establish Observability** - Metrics, logging, tracing, and alerting for all agents
5. **Deploy 50 Agents** - Prove registry and governance at scale

### Non-Objectives (Out of Scope for Phase 3)

- Self-service UI for agent creation (Phase 4)
- External partner access (Phase 4)
- Marketplace functionality (Phase 4)
- Mobile applications

---

## Technical Scope

### Component 1: Agent Registry

**Description:** Central repository for storing, discovering, and managing agent metadata and versions.

**Registry Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Registry                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      Registry API                               ││
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  ││
│  │  │   Create   │  │   Read     │  │   Update   │  │  Delete  │  ││
│  │  │   Agent    │  │   Agent    │  │   Agent    │  │  Agent   │  ││
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────┘  ││
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  ││
│  │  │   Search   │  │   List     │  │   Deploy   │  │ Rollback │  ││
│  │  │   Agents   │  │   Versions │  │   Agent    │  │  Agent   │  ││
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────┘  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Storage Layer                                ││
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  ││
│  │  │ Metadata Store │  │ Artifact Store │  │ Certificate Store│  ││
│  │  │  (PostgreSQL)  │  │    (S3)        │  │   (PostgreSQL)   │  ││
│  │  └────────────────┘  └────────────────┘  └──────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Index Layer                                  ││
│  │  ┌────────────────────────────────────────────────────────────┐ ││
│  │  │ Search Index (Elasticsearch) - Full-text + Faceted Search │ ││
│  │  └────────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

**Registry API:**

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class AgentStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

class AgentVersion(BaseModel):
    """Represents a specific version of an agent."""
    version: str
    status: AgentStatus
    spec_hash: str
    artifact_uri: str
    certificate_id: Optional[str]
    created_at: datetime
    created_by: str
    changelog: str

class RegistryEntry(BaseModel):
    """Complete registry entry for an agent."""
    agent_id: str
    name: str
    description: str
    owner_team: str
    type: str
    regulatory_scope: List[str]
    tags: List[str]
    versions: List[AgentVersion]
    current_version: str
    created_at: datetime
    updated_at: datetime
    deployment_count: int
    usage_stats: Dict[str, Any]

class SearchQuery(BaseModel):
    """Search query for finding agents."""
    query: Optional[str]
    tags: Optional[List[str]]
    regulatory_scope: Optional[List[str]]
    owner_team: Optional[str]
    status: Optional[AgentStatus]
    type: Optional[str]
    page: int = 1
    page_size: int = 20

class SearchResult(BaseModel):
    """Search result with pagination."""
    total: int
    page: int
    page_size: int
    agents: List[RegistryEntry]

class AgentRegistry:
    """Central registry for agent management."""

    def __init__(self, db: Database, artifact_store: ArtifactStore, search_index: SearchIndex):
        self._db = db
        self._artifact_store = artifact_store
        self._search_index = search_index

    async def register_agent(
        self,
        spec: AgentSpec,
        artifact_path: str,
        certificate: Optional[Certificate],
        created_by: str
    ) -> RegistryEntry:
        """Register a new agent or new version in the registry."""
        existing = await self._db.get_agent(spec.agent_id)

        if existing:
            # Add new version
            version = AgentVersion(
                version=spec.version,
                status=AgentStatus.DRAFT if not certificate else AgentStatus.ACTIVE,
                spec_hash=self._hash_spec(spec),
                artifact_uri=await self._artifact_store.upload(artifact_path),
                certificate_id=certificate.certificate_id if certificate else None,
                created_at=datetime.utcnow(),
                created_by=created_by,
                changelog=spec.metadata.get("changelog", "")
            )
            existing.versions.append(version)
            existing.updated_at = datetime.utcnow()
            await self._db.update_agent(existing)
            await self._search_index.update(existing)
            return existing
        else:
            # Create new agent entry
            entry = RegistryEntry(
                agent_id=spec.agent_id,
                name=spec.name,
                description=spec.description,
                owner_team=spec.metadata.get("team", "unknown"),
                type=spec.type,
                regulatory_scope=spec.metadata.get("regulatory_scope", []),
                tags=spec.metadata.get("tags", []),
                versions=[AgentVersion(...)],
                current_version=spec.version,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                deployment_count=0,
                usage_stats={}
            )
            await self._db.create_agent(entry)
            await self._search_index.index(entry)
            return entry

    async def get_agent(self, agent_id: str, version: Optional[str] = None) -> Optional[RegistryEntry]:
        """Get agent by ID, optionally specific version."""
        entry = await self._db.get_agent(agent_id)
        if entry and version:
            entry.current_version = version
        return entry

    async def search_agents(self, query: SearchQuery) -> SearchResult:
        """Search for agents using full-text and faceted search."""
        return await self._search_index.search(query)

    async def update_status(
        self,
        agent_id: str,
        version: str,
        new_status: AgentStatus,
        reason: str,
        updated_by: str
    ) -> RegistryEntry:
        """Update agent version status (e.g., deprecate, retire)."""
        entry = await self._db.get_agent(agent_id)
        for v in entry.versions:
            if v.version == version:
                v.status = new_status
                break
        entry.updated_at = datetime.utcnow()
        await self._db.update_agent(entry)
        await self._audit_log.record(
            action="status_change",
            agent_id=agent_id,
            version=version,
            new_status=new_status,
            reason=reason,
            actor=updated_by
        )
        return entry

    async def list_versions(self, agent_id: str) -> List[AgentVersion]:
        """List all versions of an agent."""
        entry = await self._db.get_agent(agent_id)
        return entry.versions if entry else []

    async def get_artifact(self, agent_id: str, version: str) -> bytes:
        """Download agent artifact for deployment."""
        entry = await self._db.get_agent(agent_id)
        version_info = next((v for v in entry.versions if v.version == version), None)
        if not version_info:
            raise AgentNotFoundError(f"Version {version} not found")
        return await self._artifact_store.download(version_info.artifact_uri)
```

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agents` | Register new agent |
| GET | `/agents/{id}` | Get agent by ID |
| GET | `/agents/{id}/versions` | List all versions |
| GET | `/agents/{id}/versions/{version}` | Get specific version |
| PATCH | `/agents/{id}/versions/{version}` | Update version status |
| GET | `/agents/search` | Search agents |
| POST | `/agents/{id}/deploy` | Trigger deployment |
| POST | `/agents/{id}/rollback` | Rollback to previous version |
| DELETE | `/agents/{id}` | Soft delete (retire) |

**Deliverables:**
- Registry API service
- PostgreSQL schema for metadata
- S3 integration for artifacts
- Elasticsearch integration for search
- CLI: `greenlang-registry agent list/get/search/register`
- API documentation

**Owner:** Platform Team
**Support:** DevOps (infrastructure), AI/Agent (integration)

---

### Component 2: Lifecycle Management

**Description:** Manages the complete lifecycle of agents from creation to retirement.

**Lifecycle States:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Agent Lifecycle States                            │
│                                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐   │
│  │  DRAFT  │───►│ ACTIVE  │───►│DEPRECATED│───►│    RETIRED     │   │
│  │         │    │         │    │         │    │                 │   │
│  │ Created │    │ Deployed│    │ Sunset  │    │ No longer       │   │
│  │ Testing │    │ Live    │    │ Warning │    │ available       │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────────────┘   │
│       │              │              │                               │
│       │              │              │                               │
│       ▼              ▼              ▼                               │
│  Can promote    Can rollback   Can re-activate                     │
│  to ACTIVE      to previous    (with approval)                     │
│                 ACTIVE version                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Lifecycle Manager:**

```python
class LifecycleManager:
    """Manages agent lifecycle transitions."""

    def __init__(self, registry: AgentRegistry, deployer: Deployer, governance: GovernanceEngine):
        self._registry = registry
        self._deployer = deployer
        self._governance = governance

    async def promote_to_active(
        self,
        agent_id: str,
        version: str,
        deployment_config: DeploymentConfig,
        promoted_by: str
    ) -> DeploymentResult:
        """Promote draft agent to active (deploy)."""
        # Check governance policies
        policy_check = await self._governance.check_promotion_policy(agent_id, version)
        if not policy_check.allowed:
            raise PolicyViolationError(policy_check.violations)

        # Verify certification
        entry = await self._registry.get_agent(agent_id)
        version_info = self._get_version(entry, version)
        if not version_info.certificate_id:
            raise CertificationRequiredError("Agent must be certified before deployment")

        # Deploy
        deployment_result = await self._deployer.deploy(
            agent_id=agent_id,
            version=version,
            config=deployment_config
        )

        if deployment_result.success:
            # Update status in registry
            await self._registry.update_status(
                agent_id, version, AgentStatus.ACTIVE,
                reason="Deployed to production",
                updated_by=promoted_by
            )

        return deployment_result

    async def deprecate(
        self,
        agent_id: str,
        version: str,
        sunset_date: datetime,
        replacement_id: Optional[str],
        deprecated_by: str
    ) -> RegistryEntry:
        """Mark agent version as deprecated with sunset date."""
        await self._registry.update_status(
            agent_id, version, AgentStatus.DEPRECATED,
            reason=f"Deprecated. Sunset: {sunset_date}. Replacement: {replacement_id or 'none'}",
            updated_by=deprecated_by
        )

        # Schedule notification to users
        await self._notify_deprecation(agent_id, version, sunset_date, replacement_id)

        return await self._registry.get_agent(agent_id)

    async def retire(
        self,
        agent_id: str,
        version: str,
        retired_by: str
    ) -> RegistryEntry:
        """Retire agent version (no longer available)."""
        # Verify no active deployments
        active_deployments = await self._deployer.list_deployments(agent_id, version)
        if active_deployments:
            raise ActiveDeploymentsError(f"Cannot retire: {len(active_deployments)} active deployments")

        # Undeploy and update status
        await self._deployer.undeploy(agent_id, version)
        await self._registry.update_status(
            agent_id, version, AgentStatus.RETIRED,
            reason="Retired from service",
            updated_by=retired_by
        )

        return await self._registry.get_agent(agent_id)

    async def rollback(
        self,
        agent_id: str,
        target_version: str,
        reason: str,
        rolled_back_by: str
    ) -> DeploymentResult:
        """Rollback to previous version."""
        # Verify target version exists and is deployable
        entry = await self._registry.get_agent(agent_id)
        target = self._get_version(entry, target_version)
        if target.status == AgentStatus.RETIRED:
            raise InvalidRollbackError("Cannot rollback to retired version")

        # Deploy target version
        current_version = entry.current_version
        result = await self._deployer.deploy(
            agent_id=agent_id,
            version=target_version,
            config=await self._deployer.get_config(agent_id, current_version)
        )

        if result.success:
            # Update registry
            entry.current_version = target_version
            await self._registry.update_agent(entry)
            await self._audit_log.record(
                action="rollback",
                agent_id=agent_id,
                from_version=current_version,
                to_version=target_version,
                reason=reason,
                actor=rolled_back_by
            )

        return result
```

**Deliverables:**
- `LifecycleManager` class
- Promotion workflow with approval gates
- Deprecation notification system
- Rollback mechanism with safety checks
- CLI: `greenlang-lifecycle promote/deprecate/retire/rollback`

**Owner:** Platform Team
**Support:** DevOps (deployment), AI/Agent (agent loading)

---

### Component 3: Runtime Governance

**Description:** Policy enforcement and compliance monitoring for running agents.

**Governance Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Runtime Governance Engine                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      Policy Engine                              ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       ││
│  │  │ Access Control│  │ Rate Limiting │  │  Compliance   │       ││
│  │  │    Policies   │  │   Policies    │  │   Policies    │       ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘       ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       ││
│  │  │  Deployment   │  │  Data Access  │  │   Retention   │       ││
│  │  │   Policies    │  │   Policies    │  │   Policies    │       ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘       ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                   Enforcement Layer                             ││
│  │  ┌───────────────────────────────────────────────────────────┐  ││
│  │  │  Pre-Execution   │   During Execution   │  Post-Execution │  ││
│  │  │  • Auth check    │   • Token limits     │  • Audit log    │  ││
│  │  │  • Rate check    │   • Data masking     │  • Compliance   │  ││
│  │  │  • Policy check  │   • Timeout          │  • Metrics      │  ││
│  │  └───────────────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                   Monitoring & Alerting                         ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       ││
│  │  │ Policy        │  │ Compliance    │  │ Anomaly       │       ││
│  │  │ Violations    │  │ Reports       │  │ Detection     │       ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘       ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

**Governance Engine:**

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum
from abc import ABC, abstractmethod

class PolicyType(str, Enum):
    ACCESS_CONTROL = "access_control"
    RATE_LIMITING = "rate_limiting"
    COMPLIANCE = "compliance"
    DEPLOYMENT = "deployment"
    DATA_ACCESS = "data_access"
    RETENTION = "retention"

class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"

class Policy(BaseModel):
    """Base policy definition."""
    policy_id: str
    type: PolicyType
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement: PolicyDecision  # Default decision if no rules match
    enabled: bool

class PolicyCheckResult(BaseModel):
    """Result of policy evaluation."""
    allowed: bool
    decision: PolicyDecision
    violations: List[str]
    warnings: List[str]
    policy_id: str
    evaluated_rules: int

class BasePolicy(ABC):
    """Abstract base class for policy implementations."""

    @abstractmethod
    async def evaluate(
        self,
        context: Dict[str, Any]
    ) -> PolicyCheckResult:
        pass

class AccessControlPolicy(BasePolicy):
    """Controls who can access which agents."""

    async def evaluate(self, context: Dict[str, Any]) -> PolicyCheckResult:
        user = context.get("user")
        agent_id = context.get("agent_id")
        action = context.get("action")

        # Check RBAC
        if not await self._has_permission(user, agent_id, action):
            return PolicyCheckResult(
                allowed=False,
                decision=PolicyDecision.DENY,
                violations=[f"User {user} lacks permission {action} on {agent_id}"],
                warnings=[],
                policy_id=self.policy_id,
                evaluated_rules=1
            )

        return PolicyCheckResult(allowed=True, ...)

class RateLimitingPolicy(BasePolicy):
    """Enforces rate limits on agent invocations."""

    async def evaluate(self, context: Dict[str, Any]) -> PolicyCheckResult:
        user = context.get("user")
        agent_id = context.get("agent_id")

        # Check rate limit
        current_rate = await self._get_current_rate(user, agent_id)
        limit = self._get_limit(user, agent_id)

        if current_rate >= limit:
            return PolicyCheckResult(
                allowed=False,
                decision=PolicyDecision.DENY,
                violations=[f"Rate limit exceeded: {current_rate}/{limit}"],
                warnings=[],
                policy_id=self.policy_id,
                evaluated_rules=1
            )

        if current_rate >= limit * 0.8:
            return PolicyCheckResult(
                allowed=True,
                decision=PolicyDecision.WARN,
                violations=[],
                warnings=[f"Approaching rate limit: {current_rate}/{limit}"],
                policy_id=self.policy_id,
                evaluated_rules=1
            )

        return PolicyCheckResult(allowed=True, ...)

class CompliancePolicy(BasePolicy):
    """Ensures agents meet compliance requirements."""

    async def evaluate(self, context: Dict[str, Any]) -> PolicyCheckResult:
        agent_id = context.get("agent_id")

        # Check certification status
        agent = await self._registry.get_agent(agent_id)
        version = self._get_active_version(agent)

        if not version.certificate_id:
            return PolicyCheckResult(
                allowed=False,
                decision=PolicyDecision.DENY,
                violations=["Agent not certified"],
                warnings=[],
                policy_id=self.policy_id,
                evaluated_rules=1
            )

        # Check certificate expiration
        cert = await self._get_certificate(version.certificate_id)
        if cert.expires_at < datetime.utcnow():
            return PolicyCheckResult(
                allowed=False,
                decision=PolicyDecision.DENY,
                violations=["Agent certification expired"],
                warnings=[],
                policy_id=self.policy_id,
                evaluated_rules=1
            )

        return PolicyCheckResult(allowed=True, ...)

class GovernanceEngine:
    """Orchestrates policy evaluation and enforcement."""

    def __init__(self):
        self._policies: Dict[PolicyType, List[BasePolicy]] = {}
        self._audit_log = AuditLog()

    def register_policy(self, policy: BasePolicy) -> None:
        """Register a policy for enforcement."""
        if policy.type not in self._policies:
            self._policies[policy.type] = []
        self._policies[policy.type].append(policy)

    async def check_policies(
        self,
        policy_types: List[PolicyType],
        context: Dict[str, Any]
    ) -> List[PolicyCheckResult]:
        """Evaluate multiple policies against context."""
        results = []
        for policy_type in policy_types:
            policies = self._policies.get(policy_type, [])
            for policy in policies:
                if policy.enabled:
                    result = await policy.evaluate(context)
                    results.append(result)
                    await self._audit_log.record_policy_check(policy.policy_id, context, result)
        return results

    async def enforce(
        self,
        policy_types: List[PolicyType],
        context: Dict[str, Any]
    ) -> bool:
        """Enforce policies, raising exception on violation."""
        results = await self.check_policies(policy_types, context)
        violations = [r for r in results if not r.allowed]

        if violations:
            raise PolicyViolationError(
                f"Policy violations: {[v.violations for v in violations]}"
            )

        return True

    async def get_compliance_report(
        self,
        agent_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> ComplianceReport:
        """Generate compliance report for an agent."""
        violations = await self._audit_log.get_violations(agent_id, start_date, end_date)
        invocations = await self._audit_log.get_invocations(agent_id, start_date, end_date)

        return ComplianceReport(
            agent_id=agent_id,
            period_start=start_date,
            period_end=end_date,
            total_invocations=len(invocations),
            total_violations=len(violations),
            violation_rate=len(violations) / len(invocations) if invocations else 0,
            violations_by_type=self._group_violations(violations),
            compliance_score=self._calculate_compliance_score(violations, invocations)
        )
```

**Deliverables:**
- `GovernanceEngine` class with policy registry
- Access control policy (RBAC)
- Rate limiting policy
- Compliance policy (certification checks)
- Deployment policy (approval workflow)
- Audit log integration
- Compliance reporting
- CLI: `greenlang-governance policy list/create/enable/disable`

**Owner:** DevOps/Security Team
**Support:** Platform (integration), Climate Science (compliance rules)

---

### Component 4: Observability

**Description:** Comprehensive monitoring, logging, tracing, and alerting for all agents.

**Observability Stack:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Observability Infrastructure                     │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Metrics (Prometheus)                      │  │
│  │  • Invocation count        • Error rate                       │  │
│  │  • Latency (p50/p95/p99)   • Token usage                      │  │
│  │  • Cache hit rate          • Concurrent executions            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Logging (Elasticsearch)                   │  │
│  │  • Structured logs         • Input/output samples             │  │
│  │  • Error stack traces      • Audit events                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Tracing (Jaeger)                          │  │
│  │  • End-to-end traces       • Tool invocation spans            │  │
│  │  • Cross-service calls     • LLM call spans                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Alerting (Grafana/PagerDuty)              │  │
│  │  • SLO violations          • Error rate spikes                │  │
│  │  • Latency degradation     • Certificate expiration           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Dashboards (Grafana)                      │  │
│  │  • Agent health            • Business metrics                 │  │
│  │  • Governance compliance   • Cost tracking                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| `agent_invocations_total` | Total invocations per agent | N/A (counter) |
| `agent_invocation_duration_seconds` | Latency histogram | p99 < 5s |
| `agent_errors_total` | Error count by type | <1% rate |
| `agent_tokens_used_total` | LLM token consumption | Track |
| `agent_cache_hit_ratio` | Cache effectiveness | >50% |
| `agent_concurrent_executions` | Current load | <100 |
| `agent_certificate_expiry_days` | Days until cert expires | >30 |
| `governance_violations_total` | Policy violations | <0.1% rate |

**Observability Client:**

```python
from typing import Dict, Any, Optional
import structlog
from opentelemetry import trace
from opentelemetry.metrics import get_meter
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger()
tracer = trace.get_tracer("greenlang.agents")
meter = get_meter("greenlang.agents")

class AgentObservability:
    """Observability integration for agents."""

    def __init__(self, agent_id: str, version: str):
        self.agent_id = agent_id
        self.version = version

        # Metrics
        self.invocations = Counter(
            "agent_invocations_total",
            "Total agent invocations",
            ["agent_id", "version", "status"]
        )
        self.latency = Histogram(
            "agent_invocation_duration_seconds",
            "Agent invocation latency",
            ["agent_id", "version"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        self.tokens = Counter(
            "agent_tokens_used_total",
            "LLM tokens consumed",
            ["agent_id", "version", "model"]
        )

    def record_invocation(
        self,
        status: str,
        duration_seconds: float,
        tokens_used: int,
        model: str
    ) -> None:
        """Record metrics for an invocation."""
        self.invocations.labels(
            agent_id=self.agent_id,
            version=self.version,
            status=status
        ).inc()
        self.latency.labels(
            agent_id=self.agent_id,
            version=self.version
        ).observe(duration_seconds)
        self.tokens.labels(
            agent_id=self.agent_id,
            version=self.version,
            model=model
        ).inc(tokens_used)

    def create_span(self, operation: str) -> trace.Span:
        """Create a tracing span for an operation."""
        return tracer.start_span(
            f"{self.agent_id}.{operation}",
            attributes={
                "agent.id": self.agent_id,
                "agent.version": self.version
            }
        )

    def log_event(
        self,
        event: str,
        level: str = "info",
        **kwargs
    ) -> None:
        """Log a structured event."""
        log_fn = getattr(logger, level)
        log_fn(
            event,
            agent_id=self.agent_id,
            version=self.version,
            **kwargs
        )
```

**Deliverables:**
- `AgentObservability` class with all integrations
- Prometheus metrics exporter
- Elasticsearch log shipping
- Jaeger tracing integration
- Grafana dashboards (5 dashboards)
- PagerDuty alert rules (10 rules)
- Documentation

**Owner:** DevOps/SRE Team
**Support:** ML Platform (integration), Platform (API instrumentation)

---

## Deliverables by Team

### Platform Team (Primary Owner - Registry)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Registry API design | 1 week | Week 26 | Pending |
| Registry API implementation | 3 weeks | Week 29 | Pending |
| PostgreSQL schema | 1 week | Week 26 | Pending |
| S3 artifact store | 1 week | Week 27 | Pending |
| Elasticsearch integration | 2 weeks | Week 29 | Pending |
| Lifecycle Manager | 2 weeks | Week 31 | Pending |
| Registry CLI | 1 week | Week 32 | Pending |
| Registry documentation | 1 week | Week 36 | Pending |

### DevOps/Security Team (Primary Owner - Governance)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Governance engine design | 1 week | Week 26 | Pending |
| Policy framework | 2 weeks | Week 28 | Pending |
| Access control policy | 2 weeks | Week 30 | Pending |
| Rate limiting policy | 1 week | Week 30 | Pending |
| Compliance policy | 2 weeks | Week 32 | Pending |
| Audit log system | 2 weeks | Week 30 | Pending |
| Observability infrastructure | 3 weeks | Week 32 | Pending |
| Grafana dashboards | 2 weeks | Week 34 | Pending |
| Alert rules | 1 week | Week 34 | Pending |
| Governance CLI | 1 week | Week 34 | Pending |

### AI/Agent Team (Support)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Agent SDK registry integration | 2 weeks | Week 30 | Pending |
| Agent SDK observability hooks | 1 week | Week 32 | Pending |
| Generate 40 additional agents | 4 weeks | Week 36 | Pending |

### Data Engineering Team (Support)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Audit data pipeline | 2 weeks | Week 30 | Pending |
| Metrics aggregation pipeline | 2 weeks | Week 32 | Pending |
| Compliance reporting | 2 weeks | Week 34 | Pending |

### Climate Science Team (Support)

| Deliverable | Effort | Due | Status |
|-------------|--------|-----|--------|
| Compliance rules definition | 2 weeks | Week 30 | Pending |
| Certify additional agents | 4 weeks | Week 36 | Pending |

---

## Timeline

### Sprint Breakdown (2-Week Sprints)

**Sprint 13 (Weeks 25-26): Foundation**
- Registry API design
- Governance engine design
- Database schema design
- Observability architecture

**Sprint 14 (Weeks 27-28): Core Implementation**
- Registry API implementation (partial)
- Policy framework implementation
- S3 artifact store
- Audit log system

**Sprint 15 (Weeks 29-30): Integration**
- Registry API completion
- Elasticsearch integration
- Access control + Rate limiting policies
- SDK registry integration

**Sprint 16 (Weeks 31-32): Governance**
- Lifecycle Manager
- Compliance policy
- Observability infrastructure
- SDK observability hooks

**Sprint 17 (Weeks 33-34): Scale**
- Generate additional agents (20)
- Grafana dashboards
- Alert rules
- CLI tools

**Sprint 18 (Weeks 35-36): Polish**
- Generate remaining agents (20)
- Certify all agents (50 total)
- Documentation
- Phase 3 exit review preparation

---

## Success Criteria

### Must-Have (Phase Cannot Exit Without)

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Registry operational | 100% | API serving requests |
| Agents registered | 50+ | In registry database |
| Governance enforced | 100% | All policies active |
| Observability | 100% | Metrics, logs, traces flowing |
| Uptime | 99.9% | Registry + deployed agents |
| Audit completeness | 100% | All actions logged |
| Agents deployed | 50+ | Via registry |

### Should-Have

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Search accuracy | 95%+ | Relevant results returned |
| Lifecycle automation | 100% | No manual steps |
| Dashboard coverage | 5 | Grafana dashboards |
| Alert coverage | 10+ | PagerDuty rules |

### Could-Have

| Criteria | Target | Measurement |
|----------|--------|-------------|
| API response time | <100ms | p95 latency |
| Full-text search | 1 | Elasticsearch |
| Self-service deprecation | 1 | Owner-initiated |

---

## Exit Criteria to Phase 4

**All Must Pass to Proceed to Phase 4:**

1. **Registry Production-Ready**
   - [ ] 50+ agents registered and discoverable
   - [ ] All APIs documented and tested
   - [ ] 99.9% uptime achieved
   - [ ] Search returning relevant results

2. **Lifecycle Management Complete**
   - [ ] CRUD operations working
   - [ ] Promotion workflow operational
   - [ ] Deprecation and retirement tested
   - [ ] Rollback mechanism validated

3. **Governance Enforced**
   - [ ] All policies active and enforced
   - [ ] <0.1% violation rate
   - [ ] Compliance reports generating
   - [ ] Audit log complete

4. **Observability Operational**
   - [ ] Metrics flowing to Prometheus
   - [ ] Logs searchable in Elasticsearch
   - [ ] Traces visible in Jaeger
   - [ ] Alerts triggering correctly

5. **Scale Validated**
   - [ ] 50+ agents deployed
   - [ ] Concurrent agent execution tested
   - [ ] No performance degradation

**Phase 4 Approval Contingent On:**
- Business case for self-service validated
- Market demand for partner ecosystem confirmed
- Phase 3 success (50+ agents, 99.9% uptime)

---

## Risks and Mitigations

### Phase 3 Specific Risks

| Risk | Likelihood | Impact | Owner | Mitigation |
|------|------------|--------|-------|------------|
| Registry performance issues | Medium | High | Platform | Load testing; caching; read replicas |
| Policy complexity | Medium | Medium | DevOps/Security | Start simple; iterate |
| Agent deployment failures | Low | High | DevOps | Canary deployments; rollback |
| Observability data volume | High | Medium | SRE | Sampling; retention policies |
| Certification bottleneck | Medium | Medium | Climate Science | Batch certification; more reviewers |

### Mitigation Actions

1. **Performance**
   - Load test with 100 agents, 1000 req/s
   - Implement read replicas for registry
   - Cache frequently accessed agents

2. **Deployment Safety**
   - Canary deployments (10% traffic)
   - Automatic rollback on error spike
   - Health checks before full promotion

3. **Observability Scale**
   - 1% trace sampling in production
   - 7-day log retention (30-day archive)
   - Metric aggregation at 1-minute intervals

---

## Resource Allocation

### Team Allocation by Week

| Team | W25-26 | W27-28 | W29-30 | W31-32 | W33-34 | W35-36 | Total |
|------|--------|--------|--------|--------|--------|--------|-------|
| Platform | 4 | 5 | 5 | 4 | 3 | 3 | 48 FTE-weeks |
| DevOps/Security | 4 | 5 | 5 | 5 | 4 | 3 | 52 FTE-weeks |
| AI/Agent | 2 | 2 | 3 | 3 | 4 | 4 | 36 FTE-weeks |
| Data Engineering | 1 | 2 | 2 | 2 | 2 | 1 | 20 FTE-weeks |
| Climate Science | 0 | 1 | 2 | 2 | 3 | 4 | 24 FTE-weeks |
| **Total** | 11 | 15 | 17 | 16 | 16 | 15 | **180 FTE-weeks** |

---

## Appendices

### Appendix A: Registry API Examples

**Register Agent:**
```bash
POST /api/v1/agents
Content-Type: application/json

{
  "spec_file": "gl-cbam-v2.yaml",
  "artifact_path": "/artifacts/gl-cbam-v2.tar.gz",
  "certificate_id": "CERT-GL-CBAM-V2-20260501"
}
```

**Search Agents:**
```bash
GET /api/v1/agents/search?query=cbam&regulatory_scope=CBAM&status=active

Response:
{
  "total": 5,
  "page": 1,
  "agents": [
    {
      "agent_id": "gl-cbam-calculator-v2",
      "name": "CBAM Calculator",
      "regulatory_scope": ["CBAM"],
      "current_version": "2.0.0",
      "status": "active"
    }
  ]
}
```

### Appendix B: Governance Policy Example

```yaml
policy_id: pol-001-certification-required
type: compliance
name: Certification Required for Production
description: All agents must be certified before production deployment

rules:
  - condition:
      field: deployment.environment
      operator: equals
      value: production
    requirement:
      field: agent.certificate_id
      operator: not_null

enforcement: deny
enabled: true
```

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL-ProductManager | Initial Phase 3 plan |

---

**Approvals:**

- Product Manager: ___________________
- Platform Lead: ___________________
- DevOps/Security Lead: ___________________
- Engineering Lead: ___________________
