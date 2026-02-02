# Agent Registry & Runtime - Documentation Index

**Version:** 1.0.0
**Status:** PRODUCTION
**Owner:** GL-DevOpsEngineer
**Last Updated:** 2025-12-03

---

## Overview

This directory contains comprehensive documentation for the GreenLang Agent Registry and Runtime infrastructure. The Agent Registry treats agents as first-class, versioned, governed assets, while the Runtime executes them with proper isolation, monitoring, and SLO enforcement.

---

## Documentation Structure

### 1. Architecture

#### [00-REGISTRY_OVERVIEW.md](architecture/00-REGISTRY_OVERVIEW.md)
- **Purpose:** Comprehensive registry architecture and design
- **Key Topics:**
  - Registry architecture principles
  - What gets registered (agents, versions, capabilities, evaluation results)
  - Storage backend (PostgreSQL, S3, Redis, Vector DB)
  - API surface overview
  - Multi-region replication for data residency
  - Security and performance targets

#### [01-RUNTIME_ARCHITECTURE.md](architecture/01-RUNTIME_ARCHITECTURE.md)
- **Purpose:** Production runtime execution environment
- **Key Topics:**
  - Kubernetes-native deployment and orchestration
  - Multi-tenant isolation with namespaces
  - Horizontal and vertical pod autoscaling (HPA/VPA)
  - Resource quotas and network policies
  - Monitoring with Prometheus/Grafana
  - SLO enforcement and auto-remediation
  - Deployment strategies (rolling, blue-green, canary)
  - Integration with Registry
  - Disaster recovery and backup strategies

---

### 2. API Specifications

#### [00-REGISTRY_API.md](api-specs/00-REGISTRY_API.md)
- **Purpose:** Complete API reference for Registry operations
- **Key Topics:**
  - `gl agent publish` - Publish new agent versions
  - `gl agent list` - List and filter agents
  - `gl agent search` - Semantic search across capabilities
  - `gl agent promote` - Promote agents through lifecycle states
  - `gl agent deprecate` - Deprecate old versions
  - Governance policy checks
  - Usage metrics and analytics
  - REST API, gRPC, CLI, and SDK examples
  - Error handling and rate limits

---

### 3. Lifecycle Management

#### [00-AGENT_LIFECYCLE.md](lifecycle/00-AGENT_LIFECYCLE.md)
- **Purpose:** Agent lifecycle state management
- **Key Topics:**
  - Lifecycle states: `draft → experimental → certified → deprecated`
  - Promotion criteria and workflows
    - **Draft → Experimental:** Basic evaluation, security scan
    - **Experimental → Certified:** Comprehensive evaluation, min usage threshold, error rate < 1%
  - Automated vs manual promotion
  - Fast-track promotion for critical patches
  - Deprecation policy and timelines
  - Version management strategy (semantic versioning)
  - Rollback procedures

---

### 4. Governance Controls

#### [00-GOVERNANCE_CONTROLS.md](governance/00-GOVERNANCE_CONTROLS.md)
- **Purpose:** Multi-tenant governance and policy enforcement
- **Key Topics:**
  - Per-tenant configuration (whitelists, blacklists, resource quotas)
  - Environment-based controls (dev, staging, production)
  - Lifecycle state enforcement (experimental vs certified endpoints)
  - Role-based access control (RBAC)
    - Roles: super_admin, tenant_admin, developer, operator, analyst, auditor, viewer
  - Policy-as-code for automated enforcement
  - Comprehensive audit logging (7-year retention)
  - Integration with SSO/SAML
  - Multi-region data residency compliance
  - SOC 2 and GDPR compliance reporting

---

## Quick Start

### For Agent Developers

1. **Publish an agent:**
   ```bash
   gl agent publish \
     --name "CBAM Carbon Calculator" \
     --version 2.3.1 \
     --domain sustainability.cbam \
     --type calculator
   ```

2. **Check promotion readiness:**
   ```bash
   gl agent promotion-status gl-cbam-calculator-v2@2.3.1
   ```

3. **Promote to experimental:**
   ```bash
   gl agent promote gl-cbam-calculator-v2@2.3.1 --to experimental
   ```

4. **Monitor agent:**
   ```bash
   gl agent metrics gl-cbam-calculator-v2@2.3.1 --window 30d
   ```

### For Platform Operators

1. **Deploy agent to production:**
   ```bash
   kubectl apply -f manifests/gl-cbam-calculator-v2.yaml
   ```

2. **Monitor SLOs:**
   ```bash
   gl slo status --agent gl-cbam-calculator-v2
   ```

3. **Check governance policies:**
   ```bash
   gl agent check gl-cbam-calculator-v2@2.3.1 \
     --tenant customer-abc-123 \
     --environment production
   ```

4. **View audit logs:**
   ```bash
   gl audit query \
     --tenant customer-abc-123 \
     --event-type agent.promoted \
     --start-time 2025-11-01
   ```

---

## Key Concepts

### Agent as First-Class Asset

Agents are treated as immutable, versioned artifacts:
- **Identity:** Unique `agent_id` + `version`
- **Immutability:** Published versions cannot be modified
- **Versioning:** Semantic versioning (major.minor.patch)
- **Metadata:** Rich descriptive metadata and capabilities
- **Provenance:** Full audit trail of creation and modifications

### Lifecycle States

```
DRAFT → EXPERIMENTAL → CERTIFIED → DEPRECATED
```

- **DRAFT:** Initial development, tenant-private, rapid iteration
- **EXPERIMENTAL:** Limited production testing, opt-in access
- **CERTIFIED:** Production-ready, full SLA, widely available
- **DEPRECATED:** Sunset period, migration to replacement version

### Multi-Tenant Isolation

- **Data Isolation:** Row-level security (PostgreSQL RLS)
- **Network Isolation:** Kubernetes Network Policies
- **Compute Isolation:** Dedicated namespaces per tenant
- **Resource Quotas:** Per-tenant CPU, memory, storage limits

### SLO-Driven Operations

Runtime enforces strict SLOs:
- **Availability:** 99.99% for certified agents
- **Latency:** P95 < 500ms, P99 < 2000ms
- **Error Rate:** < 0.5%
- **Auto-remediation:** Automatic rollback on SLO violations

---

## Integration Points

The Registry and Runtime integrate with:

1. **Agent Factory Build System** - Receives newly built agents
2. **Evaluation Framework** - Stores and retrieves evaluation results
3. **Governance System** - Enforces deployment policies
4. **Monitoring System** - Collects metrics and usage analytics
5. **CI/CD Pipeline** - Automates publish, test, and promotion
6. **Kubernetes** - Orchestrates agent deployment and scaling
7. **Prometheus/Grafana** - Monitors performance and SLOs

---

## Architecture Diagrams

### Registry Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Registry API                        │
│  (RESTful + gRPC for publish, search, promote, deprecate)   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                  Registry Core Services                      │
│  • Metadata Service       • Version Management               │
│  • Search & Discovery     • Lifecycle State Machine          │
│  • Governance Engine      • Promotion Pipeline               │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   Storage Backend                            │
│  • PostgreSQL (metadata)  • S3 (artifacts)                   │
│  • Redis (cache)          • Vector DB (semantic search)      │
└──────────────────────────────────────────────────────────────┘
```

### Runtime Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                     Ingress + API Gateway                      │
└────────────────────────┬──────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
│   Tenant A   │  │   Tenant B   │  │  Tenant C  │
│  Namespace   │  │  Namespace   │  │ Namespace  │
│  (Isolated)  │  │  (Isolated)  │  │ (Isolated) │
└──────┬───────┘  └──────┬──────┘  └──────┬─────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────┐
│  Shared Services: PostgreSQL, Redis, S3, Monitoring           │
└───────────────────────────────────────────────────────────────┘
```

---

## Performance Targets

### Registry SLOs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Get Agent | < 50ms | P95 latency |
| Search | < 200ms | P95 latency |
| Publish | < 2 seconds | P95 latency |
| Availability | 99.99% | Monthly uptime |
| Read Ops | > 10,000 req/sec | Throughput |
| Write Ops | > 500 req/sec | Throughput |

### Runtime SLOs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Availability | 99.99% | Certified agents |
| Latency P50 | < 100ms | Request duration |
| Latency P95 | < 500ms | Request duration |
| Latency P99 | < 2000ms | Request duration |
| Error Rate | < 0.5% | Failed requests |

---

## Security Considerations

### Access Control

- **Authentication:** API keys, OAuth 2.0/OIDC, mTLS
- **Authorization:** RBAC with fine-grained permissions
- **Encryption:** TLS 1.3, AES-256 for data at rest
- **Audit Logging:** All operations logged with 7-year retention

### Container Security

- **Image Scanning:** Automated vulnerability scanning
- **Base Images:** Hardened, regularly updated base images
- **Security Context:** Non-root users, seccomp profiles
- **Network Policies:** Egress/ingress restrictions

### Data Protection

- **Tenant Isolation:** Complete data separation
- **Data Residency:** Regional data storage enforcement
- **Encryption:** At rest and in transit
- **Backup:** Encrypted, geo-redundant backups

---

## Compliance

### Supported Standards

- **SOC 2 Type II:** Security, availability, confidentiality
- **GDPR:** Data protection and privacy
- **HIPAA:** Healthcare data protection (US tenants)
- **ISO 27001:** Information security management

### Audit Trail

All registry and runtime operations are audited:
- Agent publish, promote, deprecate
- Deployment, scaling, rollback
- Access attempts and policy violations
- Configuration changes

Audit logs retained for 7 years (2,555 days) for compliance.

---

## Monitoring and Alerting

### Dashboards

- **Registry Dashboard:** Agent metrics, usage trends, promotion pipeline
- **Runtime Dashboard:** Pod health, resource utilization, SLO compliance
- **Tenant Dashboard:** Per-tenant usage, costs, quotas
- **SLO Dashboard:** Real-time SLO tracking and error budget

### Alert Channels

- **P0 (Critical):** PagerDuty → On-call engineer
- **P1 (High):** Slack #incidents
- **P2 (Medium):** Slack #performance
- **P3 (Low):** Email

---

## Cost Management

### Cost Tracking

- **Compute:** CPU/memory usage per tenant
- **LLM:** Token usage and API costs
- **Storage:** S3, EBS, and database storage
- **Network:** Data transfer and egress

### Cost Optimization

- **Right-sizing:** VPA recommendations for pod resources
- **Spot Instances:** Non-production workloads
- **Scheduled Scaling:** Scale down outside business hours
- **LLM Caching:** Reduce redundant LLM API calls
- **Reserved Instances:** Baseline production capacity

---

## Disaster Recovery

- **RTO:** 1 hour (Recovery Time Objective)
- **RPO:** 5 minutes (Recovery Point Objective)
- **Backup Frequency:** Daily (database), every 6 hours (Kubernetes state)
- **Backup Retention:** 30 days
- **Cross-Region Replication:** All critical data

---

## Related Documentation

- [Performance SLOs](../../slo/PERFORMANCE_SLOS.md)
- [Multi-Tenancy Requirements](../../docs/planning/greenlang-2030-vision/Upgrade_needed_Agentfactory.md)
- [Evaluation Framework](../04-evaluation/)
- [Agent SDK](../02-sdk/)

---

## Support and Feedback

### Communication Channels

- **Slack:**
  - #agent-registry - Registry questions
  - #runtime-infrastructure - Runtime questions
  - #governance - Governance and compliance
  - #agent-lifecycle - Lifecycle management

- **Email:**
  - registry@greenlang.ai
  - runtime@greenlang.ai
  - governance@greenlang.ai

- **Wiki:** https://wiki.greenlang.ai/registry

### Getting Help

1. **Documentation:** Start with this README and linked documents
2. **Slack:** Ask questions in appropriate channels
3. **Email:** For detailed technical support
4. **Incident:** For production issues, page on-call via PagerDuty

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-03 | Initial comprehensive documentation |

---

**Built with production-grade DevOps best practices by GL-DevOpsEngineer**
