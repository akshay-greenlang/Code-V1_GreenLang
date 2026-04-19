# Registry Governance Controls

**Version:** 1.0.0
**Status:** PRODUCTION
**Owner:** GL-DevOpsEngineer
**Last Updated:** 2025-12-03

---

## Executive Summary

Governance controls ensure that agents are deployed safely, securely, and in compliance with organizational policies across multi-tenant environments. This document defines per-tenant configuration, access controls, policy enforcement, audit logging, and integration with enterprise multi-tenancy requirements.

**Key Governance Capabilities:**
- Tenant-specific agent whitelists/blacklists
- Environment-based deployment controls (dev/staging/prod)
- Lifecycle state enforcement (experimental vs certified)
- Role-based access control (RBAC)
- Policy-as-code for automated enforcement
- Comprehensive audit trails with 7-year retention
- Integration with SSO/SAML and identity providers

---

## Governance Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Governance Policy Engine                    │
├─────────────────────────────────────────────────────────┤
│  • Policy Evaluation                                     │
│  • Decision Caching                                      │
│  • Audit Logging                                         │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────┐
│                   Policy Store                           │
├──────────────────────────────────────────────────────────┤
│  • Tenant Policies (PostgreSQL)                          │
│  • Global Policies (Code-based)                          │
│  • Policy Versioning                                     │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────┐
│              Enforcement Points                          │
├──────────────────────────────────────────────────────────┤
│  • Registry API (publish, promote)                       │
│  • Runtime Deployment (kubectl apply)                    │
│  • Agent Execution (request validation)                  │
└──────────────────────────────────────────────────────────┘
```

---

## Per-Tenant Configuration

### Tenant Policy Model

```yaml
tenant_policy:
  tenant_id: "customer-abc-123"
  tenant_name: "Acme Corporation"
  policy_version: "1.2.0"
  active: true
  created_at: "2025-11-01T08:00:00Z"
  updated_at: "2025-12-01T10:00:00Z"

  # Agent Access Controls
  agent_access:
    mode: "whitelist"  # whitelist, blacklist, or open

    whitelist:
      # Specific agents allowed
      - agent_id: "gl-cbam-calculator-v2"
        versions: ["2.3.1", "2.4.0"]
        environments: ["development", "staging", "production"]

      - agent_id: "gl-csrd-materiality"
        versions: [">=1.5.0"]  # Semantic version range
        environments: ["development", "staging", "production"]

    blacklist:
      # Explicitly blocked agents
      - agent_id: "gl-experimental-beta"
        reason: "Not approved for production use"
        blocked_until: "2026-01-01"

  # Lifecycle State Controls
  lifecycle_policy:
    production:
      allowed_states: ["certified"]
      min_certification_age_days: 30  # Must be certified for 30 days

    staging:
      allowed_states: ["experimental", "certified"]
      min_experimental_requests: 1000  # Must have 1K requests in experimental

    development:
      allowed_states: ["draft", "experimental", "certified"]

  # Environment-Specific Controls
  environment_controls:
    production:
      require_approval: true
      approval_workflow: "dual-approval"
      approvers: ["engineering_manager", "security_lead"]
      auto_rollback_enabled: true
      canary_deployment_required: true

    staging:
      require_approval: false
      auto_rollback_enabled: true

    development:
      require_approval: false

  # Resource Quotas
  resource_quotas:
    max_concurrent_agents: 100
    max_deployments_per_day: 50
    max_api_requests_per_day: 10000000
    max_llm_tokens_per_day: 5000000

  # Security Controls
  security_policy:
    require_security_scan: true
    max_vulnerability_score: 7.0  # CVSS score
    allowed_base_images:
      - "gcr.io/greenlang/agent-base:*"
      - "gcr.io/greenlang/python-base:*"
    blocked_egress_domains:
      - "*.malicious.com"
    required_network_policies: true

  # Compliance Controls
  compliance:
    data_residency: "eu-central-1"  # EU data residency required
    audit_retention_days: 2555  # 7 years
    pii_handling: "strict"
    encryption_required: true

  # Cost Controls
  cost_controls:
    monthly_budget_usd: 50000
    alert_threshold: 0.80  # Alert at 80% of budget
    hard_limit_enabled: true
    overage_allowed: false
```

### Policy Inheritance

```yaml
policy_hierarchy:
  global_policies:
    # Platform-wide policies (cannot be overridden)
    - enforce_tls: true
    - min_kubernetes_version: "1.28"
    - container_image_scanning: true
    - audit_logging: true

  organization_policies:
    # Organization-level (overrides global, can be overridden by tenant)
    org_id: "org-enterprise-001"
    policies:
      - allowed_regions: ["eu-central-1", "eu-west-1"]
      - sso_required: true
      - mfa_required: true

  tenant_policies:
    # Tenant-specific (overrides organization)
    tenant_id: "customer-abc-123"
    policies:
      - agent_whitelist: [...]
      - environment_controls: [...]

  inheritance_order:
    1. Global (highest priority, cannot override)
    2. Organization
    3. Tenant
    4. Default
```

---

## Certified vs Experimental Endpoints

### Deployment Endpoints

```yaml
deployment_endpoints:
  certified_endpoint:
    url: "https://api.greenlang.ai/v1/agents"
    description: "Production-grade certified agents only"
    allowed_states: ["certified"]
    sla: "99.99%"
    support: "24/7"
    rate_limits:
      standard: "10,000 req/min"
      enterprise: "100,000 req/min"

  experimental_endpoint:
    url: "https://experimental.greenlang.ai/v1/agents"
    description: "Experimental agents for testing"
    allowed_states: ["experimental"]
    sla: "99.5%"
    support: "Business hours"
    rate_limits:
      standard: "1,000 req/min"
      enterprise: "10,000 req/min"
    disclaimers:
      - "Not for production use"
      - "Breaking changes may occur"
      - "Limited support available"

  development_endpoint:
    url: "https://dev.greenlang.ai/v1/agents"
    description: "Development and draft agents"
    allowed_states: ["draft", "experimental"]
    sla: "99%"
    support: "Community support only"
    rate_limits:
      standard: "100 req/min"
```

### Endpoint Routing

```python
class EndpointRouter:
    """Route agent requests to appropriate endpoint based on lifecycle state"""

    def route_agent_request(
        self,
        agent_id: str,
        version: str,
        environment: str,
        tenant_id: str
    ) -> str:
        # Get agent lifecycle state
        agent = registry.get_agent_version(agent_id, version)
        lifecycle_state = agent.lifecycle_state

        # Get tenant policy
        policy = governance.get_tenant_policy(tenant_id)

        # Check if agent state is allowed in environment
        allowed_states = policy.lifecycle_policy[environment]["allowed_states"]
        if lifecycle_state not in allowed_states:
            raise PolicyViolationError(
                f"Agent state '{lifecycle_state}' not allowed in '{environment}'. "
                f"Allowed states: {allowed_states}"
            )

        # Route to appropriate endpoint
        if lifecycle_state == "certified":
            return "https://api.greenlang.ai/v1/agents"
        elif lifecycle_state == "experimental":
            return "https://experimental.greenlang.ai/v1/agents"
        else:  # draft
            return "https://dev.greenlang.ai/v1/agents"
```

---

## Role-Based Access Control (RBAC)

### Role Definitions

```yaml
rbac_roles:
  # Platform Roles (GreenLang employees)
  super_admin:
    scope: "all_tenants"
    permissions:
      - "registry:*"
      - "governance:*"
      - "audit:*"
    use_case: "Platform operations and support"

  # Tenant Roles
  tenant_admin:
    scope: "single_tenant"
    permissions:
      - "registry:read"
      - "registry:write"
      - "registry:publish"
      - "registry:promote"
      - "governance:read"
      - "governance:update"
      - "audit:read"
    use_case: "Customer administrators"

  agent_developer:
    scope: "single_tenant"
    permissions:
      - "registry:read"
      - "registry:write"
      - "registry:publish"
      - "registry:promote:to_experimental"  # Can only promote to experimental
    use_case: "Development teams"

  qa_engineer:
    scope: "single_tenant"
    permissions:
      - "registry:read"
      - "registry:promote:to_certified"  # Can promote to certified
      - "evaluation:execute"
      - "audit:read"
    use_case: "QA and certification teams"

  operator:
    scope: "single_tenant"
    permissions:
      - "registry:read"
      - "agents:deploy"
      - "agents:monitor"
      - "agents:rollback"
    use_case: "Operations/SRE teams"

  analyst:
    scope: "single_tenant"
    permissions:
      - "registry:read"
      - "agents:execute"
      - "metrics:read"
    use_case: "Business analysts"

  auditor:
    scope: "single_tenant"
    permissions:
      - "registry:read"
      - "audit:read"
      - "compliance:read"
    use_case: "Compliance and audit teams"

  viewer:
    scope: "single_tenant"
    permissions:
      - "registry:read"
      - "metrics:read"
    use_case: "Read-only access"
```

### Permission Model

```yaml
permissions:
  # Registry permissions
  registry_read:
    - "List agents"
    - "Get agent metadata"
    - "Search agents"
    - "View versions"

  registry_write:
    - "Update agent metadata"
    - "Upload artifacts"

  registry_publish:
    - "Publish new agent versions"
    - "Create new agents"

  registry_promote:
    - "Promote agents between lifecycle states"
    - sub_permissions:
        - "registry:promote:to_experimental"
        - "registry:promote:to_certified"

  registry_deprecate:
    - "Deprecate agent versions"
    - "Set sunset dates"

  # Governance permissions
  governance_read:
    - "View tenant policies"
    - "View governance rules"

  governance_update:
    - "Update tenant policies"
    - "Configure access controls"
    - "Manage whitelists/blacklists"

  # Audit permissions
  audit_read:
    - "View audit logs"
    - "Export audit reports"
    - "Access compliance data"

  # Agent execution permissions
  agents_deploy:
    - "Deploy agents to environments"
    - "Update deployments"
    - "Delete deployments"

  agents_execute:
    - "Execute agent capabilities"
    - "Send requests to agents"

  agents_monitor:
    - "View agent metrics"
    - "Access logs"
    - "Create alerts"
```

### RBAC Enforcement

```python
class RBACEnforcer:
    """Enforce role-based access control"""

    def check_permission(
        self,
        user_id: str,
        tenant_id: str,
        resource: str,
        action: str
    ) -> bool:
        # Get user roles
        user_roles = self.get_user_roles(user_id, tenant_id)

        # Check if any role has required permission
        required_permission = f"{resource}:{action}"

        for role in user_roles:
            role_permissions = self.get_role_permissions(role)
            if required_permission in role_permissions:
                return True

        return False

    def enforce_permission(
        self,
        user_id: str,
        tenant_id: str,
        resource: str,
        action: str
    ):
        if not self.check_permission(user_id, tenant_id, resource, action):
            raise PermissionDeniedError(
                f"User {user_id} does not have permission "
                f"{resource}:{action} in tenant {tenant_id}"
            )
```

---

## Policy Enforcement

### Policy-as-Code

Policies are defined in declarative YAML and enforced automatically:

```yaml
# policy-production-certified-only.yaml
policy:
  id: "policy-prod-certified-only"
  name: "Production Certified Agents Only"
  description: "Only certified agents can be deployed to production"
  version: "1.0.0"
  active: true

  applies_to:
    tenants: ["*"]  # All tenants
    environments: ["production"]

  rules:
    - id: "rule-001"
      name: "Lifecycle state must be certified"
      condition: "agent.lifecycle_state == 'certified'"
      enforcement: "hard"  # hard = block, soft = warn
      error_message: "Only certified agents can be deployed to production"

    - id: "rule-002"
      name: "Agent must be certified for minimum period"
      condition: "agent.certified_days >= 30"
      enforcement: "hard"
      error_message: "Agent must be certified for at least 30 days"

    - id: "rule-003"
      name: "No critical vulnerabilities"
      condition: "agent.vulnerability_score < 9.0"
      enforcement: "hard"
      error_message: "Agent has critical vulnerabilities (CVSS >= 9.0)"

    - id: "rule-004"
      name: "Security scan must be recent"
      condition: "agent.security_scan_age_days < 7"
      enforcement: "hard"
      error_message: "Security scan is older than 7 days"

  exemptions:
    - reason: "Emergency security patch"
      approver: "cto@greenlang.ai"
      expiration: "2025-12-10"
      applies_to_agents: ["gl-cbam-calculator-v2@2.3.2"]
```

### Policy Evaluation Engine

```python
class PolicyEngine:
    """Evaluate policies against agent deployment requests"""

    def evaluate_policy(
        self,
        agent_id: str,
        version: str,
        environment: str,
        tenant_id: str,
        operation: str
    ) -> PolicyEvaluationResult:
        # Load applicable policies
        policies = self.load_policies(tenant_id, environment, operation)

        results = []
        for policy in policies:
            policy_result = self.evaluate_single_policy(
                policy, agent_id, version, environment, tenant_id
            )
            results.append(policy_result)

        # Aggregate results
        passed = all(r.passed for r in results if r.enforcement == "hard")
        warnings = [r for r in results if not r.passed and r.enforcement == "soft"]

        return PolicyEvaluationResult(
            allowed=passed,
            warnings=warnings,
            failed_rules=[r for r in results if not r.passed],
            policy_ids=[p.id for p in policies]
        )

    def evaluate_single_policy(
        self,
        policy: Policy,
        agent_id: str,
        version: str,
        environment: str,
        tenant_id: str
    ) -> RuleEvaluationResult:
        # Get agent metadata
        agent = registry.get_agent_version(agent_id, version)

        # Check exemptions
        if self.has_exemption(policy, agent_id, version):
            return RuleEvaluationResult(
                passed=True,
                exempted=True,
                exemption_reason="Emergency security patch"
            )

        # Evaluate each rule
        for rule in policy.rules:
            if not self.evaluate_rule(rule, agent, environment, tenant_id):
                return RuleEvaluationResult(
                    passed=False,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    enforcement=rule.enforcement,
                    error_message=rule.error_message
                )

        return RuleEvaluationResult(passed=True)
```

### Enforcement Points

```yaml
enforcement_points:
  # 1. Registry API (publish/promote)
  registry_api:
    operations:
      - "publish_agent"
      - "promote_agent"
      - "deprecate_agent"
    policies_checked:
      - "agent_publish_policy"
      - "security_policy"
      - "compliance_policy"

  # 2. Runtime Deployment
  runtime_deployment:
    operations:
      - "deploy_agent"
      - "update_deployment"
    policies_checked:
      - "deployment_policy"
      - "environment_policy"
      - "resource_quota_policy"

  # 3. Agent Execution
  agent_execution:
    operations:
      - "execute_agent"
      - "call_capability"
    policies_checked:
      - "access_control_policy"
      - "rate_limit_policy"
      - "cost_control_policy"
```

---

## Audit Logging

### Audit Log Schema

```yaml
audit_log_entry:
  log_id: "audit-2025-12-03-001234"
  timestamp: "2025-12-03T10:30:00.123Z"
  tenant_id: "customer-abc-123"
  user_id: "user@greenlang.ai"
  user_role: "agent_developer"

  # What happened
  event_type: "agent.published"
  operation: "publish_agent"
  resource_type: "agent_version"
  resource_id: "gl-cbam-calculator-v2@2.3.1"

  # Request details
  request:
    method: "POST"
    endpoint: "/api/v1/registry/agents"
    ip_address: "203.0.113.42"
    user_agent: "greenlang-cli/2.3.0"
    request_id: "req-abc123"

  # Response details
  response:
    status_code: 201
    success: true
    duration_ms: 450

  # Changes made
  changes:
    before: null  # New agent
    after:
      agent_id: "gl-cbam-calculator-v2"
      version: "2.3.1"
      lifecycle_state: "draft"

  # Policy evaluation
  policy_evaluation:
    policies_checked: ["agent_publish_policy", "security_policy"]
    passed: true
    warnings: []

  # Compliance metadata
  compliance:
    data_classification: "internal"
    retention_period_days: 2555  # 7 years
    region: "eu-central-1"
```

### Audit Events

```yaml
audit_events:
  # Agent lifecycle events
  agent_published:
    severity: "info"
    retention: "7 years"

  agent_promoted:
    severity: "info"
    retention: "7 years"

  agent_deprecated:
    severity: "warning"
    retention: "7 years"

  # Deployment events
  agent_deployed:
    severity: "info"
    retention: "7 years"

  agent_undeployed:
    severity: "info"
    retention: "7 years"

  deployment_failed:
    severity: "error"
    retention: "7 years"

  # Access events
  unauthorized_access_attempt:
    severity: "critical"
    retention: "10 years"
    alert: true

  permission_denied:
    severity: "warning"
    retention: "7 years"

  # Policy events
  policy_violation:
    severity: "error"
    retention: "7 years"
    alert: true

  policy_exemption_granted:
    severity: "warning"
    retention: "10 years"

  # Security events
  security_scan_failed:
    severity: "critical"
    retention: "10 years"
    alert: true

  vulnerability_detected:
    severity: "error"
    retention: "10 years"
    alert: true
```

### Audit Log Storage

```yaml
audit_storage:
  hot_storage:
    backend: "PostgreSQL"
    retention: "90 days"
    indexing: "Full text search"
    query_performance: "<100ms"

  warm_storage:
    backend: "S3 Standard"
    retention: "1 year"
    indexing: "Metadata only"
    query_performance: "<5 seconds"

  cold_storage:
    backend: "S3 Glacier"
    retention: "7 years (2555 days)"
    indexing: "None"
    retrieval_time: "1-5 hours"
    compliance: "GDPR, SOX, HIPAA compliant"

  backup:
    frequency: "Daily"
    retention: "7 years"
    encryption: "AES-256"
    geo_redundancy: true
```

### Audit Query API

```python
from greenlang.governance import AuditLog

# Query audit logs
logs = AuditLog.query(
    tenant_id="customer-abc-123",
    event_type="agent.published",
    start_time="2025-11-01",
    end_time="2025-12-01",
    limit=100
)

# Export audit report
report = AuditLog.export(
    tenant_id="customer-abc-123",
    format="csv",
    start_time="2025-01-01",
    end_time="2025-12-31"
)

# Search audit logs
results = AuditLog.search(
    tenant_id="customer-abc-123",
    query="agent:gl-cbam-calculator AND operation:promote",
    limit=50
)
```

---

## Integration with Multi-Tenancy

### Tenant Isolation

```yaml
tenant_isolation:
  data_isolation:
    strategy: "Row-level security (RLS)"
    enforcement: "PostgreSQL policies"
    verification: "Automated testing"

  example_rls_policy: |
    CREATE POLICY tenant_isolation ON agents
    FOR ALL
    TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant_id'));

  network_isolation:
    strategy: "Kubernetes Network Policies"
    enforcement: "CNI plugin (Calico)"
    rules:
      - "Tenants cannot access other tenant namespaces"
      - "Ingress only through API gateway"
      - "Egress restricted to whitelisted destinations"

  compute_isolation:
    strategy: "Dedicated Kubernetes namespaces"
    resource_quotas: true
    limit_ranges: true
    pod_security_policies: true
```

### Multi-Region Data Residency

```yaml
data_residency_enforcement:
  eu_tenant:
    tenant_id: "customer-eu-123"
    primary_region: "eu-central-1"
    allowed_regions: ["eu-central-1", "eu-west-1"]
    blocked_regions: ["us-*", "cn-*"]

    policy_enforcement:
      - "All data stored in EU regions only"
      - "No cross-border data transfers"
      - "EU-based support staff only"
      - "GDPR compliance enforced"

  us_tenant:
    tenant_id: "customer-us-456"
    primary_region: "us-east-1"
    allowed_regions: ["us-east-1", "us-west-2"]
    blocked_regions: ["eu-*", "cn-*"]

    policy_enforcement:
      - "All data stored in US regions only"
      - "HIPAA compliance enforced"
      - "US-based support staff only"

  china_tenant:
    tenant_id: "customer-cn-789"
    primary_region: "cn-north-1"
    allowed_regions: ["cn-north-1"]
    blocked_regions: ["*"]  # No data egress

    policy_enforcement:
      - "All data stored in China"
      - "No data egress to outside China"
      - "PIPL compliance enforced"
      - "Local LLM providers only"
```

### SSO/SAML Integration

```yaml
sso_integration:
  identity_providers:
    - provider: "Okta"
      tenant_id: "customer-abc-123"
      saml_metadata_url: "https://okta.com/app/metadata/abc123"
      attributes:
        user_id: "email"
        roles: "groups"
        tenant_id: "customAttribute1"

    - provider: "Azure AD"
      tenant_id: "customer-xyz-456"
      saml_metadata_url: "https://login.microsoftonline.com/xyz456/metadata"
      attributes:
        user_id: "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"
        roles: "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups"

  jit_provisioning:
    enabled: true
    create_users: true
    update_users: true
    deactivate_users: true
    role_mapping:
      "Engineering": "agent_developer"
      "QA": "qa_engineer"
      "Operations": "operator"
      "Admins": "tenant_admin"

  mfa_enforcement:
    required_for_roles: ["tenant_admin", "super_admin"]
    methods: ["TOTP", "SMS", "Hardware Token"]
```

---

## Compliance Reporting

### SOC 2 Compliance

```yaml
soc2_compliance:
  controls:
    - control_id: "CC6.1"
      description: "Logical and physical access controls"
      implementation:
        - "RBAC for all operations"
        - "Audit logging with 7-year retention"
        - "MFA for privileged accounts"
      evidence:
        - "Audit logs exported monthly"
        - "Access reviews conducted quarterly"

    - control_id: "CC7.2"
      description: "System monitoring"
      implementation:
        - "Real-time monitoring of all systems"
        - "Automated alerting for anomalies"
        - "24/7 SOC coverage"
      evidence:
        - "Monitoring dashboards"
        - "Incident response logs"

  reporting:
    frequency: "Annual"
    auditor: "External SOC 2 auditor"
    next_audit: "2026-Q2"
```

### GDPR Compliance

```yaml
gdpr_compliance:
  data_subject_rights:
    right_to_access:
      implementation: "Self-service portal for data export"
      sla: "30 days"

    right_to_erasure:
      implementation: "Automated data deletion pipeline"
      sla: "30 days"
      retention_exceptions: "Legal hold, audit requirements"

    right_to_portability:
      implementation: "Export in JSON/CSV format"
      sla: "30 days"

  lawful_basis:
    processing_basis: "Contractual necessity + Legitimate interest"
    documentation: "Privacy policy, DPA"

  data_protection_officer:
    name: "Jane Doe"
    email: "dpo@greenlang.ai"
```

---

## Best Practices

### For Tenants

1. **Start with Whitelists** - Use whitelist mode for production environments
2. **Enforce Lifecycle States** - Require certified agents in production
3. **Enable Auto-Rollback** - Automatically rollback failed deployments
4. **Monitor Policy Violations** - Set up alerts for policy violations
5. **Regular Audits** - Review audit logs quarterly

### For Platform Operators

1. **Default Deny** - Default to restrictive policies
2. **Audit Everything** - Log all registry and governance operations
3. **Automate Enforcement** - Use policy-as-code for consistency
4. **Regular Reviews** - Review policies and exemptions monthly
5. **Tenant Education** - Provide governance best practices documentation

---

## Related Documentation

- [Registry Overview](../architecture/00-REGISTRY_OVERVIEW.md)
- [Registry API Specification](../api-specs/00-REGISTRY_API.md)
- [Agent Lifecycle Management](../lifecycle/00-AGENT_LIFECYCLE.md)
- [Runtime Architecture](../architecture/01-RUNTIME_ARCHITECTURE.md)
- [Multi-Tenancy Requirements](C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\Upgrade_needed_Agentfactory.md)

---

**Questions or feedback?**
- Slack: #governance
- Email: governance@greenlang.ai
- Wiki: https://wiki.greenlang.ai/governance
