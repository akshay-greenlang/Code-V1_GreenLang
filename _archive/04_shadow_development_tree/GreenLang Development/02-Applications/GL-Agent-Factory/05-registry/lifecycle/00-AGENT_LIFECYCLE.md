# Agent Lifecycle Management

**Version:** 1.0.0
**Status:** PRODUCTION
**Owner:** GL-DevOpsEngineer
**Last Updated:** 2025-12-03

---

## Overview

Agent lifecycle management defines the progression of agents from initial development through production deployment and eventual deprecation. This document specifies lifecycle states, promotion criteria, deprecation policies, and version management strategies.

**Core Principle:** Agents progress through well-defined states with objective criteria for advancement, ensuring quality, security, and reliability at each stage.

---

## Lifecycle States

```
┌─────────┐
│  DRAFT  │ ← Initial state after publish
└────┬────┘
     │ Promotion criteria:
     │ - Container image published
     │ - Basic evaluation passed
     │ - No critical vulnerabilities
     ↓
┌──────────────┐
│ EXPERIMENTAL │ ← Testing with limited users
└──────┬───────┘
       │ Promotion criteria:
       │ - Comprehensive evaluation passed
       │ - Min usage threshold (>10K requests, >3 users)
       │ - Error rate < 1%
       │ - Security scan passed
       │ - Documentation complete
       ↓
┌───────────┐
│ CERTIFIED │ ← Production-ready
└─────┬─────┘
      │ Deprecation trigger:
      │ - Superseded by new version
      │ - Security vulnerability
      │ - End of support
      ↓
┌────────────┐
│ DEPRECATED │ ← Sunset period
└────────────┘
```

---

## State Definitions

### 1. DRAFT

**Purpose:** Initial development and testing

**Characteristics:**
- Agent code is complete but not production-tested
- Available only to the creating tenant
- No SLA guarantees
- Rapid iteration allowed
- Can be deleted or modified

**Access Control:**
- Creator: Full access
- Team members: Read/execute access
- Other tenants: No access

**Monitoring:**
- Basic error tracking
- No performance SLOs
- No usage quotas

**Duration:** Typically 1-7 days

**Exit Criteria:**
```yaml
draft_to_experimental:
  required:
    - container_image_published: true
    - basic_evaluation_passed: true
    - no_critical_vulnerabilities: true
    - readme_exists: true
    - input_output_schemas_defined: true

  optional:
    - integration_tests_passed: true
    - documentation_reviewed: true
```

**CLI Commands:**
```bash
# Publish as draft
gl agent publish --state draft

# Check promotion criteria
gl agent check-promotion gl-cbam-calculator-v2@2.3.1

# Promote to experimental
gl agent promote gl-cbam-calculator-v2@2.3.1 --to experimental
```

---

### 2. EXPERIMENTAL

**Purpose:** Real-world testing with limited production exposure

**Characteristics:**
- Available to selected users/tenants (opt-in)
- Limited SLA (99.5% uptime)
- Usage monitored and analyzed
- Breaking changes allowed with notification
- Cannot be deleted (only deprecated)

**Access Control:**
- Creator tenant: Full access
- Whitelisted tenants: Execute access (opt-in required)
- Other tenants: Discoverable but not executable

**Monitoring:**
- Full performance metrics
- Error tracking and alerting
- User feedback collection
- Usage analytics

**SLOs:**
```yaml
experimental_slos:
  availability: "99.5%"
  latency_p95: "<5 seconds"
  error_rate: "<5%"
  support: "Business hours (9am-5pm)"
```

**Duration:** Typically 1-4 weeks

**Exit Criteria:**
```yaml
experimental_to_certified:
  required:
    # Evaluation
    - comprehensive_evaluation_passed: true
    - performance_slos_met: true
    - quality_metrics_above_threshold: true

    # Security
    - security_scan_passed: true
    - no_high_or_critical_vulnerabilities: true
    - dependency_audit_passed: true

    # Usage
    - min_experimental_requests: 10000
    - min_experimental_users: 3
    - min_experimental_days: 7

    # Quality
    - error_rate_30d: "<1%"
    - p95_latency_meets_target: true
    - no_data_loss_incidents: true

    # Documentation
    - documentation_complete: true
    - api_docs_generated: true
    - runbook_created: true

  optional:
    - user_satisfaction_score: ">4.0/5.0"
    - support_tickets_resolved: true
```

**CLI Commands:**
```bash
# Check experimental metrics
gl agent metrics gl-cbam-calculator-v2@2.3.1 --window 30d

# View promotion readiness
gl agent promotion-status gl-cbam-calculator-v2@2.3.1

# Promote to certified
gl agent promote gl-cbam-calculator-v2@2.3.1 --to certified
```

---

### 3. CERTIFIED

**Purpose:** Production-ready agents with full SLA

**Characteristics:**
- Available to all tenants (subject to governance policies)
- Full SLA (99.9% or 99.99% depending on tier)
- Immutable (no breaking changes)
- Fully supported
- Long-term stability guarantee

**Access Control:**
- All tenants: Execute access (subject to governance policies)
- Creator tenant: Full metadata access
- GreenLang admins: Full administrative access

**Monitoring:**
- Comprehensive observability
- Real-time alerting
- Performance dashboards
- Usage analytics
- Cost tracking

**SLOs:**
```yaml
certified_slos:
  availability:
    standard: "99.9%"
    premium: "99.99%"
    mission_critical: "99.995%"

  latency:
    p50: "<100ms"
    p95: "<500ms"
    p99: "<2000ms"

  error_rate: "<0.5%"

  support:
    standard: "Business hours"
    premium: "24/7"
    mission_critical: "24/7 + dedicated TAM"
```

**Version Compatibility:**
```yaml
certified_versioning:
  major_version_change:
    - Breaking changes allowed
    - Requires new agent_id or major version bump
    - Old version remains certified until deprecated

  minor_version_change:
    - New features, backward compatible
    - Automatic promotion if criteria met
    - Replaces previous minor version

  patch_version_change:
    - Bug fixes only
    - Fast-track promotion allowed
    - Immediate rollout recommended
```

**Deprecation Triggers:**
```yaml
certified_to_deprecated:
  triggers:
    - superseded_by_new_version: true
    - critical_security_vulnerability: true
    - end_of_support_reached: true
    - performance_degradation: true
    - compliance_violation: true

  grace_period:
    standard: "90 days"
    critical_security: "30 days"
    emergency: "7 days"
```

**CLI Commands:**
```bash
# List certified agents
gl agent list --state certified

# Deploy certified agent
gl agent deploy gl-cbam-calculator-v2@2.3.1 --environment production

# Monitor certified agent
gl agent monitor gl-cbam-calculator-v2@2.3.1
```

---

### 4. DEPRECATED

**Purpose:** Sunset period before removal

**Characteristics:**
- Still functional but not recommended
- Visible deprecation warnings in all interfaces
- No new deployments allowed (existing deployments continue)
- Limited support (critical bugs only)
- Sunset date announced

**Access Control:**
- Existing users: Continue access
- New users: Blocked from deployment
- All users: See deprecation notices

**Monitoring:**
- Track remaining active deployments
- Monitor migration progress
- Alert on sunset date approach

**Sunset Process:**
```yaml
deprecation_timeline:
  day_0:
    - agent_marked_deprecated: true
    - deprecation_notice_sent: true
    - replacement_version_announced: true

  day_30:
    - reminder_sent_to_active_users: true
    - migration_guide_available: true

  day_60:
    - final_warning_sent: true
    - support_ending_notice: true

  day_90:
    - new_deployments_blocked: true
    - existing_deployments_auto_migrated: true  # if auto-migration enabled

  day_120:
    - agent_removed_from_registry: true
    - container_images_archived: true
```

**Migration Support:**
```yaml
migration_support:
  automated:
    - auto_migration_available: true  # if replacement is backward compatible
    - migration_script_provided: true
    - rollback_available: true

  manual:
    - migration_guide: "https://docs.greenlang.ai/migration/2.3-to-2.4"
    - support_hours: "Business hours"
    - dedicated_slack_channel: "#migration-support"
```

**CLI Commands:**
```bash
# Deprecate agent version
gl agent deprecate gl-cbam-calculator-v2@2.3.1 \
  --replacement 2.4.0 \
  --sunset 2026-03-01 \
  --reason "Security vulnerability CVE-2025-1234"

# Check deprecation status
gl agent deprecation-status gl-cbam-calculator-v2@2.3.1

# Migrate to replacement
gl agent migrate \
  --from gl-cbam-calculator-v2@2.3.1 \
  --to gl-cbam-calculator-v2@2.4.0
```

---

## Promotion Workflows

### Automated Promotion

```yaml
auto_promotion:
  enabled_for:
    - patch_versions  # Bug fixes
    - minor_versions_with_full_tests  # Backward compatible features

  criteria_checks:
    - evaluation_results: "PASS"
    - security_scan: "PASS"
    - test_coverage: ">90%"
    - performance_regression: "NONE"

  workflow:
    1. Agent published to draft
    2. Automated evaluation triggered
    3. If all criteria met → promote to experimental
    4. Monitor experimental for 7 days
    5. If experimental criteria met → promote to certified
    6. Send notification to stakeholders

  cli_command: |
    gl agent publish --auto-promote
```

### Manual Promotion

```yaml
manual_promotion:
  required_for:
    - major_versions  # Breaking changes
    - security_sensitive_agents
    - high_risk_changes

  workflow:
    1. Agent published to draft
    2. Creator runs evaluation: gl agent evaluate
    3. Creator reviews results
    4. Creator promotes to experimental: gl agent promote --to experimental
    5. Monitor experimental for 2-4 weeks
    6. QA team reviews metrics
    7. QA team promotes to certified: gl agent promote --to certified
    8. Stakeholders notified

  approvers:
    experimental: ["agent_creator", "team_lead"]
    certified: ["qa_lead", "engineering_manager"]
```

### Fast-Track Promotion (Emergency)

```yaml
fast_track_promotion:
  use_cases:
    - critical_security_patches
    - data_loss_bugs
    - regulatory_compliance_fixes

  criteria:
    - severity: "CRITICAL"
    - fix_verified: true
    - regression_tests_passed: true
    - security_review_completed: true

  workflow:
    1. Emergency fix committed
    2. Expedited evaluation (automated only)
    3. Security review (<4 hours)
    4. Fast-track to certified (skip experimental)
    5. Immediate deployment to production
    6. Post-deployment monitoring (24/7)

  approval_required:
    - engineering_manager
    - security_lead
    - cto  # for production deployments

  cli_command: |
    gl agent promote gl-cbam-calculator-v2@2.3.2 \
      --fast-track \
      --severity critical \
      --reason "CVE-2025-1234 security patch"
```

---

## Version Management Strategy

### Semantic Versioning

```yaml
semver_rules:
  major_version:
    - breaking_changes: true
    - examples:
      - "Changed input schema (removed field)"
      - "Changed output schema (incompatible format)"
      - "Removed public API endpoint"
    - version_bump: "2.3.1 → 3.0.0"

  minor_version:
    - new_features: true
    - backward_compatible: true
    - examples:
      - "Added new optional input field"
      - "Added new capability"
      - "Performance improvement"
    - version_bump: "2.3.1 → 2.4.0"

  patch_version:
    - bug_fixes: true
    - no_new_features: true
    - examples:
      - "Fixed calculation error"
      - "Fixed memory leak"
      - "Fixed documentation typo"
    - version_bump: "2.3.1 → 2.3.2"
```

### Version Lifecycle

```yaml
version_lifecycle:
  active_versions:
    - latest_major: "Always certified"
    - latest_minor: "Certified or experimental"
    - latest_patch: "Certified or experimental"

  deprecated_versions:
    - previous_minor: "Deprecated after new minor released"
    - previous_patch: "Immediately deprecated when new patch released"

  example:
    - version: "3.0.0"
      state: "certified"
      comment: "Latest major"

    - version: "2.4.1"
      state: "certified"
      comment: "Latest minor of previous major"

    - version: "2.4.0"
      state: "deprecated"
      comment: "Superseded by 2.4.1"

    - version: "2.3.5"
      state: "deprecated"
      comment: "Superseded by 2.4.x"
```

### Version Selection Policy

```yaml
version_selection:
  default_behavior:
    - production: "latest certified version"
    - staging: "latest certified or experimental"
    - development: "any version"

  pinning_strategies:
    exact_version:
      - format: "gl-cbam-calculator-v2@2.3.1"
      - use_case: "Production deployments"
      - risk: "Low"

    minor_version:
      - format: "gl-cbam-calculator-v2@2.3.x"
      - use_case: "Get bug fixes automatically"
      - risk: "Low"

    major_version:
      - format: "gl-cbam-calculator-v2@2.x"
      - use_case: "Get new features automatically"
      - risk: "Medium"

    latest:
      - format: "gl-cbam-calculator-v2@latest"
      - use_case: "Development/testing only"
      - risk: "High"
```

---

## Deprecation Policy

### Standard Deprecation

```yaml
standard_deprecation:
  triggers:
    - new_major_version_released: true
    - new_minor_version_with_improvements: true

  timeline:
    - deprecation_announced: "Day 0"
    - new_deployments_blocked: "Day 90"
    - support_ended: "Day 120"
    - removed_from_registry: "Day 180"

  notifications:
    - email_to_active_users: "Day 0, 30, 60, 90"
    - in_app_warnings: "Continuous"
    - api_deprecation_headers: "Continuous"
    - slack_announcements: "Day 0, 60"

  migration_support:
    - automated_migration_available: "If possible"
    - migration_guide: "Required"
    - support_team_available: "Business hours"
```

### Accelerated Deprecation

```yaml
accelerated_deprecation:
  triggers:
    - critical_security_vulnerability: true
    - data_integrity_issue: true
    - compliance_violation: true

  timeline:
    - deprecation_announced: "Day 0"
    - new_deployments_blocked: "Immediate"
    - mandatory_upgrade_notice: "Day 7"
    - forced_auto_migration: "Day 30"
    - removed_from_registry: "Day 60"

  notifications:
    - urgent_email_to_all_users: "Immediate"
    - in_app_critical_alerts: "Immediate"
    - slack_emergency_channel: "Immediate"
    - phone_calls_to_enterprise_customers: "Within 24 hours"

  migration_support:
    - automated_migration_required: true
    - 24/7_support_available: true
    - dedicated_incident_response_team: true
```

### Deprecation Communication

```email
Subject: [Action Required] Agent Deprecation Notice: gl-cbam-calculator-v2@2.3.1

Dear GreenLang Customer,

We are deprecating gl-cbam-calculator-v2@2.3.1 on 2026-03-01.

WHY:
This version has been superseded by 2.4.0, which includes:
- 30% performance improvement
- Enhanced accuracy for aluminum calculations
- Security fixes for CVE-2025-1234

TIMELINE:
- 2025-12-03: Deprecation announced
- 2026-01-01: New deployments blocked
- 2026-02-01: Support ended (critical bugs only)
- 2026-03-01: Removed from registry

ACTION REQUIRED:
Migrate to version 2.4.0 before 2026-03-01.

Migration options:
1. Automated: gl agent migrate --from 2.3.1 --to 2.4.0
2. Manual: Follow guide at https://docs.greenlang.ai/migration/2.3-to-2.4

SUPPORT:
- Migration guide: https://docs.greenlang.ai/migration/2.3-to-2.4
- Slack: #migration-support
- Email: support@greenlang.ai

Questions? Reply to this email or contact your Customer Success Manager.

Best regards,
GreenLang Platform Team
```

---

## Rollback Procedures

### Version Rollback

```yaml
rollback_scenarios:
  performance_degradation:
    trigger: "p95 latency > 2× previous version"
    action: "Automatic rollback to previous certified version"
    timeline: "Within 5 minutes"

  high_error_rate:
    trigger: "Error rate > 5%"
    action: "Automatic rollback + incident alert"
    timeline: "Within 2 minutes"

  security_incident:
    trigger: "Security vulnerability discovered"
    action: "Immediate deprecation + forced rollback"
    timeline: "Immediate"

rollback_workflow:
  1. Detect issue (automated monitoring)
  2. Alert on-call engineer
  3. Validate rollback candidate (previous certified version)
  4. Execute rollback: gl agent rollback {agent_id}@{version}
  5. Verify rollback success (health checks)
  6. Notify stakeholders
  7. Incident postmortem

cli_command: |
  # Rollback to previous version
  gl agent rollback gl-cbam-calculator-v2@2.3.1

  # Rollback to specific version
  gl agent rollback gl-cbam-calculator-v2@2.3.1 --to 2.3.0

  # Rollback with reason
  gl agent rollback gl-cbam-calculator-v2@2.3.1 \
    --reason "High error rate detected"
```

---

## Lifecycle Metrics and Monitoring

### Key Metrics

```yaml
lifecycle_metrics:
  state_duration:
    - avg_draft_duration: "3 days"
    - avg_experimental_duration: "14 days"
    - avg_certified_duration: "180 days"

  promotion_success_rate:
    - draft_to_experimental: "85%"
    - experimental_to_certified: "92%"

  deprecation_metrics:
    - avg_deprecation_notice_period: "90 days"
    - avg_migration_completion_time: "60 days"
    - forced_migrations: "<5%"

  quality_metrics:
    - avg_error_rate_experimental: "2.3%"
    - avg_error_rate_certified: "0.4%"
    - avg_certification_time: "17 days"
```

### Monitoring Dashboard

```yaml
lifecycle_dashboard:
  sections:
    - current_state_distribution:
        draft: 45
        experimental: 12
        certified: 230
        deprecated: 18

    - promotion_pipeline:
        ready_for_experimental: 8
        ready_for_certified: 3
        blocked_on_criteria: 4

    - deprecation_tracking:
        scheduled_deprecations_30d: 5
        active_migrations: 12
        sunset_approaching_7d: 2

  alerts:
    - experimental_exceeding_30d: "Review needed"
    - certification_criteria_not_met: "Investigation required"
    - deprecated_still_deployed: "Migration urgency"
```

---

## Best Practices

### For Agent Creators

1. **Start in Draft** - Iterate quickly in draft state
2. **Comprehensive Testing** - Test thoroughly before promoting to experimental
3. **Monitor Experimental** - Actively monitor experimental deployments
4. **Document Everything** - Complete documentation before certification
5. **Plan for Deprecation** - Design for backward compatibility

### For Platform Operators

1. **Automate Promotion** - Use automated promotion for low-risk changes
2. **Monitor Lifecycle** - Track lifecycle metrics and trends
3. **Enforce Quality Gates** - Don't skip promotion criteria
4. **Plan Deprecations** - Give adequate notice for deprecations
5. **Support Migrations** - Provide tools and support for migrations

---

## Related Documentation

- [Registry Overview](../architecture/00-REGISTRY_OVERVIEW.md)
- [Registry API Specification](../api-specs/00-REGISTRY_API.md)
- [Governance Controls](../governance/00-GOVERNANCE_CONTROLS.md)
- [Runtime Architecture](../architecture/01-RUNTIME_ARCHITECTURE.md)

---

**Questions or feedback?**
- Slack: #agent-lifecycle
- Email: lifecycle@greenlang.ai
- Wiki: https://wiki.greenlang.ai/lifecycle
