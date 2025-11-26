# GL-010 EMISSIONWATCH Rollback Procedure

## Document Control

| Property | Value |
|----------|-------|
| Document ID | GL-010-RUNBOOK-RB-001 |
| Version | 1.0.0 |
| Last Updated | 2025-11-26 |
| Owner | GL-010 Operations Team |
| Classification | Internal |
| Review Cycle | Quarterly |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pre-Rollback Assessment](#2-pre-rollback-assessment)
3. [Rollback Decision Matrix](#3-rollback-decision-matrix)
4. [Deployment Rollback](#4-deployment-rollback)
5. [Configuration Rollback](#5-configuration-rollback)
6. [Database Migration Rollback](#6-database-migration-rollback)
7. [Integration Update Rollback](#7-integration-update-rollback)
8. [Post-Rollback Validation](#8-post-rollback-validation)
9. [Regulatory Considerations](#9-regulatory-considerations)
10. [Emergency Rollback Procedures](#10-emergency-rollback-procedures)
11. [Rollback Communication](#11-rollback-communication)
12. [Appendices](#12-appendices)

---

## 1. Overview

### 1.1 Purpose

This Rollback Procedure document provides comprehensive guidance for safely reverting GL-010 EMISSIONWATCH system components to previous stable states. Given the critical nature of emissions monitoring and regulatory compliance, rollback procedures must ensure:

- No emissions data loss
- Continuous regulatory compliance
- Minimal service interruption
- Complete data integrity
- Audit trail maintenance

### 1.2 Scope

This document covers rollback procedures for:
- Application deployments (Kubernetes)
- Configuration changes
- Database schema migrations
- Integration connector updates
- Certificate and credential updates

### 1.3 Critical Considerations for Emissions Systems

**Regulatory Requirements During Rollback:**

| Requirement | Impact | Mitigation |
|-------------|--------|------------|
| Continuous monitoring | Data gaps trigger substitute data | Maintain backup data collection |
| Data integrity | 7-year retention requirement | Never delete emissions data |
| Audit trail | All changes must be documented | Log all rollback actions |
| Reporting deadlines | Reports must be submitted on time | Verify reporting capability |

### 1.4 Rollback Authority

| Role | Authorization Level |
|------|---------------------|
| On-Call Engineer | SEV3/SEV4 rollbacks, emergency SEV1/SEV2 |
| Operations Manager | All rollbacks |
| VP Engineering | Rollbacks affecting multiple facilities |
| Compliance Lead | Must approve rollbacks affecting compliance |

---

## 2. Pre-Rollback Assessment

### 2.1 Pre-Rollback Checklist

Before initiating any rollback, complete this assessment:

```markdown
## Pre-Rollback Assessment Checklist

**Incident Information:**
- [ ] Incident ID: _______________
- [ ] Severity: SEV___
- [ ] Components affected: _______________
- [ ] Duration of issue: _______________

**Impact Assessment:**
- [ ] Facilities affected: _______________
- [ ] Pollutants affected: _______________
- [ ] Data collection impacted: Yes / No
- [ ] Compliance calculations affected: Yes / No
- [ ] Reports pending submission: Yes / No

**Rollback Feasibility:**
- [ ] Rollback target version identified: _______________
- [ ] Rollback target known to be stable: Yes / No
- [ ] Data migration reversibility verified: Yes / No
- [ ] Configuration backup available: Yes / No
- [ ] Database backup available: Yes / No

**Risk Assessment:**
- [ ] Estimated rollback duration: _______________
- [ ] Data loss risk: None / Minimal / Significant
- [ ] Compliance impact: None / Low / High
- [ ] Regulatory notification required: Yes / No

**Approvals Required:**
- [ ] On-Call Engineer: _______________
- [ ] Operations Manager (if SEV1/SEV2): _______________
- [ ] Compliance Lead (if compliance affected): _______________

**Go/No-Go Decision:**
- [ ] PROCEED WITH ROLLBACK
- [ ] DO NOT PROCEED - Escalate to: _______________
```

### 2.2 Pre-Rollback Commands

**Capture Current State:**

```bash
# Capture current deployment state
greenlang deployment snapshot \
  --agent GL-010 \
  --output /backups/pre-rollback/deployment-$(date +%Y%m%d%H%M%S).json

# Capture current configuration
greenlang config export \
  --agent GL-010 \
  --output /backups/pre-rollback/config-$(date +%Y%m%d%H%M%S).json

# Capture database state
greenlang db snapshot \
  --database gl010-emissions-db \
  --output /backups/pre-rollback/db-$(date +%Y%m%d%H%M%S).sql

# Verify CEMS data is being buffered
greenlang cems verify-buffer \
  --all-facilities \
  --confirm-active

# Check pending reports
greenlang report list \
  --status pending,in-progress \
  --show-deadlines
```

**Verify Rollback Target:**

```bash
# Check available rollback versions
greenlang deployment versions \
  --agent GL-010 \
  --last 10

# Verify target version health history
greenlang deployment health-history \
  --agent GL-010 \
  --version {target-version}

# Check target version compatibility
greenlang deployment compatibility-check \
  --agent GL-010 \
  --from-version {current} \
  --to-version {target}
```

### 2.3 Emissions Data Continuity Check

**Critical: Ensure emissions data capture during rollback:**

```bash
# Verify backup data collection is active
greenlang cems backup-collection status

# Enable emergency data buffering if needed
greenlang cems enable-emergency-buffer \
  --all-facilities \
  --duration 4h

# Check data buffering capacity
greenlang cems buffer-capacity \
  --all-facilities

# Set up data reconciliation tracking
greenlang data track-reconciliation \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --reason "Rollback procedure"
```

---

## 3. Rollback Decision Matrix

### 3.1 When to Rollback

| Scenario | Recommendation | Considerations |
|----------|----------------|----------------|
| New version causing data collection failure | Immediate rollback | Data gaps are regulatory violations |
| New version causing calculation errors | Immediate rollback | Incorrect compliance determinations |
| New version causing performance degradation | Assess severity | May be acceptable temporarily |
| New version causing intermittent issues | Monitor closely | May resolve with configuration |
| New version causing reporting failures | Rollback if deadline imminent | Reports can be resubmitted |
| New version causing UI issues only | Generally no rollback | Unless blocking critical functions |

### 3.2 When NOT to Rollback

| Scenario | Action | Rationale |
|----------|--------|-----------|
| Minor bugs with workaround | Apply workaround | Rollback introduces its own risks |
| Performance within acceptable range | Optimize instead | Rollback may not improve |
| Issue in non-critical component | Fix forward | Less disruptive |
| Data already migrated irreversibly | Fix forward | Cannot rollback data |
| Regulatory requirement in new version | Fix forward | Rollback would cause non-compliance |

### 3.3 Decision Flowchart

```
                        ISSUE IDENTIFIED
                              │
                              ▼
              ┌───────────────────────────────────┐
              │  Is emissions data being lost?     │
              └───────────────────────────────────┘
                    │Yes              │No
                    ▼                 ▼
            ┌─────────────┐    ┌─────────────────────────┐
            │  IMMEDIATE   │    │ Is compliance affected?  │
            │  ROLLBACK    │    └─────────────────────────┘
            └─────────────┘          │Yes         │No
                                     ▼            ▼
                              ┌─────────────┐ ┌─────────────────┐
                              │  Can issue   │ │ Is there a      │
                              │  be fixed    │ │ reporting       │
                              │  within 1hr? │ │ deadline?       │
                              └─────────────┘ └─────────────────┘
                                │No    │Yes      │Yes    │No
                                ▼      ▼         ▼       ▼
                            ROLLBACK  FIX    ASSESS   ASSESS
                                           URGENCY  & FIX
```

---

## 4. Deployment Rollback

### 4.1 Kubernetes Deployment Rollback

**Scenario: Application deployment causing issues**

#### Step 1: Identify Current and Target Versions

```bash
# Check current deployment
kubectl get deployment gl-010-emissionwatch -n gl-agents -o jsonpath='{.spec.template.spec.containers[0].image}'

# Check deployment history
kubectl rollout history deployment/gl-010-emissionwatch -n gl-agents

# Identify target revision
kubectl rollout history deployment/gl-010-emissionwatch -n gl-agents --revision=3
```

#### Step 2: Execute Rollback

**Option A: Rollback to Previous Version**

```bash
# Rollback to immediately previous version
kubectl rollout undo deployment/gl-010-emissionwatch -n gl-agents

# Monitor rollback progress
kubectl rollout status deployment/gl-010-emissionwatch -n gl-agents --watch
```

**Option B: Rollback to Specific Revision**

```bash
# Rollback to specific revision
kubectl rollout undo deployment/gl-010-emissionwatch -n gl-agents --to-revision=5

# Monitor rollback progress
kubectl rollout status deployment/gl-010-emissionwatch -n gl-agents --watch
```

**Option C: Rollback Using Helm (if Helm deployed)**

```bash
# Check Helm release history
helm history gl-010-emissionwatch -n gl-agents

# Rollback to specific revision
helm rollback gl-010-emissionwatch 3 -n gl-agents

# Verify rollback
helm status gl-010-emissionwatch -n gl-agents
```

#### Step 3: Verify Rollback Success

```bash
# Check pod status
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch

# Check pod logs for errors
kubectl logs -n gl-agents -l app=gl-010-emissionwatch --tail=100

# Verify application health
greenlang health --agent GL-010 --full

# Verify CEMS connectivity
greenlang cems status --all-facilities

# Verify data flow
greenlang data flow-status --agent GL-010
```

### 4.2 Canary Deployment Rollback

**Scenario: Canary deployment showing issues**

```bash
# Check canary status
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch,track=canary

# Terminate canary
kubectl delete deployment gl-010-emissionwatch-canary -n gl-agents

# Or scale down canary
kubectl scale deployment gl-010-emissionwatch-canary -n gl-agents --replicas=0

# Verify traffic routing (if using Istio)
kubectl get virtualservice gl-010-emissionwatch -n gl-agents -o yaml

# Remove canary from traffic routing
kubectl patch virtualservice gl-010-emissionwatch -n gl-agents --type=merge -p '
{
  "spec": {
    "http": [{
      "route": [{
        "destination": {
          "host": "gl-010-emissionwatch",
          "subset": "stable"
        },
        "weight": 100
      }]
    }]
  }
}'
```

### 4.3 Blue-Green Deployment Rollback

**Scenario: Green deployment has issues, switch back to Blue**

```bash
# Check current active deployment
kubectl get service gl-010-emissionwatch -n gl-agents -o jsonpath='{.spec.selector}'

# Switch traffic back to blue
kubectl patch service gl-010-emissionwatch -n gl-agents -p '
{
  "spec": {
    "selector": {
      "app": "gl-010-emissionwatch",
      "version": "blue"
    }
  }
}'

# Verify traffic switch
kubectl get endpoints gl-010-emissionwatch -n gl-agents

# Optionally scale down green
kubectl scale deployment gl-010-emissionwatch-green -n gl-agents --replicas=0
```

### 4.4 Multi-Facility Deployment Rollback

**Scenario: Rolling deployment across facilities showing issues**

```bash
# Pause ongoing rollout
kubectl rollout pause deployment/gl-010-emissionwatch -n gl-agents

# Check which facilities are affected
greenlang deployment facility-status --agent GL-010

# Rollback affected facilities
for facility in facility-001 facility-002 facility-003; do
  greenlang deployment rollback \
    --agent GL-010 \
    --facility $facility \
    --to-version v2.3.5
done

# Resume rollout with remaining facilities (if proceeding)
# kubectl rollout resume deployment/gl-010-emissionwatch -n gl-agents

# Or complete rollback for all
kubectl rollout undo deployment/gl-010-emissionwatch -n gl-agents
```

---

## 5. Configuration Rollback

### 5.1 Application Configuration Rollback

**Scenario: Configuration change causing issues**

#### Step 1: Identify Configuration Change

```bash
# View configuration change history
greenlang config history \
  --agent GL-010 \
  --last 10

# Compare current with previous
greenlang config diff \
  --agent GL-010 \
  --version current \
  --compare-to previous

# Show specific change details
greenlang config show-change \
  --change-id config-change-20251126-001
```

#### Step 2: Execute Configuration Rollback

**Option A: Rollback to Previous Configuration**

```bash
# Rollback to previous configuration version
greenlang config rollback \
  --agent GL-010 \
  --to-version previous \
  --reason "Configuration causing calculation errors"

# Apply the rollback
greenlang config apply \
  --agent GL-010 \
  --confirm
```

**Option B: Rollback Specific Setting**

```bash
# Rollback specific configuration path
greenlang config rollback-path \
  --agent GL-010 \
  --path "emissions.NOx.o2_reference" \
  --to-version "2025-11-25T10:00:00Z" \
  --reason "Incorrect O2 reference causing calculation errors"
```

**Option C: Rollback from Backup**

```bash
# List available configuration backups
greenlang config backups \
  --agent GL-010 \
  --last 30d

# Restore from specific backup
greenlang config restore \
  --agent GL-010 \
  --backup-id backup-20251125-120000 \
  --confirm
```

#### Step 3: Verify Configuration Rollback

```bash
# Verify configuration applied
greenlang config verify \
  --agent GL-010 \
  --expected-version {version}

# Test emissions calculations
greenlang emissions test-calculation \
  --facility facility-001 \
  --pollutant NOx \
  --test-data sample

# Verify compliance checking
greenlang compliance test \
  --facility facility-001 \
  --test-data sample
```

### 5.2 Facility Configuration Rollback

**Scenario: Facility-specific configuration causing issues**

```bash
# Rollback facility configuration
greenlang config rollback \
  --agent GL-010 \
  --facility facility-001 \
  --to-version previous \
  --reason "Facility configuration error"

# Verify facility-specific settings
greenlang config verify \
  --agent GL-010 \
  --facility facility-001

# Test facility calculations
greenlang emissions verify \
  --facility facility-001 \
  --all-pollutants \
  --time-range "last-1h"
```

### 5.3 Emission Factor Rollback

**Scenario: Emission factor update causing incorrect calculations**

```bash
# View emission factor change history
greenlang emissions emission-factors history \
  --facility facility-001 \
  --pollutant NOx

# Rollback emission factor
greenlang emissions emission-factors rollback \
  --facility facility-001 \
  --pollutant NOx \
  --to-version previous \
  --reason "Stack test factor incorrectly applied"

# Recalculate affected emissions
greenlang emissions recalculate \
  --facility facility-001 \
  --pollutant NOx \
  --time-range "2025-11-26T00:00:00Z/now" \
  --reason "Emission factor rollback"
```

### 5.4 Regulatory Limit Rollback

**Scenario: Limit update was incorrect**

```bash
# View limit change history
greenlang compliance limits history \
  --facility facility-001 \
  --pollutant NOx

# Rollback limit change
greenlang compliance limits rollback \
  --facility facility-001 \
  --pollutant NOx \
  --to-version previous \
  --reason "Incorrect limit applied"

# Recalculate compliance for affected period
greenlang compliance recalculate \
  --facility facility-001 \
  --pollutant NOx \
  --time-range "2025-11-26T00:00:00Z/now"
```

---

## 6. Database Migration Rollback

### 6.1 Pre-Migration Rollback Assessment

**Critical: Assess data reversibility before rollback:**

```bash
# Check migration history
greenlang db migrations history \
  --database gl010-emissions-db \
  --last 10

# Check if migration is reversible
greenlang db migrations check-reversibility \
  --database gl010-emissions-db \
  --migration-id migration-20251126-001

# Analyze data changes
greenlang db migrations analyze-changes \
  --database gl010-emissions-db \
  --migration-id migration-20251126-001
```

### 6.2 Schema Migration Rollback

**Scenario: Database schema change causing issues**

#### Step 1: Verify Rollback Feasibility

```bash
# Check for down migration
greenlang db migrations has-down \
  --database gl010-emissions-db \
  --migration-id migration-20251126-001

# Preview rollback
greenlang db migrations preview-rollback \
  --database gl010-emissions-db \
  --migration-id migration-20251126-001
```

#### Step 2: Execute Schema Rollback

```bash
# Stop application to prevent writes
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=0

# Verify no active connections
greenlang db connections \
  --database gl010-emissions-db \
  --active-only

# Execute rollback
greenlang db migrations rollback \
  --database gl010-emissions-db \
  --migration-id migration-20251126-001 \
  --confirm

# Verify schema state
greenlang db schema-version \
  --database gl010-emissions-db
```

#### Step 3: Restore Application

```bash
# Deploy compatible application version
kubectl set image deployment/gl-010-emissionwatch \
  -n gl-agents \
  emissionwatch=greenlang/gl-010-emissionwatch:v2.3.5

# Scale up application
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=3

# Verify application health
greenlang health --agent GL-010 --full
```

### 6.3 Data Migration Rollback

**Scenario: Data migration corrupted or transformed data incorrectly**

```bash
# Check data migration status
greenlang db data-migration status \
  --database gl010-emissions-db \
  --migration-id data-migration-20251126-001

# If reversible, execute rollback
greenlang db data-migration rollback \
  --database gl010-emissions-db \
  --migration-id data-migration-20251126-001 \
  --confirm

# If not reversible, restore from backup
greenlang db restore \
  --database gl010-emissions-db \
  --backup-id backup-20251126-pre-migration \
  --tables "affected_table_1,affected_table_2"

# Verify data integrity
greenlang db verify-integrity \
  --database gl010-emissions-db \
  --tables "affected_table_1,affected_table_2"
```

### 6.4 Full Database Rollback

**Scenario: Critical database issue requiring full restoration**

**WARNING: This is a destructive operation. Use only when necessary.**

```bash
# Stop all applications
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=0

# Verify no active connections
greenlang db connections --database gl010-emissions-db --terminate-all

# List available backups
greenlang db backups list \
  --database gl010-emissions-db \
  --last 7d

# Restore from backup
greenlang db restore \
  --database gl010-emissions-db \
  --backup-id backup-20251125-000000 \
  --confirm

# Verify restoration
greenlang db verify-backup-restore \
  --database gl010-emissions-db \
  --backup-id backup-20251125-000000

# Reconcile any data collected since backup
greenlang data reconcile \
  --database gl010-emissions-db \
  --from-buffer \
  --time-range "2025-11-25T00:00:00Z/now"

# Restart application
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=3
```

---

## 7. Integration Update Rollback

### 7.1 CEMS Connector Rollback

**Scenario: CEMS connector update causing data collection issues**

```bash
# Check connector status
greenlang connector status cems-connector

# View connector version history
greenlang connector versions cems-connector --last 5

# Rollback connector
greenlang connector rollback cems-connector \
  --to-version v1.4.2 \
  --reason "Data collection failure with v1.5.0"

# Restart connector
greenlang connector restart cems-connector

# Verify data flow
greenlang cems verify-data-flow \
  --all-facilities \
  --duration 5m
```

### 7.2 Regulatory Portal Connector Rollback

**Scenario: EPA CEDRI connector update causing submission failures**

```bash
# Check integration status
greenlang integration status EPA_CEDRI

# View integration version history
greenlang integration versions EPA_CEDRI --last 5

# Rollback integration
greenlang integration rollback EPA_CEDRI \
  --to-version v2.1.0 \
  --reason "XML schema compatibility issue with v2.2.0"

# Test connection
greenlang integration test EPA_CEDRI --verbose

# Retry failed submissions
greenlang report retry-submissions \
  --portal EPA_CEDRI \
  --status failed \
  --time-range "last-24h"
```

### 7.3 OPC-UA/Modbus Protocol Update Rollback

**Scenario: Protocol library update causing communication issues**

```bash
# Check protocol library versions
greenlang cems protocol-versions

# Rollback protocol library
greenlang cems rollback-protocol \
  --protocol modbus \
  --to-version v3.2.1 \
  --reason "Timeout issues with v3.3.0"

# Restart affected connectors
greenlang connector restart-all \
  --protocol modbus

# Verify communications
greenlang cems test-all-connections \
  --protocol modbus
```

### 7.4 Certificate/Credential Update Rollback

**Scenario: New certificate causing authentication failures**

```bash
# Check certificate status
greenlang certificate status \
  --purpose cems-authentication

# Rollback to previous certificate
greenlang certificate rollback \
  --purpose cems-authentication \
  --to-version previous \
  --reason "New certificate not trusted by CEMS server"

# Restart affected connections
greenlang connector restart-all \
  --uses-certificate cems-authentication

# Verify authentication
greenlang cems test-authentication \
  --all-facilities
```

---

## 8. Post-Rollback Validation

### 8.1 Core Functionality Validation

```bash
# Run comprehensive post-rollback checks
greenlang validate post-rollback \
  --agent GL-010 \
  --comprehensive

# Or run individual checks:

# 1. System Health
greenlang health --agent GL-010 --full

# 2. CEMS Connectivity
greenlang cems status --all-facilities --verify-data

# 3. Emissions Calculations
greenlang emissions verify-calculations \
  --all-facilities \
  --time-range "last-1h" \
  --compare-with-expected

# 4. Compliance Status
greenlang compliance verify \
  --all-facilities \
  --check-limits \
  --check-averaging

# 5. Reporting Capability
greenlang report generate-test \
  --type quarterly-emissions \
  --facility facility-001 \
  --verify-schema
```

### 8.2 Data Integrity Validation

```bash
# Check for data gaps during rollback
greenlang data check-gaps \
  --agent GL-010 \
  --time-range "rollback-start/rollback-end"

# Verify data reconciliation
greenlang data verify-reconciliation \
  --agent GL-010 \
  --time-range "rollback-start/now"

# Check calculation consistency
greenlang emissions verify-consistency \
  --all-facilities \
  --time-range "rollback-start/now"

# Verify audit trail
greenlang audit verify \
  --agent GL-010 \
  --time-range "rollback-start/now"
```

### 8.3 Emissions Data Validation

```bash
# Verify emissions data completeness
greenlang emissions data-completeness \
  --all-facilities \
  --time-range "rollback-start/now"

# Compare with CEMS source data
greenlang emissions compare-with-source \
  --all-facilities \
  --time-range "rollback-start/now" \
  --tolerance 0.02

# Verify regulatory data quality
greenlang cems data-quality \
  --all-facilities \
  --period "today" \
  --check-regulatory-threshold
```

### 8.4 Compliance Validation

```bash
# Verify compliance calculations
greenlang compliance recalculate \
  --all-facilities \
  --time-range "rollback-start/now" \
  --verify-only

# Check for any new violations
greenlang compliance check-violations \
  --all-facilities \
  --time-range "rollback-start/now"

# Verify regulatory limits are correct
greenlang compliance verify-limits \
  --all-facilities \
  --compare-with-permit
```

### 8.5 Report Generation Validation

```bash
# Test report generation
greenlang report generate-test \
  --type quarterly-emissions \
  --facility facility-001 \
  --period "Q4-2025"

# Verify report data accuracy
greenlang report verify-data \
  --report-id test-quarterly-Q4-2025-facility-001 \
  --compare-with-source

# Test regulatory submission (dry run)
greenlang report submit \
  --report-id test-quarterly-Q4-2025-facility-001 \
  --portal EPA_CEDRI \
  --dry-run
```

### 8.6 Post-Rollback Validation Checklist

```markdown
## Post-Rollback Validation Checklist

**Rollback Information:**
- Rollback ID: _______________
- Rollback Time: _______________
- Rollback Duration: _______________

**System Health:**
- [ ] All pods running and healthy
- [ ] No error logs in last 15 minutes
- [ ] Resource utilization normal
- [ ] All dependencies accessible

**Data Collection:**
- [ ] CEMS data being received from all facilities
- [ ] Data quality scores within normal range
- [ ] No missing data periods detected
- [ ] Data reconciliation complete

**Calculations:**
- [ ] Emissions calculations producing expected results
- [ ] No calculation errors in logs
- [ ] Results match CEMS source data (within tolerance)
- [ ] All pollutants being calculated

**Compliance:**
- [ ] Compliance status accurate for all facilities
- [ ] Regulatory limits correctly applied
- [ ] Averaging periods calculating correctly
- [ ] No false violations detected

**Reporting:**
- [ ] Report generation working
- [ ] Report schema validation passing
- [ ] Regulatory portal connectivity verified
- [ ] Pending reports status unchanged

**Validation Completed By:** _______________
**Date/Time:** _______________
**Notes:** _______________
```

---

## 9. Regulatory Considerations

### 9.1 Notification Requirements During Outages

| Outage Duration | Notification Required | Agency | Timeline |
|-----------------|----------------------|--------|----------|
| < 1 hour | Internal documentation only | N/A | N/A |
| 1-4 hours | May require notification | State agency | Within 24 hours |
| > 4 hours | Notification required | EPA, State | Same day |
| > 24 hours | Formal notification | EPA, State | Immediate |

### 9.2 Data Gap Handling

**For Data Gaps During Rollback:**

```bash
# Identify data gaps
greenlang cems data-gaps \
  --all-facilities \
  --time-range "rollback-start/rollback-end"

# Apply substitute data per regulatory requirements
greenlang cems apply-substitute-data \
  --facility facility-001 \
  --time-range "gap-start/gap-end" \
  --method "40-CFR-75-appendix-D" \
  --reason "System rollback - no data collection"

# Document gaps for regulatory records
greenlang cems document-gaps \
  --all-facilities \
  --time-range "rollback-start/rollback-end" \
  --include-substitute-data \
  --output /reports/data-gaps/rollback-$(date +%Y%m%d).pdf
```

### 9.3 Substitute Data Procedures

**Part 75 Substitute Data Requirements:**

```bash
# Calculate substitute data for NOx
greenlang cems substitute-data calculate \
  --facility facility-001 \
  --pollutant NOx \
  --gap-start "2025-11-26T10:00:00Z" \
  --gap-end "2025-11-26T12:00:00Z" \
  --method "90th-percentile-lookback"

# Apply substitute data
greenlang cems substitute-data apply \
  --facility facility-001 \
  --pollutant NOx \
  --gap-start "2025-11-26T10:00:00Z" \
  --gap-end "2025-11-26T12:00:00Z" \
  --value 65.3 \
  --method "90th-percentile-lookback" \
  --reason "Data gap during system rollback"
```

### 9.4 Regulatory Reporting Impact

```bash
# Check impact on pending reports
greenlang report impact-assessment \
  --event "rollback" \
  --time-range "rollback-start/rollback-end"

# Update affected quarterly reports
greenlang report update-for-data-gap \
  --report-id quarterly-2025-Q4-facility-001 \
  --gap-start "2025-11-26T10:00:00Z" \
  --gap-end "2025-11-26T12:00:00Z" \
  --flag-substitute-data

# Generate regulatory notification if required
greenlang regulatory generate-notification \
  --type "system-outage" \
  --facility facility-001 \
  --outage-start "2025-11-26T10:00:00Z" \
  --outage-end "2025-11-26T12:00:00Z" \
  --cause "System rollback to address software issue" \
  --corrective-action "System restored, data reconciled"
```

### 9.5 Audit Trail Maintenance

```bash
# Ensure rollback is documented in audit trail
greenlang audit log \
  --action "system-rollback" \
  --agent GL-010 \
  --from-version v2.4.0 \
  --to-version v2.3.5 \
  --reason "Production issue with v2.4.0 calculations" \
  --authorized-by "John Smith" \
  --approval-reference "CHG-2025-1126"

# Generate audit report
greenlang audit report \
  --agent GL-010 \
  --time-range "rollback-start/rollback-end" \
  --output /reports/audit/rollback-audit-$(date +%Y%m%d).pdf
```

---

## 10. Emergency Rollback Procedures

### 10.1 Emergency Rollback (< 5 Minutes)

**Use when: Active data loss or compliance violation**

```bash
# EMERGENCY ROLLBACK PROCEDURE
# Execute as quickly as possible

# Step 1: Immediate rollback (30 seconds)
kubectl rollout undo deployment/gl-010-emissionwatch -n gl-agents

# Step 2: Verify pods starting (1 minute)
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch -w

# Step 3: Check data flow resumed (2 minutes)
greenlang cems verify-data-flow --quick

# Step 4: Notify stakeholders (1 minute)
greenlang notify emergency-rollback \
  --agent GL-010 \
  --status "complete" \
  --verify-data-flow "yes"

# Step 5: Document (ongoing)
greenlang incident update \
  --incident-id {incident-id} \
  --action "emergency-rollback-complete" \
  --notes "Rollback to v2.3.5 completed, data flow verified"
```

### 10.2 Emergency Database Rollback

**Use when: Critical database issue affecting data integrity**

```bash
# EMERGENCY DATABASE ROLLBACK
# WARNING: This will interrupt service

# Step 1: Stop application immediately
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=0

# Step 2: Enable emergency data buffering at CEMS
greenlang cems enable-emergency-buffer --all-facilities --duration 2h

# Step 3: Restore database from latest backup
greenlang db emergency-restore \
  --database gl010-emissions-db \
  --latest-backup \
  --skip-confirmation

# Step 4: Verify database
greenlang db verify-quick --database gl010-emissions-db

# Step 5: Start application
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=3

# Step 6: Reconcile buffered data
greenlang data reconcile --from-buffer --all-facilities
```

### 10.3 Emergency Configuration Rollback

**Use when: Configuration change causing immediate compliance issues**

```bash
# EMERGENCY CONFIGURATION ROLLBACK

# Step 1: Rollback to last known good configuration
greenlang config emergency-rollback \
  --agent GL-010 \
  --to "last-known-good"

# Step 2: Force configuration reload
greenlang config force-reload --agent GL-010

# Step 3: Verify critical functions
greenlang health --agent GL-010 --critical-only

# Step 4: Verify compliance calculations
greenlang compliance quick-verify --all-facilities
```

### 10.4 Emergency Contact During Rollback

```
EMERGENCY CONTACTS DURING ROLLBACK:

On-Call Engineer:     +1-555-0100
Operations Manager:   +1-555-0102
Compliance Lead:      +1-555-0103
Database Admin:       +1-555-0108
VP Engineering:       +1-555-0104

Slack: #gl-010-emergency
Bridge: https://meet.greenlang.io/emergency
```

---

## 11. Rollback Communication

### 11.1 Rollback Notification Template

**Subject:** [GL-010] System Rollback - {Rollback Type} - {Status}

```markdown
## GL-010 EMISSIONWATCH Rollback Notification

**Status:** {In Progress / Completed}
**Rollback Type:** {Deployment / Configuration / Database / Integration}
**Time Started:** {timestamp}
**Time Completed:** {timestamp or "In Progress"}

### Summary
{Brief description of why rollback was initiated}

### Impact
- **Facilities Affected:** {list or "All"}
- **Service Interruption:** {duration}
- **Data Impact:** {None / Minimal / Describe}

### Current Status
{Current state description}

### Actions Taken
1. {Action 1}
2. {Action 2}
3. {Action 3}

### Validation Status
- [ ] System health verified
- [ ] Data flow verified
- [ ] Compliance calculations verified
- [ ] Reporting capability verified

### Next Steps
{Description of next steps}

### Contact
Incident Commander: {name}
Email: {email}
Phone: {phone}

---
This notification generated by GL-010 Operations
```

### 11.2 Stakeholder Communication Matrix

| Stakeholder | Notification Trigger | Method | Timeline |
|-------------|---------------------|--------|----------|
| Operations Team | All rollbacks | Slack + Email | Immediate |
| Facility Contacts | Their facility affected | Email | Within 15 min |
| Compliance Team | Any compliance impact | Phone + Email | Immediate |
| Management | SEV1/SEV2 rollbacks | Phone + Email | Within 30 min |
| Regulatory Affairs | Regulatory impact | Phone + Email | Immediate |

### 11.3 Post-Rollback Report

```markdown
## Post-Rollback Report

**Report Date:** {date}
**Rollback ID:** {id}
**Author:** {name}

### Rollback Summary
| Property | Value |
|----------|-------|
| Type | {Deployment/Config/Database/Integration} |
| Start Time | {timestamp} |
| End Time | {timestamp} |
| Duration | {duration} |
| From Version | {version} |
| To Version | {version} |

### Reason for Rollback
{Detailed explanation of why rollback was necessary}

### Rollback Procedure Followed
1. {Step 1}
2. {Step 2}
3. {Step 3}

### Data Impact
- **Data Collection Gap:** {Yes/No - duration}
- **Substitute Data Applied:** {Yes/No}
- **Data Reconciliation Required:** {Yes/No}
- **Regulatory Notification Required:** {Yes/No}

### Validation Results
| Check | Result | Notes |
|-------|--------|-------|
| System Health | Pass/Fail | {notes} |
| Data Flow | Pass/Fail | {notes} |
| Calculations | Pass/Fail | {notes} |
| Compliance | Pass/Fail | {notes} |
| Reporting | Pass/Fail | {notes} |

### Root Cause (if known)
{Description of root cause}

### Follow-up Actions
| Action | Owner | Due Date |
|--------|-------|----------|
| {action} | {owner} | {date} |

### Lessons Learned
1. {Lesson 1}
2. {Lesson 2}

### Attachments
- Rollback logs
- Validation results
- Data reconciliation report

---
Prepared by: {name}
Reviewed by: {name}
Date: {date}
```

---

## 12. Appendices

### Appendix A: Rollback Quick Reference

| Scenario | Command | Section |
|----------|---------|---------|
| Kubernetes rollback | `kubectl rollout undo deployment/gl-010-emissionwatch -n gl-agents` | 4.1 |
| Helm rollback | `helm rollback gl-010-emissionwatch {revision} -n gl-agents` | 4.1 |
| Config rollback | `greenlang config rollback --agent GL-010 --to-version previous` | 5.1 |
| DB migration rollback | `greenlang db migrations rollback --migration-id {id}` | 6.2 |
| Integration rollback | `greenlang integration rollback {name} --to-version {version}` | 7.2 |
| Emergency rollback | `kubectl rollout undo deployment/gl-010-emissionwatch -n gl-agents` | 10.1 |

### Appendix B: Version Compatibility Matrix

| Application Version | DB Schema Version | Config Version | Compatible With |
|--------------------|-------------------|----------------|-----------------|
| v2.4.0 | v15 | v8 | v2.3.x, v2.2.x |
| v2.3.5 | v14 | v7 | v2.3.x, v2.2.x |
| v2.3.0 | v14 | v7 | v2.2.x |
| v2.2.0 | v13 | v6 | v2.1.x |

### Appendix C: Rollback Testing Schedule

| Test Type | Frequency | Last Test | Next Test |
|-----------|-----------|-----------|-----------|
| Deployment rollback | Monthly | {date} | {date} |
| Configuration rollback | Monthly | {date} | {date} |
| Database rollback | Quarterly | {date} | {date} |
| Emergency rollback drill | Quarterly | {date} | {date} |

### Appendix D: Related Documentation

| Document | Location |
|----------|----------|
| Incident Response | ./INCIDENT_RESPONSE.md |
| Troubleshooting | ./TROUBLESHOOTING.md |
| Scaling Guide | ./SCALING_GUIDE.md |
| Maintenance Guide | ./MAINTENANCE.md |
| Deployment Guide | /docs/deployment/gl-010/ |

### Appendix E: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-26 | GL-TechWriter | Initial release |

---

**Document Classification:** Internal Use Only

**Next Review Date:** 2026-02-26

**Feedback:** Submit feedback to docs@greenlang.io with subject "GL-010 Rollback Procedure Feedback"
