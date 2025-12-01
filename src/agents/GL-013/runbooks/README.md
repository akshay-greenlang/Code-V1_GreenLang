# GL-013 PREDICTMAINT - Runbooks

```
================================================================================
                    OPERATIONAL RUNBOOKS - GL-013 PREDICTMAINT
                         Standard Operating Procedures
================================================================================
```

**Version:** 1.0.0
**Last Updated:** 2024-12-01
**Owner:** Site Reliability Engineering (SRE) Team
**Review Cycle:** Quarterly

---

## Overview

This directory contains operational runbooks for GL-013 PREDICTMAINT. These runbooks provide step-by-step procedures for common operational tasks, incident response, and troubleshooting.

---

## Runbook Index

| Runbook | Purpose | On-Call Use |
|---------|---------|-------------|
| [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) | Incident handling procedures | Primary |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Diagnostic and resolution steps | Primary |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment procedures | Secondary |
| [SCALING.md](SCALING.md) | Scaling operations | Secondary |
| [BACKUP_RESTORE.md](BACKUP_RESTORE.md) | Backup and recovery | Secondary |

---

## Quick Reference

### Health Check Commands

```bash
# Check application health
curl -s http://localhost:8000/health | jq .

# Check readiness
curl -s http://localhost:8000/health/ready | jq .

# Check Kubernetes pods
kubectl get pods -n greenlang -l app=gl-013-predictmaint

# Check pod logs
kubectl logs -n greenlang -l app=gl-013-predictmaint --tail=100

# Check metrics endpoint
curl -s http://localhost:9090/metrics | head -50
```

### Common Alerts

| Alert | Severity | First Response |
|-------|----------|----------------|
| HighFailureProbability | Critical | Notify maintenance team |
| LowHealthScore | Critical | Investigate equipment |
| PredictionLatencyHigh | Warning | Check resources/cache |
| IntegrationDown | Warning | Check connector status |
| CacheHitRatioLow | Warning | Review cache strategy |

### Escalation Path

1. **L1 - On-Call Engineer** (0-15 min)
   - Initial triage
   - Basic troubleshooting
   - Escalate if unresolved

2. **L2 - Platform Team** (15-60 min)
   - Deep investigation
   - Configuration changes
   - Escalate if service impact

3. **L3 - Engineering Team** (60+ min)
   - Code-level debugging
   - Emergency patches
   - Architecture decisions

### Contact Information

| Role | Contact | Availability |
|------|---------|--------------|
| On-Call Engineer | PagerDuty | 24/7 |
| Platform Team | #platform-team | Business hours |
| Engineering Lead | @eng-lead | Business hours |
| Product Owner | @product-owner | Business hours |

---

## Runbook Standards

### Document Structure

Each runbook follows this structure:

1. **Overview** - Purpose and scope
2. **Prerequisites** - Required access and tools
3. **Procedure** - Step-by-step instructions
4. **Verification** - How to confirm success
5. **Rollback** - How to undo changes
6. **References** - Related documentation

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| SEV1 | Service down, all users affected | 15 minutes |
| SEV2 | Major feature broken, many users affected | 30 minutes |
| SEV3 | Minor feature broken, some users affected | 4 hours |
| SEV4 | Bug or enhancement request | Next sprint |

### Change Management

All changes must follow the change management process:

1. Create change request ticket
2. Get approval from change board
3. Schedule maintenance window (if needed)
4. Execute change with rollback plan
5. Verify change success
6. Close change request

---

## Environment Information

### Production Environment

| Component | Endpoint | Port |
|-----------|----------|------|
| API Gateway | api.greenlang.io | 443 |
| Metrics | metrics.greenlang.io | 9090 |
| Grafana | grafana.greenlang.io | 3000 |
| PostgreSQL | postgres.greenlang.io | 5432 |
| Redis | redis.greenlang.io | 6379 |

### Kubernetes Namespaces

| Namespace | Purpose |
|-----------|---------|
| greenlang | Production workloads |
| greenlang-staging | Staging environment |
| monitoring | Prometheus, Grafana |
| logging | ELK stack |

---

## Tools and Access

### Required Tools

- kubectl (v1.28+)
- helm (v3.12+)
- psql (PostgreSQL client)
- redis-cli (Redis client)
- jq (JSON processor)
- curl (HTTP client)

### Access Requirements

| System | Access Method | Request Process |
|--------|---------------|-----------------|
| Kubernetes | kubectl + RBAC | IT Service Desk |
| PostgreSQL | psql + credentials | Vault request |
| Redis | redis-cli + auth | Vault request |
| Grafana | SSO | Auto-provisioned |
| PagerDuty | Email invite | Manager request |

---

## Related Documentation

- [README.md](../README.md) - Main documentation
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Architecture guide
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
- [deployment/](../deployment/) - Deployment manifests

---

```
================================================================================
                    GL-013 PREDICTMAINT - Operational Runbooks
                         GreenLang Inc. - SRE Team
================================================================================
```
