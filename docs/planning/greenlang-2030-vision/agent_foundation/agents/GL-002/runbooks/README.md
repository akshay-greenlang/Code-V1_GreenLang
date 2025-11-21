# GL-002 BoilerEfficiencyOptimizer - Operations Runbooks

Comprehensive operational runbooks for GL-002 BoilerEfficiencyOptimizer production operations.

## Overview

This directory contains production operations runbooks covering troubleshooting, incident response, rollback procedures, and scaling operations for the GL-002 BoilerEfficiencyOptimizer agent.

## Runbooks Index

### 1. TROUBLESHOOTING.md

**Purpose**: Diagnose and resolve common production issues

**When to Use**:
- Agent not starting or pods crashing
- High error rates or degraded performance
- Integration failures with SCADA/ERP systems
- Database or cache connectivity issues
- Determinism failures

**Common Scenarios**:
- Configuration errors preventing startup
- Missing environment variables or secrets
- Database connection pool exhaustion
- Redis cache failures
- OOMKilled or CPU throttling
- Integration timeouts

**Quick Links**:
- [Agent Not Starting](TROUBLESHOOTING.md#agent-not-starting)
- [High Error Rates](TROUBLESHOOTING.md#high-error-rates)
- [Performance Degradation](TROUBLESHOOTING.md#performance-degradation)
- [Integration Failures](TROUBLESHOOTING.md#integration-failures)

---

### 2. INCIDENT_RESPONSE.md

**Purpose**: Emergency procedures for production incidents

**When to Use**:
- P0 critical incidents (production down)
- P1 high severity incidents (major degradation)
- P2 medium severity incidents (minor issues)
- Any production issue requiring escalation

**Severity Levels**:
- **P0 (Critical)**: Complete outage, immediate response
- **P1 (High)**: Major degradation, 15-minute response
- **P2 (Medium)**: Minor degradation, 1-hour response
- **P3 (Low)**: Monitoring alerts, 4-hour response

**Includes**:
- Severity definitions
- Response procedures by severity
- Escalation paths and contacts
- Communication templates
- Emergency rollback procedures
- Post-incident review template

**Quick Links**:
- [P0 Response Procedure](INCIDENT_RESPONSE.md#p0---critical-incident-response)
- [Escalation Paths](INCIDENT_RESPONSE.md#escalation-paths)
- [Communication Templates](INCIDENT_RESPONSE.md#communication-templates)
- [Emergency Rollback](INCIDENT_RESPONSE.md#emergency-rollback-procedures)

---

### 3. ROLLBACK_PROCEDURE.md

**Purpose**: Safe version rollback procedures

**When to Use**:
- Recent deployment causing issues (<2 hours)
- Need to revert to previous stable version
- Database migration failures
- Critical bugs discovered in production

**Rollback Methods**:
- **Quick Rollback** (5 minutes): Immediate undo to previous version
- **Specific Revision** (10 minutes): Rollback to specific version
- **Blue-Green** (15 minutes): Zero-downtime rollback
- **ConfigMap/Secret** (2 minutes): Configuration-only rollback

**Includes**:
- When to rollback decision matrix
- Pre-rollback checklist
- Step-by-step rollback procedures
- Verification and validation steps
- Database migration rollback
- Communication templates

**Quick Links**:
- [When to Rollback](ROLLBACK_PROCEDURE.md#when-to-rollback)
- [Emergency Rollback (5 min)](ROLLBACK_PROCEDURE.md#emergency-rollback-5-minutes)
- [Verification Procedures](ROLLBACK_PROCEDURE.md#verification-procedures)
- [Database Migration Rollback](ROLLBACK_PROCEDURE.md#database-migration-rollback)

---

### 4. SCALING_GUIDE.md

**Purpose**: Scale applications to handle load

**When to Use**:
- CPU/Memory usage >80%
- Response times increasing
- Error rates due to capacity
- Planned load increases (campaigns, seasonal)
- Performance optimization

**Scaling Types**:
- **Horizontal Scaling**: Add/remove pods (HPA)
- **Vertical Scaling**: Increase pod resources
- **Database Scaling**: Connection pools, read replicas
- **Cache Scaling**: Redis memory, clustering
- **Multi-Region**: Geographic distribution

**Includes**:
- Scaling triggers and thresholds
- Manual and automatic scaling procedures
- Resource optimization
- Performance testing
- Capacity planning
- Cost optimization

**Quick Links**:
- [When to Scale](SCALING_GUIDE.md#when-to-scale)
- [Horizontal Scaling (HPA)](SCALING_GUIDE.md#horizontal-scaling-hpa)
- [Vertical Scaling](SCALING_GUIDE.md#vertical-scaling-resources)
- [Performance Testing](SCALING_GUIDE.md#performance-testing)

---

## Quick Reference Guide

### Common Commands

#### Check Service Health
```bash
# Pod status
kubectl get pods -n greenlang | grep gl-002

# Health endpoint
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/health

# Error rate
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate

# Resource usage
kubectl top pods -n greenlang | grep gl-002
```

#### Emergency Rollback
```bash
# Rollback to previous version (5 minutes)
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang --timeout=5m
```

#### Scale Pods
```bash
# Manual scale up
kubectl scale deployment gl-002-boiler-efficiency --replicas=5 -n greenlang

# Check HPA status
kubectl get hpa gl-002-boiler-efficiency-hpa -n greenlang
```

#### View Logs
```bash
# Recent logs
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=100

# Follow logs
kubectl logs -f -n greenlang deployment/gl-002-boiler-efficiency

# Previous pod logs (if crashed)
kubectl logs -n greenlang <pod-name> --previous
```

#### Restart Deployment
```bash
# Graceful restart
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# Monitor restart
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
```

---

## Decision Trees

### Issue Resolution Decision Tree

```
Is the service down (all pods failing)?
├─ YES → Use INCIDENT_RESPONSE.md (P0)
│         └─ Recent deployment? → Use ROLLBACK_PROCEDURE.md
└─ NO
   └─ Is error rate >20%?
      ├─ YES → Use INCIDENT_RESPONSE.md (P1)
      │         └─ Recent deployment? → Use ROLLBACK_PROCEDURE.md
      └─ NO
         └─ Is performance degraded?
            ├─ YES → Use TROUBLESHOOTING.md (Performance Degradation)
            │         └─ Resource constraints? → Use SCALING_GUIDE.md
            └─ NO → Use TROUBLESHOOTING.md (specific issue)
```

### Scaling Decision Tree

```
What type of issue?
├─ High CPU/Memory usage → SCALING_GUIDE.md (Vertical or Horizontal Scaling)
├─ Slow response times → TROUBLESHOOTING.md first, then SCALING_GUIDE.md
├─ Database slow → SCALING_GUIDE.md (Database Scaling)
├─ Cache issues → SCALING_GUIDE.md (Cache Scaling)
└─ Planned load increase → SCALING_GUIDE.md (Capacity Planning)
```

---

## On-Call Quick Start

If you're on-call and got paged:

1. **Acknowledge the alert** in PagerDuty (within 5 minutes)

2. **Check severity** (from alert or monitoring):
   - All pods down → P0 (use INCIDENT_RESPONSE.md)
   - High error rate → P1 (use INCIDENT_RESPONSE.md)
   - Performance issues → P2 (use TROUBLESHOOTING.md)

3. **Check recent changes**:
   ```bash
   kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang
   ```
   - If deployed <2 hours ago → Consider rollback (ROLLBACK_PROCEDURE.md)

4. **Assess impact**:
   ```bash
   kubectl get pods -n greenlang | grep gl-002
   kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
     curl -s http://localhost:8000/api/v1/metrics | grep error_rate
   ```

5. **Follow appropriate runbook**:
   - P0/P1 → INCIDENT_RESPONSE.md
   - Need rollback → ROLLBACK_PROCEDURE.md
   - Troubleshooting → TROUBLESHOOTING.md
   - Scaling → SCALING_GUIDE.md

6. **Communicate** (see INCIDENT_RESPONSE.md for templates):
   - Post in #gl-002-incidents Slack channel
   - Update status page (for P0/P1)

---

## Monitoring and Alerting

### Dashboards

- **Grafana Main Dashboard**: https://grafana.greenlang.io/d/gl-002/boiler-efficiency-optimizer
- **Error Analysis**: https://grafana.greenlang.io/d/gl-002-errors/error-analysis
- **Performance Metrics**: https://grafana.greenlang.io/d/gl-002-perf/performance-metrics
- **Scaling Metrics**: https://grafana.greenlang.io/d/gl-002-scaling/scaling-operations

### Key Metrics

```bash
# Check all key metrics at once
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(error_rate|duration|cpu|memory|db_pool|cache_hit)'
```

**Target Metrics**:
- Error rate: <1%
- Response time (p95): <2 seconds
- CPU usage: <70%
- Memory usage: <80%
- Database pool utilization: <80%
- Cache hit rate: >80%

### Alerting Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Error rate | >5% | >20% | TROUBLESHOOTING.md |
| Response time (p95) | >3s | >5s | SCALING_GUIDE.md |
| CPU usage | >70% | >85% | SCALING_GUIDE.md |
| Memory usage | >80% | >90% | SCALING_GUIDE.md |
| Pod count | 0/3 | 0/3 | INCIDENT_RESPONSE.md (P0) |
| Database connections | >80% | >95% | SCALING_GUIDE.md |

---

## Escalation Contacts

### Primary On-Call
- **PagerDuty**: Service ID "GL-002 Primary"
- **Slack**: @gl-002-oncall-primary
- **Response Time**: 5 minutes

### Secondary On-Call
- **PagerDuty**: Service ID "GL-002 Secondary"
- **Slack**: @gl-002-oncall-secondary
- **Response Time**: 10 minutes (if primary doesn't respond)

### Engineering Manager
- **Escalation**: For P0 incidents lasting >1 hour or P1 incidents lasting >4 hours
- **Contact**: See INCIDENT_RESPONSE.md for details

### Director of Engineering
- **Escalation**: For unresolved P0 incidents
- **Contact**: See INCIDENT_RESPONSE.md for details

---

## Runbook Maintenance

### Update Frequency
- Review quarterly or after significant incidents
- Update after architecture changes
- Add new scenarios as discovered

### Contributors
- DevOps Team: Infrastructure procedures
- Platform Engineering: Scaling and performance
- Development Team: Application-specific troubleshooting
- SRE Team: Monitoring and alerting

### Feedback
- Found an issue? Create JIRA ticket with label "runbook-improvement"
- Have a suggestion? Post in #gl-002-team Slack channel
- Need clarification? Ask in #gl-002-ops

---

## Related Documentation

### Internal Documentation
- **Architecture**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\ARCHITECTURE.md`
- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\deployment\README.md`
- **API Documentation**: https://docs.greenlang.io/agents/gl-002/api
- **Integration Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\INTEGRATION_ARCHITECTURE.md`

### External Resources
- **Kubernetes Documentation**: https://kubernetes.io/docs/
- **PostgreSQL Tuning**: https://wiki.postgresql.org/wiki/Performance_Optimization
- **Redis Optimization**: https://redis.io/docs/management/optimization/
- **Prometheus Queries**: https://prometheus.io/docs/prometheus/latest/querying/basics/

---

## Training and Certification

### Required Training
- Kubernetes Operations Basics (2 hours)
- GL-002 Architecture Overview (1 hour)
- Incident Response Protocol (1 hour)
- On-Call Shadowing (1 week)

### Runbook Certification
To be certified for on-call:
1. Read all four runbooks
2. Complete runbook quiz (80% pass rate)
3. Shadow on-call engineer for 1 week
4. Successfully handle 3 simulated incidents
5. Review by On-Call Lead

### Simulated Incident Drills
- Monthly gameday exercises
- Practice P0/P1 incident response
- Test rollback procedures
- Validate escalation paths

---

## Changelog

### Version 1.0 (2025-11-17)
- Initial creation of all four runbooks
- TROUBLESHOOTING.md: Common issues and solutions
- INCIDENT_RESPONSE.md: Emergency procedures
- ROLLBACK_PROCEDURE.md: Version rollback guide
- SCALING_GUIDE.md: Scaling operations

### Future Improvements
- [ ] Add screenshots and diagrams
- [ ] Create video walkthroughs
- [ ] Add more integration-specific troubleshooting
- [ ] Expand multi-region deployment procedures
- [ ] Add chaos engineering scenarios

---

## Support

**Questions?** Ask in Slack:
- #gl-002-team (general questions)
- #gl-002-ops (operations questions)
- #gl-002-incidents (active incidents only)

**Urgent?** Page on-call:
- PagerDuty: Service "GL-002 Production"
- Phone: See PagerDuty escalation policy

**Documentation Issues?**
- Create JIRA ticket: Project "GL-002", Type "Documentation"
- Tag: runbook-improvement
- Assign to: TechWriter team
