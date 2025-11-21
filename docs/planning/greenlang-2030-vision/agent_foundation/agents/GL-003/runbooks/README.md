# GL-003 SteamSystemAnalyzer - Operations Runbooks

Comprehensive operational runbooks for GL-003 SteamSystemAnalyzer production operations.

## Overview

This directory contains production operations runbooks covering troubleshooting, incident response, rollback procedures, and scaling operations for the GL-003 SteamSystemAnalyzer agent.

## Runbooks Index

### 1. TROUBLESHOOTING.md

**Purpose**: Diagnose and resolve common production issues

**When to Use**:
- Agent not starting or pods crashing
- High error rates or degraded performance
- Integration failures with SCADA/Steam Meter systems
- Database or cache connectivity issues
- Steam analysis calculation errors
- Leak detection failures
- Steam trap monitoring issues

**Common Scenarios**:
- Configuration errors preventing startup
- Missing environment variables or secrets
- Database connection pool exhaustion
- Redis cache failures
- OOMKilled or CPU throttling
- Integration timeouts
- Inaccurate leak detection
- Steam trap misclassification

**Quick Links**:
- [Agent Not Starting](TROUBLESHOOTING.md#agent-not-starting)
- [High Error Rates](TROUBLESHOOTING.md#high-error-rates)
- [Performance Degradation](TROUBLESHOOTING.md#performance-degradation)
- [Steam Analysis Issues](TROUBLESHOOTING.md#steam-analysis-issues)
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
- High steam meter count (>500 meters)
- High-frequency monitoring (>10Hz)
- Large facilities (>50 zones)
- Planned load increases (new facilities, seasonal demand)
- Performance optimization

**Scaling Types**:
- **Horizontal Scaling**: Add/remove pods (HPA)
- **Vertical Scaling**: Increase pod resources
- **Database Scaling**: Connection pools, read replicas, TimescaleDB optimization
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
kubectl get pods -n greenlang | grep gl-003

# Health endpoint
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/health

# Error rate
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate

# Resource usage
kubectl top pods -n greenlang | grep gl-003
```

#### Emergency Rollback
```bash
# Rollback to previous version (5 minutes)
kubectl rollout undo deployment/gl-003-steam-system -n greenlang
kubectl rollout status deployment/gl-003-steam-system -n greenlang --timeout=5m
```

#### Scale Pods
```bash
# Manual scale up
kubectl scale deployment gl-003-steam-system --replicas=5 -n greenlang

# Check HPA status
kubectl get hpa gl-003-steam-system-hpa -n greenlang
```

#### View Logs
```bash
# Recent logs
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=100

# Follow logs
kubectl logs -f -n greenlang deployment/gl-003-steam-system

# Previous pod logs (if crashed)
kubectl logs -n greenlang <pod-name> --previous
```

#### Restart Deployment
```bash
# Graceful restart
kubectl rollout restart deployment/gl-003-steam-system -n greenlang

# Monitor restart
kubectl rollout status deployment/gl-003-steam-system -n greenlang
```

#### Check Steam System Metrics
```bash
# Steam meter connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/health

# Leak detection status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/status

# Steam trap performance
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/traps/status
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

### Steam Analysis Issues Decision Tree

```
What type of steam analysis issue?
├─ Inaccurate leak detection → TROUBLESHOOTING.md (Leak Detection Issues)
├─ Steam trap misclassification → TROUBLESHOOTING.md (Steam Trap Issues)
├─ Pressure drop calculation errors → TROUBLESHOOTING.md (Calculation Issues)
├─ Condensate optimization failures → TROUBLESHOOTING.md (Condensate Issues)
├─ SCADA/Meter connection failures → TROUBLESHOOTING.md (Integration Failures)
└─ Database query timeouts → TROUBLESHOOTING.md (Database Issues) + SCALING_GUIDE.md
```

### Scaling Decision Tree

```
What type of issue?
├─ High CPU/Memory usage → SCALING_GUIDE.md (Vertical or Horizontal Scaling)
├─ Slow response times → TROUBLESHOOTING.md first, then SCALING_GUIDE.md
├─ Database slow (TimescaleDB) → SCALING_GUIDE.md (Database Scaling)
├─ Cache issues → SCALING_GUIDE.md (Cache Scaling)
├─ High steam meter count (>500) → SCALING_GUIDE.md (High-Volume Scaling)
├─ High-frequency monitoring (>10Hz) → SCALING_GUIDE.md (Real-Time Scaling)
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
   kubectl rollout history deployment/gl-003-steam-system -n greenlang
   ```
   - If deployed <2 hours ago → Consider rollback (ROLLBACK_PROCEDURE.md)

4. **Assess impact**:
   ```bash
   kubectl get pods -n greenlang | grep gl-003
   kubectl exec -n greenlang deployment/gl-003-steam-system -- \
     curl -s http://localhost:8000/api/v1/metrics | grep error_rate
   ```

5. **Check steam system critical functions**:
   ```bash
   # Leak detection working?
   kubectl exec -n greenlang deployment/gl-003-steam-system -- \
     curl -s http://localhost:8000/api/v1/analysis/leaks/status

   # Steam meter connectivity?
   kubectl exec -n greenlang deployment/gl-003-steam-system -- \
     curl -s http://localhost:8000/api/v1/integrations/meters/health

   # Steam trap monitoring active?
   kubectl exec -n greenlang deployment/gl-003-steam-system -- \
     curl -s http://localhost:8000/api/v1/analysis/traps/status
   ```

6. **Follow appropriate runbook**:
   - P0/P1 → INCIDENT_RESPONSE.md
   - Need rollback → ROLLBACK_PROCEDURE.md
   - Troubleshooting → TROUBLESHOOTING.md
   - Scaling → SCALING_GUIDE.md

7. **Communicate** (see INCIDENT_RESPONSE.md for templates):
   - Post in #gl-003-incidents Slack channel
   - Update status page (for P0/P1)

---

## Monitoring and Alerting

### Dashboards

- **Grafana Main Dashboard**: https://grafana.greenlang.io/d/gl-003/steam-system-analyzer
- **Leak Detection Dashboard**: https://grafana.greenlang.io/d/gl-003-leaks/leak-analysis
- **Steam Trap Dashboard**: https://grafana.greenlang.io/d/gl-003-traps/trap-performance
- **Performance Metrics**: https://grafana.greenlang.io/d/gl-003-perf/performance-metrics
- **Scaling Metrics**: https://grafana.greenlang.io/d/gl-003-scaling/scaling-operations

### Key Metrics

```bash
# Check all key metrics at once
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(error_rate|duration|cpu|memory|db_pool|cache_hit|leak_detection|trap_efficiency)'
```

**Target Metrics**:
- Error rate: <1%
- Response time (p95): <2 seconds
- CPU usage: <70%
- Memory usage: <80%
- Database pool utilization: <80%
- Cache hit rate: >85%
- Leak detection accuracy: >95%
- Steam trap classification accuracy: >98%
- Steam meter connectivity: >99%

### Alerting Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Error rate | >5% | >20% | TROUBLESHOOTING.md |
| Response time (p95) | >3s | >5s | SCALING_GUIDE.md |
| CPU usage | >70% | >85% | SCALING_GUIDE.md |
| Memory usage | >80% | >90% | SCALING_GUIDE.md |
| Pod count | 0/3 | 0/3 | INCIDENT_RESPONSE.md (P0) |
| Database connections | >80% | >95% | SCALING_GUIDE.md |
| Steam meter connectivity | <95% | <90% | TROUBLESHOOTING.md |
| Leak detection failures | >5% | >10% | TROUBLESHOOTING.md |
| Steam trap misclassification | >5% | >10% | TROUBLESHOOTING.md |

### Steam System Specific Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| Critical steam leak detected | Any P0 leak | INCIDENT_RESPONSE.md (P0) |
| Steam trap mass failure | >20% traps failed | INCIDENT_RESPONSE.md (P1) |
| Distribution efficiency drop | <80% efficiency | TROUBLESHOOTING.md |
| Pressure anomaly | Deviation >10% | TROUBLESHOOTING.md |
| SCADA connection loss | >5 min offline | TROUBLESHOOTING.md |
| Condensate return rate drop | <70% return | TROUBLESHOOTING.md |

---

## Escalation Contacts

### Primary On-Call
- **PagerDuty**: Service ID "GL-003 Primary"
- **Slack**: @gl-003-oncall-primary
- **Response Time**: 5 minutes

### Secondary On-Call
- **PagerDuty**: Service ID "GL-003 Secondary"
- **Slack**: @gl-003-oncall-secondary
- **Response Time**: 10 minutes (if primary doesn't respond)

### Engineering Manager
- **Escalation**: For P0 incidents lasting >1 hour or P1 incidents lasting >4 hours
- **Contact**: See INCIDENT_RESPONSE.md for details

### Director of Engineering
- **Escalation**: For unresolved P0 incidents
- **Contact**: See INCIDENT_RESPONSE.md for details

### Steam System Subject Matter Expert (SME)
- **Contact**: For complex steam system calculation issues
- **Availability**: Business hours (8AM-6PM local time)
- **Slack**: @steam-sme

---

## Runbook Maintenance

### Update Frequency
- Review quarterly or after significant incidents
- Update after architecture changes
- Add new scenarios as discovered
- Update steam system thresholds based on operational data

### Contributors
- DevOps Team: Infrastructure procedures
- Platform Engineering: Scaling and performance
- Development Team: Application-specific troubleshooting
- SRE Team: Monitoring and alerting
- Steam System Engineers: Domain-specific procedures

### Feedback
- Found an issue? Create JIRA ticket with label "runbook-improvement"
- Have a suggestion? Post in #gl-003-team Slack channel
- Need clarification? Ask in #gl-003-ops

---

## Related Documentation

### Internal Documentation
- **Architecture**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\ARCHITECTURE.md`
- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\deployment\README.md`
- **API Documentation**: https://docs.greenlang.io/agents/gl-003/api
- **Integration Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\INTEGRATION_MODULES_DELIVERY.md`
- **Security Audit**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\SECURITY_AUDIT_REPORT.md`

### External Resources
- **Kubernetes Documentation**: https://kubernetes.io/docs/
- **PostgreSQL/TimescaleDB Tuning**: https://docs.timescale.com/timescaledb/latest/how-to-guides/optimize-performance/
- **Redis Optimization**: https://redis.io/docs/management/optimization/
- **Prometheus Queries**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **ASME Steam Tables**: https://www.asme.org/codes-standards/steam-tables
- **ISO 9001 Steam System Standards**: https://www.iso.org/standard/62085.html

### Steam System Resources
- **ASME PTC 19.5**: Application Guide for Flow Measurement
- **ASME PTC 6**: Steam Turbine Performance Test Codes
- **ASHRAE Handbook**: HVAC Systems and Equipment
- **Spirax Sarco Steam Engineering Tutorials**: https://www.spiraxsarco.com/learn-about-steam
- **Armstrong International Technical Library**: https://www.armstronginternational.com/resources/technical-library

---

## Training and Certification

### Required Training
- Kubernetes Operations Basics (2 hours)
- GL-003 Architecture Overview (2 hours)
- Steam System Fundamentals (4 hours)
- Incident Response Protocol (1 hour)
- On-Call Shadowing (1 week)

### Runbook Certification
To be certified for on-call:
1. Read all four runbooks
2. Complete runbook quiz (80% pass rate)
3. Shadow on-call engineer for 1 week
4. Successfully handle 3 simulated incidents (including steam-specific scenarios)
5. Review by On-Call Lead

### Simulated Incident Drills
- Monthly gameday exercises
- Practice P0/P1 incident response
- Test rollback procedures
- Validate escalation paths
- Steam system emergency scenarios:
  - Critical leak detection
  - Steam trap mass failure
  - SCADA connection loss
  - Distribution efficiency collapse

---

## Changelog

### Version 1.0 (2025-11-17)
- Initial creation of all five runbooks
- TROUBLESHOOTING.md: Common issues and solutions
- INCIDENT_RESPONSE.md: Emergency procedures
- ROLLBACK_PROCEDURE.md: Version rollback guide
- SCALING_GUIDE.md: Scaling operations
- README.md: Runbook index and quick reference

### Future Improvements
- [ ] Add screenshots and diagrams
- [ ] Create video walkthroughs
- [ ] Add more steam-specific troubleshooting scenarios
- [ ] Expand multi-region deployment procedures
- [ ] Add chaos engineering scenarios
- [ ] Add TimescaleDB-specific optimization procedures
- [ ] Add acoustic leak detection troubleshooting
- [ ] Add condensate recovery system procedures

---

## Support

**Questions?** Ask in Slack:
- #gl-003-team (general questions)
- #gl-003-ops (operations questions)
- #gl-003-incidents (active incidents only)
- #steam-engineering (steam system domain questions)

**Urgent?** Page on-call:
- PagerDuty: Service "GL-003 Production"
- Phone: See PagerDuty escalation policy

**Documentation Issues?**
- Create JIRA ticket: Project "GL-003", Type "Documentation"
- Tag: runbook-improvement
- Assign to: TechWriter team

---

## Emergency Contacts

### Critical Steam System Emergencies
- **Plant Safety Officer**: +1-555-SAFETY (24/7)
- **Facility Manager**: +1-555-PLANT (24/7)
- **Steam System Vendor Support**: +1-555-STEAM (Business hours)

### External Vendor Support
- **SCADA Vendor**: +1-555-SCADA (24/7 for P0)
- **Steam Meter Vendor**: +1-555-METERS (Business hours)
- **Acoustic Leak Detection Vendor**: +1-555-ACOUSTIC (Business hours)
- **TimescaleDB Support**: https://www.timescale.com/support (Enterprise customers)

**Note**: For life safety issues related to steam systems, immediately contact:
1. Plant Safety Officer
2. Emergency Services (911/local equivalent)
3. Then follow incident response procedures
