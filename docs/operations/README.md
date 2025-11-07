# GreenLang Operations Documentation

**Version:** 1.0
**Last Updated:** 2025-11-07
**Document Classification:** Operations Reference

---

## Overview

This directory contains comprehensive operational documentation for the GreenLang platform. These runbooks, playbooks, and guides are designed to enable operations teams to deploy, maintain, troubleshoot, and optimize GreenLang in production environments.

**Target Audience:**
- Operations Engineers
- Site Reliability Engineers (SRE)
- DevOps Engineers
- System Administrators
- Database Administrators
- On-Call Engineers

**Documentation Philosophy:**
- **Actionable:** Step-by-step procedures with clear commands
- **Comprehensive:** Covers normal operations and emergency scenarios
- **Tested:** All procedures validated in production-like environments
- **Maintained:** Regularly updated based on operational learnings
- **Accessible:** Available offline and online

---

## Table of Contents

### Core Operational Runbooks

1. [Disaster Recovery Runbook](disaster-recovery-runbook.md)
2. [Incident Response Playbook](incident-response-playbook.md)
3. [Production Deployment Checklist](production-deployment-checklist.md)
4. [Troubleshooting Guide](troubleshooting-guide.md)
5. [Performance Tuning Guide](performance-tuning-guide.md)
6. [Backup and Restore Procedures](backup-and-restore.md)
7. [Monitoring Setup Guide](monitoring-setup-guide.md)

---

## Quick Start for New Operations Team Members

### Day 1: Orientation

**Required Reading (2-3 hours):**
1. This README (overview)
2. [Production Deployment Checklist](production-deployment-checklist.md) - Sections 1-5
3. [Incident Response Playbook](incident-response-playbook.md) - Executive Summary and Severity Levels
4. [Troubleshooting Guide](troubleshooting-guide.md) - First 5 issues

**Required Setup (1-2 hours):**
- [ ] Access to production Kubernetes cluster
- [ ] Access to monitoring dashboards (Grafana, Prometheus)
- [ ] Access to log aggregation (Loki)
- [ ] PagerDuty account configured
- [ ] Slack channels joined (#ops, #incidents, #alerts)
- [ ] VPN access configured
- [ ] SSH keys for production servers
- [ ] Database read-only access

**Validation:**
```bash
# Verify access
kubectl get nodes
curl https://grafana.greenlang.io/api/health
psql -h db.greenlang.io -U readonly -d greenlang -c "SELECT 1;"
```

---

### Week 1: Core Operations

**Learning Path:**
1. **Day 2-3:** Complete reading of all operational runbooks
2. **Day 4:** Shadow on-call engineer for 8 hours
3. **Day 5:** Participate in deployment (read-only observer)

**Practice Exercises:**
1. Deploy to staging environment using [Production Deployment Checklist](production-deployment-checklist.md)
2. Simulate and resolve a P2 incident using [Incident Response Playbook](incident-response-playbook.md)
3. Perform a database restore test using [Backup and Restore Procedures](backup-and-restore.md)
4. Configure a new Grafana dashboard using [Monitoring Setup Guide](monitoring-setup-guide.md)

**Checkpoint:** By end of Week 1, you should be able to:
- [ ] Navigate all monitoring dashboards
- [ ] Identify and triage common issues
- [ ] Execute basic troubleshooting procedures
- [ ] Understand deployment process
- [ ] Know when and how to escalate

---

### Month 1: Operational Proficiency

**Milestones:**
- [ ] Complete on-call training
- [ ] Perform first production deployment (with supervision)
- [ ] Respond to first incident (with supervision)
- [ ] Complete disaster recovery tabletop exercise
- [ ] Contribute improvements to operational documentation

**Advanced Topics:**
- Performance optimization strategies
- Capacity planning and forecasting
- Advanced troubleshooting techniques
- Disaster recovery testing
- Security incident response

---

## Document Summaries

### 1. Disaster Recovery Runbook

**Purpose:** Comprehensive procedures for recovering from catastrophic failures

**When to Use:**
- Complete data center failure
- Database corruption or loss
- Major security breach
- Extended service outage

**Key Procedures:**
- Data center failover (RTO: 4 hours)
- Database restoration (RPO: 1 hour)
- Point-in-time recovery
- DR testing procedures

**Critical Information:**
- Recovery objectives (RTO/RPO)
- Failover procedures
- Backup restoration steps
- Validation checklists

**Length:** ~400 lines
**Read Time:** 30-45 minutes

---

### 2. Incident Response Playbook

**Purpose:** Standardized procedures for responding to production incidents

**When to Use:**
- Service degradation or outage
- High error rates
- Performance issues
- Security incidents
- Any production anomaly

**Key Procedures:**
- Incident severity assessment (P0-P3)
- Response procedures by severity
- Common incident scenarios and fixes
- Escalation paths
- Post-incident review process

**Critical Information:**
- Severity definitions
- Response time targets
- Communication templates
- On-call procedures

**Length:** ~600 lines
**Read Time:** 45-60 minutes

---

### 3. Production Deployment Checklist

**Purpose:** Comprehensive checklist ensuring safe production deployments

**When to Use:**
- Every production deployment
- Pre-deployment validation
- Post-deployment verification
- Rollback scenarios

**Key Procedures:**
- Pre-deployment checklist (100+ items)
- Blue-green deployment steps
- Rolling update procedures
- Rollback procedures
- Validation tests

**Critical Information:**
- Deployment methods comparison
- Validation gates
- Rollback triggers
- Success criteria

**Length:** ~700 lines
**Read Time:** 60-90 minutes

---

### 4. Troubleshooting Guide

**Purpose:** Solutions for common operational issues

**When to Use:**
- Performance problems
- Application errors
- Infrastructure issues
- Database problems
- Network connectivity issues

**Key Procedures:**
- Diagnostic steps for common issues
- Resolution procedures
- Debugging tools and commands
- Log analysis techniques
- Metric interpretation

**Critical Information:**
- Common issue patterns
- Quick fixes
- Diagnostic commands
- When to escalate

**Length:** ~500 lines
**Read Time:** 45-60 minutes

---

### 5. Performance Tuning Guide

**Purpose:** Optimization procedures for production performance

**When to Use:**
- Performance below targets
- Capacity planning
- Resource optimization
- Cost reduction initiatives
- Scaling preparations

**Key Procedures:**
- CPU/memory/disk optimization
- Application tuning (concurrency, caching)
- Database optimization
- Infrastructure scaling
- Capacity planning

**Critical Information:**
- Performance targets
- Optimization strategies
- Monitoring metrics
- Benchmarking procedures

**Length:** ~500 lines
**Read Time:** 45-60 minutes

---

### 6. Backup and Restore Procedures

**Purpose:** Data protection and recovery procedures

**When to Use:**
- Regular backup operations
- Data recovery needs
- Disaster recovery scenarios
- Backup testing
- Data migration

**Key Procedures:**
- Database backup (full, incremental, WAL)
- Configuration backup
- Backup validation
- Restore procedures
- Point-in-time recovery

**Critical Information:**
- Backup schedules
- Retention policies
- Recovery objectives (RTO/RPO)
- Testing procedures

**Length:** ~450 lines
**Read Time:** 45 minutes

---

### 7. Monitoring Setup Guide

**Purpose:** Complete monitoring stack configuration

**When to Use:**
- Initial monitoring setup
- Adding new metrics
- Configuring dashboards
- Setting up alerts
- Troubleshooting monitoring

**Key Procedures:**
- Prometheus setup
- Grafana configuration
- Loki log aggregation
- Jaeger tracing
- Alert configuration

**Critical Information:**
- Monitoring architecture
- Installation steps
- Dashboard templates
- Alert rules

**Length:** ~500 lines
**Read Time:** 60 minutes

---

## Operational Workflows

### Daily Operations

**Morning Checklist (15 minutes):**
```bash
# 1. Check system health
kubectl get pods -A | grep -v Running
kubectl top nodes

# 2. Review overnight alerts
# Visit: https://grafana.greenlang.io/alerting

# 3. Check key metrics
# Dashboard: https://grafana.greenlang.io/d/system-overview

# 4. Review error logs
kubectl logs -l app=greenlang-api --since=12h | grep ERROR | wc -l

# 5. Verify backups completed
aws s3 ls s3://greenlang-backups/database/ | tail -5
```

**End of Day Checklist (10 minutes):**
- [ ] Review day's incidents and actions taken
- [ ] Update on-call handoff notes (if applicable)
- [ ] Check scheduled maintenance for tonight
- [ ] Verify monitoring and alerting functional
- [ ] Document any operational issues discovered

---

### Weekly Operations

**Weekly Review (1 hour):**
- [ ] Review week's incidents and trends
- [ ] Check backup restore test results
- [ ] Review capacity metrics and forecasts
- [ ] Update operational documentation
- [ ] Review and update alert thresholds
- [ ] Check SSL certificate expiration dates
- [ ] Review cost reports

**Weekly Maintenance:**
- [ ] Database maintenance (VACUUM, ANALYZE)
- [ ] Log rotation and cleanup
- [ ] Kubernetes node health check
- [ ] Security updates review
- [ ] Performance trend analysis

---

### Monthly Operations

**Monthly Tasks (4 hours):**
- [ ] Full disaster recovery test
- [ ] Capacity planning review
- [ ] Performance benchmarking
- [ ] Security audit
- [ ] Cost optimization review
- [ ] Documentation review and updates
- [ ] Operational metrics review
- [ ] Team training/knowledge sharing session

**Monthly Reports:**
- Service availability report
- Incident summary and trends
- Performance metrics
- Cost analysis
- Capacity forecast
- Security posture

---

## Key Metrics and SLOs

### Service Level Objectives (SLOs)

| Metric | SLO | Measurement |
|--------|-----|-------------|
| **Availability** | 99.9% | Monthly uptime |
| **Latency (p95)** | < 500ms | API response time |
| **Error Rate** | < 1% | Failed requests / total |
| **Time to Recovery** | < 4 hours | MTTR for incidents |
| **Backup Success** | 100% | Daily backup completion |

### Monitoring Dashboards

**Essential Dashboards:**
1. **System Overview:** https://grafana.greenlang.io/d/system-overview
   - Request rate, error rate, latency
   - CPU, memory, disk usage
   - Active incidents

2. **API Performance:** https://grafana.greenlang.io/d/api-performance
   - Per-endpoint latency
   - Per-endpoint error rates
   - Request distribution

3. **Database Performance:** https://grafana.greenlang.io/d/database
   - Connection pool usage
   - Query duration
   - Replication lag

4. **Infrastructure:** https://grafana.greenlang.io/d/infrastructure
   - Node health
   - Pod status
   - Resource utilization

5. **Deployment Dashboard:** https://grafana.greenlang.io/d/deployments
   - Deployment history
   - Success/failure rate
   - Rollback tracking

---

## Escalation Procedures

### Escalation Matrix

| Scenario | Severity | First Contact | Escalate To (15 min) | Escalate To (1 hour) |
|----------|----------|---------------|----------------------|----------------------|
| Service outage | P0 | On-call Engineer | Incident Commander | CTO |
| Severe degradation | P1 | On-call Engineer | Engineering Lead | CTO |
| Moderate degradation | P2 | On-call Engineer | Engineering Lead | - |
| Minor issue | P3 | Create ticket | - | - |

### Contact Information

**Primary Contacts:**
- On-Call Engineer: PagerDuty rotation
- Incident Commander: [Phone]
- Engineering Lead: [Phone]
- Database Administrator: [Phone]
- Security Lead: [Phone]
- CTO: [Phone]

**Communication Channels:**
- Primary: PagerDuty
- War Room: Zoom bridge + Slack #incident-war-room
- Status Updates: #ops channel
- Customer Communication: status.greenlang.io

---

## Tools and Access

### Required Tools

**Command Line:**
- `kubectl` - Kubernetes cluster management
- `psql` - PostgreSQL client
- `redis-cli` - Redis client
- `aws-cli` - AWS management
- `helm` - Kubernetes package manager
- `curl` - API testing
- `jq` - JSON parsing

**Web Interfaces:**
- Grafana: https://grafana.greenlang.io
- Prometheus: https://prometheus.greenlang.io
- Jaeger: https://jaeger.greenlang.io
- Kubernetes Dashboard: https://k8s.greenlang.io
- AWS Console: https://console.aws.amazon.com

**Installation:**
```bash
# Install required tools
brew install kubectl postgresql redis aws-cli helm jq

# Or on Linux:
apt-get install kubectl postgresql-client redis-tools awscli helm jq
```

---

## Security and Compliance

### Access Control

**Production Access:**
- Requires: VPN + MFA + SSH key
- Approved by: Engineering Lead
- Audit logged: All commands logged
- Review period: Quarterly

**Sensitive Operations:**
- Database modifications: Require peer review
- Secret rotation: Scheduled maintenance window
- Configuration changes: Change control process
- Deployment to production: Deployment checklist

### Audit and Compliance

**Audit Logs:**
- All production access logged
- All database changes logged
- All configuration changes tracked in Git
- All deployments tracked

**Regular Audits:**
- Weekly: Access review
- Monthly: Security audit
- Quarterly: Compliance review
- Annually: External audit

---

## Continuous Improvement

### Documentation Updates

**When to Update:**
- After every incident (lessons learned)
- After every deployment issue
- Quarterly documentation review
- When procedures change
- When new tools/systems added

**How to Contribute:**
1. Identify documentation gap or improvement
2. Create issue in docs repository
3. Submit pull request with updates
4. Get review from operations lead
5. Merge and notify team

### Feedback Loop

**Channels for Feedback:**
- Post-incident reviews
- Weekly ops meetings
- Monthly retrospectives
- Quarterly planning
- Ad-hoc suggestions (Slack #ops-docs)

---

## Training and Certification

### Required Training

**Week 1:**
- [ ] GreenLang architecture overview
- [ ] Operations documentation review
- [ ] Monitoring and alerting training
- [ ] Incident response training

**Week 2:**
- [ ] Deployment procedures training
- [ ] Troubleshooting workshop
- [ ] Disaster recovery tabletop exercise
- [ ] On-call readiness assessment

**Month 1:**
- [ ] Performance tuning workshop
- [ ] Database administration basics
- [ ] Security incident response
- [ ] Shadow on-call rotation (1 week)

**Certification:**
- [ ] Complete all training modules
- [ ] Pass operations knowledge assessment
- [ ] Successfully complete on-call shift (supervised)
- [ ] Receive sign-off from operations lead

---

## Additional Resources

### Internal Documentation

- [GreenLang Architecture](../ARCHITECTURE.md)
- [API Reference](../API_REFERENCE_COMPLETE.md)
- [Security Documentation](../security/)
- [Observability Documentation](../observability/)
- [Performance Documentation](../performance/)

### External Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Site Reliability Engineering Book](https://sre.google/sre-book/table-of-contents/)

### Vendor Support

- **AWS Support:** +1-XXX-XXX-XXXX
- **Kubernetes Support:** support@kubernetes.io
- **OpenAI Support:** support@openai.com
- **Anthropic Support:** support@anthropic.com

---

## FAQ

### General Questions

**Q: Who should I contact first during an incident?**
A: Acknowledge the alert in PagerDuty. This pages the on-call engineer. For P0 incidents, immediately escalate to Incident Commander.

**Q: How do I know if I should declare an incident?**
A: Use the severity definitions in the [Incident Response Playbook](incident-response-playbook.md). When in doubt, declare the incident - you can always downgrade severity later.

**Q: What if I'm not sure how to fix an issue?**
A: 1) Check the [Troubleshooting Guide](troubleshooting-guide.md), 2) Search previous incidents in the incident log, 3) Ask in #ops Slack channel, 4) Escalate according to the escalation matrix.

**Q: How often should I test disaster recovery procedures?**
A: Monthly for database restore tests, quarterly for partial failover tests, annually for full disaster recovery tests.

**Q: Can I make changes to production without a deployment?**
A: Minor configuration changes through ConfigMaps are allowed with peer review. Code changes require full deployment process.

---

### Deployment Questions

**Q: How long does a typical production deployment take?**
A: 2-4 hours for blue-green deployment with gradual rollout. This includes all validation gates.

**Q: When should I rollback a deployment?**
A: Immediately if error rate >5%, or if any P0 trigger conditions are met. Within 30 minutes if error rate >2% sustained for 15 minutes.

**Q: Can I deploy outside the normal deployment window?**
A: Emergency hotfixes only, with approval from Engineering Lead or CTO. Use expedited deployment process.

---

### Monitoring Questions

**Q: Which metrics should I watch during deployments?**
A: Error rate, p95 latency, request rate, CPU/memory usage, database connection pool. All should remain stable.

**Q: How do I add a new alert?**
A: Follow the [Monitoring Setup Guide](monitoring-setup-guide.md) alert rules section. All new alerts require testing and documentation.

**Q: Why am I getting false positive alerts?**
A: Review alert threshold in Prometheus rules. Tune threshold based on historical data. Document tuning rationale.

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | Operations Team | Initial comprehensive operations documentation |

**Next Review:** 2025-12-07
**Document Owner:** Operations Lead
**Approvers:** [CTO], [Engineering Lead], [Operations Lead]

---

## Feedback and Improvements

**Found an issue or have a suggestion?**

1. **For urgent fixes:** Post in #ops Slack channel
2. **For improvements:** Create issue in documentation repository
3. **For major changes:** Schedule discussion in ops meeting

**Contact:** operations@greenlang.io

---

**Operational excellence is a journey, not a destination. Let's build reliability together!**
