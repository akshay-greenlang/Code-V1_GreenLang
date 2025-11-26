# GL-008 SteamTrapInspector - Operational Runbooks

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** Platform Operations Team

---

## Overview

This directory contains comprehensive operational runbooks for GL-008 SteamTrapInspector, GreenLang's AI-powered steam trap inspection and predictive maintenance application. These runbooks are designed to enable operations teams to maintain, troubleshoot, scale, and respond to incidents effectively.

---

## Runbook Inventory

### 1. INCIDENT_RESPONSE.md (1,450 lines)

**Purpose:** Incident classification, response procedures, and escalation protocols

**Contents:**
- Incident classification matrix (P0-P4)
- 8 detailed incident scenarios:
  - Mass trap failure detection
  - Sensor communication loss
  - False positive surge
  - Energy calculation anomaly
  - ML model degradation
  - Database connection failure
  - High latency
  - Security breach
- Response procedures with specific timelines
- Escalation matrix and contact information
- Post-incident review process
- Communication templates

**When to Use:**
- System outages or degradation
- Unusual error patterns
- Customer escalations
- Security incidents
- SLA breaches

**Key Features:**
- Step-by-step response procedures
- Specific diagnostic commands
- Automated remediation scripts
- Escalation criteria and paths
- Communication templates for stakeholders

---

### 2. TROUBLESHOOTING.md (1,553 lines)

**Purpose:** Diagnostic procedures and resolution steps for common issues

**Contents:**
- 15+ common issue categories:
  - Inspection job failures
  - High false positive rate
  - Slow inspection times
  - Acoustic sensor failures
  - Thermal camera issues
  - Sensor calibration drift
  - API response latency
  - Memory leaks
  - Inconsistent energy calculations
  - Missing inspection data
  - ERP connector failures
  - MQTT message delivery failures
  - ML model prediction errors
  - RUL prediction drift
  - Slow database queries
- Diagnostic commands and tools
- Step-by-step resolution procedures
- Advanced troubleshooting techniques

**When to Use:**
- Investigating recurring issues
- Performance degradation
- Sensor malfunctions
- Integration problems
- Data quality issues

**Key Features:**
- Comprehensive diagnostic scripts
- Root cause analysis procedures
- Performance profiling tools
- Network diagnostics
- Database query optimization

---

### 3. ROLLBACK_PROCEDURE.md (1,135 lines)

**Purpose:** Safe and tested rollback procedures for all deployment types

**Contents:**
- Rollback decision criteria
- Pre-rollback checklist (critical verification steps)
- 3 rollback scenarios:
  - **Fast Rollback (5 minutes):** Application-only changes
  - **Standard Rollback (15 minutes):** Config + ML model changes
  - **Full Rollback (1 hour):** Database schema changes
- Component-specific rollback procedures:
  - Application deployment rollback
  - Database schema migration rollback
  - ML model version rollback
  - Configuration and feature flag rollback
- Verification procedures
- Post-rollback monitoring
- Communication templates

**When to Use:**
- Deployment causes production issues
- Critical bugs discovered post-deployment
- Performance regression detected
- Data corruption risks

**Key Features:**
- Time-boxed procedures (5min/15min/1hr)
- Automated rollback scripts
- Database PITR (Point-In-Time Recovery)
- Multi-component coordination
- Safety checks and validations

---

### 4. SCALING_GUIDE.md (1,137 lines)

**Purpose:** Horizontal and vertical scaling strategies for growth and performance

**Contents:**
- Scaling overview and capacity baseline
- Horizontal scaling procedures:
  - Manual scaling
  - Gradual scaling strategies
  - Multi-component scaling
- Vertical scaling:
  - Resource right-sizing
  - ML service GPU allocation
  - Performance optimization
- Auto-scaling configuration:
  - Horizontal Pod Autoscaler (HPA)
  - Vertical Pod Autoscaler (VPA)
  - Custom metrics for autoscaling
- Capacity planning:
  - Planning matrix (10 sites → 1,000 sites)
  - Growth forecasting tools
  - Pre-scaling checklist
- Performance benchmarks at scale
- Multi-site deployment architecture:
  - Regional deployments
  - Cross-region failover
  - Global load balancing
- Database scaling strategies
- Cost optimization techniques

**When to Use:**
- Onboarding new customers
- Anticipated traffic spikes
- Performance optimization
- Multi-region expansion
- Cost reduction initiatives

**Key Features:**
- Capacity planning matrix for 10-1000 sites
- Automated scaling scripts
- Performance benchmarking tools
- Load and stress testing procedures
- Cost-effective scaling strategies (spot instances, off-peak scaling)

---

### 5. MAINTENANCE.md (1,585 lines)

**Purpose:** Proactive maintenance schedules and procedures

**Contents:**
- Daily checks (15 minutes, automated):
  - Kubernetes pod status
  - API health endpoints
  - Database connectivity
  - Sensor status
  - Error rate monitoring
  - API latency
  - Backup verification
  - Certificate expiry
  - Security vulnerability scanning
- Weekly maintenance (1 hour):
  - Log rotation and archival
  - Cache clearing
  - Database vacuum and analyze
  - Sensor health validation
  - ML model performance review
  - Data archival
  - Rolling pod restarts
- Monthly tasks (2 hours):
  - Database index optimization
  - Full ML model evaluation
  - Security audit
  - Capacity review
  - Dependency updates
  - Backup verification and test restore
  - Configuration drift detection
  - Performance benchmarking
  - Log analysis
- Quarterly reviews (4 hours):
  - Business review template
  - Disaster recovery testing
  - Comprehensive system audit
- Sensor calibration procedures:
  - Automated scheduling
  - On-site calibration steps
  - Verification procedures
- ML model retraining:
  - Automated monthly retraining
  - Model deployment approval process
  - A/B testing procedures
- Database maintenance:
  - Weekly optimization
  - Performance tuning
  - Index management
- Security updates and patching
- Backup and recovery procedures

**When to Use:**
- Scheduled maintenance windows
- Proactive health monitoring
- Long-term capacity planning
- Preventive maintenance

**Key Features:**
- Fully automated daily health checks
- Comprehensive maintenance scripts
- Sensor calibration workflows
- ML model retraining automation
- Database performance tuning
- Security patch procedures

---

## Quick Reference Guide

### Critical Contact Information

| Role | Primary Contact | Escalation |
|------|-----------------|------------|
| **On-Call Engineer** | PagerDuty rotation | Engineering Manager |
| **Database Admin** | DBA team | DBA Manager |
| **ML Engineer** | ML team | ML Engineering Lead |
| **Security** | Security team | CISO |
| **Product Owner** | Product team | VP Product |

**Emergency Contact:**
- PagerDuty: https://greenlang.pagerduty.com
- Phone: +1-800-GREENLANG
- Slack: #incident-response

### Essential Commands

```bash
# Quick health check
curl https://api.greenlang.io/v1/steam-trap/health | jq '.'

# Check pod status
kubectl get pods -n greenlang-gl008

# View recent errors
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=100 | grep ERROR

# Database connection test
psql $DB_URL -c "SELECT 1;"

# Sensor status
curl https://api.greenlang.io/v1/steam-trap/sensors/status | jq '.'
```

### Essential Dashboards

- **System Overview:** https://grafana.greenlang.io/d/gl008-overview
- **Performance:** https://grafana.greenlang.io/d/gl008-performance
- **ML Metrics:** https://grafana.greenlang.io/d/gl008-ml-metrics
- **Infrastructure:** https://grafana.greenlang.io/d/gl008-infrastructure
- **Status Page:** https://status.greenlang.io

---

## Runbook Usage Guidelines

### 1. Know Your Runbooks

**Daily Operations:**
- MAINTENANCE.md → Daily checks section
- TROUBLESHOOTING.md → Common issues

**Incidents:**
- INCIDENT_RESPONSE.md → Start here for all incidents
- TROUBLESHOOTING.md → Detailed diagnostics
- ROLLBACK_PROCEDURE.md → If deployment caused issue

**Growth & Planning:**
- SCALING_GUIDE.md → Capacity planning and scaling
- MAINTENANCE.md → Quarterly reviews

### 2. Follow Procedures Exactly

- Runbooks are tested and validated
- Skipping steps can cause additional issues
- Document any deviations in incident tickets

### 3. Update Runbooks

- Document new learnings after each incident
- Update procedures when infrastructure changes
- Review quarterly for accuracy

### 4. Escalate When Needed

- Don't hesitate to escalate if:
  - Issue severity is unclear
  - Resolution time exceeds expectations
  - You need additional expertise
  - Customer escalation occurs

---

## Maintenance Schedule

| Frequency | Day/Time | Duration | Runbook Section |
|-----------|----------|----------|-----------------|
| **Daily** | 9 AM UTC (automated) | 15 min | MAINTENANCE.md → Daily Checks |
| **Weekly** | Sunday 2-3 AM UTC | 1 hour | MAINTENANCE.md → Weekly Maintenance |
| **Monthly** | First Sunday 2-4 AM UTC | 2 hours | MAINTENANCE.md → Monthly Tasks |
| **Quarterly** | TBD with customers | 4 hours | MAINTENANCE.md → Quarterly Reviews |

---

## Runbook Metrics

| Runbook | Lines | Size | Procedures | Scripts |
|---------|-------|------|------------|---------|
| INCIDENT_RESPONSE.md | 1,450 | 37 KB | 8 scenarios | 15+ |
| TROUBLESHOOTING.md | 1,553 | 38 KB | 15+ issues | 20+ |
| ROLLBACK_PROCEDURE.md | 1,135 | 34 KB | 3 scenarios | 8 |
| SCALING_GUIDE.md | 1,137 | 31 KB | Multi-strategy | 12+ |
| MAINTENANCE.md | 1,585 | 45 KB | Daily/Weekly/Monthly/Quarterly | 25+ |
| **TOTAL** | **6,860** | **185 KB** | **50+** | **80+** |

---

## Runbook Review Schedule

| Runbook | Last Reviewed | Next Review | Review Frequency |
|---------|---------------|-------------|------------------|
| INCIDENT_RESPONSE.md | 2025-11-26 | 2026-02-26 | Quarterly |
| TROUBLESHOOTING.md | 2025-11-26 | 2026-02-26 | Quarterly |
| ROLLBACK_PROCEDURE.md | 2025-11-26 | 2026-02-26 | Quarterly |
| SCALING_GUIDE.md | 2025-11-26 | 2026-02-26 | Quarterly |
| MAINTENANCE.md | 2025-11-26 | 2026-02-26 | Quarterly |

---

## Training & Onboarding

### New Team Members

**Week 1:**
- Read all runbooks (full understanding)
- Review recent incident reports
- Shadow on-call engineer

**Week 2:**
- Execute MAINTENANCE.md daily checks
- Practice TROUBLESHOOTING.md procedures in staging
- Review SCALING_GUIDE.md capacity planning

**Week 3:**
- Participate in weekly maintenance
- Practice rollback procedures in staging
- Shadow incident response

**Week 4:**
- Join on-call rotation (with backup)
- Lead weekly maintenance (supervised)
- Contribute to runbook updates

### Ongoing Training

**Monthly:**
- Review updated sections of runbooks
- Participate in incident post-mortems
- Practice disaster recovery procedures

**Quarterly:**
- Full runbook refresh training
- DR test participation
- Scaling simulation exercises

---

## Related Documentation

- **Architecture:** `../architecture/SYSTEM_ARCHITECTURE.md`
- **API Documentation:** `../api/API_REFERENCE.md`
- **Deployment Guide:** `../deployment/DEPLOYMENT_GUIDE.md`
- **Monitoring & Alerts:** `../monitoring/MONITORING_GUIDE.md`
- **Security:** `../security/SECURITY_GUIDE.md`

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-11-26 | Initial release - Complete runbook suite | GL-TechWriter |

---

## Feedback & Improvements

**Submit Improvements:**
- Jira: https://greenlang.atlassian.net/projects/GL008-DOCS
- Pull Request: Submit to `docs/runbooks/` directory
- Email: documentation@greenlang.io

**Urgent Updates:**
- Slack: #gl008-runbooks
- On-call: PagerDuty escalation

---

## Legal & Compliance

**Confidentiality:** These runbooks contain proprietary operational procedures. Do not share outside GreenLang organization.

**Compliance:** Runbook procedures comply with:
- SOC 2 Type 2 requirements
- ISO 27001 controls
- GDPR data handling requirements
- Industry best practices for operational excellence

---

**Document Owner:** Platform Operations Team
**Technical Writer:** GL-TechWriter
**Maintained By:** Platform Operations Team
**Review Cycle:** Quarterly

**Last Updated:** 2025-11-26
**Next Review:** 2026-02-26
