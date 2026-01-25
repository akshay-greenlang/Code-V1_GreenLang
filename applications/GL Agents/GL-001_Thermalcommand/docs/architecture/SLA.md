# GL-001 ThermalCommand - Service Level Agreement (SLA)

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-001 |
| Agent Name | ThermalCommand |
| Version | 1.0.0 |
| Effective Date | 2025-01-01 |
| Review Date | 2026-01-01 |
| Classification | Internal - Operations |

---

## 1. Service Overview

GL-001 ThermalCommand is the master orchestrator for all process heat operations, providing:
- Plant-wide heat network allocation
- MILP load dispatch optimization
- Cascade PID control
- SIS safety integration
- Real-time recommendations with explainability

**Business Impact**: $20B value at stake across deployed installations.

---

## 2. Availability Targets

### 2.1 Service Tiers

| Tier | Availability | Monthly Downtime | Use Case |
|------|--------------|------------------|----------|
| **Platinum** | 99.99% | 4.38 minutes | Critical production systems |
| **Gold** | 99.95% | 21.9 minutes | Standard production |
| **Silver** | 99.9% | 43.8 minutes | Non-critical, development |

### 2.2 Default Availability Target

**Target: 99.99% (Platinum Tier)**

```
Annual Allowed Downtime:  52.6 minutes
Monthly Allowed Downtime: 4.38 minutes
Weekly Allowed Downtime:  1.01 minutes
```

### 2.3 Availability Calculation

```
Availability % = ((Total Minutes - Downtime Minutes) / Total Minutes) * 100

Where:
- Total Minutes = Minutes in measurement period
- Downtime Minutes = Time when service is unavailable or degraded
```

### 2.4 Exclusions from Downtime

The following are NOT counted as downtime:
- Scheduled maintenance (announced 72 hours in advance)
- Force majeure events (natural disasters, war, pandemic)
- Customer-caused issues (misconfiguration, network issues)
- Third-party service outages beyond GreenLang control
- Planned DR testing (announced 1 week in advance)

---

## 3. Performance SLOs (Service Level Objectives)

### 3.1 Response Time SLOs

| Endpoint/Operation | P50 | P95 | P99 | Max |
|--------------------|-----|-----|-----|-----|
| Health Check (`/health`) | 5ms | 20ms | 50ms | 100ms |
| Get Heat Plan | 50ms | 200ms | 500ms | 1s |
| Create Heat Plan | 100ms | 500ms | 1s | 2s |
| MILP Optimization | 500ms | 2s | 5s | 30s |
| Safety Validation | 10ms | 50ms | 100ms | 200ms |
| Kafka Message Processing | 20ms | 100ms | 200ms | 500ms |
| GraphQL Query | 100ms | 300ms | 500ms | 1s |
| gRPC Call | 50ms | 150ms | 300ms | 500ms |

### 3.2 Throughput SLOs

| Metric | Target | Minimum | Burst |
|--------|--------|---------|-------|
| API Requests/second | 1,000 | 500 | 2,000 |
| Kafka Messages/second | 10,000 | 5,000 | 20,000 |
| Optimization Runs/minute | 60 | 30 | 120 |
| Concurrent Connections | 5,000 | 2,500 | 10,000 |

### 3.3 Error Rate SLOs

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| HTTP 5xx Rate | < 0.1% | 0.5% | 1% |
| HTTP 4xx Rate | < 1% | 2% | 5% |
| Kafka Consumer Failures | < 0.01% | 0.05% | 0.1% |
| Database Connection Errors | < 0.01% | 0.05% | 0.1% |

### 3.4 Data Quality SLOs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Calculation Accuracy | 100% | Bit-perfect reproducibility |
| Audit Log Completeness | 100% | All actions logged |
| Data Freshness | < 10 seconds | Time since last sensor update |
| Provenance Hash Integrity | 100% | SHA-256 verification |

---

## 4. Recovery Objectives

### 4.1 Recovery Time Objective (RTO)

| Scenario | RTO Target | RTO Maximum |
|----------|------------|-------------|
| Pod Restart | 30 seconds | 2 minutes |
| Node Failure | 2 minutes | 5 minutes |
| Database Failover | 30 seconds | 2 minutes |
| AZ Failure | 3 minutes | 10 minutes |
| Regional Failover | 5 minutes | 30 minutes |
| Full DR Recovery | 30 minutes | 4 hours |

### 4.2 Recovery Point Objective (RPO)

| Data Type | RPO Target | RPO Maximum |
|-----------|------------|-------------|
| Transaction Data | 1 second | 1 minute |
| Optimization Results | 1 minute | 5 minutes |
| Configuration | 0 (GitOps) | 0 |
| Audit Logs | 0 (synchronous) | 0 |
| Cache Data | 5 minutes | 15 minutes |
| Analytics Data | 15 minutes | 1 hour |

---

## 5. Incident Response Times

### 5.1 Severity Definitions

| Severity | Definition | Example |
|----------|------------|---------|
| **P1 - Critical** | Complete service outage or data loss risk | All pods down, database corruption |
| **P2 - High** | Major functionality impaired | Optimization unavailable, high error rate |
| **P3 - Medium** | Minor functionality impaired | Slow responses, partial degradation |
| **P4 - Low** | Minimal impact, cosmetic issues | UI glitch, documentation error |

### 5.2 Response Time Commitments

| Severity | Acknowledgment | Initial Response | Status Update | Resolution Target |
|----------|----------------|------------------|---------------|-------------------|
| P1 | 5 minutes | 15 minutes | Every 15 min | 1 hour |
| P2 | 15 minutes | 30 minutes | Every 30 min | 4 hours |
| P3 | 1 hour | 4 hours | Every 4 hours | 24 hours |
| P4 | 4 hours | 24 hours | Every 24 hours | 5 business days |

### 5.3 Escalation Path

```
P1 Timeline:
0-5 min:   On-call SRE acknowledges
5-15 min:  SRE Lead engaged
15-30 min: Director notified
30-60 min: VP Engineering notified
60+ min:   Executive team notified
```

---

## 6. Maintenance Windows

### 6.1 Scheduled Maintenance

| Type | Window | Frequency | Notification |
|------|--------|-----------|--------------|
| Security Patches | Sunday 02:00-04:00 UTC | Weekly | 72 hours |
| Minor Updates | Sunday 02:00-06:00 UTC | Bi-weekly | 72 hours |
| Major Updates | Saturday 22:00-Sunday 06:00 UTC | Monthly | 1 week |
| Infrastructure | Saturday 22:00-Sunday 10:00 UTC | Quarterly | 2 weeks |

### 6.2 Emergency Maintenance

Emergency maintenance may be performed without advance notice for:
- Critical security vulnerabilities (CVE score >= 9.0)
- Active exploitation or breach
- Imminent data loss risk
- Regulatory compliance requirements

---

## 7. Monitoring and Reporting

### 7.1 Real-Time Monitoring

| Metric | Dashboard | Alert Threshold |
|--------|-----------|-----------------|
| Availability | Grafana: GL-001-Overview | < 99.99% (5 min) |
| Latency P99 | Grafana: GL-001-Latency | > 1s |
| Error Rate | Grafana: GL-001-Errors | > 0.5% |
| Pod Count | Grafana: GL-001-Capacity | < 2 |
| CPU Usage | Grafana: GL-001-Resources | > 80% |
| Memory Usage | Grafana: GL-001-Resources | > 85% |
| Disk Usage | Grafana: GL-001-Resources | > 75% |

### 7.2 Reporting Schedule

| Report | Frequency | Recipients | Content |
|--------|-----------|------------|---------|
| Availability Report | Daily | SRE Team | Uptime, incidents |
| SLO Dashboard | Real-time | All stakeholders | All SLOs |
| Performance Report | Weekly | Engineering | Latency, throughput |
| Incident Summary | Weekly | Management | Incidents, RCA |
| SLA Compliance | Monthly | Executives | Full SLA metrics |
| Trend Analysis | Quarterly | All | Capacity planning |

### 7.3 SLA Breach Notification

When SLA is breached:
1. Immediate PagerDuty alert to on-call
2. Slack notification to #gl-001-ops
3. Email to stakeholder list within 1 hour
4. Root Cause Analysis within 48 hours
5. Preventive action plan within 1 week

---

## 8. Capacity Commitments

### 8.1 Baseline Capacity

| Resource | Guaranteed | Burst | Notes |
|----------|------------|-------|-------|
| CPU | 6 cores | 12 cores | HPA enabled |
| Memory | 12 GB | 24 GB | Per region |
| Storage | 500 GB | 2 TB | Auto-scaling |
| Network | 1 Gbps | 10 Gbps | Dedicated |

### 8.2 Scaling Commitments

| Trigger | Action | Time |
|---------|--------|------|
| CPU > 70% | Scale up pods | 30 seconds |
| Memory > 80% | Scale up pods | 30 seconds |
| Request queue > 100 | Scale up pods | 30 seconds |
| Kafka lag > 10,000 | Scale consumers | 1 minute |

---

## 9. Support Channels

### 9.1 Contact Methods

| Channel | Purpose | Response Time |
|---------|---------|---------------|
| PagerDuty | P1/P2 incidents | 5/15 minutes |
| Slack #gl-001-support | Questions, P3/P4 | 1 hour |
| Email: gl-001@greenlang.ai | Non-urgent | 4 hours |
| JIRA | Feature requests | 24 hours |
| Documentation | Self-service | N/A |

### 9.2 Support Hours

| Tier | Hours | Coverage |
|------|-------|----------|
| P1/P2 Incidents | 24/7/365 | Global on-call |
| P3 Incidents | Business hours | Regional teams |
| General Support | Business hours | Regional teams |

---

## 10. Compliance and Audit

### 10.1 Compliance Standards

| Standard | Scope | Certification |
|----------|-------|---------------|
| ISO 27001 | Security | Annual audit |
| SOC 2 Type II | Operations | Annual audit |
| IEC 61511 | Safety | SIL 3 certified |
| ISO 50001 | Energy | Annual audit |
| NFPA 86 | Furnace safety | Compliance verified |

### 10.2 Audit Trail

All operations are logged with:
- Timestamp (UTC)
- User/service identity
- Action performed
- Before/after state
- SHA-256 hash for integrity
- 7-year retention

---

## 11. SLA Review and Amendment

### 11.1 Review Schedule

- **Quarterly**: Performance review, minor adjustments
- **Annually**: Full SLA review, major changes
- **On-demand**: Significant architecture changes

### 11.2 Amendment Process

1. Proposed changes documented
2. Stakeholder review (2 weeks)
3. Impact assessment
4. Approval by SLA committee
5. 30-day notice before implementation

---

## 12. Appendices

### 12.1 Glossary

| Term | Definition |
|------|------------|
| Availability | Percentage of time service is operational |
| Downtime | Period when service is unavailable |
| Latency | Time to process a request |
| RTO | Recovery Time Objective |
| RPO | Recovery Point Objective |
| SLO | Service Level Objective |
| SLA | Service Level Agreement |

### 12.2 Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL DevOps | Initial release |

---

## Signatures

| Role | Name | Date |
|------|------|------|
| Engineering Lead | _________________ | ____________ |
| Operations Lead | _________________ | ____________ |
| Product Owner | _________________ | ____________ |
