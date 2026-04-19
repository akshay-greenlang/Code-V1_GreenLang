# Service Level Objectives (SLO) and Service Level Agreements (SLA)
# GL-VCCI Scope 3 Carbon Intelligence Platform

**Version:** 1.0.0
**Effective Date:** 2025-11-09
**Review Cycle:** Quarterly
**Owner:** GreenLang VCCI SRE Team

---

## Executive Summary

This document defines Service Level Objectives (SLOs) and Service Level Agreements (SLAs) for the GL-VCCI Scope 3 Carbon Intelligence Platform. These metrics ensure predictable, reliable performance for enterprise carbon accounting operations.

### Key Metrics at a Glance

| Metric | SLO Target | SLA Guarantee | Measurement Window |
|--------|-----------|---------------|-------------------|
| **Availability** | 99.9% | 99.5% | Monthly |
| **API Latency (P95)** | < 500ms | < 1000ms | 5-minute intervals |
| **API Latency (P99)** | < 1000ms | < 2000ms | 5-minute intervals |
| **Error Rate** | < 0.1% | < 1.0% | 5-minute intervals |
| **Data Quality Index (DQI)** | > 3.5 | > 3.0 | Daily average |
| **Carbon Calculation Throughput** | > 100 calc/min | > 50 calc/min | 5-minute intervals |

---

## 1. Service Level Objectives (SLOs)

SLOs are internal targets that guide operational excellence. They are more stringent than SLAs to provide a buffer before customer-facing commitments are breached.

### 1.1 Availability SLO

**Objective:** Maintain 99.9% uptime (Three Nines)

**Details:**
- **Maximum Downtime:** 43.2 minutes per month
- **Measurement:** Uptime checks via `/health/ready` endpoint every 30 seconds
- **Calculation:** `(Total time - Downtime) / Total time × 100`
- **Exclusions:**
  - Scheduled maintenance (announced 7 days in advance, max 4 hours/month)
  - Force majeure events (natural disasters, internet backbone failures)
  - Client-side issues (DNS, network, browser compatibility)

**Error Budget:**
- **Monthly:** 43.2 minutes
- **Quarterly:** 129.6 minutes
- **Annual:** 8.76 hours

**Escalation:**
- < 99.9%: Warning - Review incident, identify root cause
- < 99.5%: Critical - Immediate incident response, executive notification
- < 99.0%: Emergency - All-hands incident, customer communication

---

### 1.2 API Latency SLOs

**Objective:** Deliver fast, responsive API performance

#### 1.2.1 P95 Latency (95th Percentile)

**Target:** < 500ms

**Details:**
- 95% of all API requests complete within 500 milliseconds
- Measured across all HTTP endpoints (`/api/v1/*`)
- Excludes long-running operations (batch calculations, report generation)
- Measurement window: 5-minute rolling average

**Thresholds:**
- **Green:** < 500ms (meeting SLO)
- **Yellow:** 500ms - 800ms (warning, investigate performance)
- **Red:** > 800ms (SLO breach, immediate action)

#### 1.2.2 P99 Latency (99th Percentile)

**Target:** < 1000ms (1 second)

**Details:**
- 99% of all API requests complete within 1 second
- Ensures exceptional user experience even for slowest requests
- Excludes intentionally long operations:
  - Batch carbon calculations (> 100 products)
  - Annual report generation
  - Supplier data imports (> 1000 records)

**Thresholds:**
- **Green:** < 1000ms (meeting SLO)
- **Yellow:** 1000ms - 1500ms (warning)
- **Red:** > 1500ms (SLO breach)

---

### 1.3 Error Rate SLO

**Objective:** Minimize request failures

**Target:** < 0.1% error rate (99.9% success rate)

**Details:**
- Measured as: `(5xx errors / total requests) × 100`
- Includes all HTTP 500-599 status codes
- Measurement window: 5-minute rolling average
- Client errors (4xx) excluded from error rate calculation

**Error Categories:**
- **5xx Server Errors:** System failures, database errors, timeout errors
- **Retryable Errors:** Temporary failures that succeed on retry
- **Non-Retryable Errors:** Permanent failures requiring intervention

**Thresholds:**
- **Green:** < 0.1% (meeting SLO)
- **Yellow:** 0.1% - 0.5% (warning, investigate)
- **Red:** > 0.5% (SLO breach, incident response)

**Error Budget:**
- **Per 1M requests:** Max 1,000 errors
- **Per day (100k requests):** Max 100 errors
- **Per hour (4.2k requests):** Max 4 errors

---

### 1.4 Data Quality Index (DQI) SLO

**Objective:** Ensure high-quality carbon data

**Target:** Average DQI > 3.5 (on 1-5 scale per GHG Protocol)

**Details:**
- DQI Scale (GHG Protocol Data Quality Indicators):
  - **5:** Highest quality - Supplier-specific primary data
  - **4:** High quality - Secondary data from verified sources
  - **3:** Medium quality - Industry averages with regional specificity
  - **2:** Low quality - Proxy data or outdated factors
  - **1:** Lowest quality - Default global averages

**Calculation:**
```
DQI_average = Σ(DQI_score × emissions_tCO2e) / Σ(emissions_tCO2e)
```

**Tier Distribution Targets:**
- **Tier 1 (DQI 4-5):** > 30% of total emissions
- **Tier 2 (DQI 3):** 30-50% of total emissions
- **Tier 3 (DQI 1-2):** < 20% of total emissions

**Thresholds:**
- **Green:** DQI > 3.5 (excellent quality)
- **Yellow:** DQI 3.0 - 3.5 (acceptable quality)
- **Red:** DQI < 3.0 (poor quality, corrective action needed)

---

### 1.5 Carbon Calculation Throughput SLO

**Objective:** Process carbon calculations efficiently

**Target:** > 100 calculations per minute

**Details:**
- Measures system capacity for concurrent calculations
- Includes all 15 Scope 3 categories
- Excludes batch operations (measured separately)
- Measurement window: 5-minute rolling average

**Performance Tiers:**
- **Excellent:** > 100 calc/min (meeting SLO)
- **Good:** 75-100 calc/min (acceptable)
- **Degraded:** 50-75 calc/min (warning)
- **Critical:** < 50 calc/min (capacity issue)

**Scaling Triggers:**
- **Auto-scale up:** Throughput > 150 calc/min sustained for 10 minutes
- **Auto-scale down:** Throughput < 50 calc/min sustained for 30 minutes

---

### 1.6 Dependency Health SLOs

**Objective:** Monitor critical external dependencies

#### Database (PostgreSQL)

- **Availability:** 99.95%
- **Query Latency (P95):** < 50ms
- **Connection Pool Utilization:** < 80%

#### Cache (Redis)

- **Availability:** 99.9%
- **Operation Latency (P95):** < 10ms
- **Memory Utilization:** < 85%

#### Vector Database (Weaviate)

- **Availability:** 99.9%
- **Search Latency (P95):** < 200ms
- **Index Health:** All classes indexed

#### External APIs

- **Factor Broker API:** 99.5% availability
- **LLM Provider (Claude/GPT):** 99% availability (with fallback)
- **ERP SAP Connector:** 99% availability
- **Entity Resolution (LEI/DUNS):** 95% availability

---

## 2. Service Level Agreements (SLAs)

SLAs are contractual commitments to customers. They define minimum acceptable performance and remediation for breaches.

### 2.1 Availability SLA

**Guarantee:** 99.5% uptime (Two and a Half Nines)

**Monthly Downtime Allowance:** 3 hours 36 minutes

**Measurement:**
- Calendar month basis
- Excludes scheduled maintenance windows (max 4 hours/month, announced 7 days advance)
- Measured via synthetic monitoring (external probes)

**Customer Credits for Breaches:**

| Uptime Achieved | Downtime | Service Credit |
|----------------|----------|----------------|
| < 99.5% but ≥ 99.0% | 3.6 - 7.2 hours | 10% monthly fee |
| < 99.0% but ≥ 98.0% | 7.2 - 14.4 hours | 25% monthly fee |
| < 98.0% | > 14.4 hours | 50% monthly fee |

**Credit Request Process:**
1. Customer submits claim within 30 days of incident
2. GreenLang validates downtime records
3. Credits applied to next billing cycle

---

### 2.2 Performance SLA

**Guarantee:** 95th percentile API latency < 1000ms

**Measurement:**
- 5-minute rolling windows
- Averaged over monthly period
- Excludes explicitly long-running operations (documented in API)

**Breach Threshold:**
- If P95 latency exceeds 1000ms for > 10% of measurement windows in a month

**Customer Credits:**

| P95 Latency | Impact | Service Credit |
|------------|--------|----------------|
| < 1000ms | None | 0% |
| 1000-1500ms for > 10% windows | Degraded UX | 5% monthly fee |
| > 1500ms for > 10% windows | Severe degradation | 15% monthly fee |

---

### 2.3 Data Accuracy SLA

**Guarantee:** Data Quality Index (DQI) average ≥ 3.0

**Measurement:**
- Monthly weighted average across all calculations
- Per GHG Protocol Data Quality Indicators

**Breach Definition:**
- Monthly DQI average falls below 3.0

**Customer Remediation:**
- **DQI 2.5-3.0:** Written explanation + improvement plan (no credit)
- **DQI < 2.5:** 10% monthly fee credit + mandatory data quality audit

**Exclusions:**
- Customer-provided data inaccuracies (e.g., incorrect product specifications)
- Emission factors not yet available for new products/regions
- Force majeure affecting emission factor databases

---

### 2.4 Support Response SLA

**Guarantee:** Timely response to support tickets

#### Response Time Targets

| Priority | Definition | First Response | Resolution Target |
|----------|-----------|----------------|-------------------|
| **P1 - Critical** | Platform down, data loss, security breach | 1 hour | 4 hours |
| **P2 - High** | Major feature broken, significant performance degradation | 4 hours | 24 hours |
| **P3 - Medium** | Minor feature issues, workaround available | 8 hours | 72 hours |
| **P4 - Low** | Questions, feature requests, cosmetic issues | 24 hours | Best effort |

**Support Hours:**
- **Business Hours:** Monday-Friday, 9:00 AM - 6:00 PM (Customer's timezone)
- **After-Hours:** P1 Critical only (24/7/365)

**Support Channels:**
- Email: support@greenlang.com
- Slack: #vcci-support (enterprise customers)
- Phone: P1 Critical only
- Portal: support.greenlang.com

**Escalation Path:**
1. L1 Support Engineer (0-2 hours)
2. L2 Senior Engineer (2-6 hours)
3. L3 Principal Engineer (6-12 hours)
4. VP Engineering (> 12 hours or major incident)

---

## 3. Monitoring and Alerting

### 3.1 Real-Time Monitoring

**Metrics Collection:**
- Prometheus scraping every 30 seconds
- Custom VCCI metrics + standard HTTP metrics
- Circuit breaker state tracking
- Dependency health checks

**Dashboards:**
- **Production Dashboard:** Grafana dashboard for operations team
- **Executive Dashboard:** High-level KPIs for leadership
- **Customer Dashboard:** Public status page (status.greenlang.com)

### 3.2 Alerting Rules

**Alert Severity Levels:**

| Severity | Definition | Notification | Escalation |
|----------|-----------|--------------|------------|
| **Critical** | SLA breach imminent or active | PagerDuty page | Immediate |
| **Warning** | SLO breach, approaching SLA | Slack + email | 30 minutes |
| **Info** | Anomaly detected | Slack only | None |

**Key Alerts:**
- `VCCISLAViolation`: Success rate < 99.5% for 15 minutes
- `VCCISLALatencyViolation`: P95 latency > 1000ms for 30 minutes
- `VCCIHighErrorRate`: Error rate > 0.5% for 5 minutes
- `CircuitBreakerOpen`: Critical dependency circuit open
- `VCCILowDataQualityScore`: DQI < 3.0 for 2 hours

### 3.3 Error Budget Policy

**Philosophy:** Balance innovation with reliability

**Error Budget Calculation:**
```
Error Budget = (1 - SLO) × Total time
Monthly Error Budget (99.9%) = 0.1% × 43,200 minutes = 43.2 minutes
```

**Error Budget Consumption:**
- **< 25% consumed:** Green - Fast iteration, deploy freely
- **25-50% consumed:** Yellow - Increased testing, cautious deploys
- **50-75% consumed:** Orange - Code freeze on risky features, focus on reliability
- **> 75% consumed:** Red - Deploy freeze (except critical fixes), all-hands stabilization

**Monthly Review:**
- Calculate error budget consumption
- Root cause analysis for major incidents
- Adjust SLOs if consistently under-consuming (too conservative) or over-consuming (unrealistic)

---

## 4. Incident Management

### 4.1 Incident Classification

**Severity Levels:**

| Severity | Definition | Examples | Response Time |
|----------|-----------|----------|---------------|
| **SEV-1** | Complete outage, data loss, security breach | Platform down, database corruption, data breach | < 15 min |
| **SEV-2** | Major feature broken, severe degradation | Carbon calc failures, login broken, major API errors | < 1 hour |
| **SEV-3** | Minor feature issues, degraded performance | Slow queries, intermittent errors, UI bugs | < 4 hours |
| **SEV-4** | Cosmetic issues, questions | Typos, minor UI glitches, documentation issues | < 24 hours |

### 4.2 Incident Response Process

**SEV-1 Response:**
1. **Detection:** Automated alert or customer report (< 5 min)
2. **Notification:** Page on-call engineer (< 5 min)
3. **Acknowledgment:** On-call engineer acknowledges (< 5 min)
4. **War Room:** Incident channel created, stakeholders joined (< 10 min)
5. **Investigation:** Root cause analysis begins (< 15 min)
6. **Mitigation:** Deploy fix or workaround (< 1 hour target)
7. **Resolution:** Service fully restored (< 4 hours target)
8. **Post-Mortem:** Written report within 72 hours

**Communication:**
- **Internal:** Slack #incidents channel
- **External:** Status page updates every 30 minutes
- **Customer:** Email notification for SEV-1/2 affecting them

### 4.3 Post-Incident Review

**Blameless Post-Mortems:**
- Conducted for all SEV-1 and SEV-2 incidents
- Focus on systems, processes, not individuals
- Template includes:
  - **Timeline:** Detailed incident chronology
  - **Root Cause:** 5-Why analysis
  - **Impact:** Affected customers, downtime, financial impact
  - **Action Items:** Preventative measures with owners and deadlines

**Learning Culture:**
- Post-mortems shared company-wide
- Monthly incident review meeting
- Continuous improvement of runbooks and automation

---

## 5. Capacity Planning

### 5.1 Capacity Targets

**Compute Resources:**
- **CPU Utilization:** Maintain < 70% average
- **Memory Utilization:** Maintain < 75% average
- **Headroom:** Always 2x peak capacity available

**Database:**
- **Connection Pool:** < 80% utilization
- **Storage:** < 70% full (daily growth monitored)
- **IOPS:** < 80% of provisioned capacity

**Cache (Redis):**
- **Memory:** < 85% full
- **Hit Rate:** > 90%
- **Evictions:** < 1% of sets

### 5.2 Scaling Strategy

**Horizontal Scaling (Preferred):**
- Add application pods (Kubernetes HPA)
- Read replicas for database
- Redis cluster nodes

**Vertical Scaling (Limited):**
- Increase pod resources for bursty workloads
- Database instance size for high-complexity queries

**Auto-Scaling Rules:**
- **Scale Up:** CPU > 70% for 5 minutes OR Memory > 75% for 5 minutes
- **Scale Down:** CPU < 30% for 30 minutes AND Memory < 40% for 30 minutes
- **Min Pods:** 3 (high availability)
- **Max Pods:** 20 (cost control)

---

## 6. Disaster Recovery

### 6.1 Recovery Objectives

**Recovery Time Objective (RTO):** 4 hours
- Maximum acceptable downtime before service restoration

**Recovery Point Objective (RPO):** 15 minutes
- Maximum acceptable data loss (time between backups)

### 6.2 Backup Strategy

**Database Backups:**
- **Full Backup:** Daily at 2:00 AM UTC
- **Incremental Backup:** Every 15 minutes (WAL archiving)
- **Retention:** 30 days online, 1 year archival

**Configuration Backups:**
- **Kubernetes Manifests:** Git repository (version controlled)
- **Secrets:** Vault backup daily
- **Application Config:** Stored in Git + S3

**Testing:**
- **Restore Test:** Monthly full database restoration to staging
- **DR Drill:** Quarterly full disaster recovery simulation

### 6.3 Failover Procedures

**Database Failover:**
- Automatic promotion of read replica to master (< 5 min)
- Application reconnects automatically

**Multi-Region Failover:**
- Manual process (future: automatic)
- DNS switchover to backup region
- Target: < 30 minutes RTO

---

## 7. Continuous Improvement

### 7.1 SLO Review Cadence

**Monthly:**
- Review SLO/SLA compliance
- Analyze error budget consumption
- Identify trends and anomalies

**Quarterly:**
- Formal SLO review meeting
- Adjust targets based on performance history
- Update alerting thresholds

**Annually:**
- Comprehensive SLA contract review
- Negotiate updated terms with customers
- Benchmark against industry standards

### 7.2 Performance Optimization

**Ongoing Initiatives:**
- Database query optimization (target: 10% latency reduction/quarter)
- Caching strategy improvements (target: 95% hit rate)
- Circuit breaker tuning (minimize false positives)
- Load testing (monthly, target: 2x current peak capacity)

---

## 8. Appendix

### 8.1 Measurement Tools

- **Uptime Monitoring:** Prometheus blackbox exporter + external StatusCake
- **Latency Tracking:** Prometheus histograms (P50, P95, P99)
- **Error Rates:** Prometheus counters + Sentry error tracking
- **DQI Calculation:** Custom Python scripts + PostgreSQL queries
- **Dashboards:** Grafana (internal) + custom status page (public)

### 8.2 Definitions

- **Availability:** Percentage of time the service is operational and accessible
- **Latency:** Time from request initiation to response completion
- **Error Rate:** Percentage of requests resulting in 5xx server errors
- **DQI (Data Quality Indicator):** GHG Protocol standard for carbon data quality
- **SLO (Service Level Objective):** Internal performance target
- **SLA (Service Level Agreement):** Contractual customer commitment
- **Error Budget:** Allowed failure rate within SLO parameters
- **MTTR (Mean Time To Repair):** Average time to restore service after failure
- **MTBF (Mean Time Between Failures):** Average operational time between incidents

### 8.3 Contact Information

**SRE On-Call:** sre-oncall@greenlang.com
**Incident Commander:** incidents@greenlang.com
**Support Team:** support@greenlang.com
**Escalation:** escalations@greenlang.com

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-09 | VCCI SRE Team | Initial release - comprehensive SLO/SLA definitions |

---

**Last Updated:** 2025-11-09
**Next Review:** 2026-02-09 (Quarterly)
**Owner:** GreenLang VCCI SRE Team
**Approved By:** VP Engineering, Chief Product Officer
