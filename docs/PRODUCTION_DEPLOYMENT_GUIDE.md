# GreenLang Intelligence Layer - Production Deployment Guide

**Version:** 1.0
**Last Updated:** 2025-10-01
**Target Environment:** Production
**Risk Level:** HIGH - Follow all steps carefully

---

## ⚠️ Critical Pre-Deployment Warning

**READ THIS FIRST:**

This deployment involves:
- Real customer data
- Production API costs ($$$)
- Live business operations
- Regulatory compliance requirements

**Required Approvals:**
- [ ] Engineering Lead sign-off
- [ ] Security review complete
- [ ] Budget approval ($X,XXX/month)
- [ ] Stakeholder notification
- [ ] Incident response team on standby

**Staging Requirements:**
- [ ] 7+ days successful staging operation
- [ ] Load testing complete (>10K requests)
- [ ] All acceptance criteria met
- [ ] Team trained on operations
- [ ] Runbooks and procedures documented

---

## Production Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  PRODUCTION ENVIRONMENT (High Availability)                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                    │
│  │ Load         │────────▶│ GreenLang    │                    │
│  │ Balancer     │         │ App Cluster  │                    │
│  │ (HA Proxy)   │         │ (3+ instances)│                   │
│  └──────────────┘         └──────┬───────┘                    │
│                                    │                            │
│                                    ▼                            │
│                       ┌────────────────────────┐               │
│                       │ Intelligence Layer     │               │
│                       │ - Provider Router      │               │
│                       │ - Circuit Breakers     │               │
│                       │ - Budget Enforcement   │               │
│                       └────────┬───────────────┘               │
│                                │                                │
│                     ┌──────────┴───────────┐                  │
│                     ▼                       ▼                   │
│          ┌─────────────────┐    ┌─────────────────┐           │
│          │ OpenAI Provider │    │Anthropic Provider│           │
│          │ (Production Key)│    │ (Production Key) │           │
│          └─────────────────┘    └─────────────────┘           │
│                     │                       │                   │
│                     └──────────┬───────────┘                  │
│                                ▼                                │
│                    ┌────────────────────────┐                  │
│                    │  Monitoring &  Alerts   │                  │
│                    │  - Prometheus           │                  │
│                    │  - Grafana              │                  │
│                    │  - PagerDuty            │                  │
│                    │  - CloudWatch           │                  │
│                    └────────────────────────┘                  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Pre-Deployment Checklist

### 1. Security & Compliance

- [ ] **Security Audit:** Complete security review
  - Secrets management (no hardcoded keys)
  - API key rotation procedure
  - Access control (IAM roles, least privilege)
  - Data encryption (at rest and in transit)
  - GDPR/compliance requirements

- [ ] **Penetration Testing:** External security assessment
- [ ] **Dependency Audit:** All packages scanned for vulnerabilities
  ```bash
  pip-audit
  safety check
  ```

- [ ] **Secrets Management:**
  - Use AWS Secrets Manager / HashiCorp Vault
  - NEVER commit secrets to git
  - Rotate keys every 90 days

### 2. Infrastructure

- [ ] **Production Servers:** Minimum 3 instances for HA
  - CPU: 8 cores per instance
  - RAM: 16GB per instance
  - Network: Low latency to provider APIs

- [ ] **Load Balancing:** Configure HAProxy/ALB
  - Health checks every 30s
  - Automatic failover
  - SSL termination

- [ ] **Database:** (if using persistence)
  - Primary + replica for HA
  - Automated backups every 6 hours
  - Point-in-time recovery enabled

- [ ] **Monitoring Stack:**
  - Prometheus (metrics collection)
  - Grafana (dashboards)
  - AlertManager (alert routing)
  - PagerDuty (on-call escalation)

### 3. Budget & Cost Controls

- [ ] **Budget Approval:** Signed budget approval for:
  - Monthly LLM API costs: $____
  - Infrastructure costs: $____
  - Monitoring costs: $____
  - **Total:** $____/month

- [ ] **Cost Controls:**
  - Daily budget limits configured
  - Alert thresholds set (80%, 90%, 95%)
  - Auto-cutoff at 100% budget
  - Cost anomaly detection

- [ ] **Provider Limits:**
  - OpenAI: Set organization spending limit
  - Anthropic: Configure rate limits
  - Test hard limits work as expected

### 4. Operational Readiness

- [ ] **Runbooks:** Complete and tested
  - Incident response procedures
  - Escalation matrix
  - Rollback procedures
  - Provider outage handling

- [ ] **On-Call Rotation:** Configured in PagerDuty
  - Primary: Engineering team
  - Secondary: Engineering lead
  - Escalation: CTO

- [ ] **Team Training:** All team members trained
  - Monitoring dashboard usage
  - Alert handling
  - Incident response
  - Rollback procedures

---

## Deployment Plan

### Phase 1: Pre-Deployment (Day 0)

**Timeline:** 08:00 - 10:00

```bash
# 1. Final smoke tests in staging
pytest tests/smoke/ -v --staging

# 2. Backup current production state
./scripts/backup_production.sh

# 3. Notify stakeholders
./scripts/notify_deployment_start.sh

# 4. Create deployment branch
git checkout -b deploy/intelligence-layer-v1.0
git tag -a v1.0.0 -m "Intelligence Layer Production Release"
```

### Phase 2: Infrastructure Setup (Day 0)

**Timeline:** 10:00 - 12:00

```bash
# 1. Deploy monitoring stack
kubectl apply -f k8s/monitoring/

# 2. Verify monitoring operational
curl http://prometheus:9090/-/healthy
curl http://grafana:3000/api/health

# 3. Configure alerts
kubectl apply -f k8s/alerts/intelligence-layer-alerts.yaml

# 4. Test alert routing
./scripts/test_alerts.sh
```

### Phase 3: Application Deployment (Day 0)

**Timeline:** 13:00 - 15:00

```bash
# 1. Deploy application (blue-green deployment)
kubectl apply -f k8s/intelligence-layer-green.yaml

# 2. Wait for health checks
kubectl wait --for=condition=ready pod -l app=intelligence-layer-green

# 3. Smoke tests on green deployment
./scripts/smoke_test_green.sh

# 4. Switch traffic to green (10% initially)
kubectl patch service intelligence-layer -p '{"spec":{"selector":{"version":"green","weight":"10"}}}'
```

### Phase 4: Gradual Rollout (Day 0-1)

**Timeline:** 15:00 Day 0 → 15:00 Day 1

```bash
# Hour 0: 10% traffic
kubectl patch service intelligence-layer -p '{"spec":{"selector":{"weight":"10"}}}'
# Monitor for 2 hours

# Hour 2: 25% traffic
kubectl patch service intelligence-layer -p '{"spec":{"selector":{"weight":"25"}}}'
# Monitor for 2 hours

# Hour 4: 50% traffic
kubectl patch service intelligence-layer -p '{"spec":{"selector":{"weight":"50"}}}'
# Monitor overnight

# Day 1, Hour 12: 100% traffic
kubectl patch service intelligence-layer -p '{"spec":{"selector":{"weight":"100"}}}'

# Remove blue deployment after 24h stability
kubectl delete -f k8s/intelligence-layer-blue.yaml
```

### Phase 5: Post-Deployment Validation (Day 1-7)

**Validation Windows:**
- **24 hours:** Monitor for critical issues
- **48 hours:** Verify cost projections
- **7 days:** Confirm long-term stability

**Metrics to Track:**
- Uptime: Target 99.9%
- Success rate: Target >99%
- p95 latency: Target <2000ms
- Cost per request: Within 10% of projections
- Alert noise: <5 false positives/day

---

## Monitoring & Alerts

### Critical Alerts (PagerDuty - Immediate)

| Alert | Threshold | Action |
|-------|-----------|--------|
| Service Down | 2 failed health checks | Page on-call immediately |
| Error Rate Spike | >5% for 5 minutes | Page on-call |
| Budget Exceeded | 100% of daily budget | Auto-cutoff + page lead |
| Circuit Breaker Open | Any circuit open >5min | Investigate provider |
| p95 Latency High | >5000ms for 10min | Investigate performance |

### Warning Alerts (Slack - Within 15min)

| Alert | Threshold | Action |
|-------|-----------|--------|
| Budget Warning | >80% of daily budget | Review usage patterns |
| Success Rate Drop | <98% for 15min | Monitor closely |
| Latency Increase | p95 >3000ms | Check provider status |
| JSON Retry Rate High | >10% failures | Check schema changes |

### Dashboard URLs

- **Main Dashboard:** https://grafana.company.com/d/intelligence-layer
- **Cost Dashboard:** https://grafana.company.com/d/intelligence-cost
- **Provider Health:** https://grafana.company.com/d/provider-health
- **Circuit Breakers:** https://grafana.company.com/d/circuit-breakers

---

## Rollback Procedures

### Immediate Rollback (< 5 minutes)

**Trigger Conditions:**
- Error rate >20%
- Multiple circuit breakers open
- Budget runaway (>2x expected rate)
- Data integrity issues

**Procedure:**
```bash
# 1. Switch traffic back to blue deployment
kubectl patch service intelligence-layer -p '{"spec":{"selector":{"version":"blue"}}}'

# 2. Verify blue is healthy
curl https://api.company.com/health

# 3. Notify team
./scripts/notify_rollback.sh "Emergency rollback: <reason>"

# 4. Investigate root cause
./scripts/collect_logs.sh --since=1h
```

### Partial Rollback (< 15 minutes)

**Trigger Conditions:**
- Cost higher than expected but manageable
- Minor functionality issues
- Performance degradation

**Procedure:**
```bash
# 1. Reduce traffic to green
kubectl patch service intelligence-layer -p '{"spec":{"selector":{"weight":"10"}}}'

# 2. Monitor for improvement
watch curl https://api.company.com/metrics

# 3. If improved: continue rollout
# If not improved: full rollback (see above)
```

---

## Cost Management

### Daily Budget Monitoring

```bash
# View cost dashboard
python -c "from greenlang.intelligence.runtime.dashboard import print_dashboard; print_dashboard()"

# Export cost report
gl intelligence cost-report --date=today --format=csv > cost_report.csv
```

### Expected Costs (Update with Your Projections)

| Component | Daily Cost | Monthly Cost |
|-----------|-----------|--------------|
| OpenAI API | $XXX | $X,XXX |
| Anthropic API | $XXX | $X,XXX |
| Infrastructure | $XX | $XXX |
| Monitoring | $XX | $XXX |
| **TOTAL** | **$XXX** | **$X,XXX** |

### Cost Optimization

- **Provider Router:** Saves 60-90% by routing simple queries to cheap models
- **Circuit Breaker:** Prevents wasted calls during outages
- **Budget Caps:** Hard limits prevent runaway costs
- **Context Management:** Reduces token usage for long conversations

---

## Incident Response

### Severity Levels

**P0 (Critical):** Service completely down
- Response Time: Immediate
- Escalation: Page CTO if not resolved in 30min
- Communication: Status page + customer notifications

**P1 (High):** Degraded service
- Response Time: Within 15min
- Escalation: Engineering lead after 1 hour
- Communication: Internal Slack updates

**P2 (Medium):** Minor issues
- Response Time: Within 1 hour
- Escalation: None required
- Communication: Document in ticket

### Common Incidents

#### Incident: Provider Outage

**Symptoms:** All requests to OpenAI failing, circuit breaker open

**Response:**
1. Confirm provider outage: Check https://status.openai.com/
2. Verify circuit breaker working: Should fast-fail requests
3. If prolonged (>30min): Switch to Anthropic only
   ```bash
   kubectl set env deployment/intelligence-layer GREENLANG_DISABLE_OPENAI=true
   ```
4. Monitor Anthropic capacity
5. When OpenAI recovers: Reset circuit breaker and re-enable

#### Incident: Budget Exceeded

**Symptoms:** "Budget exceeded" errors, requests being rejected

**Response:**
1. Check if legitimate usage spike or attack
2. If legitimate: Emergency budget increase
   ```bash
   kubectl set env deployment/intelligence-layer GREENLANG_BUDGET_MAX_USD=XXX
   ```
3. If attack: Rate limit or block offending IPs
4. Review cost dashboard for anomalies
5. Incident post-mortem to prevent recurrence

#### Incident: High Latency

**Symptoms:** p95 latency >5000ms, slow user experience

**Response:**
1. Check provider status pages
2. Review circuit breaker states
3. Check for context overflow (long conversations)
4. Scale up infrastructure if needed
   ```bash
   kubectl scale deployment/intelligence-layer --replicas=5
   ```
5. Enable request queuing if overloaded

---

## Post-Deployment

### Day 1 Review

- [ ] Review metrics from first 24 hours
- [ ] Verify cost tracking accuracy
- [ ] Check for any unexpected alerts
- [ ] Gather team feedback
- [ ] Document any issues or surprises

### Week 1 Review

- [ ] Weekly cost report vs projections
- [ ] Performance trends analysis
- [ ] User feedback review
- [ ] Optimization opportunities identified
- [ ] Update runbooks based on learnings

### Month 1 Review

- [ ] Monthly cost analysis
- [ ] Capacity planning review
- [ ] Provider performance comparison
- [ ] Tool usage analytics
- [ ] Roadmap planning for improvements

---

## Success Criteria

Production deployment considered successful when:

- [ ] **Uptime:** 99.9% over 30 days
- [ ] **Success Rate:** >99% LLM request success
- [ ] **Latency:** p95 <2000ms, p99 <5000ms
- [ ] **Cost:** Within ±10% of projections
- [ ] **Alerts:** <3 false positives per week
- [ ] **Incidents:** Zero P0 incidents in first 30 days
- [ ] **Customer Impact:** Zero customer-facing issues
- [ ] **Team Confidence:** Team comfortable with operations

---

## Appendix

### A. Environment Variables (Production)

```bash
# Provider API Keys (from Secrets Manager)
OPENAI_API_KEY=<from-secrets-manager>
ANTHROPIC_API_KEY=<from-secrets-manager>

# Environment
GREENLANG_ENV=production
GREENLANG_LOG_LEVEL=WARNING  # Less verbose in production

# Budget (Daily limits)
GREENLANG_BUDGET_MAX_USD=500.00
GREENLANG_BUDGET_ALERT_THRESHOLD=0.80

# Circuit Breaker
GREENLANG_CIRCUIT_BREAKER_THRESHOLD=5
GREENLANG_CIRCUIT_BREAKER_TIMEOUT=60

# Monitoring
GREENLANG_METRICS_ENABLED=true
GREENLANG_METRICS_PORT=9090
GREENLANG_METRICS_EXPORT_INTERVAL=30

# Logging
GREENLANG_LOG_FORMAT=json
GREENLANG_LOG_DESTINATION=/var/log/greenlang/intelligence.log

# Performance
GREENLANG_CONTEXT_CACHE_SIZE=1000
GREENLANG_PROVIDER_TIMEOUT_MS=30000
```

### B. Health Check Endpoints

```bash
# Application health
GET /health
Response: {"status": "healthy", "version": "1.0.0"}

# Intelligence layer health
GET /intelligence/health
Response: {
  "status": "healthy",
  "providers": {
    "openai": "healthy",
    "anthropic": "healthy"
  },
  "circuit_breakers": {
    "openai": "closed",
    "anthropic": "closed"
  }
}

# Metrics
GET /metrics
Response: Prometheus-format metrics
```

### C. Emergency Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| On-Call Engineer | TBD | +1-XXX-XXX-XXXX | oncall@company.com |
| Engineering Lead | TBD | +1-XXX-XXX-XXXX | lead@company.com |
| CTO | TBD | +1-XXX-XXX-XXXX | cto@company.com |
| PagerDuty | N/A | N/A | pagerduty.com/escalations |

---

**Last Updated:** 2025-10-01
**Approved By:** [Engineering Lead], [Security], [Operations]
**Next Review:** 2025-11-01

**Document Classification:** INTERNAL - Production Operations
