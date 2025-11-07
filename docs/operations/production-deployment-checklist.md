# GreenLang Production Deployment Checklist

**Version:** 1.0
**Last Updated:** 2025-11-07
**Document Classification:** CRITICAL - Operations
**Review Cycle:** Monthly
**Next Review:** 2025-12-07

---

## Executive Summary

This comprehensive checklist ensures safe, reliable deployments to the GreenLang production environment. It covers pre-deployment preparation, deployment execution, post-deployment validation, and rollback procedures.

**Deployment Philosophy:**
- **Safety First:** Better to delay than deploy broken code
- **Incremental:** Gradual rollout with validation gates
- **Reversible:** Always have a rollback plan
- **Monitored:** Observe metrics at every stage
- **Documented:** Record everything

**Success Criteria:**
- Zero P0/P1 incidents during deployment
- Error rate remains < 1%
- p95 latency remains < 500ms
- All smoke tests passing
- Successful rollback capability verified

---

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Deployment Day Preparation](#deployment-day-preparation)
3. [Deployment Execution](#deployment-execution)
4. [Post-Deployment Validation](#post-deployment-validation)
5. [Rollback Procedures](#rollback-procedures)
6. [Deployment Methods](#deployment-methods)

---

## Pre-Deployment Checklist

### 1. Code Quality & Testing (T-7 days)

#### Code Review
- [ ] All code reviewed by at least 2 senior engineers
- [ ] No unresolved review comments
- [ ] Architecture review completed (for major changes)
- [ ] Performance review completed (for performance-sensitive changes)
- [ ] Security review completed (for security-sensitive changes)

#### Testing - Unit Tests
- [ ] All unit tests passing
- [ ] Code coverage ≥ 80% for new code
- [ ] No skipped or ignored tests
- [ ] Test execution time < 5 minutes

```bash
# Run unit tests
pytest tests/unit/ -v --cov=greenlang --cov-report=html

# Verify coverage
coverage report --fail-under=80
```

#### Testing - Integration Tests
- [ ] All integration tests passing
- [ ] Database migration tests passing
- [ ] API contract tests passing
- [ ] Cross-service integration tests passing
- [ ] Test execution time < 30 minutes

```bash
# Run integration tests
pytest tests/integration/ -v --maxfail=1

# Run API contract tests
pytest tests/contracts/ -v
```

#### Testing - End-to-End Tests
- [ ] All E2E tests passing in staging
- [ ] Critical user flows validated
- [ ] Performance benchmarks met
- [ ] Test execution time < 60 minutes

```bash
# Run E2E tests against staging
pytest tests/e2e/ --env=staging -v --html=report.html
```

#### Testing - Performance Tests
- [ ] Load tests executed against staging
- [ ] Performance benchmarks met:
  - [ ] p50 latency < 200ms
  - [ ] p95 latency < 500ms
  - [ ] p99 latency < 1000ms
  - [ ] Throughput ≥ 200 RPS
  - [ ] Error rate < 0.1%
- [ ] Stress tests executed (150% expected load)
- [ ] Soak tests executed (24-hour sustained load)
- [ ] No memory leaks detected

```bash
# Run load tests
locust -f tests/performance/locustfile.py \
  --host=https://staging.greenlang.io \
  --users=500 --spawn-rate=10 --run-time=1h \
  --html=load-test-report.html
```

#### Testing - Security Tests
- [ ] Security scan completed (no critical/high vulnerabilities)
- [ ] Dependency audit clean
- [ ] Secrets scanner run (no secrets in code)
- [ ] OWASP Top 10 tests passing
- [ ] Penetration test completed (for major releases)

```bash
# Run security scans
bandit -r greenlang/ -f json -o security-report.json
safety check
git secrets --scan
```

---

### 2. Documentation (T-7 days)

#### User Documentation
- [ ] User-facing documentation updated
- [ ] API documentation updated
- [ ] Changelog/release notes prepared
- [ ] Migration guide written (if breaking changes)
- [ ] FAQ updated with new features

#### Technical Documentation
- [ ] Architecture docs updated
- [ ] API reference updated
- [ ] Configuration reference updated
- [ ] Runbooks updated with new procedures
- [ ] Troubleshooting guide updated

#### Deployment Documentation
- [ ] Deployment plan documented
- [ ] Rollback plan documented
- [ ] Dependencies documented
- [ ] Configuration changes documented
- [ ] Database migrations documented

---

### 3. Database (T-5 days)

#### Migration Scripts
- [ ] Migration scripts written and reviewed
- [ ] Migration scripts tested in dev environment
- [ ] Migration scripts tested in staging environment
- [ ] Rollback migrations written and tested
- [ ] Migration execution time estimated (should be < 5 minutes)
- [ ] Data validation queries prepared

```bash
# Test migration in staging
psql -h staging-db.greenlang.io -f migrations/2025-11-07-add-indexes.sql

# Verify migration
psql -h staging-db.greenlang.io -f migrations/validate-2025-11-07.sql

# Test rollback
psql -h staging-db.greenlang.io -f migrations/rollback-2025-11-07.sql
```

#### Backup & Recovery
- [ ] Pre-deployment backup scheduled
- [ ] Backup restoration tested in staging
- [ ] Database backup retention policy confirmed
- [ ] Point-in-time recovery tested
- [ ] Backup storage capacity verified

```bash
# Create pre-deployment backup
pg_dump -h db.greenlang.io -U admin greenlang \
  -F c -f backup/pre-deployment-$(date +%Y%m%d).dump

# Verify backup
pg_restore --list backup/pre-deployment-$(date +%Y%m%d).dump
```

#### Database Performance
- [ ] Query performance tested
- [ ] No long-running queries (>1s)
- [ ] Indexes added for new queries
- [ ] Query execution plans reviewed
- [ ] Database connection pool sized appropriately

```bash
# Test query performance
psql -h staging-db.greenlang.io -c "EXPLAIN ANALYZE SELECT ..."

# Check for missing indexes
psql -h staging-db.greenlang.io -f scripts/check-missing-indexes.sql
```

---

### 4. Configuration (T-5 days)

#### Environment Variables
- [ ] All required environment variables documented
- [ ] Staging environment variables validated
- [ ] Production environment variables prepared
- [ ] Secrets rotated (if needed)
- [ ] No hardcoded values in code

```bash
# Validate environment variables
./scripts/validate-env.sh production

# Expected output: All required variables present
```

#### Feature Flags
- [ ] Feature flags configured
- [ ] Gradual rollout plan defined
- [ ] Kill switches identified and tested
- [ ] Feature flag cleanup plan documented

```yaml
# Feature flag configuration
feature_flags:
  new_agent_execution_engine:
    enabled: true
    rollout_percentage: 10
    kill_switch: true
```

#### Configuration Files
- [ ] All configuration files reviewed
- [ ] Staging configuration validated
- [ ] Production configuration prepared
- [ ] Configuration backup created
- [ ] Configuration diffs reviewed

```bash
# Review configuration differences
diff -u config/production.yaml.old config/production.yaml.new
```

---

### 5. Infrastructure (T-5 days)

#### Capacity Planning
- [ ] Current resource usage reviewed
- [ ] Sufficient capacity for new deployment
- [ ] Auto-scaling configured and tested
- [ ] Resource limits increased (if needed)
- [ ] Database capacity verified

```bash
# Check current resource usage
kubectl top nodes
kubectl top pods -A

# Verify auto-scaling
kubectl get hpa
```

#### Network
- [ ] Network connectivity verified
- [ ] Firewall rules updated (if needed)
- [ ] DNS records verified
- [ ] SSL certificates valid (>30 days)
- [ ] CDN configuration verified

```bash
# Check SSL certificate expiry
echo | openssl s_client -connect api.greenlang.io:443 2>/dev/null | \
  openssl x509 -noout -dates

# Verify DNS
dig api.greenlang.io +short
```

#### Load Balancing
- [ ] Load balancer health checks configured
- [ ] Connection draining configured
- [ ] Session persistence configured (if needed)
- [ ] Load balancer logs reviewed

```bash
# Verify health check configuration
aws elbv2 describe-target-health --target-group-arn $TG_ARN
```

---

### 6. Monitoring & Alerting (T-5 days)

#### Metrics
- [ ] All metrics endpoints operational
- [ ] Prometheus scraping correctly
- [ ] New metrics added for new features
- [ ] Metric retention policy verified
- [ ] Metrics exporters tested

```bash
# Verify metrics collection
curl http://prometheus.greenlang.io/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'
# Should return empty
```

#### Dashboards
- [ ] Grafana dashboards updated
- [ ] New dashboards created for new features
- [ ] Dashboard alerts configured
- [ ] Dashboard access verified

```bash
# Test dashboard access
curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
  https://grafana.greenlang.io/api/dashboards/uid/system-overview
```

#### Alerts
- [ ] Alert rules reviewed and updated
- [ ] New alerts added for new features
- [ ] Alert thresholds validated
- [ ] Alert notification channels tested
- [ ] On-call rotation verified

```bash
# Validate alert rules
promtool check rules alerting/rules.yml

# Test alert notification
./scripts/test-alert-notification.sh
```

#### Logging
- [ ] Log aggregation operational
- [ ] Log retention policy verified
- [ ] New log statements added for new features
- [ ] Log volume estimated
- [ ] Log storage capacity verified

```bash
# Check log aggregation
curl -G "http://loki.greenlang.io/loki/api/v1/query" \
  --data-urlencode 'query={job="greenlang"}' | jq
```

---

### 7. Security (T-5 days)

#### Authentication & Authorization
- [ ] Authentication mechanisms tested
- [ ] Authorization policies reviewed
- [ ] Service accounts validated
- [ ] API keys rotated (if needed)
- [ ] OAuth flows tested (if applicable)

```bash
# Test authentication
curl -H "Authorization: Bearer $API_KEY" \
  https://api.greenlang.io/v1/agents

# Expected: 200 OK
```

#### Secrets Management
- [ ] All secrets in secure storage (no hardcoded)
- [ ] Secrets rotation schedule verified
- [ ] Secrets backup verified
- [ ] Access to secrets audited

```bash
# Verify secrets in vault
vault kv list secret/greenlang/production

# Test secret retrieval
vault kv get secret/greenlang/production/database
```

#### Network Security
- [ ] Firewall rules reviewed
- [ ] Security groups configured
- [ ] Network policies applied
- [ ] TLS/SSL enforced
- [ ] Rate limiting configured

```bash
# Verify network policies
kubectl get networkpolicies -A

# Test rate limiting
for i in {1..100}; do curl https://api.greenlang.io/v1/agents; done
# Should see 429 Too Many Requests
```

#### Vulnerability Scanning
- [ ] Container images scanned
- [ ] Dependencies scanned
- [ ] OS packages scanned
- [ ] No critical vulnerabilities
- [ ] No high vulnerabilities (or approved exceptions)

```bash
# Scan container image
trivy image greenlang/api:v1.2.3

# Scan dependencies
safety check --json
```

---

### 8. Compliance & Legal (T-5 days)

#### Data Privacy
- [ ] GDPR compliance verified
- [ ] CCPA compliance verified
- [ ] Data retention policies enforced
- [ ] Privacy policy updated
- [ ] Cookie consent implemented (if applicable)

#### Audit Logging
- [ ] Audit logs enabled
- [ ] Audit log retention configured
- [ ] Audit log access controlled
- [ ] Audit log integrity verified

#### Licensing
- [ ] Third-party license compliance verified
- [ ] Open source licenses reviewed
- [ ] Commercial licenses valid
- [ ] License attribution complete

---

### 9. Staging Validation (T-3 days)

#### Staging Deployment
- [ ] Code deployed to staging
- [ ] Database migrations applied to staging
- [ ] Configuration deployed to staging
- [ ] Staging environment stable for 48+ hours

```bash
# Deploy to staging
kubectl apply -f k8s/staging/ --namespace=greenlang-staging

# Verify deployment
kubectl get pods -n greenlang-staging
kubectl get deployments -n greenlang-staging
```

#### Staging Testing
- [ ] All automated tests passing in staging
- [ ] Manual testing completed
- [ ] User acceptance testing (UAT) completed
- [ ] Performance validated in staging
- [ ] No critical bugs found

#### Load Testing Staging
- [ ] Load tests executed at 100% expected production load
- [ ] Load tests executed at 150% expected production load
- [ ] All performance targets met
- [ ] No errors during load test
- [ ] Resource usage within limits

```bash
# Run production-level load test
locust -f tests/performance/locustfile.py \
  --host=https://staging.greenlang.io \
  --users=1000 --spawn-rate=50 --run-time=2h \
  --html=staging-load-test.html
```

---

### 10. Communication & Coordination (T-2 days)

#### Stakeholder Notification
- [ ] Engineering team notified
- [ ] Product team notified
- [ ] Customer success team notified
- [ ] Support team notified
- [ ] Executive team notified (for major releases)

```
Subject: Production Deployment - [Date] [Time]

Team,

We will be deploying GreenLang v1.2.3 to production on [Date] at [Time UTC].

Release Highlights:
- [Feature 1]
- [Feature 2]
- [Bug Fix 1]

Deployment Window: [Start Time] - [End Time] (approximately 2 hours)
Expected Downtime: None (zero-downtime deployment)
Deployment Method: Blue-green deployment with gradual rollout

Deployment Team:
- Deployment Lead: [Name]
- On-Call Engineer: [Name]
- DBA: [Name]
- Incident Commander: [Name]

Communication Channels:
- War Room: #deployment-war-room
- Video Bridge: [Zoom Link]
- Status Updates: Every 30 minutes

Questions? Reply to this thread.

Thanks,
[Your Name]
```

#### Customer Communication
- [ ] Maintenance window communicated (if downtime expected)
- [ ] Status page updated with schedule
- [ ] Customer email sent (for major releases)
- [ ] Social media posts prepared
- [ ] Support team briefed on changes

```
Subject: Upcoming GreenLang Release - New Features & Improvements

Dear GreenLang Customer,

We are excited to announce the upcoming release of GreenLang v1.2.3 on [Date].

New Features:
- [Feature 1 with user benefit]
- [Feature 2 with user benefit]

Improvements:
- [Performance improvement]
- [Bug fix]

Deployment Schedule: [Date] [Time UTC]
Expected Downtime: None
Action Required: None (automatic update)

For detailed release notes, visit:
https://docs.greenlang.io/releases/v1.2.3

If you have any questions, contact support@greenlang.io

Best regards,
GreenLang Team
```

#### Vendor Notification
- [ ] Cloud provider notified (if significant resource changes)
- [ ] Third-party API providers notified (if traffic increase expected)
- [ ] CDN provider notified (if changes to CDN config)

#### On-Call Preparation
- [ ] On-call engineer identified and briefed
- [ ] Backup on-call engineer identified
- [ ] Escalation contacts confirmed
- [ ] War room details shared
- [ ] Runbooks reviewed

---

### 11. Deployment Prerequisites (T-1 day)

#### Final Code Freeze
- [ ] Code freeze in effect (no new merges)
- [ ] Release branch created and tagged
- [ ] Release notes finalized
- [ ] Version number confirmed

```bash
# Create release branch
git checkout -b release/v1.2.3 develop
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin release/v1.2.3 --tags
```

#### Build & Artifacts
- [ ] Container images built
- [ ] Container images tagged correctly
- [ ] Container images pushed to registry
- [ ] Container images scanned (no vulnerabilities)
- [ ] Artifacts checksums verified

```bash
# Build container image
docker build -t greenlang/api:v1.2.3 .

# Scan image
trivy image greenlang/api:v1.2.3

# Push to registry
docker push greenlang/api:v1.2.3

# Verify image
docker pull greenlang/api:v1.2.3
docker inspect greenlang/api:v1.2.3
```

#### Rollback Plan
- [ ] Rollback procedure documented
- [ ] Rollback tested in staging
- [ ] Previous version artifacts available
- [ ] Rollback trigger conditions defined
- [ ] Rollback decision-makers identified

```bash
# Verify previous version available
docker pull greenlang/api:v1.2.2

# Test rollback in staging
kubectl set image deployment/greenlang-api \
  api=greenlang/api:v1.2.2 -n greenlang-staging
```

#### Backup Verification
- [ ] Database backup completed
- [ ] Database backup verified (restoration tested)
- [ ] Configuration backup completed
- [ ] Container registry backup verified
- [ ] Documentation backup completed

```bash
# Create final pre-deployment backup
pg_dump -h db.greenlang.io -U admin greenlang \
  -F c -f backup/final-pre-deployment-$(date +%Y%m%d-%H%M%S).dump

# Verify backup size and integrity
ls -lh backup/final-pre-deployment-*.dump
pg_restore --list backup/final-pre-deployment-*.dump | wc -l
```

---

## Deployment Day Preparation

### T-0 Day (Morning)

#### Pre-Deployment Meeting (2 hours before)
- [ ] All deployment team members present
- [ ] Deployment plan reviewed
- [ ] Role assignments confirmed
- [ ] Communication channels tested
- [ ] Go/No-Go decision made

**Go/No-Go Criteria:**
- [ ] All pre-deployment checklist items completed
- [ ] No P0/P1 incidents in past 48 hours
- [ ] All team members available
- [ ] Rollback plan ready
- [ ] Monitoring operational

#### Final Validation (1 hour before)
- [ ] Production environment health verified
- [ ] Monitoring dashboards operational
- [ ] Alert channels operational
- [ ] Backup completed and verified
- [ ] Deployment artifacts ready

```bash
# Verify production health
curl https://api.greenlang.io/health
kubectl get pods -A | grep -v Running
kubectl top nodes

# Verify monitoring
curl http://prometheus.greenlang.io/-/healthy
curl http://grafana.greenlang.io/api/health

# Verify backup
ls -lh backup/final-pre-deployment-*.dump
```

---

## Deployment Execution

### Method 1: Blue-Green Deployment (Recommended)

**Characteristics:**
- Zero downtime
- Instant rollback capability
- Full validation before cutover
- Higher resource usage (2x during deployment)

**Procedure:**

#### Step 1: Deploy Green Environment (30 minutes)
```bash
# Deploy new version (green) alongside existing (blue)
kubectl apply -f k8s/deployments/greenlang-api-green.yaml

# Wait for all pods ready
kubectl wait --for=condition=ready pod \
  -l app=greenlang-api,version=green \
  --timeout=600s

# Verify pod health
kubectl get pods -l app=greenlang-api,version=green
```

#### Step 2: Smoke Test Green Environment (15 minutes)
```bash
# Run smoke tests against green environment
pytest tests/smoke/ --base-url=http://greenlang-api-green.greenlang.svc.cluster.local -v

# Manual validation
curl http://greenlang-api-green.greenlang.svc.cluster.local/health
curl http://greenlang-api-green.greenlang.svc.cluster.local/v1/agents

# Check metrics
curl http://greenlang-api-green.greenlang.svc.cluster.local/metrics
```

**Green Environment Validation Checklist:**
- [ ] All pods running and ready
- [ ] Health checks passing
- [ ] Smoke tests passing
- [ ] Database connectivity verified
- [ ] External API connectivity verified
- [ ] Metrics being collected
- [ ] Logs being sent to aggregation

#### Step 3: Gradual Traffic Shift (2 hours)
```bash
# Shift 10% traffic to green
kubectl patch service greenlang-api -p '
{
  "spec": {
    "selector": {
      "app": "greenlang-api"
    }
  }
}'
kubectl apply -f k8s/canary/10-percent-green.yaml

# Wait and monitor for 15 minutes
# Check metrics:
# - Error rate < 1%
# - Latency p95 < 500ms
# - No increase in errors
```

**10% Traffic Validation Checklist:**
- [ ] Error rate stable (< 1%)
- [ ] Latency within normal range (p95 < 500ms)
- [ ] No new error types appearing
- [ ] Logs look normal
- [ ] No customer reports
- [ ] Metrics stable for 15 minutes

```bash
# If validation passes, increase to 25%
kubectl apply -f k8s/canary/25-percent-green.yaml

# Wait and monitor for 15 minutes
# Repeat validation checklist
```

**25% Traffic Validation Checklist:**
- [ ] Error rate stable (< 1%)
- [ ] Latency within normal range
- [ ] No new error types
- [ ] Logs normal
- [ ] No customer reports
- [ ] Metrics stable for 15 minutes

```bash
# If validation passes, increase to 50%
kubectl apply -f k8s/canary/50-percent-green.yaml

# Wait and monitor for 30 minutes
# Repeat validation checklist
```

**50% Traffic Validation Checklist:**
- [ ] Error rate stable (< 1%)
- [ ] Latency within normal range
- [ ] No new error types
- [ ] Logs normal
- [ ] No customer reports
- [ ] Metrics stable for 30 minutes

```bash
# If validation passes, increase to 100%
kubectl patch service greenlang-api -p '
{
  "spec": {
    "selector": {
      "app": "greenlang-api",
      "version": "green"
    }
  }
}'

# Wait and monitor for 30 minutes
# Repeat validation checklist
```

**100% Traffic Validation Checklist:**
- [ ] Error rate stable (< 1%)
- [ ] Latency within normal range
- [ ] No new error types
- [ ] Logs normal
- [ ] No customer reports
- [ ] Metrics stable for 30 minutes

#### Step 4: Remove Blue Environment (after 24 hours of stability)
```bash
# Only after 24 hours of stable operation on green
kubectl scale deployment greenlang-api-blue --replicas=0

# After another 24 hours of stability
kubectl delete -f k8s/deployments/greenlang-api-blue.yaml
```

---

### Method 2: Rolling Update

**Characteristics:**
- Zero downtime
- Gradual rollout
- Lower resource usage
- Slower rollback

**Procedure:**

```bash
# Update deployment with new image
kubectl set image deployment/greenlang-api \
  api=greenlang/api:v1.2.3

# Monitor rollout
kubectl rollout status deployment/greenlang-api

# Pause rollout if issues detected
kubectl rollout pause deployment/greenlang-api

# Resume if healthy
kubectl rollout resume deployment/greenlang-api
```

**Rolling Update Validation (after each pod):**
- [ ] New pod started successfully
- [ ] Health checks passing
- [ ] Metrics normal
- [ ] No errors in logs

---

### Method 3: Canary Deployment

**Characteristics:**
- Lowest risk
- Subset of users on new version
- Longer deployment time
- More complex

**Procedure:**

```bash
# Deploy canary (5% of pods)
kubectl apply -f k8s/canary/greenlang-api-canary.yaml

# Monitor canary for 1 hour
# If healthy, increase to 10%, 25%, 50%, 100%

# Promote canary to production
kubectl apply -f k8s/deployments/greenlang-api-production.yaml
```

---

## Post-Deployment Validation

### Immediate Validation (T+0 minutes)

#### Health Checks
```bash
# API health check
curl https://api.greenlang.io/health
# Expected: {"status": "healthy", "version": "1.2.3"}

# Database connectivity
curl https://api.greenlang.io/internal/db-health
# Expected: {"status": "healthy"}

# External API connectivity
curl https://api.greenlang.io/internal/external-api-health
# Expected: {"status": "healthy", "providers": ["openai": "healthy", "anthropic": "healthy"]}
```

**Health Check Validation Checklist:**
- [ ] API responding to requests
- [ ] Correct version deployed
- [ ] Database connectivity verified
- [ ] External APIs reachable
- [ ] All services reporting healthy

#### Smoke Tests
```bash
# Run automated smoke tests
pytest tests/smoke/ --env=production -v

# Expected: All tests passing
```

**Smoke Test Checklist:**
- [ ] Authentication working
- [ ] Agent listing working
- [ ] Agent execution working
- [ ] Configuration retrieval working
- [ ] All critical endpoints responding

#### Metrics Verification
```bash
# Check error rate
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=rate(gl_errors_total[5m]) / rate(gl_requests_total[5m])'
# Expected: < 0.01 (1%)

# Check latency
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))'
# Expected: < 0.5 (500ms)

# Check request rate
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=rate(gl_requests_total[5m])'
# Expected: Normal traffic levels
```

**Metrics Validation Checklist:**
- [ ] Error rate < 1%
- [ ] p95 latency < 500ms
- [ ] p99 latency < 1000ms
- [ ] Request rate normal
- [ ] Success rate > 99%

---

### Short-Term Validation (T+1 hour)

#### Continuous Monitoring
- [ ] Error rate stable and < 1%
- [ ] Latency within normal range
- [ ] No unusual log patterns
- [ ] No alerts firing
- [ ] Customer reports normal

#### Log Analysis
```bash
# Check for new error types
kubectl logs -l app=greenlang-api --since=1h | grep ERROR | sort | uniq -c

# Check for warnings
kubectl logs -l app=greenlang-api --since=1h | grep WARN | sort | uniq -c

# Should see no new error types or concerning warnings
```

#### Database Verification
```bash
# Check database performance
psql -h db.greenlang.io -c "
  SELECT count(*), state FROM pg_stat_activity GROUP BY state;
"
# Should show normal connection distribution

# Check for slow queries
psql -h db.greenlang.io -c "
  SELECT pid, now() - query_start AS duration, query
  FROM pg_stat_activity
  WHERE state = 'active' AND now() - query_start > interval '5 seconds'
  ORDER BY duration DESC;
"
# Should return no results or only expected long-running queries
```

---

### Medium-Term Validation (T+4 hours)

#### Performance Trending
- [ ] Latency trending normal (no degradation)
- [ ] Throughput at expected levels
- [ ] Resource usage stable (CPU, memory)
- [ ] Database performance stable
- [ ] Cache hit rates normal

#### Customer Feedback
- [ ] No unusual support tickets
- [ ] No social media complaints
- [ ] No reports of issues from customer success
- [ ] Usage patterns normal

#### Integration Verification
- [ ] All external integrations working
- [ ] Webhook deliveries succeeding
- [ ] Background jobs processing
- [ ] Scheduled tasks executing

---

### Long-Term Validation (T+24 hours)

#### Full Regression Testing
```bash
# Run full test suite against production (non-destructive tests only)
pytest tests/regression/ --env=production --markers="production-safe" -v
```

#### Performance Benchmarking
```bash
# Run performance benchmarks
./scripts/run-performance-benchmarks.sh production

# Compare to baseline
./scripts/compare-performance.sh current baseline
```

#### Cost Analysis
- [ ] Infrastructure costs within expected range
- [ ] API usage costs within expected range
- [ ] No unexpected cost spikes
- [ ] Resource utilization optimal

#### Stability Verification
- [ ] No P0/P1 incidents
- [ ] Error rate consistently < 0.5%
- [ ] Latency consistently < 400ms (p95)
- [ ] Uptime 100%
- [ ] All alerts resolved

---

## Rollback Procedures

### Rollback Decision Criteria

**Immediate Rollback (within 5 minutes):**
- Error rate > 5%
- P0 incident triggered
- Data corruption detected
- Critical security vulnerability
- Complete service outage

**Planned Rollback (within 30 minutes):**
- Error rate > 2% sustained for 15 minutes
- P1 incident with no quick fix
- Performance degradation > 50%
- Customer-reported critical issues

### Rollback Procedure (Blue-Green)

**Fastest Rollback (2 minutes):**
```bash
# Instantly switch back to blue
kubectl patch service greenlang-api -p '
{
  "spec": {
    "selector": {
      "app": "greenlang-api",
      "version": "blue"
    }
  }
}'

# Verify traffic shifted
curl https://api.greenlang.io/version
# Should show old version

# Monitor for immediate improvement
# Watch error rate and latency
```

**Rollback Validation Checklist:**
- [ ] Service responding with old version
- [ ] Error rate decreasing
- [ ] Latency improving
- [ ] Customer reports decreasing
- [ ] System stabilizing

### Rollback Procedure (Rolling Update)

```bash
# Rollback to previous version
kubectl rollout undo deployment/greenlang-api

# Monitor rollback
kubectl rollout status deployment/greenlang-api

# Verify rollback
kubectl get pods -l app=greenlang-api -o wide
```

### Post-Rollback Actions

1. **Communication**
   ```bash
   # Update status page
   curl -X POST https://status.greenlang.io/api/incidents \
     -H "Authorization: Bearer $STATUS_API_KEY" \
     -d '{
       "status": "resolved",
       "message": "Deployment has been rolled back. Service is operating normally."
     }'
   ```

2. **Investigation**
   - Preserve logs from failed deployment
   - Capture metrics/traces
   - Document what went wrong
   - Schedule post-mortem

3. **Prevention**
   - Identify root cause
   - Add tests to catch issue
   - Update deployment checklist
   - Update monitoring/alerting

---

## Deployment Methods Comparison

| Aspect | Blue-Green | Rolling Update | Canary |
|--------|-----------|----------------|---------|
| **Downtime** | Zero | Zero | Zero |
| **Rollback Speed** | Instant | Moderate | Slow |
| **Resource Usage** | 2x (temporary) | 1.5x (temporary) | 1.1x |
| **Risk** | Low | Medium | Lowest |
| **Complexity** | Medium | Low | High |
| **Validation Time** | Fast | Moderate | Slow |
| **Best For** | Production critical | Standard releases | High-risk changes |

**Recommendation:** Use blue-green for production deployments

---

## Appendix A: Deployment Roles & Responsibilities

### Deployment Lead
- Coordinates entire deployment
- Makes go/no-go decisions
- Authorizes rollback if needed
- Primary decision-maker

### Technical Lead
- Executes deployment commands
- Monitors technical metrics
- Troubleshoots issues
- Recommends technical decisions

### Database Administrator
- Executes database migrations
- Monitors database performance
- Handles database rollback if needed

### Communications Lead
- Updates status page
- Sends customer communications
- Coordinates with support team
- Manages stakeholder communications

### On-Call Engineer
- Monitors alerts
- Responds to incidents
- Escalates issues
- Documents problems

---

## Appendix B: Communication Templates

### Deployment Start
```
Subject: Production Deployment Started - GreenLang v1.2.3

Team,

Production deployment has started at [TIME UTC].

Status: Green environment deployed, starting traffic shift
Next Update: [TIME + 30 minutes]

Dashboard: https://grafana.greenlang.io/d/deployment
War Room: #deployment-war-room

[Name], Deployment Lead
```

### Deployment Progress
```
Subject: Deployment Progress Update - 50% traffic shifted

Team,

Deployment Status Update - [TIME UTC]

Current Status: 50% traffic on new version
Metrics: All green
- Error rate: 0.3%
- Latency p95: 285ms
- No alerts

Next Step: Increase to 100% in 30 minutes
Next Update: [TIME + 30 minutes]

[Name], Deployment Lead
```

### Deployment Complete
```
Subject: Production Deployment Complete - GreenLang v1.2.3

Team,

Production deployment completed successfully at [TIME UTC].

Final Status:
- 100% traffic on v1.2.3
- All metrics normal
- No incidents
- Deployment duration: [X hours Y minutes]

Deployed Features:
- [Feature 1]
- [Feature 2]

Validation Period: Next 24 hours
War Room: Closed, monitoring continues

Thank you for your support!

[Name], Deployment Lead
```

### Deployment Rollback
```
Subject: URGENT - Production Deployment Rolled Back

Team,

The production deployment has been rolled back at [TIME UTC].

Reason: [Brief explanation]
Current Status: Running v1.2.2 (previous version)
Service Status: Stable

Incident Ticket: INC-12345
Post-Mortem: Scheduled for [DATE]

The issue will be investigated and fixed before retry.

[Name], Deployment Lead
```

---

## Appendix C: Quick Reference

### Essential Commands

```bash
# Check deployment status
kubectl get deployments
kubectl get pods
kubectl rollout status deployment/greenlang-api

# Check service health
curl https://api.greenlang.io/health

# Check metrics
curl http://prometheus.greenlang.io/api/v1/query?query=gl_error_rate

# Rollback
kubectl rollout undo deployment/greenlang-api

# Scale
kubectl scale deployment/greenlang-api --replicas=10
```

### Essential Dashboards

- System Overview: https://grafana.greenlang.io/d/system-overview
- API Performance: https://grafana.greenlang.io/d/api-performance
- Error Tracking: https://grafana.greenlang.io/d/errors
- Database Performance: https://grafana.greenlang.io/d/database
- Deployment Dashboard: https://grafana.greenlang.io/d/deployment

### Emergency Contacts

- On-Call Engineer: PagerDuty
- Deployment Lead: [Phone]
- Incident Commander: [Phone]
- CTO: [Phone]

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | Operations Team | Initial comprehensive checklist |

**Next Review Date:** 2025-12-07
**Approved By:** [CTO], [Operations Lead], [Engineering Lead]

---

**Remember:**
- When in doubt, don't deploy
- It's okay to abort a deployment
- Safety over speed
- Document everything
- Learn from every deployment

**You've got this!**
