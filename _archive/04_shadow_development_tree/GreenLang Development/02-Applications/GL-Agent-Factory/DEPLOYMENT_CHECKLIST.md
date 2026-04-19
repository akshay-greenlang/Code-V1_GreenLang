# GreenLang Agent Factory - Deployment Checklist

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Production Ready
**Environment:** Kubernetes (Local/Cloud)

---

## Table of Contents

1. [Pre-Deployment Verification](#pre-deployment-verification)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Database Deployment](#database-deployment)
4. [Agent Deployment](#agent-deployment)
5. [Monitoring Setup](#monitoring-setup)
6. [Post-Deployment Validation](#post-deployment-validation)
7. [Smoke Tests](#smoke-tests)
8. [Rollback Procedures](#rollback-procedures)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Pre-Deployment Verification

### System Requirements

- [ ] **Kubernetes Cluster**
  - [ ] Cluster version: 1.24+ verified
  - [ ] kubectl configured and connected
  - [ ] Sufficient resources available:
    - CPU: 8+ cores available
    - Memory: 16GB+ available
    - Storage: 50GB+ available

- [ ] **Required Tools Installed**
  - [ ] kubectl (v1.24+)
  - [ ] helm (v3.10+)
  - [ ] docker (v20.10+)
  - [ ] psql (PostgreSQL client)
  - [ ] redis-cli (Redis client)

- [ ] **Network Requirements**
  - [ ] Internet connectivity for image pulls
  - [ ] DNS resolution working
  - [ ] Port availability checked (5432, 6379, 9090, 3000, 8000-8004)

### Code Verification

- [ ] **Repository Status**
  - [ ] Latest code pulled from main/master branch
  - [ ] All tests passing: `pytest`
  - [ ] No uncommitted changes in production code
  - [ ] Version tags applied correctly

- [ ] **Environment Configuration**
  - [ ] `.env` files reviewed and validated
  - [ ] Secrets created in Kubernetes
  - [ ] API keys configured (OpenAI, Anthropic, etc.)
  - [ ] Database credentials secure

### Documentation Review

- [ ] **Deployment Documentation**
  - [ ] Read GL-Agent-Factory/00-README.md
  - [ ] Review k8s/README.md
  - [ ] Check k8s/monitoring/README.md
  - [ ] Understand rollback procedures

---

## Infrastructure Setup

### Namespace Creation

```bash
# Create namespaces
kubectl apply -f k8s/00-namespaces.yaml

# Verify namespaces
kubectl get namespaces | grep -E "greenlang|gl-|monitoring"
```

- [ ] Namespace `greenlang` created
- [ ] Namespace `gl-cbam` created
- [ ] Namespace `gl-fuel` created
- [ ] Namespace `gl-building` created
- [ ] Namespace `gl-eudr` created
- [ ] Namespace `monitoring` created

### ConfigMaps and Secrets

```bash
# Create ConfigMaps
kubectl apply -f k8s/configmaps/

# Create Secrets (use secure method)
kubectl create secret generic openai-api-key \
  --from-literal=api-key=YOUR_KEY_HERE \
  -n greenlang

kubectl create secret generic anthropic-api-key \
  --from-literal=api-key=YOUR_KEY_HERE \
  -n greenlang
```

- [ ] All ConfigMaps created
- [ ] API key secrets created
- [ ] Database secrets created
- [ ] Verify secrets: `kubectl get secrets -n greenlang`

---

## Database Deployment

### PostgreSQL Deployment

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres/

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n greenlang --timeout=300s

# Verify PostgreSQL
kubectl get pods -n greenlang -l app=postgres
kubectl logs -n greenlang -l app=postgres --tail=50
```

- [ ] PostgreSQL pod running
- [ ] PostgreSQL logs show no errors
- [ ] Database connection test successful

### Database Initialization

```bash
# Port forward to PostgreSQL
kubectl port-forward -n greenlang svc/postgres 5432:5432 &

# Run migrations
cd sdks/python/greenlang_sdk
python -m alembic upgrade head

# Verify tables created
psql -h localhost -U greenlang -d greenlang -c "\dt"
```

- [ ] All migrations applied successfully
- [ ] Required tables created:
  - [ ] agent_runs
  - [ ] calculation_records
  - [ ] cache_entries
  - [ ] audit_logs
- [ ] Sample data loaded (if applicable)

### Redis Deployment

```bash
# Deploy Redis
kubectl apply -f k8s/redis/

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app=redis -n greenlang --timeout=300s

# Test Redis connection
kubectl port-forward -n greenlang svc/redis 6379:6379 &
redis-cli ping
```

- [ ] Redis pod running
- [ ] Redis responds to PING with PONG
- [ ] Redis persistence configured

---

## Agent Deployment

### CBAM Importer Copilot (Tier 2)

```bash
# Deploy CBAM agent
kubectl apply -f k8s/deployments/cbam/

# Wait for deployment
kubectl rollout status deployment/cbam-api -n gl-cbam

# Check pod status
kubectl get pods -n gl-cbam
kubectl logs -n gl-cbam -l app=cbam-api --tail=50
```

- [ ] CBAM deployment successful
- [ ] Pod status: Running
- [ ] Health check passing: `curl http://localhost:8000/health`
- [ ] Metrics endpoint available: `curl http://localhost:8000/metrics`

### Fuel Analyzer Agent

```bash
# Deploy Fuel agent
kubectl apply -f GL-014/k8s/

# Check deployment
kubectl get pods -n gl-fuel
kubectl logs -n gl-fuel -l app=fuel-analyzer-agent --tail=50
```

- [ ] Fuel analyzer deployment successful
- [ ] Pod status: Running
- [ ] Health check passing
- [ ] Metrics endpoint available

### Building Energy Agent

```bash
# Deploy Building Energy agent
kubectl apply -f GL-015/k8s/

# Check deployment
kubectl get pods -n gl-building
kubectl logs -n gl-building -l app=building-energy-agent --tail=50
```

- [ ] Building energy deployment successful
- [ ] Pod status: Running
- [ ] Health check passing
- [ ] Metrics endpoint available

### EUDR Compliance Agent (Tier 1 - CRITICAL)

```bash
# Deploy EUDR agent
kubectl apply -f GL-016/k8s/

# Check deployment - CRITICAL
kubectl get pods -n gl-eudr
kubectl logs -n gl-eudr -l app=eudr-api --tail=100

# Verify EUDR-specific resources
kubectl get all -n gl-eudr
```

- [ ] EUDR deployment successful
- [ ] Pod status: Running (CRITICAL)
- [ ] Health check passing
- [ ] Metrics endpoint available
- [ ] All 5 tools initialized:
  - [ ] calculate_country_risk
  - [ ] classify_commodity
  - [ ] assess_deforestation_risk
  - [ ] generate_due_diligence
  - [ ] validate_documentation
- [ ] Database connectivity verified
- [ ] LLM API connectivity verified

### Agent Verification Summary

```bash
# Check all agents
kubectl get pods -A | grep -E "cbam|fuel|building|eudr"

# Check all services
kubectl get svc -A | grep -E "cbam|fuel|building|eudr"
```

- [ ] All agent pods running
- [ ] All services exposed correctly
- [ ] No CrashLoopBackOff errors
- [ ] Resource limits appropriate

---

## Monitoring Setup

### Prometheus Deployment

```bash
# Install Prometheus using Helm
cd k8s/monitoring
./install.sh

# Or manual installation
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  --create-namespace \
  -f prometheus-values.yaml

# Wait for Prometheus
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=300s
```

- [ ] Prometheus operator installed
- [ ] Prometheus server running
- [ ] Prometheus accessible: `kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090`

### ServiceMonitors Deployment

```bash
# Deploy ServiceMonitors
kubectl apply -f k8s/monitoring/servicemonitor-cbam.yaml
kubectl apply -f k8s/monitoring/servicemonitor-fuel-analyzer.yaml
kubectl apply -f k8s/monitoring/servicemonitor-building-energy.yaml
kubectl apply -f k8s/monitoring/servicemonitor-eudr-compliance.yaml

# Verify ServiceMonitors
kubectl get servicemonitors -n monitoring
```

- [ ] All 4 ServiceMonitors created
- [ ] ServiceMonitors discovered by Prometheus
- [ ] Targets showing in Prometheus UI (Status > Targets)

### PrometheusRules Deployment

```bash
# Deploy alerting rules
kubectl apply -f k8s/monitoring/prometheus-rules.yaml

# Verify rules loaded
kubectl get prometheusrules -n monitoring
```

- [ ] PrometheusRule created
- [ ] All alert groups loaded:
  - [ ] greenlang.agent.errors
  - [ ] greenlang.agent.latency
  - [ ] greenlang.agent.availability
  - [ ] greenlang.agent.cache
  - [ ] greenlang.agent.calculations
  - [ ] greenlang.agent.resources
  - [ ] greenlang.infrastructure
  - [ ] greenlang.eudr.compliance (CRITICAL)
- [ ] Rules showing in Prometheus UI (Status > Rules)

### Grafana Setup

```bash
# Grafana should be installed with Prometheus stack
kubectl get pods -n monitoring | grep grafana

# Get Grafana admin password
kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

- [ ] Grafana accessible at http://localhost:3000
- [ ] Login successful (admin/password)
- [ ] Prometheus datasource configured

### Grafana Dashboards

```bash
# Import dashboards
# Option 1: Via ConfigMap (automated)
kubectl apply -f k8s/monitoring/prometheus-rules.yaml  # Contains dashboard ConfigMap

# Option 2: Manual import via UI
# - Go to Grafana UI
# - Dashboards > Import
# - Upload JSON files from k8s/monitoring/dashboards/
```

- [ ] Dashboard: Agent Factory Overview imported
- [ ] Dashboard: Agent Health imported
- [ ] Dashboard: Infrastructure imported
- [ ] Dashboard: EUDR Compliance Agent imported (CRITICAL)
- [ ] All dashboards showing data
- [ ] No "No Data" errors

---

## Post-Deployment Validation

### Health Checks

```bash
# Check all agent health endpoints
curl http://localhost:8000/health  # CBAM
curl http://localhost:8001/health  # Fuel
curl http://localhost:8002/health  # Building
curl http://localhost:8003/health  # EUDR (CRITICAL)
```

- [ ] All health endpoints returning 200 OK
- [ ] Response format correct
- [ ] Database connectivity confirmed in health response

### Metrics Verification

```bash
# Check metrics endpoints
curl http://localhost:8000/metrics | grep agent_requests_total
curl http://localhost:8001/metrics | grep agent_requests_total
curl http://localhost:8002/metrics | grep agent_requests_total
curl http://localhost:8003/metrics | grep agent_requests_total
```

- [ ] All metrics endpoints accessible
- [ ] Metrics format correct (Prometheus format)
- [ ] Custom metrics present:
  - [ ] agent_requests_total
  - [ ] agent_request_duration_seconds
  - [ ] agent_calculations_total
  - [ ] agent_cache_hits_total
  - [ ] agent_cache_misses_total

### Prometheus Targets

1. Open Prometheus UI: http://localhost:9090
2. Go to Status > Targets
3. Verify all targets are UP:

- [ ] cbam-importer-agent (UP)
- [ ] fuel-analyzer-agent (UP)
- [ ] building-energy-agent (UP)
- [ ] eudr-compliance-agent (UP - CRITICAL)

### Grafana Dashboards

1. Open Grafana: http://localhost:3000
2. Navigate to each dashboard:

- [ ] **Agent Factory Overview**
  - [ ] Total agents count correct
  - [ ] Request rates showing
  - [ ] Error rates visible
  - [ ] No "No Data" panels

- [ ] **Agent Health**
  - [ ] All agents showing as healthy
  - [ ] Per-agent metrics visible
  - [ ] Tool execution data present

- [ ] **EUDR Compliance Agent (CRITICAL)**
  - [ ] Deadline countdown showing correct days
  - [ ] Request rate > 0
  - [ ] Error rate < 0.5%
  - [ ] P95 latency < 300ms
  - [ ] All 5 tools showing execution data
  - [ ] Country risk distribution showing
  - [ ] Commodity classification data present

---

## Smoke Tests

### Test 1: CBAM Importer Basic Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Calculate embedded emissions for steel imported from Germany, produced using electric arc furnace with 1500 kg CO2/tonne"
      }
    ]
  }'
```

- [ ] Request successful (200 OK)
- [ ] Response contains calculation
- [ ] Provenance data included
- [ ] Metrics incremented in Prometheus

### Test 2: Fuel Analyzer Request

```bash
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Calculate emissions for 1000 liters of diesel fuel"
      }
    ]
  }'
```

- [ ] Request successful
- [ ] Emissions calculated correctly
- [ ] Tool execution logged

### Test 3: Building Energy Request

```bash
curl -X POST http://localhost:8002/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Calculate energy consumption for an office building: 5000 sqm, 200 occupants, 8 hours/day"
      }
    ]
  }'
```

- [ ] Request successful
- [ ] Energy profile calculated
- [ ] Recommendations provided

### Test 4: EUDR Compliance Request (CRITICAL)

```bash
curl -X POST http://localhost:8003/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Generate EUDR due diligence statement for coffee imported from Brazil, coordinates -15.7942,-47.8822, 1000kg, operator: ABC Trading Ltd"
      }
    ]
  }'
```

- [ ] Request successful (CRITICAL)
- [ ] All 5 tools executed:
  - [ ] Country risk calculated
  - [ ] Commodity classified correctly
  - [ ] Deforestation risk assessed
  - [ ] Due diligence statement generated
  - [ ] Documentation validated
- [ ] Response time < 2 seconds
- [ ] Due diligence statement complete and valid
- [ ] Provenance chain complete

### Test 5: Error Handling

```bash
# Test invalid request
curl -X POST http://localhost:8003/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Invalid coordinates: 999,999"
      }
    ]
  }'
```

- [ ] Error handled gracefully
- [ ] Error metrics incremented
- [ ] Appropriate error message returned
- [ ] No pod crashes

### Test 6: Load Test (Optional but Recommended)

```bash
# Simple load test using hey or ab
hey -n 100 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d @test_request.json \
  http://localhost:8003/api/v1/chat
```

- [ ] All requests completed successfully
- [ ] No timeout errors
- [ ] Error rate < 1%
- [ ] Average latency acceptable
- [ ] Pods scaled if needed (HPA)

---

## Rollback Procedures

### Identify Issues

**Symptoms requiring rollback:**
- Pod crash loops
- Error rate > 10%
- Health checks failing
- Database connection errors
- Critical EUDR agent failures

### Immediate Actions

1. **Stop Incoming Traffic (if applicable)**
   ```bash
   kubectl scale deployment/[agent-name] -n [namespace] --replicas=0
   ```

2. **Capture Current State**
   ```bash
   # Save pod logs
   kubectl logs -n [namespace] -l app=[agent-name] > rollback-logs.txt

   # Save events
   kubectl get events -n [namespace] --sort-by='.lastTimestamp' > rollback-events.txt

   # Save pod descriptions
   kubectl describe pods -n [namespace] > rollback-pods.txt
   ```

### Rollback Options

#### Option 1: Rollback Deployment

```bash
# Check deployment history
kubectl rollout history deployment/[agent-name] -n [namespace]

# Rollback to previous version
kubectl rollout undo deployment/[agent-name] -n [namespace]

# Rollback to specific revision
kubectl rollout undo deployment/[agent-name] -n [namespace] --to-revision=2

# Verify rollback
kubectl rollout status deployment/[agent-name] -n [namespace]
```

#### Option 2: Rollback via Git

```bash
# Revert to previous commit
git revert HEAD
git push origin main

# Redeploy
kubectl apply -f k8s/deployments/[agent]/
```

#### Option 3: Complete Teardown

```bash
# Delete specific agent
kubectl delete -f k8s/deployments/[agent]/

# Delete namespace (nuclear option)
kubectl delete namespace [namespace]

# Wait and verify
kubectl get all -n [namespace]
```

### Post-Rollback Verification

- [ ] Pods running after rollback
- [ ] Health checks passing
- [ ] Error rate back to normal
- [ ] Smoke tests passing
- [ ] Metrics collection resumed

### Root Cause Analysis

After successful rollback, investigate:

1. **Check Logs**
   ```bash
   kubectl logs -n [namespace] -l app=[agent-name] --previous
   ```

2. **Check Resource Issues**
   ```bash
   kubectl top pods -n [namespace]
   kubectl describe pod [pod-name] -n [namespace]
   ```

3. **Check Configuration**
   - Review ConfigMaps
   - Verify Secrets
   - Check environment variables

4. **Database State**
   ```bash
   psql -h localhost -U greenlang -d greenlang -c "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT 10;"
   ```

---

## Troubleshooting Guide

### Common Issues

#### Issue: Pod CrashLoopBackOff

**Symptoms:**
```bash
kubectl get pods -n [namespace]
NAME                        READY   STATUS             RESTARTS
agent-xyz-12345-abcde       0/1     CrashLoopBackOff   5
```

**Diagnosis:**
```bash
kubectl logs -n [namespace] agent-xyz-12345-abcde
kubectl describe pod -n [namespace] agent-xyz-12345-abcde
```

**Common Causes & Solutions:**
1. **Missing environment variables**
   - Check ConfigMaps and Secrets
   - Verify env vars in deployment YAML

2. **Database connection failure**
   - Verify PostgreSQL is running
   - Check connection string
   - Test connectivity: `psql -h postgres -U greenlang`

3. **Missing API keys**
   - Verify secrets created
   - Check secret names match deployment

4. **Out of Memory**
   - Check memory limits
   - Increase limits if needed
   - Review memory usage patterns

#### Issue: Service Unavailable (503)

**Diagnosis:**
```bash
kubectl get svc -n [namespace]
kubectl get endpoints -n [namespace]
```

**Solutions:**
- Verify service selector matches pod labels
- Check pod is running and ready
- Verify port configuration

#### Issue: Metrics Not Showing in Prometheus

**Diagnosis:**
1. Check ServiceMonitor:
   ```bash
   kubectl get servicemonitor -n monitoring
   kubectl describe servicemonitor [name] -n monitoring
   ```

2. Check Prometheus targets:
   - Open http://localhost:9090/targets
   - Look for DOWN targets

**Solutions:**
- Verify ServiceMonitor selector matches service labels
- Check namespace selector
- Verify metrics endpoint: `curl http://[pod-ip]:8000/metrics`
- Check Prometheus operator logs

#### Issue: EUDR Agent High Error Rate (CRITICAL)

**Diagnosis:**
```bash
# Check error rate
kubectl logs -n gl-eudr -l app=eudr-api --tail=100 | grep ERROR

# Check specific tool failures
curl http://localhost:8003/metrics | grep agent_calculations_total
```

**Solutions:**
1. **Tool-specific failures:**
   - `calculate_country_risk`: Check country database connectivity
   - `classify_commodity`: Verify commodity DB and CN codes
   - `assess_deforestation_risk`: Check geospatial API availability
   - `generate_due_diligence`: Verify template availability
   - `validate_documentation`: Check validation rules

2. **Database issues:**
   ```bash
   kubectl logs -n greenlang -l app=postgres --tail=50
   ```

3. **LLM API issues:**
   - Check API key validity
   - Verify rate limits not exceeded
   - Test API connectivity manually

#### Issue: High Latency

**Diagnosis:**
```bash
# Check P95 latency
curl http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(agent_request_duration_seconds_bucket[5m]))by(le,agent_id))
```

**Solutions:**
- Check database query performance
- Verify cache hit rates
- Check LLM API response times
- Scale horizontally: `kubectl scale deployment/[agent] -n [namespace] --replicas=3`
- Review resource limits

#### Issue: Database Migration Failures

**Diagnosis:**
```bash
# Check migration status
python -m alembic current
python -m alembic history
```

**Solutions:**
```bash
# Rollback migration
python -m alembic downgrade -1

# Re-run migration
python -m alembic upgrade head

# Force upgrade (dangerous)
python -m alembic stamp head
```

### Monitoring Health

**Key Metrics to Watch:**

1. **Error Rate**
   - Threshold: < 1% (< 0.5% for EUDR)
   - Alert: EudrAgentHighErrorRate

2. **Latency**
   - P95 Threshold: < 500ms (< 300ms for EUDR)
   - Alert: EudrAgentHighLatency

3. **Availability**
   - Threshold: 99.9% uptime
   - Alert: AgentDown

4. **Resource Usage**
   - Memory: < 80%
   - CPU: < 80%

### Support Contacts

- **Platform Team:** platform@greenlang.ai
- **On-Call Engineer:** PagerDuty escalation
- **EUDR Compliance Lead:** eudr@greenlang.ai
- **Slack Channels:**
  - #greenlang-ops
  - #eudr-compliance-alerts
  - #incident-response

---

## Success Criteria

Deployment is considered successful when:

- [ ] All agents deployed and running
- [ ] All health checks passing
- [ ] Monitoring collecting metrics
- [ ] Grafana dashboards showing data
- [ ] Smoke tests passing
- [ ] Error rate < 1% (< 0.5% for EUDR)
- [ ] P95 latency < 500ms (< 300ms for EUDR)
- [ ] No critical alerts firing
- [ ] Database connectivity verified
- [ ] EUDR agent fully operational (CRITICAL)

**Special EUDR Success Criteria:**
- [ ] Deadline countdown showing correctly
- [ ] All 5 tools executing successfully
- [ ] Country risk assessments working
- [ ] Commodity classifications accurate
- [ ] Due diligence statements generating
- [ ] Validation passing
- [ ] Integration with EU DDS ready (if applicable)

---

## Next Steps After Deployment

1. **Monitor for 24 Hours**
   - Watch Grafana dashboards
   - Review Prometheus alerts
   - Check error logs

2. **Performance Tuning**
   - Adjust resource limits if needed
   - Optimize cache configuration
   - Fine-tune alert thresholds

3. **Documentation**
   - Update runbooks with actual issues encountered
   - Document any configuration changes
   - Update team knowledge base

4. **Stakeholder Communication**
   - Send deployment success notification
   - Schedule post-deployment review
   - Update status page

5. **EUDR-Specific Actions**
   - Brief compliance team on system status
   - Verify integration with importer systems
   - Schedule deadline readiness review
   - Prepare regulatory reporting

---

**Document Version:** 1.0.0
**Last Updated:** 2025-12-03
**Maintained By:** GreenLang Platform Team
**Review Frequency:** After each major deployment
