# GreenLang Intelligence Layer - Staging Deployment Guide

**Version:** 1.0
**Last Updated:** 2025-10-01
**Target Environment:** Staging
**Prerequisites:** Docker, Python 3.10+, API Keys for OpenAI/Anthropic

---

## Overview

This guide walks through deploying the GreenLang Intelligence Layer to a staging environment for testing before production release.

**Staging Purpose:**
- Validate system behavior under realistic conditions
- Test with real API credentials (with budget limits)
- Verify monitoring and alerting
- Conduct load testing
- Train team on operations

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  STAGING ENVIRONMENT                                       │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐        ┌──────────────┐                 │
│  │ GreenLang   │───────▶│ Intelligence │                 │
│  │ Application │        │ Layer        │                 │
│  └─────────────┘        └──────┬───────┘                 │
│                                │                          │
│                                ▼                          │
│                    ┌────────────────────┐                 │
│                    │  LLM Providers     │                 │
│                    │  - OpenAI (staging)│                 │
│                    │  - Anthropic (stag)│                 │
│                    └────────────────────┘                 │
│                                │                          │
│                                ▼                          │
│                    ┌────────────────────┐                 │
│                    │  Monitoring        │                 │
│                    │  - Metrics         │                 │
│                    │  - Dashboards      │                 │
│                    │  - Alerts          │                 │
│                    └────────────────────┘                 │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

---

## Pre-Deployment Checklist

### 1. Environment Preparation

- [ ] **Staging Server:** Provision staging server/container
  - Minimum: 4 CPU cores, 8GB RAM
  - OS: Ubuntu 20.04+ or compatible
  - Python: 3.9 or higher

- [ ] **API Keys:** Obtain staging-specific API keys
  - OpenAI: Create organization-specific key with budget limits
  - Anthropic: Create key with rate limits for staging
  - **NEVER use production keys in staging**

- [ ] **Dependencies:** Install required packages
  ```bash
  pip install greenlang-cli[analytics]==0.3.0
  pip install prometheus-client  # For metrics export
  ```

### 2. Configuration

- [ ] **Environment Variables:**
  ```bash
  # .env.staging
  OPENAI_API_KEY=sk-staging-...
  ANTHROPIC_API_KEY=sk-ant-staging-...

  GREENLANG_ENV=staging
  GREENLANG_LOG_LEVEL=INFO
  GREENLANG_BUDGET_MAX_USD=100.00  # Staging budget limit
  GREENLANG_BUDGET_ALERT_THRESHOLD=0.20  # Alert at 20% remaining

  # Circuit Breaker Settings
  GREENLANG_CIRCUIT_BREAKER_THRESHOLD=5
  GREENLANG_CIRCUIT_BREAKER_TIMEOUT=60

  # Monitoring
  GREENLANG_METRICS_ENABLED=true
  GREENLANG_METRICS_PORT=9090
  ```

- [ ] **Budget Configuration:**
  - Set daily/monthly spending limits in provider dashboards
  - Configure alerts for budget thresholds
  - Test auto-cutoff at budget limits

### 3. Infrastructure

- [ ] **Monitoring Stack:**
  - Prometheus for metrics collection (optional)
  - Grafana for visualization (optional)
  - AlertManager for alerts (optional)
  - **OR** use built-in CLI dashboard

- [ ] **Logging:**
  - Configure log aggregation (e.g., CloudWatch, Datadog)
  - Set log retention policy (7-14 days for staging)
  - Enable structured logging for analysis

---

## Deployment Steps

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/greenlang.git
cd greenlang

# Create staging environment
python -m venv .venv-staging
source .venv-staging/bin/activate  # On Windows: .venv-staging\Scripts\activate

# Install dependencies
pip install -e ".[analytics]"
```

### Step 2: Configure Environment

```bash
# Copy staging config
cp .env.staging.template .env

# Edit configuration
nano .env

# Validate configuration
gl doctor
```

### Step 3: Run Validation Tests

```bash
# Run unit tests
pytest tests/intelligence/ -v

# Run integration tests
pytest tests/integration/test_staging_readiness.py -v

# Validate tool registry
python scripts/batch_retrofit_agents.py
```

### Step 4: Start Services

```bash
# Start GreenLang application
gl run staging_pipeline.yaml --inputs staging_inputs.json

# In separate terminal: Start metrics dashboard
python -m greenlang.intelligence.runtime.dashboard
```

### Step 5: Smoke Tests

Run critical path smoke tests:

```bash
# Test 1: Simple calculation
curl -X POST http://localhost:8000/intelligence/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Calculate emissions for 100 kWh of electricity in US"}],
    "budget_usd": 0.10
  }'

# Test 2: Tool invocation
curl -X POST http://localhost:8000/intelligence/tool \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "get_emission_factor",
    "arguments": {"country": "US", "fuel_type": "electricity", "unit": "kWh"}
  }'

# Test 3: Budget enforcement
curl -X POST http://localhost:8000/intelligence/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Analyze building carbon footprint"}],
    "budget_usd": 0.001  # Very low budget to test cutoff
  }'
```

Expected Results:
- Test 1: Returns emission calculation with units and sources
- Test 2: Returns emission factor with metadata
- Test 3: Returns budget exceeded error

### Step 6: Monitor Health

```bash
# Check dashboard
python -c "from greenlang.intelligence.runtime.dashboard import print_dashboard; print_dashboard()"

# Check metrics endpoint (if Prometheus enabled)
curl http://localhost:9090/metrics

# Check circuit breakers
curl http://localhost:8000/intelligence/health
```

---

## Monitoring and Observability

### Built-in Dashboard

```bash
# View real-time dashboard
python -m greenlang.intelligence.runtime.dashboard
```

**Key Metrics to Monitor:**
- **Budget:** Spent vs. remaining, burn rate
- **Success Rate:** % of successful LLM calls
- **Latency:** p50, p95, p99 response times
- **Circuit Breakers:** Open/closed status
- **Tool Invocations:** Success rate by tool
- **Alerts:** Active alerts and severity

### Log Monitoring

```bash
# Tail logs
tail -f /var/log/greenlang/intelligence.log

# Filter errors
grep "ERROR" /var/log/greenlang/intelligence.log

# Filter circuit breaker events
grep "Circuit breaker" /var/log/greenlang/intelligence.log
```

### Alert Configuration

Configure alerts for:
- **Budget:** < 20% remaining
- **Error Rate:** > 5%
- **Circuit Breaker:** Any open circuits
- **Latency:** p95 > 5000ms
- **JSON Retry:** >10% failure rate

---

## Testing Checklist

### Functional Tests

- [ ] **Simple Queries:** Test basic LLM interactions
- [ ] **Tool Calling:** Verify all retrofitted agents work
- [ ] **JSON Schema:** Test schema validation and retry logic
- [ ] **Context Management:** Test long conversations (>100 messages)
- [ ] **Provider Routing:** Verify cost optimization
- [ ] **Budget Enforcement:** Test hard limits

### Resilience Tests

- [ ] **Circuit Breaker:** Simulate provider outage
  ```bash
  # Set invalid API key to trigger failures
  export OPENAI_API_KEY=invalid_key
  # Make 6 requests to trip circuit breaker
  for i in {1..6}; do curl http://localhost:8000/intelligence/chat ...; done
  # Verify circuit is OPEN
  ```

- [ ] **Retry Logic:** Test JSON validation failures
- [ ] **Timeout Handling:** Test slow provider responses
- [ ] **Concurrent Requests:** Test under load (10+ parallel requests)

### Performance Tests

- [ ] **Latency:** Measure p95 latency under normal load
- [ ] **Throughput:** Measure requests/second capacity
- [ ] **Memory Usage:** Monitor for leaks over 1-hour test
- [ ] **Cost Efficiency:** Verify provider router savings

---

## Rollback Plan

If issues are discovered in staging:

### 1. Immediate Rollback

```bash
# Stop services
pkill -f "greenlang"

# Revert to previous version
git checkout <previous_tag>
pip install -e .

# Restart
gl run staging_pipeline.yaml
```

### 2. Budget Protection

```bash
# Disable intelligence layer if runaway costs
export GREENLANG_INTELLIGENCE_ENABLED=false

# Or set budget to zero
export GREENLANG_BUDGET_MAX_USD=0.0
```

### 3. Provider Fallback

```bash
# Disable specific provider
export GREENLANG_DISABLE_OPENAI=true

# Force single provider
export GREENLANG_FORCE_PROVIDER=anthropic
```

---

## Staging Acceptance Criteria

Before promoting to production, verify:

- [ ] **Uptime:** 99.5%+ uptime over 48-hour test period
- [ ] **Success Rate:** >95% successful LLM calls
- [ ] **Latency:** p95 latency < 3000ms
- [ ] **Cost:** Actual cost within 10% of projections
- [ ] **Alerts:** No critical alerts for 24 hours
- [ ] **Circuit Breakers:** Correct behavior during simulated outages
- [ ] **Monitoring:** All dashboards and alerts functioning
- [ ] **Documentation:** Team trained on operations

---

## Troubleshooting

### Issue: High Error Rate

**Symptoms:** >10% of requests failing
**Diagnosis:**
```bash
# Check error logs
grep "ERROR" /var/log/greenlang/intelligence.log | tail -20

# Check provider status
curl https://status.openai.com/
curl https://status.anthropic.com/
```

**Solution:**
- If provider issue: Wait for resolution or switch providers
- If code issue: Review recent changes, rollback if needed
- If quota issue: Increase API rate limits

### Issue: Circuit Breaker Open

**Symptoms:** "Circuit breaker is OPEN" errors
**Diagnosis:**
```bash
# Check circuit breaker status
curl http://localhost:8000/intelligence/health

# Check failure history
grep "Circuit breaker" /var/log/greenlang/intelligence.log
```

**Solution:**
- Identify root cause (provider outage, quota, timeout)
- Fix root cause
- Manually reset circuit breaker:
  ```python
  from greenlang.intelligence.providers.resilience import get_resilient_client
  client = get_resilient_client("openai")
  client.reset()
  ```

### Issue: Budget Exceeded

**Symptoms:** Budget limit reached unexpectedly
**Diagnosis:**
```bash
# Check cost dashboard
python -c "from greenlang.intelligence.runtime.dashboard import print_dashboard; print_dashboard()"

# Review cost by provider
grep "cost_usd" /var/log/greenlang/intelligence.log
```

**Solution:**
- Increase budget if legitimate usage
- Investigate unexpected usage patterns
- Check for inefficient routing (always using expensive models)
- Review provider router configuration

---

## Next Steps

After successful staging deployment and testing:

1. **Document Findings:** Note any issues or optimizations needed
2. **Update Production Plan:** Incorporate staging learnings
3. **Train Operations Team:** Conduct handoff sessions
4. **Schedule Production Deployment:** Follow production deployment guide

---

## Support

- **Documentation:** `/docs/PRODUCTION_DEPLOYMENT_GUIDE.md`
- **Monitoring Dashboard:** `python -m greenlang.intelligence.runtime.dashboard`
- **Health Check:** `curl http://localhost:8000/intelligence/health`
- **Issues:** https://github.com/your-org/greenlang/issues

---

**Last Updated:** 2025-10-01
**Maintainer:** GreenLang Intelligence Team
**Version:** 1.0
