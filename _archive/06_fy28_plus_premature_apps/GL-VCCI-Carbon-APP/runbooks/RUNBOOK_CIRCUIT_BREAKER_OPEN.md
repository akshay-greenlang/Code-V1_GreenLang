# Runbook: Circuit Breaker Open

**Severity**: High
**Owner**: Platform Operations Team
**Last Updated**: 2025-11-09

## Symptoms

### What the User/Operator Sees
- API requests failing with "Circuit breaker is OPEN" error
- Increased error rates in monitoring dashboards
- Degraded service quality or fallback responses
- Alerts firing: `CircuitBreakerOpenAlert`

### Metrics/Alerts That Fire
- **Alert**: `CircuitBreakerOpen` (Severity: High)
- **Metric**: `circuit_breaker_state = "open"`
- **Metric**: `circuit_breaker_failure_count > threshold`
- **Metric**: `error_rate > 50%` for specific service

### Example Error Messages
```
CircuitBreakerError: Circuit breaker is OPEN. Failures: 5, Last failure: 30s ago
Service unavailable due to circuit breaker protection
Falling back to degraded mode
```

## Impact

### User Impact
- **Severity**: High
- **Scope**: Affects all requests to the protected service
- **User Experience**:
  - Requests may fail immediately (fail-fast)
  - System may fall back to cached data or degraded functionality
  - Slower response times if using fallback chain

### Business Impact
- **Revenue**: Potential calculation delays for Scope 3 emissions
- **Compliance**: May affect real-time reporting capabilities
- **SLA**: Violates 99.9% availability SLA if extended
- **Reputation**: Customer-facing errors

## Diagnosis

### Step 1: Confirm Circuit Breaker State
```bash
# Check circuit breaker status
curl http://localhost:8000/health/circuit-breakers

# Expected output:
{
  "factor_broker": {
    "state": "open",
    "failure_count": 5,
    "last_failure": "2025-11-09T10:30:00Z"
  }
}
```

### Step 2: Identify the Failing Service
```bash
# Check which service is protected by the open circuit
grep "Circuit breaker OPEN" /var/log/greenlang/app.log | tail -20

# Look for patterns:
# - factor_broker: Emission factor API
# - llm_service: LLM categorization
# - erp_connector: ERP data extraction
```

### Step 3: Check Underlying Service Health
```bash
# Test the failing service directly
curl -v https://api.factor-broker.com/health

# Check recent errors
grep "factor_broker" /var/log/greenlang/errors.log | tail -50

# Common issues:
# - HTTP 503: Service temporarily unavailable
# - HTTP 429: Rate limit exceeded
# - Connection timeouts
# - DNS resolution failures
```

### Step 4: Review Failure Timeline
```bash
# Check when failures started
grep "record_failure" /var/log/greenlang/app.log | grep "factor_broker" | tail -20

# Analyze pattern:
# - Gradual increase → Degrading service
# - Sudden spike → Service outage or deployment
# - Periodic → Network issues or rate limiting
```

### Step 5: Check System Resource Health
```bash
# Check system resources
top -n 1
df -h
netstat -an | grep ESTABLISHED | wc -l

# Check for:
# - High CPU/memory usage
# - Disk space exhaustion
# - Connection pool exhaustion
```

## Resolution

### Immediate Actions (5 minutes)

#### 1. Verify System is Operating in Degraded Mode
```bash
# Check fallback mechanisms are working
curl http://localhost:8000/api/calculate \
  -H "Content-Type: application/json" \
  -d '{"supplier":"test","spend":1000}'

# Should return with degraded mode indicator:
{
  "emissions": 500,
  "degraded": true,
  "factor_source": "default"
}
```

#### 2. Attempt Manual Reset (Only if appropriate)
```bash
# Reset circuit breaker manually
curl -X POST http://localhost:8000/admin/circuit-breaker/reset \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"service": "factor_broker"}'

# ⚠️ WARNING: Only reset if underlying issue is resolved!
# Premature reset will immediately re-open the circuit
```

#### 3. Enable Enhanced Monitoring
```bash
# Increase log verbosity
curl -X POST http://localhost:8000/admin/logging/level \
  -d '{"logger": "greenlang.intelligence.providers.resilience", "level": "DEBUG"}'

# Watch logs in real-time
tail -f /var/log/greenlang/app.log | grep "circuit_breaker"
```

### Short-Term Actions (30 minutes)

#### 1. Fix Underlying Service Issue

**If Rate Limited:**
```bash
# Check rate limit quotas
curl https://api.factor-broker.com/quota

# Increase rate limits or add delay:
# Edit config: /etc/greenlang/config.yaml
rate_limiting:
  factor_broker:
    requests_per_minute: 30  # Reduce from 60
    burst: 5

# Restart service
sudo systemctl restart greenlang-api
```

**If Service Down:**
```bash
# Check service provider status
curl https://status.factor-broker.com

# Contact vendor support
# Escalate to Tier 2 if vendor outage

# Enable fallback data source:
# Edit: /etc/greenlang/config.yaml
fallback:
  factor_broker:
    enabled: true
    source: "default_factors"
```

**If Timeout Issues:**
```bash
# Increase timeout threshold
# Edit: /etc/greenlang/config.yaml
resilience:
  factor_broker:
    timeout: 60  # Increase from 30

# Restart
sudo systemctl restart greenlang-api
```

#### 2. Wait for Automatic Recovery
```bash
# Circuit breaker will attempt recovery after timeout (default: 60s)
# Monitor recovery attempts:
tail -f /var/log/greenlang/app.log | grep "HALF_OPEN\|recovery"

# Successful recovery log:
# "Circuit breaker HALF_OPEN (testing recovery)"
# "Circuit breaker recovered: HALF_OPEN → CLOSED"
```

### Long-Term Fixes (1-2 hours)

#### 1. Tune Circuit Breaker Thresholds
```yaml
# /etc/greenlang/config.yaml
resilience:
  circuit_breaker:
    failure_threshold: 10  # Increase from 5 to reduce sensitivity
    recovery_timeout: 120  # Increase from 60 for more recovery time
    success_threshold: 5   # Require more successes before closing
```

#### 2. Implement Better Fallbacks
```python
# Add robust fallback strategy
# File: /app/greenlang/services/factor_broker/client.py

async def get_emission_factor_with_fallback(category):
    try:
        return await circuit_breaker.call(get_factor_from_api, category)
    except CircuitBreakerError:
        # Fallback 1: Cache
        if cached := cache.get(f"factor:{category}"):
            return cached

        # Fallback 2: Default factors
        return DEFAULT_EMISSION_FACTORS.get(category, DEFAULT_FACTOR)
```

#### 3. Add Metrics and Alerts
```yaml
# /etc/greenlang/monitoring/alerts.yaml
- alert: CircuitBreakerHighFailureRate
  expr: circuit_breaker_failure_count > 3
  for: 1m
  severity: warning
  annotations:
    summary: "Circuit breaker approaching threshold"

- alert: CircuitBreakerOpen
  expr: circuit_breaker_state == "open"
  for: 5m
  severity: critical
  annotations:
    summary: "Circuit breaker has opened"
```

## Prevention

### How to Prevent Recurrence

#### 1. Proactive Monitoring
- Monitor failure rates before circuit opens
- Set up early warning alerts at 60% of threshold
- Track service health trends

#### 2. Capacity Planning
- Review rate limits and quotas monthly
- Load test circuit breaker behavior
- Plan for traffic spikes (e.g., month-end reporting)

#### 3. Vendor Management
- Maintain SLA with Factor Broker vendor
- Have backup data sources configured
- Regular vendor health check meetings

#### 4. Circuit Breaker Configuration
```yaml
# Recommended settings for production
resilience:
  circuit_breaker:
    failure_threshold: 10      # Higher threshold for stability
    recovery_timeout: 120      # 2 minutes for recovery
    success_threshold: 5       # Require sustained success
    half_open_max_calls: 1     # Test with single request
```

#### 5. Regular Testing
```bash
# Weekly circuit breaker drill
./scripts/test_circuit_breaker.sh

# Monthly chaos engineering
python -m tests.chaos.test_resilience_chaos
```

## Runbook Metadata

- **Version**: 1.0
- **Created**: 2025-11-09
- **Last Tested**: 2025-11-09
- **Average Resolution Time**: 15-30 minutes
- **Escalation Path**: Platform Team → Infrastructure Team → Vendor Support
- **Related Runbooks**:
  - `RUNBOOK_HIGH_FAILURE_RATE.md`
  - `RUNBOOK_DEPENDENCY_DOWN.md`

## Appendix

### Useful Commands
```bash
# Check all circuit breaker states
curl http://localhost:8000/health/circuit-breakers | jq

# View circuit breaker metrics
curl http://localhost:8000/metrics | grep circuit_breaker

# Force circuit open (testing only)
curl -X POST http://localhost:8000/admin/circuit-breaker/open -d '{"service":"factor_broker"}'

# Reset all circuit breakers
curl -X POST http://localhost:8000/admin/circuit-breaker/reset-all
```

### Contact Information
- **On-Call Engineer**: PagerDuty escalation
- **Platform Team**: platform-team@greenlang.com
- **Vendor Support**: Factor Broker Support (support@factor-broker.com)

### References
- [Circuit Breaker Pattern Documentation](../CIRCUIT_BREAKER_DEVELOPER_GUIDE.md)
- [System Architecture](../../docs/ARCHITECTURE.md)
- [Monitoring Dashboard](https://grafana.greenlang.com/d/resilience)
