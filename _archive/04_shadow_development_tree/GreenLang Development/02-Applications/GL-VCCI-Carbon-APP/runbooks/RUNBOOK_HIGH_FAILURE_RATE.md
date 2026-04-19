# Runbook: High Failure Rate

**Severity**: High
**Owner**: Platform Operations Team
**Last Updated**: 2025-11-09

## Symptoms

### What the User/Operator Sees
- Increased error rates in API responses (>10% error rate)
- Intermittent failures across multiple requests
- Alert: `HighFailureRateAlert` firing
- Dashboard showing error spike

### Metrics/Alerts That Fire
- **Alert**: `HighFailureRate` (Severity: High)
- **Metric**: `error_rate > 10%` over 5 minutes
- **Metric**: `failed_requests_count` increasing rapidly
- **Metric**: `p99_latency` increasing

## Impact

### User Impact
- Intermittent calculation failures
- Degraded user experience
- Potential data loss if retries exhausted

### Business Impact
- SLA violation risk (99.9% uptime = 0.1% acceptable error rate)
- Customer complaints
- Potential revenue impact

## Diagnosis

### Step 1: Identify Failure Pattern
```bash
# Check overall error rate
curl http://localhost:8000/metrics | grep error_rate

# Sample output:
# error_rate{service="scope3-api"} 0.25  # 25% error rate - CRITICAL!

# Check error breakdown by endpoint
grep "ERROR" /var/log/greenlang/app.log | \
  awk '{print $5}' | sort | uniq -c | sort -rn | head -10
```

### Step 2: Analyze Error Types
```bash
# Group errors by type
grep "ERROR" /var/log/greenlang/app.log | \
  grep -oP 'Exception: \K[^:]+' | sort | uniq -c | sort -rn

# Common patterns:
# - ConnectionError → Network issues
# - TimeoutError → Dependency slowness
# - RateLimitError → API quota exceeded
# - ValidationError → Bad input data
```

### Step 3: Check Dependency Health
```bash
# Check all external dependencies
for service in factor_broker llm_api erp_connector; do
  echo "=== $service ==="
  curl -s http://localhost:8000/health/dependencies | \
    jq ".${service}"
done

# Check circuit breaker states
curl http://localhost:8000/health/circuit-breakers | jq
```

### Step 4: Review Resource Utilization
```bash
# Check system resources
top -bn1 | head -20
free -h
df -h

# Check connection pool
netstat -an | grep ESTABLISHED | wc -l
netstat -an | grep TIME_WAIT | wc -l

# Check for resource exhaustion
```

## Resolution

### Immediate Actions (5 minutes)

#### 1. Enable Rate Limiting
```bash
# Temporarily reduce request rate to protect system
curl -X POST http://localhost:8000/admin/rate-limit \
  -d '{"limit": 10, "period": "minute"}'

# This gives system time to recover
```

#### 2. Scale Up Resources
```bash
# Kubernetes: Scale up pods
kubectl scale deployment greenlang-api --replicas=6

# Docker: Add more containers
docker-compose up -d --scale api=4

# Check scaling progress
kubectl get pods -w
```

#### 3. Clear Failed Tasks
```bash
# Clear retry queue if backed up
redis-cli KEYS "celery*" | xargs redis-cli DEL

# Restart workers
sudo systemctl restart greenlang-workers
```

### Short-Term Actions (30 minutes)

#### 1. Fix Specific Error Causes

**If Rate Limited:**
```bash
# Increase API quotas or add delays
# Edit: /etc/greenlang/config.yaml
rate_limiting:
  requests_per_second: 20  # Reduce from 50
  burst_size: 10

sudo systemctl restart greenlang-api
```

**If Timeouts:**
```bash
# Increase timeout values
resilience:
  default_timeout: 60  # Increase from 30
  max_retries: 5

sudo systemctl restart greenlang-api
```

**If Connection Pool Exhausted:**
```bash
# Increase pool size
database:
  pool_size: 50  # Increase from 20
  max_overflow: 100

# Restart
sudo systemctl restart greenlang-api
```

#### 2. Enable Fallback Mechanisms
```python
# Ensure all services have fallbacks configured
# Check: /app/greenlang/config.yaml
fallback:
  enabled: true
  cache_ttl: 3600
  default_factors_enabled: true
```

### Long-Term Fixes

#### 1. Root Cause Analysis
- Review error logs for patterns
- Identify systemic issues
- Schedule post-mortem meeting

#### 2. Improve Monitoring
```yaml
# Add detailed error tracking
- alert: ErrorRateIncreasing
  expr: rate(errors_total[5m]) > 0.05
  for: 2m
  severity: warning
```

#### 3. Load Testing
```bash
# Run load tests to find limits
locust -f tests/load/test_api.py --headless \
  -u 100 -r 10 --run-time 5m
```

## Prevention

### 1. Capacity Planning
- Monitor baseline error rates
- Plan for 2x expected load
- Regular load testing

### 2. Better Error Handling
```python
# Implement retry with exponential backoff
@retry(max_attempts=3, backoff_factor=2)
async def resilient_call():
    pass
```

### 3. Circuit Breaker Tuning
- Lower thresholds for early detection
- Faster recovery times
- Better fallback strategies

## Runbook Metadata

- **Version**: 1.0
- **Average Resolution Time**: 30-45 minutes
- **Escalation Path**: On-Call → Platform Lead → CTO
- **Related Runbooks**: `RUNBOOK_CIRCUIT_BREAKER_OPEN.md`
