# Runbook: External Dependency Down

**Severity**: Critical
**Owner**: Platform Operations Team
**Last Updated**: 2025-11-09

## Symptoms

### What the User/Operator Sees
- "Service temporarily unavailable" errors
- Calculations falling back to default values
- Alert: `DependencyDownAlert` firing
- Dashboard showing dependency health = RED

### Metrics/Alerts That Fire
- **Alert**: `DependencyDown` (Severity: Critical)
- **Metric**: `dependency_health{service="factor_broker"} = 0`
- **Metric**: `dependency_response_time > 30s`
- **Metric**: All requests to dependency failing

## Impact

### User Impact
- **Factor Broker Down**: Calculations use default emission factors (less accurate)
- **LLM Service Down**: Manual categorization required
- **ERP Connector Down**: Cannot fetch latest supplier data

### Business Impact
- Reduced data quality
- Compliance risk if extended
- Customer dissatisfaction

## Diagnosis

### Step 1: Confirm Dependency is Down
```bash
# Test dependency directly
curl -v https://api.factor-broker.com/health
# Expected: Connection refused, timeout, or 503

# Check from application
curl http://localhost:8000/health/dependencies | jq '.factor_broker'
# Expected: {"status": "unhealthy", "last_success": "2025-11-09T09:00:00Z"}
```

### Step 2: Identify Scope of Outage
```bash
# Check vendor status page
curl https://status.factor-broker.com/api/status

# Check multiple regions/endpoints
for region in us-east us-west eu-west; do
  echo "Testing $region..."
  curl -s https://api-${region}.factor-broker.com/health
done
```

### Step 3: Check Fallback Status
```bash
# Verify fallback is working
curl http://localhost:8000/api/calculate \
  -d '{"supplier":"test","spend":1000}' | jq

# Look for:
# "factor_source": "default" or "cached"
# "degraded": true
```

## Resolution

### Immediate Actions (2 minutes)

#### 1. Confirm Fallback is Active
```bash
# Check fallback configuration
curl http://localhost:8000/admin/fallback/status | jq

# Expected:
{
  "factor_broker": {
    "fallback_enabled": true,
    "fallback_source": "default_factors",
    "requests_served_from_fallback": 245
  }
}
```

#### 2. Notify Stakeholders
```bash
# Send status update
curl -X POST https://status.greenlang.com/api/incident \
  -d '{
    "title": "Factor Broker Unavailable - Operating in Degraded Mode",
    "status": "investigating",
    "impact": "minor"
  }'
```

### Short-Term Actions (15 minutes)

#### 1. Enable Enhanced Fallback
```yaml
# /etc/greenlang/config.yaml
fallback:
  factor_broker:
    enabled: true
    cache_ttl: 7200  # Use cache for 2 hours
    default_factors:
      electricity_grid: 0.500
      natural_gas: 0.185
      transportation: 0.120
```

#### 2. Contact Vendor
```bash
# Check vendor status
curl https://status.factor-broker.com

# Open support ticket
# Email: support@factor-broker.com
# Subject: "Production Outage - Customer ID: GREENLANG-001"
```

#### 3. Route to Backup Service (if available)
```yaml
# Switch to backup provider
services:
  factor_broker:
    primary_url: https://api.factor-broker-backup.com  # Backup
    # Original: https://api.factor-broker.com
```

### Long-Term Actions (2-4 hours)

#### 1. Monitor Recovery
```bash
# Poll vendor status
while true; do
  status=$(curl -s https://api.factor-broker.com/health | jq -r .status)
  echo "$(date): Status = $status"
  [ "$status" = "healthy" ] && break
  sleep 60
done
```

#### 2. Gradual Cutover
```bash
# When vendor recovers, gradually shift traffic
# Start with 10% of requests
curl -X POST http://localhost:8000/admin/traffic \
  -d '{"factor_broker_primary": 0.1, "factor_broker_fallback": 0.9}'

# Monitor for 10 minutes, then increase
# 25%, 50%, 75%, 100%
```

#### 3. Validate Data Quality
```bash
# Compare fallback vs primary results
python scripts/validate_fallback_accuracy.py

# Check for data drift
```

## Prevention

### 1. Multi-Region Deployment
```yaml
# Configure multiple endpoints
services:
  factor_broker:
    endpoints:
      - https://api-us-east.factor-broker.com
      - https://api-us-west.factor-broker.com
      - https://api-eu-west.factor-broker.com
    failover: automatic
```

### 2. Improved Caching
```python
# Cache emission factors for 24 hours
@cache(ttl=86400, stale_while_revalidate=True)
async def get_emission_factor(category):
    pass
```

### 3. Better Fallback Data
- Maintain high-quality default factors
- Update defaults quarterly
- Validate against industry standards

### 4. SLA Monitoring
```yaml
- alert: DependencyDowntime
  expr: dependency_uptime < 0.999
  for: 10m
  annotations:
    summary: "Dependency SLA violation"
```

## Runbook Metadata

- **Version**: 1.0
- **Average Resolution Time**: Depends on vendor (2 hours - 2 days)
- **Escalation**: Vendor Support (immediate), CTO (if >4 hours)
- **Related Runbooks**: `RUNBOOK_GRACEFUL_DEGRADATION.md`
