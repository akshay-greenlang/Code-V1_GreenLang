# Runbook: Operating in Graceful Degradation Mode

**Severity**: Medium
**Owner**: Platform Operations Team
**Last Updated**: 2025-11-09

## Symptoms

### What the User/Operator Sees
- System operational but with reduced functionality
- Responses include `"degraded": true` flag
- Using cached or default data instead of live data
- Warning messages in UI about degraded mode
- Alert: `SystemDegradedAlert`

### Metrics/Alerts That Fire
- **Alert**: `GracefulDegradation` (Severity: Medium)
- **Metric**: `degraded_mode_requests > 10%`
- **Metric**: `fallback_usage_rate > 0.5`
- **Metric**: `data_freshness > 1 hour`

## Impact

### User Impact
- **Data Freshness**: Using cached/default data (may be stale)
- **Accuracy**: Lower precision (default factors vs. specific factors)
- **Performance**: Faster responses (no external API calls)
- **Completeness**: Some features disabled

### Business Impact
- **Compliance**: Acceptable for short-term (<24 hours)
- **Quality**: 80-90% accuracy vs. 95%+ in normal mode
- **Revenue**: Minimal impact if temporary

## Diagnosis

### Step 1: Identify Degraded Services
```bash
# Check which services are degraded
curl http://localhost:8000/health/degradation | jq

# Sample output:
{
  "mode": "degraded",
  "degraded_services": [
    {
      "service": "factor_broker",
      "reason": "circuit_breaker_open",
      "fallback": "default_factors",
      "since": "2025-11-09T10:00:00Z"
    }
  ],
  "impact": "Using default emission factors"
}
```

### Step 2: Check Degradation Reason
```bash
# Common reasons:
# 1. Circuit breaker open
curl http://localhost:8000/health/circuit-breakers | jq

# 2. High latency
curl http://localhost:8000/metrics | grep latency_p99

# 3. Rate limiting
curl http://localhost:8000/metrics | grep rate_limit_exceeded

# 4. Dependency unavailable
curl http://localhost:8000/health/dependencies | jq
```

### Step 3: Assess Data Quality
```bash
# Compare degraded vs. normal mode results
python scripts/compare_degraded_quality.py

# Check data freshness
curl http://localhost:8000/admin/cache/stats | jq '.age_seconds'

# Validate default factors
cat /etc/greenlang/data/default_factors.json | jq
```

### Step 4: Estimate Recovery Time
```bash
# Check circuit breaker recovery timeout
grep "recovery_timeout" /etc/greenlang/config.yaml

# Check vendor ETA (if applicable)
curl https://status.factor-broker.com/api/incidents/latest | jq '.eta'
```

## Resolution

### Immediate Actions (5 minutes)

#### 1. Verify Degraded Mode is Safe
```bash
# Confirm fallback data quality
curl http://localhost:8000/api/calculate \
  -d '{
    "supplier": "Test Corp",
    "spend_usd": 100000,
    "category": "electricity"
  }' | jq

# Expected response:
{
  "emissions_kg_co2": 50000,
  "factor_used": 0.500,
  "factor_source": "default",
  "degraded": true,
  "confidence": "medium"
}

# Verify result is reasonable
```

#### 2. Notify Users
```bash
# Update status page
curl -X POST https://status.greenlang.com/api/update \
  -d '{
    "status": "degraded_performance",
    "message": "System operating with cached data. Accuracy may be reduced."
  }'

# Send email notification (if >1 hour)
python scripts/notify_degraded_mode.py
```

#### 3. Enable Enhanced Monitoring
```bash
# Monitor degradation metrics
watch -n 10 'curl -s http://localhost:8000/metrics | grep degraded'

# Track data quality
python scripts/monitor_data_quality.py --interval 60
```

### Short-Term Actions (30 minutes)

#### 1. Optimize Fallback Strategy
```python
# Improve fallback data quality
# File: /app/config/fallback_config.py

FALLBACK_STRATEGIES = {
    "factor_broker": {
        # Priority 1: Recent cache (last 24h)
        "cache": {"ttl": 86400, "priority": 1},

        # Priority 2: Default factors (industry standard)
        "defaults": {"priority": 2, "quality": "medium"},

        # Priority 3: Conservative estimates
        "conservative": {"priority": 3, "quality": "low"}
    }
}
```

#### 2. Refresh Cached Data
```bash
# Try to refresh cache from any available source
python scripts/refresh_cache.py --source backup_api

# Validate refreshed data
python scripts/validate_cache.py
```

#### 3. Reduce Degraded Scope
```bash
# If only some categories affected, isolate them
curl -X POST http://localhost:8000/admin/fallback/configure \
  -d '{
    "partial_degradation": true,
    "degraded_categories": ["electricity_grid"],
    "normal_categories": ["natural_gas", "transportation"]
  }'
```

### Long-Term Actions (1-4 hours)

#### 1. Monitor for Recovery Opportunity
```bash
# Poll primary service for recovery
while true; do
  health=$(curl -s http://localhost:8000/health/dependencies | \
    jq -r '.factor_broker.status')

  if [ "$health" = "healthy" ]; then
    echo "Service recovered! Initiating cutover..."
    break
  fi

  sleep 60
done
```

#### 2. Gradual Cutover to Normal Mode
```bash
# Step 1: Test with low traffic (10%)
curl -X POST http://localhost:8000/admin/cutover \
  -d '{"mode": "canary", "percentage": 10}'

# Monitor for 10 minutes
python scripts/monitor_cutover.py --duration 600

# Step 2: Increase gradually
# 10% → 25% → 50% → 75% → 100%
for pct in 25 50 75 100; do
  curl -X POST http://localhost:8000/admin/cutover \
    -d "{\"percentage\": $pct}"
  sleep 300  # Wait 5 minutes between increases
done
```

#### 3. Validate Full Recovery
```bash
# Run validation suite
pytest tests/integration/test_full_recovery.py -v

# Compare results: degraded vs. normal
python scripts/validate_recovery.py

# Check data quality metrics
curl http://localhost:8000/metrics | grep data_quality
```

## Operating Procedures in Degraded Mode

### Do's
- ✅ Continue accepting requests
- ✅ Use cached/default data
- ✅ Monitor data quality
- ✅ Log all degraded responses
- ✅ Notify users of degraded state

### Don'ts
- ❌ Don't disable system completely
- ❌ Don't hide degraded state from users
- ❌ Don't ignore data quality issues
- ❌ Don't extend degraded mode >24 hours without escalation

### Acceptable Duration
- **0-1 hour**: Normal operations, auto-recovery expected
- **1-4 hours**: Monitor closely, notify stakeholders
- **4-24 hours**: Escalate to engineering, find workaround
- **>24 hours**: Critical - involve leadership, consider backup vendors

## Prevention

### 1. Better Default Data
```python
# Maintain high-quality default factors
DEFAULT_EMISSION_FACTORS = {
    "electricity_grid_us_avg": {
        "factor": 0.385,
        "unit": "kg_co2_per_kwh",
        "source": "EPA 2024 eGrid",
        "confidence": "high",
        "last_updated": "2024-01-01"
    },
    # ... more factors
}

# Update quarterly from authoritative sources
```

### 2. Proactive Cache Warming
```bash
# Pre-populate cache daily
0 2 * * * python scripts/warm_cache.py

# Contents:
# - Top 100 supplier categories
# - Common emission factors
# - Recent calculation patterns
```

### 3. Multi-Vendor Strategy
```yaml
# Configure multiple vendors
vendors:
  primary: factor_broker_vendor_a
  secondary: factor_broker_vendor_b
  tertiary: default_factors

# Auto-failover rules
failover:
  - if: primary.latency > 5s
    then: use secondary
  - if: secondary.unavailable
    then: use tertiary
```

### 4. Degradation Testing
```bash
# Monthly degradation drills
python tests/chaos/test_degraded_mode.py

# Validate:
# - Fallback activation
# - Data quality
# - User experience
# - Recovery procedures
```

## Runbook Metadata

- **Version**: 1.0
- **Typical Duration**: 1-4 hours
- **Escalation**: >4 hours → Platform Lead, >24 hours → CTO
- **Related Runbooks**:
  - `RUNBOOK_CIRCUIT_BREAKER_OPEN.md`
  - `RUNBOOK_DEPENDENCY_DOWN.md`

## Appendix

### Default Emission Factors
```json
{
  "electricity_grid_us": 0.385,
  "electricity_grid_eu": 0.295,
  "natural_gas": 0.185,
  "transportation_road": 0.120,
  "air_freight": 0.500,
  "waste_landfill": 0.250
}
```

### Quality Metrics
- **High Quality**: Live data from primary vendor (95%+ accuracy)
- **Medium Quality**: Cached data <24h old (90%+ accuracy)
- **Low Quality**: Default factors (80%+ accuracy)
- **Unacceptable**: Default factors >7 days old or no validation

### Contact Information
- **Platform Team**: platform-ops@greenlang.com
- **Data Team**: data-quality@greenlang.com
- **On-Call**: PagerDuty escalation
