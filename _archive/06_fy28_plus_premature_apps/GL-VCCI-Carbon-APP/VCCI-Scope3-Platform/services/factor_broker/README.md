# Factor Broker Service

**Version:** 1.0.0
**Part of:** GL-VCCI Scope 3 Carbon Platform
**Phase:** 2 (Weeks 3-4)

## Overview

The Factor Broker is a critical service that provides runtime emission factor resolution with multi-source cascading, license compliance, and intelligent caching. It abstracts the complexity of managing multiple emission factor databases while ensuring compliance with licensing terms (particularly ecoinvent's no-bulk-redistribution policy).

## Key Features

### 1. Multi-Source Cascading
- **Automatic Fallback:** ecoinvent → DESNZ UK → EPA US → Proxy
- **Priority-Based:** Higher quality sources tried first
- **Transparent:** Provenance tracking shows which sources were attempted

### 2. License Compliance
- **No Bulk Export:** Runtime API access only for ecoinvent
- **24-Hour TTL:** Cache limited to 24 hours per ecoinvent license
- **Attribution:** All responses include proper source citation
- **Audit Trail:** Full provenance tracking for compliance verification

### 3. Performance Optimization
- **Redis Caching:** Sub-5ms cache hits
- **Target Latency:** <50ms p95 latency
- **Target Cache Hit Rate:** ≥85%
- **Concurrent Requests:** Async I/O for parallel processing

### 4. Data Quality
- **Pedigree Matrix:** DQI scoring based on ecoinvent methodology
- **Quality Degradation:** Proxy factors flagged with appropriate quality scores
- **Uncertainty Tracking:** All factors include uncertainty percentages
- **Provenance Chain:** SHA256 hash for reproducibility

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FactorBroker                          │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Cache    │  │   Sources   │  │   Provenance    │  │
│  │  (Redis)   │  │  Manager    │  │    Tracker      │  │
│  └────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                   │                  │
         ▼                   ▼                  ▼
  ┌───────────┐      ┌──────────────┐   ┌─────────────┐
  │  Redis    │      │   Sources    │   │  Calculation│
  │  Server   │      │              │   │  Provenance │
  └───────────┘      └──────────────┘   └─────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  ┌──────────┐        ┌──────────┐        ┌──────────┐
  │ecoinvent │        │  DESNZ   │        │   EPA    │
  │  (v3.10) │        │  (2024)  │        │  (2024)  │
  └──────────┘        └──────────┘        └──────────┘
```

## Files Structure

```
services/factor_broker/
├── __init__.py                 (49 lines)   - Package exports
├── broker.py                   (481 lines)  - Core FactorBroker class
├── models.py                   (619 lines)  - Pydantic data models
├── config.py                   (434 lines)  - Configuration management
├── cache.py                    (511 lines)  - Redis caching layer
├── exceptions.py               (502 lines)  - Custom exceptions
├── sources/
│   ├── __init__.py             (22 lines)   - Source exports
│   ├── base.py                 (323 lines)  - Abstract base class
│   ├── ecoinvent.py            (431 lines)  - ecoinvent integration
│   ├── desnz.py                (432 lines)  - DESNZ UK integration
│   ├── epa.py                  (436 lines)  - EPA US integration
│   └── proxy.py                (432 lines)  - Proxy factor calculator
└── README.md                   (this file)

Total Production Code: 4,672 lines

tests/services/factor_broker/
├── __init__.py                 (8 lines)
├── test_broker.py              (154 lines)  - Core broker tests
├── test_sources.py             (204 lines)  - Source integration tests
├── test_cache.py               (159 lines)  - Cache tests
├── test_models.py              (186 lines)  - Model validation tests
└── test_integration.py         (147 lines)  - End-to-end tests

Total Test Code: 858 lines (test stubs)
```

## Usage Examples

### Basic Factor Resolution

```python
from services.factor_broker import FactorBroker, FactorRequest, GWPStandard

# Initialize broker
broker = FactorBroker()

# Create request
request = FactorRequest(
    product="Steel",
    region="US",
    gwp_standard=GWPStandard.AR6,
    unit="kg"
)

# Resolve factor
async with broker:
    response = await broker.resolve(request)

    print(f"Factor ID: {response.factor_id}")
    print(f"Value: {response.value} {response.unit}")
    print(f"Source: {response.source}")
    print(f"Quality Score: {response.data_quality_score}/100")
    print(f"Uncertainty: ±{response.uncertainty * 100}%")
```

### GWP Standard Comparison

```python
from services.factor_broker import GWPComparisonRequest

# Compare AR5 vs AR6
comparison_request = GWPComparisonRequest(
    product="Steel",
    region="US"
)

comparison = await broker.compare_gwp_standards(comparison_request)

print(f"AR5: {comparison.ar5.value} {comparison.ar5.unit}")
print(f"AR6: {comparison.ar6.value} {comparison.ar6.unit}")
print(f"Difference: {comparison.difference_percent:.2f}%")
```

### Health Check

```python
health = await broker.health_check()

print(f"Status: {health.status}")
print(f"Cache Hit Rate: {health.cache_hit_rate * 100:.1f}%")
print(f"Average Latency: {health.average_latency_ms:.1f}ms")
```

## Configuration

### Environment Variables

```bash
# ecoinvent Configuration
ECOINVENT_API_ENDPOINT=https://api.ecoinvent.org/v3.10
ECOINVENT_API_KEY=your_api_key_here

# DESNZ Configuration
DESNZ_API_ENDPOINT=https://api.gov.uk/desnz/emission-factors

# EPA Configuration
EPA_API_ENDPOINT=https://api.epa.gov/easey/emission-factors

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=optional_password

# Cache Configuration
CACHE_ENABLED=true
CACHE_TTL_SECONDS=86400  # 24 hours (max for ecoinvent license)
CACHE_MAX_SIZE_MB=500

# Performance Configuration
TARGET_P95_LATENCY_MS=50.0
TARGET_CACHE_HIT_RATE=0.85

# Defaults
DEFAULT_GWP_STANDARD=AR6
LICENSE_COMPLIANCE_MODE=true
```

## Data Sources

### 1. ecoinvent (Priority 1)
- **Version:** 3.10
- **Coverage:** 20,000+ factors, global
- **License:** Commercial ($60k/year)
- **Quality Score:** 95/100
- **Restrictions:** No bulk redistribution, 24h cache limit

### 2. DESNZ UK (Priority 2)
- **Version:** 2024
- **Coverage:** 20,000+ factors, UK/EU focus
- **License:** Open Government License v3.0
- **Quality Score:** 90/100
- **Regions:** GB, UK, EU countries

### 3. EPA US (Priority 3)
- **Version:** 2024
- **Coverage:** 15,000+ factors, US focus
- **License:** Public Domain
- **Quality Score:** 90/100
- **Regions:** US

### 4. Proxy Calculator (Priority 4)
- **Method:** Category averages
- **Coverage:** 12 major categories
- **Quality Score:** 50/100
- **Uncertainty:** ±50%

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| p50 Latency | <10ms | 50th percentile response time |
| p95 Latency | <50ms | 95th percentile response time |
| p99 Latency | <100ms | 99th percentile response time |
| Cache Hit Rate | ≥85% | Percentage of requests served from cache |
| Throughput | 1000 req/s | Sustained request rate |
| Availability | 99.9% | Three nines uptime |

## License Compliance

### ecoinvent License Requirements

✅ **Compliant:**
- Runtime API access only (no bulk downloads)
- Caching limited to 24 hours
- Attribution in all responses
- No redistribution to third parties
- Full provenance tracking

❌ **Prohibited:**
- Bulk export endpoints
- Cache TTL > 24 hours
- Sharing factor databases
- Missing attribution

### Compliance Verification

```python
# Check configuration compliance
config = FactorBrokerConfig.from_env()
errors = config.validate()

if errors:
    print("License compliance issues:")
    for error in errors:
        print(f"  - {error}")
```

## Monitoring & Observability

### Metrics (Prometheus)

```python
# Request metrics
factor_broker_requests_total
factor_broker_cache_hits_total
factor_broker_cache_hit_rate

# Latency metrics
factor_broker_latency_seconds (histogram)

# Source metrics
factor_broker_source_usage{source="ecoinvent"}
factor_broker_proxy_usage_total
```

### Alerts

- **Low Cache Hit Rate:** Alert if <80%
- **High Latency:** Alert if p95 >100ms
- **Source Unavailable:** Alert if error rate >5%
- **High Proxy Usage:** Alert if >20% of factors are proxies

## Testing

### Run Unit Tests

```bash
pytest tests/services/factor_broker/test_*.py -v
```

### Run Integration Tests

```bash
pytest tests/services/factor_broker/test_integration.py -v --integration
```

### Run Performance Tests

```bash
pytest tests/services/factor_broker/test_integration.py -v --performance
```

## Design Decisions

### 1. Async/Await Architecture
- **Why:** Enables concurrent API calls and high throughput
- **Impact:** Better performance under load, more complex code

### 2. Redis for Caching
- **Why:** Fast, scalable, TTL support
- **Impact:** Additional infrastructure dependency

### 3. Pydantic Models
- **Why:** Type safety, validation, serialization
- **Impact:** Runtime validation overhead (minimal)

### 4. Cascading Sources
- **Why:** Maximizes factor coverage, handles source failures
- **Impact:** More complex error handling

### 5. Provenance Tracking
- **Why:** Audit trails, reproducibility, compliance
- **Impact:** Additional metadata in responses

## Next Steps

### Immediate (Week 3-4)
1. ✅ Complete implementation (DONE)
2. ⬜ Implement unit tests (stubs created)
3. ⬜ Integration with Scope3CalculatorAgent
4. ⬜ Performance testing and optimization

### Future Enhancements
1. **Fuzzy Matching:** Product name suggestions
2. **Regional Fallbacks:** Automatic fallback to global factors
3. **Unit Conversion:** Automatic conversion between units
4. **Factor Versioning:** Track factor changes over time
5. **ML-Based Proxies:** More sophisticated proxy calculations
6. **GraphQL API:** Alternative to REST for factor queries

## Troubleshooting

### Issue: Cache Not Working
```bash
# Check Redis connection
redis-cli ping

# Check cache stats
curl http://localhost:8000/api/v1/factor-broker/health
```

### Issue: High Latency
```bash
# Check source latencies
curl http://localhost:8000/api/v1/factor-broker/health

# Review cache hit rate
# Target: >=85%
```

### Issue: License Compliance Error
```bash
# Verify cache TTL is <=24 hours
echo $CACHE_TTL_SECONDS  # Should be 86400 or less
```

## Support

- **Documentation:** See `specs/factor_broker_spec.yaml`
- **Issues:** Report via GitHub Issues
- **Contact:** Platform Engineering Team

---

**Generated as part of CTO v2 Strategic Plan - Phase 2**
**GL-VCCI Scope 3 Carbon Platform**
