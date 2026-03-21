# PACK-030: Integration Guides

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [Pack Integrations (1-4)](#pack-integrations)
3. [Application Integrations (5-8)](#application-integrations)
4. [External Integrations (9-10)](#external-integrations)
5. [Platform Integrations (11-12)](#platform-integrations)
6. [Common Patterns](#common-patterns)
7. [Error Handling](#error-handling)
8. [Testing Integrations](#testing-integrations)

---

## 1. Integration Overview

| # | Integration | File | Lines | Protocol | Auth |
|---|-------------|------|-------|----------|------|
| 1 | PACK-021 Starter | `integrations/pack021_integration.py` | 700 | REST API | API Key |
| 2 | PACK-022 Acceleration | `integrations/pack022_integration.py` | 700 | REST API | API Key |
| 3 | PACK-028 Sector Pathway | `integrations/pack028_integration.py` | 700 | REST API | API Key |
| 4 | PACK-029 Interim Targets | `integrations/pack029_integration.py` | 700 | REST API | API Key |
| 5 | GL-SBTi-APP | `integrations/gl_sbti_app_integration.py` | 800 | GraphQL | OAuth 2.0 |
| 6 | GL-CDP-APP | `integrations/gl_cdp_app_integration.py` | 800 | GraphQL | OAuth 2.0 |
| 7 | GL-TCFD-APP | `integrations/gl_tcfd_app_integration.py` | 750 | GraphQL | OAuth 2.0 |
| 8 | GL-GHG-APP | `integrations/gl_ghg_app_integration.py` | 750 | GraphQL | OAuth 2.0 |
| 9 | XBRL Taxonomy | `integrations/xbrl_taxonomy_integration.py` | 600 | HTTPS | None |
| 10 | Translation Service | `integrations/translation_integration.py` | 500 | REST API | API Key |
| 11 | Orchestrator | `integrations/orchestrator_integration.py` | 400 | REST API | JWT |
| 12 | Health Check | `integrations/health_check_integration.py` | 300 | Internal | N/A |
| **Total** | | | **8,200** | | |

---

## 2. Pack Integrations

### Integration 1: PACK-021 Net Zero Starter

**Purpose:** Pull baseline emissions, GHG inventory, and activity data.

**Data Points:**

| Method | Endpoint | Returns |
|--------|----------|---------|
| `fetch_baseline()` | `GET /api/v1/baseline/{org_id}` | Base year emissions (Scope 1/2/3) |
| `fetch_inventory()` | `GET /api/v1/inventory/{org_id}/{year}` | Full GHG inventory |
| `fetch_activity_data()` | `GET /api/v1/activity/{org_id}/{year}` | Activity data by source |

**Configuration:**

```python
pack021 = Pack021Integration(
    base_url="http://pack-021:9021",
    api_key=os.environ["PACK021_API_KEY"],
    timeout=30,
    retry_count=3,
    circuit_breaker_threshold=5,
)
```

**Usage:**

```python
baseline = await pack021.fetch_baseline(
    organization_id="org-uuid",
    base_year=2019,
)
# Returns: BaselineData(scope1=Decimal("45000"), scope2=Decimal("32000"), ...)
```

### Integration 2: PACK-022 Net Zero Acceleration

**Purpose:** Pull reduction initiatives, MACC curves, and abatement costs.

| Method | Endpoint | Returns |
|--------|----------|---------|
| `fetch_initiatives()` | `GET /api/v1/initiatives/{org_id}` | Active reduction initiatives |
| `fetch_macc()` | `GET /api/v1/macc/{org_id}` | Marginal abatement cost curve |
| `fetch_abatement()` | `GET /api/v1/abatement/{org_id}` | Abatement potential by initiative |

### Integration 3: PACK-028 Sector Pathway

**Purpose:** Pull sector-specific decarbonization pathways and benchmarks.

| Method | Endpoint | Returns |
|--------|----------|---------|
| `fetch_pathways()` | `GET /api/v1/pathways/{sector}` | Sector pathway data |
| `fetch_convergence()` | `GET /api/v1/convergence/{org_id}` | Convergence analysis |
| `fetch_benchmarks()` | `GET /api/v1/benchmarks/{sector}` | Peer benchmarking data |

### Integration 4: PACK-029 Interim Targets

**Purpose:** Pull interim targets, quarterly progress, and variance analysis.

| Method | Endpoint | Returns |
|--------|----------|---------|
| `fetch_targets()` | `GET /api/v1/interim-targets/{org_id}` | 5-year and 10-year targets |
| `fetch_progress()` | `GET /api/v1/progress/{org_id}/{period}` | Progress monitoring data |
| `fetch_variance()` | `GET /api/v1/variance/{org_id}/{year}` | LMDI variance decomposition |

---

## 3. Application Integrations

### Integration 5: GL-SBTi-APP

**Purpose:** Pull validated SBTi target data, validation status, and submission history.

| Method | Returns |
|--------|---------|
| `fetch_sbti_targets()` | Near-term and long-term validated targets |
| `fetch_validation()` | 21-criteria validation results |
| `fetch_submission_history()` | Historical SBTi submissions |

**GraphQL Query Example:**

```graphql
query GetSBTiTargets($orgId: UUID!) {
  organization(id: $orgId) {
    sbtiTargets {
      nearTerm {
        scope
        baseYear
        targetYear
        reductionPct
        ambitionLevel
        validationStatus
      }
      longTerm {
        targetYear
        residualEmissionsPct
        neutralizationStrategy
      }
    }
  }
}
```

### Integration 6: GL-CDP-APP

**Purpose:** Pull historical CDP responses, scores, and peer benchmarks.

| Method | Returns |
|--------|---------|
| `fetch_cdp_history()` | Previous year CDP responses |
| `fetch_scores()` | CDP scoring history (A through D-) |
| `fetch_peer_benchmarks()` | Sector peer CDP scores |

### Integration 7: GL-TCFD-APP

**Purpose:** Pull scenario analysis data, risk assessments, and opportunity evaluations.

| Method | Returns |
|--------|---------|
| `fetch_scenarios()` | 1.5C, 2C, 4C scenario analyses |
| `fetch_risks()` | Physical and transition risk assessments |
| `fetch_opportunities()` | Climate-related opportunities |

### Integration 8: GL-GHG-APP

**Purpose:** Pull GHG inventory, emission factors, and activity data.

| Method | Returns |
|--------|---------|
| `fetch_inventory()` | Full GHG inventory (Scope 1/2/3) |
| `fetch_emission_factors()` | Active emission factor database |
| `fetch_activity_data()` | Activity data by source and scope |

---

## 4. External Integrations

### Integration 9: XBRL Taxonomy

**Purpose:** Fetch and cache official XBRL taxonomies for SEC and CSRD validation.

| Method | Returns |
|--------|---------|
| `fetch_sec_taxonomy()` | SEC climate disclosure taxonomy (elements, contexts, units) |
| `fetch_csrd_taxonomy()` | CSRD ESRS digital taxonomy |
| `validate_tags()` | Validate XBRL tags against taxonomy |

**Caching Strategy:**
- Taxonomies are cached in Redis with 24-hour TTL
- Background refresh job runs daily to check for updates
- Version changes trigger cache invalidation and alert

### Integration 10: Translation Service

**Purpose:** Multi-language narrative translation via external translation APIs.

| Method | Returns |
|--------|---------|
| `translate()` | Translated text in target language |
| `detect_language()` | Detected source language |
| `validate_quality()` | Translation quality score |

**Supported Services:**
- Primary: DeepL API (highest quality for climate terminology)
- Fallback: Google Cloud Translation API
- Cache: Redis cache for repeated translations (7-day TTL)

---

## 5. Platform Integrations

### Integration 11: Orchestrator

**Purpose:** Register PACK-030 with the GreenLang Orchestrator for DAG-based workflow management.

| Method | Purpose |
|--------|---------|
| `register_pack()` | Register pack capabilities and endpoints |
| `report_health()` | Send health status to orchestrator |
| `handle_orchestration()` | Handle orchestrated workflow requests |

### Integration 12: Health Check

**Purpose:** Monitor health of all 11 upstream integrations.

| Method | Purpose |
|--------|---------|
| `check_pack_health()` | Verify all 4 pack integrations |
| `check_app_health()` | Verify all 4 app integrations |
| `check_external_services()` | Verify XBRL and translation services |

**Health Check Response Format:**

```json
{
  "integration": "PACK-021",
  "status": "healthy",
  "latency_ms": 15,
  "last_success": "2026-03-20T10:00:00Z",
  "circuit_breaker": "closed",
  "error_rate_1h": 0.01
}
```

---

## 6. Common Patterns

### Circuit Breaker

All integrations use the circuit breaker pattern:

```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,     # Opens after 5 consecutive failures
        recovery_timeout: int = 60,      # Half-open after 60 seconds
        success_threshold: int = 3,      # Closes after 3 successes in half-open
    ):
        ...
```

### Retry with Exponential Backoff

```python
class RetryConfig:
    max_retries: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_multiplier: float = 2.0
    jitter: bool = True                  # Add randomness to prevent thundering herd
```

### Connection Pooling

```python
# HTTP connection pool for each integration
pool = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
        keepalive_expiry=30,
    ),
    timeout=httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0,
    ),
)
```

### Response Caching

```python
# Cache integration responses in Redis
@cached(ttl=3600, key_prefix="pack030:integration")
async def fetch_baseline(self, organization_id: str, base_year: int):
    ...
```

---

## 7. Error Handling

### Error Classification

| Error Type | Behavior | Example |
|-----------|----------|---------|
| `ConnectionError` | Retry (circuit breaker) | Network timeout |
| `AuthenticationError` | Fail immediately | Invalid API key |
| `RateLimitError` | Retry after delay | 429 response |
| `DataNotFoundError` | Return empty/default | Organization not in system |
| `ValidationError` | Fail with details | Invalid response format |

### Graceful Degradation

When an integration is unavailable, PACK-030 continues with available data:

```python
try:
    sbti_data = await gl_sbti_integration.fetch_sbti_targets(org_id)
except IntegrationError:
    logger.warning("GL-SBTi-APP unavailable, using cached data")
    sbti_data = await cache.get(f"sbti_targets:{org_id}")
    if not sbti_data:
        sbti_data = SBTiTargets.empty()
        report.add_warning("SBTi target data unavailable")
```

---

## 8. Testing Integrations

### Mock Strategy

All integrations have mock implementations for testing:

```python
# tests/conftest.py
@pytest.fixture
def mock_pack021():
    """Mock PACK-021 integration for testing."""
    mock = AsyncMock(spec=Pack021Integration)
    mock.fetch_baseline.return_value = BaselineData(
        scope1=Decimal("45000"),
        scope2_location=Decimal("32000"),
        scope2_market=Decimal("28000"),
        scope3=Decimal("120000"),
        base_year=2019,
    )
    return mock
```

### Integration Test Pattern

```python
@pytest.mark.integration
async def test_pack021_fetch_baseline():
    """Test PACK-021 baseline fetch (requires running PACK-021)."""
    integration = Pack021Integration(base_url="http://localhost:9021")
    result = await integration.fetch_baseline(
        organization_id="test-org-uuid",
        base_year=2019,
    )
    assert result.scope1 > 0
    assert result.base_year == 2019
```

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
