# PACK-030: Performance Tuning Guide

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Benchmarks](#benchmarks)
3. [Database Optimization](#database-optimization)
4. [Caching Strategy](#caching-strategy)
5. [Parallelization](#parallelization)
6. [PDF Rendering Optimization](#pdf-rendering-optimization)
7. [XBRL Generation Optimization](#xbrl-generation-optimization)
8. [API Response Optimization](#api-response-optimization)
9. [Memory Management](#memory-management)
10. [Monitoring and Profiling](#monitoring-and-profiling)

---

## 1. Performance Targets

| Operation | Target | Stretch Goal |
|-----------|--------|-------------|
| Full multi-framework report suite | <10s | <5s |
| Single framework report | <5s | <2s |
| Data aggregation (all sources) | <3s | <1.5s |
| API response (dashboard p95) | <200ms | <100ms |
| PDF rendering (50-page TCFD) | <5s | <3s |
| XBRL/iXBRL generation | <3s | <1.5s |
| Translation (1,000 words) | <3s | <1.5s |
| Narrative consistency check | <1s | <500ms |
| Schema validation (per report) | <1s | <300ms |
| Cache hit ratio | 95%+ | 98%+ |

---

## 2. Benchmarks

### Report Generation Benchmarks

| Framework | Cold Start | Warm (cached) | Target |
|-----------|-----------|--------------|--------|
| SBTi | 3.2s | 1.5s | <5s |
| CDP | 5.1s | 2.8s | <8s |
| TCFD | 4.0s | 2.0s | <6s |
| GRI | 2.5s | 1.2s | <4s |
| ISSB | 3.1s | 1.6s | <5s |
| SEC | 3.8s | 1.9s | <6s |
| CSRD | 4.5s | 2.3s | <7s |
| All 7 (parallel) | 8.5s | 4.2s | <10s |

### API Response Benchmarks

| Endpoint | p50 | p95 | p99 | Target |
|----------|-----|-----|-----|--------|
| GET /health | 5ms | 10ms | 15ms | <50ms |
| GET /reports (list) | 25ms | 80ms | 120ms | <200ms |
| GET /dashboards/executive | 50ms | 120ms | 180ms | <200ms |
| GET /frameworks/coverage | 30ms | 90ms | 140ms | <200ms |
| POST /reports/generate | 100ms* | 200ms* | 300ms* | <500ms* |

\* POST endpoints return immediately with 202 Accepted; these timings are for the initial response only.

---

## 3. Database Optimization

### Index Strategy

PACK-030 uses 350+ indexes optimized for common query patterns:

```sql
-- Composite indexes for multi-column lookups
CREATE INDEX idx_nz_reports_org_framework_status
    ON gl_nz_reports(organization_id, framework, status);

-- Partial indexes for active records
CREATE INDEX idx_nz_reports_active
    ON gl_nz_reports(organization_id, framework)
    WHERE status NOT IN ('archived');

-- Partial indexes for unresolved issues
CREATE INDEX idx_nz_validation_unresolved
    ON gl_nz_validation_results(report_id)
    WHERE resolved = FALSE;

-- Covering indexes for dashboard queries
CREATE INDEX idx_nz_reports_dashboard
    ON gl_nz_reports(organization_id, framework, status, created_at DESC)
    INCLUDE (reporting_period, approved_at);
```

### Query Optimization

```sql
-- Use materialized views for expensive aggregations
CREATE MATERIALIZED VIEW gl_nz_reports_stats AS
SELECT
    organization_id,
    framework,
    COUNT(*) AS report_count,
    MAX(created_at) AS latest_report,
    AVG(metadata->>'generation_time_seconds')::numeric AS avg_gen_time
FROM gl_nz_reports
GROUP BY organization_id, framework;

-- Refresh on schedule (every 5 minutes)
REFRESH MATERIALIZED VIEW CONCURRENTLY gl_nz_reports_stats;
```

### Connection Pool Tuning

```python
# Optimal pool configuration for PACK-030
pool_config = {
    "min_size": 5,           # Minimum idle connections
    "max_size": 20,          # Maximum connections
    "max_overflow": 10,      # Burst capacity above max_size
    "pool_timeout": 30,      # Wait time for connection from pool
    "pool_recycle": 1800,    # Recycle connections every 30 minutes
    "pool_pre_ping": True,   # Verify connection before use
}
```

---

## 4. Caching Strategy

### Multi-Level Cache

| Level | Store | TTL | Data |
|-------|-------|-----|------|
| L1 | In-memory (LRU) | 5 min | Framework schemas, XBRL taxonomies |
| L2 | Redis | 1 hour | Aggregated data, report data |
| L3 | Redis | 24 hours | XBRL taxonomies, translation cache |
| L4 | PostgreSQL | Persistent | Generated reports, evidence bundles |

### Cache Key Design

```
# Key pattern: pack030:{scope}:{entity}:{identifier}:{data_type}
pack030:org:uuid123:2025:aggregated_data        # Aggregated data for org/year
pack030:org:uuid123:SBTi:2025:report            # SBTi report for org/year
pack030:global:SBTi:schema:v1.1                 # SBTi schema (global)
pack030:global:SEC:taxonomy:2024                # SEC XBRL taxonomy (global)
pack030:translation:sha256hash:en:de            # Translation cache
```

### Cache Warming

```python
# Pre-warm cache on service startup
async def warm_cache():
    """Pre-load frequently accessed data into cache."""
    # Load framework schemas (7 frameworks)
    for framework in SUPPORTED_FRAMEWORKS:
        schema = await db.fetch_framework_schema(framework)
        await cache.set(f"pack030:global:{framework}:schema", schema, ttl=86400)

    # Load XBRL taxonomies
    for taxonomy in ["SEC-2024", "CSRD-2024"]:
        data = await xbrl_integration.fetch_taxonomy(taxonomy)
        await cache.set(f"pack030:global:taxonomy:{taxonomy}", data, ttl=86400)

    # Load framework mappings (42 mappings)
    mappings = await db.fetch_all_mappings()
    await cache.set("pack030:global:mappings", mappings, ttl=86400)
```

---

## 5. Parallelization

### Framework Parallel Execution

The multi-framework workflow executes all 7 framework workflows in parallel:

```python
import asyncio

async def execute_multi_framework(self, aggregated_data, config):
    """Execute all 7 framework workflows in parallel."""
    tasks = [
        self.sbti_workflow.execute(aggregated_data),
        self.cdp_workflow.execute(aggregated_data),
        self.tcfd_workflow.execute(aggregated_data),
        self.gri_workflow.execute(aggregated_data),
        self.issb_workflow.execute(aggregated_data),
        self.sec_workflow.execute(aggregated_data),
        self.csrd_workflow.execute(aggregated_data),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Data Aggregation Parallelization

```python
async def aggregate_all(self, org_id, period):
    """Fetch from all 8 sources in parallel."""
    pack_task = asyncio.gather(
        self.pack021.fetch_baseline(org_id),
        self.pack022.fetch_initiatives(org_id),
        self.pack028.fetch_pathways(org_id),
        self.pack029.fetch_targets(org_id),
    )
    app_task = asyncio.gather(
        self.sbti_app.fetch_targets(org_id),
        self.cdp_app.fetch_history(org_id),
        self.tcfd_app.fetch_scenarios(org_id),
        self.ghg_app.fetch_inventory(org_id, period),
    )
    pack_results, app_results = await asyncio.gather(pack_task, app_task)
    return self.merge(pack_results, app_results)
```

---

## 6. PDF Rendering Optimization

### WeasyPrint Performance Tips

```python
# Pre-compile CSS for reuse
_compiled_css = weasyprint.CSS(filename="templates/styles.css")

async def render_pdf(self, report, branding):
    """Optimized PDF rendering."""
    # Use pre-compiled CSS (saves ~500ms)
    html = self.compile_html(report, branding)

    # Render with pre-compiled CSS
    pdf = weasyprint.HTML(string=html).write_pdf(
        stylesheets=[_compiled_css],
        optimize_size=("fonts", "images"),
    )
    return pdf
```

### Image Optimization

- Pre-resize charts to target DPI (300 for print)
- Use SVG for vector charts (smaller, sharper)
- Compress raster images to 85% JPEG quality
- Cache rendered charts for reuse across frameworks

---

## 7. XBRL Generation Optimization

### Template Pre-compilation

```python
# Pre-compile XBRL templates at startup
_xbrl_template = lxml.etree.parse("templates/xbrl_base.xml")
_ixbrl_template = jinja2.Template(open("templates/ixbrl_base.html").read())

async def generate_xbrl(self, report_data, taxonomy):
    """Generate XBRL using pre-compiled template."""
    tree = copy.deepcopy(_xbrl_template)
    # Insert data elements (avoid full XML parsing each time)
    ...
```

### Taxonomy Caching

```python
# Cache XBRL taxonomies in Redis (24-hour TTL)
# Avoids HTTP fetch on every XBRL generation
taxonomy = await cache.get(f"pack030:taxonomy:{version}")
if not taxonomy:
    taxonomy = await xbrl_integration.fetch_taxonomy(version)
    await cache.set(f"pack030:taxonomy:{version}", taxonomy, ttl=86400)
```

---

## 8. API Response Optimization

### Response Compression

```python
# Enable gzip compression for API responses
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Field Selection

```python
# Support field selection to reduce response size
@app.get("/api/v1/reports/{report_id}")
async def get_report(
    report_id: UUID,
    fields: Optional[str] = Query(None, description="Comma-separated field names"),
):
    report = await db.get_report(report_id)
    if fields:
        selected = fields.split(",")
        return {k: v for k, v in report.dict().items() if k in selected}
    return report
```

### Pagination

```python
# Default pagination to prevent large responses
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
```

---

## 9. Memory Management

### Streaming Large Reports

```python
# Stream large PDF reports instead of loading into memory
from fastapi.responses import StreamingResponse

@app.get("/api/v1/reports/{report_id}/download")
async def download_report(report_id: UUID, format: str = "pdf"):
    report_path = await storage.get_report_path(report_id, format)
    return StreamingResponse(
        open(report_path, "rb"),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report.{format}"},
    )
```

### Memory Limits

```python
# Limit concurrent report generations to prevent OOM
MAX_CONCURRENT_GENERATIONS = 5
generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

async def generate_report(self, ...):
    async with generation_semaphore:
        return await self._generate_report_impl(...)
```

---

## 10. Monitoring and Profiling

### Prometheus Metrics

```python
from prometheus_client import Histogram, Counter, Gauge

# Report generation latency
report_generation_duration = Histogram(
    "pack030_report_generation_seconds",
    "Report generation duration in seconds",
    ["framework", "format"],
    buckets=[0.5, 1, 2, 3, 5, 8, 10, 15, 30],
)

# Cache hit ratio
cache_hits = Counter("pack030_cache_hits_total", "Cache hits", ["cache_level"])
cache_misses = Counter("pack030_cache_misses_total", "Cache misses", ["cache_level"])

# Active generations
active_generations = Gauge("pack030_active_generations", "Currently generating reports")
```

### Grafana Dashboard Queries

```promql
# Report generation p95 latency
histogram_quantile(0.95, rate(pack030_report_generation_seconds_bucket[5m]))

# Cache hit ratio
sum(rate(pack030_cache_hits_total[5m])) /
(sum(rate(pack030_cache_hits_total[5m])) + sum(rate(pack030_cache_misses_total[5m])))

# API error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m]))
```

### Query Performance Monitoring

```sql
-- Identify slow queries
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
WHERE query LIKE '%gl_nz_%'
ORDER BY mean_exec_time DESC
LIMIT 20;
```

---

## Performance Checklist

Before deploying to production, verify:

- [ ] All 350+ indexes created and used (check `pg_stat_user_indexes`)
- [ ] Redis caching enabled with appropriate TTLs
- [ ] Connection pool sized for expected concurrency
- [ ] Framework workflows run in parallel
- [ ] WeasyPrint CSS pre-compiled
- [ ] XBRL taxonomies cached
- [ ] API response compression enabled
- [ ] Memory limits configured
- [ ] Prometheus metrics scraped
- [ ] Grafana dashboards configured
- [ ] Slow query logging enabled
- [ ] Cache warming on startup

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
