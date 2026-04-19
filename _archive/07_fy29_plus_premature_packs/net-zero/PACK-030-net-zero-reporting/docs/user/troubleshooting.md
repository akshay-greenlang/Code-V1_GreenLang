# PACK-030: Troubleshooting Guide

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Installation Issues](#1-installation-issues)
2. [Database Issues](#2-database-issues)
3. [Data Aggregation Issues](#3-data-aggregation-issues)
4. [Report Generation Issues](#4-report-generation-issues)
5. [Format Rendering Issues](#5-format-rendering-issues)
6. [XBRL/iXBRL Issues](#6-xbrlixbrl-issues)
7. [Narrative Generation Issues](#7-narrative-generation-issues)
8. [Translation Issues](#8-translation-issues)
9. [Dashboard Issues](#9-dashboard-issues)
10. [Performance Issues](#10-performance-issues)
11. [Authentication Issues](#11-authentication-issues)
12. [Integration Issues](#12-integration-issues)
13. [Diagnostic Commands](#13-diagnostic-commands)

---

## 1. Installation Issues

### WeasyPrint Installation Fails

**Symptom:** `ImportError: cannot import name 'HTML' from 'weasyprint'`

**Cause:** WeasyPrint requires system-level dependencies (Pango, GDK-Pixbuf, Cairo).

**Solution:**

```bash
# Ubuntu/Debian
sudo apt-get install -y libpango-1.0-0 libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 libcairo2 libffi-dev

# macOS
brew install pango cairo gdk-pixbuf libffi

# Windows
# Install GTK3 runtime from https://github.com/nickvdyck/weasyprint-win
```

### Missing Python Dependencies

**Symptom:** `ModuleNotFoundError: No module named 'psycopg'`

**Solution:**

```bash
pip install -r requirements.txt

# If psycopg binary fails, install from source
pip install "psycopg[c]>=3.1.0"
```

### TimescaleDB Extension Not Found

**Symptom:** `ERROR: extension "timescaledb" is not available`

**Solution:**

```bash
# Install TimescaleDB extension
sudo apt-get install timescaledb-2-postgresql-16

# Enable in postgresql.conf
echo "shared_preload_libraries = 'timescaledb'" >> /etc/postgresql/16/main/postgresql.conf

# Restart PostgreSQL
sudo systemctl restart postgresql
```

---

## 2. Database Issues

### Migration Fails at V219 (Indexes)

**Symptom:** `ERROR: relation "gl_nz_reports" does not exist`

**Cause:** Previous migrations (V211-V218) did not complete successfully.

**Solution:**

```bash
# Check current migration version
python scripts/check_migration_version.py

# Re-apply from the beginning
python scripts/apply_migrations.py --start V211 --end V225 --force
```

### RLS Policy Blocks Queries

**Symptom:** `ERROR: new row violates row-level security policy`

**Cause:** The `app.current_organization_id` session variable is not set.

**Solution:**

```sql
-- Set the organization ID before queries
SET app.current_organization_id = 'your-org-uuid';

-- Or in Python
await conn.execute("SET app.current_organization_id = $1", org_id)
```

### Connection Pool Exhausted

**Symptom:** `TimeoutError: Connection pool exhausted`

**Solution:**

```bash
# Increase pool size
export DATABASE_POOL_SIZE=40
export DATABASE_MAX_OVERFLOW=20

# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'greenlang';"
```

---

## 3. Data Aggregation Issues

### Pack Connection Timeout

**Symptom:** `TimeoutError: PACK-021 did not respond within 30s`

**Cause:** Prerequisite pack is not running or network is slow.

**Solution:**

```bash
# Verify pack is running
curl http://pack-021:9021/health

# Increase timeout
export DATA_AGGREGATION_TIMEOUT=60

# Check network connectivity
ping pack-021
```

### Data Reconciliation Mismatch

**Symptom:** `ReconciliationError: Scope 1 total mismatch: PACK-021 (45000) vs GL-GHG-APP (45250)`

**Cause:** Source systems have different data due to timing or methodology differences.

**Solution:**

```python
# Review reconciliation report
result = await engine.aggregate_all(organization_id="your-org-uuid", ...)
for issue in result.reconciliation_issues:
    print(f"Source A: {issue.source_a} = {issue.value_a}")
    print(f"Source B: {issue.source_b} = {issue.value_b}")
    print(f"Difference: {issue.difference}")

# Accept the higher-confidence source
await engine.resolve_reconciliation(
    issue_id=issue.id,
    resolution="accept_source_a",
    reason="PACK-021 is the master source for baseline data"
)
```

### Missing Data Points

**Symptom:** `DataGapWarning: 15 required metrics missing for CDP framework`

**Solution:**

```python
# Check which metrics are missing
gaps = await engine.get_data_gaps(
    organization_id="your-org-uuid",
    framework="CDP",
)
for gap in gaps:
    print(f"Missing: {gap.metric_name} (required by {gap.framework})")
    print(f"Suggested source: {gap.suggested_source}")
```

---

## 4. Report Generation Issues

### Report Generation Exceeds 10s Timeout

**Symptom:** `TimeoutError: Multi-framework report generation exceeded 10s`

**Solution:**

```bash
# Increase timeout
export REPORT_GENERATION_TIMEOUT=30

# Enable caching to speed up subsequent runs
export CACHE_ENABLED=true
export CACHE_TTL=3600

# Reduce frameworks for faster generation
# Generate only needed frameworks instead of all 7
```

### Schema Validation Failure

**Symptom:** `ValidationError: Report does not conform to SBTi schema v1.1`

**Solution:**

```python
# Get detailed validation errors
validation = await validator.validate(
    report_data=report.data,
    framework="SBTi",
)
for error in validation.errors:
    print(f"Field: {error.field_path}")
    print(f"Error: {error.message}")
    print(f"Expected: {error.expected}")
    print(f"Actual: {error.actual}")
```

### Consistency Score Below 95%

**Symptom:** `ConsistencyWarning: Cross-framework consistency score is 87%`

**Solution:**

```python
# Get specific inconsistencies
consistency = await validator.validate_consistency(
    reports=multi_result.reports,
)
for issue in consistency.issues:
    print(f"Inconsistency: {issue.description}")
    print(f"  {issue.framework_a}: {issue.value_a}")
    print(f"  {issue.framework_b}: {issue.value_b}")
    print(f"  Suggestion: {issue.suggestion}")
```

---

## 5. Format Rendering Issues

### PDF Rendering Fails

**Symptom:** `RenderingError: PDF generation failed`

**Cause:** WeasyPrint dependency issue or invalid HTML template.

**Solution:**

```bash
# Test WeasyPrint directly
python -c "from weasyprint import HTML; HTML(string='<h1>Test</h1>').write_pdf('test.pdf')"

# Check for CSS/HTML errors
python scripts/validate_templates.py
```

### Excel Export Missing Charts

**Symptom:** Charts not appearing in exported Excel file.

**Solution:**

```python
# Ensure openpyxl chart support
result = await renderer.render_excel(
    report=report,
    include_charts=True,
    chart_engine="openpyxl",
)
```

### HTML Dashboard Not Interactive

**Symptom:** Charts are static images instead of interactive.

**Solution:**

```python
# Ensure JavaScript is included
result = await renderer.render_html(
    report=report,
    include_javascript=True,
    chart_library="plotly",
)
```

---

## 6. XBRL/iXBRL Issues

### XBRL Taxonomy Validation Fails

**Symptom:** `XBRLError: Element 'ghg:Scope1Emissions' not found in taxonomy`

**Cause:** XBRL taxonomy cache is outdated.

**Solution:**

```bash
# Update taxonomy cache
python scripts/update_xbrl_taxonomies.py

# Or force refresh
python scripts/update_xbrl_taxonomies.py --force

# Verify taxonomy version
python scripts/check_taxonomy_version.py
```

### iXBRL Rendering Issues

**Symptom:** iXBRL tags not visible in browser viewer.

**Solution:**

```python
# Enable tag highlighting for debugging
result = await renderer.render_ixbrl(
    report=report,
    taxonomy="SEC-2024",
    highlight_tags=True,
)
```

---

## 7. Narrative Generation Issues

### Narrative Contains Contradictions

**Symptom:** Consistency validator flags contradictory statements.

**Solution:**

```python
# Review and fix contradictions
consistency = await narrative_engine.validate_consistency(
    narratives=[tcfd_narrative, cdp_narrative],
)
for contradiction in consistency.contradictions:
    print(f"Statement A: {contradiction.statement_a}")
    print(f"Statement B: {contradiction.statement_b}")
    print(f"Suggestion: {contradiction.harmonization_suggestion}")

# Regenerate with consistency enforced
narrative = await narrative_engine.generate_narrative(
    section_type="governance",
    framework="TCFD",
    consistency_context=existing_narratives,
)
```

### Citations Missing

**Symptom:** Narrative text has claims without source citations.

**Solution:**

```python
# Enable strict citation mode
narrative = await narrative_engine.generate_narrative(
    section_type="metrics",
    framework="TCFD",
    require_citations=True,
    source_data=aggregated_data,
)
```

---

## 8. Translation Issues

### Translation Quality Low

**Symptom:** Translation quality score below 98%.

**Solution:**

```python
# Use professional translation service
from packs.net_zero.pack030.engines import TranslationEngine

engine = TranslationEngine(config=config)
translation = await engine.translate_narrative(
    source_text=narrative.content,
    source_language="en",
    target_language="de",
    quality_threshold=0.98,
    use_glossary=True,
)

# Review translation
if translation.quality_score < 0.98:
    print("Translation quality below threshold")
    for issue in translation.quality_issues:
        print(f"  {issue.original} -> {issue.translated} ({issue.issue})")
```

### Climate Terminology Incorrect

**Symptom:** Technical climate terms translated incorrectly.

**Solution:**

```bash
# Update climate glossary
python scripts/update_climate_glossary.py

# Add custom terms to glossary
python scripts/add_glossary_term.py \
  --term "net-zero" \
  --en "net-zero" \
  --de "Netto-Null" \
  --fr "neutralite carbone" \
  --es "cero neto"
```

---

## 9. Dashboard Issues

### Dashboard Not Loading

**Symptom:** Dashboard shows loading spinner indefinitely.

**Solution:**

```bash
# Check API connectivity
curl http://localhost:8030/api/v1/dashboards/executive?organization_id=your-org-uuid

# Check browser console for JavaScript errors
# Ensure Plotly CDN is accessible
```

### Heatmap Data Missing

**Symptom:** Framework coverage heatmap shows no data.

**Solution:**

```bash
# Verify reports exist for the organization
curl http://localhost:8030/api/v1/reports?organization_id=your-org-uuid

# Generate reports first, then refresh dashboard
```

---

## 10. Performance Issues

### Slow Report Generation

**Solution:**

```bash
# Enable Redis caching
export CACHE_ENABLED=true
export REDIS_URL=redis://localhost:6379/0

# Enable parallel framework execution
export FRAMEWORK_PARALLEL=true
export MAX_WORKERS=8

# Pre-warm cache
python scripts/prewarm_cache.py
```

### High Memory Usage

**Solution:**

```bash
# Reduce parallel workers
export MAX_WORKERS=4

# Limit concurrent report generations
export MAX_CONCURRENT_REPORTS=5

# Enable streaming for large reports
export STREAMING_ENABLED=true
```

### Slow Database Queries

**Solution:**

```bash
# Check for missing indexes
python scripts/check_indexes.py

# Analyze query performance
python scripts/query_analyzer.py

# Vacuum and analyze tables
psql -c "VACUUM ANALYZE gl_nz_reports;"
psql -c "VACUUM ANALYZE gl_nz_report_metrics;"
```

---

## 11. Authentication Issues

### Token Expired

**Symptom:** `401 Unauthorized: Token has expired`

**Solution:** Refresh the token using the refresh endpoint or re-authenticate.

### Insufficient Permissions

**Symptom:** `403 Forbidden: Missing permission report:write`

**Solution:** Contact your administrator to assign the required role/permission.

---

## 12. Integration Issues

### Circuit Breaker Open

**Symptom:** `CircuitBreakerOpen: PACK-021 circuit breaker is open`

**Cause:** Too many consecutive failures to PACK-021.

**Solution:**

```bash
# Check PACK-021 health
curl http://pack-021:9021/health

# Wait for circuit breaker to half-open (default: 60s)
# Or reset manually
python scripts/reset_circuit_breaker.py --integration pack021
```

### Rate Limit Exceeded on External API

**Symptom:** `RateLimitError: DeepL API rate limit exceeded`

**Solution:**

```bash
# Reduce translation request rate
export TRANSLATION_RATE_LIMIT=5  # requests per second

# Use cached translations
export TRANSLATION_CACHE_ENABLED=true
```

---

## 13. Diagnostic Commands

```bash
# Full system diagnostics
python scripts/diagnostics.py

# Check all integration connections
python scripts/check_integrations.py

# Verify database schema
python scripts/verify_migrations.py --version V225

# Test report generation (dry run)
python scripts/smoke_test.py --dry-run

# Export diagnostic report
python scripts/diagnostics.py --export diagnostics_report.json

# Check XBRL taxonomy status
python scripts/check_taxonomy_version.py

# Validate all templates
python scripts/validate_templates.py

# Check cache status
python scripts/check_cache.py
```

---

## Getting Support

If the troubleshooting steps above do not resolve your issue:

1. Run `python scripts/diagnostics.py --export` and save the output
2. Check the application logs: `journalctl -u pack-030 -f`
3. Review the Grafana dashboard for anomalies
4. Consult the `docs/technical/architecture.md` for system design details
5. Contact `net-zero-team@greenlang.io` with the diagnostic report

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
