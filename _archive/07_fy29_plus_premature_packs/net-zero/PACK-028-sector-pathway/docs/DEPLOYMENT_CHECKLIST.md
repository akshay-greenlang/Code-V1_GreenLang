# PACK-028 Sector Pathway Pack -- Deployment Checklist

**Pack ID:** PACK-028-sector-pathway
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Pre-Deployment Prerequisites](#pre-deployment-prerequisites)
2. [Infrastructure Verification](#infrastructure-verification)
3. [Database Migration Steps](#database-migration-steps)
4. [Reference Data Loading](#reference-data-loading)
5. [Environment Configuration](#environment-configuration)
6. [Health Check Procedures](#health-check-procedures)
7. [Integration Verification](#integration-verification)
8. [Smoke Testing](#smoke-testing)
9. [Performance Validation](#performance-validation)
10. [Security Checklist](#security-checklist)
11. [Rollback Procedures](#rollback-procedures)
12. [Post-Deployment Monitoring](#post-deployment-monitoring)
13. [Troubleshooting Guide](#troubleshooting-guide)
14. [Sign-Off](#sign-off)

---

## Pre-Deployment Prerequisites

### Platform Version Requirements

| Component | Minimum Version | Required | Verified |
|-----------|----------------|----------|----------|
| GreenLang Platform | v1.0.0 | Yes | [ ] |
| Python | 3.11+ | Yes | [ ] |
| PostgreSQL | 16.x | Yes | [ ] |
| TimescaleDB Extension | 2.13+ | Yes | [ ] |
| Redis | 7.x | Yes | [ ] |
| Kubernetes | 1.28+ | Yes (if K8s deploy) | [ ] |
| Docker | 24.x+ | Yes (if containerized) | [ ] |

### Platform Migrations

| Migration Range | Description | Applied | Verified |
|----------------|-------------|---------|----------|
| V001-V006 | Core platform tables | [ ] | [ ] |
| V007-V008 | Feature flags + Agent Factory | [ ] | [ ] |
| V009-V010 | Auth + RBAC | [ ] | [ ] |
| V011-V018 | Security components | [ ] | [ ] |
| V019-V020 | Observability | [ ] | [ ] |
| V021-V030 | AGENT-FOUND tables | [ ] | [ ] |
| V031-V050 | AGENT-DATA tables | [ ] | [ ] |
| V051-V081 | AGENT-MRV tables | [ ] | [ ] |
| V082-V088 | Application tables | [ ] | [ ] |
| V089-V128 | AGENT-EUDR tables | [ ] | [ ] |

### Dependent Packs (Optional)

| Pack | Version | Required | Installed | Verified |
|------|---------|----------|-----------|----------|
| PACK-021 Net Zero Starter | 1.0.0 | Recommended | [ ] | [ ] |
| PACK-022 Net Zero Acceleration | 1.0.0 | Optional | [ ] | [ ] |

### Verification Commands

```bash
# Check Python version
python --version
# Expected: Python 3.11.x or higher

# Check PostgreSQL version and extensions
psql -h $DB_HOST -U $DB_USER -d greenlang -c "SELECT version();"
psql -h $DB_HOST -U $DB_USER -d greenlang -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';"

# Check Redis version
redis-cli -h $REDIS_HOST INFO server | grep redis_version

# Check latest applied migration
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT version, description, applied_at FROM schema_migrations ORDER BY version DESC LIMIT 5;"

# Check Kubernetes version (if applicable)
kubectl version --short
```

---

## Infrastructure Verification

### Database

| Check | Command | Expected | Verified |
|-------|---------|----------|----------|
| Connection | `psql -h $DB_HOST -c "SELECT 1"` | Returns 1 | [ ] |
| TimescaleDB | `SELECT extversion FROM pg_extension WHERE extname='timescaledb'` | 2.13+ | [ ] |
| Disk space | `SELECT pg_database_size('greenlang')` | <80% capacity | [ ] |
| Max connections | `SHOW max_connections` | 100+ | [ ] |
| WAL archiving | `SHOW archive_mode` | on (production) | [ ] |

### Redis Cache

| Check | Command | Expected | Verified |
|-------|---------|----------|----------|
| Connection | `redis-cli -h $REDIS_HOST PING` | PONG | [ ] |
| Version | `redis-cli -h $REDIS_HOST INFO server` | 7.x | [ ] |
| Max memory | `redis-cli -h $REDIS_HOST CONFIG GET maxmemory` | 2GB+ | [ ] |
| Eviction policy | `redis-cli -h $REDIS_HOST CONFIG GET maxmemory-policy` | allkeys-lru | [ ] |

### Network

| Check | Description | Verified |
|-------|-------------|----------|
| Database accessible | Can connect from app server to DB | [ ] |
| Redis accessible | Can connect from app server to Redis | [ ] |
| PACK-021 reachable | Can reach PACK-021 API (if integrated) | [ ] |
| MRV agents reachable | Can reach MRV agent APIs | [ ] |
| DATA agents reachable | Can reach DATA agent APIs | [ ] |
| Outbound HTTPS | Can reach external APIs (SBTi, IEA) | [ ] |

---

## Database Migration Steps

### Step 1: Backup Current Database

```bash
# Create backup before migration
pg_dump -h $DB_HOST -U $DB_USER -d greenlang \
  --format=custom \
  --file=greenlang_pre_pack028_$(date +%Y%m%d_%H%M%S).backup

# Verify backup
pg_restore --list greenlang_pre_pack028_*.backup | head -20
echo "Backup verified: $(ls -lh greenlang_pre_pack028_*.backup)"
```

**Verified:** [ ] Backup created and verified

### Step 2: Apply PACK-028 Migrations

```bash
# Navigate to migration directory
cd packs/net-zero/PACK-028-sector-pathway/migrations

# Apply migrations sequentially
echo "Applying V181-PACK028-001: Sector Classifications..."
psql -h $DB_HOST -U $DB_USER -d greenlang -f V181-PACK028-001.sql
echo "Result: $?"

echo "Applying V181-PACK028-002: Sector Intensities..."
psql -h $DB_HOST -U $DB_USER -d greenlang -f V181-PACK028-002.sql
echo "Result: $?"

echo "Applying V181-PACK028-003: Sector Pathways..."
psql -h $DB_HOST -U $DB_USER -d greenlang -f V181-PACK028-003.sql
echo "Result: $?"

echo "Applying V181-PACK028-004: Sector Benchmarks..."
psql -h $DB_HOST -U $DB_USER -d greenlang -f V181-PACK028-004.sql
echo "Result: $?"

echo "Applying V181-PACK028-005: Technology Roadmaps..."
psql -h $DB_HOST -U $DB_USER -d greenlang -f V181-PACK028-005.sql
echo "Result: $?"

echo "Applying V181-PACK028-006: Abatement Waterfalls..."
psql -h $DB_HOST -U $DB_USER -d greenlang -f V181-PACK028-006.sql
echo "Result: $?"
```

**Verified:** [ ] All 6 migrations applied successfully

### Step 3: Verify Migrations

```bash
# Verify all tables created
psql -h $DB_HOST -U $DB_USER -d greenlang -c "
  SELECT table_name
  FROM information_schema.tables
  WHERE table_schema = 'sector_pathway'
  ORDER BY table_name;
"

# Expected tables:
# gl_abatement_waterfalls
# gl_sector_benchmarks
# gl_sector_classifications
# gl_sector_intensities
# gl_sector_pathways
# gl_technology_roadmaps

# Verify row-level security
psql -h $DB_HOST -U $DB_USER -d greenlang -c "
  SELECT tablename, rowsecurity
  FROM pg_tables
  WHERE schemaname = 'sector_pathway';
"
# All tables should show rowsecurity = true

# Verify indexes
psql -h $DB_HOST -U $DB_USER -d greenlang -c "
  SELECT indexname, tablename
  FROM pg_indexes
  WHERE schemaname = 'sector_pathway'
  ORDER BY tablename, indexname;
"
```

**Verified:** [ ] All tables, indexes, and RLS verified

---

## Reference Data Loading

### Step 1: Load SBTi SDA Data

```bash
# Verify SBTi data directory
ls -la $SECTOR_PATHWAY_SBTI_DATA_DIR/v3.0/

# Load SBTi reference data
python -c "
from integrations.sbti_sda_bridge import SBTiSDABridge
bridge = SBTiSDABridge(data_dir='$SECTOR_PATHWAY_SBTI_DATA_DIR')
status = bridge.load_and_verify()
print(f'Sectors loaded: {status.sectors_loaded}')
print(f'Convergence factors: {status.convergence_factors_count}')
print(f'Checksum verified: {status.checksums_valid}')
"
```

**Verified:** [ ] SBTi SDA data loaded (12 sectors, 504 factors)

### Step 2: Load IEA NZE Data

```bash
# Verify IEA data directory
ls -la $SECTOR_PATHWAY_IEA_DATA_DIR/2023_update/

# Load IEA reference data
python -c "
from integrations.iea_nze_bridge import IEANZEBridge
bridge = IEANZEBridge(data_dir='$SECTOR_PATHWAY_IEA_DATA_DIR')
status = bridge.load_and_verify()
print(f'Sectors covered: {status.sectors_covered}')
print(f'Milestones loaded: {status.milestone_count}')
print(f'Scenarios available: {status.scenarios_available}')
print(f'Checksum verified: {status.checksums_valid}')
"
```

**Verified:** [ ] IEA NZE data loaded (15 sectors, 428 milestones, 5 scenarios)

### Step 3: Load IPCC AR6 Data

```bash
python -c "
from integrations.ipcc_ar6_bridge import IPCCAR6Bridge
bridge = IPCCAR6Bridge(data_dir='$SECTOR_PATHWAY_IPCC_DATA_DIR')
status = bridge.load_and_verify()
print(f'GWP values loaded: {status.gwp_count}')
print(f'Emission factors: {status.emission_factor_count}')
print(f'Checksum verified: {status.checksums_valid}')
"
```

**Verified:** [ ] IPCC AR6 data loaded (42 GWP values, 1200+ emission factors)

---

## Environment Configuration

### Required Environment Variables

| Variable | Value | Set | Verified |
|----------|-------|-----|----------|
| `SECTOR_PATHWAY_DB_HOST` | Database hostname | [ ] | [ ] |
| `SECTOR_PATHWAY_DB_PORT` | Database port (5432) | [ ] | [ ] |
| `SECTOR_PATHWAY_DB_NAME` | Database name (greenlang) | [ ] | [ ] |
| `SECTOR_PATHWAY_REDIS_HOST` | Redis hostname | [ ] | [ ] |
| `SECTOR_PATHWAY_REDIS_PORT` | Redis port (6379) | [ ] | [ ] |
| `SECTOR_PATHWAY_LOG_LEVEL` | Log level (INFO) | [ ] | [ ] |
| `SECTOR_PATHWAY_PROVENANCE` | Provenance enabled (true) | [ ] | [ ] |
| `SECTOR_PATHWAY_SBTI_DATA_DIR` | SBTi data directory | [ ] | [ ] |
| `SECTOR_PATHWAY_IEA_DATA_DIR` | IEA data directory | [ ] | [ ] |
| `SECTOR_PATHWAY_IPCC_DATA_DIR` | IPCC data directory | [ ] | [ ] |

### Optional Environment Variables

| Variable | Value | Set | Verified |
|----------|-------|-----|----------|
| `SECTOR_PATHWAY_PACK021_ENABLED` | PACK-021 integration (true/false) | [ ] | [ ] |
| `SECTOR_PATHWAY_PACK021_BASE_URL` | PACK-021 API URL | [ ] | [ ] |
| `SECTOR_PATHWAY_CACHE_TTL` | Cache TTL in seconds (3600) | [ ] | [ ] |
| `SECTOR_PATHWAY_MAX_SCENARIOS` | Max concurrent scenarios (5) | [ ] | [ ] |

---

## Health Check Procedures

### Full Health Check

```python
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run()

print(f"Overall Score: {result.overall_score}/100")
print(f"Status: {result.status}")

all_pass = True
for category in result.categories:
    icon = "PASS" if category.score >= 90 else "WARN" if category.score >= 70 else "FAIL"
    print(f"  [{icon}] {category.name}: {category.score}/100")
    if category.score < 90:
        all_pass = False
        for issue in category.issues:
            print(f"         {issue.description}")

assert all_pass, "Health check has failures -- resolve before proceeding"
```

### Expected Health Check Results

| # | Category | Expected Score | Verified |
|---|----------|---------------|----------|
| 1 | Database Connectivity | 100 | [ ] |
| 2 | Redis Cache | 100 | [ ] |
| 3 | SBTi SDA Data | 100 | [ ] |
| 4 | IEA NZE Data | 100 | [ ] |
| 5 | IPCC AR6 Data | 100 | [ ] |
| 6 | MRV Agent Connectivity | 100 | [ ] |
| 7 | DATA Agent Connectivity | 100 | [ ] |
| 8 | FOUND Agent Connectivity | 100 | [ ] |
| 9 | Engine Availability | 100 | [ ] |
| 10 | Workflow Availability | 100 | [ ] |
| 11 | Template Availability | 100 | [ ] |
| 12 | Integration Availability | 100 | [ ] |
| 13 | Migration Status | 100 | [ ] |
| 14 | PACK-021 Bridge | 100 (or N/A) | [ ] |
| 15 | Sector Data Freshness | 100 | [ ] |
| 16 | Benchmark Data Freshness | 100 | [ ] |
| 17 | Emission Factor Data | 100 | [ ] |
| 18 | Cache Performance | 100 | [ ] |
| 19 | API Response Time | 100 | [ ] |
| 20 | Provenance Integrity | 100 | [ ] |

**Overall Health Score:** [ ] ___ /100
**Status:** [ ] HEALTHY

---

## Integration Verification

### Bridge Connectivity

| Bridge | Connected | Data Loaded | Verified |
|--------|-----------|-------------|----------|
| SBTi SDA Bridge | [ ] | [ ] | [ ] |
| IEA NZE Bridge | [ ] | [ ] | [ ] |
| IPCC AR6 Bridge | [ ] | [ ] | [ ] |
| PACK-021 Bridge | [ ] | [ ] (or N/A) | [ ] |
| MRV Bridge (30 agents) | [ ] | [ ] | [ ] |
| DATA Bridge (20 agents) | [ ] | [ ] | [ ] |
| FOUND Bridge (10 agents) | [ ] | [ ] | [ ] |
| Decarb Bridge | [ ] | [ ] | [ ] |

### Verification Commands

```python
# Verify all bridges
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run(categories=["integrations_only"])

for category in result.categories:
    print(f"  [{category.status}] {category.name}: {category.score}/100")
```

---

## Smoke Testing

### Smoke Test 1: Sector Classification

```python
from engines.sector_classification_engine import SectorClassificationEngine

engine = SectorClassificationEngine()
result = engine.classify({"nace_codes": ["C24.10"]})
assert result.primary_sector == "steel", f"Expected steel, got {result.primary_sector}"
assert result.sda_eligible == True
print("Smoke Test 1: PASS - Sector classification working")
```

**Verified:** [ ] Sector classification returns correct sector

### Smoke Test 2: Pathway Generation

```python
from engines.pathway_generator_engine import PathwayGeneratorEngine

engine = PathwayGeneratorEngine()
result = engine.generate(
    sector="steel",
    base_year=2023,
    base_year_intensity=1.85,
    target_year_near=2030,
    target_year_long=2050,
    scenario="nze_15c",
)
assert result.target_2030 == 1.25, f"Expected 1.25, got {result.target_2030}"
assert len(result.annual_pathway) > 0
print("Smoke Test 2: PASS - Pathway generation working")
```

**Verified:** [ ] Pathway generates correct 2030 target

### Smoke Test 3: Report Generation

```python
from templates.sector_pathway_report import SectorPathwayReport

report = SectorPathwayReport()
output = report.render(pathway=result, format="md")
assert len(output.content) > 1000, "Report too short"
print(f"Smoke Test 3: PASS - Report generated ({len(output.content)} chars)")
```

**Verified:** [ ] Report renders successfully

### Smoke Test 4: Full Workflow

```python
from workflows.sector_pathway_design_workflow import SectorPathwayDesignWorkflow
from config.pack_config import PackConfig

config = PackConfig.from_preset("heavy_industry")
workflow = SectorPathwayDesignWorkflow(config=config)
result = workflow.execute(
    company_profile={
        "name": "Smoke Test Corp",
        "nace_codes": ["C24.10"],
        "base_year": 2023,
        "base_year_production_tonnes": 1_000_000,
        "base_year_emissions_tco2e": 1_850_000,
    },
    target_scenario="nze_15c",
    target_year_near=2030,
    target_year_long=2050,
)
assert result.status == "completed", f"Workflow failed: {result.status}"
print("Smoke Test 4: PASS - Full workflow completed")
```

**Verified:** [ ] Full workflow executes successfully

---

## Performance Validation

### Latency Checks

| Operation | Target | Measured | Pass | Verified |
|-----------|--------|----------|------|----------|
| Sector classification | <2s | ___ s | [ ] | [ ] |
| Intensity calculation | <5s | ___ s | [ ] | [ ] |
| Pathway generation | <30s | ___ s | [ ] | [ ] |
| Convergence analysis | <10s | ___ s | [ ] | [ ] |
| Technology roadmap | <60s | ___ s | [ ] | [ ] |
| API response (p95) | <2s | ___ s | [ ] | [ ] |

---

## Security Checklist

| # | Check | Description | Verified |
|---|-------|-------------|----------|
| 1 | JWT authentication | All endpoints require valid JWT | [ ] |
| 2 | RBAC enforcement | Role-based access controls active | [ ] |
| 3 | TLS 1.3 | All API communication encrypted | [ ] |
| 4 | AES-256-GCM | Data at rest encrypted | [ ] |
| 5 | Provenance hashing | SHA-256 hashing enabled | [ ] |
| 6 | Audit trail | Immutable audit log active | [ ] |
| 7 | Reference data integrity | SHA-256 checksums verified | [ ] |
| 8 | RLS enabled | Row-level security on all pack tables | [ ] |
| 9 | Vault integration | Secrets stored in Vault | [ ] |
| 10 | Log sanitization | No PII or credentials in logs | [ ] |

---

## Rollback Procedures

### Step 1: Stop Services

```bash
# If using Kubernetes
kubectl scale deployment pack-028-sector-pathway --replicas=0

# If using systemd
systemctl stop pack-028-sector-pathway
```

### Step 2: Rollback Migrations

```bash
# Rollback in reverse order
psql -h $DB_HOST -U $DB_USER -d greenlang -f migrations/V181-PACK028-006.down.sql
psql -h $DB_HOST -U $DB_USER -d greenlang -f migrations/V181-PACK028-005.down.sql
psql -h $DB_HOST -U $DB_USER -d greenlang -f migrations/V181-PACK028-004.down.sql
psql -h $DB_HOST -U $DB_USER -d greenlang -f migrations/V181-PACK028-003.down.sql
psql -h $DB_HOST -U $DB_USER -d greenlang -f migrations/V181-PACK028-002.down.sql
psql -h $DB_HOST -U $DB_USER -d greenlang -f migrations/V181-PACK028-001.down.sql
```

### Step 3: Restore from Backup (if needed)

```bash
# Restore from pre-deployment backup
pg_restore -h $DB_HOST -U $DB_USER -d greenlang \
  --clean --if-exists \
  greenlang_pre_pack028_*.backup
```

---

## Post-Deployment Monitoring

### First 24 Hours

| Check | Frequency | Action |
|-------|-----------|--------|
| Health check score | Every 15 min | Alert if <90 |
| API response time (p95) | Every 5 min | Alert if >2s |
| Error rate | Every 5 min | Alert if >1% |
| Cache hit ratio | Every 30 min | Alert if <80% |
| Memory usage | Every 15 min | Alert if >80% ceiling |

### First 7 Days

| Check | Frequency | Action |
|-------|-----------|--------|
| Full health check | Daily | Review all 20 categories |
| Performance baseline | Daily | Establish p50/p95/p99 baselines |
| User feedback | Daily | Collect and address issues |
| Error log review | Daily | Review and categorize errors |
| Cache performance | Daily | Tune TTL if needed |

---

## Troubleshooting Guide

### Migration Failures

**Problem:** Migration script fails with "table already exists"
```bash
# Check if partial migration was applied
psql -h $DB_HOST -U $DB_USER -d greenlang -c "
  SELECT table_name FROM information_schema.tables
  WHERE table_schema = 'sector_pathway';
"

# Drop partial tables and re-apply
psql -h $DB_HOST -U $DB_USER -d greenlang -c "
  DROP SCHEMA sector_pathway CASCADE;
  CREATE SCHEMA sector_pathway;
"
# Then re-run migrations
```

### Health Check Failures

**Problem:** SBTi SDA Data score below 100
```bash
# Check data directory
ls -la $SECTOR_PATHWAY_SBTI_DATA_DIR/v3.0/pathways/
# Verify all 12 sector CSV files exist

# Verify checksums
python -c "
from integrations.sbti_sda_bridge import SBTiSDABridge
bridge = SBTiSDABridge()
bridge.verify_checksums()
"
```

**Problem:** MRV Agent Connectivity score below 100
```bash
# Check which agents are unavailable
python -c "
from integrations.mrv_bridge import MRVBridge
bridge = MRVBridge()
status = bridge.verify()
for agent in status.agents:
    if not agent.available:
        print(f'UNAVAILABLE: {agent.agent_id}')
"
```

### Performance Issues

**Problem:** Pathway generation exceeds 30 second target
```bash
# Check Redis cache performance
redis-cli -h $REDIS_HOST INFO stats | grep -E "keyspace_hits|keyspace_misses"

# Clear and warm cache
python -c "
from integrations.health_check import HealthCheck
hc = HealthCheck()
hc.warm_cache()
"
```

---

## Sign-Off

### Deployment Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| DevOps Engineer | ________________ | ____/____/2026 | _________ |
| QA Lead | ________________ | ____/____/2026 | _________ |
| Product Owner | ________________ | ____/____/2026 | _________ |
| Security Reviewer | ________________ | ____/____/2026 | _________ |

### Deployment Summary

| Item | Value |
|------|-------|
| Deployment Date | ____/____/2026 |
| Environment | [ ] Development [ ] Staging [ ] Production |
| Version Deployed | 1.0.0 |
| Health Check Score | ____/100 |
| Smoke Tests | [ ] All Passed |
| Performance | [ ] All Targets Met |
| Security | [ ] All Checks Passed |
| Rollback Plan | [ ] Tested and Ready |
| Monitoring | [ ] Configured and Active |

---

**End of Deployment Checklist**
