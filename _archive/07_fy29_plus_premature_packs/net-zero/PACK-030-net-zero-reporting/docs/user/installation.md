# PACK-030: Installation Guide

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Database Setup](#database-setup)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Upgrade Guide](#upgrade-guide)
8. [Uninstallation](#uninstallation)

---

## 1. System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Disk | 50 GB SSD | 100+ GB SSD |
| Network | 100 Mbps | 1 Gbps |

### Software Requirements

| Software | Minimum Version | Recommended Version |
|----------|----------------|---------------------|
| Python | 3.11 | 3.12+ |
| PostgreSQL | 16.0 | 16.2+ |
| TimescaleDB | 2.13 | 2.14+ |
| Redis | 7.0 | 7.2+ |
| Kubernetes | 1.28 | 1.29+ |
| Docker | 24.0 | 25.0+ |
| WeasyPrint | 60.0 | 61.0+ |

### Python Dependencies

```
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.5.0
psycopg[binary]>=3.1.0
psycopg_pool>=3.1.0
redis>=5.0.0
httpx>=0.26.0
weasyprint>=60.0
openpyxl>=3.1.0
jinja2>=3.1.0
plotly>=5.18.0
lxml>=5.1.0
PyYAML>=6.0
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
```

---

## 2. Prerequisites

### Required Packs

PACK-030 requires the following packs to be installed and operational:

| Pack | ID | Purpose | Required Version |
|------|----|---------|-----------------|
| Net Zero Starter Pack | PACK-021 | Baseline emissions, inventory data | 1.0.0+ |
| Net Zero Acceleration Pack | PACK-022 | Reduction initiatives, MACC curves | 1.0.0+ |
| Sector Pathway Pack | PACK-028 | Sector pathways, convergence data | 1.0.0+ |
| Interim Targets Pack | PACK-029 | Interim targets, progress monitoring | 1.0.0+ |

### Required Applications

| Application | Purpose | API Endpoint |
|-------------|---------|-------------|
| GL-SBTi-APP | SBTi target management | `/api/v1/sbti/` |
| GL-CDP-APP | CDP questionnaire management | `/api/v1/cdp/` |
| GL-TCFD-APP | TCFD scenario analysis | `/api/v1/tcfd/` |
| GL-GHG-APP | GHG inventory management | `/api/v1/ghg/` |

### Verify Prerequisites

```bash
# Check Python version
python --version  # Must be 3.11+

# Check PostgreSQL version
psql --version  # Must be 16+

# Check Redis version
redis-server --version  # Must be 7+

# Verify prerequisite packs
python -m greenlang.packs verify PACK-021
python -m greenlang.packs verify PACK-022
python -m greenlang.packs verify PACK-028
python -m greenlang.packs verify PACK-029

# Verify application availability
curl -s http://localhost:8001/api/v1/sbti/health
curl -s http://localhost:8002/api/v1/cdp/health
curl -s http://localhost:8003/api/v1/tcfd/health
curl -s http://localhost:8004/api/v1/ghg/health
```

---

## 3. Installation Steps

### Step 1: Clone or Copy Pack Files

```bash
# If installing from repository
cd packs/net-zero/
git pull origin master

# Verify pack directory exists
ls PACK-030-net-zero-reporting/
```

### Step 2: Install Python Dependencies

```bash
cd packs/net-zero/PACK-030-net-zero-reporting

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install WeasyPrint system dependencies (Ubuntu/Debian)
sudo apt-get install -y libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0

# Install WeasyPrint system dependencies (macOS)
brew install pango

# Install WeasyPrint system dependencies (Windows)
# Follow WeasyPrint Windows installation guide
```

### Step 3: Configure Environment Variables

```bash
# Create .env file from template
cp .env.example .env

# Edit with your settings
nano .env
```

Required environment variables:

```env
# Database
DATABASE_URL=postgresql://greenlang:password@localhost:5432/greenlang
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_TTL=3600

# Application endpoints
SBTI_APP_URL=http://localhost:8001
CDP_APP_URL=http://localhost:8002
TCFD_APP_URL=http://localhost:8003
GHG_APP_URL=http://localhost:8004

# Pack endpoints
PACK021_URL=http://localhost:9021
PACK022_URL=http://localhost:9022
PACK028_URL=http://localhost:9028
PACK029_URL=http://localhost:9029

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=RS256
ENCRYPTION_KEY=your-256-bit-key

# Translation (optional)
DEEPL_API_KEY=your-deepl-key
GOOGLE_TRANSLATE_KEY=your-google-key

# Storage
S3_BUCKET=greenlang-reports
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# XBRL
SEC_TAXONOMY_URL=https://xbrl.sec.gov/taxonomy/
CSRD_TAXONOMY_URL=https://www.efrag.org/taxonomy/
```

### Step 4: Apply Database Migrations

```bash
# Apply all 15 migrations (V211-V225)
python scripts/apply_migrations.py --start V211 --end V225

# Expected output:
# Applying V211__PACK030_core_tables.sql ... OK
# Applying V212__PACK030_framework_tables.sql ... OK
# Applying V213__PACK030_narrative_tables.sql ... OK
# Applying V214__PACK030_assurance_tables.sql ... OK
# Applying V215__PACK030_audit_tables.sql ... OK
# Applying V216__PACK030_xbrl_tables.sql ... OK
# Applying V217__PACK030_validation_tables.sql ... OK
# Applying V218__PACK030_config_tables.sql ... OK
# Applying V219__PACK030_indexes.sql ... OK
# Applying V220__PACK030_views.sql ... OK
# Applying V221__PACK030_rls_policies.sql ... OK
# Applying V222__PACK030_functions.sql ... OK
# Applying V223__PACK030_triggers.sql ... OK
# Applying V224__PACK030_seed_data.sql ... OK
# Applying V225__PACK030_permissions.sql ... OK
# All 15 migrations applied successfully.
```

### Step 5: Seed Reference Data

```bash
# Load framework schemas, mappings, and deadlines
python scripts/seed_reference_data.py

# Expected output:
# Loading framework schemas ... 7 frameworks loaded
# Loading framework mappings ... 42 mappings loaded
# Loading deadline calendar ... 7 deadlines loaded
# Loading XBRL taxonomies ... 2 taxonomies cached
# Reference data seeded successfully.
```

### Step 6: Start the Service

```bash
# Development mode
python -m uvicorn pack030.main:app --host 0.0.0.0 --port 8030 --reload

# Production mode
python -m uvicorn pack030.main:app --host 0.0.0.0 --port 8030 --workers 4

# Docker mode
docker build -t greenlang/pack-030-net-zero-reporting:1.0.0 .
docker run -p 8030:8030 --env-file .env greenlang/pack-030-net-zero-reporting:1.0.0
```

---

## 4. Database Setup

### Tables Created (15)

| Table | Purpose |
|-------|---------|
| `gl_nz_reports` | Report metadata and status tracking |
| `gl_nz_report_sections` | Report sections with narrative content |
| `gl_nz_report_metrics` | Quantitative metrics with provenance |
| `gl_nz_narratives` | Narrative library across frameworks |
| `gl_nz_framework_mappings` | Cross-framework metric mappings |
| `gl_nz_framework_schemas` | Framework JSON schemas |
| `gl_nz_framework_deadlines` | Reporting deadline calendar |
| `gl_nz_assurance_evidence` | Assurance evidence files |
| `gl_nz_data_lineage` | Source-to-report data lineage |
| `gl_nz_audit_trail` | Immutable audit log |
| `gl_nz_translations` | Translation cache |
| `gl_nz_xbrl_tags` | XBRL tag assignments |
| `gl_nz_validation_results` | Validation errors and warnings |
| `gl_nz_report_config` | Per-organization framework configuration |
| `gl_nz_dashboard_views` | Dashboard view configurations |

### Views Created (5)

| View | Purpose |
|------|---------|
| `gl_nz_reports_summary` | Report overview with section/metric counts |
| `gl_nz_framework_coverage` | Framework completeness percentage |
| `gl_nz_validation_issues` | Open validation issues by severity |
| `gl_nz_upcoming_deadlines` | Upcoming deadlines with days remaining |
| `gl_nz_lineage_summary` | Data lineage summary per metric |

### Verify Database

```bash
# Verify migrations
python scripts/verify_migrations.py --version V225

# Expected output:
# Tables: 15/15 created
# Views: 5/5 created
# Indexes: 350+ created
# RLS Policies: 30/30 enabled
# Functions: All created
# Triggers: All created
# Seed data: All loaded
# Database verification: PASSED
```

---

## 5. Configuration

### Minimal Configuration

```yaml
# config/pack_config.yaml
pack_id: "PACK-030-net-zero-reporting"
organization_id: "your-org-uuid"
frameworks:
  - SBTi
  - CDP
  - TCFD
languages:
  - en
output_formats:
  - PDF
  - HTML
```

### Production Configuration

See `docs/user/configuration.md` for the full configuration reference including:
- Framework-specific settings
- Branding customization
- Notification channels
- Assurance settings
- Performance tuning

---

## 6. Verification

### Run Health Check

```bash
curl http://localhost:8030/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "pack_id": "PACK-030-net-zero-reporting",
#   "database": "connected",
#   "redis": "connected",
#   "packs": {
#     "PACK-021": "available",
#     "PACK-022": "available",
#     "PACK-028": "available",
#     "PACK-029": "available"
#   },
#   "applications": {
#     "GL-SBTi-APP": "available",
#     "GL-CDP-APP": "available",
#     "GL-TCFD-APP": "available",
#     "GL-GHG-APP": "available"
#   }
# }
```

### Run Test Suite

```bash
# Full test suite
pytest tests/ -v --cov=. --cov-report=term

# Expected: 2,020+ tests, 90%+ coverage
# Run time: <5 minutes
```

### Smoke Test

```bash
# Generate a test report
python scripts/smoke_test.py

# Expected output:
# Generating SBTi progress report ... OK (2.1s)
# Generating CDP questionnaire ... OK (3.5s)
# Generating TCFD disclosure ... OK (2.8s)
# Generating GRI 305 disclosure ... OK (1.9s)
# Generating ISSB IFRS S2 ... OK (2.3s)
# Generating SEC climate disclosure ... OK (2.7s)
# Generating CSRD ESRS E1 ... OK (3.1s)
# Generating executive dashboard ... OK (1.8s)
# Generating evidence bundle ... OK (2.4s)
# All smoke tests passed.
```

---

## 7. Upgrade Guide

### From Pre-release to 1.0.0

```bash
# Back up database
pg_dump greenlang > backup_pre_v1.sql

# Pull latest code
git pull origin master

# Apply any new migrations
python scripts/apply_migrations.py --start V211 --end V225

# Restart service
systemctl restart pack-030
```

### Rolling Updates (Kubernetes)

```bash
# Update container image
kubectl set image deployment/pack-030 \
  pack-030=greenlang/pack-030-net-zero-reporting:1.0.1 \
  -n pack-030-production

# Monitor rollout
kubectl rollout status deployment/pack-030 -n pack-030-production
```

---

## 8. Uninstallation

### Remove Pack Service

```bash
# Stop service
systemctl stop pack-030

# Remove Docker container
docker stop pack-030 && docker rm pack-030

# Kubernetes
kubectl delete -f kubernetes/ -n pack-030-production
```

### Remove Database Objects

```bash
# Run down migrations (V225 to V211)
python scripts/apply_migrations.py --down --start V225 --end V211

# WARNING: This will permanently delete all PACK-030 data!
```

### Clean Up Files

```bash
# Remove pack directory
rm -rf packs/net-zero/PACK-030-net-zero-reporting/

# Remove cached data
redis-cli KEYS "pack030:*" | xargs redis-cli DEL
```

---

## Support

For installation issues:
- Check `docs/user/troubleshooting.md` for common problems
- Review system logs: `journalctl -u pack-030 -f`
- Kubernetes logs: `kubectl logs -f deployment/pack-030 -n pack-030-production`
- Run diagnostics: `python scripts/diagnostics.py`

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
