# PACK-027 Enterprise Net Zero Pack - Deployment Checklist

**Version**: 1.0.0
**Date**: March 2026
**Status**: Production Ready

---

## Overview

This checklist guides you through deploying PACK-027: Enterprise Net Zero Pack to production. The pack includes 12 calculation engines, 10 workflows, 12 templates, and 16 integrations for enterprise-grade net zero management.

**Total Deployment Time**: ~30-45 minutes
**Prerequisites**: Docker Desktop, PostgreSQL, Kubernetes cluster

---

## Pre-Deployment Checklist

### ✅ Prerequisites Verification

- [ ] **Docker Desktop** is installed and running
  ```bash
  docker info
  # Should show: Server Version: 20.10+
  ```

- [ ] **PostgreSQL** is accessible
  ```bash
  psql postgresql://greenlang:greenlang@localhost:5432/greenlang -c "SELECT version();"
  # Should connect successfully
  ```

- [ ] **Kubernetes cluster** is configured
  ```bash
  kubectl cluster-info
  # Should show cluster endpoint
  ```

- [ ] **Python 3.11+** with required packages
  ```bash
  python --version
  pip install -r requirements.txt
  ```

- [ ] **Database connection** credentials are set
  ```bash
  export DATABASE_URL="postgresql://greenlang:greenlang@localhost:5432/greenlang"
  export API_URL="http://localhost:8000"
  ```

---

## Deployment Steps

### Step 1: Database Migrations (V166-V180)

**Duration**: 5-10 minutes
**Critical**: Yes - Must complete before deployment

#### 1.1 Dry-Run Verification (Recommended)

```bash
cd packs/net-zero/PACK-027-enterprise-net-zero
python scripts/apply_migrations.py --dry-run
```

**Expected Output**:
```
=======================================================================
PACK-027 Database Migrations (V166-V180)
=======================================================================

✓ schema_migrations table ready
Applying V166: PACK027_enterprise_schema_and_profiles...
  [DRY RUN] Would apply V166: PACK027_enterprise_schema_and_profiles
    SQL length: 15234 bytes
...
[DRY RUN] No changes made to database
```

#### 1.2 Apply Migrations

```bash
python scripts/apply_migrations.py
```

**Expected Output**:
```
=======================================================================
MIGRATION SUMMARY
=======================================================================

  Total migrations: 15
  Applied:          15
  Skipped:          0

✅ Successfully applied 15 migrations
```

**Tables Created**:
- `gl_enterprise_profiles` - Enterprise organization profiles
- `gl_multi_entity_hierarchy` - Multi-entity structures
- `gl_comprehensive_baselines` - Historical baselines
- `gl_sbti_targets` - Science-based targets
- `gl_scenario_models` - Scenario modeling
- `gl_carbon_pricing` - Internal carbon pricing
- `gl_scope4_avoided_emissions` - Scope 4 projects
- `gl_supply_chain_mapping` - Supply chain data
- `gl_financial_integration` - Financial system links
- `gl_enterprise_risk_assessments` - Climate risks
- `gl_regulatory_compliance` - Regulatory tracking
- `gl_assurance_records` - Assurance documentation
- `gl_board_reporting` - Board-level reports
- `gl_enterprise_data_quality` - DQ tracking
- Plus 3 views and 200+ indexes

#### 1.3 Verify Migrations

```bash
python scripts/verify_migrations.py
```

**Expected Output**:
```
=======================================================================
PACK-027 Migration Verification
=======================================================================

✓ All 15 migrations applied successfully
✓ 18 tables exist
✓ 200+ indexes created
✓ 36 RLS policies active
✓ 3 views functional

✅ Database schema ready for deployment
```

**Troubleshooting**:
- If migration fails: Check `schema_migrations` table for partial completion
- If table exists error: Run verify_migrations.py to check current state
- If connection fails: Verify DATABASE_URL and PostgreSQL service

---

### Step 2: Docker Container Build

**Duration**: 10-15 minutes
**Critical**: Yes - Required for Kubernetes deployment

#### 2.1 Build Container Image

```bash
docker build -t greenlang/pack-027-enterprise-net-zero:1.0.0 .
```

**Expected Output**:
```
[+] Building 45.2s (15/15) FINISHED
 => [internal] load build definition
 => => transferring dockerfile: 1.2kB
 ...
 => exporting to image
 => => naming to greenlang/pack-027-enterprise-net-zero:1.0.0
```

#### 2.2 Verify Container

```bash
docker images greenlang/pack-027-enterprise-net-zero:1.0.0
```

**Expected Output**:
```
REPOSITORY                              TAG       IMAGE ID       CREATED         SIZE
greenlang/pack-027-enterprise-net-zero  1.0.0     abc123def456   1 minute ago    850MB
```

#### 2.3 Test Container Locally (Optional)

```bash
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://greenlang:greenlang@host.docker.internal:5432/greenlang" \
  greenlang/pack-027-enterprise-net-zero:1.0.0
```

**Expected Output**:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test API**:
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","pack":"PACK-027","version":"1.0.0"}
```

Stop container: `Ctrl+C` or `docker stop <container_id>`

---

### Step 3: Kubernetes Deployment

**Duration**: 5-10 minutes
**Critical**: Yes - Production deployment

#### 3.1 Review Kubernetes Configuration

```bash
cat kubernetes/deployment.yaml
```

**Key Configuration**:
- Replicas: 3 (production)
- Resource Requests: 1 CPU, 2Gi RAM
- Resource Limits: 2 CPU, 4Gi RAM
- Health Checks: Liveness + Readiness probes
- HPA: 3-10 replicas based on CPU >70%
- PDB: Min 2 available pods

#### 3.2 Deploy to Kubernetes

```bash
bash scripts/deploy.sh production
```

**Expected Output**:
```
=======================================================================
PACK-027 Kubernetes Deployment - PRODUCTION
=======================================================================

✓ Creating namespace: pack-027-production
✓ Applying ConfigMap
✓ Applying Secret
✓ Applying Deployment (3 replicas)
✓ Applying Service (ClusterIP)
✓ Applying HPA (3-10 replicas)
✓ Applying PDB (min 2 available)
✓ Applying Ingress

Waiting for rollout...
deployment "pack-027-enterprise-net-zero" successfully rolled out

✅ PACK-027 deployed successfully to production
```

#### 3.3 Verify Deployment

```bash
kubectl get pods -n pack-027-production
```

**Expected Output**:
```
NAME                                          READY   STATUS    RESTARTS   AGE
pack-027-enterprise-net-zero-abc123-xyz       1/1     Running   0          2m
pack-027-enterprise-net-zero-def456-uvw       1/1     Running   0          2m
pack-027-enterprise-net-zero-ghi789-rst       1/1     Running   0          2m
```

```bash
kubectl get svc -n pack-027-production
```

**Expected Output**:
```
NAME                           TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
pack-027-enterprise-net-zero   ClusterIP   10.96.123.45    <none>        8000/TCP   2m
```

**Troubleshooting**:
- If pods CrashLoopBackOff: Check logs with `kubectl logs -n pack-027-production <pod-name>`
- If ImagePullBackOff: Verify container image exists in registry
- If service unreachable: Check service endpoints with `kubectl get endpoints -n pack-027-production`

---

### Step 4: Health Checks

**Duration**: 2-3 minutes
**Critical**: Yes - Verify production readiness

#### 4.1 Run Comprehensive Health Check

```bash
python scripts/health_check.py --api-url http://localhost:8000
```

**Expected Output**:
```
=======================================================================
PACK-027 Enterprise Net Zero Pack - Health Check
=======================================================================

CATEGORY: Database Connection
  ✓ PostgreSQL connection successful
  ✓ All 18 tables exist
  ✓ 200+ indexes operational
  ✓ RLS policies active

CATEGORY: Calculation Engines
  ✓ EnterpriseBaselineEngine operational
  ✓ SBTiTargetEngine operational
  ✓ ScenarioModelingEngine operational
  ✓ CarbonPricingEngine operational
  ✓ Scope4AvoidedEmissionsEngine operational
  ✓ SupplyChainMappingEngine operational
  ✓ MultiEntityConsolidationEngine operational
  ✓ FinancialIntegrationEngine operational
  ✓ DataQualityEngine operational
  ✓ RegulatoryComplianceEngine operational
  ✓ AssuranceReadinessEngine operational
  ✓ RiskAssessmentEngine operational

CATEGORY: Workflows
  ✓ All 10 workflows registered
  ✓ DAG dependencies validated

CATEGORY: Templates
  ✓ All 12 templates loaded
  ✓ Multi-format rendering functional

CATEGORY: Integrations
  ✓ SAP connector ready
  ✓ Oracle connector ready
  ✓ Workday connector ready
  ✓ CDP bridge ready
  ✓ SBTi bridge ready
  ✓ Assurance provider bridge ready
  ✓ All 16 integrations operational

CATEGORY: API Endpoints
  ✓ /health responds 200 OK
  ✓ /docs accessible
  ✓ /api/v1/enterprise/profiles responds
  ✓ /api/v1/enterprise/baseline responds

CATEGORY: Prometheus Metrics
  ✓ Metrics endpoint /metrics accessible
  ✓ Custom metrics registered

=======================================================================
OVERALL HEALTH: ✅ HEALTHY (16/16 categories passed)
=======================================================================
```

#### 4.2 Check Prometheus Metrics

```bash
curl http://localhost:8000/metrics | grep pack_027
```

**Expected Output**:
```
# HELP pack_027_baseline_calculations_total Total enterprise baseline calculations
# TYPE pack_027_baseline_calculations_total counter
pack_027_baseline_calculations_total 0.0

# HELP pack_027_sbti_validations_total Total SBTi target validations
# TYPE pack_027_sbti_validations_total counter
pack_027_sbti_validations_total 0.0
...
```

#### 4.3 Validate API Documentation

```bash
curl http://localhost:8000/docs
# Opens interactive API documentation (Swagger UI)
```

**Troubleshooting**:
- If health check fails: Review specific category failures
- If database unreachable: Verify DATABASE_URL and PostgreSQL connectivity
- If engines fail to load: Check import errors in application logs

---

### Step 5: Enterprise Onboarding

**Duration**: 10-15 minutes
**Critical**: Optional - For first enterprise setup

#### 5.1 Run Interactive Onboarding Wizard

```bash
python scripts/enterprise_onboarding.py --api-url http://localhost:8000
```

**Interactive Steps**:

**Step 1/8: Organization Profile**
```
Organization Legal Name: Acme Corporation
Headquarters Country: United States
Primary Sector: manufacturing
Total Employees: 5000
Annual Revenue (USD): 1200000000
Fiscal Year End (MM-DD): 12-31

✓ Organization profile created: Acme Corporation
```

**Step 2/8: Multi-Entity Hierarchy**
```
Number of legal entities (including parent): 3

Entity 1/3:
  Entity Name: Acme Corporation (Parent)
  Country: United States
  Ownership % (0-100): 100

Entity 2/3:
  Entity Name: Acme Europe GmbH
  Country: Germany
  Ownership % (0-100): 100

Entity 3/3:
  Entity Name: Acme Asia Ltd
  Country: Singapore
  Ownership % (0-100): 80

✓ 3 entities defined
```

**Step 3/8: Consolidation Approach**
```
Select your GHG Protocol consolidation approach:

  1. Financial Control (recommended for most enterprises)
  2. Operational Control
  3. Equity Share

Selection (1-3): 1

✓ Consolidation approach: FINANCIAL_CONTROL
```

**Step 4/8: Data Sources & ERP Integration**
```
Primary ERP System:
  1. SAP
  2. Oracle
  3. Workday
  4. None

Selection (1-4): 1

✓ SAP integration will be configured
  (API credentials will be configured in deployment step)
```

**Step 5/8: Baseline Year Selection**
```
Baseline Year (e.g., 2019): 2019
Current Reporting Year (e.g., 2025): 2025

✓ Baseline year: 2019
✓ Reporting year: 2025
```

**Step 6/8: Target Setting Strategy**
```
SBTi Pathway:
  1. ACA_15C (Absolute Contraction Approach - 1.5°C)
  2. ACA_WB2C (Absolute Contraction Approach - Well Below 2°C)
  3. SDA (Sectoral Decarbonization Approach)
  4. MIXED (ACA + SDA for different scopes)

Selection (1-4): 1

Near-term target year (e.g., 2030): 2030
Net zero target year (e.g., 2050): 2050

✓ SBTi pathway: ACA_15C
✓ Near-term target: 2030
✓ Net zero target: 2050
```

**Step 7/8: External Assurance**
```
Assurance Level:
  1. LIMITED (ISO 14064-3 limited assurance)
  2. REASONABLE (ISO 14064-3 reasonable assurance)
  3. NONE (No external assurance required)

Selection (1-3): 2

✓ Assurance level: REASONABLE
```

**Step 8/8: Configuration Review & Deployment**
```
ORGANIZATION:
  Name:           Acme Corporation
  Sector:         manufacturing
  Employees:      5,000
  Revenue:        $1,200,000,000

ENTITIES:
  Total Entities: 3
  Consolidation:  FINANCIAL_CONTROL

DATA SOURCES:
  ERP System:     SAP

BASELINE:
  Baseline Year:  2019
  Reporting Year: 2025

TARGETS:
  SBTi Pathway:   ACA_15C
  Target Year:    2030
  Net Zero Year:  2050

ASSURANCE:
  Enabled:        True
  Level:          REASONABLE

✓ Configuration saved to: config/enterprise_config_abc12345.json

=======================================================================
✅ Enterprise onboarding complete!
=======================================================================
```

#### 5.2 Verify Configuration

```bash
cat config/enterprise_config_*.json
```

**Expected Output**: JSON configuration with all enterprise settings

---

## Post-Deployment Validation

### ✅ Final Verification Checklist

- [ ] **Database**: All 15 migrations applied (V166-V180)
- [ ] **Database**: 18 tables created with 200+ indexes
- [ ] **Database**: 36 RLS policies active
- [ ] **Container**: Docker image built and tagged
- [ ] **Kubernetes**: 3 pods running in production namespace
- [ ] **Kubernetes**: HPA configured (3-10 replicas)
- [ ] **Kubernetes**: Service accessible on ClusterIP
- [ ] **Health**: All 16 health check categories pass
- [ ] **API**: /health endpoint returns 200 OK
- [ ] **API**: /docs accessible with full API documentation
- [ ] **Metrics**: Prometheus metrics endpoint functional
- [ ] **Onboarding**: Configuration saved for first enterprise (if applicable)

---

## Rollback Procedure

If deployment fails or issues are detected:

### Database Rollback

```bash
# Each migration has a down script
cd deployment/database/migrations/sql
psql $DATABASE_URL -f V180__PACK027_views_and_indexes.down.sql
psql $DATABASE_URL -f V179__PACK027_data_quality_tracking.down.sql
# ... continue in reverse order to V166
```

### Kubernetes Rollback

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/pack-027-enterprise-net-zero -n pack-027-production

# Or delete deployment entirely
kubectl delete namespace pack-027-production
```

### Container Cleanup

```bash
# Remove container image
docker rmi greenlang/pack-027-enterprise-net-zero:1.0.0

# Clean build cache
docker builder prune -a
```

---

## Monitoring & Maintenance

### Daily Health Checks

```bash
# Automated daily health check (add to cron)
0 9 * * * cd /path/to/pack && python scripts/health_check.py --api-url http://localhost:8000 >> logs/daily_health.log 2>&1
```

### Prometheus Alerts

Key metrics to monitor:
- `pack_027_baseline_calculations_total` - Should increase with usage
- `pack_027_api_request_duration_seconds` - Should stay <2s p95
- `pack_027_database_query_duration_seconds` - Should stay <1s p95
- `pack_027_errors_total` - Should stay near 0

### Log Monitoring

```bash
# Kubernetes logs
kubectl logs -n pack-027-production -l app=pack-027-enterprise-net-zero --tail=100 -f

# Application logs
tail -f logs/pack-027-production.log
```

---

## Support & Troubleshooting

### Common Issues

**Issue**: Pod CrashLoopBackOff
**Solution**: Check DATABASE_URL is correct in Secret/ConfigMap

**Issue**: 500 Internal Server Error
**Solution**: Check application logs for Python exceptions

**Issue**: Migration fails with "relation already exists"
**Solution**: Run verify_migrations.py to check current state, manually fix conflicts

**Issue**: Health check fails on engine import
**Solution**: Verify all dependencies installed: `pip install -r requirements.txt`

**Issue**: SAP/Oracle connector fails
**Solution**: Verify ERP credentials in configuration, check network connectivity

### Getting Help

- **Documentation**: `docs/` directory
- **API Reference**: http://localhost:8000/docs
- **Validation Report**: `docs/VALIDATION_REPORT.md`
- **Test Coverage**: Run `pytest tests/ --cov` for detailed coverage report

---

## Deployment Success Criteria

✅ **DEPLOYMENT SUCCESSFUL** when all of the following are true:

1. All 15 database migrations applied (V166-V180)
2. All 18 tables exist with proper indexes and RLS policies
3. 3+ Kubernetes pods running in production namespace
4. Health check passes all 16 categories
5. API endpoints respond with 200 OK
6. Prometheus metrics accessible
7. No CrashLoopBackOff or ImagePullBackOff errors
8. Logs show no critical errors
9. Enterprise onboarding configuration saved (if applicable)
10. All integration connectors ready (SAP, Oracle, Workday, CDP, SBTi)

**Estimated Total Time**: 30-45 minutes
**Risk Level**: Low (comprehensive rollback procedures available)
**Production Ready**: Yes

---

**Last Updated**: March 2026
**Pack Version**: 1.0.0
**Deployment Template**: v1.0
