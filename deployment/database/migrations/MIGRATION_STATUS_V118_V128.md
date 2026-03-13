# Database Migration Status: V118-V128

## Overview
This document tracks the status of database migrations V118 through V128 for EUDR agents 030-040 (Documentation & Reporting category).

**Migration Range**: V118 - V128 (11 migrations)
**Category**: AGENT-EUDR Documentation & Reporting
**Current Database Version**: V117 (as of 2026-03-13)
**Target Database Version**: V128

---

## Migration System Details

### System Identified
- **Migration Tool**: Flyway 9.22.3
- **Configuration**: `C:\Users\aksha\Code-V1_GreenLang\deployment\database\migrations\flyway.conf`
- **Migration Location**: `C:\Users\aksha\Code-V1_GreenLang\deployment\database\migrations\sql\`
- **Deployment Method**: Docker-based (Kubernetes Job or local Docker)

### Flyway Configuration
- **Database**: PostgreSQL with TimescaleDB extension
- **Connection**: Via PgBouncer (production) or direct (development)
- **Schemas**: public, metrics, audit
- **Validation**: Enabled (`validateOnMigrate=true`)
- **Out-of-Order**: Disabled for production safety
- **Baseline**: Enabled for existing databases

---

## Migration Files Status

All migration files have been verified to exist in the migrations directory:

| Version | File Name | Agent ID | Component | Status |
|---------|-----------|----------|-----------|--------|
| V118 | `V118__agent_eudr_documentation_generator.sql` | EUDR-030 | Documentation Generator | ✅ EXISTS |
| V119 | `V119__agent_eudr_stakeholder_engagement.sql` | EUDR-031 | Stakeholder Engagement Tool | ✅ EXISTS |
| V120 | `V120__agent_eudr_grievance_mechanism_manager.sql` | EUDR-032 | Grievance Mechanism Manager | ✅ EXISTS |
| V121 | `V121__agent_eudr_continuous_monitoring.sql` | EUDR-033 | Continuous Monitoring Agent | ✅ EXISTS |
| V122 | `V122__agent_eudr_annual_review_scheduler.sql` | EUDR-034 | Annual Review Scheduler | ✅ EXISTS |
| V123 | `V123__agent_eudr_improvement_plan_creator.sql` | EUDR-035 | Improvement Plan Creator | ✅ EXISTS |
| V124 | `V124__agent_eudr_eu_information_system_interface.sql` | EUDR-036 | EU Information System Interface | ✅ EXISTS |
| V125 | `V125__agent_eudr_due_diligence_statement_creator.sql` | EUDR-037 | Due Diligence Statement Creator | ✅ EXISTS |
| V126 | `V126__agent_eudr_reference_number_generator.sql` | EUDR-038 | Reference Number Generator | ✅ EXISTS |
| V127 | `V127__agent_eudr_customs_declaration_support.sql` | EUDR-039 | Customs Declaration Support | ✅ EXISTS |
| V128 | `V128__agent_eudr_authority_communication_manager.sql` | EUDR-040 | Authority Communication Manager | ✅ EXISTS |

---

## Migration Scripts

Two migration scripts have been created to apply these migrations:

### 1. Bash Script (Linux/macOS/Git Bash)
**Location**: `C:\Users\aksha\Code-V1_GreenLang\deployment\database\migrations\apply_v118_v128.sh`

**Features**:
- Pre-flight checks (Docker, database connectivity)
- Current migration status display
- Pending migrations preview
- Migration validation
- User confirmation prompt
- Post-migration status verification

**Usage**:
```bash
# Make executable (first time only)
chmod +x deployment/database/migrations/apply_v118_v128.sh

# Apply to development database
./deployment/database/migrations/apply_v118_v128.sh dev

# Apply to staging database
./deployment/database/migrations/apply_v118_v128.sh staging

# Apply to production database
./deployment/database/migrations/apply_v118_v128.sh prod
```

### 2. PowerShell Script (Windows)
**Location**: `C:\Users\aksha\Code-V1_GreenLang\deployment\database\migrations\apply_v118_v128.ps1`

**Features**:
- Same features as Bash script
- Windows-native execution
- Colored console output

**Usage**:
```powershell
# Apply to development database
.\deployment\database\migrations\apply_v118_v128.ps1

# Apply to specific environment
.\deployment\database\migrations\apply_v118_v128.ps1 -Environment dev
.\deployment\database\migrations\apply_v118_v128.ps1 -Environment staging
.\deployment\database\migrations\apply_v118_v128.ps1 -Environment prod
```

---

## Prerequisites

Before running migrations, ensure:

1. **Docker Desktop is running**
   - Check: `docker --version`
   - Start Docker Desktop if not running

2. **PostgreSQL database is accessible**
   - Development: localhost:5432
   - Staging/Production: Configure via environment variables

3. **Database credentials are available**
   - Set via environment variables or use defaults
   - `FLYWAY_DB_HOST`, `FLYWAY_DB_PORT`, `FLYWAY_DB_NAME`
   - `FLYWAY_DB_USER`, `FLYWAY_DB_PASSWORD`

4. **Current migration version is V117**
   - Verify: Check `flyway_schema_history` table
   - Or run: `flyway info`

---

## Environment Variables

### Database Connection
```bash
export FLYWAY_DB_HOST="localhost"        # Database host
export FLYWAY_DB_PORT="5432"             # Database port
export FLYWAY_DB_NAME="greenlang_platform"  # Database name
export FLYWAY_DB_USER="greenlang_admin"  # Database user
export FLYWAY_DB_PASSWORD="your_password"  # Database password
```

### Flyway Configuration
```bash
export FLYWAY_IMAGE="flyway/flyway:9.22.3"  # Flyway Docker image
```

---

## Migration Process

### Step 1: Pre-Migration Checklist
- [ ] Backup database (recommended for production)
- [ ] Verify current version is V117
- [ ] Review migration files (V118-V128)
- [ ] Check database connectivity
- [ ] Ensure Docker is running

### Step 2: Dry Run (Recommended)
```bash
# View pending migrations without applying
docker run --rm \
  --network host \
  -v "$(pwd)/deployment/database/migrations/sql:/flyway/sql:ro" \
  -v "$(pwd)/deployment/database/migrations/flyway.conf:/flyway/conf/flyway.conf:ro" \
  flyway/flyway:9.22.3 \
  -url="jdbc:postgresql://localhost:5432/greenlang_platform" \
  -user="greenlang_admin" \
  -password="your_password" \
  -locations="filesystem:/flyway/sql" \
  -configFiles="/flyway/conf/flyway.conf" \
  info
```

### Step 3: Apply Migrations
```bash
# Using the provided script (recommended)
./deployment/database/migrations/apply_v118_v128.sh dev

# Or using PowerShell on Windows
.\deployment\database\migrations\apply_v118_v128.ps1
```

### Step 4: Verify Migrations
```bash
# Check final migration status
docker run --rm \
  --network host \
  -v "$(pwd)/deployment/database/migrations/sql:/flyway/sql:ro" \
  -v "$(pwd)/deployment/database/migrations/flyway.conf:/flyway/conf/flyway.conf:ro" \
  flyway/flyway:9.22.3 \
  -url="jdbc:postgresql://localhost:5432/greenlang_platform" \
  -user="greenlang_admin" \
  -password="your_password" \
  -locations="filesystem:/flyway/sql" \
  -configFiles="/flyway/conf/flyway.conf" \
  info
```

---

## Expected Migration Results

### Database Objects Created

Each migration creates the following database objects for its respective agent:

#### Common Pattern (per agent)
- **Tables**: 8-10 tables per agent
  - Main entity table
  - Configuration table
  - Execution log table
  - Results/outputs table
  - Audit log table (TimescaleDB hypertable)
  - Supporting tables (varies by agent)

- **Indexes**: ~100-150 indexes per agent
  - Primary keys
  - Foreign keys
  - Search indexes (JSONB, text)
  - Performance indexes (timestamps, status)
  - Composite indexes

- **Comments**: Documentation on all tables and columns

#### Total Objects (V118-V128)
- **Tables**: ~90-110 tables (including hypertables)
- **Indexes**: ~1,200-1,500 indexes
- **Schemas**: public (tables), metrics, audit

### Sample Migration: V118 (Documentation Generator)

**Tables Created**:
1. `eudr_dgn_dds_documents` - Due Diligence Statement records
2. `eudr_dgn_article9_packages` - Article 9 information packages
3. `eudr_dgn_risk_assessment_docs` - Risk assessment documentation
4. `eudr_dgn_mitigation_docs` - Mitigation documentation
5. `eudr_dgn_compliance_packages` - Complete compliance packages
6. `eudr_dgn_document_versions` - Document version tracking
7. `eudr_dgn_submission_lifecycle` - EU system submission tracking
8. `eudr_dgn_schema_validation` - EUDR schema validation results
9. `eudr_dgn_audit_log` - Audit trail (TimescaleDB hypertable)

**Key Features**:
- JSONB storage for flexible data structures
- Full-text search capabilities
- TimescaleDB hypertables for audit logs
- Comprehensive indexing for performance
- EUDR regulatory compliance tracking

---

## Troubleshooting

### Docker Not Running
```
Error: Cannot connect to Docker API
Solution: Start Docker Desktop
```

### Database Connection Failed
```
Error: Cannot connect to database at localhost:5432
Solutions:
1. Start PostgreSQL: docker-compose up -d postgres
2. Check connection settings: verify host, port, credentials
3. Check firewall: ensure port 5432 is accessible
```

### Migration Validation Failed
```
Error: Migration validation failed
Solutions:
1. Check for modified migration files (checksum mismatch)
2. Verify current database version
3. Review flyway_schema_history table
4. Check for out-of-order migrations
```

### Permission Denied
```
Error: Permission denied on database
Solutions:
1. Verify database user has migration permissions
2. Check user role: should have CREATE, ALTER permissions
3. Verify schema permissions
```

---

## Kubernetes Deployment

For Kubernetes-based deployments, use the Helm chart:

```bash
# Apply migrations via Helm
helm upgrade --install flyway \
  deployment/helm/flyway \
  --namespace greenlang \
  --set database.host=postgres.database.svc.cluster.local \
  --set database.port=5432 \
  --set database.name=greenlang_platform

# Monitor migration job
kubectl logs -n greenlang -l app=flyway,component=database-migration -f
```

---

## Verification Queries

After migration, run these queries to verify:

### Check Migration Status
```sql
-- View migration history
SELECT version, description, type, installed_on, execution_time, success
FROM flyway_schema_history
WHERE version::integer >= 118
ORDER BY installed_rank DESC;

-- Verify V128 is latest
SELECT MAX(version::integer) as latest_version
FROM flyway_schema_history
WHERE success = true;
```

### Verify EUDR Tables
```sql
-- Count EUDR tables created by V118-V128
SELECT COUNT(*) as eudr_table_count
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'eudr_%'
  AND table_name ~ '(dgn|set|gmm|cmt|ars|ipc|eis|dsc|rng|cds|acm)_';

-- List all EUDR agent tables
SELECT table_name,
       pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'eudr_%'
ORDER BY table_name;
```

### Verify Indexes
```sql
-- Count indexes on EUDR tables
SELECT schemaname, tablename, COUNT(*) as index_count
FROM pg_indexes
WHERE tablename LIKE 'eudr_%'
  AND tablename ~ '(dgn|set|gmm|cmt|ars|ipc|eis|dsc|rng|cds|acm)_'
GROUP BY schemaname, tablename
ORDER BY tablename;
```

### Verify TimescaleDB Hypertables
```sql
-- List TimescaleDB hypertables created
SELECT hypertable_schema, hypertable_name, num_dimensions
FROM timescaledb_information.hypertables
WHERE hypertable_name LIKE 'eudr_%audit_log'
  AND hypertable_name ~ '(dgn|set|gmm|cmt|ars|ipc|eis|dsc|rng|cds|acm)_'
ORDER BY hypertable_name;
```

---

## Rollback Plan

Flyway does not support automatic rollback. For rollback:

### Manual Rollback (if needed)
1. **Restore from backup** (recommended)
   ```bash
   # Restore database from backup taken before migration
   psql -U greenlang_admin -d greenlang_platform < backup_pre_v118.sql
   ```

2. **Create undo migrations** (Flyway Teams only)
   - Requires Flyway Teams license
   - Create U118-U128 undo scripts

3. **Manual cleanup**
   ```sql
   -- Drop tables created by V118-V128 (example)
   DROP TABLE IF EXISTS eudr_dgn_dds_documents CASCADE;
   DROP TABLE IF EXISTS eudr_dgn_article9_packages CASCADE;
   -- ... repeat for all tables

   -- Remove from migration history
   DELETE FROM flyway_schema_history WHERE version::integer >= 118;
   ```

---

## Post-Migration Actions

After successful migration:

1. **Update MEMORY.md**
   ```markdown
   ## DB Migrations (Current: V128)
   - V118-V128: AGENT-EUDR-030 through 040 (Documentation & Reporting)
   ```

2. **Update application configuration**
   - Agent registry entries
   - RBAC permissions
   - Auth integration

3. **Run tests**
   ```bash
   pytest tests/agents/eudr/ -v
   ```

4. **Deploy updated agents**
   ```bash
   kubectl rollout restart deployment -n greenlang -l app=eudr-agents
   ```

---

## Contact & Support

**Migration Issues**: Review this document and check Flyway logs
**Database Issues**: Check PostgreSQL logs and connection settings
**Application Issues**: Verify agent deployments and configurations

---

## Appendix: EUDR Agents 030-040 Summary

### Documentation & Reporting Category

| Agent ID | Name | Purpose | Key Tables |
|----------|------|---------|------------|
| EUDR-030 | Documentation Generator | Produces DDS documents, Article 9 packages | eudr_dgn_* (9 tables) |
| EUDR-031 | Stakeholder Engagement Tool | Manages stakeholder communication | eudr_set_* (9 tables) |
| EUDR-032 | Grievance Mechanism Manager | Handles complaints and grievances | eudr_gmm_* (9 tables) |
| EUDR-033 | Continuous Monitoring Agent | Ongoing compliance monitoring | eudr_cmt_* (9 tables) |
| EUDR-034 | Annual Review Scheduler | Schedules annual due diligence reviews | eudr_ars_* (9 tables) |
| EUDR-035 | Improvement Plan Creator | Creates improvement action plans | eudr_ipc_* (9 tables) |
| EUDR-036 | EU Information System Interface | Interfaces with EU EUDR system | eudr_eis_* (9 tables) |
| EUDR-037 | Due Diligence Statement Creator | Creates formal DDS submissions | eudr_dsc_* (9 tables) |
| EUDR-038 | Reference Number Generator | Generates unique DDS reference numbers | eudr_rng_* (9 tables) |
| EUDR-039 | Customs Declaration Support | Supports customs documentation | eudr_cds_* (9 tables) |
| EUDR-040 | Authority Communication Manager | Manages authority communications | eudr_acm_* (9 tables) |

---

**Document Version**: 1.0
**Last Updated**: 2026-03-13
**Status**: Ready for Migration
