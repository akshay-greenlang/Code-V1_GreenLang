# GL-VCCI Database Migration Guide
## Scope 3 Carbon Intelligence Platform v2.0

**Version:** 2.0.0
**Database:** PostgreSQL 15+
**Migration Tool:** Alembic 1.12+
**Last Updated:** November 8, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Migration Strategy](#migration-strategy)
3. [Pre-Migration Checklist](#pre-migration-checklist)
4. [Running Migrations](#running-migrations)
5. [Rollback Procedures](#rollback-procedures)
6. [Zero-Downtime Migrations](#zero-downtime-migrations)
7. [Common Migration Patterns](#common-migration-patterns)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Database Schema

GL-VCCI uses a multi-tenant PostgreSQL database with the following architecture:

- **Schema Version:** Managed by Alembic
- **Tables:** 32 core tables
- **Indexes:** 47 indexes (including partial and composite)
- **Partitions:** 2 partitioned tables (emissions, audit_logs)
- **Extensions:** pgcrypto, uuid-ossp, pg_trgm, btree_gin

### Migration Philosophy

- **Zero-Downtime:** All migrations should be backwards compatible
- **Rollback-Safe:** Every migration must have a working downgrade path
- **Idempotent:** Migrations can be run multiple times safely
- **Tested:** All migrations tested in staging before production
- **Fast:** Migrations should complete in < 5 minutes (or use online schema changes)

---

## Migration Strategy

### Development → Staging → Production

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Development │ -> │   Staging   │ -> │ Production  │
│   (Local)   │    │ (Pre-prod)  │    │    (GA)     │
└─────────────┘    └─────────────┘    └─────────────┘
      │                   │                   │
      v                   v                   v
  Test First         Validate            Deploy
  Auto-migrate    Manual Review      Scheduled Window
```

### Migration Types

1. **Additive Migrations** (Safe for zero-downtime)
   - Adding new tables
   - Adding new columns (with defaults)
   - Adding new indexes (CONCURRENTLY)
   - Adding new constraints (NOT VALID, then VALIDATE)

2. **Destructive Migrations** (Require downtime or multi-phase)
   - Dropping tables
   - Dropping columns
   - Renaming columns
   - Changing column types

3. **Data Migrations** (Require careful planning)
   - Backfilling data
   - Data transformations
   - Large-scale updates

---

## Pre-Migration Checklist

### Before Running ANY Migration

- [ ] **1. Backup Database**
  ```bash
  # Create full backup
  pg_dump -h $DB_HOST -U $DB_USER -d vcci_production -F c -f backup_$(date +%Y%m%d_%H%M%S).dump

  # Verify backup
  pg_restore --list backup_*.dump | head -20
  ```

- [ ] **2. Check Database Size**
  ```sql
  SELECT pg_size_pretty(pg_database_size('vcci_production'));
  ```

- [ ] **3. Check Active Connections**
  ```sql
  SELECT count(*) FROM pg_stat_activity WHERE datname = 'vcci_production';
  ```

- [ ] **4. Check Long-Running Queries**
  ```sql
  SELECT pid, now() - query_start AS duration, query
  FROM pg_stat_activity
  WHERE state = 'active' AND now() - query_start > interval '1 minute';
  ```

- [ ] **5. Review Migration Script**
  ```bash
  # Show SQL that will be executed
  alembic upgrade head --sql
  ```

- [ ] **6. Verify Disk Space**
  ```bash
  # Ensure at least 50% free space
  df -h /var/lib/postgresql
  ```

- [ ] **7. Check Migration History**
  ```bash
  # Show current version
  alembic current

  # Show migration history
  alembic history
  ```

- [ ] **8. Schedule Maintenance Window** (if needed)
  - Notify users 48 hours in advance
  - Choose low-traffic window (weekends, off-hours)
  - Prepare rollback plan

### Staging Environment Validation

- [ ] **1. Apply to Staging**
  ```bash
  # Connect to staging database
  export DATABASE_URL=$STAGING_DATABASE_URL

  # Run migration
  alembic upgrade head
  ```

- [ ] **2. Run Integration Tests**
  ```bash
  pytest tests/integration/ -v
  ```

- [ ] **3. Check Performance Impact**
  ```bash
  # Run load tests
  locust -f tests/load/test_api.py --headless -u 100 -r 10 -t 5m
  ```

- [ ] **4. Validate Data Integrity**
  ```sql
  -- Check row counts
  SELECT 'tenants' AS table, count(*) FROM tenants
  UNION ALL
  SELECT 'suppliers', count(*) FROM suppliers
  UNION ALL
  SELECT 'emissions', count(*) FROM emissions;

  -- Check for null violations
  SELECT count(*) FROM suppliers WHERE name IS NULL;
  ```

---

## Running Migrations

### Standard Migration Process

#### 1. Development (Create Migration)

```bash
# Create a new migration
alembic revision -m "add_supplier_contact_table"

# Edit the migration file
# database/migrations/versions/XXXXXX_add_supplier_contact_table.py

# Apply migration locally
alembic upgrade head

# Test the migration
pytest tests/

# Test rollback
alembic downgrade -1
alembic upgrade head
```

#### 2. Staging (Validate)

```bash
# Connect to staging
export DATABASE_URL=$STAGING_DATABASE_URL

# Show pending migrations
alembic current
alembic heads

# Apply migration
alembic upgrade head

# Verify
alembic current

# Run full test suite
pytest tests/integration/ -v --cov

# Monitor for issues
tail -f /var/log/vcci/application.log
```

#### 3. Production (Deploy)

```bash
# PRE-DEPLOYMENT CHECKLIST (see above)

# 1. Backup database (CRITICAL!)
./scripts/backup_database.sh production

# 2. Enable maintenance mode (optional)
kubectl scale deployment/backend-api --replicas=0 -n vcci-production

# 3. Connect to production
export DATABASE_URL=$PRODUCTION_DATABASE_URL

# 4. Show current version
alembic current

# 5. Show SQL that will run (review carefully!)
alembic upgrade head --sql > migration_$(date +%Y%m%d_%H%M%S).sql
cat migration_*.sql

# 6. Run migration
alembic upgrade head

# 7. Verify migration
alembic current
psql $DATABASE_URL -c "\dt"

# 8. Re-enable application
kubectl scale deployment/backend-api --replicas=3 -n vcci-production

# 9. Monitor logs
kubectl logs -f deployment/backend-api -n vcci-production

# 10. Smoke test
curl https://api.vcci.greenlang.io/health/ready
```

### Migration Commands Reference

```bash
# Show current database version
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic heads

# Upgrade to latest
alembic upgrade head

# Upgrade to specific version
alembic upgrade <revision>

# Upgrade one version at a time
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Downgrade to specific version
alembic downgrade <revision>

# Downgrade to base (DANGER!)
alembic downgrade base

# Show SQL without executing
alembic upgrade head --sql

# Stamp database with version (without running migration)
alembic stamp head
```

---

## Rollback Procedures

### Automatic Rollback (< 5 minutes after migration)

If issues detected immediately after migration:

```bash
# 1. Stop application
kubectl scale deployment/backend-api --replicas=0 -n vcci-production

# 2. Rollback migration
alembic downgrade -1

# 3. Verify rollback
alembic current

# 4. Restore application
kubectl scale deployment/backend-api --replicas=3 -n vcci-production

# 5. Verify health
curl https://api.vcci.greenlang.io/health/ready
```

### Backup Restoration (> 5 minutes after migration)

If data corruption or significant issues:

```bash
# 1. Stop application
kubectl scale deployment/backend-api --replicas=0 -n vcci-production
kubectl scale deployment/worker --replicas=0 -n vcci-production

# 2. Terminate all connections
psql $DATABASE_URL -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'vcci_production' AND pid <> pg_backend_pid();"

# 3. Restore from backup
pg_restore -h $DB_HOST -U $DB_USER -d vcci_production -c backup_YYYYMMDD_HHMMSS.dump

# 4. Verify restoration
psql $DATABASE_URL -c "SELECT count(*) FROM suppliers;"

# 5. Stamp database with correct version
alembic stamp <previous_version>

# 6. Restore application
kubectl scale deployment/backend-api --replicas=3 -n vcci-production
kubectl scale deployment/worker --replicas=3 -n vcci-production

# 7. Monitor
kubectl logs -f deployment/backend-api -n vcci-production
```

### Rollback Decision Matrix

| Time Since Migration | Data Loss | Recommended Action |
|---------------------|-----------|-------------------|
| < 5 minutes | None | Alembic downgrade |
| 5-30 minutes | Minimal | Alembic downgrade (accept data loss) |
| 30-60 minutes | Moderate | Backup restore (lose last hour) |
| > 60 minutes | Significant | Backup restore + manual data recovery |

---

## Zero-Downtime Migrations

### Adding a New Column

**BAD (causes downtime):**
```python
# DON'T DO THIS - will lock table
def upgrade():
    op.add_column('suppliers', sa.Column('priority', sa.Integer(), nullable=False, server_default='1'))
```

**GOOD (zero-downtime):**
```python
# Phase 1: Add column as nullable with default
def upgrade():
    op.add_column('suppliers', sa.Column('priority', sa.Integer(), nullable=True, server_default='1'))

# Phase 2 (next deployment): Make column not null
def upgrade():
    op.execute("UPDATE suppliers SET priority = 1 WHERE priority IS NULL")
    op.alter_column('suppliers', 'priority', nullable=False)
```

### Adding an Index

**BAD (locks table):**
```python
# DON'T DO THIS - acquires exclusive lock
def upgrade():
    op.create_index('idx_suppliers_name', 'suppliers', ['name'])
```

**GOOD (zero-downtime):**
```python
# Use CONCURRENTLY to avoid locks
def upgrade():
    op.create_index(
        'idx_suppliers_name',
        'suppliers',
        ['name'],
        postgresql_concurrently=True
    )

# IMPORTANT: Must be run outside transaction
def upgrade():
    connection = op.get_bind()
    connection.execute("COMMIT")
    connection.execute("CREATE INDEX CONCURRENTLY idx_suppliers_name ON suppliers(name)")
    connection.execute("BEGIN")
```

### Dropping a Column

**BAD (breaks running code):**
```python
# DON'T DO THIS - breaks running app
def upgrade():
    op.drop_column('suppliers', 'old_field')
```

**GOOD (multi-phase):**
```python
# Phase 1 (deployment N): Make column nullable, stop writing to it
# (Application code stops using the column)

# Phase 2 (deployment N+1): Drop column
def upgrade():
    op.drop_column('suppliers', 'old_field')
```

### Renaming a Column

**BAD (breaks running code):**
```python
# DON'T DO THIS - breaks running app
def upgrade():
    op.alter_column('suppliers', 'old_name', new_column_name='new_name')
```

**GOOD (multi-phase):**
```python
# Phase 1: Add new column, copy data
def upgrade():
    op.add_column('suppliers', sa.Column('new_name', sa.String(), nullable=True))
    op.execute("UPDATE suppliers SET new_name = old_name")

# Phase 2 (next deployment): Application reads from new_name, writes to both

# Phase 3 (next deployment): Drop old column
def upgrade():
    op.drop_column('suppliers', 'old_name')
```

### Changing Column Type

**BAD (locks table, may fail):**
```python
# DON'T DO THIS - rewrites entire table
def upgrade():
    op.alter_column('suppliers', 'id', type_=sa.BigInteger())
```

**GOOD (multi-phase):**
```python
# Phase 1: Add new column
def upgrade():
    op.add_column('suppliers', sa.Column('id_new', sa.BigInteger(), nullable=True))

# Phase 2: Backfill data (in batches)
def upgrade():
    connection = op.get_bind()
    connection.execute("""
        UPDATE suppliers SET id_new = id::bigint
        WHERE id_new IS NULL
        LIMIT 10000
    """)
    # Repeat until all rows updated

# Phase 3: Swap columns (requires downtime)
def upgrade():
    op.drop_column('suppliers', 'id')
    op.alter_column('suppliers', 'id_new', new_column_name='id')
```

---

## Common Migration Patterns

### 1. Adding a Table

```python
"""Add supplier_contacts table

Revision ID: 001_add_contacts
Revises: base
Create Date: 2025-11-08 10:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_add_contacts'
down_revision = 'base'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'supplier_contacts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('supplier_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('phone', sa.String(50), nullable=True),
        sa.Column('role', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['supplier_id'], ['suppliers.id'], ondelete='CASCADE'),
    )

    # Create indexes
    op.create_index('idx_supplier_contacts_supplier_id', 'supplier_contacts', ['supplier_id'])
    op.create_index('idx_supplier_contacts_email', 'supplier_contacts', ['email'])

def downgrade():
    op.drop_index('idx_supplier_contacts_email', table_name='supplier_contacts')
    op.drop_index('idx_supplier_contacts_supplier_id', table_name='supplier_contacts')
    op.drop_table('supplier_contacts')
```

### 2. Data Migration (Backfilling)

```python
"""Backfill supplier carbon_intensity scores

Revision ID: 002_backfill_scores
Revises: 001_add_contacts
Create Date: 2025-11-08 11:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = '002_backfill_scores'
down_revision = '001_add_contacts'

def upgrade():
    # Add column
    op.add_column('suppliers', sa.Column('carbon_intensity', sa.Float(), nullable=True))

    # Backfill in batches (avoid long locks)
    connection = op.get_bind()

    # Calculate carbon intensity from emissions data
    connection.execute("""
        UPDATE suppliers s
        SET carbon_intensity = (
            SELECT SUM(e.total_emissions) / SUM(e.spend_amount)
            FROM emissions e
            WHERE e.supplier_id = s.id
            GROUP BY e.supplier_id
        )
        WHERE EXISTS (SELECT 1 FROM emissions e WHERE e.supplier_id = s.id)
    """)

    # Set default for suppliers with no emissions
    connection.execute("""
        UPDATE suppliers
        SET carbon_intensity = 0.0
        WHERE carbon_intensity IS NULL
    """)

def downgrade():
    op.drop_column('suppliers', 'carbon_intensity')
```

### 3. Adding an Enum Type

```python
"""Add supplier_status enum

Revision ID: 003_add_status_enum
Revises: 002_backfill_scores
Create Date: 2025-11-08 12:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = '003_add_status_enum'
down_revision = '002_backfill_scores'

# Define enum
supplier_status = sa.Enum('active', 'inactive', 'pending', 'suspended', name='supplier_status')

def upgrade():
    # Create enum type
    supplier_status.create(op.get_bind(), checkfirst=True)

    # Add column
    op.add_column('suppliers', sa.Column('status', supplier_status, nullable=True, server_default='active'))

    # Backfill existing data
    op.execute("UPDATE suppliers SET status = 'active' WHERE status IS NULL")

    # Make not null
    op.alter_column('suppliers', 'status', nullable=False)

def downgrade():
    op.drop_column('suppliers', 'status')
    supplier_status.drop(op.get_bind(), checkfirst=True)
```

### 4. Adding a Constraint

```python
"""Add unique constraint on supplier email

Revision ID: 004_unique_email
Revises: 003_add_status_enum
Create Date: 2025-11-08 13:00:00
"""

from alembic import op

revision = '004_unique_email'
down_revision = '003_add_status_enum'

def upgrade():
    # Phase 1: Add constraint as NOT VALID (doesn't check existing rows)
    op.execute("""
        ALTER TABLE suppliers
        ADD CONSTRAINT uq_suppliers_email
        UNIQUE (email)
        NOT VALID
    """)

    # Phase 2: Validate constraint (checks existing rows, can be done online)
    op.execute("ALTER TABLE suppliers VALIDATE CONSTRAINT uq_suppliers_email")

def downgrade():
    op.drop_constraint('uq_suppliers_email', 'suppliers', type_='unique')
```

### 5. Partitioning an Existing Table

```python
"""Partition emissions table by reporting_period

Revision ID: 005_partition_emissions
Revises: 004_unique_email
Create Date: 2025-11-08 14:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = '005_partition_emissions'
down_revision = '004_unique_email'

def upgrade():
    # Create new partitioned table
    op.execute("""
        CREATE TABLE emissions_new (
            LIKE emissions INCLUDING ALL
        ) PARTITION BY RANGE (reporting_period)
    """)

    # Create partitions for each quarter (last 2 years)
    for year in [2024, 2025, 2026]:
        for quarter in [1, 2, 3, 4]:
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3 + 1

            op.execute(f"""
                CREATE TABLE emissions_{year}_q{quarter}
                PARTITION OF emissions_new
                FOR VALUES FROM ('{year}-{start_month:02d}-01') TO ('{year}-{end_month:02d}-01')
            """)

    # Copy data (this will take time - consider doing in batches)
    op.execute("INSERT INTO emissions_new SELECT * FROM emissions")

    # Swap tables (requires downtime)
    op.execute("ALTER TABLE emissions RENAME TO emissions_old")
    op.execute("ALTER TABLE emissions_new RENAME TO emissions")

    # Drop old table after verification
    # op.execute("DROP TABLE emissions_old")

def downgrade():
    # Restore from emissions_old if it exists
    op.execute("ALTER TABLE emissions RENAME TO emissions_partitioned")
    op.execute("ALTER TABLE emissions_old RENAME TO emissions")
    # op.execute("DROP TABLE emissions_partitioned")
```

---

## Troubleshooting

### Issue: Migration Hangs

**Symptoms:** Migration command doesn't return after several minutes

**Cause:** Table lock conflict with running queries

**Solution:**
```sql
-- Check for locks
SELECT
    locktype,
    relation::regclass AS table,
    mode,
    granted,
    pid,
    query
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE relation = 'suppliers'::regclass;

-- Kill blocking queries (CAREFUL!)
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'vcci_production'
AND pid <> pg_backend_pid()
AND state = 'idle in transaction';
```

### Issue: Out of Disk Space

**Symptoms:** Migration fails with "No space left on device"

**Solution:**
```bash
# Check disk usage
df -h

# Clean up old WAL files
psql $DATABASE_URL -c "CHECKPOINT;"

# Vacuum to reclaim space
psql $DATABASE_URL -c "VACUUM FULL;"

# Resize disk (if on cloud)
aws rds modify-db-instance --db-instance-identifier vcci-prod --allocated-storage 500
```

### Issue: Constraint Violation

**Symptoms:** Migration fails with "violates check constraint" or "violates foreign key"

**Solution:**
```sql
-- Find violating rows
SELECT * FROM suppliers WHERE email IS NULL;
SELECT * FROM emissions WHERE supplier_id NOT IN (SELECT id FROM suppliers);

-- Fix data before re-running migration
UPDATE suppliers SET email = 'unknown@example.com' WHERE email IS NULL;
DELETE FROM emissions WHERE supplier_id NOT IN (SELECT id FROM suppliers);
```

### Issue: Migration Applied but alembic_version Not Updated

**Symptoms:** `alembic current` shows old version but tables exist

**Solution:**
```bash
# Manually stamp database with correct version
alembic stamp <correct_revision>

# Verify
alembic current
```

### Issue: Need to Skip a Migration

**Symptoms:** Migration no longer relevant or already applied manually

**Solution:**
```bash
# Mark migration as applied without running it
alembic stamp <revision_to_skip>

# Or upgrade to next version
alembic upgrade <next_revision>
```

---

## Migration Checklist Template

Copy this for each production migration:

```markdown
## Migration: [NAME]
**Revision:** [REVISION_ID]
**Date:** [YYYY-MM-DD]
**Engineer:** [NAME]

### Pre-Migration
- [ ] Migration tested in development
- [ ] Migration tested in staging
- [ ] Database backup created
- [ ] Disk space verified (>50% free)
- [ ] Long-running queries checked
- [ ] Maintenance window scheduled
- [ ] Users notified (if downtime)
- [ ] Rollback plan documented

### Migration Execution
- [ ] Application stopped (if needed)
- [ ] Migration SQL reviewed
- [ ] Migration executed: `alembic upgrade head`
- [ ] Version verified: `alembic current`
- [ ] Data integrity checked
- [ ] Application started

### Post-Migration
- [ ] Health check passed
- [ ] Integration tests passed
- [ ] Performance validated
- [ ] No error spikes in logs
- [ ] Backup verified
- [ ] Documentation updated

### Rollback (if needed)
- [ ] Rollback executed: `alembic downgrade -1`
- [ ] Version verified
- [ ] Data restored (if needed)
- [ ] Incident report filed
```

---

## Best Practices

1. **Always Backup Before Migration**
   - Automated backups may not be recent enough
   - Manual backup gives you a known restore point

2. **Test in Staging First**
   - Never run untested migrations in production
   - Staging should mirror production data volume

3. **Use Transactions Carefully**
   - Most DDL in PostgreSQL is transactional
   - Some operations (CREATE INDEX CONCURRENTLY) must run outside transactions

4. **Monitor During Migration**
   - Watch for locks, disk space, memory usage
   - Be ready to cancel if issues arise

5. **Communicate Changes**
   - Notify team of schema changes
   - Update API documentation if schema affects API

6. **Version Your Migrations**
   - Never edit a migration after it's been applied
   - Create a new migration to fix issues

7. **Keep Migrations Fast**
   - Large data migrations should be batched
   - Use CONCURRENTLY for index creation
   - Consider online schema change tools for huge tables

8. **Document Complex Migrations**
   - Add comments explaining non-obvious logic
   - Document manual steps required

---

## Support

**Questions?** Contact the DevOps team:
- Email: devops@greenlang.io
- Slack: #vcci-devops
- On-call: oncall-devops@greenlang.io

**Emergency Rollback:** See `docs/runbooks/DEPLOYMENT_ROLLBACK.md`

---

**Document Version:** 2.0.0
**Last Updated:** November 8, 2025
**Next Review:** Quarterly
