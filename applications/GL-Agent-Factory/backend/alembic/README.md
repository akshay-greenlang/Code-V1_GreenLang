# Database Migrations

This directory contains Alembic database migrations for the GL-Agent-Factory.

## Directory Structure

```
alembic/
    env.py           # Alembic environment configuration
    script.py.mako   # Template for new migrations
    versions/        # Migration scripts
    README.md        # This file
```

## Quick Start

### Using Makefile (Recommended)

```bash
# Apply all pending migrations
make db-upgrade

# Rollback last migration
make db-downgrade

# Create new auto-generated migration
make db-revision MSG="add new column to agents"

# Create empty migration for custom SQL
make db-revision-empty MSG="add custom function"

# Show current migration status
make db-current

# Show migration history
make db-history
```

### Using Alembic Directly

```bash
cd backend

# Apply all migrations
alembic upgrade head

# Apply specific migration
alembic upgrade 20241208_130000_001

# Rollback last migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade 20241208_130000_001

# Rollback all migrations
alembic downgrade base

# Show current revision
alembic current

# Show migration history
alembic history --verbose

# Generate SQL without executing (dry run)
alembic upgrade head --sql

# Create new migration (auto-detect changes)
alembic revision --autogenerate -m "description"

# Create empty migration
alembic revision -m "description"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALEMBIC_DATABASE_URL` | Primary database URL (highest priority) | - |
| `DATABASE_URL` | Alternative database URL | - |
| `sqlalchemy.url` | Config file URL (lowest priority) | `postgresql+asyncpg://...` |

### Database URL Format

```
postgresql+asyncpg://username:password@host:port/database
```

## Migration Conventions

### File Naming

Migrations follow the pattern:
```
YYYYMMDD_HHMMSS_<revision>_<slug>.py
```

Example: `20241208_130000_001_initial_schema.py`

### Writing Migrations

1. **Always include both upgrade() and downgrade()**
   - Ensure migrations can be rolled back
   - Test both directions in development

2. **Use transactions**
   - Each migration runs in a transaction
   - Failed migrations are automatically rolled back

3. **Be explicit with indexes**
   - Create indexes explicitly rather than relying on ORM defaults
   - Use appropriate index types (btree, gin, gist)

4. **Handle data migrations carefully**
   - Separate schema changes from data migrations
   - Use batch operations for large data sets

5. **Document complex migrations**
   - Add comments explaining non-obvious changes
   - Reference related issues/tickets

### Example Migration

```python
"""Add status column to agents table

Revision ID: 20241209_100000_002
Revises: 20241208_130000_001
Create Date: 2024-12-09 10:00:00 UTC

Adds a new status column to track agent operational state.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '20241209_100000_002'
down_revision: Union[str, None] = '20241208_130000_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add status column with default value."""
    op.add_column(
        'agents',
        sa.Column(
            'operational_status',
            sa.String(50),
            nullable=False,
            server_default='active',
            comment='Operational status of the agent'
        )
    )
    op.create_index(
        'ix_agents_operational_status',
        'agents',
        ['operational_status']
    )


def downgrade() -> None:
    """Remove status column."""
    op.drop_index('ix_agents_operational_status', table_name='agents')
    op.drop_column('agents', 'operational_status')
```

## Production Deployment

### Pre-deployment Checklist

1. [ ] Test migration in staging environment
2. [ ] Verify rollback works correctly
3. [ ] Check migration duration for large tables
4. [ ] Review for breaking changes
5. [ ] Coordinate with application deployment

### Deployment Commands

```bash
# Production with environment variable
ALEMBIC_DATABASE_URL="postgresql+asyncpg://prod_user:secret@prod-db:5432/agent_factory" \
    make db-upgrade

# Check current state
make db-current

# Generate SQL for review (without executing)
make db-show REV=head
```

### Emergency Rollback

```bash
# Rollback last migration
make db-downgrade

# Rollback multiple migrations
cd backend && alembic downgrade -3

# Rollback to specific revision
cd backend && alembic downgrade 20241208_130000_001
```

## Troubleshooting

### Common Issues

1. **"Target database is not up to date"**
   ```bash
   # Check current revision
   make db-current

   # Apply pending migrations
   make db-upgrade
   ```

2. **"Can't locate revision"**
   - Verify migration file exists in `versions/`
   - Check revision ID matches file content

3. **"Relation already exists"**
   - Migration may have partially applied
   - Check database state and use `alembic stamp` if needed

4. **Migration conflicts**
   ```bash
   # Show branch points
   make db-branches

   # Merge branches
   make db-merge REVS="rev1 rev2" MSG="merge branches"
   ```

### Database State Reset (Development Only)

```bash
# WARNING: This destroys all data!
make db-downgrade-all

# Recreate from scratch
make db-upgrade
```

## Models Covered

The initial migration creates tables for:

- `tenants` - Multi-tenant organizations
- `users` - User accounts
- `agents` - Registered AI agents
- `agent_versions` - Agent version history
- `executions` - Agent execution records
- `audit_logs` - Compliance audit trail
- `tenant_usage_logs` - Usage tracking
- `tenant_invitations` - User invitations

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
