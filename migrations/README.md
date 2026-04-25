# GreenLang Factors — Alembic Migration Tree

This directory is the **operator-facing Alembic tree for the FY27 Factors
product** (`factors_v0_1` schema and beyond). It is intentionally separate
from `data/migrations/` (Phase-4 auth) and `greenlang/auth/migrations/`
(auth service) so the Factors product can be promoted independently.

The canonical SQL DDL lives in
`deployment/database/migrations/sql/V###__*.sql`. Those files remain the
source of truth. Each Alembic revision in `versions/` is a thin Python
wrapper that loads and executes the matching `V###` SQL file via raw
`op.execute(...)`. This gives operators a single, idempotent
`alembic upgrade head` command without forcing them to know the V###
filename layout.

## When to use Alembic vs raw SQL

| Use case | Tool | Why |
| --- | --- | --- |
| Fresh dev / staging / prod database | `alembic upgrade head` | Runs every revision in order, records `factors_v0_1.alembic_version` |
| Hotfix / DBA-led recovery | `psql -f V###.sql` | Bypasses Alembic state; manual catch-up required afterwards |
| New schema work in PR | New revision in `versions/` + matching `V###.sql` | Keeps Alembic and SQL trees in lockstep |
| Local schema inspection | `alembic upgrade head --sql > /tmp/all.sql` | Offline mode, dumps the SQL Alembic *would* run |

## Pointing Alembic at staging vs prod

The factors `env.py` resolves the database URL with this precedence (highest first):

1. `ALEMBIC_SQLALCHEMY_URL` env var (preferred — keeps it scoped to a single command)
2. `DATABASE_URL` env var (fallback — re-uses app config)
3. `sqlalchemy.url` from `alembic.ini` (placeholder; do NOT commit real creds here)

Examples:

```bash
# Staging
ALEMBIC_SQLALCHEMY_URL="postgresql+psycopg://gl_app:***@staging-db.internal:5432/greenlang" \
  alembic upgrade head

# Prod (always dry-run first)
ALEMBIC_SQLALCHEMY_URL="postgresql+psycopg://gl_app:***@prod-db.internal:5432/greenlang" \
  alembic upgrade head --sql > /tmp/factors_upgrade.sql
# review /tmp/factors_upgrade.sql, then:
ALEMBIC_SQLALCHEMY_URL="postgresql+psycopg://gl_app:***@prod-db.internal:5432/greenlang" \
  alembic upgrade head
```

The `alembic_version` row is stored in `factors_v0_1.alembic_version`
(separate from any other Alembic tree's version table).

## Writing a new revision

1. Drop your new `V###__feature.sql` (and `V###__feature_DOWN.sql`) into
   `deployment/database/migrations/sql/`.
2. Create the matching Alembic revision:
   ```bash
   alembic revision -m "factors v0.X feature"
   ```
   This generates `versions/<rev_id>_factors_v0_X_feature.py`.
3. In the new revision file:
   * Set `down_revision` to the previous factors revision id.
   * In `upgrade()`, call `_execute_sql_file("V###__feature.sql")`
     (copy the helper from `0001_factors_v0_1_initial.py` or import it).
   * In `downgrade()`, call `_execute_sql_file("V###__feature_DOWN.sql")`.
4. Run the static test:
   `pytest tests/factors/v0_1_alpha/test_alembic_revision_0001.py`.
5. (Optional) Apply against a local Postgres before merging:
   `ALEMBIC_SQLALCHEMY_URL=postgresql+psycopg://localhost/factors_dev alembic upgrade head`.

`alembic revision --autogenerate` is **not** the default workflow for the
factors tree (raw-SQL wins), but it is supported for ORM models that may
land later: `target_metadata` in `env.py` is currently `None`; once
SQLAlchemy models are introduced, point it at their `Base.metadata` and
autogenerate will start emitting useful diffs.

## Convention: one revision per V### file

The V### scheme remains the source of truth for SQL. Alembic re-applies
those files in revision order. Do **not** put DDL directly in the
revision file — keep it in a `V###__*.sql` companion so DBAs can run the
same statements without Alembic if needed.
